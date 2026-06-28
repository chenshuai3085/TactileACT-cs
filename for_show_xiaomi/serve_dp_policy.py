"""
Diffusion Policy TCP Server (runs on GPU machine).

Supports all three DP variants:
  - official_no_tactile: ResNet18 + AvgPool (global + wrist)
  - clip_tactile_image:  CLIP ResNet18 (global + wrist + gelsight image)
  - tactile_vae_frozen:  ResNet18 + AvgPool + frozen TactileVAE (marker_offset)

Usage:
    cd /path/to/TactileACT-cs
    conda activate TactileACT
    python -m for_show_xiaomi.serve_dp_policy --ckpt_dir /path/to/dp_save_dir

ckpt_dir should contain:
    - config.json   (training config with variant, norm_stats, etc.)
    - dp_best.pth   (model checkpoint, preferably with EMA weights)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import deque

import numpy as np
import torch
import torchvision.transforms as transforms
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_DIFFUSION = os.path.join(_ROOT, "diffusion")
if _DIFFUSION not in sys.path:
    sys.path.insert(0, _DIFFUSION)

from utils import set_seed
from network import ConditionalUnet1D
from for_show_xiaomi.ws_server import TactileACTServer, ClientDisconnected


_IMG_MEAN = [0.485, 0.456, 0.406]
_IMG_STD = [0.229, 0.224, 0.225]
_IMG_NORM = transforms.Normalize(mean=_IMG_MEAN, std=_IMG_STD)


def apply_fixed_vision_enhance(image_t, config):
    """Apply deterministic image enhancement used by augmented DP training."""
    if not config.get("vision_enhance", False):
        return image_t
    out = image_t.clamp(0.0, 1.0)
    gamma = float(config.get("vision_gamma", 1.0))
    if gamma > 0 and abs(gamma - 1.0) > 1e-6:
        out = out.clamp_min(1e-6).pow(gamma)
    contrast = float(config.get("vision_contrast", 1.0))
    if abs(contrast - 1.0) > 1e-6:
        mean = out.mean(dim=(1, 2), keepdim=True)
        out = (out - mean) * contrast + mean
    brightness = float(config.get("vision_brightness", 0.0))
    if abs(brightness) > 1e-6:
        out = out + brightness
    return out.clamp(0.0, 1.0)


def build_dp_model(config: dict, device: torch.device):
    """
    Build DP models from saved training config.

    Returns:
        (vision_encoder, noise_pred_net, tac_encoder_or_None)
    """
    variant = config["variant"]
    camera_names = config["camera_names"]
    if isinstance(camera_names, str):
        camera_names = camera_names.split(",")
    action_dim = config["action_dim"]
    global_cond_dim = config["global_cond_dim"]
    down_dims = config["down_dims"]
    if isinstance(down_dims, str):
        down_dims = [int(x) for x in down_dims.split(",")]
    diffusion_step_embed_dim = config.get("diffusion_step_embed_dim", 128)

    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=global_cond_dim,
        diffusion_step_embed_dim=diffusion_step_embed_dim,
        down_dims=down_dims,
        kernel_size=5,
    ).to(device)

    tac_encoder = None

    if variant in ("official_no_tactile", "dagger_beta_decay"):
        from train_dp_official import OfficialVisionEncoder

        vision_encoder = OfficialVisionEncoder(camera_names).to(device)

    elif variant == "clip_tactile_image":
        from train_dp_tac_img import CLIPVisionEncoder

        vision_encoder = CLIPVisionEncoder(camera_names).to(device)

    elif variant == "tactile_vae_frozen":
        from train_dp_official import OfficialVisionEncoder
        from train_dp_tac_vae import FrozenTactileVAEEncoder

        vis_cams = [c for c in camera_names if c != "gelsight"]
        vision_encoder = OfficialVisionEncoder(vis_cams).to(device)

        vae_checkpoint = config.get("vae_checkpoint", "")
        vae_latent_dim = config.get("vae_latent_dim", 16)
        tac_history = config.get("tac_history", 8)
        tac_encoder = FrozenTactileVAEEncoder(
            vae_checkpoint, latent_dim=vae_latent_dim, temporal_window=tac_history
        ).to(device)

    else:
        raise ValueError(f"Unknown variant: {variant}")

    return vision_encoder, noise_pred_net, tac_encoder


def _remap_legacy_vision_keys(sd, camera_names):
    """
    Remap old shared-encoder checkpoint keys to new per-camera format.

    Old format: shared_vision.* + gelsight_encoder.*
    New format: encoders.{cam_name}.*
    """
    has_shared = any(k.startswith("shared_vision.") for k in sd)
    if not has_shared:
        return sd

    print("  [compat] detected legacy shared_vision format, remapping keys...")
    new_sd = {}
    for k, v in sd.items():
        if k.startswith("shared_vision."):
            suffix = k[len("shared_vision."):]
            for cam in camera_names:
                if cam == "gelsight":
                    continue
                new_sd[f"encoders.{cam}.{suffix}"] = v.clone()
        elif k.startswith("gelsight_encoder."):
            suffix = k[len("gelsight_encoder."):]
            new_sd[f"encoders.gelsight.{suffix}"] = v
        else:
            new_sd[k] = v
    return new_sd


def load_checkpoint(ckpt_path, vision_encoder, noise_pred_net, camera_names, device):
    """Load checkpoint, preferring EMA weights."""
    ckpt = torch.load(ckpt_path, map_location="cpu")

    vis_sd = ckpt.get("ema_vis", ckpt.get("vision_encoder"))
    vis_label = "EMA" if "ema_vis" in ckpt else "raw"
    vis_sd = _remap_legacy_vision_keys(vis_sd, camera_names)
    vision_encoder.load_state_dict(vis_sd)
    print(f"  loaded {vis_label} vision weights")

    if "ema_net" in ckpt:
        noise_pred_net.load_state_dict(ckpt["ema_net"])
        print(f"  loaded EMA noise_pred_net weights")
    else:
        noise_pred_net.load_state_dict(ckpt["noise_pred_net"])
        print(f"  loaded raw noise_pred_net weights (no EMA)")

    epoch = ckpt.get("epoch", "?")
    val_loss = ckpt.get("val_loss", "?")
    print(f"  checkpoint epoch={epoch}, val_loss={val_loss}")
    del ckpt


def preprocess_image(raw_img, resize_tf=None, crop_tf=None, vision_config=None):
    """uint8 HWC → normalized CHW float tensor (1, C, H, W)."""
    img = np.asarray(raw_img, dtype=np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    t = torch.from_numpy(img).permute(2, 0, 1).float()
    if resize_tf is not None:
        t = resize_tf(t)
    if vision_config is not None:
        t = apply_fixed_vision_enhance(t, vision_config)
    if crop_tf is not None:
        t = crop_tf(t)
    return _IMG_NORM(t).unsqueeze(0)


def _raw_image_to_uint8_hwc(raw_img):
    arr = np.asarray(raw_img)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected HWC image with 3 channels, got shape={arr.shape}")
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    if arr.max(initial=0.0) <= 1.0:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def _save_hwc_png(path, image_hwc_uint8):
    try:
        from PIL import Image

        Image.fromarray(image_hwc_uint8).save(path)
    except Exception as exc:
        np.save(path + ".npy", image_hwc_uint8)
        print(f"[dp-server][vision-debug] PNG save failed ({exc}); saved npy: {path}.npy")


def _denorm_image_tensor_to_uint8_hwc(image_t):
    t = image_t.detach().float().cpu().squeeze(0)
    mean = torch.tensor(_IMG_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(_IMG_STD, dtype=torch.float32).view(3, 1, 1)
    t = (t * std + mean).clamp(0.0, 1.0)
    return (t.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)


def dump_first_vision_obs(obs, processed, camera_names, variant, dump_dir):
    """Print and save the first real camera frame seen by the server."""
    os.makedirs(dump_dir, exist_ok=True)
    print(f"[dp-server][vision-debug] dumping first vision obs to: {dump_dir}")
    for cam_idx, cam in enumerate(camera_names):
        if cam == "gelsight":
            tac = obs.get("tac", {})
            if not tac:
                continue
            side = list(tac.keys())[0]
            side_data = tac[side]
            raw = side_data["img"] if isinstance(side_data, dict) else side_data
        else:
            raw = obs["images"][cam]

        raw_u8 = _raw_image_to_uint8_hwc(raw)
        channel_mean = raw_u8.reshape(-1, 3).mean(axis=0)
        print(
            f"[dp-server][vision-debug] raw {cam}: "
            f"shape={raw_u8.shape}, dtype={raw_u8.dtype}, "
            f"min={raw_u8.min()}, max={raw_u8.max()}, "
            f"mean={raw_u8.mean():.3f}, channel_mean={channel_mean.round(3).tolist()}"
        )
        _save_hwc_png(os.path.join(dump_dir, f"first_raw_{cam}.png"), raw_u8)

        image_t = None
        if variant == "clip_tactile_image":
            image_t = processed["images_list"][cam_idx]
        elif cam != "gelsight" and "images_dict" in processed:
            image_t = processed["images_dict"].get(cam)

        if image_t is None:
            continue
        flat = image_t.detach().float().cpu().flatten()
        print(
            f"[dp-server][vision-debug] model_input {cam}: "
            f"shape={tuple(image_t.shape)}, dtype={image_t.dtype}, "
            f"min={float(flat.min()):.3f}, max={float(flat.max()):.3f}, "
            f"mean={float(flat.mean()):.3f}, std={float(flat.std(unbiased=False)):.3f}"
        )
        denorm_u8 = _denorm_image_tensor_to_uint8_hwc(image_t)
        _save_hwc_png(os.path.join(dump_dir, f"first_model_input_denorm_{cam}.png"), denorm_u8)


def preprocess_obs(obs, camera_names, variant, device,
                   resize_tf=None, crop_tf=None, vision_config=None):
    """
    Convert raw obs dict into preprocessed tensors.

    Returns dict with:
      - 'images_dict': {cam: (1,C,H,W)} for official/tac_vae
      - 'images_list': [(1,C,H,W), ...] for clip_tactile_image
      - 'qpos': (1, qpos_dim) raw float32
      - 'marker_offset': (9, 9, 2) raw float32 (only for tac_vae)
    """
    result = {}

    qpos_raw = np.asarray(obs["qpos"], dtype=np.float32)
    result["qpos_raw"] = qpos_raw

    if variant == "clip_tactile_image":
        images_list = []
        for cam in camera_names:
            if cam == "gelsight":
                tac = obs["tac"]
                side = list(tac.keys())[0]
                side_data = tac[side]
                raw = side_data["img"] if isinstance(side_data, dict) else side_data
                images_list.append(preprocess_image(raw, vision_config=vision_config).to(device))
            else:
                raw = obs["images"][cam]
                images_list.append(preprocess_image(raw, resize_tf, crop_tf, vision_config).to(device))
        result["images_list"] = images_list
    else:
        images_dict = {}
        for cam in camera_names:
            if cam == "gelsight":
                continue
            raw = obs["images"][cam]
            images_dict[cam] = preprocess_image(raw, resize_tf, crop_tf, vision_config).to(device)
        result["images_dict"] = images_dict

    if variant == "tactile_vae_frozen":
        tac = obs["tac"]
        side = list(tac.keys())[0]
        side_data = tac[side]
        if isinstance(side_data, dict):
            if "marker_offset" not in side_data:
                raise KeyError(
                    f"tactile_vae_frozen requires 'marker_offset' in tac[{side}], "
                    f"but only got keys: {list(side_data.keys())}. "
                    f"Check that RealmanEnv provides marker_offset data.")
            marker = side_data["marker_offset"]
        else:
            marker = side_data
        result["marker_offset"] = np.asarray(marker, dtype=np.float32)

    return result


def build_obs_cond(obs_buffer, vision_encoder, tac_encoder, variant,
                   camera_names, qpos_min, qpos_max, tac_history,
                   marker_buffer, device, temporal_stride=1):
    """
    Build obs_cond from the observation buffer.

    obs_buffer: deque of preprocessed obs dicts (length = obs_horizon)
    marker_buffer: deque of raw marker_offset frames (for tac_vae)
    """
    obs_feats = []

    for t_idx in range(len(obs_buffer)):
        frame = obs_buffer[t_idx]

        if variant == "clip_tactile_image":
            vf = vision_encoder(frame["images_list"])
        else:
            vf = vision_encoder(frame["images_dict"])

        qpos_raw = frame["qpos_raw"]
        qpos_norm = (qpos_raw - qpos_min) / (qpos_max - qpos_min + 1e-8) * 2 - 1
        qpos_t = torch.tensor(qpos_norm, dtype=torch.float32).unsqueeze(0).to(device)

        if variant == "tactile_vae_frozen" and tac_encoder is not None:
            marker_end = frame["_marker_idx"]
            frames = []
            for k in range(tac_history):
                idx = marker_end - (tac_history - 1 - k) * temporal_stride
                idx = max(0, idx)
                idx = min(idx, len(marker_buffer) - 1)
                frames.append(marker_buffer[idx])
            marker_seq = np.stack(frames, axis=0)
            marker_t = torch.tensor(marker_seq, dtype=torch.float32).unsqueeze(0).to(device)
            tf = tac_encoder(marker_t)
            obs_feats.append(torch.cat([vf, tf, qpos_t], dim=-1))
        else:
            obs_feats.append(torch.cat([vf, qpos_t], dim=-1))

    return torch.cat(obs_feats, dim=-1)


@torch.no_grad()
def ddpm_inference(noise_pred_net, noise_scheduler, obs_cond, action_dim,
                   pred_horizon, num_inference_steps, device):
    """Run DDPM denoising to generate action chunk."""
    noise_scheduler.set_timesteps(num_inference_steps)
    action = torch.randn((1, pred_horizon, action_dim), device=device)

    for t in noise_scheduler.timesteps:
        noise_pred = noise_pred_net(
            action, t.unsqueeze(0).to(device), global_cond=obs_cond
        )
        action = noise_scheduler.step(noise_pred, t, action).prev_sample

    return action


def main():
    parser = argparse.ArgumentParser(description="Diffusion Policy Server")
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--ckpt_name", type=str, default="dp_best.pth")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--action_horizon", type=int, default=8,
                        help="Execute N steps of pred_horizon, then re-plan (receding horizon)")
    parser.add_argument("--action_skip", type=int, default=0,
                        help="Skip first N steps of predicted action chunk before executing")
    parser.add_argument("--temporal_agg", action="store_true")
    parser.add_argument("--num_inference_steps", type=int, default=None,
                        help="Override DDPM denoising steps (default: use config value)")
    parser.add_argument("--max_timesteps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--debug_dump_first_obs", action="store_true",
                        help="Print and save the first received raw/model-input camera frames.")
    parser.add_argument("--debug_dump_dir", type=str,
                        default="/tmp/tactileact_dp_first_obs",
                        help="Directory used by --debug_dump_first_obs.")
    cli = parser.parse_args()

    set_seed(cli.seed)
    device = torch.device(
        f"cuda:{cli.gpu}" if torch.cuda.is_available() and cli.gpu >= 0 else "cpu"
    )
    print(f"[dp-server] device: {device}")

    config_path = os.path.join(cli.ckpt_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    variant = config["variant"]
    camera_names = config["camera_names"]
    if isinstance(camera_names, str):
        camera_names = camera_names.split(",")
    action_dim = config["action_dim"]
    pred_horizon = config["pred_horizon"]
    obs_horizon = config.get("obs_horizon", 2)
    temporal_stride = int(config.get("temporal_stride", 1))
    if temporal_stride < 1:
        raise ValueError(f"Invalid temporal_stride in config: {temporal_stride}")
    num_train_timesteps = config.get("num_train_timesteps", 100)
    num_inference_steps = cli.num_inference_steps or config.get("num_inference_steps", 100)
    tac_history = config.get("tac_history", 8)

    ns = config["norm_stats"]
    action_min = np.array(ns["action_min"], dtype=np.float32)
    action_max = np.array(ns["action_max"], dtype=np.float32)
    qpos_min = np.array(ns["qpos_min"], dtype=np.float32)
    qpos_max = np.array(ns["qpos_max"], dtype=np.float32)

    resize_shape = config.get("resize_shape")
    crop_shape = config.get("crop_shape")
    resize_tf = transforms.Resize(tuple(resize_shape)) if resize_shape else None
    crop_tf = transforms.CenterCrop(tuple(crop_shape)) if crop_shape else None

    print(f"[dp-server] variant={variant}")
    print(f"[dp-server] cameras={camera_names}")
    print(f"[dp-server] action_dim={action_dim}, pred_horizon={pred_horizon}, "
          f"obs_horizon={obs_horizon}, temporal_stride={temporal_stride}")
    if resize_shape:
        print(f"[dp-server] resize={resize_shape}, crop={crop_shape}")

    vision_encoder, noise_pred_net, tac_encoder = build_dp_model(config, device)

    ckpt_path = os.path.join(cli.ckpt_dir, cli.ckpt_name)
    load_checkpoint(ckpt_path, vision_encoder, noise_pred_net, camera_names, device)

    vision_encoder.eval()
    noise_pred_net.eval()
    if tac_encoder is not None:
        tac_encoder.eval()
    print(f"[dp-server] models loaded and set to eval mode")

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )
    noise_scheduler.set_timesteps(num_inference_steps)
    print(f"[dp-server] DDPM scheduler: {num_train_timesteps} train steps, "
          f"{num_inference_steps} inference steps")

    temporal_agg = cli.temporal_agg
    action_skip = cli.action_skip
    action_horizon = min(cli.action_horizon, pred_horizon - action_skip)
    query_freq = 1 if temporal_agg else action_horizon
    print(f"[dp-server] action_skip={action_skip}, action_horizon={action_horizon}, "
          f"query_freq={query_freq}, temporal_agg={temporal_agg}")

    server = TactileACTServer(
        host=cli.host,
        port=cli.port,
        metadata={
            "protocol": "diffusion_policy",
            "variant": variant,
            "camera_names": camera_names,
            "action_dim": action_dim,
            "pred_horizon": pred_horizon,
            "temporal_stride": temporal_stride,
            "action_skip": action_skip,
            "action_horizon": action_horizon,
            "temporal_agg": temporal_agg,
        },
    )
    server.start()
    print(f"[dp-server] listening on {cli.host}:{cli.port}  (waiting for client...)")

    try:
        ep = 0
        while True:
            try:
                print(f"\n[dp-server] === episode {ep} ===")
                obs = server.recv_obs()
            except ClientDisconnected:
                print("[dp-server] client gone before first obs, waiting...")
                continue

            obs_buffer = deque(maxlen=(obs_horizon - 1) * temporal_stride + 1)
            marker_buffer = []
            marker_step = 0
            dumped_first_obs = False

            if temporal_agg:
                all_time_actions = torch.zeros(
                    [cli.max_timesteps, cli.max_timesteps + pred_horizon, action_dim],
                    device=device,
                )

            with torch.inference_mode():
                try:
                    for step in range(cli.max_timesteps):
                        processed = preprocess_obs(
                            obs, camera_names, variant, device,
                            resize_tf=resize_tf, crop_tf=crop_tf,
                            vision_config=config,
                        )
                        if cli.debug_dump_first_obs and not dumped_first_obs:
                            dump_first_vision_obs(
                                obs,
                                processed,
                                camera_names,
                                variant,
                                cli.debug_dump_dir,
                            )
                            dumped_first_obs = True

                        if variant == "tactile_vae_frozen":
                            marker_buffer.append(processed["marker_offset"])
                            processed["_marker_idx"] = marker_step
                            marker_step += 1

                        obs_buffer.append(processed)

                        while len(obs_buffer) < (obs_horizon - 1) * temporal_stride + 1:
                            pad = dict(processed)
                            if variant == "tactile_vae_frozen":
                                pad["_marker_idx"] = 0
                            obs_buffer.appendleft(pad)

                        obs_stride_buffer = list(obs_buffer)[-(obs_horizon - 1) * temporal_stride - 1::temporal_stride]
                        obs_cond = build_obs_cond(
                            obs_stride_buffer, vision_encoder, tac_encoder, variant,
                            camera_names, qpos_min, qpos_max, tac_history,
                            marker_buffer, device, temporal_stride=temporal_stride,
                        )

                        if step % query_freq == 0:
                            all_actions = ddpm_inference(
                                noise_pred_net, noise_scheduler, obs_cond,
                                action_dim, pred_horizon,
                                num_inference_steps, device,
                            )

                        if temporal_agg:
                            all_time_actions[step, step:step + pred_horizon] = (
                                all_actions.squeeze(0)
                            )
                            col = all_time_actions[:, step]
                            mask = torch.all(col != 0, dim=1)
                            col = col[mask]
                            k = 0.9
                            w = np.exp(-k * np.arange(len(col)))
                            w = w / w.sum()
                            w_t = torch.from_numpy(w).to(device).unsqueeze(1).float()
                            raw = (col * w_t).sum(dim=0, keepdim=True)
                        else:
                            raw = all_actions[:, action_skip + step % query_freq]

                        raw_np = raw.squeeze(0).cpu().numpy()

                        action = (raw_np + 1) / 2 * (action_max - action_min) + action_min
                        action = action.astype(np.float32)

                        server.send_action({
                            "actions": action[None, :],
                            "step": step,
                        })

                        if step + 1 < cli.max_timesteps:
                            obs = server.recv_obs()

                except ClientDisconnected:
                    print(f"[dp-server] client disconnected at step {step}")

            ep += 1
    except KeyboardInterrupt:
        print("\n[dp-server] shutting down")
    finally:
        server.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
