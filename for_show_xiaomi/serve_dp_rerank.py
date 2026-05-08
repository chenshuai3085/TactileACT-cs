"""
Diffusion Policy Reranking TCP Server (runs on GPU machine).

Extends serve_dp_policy with CQF-based action reranking:
  1. DP generates K candidate action trajectories (DDPM sampling)
  2. LatentForesight predicts future tactile latent for each candidate
  3. CQF scores each (state, action, z_cur, z_pred)
  4. Select highest-scored trajectory and execute via receding horizon

Only supports variant: tactile_vae_frozen

Usage:
    cd /path/to/TactileACT-cs
    conda activate TactileACT
    python -m for_show_xiaomi.serve_dp_rerank \
      --ckpt_dir /path/to/dp_save_dir \
      --foresight_ckpt /path/to/foresight_best.ckpt \
      --foresight_dir /path/to/foresight_dir \
      --cqf_ckpt /path/to/cqf_latent_best.pt \
      --K 16 --port 8766
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
from collections import deque

import numpy as np
import torch
import torchvision.transforms as transforms
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_TFAC_V5 = os.path.join(_ROOT, "TFAC_V5")
if _TFAC_V5 not in sys.path:
    sys.path.insert(0, _TFAC_V5)

# Import TFAC_V5 modules first (before diffusion/ goes on path)
from cqf_model import ContactQualityScorer
from pretrain_latent_foresight import LatentForesightPretrainModel

_DIFFUSION = os.path.join(_ROOT, "diffusion")
if _DIFFUSION not in sys.path:
    sys.path.insert(0, _DIFFUSION)

from utils import set_seed
from network import ConditionalUnet1D
from for_show_xiaomi.ws_server import TactileACTServer, ClientDisconnected
from for_show_xiaomi.serve_dp_policy import (
    build_dp_model, load_checkpoint, _remap_legacy_vision_keys,
    preprocess_image, preprocess_obs, build_obs_cond,
)

_IMG_NORM = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)

TAC_MEAN = np.array([0.2102, -0.6422], dtype=np.float32)
TAC_STD = np.array([1.6805, 3.6717], dtype=np.float32)


def load_foresight(ckpt_path, foresight_dir, device):
    """Load LatentForesightPretrainModel and its norm stats."""
    if foresight_dir is None:
        foresight_dir = os.path.dirname(ckpt_path)

    args_path = os.path.join(foresight_dir, "args.json")
    with open(args_path) as f:
        config = json.load(f)

    camera_names = config['camera_names']
    cam_backbone_mapping = {cam: 0 for cam in camera_names}

    model = LatentForesightPretrainModel(
        camera_names=camera_names,
        cam_backbone_mapping=cam_backbone_mapping,
        hidden_dim=config['hidden_dim'],
        state_dim=config['state_dim'],
        foresight_layers=config.get('foresight_layers', 3),
        foresight_nheads=config.get('foresight_nheads', 8),
        foresight_dim_feedforward=config.get('foresight_dim_feedforward', 2048),
        dropout=config.get('dropout', 0.1),
        tactile_mode=config.get('tactile_mode', 'marker'),
        max_history=config.get('max_history', 8),
        predict_horizon=config.get('predict_horizon', 1),
        tactile_vae_ckpt=config.get('tactile_vae_ckpt'),
        tactile_vae_latent_dim=config.get('tactile_vae_latent_dim', 16),
        use_delta_pred=config.get('use_delta_pred', False),
        residual_prediction=config.get('residual_prediction', False),
    ).to(device)

    state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    n_loaded = len(state_dict) - len(unexpected)
    print(f"[rerank] Foresight loaded: {n_loaded} keys, "
          f"{len(missing)} missing (backbone/VAE), {len(unexpected)} unexpected")
    model.eval()

    stats_path = os.path.join(foresight_dir, "dataset_stats.pkl")
    if os.path.exists(stats_path):
        with open(stats_path, 'rb') as f:
            ns = pickle.load(f)
        print(f"[rerank] Foresight norm stats from dataset_stats.pkl")
    else:
        ns = config.get('norm_stats', config)
        print(f"[rerank] Foresight norm stats from args.json")

    norm_stats = {
        'qpos_mean': torch.tensor(ns['qpos_mean'], dtype=torch.float32, device=device),
        'qpos_std': torch.tensor(ns['qpos_std'], dtype=torch.float32, device=device),
        'action_mean': torch.tensor(ns['action_mean'], dtype=torch.float32, device=device),
        'action_std': torch.tensor(ns['action_std'], dtype=torch.float32, device=device),
    }

    fs_config = {
        'chunk_size': config.get('chunk_size', 10),
        'vae_window': config.get('tactile_vae_window', 8) or 8,
    }

    return model, norm_stats, fs_config


def load_cqf(ckpt_path, device):
    """Load ContactQualityScorer from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    model = ContactQualityScorer(
        tac_dim=args.get("tac_dim", 144),
        chunk_size=args.get("chunk_size", 16),
        hidden=args.get("hidden", 256),
        action_dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    metrics = ckpt.get('val_metrics', {})
    print(f"[rerank] CQF loaded (epoch {ckpt.get('epoch', '?')}, "
          f"rank1={metrics.get('rank1_acc', '?')}, "
          f"corr={metrics.get('spearman_corr', '?')})")
    return model


@torch.no_grad()
def ddpm_inference_K(noise_pred_net, noise_scheduler, obs_cond, action_dim,
                     pred_horizon, num_inference_steps, K, device):
    """Run DDPM denoising to generate K candidate action chunks."""
    noise_scheduler.set_timesteps(num_inference_steps)
    obs_cond_K = obs_cond.expand(K, -1)
    action = torch.randn((K, pred_horizon, action_dim), device=device)

    for t in noise_scheduler.timesteps:
        noise_pred = noise_pred_net(
            action, t.unsqueeze(0).to(device), global_cond=obs_cond_K
        )
        action = noise_scheduler.step(noise_pred, t, action).prev_sample

    return action  # (K, pred_horizon, action_dim) in [-1, 1]


@torch.no_grad()
def score_and_select(candidates_norm, qpos_raw, marker_buffer, marker_step,
                     obs_raw_images, foresight, fs_norm, fs_config,
                     cqf, action_min_t, action_max_t, device):
    """Score K candidates with Foresight+CQF and return best index.

    Args:
        candidates_norm: (K, pred_horizon, 7) in [-1, 1]
        qpos_raw: (7,) numpy
        marker_buffer: list of raw marker_offset frames
        marker_step: current step index in marker_buffer
        obs_raw_images: dict {cam: (H,W,3) uint8} current frame raw images
        foresight: LatentForesightPretrainModel
        fs_norm: dict with qpos_mean/std, action_mean/std tensors
        fs_config: dict with chunk_size, vae_window
        cqf: ContactQualityScorer
        action_min_t, action_max_t: tensors for denormalization
        device: torch device
    """
    K = candidates_norm.shape[0]

    # Denormalize candidates to raw action space
    actions_raw = (candidates_norm + 1) / 2 * (action_max_t - action_min_t) + action_min_t

    # --- Prepare foresight inputs ---
    # 1. Images: global + wrist at original resolution (no DP resize/crop)
    fs_images = []
    for cam_name in ['global', 'wrist']:
        if cam_name in obs_raw_images:
            img_uint8 = obs_raw_images[cam_name]
            img = np.asarray(img_uint8, dtype=np.float32)
            if img.max() > 1.0:
                img = img / 255.0
            img_t = torch.from_numpy(img).permute(2, 0, 1).float()
            img_t = _IMG_NORM(img_t).unsqueeze(0).to(device)
        else:
            img_t = torch.zeros(1, 3, 200, 266, device=device)
        fs_images.append(img_t.expand(K, -1, -1, -1))

    # 2. Marker window for foresight (normalized)
    vae_window = fs_config['vae_window']
    marker_frames = []
    for i in range(vae_window):
        idx = marker_step - vae_window + 1 + i
        idx = max(0, min(idx, len(marker_buffer) - 1))
        marker_frames.append(marker_buffer[idx])
    marker_window = np.stack(marker_frames)  # (W, 9, 9, 2)
    marker_window_t = torch.tensor(marker_window, dtype=torch.float32)
    tac_mean_t = torch.tensor(TAC_MEAN).reshape(1, 1, 1, 2)
    tac_std_t = torch.tensor(TAC_STD).reshape(1, 1, 1, 2)
    marker_window_norm = (marker_window_t - tac_mean_t) / tac_std_t
    marker_window_dev = marker_window_norm.unsqueeze(0).to(device)  # (1, W, 9, 9, 2)
    fs_images.append(marker_window_dev.expand(K, -1, -1, -1, -1))

    # 3. Normalize actions for foresight (mean/std)
    fs_chunk = fs_config['chunk_size']
    action_fs = actions_raw[:, :fs_chunk, :]
    action_fs_norm = (action_fs - fs_norm['action_mean']) / fs_norm['action_std']

    # 4. Normalize qpos for foresight
    qpos_t = torch.tensor(qpos_raw, dtype=torch.float32, device=device)
    qpos_fs_norm = ((qpos_t - fs_norm['qpos_mean']) / fs_norm['qpos_std']).unsqueeze(0).expand(K, -1)

    # --- Foresight forward ---
    z_pred, _, _, z_current, _, _ = foresight(fs_images, action_fs_norm, qpos=qpos_fs_norm)
    # z_pred: (K, 144), z_current: (K, 144)

    # Use z_current from foresight (it encodes current marker window via TactileVAE)
    z_cur = z_current  # (K, 144) — already batch-expanded inside foresight

    # --- CQF scoring ---
    qpos_K = qpos_t.unsqueeze(0).expand(K, -1)
    action_cqf = actions_raw[:, :16, :]

    scores = cqf.predict(qpos_K, action_cqf, z_cur, z_pred).squeeze(-1)  # (K,)

    best_idx = scores.argmax().item()
    return best_idx, scores


def main():
    parser = argparse.ArgumentParser(description="Diffusion Policy Reranking Server")
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--ckpt_name", type=str, default="dp_epoch250.pth")
    parser.add_argument("--foresight_ckpt", type=str, required=True)
    parser.add_argument("--foresight_dir", type=str, default=None)
    parser.add_argument("--cqf_ckpt", type=str, required=True)
    parser.add_argument("--K", type=int, default=16,
                        help="Number of candidates to generate and score")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--action_horizon", type=int, default=8)
    parser.add_argument("--action_skip", type=int, default=0)
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--max_timesteps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    cli = parser.parse_args()

    set_seed(cli.seed)
    device = torch.device(
        f"cuda:{cli.gpu}" if torch.cuda.is_available() and cli.gpu >= 0 else "cpu"
    )
    print(f"[rerank-server] device: {device}")

    # --- Load DP config ---
    config_path = os.path.join(cli.ckpt_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    variant = config["variant"]
    assert variant == "tactile_vae_frozen", \
        f"Reranking server only supports tactile_vae_frozen, got {variant}"

    camera_names = config["camera_names"]
    if isinstance(camera_names, str):
        camera_names = camera_names.split(",")
    action_dim = config["action_dim"]
    pred_horizon = config["pred_horizon"]
    obs_horizon = config.get("obs_horizon", 2)
    num_train_timesteps = config.get("num_train_timesteps", 100)
    num_inference_steps = cli.num_inference_steps or config.get("num_inference_steps", 100)
    tac_history = config.get("tac_history", 8)

    ns = config["norm_stats"]
    action_min = np.array(ns["action_min"], dtype=np.float32)
    action_max = np.array(ns["action_max"], dtype=np.float32)
    qpos_min = np.array(ns["qpos_min"], dtype=np.float32)
    qpos_max = np.array(ns["qpos_max"], dtype=np.float32)
    action_min_t = torch.tensor(action_min, device=device)
    action_max_t = torch.tensor(action_max, device=device)

    resize_shape = config.get("resize_shape")
    crop_shape = config.get("crop_shape")
    resize_tf = transforms.Resize(tuple(resize_shape)) if resize_shape else None
    crop_tf = transforms.CenterCrop(tuple(crop_shape)) if crop_shape else None

    print(f"[rerank-server] variant={variant}, cameras={camera_names}")
    print(f"[rerank-server] action_dim={action_dim}, pred_horizon={pred_horizon}, "
          f"obs_horizon={obs_horizon}, K={cli.K}")

    # --- Load DP models ---
    vision_encoder, noise_pred_net, tac_encoder = build_dp_model(config, device)
    ckpt_path = os.path.join(cli.ckpt_dir, cli.ckpt_name)
    load_checkpoint(ckpt_path, vision_encoder, noise_pred_net, camera_names, device)
    vision_encoder.eval()
    noise_pred_net.eval()
    if tac_encoder is not None:
        tac_encoder.eval()

    # --- Load Foresight ---
    foresight, fs_norm, fs_config = load_foresight(
        cli.foresight_ckpt, cli.foresight_dir, device)

    # --- Load CQF ---
    cqf = load_cqf(cli.cqf_ckpt, device)

    # --- DDPM scheduler ---
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )
    print(f"[rerank-server] DDPM: {num_inference_steps} inference steps")

    # --- Receding horizon ---
    action_skip = cli.action_skip
    action_horizon = min(cli.action_horizon, pred_horizon - action_skip)
    query_freq = action_horizon
    print(f"[rerank-server] action_horizon={action_horizon}, query_freq={query_freq}")

    # --- TCP Server ---
    server = TactileACTServer(
        host=cli.host,
        port=cli.port,
        metadata={
            "protocol": "diffusion_policy_rerank",
            "variant": variant,
            "camera_names": camera_names,
            "action_dim": action_dim,
            "pred_horizon": pred_horizon,
            "action_horizon": action_horizon,
            "K": cli.K,
        },
    )
    server.start()
    print(f"[rerank-server] listening on {cli.host}:{cli.port}  (waiting for client...)")

    try:
        ep = 0
        while True:
            try:
                print(f"\n[rerank-server] === episode {ep} ===")
                obs = server.recv_obs()
            except ClientDisconnected:
                print("[rerank-server] client gone before first obs, waiting...")
                continue

            obs_buffer = deque(maxlen=obs_horizon)
            marker_buffer = []
            marker_step = 0
            current_raw_images = {}

            with torch.inference_mode():
                try:
                    for step in range(cli.max_timesteps):
                        # Save raw images for foresight (before DP preprocessing)
                        current_raw_images = obs.get("images", {})

                        processed = preprocess_obs(
                            obs, camera_names, variant, device,
                            resize_tf=resize_tf, crop_tf=crop_tf,
                        )

                        if variant == "tactile_vae_frozen":
                            marker_buffer.append(processed["marker_offset"])
                            processed["_marker_idx"] = marker_step
                            marker_step += 1

                        obs_buffer.append(processed)

                        while len(obs_buffer) < obs_horizon:
                            pad = dict(processed)
                            pad["_marker_idx"] = 0
                            obs_buffer.appendleft(pad)

                        obs_cond = build_obs_cond(
                            obs_buffer, vision_encoder, tac_encoder, variant,
                            camera_names, qpos_min, qpos_max, tac_history,
                            marker_buffer, device,
                        )

                        if step % query_freq == 0:
                            # Generate K candidates
                            all_candidates = ddpm_inference_K(
                                noise_pred_net, noise_scheduler, obs_cond,
                                action_dim, pred_horizon,
                                num_inference_steps, cli.K, device,
                            )  # (K, pred_horizon, 7) normalized

                            # Score and select best
                            best_idx, scores = score_and_select(
                                all_candidates,
                                processed["qpos_raw"],
                                marker_buffer, marker_step - 1,
                                current_raw_images,
                                foresight, fs_norm, fs_config,
                                cqf, action_min_t, action_max_t, device,
                            )

                            best_actions = all_candidates[best_idx:best_idx+1]  # (1, H, 7)

                            if step % (query_freq * 5) == 0:
                                score_spread = scores.max() - scores.min()
                                print(f"  step {step}: best={best_idx}, "
                                      f"score={scores[best_idx]:.3f}, "
                                      f"spread={score_spread:.3f}")

                        # Extract current step's action from best trajectory
                        raw = best_actions[:, action_skip + step % query_freq]
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
                    print(f"[rerank-server] client disconnected at step {step}")

            ep += 1
    except KeyboardInterrupt:
        print("\n[rerank-server] shutting down")
    finally:
        server.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
