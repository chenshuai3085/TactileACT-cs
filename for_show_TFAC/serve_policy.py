"""
TFAC TCP Policy Server (runs on GPU machine)

Usage:
    cd /path/to/TactileACT-cs
    conda activate TactileACT
    python -m for_show_TFAC.serve_policy --ckpt_dir /path/to/ckpt_dir

ckpt_dir should contain:
    - args.json          (training config)
    - dataset_stats.pkl  (normalization stats)
    - policy_best.ckpt   (model weights)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms

# Project root
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from TFAC.tfac_policy import TFACPolicy
from utils import NormalizeSeparate, set_seed
from for_show_xiaomi.ws_server import TactileACTServer, ClientDisconnected


FREEZE_TACTILE = True


def build_policy(args: dict) -> TFACPolicy:
    """Build TFACPolicy from saved training args."""
    camera_names = args["camera_names"]
    state_dim = args["state_dim"]

    pretrained_backbones = None
    camera_backbone_mapping = None
    tactile_mode = args.get("tactile_mode", "image")

    if args.get("backbone") == "clip_backbone":
        try:
            from clip_pretraining_xiaomi import modified_resnet18
        except ImportError:
            from clip_pretraining import modified_resnet18
        vision_model = modified_resnet18()
        camera_backbone_mapping = {c: 0 for c in camera_names}

        if tactile_mode == "image":
            gelsight_model = modified_resnet18()
            camera_backbone_mapping["gelsight"] = 1
            if FREEZE_TACTILE:
                gelsight_model.requires_grad_(False)
            pretrained_backbones = [vision_model, gelsight_model]
        else:
            camera_backbone_mapping["gelsight"] = 0
            pretrained_backbones = [vision_model]

    return TFACPolicy(
        state_dim=state_dim,
        hidden_dim=args.get("hidden_dim", 512),
        position_embedding_type=args.get("position_embedding", "sine"),
        lr_backbone=args.get("lr_backbone", 1e-5),
        masks=args.get("masks", False),
        backbone_type=args.get("backbone", "resnet18"),
        dilation=args.get("dilation", False),
        dropout=args.get("dropout", 0.1),
        nheads=args.get("nheads", 8),
        dim_feedforward=args.get("dim_feedforward", 2048),
        num_enc_layers=args.get("enc_layers", 4),
        num_dec_layers=args.get("dec_layers", 7),
        pre_norm=args.get("pre_norm", False),
        num_queries=args.get("chunk_size", 20),
        camera_names=camera_names,
        z_dimension=args.get("z_dimension", 32),
        lr=args.get("lr", 1e-5),
        weight_decay=args.get("weight_decay", 1e-4),
        kl_weight=args.get("kl_weight", 10),
        pretrained_backbones=pretrained_backbones,
        cam_backbone_mapping=camera_backbone_mapping,
        # TFAC specific
        foresight_layers=args.get("foresight_layers", 2),
        foresight_nheads=args.get("foresight_nheads", 4),
        foresight_dim_feedforward=args.get("foresight_dim_feedforward", 2048),
        proj_dim=args.get("proj_dim", 128),
        contrastive_temperature=args.get("contrastive_temperature", 0.07),
        curriculum_ratio=args.get("curriculum_ratio", 0.75),
        lambda_draft=args.get("lambda_draft", 0.5),
        lambda_foresight=args.get("lambda_foresight", 1.0),
        lambda_foresight_vis=args.get("lambda_foresight_vis", 0.3),
        lambda_contrastive=args.get("lambda_contrastive", 0.1),
        num_dec_layers_draft=args.get("dec_layers_draft", None),
        foresight_change_weight=args.get("foresight_change_weight", False),
        # V4 modularity
        tactile_mode=args.get("tactile_mode", "image"),
        marker_encoder_type=args.get("marker_encoder_type", "conv2d"),
        fusion_mode=args.get("fusion_mode", "gate"),
        foresight_tac_decoder=args.get("foresight_tac_decoder", "linear"),
        a2_init=args.get("a2_init", "zero"),
    )


_IMG_NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])


def preprocess_images(obs: dict, camera_names: list,
                      norm_stats: dict, device: torch.device,
                      tactile_mode: str = "image") -> list:
    """
    Convert raw obs into list of tensors matching TFACPolicy input.

    obs format from client:
      obs["images"][cam_name] -> (H,W,3) uint8
      obs["tac"]["left"]["img"] -> (H,W,3) uint8   (maps to "gelsight" camera)
      obs["tac"]["left"]["marker_offset"] -> (9,9,2) float32  (marker mode)
    """
    all_images = []
    for cam_name in camera_names:
        if cam_name == "gelsight":
            if tactile_mode == "marker":
                # --- marker_offset mode ---
                if "tac" in obs:
                    tac = obs["tac"]
                    side = list(tac.keys())[0]
                    side_data = tac[side]
                    mo = side_data.get("marker_offset") if isinstance(side_data, dict) else None
                    if mo is None:
                        mo = np.zeros((9, 9, 2), dtype=np.float32)
                    mo = np.asarray(mo, dtype=np.float32)
                else:
                    mo = np.zeros((9, 9, 2), dtype=np.float32)
                t = torch.from_numpy(mo).float()
                # normalize marker_offset if stats available
                mo_mean = norm_stats.get("marker_offset_mean")
                mo_std = norm_stats.get("marker_offset_std")
                if mo_mean is not None:
                    t = (t - torch.tensor(mo_mean, dtype=torch.float32)) / torch.tensor(mo_std, dtype=torch.float32)
                all_images.append(t.unsqueeze(0).to(device))  # (1, 9, 9, 2)
            else:
                # --- tactile image mode ---
                if "tac" in obs:
                    tac = obs["tac"]
                    side = list(tac.keys())[0]
                    side_data = tac[side]
                    img = side_data["img"] if isinstance(side_data, dict) else side_data
                    img = np.asarray(img, dtype=np.float32)
                    if img.max() > 1.0:
                        img = img / 255.0
                    t = torch.from_numpy(img).permute(2, 0, 1).float()
                    t = _IMG_NORM(t)
                elif "gelsight" in obs:
                    gs = np.asarray(obs["gelsight"], dtype=np.float32)
                    gs_mean = norm_stats.get("gelsight_mean")
                    gs_std = norm_stats.get("gelsight_std")
                    if gs_mean is not None:
                        gs = (gs - gs_mean) / gs_std
                    t = torch.from_numpy(gs).permute(2, 0, 1).float()
                else:
                    raise KeyError(f"No tactile data for 'gelsight' camera. obs keys: {list(obs.keys())}")
                all_images.append(t.unsqueeze(0).to(device))

        elif cam_name == "blank":
            all_images.append(torch.zeros(1, 3, 480, 640, device=device))

        else:
            # --- regular camera ---
            if "images" in obs and cam_name in obs["images"]:
                img = obs["images"][cam_name]
            elif cam_name in obs:
                img = obs[cam_name]
            else:
                raise KeyError(f"Camera '{cam_name}' not in obs")
            img = np.asarray(img, dtype=np.float32)
            if img.max() > 1.0:
                img = img / 255.0
            t = torch.from_numpy(img).permute(2, 0, 1).float()
            t = _IMG_NORM(t)
            all_images.append(t.unsqueeze(0).to(device))

    return all_images


def main():
    parser = argparse.ArgumentParser(description="TFAC Policy Server")
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--ckpt_name", type=str, default="policy_best.ckpt")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--temporal_agg", action="store_true")
    parser.add_argument("--max_timesteps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1)
    cli = parser.parse_args()

    set_seed(cli.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[server] device: {device}")

    # --- load training config ---
    args_path = os.path.join(cli.ckpt_dir, "args.json")
    with open(args_path) as f:
        train_args = json.load(f)

    stats_path = os.path.join(cli.ckpt_dir, "dataset_stats.pkl")
    with open(stats_path, "rb") as f:
        norm_stats = pickle.load(f)

    # norm_stats embedded in args.json (as lists) -> convert to numpy
    if "norm_stats" in train_args:
        for k, v in train_args["norm_stats"].items():
            if k not in norm_stats:
                norm_stats[k] = np.array(v)

    normalizer = NormalizeSeparate(norm_stats)
    camera_names = train_args["camera_names"]
    state_dim = train_args["state_dim"]
    chunk_size = train_args["chunk_size"]
    temporal_agg = cli.temporal_agg or train_args.get("temporal_agg", False)
    tactile_mode = train_args.get("tactile_mode", "image")

    print(f"[server] TFAC model | cameras={camera_names}  state_dim={state_dim}  "
          f"chunk={chunk_size}  temporal_agg={temporal_agg}")

    # --- build & load model ---
    policy = build_policy(train_args)
    ckpt_path = os.path.join(cli.ckpt_dir, cli.ckpt_name)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        ckpt = ckpt["model"]
    result = policy.load_state_dict(ckpt, strict=False)
    if result.missing_keys:
        print(f"  warning missing keys: {result.missing_keys}")
    if result.unexpected_keys:
        print(f"  warning unexpected keys: {result.unexpected_keys}")
    policy.to(device)
    policy.eval()
    print(f"[server] loaded {ckpt_path}")

    query_freq = 1 if temporal_agg else chunk_size

    # --- start server ---
    server = TactileACTServer(
        host=cli.host, port=cli.port,
        metadata={"protocol": "tfac", "camera_names": camera_names,
                   "state_dim": state_dim, "chunk_size": chunk_size,
                   "temporal_agg": temporal_agg},
    )
    server.start()
    print(f"[server] listening on {cli.host}:{cli.port}  (waiting for client...)")

    # Logging setup
    has_gate = hasattr(policy.model, 'gated_fusion')
    log_dir = os.path.join(cli.ckpt_dir, "inference_logs")
    os.makedirs(log_dir, exist_ok=True)
    if has_gate:
        print(f"[server] gate fusion detected, will save weight plots to {log_dir}")

    # Enable cross-attention hook for token fusion mode
    fusion_mode = train_args.get("fusion_mode", "gate")
    policy.model.enable_attn_hooks()
    print(f"[server] attention hooks enabled (fusion_mode={fusion_mode})")

    try:
        ep = 0
        while True:
            try:
                print(f"\n[server] === episode {ep} ===")
                obs = server.recv_obs()
            except ClientDisconnected:
                print("[server] client gone before first obs, waiting...")
                continue

            if temporal_agg:
                all_time_actions = torch.zeros(
                    [cli.max_timesteps, cli.max_timesteps + chunk_size, state_dim],
                    device=device)

            gate_history = []
            attn_history = []  # cross-attention maps per query step
            obs_history = []   # raw input data per step (images + tactile)

            with torch.inference_mode():
                try:
                    for t in range(cli.max_timesteps):
                        # Save raw input data for visualization
                        obs_record = {"step": t}
                        # Camera images
                        for cam_name in camera_names:
                            if cam_name == "gelsight":
                                continue  # tactile handled separately
                            if "images" in obs and cam_name in obs["images"]:
                                obs_record[cam_name] = np.asarray(obs["images"][cam_name], dtype=np.uint8)
                            elif cam_name in obs:
                                obs_record[cam_name] = np.asarray(obs[cam_name], dtype=np.uint8)
                        # Tactile data
                        if "tac" in obs:
                            side = list(obs["tac"].keys())[0]
                            side_data = obs["tac"][side]
                            if isinstance(side_data, dict):
                                if "marker_offset" in side_data:
                                    obs_record["marker_offset"] = np.asarray(
                                        side_data["marker_offset"], dtype=np.float32)
                                if "img" in side_data:
                                    obs_record["tac_img"] = np.asarray(
                                        side_data["img"], dtype=np.uint8)
                        obs_history.append(obs_record)

                        # preprocess
                        qpos_raw = np.asarray(obs["qpos"], dtype=np.float32)
                        qpos_n = normalizer.normalize_qpos(qpos_raw)
                        qpos_t = torch.from_numpy(qpos_n).float().unsqueeze(0).to(device)
                        imgs = preprocess_images(obs, camera_names, norm_stats, device,
                                                 tactile_mode=tactile_mode)

                        # inference: TFACPolicy returns A2 (refined action)
                        if t % query_freq == 0:
                            all_actions = policy(qpos_t, imgs)  # (1, chunk, dim)

                            # Collect cross-attention weights (only on query steps)
                            attn_w = policy.model._attn_weights.get('decoder2_cross')
                            if attn_w is not None:
                                attn_history.append((t, attn_w.numpy()))  # (B, num_queries, memory_len)

                        # Record gate weights
                        if has_gate:
                            gm, ga, gf = policy.model.gated_fusion._last_gate_means
                            gate_history.append((gm, ga, gf))

                        if temporal_agg:
                            all_time_actions[t, t:t+chunk_size] = all_actions.squeeze(0)
                            col = all_time_actions[:, t]
                            mask = torch.all(col != 0, dim=1)
                            col = col[mask]
                            k = 0.9
                            w = np.exp(-k * np.arange(len(col)))
                            w = w / w.sum()
                            w = torch.from_numpy(w).to(device).unsqueeze(1).float()
                            raw = (col * w).sum(dim=0, keepdim=True)
                        else:
                            raw = all_actions[:, t % query_freq]

                        raw_np = raw.squeeze(0).cpu().numpy()

                        # un-normalize -> absolute action (joint angles)
                        action = normalizer.unnormalize_action(raw_np)
                        action = action.astype(np.float32)

                        server.send_action({
                            "actions": action[None, :],
                            "step": t,
                        })

                        if t + 1 < cli.max_timesteps:
                            obs = server.recv_obs()

                except ClientDisconnected:
                    print(f"[server] client disconnected at step {t}")

            # Save gate weight plot after each episode
            if gate_history:
                gate_arr = np.array(gate_history)  # (T, 3)
                fig, ax = plt.subplots(figsize=(14, 5))
                ax.plot(gate_arr[:, 0], label='memory', alpha=0.8)
                ax.plot(gate_arr[:, 1], label='a1_draft', alpha=0.8)
                ax.plot(gate_arr[:, 2], label='future_tac', alpha=0.8)
                ax.set_xlabel('Timestep')
                ax.set_ylabel('Gate Weight')
                ax.set_title(f'Episode {ep} — Gate Weights (Real Robot Inference)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plot_path = os.path.join(log_dir, f'ep{ep}_gate_weights.png')
                plt.savefig(plot_path, dpi=150)
                plt.close()
                print(f"[server] gate weights saved: {plot_path}")
                print(f"[server] gate avg: mem={gate_arr[:,0].mean():.3f} "
                      f"a1={gate_arr[:,1].mean():.3f} fut={gate_arr[:,2].mean():.3f}")

            # Save cross-attention maps after each episode
            if attn_history:
                # attn_w shape: (B, num_queries, memory_len), B=1 at inference
                # For token fusion: last memory position = foresight token
                timesteps = [item[0] for item in attn_history]
                # Average attention across all 20 action queries → (memory_len,) per step
                attn_avg_per_step = []
                for _, aw in attn_history:
                    attn_avg_per_step.append(aw[0].mean(axis=0))  # (memory_len,)
                attn_matrix = np.array(attn_avg_per_step)  # (num_steps, memory_len)
                memory_len = attn_matrix.shape[1]

                fig, axes = plt.subplots(2, 1, figsize=(16, 10))

                # Top: full attention heatmap over time
                ax = axes[0]
                im = ax.imshow(attn_matrix.T, aspect='auto', cmap='hot',
                               interpolation='nearest')
                ax.set_xlabel('Query Timestep')
                ax.set_ylabel('Memory Token Index')
                ax.set_title(f'Episode {ep} — Decoder₂ Cross-Attention Map')
                # Label key memory regions
                ax.axhline(y=1.5, color='cyan', linestyle='--', alpha=0.5, label='latent|proprio')
                if fusion_mode == "token":
                    ax.axhline(y=memory_len - 1.5, color='lime', linestyle='--',
                               alpha=0.7, label='foresight token')
                ax.legend(loc='upper right', fontsize=8)
                plt.colorbar(im, ax=ax, label='attention weight')

                # Bottom: foresight token attention weight over time
                ax2 = axes[1]
                if fusion_mode == "token":
                    foresight_attn = attn_matrix[:, -1]  # last token = foresight
                    ax2.plot(timesteps, foresight_attn, color='green', linewidth=2,
                             label='foresight token')
                # Also plot average attention to vision / tactile tokens
                # memory layout: [latent, proprio, vision_tokens..., tactile_tokens..., (foresight)]
                n_prefix = 2  # latent + proprio
                end_idx = memory_len - 1 if fusion_mode == "token" else memory_len
                if end_idx > n_prefix:
                    other_avg = attn_matrix[:, n_prefix:end_idx].mean(axis=1)
                    ax2.plot(timesteps, other_avg, color='blue', alpha=0.6,
                             label='vision+tactile avg')
                ax2.set_xlabel('Timestep')
                ax2.set_ylabel('Attention Weight')
                ax2.set_title(f'Episode {ep} — Foresight Token Attention over Time')
                ax2.legend()
                ax2.grid(True, alpha=0.3)

                plt.tight_layout()
                attn_path = os.path.join(log_dir, f'ep{ep}_cross_attention.png')
                plt.savefig(attn_path, dpi=150)
                plt.close()
                print(f"[server] cross-attention map saved: {attn_path}")

                # Save raw data for further analysis
                np.savez(os.path.join(log_dir, f'ep{ep}_cross_attention.npz'),
                         timesteps=np.array(timesteps),
                         attn_matrix=attn_matrix)

            # Save input observations (images + tactile) for each episode
            if obs_history:
                ep_obs_dir = os.path.join(log_dir, f'ep{ep}_obs')
                os.makedirs(ep_obs_dir, exist_ok=True)

                marker_offsets = []
                for rec in obs_history:
                    step_i = rec["step"]
                    # Save camera images (every 10 steps to avoid too many files)
                    if step_i % 10 == 0:
                        for cam_name in camera_names:
                            if cam_name == "gelsight":
                                continue
                            if cam_name in rec:
                                from PIL import Image as PILImage
                                img = PILImage.fromarray(rec[cam_name])
                                img.save(os.path.join(ep_obs_dir,
                                         f'step{step_i:03d}_{cam_name}.jpg'), quality=85)
                        # Save tactile image if available
                        if "tac_img" in rec:
                            from PIL import Image as PILImage
                            tac_img = PILImage.fromarray(rec["tac_img"])
                            tac_img.save(os.path.join(ep_obs_dir,
                                         f'step{step_i:03d}_tactile.jpg'), quality=85)
                    # Collect marker offsets (all steps)
                    if "marker_offset" in rec:
                        marker_offsets.append(rec["marker_offset"])

                # Save all marker offsets as single npz
                if marker_offsets:
                    mo_arr = np.stack(marker_offsets)  # (T, 9, 9, 2)
                    np.save(os.path.join(ep_obs_dir, 'marker_offsets.npy'), mo_arr)

                    # Visualize marker offset magnitude over time
                    magnitudes = np.sqrt((mo_arr ** 2).sum(axis=-1))  # (T, 9, 9)
                    mean_mag = magnitudes.mean(axis=(1, 2))  # (T,)
                    max_mag = magnitudes.max(axis=(1, 2))    # (T,)

                    fig, ax = plt.subplots(figsize=(14, 4))
                    ax.plot(mean_mag, label='mean magnitude', alpha=0.8)
                    ax.plot(max_mag, label='max magnitude', alpha=0.6)
                    ax.set_xlabel('Timestep')
                    ax.set_ylabel('Marker Offset Magnitude (px)')
                    ax.set_title(f'Episode {ep} — Tactile Marker Offset over Time')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(ep_obs_dir, 'marker_offset_timeline.png'), dpi=150)
                    plt.close()

                print(f"[server] obs data saved: {ep_obs_dir}/ "
                      f"({len(obs_history)} steps, images every 10 steps)")

            ep += 1
    except KeyboardInterrupt:
        print("\n[server] shutting down")
    finally:
        policy.model.disable_attn_hooks()
        server.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
