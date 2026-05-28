"""
Pi0-TacForesight Inference Server.

10-step flow matching ODE with tactile-augmented prefix.
Connects to robot via WebSocket (same protocol as serve_dp_policy.py).

Usage:
    python -m pi0_tactile.serve \
        --ckpt_dir /home/chenshuai/Project/output/pi0_tactile_run1/step_30000 \
        --pi0_weights /home/chenshuai/Project/output/pi0_checkpoints/pi0_base
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
import torchvision.transforms as T

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pi0_tactile.config import Pi0TactileConfig
from pi0_tactile.model import Pi0Tactile
from for_show_xiaomi.ws_server import TactileACTServer, ClientDisconnected


def build_model(config: Pi0TactileConfig, ckpt_path: str, pi0_weights: str, device: torch.device):
    """Build and load Pi0Tactile model."""
    model = Pi0Tactile(config).to(device)

    # Load pi0 base weights first
    if pi0_weights:
        model.load_pi0_weights(pi0_weights)

    # Load fine-tuned checkpoint (overrides)
    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)
        step = ckpt.get("step", "?")
        print(f"[pi0-serve] Loaded checkpoint step={step}")

    model.eval()
    return model


def preprocess_image(raw_img: np.ndarray, transform) -> torch.Tensor:
    """Convert raw uint8 HWC image to CHW tensor in [-1, 1]."""
    if raw_img.dtype == np.uint8:
        img = raw_img.astype(np.float32) / 255.0
    else:
        img = np.asarray(raw_img, dtype=np.float32)
        if img.max() > 1.0:
            img = img / 255.0
    t = torch.from_numpy(img).permute(2, 0, 1).float()
    return transform(t)


def main():
    parser = argparse.ArgumentParser(description="Pi0-TacForesight Server")
    parser.add_argument("--ckpt_dir", type=str, required=True,
                        help="Checkpoint directory (contains checkpoint.pth + config.json)")
    parser.add_argument("--pi0_weights", type=str, default="",
                        help="Pi0 base weights directory")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--num_flow_steps", type=int, default=10)
    parser.add_argument("--action_horizon_exec", type=int, default=8,
                        help="Execute N steps of predicted chunk, then re-plan")
    parser.add_argument("--max_timesteps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )
    print(f"[pi0-serve] device: {device}")

    # Load config
    config_path = os.path.join(args.ckpt_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg_dict = json.load(f)
        config = Pi0TactileConfig(**{
            k: v for k, v in cfg_dict.items()
            if k in Pi0TactileConfig.__dataclass_fields__
        })
    else:
        config = Pi0TactileConfig()

    # Build model
    ckpt_path = os.path.join(args.ckpt_dir, "checkpoint.pth")
    model = build_model(config, ckpt_path, args.pi0_weights, device)
    print(f"[pi0-serve] Model loaded, action_dim={config.action_dim}, "
          f"action_horizon={config.action_horizon}")

    # Normalization stats
    import pickle
    stats_path = os.path.join(config.dataset_dir, "dataset_stats.pkl")
    if os.path.exists(stats_path):
        with open(stats_path, "rb") as f:
            norm_stats = pickle.load(f)
        action_min = np.array(norm_stats["action_min"], dtype=np.float32)
        action_max = np.array(norm_stats["action_max"], dtype=np.float32)
        qpos_min = np.array(norm_stats["qpos_min"], dtype=np.float32)
        qpos_max = np.array(norm_stats["qpos_max"], dtype=np.float32)
    else:
        print("[pi0-serve] WARNING: norm stats not found, using identity")
        action_min = np.zeros(config.action_dim, dtype=np.float32)
        action_max = np.ones(config.action_dim, dtype=np.float32)
        qpos_min = np.zeros(config.action_dim, dtype=np.float32)
        qpos_max = np.ones(config.action_dim, dtype=np.float32)

    marker_mean = np.array(config.marker_mean, dtype=np.float32)
    marker_std = np.array(config.marker_std, dtype=np.float32)

    # Image transform
    img_transform = T.Compose([
        T.Resize(config.image_size),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Fixed prompt tokens (placeholder)
    prompt_tokens = torch.zeros(1, config.max_token_len, dtype=torch.int32, device=device)
    prompt_mask = torch.zeros(1, config.max_token_len, dtype=torch.bool, device=device)
    prompt_text = config.fixed_prompt[:config.max_token_len]
    for i, c in enumerate(prompt_text):
        prompt_tokens[0, i] = ord(c)
        prompt_mask[0, i] = True

    # Inference params
    action_horizon_exec = min(args.action_horizon_exec, config.action_horizon)
    query_freq = action_horizon_exec

    print(f"[pi0-serve] flow_steps={args.num_flow_steps}, "
          f"exec_horizon={action_horizon_exec}, query_freq={query_freq}")

    # WebSocket server
    server = TactileACTServer(
        host=args.host,
        port=args.port,
        metadata={
            "protocol": "pi0_tactile",
            "action_dim": config.action_dim,
            "action_horizon": config.action_horizon,
            "num_flow_steps": args.num_flow_steps,
        },
    )
    server.start()
    print(f"[pi0-serve] listening on {args.host}:{args.port}")

    try:
        ep = 0
        while True:
            try:
                print(f"\n[pi0-serve] === episode {ep} ===")
                obs = server.recv_obs()
            except ClientDisconnected:
                continue

            marker_buffer = []

            with torch.inference_mode():
                try:
                    all_actions = None

                    for step in range(args.max_timesteps):
                        # === Preprocess observation ===
                        # Images
                        global_img = preprocess_image(
                            obs["images"]["global"], img_transform
                        ).unsqueeze(0).to(device)
                        wrist_img = preprocess_image(
                            obs["images"]["wrist"], img_transform
                        ).unsqueeze(0).to(device)

                        images_list = [global_img, wrist_img]
                        img_masks_list = [
                            torch.ones(1, dtype=torch.bool, device=device),
                            torch.ones(1, dtype=torch.bool, device=device),
                        ]

                        # qpos
                        qpos_raw = np.asarray(obs["qpos"], dtype=np.float32)
                        qpos_norm = (qpos_raw - qpos_min) / (qpos_max - qpos_min + 1e-8) * 2 - 1
                        state = torch.tensor(qpos_norm, dtype=torch.float32).unsqueeze(0).to(device)

                        # Marker offset
                        tac = obs["tac"]
                        side = list(tac.keys())[0]
                        side_data = tac[side]
                        marker_raw = side_data["marker_offset"] if isinstance(side_data, dict) else side_data
                        marker_raw = np.asarray(marker_raw, dtype=np.float32)
                        marker_norm = (marker_raw - marker_mean) / (marker_std + 1e-8)
                        marker_buffer.append(marker_norm)

                        # Build tactile window
                        T_hist = config.tac_history
                        if len(marker_buffer) >= T_hist:
                            marker_seq = np.stack(marker_buffer[-T_hist:], axis=0)
                        else:
                            pad_count = T_hist - len(marker_buffer)
                            pad = [marker_buffer[0]] * pad_count
                            marker_seq = np.stack(pad + list(marker_buffer), axis=0)

                        marker_tensor = torch.tensor(
                            marker_seq, dtype=torch.float32
                        ).unsqueeze(0).to(device)  # (1, T_hist, 9, 9, 2)

                        # === Inference (every query_freq steps) ===
                        if step % query_freq == 0:
                            all_actions = model.sample_actions(
                                images=images_list,
                                img_masks=img_masks_list,
                                lang_tokens=prompt_tokens,
                                lang_masks=prompt_mask,
                                state=state,
                                marker_offset=marker_tensor,
                                num_steps=args.num_flow_steps,
                            )  # (1, action_horizon, action_dim)

                        # Get current action from chunk
                        action_idx = step % query_freq
                        raw = all_actions[0, action_idx].cpu().numpy()

                        # Denormalize: [-1, 1] → absolute
                        action = (raw + 1) / 2 * (action_max - action_min) + action_min
                        action = action.astype(np.float32)

                        # Send to robot
                        server.send_action({
                            "actions": action[None, :],
                            "step": step,
                        })

                        # Receive next observation
                        if step + 1 < args.max_timesteps:
                            obs = server.recv_obs()

                except ClientDisconnected:
                    print(f"[pi0-serve] client disconnected at step {step}")

            ep += 1

    except KeyboardInterrupt:
        print("\n[pi0-serve] shutting down")
    finally:
        server.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
