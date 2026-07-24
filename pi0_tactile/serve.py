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

import numpy as np
import torch
import torchvision.transforms as T

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pi0_tactile.config import Pi0TactileConfig
from pi0_tactile.model import Pi0Tactile
from pi0_tactile.score_bridge import Pi0TactileForesightScoreBridge
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


def fit_stat_dim(arr: np.ndarray, dim: int, pad_value: float) -> np.ndarray:
    """Slice or pad a 1-D normalization stat to the executable robot dim."""
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    if arr.shape[0] >= dim:
        return arr[:dim]
    pad = np.full(dim - arr.shape[0], float(pad_value), dtype=np.float32)
    return np.concatenate([arr, pad], axis=0)


def build_scorer(runtime: str, checkpoint: str, device: torch.device):
    """Build a TacQuality scorer runtime for flow guidance."""
    runtime = runtime.lower()
    if runtime == "force_band":
        from TFAC_V5.tac_quality_energy.force_band_runtime import ForceBandTacQualityEnergyRuntime

        return ForceBandTacQualityEnergyRuntime(checkpoint, device=str(device)).to(device)
    if runtime == "distilled":
        from TFAC_V5.tac_quality_energy.runtime import DistilledTacQualityEnergyRuntime

        return DistilledTacQualityEnergyRuntime(checkpoint, device=str(device)).to(device)
    raise ValueError(f"Unsupported flow guidance scorer runtime: {runtime}")


def main():
    parser = argparse.ArgumentParser(description="Pi0-TacForesight Server")
    parser.add_argument("--ckpt_dir", type=str, required=True,
                        help="Checkpoint directory (contains checkpoint.pth + config.json)")
    parser.add_argument("--pi0_weights", type=str, default="",
                        help="Pi0 base weights directory")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--num_flow_steps", type=int, default=10)
    parser.add_argument("--robot_action_dim", type=int, default=None,
                        help="Executable robot action dim; pi0.5 model dim may stay 32.")
    parser.add_argument("--action_horizon_exec", type=int, default=8,
                        help="Execute N steps of predicted chunk, then re-plan")
    parser.add_argument("--max_timesteps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--flow_guidance_steps", type=int, default=None,
                        help="Late flow steps to guide; 0 disables guidance.")
    parser.add_argument("--flow_guidance_scale", type=float, default=None)
    parser.add_argument("--flow_guidance_max_total_delta", type=float, default=None)
    parser.add_argument("--flow_guidance_lambda_smooth", type=float, default=None)
    parser.add_argument("--flow_guidance_scorer_ckpt", type=str, default="",
                        help="TacQuality scorer checkpoint. Required to enable real flow guidance.")
    parser.add_argument("--flow_guidance_scorer_runtime", choices=["force_band", "distilled"], default="force_band")
    parser.add_argument("--flow_guidance_score_mode", type=str, default="energy_clipped")
    parser.add_argument("--flow_guidance_score_window", type=int, default=8)
    parser.add_argument("--flow_guidance_task_id", type=int, default=None)
    parser.add_argument("--send_guidance_report", action="store_true")
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
    if args.robot_action_dim is not None:
        config.robot_action_dim = int(args.robot_action_dim)
    if args.flow_guidance_steps is not None:
        config.flow_guidance_steps = int(args.flow_guidance_steps)
    if args.flow_guidance_scale is not None:
        config.flow_guidance_scale = float(args.flow_guidance_scale)
    if args.flow_guidance_max_total_delta is not None:
        config.flow_guidance_max_total_delta = float(args.flow_guidance_max_total_delta)
    if args.flow_guidance_lambda_smooth is not None:
        config.flow_guidance_lambda_smooth = float(args.flow_guidance_lambda_smooth)

    # Build model
    ckpt_path = os.path.join(args.ckpt_dir, "checkpoint.pth")
    model = build_model(config, ckpt_path, args.pi0_weights, device)
    print(f"[pi0-serve] Model loaded, action_dim={config.action_dim}, "
          f"robot_action_dim={config.robot_action_dim}, "
          f"action_horizon={config.action_horizon}")

    flow_scorer = None
    flow_guidance_enabled = (
        int(config.flow_guidance_steps) > 0
        and float(config.flow_guidance_scale) > 0.0
        and bool(args.flow_guidance_scorer_ckpt)
    )
    if flow_guidance_enabled:
        flow_scorer = build_scorer(args.flow_guidance_scorer_runtime, args.flow_guidance_scorer_ckpt, device)
        print(
            f"[pi0-serve] flow guidance enabled: steps={config.flow_guidance_steps}, "
            f"scale={config.flow_guidance_scale}, scorer={args.flow_guidance_scorer_runtime}, "
            f"mode={args.flow_guidance_score_mode}"
        )
    elif int(config.flow_guidance_steps) > 0 and float(config.flow_guidance_scale) > 0.0:
        print("[pi0-serve] flow guidance requested but no scorer checkpoint was provided; disabled")

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
        action_min = np.zeros(config.robot_action_dim, dtype=np.float32)
        action_max = np.ones(config.robot_action_dim, dtype=np.float32)
        qpos_min = np.zeros(config.robot_action_dim, dtype=np.float32)
        qpos_max = np.ones(config.robot_action_dim, dtype=np.float32)

    action_min = fit_stat_dim(action_min, config.robot_action_dim, 0.0)
    action_max = fit_stat_dim(action_max, config.robot_action_dim, 1.0)
    qpos_min = fit_stat_dim(qpos_min, config.robot_action_dim, 0.0)
    qpos_max = fit_stat_dim(qpos_max, config.robot_action_dim, 1.0)
    action_min_t = torch.tensor(action_min, dtype=torch.float32, device=device).view(1, 1, -1)
    action_max_t = torch.tensor(action_max, dtype=torch.float32, device=device).view(1, 1, -1)

    def denormalize_robot_action(action_norm: torch.Tensor) -> torch.Tensor:
        return (action_norm + 1.0) * 0.5 * (action_max_t - action_min_t) + action_min_t

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
            "model_action_dim": config.action_dim,
            "robot_action_dim": config.robot_action_dim,
            "action_horizon": config.action_horizon,
            "num_flow_steps": args.num_flow_steps,
            "flow_guidance_enabled": flow_guidance_enabled,
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

            with torch.no_grad():
                try:
                    all_actions = None
                    guidance_report = None

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
                        qpos_raw = np.asarray(obs["qpos"], dtype=np.float32)[: config.robot_action_dim]
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
                            flow_guidance = None
                            score_fn = None
                            if flow_guidance_enabled and flow_scorer is not None:
                                flow_guidance = model.build_flow_guidance()
                                score_bridge = Pi0TactileForesightScoreBridge(
                                    model,
                                    marker_offset_norm=marker_tensor,
                                    qpos_norm=state,
                                    scorer=flow_scorer,
                                    action_to_scorer=denormalize_robot_action,
                                    marker_mean=tuple(float(x) for x in marker_mean),
                                    marker_std=tuple(float(x) for x in marker_std),
                                    score_mode=args.flow_guidance_score_mode,
                                    task_id=args.flow_guidance_task_id,
                                    score_window=args.flow_guidance_score_window,
                                )
                                score_fn = score_bridge
                            sample_out = model.sample_actions(
                                images=images_list,
                                img_masks=img_masks_list,
                                lang_tokens=prompt_tokens,
                                lang_masks=prompt_mask,
                                state=state,
                                marker_offset=marker_tensor,
                                num_steps=args.num_flow_steps,
                                flow_guidance=flow_guidance,
                                score_fn=score_fn,
                                return_guidance_report=args.send_guidance_report or flow_guidance_enabled,
                            )
                            if isinstance(sample_out, tuple):
                                all_actions, guidance_report = sample_out
                            else:
                                all_actions = sample_out
                                guidance_report = None

                        # Get current action from chunk
                        action_idx = step % query_freq
                        robot_actions = model.action_adapter.slice_robot_action(all_actions)
                        raw = robot_actions[0, action_idx].cpu().numpy()

                        # Denormalize: [-1, 1] → absolute
                        action = (raw + 1) / 2 * (action_max - action_min) + action_min
                        action = action.astype(np.float32)

                        # Send to robot
                        packet = {
                            "actions": action[None, :],
                            "step": step,
                        }
                        if args.send_guidance_report and guidance_report is not None and action_idx == 0:
                            packet["guidance_report"] = guidance_report
                        server.send_action(packet)

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
