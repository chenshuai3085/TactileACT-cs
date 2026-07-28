"""WebSocket server for pi0/pi0.5 tactile foresight policies."""
from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any

import numpy as np
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from for_show_xiaomi.ws_server import ClientDisconnected, TactileACTServer
from pi0_tactile.runtime import Pi0TactileRuntime


def config_overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "robot_action_dim": args.robot_action_dim,
        "action_dim": args.action_dim,
        "pi05": True if args.pi05 else None,
        "action_horizon": args.action_horizon,
        "num_flow_steps": args.num_flow_steps,
        "tokenizer_backend": args.tokenizer_backend,
        "flow_guidance_steps": args.flow_guidance_steps,
        "flow_guidance_scale": args.flow_guidance_scale,
        "flow_guidance_max_total_delta": args.flow_guidance_max_total_delta,
        "flow_guidance_lambda_smooth": args.flow_guidance_lambda_smooth,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve pi0/pi0.5 tactile foresight policy")
    parser.add_argument("--ckpt_dir", type=str, required=True,
                        help="Directory containing checkpoint.pth and config.json")
    parser.add_argument("--pi0_weights", type=str, default="")
    parser.add_argument("--stats_path", type=str, default="",
                        help="Optional dataset_stats.pkl override")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max_timesteps", type=int, default=300)
    parser.add_argument("--action_horizon_exec", type=int, default=8,
                        help="How many predicted actions to execute before replanning")

    parser.add_argument("--action_dim", type=int, default=None,
                        help="Full pi0/pi0.5 model action dim, e.g. 32 for pi0.5")
    parser.add_argument("--pi05", action="store_true",
                        help="Force pi0.5 mode when config.json is absent or generic")
    parser.add_argument("--robot_action_dim", type=int, default=None,
                        help="Executable robot/scorer action dim, e.g. 7")
    parser.add_argument("--action_horizon", type=int, default=None)
    parser.add_argument("--num_flow_steps", type=int, default=None)
    parser.add_argument("--tokenizer_backend", type=str, default=None,
                        choices=["ascii", "paligemma", "official", "openpi"])

    parser.add_argument("--flow_guidance_steps", type=int, default=None)
    parser.add_argument("--flow_guidance_scale", type=float, default=None)
    parser.add_argument("--flow_guidance_max_total_delta", type=float, default=None)
    parser.add_argument("--flow_guidance_lambda_smooth", type=float, default=None)
    parser.add_argument("--flow_guidance_scorer_ckpt", type=str, default="")
    parser.add_argument("--flow_guidance_scorer_runtime", choices=["force_band", "distilled"], default="force_band")
    parser.add_argument("--flow_guidance_score_mode", type=str, default="energy_clipped")
    parser.add_argument("--flow_guidance_score_window", type=int, default=8)
    parser.add_argument("--flow_guidance_task_id", type=int, default=None)
    parser.add_argument("--send_guidance_report", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[pi0-serve] device={device}")

    runtime = Pi0TactileRuntime.from_checkpoint_dir(
        args.ckpt_dir,
        device=device,
        pi0_weights=args.pi0_weights,
        stats_path=args.stats_path,
        config_overrides=config_overrides_from_args(args),
        flow_guidance_scorer_ckpt=args.flow_guidance_scorer_ckpt,
        flow_guidance_scorer_runtime=args.flow_guidance_scorer_runtime,
        flow_guidance_score_mode=args.flow_guidance_score_mode,
        flow_guidance_score_window=args.flow_guidance_score_window,
        flow_guidance_task_id=args.flow_guidance_task_id,
    )
    config = runtime.config
    exec_horizon = min(int(args.action_horizon_exec), int(config.action_horizon))
    print(
        "[pi0-serve] loaded "
        f"pi05={config.pi05} action_dim={config.action_dim} "
        f"robot_action_dim={config.robot_action_dim} horizon={config.action_horizon} "
        f"exec_horizon={exec_horizon} guidance={runtime.guidance_enabled}"
    )
    if runtime.guidance_requested and not runtime.guidance_enabled:
        print("[pi0-serve] flow guidance requested but scorer checkpoint was not provided; disabled")

    server = TactileACTServer(
        host=args.host,
        port=args.port,
        metadata={
            "protocol": "pi0_tactile",
            "pi05": config.pi05,
            "model_action_dim": config.action_dim,
            "robot_action_dim": config.robot_action_dim,
            "action_horizon": config.action_horizon,
            "num_flow_steps": config.num_flow_steps,
            "flow_guidance_enabled": runtime.guidance_enabled,
        },
    )
    server.start()
    print(f"[pi0-serve] listening on {args.host}:{args.port}")

    try:
        episode = 0
        while True:
            try:
                print(f"\n[pi0-serve] === episode {episode} ===")
                obs = server.recv_obs()
            except ClientDisconnected:
                continue

            runtime.reset()
            chunk = None
            guidance_report = None
            try:
                for step in range(args.max_timesteps):
                    if step % exec_horizon == 0 or chunk is None:
                        out = runtime.sample_action_chunk(
                            obs,
                            num_flow_steps=config.num_flow_steps,
                            return_guidance_report=args.send_guidance_report,
                        )
                        chunk = out["actions_robot_raw"].detach().cpu().numpy()[0]
                        guidance_report = out["guidance_report"]

                    action_idx = step % exec_horizon
                    action = chunk[action_idx].astype(np.float32)
                    packet = {"actions": action[None, :], "step": step}
                    if args.send_guidance_report and guidance_report is not None and action_idx == 0:
                        packet["guidance_report"] = guidance_report
                    server.send_action(packet)

                    if step + 1 < args.max_timesteps:
                        obs = server.recv_obs()
            except ClientDisconnected:
                print(f"[pi0-serve] client disconnected during episode {episode}")
            episode += 1
    except KeyboardInterrupt:
        print("\n[pi0-serve] shutting down")
    finally:
        server.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
