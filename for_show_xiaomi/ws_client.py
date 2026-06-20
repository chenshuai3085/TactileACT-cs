"""
TactileACT TCP Client for Realman Robot (runs on robot machine)

Usage:
    python ws_client.py --host <gpu_server_ip> --port 8765

Prerequisites on robot machine:
    - realman_env package (from miACT project)
    - Robotic_Arm SDK
    - tcp_client.py (same directory)

The obs dict sent to server must match serve_policy.py expectations:
    {
        "images": {"global": (H,W,3) uint8, "wrist": (H,W,3) uint8},
        "tac":    {"left": {"img": (240,240,3) uint8}},
        "qpos":   (state_dim,) float32,
    }
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from tcp_client import (
    connect_to_server, recv_metadata,
    send_obs, recv_action,
)


class RobotEnv:
    """Wraps miACT's RealmanEnv, adapting obs format for TactileACT server."""

    def __init__(self, action_mode: str = "joint"):
        from realman_env.envs.realman_env import RealmanEnv, Config

        cfg = Config()
        cfg.ACTION_MODE = action_mode
        self.env = RealmanEnv(cfg)
        self._printed_obs_schema = False

    def reset(self) -> dict:
        """Move to home, return first obs."""
        obs = self.env.reset()
        return self._build_obs(obs)

    def step(self, action: np.ndarray) -> dict:
        """Execute action, return new obs."""
        obs, _ = self.env.step(action)
        return self._build_obs(obs)

    def _build_obs(self, raw_obs: dict) -> dict:
        """Convert RealmanEnv obs to TactileACT server format.

        RealmanEnv produces:
            raw_obs["images"][cam]          -> (H,W,3) uint8
            raw_obs["proprio"]              -> (7,) float32 (joint deg)
            raw_obs["eef"]                  -> (6,) or (7,) float32 (xyz + euler/quat)
            raw_obs["tactile"][side]["img"]  -> (240,240,3) uint8

        Server expects:
            obs["images"][cam]    -> (H,W,3) uint8
            obs["qpos"]          -> (state_dim,) float32
            obs["eef"]           -> (6,) or (7,) float32 (xyz + orientation)
            obs["tac"][side]["img"]           -> (240,240,3) uint8
            obs["tac"][side]["marker_offset"] -> (9,9,2) float32  (marker mode)
            obs["tac"][side]["force6d"]       -> (6,) float32      (if available)
        """
        obs = {
            "images": raw_obs.get("images", {}),
            "qpos": np.asarray(raw_obs["proprio"], dtype=np.float32),
        }
        if "ft" in raw_obs:
            obs["ft"] = np.asarray(raw_obs["ft"], dtype=np.float32)

        eef_source = None
        if "eef_pose" in raw_obs:
            obs["eef"] = np.asarray(raw_obs["eef_pose"], dtype=np.float32)
            eef_source = "raw_obs.eef_pose"
        elif "eef" in raw_obs:
            obs["eef"] = np.asarray(raw_obs["eef"], dtype=np.float32)
            eef_source = "raw_obs.eef"
        else:
            current_pose = getattr(self.env, "current_pose", None)
            if current_pose is not None:
                obs["eef"] = np.asarray(current_pose, dtype=np.float32)
                eef_source = "env.current_pose"

        # tactile: rename "tactile" -> "tac", send img/marker/force fields
        tactile = raw_obs.get("tactile")
        if tactile is not None:
            tac = {}
            for side, side_data in tactile.items():
                if not isinstance(side_data, dict) or "img" not in side_data:
                    raise KeyError(f"tactile[{side}] missing 'img' field")
                entry = {"img": side_data["img"]}
                if "marker_offset" in side_data:
                    entry["marker_offset"] = np.asarray(
                        side_data["marker_offset"], dtype=np.float32)
                if "force6d" in side_data:
                    entry["force6d"] = np.asarray(side_data["force6d"], dtype=np.float32)
                tac[side] = entry
            if tac:
                obs["tac"] = tac

        if not self._printed_obs_schema:
            self._printed_obs_schema = True
            image_keys = list(raw_obs.get("images", {}).keys())
            tactile = raw_obs.get("tactile")
            tactile_keys = list(tactile.keys()) if isinstance(tactile, dict) else []
            tactile_field_keys = {}
            if isinstance(tactile, dict):
                for side, side_data in tactile.items():
                    if isinstance(side_data, dict):
                        tactile_field_keys[side] = list(side_data.keys())
            logging.info("[client] raw_obs keys: %s", list(raw_obs.keys()))
            logging.info("[client] raw image keys: %s", image_keys)
            logging.info("[client] raw tactile sides: %s", tactile_keys)
            logging.info("[client] raw tactile fields: %s", tactile_field_keys)
            logging.info("[client] eef_pose present: %s, eef present: %s",
                         "eef_pose" in raw_obs, "eef" in raw_obs)
            logging.info("[client] outgoing eef source: %s", eef_source)
            logging.info("[client] outgoing obs keys: %s", list(obs.keys()))
            if "eef" in obs:
                logging.info("[client] outgoing eef shape: %s, first values: %s",
                             obs["eef"].shape, obs["eef"][:3])
            if "tac" in obs:
                outgoing_tac_fields = {
                    side: list(side_data.keys())
                    for side, side_data in obs["tac"].items()
                }
                logging.info("[client] outgoing tac fields: %s", outgoing_tac_fields)

        return obs


import select
import sys
import termios
import tty


class ForceEpisodeLogger:
    """Record real force traces for one robot rollout/trial."""

    def __init__(
        self,
        root_dir: str,
        *,
        trial: int,
        host: str,
        port: int,
        control_hz: float,
        action_mode: str,
        server_metadata: dict | None,
    ):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.root = Path(root_dir).expanduser()
        self.trial_dir = self.root / f"{ts}_port{port}_trial{trial:04d}"
        self.trial_dir.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()
        self.records = []
        self.actions = []
        self.metadata = {
            "created_at": ts,
            "trial": int(trial),
            "host": str(host),
            "port": int(port),
            "control_hz": float(control_hz),
            "action_mode": str(action_mode),
            "server_metadata": server_metadata or {},
            "schema": {
                "ft": "robot wrist force/torque if present; first 3 dims are Fx,Fy,Fz",
                "left_force6d": "left tactile force6d if present",
                "right_force6d": "right tactile force6d if present",
            },
        }
        if server_metadata and isinstance(server_metadata.get("client_rollout_metadata"), dict):
            self.metadata.update(server_metadata["client_rollout_metadata"])

    @staticmethod
    def _vec(obs: dict, key: str, n: int = 6):
        value = obs.get(key)
        if value is None:
            return [float("nan")] * n
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        out = [float(x) for x in arr[:n]]
        if len(out) < n:
            out.extend([float("nan")] * (n - len(out)))
        return out

    @staticmethod
    def _tac_force(obs: dict, side: str, n: int = 6):
        tac = obs.get("tac", {})
        side_data = tac.get(side, {}) if isinstance(tac, dict) else {}
        value = side_data.get("force6d") if isinstance(side_data, dict) else None
        if value is None:
            return [float("nan")] * n
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        out = [float(x) for x in arr[:n]]
        if len(out) < n:
            out.extend([float("nan")] * (n - len(out)))
        return out

    @staticmethod
    def _mag3(values):
        arr = np.asarray(values[:3], dtype=np.float64)
        if not np.isfinite(arr).all():
            return float("nan")
        return float(np.linalg.norm(arr))

    def record(self, step: int, obs: dict, action: np.ndarray | None) -> None:
        ft = self._vec(obs, "ft")
        left = self._tac_force(obs, "left")
        right = self._tac_force(obs, "right")
        now = time.time()
        row = {
            "step": int(step),
            "wall_time": float(now),
            "t": float(now - self.t0),
            "ft_fx": ft[0],
            "ft_fy": ft[1],
            "ft_fz": ft[2],
            "ft_tx": ft[3],
            "ft_ty": ft[4],
            "ft_tz": ft[5],
            "ft_f_mag": self._mag3(ft),
            "left_fx": left[0],
            "left_fy": left[1],
            "left_fz": left[2],
            "left_tx": left[3],
            "left_ty": left[4],
            "left_tz": left[5],
            "left_f_mag": self._mag3(left),
            "right_fx": right[0],
            "right_fy": right[1],
            "right_fz": right[2],
            "right_tx": right[3],
            "right_ty": right[4],
            "right_tz": right[5],
            "right_f_mag": self._mag3(right),
        }
        if action is not None:
            action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
            self.actions.append(action_arr)
            for i, value in enumerate(action_arr):
                row[f"action_{i}"] = float(value)
        self.records.append(row)

    @staticmethod
    def _finite_array(records, key):
        vals = np.array([r.get(key, np.nan) for r in records], dtype=np.float64)
        return vals[np.isfinite(vals)]

    def _summarize(self) -> dict:
        summary = {"n_samples": int(len(self.records))}
        for key in ("ft_fz", "ft_f_mag", "left_fz", "left_f_mag", "right_fz", "right_f_mag"):
            vals = self._finite_array(self.records, key)
            if len(vals):
                d = np.diff(vals) if len(vals) > 1 else np.array([], dtype=np.float64)
                summary[key] = {
                    "mean": float(vals.mean()),
                    "std": float(vals.std()),
                    "min": float(vals.min()),
                    "max": float(vals.max()),
                    "p05": float(np.percentile(vals, 5)),
                    "p50": float(np.percentile(vals, 50)),
                    "p95": float(np.percentile(vals, 95)),
                    "delta_abs_mean": float(np.abs(d).mean()) if len(d) else 0.0,
                    "delta_abs_p95": float(np.percentile(np.abs(d), 95)) if len(d) else 0.0,
                }
        return summary

    def _write_csv(self) -> Path:
        csv_path = self.trial_dir / "force_trace.csv"
        keys = sorted({k for row in self.records for k in row.keys()})
        preferred = [
            "step", "t", "wall_time",
            "ft_fx", "ft_fy", "ft_fz", "ft_f_mag",
            "left_fx", "left_fy", "left_fz", "left_f_mag",
            "right_fx", "right_fy", "right_fz", "right_f_mag",
        ]
        fieldnames = preferred + [k for k in keys if k not in preferred]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.records)
        return csv_path

    def _write_npz(self) -> Path:
        npz_path = self.trial_dir / "force_trace.npz"
        arrays = {}
        if self.records:
            keys = sorted({k for row in self.records for k in row.keys()})
            for key in keys:
                arrays[key] = np.array([row.get(key, np.nan) for row in self.records], dtype=np.float32)
        if self.actions:
            arrays["actions"] = np.stack(self.actions).astype(np.float32)
        np.savez_compressed(npz_path, **arrays)
        return npz_path

    def _write_plot(self) -> str | None:
        if not self.records:
            return None
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            logging.warning("[force-log] matplotlib unavailable, skip plot: %s", exc)
            return None

        t = np.array([r["t"] for r in self.records], dtype=np.float64)
        plot_path = self.trial_dir / "force_curve.png"
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        for key, label in [
            ("ft_fz", "robot ft Fz"),
            ("left_fz", "left tactile Fz"),
            ("right_fz", "right tactile Fz"),
        ]:
            y = np.array([r.get(key, np.nan) for r in self.records], dtype=np.float64)
            if np.isfinite(y).any():
                axes[0].plot(t, y, label=label, linewidth=1.5)
        axes[0].set_ylabel("Fz")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="best")

        for key, label in [
            ("ft_f_mag", "robot |Fxyz|"),
            ("left_f_mag", "left tactile |Fxyz|"),
            ("right_f_mag", "right tactile |Fxyz|"),
        ]:
            y = np.array([r.get(key, np.nan) for r in self.records], dtype=np.float64)
            if np.isfinite(y).any():
                axes[1].plot(t, y, label=label, linewidth=1.5)
        axes[1].set_xlabel("time (s)")
        axes[1].set_ylabel("force magnitude")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="best")
        fig.suptitle(self.trial_dir.name)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
        return str(plot_path)

    def finalize(self, *, steps: int, stop_reason: str) -> dict:
        self.metadata["steps"] = int(steps)
        self.metadata["stop_reason"] = str(stop_reason)
        self.metadata["summary"] = self._summarize()
        csv_path = self._write_csv()
        npz_path = self._write_npz()
        png_path = self._write_plot()
        self.metadata["artifacts"] = {
            "csv": str(csv_path),
            "npz": str(npz_path),
            "plot": png_path,
        }
        meta_path = self.trial_dir / "metadata.json"
        meta_path.write_text(json.dumps(self.metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        logging.info("[force-log] saved trial force data: %s", self.trial_dir)
        return self.metadata


def _check_key():
    """非阻塞检测键盘输入，返回按键字符或None。"""
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1)
    return None


def _run_one_episode(
    env,
    host,
    port,
    max_steps,
    dt,
    *,
    trial,
    control_hz,
    action_mode,
    force_log_dir,
    disable_force_log,
    rollout_metadata,
):
    """执行一次完整episode，空格键可中途停止。返回实际步数。"""
    sock = connect_to_server(host, port)
    logger = None
    executed_steps = 0
    stop_reason = "max_steps"
    prev_action = None
    try:
        metadata = recv_metadata(sock)
        logging.info("[client] server metadata: %s", metadata)
        if rollout_metadata:
            metadata = dict(metadata)
            metadata["client_rollout_metadata"] = dict(rollout_metadata)

        obs = env.reset()
        if rollout_metadata:
            obs["rollout_metadata"] = dict(rollout_metadata)
        if not disable_force_log:
            logger = ForceEpisodeLogger(
                force_log_dir,
                trial=trial,
                host=host,
                port=port,
                control_hz=control_hz,
                action_mode=action_mode,
                server_metadata=metadata,
            )
            logger.record(0, obs, None)

        for step in range(max_steps):
            t0 = time.perf_counter()

            # 检测空格键 → 停止当前episode
            key = _check_key()
            if key == ' ':
                print(f"\n[client] STOPPED by user at step {step + 1}")
                stop_reason = "user_space"
                break

            if rollout_metadata:
                obs["rollout_metadata"] = dict(rollout_metadata)
            send_obs(sock, obs)
            msg = recv_action(sock)

            action = np.asarray(msg["actions"], dtype=np.float32)
            if action.ndim >= 2:
                action = action[0]

            obs = env.step(action)
            if rollout_metadata:
                obs["rollout_metadata"] = dict(rollout_metadata)
            prev_action = action
            executed_steps += 1
            if logger is not None:
                logger.record(executed_steps, obs, prev_action)

            logging.info("[client] step %d/%d", step + 1, max_steps)

            elapsed = time.perf_counter() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)
    finally:
        sock.close()
        if logger is not None:
            logger.finalize(steps=executed_steps, stop_reason=stop_reason)

    return executed_steps


def main():
    parser = argparse.ArgumentParser(
        description="TactileACT Realman client (runs on robot machine)",
    )
    parser.add_argument("--host", type=str, required=True,
                        help="GPU server IP address")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--control_hz", type=float, default=20.0)
    parser.add_argument("--action_mode", type=str, default="joint",
                        choices=["joint", "eef_rel"])
    parser.add_argument(
        "--force_log_dir",
        type=str,
        default="/home/chenshuai/Project/output/board_force_rollouts",
        help="Directory for per-trial force CSV/NPZ/PNG logs during real robot tests.",
    )
    parser.add_argument(
        "--disable_force_log",
        action="store_true",
        help="Disable per-trial force logging.",
    )
    parser.add_argument("--rollout_pair_id", default=None,
                        help="Optional manifest pair_id forwarded to the server-side rollout metadata.")
    parser.add_argument("--rollout_trial_order", type=int, default=None,
                        help="Optional manifest trial_order forwarded to the server-side rollout metadata.")
    parser.add_argument("--rollout_task", default=None,
                        help="Optional manifest task forwarded to the server-side rollout metadata.")
    parser.add_argument("--rollout_group", choices=["baseline", "guided"], default=None,
                        help="Optional manifest group forwarded to the server-side rollout metadata.")
    parser.add_argument("--rollout_server_arm", default=None,
                        help="Optional expected server arm forwarded to the server-side rollout metadata.")
    parser.add_argument("--rollout_manifest_csv", default=None,
                        help="Optional manifest CSV path forwarded to the server-side rollout metadata.")
    args = parser.parse_args()

    rollout_metadata = {
        key: value
        for key, value in {
            "pair_id": args.rollout_pair_id,
            "manifest_trial_order": args.rollout_trial_order,
            "manifest_task": args.rollout_task,
            "manifest_group": args.rollout_group,
            "manifest_server_arm": args.rollout_server_arm,
            "manifest_source_csv": args.rollout_manifest_csv,
            "pair_id_source": "ws_client_rollout_metadata" if args.rollout_pair_id else None,
        }.items()
        if value is not None
    }

    env = RobotEnv(action_mode=args.action_mode)
    dt = 1.0 / args.control_hz

    # 设置终端为raw模式以支持非阻塞按键检测
    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    trial = 0
    try:
        while True:
            trial += 1
            print(f"\n{'='*50}")
            print(f"  Trial #{trial} — Press Enter to START, Ctrl+C to QUIT")
            print(f"  (During execution: Space to STOP current trial)")
            print(f"{'='*50}")

            # 等待Enter开始
            while True:
                key = sys.stdin.read(1)
                if key == '\n' or key == '\r':
                    break

            print(f"[client] Starting trial #{trial}...")
            steps = _run_one_episode(env, args.host, args.port,
                                     args.max_steps, dt,
                                     trial=trial,
                                     control_hz=args.control_hz,
                                     action_mode=args.action_mode,
                                     force_log_dir=args.force_log_dir,
                                     disable_force_log=args.disable_force_log,
                                     rollout_metadata=rollout_metadata)
            print(f"[client] Trial #{trial} finished: {steps} steps")

    except KeyboardInterrupt:
        print("\n[client] Exiting.")
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
