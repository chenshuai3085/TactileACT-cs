#!/usr/bin/env python3
"""
Generate card-swipe positive EEF trajectories from real success episodes.

This script keeps each source success trajectory's step count and route skeleton.
It is intended for card swiping, where the slot/entry geometry matters and the
trajectory should stay very close to demonstrations that already completed the task.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


DEFAULT_SUCCESS_DIR = Path("/media/chenshuai/EXTERNAL_USB/pih_dataset/260615_v8l_card/success")
CONTROL_HZ = 20


def _episode_index(path: Path) -> int:
    try:
        return int(path.stem.split("_")[-1])
    except ValueError:
        return 0


def _wrap_to_pi(angles: np.ndarray) -> np.ndarray:
    return (angles + np.pi) % (2 * np.pi) - np.pi


def _parse_vector(text: str, expected_len: int, name: str) -> np.ndarray:
    try:
        values = [float(x.strip()) for x in text.split(",")]
    except ValueError as exc:
        raise ValueError(f"{name} should be comma separated floats, got: {text}") from exc
    if len(values) != expected_len:
        raise ValueError(f"{name} should have {expected_len} values, got: {text}")
    return np.asarray(values, dtype=np.float64)


def _smooth_noise(length: int, dim: int, knot_interval: int, scale: np.ndarray, rng) -> np.ndarray:
    knots = np.arange(0, length + knot_interval, knot_interval)
    values = rng.normal(0.0, scale, size=(len(knots), dim))
    t = np.arange(length)
    noise = np.stack([np.interp(t, knots, values[:, i]) for i in range(dim)], axis=1)
    envelope = np.sin(np.linspace(0.0, np.pi, length))[:, None]
    return noise * envelope


def load_success_trajectory(path: Path) -> np.ndarray:
    import h5py

    with h5py.File(str(path), "r") as f:
        if "actions/eef_abs" in f:
            traj = f["actions/eef_abs"][:]
        elif "observations/proprio_eef" in f:
            traj = f["observations/proprio_eef"][:]
        else:
            raise KeyError(f"{path} lacks actions/eef_abs and observations/proprio_eef")
    traj = np.asarray(traj, dtype=np.float64)
    if traj.ndim != 2 or traj.shape[1] != 6:
        raise ValueError(f"Unexpected trajectory shape in {path}: {traj.shape}")
    return traj


def perturb_keep_steps(
    source: np.ndarray,
    rng,
    xyz_noise_mm: np.ndarray,
    euler_noise_rad: np.ndarray,
    knot_interval: int,
    lock_start_steps: int,
    lock_end_steps: int,
    no_noise: bool,
) -> np.ndarray:
    traj = source.copy()
    if no_noise:
        return traj.astype(np.float32)

    xyz_scale = xyz_noise_mm / 1000.0
    traj[:, :3] += _smooth_noise(len(traj), 3, knot_interval, xyz_scale, rng)

    euler_cont = np.unwrap(traj[:, 3:], axis=0)
    euler_cont += _smooth_noise(len(traj), 3, knot_interval, euler_noise_rad, rng)
    traj[:, 3:] = _wrap_to_pi(euler_cont)

    start_n = min(max(int(lock_start_steps), 0), len(traj))
    end_n = min(max(int(lock_end_steps), 0), len(traj))
    if start_n:
        traj[:start_n] = source[:start_n]
    if end_n:
        traj[-end_n:] = source[-end_n:]
    return traj.astype(np.float32)


def trajectory_stats(traj: np.ndarray, source: np.ndarray) -> dict:
    speed = np.linalg.norm(np.diff(traj[:, :3], axis=0), axis=1) * CONTROL_HZ * 1000.0
    xyz_err = np.linalg.norm(traj[:, :3] - source[:, :3], axis=1) * 1000.0
    return {
        "steps": int(len(traj)),
        "duration_sec": float(len(traj) / CONTROL_HZ),
        "xyz_min_mm": (traj[:, :3].min(axis=0) * 1000.0).round(3).tolist(),
        "xyz_max_mm": (traj[:, :3].max(axis=0) * 1000.0).round(3).tolist(),
        "speed_mean_mm_s": float(speed.mean()) if len(speed) else 0.0,
        "speed_max_mm_s": float(speed.max()) if len(speed) else 0.0,
        "xyz_err_mean_mm": float(xyz_err.mean()),
        "xyz_err_p95_mm": float(np.quantile(xyz_err, 0.95)),
        "xyz_err_max_mm": float(xyz_err.max()),
    }


def visualize(traj: np.ndarray, source: np.ndarray, save_path: Path) -> None:
    import matplotlib.pyplot as plt

    t = np.arange(len(traj)) / CONTROL_HZ
    speed = np.linalg.norm(np.diff(traj[:, :3], axis=0), axis=1) * CONTROL_HZ * 1000.0

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(save_path.stem)

    ax = axes[0, 0]
    ax.plot(source[:, 0], source[:, 1], color="0.7", linewidth=1.0, label="source")
    ax.plot(traj[:, 0], traj[:, 1], "b-", linewidth=0.8, label="generated")
    ax.plot(traj[0, 0], traj[0, 1], "go", markersize=6)
    ax.plot(traj[-1, 0], traj[-1, 1], "r^", markersize=6)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("XY")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t, source[:, 2] * 1000.0, color="0.7", linewidth=1.0, label="source")
    ax.plot(t, traj[:, 2] * 1000.0, "r-", linewidth=0.8, label="generated")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Z (mm)")
    ax.set_title("Z vs Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(source[:, 0], source[:, 2] * 1000.0, color="0.7", linewidth=1.0)
    ax.plot(traj[:, 0], traj[:, 2] * 1000.0, "g-", linewidth=0.8)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (mm)")
    ax.set_title("XZ")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(t[1:], speed, "purple", linewidth=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (mm/s)")
    ax.set_title(f"Speed mean={speed.mean():.1f}, max={speed.max():.1f} mm/s")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate card-swipe positive trajectories preserving success steps/routes.")
    parser.add_argument("--success_dir", type=Path, default=DEFAULT_SUCCESS_DIR)
    parser.add_argument("--save_dir", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=100)
    parser.add_argument("--seed", type=int, default=260626)
    parser.add_argument("--xyz_noise_mm", type=str, default="0.20,0.20,0.05")
    parser.add_argument("--euler_noise_rad", type=str, default="0.0010,0.0005,0.0010")
    parser.add_argument("--knot_interval", type=int, default=45)
    parser.add_argument("--lock_start_steps", type=int, default=20)
    parser.add_argument("--lock_end_steps", type=int, default=10)
    parser.add_argument("--no_noise", action="store_true", help="Copy source trajectories exactly.")
    parser.add_argument("--visualize", action="store_true", help="Save previews for the first three trajectories.")
    args = parser.parse_args()

    success_files = sorted(args.success_dir.glob("episode_*.hdf5"), key=_episode_index)
    if not success_files:
        raise FileNotFoundError(f"No episode_*.hdf5 found in {args.success_dir}")

    xyz_noise_mm = _parse_vector(args.xyz_noise_mm, 3, "--xyz_noise_mm")
    euler_noise_rad = _parse_vector(args.euler_noise_rad, 3, "--euler_noise_rad")

    args.save_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    all_stats = []

    for out_idx in range(args.batch):
        rng = np.random.default_rng(args.seed + out_idx)
        source_idx = out_idx % len(success_files)
        source_path = success_files[source_idx]
        source = load_success_trajectory(source_path)
        traj = perturb_keep_steps(
            source=source,
            rng=rng,
            xyz_noise_mm=xyz_noise_mm,
            euler_noise_rad=euler_noise_rad,
            knot_interval=args.knot_interval,
            lock_start_steps=args.lock_start_steps,
            lock_end_steps=args.lock_end_steps,
            no_noise=args.no_noise,
        )

        out_name = f"card_pos_{out_idx:03d}.npy"
        out_path = args.save_dir / out_name
        np.save(out_path, traj)
        if args.visualize and out_idx < 3:
            visualize(traj, source, args.save_dir / f"card_pos_{out_idx:03d}.png")

        stats = trajectory_stats(traj.astype(np.float64), source)
        all_stats.append(stats)
        row = {
            "output": out_name,
            "source_idx": source_idx,
            "source_file": source_path.name,
            **stats,
        }
        rows.append(row)
        print(
            f"[+] {out_name}: source={source_path.name}, steps={stats['steps']}, "
            f"max_speed={stats['speed_max_mm_s']:.1f}mm/s, "
            f"xyz_err_p95={stats['xyz_err_p95_mm']:.3f}mm"
        )

    csv_path = args.save_dir / "manifest.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "source_dir": str(args.success_dir),
        "output_dir": str(args.save_dir),
        "count": len(rows),
        "source_count": len(success_files),
        "generation": "success_keep_steps_same_route",
        "no_time_scaling": True,
        "xyz_noise_mm": xyz_noise_mm.tolist(),
        "euler_noise_rad": euler_noise_rad.tolist(),
        "lock_start_steps": args.lock_start_steps,
        "lock_end_steps": args.lock_end_steps,
        "steps": {
            "mean": float(np.mean([s["steps"] for s in all_stats])),
            "min": int(np.min([s["steps"] for s in all_stats])),
            "max": int(np.max([s["steps"] for s in all_stats])),
        },
        "speed_max_mm_s": {
            "mean": float(np.mean([s["speed_max_mm_s"] for s in all_stats])),
            "max": float(np.max([s["speed_max_mm_s"] for s in all_stats])),
        },
        "xyz_err_mm": {
            "mean": float(np.mean([s["xyz_err_mean_mm"] for s in all_stats])),
            "p95_mean": float(np.mean([s["xyz_err_p95_mm"] for s in all_stats])),
            "max": float(np.max([s["xyz_err_max_mm"] for s in all_stats])),
        },
    }
    (args.save_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    readme = f"""# Card Positive Keep-Steps Trajectories

Generated from `{args.success_dir}`.

These trajectories preserve each source success trajectory's original step count and route skeleton.
There is no time scaling or lengthening. Output `i` uses source `i % {len(success_files)}`.

- Count: {summary['count']}
- Source count: {summary['source_count']}
- Step range: {summary['steps']['min']}..{summary['steps']['max']}
- Max speed: mean {summary['speed_max_mm_s']['mean']:.3f} mm/s, max {summary['speed_max_mm_s']['max']:.3f} mm/s
- XYZ deviation from source: mean {summary['xyz_err_mm']['mean']:.3f} mm, max {summary['xyz_err_mm']['max']:.3f} mm

Command:

```bash
/home/chenshuai/miniconda3/envs/TactileACT/bin/python data_collection/generate_card_success_keepsteps.py --save_dir {args.save_dir} --batch {args.batch} --seed {args.seed} --visualize
```
"""
    (args.save_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"\nSaved {len(rows)} trajectories to {args.save_dir}")


if __name__ == "__main__":
    main()
