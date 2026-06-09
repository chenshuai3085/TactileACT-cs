"""Calibrate board-wiping target force for real rollout quality gates.

The board task has no human good/bad labels.  For rollout evaluation we need a
stable "suitable force" reference so a weak baseline does not define the
acceptable force band.  This script estimates that reference from the
demonstration/offline board dataset using the same convention as
eval_board_quality_label_schemes.py:

  target_force = q55(force_mean)
  force_sigma  = q75(force_mean) - q25(force_mean)

The output is an explicit calibration artifact used by real rollout gates via:

  --board_target_force <target_force>
  --board_force_sigma <force_sigma>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np


BOARD_DIR = Path("/home/chenshuai/data/dataset/260522_v8l_caheiban")
OUT_DIR = Path("/home/chenshuai/Project/output/board_target_force_calibration")


def safe_get(f: h5py.File, key: str) -> Optional[np.ndarray]:
    if key not in f:
        return None
    return np.asarray(f[key][:], dtype=np.float32)


def force_mag(force6: np.ndarray) -> np.ndarray:
    arr = np.asarray(force6, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None]
    return np.linalg.norm(arr[:, :3], axis=-1)


def discover_hdf5(root: Path) -> List[Path]:
    if root.is_file() and root.suffix in {".hdf5", ".h5"}:
        return [root]
    return sorted([*root.rglob("*.hdf5"), *root.rglob("*.h5")])


def episode_force_row(path: Path, source: str) -> Optional[Dict[str, float]]:
    with h5py.File(path, "r") as f:
        if source == "ft":
            force = safe_get(f, "ft")
        elif source == "left_force":
            force = safe_get(f, "observations/tac/left/force6d")
        elif source == "right_force":
            force = safe_get(f, "observations/tac/right/force6d")
        else:
            raise ValueError(source)
    if force is None or len(force) == 0:
        return None
    mag = force_mag(force)
    delta = np.abs(np.diff(mag)) if len(mag) > 1 else np.zeros(1, dtype=np.float32)
    jerk = np.abs(np.diff(delta)) if len(delta) > 1 else np.zeros(1, dtype=np.float32)
    return {
        "path": str(path),
        "stem": path.stem,
        "n": int(len(mag)),
        "force_mean": float(mag.mean()),
        "force_p50": float(np.percentile(mag, 50)),
        "force_p95": float(np.percentile(mag, 95)),
        "force_max": float(mag.max()),
        "force_delta_mean": float(delta.mean()),
        "force_jerk_mean": float(jerk.mean()),
    }


def summarize(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "q05": float(np.quantile(values, 0.05)),
        "q20": float(np.quantile(values, 0.20)),
        "q25": float(np.quantile(values, 0.25)),
        "q50": float(np.quantile(values, 0.50)),
        "q55": float(np.quantile(values, 0.55)),
        "q75": float(np.quantile(values, 0.75)),
        "q80": float(np.quantile(values, 0.80)),
        "q95": float(np.quantile(values, 0.95)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def calibrate(args: argparse.Namespace) -> Dict:
    paths = discover_hdf5(Path(args.data_dir))
    if args.success_only:
        paths = [p for p in paths if "success" in p.parts]
    rows = [episode_force_row(path, args.force_source) for path in paths]
    rows = [row for row in rows if row is not None]
    if not rows:
        raise RuntimeError(f"No usable force rows under {args.data_dir}")
    force_mean = np.array([row["force_mean"] for row in rows], dtype=np.float64)
    force_p95 = np.array([row["force_p95"] for row in rows], dtype=np.float64)
    force_delta = np.array([row["force_delta_mean"] for row in rows], dtype=np.float64)
    target = float(np.quantile(force_mean, args.target_quantile))
    sigma = float(max(np.quantile(force_mean, 0.75) - np.quantile(force_mean, 0.25), args.min_sigma))
    lower = target - 2.0 * sigma
    upper = target + 2.0 * sigma
    peak_upper = target + 2.5 * sigma
    result = {
        "purpose": "Board target force calibration for real rollout quality gates.",
        "scientific_scope": (
            "Weak calibration from available board dataset; final thresholds should be "
            "reviewed against robot force units and task success."
        ),
        "data_dir": str(Path(args.data_dir)),
        "force_source": args.force_source,
        "success_only": bool(args.success_only),
        "n_episodes": len(rows),
        "target_quantile": args.target_quantile,
        "recommended": {
            "board_target_force": target,
            "board_force_sigma": sigma,
            "acceptable_mean_force_range": [float(lower), float(upper)],
            "acceptable_force_p95_upper": float(peak_upper),
        },
        "summaries": {
            "force_mean": summarize(force_mean),
            "force_p95": summarize(force_p95),
            "force_delta_mean": summarize(force_delta),
        },
        "commands": {
            "two_arm_gate_args": f"--board_target_force {target:.8g} --board_force_sigma {sigma:.8g}",
            "three_arm_gate_args": f"--board_target_force {target:.8g} --board_force_sigma {sigma:.8g}",
        },
        "rows_preview": rows[:20],
    }
    return result


def write_markdown(result: Dict, path: Path) -> None:
    rec = result["recommended"]
    lines = [
        "# Board Target Force Calibration",
        "",
        f"- data_dir: `{result['data_dir']}`",
        f"- force_source: `{result['force_source']}`",
        f"- success_only: `{result['success_only']}`",
        f"- n_episodes: `{result['n_episodes']}`",
        f"- board_target_force: `{rec['board_target_force']:.8g}`",
        f"- board_force_sigma: `{rec['board_force_sigma']:.8g}`",
        f"- acceptable_mean_force_range: `{rec['acceptable_mean_force_range']}`",
        f"- acceptable_force_p95_upper: `{rec['acceptable_force_p95_upper']:.8g}`",
        "",
        "## Gate Args",
        "",
        "```bash",
        result["commands"]["two_arm_gate_args"],
        "```",
        "",
        "## Force Mean Summary",
        "",
        "```json",
        json.dumps(result["summaries"]["force_mean"], ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default=str(BOARD_DIR))
    parser.add_argument("--output_dir", default=str(OUT_DIR))
    parser.add_argument("--force_source", choices=["ft", "left_force", "right_force"], default="left_force")
    parser.add_argument("--target_quantile", type=float, default=0.55)
    parser.add_argument("--min_sigma", type=float, default=1e-6)
    parser.add_argument("--success_only", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = calibrate(args)
    json_path = out_dir / "board_target_force_calibration.json"
    md_path = out_dir / "board_target_force_calibration.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    print(
        json.dumps(
            {
                "n_episodes": result["n_episodes"],
                "board_target_force": result["recommended"]["board_target_force"],
                "board_force_sigma": result["recommended"]["board_force_sigma"],
                "json": str(json_path),
                "markdown": str(md_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
