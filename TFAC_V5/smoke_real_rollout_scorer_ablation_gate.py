"""Smoke-test the three-arm real rollout scorer ablation gate.

This creates synthetic but schema-compatible HDF5 rollouts for both insertion
and board tasks, then runs eval_real_rollout_scorer_ablation_gate.py in process.

The smoke data are not scientific evidence.  They only verify that the formal
three-arm evaluator can read HDF5 rollouts, apply pairing/metadata, compute
quality deltas, and select the expected best arm.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Tuple

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TFAC_V5.eval_real_rollout_scorer_ablation_gate import build_result, write_markdown
from TFAC_V5.eval_real_rollout_quality_gate import write_csv


OUT_DIR = Path("/home/chenshuai/Project/output/real_rollout_scorer_ablation_smoke")
ARMS = ["baseline", "default_guided", "distilled_guided"]


def mkdir_clean(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def smooth_signal(rng: np.random.Generator, steps: int, dim: int, scale: float, roughness: float) -> np.ndarray:
    base = np.linspace(0.0, 1.0, steps, dtype=np.float32)[:, None] * rng.normal(0.0, scale, size=(1, dim))
    walk = np.cumsum(rng.normal(0.0, roughness, size=(steps, dim)), axis=0)
    jitter = rng.normal(0.0, roughness * 0.35, size=(steps, dim))
    return (base + walk + jitter).astype(np.float32)


def write_rollout(path: Path, *, task: str, quality_rank: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    steps = 40
    path.parent.mkdir(parents=True, exist_ok=True)
    # quality_rank: 0=worst baseline, 1=default, 2=distilled best.
    if task == "board":
        force_level = [0.18, 0.55, 0.58][quality_rank]
        force_noise = [0.20, 0.055, 0.025][quality_rank]
        action_rough = [0.055, 0.010, 0.002][quality_rank]
        marker_scale = [0.18, 0.08, 0.035][quality_rank]
    else:
        force_level = [0.75, 0.42, 0.28][quality_rank]
        force_noise = [0.22, 0.08, 0.035][quality_rank]
        action_rough = [0.055, 0.014, 0.004][quality_rank]
        marker_scale = [0.28, 0.10, 0.045][quality_rank]

    force_xyz = rng.normal(force_level, force_noise, size=(steps, 3)).astype(np.float32)
    force_rot = rng.normal(0.0, 0.01, size=(steps, 3)).astype(np.float32)
    force6 = np.concatenate([force_xyz, force_rot], axis=-1)
    left_marker = smooth_signal(rng, steps, 9 * 9 * 2, marker_scale, marker_scale * 0.08).reshape(steps, 9, 9, 2)
    right_marker = smooth_signal(rng, steps, 9 * 9 * 2, marker_scale * 0.9, marker_scale * 0.07).reshape(steps, 9, 9, 2)
    joint = smooth_signal(rng, steps, 7, 0.08, action_rough)
    eef = smooth_signal(rng, steps, 6, 0.05, action_rough * 0.8)

    with h5py.File(path, "w") as f:
        f.create_dataset("ft", data=force6)
        f.create_dataset("observations/tac/left/force6d", data=force6)
        f.create_dataset("observations/tac/right/force6d", data=force6 * 0.95)
        f.create_dataset("observations/tac/left/marker_offset", data=left_marker.astype(np.float32))
        f.create_dataset("observations/tac/right/marker_offset", data=right_marker.astype(np.float32))
        f.create_dataset("actions/joint_abs", data=joint.astype(np.float32))
        f.create_dataset("actions/eef_abs", data=eef.astype(np.float32))
        f.attrs["success"] = 1.0
        f.attrs["stopped_early"] = 0.0


def write_pairing_and_metadata(root: Path, n_pairs: int) -> Tuple[Path, Path]:
    pairing = root / "three_arm_pairing.csv"
    metadata = root / "metadata.csv"
    with open(pairing, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pair_id", "baseline", "default_guided", "distilled_guided"])
        for i in range(n_pairs):
            name = f"episode_{i + 1:03d}.hdf5"
            writer.writerow([f"trial_{i + 1:03d}", name, name, name])
    with open(metadata, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["stem", "success", "stopped_early"])
        for i in range(n_pairs):
            writer.writerow([f"episode_{i + 1:03d}", "1", "0"])
    return pairing, metadata


def make_task_data(root: Path, task: str, n_pairs: int, seed: int) -> Dict[str, Path]:
    task_root = root / task
    mkdir_clean(task_root)
    dirs = {arm: task_root / arm for arm in ARMS}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    for i in range(n_pairs):
        for rank, arm in enumerate(ARMS):
            write_rollout(
                dirs[arm] / f"episode_{i + 1:03d}.hdf5",
                task=task,
                quality_rank=rank,
                seed=seed + i * 17 + rank,
            )
    pairing, metadata = write_pairing_and_metadata(task_root, n_pairs)
    return {**dirs, "pairing": pairing, "metadata": metadata}


def run_gate(task: str, paths: Dict[str, Path], out_dir: Path, n_pairs: int, seed: int) -> Dict:
    args = SimpleNamespace(
        task=task,
        baseline_dir=str(paths["baseline"]),
        default_guided_dir=str(paths["default_guided"]),
        distilled_guided_dir=str(paths["distilled_guided"]),
        pairing_csv=str(paths["pairing"]),
        metadata_csv=str(paths["metadata"]),
        output_dir=str(out_dir),
        tag=f"{task}_synthetic_smoke",
        min_episodes=n_pairs,
        min_quality_delta=0.02,
        max_bad_rate_increase=0.10,
        max_success_rate_drop=0.0,
        bootstrap_samples=500,
        board_target_force=1.0 if task == "board" else None,
        board_force_sigma=0.20 if task == "board" else None,
        seed=seed,
    )
    result = build_result(args)
    rows = result.pop("all_rows")
    task_out = out_dir / f"{task}_synthetic_smoke"
    task_out.mkdir(parents=True, exist_ok=True)
    json_path = task_out / "real_rollout_scorer_ablation_gate.json"
    md_path = task_out / "real_rollout_scorer_ablation_gate.md"
    csv_path = task_out / "real_rollout_scorer_ablation_episode_metrics.csv"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    write_csv(rows, csv_path)
    result["outputs"] = {"json": str(json_path), "markdown": str(md_path), "csv": str(csv_path)}
    return result


def smoke(args: argparse.Namespace) -> Dict:
    out_dir = Path(args.output_dir)
    data_root = out_dir / "synthetic_hdf5"
    mkdir_clean(data_root)
    results = {}
    for offset, task in enumerate(["insertion", "board"]):
        paths = make_task_data(data_root, task, args.n_pairs, args.seed + offset * 1000)
        results[task] = run_gate(task, paths, out_dir, args.n_pairs, args.seed + offset * 1000)
    insertion = results["insertion"]
    board = results["board"]
    # Insertion quality can saturate once both guided arms remove risk; this is
    # acceptable for smoke.  Board has an explicit target force, so it should
    # identify the smoother/closer distilled arm.
    passed = bool(
        insertion["production_ablation_pass"]
        and insertion["recommended_real_scorer"] in {"default_guided", "distilled_guided"}
        and board["production_ablation_pass"]
        and board["recommended_real_scorer"] == "distilled_guided"
        and board["guided_arm_comparison"]["winner"] == "distilled_guided"
    )
    summary = {
        "purpose": "Synthetic smoke test for three-arm real rollout scorer ablation gate.",
        "scientific_evidence": False,
        "n_pairs": args.n_pairs,
        "overall_pass": bool(passed),
        "expected_winner": {
            "insertion": "default_or_distilled_guided",
            "board": "distilled_guided",
        },
        "tasks": {
            task: {
                "production_ablation_pass": row["production_ablation_pass"],
                "recommended_real_scorer": row["recommended_real_scorer"],
                "guided_arm_winner": row["guided_arm_comparison"]["winner"],
                "outputs": row["outputs"],
            }
            for task, row in results.items()
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "real_rollout_scorer_ablation_smoke.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**summary, "summary_json": str(summary_path)}, ensure_ascii=False, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default=str(OUT_DIR))
    parser.add_argument("--n_pairs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    smoke(parse_args())
