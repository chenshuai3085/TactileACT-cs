"""Smoke-test formal gate runner with generated pairing CSVs.

This creates synthetic HDF5 rollouts in temporary formal-like directories,
builds concrete pairing/metadata CSVs with
``build_tac_quality_rollout_pairing.py``, and then runs
``run_formal_tac_quality_rollout_gates.py --use_generated_pairing --run_gates``.

The synthetic data are not scientific evidence.  The smoke only verifies that
the post-collection evaluation plumbing works end to end.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TFAC_V5.run_formal_tac_quality_rollout_gates import DEFAULT_PACKET, load_json  # noqa: E402


OUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_generated_pairing_gate_runner_smoke")
ARMS = ("baseline", "default_guided", "distilled_guided", "action_aware_guided")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


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
    if task == "board":
        force_level = [0.18, 0.55, 0.58, 0.60][quality_rank]
        force_noise = [0.20, 0.055, 0.025, 0.020][quality_rank]
        action_rough = [0.055, 0.010, 0.002, 0.0015][quality_rank]
        marker_scale = [0.18, 0.08, 0.035, 0.030][quality_rank]
    else:
        force_level = [0.75, 0.42, 0.28, 0.24][quality_rank]
        force_noise = [0.22, 0.08, 0.035, 0.030][quality_rank]
        action_rough = [0.055, 0.014, 0.004, 0.003][quality_rank]
        marker_scale = [0.28, 0.10, 0.045, 0.040][quality_rank]

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


def make_rollouts(root: Path, n_pairs: int, seed: int) -> Dict[str, Dict[str, str]]:
    all_dirs: Dict[str, Dict[str, str]] = {}
    for task_idx, task in enumerate(["insertion", "board"]):
        all_dirs[task] = {}
        for rank, arm in enumerate(ARMS):
            arm_dir = root / task / arm
            arm_dir.mkdir(parents=True, exist_ok=True)
            all_dirs[task][arm] = str(arm_dir)
            for i in range(n_pairs):
                write_rollout(
                    arm_dir / f"episode_{i + 1:03d}.hdf5",
                    task=task,
                    quality_rank=rank,
                    seed=seed + task_idx * 1000 + i * 17 + rank,
                )
    return all_dirs


def build_launch_sheet(base_launch_sheet: Dict[str, Any], rollout_dirs: Dict[str, Dict[str, str]], path: Path) -> None:
    launch = json.loads(json.dumps(base_launch_sheet))
    launch["rollout_root"] = str(path.parent / "synthetic_rollouts")
    for task in ["insertion", "board"]:
        launch["tasks"][task]["rollout_dirs"] = rollout_dirs[task]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(launch, ensure_ascii=False, indent=2), encoding="utf-8")


def run_cmd(cmd: List[str]) -> Dict[str, Any]:
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
    return {
        "command": " ".join(cmd),
        "returncode": proc.returncode,
        "passed": proc.returncode == 0,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def smoke(args: argparse.Namespace) -> Dict[str, Any]:
    out_root = Path(args.output_dir) / args.tag
    mkdir_clean(out_root)
    rollout_root = out_root / "synthetic_rollouts"
    pairing_root = out_root / "generated_pairing"
    gate_out = out_root / "gate_runner"
    quality_gate_out = out_root / "quality_gate_results"
    ablation_gate_out = out_root / "ablation_gate_results"
    rollout_dirs = make_rollouts(rollout_root, args.n_pairs, args.seed)

    base_launch_sheet = load_json(Path(args.launch_sheet))
    smoke_launch_sheet = out_root / "synthetic_launch_sheet.json"
    build_launch_sheet(base_launch_sheet, rollout_dirs, smoke_launch_sheet)

    pairing_cmd = [
        sys.executable,
        "TFAC_V5/build_tac_quality_rollout_pairing.py",
        "--launch_sheet",
        str(smoke_launch_sheet),
        "--output_dir",
        str(pairing_root),
        "--tag",
        "synthetic",
    ]
    pairing_result = run_cmd(pairing_cmd)
    pairing_json = pairing_root / "synthetic" / "tac_quality_rollout_pairing.json"
    pairing_report = load_json(pairing_json) if pairing_json.exists() else {}

    gate_cmd = [
        sys.executable,
        "TFAC_V5/run_formal_tac_quality_rollout_gates.py",
        "--packet",
        str(args.packet),
        "--output_dir",
        str(gate_out),
        "--tag",
        "synthetic_generated_pairing",
        "--min_episodes",
        str(args.n_pairs),
        "--bootstrap_samples",
        str(args.bootstrap_samples),
        "--use_generated_pairing",
        "--generated_pairing_dir",
        str(pairing_root / "synthetic"),
        "--quality_gate_output_dir",
        str(quality_gate_out),
        "--ablation_gate_output_dir",
        str(ablation_gate_out),
        "--run_gates",
    ]
    for task in ["insertion", "board"]:
        gate_cmd.extend(
            [
                f"--{task}_baseline_dir",
                rollout_dirs[task]["baseline"],
                f"--{task}_default_guided_dir",
                rollout_dirs[task]["default_guided"],
                f"--{task}_distilled_guided_dir",
                rollout_dirs[task]["distilled_guided"],
            ]
        )
    gate_result = run_cmd(gate_cmd)
    gate_json = gate_out / "synthetic_generated_pairing" / "formal_tac_quality_rollout_gate_runner.json"
    gate_report = load_json(gate_json) if gate_json.exists() else {}

    gate_passed = bool(
        gate_report.get("use_generated_pairing") is True
        and gate_report.get("preflight_ready") is True
        and gate_report.get("all_requested_gates_passed") is True
        and gate_report.get("scientific_evidence") is True
        and all(row.get("passed") for row in gate_report.get("gate_results", {}).values())
    )
    summary = {
        "purpose": "Synthetic smoke for generated-pairing formal TacQuality gate runner.",
        "scientific_evidence": False,
        "git_commit": git_commit(),
        "n_pairs": args.n_pairs,
        "overall_pass": bool(pairing_result["passed"] and pairing_report.get("overall_ready") is True and gate_passed),
        "pairing_command": pairing_result,
        "gate_command": gate_result,
        "pairing_report": {
            "path": str(pairing_json),
            "overall_ready": pairing_report.get("overall_ready"),
            "tasks": {
                task: {
                    "ready": pairing_report.get("tasks", {}).get(task, {}).get("ready"),
                    "n_pairs": pairing_report.get("tasks", {}).get(task, {}).get("n_pairs"),
                    "n_hdf5": pairing_report.get("tasks", {}).get(task, {}).get("n_hdf5"),
                }
                for task in ["insertion", "board"]
            },
        },
        "gate_report": {
            "path": str(gate_json),
            "preflight_ready": gate_report.get("preflight_ready"),
            "use_generated_pairing": gate_report.get("use_generated_pairing"),
            "all_requested_gates_passed": gate_report.get("all_requested_gates_passed"),
            "scientific_evidence_in_gate_report": gate_report.get("scientific_evidence"),
            "gate_result_passes": {
                name: row.get("passed")
                for name, row in gate_report.get("gate_results", {}).items()
            },
        },
        "note": "Synthetic smoke only; not real robot or production rollout evidence.",
    }
    summary_json = out_root / "tac_quality_generated_pairing_gate_runner_smoke.json"
    summary_md = out_root / "tac_quality_generated_pairing_gate_runner_smoke.md"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(summary, summary_md)
    print(
        json.dumps(
            {
                "overall_pass": summary["overall_pass"],
                "pairing_ready": summary["pairing_report"]["overall_ready"],
                "gate_preflight_ready": summary["gate_report"]["preflight_ready"],
                "gate_all_requested_passed": summary["gate_report"]["all_requested_gates_passed"],
                "json": str(summary_json),
                "markdown": str(summary_md),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return summary


def write_markdown(summary: Dict[str, Any], path: Path) -> None:
    lines = [
        "# Generated-Pairing Gate Runner Smoke",
        "",
        f"- overall_pass: `{summary['overall_pass']}`",
        f"- scientific_evidence: `{summary['scientific_evidence']}`",
        f"- n_pairs: `{summary['n_pairs']}`",
        f"- pairing_ready: `{summary['pairing_report']['overall_ready']}`",
        f"- gate_preflight_ready: `{summary['gate_report']['preflight_ready']}`",
        f"- gate_all_requested_gates_passed: `{summary['gate_report']['all_requested_gates_passed']}`",
        f"- note: {summary['note']}",
        "",
        "## Gate Result Passes",
        "",
        "| gate | passed |",
        "|---|---:|",
    ]
    for name, passed in summary["gate_report"]["gate_result_passes"].items():
        lines.append(f"| {name} | {passed} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch_sheet", default="/home/chenshuai/Project/output/tac_quality_formal_launch_sheet/formal_paired12/tac_quality_formal_launch_sheet.json")
    parser.add_argument("--packet", default=str(DEFAULT_PACKET))
    parser.add_argument("--output_dir", default=str(OUT_DIR))
    parser.add_argument("--tag", default="synthetic_n10")
    parser.add_argument("--n_pairs", type=int, default=10)
    parser.add_argument("--bootstrap_samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    smoke(parse_args())
