"""Preflight and optionally run formal TacQuality rollout gates.

This script is the single entry point after collecting real/production HDF5
rollouts for the remaining TacQuality validation blockers.  It does not run a
robot.  It either:

  1. preflights the required directories, pairing CSVs, metadata CSVs, and
     expected HDF5 counts; or
  2. when --run_gates is set and all inputs are present, invokes the existing
     two-arm and three-arm rollout gate evaluators.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TFAC_V5.eval_real_rollout_quality_gate import discover_hdf5  # noqa: E402


DEFAULT_PACKET = Path(
    "/home/chenshuai/Project/output/real_rollout_experiment_packet/"
    "formal_paired12/real_rollout_experiment_packet.json"
)
DEFAULT_OUT_DIR = Path("/home/chenshuai/Project/output/formal_tac_quality_rollout_gate_runner")


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def file_info(path: Path) -> Dict[str, Any]:
    exists = path.exists()
    return {
        "path": str(path),
        "exists": bool(exists),
        "bytes": int(path.stat().st_size) if exists and path.is_file() else None,
    }


def dir_check(path_value: Optional[str], min_episodes: int) -> Dict[str, Any]:
    if not path_value:
        return {
            "path": None,
            "exists": False,
            "n_hdf5": 0,
            "ready": False,
            "reason": "directory argument not provided",
        }
    path = Path(path_value)
    exists = path.exists()
    files = discover_hdf5(path) if exists else []
    ready = exists and len(files) >= min_episodes
    return {
        "path": str(path),
        "exists": bool(exists),
        "n_hdf5": len(files),
        "ready": bool(ready),
        "reason": "ok" if ready else f"need at least {min_episodes} HDF5 files",
        "sample_files": [str(p) for p in files[:5]],
    }


def add_board_args(cmd: List[str], task_packet: Dict[str, Any]) -> None:
    text = task_packet.get("gate_command", "")
    parts = text.split()
    if "--board_target_force" in parts:
        idx = parts.index("--board_target_force")
        cmd.extend(["--board_target_force", parts[idx + 1]])
    if "--board_force_sigma" in parts:
        idx = parts.index("--board_force_sigma")
        cmd.extend(["--board_force_sigma", parts[idx + 1]])


def gate_commands(args: argparse.Namespace, packet: Dict[str, Any]) -> Dict[str, Optional[List[str]]]:
    commands: Dict[str, Optional[List[str]]] = {}
    for task in ["insertion", "board"]:
        task_packet = packet["tasks"][task]
        baseline = getattr(args, f"{task}_baseline_dir") or f"<{task}_baseline_rollout_dir>"
        default_guided = getattr(args, f"{task}_default_guided_dir") or f"<{task}_default_guided_rollout_dir>"
        distilled = getattr(args, f"{task}_distilled_guided_dir") or f"<{task}_distilled_guided_rollout_dir>"
        two_arm = [
            sys.executable,
            "TFAC_V5/eval_real_rollout_quality_gate.py",
            "--task",
            task,
            "--baseline_dir",
            baseline or "",
            "--guided_dir",
            default_guided or "",
            "--pairing_csv",
            task_packet["pairing_template"],
            "--metadata_csv",
            task_packet["metadata_template"],
            "--output_dir",
            "/home/chenshuai/Project/output/real_rollout_quality_gate",
            "--tag",
            f"{task}_baseline_vs_guided",
            "--min_episodes",
            str(args.min_episodes),
            "--bootstrap_samples",
            str(args.bootstrap_samples),
        ]
        three_arm = [
            sys.executable,
            "TFAC_V5/eval_real_rollout_scorer_ablation_gate.py",
            "--task",
            task,
            "--baseline_dir",
            baseline or "",
            "--default_guided_dir",
            default_guided or "",
            "--distilled_guided_dir",
            distilled or "",
            "--pairing_csv",
            task_packet["three_arm_pairing_template"],
            "--metadata_csv",
            task_packet["metadata_template"],
            "--output_dir",
            "/home/chenshuai/Project/output/real_rollout_scorer_ablation_gate",
            "--tag",
            f"{task}_baseline_vs_default_vs_distilled",
            "--min_episodes",
            str(args.min_episodes),
            "--bootstrap_samples",
            str(args.bootstrap_samples),
        ]
        if task == "board":
            add_board_args(two_arm, task_packet)
            add_board_args(three_arm, task_packet)
        commands[f"{task}_two_arm"] = two_arm
        commands[f"{task}_three_arm"] = three_arm
    return commands


def run_command(cmd: List[str]) -> Dict[str, Any]:
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
    return {
        "command": " ".join(cmd),
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "passed": proc.returncode == 0,
    }


def build_preflight(args: argparse.Namespace) -> Dict[str, Any]:
    packet = load_json(Path(args.packet))
    commands = gate_commands(args, packet)
    tasks: Dict[str, Any] = {}
    for task in ["insertion", "board"]:
        task_packet = packet["tasks"][task]
        checks = {
            "baseline": dir_check(getattr(args, f"{task}_baseline_dir"), args.min_episodes),
            "default_guided": dir_check(getattr(args, f"{task}_default_guided_dir"), args.min_episodes),
            "distilled_guided": dir_check(getattr(args, f"{task}_distilled_guided_dir"), args.min_episodes),
            "pairing_csv": file_info(Path(task_packet["pairing_template"])),
            "three_arm_pairing_csv": file_info(Path(task_packet["three_arm_pairing_template"])),
            "metadata_csv": file_info(Path(task_packet["metadata_template"])),
        }
        two_arm_ready = (
            checks["baseline"]["ready"]
            and checks["default_guided"]["ready"]
            and checks["pairing_csv"]["exists"]
            and checks["metadata_csv"]["exists"]
        )
        three_arm_ready = (
            two_arm_ready
            and checks["distilled_guided"]["ready"]
            and checks["three_arm_pairing_csv"]["exists"]
        )
        tasks[task] = {
            "two_arm_ready": bool(two_arm_ready),
            "three_arm_ready": bool(three_arm_ready),
            "checks": checks,
            "commands": {
                "two_arm": " ".join(commands[f"{task}_two_arm"]),
                "three_arm": " ".join(commands[f"{task}_three_arm"]),
            },
        }

    result: Dict[str, Any] = {
        "purpose": "Preflight and optional execution wrapper for formal TacQuality real-rollout gates.",
        "scientific_evidence": False,
        "git_commit": git_commit(),
        "packet": str(args.packet),
        "run_gates_requested": bool(args.run_gates),
        "min_episodes": args.min_episodes,
        "bootstrap_samples": args.bootstrap_samples,
        "tasks": tasks,
        "preflight_ready": bool(
            all(row["two_arm_ready"] and row["three_arm_ready"] for row in tasks.values())
        ),
        "gate_results": {},
    }
    if args.run_gates and result["preflight_ready"]:
        for name, cmd in commands.items():
            result["gate_results"][name] = run_command(cmd)
        result["all_requested_gates_passed"] = all(row["passed"] for row in result["gate_results"].values())
        result["scientific_evidence"] = bool(result["all_requested_gates_passed"])
    elif args.run_gates:
        result["all_requested_gates_passed"] = False
        result["run_skip_reason"] = "preflight_ready=false"
    else:
        result["all_requested_gates_passed"] = None
        result["run_skip_reason"] = "preflight only; pass --run_gates after collecting rollout HDF5 directories"
    return result


def write_markdown(result: Dict[str, Any], path: Path) -> None:
    lines = [
        "# Formal TacQuality Rollout Gate Runner",
        "",
        f"- preflight_ready: `{result['preflight_ready']}`",
        f"- run_gates_requested: `{result['run_gates_requested']}`",
        f"- scientific_evidence: `{result['scientific_evidence']}`",
        f"- run_skip_reason: {result.get('run_skip_reason')}",
        "",
        "## Tasks",
        "",
        "| task | two_arm_ready | three_arm_ready | baseline n | default n | distilled n |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for task, row in result["tasks"].items():
        checks = row["checks"]
        lines.append(
            f"| {task} | {row['two_arm_ready']} | {row['three_arm_ready']} | "
            f"{checks['baseline']['n_hdf5']} | {checks['default_guided']['n_hdf5']} | "
            f"{checks['distilled_guided']['n_hdf5']} |"
        )
    lines.extend(["", "## Commands", ""])
    for task, row in result["tasks"].items():
        lines.extend(
            [
                f"### {task}",
                "",
                "```bash",
                row["commands"]["two_arm"],
                row["commands"]["three_arm"],
                "```",
                "",
            ]
        )
    if result["gate_results"]:
        lines.extend(["## Gate Results", "", "```json", json.dumps(result["gate_results"], ensure_ascii=False, indent=2), "```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", default=str(DEFAULT_PACKET))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--tag", default="formal_paired12_preflight")
    parser.add_argument("--min_episodes", type=int, default=10)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--run_gates", action="store_true")
    for task in ["insertion", "board"]:
        parser.add_argument(f"--{task}_baseline_dir", default=None)
        parser.add_argument(f"--{task}_default_guided_dir", default=None)
        parser.add_argument(f"--{task}_distilled_guided_dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_preflight(args)
    out_dir = Path(args.output_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "formal_tac_quality_rollout_gate_runner.json"
    md_path = out_dir / "formal_tac_quality_rollout_gate_runner.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    print(
        json.dumps(
            {
                "preflight_ready": result["preflight_ready"],
                "run_gates_requested": result["run_gates_requested"],
                "scientific_evidence": result["scientific_evidence"],
                "json": str(json_path),
                "markdown": str(md_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
