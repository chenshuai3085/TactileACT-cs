"""Preflight and optionally evaluate optional ActionAware real rollouts.

This is deliberately separate from the formal TacQuality three-arm gate:

  baseline
  default_guided
  distilled_guided

ActionAware is an optional fourth-arm ablation candidate.  It should not be
required to close the formal validation gate, but if real/production rollouts
are collected for action_aware_guided, this script evaluates:

  baseline DP vs ActionAware quality-mode line-search guided DP

using the same two-arm real rollout quality gate.
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
DEFAULT_ROLLOUT_ROOT = Path("/home/chenshuai/Project/output/tac_quality_formal_rollouts")
DEFAULT_OUT_DIR = Path("/home/chenshuai/Project/output/optional_action_aware_rollout_gate_runner")
DEFAULT_QUALITY_GATE_OUT = Path("/home/chenshuai/Project/output/optional_action_aware_rollout_quality_gate")
DEFAULT_ACTION_AWARE_PAIRING_DIR = Path(
    "/home/chenshuai/Project/output/tac_quality_rollout_pairing/formal_paired12"
)


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def discover_count(path_value: Optional[str], min_episodes: int) -> Dict[str, Any]:
    if not path_value:
        return {
            "path": None,
            "exists": False,
            "n_hdf5": 0,
            "ready": False,
            "reason": "directory argument not provided",
        }
    path = Path(path_value)
    files = discover_hdf5(path) if path.exists() else []
    ready = path.exists() and len(files) >= min_episodes
    return {
        "path": str(path),
        "exists": bool(path.exists()),
        "n_hdf5": len(files),
        "ready": bool(ready),
        "reason": "ok" if ready else f"need at least {min_episodes} HDF5 files",
        "sample_files": [str(p) for p in files[:5]],
    }


def default_dir(task: str, arm: str, rollout_root: Path) -> str:
    return str(rollout_root / task / arm)


def pairing_paths(args: argparse.Namespace, task: str) -> Dict[str, str]:
    if args.action_aware_pairing_dir:
        root = Path(args.action_aware_pairing_dir) / task
        return {
            "pairing_csv": str(root / "action_aware_pairing.csv"),
            "metadata_csv": str(root / "metadata_generated.csv"),
        }
    packet = load_json(Path(args.packet))
    return {
        "pairing_csv": packet["tasks"][task]["pairing_template"],
        "metadata_csv": packet["tasks"][task]["metadata_template"],
    }


def file_info(path_value: str) -> Dict[str, Any]:
    path = Path(path_value)
    return {
        "path": str(path),
        "exists": bool(path.exists()),
        "bytes": int(path.stat().st_size) if path.exists() and path.is_file() else None,
    }


def add_board_args(cmd: List[str], task: str, packet_path: str) -> None:
    if task != "board":
        return
    packet = load_json(Path(packet_path))
    text = packet["tasks"]["board"].get("gate_command", "")
    parts = text.split()
    if "--board_target_force" in parts:
        idx = parts.index("--board_target_force")
        cmd.extend(["--board_target_force", parts[idx + 1]])
    if "--board_force_sigma" in parts:
        idx = parts.index("--board_force_sigma")
        cmd.extend(["--board_force_sigma", parts[idx + 1]])


def gate_command(args: argparse.Namespace, task: str, baseline_dir: str, action_aware_dir: str) -> List[str]:
    paths = pairing_paths(args, task)
    cmd = [
        sys.executable,
        "-m",
        "TFAC_V5.eval_real_rollout_quality_gate",
        "--task",
        task,
        "--baseline_dir",
        baseline_dir,
        "--guided_dir",
        action_aware_dir,
        "--pairing_csv",
        paths["pairing_csv"],
        "--metadata_csv",
        paths["metadata_csv"],
        "--output_dir",
        str(args.quality_gate_output_dir),
        "--tag",
        f"{task}_baseline_vs_action_aware",
        "--min_episodes",
        str(args.min_episodes),
        "--bootstrap_samples",
        str(args.bootstrap_samples),
    ]
    add_board_args(cmd, task, args.packet)
    return cmd


def run_command(cmd: List[str]) -> Dict[str, Any]:
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
    return {
        "command": " ".join(cmd),
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "passed": proc.returncode == 0,
    }


def build(args: argparse.Namespace) -> Dict[str, Any]:
    rollout_root = Path(args.rollout_root)
    tasks: Dict[str, Any] = {}
    for task in ["insertion", "board"]:
        baseline_dir = getattr(args, f"{task}_baseline_dir") or default_dir(task, "baseline", rollout_root)
        action_aware_dir = (
            getattr(args, f"{task}_action_aware_guided_dir")
            or default_dir(task, "action_aware_guided", rollout_root)
        )
        csvs = pairing_paths(args, task)
        checks = {
            "baseline": discover_count(baseline_dir, args.min_episodes),
            "action_aware_guided": discover_count(action_aware_dir, args.min_episodes),
            "pairing_csv": file_info(csvs["pairing_csv"]),
            "metadata_csv": file_info(csvs["metadata_csv"]),
        }
        ready = (
            checks["baseline"]["ready"]
            and checks["action_aware_guided"]["ready"]
            and checks["pairing_csv"]["exists"]
            and checks["metadata_csv"]["exists"]
        )
        cmd = gate_command(args, task, baseline_dir, action_aware_dir)
        tasks[task] = {
            "ready": bool(ready),
            "checks": checks,
            "command": " ".join(cmd),
            "gate_result": run_command(cmd) if args.run_gates and ready else None,
        }
        if args.run_gates and not ready:
            tasks[task]["run_skip_reason"] = "preflight ready=false"

    ran = [row["gate_result"] for row in tasks.values() if row["gate_result"] is not None]
    result = {
        "purpose": "Optional ActionAware baseline-vs-guided rollout gate runner.",
        "scientific_evidence": False,
        "formal_gate_dependency": False,
        "git_commit": git_commit(),
        "packet": str(args.packet),
        "rollout_root": str(rollout_root),
        "action_aware_pairing_dir": str(args.action_aware_pairing_dir) if args.action_aware_pairing_dir else None,
        "min_episodes": int(args.min_episodes),
        "bootstrap_samples": int(args.bootstrap_samples),
        "run_gates_requested": bool(args.run_gates),
        "quality_gate_output_dir": str(args.quality_gate_output_dir),
        "tasks": tasks,
        "preflight_ready": bool(all(row["ready"] for row in tasks.values())),
        "all_requested_gates_passed": None
        if not args.run_gates
        else bool(ran and all(row["passed"] for row in ran) and len(ran) == len(tasks)),
        "interpretation": (
            "Optional ActionAware evidence only.  Formal completion still depends on "
            "baseline/default/distilled real rollout gates."
        ),
    }
    result["scientific_evidence"] = bool(result["all_requested_gates_passed"] is True)
    return result


def write_markdown(result: Dict[str, Any], path: Path) -> None:
    lines = [
        "# Optional ActionAware Rollout Gate Runner",
        "",
        f"- preflight_ready: `{result['preflight_ready']}`",
        f"- run_gates_requested: `{result['run_gates_requested']}`",
        f"- all_requested_gates_passed: `{result['all_requested_gates_passed']}`",
        f"- scientific_evidence: `{result['scientific_evidence']}`",
        f"- formal_gate_dependency: `{result['formal_gate_dependency']}`",
        "",
        "| task | ready | baseline n | action_aware n |",
        "|---|---:|---:|---:|",
    ]
    for task, row in result["tasks"].items():
        checks = row["checks"]
        lines.append(
            f"| {task} | {row['ready']} | {checks['baseline']['n_hdf5']} | "
            f"{checks['action_aware_guided']['n_hdf5']} |"
        )
    lines.extend(["", "## Commands", ""])
    for task, row in result["tasks"].items():
        lines.extend([f"### {task}", "", "```bash", row["command"], "```", ""])
    lines.extend(["## Interpretation", "", result["interpretation"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", default=str(DEFAULT_PACKET))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--tag", default="formal_paired12_preflight")
    parser.add_argument("--rollout_root", default=str(DEFAULT_ROLLOUT_ROOT))
    parser.add_argument("--action_aware_pairing_dir", default=str(DEFAULT_ACTION_AWARE_PAIRING_DIR))
    parser.add_argument("--quality_gate_output_dir", default=str(DEFAULT_QUALITY_GATE_OUT))
    parser.add_argument("--min_episodes", type=int, default=10)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--run_gates", action="store_true")
    for task in ["insertion", "board"]:
        parser.add_argument(f"--{task}_baseline_dir", default=None)
        parser.add_argument(f"--{task}_action_aware_guided_dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build(args)
    out_dir = Path(args.output_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "optional_action_aware_rollout_gate_runner.json"
    md_path = out_dir / "optional_action_aware_rollout_gate_runner.md"
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
