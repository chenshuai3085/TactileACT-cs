"""Build a formal collection readiness dashboard for TacQuality rollouts.

This script does not evaluate policy quality and does not claim scientific
evidence.  It checks whether the formal baseline/default/distilled rollout
directories and pairing/metadata templates are ready for the final two-arm and
three-arm real-rollout gates.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TFAC_V5.eval_real_rollout_quality_gate import discover_hdf5  # noqa: E402


DEFAULT_LAUNCH_SHEET = Path(
    "/home/chenshuai/Project/output/tac_quality_formal_launch_sheet/"
    "formal_paired12/tac_quality_formal_launch_sheet.json"
)
DEFAULT_PACKET = Path(
    "/home/chenshuai/Project/output/real_rollout_experiment_packet/"
    "formal_paired12/real_rollout_experiment_packet.json"
)
DEFAULT_OUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_collection_readiness")
DEFAULT_PAIRING_REPORT = Path(
    "/home/chenshuai/Project/output/tac_quality_rollout_pairing/"
    "formal_paired12/tac_quality_rollout_pairing.json"
)
DEFAULT_PAIRING_METADATA_AUDIT = Path(
    "/home/chenshuai/Project/output/tac_quality_pairing_metadata_audit/"
    "tac_quality_pairing_metadata_audit.json"
)


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


def dir_info(path: Path, needed: int, create_dirs: bool) -> Dict[str, Any]:
    if create_dirs:
        path.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    files = discover_hdf5(path) if exists else []
    ready = exists and len(files) >= needed
    return {
        "path": str(path),
        "exists": bool(exists),
        "n_hdf5": len(files),
        "needed": int(needed),
        "missing_hdf5": max(0, int(needed) - len(files)),
        "ready": bool(ready),
        "sample_files": [str(p) for p in files[:5]],
    }


def missing_items_for_task(task: str, row: Dict[str, Any]) -> List[str]:
    missing: List[str] = []
    for arm, info in row["arms"].items():
        if not info["exists"]:
            missing.append(f"{task}/{arm}: create rollout directory {info['path']}")
        if info["missing_hdf5"] > 0:
            missing.append(
                f"{task}/{arm}: collect {info['missing_hdf5']} more HDF5 "
                f"(have {info['n_hdf5']}, need {info['needed']})"
            )
    for name, info in row["templates"].items():
        if not info["exists"]:
            missing.append(f"{task}/{name}: missing template {info['path']}")
    return missing


def build_readiness(args: argparse.Namespace) -> Dict[str, Any]:
    launch = load_json(Path(args.launch_sheet))
    packet = load_json(Path(args.packet))
    pairing_report = load_json(Path(args.pairing_report)) if Path(args.pairing_report).exists() else None
    pairing_metadata_audit = (
        load_json(Path(args.pairing_metadata_audit))
        if Path(args.pairing_metadata_audit).exists()
        else None
    )
    tasks: Dict[str, Any] = {}
    all_missing: List[str] = []

    for task in ["insertion", "board"]:
        launch_task = launch["tasks"][task]
        packet_task = packet["tasks"][task]
        needed = int(args.min_episodes or packet_task.get("paired_n_pairs") or launch_task.get("paired_n_pairs") or 10)
        arms = {
            arm: dir_info(Path(path), needed, args.create_dirs)
            for arm, path in launch_task["rollout_dirs"].items()
        }
        templates = {
            "pairing_csv": file_info(Path(packet_task["pairing_template"])),
            "three_arm_pairing_csv": file_info(Path(packet_task["three_arm_pairing_template"])),
            "metadata_csv": file_info(Path(packet_task["metadata_template"])),
        }
        two_arm_ready = (
            arms["baseline"]["ready"]
            and arms["default_guided"]["ready"]
            and templates["pairing_csv"]["exists"]
            and templates["metadata_csv"]["exists"]
        )
        three_arm_ready = (
            two_arm_ready
            and arms["distilled_guided"]["ready"]
            and templates["three_arm_pairing_csv"]["exists"]
        )
        row = {
            "needed_per_arm": needed,
            "arms": arms,
            "templates": templates,
            "two_arm_ready": bool(two_arm_ready),
            "three_arm_ready": bool(three_arm_ready),
            "gate_commands": {
                "two_arm_template": packet_task["gate_command"],
                "three_arm_template": packet_task["ablation_gate_command"],
            },
            "server_commands": launch_task.get("launch_commands", {}),
            "ports": launch_task.get("ports", {}),
        }
        row["missing_items"] = missing_items_for_task(task, row)
        all_missing.extend(row["missing_items"])
        tasks[task] = row

    all_collection_dirs_exist = all(
        arm["exists"] for task in tasks.values() for arm in task["arms"].values()
    )
    result: Dict[str, Any] = {
        "purpose": "Collection readiness for formal TacQuality real-rollout gates.",
        "scientific_evidence": False,
        "git_commit": git_commit(),
        "launch_sheet": str(args.launch_sheet),
        "packet": str(args.packet),
        "rollout_root": launch.get("rollout_root"),
        "create_dirs": bool(args.create_dirs),
        "min_episodes": int(args.min_episodes),
        "pairing_report": str(args.pairing_report),
        "pairing_report_exists": pairing_report is not None,
        "pairing_report_overall_ready": pairing_report.get("overall_ready") if pairing_report else None,
        "pairing_metadata_audit": str(args.pairing_metadata_audit),
        "pairing_metadata_audit_exists": pairing_metadata_audit is not None,
        "pairing_metadata_all_tasks_ready": pairing_metadata_audit.get("all_tasks_ready") if pairing_metadata_audit else None,
        "tasks": tasks,
        "all_collection_dirs_exist": bool(all_collection_dirs_exist),
        "all_templates_exist": all(
            info["exists"] for task in tasks.values() for info in task["templates"].values()
        ),
        "ready_for_two_arm_gates": all(task["two_arm_ready"] for task in tasks.values()),
        "ready_for_three_arm_gates": all(task["three_arm_ready"] for task in tasks.values()),
        "ready_for_gate_runner": all(task["three_arm_ready"] for task in tasks.values()),
        "missing_items": all_missing,
        "next_required_step": (
            "Collect the missing HDF5 rollouts listed in missing_items, then run "
            "TFAC_V5/build_tac_quality_rollout_pairing.py to generate concrete "
            "pairing/metadata CSVs before running the formal gates."
        ),
        "post_collection_pairing_command": (
            "python TFAC_V5/build_tac_quality_rollout_pairing.py "
            f"--launch_sheet {args.launch_sheet} --tag {args.tag}"
        ),
    }
    if result["ready_for_gate_runner"]:
        result["next_required_step"] = (
            "Run TFAC_V5/build_tac_quality_rollout_pairing.py, review blank "
            "metadata cells, run TFAC_V5/audit_tac_quality_pairing_metadata.py, "
            "then run TFAC_V5/run_formal_tac_quality_rollout_gates.py --run_gates."
        )
    return result


def write_markdown(result: Dict[str, Any], path: Path) -> None:
    lines = [
        "# TacQuality Formal Collection Readiness",
        "",
        f"- ready_for_gate_runner: `{result['ready_for_gate_runner']}`",
        f"- ready_for_two_arm_gates: `{result['ready_for_two_arm_gates']}`",
        f"- ready_for_three_arm_gates: `{result['ready_for_three_arm_gates']}`",
        f"- scientific_evidence: `{result['scientific_evidence']}`",
        f"- rollout_root: `{result['rollout_root']}`",
        f"- pairing_report_exists: `{result['pairing_report_exists']}`",
        f"- pairing_report_overall_ready: `{result['pairing_report_overall_ready']}`",
        f"- pairing_metadata_audit_exists: `{result['pairing_metadata_audit_exists']}`",
        f"- pairing_metadata_all_tasks_ready: `{result['pairing_metadata_all_tasks_ready']}`",
        f"- next_required_step: {result['next_required_step']}",
        "",
        "## HDF5 Counts",
        "",
        "| task | arm | exists | n_hdf5 | needed | missing | ready |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for task, task_row in result["tasks"].items():
        for arm, info in task_row["arms"].items():
            lines.append(
                f"| {task} | {arm} | {info['exists']} | {info['n_hdf5']} | "
                f"{info['needed']} | {info['missing_hdf5']} | {info['ready']} |"
            )
    lines.extend(
        [
            "",
            "## Templates",
            "",
            "| task | template | exists | path |",
            "|---|---|---:|---|",
        ]
    )
    for task, task_row in result["tasks"].items():
        for name, info in task_row["templates"].items():
            lines.append(f"| {task} | {name} | {info['exists']} | `{info['path']}` |")
    lines.extend(["", "## Missing Items", ""])
    if result["missing_items"]:
        for item in result["missing_items"]:
            lines.append(f"- {item}")
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Post-Collection Pairing",
            "",
            "```bash",
            result["post_collection_pairing_command"],
            "```",
            "",
        ]
    )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch_sheet", default=str(DEFAULT_LAUNCH_SHEET))
    parser.add_argument("--packet", default=str(DEFAULT_PACKET))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--tag", default="formal_paired12")
    parser.add_argument("--min_episodes", type=int, default=10)
    parser.add_argument("--pairing_report", default=str(DEFAULT_PAIRING_REPORT))
    parser.add_argument("--pairing_metadata_audit", default=str(DEFAULT_PAIRING_METADATA_AUDIT))
    parser.add_argument("--create_dirs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_readiness(args)
    out_dir = Path(args.output_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "tac_quality_collection_readiness.json"
    md_path = out_dir / "tac_quality_collection_readiness.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    print(
        json.dumps(
            {
                "ready_for_gate_runner": result["ready_for_gate_runner"],
                "ready_for_two_arm_gates": result["ready_for_two_arm_gates"],
                "ready_for_three_arm_gates": result["ready_for_three_arm_gates"],
                "all_collection_dirs_exist": result["all_collection_dirs_exist"],
                "n_missing_items": len(result["missing_items"]),
                "json": str(json_path),
                "markdown": str(md_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
