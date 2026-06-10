"""Build the formal TacQuality rollout execution runbook.

This artifact is the operator-facing bridge between the acceptance protocol and
the actual rollout collection step.  It does not evaluate policy quality.  It
collects the exact server commands, rollout directories, required counts,
post-collection pairing commands, gate commands, and pass criteria into one
machine-readable/runbook file.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional


OUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_formal_rollout_runbook")

PATHS = {
    "acceptance_protocol": Path(
        "/home/chenshuai/Project/output/tac_quality_real_rollout_acceptance_protocol/"
        "tac_quality_real_rollout_acceptance_protocol.json"
    ),
    "launch_sheet": Path(
        "/home/chenshuai/Project/output/tac_quality_formal_launch_sheet/"
        "formal_paired12/tac_quality_formal_launch_sheet.json"
    ),
    "collection_readiness": Path(
        "/home/chenshuai/Project/output/tac_quality_collection_readiness/"
        "formal_paired12/tac_quality_collection_readiness.json"
    ),
    "post_collection_pipeline": Path(
        "/home/chenshuai/Project/output/tac_quality_post_collection_pipeline/"
        "formal_paired12/tac_quality_post_collection_pipeline.json"
    ),
    "collection_schedule": Path(
        "/home/chenshuai/Project/output/tac_quality_collection_schedule/"
        "formal_paired12/tac_quality_collection_schedule.json"
    ),
    "collection_progress": Path(
        "/home/chenshuai/Project/output/tac_quality_collection_progress/"
        "formal_paired12/tac_quality_collection_progress.json"
    ),
    "next_collection_step": Path(
        "/home/chenshuai/Project/output/tac_quality_next_collection_step/"
        "formal_paired12/tac_quality_next_collection_step.json"
    ),
}


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get(d: Optional[Dict[str, Any]], dotted: str, default=None):
    cur: Any = d
    if cur is None:
        return default
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


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


def build_task_runbook(task: str, protocol: Dict[str, Any], launch: Dict[str, Any], readiness: Dict[str, Any]) -> Dict[str, Any]:
    launch_task = get(launch, f"tasks.{task}", {}) or {}
    readiness_task = get(readiness, f"tasks.{task}", {}) or {}
    protocol_task = get(protocol, f"tasks.{task}", {}) or {}
    rollout_dirs = launch_task.get("rollout_dirs", {})
    launch_commands = launch_task.get("launch_commands", {})
    arms = readiness_task.get("arms", {})
    collection_order = launch_task.get("collection_order", ["baseline", "default_guided", "distilled_guided"])
    arm_rows = []
    for arm in collection_order:
        info = arms.get(arm, {})
        arm_rows.append(
            {
                "arm": arm,
                "rollout_dir": rollout_dirs.get(arm),
                "launch_command": launch_commands.get(arm),
                "needed_hdf5": info.get("needed", get(protocol_task, "collection.paired_n_pairs")),
                "current_hdf5": info.get("n_hdf5"),
                "missing_hdf5": info.get("missing_hdf5"),
                "ready": info.get("ready"),
            }
        )
    optional_arm = {
        "arm": "action_aware_guided",
        "rollout_dir": rollout_dirs.get("action_aware_guided"),
        "launch_command": launch_commands.get("action_aware_guided"),
        "status": "optional_after_formal_three_arm_review",
    }
    return {
        "task": task,
        "paired_n_pairs": get(protocol_task, "collection.paired_n_pairs"),
        "collection_order": collection_order,
        "arms": arm_rows,
        "optional_arm": optional_arm,
        "pairing": {
            "pairing_template": launch_task.get("pairing_template"),
            "three_arm_pairing_template": launch_task.get("three_arm_pairing_template"),
            "metadata_template": launch_task.get("metadata_template"),
        },
        "two_arm_gate": get(protocol_task, "two_arm_gate"),
        "three_arm_ablation_gate": get(protocol_task, "three_arm_ablation_gate"),
        "readiness": {
            "two_arm_ready": readiness_task.get("two_arm_ready"),
            "three_arm_ready": readiness_task.get("three_arm_ready"),
            "missing_items": readiness_task.get("missing_items"),
        },
    }


def build(paths: Dict[str, Path]) -> Dict[str, Any]:
    protocol = load_json(paths["acceptance_protocol"]) or {}
    launch = load_json(paths["launch_sheet"]) or {}
    readiness = load_json(paths["collection_readiness"]) or {}
    pipeline = load_json(paths["post_collection_pipeline"]) or {}
    schedule = load_json(paths["collection_schedule"]) or {}
    progress = load_json(paths["collection_progress"]) or {}
    next_step = load_json(paths["next_collection_step"]) or {}
    tasks = {
        task: build_task_runbook(task, protocol, launch, readiness)
        for task in ["insertion", "board"]
    }
    collection_steps = [
        "Open the formal launch sheet and create/confirm every rollout directory.",
        "Run the next-step artifact before each rollout to get the exact scheduled task, arm, launch command, and recommended_path.",
        "Run the next-step dry-run runner before touching the robot; inspect the smoke JSON if it fails.",
        "For each task, collect baseline, default_guided, and distilled_guided HDF5 rollouts with the listed server commands.",
        "After the robot/client saves a raw HDF5, run the finalize command to copy it to the current schedule recommended_path.",
        "Use matched task setup within each pair/triple; keep pair_id notes so pairing can be reviewed.",
        "Follow the counterbalanced collection schedule CSV; do not change arm order after seeing outcomes.",
        "Rerun the collection progress and next-step artifacts after each finalized HDF5.",
        "After collection, run the post-collection pipeline without --run_gates.",
        "Review generated pairing and metadata CSVs; fill missing success/stopped_early cells if HDF5 attrs are absent.",
        "Run the post-collection pipeline or formal gate runner with --run_gates only after preflight is ready.",
        "Review two-arm gates first, then three-arm scorer ablation gates.",
        "Rerun the goal audit after the four formal gate artifacts exist.",
    ]
    post_collection_commands = {
        "build_collection_schedule": (
            "python TFAC_V5/build_tac_quality_collection_schedule.py "
            "--tag formal_paired12"
        ),
        "collection_progress": "python TFAC_V5/build_tac_quality_collection_progress.py --tag formal_paired12",
        "next_collection_step": "python TFAC_V5/build_tac_quality_next_collection_step.py --tag formal_paired12",
        "next_collection_step_smoke_runner": (
            "conda run -n TactileACT python "
            "TFAC_V5/run_tac_quality_next_collection_step_smoke.py --tag formal_paired12"
        ),
        "current_collection_gate": (
            "conda run -n TactileACT python "
            "TFAC_V5/build_tac_quality_current_collection_gate.py --tag formal_paired12"
        ),
        "current_collection_handoff": (
            "conda run -n TactileACT python "
            "TFAC_V5/build_tac_quality_current_collection_handoff.py --tag formal_paired12"
        ),
        "finalize_collected_hdf5": "python TFAC_V5/finalize_tac_quality_collected_hdf5.py --source <collected_episode.hdf5>",
        "finalize_newest_from_dir": "python TFAC_V5/finalize_tac_quality_collected_hdf5.py --source_dir <collection_output_dir>",
        "pairing": get(launch, "post_collection_pairing_command"),
        "pipeline_preflight": (
            "python TFAC_V5/run_tac_quality_post_collection_pipeline.py "
            "--tag formal_paired12"
        ),
        "pipeline_run_gates": (
            "python TFAC_V5/run_tac_quality_post_collection_pipeline.py "
            "--tag formal_paired12 --run_gates"
        ),
        "formal_gate_runner_preflight": get(launch, "all_tasks_gate_runner_command"),
        "formal_gate_runner_run_gates": (
            str(get(launch, "all_tasks_gate_runner_command", "")) + " --run_gates"
            if get(launch, "all_tasks_gate_runner_command")
            else None
        ),
        "goal_audit": "python TFAC_V5/audit_tac_quality_goal_completion.py",
    }
    pass_criteria = {
        "two_arm": {
            "quality_delta_mean_at_least": get(
                protocol, "tasks.insertion.two_arm_gate.pass_conditions.quality_delta_mean_at_least"
            ),
            "bootstrap_ci95_low_must_be_positive": get(
                protocol, "tasks.insertion.two_arm_gate.pass_conditions.bootstrap_ci95_low_must_be_positive"
            ),
            "max_bad_rate_increase": get(
                protocol, "tasks.insertion.two_arm_gate.pass_conditions.max_bad_rate_increase"
            ),
            "max_success_rate_drop": get(
                protocol, "tasks.insertion.two_arm_gate.pass_conditions.max_success_rate_drop"
            ),
            "debug_or_underpowered_must_be_false": True,
        },
        "three_arm": get(protocol, "tasks.insertion.three_arm_ablation_gate.pass_conditions"),
    }
    runbook = {
        "name": "TacQuality formal paired12 rollout runbook",
        "purpose": "Single execution handoff for collecting the final real-rollout evidence for TacQuality DP guidance.",
        "scientific_evidence": False,
        "git_commit": git_commit(),
        "protocol_pass": get(protocol, "protocol_pass"),
        "launch_sheet_ready": get(launch, "launch_sheet_ready"),
        "pipeline_pass": get(pipeline, "pipeline_pass"),
        "ready_for_gate_runner": get(readiness, "ready_for_gate_runner"),
        "rollout_root": get(launch, "rollout_root"),
        "generated_pairing_dir": get(launch, "generated_pairing_dir"),
        "collection_schedule": {
            "schedule_pass": get(schedule, "schedule_pass"),
            "json": str(paths["collection_schedule"]),
            "csv": str(paths["collection_schedule"]).replace(".json", ".csv"),
            "markdown": str(paths["collection_schedule"]).replace(".json", ".md"),
            "task_order_policy": get(schedule, "task_order_policy"),
            "within_triplet_policy": get(schedule, "within_triplet_policy"),
        },
        "collection_progress": {
            "progress_pass": get(progress, "progress_pass"),
            "json": str(paths["collection_progress"]),
            "markdown": str(paths["collection_progress"]).replace(".json", ".md"),
            "n_completed_rows": get(progress, "n_completed_rows"),
            "n_scheduled_rows": get(progress, "n_scheduled_rows"),
            "ready_for_post_collection": get(progress, "ready_for_post_collection"),
            "next_row": get(progress, "next_row"),
        },
        "next_collection_step": {
            "next_step_pass": get(next_step, "next_step_pass"),
            "json": str(paths["next_collection_step"]),
            "markdown": str(paths["next_collection_step"]).replace(".json", ".md"),
            "recommended_path": get(next_step, "recommended_path"),
            "pre_collection_dry_run_required": get(next_step, "pre_collection_dry_run_required"),
            "pre_collection_dry_run_command": get(next_step, "pre_collection_dry_run_command"),
            "pre_collection_dry_run_output": get(next_step, "pre_collection_dry_run_output"),
            "finalize_command_template": get(next_step, "finalize_command_template"),
            "finalize_newest_from_dir_template": get(next_step, "finalize_newest_from_dir_template"),
        },
        "tasks": tasks,
        "collection_steps": collection_steps,
        "post_collection_commands": post_collection_commands,
        "pass_criteria": pass_criteria,
        "completion_blockers_to_close": get(protocol, "completion_blockers_to_close"),
        "cannot_count_as_completion": get(protocol, "cannot_count_as_completion"),
        "artifacts": {name: file_info(path) for name, path in paths.items()},
    }
    runbook["runbook_pass"] = bool(
        runbook["protocol_pass"] is True
        and runbook["launch_sheet_ready"] is True
        and runbook["pipeline_pass"] is True
        and len(runbook["completion_blockers_to_close"] or []) == 4
        and all(
            task_row["paired_n_pairs"] and task_row["paired_n_pairs"] >= 10
            for task_row in tasks.values()
        )
        and all(
            arm.get("launch_command") and arm.get("rollout_dir")
            for task_row in tasks.values()
            for arm in task_row["arms"]
        )
        and post_collection_commands["pipeline_run_gates"]
        and get(schedule, "schedule_pass") is True
        and get(progress, "progress_pass") is True
        and get(next_step, "next_step_pass") is True
        and "--dry_run_guidance_smoke" in str(get(next_step, "pre_collection_dry_run_command"))
        and "--smoke_output" in str(get(next_step, "pre_collection_dry_run_command"))
        and "conda run -n TactileACT" in post_collection_commands["next_collection_step_smoke_runner"]
        and "run_tac_quality_next_collection_step_smoke.py" in post_collection_commands["next_collection_step_smoke_runner"]
        and "build_tac_quality_current_collection_gate.py" in post_collection_commands["current_collection_gate"]
        and "build_tac_quality_current_collection_handoff.py" in post_collection_commands["current_collection_handoff"]
        and "finalize_tac_quality_collected_hdf5.py" in post_collection_commands["finalize_collected_hdf5"]
    )
    return runbook


def write_markdown(runbook: Dict[str, Any], path: Path) -> None:
    lines = [
        "# TacQuality Formal Paired12 Rollout Runbook",
        "",
        f"- runbook_pass: `{runbook['runbook_pass']}`",
        f"- scientific_evidence: `{runbook['scientific_evidence']}`",
        f"- protocol_pass: `{runbook['protocol_pass']}`",
        f"- launch_sheet_ready: `{runbook['launch_sheet_ready']}`",
        f"- ready_for_gate_runner: `{runbook['ready_for_gate_runner']}`",
        f"- rollout_root: `{runbook['rollout_root']}`",
        f"- collection_schedule_csv: `{runbook['collection_schedule']['csv']}`",
        f"- collection_progress: `{runbook['collection_progress']['json']}`",
        f"- next_collection_step: `{runbook['next_collection_step']['json']}`",
        f"- pre_collection_dry_run_output: `{runbook['next_collection_step']['pre_collection_dry_run_output']}`",
        "",
        "## Collection Steps",
        "",
    ]
    for i, step in enumerate(runbook["collection_steps"], start=1):
        lines.append(f"{i}. {step}")
    if runbook["next_collection_step"].get("pre_collection_dry_run_command"):
        lines.extend(
            [
                "",
                "## Current Pre-Collection Dry Run",
                "",
                "```bash",
                runbook["next_collection_step"]["pre_collection_dry_run_command"],
                "```",
            ]
        )
    lines.extend(["", "## Tasks", ""])
    for task, task_row in runbook["tasks"].items():
        lines.extend(
            [
                f"### {task}",
                "",
                f"- paired_n_pairs: `{task_row['paired_n_pairs']}`",
                f"- follow schedule: `{runbook['collection_schedule']['csv']}`",
                "",
                "| arm | needed | current | missing | ready | rollout_dir |",
                "|---|---:|---:|---:|---:|---|",
            ]
        )
        for arm in task_row["arms"]:
            lines.append(
                f"| {arm['arm']} | {arm['needed_hdf5']} | {arm['current_hdf5']} | "
                f"{arm['missing_hdf5']} | {arm['ready']} | `{arm['rollout_dir']}` |"
            )
        for arm in task_row["arms"]:
            lines.extend(
                [
                    "",
                    f"Launch `{task}/{arm['arm']}`:",
                    "",
                    "```bash",
                    arm["launch_command"] or "",
                    "```",
                ]
            )
    lines.extend(["", "## Post Collection", ""])
    for name, cmd in runbook["post_collection_commands"].items():
        lines.extend(["", f"### {name}", "", "```bash", cmd or "", "```"])
    lines.extend(["", "## Cannot Count As Completion", ""])
    for item in runbook["cannot_count_as_completion"] or []:
        lines.append(f"- {item}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default=str(OUT_DIR))
    parser.add_argument("--tag", default="formal_paired12")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    runbook = build(PATHS)
    json_path = out_dir / "tac_quality_formal_rollout_runbook.json"
    md_path = out_dir / "tac_quality_formal_rollout_runbook.md"
    json_path.write_text(json.dumps(runbook, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(runbook, md_path)
    print(
        json.dumps(
            {
                "runbook_pass": runbook["runbook_pass"],
                "scientific_evidence": runbook["scientific_evidence"],
                "ready_for_gate_runner": runbook["ready_for_gate_runner"],
                "rollout_root": runbook["rollout_root"],
                "json": str(json_path),
                "markdown": str(md_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
