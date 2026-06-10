"""Build live progress and next-trial instructions for formal TacQuality rollout collection.

This tool reads the counterbalanced schedule and the current rollout
directories, then reports which scheduled rows are already represented by HDF5
files and which row should be collected next.  It does not evaluate quality and
does not claim scientific evidence.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TFAC_V5.eval_real_rollout_quality_gate import discover_hdf5  # noqa: E402


OUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_collection_progress")
DEFAULT_SCHEDULE = Path(
    "/home/chenshuai/Project/output/tac_quality_collection_schedule/"
    "formal_paired12/tac_quality_collection_schedule.json"
)
FORMAL_ARMS = ("baseline", "default_guided", "distilled_guided")


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def hdf5_count(root: str) -> int:
    path = Path(root)
    if not path.exists():
        return 0
    return len(discover_hdf5(path))


def recommended_exists(row: Dict[str, Any]) -> bool:
    path = row.get("recommended_path")
    return bool(path and Path(path).exists())


def build(args: argparse.Namespace) -> Dict[str, Any]:
    schedule = load_json(Path(args.schedule))
    rows = schedule.get("long_schedule_rows", [])
    counts: Dict[tuple[str, str], int] = {}
    completed_rows: List[Dict[str, Any]] = []
    pending_rows: List[Dict[str, Any]] = []
    arm_totals: Dict[str, Dict[str, Any]] = {}

    for row in rows:
        key = (row["task"], row["arm"])
        if key not in counts:
            counts[key] = hdf5_count(row["rollout_dir"])

    seen_per_arm: Dict[tuple[str, str], int] = {}
    for row in rows:
        key = (row["task"], row["arm"])
        seen = seen_per_arm.get(key, 0) + 1
        seen_per_arm[key] = seen
        row_report = dict(row)
        row_report["hdf5_count_for_task_arm"] = counts[key]
        row_report["scheduled_index_for_task_arm"] = seen
        row_report["recommended_path_exists"] = recommended_exists(row)
        row_report["completed_by_recommended_path"] = row_report["recommended_path_exists"]
        row_report["completed_by_count_fallback"] = counts[key] >= seen
        row_report["completed"] = bool(
            row_report["completed_by_recommended_path"]
            or row_report["completed_by_count_fallback"]
        )
        if row_report["completed"]:
            completed_rows.append(row_report)
        else:
            pending_rows.append(row_report)

    for task, task_data in schedule.get("tasks", {}).items():
        arm_totals[task] = {}
        for arm in FORMAL_ARMS:
            rollout_dir = None
            needed = 0
            for row in rows:
                if row["task"] == task and row["arm"] == arm:
                    rollout_dir = row["rollout_dir"]
                    needed += 1
            have = hdf5_count(rollout_dir) if rollout_dir else 0
            arm_totals[task][arm] = {
                "rollout_dir": rollout_dir,
                "needed": needed,
                "have": have,
                "missing": max(0, needed - have),
                "ready": have >= needed,
            }

    next_row = pending_rows[0] if pending_rows else None
    progress = {
        "purpose": "Live collection progress for formal TacQuality paired12 rollout schedule.",
        "scientific_evidence": False,
        "git_commit": git_commit(),
        "schedule": str(args.schedule),
        "schedule_pass": schedule.get("schedule_pass"),
        "n_scheduled_rows": len(rows),
        "n_completed_rows": len(completed_rows),
        "n_pending_rows": len(pending_rows),
        "completion_fraction": float(len(completed_rows) / len(rows)) if rows else 0.0,
        "ready_for_post_collection": len(rows) > 0 and len(pending_rows) == 0,
        "next_row": next_row,
        "arm_totals": arm_totals,
        "completed_rows_tail": completed_rows[-5:],
        "pending_rows_head": pending_rows[:10],
        "post_collection_command": (
            "python TFAC_V5/run_tac_quality_post_collection_pipeline.py --tag formal_paired12"
        ),
        "guardrails": [
            "Do not skip pending schedule rows unless the run is formally marked invalid and recollected.",
            "Do not change arm order after seeing rollout outcomes.",
            "This progress report is not quality evidence; it only tracks HDF5 collection coverage.",
        ],
    }
    progress["progress_pass"] = bool(
        schedule.get("schedule_pass") is True
        and len(rows) == 72
        and set(progress["arm_totals"]) == {"insertion", "board"}
        and all(
            set(task_rows) == set(FORMAL_ARMS)
            for task_rows in progress["arm_totals"].values()
        )
        and (
            next_row is None
            or {
                "task",
                "arm",
                "pair_id",
                "rollout_dir",
                "recommended_filename",
                "recommended_path",
                "launch_command",
            }
            <= set(next_row)
        )
    )
    return progress


def write_markdown(progress: Dict[str, Any], path: Path) -> None:
    lines = [
        "# TacQuality Formal Collection Progress",
        "",
        f"- progress_pass: `{progress['progress_pass']}`",
        f"- scientific_evidence: `{progress['scientific_evidence']}`",
        f"- n_completed_rows: `{progress['n_completed_rows']}` / `{progress['n_scheduled_rows']}`",
        f"- ready_for_post_collection: `{progress['ready_for_post_collection']}`",
        "",
    ]
    next_row = progress.get("next_row")
    lines.extend(["## Next Row", ""])
    if next_row:
        lines.extend(
            [
                f"- global_step: `{next_row['global_step']}`",
                f"- task: `{next_row['task']}`",
                f"- pair_id: `{next_row['pair_id']}`",
                f"- within_pair_order: `{next_row['within_pair_order']}`",
                f"- arm: `{next_row['arm']}`",
                f"- rollout_dir: `{next_row['rollout_dir']}`",
                f"- recommended_filename: `{next_row.get('recommended_filename')}`",
                f"- recommended_path: `{next_row.get('recommended_path')}`",
                "",
                "Launch command:",
                "",
                "```bash",
                next_row["launch_command"],
                "```",
                "",
            ]
        )
    else:
        lines.append("- None. Schedule coverage is complete; run post-collection preflight.")
    lines.extend(
        [
            "",
            "## Arm Totals",
            "",
            "| task | arm | have | needed | missing | ready |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for task, task_rows in progress["arm_totals"].items():
        for arm, row in task_rows.items():
            lines.append(
                f"| {task} | {arm} | {row['have']} | {row['needed']} | {row['missing']} | {row['ready']} |"
            )
    lines.extend(["", "## Pending Head", ""])
    for row in progress["pending_rows_head"]:
        lines.append(
            f"- step {row['global_step']}: {row['task']} {row['pair_id']} "
            f"{row['within_pair_order']} {row['arm']} -> {row.get('recommended_filename')}"
        )
    lines.extend(["", "## Guardrails", ""])
    for item in progress["guardrails"]:
        lines.append(f"- {item}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", default=str(DEFAULT_SCHEDULE))
    parser.add_argument("--output_dir", default=str(OUT_DIR))
    parser.add_argument("--tag", default="formal_paired12")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    progress = build(args)
    out_dir = Path(args.output_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "tac_quality_collection_progress.json"
    md_path = out_dir / "tac_quality_collection_progress.md"
    json_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(progress, md_path)
    print(
        json.dumps(
            {
                "progress_pass": progress["progress_pass"],
                "scientific_evidence": progress["scientific_evidence"],
                "n_completed_rows": progress["n_completed_rows"],
                "n_scheduled_rows": progress["n_scheduled_rows"],
                "ready_for_post_collection": progress["ready_for_post_collection"],
                "next_row": progress["next_row"],
                "json": str(json_path),
                "markdown": str(md_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
