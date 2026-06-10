"""Build a counterbalanced formal TacQuality rollout collection schedule.

The final paired12 rollout validation compares baseline/default/distilled arms.
If all baseline trials are collected first, time drift, sensor warm-up, operator
fatigue, or board/socket condition changes can bias the gate.  This schedule
predefines a deterministic counterbalanced order for the 12 paired triplets.

It is an execution artifact only, not scientific evidence by itself.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List


OUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_collection_schedule")
DEFAULT_RUNBOOK = Path(
    "/home/chenshuai/Project/output/tac_quality_formal_rollout_runbook/"
    "formal_paired12/tac_quality_formal_rollout_runbook.json"
)
FORMAL_ARMS = ("baseline", "default_guided", "distilled_guided")
BASE_PERMUTATIONS = [
    ("baseline", "default_guided", "distilled_guided"),
    ("baseline", "distilled_guided", "default_guided"),
    ("default_guided", "baseline", "distilled_guided"),
    ("default_guided", "distilled_guided", "baseline"),
    ("distilled_guided", "baseline", "default_guided"),
    ("distilled_guided", "default_guided", "baseline"),
]


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "global_step",
        "task",
        "pair_id",
        "within_pair_order",
        "arm",
        "rollout_dir",
        "launch_command",
        "operator_note",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def task_schedule(task: str, task_row: Dict[str, Any], paired_n: int) -> Dict[str, Any]:
    by_arm = {row["arm"]: row for row in task_row["arms"]}
    triplets: List[Dict[str, Any]] = []
    long_rows: List[Dict[str, Any]] = []
    position_counts = {arm: {1: 0, 2: 0, 3: 0} for arm in FORMAL_ARMS}
    pair_order = [BASE_PERMUTATIONS[i % len(BASE_PERMUTATIONS)] for i in range(paired_n)]
    for pair_idx, arms in enumerate(pair_order, start=1):
        pair_id = f"trial_{pair_idx:03d}"
        triplet = {"pair_id": pair_id, "order": list(arms)}
        triplets.append(triplet)
        for pos, arm in enumerate(arms, start=1):
            position_counts[arm][pos] += 1
            arm_info = by_arm[arm]
            long_rows.append(
                {
                    "global_step": len(long_rows) + 1,
                    "task": task,
                    "pair_id": pair_id,
                    "within_pair_order": pos,
                    "arm": arm,
                    "rollout_dir": arm_info.get("rollout_dir"),
                    "launch_command": arm_info.get("launch_command"),
                    "operator_note": (
                        "Keep initial setup matched within this triplet; save the HDF5 into rollout_dir."
                    ),
                }
            )
    balanced = all(count == paired_n // len(FORMAL_ARMS) for arm in FORMAL_ARMS for count in position_counts[arm].values())
    return {
        "task": task,
        "paired_n_pairs": int(paired_n),
        "formal_arms": list(FORMAL_ARMS),
        "triplets": triplets,
        "rows": long_rows,
        "position_counts": position_counts,
        "counterbalance_pass": bool(balanced),
    }


def build(runbook_path: Path) -> Dict[str, Any]:
    runbook = load_json(runbook_path)
    tasks: Dict[str, Any] = {}
    all_rows: List[Dict[str, Any]] = []
    for task, task_row in runbook["tasks"].items():
        paired_n = int(task_row.get("paired_n_pairs") or 12)
        report = task_schedule(task, task_row, paired_n)
        tasks[task] = {
            key: value
            for key, value in report.items()
            if key != "rows"
        }
        all_rows.extend(report["rows"])
    schedule = {
        "name": "TacQuality formal paired12 counterbalanced collection schedule",
        "purpose": "Reduce collection-order bias before final real-rollout gates.",
        "scientific_evidence": False,
        "git_commit": git_commit(),
        "runbook": str(runbook_path),
        "task_order_policy": "collect task blocks separately; within each task follow the listed counterbalanced triplets",
        "within_triplet_policy": (
            "For a given pair_id, collect all three arms with matched initial setup before moving to the next pair_id."
        ),
        "formal_arms": list(FORMAL_ARMS),
        "tasks": tasks,
        "long_schedule_rows": all_rows,
        "cannot_count_as_completion": [
            "Schedule existence without collected HDF5 rollouts",
            "Unpaired collection that ignores pair_id matching",
            "Changing arm order after seeing rollout outcomes",
        ],
    }
    schedule["schedule_pass"] = bool(
        runbook.get("runbook_pass") is True
        and all(task["paired_n_pairs"] == 12 for task in tasks.values())
        and all(task["counterbalance_pass"] for task in tasks.values())
        and len(all_rows) == 2 * 12 * 3
    )
    return schedule


def write_markdown(schedule: Dict[str, Any], path: Path) -> None:
    lines = [
        "# TacQuality Formal Paired12 Collection Schedule",
        "",
        f"- schedule_pass: `{schedule['schedule_pass']}`",
        f"- scientific_evidence: `{schedule['scientific_evidence']}`",
        f"- task_order_policy: {schedule['task_order_policy']}",
        f"- within_triplet_policy: {schedule['within_triplet_policy']}",
        "",
        "## Per-Task Counterbalance",
        "",
    ]
    for task, row in schedule["tasks"].items():
        lines.extend(
            [
                f"### {task}",
                "",
                f"- paired_n_pairs: `{row['paired_n_pairs']}`",
                f"- counterbalance_pass: `{row['counterbalance_pass']}`",
                "",
                "| arm | pos1 | pos2 | pos3 |",
                "|---|---:|---:|---:|",
            ]
        )
        for arm, counts in row["position_counts"].items():
            lines.append(f"| {arm} | {counts[1]} | {counts[2]} | {counts[3]} |")
        lines.extend(["", "| pair_id | order |", "|---|---|"])
        for triplet in row["triplets"]:
            lines.append(f"| {triplet['pair_id']} | {' -> '.join(triplet['order'])} |")
        lines.append("")
    lines.extend(["## Guardrails", ""])
    for item in schedule["cannot_count_as_completion"]:
        lines.append(f"- {item}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runbook", default=str(DEFAULT_RUNBOOK))
    parser.add_argument("--output_dir", default=str(OUT_DIR))
    parser.add_argument("--tag", default="formal_paired12")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    schedule = build(Path(args.runbook))
    out_dir = Path(args.output_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "tac_quality_collection_schedule.json"
    md_path = out_dir / "tac_quality_collection_schedule.md"
    csv_path = out_dir / "tac_quality_collection_schedule.csv"
    json_path.write_text(json.dumps(schedule, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(schedule, md_path)
    write_csv(csv_path, schedule["long_schedule_rows"])
    print(
        json.dumps(
            {
                "schedule_pass": schedule["schedule_pass"],
                "scientific_evidence": schedule["scientific_evidence"],
                "n_rows": len(schedule["long_schedule_rows"]),
                "json": str(json_path),
                "markdown": str(md_path),
                "csv": str(csv_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
