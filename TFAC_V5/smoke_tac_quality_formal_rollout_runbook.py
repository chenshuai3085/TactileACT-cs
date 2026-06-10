"""Smoke-test the formal TacQuality rollout runbook.

The launch-sheet smoke executes the server dry-runs.  This lighter smoke checks
that the runbook itself is a complete and consistent operator handoff: formal
arms, directories, commands, post-collection gate commands, guardrails, and the
existing launch-sheet smoke result all line up.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_RUNBOOK = Path(
    "/home/chenshuai/Project/output/tac_quality_formal_rollout_runbook/"
    "formal_paired12/tac_quality_formal_rollout_runbook.json"
)
DEFAULT_LAUNCH_SMOKE = Path(
    "/home/chenshuai/Project/output/tac_quality_formal_launch_sheet_smoke/"
    "formal_paired12/tac_quality_formal_launch_sheet_smoke.json"
)
OUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_formal_rollout_runbook_smoke")


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


def command_has(cmd: Optional[str], *tokens: str) -> bool:
    text = cmd or ""
    return all(token in text for token in tokens)


def task_checks(task: str, task_row: Dict[str, Any]) -> Dict[str, Any]:
    arms = task_row.get("arms", [])
    by_arm = {row.get("arm"): row for row in arms}
    formal_arms = ["baseline", "default_guided", "distilled_guided"]
    arm_checks: List[Dict[str, Any]] = []
    for arm in formal_arms:
        row = by_arm.get(arm, {})
        cmd = row.get("launch_command")
        arm_checks.append(
            {
                "arm": arm,
                "has_rollout_dir": bool(row.get("rollout_dir")),
                "has_launch_command": bool(cmd),
                "needed_hdf5_ok": (row.get("needed_hdf5") or 0) >= 10,
                "baseline_disables_guidance": command_has(cmd, "--disable_guidance") if arm == "baseline" else True,
                "guided_enabled": "--disable_guidance" not in (cmd or "") if arm != "baseline" else True,
                "uses_guided_server": command_has(cmd, "for_show_xiaomi.serve_dp_tac_quality_guided"),
                "mentions_arm": arm in (cmd or ""),
            }
        )
    two_arm_cmd = get(task_row, "two_arm_gate.command")
    three_arm_cmd = get(task_row, "three_arm_ablation_gate.command")
    checks = {
        "paired_n_pairs_ok": (task_row.get("paired_n_pairs") or 0) >= 10,
        "formal_arms_present": sorted(by_arm) == formal_arms,
        "all_arm_fields_present": all(
            all(row.get(key) is True for key in ["has_rollout_dir", "has_launch_command", "needed_hdf5_ok"])
            for row in arm_checks
        ),
        "baseline_disables_guidance": all(row["baseline_disables_guidance"] for row in arm_checks),
        "guided_arms_enabled": all(row["guided_enabled"] for row in arm_checks),
        "all_use_guided_server": all(row["uses_guided_server"] for row in arm_checks),
        "all_commands_mention_arm": all(row["mentions_arm"] for row in arm_checks),
        "two_arm_gate_command_present": command_has(two_arm_cmd, "eval_real_rollout_quality_gate.py"),
        "three_arm_gate_command_present": command_has(three_arm_cmd, "eval_real_rollout_scorer_ablation_gate.py"),
    }
    return {
        "task": task,
        "arm_checks": arm_checks,
        "checks": checks,
        "task_pass": all(checks.values()),
    }


def build(args: argparse.Namespace) -> Dict[str, Any]:
    runbook = load_json(Path(args.runbook)) or {}
    launch_smoke = load_json(Path(args.launch_smoke)) or {}
    task_reports = {
        task: task_checks(task, row)
        for task, row in (runbook.get("tasks") or {}).items()
    }
    checks = {
        "runbook_pass": get(runbook, "runbook_pass") is True,
        "scientific_evidence_false": get(runbook, "scientific_evidence") is False,
        "ready_for_gate_runner_false_until_collection": get(runbook, "ready_for_gate_runner") is False,
        "two_tasks_present": sorted(task_reports) == ["board", "insertion"],
        "all_tasks_pass": all(row["task_pass"] for row in task_reports.values()),
        "completion_blockers_four": len(runbook.get("completion_blockers_to_close") or []) == 4,
        "cannot_count_guardrails_present": all(
            token in str(runbook.get("cannot_count_as_completion") or [])
            for token in ["synthetic HDF5 smoke outputs", "frame-level random cross validation"]
        ),
        "post_collection_preflight_command_present": command_has(
            get(runbook, "post_collection_commands.pipeline_preflight"),
            "run_tac_quality_post_collection_pipeline.py",
        ),
        "post_collection_run_gates_command_present": command_has(
            get(runbook, "post_collection_commands.pipeline_run_gates"),
            "run_tac_quality_post_collection_pipeline.py",
            "--run_gates",
        ),
        "goal_audit_command_present": command_has(
            get(runbook, "post_collection_commands.goal_audit"),
            "audit_tac_quality_goal_completion.py",
        ),
        "launch_sheet_smoke_pass": get(launch_smoke, "overall_pass") is True,
        "launch_sheet_smoke_scientific_evidence_false": get(launch_smoke, "scientific_evidence") is False,
        "launch_sheet_smoke_all_commands_present": get(launch_smoke, "checks.all_commands_present") is True,
    }
    result = {
        "purpose": "Structure smoke for the formal TacQuality rollout runbook.",
        "scientific_evidence": False,
        "git_commit": git_commit(),
        "runbook": str(args.runbook),
        "launch_smoke": str(args.launch_smoke),
        "checks": checks,
        "tasks": task_reports,
    }
    result["overall_pass"] = all(checks.values())
    return result


def write_markdown(result: Dict[str, Any], path: Path) -> None:
    lines = [
        "# TacQuality Formal Rollout Runbook Smoke",
        "",
        f"- overall_pass: `{result['overall_pass']}`",
        f"- scientific_evidence: `{result['scientific_evidence']}`",
        f"- git_commit: `{result['git_commit']}`",
        "",
        "## Checks",
        "",
        "| check | pass |",
        "|---|---:|",
    ]
    for name, passed in result["checks"].items():
        lines.append(f"| {name} | {passed} |")
    lines.extend(["", "## Tasks", ""])
    for task, row in result["tasks"].items():
        lines.extend([f"### {task}", "", f"- task_pass: `{row['task_pass']}`", ""])
        lines.extend(["| arm | rollout_dir | command | needed | guidance |", "|---|---:|---:|---:|---:|"])
        for arm in row["arm_checks"]:
            lines.append(
                f"| {arm['arm']} | {arm['has_rollout_dir']} | {arm['has_launch_command']} | "
                f"{arm['needed_hdf5_ok']} | {arm['baseline_disables_guidance'] and arm['guided_enabled']} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runbook", default=str(DEFAULT_RUNBOOK))
    parser.add_argument("--launch_smoke", default=str(DEFAULT_LAUNCH_SMOKE))
    parser.add_argument("--output_dir", default=str(OUT_DIR))
    parser.add_argument("--tag", default="formal_paired12")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    result = build(args)
    json_path = out_dir / "tac_quality_formal_rollout_runbook_smoke.json"
    md_path = out_dir / "tac_quality_formal_rollout_runbook_smoke.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    print(
        json.dumps(
            {
                "overall_pass": result["overall_pass"],
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
