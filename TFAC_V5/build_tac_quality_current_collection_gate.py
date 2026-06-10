"""Build a go/no-go gate for the current formal TacQuality collection row.

This is the final operator-facing pre-robot check for the current scheduled
rollout.  It combines collection progress, next-step handoff, and the executed
dry-run smoke runner.  Passing this gate means the next row is ready to collect;
it does not mean the scorer/guidance objective is scientifically validated.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional


OUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_current_collection_gate")
DEFAULT_PROGRESS = Path(
    "/home/chenshuai/Project/output/tac_quality_collection_progress/"
    "formal_paired12/tac_quality_collection_progress.json"
)
DEFAULT_NEXT_STEP = Path(
    "/home/chenshuai/Project/output/tac_quality_next_collection_step/"
    "formal_paired12/tac_quality_next_collection_step.json"
)
DEFAULT_RUNNER = Path(
    "/home/chenshuai/Project/output/tac_quality_next_collection_step_smoke/"
    "formal_paired12/tac_quality_next_collection_step_smoke_runner.json"
)


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


def dir_writable(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    return os.access(path, os.W_OK)


def same_current_row(progress: Dict[str, Any], next_step: Dict[str, Any], runner: Dict[str, Any]) -> bool:
    keys = ["task", "pair_id", "arm", "recommended_path", "global_step", "within_pair_order"]
    return all(
        get(progress, f"next_row.{key}") == get(next_step, f"next_row.{key}") == get(runner, f"next_row.{key}")
        for key in keys
    )


def build(args: argparse.Namespace) -> Dict[str, Any]:
    progress = load_json(Path(args.progress)) or {}
    next_step = load_json(Path(args.next_step)) or {}
    runner = load_json(Path(args.runner)) or {}
    row = next_step.get("next_row") or {}
    recommended_path = Path(str(next_step.get("recommended_path") or row.get("recommended_path") or ""))
    rollout_dir = Path(str(row.get("rollout_dir") or ""))
    launch_command = str(next_step.get("launch_command") or row.get("launch_command") or "")
    arm = str(row.get("arm") or "")
    baseline = arm == "baseline"
    checks = {
        "progress_pass": progress.get("progress_pass") is True,
        "next_step_pass": next_step.get("next_step_pass") is True,
        "runner_pass": runner.get("overall_pass") is True,
        "runner_process_pass": get(runner, "process.passed_process") is True
        and get(runner, "process.returncode") == 0,
        "runner_smoke_audit_pass": get(runner, "smoke_audit.overall_pass") is True,
        "has_next_step": next_step.get("has_next_step") is True,
        "same_current_row": same_current_row(progress, next_step, runner),
        "recommended_path_declared": str(recommended_path).endswith(".hdf5"),
        "recommended_path_not_exists": not recommended_path.exists(),
        "rollout_dir_exists": rollout_dir.exists() and rollout_dir.is_dir(),
        "rollout_dir_writable": dir_writable(rollout_dir),
        "launch_uses_guided_server": "serve_dp_tac_quality_guided" in launch_command,
        "baseline_guidance_disabled": ("--disable_guidance" in launch_command) if baseline else True,
        "guided_guidance_enabled": ("--disable_guidance" not in launch_command) if not baseline else True,
        "dry_run_output_matches_runner": next_step.get("pre_collection_dry_run_output")
        == runner.get("pre_collection_dry_run_output"),
        "not_ready_for_post_collection": progress.get("ready_for_post_collection") is False,
        "scheduled_count_positive": (progress.get("n_scheduled_rows", 0) or 0) > 0,
    }
    result = {
        "purpose": "Go/no-go gate for collecting the current formal TacQuality rollout row.",
        "scientific_evidence": False,
        "git_commit": git_commit(),
        "progress": str(args.progress),
        "next_step": str(args.next_step),
        "runner": str(args.runner),
        "current_row": row,
        "recommended_path": str(recommended_path),
        "rollout_dir": str(rollout_dir),
        "operator_go_no_go": "go" if all(checks.values()) else "no_go",
        "checks": checks,
        "next_actions": [
            "Run the launch_command_with_save_path_hint and collect the robot/client HDF5.",
            "Finalize the raw HDF5 to recommended_path.",
            "Rerun collection progress, next-step, runner, and this gate before the next row.",
        ],
        "guardrails": [
            "This gate is only a pre-collection execution check, not policy quality evidence.",
            "Do not collect if recommended_path already exists.",
            "Do not skip counterbalanced schedule order after seeing outcomes.",
        ],
    }
    result["current_collection_gate_pass"] = all(checks.values())
    return result


def write_markdown(result: Dict[str, Any], path: Path) -> None:
    row = result["current_row"] or {}
    lines = [
        "# TacQuality Current Collection Gate",
        "",
        f"- current_collection_gate_pass: `{result['current_collection_gate_pass']}`",
        f"- operator_go_no_go: `{result['operator_go_no_go']}`",
        f"- scientific_evidence: `{result['scientific_evidence']}`",
        f"- task: `{row.get('task')}`",
        f"- pair_id: `{row.get('pair_id')}`",
        f"- arm: `{row.get('arm')}`",
        f"- recommended_path: `{result['recommended_path']}`",
        f"- rollout_dir: `{result['rollout_dir']}`",
        "",
        "## Checks",
        "",
        "| check | pass |",
        "|---|---:|",
    ]
    for name, passed in result["checks"].items():
        lines.append(f"| {name} | {passed} |")
    lines.extend(["", "## Next Actions", ""])
    for action in result["next_actions"]:
        lines.append(f"- {action}")
    lines.extend(["", "## Guardrails", ""])
    for item in result["guardrails"]:
        lines.append(f"- {item}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress", default=str(DEFAULT_PROGRESS))
    parser.add_argument("--next_step", default=str(DEFAULT_NEXT_STEP))
    parser.add_argument("--runner", default=str(DEFAULT_RUNNER))
    parser.add_argument("--output_dir", default=str(OUT_DIR))
    parser.add_argument("--tag", default="formal_paired12")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    result = build(args)
    json_path = out_dir / "tac_quality_current_collection_gate.json"
    md_path = out_dir / "tac_quality_current_collection_gate.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    print(
        json.dumps(
            {
                "current_collection_gate_pass": result["current_collection_gate_pass"],
                "operator_go_no_go": result["operator_go_no_go"],
                "scientific_evidence": result["scientific_evidence"],
                "recommended_path": result["recommended_path"],
                "json": str(json_path),
                "markdown": str(md_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
