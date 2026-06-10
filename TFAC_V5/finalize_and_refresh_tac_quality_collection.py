"""Finalize one TacQuality rollout HDF5 and refresh the next collection gate.

This is the operator-safe wrapper to run immediately after a robot/client HDF5
has been collected.  It calls ``finalize_tac_quality_collected_hdf5.py`` and,
only after a successful finalize, regenerates the artifacts needed for the next
scheduled row:

  - collection progress;
  - next collection step;
  - next-step dry-run smoke;
  - current collection gate;
  - current collection handoff;
  - deployment manifest;
  - goal audit.

It does not run real rollout quality gates and does not create scientific
evidence by itself.
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

from TFAC_V5.finalize_tac_quality_collected_hdf5 import build as build_finalize  # noqa: E402
from TFAC_V5.finalize_tac_quality_collected_hdf5 import write_markdown as write_finalize_markdown  # noqa: E402


OUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_finalize_and_refresh")
DEFAULT_NEXT_STEP = Path(
    "/home/chenshuai/Project/output/tac_quality_next_collection_step/"
    "formal_paired12/tac_quality_next_collection_step.json"
)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def run_cmd(cmd: List[str]) -> Dict[str, Any]:
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
    return {
        "command": " ".join(cmd),
        "returncode": proc.returncode,
        "passed": proc.returncode == 0,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def refresh_commands(tag: str) -> List[List[str]]:
    return [
        [sys.executable, "TFAC_V5/build_tac_quality_collection_progress.py", "--tag", tag],
        [sys.executable, "TFAC_V5/build_tac_quality_next_collection_step.py", "--tag", tag],
        [sys.executable, "TFAC_V5/run_tac_quality_next_collection_step_smoke.py", "--tag", tag],
        [sys.executable, "TFAC_V5/build_tac_quality_current_collection_gate.py", "--tag", tag],
        [sys.executable, "TFAC_V5/build_tac_quality_current_collection_handoff.py", "--tag", tag],
        [sys.executable, "TFAC_V5/build_tac_quality_guidance_manifest.py"],
        [sys.executable, "TFAC_V5/audit_tac_quality_goal_completion.py"],
    ]


def build(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.output_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    finalize_args = argparse.Namespace(
        next_step=args.next_step,
        source=args.source,
        source_dir=args.source_dir,
        output_dir=str(out_dir),
        tag="finalize",
        move=args.move,
        overwrite=args.overwrite,
        allow_bad_schema=args.allow_bad_schema,
        dry_run=args.dry_run,
        min_steps=args.min_steps,
        success=getattr(args, "success", None),
        stopped_early=getattr(args, "stopped_early", None),
        scorer_freeze_manifest=getattr(args, "scorer_freeze_manifest", None),
    )
    finalize = build_finalize(finalize_args)
    finalize_dir = out_dir / "finalize"
    finalize_dir.mkdir(parents=True, exist_ok=True)
    finalize_json = finalize_dir / "tac_quality_finalize_collected_hdf5.json"
    finalize_md = finalize_dir / "tac_quality_finalize_collected_hdf5.md"
    finalize_json.write_text(json.dumps(finalize, ensure_ascii=False, indent=2), encoding="utf-8")
    write_finalize_markdown(finalize, finalize_md)

    command_results: List[Dict[str, Any]] = []
    if finalize.get("finalize_pass") and not args.skip_refresh:
        for cmd in refresh_commands(args.tag):
            command_results.append(run_cmd(cmd))

    progress_path = (
        Path("/home/chenshuai/Project/output/tac_quality_collection_progress")
        / args.tag
        / "tac_quality_collection_progress.json"
    )
    next_step_path = (
        Path("/home/chenshuai/Project/output/tac_quality_next_collection_step")
        / args.tag
        / "tac_quality_next_collection_step.json"
    )
    gate_path = (
        Path("/home/chenshuai/Project/output/tac_quality_current_collection_gate")
        / args.tag
        / "tac_quality_current_collection_gate.json"
    )
    handoff_path = (
        Path("/home/chenshuai/Project/output/tac_quality_current_collection_handoff")
        / args.tag
        / "tac_quality_current_collection_handoff.json"
    )
    manifest_path = Path("/home/chenshuai/Project/output/tac_quality_guidance_manifest/tac_quality_guidance_manifest.json")
    goal_audit_path = Path("/home/chenshuai/Project/output/tac_quality_goal_audit/tac_quality_goal_completion_audit.json")

    progress = load_json(progress_path)
    next_step = load_json(next_step_path)
    gate = load_json(gate_path)
    handoff = load_json(handoff_path)
    manifest = load_json(manifest_path)
    goal_audit = load_json(goal_audit_path)

    refresh_pass = bool(command_results and all(row["passed"] for row in command_results))
    if args.skip_refresh:
        refresh_pass = False

    result = {
        "purpose": "Finalize a collected TacQuality HDF5 and refresh next-row collection artifacts.",
        "scientific_evidence": False,
        "git_commit": git_commit(),
        "dry_run": bool(args.dry_run),
        "skip_refresh": bool(args.skip_refresh),
        "finalize": {
            "json": str(finalize_json),
            "markdown": str(finalize_md),
            "finalize_pass": finalize.get("finalize_pass"),
            "operation": finalize.get("operation"),
            "source": finalize.get("source"),
            "target": finalize.get("target"),
            "refusal_reason": finalize.get("refusal_reason"),
            "explicit_outcome_attrs": finalize.get("explicit_outcome_attrs"),
            "scorer_freeze_attrs": finalize.get("scorer_freeze_attrs"),
            "next_row": finalize.get("next_row"),
        },
        "refresh_commands": command_results,
        "refreshed_artifacts": {
            "progress": str(progress_path),
            "next_step": str(next_step_path),
            "gate": str(gate_path),
            "handoff": str(handoff_path),
            "manifest": str(manifest_path),
            "goal_audit": str(goal_audit_path),
        },
        "post_refresh_status": {
            "progress_pass": progress.get("progress_pass"),
            "completed": progress.get("n_completed_rows"),
            "scheduled": progress.get("n_scheduled_rows"),
            "ready_for_post_collection": progress.get("ready_for_post_collection"),
            "next_task": (next_step.get("next_row") or {}).get("task"),
            "next_pair_id": (next_step.get("next_row") or {}).get("pair_id"),
            "next_arm": (next_step.get("next_row") or {}).get("arm"),
            "current_collection_gate_pass": gate.get("current_collection_gate_pass"),
            "operator_go_no_go": gate.get("operator_go_no_go"),
            "handoff_pass": handoff.get("handoff_pass"),
            "deployment_manifest_pass": manifest.get("deployment_manifest_pass"),
            "objective_complete": goal_audit.get("objective_complete"),
            "n_blockers": len(goal_audit.get("blockers", []) or []),
        },
        "next_required_step": (
            "Review the refreshed handoff and collect the next scheduled row."
            if finalize.get("finalize_pass")
            and refresh_pass
            and progress.get("ready_for_post_collection") is False
            else (
                "All scheduled rows appear covered; run the post-collection pipeline preflight."
                if finalize.get("finalize_pass")
                and refresh_pass
                and progress.get("ready_for_post_collection") is True
                else "Fix finalize/refresh failure before collecting another rollout."
            )
        ),
        "guardrails": [
            "This wrapper does not run formal quality gates and is not scientific evidence.",
            "It refreshes the next-row gate after file finalization so the operator does not reuse stale handoff artifacts.",
            "Do not use --overwrite or --move unless the collected file has been manually checked.",
        ],
    }
    result["finalize_and_refresh_pass"] = bool(
        finalize.get("finalize_pass")
        and (args.skip_refresh or refresh_pass)
        and result["scientific_evidence"] is False
        and (
            args.skip_refresh
            or (
                manifest.get("deployment_manifest_pass") is True
                and goal_audit.get("objective_complete") is False
                and handoff.get("handoff_pass") in {True, None}
            )
        )
    )
    return result


def write_markdown(result: Dict[str, Any], path: Path) -> None:
    lines = [
        "# TacQuality Finalize And Refresh",
        "",
        f"- finalize_and_refresh_pass: `{result['finalize_and_refresh_pass']}`",
        f"- scientific_evidence: `{result['scientific_evidence']}`",
        f"- dry_run: `{result['dry_run']}`",
        f"- finalize_pass: `{result['finalize']['finalize_pass']}`",
        f"- operation: `{result['finalize']['operation']}`",
        f"- explicit_outcome_attrs: `{result['finalize']['explicit_outcome_attrs']}`",
        f"- scorer_freeze_attrs: `{result['finalize']['scorer_freeze_attrs']}`",
        f"- source: `{result['finalize']['source']}`",
        f"- target: `{result['finalize']['target']}`",
        f"- next_required_step: {result['next_required_step']}",
        "",
        "## Post Refresh Status",
        "",
    ]
    for key, value in result["post_refresh_status"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Refresh Commands", "", "| passed | command |", "|---:|---|"])
    for row in result["refresh_commands"]:
        lines.append(f"| {row['passed']} | `{row['command']}` |")
    lines.extend(["", "## Guardrails", ""])
    for item in result["guardrails"]:
        lines.append(f"- {item}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--next_step", default=str(DEFAULT_NEXT_STEP))
    parser.add_argument("--source", default=None)
    parser.add_argument("--source_dir", default=None)
    parser.add_argument("--output_dir", default=str(OUT_DIR))
    parser.add_argument("--tag", default="formal_paired12")
    parser.add_argument("--move", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow_bad_schema", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_refresh", action="store_true")
    parser.add_argument("--min_steps", type=int, default=3)
    parser.add_argument("--success", default=None, help="Optional explicit rollout success attr: true/false.")
    parser.add_argument("--stopped_early", default=None, help="Optional explicit stopped_early attr: true/false.")
    parser.add_argument(
        "--scorer_freeze_manifest",
        default="/home/chenshuai/Project/output/tac_quality_scorer_freeze_manifest/tac_quality_scorer_freeze_manifest.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build(args)
    out_dir = Path(args.output_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "tac_quality_finalize_and_refresh.json"
    md_path = out_dir / "tac_quality_finalize_and_refresh.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    print(
        json.dumps(
            {
                "finalize_and_refresh_pass": result["finalize_and_refresh_pass"],
                "scientific_evidence": result["scientific_evidence"],
                "finalize_pass": result["finalize"]["finalize_pass"],
                "refresh_commands_passed": all(row["passed"] for row in result["refresh_commands"])
                if result["refresh_commands"]
                else None,
                "next_required_step": result["next_required_step"],
                "json": str(json_path),
                "markdown": str(md_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
