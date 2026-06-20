#!/usr/bin/env python3
"""Refresh the current TacQuality evidence bundle.

This script is intentionally a bookkeeping wrapper.  It reruns the existing
offline scorer, guidance-state, rollout-coverage, server-log-schema, and
scorecard audits, then writes one compact JSON/Markdown bundle that is easy to
check before and after real robot trials.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_evidence_bundle")
DEFAULT_TAG = "current_tac_quality_evidence_bundle"

DEFAULT_SCORECARD = Path(
    "/home/chenshuai/Project/output/tac_quality_current_scorecard/"
    "current_tac_quality_scorecard.json"
)
DEFAULT_SCORER_AUDIT = Path(
    "/home/chenshuai/Project/output/tac_quality_current_scorer_audit/"
    "current_tac_quality_scorer_audit.json"
)
DEFAULT_GUIDANCE_STATE = Path(
    "/home/chenshuai/Project/output/tac_quality_guidance_state_audit/"
    "tac_quality_guidance_state_audit.json"
)
DEFAULT_COVERAGE = Path(
    "/home/chenshuai/Project/output/tac_quality_real_rollout_coverage/"
    "current_s12_good_margin_coverage/tac_quality_real_rollout_coverage.json"
)
DEFAULT_SCHEMA_AUDIT = Path(
    "/home/chenshuai/Project/output/tac_quality_server_rollout_schema_audit/"
    "current_schema_smoke/tac_quality_server_rollout_schema_audit.json"
)
DEFAULT_DP_STATUS = Path(
    "/media/chenshuai/EXTERNAL_USB/pih_output/"
    "dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_"
    "20260619_stable_fullwindow_slowlr/training_status_latest.json"
)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"_missing": True, "_path": str(path)}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_error": repr(exc), "_path": str(path)}
    if isinstance(data, dict):
        data.setdefault("_source_path", str(path))
        return data
    return {"_error": "JSON root is not an object", "_path": str(path)}


def get(data: Mapping[str, Any] | None, dotted: str, default: Any = None) -> Any:
    cur: Any = data
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return default
        cur = cur[part]
    return cur


def boolish(value: Any) -> bool:
    return bool(value is True or str(value).lower() == "true")


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def run_step(name: str, command: list[str], cwd: Path, dry_run: bool) -> dict[str, Any]:
    record: dict[str, Any] = {
        "name": name,
        "command": command,
        "cwd": str(cwd),
        "dry_run": dry_run,
    }
    print(f"\n[{name}] {' '.join(command)}", flush=True)
    if dry_run:
        record.update({"returncode": 0, "stdout_tail": "", "stderr_tail": ""})
        return record

    proc = subprocess.run(command, cwd=str(cwd), text=True, capture_output=True)
    record["returncode"] = int(proc.returncode)
    record["stdout_tail"] = proc.stdout[-6000:]
    record["stderr_tail"] = proc.stderr[-6000:]
    if proc.stdout:
        print(proc.stdout[-3000:], end="" if proc.stdout.endswith("\n") else "\n")
    if proc.stderr:
        print(proc.stderr[-3000:], end="" if proc.stderr.endswith("\n") else "\n", file=sys.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"Step failed: {name}, returncode={proc.returncode}")
    return record


def build_commands(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    python_cmd = args.python_cmd
    return [
        (
            "current_scorer_audit",
            python_cmd + ["TFAC_V5/tac_quality_energy/audit_current_tac_quality_scorers.py"],
        ),
        (
            "guidance_state_audit",
            python_cmd + ["for_show_xiaomi/audit_tac_quality_guidance_state.py"],
        ),
        (
            "real_rollout_coverage_audit",
            python_cmd + ["for_show_xiaomi/audit_real_rollout_coverage.py"],
        ),
        (
            "server_rollout_schema_audit",
            python_cmd + [
                "for_show_xiaomi/audit_server_rollout_schema.py",
                "--tag",
                args.schema_tag,
            ],
        ),
        (
            "current_scorecard",
            python_cmd + ["TFAC_V5/tac_quality_energy/build_current_scorecard.py"],
        ),
    ]


def build_bundle(args: argparse.Namespace, steps: list[dict[str, Any]]) -> dict[str, Any]:
    scorecard = load_json(args.scorecard)
    scorer_audit = load_json(args.scorer_audit)
    guidance_state = load_json(args.guidance_state)
    coverage = load_json(args.coverage)
    schema_audit = load_json(args.schema_audit)
    dp_status = load_json(args.dp_status)

    levels = get(scorecard, "evidence_levels", {})
    board = get(scorecard, "task_scorecards.board", {})
    insertion = get(scorecard, "task_scorecards.insertion", {})
    coverage_summary = get(coverage, "summary", {})
    step_failures = [step for step in steps if int(step.get("returncode", 1)) != 0]

    bundle = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "tag": args.tag,
        "repo_root": str(REPO_ROOT),
        "refresh_steps": steps,
        "refresh_pass": not step_failures,
        "step_failures": step_failures,
        "evidence_levels": {
            "offline_scorer_ready": boolish(get(levels, "offline_scorer_ready", False)),
            "gradient_guidance_ready": boolish(get(levels, "gradient_guidance_ready", False)),
            "server_rollout_schema_ready": boolish(get(levels, "server_rollout_schema_ready", False)),
            "real_evidence_pipeline_ready": boolish(get(levels, "real_evidence_pipeline_ready", False)),
            "real_paired_rollout_complete": boolish(get(levels, "real_paired_rollout_complete", False)),
            "goal_complete": boolish(get(levels, "goal_complete", False)),
        },
        "recommended_runtime": {
            "board": {
                "arm": get(board, "arm"),
                "runtime": get(board, "runtime"),
                "score_mode": get(board, "score_mode"),
                "dp_ckpt": get(board, "dp_checkpoint_policy.recommended_ckpt"),
                "avoid_ckpt": get(board, "dp_checkpoint_policy.avoid_default_ckpt"),
            },
            "insertion": {
                "arm": get(insertion, "arm"),
                "runtime": get(insertion, "runtime"),
                "score_mode": get(insertion, "score_mode"),
            },
        },
        "key_metrics": {
            "board": {
                "auc": get(board, "offline_metrics.binary_auc"),
                "balanced_accuracy": get(board, "offline_metrics.binary_balanced_accuracy"),
                "quality_spearman": get(board, "offline_metrics.quality_spearman"),
                "pred_gt_spearman": get(board, "foresight_alignment.pred_gt_spearman"),
                "guidance_improved_rate": get(board, "guidance_metrics.improved_rate"),
                "best_val_epoch": get(board, "dp_checkpoint_policy.best_val_epoch"),
                "best_val_loss": get(board, "dp_checkpoint_policy.best_val_loss"),
                "final_epoch": get(board, "dp_checkpoint_policy.final_epoch"),
                "final_val_loss": get(board, "dp_checkpoint_policy.final_val_loss"),
            },
            "insertion": {
                "auc": get(insertion, "offline_metrics.binary_auc"),
                "balanced_accuracy": get(insertion, "offline_metrics.binary_balanced_accuracy"),
                "quality_corr": get(insertion, "offline_metrics.quality_corr"),
                "good_margin_improve_rate": get(insertion, "guidance_metrics.good_margin_improve_rate"),
            },
        },
        "real_rollout_coverage": {
            "status_counts": get(coverage_summary, "status_counts", {}),
            "planned_counts": get(coverage_summary, "planned_counts", {}),
            "observed_counts": get(coverage_summary, "observed_counts", {}),
            "pair_summary": get(coverage_summary, "pair_summary", {}),
            "real_rollout_evidence_complete": boolish(
                get(coverage_summary, "real_rollout_evidence_complete", False)
            ),
        },
        "server_rollout_schema": {
            "schema_pass": boolish(get(schema_audit, "summary.schema_pass", False)),
            "synthetic_count": get(schema_audit, "summary.synthetic_count"),
            "real_count": get(schema_audit, "summary.real_count"),
            "failed_trials": get(schema_audit, "summary.failed_trials", []),
        },
        "dp_training_status": {
            "best_metric_name": get(dp_status, "best_metric_name"),
            "best_metric": get(dp_status, "best_metric"),
            "latest": get(dp_status, "latest"),
            "best_val_epoch": get(dp_status, "best_val_epoch"),
        },
        "evidence_boundary": {
            "can_claim_now": get(scorecard, "evidence_boundary.can_claim_now", []),
            "cannot_claim_yet": get(scorecard, "evidence_boundary.cannot_claim_yet", []),
        },
        "input_paths": {
            "scorecard": str(args.scorecard),
            "scorer_audit": str(args.scorer_audit),
            "guidance_state": str(args.guidance_state),
            "coverage": str(args.coverage),
            "schema_audit": str(args.schema_audit),
            "dp_status": str(args.dp_status),
        },
        "source_snapshots": {
            "scorecard_created_at": get(scorecard, "created_at"),
            "scorer_audit_created_at": get(scorer_audit, "created_at"),
            "guidance_state_created_at": get(guidance_state, "created_at"),
            "coverage_created_at": get(coverage, "created_at"),
            "schema_audit_created_at": get(schema_audit, "created_at"),
        },
    }
    return bundle


def write_markdown(bundle: Mapping[str, Any], path: Path) -> None:
    levels = get(bundle, "evidence_levels", {})
    board = get(bundle, "recommended_runtime.board", {})
    insertion = get(bundle, "recommended_runtime.insertion", {})
    board_metrics = get(bundle, "key_metrics.board", {})
    insertion_metrics = get(bundle, "key_metrics.insertion", {})
    coverage = get(bundle, "real_rollout_coverage", {})
    schema = get(bundle, "server_rollout_schema", {})

    lines = [
        "# TacQuality Evidence Bundle",
        "",
        f"- generated: `{bundle['created_at']}`",
        f"- tag: `{bundle['tag']}`",
        f"- refresh_pass: `{bundle['refresh_pass']}`",
        "",
        "## Evidence Levels",
        "",
        "| level | status |",
        "|---|---:|",
    ]
    for key, value in levels.items():
        lines.append(f"| `{key}` | `{value}` |")

    lines.extend([
        "",
        "## Current Runtime Choices",
        "",
        "| task | arm | runtime | score mode | checkpoint note |",
        "|---|---|---|---|---|",
        (
            f"| board | `{board.get('arm')}` | `{board.get('runtime')}` | "
            f"`{board.get('score_mode')}` | use `{board.get('dp_ckpt')}` |"
        ),
        (
            f"| insertion | `{insertion.get('arm')}` | `{insertion.get('runtime')}` | "
            f"`{insertion.get('score_mode')}` | existing insertion DP chain |"
        ),
        "",
        "## Key Metrics",
        "",
        "| task | offline classifier | quality metric | guidance/foresight metric |",
        "|---|---|---|---|",
        (
            "| board | "
            f"AUC `{fmt(board_metrics.get('auc'))}`, bACC `{fmt(board_metrics.get('balanced_accuracy'))}` | "
            f"rho `{fmt(board_metrics.get('quality_spearman'))}` | "
            f"pred-GT rho `{fmt(board_metrics.get('pred_gt_spearman'))}`, "
            f"guidance improve `{fmt(board_metrics.get('guidance_improved_rate'))}` |"
        ),
        (
            "| insertion | "
            f"AUC `{fmt(insertion_metrics.get('auc'))}`, bACC `{fmt(insertion_metrics.get('balanced_accuracy'))}` | "
            f"corr `{fmt(insertion_metrics.get('quality_corr'))}` | "
            f"good-margin improve `{fmt(insertion_metrics.get('good_margin_improve_rate'))}` |"
        ),
        "",
        "## 260617 DP Checkpoint Policy",
        "",
        f"- best val epoch/loss: `{board_metrics.get('best_val_epoch')}` / `{fmt(board_metrics.get('best_val_loss'), 6)}`",
        f"- final epoch/val loss: `{board_metrics.get('final_epoch')}` / `{fmt(board_metrics.get('final_val_loss'), 6)}`",
        f"- default rollout ckpt: `{board.get('dp_ckpt')}`",
        f"- avoid as default: `{board.get('avoid_ckpt')}`",
        "",
        "## Real Rollout Coverage",
        "",
        f"- status_counts: `{json.dumps(coverage.get('status_counts'), ensure_ascii=False)}`",
        f"- planned_counts: `{json.dumps(coverage.get('planned_counts'), ensure_ascii=False)}`",
        f"- observed_counts: `{json.dumps(coverage.get('observed_counts'), ensure_ascii=False)}`",
        f"- real_rollout_evidence_complete: `{coverage.get('real_rollout_evidence_complete')}`",
        "",
        "## Server Rollout Schema",
        "",
        f"- schema_pass: `{schema.get('schema_pass')}`",
        f"- synthetic_count: `{schema.get('synthetic_count')}`",
        f"- real_count: `{schema.get('real_count')}`",
        f"- failed_trials: `{json.dumps(schema.get('failed_trials'), ensure_ascii=False)}`",
        "",
        "## Evidence Boundary",
        "",
        "Can claim now:",
    ])
    for item in get(bundle, "evidence_boundary.can_claim_now", []):
        lines.append(f"- {item}")
    lines.extend(["", "Cannot claim yet:"])
    for item in get(bundle, "evidence_boundary.cannot_claim_yet", []):
        lines.append(f"- {item}")

    lines.extend(["", "## Refreshed Steps", ""])
    for step in bundle["refresh_steps"]:
        lines.append(f"- `{step['name']}`: returncode `{step['returncode']}`")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--schema_tag", default="current_schema_smoke")
    parser.add_argument("--scorecard", type=Path, default=DEFAULT_SCORECARD)
    parser.add_argument("--scorer_audit", type=Path, default=DEFAULT_SCORER_AUDIT)
    parser.add_argument("--guidance_state", type=Path, default=DEFAULT_GUIDANCE_STATE)
    parser.add_argument("--coverage", type=Path, default=DEFAULT_COVERAGE)
    parser.add_argument("--schema_audit", type=Path, default=DEFAULT_SCHEMA_AUDIT)
    parser.add_argument("--dp_status", type=Path, default=DEFAULT_DP_STATUS)
    parser.add_argument(
        "--python_cmd",
        nargs="+",
        default=["conda", "run", "--no-capture-output", "-n", "TactileACT", "python"],
        help="Python command prefix used to run audit scripts.",
    )
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    steps: list[dict[str, Any]] = []
    for name, command in build_commands(args):
        steps.append(run_step(name, command, REPO_ROOT, args.dry_run))

    bundle = build_bundle(args, steps)
    json_path = args.output_dir / f"{args.tag}.json"
    md_path = args.output_dir / f"{args.tag}.md"
    json_path.write_text(json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(bundle, md_path)

    print(json.dumps({
        "json": str(json_path),
        "markdown": str(md_path),
        "refresh_pass": bundle["refresh_pass"],
        "offline_scorer_ready": bundle["evidence_levels"]["offline_scorer_ready"],
        "gradient_guidance_ready": bundle["evidence_levels"]["gradient_guidance_ready"],
        "server_rollout_schema_ready": bundle["evidence_levels"]["server_rollout_schema_ready"],
        "real_paired_rollout_complete": bundle["evidence_levels"]["real_paired_rollout_complete"],
        "goal_complete": bundle["evidence_levels"]["goal_complete"],
        "rollout_status_counts": bundle["real_rollout_coverage"]["status_counts"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
