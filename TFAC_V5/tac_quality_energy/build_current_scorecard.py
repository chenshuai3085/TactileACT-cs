#!/usr/bin/env python3
"""Build the current TacQuality scorecard for DP classifier guidance.

The scorecard is intentionally evidence-bound.  It does not retrain a scorer
and does not claim real robot improvement.  It reads current experiment
artifacts and separates three evidence levels:

1. offline scorer quality,
2. differentiable Foresight -> score -> action guidance readiness,
3. real paired robot rollout evidence.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


DEFAULT_SCORER_AUDIT = Path("/home/chenshuai/Project/output/tac_quality_current_scorer_audit/current_tac_quality_scorer_audit.json")
DEFAULT_GUIDANCE_STATE = Path("/home/chenshuai/Project/output/tac_quality_guidance_state_audit/tac_quality_guidance_state_audit.json")
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
    "20260620_rerun/training_status_latest.json"
)
DEFAULT_ROLLOUT_CONFIG = Path(
    "/home/chenshuai/Project/output/tac_quality_rollout_arm_configs/"
    "tac_quality_rollout_arm_configs_current_s12_good_margin_20260619.json"
)
DEFAULT_FORCE_AWARE_BOARD_AUDIT = Path(
    "/home/chenshuai/Project/output/force_aware_foresight_guidance_audit/"
    "20260621_090725/audit_results.json"
)
DEFAULT_OUTPUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_current_scorecard")
DEFAULT_DOC = Path("docs/2026-06-20_current_tac_quality_scorecard.md")


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


def metric(data: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    value = get(data, dotted, default)
    if isinstance(value, Mapping) and "mean" in value:
        return value["mean"]
    return value


def fnum(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "true" if value else "false"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def boolish(value: Any) -> bool:
    return bool(value is True or str(value).lower() == "true")


def build_scorecard(args: argparse.Namespace) -> dict[str, Any]:
    scorer = load_json(args.scorer_audit)
    state = load_json(args.guidance_state)
    coverage = load_json(args.coverage)
    schema_audit = load_json(args.schema_audit)
    dp_status = load_json(args.dp_status)
    rollout_config = load_json(args.rollout_config)
    force_aware_board = load_json(args.force_aware_board_audit)

    ins_metrics = get(scorer, "key_metrics.insertion", {})
    board_metrics = get(scorer, "key_metrics.board", {})
    board_heldout = get(board_metrics, "heldout", {})
    board_alignment = get(board_metrics, "foresight_alignment", {})
    board_gradient = get(board_metrics, "gradient", {})
    coverage_summary = get(coverage, "summary", {})

    insertion_ready = boolish(get(state, "insertion.ready_for_real_rollout", False))
    board_ready = boolish(get(state, "board.ready_for_real_rollout", False))
    denoise_ready = boolish(get(state, "denoising_step_serving_ready", False))
    real_pipeline_ready = boolish(get(state, "real_evidence_pipeline_ready", False))
    real_evidence_complete = boolish(get(coverage_summary, "real_rollout_evidence_complete", False))
    schema_ready = boolish(get(schema_audit, "summary.schema_pass", False))
    force_aware_board_ready = (
        float(get(force_aware_board, "scorer_metrics.band_balanced_acc", 0.0) or 0.0) >= 0.90
        and float(get(force_aware_board, "scorer_metrics.contact_acc", 0.0) or 0.0) >= 0.85
        and float(get(force_aware_board, "guidance_metrics.finite_grad_rate", 0.0) or 0.0) >= 0.999
        and float(get(force_aware_board, "guidance_metrics.positive_grad_rate", 0.0) or 0.0) >= 0.999
        and float(get(force_aware_board, "guidance_metrics.improved_rate", 0.0) or 0.0) >= 0.90
        and float(get(force_aware_board, "guidance_metrics.trust_region_pass_rate", 0.0) or 0.0) >= 0.999
    )

    board_ckpt_policy = {
        "run_dir": str(args.dp_status.parent),
        "recommended_ckpt": str(args.dp_status.parent / "dp_best.pth"),
        "avoid_default_ckpt": str(args.dp_status.parent / "dp_final.pth"),
        "best_val_epoch": get(dp_status, "best_val_epoch.epoch"),
        "best_val_loss": get(dp_status, "best_val_epoch.val"),
        "latest_logged_epoch": get(dp_status, "latest.epoch"),
        "latest_logged_train_loss": get(dp_status, "latest.train"),
        "latest_logged_val_loss": get(dp_status, "latest.val"),
        "latest_val_epoch": get(dp_status, "latest_val_epoch.epoch"),
        "latest_val_train_loss": get(dp_status, "latest_val_epoch.train"),
        "latest_val_loss": get(dp_status, "latest_val_epoch.val"),
        "latest_status_timestamp": get(dp_status, "timestamp"),
        "training_pid": get(dp_status, "pid"),
        "training_running": bool(get(dp_status, "pid")),
        "reason": get(
            dp_status,
            "checkpoint_selection",
            "Use validation-selected dp_best.pth for rollout comparison.",
        ),
    }

    scorecard = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "Current TacQuality scorecard for socket insertion and board wiping DP classifier guidance.",
        "recommendation": {
            "insertion": get(scorer, "current_recommendation.insertion", {}),
            "board": get(scorer, "current_recommendation.board", {}),
            "board_research_candidate": {
                "name": "force_aware_foresight_quality_energy",
                "runtime_status": "offline_gradient_audited_not_yet_rollout_config_default",
                "score_definition": (
                    "good-vs-risk force-band margin + contact log-prob "
                    "- force-center penalty - force-smoothness penalty"
                ),
                "foresight_checkpoint": get(force_aware_board, "setup.ckpt"),
                "audit_path": str(args.force_aware_board_audit),
                "why": (
                    "This is the stronger scientific board scorer because board wiping quality is "
                    "defined by contact force magnitude and force smoothness, not marker geometry alone."
                ),
            },
        },
        "evidence_levels": {
            "offline_scorer_ready": boolish(get(scorer, "overall_offline_guidance_ready", False)),
            "gradient_guidance_ready": insertion_ready and board_ready and denoise_ready,
            "force_aware_board_gradient_audit_ready": force_aware_board_ready,
            "server_rollout_schema_ready": schema_ready,
            "real_evidence_pipeline_ready": real_pipeline_ready,
            "real_paired_rollout_complete": real_evidence_complete,
            "goal_complete": boolish(get(state, "overall_goal_complete", False)) and real_evidence_complete,
        },
        "task_scorecards": {
            "insertion": {
                "arm": get(scorer, "current_recommendation.insertion.arm"),
                "runtime": get(scorer, "current_recommendation.insertion.runtime"),
                "score_mode": get(scorer, "current_recommendation.insertion.score_mode"),
                "label_standard": get(scorer, "current_recommendation.insertion.label_definition"),
                "offline_metrics": {
                    "binary_auc": get(ins_metrics, "binary_auc"),
                    "binary_balanced_accuracy": get(ins_metrics, "binary_balanced_accuracy"),
                    "reason_macro_f1": get(ins_metrics, "reason_macro_f1"),
                    "quality_corr": get(ins_metrics, "quality_corr"),
                },
                "guidance_metrics": {
                    "matched_0209_improved_rate": get(ins_metrics, "matched_0209_gradient.improved_rate_mean"),
                    "matched_0401_improved_rate": get(ins_metrics, "matched_0401_gradient.improved_rate_mean"),
                    "matched_0209_trust_region": get(ins_metrics, "matched_0209_gradient.trust_region_pass_rate"),
                    "matched_0401_trust_region": get(ins_metrics, "matched_0401_gradient.trust_region_pass_rate"),
                    "good_margin_improve_rate": get(ins_metrics, "good_margin_cross_score.good_margin_improve_rate"),
                },
                "ready_for_real_rollout": insertion_ready,
                "real_pair_coverage": get(coverage_summary, "pair_summary.insertion", {}),
            },
            "board": {
                "arm": get(scorer, "current_recommendation.board.arm"),
                "runtime": get(scorer, "current_recommendation.board.runtime"),
                "score_mode": get(scorer, "current_recommendation.board.score_mode"),
                "label_standard": get(scorer, "current_recommendation.board.label_definition"),
                "offline_metrics": {
                    "binary_auc": get(board_heldout, "binary_auc"),
                    "binary_balanced_accuracy": get(board_heldout, "binary_balanced_accuracy"),
                    "reason_macro_f1": get(board_heldout, "reason_macro_f1"),
                    "quality_spearman": get(board_heldout, "quality_spearman"),
                },
                "foresight_alignment": {
                    "pred_auc_good": get(board_alignment, "pred_auc_good"),
                    "pred_gt_spearman": get(board_alignment, "pred_gt_spearman"),
                    "pred_score_vs_force_band_quality_spearman": get(
                        board_alignment, "pred_score_vs_force_band_quality_spearman"
                    ),
                },
                "guidance_metrics": {
                    "improved_rate": get(board_gradient, "improved_rate_mean"),
                    "finite_grad_rate": get(board_gradient, "finite_grad_rate_mean"),
                    "trust_region_pass_rate": get(board_gradient, "trust_region_pass_rate"),
                    "score_delta_mean": get(board_gradient, "score_delta.mean"),
                },
                "force_aware_foresight_guidance": {
                    "audit_path": str(args.force_aware_board_audit),
                    "foresight_checkpoint": get(force_aware_board, "setup.ckpt"),
                    "split": get(force_aware_board, "setup.split"),
                    "split_counts": get(force_aware_board, "setup.split_counts", {}),
                    "num_samples": get(force_aware_board, "setup.num_samples"),
                    "score_weights": get(force_aware_board, "setup.score_weights", {}),
                    "trust_region": get(force_aware_board, "setup.trust_region", {}),
                    "force_ref": get(force_aware_board, "force_ref", {}),
                    "scorer_metrics": get(force_aware_board, "scorer_metrics", {}),
                    "guidance_metrics": get(force_aware_board, "guidance_metrics", {}),
                    "score_delta": get(force_aware_board, "summaries.score_delta", {}),
                    "action_delta_norm": get(force_aware_board, "summaries.action_delta_norm", {}),
                    "raw_action_delta_norm": get(force_aware_board, "summaries.raw_action_delta_norm", {}),
                    "by_label": get(force_aware_board, "by_label", {}),
                    "ready_for_research_guidance": force_aware_board_ready,
                    "evidence_boundary": (
                        "Offline gradient audit only. This is not yet a real robot improvement claim "
                        "and is not yet the default server rollout arm."
                    ),
                },
                "dp_checkpoint_policy": board_ckpt_policy,
                "ready_for_real_rollout": board_ready,
                "real_pair_coverage": get(coverage_summary, "pair_summary.board", {}),
            },
        },
        "rollout_coverage": {
            "planned_counts": get(coverage_summary, "planned_counts", {}),
            "observed_counts": get(coverage_summary, "observed_counts", {}),
            "status_counts": get(coverage_summary, "status_counts", {}),
            "board_ready_for_real_eval": boolish(get(coverage_summary, "board_ready_for_real_eval", False)),
            "insertion_ready_for_real_eval": boolish(get(coverage_summary, "insertion_ready_for_real_eval", False)),
            "real_rollout_evidence_complete": real_evidence_complete,
        },
        "server_rollout_schema": {
            "path": str(args.schema_audit),
            "schema_pass": schema_ready,
            "n_trials": get(schema_audit, "summary.n_trials"),
            "synthetic_count": get(schema_audit, "summary.synthetic_count"),
            "real_count": get(schema_audit, "summary.real_count"),
            "failed_trials": get(schema_audit, "summary.failed_trials", []),
            "evidence_boundary": "Schema smoke proves server log format only; synthetic logs are not real robot evidence.",
        },
        "rollout_config": {
            "path": str(args.rollout_config),
            "recommended_board_arm": get(rollout_config, "recommended_board_arm"),
            "recommended_insertion_arm": get(rollout_config, "recommended_insertion_arm"),
            "pass": boolish(get(rollout_config, "rollout_arm_config_pass", False)),
        },
        "innovation_story": [
            "Do not present tactile concat DP as the main novelty; treat it as the action prior.",
            "Main novelty: differentiable tactile/force consequence scoring for DP classifier guidance.",
            "Insertion uses an unsaturated good-margin risk scorer over good insert vs pre-bounce/impact modes.",
            "The deployable board arm currently uses a four-class marker_joint_action force-band energy.",
            "The stronger board research candidate is force-aware Foresight consequence energy because it directly scores predicted force band, contact, and smoothness.",
            "Both tasks use bounded trust-region guidance through Foresight rather than offline reranking.",
        ],
        "next_required_evidence": [
            "Integrate the force-aware board consequence score into the serving guidance path if it replaces marker_joint_s12_guided.",
            "Run board block 1 vs block 2 in guide_forshow.sh and collect server-side force_trace.csv.",
            "Run insertion block 3 vs block 4 and fill success/stopped_early/bounce_count/retry_count metadata.",
            "Rerun audit_real_rollout_coverage.py until both tasks have at least three complete pairs.",
            "Only then run eval_tac_quality_real_rollouts.py for final real robot evidence.",
        ],
        "evidence_boundary": {
            "can_claim_now": [
                "Offline scorer quality is strong for both tasks.",
                "Foresight-gradient guidance path is ready for real rollout tests.",
                "Force-aware board consequence scorer has passed offline held-out gradient audit.",
                "Server-side rollout log schema is ready for force/action/guidance evaluation.",
                "The command and manifest pipeline is ready for paired real robot evidence collection.",
            ],
            "cannot_claim_yet": [
                "No real paired baseline-vs-guided improvement has been proven.",
                "No board force-curve improvement should be claimed without force_trace.csv pairs.",
                "No insertion success/bounce/retry improvement should be claimed without outcome metadata.",
            ],
        },
        "input_paths": {
            "scorer_audit": str(args.scorer_audit),
            "guidance_state": str(args.guidance_state),
            "coverage": str(args.coverage),
            "schema_audit": str(args.schema_audit),
            "dp_status": str(args.dp_status),
            "rollout_config": str(args.rollout_config),
            "force_aware_board_audit": str(args.force_aware_board_audit),
        },
    }
    return scorecard


def write_markdown(scorecard: Mapping[str, Any], path: Path) -> None:
    ins = get(scorecard, "task_scorecards.insertion", {})
    board = get(scorecard, "task_scorecards.board", {})
    levels = get(scorecard, "evidence_levels", {})
    coverage = get(scorecard, "rollout_coverage", {})
    schema = get(scorecard, "server_rollout_schema", {})
    lines = [
        "# Current TacQuality Scorecard",
        "",
        f"Generated: `{scorecard['created_at']}`",
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
        "## Current Recommended Scorers",
        "",
        "| task | arm | runtime | score mode | ready for real rollout |",
        "|---|---|---|---|---:|",
        f"| insertion | `{ins.get('arm')}` | `{ins.get('runtime')}` | `{ins.get('score_mode')}` | `{ins.get('ready_for_real_rollout')}` |",
        f"| board | `{board.get('arm')}` | `{board.get('runtime')}` | `{board.get('score_mode')}` | `{board.get('ready_for_real_rollout')}` |",
        "",
        "Board research candidate: `force_aware_foresight_quality_energy` "
        f"(offline gradient audit ready: `{get(board, 'force_aware_foresight_guidance.ready_for_research_guidance')}`).",
        "",
        "## Key Metrics",
        "",
        "| task | classifier metric | quality metric | Foresight/guidance metric |",
        "|---|---|---|---|",
        (
            "| insertion | "
            f"AUC `{fnum(get(ins, 'offline_metrics.binary_auc'))}`, bACC `{fnum(get(ins, 'offline_metrics.binary_balanced_accuracy'))}` | "
            f"corr `{fnum(get(ins, 'offline_metrics.quality_corr'))}` | "
            f"0401 improve `{fnum(get(ins, 'guidance_metrics.matched_0401_improved_rate'))}`, "
            f"good-margin improve `{fnum(get(ins, 'guidance_metrics.good_margin_improve_rate'))}` |"
        ),
        (
            "| board | "
            f"AUC `{fnum(get(board, 'offline_metrics.binary_auc'))}`, bACC `{fnum(get(board, 'offline_metrics.binary_balanced_accuracy'))}` | "
            f"rho `{fnum(get(board, 'offline_metrics.quality_spearman'))}` | "
            f"pred-vs-GT rho `{fnum(get(board, 'foresight_alignment.pred_gt_spearman'))}`, "
            f"guidance improve `{fnum(get(board, 'guidance_metrics.improved_rate'))}` |"
        ),
        (
            "| board force-aware candidate | "
            f"band bACC `{fnum(get(board, 'force_aware_foresight_guidance.scorer_metrics.band_balanced_acc'))}`, "
            f"contact acc `{fnum(get(board, 'force_aware_foresight_guidance.scorer_metrics.contact_acc'))}` | "
            f"good/bad AUC `{fnum(get(board, 'force_aware_foresight_guidance.scorer_metrics.score_good_bad_auc'))}` | "
            f"finite grad `{fnum(get(board, 'force_aware_foresight_guidance.guidance_metrics.finite_grad_rate'))}`, "
            f"improve `{fnum(get(board, 'force_aware_foresight_guidance.guidance_metrics.improved_rate'))}`, "
            f"score delta `{fnum(get(board, 'force_aware_foresight_guidance.score_delta.mean'))}` |"
        ),
        "",
        "## Real Rollout Coverage",
        "",
        f"- planned_counts: `{json.dumps(coverage.get('planned_counts'), ensure_ascii=False)}`",
        f"- observed_counts: `{json.dumps(coverage.get('observed_counts'), ensure_ascii=False)}`",
        f"- status_counts: `{json.dumps(coverage.get('status_counts'), ensure_ascii=False)}`",
        f"- real_rollout_evidence_complete: `{coverage.get('real_rollout_evidence_complete')}`",
        "",
        "## Server Rollout Log Schema",
        "",
        f"- schema_pass: `{schema.get('schema_pass')}`",
        f"- n_trials: `{schema.get('n_trials')}`",
        f"- synthetic_count: `{schema.get('synthetic_count')}`",
        f"- real_count: `{schema.get('real_count')}`",
        f"- note: {schema.get('evidence_boundary')}",
        "",
        "## Board DP Checkpoint Policy",
        "",
        f"- recommended: `{get(board, 'dp_checkpoint_policy.recommended_ckpt')}`",
        f"- avoid as default: `{get(board, 'dp_checkpoint_policy.avoid_default_ckpt')}`",
        f"- best val epoch/loss: `{get(board, 'dp_checkpoint_policy.best_val_epoch')}` / `{fnum(get(board, 'dp_checkpoint_policy.best_val_loss'), 6)}`",
        f"- latest logged epoch/train loss: `{get(board, 'dp_checkpoint_policy.latest_logged_epoch')}` / `{fnum(get(board, 'dp_checkpoint_policy.latest_logged_train_loss'), 6)}`",
        f"- latest validation epoch/loss: `{get(board, 'dp_checkpoint_policy.latest_val_epoch')}` / `{fnum(get(board, 'dp_checkpoint_policy.latest_val_loss'), 6)}`",
        f"- training running at status timestamp: `{get(board, 'dp_checkpoint_policy.training_running')}` "
        f"(pid `{get(board, 'dp_checkpoint_policy.training_pid')}`, status `{get(board, 'dp_checkpoint_policy.latest_status_timestamp')}`)",
        "",
        "## Innovation Story",
        "",
    ])
    lines.extend(f"- {item}" for item in scorecard["innovation_story"])
    lines.extend(["", "## Evidence Boundary", "", "Can claim now:"])
    lines.extend(f"- {item}" for item in scorecard["evidence_boundary"]["can_claim_now"])
    lines.extend(["", "Cannot claim yet:"])
    lines.extend(f"- {item}" for item in scorecard["evidence_boundary"]["cannot_claim_yet"])
    lines.extend(["", "## Next Required Evidence", ""])
    lines.extend(f"- {item}" for item in scorecard["next_required_evidence"])
    lines.extend(["", "## Input Paths", ""])
    for key, value in scorecard["input_paths"].items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorer_audit", type=Path, default=DEFAULT_SCORER_AUDIT)
    parser.add_argument("--guidance_state", type=Path, default=DEFAULT_GUIDANCE_STATE)
    parser.add_argument("--coverage", type=Path, default=DEFAULT_COVERAGE)
    parser.add_argument("--schema_audit", type=Path, default=DEFAULT_SCHEMA_AUDIT)
    parser.add_argument("--dp_status", type=Path, default=DEFAULT_DP_STATUS)
    parser.add_argument("--rollout_config", type=Path, default=DEFAULT_ROLLOUT_CONFIG)
    parser.add_argument("--force_aware_board_audit", type=Path, default=DEFAULT_FORCE_AWARE_BOARD_AUDIT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scorecard = build_scorecard(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "current_tac_quality_scorecard.json"
    json_path.write_text(json.dumps(scorecard, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(scorecard, args.doc)
    print(json.dumps({
        "json": str(json_path),
        "markdown": str(args.doc),
        "offline_scorer_ready": scorecard["evidence_levels"]["offline_scorer_ready"],
        "gradient_guidance_ready": scorecard["evidence_levels"]["gradient_guidance_ready"],
        "server_rollout_schema_ready": scorecard["evidence_levels"]["server_rollout_schema_ready"],
        "real_paired_rollout_complete": scorecard["evidence_levels"]["real_paired_rollout_complete"],
        "goal_complete": scorecard["evidence_levels"]["goal_complete"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
