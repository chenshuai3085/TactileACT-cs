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
    "current_forceaware_goodmargin_coverage/tac_quality_real_rollout_coverage.json"
)
DEFAULT_SCHEMA_AUDIT = Path(
    "/home/chenshuai/Project/output/tac_quality_server_rollout_schema_audit/"
    "current_schema_smoke/tac_quality_server_rollout_schema_audit.json"
)
DEFAULT_DP_STATUS = Path(
    "/media/chenshuai/EXTERNAL_USB/pih_output/"
    "dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_"
    "20260621_codex/training_status_latest.json"
)
DEFAULT_ROLLOUT_CONFIG = Path(
    "/home/chenshuai/Project/output/tac_quality_rollout_arm_configs/"
    "tac_quality_rollout_arm_configs_current_s12_good_margin_forceaware_board_20260621.json"
)
DEFAULT_INSERTION_CONFIG_CONSISTENCY = Path(
    "/home/chenshuai/Project/output/insertion_config_consistency/"
    "20260621_113544/insertion_config_consistency.json"
)
DEFAULT_FORCE_AWARE_BOARD_AUDIT = Path(
    "/home/chenshuai/Project/output/force_aware_foresight_guidance_audit/"
    "20260621_090725/audit_results.json"
)
DEFAULT_FORCE_AWARE_BOARD_SMOKE = Path(
    "/home/chenshuai/Project/output/tac_quality_guided_server_packet/"
    "board_force_aware_guided_smoke_20260621/guided_server_dry_run_smoke.json"
)
DEFAULT_FORCE_AWARE_BOARD_DENOISE_SMOKE = Path(
    "/home/chenshuai/Project/output/tac_quality_guided_server_packet/"
    "board_force_aware_denoising_step_smoke_20260621/guided_server_dry_run_smoke.json"
)
DEFAULT_FORCE_AWARE_DENOISING_REAL_WINDOW = Path(
    "/home/chenshuai/Project/output/force_aware_denoising_real_window_audit/"
    "20260621_173857/force_aware_denoising_real_window_audit.json"
)
DEFAULT_FORCE_AWARE_SERVING_REAL_WINDOW = Path(
    "/home/chenshuai/Project/output/force_aware_serving_real_window_audit/"
    "20260621_110440/force_aware_serving_real_window_audit.json"
)
DEFAULT_FORCE_AWARE_ROLLOUT_MANIFEST = Path(
    "/home/chenshuai/Project/output/tac_quality_real_rollout_manifest/"
    "board_force_aware_manifest/tac_quality_rollout_manifest.json"
)
DEFAULT_FORCE_AWARE_ROLLOUT_COVERAGE = Path(
    "/home/chenshuai/Project/output/tac_quality_real_rollout_coverage/"
    "board_force_aware_coverage/tac_quality_real_rollout_coverage.json"
)
DEFAULT_FORCE_AWARE_ROLLOUT_EVAL = Path(
    "/home/chenshuai/Project/output/tac_quality_real_rollout_eval/"
    "board_force_aware_tac_quality_precheck/tac_quality_real_rollout_eval.json"
)
DEFAULT_FORCE_AWARE_WEIGHT_SWEEP = Path(
    "/home/chenshuai/Project/output/force_aware_score_weight_sweep/"
    "20260621_104503/force_aware_score_weight_sweep.json"
)
DEFAULT_FORCE_AWARE_CONFIG_CONSISTENCY = Path(
    "/home/chenshuai/Project/output/force_aware_config_consistency/"
    "20260621_111101/force_aware_config_consistency.json"
)
DEFAULT_FORCE_AWARE_LABEL_SEPARATION = Path(
    "/home/chenshuai/Project/output/force_aware_label_separation/"
    "20260621_180426/force_aware_label_separation.json"
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


def as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def summarize_manifest_rows(manifest: Mapping[str, Any], task: str) -> dict[str, Any]:
    rows = [row for row in manifest.get("rows", []) if isinstance(row, Mapping) and row.get("task") == task]
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get("group")), []).append(row)

    def unique(group: str, key: str) -> list[Any]:
        return sorted({row.get(key) for row in groups.get(group, []) if row.get(key) is not None})

    return {
        "n_rows": len(rows),
        "groups": {group: len(items) for group, items in groups.items()},
        "baseline_arms": unique("baseline", "server_arm"),
        "guided_arms": unique("guided", "server_arm"),
        "baseline_ports": unique("baseline", "server_port"),
        "guided_ports": unique("guided", "server_port"),
        "baseline_roots": unique("baseline", "server_log_root"),
        "guided_roots": unique("guided", "server_log_root"),
        "pair_ids": sorted({row.get("pair_id") for row in rows if row.get("pair_id") is not None}),
    }


def signal_check(
    *,
    score_delta_mean: Any,
    action_delta_mean: Any,
    min_score_delta_mean: float,
    min_action_delta_mean: float,
    score_scale: str,
    note: str,
) -> dict[str, Any]:
    """Separate nonzero gradients from practically meaningful guidance signal."""

    score_delta = as_float(score_delta_mean)
    action_delta = as_float(action_delta_mean)
    score_pass = score_delta is not None and score_delta >= min_score_delta_mean
    action_pass = action_delta is not None and action_delta >= min_action_delta_mean
    strong = bool(score_pass and action_pass)
    return {
        "score_delta_mean": score_delta,
        "action_delta_mean": action_delta,
        "min_score_delta_mean": min_score_delta_mean,
        "min_action_delta_mean": min_action_delta_mean,
        "score_delta_pass": score_pass,
        "action_delta_pass": action_pass,
        "strong_signal": strong,
        "status": "strong" if strong else "weak_or_unproven",
        "score_scale": score_scale,
        "note": note,
    }


def build_scorecard(args: argparse.Namespace) -> dict[str, Any]:
    scorer = load_json(args.scorer_audit)
    state = load_json(args.guidance_state)
    coverage = load_json(args.coverage)
    schema_audit = load_json(args.schema_audit)
    dp_status = load_json(args.dp_status)
    rollout_config = load_json(args.rollout_config)
    insertion_config_consistency = load_json(args.insertion_config_consistency)
    force_aware_board = load_json(args.force_aware_board_audit)
    force_aware_smoke = load_json(args.force_aware_board_smoke)
    force_aware_denoise_smoke = load_json(args.force_aware_board_denoise_smoke)
    force_aware_denoise_real_window = load_json(args.force_aware_denoising_real_window)
    force_aware_real_window = load_json(args.force_aware_serving_real_window)
    force_aware_manifest = load_json(args.force_aware_rollout_manifest)
    force_aware_coverage = load_json(args.force_aware_rollout_coverage)
    force_aware_eval = load_json(args.force_aware_rollout_eval)
    force_aware_weight_sweep = load_json(args.force_aware_weight_sweep)
    force_aware_config_consistency = load_json(args.force_aware_config_consistency)
    force_aware_label_separation = load_json(args.force_aware_label_separation)

    ins_metrics = get(scorer, "key_metrics.insertion", {})
    board_metrics = get(scorer, "key_metrics.board", {})
    board_heldout = get(board_metrics, "heldout", {})
    board_alignment = get(board_metrics, "foresight_alignment", {})
    board_gradient = get(board_metrics, "gradient", {})
    coverage_summary = get(coverage, "summary", {})

    insertion_ready = boolish(get(state, "insertion.ready_for_real_rollout", False))
    board_ready = boolish(get(state, "board.ready_for_real_rollout", False))
    denoise_ready = boolish(get(state, "denoising_step_serving_ready", False))
    current_gradient_guidance_ready = insertion_ready and board_ready
    real_pipeline_ready = boolish(get(state, "real_evidence_pipeline_ready", False))
    real_evidence_complete = boolish(get(coverage_summary, "real_rollout_evidence_complete", False))
    schema_ready = boolish(get(schema_audit, "summary.schema_pass", False))
    insertion_config_consistent = (
        boolish(get(insertion_config_consistency, "pass", False))
        and get(insertion_config_consistency, "arm") == "good_margin_guided"
        and get(insertion_config_consistency, "expected.runtime") == "InsertionRiskScorerRuntime"
        and get(insertion_config_consistency, "expected.score_mode") == "good_margin"
    )
    force_aware_board_ready = (
        float(get(force_aware_board, "scorer_metrics.band_balanced_acc", 0.0) or 0.0) >= 0.90
        and float(get(force_aware_board, "scorer_metrics.contact_acc", 0.0) or 0.0) >= 0.85
        and float(get(force_aware_board, "guidance_metrics.finite_grad_rate", 0.0) or 0.0) >= 0.999
        and float(get(force_aware_board, "guidance_metrics.positive_grad_rate", 0.0) or 0.0) >= 0.999
        and float(get(force_aware_board, "guidance_metrics.improved_rate", 0.0) or 0.0) >= 0.90
        and float(get(force_aware_board, "guidance_metrics.trust_region_pass_rate", 0.0) or 0.0) >= 0.999
    )
    force_aware_label_separation_ready = (
        boolish(get(force_aware_label_separation, "pass", False))
        and float(get(force_aware_label_separation, "scorer_metrics.score_good_bad_auc", 0.0) or 0.0) >= 0.99
        and float(get(force_aware_label_separation, "scorer_metrics.band_balanced_acc", 0.0) or 0.0) >= 0.95
        and float(get(force_aware_label_separation, "separation.good_vs_worst_bad_margin", 0.0) or 0.0) >= 10.0
        and float(get(force_aware_label_separation, "separation.good_score_mean", -1.0) or -1.0) > 0.0
        and float(get(force_aware_label_separation, "separation.worst_bad_score_mean", 1.0) or 1.0) < 0.0
    )
    force_aware_serving_ready = (
        boolish(get(force_aware_smoke, "dry_run_guidance_smoke_pass", False))
        and get(force_aware_smoke, "report.scorer_runtime") == "ForceAwareForesightGuidanceRuntime"
        and get(force_aware_smoke, "report.adapter_policy") == "force_aware_foresight_trust_region_refinement"
        and boolish(get(force_aware_smoke, "not_reranking", False))
        and float(get(force_aware_smoke, "report.finite_grad_rate", 0.0) or 0.0) >= 0.999
        and float(get(force_aware_smoke, "report.positive_grad_rate", 0.0) or 0.0) >= 0.999
    )
    force_aware_denoise_serving_ready = (
        boolish(get(force_aware_denoise_smoke, "dry_run_guidance_smoke_pass", False))
        and get(force_aware_denoise_smoke, "task") == "board"
        and get(force_aware_denoise_smoke, "arm") == "force_aware_guided"
        and get(force_aware_denoise_smoke, "report.scorer_runtime") == "ForceAwareForesightGuidanceRuntime"
        and get(force_aware_denoise_smoke, "report.adapter_policy")
        == "denoising_step_force_aware_tac_quality_guidance"
        and boolish(get(force_aware_denoise_smoke, "report.every_step_ddpm_guidance", False))
        and boolish(get(force_aware_denoise_smoke, "not_reranking", False))
        and str(get(force_aware_denoise_smoke, "guidance_location", "")).startswith(
            "inside DP denoising loop"
        )
        and float(get(force_aware_denoise_smoke, "report.finite_grad_rate", 0.0) or 0.0) >= 0.999
        and float(get(force_aware_denoise_smoke, "report.positive_grad_rate", 0.0) or 0.0) >= 0.999
        and float(get(force_aware_denoise_smoke, "report.accept_rate", 0.0) or 0.0) >= 0.999
    )
    force_aware_denoise_real_window_ready = (
        boolish(get(force_aware_denoise_real_window, "summary.pass", False))
        and boolish(get(force_aware_denoise_real_window, "checks.runtime_is_force_aware", False))
        and boolish(get(force_aware_denoise_real_window, "checks.adapter_policy_ok", False))
        and boolish(get(force_aware_denoise_real_window, "checks.denoising_location_ok", False))
        and boolish(get(force_aware_denoise_real_window, "checks.every_step_ddpm_guidance", False))
        and boolish(get(force_aware_denoise_real_window, "checks.not_reranking", False))
        and boolish(get(force_aware_denoise_real_window, "checks.guided_steps_executed_positive", False))
        and float(get(force_aware_denoise_real_window, "summary.finite_grad_rate_mean", 0.0) or 0.0) >= 0.999
        and float(get(force_aware_denoise_real_window, "summary.positive_grad_rate_mean", 0.0) or 0.0) >= 0.999
        and float(get(force_aware_denoise_real_window, "summary.accept_rate_mean", 0.0) or 0.0) >= 0.999
        and float(get(force_aware_denoise_real_window, "summary.trust_region_pass_rate", 0.0) or 0.0) >= 0.999
    )
    force_aware_real_window_ready = (
        boolish(get(force_aware_real_window, "summary.pass", False))
        and get(force_aware_real_window, "setup.arm") == "force_aware_guided"
        and boolish(get(force_aware_real_window, "checks.runtime_is_force_aware", False))
        and boolish(get(force_aware_real_window, "checks.adapter_policy_ok", False))
        and boolish(get(force_aware_real_window, "checks.not_reranking", False))
        and float(get(force_aware_real_window, "summary.finite_grad_rate_mean", 0.0) or 0.0) >= 0.999
        and float(get(force_aware_real_window, "summary.positive_grad_rate_mean", 0.0) or 0.0) >= 0.999
        and float(get(force_aware_real_window, "summary.improved_rate_mean", 0.0) or 0.0) >= 0.80
        and float(get(force_aware_real_window, "summary.trust_region_pass_rate", 0.0) or 0.0) >= 0.999
    )
    force_aware_manifest_rows = summarize_manifest_rows(force_aware_manifest, "board")
    force_aware_manifest_ready = (
        not boolish(get(force_aware_manifest, "_missing", False))
        and get(force_aware_manifest, "board_pairs") == 3
        and force_aware_manifest_rows["groups"].get("baseline") == 3
        and force_aware_manifest_rows["groups"].get("guided") == 3
        and force_aware_manifest_rows["baseline_arms"] == ["baseline"]
        and force_aware_manifest_rows["guided_arms"] == ["force_aware_guided"]
        and force_aware_manifest_rows["baseline_ports"] == [8765]
        and force_aware_manifest_rows["guided_ports"] == [8769]
        and force_aware_manifest_rows["baseline_roots"]
        == ["/home/chenshuai/Project/output/board_force_rollouts/260617_only_force_aware_scorer"]
        and force_aware_manifest_rows["guided_roots"]
        == ["/home/chenshuai/Project/output/board_force_rollouts/260617_only_force_aware_scorer"]
    )
    force_aware_rollout_complete = boolish(
        get(force_aware_coverage, "summary.real_rollout_evidence_complete", False)
    ) and boolish(get(force_aware_eval, "real_rollout_evidence_complete", False))
    force_aware_config_consistent = (
        boolish(get(force_aware_config_consistency, "pass", False))
        and get(force_aware_config_consistency, "arm") == "force_aware_guided"
    )

    insertion_signal = signal_check(
        score_delta_mean=get(ins_metrics, "matched_0401_gradient.score_delta.mean"),
        action_delta_mean=get(ins_metrics, "matched_0401_gradient.action_delta_norm.mean"),
        min_score_delta_mean=0.05,
        min_action_delta_mean=0.02,
        score_scale="Insertion profile/good-margin energy scale",
        note=(
            "Checks whether matched Foresight guidance changes both score and action "
            "by a nontrivial amount, not just with the correct sign."
        ),
    )
    board_deploy_signal = signal_check(
        score_delta_mean=get(board_gradient, "score_delta.mean"),
        action_delta_mean=get(board_gradient, "action_delta_norm.mean"),
        min_score_delta_mean=0.01,
        min_action_delta_mean=0.005,
        score_scale="Board deploy sigmoid quality score in [0, 1]",
        note=(
            "The marker_joint_s12 scorer classifies well, but this audit guards "
            "against a numerically tiny guidance update."
        ),
    )
    force_aware_signal = signal_check(
        score_delta_mean=get(force_aware_board, "summaries.score_delta.mean"),
        action_delta_mean=get(force_aware_board, "summaries.action_delta_norm.mean"),
        min_score_delta_mean=0.25,
        min_action_delta_mean=0.01,
        score_scale="Force-aware margin/contact/penalty energy scale",
        note=(
            "Force-aware score is not probability-bounded; threshold checks that "
            "guidance produces a visibly useful energy increase under trust region."
        ),
    )
    force_aware_real_window_signal = signal_check(
        score_delta_mean=get(force_aware_real_window, "summary.score_delta.mean"),
        action_delta_mean=get(force_aware_real_window, "summary.normalized_action_delta.mean"),
        min_score_delta_mean=0.25,
        min_action_delta_mean=0.005,
        score_scale="Force-aware serving audit on stratified real HDF5 windows",
        note=(
            "This is the closest offline proxy to deployment-time board guidance "
            "without claiming real robot improvement."
        ),
    )

    dp_artifacts = get(dp_status, "artifacts", {})
    recommended_ckpt = get(dp_artifacts, "dp_best.pth") if isinstance(dp_artifacts, Mapping) else None
    avoid_ckpt = get(dp_artifacts, "dp_final.pth") if isinstance(dp_artifacts, Mapping) else None
    board_ckpt_policy = {
        "run_dir": str(args.dp_status.parent),
        "recommended_ckpt": recommended_ckpt or str(args.dp_status.parent / "dp_best.pth"),
        "avoid_default_ckpt": avoid_ckpt or str(args.dp_status.parent / "dp_final.pth"),
        "best_val_epoch": get(dp_status, "best_val_epoch.epoch"),
        "best_val_loss": get(dp_status, "best_val_epoch.val"),
        "latest_logged_epoch": get(dp_status, "latest.epoch"),
        "latest_logged_train_loss": get(dp_status, "latest.train"),
        "latest_logged_val_loss": get(dp_status, "latest.val"),
        "latest_val_epoch": get(dp_status, "latest_val_epoch.epoch", get(dp_status, "latest.epoch")),
        "latest_val_train_loss": get(dp_status, "latest_val_epoch.train", get(dp_status, "latest.train")),
        "latest_val_loss": get(dp_status, "latest_val_epoch.val", get(dp_status, "latest.val")),
        "latest_status_timestamp": get(dp_status, "timestamp"),
        "training_pid": get(dp_status, "pid"),
        "training_running": bool(get(dp_status, "pid")),
        "reason": get(
            dp_status,
            "checkpoint_selection",
            "Use validation-selected dp_best.pth for rollout comparison.",
        ),
    }

    force_aware_board_recommendation = {
        "arm": "force_aware_guided",
        "runtime": "ForceAwareForesightGuidanceRuntime",
        "score_preset": get(force_aware_weight_sweep, "best.name", "margin_only"),
        "score_weights": get(force_aware_weight_sweep, "best.weights", {}),
        "weight_sweep_path": str(args.force_aware_weight_sweep),
        "status": "preferred_research_candidate_not_real_robot_proven",
        "why": (
            "Board quality is explicitly force-band and smoothness based; "
            "force_aware_guided has much stronger guidance signal than the "
            "deployable marker_joint_s12 scorer.  The current offline weight "
            "sweep selects the force-band good-vs-risk margin as the strongest "
            "bounded guidance score; extra contact/center/smooth penalties are "
            "kept as hypotheses for real force-trace validation rather than "
            "assumed improvements."
        ),
    }
    board_integrated_fallback = get(scorer, "current_recommendation.board", {})

    scorecard = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "Current TacQuality scorecard for socket insertion and board wiping DP classifier guidance.",
        "recommendation": {
            "insertion": get(scorer, "current_recommendation.insertion", {}),
            "board": force_aware_board_recommendation,
            "board_scientific_preference": force_aware_board_recommendation,
            "board_integrated_fallback": board_integrated_fallback,
            "board_research_candidate": {
                "name": "force_aware_foresight_quality_energy",
                "runtime_status": "scientific_priority_not_real_robot_proven",
                "score_definition": (
                    "good-vs-risk force-band margin + contact log-prob "
                    "- force-center penalty - force-smoothness penalty"
                ),
                "foresight_checkpoint": get(force_aware_board, "setup.ckpt"),
                "audit_path": str(args.force_aware_board_audit),
                "server_smoke_path": str(args.force_aware_board_smoke),
                "server_denoising_step_smoke_path": str(args.force_aware_board_denoise_smoke),
                "why": (
                    "This is the stronger scientific board scorer because board wiping quality is "
                    "defined by contact force magnitude and force smoothness, not marker geometry alone."
                ),
            },
        },
        "evidence_levels": {
            "offline_scorer_ready": boolish(get(scorer, "overall_offline_guidance_ready", False)),
            "gradient_guidance_ready": current_gradient_guidance_ready,
            "denoising_step_serving_ready": denoise_ready,
            "insertion_guidance_signal_strong": bool(insertion_signal["strong_signal"]),
            "insertion_config_consistent": insertion_config_consistent,
            "board_deploy_guidance_signal_strong": bool(board_deploy_signal["strong_signal"]),
            "force_aware_board_guidance_signal_strong": bool(
                force_aware_signal["strong_signal"] and force_aware_real_window_signal["strong_signal"]
            ),
            "force_aware_board_gradient_audit_ready": force_aware_board_ready,
            "force_aware_board_label_separation_ready": force_aware_label_separation_ready,
            "force_aware_board_serving_smoke_ready": force_aware_serving_ready,
            "force_aware_board_denoising_step_serving_ready": force_aware_denoise_serving_ready,
            "force_aware_board_denoising_real_window_ready": force_aware_denoise_real_window_ready,
            "force_aware_board_real_window_serving_ready": force_aware_real_window_ready,
            "force_aware_board_config_consistent": force_aware_config_consistent,
            "force_aware_board_rollout_manifest_ready": force_aware_manifest_ready,
            "force_aware_board_real_rollout_complete": force_aware_rollout_complete,
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
                "guidance_signal_strength": insertion_signal,
                "config_consistency": {
                    "path": str(args.insertion_config_consistency),
                    "pass": insertion_config_consistent,
                    "checks": get(insertion_config_consistency, "checks", []),
                    "summary": get(insertion_config_consistency, "summary", {}),
                    "expected": get(insertion_config_consistency, "expected", {}),
                    "evidence_boundary": get(insertion_config_consistency, "evidence_boundary"),
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
                "guidance_signal_strength": board_deploy_signal,
                "force_aware_foresight_guidance": {
                    "audit_path": str(args.force_aware_board_audit),
                    "foresight_checkpoint": get(force_aware_board, "setup.ckpt"),
                    "split": get(force_aware_board, "setup.split"),
                    "split_counts": get(force_aware_board, "setup.split_counts", {}),
                    "num_samples": get(force_aware_board, "setup.num_samples"),
                    "score_weights": get(
                        force_aware_real_window,
                        "setup.score_weights",
                        get(force_aware_board, "setup.score_weights", {}),
                    ),
                    "base_audit_score_weights": get(force_aware_board, "setup.score_weights", {}),
                    "trust_region": get(force_aware_board, "setup.trust_region", {}),
                    "force_ref": get(force_aware_board, "force_ref", {}),
                    "scorer_metrics": get(force_aware_board, "scorer_metrics", {}),
                    "guidance_metrics": get(force_aware_board, "guidance_metrics", {}),
                    "score_delta": get(force_aware_board, "summaries.score_delta", {}),
                    "action_delta_norm": get(force_aware_board, "summaries.action_delta_norm", {}),
                    "raw_action_delta_norm": get(force_aware_board, "summaries.raw_action_delta_norm", {}),
                    "guidance_signal_strength": force_aware_signal,
                    "score_weight_sweep": {
                        "path": str(args.force_aware_weight_sweep),
                        "created_at": get(force_aware_weight_sweep, "created_at"),
                        "best": get(force_aware_weight_sweep, "best", {}),
                        "top_presets": get(force_aware_weight_sweep, "rows_sorted", [])[:5],
                        "evidence_boundary": get(force_aware_weight_sweep, "evidence_boundary"),
                    },
                    "label_separation": {
                        "path": str(args.force_aware_label_separation),
                        "pass": force_aware_label_separation_ready,
                        "source_audit": get(force_aware_label_separation, "source_audit"),
                        "split": get(force_aware_label_separation, "split"),
                        "num_samples": get(force_aware_label_separation, "num_samples"),
                        "score_weights": get(force_aware_label_separation, "score_weights", {}),
                        "scorer_metrics": get(force_aware_label_separation, "scorer_metrics", {}),
                        "separation": get(force_aware_label_separation, "separation", {}),
                        "labels": get(force_aware_label_separation, "labels", {}),
                        "checks": get(force_aware_label_separation, "checks", []),
                        "evidence_boundary": get(force_aware_label_separation, "evidence_boundary"),
                    },
                    "config_consistency": {
                        "path": str(args.force_aware_config_consistency),
                        "pass": force_aware_config_consistent,
                        "checks": get(force_aware_config_consistency, "checks", []),
                        "summary": get(force_aware_config_consistency, "summary", {}),
                        "expected": get(force_aware_config_consistency, "expected", {}),
                        "evidence_boundary": get(force_aware_config_consistency, "evidence_boundary"),
                    },
                    "by_label": get(force_aware_board, "by_label", {}),
                    "ready_for_research_guidance": force_aware_board_ready,
                    "serving_smoke": {
                        "path": str(args.force_aware_board_smoke),
                        "pass": boolish(get(force_aware_smoke, "dry_run_guidance_smoke_pass", False)),
                        "task": get(force_aware_smoke, "task"),
                        "arm": get(force_aware_smoke, "arm"),
                        "runtime": get(force_aware_smoke, "report.scorer_runtime"),
                        "adapter_policy": get(force_aware_smoke, "report.adapter_policy"),
                        "not_reranking": boolish(get(force_aware_smoke, "not_reranking", False)),
                        "finite_grad_rate": get(force_aware_smoke, "report.finite_grad_rate"),
                        "positive_grad_rate": get(force_aware_smoke, "report.positive_grad_rate"),
                        "improved_rate": get(force_aware_smoke, "report.improved_rate"),
                        "score_delta": get(force_aware_smoke, "report.score_delta", {}),
                        "raw_action_delta": get(force_aware_smoke, "report.raw_action_delta", {}),
                        "integration_contract": get(force_aware_smoke, "report.integration_contract", {}),
                        "ready_for_optional_server_trial": force_aware_serving_ready,
                    },
                    "serving_denoising_step_smoke": {
                        "path": str(args.force_aware_board_denoise_smoke),
                        "pass": boolish(get(force_aware_denoise_smoke, "dry_run_guidance_smoke_pass", False)),
                        "task": get(force_aware_denoise_smoke, "task"),
                        "arm": get(force_aware_denoise_smoke, "arm"),
                        "runtime": get(force_aware_denoise_smoke, "report.scorer_runtime"),
                        "adapter_policy": get(force_aware_denoise_smoke, "report.adapter_policy"),
                        "score_mode": get(force_aware_denoise_smoke, "report.score_mode"),
                        "guidance_location": get(force_aware_denoise_smoke, "guidance_location"),
                        "every_step_ddpm_guidance": boolish(
                            get(force_aware_denoise_smoke, "report.every_step_ddpm_guidance", False)
                        ),
                        "ddpm_guidance": get(force_aware_denoise_smoke, "report.ddpm_guidance", {}),
                        "not_reranking": boolish(get(force_aware_denoise_smoke, "not_reranking", False)),
                        "finite_grad_rate": get(force_aware_denoise_smoke, "report.finite_grad_rate"),
                        "positive_grad_rate": get(force_aware_denoise_smoke, "report.positive_grad_rate"),
                        "accept_rate": get(force_aware_denoise_smoke, "report.accept_rate"),
                        "score_delta": get(force_aware_denoise_smoke, "report.score_delta", {}),
                        "raw_action_delta": get(force_aware_denoise_smoke, "report.raw_action_delta", {}),
                        "normalized_action_delta": get(
                            force_aware_denoise_smoke, "report.normalized_action_delta", {}
                        ),
                        "max_delta_within_trust_region": boolish(
                            get(force_aware_denoise_smoke, "report.max_delta_within_trust_region", False)
                        ),
                        "ready_for_optional_server_trial": force_aware_denoise_serving_ready,
                        "evidence_boundary": (
                            "Dry-run smoke proves the force-aware board scorer can be called inside "
                            "the DP denoising loop on predicted clean action x0. It is not real robot "
                            "evidence and uses a small smoke batch."
                        ),
                    },
                    "denoising_real_window_audit": {
                        "path": str(args.force_aware_denoising_real_window),
                        "pass": boolish(get(force_aware_denoise_real_window, "summary.pass", False)),
                        "ready_for_optional_server_trial": force_aware_denoise_real_window_ready,
                        "num_windows": get(force_aware_denoise_real_window, "setup.num_windows"),
                        "labels": get(force_aware_denoise_real_window, "summary.labels", {}),
                        "scheduler": get(force_aware_denoise_real_window, "setup.scheduler"),
                        "num_inference_steps": get(force_aware_denoise_real_window, "setup.num_inference_steps"),
                        "ddpm_guidance_steps": get(force_aware_denoise_real_window, "setup.ddpm_guidance_steps"),
                        "runtime_is_force_aware": boolish(
                            get(force_aware_denoise_real_window, "checks.runtime_is_force_aware", False)
                        ),
                        "adapter_policy_ok": boolish(
                            get(force_aware_denoise_real_window, "checks.adapter_policy_ok", False)
                        ),
                        "denoising_location_ok": boolish(
                            get(force_aware_denoise_real_window, "checks.denoising_location_ok", False)
                        ),
                        "every_step_ddpm_guidance": boolish(
                            get(force_aware_denoise_real_window, "checks.every_step_ddpm_guidance", False)
                        ),
                        "not_reranking": boolish(
                            get(force_aware_denoise_real_window, "checks.not_reranking", False)
                        ),
                        "guided_steps_executed_positive": boolish(
                            get(force_aware_denoise_real_window, "checks.guided_steps_executed_positive", False)
                        ),
                        "finite_grad_rate": get(force_aware_denoise_real_window, "summary.finite_grad_rate_mean"),
                        "positive_grad_rate": get(force_aware_denoise_real_window, "summary.positive_grad_rate_mean"),
                        "accept_rate": get(force_aware_denoise_real_window, "summary.accept_rate_mean"),
                        "trust_region_pass_rate": get(
                            force_aware_denoise_real_window, "summary.trust_region_pass_rate"
                        ),
                        "score_delta": get(force_aware_denoise_real_window, "summary.score_delta", {}),
                        "normalized_action_delta": get(
                            force_aware_denoise_real_window, "summary.normalized_action_delta", {}
                        ),
                        "contact_metric": get(force_aware_denoise_real_window, "summary.contact_metric", {}),
                        "evidence_boundary": get(force_aware_denoise_real_window, "evidence_boundary"),
                    },
                    "serving_real_window_audit": {
                        "path": str(args.force_aware_serving_real_window),
                        "pass": boolish(get(force_aware_real_window, "summary.pass", False)),
                        "split": get(force_aware_real_window, "setup.split"),
                        "num_windows": get(force_aware_real_window, "setup.num_windows"),
                        "labels": get(force_aware_real_window, "summary.labels", {}),
                        "selected_label_counts": get(force_aware_real_window, "setup.selected_label_counts", {}),
                        "runtime_is_force_aware": boolish(get(force_aware_real_window, "checks.runtime_is_force_aware", False)),
                        "adapter_policy_ok": boolish(get(force_aware_real_window, "checks.adapter_policy_ok", False)),
                        "not_reranking": boolish(get(force_aware_real_window, "checks.not_reranking", False)),
                        "finite_grad_rate": get(force_aware_real_window, "summary.finite_grad_rate_mean"),
                        "positive_grad_rate": get(force_aware_real_window, "summary.positive_grad_rate_mean"),
                        "improved_rate": get(force_aware_real_window, "summary.improved_rate_mean"),
                        "trust_region_pass_rate": get(force_aware_real_window, "summary.trust_region_pass_rate"),
                        "score_delta": get(force_aware_real_window, "summary.score_delta", {}),
                        "raw_action_delta": get(force_aware_real_window, "summary.raw_action_delta", {}),
                        "normalized_action_delta": get(force_aware_real_window, "summary.normalized_action_delta", {}),
                        "guidance_signal_strength": force_aware_real_window_signal,
                        "contact_metric": get(force_aware_real_window, "summary.contact_metric", {}),
                        "ready_for_optional_server_trial": force_aware_real_window_ready,
                        "evidence_boundary": get(force_aware_real_window, "evidence_boundary"),
                    },
                    "real_rollout_manifest": {
                        "manifest_path": str(args.force_aware_rollout_manifest),
                        "coverage_path": str(args.force_aware_rollout_coverage),
                        "eval_precheck_path": str(args.force_aware_rollout_eval),
                        "ready_for_collection": force_aware_manifest_ready,
                        "real_rollout_complete": force_aware_rollout_complete,
                        "tag": get(force_aware_manifest, "tag"),
                        "tasks": get(force_aware_manifest, "tasks", []),
                        "board_pairs": get(force_aware_manifest, "board_pairs"),
                        "n_trials": get(force_aware_manifest, "n_trials"),
                        "baseline_arm": (
                            force_aware_manifest_rows["baseline_arms"][0]
                            if force_aware_manifest_rows["baseline_arms"]
                            else None
                        ),
                        "guided_arm": (
                            force_aware_manifest_rows["guided_arms"][0]
                            if force_aware_manifest_rows["guided_arms"]
                            else None
                        ),
                        "baseline_port": (
                            force_aware_manifest_rows["baseline_ports"][0]
                            if force_aware_manifest_rows["baseline_ports"]
                            else None
                        ),
                        "guided_port": (
                            force_aware_manifest_rows["guided_ports"][0]
                            if force_aware_manifest_rows["guided_ports"]
                            else None
                        ),
                        "rollout_root": (
                            force_aware_manifest_rows["baseline_roots"][0]
                            if force_aware_manifest_rows["baseline_roots"]
                            == force_aware_manifest_rows["guided_roots"]
                            and force_aware_manifest_rows["baseline_roots"]
                            else None
                        ),
                        "row_summary": force_aware_manifest_rows,
                        "coverage_status_counts": get(force_aware_coverage, "summary.status_counts", {}),
                        "coverage_pair_summary": get(force_aware_coverage, "summary.pair_summary.board", {}),
                        "eval_board_ready": get(force_aware_eval, "board.real_comparison_ready"),
                        "eval_missing_force_trace": get(force_aware_eval, "board.missing_force_trace"),
                        "eval_detail": get(force_aware_eval, "board.detail"),
                        "evidence_boundary": (
                            "This proves the paired real-rollout collection route is specified. "
                            "It does not prove real robot improvement until force_trace.csv files exist "
                            "for the planned baseline/guided pairs and the evaluator passes."
                        ),
                    },
                    "evidence_boundary": (
                        "Offline gradient audit, serving dry-run, real-HDF5-window serving audit, "
                        "and real-rollout manifest readiness only. "
                        "This is not yet a real robot improvement claim and is not yet the default board arm."
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
            "scientific_priority_board_arm": force_aware_board_recommendation.get("arm"),
            "integrated_fallback_board_arm": get(board_integrated_fallback, "arm"),
            "recommended_insertion_arm": get(rollout_config, "recommended_insertion_arm"),
            "pass": boolish(get(rollout_config, "rollout_arm_config_pass", False)),
        },
        "innovation_story": [
            "Do not present tactile concat DP as the main novelty; treat it as the action prior.",
            "Main novelty: differentiable tactile/force consequence scoring for DP classifier guidance.",
            "Insertion uses an unsaturated good-margin risk scorer over good insert vs pre-bounce/impact modes.",
            "The board scientific priority is force-aware Foresight consequence energy because it directly scores predicted force band, contact, and smoothness.",
            "The marker_joint_action force-band energy remains an integrated comparison/fallback arm.",
            "Do not treat classification accuracy alone as sufficient; guidance signal strength must be nontrivial.",
            "Both tasks use bounded trust-region guidance through Foresight rather than offline reranking.",
        ],
        "next_required_evidence": [
            "Run board block 1b vs block 2c in guide_forshow.sh and collect server-side force_trace.csv under the force-aware rollout root.",
            "Keep marker_joint_s12 block 1 vs block 2 only as an integrated comparison/fallback ablation.",
            "Run insertion block 3 vs block 4 and fill success/stopped_early/bounce_count/retry_count metadata.",
            "Rerun audit_real_rollout_coverage.py until both tasks have at least three complete pairs.",
            "Only then run eval_tac_quality_real_rollouts.py for final real robot evidence.",
        ],
        "evidence_boundary": {
            "can_claim_now": [
                "Offline scorer quality is strong for both tasks.",
                "Foresight-gradient guidance path is ready for real rollout tests.",
                "Force-aware board consequence scorer has passed offline held-out gradient audit.",
                "Force-aware board consequence scorer has an optional serving arm whose dry-run smoke passed.",
                "Force-aware board serving arm has passed a stratified real-HDF5-window audit across five board labels.",
                "Force-aware board paired real-rollout manifest is prepared for three baseline/guided board pairs.",
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
            "insertion_config_consistency": str(args.insertion_config_consistency),
            "force_aware_board_audit": str(args.force_aware_board_audit),
            "force_aware_board_smoke": str(args.force_aware_board_smoke),
            "force_aware_board_denoise_smoke": str(args.force_aware_board_denoise_smoke),
            "force_aware_denoising_real_window": str(args.force_aware_denoising_real_window),
            "force_aware_serving_real_window": str(args.force_aware_serving_real_window),
            "force_aware_rollout_manifest": str(args.force_aware_rollout_manifest),
            "force_aware_rollout_coverage": str(args.force_aware_rollout_coverage),
            "force_aware_rollout_eval": str(args.force_aware_rollout_eval),
            "force_aware_weight_sweep": str(args.force_aware_weight_sweep),
            "force_aware_config_consistency": str(args.force_aware_config_consistency),
            "force_aware_label_separation": str(args.force_aware_label_separation),
        },
    }
    return scorecard


def write_markdown(scorecard: Mapping[str, Any], path: Path) -> None:
    ins = get(scorecard, "task_scorecards.insertion", {})
    board = get(scorecard, "task_scorecards.board", {})
    levels = get(scorecard, "evidence_levels", {})
    coverage = get(scorecard, "rollout_coverage", {})
    schema = get(scorecard, "server_rollout_schema", {})
    board_preference = get(scorecard, "recommendation.board", {})
    board_fallback = get(scorecard, "recommendation.board_integrated_fallback", {})
    weight_sweep_best = get(board, "force_aware_foresight_guidance.score_weight_sweep.best", {})
    weight_sweep_top = get(board, "force_aware_foresight_guidance.score_weight_sweep.top_presets", [])
    insertion_config_consistency = get(ins, "config_consistency", {})
    config_consistency = get(board, "force_aware_foresight_guidance.config_consistency", {})
    force_aware_denoise = get(board, "force_aware_foresight_guidance.serving_denoising_step_smoke", {})
    force_aware_denoise_real = get(board, "force_aware_foresight_guidance.denoising_real_window_audit", {})
    label_sep = get(board, "force_aware_foresight_guidance.label_separation", {})
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
        f"| board | `{board_preference.get('arm')}` | `{board_preference.get('runtime')}` | `{board_preference.get('score_preset')}` | `{get(board, 'force_aware_foresight_guidance.real_rollout_manifest.ready_for_collection')}` |",
        f"| board fallback/comparison | `{board_fallback.get('arm')}` | `{board_fallback.get('runtime')}` | `{board_fallback.get('score_mode')}` | `{board.get('ready_for_real_rollout')}` |",
        "",
        "Board scientific priority: `force_aware_foresight_quality_energy` "
        f"(offline gradient audit ready: `{get(board, 'force_aware_foresight_guidance.ready_for_research_guidance')}`, "
        f"serving smoke ready: `{get(board, 'force_aware_foresight_guidance.serving_smoke.ready_for_optional_server_trial')}`, "
        f"denoising-step smoke ready: `{get(board, 'force_aware_foresight_guidance.serving_denoising_step_smoke.ready_for_optional_server_trial')}`, "
        f"real-window serving ready: `{get(board, 'force_aware_foresight_guidance.serving_real_window_audit.ready_for_optional_server_trial')}`, "
        f"paired rollout manifest ready: `{get(board, 'force_aware_foresight_guidance.real_rollout_manifest.ready_for_collection')}`).",
        "",
        "Scientific board preference: "
        f"`{board_preference.get('arm')}` / `{board_preference.get('runtime')}` "
        f"with score preset `{board_preference.get('score_preset')}` "
        f"({board_preference.get('status')}). "
        f"{board_preference.get('why')}",
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
            f"score delta `{fnum(get(board, 'force_aware_foresight_guidance.score_delta.mean'))}`; "
            f"smoke score delta `{fnum(get(board, 'force_aware_foresight_guidance.serving_smoke.score_delta.mean'))}`, "
            f"denoise smoke score delta `{fnum(get(board, 'force_aware_foresight_guidance.serving_denoising_step_smoke.score_delta.mean'))}`, "
            f"real-window score delta `{fnum(get(board, 'force_aware_foresight_guidance.serving_real_window_audit.score_delta.mean'))}` |"
        ),
        "",
        "## Guidance Signal Strength",
        "",
        "This separates classification quality from whether the scorer provides a nontrivial denoising guidance signal.",
        "",
        "| scorer | status | score delta mean | action delta mean | threshold |",
        "|---|---|---:|---:|---|",
        (
            "| insertion good-margin | "
            f"`{get(ins, 'guidance_signal_strength.status')}` | "
            f"`{fnum(get(ins, 'guidance_signal_strength.score_delta_mean'), 6)}` | "
            f"`{fnum(get(ins, 'guidance_signal_strength.action_delta_mean'), 6)}` | "
            f"score >= `{get(ins, 'guidance_signal_strength.min_score_delta_mean')}`, "
            f"action >= `{get(ins, 'guidance_signal_strength.min_action_delta_mean')}` |"
        ),
        (
            "| board marker_joint_s12 deployable | "
            f"`{get(board, 'guidance_signal_strength.status')}` | "
            f"`{fnum(get(board, 'guidance_signal_strength.score_delta_mean'), 6)}` | "
            f"`{fnum(get(board, 'guidance_signal_strength.action_delta_mean'), 6)}` | "
            f"score >= `{get(board, 'guidance_signal_strength.min_score_delta_mean')}`, "
            f"action >= `{get(board, 'guidance_signal_strength.min_action_delta_mean')}` |"
        ),
        (
            "| board force-aware audit | "
            f"`{get(board, 'force_aware_foresight_guidance.guidance_signal_strength.status')}` | "
            f"`{fnum(get(board, 'force_aware_foresight_guidance.guidance_signal_strength.score_delta_mean'), 6)}` | "
            f"`{fnum(get(board, 'force_aware_foresight_guidance.guidance_signal_strength.action_delta_mean'), 6)}` | "
            f"score >= `{get(board, 'force_aware_foresight_guidance.guidance_signal_strength.min_score_delta_mean')}`, "
            f"action >= `{get(board, 'force_aware_foresight_guidance.guidance_signal_strength.min_action_delta_mean')}` |"
        ),
        (
            "| board force-aware real-window serving | "
            f"`{get(board, 'force_aware_foresight_guidance.serving_real_window_audit.guidance_signal_strength.status')}` | "
            f"`{fnum(get(board, 'force_aware_foresight_guidance.serving_real_window_audit.guidance_signal_strength.score_delta_mean'), 6)}` | "
            f"`{fnum(get(board, 'force_aware_foresight_guidance.serving_real_window_audit.guidance_signal_strength.action_delta_mean'), 6)}` | "
            f"score >= `{get(board, 'force_aware_foresight_guidance.serving_real_window_audit.guidance_signal_strength.min_score_delta_mean')}`, "
            f"action >= `{get(board, 'force_aware_foresight_guidance.serving_real_window_audit.guidance_signal_strength.min_action_delta_mean')}` |"
        ),
        "",
        "Interpretation: the deployable `marker_joint_s12_guided` board scorer remains useful for real A/B testing because it is integrated, "
        "but its current gradient update is numerically weak.  The force-aware scorer is the better scientific candidate for the final "
        "TacQuality guidance story because it produces a stronger bounded action update and directly scores force/contact consequences.",
        "",
        "## Force-Aware Denoising-Step Smoke",
        "",
        "This is the strictest current serving smoke for the board scorer: it verifies that the scorer is called inside the DP denoising loop "
        "on the predicted clean action `x0`, rather than as a post-hoc reranker.",
        "",
        f"- path: `{force_aware_denoise.get('path')}`",
        f"- pass: `{force_aware_denoise.get('pass')}`",
        f"- ready_for_optional_server_trial: `{force_aware_denoise.get('ready_for_optional_server_trial')}`",
        f"- runtime: `{force_aware_denoise.get('runtime')}`",
        f"- adapter_policy: `{force_aware_denoise.get('adapter_policy')}`",
        f"- guidance_location: `{force_aware_denoise.get('guidance_location')}`",
        f"- every_step_ddpm_guidance: `{force_aware_denoise.get('every_step_ddpm_guidance')}`",
        f"- not_reranking: `{force_aware_denoise.get('not_reranking')}`",
        f"- finite_grad_rate / positive_grad_rate / accept_rate: "
        f"`{fnum(force_aware_denoise.get('finite_grad_rate'))}` / "
        f"`{fnum(force_aware_denoise.get('positive_grad_rate'))}` / "
        f"`{fnum(force_aware_denoise.get('accept_rate'))}`",
        f"- score_delta_mean: `{fnum(get(force_aware_denoise, 'score_delta.mean'), 6)}`",
        f"- normalized_action_delta_mean: `{fnum(get(force_aware_denoise, 'normalized_action_delta.mean'), 6)}`",
        f"- max_delta_within_trust_region: `{force_aware_denoise.get('max_delta_within_trust_region')}`",
        f"- evidence boundary: {force_aware_denoise.get('evidence_boundary')}",
        "",
        "## Force-Aware Denoising Real-Window Audit",
        "",
        "This is stronger than the single synthetic smoke: it runs the same denoising-step guidance path on real board HDF5 image/qpos/tactile windows.",
        "",
        f"- path: `{force_aware_denoise_real.get('path')}`",
        f"- pass: `{force_aware_denoise_real.get('pass')}`",
        f"- ready_for_optional_server_trial: `{force_aware_denoise_real.get('ready_for_optional_server_trial')}`",
        f"- windows: `{force_aware_denoise_real.get('num_windows')}`",
        f"- labels: `{json.dumps(force_aware_denoise_real.get('labels'), ensure_ascii=False)}`",
        f"- scheduler / inference steps / guided steps: "
        f"`{force_aware_denoise_real.get('scheduler')}` / "
        f"`{force_aware_denoise_real.get('num_inference_steps')}` / "
        f"`{force_aware_denoise_real.get('ddpm_guidance_steps')}`",
        f"- runtime/location/not-reranking checks: "
        f"`{force_aware_denoise_real.get('runtime_is_force_aware')}` / "
        f"`{force_aware_denoise_real.get('denoising_location_ok')}` / "
        f"`{force_aware_denoise_real.get('not_reranking')}`",
        f"- finite_grad_rate / positive_grad_rate / accept_rate / trust_region: "
        f"`{fnum(force_aware_denoise_real.get('finite_grad_rate'))}` / "
        f"`{fnum(force_aware_denoise_real.get('positive_grad_rate'))}` / "
        f"`{fnum(force_aware_denoise_real.get('accept_rate'))}` / "
        f"`{fnum(force_aware_denoise_real.get('trust_region_pass_rate'))}`",
        f"- score_delta_mean: `{fnum(get(force_aware_denoise_real, 'score_delta.mean'), 6)}`",
        f"- normalized_action_delta_mean: `{fnum(get(force_aware_denoise_real, 'normalized_action_delta.mean'), 6)}`",
        f"- evidence boundary: {force_aware_denoise_real.get('evidence_boundary')}",
        "",
        "## Insertion Config Consistency",
        "",
        f"- audit path: `{insertion_config_consistency.get('path')}`",
        f"- pass: `{insertion_config_consistency.get('pass')}`",
        f"- expected arm/runtime/score mode: "
        f"`{get(insertion_config_consistency, 'expected.arm')}` / "
        f"`{get(insertion_config_consistency, 'expected.runtime')}` / "
        f"`{get(insertion_config_consistency, 'expected.score_mode')}`",
        f"- ablation good-margin delta mean: `{fnum(get(insertion_config_consistency, 'summary.ablation_good_margin_delta_mean'), 6)}`",
        f"- ablation p_good delta mean: `{fnum(get(insertion_config_consistency, 'summary.ablation_p_good_delta_mean'), 6)}`",
        f"- DDPM final-score improve rate: `{fnum(get(insertion_config_consistency, 'summary.ddpm_final_score_improve_rate'))}`",
        f"- final-action smoke score delta mean: `{fnum(get(insertion_config_consistency, 'summary.final_action_smoke_score_delta_mean'))}`",
        f"- denoising-step smoke score delta mean: `{fnum(get(insertion_config_consistency, 'summary.denoising_step_smoke_score_delta_mean'))}`",
        f"- evidence boundary: {insertion_config_consistency.get('evidence_boundary')}",
        "",
        "## Force-Aware Score Weight Sweep",
        "",
        f"- sweep path: `{get(board, 'force_aware_foresight_guidance.score_weight_sweep.path')}`",
        f"- best preset: `{weight_sweep_best.get('name')}`",
        f"- best weights: `{weight_sweep_best.get('weights')}`",
        f"- best ranking score: `{fnum(weight_sweep_best.get('ranking_score'))}`",
        f"- best improve / score delta / action delta: "
        f"`{fnum(weight_sweep_best.get('improved_rate'))}` / "
        f"`{fnum(weight_sweep_best.get('score_delta_mean'))}` / "
        f"`{fnum(weight_sweep_best.get('action_delta_norm_mean'))}`",
        "",
        "| rank | preset | ranking | improve | score delta | action delta | raw delta |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ])
    for i, row in enumerate(weight_sweep_top[:5], start=1):
        lines.append(
            f"| {i} | `{row.get('name')}` | `{fnum(row.get('ranking_score'))}` | "
            f"`{fnum(row.get('improved_rate'))}` | `{fnum(row.get('score_delta_mean'))}` | "
            f"`{fnum(row.get('action_delta_norm_mean'))}` | `{fnum(row.get('raw_action_delta_norm_mean'))}` |"
        )
    lines.extend([
        "",
        "Interpretation: on the full validation sweep, the plain force-band good-vs-risk margin is the strongest offline guidance score. "
        "Contact, force-center, force-smooth, and action-smooth penalties remain useful design hypotheses, but they did not improve "
        "the current offline guidance ranking and must be justified by paired real force_trace rollouts before becoming the default.",
        "",
        "## Force-Aware Label Separation",
        "",
        "This checks whether the selected board score matches the intended quality labels, not only whether it has gradients.",
        "",
        f"- path: `{label_sep.get('path')}`",
        f"- pass: `{label_sep.get('pass')}`",
        f"- split / samples: `{label_sep.get('split')}` / `{label_sep.get('num_samples')}`",
        f"- AUC / band bACC: `{fnum(get(label_sep, 'scorer_metrics.score_good_bad_auc'))}` / `{fnum(get(label_sep, 'scorer_metrics.band_balanced_acc'))}`",
        f"- good score mean: `{fnum(get(label_sep, 'separation.good_score_mean'))}`",
        f"- worst bad score mean: `{fnum(get(label_sep, 'separation.worst_bad_score_mean'))}`",
        f"- good-vs-worst-bad margin: `{fnum(get(label_sep, 'separation.good_vs_worst_bad_margin'))}`",
        f"- good prob / worst bad good prob: `{fnum(get(label_sep, 'separation.good_prob_mean'))}` / `{fnum(get(label_sep, 'separation.worst_bad_good_prob_mean'), 8)}`",
        f"- evidence boundary: {label_sep.get('evidence_boundary')}",
        "",
        "## Force-Aware Config Consistency",
        "",
        f"- audit path: `{config_consistency.get('path')}`",
        f"- pass: `{config_consistency.get('pass')}`",
        f"- expected preset: `{get(config_consistency, 'expected.score_preset')}`",
        f"- expected weights: `{get(config_consistency, 'expected.weights')}`",
        f"- serving score delta mean: `{fnum(get(config_consistency, 'summary.score_delta_mean'))}`",
        f"- serving normalized action delta mean: `{fnum(get(config_consistency, 'summary.normalized_action_delta_mean'))}`",
        f"- evidence boundary: {config_consistency.get('evidence_boundary')}",
        "",
        "## Real Rollout Coverage",
        "",
        f"- planned_counts: `{json.dumps(coverage.get('planned_counts'), ensure_ascii=False)}`",
        f"- observed_counts: `{json.dumps(coverage.get('observed_counts'), ensure_ascii=False)}`",
        f"- status_counts: `{json.dumps(coverage.get('status_counts'), ensure_ascii=False)}`",
        f"- real_rollout_evidence_complete: `{coverage.get('real_rollout_evidence_complete')}`",
        "",
        "## Force-Aware Board Rollout Manifest",
        "",
        f"- manifest ready: `{get(board, 'force_aware_foresight_guidance.real_rollout_manifest.ready_for_collection')}`",
        f"- real rollout complete: `{get(board, 'force_aware_foresight_guidance.real_rollout_manifest.real_rollout_complete')}`",
        f"- planned board pairs/trials: `{get(board, 'force_aware_foresight_guidance.real_rollout_manifest.board_pairs')}` / `{get(board, 'force_aware_foresight_guidance.real_rollout_manifest.n_trials')}`",
        f"- arms: baseline `{get(board, 'force_aware_foresight_guidance.real_rollout_manifest.baseline_arm')}`, guided `{get(board, 'force_aware_foresight_guidance.real_rollout_manifest.guided_arm')}`",
        f"- ports: baseline `{get(board, 'force_aware_foresight_guidance.real_rollout_manifest.baseline_port')}`, guided `{get(board, 'force_aware_foresight_guidance.real_rollout_manifest.guided_port')}`",
        f"- rollout root: `{get(board, 'force_aware_foresight_guidance.real_rollout_manifest.rollout_root')}`",
        f"- coverage status: `{json.dumps(get(board, 'force_aware_foresight_guidance.real_rollout_manifest.coverage_status_counts'), ensure_ascii=False)}`",
        f"- precheck detail: {get(board, 'force_aware_foresight_guidance.real_rollout_manifest.eval_detail')}",
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
    parser.add_argument("--insertion_config_consistency", type=Path, default=DEFAULT_INSERTION_CONFIG_CONSISTENCY)
    parser.add_argument("--force_aware_board_audit", type=Path, default=DEFAULT_FORCE_AWARE_BOARD_AUDIT)
    parser.add_argument("--force_aware_board_smoke", type=Path, default=DEFAULT_FORCE_AWARE_BOARD_SMOKE)
    parser.add_argument("--force_aware_board_denoise_smoke", type=Path, default=DEFAULT_FORCE_AWARE_BOARD_DENOISE_SMOKE)
    parser.add_argument("--force_aware_denoising_real_window", type=Path, default=DEFAULT_FORCE_AWARE_DENOISING_REAL_WINDOW)
    parser.add_argument("--force_aware_serving_real_window", type=Path, default=DEFAULT_FORCE_AWARE_SERVING_REAL_WINDOW)
    parser.add_argument("--force_aware_rollout_manifest", type=Path, default=DEFAULT_FORCE_AWARE_ROLLOUT_MANIFEST)
    parser.add_argument("--force_aware_rollout_coverage", type=Path, default=DEFAULT_FORCE_AWARE_ROLLOUT_COVERAGE)
    parser.add_argument("--force_aware_rollout_eval", type=Path, default=DEFAULT_FORCE_AWARE_ROLLOUT_EVAL)
    parser.add_argument("--force_aware_weight_sweep", type=Path, default=DEFAULT_FORCE_AWARE_WEIGHT_SWEEP)
    parser.add_argument("--force_aware_config_consistency", type=Path, default=DEFAULT_FORCE_AWARE_CONFIG_CONSISTENCY)
    parser.add_argument("--force_aware_label_separation", type=Path, default=DEFAULT_FORCE_AWARE_LABEL_SEPARATION)
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
