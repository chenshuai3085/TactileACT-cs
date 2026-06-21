#!/usr/bin/env python3
"""Build a gap audit for the current TacQuality DP guidance candidate.

This script does not retrain models.  It reads the current scorer/guidance
evidence artifacts and writes a concise, evidence-bound audit of what is strong,
what is still weak, and which experiment should be run next.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


DEFAULT_SCORER_AUDIT = Path(
    "/home/chenshuai/Project/output/tac_quality_current_scorer_audit/current_tac_quality_scorer_audit.json"
)
DEFAULT_SCORECARD = Path(
    "/home/chenshuai/Project/output/tac_quality_current_scorecard/current_tac_quality_scorecard.json"
)
DEFAULT_EVIDENCE_BUNDLE = Path(
    "/home/chenshuai/Project/output/tac_quality_evidence_bundle/current_tac_quality_evidence_bundle.json"
)
DEFAULT_ROLLOUT_CONFIG = Path(
    "/home/chenshuai/Project/output/tac_quality_rollout_arm_configs/"
    "tac_quality_rollout_arm_configs_current_s12_good_margin_forceaware_board_20260621.json"
)
DEFAULT_MAIN_ROLLOUT_COVERAGE = Path(
    "/home/chenshuai/Project/output/tac_quality_real_rollout_coverage/"
    "current_s12_good_margin_coverage/tac_quality_real_rollout_coverage.json"
)
DEFAULT_FORCE_AWARE_ROLLOUT_COVERAGE = Path(
    "/home/chenshuai/Project/output/tac_quality_real_rollout_coverage/"
    "board_force_aware_coverage/tac_quality_real_rollout_coverage.json"
)
DEFAULT_OUTPUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_guidance_gap_audit")
DEFAULT_DOC = Path("docs/2026-06-20_tac_quality_guidance_gap_audit.md")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"_missing": True, "_path": str(path)}
    data = json.loads(path.read_text(encoding="utf-8"))
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


def fnum(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def maybe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def risk_level(value: float, warn: float, high: float, *, larger_is_worse: bool = True) -> str:
    if larger_is_worse:
        if value >= high:
            return "high"
        if value >= warn:
            return "medium"
        return "low"
    if value <= high:
        return "high"
    if value <= warn:
        return "medium"
    return "low"


def coverage_summary(data: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(data, Mapping):
        return {}
    summary = data.get("summary")
    if isinstance(summary, Mapping):
        return summary
    return data


def missing_count(data: Mapping[str, Any] | None) -> int:
    summary = coverage_summary(data)
    counts = summary.get("status_counts", {})
    if not isinstance(counts, Mapping):
        return 0
    return int(counts.get("missing", 0) or 0)


def bool_status(value: Any) -> str:
    if value is True:
        return "pass"
    if value is False:
        return "missing"
    if value is None:
        return "unknown"
    return str(value)


def build_audit(args: argparse.Namespace) -> dict[str, Any]:
    scorer = load_json(args.scorer_audit)
    scorecard = load_json(args.scorecard)
    bundle = load_json(args.evidence_bundle)
    rollout_config = load_json(args.rollout_config)
    main_coverage = load_json(args.main_rollout_coverage)
    force_aware_coverage = load_json(args.force_aware_rollout_coverage)

    evidence = get(scorecard, "evidence_levels", {}) or {}
    insertion = get(scorecard, "task_scorecards.insertion", {}) or {}
    board = get(scorecard, "task_scorecards.board", {}) or {}
    board_force = get(board, "force_aware_foresight_guidance", {}) or {}
    rollout_counts = get(scorecard, "rollout_coverage.status_counts", {}) or {}
    main_cov_summary = coverage_summary(main_coverage)
    force_cov_summary = coverage_summary(force_aware_coverage)

    # Fallback to the older scorer audit only when the current scorecard is absent.
    if not board and not insertion:
        board_metrics = get(scorer, "key_metrics.board", {})
        insertion_metrics = get(scorer, "key_metrics.insertion", {})
        board = {
            "guidance_signal_strength": {
                "score_delta_mean": get(board_metrics, "gradient.score_delta.mean", 0.0),
                "action_delta_mean": get(board_metrics, "gradient.action_delta_norm.mean", 0.0),
                "strong_signal": False,
            },
            "foresight_alignment": {
                "pred_gt_spearman": get(board_metrics, "foresight_alignment.pred_gt_spearman", 0.0),
                "pred_score_vs_force_band_quality_spearman": get(
                    board_metrics,
                    "foresight_alignment.pred_score_vs_force_band_quality_spearman",
                    0.0,
                ),
                "force_band_quality_auc_good": get(
                    board_metrics,
                    "foresight_alignment.force_band_quality_auc_good",
                    0.0,
                ),
            },
        }
        insertion = {
            "offline_metrics": {
                "binary_auc": get(insertion_metrics, "binary_auc"),
                "reason_macro_f1": get(insertion_metrics, "reason_macro_f1"),
            },
            "guidance_signal_strength": {
                "score_delta_mean": get(insertion_metrics, "matched_0401_gradient.score_delta.mean", 0.0),
                "action_delta_mean": get(insertion_metrics, "matched_0401_gradient.action_delta_norm.mean", 0.0),
            },
        }

    board_signal = get(board, "guidance_signal_strength", {}) or {}
    board_align = get(board, "foresight_alignment", {}) or {}
    insertion_signal = get(insertion, "guidance_signal_strength", {}) or {}
    insertion_offline = get(insertion, "offline_metrics", {}) or {}
    insertion_config = get(insertion, "config_consistency", {}) or {}
    force_signal = get(board_force, "guidance_signal_strength", {}) or {}
    force_sweep_best = get(board_force, "score_weight_sweep.best", {}) or {}
    force_real_window = get(board_force, "serving_real_window_audit", {}) or {}

    board_score_delta = float(get(board_signal, "score_delta_mean", 0.0) or 0.0)
    board_action_delta = float(get(board_signal, "action_delta_mean", 0.0) or 0.0)
    board_pred_gt_rho = float(get(board_align, "pred_gt_spearman", 0.0) or 0.0)
    board_force_rho = float(get(board_align, "pred_score_vs_force_band_quality_spearman", 0.0) or 0.0)
    board_force_auc = maybe_float(get(board_align, "force_band_quality_auc_good"))
    insertion_score_delta = float(get(insertion_signal, "score_delta_mean", 0.0) or 0.0)
    insertion_action_delta = float(get(insertion_signal, "action_delta_mean", 0.0) or 0.0)
    force_score_delta = float(get(force_signal, "score_delta_mean", 0.0) or 0.0)
    force_action_delta = float(get(force_signal, "action_delta_mean", 0.0) or 0.0)
    force_sweep_score_delta = float(get(force_sweep_best, "score_delta_mean", 0.0) or 0.0)
    force_window_score_delta = float(
        get(
            force_real_window,
            "score_delta.mean",
            get(force_real_window, "guidance_signal_strength.score_delta_mean", 0.0),
        )
        or 0.0
    )
    missing_rollouts = int(rollout_counts.get("missing", missing_count(main_coverage)) or 0)
    force_missing_rollouts = missing_count(force_aware_coverage)

    board_deploy_gaps = [
        {
            "gap": "deploy board real force evidence missing",
            "risk": "high" if missing_rollouts else "low",
            "evidence": f"main_rollout_status_counts={rollout_counts}",
            "action": "Run the marker_joint_s12 board baseline/guided pairs only if this deployable arm remains a candidate.",
        },
        {
            "gap": "deploy board guidance signal is too weak for the current objective",
            "risk": "high" if not get(board_signal, "strong_signal", False) else "low",
            "evidence": f"score_delta_mean={board_score_delta:.8f}, action_delta_norm_mean={board_action_delta:.8f}",
            "action": "Do not present marker_joint_s12 as the scientific board solution; keep it as a deployable baseline unless stronger real force traces prove otherwise.",
        },
        {
            "gap": "deploy board score is not a direct force-consequence scorer",
            "risk": "medium" if board_force_rho < 0.50 else "low",
            "evidence": f"pred_score_vs_force_band_quality_spearman={board_force_rho:.4f}, pred_gt_spearman={board_pred_gt_rho:.4f}",
            "action": "Use the force-aware Foresight branch for the main board guidance story.",
        },
        {
            "gap": "force-band quality alone does not explain good-label AUC",
            "risk": "medium" if board_force_auc is None or board_force_auc < 0.60 else "low",
            "evidence": f"force_band_quality_auc_good={fnum(board_force_auc)}",
            "action": "Keep semantic labels and force metrics separate in the paper; do not claim force-band metric alone defines board success.",
        },
    ]

    board_force_gaps = [
        {
            "gap": "force-aware board real rollout evidence missing",
            "risk": "high" if force_missing_rollouts else "low",
            "evidence": f"force_aware_rollout_status_counts={get(force_cov_summary, 'status_counts', {})}",
            "action": "Run three paired baseline/force_aware_guided board trials and evaluate force_trace.csv.",
        },
        {
            "gap": "force-aware board is offline/serving-ready but not robot-proven",
            "risk": "medium",
            "evidence": (
                f"config_consistent={evidence.get('force_aware_board_config_consistent')}, "
                f"serving_real_window_ready={evidence.get('force_aware_board_real_window_serving_ready')}, "
                f"score_delta_mean={force_window_score_delta:.4f}"
            ),
            "action": "Claim only preflight readiness until real paired force traces show improvement.",
        },
        {
            "gap": "force-aware score currently selects margin-only objective",
            "risk": "low",
            "evidence": (
                f"best_preset={get(force_sweep_best, 'name')}, "
                f"sweep_score_delta_mean={force_sweep_score_delta:.4f}, "
                f"score_weights={get(scorecard, 'recommendation.board_scientific_preference.score_weights', {})}"
            ),
            "action": "Keep smooth/contact penalties as hypotheses for real-force evaluation; do not assume they improve deployment before force traces.",
        },
    ]

    insertion_gaps = [
        {
            "gap": "real paired insertion success/bounce evidence missing",
            "risk": "high" if missing_rollouts else "low",
            "evidence": f"rollout_status_counts={rollout_counts}",
            "action": "Run paired baseline/good_margin_guided insertion rollouts with success, bounce_count, retry_count, and stopped_early metadata.",
        },
        {
            "gap": "probability heads saturate; use logit margin for guidance",
            "risk": "low",
            "evidence": (
                f"config_pass={get(insertion_config, 'pass')}, "
                f"score_mode={get(insertion, 'score_mode')}, "
                f"good_margin_improve_rate={fnum(get(insertion, 'guidance_metrics.good_margin_improve_rate'))}"
            ),
            "action": "Keep good_margin as default and treat p_good as a reporting metric, not guidance objective.",
        },
        {
            "gap": "insertion reason classifier is weaker than binary classifier",
            "risk": "medium",
            "evidence": (
                f"reason_macro_f1={fnum(get(insertion_offline, 'reason_macro_f1'))}, "
                f"binary_auc={fnum(get(insertion_offline, 'binary_auc'))}"
            ),
            "action": "For paper claims, emphasize good-vs-risk guidance; use reason labels mainly for interpretation unless reason F1 improves.",
        },
    ]

    priorities = [
        {
            "rank": 1,
            "experiment": "force-aware board paired real rollout evaluation",
            "why": "It is the strongest board scientific candidate and the missing evidence is real force_trace improvement.",
            "success_criterion": "At least 3 complete baseline/force_aware_guided pairs; Fz band occupancy and smoothness improve without task/safety regression.",
        },
        {
            "rank": 2,
            "experiment": "insertion paired real rollout evaluation",
            "why": "Good-margin insertion is config-consistent and has strong offline/serving signal; it still lacks success/bounce/retry evidence.",
            "success_criterion": "At least 3 complete baseline/good_margin_guided pairs; success increases or bounce/retry decreases with complete metadata.",
        },
        {
            "rank": 3,
            "experiment": "board action_horizon/reactivity ablation",
            "why": "Recent work emphasizes reactive tactile policies; current action_horizon=8 may be slow for contact correction.",
            "success_criterion": "Compare action_horizon 4/6/8 under identical force-aware scoring; select the shortest horizon that preserves trajectory completion and improves force smoothness.",
        },
        {
            "rank": 4,
            "experiment": "force-aware score penalty validation",
            "why": "Offline sweep favored margin_only, but smoothness/contact penalties are still physically meaningful hypotheses.",
            "success_criterion": "Use real force traces to decide whether margin_only, margin_smooth, or margin_contact best improves Fz smoothness and contact continuity.",
        },
    ]

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "Gap audit for current TacQuality scorer/guidance candidates, keyed to the latest scorecard.",
        "current_candidates": {
            "insertion": get(scorecard, "recommendation.insertion", {}),
            "board_deploy": get(scorecard, "recommendation.board", {}),
            "board_scientific_preference": get(scorecard, "recommendation.board_scientific_preference", {}),
        },
        "summary": {
            "offline_scorer_ready": evidence.get("offline_scorer_ready"),
            "gradient_guidance_ready": evidence.get("gradient_guidance_ready"),
            "server_rollout_schema_ready": evidence.get("server_rollout_schema_ready"),
            "insertion_guidance_signal_strong": evidence.get("insertion_guidance_signal_strong"),
            "insertion_config_consistent": evidence.get("insertion_config_consistent"),
            "board_deploy_guidance_signal_strong": evidence.get("board_deploy_guidance_signal_strong"),
            "force_aware_board_guidance_signal_strong": evidence.get("force_aware_board_guidance_signal_strong"),
            "force_aware_board_real_window_serving_ready": evidence.get("force_aware_board_real_window_serving_ready"),
            "force_aware_board_config_consistent": evidence.get("force_aware_board_config_consistent"),
            "force_aware_board_real_rollout_complete": evidence.get("force_aware_board_real_rollout_complete"),
            "real_paired_rollout_complete": evidence.get("real_paired_rollout_complete"),
            "goal_complete": evidence.get("goal_complete"),
            "missing_rollouts": missing_rollouts,
            "force_aware_missing_rollouts": force_missing_rollouts,
            "board_score_delta_mean": board_score_delta,
            "board_action_delta_norm_mean": board_action_delta,
            "board_pred_gt_spearman": board_pred_gt_rho,
            "board_force_band_spearman": board_force_rho,
            "board_force_band_auc_good": board_force_auc,
            "force_aware_score_delta_mean": force_score_delta,
            "force_aware_action_delta_mean": force_action_delta,
            "force_aware_sweep_score_delta_mean": force_sweep_score_delta,
            "force_aware_real_window_score_delta_mean": force_window_score_delta,
            "insertion_score_delta_mean": insertion_score_delta,
            "insertion_action_delta_norm_mean": insertion_action_delta,
        },
        "board_deploy_gaps": board_deploy_gaps,
        "board_force_aware_gaps": board_force_gaps,
        "insertion_gaps": insertion_gaps,
        "priority_experiments": priorities,
        "rollout_config": {
            "recommended_board_arm": get(rollout_config, "recommended_board_arm"),
            "recommended_insertion_arm": get(rollout_config, "recommended_insertion_arm"),
            "force_aware_board_arm_present": get(rollout_config, "tasks.board.force_aware_guided.arm") == "force_aware_guided",
            "board_score_mode": get(
                rollout_config,
                f"tasks.board.{get(rollout_config, 'recommended_board_arm')}.refiner.score_mode",
            ),
            "force_aware_score_preset": get(rollout_config, "tasks.board.force_aware_guided.refiner.energy.score_preset"),
            "insertion_score_mode": get(
                rollout_config,
                f"tasks.insertion.{get(rollout_config, 'recommended_insertion_arm')}.refiner.score_mode",
            ),
        },
        "rollout_coverage": {
            "main": main_cov_summary,
            "force_aware_board": force_cov_summary,
        },
        "evidence_paths": {
            "scorer_audit": str(args.scorer_audit),
            "scorecard": str(args.scorecard),
            "evidence_bundle": str(args.evidence_bundle),
            "rollout_config": str(args.rollout_config),
            "main_rollout_coverage": str(args.main_rollout_coverage),
            "force_aware_rollout_coverage": str(args.force_aware_rollout_coverage),
        },
        "claim_boundary": {
            "can_claim": get(scorecard, "evidence_boundary.can_claim_now", [])
            or [
                "Offline task-specific scorers are ready for controlled real rollout tests.",
                "The guidance implementation is gradient-based and bounded, not offline reranking.",
            ],
            "cannot_claim": get(scorecard, "evidence_boundary.cannot_claim_yet", [])
            or [
                "Real robot improvement over baseline has not been proven.",
                "Board force-curve improvement has not been proven.",
                "Insertion success/bounce/retry improvement has not been proven.",
            ],
        },
    }


def write_markdown(result: Mapping[str, Any], path: Path) -> None:
    summary = result["summary"]
    lines = [
        "# TacQuality Guidance Gap Audit",
        "",
        f"Generated: `{result['created_at']}`",
        "",
        "## Current State",
        "",
        "| item | value |",
        "|---|---:|",
        f"| offline_scorer_ready | `{bool_status(summary['offline_scorer_ready'])}` |",
        f"| gradient_guidance_ready | `{bool_status(summary['gradient_guidance_ready'])}` |",
        f"| server_rollout_schema_ready | `{bool_status(summary['server_rollout_schema_ready'])}` |",
        f"| insertion_config_consistent | `{bool_status(summary['insertion_config_consistent'])}` |",
        f"| force_aware_board_config_consistent | `{bool_status(summary['force_aware_board_config_consistent'])}` |",
        f"| real_paired_rollout_complete | `{bool_status(summary['real_paired_rollout_complete'])}` |",
        f"| goal_complete | `{bool_status(summary['goal_complete'])}` |",
        f"| main missing real rollout records | `{summary['missing_rollouts']}` |",
        f"| force-aware board missing real rollout records | `{summary['force_aware_missing_rollouts']}` |",
        "",
        "## Insertion Good-Margin",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| guidance signal strong | `{bool_status(summary['insertion_guidance_signal_strong'])}` |",
        f"| matched/offline score_delta mean | `{fnum(summary['insertion_score_delta_mean'])}` |",
        f"| matched/offline action_delta_norm mean | `{fnum(summary['insertion_action_delta_norm_mean'])}` |",
        "",
        "| gap | risk | evidence | next action |",
        "|---|---|---|---|",
    ]
    for row in result["insertion_gaps"]:
        lines.append(f"| {row['gap']} | `{row['risk']}` | {row['evidence']} | {row['action']} |")

    lines.extend(
        [
            "",
            "## Board Deploy Candidate",
            "",
            "This is the deployable marker/action scorer, not the preferred scientific board scorer.",
            "",
            "| metric | value |",
            "|---|---:|",
            f"| guidance signal strong | `{bool_status(summary['board_deploy_guidance_signal_strong'])}` |",
            f"| clean-action score_delta mean | `{fnum(summary['board_score_delta_mean'], 8)}` |",
            f"| clean-action action_delta_norm mean | `{fnum(summary['board_action_delta_norm_mean'], 8)}` |",
            f"| Foresight pred-vs-GT score Spearman | `{fnum(summary['board_pred_gt_spearman'])}` |",
            f"| pred score vs force-band quality Spearman | `{fnum(summary['board_force_band_spearman'])}` |",
            f"| force-band quality AUC good | `{fnum(summary['board_force_band_auc_good'])}` |",
            "",
            "| gap | risk | evidence | next action |",
            "|---|---|---|---|",
        ]
    )
    for row in result["board_deploy_gaps"]:
        lines.append(f"| {row['gap']} | `{row['risk']}` | {row['evidence']} | {row['action']} |")

    lines.extend(
        [
            "",
            "## Board Force-Aware Scientific Candidate",
            "",
            "This is the current preferred research candidate for board wiping because it scores predicted force/contact consequences.",
            "",
            "| metric | value |",
            "|---|---:|",
            f"| guidance signal strong | `{bool_status(summary['force_aware_board_guidance_signal_strong'])}` |",
            f"| real-window serving ready | `{bool_status(summary['force_aware_board_real_window_serving_ready'])}` |",
            f"| offline score_delta mean | `{fnum(summary['force_aware_score_delta_mean'])}` |",
            f"| offline action_delta_norm mean | `{fnum(summary['force_aware_action_delta_mean'])}` |",
            f"| weight-sweep best score_delta mean | `{fnum(summary['force_aware_sweep_score_delta_mean'])}` |",
            f"| real-HDF5-window serving score_delta mean | `{fnum(summary['force_aware_real_window_score_delta_mean'])}` |",
            f"| real rollout complete | `{bool_status(summary['force_aware_board_real_rollout_complete'])}` |",
            "",
            "| gap | risk | evidence | next action |",
            "|---|---|---|---|",
        ]
    )
    for row in result["board_force_aware_gaps"]:
        lines.append(f"| {row['gap']} | `{row['risk']}` | {row['evidence']} | {row['action']} |")

    lines.extend(
        [
            "",
            "## Rollout Config",
            "",
            "| item | value |",
            "|---|---|",
            f"| recommended_board_arm | `{get(result, 'rollout_config.recommended_board_arm')}` |",
            f"| recommended_insertion_arm | `{get(result, 'rollout_config.recommended_insertion_arm')}` |",
            f"| force_aware_board_arm_present | `{get(result, 'rollout_config.force_aware_board_arm_present')}` |",
            f"| board_score_mode | `{get(result, 'rollout_config.board_score_mode')}` |",
            f"| force_aware_score_preset | `{get(result, 'rollout_config.force_aware_score_preset')}` |",
            f"| insertion_score_mode | `{get(result, 'rollout_config.insertion_score_mode')}` |",
        ]
    )

    lines.extend(
        [
            "",
            "## Priority Experiments",
            "",
            "| rank | experiment | why | success criterion |",
            "|---:|---|---|---|",
        ]
    )
    for row in result["priority_experiments"]:
        lines.append(f"| {row['rank']} | {row['experiment']} | {row['why']} | {row['success_criterion']} |")

    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "Can claim now:",
        ]
    )
    lines.extend(f"- {item}" for item in result["claim_boundary"]["can_claim"])
    lines.extend(["", "Cannot claim yet:"])
    lines.extend(f"- {item}" for item in result["claim_boundary"]["cannot_claim"])
    lines.extend(["", "## Evidence Paths"])
    lines.extend(f"- {key}: `{value}`" for key, value in result["evidence_paths"].items())

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scorer_audit", type=Path, default=DEFAULT_SCORER_AUDIT)
    parser.add_argument("--scorecard", type=Path, default=DEFAULT_SCORECARD)
    parser.add_argument("--evidence_bundle", type=Path, default=DEFAULT_EVIDENCE_BUNDLE)
    parser.add_argument("--rollout_config", type=Path, default=DEFAULT_ROLLOUT_CONFIG)
    parser.add_argument("--main_rollout_coverage", type=Path, default=DEFAULT_MAIN_ROLLOUT_COVERAGE)
    parser.add_argument("--force_aware_rollout_coverage", type=Path, default=DEFAULT_FORCE_AWARE_ROLLOUT_COVERAGE)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    args = parser.parse_args()

    result = build_audit(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "tac_quality_guidance_gap_audit.json"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(result, args.doc)
    print(
        json.dumps(
            {
                "json": str(json_path),
                "markdown": str(args.doc),
                "real_paired_rollout_complete": result["summary"]["real_paired_rollout_complete"],
                "missing_rollouts": result["summary"]["missing_rollouts"],
                "force_aware_missing_rollouts": result["summary"]["force_aware_missing_rollouts"],
                "board_score_delta_mean": result["summary"]["board_score_delta_mean"],
                "force_aware_real_window_score_delta_mean": result["summary"][
                    "force_aware_real_window_score_delta_mean"
                ],
                "top_priority": result["priority_experiments"][0]["experiment"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
