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
    "tac_quality_rollout_arm_configs_current_s12_good_margin_20260619.json"
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


def build_audit(args: argparse.Namespace) -> dict[str, Any]:
    scorer = load_json(args.scorer_audit)
    scorecard = load_json(args.scorecard)
    bundle = load_json(args.evidence_bundle)
    rollout_config = load_json(args.rollout_config)

    board_metrics = get(scorer, "key_metrics.board", {})
    insertion_metrics = get(scorer, "key_metrics.insertion", {})
    board_grad = get(board_metrics, "gradient", {})
    board_align = get(board_metrics, "foresight_alignment", {})
    insertion_grad = get(insertion_metrics, "matched_0401_gradient", {})
    rollout_counts = get(bundle, "real_rollout_coverage.status_counts", {})

    board_score_delta = float(get(board_grad, "score_delta.mean", 0.0) or 0.0)
    board_action_delta = float(get(board_grad, "action_delta_norm.mean", 0.0) or 0.0)
    board_pred_gt_rho = float(get(board_align, "pred_gt_spearman", 0.0) or 0.0)
    board_force_rho = float(get(board_align, "pred_score_vs_force_band_quality_spearman", 0.0) or 0.0)
    board_force_auc = float(get(board_align, "force_band_quality_auc_good", 0.0) or 0.0)
    insertion_score_delta = float(get(insertion_grad, "score_delta.mean", 0.0) or 0.0)
    insertion_action_delta = float(get(insertion_grad, "action_delta_norm.mean", 0.0) or 0.0)
    missing_rollouts = int(rollout_counts.get("missing", 0) or 0)

    board_gaps = [
        {
            "gap": "real paired board force evidence missing",
            "risk": "high" if missing_rollouts else "low",
            "evidence": f"rollout_status_counts={rollout_counts}",
            "action": "Run paired baseline/guided board rollouts and evaluate force_trace.csv.",
        },
        {
            "gap": "board clean-action guidance magnitude is small",
            "risk": "medium" if board_score_delta < 1e-3 else "low",
            "evidence": f"score_delta_mean={board_score_delta:.8f}, action_delta_norm_mean={board_action_delta:.8f}",
            "action": "Sweep board score modes / step sizes inside DDPM-step guidance and require nontrivial score/action deltas under trust-region limits.",
        },
        {
            "gap": "board force-band continuous alignment is only moderate",
            "risk": "medium" if board_force_rho < 0.50 else "low",
            "evidence": f"pred_score_vs_force_band_quality_spearman={board_force_rho:.4f}, pred_gt_spearman={board_pred_gt_rho:.4f}",
            "action": "Train or audit a force-aware Foresight head or force-proxy head so the score targets force-band quality more directly.",
        },
        {
            "gap": "force-band quality alone does not explain good-label AUC",
            "risk": "medium" if board_force_auc < 0.60 else "low",
            "evidence": f"force_band_quality_auc_good={board_force_auc:.4f}",
            "action": "Keep semantic labels and force metrics separate in the paper; do not claim force-band metric alone defines board success.",
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
            "evidence": "p_good/log_p_good deltas are zero in score-mode ablation; good_margin improve_rate=0.9375.",
            "action": "Keep good_margin as default and treat p_good as a reporting metric, not guidance objective.",
        },
        {
            "gap": "insertion reason classifier is weaker than binary classifier",
            "risk": "medium",
            "evidence": f"reason_macro_f1={fnum(get(insertion_metrics, 'reason_macro_f1'))}, binary_auc={fnum(get(insertion_metrics, 'binary_auc'))}",
            "action": "For paper claims, emphasize good-vs-risk guidance; use reason labels mainly for interpretation unless reason F1 improves.",
        },
    ]

    priorities = [
        {
            "rank": 1,
            "experiment": "paired real rollout evaluation",
            "why": "It is the only missing evidence level and blocks the final claim.",
            "success_criterion": "At least 3 complete baseline/guided pairs per task; board force metrics and insertion outcomes improve without safety regressions.",
        },
        {
            "rank": 2,
            "experiment": "board DDPM-step guidance sweep with stronger but bounded settings",
            "why": "Current board gradient is finite but very small in clean-action audit.",
            "success_criterion": "Positive score delta with meaningful action_delta_norm, trust_region_pass_rate>=0.999, and no final-score regression after accept filtering.",
        },
        {
            "rank": 3,
            "experiment": "force-aware board Foresight / force-proxy scorer",
            "why": "Board quality is physically force-band based, but current runtime relies on marker/action proxies and marker-only Foresight.",
            "success_criterion": "Improve pred_score_vs_force_band_quality_spearman beyond 0.50 and preserve held-out episode-level classification.",
        },
        {
            "rank": 4,
            "experiment": "insertion reason-head refinement",
            "why": "Binary guidance is strong; reason labels are less reliable but useful for interpretability.",
            "success_criterion": "Improve GroupKFold reason_macro_f1 without reducing binary AUC or good_margin gradient quality.",
        },
    ]

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "Gap audit for current TacQuality scorer/guidance candidates.",
        "current_candidates": {
            "board": get(scorer, "current_recommendation.board", {}),
            "insertion": get(scorer, "current_recommendation.insertion", {}),
        },
        "summary": {
            "offline_scorer_ready": get(scorecard, "evidence_levels.offline_scorer_ready"),
            "gradient_guidance_ready": get(scorecard, "evidence_levels.gradient_guidance_ready"),
            "server_rollout_schema_ready": get(scorecard, "evidence_levels.server_rollout_schema_ready"),
            "real_paired_rollout_complete": get(scorecard, "evidence_levels.real_paired_rollout_complete"),
            "missing_rollouts": missing_rollouts,
            "board_score_delta_mean": board_score_delta,
            "board_action_delta_norm_mean": board_action_delta,
            "board_pred_gt_spearman": board_pred_gt_rho,
            "board_force_band_spearman": board_force_rho,
            "board_force_band_auc_good": board_force_auc,
            "insertion_score_delta_mean": insertion_score_delta,
            "insertion_action_delta_norm_mean": insertion_action_delta,
        },
        "board_gaps": board_gaps,
        "insertion_gaps": insertion_gaps,
        "priority_experiments": priorities,
        "rollout_config": {
            "recommended_board_arm": get(rollout_config, "recommended_board_arm"),
            "recommended_insertion_arm": get(rollout_config, "recommended_insertion_arm"),
            "board_score_mode": get(
                rollout_config,
                f"tasks.board.{get(rollout_config, 'recommended_board_arm')}.refiner.score_mode",
            ),
            "insertion_score_mode": get(
                rollout_config,
                f"tasks.insertion.{get(rollout_config, 'recommended_insertion_arm')}.refiner.score_mode",
            ),
        },
        "evidence_paths": {
            "scorer_audit": str(args.scorer_audit),
            "scorecard": str(args.scorecard),
            "evidence_bundle": str(args.evidence_bundle),
            "rollout_config": str(args.rollout_config),
        },
        "claim_boundary": {
            "can_claim": [
                "The current offline task-specific scorers are strong enough for controlled real rollout tests.",
                "The current Foresight-to-score-to-action gradient path is finite and trust-region bounded.",
                "The implementation path is gradient guidance, not offline reranking.",
            ],
            "cannot_claim": [
                "Real robot improvement over baseline has not been proven.",
                "Board force-curve improvement has not been proven.",
                "Insertion success/bounce/retry improvement has not been proven.",
                "The board scorer is not yet a direct force-prediction scorer; it is a marker/action proxy scorer trained from force-band labels.",
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
        f"| offline_scorer_ready | `{summary['offline_scorer_ready']}` |",
        f"| gradient_guidance_ready | `{summary['gradient_guidance_ready']}` |",
        f"| server_rollout_schema_ready | `{summary['server_rollout_schema_ready']}` |",
        f"| real_paired_rollout_complete | `{summary['real_paired_rollout_complete']}` |",
        f"| missing real rollout records | `{summary['missing_rollouts']}` |",
        "",
        "## Board",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| clean-action score_delta mean | `{fnum(summary['board_score_delta_mean'], 8)}` |",
        f"| clean-action action_delta_norm mean | `{fnum(summary['board_action_delta_norm_mean'], 8)}` |",
        f"| Foresight pred-vs-GT score Spearman | `{fnum(summary['board_pred_gt_spearman'])}` |",
        f"| pred score vs force-band quality Spearman | `{fnum(summary['board_force_band_spearman'])}` |",
        f"| force-band quality AUC good | `{fnum(summary['board_force_band_auc_good'])}` |",
        "",
        "| gap | risk | evidence | next action |",
        "|---|---|---|---|",
    ]
    for row in result["board_gaps"]:
        lines.append(f"| {row['gap']} | `{row['risk']}` | {row['evidence']} | {row['action']} |")

    lines.extend(
        [
            "",
            "## Insertion",
            "",
            "| metric | value |",
            "|---|---:|",
            f"| matched 0401 score_delta mean | `{fnum(summary['insertion_score_delta_mean'])}` |",
            f"| matched 0401 action_delta_norm mean | `{fnum(summary['insertion_action_delta_norm_mean'])}` |",
            "",
            "| gap | risk | evidence | next action |",
            "|---|---|---|---|",
        ]
    )
    for row in result["insertion_gaps"]:
        lines.append(f"| {row['gap']} | `{row['risk']}` | {row['evidence']} | {row['action']} |")

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
                "board_score_delta_mean": result["summary"]["board_score_delta_mean"],
                "board_force_band_spearman": result["summary"]["board_force_band_spearman"],
                "top_priority": result["priority_experiments"][0]["experiment"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
