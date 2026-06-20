#!/usr/bin/env python3
"""Audit the current task-specific TacQuality scorer recommendations.

This audit is intentionally evidence-bound.  It reads existing experiment JSON
artifacts and writes a compact current-state report for the two active tasks:

* socket insertion: InsertionRiskScorerRuntime, score_mode=good_margin
* board wiping: ForceBandTacQualityEnergyRuntime(marker_joint_action,s12),
  score_mode=quality

It does not retrain models and does not claim real robot improvement.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


DEFAULT_INSERTION_EVAL = Path("/home/chenshuai/Project/output/insertion_risk_scorer/insertion_risk_scorer_eval.json")
DEFAULT_INSERTION_SCORE_ABLATION = Path(
    "/home/chenshuai/Project/output/tac_quality_score_mode_ablation/"
    "insertion_0401_profile_pgood_energy_goodmargin_cross_score_20260619/"
    "insertion_score_mode_ablation.json"
)
DEFAULT_INSERTION_GRAD_0209 = Path(
    "/home/chenshuai/Project/output/insertion_guidance_gradient_audit_0209_matched_20260619/"
    "guidance_gradient_audit.json"
)
DEFAULT_INSERTION_GRAD_0401 = Path(
    "/home/chenshuai/Project/output/insertion_guidance_gradient_audit_0401_matched_20260619/"
    "guidance_gradient_audit.json"
)
DEFAULT_BOARD_EVAL = Path(
    "/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260619_s12/"
    "train_result.json"
)
DEFAULT_BOARD_ALIGNMENT = Path(
    "/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260619_s12/"
    "foresight_alignment_quality/foresight_score_alignment.json"
)
DEFAULT_BOARD_GRAD = Path(
    "/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260619_s12/"
    "guidance_gradient_audit_quality/guidance_gradient_audit.json"
)
DEFAULT_BOARD_FEATURES = Path(
    "/home/chenshuai/Project/output/tac_quality_board_force_band_eval_mlp/"
    "board_force_band_scorer_eval.json"
)
DEFAULT_CONFIG = Path(
    "/home/chenshuai/Project/output/tac_quality_rollout_arm_configs/"
    "tac_quality_rollout_arm_configs_current_s12_good_margin_20260619.json"
)
DEFAULT_OUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_current_scorer_audit")
DEFAULT_DOC = Path("docs/2026-06-20_current_tac_quality_scorer_audit.md")


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
    return {"_error": "not a JSON object", "_path": str(path)}


def get(data: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def metric_mean(data: Mapping[str, Any], *keys: str) -> Any:
    value = get(data, *keys)
    if isinstance(value, Mapping) and "mean" in value:
        return value["mean"]
    return value


def passed(value: Any, threshold: float) -> bool:
    try:
        return float(value) >= threshold
    except Exception:
        return False


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def status_row(name: str, value: Any, threshold: float) -> dict[str, Any]:
    return {
        "name": name,
        "value": value,
        "threshold": threshold,
        "pass": passed(value, threshold),
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    insertion_eval = load_json(args.insertion_eval)
    insertion_ablation = load_json(args.insertion_score_ablation)
    insertion_grad_0209 = load_json(args.insertion_grad_0209)
    insertion_grad_0401 = load_json(args.insertion_grad_0401)
    board_eval = load_json(args.board_eval)
    board_alignment = load_json(args.board_alignment)
    board_grad = load_json(args.board_grad)
    board_features = load_json(args.board_features)
    rollout_config = load_json(args.rollout_config)

    insertion_cv = get(insertion_eval, "mixed_group_cv", default={})
    insertion_good_margin = get(insertion_ablation, "summary", "good_margin", default={})
    insertion_cross = get(insertion_good_margin, "cross_scores", default={})

    board_best = get(board_eval, "best", "val", default={})
    board_summary = get(board_alignment, "summary", default={})
    board_grad_summary = get(board_grad, "summary", default={})
    feature_labels = get(board_features, "label_counts", default={})
    quality_ref = get(board_features, "quality_reference", default={})

    checks = {
        "insertion_offline": [
            status_row("binary_auc", metric_mean(insertion_cv, "binary_auc"), 0.95),
            status_row("binary_balanced_accuracy", metric_mean(insertion_cv, "binary_balanced_accuracy"), 0.90),
            status_row("quality_corr", metric_mean(insertion_cv, "quality_corr"), 0.70),
        ],
        "insertion_gradient": [
            status_row("matched_0209_improved_rate", get(insertion_grad_0209, "summary", "improved_rate_mean"), 0.80),
            status_row("matched_0209_finite_grad", get(insertion_grad_0209, "summary", "finite_grad_rate_mean"), 0.999),
            status_row("matched_0401_improved_rate", get(insertion_grad_0401, "summary", "improved_rate_mean"), 0.80),
            status_row("matched_0401_finite_grad", get(insertion_grad_0401, "summary", "finite_grad_rate_mean"), 0.999),
        ],
        "board_offline": [
            status_row("binary_auc", board_best.get("binary_auc"), 0.98),
            status_row("binary_balanced_accuracy", board_best.get("binary_balanced_accuracy"), 0.95),
            status_row("reason_macro_f1", board_best.get("reason_macro_f1"), 0.95),
            status_row("quality_spearman", board_best.get("quality_spearman"), 0.90),
        ],
        "board_foresight": [
            status_row("pred_auc_good", board_summary.get("pred_auc_good"), 0.95),
            status_row("pred_gt_spearman", board_summary.get("pred_gt_spearman"), 0.45),
            status_row("pred_score_vs_force_band_quality_spearman", board_summary.get("pred_score_vs_force_band_quality_spearman"), 0.30),
        ],
        "board_gradient": [
            status_row("improved_rate", board_grad_summary.get("improved_rate_mean"), 0.80),
            status_row("finite_grad", board_grad_summary.get("finite_grad_rate_mean"), 0.999),
            status_row("trust_region", board_grad_summary.get("trust_region_pass_rate"), 0.999),
        ],
    }

    result = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "Current task-specific TacQuality scorer audit for DP classifier/scorer guidance.",
        "current_recommendation": {
            "insertion": {
                "arm": "good_margin_guided",
                "runtime": "InsertionRiskScorerRuntime",
                "checkpoint": "/home/chenshuai/Project/output/insertion_risk_scorer/insertion_risk_scorer_final.pt",
                "score_mode": "good_margin",
                "label_definition": {
                    "good": "good_insert",
                    "bad": ["pre_bounce_risk", "impact_or_recovery"],
                    "neutral_excluded_from_binary": "weak_approach",
                },
                "why": "good_margin is an unsaturated logit margin; p_good separated labels but saturated in matched DDPM/Foresight evaluation.",
            },
            "board": {
                "arm": "marker_joint_s12_guided",
                "runtime": "ForceBandTacQualityEnergyRuntime",
                "checkpoint": str(args.board_eval.parent / "force_band_tac_quality_energy_best.pt"),
                "score_mode": "quality",
                "label_definition": {
                    "good": "positive",
                    "bad": ["too_small", "too_large", "oscillate"],
                    "quality": get(quality_ref, "definition"),
                },
                "why": "Four-class force-band board scorer covers too-light, too-heavy, and unstable contact while staying deployable from marker_joint_action features.",
            },
        },
        "checks": checks,
        "pass_summary": {
            key: all(bool(row["pass"]) for row in rows)
            for key, rows in checks.items()
        },
        "overall_offline_guidance_ready": all(
            bool(row["pass"]) for rows in checks.values() for row in rows
        ),
        "production_validated_by_real_rollouts": False,
        "key_metrics": {
            "insertion": {
                "binary_auc": metric_mean(insertion_cv, "binary_auc"),
                "binary_balanced_accuracy": metric_mean(insertion_cv, "binary_balanced_accuracy"),
                "reason_macro_f1": metric_mean(insertion_cv, "reason_macro_f1"),
                "quality_corr": metric_mean(insertion_cv, "quality_corr"),
                "good_margin_cross_score": {
                    "n_rows": get(insertion_good_margin, "n_rows"),
                    "final_accept_rate": get(insertion_good_margin, "final_accept_rate"),
                    "action_delta_norm_mean": get(insertion_good_margin, "guided_action_delta_norm", "mean"),
                    "profile_delta_mean": get(insertion_cross, "profile", "delta", "mean"),
                    "energy_delta_mean": get(insertion_cross, "energy", "delta", "mean"),
                    "good_margin_delta_mean": get(insertion_cross, "good_margin", "delta", "mean"),
                    "good_margin_improve_rate": get(insertion_cross, "good_margin", "improve_rate"),
                    "quality_logit_delta_mean": get(insertion_cross, "quality_logit", "delta", "mean"),
                    "quality_logit_delta_min": get(insertion_cross, "quality_logit", "delta", "min"),
                },
                "matched_0209_gradient": get(insertion_grad_0209, "summary", default={}),
                "matched_0401_gradient": get(insertion_grad_0401, "summary", default={}),
            },
            "board": {
                "heldout": board_best,
                "foresight_alignment": board_summary,
                "gradient": board_grad_summary,
                "feature_label_counts": feature_labels,
                "quality_by_label": get(quality_ref, "quality_by_label", default={}),
            },
        },
        "evidence_paths": {
            "insertion_eval": str(args.insertion_eval),
            "insertion_score_ablation": str(args.insertion_score_ablation),
            "insertion_grad_0209": str(args.insertion_grad_0209),
            "insertion_grad_0401": str(args.insertion_grad_0401),
            "board_eval": str(args.board_eval),
            "board_alignment": str(args.board_alignment),
            "board_grad": str(args.board_grad),
            "board_features": str(args.board_features),
            "rollout_config": str(args.rollout_config),
        },
        "rollout_config_exists": not bool(rollout_config.get("_missing")),
        "evidence_boundary": [
            "Offline scorer and Foresight-gradient evidence are ready for controlled real rollout tests.",
            "No formal paired baseline-vs-guided robot rollout gate is present here.",
            "Do not claim real success-rate, bounce-rate, or wiping-force improvement until paired robot logs are evaluated.",
        ],
        "next_required_evidence": [
            "Insertion paired baseline vs good_margin_guided rollouts with success/bounce/retry metadata.",
            "Board paired baseline vs marker_joint_s12_guided rollouts with server-side force curves.",
            "Use the same DP checkpoint per task when comparing guided vs unguided.",
        ],
    }
    return result


def write_markdown(result: Mapping[str, Any], path: Path) -> None:
    ins = result["key_metrics"]["insertion"]
    board = result["key_metrics"]["board"]
    board_held = board["heldout"]
    board_align = board["foresight_alignment"]
    board_grad = board["gradient"]
    lines = [
        "# Current TacQuality Scorer Audit",
        "",
        f"Generated: `{result['created_at']}`",
        "",
        "## Current Recommendation",
        "",
        "| task | arm | runtime | score mode | role |",
        "|---|---|---|---|---|",
        "| insertion | `good_margin_guided` | `InsertionRiskScorerRuntime` | `good_margin` | unsaturated good-vs-risk logit margin |",
        "| board | `marker_joint_s12_guided` | `ForceBandTacQualityEnergyRuntime` | `quality` | four-class force-band tactile quality |",
        "",
        "This replaces the older interpretation that the distilled/manual-board scorer is the main board candidate.  The distilled scorer remains an ablation; the current board default is the four-class ForceBand scorer.",
        "",
        "## Pass Summary",
        "",
        "| category | pass |",
        "|---|---:|",
    ]
    for key, value in result["pass_summary"].items():
        lines.append(f"| {key} | `{value}` |")
    lines.extend(
        [
            f"| overall offline guidance ready | `{result['overall_offline_guidance_ready']}` |",
            f"| production validated by real rollouts | `{result['production_validated_by_real_rollouts']}` |",
            "",
            "## Key Metrics",
            "",
            "| task | binary AUC | bACC | reason F1 | quality metric | gradient evidence |",
            "|---|---:|---:|---:|---:|---|",
            f"| insertion | {fmt(ins['binary_auc'])} | {fmt(ins['binary_balanced_accuracy'])} | {fmt(ins['reason_macro_f1'])} | corr {fmt(ins['quality_corr'])} | 0209 improve {fmt(get(ins, 'matched_0209_gradient', 'improved_rate_mean'))}, 0401 improve {fmt(get(ins, 'matched_0401_gradient', 'improved_rate_mean'))} |",
            f"| board | {fmt(board_held.get('binary_auc'))} | {fmt(board_held.get('binary_balanced_accuracy'))} | {fmt(board_held.get('reason_macro_f1'))} | rho {fmt(board_held.get('quality_spearman'))} | improve {fmt(board_grad.get('improved_rate_mean'))}, finite {fmt(board_grad.get('finite_grad_rate_mean'))} |",
            "",
            "## Insertion Good-Margin Cross-Score Ablation",
            "",
            f"- rows: `{fmt(get(ins, 'good_margin_cross_score', 'n_rows'), 0)}`",
            f"- final accept rate: `{fmt(get(ins, 'good_margin_cross_score', 'final_accept_rate'))}`",
            f"- guided action delta norm mean: `{fmt(get(ins, 'good_margin_cross_score', 'action_delta_norm_mean'), 8)}`",
            f"- own good-margin delta mean: `{fmt(get(ins, 'good_margin_cross_score', 'good_margin_delta_mean'), 8)}`",
            f"- good-margin improve rate: `{fmt(get(ins, 'good_margin_cross_score', 'good_margin_improve_rate'))}`",
            f"- profile delta mean: `{fmt(get(ins, 'good_margin_cross_score', 'profile_delta_mean'), 8)}`",
            f"- energy delta mean: `{fmt(get(ins, 'good_margin_cross_score', 'energy_delta_mean'), 8)}`",
            f"- quality-logit delta mean/min: `{fmt(get(ins, 'good_margin_cross_score', 'quality_logit_delta_mean'), 8)}` / `{fmt(get(ins, 'good_margin_cross_score', 'quality_logit_delta_min'), 8)}`",
            "",
            "## Board Four-Class Coverage",
            "",
            "| label | count | quality mean |",
            "|---|---:|---:|",
        ]
    )
    counts = board.get("feature_label_counts", {})
    q_by = board.get("quality_by_label", {})
    for label in sorted(counts):
        lines.append(f"| {label} | {counts[label]} | {fmt(get(q_by, label, 'mean'))} |")
    lines.extend(
        [
            "",
            "Board labels:",
            "",
            "- `positive`: proper contact force band and smooth wiping.",
            "- `too_small`: pressure too small / insufficient wiping.",
            "- `too_large`: pressure too large.",
            "- `oscillate`: unstable force/contact transition.",
            "",
            "## Foresight-Chain Board Alignment",
            "",
            f"- pred AUC good: `{fmt(board_align.get('pred_auc_good'))}`",
            f"- GT AUC good: `{fmt(board_align.get('gt_auc_good'))}`",
            f"- predicted-vs-GT score Spearman: `{fmt(board_align.get('pred_gt_spearman'))}`",
            f"- predicted score vs force-band quality Spearman: `{fmt(board_align.get('pred_score_vs_force_band_quality_spearman'))}`",
            "",
            "## Evidence Boundary",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in result["evidence_boundary"])
    lines.extend(["", "## Next Required Evidence", ""])
    lines.extend(f"- {item}" for item in result["next_required_evidence"])
    lines.extend(["", "## Evidence Paths", ""])
    for key, value in result["evidence_paths"].items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--insertion_eval", type=Path, default=DEFAULT_INSERTION_EVAL)
    parser.add_argument("--insertion_score_ablation", type=Path, default=DEFAULT_INSERTION_SCORE_ABLATION)
    parser.add_argument("--insertion_grad_0209", type=Path, default=DEFAULT_INSERTION_GRAD_0209)
    parser.add_argument("--insertion_grad_0401", type=Path, default=DEFAULT_INSERTION_GRAD_0401)
    parser.add_argument("--board_eval", type=Path, default=DEFAULT_BOARD_EVAL)
    parser.add_argument("--board_alignment", type=Path, default=DEFAULT_BOARD_ALIGNMENT)
    parser.add_argument("--board_grad", type=Path, default=DEFAULT_BOARD_GRAD)
    parser.add_argument("--board_features", type=Path, default=DEFAULT_BOARD_FEATURES)
    parser.add_argument("--rollout_config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    args = parser.parse_args()

    result = audit(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "current_tac_quality_scorer_audit.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(result, args.doc)
    print(f"Saved: {out_json}")
    print(f"Saved: {args.doc}")
    print(f"overall_offline_guidance_ready={result['overall_offline_guidance_ready']}")


if __name__ == "__main__":
    main()
