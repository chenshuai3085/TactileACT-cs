"""Build an auditable TacQuality scorer decision matrix.

This is a selection artifact, not a new training run.  It compresses the
existing episode-level evaluation, differentiability, local-guidance, and
real-rollout evidence into a single matrix that is easier to audit before
using a scorer for DP classifier guidance.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_scorer_decision_matrix")

PATHS = {
    "split_leakage_audit": Path(
        "/home/chenshuai/Project/output/tac_quality_split_leakage_audit/tac_quality_split_leakage_audit.json"
    ),
    "ptg_proxy_eval": Path("/home/chenshuai/Project/output/ptg_proxy_scorer_v2/ptg_proxy_scorer_v2_eval.json"),
    "distilled_eval": Path(
        "/home/chenshuai/Project/output/distilled_tac_quality_energy/distilled_tac_quality_energy_eval.json"
    ),
    "distilled_runtime": Path("/home/chenshuai/Project/output/distilled_tac_quality_energy/runtime_sanity.json"),
    "distilled_guidance": Path(
        "/home/chenshuai/Project/output/distilled_energy_guidance_comparison/distilled_energy_guidance_comparison.json"
    ),
    "insertion_clean": Path(
        "/home/chenshuai/Project/output/insertion_distilled_clean_refine_comparison/"
        "n24_k4/insertion_distilled_clean_refine_comparison.json"
    ),
    "board_dp_clean": Path(
        "/home/chenshuai/Project/output/board_dp_distilled_clean_refine_comparison/"
        "fast20_heldout32_n64/board_dp_distilled_clean_refine_comparison.json"
    ),
    "action_aware_eval": Path(
        "/home/chenshuai/Project/output/action_aware_marker_scorer/action_aware_marker_scorer_eval.json"
    ),
    "action_aware_runtime": Path(
        "/home/chenshuai/Project/output/action_aware_marker_scorer/runtime_gradient_sanity.json"
    ),
    "action_aware_guidance": Path(
        "/home/chenshuai/Project/output/action_aware_guidance_suitability/"
        "line_search_default/action_aware_guidance_suitability.json"
    ),
    "real_rollout_insertion": Path(
        "/home/chenshuai/Project/output/real_rollout_quality_gate/"
        "insertion_baseline_vs_guided/real_rollout_quality_gate.json"
    ),
    "real_rollout_board": Path(
        "/home/chenshuai/Project/output/real_rollout_quality_gate/"
        "board_baseline_vs_guided/real_rollout_quality_gate.json"
    ),
    "scorer_ablation_insertion": Path(
        "/home/chenshuai/Project/output/real_rollout_scorer_ablation_gate/"
        "insertion_baseline_vs_default_vs_distilled/real_rollout_scorer_ablation_gate.json"
    ),
    "scorer_ablation_board": Path(
        "/home/chenshuai/Project/output/real_rollout_scorer_ablation_gate/"
        "board_baseline_vs_default_vs_distilled/real_rollout_scorer_ablation_gate.json"
    ),
}

CRITERIA = [
    "episode_generalization",
    "quality_alignment",
    "differentiable_gradient",
    "local_guidance_stability",
    "task_coverage",
    "cross_task_transfer",
    "real_rollout_evidence",
    "innovation_value",
]


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


def metric(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def norm(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return clip01((value - low) / (high - low))


def bool_score(value: Any) -> float:
    return 1.0 if value is True else 0.0


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def real_evidence_score(*reports: Optional[Dict[str, Any]]) -> float:
    if not reports or all(r is None for r in reports):
        return 0.0
    passed = 0
    total = 0
    for report in reports:
        if report is None:
            total += 1
            continue
        total += 1
        if (
            get(report, "decision.production_validation_pass") is True
            or get(report, "production_ablation_pass") is True
        ) and get(report, "debug_or_underpowered", False) is False:
            passed += 1
    return passed / max(total, 1)


def weighted_score(row: Dict[str, Any], weights: Dict[str, float]) -> float:
    total = sum(weights.values())
    return float(sum(row["criteria_scores"][k] * weight for k, weight in weights.items()) / total)


def build_matrix(paths: Dict[str, Path]) -> Dict[str, Any]:
    data = {name: load_json(path) for name, path in paths.items()}
    split = data["split_leakage_audit"]
    ptg = data["ptg_proxy_eval"]
    distill = data["distilled_eval"]
    distill_runtime = data["distilled_runtime"]
    distill_guidance = data["distilled_guidance"]
    insertion_clean = data["insertion_clean"]
    board_dp = data["board_dp_clean"]
    action_eval = data["action_aware_eval"]
    action_runtime = data["action_aware_runtime"]
    action_guidance = data["action_aware_guidance"]

    best_teacher = (get(split, "best_candidates", []) or [{}])[0]
    rf_teacher_auc = metric(best_teacher.get("group_good_vs_bad_auc"))
    rf_teacher_bal = metric(best_teacher.get("group_balanced_accuracy"))
    rf_teacher_quality = metric(best_teacher.get("group_score_quality_spearman"))

    rows: List[Dict[str, Any]] = [
        {
            "scorer": "RFTeacher",
            "role": "offline_teacher_upper_bound",
            "deployment_status": "not_deployable_for_gradient_guidance",
            "criteria_scores": {
                "episode_generalization": norm(rf_teacher_bal, 0.70, 0.95),
                "quality_alignment": norm(rf_teacher_quality, 0.50, 0.90),
                "differentiable_gradient": 0.0,
                "local_guidance_stability": 0.0,
                "task_coverage": 1.0,
                "cross_task_transfer": 0.5,
                "real_rollout_evidence": real_evidence_score(data["real_rollout_insertion"], data["real_rollout_board"]),
                "innovation_value": 0.55,
            },
            "raw_metrics": {
                "group_balanced_accuracy": rf_teacher_bal,
                "group_auc": rf_teacher_auc,
                "group_quality_spearman": rf_teacher_quality,
                "differentiable": False,
            },
        },
        {
            "scorer": "InsertionRiskScorerRuntime",
            "role": "task_specific_insertion_default",
            "deployment_status": "formal_default_for_insertion_pending_real_rollout",
            "criteria_scores": {
                "episode_generalization": 0.88,
                "quality_alignment": 0.78,
                "differentiable_gradient": 1.0,
                "local_guidance_stability": bool_score(
                    get(insertion_clean, "insertion_risk.passes_insertion_clean_refine_smoke")
                ),
                "task_coverage": 0.5,
                "cross_task_transfer": 0.2,
                "real_rollout_evidence": real_evidence_score(data["real_rollout_insertion"]),
                "innovation_value": 0.50,
            },
            "raw_metrics": {
                "clean_refine_score_delta_mean": metric(get(insertion_clean, "insertion_risk.summary.score_delta.mean")),
                "clean_refine_improved_rate": metric(get(insertion_clean, "insertion_risk.summary.refined_beats_base_rate")),
                "smoothness_delta_mean": metric(get(insertion_clean, "insertion_risk.summary.smoothness_delta.mean")),
                "differentiable": True,
            },
        },
        {
            "scorer": "PTGProxyScorerV2Runtime",
            "role": "task_conditioned_current_board_default",
            "deployment_status": "formal_default_for_board_pending_real_rollout",
            "criteria_scores": {
                "episode_generalization": norm(metric(get(ptg, "mixed_group_cv.binary_auc.mean")), 0.85, 0.99),
                "quality_alignment": norm(metric(get(ptg, "mixed_group_cv.quality_corr.mean")), 0.50, 0.85),
                "differentiable_gradient": bool_score(get(ptg, "gradient_sanity.usable_for_feature_guidance")),
                "local_guidance_stability": bool_score(get(board_dp, "ptg_proxy_v2.passes_board_dp_clean_refine_smoke")),
                "task_coverage": 0.75,
                "cross_task_transfer": 0.55,
                "real_rollout_evidence": real_evidence_score(data["real_rollout_board"]),
                "innovation_value": 0.68,
            },
            "raw_metrics": {
                "mixed_auc": metric(get(ptg, "mixed_group_cv.binary_auc.mean")),
                "quality_corr": metric(get(ptg, "mixed_group_cv.quality_corr.mean")),
                "board_dp_score_delta_mean": metric(get(board_dp, "ptg_proxy_v2.summary.score_delta.mean")),
                "board_dp_improved_rate": metric(get(board_dp, "ptg_proxy_v2.summary.guided_beats_base_rate")),
                "differentiable": True,
            },
        },
        {
            "scorer": "DistilledTacQualityEnergyRuntime",
            "role": "differentiable_rf_teacher_energy_ablation",
            "deployment_status": "promoted_ablation_candidate_pending_real_rollout",
            "criteria_scores": {
                "episode_generalization": norm(metric(get(distill, "mixed_episode_group_cv.energy_binary_auc.mean")), 0.85, 0.99),
                "quality_alignment": norm(metric(get(distill, "mixed_episode_group_cv.energy_quality_spearman.mean")), 0.45, 0.75),
                "differentiable_gradient": min(
                    bool_score(get(distill, "guidance_sanity.usable_for_feature_guidance"))
                    + bool_score(get(distill_runtime, "usable_for_guidance")),
                    1.0,
                ),
                "local_guidance_stability": np.mean(
                    [
                        bool_score(get(distill_guidance, "distilled_energy.passes_local_guidance")),
                        bool_score(get(insertion_clean, "distilled_energy.passes_insertion_clean_refine_smoke")),
                        bool_score(get(board_dp, "distilled_energy.passes_board_dp_clean_refine_smoke")),
                    ]
                ).item(),
                "task_coverage": 1.0,
                "cross_task_transfer": 0.70,
                "real_rollout_evidence": real_evidence_score(
                    data["scorer_ablation_insertion"], data["scorer_ablation_board"]
                ),
                "innovation_value": 0.90,
            },
            "raw_metrics": {
                "energy_binary_auc": metric(get(distill, "mixed_episode_group_cv.energy_binary_auc.mean")),
                "teacher_pred_corr": metric(get(distill, "mixed_episode_group_cv.teacher_pred_corr.mean")),
                "energy_teacher_spearman": metric(get(distill, "mixed_episode_group_cv.energy_teacher_spearman.mean")),
                "energy_quality_spearman": metric(get(distill, "mixed_episode_group_cv.energy_quality_spearman.mean")),
                "insertion_score_delta_mean": metric(get(insertion_clean, "distilled_energy.summary.score_delta.mean")),
                "board_dp_score_delta_mean": metric(get(board_dp, "distilled_energy.summary.score_delta.mean")),
                "differentiable": True,
            },
        },
        {
            "scorer": "ActionAwareScorerRuntime",
            "role": "unified_action_conditioned_line_search_candidate",
            "deployment_status": "optional_fourth_arm_candidate_pending_real_rollout",
            "criteria_scores": {
                "episode_generalization": norm(metric(get(action_eval, "mixed_group_cv.binary_auc.mean")), 0.85, 0.99),
                "quality_alignment": norm(metric(get(action_eval, "mixed_group_cv.score_corr.mean")), 0.50, 0.85),
                "differentiable_gradient": bool_score(get(action_runtime, "usable_for_guidance")),
                "local_guidance_stability": bool_score(get(action_guidance, "passes_guidance_suitability")),
                "task_coverage": 1.0,
                "cross_task_transfer": np.mean(
                    [
                        norm(metric(get(action_eval, "cross_task.insertion_to_board.binary_macro_f1")), 0.40, 0.75),
                        norm(metric(get(action_eval, "cross_task.board_to_insertion.binary_macro_f1")), 0.40, 0.75),
                    ]
                ).item(),
                "real_rollout_evidence": 0.0,
                "innovation_value": 0.95,
            },
            "raw_metrics": {
                "mixed_auc": metric(get(action_eval, "mixed_group_cv.binary_auc.mean")),
                "score_corr": metric(get(action_eval, "mixed_group_cv.score_corr.mean")),
                "cross_insertion_to_board_macro_f1": metric(
                    get(action_eval, "cross_task.insertion_to_board.binary_macro_f1")
                ),
                "cross_board_to_insertion_macro_f1": metric(
                    get(action_eval, "cross_task.board_to_insertion.binary_macro_f1")
                ),
                "recommended_mode": get(action_guidance, "recommended_mode"),
                "line_search_accepted_rate": metric(
                    get(action_guidance, "modes.quality.gradient_probe.mixed.line_search.accepted_rate")
                ),
                "differentiable": True,
            },
        },
    ]

    weights = {
        "episode_generalization": 1.15,
        "quality_alignment": 1.05,
        "differentiable_gradient": 1.25,
        "local_guidance_stability": 1.20,
        "task_coverage": 0.70,
        "cross_task_transfer": 0.55,
        "real_rollout_evidence": 1.45,
        "innovation_value": 0.65,
    }
    for row in rows:
        row["offline_score_without_real_rollout"] = weighted_score(
            row,
            {k: v for k, v in weights.items() if k != "real_rollout_evidence"},
        )
        row["deployment_score_with_real_rollout"] = weighted_score(row, weights)

    ranked_offline = sorted(rows, key=lambda r: r["offline_score_without_real_rollout"], reverse=True)
    ranked_deploy = sorted(rows, key=lambda r: r["deployment_score_with_real_rollout"], reverse=True)
    result = {
        "name": "TacQuality scorer decision matrix",
        "purpose": (
            "Auditable scorer selection for DP classifier guidance on socket insertion and board wiping."
        ),
        "scientific_evidence": False,
        "git_commit": git_commit(),
        "criteria": CRITERIA,
        "weights": weights,
        "rows": rows,
        "ranked_offline": [
            {
                "rank": i + 1,
                "scorer": row["scorer"],
                "score": row["offline_score_without_real_rollout"],
                "role": row["role"],
            }
            for i, row in enumerate(ranked_offline)
        ],
        "ranked_deployment": [
            {
                "rank": i + 1,
                "scorer": row["scorer"],
                "score": row["deployment_score_with_real_rollout"],
                "role": row["role"],
            }
            for i, row in enumerate(ranked_deploy)
        ],
        "recommendation": {
            "formal_default_insertion": "InsertionRiskScorerRuntime",
            "formal_default_board": "PTGProxyScorerV2Runtime",
            "innovation_ablation": "DistilledTacQualityEnergyRuntime",
            "optional_unified_action_conditioned_ablation": "ActionAwareScorerRuntime",
            "reason": (
                "Distilled and ActionAware are stronger innovation candidates, but formal deployment choice "
                "cannot be closed because real rollout evidence is still missing.  Keep task-specific "
                "defaults for formal gates and compare distilled/action-aware in paired ablations."
            ),
        },
        "passes_decision_matrix_gate": (
            ranked_offline[0]["offline_score_without_real_rollout"] >= 0.70
            and any(row["scorer"] == "DistilledTacQualityEnergyRuntime" for row in ranked_offline[:3])
            and any(row["scorer"] == "ActionAwareScorerRuntime" for row in ranked_offline[:3])
            and all(row["criteria_scores"]["real_rollout_evidence"] == 0.0 for row in rows)
        ),
        "remaining_gap": "Real baseline-vs-guided and scorer-ablation rollout evidence is required before final scorer promotion.",
        "paths": {name: str(path) for name, path in paths.items()},
    }
    return result


def plot_heatmap(result: Dict[str, Any], path: Path) -> None:
    labels = [row["scorer"] for row in result["rows"]]
    mat = np.array([[row["criteria_scores"][criterion] for criterion in CRITERIA] for row in result["rows"]])
    fig, ax = plt.subplots(figsize=(12, 5.8))
    im = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(np.arange(len(CRITERIA)))
    ax.set_xticklabels([c.replace("_", "\n") for c in CRITERIA], fontsize=8)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title("TacQuality scorer decision matrix")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color="white", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_markdown(result: Dict[str, Any], path: Path) -> None:
    lines = [
        "# TacQuality Scorer Decision Matrix",
        "",
        f"- passes_decision_matrix_gate: `{result['passes_decision_matrix_gate']}`",
        f"- scientific_evidence: `{result['scientific_evidence']}`",
        f"- remaining_gap: {result['remaining_gap']}",
        "",
        "## Recommendation",
        "",
    ]
    for key, value in result["recommendation"].items():
        lines.append(f"- {key}: `{value}`" if key != "reason" else f"- reason: {value}")
    lines.extend(
        [
            "",
            "## Offline Ranking",
            "",
            "| rank | scorer | score | role |",
            "|---:|---|---:|---|",
        ]
    )
    for row in result["ranked_offline"]:
        lines.append(f"| {row['rank']} | {row['scorer']} | {row['score']:.4f} | {row['role']} |")
    lines.extend(
        [
            "",
            "## Criterion Scores",
            "",
            "| scorer | "
            + " | ".join(c.replace("_", " ") for c in CRITERIA)
            + " | offline score | deployment score |",
            "|---|" + "---:|" * (len(CRITERIA) + 2),
        ]
    )
    for row in result["rows"]:
        vals = " | ".join(f"{row['criteria_scores'][c]:.3f}" for c in CRITERIA)
        lines.append(
            f"| {row['scorer']} | {vals} | "
            f"{row['offline_score_without_real_rollout']:.4f} | "
            f"{row['deployment_score_with_real_rollout']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- RFTeacher remains useful as an offline teacher but cannot guide DP gradients directly.",
            "- Task-specific defaults remain the formal rollout choices until real evidence exists.",
            "- DistilledTacQualityEnergy is the most defensible novelty ablation because it distills a strong teacher into a differentiable energy.",
            "- ActionAwareScorerRuntime is conceptually closest to action-gradient guidance, but weak zero-shot transfer keeps it optional.",
            "",
            "## Figure",
            "",
            "- `tac_quality_scorer_decision_matrix.png`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default=str(OUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = build_matrix(PATHS)
    json_path = out_dir / "tac_quality_scorer_decision_matrix.json"
    md_path = out_dir / "tac_quality_scorer_decision_matrix.md"
    fig_path = out_dir / "tac_quality_scorer_decision_matrix.png"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    plot_heatmap(result, fig_path)
    print(
        json.dumps(
            {
                "passes_decision_matrix_gate": result["passes_decision_matrix_gate"],
                "best_offline": result["ranked_offline"][0],
                "best_deployment_current_evidence": result["ranked_deployment"][0],
                "json": str(json_path),
                "markdown": str(md_path),
                "figure": str(fig_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
