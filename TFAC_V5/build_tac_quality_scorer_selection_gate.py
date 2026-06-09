"""Build a scorer-selection gate for TacQuality DP gradient guidance.

This gate aggregates the current classification, differentiability, and
offline guidance evidence into one machine-readable decision.  It deliberately
separates three roles:

  - RF teacher: strongest non-differentiable classifier/soft-label source;
  - PTGProxyV2: current default differentiable board scorer;
  - DistilledTacQualityEnergy: differentiable RF-teacher energy candidate.

The output is a promotion decision for offline/robot-ablation planning, not a
claim that real baseline-vs-guided rollouts have passed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional


OUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_scorer_selection_gate")

PATHS = {
    "split_leakage_audit": Path(
        "/home/chenshuai/Project/output/tac_quality_split_leakage_audit/tac_quality_split_leakage_audit.json"
    ),
    "ptg_proxy_eval": Path("/home/chenshuai/Project/output/ptg_proxy_scorer_v2/ptg_proxy_scorer_v2_eval.json"),
    "distilled_eval": Path(
        "/home/chenshuai/Project/output/distilled_tac_quality_energy/distilled_tac_quality_energy_eval.json"
    ),
    "distilled_runtime": Path("/home/chenshuai/Project/output/distilled_tac_quality_energy/runtime_sanity.json"),
    "feature_guidance_comparison": Path(
        "/home/chenshuai/Project/output/distilled_energy_guidance_comparison/distilled_energy_guidance_comparison.json"
    ),
    "insertion_clean_refine_comparison": Path(
        "/home/chenshuai/Project/output/insertion_distilled_clean_refine_comparison/"
        "n24_k4/insertion_distilled_clean_refine_comparison.json"
    ),
    "board_surrogate_comparison": Path(
        "/home/chenshuai/Project/output/board_surrogate_distilled_comparison/board_surrogate_distilled_comparison.json"
    ),
    "board_dp_clean_refine_comparison": Path(
        "/home/chenshuai/Project/output/board_dp_distilled_clean_refine_comparison/"
        "fast20_heldout32_n64/board_dp_distilled_clean_refine_comparison.json"
    ),
    "real_rollout_insertion": Path(
        "/home/chenshuai/Project/output/real_rollout_quality_gate/"
        "insertion_baseline_vs_guided/real_rollout_quality_gate.json"
    ),
    "real_rollout_board": Path(
        "/home/chenshuai/Project/output/real_rollout_quality_gate/"
        "board_baseline_vs_guided/real_rollout_quality_gate.json"
    ),
}


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


def file_info(path: Path) -> Dict[str, Any]:
    exists = path.exists()
    return {
        "path": str(path),
        "exists": bool(exists),
        "bytes": int(path.stat().st_size) if exists and path.is_file() else None,
    }


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def metric(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def build_gate(paths: Dict[str, Path]) -> Dict[str, Any]:
    data = {name: load_json(path) for name, path in paths.items()}
    split = data["split_leakage_audit"]
    ptg = data["ptg_proxy_eval"]
    distill = data["distilled_eval"]
    runtime = data["distilled_runtime"]
    feature = data["feature_guidance_comparison"]
    insertion_clean = data["insertion_clean_refine_comparison"]
    surrogate = data["board_surrogate_comparison"]
    board_dp = data["board_dp_clean_refine_comparison"]

    best = (get(split, "best_candidates", []) or [{}])[0]

    evidence = {
        "rf_teacher": {
            "role": "offline_teacher_and_upper_bound",
            "differentiable": False,
            "episode_group_balanced_accuracy": metric(best.get("group_balanced_accuracy")),
            "episode_group_auc": metric(best.get("group_good_vs_bad_auc")),
            "episode_group_quality_spearman": metric(best.get("group_score_quality_spearman")),
            "frame_random_minus_group_balanced_accuracy": metric(best.get("random_minus_group_bal_acc")),
            "source": str(paths["split_leakage_audit"]),
        },
        "ptg_proxy_v2": {
            "role": "current_default_board_scorer",
            "differentiable": True,
            "episode_group_binary_auc": metric(get(ptg, "mixed_group_cv.binary_auc.mean")),
            "episode_group_balanced_accuracy": metric(get(ptg, "mixed_group_cv.binary_balanced_accuracy.mean")),
            "episode_group_quality_corr": metric(get(ptg, "mixed_group_cv.quality_corr.mean")),
            "feature_gradient_usable": bool(get(ptg, "gradient_sanity.usable_for_feature_guidance", False)),
            "feature_guidance": {
                "passes": bool(get(feature, "ptg_proxy_v2.passes_local_guidance", False)),
                "improved_rate": metric(get(feature, "ptg_proxy_v2.recommended_improved_rate")),
                "score_delta_mean": metric(get(feature, "ptg_proxy_v2.recommended_score_delta_mean")),
            },
            "board_surrogate": {
                "passes": bool(get(surrogate, "ptg_proxy_v2.passes_board_surrogate_action_refinement", False)),
                "improved_rate": metric(get(surrogate, "ptg_proxy_v2.summary.score_improved_rate")),
                "score_delta_mean": metric(get(surrogate, "ptg_proxy_v2.summary.score_delta.mean")),
            },
            "board_dp_clean_refine": {
                "passes": bool(get(board_dp, "ptg_proxy_v2.passes_board_dp_clean_refine_smoke", False)),
                "improved_rate": metric(get(board_dp, "ptg_proxy_v2.summary.guided_beats_base_rate")),
                "score_delta_mean": metric(get(board_dp, "ptg_proxy_v2.summary.score_delta.mean")),
                "smoothness_delta_mean": metric(get(board_dp, "ptg_proxy_v2.summary.smoothness_delta.mean")),
                "range_violation_max": metric(get(board_dp, "ptg_proxy_v2.summary.range_violation.max")),
            },
        },
        "insertion_risk": {
            "role": "current_default_insertion_scorer",
            "differentiable": True,
            "insertion_clean_refine": {
                "passes": bool(get(insertion_clean, "insertion_risk.passes_insertion_clean_refine_smoke", False)),
                "improved_rate": metric(get(insertion_clean, "insertion_risk.summary.refined_beats_base_rate")),
                "score_delta_mean": metric(get(insertion_clean, "insertion_risk.summary.score_delta.mean")),
                "smoothness_delta_mean": metric(get(insertion_clean, "insertion_risk.summary.smoothness_delta.mean")),
                "action_delta_p95": metric(get(insertion_clean, "insertion_risk.summary.action_delta_norm.p95")),
                "range_violation_p95": metric(
                    get(insertion_clean, "insertion_risk.summary.refined_hard_range_violation.p95")
                ),
            },
        },
        "distilled_energy": {
            "role": "promoted_ablation_candidate",
            "differentiable": True,
            "episode_group_energy_auc": metric(get(distill, "mixed_episode_group_cv.energy_binary_auc.mean")),
            "episode_group_binary_auc": metric(get(distill, "mixed_episode_group_cv.binary_auc.mean")),
            "teacher_prediction_corr": metric(get(distill, "mixed_episode_group_cv.teacher_pred_corr.mean")),
            "energy_teacher_spearman": metric(get(distill, "mixed_episode_group_cv.energy_teacher_spearman.mean")),
            "energy_quality_spearman": metric(get(distill, "mixed_episode_group_cv.energy_quality_spearman.mean")),
            "feature_gradient_usable": bool(get(distill, "guidance_sanity.usable_for_feature_guidance", False)),
            "runtime_gradient_usable": bool(get(runtime, "usable_for_guidance", False)),
            "runtime_grad_norms": {
                "eef": metric(get(runtime, "eef_grad_norm")),
                "joint": metric(get(runtime, "joint_grad_norm")),
                "left_tactile": metric(get(runtime, "left_grad_norm")),
                "right_tactile": metric(get(runtime, "right_grad_norm")),
            },
            "feature_guidance": {
                "passes": bool(get(feature, "distilled_energy.passes_local_guidance", False)),
                "improved_rate": metric(get(feature, "distilled_energy.recommended_improved_rate")),
                "score_delta_mean": metric(get(feature, "distilled_energy.recommended_score_delta_mean")),
            },
            "insertion_clean_refine": {
                "passes": bool(get(insertion_clean, "distilled_energy.passes_insertion_clean_refine_smoke", False)),
                "improved_rate": metric(get(insertion_clean, "distilled_energy.summary.refined_beats_base_rate")),
                "score_delta_mean": metric(get(insertion_clean, "distilled_energy.summary.score_delta.mean")),
                "smoothness_delta_mean": metric(get(insertion_clean, "distilled_energy.summary.smoothness_delta.mean")),
                "action_delta_p95": metric(get(insertion_clean, "distilled_energy.summary.action_delta_norm.p95")),
                "range_violation_p95": metric(
                    get(insertion_clean, "distilled_energy.summary.refined_hard_range_violation.p95")
                ),
            },
            "board_surrogate": {
                "passes": bool(get(surrogate, "distilled_energy.passes_board_surrogate_action_refinement", False)),
                "improved_rate": metric(get(surrogate, "distilled_energy.summary.score_improved_rate")),
                "score_delta_mean": metric(get(surrogate, "distilled_energy.summary.score_delta.mean")),
            },
            "board_dp_clean_refine": {
                "passes": bool(get(board_dp, "distilled_energy.passes_board_dp_clean_refine_smoke", False)),
                "improved_rate": metric(get(board_dp, "distilled_energy.summary.guided_beats_base_rate")),
                "score_delta_mean": metric(get(board_dp, "distilled_energy.summary.score_delta.mean")),
                "smoothness_delta_mean": metric(get(board_dp, "distilled_energy.summary.smoothness_delta.mean")),
                "range_violation_max": metric(get(board_dp, "distilled_energy.summary.range_violation.max")),
            },
        },
    }

    checks = [
        {
            "name": "required_evidence_files_exist",
            "passed": all(paths[name].exists() for name in paths if not name.startswith("real_rollout_")),
            "evidence": {name: file_info(path) for name, path in paths.items()},
        },
        {
            "name": "episode_group_generalization_is_authoritative",
            "passed": get(split, "authoritative_split") == "episode_group_kfold"
            and evidence["rf_teacher"]["episode_group_auc"] >= 0.97
            and evidence["rf_teacher"]["frame_random_minus_group_balanced_accuracy"] <= 0.03,
            "evidence": evidence["rf_teacher"],
        },
        {
            "name": "insertion_default_passes_clean_refine_gate",
            "passed": evidence["insertion_risk"]["insertion_clean_refine"]["passes"]
            and evidence["insertion_risk"]["insertion_clean_refine"]["improved_rate"] >= 0.95
            and evidence["insertion_risk"]["insertion_clean_refine"]["range_violation_p95"] <= 1e-5,
            "evidence": evidence["insertion_risk"],
        },
        {
            "name": "ptg_proxy_v2_default_passes_core_gates",
            "passed": evidence["ptg_proxy_v2"]["episode_group_binary_auc"] >= 0.95
            and evidence["ptg_proxy_v2"]["episode_group_quality_corr"] >= 0.70
            and evidence["ptg_proxy_v2"]["feature_gradient_usable"]
            and evidence["ptg_proxy_v2"]["feature_guidance"]["passes"]
            and evidence["ptg_proxy_v2"]["board_surrogate"]["passes"]
            and evidence["ptg_proxy_v2"]["board_dp_clean_refine"]["passes"],
            "evidence": evidence["ptg_proxy_v2"],
        },
        {
            "name": "distilled_energy_passes_candidate_promotion_gates",
            "passed": evidence["distilled_energy"]["episode_group_energy_auc"] >= 0.95
            and evidence["distilled_energy"]["teacher_prediction_corr"] >= 0.90
            and evidence["distilled_energy"]["energy_teacher_spearman"] >= 0.85
            and evidence["distilled_energy"]["feature_gradient_usable"]
            and evidence["distilled_energy"]["runtime_gradient_usable"]
            and evidence["distilled_energy"]["feature_guidance"]["passes"]
            and evidence["distilled_energy"]["insertion_clean_refine"]["passes"]
            and evidence["distilled_energy"]["board_surrogate"]["passes"]
            and evidence["distilled_energy"]["board_dp_clean_refine"]["passes"],
            "evidence": evidence["distilled_energy"],
        },
        {
            "name": "real_rollout_validation_not_claimed",
            "passed": data["real_rollout_insertion"] is None and data["real_rollout_board"] is None,
            "evidence": {
                "insertion_rollout_gate_exists": data["real_rollout_insertion"] is not None,
                "board_rollout_gate_exists": data["real_rollout_board"] is not None,
            },
        },
    ]

    distilled_delta_advantage = (
        evidence["distilled_energy"]["board_dp_clean_refine"]["score_delta_mean"]
        - evidence["ptg_proxy_v2"]["board_dp_clean_refine"]["score_delta_mean"]
    )
    distilled_insertion_delta_gap = (
        evidence["insertion_risk"]["insertion_clean_refine"]["score_delta_mean"]
        - evidence["distilled_energy"]["insertion_clean_refine"]["score_delta_mean"]
    )
    selection = {
        "current_default_insertion_scorer": "InsertionRiskScorerRuntime",
        "current_default_board_scorer": "PTGProxyScorerV2Runtime",
        "rf_teacher_role": "non_differentiable_upper_bound_and_soft_label_teacher",
        "promoted_ablation_candidate": "DistilledTacQualityEnergyRuntime",
        "distilled_replacement_status": "not_yet_replacement",
        "recommended_dp_guidance_mode": "final_clean_action_trust_region_refinement",
        "why_not_every_step_ddpm_guidance": (
            "Offline diagnostics show local score gains are reliable, but every-step "
            "denoising guidance can move samples off the learned DP manifold.  Current "
            "safe mode is bounded accept-only refinement after clean-action prediction."
        ),
        "why_distilled_is_novel_candidate": (
            "It distills a strong episode-generalizing non-differentiable RF teacher "
            "into a differentiable energy that preserves teacher ordering while exposing "
            "stable gradients through Foresight to action."
        ),
        "why_not_replace_default_yet": (
            "Distilled energy gives larger internal board DP clean-refine score gain "
            f"({distilled_delta_advantage:.6f} over PTGProxyV2), but score scales differ "
            "and no formal real baseline-vs-guided rollout gate exists.  On insertion, "
            "the task-specific scorer has a much stronger clean-refine score response "
            f"({distilled_insertion_delta_gap:.6f} higher mean delta), so the distilled "
            "scorer should stay an ablation candidate rather than replacing insertion default."
        ),
        "next_required_experiment": (
            "Run paired real rollout ablation: baseline DP vs PTGProxyV2-guided vs "
            "DistilledTacQualityEnergy-guided for insertion and board."
        ),
    }

    gate = {
        "name": "TacQuality scorer selection gate",
        "git_commit": git_commit(),
        "objective": (
            "Select a scientifically defensible tactile quality classifier/scorer for "
            "DP gradient guidance across socket insertion and board wiping."
        ),
        "selection": selection,
        "evidence": evidence,
        "checks": checks,
        "selection_gate_pass": all(item["passed"] for item in checks),
        "status": "offline_candidate_selected_not_real_rollout_validated",
        "paths": {name: str(path) for name, path in paths.items()},
    }
    return gate


def write_markdown(gate: Dict[str, Any], path: Path) -> None:
    selection = gate["selection"]
    evidence = gate["evidence"]
    lines = [
        "# TacQuality Scorer Selection Gate",
        "",
        f"- selection_gate_pass: `{gate['selection_gate_pass']}`",
        f"- status: `{gate['status']}`",
        f"- git_commit: `{gate['git_commit']}`",
        "",
        "## Decision",
        "",
        f"- current_default_insertion_scorer: `{selection['current_default_insertion_scorer']}`",
        f"- current_default_board_scorer: `{selection['current_default_board_scorer']}`",
        f"- rf_teacher_role: `{selection['rf_teacher_role']}`",
        f"- promoted_ablation_candidate: `{selection['promoted_ablation_candidate']}`",
        f"- distilled_replacement_status: `{selection['distilled_replacement_status']}`",
        f"- recommended_dp_guidance_mode: `{selection['recommended_dp_guidance_mode']}`",
        "",
        "## Rationale",
        "",
        f"- why_distilled_is_novel_candidate: {selection['why_distilled_is_novel_candidate']}",
        f"- why_not_replace_default_yet: {selection['why_not_replace_default_yet']}",
        f"- why_not_every_step_ddpm_guidance: {selection['why_not_every_step_ddpm_guidance']}",
        f"- next_required_experiment: {selection['next_required_experiment']}",
        "",
        "## Key Evidence",
        "",
        "| scorer | role | AUC / teacher | feature guidance | insertion clean-refine | board surrogate | board DP clean-refine |",
        "|---|---|---:|---:|---:|---:|---:|",
        (
            "| RF teacher | offline upper bound | "
            f"{evidence['rf_teacher']['episode_group_auc']:.4f} | n/a | n/a | n/a | n/a |"
        ),
        (
            "| InsertionRisk | insertion default | n/a | n/a | "
            f"{evidence['insertion_risk']['insertion_clean_refine']['improved_rate']:.4f} | n/a | n/a |"
        ),
        (
            "| PTGProxyV2 | current default | "
            f"{evidence['ptg_proxy_v2']['episode_group_binary_auc']:.4f} | "
            f"{evidence['ptg_proxy_v2']['feature_guidance']['improved_rate']:.4f} | n/a | "
            f"{evidence['ptg_proxy_v2']['board_surrogate']['improved_rate']:.4f} | "
            f"{evidence['ptg_proxy_v2']['board_dp_clean_refine']['improved_rate']:.4f} |"
        ),
        (
            "| DistilledEnergy | ablation candidate | "
            f"{evidence['distilled_energy']['episode_group_energy_auc']:.4f} / "
            f"{evidence['distilled_energy']['energy_teacher_spearman']:.4f} | "
            f"{evidence['distilled_energy']['feature_guidance']['improved_rate']:.4f} | "
            f"{evidence['distilled_energy']['insertion_clean_refine']['improved_rate']:.4f} | "
            f"{evidence['distilled_energy']['board_surrogate']['improved_rate']:.4f} | "
            f"{evidence['distilled_energy']['board_dp_clean_refine']['improved_rate']:.4f} |"
        ),
        "",
        "## Checks",
        "",
    ]
    for item in gate["checks"]:
        status = "PASS" if item["passed"] else "FAIL"
        lines.append(f"- **{status}** {item['name']}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default=str(OUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gate = build_gate(PATHS)
    json_path = out_dir / "tac_quality_scorer_selection_gate.json"
    md_path = out_dir / "tac_quality_scorer_selection_gate.md"
    json_path.write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(gate, md_path)
    print(
        json.dumps(
            {
                "selection_gate_pass": gate["selection_gate_pass"],
                "status": gate["status"],
                "default_board_scorer": gate["selection"]["current_default_board_scorer"],
                "promoted_ablation_candidate": gate["selection"]["promoted_ablation_candidate"],
                "json": str(json_path),
                "markdown": str(md_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
