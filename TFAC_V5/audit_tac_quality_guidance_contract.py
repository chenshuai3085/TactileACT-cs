"""Audit the TacQuality scorer contract required for DP guidance.

This audit is intentionally narrower than the full goal-completion audit.  It
answers one question:

  Is the current scorer package a valid differentiable guidance objective,
  rather than only a classifier with high offline accuracy?

The answer can be true while the overall research objective remains incomplete,
because real paired robot rollouts are still required for final validation.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional


OUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_guidance_contract")

PATHS = {
    "label_registry": Path(
        "/home/chenshuai/Project/output/tac_quality_label_standard_registry/"
        "tac_quality_label_standard_registry.json"
    ),
    "label_compliance": Path(
        "/home/chenshuai/Project/output/tac_quality_label_standard_compliance/"
        "tac_quality_label_standard_compliance.json"
    ),
    "scorer_decision_matrix": Path(
        "/home/chenshuai/Project/output/tac_quality_scorer_decision_matrix/"
        "tac_quality_scorer_decision_matrix.json"
    ),
    "runtime_contract": Path(
        "/home/chenshuai/Project/output/tac_quality_guidance_runtime/"
        "runtime_contract_sanity.json"
    ),
    "score_calibration": Path(
        "/home/chenshuai/Project/output/tac_quality_score_calibration/"
        "tac_quality_score_calibration.json"
    ),
    "score_landscape": Path(
        "/home/chenshuai/Project/output/tac_quality_score_landscape/"
        "tac_quality_score_landscape.json"
    ),
    "scale_sweep": Path(
        "/home/chenshuai/Project/output/tac_quality_guidance_scale_sweep/"
        "tac_quality_guidance_scale_sweep.json"
    ),
    "robustness": Path(
        "/home/chenshuai/Project/output/tac_quality_guidance_robustness/"
        "tac_quality_guidance_robustness.json"
    ),
    "trust_region": Path(
        "/home/chenshuai/Project/output/tac_quality_trust_region_guidance/"
        "trust_region_sanity.json"
    ),
    "dp_adapter": Path(
        "/home/chenshuai/Project/output/tac_quality_dp_integration_adapter/"
        "integration_adapter_sanity.json"
    ),
    "foresight_bridge": Path(
        "/home/chenshuai/Project/output/tac_quality_foresight_bridge/"
        "foresight_bridge_sanity.json"
    ),
    "guided_server_smoke": Path(
        "/home/chenshuai/Project/output/tac_quality_guided_server_real_foresight_smoke/"
        "auto_discovered_all_arms/tac_quality_guided_server_real_foresight_smoke.json"
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


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def check(name: str, passed: bool, evidence: Dict[str, Any], required: bool = True) -> Dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "required_for_guidance_contract": bool(required),
        "evidence": evidence,
    }


def build_audit(paths: Dict[str, Path]) -> Dict[str, Any]:
    data = {name: load_json(path) for name, path in paths.items()}

    checks = [
        check(
            "task_quality_standards_defined",
            bool(get(data["label_registry"], "registry_pass", False))
            and get(data["label_registry"], "tasks.insertion.binary_policy.bad")
            == ["pre_bounce_risk", "impact_or_recovery"]
            and get(data["label_registry"], "tasks.board.binary_policy.good") == ["good_smooth"]
            and get(data["label_registry"], "tasks.board.target_force") is not None,
            {
                "registry_pass": get(data["label_registry"], "registry_pass"),
                "insertion_binary_policy": get(data["label_registry"], "tasks.insertion.binary_policy"),
                "board_binary_policy": get(data["label_registry"], "tasks.board.binary_policy"),
                "board_target_force": get(data["label_registry"], "tasks.board.target_force"),
            },
        ),
        check(
            "cached_labels_match_quality_standard",
            bool(get(data["label_compliance"], "compliance_pass", False))
            and get(data["label_compliance"], "required_checks_pass") is True
            and get(data["label_compliance"], "compatible_deviations_pass") is True,
            {
                "compliance_pass": get(data["label_compliance"], "compliance_pass"),
                "summary": get(data["label_compliance"], "summary"),
                "interpretation": get(data["label_compliance"], "interpretation"),
            },
        ),
        check(
            "candidate_selection_uses_guidance_criteria",
            bool(get(data["scorer_decision_matrix"], "passes_decision_matrix_gate", False))
            and get(data["scorer_decision_matrix"], "recommendation.formal_default_insertion")
            == "InsertionRiskScorerRuntime"
            and get(data["scorer_decision_matrix"], "recommendation.formal_default_board")
            == "PTGProxyScorerV2Runtime"
            and get(data["scorer_decision_matrix"], "recommendation.innovation_ablation")
            == "DistilledTacQualityEnergyRuntime",
            {
                "passes": get(data["scorer_decision_matrix"], "passes_decision_matrix_gate"),
                "criteria": get(data["scorer_decision_matrix"], "criteria"),
                "recommendation": get(data["scorer_decision_matrix"], "recommendation"),
                "remaining_gap": get(data["scorer_decision_matrix"], "remaining_gap"),
            },
        ),
        check(
            "runtime_score_is_differentiable_wrt_action",
            bool(get(data["runtime_contract"], "passes_runtime_contract_sanity", False))
            and (get(data["runtime_contract"], "insertion.action_grad_norm_mean", 0.0) or 0.0) > 0
            and (get(data["runtime_contract"], "board.joint_action_grad_norm_mean", 0.0) or 0.0) > 0
            and (get(data["runtime_contract"], "board.eef_action_grad_norm_mean", 0.0) or 0.0) > 0,
            {
                "runtime_pass": get(data["runtime_contract"], "passes_runtime_contract_sanity"),
                "insertion_action_grad_norm_mean": get(
                    data["runtime_contract"], "insertion.action_grad_norm_mean"
                ),
                "board_joint_action_grad_norm_mean": get(
                    data["runtime_contract"], "board.joint_action_grad_norm_mean"
                ),
                "board_eef_action_grad_norm_mean": get(
                    data["runtime_contract"], "board.eef_action_grad_norm_mean"
                ),
            },
        ),
        check(
            "score_mode_is_calibrated_for_guidance",
            get(data["score_calibration"], "recommendation.insertion") == "energy"
            and get(data["score_calibration"], "recommendation.board") == "quality",
            {
                "recommendation": get(data["score_calibration"], "recommendation"),
                "insertion": get(data["score_calibration"], "insertion"),
                "board": get(data["score_calibration"], "board"),
            },
        ),
        check(
            "local_gradient_landscape_is_usable",
            bool(get(data["score_landscape"], "overall_pass", False))
            and get(data["score_landscape"], "insertion.passes_score_landscape") is True
            and get(data["score_landscape"], "board.passes_score_landscape") is True,
            {
                "overall_pass": get(data["score_landscape"], "overall_pass"),
                "insertion_grad_norm": get(data["score_landscape"], "insertion.gradient.grad_norm.mean"),
                "board_grad_norm": get(data["score_landscape"], "board.gradient.grad_norm.mean"),
            },
        ),
        check(
            "guidance_scale_law_passes_on_real_samples",
            bool(get(data["scale_sweep"], "overall_pass", False))
            and get(data["scale_sweep"], "insertion.passes_guidance_scale_sweep") is True
            and get(data["scale_sweep"], "board.passes_guidance_scale_sweep") is True,
            {
                "overall_pass": get(data["scale_sweep"], "overall_pass"),
                "insertion_recommended_scale": get(data["scale_sweep"], "insertion.recommended_scale"),
                "board_recommended_scale": get(data["scale_sweep"], "board.recommended_scale"),
                "insertion_improved_rate": get(
                    data["scale_sweep"], "insertion.recommended_improved_rate"
                ),
                "board_improved_rate": get(data["scale_sweep"], "board.recommended_improved_rate"),
            },
        ),
        check(
            "current_gradient_robustness_passes_under_noise",
            bool(get(data["robustness"], "overall_pass", False))
            and get(data["robustness"], "insertion.passes_current_gradient_robustness") is True
            and get(data["robustness"], "board.passes_current_gradient_robustness") is True,
            {
                "overall_pass": get(data["robustness"], "overall_pass"),
                "interpretation": get(data["robustness"], "interpretation"),
                "insertion_worst_perturbed_gradient_improved_rate": get(
                    data["robustness"], "insertion.worst_perturbed_gradient_improved_rate"
                ),
                "board_worst_perturbed_gradient_improved_rate": get(
                    data["robustness"], "board.worst_perturbed_gradient_improved_rate"
                ),
            },
        ),
        check(
            "bounded_accept_only_trust_region_refinement_passes",
            bool(get(data["trust_region"], "passes_trust_region_guidance_sanity", False))
            and get(data["trust_region"], "insertion.config.accept_only_improved") is True
            and get(data["trust_region"], "board.config.accept_only_improved") is True
            and get(data["trust_region"], "insertion.max_delta_within_trust_region") is True
            and get(data["trust_region"], "board.max_delta_within_trust_region") is True
            and (get(data["trust_region"], "insertion.improved_rate", 0.0) or 0.0) >= 0.99
            and (get(data["trust_region"], "board.improved_rate", 0.0) or 0.0) >= 0.99,
            {
                "trust_region_pass": get(data["trust_region"], "passes_trust_region_guidance_sanity"),
                "insertion_config": get(data["trust_region"], "insertion.config"),
                "board_config": get(data["trust_region"], "board.config"),
                "insertion_improved_rate": get(data["trust_region"], "insertion.improved_rate"),
                "board_improved_rate": get(data["trust_region"], "board.improved_rate"),
            },
        ),
        check(
            "dp_adapter_contract_is_not_reranking_or_every_step_guidance",
            bool(get(data["dp_adapter"], "passes_integration_adapter_sanity", False))
            and get(data["dp_adapter"], "insertion.integration_contract.reranking") is False
            and get(data["dp_adapter"], "insertion.integration_contract.every_step_ddpm_guidance") is False
            and get(data["dp_adapter"], "insertion.config.recompute_gradient_every_call") is True
            and get(data["dp_adapter"], "board.config.recompute_gradient_every_call") is True,
            {
                "adapter_pass": get(data["dp_adapter"], "passes_integration_adapter_sanity"),
                "insertion_contract": get(data["dp_adapter"], "insertion.integration_contract"),
                "insertion_config": get(data["dp_adapter"], "insertion.config"),
                "board_config": get(data["dp_adapter"], "board.config"),
            },
        ),
        check(
            "foresight_bridge_preserves_action_to_tactile_to_score_gradient",
            bool(get(data["foresight_bridge"], "passes_foresight_bridge_sanity", False))
            and get(data["foresight_bridge"], "not_reranking") is True
            and get(data["foresight_bridge"], "not_every_step_ddpm_guidance") is True
            and (get(data["foresight_bridge"], "insertion.bridge_grad.positive_grad_rate", 0.0) or 0.0)
            >= 0.999
            and (get(data["foresight_bridge"], "board.bridge_grad.positive_grad_rate", 0.0) or 0.0)
            >= 0.999,
            {
                "bridge_pass": get(data["foresight_bridge"], "passes_foresight_bridge_sanity"),
                "guidance_mode": get(data["foresight_bridge"], "guidance_mode"),
                "not_reranking": get(data["foresight_bridge"], "not_reranking"),
                "not_every_step_ddpm_guidance": get(data["foresight_bridge"], "not_every_step_ddpm_guidance"),
                "insertion_grad_rate": get(
                    data["foresight_bridge"], "insertion.bridge_grad.positive_grad_rate"
                ),
                "board_grad_rate": get(data["foresight_bridge"], "board.bridge_grad.positive_grad_rate"),
            },
        ),
        check(
            "guided_server_dry_run_preserves_guidance_contract",
            bool(get(data["guided_server_smoke"], "overall_pass", False))
            and get(data["guided_server_smoke"], "not_reranking") is True
            and get(data["guided_server_smoke"], "not_every_step_ddpm_guidance") is True
            and get(data["guided_server_smoke"], "checks.all_guided_arms_pass_real_foresight_smoke")
            is True,
            {
                "overall_pass": get(data["guided_server_smoke"], "overall_pass"),
                "scientific_evidence": get(data["guided_server_smoke"], "scientific_evidence"),
                "checks": get(data["guided_server_smoke"], "checks"),
            },
        ),
        check(
            "formal_real_rollout_validation_still_missing",
            data["real_rollout_insertion"] is None and data["real_rollout_board"] is None,
            {
                "insertion_gate_exists": data["real_rollout_insertion"] is not None,
                "board_gate_exists": data["real_rollout_board"] is not None,
                "meaning": (
                    "Guidance contract can pass offline, but final scientific validation still "
                    "requires paired real rollout HDF5 gates."
                ),
            },
            required=False,
        ),
    ]

    required_checks = [row for row in checks if row["required_for_guidance_contract"]]
    guidance_contract_pass = all(row["passed"] for row in required_checks)
    real_rollout_validation_complete = (
        get(data["real_rollout_insertion"], "decision.production_validation_pass") is True
        and get(data["real_rollout_board"], "decision.production_validation_pass") is True
    )
    result = {
        "name": "TacQuality DP guidance contract audit",
        "git_commit": git_commit(),
        "guidance_contract_pass": bool(guidance_contract_pass),
        "real_rollout_validation_complete": bool(real_rollout_validation_complete),
        "scientific_evidence_complete": bool(guidance_contract_pass and real_rollout_validation_complete),
        "deployment_interpretation": (
            "Offline scorer-guidance contract is satisfied for final clean-action trust-region refinement. "
            "This proves the scorer is suitable for DP gradient-guidance experiments, not that it has "
            "already improved real robot rollouts."
            if guidance_contract_pass
            else "The scorer package is missing at least one required DP guidance contract check."
        ),
        "recommended_guidance_mode": "final_clean_action_trust_region_refinement",
        "not_recommended_modes": [
            "reranking_only",
            "unbounded_noisy_step_guidance",
            "stale_gradient_reuse",
            "large_scale_binary_logprob_only_guidance",
        ],
        "checks": checks,
        "failed_required_checks": [row for row in required_checks if not row["passed"]],
        "paths": {name: str(path) for name, path in paths.items()},
    }
    return result


def write_markdown(result: Dict[str, Any], path: Path) -> None:
    lines = [
        "# TacQuality DP Guidance Contract Audit",
        "",
        f"- guidance_contract_pass: `{result['guidance_contract_pass']}`",
        f"- scientific_evidence_complete: `{result['scientific_evidence_complete']}`",
        f"- recommended_guidance_mode: `{result['recommended_guidance_mode']}`",
        f"- interpretation: {result['deployment_interpretation']}",
        "",
        "## Checks",
        "",
        "| required | passed | check | evidence |",
        "|---|---:|---|---|",
    ]
    for row in result["checks"]:
        evidence = json.dumps(row["evidence"], ensure_ascii=False)
        lines.append(
            f"| {row['required_for_guidance_contract']} | {row['passed']} | "
            f"{row['name']} | {evidence.replace('|', '/')} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default=str(OUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = build_audit(PATHS)
    json_path = out_dir / "tac_quality_guidance_contract_audit.json"
    md_path = out_dir / "tac_quality_guidance_contract_audit.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    print(
        json.dumps(
            {
                "guidance_contract_pass": result["guidance_contract_pass"],
                "scientific_evidence_complete": result["scientific_evidence_complete"],
                "n_checks": len(result["checks"]),
                "n_failed_required": len(result["failed_required_checks"]),
                "recommended_guidance_mode": result["recommended_guidance_mode"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"Saved {json_path}")
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
