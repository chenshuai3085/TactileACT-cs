"""Audit completion of the TacQualityEnergy guidance objective.

This is stricter than the offline production gate.  The gate answers:
"is the current scorer/guidance stack ready for robot dry-run?"  This audit
answers the user-level objective:

  design, evaluate, and record a tactile quality classifier/scorer for DP
  classifier guidance on socket insertion and board wiping, with evidence that
  it works and is innovative enough for the intended gradient-guidance use.

The audit deliberately keeps the objective incomplete until real baseline-vs-
guided production/robot rollouts exist for both tasks.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional


OUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_goal_audit")

PATHS = {
    "insertion_eval": Path("/home/chenshuai/Project/output/ptg_quality_eval/tactile_quality_model_eval.json"),
    "board_scheme_eval": Path("/home/chenshuai/Project/output/board_quality_label_schemes/w32_s16/board_quality_scheme_eval.json"),
    "board_target_force_calibration": Path(
        "/home/chenshuai/Project/output/board_target_force_calibration/board_target_force_calibration.json"
    ),
    "label_standard_registry": Path(
        "/home/chenshuai/Project/output/tac_quality_label_standard_registry/"
        "tac_quality_label_standard_registry.json"
    ),
    "label_standard_compliance": Path(
        "/home/chenshuai/Project/output/tac_quality_label_standard_compliance/"
        "tac_quality_label_standard_compliance.json"
    ),
    "ptg_proxy_eval": Path("/home/chenshuai/Project/output/ptg_proxy_scorer_v2/ptg_proxy_scorer_v2_eval.json"),
    "score_calibration": Path("/home/chenshuai/Project/output/tac_quality_score_calibration/tac_quality_score_calibration.json"),
    "scale_sweep": Path("/home/chenshuai/Project/output/tac_quality_guidance_scale_sweep/tac_quality_guidance_scale_sweep.json"),
    "robustness": Path("/home/chenshuai/Project/output/tac_quality_guidance_robustness/tac_quality_guidance_robustness.json"),
    "guidance_contract": Path(
        "/home/chenshuai/Project/output/tac_quality_guidance_contract/"
        "tac_quality_guidance_contract_audit.json"
    ),
    "dp_integration_adapter": Path(
        "/home/chenshuai/Project/output/tac_quality_dp_integration_adapter/integration_adapter_sanity.json"
    ),
    "foresight_bridge": Path(
        "/home/chenshuai/Project/output/tac_quality_foresight_bridge/foresight_bridge_sanity.json"
    ),
    "score_landscape": Path("/home/chenshuai/Project/output/tac_quality_score_landscape/tac_quality_score_landscape.json"),
    "proxy_alignment_audit": Path(
        "/home/chenshuai/Project/output/tac_quality_proxy_alignment_audit/"
        "tac_quality_proxy_alignment_audit.json"
    ),
    "runtime_visualization": Path(
        "/home/chenshuai/Project/output/tac_quality_runtime_visualization/tac_quality_runtime_visualization.json"
    ),
    "offline_gate": Path("/home/chenshuai/Project/output/ptg_offline_production_gate/ptg_offline_production_gate.json"),
    "scorer_selection_gate": Path(
        "/home/chenshuai/Project/output/tac_quality_scorer_selection_gate/tac_quality_scorer_selection_gate.json"
    ),
    "scorer_decision_matrix": Path(
        "/home/chenshuai/Project/output/tac_quality_scorer_decision_matrix/"
        "tac_quality_scorer_decision_matrix.json"
    ),
    "action_aware_eval": Path(
        "/home/chenshuai/Project/output/action_aware_marker_scorer/action_aware_marker_scorer_eval.json"
    ),
    "action_aware_runtime": Path(
        "/home/chenshuai/Project/output/action_aware_marker_scorer/runtime_gradient_sanity.json"
    ),
    "action_aware_guidance_suitability": Path(
        "/home/chenshuai/Project/output/action_aware_guidance_suitability/"
        "line_search_default/action_aware_guidance_suitability.json"
    ),
    "insertion_distilled_clean_refine": Path(
        "/home/chenshuai/Project/output/insertion_distilled_clean_refine_comparison/"
        "n24_k4/insertion_distilled_clean_refine_comparison.json"
    ),
    "manifest": Path("/home/chenshuai/Project/output/tac_quality_guidance_manifest/tac_quality_guidance_manifest.json"),
    "evidence_summary": Path("/home/chenshuai/Project/output/ptg_guidance_evidence/ptg_guidance_evidence_summary.json"),
    "real_rollout_insertion": Path("/home/chenshuai/Project/output/real_rollout_quality_gate/insertion_baseline_vs_guided/real_rollout_quality_gate.json"),
    "real_rollout_board": Path("/home/chenshuai/Project/output/real_rollout_quality_gate/board_baseline_vs_guided/real_rollout_quality_gate.json"),
    "real_rollout_prep_smoke": Path("/home/chenshuai/Project/output/real_rollout_validation_ready/board_smoke_ready_with_csv/real_rollout_validation_readiness.json"),
    "insertion_sample_size_plan": Path("/home/chenshuai/Project/output/real_rollout_sample_size_plan/insertion_default_plan/real_rollout_sample_size_plan.json"),
    "board_sample_size_plan": Path("/home/chenshuai/Project/output/real_rollout_sample_size_plan/board_default_plan/real_rollout_sample_size_plan.json"),
    "real_rollout_experiment_packet": Path("/home/chenshuai/Project/output/real_rollout_experiment_packet/formal_paired12/real_rollout_experiment_packet.json"),
    "scorer_ablation_insertion": Path(
        "/home/chenshuai/Project/output/real_rollout_scorer_ablation_gate/"
        "insertion_baseline_vs_default_vs_distilled/real_rollout_scorer_ablation_gate.json"
    ),
    "scorer_ablation_board": Path(
        "/home/chenshuai/Project/output/real_rollout_scorer_ablation_gate/"
        "board_baseline_vs_default_vs_distilled/real_rollout_scorer_ablation_gate.json"
    ),
    "scorer_ablation_smoke": Path(
        "/home/chenshuai/Project/output/real_rollout_scorer_ablation_smoke/"
        "real_rollout_scorer_ablation_smoke.json"
    ),
    "rollout_arm_configs": Path(
        "/home/chenshuai/Project/output/tac_quality_rollout_arm_configs/tac_quality_rollout_arm_configs.json"
    ),
    "rollout_arm_config_smoke": Path(
        "/home/chenshuai/Project/output/tac_quality_rollout_arm_config_smoke/"
        "tac_quality_rollout_arm_config_smoke.json"
    ),
    "deployment_bridge_smoke": Path(
        "/home/chenshuai/Project/output/tac_quality_deployment_bridge_smoke/"
        "tac_quality_deployment_bridge_smoke.json"
    ),
    "serving_packet": Path(
        "/home/chenshuai/Project/output/tac_quality_serving_packet/"
        "auto_discovered/tac_quality_serving_packet.json"
    ),
    "guided_server_packet": Path(
        "/home/chenshuai/Project/output/tac_quality_guided_server_packet/"
        "auto_discovered/tac_quality_guided_server_packet.json"
    ),
    "guided_server_insertion_real_foresight_smoke": Path(
        "/home/chenshuai/Project/output/tac_quality_guided_server_packet/"
        "auto_discovered/insertion_guided_server_real_foresight_smoke.json"
    ),
    "guided_server_board_real_foresight_smoke": Path(
        "/home/chenshuai/Project/output/tac_quality_guided_server_packet/"
        "auto_discovered/board_guided_server_real_foresight_smoke.json"
    ),
    "guided_server_all_arms_real_foresight_smoke": Path(
        "/home/chenshuai/Project/output/tac_quality_guided_server_real_foresight_smoke/"
        "auto_discovered_all_arms/tac_quality_guided_server_real_foresight_smoke.json"
    ),
    "guided_server_insertion_baseline_no_guidance_smoke": Path(
        "/home/chenshuai/Project/output/tac_quality_guided_server_packet/"
        "auto_discovered/insertion_baseline_no_guidance_smoke.json"
    ),
    "guided_server_board_baseline_no_guidance_smoke": Path(
        "/home/chenshuai/Project/output/tac_quality_guided_server_packet/"
        "auto_discovered/board_baseline_no_guidance_smoke.json"
    ),
    "formal_rollout_gate_runner": Path(
        "/home/chenshuai/Project/output/formal_tac_quality_rollout_gate_runner/"
        "formal_paired12_preflight/formal_tac_quality_rollout_gate_runner.json"
    ),
    "formal_launch_sheet": Path(
        "/home/chenshuai/Project/output/tac_quality_formal_launch_sheet/"
        "formal_paired12/tac_quality_formal_launch_sheet.json"
    ),
    "formal_launch_sheet_smoke": Path(
        "/home/chenshuai/Project/output/tac_quality_formal_launch_sheet_smoke/"
        "formal_paired12/tac_quality_formal_launch_sheet_smoke.json"
    ),
    "formal_collection_readiness": Path(
        "/home/chenshuai/Project/output/tac_quality_collection_readiness/"
        "formal_paired12/tac_quality_collection_readiness.json"
    ),
    "formal_rollout_pairing": Path(
        "/home/chenshuai/Project/output/tac_quality_rollout_pairing/"
        "formal_paired12/tac_quality_rollout_pairing.json"
    ),
    "pairing_metadata_audit": Path(
        "/home/chenshuai/Project/output/tac_quality_pairing_metadata_audit/"
        "tac_quality_pairing_metadata_audit.json"
    ),
    "hdf5_schema_audit": Path(
        "/home/chenshuai/Project/output/tac_quality_rollout_hdf5_schema_audit/"
        "tac_quality_rollout_hdf5_schema_audit.json"
    ),
    "generated_pairing_gate_runner_smoke": Path(
        "/home/chenshuai/Project/output/tac_quality_generated_pairing_gate_runner_smoke/"
        "synthetic_n10/tac_quality_generated_pairing_gate_runner_smoke.json"
    ),
    "real_rollout_source_audit": Path(
        "/home/chenshuai/Project/output/tac_quality_real_rollout_source_audit/"
        "tac_quality_real_rollout_source_audit.json"
    ),
    "post_collection_pipeline": Path(
        "/home/chenshuai/Project/output/tac_quality_post_collection_pipeline/"
        "formal_paired12/tac_quality_post_collection_pipeline.json"
    ),
    "post_collection_pipeline_smoke": Path(
        "/home/chenshuai/Project/output/tac_quality_post_collection_pipeline_smoke/"
        "synthetic_n10/tac_quality_post_collection_pipeline_smoke.json"
    ),
    "finalize_collected_hdf5_smoke": Path(
        "/home/chenshuai/Project/output/tac_quality_finalize_collected_hdf5_smoke/"
        "synthetic/tac_quality_finalize_collected_hdf5_smoke.json"
    ),
    "finalize_and_refresh_smoke": Path(
        "/home/chenshuai/Project/output/tac_quality_finalize_and_refresh_smoke/"
        "synthetic/tac_quality_finalize_and_refresh_smoke.json"
    ),
    "real_rollout_acceptance_protocol": Path(
        "/home/chenshuai/Project/output/tac_quality_real_rollout_acceptance_protocol/"
        "tac_quality_real_rollout_acceptance_protocol.json"
    ),
    "formal_rollout_runbook": Path(
        "/home/chenshuai/Project/output/tac_quality_formal_rollout_runbook/"
        "formal_paired12/tac_quality_formal_rollout_runbook.json"
    ),
    "formal_rollout_runbook_smoke": Path(
        "/home/chenshuai/Project/output/tac_quality_formal_rollout_runbook_smoke/"
        "formal_paired12/tac_quality_formal_rollout_runbook_smoke.json"
    ),
    "formal_collection_schedule": Path(
        "/home/chenshuai/Project/output/tac_quality_collection_schedule/"
        "formal_paired12/tac_quality_collection_schedule.json"
    ),
    "formal_collection_progress": Path(
        "/home/chenshuai/Project/output/tac_quality_collection_progress/"
        "formal_paired12/tac_quality_collection_progress.json"
    ),
    "formal_next_collection_step": Path(
        "/home/chenshuai/Project/output/tac_quality_next_collection_step/"
        "formal_paired12/tac_quality_next_collection_step.json"
    ),
    "formal_next_collection_step_smoke": Path(
        "/home/chenshuai/Project/output/tac_quality_next_collection_step_smoke/"
        "formal_paired12/tac_quality_next_collection_step_smoke.json"
    ),
    "formal_next_collection_step_smoke_runner": Path(
        "/home/chenshuai/Project/output/tac_quality_next_collection_step_smoke/"
        "formal_paired12/tac_quality_next_collection_step_smoke_runner.json"
    ),
    "formal_current_collection_gate": Path(
        "/home/chenshuai/Project/output/tac_quality_current_collection_gate/"
        "formal_paired12/tac_quality_current_collection_gate.json"
    ),
    "formal_current_collection_handoff": Path(
        "/home/chenshuai/Project/output/tac_quality_current_collection_handoff/"
        "formal_paired12/tac_quality_current_collection_handoff.json"
    ),
    "optional_action_aware_rollout_gate_runner": Path(
        "/home/chenshuai/Project/output/optional_action_aware_rollout_gate_runner/"
        "formal_paired12_preflight/optional_action_aware_rollout_gate_runner.json"
    ),
    "record": Path("/home/chenshuai/Project/TactileACT-cs/工作记录codex.txt"),
    "eval_doc": Path("/home/chenshuai/Project/TactileACT-cs/research/PTG_触觉质量分类器评估方案与实验记录.md"),
    "deploy_doc": Path("/home/chenshuai/Project/TactileACT-cs/research/PTG_TacQualityEnergy_部署策略与运行手册.md"),
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


def file_contains(path: Path, snippets: List[str]) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8", errors="ignore")
    return all(s in text for s in snippets)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def item(requirement: str, status: str, evidence: str, artifact: str) -> Dict[str, Any]:
    if status not in {"satisfied", "incomplete", "weak", "missing", "contradicted"}:
        raise ValueError(f"Bad audit status: {status}")
    return {
        "requirement": requirement,
        "status": status,
        "passed": status == "satisfied",
        "evidence": evidence,
        "artifact": artifact,
    }


def real_rollout_status(d: Optional[Dict[str, Any]], task: str) -> Dict[str, Any]:
    if d is None:
        return {
            "status": "missing",
            "evidence": f"No formal {task} baseline-vs-guided rollout gate report found.",
        }
    path_text = " ".join(
        str(d.get(key, ""))
        for key in ["baseline_dir", "guided_dir", "pairing_csv", "metadata_csv"]
    ).lower()
    if any(token in path_text for token in ["synthetic", "smoke"]):
        return {
            "status": "contradicted",
            "evidence": (
                "Found synthetic/smoke paths in the formal rollout gate report; "
                "this cannot count as real production/robot evidence."
            ),
        }
    decision = get(d, "decision.production_validation_pass", False)
    debug = bool(get(d, "debug_or_underpowered", True))
    if decision and not debug:
        return {"status": "satisfied", "evidence": "production_validation_pass=true and debug_or_underpowered=false"}
    return {
        "status": "incomplete",
        "evidence": (
            f"production_validation_pass={decision}, "
            f"debug_or_underpowered={debug}, "
            f"decision={get(d, 'decision')}"
        ),
    }


def ablation_is_real(d: Optional[Dict[str, Any]]) -> bool:
    if d is None:
        return False
    path_text = " ".join(
        str(d.get(key, ""))
        for key in [
            "baseline_dir",
            "default_guided_dir",
            "distilled_guided_dir",
            "pairing_csv",
            "metadata_csv",
        ]
    ).lower()
    return not any(token in path_text for token in ["synthetic", "smoke"])


def build_audit(paths: Dict[str, Path]) -> Dict[str, Any]:
    data = {name: load_json(path) for name, path in paths.items() if path.suffix == ".json"}

    insertion = data["insertion_eval"]
    board = data["board_scheme_eval"]
    board_calibration = data["board_target_force_calibration"]
    label_registry = data["label_standard_registry"]
    label_compliance = data["label_standard_compliance"]
    ptg = data["ptg_proxy_eval"]
    scale = data["scale_sweep"]
    robust = data["robustness"]
    guidance_contract = data["guidance_contract"]
    dp_integration_adapter = data["dp_integration_adapter"]
    foresight_bridge = data["foresight_bridge"]
    score_landscape = data["score_landscape"]
    proxy_alignment = data["proxy_alignment_audit"]
    runtime_visualization = data["runtime_visualization"]
    offline = data["offline_gate"]
    selection_gate = data["scorer_selection_gate"]
    decision_matrix = data["scorer_decision_matrix"]
    action_aware_eval = data["action_aware_eval"]
    action_aware_runtime = data["action_aware_runtime"]
    action_aware_guidance = data["action_aware_guidance_suitability"]
    insertion_distilled = data["insertion_distilled_clean_refine"]
    manifest = data["manifest"]
    summary = data["evidence_summary"]
    rr_ins = data["real_rollout_insertion"]
    rr_board = data["real_rollout_board"]
    prep_smoke = data["real_rollout_prep_smoke"]
    insertion_plan = data["insertion_sample_size_plan"]
    board_plan = data["board_sample_size_plan"]
    experiment_packet = data["real_rollout_experiment_packet"]
    ablation_ins = data["scorer_ablation_insertion"]
    ablation_board = data["scorer_ablation_board"]
    ablation_smoke = data["scorer_ablation_smoke"]
    rollout_arm_configs = data["rollout_arm_configs"]
    rollout_arm_config_smoke = data["rollout_arm_config_smoke"]
    deployment_bridge_smoke = data["deployment_bridge_smoke"]
    serving_packet = data["serving_packet"]
    guided_server_packet = data["guided_server_packet"]
    guided_server_all_arms_smoke = data["guided_server_all_arms_real_foresight_smoke"]
    formal_rollout_gate_runner = data["formal_rollout_gate_runner"]
    formal_launch_sheet = data["formal_launch_sheet"]
    formal_launch_sheet_smoke = data["formal_launch_sheet_smoke"]
    formal_collection_readiness = data["formal_collection_readiness"]
    formal_rollout_pairing = data["formal_rollout_pairing"]
    pairing_metadata_audit = data["pairing_metadata_audit"]
    hdf5_schema_audit = data["hdf5_schema_audit"]
    generated_pairing_gate_runner_smoke = data["generated_pairing_gate_runner_smoke"]
    real_rollout_source_audit = data["real_rollout_source_audit"]
    post_collection_pipeline = data["post_collection_pipeline"]
    post_collection_pipeline_smoke = data["post_collection_pipeline_smoke"]
    finalize_collected_hdf5_smoke = data["finalize_collected_hdf5_smoke"]
    finalize_and_refresh_smoke = data["finalize_and_refresh_smoke"]
    real_rollout_acceptance_protocol = data["real_rollout_acceptance_protocol"]
    formal_rollout_runbook = data["formal_rollout_runbook"]
    formal_rollout_runbook_smoke = data["formal_rollout_runbook_smoke"]
    formal_collection_schedule = data["formal_collection_schedule"]
    formal_collection_progress = data["formal_collection_progress"]
    formal_next_collection_step = data["formal_next_collection_step"]
    formal_next_collection_step_smoke = data["formal_next_collection_step_smoke"]
    formal_next_collection_step_smoke_runner = data["formal_next_collection_step_smoke_runner"]
    formal_current_collection_gate = data["formal_current_collection_gate"]
    formal_current_collection_handoff = data["formal_current_collection_handoff"]
    optional_action_aware_runner = data["optional_action_aware_rollout_gate_runner"]

    requirements = [
        item(
            "Socket insertion scorer is evaluated with episode-level generalization.",
            "satisfied"
            if (get(insertion, "group_cv.LDA.balanced_accuracy.mean", 0.0) or 0.0) >= 0.88
            and (get(insertion, "data.n_groups", 0) or 0) >= 100
            else "incomplete",
            "GroupKFold best LDA balanced_acc="
            f"{get(insertion, 'group_cv.LDA.balanced_accuracy.mean')}; "
            f"n_groups={get(insertion, 'data.n_groups')}",
            str(paths["insertion_eval"]),
        ),
        item(
            "Board rollout quality gate has explicit target-force calibration.",
            "satisfied"
            if get(board_calibration, "recommended.board_target_force") is not None
            and get(board_calibration, "recommended.board_force_sigma") is not None
            and (get(board_calibration, "n_episodes", 0) or 0) >= 20
            else "incomplete",
            "target="
            f"{get(board_calibration, 'recommended.board_target_force')}; "
            f"sigma={get(board_calibration, 'recommended.board_force_sigma')}; "
            f"n_episodes={get(board_calibration, 'n_episodes')}; "
            f"source={get(board_calibration, 'force_source')}",
            str(paths["board_target_force_calibration"]),
        ),
        item(
            "Board wiping quality labels and scorer target are defined from force magnitude and smoothness.",
            "satisfied"
            if get(board, "best_classification.balanced_accuracy", 0.0) >= 0.88
            and get(board, "best_regression.quality_corr", 0.0) >= 0.95
            else "incomplete",
            "best_classification balanced_acc="
            f"{get(board, 'best_classification.balanced_accuracy')}; "
            f"best_regression quality_corr={get(board, 'best_regression.quality_corr')}; "
            f"scheme={get(board, 'best_classification.scheme')}",
            str(paths["board_scheme_eval"]),
        ),
        item(
            "TacQuality label/score standard registry explicitly defines insertion and board good/bad/quality targets before scorer promotion.",
            "satisfied"
            if bool(get(label_registry, "registry_pass", False))
            and get(label_registry, "scientific_evidence") is False
            and get(label_registry, "tasks.insertion.binary_policy.bad") == [
                "pre_bounce_risk",
                "impact_or_recovery",
            ]
            and get(label_registry, "tasks.board.binary_policy.good") == ["good_smooth"]
            and get(label_registry, "tasks.board.target_force") is not None
            and "formal insertion baseline-vs-guided real rollout gate"
            in str(get(label_registry, "shared_guidance_policy.required_for_final_completion", ""))
            else "incomplete",
            "registry_pass="
            f"{get(label_registry, 'registry_pass')}; "
            f"insertion_policy={get(label_registry, 'tasks.insertion.binary_policy')}; "
            f"board_policy={get(label_registry, 'tasks.board.binary_policy')}; "
            f"board_target={get(label_registry, 'tasks.board.target_force')}; "
            f"required_final={get(label_registry, 'shared_guidance_policy.required_for_final_completion')}",
            str(paths["label_standard_registry"]),
        ),
        item(
            "Cached scorer labels comply with the TacQuality label/score standard registry, with compatible deviations explicitly documented.",
            "satisfied"
            if bool(get(label_compliance, "compliance_pass", False))
            and get(label_compliance, "scientific_evidence") is False
            and get(label_compliance, "required_checks_pass") is True
            and get(label_compliance, "compatible_deviations_pass") is True
            and (get(label_compliance, "summary.n_checks", 0) or 0) >= 10
            else "incomplete",
            "compliance_pass="
            f"{get(label_compliance, 'compliance_pass')}; "
            f"summary={get(label_compliance, 'summary')}; "
            f"interpretation={get(label_compliance, 'interpretation')}",
            str(paths["label_standard_compliance"]),
        ),
        item(
            "Unified task-conditioned differentiable scorer is trained/evaluated across insertion and board.",
            "satisfied"
            if (get(ptg, "mixed_group_cv.binary_auc.mean", 0.0) or 0.0) >= 0.95
            and (get(ptg, "mixed_group_cv.quality_corr.mean", 0.0) or 0.0) >= 0.70
            and bool(get(ptg, "gradient_sanity.usable_for_feature_guidance", False))
            else "incomplete",
            "binary_auc="
            f"{get(ptg, 'mixed_group_cv.binary_auc.mean')}; "
            f"quality_corr={get(ptg, 'mixed_group_cv.quality_corr.mean')}; "
            f"gradient_sanity={get(ptg, 'gradient_sanity')}",
            str(paths["ptg_proxy_eval"]),
        ),
        item(
            "Scorer behaves as a local action-gradient energy on both tasks.",
            "satisfied"
            if bool(get(scale, "overall_pass", False))
            and bool(get(robust, "overall_pass", False))
            and bool(get(score_landscape, "overall_pass", False))
            else "incomplete",
            "scale_sweep overall="
            f"{get(scale, 'overall_pass')}, insertion_improved={get(scale, 'insertion.recommended_improved_rate')}, "
            f"board_improved={get(scale, 'board.recommended_improved_rate')}; "
            "robustness overall="
            f"{get(robust, 'overall_pass')}, insertion_worst={get(robust, 'insertion.worst_perturbed_gradient_improved_rate')}, "
            f"board_worst={get(robust, 'board.worst_perturbed_gradient_improved_rate')}; "
            "landscape overall="
            f"{get(score_landscape, 'overall_pass')}, "
            f"insertion_grad_mean={get(score_landscape, 'insertion.gradient.grad_norm.mean')}, "
            f"board_grad_mean={get(score_landscape, 'board.gradient.grad_norm.mean')}",
            f"{paths['scale_sweep']} ; {paths['robustness']} ; {paths['score_landscape']}",
        ),
        item(
            "TacQuality scorer package satisfies the DP guidance contract, not merely offline classification accuracy.",
            "satisfied"
            if bool(get(guidance_contract, "guidance_contract_pass", False))
            and get(guidance_contract, "scientific_evidence_complete") is False
            and get(guidance_contract, "recommended_guidance_mode")
            == "final_clean_action_trust_region_refinement"
            and not (get(guidance_contract, "failed_required_checks", []) or [])
            else "incomplete",
            "guidance_contract_pass="
            f"{get(guidance_contract, 'guidance_contract_pass')}; "
            f"scientific_evidence_complete={get(guidance_contract, 'scientific_evidence_complete')}; "
            f"recommended_mode={get(guidance_contract, 'recommended_guidance_mode')}; "
            f"failed_required={get(guidance_contract, 'failed_required_checks')}; "
            f"interpretation={get(guidance_contract, 'deployment_interpretation')}",
            str(paths["guidance_contract"]),
        ),
        item(
            "Scorer-gradient refinement improves scorer outputs while preserving task quality proxies such as smoothness and action range.",
            "satisfied"
            if bool(get(proxy_alignment, "proxy_alignment_pass", False))
            and get(proxy_alignment, "scientific_evidence") is False
            and get(proxy_alignment, "task_pass.insertion") is True
            and get(proxy_alignment, "task_pass.board") is True
            and "Offline proxy alignment" in str(get(proxy_alignment, "remaining_gap", ""))
            else "incomplete",
            "proxy_alignment_pass="
            f"{get(proxy_alignment, 'proxy_alignment_pass')}; "
            f"task_pass={get(proxy_alignment, 'task_pass')}; "
            f"rows={get(proxy_alignment, 'rows')}; "
            f"remaining_gap={get(proxy_alignment, 'remaining_gap')}",
            str(paths["proxy_alignment_audit"]),
        ),
        item(
            "DP integration adapter exposes final clean-action TacQuality guidance contract.",
            "satisfied"
            if bool(get(dp_integration_adapter, "passes_integration_adapter_sanity", False))
            and (get(dp_integration_adapter, "insertion.improved_rate", 0.0) or 0.0) >= 0.95
            and (get(dp_integration_adapter, "board.improved_rate", 0.0) or 0.0) >= 0.95
            and (get(dp_integration_adapter, "minmax_insertion.improved_rate", 0.0) or 0.0) >= 0.95
            and (get(dp_integration_adapter, "normalizer_roundtrip.max_abs_error", 1.0) or 1.0) <= 1e-6
            and get(dp_integration_adapter, "insertion.integration_contract.reranking") is False
            and get(dp_integration_adapter, "insertion.integration_contract.every_step_ddpm_guidance") is False
            else "incomplete",
            "adapter_pass="
            f"{get(dp_integration_adapter, 'passes_integration_adapter_sanity')}; "
            f"insertion_improved={get(dp_integration_adapter, 'insertion.improved_rate')}; "
            f"board_improved={get(dp_integration_adapter, 'board.improved_rate')}; "
            f"minmax_improved={get(dp_integration_adapter, 'minmax_insertion.improved_rate')}; "
            f"roundtrip_error={get(dp_integration_adapter, 'normalizer_roundtrip.max_abs_error')}; "
            f"contract={get(dp_integration_adapter, 'insertion.integration_contract')}",
            str(paths["dp_integration_adapter"]),
        ),
        item(
            "Foresight bridge converts latent predictions into differentiable TacQuality marker/action inputs.",
            "satisfied"
            if bool(get(foresight_bridge, "passes_foresight_bridge_sanity", False))
            and get(foresight_bridge, "not_reranking") is True
            and get(foresight_bridge, "not_every_step_ddpm_guidance") is True
            and (get(foresight_bridge, "insertion.bridge_grad.positive_grad_rate", 0.0) or 0.0) >= 0.999
            and (get(foresight_bridge, "board.bridge_grad.positive_grad_rate", 0.0) or 0.0) >= 0.999
            and (get(foresight_bridge, "insertion.adapter_report.improved_rate", 0.0) or 0.0) >= 0.90
            and (get(foresight_bridge, "board.adapter_report.improved_rate", 0.0) or 0.0) >= 0.90
            else "incomplete",
            "bridge_pass="
            f"{get(foresight_bridge, 'passes_foresight_bridge_sanity')}; "
            f"mode={get(foresight_bridge, 'guidance_mode')}; "
            f"insertion_shapes={get(foresight_bridge, 'insertion.tactile_shapes')}; "
            f"board_shapes={get(foresight_bridge, 'board.tactile_shapes')}; "
            f"insertion_grad={get(foresight_bridge, 'insertion.bridge_grad.positive_grad_rate')}; "
            f"board_grad={get(foresight_bridge, 'board.bridge_grad.positive_grad_rate')}; "
            f"insertion_improved={get(foresight_bridge, 'insertion.adapter_report.improved_rate')}; "
            f"board_improved={get(foresight_bridge, 'board.adapter_report.improved_rate')}",
            str(paths["foresight_bridge"]),
        ),
        item(
            "Distilled scorer is checked on insertion clean-action refinement, not only board.",
            "satisfied"
            if bool(get(insertion_distilled, "overall_pass", False))
            and bool(get(insertion_distilled, "insertion_risk.passes_insertion_clean_refine_smoke", False))
            and bool(get(insertion_distilled, "distilled_energy.passes_insertion_clean_refine_smoke", False))
            else "incomplete",
            "overall_pass="
            f"{get(insertion_distilled, 'overall_pass')}; "
            f"insertion_default_delta={get(insertion_distilled, 'insertion_risk.summary.score_delta.mean')}; "
            f"distilled_delta={get(insertion_distilled, 'distilled_energy.summary.score_delta.mean')}; "
            f"distilled_improved={get(insertion_distilled, 'distilled_energy.summary.refined_beats_base_rate')}",
            str(paths["insertion_distilled_clean_refine"]),
        ),
        item(
            "Scorer-selection gate chooses a default scorer and a differentiable distilled ablation candidate.",
            "satisfied"
            if bool(get(selection_gate, "selection_gate_pass", False))
            and get(selection_gate, "selection.current_default_board_scorer") == "PTGProxyScorerV2Runtime"
            and get(selection_gate, "selection.promoted_ablation_candidate") == "DistilledTacQualityEnergyRuntime"
            and get(selection_gate, "selection.distilled_replacement_status") == "not_yet_replacement"
            and get(selection_gate, "selection.action_aware_marker_status")
            == "line_search_quality_mode_guidance_candidate"
            and get(selection_gate, "evidence.gradient_guidance_contract.scale_sweep.overall_pass") is True
            and get(selection_gate, "evidence.gradient_guidance_contract.robustness.overall_pass") is True
            and (
                get(selection_gate, "evidence.gradient_guidance_contract.scale_sweep.insertion_improved_rate", 0.0)
                or 0.0
            )
            >= 0.95
            and (
                get(selection_gate, "evidence.gradient_guidance_contract.scale_sweep.board_improved_rate", 0.0)
                or 0.0
            )
            >= 0.95
            else "incomplete",
            "selection_gate_pass="
            f"{get(selection_gate, 'selection_gate_pass')}; "
            f"default={get(selection_gate, 'selection.current_default_board_scorer')}; "
            f"candidate={get(selection_gate, 'selection.promoted_ablation_candidate')}; "
            f"replacement_status={get(selection_gate, 'selection.distilled_replacement_status')}; "
            f"action_aware_status={get(selection_gate, 'selection.action_aware_marker_status')}; "
            "gradient_scale_pass="
            f"{get(selection_gate, 'evidence.gradient_guidance_contract.scale_sweep.overall_pass')}; "
            "gradient_robustness_pass="
            f"{get(selection_gate, 'evidence.gradient_guidance_contract.robustness.overall_pass')}",
            str(paths["scorer_selection_gate"]),
        ),
        item(
            "Scorer decision matrix ranks candidates by episode generalization, quality alignment, differentiability, local guidance stability, task coverage, cross-task transfer, real evidence, and novelty.",
            "satisfied"
            if bool(get(decision_matrix, "passes_decision_matrix_gate", False))
            and get(decision_matrix, "scientific_evidence") is False
            and get(decision_matrix, "recommendation.formal_default_insertion") == "InsertionRiskScorerRuntime"
            and get(decision_matrix, "recommendation.formal_default_board") == "PTGProxyScorerV2Runtime"
            and get(decision_matrix, "recommendation.innovation_ablation") == "DistilledTacQualityEnergyRuntime"
            and get(decision_matrix, "recommendation.optional_unified_action_conditioned_ablation")
            == "ActionAwareScorerRuntime"
            and "Real baseline-vs-guided" in str(get(decision_matrix, "remaining_gap", ""))
            else "incomplete",
            "passes="
            f"{get(decision_matrix, 'passes_decision_matrix_gate')}; "
            f"ranked_offline={get(decision_matrix, 'ranked_offline')}; "
            f"recommendation={get(decision_matrix, 'recommendation')}; "
            f"remaining_gap={get(decision_matrix, 'remaining_gap')}",
            str(paths["scorer_decision_matrix"]),
        ),
        item(
            "Action-aware unified scorer candidate is evaluated with episode-level metrics and differentiable runtime gradients; quality-mode line-search guidance passes, but it is not promoted because zero-shot cross-task transfer remains weak.",
            "satisfied"
            if (get(action_aware_eval, "mixed_group_cv.binary_auc.mean", 0.0) or 0.0) >= 0.95
            and (get(action_aware_eval, "mixed_group_cv.score_corr.mean", 0.0) or 0.0) >= 0.70
            and bool(get(action_aware_runtime, "usable_for_guidance", False))
            and (get(action_aware_eval, "cross_task.insertion_to_board.binary_macro_f1", 1.0) or 1.0) < 0.60
            and (get(action_aware_eval, "cross_task.board_to_insertion.binary_macro_f1", 1.0) or 1.0) < 0.60
            and get(action_aware_guidance, "passes_guidance_suitability") is True
            and get(action_aware_guidance, "recommended_mode") == "quality"
            and (get(action_aware_guidance, "modes.quality.gradient_probe.mixed.improved_rate", 1.0) or 1.0) < 0.95
            and (get(action_aware_guidance, "modes.quality.gradient_probe.mixed.line_search.accepted_rate", 0.0) or 0.0) >= 0.95
            else "incomplete",
            "mixed_auc="
            f"{get(action_aware_eval, 'mixed_group_cv.binary_auc.mean')}; "
            f"score_corr={get(action_aware_eval, 'mixed_group_cv.score_corr.mean')}; "
            f"usable={get(action_aware_runtime, 'usable_for_guidance')}; "
            f"i2b_macro_f1={get(action_aware_eval, 'cross_task.insertion_to_board.binary_macro_f1')}; "
            f"b2i_macro_f1={get(action_aware_eval, 'cross_task.board_to_insertion.binary_macro_f1')}; "
            f"guidance_pass={get(action_aware_guidance, 'passes_guidance_suitability')}; "
            f"recommended={get(action_aware_guidance, 'recommended_mode')}; "
            f"quality_fixed={get(action_aware_guidance, 'modes.quality.gradient_probe.mixed.improved_rate')}; "
            f"quality_line_search={get(action_aware_guidance, 'modes.quality.gradient_probe.mixed.line_search.accepted_rate')}",
            str(paths["action_aware_guidance_suitability"]),
        ),
        item(
            "Offline production-readiness gate passes while preserving the real-robot validation gap.",
            "satisfied"
            if bool(get(offline, "offline_production_gate_pass", False))
            and get(offline, "remaining_required_step") == "Real robot / final production policy validation."
            else "incomplete",
            "offline_gate="
            f"{get(offline, 'offline_production_gate_pass')}; "
            f"remaining={get(offline, 'remaining_required_step')}",
            str(paths["offline_gate"]),
        ),
        item(
            "Deployment manifest exists and records the offline-ready but not robot-validated package.",
            "satisfied"
            if bool(get(manifest, "deployment_manifest_pass", False))
            and get(manifest, "deployment_policy.completion_status") == "offline_ready_not_robot_validated"
            else "incomplete",
            "manifest_pass="
            f"{get(manifest, 'deployment_manifest_pass')}; "
            f"completion_status={get(manifest, 'deployment_policy.completion_status')}; "
            f"manifest_git_commit={get(manifest, 'git_commit')}",
            str(paths["manifest"]),
        ),
        item(
            "Evidence summary explicitly keeps the objective incomplete until real rollout validation.",
            "satisfied"
            if get(summary, "completion_assessment.objective_complete", get(summary, "objective_complete")) is False
            and any(
                token
                in str(get(summary, "completion_assessment.reason", get(summary, "reason", ""))).lower()
                for token in ["real-robot", "real rollout", "real-rollout", "acceptance protocol"]
            )
            else "incomplete",
            "objective_complete="
            f"{get(summary, 'completion_assessment.objective_complete', get(summary, 'objective_complete'))}; "
            f"reason={get(summary, 'completion_assessment.reason', get(summary, 'reason'))}",
            str(paths["evidence_summary"]),
        ),
        item(
            "Work and research records document design, standards, experiments, and deployment policy.",
            "satisfied"
            if file_contains(paths["record"], ["TacQualityEnergy", "GroupKFold", "offline production gate"])
            and file_contains(paths["eval_doc"], ["TacQualityEnergy", "GroupKFold", "可梯度引导"])
            and file_contains(paths["deploy_doc"], ["TacQualityEnergy", "offline production gate", "Real robot"])
            else "incomplete",
            "Required snippets found in 工作记录codex.txt, PTG evaluation doc, and deployment manual.",
            f"{paths['record']} ; {paths['eval_doc']} ; {paths['deploy_doc']}",
        ),
        item(
            "Runtime scorer visualizations exist for score, task space, quality proxies, and gradient norms.",
            "satisfied"
            if bool(get(runtime_visualization, "visualization_pass", False))
            and get(runtime_visualization, "figures.runtime_pca_task_score_grad") is not None
            and get(runtime_visualization, "figures.insertion_runtime_quality_reason") is not None
            and get(runtime_visualization, "figures.board_runtime_force_smoothness") is not None
            else "incomplete",
            "visualization_pass="
            f"{get(runtime_visualization, 'visualization_pass')}; "
            f"figures={get(runtime_visualization, 'figures')}",
            str(paths["runtime_visualization"]),
        ),
        item(
            "Formal rollout validation preparation tool exists and produces gate-ready templates/commands.",
            "satisfied"
            if prep_smoke is not None
            and get(prep_smoke, "task") == "board"
            and get(prep_smoke, "ready_for_quality_gate") is True
            and get(prep_smoke, "outputs.pairing_csv_template") is not None
            and get(prep_smoke, "outputs.metadata_csv_template") is not None
            and "eval_real_rollout_quality_gate.py" in str(get(prep_smoke, "gate_command", ""))
            else "incomplete",
            "prep_smoke="
            f"task={get(prep_smoke, 'task')}, "
            f"ready={get(prep_smoke, 'ready_for_quality_gate')}, "
            f"baseline_n={get(prep_smoke, 'baseline.n')}, guided_n={get(prep_smoke, 'guided.n')}, "
            f"pairing_ready={get(prep_smoke, 'pairing_csv_check.ready')}, "
            f"metadata_ready={get(prep_smoke, 'metadata_csv_check.ready')}, "
            f"pairing={get(prep_smoke, 'outputs.pairing_csv_template')}, "
            f"metadata={get(prep_smoke, 'outputs.metadata_csv_template')}",
            str(paths["real_rollout_prep_smoke"]),
        ),
        item(
            "Formal rollout sample-size plans exist for insertion and board.",
            "satisfied"
            if get(insertion_plan, "recommendation.paired_n_pairs", 0) >= 10
            and get(board_plan, "recommendation.paired_n_pairs", 0) >= 10
            and "eval_real_rollout_quality_gate.py" in str(get(insertion_plan, "commands.gate", ""))
            and "eval_real_rollout_quality_gate.py" in str(get(board_plan, "commands.gate", ""))
            else "incomplete",
            "insertion_paired_n="
            f"{get(insertion_plan, 'recommendation.paired_n_pairs')}, "
            f"board_paired_n={get(board_plan, 'recommendation.paired_n_pairs')}, "
            f"insertion_unpaired_n={get(insertion_plan, 'recommendation.unpaired_n_per_group')}, "
            f"board_unpaired_n={get(board_plan, 'recommendation.unpaired_n_per_group')}",
            f"{paths['insertion_sample_size_plan']} ; {paths['board_sample_size_plan']}",
        ),
        item(
            "Formal paired rollout experiment packet exists for both tasks.",
            "satisfied"
            if get(experiment_packet, "tasks.insertion.paired_n_pairs", 0) >= 10
            and get(experiment_packet, "tasks.board.paired_n_pairs", 0) >= 10
            and "eval_real_rollout_quality_gate.py" in str(get(experiment_packet, "tasks.insertion.gate_command", ""))
            and "eval_real_rollout_quality_gate.py" in str(get(experiment_packet, "tasks.board.gate_command", ""))
            and "eval_real_rollout_scorer_ablation_gate.py"
            in str(get(experiment_packet, "tasks.insertion.ablation_gate_command", ""))
            and "eval_real_rollout_scorer_ablation_gate.py"
            in str(get(experiment_packet, "tasks.board.ablation_gate_command", ""))
            else "incomplete",
            "packet="
            f"{get(experiment_packet, 'out_dir')}, "
            f"insertion_pairs={get(experiment_packet, 'tasks.insertion.paired_n_pairs')}, "
            f"board_pairs={get(experiment_packet, 'tasks.board.paired_n_pairs')}, "
            f"insertion_ablation_cmd={get(experiment_packet, 'tasks.insertion.ablation_gate_command') is not None}, "
            f"board_ablation_cmd={get(experiment_packet, 'tasks.board.ablation_gate_command') is not None}",
            str(paths["real_rollout_experiment_packet"]),
        ),
        item(
            "Formal rollout gate runner can preflight and execute the remaining two-arm and three-arm gates.",
            "satisfied"
            if formal_rollout_gate_runner is not None
            and get(formal_rollout_gate_runner, "preflight_ready") is False
            and get(formal_rollout_gate_runner, "scientific_evidence") is False
            and "run_gates" in str(get(formal_rollout_gate_runner, "run_skip_reason"))
            and get(formal_rollout_gate_runner, "tasks.insertion.commands.two_arm") is not None
            and get(formal_rollout_gate_runner, "tasks.board.commands.three_arm") is not None
            else "incomplete",
            "preflight_ready="
            f"{get(formal_rollout_gate_runner, 'preflight_ready')}; "
            f"use_generated_pairing={get(formal_rollout_gate_runner, 'use_generated_pairing')}; "
            f"scientific_evidence={get(formal_rollout_gate_runner, 'scientific_evidence')}; "
            f"run_skip_reason={get(formal_rollout_gate_runner, 'run_skip_reason')}",
            str(paths["formal_rollout_gate_runner"]),
        ),
        item(
            "Formal launch sheet lists baseline/default/distilled server commands, rollout dirs, and all-task gate runner command.",
            "satisfied"
            if bool(get(formal_launch_sheet, "launch_sheet_ready", False))
            and get(formal_launch_sheet, "scientific_evidence") is False
            and get(formal_launch_sheet, "not_reranking") is True
            and get(formal_launch_sheet, "not_every_step_ddpm_guidance") is True
            and "serve_dp_tac_quality_guided" in str(get(formal_launch_sheet, "tasks.insertion.launch_commands.baseline", ""))
            and "--disable_guidance" in str(get(formal_launch_sheet, "tasks.insertion.launch_commands.baseline", ""))
            and "serve_dp_tac_quality_guided" in str(get(formal_launch_sheet, "tasks.insertion.launch_commands.default_guided", ""))
            and "serve_dp_tac_quality_guided" in str(get(formal_launch_sheet, "tasks.insertion.launch_commands.distilled_guided", ""))
            and "serve_dp_tac_quality_guided" in str(get(formal_launch_sheet, "tasks.board.launch_commands.baseline", ""))
            and "--disable_guidance" in str(get(formal_launch_sheet, "tasks.board.launch_commands.baseline", ""))
            and "serve_dp_tac_quality_guided" in str(get(formal_launch_sheet, "tasks.board.launch_commands.default_guided", ""))
            and "serve_dp_tac_quality_guided" in str(get(formal_launch_sheet, "tasks.board.launch_commands.distilled_guided", ""))
            and get(load_json(paths["guided_server_insertion_baseline_no_guidance_smoke"]), "dry_run_guidance_smoke_pass") is True
            and get(load_json(paths["guided_server_board_baseline_no_guidance_smoke"]), "dry_run_guidance_smoke_pass") is True
            and "run_formal_tac_quality_rollout_gates.py" in str(get(formal_launch_sheet, "all_tasks_gate_runner_command", ""))
            and "--use_generated_pairing" in str(get(formal_launch_sheet, "all_tasks_gate_runner_command", ""))
            and "build_tac_quality_rollout_pairing.py" in str(get(formal_launch_sheet, "post_collection_pairing_command", ""))
            else "incomplete",
            "launch_sheet_ready="
            f"{get(formal_launch_sheet, 'launch_sheet_ready')}; "
            f"rollout_root={get(formal_launch_sheet, 'rollout_root')}; "
            f"generated_pairing_dir={get(formal_launch_sheet, 'generated_pairing_dir')}; "
            f"insertion_ports={get(formal_launch_sheet, 'tasks.insertion.ports')}; "
            f"board_ports={get(formal_launch_sheet, 'tasks.board.ports')}; "
            f"post_pairing={get(formal_launch_sheet, 'post_collection_pairing_command')}; "
            "baseline_smoke="
            f"{get(load_json(paths['guided_server_insertion_baseline_no_guidance_smoke']), 'dry_run_guidance_smoke_pass')}/"
            f"{get(load_json(paths['guided_server_board_baseline_no_guidance_smoke']), 'dry_run_guidance_smoke_pass')}",
            str(paths["formal_launch_sheet"]),
        ),
        item(
            "Formal launch sheet server commands all pass dry-run command smoke.",
            "satisfied"
            if bool(get(formal_launch_sheet_smoke, "overall_pass", False))
            and get(formal_launch_sheet_smoke, "scientific_evidence") is False
            and get(formal_launch_sheet_smoke, "checks.all_commands_present") is True
            and get(formal_launch_sheet_smoke, "checks.formal_six_commands_present") is True
            and get(formal_launch_sheet_smoke, "checks.optional_action_aware_commands_present") is True
            and get(formal_launch_sheet_smoke, "checks.all_commands_pass_process") is True
            and get(formal_launch_sheet_smoke, "checks.all_commands_pass_output_contract") is True
            and get(formal_launch_sheet_smoke, "checks.baseline_commands_disable_guidance") is True
            and get(formal_launch_sheet_smoke, "checks.guided_commands_have_gradients") is True
            else "incomplete",
            "overall_pass="
            f"{get(formal_launch_sheet_smoke, 'overall_pass')}; "
            f"checks={get(formal_launch_sheet_smoke, 'checks')}; "
            "commands="
            f"{[(row.get('task'), row.get('arm'), row.get('passes_launch_command_smoke'), get(row, 'output.guidance_disabled')) for row in (get(formal_launch_sheet_smoke, 'commands', []) or [])]}",
            str(paths["formal_launch_sheet_smoke"]),
        ),
        item(
            "Formal collection readiness dashboard tracks HDF5 counts, templates, and gate readiness.",
            "satisfied"
            if formal_collection_readiness is not None
            and get(formal_collection_readiness, "scientific_evidence") is False
            and get(formal_collection_readiness, "all_collection_dirs_exist") is True
            and get(formal_collection_readiness, "all_templates_exist") is True
            and get(formal_collection_readiness, "tasks.insertion.arms.baseline.needed") is not None
            and get(formal_collection_readiness, "tasks.board.arms.distilled_guided.needed") is not None
            and get(formal_collection_readiness, "ready_for_gate_runner") is False
            else "incomplete",
            "dirs_exist="
            f"{get(formal_collection_readiness, 'all_collection_dirs_exist')}; "
            f"templates_exist={get(formal_collection_readiness, 'all_templates_exist')}; "
            f"ready_for_gate_runner={get(formal_collection_readiness, 'ready_for_gate_runner')}; "
            f"missing_items={get(formal_collection_readiness, 'missing_items')}",
            str(paths["formal_collection_readiness"]),
        ),
        item(
            "Formal rollout pairing generator can create concrete pairing/metadata CSVs from collected HDF5s.",
            "satisfied"
            if formal_rollout_pairing is not None
            and get(formal_rollout_pairing, "scientific_evidence") is False
            and get(formal_rollout_pairing, "overall_ready") is False
            and get(formal_rollout_pairing, "schedule_used") is True
            and get(formal_rollout_pairing, "tasks.insertion.schedule_mode") is True
            and get(formal_rollout_pairing, "tasks.board.schedule_mode") is True
            and get(formal_rollout_pairing, "tasks.insertion.outputs.pairing_csv") is not None
            and get(formal_rollout_pairing, "tasks.insertion.outputs.three_arm_pairing_csv") is not None
            and get(formal_rollout_pairing, "tasks.board.outputs.pairing_csv") is not None
            and get(formal_rollout_pairing, "tasks.board.outputs.three_arm_pairing_csv") is not None
            else "incomplete",
            "overall_ready="
            f"{get(formal_rollout_pairing, 'overall_ready')}; "
            f"schedule_used={get(formal_rollout_pairing, 'schedule_used')}; "
            f"insertion_pairs={get(formal_rollout_pairing, 'tasks.insertion.n_pairs')}; "
            f"board_pairs={get(formal_rollout_pairing, 'tasks.board.n_pairs')}; "
            f"next={get(formal_rollout_pairing, 'next_required_step')}",
            str(paths["formal_rollout_pairing"]),
        ),
        item(
            "Generated pairing/metadata completeness audit tracks whether formal gate inputs are manually review-ready.",
            "satisfied"
            if pairing_metadata_audit is not None
            and get(pairing_metadata_audit, "scientific_evidence") is False
            and get(pairing_metadata_audit, "all_tasks_ready") is False
            and get(pairing_metadata_audit, "tasks.insertion.counts.two_arm_rows") is not None
            and get(pairing_metadata_audit, "tasks.board.counts.metadata_rows") is not None
            else "incomplete",
            "all_tasks_ready="
            f"{get(pairing_metadata_audit, 'all_tasks_ready')}; "
            f"insertion_counts={get(pairing_metadata_audit, 'tasks.insertion.counts')}; "
            f"board_counts={get(pairing_metadata_audit, 'tasks.board.counts')}",
            str(paths["pairing_metadata_audit"]),
        ),
        item(
            "Formal rollout HDF5 schema audit tracks whether collected files contain force, tactile marker, and action fields.",
            "satisfied"
            if hdf5_schema_audit is not None
            and get(hdf5_schema_audit, "scientific_evidence") is False
            and get(hdf5_schema_audit, "all_tasks_ready") is False
            and get(hdf5_schema_audit, "tasks.insertion.arms.baseline.n_hdf5") is not None
            and get(hdf5_schema_audit, "tasks.board.arms.distilled_guided.n_hdf5") is not None
            else "incomplete",
            "all_tasks_ready="
            f"{get(hdf5_schema_audit, 'all_tasks_ready')}; "
            f"insertion_baseline_n={get(hdf5_schema_audit, 'tasks.insertion.arms.baseline.n_hdf5')}; "
            f"board_distilled_n={get(hdf5_schema_audit, 'tasks.board.arms.distilled_guided.n_hdf5')}",
            str(paths["hdf5_schema_audit"]),
        ),
        item(
            "Generated-pairing formal gate runner passes synthetic end-to-end smoke.",
            "satisfied"
            if generated_pairing_gate_runner_smoke is not None
            and get(generated_pairing_gate_runner_smoke, "overall_pass") is True
            and get(generated_pairing_gate_runner_smoke, "scientific_evidence") is False
            and get(generated_pairing_gate_runner_smoke, "pairing_report.overall_ready") is True
            and get(generated_pairing_gate_runner_smoke, "gate_report.use_generated_pairing") is True
            and get(generated_pairing_gate_runner_smoke, "gate_report.preflight_ready") is True
            and get(generated_pairing_gate_runner_smoke, "gate_report.all_requested_gates_passed") is True
            else "incomplete",
            "overall_pass="
            f"{get(generated_pairing_gate_runner_smoke, 'overall_pass')}; "
            f"pairing_ready={get(generated_pairing_gate_runner_smoke, 'pairing_report.overall_ready')}; "
            f"use_generated_pairing={get(generated_pairing_gate_runner_smoke, 'gate_report.use_generated_pairing')}; "
            f"all_requested_gates_passed={get(generated_pairing_gate_runner_smoke, 'gate_report.all_requested_gates_passed')}",
            str(paths["generated_pairing_gate_runner_smoke"]),
        ),
        item(
            "Real rollout source audit prevents synthetic/smoke artifacts from closing the validation gap.",
            "satisfied"
            if real_rollout_source_audit is not None
            and get(real_rollout_source_audit, "synthetic_guardrail_pass") is True
            and get(real_rollout_source_audit, "all_four_real_evidence_present") is False
            and (get(real_rollout_source_audit, "n_blockers", 0) or 0) == 4
            else "incomplete",
            "all_four_real_evidence_present="
            f"{get(real_rollout_source_audit, 'all_four_real_evidence_present')}; "
            f"n_real_evidence={get(real_rollout_source_audit, 'n_real_evidence')}; "
            f"n_blockers={get(real_rollout_source_audit, 'n_blockers')}; "
            f"synthetic_guardrail_pass={get(real_rollout_source_audit, 'synthetic_guardrail_pass')}",
            str(paths["real_rollout_source_audit"]),
        ),
        item(
            "Post-collection pipeline orchestrates pairing, metadata audit, gate preflight, and source audit.",
            "satisfied"
            if post_collection_pipeline is not None
            and get(post_collection_pipeline, "pipeline_pass") is True
            and get(post_collection_pipeline, "run_gates_requested") is False
            and get(post_collection_pipeline, "run_optional_action_aware_gate_requested") is False
            and get(post_collection_pipeline, "can_run_gates") is False
            and get(post_collection_pipeline, "metadata_audit.all_tasks_ready") is False
            and get(post_collection_pipeline, "hdf5_schema_audit.all_tasks_ready") is False
            and get(post_collection_pipeline, "optional_action_aware.formal_gate_dependency") is False
            and get(post_collection_pipeline, "optional_action_aware.run_gates_requested") is False
            and get(post_collection_pipeline, "optional_action_aware.tasks.insertion.checks.action_aware_guided.n_hdf5")
            is not None
            and get(post_collection_pipeline, "optional_action_aware.tasks.board.checks.action_aware_guided.n_hdf5")
            is not None
            and get(post_collection_pipeline, "source_audit.n_blockers") == 4
            else "incomplete",
            "pipeline_pass="
            f"{get(post_collection_pipeline, 'pipeline_pass')}; "
            f"can_run_gates={get(post_collection_pipeline, 'can_run_gates')}; "
            f"metadata_ready={get(post_collection_pipeline, 'metadata_audit.all_tasks_ready')}; "
            f"schema_ready={get(post_collection_pipeline, 'hdf5_schema_audit.all_tasks_ready')}; "
            f"action_aware_preflight={get(post_collection_pipeline, 'optional_action_aware.preflight_ready')}; "
            f"action_aware_formal_dep={get(post_collection_pipeline, 'optional_action_aware.formal_gate_dependency')}; "
            f"source_blockers={get(post_collection_pipeline, 'source_audit.n_blockers')}",
            str(paths["post_collection_pipeline"]),
        ),
        item(
            "Post-collection pipeline passes synthetic run-gates smoke while preserving the real-evidence gap.",
            "satisfied"
            if post_collection_pipeline_smoke is not None
            and get(post_collection_pipeline_smoke, "overall_pass") is True
            and get(post_collection_pipeline_smoke, "scientific_evidence") is False
            and get(post_collection_pipeline_smoke, "checks.pipeline_pass") is True
            and get(post_collection_pipeline_smoke, "checks.can_run_gates") is True
            and get(post_collection_pipeline_smoke, "checks.schema_ready") is True
            and get(post_collection_pipeline_smoke, "checks.gates_passed") is True
            and get(post_collection_pipeline_smoke, "checks.optional_action_aware_preflight_ready") is True
            and get(post_collection_pipeline_smoke, "checks.optional_action_aware_gates_passed") is True
            and get(post_collection_pipeline_smoke, "checks.optional_action_aware_not_formal_dependency") is True
            and get(post_collection_pipeline_smoke, "checks.source_guardrail_keeps_formal_gap") is True
            else "incomplete",
            "overall_pass="
            f"{get(post_collection_pipeline_smoke, 'overall_pass')}; "
            f"checks={get(post_collection_pipeline_smoke, 'checks')}",
            str(paths["post_collection_pipeline_smoke"]),
        ),
        item(
            "Collected-HDF5 finalize utility passes synthetic copy/refuse/source-dir smoke before real rollout collection.",
            "satisfied"
            if finalize_collected_hdf5_smoke is not None
            and get(finalize_collected_hdf5_smoke, "overall_pass") is True
            and get(finalize_collected_hdf5_smoke, "scientific_evidence") is False
            and get(finalize_collected_hdf5_smoke, "checks.copy_finalize_pass") is True
            and get(finalize_collected_hdf5_smoke, "checks.copy_operation") is True
            and get(finalize_collected_hdf5_smoke, "checks.copy_keeps_source") is True
            and get(finalize_collected_hdf5_smoke, "checks.refuse_existing_target") is True
            and get(finalize_collected_hdf5_smoke, "checks.source_dir_finalize_pass") is True
            and get(finalize_collected_hdf5_smoke, "checks.source_dir_picks_newest") is True
            else "incomplete",
            "overall_pass="
            f"{get(finalize_collected_hdf5_smoke, 'overall_pass')}; "
            f"checks={get(finalize_collected_hdf5_smoke, 'checks')}; "
            f"reports={get(finalize_collected_hdf5_smoke, 'reports')}",
            str(paths["finalize_collected_hdf5_smoke"]),
        ),
        item(
            "Finalize-and-refresh wrapper passes synthetic smoke so next-row artifacts can be regenerated after each collected HDF5.",
            "satisfied"
            if finalize_and_refresh_smoke is not None
            and get(finalize_and_refresh_smoke, "overall_pass") is True
            and get(finalize_and_refresh_smoke, "scientific_evidence") is False
            and get(finalize_and_refresh_smoke, "checks.wrapper_pass") is True
            and get(finalize_and_refresh_smoke, "checks.target_exists") is True
            and get(finalize_and_refresh_smoke, "checks.source_kept_by_copy") is True
            and get(finalize_and_refresh_smoke, "checks.skip_refresh_used") is True
            else "incomplete",
            "overall_pass="
            f"{get(finalize_and_refresh_smoke, 'overall_pass')}; "
            f"checks={get(finalize_and_refresh_smoke, 'checks')}; "
            f"runner={get(finalize_and_refresh_smoke, 'runner_summary')}",
            str(paths["finalize_and_refresh_smoke"]),
        ),
        item(
            "Real-rollout acceptance protocol defines the exact blocker-closing two-arm and three-arm evidence criteria.",
            "satisfied"
            if real_rollout_acceptance_protocol is not None
            and get(real_rollout_acceptance_protocol, "protocol_pass") is True
            and get(real_rollout_acceptance_protocol, "scientific_evidence") is False
            and get(real_rollout_acceptance_protocol, "tasks.insertion.collection.paired_n_pairs", 0) >= 10
            and get(real_rollout_acceptance_protocol, "tasks.board.collection.paired_n_pairs", 0) >= 10
            and len(get(real_rollout_acceptance_protocol, "completion_blockers_to_close", []) or []) == 4
            and "synthetic HDF5 smoke outputs"
            in str(get(real_rollout_acceptance_protocol, "cannot_count_as_completion", ""))
            and "eval_real_rollout_quality_gate.py"
            in str(get(real_rollout_acceptance_protocol, "tasks.insertion.two_arm_gate.command", ""))
            and "eval_real_rollout_scorer_ablation_gate.py"
            in str(get(real_rollout_acceptance_protocol, "tasks.board.three_arm_ablation_gate.command", ""))
            else "incomplete",
            "protocol_pass="
            f"{get(real_rollout_acceptance_protocol, 'protocol_pass')}; "
            f"scientific_evidence={get(real_rollout_acceptance_protocol, 'scientific_evidence')}; "
            "paired_n_pairs="
            f"{get(real_rollout_acceptance_protocol, 'tasks.insertion.collection.paired_n_pairs')}/"
            f"{get(real_rollout_acceptance_protocol, 'tasks.board.collection.paired_n_pairs')}; "
            "blockers="
            f"{get(real_rollout_acceptance_protocol, 'completion_blockers_to_close')}",
            str(paths["real_rollout_acceptance_protocol"]),
        ),
        item(
            "Formal paired12 rollout runbook gives a single executable handoff from protocol to collection and gates.",
            "satisfied"
            if formal_rollout_runbook is not None
            and get(formal_rollout_runbook, "runbook_pass") is True
            and get(formal_rollout_runbook, "scientific_evidence") is False
            and get(formal_rollout_runbook, "protocol_pass") is True
            and get(formal_rollout_runbook, "launch_sheet_ready") is True
            and get(formal_rollout_runbook, "pipeline_pass") is True
            and get(formal_rollout_runbook, "tasks.insertion.paired_n_pairs", 0) >= 10
            and get(formal_rollout_runbook, "tasks.board.paired_n_pairs", 0) >= 10
            and len(get(formal_rollout_runbook, "completion_blockers_to_close", []) or []) == 4
            and get(formal_rollout_runbook, "collection_schedule.schedule_pass") is True
            and str(get(formal_rollout_runbook, "collection_schedule.csv", "")).endswith(
                "tac_quality_collection_schedule.csv"
            )
            and "run_tac_quality_post_collection_pipeline.py"
            in str(get(formal_rollout_runbook, "post_collection_commands.pipeline_run_gates", ""))
            and "build_tac_quality_collection_schedule.py"
            in str(get(formal_rollout_runbook, "post_collection_commands.build_collection_schedule", ""))
            and "build_tac_quality_next_collection_step.py"
            in str(get(formal_rollout_runbook, "post_collection_commands.next_collection_step", ""))
            and "run_tac_quality_next_collection_step_smoke.py"
            in str(get(formal_rollout_runbook, "post_collection_commands.next_collection_step_smoke_runner", ""))
            and "conda run -n TactileACT"
            in str(get(formal_rollout_runbook, "post_collection_commands.next_collection_step_smoke_runner", ""))
            and "build_tac_quality_current_collection_gate.py"
            in str(get(formal_rollout_runbook, "post_collection_commands.current_collection_gate", ""))
            and "build_tac_quality_current_collection_handoff.py"
            in str(get(formal_rollout_runbook, "post_collection_commands.current_collection_handoff", ""))
            and "finalize_and_refresh_tac_quality_collection.py"
            in str(get(formal_rollout_runbook, "post_collection_commands.finalize_and_refresh_collected_hdf5", ""))
            and "finalize_tac_quality_collected_hdf5.py"
            in str(get(formal_rollout_runbook, "post_collection_commands.finalize_collected_hdf5", ""))
            and get(formal_rollout_runbook, "collection_progress.progress_pass") is True
            and get(formal_rollout_runbook, "next_collection_step.next_step_pass") is True
            and "--dry_run_guidance_smoke"
            in str(get(formal_rollout_runbook, "next_collection_step.pre_collection_dry_run_command", ""))
            and "--smoke_output"
            in str(get(formal_rollout_runbook, "next_collection_step.pre_collection_dry_run_command", ""))
            and "serve_dp_tac_quality_guided"
            in str(get(formal_rollout_runbook, "tasks.insertion.arms", ""))
            else "incomplete",
            "runbook_pass="
            f"{get(formal_rollout_runbook, 'runbook_pass')}; "
            f"ready_for_gate_runner={get(formal_rollout_runbook, 'ready_for_gate_runner')}; "
            f"rollout_root={get(formal_rollout_runbook, 'rollout_root')}; "
            "paired_n_pairs="
            f"{get(formal_rollout_runbook, 'tasks.insertion.paired_n_pairs')}/"
            f"{get(formal_rollout_runbook, 'tasks.board.paired_n_pairs')}; "
            "post_collection="
            f"{get(formal_rollout_runbook, 'post_collection_commands.pipeline_run_gates')}; "
            f"pre_collection_dry_run={get(formal_rollout_runbook, 'next_collection_step.pre_collection_dry_run_command')}; "
            f"finalize_and_refresh={get(formal_rollout_runbook, 'post_collection_commands.finalize_and_refresh_collected_hdf5')}; "
            f"finalize={get(formal_rollout_runbook, 'post_collection_commands.finalize_collected_hdf5')}",
            str(paths["formal_rollout_runbook"]),
        ),
        item(
            "Formal paired12 rollout runbook smoke checks command, gate, and guardrail consistency before collection.",
            "satisfied"
            if formal_rollout_runbook_smoke is not None
            and get(formal_rollout_runbook_smoke, "overall_pass") is True
            and get(formal_rollout_runbook_smoke, "scientific_evidence") is False
            and get(formal_rollout_runbook_smoke, "checks.runbook_pass") is True
            and get(formal_rollout_runbook_smoke, "checks.ready_for_gate_runner_false_until_collection") is True
            and get(formal_rollout_runbook_smoke, "checks.all_tasks_pass") is True
            and get(formal_rollout_runbook_smoke, "checks.completion_blockers_four") is True
            and get(formal_rollout_runbook_smoke, "checks.post_collection_run_gates_command_present") is True
            and get(formal_rollout_runbook_smoke, "checks.build_schedule_command_present") is True
            and get(formal_rollout_runbook_smoke, "checks.collection_progress_command_present") is True
            and get(formal_rollout_runbook_smoke, "checks.next_collection_step_command_present") is True
            and get(formal_rollout_runbook_smoke, "checks.next_collection_step_smoke_runner_command_present") is True
            and get(formal_rollout_runbook_smoke, "checks.current_collection_gate_command_present") is True
            and get(formal_rollout_runbook_smoke, "checks.current_collection_handoff_command_present") is True
            and get(formal_rollout_runbook_smoke, "checks.finalize_and_refresh_command_present") is True
            and get(formal_rollout_runbook_smoke, "checks.finalize_and_refresh_source_dir_command_present") is True
            and get(formal_rollout_runbook_smoke, "checks.finalize_collected_hdf5_command_present") is True
            and get(formal_rollout_runbook_smoke, "checks.finalize_source_dir_command_present") is True
            and get(formal_rollout_runbook_smoke, "checks.next_step_finalize_template_present") is True
            and get(formal_rollout_runbook_smoke, "checks.next_step_pre_collection_dry_run_present") is True
            and get(formal_rollout_runbook_smoke, "checks.next_step_pre_collection_dry_run_output_present") is True
            and get(formal_rollout_runbook_smoke, "checks.collection_steps_require_pre_collection_dry_run") is True
            and get(formal_rollout_runbook_smoke, "checks.runbook_references_schedule_csv") is True
            and get(formal_rollout_runbook_smoke, "checks.runbook_references_collection_progress") is True
            and get(formal_rollout_runbook_smoke, "checks.runbook_references_next_collection_step") is True
            and get(formal_rollout_runbook_smoke, "checks.schedule_pass") is True
            and get(formal_rollout_runbook_smoke, "checks.schedule_counterbalances_both_tasks") is True
            and get(formal_rollout_runbook_smoke, "checks.launch_sheet_smoke_pass") is True
            else "incomplete",
            "overall_pass="
            f"{get(formal_rollout_runbook_smoke, 'overall_pass')}; "
            f"checks={get(formal_rollout_runbook_smoke, 'checks')}",
            str(paths["formal_rollout_runbook_smoke"]),
        ),
        item(
            "Formal paired12 collection schedule counterbalances arm order before real rollout collection.",
            "satisfied"
            if formal_collection_schedule is not None
            and get(formal_collection_schedule, "schedule_pass") is True
            and get(formal_collection_schedule, "scientific_evidence") is False
            and get(formal_collection_schedule, "tasks.insertion.counterbalance_pass") is True
            and get(formal_collection_schedule, "tasks.board.counterbalance_pass") is True
            and get(formal_collection_schedule, "tasks.insertion.paired_n_pairs") == 12
            and get(formal_collection_schedule, "tasks.board.paired_n_pairs") == 12
            and len(get(formal_collection_schedule, "long_schedule_rows", []) or []) == 72
            and str(((get(formal_collection_schedule, "long_schedule_rows", []) or [{}])[0]).get("recommended_filename", "")).endswith(".hdf5")
            and str(((get(formal_collection_schedule, "long_schedule_rows", []) or [{}])[0]).get("recommended_path", "")).endswith(".hdf5")
            and "Changing arm order after seeing rollout outcomes"
            in str(get(formal_collection_schedule, "cannot_count_as_completion", ""))
            else "incomplete",
            "schedule_pass="
            f"{get(formal_collection_schedule, 'schedule_pass')}; "
            "position_counts="
            f"{get(formal_collection_schedule, 'tasks.insertion.position_counts')}/"
            f"{get(formal_collection_schedule, 'tasks.board.position_counts')}; "
            f"n_rows={len(get(formal_collection_schedule, 'long_schedule_rows', []) or [])}",
            str(paths["formal_collection_schedule"]),
        ),
        item(
            "Formal paired12 collection progress tracker reports current coverage and next scheduled trial.",
            "satisfied"
            if formal_collection_progress is not None
            and get(formal_collection_progress, "progress_pass") is True
            and get(formal_collection_progress, "scientific_evidence") is False
            and get(formal_collection_progress, "schedule_pass") is True
            and get(formal_collection_progress, "n_scheduled_rows") == 72
            and get(formal_collection_progress, "ready_for_post_collection") is False
            and get(formal_collection_progress, "next_row.task") is not None
            and str(get(formal_collection_progress, "next_row.recommended_filename", "")).endswith(".hdf5")
            and str(get(formal_collection_progress, "next_row.recommended_path", "")).endswith(".hdf5")
            and "Do not change arm order after seeing rollout outcomes"
            in str(get(formal_collection_progress, "guardrails", ""))
            else "incomplete",
            "progress_pass="
            f"{get(formal_collection_progress, 'progress_pass')}; "
            f"completed={get(formal_collection_progress, 'n_completed_rows')}/"
            f"{get(formal_collection_progress, 'n_scheduled_rows')}; "
            f"ready_for_post_collection={get(formal_collection_progress, 'ready_for_post_collection')}; "
            f"next_row={get(formal_collection_progress, 'next_row')}",
            str(paths["formal_collection_progress"]),
        ),
        item(
            "Formal next-collection-step artifact turns the current schedule row into an executable launch/save/verify handoff.",
            "satisfied"
            if formal_next_collection_step is not None
            and get(formal_next_collection_step, "next_step_pass") is True
            and get(formal_next_collection_step, "scientific_evidence") is False
            and get(formal_next_collection_step, "progress_pass") is True
            and get(formal_next_collection_step, "has_next_step") is True
            and get(formal_next_collection_step, "next_row.task")
            == get(formal_collection_progress, "next_row.task")
            and get(formal_next_collection_step, "next_row.arm")
            == get(formal_collection_progress, "next_row.arm")
            and get(formal_next_collection_step, "next_row.pair_id")
            == get(formal_collection_progress, "next_row.pair_id")
            and get(formal_next_collection_step, "recommended_path")
            == get(formal_collection_progress, "next_row.recommended_path")
            and "serve_dp_tac_quality_guided"
            in str(get(formal_next_collection_step, "launch_command", ""))
            and "--dry_run_guidance_smoke" in str(get(formal_next_collection_step, "pre_collection_dry_run_command", ""))
            and "--smoke_output" in str(get(formal_next_collection_step, "pre_collection_dry_run_command", ""))
            and str(get(formal_next_collection_step, "pre_collection_dry_run_output", "")).endswith("_smoke.json")
            and "build_tac_quality_collection_progress.py"
            in str(get(formal_next_collection_step, "post_run_commands", ""))
            and "finalize_tac_quality_collected_hdf5.py"
            in str(get(formal_next_collection_step, "post_run_commands", ""))
            and "finalize_tac_quality_collected_hdf5.py"
            in str(get(formal_next_collection_step, "finalize_command_template", ""))
            else "incomplete",
            "next_step_pass="
            f"{get(formal_next_collection_step, 'next_step_pass')}; "
            f"next_row={get(formal_next_collection_step, 'next_row')}; "
            f"recommended_path={get(formal_next_collection_step, 'recommended_path')}; "
            f"path_exists={get(formal_next_collection_step, 'recommended_path_exists')}; "
            f"hdf5_audit={get(formal_next_collection_step, 'hdf5_audit')}; "
            f"pre_collection_dry_run={get(formal_next_collection_step, 'pre_collection_dry_run_command')}; "
            f"finalize={get(formal_next_collection_step, 'finalize_command_template')}",
            str(paths["formal_next_collection_step"]),
        ),
        item(
            "Current formal next-step pre-collection dry-run smoke has passed before robot collection.",
            "satisfied"
            if formal_next_collection_step_smoke is not None
            and get(formal_next_collection_step_smoke, "overall_pass") is True
            and get(formal_next_collection_step_smoke, "scientific_evidence") is False
            and get(formal_next_collection_step_smoke, "checks.smoke_pass") is True
            and get(formal_next_collection_step_smoke, "checks.task_matches") is True
            and get(formal_next_collection_step_smoke, "checks.arm_matches") is True
            and get(formal_next_collection_step_smoke, "checks.not_reranking") is True
            and get(formal_next_collection_step_smoke, "checks.final_clean_action_guidance") is True
            and get(formal_next_collection_step_smoke, "checks.baseline_guidance_disabled") is True
            and get(formal_next_collection_step_smoke, "checks.guided_grad_contract") is True
            else "incomplete",
            "overall_pass="
            f"{get(formal_next_collection_step_smoke, 'overall_pass')}; "
            f"output={get(formal_next_collection_step_smoke, 'pre_collection_dry_run_output')}; "
            f"checks={get(formal_next_collection_step_smoke, 'checks')}; "
            f"smoke_summary={get(formal_next_collection_step_smoke, 'smoke_summary')}",
            str(paths["formal_next_collection_step_smoke"]),
        ),
        item(
            "Current formal next-step dry-run runner executes the pre-collection smoke command and audits the output.",
            "satisfied"
            if formal_next_collection_step_smoke_runner is not None
            and get(formal_next_collection_step_smoke_runner, "overall_pass") is True
            and get(formal_next_collection_step_smoke_runner, "scientific_evidence") is False
            and get(formal_next_collection_step_smoke_runner, "process.passed_process") is True
            and get(formal_next_collection_step_smoke_runner, "process.returncode") == 0
            and get(formal_next_collection_step_smoke_runner, "smoke_audit.overall_pass") is True
            and "--dry_run_guidance_smoke"
            in str(get(formal_next_collection_step_smoke_runner, "pre_collection_dry_run_command", ""))
            else "incomplete",
            "overall_pass="
            f"{get(formal_next_collection_step_smoke_runner, 'overall_pass')}; "
            f"process={get(formal_next_collection_step_smoke_runner, 'process')}; "
            f"smoke_audit={get(formal_next_collection_step_smoke_runner, 'smoke_audit')}",
            str(paths["formal_next_collection_step_smoke_runner"]),
        ),
        item(
            "Current formal collection row has an operator go/no-go gate before robot execution.",
            "satisfied"
            if formal_current_collection_gate is not None
            and get(formal_current_collection_gate, "current_collection_gate_pass") is True
            and get(formal_current_collection_gate, "scientific_evidence") is False
            and get(formal_current_collection_gate, "operator_go_no_go") == "go"
            and get(formal_current_collection_gate, "checks.runner_pass") is True
            and get(formal_current_collection_gate, "checks.recommended_path_not_exists") is True
            and get(formal_current_collection_gate, "checks.rollout_dir_writable") is True
            and get(formal_current_collection_gate, "checks.same_current_row") is True
            else "incomplete",
            "gate_pass="
            f"{get(formal_current_collection_gate, 'current_collection_gate_pass')}; "
            f"go_no_go={get(formal_current_collection_gate, 'operator_go_no_go')}; "
            f"current_row={get(formal_current_collection_gate, 'current_row')}; "
            f"checks={get(formal_current_collection_gate, 'checks')}",
            str(paths["formal_current_collection_gate"]),
        ),
        item(
            "Current formal collection row has a single operator handoff packet for preflight, launch, finalize, and refresh.",
            "satisfied"
            if formal_current_collection_handoff is not None
            and get(formal_current_collection_handoff, "handoff_pass") is True
            and get(formal_current_collection_handoff, "scientific_evidence") is False
            and get(formal_current_collection_handoff, "operator_go_no_go") == "go"
            and "TACQUALITY_RECOMMENDED_HDF5="
            in str(get(formal_current_collection_handoff, "launch_command_with_save_path_hint", ""))
            and "finalize_tac_quality_collected_hdf5.py"
            in str(get(formal_current_collection_handoff, "finalize_commands", ""))
            and "build_tac_quality_current_collection_gate.py"
            in str(get(formal_current_collection_handoff, "post_finalize_commands", ""))
            else "incomplete",
            "handoff_pass="
            f"{get(formal_current_collection_handoff, 'handoff_pass')}; "
            f"go_no_go={get(formal_current_collection_handoff, 'operator_go_no_go')}; "
            f"current_row={get(formal_current_collection_handoff, 'current_row')}; "
            f"recommended_path={get(formal_current_collection_handoff, 'recommended_path')}; "
            f"checks={get(formal_current_collection_handoff, 'checks')}",
            str(paths["formal_current_collection_handoff"]),
        ),
    ]

    ins_rr = real_rollout_status(rr_ins, "insertion")
    board_rr = real_rollout_status(rr_board, "board")
    requirements.extend(
        [
            item(
                "Formal socket insertion baseline-vs-guided production/robot rollout validation passes.",
                ins_rr["status"],
                ins_rr["evidence"],
                str(paths["real_rollout_insertion"]),
            ),
            item(
                "Formal board wiping baseline-vs-guided production/robot rollout validation passes.",
                board_rr["status"],
                board_rr["evidence"],
                str(paths["real_rollout_board"]),
            ),
            item(
                "Formal three-arm rollout policy/scorer configs plus optional ActionAware fourth-arm candidate are machine-readable and complete.",
                "satisfied"
                if bool(get(rollout_arm_configs, "rollout_arm_config_pass", False))
                and get(rollout_arm_configs, "tasks.insertion.default_guided.scorer_runtime")
                == "InsertionRiskScorerRuntime"
                and get(rollout_arm_configs, "tasks.board.default_guided.scorer_runtime")
                == "PTGProxyScorerV2Runtime"
                and get(rollout_arm_configs, "tasks.insertion.distilled_guided.scorer_runtime")
                == "DistilledTacQualityEnergyRuntime"
                and get(rollout_arm_configs, "tasks.board.distilled_guided.scorer_runtime")
                == "DistilledTacQualityEnergyRuntime"
                and get(rollout_arm_configs, "tasks.insertion.action_aware_guided.scorer_runtime")
                == "ActionAwareScorerRuntime"
                and get(rollout_arm_configs, "tasks.board.action_aware_guided.scorer_runtime")
                == "ActionAwareScorerRuntime"
                else "incomplete",
                "pass="
                f"{get(rollout_arm_configs, 'rollout_arm_config_pass')}; "
                f"insertion_default={get(rollout_arm_configs, 'tasks.insertion.default_guided.scorer_runtime')}; "
                f"board_default={get(rollout_arm_configs, 'tasks.board.default_guided.scorer_runtime')}; "
                f"candidate={get(rollout_arm_configs, 'selection_summary.promoted_ablation_candidate')}; "
                f"action_aware={get(rollout_arm_configs, 'selection_summary.action_aware_marker_status')}",
                str(paths["rollout_arm_configs"]),
            ),
            item(
                "Three-arm scorer ablation evaluator passes synthetic HDF5 smoke for insertion and board.",
                "satisfied"
                if bool(get(ablation_smoke, "overall_pass", False))
                and get(ablation_smoke, "scientific_evidence") is False
                and get(ablation_smoke, "tasks.insertion.production_ablation_pass") is True
                and get(ablation_smoke, "tasks.board.production_ablation_pass") is True
                else "incomplete",
                "overall_pass="
                f"{get(ablation_smoke, 'overall_pass')}; "
                f"scientific_evidence={get(ablation_smoke, 'scientific_evidence')}; "
                f"insertion={get(ablation_smoke, 'tasks.insertion.recommended_real_scorer')}; "
                f"board={get(ablation_smoke, 'tasks.board.recommended_real_scorer')}",
                str(paths["scorer_ablation_smoke"]),
            ),
            item(
                "Every guided rollout arm config can instantiate its scorer and provide finite non-zero gradients.",
                "satisfied"
                if bool(get(rollout_arm_config_smoke, "overall_pass", False))
                and get(rollout_arm_config_smoke, "scientific_evidence") is False
                and get(rollout_arm_config_smoke, "checks.all_guided_arms_pass_gradient_smoke") is True
                and get(rollout_arm_config_smoke, "checks.all_guided_arms_present") is True
                and get(rollout_arm_config_smoke, "checks.optional_action_aware_arms_present") is True
                else "incomplete",
                "overall_pass="
                f"{get(rollout_arm_config_smoke, 'overall_pass')}; "
                f"checks={get(rollout_arm_config_smoke, 'checks')}",
                str(paths["rollout_arm_config_smoke"]),
            ),
            item(
                "Every guided rollout arm can run through Foresight bridge and final-action DP adapter.",
                "satisfied"
                if bool(get(deployment_bridge_smoke, "overall_pass", False))
                and get(deployment_bridge_smoke, "scientific_evidence") is False
                and get(deployment_bridge_smoke, "not_reranking") is True
                and get(deployment_bridge_smoke, "not_every_step_ddpm_guidance") is True
                and get(deployment_bridge_smoke, "checks.all_guided_arms_present") is True
                and get(deployment_bridge_smoke, "checks.optional_action_aware_arms_present") is True
                and get(deployment_bridge_smoke, "checks.all_guided_arms_pass_deployment_bridge_smoke") is True
                and all(
                    (not row.get("guidance_enabled", True))
                    or (
                        get(row, "adapter.called_from_inference_mode") is True
                        and get(row, "adapter.returned_requires_grad") is False
                    )
                    for row in (get(deployment_bridge_smoke, "arms", []) or [])
                )
                else "incomplete",
                "overall_pass="
                f"{get(deployment_bridge_smoke, 'overall_pass')}; "
                f"scientific_evidence={get(deployment_bridge_smoke, 'scientific_evidence')}; "
                f"guidance_mode={get(deployment_bridge_smoke, 'guidance_mode')}; "
                f"checks={get(deployment_bridge_smoke, 'checks')}; "
                "inference_mode_boundary="
                f"{[(row.get('task'), row.get('arm'), get(row, 'adapter.called_from_inference_mode'), get(row, 'adapter.returned_requires_grad')) for row in (get(deployment_bridge_smoke, 'arms', []) or []) if row.get('guidance_enabled', True)]}",
                str(paths["deployment_bridge_smoke"]),
            ),
            item(
                "Serving integration packet auto-discovers strict DP/Foresight preflight inputs for both tasks.",
                "satisfied"
                if bool(get(serving_packet, "auto_discover", False))
                and get(serving_packet, "serving_ready") is True
                and get(serving_packet, "checks.auto_insertion_ready") is True
                and get(serving_packet, "checks.auto_board_ready") is True
                and get(serving_packet, "packets.insertion.checks.dp_action_dim_matches_foresight_action_dim") is True
                and get(serving_packet, "packets.board.checks.dp_action_dim_matches_foresight_action_dim") is True
                else "incomplete",
                "serving_ready="
                f"{get(serving_packet, 'serving_ready')}; "
                f"auto_discover={get(serving_packet, 'auto_discover')}; "
                f"checks={get(serving_packet, 'checks')}; "
                f"auto_pairs={get(serving_packet, 'auto_pairs')}; "
                f"next_step={get(serving_packet, 'next_step')}",
                str(paths["serving_packet"]),
            ),
            item(
                "Guided server entrypoint exists, passes real-Foresight dry-run smoke, and separates final-action guidance from reranking servers.",
                "satisfied"
                if bool(get(guided_server_packet, "launch_packet_ready", False))
                and get(guided_server_packet, "serving_packet_ready") is True
                and get(guided_server_packet, "guided_server_ready") is True
                and get(guided_server_packet, "not_reranking") is True
                and get(guided_server_packet, "not_every_step_ddpm_guidance") is True
                and "action_aware_guided" in str(
                    get(guided_server_packet, "tasks.insertion.action_aware_guided_command_template", "")
                )
                and "action_aware_guided" in str(
                    get(guided_server_packet, "tasks.board.action_aware_guided_command_template", "")
                )
                and get(load_json(paths["guided_server_insertion_real_foresight_smoke"]), "dry_run_guidance_smoke_pass") is True
                and get(load_json(paths["guided_server_board_real_foresight_smoke"]), "dry_run_guidance_smoke_pass") is True
                else "incomplete",
                "launch_packet_ready="
                f"{get(guided_server_packet, 'launch_packet_ready')}; "
                f"guided_server_ready={get(guided_server_packet, 'guided_server_ready')}; "
                f"next_step={get(guided_server_packet, 'next_step')}; "
                "action_aware_commands="
                f"{'action_aware_guided' in str(get(guided_server_packet, 'tasks.insertion.action_aware_guided_command_template', ''))}/"
                f"{'action_aware_guided' in str(get(guided_server_packet, 'tasks.board.action_aware_guided_command_template', ''))}; "
                "real_foresight_smoke="
                f"{get(load_json(paths['guided_server_insertion_real_foresight_smoke']), 'dry_run_guidance_smoke_pass')}/"
                f"{get(load_json(paths['guided_server_board_real_foresight_smoke']), 'dry_run_guidance_smoke_pass')}",
                str(paths["guided_server_packet"]),
            ),
            item(
                "Optional ActionAware rollout gate runner exists without becoming a formal completion dependency.",
                "satisfied"
                if optional_action_aware_runner is not None
                and get(optional_action_aware_runner, "scientific_evidence") is False
                and get(optional_action_aware_runner, "formal_gate_dependency") is False
                and get(optional_action_aware_runner, "run_gates_requested") is False
                and get(optional_action_aware_runner, "tasks.insertion.checks.action_aware_guided.n_hdf5") is not None
                and get(optional_action_aware_runner, "tasks.board.checks.action_aware_guided.n_hdf5") is not None
                else "incomplete",
                "preflight_ready="
                f"{get(optional_action_aware_runner, 'preflight_ready')}; "
                f"formal_gate_dependency={get(optional_action_aware_runner, 'formal_gate_dependency')}; "
                f"scientific_evidence={get(optional_action_aware_runner, 'scientific_evidence')}; "
                f"interpretation={get(optional_action_aware_runner, 'interpretation')}",
                str(paths["optional_action_aware_rollout_gate_runner"]),
            ),
            item(
                "All four guided server arms pass real-Foresight dry-run smoke before formal three-arm rollout ablation.",
                "satisfied"
                if bool(get(guided_server_all_arms_smoke, "overall_pass", False))
                and get(guided_server_all_arms_smoke, "scientific_evidence") is False
                and get(guided_server_all_arms_smoke, "not_reranking") is True
                and get(guided_server_all_arms_smoke, "not_every_step_ddpm_guidance") is True
                and get(guided_server_all_arms_smoke, "checks.all_four_guided_arms_present") is True
                and get(guided_server_all_arms_smoke, "checks.all_guided_arms_pass_real_foresight_smoke") is True
                else "incomplete",
                "overall_pass="
                f"{get(guided_server_all_arms_smoke, 'overall_pass')}; "
                f"checks={get(guided_server_all_arms_smoke, 'checks')}; "
                "arms="
                f"{[(row.get('task'), row.get('arm'), row.get('passes_all_arm_real_foresight_smoke'), get(row, 'report.improved_rate')) for row in (get(guided_server_all_arms_smoke, 'arms', []) or [])]}",
                str(paths["guided_server_all_arms_real_foresight_smoke"]),
            ),
            item(
                "Formal insertion three-arm scorer ablation identifies the best real guided scorer.",
                "satisfied"
                if get(ablation_ins, "production_ablation_pass") is True
                and get(ablation_ins, "debug_or_underpowered") is False
                and ablation_is_real(ablation_ins)
                and get(ablation_ins, "recommended_real_scorer") is not None
                else ("missing" if ablation_ins is None else "incomplete"),
                "production_ablation_pass="
                f"{get(ablation_ins, 'production_ablation_pass')}; "
                f"debug_or_underpowered={get(ablation_ins, 'debug_or_underpowered')}; "
                f"real_paths={ablation_is_real(ablation_ins)}; "
                f"recommended_real_scorer={get(ablation_ins, 'recommended_real_scorer')}",
                str(paths["scorer_ablation_insertion"]),
            ),
            item(
                "Formal board three-arm scorer ablation identifies the best real guided scorer.",
                "satisfied"
                if get(ablation_board, "production_ablation_pass") is True
                and get(ablation_board, "debug_or_underpowered") is False
                and ablation_is_real(ablation_board)
                and get(ablation_board, "recommended_real_scorer") is not None
                else ("missing" if ablation_board is None else "incomplete"),
                "production_ablation_pass="
                f"{get(ablation_board, 'production_ablation_pass')}; "
                f"debug_or_underpowered={get(ablation_board, 'debug_or_underpowered')}; "
                f"real_paths={ablation_is_real(ablation_board)}; "
                f"recommended_real_scorer={get(ablation_board, 'recommended_real_scorer')}",
                str(paths["scorer_ablation_board"]),
            ),
        ]
    )

    objective_complete = all(r["status"] == "satisfied" for r in requirements)
    blockers = [r for r in requirements if r["status"] != "satisfied"]
    result = {
        "objective": (
            "Design, evaluate, and record a tactile quality classifier/scorer for DP classifier guidance "
            "on socket insertion and board wiping, balancing effectiveness and novelty."
        ),
        "git_commit": git_commit(),
        "objective_complete": bool(objective_complete),
        "status": "complete" if objective_complete else "incomplete",
        "requirements": requirements,
        "blockers": blockers,
        "next_required_step": (
            "Collect formal baseline-vs-guided production/robot rollouts for insertion and board, "
            "then run TFAC_V5/eval_real_rollout_quality_gate.py and "
            "TFAC_V5/eval_real_rollout_scorer_ablation_gate.py with metadata/pairing if available."
            if blockers
            else None
        ),
        "paths": {name: str(path) for name, path in paths.items()},
    }
    return result


def write_markdown(result: Dict[str, Any], path: Path) -> None:
    lines = [
        "# TacQuality Goal Completion Audit",
        "",
        f"- objective_complete: `{result['objective_complete']}`",
        f"- status: `{result['status']}`",
        f"- git_commit: `{result['git_commit']}`",
        f"- next_required_step: {result['next_required_step']}",
        "",
        "## Requirements",
        "",
        "| status | requirement | evidence |",
        "|---|---|---|",
    ]
    for row in result["requirements"]:
        lines.append(
            f"| {row['status']} | {row['requirement']} | {str(row['evidence']).replace('|', '/')} |"
        )
    lines.extend(["", "## Blockers", ""])
    for row in result["blockers"]:
        lines.append(f"- **{row['status']}** {row['requirement']}: {row['evidence']}")
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
    result = build_audit(PATHS)
    json_path = out_dir / "tac_quality_goal_completion_audit.json"
    md_path = out_dir / "tac_quality_goal_completion_audit.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    print(
        json.dumps(
            {
                "objective_complete": result["objective_complete"],
                "status": result["status"],
                "n_requirements": len(result["requirements"]),
                "n_blockers": len(result["blockers"]),
                "next_required_step": result["next_required_step"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"Saved {json_path}")
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
