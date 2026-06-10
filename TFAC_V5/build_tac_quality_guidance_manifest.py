"""Build a deployment manifest for TacQuality DP classifier guidance.

The manifest is a machine-readable handoff artifact for connecting the current
scorer/guidance stack to a DP policy or robot dry-run.  It records:

  - selected task-conditioned scorers and checkpoints;
  - runtime/refiner/config modules;
  - evidence JSON files and their pass/fail status;
  - the exact score API and remaining validation gap.

It intentionally does not claim real-robot completion.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional


OUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_guidance_manifest")

FUTURE_ROLLOUT_OUTPUTS = {
    "scorer_ablation_gate_insertion",
    "scorer_ablation_gate_board",
}


PATHS = {
    "insertion_scorer_ckpt": Path("/home/chenshuai/Project/output/insertion_risk_scorer/insertion_risk_scorer_final.pt"),
    "board_scorer_ckpt": Path("/home/chenshuai/Project/output/ptg_proxy_scorer_v2/ptg_proxy_scorer_v2_final.pt"),
    "distilled_energy_ckpt": Path(
        "/home/chenshuai/Project/output/distilled_tac_quality_energy/distilled_tac_quality_energy_final.pt"
    ),
    "action_aware_ckpt": Path(
        "/home/chenshuai/Project/output/action_aware_marker_scorer/action_aware_marker_scorer_final.pt"
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
    "scorer_selection_gate": Path(
        "/home/chenshuai/Project/output/tac_quality_scorer_selection_gate/tac_quality_scorer_selection_gate.json"
    ),
    "scorer_decision_matrix": Path(
        "/home/chenshuai/Project/output/tac_quality_scorer_decision_matrix/"
        "tac_quality_scorer_decision_matrix.json"
    ),
    "runtime_contract": Path("/home/chenshuai/Project/output/tac_quality_guidance_runtime/runtime_contract_sanity.json"),
    "guidance_contract": Path(
        "/home/chenshuai/Project/output/tac_quality_guidance_contract/"
        "tac_quality_guidance_contract_audit.json"
    ),
    "trust_region_guidance": Path("/home/chenshuai/Project/output/tac_quality_trust_region_guidance/trust_region_sanity.json"),
    "dp_guidance_controller": Path("/home/chenshuai/Project/output/tac_quality_dp_guidance_controller/controller_sanity.json"),
    "dp_guidance_controller_real_sample": Path("/home/chenshuai/Project/output/tac_quality_dp_guidance_controller/controller_real_sample_audit.json"),
    "dp_integration_adapter": Path(
        "/home/chenshuai/Project/output/tac_quality_dp_integration_adapter/integration_adapter_sanity.json"
    ),
    "foresight_bridge": Path(
        "/home/chenshuai/Project/output/tac_quality_foresight_bridge/foresight_bridge_sanity.json"
    ),
    "score_calibration": Path("/home/chenshuai/Project/output/tac_quality_score_calibration/tac_quality_score_calibration.json"),
    "score_landscape": Path("/home/chenshuai/Project/output/tac_quality_score_landscape/tac_quality_score_landscape.json"),
    "proxy_alignment_audit": Path(
        "/home/chenshuai/Project/output/tac_quality_proxy_alignment_audit/"
        "tac_quality_proxy_alignment_audit.json"
    ),
    "runtime_visualization": Path(
        "/home/chenshuai/Project/output/tac_quality_runtime_visualization/tac_quality_runtime_visualization.json"
    ),
    "evidence_summary": Path("/home/chenshuai/Project/output/ptg_guidance_evidence/ptg_guidance_evidence_summary.json"),
    "offline_gate": Path("/home/chenshuai/Project/output/ptg_offline_production_gate/ptg_offline_production_gate.json"),
    "insertion_full_chain": Path("/home/chenshuai/Project/output/full_chain_guidance_gradient/insertion_full_chain_energy_clipped_K8_N16.json"),
    "insertion_clean_refine": Path("/home/chenshuai/Project/output/clean_action_energy_refinement/insertion_clean_refine_constrained_K4_N40.json"),
    "insertion_distilled_clean_refine": Path(
        "/home/chenshuai/Project/output/insertion_distilled_clean_refine_comparison/"
        "n24_k4/insertion_distilled_clean_refine_comparison.json"
    ),
    "board_full_chain_fast100": Path("/home/chenshuai/Project/output/board_dp_denoising_full_chain_smoke/board_dp_feature_cache_full80_fast32ema_w4096_e5_fast100_heldout32_K4_N64.json"),
    "board_distilled_clean_refine": Path(
        "/home/chenshuai/Project/output/board_dp_distilled_clean_refine_comparison/"
        "fast20_heldout32_n64/board_dp_distilled_clean_refine_comparison.json"
    ),
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
    "scorer_ablation_gate_insertion": Path(
        "/home/chenshuai/Project/output/real_rollout_scorer_ablation_gate/"
        "insertion_baseline_vs_default_vs_distilled/real_rollout_scorer_ablation_gate.json"
    ),
    "scorer_ablation_gate_board": Path(
        "/home/chenshuai/Project/output/real_rollout_scorer_ablation_gate/"
        "board_baseline_vs_default_vs_distilled/real_rollout_scorer_ablation_gate.json"
    ),
    "scorer_ablation_gate_smoke": Path(
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
    "metadata_review_sheet_smoke": Path(
        "/home/chenshuai/Project/output/tac_quality_metadata_review_sheet_smoke/"
        "synthetic/tac_quality_metadata_review_sheet_smoke.json"
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
    "outcome_label_card": Path(
        "/home/chenshuai/Project/output/tac_quality_outcome_label_card/"
        "tac_quality_outcome_label_card.json"
    ),
    "outcome_label_card_smoke": Path(
        "/home/chenshuai/Project/output/tac_quality_outcome_label_card_smoke/"
        "synthetic/tac_quality_outcome_label_card_smoke.json"
    ),
    "scorer_freeze_manifest": Path(
        "/home/chenshuai/Project/output/tac_quality_scorer_freeze_manifest/"
        "tac_quality_scorer_freeze_manifest.json"
    ),
    "scorer_freeze_manifest_smoke": Path(
        "/home/chenshuai/Project/output/tac_quality_scorer_freeze_manifest_smoke/"
        "synthetic/tac_quality_scorer_freeze_manifest_smoke.json"
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
    "goal_completion_audit": Path("/home/chenshuai/Project/output/tac_quality_goal_audit/tac_quality_goal_completion_audit.json"),
}


MODULES = {
    "guidance_config": Path("TFAC_V5/tac_quality_guidance_config.py"),
    "guidance_runtime": Path("TFAC_V5/tac_quality_guidance_runtime.py"),
    "action_aware_runtime": Path("TFAC_V5/action_aware_scorer_runtime.py"),
    "action_aware_guidance_suitability": Path("TFAC_V5/eval_action_aware_guidance_suitability.py"),
    "trust_region_refiner": Path("TFAC_V5/tac_quality_trust_region_guidance.py"),
    "dp_guidance_controller": Path("TFAC_V5/tac_quality_dp_guidance_controller.py"),
    "dp_integration_adapter": Path("TFAC_V5/tac_quality_dp_integration_adapter.py"),
    "serving_guidance": Path("TFAC_V5/tac_quality_serving_guidance.py"),
    "serving_packet": Path("TFAC_V5/build_tac_quality_serving_packet.py"),
    "guided_server_packet": Path("TFAC_V5/build_tac_quality_guided_server_packet.py"),
    "guided_server_real_foresight_smoke": Path("TFAC_V5/smoke_tac_quality_guided_server_real_foresight.py"),
    "foresight_bridge": Path("TFAC_V5/tac_quality_foresight_bridge.py"),
    "real_rollout_quality_gate": Path("TFAC_V5/eval_real_rollout_quality_gate.py"),
    "real_rollout_scorer_ablation_gate": Path("TFAC_V5/eval_real_rollout_scorer_ablation_gate.py"),
    "real_rollout_scorer_ablation_smoke": Path("TFAC_V5/smoke_real_rollout_scorer_ablation_gate.py"),
    "real_rollout_validation_prep": Path("TFAC_V5/prepare_real_rollout_validation.py"),
    "real_rollout_sample_size_plan": Path("TFAC_V5/plan_real_rollout_sample_size.py"),
    "real_rollout_experiment_packet": Path("TFAC_V5/build_real_rollout_experiment_packet.py"),
    "formal_rollout_gate_runner": Path("TFAC_V5/run_formal_tac_quality_rollout_gates.py"),
    "optional_action_aware_rollout_gate_runner": Path("TFAC_V5/run_optional_action_aware_rollout_gate.py"),
    "formal_launch_sheet": Path("TFAC_V5/build_tac_quality_formal_launch_sheet.py"),
    "formal_launch_sheet_smoke": Path("TFAC_V5/smoke_tac_quality_formal_launch_sheet.py"),
    "formal_collection_readiness": Path("TFAC_V5/build_tac_quality_collection_readiness.py"),
    "formal_rollout_pairing": Path("TFAC_V5/build_tac_quality_rollout_pairing.py"),
    "pairing_metadata_audit": Path("TFAC_V5/audit_tac_quality_pairing_metadata.py"),
    "hdf5_schema_audit": Path("TFAC_V5/audit_tac_quality_rollout_hdf5_schema.py"),
    "generated_pairing_gate_runner_smoke": Path("TFAC_V5/smoke_tac_quality_generated_pairing_gate_runner.py"),
    "real_rollout_source_audit": Path("TFAC_V5/audit_tac_quality_real_rollout_sources.py"),
    "post_collection_pipeline": Path("TFAC_V5/run_tac_quality_post_collection_pipeline.py"),
    "post_collection_pipeline_smoke": Path("TFAC_V5/smoke_tac_quality_post_collection_pipeline.py"),
    "metadata_review_sheet": Path("TFAC_V5/build_tac_quality_metadata_review_sheet.py"),
    "metadata_review_sheet_smoke": Path("TFAC_V5/smoke_tac_quality_metadata_review_sheet.py"),
    "finalize_collected_hdf5_smoke": Path("TFAC_V5/smoke_tac_quality_finalize_collected_hdf5.py"),
    "finalize_and_refresh": Path("TFAC_V5/finalize_and_refresh_tac_quality_collection.py"),
    "finalize_and_refresh_smoke": Path("TFAC_V5/smoke_finalize_and_refresh_tac_quality_collection.py"),
    "real_rollout_acceptance_protocol": Path("TFAC_V5/build_tac_quality_real_rollout_acceptance_protocol.py"),
    "outcome_label_card": Path("TFAC_V5/build_tac_quality_outcome_label_card.py"),
    "outcome_label_card_smoke": Path("TFAC_V5/smoke_tac_quality_outcome_label_card.py"),
    "scorer_freeze_manifest": Path("TFAC_V5/build_tac_quality_scorer_freeze_manifest.py"),
    "scorer_freeze_manifest_smoke": Path("TFAC_V5/smoke_tac_quality_scorer_freeze_manifest.py"),
    "formal_rollout_runbook": Path("TFAC_V5/build_tac_quality_formal_rollout_runbook.py"),
    "formal_rollout_runbook_smoke": Path("TFAC_V5/smoke_tac_quality_formal_rollout_runbook.py"),
    "formal_collection_schedule": Path("TFAC_V5/build_tac_quality_collection_schedule.py"),
    "formal_collection_progress": Path("TFAC_V5/build_tac_quality_collection_progress.py"),
    "formal_next_collection_step": Path("TFAC_V5/build_tac_quality_next_collection_step.py"),
    "formal_next_collection_step_smoke": Path("TFAC_V5/smoke_tac_quality_next_collection_step.py"),
    "formal_next_collection_step_smoke_runner": Path("TFAC_V5/run_tac_quality_next_collection_step_smoke.py"),
    "formal_current_collection_gate": Path("TFAC_V5/build_tac_quality_current_collection_gate.py"),
    "formal_current_collection_handoff": Path("TFAC_V5/build_tac_quality_current_collection_handoff.py"),
    "finalize_collected_hdf5": Path("TFAC_V5/finalize_tac_quality_collected_hdf5.py"),
    "board_target_force_calibration": Path("TFAC_V5/calibrate_board_target_force.py"),
    "label_standard_registry": Path("TFAC_V5/build_tac_quality_label_standard_registry.py"),
    "label_standard_compliance": Path("TFAC_V5/audit_tac_quality_label_standard_compliance.py"),
    "rollout_arm_configs": Path("TFAC_V5/build_tac_quality_rollout_arm_configs.py"),
    "rollout_arm_config_smoke": Path("TFAC_V5/smoke_tac_quality_rollout_arm_configs.py"),
    "deployment_bridge_smoke": Path("TFAC_V5/smoke_tac_quality_deployment_bridge.py"),
    "scorer_selection_gate": Path("TFAC_V5/build_tac_quality_scorer_selection_gate.py"),
    "scorer_decision_matrix": Path("TFAC_V5/build_tac_quality_scorer_decision_matrix.py"),
    "score_landscape": Path("TFAC_V5/eval_tac_quality_score_landscape.py"),
    "guidance_contract_audit": Path("TFAC_V5/audit_tac_quality_guidance_contract.py"),
    "proxy_alignment_audit": Path("TFAC_V5/audit_tac_quality_proxy_alignment.py"),
    "runtime_visualization": Path("TFAC_V5/visualize_tac_quality_runtime.py"),
    "summary_builder": Path("TFAC_V5/summarize_ptg_guidance_evidence.py"),
    "goal_completion_audit": Path("TFAC_V5/audit_tac_quality_goal_completion.py"),
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


def build_manifest() -> Dict[str, Any]:
    data = {name: load_json(path) for name, path in PATHS.items() if path.suffix == ".json"}
    missing = {
        name: str(path)
        for name, path in {**PATHS, **MODULES}.items()
        if not path.exists() and name not in FUTURE_ROLLOUT_OUTPUTS
    }

    checks = [
        {
            "name": "rollout_arm_configs_pass",
            "passed": bool(get(data["rollout_arm_configs"], "rollout_arm_config_pass", False))
            and get(data["rollout_arm_configs"], "tasks.insertion.default_guided.scorer_runtime")
            == "InsertionRiskScorerRuntime"
            and get(data["rollout_arm_configs"], "tasks.board.default_guided.scorer_runtime")
            == "PTGProxyScorerV2Runtime"
            and get(data["rollout_arm_configs"], "tasks.board.distilled_guided.scorer_runtime")
            == "DistilledTacQualityEnergyRuntime"
            and get(data["rollout_arm_configs"], "tasks.insertion.action_aware_guided.scorer_runtime")
            == "ActionAwareScorerRuntime"
            and get(data["rollout_arm_configs"], "tasks.board.action_aware_guided.scorer_runtime")
            == "ActionAwareScorerRuntime",
            "evidence": {
                "pass": get(data["rollout_arm_configs"], "rollout_arm_config_pass"),
                "selection_summary": get(data["rollout_arm_configs"], "selection_summary"),
            },
        },
        {
            "name": "rollout_arm_config_gradient_smoke_pass",
            "passed": bool(get(data["rollout_arm_config_smoke"], "overall_pass", False))
            and get(data["rollout_arm_config_smoke"], "scientific_evidence") is False
            and get(data["rollout_arm_config_smoke"], "checks.optional_action_aware_arms_present") is True
            and get(data["rollout_arm_config_smoke"], "checks.all_guided_arms_pass_gradient_smoke") is True,
            "evidence": {
                "overall_pass": get(data["rollout_arm_config_smoke"], "overall_pass"),
                "scientific_evidence": get(data["rollout_arm_config_smoke"], "scientific_evidence"),
                "checks": get(data["rollout_arm_config_smoke"], "checks"),
            },
        },
        {
            "name": "deployment_bridge_smoke_pass",
            "passed": bool(get(data["deployment_bridge_smoke"], "overall_pass", False))
            and get(data["deployment_bridge_smoke"], "scientific_evidence") is False
            and get(data["deployment_bridge_smoke"], "not_reranking") is True
            and get(data["deployment_bridge_smoke"], "not_every_step_ddpm_guidance") is True
            and get(data["deployment_bridge_smoke"], "checks.all_guided_arms_present") is True
            and get(data["deployment_bridge_smoke"], "checks.optional_action_aware_arms_present") is True
            and get(data["deployment_bridge_smoke"], "checks.all_guided_arms_pass_deployment_bridge_smoke") is True
            and all(
                (not row.get("guidance_enabled", True))
                or (
                    get(row, "adapter.called_from_inference_mode") is True
                    and get(row, "adapter.returned_requires_grad") is False
                )
                for row in (get(data["deployment_bridge_smoke"], "arms", []) or [])
            ),
            "evidence": {
                "overall_pass": get(data["deployment_bridge_smoke"], "overall_pass"),
                "scientific_evidence": get(data["deployment_bridge_smoke"], "scientific_evidence"),
                "guidance_mode": get(data["deployment_bridge_smoke"], "guidance_mode"),
                "checks": get(data["deployment_bridge_smoke"], "checks"),
                "inference_mode_boundary_checked": [
                    {
                        "task": row.get("task"),
                        "arm": row.get("arm"),
                        "called_from_inference_mode": get(row, "adapter.called_from_inference_mode"),
                        "returned_requires_grad": get(row, "adapter.returned_requires_grad"),
                    }
                    for row in (get(data["deployment_bridge_smoke"], "arms", []) or [])
                    if row.get("guidance_enabled", True)
                ],
            },
        },
        {
            "name": "serving_packet_auto_discovered_ready",
            "passed": bool(get(data["serving_packet"], "auto_discover", False))
            and get(data["serving_packet"], "serving_ready") is True
            and get(data["serving_packet"], "checks.auto_insertion_ready") is True
            and get(data["serving_packet"], "checks.auto_board_ready") is True
            and get(data["serving_packet"], "packets.insertion.checks.dp_action_dim_matches_foresight_action_dim") is True
            and get(data["serving_packet"], "packets.board.checks.dp_action_dim_matches_foresight_action_dim") is True,
            "evidence": {
                "serving_ready": get(data["serving_packet"], "serving_ready"),
                "auto_discover": get(data["serving_packet"], "auto_discover"),
                "checks": get(data["serving_packet"], "checks"),
                "auto_pairs": get(data["serving_packet"], "auto_pairs"),
                "next_step": get(data["serving_packet"], "next_step"),
            },
        },
        {
            "name": "guided_server_launch_packet_exists",
            "passed": bool(get(data["guided_server_packet"], "launch_packet_ready", False))
            and get(data["guided_server_packet"], "serving_packet_ready") is True
            and get(data["guided_server_packet"], "guided_server_ready") is True
            and get(data["guided_server_packet"], "not_reranking") is True
            and get(data["guided_server_packet"], "not_every_step_ddpm_guidance") is True
            and "action_aware_guided" in str(
                get(data["guided_server_packet"], "tasks.insertion.action_aware_guided_command_template", "")
            )
            and "action_aware_guided" in str(
                get(data["guided_server_packet"], "tasks.board.action_aware_guided_command_template", "")
            )
            and get(data["guided_server_insertion_real_foresight_smoke"], "dry_run_guidance_smoke_pass") is True
            and get(data["guided_server_board_real_foresight_smoke"], "dry_run_guidance_smoke_pass") is True,
            "evidence": {
                "launch_packet_ready": get(data["guided_server_packet"], "launch_packet_ready"),
                "guided_server_ready": get(data["guided_server_packet"], "guided_server_ready"),
                "next_step": get(data["guided_server_packet"], "next_step"),
                "tasks": get(data["guided_server_packet"], "tasks"),
                "action_aware_commands_present": {
                    "insertion": "action_aware_guided"
                    in str(get(data["guided_server_packet"], "tasks.insertion.action_aware_guided_command_template", "")),
                    "board": "action_aware_guided"
                    in str(get(data["guided_server_packet"], "tasks.board.action_aware_guided_command_template", "")),
                },
                "insertion_real_foresight_smoke": get(data["guided_server_insertion_real_foresight_smoke"], "dry_run_guidance_smoke_pass"),
                "board_real_foresight_smoke": get(data["guided_server_board_real_foresight_smoke"], "dry_run_guidance_smoke_pass"),
            },
        },
        {
            "name": "guided_server_all_arms_real_foresight_smoke_pass",
            "passed": bool(get(data["guided_server_all_arms_real_foresight_smoke"], "overall_pass", False))
            and get(data["guided_server_all_arms_real_foresight_smoke"], "scientific_evidence") is False
            and get(data["guided_server_all_arms_real_foresight_smoke"], "not_reranking") is True
            and get(data["guided_server_all_arms_real_foresight_smoke"], "not_every_step_ddpm_guidance") is True
            and get(data["guided_server_all_arms_real_foresight_smoke"], "checks.all_four_guided_arms_present") is True
            and get(data["guided_server_all_arms_real_foresight_smoke"], "checks.all_guided_arms_pass_real_foresight_smoke") is True,
            "evidence": {
                "overall_pass": get(data["guided_server_all_arms_real_foresight_smoke"], "overall_pass"),
                "checks": get(data["guided_server_all_arms_real_foresight_smoke"], "checks"),
                "arms": [
                    {
                        "task": row.get("task"),
                        "arm": row.get("arm"),
                        "pass": row.get("passes_all_arm_real_foresight_smoke"),
                        "improved_rate": get(row, "report.improved_rate"),
                        "finite_grad_rate": get(row, "report.finite_grad_rate"),
                        "positive_grad_rate": get(row, "report.positive_grad_rate"),
                    }
                    for row in (get(data["guided_server_all_arms_real_foresight_smoke"], "arms", []) or [])
                ],
            },
        },
        {
            "name": "board_target_force_calibration_exists",
            "passed": get(data["board_target_force_calibration"], "recommended.board_target_force") is not None
            and get(data["board_target_force_calibration"], "recommended.board_force_sigma") is not None
            and (get(data["board_target_force_calibration"], "n_episodes", 0) or 0) >= 20,
            "evidence": {
                "n_episodes": get(data["board_target_force_calibration"], "n_episodes"),
                "target": get(data["board_target_force_calibration"], "recommended.board_target_force"),
                "sigma": get(data["board_target_force_calibration"], "recommended.board_force_sigma"),
            },
        },
        {
            "name": "label_standard_registry_pass",
            "passed": bool(get(data["label_standard_registry"], "registry_pass", False))
            and get(data["label_standard_registry"], "scientific_evidence") is False
            and get(data["label_standard_registry"], "tasks.insertion.binary_policy.bad") == [
                "pre_bounce_risk",
                "impact_or_recovery",
            ]
            and get(data["label_standard_registry"], "tasks.board.binary_policy.good") == ["good_smooth"]
            and get(data["label_standard_registry"], "tasks.board.target_force") is not None
            and "formal insertion baseline-vs-guided real rollout gate"
            in str(get(data["label_standard_registry"], "shared_guidance_policy.required_for_final_completion", "")),
            "evidence": {
                "registry_pass": get(data["label_standard_registry"], "registry_pass"),
                "insertion_binary_policy": get(data["label_standard_registry"], "tasks.insertion.binary_policy"),
                "board_binary_policy": get(data["label_standard_registry"], "tasks.board.binary_policy"),
                "board_target_force": get(data["label_standard_registry"], "tasks.board.target_force"),
                "required_for_final_completion": get(
                    data["label_standard_registry"], "shared_guidance_policy.required_for_final_completion"
                ),
            },
        },
        {
            "name": "label_standard_compliance_pass",
            "passed": bool(get(data["label_standard_compliance"], "compliance_pass", False))
            and get(data["label_standard_compliance"], "scientific_evidence") is False
            and get(data["label_standard_compliance"], "required_checks_pass") is True
            and get(data["label_standard_compliance"], "compatible_deviations_pass") is True
            and (get(data["label_standard_compliance"], "summary.n_checks", 0) or 0) >= 10,
            "evidence": {
                "compliance_pass": get(data["label_standard_compliance"], "compliance_pass"),
                "summary": get(data["label_standard_compliance"], "summary"),
                "checks": get(data["label_standard_compliance"], "checks"),
                "interpretation": get(data["label_standard_compliance"], "interpretation"),
            },
        },
        {
            "name": "scorer_selection_gate_pass",
            "passed": bool(get(data["scorer_selection_gate"], "selection_gate_pass", False))
            and get(data["scorer_selection_gate"], "selection.current_default_board_scorer")
            == "PTGProxyScorerV2Runtime"
            and get(data["scorer_selection_gate"], "selection.promoted_ablation_candidate")
            == "DistilledTacQualityEnergyRuntime"
            and get(data["scorer_selection_gate"], "selection.distilled_replacement_status") == "not_yet_replacement"
            and get(data["scorer_selection_gate"], "selection.action_aware_marker_status")
            == "line_search_quality_mode_guidance_candidate"
            and get(data["scorer_selection_gate"], "evidence.gradient_guidance_contract.scale_sweep.overall_pass")
            is True
            and get(data["scorer_selection_gate"], "evidence.gradient_guidance_contract.robustness.overall_pass")
            is True
            and (
                get(
                    data["scorer_selection_gate"],
                    "evidence.gradient_guidance_contract.scale_sweep.insertion_improved_rate",
                    0.0,
                )
                or 0.0
            )
            >= 0.95
            and (
                get(
                    data["scorer_selection_gate"],
                    "evidence.gradient_guidance_contract.scale_sweep.board_improved_rate",
                    0.0,
                )
                or 0.0
            )
            >= 0.95,
            "evidence": {
                "selection_gate_pass": get(data["scorer_selection_gate"], "selection_gate_pass"),
                "status": get(data["scorer_selection_gate"], "status"),
                "current_default_board_scorer": get(
                    data["scorer_selection_gate"], "selection.current_default_board_scorer"
                ),
                "promoted_ablation_candidate": get(
                    data["scorer_selection_gate"], "selection.promoted_ablation_candidate"
                ),
                "distilled_replacement_status": get(
                    data["scorer_selection_gate"], "selection.distilled_replacement_status"
                ),
                "action_aware_marker_status": get(
                    data["scorer_selection_gate"], "selection.action_aware_marker_status"
                ),
                "gradient_guidance_contract": get(
                    data["scorer_selection_gate"], "evidence.gradient_guidance_contract"
                ),
            },
        },
        {
            "name": "proxy_alignment_audit_pass",
            "passed": bool(get(data["proxy_alignment_audit"], "proxy_alignment_pass", False))
            and get(data["proxy_alignment_audit"], "scientific_evidence") is False
            and get(data["proxy_alignment_audit"], "task_pass.insertion") is True
            and get(data["proxy_alignment_audit"], "task_pass.board") is True
            and "Offline proxy alignment" in str(get(data["proxy_alignment_audit"], "remaining_gap", "")),
            "evidence": {
                "proxy_alignment_pass": get(data["proxy_alignment_audit"], "proxy_alignment_pass"),
                "task_pass": get(data["proxy_alignment_audit"], "task_pass"),
                "rows": get(data["proxy_alignment_audit"], "rows"),
                "remaining_gap": get(data["proxy_alignment_audit"], "remaining_gap"),
            },
        },
        {
            "name": "scorer_decision_matrix_pass",
            "passed": bool(get(data["scorer_decision_matrix"], "passes_decision_matrix_gate", False))
            and get(data["scorer_decision_matrix"], "scientific_evidence") is False
            and get(data["scorer_decision_matrix"], "recommendation.formal_default_insertion")
            == "InsertionRiskScorerRuntime"
            and get(data["scorer_decision_matrix"], "recommendation.formal_default_board")
            == "PTGProxyScorerV2Runtime"
            and get(data["scorer_decision_matrix"], "recommendation.innovation_ablation")
            == "DistilledTacQualityEnergyRuntime"
            and get(data["scorer_decision_matrix"], "recommendation.optional_unified_action_conditioned_ablation")
            == "ActionAwareScorerRuntime"
            and "Real baseline-vs-guided" in str(get(data["scorer_decision_matrix"], "remaining_gap", "")),
            "evidence": {
                "passes": get(data["scorer_decision_matrix"], "passes_decision_matrix_gate"),
                "ranked_offline": get(data["scorer_decision_matrix"], "ranked_offline"),
                "recommendation": get(data["scorer_decision_matrix"], "recommendation"),
                "remaining_gap": get(data["scorer_decision_matrix"], "remaining_gap"),
            },
        },
        {
            "name": "action_aware_runtime_candidate_tracked",
            "passed": PATHS["action_aware_ckpt"].exists()
            and bool(get(data["action_aware_runtime"], "usable_for_guidance", False))
            and (get(data["action_aware_eval"], "mixed_group_cv.binary_auc.mean", 0.0) or 0.0) >= 0.95
            and (get(data["action_aware_eval"], "mixed_group_cv.score_corr.mean", 0.0) or 0.0) >= 0.70
            and get(data["action_aware_guidance_suitability"], "passes_guidance_suitability") is True
            and get(data["action_aware_guidance_suitability"], "recommended_mode") == "quality",
            "evidence": {
                "checkpoint": file_info(PATHS["action_aware_ckpt"]),
                "usable_for_guidance": get(data["action_aware_runtime"], "usable_for_guidance"),
                "grad_action_norm": get(data["action_aware_runtime"], "grad_action_norm"),
                "mixed_auc": get(data["action_aware_eval"], "mixed_group_cv.binary_auc.mean"),
                "mixed_score_corr": get(data["action_aware_eval"], "mixed_group_cv.score_corr.mean"),
                "guidance_suitability_pass": get(
                    data["action_aware_guidance_suitability"], "passes_guidance_suitability"
                ),
                "recommended_mode": get(data["action_aware_guidance_suitability"], "recommended_mode"),
                "hybrid_improved_rate": get(
                    data["action_aware_guidance_suitability"],
                    "modes.hybrid.gradient_probe.mixed.improved_rate",
                ),
                "quality_line_search_accepted_rate": get(
                    data["action_aware_guidance_suitability"],
                    "modes.quality.gradient_probe.mixed.line_search.accepted_rate",
                ),
                "cross_insertion_to_board_macro_f1": get(
                    data["action_aware_eval"], "cross_task.insertion_to_board.binary_macro_f1"
                ),
                "cross_board_to_insertion_macro_f1": get(
                    data["action_aware_eval"], "cross_task.board_to_insertion.binary_macro_f1"
                ),
            },
        },
        {
            "name": "three_arm_scorer_ablation_smoke_pass",
            "passed": bool(get(data["scorer_ablation_gate_smoke"], "overall_pass", False))
            and get(data["scorer_ablation_gate_smoke"], "scientific_evidence") is False,
            "evidence": {
                "overall_pass": get(data["scorer_ablation_gate_smoke"], "overall_pass"),
                "scientific_evidence": get(data["scorer_ablation_gate_smoke"], "scientific_evidence"),
                "tasks": get(data["scorer_ablation_gate_smoke"], "tasks"),
            },
        },
        {
            "name": "formal_rollout_gate_runner_preflight_exists",
            "passed": get(data["formal_rollout_gate_runner"], "preflight_ready") is False
            and get(data["formal_rollout_gate_runner"], "scientific_evidence") is False
            and get(data["formal_rollout_gate_runner"], "run_skip_reason") is not None
            and "--require_outcome_metadata"
            in str(get(data["formal_rollout_gate_runner"], "tasks.insertion.commands.two_arm", ""))
            and "--require_scorer_freeze_metadata"
            in str(get(data["formal_rollout_gate_runner"], "tasks.insertion.commands.two_arm", ""))
            and "--require_outcome_metadata"
            in str(get(data["formal_rollout_gate_runner"], "tasks.insertion.commands.three_arm", ""))
            and "--require_scorer_freeze_metadata"
            in str(get(data["formal_rollout_gate_runner"], "tasks.insertion.commands.three_arm", ""))
            and "--require_outcome_metadata"
            in str(get(data["formal_rollout_gate_runner"], "tasks.board.commands.two_arm", ""))
            and "--require_scorer_freeze_metadata"
            in str(get(data["formal_rollout_gate_runner"], "tasks.board.commands.two_arm", ""))
            and "--require_outcome_metadata"
            in str(get(data["formal_rollout_gate_runner"], "tasks.board.commands.three_arm", ""))
            and "--require_scorer_freeze_metadata"
            in str(get(data["formal_rollout_gate_runner"], "tasks.board.commands.three_arm", "")),
            "evidence": {
                "preflight_ready": get(data["formal_rollout_gate_runner"], "preflight_ready"),
                "use_generated_pairing": get(data["formal_rollout_gate_runner"], "use_generated_pairing"),
                "scientific_evidence": get(data["formal_rollout_gate_runner"], "scientific_evidence"),
                "tasks": get(data["formal_rollout_gate_runner"], "tasks"),
                "run_skip_reason": get(data["formal_rollout_gate_runner"], "run_skip_reason"),
            },
        },
        {
            "name": "optional_action_aware_rollout_gate_runner_preflight_exists",
            "passed": get(data["optional_action_aware_rollout_gate_runner"], "scientific_evidence") is False
            and get(data["optional_action_aware_rollout_gate_runner"], "formal_gate_dependency") is False
            and get(data["optional_action_aware_rollout_gate_runner"], "run_gates_requested") is False
            and get(data["optional_action_aware_rollout_gate_runner"], "tasks.insertion.checks.action_aware_guided.n_hdf5")
            is not None
            and get(data["optional_action_aware_rollout_gate_runner"], "tasks.board.checks.action_aware_guided.n_hdf5")
            is not None,
            "evidence": {
                "preflight_ready": get(data["optional_action_aware_rollout_gate_runner"], "preflight_ready"),
                "scientific_evidence": get(data["optional_action_aware_rollout_gate_runner"], "scientific_evidence"),
                "formal_gate_dependency": get(data["optional_action_aware_rollout_gate_runner"], "formal_gate_dependency"),
                "tasks": get(data["optional_action_aware_rollout_gate_runner"], "tasks"),
                "interpretation": get(data["optional_action_aware_rollout_gate_runner"], "interpretation"),
            },
        },
        {
            "name": "formal_launch_sheet_ready",
            "passed": bool(get(data["formal_launch_sheet"], "launch_sheet_ready", False))
            and get(data["formal_launch_sheet"], "scientific_evidence") is False
            and get(data["formal_launch_sheet"], "not_reranking") is True
            and get(data["formal_launch_sheet"], "not_every_step_ddpm_guidance") is True
            and "serve_dp_tac_quality_guided" in str(get(data["formal_launch_sheet"], "tasks.insertion.launch_commands.baseline", ""))
            and "--disable_guidance" in str(get(data["formal_launch_sheet"], "tasks.insertion.launch_commands.baseline", ""))
            and "serve_dp_tac_quality_guided" in str(get(data["formal_launch_sheet"], "tasks.insertion.launch_commands.default_guided", ""))
            and "serve_dp_tac_quality_guided" in str(get(data["formal_launch_sheet"], "tasks.insertion.launch_commands.distilled_guided", ""))
            and "action_aware_guided" in str(get(data["formal_launch_sheet"], "tasks.insertion.launch_commands.action_aware_guided", ""))
            and "serve_dp_tac_quality_guided" in str(get(data["formal_launch_sheet"], "tasks.board.launch_commands.baseline", ""))
            and "--disable_guidance" in str(get(data["formal_launch_sheet"], "tasks.board.launch_commands.baseline", ""))
            and "serve_dp_tac_quality_guided" in str(get(data["formal_launch_sheet"], "tasks.board.launch_commands.default_guided", ""))
            and "serve_dp_tac_quality_guided" in str(get(data["formal_launch_sheet"], "tasks.board.launch_commands.distilled_guided", ""))
            and "action_aware_guided" in str(get(data["formal_launch_sheet"], "tasks.board.launch_commands.action_aware_guided", ""))
            and get(data["guided_server_insertion_baseline_no_guidance_smoke"], "dry_run_guidance_smoke_pass") is True
            and get(data["guided_server_board_baseline_no_guidance_smoke"], "dry_run_guidance_smoke_pass") is True
            and "run_formal_tac_quality_rollout_gates.py" in str(get(data["formal_launch_sheet"], "all_tasks_gate_runner_command", ""))
            and "--use_generated_pairing" in str(get(data["formal_launch_sheet"], "all_tasks_gate_runner_command", ""))
            and "build_tac_quality_rollout_pairing.py" in str(get(data["formal_launch_sheet"], "post_collection_pairing_command", "")),
            "evidence": {
                "launch_sheet_ready": get(data["formal_launch_sheet"], "launch_sheet_ready"),
                "rollout_root": get(data["formal_launch_sheet"], "rollout_root"),
                "generated_pairing_dir": get(data["formal_launch_sheet"], "generated_pairing_dir"),
                "post_collection_pairing_command": get(data["formal_launch_sheet"], "post_collection_pairing_command"),
                "insertion_ports": get(data["formal_launch_sheet"], "tasks.insertion.ports"),
                "board_ports": get(data["formal_launch_sheet"], "tasks.board.ports"),
                "optional_action_aware_commands": {
                    "insertion": get(data["formal_launch_sheet"], "tasks.insertion.launch_commands.action_aware_guided"),
                    "board": get(data["formal_launch_sheet"], "tasks.board.launch_commands.action_aware_guided"),
                },
                "insertion_baseline_no_guidance_smoke": get(data["guided_server_insertion_baseline_no_guidance_smoke"], "dry_run_guidance_smoke_pass"),
                "board_baseline_no_guidance_smoke": get(data["guided_server_board_baseline_no_guidance_smoke"], "dry_run_guidance_smoke_pass"),
                "all_tasks_gate_runner_command": get(data["formal_launch_sheet"], "all_tasks_gate_runner_command"),
            },
        },
        {
            "name": "formal_launch_sheet_command_smoke_pass",
            "passed": bool(get(data["formal_launch_sheet_smoke"], "overall_pass", False))
            and get(data["formal_launch_sheet_smoke"], "scientific_evidence") is False
            and get(data["formal_launch_sheet_smoke"], "checks.all_commands_present") is True
            and get(data["formal_launch_sheet_smoke"], "checks.formal_six_commands_present") is True
            and get(data["formal_launch_sheet_smoke"], "checks.optional_action_aware_commands_present") is True
            and get(data["formal_launch_sheet_smoke"], "checks.all_commands_pass_process") is True
            and get(data["formal_launch_sheet_smoke"], "checks.all_commands_pass_output_contract") is True
            and get(data["formal_launch_sheet_smoke"], "checks.baseline_commands_disable_guidance") is True
            and get(data["formal_launch_sheet_smoke"], "checks.guided_commands_have_gradients") is True,
            "evidence": {
                "overall_pass": get(data["formal_launch_sheet_smoke"], "overall_pass"),
                "checks": get(data["formal_launch_sheet_smoke"], "checks"),
                "commands": [
                    {
                        "task": row.get("task"),
                        "arm": row.get("arm"),
                        "pass": row.get("passes_launch_command_smoke"),
                        "guidance_disabled": get(row, "output.guidance_disabled"),
                        "finite_grad_rate": get(row, "output.finite_grad_rate"),
                        "positive_grad_rate": get(row, "output.positive_grad_rate"),
                    }
                    for row in (get(data["formal_launch_sheet_smoke"], "commands", []) or [])
                ],
            },
        },
        {
            "name": "formal_collection_readiness_exists",
            "passed": get(data["formal_collection_readiness"], "scientific_evidence") is False
            and get(data["formal_collection_readiness"], "all_collection_dirs_exist") is True
            and get(data["formal_collection_readiness"], "all_templates_exist") is True
            and get(data["formal_collection_readiness"], "ready_for_gate_runner") is False
            and get(data["formal_collection_readiness"], "next_required_step") is not None,
            "evidence": {
                "all_collection_dirs_exist": get(data["formal_collection_readiness"], "all_collection_dirs_exist"),
                "all_templates_exist": get(data["formal_collection_readiness"], "all_templates_exist"),
                "ready_for_two_arm_gates": get(data["formal_collection_readiness"], "ready_for_two_arm_gates"),
                "ready_for_three_arm_gates": get(data["formal_collection_readiness"], "ready_for_three_arm_gates"),
                "ready_for_gate_runner": get(data["formal_collection_readiness"], "ready_for_gate_runner"),
                "pairing_report_exists": get(data["formal_collection_readiness"], "pairing_report_exists"),
                "post_collection_pairing_command": get(data["formal_collection_readiness"], "post_collection_pairing_command"),
                "missing_items": get(data["formal_collection_readiness"], "missing_items"),
            },
        },
        {
            "name": "formal_rollout_pairing_generator_exists",
            "passed": get(data["formal_rollout_pairing"], "scientific_evidence") is False
            and get(data["formal_rollout_pairing"], "overall_ready") is False
            and get(data["formal_rollout_pairing"], "schedule_used") is True
            and get(data["formal_rollout_pairing"], "tasks.insertion.schedule_mode") is True
            and get(data["formal_rollout_pairing"], "tasks.board.schedule_mode") is True
            and get(data["formal_rollout_pairing"], "tasks.insertion.outputs.pairing_csv") is not None
            and get(data["formal_rollout_pairing"], "tasks.board.outputs.three_arm_pairing_csv") is not None,
            "evidence": {
                "overall_ready": get(data["formal_rollout_pairing"], "overall_ready"),
                "schedule_used": get(data["formal_rollout_pairing"], "schedule_used"),
                "insertion": get(data["formal_rollout_pairing"], "tasks.insertion"),
                "board": get(data["formal_rollout_pairing"], "tasks.board"),
            },
        },
        {
            "name": "pairing_metadata_audit_tracks_gate_input_completeness",
            "passed": get(data["pairing_metadata_audit"], "scientific_evidence") is False
            and get(data["pairing_metadata_audit"], "all_tasks_ready") is False
            and get(data["pairing_metadata_audit"], "tasks.insertion.counts.two_arm_rows") is not None
            and get(data["pairing_metadata_audit"], "tasks.board.counts.metadata_rows") is not None,
            "evidence": {
                "all_tasks_ready": get(data["pairing_metadata_audit"], "all_tasks_ready"),
                "insertion": get(data["pairing_metadata_audit"], "tasks.insertion"),
                "board": get(data["pairing_metadata_audit"], "tasks.board"),
                "next_required_step": get(data["pairing_metadata_audit"], "next_required_step"),
            },
        },
        {
            "name": "hdf5_schema_audit_tracks_rollout_field_completeness",
            "passed": get(data["hdf5_schema_audit"], "scientific_evidence") is False
            and get(data["hdf5_schema_audit"], "all_tasks_ready") is False
            and get(data["hdf5_schema_audit"], "tasks.insertion.arms.baseline.n_hdf5") is not None
            and get(data["hdf5_schema_audit"], "tasks.board.arms.distilled_guided.n_hdf5") is not None,
            "evidence": {
                "all_tasks_ready": get(data["hdf5_schema_audit"], "all_tasks_ready"),
                "insertion": get(data["hdf5_schema_audit"], "tasks.insertion"),
                "board": get(data["hdf5_schema_audit"], "tasks.board"),
                "next_required_step": get(data["hdf5_schema_audit"], "next_required_step"),
            },
        },
        {
            "name": "generated_pairing_gate_runner_synthetic_smoke_pass",
            "passed": get(data["generated_pairing_gate_runner_smoke"], "overall_pass") is True
            and get(data["generated_pairing_gate_runner_smoke"], "scientific_evidence") is False
            and get(data["generated_pairing_gate_runner_smoke"], "pairing_report.overall_ready") is True
            and get(data["generated_pairing_gate_runner_smoke"], "gate_report.use_generated_pairing") is True
            and get(data["generated_pairing_gate_runner_smoke"], "gate_report.preflight_ready") is True
            and get(data["generated_pairing_gate_runner_smoke"], "gate_report.all_requested_gates_passed") is True
            and get(
                data["generated_pairing_gate_runner_smoke"],
                "gate_report.strict_outcome_metadata_commands.all_require_outcome_metadata",
            )
            is True
            and get(
                data["generated_pairing_gate_runner_smoke"],
                "gate_report.strict_outcome_metadata_commands.all_require_scorer_freeze_metadata",
            )
            is True
            and get(
                data["generated_pairing_gate_runner_smoke"],
                "gate_report.strict_outcome_metadata_outputs.all_gate_outputs_require_and_pass_outcome_metadata",
            )
            is True,
            "evidence": {
                "overall_pass": get(data["generated_pairing_gate_runner_smoke"], "overall_pass"),
                "scientific_evidence": get(data["generated_pairing_gate_runner_smoke"], "scientific_evidence"),
                "pairing_report": get(data["generated_pairing_gate_runner_smoke"], "pairing_report"),
                "gate_report": get(data["generated_pairing_gate_runner_smoke"], "gate_report"),
            },
        },
        {
            "name": "real_rollout_source_audit_preserves_gap",
            "passed": get(data["real_rollout_source_audit"], "synthetic_guardrail_pass") is True
            and get(data["real_rollout_source_audit"], "all_four_real_evidence_present") is False
            and (get(data["real_rollout_source_audit"], "n_blockers", 0) or 0) == 4,
            "evidence": {
                "all_four_real_evidence_present": get(data["real_rollout_source_audit"], "all_four_real_evidence_present"),
                "n_real_evidence": get(data["real_rollout_source_audit"], "n_real_evidence"),
                "n_blockers": get(data["real_rollout_source_audit"], "n_blockers"),
                "synthetic_guardrail_pass": get(data["real_rollout_source_audit"], "synthetic_guardrail_pass"),
                "artifacts": get(data["real_rollout_source_audit"], "artifacts"),
            },
        },
        {
            "name": "post_collection_pipeline_preflight_runs_and_preserves_gap",
            "passed": get(data["post_collection_pipeline"], "pipeline_pass") is True
            and get(data["post_collection_pipeline"], "run_gates_requested") is False
            and get(data["post_collection_pipeline"], "run_optional_action_aware_gate_requested") is False
            and get(data["post_collection_pipeline"], "can_run_gates") is False
            and get(data["post_collection_pipeline"], "metadata_audit.all_tasks_ready") is False
            and get(data["post_collection_pipeline"], "hdf5_schema_audit.all_tasks_ready") is False
            and get(data["post_collection_pipeline"], "optional_action_aware.formal_gate_dependency") is False
            and get(data["post_collection_pipeline"], "optional_action_aware.run_gates_requested") is False
            and get(data["post_collection_pipeline"], "optional_action_aware.tasks.insertion.checks.action_aware_guided.n_hdf5")
            is not None
            and get(data["post_collection_pipeline"], "optional_action_aware.tasks.board.checks.action_aware_guided.n_hdf5")
            is not None
            and get(data["post_collection_pipeline"], "source_audit.n_blockers") == 4,
            "evidence": {
                "pipeline_pass": get(data["post_collection_pipeline"], "pipeline_pass"),
                "can_run_gates": get(data["post_collection_pipeline"], "can_run_gates"),
                "run_gates_requested": get(data["post_collection_pipeline"], "run_gates_requested"),
                "run_optional_action_aware_gate_requested": get(
                    data["post_collection_pipeline"], "run_optional_action_aware_gate_requested"
                ),
                "pairing_ready": get(data["post_collection_pipeline"], "pairing.overall_ready"),
                "metadata_ready": get(data["post_collection_pipeline"], "metadata_audit.all_tasks_ready"),
                "schema_ready": get(data["post_collection_pipeline"], "hdf5_schema_audit.all_tasks_ready"),
                "preflight_ready": get(data["post_collection_pipeline"], "gate_runner.preflight_ready"),
                "optional_action_aware": get(data["post_collection_pipeline"], "optional_action_aware"),
                "source_audit": get(data["post_collection_pipeline"], "source_audit"),
            },
        },
        {
            "name": "post_collection_pipeline_run_gates_synthetic_smoke_pass",
            "passed": get(data["post_collection_pipeline_smoke"], "overall_pass") is True
            and get(data["post_collection_pipeline_smoke"], "scientific_evidence") is False
            and get(data["post_collection_pipeline_smoke"], "checks.pipeline_pass") is True
            and get(data["post_collection_pipeline_smoke"], "checks.can_run_gates") is True
            and get(data["post_collection_pipeline_smoke"], "checks.metadata_review_present") is True
            and get(data["post_collection_pipeline_smoke"], "checks.metadata_review_needed_zero") is True
            and get(data["post_collection_pipeline_smoke"], "checks.schema_ready") is True
            and get(data["post_collection_pipeline_smoke"], "checks.formal_gate_commands_require_outcome_metadata")
            is True
            and get(data["post_collection_pipeline_smoke"], "checks.formal_gate_commands_require_scorer_freeze_metadata")
            is True
            and get(data["post_collection_pipeline_smoke"], "checks.gates_passed") is True
            and get(data["post_collection_pipeline_smoke"], "checks.formal_gate_outputs_have_outcome_metadata")
            is True
            and get(data["post_collection_pipeline_smoke"], "checks.optional_action_aware_preflight_ready") is True
            and get(data["post_collection_pipeline_smoke"], "checks.optional_action_aware_gates_passed") is True
            and get(data["post_collection_pipeline_smoke"], "checks.optional_action_aware_not_formal_dependency") is True
            and get(data["post_collection_pipeline_smoke"], "checks.source_guardrail_keeps_formal_gap") is True,
            "evidence": {
                "overall_pass": get(data["post_collection_pipeline_smoke"], "overall_pass"),
                "checks": get(data["post_collection_pipeline_smoke"], "checks"),
                "pipeline_summary": get(data["post_collection_pipeline_smoke"], "pipeline_summary"),
            },
        },
        {
            "name": "metadata_review_sheet_synthetic_smoke_pass",
            "passed": get(data["metadata_review_sheet_smoke"], "overall_pass") is True
            and get(data["metadata_review_sheet_smoke"], "scientific_evidence") is False
            and get(data["metadata_review_sheet_smoke"], "checks.first_detects_four_rows") is True
            and get(data["metadata_review_sheet_smoke"], "checks.second_applies_four_rows") is True
            and get(data["metadata_review_sheet_smoke"], "checks.second_merged_complete") is True
            and get(data["metadata_review_sheet_smoke"], "checks.does_not_auto_infer") is True,
            "evidence": {
                "overall_pass": get(data["metadata_review_sheet_smoke"], "overall_pass"),
                "checks": get(data["metadata_review_sheet_smoke"], "checks"),
            },
        },
        {
            "name": "finalize_collected_hdf5_synthetic_smoke_pass",
            "passed": get(data["finalize_collected_hdf5_smoke"], "overall_pass") is True
            and get(data["finalize_collected_hdf5_smoke"], "scientific_evidence") is False
            and get(data["finalize_collected_hdf5_smoke"], "checks.copy_finalize_pass") is True
            and get(data["finalize_collected_hdf5_smoke"], "checks.copy_operation") is True
            and get(data["finalize_collected_hdf5_smoke"], "checks.copy_keeps_source") is True
            and get(data["finalize_collected_hdf5_smoke"], "checks.copy_writes_freeze_attrs") is True
            and get(data["finalize_collected_hdf5_smoke"], "checks.refuse_existing_target") is True
            and get(data["finalize_collected_hdf5_smoke"], "checks.source_dir_finalize_pass") is True
            and get(data["finalize_collected_hdf5_smoke"], "checks.source_dir_picks_newest") is True
            and get(data["finalize_collected_hdf5_smoke"], "checks.source_dir_writes_freeze_attrs") is True,
            "evidence": {
                "overall_pass": get(data["finalize_collected_hdf5_smoke"], "overall_pass"),
                "checks": get(data["finalize_collected_hdf5_smoke"], "checks"),
                "reports": get(data["finalize_collected_hdf5_smoke"], "reports"),
                "note": get(data["finalize_collected_hdf5_smoke"], "note"),
            },
        },
        {
            "name": "finalize_and_refresh_synthetic_smoke_pass",
            "passed": get(data["finalize_and_refresh_smoke"], "overall_pass") is True
            and get(data["finalize_and_refresh_smoke"], "scientific_evidence") is False
            and get(data["finalize_and_refresh_smoke"], "checks.wrapper_pass") is True
            and get(data["finalize_and_refresh_smoke"], "checks.target_exists") is True
            and get(data["finalize_and_refresh_smoke"], "checks.source_kept_by_copy") is True
            and get(data["finalize_and_refresh_smoke"], "checks.freeze_attrs_written") is True
            and get(data["finalize_and_refresh_smoke"], "checks.skip_refresh_used") is True,
            "evidence": {
                "overall_pass": get(data["finalize_and_refresh_smoke"], "overall_pass"),
                "checks": get(data["finalize_and_refresh_smoke"], "checks"),
                "note": get(data["finalize_and_refresh_smoke"], "note"),
                "runner_summary": get(data["finalize_and_refresh_smoke"], "runner_summary"),
            },
        },
        {
            "name": "outcome_label_card_pass",
            "passed": get(data["outcome_label_card"], "outcome_label_card_pass") is True
            and get(data["outcome_label_card"], "scientific_evidence") is False
            and get(data["outcome_label_card_smoke"], "overall_pass") is True
            and get(data["outcome_label_card_smoke"], "scientific_evidence") is False
            and get(data["outcome_label_card_smoke"], "checks.manual_no_auto_inference_guardrail") is True,
            "evidence": {
                "outcome_label_card_pass": get(data["outcome_label_card"], "outcome_label_card_pass"),
                "tasks": list((get(data["outcome_label_card"], "tasks", {}) or {}).keys()),
                "smoke_checks": get(data["outcome_label_card_smoke"], "checks"),
            },
        },
        {
            "name": "scorer_freeze_manifest_pass",
            "passed": get(data["scorer_freeze_manifest"], "scorer_freeze_manifest_pass") is True
            and get(data["scorer_freeze_manifest"], "scientific_evidence") is False
            and get(data["scorer_freeze_manifest_smoke"], "overall_pass") is True
            and get(data["scorer_freeze_manifest_smoke"], "scientific_evidence") is False
            and get(data["scorer_freeze_manifest_smoke"], "checks.guided_checkpoints_have_sha256") is True
            and get(data["scorer_freeze_manifest_smoke"], "checks.runtime_modules_have_sha256") is True,
            "evidence": {
                "scorer_freeze_manifest_pass": get(data["scorer_freeze_manifest"], "scorer_freeze_manifest_pass"),
                "n_arms": len(get(data["scorer_freeze_manifest"], "arms", {}) or {}),
                "n_runtime_modules": len(get(data["scorer_freeze_manifest"], "runtime_modules", {}) or {}),
                "smoke_checks": get(data["scorer_freeze_manifest_smoke"], "checks"),
            },
        },
        {
            "name": "real_rollout_acceptance_protocol_pass",
            "passed": get(data["real_rollout_acceptance_protocol"], "protocol_pass") is True
            and get(data["real_rollout_acceptance_protocol"], "scientific_evidence") is False
            and get(data["real_rollout_acceptance_protocol"], "outcome_label_card.outcome_label_card_pass") is True
            and get(data["real_rollout_acceptance_protocol"], "tasks.insertion.collection.paired_n_pairs", 0) >= 10
            and get(data["real_rollout_acceptance_protocol"], "tasks.board.collection.paired_n_pairs", 0) >= 10
            and len(get(data["real_rollout_acceptance_protocol"], "completion_blockers_to_close", []) or []) == 4
            and "synthetic HDF5 smoke outputs"
            in str(get(data["real_rollout_acceptance_protocol"], "cannot_count_as_completion", ""))
            and "eval_real_rollout_quality_gate.py"
            in str(get(data["real_rollout_acceptance_protocol"], "tasks.insertion.two_arm_gate.command", ""))
            and "eval_real_rollout_scorer_ablation_gate.py"
            in str(get(data["real_rollout_acceptance_protocol"], "tasks.board.three_arm_ablation_gate.command", "")),
            "evidence": {
                "protocol_pass": get(data["real_rollout_acceptance_protocol"], "protocol_pass"),
                "scientific_evidence": get(data["real_rollout_acceptance_protocol"], "scientific_evidence"),
                "paired_n_pairs": {
                    "insertion": get(
                        data["real_rollout_acceptance_protocol"],
                        "tasks.insertion.collection.paired_n_pairs",
                    ),
                    "board": get(
                        data["real_rollout_acceptance_protocol"],
                        "tasks.board.collection.paired_n_pairs",
                    ),
                },
                "completion_blockers_to_close": get(
                    data["real_rollout_acceptance_protocol"], "completion_blockers_to_close"
                ),
                "review_sequence": get(data["real_rollout_acceptance_protocol"], "review_sequence"),
                "outcome_label_card": get(data["real_rollout_acceptance_protocol"], "outcome_label_card"),
            },
        },
        {
            "name": "formal_rollout_runbook_pass",
            "passed": get(data["formal_rollout_runbook"], "runbook_pass") is True
            and get(data["formal_rollout_runbook"], "scientific_evidence") is False
            and get(data["formal_rollout_runbook"], "protocol_pass") is True
            and get(data["formal_rollout_runbook"], "launch_sheet_ready") is True
            and get(data["formal_rollout_runbook"], "pipeline_pass") is True
            and get(data["formal_rollout_runbook"], "outcome_label_card.outcome_label_card_pass") is True
            and get(data["formal_rollout_runbook"], "scorer_freeze_manifest.scorer_freeze_manifest_pass") is True
            and get(data["formal_rollout_runbook"], "tasks.insertion.paired_n_pairs", 0) >= 10
            and get(data["formal_rollout_runbook"], "tasks.board.paired_n_pairs", 0) >= 10
            and len(get(data["formal_rollout_runbook"], "completion_blockers_to_close", []) or []) == 4
            and get(data["formal_rollout_runbook"], "collection_schedule.schedule_pass") is True
            and str(get(data["formal_rollout_runbook"], "collection_schedule.csv", "")).endswith(
                "tac_quality_collection_schedule.csv"
            )
            and "run_tac_quality_post_collection_pipeline.py"
            in str(get(data["formal_rollout_runbook"], "post_collection_commands.pipeline_run_gates", ""))
            and "build_tac_quality_collection_schedule.py"
            in str(get(data["formal_rollout_runbook"], "post_collection_commands.build_collection_schedule", ""))
            and "build_tac_quality_next_collection_step.py"
            in str(get(data["formal_rollout_runbook"], "post_collection_commands.next_collection_step", ""))
            and "run_tac_quality_next_collection_step_smoke.py"
            in str(get(data["formal_rollout_runbook"], "post_collection_commands.next_collection_step_smoke_runner", ""))
            and "conda run -n TactileACT"
            in str(get(data["formal_rollout_runbook"], "post_collection_commands.next_collection_step_smoke_runner", ""))
            and "build_tac_quality_current_collection_gate.py"
            in str(get(data["formal_rollout_runbook"], "post_collection_commands.current_collection_gate", ""))
            and "build_tac_quality_current_collection_handoff.py"
            in str(get(data["formal_rollout_runbook"], "post_collection_commands.current_collection_handoff", ""))
            and "build_tac_quality_outcome_label_card.py"
            in str(get(data["formal_rollout_runbook"], "post_collection_commands.outcome_label_card", ""))
            and "build_tac_quality_scorer_freeze_manifest.py"
            in str(get(data["formal_rollout_runbook"], "post_collection_commands.scorer_freeze_manifest", ""))
            and "finalize_and_refresh_tac_quality_collection.py"
            in str(get(data["formal_rollout_runbook"], "post_collection_commands.finalize_and_refresh_collected_hdf5", ""))
            and "finalize_tac_quality_collected_hdf5.py"
            in str(get(data["formal_rollout_runbook"], "post_collection_commands.finalize_collected_hdf5", ""))
            and get(data["formal_rollout_runbook"], "collection_progress.progress_pass") is True
            and get(data["formal_rollout_runbook"], "next_collection_step.next_step_pass") is True
            and "--dry_run_guidance_smoke"
            in str(get(data["formal_rollout_runbook"], "next_collection_step.pre_collection_dry_run_command", ""))
            and "--smoke_output"
            in str(get(data["formal_rollout_runbook"], "next_collection_step.pre_collection_dry_run_command", ""))
            and "serve_dp_tac_quality_guided"
            in str(get(data["formal_rollout_runbook"], "tasks.insertion.arms", "")),
            "evidence": {
                "runbook_pass": get(data["formal_rollout_runbook"], "runbook_pass"),
                "ready_for_gate_runner": get(data["formal_rollout_runbook"], "ready_for_gate_runner"),
                "rollout_root": get(data["formal_rollout_runbook"], "rollout_root"),
                "paired_n_pairs": {
                    "insertion": get(data["formal_rollout_runbook"], "tasks.insertion.paired_n_pairs"),
                    "board": get(data["formal_rollout_runbook"], "tasks.board.paired_n_pairs"),
                },
                "post_collection_commands": get(data["formal_rollout_runbook"], "post_collection_commands"),
                "collection_schedule": get(data["formal_rollout_runbook"], "collection_schedule"),
                "collection_progress": get(data["formal_rollout_runbook"], "collection_progress"),
                "next_collection_step": get(data["formal_rollout_runbook"], "next_collection_step"),
                "outcome_label_card": get(data["formal_rollout_runbook"], "outcome_label_card"),
                "scorer_freeze_manifest": get(data["formal_rollout_runbook"], "scorer_freeze_manifest"),
            },
        },
        {
            "name": "formal_rollout_runbook_smoke_pass",
            "passed": get(data["formal_rollout_runbook_smoke"], "overall_pass") is True
            and get(data["formal_rollout_runbook_smoke"], "scientific_evidence") is False
            and get(data["formal_rollout_runbook_smoke"], "checks.runbook_pass") is True
            and get(data["formal_rollout_runbook_smoke"], "checks.ready_for_gate_runner_false_until_collection") is True
            and get(data["formal_rollout_runbook_smoke"], "checks.all_tasks_pass") is True
            and get(data["formal_rollout_runbook_smoke"], "checks.completion_blockers_four") is True
            and get(data["formal_rollout_runbook_smoke"], "checks.post_collection_run_gates_command_present") is True
            and get(data["formal_rollout_runbook_smoke"], "checks.build_schedule_command_present") is True
            and get(data["formal_rollout_runbook_smoke"], "checks.collection_progress_command_present") is True
            and get(data["formal_rollout_runbook_smoke"], "checks.next_collection_step_command_present") is True
            and get(data["formal_rollout_runbook_smoke"], "checks.next_collection_step_smoke_runner_command_present") is True
            and get(data["formal_rollout_runbook_smoke"], "checks.current_collection_gate_command_present") is True
            and get(data["formal_rollout_runbook_smoke"], "checks.current_collection_handoff_command_present") is True
            and get(data["formal_rollout_runbook_smoke"], "checks.outcome_label_card_command_present") is True
            and get(data["formal_rollout_runbook_smoke"], "checks.outcome_label_card_present") is True
            and get(data["formal_rollout_runbook_smoke"], "checks.outcome_label_card_smoke_pass") is True
            and get(data["formal_rollout_runbook_smoke"], "checks.scorer_freeze_manifest_command_present") is True
            and get(data["formal_rollout_runbook_smoke"], "checks.scorer_freeze_manifest_present") is True
            and get(data["formal_rollout_runbook_smoke"], "checks.scorer_freeze_smoke_pass") is True
            and get(data["formal_rollout_runbook_smoke"], "checks.finalize_and_refresh_command_present") is True
            and get(data["formal_rollout_runbook_smoke"], "checks.finalize_and_refresh_source_dir_command_present") is True
            and get(data["formal_rollout_runbook_smoke"], "checks.finalize_collected_hdf5_command_present") is True
            and get(data["formal_rollout_runbook_smoke"], "checks.finalize_source_dir_command_present") is True
            and get(data["formal_rollout_runbook_smoke"], "checks.next_step_finalize_template_present") is True
            and get(data["formal_rollout_runbook_smoke"], "checks.next_step_pre_collection_dry_run_present") is True
            and get(data["formal_rollout_runbook_smoke"], "checks.next_step_pre_collection_dry_run_output_present") is True
            and get(data["formal_rollout_runbook_smoke"], "checks.collection_steps_require_pre_collection_dry_run") is True
            and get(data["formal_rollout_runbook_smoke"], "checks.runbook_references_schedule_csv") is True
            and get(data["formal_rollout_runbook_smoke"], "checks.runbook_references_collection_progress") is True
            and get(data["formal_rollout_runbook_smoke"], "checks.runbook_references_next_collection_step") is True
            and get(data["formal_rollout_runbook_smoke"], "checks.schedule_pass") is True
            and get(data["formal_rollout_runbook_smoke"], "checks.schedule_counterbalances_both_tasks") is True
            and get(data["formal_rollout_runbook_smoke"], "checks.launch_sheet_smoke_pass") is True,
            "evidence": {
                "overall_pass": get(data["formal_rollout_runbook_smoke"], "overall_pass"),
                "checks": get(data["formal_rollout_runbook_smoke"], "checks"),
                "tasks": get(data["formal_rollout_runbook_smoke"], "tasks"),
            },
        },
        {
            "name": "formal_collection_schedule_counterbalanced",
            "passed": get(data["formal_collection_schedule"], "schedule_pass") is True
            and get(data["formal_collection_schedule"], "scientific_evidence") is False
            and get(data["formal_collection_schedule"], "tasks.insertion.counterbalance_pass") is True
            and get(data["formal_collection_schedule"], "tasks.board.counterbalance_pass") is True
            and get(data["formal_collection_schedule"], "tasks.insertion.paired_n_pairs") == 12
            and get(data["formal_collection_schedule"], "tasks.board.paired_n_pairs") == 12
            and len(get(data["formal_collection_schedule"], "long_schedule_rows", []) or []) == 72
            and str(((get(data["formal_collection_schedule"], "long_schedule_rows", []) or [{}])[0]).get("recommended_filename", "")).endswith(".hdf5")
            and str(((get(data["formal_collection_schedule"], "long_schedule_rows", []) or [{}])[0]).get("recommended_path", "")).endswith(".hdf5")
            and "Changing arm order after seeing rollout outcomes"
            in str(get(data["formal_collection_schedule"], "cannot_count_as_completion", "")),
            "evidence": {
                "schedule_pass": get(data["formal_collection_schedule"], "schedule_pass"),
                "task_order_policy": get(data["formal_collection_schedule"], "task_order_policy"),
                "within_triplet_policy": get(data["formal_collection_schedule"], "within_triplet_policy"),
                "insertion_position_counts": get(data["formal_collection_schedule"], "tasks.insertion.position_counts"),
                "board_position_counts": get(data["formal_collection_schedule"], "tasks.board.position_counts"),
            },
        },
        {
            "name": "formal_collection_progress_tracks_next_trial",
            "passed": get(data["formal_collection_progress"], "progress_pass") is True
            and get(data["formal_collection_progress"], "scientific_evidence") is False
            and get(data["formal_collection_progress"], "schedule_pass") is True
            and get(data["formal_collection_progress"], "n_scheduled_rows") == 72
            and get(data["formal_collection_progress"], "ready_for_post_collection") is False
            and get(data["formal_collection_progress"], "next_row.task") is not None
            and str(get(data["formal_collection_progress"], "next_row.recommended_filename", "")).endswith(".hdf5")
            and str(get(data["formal_collection_progress"], "next_row.recommended_path", "")).endswith(".hdf5")
            and "Do not change arm order after seeing rollout outcomes"
            in str(get(data["formal_collection_progress"], "guardrails", "")),
            "evidence": {
                "progress_pass": get(data["formal_collection_progress"], "progress_pass"),
                "n_completed_rows": get(data["formal_collection_progress"], "n_completed_rows"),
                "n_scheduled_rows": get(data["formal_collection_progress"], "n_scheduled_rows"),
                "ready_for_post_collection": get(data["formal_collection_progress"], "ready_for_post_collection"),
                "next_row": get(data["formal_collection_progress"], "next_row"),
                "arm_totals": get(data["formal_collection_progress"], "arm_totals"),
            },
        },
        {
            "name": "formal_next_collection_step_ready",
            "passed": get(data["formal_next_collection_step"], "next_step_pass") is True
            and get(data["formal_next_collection_step"], "scientific_evidence") is False
            and get(data["formal_next_collection_step"], "progress_pass") is True
            and get(data["formal_next_collection_step"], "has_next_step") is True
            and get(data["formal_next_collection_step"], "next_row.task")
            == get(data["formal_collection_progress"], "next_row.task")
            and get(data["formal_next_collection_step"], "next_row.arm")
            == get(data["formal_collection_progress"], "next_row.arm")
            and get(data["formal_next_collection_step"], "next_row.pair_id")
            == get(data["formal_collection_progress"], "next_row.pair_id")
            and get(data["formal_next_collection_step"], "recommended_path")
            == get(data["formal_collection_progress"], "next_row.recommended_path")
            and str(get(data["formal_next_collection_step"], "recommended_path", "")).endswith(".hdf5")
            and "serve_dp_tac_quality_guided"
            in str(get(data["formal_next_collection_step"], "launch_command", ""))
            and "--dry_run_guidance_smoke"
            in str(get(data["formal_next_collection_step"], "pre_collection_dry_run_command", ""))
            and "--smoke_output"
            in str(get(data["formal_next_collection_step"], "pre_collection_dry_run_command", ""))
            and str(get(data["formal_next_collection_step"], "pre_collection_dry_run_output", "")).endswith(
                "_smoke.json"
            )
            and "build_tac_quality_collection_progress.py"
            in str(get(data["formal_next_collection_step"], "post_run_commands", ""))
            and "finalize_tac_quality_collected_hdf5.py"
            in str(get(data["formal_next_collection_step"], "post_run_commands", ""))
            and "finalize_tac_quality_collected_hdf5.py"
            in str(get(data["formal_next_collection_step"], "finalize_command_template", "")),
            "evidence": {
                "next_step_pass": get(data["formal_next_collection_step"], "next_step_pass"),
                "has_next_step": get(data["formal_next_collection_step"], "has_next_step"),
                "next_row": get(data["formal_next_collection_step"], "next_row"),
                "recommended_path": get(data["formal_next_collection_step"], "recommended_path"),
                "recommended_path_exists": get(data["formal_next_collection_step"], "recommended_path_exists"),
                "hdf5_audit": get(data["formal_next_collection_step"], "hdf5_audit"),
                "pre_collection_dry_run_command": get(
                    data["formal_next_collection_step"], "pre_collection_dry_run_command"
                ),
                "pre_collection_dry_run_output": get(
                    data["formal_next_collection_step"], "pre_collection_dry_run_output"
                ),
                "finalize_command_template": get(data["formal_next_collection_step"], "finalize_command_template"),
                "finalize_newest_from_dir_template": get(
                    data["formal_next_collection_step"], "finalize_newest_from_dir_template"
                ),
                "post_run_commands": get(data["formal_next_collection_step"], "post_run_commands"),
            },
        },
        {
            "name": "formal_next_collection_step_pre_collection_smoke_pass",
            "passed": get(data["formal_next_collection_step_smoke"], "overall_pass") is True
            and get(data["formal_next_collection_step_smoke"], "scientific_evidence") is False
            and get(data["formal_next_collection_step_smoke"], "checks.smoke_pass") is True
            and get(data["formal_next_collection_step_smoke"], "checks.task_matches") is True
            and get(data["formal_next_collection_step_smoke"], "checks.arm_matches") is True
            and get(data["formal_next_collection_step_smoke"], "checks.not_reranking") is True
            and get(data["formal_next_collection_step_smoke"], "checks.final_clean_action_guidance") is True
            and get(data["formal_next_collection_step_smoke"], "checks.baseline_guidance_disabled") is True
            and get(data["formal_next_collection_step_smoke"], "checks.guided_grad_contract") is True,
            "evidence": {
                "overall_pass": get(data["formal_next_collection_step_smoke"], "overall_pass"),
                "pre_collection_dry_run_output": get(
                    data["formal_next_collection_step_smoke"], "pre_collection_dry_run_output"
                ),
                "checks": get(data["formal_next_collection_step_smoke"], "checks"),
                "smoke_summary": get(data["formal_next_collection_step_smoke"], "smoke_summary"),
            },
        },
        {
            "name": "formal_next_collection_step_smoke_runner_pass",
            "passed": get(data["formal_next_collection_step_smoke_runner"], "overall_pass") is True
            and get(data["formal_next_collection_step_smoke_runner"], "scientific_evidence") is False
            and get(data["formal_next_collection_step_smoke_runner"], "process.passed_process") is True
            and get(data["formal_next_collection_step_smoke_runner"], "process.returncode") == 0
            and get(data["formal_next_collection_step_smoke_runner"], "smoke_audit.overall_pass") is True
            and "--dry_run_guidance_smoke"
            in str(get(data["formal_next_collection_step_smoke_runner"], "pre_collection_dry_run_command", "")),
            "evidence": {
                "overall_pass": get(data["formal_next_collection_step_smoke_runner"], "overall_pass"),
                "process": get(data["formal_next_collection_step_smoke_runner"], "process"),
                "smoke_audit": get(data["formal_next_collection_step_smoke_runner"], "smoke_audit"),
            },
        },
        {
            "name": "formal_current_collection_gate_pass",
            "passed": get(data["formal_current_collection_gate"], "current_collection_gate_pass") is True
            and get(data["formal_current_collection_gate"], "scientific_evidence") is False
            and get(data["formal_current_collection_gate"], "operator_go_no_go") == "go"
            and get(data["formal_current_collection_gate"], "checks.runner_pass") is True
            and get(data["formal_current_collection_gate"], "checks.recommended_path_not_exists") is True
            and get(data["formal_current_collection_gate"], "checks.rollout_dir_writable") is True
            and get(data["formal_current_collection_gate"], "checks.same_current_row") is True,
            "evidence": {
                "gate_pass": get(data["formal_current_collection_gate"], "current_collection_gate_pass"),
                "operator_go_no_go": get(data["formal_current_collection_gate"], "operator_go_no_go"),
                "current_row": get(data["formal_current_collection_gate"], "current_row"),
                "checks": get(data["formal_current_collection_gate"], "checks"),
            },
        },
        {
            "name": "formal_current_collection_handoff_pass",
            "passed": get(data["formal_current_collection_handoff"], "handoff_pass") is True
            and get(data["formal_current_collection_handoff"], "scientific_evidence") is False
            and get(data["formal_current_collection_handoff"], "operator_go_no_go") == "go"
            and "TACQUALITY_RECOMMENDED_HDF5="
            in str(get(data["formal_current_collection_handoff"], "launch_command_with_save_path_hint", ""))
            and "finalize_tac_quality_collected_hdf5.py"
            in str(get(data["formal_current_collection_handoff"], "finalize_commands", ""))
            and "build_tac_quality_current_collection_gate.py"
            in str(get(data["formal_current_collection_handoff"], "post_finalize_commands", "")),
            "evidence": {
                "handoff_pass": get(data["formal_current_collection_handoff"], "handoff_pass"),
                "operator_go_no_go": get(data["formal_current_collection_handoff"], "operator_go_no_go"),
                "current_row": get(data["formal_current_collection_handoff"], "current_row"),
                "recommended_path": get(data["formal_current_collection_handoff"], "recommended_path"),
                "checks": get(data["formal_current_collection_handoff"], "checks"),
            },
        },
        {
            "name": "all_manifest_files_exist",
            "passed": not missing,
            "evidence": missing,
        },
        {
            "name": "runtime_contract_pass",
            "passed": bool(get(data["runtime_contract"], "passes_runtime_contract_sanity", False)),
            "evidence": {
                "insertion_action_grad": get(data["runtime_contract"], "insertion.action_grad_norm_mean"),
                "board_joint_grad": get(data["runtime_contract"], "board.joint_action_grad_norm_mean"),
                "board_eef_grad": get(data["runtime_contract"], "board.eef_action_grad_norm_mean"),
            },
        },
        {
            "name": "guidance_contract_audit_pass",
            "passed": bool(get(data["guidance_contract"], "guidance_contract_pass", False))
            and get(data["guidance_contract"], "scientific_evidence_complete") is False
            and get(data["guidance_contract"], "recommended_guidance_mode")
            == "final_clean_action_trust_region_refinement"
            and not (get(data["guidance_contract"], "failed_required_checks", []) or []),
            "evidence": {
                "guidance_contract_pass": get(data["guidance_contract"], "guidance_contract_pass"),
                "scientific_evidence_complete": get(
                    data["guidance_contract"], "scientific_evidence_complete"
                ),
                "recommended_guidance_mode": get(data["guidance_contract"], "recommended_guidance_mode"),
                "failed_required_checks": get(data["guidance_contract"], "failed_required_checks"),
                "interpretation": get(data["guidance_contract"], "deployment_interpretation"),
            },
        },
        {
            "name": "trust_region_guidance_pass",
            "passed": bool(get(data["trust_region_guidance"], "passes_trust_region_guidance_sanity", False)),
            "evidence": {
                "insertion_improved": get(data["trust_region_guidance"], "insertion.improved_rate"),
                "board_improved": get(data["trust_region_guidance"], "board.improved_rate"),
            },
        },
        {
            "name": "dp_guidance_controller_pass",
            "passed": bool(get(data["dp_guidance_controller"], "passes_controller_sanity", False))
            and bool(get(data["dp_guidance_controller_real_sample"], "overall_pass", False)),
            "evidence": {
                "sanity_pass": get(data["dp_guidance_controller"], "passes_controller_sanity"),
                "real_sample_pass": get(data["dp_guidance_controller_real_sample"], "overall_pass"),
                "insertion_improved": get(data["dp_guidance_controller_real_sample"], "insertion.report.improved_rate"),
                "board_improved": get(data["dp_guidance_controller_real_sample"], "board.report.improved_rate"),
                "stale_gradient_reuse_allowed": get(data["dp_guidance_controller"], "insertion.guardrails.stale_gradient_reuse_allowed"),
            },
        },
        {
            "name": "dp_integration_adapter_pass",
            "passed": bool(get(data["dp_integration_adapter"], "passes_integration_adapter_sanity", False))
            and get(data["dp_integration_adapter"], "insertion.improved_rate", 0.0) >= 0.95
            and get(data["dp_integration_adapter"], "board.improved_rate", 0.0) >= 0.95
            and get(data["dp_integration_adapter"], "minmax_insertion.improved_rate", 0.0) >= 0.95
            and get(data["dp_integration_adapter"], "normalizer_roundtrip.max_abs_error", 1.0) <= 1e-6,
            "evidence": {
                "pass": get(data["dp_integration_adapter"], "passes_integration_adapter_sanity"),
                "insertion_improved": get(data["dp_integration_adapter"], "insertion.improved_rate"),
                "board_improved": get(data["dp_integration_adapter"], "board.improved_rate"),
                "minmax_insertion_improved": get(data["dp_integration_adapter"], "minmax_insertion.improved_rate"),
                "normalizer_roundtrip": get(data["dp_integration_adapter"], "normalizer_roundtrip"),
                "contract": get(data["dp_integration_adapter"], "insertion.integration_contract"),
            },
        },
        {
            "name": "foresight_bridge_pass",
            "passed": bool(get(data["foresight_bridge"], "passes_foresight_bridge_sanity", False))
            and get(data["foresight_bridge"], "not_reranking") is True
            and get(data["foresight_bridge"], "not_every_step_ddpm_guidance") is True
            and get(data["foresight_bridge"], "insertion.bridge_grad.positive_grad_rate", 0.0) >= 0.999
            and get(data["foresight_bridge"], "board.bridge_grad.positive_grad_rate", 0.0) >= 0.999
            and get(data["foresight_bridge"], "insertion.adapter_report.improved_rate", 0.0) >= 0.90
            and get(data["foresight_bridge"], "board.adapter_report.improved_rate", 0.0) >= 0.90,
            "evidence": {
                "pass": get(data["foresight_bridge"], "passes_foresight_bridge_sanity"),
                "guidance_mode": get(data["foresight_bridge"], "guidance_mode"),
                "not_reranking": get(data["foresight_bridge"], "not_reranking"),
                "not_every_step_ddpm_guidance": get(data["foresight_bridge"], "not_every_step_ddpm_guidance"),
                "insertion_shapes": get(data["foresight_bridge"], "insertion.tactile_shapes"),
                "board_shapes": get(data["foresight_bridge"], "board.tactile_shapes"),
                "insertion_improved": get(data["foresight_bridge"], "insertion.adapter_report.improved_rate"),
                "board_improved": get(data["foresight_bridge"], "board.adapter_report.improved_rate"),
            },
        },
        {
            "name": "score_landscape_pass",
            "passed": bool(get(data["score_landscape"], "overall_pass", False))
            and bool(get(data["score_landscape"], "insertion.passes_score_landscape", False))
            and bool(get(data["score_landscape"], "board.passes_score_landscape", False)),
            "evidence": {
                "overall_pass": get(data["score_landscape"], "overall_pass"),
                "insertion_grad_norm_mean": get(data["score_landscape"], "insertion.gradient.grad_norm.mean"),
                "board_grad_norm_mean": get(data["score_landscape"], "board.gradient.grad_norm.mean"),
                "insertion_thresholds": get(data["score_landscape"], "insertion.pass_thresholds"),
                "board_thresholds": get(data["score_landscape"], "board.pass_thresholds"),
            },
        },
        {
            "name": "runtime_visualization_pass",
            "passed": bool(get(data["runtime_visualization"], "visualization_pass", False))
            and get(data["runtime_visualization"], "figures.runtime_pca_task_score_grad") is not None
            and get(data["runtime_visualization"], "figures.runtime_score_grad_distributions") is not None,
            "evidence": {
                "visualization_pass": get(data["runtime_visualization"], "visualization_pass"),
                "figures": get(data["runtime_visualization"], "figures"),
                "diagnostics": get(data["runtime_visualization"], "diagnostics"),
            },
        },
        {
            "name": "score_calibration_pass",
            "passed": get(data["score_calibration"], "recommendation.insertion") == "energy"
            and get(data["score_calibration"], "recommendation.board") == "quality",
            "evidence": get(data["score_calibration"], "recommendation"),
        },
        {
            "name": "offline_gate_pass",
            "passed": bool(get(data["offline_gate"], "offline_production_gate_pass", False)),
            "evidence": get(data["offline_gate"], "remaining_required_step"),
        },
        {
            "name": "evidence_summary_keeps_real_robot_gap",
            "passed": (
                get(data["evidence_summary"], "completion_assessment.objective_complete", get(data["evidence_summary"], "objective_complete"))
                is False
            )
            and any(
                token
                in str(
                    get(
                        data["evidence_summary"],
                        "completion_assessment.reason",
                        get(data["evidence_summary"], "reason", ""),
                    )
                ).lower()
                for token in ["real-robot", "real rollout", "real-rollout", "acceptance protocol"]
            ),
            "evidence": get(data["evidence_summary"], "completion_assessment", data["evidence_summary"]),
        },
        {
            "name": "goal_completion_audit_keeps_real_rollout_gap",
            "passed": get(data["goal_completion_audit"], "objective_complete") is False
            and get(data["goal_completion_audit"], "status") == "incomplete"
            and get(data["goal_completion_audit"], "next_required_step") is not None,
            "evidence": {
                "objective_complete": get(data["goal_completion_audit"], "objective_complete"),
                "status": get(data["goal_completion_audit"], "status"),
                "n_blockers": len(get(data["goal_completion_audit"], "blockers", []) or []),
                "next_required_step": get(data["goal_completion_audit"], "next_required_step"),
            },
        },
    ]

    manifest = {
        "name": "TacQualityEnergy DP classifier guidance manifest",
        "git_commit": git_commit(),
        "scope": (
            "Offline-ready scorer/guidance package for socket insertion and board wiping. "
            "This manifest does not claim real-robot validation."
        ),
        "deployment_policy": {
            "recommended_mode": "final_clean_action_trust_region_refinement",
            "allowed_for_robot_dry_run": [
                "Run the original DP denoising sampler first.",
                "Predict tactile consequence with the current Foresight model.",
                "Score with task-conditioned TacQualityEnergy.",
                "Apply bounded accept-only refinement on the clean/final action.",
                "Recompute action -> Foresight -> TacQualityEnergy gradient for every accepted update.",
            ],
            "research_only_modes": [
                "late_step_denoising_controller_guidance",
                "controller_in_every_ddpm_step",
            ],
            "not_recommended_yet": [
                "Unconditional guidance at every DDPM denoising step.",
                "Large guidance scale without trust-region projection.",
                "Using p_good/log_p_good alone as the guidance potential.",
                "Reusing cached or stale gradients across denoising steps.",
                "Using a single cross-task probability threshold as the final quality decision.",
            ],
            "reason": (
                "Offline scorer quality, full-chain gradients, clean-action refinement, "
                "board heldout chain, and production gate pass.  However controller-in-denoising "
                "diagnostics show local step score gains do not reliably translate into final "
                "denoised sample improvement, so final/clean-action refinement is the current "
                "safe deployment mode."
            ),
            "completion_status": "offline_ready_not_robot_validated",
        },
        "score_api": {
            "runtime": "TFAC_V5.tac_quality_guidance_runtime.TacQualityGuidanceRuntime",
            "score_call": "runtime.score(task, predicted_tactile, action, mode='profile')",
            "refiner": "TFAC_V5.tac_quality_trust_region_guidance.TacQualityTrustRegionRefiner",
            "refine_call": "refiner.refine(action, score_fn)",
            "dp_controller": "TFAC_V5.tac_quality_dp_guidance_controller.TacQualityDPGuidanceController",
            "dp_controller_call": "guided_action, report = controller.guide(action, current_score_fn)",
            "serving_helper": "TFAC_V5.tac_quality_serving_guidance.TacQualityServingGuidance",
            "serving_helper_call": "guided_action_norm, report = helper.guide_action_chunk(action_norm, bridge)",
            "serving_packet": (
                "python TFAC_V5/build_tac_quality_serving_packet.py "
                "--dp_ckpt_dir <dp_ckpt_dir> --foresight_dir <foresight_dir> "
                "--foresight_ckpt <foresight_ckpt>"
            ),
            "foresight_bridge": "TFAC_V5.tac_quality_foresight_bridge.ForesightTacQualityBridge",
            "foresight_bridge_call": "tactile = bridge(action_raw)",
            "guardrail": "Do not pass cached gradients; current_score_fn must recompute action -> Foresight -> TacQuality score each guidance step.",
            "real_rollout_gate": (
                "python TFAC_V5/eval_real_rollout_quality_gate.py "
                "--task {insertion,board} --baseline_dir <baseline_hdf5_dir> "
                "--guided_dir <guided_hdf5_dir>"
            ),
            "real_rollout_validation_prep": (
                "python TFAC_V5/prepare_real_rollout_validation.py "
                "--task {insertion,board} --baseline_dir <baseline_hdf5_dir> "
                "--guided_dir <guided_hdf5_dir>"
            ),
            "real_rollout_sample_size_plan": (
                "python TFAC_V5/plan_real_rollout_sample_size.py "
                "--task {insertion,board}"
            ),
            "real_rollout_experiment_packet": (
                "python TFAC_V5/build_real_rollout_experiment_packet.py --tag formal_paired12"
            ),
            "real_rollout_scorer_ablation_gate": (
                "python TFAC_V5/eval_real_rollout_scorer_ablation_gate.py "
                "--task {insertion,board} --baseline_dir <baseline_hdf5_dir> "
                "--default_guided_dir <default_guided_hdf5_dir> "
                "--distilled_guided_dir <distilled_guided_hdf5_dir>"
            ),
        },
        "tasks": {
            "insertion": {
                "scorer": "InsertionRiskScorerRuntime",
                "checkpoint": file_info(PATHS["insertion_scorer_ckpt"]),
                "profile_energy": "0.50*quality_logit + 0.10*binary_margin",
                "calibration_mode": get(data["score_calibration"], "recommendation.insertion"),
                "trust_region": {
                    "steps": get(data["trust_region_guidance"], "insertion.config.steps"),
                    "step_size": get(data["trust_region_guidance"], "insertion.config.step_size"),
                    "max_total_delta": get(data["trust_region_guidance"], "insertion.config.max_total_delta"),
                    "accept_only_improved": get(data["trust_region_guidance"], "insertion.config.accept_only_improved"),
                },
            },
            "board": {
                "scorer": "PTGProxyScorerV2Runtime",
                "checkpoint": file_info(PATHS["board_scorer_ckpt"]),
                "ablation_candidate": {
                    "scorer": "DistilledTacQualityEnergyRuntime",
                    "checkpoint": file_info(PATHS["distilled_energy_ckpt"]),
                    "status": get(
                        data["scorer_selection_gate"],
                        "selection.distilled_replacement_status",
                        "not_evaluated",
                    ),
                    "selection_gate": str(PATHS["scorer_selection_gate"]),
                },
                "profile_energy": "0.75*quality_logit + 0.10*binary_margin",
                "calibration_mode": get(data["score_calibration"], "recommendation.board"),
                "trust_region": {
                    "steps": get(data["trust_region_guidance"], "board.config.steps"),
                    "step_size": get(data["trust_region_guidance"], "board.config.step_size"),
                    "max_total_delta": get(data["trust_region_guidance"], "board.config.max_total_delta"),
                    "accept_only_improved": get(data["trust_region_guidance"], "board.config.accept_only_improved"),
                },
            },
        },
        "modules": {name: file_info(path) for name, path in MODULES.items()},
        "evidence_files": {name: file_info(path) for name, path in PATHS.items()},
        "future_rollout_outputs": {
            name: file_info(PATHS[name])
            for name in sorted(FUTURE_ROLLOUT_OUTPUTS)
        },
        "checks": checks,
        "deployment_manifest_pass": all(item["passed"] for item in checks),
        "remaining_required_step": "Real robot / final production policy validation.",
    }
    return manifest


def write_markdown(manifest: Dict[str, Any], path: Path) -> None:
    lines = [
        "# TacQuality Guidance Deployment Manifest",
        "",
        f"- deployment_manifest_pass: `{manifest['deployment_manifest_pass']}`",
        f"- git_commit: `{manifest['git_commit']}`",
        f"- scope: {manifest['scope']}",
        f"- remaining_required_step: {manifest['remaining_required_step']}",
        "",
        "## Deployment Policy",
        "",
        f"- recommended_mode: `{manifest['deployment_policy']['recommended_mode']}`",
        f"- completion_status: `{manifest['deployment_policy']['completion_status']}`",
        f"- reason: {manifest['deployment_policy']['reason']}",
        "",
        "### Allowed Robot Dry-Run Flow",
        "",
    ]
    for item in manifest["deployment_policy"]["allowed_for_robot_dry_run"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "### Research-Only Modes",
            "",
        ]
    )
    for item in manifest["deployment_policy"]["research_only_modes"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "### Not Recommended Yet",
            "",
        ]
    )
    for item in manifest["deployment_policy"]["not_recommended_yet"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## API",
            "",
            f"- score: `{manifest['score_api']['score_call']}`",
            f"- refine: `{manifest['score_api']['refine_call']}`",
            f"- real rollout gate: `{manifest['score_api']['real_rollout_gate']}`",
            "",
            "## Tasks",
            "",
        ]
    )
    for task, info in manifest["tasks"].items():
        lines.extend(
            [
                f"### {task}",
                "",
                f"- scorer: `{info['scorer']}`",
                f"- checkpoint: `{info['checkpoint']['path']}`",
                f"- profile_energy: `{info['profile_energy']}`",
                f"- calibration_mode: `{info['calibration_mode']}`",
                f"- trust_region: `{json.dumps(info['trust_region'], ensure_ascii=False)}`",
                "",
            ]
        )
    lines.extend(["## Checks", ""])
    for item in manifest["checks"]:
        status = "PASS" if item["passed"] else "FAIL"
        lines.append(f"- **{status}** {item['name']}: `{json.dumps(item['evidence'], ensure_ascii=False)}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default=str(OUT_DIR))
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest()
    json_path = out_dir / "tac_quality_guidance_manifest.json"
    md_path = out_dir / "tac_quality_guidance_manifest.md"
    json_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(manifest, md_path)
    print(json.dumps({
        "deployment_manifest_pass": manifest["deployment_manifest_pass"],
        "remaining_required_step": manifest["remaining_required_step"],
        "json": str(json_path),
        "markdown": str(md_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
