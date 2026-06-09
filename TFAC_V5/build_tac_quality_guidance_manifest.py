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
    "scorer_selection_gate": Path(
        "/home/chenshuai/Project/output/tac_quality_scorer_selection_gate/tac_quality_scorer_selection_gate.json"
    ),
    "runtime_contract": Path("/home/chenshuai/Project/output/tac_quality_guidance_runtime/runtime_contract_sanity.json"),
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
    "goal_completion_audit": Path("/home/chenshuai/Project/output/tac_quality_goal_audit/tac_quality_goal_completion_audit.json"),
}


MODULES = {
    "guidance_config": Path("TFAC_V5/tac_quality_guidance_config.py"),
    "guidance_runtime": Path("TFAC_V5/tac_quality_guidance_runtime.py"),
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
    "formal_launch_sheet": Path("TFAC_V5/build_tac_quality_formal_launch_sheet.py"),
    "formal_launch_sheet_smoke": Path("TFAC_V5/smoke_tac_quality_formal_launch_sheet.py"),
    "board_target_force_calibration": Path("TFAC_V5/calibrate_board_target_force.py"),
    "rollout_arm_configs": Path("TFAC_V5/build_tac_quality_rollout_arm_configs.py"),
    "rollout_arm_config_smoke": Path("TFAC_V5/smoke_tac_quality_rollout_arm_configs.py"),
    "deployment_bridge_smoke": Path("TFAC_V5/smoke_tac_quality_deployment_bridge.py"),
    "scorer_selection_gate": Path("TFAC_V5/build_tac_quality_scorer_selection_gate.py"),
    "score_landscape": Path("TFAC_V5/eval_tac_quality_score_landscape.py"),
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
            == "DistilledTacQualityEnergyRuntime",
            "evidence": {
                "pass": get(data["rollout_arm_configs"], "rollout_arm_config_pass"),
                "selection_summary": get(data["rollout_arm_configs"], "selection_summary"),
            },
        },
        {
            "name": "rollout_arm_config_gradient_smoke_pass",
            "passed": bool(get(data["rollout_arm_config_smoke"], "overall_pass", False))
            and get(data["rollout_arm_config_smoke"], "scientific_evidence") is False
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
            and get(data["guided_server_insertion_real_foresight_smoke"], "dry_run_guidance_smoke_pass") is True
            and get(data["guided_server_board_real_foresight_smoke"], "dry_run_guidance_smoke_pass") is True,
            "evidence": {
                "launch_packet_ready": get(data["guided_server_packet"], "launch_packet_ready"),
                "guided_server_ready": get(data["guided_server_packet"], "guided_server_ready"),
                "next_step": get(data["guided_server_packet"], "next_step"),
                "tasks": get(data["guided_server_packet"], "tasks"),
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
            "name": "scorer_selection_gate_pass",
            "passed": bool(get(data["scorer_selection_gate"], "selection_gate_pass", False))
            and get(data["scorer_selection_gate"], "selection.current_default_board_scorer")
            == "PTGProxyScorerV2Runtime"
            and get(data["scorer_selection_gate"], "selection.promoted_ablation_candidate")
            == "DistilledTacQualityEnergyRuntime"
            and get(data["scorer_selection_gate"], "selection.distilled_replacement_status") == "not_yet_replacement",
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
            and get(data["formal_rollout_gate_runner"], "run_skip_reason") is not None,
            "evidence": {
                "preflight_ready": get(data["formal_rollout_gate_runner"], "preflight_ready"),
                "scientific_evidence": get(data["formal_rollout_gate_runner"], "scientific_evidence"),
                "tasks": get(data["formal_rollout_gate_runner"], "tasks"),
                "run_skip_reason": get(data["formal_rollout_gate_runner"], "run_skip_reason"),
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
            and "serve_dp_tac_quality_guided" in str(get(data["formal_launch_sheet"], "tasks.board.launch_commands.baseline", ""))
            and "--disable_guidance" in str(get(data["formal_launch_sheet"], "tasks.board.launch_commands.baseline", ""))
            and "serve_dp_tac_quality_guided" in str(get(data["formal_launch_sheet"], "tasks.board.launch_commands.default_guided", ""))
            and "serve_dp_tac_quality_guided" in str(get(data["formal_launch_sheet"], "tasks.board.launch_commands.distilled_guided", ""))
            and get(data["guided_server_insertion_baseline_no_guidance_smoke"], "dry_run_guidance_smoke_pass") is True
            and get(data["guided_server_board_baseline_no_guidance_smoke"], "dry_run_guidance_smoke_pass") is True
            and "run_formal_tac_quality_rollout_gates.py" in str(get(data["formal_launch_sheet"], "all_tasks_gate_runner_command", "")),
            "evidence": {
                "launch_sheet_ready": get(data["formal_launch_sheet"], "launch_sheet_ready"),
                "rollout_root": get(data["formal_launch_sheet"], "rollout_root"),
                "insertion_ports": get(data["formal_launch_sheet"], "tasks.insertion.ports"),
                "board_ports": get(data["formal_launch_sheet"], "tasks.board.ports"),
                "insertion_baseline_no_guidance_smoke": get(data["guided_server_insertion_baseline_no_guidance_smoke"], "dry_run_guidance_smoke_pass"),
                "board_baseline_no_guidance_smoke": get(data["guided_server_board_baseline_no_guidance_smoke"], "dry_run_guidance_smoke_pass"),
                "all_tasks_gate_runner_command": get(data["formal_launch_sheet"], "all_tasks_gate_runner_command"),
            },
        },
        {
            "name": "formal_launch_sheet_command_smoke_pass",
            "passed": bool(get(data["formal_launch_sheet_smoke"], "overall_pass", False))
            and get(data["formal_launch_sheet_smoke"], "scientific_evidence") is False
            and get(data["formal_launch_sheet_smoke"], "checks.six_commands_present") is True
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
            and "real-robot" in str(
                get(data["evidence_summary"], "completion_assessment.reason", get(data["evidence_summary"], "reason", ""))
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
