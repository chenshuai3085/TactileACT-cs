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
    "ptg_proxy_eval": Path("/home/chenshuai/Project/output/ptg_proxy_scorer_v2/ptg_proxy_scorer_v2_eval.json"),
    "score_calibration": Path("/home/chenshuai/Project/output/tac_quality_score_calibration/tac_quality_score_calibration.json"),
    "scale_sweep": Path("/home/chenshuai/Project/output/tac_quality_guidance_scale_sweep/tac_quality_guidance_scale_sweep.json"),
    "robustness": Path("/home/chenshuai/Project/output/tac_quality_guidance_robustness/tac_quality_guidance_robustness.json"),
    "dp_integration_adapter": Path(
        "/home/chenshuai/Project/output/tac_quality_dp_integration_adapter/integration_adapter_sanity.json"
    ),
    "foresight_bridge": Path(
        "/home/chenshuai/Project/output/tac_quality_foresight_bridge/foresight_bridge_sanity.json"
    ),
    "score_landscape": Path("/home/chenshuai/Project/output/tac_quality_score_landscape/tac_quality_score_landscape.json"),
    "runtime_visualization": Path(
        "/home/chenshuai/Project/output/tac_quality_runtime_visualization/tac_quality_runtime_visualization.json"
    ),
    "offline_gate": Path("/home/chenshuai/Project/output/ptg_offline_production_gate/ptg_offline_production_gate.json"),
    "scorer_selection_gate": Path(
        "/home/chenshuai/Project/output/tac_quality_scorer_selection_gate/tac_quality_scorer_selection_gate.json"
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


def build_audit(paths: Dict[str, Path]) -> Dict[str, Any]:
    data = {name: load_json(path) for name, path in paths.items() if path.suffix == ".json"}

    insertion = data["insertion_eval"]
    board = data["board_scheme_eval"]
    board_calibration = data["board_target_force_calibration"]
    ptg = data["ptg_proxy_eval"]
    scale = data["scale_sweep"]
    robust = data["robustness"]
    dp_integration_adapter = data["dp_integration_adapter"]
    foresight_bridge = data["foresight_bridge"]
    score_landscape = data["score_landscape"]
    runtime_visualization = data["runtime_visualization"]
    offline = data["offline_gate"]
    selection_gate = data["scorer_selection_gate"]
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
            else "incomplete",
            "selection_gate_pass="
            f"{get(selection_gate, 'selection_gate_pass')}; "
            f"default={get(selection_gate, 'selection.current_default_board_scorer')}; "
            f"candidate={get(selection_gate, 'selection.promoted_ablation_candidate')}; "
            f"replacement_status={get(selection_gate, 'selection.distilled_replacement_status')}",
            str(paths["scorer_selection_gate"]),
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
            and "real-robot" in str(get(summary, "completion_assessment.reason", get(summary, "reason", "")))
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
            else "incomplete",
            "launch_sheet_ready="
            f"{get(formal_launch_sheet, 'launch_sheet_ready')}; "
            f"rollout_root={get(formal_launch_sheet, 'rollout_root')}; "
            f"insertion_ports={get(formal_launch_sheet, 'tasks.insertion.ports')}; "
            f"board_ports={get(formal_launch_sheet, 'tasks.board.ports')}; "
            "baseline_smoke="
            f"{get(load_json(paths['guided_server_insertion_baseline_no_guidance_smoke']), 'dry_run_guidance_smoke_pass')}/"
            f"{get(load_json(paths['guided_server_board_baseline_no_guidance_smoke']), 'dry_run_guidance_smoke_pass')}",
            str(paths["formal_launch_sheet"]),
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
                "Three-arm rollout policy/scorer configs are machine-readable and complete.",
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
                else "incomplete",
                "pass="
                f"{get(rollout_arm_configs, 'rollout_arm_config_pass')}; "
                f"insertion_default={get(rollout_arm_configs, 'tasks.insertion.default_guided.scorer_runtime')}; "
                f"board_default={get(rollout_arm_configs, 'tasks.board.default_guided.scorer_runtime')}; "
                f"candidate={get(rollout_arm_configs, 'selection_summary.promoted_ablation_candidate')}",
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
                and get(load_json(paths["guided_server_insertion_real_foresight_smoke"]), "dry_run_guidance_smoke_pass") is True
                and get(load_json(paths["guided_server_board_real_foresight_smoke"]), "dry_run_guidance_smoke_pass") is True
                else "incomplete",
                "launch_packet_ready="
                f"{get(guided_server_packet, 'launch_packet_ready')}; "
                f"guided_server_ready={get(guided_server_packet, 'guided_server_ready')}; "
                f"next_step={get(guided_server_packet, 'next_step')}; "
                "real_foresight_smoke="
                f"{get(load_json(paths['guided_server_insertion_real_foresight_smoke']), 'dry_run_guidance_smoke_pass')}/"
                f"{get(load_json(paths['guided_server_board_real_foresight_smoke']), 'dry_run_guidance_smoke_pass')}",
                str(paths["guided_server_packet"]),
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
                and get(ablation_ins, "recommended_real_scorer") is not None
                else ("missing" if ablation_ins is None else "incomplete"),
                "production_ablation_pass="
                f"{get(ablation_ins, 'production_ablation_pass')}; "
                f"debug_or_underpowered={get(ablation_ins, 'debug_or_underpowered')}; "
                f"recommended_real_scorer={get(ablation_ins, 'recommended_real_scorer')}",
                str(paths["scorer_ablation_insertion"]),
            ),
            item(
                "Formal board three-arm scorer ablation identifies the best real guided scorer.",
                "satisfied"
                if get(ablation_board, "production_ablation_pass") is True
                and get(ablation_board, "debug_or_underpowered") is False
                and get(ablation_board, "recommended_real_scorer") is not None
                else ("missing" if ablation_board is None else "incomplete"),
                "production_ablation_pass="
                f"{get(ablation_board, 'production_ablation_pass')}; "
                f"debug_or_underpowered={get(ablation_board, 'debug_or_underpowered')}; "
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
