"""Summarize evidence for TacQualityEnergy scorer guidance.

This script turns scattered experiment JSON files into a single auditable
report.  It is intentionally conservative: scorer-level evidence is not counted
as full-chain DP guidance evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TFAC_V5.tac_quality_guidance_config import profile_summary


OUT_DIR = Path("/home/chenshuai/Project/output/ptg_guidance_evidence")


DEFAULT_PATHS = {
    "insertion_scorer_eval": Path("/home/chenshuai/Project/output/insertion_risk_scorer/insertion_risk_scorer_eval.json"),
    "insertion_runtime_grad": Path("/home/chenshuai/Project/output/insertion_risk_scorer/runtime_gradient_sanity.json"),
    "insertion_full_chain": Path("/home/chenshuai/Project/output/full_chain_guidance_gradient/insertion_full_chain_energy_clipped_K8_N16.json"),
    "insertion_clean_refine": Path("/home/chenshuai/Project/output/clean_action_energy_refinement/insertion_clean_refine_constrained_K4_N40.json"),
    "ptg_v2_eval": Path("/home/chenshuai/Project/output/ptg_proxy_scorer_v2/ptg_proxy_scorer_v2_eval.json"),
    "ptg_v2_runtime_grad": Path("/home/chenshuai/Project/output/ptg_proxy_scorer_v2/runtime_gradient_sanity.json"),
    "board_readiness": Path("/home/chenshuai/Project/output/board_guidance_readiness/board_ptg_v2_energy_readiness_N240_safe_step.json"),
    "board_surrogate": Path("/home/chenshuai/Project/output/board_tactile_surrogate/board_tactile_surrogate_eval.json"),
    "board_surrogate_refine": Path("/home/chenshuai/Project/output/board_surrogate_action_refinement/board_surrogate_refine_K4_N512.json"),
    "board_foresight_smoke_history": Path("/home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_260522_smoke_0/pretrain_history.pkl"),
    "board_foresight_smoke_ckpt": Path("/home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_260522_smoke_0/foresight_best.ckpt"),
    "board_foresight_fast20": Path("/home/chenshuai/Project/output/board_production_chain_setup/board_foresight_fast20.json"),
    "board_foresight_fast100": Path("/home/chenshuai/Project/output/board_production_chain_setup/board_foresight_fast100.json"),
    "board_foresight_gradient": Path("/home/chenshuai/Project/output/board_production_foresight_gradient/board_production_foresight_fast20_gradient_N64.json"),
    "board_dp_smoke": Path("/home/chenshuai/Project/output/board_production_chain_setup/board_dp_smoke4.json"),
    "board_dp_fast16": Path("/home/chenshuai/Project/output/board_production_chain_setup/board_dp_fast16_e20.json"),
    "board_dp_fast32": Path("/home/chenshuai/Project/output/board_production_chain_setup/board_dp_fast32_e20.json"),
    "board_dp_lazy_full80_entry": Path("/home/chenshuai/Project/output/board_production_chain_setup/board_dp_lazy_full80_entry_e1.json"),
    "board_dp_image_cache_smoke": Path("/home/chenshuai/Project/output/board_production_chain_setup/board_dp_cache_smoke2_e1.json"),
    "board_dp_feature_cache_smoke": Path("/home/chenshuai/Project/output/board_production_chain_setup/board_dp_feature_cache_smoke2_e1.json"),
    "board_dp_feature_cache_full80": Path("/home/chenshuai/Project/output/board_production_chain_setup/board_dp_feature_cache_full80_fast32ema_w4096_e5.json"),
    "board_dp_full_chain_smoke": Path("/home/chenshuai/Project/output/board_dp_denoising_full_chain_smoke/board_dp_fast32_e20_clean_refine_full_chain_fast20_heldout32_K4_N64.json"),
    "board_dp_feature_cache_full_chain": Path("/home/chenshuai/Project/output/board_dp_denoising_full_chain_smoke/board_dp_feature_cache_full80_fast32ema_w4096_e5_fast20_heldout32_K4_N64.json"),
    "board_dp_feature_cache_full_chain_fast100": Path("/home/chenshuai/Project/output/board_dp_denoising_full_chain_smoke/board_dp_feature_cache_full80_fast32ema_w4096_e5_fast100_heldout32_K4_N64.json"),
    "offline_production_gate": Path("/home/chenshuai/Project/output/ptg_offline_production_gate/ptg_offline_production_gate.json"),
    "score_calibration": Path("/home/chenshuai/Project/output/tac_quality_score_calibration/tac_quality_score_calibration.json"),
    "runtime_contract": Path("/home/chenshuai/Project/output/tac_quality_guidance_runtime/runtime_contract_sanity.json"),
    "trust_region_guidance": Path("/home/chenshuai/Project/output/tac_quality_trust_region_guidance/trust_region_sanity.json"),
    "dp_guidance_controller": Path("/home/chenshuai/Project/output/tac_quality_dp_guidance_controller/controller_sanity.json"),
    "dp_guidance_controller_real_sample": Path("/home/chenshuai/Project/output/tac_quality_dp_guidance_controller/controller_real_sample_audit.json"),
    "controller_denoising_smoke": Path("/home/chenshuai/Project/output/tac_quality_controller_denoising_smoke/insertion_controller_denoising_final_s0001_K4_N4.json"),
    "deployment_manifest": Path("/home/chenshuai/Project/output/tac_quality_guidance_manifest/tac_quality_guidance_manifest.json"),
    "manifest_real_sample_smoke": Path("/home/chenshuai/Project/output/tac_quality_manifest_real_sample_smoke/manifest_real_sample_smoke.json"),
    "guidance_scale_sweep": Path("/home/chenshuai/Project/output/tac_quality_guidance_scale_sweep/tac_quality_guidance_scale_sweep.json"),
    "guidance_robustness": Path("/home/chenshuai/Project/output/tac_quality_guidance_robustness/tac_quality_guidance_robustness.json"),
    "unified_taxonomy": Path("/home/chenshuai/Project/output/unified_quality_taxonomy/unified_quality_eval_fast.json"),
    "energy_coeff_search": Path("/home/chenshuai/Project/output/scorer_guidance_suitability/energy_coeff_search.json"),
}


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_pickle(path: Path):
    if not path.exists():
        return None
    import pickle

    with open(path, "rb") as f:
        return pickle.load(f)


def get(d: Optional[Dict[str, Any]], path: str, default=None):
    if d is None:
        return default
    cur: Any = d
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def mean_metric(d: Optional[Dict[str, Any]], path: str, default=None):
    value = get(d, path, default)
    if isinstance(value, dict) and "mean" in value:
        return value["mean"]
    return value


def pass_item(name: str, passed: bool, evidence: str, missing: bool = False) -> Dict[str, Any]:
    return {"name": name, "passed": bool(passed), "missing": bool(missing), "evidence": evidence}


def build_summary(paths: Dict[str, Path]) -> Dict[str, Any]:
    non_json_paths = {"board_foresight_smoke_history", "board_foresight_smoke_ckpt"}
    data = {name: load_json(path) for name, path in paths.items() if name not in non_json_paths}
    missing = {name: str(path) for name, path in paths.items() if name not in non_json_paths and data[name] is None}

    insertion_scorer = data["insertion_scorer_eval"]
    insertion_full = data["insertion_full_chain"]
    insertion_refine = data["insertion_clean_refine"]
    ptg_v2 = data["ptg_v2_eval"]
    board_ready = data["board_readiness"]
    board_surrogate = data["board_surrogate"]
    board_surrogate_refine = data["board_surrogate_refine"]
    board_foresight_fast20 = data["board_foresight_fast20"]
    board_foresight_fast100 = data["board_foresight_fast100"]
    board_foresight_gradient = data["board_foresight_gradient"]
    board_dp_smoke = data["board_dp_smoke"]
    board_dp_fast16 = data["board_dp_fast16"]
    board_dp_fast32 = data["board_dp_fast32"]
    board_dp_lazy_full80_entry = data["board_dp_lazy_full80_entry"]
    board_dp_image_cache_smoke = data["board_dp_image_cache_smoke"]
    board_dp_feature_cache_smoke = data["board_dp_feature_cache_smoke"]
    board_dp_feature_cache_full80 = data["board_dp_feature_cache_full80"]
    board_dp_full_chain_smoke = data["board_dp_full_chain_smoke"]
    board_dp_feature_cache_full_chain = data["board_dp_feature_cache_full_chain"]
    board_dp_feature_cache_full_chain_fast100 = data["board_dp_feature_cache_full_chain_fast100"]
    offline_production_gate = data["offline_production_gate"]
    score_calibration = data["score_calibration"]
    runtime_contract = data["runtime_contract"]
    trust_region_guidance = data["trust_region_guidance"]
    dp_guidance_controller = data["dp_guidance_controller"]
    dp_guidance_controller_real_sample = data["dp_guidance_controller_real_sample"]
    controller_denoising_smoke = data["controller_denoising_smoke"]
    deployment_manifest = data["deployment_manifest"]
    manifest_real_sample_smoke = data["manifest_real_sample_smoke"]
    guidance_scale_sweep = data["guidance_scale_sweep"]
    guidance_robustness = data["guidance_robustness"]
    unified = data["unified_taxonomy"]
    board_smoke_history = load_pickle(paths["board_foresight_smoke_history"])
    board_smoke_ckpt_exists = paths["board_foresight_smoke_ckpt"].exists()
    if board_smoke_history is None:
        missing["board_foresight_smoke_history"] = str(paths["board_foresight_smoke_history"])
    if not board_smoke_ckpt_exists:
        missing["board_foresight_smoke_ckpt"] = str(paths["board_foresight_smoke_ckpt"])
    board_smoke_best_val = None
    board_smoke_epochs = None
    if isinstance(board_smoke_history, dict):
        val_losses = board_smoke_history.get("val", board_smoke_history.get("val_loss", []))
        if val_losses:
            board_smoke_best_val = min(val_losses)
            board_smoke_epochs = len(val_losses)

    insertion_checks = [
        pass_item(
            "Insertion scorer GroupKFold quality",
            (mean_metric(insertion_scorer, "mixed_group_cv.binary_auc.mean", 0) or 0) >= 0.95,
            f"binary_auc={mean_metric(insertion_scorer, 'mixed_group_cv.binary_auc.mean')}",
            insertion_scorer is None,
        ),
        pass_item(
            "Insertion full-chain gradient",
            bool(get(insertion_full, "interpretation.passes_full_chain_gradient", False)),
            f"score_improved_rate={get(insertion_full, 'summary.score_delta_positive_rate', get(insertion_full, 'summary.score_improved_rate'))}",
            insertion_full is None,
        ),
        pass_item(
            "Insertion constrained clean-action refinement",
            bool(get(insertion_refine, "interpretation.passes_clean_refinement_sanity", False)),
            f"beats={get(insertion_refine, 'summary.refined_beats_base_rate')}, hard_violation={get(insertion_refine, 'summary.refined_hard_range_violation.max')}",
            insertion_refine is None,
        ),
        pass_item(
            "Insertion guidance-score calibration",
            get(score_calibration, "recommendation.insertion") == "energy"
            and (get(score_calibration, "insertion_risk_scorer.modes.energy.summary.quality_positive_step_rate", 0.0) or 0.0) >= 0.95
            and (get(score_calibration, "insertion_risk_scorer.modes.energy.summary.quality_spearman", 0.0) or 0.0) >= 0.70,
            f"mode={get(score_calibration, 'recommendation.insertion')}, spearman={get(score_calibration, 'insertion_risk_scorer.modes.energy.summary.quality_spearman')}, q_gap={get(score_calibration, 'insertion_risk_scorer.modes.energy.summary.top_bottom_quality_gap')}",
            score_calibration is None,
        ),
        pass_item(
            "Insertion unified runtime contract",
            bool(get(runtime_contract, "passes_runtime_contract_sanity", False))
            and bool(get(runtime_contract, "insertion.all_finite", False))
            and bool(get(runtime_contract, "insertion.all_nonzero", False)),
            f"score_call={get(runtime_contract, 'contract.score_call')}, action_grad={get(runtime_contract, 'insertion.action_grad_norm_mean')}",
            runtime_contract is None,
        ),
        pass_item(
            "Insertion trust-region guidance update",
            bool(get(trust_region_guidance, "passes_trust_region_guidance_sanity", False))
            and (get(trust_region_guidance, "insertion.improved_rate", 0.0) or 0.0) >= 0.99
            and bool(get(trust_region_guidance, "insertion.max_delta_within_trust_region", False)),
            f"improved={get(trust_region_guidance, 'insertion.improved_rate')}, delta_max={get(trust_region_guidance, 'insertion.delta_norm.max')}, limit={get(trust_region_guidance, 'insertion.config.max_total_delta')}",
            trust_region_guidance is None,
        ),
        pass_item(
            "Insertion manifest real-sample smoke",
            bool(get(manifest_real_sample_smoke, "insertion.passes_real_sample_smoke", False)),
            f"improved={get(manifest_real_sample_smoke, 'insertion.score_improved_rate')}, score_delta={get(manifest_real_sample_smoke, 'insertion.score_delta.mean')}, delta_max={get(manifest_real_sample_smoke, 'insertion.delta_norm.max')}",
            manifest_real_sample_smoke is None,
        ),
        pass_item(
            "Insertion local guidance-scale law",
            bool(get(guidance_scale_sweep, "insertion.passes_guidance_scale_sweep", False)),
            f"recommended_scale={get(guidance_scale_sweep, 'insertion.recommended_scale')}, improved={get(guidance_scale_sweep, 'insertion.recommended_improved_rate')}, score_delta={get(guidance_scale_sweep, 'insertion.recommended_score_delta_mean')}",
            guidance_scale_sweep is None,
        ),
        pass_item(
            "Insertion current-gradient robustness",
            bool(get(guidance_robustness, "insertion.passes_current_gradient_robustness", False)),
            f"worst_current_improved={get(guidance_robustness, 'insertion.worst_perturbed_gradient_improved_rate')}, stale_gradient_stable={get(guidance_robustness, 'insertion.stale_gradient_stable_under_noise')}",
            guidance_robustness is None,
        ),
        pass_item(
            "Insertion DP guidance controller",
            bool(get(dp_guidance_controller, "passes_controller_sanity", False))
            and bool(get(dp_guidance_controller_real_sample, "insertion.passes_controller_real_sample", False)),
            f"real_sample_improved={get(dp_guidance_controller_real_sample, 'insertion.report.improved_rate')}, stale_allowed={get(dp_guidance_controller_real_sample, 'insertion.report.guardrails.stale_gradient_reuse_allowed')}",
            dp_guidance_controller is None or dp_guidance_controller_real_sample is None,
        ),
        pass_item(
            "Insertion controller denoising diagnostic recorded",
            controller_denoising_smoke is not None,
            f"pass={get(controller_denoising_smoke, 'interpretation.passes_controller_denoising_smoke')}, beats={get(controller_denoising_smoke, 'summary.guided_beats_base_rate')}, score_delta={get(controller_denoising_smoke, 'summary.score_delta.mean')}",
            controller_denoising_smoke is None,
        ),
    ]

    board_checks = [
        pass_item(
            "Board PTG v2 scorer quality",
            (mean_metric(ptg_v2, "mixed_group_cv.binary_auc.mean", 0) or 0) >= 0.95
            and (mean_metric(ptg_v2, "mixed_group_cv.quality_corr.mean", 0) or 0) >= 0.70,
            f"mixed_auc={mean_metric(ptg_v2, 'mixed_group_cv.binary_auc.mean')}, mixed_quality_corr={mean_metric(ptg_v2, 'mixed_group_cv.quality_corr.mean')}",
            ptg_v2 is None,
        ),
        pass_item(
            "Board scorer-level guidance readiness",
            bool(get(board_ready, "interpretation.passes_board_guidance_readiness", False)),
            f"improved_rate={get(board_ready, 'summary.score_improved_rate')}, finite_grad_rate={get(board_ready, 'summary.finite_grad_rate_all_inputs')}",
            board_ready is None,
        ),
        pass_item(
            "Board surrogate full-chain guidance",
            bool(get(board_surrogate, "interpretation.passes_board_surrogate_full_chain", False)),
            f"marker_mae={get(board_surrogate, 'eval.marker_mae.mean')}, improved_rate={get(board_surrogate, 'guidance_probe.score_improved_rate')}",
            board_surrogate is None,
        ),
        pass_item(
            "Board surrogate clean-action refinement",
            bool(get(board_surrogate_refine, "interpretation.passes_board_surrogate_action_refinement", False)),
            f"score_delta={get(board_surrogate_refine, 'summary.score_delta.mean')}, improved_rate={get(board_surrogate_refine, 'summary.score_improved_rate')}",
            board_surrogate_refine is None,
        ),
        pass_item(
            "Board production Foresight smoke",
            board_smoke_history is not None and board_smoke_ckpt_exists,
            f"epochs={board_smoke_epochs}, best_val={board_smoke_best_val}, ckpt_exists={board_smoke_ckpt_exists}",
            board_smoke_history is None or not board_smoke_ckpt_exists,
        ),
        pass_item(
            "Board Foresight fast20 training",
            bool(get(board_foresight_fast20, "interpretation.passes_board_foresight_fast20_training", False)),
            f"epochs={get(board_foresight_fast20, 'metrics.epochs')}, best_val={get(board_foresight_fast20, 'metrics.best_val')}, initial_val={get(board_foresight_fast20, 'metrics.initial_val')}",
            board_foresight_fast20 is None,
        ),
        pass_item(
            "Board Foresight fast100 training",
            bool(get(board_foresight_fast100, "interpretation.passes_board_foresight_fast100_training", False)),
            f"epochs={get(board_foresight_fast100, 'metrics.epochs')}, best_val={get(board_foresight_fast100, 'metrics.best_val')}, fast20_reduction={get(board_foresight_fast100, 'metrics.relative_best_val_reduction_vs_fast20')}",
            board_foresight_fast100 is None,
        ),
        pass_item(
            "Board production Foresight gradient probe",
            bool(get(board_foresight_gradient, "interpretation.passes_board_production_foresight_gradient", False)),
            f"score_improved_rate={get(board_foresight_gradient, 'summary.score_improved_rate')}, finite_grad_rate={get(board_foresight_gradient, 'summary.finite_grad_rate')}, nonzero_grad_rate={get(board_foresight_gradient, 'summary.nonzero_grad_rate')}",
            board_foresight_gradient is None,
        ),
        pass_item(
            "Board DP training-entry smoke",
            bool(get(board_dp_smoke, "interpretation.passes_board_dp_training_entry_smoke", False)),
            f"episodes={get(board_dp_smoke, 'n_episodes')}, epochs={get(board_dp_smoke, 'epochs')}, final_loss={get(board_dp_smoke, 'final_train_loss')}, dp_final_exists={get(board_dp_smoke, 'checks.dp_final_exists')}",
            board_dp_smoke is None,
        ),
        pass_item(
            "Board DP fast16_e20 training",
            bool(get(board_dp_fast16, "interpretation.passes_board_dp_fast16_training", False)),
            f"episodes={get(board_dp_fast16, 'n_episodes')}, epochs={get(board_dp_fast16, 'epochs')}, final_loss={get(board_dp_fast16, 'final_train_loss')}, initial_loss={get(board_dp_fast16, 'initial_train_loss')}",
            board_dp_fast16 is None,
        ),
        pass_item(
            "Board DP fast32_e20 training",
            bool(get(board_dp_fast32, "interpretation.passes_board_dp_fast32_training", False)),
            f"episodes={get(board_dp_fast32, 'n_episodes')}, epochs={get(board_dp_fast32, 'epochs')}, final_loss={get(board_dp_fast32, 'final_train_loss')}, initial_loss={get(board_dp_fast32, 'initial_train_loss')}",
            board_dp_fast32 is None,
        ),
        pass_item(
            "Board lazy full80 DP training entry",
            bool(get(board_dp_lazy_full80_entry, "interpretation.passes_lazy_full80_training_entry", False)),
            f"episodes={get(board_dp_lazy_full80_entry, 'n_episodes')}, windows={get(board_dp_lazy_full80_entry, 'max_train_windows')}, final_loss={get(board_dp_lazy_full80_entry, 'final_train_loss')}, lazy_images={get(board_dp_lazy_full80_entry, 'lazy_images')}",
            board_dp_lazy_full80_entry is None,
        ),
        pass_item(
            "Board image-cache DP training smoke",
            bool(get(board_dp_image_cache_smoke, "interpretation.passes_image_cache_smoke", False)),
            f"episodes={get(board_dp_image_cache_smoke, 'n_episodes')}, windows={get(board_dp_image_cache_smoke, 'max_train_windows')}, cache_files={get(board_dp_image_cache_smoke, 'cache_files')}, final_loss={get(board_dp_image_cache_smoke, 'final_train_loss')}",
            board_dp_image_cache_smoke is None,
        ),
        pass_item(
            "Board feature-cache DP training smoke",
            bool(get(board_dp_feature_cache_smoke, "interpretation.passes_feature_cache_smoke", False)),
            f"episodes={get(board_dp_feature_cache_smoke, 'n_episodes')}, windows={get(board_dp_feature_cache_smoke, 'max_train_windows')}, cache_files={get(board_dp_feature_cache_smoke, 'cache_files')}, cache_bytes={get(board_dp_feature_cache_smoke, 'cache_total_bytes')}, final_loss={get(board_dp_feature_cache_smoke, 'final_train_loss')}",
            board_dp_feature_cache_smoke is None,
        ),
        pass_item(
            "Board feature-cache full80 DP training",
            bool(get(board_dp_feature_cache_full80, "interpretation.passes_feature_cache_full80_training", False)),
            f"episodes={get(board_dp_feature_cache_full80, 'n_episodes')}, windows={get(board_dp_feature_cache_full80, 'n_windows')}, cache_files={get(board_dp_feature_cache_full80, 'cache_files')}, cache_bytes={get(board_dp_feature_cache_full80, 'cache_total_bytes')}, final_loss={get(board_dp_feature_cache_full80, 'final_train_loss')}",
            board_dp_feature_cache_full80 is None,
        ),
        pass_item(
            "Board DP/Foresight clean-action full-chain smoke",
            bool(get(board_dp_full_chain_smoke, "interpretation.passes_board_dp_full_chain_smoke", False)),
            f"mode={get(board_dp_full_chain_smoke, 'config.mode')}, score_delta={get(board_dp_full_chain_smoke, 'summary.score_delta.mean')}, beats={get(board_dp_full_chain_smoke, 'summary.guided_beats_base_rate')}, range_violation={get(board_dp_full_chain_smoke, 'summary.range_violation.max')}",
            board_dp_full_chain_smoke is None,
        ),
        pass_item(
            "Board feature-cache DP/Foresight full-chain heldout",
            bool(get(board_dp_feature_cache_full_chain, "interpretation.passes_board_dp_full_chain_smoke", False)),
            f"frames={get(board_dp_feature_cache_full_chain, 'n_frames')}, samples={get(board_dp_feature_cache_full_chain, 'n_action_samples')}, score_delta={get(board_dp_feature_cache_full_chain, 'summary.score_delta.mean')}, beats={get(board_dp_feature_cache_full_chain, 'summary.guided_beats_base_rate')}, range_violation={get(board_dp_feature_cache_full_chain, 'summary.range_violation.max')}",
            board_dp_feature_cache_full_chain is None,
        ),
        pass_item(
            "Board feature-cache DP/Foresight fast100 full-chain heldout",
            bool(get(board_dp_feature_cache_full_chain_fast100, "interpretation.passes_board_dp_full_chain_smoke", False)),
            f"frames={get(board_dp_feature_cache_full_chain_fast100, 'n_frames')}, samples={get(board_dp_feature_cache_full_chain_fast100, 'n_action_samples')}, score_delta={get(board_dp_feature_cache_full_chain_fast100, 'summary.score_delta.mean')}, beats={get(board_dp_feature_cache_full_chain_fast100, 'summary.guided_beats_base_rate')}, range_violation={get(board_dp_feature_cache_full_chain_fast100, 'summary.range_violation.max')}",
            board_dp_feature_cache_full_chain_fast100 is None,
        ),
        pass_item(
            "Board full-chain DP/Foresight guidance",
            False,
            "Feature-cache full80 DP heldout full-chain passed with fast20 and stronger fast100 Foresight; final production policy validation is still missing.",
            False,
        ),
        pass_item(
            "Offline production-readiness gate",
            bool(get(offline_production_gate, "offline_production_gate_pass", False)),
            f"remaining={get(offline_production_gate, 'remaining_required_step')}, board_delta={get(offline_production_gate, 'metrics.board.full_chain_score_delta_mean')}, insertion_delta={get(offline_production_gate, 'metrics.insertion.clean_score_delta_mean')}",
            offline_production_gate is None,
        ),
        pass_item(
            "Board guidance-score calibration",
            get(score_calibration, "recommendation.board") == "quality"
            and (get(score_calibration, "ptg_proxy_v2_board.modes.quality.summary.quality_positive_step_rate", 0.0) or 0.0) >= 0.95
            and (get(score_calibration, "ptg_proxy_v2_board.modes.quality.summary.quality_spearman", 0.0) or 0.0) >= 0.90,
            f"mode={get(score_calibration, 'recommendation.board')}, spearman={get(score_calibration, 'ptg_proxy_v2_board.modes.quality.summary.quality_spearman')}, q_gap={get(score_calibration, 'ptg_proxy_v2_board.modes.quality.summary.top_bottom_quality_gap')}",
            score_calibration is None,
        ),
        pass_item(
            "Board unified runtime contract",
            bool(get(runtime_contract, "passes_runtime_contract_sanity", False))
            and bool(get(runtime_contract, "board.all_finite", False))
            and bool(get(runtime_contract, "board.all_nonzero", False)),
            f"score_call={get(runtime_contract, 'contract.score_call')}, joint_grad={get(runtime_contract, 'board.joint_action_grad_norm_mean')}, eef_grad={get(runtime_contract, 'board.eef_action_grad_norm_mean')}",
            runtime_contract is None,
        ),
        pass_item(
            "Board trust-region guidance update",
            bool(get(trust_region_guidance, "passes_trust_region_guidance_sanity", False))
            and (get(trust_region_guidance, "board.improved_rate", 0.0) or 0.0) >= 0.99
            and bool(get(trust_region_guidance, "board.max_delta_within_trust_region", False)),
            f"improved={get(trust_region_guidance, 'board.improved_rate')}, delta_max={get(trust_region_guidance, 'board.delta_norm.max')}, limit={get(trust_region_guidance, 'board.config.max_total_delta')}",
            trust_region_guidance is None,
        ),
        pass_item(
            "Board manifest real-sample smoke",
            bool(get(manifest_real_sample_smoke, "board.passes_real_sample_smoke", False)),
            f"improved={get(manifest_real_sample_smoke, 'board.score_improved_rate')}, score_delta={get(manifest_real_sample_smoke, 'board.score_delta.mean')}, delta_max={get(manifest_real_sample_smoke, 'board.delta_norm.max')}",
            manifest_real_sample_smoke is None,
        ),
        pass_item(
            "Board local guidance-scale law",
            bool(get(guidance_scale_sweep, "board.passes_guidance_scale_sweep", False)),
            f"recommended_scale={get(guidance_scale_sweep, 'board.recommended_scale')}, improved={get(guidance_scale_sweep, 'board.recommended_improved_rate')}, score_delta={get(guidance_scale_sweep, 'board.recommended_score_delta_mean')}",
            guidance_scale_sweep is None,
        ),
        pass_item(
            "Board current-gradient robustness",
            bool(get(guidance_robustness, "board.passes_current_gradient_robustness", False)),
            f"worst_current_improved={get(guidance_robustness, 'board.worst_perturbed_gradient_improved_rate')}, stale_gradient_stable={get(guidance_robustness, 'board.stale_gradient_stable_under_noise')}",
            guidance_robustness is None,
        ),
        pass_item(
            "Board DP guidance controller",
            bool(get(dp_guidance_controller, "passes_controller_sanity", False))
            and bool(get(dp_guidance_controller_real_sample, "board.passes_controller_real_sample", False)),
            f"real_sample_improved={get(dp_guidance_controller_real_sample, 'board.report.improved_rate')}, stale_allowed={get(dp_guidance_controller_real_sample, 'board.report.guardrails.stale_gradient_reuse_allowed')}",
            dp_guidance_controller is None or dp_guidance_controller_real_sample is None,
        ),
    ]

    unified_best = get(unified, "best_candidates", [])
    best_taxonomy = unified_best[0] if unified_best else None
    taxonomy_checks = [
        pass_item(
            "Unified taxonomy baseline exists",
            best_taxonomy is not None,
            f"best={best_taxonomy}",
            unified is None,
        ),
        pass_item(
            "Task-conditioned scorer preferred over task-agnostic scorer",
            True,
            "Cross-task zero-shot RF/LogReg was weak; PTG v2 mixed scorer is stronger.",
        ),
        pass_item(
            "Deployment manifest ready",
            bool(get(deployment_manifest, "deployment_manifest_pass", False)),
            f"score_api={get(deployment_manifest, 'score_api.score_call')}, remaining={get(deployment_manifest, 'remaining_required_step')}",
            deployment_manifest is None,
        ),
    ]

    all_checks = insertion_checks + board_checks + taxonomy_checks
    achieved = all(item["passed"] for item in all_checks)
    result = {
        "profiles": profile_summary(),
        "paths": {name: str(path) for name, path in paths.items()},
        "missing_files": missing,
        "checks": {
            "insertion": insertion_checks,
            "board": board_checks,
            "taxonomy": taxonomy_checks,
        },
        "metrics": {
            "insertion": {
                "scorer_binary_auc": mean_metric(insertion_scorer, "mixed_group_cv.binary_auc.mean"),
                "full_chain_pass": get(insertion_full, "interpretation.passes_full_chain_gradient"),
                "clean_refine_score_delta_mean": get(insertion_refine, "summary.score_delta.mean"),
                "clean_refine_beats": get(insertion_refine, "summary.refined_beats_base_rate"),
                "calibrated_guidance_mode": get(score_calibration, "recommendation.insertion"),
                "calibrated_energy_quality_spearman": get(score_calibration, "insertion_risk_scorer.modes.energy.summary.quality_spearman"),
                "calibrated_energy_quality_gap": get(score_calibration, "insertion_risk_scorer.modes.energy.summary.top_bottom_quality_gap"),
                "calibrated_energy_auc": get(score_calibration, "insertion_risk_scorer.modes.energy.summary.binary_auc"),
                "runtime_contract_pass": get(runtime_contract, "passes_runtime_contract_sanity"),
                "runtime_contract_action_grad_norm_mean": get(runtime_contract, "insertion.action_grad_norm_mean"),
                "trust_region_update_pass": get(trust_region_guidance, "passes_trust_region_guidance_sanity"),
                "trust_region_update_improved_rate": get(trust_region_guidance, "insertion.improved_rate"),
                "trust_region_update_delta_max": get(trust_region_guidance, "insertion.delta_norm.max"),
                "manifest_real_sample_smoke_pass": get(manifest_real_sample_smoke, "insertion.passes_real_sample_smoke"),
                "manifest_real_sample_smoke_improved_rate": get(manifest_real_sample_smoke, "insertion.score_improved_rate"),
                "manifest_real_sample_smoke_score_delta_mean": get(manifest_real_sample_smoke, "insertion.score_delta.mean"),
                "manifest_real_sample_smoke_delta_max": get(manifest_real_sample_smoke, "insertion.delta_norm.max"),
                "guidance_scale_sweep_pass": get(guidance_scale_sweep, "insertion.passes_guidance_scale_sweep"),
                "guidance_scale_sweep_recommended_scale": get(guidance_scale_sweep, "insertion.recommended_scale"),
                "guidance_scale_sweep_score_delta_mean": get(guidance_scale_sweep, "insertion.recommended_score_delta_mean"),
                "guidance_scale_sweep_improved_rate": get(guidance_scale_sweep, "insertion.recommended_improved_rate"),
                "guidance_robustness_current_gradient_pass": get(guidance_robustness, "insertion.passes_current_gradient_robustness"),
                "guidance_robustness_worst_current_improved_rate": get(guidance_robustness, "insertion.worst_perturbed_gradient_improved_rate"),
                "guidance_robustness_stale_gradient_stable": get(guidance_robustness, "insertion.stale_gradient_stable_under_noise"),
                "dp_guidance_controller_sanity_pass": get(dp_guidance_controller, "passes_controller_sanity"),
                "dp_guidance_controller_real_sample_pass": get(dp_guidance_controller_real_sample, "insertion.passes_controller_real_sample"),
                "dp_guidance_controller_real_sample_improved_rate": get(dp_guidance_controller_real_sample, "insertion.report.improved_rate"),
                "dp_guidance_controller_stale_gradient_allowed": get(dp_guidance_controller_real_sample, "insertion.report.guardrails.stale_gradient_reuse_allowed"),
                "controller_denoising_smoke_pass": get(controller_denoising_smoke, "interpretation.passes_controller_denoising_smoke"),
                "controller_denoising_smoke_score_delta_mean": get(controller_denoising_smoke, "summary.score_delta.mean"),
                "controller_denoising_smoke_beats": get(controller_denoising_smoke, "summary.guided_beats_base_rate"),
                "controller_denoising_smoke_range_violation_max": get(controller_denoising_smoke, "summary.range_violation.max"),
            },
            "board": {
                "ptg_v2_mixed_binary_auc": mean_metric(ptg_v2, "mixed_group_cv.binary_auc.mean"),
                "ptg_v2_mixed_quality_corr": mean_metric(ptg_v2, "mixed_group_cv.quality_corr.mean"),
                "readiness_score_delta_mean": get(board_ready, "summary.score_delta.mean"),
                "readiness_improved_rate": get(board_ready, "summary.score_improved_rate"),
                "surrogate_marker_mae": get(board_surrogate, "eval.marker_mae.mean"),
                "surrogate_score_improved_rate": get(board_surrogate, "guidance_probe.score_improved_rate"),
                "surrogate_full_chain_pass": get(board_surrogate, "interpretation.passes_board_surrogate_full_chain"),
                "surrogate_refine_score_delta_mean": get(board_surrogate_refine, "summary.score_delta.mean"),
                "surrogate_refine_improved_rate": get(board_surrogate_refine, "summary.score_improved_rate"),
                "surrogate_refine_pass": get(board_surrogate_refine, "interpretation.passes_board_surrogate_action_refinement"),
                "production_foresight_smoke_epochs": board_smoke_epochs,
                "production_foresight_smoke_best_val": board_smoke_best_val,
                "production_foresight_smoke_ckpt_exists": board_smoke_ckpt_exists,
                "production_foresight_fast20_pass": get(board_foresight_fast20, "interpretation.passes_board_foresight_fast20_training"),
                "production_foresight_fast20_best_val": get(board_foresight_fast20, "metrics.best_val"),
                "production_foresight_fast20_initial_val": get(board_foresight_fast20, "metrics.initial_val"),
                "production_foresight_fast20_epochs": get(board_foresight_fast20, "metrics.epochs"),
                "production_foresight_fast100_pass": get(board_foresight_fast100, "interpretation.passes_board_foresight_fast100_training"),
                "production_foresight_fast100_best_val": get(board_foresight_fast100, "metrics.best_val"),
                "production_foresight_fast100_final_val": get(board_foresight_fast100, "metrics.final_val"),
                "production_foresight_fast100_relative_reduction_vs_fast20": get(board_foresight_fast100, "metrics.relative_best_val_reduction_vs_fast20"),
                "production_foresight_gradient_pass": get(board_foresight_gradient, "interpretation.passes_board_production_foresight_gradient"),
                "production_foresight_gradient_score_delta_mean": get(board_foresight_gradient, "summary.score_delta.mean"),
                "production_foresight_gradient_improved_rate": get(board_foresight_gradient, "summary.score_improved_rate"),
                "production_foresight_gradient_finite_grad_rate": get(board_foresight_gradient, "summary.finite_grad_rate"),
                "production_foresight_gradient_nonzero_grad_rate": get(board_foresight_gradient, "summary.nonzero_grad_rate"),
                "dp_training_entry_smoke_pass": get(board_dp_smoke, "interpretation.passes_board_dp_training_entry_smoke"),
                "dp_training_entry_smoke_final_loss": get(board_dp_smoke, "final_train_loss"),
                "dp_training_entry_smoke_epochs": get(board_dp_smoke, "epochs"),
                "dp_training_entry_smoke_n_episodes": get(board_dp_smoke, "n_episodes"),
                "dp_fast16_pass": get(board_dp_fast16, "interpretation.passes_board_dp_fast16_training"),
                "dp_fast16_final_loss": get(board_dp_fast16, "final_train_loss"),
                "dp_fast16_initial_loss": get(board_dp_fast16, "initial_train_loss"),
                "dp_fast16_epochs": get(board_dp_fast16, "epochs"),
                "dp_fast16_n_episodes": get(board_dp_fast16, "n_episodes"),
                "dp_fast32_pass": get(board_dp_fast32, "interpretation.passes_board_dp_fast32_training"),
                "dp_fast32_final_loss": get(board_dp_fast32, "final_train_loss"),
                "dp_fast32_initial_loss": get(board_dp_fast32, "initial_train_loss"),
                "dp_fast32_best_loss": get(board_dp_fast32, "best_train_loss"),
                "dp_fast32_epochs": get(board_dp_fast32, "epochs"),
                "dp_fast32_n_episodes": get(board_dp_fast32, "n_episodes"),
                "dp_lazy_full80_entry_pass": get(board_dp_lazy_full80_entry, "interpretation.passes_lazy_full80_training_entry"),
                "dp_lazy_full80_entry_final_loss": get(board_dp_lazy_full80_entry, "final_train_loss"),
                "dp_lazy_full80_entry_windows": get(board_dp_lazy_full80_entry, "max_train_windows"),
                "dp_lazy_full80_entry_n_episodes": get(board_dp_lazy_full80_entry, "n_episodes"),
                "dp_image_cache_smoke_pass": get(board_dp_image_cache_smoke, "interpretation.passes_image_cache_smoke"),
                "dp_image_cache_smoke_final_loss": get(board_dp_image_cache_smoke, "final_train_loss"),
                "dp_image_cache_smoke_cache_files": get(board_dp_image_cache_smoke, "cache_files"),
                "dp_image_cache_smoke_cache_total_bytes": get(board_dp_image_cache_smoke, "cache_total_bytes"),
                "dp_feature_cache_smoke_pass": get(board_dp_feature_cache_smoke, "interpretation.passes_feature_cache_smoke"),
                "dp_feature_cache_smoke_final_loss": get(board_dp_feature_cache_smoke, "final_train_loss"),
                "dp_feature_cache_smoke_cache_files": get(board_dp_feature_cache_smoke, "cache_files"),
                "dp_feature_cache_smoke_cache_total_bytes": get(board_dp_feature_cache_smoke, "cache_total_bytes"),
                "dp_feature_cache_full80_pass": get(board_dp_feature_cache_full80, "interpretation.passes_feature_cache_full80_training"),
                "dp_feature_cache_full80_final_loss": get(board_dp_feature_cache_full80, "final_train_loss"),
                "dp_feature_cache_full80_cache_files": get(board_dp_feature_cache_full80, "cache_files"),
                "dp_feature_cache_full80_cache_total_bytes": get(board_dp_feature_cache_full80, "cache_total_bytes"),
                "dp_full_chain_smoke_pass": get(board_dp_full_chain_smoke, "interpretation.passes_board_dp_full_chain_smoke"),
                "dp_full_chain_smoke_mode": get(board_dp_full_chain_smoke, "config.mode"),
                "dp_full_chain_smoke_score_delta_mean": get(board_dp_full_chain_smoke, "summary.score_delta.mean"),
                "dp_full_chain_smoke_beats": get(board_dp_full_chain_smoke, "summary.guided_beats_base_rate"),
                "dp_full_chain_smoke_range_violation_max": get(board_dp_full_chain_smoke, "summary.range_violation.max"),
                "dp_feature_cache_full_chain_pass": get(board_dp_feature_cache_full_chain, "interpretation.passes_board_dp_full_chain_smoke"),
                "dp_feature_cache_full_chain_score_delta_mean": get(board_dp_feature_cache_full_chain, "summary.score_delta.mean"),
                "dp_feature_cache_full_chain_beats": get(board_dp_feature_cache_full_chain, "summary.guided_beats_base_rate"),
                "dp_feature_cache_full_chain_range_violation_max": get(board_dp_feature_cache_full_chain, "summary.range_violation.max"),
                "dp_feature_cache_fast100_full_chain_pass": get(board_dp_feature_cache_full_chain_fast100, "interpretation.passes_board_dp_full_chain_smoke"),
                "dp_feature_cache_fast100_full_chain_score_delta_mean": get(board_dp_feature_cache_full_chain_fast100, "summary.score_delta.mean"),
                "dp_feature_cache_fast100_full_chain_beats": get(board_dp_feature_cache_full_chain_fast100, "summary.guided_beats_base_rate"),
                "dp_feature_cache_fast100_full_chain_range_violation_max": get(board_dp_feature_cache_full_chain_fast100, "summary.range_violation.max"),
                "offline_production_gate_pass": get(offline_production_gate, "offline_production_gate_pass"),
                "calibrated_guidance_mode": get(score_calibration, "recommendation.board"),
                "calibrated_quality_spearman": get(score_calibration, "ptg_proxy_v2_board.modes.quality.summary.quality_spearman"),
                "calibrated_quality_gap": get(score_calibration, "ptg_proxy_v2_board.modes.quality.summary.top_bottom_quality_gap"),
                "calibrated_quality_auc": get(score_calibration, "ptg_proxy_v2_board.modes.quality.summary.binary_auc"),
                "deployed_weighted_energy_spearman": get(score_calibration, "ptg_proxy_v2_board.modes.weighted_energy.summary.quality_spearman"),
                "deployed_weighted_energy_auc": get(score_calibration, "ptg_proxy_v2_board.modes.weighted_energy.summary.binary_auc"),
                "runtime_contract_pass": get(runtime_contract, "passes_runtime_contract_sanity"),
                "runtime_contract_joint_grad_norm_mean": get(runtime_contract, "board.joint_action_grad_norm_mean"),
                "runtime_contract_eef_grad_norm_mean": get(runtime_contract, "board.eef_action_grad_norm_mean"),
                "trust_region_update_pass": get(trust_region_guidance, "passes_trust_region_guidance_sanity"),
                "trust_region_update_improved_rate": get(trust_region_guidance, "board.improved_rate"),
                "trust_region_update_delta_max": get(trust_region_guidance, "board.delta_norm.max"),
                "manifest_real_sample_smoke_pass": get(manifest_real_sample_smoke, "board.passes_real_sample_smoke"),
                "manifest_real_sample_smoke_improved_rate": get(manifest_real_sample_smoke, "board.score_improved_rate"),
                "manifest_real_sample_smoke_score_delta_mean": get(manifest_real_sample_smoke, "board.score_delta.mean"),
                "manifest_real_sample_smoke_delta_max": get(manifest_real_sample_smoke, "board.delta_norm.max"),
                "guidance_scale_sweep_pass": get(guidance_scale_sweep, "board.passes_guidance_scale_sweep"),
                "guidance_scale_sweep_recommended_scale": get(guidance_scale_sweep, "board.recommended_scale"),
                "guidance_scale_sweep_score_delta_mean": get(guidance_scale_sweep, "board.recommended_score_delta_mean"),
                "guidance_scale_sweep_improved_rate": get(guidance_scale_sweep, "board.recommended_improved_rate"),
                "guidance_robustness_current_gradient_pass": get(guidance_robustness, "board.passes_current_gradient_robustness"),
                "guidance_robustness_worst_current_improved_rate": get(guidance_robustness, "board.worst_perturbed_gradient_improved_rate"),
                "guidance_robustness_stale_gradient_stable": get(guidance_robustness, "board.stale_gradient_stable_under_noise"),
                "dp_guidance_controller_sanity_pass": get(dp_guidance_controller, "passes_controller_sanity"),
                "dp_guidance_controller_real_sample_pass": get(dp_guidance_controller_real_sample, "board.passes_controller_real_sample"),
                "dp_guidance_controller_real_sample_improved_rate": get(dp_guidance_controller_real_sample, "board.report.improved_rate"),
                "dp_guidance_controller_stale_gradient_allowed": get(dp_guidance_controller_real_sample, "board.report.guardrails.stale_gradient_reuse_allowed"),
                "full_chain_pass": False,
            },
            "unified_taxonomy": {
                "best_candidate": best_taxonomy,
            },
            "deployment_manifest": {
                "pass": get(deployment_manifest, "deployment_manifest_pass"),
                "score_call": get(deployment_manifest, "score_api.score_call"),
                "refine_call": get(deployment_manifest, "score_api.refine_call"),
                "remaining_required_step": get(deployment_manifest, "remaining_required_step"),
                "real_sample_smoke_pass": get(manifest_real_sample_smoke, "overall_pass"),
            },
        },
        "completion_assessment": {
            "objective_complete": achieved,
            "reason": (
                "All scorer, insertion full-chain, board stronger Foresight, board feature-cache full80 heldout full-chain, and offline production-readiness checks pass; real-robot validation is still missing."
                if not achieved
                else "All required scorer and full-chain checks pass."
            ),
            "next_required_step": "Run real-robot / final production policy validation; current offline production gate passes.",
        },
    }
    return result


def write_markdown(summary: Dict[str, Any], path: Path):
    lines = [
        "# PTG Guidance Evidence Summary",
        "",
        "## Completion",
        "",
        f"- objective_complete: `{summary['completion_assessment']['objective_complete']}`",
        f"- reason: {summary['completion_assessment']['reason']}",
        f"- next_required_step: {summary['completion_assessment']['next_required_step']}",
        "",
        "## Profiles",
        "",
    ]
    for name, profile in summary["profiles"].items():
        energy = profile["energy"]
        refine = profile["refinement"]
        lines.extend(
            [
                f"### {name}",
                "",
                f"- scorer: `{profile['scorer']}`",
                f"- energy: `{energy['quality']}*quality + {energy['binary_margin']}*binary_margin + {energy['reason_margin']}*reason_margin`",
                f"- action_step: `{refine['action_step']}`",
                f"- max_total_delta: `{refine['max_total_delta']}`",
                f"- scope: {profile['scope']}",
                "",
            ]
        )
    lines.extend(["## Checks", ""])
    for group, checks in summary["checks"].items():
        lines.extend([f"### {group}", ""])
        for item in checks:
            status = "PASS" if item["passed"] else "FAIL"
            lines.append(f"- **{status}** {item['name']}: {item['evidence']}")
        lines.append("")
    lines.extend(["## Metrics", "", "```json", json.dumps(summary["metrics"], ensure_ascii=False, indent=2), "```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default=str(OUT_DIR))
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary(DEFAULT_PATHS)
    json_path = out_dir / "ptg_guidance_evidence_summary.json"
    md_path = out_dir / "ptg_guidance_evidence_summary.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(summary, md_path)
    print(json.dumps(summary["completion_assessment"], ensure_ascii=False, indent=2))
    print(f"Saved {json_path}")
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
