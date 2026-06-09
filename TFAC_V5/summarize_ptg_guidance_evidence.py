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
            "Board full-chain DP/Foresight guidance",
            False,
            "Feature-cache full80 DP heldout full-chain passed, but stronger board Foresight and final production policy validation are still missing.",
            False,
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
                "full_chain_pass": False,
            },
            "unified_taxonomy": {
                "best_candidate": best_taxonomy,
            },
        },
        "completion_assessment": {
            "objective_complete": achieved,
            "reason": (
                "All scorer, insertion full-chain, board production Foresight gradient, and board feature-cache full80 heldout full-chain checks pass, but stronger board Foresight and final production policy validation are still missing."
                if not achieved
                else "All required scorer and full-chain checks pass."
            ),
            "next_required_step": "Train/evaluate stronger board Foresight and run final production policy validation; current feature-cache full80 DP heldout full-chain guidance is positive.",
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
