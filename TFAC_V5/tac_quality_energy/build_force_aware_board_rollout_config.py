#!/usr/bin/env python3
"""Build rollout config with the force-aware board guidance research arm.

This keeps the current deployable board arm (`marker_joint_s12_guided`) and the
current insertion arm (`good_margin_guided`) unchanged, then adds an optional
board research arm:

    force_aware_guided

The arm is explicit about its score preset.  The current offline weight sweep
selects `margin_only`, so serving should not silently fall back to the older
mixed contact/center/smooth penalty weights.
"""

from __future__ import annotations

import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


BASE_CONFIG = Path(
    "/home/chenshuai/Project/output/tac_quality_rollout_arm_configs/"
    "tac_quality_rollout_arm_configs_current_s12_good_margin_20260619.json"
)
OUTPUT_CONFIG = Path(
    "/home/chenshuai/Project/output/tac_quality_rollout_arm_configs/"
    "tac_quality_rollout_arm_configs_current_s12_good_margin_forceaware_board_20260621.json"
)
FORESIGHT_DIR = Path(
    "/home/chenshuai/Project/output/foresight_ckpt/"
    "latent_foresight_board_forceaware_multistep16_boardvae_e100_bs16_0"
)
FORESIGHT_CKPT = FORESIGHT_DIR / "foresight_force_best.ckpt"
FORESIGHT_SUMMARY = FORESIGHT_DIR / "force_aware_foresight_summary.json"
FORCE_AWARE_AUDIT = Path(
    "/home/chenshuai/Project/output/force_aware_foresight_guidance_audit/"
    "20260621_090725/audit_results.json"
)
WEIGHT_SWEEP = Path(
    "/home/chenshuai/Project/output/force_aware_score_weight_sweep/"
    "20260621_104503/force_aware_score_weight_sweep.json"
)
BOARD_FORCE_CALIBRATION = Path(
    "/home/chenshuai/Project/output/board_target_force_calibration/"
    "board_target_force_calibration.json"
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def get(data: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = data
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return default
        cur = cur[part]
    return cur


def main() -> None:
    for path in [
        BASE_CONFIG,
        FORESIGHT_CKPT,
        FORESIGHT_SUMMARY,
        FORCE_AWARE_AUDIT,
        WEIGHT_SWEEP,
    ]:
        require(path)

    cfg = load_json(BASE_CONFIG)
    audit = load_json(FORCE_AWARE_AUDIT)
    sweep = load_json(WEIGHT_SWEEP)
    summary = load_json(FORESIGHT_SUMMARY)
    best = sweep.get("best", {})
    if best.get("name") != "margin_only":
        raise ValueError(f"Unexpected force-aware best preset: {best.get('name')!r}")
    weights = best.get("weights", {})
    expected = {
        "band_margin": 1.0,
        "contact_logprob": 0.0,
        "force_center": 0.0,
        "force_smooth": 0.0,
        "action_smooth": 0.0,
    }
    if weights != expected:
        raise ValueError(f"Unexpected margin_only weights: {weights!r}")

    board = cfg["tasks"]["board"]
    base_board = board["marker_joint_s12_guided"]
    arm = copy.deepcopy(base_board)
    arm.update(
        {
            "arm": "force_aware_guided",
            "policy": "dp_with_final_clean_action_force_aware_foresight_trust_region_refinement",
            "guidance_enabled": True,
            "scorer_runtime": "ForceAwareForesightGuidanceRuntime",
            "checkpoint": {
                "path": str(FORESIGHT_CKPT),
                "exists": FORESIGHT_CKPT.exists(),
                "bytes": FORESIGHT_CKPT.stat().st_size,
            },
            "force_aware_foresight": {
                "foresight_dir": str(FORESIGHT_DIR),
                "audit_json": str(FORCE_AWARE_AUDIT),
                "summary_json": str(FORESIGHT_SUMMARY),
                "weight_sweep_json": str(WEIGHT_SWEEP),
            },
            "refiner": {
                "profile": "board_force_aware",
                "task_id": 1,
                "score_mode": "force_aware_quality",
                "refinement": {
                    "refine_steps": 4,
                    "marker_step": 0.0,
                    "action_step": 0.02,
                    "max_total_delta": 0.08,
                    "smooth_weight": 0.0,
                    "joint_limit_weight": 0.0,
                    "joint_margin_frac": 0.03,
                    "accept_only_improved": True,
                },
                "energy": {
                    "source": "ForceAwareForesightGuidanceRuntime.force_aware_quality",
                    "score_preset": "margin_only",
                    "definition": "good-vs-risk force-band logit margin",
                    "weights": weights,
                    "selected_by": str(WEIGHT_SWEEP),
                    "weight_sweep_summary": {
                        "ranking_score": best.get("ranking_score"),
                        "improved_rate": best.get("improved_rate"),
                        "score_delta_mean": best.get("score_delta_mean"),
                        "action_delta_norm_mean": best.get("action_delta_norm_mean"),
                        "raw_action_delta_norm_mean": best.get("raw_action_delta_norm_mean"),
                        "band_balanced_acc": best.get("band_balanced_acc"),
                        "contact_acc": best.get("contact_acc"),
                        "score_good_bad_auc": best.get("score_good_bad_auc"),
                    },
                    "heldout_audit": {
                        "band_balanced_acc": get(audit, "scorer_metrics.band_balanced_acc"),
                        "contact_acc": get(audit, "scorer_metrics.contact_acc"),
                        "score_good_bad_auc": get(audit, "scorer_metrics.score_good_bad_auc"),
                        "finite_grad_rate": get(audit, "guidance_metrics.finite_grad_rate"),
                        "positive_grad_rate": get(audit, "guidance_metrics.positive_grad_rate"),
                        "improved_rate": get(audit, "guidance_metrics.improved_rate"),
                        "score_delta_mean": get(audit, "summaries.score_delta.mean"),
                    },
                },
                "evidence": (
                    "2026-06-21 force-aware Foresight held-out gradient audit and "
                    "score-weight sweep select margin_only as the strongest offline "
                    "bounded guidance score. This arm is still a research candidate "
                    "and requires paired real rollout force traces."
                ),
                "scope": (
                    "Optional force-aware board guidance arm. Not the default board "
                    "recommendation until real baseline/guided force-curve evidence passes."
                ),
            },
            "board_force_gate": {
                "force_ref": summary.get("force_ref", {}),
                "calibration_json": str(BOARD_FORCE_CALIBRATION),
            },
            "expected_rollout_dir_placeholder": "<board_force_aware_guided_rollout_dir>",
            "optional_research_arm": True,
        }
    )
    board["force_aware_guided"] = arm

    cfg["purpose"] = (
        "Current machine-readable TacQuality rollout arm config for board "
        "marker_joint_s12_guided, insertion good_margin_guided, and optional "
        "force_aware_guided board research tests."
    )
    cfg["generated_by"] = "TFAC_V5/tac_quality_energy/build_force_aware_board_rollout_config.py"
    cfg["force_aware_guided_evidence"] = {
        "foresight_audit": str(FORCE_AWARE_AUDIT),
        "score_weight_sweep": str(WEIGHT_SWEEP),
        "selected_score_preset": "margin_only",
        "selected_score_weights": weights,
        "evidence_boundary": (
            "offline gradient audit and real-HDF5-window serving checks only; "
            "not a real robot improvement claim"
        ),
    }
    cfg.setdefault("selection_summary", {})
    cfg["selection_summary"]["board_force_aware_guided"] = (
        "Optional research arm using ForceAwareForesightGuidanceRuntime with the "
        "margin_only score preset selected by the 2026-06-21 weight sweep. "
        "It is not the default board arm until paired real force traces pass."
    )
    cfg["checks"] = dict(cfg.get("checks", {}))
    cfg["checks"].update(
        {
            "force_aware_guided_arm_exists": True,
            "force_aware_checkpoint_exists": FORESIGHT_CKPT.exists(),
            "force_aware_audit_exists": FORCE_AWARE_AUDIT.exists(),
            "force_aware_weight_sweep_exists": WEIGHT_SWEEP.exists(),
            "force_aware_score_preset_is_margin_only": True,
        }
    )
    cfg["rollout_arm_config_pass"] = all(
        bool(cfg["checks"].get(key))
        for key in [
            "all_checkpoints_exist",
            "marker_joint_s12_checkpoint_exists",
            "good_margin_guided_arm_exists",
            "force_aware_guided_arm_exists",
            "force_aware_checkpoint_exists",
            "force_aware_audit_exists",
            "force_aware_weight_sweep_exists",
            "force_aware_score_preset_is_margin_only",
        ]
    )
    cfg["updated_by"] = "codex"
    cfg["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M CST")

    OUTPUT_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_CONFIG.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(OUTPUT_CONFIG),
                "arm": "force_aware_guided",
                "score_preset": "margin_only",
                "weights": weights,
                "pass": cfg["rollout_arm_config_pass"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
