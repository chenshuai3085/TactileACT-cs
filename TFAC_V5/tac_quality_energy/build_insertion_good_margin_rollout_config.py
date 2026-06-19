#!/usr/bin/env python3
"""Build rollout config with insertion good_margin guidance arm.

The good_margin arm is an insertion ablation candidate selected after the
2026-06-19 cross-score audit.  It keeps the same scorer checkpoint and trust
region as the conservative insertion profile arm, but changes the score mode to
the unsaturated binary logit margin.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path


DEFAULT_BASE_CONFIG = Path(
    "/home/chenshuai/Project/output/tac_quality_rollout_arm_configs/"
    "tac_quality_rollout_arm_configs_marker_joint_20260619_semantic_pgood_s12.json"
)
DEFAULT_OUTPUT = Path(
    "/home/chenshuai/Project/output/tac_quality_rollout_arm_configs/"
    "tac_quality_rollout_arm_configs_marker_joint_20260619_insertion_good_margin.json"
)
GOOD_MARGIN_ABLATION = Path(
    "/home/chenshuai/Project/output/tac_quality_score_mode_ablation/"
    "insertion_0401_profile_pgood_energy_goodmargin_cross_score_20260619/"
    "insertion_score_mode_ablation.json"
)
GOOD_MARGIN_DDPM_SWEEP = Path(
    "/home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/"
    "insertion_0401_good_margin_protected_multiep8_start2_seed2_t0_s001/"
    "insertion_ddpm_step_guidance_sweep.json"
)


def main() -> None:
    cfg = json.loads(DEFAULT_BASE_CONFIG.read_text(encoding="utf-8"))
    insertion = cfg["tasks"]["insertion"]
    base = insertion["default_guided"]
    arm = copy.deepcopy(base)
    arm["arm"] = "good_margin_guided"
    arm["optional_ablation_arm"] = True
    arm["policy"] = "dp_with_final_clean_action_trust_region_refinement"
    arm["refiner"]["score_mode"] = "good_margin"
    arm["refiner"]["energy"] = {
        "source": "InsertionRiskScorerRuntime.good_margin",
        "definition": "binary_logits[:, good] - binary_logits[:, bad]",
        "reason": "unsaturated binary logit margin avoids p_good probability saturation",
        "selected_by": str(GOOD_MARGIN_ABLATION),
        "ddpm_step_sweep": str(GOOD_MARGIN_DDPM_SWEEP),
        "cross_score_summary": {
            "own_delta_mean": 0.001154,
            "own_improve_rate": 0.9375,
            "profile_delta_mean": 0.000093,
            "energy_delta_mean": 0.000394,
            "quality_logit_delta_mean": 0.000557,
            "quality_logit_delta_min": 0.0,
            "action_delta_norm_mean": 0.000102,
        },
    }
    arm["refiner"]["evidence"] = (
        "2026-06-19 offline DDPM-step and cross-score ablation: good_margin is "
        "the strongest unsaturated insertion guidance candidate; real robot "
        "A/B success/bounce evidence is still required."
    )
    arm["refiner"]["scope"] = (
        "Insertion A/B candidate only. Do not claim real robot improvement "
        "without paired baseline/guided rollouts."
    )
    arm["expected_rollout_dir_placeholder"] = "<insertion_good_margin_guided_rollout_dir>"
    insertion["good_margin_guided"] = arm
    cfg["purpose"] = (
        "Machine-readable rollout arm configs for TacQuality scorer validation, "
        "including insertion good_margin unsaturated guidance candidate."
    )
    cfg["generated_by"] = "TFAC_V5/tac_quality_energy/build_insertion_good_margin_rollout_config.py"
    cfg["good_margin_evidence"] = {
        "score_mode_ablation": str(GOOD_MARGIN_ABLATION),
        "ddpm_step_sweep": str(GOOD_MARGIN_DDPM_SWEEP),
        "evidence_boundary": "offline sampler/cross-score only, not real robot outcome proof",
    }
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(DEFAULT_OUTPUT), "arm": "good_margin_guided"}, indent=2))


if __name__ == "__main__":
    main()
