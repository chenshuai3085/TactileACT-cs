"""Serving helper for TacQuality final-action DP guidance.

The existing DP servers usually run their control loop under
``torch.inference_mode()``.  TacQuality classifier guidance needs autograd for:

  action -> Foresight bridge -> TacQuality score -> d score / d action

This helper keeps that boundary explicit.  Call ``guide_action_chunk`` after a
DP clean action chunk is produced; it temporarily disables inference mode for
the bounded refinement step, then returns a detached guided action chunk.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TFAC_V5.distilled_tac_quality_energy_runtime import (  # noqa: E402
    DistilledTacQualityEnergyRuntime,
    TASK_TO_ID as DISTILLED_TASK_TO_ID,
)
from TFAC_V5.tac_quality_dp_guidance_controller import from_guidance_profile  # noqa: E402
from TFAC_V5.tac_quality_dp_integration_adapter import (  # noqa: E402
    ActionNormalizer,
    ForesightPredictFn,
    TacQualityDPIntegrationAdapter,
    summarize_tensor,
)
from TFAC_V5.tac_quality_guidance_runtime import TacQualityGuidanceRuntime  # noqa: E402


DEFAULT_ROLLOUT_ARM_CONFIG = Path(
    "/home/chenshuai/Project/output/tac_quality_rollout_arm_configs/tac_quality_rollout_arm_configs.json"
)


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


class DistilledTacQualityDPIntegrationAdapter:
    """DP adapter variant for the distilled TacQualityEnergy ablation scorer."""

    def __init__(
        self,
        task: str,
        *,
        scorer: DistilledTacQualityEnergyRuntime,
        action_normalizer: Optional[ActionNormalizer] = None,
    ):
        self.task = task.lower()
        self.scorer = scorer
        self.action_normalizer = action_normalizer or ActionNormalizer(mode="identity")
        self.controller = from_guidance_profile(self.task, clamp_norm_action=False)

    def score_from_prediction(self, tactile: Dict[str, torch.Tensor], action_raw: torch.Tensor) -> torch.Tensor:
        task_id = torch.full(
            (action_raw.shape[0],),
            DISTILLED_TASK_TO_ID[self.task],
            dtype=torch.long,
            device=self.scorer.device,
        )
        return self.scorer.score(
            tactile["left_marker_seq"],
            right_marker_seq=tactile.get("right_marker_seq"),
            eef_action_seq=tactile.get("eef_action_seq"),
            joint_action_seq=action_raw,
            task_id=task_id,
            mode="energy_clipped",
        )

    def guide_final_action(
        self,
        action_norm: torch.Tensor,
        foresight_predict_fn: ForesightPredictFn,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        action_raw = self.action_normalizer.denormalize(action_norm).detach()

        def current_score_fn(candidate_raw: torch.Tensor) -> torch.Tensor:
            tactile = foresight_predict_fn(candidate_raw)
            return self.score_from_prediction(tactile, candidate_raw)

        guided_raw, report = self.controller.guide(action_raw, current_score_fn)
        guided_norm = self.action_normalizer.normalize(guided_raw)
        base_score = current_score_fn(action_raw.detach().clone().requires_grad_(True)).detach()
        guided_score = current_score_fn(guided_raw.detach().clone().requires_grad_(True)).detach()
        report.update(
            {
                "task": self.task,
                "score_mode": "energy_clipped",
                "adapter_policy": "final_clean_action_trust_region_refinement",
                "scorer_runtime": "DistilledTacQualityEnergyRuntime",
                "base_score_recomputed": summarize_tensor(base_score),
                "guided_score_recomputed": summarize_tensor(guided_score),
                "raw_action_delta": summarize_tensor((guided_raw - action_raw).flatten(1).norm(dim=1)),
                "normalized_action_delta": summarize_tensor((guided_norm - action_norm).flatten(1).norm(dim=1)),
                "action_normalizer": self.action_normalizer.summary(),
                "integration_contract": {
                    "foresight_predict_fn_input": "raw action tensor, shape (B,H,A)",
                    "foresight_predict_fn_output": "dict with left_marker_seq and optional right_marker_seq/eef_action_seq",
                    "guidance_location": "after DP denoising has produced clean/final action",
                    "reranking": False,
                    "every_step_ddpm_guidance": False,
                    "recompute_foresight_each_guidance_call": True,
                },
            }
        )
        return guided_norm.detach(), report


@dataclass
class TacQualityServingGuidance:
    """Small serving-facing wrapper around a task/arm adapter."""

    task: str
    arm: str
    scorer_runtime: str
    adapter: Any

    def guide_action_chunk(
        self,
        action_norm: torch.Tensor,
        foresight_predict_fn: ForesightPredictFn,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        was_inference_mode = torch.is_inference_mode_enabled()
        with torch.inference_mode(False):
            with torch.enable_grad():
                action_for_grad = action_norm.detach().clone()
                guided, report = self.adapter.guide_final_action(action_for_grad, foresight_predict_fn)
        report.update(
            {
                "serving_helper": "TacQualityServingGuidance",
                "task": self.task,
                "arm": self.arm,
                "scorer_runtime": self.scorer_runtime,
                "called_from_inference_mode": bool(was_inference_mode),
                "returned_requires_grad": bool(guided.requires_grad),
            }
        )
        return guided.detach(), report


def build_action_normalizer(dp_norm_stats: Mapping[str, Any], mode: str = "minmax") -> ActionNormalizer:
    return ActionNormalizer.from_norm_stats(dp_norm_stats, mode=mode)


def build_serving_guidance_from_arm(
    task: str,
    arm_name: str,
    *,
    dp_norm_stats: Mapping[str, Any],
    rollout_config: Mapping[str, Any],
    device: str = "cuda:0",
    norm_mode: str = "minmax",
    runtime: Optional[TacQualityGuidanceRuntime] = None,
) -> TacQualityServingGuidance:
    arm = rollout_config["tasks"][task][arm_name]
    if not arm.get("guidance_enabled", False):
        raise ValueError(f"Arm {task}/{arm_name} has guidance_enabled=false")
    normalizer = build_action_normalizer(dp_norm_stats, mode=norm_mode)
    scorer_runtime = arm["scorer_runtime"]
    if scorer_runtime in {"InsertionRiskScorerRuntime", "PTGProxyScorerV2Runtime"}:
        runtime = runtime or TacQualityGuidanceRuntime(device=device)
        adapter = TacQualityDPIntegrationAdapter(task, runtime=runtime, action_normalizer=normalizer)
    elif scorer_runtime == "DistilledTacQualityEnergyRuntime":
        scorer = DistilledTacQualityEnergyRuntime(arm["checkpoint"]["path"], device=device)
        adapter = DistilledTacQualityDPIntegrationAdapter(task, scorer=scorer, action_normalizer=normalizer)
    else:
        raise KeyError(f"Unsupported scorer runtime: {scorer_runtime}")
    return TacQualityServingGuidance(
        task=task,
        arm=arm_name,
        scorer_runtime=scorer_runtime,
        adapter=adapter,
    )


def load_rollout_arm_config(path: Path = DEFAULT_ROLLOUT_ARM_CONFIG) -> Dict[str, Any]:
    return load_json(path)
