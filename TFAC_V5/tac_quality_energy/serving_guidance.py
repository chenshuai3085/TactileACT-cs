"""Serving helper for final-action TacQuality classifier guidance."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import torch

from .model import TASK_TO_ID
from .board_proxy_energy import BoardProxyEnergyRuntime
from .force_band_runtime import ForceBandTacQualityEnergyRuntime
from .insertion_runtime import InsertionRiskScorerRuntime
from .ptg_proxy_runtime import PTGProxyScorerV2Runtime
from .runtime import DistilledTacQualityEnergyRuntime
from .trust_region import TacQualityTrustRegionRefiner, TrustRegionConfig, summarize_tensor


DEFAULT_ROLLOUT_ARM_CONFIG = Path(
    "/home/chenshuai/Project/output/tac_quality_rollout_arm_configs/tac_quality_rollout_arm_configs.json"
)


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@dataclass(frozen=True)
class ActionNormalizer:
    action_min: Optional[torch.Tensor] = None
    action_max: Optional[torch.Tensor] = None
    action_mean: Optional[torch.Tensor] = None
    action_std: Optional[torch.Tensor] = None
    mode: str = "identity"

    @staticmethod
    def _tensor_or_none(value) -> Optional[torch.Tensor]:
        if value is None:
            return None
        return torch.as_tensor(value, dtype=torch.float32)

    @classmethod
    def from_norm_stats(cls, norm_stats: Mapping[str, object], mode: str = "minmax") -> "ActionNormalizer":
        mode = mode.lower()
        if mode == "identity":
            return cls(mode="identity")
        if mode == "minmax":
            return cls(
                action_min=cls._tensor_or_none(norm_stats.get("action_min")),
                action_max=cls._tensor_or_none(norm_stats.get("action_max")),
                mode="minmax",
            )
        if mode == "standard":
            return cls(
                action_mean=cls._tensor_or_none(norm_stats.get("action_mean")),
                action_std=cls._tensor_or_none(norm_stats.get("action_std")),
                mode="standard",
            )
        raise ValueError(f"Unknown action normalization mode: {mode}")

    def denormalize(self, action: torch.Tensor) -> torch.Tensor:
        if self.mode == "identity":
            return action
        if self.mode == "minmax":
            if self.action_min is None or self.action_max is None:
                raise ValueError("minmax normalizer requires action_min and action_max")
            lo = self.action_min.to(action.device).view(1, 1, -1)
            hi = self.action_max.to(action.device).view(1, 1, -1)
            return (action + 1.0) * 0.5 * (hi - lo) + lo
        if self.mode == "standard":
            if self.action_mean is None or self.action_std is None:
                raise ValueError("standard normalizer requires action_mean and action_std")
            mean = self.action_mean.to(action.device).view(1, 1, -1)
            std = self.action_std.to(action.device).view(1, 1, -1)
            return action * std + mean
        raise ValueError(f"Unknown action normalization mode: {self.mode}")

    def normalize(self, action_raw: torch.Tensor) -> torch.Tensor:
        if self.mode == "identity":
            return action_raw
        if self.mode == "minmax":
            if self.action_min is None or self.action_max is None:
                raise ValueError("minmax normalizer requires action_min and action_max")
            lo = self.action_min.to(action_raw.device).view(1, 1, -1)
            hi = self.action_max.to(action_raw.device).view(1, 1, -1)
            return 2.0 * (action_raw - lo) / (hi - lo).clamp_min(1e-8) - 1.0
        if self.mode == "standard":
            if self.action_mean is None or self.action_std is None:
                raise ValueError("standard normalizer requires action_mean and action_std")
            mean = self.action_mean.to(action_raw.device).view(1, 1, -1)
            std = self.action_std.to(action_raw.device).view(1, 1, -1)
            return (action_raw - mean) / std.clamp_min(1e-8)
        raise ValueError(f"Unknown action normalization mode: {self.mode}")

    def summary(self) -> Dict[str, object]:
        return {
            "mode": self.mode,
            "has_action_min": self.action_min is not None,
            "has_action_max": self.action_max is not None,
            "has_action_mean": self.action_mean is not None,
            "has_action_std": self.action_std is not None,
        }


def _refiner_config(arm: Mapping[str, Any]) -> TrustRegionConfig:
    ref = arm.get("refiner", {}).get("refinement", {})
    return TrustRegionConfig(
        steps=int(ref.get("refine_steps", 4)),
        step_size=float(ref.get("action_step", 0.0002)),
        max_total_delta=float(ref.get("max_total_delta", 0.02)),
        accept_only_improved=bool(ref.get("accept_only_improved", True)),
    )


class EnergyGuidanceAdapter:
    """Final-clean-action adapter around package scorer runtimes."""

    def __init__(
        self,
        task: str,
        *,
        scorer: torch.nn.Module,
        action_normalizer: ActionNormalizer,
        config: TrustRegionConfig,
        score_mode: str = "energy_clipped",
        profile_energy: Optional[Mapping[str, Any]] = None,
    ):
        self.task = task.lower()
        self.scorer = scorer
        self.action_normalizer = action_normalizer
        self.refiner = TacQualityTrustRegionRefiner(config)
        self.score_mode = score_mode
        self.profile_energy = dict(profile_energy or {})

    def _profile_score(self, tactile: Dict[str, torch.Tensor], action_raw: torch.Tensor, task_id: torch.Tensor) -> torch.Tensor:
        if hasattr(self.scorer, "weighted_energy_score"):
            return self.scorer.weighted_energy_score(
                tactile["left_marker_seq"],
                right_marker_seq=tactile.get("right_marker_seq"),
                eef_action_seq=tactile.get("eef_action_seq"),
                joint_action_seq=action_raw,
                task_id=task_id,
                quality_weight=float(self.profile_energy.get("quality", 0.75)),
                binary_weight=float(self.profile_energy.get("binary_margin", 0.10)),
                reason_weight=float(self.profile_energy.get("reason_margin", 0.0)),
                clip=True,
            )
        return self.scorer.score(
            tactile["left_marker_seq"],
            right_marker_seq=tactile.get("right_marker_seq"),
            eef_action_seq=tactile.get("eef_action_seq"),
            joint_action_seq=action_raw,
            task_id=task_id,
            mode="energy_clipped",
        )

    def score_from_prediction(self, tactile: Dict[str, torch.Tensor], action_raw: torch.Tensor) -> torch.Tensor:
        task_id = torch.full(
            (action_raw.shape[0],),
            TASK_TO_ID.get(self.task, 1),
            dtype=torch.long,
            device=action_raw.device,
        )
        if self.score_mode == "profile":
            return self._profile_score(tactile, action_raw, task_id)
        return self.scorer.score(
            tactile["left_marker_seq"],
            right_marker_seq=tactile.get("right_marker_seq"),
            eef_action_seq=tactile.get("eef_action_seq"),
            joint_action_seq=action_raw,
            task_id=task_id,
            mode=self.score_mode,
        )

    def guide_final_action(self, action_norm: torch.Tensor, foresight_predict_fn) -> Tuple[torch.Tensor, Dict[str, object]]:
        action_raw = self.action_normalizer.denormalize(action_norm).detach()

        def score_fn(candidate_raw: torch.Tensor) -> torch.Tensor:
            tactile = foresight_predict_fn(candidate_raw)
            return self.score_from_prediction(tactile, candidate_raw)

        guided_raw, report = self.refiner.refine(action_raw, score_fn)
        guided_norm = self.action_normalizer.normalize(guided_raw)
        report.update(
            {
                "task": self.task,
                "score_mode": self.score_mode,
                "adapter_policy": "final_clean_action_trust_region_refinement",
                "scorer_runtime": type(self.scorer).__name__,
                "raw_action_delta": summarize_tensor((guided_raw - action_raw).flatten(1).norm(dim=1)),
                "normalized_action_delta": summarize_tensor((guided_norm - action_norm).flatten(1).norm(dim=1)),
                "action_normalizer": self.action_normalizer.summary(),
                "profile_energy": self.profile_energy if self.score_mode == "profile" else None,
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
    task: str
    arm: str
    scorer_runtime: str
    adapter: EnergyGuidanceAdapter

    def guide_action_chunk(self, action_norm: torch.Tensor, foresight_predict_fn) -> Tuple[torch.Tensor, Dict[str, object]]:
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
                "configured_scorer_runtime": self.scorer_runtime,
                "called_from_inference_mode": bool(was_inference_mode),
                "returned_requires_grad": bool(guided.requires_grad),
            }
        )
        return guided.detach(), report


def build_serving_guidance_from_arm(
    task: str,
    arm_name: str,
    *,
    dp_norm_stats: Mapping[str, Any],
    rollout_config: Mapping[str, Any],
    device: str = "cuda:0",
    norm_mode: str = "minmax",
) -> TacQualityServingGuidance:
    arm = rollout_config["tasks"][task][arm_name]
    if not arm.get("guidance_enabled", False):
        raise ValueError(f"Arm {task}/{arm_name} has guidance_enabled=false")
    normalizer = ActionNormalizer.from_norm_stats(dp_norm_stats, mode=norm_mode)
    runtime_name = arm["scorer_runtime"]
    checkpoint = arm["checkpoint"]["path"]
    refiner = arm.get("refiner", {})
    configured_score_mode = str(refiner.get("score_mode", "energy_clipped"))
    if runtime_name == "PTGProxyScorerV2Runtime":
        scorer = PTGProxyScorerV2Runtime(checkpoint, device=device)
    elif runtime_name == "ForceBandTacQualityEnergyRuntime":
        scorer = ForceBandTacQualityEnergyRuntime(checkpoint, device=device)
    elif runtime_name == "BoardProxyEnergyRuntime":
        scorer = BoardProxyEnergyRuntime(device=device)
    elif runtime_name == "InsertionRiskScorerRuntime":
        scorer = InsertionRiskScorerRuntime(checkpoint, device=device)
    elif runtime_name == "DistilledTacQualityEnergyRuntime":
        scorer = DistilledTacQualityEnergyRuntime(checkpoint, device=device)
    else:
        raise KeyError(
            f"Unsupported scorer runtime {runtime_name!r} in package serving helper. "
            "Currently supported: InsertionRiskScorerRuntime, PTGProxyScorerV2Runtime, "
            "ForceBandTacQualityEnergyRuntime, DistilledTacQualityEnergyRuntime."
        )
    adapter = EnergyGuidanceAdapter(
        task,
        scorer=scorer,
        action_normalizer=normalizer,
        config=_refiner_config(arm),
        score_mode=configured_score_mode,
        profile_energy=refiner.get("energy"),
    )
    return TacQualityServingGuidance(task=task, arm=arm_name, scorer_runtime=runtime_name, adapter=adapter)


def load_rollout_arm_config(path: Path = DEFAULT_ROLLOUT_ARM_CONFIG) -> Dict[str, Any]:
    return load_json(path)
