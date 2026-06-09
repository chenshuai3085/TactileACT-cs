"""Central configuration for TacQualityEnergy guidance.

The experiments in this repository use several scorer runtimes, but the
guidance objective should stay consistent:

    energy = wq * quality_logit + wb * binary_margin + wr * reason_margin
    energy_clipped = clip_scale * tanh(energy / clip_scale)

This module records the currently validated task-specific coefficients and
trust-region defaults.  It is intentionally lightweight so evaluation scripts
and future DP inference code can share the same guidance contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

import torch


@dataclass(frozen=True)
class EnergyWeights:
    quality: float
    binary_margin: float
    reason_margin: float
    clip_scale: float = 4.0


@dataclass(frozen=True)
class RefinementDefaults:
    refine_steps: int
    marker_step: float
    action_step: float
    max_total_delta: float
    smooth_weight: float
    joint_limit_weight: float
    joint_margin_frac: float
    accept_only_improved: bool = True


@dataclass(frozen=True)
class GuidanceProfile:
    name: str
    task_id: int
    scorer: str
    energy: EnergyWeights
    refinement: RefinementDefaults
    evidence: str
    scope: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


GUIDANCE_PROFILES: Dict[str, GuidanceProfile] = {
    "insertion": GuidanceProfile(
        name="insertion",
        task_id=0,
        scorer="InsertionRiskScorerRuntime",
        energy=EnergyWeights(quality=0.50, binary_margin=0.10, reason_margin=0.0),
        refinement=RefinementDefaults(
            refine_steps=4,
            marker_step=0.0,
            action_step=0.02,
            max_total_delta=0.08,
            smooth_weight=0.02,
            joint_limit_weight=10.0,
            joint_margin_frac=0.03,
        ),
        evidence="full-chain gradient + constrained clean-action refinement passed on socket insertion",
        scope="Validated with insertion DP/Foresight. Use as default insertion clean-action guidance.",
    ),
    "board": GuidanceProfile(
        name="board",
        task_id=1,
        scorer="PTGProxyScorerV2Runtime",
        energy=EnergyWeights(quality=0.75, binary_margin=0.10, reason_margin=0.0),
        refinement=RefinementDefaults(
            refine_steps=4,
            marker_step=0.005,
            action_step=0.0002,
            max_total_delta=0.02,
            smooth_weight=0.02,
            joint_limit_weight=0.0,
            joint_margin_frac=0.03,
        ),
        evidence="scorer-level board guidance readiness passed on real board windows",
        scope="Scorer-level validated. Full-chain use requires board-specific Foresight/DP.",
    ),
    "mixed": GuidanceProfile(
        name="mixed",
        task_id=-1,
        scorer="PTGProxyScorerV2Runtime",
        energy=EnergyWeights(quality=0.50, binary_margin=0.0, reason_margin=0.20),
        refinement=RefinementDefaults(
            refine_steps=4,
            marker_step=0.001,
            action_step=0.0002,
            max_total_delta=0.02,
            smooth_weight=0.02,
            joint_limit_weight=0.0,
            joint_margin_frac=0.03,
        ),
        evidence="energy coefficient search favored reason margin for mixed-task semantics",
        scope="Use for analysis or task-conditioned fallback; prefer task-specific profiles for deployment.",
    ),
}


def get_guidance_profile(task: str) -> GuidanceProfile:
    key = task.lower()
    if key not in GUIDANCE_PROFILES:
        raise KeyError(f"Unknown guidance profile {task!r}. Available: {sorted(GUIDANCE_PROFILES)}")
    return GUIDANCE_PROFILES[key]


def weighted_logit_energy(
    quality_logit: torch.Tensor,
    binary_margin: torch.Tensor,
    reason_margin: torch.Tensor,
    weights: EnergyWeights,
    clip: bool = True,
) -> torch.Tensor:
    energy = (
        weights.quality * quality_logit
        + weights.binary_margin * binary_margin
        + weights.reason_margin * reason_margin
    )
    if clip:
        return torch.tanh(energy / weights.clip_scale) * weights.clip_scale
    return energy


def profile_summary() -> Dict[str, Dict[str, object]]:
    return {name: profile.to_dict() for name, profile in GUIDANCE_PROFILES.items()}


def main():
    import json

    print(json.dumps(profile_summary(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
