"""Flow-step tactile foresight guidance utilities for pi0/pi0.5.

The pi0.5 checkpoints commonly keep a 32-D model action space while many robot
datasets execute only the first 7 or 8 dimensions.  This module keeps the flow
state in the full model space, but lets tactile foresight/scoring read and
modify only the executable robot action dimensions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch


ScoreFn = Callable[[torch.Tensor], torch.Tensor]


def _summarize_tensor(x: torch.Tensor) -> Dict[str, float]:
    arr = x.detach().float().flatten()
    if arr.numel() == 0:
        return {"n": 0}
    return {
        "n": int(arr.numel()),
        "mean": float(arr.mean().cpu()),
        "std": float(arr.std(unbiased=False).cpu()),
        "min": float(arr.min().cpu()),
        "max": float(arr.max().cpu()),
    }


@dataclass(frozen=True)
class Pi0ActionAdapter:
    """Maps between pi0.5 model actions and executable robot actions."""

    model_action_dim: int = 32
    robot_action_dim: int = 7
    pad_value: float = 0.0

    def __post_init__(self):
        if self.model_action_dim <= 0:
            raise ValueError("model_action_dim must be positive")
        if self.robot_action_dim <= 0:
            raise ValueError("robot_action_dim must be positive")
        if self.robot_action_dim > self.model_action_dim:
            raise ValueError(
                f"robot_action_dim={self.robot_action_dim} exceeds "
                f"model_action_dim={self.model_action_dim}"
            )

    def slice_robot_action(self, action_model: torch.Tensor) -> torch.Tensor:
        """Return the executable robot dimensions from a full model action."""
        if action_model.shape[-1] < self.robot_action_dim:
            raise ValueError(
                f"Cannot slice {self.robot_action_dim} dims from action shape "
                f"{tuple(action_model.shape)}"
            )
        return action_model[..., : self.robot_action_dim]

    def pad_robot_action(self, action_robot: torch.Tensor) -> torch.Tensor:
        """Zero-pad an executable action/state to the full model action dim."""
        dim = action_robot.shape[-1]
        if dim == self.model_action_dim:
            return action_robot
        if dim != self.robot_action_dim:
            raise ValueError(
                f"Expected last dim {self.robot_action_dim} or "
                f"{self.model_action_dim}, got {dim}"
            )
        pad_shape = (*action_robot.shape[:-1], self.model_action_dim - dim)
        pad = torch.full(
            pad_shape,
            float(self.pad_value),
            dtype=action_robot.dtype,
            device=action_robot.device,
        )
        return torch.cat([action_robot, pad], dim=-1)

    def robot_mask_like(self, action_model: torch.Tensor) -> torch.Tensor:
        """Mask with ones on robot dimensions and zeros on padded dimensions."""
        if action_model.shape[-1] != self.model_action_dim:
            raise ValueError(
                f"Expected model action dim {self.model_action_dim}, got "
                f"{action_model.shape[-1]}"
            )
        mask = torch.zeros_like(action_model)
        mask[..., : self.robot_action_dim] = 1.0
        return mask


@dataclass(frozen=True)
class Pi0FlowGuidanceConfig:
    """Configuration for late-flow-step tactile guidance."""

    guidance_steps: int = 0
    guidance_scale: float = 0.0
    max_total_delta: float = 0.08
    normalize_grad: bool = True
    max_grad_norm: float = 1.0
    accept_only_improved: bool = True
    clamp_min: Optional[float] = None
    clamp_max: Optional[float] = None
    lambda_smooth: float = 0.0
    min_grad_norm: float = 1e-8
    detach_velocity: bool = True
    recompute_velocity_after_guidance: bool = True

    @property
    def enabled(self) -> bool:
        return self.guidance_steps > 0 and self.guidance_scale > 0.0


def action_smoothness(actions: torch.Tensor) -> torch.Tensor:
    """Second-difference smoothness penalty for action chunks."""
    if actions.shape[1] < 3:
        return torch.zeros(actions.shape[0], dtype=actions.dtype, device=actions.device)
    accel = actions[:, 2:] - 2.0 * actions[:, 1:-1] + actions[:, :-2]
    return torch.linalg.norm(accel, dim=-1).mean(dim=1)


def _project_robot_trust_region(
    proposal: torch.Tensor,
    base: torch.Tensor,
    mask: torch.Tensor,
    max_total_delta: float,
) -> torch.Tensor:
    if max_total_delta <= 0.0:
        return proposal
    delta = (proposal - base) * mask
    norm = delta.flatten(1).norm(dim=1).view(-1, *([1] * (delta.ndim - 1))).clamp_min(1e-8)
    scale = torch.clamp(float(max_total_delta) / norm, max=1.0)
    return base + delta * scale


class Pi0FlowStepGuidance:
    """Late-step guidance for pi0/pi0.5 flow matching action generation.

    The clean action estimate follows the pi0 flow convention:

    ``x0_est = x_t - t * v_t``.

    By default ``v_t`` is detached, matching the DP guidance policy where the
    model prediction is treated as a local clean-action estimator and gradients
    update only the sampled action trajectory.
    """

    def __init__(
        self,
        config: Pi0FlowGuidanceConfig,
        action_adapter: Pi0ActionAdapter,
    ):
        self.config = config
        self.action_adapter = action_adapter

    def should_guide(self, step_idx: int, total_steps: int) -> bool:
        if not self.config.enabled:
            return False
        guide_start = max(0, int(total_steps) - int(self.config.guidance_steps))
        return int(step_idx) >= guide_start

    def clean_action_estimate(
        self,
        x_t: torch.Tensor,
        v_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        v_for_clean = v_t.detach() if self.config.detach_velocity else v_t
        return x_t - timestep.view(-1, 1, 1).to(x_t.dtype) * v_for_clean

    def _score_clean_action(self, clean_action_model: torch.Tensor, score_fn: ScoreFn) -> torch.Tensor:
        robot_action = self.action_adapter.slice_robot_action(clean_action_model)
        score = score_fn(robot_action)
        if score.ndim == 0:
            score = score.expand(robot_action.shape[0])
        if score.ndim != 1 or score.shape[0] != robot_action.shape[0]:
            raise ValueError(
                f"score_fn must return shape (B,), got {tuple(score.shape)} "
                f"for robot action {tuple(robot_action.shape)}"
            )
        return score

    def _masked_update(self, grad: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0) * mask
        grad_norm = grad.flatten(1).norm(dim=1)
        norm_view = grad_norm.view(-1, *([1] * (grad.ndim - 1))).clamp_min(self.config.min_grad_norm)
        if self.config.normalize_grad:
            update = grad / norm_view
            clipped_norm = torch.ones_like(grad_norm)
        else:
            coef = (float(self.config.max_grad_norm) / norm_view).clamp(max=1.0)
            update = grad * coef
            clipped_norm = grad_norm * coef.flatten()
        return update, grad_norm, clipped_norm

    def guide(
        self,
        x_t: torch.Tensor,
        v_t: torch.Tensor,
        timestep: torch.Tensor,
        *,
        step_idx: int,
        total_steps: int,
        score_fn: ScoreFn,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        """Optionally update ``x_t`` using tactile foresight score gradients."""
        if not self.should_guide(step_idx, total_steps):
            return x_t, {"applied": False, "step_idx": int(step_idx)}

        base = x_t.detach()
        x = base.clone().requires_grad_(True)
        v_for_clean = v_t.detach() if self.config.detach_velocity else v_t
        clean = x - timestep.view(-1, 1, 1).to(x.dtype) * v_for_clean
        score = self._score_clean_action(clean, score_fn)

        robot_clean = self.action_adapter.slice_robot_action(clean)
        smooth_penalty = action_smoothness(robot_clean)
        objective = score - float(self.config.lambda_smooth) * smooth_penalty
        grad = torch.autograd.grad(objective.sum(), x, retain_graph=False)[0]

        mask = self.action_adapter.robot_mask_like(x)
        update, grad_norm, clipped_norm = self._masked_update(grad, mask)
        proposal = x + float(self.config.guidance_scale) * update
        proposal = _project_robot_trust_region(
            proposal,
            base,
            mask,
            float(self.config.max_total_delta),
        )
        if self.config.clamp_min is not None or self.config.clamp_max is not None:
            lo = -float("inf") if self.config.clamp_min is None else float(self.config.clamp_min)
            hi = float("inf") if self.config.clamp_max is None else float(self.config.clamp_max)
            clamped = proposal.clamp(lo, hi)
            proposal = torch.where(mask.bool(), clamped, base)

        with torch.no_grad():
            proposal_clean = proposal - timestep.view(-1, 1, 1).to(proposal.dtype) * v_for_clean
            score_new = self._score_clean_action(proposal_clean, score_fn)
            finite = torch.isfinite(grad).flatten(1).all(dim=1)
            if self.config.accept_only_improved:
                accept = (score_new > score.detach()) & finite
            else:
                accept = finite
            guided = torch.where(accept.view(-1, *([1] * (x.ndim - 1))), proposal, base)

        delta = (guided - base) * mask
        report = {
            "applied": True,
            "step_idx": int(step_idx),
            "timestep_mean": float(timestep.detach().float().mean().cpu()),
            "score": _summarize_tensor(score),
            "score_after": _summarize_tensor(score_new),
            "score_delta": _summarize_tensor(score_new - score.detach()),
            "accept_rate": float(accept.float().mean().cpu()),
            "finite_grad_rate": float(finite.float().mean().cpu()),
            "positive_grad_rate": float((grad_norm > self.config.min_grad_norm).float().mean().cpu()),
            "grad_norm": _summarize_tensor(grad_norm),
            "clipped_grad_norm": _summarize_tensor(clipped_norm),
            "applied_delta_norm": _summarize_tensor(delta.flatten(1).norm(dim=1)),
            "smoothness_penalty": _summarize_tensor(smooth_penalty),
        }
        return guided.detach(), report

    def summarize(self, reports: List[Dict[str, object]]) -> Dict[str, object]:
        applied = [r for r in reports if r.get("applied")]
        if not applied:
            return {
                "enabled": self.config.enabled,
                "applied_steps": 0,
                "config": asdict(self.config),
                "action_adapter": asdict(self.action_adapter),
                "steps": reports,
            }
        return {
            "enabled": self.config.enabled,
            "applied_steps": len(applied),
            "accept_rate_mean": float(sum(float(r["accept_rate"]) for r in applied) / len(applied)),
            "finite_grad_rate_mean": float(sum(float(r["finite_grad_rate"]) for r in applied) / len(applied)),
            "positive_grad_rate_mean": float(sum(float(r["positive_grad_rate"]) for r in applied) / len(applied)),
            "config": asdict(self.config),
            "action_adapter": asdict(self.action_adapter),
            "steps": reports,
            "integration_contract": {
                "flow_clean_action_estimate": "x0_est = x_t - t * v_t",
                "gradient_target": "x_t",
                "velocity_gradient_policy": "detach_v_t" if self.config.detach_velocity else "through_v_t",
                "guided_dims": f"first {self.action_adapter.robot_action_dim} model action dimensions",
                "padded_dims": "not scored and receive zero guidance gradient",
                "reranking": False,
            },
        }
