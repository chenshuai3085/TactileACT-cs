"""Trust-region action updates for TacQualityEnergy classifier guidance."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch


ScoreFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class TrustRegionConfig:
    steps: int = 4
    step_size: float = 0.02
    max_total_delta: float = 0.08
    accept_only_improved: bool = True
    clamp_min: Optional[float] = None
    clamp_max: Optional[float] = None
    min_grad_norm: float = 1e-8


def action_smoothness(actions: torch.Tensor) -> torch.Tensor:
    if actions.shape[1] < 3:
        return torch.zeros(actions.shape[0], dtype=actions.dtype, device=actions.device)
    accel = actions[:, 2:] - 2 * actions[:, 1:-1] + actions[:, :-2]
    return torch.linalg.norm(accel, dim=-1).mean(dim=1)


def project_trust_region(proposal: torch.Tensor, base: torch.Tensor, max_total_delta: float) -> torch.Tensor:
    if max_total_delta <= 0:
        return proposal
    delta = proposal - base
    norm = delta.flatten(1).norm(dim=1).view(-1, *([1] * (delta.ndim - 1))).clamp_min(1e-8)
    scale = torch.clamp(max_total_delta / norm, max=1.0)
    return base + delta * scale


def unit_gradient_step(
    action: torch.Tensor,
    grad: torch.Tensor,
    step_size: float,
    min_grad_norm: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    grad_norm = grad.flatten(1).norm(dim=1)
    grad_unit = grad / grad_norm.view(-1, *([1] * (grad.ndim - 1))).clamp_min(min_grad_norm)
    return action + step_size * grad_unit, grad_norm


def summarize_tensor(x: torch.Tensor) -> Dict[str, float]:
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


class TacQualityTrustRegionRefiner:
    """Accepted gradient-ascent optimizer for DP action tensors."""

    def __init__(self, config: TrustRegionConfig = TrustRegionConfig()):
        self.config = config

    def refine(self, action: torch.Tensor, score_fn: ScoreFn) -> Tuple[torch.Tensor, Dict[str, object]]:
        cfg = self.config
        base = action.detach()
        current = base.clone()
        logs: List[Dict[str, float]] = []

        for step_idx in range(cfg.steps):
            x = current.detach().clone().requires_grad_(True)
            score = score_fn(x)
            if score.ndim != 1 or score.shape[0] != x.shape[0]:
                raise ValueError(f"score_fn must return shape (B,), got {tuple(score.shape)} for action {tuple(x.shape)}")
            grad = torch.autograd.grad(score.sum(), x, retain_graph=False)[0]
            proposal, grad_norm = unit_gradient_step(x, grad, cfg.step_size, cfg.min_grad_norm)
            proposal = project_trust_region(proposal, base, cfg.max_total_delta)
            if cfg.clamp_min is not None or cfg.clamp_max is not None:
                lo = -float("inf") if cfg.clamp_min is None else cfg.clamp_min
                hi = float("inf") if cfg.clamp_max is None else cfg.clamp_max
                proposal = proposal.clamp(lo, hi)

            score_new = score_fn(proposal.detach().clone().requires_grad_(True))
            accept = score_new.detach() > score.detach() if cfg.accept_only_improved else torch.ones_like(score, dtype=torch.bool)
            with torch.no_grad():
                current = torch.where(accept.view(-1, *([1] * (x.ndim - 1))), proposal, current)

            logs.append(
                {
                    "step": float(step_idx),
                    "score_mean": float(score.detach().mean().cpu()),
                    "score_after_mean": float(score_new.detach().mean().cpu()),
                    "score_delta_mean": float((score_new.detach() - score.detach()).mean().cpu()),
                    "accept_rate": float(accept.float().mean().cpu()),
                    "grad_norm_mean": float(grad_norm.detach().mean().cpu()),
                    "positive_grad_rate": float((grad_norm.detach() > cfg.min_grad_norm).float().mean().cpu()),
                    "finite_grad_rate": float(torch.isfinite(grad).flatten(1).all(dim=1).float().mean().cpu()),
                }
            )

        final_score = score_fn(current.detach().clone().requires_grad_(True)).detach()
        base_score = score_fn(base.detach().clone().requires_grad_(True)).detach()
        delta = current - base
        delta_norm = delta.flatten(1).norm(dim=1)
        report = {
            "config": asdict(cfg),
            "base_score": summarize_tensor(base_score),
            "final_score": summarize_tensor(final_score),
            "score_delta": summarize_tensor(final_score - base_score),
            "improved_rate": float((final_score > base_score).float().mean().cpu()),
            "accept_rate": float(sum(row["accept_rate"] for row in logs) / len(logs)) if logs else 0.0,
            "finite_grad_rate": float(sum(row["finite_grad_rate"] for row in logs) / len(logs)) if logs else 0.0,
            "positive_grad_rate": float(sum(row["positive_grad_rate"] for row in logs) / len(logs)) if logs else 0.0,
            "delta_norm": summarize_tensor(delta_norm),
            "max_delta_within_trust_region": bool(
                delta_norm.max().item() <= cfg.max_total_delta + 1e-6 if cfg.max_total_delta > 0 else True
            ),
            "logs": logs,
        }
        return current.detach(), report
