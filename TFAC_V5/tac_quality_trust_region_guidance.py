"""Trust-region action guidance utilities for TacQuality scores.

The scorer runtime answers "which direction improves tactile quality".  This
module answers the deployment question "how far do we move the DP action, and
when do we accept the update?"  It is intentionally independent of a specific
task, Foresight model, or DP implementation.

Usage:
  refiner = TacQualityTrustRegionRefiner(...)
  refined, report = refiner.refine(action, score_fn)

where score_fn(action) returns one differentiable scalar per batch item.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TFAC_V5.tac_quality_guidance_config import get_guidance_profile  # noqa: E402


DEFAULT_OUT = "/home/chenshuai/Project/output/tac_quality_trust_region_guidance/trust_region_sanity.json"


ScoreFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class TrustRegionConfig:
    steps: int
    step_size: float
    max_total_delta: float
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


def unit_gradient_step(action: torch.Tensor, grad: torch.Tensor, step_size: float, min_grad_norm: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
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
    """Small accepted gradient-ascent optimizer for DP action tensors."""

    def __init__(self, config: TrustRegionConfig):
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
            if cfg.accept_only_improved:
                accept = score_new.detach() > score.detach()
            else:
                accept = torch.ones_like(score, dtype=torch.bool)
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
            "delta_norm": summarize_tensor(delta_norm),
            "max_delta_within_trust_region": bool(delta_norm.max().item() <= cfg.max_total_delta + 1e-6 if cfg.max_total_delta > 0 else True),
            "logs": logs,
        }
        return current.detach(), report


def from_guidance_profile(task: str, *, clamp_norm_action: bool = False) -> TacQualityTrustRegionRefiner:
    profile = get_guidance_profile(task)
    refine = profile.refinement
    config = TrustRegionConfig(
        steps=refine.refine_steps,
        step_size=refine.action_step,
        max_total_delta=refine.max_total_delta,
        accept_only_improved=refine.accept_only_improved,
        clamp_min=-1.0 if clamp_norm_action else None,
        clamp_max=1.0 if clamp_norm_action else None,
    )
    return TacQualityTrustRegionRefiner(config)


def quadratic_score(target: torch.Tensor) -> ScoreFn:
    def score_fn(action: torch.Tensor) -> torch.Tensor:
        return -torch.square(action - target).flatten(1).mean(dim=1)

    return score_fn


def sanity(args) -> Dict[str, object]:
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    batch = args.batch_size
    horizon = args.horizon
    dim = args.action_dim
    base = torch.randn(batch, horizon, dim, device=device) * 0.25
    target = base + torch.randn_like(base) * 0.2

    insertion_refiner = from_guidance_profile("insertion", clamp_norm_action=True)
    insertion_refined, insertion_report = insertion_refiner.refine(base, quadratic_score(target))

    board_base = torch.randn(batch, horizon, dim, device=device) * 0.05
    board_target = board_base + torch.randn_like(board_base) * 0.02
    board_refiner = from_guidance_profile("board", clamp_norm_action=True)
    board_refined, board_report = board_refiner.refine(board_base, quadratic_score(board_target))

    result = {
        "purpose": "Trust-region action update sanity for TacQuality classifier guidance.",
        "device": str(device),
        "insertion": insertion_report,
        "board": board_report,
        "smoothness": {
            "insertion_base": summarize_tensor(action_smoothness(base)),
            "insertion_refined": summarize_tensor(action_smoothness(insertion_refined)),
            "board_base": summarize_tensor(action_smoothness(board_base)),
            "board_refined": summarize_tensor(action_smoothness(board_refined)),
        },
        "passes_trust_region_guidance_sanity": bool(
            insertion_report["improved_rate"] >= 0.99
            and board_report["improved_rate"] >= 0.99
            and insertion_report["max_delta_within_trust_region"]
            and board_report["max_delta_within_trust_region"]
        ),
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=DEFAULT_OUT)
    return parser.parse_args()


if __name__ == "__main__":
    sanity(parse_args())
