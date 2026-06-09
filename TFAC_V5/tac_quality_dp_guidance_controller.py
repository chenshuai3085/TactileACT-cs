"""DP-facing TacQuality classifier-guidance controller.

This module is the deployment-side guardrail around TacQuality scores.  It
intentionally does not accept precomputed gradients.  Each call recomputes the
current score and d score / d action from the provided score_fn, then applies a
bounded accept-only update.  This encodes the robustness audit finding:

  recompute current gradients every guidance step; do not reuse stale gradients.
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
from TFAC_V5.tac_quality_trust_region_guidance import action_smoothness, project_trust_region  # noqa: E402


DEFAULT_OUT = "/home/chenshuai/Project/output/tac_quality_dp_guidance_controller/controller_sanity.json"


ScoreFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class DPGuidanceControllerConfig:
    task: str
    guidance_scale: float
    max_total_delta: float
    accept_only_improved: bool = True
    clamp_min: Optional[float] = None
    clamp_max: Optional[float] = None
    min_grad_norm: float = 1e-8
    stale_gradient_reuse_allowed: bool = False
    recompute_gradient_every_call: bool = True


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


class TacQualityDPGuidanceController:
    """One-step DP classifier-guidance update with guardrails.

    A DP denoising loop should call this after a denoising step:

      action = scheduler.step(...).prev_sample
      action, report = controller.guide(action, current_score_fn)

    current_score_fn must rebuild the current action -> Foresight -> predicted
    tactile -> TacQuality score computation.  Passing cached gradients is not
    supported by design.
    """

    def __init__(self, config: DPGuidanceControllerConfig):
        if config.stale_gradient_reuse_allowed:
            raise ValueError("TacQualityDPGuidanceController forbids stale gradient reuse.")
        self.config = config

    def guide(self, action: torch.Tensor, score_fn: ScoreFn) -> Tuple[torch.Tensor, Dict[str, object]]:
        cfg = self.config
        base = action.detach()
        x = base.clone().requires_grad_(True)
        score = score_fn(x)
        if score.ndim != 1 or score.shape[0] != x.shape[0]:
            raise ValueError(f"score_fn must return shape (B,), got {tuple(score.shape)}")
        grad = torch.autograd.grad(score.sum(), x, retain_graph=False)[0]
        grad_norm = grad.flatten(1).norm(dim=1)
        grad_unit = grad / grad_norm.view(-1, *([1] * (grad.ndim - 1))).clamp_min(cfg.min_grad_norm)
        proposal = x.detach() + cfg.guidance_scale * grad_unit.detach()
        proposal = project_trust_region(proposal, base, cfg.max_total_delta)
        if cfg.clamp_min is not None or cfg.clamp_max is not None:
            lo = -float("inf") if cfg.clamp_min is None else cfg.clamp_min
            hi = float("inf") if cfg.clamp_max is None else cfg.clamp_max
            proposal = proposal.clamp(lo, hi)
        proposal_score = score_fn(proposal.detach().clone().requires_grad_(True)).detach()
        if cfg.accept_only_improved:
            accept = proposal_score > score.detach()
        else:
            accept = torch.ones_like(score, dtype=torch.bool)
        guided = torch.where(accept.view(-1, *([1] * (base.ndim - 1))), proposal, base)
        final_score = score_fn(guided.detach().clone().requires_grad_(True)).detach()
        delta = guided - base
        delta_norm = delta.flatten(1).norm(dim=1)
        report = {
            "config": asdict(cfg),
            "base_score": summarize_tensor(score.detach()),
            "proposal_score": summarize_tensor(proposal_score),
            "final_score": summarize_tensor(final_score),
            "score_delta": summarize_tensor(final_score - score.detach()),
            "accept_rate": float(accept.float().mean().cpu()),
            "improved_rate": float((final_score > score.detach()).float().mean().cpu()),
            "grad_norm": summarize_tensor(grad_norm),
            "finite_grad_rate": float(torch.isfinite(grad).flatten(1).all(dim=1).float().mean().cpu()),
            "positive_grad_rate": float((grad_norm > cfg.min_grad_norm).float().mean().cpu()),
            "delta_norm": summarize_tensor(delta_norm),
            "max_delta_within_trust_region": bool(
                delta_norm.max().item() <= cfg.max_total_delta + max(1e-5, 1e-4 * cfg.max_total_delta)
                if cfg.max_total_delta > 0
                else True
            ),
            "smoothness_delta": summarize_tensor(action_smoothness(guided) - action_smoothness(base)),
            "guardrails": {
                "stale_gradient_reuse_allowed": False,
                "gradient_source": "computed inside guide() from current score_fn(action)",
                "requires_current_score_fn": True,
            },
        }
        return guided.detach(), report


def from_guidance_profile(task: str, *, scale: Optional[float] = None, clamp_norm_action: bool = False) -> TacQualityDPGuidanceController:
    profile = get_guidance_profile(task)
    refine = profile.refinement
    config = DPGuidanceControllerConfig(
        task=task,
        guidance_scale=refine.action_step if scale is None else float(scale),
        max_total_delta=refine.max_total_delta,
        accept_only_improved=refine.accept_only_improved,
        clamp_min=-1.0 if clamp_norm_action else None,
        clamp_max=1.0 if clamp_norm_action else None,
    )
    return TacQualityDPGuidanceController(config)


def quadratic_score(target: torch.Tensor) -> ScoreFn:
    def score_fn(action: torch.Tensor) -> torch.Tensor:
        return -torch.square(action - target).flatten(1).mean(dim=1)

    return score_fn


def sanity(args) -> Dict[str, object]:
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    base = torch.randn(args.batch_size, args.horizon, args.action_dim, device=device) * 0.2
    target = base + torch.randn_like(base) * 0.1
    insertion = from_guidance_profile("insertion", clamp_norm_action=False)
    guided_i, report_i = insertion.guide(base, quadratic_score(target))

    board_base = torch.randn(args.batch_size, args.horizon, args.action_dim, device=device) * 0.05
    board_target = board_base + torch.randn_like(board_base) * 0.02
    board = from_guidance_profile("board", clamp_norm_action=False)
    guided_b, report_b = board.guide(board_base, quadratic_score(board_target))

    result = {
        "purpose": "DP-facing TacQuality guidance controller sanity.",
        "device": str(device),
        "insertion": report_i,
        "board": report_b,
        "passes_controller_sanity": bool(
            report_i["improved_rate"] >= 0.99
            and report_b["improved_rate"] >= 0.99
            and report_i["finite_grad_rate"] >= 0.999
            and report_b["finite_grad_rate"] >= 0.999
            and report_i["max_delta_within_trust_region"]
            and report_b["max_delta_within_trust_region"]
            and not report_i["guardrails"]["stale_gradient_reuse_allowed"]
            and not report_b["guardrails"]["stale_gradient_reuse_allowed"]
        ),
        "smoothness": {
            "insertion_base": summarize_tensor(action_smoothness(base)),
            "insertion_guided": summarize_tensor(action_smoothness(guided_i)),
            "board_base": summarize_tensor(action_smoothness(board_base)),
            "board_guided": summarize_tensor(action_smoothness(guided_b)),
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
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
