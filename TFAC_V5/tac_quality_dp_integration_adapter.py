"""DP integration adapter for TacQuality classifier guidance.

This module is the narrow bridge between a DP policy and the TacQuality scorer.
It deliberately does not own the DP sampler or the Foresight model.  Deployment
code supplies a differentiable foresight_predict_fn(action), and this adapter
turns it into a bounded final-clean-action refinement:

  action_norm --denormalize--> action_raw
  action_raw --foresight_predict_fn--> predicted tactile consequence
  predicted tactile + action_raw --TacQuality--> score
  d score / d action_raw --trust region--> guided action_raw
  guided action_raw --normalize--> guided action_norm

The recommended production mode remains final clean-action refinement, not
reranking and not unconditional guidance at every DDPM step.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Tuple

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TFAC_V5.tac_quality_dp_guidance_controller import (  # noqa: E402
    TacQualityDPGuidanceController,
    from_guidance_profile,
)
from TFAC_V5.tac_quality_guidance_config import get_guidance_profile  # noqa: E402
from TFAC_V5.tac_quality_guidance_runtime import TacQualityGuidanceRuntime  # noqa: E402


DEFAULT_OUT = "/home/chenshuai/Project/output/tac_quality_dp_integration_adapter/integration_adapter_sanity.json"


ForesightPredictFn = Callable[[torch.Tensor], Dict[str, torch.Tensor]]


@dataclass(frozen=True)
class ActionNormalizer:
    """Convert between DP normalized action and raw robot action."""

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
        """Create an action normalizer from DP config/dataset norm_stats.

        Most DP checkpoints in this repo use min-max normalized actions in
        [-1, 1], while Foresight uses raw actions normalized separately by
        action_mean/action_std.  This factory keeps the DP side explicit.
        """

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
            lo = self.action_min.to(action_raw.device).view(1, 1, -1)
            hi = self.action_max.to(action_raw.device).view(1, 1, -1)
            return 2.0 * (action_raw - lo) / (hi - lo).clamp_min(1e-8) - 1.0
        if self.mode == "standard":
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


class TacQualityDPIntegrationAdapter:
    """Final-clean-action TacQuality guidance adapter for DP inference."""

    def __init__(
        self,
        task: str,
        *,
        runtime: TacQualityGuidanceRuntime,
        controller: Optional[TacQualityDPGuidanceController] = None,
        action_normalizer: Optional[ActionNormalizer] = None,
        score_mode: str = "profile",
    ):
        self.task = task.lower()
        self.profile = get_guidance_profile(self.task)
        self.runtime = runtime
        self.controller = controller or from_guidance_profile(self.task, clamp_norm_action=False)
        self.action_normalizer = action_normalizer or ActionNormalizer(mode="identity")
        self.score_mode = score_mode

    def score_from_prediction(self, tactile: Dict[str, torch.Tensor], action_raw: torch.Tensor) -> torch.Tensor:
        if self.task == "insertion":
            return self.runtime.score(
                "insertion",
                tactile["left_marker_seq"],
                action_raw,
                mode=self.score_mode,
            )
        if self.task == "board":
            return self.runtime.score(
                "board",
                tactile["left_marker_seq"],
                action_raw,
                right_marker_seq=tactile.get("right_marker_seq"),
                eef_action_seq=tactile.get("eef_action_seq"),
                mode=self.score_mode,
            )
        raise KeyError(f"Unknown task {self.task!r}")

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
        delta_raw = guided_raw - action_raw
        delta_norm = guided_norm - action_norm
        report.update(
            {
                "task": self.task,
                "profile": self.profile.to_dict(),
                "score_mode": self.score_mode,
                "adapter_policy": "final_clean_action_trust_region_refinement",
                "base_score_recomputed": summarize_tensor(base_score),
                "guided_score_recomputed": summarize_tensor(guided_score),
                "raw_action_delta": summarize_tensor(delta_raw.flatten(1).norm(dim=1)),
                "normalized_action_delta": summarize_tensor(delta_norm.flatten(1).norm(dim=1)),
                "action_normalizer": self.action_normalizer.summary(),
                "integration_contract": {
                    "foresight_predict_fn_input": "raw action tensor, shape (B,H,A)",
                    "foresight_predict_fn_output": (
                        "dict with left_marker_seq and optional right_marker_seq/eef_action_seq, "
                        "each differentiably connected to the input action"
                    ),
                    "guidance_location": "after DP denoising has produced clean/final action",
                    "reranking": False,
                    "every_step_ddpm_guidance": False,
                    "recompute_foresight_each_guidance_call": True,
                },
            }
        )
        return guided_norm.detach(), report

    @classmethod
    def from_dp_norm_stats(
        cls,
        task: str,
        *,
        runtime: TacQualityGuidanceRuntime,
        norm_stats: Mapping[str, object],
        norm_mode: str = "minmax",
        controller: Optional[TacQualityDPGuidanceController] = None,
        score_mode: str = "profile",
    ) -> "TacQualityDPIntegrationAdapter":
        return cls(
            task,
            runtime=runtime,
            controller=controller,
            action_normalizer=ActionNormalizer.from_norm_stats(norm_stats, mode=norm_mode),
            score_mode=score_mode,
        )


class SyntheticInsertionForesight(torch.nn.Module):
    def __init__(self, window: int = 8):
        super().__init__()
        self.window = window
        self.proj = torch.nn.Linear(7, 9 * 9 * 2)
        with torch.no_grad():
            self.proj.weight.normal_(0.0, 0.03)
            self.proj.bias.zero_()

    def forward(self, action_raw: torch.Tensor) -> Dict[str, torch.Tensor]:
        marker = self.proj(action_raw[..., :7]).view(action_raw.shape[0], action_raw.shape[1], 9, 9, 2)
        return {"left_marker_seq": marker[:, -self.window :]}


class SyntheticBoardForesight(torch.nn.Module):
    def __init__(self, window: int = 8):
        super().__init__()
        self.window = window
        self.left = torch.nn.Linear(7, 9 * 9 * 2)
        self.right = torch.nn.Linear(7, 9 * 9 * 2)
        with torch.no_grad():
            self.left.weight.normal_(0.0, 0.02)
            self.right.weight.normal_(0.0, 0.02)
            self.left.bias.zero_()
            self.right.bias.zero_()

    def forward(self, action_raw: torch.Tensor) -> Dict[str, torch.Tensor]:
        left = self.left(action_raw[..., :7]).view(action_raw.shape[0], action_raw.shape[1], 9, 9, 2)
        right = self.right(action_raw[..., :7]).view(action_raw.shape[0], action_raw.shape[1], 9, 9, 2)
        eef = action_raw[..., :6] if action_raw.shape[-1] >= 6 else torch.nn.functional.pad(action_raw, (0, 6 - action_raw.shape[-1]))
        return {
            "left_marker_seq": left[:, -self.window :],
            "right_marker_seq": right[:, -self.window :],
            "eef_action_seq": eef[:, -self.window :],
        }


def sanity(args) -> Dict[str, object]:
    torch.manual_seed(args.seed)
    runtime = TacQualityGuidanceRuntime(device=args.device)
    device = runtime.device
    action = torch.randn(args.batch_size, args.horizon, args.action_dim, device=device) * 0.1
    normalizer = ActionNormalizer(mode="identity")

    insertion_adapter = TacQualityDPIntegrationAdapter(
        "insertion",
        runtime=runtime,
        action_normalizer=normalizer,
    )
    insertion_foresight = SyntheticInsertionForesight(window=args.window).to(device)
    guided_i, report_i = insertion_adapter.guide_final_action(action, insertion_foresight)

    board_adapter = TacQualityDPIntegrationAdapter(
        "board",
        runtime=runtime,
        action_normalizer=normalizer,
    )
    board_foresight = SyntheticBoardForesight(window=args.window).to(device)
    guided_b, report_b = board_adapter.guide_final_action(action, board_foresight)

    action_min = torch.linspace(-0.8, -0.2, args.action_dim)
    action_max = torch.linspace(0.2, 0.8, args.action_dim)
    norm_stats = {
        "action_min": action_min,
        "action_max": action_max,
        "action_mean": torch.zeros(args.action_dim),
        "action_std": torch.ones(args.action_dim),
    }
    action_norm = torch.empty_like(action).uniform_(-0.25, 0.25)
    minmax_adapter = TacQualityDPIntegrationAdapter.from_dp_norm_stats(
        "insertion",
        runtime=runtime,
        norm_stats=norm_stats,
        norm_mode="minmax",
    )
    guided_minmax, report_minmax = minmax_adapter.guide_final_action(action_norm, insertion_foresight)
    roundtrip = minmax_adapter.action_normalizer.normalize(
        minmax_adapter.action_normalizer.denormalize(action_norm)
    )
    roundtrip_error = torch.max(torch.abs(roundtrip - action_norm))

    result = {
        "purpose": "TacQuality DP integration adapter sanity with differentiable synthetic Foresight.",
        "device": str(device),
        "insertion": report_i,
        "board": report_b,
        "minmax_insertion": report_minmax,
        "normalizer_roundtrip": {
            "mode": "minmax",
            "max_abs_error": float(roundtrip_error.detach().cpu()),
            "input_norm_range": summarize_tensor(action_norm),
            "guided_norm_range": summarize_tensor(guided_minmax),
        },
        "passes_integration_adapter_sanity": bool(
            report_i["improved_rate"] >= 0.95
            and report_b["improved_rate"] >= 0.95
            and report_minmax["improved_rate"] >= 0.95
            and report_i["finite_grad_rate"] >= 0.999
            and report_b["finite_grad_rate"] >= 0.999
            and report_minmax["finite_grad_rate"] >= 0.999
            and report_i["max_delta_within_trust_region"]
            and report_b["max_delta_within_trust_region"]
            and report_minmax["max_delta_within_trust_region"]
            and roundtrip_error.item() <= 1e-6
            and torch.isfinite(guided_i).all().item()
            and torch.isfinite(guided_b).all().item()
            and torch.isfinite(guided_minmax).all().item()
        ),
        "note": (
            "Synthetic Foresight only verifies integration mechanics.  Real deployment must "
            "replace Synthetic*Foresight with the trained differentiable Foresight wrapper."
        ),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "passes_integration_adapter_sanity": result["passes_integration_adapter_sanity"],
                "insertion_improved_rate": report_i["improved_rate"],
                "board_improved_rate": report_b["improved_rate"],
                "minmax_insertion_improved_rate": report_minmax["improved_rate"],
                "minmax_roundtrip_error": result["normalizer_roundtrip"]["max_abs_error"],
                "json": str(out),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--output", default=DEFAULT_OUT)
    return parser.parse_args()


if __name__ == "__main__":
    sanity(parse_args())
