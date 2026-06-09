"""Unified runtime contract for TacQuality classifier guidance.

This module keeps the task-specific scorer details behind one interface so a
DP policy can call a single guidance object:

  score = runtime.score(task, predicted_tactile, action)
  grad = d score / d action

It does not replace the trained scorers.  It wraps the currently validated
runtime modules and applies the task-specific guidance profile from
tac_quality_guidance_config.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TFAC_V5.insertion_risk_scorer_runtime import InsertionRiskScorerRuntime  # noqa: E402
from TFAC_V5.ptg_proxy_scorer_v2_runtime import PTGProxyScorerV2Runtime, TASK_TO_ID  # noqa: E402
from TFAC_V5.tac_quality_guidance_config import get_guidance_profile, weighted_logit_energy  # noqa: E402


DEFAULT_INSERTION_CKPT = "/home/chenshuai/Project/output/insertion_risk_scorer/insertion_risk_scorer_final.pt"
DEFAULT_BOARD_CKPT = "/home/chenshuai/Project/output/ptg_proxy_scorer_v2/ptg_proxy_scorer_v2_final.pt"
DEFAULT_OUT = "/home/chenshuai/Project/output/tac_quality_guidance_runtime/runtime_contract_sanity.json"


def freeze(module: nn.Module) -> None:
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)


def _as_action(action: torch.Tensor, dim: int) -> torch.Tensor:
    if action.shape[-1] > dim:
        return action[..., :dim]
    if action.shape[-1] < dim:
        pad = torch.zeros(*action.shape[:-1], dim - action.shape[-1], dtype=action.dtype, device=action.device)
        return torch.cat([action, pad], dim=-1)
    return action


class TacQualityGuidanceRuntime(nn.Module):
    """Task-conditioned scorer wrapper for DP guidance.

    Supported task inputs:
      - insertion: left tactile marker sequence and joint action sequence.
      - board: left/right marker sequences and eef/joint action sequences.

    The returned score is a differentiable per-sample tensor.  Callers can use
    autograd to backpropagate through Foresight into the DP action variable.
    """

    def __init__(
        self,
        insertion_ckpt: str = DEFAULT_INSERTION_CKPT,
        board_ckpt: str = DEFAULT_BOARD_CKPT,
        device: str = "cuda:0",
    ):
        super().__init__()
        self.device_name = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self.device = torch.device(self.device_name)
        self.insertion = InsertionRiskScorerRuntime(insertion_ckpt, device=self.device_name)
        self.board = PTGProxyScorerV2Runtime(board_ckpt, device=self.device_name)
        freeze(self.insertion)
        freeze(self.board)

    def score(
        self,
        task: str,
        left_marker_seq: torch.Tensor,
        action_seq: torch.Tensor,
        *,
        right_marker_seq: Optional[torch.Tensor] = None,
        eef_action_seq: Optional[torch.Tensor] = None,
        mode: str = "profile",
        clip: bool = True,
    ) -> torch.Tensor:
        task = task.lower()
        if task == "insertion":
            return self._score_insertion(left_marker_seq, action_seq, mode=mode, clip=clip)
        if task == "board":
            return self._score_board(
                left_marker_seq,
                action_seq,
                right_marker_seq=right_marker_seq,
                eef_action_seq=eef_action_seq,
                mode=mode,
                clip=clip,
            )
        raise KeyError(f"Unknown task {task!r}; expected 'insertion' or 'board'.")

    def diagnostics(
        self,
        task: str,
        left_marker_seq: torch.Tensor,
        action_seq: torch.Tensor,
        *,
        right_marker_seq: Optional[torch.Tensor] = None,
        eef_action_seq: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        task = task.lower()
        if task == "insertion":
            out = self.insertion.forward(left_marker_seq, action_seq)
            return {
                "quality": out["quality_score"],
                "p_good": out["p_good"],
                "risk_prob": out["risk_prob"],
                "quality_logit": out["quality_logit"],
                "binary_margin": out["good_logit_margin"],
                "reason_margin": out["reason_logit_margin"],
                "profile_energy": self._score_insertion(left_marker_seq, action_seq, mode="profile"),
            }
        if task == "board":
            task_id = torch.full(
                (left_marker_seq.shape[0],),
                TASK_TO_ID["board"],
                dtype=torch.long,
                device=self.device,
            )
            out = self.board.forward(
                left_marker_seq,
                right_marker_seq=right_marker_seq,
                eef_action_seq=eef_action_seq,
                joint_action_seq=action_seq,
                task_id=task_id,
            )
            return {
                "quality": out["quality_score"],
                "p_good": out["p_good"],
                "reason_good": out["reason_prob"][:, 1],
                "quality_logit": out["quality_logit"],
                "binary_margin": out["good_logit_margin"],
                "reason_margin": out["reason_logit_margin"],
                "profile_energy": self._score_board(
                    left_marker_seq,
                    action_seq,
                    right_marker_seq=right_marker_seq,
                    eef_action_seq=eef_action_seq,
                    mode="profile",
                ),
            }
        raise KeyError(f"Unknown task {task!r}; expected 'insertion' or 'board'.")

    def _score_insertion(self, marker_seq: torch.Tensor, action_seq: torch.Tensor, mode: str, clip: bool = True) -> torch.Tensor:
        marker_seq = marker_seq.to(self.device).float()
        action_seq = _as_action(action_seq.to(self.device).float(), self.insertion.action_dim)
        out = self.insertion.forward(marker_seq, action_seq)
        if mode == "profile":
            profile = get_guidance_profile("insertion")
            return weighted_logit_energy(
                out["quality_logit"],
                out["good_logit_margin"],
                out["reason_logit_margin"],
                profile.energy,
                clip=clip,
            )
        if mode == "calibrated":
            return out["energy_score"]
        return self.insertion.score(marker_seq, action_seq, mode=mode)

    def _score_board(
        self,
        left_marker_seq: torch.Tensor,
        action_seq: torch.Tensor,
        *,
        right_marker_seq: Optional[torch.Tensor],
        eef_action_seq: Optional[torch.Tensor],
        mode: str,
        clip: bool = True,
    ) -> torch.Tensor:
        left_marker_seq = left_marker_seq.to(self.device).float()
        action_seq = action_seq.to(self.device).float()
        if right_marker_seq is not None:
            right_marker_seq = right_marker_seq.to(self.device).float()
        if eef_action_seq is not None:
            eef_action_seq = eef_action_seq.to(self.device).float()
        task_id = torch.full((left_marker_seq.shape[0],), TASK_TO_ID["board"], dtype=torch.long, device=self.device)
        if mode == "profile":
            profile = get_guidance_profile("board")
            return self.board.weighted_energy_score(
                left_marker_seq,
                right_marker_seq=right_marker_seq,
                eef_action_seq=eef_action_seq,
                joint_action_seq=action_seq,
                task_id=task_id,
                quality_weight=profile.energy.quality,
                binary_weight=profile.energy.binary_margin,
                reason_weight=profile.energy.reason_margin,
                clip=clip,
            )
        if mode == "calibrated":
            return self.board.score(
                left_marker_seq,
                right_marker_seq=right_marker_seq,
                eef_action_seq=eef_action_seq,
                joint_action_seq=action_seq,
                task_id=task_id,
                mode="quality",
            )
        return self.board.score(
            left_marker_seq,
            right_marker_seq=right_marker_seq,
            eef_action_seq=eef_action_seq,
            joint_action_seq=action_seq,
            task_id=task_id,
            mode=mode,
        )


def _grad_summary(score: torch.Tensor, tensors: Dict[str, torch.Tensor]) -> Dict[str, object]:
    grads = torch.autograd.grad(score.sum(), list(tensors.values()), retain_graph=False, allow_unused=False)
    result: Dict[str, object] = {
        "score_mean": float(score.detach().mean().cpu()),
        "score_std": float(score.detach().std().cpu()),
        "all_finite": True,
        "all_nonzero": True,
    }
    for name, grad in zip(tensors.keys(), grads):
        finite = bool(torch.isfinite(grad).all().item())
        norm = float(grad.detach().flatten(1).norm(dim=1).mean().cpu())
        result[f"{name}_grad_finite"] = finite
        result[f"{name}_grad_norm_mean"] = norm
        result["all_finite"] = bool(result["all_finite"] and finite)
        result["all_nonzero"] = bool(result["all_nonzero"] and norm > 1e-8)
    return result


def sanity(args) -> Dict[str, object]:
    torch.manual_seed(args.seed)
    runtime = TacQualityGuidanceRuntime(args.insertion_ckpt, args.board_ckpt, args.device)
    device = runtime.device
    batch = args.batch_size
    window = args.window

    insertion_marker = torch.randn(batch, window, 9, 9, 2, device=device, requires_grad=True)
    insertion_action = torch.randn(batch, window, runtime.insertion.action_dim, device=device, requires_grad=True)
    insertion_score = runtime.score("insertion", insertion_marker, insertion_action, mode="profile")
    insertion_grad = _grad_summary(
        insertion_score,
        {"marker": insertion_marker, "action": insertion_action},
    )

    board_left = torch.randn(batch, window, 9, 9, 2, device=device, requires_grad=True)
    board_right = torch.randn(batch, window, 9, 9, 2, device=device, requires_grad=True)
    board_eef = torch.randn(batch, window, 6, device=device, requires_grad=True)
    board_joint = torch.randn(batch, window, 7, device=device, requires_grad=True)
    board_score = runtime.score(
        "board",
        board_left,
        board_joint,
        right_marker_seq=board_right,
        eef_action_seq=board_eef,
        mode="profile",
    )
    board_grad = _grad_summary(
        board_score,
        {
            "left_marker": board_left,
            "right_marker": board_right,
            "eef_action": board_eef,
            "joint_action": board_joint,
        },
    )

    result = {
        "contract": {
            "score_call": "runtime.score(task, predicted_tactile, action, mode='profile')",
            "insertion_profile": get_guidance_profile("insertion").to_dict(),
            "board_profile": get_guidance_profile("board").to_dict(),
            "mode_profile": "uses task-specific validated deployment energy",
            "mode_calibrated": "uses calibration-best analysis score: insertion energy, board quality",
        },
        "device": str(device),
        "insertion": insertion_grad,
        "board": board_grad,
        "passes_runtime_contract_sanity": bool(
            insertion_grad["all_finite"]
            and insertion_grad["all_nonzero"]
            and board_grad["all_finite"]
            and board_grad["all_nonzero"]
        ),
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--insertion_ckpt", default=DEFAULT_INSERTION_CKPT)
    parser.add_argument("--board_ckpt", default=DEFAULT_BOARD_CKPT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=DEFAULT_OUT)
    return parser.parse_args()


if __name__ == "__main__":
    sanity(parse_args())
