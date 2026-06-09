"""Runtime wrapper for the PTG proxy scorer v2.

This module keeps dataset code out of inference.  It loads
ptg_proxy_scorer_v2_final.pt and computes the same differentiable proxy
features in torch:

  left marker proxy + right marker proxy + abs difference
  eef action proxy + joint action proxy

The returned quality / good probability can be used for candidate reranking or
as a gradient source inside DP guidance once marker/action predictions are
connected to Foresight and the denoising action variable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TFAC_V5.action_aware_scorer_runtime import (
    action_proxy_features_torch,
    marker_proxy_features_torch,
)
from TFAC_V5.tac_quality_guidance_config import EnergyWeights, weighted_logit_energy


TASK_TO_ID = {"insertion": 0, "board": 1}
DEFAULT_CKPT = "/home/chenshuai/Project/output/ptg_proxy_scorer_v2/ptg_proxy_scorer_v2_final.pt"


class PTGProxyScorerV2(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 192, dropout: float = 0.0):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim + 2, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.SiLU(),
        )
        self.binary_head = nn.Linear(hidden // 2, 2)
        self.reason_head = nn.Linear(hidden // 2, 5)
        self.quality_head = nn.Linear(hidden // 2, 1)

    def forward(self, x, task_id):
        task_oh = F.one_hot(task_id.long(), num_classes=2).float()
        h = self.encoder(torch.cat([x, task_oh], dim=-1))
        return {
            "binary_logits": self.binary_head(h),
            "reason_logits": self.reason_head(h),
            "quality": self.quality_head(h).squeeze(-1),
        }


def _as_action(action: Optional[torch.Tensor], batch: int, window: int, dim: int, device) -> torch.Tensor:
    if action is None:
        return torch.zeros(batch, window, dim, device=device)
    action = action.to(device).float()
    if action.shape[-1] > dim:
        action = action[..., :dim]
    elif action.shape[-1] < dim:
        pad = torch.zeros(*action.shape[:-1], dim - action.shape[-1], device=device, dtype=action.dtype)
        action = torch.cat([action, pad], dim=-1)
    return action


class PTGProxyScorerV2Runtime(nn.Module):
    def __init__(self, checkpoint_path: str = DEFAULT_CKPT, device: str = "cuda"):
        super().__init__()
        self.device_name = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self.device = torch.device(self.device_name)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.feature_dim = int(ckpt["feature_dim"])
        self.model = PTGProxyScorerV2(
            self.feature_dim,
            hidden=int(ckpt.get("hidden", 192)),
            dropout=0.0,
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.task_to_id = ckpt.get("task_to_id", TASK_TO_ID)
        self.reason_names = ckpt.get("reason_names", {})
        self.register_buffer("scaler_mean", torch.as_tensor(ckpt["scaler_mean"], dtype=torch.float32, device=self.device))
        self.register_buffer("scaler_scale", torch.as_tensor(ckpt["scaler_scale"], dtype=torch.float32, device=self.device))

    def proxy_features(
        self,
        left_marker_seq: torch.Tensor,
        right_marker_seq: Optional[torch.Tensor] = None,
        eef_action_seq: Optional[torch.Tensor] = None,
        joint_action_seq: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        left = left_marker_seq.to(self.device).float()
        right = right_marker_seq.to(self.device).float() if right_marker_seq is not None else left
        batch = left.shape[0]
        window = left.shape[1]
        left_feat = marker_proxy_features_torch(left)
        right_feat = marker_proxy_features_torch(right)
        diff_feat = (left_feat - right_feat).abs()
        eef_action = _as_action(eef_action_seq, batch, window, 6, self.device)
        joint_action = _as_action(joint_action_seq, batch, window, 7, self.device)
        eef_feat = action_proxy_features_torch(eef_action, action_dim=6)
        joint_feat = action_proxy_features_torch(joint_action, action_dim=7)
        return torch.cat([left_feat, right_feat, diff_feat, eef_feat, joint_feat], dim=-1)

    def forward(
        self,
        left_marker_seq: torch.Tensor,
        right_marker_seq: Optional[torch.Tensor] = None,
        eef_action_seq: Optional[torch.Tensor] = None,
        joint_action_seq: Optional[torch.Tensor] = None,
        task_id: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        feat = self.proxy_features(left_marker_seq, right_marker_seq, eef_action_seq, joint_action_seq)
        feat_norm = (feat - self.scaler_mean.view(1, -1)) / (self.scaler_scale.view(1, -1) + 1e-8)
        if task_id is None:
            task_id = torch.zeros(feat.shape[0], dtype=torch.long, device=self.device)
        else:
            task_id = task_id.to(self.device).long()
        raw = self.model(feat_norm, task_id)
        p_good = torch.softmax(raw["binary_logits"], dim=-1)[:, 1]
        reason_prob = torch.softmax(raw["reason_logits"], dim=-1)
        quality = torch.sigmoid(raw["quality"])
        good_logit_margin = raw["binary_logits"][:, 1] - raw["binary_logits"][:, 0]
        bad_reason_logits = torch.stack(
            [
                raw["reason_logits"][:, 0],
                torch.logsumexp(raw["reason_logits"][:, 2:], dim=-1),
            ],
            dim=-1,
        )
        reason_logit_margin = raw["reason_logits"][:, 1] - torch.logsumexp(bad_reason_logits, dim=-1)
        quality_logit = raw["quality"]
        energy_score = quality_logit + 0.25 * good_logit_margin + 0.25 * reason_logit_margin
        raw.update(
            {
                "p_good": p_good,
                "log_p_good": torch.log(p_good.clamp_min(1e-8)),
                "reason_prob": reason_prob,
                "quality_score": quality,
                "quality_logit": quality_logit,
                "good_logit_margin": good_logit_margin,
                "reason_logit_margin": reason_logit_margin,
                "energy_score": energy_score,
                "proxy_features": feat,
                "proxy_features_norm": feat_norm,
            }
        )
        return raw

    def guidance_score(
        self,
        left_marker_seq: torch.Tensor,
        right_marker_seq: Optional[torch.Tensor] = None,
        eef_action_seq: Optional[torch.Tensor] = None,
        joint_action_seq: Optional[torch.Tensor] = None,
        task_id: Optional[torch.Tensor] = None,
        quality_weight: float = 1.0,
        good_weight: float = 0.25,
        reason_weight: float = 0.25,
        action_smooth_weight: float = 0.0,
    ) -> torch.Tensor:
        out = self.forward(left_marker_seq, right_marker_seq, eef_action_seq, joint_action_seq, task_id)
        score = quality_weight * out["quality_score"] + good_weight * out["log_p_good"]
        score = score + reason_weight * out["reason_prob"][:, 1]
        if action_smooth_weight and (eef_action_seq is not None or joint_action_seq is not None):
            action = joint_action_seq if joint_action_seq is not None else eef_action_seq
            action = action.to(self.device).float()
            if action.shape[1] > 2:
                accel = action[:, 2:] - 2 * action[:, 1:-1] + action[:, :-2]
                score = score - action_smooth_weight * torch.linalg.norm(accel, dim=-1).mean(dim=1)
        return score

    def score(
        self,
        left_marker_seq: torch.Tensor,
        right_marker_seq: Optional[torch.Tensor] = None,
        eef_action_seq: Optional[torch.Tensor] = None,
        joint_action_seq: Optional[torch.Tensor] = None,
        task_id: Optional[torch.Tensor] = None,
        mode: str = "energy_clipped",
    ) -> torch.Tensor:
        out = self.forward(left_marker_seq, right_marker_seq, eef_action_seq, joint_action_seq, task_id)
        if mode == "quality":
            return out["quality_score"]
        if mode == "p_good":
            return out["p_good"]
        if mode == "log_p_good":
            return out["log_p_good"]
        if mode == "reason_good":
            return out["reason_prob"][:, 1]
        if mode == "energy":
            return out["energy_score"]
        if mode == "energy_clipped":
            return torch.tanh(out["energy_score"] / 4.0) * 4.0
        raise ValueError(mode)

    def weighted_energy_score(
        self,
        left_marker_seq: torch.Tensor,
        right_marker_seq: Optional[torch.Tensor] = None,
        eef_action_seq: Optional[torch.Tensor] = None,
        joint_action_seq: Optional[torch.Tensor] = None,
        task_id: Optional[torch.Tensor] = None,
        quality_weight: float = 0.75,
        binary_weight: float = 0.1,
        reason_weight: float = 0.0,
        clip: bool = True,
    ) -> torch.Tensor:
        out = self.forward(left_marker_seq, right_marker_seq, eef_action_seq, joint_action_seq, task_id)
        weights = EnergyWeights(
            quality=quality_weight,
            binary_margin=binary_weight,
            reason_margin=reason_weight,
        )
        return weighted_logit_energy(
            out["quality_logit"],
            out["good_logit_margin"],
            out["reason_logit_margin"],
            weights,
            clip=clip,
        )


def sanity(args):
    torch.manual_seed(0)
    runtime = PTGProxyScorerV2Runtime(args.checkpoint, device=args.device)
    device = runtime.device
    batch = 8
    left = torch.randn(batch, 8, 9, 9, 2, device=device, requires_grad=True)
    right = torch.randn(batch, 8, 9, 9, 2, device=device, requires_grad=True)
    eef = torch.randn(batch, 8, 6, device=device, requires_grad=True)
    joint = torch.randn(batch, 8, 7, device=device, requires_grad=True)
    task_id = torch.tensor([0, 1] * (batch // 2), dtype=torch.long, device=device)
    score = runtime.guidance_score(
        left,
        right,
        eef_action_seq=eef,
        joint_action_seq=joint,
        task_id=task_id,
        action_smooth_weight=0.02,
    ).mean()
    grads = torch.autograd.grad(score, [left, right, eef, joint], retain_graph=False)
    result = {
        "checkpoint": args.checkpoint,
        "device": str(device),
        "score": float(score.detach().cpu()),
        "grad_left_norm": float(grads[0].norm().detach().cpu()),
        "grad_right_norm": float(grads[1].norm().detach().cpu()),
        "grad_eef_norm": float(grads[2].norm().detach().cpu()),
        "grad_joint_norm": float(grads[3].norm().detach().cpu()),
        "usable_for_guidance": bool(all(torch.isfinite(g).all().item() and g.norm().item() > 1e-8 for g in grads)),
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="/home/chenshuai/Project/output/ptg_proxy_scorer_v2/runtime_gradient_sanity.json")
    return parser.parse_args()


if __name__ == "__main__":
    sanity(parse_args())
