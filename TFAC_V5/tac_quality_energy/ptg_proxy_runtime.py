"""PTGProxyV2 runtime kept inside the TacQualityEnergy package.

This is the board default scorer used by the current rollout-arm config.  It
shares the same differentiable marker/action proxy features as
``DistilledTacQualityEnergyRuntime`` but loads the older three-head
``ptg_proxy_scorer_v2_final.pt`` checkpoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .proxy_features import action_proxy_features_torch, marker_proxy_features_torch


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

    def forward(self, x: torch.Tensor, task_id: torch.Tensor) -> Dict[str, torch.Tensor]:
        task_oh = F.one_hot(task_id.long(), num_classes=2).float()
        h = self.encoder(torch.cat([x, task_oh], dim=-1))
        return {
            "binary_logits": self.binary_head(h),
            "reason_logits": self.reason_head(h),
            "quality": self.quality_head(h).squeeze(-1),
        }


def _as_action(
    action: Optional[torch.Tensor],
    batch: int,
    window: int,
    dim: int,
    device: torch.device,
) -> torch.Tensor:
    if action is None:
        return torch.zeros(batch, window, dim, device=device)
    action = action.to(device).float()
    if action.shape[-1] > dim:
        return action[..., :dim]
    if action.shape[-1] < dim:
        pad = torch.zeros(*action.shape[:-1], dim - action.shape[-1], dtype=action.dtype, device=device)
        return torch.cat([action, pad], dim=-1)
    return action


class PTGProxyScorerV2Runtime(nn.Module):
    """Checkpoint-backed differentiable PTGProxyV2 scorer runtime."""

    def __init__(self, checkpoint_path: str = DEFAULT_CKPT, device: str = "cuda:0"):
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
        for param in self.model.parameters():
            param.requires_grad_(False)
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
            task_id = torch.full((feat.shape[0],), self.task_to_id.get("board", 1), dtype=torch.long, device=self.device)
        else:
            task_id = task_id.to(self.device).long()
        raw = self.model(feat_norm, task_id)
        p_good = torch.softmax(raw["binary_logits"], dim=-1)[:, 1]
        reason_prob = torch.softmax(raw["reason_logits"], dim=-1)
        quality_logit = raw["quality"]
        good_margin = raw["binary_logits"][:, 1] - raw["binary_logits"][:, 0]
        bad_reason_logits = torch.stack(
            [
                raw["reason_logits"][:, 0],
                torch.logsumexp(raw["reason_logits"][:, 2:], dim=-1),
            ],
            dim=-1,
        )
        reason_margin = raw["reason_logits"][:, 1] - torch.logsumexp(bad_reason_logits, dim=-1)
        energy = quality_logit + 0.25 * good_margin + 0.25 * reason_margin
        raw.update(
            {
                "p_good": p_good,
                "log_p_good": torch.log(p_good.clamp_min(1e-8)),
                "reason_prob": reason_prob,
                "quality_score": torch.sigmoid(quality_logit),
                "quality_logit": quality_logit,
                "good_margin": good_margin,
                "reason_margin": reason_margin,
                "energy_logit": energy,
                "energy_clipped": torch.tanh(energy / 4.0) * 4.0,
                "proxy_features": feat,
                "proxy_features_norm": feat_norm,
            }
        )
        return raw

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
        if mode == "energy_clipped":
            return out["energy_clipped"]
        if mode == "energy":
            return out["energy_logit"]
        if mode == "quality":
            return out["quality_score"]
        if mode == "p_good":
            return out["p_good"]
        if mode == "log_p_good":
            return out["log_p_good"]
        if mode == "reason_good":
            return out["reason_prob"][:, 1]
        raise ValueError(f"Unknown PTGProxyScorerV2 score mode: {mode}")

    def weighted_energy_score(
        self,
        left_marker_seq: torch.Tensor,
        right_marker_seq: Optional[torch.Tensor] = None,
        eef_action_seq: Optional[torch.Tensor] = None,
        joint_action_seq: Optional[torch.Tensor] = None,
        task_id: Optional[torch.Tensor] = None,
        quality_weight: float = 0.75,
        binary_weight: float = 0.10,
        reason_weight: float = 0.0,
        clip: bool = True,
    ) -> torch.Tensor:
        out = self.forward(left_marker_seq, right_marker_seq, eef_action_seq, joint_action_seq, task_id)
        energy = (
            quality_weight * out["quality_logit"]
            + binary_weight * out["good_margin"]
            + reason_weight * out["reason_margin"]
        )
        return torch.tanh(energy / 4.0) * 4.0 if clip else energy


def load_runtime(checkpoint_path: str | Path = DEFAULT_CKPT, device: str = "cuda:0") -> PTGProxyScorerV2Runtime:
    return PTGProxyScorerV2Runtime(str(checkpoint_path), device=device)
