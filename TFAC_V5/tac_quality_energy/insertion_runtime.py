"""Insertion-specific TacQuality runtime.

This runtime keeps the socket-insertion scorer available from the formal
``TFAC_V5.tac_quality_energy`` package.  It uses dense marker/action encoders
plus differentiable proxy features, and exposes the same ``score`` call shape
used by the serving guidance adapter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .proxy_features import action_proxy_features_torch, marker_proxy_features_torch


DEFAULT_CKPT = "/home/chenshuai/Project/output/insertion_risk_scorer/insertion_risk_scorer_final.pt"
WINDOW = 8
ACTION_DIM = 7


class InsertionRiskScorer(nn.Module):
    def __init__(
        self,
        marker_proxy_dim: int,
        action_proxy_dim: int,
        action_dim: int = ACTION_DIM,
        hidden: int = 160,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.marker_encoder = nn.Sequential(
            nn.Conv3d(2, 24, kernel_size=(3, 3, 3), padding=1),
            nn.GroupNorm(6, 24),
            nn.SiLU(),
            nn.Conv3d(24, 48, kernel_size=(3, 3, 3), padding=1),
            nn.GroupNorm(8, 48),
            nn.SiLU(),
            nn.AdaptiveAvgPool3d((2, 3, 3)),
            nn.Flatten(),
        )
        self.action_encoder = nn.Sequential(
            nn.Conv1d(action_dim, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv1d(32, 48, kernel_size=3, padding=1),
            nn.GroupNorm(8, 48),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(4),
            nn.Flatten(),
        )
        in_dim = 48 * 2 * 3 * 3 + 48 * 4 + marker_proxy_dim + action_proxy_dim
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
        )
        self.binary_head = nn.Linear(hidden // 2, 2)
        self.reason_head = nn.Linear(hidden // 2, 4)
        self.quality_head = nn.Linear(hidden // 2, 1)

    def forward(
        self,
        marker: torch.Tensor,
        marker_proxy: torch.Tensor,
        action: torch.Tensor,
        action_proxy: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        marker_feat = self.marker_encoder(marker.permute(0, 4, 1, 2, 3))
        action_feat = self.action_encoder(action.permute(0, 2, 1))
        h = self.encoder(torch.cat([marker_feat, action_feat, marker_proxy, action_proxy], dim=-1))
        return {
            "binary_logits": self.binary_head(h),
            "reason_logits": self.reason_head(h),
            "quality": self.quality_head(h).squeeze(-1),
        }


def _ensure_action(action_seq: Optional[torch.Tensor], batch: int, window: int, action_dim: int, device: torch.device) -> torch.Tensor:
    if action_seq is None:
        return torch.zeros(batch, window, action_dim, device=device)
    action_seq = action_seq.to(device).float()
    if action_seq.shape[-1] > action_dim:
        return action_seq[..., :action_dim]
    if action_seq.shape[-1] < action_dim:
        pad = torch.zeros(
            *action_seq.shape[:-1],
            action_dim - action_seq.shape[-1],
            dtype=action_seq.dtype,
            device=device,
        )
        return torch.cat([action_seq, pad], dim=-1)
    return action_seq


class InsertionRiskScorerRuntime(nn.Module):
    """Checkpoint-backed insertion risk scorer with differentiable preprocessing."""

    def __init__(self, checkpoint_path: str = DEFAULT_CKPT, device: str = "cuda:0"):
        super().__init__()
        self.device_name = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self.device = torch.device(self.device_name)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.action_dim = int(ckpt.get("action_dim", ACTION_DIM))
        self.model = InsertionRiskScorer(
            marker_proxy_dim=int(ckpt["marker_proxy_dim"]),
            action_proxy_dim=int(ckpt["action_proxy_dim"]),
            action_dim=self.action_dim,
            hidden=int(ckpt.get("hidden", 160)),
            dropout=0.0,
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

        self.reason_names = ckpt.get("reason_names", {})
        self.register_buffer("marker_mean", torch.as_tensor(ckpt["marker_mean"], dtype=torch.float32, device=self.device).view(1, 1, 1, 1, 2))
        self.register_buffer("marker_std", torch.as_tensor(ckpt["marker_std"], dtype=torch.float32, device=self.device).view(1, 1, 1, 1, 2))
        self.register_buffer("action_mean", torch.as_tensor(ckpt["action_mean"], dtype=torch.float32, device=self.device).view(1, 1, self.action_dim))
        self.register_buffer("action_std", torch.as_tensor(ckpt["action_std"], dtype=torch.float32, device=self.device).view(1, 1, self.action_dim))
        self.register_buffer("marker_proxy_mean", torch.as_tensor(ckpt["marker_proxy_mean"], dtype=torch.float32, device=self.device).view(1, -1))
        self.register_buffer("marker_proxy_scale", torch.as_tensor(ckpt["marker_proxy_scale"], dtype=torch.float32, device=self.device).view(1, -1))
        self.register_buffer("action_proxy_mean", torch.as_tensor(ckpt["action_proxy_mean"], dtype=torch.float32, device=self.device).view(1, -1))
        self.register_buffer("action_proxy_scale", torch.as_tensor(ckpt["action_proxy_scale"], dtype=torch.float32, device=self.device).view(1, -1))

    def forward(
        self,
        left_marker_seq: torch.Tensor,
        right_marker_seq: Optional[torch.Tensor] = None,
        eef_action_seq: Optional[torch.Tensor] = None,
        joint_action_seq: Optional[torch.Tensor] = None,
        task_id: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        del right_marker_seq, task_id
        marker_seq = left_marker_seq.to(self.device).float()
        batch, window = marker_seq.shape[:2]
        action_source = joint_action_seq if joint_action_seq is not None else eef_action_seq
        action_seq = _ensure_action(action_source, batch, window, self.action_dim, self.device)

        marker_proxy = marker_proxy_features_torch(marker_seq)
        action_proxy = action_proxy_features_torch(action_seq, self.action_dim)
        marker_norm = (marker_seq - self.marker_mean) / self.marker_std.clamp_min(1e-8)
        action_norm = (action_seq - self.action_mean) / self.action_std.clamp_min(1e-8)
        marker_proxy_norm = (marker_proxy - self.marker_proxy_mean) / (self.marker_proxy_scale + 1e-8)
        action_proxy_norm = (action_proxy - self.action_proxy_mean) / (self.action_proxy_scale + 1e-8)

        out = self.model(marker_norm, marker_proxy_norm, action_norm, action_proxy_norm)
        p_good = torch.softmax(out["binary_logits"], dim=-1)[:, 1]
        reason_prob = torch.softmax(out["reason_logits"], dim=-1)
        risk_prob = reason_prob[:, 2] + reason_prob[:, 3]
        quality_logit = out["quality"]
        good_margin = out["binary_logits"][:, 1] - out["binary_logits"][:, 0]
        reason_margin = out["reason_logits"][:, 1] - torch.logsumexp(out["reason_logits"][:, 2:4], dim=-1)
        energy = 0.5 * quality_logit + 0.1 * good_margin

        out.update(
            {
                "p_good": p_good,
                "log_p_good": torch.log(p_good.clamp_min(1e-8)),
                "reason_prob": reason_prob,
                "risk_prob": risk_prob,
                "quality_score": torch.sigmoid(quality_logit),
                "quality_logit": quality_logit,
                "good_margin": good_margin,
                "good_logit_margin": good_margin,
                "reason_margin": reason_margin,
                "reason_logit_margin": reason_margin,
                "energy_logit": energy,
                "energy_score": energy,
                "energy_clipped": torch.tanh(energy / 4.0) * 4.0,
                "marker_proxy": marker_proxy,
                "action_proxy": action_proxy,
            }
        )
        return out

    def profile_score(
        self,
        left_marker_seq: torch.Tensor,
        right_marker_seq: Optional[torch.Tensor] = None,
        eef_action_seq: Optional[torch.Tensor] = None,
        joint_action_seq: Optional[torch.Tensor] = None,
        task_id: Optional[torch.Tensor] = None,
        *,
        quality_weight: float = 0.5,
        binary_weight: float = 0.1,
        reason_weight: float = 0.0,
        risk_weight: float = 0.0,
        clip: bool = True,
    ) -> torch.Tensor:
        """Weighted insertion guidance score used by rollout profile configs."""
        out = self.forward(left_marker_seq, right_marker_seq, eef_action_seq, joint_action_seq, task_id)
        score = (
            float(quality_weight) * out["quality_logit"]
            + float(binary_weight) * out["good_margin"]
            + float(reason_weight) * out["reason_margin"]
            - float(risk_weight) * out["risk_prob"]
        )
        return torch.tanh(score / 4.0) * 4.0 if clip else score

    def score(
        self,
        left_marker_seq: torch.Tensor,
        right_marker_seq: Optional[torch.Tensor] = None,
        eef_action_seq: Optional[torch.Tensor] = None,
        joint_action_seq: Optional[torch.Tensor] = None,
        task_id: Optional[torch.Tensor] = None,
        mode: str = "energy_clipped",
    ) -> torch.Tensor:
        if mode == "profile":
            return self.profile_score(
                left_marker_seq,
                right_marker_seq,
                eef_action_seq,
                joint_action_seq,
                task_id,
            )
        out = self.forward(left_marker_seq, right_marker_seq, eef_action_seq, joint_action_seq, task_id)
        if mode == "quality":
            return out["quality_score"]
        if mode == "p_good":
            return out["p_good"]
        if mode == "log_p_good":
            return out["log_p_good"]
        if mode == "neg_risk":
            return -out["risk_prob"]
        if mode == "risk_guidance":
            return out["quality_score"] + 0.35 * out["log_p_good"] - 0.5 * out["risk_prob"]
        if mode == "energy":
            return out["energy_logit"]
        if mode == "energy_clipped":
            return out["energy_clipped"]
        raise ValueError(f"Unknown insertion risk score mode: {mode}")


def load_runtime(checkpoint_path: str | Path = DEFAULT_CKPT, device: str = "cuda:0") -> InsertionRiskScorerRuntime:
    return InsertionRiskScorerRuntime(str(checkpoint_path), device=device)
