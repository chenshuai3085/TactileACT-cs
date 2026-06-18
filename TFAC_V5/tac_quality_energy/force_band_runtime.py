"""Force-band-aware board TacQuality scorer runtime.

This module is the deployable, differentiable counterpart of the offline
force-band board scorer audit.  It intentionally keeps the runtime input to
marker/action proxy features so score(action) can be differentiated through
Foresight-predicted marker fields during DP classifier guidance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .proxy_features import action_proxy_features_torch, marker_proxy_features_torch


DEFAULT_CKPT = "/home/chenshuai/Project/output/board_force_band_tac_quality_energy/force_band_tac_quality_energy_best.pt"
REASON_TO_ID = {"too_small": 0, "positive": 1, "too_large": 2, "oscillate": 3}


def _as_action(
    action: Optional[torch.Tensor],
    batch: int,
    window: int,
    dim: int,
    device: torch.device,
) -> torch.Tensor:
    if action is None:
        return torch.zeros(batch, window, dim, dtype=torch.float32, device=device)
    action = action.to(device).float()
    if action.shape[1] != window:
        if action.shape[1] > window:
            action = action[:, -window:]
        else:
            pad = action[:, :1].expand(-1, window - action.shape[1], -1)
            action = torch.cat([pad, action], dim=1)
    if action.shape[-1] > dim:
        return action[..., :dim]
    if action.shape[-1] < dim:
        pad = torch.zeros(*action.shape[:-1], dim - action.shape[-1], dtype=action.dtype, device=device)
        return torch.cat([action, pad], dim=-1)
    return action


class ForceBandTacQualityEnergy(nn.Module):
    """Small differentiable multi-head scorer for board wiping quality."""

    def __init__(self, in_dim: int = 74, hidden: int = 192, dropout: float = 0.10):
        super().__init__()
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
            nn.LayerNorm(hidden // 2),
            nn.SiLU(),
        )
        self.binary_head = nn.Linear(hidden // 2, 2)
        self.reason_head = nn.Linear(hidden // 2, 4)
        self.quality_head = nn.Linear(hidden // 2, 1)
        self.teacher_head = nn.Linear(hidden // 2, 1)
        self.energy_head = nn.Linear(hidden // 2, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.encoder(x)
        binary_logits = self.binary_head(h)
        reason_logits = self.reason_head(h)
        quality_logit = self.quality_head(h).squeeze(-1)
        teacher_logit = self.teacher_head(h).squeeze(-1)
        free_energy = self.energy_head(h).squeeze(-1)

        good_margin = binary_logits[:, 1] - binary_logits[:, 0]
        bad_reason = torch.logsumexp(
            torch.stack(
                [
                    reason_logits[:, REASON_TO_ID["too_small"]],
                    reason_logits[:, REASON_TO_ID["too_large"]],
                    reason_logits[:, REASON_TO_ID["oscillate"]],
                ],
                dim=-1,
            ),
            dim=-1,
        )
        reason_margin = reason_logits[:, REASON_TO_ID["positive"]] - bad_reason
        energy_logit = (
            0.40 * quality_logit
            + 0.25 * good_margin
            + 0.20 * reason_margin
            + 0.10 * teacher_logit
            + 0.05 * free_energy
        )
        return {
            "binary_logits": binary_logits,
            "reason_logits": reason_logits,
            "quality_logit": quality_logit,
            "teacher_logit": teacher_logit,
            "free_energy": free_energy,
            "good_margin": good_margin,
            "reason_margin": reason_margin,
            "energy_logit": energy_logit,
            "energy_clipped": torch.tanh(energy_logit / 4.0) * 4.0,
        }


class ForceBandTacQualityEnergyRuntime(nn.Module):
    """Checkpoint-backed differentiable scorer used for DP guidance."""

    def __init__(self, checkpoint_path: str = DEFAULT_CKPT, device: str = "cuda:0"):
        super().__init__()
        self.device_name = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self.device = torch.device(self.device_name)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.feature_dim = int(ckpt["feature_dim"])
        self.model = ForceBandTacQualityEnergy(
            in_dim=self.feature_dim,
            hidden=int(ckpt.get("hidden", 192)),
            dropout=0.0,
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.reason_to_id = ckpt.get("reason_to_id", REASON_TO_ID)
        self.feature_variant = ckpt.get("feature_variant", "marker_action")
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
        marker_window = left.shape[1]
        left_feat = marker_proxy_features_torch(left)
        right_feat = marker_proxy_features_torch(right)
        diff_feat = (left_feat - right_feat).abs()
        eef_window = eef_action_seq.shape[1] if eef_action_seq is not None else marker_window
        joint_window = joint_action_seq.shape[1] if joint_action_seq is not None else marker_window
        eef_action = _as_action(eef_action_seq, batch, eef_window, 6, self.device)
        joint_action = _as_action(joint_action_seq, batch, joint_window, 7, self.device)
        eef_feat = action_proxy_features_torch(eef_action, action_dim=6, window=eef_window)
        joint_feat = action_proxy_features_torch(joint_action, action_dim=7, window=joint_window)
        if self.feature_variant == "marker_left":
            return left_feat
        if self.feature_variant == "left_marker_action":
            return torch.cat([left_feat, joint_feat, eef_feat], dim=-1)
        return torch.cat([left_feat, right_feat, diff_feat, joint_feat, eef_feat], dim=-1)

    def forward(
        self,
        left_marker_seq: torch.Tensor,
        right_marker_seq: Optional[torch.Tensor] = None,
        eef_action_seq: Optional[torch.Tensor] = None,
        joint_action_seq: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        feat = self.proxy_features(left_marker_seq, right_marker_seq, eef_action_seq, joint_action_seq)
        feat_norm = (feat - self.scaler_mean.view(1, -1)) / (self.scaler_scale.view(1, -1) + 1e-8)
        raw = self.model(feat_norm)
        p_good = torch.softmax(raw["binary_logits"], dim=-1)[:, 1]
        reason_prob = torch.softmax(raw["reason_logits"], dim=-1)
        raw.update(
            {
                "p_good": p_good,
                "log_p_good": torch.log(p_good.clamp_min(1e-8)),
                "reason_prob": reason_prob,
                "quality_score": torch.sigmoid(raw["quality_logit"]),
                "teacher_score": torch.sigmoid(raw["teacher_logit"]),
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
        del task_id
        out = self.forward(left_marker_seq, right_marker_seq, eef_action_seq, joint_action_seq)
        if mode == "energy_clipped":
            return out["energy_clipped"]
        if mode == "energy":
            return out["energy_logit"]
        if mode == "quality":
            return out["quality_score"]
        if mode == "teacher":
            return out["teacher_score"]
        if mode == "p_good":
            return out["p_good"]
        if mode == "log_p_good":
            return out["log_p_good"]
        if mode == "reason_good":
            return out["reason_prob"][:, self.reason_to_id.get("positive", 1)]
        if mode == "profile":
            return 0.5 * out["quality_logit"] + 0.25 * out["good_margin"] + 0.25 * out["reason_margin"]
        raise ValueError(f"Unknown ForceBandTacQualityEnergy score mode: {mode}")


def load_runtime(checkpoint_path: str | Path = DEFAULT_CKPT, device: str = "cuda:0") -> ForceBandTacQualityEnergyRuntime:
    return ForceBandTacQualityEnergyRuntime(str(checkpoint_path), device=device)
