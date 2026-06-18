"""Continuous proxy-energy board scorer for non-saturated guidance audits."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from .proxy_features import action_proxy_features_torch, marker_proxy_features_torch


@dataclass(frozen=True)
class BoardProxyEnergyConfig:
    # Positive-union marker/action centers from 260609 old positive + 260617 positive.
    mean_mag_center: float = 4.714370
    mean_mag_scale: float = 2.298754
    area_center: float = 0.95
    area_scale: float = 0.12
    marker_delta_center: float = 0.184770
    marker_delta_scale: float = 0.094409
    centroid_delta_center: float = 0.034423
    centroid_delta_scale: float = 0.023846
    drift_center: float = 3.156348
    drift_scale: float = 4.973095
    action_speed_center: float = 0.217204
    action_speed_scale: float = 0.104642
    action_accel_center: float = 0.043671
    action_accel_scale: float = 0.184535

    # Weights are deliberately explicit so the score is interpretable.
    contact_weight: float = 0.35
    area_weight: float = 0.10
    smooth_weight: float = 0.25
    center_weight: float = 0.10
    drift_weight: float = 0.08
    action_weight: float = 0.12


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


def _band_score(value: torch.Tensor, center: float, scale: float) -> torch.Tensor:
    return torch.exp(-0.5 * ((value - center) / max(scale, 1e-6)).square())


def _high_score(value: torch.Tensor, center: float, scale: float) -> torch.Tensor:
    return torch.sigmoid((value - center) / max(scale, 1e-6))


def _low_score(value: torch.Tensor, center: float, scale: float) -> torch.Tensor:
    return torch.exp(-torch.relu(value - center) / max(scale, 1e-6))


class BoardProxyEnergyRuntime(nn.Module):
    """Hand-calibrated continuous board score from marker/action proxies.

    This runtime is not intended to replace the learned ForceBand scorer.  It is
    an audit/control candidate for DP guidance: the score is continuous,
    non-saturating over typical marker ranges, differentiable, and decomposed
    into interpretable contact, smoothness, and action-stability terms.
    """

    def __init__(self, checkpoint_path: str | None = None, device: str = "cuda:0", config: BoardProxyEnergyConfig | None = None):
        super().__init__()
        del checkpoint_path
        self.device_name = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self.device = torch.device(self.device_name)
        self.config = config or BoardProxyEnergyConfig()

    def proxy_features(
        self,
        left_marker_seq: torch.Tensor,
        right_marker_seq: Optional[torch.Tensor] = None,
        eef_action_seq: Optional[torch.Tensor] = None,
        joint_action_seq: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        left = left_marker_seq.to(self.device).float()
        right = right_marker_seq.to(self.device).float() if right_marker_seq is not None else left
        batch, window = left.shape[:2]
        left_feat = marker_proxy_features_torch(left)
        right_feat = marker_proxy_features_torch(right)
        joint_action = _as_action(joint_action_seq, batch, window, 7, self.device)
        eef_action = _as_action(eef_action_seq, batch, window, 6, self.device)
        return {
            "left": left_feat,
            "right": right_feat,
            "diff": (left_feat - right_feat).abs(),
            "joint_action": action_proxy_features_torch(joint_action, action_dim=7),
            "eef_action": action_proxy_features_torch(eef_action, action_dim=6),
        }

    def forward(
        self,
        left_marker_seq: torch.Tensor,
        right_marker_seq: Optional[torch.Tensor] = None,
        eef_action_seq: Optional[torch.Tensor] = None,
        joint_action_seq: Optional[torch.Tensor] = None,
        task_id: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        del task_id
        cfg = self.config
        feat = self.proxy_features(left_marker_seq, right_marker_seq, eef_action_seq, joint_action_seq)
        left = feat["left"]
        right = feat["right"]
        joint = feat["joint_action"]

        mean_mag = 0.5 * (left[:, 0] + right[:, 0])
        area = 0.5 * (left[:, 6] + right[:, 6])
        marker_delta = 0.5 * (left[:, 12] + right[:, 12])
        centroid_delta = 0.5 * (left[:, 14] + right[:, 14])
        drift = 0.5 * (left[:, 17] + right[:, 17])
        action_speed = joint[:, 2]
        action_accel = joint[:, 5]

        contact = _band_score(mean_mag, cfg.mean_mag_center, cfg.mean_mag_scale)
        area_score = _high_score(area, cfg.area_center, cfg.area_scale)
        marker_smooth = _low_score(marker_delta, cfg.marker_delta_center, cfg.marker_delta_scale)
        centroid_smooth = _low_score(centroid_delta, cfg.centroid_delta_center, cfg.centroid_delta_scale)
        drift_score = _low_score(drift, cfg.drift_center, cfg.drift_scale)
        action_speed_score = _band_score(action_speed, cfg.action_speed_center, cfg.action_speed_scale)
        action_accel_score = _low_score(action_accel, cfg.action_accel_center, cfg.action_accel_scale)

        smooth = 0.75 * marker_smooth + 0.25 * centroid_smooth
        action = 0.55 * action_speed_score + 0.45 * action_accel_score
        score = (
            cfg.contact_weight * contact
            + cfg.area_weight * area_score
            + cfg.smooth_weight * smooth
            + cfg.center_weight * centroid_smooth
            + cfg.drift_weight * drift_score
            + cfg.action_weight * action
        )
        score = score / (
            cfg.contact_weight
            + cfg.area_weight
            + cfg.smooth_weight
            + cfg.center_weight
            + cfg.drift_weight
            + cfg.action_weight
        )

        return {
            "score": score,
            "energy_logit": torch.logit(score.clamp(1e-5, 1.0 - 1e-5)),
            "energy_clipped": torch.tanh(torch.logit(score.clamp(1e-5, 1.0 - 1e-5)) / 4.0) * 4.0,
            "quality_score": score,
            "p_good": score,
            "reason_prob": torch.stack([1.0 - score, score, torch.zeros_like(score), torch.zeros_like(score)], dim=-1),
            "components": {
                "contact": contact,
                "area": area_score,
                "smooth": smooth,
                "centroid_smooth": centroid_smooth,
                "drift": drift_score,
                "action": action,
            },
            "raw_proxy": {
                "mean_mag": mean_mag,
                "area": area,
                "marker_delta": marker_delta,
                "centroid_delta": centroid_delta,
                "drift": drift,
                "action_speed": action_speed,
                "action_accel": action_accel,
            },
            "config": asdict(cfg),
        }

    def score(
        self,
        left_marker_seq: torch.Tensor,
        right_marker_seq: Optional[torch.Tensor] = None,
        eef_action_seq: Optional[torch.Tensor] = None,
        joint_action_seq: Optional[torch.Tensor] = None,
        task_id: Optional[torch.Tensor] = None,
        mode: str = "quality",
    ) -> torch.Tensor:
        out = self.forward(left_marker_seq, right_marker_seq, eef_action_seq, joint_action_seq, task_id)
        if mode in {"quality", "score", "p_good", "reason_good"}:
            return out["score"]
        if mode == "energy":
            return out["energy_logit"]
        if mode == "energy_clipped":
            return out["energy_clipped"]
        if mode == "profile":
            return out["energy_clipped"]
        raise ValueError(f"Unknown BoardProxyEnergy score mode: {mode}")


def load_runtime(device: str = "cuda:0", config: BoardProxyEnergyConfig | None = None) -> BoardProxyEnergyRuntime:
    return BoardProxyEnergyRuntime(device=device, config=config)
