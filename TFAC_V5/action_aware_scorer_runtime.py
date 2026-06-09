"""Runtime utilities for the action-aware tactile quality scorer.

This module is intentionally free of dataset-loading code.  It is meant to be
used inside DP reranking / classifier guidance:

    action -> Foresight -> predicted marker -> ActionAwareScorerRuntime.score()
                                      ^                 |
                                      |_________________|
                                             gradient

The scorer is differentiable with respect to both marker and action because the
physical proxy features are implemented in torch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ACTION_DIM = 6
WINDOW = 8
TASK_TO_ID = {"insertion": 0, "board": 1}


class ActionAwareMarkerScorer(nn.Module):
    def __init__(self, marker_proxy_dim=18, action_proxy_dim=10, action_dim=ACTION_DIM, hidden=160, dropout=0.0):
        super().__init__()
        self.action_dim = action_dim
        self.marker_encoder = nn.Sequential(
            nn.Conv3d(2, 24, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.GroupNorm(6, 24),
            nn.SiLU(),
            nn.Conv3d(24, 48, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
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
        in_dim = 48 * 2 * 3 * 3 + 48 * 4 + marker_proxy_dim + action_proxy_dim + 2
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
        self.t4_head = nn.Linear(hidden // 2, 4)
        self.score_head = nn.Linear(hidden // 2, 1)

    def forward(self, marker, marker_proxy, action, action_proxy, task_id):
        marker_feat = self.marker_encoder(marker.permute(0, 4, 1, 2, 3))
        action_feat = self.action_encoder(action.permute(0, 2, 1))
        task_oh = F.one_hot(task_id.long(), num_classes=2).float()
        h = self.encoder(torch.cat([marker_feat, action_feat, marker_proxy, action_proxy, task_oh], dim=-1))
        return {
            "binary_logits": self.binary_head(h),
            "t4_logits": self.t4_head(h),
            "score": self.score_head(h).squeeze(-1),
        }


def _ensure_window(x: torch.Tensor, window: int = WINDOW) -> torch.Tensor:
    if x.shape[1] == window:
        return x
    if x.shape[1] > window:
        return x[:, -window:]
    pad = x[:, :1].expand(-1, window - x.shape[1], *x.shape[2:])
    return torch.cat([pad, x], dim=1)


def marker_proxy_features_torch(marker_seq: torch.Tensor) -> torch.Tensor:
    """Differentiable marker proxy features.

    Args:
        marker_seq: (B, T, 9, 9, 2), raw marker_offset.
    Returns:
        (B, 18) feature tensor matching evaluate_marker_proxy_scorer.py.
    """
    marker_seq = _ensure_window(marker_seq.float())
    bsz, timesteps = marker_seq.shape[:2]
    mag = torch.linalg.norm(marker_seq, dim=-1)
    flat = mag.reshape(bsz, timesteps, -1)
    mean_mag_t = flat.mean(dim=-1)
    max_mag_t = flat.max(dim=-1).values
    p90_mag_t = torch.quantile(flat, 0.90, dim=-1)

    thresh = torch.clamp(0.25 * max_mag_t, min=1e-6)
    area_t = (flat > thresh.unsqueeze(-1)).float().mean(dim=-1)

    yy, xx = torch.meshgrid(
        torch.arange(9, device=marker_seq.device, dtype=marker_seq.dtype),
        torch.arange(9, device=marker_seq.device, dtype=marker_seq.dtype),
        indexing="ij",
    )
    coords = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
    weights = flat + 1e-6
    wsum = weights.sum(dim=-1, keepdim=True)
    centroid = weights @ coords / wsum
    centered = coords.view(1, 1, 81, 2) - centroid.unsqueeze(2)
    spread = torch.sqrt((weights.unsqueeze(-1) * centered.square()).sum(dim=2) / wsum)

    if timesteps > 1:
        d_marker = torch.linalg.norm(marker_seq[:, 1:] - marker_seq[:, :-1], dim=-1).reshape(bsz, timesteps - 1, -1)
        d_centroid = torch.linalg.norm(centroid[:, 1:] - centroid[:, :-1], dim=-1)
        d_mean_mag = (mean_mag_t[:, 1:] - mean_mag_t[:, :-1]).abs()
    else:
        d_marker = torch.zeros(bsz, 1, 81, device=marker_seq.device, dtype=marker_seq.dtype)
        d_centroid = torch.zeros(bsz, 1, device=marker_seq.device, dtype=marker_seq.dtype)
        d_mean_mag = torch.zeros(bsz, 1, device=marker_seq.device, dtype=marker_seq.dtype)

    half = max(1, timesteps // 2)
    first_mean = torch.linalg.norm(marker_seq[:, :half], dim=-1).mean(dim=(1, 2, 3))
    second_mean = torch.linalg.norm(marker_seq[:, half:], dim=-1).mean(dim=(1, 2, 3))

    return torch.stack(
        [
            mean_mag_t.mean(dim=1),
            mean_mag_t.std(dim=1, unbiased=False),
            mean_mag_t[:, -1],
            max_mag_t.mean(dim=1),
            max_mag_t[:, -1],
            p90_mag_t.mean(dim=1),
            area_t.mean(dim=1),
            area_t[:, -1],
            centroid[:, :, 0].mean(dim=1) / 8.0,
            centroid[:, :, 1].mean(dim=1) / 8.0,
            spread[:, :, 0].mean(dim=1) / 8.0,
            spread[:, :, 1].mean(dim=1) / 8.0,
            d_marker.mean(dim=(1, 2)),
            torch.quantile(d_marker.reshape(bsz, -1), 0.90, dim=1),
            d_centroid.mean(dim=1),
            d_mean_mag.mean(dim=1),
            second_mean - first_mean,
            torch.linalg.norm((marker_seq[:, -1] - marker_seq[:, 0]).reshape(bsz, -1), dim=1),
        ],
        dim=-1,
    )


def action_proxy_features_torch(action_seq: torch.Tensor, action_dim: int = ACTION_DIM) -> torch.Tensor:
    """Differentiable action proxy features from candidate action chunks."""
    action_seq = _ensure_window(action_seq.float())
    if action_seq.shape[-1] > action_dim:
        action_seq = action_seq[..., :action_dim]
    elif action_seq.shape[-1] < action_dim:
        pad = torch.zeros(*action_seq.shape[:-1], action_dim - action_seq.shape[-1], device=action_seq.device)
        action_seq = torch.cat([action_seq, pad], dim=-1)
    delta = action_seq[:, 1:] - action_seq[:, :-1] if action_seq.shape[1] > 1 else torch.zeros_like(action_seq[:, :1])
    speed = torch.linalg.norm(delta, dim=-1)
    accel = delta[:, 1:] - delta[:, :-1] if delta.shape[1] > 1 else torch.zeros_like(delta[:, :1])
    accel_norm = torch.linalg.norm(accel, dim=-1)
    return torch.stack(
        [
            torch.linalg.norm(action_seq, dim=-1).mean(dim=1),
            torch.linalg.norm(action_seq, dim=-1).std(dim=1, unbiased=False),
            speed.mean(dim=1),
            speed.std(dim=1, unbiased=False),
            torch.quantile(speed, 0.90, dim=1),
            accel_norm.mean(dim=1),
            torch.quantile(accel_norm, 0.90, dim=1),
            torch.linalg.norm(action_seq[:, -1] - action_seq[:, 0], dim=-1),
            delta.abs().mean(dim=(1, 2)),
            delta.abs().amax(dim=(1, 2)),
        ],
        dim=-1,
    )


def _to_tensor(value, device):
    return torch.as_tensor(value, dtype=torch.float32, device=device)


class ActionAwareScorerRuntime(nn.Module):
    """Checkpoint-backed scorer with differentiable preprocessing."""

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        super().__init__()
        self.device_name = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self.device = torch.device(self.device_name)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.action_dim = int(ckpt.get("action_dim", ACTION_DIM))
        self.model = ActionAwareMarkerScorer(
            marker_proxy_dim=int(ckpt["marker_proxy_dim"]),
            action_proxy_dim=int(ckpt["action_proxy_dim"]),
            action_dim=self.action_dim,
            dropout=0.0,
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.register_buffer("marker_mean", _to_tensor(ckpt["marker_mean"], self.device).view(1, 1, 1, 1, 2))
        self.register_buffer("marker_std", _to_tensor(ckpt["marker_std"], self.device).view(1, 1, 1, 1, 2))
        self.register_buffer("action_mean", _to_tensor(ckpt["action_mean"], self.device).view(1, 1, self.action_dim))
        self.register_buffer("action_std", _to_tensor(ckpt["action_std"], self.device).view(1, 1, self.action_dim))
        self.register_buffer("marker_proxy_mean", _to_tensor(ckpt["marker_proxy_mean"], self.device).view(1, -1))
        self.register_buffer("marker_proxy_scale", _to_tensor(ckpt["marker_proxy_scale"], self.device).view(1, -1))
        self.register_buffer("action_proxy_mean", _to_tensor(ckpt["action_proxy_mean"], self.device).view(1, -1))
        self.register_buffer("action_proxy_scale", _to_tensor(ckpt["action_proxy_scale"], self.device).view(1, -1))

    def forward(self, marker_seq: torch.Tensor, action_seq: torch.Tensor, task_id: torch.Tensor) -> Dict[str, torch.Tensor]:
        marker_seq = _ensure_window(marker_seq.to(self.device).float())
        action_seq = _ensure_window(action_seq.to(self.device).float())
        if action_seq.shape[-1] > self.action_dim:
            action_seq = action_seq[..., : self.action_dim]
        elif action_seq.shape[-1] < self.action_dim:
            pad = torch.zeros(*action_seq.shape[:-1], self.action_dim - action_seq.shape[-1], device=action_seq.device)
            action_seq = torch.cat([action_seq, pad], dim=-1)

        marker_proxy = marker_proxy_features_torch(marker_seq)
        action_proxy = action_proxy_features_torch(action_seq, self.action_dim)
        marker_norm = (marker_seq - self.marker_mean) / self.marker_std
        action_norm = (action_seq - self.action_mean) / self.action_std
        marker_proxy_norm = (marker_proxy - self.marker_proxy_mean) / self.marker_proxy_scale
        action_proxy_norm = (action_proxy - self.action_proxy_mean) / self.action_proxy_scale
        return self.model(marker_norm, marker_proxy_norm, action_norm, action_proxy_norm, task_id.to(self.device))

    def score(self, marker_seq: torch.Tensor, action_seq: torch.Tensor, task_id: torch.Tensor, mode: str = "log_p_good") -> torch.Tensor:
        out = self.forward(marker_seq, action_seq, task_id)
        if mode == "log_p_good":
            return F.log_softmax(out["binary_logits"], dim=-1)[:, 1]
        if mode == "p_good":
            return torch.softmax(out["binary_logits"], dim=-1)[:, 1]
        if mode == "quality":
            return torch.sigmoid(out["score"])
        if mode == "hybrid":
            return F.log_softmax(out["binary_logits"], dim=-1)[:, 1] + 0.5 * torch.sigmoid(out["score"])
        raise ValueError(f"Unknown score mode: {mode}")


def self_test(checkpoint_path: str, out_path: str, device: str):
    runtime = ActionAwareScorerRuntime(checkpoint_path, device=device)
    bsz = 4
    marker = torch.randn(bsz, WINDOW, 9, 9, 2, device=runtime.device, requires_grad=True)
    action = torch.randn(bsz, WINDOW, runtime.action_dim, device=runtime.device, requires_grad=True)
    task_id = torch.tensor([0, 0, 1, 1], device=runtime.device)
    score = runtime.score(marker, action, task_id, mode="hybrid")
    grad_marker, grad_action = torch.autograd.grad(score.sum(), [marker, action])
    result = {
        "checkpoint": checkpoint_path,
        "device": str(runtime.device),
        "score": score.detach().cpu().tolist(),
        "grad_marker_norm": float(grad_marker.norm().detach().cpu()),
        "grad_action_norm": float(grad_action.norm().detach().cpu()),
        "grad_marker_finite": bool(torch.isfinite(grad_marker).all().detach().cpu()),
        "grad_action_finite": bool(torch.isfinite(grad_action).all().detach().cpu()),
        "usable_for_guidance": bool(
            torch.isfinite(grad_action).all().detach().cpu() and grad_action.norm().detach().cpu() > 0
        ),
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="/home/chenshuai/Project/output/action_aware_marker_scorer/action_aware_marker_scorer_final.pt",
    )
    parser.add_argument(
        "--out",
        default="/home/chenshuai/Project/output/action_aware_marker_scorer/runtime_gradient_sanity.json",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    self_test(args.checkpoint, args.out, args.device)
