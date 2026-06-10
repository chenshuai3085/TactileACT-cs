"""Differentiable tactile/action proxy features for TacQualityEnergy."""

from __future__ import annotations

import torch


DEFAULT_WINDOW = 8


def ensure_window(x: torch.Tensor, window: int = DEFAULT_WINDOW) -> torch.Tensor:
    if x.shape[1] == window:
        return x
    if x.shape[1] > window:
        return x[:, -window:]
    pad = x[:, :1].expand(-1, window - x.shape[1], *x.shape[2:])
    return torch.cat([pad, x], dim=1)


def marker_proxy_features_torch(marker_seq: torch.Tensor, window: int = DEFAULT_WINDOW) -> torch.Tensor:
    """Compute differentiable marker proxy features.

    Args:
        marker_seq: tensor shaped ``(B, T, 9, 9, 2)``.

    Returns:
        Tensor shaped ``(B, 18)`` summarizing contact intensity, area, center,
        spread, temporal smoothness, and first-to-last drift.
    """

    marker_seq = ensure_window(marker_seq.float(), window)
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


def action_proxy_features_torch(
    action_seq: torch.Tensor,
    action_dim: int,
    window: int = DEFAULT_WINDOW,
) -> torch.Tensor:
    """Compute differentiable action chunk proxy features."""

    action_seq = ensure_window(action_seq.float(), window)
    if action_seq.shape[-1] > action_dim:
        action_seq = action_seq[..., :action_dim]
    elif action_seq.shape[-1] < action_dim:
        pad = torch.zeros(*action_seq.shape[:-1], action_dim - action_seq.shape[-1], dtype=action_seq.dtype, device=action_seq.device)
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
