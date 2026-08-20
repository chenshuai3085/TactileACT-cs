"""Differentiable Foresight-to-TacQuality bridge.

The bridge implements the serving-time contract:

The formal path is ``raw action -> Foresight -> predicted tactile latent``.
The decoded-marker output remains available for legacy action-aware scorers.

It is intentionally narrow and does not do candidate reranking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ForesightBridgeConfig:
    task: str = "board"
    window: int = 8
    action_chunk: int = 10
    latent_dim: int = 16
    marker_mean: Tuple[float, float] = (0.2102, -0.6422)
    marker_std: Tuple[float, float] = (1.6805, 3.6717)
    residual_prediction: bool = False
    board_right_source: str = "mirror_left"


def _normalize_action(action_raw: torch.Tensor, fs_norm: Mapping[str, Any]) -> torch.Tensor:
    mean = torch.as_tensor(fs_norm["action_mean"], dtype=action_raw.dtype, device=action_raw.device).view(1, 1, -1)
    std = torch.as_tensor(fs_norm["action_std"], dtype=action_raw.dtype, device=action_raw.device).view(1, 1, -1)
    return (action_raw - mean) / std.clamp_min(1e-8)


def _normalize_qpos(qpos_raw: torch.Tensor, fs_norm: Mapping[str, Any]) -> torch.Tensor:
    mean = torch.as_tensor(fs_norm["qpos_mean"], dtype=qpos_raw.dtype, device=qpos_raw.device).view(1, -1)
    std = torch.as_tensor(fs_norm["qpos_std"], dtype=qpos_raw.dtype, device=qpos_raw.device).view(1, -1)
    return (qpos_raw - mean) / std.clamp_min(1e-8)


def _expand_batch(tensor: torch.Tensor, batch: int) -> torch.Tensor:
    if tensor.shape[0] == batch:
        return tensor
    if tensor.shape[0] != 1:
        raise ValueError(f"Cannot expand tensor with batch {tensor.shape[0]} to {batch}")
    return tensor.expand(batch, *tensor.shape[1:])


def _last_or_sequence(z_pred: torch.Tensor) -> torch.Tensor:
    if z_pred.dim() == 2:
        return z_pred.unsqueeze(1)
    if z_pred.dim() == 3:
        return z_pred
    raise ValueError(f"Expected z_pred shape (B,D) or (B,L,D), got {tuple(z_pred.shape)}")


class ForesightTacQualityBridge(nn.Module):
    """Wrap a trained Foresight model as a differentiable TacQuality predict fn."""

    def __init__(
        self,
        foresight: nn.Module,
        fs_norm: Mapping[str, Any],
        *,
        qpos_raw: torch.Tensor,
        foresight_images: Sequence[torch.Tensor],
        marker_window_norm: Optional[torch.Tensor] = None,
        config: Optional[ForesightBridgeConfig] = None,
    ):
        super().__init__()
        self.foresight = foresight
        self.fs_norm = fs_norm
        self.config = config or ForesightBridgeConfig()
        device = next(foresight.parameters(), torch.empty(0)).device
        # Serving can build this bridge while the outer DP loop is in
        # torch.inference_mode().  Classifier guidance later backpropagates
        # through the bridge, so cached constants must be normal tensors.
        with torch.inference_mode(False):
            qpos_buf = qpos_raw.detach().clone().float().to(device).view(1, -1)
            image_bufs = [img.detach().clone().float().to(device) for img in foresight_images]
            marker_window_buf = (
                marker_window_norm.detach().clone().float().to(device)
                if marker_window_norm is not None
                else None
            )
            marker_mean = torch.tensor(self.config.marker_mean, dtype=torch.float32, device=device).view(1, 1, 1, 1, 2)
            marker_std = torch.tensor(self.config.marker_std, dtype=torch.float32, device=device).view(1, 1, 1, 1, 2)

        self.register_buffer("qpos_raw", qpos_buf, persistent=False)
        self.foresight_images = image_bufs
        if marker_window_buf is not None:
            self.register_buffer("marker_window_norm", marker_window_buf, persistent=False)
        else:
            self.marker_window_norm = None
        self.register_buffer("marker_mean", marker_mean, persistent=False)
        self.register_buffer("marker_std", marker_std, persistent=False)

    @property
    def device(self) -> torch.device:
        return self.qpos_raw.device

    def _build_images(self, batch: int) -> List[torch.Tensor]:
        images = [_expand_batch(img.to(self.device), batch) for img in self.foresight_images]
        if self.marker_window_norm is not None:
            images.append(_expand_batch(self.marker_window_norm.to(self.device), batch))
        return images

    def _call_foresight(self, images: Sequence[torch.Tensor], action_fs_norm: torch.Tensor, qpos_norm: torch.Tensor):
        try:
            return self.foresight(images, action_fs_norm, future_images=None, qpos=qpos_norm)
        except TypeError:
            return self.foresight(images, action_fs_norm, qpos=qpos_norm)

    def _apply_residual_if_needed(self, z_seq: torch.Tensor, batch: int) -> torch.Tensor:
        if not self.config.residual_prediction:
            return z_seq
        if self.marker_window_norm is None or not hasattr(self.foresight, "tactile_vae"):
            raise ValueError("residual_prediction requires marker_window_norm and foresight.tactile_vae")
        marker_win = _expand_batch(self.marker_window_norm.to(self.device), batch)
        z_cur_raw, _ = self.foresight.tactile_vae.encode_single_frame(marker_win)
        z_cur = z_cur_raw.reshape(batch, 1, -1)
        return z_seq + z_cur.expand_as(z_seq)

    def decode_latent_to_marker_seq(self, z_pred: torch.Tensor, *, apply_residual: bool = True) -> torch.Tensor:
        z_seq = _last_or_sequence(z_pred)
        batch, length, dim = z_seq.shape
        expected = self.config.latent_dim * 3 * 3
        if dim != expected:
            raise ValueError(f"Expected latent dim {expected}, got {dim}")
        if apply_residual:
            z_seq = self._apply_residual_if_needed(z_seq, batch)
        z_spatial = z_seq.reshape(batch * length, self.config.latent_dim, 3, 3)
        marker_norm = self.foresight.tactile_vae.decoder(z_spatial)
        marker_norm = marker_norm.view(batch, length, 9, 9, 2)
        marker_raw = marker_norm * self.marker_std + self.marker_mean
        if length >= self.config.window:
            return marker_raw[:, -self.config.window :]
        return marker_raw[:, -1:].expand(batch, self.config.window, 9, 9, 2)

    def predict_latent(self, action_raw: torch.Tensor) -> torch.Tensor:
        """Predict the tactile latent consequence while preserving action gradients."""

        action_raw = action_raw.to(self.device).float()
        batch = action_raw.shape[0]
        action_fs = action_raw[:, : self.config.action_chunk, :]
        action_fs_norm = _normalize_action(action_fs, self.fs_norm)
        qpos = _expand_batch(self.qpos_raw, batch)
        qpos_norm = _normalize_qpos(qpos, self.fs_norm)
        outputs = self._call_foresight(self._build_images(batch), action_fs_norm, qpos_norm)
        z_pred = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        z_seq = _last_or_sequence(z_pred)
        expected = self.config.latent_dim * 3 * 3
        if z_seq.shape[-1] != expected:
            raise ValueError(f"Expected latent dim {expected}, got {z_seq.shape[-1]}")
        return self._apply_residual_if_needed(z_seq, batch)

    def forward(self, action_raw: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Legacy decoded-marker contract used by action-aware ablations."""

        action_raw = action_raw.to(self.device).float()
        z_pred = self.predict_latent(action_raw)
        left_marker = self.decode_latent_to_marker_seq(z_pred, apply_residual=False)
        action_seq = action_raw[:, : self.config.window, :]
        result: Dict[str, torch.Tensor] = {
            "left_marker_seq": left_marker,
            "eef_action_seq": action_seq[..., :6] if action_seq.shape[-1] >= 6 else F.pad(action_seq, (0, 6 - action_seq.shape[-1])),
        }
        if self.config.task.lower() == "board":
            if self.config.board_right_source != "mirror_left":
                raise ValueError(f"Unsupported board_right_source={self.config.board_right_source!r}")
            result["right_marker_seq"] = left_marker
        return result


class ForesightTactileOnlyLatentBridge(ForesightTacQualityBridge):
    """Formal bridge whose output is only the predicted tactile latent chunk."""

    def forward(self, action_raw: torch.Tensor) -> torch.Tensor:
        return self.predict_latent(action_raw)


class SyntheticTactileVAE(nn.Module):
    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.latent_dim = latent_dim
        self.decoder_head = nn.Linear(latent_dim * 3 * 3, 9 * 9 * 2)

    def decoder(self, z_spatial: torch.Tensor) -> torch.Tensor:
        flat = z_spatial.flatten(1)
        return self.decoder_head(flat).view(z_spatial.shape[0], 9, 9, 2)


class SyntheticLatentForesight(nn.Module):
    """Tiny differentiable Foresight stand-in for serving dry-run smoke tests."""

    def __init__(self, action_dim: int = 7, latent_dim: int = 16, pred_steps: int = 4):
        super().__init__()
        self.pred_steps = pred_steps
        self.latent_dim = latent_dim
        self.tactile_vae = SyntheticTactileVAE(latent_dim)
        self.action_proj = nn.Linear(action_dim, pred_steps * latent_dim * 3 * 3)
        self.qpos_proj = nn.Linear(action_dim, pred_steps * latent_dim * 3 * 3)

    def forward(self, images: Sequence[torch.Tensor], action: torch.Tensor, future_images=None, qpos=None):
        del images, future_images
        pooled = action.mean(dim=1)
        qpos_term = 0.0 if qpos is None else self.qpos_proj(qpos)
        z = self.action_proj(pooled) + qpos_term
        z = z.view(action.shape[0], self.pred_steps, self.latent_dim * 3 * 3)
        z_current = z[:, 0]
        return z, None, None, z_current, None, None
