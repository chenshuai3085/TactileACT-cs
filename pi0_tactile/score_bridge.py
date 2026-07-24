"""Differentiable pi0.5 action -> tactile foresight -> quality score bridge."""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn

from pi0_tactile.model import Pi0Tactile


ActionTransform = Callable[[torch.Tensor], torch.Tensor]


class Pi0TactileForesightScoreBridge(nn.Module):
    """Score pi0/pi0.5 clean robot actions through TactileVAE/Foresight.

    The input action is the executable robot slice from pi0.5's model action
    space, usually normalized and shaped ``(B, H, 7)``.  The bridge predicts a
    future tactile latent with ``model.foresight_module``, decodes it with the
    TactileVAE decoder, converts the marker output to raw marker units, and
    calls a differentiable TacQuality scorer.
    """

    def __init__(
        self,
        model: Pi0Tactile,
        *,
        marker_offset_norm: torch.Tensor,
        qpos_norm: torch.Tensor,
        scorer: nn.Module,
        action_to_scorer: Optional[ActionTransform] = None,
        marker_mean: tuple[float, float] = (0.572, -1.786),
        marker_std: tuple[float, float] = (1.596, 3.845),
        score_mode: str = "energy_clipped",
        task_id: Optional[int] = None,
        score_window: int = 8,
        mirror_right_marker: bool = True,
    ):
        super().__init__()
        self.model_ref = model
        self.scorer = scorer
        self.action_to_scorer = action_to_scorer
        self.score_mode = score_mode
        self.task_id_value = task_id
        self.score_window = int(score_window)
        self.mirror_right_marker = bool(mirror_right_marker)

        device = next(model.parameters(), torch.empty(0)).device
        marker = marker_offset_norm.detach().clone().float().to(device)
        qpos = qpos_norm.detach().clone().float().to(device)
        if qpos.shape[-1] != model.config.robot_action_dim:
            qpos = model.action_adapter.slice_robot_action(qpos)

        self.register_buffer("marker_offset_norm", marker, persistent=False)
        self.register_buffer("qpos_norm", qpos, persistent=False)
        self.register_buffer(
            "marker_mean",
            torch.tensor(marker_mean, dtype=torch.float32, device=device).view(1, 1, 1, 1, 2),
            persistent=False,
        )
        self.register_buffer(
            "marker_std",
            torch.tensor(marker_std, dtype=torch.float32, device=device).view(1, 1, 1, 1, 2),
            persistent=False,
        )

        # Guidance is a serving-time optimizer; freeze consequence/scorer params.
        self.model_ref.foresight_module.eval().requires_grad_(False)
        self.model_ref.tactile_encoder.vae.eval().requires_grad_(False)
        self.scorer.eval().requires_grad_(False)

        with torch.no_grad():
            z_current, _ = self.model_ref.tactile_encoder.vae.encode_single_frame(self.marker_offset_norm)
        self.register_buffer("z_current", z_current.detach(), persistent=False)

    def _expand_batch(self, tensor: torch.Tensor, batch: int) -> torch.Tensor:
        if tensor.shape[0] == batch:
            return tensor
        if tensor.shape[0] != 1:
            raise ValueError(f"Cannot expand batch {tensor.shape[0]} to {batch}")
        return tensor.expand(batch, *tensor.shape[1:])

    def _decode_marker_raw(self, z_pred: torch.Tensor) -> torch.Tensor:
        if z_pred.dim() == 3:
            z_pred = z_pred[:, -1]
        if z_pred.dim() != 2:
            raise ValueError(f"Expected z_pred shape (B,D) or (B,L,D), got {tuple(z_pred.shape)}")
        batch = z_pred.shape[0]
        latent_dim = self.model_ref.config.vae_latent_dim
        z_spatial = z_pred.reshape(batch, latent_dim, 3, 3)
        marker_norm = self.model_ref.tactile_encoder.vae.decoder(z_spatial)
        marker_norm = marker_norm.view(batch, 1, 9, 9, 2)
        marker_raw = marker_norm * self.marker_std + self.marker_mean
        return marker_raw.expand(batch, self.score_window, 9, 9, 2)

    def forward(self, action_robot_norm: torch.Tensor) -> torch.Tensor:
        action_robot_norm = action_robot_norm.float()
        batch = action_robot_norm.shape[0]
        z_current = self._expand_batch(self.z_current, batch)
        qpos = self._expand_batch(self.qpos_norm, batch)
        z_pred = self.model_ref.foresight_module(
            action_pred=action_robot_norm,
            tac_latent=z_current,
            qpos=qpos,
        )
        left_marker = self._decode_marker_raw(z_pred)
        right_marker = left_marker if self.mirror_right_marker else None

        if self.action_to_scorer is not None:
            action_for_score = self.action_to_scorer(action_robot_norm)
        else:
            action_for_score = action_robot_norm
        action_for_score = action_for_score[:, : self.score_window]
        if action_for_score.shape[1] < self.score_window:
            pad = action_for_score[:, -1:].expand(
                batch,
                self.score_window - action_for_score.shape[1],
                action_for_score.shape[-1],
            )
            action_for_score = torch.cat([action_for_score, pad], dim=1)

        eef_action = action_for_score[..., :6]
        if eef_action.shape[-1] < 6:
            eef_action = torch.nn.functional.pad(eef_action, (0, 6 - eef_action.shape[-1]))
        task_id = None
        if self.task_id_value is not None:
            task_id = torch.full(
                (batch,),
                int(self.task_id_value),
                dtype=torch.long,
                device=action_robot_norm.device,
            )
        return self.scorer.score(
            left_marker,
            right_marker_seq=right_marker,
            eef_action_seq=eef_action,
            joint_action_seq=action_for_score,
            task_id=task_id,
            mode=self.score_mode,
        )
