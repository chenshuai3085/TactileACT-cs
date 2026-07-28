"""
ForesightModule: wraps the ForesightTransformer for use with Pi0.

During training, takes the predicted clean actions (x_0) from flow matching
and predicts future tactile latent. The auxiliary loss guides action generation
toward tactile-aware trajectories.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from TFAC_V5.foresight_transformer import ForesightTransformer
from TFAC_V5.foresight_multistep import MultiStepSpatialForesightTransformer


class ForesightModule(nn.Module):
    """
    Lightweight wrapper around ForesightTransformer for Pi0 integration.

    Takes:
      - action_pred: (B, action_horizon, action_dim) — predicted clean actions
      - tac_latent_tokens: (B, 9, latent_dim) — current tactile spatial tokens (from VAE)
      - qpos: (B, state_dim) — current proprioceptive state

    Returns:
      - z_pred: (B, foresight_out_dim) — predicted future tactile latent
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        action_dim: int = 7,
        num_layers: int = 3,
        nheads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        vae_latent_dim: int = 16,
        n_tactile_spatial: int = 9,
        predict_horizon: int = 1,
        max_action_len: int = 30,
        checkpoint_path: str = "",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vae_latent_dim = vae_latent_dim
        self.n_tactile_spatial = n_tactile_spatial
        self.foresight_out_dim = vae_latent_dim * 3 * 3  # 144
        self.predict_horizon = int(predict_horizon)

        # Projection: VAE latent_dim → hidden_dim (for tactile tokens)
        self.tac_proj = nn.Linear(vae_latent_dim, hidden_dim)
        self.tac_spatial_pos = nn.Parameter(
            torch.randn(n_tactile_spatial, 1, hidden_dim) * 0.02
        )

        if self.predict_horizon > 1:
            self.foresight = MultiStepSpatialForesightTransformer(
                d_model=hidden_dim,
                action_dim=action_dim,
                num_layers=num_layers,
                nhead=nheads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                latent_dim=vae_latent_dim,
                n_tactile_spatial=n_tactile_spatial,
                predict_horizon=self.predict_horizon,
                max_history=1,
                state_dim=action_dim,
                max_action_len=max_action_len,
            )
        else:
            # ForesightTransformer (reuse existing single-step implementation).
            self.foresight = ForesightTransformer(
                d_model=hidden_dim,
                action_dim=action_dim,
                num_layers=num_layers,
                nhead=nheads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                tactile_out_dim=self.foresight_out_dim,
                tactile_decoder_type="linear",
                max_history=1,
                predict_horizon=1,
                state_dim=action_dim,
                n_tactile_spatial=n_tactile_spatial,
                max_action_len=max_action_len,
            )

        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_foresight_weights(checkpoint_path)

    def _load_foresight_weights(self, ckpt_path: str):
        """Load pretrained foresight weights (from TFAC_V5 pretraining)."""
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt)

        def strip_prefixes(prefixes: tuple[str, ...]) -> dict[str, torch.Tensor]:
            out = {}
            for key, value in state_dict.items():
                for prefix in prefixes:
                    if key.startswith(prefix):
                        out[key[len(prefix):]] = value
                        break
            return out

        # Extract foresight-related weights from direct, DDP, or wrapped checkpoints.
        foresight_keys = strip_prefixes((
            "foresight.",
            "module.foresight.",
            "model.foresight.",
        ))
        if not foresight_keys:
            own_keys = set(self.foresight.state_dict().keys())
            foresight_keys = {
                key: value for key, value in state_dict.items()
                if key in own_keys
            }
        own_state = self.foresight.state_dict()
        skipped_shape = [
            key for key, value in foresight_keys.items()
            if key in own_state and own_state[key].shape != value.shape
        ]
        foresight_keys = {
            key: value for key, value in foresight_keys.items()
            if key in own_state and own_state[key].shape == value.shape
        }

        if foresight_keys:
            missing, unexpected = self.foresight.load_state_dict(
                foresight_keys, strict=False
            )
            print(f"[ForesightModule] Loaded foresight weights: "
                  f"{len(foresight_keys)} keys, "
                  f"missing={len(missing)}, unexpected={len(unexpected)}, "
                  f"shape_skipped={len(skipped_shape)}")
        else:
            print(f"[ForesightModule] Warning: no 'foresight.*' keys in {ckpt_path}")

        # Try to load tactile latent projection weights.
        tac_proj_keys = strip_prefixes((
            "tac_latent_proj.",
            "module.tac_latent_proj.",
            "model.tac_latent_proj.",
            "tac_proj.",
        ))
        if tac_proj_keys:
            missing, unexpected = self.tac_proj.load_state_dict(tac_proj_keys, strict=False)
            print(
                "[ForesightModule] Loaded tac_proj weights: "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )

        for key in (
            "tac_spatial_pos_embed",
            "module.tac_spatial_pos_embed",
            "model.tac_spatial_pos_embed",
            "tac_spatial_pos",
        ):
            if key in state_dict and state_dict[key].shape == self.tac_spatial_pos.shape:
                with torch.no_grad():
                    self.tac_spatial_pos.copy_(state_dict[key])
                print("[ForesightModule] Loaded tactile spatial position embedding")
                break

    def forward(
        self,
        action_pred: torch.Tensor,
        tac_latent: torch.Tensor,
        qpos: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Predict future tactile latent given predicted actions and current tactile state.

        Args:
            action_pred: (B, action_horizon, action_dim) — predicted clean actions
            tac_latent:  (B, C, 3, 3) — current tactile VAE latent (from encoder)
            qpos:        (B, state_dim) — proprioceptive state

        Returns:
            z_pred: (B, 144) or (B, H, 144) — predicted future tactile latent
        """
        B = action_pred.shape[0]
        C = self.vae_latent_dim

        # Reshape VAE latent to spatial tokens: (B, C, 3, 3) → (9, B, C) → project → (9, B, D)
        z_flat = tac_latent.reshape(B, C, 9).permute(2, 0, 1)  # (9, B, C)
        t_tokens = self.tac_proj(z_flat)  # (9, B, hidden_dim)
        t_tokens = t_tokens + self.tac_spatial_pos  # add spatial pos

        # No vision tokens in this standalone module (pi0 handles vision in prefix)
        # Use empty vision tokens placeholder
        v_tokens = torch.zeros(
            0, B, self.hidden_dim,
            device=action_pred.device, dtype=action_pred.dtype
        )

        # ForesightTransformer expects:
        #   v_tokens: (N_v, B, D), t_tokens: (N_t, B, D), a1: (B, T, action_dim)
        t_hat, _, _ = self.foresight(
            v_tokens, t_tokens, action_pred, n_v=0, proprio=qpos
        )

        return t_hat  # (B, 144)
