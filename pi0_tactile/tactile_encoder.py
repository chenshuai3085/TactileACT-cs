"""
TactileEncoder: wraps the frozen TactileVAE to produce tokens
that can be prepended to Pi0's prefix (alongside image/lang tokens).

Input:  marker_offset sequence (B, T_hist, 9, 9, 2)
Output: tac_tokens (B, N_tac, width) ready for PaliGemma attention
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from TFAC_V5.tactile_vae import TactileVAE


class TactileTokenEncoder(nn.Module):
    """
    Frozen TactileVAE encoder → trainable projection to PaliGemma width.

    VAE maps (B, T, 9, 9, 2) → latent (B, C, T', 3, 3).
    We take the last temporal frame's latent (B, C, 3, 3),
    reshape to 9 spatial tokens of dim C, then project to `width`.
    """

    def __init__(
        self,
        vae_checkpoint: str,
        vae_latent_dim: int = 16,
        tac_history: int = 8,
        width: int = 1024,  # Action Expert width (Gemma 300M)
        freeze_vae: bool = True,
    ):
        super().__init__()
        self.vae_latent_dim = vae_latent_dim
        self.tac_history = tac_history
        self.n_spatial_tokens = 9  # 3x3

        # Frozen TactileVAE
        self.vae = TactileVAE(latent_dim=vae_latent_dim, temporal_window=tac_history)
        if vae_checkpoint and os.path.exists(vae_checkpoint):
            ckpt = torch.load(vae_checkpoint, map_location="cpu")
            state_dict = ckpt.get("model_state_dict", ckpt)
            self.vae.load_state_dict(state_dict)
            print(f"[TactileTokenEncoder] Loaded VAE from: {vae_checkpoint}")
        if freeze_vae:
            self.vae.requires_grad_(False)
            self.vae.eval()

        # Trainable projection: latent_dim → width
        self.proj = nn.Linear(vae_latent_dim, width)

        # Learnable spatial position embedding for 9 tokens
        self.spatial_pos_embed = nn.Parameter(
            torch.randn(1, self.n_spatial_tokens, width) * 0.02
        )

    def forward(self, marker_offset: torch.Tensor) -> torch.Tensor:
        """
        Args:
            marker_offset: (B, T_hist, 9, 9, 2) marker displacement sequence

        Returns:
            tac_tokens: (B, 9, width) tactile tokens ready for prefix
        """
        B = marker_offset.shape[0]

        with torch.no_grad():
            # (B, T, 9, 9, 2) → z_last (B, C, 3, 3)
            z_last, _ = self.vae.encode_single_frame(marker_offset)

        # Reshape: (B, C, 3, 3) → (B, 9, C)
        z_flat = z_last.reshape(B, self.vae_latent_dim, self.n_spatial_tokens)
        z_flat = z_flat.permute(0, 2, 1)  # (B, 9, C)

        # Project to width + add positional embedding
        tokens = self.proj(z_flat)  # (B, 9, width)
        tokens = tokens + self.spatial_pos_embed

        return tokens

    def encode_latent_flat(self, marker_offset: torch.Tensor) -> torch.Tensor:
        """
        Encode and return flat latent (for foresight GT computation).

        Args:
            marker_offset: (B, T_hist, 9, 9, 2)
        Returns:
            z_flat: (B, C*3*3) = (B, 144)
        """
        B = marker_offset.shape[0]
        with torch.no_grad():
            z_last, _ = self.vae.encode_single_frame(marker_offset)
        return z_last.reshape(B, -1)
