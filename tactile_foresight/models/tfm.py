"""
Tactile Foresight Model (TFM)

Predicts future tactile embeddings from current visual (+ optional current
tactile) observations directly in DINOv2 768-dim feature space.

Pipeline:
    1. Precomputed DINOv2 features (768-dim) as input
    2. Trainable Transformer prediction head → predicted future tactile (768-dim)

Training objective: MSE loss in DINOv2 feature space:
    L = || T̂_{t+h} - DINOv2(T_{t+h}) ||^2

No alignment stage needed — operates directly in DINOv2 space.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TFMPredictionHead(nn.Module):
    """Transformer-based prediction head for future tactile embedding.

    Operates directly in DINOv2 768-dim space.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        hidden_dim: int = 512,
        num_layers: int = 4,
        nheads: int = 8,
        num_horizons: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # Project DINOv2 features to hidden_dim
        self.vision_proj = nn.Linear(embed_dim, hidden_dim)
        self.tactile_proj = nn.Linear(embed_dim, hidden_dim)

        # Learnable mask token (used when current tactile is masked)
        self.mask_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # Learnable horizon embeddings
        self.horizon_embed = nn.Embedding(num_horizons, hidden_dim)

        # Learnable query token for predicting future tactile
        self.query_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.decoder_norm = nn.LayerNorm(hidden_dim)

        # Output projection back to DINOv2 space
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        vision_feat: torch.Tensor,
        tactile_feat: Optional[torch.Tensor],
        horizon_idx: torch.Tensor,
        tac_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            vision_feat: (B, 768) DINOv2 vision feature
            tactile_feat: (B, 768) DINOv2 tactile feature, or None
            horizon_idx: (B,) long tensor, index into horizon_embed
            tac_mask: (B,) bool tensor, True = mask current tactile

        Returns:
            (B, 768) predicted future tactile embedding in DINOv2 space
        """
        B = vision_feat.shape[0]

        v = self.vision_proj(vision_feat).unsqueeze(1)  # (B, 1, hidden)

        if tactile_feat is not None and tac_mask is not None:
            t_proj = self.tactile_proj(tactile_feat).unsqueeze(1)
            mask_expanded = tac_mask.unsqueeze(1).unsqueeze(2)
            t = torch.where(mask_expanded, self.mask_token.expand(B, -1, -1), t_proj)
        elif tactile_feat is not None:
            t = self.tactile_proj(tactile_feat).unsqueeze(1)
        else:
            t = self.mask_token.expand(B, -1, -1)

        h = self.horizon_embed(horizon_idx).unsqueeze(1)

        memory = torch.cat([v, t, h], dim=1)  # (B, 3, hidden)
        query = self.query_token.expand(B, -1, -1)  # (B, 1, hidden)

        out = self.decoder(query, memory)
        out = self.decoder_norm(out)
        pred = self.output_proj(out.squeeze(1))  # (B, 768)
        return pred


class TactileForesightModel(nn.Module):
    """TFM: predicts future tactile in DINOv2 768-dim space.

    No alignment stage needed. Directly uses precomputed DINOv2 features.

    Usage:
        tfm = TactileForesightModel(horizons=[4, 8, 12], device="cuda")

        # Training (with precomputed features)
        loss = tfm.compute_loss(v_feat, t_cur_feat, t_fut_feat, horizon, tac_mask)

        # Inference
        pred = tfm.predict(v_feat, t_cur_feat, horizon, tac_mask)
    """

    def __init__(
        self,
        dino_dim: int = 768,
        horizons: list = None,
        hidden_dim: int = 512,
        num_layers: int = 4,
        nheads: int = 8,
        dropout: float = 0.1,
        device: str = "cpu",
    ):
        super().__init__()
        if horizons is None:
            horizons = [4, 8, 12]
        self.horizons = horizons
        self.horizon_to_idx = {h: i for i, h in enumerate(horizons)}
        self.dino_dim = dino_dim

        # Trainable prediction head (in DINOv2 space)
        self.pred_head = TFMPredictionHead(
            embed_dim=dino_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            nheads=nheads,
            num_horizons=len(horizons),
            dropout=dropout,
        )

        self.to(device)

    def _horizon_to_idx(self, horizons: torch.Tensor) -> torch.Tensor:
        idx = torch.zeros_like(horizons)
        for h_val, h_idx in self.horizon_to_idx.items():
            idx[horizons == h_val] = h_idx
        return idx

    def predict(
        self,
        vision_feat: torch.Tensor,
        tactile_feat: Optional[torch.Tensor],
        horizon: torch.Tensor,
        tac_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict future tactile embedding in DINOv2 space.

        Args:
            vision_feat: (B, 768) precomputed DINOv2 vision feature
            tactile_feat: (B, 768) precomputed DINOv2 tactile feature, or None
            horizon: (B,) horizon values (e.g., 4, 8, 12)
            tac_mask: (B,) bool, True = mask current tactile

        Returns: (B, 768)
        """
        h_idx = self._horizon_to_idx(horizon)
        return self.pred_head(vision_feat, tactile_feat, h_idx, tac_mask)

    def compute_loss(
        self,
        vision_feat: torch.Tensor,
        tactile_current_feat: torch.Tensor,
        tactile_future_feat: torch.Tensor,
        horizon: torch.Tensor,
        tac_mask: torch.Tensor,
    ) -> dict:
        """MSE loss in DINOv2 feature space.

        Args:
            vision_feat: (B, 768) precomputed DINOv2 vision feature
            tactile_current_feat: (B, 768) current tactile DINOv2 feature
            tactile_future_feat: (B, 768) future tactile DINOv2 feature (target)
            horizon: (B,) horizon values
            tac_mask: (B,) bool, True = mask current tactile

        Returns: dict with 'loss', 'mse_loss', 'cosine_sim'
        """
        t_pred = self.predict(vision_feat, tactile_current_feat, horizon, tac_mask)

        mse_loss = F.mse_loss(t_pred, tactile_future_feat)

        with torch.no_grad():
            cos_sim = F.cosine_similarity(t_pred, tactile_future_feat, dim=-1).mean()

        return {
            "loss": mse_loss,
            "mse_loss": mse_loss,
            "cosine_sim": cos_sim,
        }

    def trainable_parameters(self):
        return self.pred_head.parameters()

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.pred_head.parameters() if p.requires_grad)

    def save_model(self, path: str):
        """Save prediction head state dict."""
        torch.save({
            "pred_head": self.pred_head.state_dict(),
            "dino_dim": self.dino_dim,
            "horizons": self.horizons,
        }, path)

    def load_model(self, path: str):
        """Load prediction head state dict."""
        ckpt = torch.load(path, map_location="cpu")
        self.pred_head.load_state_dict(ckpt["pred_head"])


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Building TFM on {device}...")

    tfm = TactileForesightModel(horizons=[4, 8, 12], device=device)
    print(f"Trainable parameters: {tfm.num_trainable_params():,}")

    B = 4
    v_feat = torch.randn(B, 768, device=device)
    t_cur = torch.randn(B, 768, device=device)
    t_fut = torch.randn(B, 768, device=device)
    horizon = torch.tensor([4, 8, 12, 4], device=device)
    mask = torch.tensor([True, False, True, False], device=device)

    losses = tfm.compute_loss(v_feat, t_cur, t_fut, horizon, mask)
    print(f"MSE loss: {losses['mse_loss'].item():.4f}")
    print(f"Cosine sim: {losses['cosine_sim'].item():.4f}")

    pred = tfm.predict(v_feat, t_cur, horizon, mask)
    print(f"Predicted shape: {pred.shape}")  # (4, 768)
    print("TFM OK!")
