"""Board chunk latent tactile consequence energy model."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalMLPEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.flatten(1))


class BoardLatentEnergyScorer(nn.Module):
    """Prototype energy scorer over action chunks and tactile latent chunks.

    Inputs:
        action_chunk: ``(B, H, A)`` normalized joint action chunk.
        latent_chunk: ``(B, H, Z)`` normalized old-TactileVAE latent chunk.
    """

    def __init__(
        self,
        chunk_len: int = 16,
        action_dim: int = 7,
        latent_dim: int = 144,
        embed_dim: int = 128,
        hidden: int = 256,
        dropout: float = 0.10,
        temperature: float = 0.10,
        num_classes: int = 4,
    ):
        super().__init__()
        self.chunk_len = int(chunk_len)
        self.action_dim = int(action_dim)
        self.latent_dim = int(latent_dim)
        self.embed_dim = int(embed_dim)
        self.temperature = float(temperature)
        half = max(32, embed_dim // 2)
        self.action_encoder = TemporalMLPEncoder(self.chunk_len * self.action_dim, hidden, half, dropout)
        self.latent_encoder = TemporalMLPEncoder(self.chunk_len * self.latent_dim, hidden, half, dropout)
        self.fusion = nn.Sequential(
            nn.Linear(2 * half, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
        )
        self.prototypes = nn.Parameter(torch.randn(num_classes, embed_dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def encode(self, action_chunk: torch.Tensor, latent_chunk: torch.Tensor) -> torch.Tensor:
        action_feat = self.action_encoder(action_chunk.float())
        latent_feat = self.latent_encoder(latent_chunk.float())
        z = self.fusion(torch.cat([action_feat, latent_feat], dim=-1))
        return F.normalize(z, dim=-1)

    def forward(self, action_chunk: torch.Tensor, latent_chunk: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.encode(action_chunk, latent_chunk)
        proto = F.normalize(self.prototypes, dim=-1)
        logits = z @ proto.T / max(self.temperature, 1e-6) + self.bias
        score_good = logits[:, 0]
        negative_logsumexp = torch.logsumexp(logits[:, 1:], dim=-1)
        expert_margin = score_good - negative_logsumexp
        quality_0_100 = torch.sigmoid(expert_margin) * 100.0
        return {
            "embedding": z,
            "logits": logits,
            "score_good": score_good,
            "energy": -score_good,
            "expert_margin": expert_margin,
            "margin_energy": -expert_margin,
            "quality_0_100": quality_0_100,
            "prob": torch.softmax(logits, dim=-1),
        }
