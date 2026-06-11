"""Board chunk tactile consequence energy model."""

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


class BoardChunkEnergyScorer(nn.Module):
    """Prototype energy scorer for board wiping chunks.

    Inputs:
        action_chunk: ``(B, H, A)`` normalized joint action chunk.
        marker_chunk: ``(B, H, 9, 9, 2)`` normalized future/observed marker chunk.

    Outputs include class logits, expert-likeness score, and energy
    ``-score_good``.  Class 0 is always expert/straight.
    """

    def __init__(
        self,
        chunk_len: int = 16,
        action_dim: int = 7,
        marker_shape=(9, 9, 2),
        embed_dim: int = 128,
        hidden: int = 256,
        dropout: float = 0.10,
        temperature: float = 0.10,
        num_classes: int = 4,
    ):
        super().__init__()
        self.chunk_len = int(chunk_len)
        self.action_dim = int(action_dim)
        self.marker_shape = tuple(marker_shape)
        self.embed_dim = int(embed_dim)
        self.temperature = float(temperature)
        marker_flat_dim = self.chunk_len
        for dim in self.marker_shape:
            marker_flat_dim *= int(dim)
        action_flat_dim = self.chunk_len * self.action_dim

        half = max(32, embed_dim // 2)
        self.marker_encoder = TemporalMLPEncoder(marker_flat_dim, hidden, half, dropout)
        self.action_encoder = TemporalMLPEncoder(action_flat_dim, hidden, half, dropout)
        self.fusion = nn.Sequential(
            nn.Linear(2 * half, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
        )
        self.prototypes = nn.Parameter(torch.randn(num_classes, embed_dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def encode(self, action_chunk: torch.Tensor, marker_chunk: torch.Tensor) -> torch.Tensor:
        action_feat = self.action_encoder(action_chunk.float())
        marker_feat = self.marker_encoder(marker_chunk.float())
        z = self.fusion(torch.cat([action_feat, marker_feat], dim=-1))
        return F.normalize(z, dim=-1)

    def forward(self, action_chunk: torch.Tensor, marker_chunk: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.encode(action_chunk, marker_chunk)
        proto = F.normalize(self.prototypes, dim=-1)
        logits = z @ proto.T / max(self.temperature, 1e-6) + self.bias
        score_good = logits[:, 0]
        return {
            "embedding": z,
            "logits": logits,
            "score_good": score_good,
            "energy": -score_good,
            "prob": torch.softmax(logits, dim=-1),
        }
