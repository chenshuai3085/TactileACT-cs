"""
V4 Spatial Marker Encoder — Core Innovation C1.

Encodes marker_offset (B, 9, 9, 2) into 9 spatial tokens (9, B, D) instead of
V3's single global token (1, B, D). Preserves spatial structure for:
  - Patch-level foresight prediction (latent space)
  - Spatial contrastive learning
  - Contact-aware fusion

Architecture:
  (B, 9, 9, 2) → per-point MLP → +learnable_pos_embed
  → 2-layer SelfAttn (81 tokens interact)
  → 3×3 patch pooling → 9 patch tokens (9, B, D)

Also retains V3's MarkerEncoderConv2D, MarkerEncoderPointNet, LTDEncoder for
backward compatibility and ablation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SpatialMarkerEncoder(nn.Module):
    """
    Spatial Token Marker Encoder (V4 C1).

    (B, 9, 9, 2) → 9 spatial tokens (9, B, D), preserving 3×3 patch structure.

    Pipeline:
      1. Per-point MLP: (B, 81, 4) → (B, 81, D)  [offset + grid coords]
      2. + learnable positional embedding (81, D)
      3. 2-layer self-attention among 81 tokens
      4. 3×3 patch pooling: 81 tokens → 9 patch tokens (mean pool within each 3×3 patch)
    """

    def __init__(self, hidden_dim: int = 512, num_self_attn_layers: int = 2,
                 nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        in_dim = 4  # offset_x, offset_y, grid_i, grid_j

        # Per-point MLP: (B, 81, 4) → (B, 81, D)
        self.point_mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, hidden_dim),
        )

        # Learnable positional embedding for 81 marker points
        self.pos_embed = nn.Parameter(torch.randn(81, hidden_dim) * 0.02)

        # Self-attention layers for inter-point interaction
        self.self_attn_layers = nn.ModuleList()
        self.self_attn_norms = nn.ModuleList()
        for _ in range(num_self_attn_layers):
            self.self_attn_layers.append(
                nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout, batch_first=False)
            )
            self.self_attn_norms.append(nn.LayerNorm(hidden_dim))

        # Pre-compute normalized grid coordinates (register as buffer)
        grid_i, grid_j = torch.meshgrid(
            torch.linspace(0, 1, 9), torch.linspace(0, 1, 9), indexing='ij'
        )
        # (81, 2)
        self.register_buffer('grid_coords', torch.stack([grid_i, grid_j], dim=-1).reshape(81, 2))

        # Patch index mapping: 81 points → 9 patches (3×3 each)
        # patch_ids[i] = which patch point i belongs to
        patch_ids = []
        for row in range(9):
            for col in range(9):
                patch_row = row // 3
                patch_col = col // 3
                patch_ids.append(patch_row * 3 + patch_col)
        self.register_buffer('patch_ids', torch.tensor(patch_ids, dtype=torch.long))  # (81,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode marker_offset to 9 spatial tokens.

        Args:
            x: (B, 9, 9, 2) marker displacement grid
        Returns:
            tokens: (9, B, D) — 9 spatial patch tokens
        """
        B = x.shape[0]

        # Flatten grid: (B, 9, 9, 2) → (B, 81, 2)
        x_flat = x.reshape(B, 81, 2)

        # Concat grid coordinates: (B, 81, 2) + (81, 2) → (B, 81, 4)
        coords = self.grid_coords.unsqueeze(0).expand(B, -1, -1)  # (B, 81, 2)
        x_cat = torch.cat([x_flat, coords], dim=-1)  # (B, 81, 4)

        # Per-point MLP
        tokens = self.point_mlp(x_cat)  # (B, 81, D)

        # Add positional embedding
        tokens = tokens + self.pos_embed.unsqueeze(0)  # (B, 81, D)

        # Self-attention: (81, B, D) format for nn.MultiheadAttention
        tokens = tokens.permute(1, 0, 2)  # (81, B, D)
        for attn, norm in zip(self.self_attn_layers, self.self_attn_norms):
            residual = tokens
            tokens2 = attn(tokens, tokens, tokens)[0]
            tokens = norm(residual + tokens2)

        # 3×3 patch pooling: 81 tokens → 9 patch tokens
        # tokens: (81, B, D)
        tokens = tokens.permute(1, 0, 2)  # (B, 81, D)
        patch_tokens = self._patch_pool(tokens)  # (B, 9, D)
        patch_tokens = patch_tokens.permute(1, 0, 2)  # (9, B, D)

        return patch_tokens

    def _patch_pool(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Mean pool 81 tokens into 9 patches (each 3×3 region).

        Args:
            tokens: (B, 81, D)
        Returns:
            patch_tokens: (B, 9, D)
        """
        B, _, D = tokens.shape
        patch_tokens = torch.zeros(B, 9, D, device=tokens.device, dtype=tokens.dtype)
        for pid in range(9):
            mask = (self.patch_ids == pid)  # (81,) bool
            patch_tokens[:, pid] = tokens[:, mask].mean(dim=1)
        return patch_tokens

    def forward_global(self, x: torch.Tensor) -> torch.Tensor:
        """
        Global pooled encoding for dynamics model / contrastive / compatibility.

        Args:
            x: (B, 9, 9, 2)
        Returns:
            feat: (B, D)
        """
        patch_tokens = self.forward(x)  # (9, B, D)
        return patch_tokens.mean(dim=0)  # (B, D)


# ---------------------------------------------------------------------------
# V3-compatible encoders (retained for ablation)
# ---------------------------------------------------------------------------

class MarkerEncoderConv2D(nn.Module):
    """Conv2D encoder: (B, 9, 9, 2) → (B, hidden_dim). From V3."""

    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, hidden_dim, kernel_size=3),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.flatten(1)
        return self.fc(x)


class MarkerEncoderPointNet(nn.Module):
    """PointNet encoder: (B, 9, 9, 2) → (B, hidden_dim). From V3."""

    def __init__(self, hidden_dim: int = 512, use_grid_coords: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_grid_coords = use_grid_coords
        in_dim = 4 if use_grid_coords else 2
        self.shared_mlp = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 128), nn.ReLU(inplace=True),
            nn.Linear(128, hidden_dim), nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim))
        grid_i, grid_j = torch.meshgrid(
            torch.linspace(0, 1, 9), torch.linspace(0, 1, 9), indexing='ij')
        self.register_buffer('grid_coords', torch.stack([grid_i, grid_j], dim=-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        if self.use_grid_coords:
            coords = self.grid_coords.unsqueeze(0).expand(B, -1, -1, -1)
            x = torch.cat([x, coords], dim=-1)
        x = x.reshape(B, 81, -1)
        x = self.shared_mlp(x)
        x = x.max(dim=1).values
        return self.head(x)


# ---------------------------------------------------------------------------
# LTD Encoder (from V3, unchanged)
# ---------------------------------------------------------------------------

class LTDEncoder(nn.Module):
    """Latent Tactile Differential Encoder: concat(current, predicted, diff) → (B, D)."""

    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t_current: torch.Tensor, t_predicted: torch.Tensor) -> torch.Tensor:
        diff = t_predicted - t_current
        concat = torch.cat([t_current, t_predicted, diff], dim=-1)
        return self.proj(concat)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_ENCODER_REGISTRY = {
    'spatial': SpatialMarkerEncoder,
    'conv2d': MarkerEncoderConv2D,
    'pointnet': MarkerEncoderPointNet,
}


def build_marker_encoder(encoder_type: str = 'spatial',
                          hidden_dim: int = 512,
                          **kwargs) -> nn.Module:
    """
    Factory function for marker encoders.

    Args:
        encoder_type: 'spatial' (V4 default), 'conv2d', or 'pointnet'
        hidden_dim: output feature dimension
    Returns:
        encoder: nn.Module
            - 'spatial': (B, 9, 9, 2) → (9, B, D) [call forward()]
                         or (B, D) [call forward_global()]
            - 'conv2d'/'pointnet': (B, 9, 9, 2) → (B, D)
    """
    if encoder_type not in _ENCODER_REGISTRY:
        raise ValueError(
            f"Unknown marker encoder type '{encoder_type}'. "
            f"Choose from: {list(_ENCODER_REGISTRY.keys())}"
        )
    return _ENCODER_REGISTRY[encoder_type](hidden_dim=hidden_dim, **kwargs)
