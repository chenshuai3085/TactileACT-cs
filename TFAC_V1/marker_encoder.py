"""
Modular Marker Displacement Encoders.

Encode marker_offset (B, 9, 9, 2) -> (B, hidden_dim).

Supported encoders:
  - conv2d:   Conv2D on 9x9 grid, exploits spatial locality (OmniVTA style)
  - pointnet: PointNet with shared MLP + max pooling (Chen T-RO 2024 style)

Usage:
    encoder = build_marker_encoder("conv2d", hidden_dim=512)
    feat = encoder(marker_offset)  # (B, 9, 9, 2) -> (B, 512)
"""

import torch
import torch.nn as nn


class MarkerEncoderConv2D(nn.Module):
    """
    Conv2D encoder for marker_offset (B, 9, 9, 2) -> (B, hidden_dim).

    Exploits the regular 9x9 grid spatial locality.
    Spatial position is implicit in grid coordinates.

    Ref: OmniVTA (Conv-based > PointNet-AE > PCA on regular grids)
    """

    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),      # (B, 32, 9, 9)
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),     # (B, 64, 9, 9)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=3),     # (B, 128, 3, 3)
            nn.ReLU(inplace=True),
            nn.Conv2d(128, hidden_dim, kernel_size=3),       # (B, hidden_dim, 1, 1)
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 9, 9, 2) marker displacement grid
        Returns:
            feat: (B, hidden_dim)
        """
        x = x.permute(0, 3, 1, 2)      # (B, 2, 9, 9)
        x = self.conv(x)                # (B, hidden_dim, 1, 1)
        x = x.flatten(1)                # (B, hidden_dim)
        feat = self.fc(x)               # (B, hidden_dim)
        return feat


class MarkerEncoderPointNet(nn.Module):
    """
    PointNet encoder for marker_offset (B, 9, 9, 2) -> (B, hidden_dim).

    Treats each marker as a point. Adds normalized grid coordinates (i/8, j/8)
    as extra input channels to provide spatial information, since our data only
    contains offsets without original marker positions.

    Ref: Chen et al. T-RO 2024 (PointNet on marker displacement, k=32 latent)
         "raw marker positions contain spatial information crucial for
          inferring the contact status"

    Note: Our markers are a fixed 9x9 grid (no missing/variable-length),
    so PointNet's permutation invariance is not strictly needed. We include
    this encoder for ablation comparison with Conv2D.
    """

    def __init__(self, hidden_dim: int = 512, use_grid_coords: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_grid_coords = use_grid_coords

        in_dim = 4 if use_grid_coords else 2  # (offset_x, offset_y[, grid_i, grid_j])

        # Shared MLP per marker (Chen T-RO: 64, 64, 64, 128, 512)
        self.shared_mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Post-pooling MLP
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Pre-compute normalized grid coordinates
        grid_i, grid_j = torch.meshgrid(
            torch.linspace(0, 1, 9), torch.linspace(0, 1, 9), indexing='ij'
        )
        # (9, 9, 2)
        self.register_buffer('grid_coords', torch.stack([grid_i, grid_j], dim=-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 9, 9, 2) marker displacement grid
        Returns:
            feat: (B, hidden_dim)
        """
        B = x.shape[0]

        if self.use_grid_coords:
            # Concat grid coords: (B, 9, 9, 2) + (9, 9, 2) -> (B, 9, 9, 4)
            coords = self.grid_coords.unsqueeze(0).expand(B, -1, -1, -1)
            x = torch.cat([x, coords], dim=-1)   # (B, 9, 9, 4)

        x = x.reshape(B, 81, -1)                 # (B, 81, in_dim)
        x = self.shared_mlp(x)                    # (B, 81, hidden_dim)
        x = x.max(dim=1).values                   # (B, hidden_dim)  max pooling
        feat = self.head(x)                        # (B, hidden_dim)
        return feat


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_ENCODER_REGISTRY = {
    'conv2d': MarkerEncoderConv2D,
    'pointnet': MarkerEncoderPointNet,
}


def build_marker_encoder(encoder_type: str = 'conv2d',
                          hidden_dim: int = 512,
                          **kwargs) -> nn.Module:
    """
    Factory function to build a marker displacement encoder.

    Args:
        encoder_type: 'conv2d' or 'pointnet'
        hidden_dim: output feature dimension
        **kwargs: extra args forwarded to the encoder constructor
    Returns:
        encoder: nn.Module, (B, 9, 9, 2) -> (B, hidden_dim)
    """
    if encoder_type not in _ENCODER_REGISTRY:
        raise ValueError(
            f"Unknown marker encoder type '{encoder_type}'. "
            f"Choose from: {list(_ENCODER_REGISTRY.keys())}"
        )
    return _ENCODER_REGISTRY[encoder_type](hidden_dim=hidden_dim, **kwargs)


# ---------------------------------------------------------------------------
# LTD Encoder (independent of marker encoder type)
# ---------------------------------------------------------------------------

class LTDEncoder(nn.Module):
    """
    Latent Tactile Differential Encoder (OmniVTA style).
    concat(t_current, t_predicted, t_predicted - t_current) -> Linear -> (B, D)

    Explicitly exposes predicted change. Replaces competitive GatedFusion.
    """

    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t_current: torch.Tensor,
                t_predicted: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t_current:   (B, D) -- current tactile feature from MarkerEncoder
            t_predicted: (B, D) -- predicted future tactile feature
        Returns:
            f_tac: (B, D) -- fused tactile condition vector
        """
        diff = t_predicted - t_current
        concat = torch.cat([t_current, t_predicted, diff], dim=-1)  # (B, 3D)
        return self.proj(concat)  # (B, D)
