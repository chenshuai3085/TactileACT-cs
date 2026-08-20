"""Board tactile-only latent consequence energy scorer."""

from .model import BoardLatentEnergyScorer
from .runtime import BoardLatentEnergyRuntime

__all__ = [
    "BoardLatentEnergyScorer",
    "BoardLatentEnergyRuntime",
]
