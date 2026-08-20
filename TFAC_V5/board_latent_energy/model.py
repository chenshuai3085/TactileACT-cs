"""Board alias for the canonical tactile-only latent scorer."""

from TFAC_V5.tac_quality_energy.tactile_only_latent import TactileOnlyLatentScorer


BoardLatentEnergyScorer = TactileOnlyLatentScorer

__all__ = ["BoardLatentEnergyScorer"]
