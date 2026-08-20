"""Board runtime for strict schema-v2 tactile-only latent checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from TFAC_V5.tac_quality_energy.tactile_only_latent import (
    TACTILE_ONLY_INPUT_MODE,
    TACTILE_ONLY_SCHEMA_VERSION,
    TactileOnlyLatentRuntime,
)


class BoardLatentEnergyRuntime(TactileOnlyLatentRuntime):
    """Canonical tactile-only runtime locked to the board task."""

    schema_version = TACTILE_ONLY_SCHEMA_VERSION
    input_mode = TACTILE_ONLY_INPUT_MODE

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cuda:0",
        *,
        expected_horizon: Optional[int] = None,
        expected_latent_shape: Optional[Sequence[int]] = None,
        expected_temporal_stride: Optional[int] = None,
        expected_future_offset: Optional[int] = None,
        expected_vae_identity: Optional[str] = None,
    ):
        super().__init__(
            checkpoint_path,
            device=device,
            expected_task="board",
            expected_horizon=expected_horizon,
            expected_latent_shape=expected_latent_shape,
            expected_temporal_stride=expected_temporal_stride,
            expected_future_offset=expected_future_offset,
            expected_vae_identity=expected_vae_identity,
        )


__all__ = ["BoardLatentEnergyRuntime"]
