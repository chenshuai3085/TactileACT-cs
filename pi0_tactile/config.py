"""Configuration for Pi0-TacForesight model."""
from __future__ import annotations

import dataclasses
from typing import Optional


@dataclasses.dataclass
class Pi0TactileConfig:
    # === Pi0 base model ===
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    action_dim: int = 7
    action_horizon: int = 20
    max_token_len: int = 48
    pi05: bool = False
    dtype: str = "bfloat16"

    # === Vision ===
    camera_names: list = dataclasses.field(
        default_factory=lambda: ["global", "wrist"]
    )
    image_size: tuple = (224, 224)

    # === Tactile (TactileVAE) ===
    vae_checkpoint: str = ""
    vae_latent_dim: int = 16
    tac_history: int = 8
    tac_spatial_size: int = 3  # 3x3 spatial latent map
    tac_token_num: int = 9    # 3x3 = 9 spatial tokens
    freeze_vae: bool = True

    # === Foresight ===
    foresight_checkpoint: str = ""
    foresight_hidden_dim: int = 512
    foresight_layers: int = 3
    foresight_nheads: int = 8
    foresight_dim_feedforward: int = 2048
    foresight_horizon: int = 10
    lambda_foresight: float = 0.1
    foresight_t_threshold: float = 0.3
    foresight_warmup_steps: int = 1000
    foresight_out_dim: int = 144  # vae_latent_dim * 3 * 3

    # === Training ===
    freeze_paligemma: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    num_flow_steps: int = 10  # inference ODE steps
    pytorch_compile_mode: Optional[str] = None  # disable compile for debug

    # === Data ===
    dataset_dir: str = ""
    chunk_size: int = 20
    obs_horizon: int = 2
    fixed_prompt: str = "grasp the object with tactile feedback"

    # === Normalization ===
    marker_mean: list = dataclasses.field(
        default_factory=lambda: [0.572, -1.786]
    )
    marker_std: list = dataclasses.field(
        default_factory=lambda: [1.596, 3.845]
    )

    @property
    def tactile_latent_flat_dim(self) -> int:
        return self.vae_latent_dim * self.tac_spatial_size * self.tac_spatial_size
