#!/usr/bin/env python3
"""Run the multi-step action-conditioned foresight trainer with TactileVAE V2."""

import argparse
import json
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from TFAC_V5 import pretrain_latent_foresight_multistep as trainer
from TFAC_V5.tactile_vae_v2 import TactileVAEv2


class V2MultiStepLatentForesightModel(trainer.MultiStepLatentForesightModel):
    def __init__(self, *args, tactile_vae_ckpt=None, tactile_vae_latent_dim=16,
                 tactile_vae_window=8, **kwargs):
        super().__init__(
            *args,
            tactile_vae_ckpt=None,
            tactile_vae_latent_dim=tactile_vae_latent_dim,
            tactile_vae_window=tactile_vae_window,
            **kwargs,
        )
        self.tactile_vae = TactileVAEv2(
            latent_dim=tactile_vae_latent_dim,
            temporal_window=int(tactile_vae_window),
        )
        if not tactile_vae_ckpt or not os.path.isfile(tactile_vae_ckpt):
            raise FileNotFoundError(tactile_vae_ckpt)
        checkpoint = torch.load(tactile_vae_ckpt, map_location="cpu", weights_only=False)
        self.tactile_vae.load_state_dict(checkpoint["model_state_dict"])
        self.tactile_vae.requires_grad_(False)
        print(f"Loaded TactileVAE V2 from: {tactile_vae_ckpt}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("tactile_vae_version") != "v2":
        raise ValueError("V2 runner requires tactile_vae_version='v2'")
    trainer.MultiStepLatentForesightModel = V2MultiStepLatentForesightModel
    trainer.main(config)


if __name__ == "__main__":
    main()
