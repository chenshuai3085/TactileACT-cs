"""Explicit loader for legacy action-aware board latent-energy checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

from .legacy_model import LegacyActionAwareBoardLatentEnergyScorer


class LegacyActionAwareBoardLatentEnergyRuntime(nn.Module):
    """Loads only old checkpoints that predate the schema-v2 contract."""

    def __init__(self, checkpoint_path: str | Path, device: str = "cuda:0"):
        super().__init__()
        self.device_name = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self.device = torch.device(self.device_name)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if ckpt.get("schema_version") == 2 or ckpt.get("input_mode") == "tactile_only_latent":
            raise ValueError("Schema-v2 tactile-only checkpoints must use BoardLatentEnergyRuntime.")
        self.model = LegacyActionAwareBoardLatentEnergyScorer(**ckpt["model_config"]).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        norm = ckpt["norm"]
        self.register_buffer("action_mean", torch.as_tensor(norm["action_mean"], dtype=torch.float32, device=self.device).view(1, 1, -1))
        self.register_buffer("action_std", torch.as_tensor(norm["action_std"], dtype=torch.float32, device=self.device).view(1, 1, -1))
        self.register_buffer("latent_mean", torch.as_tensor(norm["latent_mean"], dtype=torch.float32, device=self.device).view(1, 1, -1))
        self.register_buffer("latent_std", torch.as_tensor(norm["latent_std"], dtype=torch.float32, device=self.device).view(1, 1, -1))
        self.class_names = ckpt.get("class_names", ["expert", "pressure_too_small", "pressure_too_large", "pressure_unstable"])

    def forward(self, action_chunk: torch.Tensor, latent_chunk: torch.Tensor, normalized: bool = False) -> Dict[str, torch.Tensor]:
        action = action_chunk.to(self.device).float()
        latent = latent_chunk.to(self.device).float()
        if latent.dim() == 5:
            latent = latent.flatten(2)
        if not normalized:
            action = (action - self.action_mean) / self.action_std.clamp_min(1e-6)
            latent = (latent - self.latent_mean) / self.latent_std.clamp_min(1e-6)
        return self.model(action, latent)

    def score(
        self,
        action_chunk: torch.Tensor,
        latent_chunk: torch.Tensor,
        normalized: bool = False,
        mode: str = "score_good",
    ) -> torch.Tensor:
        out = self.forward(action_chunk, latent_chunk, normalized=normalized)
        aliases = {"p_expert": "prob"}
        if mode == "p_expert":
            return out[aliases[mode]][:, 0]
        if mode not in {"score_good", "energy", "expert_margin", "margin_energy", "quality_0_100"}:
            raise ValueError(f"Unknown score mode: {mode}")
        return out[mode]
