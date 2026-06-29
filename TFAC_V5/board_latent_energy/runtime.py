"""Checkpoint-backed runtime for board latent energy guidance."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

from .model import BoardLatentEnergyScorer


class BoardLatentEnergyRuntime(nn.Module):
    def __init__(self, checkpoint_path: str | Path, device: str = "cuda:0"):
        super().__init__()
        self.device_name = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self.device = torch.device(self.device_name)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model = BoardLatentEnergyScorer(**ckpt["model_config"]).to(self.device)
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
        self.vae_meta = ckpt.get("vae_meta", {})
        self.vae_checkpoint = ckpt.get("vae_checkpoint")
        self.data_config = ckpt.get("data_config", ckpt.get("args", {}))
        self.temporal_stride = int(self.data_config.get("temporal_stride", 1))

    def normalize_inputs(self, action_chunk: torch.Tensor, latent_chunk: torch.Tensor):
        action = action_chunk.to(self.device).float()
        latent = latent_chunk.to(self.device).float()
        if latent.dim() == 5:
            latent = latent.flatten(2)
        action = (action - self.action_mean) / self.action_std.clamp_min(1e-6)
        latent = (latent - self.latent_mean) / self.latent_std.clamp_min(1e-6)
        return action, latent

    def forward(self, action_chunk: torch.Tensor, latent_chunk: torch.Tensor, normalized: bool = False) -> Dict[str, torch.Tensor]:
        if normalized:
            action = action_chunk.to(self.device).float()
            latent = latent_chunk.to(self.device).float()
            if latent.dim() == 5:
                latent = latent.flatten(2)
        else:
            action, latent = self.normalize_inputs(action_chunk, latent_chunk)
        return self.model(action, latent)

    def score(self, action_chunk: torch.Tensor, latent_chunk: torch.Tensor, normalized: bool = False, mode: str = "score_good") -> torch.Tensor:
        out = self.forward(action_chunk, latent_chunk, normalized=normalized)
        if mode == "score_good":
            return out["score_good"]
        if mode == "energy":
            return out["energy"]
        if mode == "expert_margin":
            return out["expert_margin"]
        if mode == "margin_energy":
            return out["margin_energy"]
        if mode == "quality_0_100":
            return out["quality_0_100"]
        if mode == "p_expert":
            return out["prob"][:, 0]
        raise ValueError(f"Unknown score mode: {mode}")
