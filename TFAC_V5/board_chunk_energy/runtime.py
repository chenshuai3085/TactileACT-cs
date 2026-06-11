"""Checkpoint-backed runtime for board chunk energy guidance."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

from .model import BoardChunkEnergyScorer


class BoardChunkEnergyRuntime(nn.Module):
    def __init__(self, checkpoint_path: str | Path, device: str = "cuda:0"):
        super().__init__()
        self.device_name = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self.device = torch.device(self.device_name)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        config = ckpt["model_config"]
        self.model = BoardChunkEnergyScorer(**config).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        norm = ckpt["norm"]
        self.register_buffer("marker_mean", torch.as_tensor(norm["marker_mean"], dtype=torch.float32, device=self.device).view(1, 1, 1, 1, 2))
        self.register_buffer("marker_std", torch.as_tensor(norm["marker_std"], dtype=torch.float32, device=self.device).view(1, 1, 1, 1, 2))
        self.register_buffer("action_mean", torch.as_tensor(norm["action_mean"], dtype=torch.float32, device=self.device).view(1, 1, -1))
        self.register_buffer("action_std", torch.as_tensor(norm["action_std"], dtype=torch.float32, device=self.device).view(1, 1, -1))
        self.class_names = ckpt.get("class_names", ["expert", "pressure_too_small", "pressure_too_large", "pressure_unstable"])

    def normalize_inputs(self, action_chunk: torch.Tensor, marker_chunk: torch.Tensor):
        action = action_chunk.to(self.device).float()
        marker = marker_chunk.to(self.device).float()
        action = (action - self.action_mean) / self.action_std.clamp_min(1e-6)
        marker = (marker - self.marker_mean) / self.marker_std.clamp_min(1e-6)
        return action, marker

    def forward(self, action_chunk: torch.Tensor, marker_chunk: torch.Tensor, normalized: bool = False) -> Dict[str, torch.Tensor]:
        if normalized:
            action, marker = action_chunk.to(self.device).float(), marker_chunk.to(self.device).float()
        else:
            action, marker = self.normalize_inputs(action_chunk, marker_chunk)
        return self.model(action, marker)

    def score(self, action_chunk: torch.Tensor, marker_chunk: torch.Tensor, normalized: bool = False, mode: str = "score_good") -> torch.Tensor:
        out = self.forward(action_chunk, marker_chunk, normalized=normalized)
        if mode == "score_good":
            return out["score_good"]
        if mode == "energy":
            return out["energy"]
        if mode == "p_expert":
            return out["prob"][:, 0]
        raise ValueError(f"Unknown score mode: {mode}")
