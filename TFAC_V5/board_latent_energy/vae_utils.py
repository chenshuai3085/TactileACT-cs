"""Utilities for frozen TactileVAE latent extraction."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Mapping, Tuple

import numpy as np
import torch

from TFAC_V5.tactile_vae import TactileVAE
from TFAC_V5.tactile_vae_v2 import TactileVAEv2


DEFAULT_TACTILE_VAE_CKPT = (
    "/home/chenshuai/Project/output/"
    "tactile_vae_board_260609_260610_left_tw8_ld16_s2_e150/best_tactile_vae.pt"
)


def vae_checkpoint_identity(path: str | Path) -> str:
    """Return the immutable identity stored in schema-v2 scorer artifacts."""

    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"TactileVAE checkpoint does not exist: {checkpoint}")
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def load_tactile_vae_checkpoint(path: str | Path, device: torch.device | str):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    latent_dim = int(config.get("latent_dim", 16))
    temporal_window = int(config.get("temporal_window", 8))
    model_version = str(config.get("model_version", "v1")).lower()
    if model_version == "v2":
        model = TactileVAEv2(
            latent_dim=latent_dim,
            temporal_window=temporal_window,
            decoder_hidden=int(config.get("decoder_hidden", 128)),
            decoder_heads=int(config.get("decoder_heads", 4)),
            decoder_layers=int(config.get("decoder_layers", 2)),
            kl_weight=float(config.get("kl_weight", 1e-6)),
            direction_weight=float(config.get("direction_weight", 0.2)),
            intensity_weight=float(config.get("intensity_weight", 0.1)),
            rank_weight=float(config.get("rank_weight", 0.05)),
        ).to(device)
    elif model_version == "v1":
        model = TactileVAE(
            latent_dim=latent_dim,
            temporal_window=temporal_window,
            num_freqs=int(config.get("num_freqs", 4)),
            inr_hidden=int(config.get("inr_hidden", 64)),
            kl_weight=float(config.get("kl_weight", 1e-6)),
            direction_weight=float(config.get("direction_weight", 0.2)),
        ).to(device)
    else:
        raise ValueError(f"Unsupported TactileVAE model_version: {model_version}")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    norm_stats = ckpt.get("norm_stats", {})
    mean = np.asarray(norm_stats.get("mean", [0.0, 0.0]), dtype=np.float32)
    std = np.asarray(norm_stats.get("std", [1.0, 1.0]), dtype=np.float32)
    return model, {
        "config": config,
        "model_version": model_version,
        "latent_dim": latent_dim,
        "temporal_window": temporal_window,
        "latent_flat_dim": latent_dim * 3 * 3,
        "mean": mean,
        "std": std,
    }


def encode_marker_chunk_to_latents(
    marker: np.ndarray,
    start: int,
    chunk_len: int,
    vae: TactileVAE | TactileVAEv2,
    vae_info: Mapping[str, object],
    device: torch.device,
    temporal_stride: int = 1,
) -> np.ndarray:
    """Encode a future marker chunk into one latent per future step.

    For step ``s`` in ``start + arange(chunk_len) * temporal_stride``, the
    latent is produced from the old TactileVAE temporal window
    ``marker[s-T+1:s+1]``.  The first frames are left-padded by repeating frame
    0 when needed.
    """

    temporal_stride = int(temporal_stride)
    if temporal_stride < 1:
        raise ValueError(f"temporal_stride must be >= 1, got {temporal_stride}")
    temporal_window = int(vae_info["temporal_window"])
    mean = np.asarray(vae_info["mean"], dtype=np.float32).reshape(1, 1, 1, 2)
    std = np.asarray(vae_info["std"], dtype=np.float32).reshape(1, 1, 1, 2)
    latent_flat_dim = int(vae_info["latent_flat_dim"])
    windows = []
    for offset in range(chunk_len):
        target_t = start + offset * temporal_stride
        end = target_t + 1
        begin = end - temporal_window
        if begin < 0:
            pad = np.repeat(marker[0:1], -begin, axis=0)
            window = np.concatenate([pad, marker[0:end]], axis=0)
        else:
            window = marker[begin:end]
        if window.shape[0] != temporal_window:
            raise ValueError(
                "Bad VAE window length "
                f"{window.shape[0]} at start={start}, offset={offset}, "
                f"temporal_stride={temporal_stride}"
            )
        window = (window.astype(np.float32) - mean) / np.maximum(std, 1e-6)
        windows.append(window)

    batch = torch.from_numpy(np.stack(windows, axis=0)).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        z, _ = vae.encode_single_frame(batch)
    return z.reshape(chunk_len, latent_flat_dim).detach().cpu().numpy().astype(np.float32)


def normalize_latent(latent: np.ndarray, norm: Mapping[str, np.ndarray]) -> np.ndarray:
    mean = np.asarray(norm["latent_mean"], dtype=np.float32).reshape(1, -1)
    std = np.asarray(norm["latent_std"], dtype=np.float32).reshape(1, -1)
    return ((latent.astype(np.float32) - mean) / np.maximum(std, 1e-6)).astype(np.float32)


def infer_vae_meta(vae_info: Mapping[str, object]) -> Dict[str, object]:
    return {
        "model_version": str(vae_info.get("model_version", "v1")),
        "latent_dim": int(vae_info["latent_dim"]),
        "temporal_window": int(vae_info["temporal_window"]),
        "latent_flat_dim": int(vae_info["latent_flat_dim"]),
        "norm_mean": np.asarray(vae_info["mean"], dtype=np.float32).tolist(),
        "norm_std": np.asarray(vae_info["std"], dtype=np.float32).tolist(),
    }
