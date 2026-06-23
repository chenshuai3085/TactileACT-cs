"""Latent chunk dataset for board tactile consequence energy training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from TFAC_V5.board_chunk_energy.dataset import (
    DEFAULT_ACTION_KEY,
    DEFAULT_MARKER_KEY,
    ChunkIndexRow,
    ChunkDatasetConfig,
    build_default_index,
    rows_from_jsonable,
    rows_to_jsonable,
    split_rows_by_episode,
)
from TFAC_V5.board_latent_energy.vae_utils import (
    encode_marker_chunk_to_latents,
    normalize_action,
    normalize_latent,
)


def compute_action_normalization(
    rows: Sequence[ChunkIndexRow],
    action_key: str = DEFAULT_ACTION_KEY,
    max_windows: int = 4096,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    if not rows:
        raise ValueError("Cannot compute normalization from empty rows")
    rng = np.random.default_rng(seed)
    sample_rows = list(rows)
    if len(sample_rows) > max_windows:
        idx = rng.choice(len(sample_rows), max_windows, replace=False)
        sample_rows = [sample_rows[i] for i in sorted(idx.tolist())]

    action_sum = None
    action_sumsq = None
    action_count = 0
    for row in sample_rows:
        with h5py.File(row.path, "r") as f:
            action = f[action_key][row.start:row.start + row.length].astype(np.float64)
        flat = action.reshape(-1, action.shape[-1])
        if action_sum is None:
            action_sum = np.zeros((flat.shape[-1],), dtype=np.float64)
            action_sumsq = np.zeros((flat.shape[-1],), dtype=np.float64)
        action_sum += flat.sum(axis=0)
        action_sumsq += np.square(flat).sum(axis=0)
        action_count += flat.shape[0]

    assert action_sum is not None and action_sumsq is not None
    mean = action_sum / max(1, action_count)
    var = action_sumsq / max(1, action_count) - np.square(mean)
    return {
        "action_mean": mean.astype(np.float32),
        "action_std": np.sqrt(np.maximum(var, 1e-8)).astype(np.float32),
    }


def compute_latent_normalization(
    rows: Sequence[ChunkIndexRow],
    vae,
    vae_info: Mapping[str, object],
    device: torch.device,
    marker_key: str = DEFAULT_MARKER_KEY,
    max_windows: int = 4096,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    if not rows:
        raise ValueError("Cannot compute latent normalization from empty rows")
    rng = np.random.default_rng(seed)
    sample_rows = list(rows)
    if len(sample_rows) > max_windows:
        idx = rng.choice(len(sample_rows), max_windows, replace=False)
        sample_rows = [sample_rows[i] for i in sorted(idx.tolist())]

    latent_dim = int(vae_info["latent_flat_dim"])
    latent_sum = np.zeros((latent_dim,), dtype=np.float64)
    latent_sumsq = np.zeros((latent_dim,), dtype=np.float64)
    latent_count = 0
    by_path: Dict[str, np.ndarray] = {}
    for row in tqdm(sample_rows, desc="latent norm", leave=False):
        if row.path not in by_path:
            with h5py.File(row.path, "r") as f:
                by_path[row.path] = f[marker_key][:].astype(np.float32)
        latent = encode_marker_chunk_to_latents(by_path[row.path], row.start, row.length, vae, vae_info, device)
        flat = latent.reshape(-1, latent.shape[-1]).astype(np.float64)
        latent_sum += flat.sum(axis=0)
        latent_sumsq += np.square(flat).sum(axis=0)
        latent_count += flat.shape[0]
    mean = latent_sum / max(1, latent_count)
    var = latent_sumsq / max(1, latent_count) - np.square(mean)
    return {
        "latent_mean": mean.astype(np.float32),
        "latent_std": np.sqrt(np.maximum(var, 1e-8)).astype(np.float32),
    }


def compute_normalization(
    rows: Sequence[ChunkIndexRow],
    vae,
    vae_info: Mapping[str, object],
    device: torch.device,
    max_windows: int = 4096,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    norm = compute_action_normalization(rows, max_windows=max_windows, seed=seed)
    norm.update(compute_latent_normalization(rows, vae, vae_info, device, max_windows=max_windows, seed=seed))
    return norm


class BoardLatentChunkDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[ChunkIndexRow],
        norm: Mapping[str, np.ndarray],
        vae,
        vae_info: Mapping[str, object],
        device: torch.device,
        marker_key: str = DEFAULT_MARKER_KEY,
        action_key: str = DEFAULT_ACTION_KEY,
        include_path: bool = False,
        preload: bool = True,
    ):
        self.rows = list(rows)
        self.norm = norm
        self.vae = vae
        self.vae_info = vae_info
        self.device = device
        self.marker_key = marker_key
        self.action_key = action_key
        self.include_path = include_path
        self.preload = preload
        self._cache = None
        if preload:
            self._cache = []
            by_path: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
            for row in tqdm(self.rows, desc="preload latent", leave=False):
                if row.path not in by_path:
                    with h5py.File(row.path, "r") as f:
                        marker = f[self.marker_key][:].astype(np.float32)
                        action = f[self.action_key][:].astype(np.float32)
                    by_path[row.path] = (marker, action)
                marker, action = by_path[row.path]
                self._cache.append(self._read_and_normalize(row, marker=marker, action=action))

    def __len__(self) -> int:
        return len(self.rows)

    def _read_and_normalize(
        self,
        row: ChunkIndexRow,
        marker: np.ndarray | None = None,
        action: np.ndarray | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if marker is None or action is None:
            with h5py.File(row.path, "r") as f:
                marker = f[self.marker_key][:].astype(np.float32)
                action = f[self.action_key][:].astype(np.float32)
        action_chunk = action[row.start:row.start + row.length].astype(np.float32)
        latent_chunk = encode_marker_chunk_to_latents(marker, row.start, row.length, self.vae, self.vae_info, self.device)
        return (
            normalize_latent(latent_chunk, self.norm),
            normalize_action(action_chunk, self.norm),
        )

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.rows[index]
        if self._cache is None:
            latent, action = self._read_and_normalize(row)
        else:
            latent, action = self._cache[index]
        item: Dict[str, object] = {
            "latent": torch.from_numpy(latent),
            "action": torch.from_numpy(action),
            "label": torch.tensor(row.label, dtype=torch.long),
        }
        if self.include_path:
            item.update({"path": row.path, "start": row.start, "class_name": row.class_name})
        return item


def write_manifest(
    path: Path,
    *,
    rows: Sequence[ChunkIndexRow],
    train_rows: Sequence[ChunkIndexRow],
    val_rows: Sequence[ChunkIndexRow],
    audit: Mapping[str, object],
    split_meta: Mapping[str, object],
    norm: Mapping[str, np.ndarray],
    vae_meta: Mapping[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "rows": rows_to_jsonable(rows),
        "train_rows": rows_to_jsonable(train_rows),
        "val_rows": rows_to_jsonable(val_rows),
        "audit": audit,
        "split": split_meta,
        "norm": {key: value.tolist() for key, value in norm.items()},
        "vae_meta": dict(vae_meta),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_manifest(path: Path) -> Dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["rows"] = rows_from_jsonable(payload["rows"])
    payload["train_rows"] = rows_from_jsonable(payload["train_rows"])
    payload["val_rows"] = rows_from_jsonable(payload["val_rows"])
    payload["norm"] = {key: np.asarray(value, dtype=np.float32) for key, value in payload["norm"].items()}
    return payload


def build_rows(cfg: ChunkDatasetConfig):
    rows, audit = build_default_index(cfg)
    return rows, audit
