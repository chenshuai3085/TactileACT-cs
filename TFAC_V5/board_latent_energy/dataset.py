"""Tactile-only latent chunk dataset for board energy training."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from TFAC_V5.board_chunk_energy.labels import BOARD_CLASS_NAMES, ClassSpec, default_class_specs
from TFAC_V5.board_latent_energy.vae_utils import encode_marker_chunk_to_latents, normalize_latent


SCHEMA_VERSION = 2
INPUT_MODE = "tactile_only_latent"
TASK = "board"
DEFAULT_MARKER_KEY = "observations/tac/left/marker_offset"


@dataclass(frozen=True)
class ChunkIndexRow:
    path: str
    label: int
    class_name: str
    start: int
    length: int
    episode_id: str
    temporal_stride: int = 1
    future_offset: int = 0


@dataclass(frozen=True)
class ChunkDatasetConfig:
    chunk_len: int = 16
    stride: int = 4
    temporal_stride: int = 1
    future_offset: int = 0
    marker_key: str = DEFAULT_MARKER_KEY
    contact_only: bool = True
    contact_quantile: float = 0.50
    min_contact_ratio: float = 0.25
    min_marker_norm: float = 1e-5
    max_episodes_per_class: Optional[int] = None
    seed: int = 42


def _episode_id(path: Path) -> str:
    return f"{path.parent.name}/{path.stem}"


def _read_marker_shape(path: Path, marker_key: str) -> Tuple[int, Tuple[int, ...]]:
    with h5py.File(path, "r") as f:
        if marker_key not in f:
            raise KeyError(f"{path}: missing marker key {marker_key!r}")
        shape = tuple(f[marker_key].shape)
    return int(shape[0]), shape


def _target_indices(start: int, length: int, temporal_stride: int, future_offset: int) -> np.ndarray:
    return start + future_offset + np.arange(length, dtype=np.int64) * temporal_stride


def _candidate_starts(cfg: ChunkDatasetConfig, n: int) -> List[int]:
    required_span = cfg.future_offset + (cfg.chunk_len - 1) * cfg.temporal_stride + 1
    return list(range(0, max(0, n - required_span + 1), cfg.stride))


def _contact_starts(path: Path, cfg: ChunkDatasetConfig, n: int) -> List[int]:
    starts = _candidate_starts(cfg, n)
    if not cfg.contact_only or not starts:
        return starts
    with h5py.File(path, "r") as f:
        marker = f[cfg.marker_key][:n].astype(np.float32)
    magnitude = np.linalg.norm(marker, axis=-1).mean(axis=(1, 2))
    threshold = max(float(np.quantile(magnitude, cfg.contact_quantile)), float(cfg.min_marker_norm))
    return [
        start
        for start in starts
        if float(np.mean(magnitude[_target_indices(start, cfg.chunk_len, cfg.temporal_stride, cfg.future_offset)] >= threshold))
        >= cfg.min_contact_ratio
    ]


def build_index(class_specs: Sequence[ClassSpec], cfg: ChunkDatasetConfig) -> Tuple[List[ChunkIndexRow], Dict[str, object]]:
    if cfg.chunk_len < 1:
        raise ValueError(f"chunk_len must be >= 1, got {cfg.chunk_len}")
    if cfg.stride < 1:
        raise ValueError(f"stride must be >= 1, got {cfg.stride}")
    if cfg.temporal_stride < 1:
        raise ValueError(f"temporal_stride must be >= 1, got {cfg.temporal_stride}")
    if cfg.future_offset < 0:
        raise ValueError(f"future_offset must be >= 0, got {cfg.future_offset}")

    rng = np.random.default_rng(cfg.seed)
    rows: List[ChunkIndexRow] = []
    audit: Dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "input_mode": INPUT_MODE,
        "task": TASK,
        "config": asdict(cfg),
        "class_names": list(BOARD_CLASS_NAMES),
        "classes": {},
    }
    required_span = cfg.future_offset + (cfg.chunk_len - 1) * cfg.temporal_stride + 1
    for spec in class_specs:
        files = sorted(Path(spec.root).glob("*.hdf5"))
        if cfg.max_episodes_per_class is not None and len(files) > cfg.max_episodes_per_class:
            selected = sorted(rng.choice(len(files), cfg.max_episodes_per_class, replace=False).tolist())
            files = [files[index] for index in selected]
        class_rows: List[ChunkIndexRow] = []
        lengths: List[int] = []
        marker_shapes: List[Tuple[int, ...]] = []
        skipped: List[Dict[str, object]] = []
        for path in files:
            try:
                n, marker_shape = _read_marker_shape(path, cfg.marker_key)
                marker_shapes.append(marker_shape)
                lengths.append(n)
                if n < required_span:
                    skipped.append({"path": str(path), "reason": f"length {n} < required_span {required_span}"})
                    continue
                for start in _contact_starts(path, cfg, n):
                    class_rows.append(
                        ChunkIndexRow(
                            path=str(path),
                            label=spec.label,
                            class_name=spec.name,
                            start=int(start),
                            length=int(cfg.chunk_len),
                            episode_id=_episode_id(path),
                            temporal_stride=int(cfg.temporal_stride),
                            future_offset=int(cfg.future_offset),
                        )
                    )
            except Exception as exc:
                skipped.append({"path": str(path), "reason": repr(exc)})
        rows.extend(class_rows)
        audit["classes"][spec.name] = {
            "label": spec.label,
            "root": str(spec.root),
            "episodes": len(files),
            "usable_episodes": len(lengths),
            "windows": len(class_rows),
            "length_min": int(min(lengths)) if lengths else None,
            "length_max": int(max(lengths)) if lengths else None,
            "length_mean": float(np.mean(lengths)) if lengths else None,
            "marker_shape_examples": [list(shape) for shape in marker_shapes[:3]],
            "skipped_preview": skipped[:10],
        }
    audit["total_windows"] = len(rows)
    audit["total_episodes"] = len({row.episode_id for row in rows})
    return rows, audit


def build_rows(cfg: ChunkDatasetConfig):
    return build_index(default_class_specs(), cfg)


def split_rows_by_episode(
    rows: Sequence[ChunkIndexRow], val_ratio: float = 0.20, seed: int = 42
) -> Tuple[List[ChunkIndexRow], List[ChunkIndexRow], Dict[str, object]]:
    rng = np.random.default_rng(seed)
    by_class: Dict[int, Dict[str, List[ChunkIndexRow]]] = {}
    for row in rows:
        by_class.setdefault(row.label, {}).setdefault(row.episode_id, []).append(row)
    train: List[ChunkIndexRow] = []
    val: List[ChunkIndexRow] = []
    split_meta: Dict[str, object] = {"val_ratio": val_ratio, "classes": {}}
    for label, episode_map in sorted(by_class.items()):
        episodes = sorted(episode_map)
        shuffled = episodes.copy()
        rng.shuffle(shuffled)
        n_val = max(1, int(math.ceil(len(shuffled) * val_ratio))) if len(shuffled) > 1 else 0
        val_episodes = set(shuffled[:n_val])
        for episode in episodes:
            (val if episode in val_episodes else train).extend(episode_map[episode])
        split_meta["classes"][BOARD_CLASS_NAMES[label]] = {
            "train_episodes": sum(episode not in val_episodes for episode in episodes),
            "val_episodes": len(val_episodes),
            "train_windows": sum(len(episode_map[episode]) for episode in episodes if episode not in val_episodes),
            "val_windows": sum(len(episode_map[episode]) for episode in episodes if episode in val_episodes),
        }
    return train, val, split_meta


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
        raise ValueError("Cannot compute normalization from empty rows")
    rng = np.random.default_rng(seed)
    sample_rows = list(rows)
    if len(sample_rows) > max_windows:
        selected = rng.choice(len(sample_rows), max_windows, replace=False)
        sample_rows = [sample_rows[index] for index in sorted(selected.tolist())]
    latent_dim = int(vae_info["latent_flat_dim"])
    latent_sum = np.zeros((latent_dim,), dtype=np.float64)
    latent_sumsq = np.zeros((latent_dim,), dtype=np.float64)
    latent_count = 0
    marker_by_path: Dict[str, np.ndarray] = {}
    for row in tqdm(sample_rows, desc="latent norm", leave=False):
        if row.path not in marker_by_path:
            with h5py.File(row.path, "r") as f:
                marker_by_path[row.path] = f[marker_key][:].astype(np.float32)
        latent = encode_marker_chunk_to_latents(
            marker_by_path[row.path],
            row.start + row.future_offset,
            row.length,
            vae,
            vae_info,
            device,
            temporal_stride=row.temporal_stride,
        )
        flat = latent.reshape(-1, latent.shape[-1]).astype(np.float64)
        latent_sum += flat.sum(axis=0)
        latent_sumsq += np.square(flat).sum(axis=0)
        latent_count += flat.shape[0]
    mean = latent_sum / max(1, latent_count)
    variance = latent_sumsq / max(1, latent_count) - np.square(mean)
    return {
        "latent_mean": mean.astype(np.float32),
        "latent_std": np.sqrt(np.maximum(variance, 1e-8)).astype(np.float32),
    }


compute_normalization = compute_latent_normalization


class BoardLatentChunkDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[ChunkIndexRow],
        norm: Mapping[str, np.ndarray],
        vae,
        vae_info: Mapping[str, object],
        device: torch.device,
        marker_key: str = DEFAULT_MARKER_KEY,
        include_path: bool = False,
        preload: bool = True,
    ):
        unexpected_norm = set(norm) - {"latent_mean", "latent_std"}
        if unexpected_norm:
            raise ValueError(f"Tactile-only dataset rejects non-latent normalization keys: {sorted(unexpected_norm)}")
        latent_dim = int(vae_info["latent_flat_dim"])
        if set(norm) != {"latent_mean", "latent_std"}:
            raise ValueError("Tactile-only dataset requires latent_mean and latent_std")
        latent_mean = np.asarray(norm["latent_mean"], dtype=np.float32).reshape(-1)
        latent_std = np.asarray(norm["latent_std"], dtype=np.float32).reshape(-1)
        if latent_mean.size != latent_dim or latent_std.size != latent_dim:
            raise ValueError(f"Latent normalization must contain {latent_dim} values")
        if not np.isfinite(latent_mean).all() or not np.isfinite(latent_std).all() or not (latent_std > 0).all():
            raise ValueError("Latent normalization must be finite and latent_std must be positive")
        self.rows = list(rows)
        self.norm = norm
        self.vae = vae
        self.vae_info = vae_info
        self.device = device
        self.marker_key = marker_key
        self.include_path = include_path
        self._cache: Optional[List[np.ndarray]] = None
        if preload:
            self._cache = []
            marker_by_path: Dict[str, np.ndarray] = {}
            for row in tqdm(self.rows, desc="preload latent", leave=False):
                if row.path not in marker_by_path:
                    with h5py.File(row.path, "r") as f:
                        marker_by_path[row.path] = f[self.marker_key][:].astype(np.float32)
                self._cache.append(self._read_and_normalize(row, marker_by_path[row.path]))

    def __len__(self) -> int:
        return len(self.rows)

    def _read_and_normalize(self, row: ChunkIndexRow, marker: Optional[np.ndarray] = None) -> np.ndarray:
        if marker is None:
            with h5py.File(row.path, "r") as f:
                marker = f[self.marker_key][:].astype(np.float32)
        latent = encode_marker_chunk_to_latents(
            marker,
            row.start + row.future_offset,
            row.length,
            self.vae,
            self.vae_info,
            self.device,
            temporal_stride=row.temporal_stride,
        )
        return normalize_latent(latent, self.norm)

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.rows[index]
        latent = self._read_and_normalize(row) if self._cache is None else self._cache[index]
        item: Dict[str, object] = {
            "latent": torch.from_numpy(latent),
            "label": torch.tensor(row.label, dtype=torch.long),
        }
        if self.include_path:
            item.update({"path": row.path, "start": row.start, "class_name": row.class_name})
        return item


def rows_to_jsonable(rows: Sequence[ChunkIndexRow]) -> List[Dict[str, object]]:
    return [asdict(row) for row in rows]


def rows_from_jsonable(items: Iterable[Mapping[str, object]]) -> List[ChunkIndexRow]:
    return [ChunkIndexRow(**dict(item)) for item in items]


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
    vae_identity: str,
    horizon: int,
    latent_shape: Sequence[int],
    window_stride: int,
    temporal_stride: int,
    future_offset: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "input_mode": INPUT_MODE,
        "task": TASK,
        "horizon": int(horizon),
        "latent_shape": [int(value) for value in latent_shape],
        "window_stride": int(window_stride),
        "temporal_stride": int(temporal_stride),
        "future_offset": int(future_offset),
        "vae_identity": vae_identity,
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
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("input_mode") != INPUT_MODE:
        raise ValueError(f"{path} is not a schema-v2 tactile-only latent manifest")
    required = {"task", "horizon", "latent_shape", "window_stride", "temporal_stride", "future_offset", "vae_identity"}
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"{path} is missing required tactile-only metadata: {missing}")
    if payload["task"] != TASK:
        raise ValueError(f"Expected task={TASK!r}, got {payload['task']!r}")
    if isinstance(payload["horizon"], bool) or not isinstance(payload["horizon"], int) or payload["horizon"] < 1:
        raise ValueError("Schema-v2 manifest horizon must be an integer >= 1")
    if (
        not isinstance(payload["latent_shape"], list)
        or len(payload["latent_shape"]) != 1
        or isinstance(payload["latent_shape"][0], bool)
        or not isinstance(payload["latent_shape"][0], int)
        or payload["latent_shape"][0] < 1
    ):
        raise ValueError("Schema-v2 manifest latent_shape must be [positive latent dimension]")
    for key, minimum in (("window_stride", 1), ("temporal_stride", 1), ("future_offset", 0)):
        value = payload[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"Schema-v2 manifest {key} must be an integer >= {minimum}")
    if not isinstance(payload["vae_identity"], str) or not payload["vae_identity"].strip():
        raise ValueError("Schema-v2 manifest vae_identity must be a non-empty string")
    if set(payload["norm"]) != {"latent_mean", "latent_std"}:
        raise ValueError("Schema-v2 manifest norm must contain only latent_mean and latent_std")
    payload["rows"] = rows_from_jsonable(payload["rows"])
    payload["train_rows"] = rows_from_jsonable(payload["train_rows"])
    payload["val_rows"] = rows_from_jsonable(payload["val_rows"])
    payload["norm"] = {key: np.asarray(value, dtype=np.float32) for key, value in payload["norm"].items()}
    latent_dim = payload["latent_shape"][0]
    for key in ("latent_mean", "latent_std"):
        value = payload["norm"][key].reshape(-1)
        if value.size != latent_dim or not np.isfinite(value).all():
            raise ValueError(f"Manifest {key} must contain {latent_dim} finite values")
    if not (payload["norm"]["latent_std"] > 0).all():
        raise ValueError("Manifest latent_std must be positive")
    if int(payload.get("vae_meta", {}).get("latent_flat_dim", latent_dim)) != latent_dim:
        raise ValueError("Manifest VAE latent dimension does not match latent_shape")
    for split_name in ("rows", "train_rows", "val_rows"):
        for row in payload[split_name]:
            if row.length != payload["horizon"]:
                raise ValueError(f"Manifest {split_name} row horizon mismatch: {row.length}")
            if row.temporal_stride != payload["temporal_stride"]:
                raise ValueError(f"Manifest {split_name} row temporal_stride mismatch: {row.temporal_stride}")
            if row.future_offset != payload["future_offset"]:
                raise ValueError(f"Manifest {split_name} row future_offset mismatch: {row.future_offset}")
    return payload
