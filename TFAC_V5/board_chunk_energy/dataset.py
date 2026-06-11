"""HDF5 chunk dataset for board tactile consequence energy training.

The first version intentionally uses only:

  joint action chunk + future left marker chunk -> quality class

History marker is kept out of the default input so the scorer focuses on the
predicted tactile consequence that will be available from Foresight at
deployment time.
"""

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

from .labels import BOARD_CLASS_NAMES, ClassSpec, default_class_specs


DEFAULT_MARKER_KEY = "observations/tac/left/marker_offset"
DEFAULT_ACTION_KEY = "actions/joint_abs"


@dataclass(frozen=True)
class ChunkIndexRow:
    path: str
    label: int
    class_name: str
    start: int
    length: int
    episode_id: str


@dataclass(frozen=True)
class ChunkDatasetConfig:
    chunk_len: int = 16
    stride: int = 4
    marker_key: str = DEFAULT_MARKER_KEY
    action_key: str = DEFAULT_ACTION_KEY
    contact_only: bool = True
    contact_quantile: float = 0.50
    min_contact_ratio: float = 0.25
    min_marker_norm: float = 1e-5
    max_episodes_per_class: Optional[int] = None
    seed: int = 42


def _episode_id(path: Path) -> str:
    return f"{path.parent.name}/{path.stem}"


def _read_len(path: Path, marker_key: str, action_key: str) -> Tuple[int, Tuple[int, ...], Tuple[int, ...]]:
    with h5py.File(path, "r") as f:
        if marker_key not in f:
            raise KeyError(f"{path}: missing marker key {marker_key!r}")
        if action_key not in f:
            raise KeyError(f"{path}: missing action key {action_key!r}")
        marker_shape = tuple(f[marker_key].shape)
        action_shape = tuple(f[action_key].shape)
        return min(marker_shape[0], action_shape[0]), marker_shape, action_shape


def _contact_starts(path: Path, cfg: ChunkDatasetConfig, n: int) -> List[int]:
    starts = list(range(0, max(0, n - cfg.chunk_len + 1), cfg.stride))
    if not cfg.contact_only or not starts:
        return starts

    with h5py.File(path, "r") as f:
        marker = f[cfg.marker_key][:n].astype(np.float32)
    mag = np.linalg.norm(marker, axis=-1).mean(axis=(1, 2))
    threshold = max(float(np.quantile(mag, cfg.contact_quantile)), float(cfg.min_marker_norm))

    keep: List[int] = []
    for start in starts:
        window = mag[start:start + cfg.chunk_len]
        if float(np.mean(window >= threshold)) >= cfg.min_contact_ratio:
            keep.append(start)
    return keep


def build_index(
    class_specs: Sequence[ClassSpec],
    cfg: ChunkDatasetConfig,
) -> Tuple[List[ChunkIndexRow], Dict[str, object]]:
    rng = np.random.default_rng(cfg.seed)
    rows: List[ChunkIndexRow] = []
    audit: Dict[str, object] = {
        "config": asdict(cfg),
        "class_names": list(BOARD_CLASS_NAMES),
        "classes": {},
    }

    for spec in class_specs:
        files = sorted(Path(spec.root).glob("*.hdf5"))
        if cfg.max_episodes_per_class is not None and len(files) > cfg.max_episodes_per_class:
            chosen = sorted(rng.choice(len(files), cfg.max_episodes_per_class, replace=False).tolist())
            files = [files[i] for i in chosen]

        class_rows: List[ChunkIndexRow] = []
        lengths: List[int] = []
        skipped: List[Dict[str, object]] = []
        marker_shapes: List[Tuple[int, ...]] = []
        action_shapes: List[Tuple[int, ...]] = []

        for path in files:
            try:
                n, marker_shape, action_shape = _read_len(path, cfg.marker_key, cfg.action_key)
                marker_shapes.append(marker_shape)
                action_shapes.append(action_shape)
                lengths.append(n)
                if n < cfg.chunk_len:
                    skipped.append({"path": str(path), "reason": f"length {n} < chunk_len"})
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
            "marker_shape_examples": [list(x) for x in marker_shapes[:3]],
            "action_shape_examples": [list(x) for x in action_shapes[:3]],
            "skipped_preview": skipped[:10],
        }

    audit["total_windows"] = len(rows)
    audit["total_episodes"] = len({row.episode_id for row in rows})
    return rows, audit


def split_rows_by_episode(
    rows: Sequence[ChunkIndexRow],
    val_ratio: float = 0.20,
    seed: int = 42,
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
        val_eps = set(shuffled[:n_val])
        for ep in episodes:
            target = val if ep in val_eps else train
            target.extend(episode_map[ep])
        split_meta["classes"][BOARD_CLASS_NAMES[label]] = {
            "train_episodes": len([ep for ep in episodes if ep not in val_eps]),
            "val_episodes": len(val_eps),
            "train_windows": sum(len(episode_map[ep]) for ep in episodes if ep not in val_eps),
            "val_windows": sum(len(episode_map[ep]) for ep in episodes if ep in val_eps),
        }

    return train, val, split_meta


def compute_normalization(
    rows: Sequence[ChunkIndexRow],
    marker_key: str = DEFAULT_MARKER_KEY,
    action_key: str = DEFAULT_ACTION_KEY,
    max_windows: int = 4096,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    if not rows:
        raise ValueError("Cannot compute normalization from empty rows")
    rng = np.random.default_rng(seed)
    if len(rows) > max_windows:
        idx = rng.choice(len(rows), max_windows, replace=False)
        sample_rows = [rows[i] for i in sorted(idx.tolist())]
    else:
        sample_rows = list(rows)

    marker_sum = np.zeros((2,), dtype=np.float64)
    marker_sumsq = np.zeros((2,), dtype=np.float64)
    marker_count = 0
    action_sum: Optional[np.ndarray] = None
    action_sumsq: Optional[np.ndarray] = None
    action_count = 0

    for row in sample_rows:
        with h5py.File(row.path, "r") as f:
            marker = f[marker_key][row.start:row.start + row.length].astype(np.float64)
            action = f[action_key][row.start:row.start + row.length].astype(np.float64)
        marker_2d = marker.reshape(-1, marker.shape[-1])
        marker_sum += marker_2d.sum(axis=0)
        marker_sumsq += np.square(marker_2d).sum(axis=0)
        marker_count += marker_2d.shape[0]

        action_2d = action.reshape(-1, action.shape[-1])
        if action_sum is None:
            action_sum = np.zeros((action_2d.shape[-1],), dtype=np.float64)
            action_sumsq = np.zeros((action_2d.shape[-1],), dtype=np.float64)
        action_sum += action_2d.sum(axis=0)
        action_sumsq += np.square(action_2d).sum(axis=0)
        action_count += action_2d.shape[0]

    assert action_sum is not None and action_sumsq is not None
    marker_mean = marker_sum / max(1, marker_count)
    marker_var = marker_sumsq / max(1, marker_count) - np.square(marker_mean)
    action_mean = action_sum / max(1, action_count)
    action_var = action_sumsq / max(1, action_count) - np.square(action_mean)
    return {
        "marker_mean": marker_mean.astype(np.float32),
        "marker_std": np.sqrt(np.maximum(marker_var, 1e-8)).astype(np.float32),
        "action_mean": action_mean.astype(np.float32),
        "action_std": np.sqrt(np.maximum(action_var, 1e-8)).astype(np.float32),
    }


class BoardChunkDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[ChunkIndexRow],
        norm: Mapping[str, np.ndarray],
        marker_key: str = DEFAULT_MARKER_KEY,
        action_key: str = DEFAULT_ACTION_KEY,
        include_path: bool = False,
        preload: bool = False,
    ):
        self.rows = list(rows)
        self.marker_key = marker_key
        self.action_key = action_key
        self.include_path = include_path
        self.preload = preload
        self.marker_mean = np.asarray(norm["marker_mean"], dtype=np.float32).reshape(1, 1, 1, 2)
        self.marker_std = np.asarray(norm["marker_std"], dtype=np.float32).reshape(1, 1, 1, 2)
        self.action_mean = np.asarray(norm["action_mean"], dtype=np.float32).reshape(1, -1)
        self.action_std = np.asarray(norm["action_std"], dtype=np.float32).reshape(1, -1)
        self._cache: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
        if preload:
            self._cache = []
            for row in self.rows:
                self._cache.append(self._read_and_normalize(row))

    def __len__(self) -> int:
        return len(self.rows)

    def _read_and_normalize(self, row: ChunkIndexRow) -> Tuple[np.ndarray, np.ndarray]:
        with h5py.File(row.path, "r") as f:
            marker = f[self.marker_key][row.start:row.start + row.length].astype(np.float32)
            action = f[self.action_key][row.start:row.start + row.length].astype(np.float32)
        marker = (marker - self.marker_mean) / np.maximum(self.marker_std, 1e-6)
        action = (action - self.action_mean) / np.maximum(self.action_std, 1e-6)
        return marker.astype(np.float32), action.astype(np.float32)

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.rows[index]
        if self._cache is None:
            marker, action = self._read_and_normalize(row)
        else:
            marker, action = self._cache[index]
        item: Dict[str, object] = {
            "marker": torch.from_numpy(marker),
            "action": torch.from_numpy(action),
            "label": torch.tensor(row.label, dtype=torch.long),
        }
        if self.include_path:
            item.update({"path": row.path, "start": row.start, "class_name": row.class_name})
        return item


def rows_to_jsonable(rows: Sequence[ChunkIndexRow]) -> List[Dict[str, object]]:
    return [asdict(row) for row in rows]


def rows_from_jsonable(items: Iterable[Mapping[str, object]]) -> List[ChunkIndexRow]:
    return [
        ChunkIndexRow(
            path=str(item["path"]),
            label=int(item["label"]),
            class_name=str(item["class_name"]),
            start=int(item["start"]),
            length=int(item["length"]),
            episode_id=str(item["episode_id"]),
        )
        for item in items
    ]


def write_manifest(
    path: Path,
    *,
    rows: Sequence[ChunkIndexRow],
    train_rows: Sequence[ChunkIndexRow],
    val_rows: Sequence[ChunkIndexRow],
    audit: Mapping[str, object],
    split_meta: Mapping[str, object],
    norm: Mapping[str, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "rows": rows_to_jsonable(rows),
        "train_rows": rows_to_jsonable(train_rows),
        "val_rows": rows_to_jsonable(val_rows),
        "audit": audit,
        "split": split_meta,
        "norm": {key: value.tolist() for key, value in norm.items()},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_manifest(path: Path) -> Dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["rows"] = rows_from_jsonable(payload["rows"])
    payload["train_rows"] = rows_from_jsonable(payload["train_rows"])
    payload["val_rows"] = rows_from_jsonable(payload["val_rows"])
    payload["norm"] = {key: np.asarray(value, dtype=np.float32) for key, value in payload["norm"].items()}
    return payload


def build_default_index(cfg: ChunkDatasetConfig) -> Tuple[List[ChunkIndexRow], Dict[str, object]]:
    return build_index(default_class_specs(), cfg)
