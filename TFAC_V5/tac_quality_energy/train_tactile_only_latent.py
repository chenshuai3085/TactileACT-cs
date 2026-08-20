"""Train one independent tactile-only latent scorer from an explicit manifest.

The manifest is intentionally task-scoped: task selection happens outside the
model, while every sample contains only a marker HDF5 path, window coordinates,
and a label.  This entry point is shared by Board/Vase/Card/Chip/Socket but
never mixes their checkpoints or label spaces.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TFAC_V5.board_latent_energy.vae_utils import (
    encode_marker_chunk_to_latents,
    infer_vae_meta,
    load_tactile_vae_checkpoint,
    vae_checkpoint_identity,
)
from TFAC_V5.tac_quality_energy.tactile_only_latent import (
    FORMAL_TASKS,
    TactileOnlyLatentScorer,
    build_tactile_only_checkpoint,
)


FORBIDDEN_ROW_FIELDS = {"action", "action_path", "eef", "eef_action", "joint_action", "qpos", "task_id", "force", "force6d"}


def load_manifest(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    task = str(payload.get("task", "")).lower()
    if task not in FORMAL_TASKS:
        raise ValueError(f"manifest task must be one of {FORMAL_TASKS}, got {task!r}")
    class_names = payload.get("class_names")
    if not isinstance(class_names, list) or len(class_names) < 2:
        raise ValueError("manifest class_names must contain expert plus at least one negative class")
    for split in ("train_rows", "val_rows"):
        rows = payload.get(split)
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"manifest {split} must be a non-empty list")
        for row in rows:
            forbidden = sorted(FORBIDDEN_ROW_FIELDS.intersection(row))
            if forbidden:
                raise ValueError(f"manifest row contains forbidden non-tactile fields: {forbidden}")
            for key in ("path", "start", "label"):
                if key not in row:
                    raise ValueError(f"manifest row missing {key!r}")
    required_contract = {"horizon", "temporal_stride", "future_offset"}
    missing_contract = sorted(required_contract.difference(payload))
    if missing_contract:
        raise ValueError(f"manifest missing temporal contract fields: {missing_contract}")
    horizon = int(payload["horizon"])
    temporal_stride = int(payload["temporal_stride"])
    future_offset = int(payload["future_offset"])
    if horizon < 1 or temporal_stride < 1 or future_offset < 0:
        raise ValueError("manifest horizon/temporal_stride/future_offset are invalid")
    payload["task"] = task
    payload["horizon"] = horizon
    payload["temporal_stride"] = temporal_stride
    payload["future_offset"] = future_offset
    payload["marker_key"] = str(payload.get("marker_key", "observations/tac/left/marker_offset"))
    return payload


def _read_latents(rows: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any], vae, vae_info, device: torch.device):
    horizon = int(manifest["horizon"])
    temporal_stride = int(manifest["temporal_stride"])
    future_offset = int(manifest["future_offset"])
    marker_key = str(manifest["marker_key"])
    features, labels = [], []
    cache: Dict[str, np.ndarray] = {}
    for row in rows:
        path = str(row["path"])
        if path not in cache:
            with h5py.File(path, "r") as handle:
                if marker_key not in handle:
                    raise KeyError(f"{path}: missing marker key {marker_key!r}")
                cache[path] = handle[marker_key][:].astype(np.float32)
        marker = cache[path]
        start = int(row["start"]) + future_offset
        last = start + (horizon - 1) * temporal_stride
        if start < 0 or last >= marker.shape[0]:
            raise ValueError(f"row window exceeds marker sequence: {path}, start={row['start']}")
        latent = encode_marker_chunk_to_latents(marker, start, horizon, vae, vae_info, device, temporal_stride)
        features.append(latent)
        label = int(row["label"])
        if label < 0 or label >= len(manifest["class_names"]):
            raise ValueError(f"label {label} outside manifest class_names")
        labels.append(label)
    return np.stack(features).astype(np.float32), np.asarray(labels, dtype=np.int64)


class LatentRows(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, mean: np.ndarray, std: np.ndarray):
        self.features = torch.from_numpy((features - mean.reshape(1, 1, -1)) / np.maximum(std.reshape(1, 1, -1), 1e-6)).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {"latent": self.features[index], "label": self.labels[index]}


def run(args: argparse.Namespace) -> Dict[str, Any]:
    manifest = load_manifest(Path(args.manifest))
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    vae, vae_info = load_tactile_vae_checkpoint(args.tactile_vae_ckpt, device)
    vae_identity = vae_checkpoint_identity(args.tactile_vae_ckpt)
    train_x, train_y = _read_latents(manifest["train_rows"], manifest, vae, vae_info, device)
    val_x, val_y = _read_latents(manifest["val_rows"], manifest, vae, vae_info, device)
    latent_dim = int(train_x.shape[-1])
    if val_x.shape[-1] != latent_dim:
        raise ValueError("train/val latent dimensions differ")
    mean = train_x.reshape(-1, latent_dim).mean(axis=0).astype(np.float32)
    std = train_x.reshape(-1, latent_dim).std(axis=0).clip(min=1e-6).astype(np.float32)
    model = TactileOnlyLatentScorer(
        chunk_len=int(manifest["horizon"]), latent_dim=latent_dim,
        embed_dim=args.embed_dim, hidden=args.hidden, dropout=args.dropout,
        temperature=args.temperature, num_classes=len(manifest["class_names"]),
    ).to(device)
    train_loader = DataLoader(LatentRows(train_x, train_y, mean, std), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(LatentRows(val_x, val_y, mean, std), batch_size=args.batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = -float("inf")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            latent, labels = batch["latent"].to(device), batch["label"].to(device)
            loss = F.cross_entropy(model(latent)["logits"], labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_logits = torch.cat([model(batch["latent"].to(device))["logits"] for batch in val_loader])
        val_labels = torch.from_numpy(val_y).to(device)
        val_acc = float((val_logits.argmax(1) == val_labels).float().mean().cpu())
        history.append({"epoch": epoch, "val_accuracy": val_acc})
        if val_acc >= best:
            best = val_acc
            checkpoint = build_tactile_only_checkpoint(
                model, task=manifest["task"], latent_mean=mean, latent_std=std,
                class_names=manifest["class_names"], temporal_stride=manifest["temporal_stride"],
                future_offset=manifest["future_offset"], vae_identity=vae_identity,
            )
            checkpoint.update({"window_stride": int(manifest.get("window_stride", 1)), "vae_meta": infer_vae_meta(vae_info), "history": history})
            torch.save(checkpoint, output)
    return {"checkpoint": str(output), "task": manifest["task"], "vae_identity": vae_identity, "best_val_accuracy": best, "history": history}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--tactile_vae_ckpt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))
