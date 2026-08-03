#!/usr/bin/env python3
"""Train a five-task, 5-channel TacVAE from an explicit episode manifest.

The manifest is JSONL (one episode per line) or a JSON list. Required fields:
``path``, ``task``, and ``split``. ``condition`` defaults to the parent
directory name. ``episode_id``, ``source_episode_id``, and ``marker_sha256``
are optional but recommended for provenance and leakage checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from tactile_vae import TactileVAE  # noqa: E402


DEFAULT_MANIFEST = Path("outputs/multitask_tacvae_5task_20260803/manifest.jsonl")
DEFAULT_TASKS = ("board", "vase", "card", "chip", "socket")
LEFT_MARKER_KEY = "observations/tac/left/marker_offset"
RIGHT_MARKER_KEY = "observations/tac/right/marker_offset"


@dataclass(frozen=True)
class EpisodeRecord:
    path: str
    task: str
    condition: str
    split: str
    episode_id: str
    source_episode_id: str
    marker_sha256: str = ""


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_raw_manifest(path: Path) -> List[Mapping[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        if isinstance(payload, dict):
            payload = payload.get("episodes", payload.get("records"))
        if not isinstance(payload, list):
            raise ValueError("JSON manifest must be a list or contain an episodes/records list")
        return payload

    rows: List[Mapping[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"manifest line {line_no} is not a JSON object")
        rows.append(row)
    return rows


def load_manifest(path: Path, expected_tasks: Sequence[str]) -> Tuple[List[EpisodeRecord], str]:
    records: List[EpisodeRecord] = []
    seen_paths: Dict[str, str] = {}
    seen_episode_ids: Dict[str, str] = {}
    seen_source_ids: Dict[str, str] = {}
    seen_hashes: Dict[str, str] = {}
    allowed_splits = {"train", "val", "test"}

    for row_no, row in enumerate(_load_raw_manifest(path), start=1):
        missing = [key for key in ("path", "task", "split") if not row.get(key)]
        if missing:
            raise ValueError(f"manifest row {row_no} missing fields: {missing}")
        episode_path = Path(str(row["path"])).expanduser().resolve()
        task = str(row["task"]).strip().lower()
        split = str(row["split"]).strip().lower()
        condition = str(row.get("condition") or episode_path.parent.name).strip()
        side = str(row.get("side") or "left").strip().lower()
        episode_id = str(row.get("episode_id") or episode_path).strip()
        source_id = str(row.get("source_episode_id") or episode_id).strip()
        marker_hash = str(row.get("marker_sha256") or "").strip().lower()
        if split not in allowed_splits:
            raise ValueError(f"manifest row {row_no} has invalid split={split!r}")
        if task not in expected_tasks:
            raise ValueError(f"manifest row {row_no} has unexpected task={task!r}")
        if side != "left":
            raise ValueError(f"manifest row {row_no} has side={side!r}; this protocol is left-only")
        if not episode_path.is_file():
            raise FileNotFoundError(f"manifest row {row_no} missing episode: {episode_path}")

        canonical_path = str(episode_path)
        if canonical_path in seen_paths:
            raise ValueError(f"duplicate episode path: {canonical_path}")
        if episode_id in seen_episode_ids:
            raise ValueError(
                f"duplicate episode_id={episode_id!r}: {seen_episode_ids[episode_id]} and {canonical_path}"
            )
        if source_id in seen_source_ids:
            raise ValueError(
                f"duplicate source_episode_id={source_id!r}: {seen_source_ids[source_id]} and {canonical_path}"
            )
        if marker_hash and marker_hash in seen_hashes:
            raise ValueError(f"duplicate marker_sha256={marker_hash}: {seen_hashes[marker_hash]} and {canonical_path}")
        seen_paths[canonical_path] = split
        seen_episode_ids[episode_id] = canonical_path
        seen_source_ids[source_id] = canonical_path
        if marker_hash:
            seen_hashes[marker_hash] = canonical_path
        records.append(
            EpisodeRecord(
                path=canonical_path,
                task=task,
                condition=condition,
                split=split,
                episode_id=episode_id,
                source_episode_id=source_id,
                marker_sha256=marker_hash,
            )
        )

    if not records:
        raise ValueError("manifest has no episode records")
    for split in ("train", "val"):
        present = {record.task for record in records if record.split == split}
        missing_tasks = sorted(set(expected_tasks) - present)
        if missing_tasks:
            raise ValueError(f"split={split} is missing tasks: {missing_tasks}")

    canonical_rows = [asdict(record) for record in sorted(records, key=lambda record: record.episode_id)]
    canonical = json.dumps(canonical_rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    manifest_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return records, manifest_hash


def marker_digest(left: np.ndarray, right: np.ndarray) -> str:
    """Match build_multitask_tacvae_5task_manifest.py exactly."""
    digest = hashlib.sha256()
    digest.update(str(left.shape).encode("ascii"))
    digest.update(left.tobytes(order="C"))
    digest.update(right.tobytes(order="C"))
    return digest.hexdigest()


def load_marker(record: EpisodeRecord, temporal_window: int) -> Tuple[np.ndarray, str]:
    try:
        with h5py.File(record.path, "r") as h5:
            arrays = []
            for key in (LEFT_MARKER_KEY, RIGHT_MARKER_KEY):
                if key not in h5:
                    raise ValueError(f"missing {key}")
                marker = np.ascontiguousarray(h5[key][()].astype(np.float32, copy=False))
                if marker.ndim != 4 or marker.shape[1:] != (9, 9, 2):
                    raise ValueError(f"invalid {key} shape {marker.shape}, expected (T,9,9,2)")
                if marker.shape[0] < temporal_window:
                    raise ValueError(f"{key} length {marker.shape[0]} < temporal_window={temporal_window}")
                if not np.isfinite(marker).all():
                    raise ValueError(f"{key} contains NaN or Inf")
                arrays.append(marker)
    except OSError as exc:
        raise ValueError(f"cannot read HDF5") from exc
    left, right = arrays
    if left.shape[0] != right.shape[0]:
        raise ValueError(f"left/right length mismatch: {left.shape[0]} vs {right.shape[0]}")
    return left, marker_digest(left, right)


def preload_markers(
    records: Sequence[EpisodeRecord], temporal_window: int
) -> Dict[str, np.ndarray]:
    markers: Dict[str, np.ndarray] = {}
    errors: List[str] = []
    seen_hashes: Dict[str, str] = {}
    for record in records:
        try:
            left, actual_hash = load_marker(record, temporal_window)
            if record.marker_sha256 and actual_hash != record.marker_sha256:
                raise ValueError(
                    f"marker_sha256 mismatch: manifest={record.marker_sha256} actual={actual_hash}"
                )
            if actual_hash in seen_hashes:
                raise ValueError(f"duplicate marker sequence: {seen_hashes[actual_hash]} and {record.path}")
            seen_hashes[actual_hash] = record.path
            markers[record.episode_id] = left
        except ValueError as exc:
            errors.append(f"{record.path}: {exc}")
    if errors:
        preview = "\n".join(errors[:20])
        suffix = f"\n... and {len(errors) - 20} more" if len(errors) > 20 else ""
        raise ValueError(f"invalid manifest episodes:\n{preview}{suffix}")
    return markers


def compute_task_balanced_norm(
    train_records: Sequence[EpisodeRecord], markers: Mapping[str, np.ndarray], tasks: Sequence[str]
) -> Dict[str, Any]:
    per_task: Dict[str, Dict[str, Any]] = {}
    task_means = []
    task_seconds = []
    for task in tasks:
        condition_records: Dict[str, List[EpisodeRecord]] = defaultdict(list)
        for record in train_records:
            if record.task == task:
                condition_records[record.condition].append(record)
        if not condition_records:
            raise ValueError(f"task={task} has no training episodes")

        condition_means = []
        condition_seconds = []
        condition_summary: Dict[str, Any] = {}
        for condition in sorted(condition_records):
            episode_means = []
            episode_seconds = []
            for record in condition_records[condition]:
                flat = markers[record.episode_id].reshape(-1, 2).astype(np.float64)
                episode_means.append(flat.mean(axis=0))
                episode_seconds.append(np.square(flat).mean(axis=0))
            condition_mean = np.mean(np.stack(episode_means), axis=0)
            condition_second = np.mean(np.stack(episode_seconds), axis=0)
            condition_means.append(condition_mean)
            condition_seconds.append(condition_second)
            condition_summary[condition] = {
                "mean": condition_mean.tolist(),
                "second_moment": condition_second.tolist(),
                "episodes": len(episode_means),
            }
        task_mean = np.mean(np.stack(condition_means), axis=0)
        task_second = np.mean(np.stack(condition_seconds), axis=0)
        task_means.append(task_mean)
        task_seconds.append(task_second)
        per_task[task] = {
            "mean": task_mean.tolist(),
            "second_moment": task_second.tolist(),
            "conditions": condition_summary,
        }

    mean = np.mean(np.stack(task_means), axis=0)
    second = np.mean(np.stack(task_seconds), axis=0)
    std = np.sqrt(np.maximum(second - np.square(mean), 1e-12))
    return {
        "mean": mean.astype(np.float32).tolist(),
        "std": std.astype(np.float32).tolist(),
        "weighting": "equal_task_then_equal_condition_then_equal_episode",
        "per_task": per_task,
    }


class ManifestWindowDataset(Dataset):
    def __init__(
        self,
        records: Sequence[EpisodeRecord],
        markers: Mapping[str, np.ndarray],
        split: str,
        temporal_window: int,
        window_stride: int,
        mean: Sequence[float],
        std: Sequence[float],
    ):
        self.temporal_window = temporal_window
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 1, 2)
        self.std = np.asarray(std, dtype=np.float32).reshape(1, 1, 1, 2)
        self.records = [record for record in records if record.split == split]
        self.markers = markers
        self.items: List[Tuple[int, int]] = []
        self.indices: Dict[str, Dict[str, Dict[str, List[int]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        for record_idx, record in enumerate(self.records):
            marker = markers[record.episode_id]
            for start in range(0, marker.shape[0] - temporal_window + 1, window_stride):
                item_idx = len(self.items)
                self.items.append((record_idx, start))
                self.indices[record.task][record.condition][record.episode_id].append(item_idx)
        if not self.items:
            raise ValueError(f"split={split} has no valid temporal windows")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        record_idx, start = self.items[idx]
        record = self.records[record_idx]
        marker = self.markers[record.episode_id]
        seq = marker[start : start + self.temporal_window]
        seq = (seq - self.mean) / self.std
        return torch.from_numpy(seq.astype(np.float32)), record.task


class HierarchicalBalancedBatchSampler(Sampler[List[int]]):
    """Sample task -> condition -> episode -> window with exact task balance."""

    def __init__(
        self,
        dataset: ManifestWindowDataset,
        tasks: Sequence[str],
        batch_size: int,
        steps_per_epoch: int,
        seed: int,
    ):
        if batch_size % len(tasks) != 0:
            raise ValueError(f"batch_size={batch_size} must be divisible by {len(tasks)} tasks")
        if steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch must be positive")
        self.dataset = dataset
        self.tasks = tuple(tasks)
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.seed = seed
        self.epoch = 0
        missing = [task for task in self.tasks if task not in dataset.indices]
        if missing:
            raise ValueError(f"training dataset missing sampler tasks: {missing}")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch]))
        per_task = self.batch_size // len(self.tasks)
        for _ in range(self.steps_per_epoch):
            batch: List[int] = []
            for task in self.tasks:
                condition_map = self.dataset.indices[task]
                conditions = tuple(sorted(condition_map))
                for _sample in range(per_task):
                    condition = conditions[int(rng.integers(len(conditions)))]
                    episode_map = condition_map[condition]
                    episodes = tuple(sorted(episode_map))
                    episode = episodes[int(rng.integers(len(episodes)))]
                    windows = episode_map[episode]
                    batch.append(windows[int(rng.integers(len(windows)))])
            rng.shuffle(batch)
            yield batch


@torch.no_grad()
def evaluate(
    model: TactileVAE,
    dataset: ManifestWindowDataset,
    batch_size: int,
    device: torch.device,
    num_workers: int = 0,
) -> Dict[str, Any]:
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    task_sse: Dict[str, float] = defaultdict(float)
    task_count: Dict[str, int] = defaultdict(int)
    for x, task_names in loader:
        x = x.to(device)
        recon, _, _ = model(x)
        gt = TactileVAE.align_gt(x, temporal_stride=2)
        squared = torch.square(recon - gt).flatten(1).sum(dim=1).cpu().numpy()
        elements = int(np.prod(gt.shape[1:]))
        for index, task in enumerate(task_names):
            task_sse[str(task)] += float(squared[index])
            task_count[str(task)] += elements
    per_task_mse = {task: task_sse[task] / task_count[task] for task in sorted(task_sse)}
    if not per_task_mse or not all(math.isfinite(value) for value in per_task_mse.values()):
        raise RuntimeError(f"validation produced invalid per-task MSE: {per_task_mse}")
    macro = float(np.mean(list(per_task_mse.values())))
    micro = float(sum(task_sse.values()) / sum(task_count.values()))
    return {
        "macro_mse": macro,
        "micro_mse": micro,
        "per_task_mse": per_task_mse,
        "per_task_elements": dict(sorted(task_count.items())),
    }


def _rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def resume_signature(args: argparse.Namespace, tasks: Sequence[str]) -> Dict[str, Any]:
    return {
        "tasks": list(tasks),
        "temporal_window": args.temporal_window,
        "window_stride": args.window_stride,
        "val_window_stride": args.val_window_stride,
        "latent_channels": 5,
        "inr_hidden": args.inr_hidden,
        "kl_weight": args.kl_weight,
        "batch_size": args.batch_size,
        "steps_per_epoch": args.steps_per_epoch,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "patience": args.patience,
        "seed": args.seed,
    }


def build_checkpoint(
    model: TactileVAE,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_macro_mse: float,
    patience_remaining: int,
    history: Sequence[Mapping[str, Any]],
    manifest_hash: str,
    norm_stats: Mapping[str, Any],
    signature: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "completed_epoch": epoch,
        "best_macro_mse": best_macro_mse,
        "patience_remaining": patience_remaining,
        "history": list(history),
        "rng_state": _rng_state(),
        "manifest_hash": manifest_hash,
        "norm_stats": dict(norm_stats),
        "signature": dict(signature),
    }


def train(args: argparse.Namespace) -> Dict[str, Any]:
    tasks = tuple(task.strip().lower() for task in args.tasks.split(",") if task.strip())
    if tasks != DEFAULT_TASKS:
        raise ValueError(f"this entry point requires tasks={','.join(DEFAULT_TASKS)} in that order")
    if args.latent_channels != 5:
        raise ValueError("five-task TacVAE must use latent_channels=5 (45D per frame)")
    if args.temporal_window != 8:
        raise ValueError("five-task protocol requires temporal_window=8")
    if args.window_stride <= 0 or args.val_window_stride <= 0:
        raise ValueError("window_stride and val_window_stride must be positive")
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")

    set_seed(args.seed)
    records, manifest_hash = load_manifest(Path(args.manifest), tasks)
    markers = preload_markers(records, args.temporal_window)
    train_records = [record for record in records if record.split == "train"]
    norm_stats = compute_task_balanced_norm(train_records, markers, tasks)
    train_dataset = ManifestWindowDataset(
        records, markers, "train", args.temporal_window, args.window_stride,
        norm_stats["mean"], norm_stats["std"],
    )
    val_dataset = ManifestWindowDataset(
        records, markers, "val", args.temporal_window, args.val_window_stride,
        norm_stats["mean"], norm_stats["std"],
    )
    sampler = HierarchicalBalancedBatchSampler(
        train_dataset, tasks, args.batch_size, args.steps_per_epoch, args.seed
    )
    loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )

    device = torch.device(args.device)
    model = TactileVAE(
        latent_dim=5,
        temporal_window=8,
        num_freqs=4,
        inr_hidden=args.inr_hidden,
        kl_weight=args.kl_weight,
        direction_weight=0.0,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    signature = resume_signature(args, tasks)
    out_dir = Path(args.out_dir)
    latest_path = out_dir / "latest.pt"
    best_path = out_dir / "best_macro.pt"
    start_epoch = 1
    best_macro = float("inf")
    patience_remaining = args.patience
    history: List[Dict[str, Any]] = []

    if args.resume:
        if Path(args.resume).expanduser().resolve().parent != out_dir.expanduser().resolve():
            raise ValueError("resume checkpoint must be inside out_dir so best/latest remain a consistent pair")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint.get("manifest_hash") != manifest_hash:
            raise ValueError("resume checkpoint manifest hash does not match current manifest")
        if checkpoint.get("signature") != signature:
            raise ValueError(
                f"resume signature mismatch: checkpoint={checkpoint.get('signature')} current={signature}"
            )
        checkpoint_norm = checkpoint.get("norm_stats", {})
        if not np.allclose(checkpoint_norm.get("mean"), norm_stats["mean"], rtol=0, atol=1e-7) or not np.allclose(
            checkpoint_norm.get("std"), norm_stats["std"], rtol=0, atol=1e-7
        ):
            raise ValueError("resume checkpoint normalization does not match current training data")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["completed_epoch"]) + 1
        best_macro = float(checkpoint["best_macro_mse"])
        patience_remaining = int(checkpoint["patience_remaining"])
        history = list(checkpoint.get("history", []))
        _restore_rng_state(checkpoint["rng_state"])
        print(f"Resumed from {args.resume} at epoch {start_epoch}", flush=True)
    elif latest_path.exists() or best_path.exists():
        raise FileExistsError(
            f"scratch output already contains a checkpoint: {out_dir}; choose a new out_dir or use --resume"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_save(norm_stats, out_dir / "global_norm_stats.json")
    atomic_json_save(
        {
            "manifest": str(Path(args.manifest).resolve()),
            "manifest_hash": manifest_hash,
            "signature": signature,
            "train_episodes": len(train_dataset.records),
            "val_episodes": len(val_dataset.records),
            "train_windows": len(train_dataset),
            "val_windows": len(val_dataset),
        },
        out_dir / "run_config.json",
    )

    for epoch in range(start_epoch, args.epochs + 1):
        sampler.set_epoch(epoch)
        model.train()
        loss_sum = 0.0
        recon_sum = 0.0
        sample_count = 0
        for x, _task_names in loader:
            x = x.to(device)
            recon, mu, logvar = model(x)
            gt = TactileVAE.align_gt(x, temporal_stride=2)
            loss, recon_loss, _kl_loss, _direction_loss = model.loss(recon, gt, mu, logvar)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            batch_count = x.shape[0]
            loss_sum += float(loss.item()) * batch_count
            recon_sum += float(recon_loss.item()) * batch_count
            sample_count += batch_count

        val = evaluate(model, val_dataset, args.eval_batch_size, device, args.eval_num_workers)
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / sample_count,
            "train_recon": recon_sum / sample_count,
            **val,
        }
        if not math.isfinite(row["train_loss"]) or not math.isfinite(row["train_recon"]):
            raise RuntimeError(f"training produced non-finite metrics at epoch {epoch}: {row}")
        history.append(row)
        improved = val["macro_mse"] < best_macro
        if improved:
            best_macro = val["macro_mse"]
            patience_remaining = args.patience
        elif args.patience > 0:
            patience_remaining -= 1

        checkpoint = build_checkpoint(
            model, optimizer, epoch, best_macro, patience_remaining, history,
            manifest_hash, norm_stats, signature,
        )
        if improved:
            atomic_torch_save(checkpoint, best_path)
        atomic_torch_save(checkpoint, latest_path)
        atomic_json_save({"history": history, "best_macro_mse": best_macro}, out_dir / "metrics.json")
        print(
            f"epoch={epoch:03d} train={row['train_loss']:.6f} "
            f"val_macro={val['macro_mse']:.6f} val_micro={val['micro_mse']:.6f} "
            f"per_task={val['per_task_mse']}",
            flush=True,
        )
        if args.patience > 0 and patience_remaining <= 0:
            print(f"Early stopping at epoch {epoch}", flush=True)
            break

    return {
        "latest": str(latest_path),
        "best": str(best_path),
        "completed_epoch": history[-1]["epoch"] if history else start_epoch - 1,
        "best_macro_mse": best_macro,
        "manifest_hash": manifest_hash,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--out_dir", default="outputs/multitask_tacvae_5task_20260803/train")
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--temporal_window", type=int, default=8)
    parser.add_argument("--window_stride", type=int, default=2)
    parser.add_argument("--val_window_stride", type=int, default=8)
    parser.add_argument("--latent_channels", type=int, default=5)
    parser.add_argument("--inr_hidden", type=int, default=64)
    parser.add_argument("--kl_weight", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--eval_batch_size", type=int, default=512)
    parser.add_argument("--steps_per_epoch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", default="", help="Resume from a latest.pt checkpoint")
    return parser.parse_args()


if __name__ == "__main__":
    result = train(parse_args())
    print(json.dumps(result, indent=2), flush=True)
