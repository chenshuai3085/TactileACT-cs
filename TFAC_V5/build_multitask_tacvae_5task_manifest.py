#!/usr/bin/env python3
"""Build a validated, immutable episode manifest for five-task TacVAE training."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np


TASKS = ("board", "vase", "card", "chip", "socket")


def natural_key(path: Path) -> list[Any]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def canonical_json_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def marker_digest(left: np.ndarray, right: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(left.shape).encode("ascii"))
    digest.update(left.tobytes(order="C"))
    digest.update(right.tobytes(order="C"))
    return digest.hexdigest()


def validate_episode(path: Path, window: int) -> tuple[int, str, bool]:
    with h5py.File(path, "r") as h5:
        arrays = []
        for side in ("left", "right"):
            key = f"observations/tac/{side}/marker_offset"
            if key not in h5:
                raise ValueError(f"missing {key}")
            marker = h5[key][()]
            if marker.ndim != 4 or marker.shape[1:] != (9, 9, 2):
                raise ValueError(f"invalid {key} shape {marker.shape}")
            if marker.shape[0] < window:
                raise ValueError(f"short {key}: T={marker.shape[0]} < window={window}")
            if marker.dtype.kind not in "fc" or not np.isfinite(marker).all():
                raise ValueError(f"non-finite or non-float {key}: dtype={marker.dtype}")
            arrays.append(np.ascontiguousarray(marker.astype(np.float32, copy=False)))
        if arrays[0].shape[0] != arrays[1].shape[0]:
            raise ValueError(f"left/right length mismatch: {arrays[0].shape[0]} vs {arrays[1].shape[0]}")
        force_available = all(
            key in h5
            for key in ("ft", "observations/tac/left/force6d", "observations/tac/right/force6d")
        )
    return int(arrays[0].shape[0]), marker_digest(arrays[0], arrays[1]), force_available


def split_source(rows: list[dict[str, Any]], seed: int, source_id: str) -> None:
    if len(rows) < 3:
        raise ValueError(f"source {source_id} needs at least 3 valid episodes, got {len(rows)}")
    local_seed = int(hashlib.sha256(f"{seed}:{source_id}".encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(local_seed)
    order = rng.permutation(len(rows))
    n_test = max(1, int(round(0.10 * len(rows))))
    n_val = max(1, int(round(0.10 * len(rows))))
    if n_test + n_val >= len(rows):
        n_test = n_val = 1
    for rank, index in enumerate(order):
        split = "test" if rank < n_test else "val" if rank < n_test + n_val else "train"
        rows[int(index)]["split"] = split


def build_manifest(config: dict[str, Any], window: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    seen_hashes: dict[str, str] = {}
    invalid: list[dict[str, str]] = []
    for source in config["sources"]:
        source_id = source["source_id"]
        root = Path(source["root"])
        if not root.is_dir():
            raise FileNotFoundError(f"missing source root {source_id}: {root}")
        paths = sorted(root.glob(source.get("pattern", "episode_*.hdf5")), key=natural_key)
        excluded = set(source.get("exclude", []))
        paths = [path for path in paths if path.name not in excluded]
        if not paths:
            raise ValueError(f"source {source_id} has no files")
        source_rows = []
        for path in paths:
            resolved = str(path.resolve())
            if resolved in seen_paths:
                raise ValueError(f"duplicate canonical path: {resolved}")
            seen_paths.add(resolved)
            try:
                frames, digest, force_available = validate_episode(path, window)
            except (OSError, ValueError) as exc:
                invalid.append({"source_id": source_id, "path": resolved, "reason": str(exc)})
                continue
            if digest in seen_hashes:
                raise ValueError(f"duplicate marker trace: {resolved} == {seen_hashes[digest]}")
            seen_hashes[digest] = resolved
            item = {
                "episode_id": f"{source_id}/{path.stem}",
                "source_episode_id": f"{source_id}/{path.stem}",
                "source_id": source_id,
                "task": source["task"],
                "domain": source["domain"],
                "condition": source["condition"],
                "path": resolved,
                "frames": frames,
                "marker_sha256": digest,
                "force_available": force_available,
            }
            for key in ("connector", "outcome"):
                if key in source:
                    item[key] = source[key]
            source_rows.append(item)
        if not source_rows:
            raise ValueError(f"source {source_id} has no valid episodes")
        expected_episodes = source.get("expected_valid_episodes")
        expected_frames = source.get("expected_frames")
        if expected_episodes is not None and len(source_rows) != expected_episodes:
            raise ValueError(
                f"source {source_id} episode mismatch: {len(source_rows)} != {expected_episodes}; "
                f"invalid={invalid[-5:]}"
            )
        source_frames = sum(row["frames"] for row in source_rows)
        if expected_frames is not None and source_frames != expected_frames:
            raise ValueError(f"source {source_id} frame mismatch: {source_frames} != {expected_frames}")
        split_source(source_rows, seed, source_id)
        rows.extend(source_rows)

    task_counts = Counter(row["task"] for row in rows)
    unknown = sorted(set(task_counts) - set(TASKS))
    missing = sorted(set(TASKS) - set(task_counts))
    if unknown or missing:
        raise ValueError(f"task coverage mismatch: missing={missing}, unknown={unknown}")
    expected = config.get("expected", {})
    expected_totals = config.get("expected_totals", {})
    episodes_by_task = dict(expected.get("episodes_by_task", {}))
    frames_by_task_expected = dict(expected.get("frames_by_task", {}))
    for task, payload in expected_totals.get("by_task", {}).items():
        episodes_by_task[task] = payload["valid_episodes"]
        frames_by_task_expected[task] = payload["frames"]
    for task, expected_count in episodes_by_task.items():
        if task_counts[task] != expected_count:
            raise ValueError(f"{task} episode mismatch: {task_counts[task]} != {expected_count}")
    frames_by_task = Counter()
    for row in rows:
        frames_by_task[row["task"]] += row["frames"]
    for task, expected_frames in frames_by_task_expected.items():
        if frames_by_task[task] != expected_frames:
            raise ValueError(f"{task} frame mismatch: {frames_by_task[task]} != {expected_frames}")
    if expected_totals.get("valid_episodes") not in (None, len(rows)):
        raise ValueError(f"total episode mismatch: {len(rows)} != {expected_totals['valid_episodes']}")
    total_frames = int(sum(row["frames"] for row in rows))
    if expected_totals.get("frames") not in (None, total_frames):
        raise ValueError(f"total frame mismatch: {total_frames} != {expected_totals['frames']}")

    split_counts = defaultdict(Counter)
    for row in rows:
        split_counts[row["task"]][row["split"]] += 1
    summary = {
        "schema_version": 1,
        "seed": seed,
        "window": window,
        "config_sha256": canonical_json_hash(config),
        "episodes": len(rows),
        "frames": total_frames,
        "episodes_by_task": dict(sorted(task_counts.items())),
        "frames_by_task": dict(sorted(frames_by_task.items())),
        "split_counts_by_task": {task: dict(sorted(counts.items())) for task, counts in sorted(split_counts.items())},
        "invalid_excluded": invalid,
        "configured_excluded_sources": config.get(
            "excluded_sources", config.get("excluded_mirrors", [])
        ),
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("TFAC_V5/config_multitask_tacvae_5task_sources.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/multitask_tacvae_5task_20260803"))
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows, summary = build_manifest(config, args.window, args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.jsonl"
    manifest_text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    manifest_path.write_text(manifest_text, encoding="utf-8")
    summary["manifest_sha256"] = hashlib.sha256(manifest_text.encode()).hexdigest()
    (args.out_dir / "manifest_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
