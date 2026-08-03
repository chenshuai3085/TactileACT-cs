from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from TFAC_V5.train_multitask_tacvae_5task import (
    DEFAULT_TASKS,
    EpisodeRecord,
    HierarchicalBalancedBatchSampler,
    ManifestWindowDataset,
    compute_task_balanced_norm,
    evaluate,
    load_manifest,
    marker_digest,
    preload_markers,
)


def _write_episode(path: Path, left: np.ndarray, right: np.ndarray) -> None:
    with h5py.File(path, "w") as h5:
        h5.create_dataset("observations/tac/left/marker_offset", data=left)
        h5.create_dataset("observations/tac/right/marker_offset", data=right)


def _make_manifest(root: Path) -> Path:
    rows = []
    rng = np.random.default_rng(12)
    for task_index, task in enumerate(DEFAULT_TASKS):
        for split in ("train", "val"):
            left = np.ascontiguousarray(
                rng.normal(task_index, 0.1, size=(10, 9, 9, 2)).astype(np.float32)
            )
            right = np.ascontiguousarray((left * 0.25).astype(np.float32))
            path = root / f"{task}_{split}.hdf5"
            _write_episode(path, left, right)
            rows.append(
                {
                    "path": str(path),
                    "task": task,
                    "condition": f"{task}_condition",
                    "split": split,
                    "episode_id": f"{task}_{split}",
                    "source_episode_id": f"{task}_{split}",
                    "marker_sha256": marker_digest(left, right),
                }
            )
    manifest = root / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return manifest


class _EvalDataset(Dataset):
    def __init__(self) -> None:
        self.items = [(0.0, "board")] + [(2.0, "socket")] * 9

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        value, task = self.items[index]
        return torch.full((8, 9, 9, 2), value, dtype=torch.float32), task


class _ZeroTacVAE(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        shape = (x.shape[0], 4, 9, 9, 2)
        return torch.zeros(shape, dtype=x.dtype, device=x.device), None, None


class FiveTaskTacVAETrainingTests(unittest.TestCase):
    def test_manifest_hash_and_hierarchical_sampler_are_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records, _ = load_manifest(_make_manifest(root), DEFAULT_TASKS)
            markers = preload_markers(records, temporal_window=8)
            norm = compute_task_balanced_norm(
                [record for record in records if record.split == "train"], markers, DEFAULT_TASKS
            )
            dataset = ManifestWindowDataset(
                records, markers, "train", 8, 2, norm["mean"], norm["std"]
            )
            sampler = HierarchicalBalancedBatchSampler(dataset, DEFAULT_TASKS, 10, 4, seed=42)
            sampler.set_epoch(3)
            batches = list(sampler)

            for batch in batches:
                counts = {task: 0 for task in DEFAULT_TASKS}
                for item_index in batch:
                    record_index, _ = dataset.items[item_index]
                    counts[dataset.records[record_index].task] += 1
                self.assertEqual(counts, {task: 2 for task in DEFAULT_TASKS})

            repeated = HierarchicalBalancedBatchSampler(dataset, DEFAULT_TASKS, 10, 4, seed=42)
            repeated.set_epoch(3)
            self.assertEqual(batches, list(repeated))
            repeated.set_epoch(4)
            self.assertNotEqual(batches, list(repeated))

            first = records[0]
            with h5py.File(first.path, "r+") as h5:
                h5["observations/tac/right/marker_offset"][0, 0, 0, 0] += 1.0
            with self.assertRaisesRegex(ValueError, "marker_sha256 mismatch"):
                preload_markers(records, temporal_window=8)

    def test_norm_is_equal_task_condition_episode_weighted(self) -> None:
        records = []
        markers = {}

        def add(task: str, condition: str, episode: str, value: float, frames: int) -> None:
            records.append(EpisodeRecord(episode, task, condition, "train", episode, episode))
            markers[episode] = np.full((frames, 9, 9, 2), value, dtype=np.float32)

        add("board", "c1", "b0", 0.0, 8)
        add("board", "c1", "b2", 2.0, 80)
        add("board", "c2", "b10", 10.0, 16)
        for task, value in zip(DEFAULT_TASKS[1:], (1.0, 2.0, 3.0, 4.0)):
            add(task, "only", task, value, 8)

        norm = compute_task_balanced_norm(records, markers, DEFAULT_TASKS)
        expected_mean = (5.5 + 1.0 + 2.0 + 3.0 + 4.0) / 5.0
        expected_second = (51.0 + 1.0 + 4.0 + 9.0 + 16.0) / 5.0
        self.assertEqual(norm["weighting"], "equal_task_then_equal_condition_then_equal_episode")
        np.testing.assert_allclose(norm["mean"], [expected_mean, expected_mean])
        np.testing.assert_allclose(
            norm["std"], [math.sqrt(expected_second - expected_mean**2)] * 2, rtol=1e-6
        )
        np.testing.assert_allclose(norm["per_task"]["board"]["mean"], [5.5, 5.5])

    def test_macro_validation_is_not_sample_count_weighted(self) -> None:
        result = evaluate(
            _ZeroTacVAE(), _EvalDataset(), batch_size=4, device=torch.device("cpu")
        )
        self.assertEqual(result["per_task_mse"], {"board": 0.0, "socket": 4.0})
        self.assertAlmostEqual(result["macro_mse"], 2.0)
        self.assertAlmostEqual(result["micro_mse"], 3.6)


if __name__ == "__main__":
    unittest.main()
