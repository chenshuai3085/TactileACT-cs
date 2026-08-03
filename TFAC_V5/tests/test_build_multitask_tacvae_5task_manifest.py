import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from TFAC_V5.build_multitask_tacvae_5task_manifest import TASKS, build_manifest


def write_episode(path: Path, value: float) -> None:
    marker = np.full((10, 9, 9, 2), value, dtype=np.float32)
    with h5py.File(path, "w") as h5:
        h5.create_dataset("observations/tac/left/marker_offset", data=marker)
        h5.create_dataset("observations/tac/right/marker_offset", data=marker + 0.25)


def make_config(tmp_path: Path) -> dict:
    sources = []
    for task_index, task in enumerate(TASKS):
        root = tmp_path / task
        root.mkdir()
        for episode in range(3):
            write_episode(root / f"episode_{episode}.hdf5", task_index * 10 + episode)
        sources.append(
            {
                "source_id": f"{task}_source",
                "task": task,
                "domain": "synthetic",
                "condition": "synthetic",
                "root": str(root),
                "expected_valid_episodes": 3,
                "expected_frames": 30,
            }
        )
    return {"sources": sources}


class ManifestBuilderTest(unittest.TestCase):
    def test_build_manifest_is_five_task_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = make_config(Path(directory))
            rows_a, summary_a = build_manifest(config, window=8, seed=42)
            rows_b, summary_b = build_manifest(json.loads(json.dumps(config)), window=8, seed=42)

            self.assertEqual(rows_a, rows_b)
            self.assertEqual(summary_a, summary_b)
            self.assertEqual(summary_a["episodes"], 15)
            self.assertEqual(summary_a["frames"], 150)
            self.assertEqual(set(summary_a["episodes_by_task"]), set(TASKS))
            for task in TASKS:
                task_rows = [row for row in rows_a if row["task"] == task]
                self.assertEqual({row["split"] for row in task_rows}, {"train", "val", "test"})

    def test_build_manifest_rejects_duplicate_marker_trace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            config = make_config(tmp_path)
            duplicate = tmp_path / "vase" / "episode_2.hdf5"
            duplicate.unlink()
            original = tmp_path / "board" / "episode_1.hdf5"
            duplicate.write_bytes(original.read_bytes())

            with self.assertRaisesRegex(ValueError, "duplicate marker trace"):
                build_manifest(config, window=8, seed=42)


if __name__ == "__main__":
    unittest.main()
