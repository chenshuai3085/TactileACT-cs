import tempfile
import unittest
from pathlib import Path

import numpy as np

from TFAC_V5.plot_contact_quality_task_tsne import STATES, TASKS, contingency, load_archive


class ContactQualityTaskTsneTest(unittest.TestCase):
    def test_load_archive_and_contingency(self):
        n = len(STATES) * len(TASKS)
        labels = np.asarray([state for state in STATES for _ in TASKS])
        tasks = np.asarray(TASKS * len(STATES))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.npz"
            np.savez_compressed(
                path,
                latent=np.zeros((n, 45), dtype=np.float32),
                tsne=np.arange(n * 2, dtype=np.float32).reshape(n, 2),
                labels=labels,
                tasks=tasks,
                episodes=np.asarray([f"episode-{index}" for index in range(n)]),
            )
            coords, loaded_labels, loaded_tasks, episodes = load_archive(path)

        self.assertEqual(coords.shape, (n, 2))
        self.assertEqual(len(episodes), n)
        counts = contingency(loaded_labels, loaded_tasks)
        self.assertEqual({value for row in counts.values() for value in row.values()}, {1})

    def test_rejects_unknown_task(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.npz"
            np.savez_compressed(
                path,
                latent=np.zeros((1, 45), dtype=np.float32),
                tsne=np.zeros((1, 2), dtype=np.float32),
                labels=np.asarray([STATES[0]]),
                tasks=np.asarray(["unknown"]),
                episodes=np.asarray(["episode-0"]),
            )
            with self.assertRaisesRegex(ValueError, "Unknown labels"):
                load_archive(path)


if __name__ == "__main__":
    unittest.main()
