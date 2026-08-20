import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from TFAC_V5.tac_quality_energy.tactile_only_latent import TactileOnlyLatentRuntime
from TFAC_V5.tac_quality_energy.train_tactile_only_latent import load_manifest, run


class TactileOnlyTrainerContractTest(unittest.TestCase):
    def _write(self, payload):
        directory = tempfile.TemporaryDirectory()
        path = Path(directory.name) / "manifest.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return directory, path

    def test_manifest_is_task_scoped_and_marker_only(self):
        payload = {
            "task": "socket",
            "class_names": ["expert", "jam"],
            "horizon": 4,
            "temporal_stride": 2,
            "future_offset": 1,
            "train_rows": [{"path": "a.hdf5", "start": 0, "label": 0}],
            "val_rows": [{"path": "b.hdf5", "start": 0, "label": 1}],
        }
        directory, path = self._write(payload)
        try:
            loaded = load_manifest(path)
            self.assertEqual(loaded["task"], "socket")
            self.assertEqual(loaded["marker_key"], "observations/tac/left/marker_offset")
        finally:
            directory.cleanup()

    def test_manifest_rejects_action_features(self):
        payload = {
            "task": "board",
            "class_names": ["expert", "bad"],
            "horizon": 4,
            "temporal_stride": 1,
            "future_offset": 1,
            "train_rows": [{"path": "a.hdf5", "start": 0, "label": 0, "action": [0]}],
            "val_rows": [{"path": "b.hdf5", "start": 0, "label": 1}],
        }
        directory, path = self._write(payload)
        try:
            with self.assertRaisesRegex(ValueError, "forbidden non-tactile"):
                load_manifest(path)
        finally:
            directory.cleanup()

    def test_one_step_training_saves_strict_task_checkpoint(self):
        payload = {
            "task": "card",
            "class_names": ["expert", "slip"],
            "horizon": 2,
            "temporal_stride": 1,
            "future_offset": 1,
            "train_rows": [
                {"path": "train-a.hdf5", "start": 0, "label": 0},
                {"path": "train-b.hdf5", "start": 0, "label": 1},
            ],
            "val_rows": [
                {"path": "val-a.hdf5", "start": 0, "label": 0},
                {"path": "val-b.hdf5", "start": 0, "label": 1},
            ],
        }
        directory, manifest = self._write(payload)
        try:
            output = Path(directory.name) / "card.pt"
            train_features = np.arange(2 * 2 * 18, dtype=np.float32).reshape(2, 2, 18)
            val_features = train_features[::-1].copy()
            args = SimpleNamespace(
                manifest=str(manifest), tactile_vae_ckpt="synthetic-vae.pt", output=str(output),
                device="cpu", epochs=1, batch_size=2, embed_dim=8, hidden=16,
                dropout=0.0, temperature=0.1, lr=1e-3, weight_decay=0.0,
            )
            vae_info = {
                "latent_dim": 2, "temporal_window": 8, "latent_flat_dim": 18,
                "mean": np.zeros(2), "std": np.ones(2),
            }
            with mock.patch(
                "TFAC_V5.tac_quality_energy.train_tactile_only_latent.load_tactile_vae_checkpoint",
                return_value=(object(), vae_info),
            ), mock.patch(
                "TFAC_V5.tac_quality_energy.train_tactile_only_latent.vae_checkpoint_identity",
                return_value="sha256:" + "a" * 64,
            ), mock.patch(
                "TFAC_V5.tac_quality_energy.train_tactile_only_latent._read_latents",
                side_effect=[
                    (train_features, np.asarray([0, 1], dtype=np.int64)),
                    (val_features, np.asarray([0, 1], dtype=np.int64)),
                ],
            ):
                result = run(args)
            runtime = TactileOnlyLatentRuntime(output, device="cpu", expected_task="card")
            self.assertEqual(result["task"], "card")
            self.assertEqual(runtime.horizon, 2)
            self.assertEqual(runtime.latent_shape, (18,))
        finally:
            directory.cleanup()


if __name__ == "__main__":
    unittest.main()
