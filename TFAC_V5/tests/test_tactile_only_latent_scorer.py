import inspect
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
import torch

from TFAC_V5.board_latent_energy.dataset import (
    BoardLatentChunkDataset,
    ChunkIndexRow,
    DEFAULT_MARKER_KEY,
)

from TFAC_V5.tac_quality_energy.foresight_bridge import (
    ForesightBridgeConfig,
    ForesightTactileOnlyLatentBridge,
    SyntheticLatentForesight,
)
from TFAC_V5.tac_quality_energy.serving_guidance import (
    ActionNormalizer,
    TactileOnlyLatentGuidanceAdapter,
    build_serving_guidance_from_arm,
)
from TFAC_V5.tac_quality_energy.tactile_only_latent import (
    FORMAL_TASKS,
    TACTILE_ONLY_INPUT_MODE,
    TACTILE_ONLY_SCHEMA_VERSION,
    TactileOnlyLatentRuntime,
    TactileOnlyLatentScorer,
    build_tactile_only_checkpoint,
)
from TFAC_V5.tac_quality_energy.trust_region import TrustRegionConfig


CLASS_NAMES = ["expert", "low_quality", "high_risk", "unstable"]


def _make_checkpoint(task: str, *, chunk_len: int = 4, latent_dim: int = 18):
    torch.manual_seed(7)
    model = TactileOnlyLatentScorer(
        chunk_len=chunk_len,
        latent_dim=latent_dim,
        embed_dim=12,
        hidden=24,
        dropout=0.0,
        num_classes=len(CLASS_NAMES),
    )
    return build_tactile_only_checkpoint(
        model,
        task=task,
        latent_mean=torch.zeros(latent_dim),
        latent_std=torch.ones(latent_dim),
        class_names=CLASS_NAMES,
        temporal_stride=2,
        future_offset=1,
        vae_identity=f"sha256:{task:0<64}"[:71],
    )


class TactileOnlyLatentScorerTest(unittest.TestCase):
    def test_board_dataset_does_not_require_action_hdf5(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "marker_only.hdf5"
            with h5py.File(path, "w") as hdf5_file:
                hdf5_file.create_dataset(
                    DEFAULT_MARKER_KEY,
                    data=np.zeros((8, 9, 9, 2), dtype=np.float32),
                )
            row = ChunkIndexRow(
                path=str(path),
                label=0,
                class_name="expert",
                start=0,
                length=4,
                episode_id="synthetic/marker_only",
            )
            encoded = np.arange(4 * 18, dtype=np.float32).reshape(4, 18)
            with mock.patch(
                "TFAC_V5.board_latent_energy.dataset.encode_marker_chunk_to_latents",
                return_value=encoded,
            ):
                dataset = BoardLatentChunkDataset(
                    [row],
                    {"latent_mean": np.zeros(18), "latent_std": np.ones(18)},
                    vae=object(),
                    vae_info={"latent_flat_dim": 18},
                    device=torch.device("cpu"),
                    preload=True,
                )
            item = dataset[0]
            self.assertEqual(set(item), {"latent", "label"})
            self.assertEqual(tuple(item["latent"].shape), (4, 18))

    def test_five_tasks_use_independent_strict_checkpoint_metadata(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = []
            for task in FORMAL_TASKS:
                checkpoint = _make_checkpoint(task)
                self.assertEqual(checkpoint["schema_version"], TACTILE_ONLY_SCHEMA_VERSION)
                self.assertEqual(checkpoint["input_mode"], TACTILE_ONLY_INPUT_MODE)
                self.assertEqual(checkpoint["task"], task)
                self.assertEqual(checkpoint["horizon"], 4)
                self.assertEqual(checkpoint["latent_shape"], [18])
                self.assertNotIn("action_mean", checkpoint["norm"])
                path = Path(tmp_dir) / f"{task}.pt"
                torch.save(checkpoint, path)
                paths.append(path)

                runtime = TactileOnlyLatentRuntime(
                    path,
                    device="cpu",
                    expected_task=task,
                    expected_horizon=4,
                    expected_latent_shape=[18],
                    expected_temporal_stride=2,
                    expected_future_offset=1,
                    expected_vae_identity=checkpoint["vae_identity"],
                )
                self.assertEqual(runtime.task, task)
                self.assertEqual(runtime.model.chunk_len, 4)
                self.assertEqual(runtime.model.latent_dim, 18)
            self.assertEqual(len(set(paths)), len(FORMAL_TASKS))

    def test_public_model_api_is_strictly_tactile_only(self):
        model_parameters = list(inspect.signature(TactileOnlyLatentScorer.forward).parameters)
        self.assertEqual(model_parameters, ["self", "latent_chunk"])
        parameter_names = [name for name, _ in TactileOnlyLatentScorer().named_parameters()]
        self.assertFalse(any("action" in name or "task" in name for name in parameter_names))

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "board.pt"
            torch.save(_make_checkpoint("board"), path)
            runtime = TactileOnlyLatentRuntime(path, device="cpu")
            forward_parameters = list(inspect.signature(runtime.forward).parameters)
            score_parameters = list(inspect.signature(runtime.score).parameters)
            self.assertEqual(forward_parameters, ["latent_chunk", "normalized"])
            self.assertEqual(score_parameters, ["latent_chunk", "mode", "normalized"])

            latent = torch.randn(2, 4, 18)
            first = runtime.score(latent)
            second = runtime.score(latent)
            torch.testing.assert_close(first, second)
            with self.assertRaises(TypeError):
                runtime.score(latent, action_chunk=torch.zeros(2, 4, 7))

    def test_runtime_accepts_spatial_latent_and_rejects_contract_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "card.pt"
            torch.save(_make_checkpoint("card"), path)
            runtime = TactileOnlyLatentRuntime(path, device="cpu")
            output = runtime(torch.randn(2, 4, 2, 3, 3))
            self.assertEqual(tuple(output["logits"].shape), (2, len(CLASS_NAMES)))

            with self.assertRaisesRegex(ValueError, "contract mismatch"):
                TactileOnlyLatentRuntime(path, device="cpu", expected_task="board")
            with self.assertRaisesRegex(ValueError, "contract mismatch"):
                TactileOnlyLatentRuntime(path, device="cpu", expected_horizon=16)
            with self.assertRaisesRegex(ValueError, "contract mismatch"):
                TactileOnlyLatentRuntime(path, device="cpu", expected_latent_shape=[144])
            with self.assertRaisesRegex(ValueError, "contract mismatch"):
                TactileOnlyLatentRuntime(path, device="cpu", expected_temporal_stride=3)
            with self.assertRaisesRegex(ValueError, "contract mismatch"):
                TactileOnlyLatentRuntime(path, device="cpu", expected_future_offset=0)
            with self.assertRaisesRegex(ValueError, "contract mismatch"):
                TactileOnlyLatentRuntime(path, device="cpu", expected_vae_identity="another-vae")

    def test_legacy_action_aware_checkpoint_is_rejected(self):
        legacy_checkpoint = {
            "model_config": {"chunk_len": 4, "action_dim": 7, "latent_dim": 18},
            "model_state_dict": {},
            "norm": {
                "action_mean": torch.zeros(7),
                "action_std": torch.ones(7),
                "latent_mean": torch.zeros(18),
                "latent_std": torch.ones(18),
            },
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "legacy.pt"
            torch.save(legacy_checkpoint, path)
            with self.assertRaisesRegex(ValueError, "legacy action-aware checkpoint"):
                TactileOnlyLatentRuntime(path, device="cpu")

    def test_action_gradient_reaches_scorer_only_through_foresight_latent(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "board.pt"
            torch.save(_make_checkpoint("board"), path)
            runtime = TactileOnlyLatentRuntime(path, device="cpu")

            foresight = SyntheticLatentForesight(action_dim=3, latent_dim=2, pred_steps=4)
            bridge = ForesightTactileOnlyLatentBridge(
                foresight,
                fs_norm={
                    "action_mean": torch.zeros(3),
                    "action_std": torch.ones(3),
                    "qpos_mean": torch.zeros(3),
                    "qpos_std": torch.ones(3),
                },
                qpos_raw=torch.zeros(3),
                foresight_images=[],
                config=ForesightBridgeConfig(action_chunk=4, window=4, latent_dim=2),
            )
            action = torch.randn(2, 4, 3, requires_grad=True)
            predicted_latent = bridge(action)
            self.assertEqual(tuple(predicted_latent.shape), (2, 4, 18))
            score = runtime.score(predicted_latent, mode="expert_margin")
            gradient = torch.autograd.grad(score.sum(), action)[0]
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_formal_serving_adapter_loads_runtime_and_passes_only_latent(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "board.pt"
            torch.save(_make_checkpoint("board"), path)
            rollout_config = {
                "tasks": {
                    "board": {
                        "formal": {
                            "guidance_enabled": True,
                            "scorer_runtime": "TactileOnlyLatentRuntime",
                            "checkpoint": {"path": str(path)},
                            "refiner": {
                                "score_mode": "expert_margin",
                                "refinement": {
                                    "refine_steps": 1,
                                    "action_step": 0.01,
                                    "max_total_delta": 0.02,
                                    "accept_only_improved": False,
                                },
                            },
                        }
                    }
                }
            }
            guidance = build_serving_guidance_from_arm(
                "board",
                "formal",
                dp_norm_stats={},
                rollout_config=rollout_config,
                device="cpu",
                norm_mode="identity",
            )
            self.assertIsInstance(guidance.adapter, TactileOnlyLatentGuidanceAdapter)
            self.assertEqual(guidance.adapter.scorer.task, "board")

            projection = torch.nn.Linear(3, 4 * 18, bias=False)

            def foresight_predict_fn(action_raw):
                return projection(action_raw.mean(dim=1)).view(action_raw.shape[0], 4, 18)

            guided, report = guidance.guide_action_chunk(torch.randn(2, 4, 3), foresight_predict_fn)
            self.assertEqual(tuple(guided.shape), (2, 4, 3))
            self.assertEqual(report["integration_contract"]["scorer_input"], "predicted tactile latent only")
            self.assertGreater(report["positive_grad_rate"], 0.0)

    def test_adapter_rejects_cross_task_scorer(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "vase.pt"
            torch.save(_make_checkpoint("vase"), path)
            runtime = TactileOnlyLatentRuntime(path, device="cpu")
            with self.assertRaisesRegex(ValueError, "does not match scorer task"):
                TactileOnlyLatentGuidanceAdapter(
                    "board",
                    scorer=runtime,
                    action_normalizer=ActionNormalizer(mode="identity"),
                    config=TrustRegionConfig(steps=1),
                )

    def test_serving_aliases_insertion_to_socket_outside_scorer(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "socket.pt"
            torch.save(_make_checkpoint("socket"), path)
            rollout_config = {
                "tasks": {
                    "insertion": {
                        "formal": {
                            "guidance_enabled": True,
                            "scorer_runtime": "TactileOnlyLatentRuntime",
                            "checkpoint": {"path": str(path)},
                            "refiner": {"score_mode": "expert_margin", "refinement": {"refine_steps": 1}},
                        }
                    }
                }
            }
            guidance = build_serving_guidance_from_arm(
                "insertion", "formal", dp_norm_stats={}, rollout_config=rollout_config,
                device="cpu", norm_mode="identity",
            )
            self.assertEqual(guidance.adapter.task, "socket")
            self.assertEqual(guidance.adapter.scorer.task, "socket")

    def test_formal_serving_rejects_unmarked_action_aware_runtime(self):
        rollout_config = {
            "tasks": {
                "board": {
                    "formal": {
                        "guidance_enabled": True,
                        "scorer_runtime": "ForceBandTacQualityEnergyRuntime",
                        "checkpoint": {"path": "legacy.pt"},
                    }
                }
            }
        }
        with self.assertRaisesRegex(ValueError, "action-aware legacy code"):
            build_serving_guidance_from_arm(
                "board", "formal", dp_norm_stats={}, rollout_config=rollout_config,
                device="cpu", norm_mode="identity",
            )


if __name__ == "__main__":
    unittest.main()
