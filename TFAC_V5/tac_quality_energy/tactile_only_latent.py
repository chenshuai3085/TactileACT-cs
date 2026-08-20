"""Canonical tactile-only latent scorer and strict checkpoint runtime.

The scorer intentionally accepts one model input: a predicted tactile latent
chunk.  Task selection and action-conditioned Foresight stay outside the
scorer so action can influence a score only through its predicted tactile
consequence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


TACTILE_ONLY_SCHEMA_VERSION = 2
TACTILE_ONLY_INPUT_MODE = "tactile_only_latent"
FORMAL_TASKS = ("board", "vase", "card", "chip", "socket")
SCORE_MODES = (
    "score_good",
    "energy",
    "expert_margin",
    "margin_energy",
    "quality_0_100",
    "p_expert",
)

_REQUIRED_CHECKPOINT_FIELDS = (
    "schema_version",
    "input_mode",
    "task",
    "horizon",
    "latent_shape",
    "temporal_stride",
    "future_offset",
    "vae_identity",
    "model_config",
    "model_state_dict",
    "norm",
    "class_names",
)
_MODEL_CONFIG_FIELDS = {
    "chunk_len",
    "latent_dim",
    "embed_dim",
    "hidden",
    "dropout",
    "temperature",
    "num_classes",
}
_FORBIDDEN_MODEL_CONFIG_FIELDS = {
    "action_dim",
    "task_dim",
    "task_embed_dim",
    "eef_action_dim",
    "joint_action_dim",
}


def _flatten_latent_chunk(latent_chunk: torch.Tensor) -> torch.Tensor:
    if latent_chunk.dim() == 3:
        return latent_chunk
    if latent_chunk.dim() == 5:
        return latent_chunk.flatten(2)
    raise ValueError(
        "latent_chunk must have shape (B,H,Z) or (B,H,C,3,3); "
        f"got {tuple(latent_chunk.shape)}"
    )


class _TemporalLatentEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, latent_chunk: torch.Tensor) -> torch.Tensor:
        return self.net(latent_chunk.flatten(1))


class TactileOnlyLatentScorer(nn.Module):
    """Prototype quality scorer over only a tactile latent chunk."""

    def __init__(
        self,
        chunk_len: int = 16,
        latent_dim: int = 144,
        embed_dim: int = 128,
        hidden: int = 256,
        dropout: float = 0.10,
        temperature: float = 0.10,
        num_classes: int = 4,
    ):
        super().__init__()
        if chunk_len < 1 or latent_dim < 1 or embed_dim < 1 or hidden < 1:
            raise ValueError("chunk_len, latent_dim, embed_dim, and hidden must be positive")
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2 (expert plus a negative class)")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.chunk_len = int(chunk_len)
        self.latent_dim = int(latent_dim)
        self.embed_dim = int(embed_dim)
        self.temperature = float(temperature)
        self.num_classes = int(num_classes)
        self.latent_encoder = _TemporalLatentEncoder(
            self.chunk_len * self.latent_dim,
            int(hidden),
            self.embed_dim,
            float(dropout),
        )
        self.prototypes = nn.Parameter(torch.randn(self.num_classes, self.embed_dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(self.num_classes))

    def _validate_input(self, latent_chunk: torch.Tensor) -> torch.Tensor:
        latent = _flatten_latent_chunk(latent_chunk)
        if latent.shape[1] != self.chunk_len or latent.shape[2] != self.latent_dim:
            raise ValueError(
                "latent_chunk contract mismatch: expected "
                f"(B,{self.chunk_len},{self.latent_dim}), got {tuple(latent.shape)}"
            )
        return latent.float()

    def encode(self, latent_chunk: torch.Tensor) -> torch.Tensor:
        latent = self._validate_input(latent_chunk)
        return F.normalize(self.latent_encoder(latent), dim=-1)

    def forward(self, latent_chunk: torch.Tensor) -> Dict[str, torch.Tensor]:
        embedding = self.encode(latent_chunk)
        prototypes = F.normalize(self.prototypes, dim=-1)
        logits = embedding @ prototypes.T / self.temperature + self.bias
        score_good = logits[:, 0]
        negative_logsumexp = torch.logsumexp(logits[:, 1:], dim=-1)
        expert_margin = score_good - negative_logsumexp
        return {
            "embedding": embedding,
            "logits": logits,
            "score_good": score_good,
            "energy": -score_good,
            "expert_margin": expert_margin,
            "margin_energy": -expert_margin,
            "quality_0_100": torch.sigmoid(expert_margin) * 100.0,
            "prob": torch.softmax(logits, dim=-1),
        }


def build_tactile_only_checkpoint(
    model: TactileOnlyLatentScorer,
    *,
    task: str,
    latent_mean: torch.Tensor | Sequence[float],
    latent_std: torch.Tensor | Sequence[float],
    class_names: Sequence[str],
    temporal_stride: int,
    future_offset: int,
    vae_identity: str,
    model_config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the canonical schema-v2 payload used by every formal task."""

    if model_config is None:
        first_linear = model.latent_encoder.net[0]
        model_config = {
            "chunk_len": model.chunk_len,
            "latent_dim": model.latent_dim,
            "embed_dim": model.embed_dim,
            "hidden": first_linear.out_features,
            "dropout": float(model.latent_encoder.net[3].p),
            "temperature": model.temperature,
            "num_classes": model.num_classes,
        }
    payload: Dict[str, Any] = {
        "schema_version": TACTILE_ONLY_SCHEMA_VERSION,
        "input_mode": TACTILE_ONLY_INPUT_MODE,
        "task": task,
        "horizon": model.chunk_len,
        "latent_shape": [model.latent_dim],
        "temporal_stride": int(temporal_stride),
        "future_offset": int(future_offset),
        "vae_identity": vae_identity,
        "model_config": dict(model_config),
        "model_state_dict": model.state_dict(),
        "norm": {
            "latent_mean": torch.as_tensor(latent_mean, dtype=torch.float32).reshape(-1),
            "latent_std": torch.as_tensor(latent_std, dtype=torch.float32).reshape(-1),
        },
        "class_names": list(class_names),
    }
    _validate_checkpoint(payload)
    return payload


def _require_int(checkpoint: Mapping[str, Any], field: str, minimum: int) -> int:
    value = checkpoint[field]
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"checkpoint {field!r} must be an integer >= {minimum}, got {value!r}")
    return value


def _validate_checkpoint(checkpoint: Mapping[str, Any]) -> None:
    missing = [field for field in _REQUIRED_CHECKPOINT_FIELDS if field not in checkpoint]
    if missing:
        legacy_hint = " This looks like a legacy action-aware checkpoint." if "input_mode" in missing else ""
        raise ValueError(f"tactile-only checkpoint is missing required fields: {missing}.{legacy_hint}")
    if checkpoint["schema_version"] != TACTILE_ONLY_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported tactile-only schema_version={checkpoint['schema_version']!r}; "
            f"expected {TACTILE_ONLY_SCHEMA_VERSION}. Legacy checkpoints require a Legacy*Runtime."
        )
    if checkpoint["input_mode"] != TACTILE_ONLY_INPUT_MODE:
        raise ValueError(
            f"checkpoint input_mode={checkpoint['input_mode']!r} is not tactile-only; "
            f"expected {TACTILE_ONLY_INPUT_MODE!r}. Legacy checkpoints require a Legacy*Runtime."
        )

    task = checkpoint["task"]
    if task not in FORMAL_TASKS:
        raise ValueError(f"checkpoint task must be one of {FORMAL_TASKS}, got {task!r}")
    horizon = _require_int(checkpoint, "horizon", 1)
    temporal_stride = _require_int(checkpoint, "temporal_stride", 1)
    future_offset = _require_int(checkpoint, "future_offset", 0)
    del temporal_stride, future_offset
    if not isinstance(checkpoint["vae_identity"], str) or not checkpoint["vae_identity"].strip():
        raise ValueError("checkpoint vae_identity must be a non-empty string")

    model_config = checkpoint["model_config"]
    if not isinstance(model_config, Mapping):
        raise ValueError("checkpoint model_config must be a mapping")
    forbidden = sorted(_FORBIDDEN_MODEL_CONFIG_FIELDS.intersection(model_config))
    if forbidden:
        raise ValueError(f"tactile-only model_config contains forbidden action/task fields: {forbidden}")
    unknown = sorted(set(model_config).difference(_MODEL_CONFIG_FIELDS))
    if unknown:
        raise ValueError(f"tactile-only model_config contains unknown fields: {unknown}")
    required_model_fields = {"chunk_len", "latent_dim"}
    missing_model_fields = sorted(required_model_fields.difference(model_config))
    if missing_model_fields:
        raise ValueError(f"model_config is missing required fields: {missing_model_fields}")
    if int(model_config["chunk_len"]) != horizon:
        raise ValueError("checkpoint horizon must equal model_config.chunk_len")

    latent_shape = checkpoint["latent_shape"]
    expected_shape = [int(model_config["latent_dim"])]
    if not isinstance(latent_shape, (list, tuple)) or list(latent_shape) != expected_shape:
        raise ValueError(
            f"checkpoint latent_shape must equal model_config latent shape {expected_shape}, got {latent_shape!r}"
        )
    class_names = checkpoint["class_names"]
    num_classes = int(model_config.get("num_classes", 4))
    if not isinstance(class_names, (list, tuple)) or len(class_names) != num_classes:
        raise ValueError(f"checkpoint class_names must contain {num_classes} names")

    norm = checkpoint["norm"]
    if not isinstance(norm, Mapping) or set(norm) != {"latent_mean", "latent_std"}:
        raise ValueError("checkpoint norm must contain only latent_mean and latent_std")
    latent_dim = expected_shape[0]
    mean = torch.as_tensor(norm["latent_mean"])
    std = torch.as_tensor(norm["latent_std"])
    if mean.numel() != latent_dim or std.numel() != latent_dim:
        raise ValueError(f"latent normalization must contain {latent_dim} values")
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not (std > 0).all():
        raise ValueError("latent normalization must be finite and latent_std must be positive")


class TactileOnlyLatentRuntime(nn.Module):
    """Strict schema-v2 runtime whose public model input is tactile only."""

    schema_version = TACTILE_ONLY_SCHEMA_VERSION
    input_mode = TACTILE_ONLY_INPUT_MODE

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cuda:0",
        *,
        expected_task: Optional[str] = None,
        expected_horizon: Optional[int] = None,
        expected_latent_shape: Optional[Sequence[int]] = None,
        expected_temporal_stride: Optional[int] = None,
        expected_future_offset: Optional[int] = None,
        expected_vae_identity: Optional[str] = None,
    ):
        super().__init__()
        device_name = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self.device = torch.device(device_name)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if not isinstance(checkpoint, Mapping):
            raise ValueError("tactile-only checkpoint root must be a mapping")
        _validate_checkpoint(checkpoint)

        self.task = str(checkpoint["task"])
        self.temporal_stride = int(checkpoint["temporal_stride"])
        self.future_offset = int(checkpoint["future_offset"])
        self.vae_identity = str(checkpoint["vae_identity"])
        self.horizon = int(checkpoint["horizon"])
        self.latent_shape = tuple(int(value) for value in checkpoint["latent_shape"])
        self.class_names = tuple(str(name) for name in checkpoint["class_names"])
        self._check_expected_contract(
            expected_task=expected_task,
            expected_horizon=expected_horizon,
            expected_latent_shape=expected_latent_shape,
            expected_temporal_stride=expected_temporal_stride,
            expected_future_offset=expected_future_offset,
            expected_vae_identity=expected_vae_identity,
        )

        self.model = TactileOnlyLatentScorer(**dict(checkpoint["model_config"])).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        norm = checkpoint["norm"]
        self.register_buffer(
            "latent_mean",
            torch.as_tensor(norm["latent_mean"], dtype=torch.float32, device=self.device).view(1, 1, -1),
        )
        self.register_buffer(
            "latent_std",
            torch.as_tensor(norm["latent_std"], dtype=torch.float32, device=self.device).view(1, 1, -1),
        )

    @property
    def device_name(self) -> str:
        return str(self.device)

    def _check_expected_contract(
        self,
        *,
        expected_task: Optional[str],
        expected_horizon: Optional[int],
        expected_latent_shape: Optional[Sequence[int]],
        expected_temporal_stride: Optional[int],
        expected_future_offset: Optional[int],
        expected_vae_identity: Optional[str],
    ) -> None:
        expected = {
            "task": expected_task,
            "horizon": expected_horizon,
            "latent_shape": tuple(expected_latent_shape) if expected_latent_shape is not None else None,
            "temporal_stride": expected_temporal_stride,
            "future_offset": expected_future_offset,
            "vae_identity": expected_vae_identity,
        }
        actual = {
            "task": self.task,
            "horizon": self.horizon,
            "latent_shape": self.latent_shape,
            "temporal_stride": self.temporal_stride,
            "future_offset": self.future_offset,
            "vae_identity": self.vae_identity,
        }
        mismatches = [f"{key}: expected {value!r}, got {actual[key]!r}" for key, value in expected.items() if value is not None and value != actual[key]]
        if mismatches:
            raise ValueError("tactile-only scorer contract mismatch: " + "; ".join(mismatches))

    def normalize_latent(self, latent_chunk: torch.Tensor) -> torch.Tensor:
        latent = _flatten_latent_chunk(latent_chunk.to(self.device).float())
        if latent.shape[1] != self.model.chunk_len or latent.shape[2] != self.model.latent_dim:
            raise ValueError(
                "latent_chunk contract mismatch: expected "
                f"(B,{self.model.chunk_len},{self.model.latent_dim}), got {tuple(latent.shape)}"
            )
        return (latent - self.latent_mean) / self.latent_std.clamp_min(1e-6)

    def forward(self, latent_chunk: torch.Tensor, normalized: bool = False) -> Dict[str, torch.Tensor]:
        latent = _flatten_latent_chunk(latent_chunk.to(self.device).float()) if normalized else self.normalize_latent(latent_chunk)
        return self.model(latent)

    def score(
        self,
        latent_chunk: torch.Tensor,
        mode: str = "expert_margin",
        normalized: bool = False,
    ) -> torch.Tensor:
        output = self.forward(latent_chunk, normalized=normalized)
        if mode == "p_expert":
            return output["prob"][:, 0]
        if mode not in SCORE_MODES:
            raise ValueError(f"unknown tactile-only score mode {mode!r}; expected one of {SCORE_MODES}")
        return output[mode]
