"""Reusable runtime for pi0/pi0.5 tactile foresight serving and offline rollout."""
from __future__ import annotations

import dataclasses
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch
import torchvision.transforms as T

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pi0_tactile.config import Pi0TactileConfig
from pi0_tactile.prompt import PromptTokenizer


CAMERA_ALIASES = {
    "base_0_rgb": "global",
    "left_wrist_0_rgb": "wrist",
    "right_wrist_0_rgb": "right_wrist",
}


def fit_stat_dim(arr: np.ndarray, dim: int, pad_value: float) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    if arr.shape[0] >= dim:
        return arr[:dim]
    pad = np.full(dim - arr.shape[0], float(pad_value), dtype=np.float32)
    return np.concatenate([arr, pad], axis=0)


def minmax_norm(x: np.ndarray, xmin: np.ndarray, xmax: np.ndarray) -> np.ndarray:
    return (x - xmin) / (xmax - xmin + 1e-8) * 2.0 - 1.0


def minmax_denorm_torch(x: torch.Tensor, xmin: torch.Tensor, xmax: torch.Tensor) -> torch.Tensor:
    return (x + 1.0) * 0.5 * (xmax - xmin) + xmin


def preprocess_image(raw_img: Any, transform: T.Compose) -> torch.Tensor:
    img = np.asarray(raw_img)
    if img.ndim != 3:
        raise ValueError(f"Expected HWC image, got shape {img.shape}")
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).float()
    return transform(tensor)


def load_config_from_dir(ckpt_dir: str | os.PathLike[str]) -> Pi0TactileConfig:
    path = Path(ckpt_dir) / "config.json"
    if not path.exists():
        return Pi0TactileConfig()
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return Pi0TactileConfig(**{
        k: v for k, v in raw.items()
        if k in Pi0TactileConfig.__dataclass_fields__
    })


def build_scorer(runtime: str, checkpoint: str, device: torch.device) -> torch.nn.Module:
    runtime = runtime.lower()
    if runtime == "force_band":
        from TFAC_V5.tac_quality_energy.force_band_runtime import ForceBandTacQualityEnergyRuntime

        return ForceBandTacQualityEnergyRuntime(checkpoint, device=str(device)).to(device)
    if runtime == "distilled":
        from TFAC_V5.tac_quality_energy.runtime import DistilledTacQualityEnergyRuntime

        return DistilledTacQualityEnergyRuntime(checkpoint, device=str(device)).to(device)
    raise ValueError(f"Unsupported TacQuality scorer runtime: {runtime}")


class Pi0TactileRuntime:
    """Online/offline pi0 tactile action chunk generator."""

    def __init__(
        self,
        config: Pi0TactileConfig,
        *,
        device: torch.device,
        ckpt_path: str = "",
        pi0_weights: str = "",
        stats_path: str = "",
        flow_guidance_scorer_ckpt: str = "",
        flow_guidance_scorer_runtime: str = "force_band",
        flow_guidance_score_mode: str = "energy_clipped",
        flow_guidance_score_window: int = 8,
        flow_guidance_task_id: Optional[int] = None,
    ):
        from pi0_tactile.model import Pi0Tactile

        self.config = config
        self.device = device
        self.flow_guidance_score_mode = flow_guidance_score_mode
        self.flow_guidance_score_window = int(flow_guidance_score_window)
        self.flow_guidance_task_id = flow_guidance_task_id
        self.marker_buffer: list[np.ndarray] = []

        self.model = Pi0Tactile(config).to(device)
        if pi0_weights:
            self.model.load_pi0_weights(pi0_weights)
        if ckpt_path and os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            state = ckpt.get("model_state_dict", ckpt)
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            print(
                f"[pi0-runtime] loaded checkpoint {ckpt_path} "
                f"missing={len(missing)} unexpected={len(unexpected)} "
                f"step={ckpt.get('step', '?') if isinstance(ckpt, dict) else '?'}"
            )
        self.model.eval()

        self.norm_stats = self._load_norm_stats(stats_path)
        self.action_min = fit_stat_dim(self.norm_stats["action_min"], config.robot_action_dim, 0.0)
        self.action_max = fit_stat_dim(self.norm_stats["action_max"], config.robot_action_dim, 1.0)
        self.qpos_min = fit_stat_dim(self.norm_stats["qpos_min"], config.robot_action_dim, 0.0)
        self.qpos_max = fit_stat_dim(self.norm_stats["qpos_max"], config.robot_action_dim, 1.0)
        self.action_min_t = torch.as_tensor(
            self.action_min, dtype=torch.float32, device=device).view(1, 1, -1)
        self.action_max_t = torch.as_tensor(
            self.action_max, dtype=torch.float32, device=device).view(1, 1, -1)

        self.marker_mean = np.asarray(config.marker_mean, dtype=np.float32)
        self.marker_std = np.asarray(config.marker_std, dtype=np.float32)
        self.img_transform = T.Compose([
            T.Resize(config.image_size),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.prompt_tokenizer = PromptTokenizer(
            max_token_len=config.max_token_len,
            pi05=config.pi05,
            backend=config.tokenizer_backend,
        )

        self.flow_scorer = None
        if self.guidance_requested and flow_guidance_scorer_ckpt:
            self.flow_scorer = build_scorer(
                flow_guidance_scorer_runtime,
                flow_guidance_scorer_ckpt,
                device,
            )
            self.flow_scorer.eval().requires_grad_(False)

    @classmethod
    def from_checkpoint_dir(
        cls,
        ckpt_dir: str,
        *,
        device: torch.device,
        pi0_weights: str = "",
        config_overrides: Optional[Mapping[str, Any]] = None,
        stats_path: str = "",
        **kwargs: Any,
    ) -> "Pi0TactileRuntime":
        config = load_config_from_dir(ckpt_dir)
        overrides = dict(config_overrides or {})
        for key, value in overrides.items():
            if value is not None and key in Pi0TactileConfig.__dataclass_fields__:
                setattr(config, key, value)
        if config.pi05 and config.action_dim == 7:
            config.action_dim = 32
        config = Pi0TactileConfig(**dataclasses.asdict(config))
        ckpt_path = os.path.join(ckpt_dir, "checkpoint.pth")
        return cls(
            config,
            device=device,
            ckpt_path=ckpt_path,
            pi0_weights=pi0_weights,
            stats_path=stats_path,
            **kwargs,
        )

    @property
    def guidance_requested(self) -> bool:
        return self.config.flow_guidance_steps > 0 and self.config.flow_guidance_scale > 0.0

    @property
    def guidance_enabled(self) -> bool:
        return self.guidance_requested and self.flow_scorer is not None

    def _load_norm_stats(self, stats_path: str) -> dict[str, np.ndarray]:
        candidates = []
        if stats_path:
            candidates.append(Path(stats_path))
        if self.config.dataset_dir:
            candidates.append(Path(self.config.dataset_dir) / "dataset_stats.pkl")
        for path in candidates:
            if path.exists():
                with open(path, "rb") as f:
                    stats = pickle.load(f)
                return {k: np.asarray(v, dtype=np.float32) for k, v in stats.items()}
        dim = self.config.robot_action_dim
        print("[pi0-runtime] WARNING: dataset_stats.pkl not found; using identity min-max stats")
        return {
            "action_min": np.zeros(dim, dtype=np.float32),
            "action_max": np.ones(dim, dtype=np.float32),
            "qpos_min": np.zeros(dim, dtype=np.float32),
            "qpos_max": np.ones(dim, dtype=np.float32),
        }

    def reset(self) -> None:
        self.marker_buffer.clear()

    def _extract_image(self, obs: Mapping[str, Any], camera_name: str) -> Any:
        images = obs.get("images", obs.get("image", {}))
        if not isinstance(images, Mapping):
            raise KeyError("Observation must contain an images/image mapping")
        candidates = [camera_name, CAMERA_ALIASES.get(camera_name, camera_name)]
        for candidate in candidates:
            if candidate in images:
                return images[candidate]
        raise KeyError(f"Missing image camera {camera_name}; tried {candidates}")

    def _extract_qpos(self, obs: Mapping[str, Any]) -> np.ndarray:
        for key in ("qpos", "state", "proprio", "proprio_joint"):
            if key in obs:
                qpos = np.asarray(obs[key], dtype=np.float32).reshape(-1)
                return qpos[: self.config.robot_action_dim]
        raise KeyError("Observation missing qpos/state/proprio")

    def _extract_marker(self, obs: Mapping[str, Any]) -> np.ndarray:
        if "marker_offset" in obs:
            return np.asarray(obs["marker_offset"], dtype=np.float32)
        tac = obs.get("tac")
        if isinstance(tac, Mapping):
            if "marker_offset" in tac:
                return np.asarray(tac["marker_offset"], dtype=np.float32)
            for value in tac.values():
                if isinstance(value, Mapping) and "marker_offset" in value:
                    return np.asarray(value["marker_offset"], dtype=np.float32)
                if not isinstance(value, Mapping):
                    return np.asarray(value, dtype=np.float32)
        raise KeyError("Observation missing tactile marker_offset")

    def _marker_window(self, marker_norm: np.ndarray) -> np.ndarray:
        self.marker_buffer.append(marker_norm)
        history = int(self.config.tac_history)
        if len(self.marker_buffer) >= history:
            return np.stack(self.marker_buffer[-history:], axis=0)
        pad = [self.marker_buffer[0]] * (history - len(self.marker_buffer))
        return np.stack(pad + self.marker_buffer, axis=0)

    def build_model_inputs(
        self,
        obs: Mapping[str, Any],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        images = [
            preprocess_image(self._extract_image(obs, cam), self.img_transform)
            .unsqueeze(0)
            .to(self.device)
            for cam in self.config.camera_names
        ]
        img_masks = [
            torch.ones(1, dtype=torch.bool, device=self.device)
            for _ in self.config.camera_names
        ]

        qpos_raw = self._extract_qpos(obs)
        qpos_norm = minmax_norm(qpos_raw, self.qpos_min, self.qpos_max)
        state = torch.as_tensor(qpos_norm, dtype=torch.float32, device=self.device).view(1, -1)
        prompt_state = qpos_norm
        if self.config.pi05 and self.config.action_dim > prompt_state.shape[-1]:
            prompt_state = np.concatenate([
                prompt_state,
                np.zeros(self.config.action_dim - prompt_state.shape[-1], dtype=np.float32),
            ])

        marker_raw = self._extract_marker(obs)
        marker_norm = (marker_raw - self.marker_mean) / (self.marker_std + 1e-8)
        marker_seq = self._marker_window(marker_norm)
        marker = torch.as_tensor(marker_seq, dtype=torch.float32, device=self.device).unsqueeze(0)

        tokens, mask = self.prompt_tokenizer.tokenize_torch(
            self.config.fixed_prompt,
            state=prompt_state if self.config.pi05 else None,
            device=self.device,
        )
        return images, img_masks, tokens.unsqueeze(0), mask.unsqueeze(0), state, marker

    def denormalize_robot_action(self, action_norm: torch.Tensor) -> torch.Tensor:
        return minmax_denorm_torch(action_norm, self.action_min_t, self.action_max_t)

    def _score_fn(self, marker: torch.Tensor, state: torch.Tensor) -> Optional[torch.nn.Module]:
        if not self.guidance_enabled:
            return None
        from pi0_tactile.score_bridge import Pi0TactileForesightScoreBridge

        return Pi0TactileForesightScoreBridge(
            self.model,
            marker_offset_norm=marker,
            qpos_norm=state,
            scorer=self.flow_scorer,
            action_to_scorer=self.denormalize_robot_action,
            marker_mean=tuple(float(x) for x in self.marker_mean),
            marker_std=tuple(float(x) for x in self.marker_std),
            score_mode=self.flow_guidance_score_mode,
            task_id=self.flow_guidance_task_id,
            score_window=self.flow_guidance_score_window,
        )

    def sample_action_chunk(
        self,
        obs: Mapping[str, Any],
        *,
        num_flow_steps: Optional[int] = None,
        return_guidance_report: bool = False,
    ) -> dict[str, Any]:
        images, img_masks, tokens, mask, state, marker = self.build_model_inputs(obs)
        flow_guidance = self.model.build_flow_guidance() if self.guidance_enabled else None
        score_fn = self._score_fn(marker, state)
        sample = self.model.sample_actions(
            images=images,
            img_masks=img_masks,
            lang_tokens=tokens,
            lang_masks=mask,
            state=state,
            marker_offset=marker,
            num_steps=int(num_flow_steps or self.config.num_flow_steps),
            flow_guidance=flow_guidance,
            score_fn=score_fn,
            return_guidance_report=return_guidance_report or self.guidance_enabled,
        )
        guidance_report = None
        if isinstance(sample, tuple):
            actions_model_norm, guidance_report = sample
        else:
            actions_model_norm = sample
        actions_robot_norm = self.model.action_adapter.slice_robot_action(actions_model_norm)
        actions_robot_raw = self.denormalize_robot_action(actions_robot_norm)
        return {
            "actions_model_norm": actions_model_norm,
            "actions_robot_norm": actions_robot_norm,
            "actions_robot_raw": actions_robot_raw,
            "guidance_report": guidance_report,
        }
