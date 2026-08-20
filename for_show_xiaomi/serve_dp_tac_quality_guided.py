"""Diffusion Policy server with TacQuality final-action classifier guidance.

This entrypoint is intentionally different from the older reranking servers:

  DP denoising -> clean action chunk -> TacQuality gradient refinement -> action

The TacQuality step is a bounded, accept-only trust-region update through:

  action_raw -> Foresight -> decoded tactile marker -> TacQuality score

It does not generate K candidates and it does not use cached/stale gradients.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_DIFFUSION = os.path.join(_ROOT, "diffusion")
if _DIFFUSION not in sys.path:
    sys.path.append(_DIFFUSION)

from network import ConditionalUnet1D  # noqa: E402
from utils import set_seed  # noqa: E402
from diffusion.train_dp_tac_concat import FrozenTactileVAEEncoder, OfficialVisionEncoder  # noqa: E402
from for_show_xiaomi.ws_server import ClientDisconnected, TactileACTServer  # noqa: E402
from for_show_xiaomi.server_rollout_logger import ServerRolloutLogger  # noqa: E402
from TFAC_V5.pretrain_latent_foresight import LatentForesightPretrainModel  # noqa: E402
from TFAC_V5.pretrain_latent_foresight_multistep import MultiStepLatentForesightModel  # noqa: E402
from TFAC_V5.tac_quality_energy.foresight_bridge import (  # noqa: E402
    ForesightBridgeConfig,
    ForesightTacQualityBridge,
    ForesightTactileOnlyLatentBridge,
    SyntheticLatentForesight,
)
from TFAC_V5.board_latent_energy.vae_utils import vae_checkpoint_identity  # noqa: E402
from TFAC_V5.tac_quality_energy.serving_guidance import (  # noqa: E402
    build_serving_guidance_from_arm,
    canonical_scorer_task,
    load_rollout_arm_config,
)
from TFAC_V5.tac_quality_energy.trust_region import summarize_tensor  # noqa: E402


_IMG_NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
DEFAULT_BOARD_ROLLOUT_LOG_DIR = (
    "/home/chenshuai/Project/output/board_force_rollouts/"
    "260617_only_marker_joint_s12_scorer"
)
DEFAULT_INSERTION_ROLLOUT_LOG_DIR = (
    "/home/chenshuai/Project/output/insertion_rollouts/"
    "good_margin_risk_scorer"
)


def freeze(module: torch.nn.Module) -> None:
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_pickle(path: str | Path) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def parse_shape(value: Any) -> tuple[int, int]:
    if value is None:
        raise ValueError("shape value is missing")
    if isinstance(value, str):
        return tuple(int(x) for x in value.split(","))  # type: ignore[return-value]
    return tuple(int(x) for x in value)  # type: ignore[return-value]


def camera_list(value: Any) -> List[str]:
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    return list(value)


def tensor_stats(x: torch.Tensor) -> Dict[str, float]:
    y = x.detach().float().flatten()
    return {
        "mean": float(y.mean().cpu()),
        "std": float(y.std(unbiased=False).cpu()),
        "min": float(y.min().cpu()),
        "max": float(y.max().cpu()),
    }


def predict_x0_from_eps(
    sample: torch.Tensor,
    eps: torch.Tensor,
    timestep: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    *,
    clip: bool = True,
) -> torch.Tensor:
    alpha = alphas_cumprod[timestep.to(alphas_cumprod.device).long()].to(sample.device, sample.dtype)
    while alpha.ndim < sample.ndim:
        alpha = alpha.view(*alpha.shape, 1)
    x0 = (sample - (1.0 - alpha).sqrt() * eps) / alpha.sqrt().clamp_min(1e-8)
    return x0.clamp(-1.0, 1.0) if clip else x0


def unit_guidance_update(
    grad: torch.Tensor,
    *,
    scale: float,
    min_grad_norm: float,
    max_delta_norm: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    finite_mask = torch.isfinite(grad).flatten(1).all(dim=1)
    safe_grad = torch.where(torch.isfinite(grad), grad, torch.zeros_like(grad))
    grad_norm = safe_grad.flatten(1).norm(dim=1)
    update = safe_grad / grad_norm.view(-1, *([1] * (safe_grad.ndim - 1))).clamp_min(min_grad_norm)
    update = update * float(scale)
    update = torch.where(finite_mask.view(-1, *([1] * (update.ndim - 1))), update, torch.zeros_like(update))
    update_norm = update.flatten(1).norm(dim=1)
    if max_delta_norm > 0:
        coef = (float(max_delta_norm) / update_norm.clamp_min(1e-8)).clamp(max=1.0)
        update = update * coef.view(-1, *([1] * (update.ndim - 1)))
        update_norm = update.flatten(1).norm(dim=1)
    positive_rate = (grad_norm > min_grad_norm).float().mean()
    return update, {
        "grad_norm": float(grad_norm.mean().detach().cpu()),
        "grad_norm_max": float(grad_norm.max().detach().cpu()),
        "finite_grad_rate": float(finite_mask.float().mean().detach().cpu()),
        "positive_grad_rate": float(positive_rate.detach().cpu()),
        "update_norm": float(update_norm.mean().detach().cpu()),
        "update_norm_max": float(update_norm.max().detach().cpu()),
    }


def summarize_contact_gate(metric: float, low: float, high: float) -> Dict[str, Any]:
    if not np.isfinite(metric):
        gate = 1.0
        reason = "metric_not_finite_enable_guidance"
    elif high <= low:
        gate = 1.0
        reason = "invalid_thresholds_enable_guidance"
    elif metric <= low:
        gate = 0.0
        reason = "below_low_threshold_skip_guidance"
    elif metric >= high:
        gate = 1.0
        reason = "above_high_threshold_full_guidance"
    else:
        gate = float((metric - low) / (high - low))
        reason = "between_thresholds_scaled_guidance"
    return {
        "contact_gate_enabled": True,
        "contact_gate_metric": float(metric),
        "contact_gate_low": float(low),
        "contact_gate_high": float(high),
        "contact_gate_value": float(np.clip(gate, 0.0, 1.0)),
        "contact_gate_reason": reason,
    }


def marker_stats_from_foresight_config(fs_config: Mapping[str, Any]) -> tuple[tuple[float, float], tuple[float, float]]:
    stats = fs_config.get("norm_stats", fs_config)
    mean = stats.get("marker_offset_mean", [-0.3398614525794983, -2.9208483695983887])
    std = stats.get("marker_offset_std", [1.9804853200912476, 2.767117738723755])
    if len(mean) != 2 or len(std) != 2:
        raise ValueError(f"marker_offset stats must be length-2, got mean={mean}, std={std}")
    return (float(mean[0]), float(mean[1])), (float(std[0]), float(std[1]))


def preprocess_image(raw_img: Any, resize_tf=None, crop_tf=None) -> torch.Tensor:
    img = np.asarray(raw_img, dtype=np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    t = torch.from_numpy(img).permute(2, 0, 1).float()
    t = _IMG_NORM(t)
    if resize_tf is not None:
        t = resize_tf(t)
    if crop_tf is not None:
        t = crop_tf(t)
    return t.unsqueeze(0)


def load_foresight_model(foresight_ckpt: str, foresight_dir: str, device: torch.device):
    fs_config = load_json(Path(foresight_dir) / "args.json")
    camera_names = fs_config.get("camera_names", ["global", "wrist", "gelsight"])
    predict_horizon = int(fs_config.get("predict_horizon", fs_config.get("foresight_horizon", 1)))
    common_kwargs = dict(
        camera_names=camera_names,
        cam_backbone_mapping={cam: 0 for cam in camera_names},
        hidden_dim=int(fs_config.get("hidden_dim", 512)),
        state_dim=int(fs_config.get("state_dim", 7)),
        foresight_layers=int(fs_config.get("foresight_layers", 3)),
        foresight_nheads=int(fs_config.get("foresight_nheads", 8)),
        foresight_dim_feedforward=int(fs_config.get("foresight_dim_feedforward", 2048)),
        dropout=float(fs_config.get("dropout", 0.1)),
        tactile_mode=fs_config.get("tactile_mode", "marker"),
        max_history=int(fs_config.get("max_history", 8)),
        predict_horizon=predict_horizon,
        tactile_vae_ckpt=fs_config.get("tactile_vae_ckpt"),
        tactile_vae_latent_dim=int(fs_config.get("tactile_vae_latent_dim", 16)),
    )
    if predict_horizon > 1:
        model = MultiStepLatentForesightModel(**common_kwargs).to(device)
        model_kind = "multistep"
    else:
        model = LatentForesightPretrainModel(
            **common_kwargs,
            use_delta_pred=bool(fs_config.get("use_delta_pred", False)),
            residual_prediction=bool(fs_config.get("residual_prediction", False)),
        ).to(device)
        model_kind = "single_step"
    state = torch.load(foresight_ckpt, map_location=device, weights_only=False)
    if isinstance(state, Mapping) and "model_state_dict" in state:
        state = state["model_state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(
        f"[tac-guided] foresight loaded: kind={model_kind}, "
        f"predict_horizon={predict_horizon}, missing={len(missing)}, unexpected={len(unexpected)}"
    )
    freeze(model)
    return model, fs_config


def load_foresight_norm_stats(foresight_dir: str, device: torch.device) -> Dict[str, torch.Tensor]:
    stats = load_pickle(Path(foresight_dir) / "dataset_stats.pkl")
    return {
        key: torch.tensor(stats[key], dtype=torch.float32, device=device)
        for key in ["action_mean", "action_std", "qpos_mean", "qpos_std"]
    }


class GuidedDPStack:
    """Owns DP/Foresight/scorer wiring used by both server and dry-run smoke."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device(
            f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
        )
        self.config = load_json(Path(args.ckpt_dir) / "config.json")
        self.variant = str(self.config.get("variant", ""))
        self.camera_names = camera_list(self.config["camera_names"])
        self.action_dim = int(self.config["action_dim"])
        self.pred_horizon = int(self.config["pred_horizon"])
        self.obs_horizon = int(self.config.get("obs_horizon", 2))
        self.tac_history = int(self.config.get("tac_history", 8))
        self.num_train_timesteps = int(self.config.get("num_train_timesteps", 100))
        self.num_inference_steps = int(args.num_inference_steps or self.config.get("num_inference_steps", 100))
        self.resize_tf = None
        self.crop_tf = None
        if self.config.get("resize_shape"):
            self.resize_tf = transforms.Resize(parse_shape(self.config["resize_shape"]))
        if self.config.get("crop_shape"):
            self.crop_tf = transforms.CenterCrop(parse_shape(self.config["crop_shape"]))

        ns = self.config["norm_stats"]
        self.action_min = torch.tensor(ns["action_min"], dtype=torch.float32, device=self.device)
        self.action_max = torch.tensor(ns["action_max"], dtype=torch.float32, device=self.device)
        self.qpos_min = torch.tensor(ns["qpos_min"], dtype=torch.float32, device=self.device)
        self.qpos_max = torch.tensor(ns["qpos_max"], dtype=torch.float32, device=self.device)

        self.guidance = None
        self.rollout_config = load_rollout_arm_config(Path(args.rollout_arm_config))
        if not args.disable_guidance:
            self.guidance = build_serving_guidance_from_arm(
                args.task,
                args.arm,
                dp_norm_stats=ns,
                rollout_config=self.rollout_config,
                device=str(self.device),
                norm_mode=args.dp_norm_mode,
            )
        self._load_dp()
        if self.uses_force_aware_guidance:
            self.foresight = None
            self.fs_config = {}
            self.fs_norm = {}
            self.fs_marker_mean = torch.zeros(1, 1, 1, 1, 2, dtype=torch.float32, device=self.device)
            self.fs_marker_std = torch.ones(1, 1, 1, 1, 2, dtype=torch.float32, device=self.device)
        else:
            self.foresight, self.fs_config = load_foresight_model(args.foresight_ckpt, args.foresight_dir, self.device)
            self.fs_norm = load_foresight_norm_stats(args.foresight_dir, self.device)
            fs_marker_mean, fs_marker_std = marker_stats_from_foresight_config(self.fs_config)
            self.fs_marker_mean = torch.tensor(fs_marker_mean, dtype=torch.float32, device=self.device).view(1, 1, 1, 1, 2)
            self.fs_marker_std = torch.tensor(fs_marker_std, dtype=torch.float32, device=self.device).view(1, 1, 1, 1, 2)
            self._validate_tactile_only_contract()
        self.noise_scheduler = self._build_scheduler(args.scheduler)

    def _validate_tactile_only_contract(self) -> None:
        if self.guidance is None or self.guidance.scorer_runtime != "TactileOnlyLatentRuntime":
            return
        scorer = self.guidance.adapter.scorer
        horizon = int(self.fs_config.get("predict_horizon", self.fs_config.get("foresight_horizon", 1)))
        latent_dim = int(self.fs_config.get("tactile_vae_latent_dim", 16)) * 3 * 3
        stride = int(self.fs_config.get("temporal_stride", 1))
        offset = int(self.fs_config.get("future_offset", 1))
        task_aliases = {
            "board_wiping": "board", "huaping_wiping": "vase", "vase_wiping": "vase",
            "card_swipe": "card", "card_swiping": "card", "chip_grasp": "chip",
            "socket_insertion": "socket", "insertion": "socket",
        }
        serving_task = canonical_scorer_task(self.args.task)
        foresight_task = task_aliases.get(str(self.fs_config.get("task", serving_task)).lower(), str(self.fs_config.get("task", serving_task)).lower())
        vae_path = self.fs_config.get("tactile_vae_ckpt")
        vae_identity = vae_checkpoint_identity(vae_path) if vae_path else None
        expected = {
            "task": serving_task,
            "horizon": horizon,
            "latent_shape": (latent_dim,),
            "temporal_stride": stride,
            "future_offset": offset,
            "vae_identity": vae_identity,
        }
        actual = {
            "task": scorer.task,
            "horizon": scorer.horizon,
            "latent_shape": scorer.latent_shape,
            "temporal_stride": scorer.temporal_stride,
            "future_offset": scorer.future_offset,
            "vae_identity": scorer.vae_identity,
        }
        if foresight_task != serving_task:
            raise ValueError(f"Foresight task {foresight_task!r} != serving task {serving_task!r}")
        mismatches = [f"{key}: expected {value!r}, got {actual[key]!r}" for key, value in expected.items() if value is not None and value != actual[key]]
        if mismatches:
            raise ValueError("tactile-only serving contract mismatch: " + "; ".join(mismatches))

    @property
    def uses_force_aware_guidance(self) -> bool:
        return bool(
            self.guidance is not None
            and getattr(self.guidance, "scorer_runtime", None) == "ForceAwareForesightGuidanceRuntime"
        )

    @property
    def uses_feature_cache_dp(self) -> bool:
        return self.variant.startswith("feature_cache")

    def _load_dp(self) -> None:
        down_dims = self.config.get("down_dims", [512, 1024, 2048])
        if isinstance(down_dims, str):
            down_dims = [int(x) for x in down_dims.split(",")]
        vae_checkpoint = self.args.vae_checkpoint_override or self.config.get("vae_checkpoint", "")
        self.vision_encoder = OfficialVisionEncoder(self.camera_names).to(self.device)
        self.tac_encoder = FrozenTactileVAEEncoder(
            vae_checkpoint,
            latent_dim=int(self.config.get("vae_latent_dim", 16)),
            temporal_window=self.tac_history,
        ).to(self.device)
        self.tac_vae_checkpoint = vae_checkpoint
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.action_dim,
            global_cond_dim=int(self.config["global_cond_dim"]),
            diffusion_step_embed_dim=int(self.config.get("diffusion_step_embed_dim", 128)),
            down_dims=down_dims,
            kernel_size=5,
        ).to(self.device)

        ckpt = torch.load(Path(self.args.ckpt_dir) / self.args.ckpt_name, map_location=self.device, weights_only=False)
        if "ema_net" in ckpt and not self.args.no_ema:
            self.noise_pred_net.load_state_dict(ckpt["ema_net"])
            self.dp_weight_source = "ema_net"
        else:
            self.noise_pred_net.load_state_dict(ckpt["noise_pred_net"])
            self.dp_weight_source = "noise_pred_net"

        if self.uses_feature_cache_dp:
            vision_ckpt = self.config.get("vision_ckpt")
            if not vision_ckpt:
                raise ValueError("feature-cache DP config must contain vision_ckpt for online serving")
            vision_state = torch.load(vision_ckpt, map_location=self.device, weights_only=False)
            if "ema_vis" in vision_state and not self.args.no_ema:
                self.vision_encoder.load_state_dict(vision_state["ema_vis"])
                self.dp_vision_source = f"{vision_ckpt}:ema_vis"
            elif "vision_encoder" in vision_state:
                self.vision_encoder.load_state_dict(vision_state["vision_encoder"])
                self.dp_vision_source = f"{vision_ckpt}:vision_encoder"
            else:
                raise KeyError(f"No ema_vis/vision_encoder in {vision_ckpt}")
        else:
            vis_sd = ckpt.get("ema_vis", ckpt.get("vision_encoder"))
            if vis_sd is None:
                raise KeyError("DP checkpoint has no ema_vis/vision_encoder")
            self.vision_encoder.load_state_dict(vis_sd)
            self.dp_vision_source = "ema_vis" if "ema_vis" in ckpt else "vision_encoder"

        freeze(self.vision_encoder)
        freeze(self.tac_encoder)
        freeze(self.noise_pred_net)
        print(
            f"[tac-guided] DP loaded: variant={self.variant}, "
            f"net={self.dp_weight_source}, vision={self.dp_vision_source}, "
            f"tactile_vae={self.tac_vae_checkpoint}"
        )

    def _build_scheduler(self, name: str):
        cls = DDIMScheduler if name == "ddim" else DDPMScheduler
        return cls(
            num_train_timesteps=self.num_train_timesteps,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

    def normalize_qpos(self, qpos_raw: torch.Tensor) -> torch.Tensor:
        return (qpos_raw - self.qpos_min) / (self.qpos_max - self.qpos_min + 1e-8) * 2 - 1

    def denormalize_action(self, action_norm: torch.Tensor) -> torch.Tensor:
        return (action_norm + 1) * 0.5 * (self.action_max.view(1, -1) - self.action_min.view(1, -1)) + self.action_min.view(1, -1)

    def build_obs_cond(self, obs_buffer: Sequence[Dict[str, Any]], marker_buffer: Sequence[np.ndarray]) -> torch.Tensor:
        feats = []
        for frame in obs_buffer:
            vf = self.vision_encoder(frame["images_dict"])
            marker_end = int(frame["_marker_idx"])
            frames = []
            for k in range(self.tac_history):
                idx = max(0, min(marker_end - self.tac_history + 1 + k, len(marker_buffer) - 1))
                frames.append(marker_buffer[idx])
            marker_seq = torch.tensor(np.stack(frames), dtype=torch.float32, device=self.device).unsqueeze(0)
            tf = self.tac_encoder(marker_seq)
            qpos = torch.tensor(frame["qpos_raw"], dtype=torch.float32, device=self.device).view(1, -1)
            feats.append(torch.cat([vf, tf, self.normalize_qpos(qpos)], dim=-1))
        return torch.cat(feats, dim=-1)

    @torch.no_grad()
    def ddpm_inference(self, obs_cond: torch.Tensor) -> torch.Tensor:
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        action = torch.randn((1, self.pred_horizon, self.action_dim), device=self.device)
        for t in self.noise_scheduler.timesteps:
            noise_pred = self.noise_pred_net(action, t.unsqueeze(0).to(self.device), global_cond=obs_cond)
            action = self.noise_scheduler.step(noise_pred, t, action).prev_sample
        return action

    def score_x0(self, x0_norm: torch.Tensor, bridge: ForesightTacQualityBridge) -> torch.Tensor:
        if self.guidance is None:
            raise RuntimeError("TacQuality guidance must be enabled before scoring x0")
        action_raw = self.guidance.adapter.action_normalizer.denormalize(x0_norm)
        tactile = bridge(action_raw)
        return self.guidance.adapter.score_from_prediction(tactile, action_raw)

    def score_force_aware_x0(
        self,
        x0_norm: torch.Tensor,
        processed: Mapping[str, Any],
        marker_buffer: Sequence[np.ndarray],
    ) -> torch.Tensor:
        if self.guidance is None or not self.uses_force_aware_guidance:
            raise RuntimeError("Force-aware TacQuality guidance must be enabled before scoring x0")
        action_raw = self.guidance.adapter.action_normalizer.denormalize(x0_norm)
        qpos = torch.tensor(processed["qpos_raw"], dtype=torch.float32, device=self.device).view(1, -1)
        marker_window = self.marker_window_tensor(marker_buffer)
        return self.guidance.adapter.runtime.forward_score(
            action_raw,
            qpos_raw=qpos,
            marker_window_raw=marker_window,
        )

    def _contact_gate_disabled_report(self, contact_gate: Mapping[str, Any]) -> Dict[str, Any]:
        gate = dict(contact_gate)
        report = {
            "task": self.args.task,
            "arm": self.args.arm,
            "adapter_policy": "contact_gate_skip_guidance",
            "scorer_runtime": None if self.guidance is None else getattr(self.guidance, "scorer_runtime", None),
            "guidance_disabled": self.guidance is None,
            "contact_gate_skipped": True,
            "reranking": False,
            "every_step_ddpm_guidance": self.args.guidance_location == "denoising_step",
            "returned_requires_grad": False,
            "raw_action_delta": {"n": 1, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0},
            "normalized_action_delta": {"n": 1, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0},
        }
        report.update(gate)
        return report

    def ddpm_inference_with_tac_guidance(
        self,
        obs_cond: torch.Tensor,
        bridge: ForesightTacQualityBridge,
        contact_gate: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        was_inference_mode = torch.is_inference_mode_enabled()
        if was_inference_mode:
            with torch.inference_mode(False):
                with torch.enable_grad():
                    action, report = self.ddpm_inference_with_tac_guidance(obs_cond.detach(), bridge, contact_gate)
            report["called_from_inference_mode"] = True
            return action.detach(), report

        if self.guidance is None:
            action = self.ddpm_inference(obs_cond)
            return action, {
                "guidance_disabled": True,
                "task": self.args.task,
                "arm": self.args.arm,
                "adapter_policy": "baseline_no_tac_quality_guidance",
                "reranking": False,
                "every_step_ddpm_guidance": False,
                "returned_requires_grad": False,
            }

        gate = dict(contact_gate or {})
        gate_value = float(np.clip(float(gate.get("contact_gate_value", 1.0)), 0.0, 1.0))
        if gate.get("contact_gate_enabled") and gate_value <= 0.0:
            action = self.ddpm_inference(obs_cond)
            return action.detach(), self._contact_gate_disabled_report(gate)

        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        timesteps = list(self.noise_scheduler.timesteps)
        alphas = self.noise_scheduler.alphas_cumprod.to(self.device)
        action = torch.randn((1, self.pred_horizon, self.action_dim), device=self.device)
        guide_start = max(0, len(timesteps) - int(self.args.ddpm_guidance_steps))
        logs: List[Dict[str, Any]] = []

        for step_idx, t in enumerate(timesteps):
            t_batch = t.reshape(1).to(self.device)
            with torch.no_grad():
                eps = self.noise_pred_net(action, t_batch, global_cond=obs_cond)
            do_guide = (
                self.args.ddpm_guidance_steps > 0
                and self.args.ddpm_guidance_scale > 0.0
                and gate_value > 0.0
                and step_idx >= guide_start
            )
            if do_guide:
                action_for_grad = action.detach().clone().requires_grad_(True)
                x0_before = predict_x0_from_eps(
                    action_for_grad,
                    eps.detach(),
                    t,
                    alphas,
                    clip=not self.args.disable_ddpm_x0_clip,
                )
                score_before = self.score_x0(x0_before, bridge)
                grad = torch.autograd.grad(score_before.mean(), action_for_grad, retain_graph=False)[0]
                update, grad_report = unit_guidance_update(
                    grad,
                    scale=float(self.args.ddpm_guidance_scale) * gate_value,
                    min_grad_norm=float(self.args.ddpm_min_grad_norm),
                    max_delta_norm=float(self.args.ddpm_max_delta_norm),
                )
                proposal = (action_for_grad.detach() + update).clamp(
                    -float(self.args.ddpm_sample_clip),
                    float(self.args.ddpm_sample_clip),
                )
                with torch.no_grad():
                    eps_after = self.noise_pred_net(proposal, t_batch, global_cond=obs_cond)
                    x0_after = predict_x0_from_eps(
                        proposal,
                        eps_after,
                        t,
                        alphas,
                        clip=not self.args.disable_ddpm_x0_clip,
                    )
                    score_after = self.score_x0(x0_after, bridge)
                    accepted = bool(
                        self.args.disable_ddpm_accept_only
                        or torch.all(score_after >= score_before.detach()).item()
                    )
                logs.append(
                    {
                        "step_idx": int(step_idx),
                        "timestep": int(t.item()),
                        "score_before": float(score_before.mean().detach().cpu()),
                        "score_after": float(score_after.mean().detach().cpu()),
                        "score_delta": float((score_after - score_before.detach()).mean().detach().cpu()),
                        "accepted": accepted,
                        "contact_gate_value": gate_value,
                        **grad_report,
                    }
                )
                if accepted:
                    action = proposal.detach()
                    eps = eps_after.detach()
            with torch.no_grad():
                action = self.noise_scheduler.step(eps, t, action).prev_sample.detach()

        final_score = self.score_x0(action.detach().clone().requires_grad_(True), bridge).detach()
        score_deltas = torch.tensor([row["score_delta"] for row in logs], dtype=torch.float32, device=self.device)
        action_delta_norm = torch.tensor([row["update_norm"] for row in logs], dtype=torch.float32, device=self.device)
        accept_values = torch.tensor([1.0 if row["accepted"] else 0.0 for row in logs], dtype=torch.float32, device=self.device)
        finite_values = torch.tensor([row["finite_grad_rate"] for row in logs], dtype=torch.float32, device=self.device)
        positive_values = torch.tensor([row["positive_grad_rate"] for row in logs], dtype=torch.float32, device=self.device)
        force_runtime = self.guidance.adapter.runtime
        report = {
            "task": self.args.task,
            "arm": self.args.arm,
            "adapter_policy": "denoising_step_tac_quality_guidance",
            "scorer_runtime": getattr(self.guidance, "scorer_runtime", None),
            "guidance_disabled": False,
            "contact_gate_skipped": False,
            "reranking": False,
            "every_step_ddpm_guidance": True,
            "returned_requires_grad": False,
            "score_mode": getattr(self.guidance.adapter, "score_mode", None),
            "guidance_location": "inside DP denoising loop on predicted clean action x0",
            "ddpm_guidance": {
                "scheduler": self.args.scheduler,
                "num_inference_steps": int(self.num_inference_steps),
                "guided_steps_requested": int(self.args.ddpm_guidance_steps),
                "guided_steps_executed": len(logs),
                "guidance_scale": float(self.args.ddpm_guidance_scale),
                "max_delta_norm": float(self.args.ddpm_max_delta_norm),
                "sample_clip": float(self.args.ddpm_sample_clip),
                "accept_only_improved": not bool(self.args.disable_ddpm_accept_only),
                "x0_clip": not bool(self.args.disable_ddpm_x0_clip),
            },
            "final_score": summarize_tensor(final_score),
            "score_delta": summarize_tensor(score_deltas),
            "accept_rate": float(accept_values.mean().detach().cpu()) if logs else 0.0,
            "finite_grad_rate": float(finite_values.mean().detach().cpu()) if logs else 0.0,
            "positive_grad_rate": float(positive_values.mean().detach().cpu()) if logs else 0.0,
            "raw_action_delta": summarize_tensor(action_delta_norm),
            "normalized_action_delta": summarize_tensor(action_delta_norm),
            "max_delta_within_trust_region": bool(
                action_delta_norm.max().item() <= float(self.args.ddpm_max_delta_norm) + 1e-6 if logs and self.args.ddpm_max_delta_norm > 0 else True
            ),
            "logs": logs,
        }
        report.update(gate)
        report["called_from_inference_mode"] = bool(was_inference_mode)
        return action.detach(), report

    def ddpm_inference_with_force_aware_tac_guidance(
        self,
        obs_cond: torch.Tensor,
        processed: Mapping[str, Any],
        marker_buffer: Sequence[np.ndarray],
        contact_gate: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        was_inference_mode = torch.is_inference_mode_enabled()
        if was_inference_mode:
            with torch.inference_mode(False):
                with torch.enable_grad():
                    action, report = self.ddpm_inference_with_force_aware_tac_guidance(
                        obs_cond.detach(),
                        processed,
                        marker_buffer,
                        contact_gate,
                    )
            report["called_from_inference_mode"] = True
            return action.detach(), report

        if self.guidance is None:
            action = self.ddpm_inference(obs_cond)
            return action, {
                "guidance_disabled": True,
                "task": self.args.task,
                "arm": self.args.arm,
                "adapter_policy": "baseline_no_tac_quality_guidance",
                "reranking": False,
                "every_step_ddpm_guidance": False,
                "returned_requires_grad": False,
            }
        if not self.uses_force_aware_guidance:
            raise RuntimeError("ddpm_inference_with_force_aware_tac_guidance requires force-aware guidance")

        gate = dict(contact_gate or {})
        gate_value = float(np.clip(float(gate.get("contact_gate_value", 1.0)), 0.0, 1.0))
        if gate.get("contact_gate_enabled") and gate_value <= 0.0:
            action = self.ddpm_inference(obs_cond)
            return action.detach(), self._contact_gate_disabled_report(gate)

        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        timesteps = list(self.noise_scheduler.timesteps)
        alphas = self.noise_scheduler.alphas_cumprod.to(self.device)
        action = torch.randn((1, self.pred_horizon, self.action_dim), device=self.device)
        guide_start = max(0, len(timesteps) - int(self.args.ddpm_guidance_steps))
        logs: List[Dict[str, Any]] = []

        for step_idx, t in enumerate(timesteps):
            t_batch = t.reshape(1).to(self.device)
            with torch.no_grad():
                eps = self.noise_pred_net(action, t_batch, global_cond=obs_cond)
            do_guide = (
                self.args.ddpm_guidance_steps > 0
                and self.args.ddpm_guidance_scale > 0.0
                and gate_value > 0.0
                and step_idx >= guide_start
            )
            if do_guide:
                action_for_grad = action.detach().clone().requires_grad_(True)
                x0_before = predict_x0_from_eps(
                    action_for_grad,
                    eps.detach(),
                    t,
                    alphas,
                    clip=not self.args.disable_ddpm_x0_clip,
                )
                score_before = self.score_force_aware_x0(x0_before, processed, marker_buffer)
                grad = torch.autograd.grad(score_before.mean(), action_for_grad, retain_graph=False)[0]
                update, grad_report = unit_guidance_update(
                    grad,
                    scale=float(self.args.ddpm_guidance_scale) * gate_value,
                    min_grad_norm=float(self.args.ddpm_min_grad_norm),
                    max_delta_norm=float(self.args.ddpm_max_delta_norm),
                )
                proposal = (action_for_grad.detach() + update).clamp(
                    -float(self.args.ddpm_sample_clip),
                    float(self.args.ddpm_sample_clip),
                )
                with torch.no_grad():
                    eps_after = self.noise_pred_net(proposal, t_batch, global_cond=obs_cond)
                    x0_after = predict_x0_from_eps(
                        proposal,
                        eps_after,
                        t,
                        alphas,
                        clip=not self.args.disable_ddpm_x0_clip,
                    )
                    score_after = self.score_force_aware_x0(x0_after, processed, marker_buffer)
                    accepted = bool(
                        self.args.disable_ddpm_accept_only
                        or torch.all(score_after >= score_before.detach()).item()
                    )
                logs.append(
                    {
                        "step_idx": int(step_idx),
                        "timestep": int(t.item()),
                        "score_before": float(score_before.mean().detach().cpu()),
                        "score_after": float(score_after.mean().detach().cpu()),
                        "score_delta": float((score_after - score_before.detach()).mean().detach().cpu()),
                        "accepted": accepted,
                        "contact_gate_value": gate_value,
                        **grad_report,
                    }
                )
                if accepted:
                    action = proposal.detach()
                    eps = eps_after.detach()
            with torch.no_grad():
                action = self.noise_scheduler.step(eps, t, action).prev_sample.detach()

        final_score = self.score_force_aware_x0(
            action.detach().clone().requires_grad_(True),
            processed,
            marker_buffer,
        ).detach()
        score_deltas = torch.tensor([row["score_delta"] for row in logs], dtype=torch.float32, device=self.device)
        action_delta_norm = torch.tensor([row["update_norm"] for row in logs], dtype=torch.float32, device=self.device)
        accept_values = torch.tensor([1.0 if row["accepted"] else 0.0 for row in logs], dtype=torch.float32, device=self.device)
        finite_values = torch.tensor([row["finite_grad_rate"] for row in logs], dtype=torch.float32, device=self.device)
        positive_values = torch.tensor([row["positive_grad_rate"] for row in logs], dtype=torch.float32, device=self.device)
        report = {
            "task": self.args.task,
            "arm": self.args.arm,
            "adapter_policy": "denoising_step_force_aware_tac_quality_guidance",
            "scorer_runtime": getattr(self.guidance, "scorer_runtime", None),
            "guidance_disabled": False,
            "contact_gate_skipped": False,
            "reranking": False,
            "every_step_ddpm_guidance": True,
            "returned_requires_grad": False,
            "score_mode": "force_aware_quality",
            "score_preset": getattr(force_runtime, "score_preset", None),
            "score_weights": getattr(force_runtime, "summary")().get("score_weights"),
            "guidance_location": "inside DP denoising loop on predicted clean action x0",
            "ddpm_guidance": {
                "scheduler": self.args.scheduler,
                "num_inference_steps": int(self.num_inference_steps),
                "guided_steps_requested": int(self.args.ddpm_guidance_steps),
                "guided_steps_executed": len(logs),
                "guidance_scale": float(self.args.ddpm_guidance_scale),
                "max_delta_norm": float(self.args.ddpm_max_delta_norm),
                "sample_clip": float(self.args.ddpm_sample_clip),
                "accept_only_improved": not bool(self.args.disable_ddpm_accept_only),
                "x0_clip": not bool(self.args.disable_ddpm_x0_clip),
            },
            "final_score": summarize_tensor(final_score),
            "score_delta": summarize_tensor(score_deltas),
            "accept_rate": float(accept_values.mean().detach().cpu()) if logs else 0.0,
            "finite_grad_rate": float(finite_values.mean().detach().cpu()) if logs else 0.0,
            "positive_grad_rate": float(positive_values.mean().detach().cpu()) if logs else 0.0,
            "raw_action_delta": summarize_tensor(action_delta_norm),
            "normalized_action_delta": summarize_tensor(action_delta_norm),
            "max_delta_within_trust_region": bool(
                action_delta_norm.max().item() <= float(self.args.ddpm_max_delta_norm) + 1e-6
                if logs and self.args.ddpm_max_delta_norm > 0
                else True
            ),
            "logs": logs,
        }
        report.update(gate)
        report["called_from_inference_mode"] = bool(was_inference_mode)
        return action.detach(), report

    def preprocess_obs(self, obs: Mapping[str, Any]) -> Dict[str, Any]:
        images_dict = {}
        images_fs = {}
        for cam in self.camera_names:
            if cam == "gelsight":
                continue
            raw = obs["images"][cam]
            images_dict[cam] = preprocess_image(raw, self.resize_tf, self.crop_tf).to(self.device)
            images_fs[cam] = preprocess_image(raw, None, None).to(self.device)
        tac = obs["tac"]
        side = list(tac.keys())[0]
        side_data = tac[side]
        marker = side_data["marker_offset"] if isinstance(side_data, Mapping) else side_data
        return {
            "images_dict": images_dict,
            "images_fs": images_fs,
            "qpos_raw": np.asarray(obs["qpos"], dtype=np.float32),
            "marker_offset": np.asarray(marker, dtype=np.float32),
        }

    def marker_window_tensor(self, marker_buffer: Sequence[np.ndarray]) -> torch.Tensor:
        frames = []
        for k in range(self.tac_history):
            idx = max(0, min(len(marker_buffer) - self.tac_history + k, len(marker_buffer) - 1))
            frames.append(marker_buffer[idx])
        return torch.tensor(np.stack(frames), dtype=torch.float32, device=self.device).unsqueeze(0)

    def marker_contact_metric(self, marker_buffer: Sequence[np.ndarray]) -> float:
        if not marker_buffer:
            return float("nan")
        window = self.marker_window_tensor(marker_buffer)
        mag = torch.linalg.norm(window.float(), dim=-1)
        return float(mag.mean().detach().cpu())

    def contact_gate_report(self, marker_buffer: Sequence[np.ndarray]) -> Dict[str, Any]:
        if self.args.disable_contact_gate or self.args.task != "board":
            return {
                "contact_gate_enabled": False,
                "contact_gate_value": 1.0,
                "contact_gate_reason": "disabled_or_non_board_task",
            }
        return summarize_contact_gate(
            self.marker_contact_metric(marker_buffer),
            self.args.contact_gate_low,
            self.args.contact_gate_high,
        )

    def normalize_foresight_marker_window(self, marker_window_raw: torch.Tensor) -> torch.Tensor:
        return (marker_window_raw - self.fs_marker_mean) / self.fs_marker_std.clamp_min(1e-8)

    def make_bridge(self, processed: Mapping[str, Any], marker_buffer: Sequence[np.ndarray]) -> ForesightTacQualityBridge:
        fs_camera_names = camera_list(self.fs_config.get("camera_names", ["global", "wrist", "gelsight"]))
        foresight_images = []
        for cam in fs_camera_names:
            if cam == "gelsight":
                continue
            if cam in processed["images_fs"]:
                foresight_images.append(processed["images_fs"][cam])
        qpos = torch.tensor(processed["qpos_raw"], dtype=torch.float32, device=self.device).view(1, -1)
        marker_window = self.normalize_foresight_marker_window(self.marker_window_tensor(marker_buffer))
        marker_mean, marker_std = marker_stats_from_foresight_config(self.fs_config)
        bridge_cls = (
            ForesightTactileOnlyLatentBridge
            if self.guidance is not None and self.guidance.scorer_runtime == "TactileOnlyLatentRuntime"
            else ForesightTacQualityBridge
        )
        return bridge_cls(
            self.foresight,
            self.fs_norm,
            qpos_raw=qpos,
            foresight_images=foresight_images,
            marker_window_norm=marker_window,
            config=ForesightBridgeConfig(
                task=self.args.task,
                window=self.tac_history,
                action_chunk=int(self.fs_config.get("chunk_size", min(10, self.pred_horizon))),
                latent_dim=int(self.fs_config.get("tactile_vae_latent_dim", 16)),
                marker_mean=marker_mean,
                marker_std=marker_std,
                residual_prediction=bool(self.fs_config.get("residual_prediction", False)),
            ),
        )

    def guide_chunk(
        self,
        action_norm: torch.Tensor,
        bridge: ForesightTacQualityBridge,
        contact_gate: Optional[Mapping[str, Any]] = None,
    ):
        if self.guidance is None:
            return action_norm.detach(), {
                "guidance_disabled": True,
                "task": self.args.task,
                "arm": self.args.arm,
                "adapter_policy": "baseline_no_tac_quality_guidance",
                "reranking": False,
                "every_step_ddpm_guidance": False,
            }
        gate = dict(contact_gate or {})
        gate_value = float(gate.get("contact_gate_value", 1.0))
        gate_value = float(np.clip(gate_value, 0.0, 1.0))
        if gate.get("contact_gate_enabled") and gate_value <= 0.0:
            report = {
                "task": self.args.task,
                "arm": self.args.arm,
                "adapter_policy": "contact_gate_skip_guidance",
                "scorer_runtime": getattr(self.guidance, "scorer_runtime", None),
                "guidance_disabled": False,
                "contact_gate_skipped": True,
                "reranking": False,
                "every_step_ddpm_guidance": False,
                "returned_requires_grad": False,
                "raw_action_delta": {"n": int(action_norm.shape[0]), "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0},
                "normalized_action_delta": {"n": int(action_norm.shape[0]), "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0},
            }
            report.update(gate)
            return action_norm.detach(), report

        guided_norm, report = self.guidance.guide_action_chunk(action_norm, bridge)
        if gate.get("contact_gate_enabled") and gate_value < 1.0:
            ungated_norm = guided_norm.detach()
            guided_norm = action_norm.detach() + gate_value * (ungated_norm - action_norm.detach())
            guided_raw = self.guidance.adapter.action_normalizer.denormalize(guided_norm)
            action_raw = self.guidance.adapter.action_normalizer.denormalize(action_norm.detach())
            report["contact_gate_scaled"] = True
            report["raw_action_delta_before_contact_gate"] = report.get("raw_action_delta")
            report["normalized_action_delta_before_contact_gate"] = report.get("normalized_action_delta")
            report["raw_action_delta"] = summarize_tensor((guided_raw - action_raw).flatten(1).norm(dim=1))
            report["normalized_action_delta"] = summarize_tensor((guided_norm - action_norm.detach()).flatten(1).norm(dim=1))
        else:
            report["contact_gate_scaled"] = False
        report["contact_gate_skipped"] = False
        report.update(gate)
        return guided_norm.detach(), report

    def guide_force_aware_chunk(
        self,
        action_norm: torch.Tensor,
        processed: Mapping[str, Any],
        marker_buffer: Sequence[np.ndarray],
        contact_gate: Optional[Mapping[str, Any]] = None,
    ):
        if self.guidance is None:
            return action_norm.detach(), {
                "guidance_disabled": True,
                "task": self.args.task,
                "arm": self.args.arm,
                "adapter_policy": "baseline_no_tac_quality_guidance",
                "reranking": False,
                "every_step_ddpm_guidance": False,
            }
        gate = dict(contact_gate or {})
        gate_value = float(np.clip(float(gate.get("contact_gate_value", 1.0)), 0.0, 1.0))
        if gate.get("contact_gate_enabled") and gate_value <= 0.0:
            report = {
                "task": self.args.task,
                "arm": self.args.arm,
                "adapter_policy": "contact_gate_skip_guidance",
                "scorer_runtime": getattr(self.guidance, "scorer_runtime", None),
                "guidance_disabled": False,
                "contact_gate_skipped": True,
                "reranking": False,
                "every_step_ddpm_guidance": False,
                "returned_requires_grad": False,
                "raw_action_delta": {"n": int(action_norm.shape[0]), "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0},
                "normalized_action_delta": {"n": int(action_norm.shape[0]), "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0},
            }
            report.update(gate)
            return action_norm.detach(), report

        qpos = torch.tensor(processed["qpos_raw"], dtype=torch.float32, device=self.device).view(1, -1)
        marker_window = self.marker_window_tensor(marker_buffer)
        guided_norm, report = self.guidance.guide_force_aware_action_chunk(
            action_norm,
            qpos_raw=qpos,
            marker_window_raw=marker_window,
        )
        if gate.get("contact_gate_enabled") and gate_value < 1.0:
            ungated_norm = guided_norm.detach()
            guided_norm = action_norm.detach() + gate_value * (ungated_norm - action_norm.detach())
            guided_raw = self.guidance.adapter.action_normalizer.denormalize(guided_norm)
            action_raw = self.guidance.adapter.action_normalizer.denormalize(action_norm.detach())
            report["contact_gate_scaled"] = True
            report["raw_action_delta_before_contact_gate"] = report.get("raw_action_delta")
            report["normalized_action_delta_before_contact_gate"] = report.get("normalized_action_delta")
            report["raw_action_delta"] = summarize_tensor((guided_raw - action_raw).flatten(1).norm(dim=1))
            report["normalized_action_delta"] = summarize_tensor((guided_norm - action_norm.detach()).flatten(1).norm(dim=1))
        else:
            report["contact_gate_scaled"] = False
        report["contact_gate_skipped"] = False
        report.update(gate)
        return guided_norm.detach(), report


def make_synthetic_obs(stack: GuidedDPStack, marker_value: float = 3.0) -> Dict[str, Any]:
    resize_h, resize_w = 240, 320
    obs = {
        "images": {
            cam: np.zeros((resize_h, resize_w, 3), dtype=np.uint8)
            for cam in stack.camera_names
            if cam != "gelsight"
        },
        "qpos": np.zeros((stack.action_dim,), dtype=np.float32),
        "tac": {"left": {"marker_offset": np.full((9, 9, 2), marker_value, dtype=np.float32)}},
    }
    return obs


def dry_run_guidance_smoke(args: argparse.Namespace) -> Dict[str, Any]:
    stack = GuidedDPStack(args)
    if args.synthetic_foresight_for_smoke:
        stack.foresight = SyntheticLatentForesight(action_dim=stack.action_dim).to(stack.device)
        freeze(stack.foresight)
        stack.fs_norm = {
            "action_mean": torch.zeros(stack.action_dim, device=stack.device),
            "action_std": torch.ones(stack.action_dim, device=stack.device),
            "qpos_mean": torch.zeros(stack.action_dim, device=stack.device),
            "qpos_std": torch.ones(stack.action_dim, device=stack.device),
        }
        stack.fs_config = {
            "camera_names": ["global", "wrist", "gelsight"],
            "chunk_size": min(10, stack.pred_horizon),
            "tactile_vae_latent_dim": 16,
            "norm_stats": {
                "marker_offset_mean": [0.0, 0.0],
                "marker_offset_std": [1.0, 1.0],
            },
        }
        stack.fs_marker_mean = torch.zeros(1, 1, 1, 1, 2, device=stack.device)
        stack.fs_marker_std = torch.ones(1, 1, 1, 1, 2, device=stack.device)

    obs_buffer: deque[Dict[str, Any]] = deque(maxlen=stack.obs_horizon)
    marker_buffer: List[np.ndarray] = []
    obs = make_synthetic_obs(stack, marker_value=float(args.smoke_marker_value))
    for i in range(stack.obs_horizon):
        processed = stack.preprocess_obs(obs)
        marker_buffer.append(processed["marker_offset"])
        processed["_marker_idx"] = i
        obs_buffer.append(processed)

    obs_cond = stack.build_obs_cond(list(obs_buffer), marker_buffer)
    contact_gate = stack.contact_gate_report(marker_buffer)
    if args.guidance_location == "denoising_step":
        if stack.uses_force_aware_guidance:
            with torch.inference_mode():
                guided_norm, report = stack.ddpm_inference_with_force_aware_tac_guidance(
                    obs_cond,
                    obs_buffer[-1],
                    marker_buffer,
                    contact_gate,
                )
        else:
            bridge = stack.make_bridge(obs_buffer[-1], marker_buffer)
            with torch.inference_mode():
                guided_norm, report = stack.ddpm_inference_with_tac_guidance(obs_cond, bridge, contact_gate)
        action_norm = guided_norm.detach()
    else:
        action_norm = torch.zeros((1, stack.pred_horizon, stack.action_dim), dtype=torch.float32, device=stack.device)
        with torch.inference_mode():
            if stack.uses_force_aware_guidance:
                guided_norm, report = stack.guide_force_aware_chunk(action_norm, obs_buffer[-1], marker_buffer, contact_gate)
            else:
                bridge = stack.make_bridge(obs_buffer[-1], marker_buffer)
                guided_norm, report = stack.guide_chunk(action_norm, bridge, contact_gate)
    if args.disable_guidance:
        smoke_pass = bool(
            report.get("guidance_disabled") is True
            and torch.isfinite(guided_norm).all().item()
            and not guided_norm.requires_grad
            and torch.allclose(guided_norm, action_norm)
        )
    else:
        if report.get("contact_gate_skipped") is True:
            smoke_pass = bool(
                torch.isfinite(guided_norm).all().item()
                and not guided_norm.requires_grad
                and torch.allclose(guided_norm, action_norm)
                and report.get("returned_requires_grad") is False
            )
        else:
            smoke_pass = bool(
                report.get("finite_grad_rate", 0.0) >= args.min_finite_grad_rate
                and report.get("positive_grad_rate", 0.0) >= args.min_positive_grad_rate
                and report.get("max_delta_within_trust_region") is True
                and (report.get("called_from_inference_mode") is True or args.guidance_location == "denoising_step")
                and report.get("returned_requires_grad") is False
                and torch.isfinite(guided_norm).all().item()
            )
    result = {
        "dry_run_guidance_smoke_pass": smoke_pass,
        "task": args.task,
        "arm": args.arm,
        "guidance_disabled": bool(args.disable_guidance),
        "device": str(stack.device),
        "variant": stack.variant,
        "obs_cond_shape": list(obs_cond.shape),
        "action_norm_shape": list(action_norm.shape),
        "guided_norm_shape": list(guided_norm.shape),
        "guided_norm_stats": tensor_stats(guided_norm),
        "contact_gate": contact_gate,
        "report": report,
        "not_reranking": True,
        "guidance_location": report.get("guidance_location", "after DP clean action chunk"),
    }
    if args.smoke_output:
        out = Path(args.smoke_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["dry_run_guidance_smoke_pass", "task", "arm", "device"]}, ensure_ascii=False, indent=2))
    return result


def default_rollout_log_dir(task: str) -> str:
    if task == "board":
        return DEFAULT_BOARD_ROLLOUT_LOG_DIR
    if task == "insertion":
        return DEFAULT_INSERTION_ROLLOUT_LOG_DIR
    raise ValueError(f"Unsupported task for rollout logging: {task}")


def resolve_rollout_log_dir(args: argparse.Namespace) -> str:
    if args.server_rollout_log_dir:
        return str(args.server_rollout_log_dir)
    return default_rollout_log_dir(args.task)


def extract_rollout_metadata(obs: Mapping[str, Any], *, task: str, arm: str) -> Dict[str, Any]:
    """Read optional client-provided manifest metadata from the first obs."""

    raw = obs.get("rollout_metadata")
    if not isinstance(raw, Mapping):
        return {}
    allowed_keys = {
        "pair_id",
        "manifest_trial_order",
        "manifest_task",
        "manifest_group",
        "manifest_server_arm",
        "manifest_server_port",
        "manifest_source_csv",
        "pair_id_source",
        "success",
        "stopped_early",
        "bounce_count",
        "retry_count",
        "notes",
    }
    out: Dict[str, Any] = {str(k): v for k, v in raw.items() if str(k) in allowed_keys and v is not None}
    checks = {
        "task_match": (not out.get("manifest_task")) or str(out.get("manifest_task")) == str(task),
        "arm_match": (not out.get("manifest_server_arm")) or str(out.get("manifest_server_arm")) == str(arm),
    }
    group = str(out.get("manifest_group") or "")
    if group:
        checks["group_valid"] = group in {"baseline", "guided"}
    out["rollout_metadata_source"] = "client_obs_rollout_metadata"
    out["rollout_metadata_checks"] = checks
    out["rollout_metadata_ok"] = all(bool(v) for v in checks.values())
    return out


def run_server(args: argparse.Namespace) -> None:
    stack = GuidedDPStack(args)
    action_skip = args.action_skip
    action_horizon = min(args.action_horizon, stack.pred_horizon - action_skip)
    query_freq = action_horizon
    rollout_log_dir = None if args.disable_server_rollout_log else resolve_rollout_log_dir(args)
    server_metadata = {
        "protocol": "dp_tac_quality_guided",
        "task": args.task,
        "arm": args.arm,
        "variant": stack.variant,
        "action_dim": stack.action_dim,
        "pred_horizon": stack.pred_horizon,
        "action_skip": action_skip,
        "action_horizon": action_horizon,
        "guidance": (
            "denoising_step_tac_quality_guidance"
            if args.guidance_location == "denoising_step"
            else "final_clean_action_trust_region_refinement"
        ),
        "guidance_location": args.guidance_location,
        "ddpm_guidance": {
            "enabled": args.guidance_location == "denoising_step" and not args.disable_guidance,
            "guided_steps": args.ddpm_guidance_steps,
            "guidance_scale": args.ddpm_guidance_scale,
            "max_delta_norm": args.ddpm_max_delta_norm,
            "sample_clip": args.ddpm_sample_clip,
            "accept_only_improved": not args.disable_ddpm_accept_only,
        },
        "reranking": False,
        "server_rollout_logging": not args.disable_server_rollout_log,
        "server_rollout_log_dir": rollout_log_dir,
        "contact_gate": {
            "enabled": (args.task == "board" and not args.disable_guidance and not args.disable_contact_gate),
            "low": args.contact_gate_low,
            "high": args.contact_gate_high,
            "metric": "mean_marker_magnitude_over_current_window",
        },
    }
    server = TactileACTServer(
        host=args.host,
        port=args.port,
        metadata=server_metadata,
    )
    server.start()
    print(f"[tac-guided] listening on {args.host}:{args.port}")

    try:
        ep = 0
        while True:
            try:
                print(f"\n[tac-guided] === episode {ep} ===")
                obs = server.recv_obs()
            except ClientDisconnected:
                print("[tac-guided] client gone before first obs, waiting...")
                continue

            obs_buffer: deque[Dict[str, Any]] = deque(maxlen=stack.obs_horizon)
            marker_buffer: List[np.ndarray] = []
            marker_step = 0
            action_chunk = None
            last_report = None
            rollout_logger = None
            if not args.disable_server_rollout_log:
                rollout_logger = ServerRolloutLogger(
                    rollout_log_dir,
                    episode=ep,
                    host=args.host,
                    port=args.port,
                    task=args.task,
                    arm=args.arm,
                    server_metadata=server_metadata,
                )
                rollout_metadata = extract_rollout_metadata(obs, task=args.task, arm=args.arm)
                if rollout_metadata:
                    rollout_logger.metadata.update(rollout_metadata)
                print(f"[tac-guided] server rollout log: {rollout_logger.trial_dir}")
            final_step = 0
            stop_reason = "max_timesteps"

            with torch.inference_mode():
                try:
                    for step in range(args.max_timesteps):
                        final_step = step
                        processed = stack.preprocess_obs(obs)
                        marker_buffer.append(processed["marker_offset"])
                        processed["_marker_idx"] = marker_step
                        marker_step += 1
                        obs_buffer.append(processed)
                        while len(obs_buffer) < stack.obs_horizon:
                            pad = dict(processed)
                            pad["_marker_idx"] = 0
                            obs_buffer.appendleft(pad)

                        obs_cond = stack.build_obs_cond(list(obs_buffer), marker_buffer)
                        if step % query_freq == 0 or action_chunk is None:
                            contact_gate = stack.contact_gate_report(marker_buffer)
                            if args.guidance_location == "denoising_step":
                                if stack.uses_force_aware_guidance:
                                    guided_actions, last_report = stack.ddpm_inference_with_force_aware_tac_guidance(
                                        obs_cond,
                                        processed,
                                        marker_buffer,
                                        contact_gate,
                                    )
                                else:
                                    bridge = stack.make_bridge(processed, marker_buffer)
                                    guided_actions, last_report = stack.ddpm_inference_with_tac_guidance(obs_cond, bridge, contact_gate)
                            else:
                                base_actions = stack.ddpm_inference(obs_cond)
                                if stack.uses_force_aware_guidance:
                                    guided_actions, last_report = stack.guide_force_aware_chunk(base_actions, processed, marker_buffer, contact_gate)
                                else:
                                    bridge = stack.make_bridge(processed, marker_buffer)
                                    guided_actions, last_report = stack.guide_chunk(base_actions, bridge, contact_gate)
                            action_chunk = guided_actions
                            if step % max(1, query_freq * 5) == 0:
                                print(
                                    f"  step {step}: score_delta={last_report.get('score_delta')}, "
                                    f"accept={last_report.get('accept_rate')}, "
                                    f"contact_gate={last_report.get('contact_gate_value')}"
                                )

                        raw_norm = action_chunk[:, action_skip + step % query_freq]
                        action = stack.denormalize_action(raw_norm).squeeze(0).detach().cpu().numpy().astype(np.float32)
                        msg = {"actions": action[None, :], "step": step}
                        if last_report is not None and args.send_guidance_report:
                            msg["guidance_report"] = last_report
                        if rollout_logger is not None:
                            rollout_logger.record(
                                step=step,
                                obs=obs,
                                action=action,
                                action_norm=raw_norm.squeeze(0).detach().cpu().numpy().astype(np.float32),
                                guidance_report=last_report,
                            )
                        server.send_action(msg)
                        if step + 1 < args.max_timesteps:
                            obs = server.recv_obs()
                except ClientDisconnected:
                    print(f"[tac-guided] client disconnected at step {step}")
                    stop_reason = "client_disconnected"
                finally:
                    if rollout_logger is not None:
                        rollout_logger.finalize(steps=final_step + 1, stop_reason=stop_reason)
            ep += 1
    except KeyboardInterrupt:
        print("\n[tac-guided] shutting down")
    finally:
        server.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["insertion", "board"], required=True)
    parser.add_argument("--arm", default="default_guided")
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--ckpt_name", default="dp_final.pth")
    parser.add_argument("--vae_checkpoint_override", default=None,
                        help="Override DP config['vae_checkpoint'] when old absolute paths are missing on this machine.")
    parser.add_argument("--foresight_dir", required=True)
    parser.add_argument("--foresight_ckpt", required=True)
    parser.add_argument(
        "--rollout_arm_config",
        default=(
            "/home/chenshuai/Project/output/tac_quality_rollout_arm_configs/"
            "tac_quality_rollout_arm_configs_current_s12_good_margin_20260619.json"
        ),
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--scheduler", choices=["ddpm", "ddim"], default="ddim")
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--action_horizon", type=int, default=8)
    parser.add_argument("--action_skip", type=int, default=0)
    parser.add_argument("--max_timesteps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dp_norm_mode", choices=["minmax", "standard", "identity"], default="minmax")
    parser.add_argument("--no_ema", action="store_true")
    parser.add_argument("--send_guidance_report", action="store_true")
    parser.add_argument("--guidance_location", choices=["final_action", "denoising_step"], default="final_action",
                        help="final_action keeps the stable post-DP refinement path; denoising_step applies TacQuality gradients inside the DP sampling loop.")
    parser.add_argument("--ddpm_guidance_steps", type=int, default=1,
                        help="Number of final/low-noise denoising steps to guide when --guidance_location denoising_step.")
    parser.add_argument("--ddpm_guidance_scale", type=float, default=0.001,
                        help="Unit-gradient step scale in normalized action sample space for denoising-step guidance.")
    parser.add_argument("--ddpm_max_delta_norm", type=float, default=0.01,
                        help="Per-guided-step trust-region cap for denoising-step guidance.")
    parser.add_argument("--ddpm_sample_clip", type=float, default=1.0,
                        help="Clamp normalized diffusion samples during denoising-step guidance.")
    parser.add_argument("--ddpm_min_grad_norm", type=float, default=1e-8)
    parser.add_argument("--disable_ddpm_accept_only", action="store_true",
                        help="For ablations only: accept denoising-step TacQuality updates even if the step score decreases.")
    parser.add_argument("--disable_ddpm_x0_clip", action="store_true",
                        help="For ablations only: do not clip predicted clean x0 before TacQuality scoring.")
    parser.add_argument(
        "--server_rollout_log_dir",
        default=None,
        help=(
            "Root for server-side force_trace.csv logs. Defaults to the current "
            "task-specific TacQuality evaluation root."
        ),
    )
    parser.add_argument("--disable_server_rollout_log", action="store_true",
                        help="Disable server-side saving of each real rollout trajectory/force trace.")
    parser.add_argument("--disable_guidance", action="store_true",
                        help="Run the same DP serving stack without TacQuality refinement; useful for feature-cache baselines.")
    parser.add_argument("--disable_contact_gate", action="store_true",
                        help="Disable the board contact-phase gate and run TacQuality guidance on every queried chunk.")
    parser.add_argument("--contact_gate_low", type=float, default=1.8,
                        help="Board marker magnitude below this skips TacQuality guidance.")
    parser.add_argument("--contact_gate_high", type=float, default=2.3,
                        help="Board marker magnitude above this applies full TacQuality guidance; between low/high is linearly scaled.")
    parser.add_argument("--dry_run_guidance_smoke", action="store_true")
    parser.add_argument("--synthetic_foresight_for_smoke", action="store_true")
    parser.add_argument("--smoke_marker_value", type=float, default=3.0)
    parser.add_argument("--smoke_output", default="/home/chenshuai/Project/output/tac_quality_guided_server_packet/auto_discovered/guided_server_dry_run_smoke.json")
    parser.add_argument("--min_finite_grad_rate", type=float, default=0.999)
    parser.add_argument("--min_positive_grad_rate", type=float, default=0.999)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.dry_run_guidance_smoke:
        dry_run_guidance_smoke(args)
    else:
        run_server(args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
