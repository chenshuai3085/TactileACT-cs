"""Board DP server with Foresight latent-energy gradient guidance.

This is the online serving counterpart of
``TFAC_V5/eval_board_dp_foresight_guidance.py``:

    DP denoising x0 -> Foresight predicted tactile latent
    -> board latent energy scorer -> gradient wrt diffusion sample
    -> guided denoising -> action

The script is board-specific by design.  It keeps the original
``serve_dp_policy.py`` untouched and wires only the current board checkpoints.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import pickle
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import h5py
import numpy as np
import torch
import torchvision.transforms as transforms
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DIFFUSION_DIR = ROOT / "diffusion"
if str(DIFFUSION_DIR) not in sys.path:
    sys.path.insert(0, str(DIFFUSION_DIR))

from diffusion.network import ConditionalUnet1D  # noqa: E402
from diffusion.train_dp_tac_concat import FrozenTactileVAEEncoder, OfficialVisionEncoder  # noqa: E402
from for_show_xiaomi.server_rollout_logger import ServerRolloutLogger  # noqa: E402
from for_show_xiaomi.ws_server import ClientDisconnected, TactileACTServer  # noqa: E402
from TFAC_V5.board_latent_energy.runtime import BoardLatentEnergyRuntime  # noqa: E402
from TFAC_V5.board_latent_energy.vae_utils import vae_checkpoint_identity  # noqa: E402
from TFAC_V5.pretrain_latent_foresight_multistep import MultiStepLatentForesightModel  # noqa: E402
from TFAC_V5.tactile_vae_v2 import TactileVAEv2  # noqa: E402
from utils import set_seed  # noqa: E402


DEFAULT_DP_CKPT = (
    "/home/chenshuai/Project/output/"
    "dp_tac_concat_board_260609_260610_left_boardvae_rawimg200x266_ph16_oh2_e1000/"
    "dp_best.pth"
)
DEFAULT_FORESIGHT_DIR = (
    "/home/chenshuai/Project/output/foresight_ckpt/"
    "v2_board_action_conditioned_h16_e100_20260820"
)
DEFAULT_SCORER_CKPT = (
    "/home/chenshuai/Project/output/v2_intensity_rank_scorer_ablation_20260820/full_v2/"
    "board_latent_energy_best.pt"
)
DEFAULT_SMOKE_OUTPUT = (
    "/home/chenshuai/Project/output/board_dp_foresight_guided_server/"
    "dataset_smoke.json"
)
DEFAULT_ROLLOUT_LOG_DIR = (
    "/home/chenshuai/Project/output/board_force_rollouts/"
    "board_260615_latent_energy_scorer"
)

IMG_NORM = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


def apply_fixed_vision_enhance(image_t: torch.Tensor, config: Mapping[str, Any] | None) -> torch.Tensor:
    """Apply deterministic image enhancement recorded in a DP config."""
    if not config or not bool(config.get("vision_enhance", False)):
        return image_t
    out = image_t.clamp(0.0, 1.0)
    gamma = float(config.get("vision_gamma", 1.0))
    if gamma > 0 and abs(gamma - 1.0) > 1e-6:
        out = out.clamp_min(1e-6).pow(gamma)
    contrast = float(config.get("vision_contrast", 1.0))
    if abs(contrast - 1.0) > 1e-6:
        mean = out.mean(dim=(1, 2), keepdim=True)
        out = (out - mean) * contrast + mean
    brightness = float(config.get("vision_brightness", 0.0))
    if abs(brightness) > 1e-6:
        out = out + brightness
    return out.clamp(0.0, 1.0)


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_pickle(path: str | Path) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def camera_list(value: Sequence[str] | str) -> List[str]:
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    return list(value)


def parse_shape(value: Sequence[int] | str) -> Tuple[int, int]:
    if isinstance(value, str):
        parts = [int(x) for x in value.split(",")]
    else:
        parts = [int(x) for x in value]
    if len(parts) != 2:
        raise ValueError(f"Expected 2-D shape, got {value}")
    return parts[0], parts[1]


def optional_int(value: Any, default: int) -> int:
    return default if value is None else int(value)


def as_numpy_stats(stats: Dict[str, Any]) -> Dict[str, np.ndarray]:
    return {k: np.asarray(v, dtype=np.float32) for k, v in stats.items()}


def freeze(module: torch.nn.Module) -> None:
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)


def finite_float(value: torch.Tensor | float) -> float:
    if torch.is_tensor(value):
        out = float(value.detach().cpu())
    else:
        out = float(value)
    return out if math.isfinite(out) else float("nan")


def tensor_stats(x: torch.Tensor) -> Dict[str, float]:
    y = x.detach().float().flatten()
    return {
        "mean": float(y.mean().cpu()),
        "std": float(y.std(unbiased=False).cpu()),
        "min": float(y.min().cpu()),
        "max": float(y.max().cpu()),
    }


def select_score(output: Mapping[str, torch.Tensor], mode: str) -> torch.Tensor:
    return output["prob"][:, 0] if mode == "p_expert" else output[mode]


def extract_rollout_metadata(obs: Mapping[str, Any], *, arm: str) -> Dict[str, Any]:
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
        "task_match": (not out.get("manifest_task")) or str(out.get("manifest_task")) == "board",
        "arm_match": (not out.get("manifest_server_arm")) or str(out.get("manifest_server_arm")) == str(arm),
    }
    group = str(out.get("manifest_group") or "")
    if group:
        checks["group_valid"] = group in {"baseline", "guided"}
    out["rollout_metadata_source"] = "client_obs_rollout_metadata"
    out["rollout_metadata_checks"] = checks
    out["rollout_metadata_ok"] = all(bool(v) for v in checks.values())
    return out


def preprocess_image(raw_img: Any, resize_tf=None, crop_tf=None, vision_config=None) -> torch.Tensor:
    img = np.asarray(raw_img, dtype=np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    t = torch.from_numpy(img).permute(2, 0, 1).float()
    if resize_tf is not None:
        t = resize_tf(t)
    t = apply_fixed_vision_enhance(t, vision_config)
    if crop_tf is not None:
        t = crop_tf(t)
    t = IMG_NORM(t)
    return t.unsqueeze(0)


def minmax_norm(x: torch.Tensor, xmin: torch.Tensor, xmax: torch.Tensor) -> torch.Tensor:
    return (x - xmin) / (xmax - xmin + 1e-8) * 2.0 - 1.0


def minmax_denorm(x: torch.Tensor, xmin: torch.Tensor, xmax: torch.Tensor) -> torch.Tensor:
    return (x + 1.0) * 0.5 * (
        xmax.view(1, 1, -1) - xmin.view(1, 1, -1)
    ) + xmin.view(1, 1, -1)


def tensor_standard_norm(x: torch.Tensor, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    mean_t = torch.as_tensor(mean, dtype=torch.float32, device=x.device).view(1, 1, -1)
    std_t = torch.as_tensor(std, dtype=torch.float32, device=x.device).view(1, 1, -1)
    return (x - mean_t) / std_t.clamp_min(1e-8)


def predict_x0_from_eps(
    x_t: torch.Tensor,
    eps_pred: torch.Tensor,
    timestep: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    clip: bool = True,
) -> torch.Tensor:
    t_idx = int(timestep.item())
    alpha_prod = alphas_cumprod[t_idx].to(device=x_t.device, dtype=x_t.dtype)
    x0 = (x_t - torch.sqrt(1.0 - alpha_prod) * eps_pred) / torch.sqrt(alpha_prod)
    if clip:
        x0 = x0.clamp(-1.0, 1.0)
    return x0


def tensor_align_action(action_raw: torch.Tensor, horizon: int, mode: str) -> torch.Tensor:
    if mode == "none":
        aligned = action_raw[:, :horizon]
    elif mode == "shift1":
        aligned = action_raw[:, 1:horizon + 1]
    else:
        raise ValueError(f"Unknown alignment mode: {mode}")
    if aligned.shape[1] < horizon:
        pad = aligned[:, -1:].expand(-1, horizon - aligned.shape[1], -1)
        aligned = torch.cat([aligned, pad], dim=1)
    return aligned


def resolve_alignment_mode(requested: str, fs_config: Mapping[str, Any]) -> str:
    if requested != "auto":
        if requested not in {"none", "shift1"}:
            raise ValueError(f"Unknown alignment mode: {requested}")
        return requested

    use_state_traj = bool(fs_config.get("use_state_trajectory", False))
    future_offset = int(fs_config.get("future_offset", 1))
    default_action_offset = future_offset if use_state_traj else 0
    action_offset = int(fs_config.get("action_offset", default_action_offset))
    if action_offset == 0:
        return "none"
    if action_offset == 1:
        return "shift1"
    raise ValueError(
        "Cannot auto-resolve action alignment for "
        f"action_offset={action_offset}; pass --alignment explicitly."
    )


def dp_normalize_action(
    action_raw: torch.Tensor,
    action_min: torch.Tensor,
    action_max: torch.Tensor,
) -> torch.Tensor:
    scale = (action_max - action_min).clamp_min(1e-6).view(1, 1, -1)
    return 2.0 * (action_raw - action_min.view(1, 1, -1)) / scale - 1.0


def canonical_task(value: Any) -> str:
    task = str(value or "board").strip().lower()
    aliases = {"board_wiping": "board", "board_wipe": "board", "wiping": "board"}
    return aliases.get(task, task)


def canonical_vae_identity(value: Any) -> str:
    if not value:
        raise ValueError("Missing tactile VAE checkpoint in model contract")
    return vae_checkpoint_identity(os.path.realpath(os.path.expanduser(str(value))))


def marker_history_raw(marker_all: np.ndarray, t: int, window: int) -> np.ndarray:
    frames = []
    T = marker_all.shape[0]
    for k in range(window):
        idx = max(0, min(t - window + 1 + k, T - 1))
        frames.append(marker_all[idx])
    return np.stack(frames, axis=0).astype(np.float32)


def marker_history_norm(
    marker_all: np.ndarray,
    t: int,
    window: int,
    stats: Dict[str, np.ndarray],
) -> np.ndarray:
    raw = marker_history_raw(marker_all, t, window)
    return (raw - stats["marker_offset_mean"]) / (stats["marker_offset_std"] + 1e-8)


def pad_chunk(x: np.ndarray, length: int) -> np.ndarray:
    if x.shape[0] >= length:
        return x[:length].astype(np.float32)
    if x.shape[0] == 0:
        raise ValueError("Cannot pad an empty action/state chunk")
    pad = np.repeat(x[-1:], length - x.shape[0], axis=0)
    return np.concatenate([x, pad], axis=0).astype(np.float32)


def scan_episodes(dataset_dirs: Sequence[str]) -> List[str]:
    paths: List[str] = []
    for ds_dir in dataset_dirs:
        p = Path(ds_dir)
        if not p.exists():
            raise FileNotFoundError(ds_dir)
        direct = sorted(p.glob("episode_*.hdf5"))
        if direct:
            paths.extend(str(x) for x in direct)
            continue
        for child in sorted(p.iterdir()):
            if child.is_dir():
                paths.extend(str(x) for x in sorted(child.glob("episode_*.hdf5")))
    if not paths:
        raise RuntimeError(f"No episode_*.hdf5 found under {dataset_dirs}")
    return paths


def choose_smoke_point(
    episode_paths: Sequence[str],
    horizon: int,
    obs_horizon: int,
    seed: int,
) -> Tuple[str, int]:
    rng = random.Random(seed)
    shuffled = list(episode_paths)
    rng.shuffle(shuffled)
    for ep_path in shuffled:
        try:
            with h5py.File(ep_path, "r") as f:
                T = int(f["actions/joint_abs"].shape[0])
        except (OSError, KeyError):
            continue
        lo = max(0, obs_horizon - 1)
        hi = T - horizon - 2
        if hi >= lo:
            return ep_path, rng.randint(lo, hi)
    raise RuntimeError("No valid smoke point found")


def load_foresight_stack(
    foresight_dir: str,
    foresight_ckpt: str | None,
    device: torch.device,
) -> Dict[str, Any]:
    fs_config = load_json(Path(foresight_dir) / "args.json")
    fs_stats = as_numpy_stats(load_pickle(Path(foresight_dir) / "dataset_stats.pkl"))
    camera_names = camera_list(fs_config["camera_names"])
    if camera_names != ["gelsight"]:
        raise ValueError(
            "Board guided serving currently expects marker-only foresight "
            f"with camera_names=['gelsight'], got {camera_names}"
        )
    model_cls = MultiStepLatentForesightModel
    if str(fs_config.get("tactile_vae_version", "v1")).lower() == "v2":
        class V2MultiStepLatentForesightModel(MultiStepLatentForesightModel):
            def __init__(self, **kwargs):
                vae_ckpt = kwargs.pop("tactile_vae_ckpt")
                latent_dim = kwargs.get("tactile_vae_latent_dim", 16)
                window = kwargs.get("max_history", 8)
                super().__init__(**kwargs, tactile_vae_ckpt=None)
                self.tactile_vae = TactileVAEv2(latent_dim=latent_dim, temporal_window=window)
                checkpoint = torch.load(vae_ckpt, map_location="cpu", weights_only=False)
                self.tactile_vae.load_state_dict(checkpoint["model_state_dict"], strict=True)
                self.tactile_vae.requires_grad_(False)
        model_cls = V2MultiStepLatentForesightModel
    model = model_cls(
        camera_names=camera_names,
        cam_backbone_mapping={cam: 0 for cam in camera_names},
        hidden_dim=int(fs_config.get("hidden_dim", 512)),
        state_dim=int(fs_config.get("state_dim", 7)),
        foresight_layers=int(fs_config.get("foresight_layers", 3)),
        foresight_nheads=int(fs_config.get("foresight_nheads", 8)),
        foresight_dim_feedforward=int(fs_config.get("foresight_dim_feedforward", 2048)),
        dropout=float(fs_config.get("dropout", 0.1)),
        tactile_mode=fs_config.get("tactile_mode", "marker"),
        max_history=optional_int(fs_config.get("max_history"), 8),
        predict_horizon=int(fs_config.get("predict_horizon", fs_config.get("foresight_horizon", 16))),
        tactile_vae_ckpt=fs_config.get("tactile_vae_ckpt"),
        tactile_vae_latent_dim=int(fs_config.get("tactile_vae_latent_dim", 16)),
    ).to(device)
    ckpt_path = foresight_ckpt or str(Path(foresight_dir) / "foresight_best.ckpt")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, Mapping) and "model_state_dict" in state:
        state = state["model_state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        print(f"[board-guided] foresight load_state_dict missing={len(missing)} unexpected={len(unexpected)}")
    freeze(model)
    return {
        "config": fs_config,
        "stats": fs_stats,
        "model": model,
        "ckpt_path": ckpt_path,
    }


class BoardGuidedDPStack:
    """Owns DP, Foresight, scorer and tensor contracts for serving."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        if args.device == "cpu":
            self.device = torch.device("cpu")
        elif args.device == "cuda":
            self.device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 else "cuda")
        else:
            self.device = torch.device(
                f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
            )

        self.dp_ckpt = str(args.dp_ckpt)
        self.ckpt_dir = str(Path(self.dp_ckpt).parent)
        self.config = load_json(Path(self.ckpt_dir) / "config.json")
        self.variant = str(self.config.get("variant", ""))
        if self.variant != "tactile_vae_frozen":
            raise ValueError(f"Expected tactile_vae_frozen DP config, got variant={self.variant}")

        self.camera_names = camera_list(self.config["camera_names"])
        self.action_dim = int(self.config["action_dim"])
        self.pred_horizon = int(self.config["pred_horizon"])
        self.obs_horizon = int(self.config.get("obs_horizon", 2))
        self.tac_history = int(self.config.get("tac_history", 8))
        self.temporal_stride = int(self.config.get("temporal_stride", 1))
        if self.temporal_stride < 1:
            raise ValueError(f"Invalid temporal_stride in DP config: {self.temporal_stride}")
        self.tac_side = str(self.config.get("tac_side", "left"))
        self.proprio_key = str(self.config.get("proprio_key", "proprio_joint"))
        self.num_train_timesteps = int(self.config.get("num_train_timesteps", 100))
        self.num_inference_steps = int(args.num_inference_steps or self.config.get("num_inference_steps", 100))

        resize_shape = self.config.get("resize_shape")
        crop_shape = self.config.get("crop_shape")
        self.resize_tf = transforms.Resize(parse_shape(resize_shape)) if resize_shape else None
        self.crop_tf = transforms.CenterCrop(parse_shape(crop_shape)) if crop_shape else None

        ns = self.config["norm_stats"]
        self.action_min = torch.as_tensor(ns["action_min"], dtype=torch.float32, device=self.device)
        self.action_max = torch.as_tensor(ns["action_max"], dtype=torch.float32, device=self.device)
        self.qpos_min = torch.as_tensor(ns["qpos_min"], dtype=torch.float32, device=self.device)
        self.qpos_max = torch.as_tensor(ns["qpos_max"], dtype=torch.float32, device=self.device)

        self._load_dp()
        self.fs = load_foresight_stack(args.foresight_dir, args.foresight_ckpt, self.device)
        self.scorer = BoardLatentEnergyRuntime(args.scorer_ckpt, device=str(self.device)).to(self.device)
        freeze(self.scorer)

        self.horizon = int(
            self.fs["config"].get(
                "predict_horizon",
                self.fs["config"].get("foresight_horizon", 16),
            )
        )
        self.alignment = resolve_alignment_mode(args.alignment, self.fs["config"])
        self._contract_checks()
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_train_timesteps,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )
        print(
            "[board-guided] loaded: "
            f"device={self.device}, pred_horizon={self.pred_horizon}, "
            f"obs_horizon={self.obs_horizon}, temporal_stride={self.temporal_stride}, "
            f"foresight_horizon={self.horizon}, score_input=tactile_only, "
            f"scale={args.guidance_scale}, steps={args.guidance_steps}, "
            f"alignment={self.alignment}"
        )

    def _load_dp(self) -> None:
        down_dims = self.config.get("down_dims", [512, 1024, 2048])
        if isinstance(down_dims, str):
            down_dims = [int(x) for x in down_dims.split(",")]
        self.vision_encoder = OfficialVisionEncoder(self.camera_names).to(self.device)
        self.tac_encoder = FrozenTactileVAEEncoder(
            self.config.get("vae_checkpoint", ""),
            latent_dim=int(self.config.get("vae_latent_dim", 16)),
            temporal_window=self.tac_history,
        ).to(self.device)
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.action_dim,
            global_cond_dim=int(self.config["global_cond_dim"]),
            diffusion_step_embed_dim=int(self.config.get("diffusion_step_embed_dim", 128)),
            down_dims=down_dims,
            kernel_size=5,
        ).to(self.device)

        ckpt = torch.load(self.dp_ckpt, map_location="cpu", weights_only=False)
        net_key = "ema_net" if self.args.use_ema and "ema_net" in ckpt else "noise_pred_net"
        vis_key = "ema_vis" if self.args.use_ema and "ema_vis" in ckpt else "vision_encoder"
        self.noise_pred_net.load_state_dict(ckpt[net_key])
        self.vision_encoder.load_state_dict(ckpt[vis_key])
        freeze(self.vision_encoder)
        freeze(self.tac_encoder)
        freeze(self.noise_pred_net)
        self.dp_meta = {
            "epoch": ckpt.get("epoch"),
            "train_loss": ckpt.get("train_loss"),
            "val_loss": ckpt.get("val_loss"),
            "net_key": net_key,
            "vis_key": vis_key,
        }
        print(f"[board-guided] DP loaded: {self.dp_meta}")
        del ckpt

    def _contract_checks(self) -> None:
        fs_config = self.fs["config"]
        schema_version = int(getattr(self.scorer, "schema_version", 0))
        if schema_version != 2:
            raise ValueError(f"Expected tactile-only scorer schema_version=2, got {schema_version}")
        dp_vae = canonical_vae_identity(self.config.get("vae_checkpoint"))
        fs_vae = canonical_vae_identity(fs_config.get("tactile_vae_ckpt"))
        scorer_vae = str(self.scorer.vae_identity)
        input_mode = str(getattr(self.scorer, "input_mode", ""))
        if input_mode != "tactile_only_latent":
            raise ValueError(
                "Board guidance requires a tactile-only scorer checkpoint; "
                f"got input_mode={input_mode!r}"
            )
        if int(fs_config.get("state_dim", self.action_dim)) != self.action_dim:
            raise ValueError("DP action_dim and Foresight state_dim mismatch")
        if self.pred_horizon < self.horizon:
            raise ValueError(
                f"DP pred_horizon={self.pred_horizon} shorter than foresight horizon={self.horizon}"
            )
        if int(self.scorer.horizon) != self.horizon or int(self.scorer.model.chunk_len) != self.horizon:
            raise ValueError(
                f"Scorer horizon={self.scorer.horizon}/chunk_len={self.scorer.model.chunk_len} "
                f"!= Foresight horizon={self.horizon}"
            )
        expected_latent_dim = int(fs_config.get("tactile_vae_latent_dim", 16)) * 3 * 3
        if tuple(self.scorer.latent_shape) != (expected_latent_dim,) or int(self.scorer.model.latent_dim) != expected_latent_dim:
            raise ValueError(
                f"Scorer latent_shape={self.scorer.latent_shape}/latent_dim={self.scorer.model.latent_dim} != "
                f"Foresight latent_dim={expected_latent_dim}"
            )
        if fs_vae != scorer_vae:
            raise ValueError(
                "TactileVAE checkpoint mismatch between Foresight and scorer: "
                f"foresight={fs_vae}, scorer={scorer_vae}"
            )
        fs_future_offset = int(fs_config.get("future_offset", 1))
        fs_temporal_stride = int(fs_config.get("temporal_stride", 1))
        scorer_future_offset = int(self.scorer.future_offset)
        scorer_temporal_stride = int(getattr(self.scorer, "temporal_stride"))
        if fs_temporal_stride != self.temporal_stride:
            raise ValueError(
                f"DP temporal_stride={self.temporal_stride} != "
                f"Foresight temporal_stride={fs_temporal_stride}"
            )
        if scorer_temporal_stride != self.temporal_stride:
            raise ValueError(
                f"DP temporal_stride={self.temporal_stride} != "
                f"scorer temporal_stride={scorer_temporal_stride}"
            )
        if scorer_future_offset != fs_future_offset:
            raise ValueError(
                f"Scorer future_offset={scorer_future_offset} != "
                f"Foresight future_offset={fs_future_offset}"
            )
        tasks = {
            "dp": canonical_task(self.config.get("task")),
            "foresight": canonical_task(fs_config.get("task")),
            "scorer": canonical_task(self.scorer.task),
        }
        if len(set(tasks.values())) != 1 or tasks["dp"] != "board":
            raise ValueError(f"Task contract mismatch: {tasks}")
        use_state_traj = bool(fs_config.get("use_state_trajectory", False))
        default_action_offset = fs_future_offset if use_state_traj else 0
        fs_action_offset = int(fs_config.get("action_offset", default_action_offset))
        expected_alignment = "none" if fs_action_offset == 0 else "shift1" if fs_action_offset == 1 else None
        if expected_alignment is None:
            raise ValueError(f"Unsupported Foresight action_offset={fs_action_offset}")
        if self.alignment != expected_alignment:
            raise ValueError(
                f"Foresight action_offset={fs_action_offset} requires alignment="
                f"{expected_alignment}, got {self.alignment}"
            )

    def preprocess_obs(self, obs: Mapping[str, Any]) -> Dict[str, Any]:
        images_dict = {}
        for cam in self.camera_names:
            if cam == "gelsight":
                continue
            raw = obs["images"][cam]
            images_dict[cam] = preprocess_image(raw, self.resize_tf, self.crop_tf, self.config).to(self.device)

        tac = obs["tac"]
        side = self.tac_side if self.tac_side in tac else list(tac.keys())[0]
        side_data = tac[side]
        if isinstance(side_data, Mapping):
            if "marker_offset" not in side_data:
                raise KeyError(f"obs['tac'][{side!r}] missing marker_offset")
            marker = side_data["marker_offset"]
        else:
            marker = side_data

        return {
            "images_dict": images_dict,
            "qpos_raw": np.asarray(obs["qpos"], dtype=np.float32),
            "marker_offset": np.asarray(marker, dtype=np.float32),
        }

    def build_obs_cond(
        self,
        obs_buffer: Sequence[Dict[str, Any]],
        marker_buffer: Sequence[np.ndarray],
    ) -> torch.Tensor:
        feats = []
        with torch.no_grad():
            for frame in obs_buffer:
                vf = self.vision_encoder(frame["images_dict"])
                marker_end = int(frame["_marker_idx"])
                frames = []
                for k in range(self.tac_history):
                    idx = marker_end - (self.tac_history - 1 - k) * self.temporal_stride
                    idx = max(0, min(idx, len(marker_buffer) - 1))
                    frames.append(marker_buffer[idx])
                marker_seq = torch.as_tensor(
                    np.stack(frames, axis=0),
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)
                tf = self.tac_encoder(marker_seq)
                qpos_raw = torch.as_tensor(
                    frame["qpos_raw"],
                    dtype=torch.float32,
                    device=self.device,
                ).view(1, -1)
                qpos_norm = minmax_norm(qpos_raw, self.qpos_min, self.qpos_max)
                feats.append(torch.cat([vf, tf, qpos_norm], dim=-1))
        return torch.cat(feats, dim=-1)

    def make_foresight_context(
        self,
        qpos_raw: np.ndarray,
        marker_buffer: Sequence[np.ndarray],
    ) -> Dict[str, Any]:
        fs_config = self.fs["config"]
        fs_stats = self.fs["stats"]
        window = int(fs_config.get("tactile_vae_window", 8))
        marker_all = np.asarray(marker_buffer, dtype=np.float32)
        cur_marker = marker_history_norm(marker_all, len(marker_all) - 1, window, fs_stats)
        qpos_norm = (
            np.asarray(qpos_raw, dtype=np.float32) - fs_stats["qpos_mean"]
        ) / (fs_stats["qpos_std"] + 1e-8)
        return {
            "images": [torch.as_tensor(cur_marker, dtype=torch.float32, device=self.device).unsqueeze(0)],
            "qpos": torch.as_tensor(qpos_norm, dtype=torch.float32, device=self.device).unsqueeze(0),
            "future_images": None,
        }

    def foresight_predict_latent(
        self,
        action_aligned_raw: torch.Tensor,
        context: Dict[str, Any],
    ) -> torch.Tensor:
        fs_config = self.fs["config"]
        fs_stats = self.fs["stats"]
        if bool(fs_config.get("use_state_trajectory", False)):
            action_norm = tensor_standard_norm(
                action_aligned_raw,
                fs_stats["qpos_mean"],
                fs_stats["qpos_std"],
            )
        else:
            action_norm = tensor_standard_norm(
                action_aligned_raw,
                fs_stats["action_mean"],
                fs_stats["action_std"],
            )
        pred, _, _, _ = self.fs["model"](
            context["images"],
            action_norm,
            future_images=None,
            qpos=context["qpos"],
        )
        return pred

    def score_from_foresight(
        self,
        action_aligned_raw: torch.Tensor,
        context: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        pred = self.foresight_predict_latent(action_aligned_raw, context)
        out = self.scorer(pred, normalized=False)
        return select_score(out, self.args.score_mode), out

    def apply_guidance_update(
        self,
        action: torch.Tensor,
        grad: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        flat = grad.flatten(1)
        grad_norm = flat.norm(dim=1).clamp_min(1e-12)
        finite = torch.isfinite(grad).flatten(1).all(dim=1)
        if self.args.normalize_guidance_grad:
            update = grad / grad_norm.view(-1, 1, 1)
            clipped_norm = torch.ones_like(grad_norm)
        else:
            coef = (float(self.args.max_grad_norm) / grad_norm).clamp(max=1.0)
            update = grad * coef.view(-1, 1, 1)
            clipped_norm = grad_norm * coef
        update = update * float(self.args.guidance_scale)
        guided = (action + update).clamp(
            -float(self.args.sample_clip),
            float(self.args.sample_clip),
        ).detach()
        return guided, {
            "grad_norm": float(grad_norm.mean().detach().cpu()),
            "grad_norm_max": float(grad_norm.max().detach().cpu()),
            "grad_finite": float(finite.float().mean().detach().cpu()),
            "applied_update_norm": float(update.flatten(1).norm(dim=1).mean().detach().cpu()),
            "clipped_grad_norm": float(clipped_norm.mean().detach().cpu()),
        }

    def guided_ddpm_inference(
        self,
        obs_cond: torch.Tensor,
        context: Dict[str, Any],
        seed: int | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(seed))
        action = torch.randn(
            (1, self.pred_horizon, self.action_dim),
            generator=generator,
            device=self.device,
        )
        alphas = self.noise_scheduler.alphas_cumprod.to(self.device)
        total_steps = len(self.noise_scheduler.timesteps)
        guide_start = max(0, total_steps - int(self.args.guidance_steps))
        grad_logs: List[Dict[str, float]] = []

        for step_idx, t in enumerate(self.noise_scheduler.timesteps):
            t_batch = t.reshape(1).to(self.device)
            with torch.no_grad():
                eps = self.noise_pred_net(action, t_batch, global_cond=obs_cond)

            do_guide = (
                float(self.args.guidance_scale) > 0.0
                and int(self.args.guidance_steps) > 0
                and step_idx >= guide_start
            )
            if do_guide:
                action_for_grad = action.detach().requires_grad_(True)
                x0_norm = predict_x0_from_eps(action_for_grad, eps.detach(), t, alphas, clip=True)
                x0_raw = minmax_denorm(x0_norm, self.action_min, self.action_max)
                action_aligned = tensor_align_action(x0_raw, self.horizon, self.alignment)
                score, out = self.score_from_foresight(
                    action_aligned,
                    context,
                )
                objective = score.mean()
                smooth_penalty = torch.zeros((), device=self.device)
                if self.args.lambda_smooth > 0:
                    action_norm = dp_normalize_action(
                        action_aligned,
                        self.action_min,
                        self.action_max,
                    )
                    smooth_penalty = (action_norm[:, 1:] - action_norm[:, :-1]).pow(2).mean()
                    objective = objective - float(self.args.lambda_smooth) * smooth_penalty

                grad = torch.autograd.grad(objective, action_for_grad, retain_graph=False)[0]
                action, log = self.apply_guidance_update(action_for_grad.detach(), grad.detach())

                with torch.no_grad():
                    guided_x0 = predict_x0_from_eps(action, eps.detach(), t, alphas, clip=True)
                    guided_raw = minmax_denorm(guided_x0, self.action_min, self.action_max)
                    guided_aligned = tensor_align_action(guided_raw, self.horizon, self.alignment)
                    _, after_out = self.score_from_foresight(
                        guided_aligned,
                        context,
                    )
                    eps = self.noise_pred_net(action, t_batch, global_cond=obs_cond)

                log.update({
                    "step_idx": float(step_idx),
                    "timestep": float(t.item()),
                    "pre_score": finite_float(score.mean()),
                    "pre_objective": finite_float(objective),
                    "smoothness_penalty": finite_float(smooth_penalty),
                    "pre_expert_margin": finite_float(out["expert_margin"].mean()),
                    "post_expert_margin": finite_float(after_out["expert_margin"].mean()),
                    "post_quality": finite_float(after_out["quality_0_100"].mean()),
                })
                grad_logs.append(log)

            with torch.no_grad():
                action = self.noise_scheduler.step(eps, t, action).prev_sample.detach()

        summary = self._summarize_guidance_logs(grad_logs)
        summary.update({
            "score_input": "predicted_tactile_latent_only",
            "guidance_scale": float(self.args.guidance_scale),
            "guidance_steps": int(self.args.guidance_steps),
            "score_mode": self.args.score_mode,
            "alignment": self.alignment,
            "num_inference_steps": int(self.num_inference_steps),
        })
        return action.detach(), summary

    @staticmethod
    def _summarize_guidance_logs(logs: Sequence[Dict[str, float]]) -> Dict[str, Any]:
        if not logs:
            return {
                "guided_steps": 0,
                "grad_finite_mean": 1.0,
                "grad_norm_mean": 0.0,
                "grad_norm_max": 0.0,
                "update_norm_mean": 0.0,
                "pre_expert_margin_mean": float("nan"),
                "post_expert_margin_mean": float("nan"),
                "post_quality_mean": float("nan"),
            }
        return {
            "guided_steps": len(logs),
            "grad_finite_mean": float(np.mean([x["grad_finite"] for x in logs])),
            "grad_norm_mean": float(np.mean([x["grad_norm"] for x in logs])),
            "grad_norm_max": float(np.max([x["grad_norm_max"] for x in logs])),
            "update_norm_mean": float(np.mean([x["applied_update_norm"] for x in logs])),
            "pre_expert_margin_mean": float(np.mean([x["pre_expert_margin"] for x in logs])),
            "post_expert_margin_mean": float(np.mean([x["post_expert_margin"] for x in logs])),
            "post_quality_mean": float(np.mean([x["post_quality"] for x in logs])),
            "last_step": logs[-1],
        }

    def denormalize_action_step(self, action_norm_step: torch.Tensor) -> np.ndarray:
        raw = (action_norm_step + 1.0) * 0.5 * (
            self.action_max.view(1, -1) - self.action_min.view(1, -1)
        ) + self.action_min.view(1, -1)
        return raw.squeeze(0).detach().cpu().numpy().astype(np.float32)


def build_dataset_obs_buffer(
    stack: BoardGuidedDPStack,
    ep_path: str,
    start_t: int,
) -> Tuple[List[Dict[str, Any]], List[np.ndarray]]:
    obs_buffer: deque[Dict[str, Any]] = deque(maxlen=stack.obs_horizon)
    marker_buffer: List[np.ndarray] = []
    with h5py.File(ep_path, "r") as f:
        marker_all = f[f"observations/tac/{stack.tac_side}/marker_offset"][()].astype(np.float32)
        obs_indices = [
            max(0, start_t - (stack.obs_horizon - 1 - k) * stack.temporal_stride)
            for k in range(stack.obs_horizon)
        ]
        marker_prefix_end = obs_indices[-1]
        for t in range(marker_prefix_end + 1):
            marker_buffer.append(marker_all[t])
        for t in obs_indices:
            obs = {
                "images": {
                    cam: f[f"observations/images/{cam}"][t]
                    for cam in stack.camera_names
                    if cam != "gelsight"
                },
                "qpos": f[f"observations/{stack.proprio_key}"][t].astype(np.float32),
                "tac": {stack.tac_side: {"marker_offset": marker_all[t]}},
            }
            processed = stack.preprocess_obs(obs)
            processed["_marker_idx"] = int(t)
            obs_buffer.append(processed)
    return list(obs_buffer), marker_buffer


def dry_run_dataset_smoke(args: argparse.Namespace) -> Dict[str, Any]:
    stack = BoardGuidedDPStack(args)
    if args.dataset_dirs:
        dataset_dirs = [x.strip() for x in args.dataset_dirs.split(",") if x.strip()]
    else:
        dataset_dirs = list(stack.config.get("dataset_dirs", []))
    episode_paths = scan_episodes(dataset_dirs)
    ep_path, start_t = choose_smoke_point(
        episode_paths,
        horizon=stack.horizon,
        obs_horizon=stack.obs_horizon,
        seed=args.seed,
    )
    obs_buffer, marker_buffer = build_dataset_obs_buffer(stack, ep_path, start_t)
    obs_cond = stack.build_obs_cond(obs_buffer, marker_buffer)
    context = stack.make_foresight_context(obs_buffer[-1]["qpos_raw"], marker_buffer)

    def evaluate_action(action_norm: torch.Tensor) -> Dict[str, Any]:
        action_raw = minmax_denorm(action_norm, stack.action_min, stack.action_max)
        aligned_raw = tensor_align_action(action_raw, stack.horizon, stack.alignment)
        with torch.no_grad():
            pred = stack.foresight_predict_latent(aligned_raw, context)
            score_out = stack.scorer(pred, normalized=False)
        return {
            "action_norm_shape": list(action_norm.shape),
            "action_norm_stats": tensor_stats(action_norm),
            "action_raw_aligned_shape": list(aligned_raw.shape),
            "pred_latent_shape": list(pred.shape),
            "finite_action": bool(torch.isfinite(action_norm).all().item()),
            "finite_pred_latent": bool(torch.isfinite(pred).all().item()),
            "score": {
                "expert_margin": finite_float(score_out["expert_margin"].mean()),
                "quality_0_100": finite_float(score_out["quality_0_100"].mean()),
                "score_good": finite_float(score_out["score_good"].mean()),
                "p_expert": finite_float(score_out["prob"][:, 0].mean()),
            },
        }

    requested_guidance_scale = float(args.guidance_scale)
    requested_guidance_steps = int(args.guidance_steps)
    args.guidance_scale = 0.0
    args.guidance_steps = 0
    unguided_norm, unguided_report = stack.guided_ddpm_inference(obs_cond, context, seed=args.seed)
    args.guidance_scale = requested_guidance_scale
    args.guidance_steps = requested_guidance_steps
    guided_norm, report = stack.guided_ddpm_inference(obs_cond, context, seed=args.seed)

    unguided_eval = evaluate_action(unguided_norm)
    guided_eval = evaluate_action(guided_norm)
    score_delta = {
        key: guided_eval["score"][key] - unguided_eval["score"][key]
        for key in guided_eval["score"]
    }

    smoke_pass = bool(
        unguided_eval["finite_action"]
        and unguided_eval["finite_pred_latent"]
        and guided_eval["finite_action"]
        and guided_eval["finite_pred_latent"]
        and report.get("grad_finite_mean", 0.0) >= args.min_finite_grad_rate
        and int(report.get("guided_steps", 0)) == int(requested_guidance_steps if requested_guidance_scale > 0 else 0)
    )
    result = {
        "dry_run_dataset_smoke_pass": smoke_pass,
        "episode": ep_path,
        "frame_t": int(start_t),
        "device": str(stack.device),
        "dp_ckpt": stack.dp_ckpt,
        "foresight_ckpt": stack.fs["ckpt_path"],
        "scorer_ckpt": str(args.scorer_ckpt),
        "obs_cond_shape": list(obs_cond.shape),
        "unguided": unguided_eval,
        "guided": guided_eval,
        "score_delta_guided_minus_unguided": score_delta,
        "unguided_report": unguided_report,
        "guidance_report": report,
    }
    if args.smoke_output:
        out_path = Path(args.smoke_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "dry_run_dataset_smoke_pass": smoke_pass,
        "device": str(stack.device),
        "guided_steps": report.get("guided_steps"),
        "grad_finite_mean": report.get("grad_finite_mean"),
        "unguided_expert_margin": unguided_eval["score"]["expert_margin"],
        "guided_expert_margin": guided_eval["score"]["expert_margin"],
        "expert_margin_delta": score_delta["expert_margin"],
        "unguided_quality_0_100": unguided_eval["score"]["quality_0_100"],
        "guided_quality_0_100": guided_eval["score"]["quality_0_100"],
        "quality_delta": score_delta["quality_0_100"],
        "smoke_output": args.smoke_output,
    }, ensure_ascii=False, indent=2))
    return result


def run_server(args: argparse.Namespace) -> None:
    stack = BoardGuidedDPStack(args)
    action_skip = int(args.action_skip)
    action_horizon = min(int(args.action_horizon), stack.pred_horizon - action_skip)
    query_freq = action_horizon
    arm = str(args.arm)
    rollout_log_dir = None if args.disable_server_rollout_log else str(args.server_rollout_log_dir)
    server_metadata = {
        "protocol": "board_dp_foresight_guided",
        "arm": arm,
        "variant": stack.variant,
        "camera_names": stack.camera_names,
        "action_dim": stack.action_dim,
        "pred_horizon": stack.pred_horizon,
        "obs_horizon": stack.obs_horizon,
        "temporal_stride": stack.temporal_stride,
        "action_skip": action_skip,
        "action_horizon": action_horizon,
        "max_timesteps": int(args.max_timesteps),
        "guidance": {
            "enabled": not args.disable_guidance,
            "location": "inside_ddpm_denoising",
            "score_input": "predicted_tactile_latent_only",
            "scale": args.guidance_scale,
            "steps": args.guidance_steps,
            "score_mode": args.score_mode,
            "alignment": stack.alignment,
        },
        "reranking": False,
        "server_rollout_logging": not args.disable_server_rollout_log,
        "server_rollout_log_dir": rollout_log_dir,
    }
    server = TactileACTServer(
        host=args.host,
        port=args.port,
        metadata=server_metadata,
    )
    server.start()
    print(f"[board-guided] listening on {args.host}:{args.port}")

    try:
        ep = 0
        while True:
            try:
                print(f"\n[board-guided] === episode {ep} ===")
                obs = server.recv_obs()
            except ClientDisconnected:
                print("[board-guided] client gone before first obs, waiting...")
                continue

            obs_buffer: deque[Dict[str, Any]] = deque(
                maxlen=(stack.obs_horizon - 1) * stack.temporal_stride + 1
            )
            marker_buffer: List[np.ndarray] = []
            marker_step = 0
            action_chunk: torch.Tensor | None = None
            last_report: Dict[str, Any] | None = None
            rollout_logger = None
            first_transport_wall = obs.get("_server_transport_recv_wall_time")
            if isinstance(first_transport_wall, (int, float, np.integer, np.floating)) and np.isfinite(first_transport_wall):
                episode_start_wall_time = float(first_transport_wall)
            else:
                episode_start_wall_time = time.time()
            first_transport_perf = obs.get("_server_transport_recv_perf")
            if isinstance(first_transport_perf, (int, float, np.integer, np.floating)) and np.isfinite(first_transport_perf):
                episode_start_perf = float(first_transport_perf)
            else:
                episode_start_perf = time.perf_counter()
            prev_obs_recv_perf: float | None = None
            if not args.disable_server_rollout_log:
                rollout_logger = ServerRolloutLogger(
                    rollout_log_dir,
                    episode=ep,
                    host=args.host,
                    port=args.port,
                    task="board",
                    arm=arm,
                    server_metadata=server_metadata,
                    episode_start_wall_time=episode_start_wall_time,
                )
                rollout_metadata = extract_rollout_metadata(obs, arm=arm)
                if rollout_metadata:
                    rollout_logger.metadata.update(rollout_metadata)
                print(f"[board-guided] server rollout log: {rollout_logger.trial_dir}")
            final_step = 0
            stop_reason = "max_timesteps"

            try:
                for step in range(int(args.max_timesteps)):
                    server_step_t0 = time.perf_counter()
                    server_loop_obs_start_wall_time = time.time()
                    recv_wall = obs.get("_server_transport_recv_wall_time")
                    if isinstance(recv_wall, (int, float, np.integer, np.floating)) and np.isfinite(recv_wall):
                        server_obs_recv_wall_time = float(recv_wall)
                    else:
                        server_obs_recv_wall_time = server_loop_obs_start_wall_time
                    recv_perf = obs.get("_server_transport_recv_perf")
                    if isinstance(recv_perf, (int, float, np.integer, np.floating)) and np.isfinite(recv_perf):
                        server_obs_recv_perf = float(recv_perf)
                    else:
                        server_obs_recv_perf = time.perf_counter()
                    server_obs_recv_episode_t = server_obs_recv_perf - episode_start_perf
                    if prev_obs_recv_perf is None:
                        server_obs_interval_ms = 0.0
                    else:
                        server_obs_interval_ms = (server_obs_recv_perf - prev_obs_recv_perf) * 1000.0
                    prev_obs_recv_perf = server_obs_recv_perf
                    final_step = step
                    processed = stack.preprocess_obs(obs)
                    marker_buffer.append(processed["marker_offset"])
                    processed["_marker_idx"] = marker_step
                    marker_step += 1
                    obs_buffer.append(processed)
                    required_obs_buffer = (stack.obs_horizon - 1) * stack.temporal_stride + 1
                    while len(obs_buffer) < required_obs_buffer:
                        pad = dict(processed)
                        pad["_marker_idx"] = 0
                        obs_buffer.appendleft(pad)

                    obs_stride_buffer = list(obs_buffer)[
                        -(stack.obs_horizon - 1) * stack.temporal_stride - 1::stack.temporal_stride
                    ]
                    obs_cond = stack.build_obs_cond(obs_stride_buffer, marker_buffer)
                    is_replan_step = bool(step % query_freq == 0 or action_chunk is None)
                    chunk_index = int(step // query_freq)
                    server_replan_time_ms = 0.0
                    if is_replan_step:
                        context = stack.make_foresight_context(processed["qpos_raw"], marker_buffer)
                        saved_scale = args.guidance_scale
                        saved_steps = args.guidance_steps
                        if args.disable_guidance:
                            args.guidance_scale = 0.0
                            args.guidance_steps = 0
                        replan_t0 = time.perf_counter()
                        try:
                            action_chunk, last_report = stack.guided_ddpm_inference(obs_cond, context)
                        finally:
                            if args.disable_guidance:
                                args.guidance_scale = saved_scale
                                args.guidance_steps = saved_steps
                        server_replan_time_ms = (time.perf_counter() - replan_t0) * 1000.0
                        if args.disable_guidance:
                            last_report["guidance_disabled"] = True
                            last_report["arm"] = arm
                            last_report["reranking"] = False
                        if step % max(1, query_freq * 5) == 0:
                            print(
                                f"  step {step}: guided_steps={last_report.get('guided_steps')}, "
                                f"grad_finite={last_report.get('grad_finite_mean')}, "
                                f"post_margin={last_report.get('post_expert_margin_mean')}"
                            )

                    raw_norm = action_chunk[:, action_skip + step % query_freq]
                    action = stack.denormalize_action_step(raw_norm)
                    msg = {
                        "actions": action[None, :],
                        "step": step,
                        "timing": {
                            "server_episode_start_wall_time": float(episode_start_wall_time),
                            "server_obs_recv_wall_time": float(server_obs_recv_wall_time),
                            "server_obs_recv_episode_t": float(server_obs_recv_episode_t),
                            "server_obs_interval_ms": float(server_obs_interval_ms),
                            "server_loop_obs_start_wall_time": float(server_loop_obs_start_wall_time),
                            "server_is_replan_step": int(is_replan_step),
                            "server_chunk_index": chunk_index,
                            "server_chunk_action_index": int(step % query_freq),
                            "server_replan_time_ms": float(server_replan_time_ms),
                        },
                    }
                    if last_report is not None and args.send_guidance_report:
                        msg["guidance_report"] = last_report
                    server_step_time_ms = (time.perf_counter() - server_step_t0) * 1000.0
                    timing = {
                        "server_episode_start_wall_time": float(episode_start_wall_time),
                        "server_obs_recv_wall_time": float(server_obs_recv_wall_time),
                        "server_obs_recv_episode_t": float(server_obs_recv_episode_t),
                        "server_obs_interval_ms": float(server_obs_interval_ms),
                        "server_loop_obs_start_wall_time": float(server_loop_obs_start_wall_time),
                        "server_is_replan_step": int(is_replan_step),
                        "server_chunk_index": chunk_index,
                        "server_chunk_action_index": int(step % query_freq),
                        "server_replan_time_ms": float(server_replan_time_ms),
                        "server_step_time_ms": float(server_step_time_ms),
                        "server_action_enqueue_wall_time": float(time.time()),
                    }
                    msg["timing"].update(timing)
                    if rollout_logger is not None:
                        rollout_logger.record(
                            step=step,
                            obs=obs,
                            action=action,
                            action_norm=raw_norm.squeeze(0).detach().cpu().numpy().astype(np.float32),
                            guidance_report=last_report,
                            timing=timing,
                        )
                    server.send_action(msg)
                    if step + 1 < int(args.max_timesteps):
                        obs = server.recv_obs()
            except ClientDisconnected:
                print(f"[board-guided] client disconnected at step {step}")
                stop_reason = "client_disconnected"
            finally:
                if rollout_logger is not None:
                    rollout_logger.finalize(steps=final_step + 1, stop_reason=stop_reason)
            ep += 1
    except KeyboardInterrupt:
        print("\n[board-guided] shutting down")
    finally:
        server.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dp_ckpt", default=DEFAULT_DP_CKPT)
    parser.add_argument("--foresight_dir", default=DEFAULT_FORESIGHT_DIR)
    parser.add_argument("--foresight_ckpt", default=None)
    parser.add_argument("--scorer_ckpt", default=DEFAULT_SCORER_CKPT)
    parser.add_argument("--arm", default="latent_energy_guided")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--guidance_steps", type=int, default=5)
    parser.add_argument("--guidance_scale", type=float, default=0.003)
    parser.add_argument("--score_mode", default="expert_margin",
                        choices=["score_good", "expert_margin", "p_expert", "quality_0_100"])
    parser.add_argument(
        "--alignment",
        default="auto",
        choices=["auto", "shift1", "none"],
        help="Action-to-foresight temporal alignment. auto uses the foresight action_offset; action-conditioned checkpoints usually resolve to none, old state-trajectory checkpoints with offset 1 resolve to shift1.",
    )
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--normalize_guidance_grad", action="store_true", default=True)
    parser.add_argument("--raw_guidance_grad", action="store_false", dest="normalize_guidance_grad")
    parser.add_argument("--sample_clip", type=float, default=1.5)
    parser.add_argument(
        "--lambda_smooth",
        type=float,
        default=0.0,
        help="Penalty weight for temporal smoothness in DP-normalized action space.",
    )
    parser.add_argument("--action_horizon", type=int, default=8)
    parser.add_argument("--action_skip", type=int, default=0)
    parser.add_argument("--max_timesteps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--no_ema", action="store_false", dest="use_ema")
    parser.add_argument("--send_guidance_report", action="store_true")
    parser.add_argument("--disable_guidance", action="store_true",
                        help="Run the same board DP/Foresight serving stack without latent-energy guidance.")
    parser.add_argument("--server_rollout_log_dir", default=DEFAULT_ROLLOUT_LOG_DIR,
                        help="Root for server-side force_trace.csv logs.")
    parser.add_argument("--disable_server_rollout_log", action="store_true",
                        help="Disable server-side saving of each real rollout trajectory/force trace.")
    parser.add_argument("--dry_run_dataset_smoke", action="store_true")
    parser.add_argument("--dataset_dirs", default=None)
    parser.add_argument("--smoke_output", default=DEFAULT_SMOKE_OUTPUT)
    parser.add_argument("--min_finite_grad_rate", type=float, default=0.999)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.dry_run_dataset_smoke:
        dry_run_dataset_smoke(args)
    else:
        run_server(args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
