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
from for_show_xiaomi.ws_server import ClientDisconnected, TactileACTServer  # noqa: E402
from TFAC_V5.board_latent_energy.runtime import BoardLatentEnergyRuntime  # noqa: E402
from TFAC_V5.pretrain_latent_foresight_multistep import MultiStepLatentForesightModel  # noqa: E402
from utils import set_seed  # noqa: E402


DEFAULT_DP_CKPT = (
    "/home/chenshuai/Project/output/"
    "dp_tac_concat_board_260609_260610_left_boardvae_rawimg200x266_ph16_oh2_e1000/"
    "dp_best.pth"
)
DEFAULT_FORESIGHT_DIR = (
    "/home/chenshuai/Project/output/foresight_ckpt/"
    "latent_foresight_board_260609_260610_multistep16_boardvae_marker_only_e100_bs16_preload"
)
DEFAULT_SCORER_CKPT = (
    "/home/chenshuai/Project/output/board_latent_energy/ce_margin_e10/"
    "board_latent_energy_best.pt"
)
DEFAULT_SMOKE_OUTPUT = (
    "/home/chenshuai/Project/output/board_dp_foresight_guided_server/"
    "dataset_smoke.json"
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


def scorer_normalize_action(
    scorer: BoardLatentEnergyRuntime,
    action_raw: torch.Tensor,
) -> torch.Tensor:
    return (action_raw - scorer.action_mean) / scorer.action_std.clamp_min(1e-6)


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
    model = MultiStepLatentForesightModel(
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
    missing, unexpected = model.load_state_dict(state, strict=False)
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
            f"foresight_horizon={self.horizon}, guidance_path={args.guidance_path}, "
            f"scale={args.guidance_scale}, steps={args.guidance_steps}"
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
        dp_vae = os.path.abspath(str(self.config.get("vae_checkpoint")))
        fs_vae = os.path.abspath(str(fs_config.get("tactile_vae_ckpt")))
        if int(fs_config.get("state_dim", self.action_dim)) != self.action_dim:
            raise ValueError("DP action_dim and Foresight state_dim mismatch")
        if self.pred_horizon < self.horizon:
            raise ValueError(
                f"DP pred_horizon={self.pred_horizon} shorter than foresight horizon={self.horizon}"
            )
        if int(self.scorer.model.chunk_len) != self.horizon:
            raise ValueError(
                f"Scorer chunk_len={self.scorer.model.chunk_len} != foresight horizon={self.horizon}"
            )
        if int(self.scorer.model.action_dim) != self.action_dim:
            raise ValueError("Scorer action_dim and DP action_dim mismatch")
        if dp_vae != fs_vae:
            print("[board-guided][warn] DP and Foresight point to different TactileVAE checkpoints.")
        if bool(fs_config.get("use_state_trajectory", False)) and self.args.alignment == "none":
            print("[board-guided][warn] Foresight uses qpos[t+1:t+H+1]; alignment=none may be off by one.")

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
                    idx = marker_end - self.tac_history + 1 + k
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
        guidance_path: str,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        pred = self.foresight_predict_latent(action_aligned_raw, context)
        score_action = action_aligned_raw
        score_latent = pred
        if guidance_path == "latent_only":
            score_action = score_action.detach()
        elif guidance_path == "action_only":
            score_latent = score_latent.detach()
        elif guidance_path != "full":
            raise ValueError(f"Unknown guidance_path: {guidance_path}")
        out = self.scorer(score_action, score_latent, normalized=False)
        return out[self.args.score_mode], out

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
                action_aligned = tensor_align_action(x0_raw, self.horizon, self.args.alignment)
                score, out = self.score_from_foresight(
                    action_aligned,
                    context,
                    guidance_path=self.args.guidance_path,
                )
                objective = score.mean()
                smooth_penalty = torch.zeros((), device=self.device)
                if self.args.lambda_smooth > 0:
                    action_norm = scorer_normalize_action(self.scorer, action_aligned)
                    smooth_penalty = (action_norm[:, 1:] - action_norm[:, :-1]).pow(2).mean()
                    objective = objective - float(self.args.lambda_smooth) * smooth_penalty

                grad = torch.autograd.grad(objective, action_for_grad, retain_graph=False)[0]
                action, log = self.apply_guidance_update(action_for_grad.detach(), grad.detach())

                with torch.no_grad():
                    guided_x0 = predict_x0_from_eps(action, eps.detach(), t, alphas, clip=True)
                    guided_raw = minmax_denorm(guided_x0, self.action_min, self.action_max)
                    guided_aligned = tensor_align_action(guided_raw, self.horizon, self.args.alignment)
                    _, after_out = self.score_from_foresight(
                        guided_aligned,
                        context,
                        guidance_path="full",
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
            "guidance_path": self.args.guidance_path,
            "guidance_scale": float(self.args.guidance_scale),
            "guidance_steps": int(self.args.guidance_steps),
            "score_mode": self.args.score_mode,
            "alignment": self.args.alignment,
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
        obs_indices = [max(0, start_t - stack.obs_horizon + 1 + k) for k in range(stack.obs_horizon)]
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
        aligned_raw = tensor_align_action(action_raw, stack.horizon, args.alignment)
        with torch.no_grad():
            pred = stack.foresight_predict_latent(aligned_raw, context)
            score_out = stack.scorer(aligned_raw, pred, normalized=False)
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
    server = TactileACTServer(
        host=args.host,
        port=args.port,
        metadata={
            "protocol": "board_dp_foresight_guided",
            "variant": stack.variant,
            "camera_names": stack.camera_names,
            "action_dim": stack.action_dim,
            "pred_horizon": stack.pred_horizon,
            "action_skip": action_skip,
            "action_horizon": action_horizon,
            "guidance": {
                "location": "inside_ddpm_denoising",
                "path": args.guidance_path,
                "scale": args.guidance_scale,
                "steps": args.guidance_steps,
                "score_mode": args.score_mode,
                "alignment": args.alignment,
            },
        },
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

            obs_buffer: deque[Dict[str, Any]] = deque(maxlen=stack.obs_horizon)
            marker_buffer: List[np.ndarray] = []
            marker_step = 0
            action_chunk: torch.Tensor | None = None
            last_report: Dict[str, Any] | None = None

            try:
                for step in range(int(args.max_timesteps)):
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
                        context = stack.make_foresight_context(processed["qpos_raw"], marker_buffer)
                        action_chunk, last_report = stack.guided_ddpm_inference(obs_cond, context)
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
                    }
                    if last_report is not None and args.send_guidance_report:
                        msg["guidance_report"] = last_report
                    server.send_action(msg)
                    if step + 1 < int(args.max_timesteps):
                        obs = server.recv_obs()
            except ClientDisconnected:
                print(f"[board-guided] client disconnected at step {step}")
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
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--guidance_steps", type=int, default=5)
    parser.add_argument("--guidance_scale", type=float, default=0.003)
    parser.add_argument("--guidance_path", default="latent_only", choices=["latent_only", "full", "action_only"])
    parser.add_argument("--score_mode", default="expert_margin",
                        choices=["score_good", "expert_margin", "p_expert", "quality_0_100"])
    parser.add_argument("--alignment", default="shift1", choices=["shift1", "none"])
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--normalize_guidance_grad", action="store_true", default=True)
    parser.add_argument("--raw_guidance_grad", action="store_false", dest="normalize_guidance_grad")
    parser.add_argument("--sample_clip", type=float, default=1.5)
    parser.add_argument("--lambda_smooth", type=float, default=0.0)
    parser.add_argument("--action_horizon", type=int, default=8)
    parser.add_argument("--action_skip", type=int, default=0)
    parser.add_argument("--max_timesteps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--no_ema", action="store_false", dest="use_ema")
    parser.add_argument("--send_guidance_report", action="store_true")
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
