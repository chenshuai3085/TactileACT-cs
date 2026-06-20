"""
Force-aware multi-step tactile foresight pretraining.

This file is a parallel path to ``pretrain_latent_foresight_multistep.py``.
It keeps the marker-latent prediction objective, then adds force/contact heads:

    history marker + qpos + candidate future qpos/action chunk
        -> future marker latent sequence
        -> decoded future marker sequence
        -> future force proxy
        -> force-band logits: too_small / good / too_large / oscillate
        -> contact gate logits

The purpose is not to replace the current marker-only Foresight.  The purpose is
to provide a differentiable consequence model whose score is aligned with board
wiping quality: force magnitude in a good band and temporally smooth contact.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TFAC_V5.foresight_multistep import MultiStepSpatialForesightTransformer
from TFAC_V5.tactile_vae import TactileVAE
from utils import NormalizeSeparate, set_seed


DEFAULT_DATASETS = {
    "positive_old": "/media/chenshuai/EXTERNAL_USB/pih_dataset/260609/wipe_pos_straight_z124_125_150_20260609",
    "positive_260617": "/media/chenshuai/EXTERNAL_USB/pih_dataset/260617_v8l_caheiban/peg_in_hole_0617",
    "too_small": "/media/chenshuai/EXTERNAL_USB/pih_dataset/260609/z_too_high",
    "too_large": "/media/chenshuai/EXTERNAL_USB/pih_dataset/260610/z_too_low",
    "oscillate": "/media/chenshuai/EXTERNAL_USB/pih_dataset/260610/z_too_oscillate",
}

REASON_TO_ID = {"too_small": 0, "good": 1, "too_large": 2, "oscillate": 3}
LABEL_TO_REASON = {
    "positive_old": REASON_TO_ID["good"],
    "positive_260617": REASON_TO_ID["good"],
    "too_small": REASON_TO_ID["too_small"],
    "too_large": REASON_TO_ID["too_large"],
    "oscillate": REASON_TO_ID["oscillate"],
}


@dataclass
class EpisodeRef:
    path: str
    label: str


def scan_episode_paths(dataset_dir: str) -> List[str]:
    direct = sorted(glob.glob(os.path.join(dataset_dir, "episode_*.hdf5")))
    if direct:
        return direct
    return sorted(glob.glob(os.path.join(dataset_dir, "**", "episode_*.hdf5"), recursive=True))


def parse_dataset_roots(config: Dict) -> Dict[str, str]:
    if "labeled_dataset_dirs" in config:
        return {str(k): str(v) for k, v in config["labeled_dataset_dirs"].items()}
    if "dataset_dirs" in config:
        roots = {}
        for item in config["dataset_dirs"]:
            if isinstance(item, dict):
                roots[str(item["label"])] = str(item["path"])
            else:
                raise ValueError("dataset_dirs entries must be {label, path} for force-aware training")
        return roots
    return DEFAULT_DATASETS.copy()


def scan_labeled_episodes(dataset_roots: Dict[str, str]) -> List[EpisodeRef]:
    episodes: List[EpisodeRef] = []
    for label, root in dataset_roots.items():
        if label not in LABEL_TO_REASON:
            raise ValueError(f"Unknown label {label!r}; expected one of {sorted(LABEL_TO_REASON)}")
        paths = scan_episode_paths(root)
        print(f"{label}: {len(paths)} episodes from {root}")
        episodes.extend(EpisodeRef(path=p, label=label) for p in paths)
    if not episodes:
        raise RuntimeError("No episode_*.hdf5 files found")
    return episodes


def load_hdf5_array(root: h5py.File, key: str) -> np.ndarray:
    if key in root:
        return root[key][()]
    if key.startswith("/") and key[1:] in root:
        return root[key[1:]][()]
    raise KeyError(key)


def infer_meta(path: str, config: Dict) -> Dict:
    with h5py.File(path, "r") as f:
        proprio_key = str(config.get("proprio_key", "proprio_joint"))
        action_key = str(config.get("action_key", "actions/joint_abs"))
        tac_side = str(config.get("tac_side", "left"))
        if f"observations/{proprio_key}" not in f:
            for candidate in ("proprio_joint", "qpos", "proprio_eef"):
                if f"observations/{candidate}" in f:
                    proprio_key = candidate
                    break
        if action_key not in f:
            action_key = "actions/joint_abs" if "actions/joint_abs" in f else "action"
        state_dim = int(f[f"observations/{proprio_key}"].shape[-1])
        action_dim = int(f[action_key].shape[-1])
        force_key = str(config.get("force_key", f"observations/tac/{tac_side}/force6d"))
        if force_key not in f and "ft" in f:
            force_key = "ft"
    return {
        "proprio_key": proprio_key,
        "action_key": action_key,
        "force_key": force_key,
        "tac_side": tac_side,
        "state_dim": state_dim,
        "action_dim": action_dim,
    }


def compute_norm_stats(
    episodes: Iterable[EpisodeRef],
    proprio_key: str,
    action_key: str,
    marker_key: str,
    force_key: str,
    max_episodes: int = 200,
) -> Dict[str, np.ndarray]:
    refs = list(episodes)
    if len(refs) > max_episodes:
        idx = np.linspace(0, len(refs) - 1, max_episodes, dtype=int)
        refs = [refs[i] for i in idx]
    qpos_list, action_list, marker_list, force_list = [], [], [], []
    for ref in refs:
        try:
            with h5py.File(ref.path, "r") as f:
                qpos_list.append(load_hdf5_array(f, f"observations/{proprio_key}").astype(np.float32))
                action_list.append(load_hdf5_array(f, action_key).astype(np.float32))
                marker_list.append(load_hdf5_array(f, marker_key).astype(np.float32))
                force_list.append(load_hdf5_array(f, force_key).astype(np.float32))
        except (OSError, KeyError):
            continue
    if not qpos_list:
        raise RuntimeError("Could not compute normalization stats; no readable episodes")
    qpos = np.concatenate(qpos_list, axis=0)
    action = np.concatenate(action_list, axis=0)
    marker = np.concatenate(marker_list, axis=0)
    force = np.concatenate(force_list, axis=0)
    return {
        "qpos_mean": qpos.mean(axis=0).astype(np.float32),
        "qpos_std": np.clip(qpos.std(axis=0), 1e-4, None).astype(np.float32),
        "action_mean": action.mean(axis=0).astype(np.float32),
        "action_std": np.clip(action.std(axis=0), 1e-4, None).astype(np.float32),
        "marker_offset_mean": marker.mean(axis=(0, 1, 2)).astype(np.float32),
        "marker_offset_std": np.clip(marker.std(axis=(0, 1, 2)), 1e-4, None).astype(np.float32),
        "force_mean": force.mean(axis=0).astype(np.float32),
        "force_std": np.clip(force.std(axis=0), 1e-4, None).astype(np.float32),
    }


def force_proxy_np(force_window: np.ndarray) -> np.ndarray:
    """Continuous force targets per future step.

    Input is ``(H, 6)``.  Output is ``(H, 6)``:
    force magnitude, fz, translational force delta, |fz delta|, torque magnitude,
    and translational force jerk proxy.
    """

    force = force_window.astype(np.float32)
    trans = force[:, :3]
    torque = force[:, 3:6] if force.shape[-1] >= 6 else np.zeros_like(trans)
    mag = np.linalg.norm(trans, axis=-1)
    fz = trans[:, 2]
    torque_mag = np.linalg.norm(torque, axis=-1)
    if len(force) > 1:
        d_trans = np.linalg.norm(np.diff(trans, axis=0, prepend=trans[:1]), axis=-1)
        d_fz = np.abs(np.diff(fz, prepend=fz[:1]))
        jerk = np.abs(np.diff(d_trans, prepend=d_trans[:1]))
    else:
        d_trans = np.zeros_like(mag)
        d_fz = np.zeros_like(mag)
        jerk = np.zeros_like(mag)
    return np.stack([mag, fz, d_trans, d_fz, torque_mag, jerk], axis=-1).astype(np.float32)


def marker_contact_signal(marker: np.ndarray) -> np.ndarray:
    mag = np.linalg.norm(marker.reshape(marker.shape[0], -1, 2), axis=-1)
    return mag.mean(axis=-1).astype(np.float32)


def build_force_reference(episodes: List[EpisodeRef], meta: Dict, config: Dict) -> Dict:
    """Derive robust positive-set force/contact references for labels and scaling."""

    max_eps = int(config.get("force_ref_max_episodes", 80))
    force_key = meta["force_key"]
    marker_key = f"observations/tac/{meta['tac_side']}/marker_offset"
    pos_mag, pos_delta, pos_contact = [], [], []
    all_mag, all_contact = [], []
    for ref in episodes:
        if not ref.label.startswith("positive") and len(all_mag) > max_eps:
            continue
        try:
            with h5py.File(ref.path, "r") as f:
                force = load_hdf5_array(f, force_key).astype(np.float32)
                marker = load_hdf5_array(f, marker_key).astype(np.float32)
        except (OSError, KeyError):
            continue
        length = min(len(force), len(marker))
        if length < 4:
            continue
        start = int(length * float(config.get("phase_start_frac", 0.25)))
        end = int(length * float(config.get("phase_end_frac", 0.92)))
        force = force[start:end]
        marker = marker[start:end]
        proxy = force_proxy_np(force)
        contact = marker_contact_signal(marker)
        all_mag.append(proxy[:, 0])
        all_contact.append(contact)
        if ref.label.startswith("positive"):
            pos_mag.append(proxy[:, 0])
            pos_delta.append(proxy[:, 2])
            pos_contact.append(contact)
        if len(pos_mag) >= max_eps:
            break

    if not pos_mag:
        raise RuntimeError("No positive episodes available for force reference")
    pos_mag_arr = np.concatenate(pos_mag)
    pos_delta_arr = np.concatenate(pos_delta)
    pos_contact_arr = np.concatenate(pos_contact)
    all_mag_arr = np.concatenate(all_mag) if all_mag else pos_mag_arr
    all_contact_arr = np.concatenate(all_contact) if all_contact else pos_contact_arr

    center = float(np.median(pos_mag_arr))
    mad = float(np.median(np.abs(pos_mag_arr - center)))
    sigma = max(1.4826 * mad, float(np.std(pos_mag_arr)), 0.5)
    low = float(config.get("force_good_low", center - 1.25 * sigma))
    high = float(config.get("force_good_high", center + 1.25 * sigma))
    delta_hi = float(config.get("force_delta_oscillate_q", np.quantile(pos_delta_arr, 0.90) * 2.0))
    contact_thr = float(config.get("contact_threshold", max(np.quantile(pos_contact_arr, 0.20), np.quantile(all_contact_arr, 0.55))))
    force_contact_thr = float(config.get("force_contact_threshold", max(np.quantile(pos_mag_arr, 0.20), np.quantile(all_mag_arr, 0.55))))

    return {
        "force_mag_center": center,
        "force_mag_sigma": sigma,
        "force_good_low": low,
        "force_good_high": high,
        "force_delta_oscillate": max(delta_hi, 1e-4),
        "marker_contact_threshold": contact_thr,
        "force_contact_threshold": force_contact_thr,
    }


class ForceAwareForesightDataset(Dataset):
    def __init__(
        self,
        episodes: List[EpisodeRef],
        norm_stats: Dict,
        meta: Dict,
        force_ref: Dict,
        chunk_size: int = 16,
        horizon: int = 16,
        tactile_vae_window: int = 8,
        samples_per_episode: int = 30,
        phase_start_frac: float = 0.20,
        phase_end_frac: float = 0.95,
        contact_only: bool = True,
        use_state_trajectory: bool = True,
        preload: bool = False,
    ):
        self.episodes = episodes
        self.norm = NormalizeSeparate(norm_stats)
        self.norm_stats = norm_stats
        self.meta = meta
        self.force_ref = force_ref
        self.chunk_size = int(chunk_size)
        self.horizon = int(horizon)
        self.tactile_vae_window = int(tactile_vae_window)
        self.samples_per_episode = int(samples_per_episode)
        self.phase_start_frac = float(phase_start_frac)
        self.phase_end_frac = float(phase_end_frac)
        self.contact_only = bool(contact_only)
        self.use_state_trajectory = bool(use_state_trajectory)
        self.marker_key = f"observations/tac/{meta['tac_side']}/marker_offset"
        self.cache: Dict[str, Dict[str, np.ndarray]] = {}
        if preload:
            self._preload()

    def _preload(self) -> None:
        print(f"Preloading force-aware foresight dataset: {len(self.episodes)} episodes")
        skipped = 0
        for ref in tqdm(self.episodes, desc="force-aware preload"):
            try:
                self.cache[ref.path] = self._read_episode(ref.path)
            except (OSError, KeyError):
                skipped += 1
        if skipped:
            print(f"WARNING: skipped {skipped} unreadable episodes during preload")

    def _read_episode(self, path: str) -> Dict[str, np.ndarray]:
        with h5py.File(path, "r") as f:
            marker = load_hdf5_array(f, self.marker_key).astype(np.float32)
            qpos = load_hdf5_array(f, f"observations/{self.meta['proprio_key']}").astype(np.float32)
            action = load_hdf5_array(f, self.meta["action_key"]).astype(np.float32)
            force = load_hdf5_array(f, self.meta["force_key"]).astype(np.float32)
        length = min(len(marker), len(qpos), len(action), len(force))
        return {
            "marker": marker[:length],
            "qpos": qpos[:length],
            "action": action[:length],
            "force": force[:length],
        }

    def __len__(self) -> int:
        return max(1, len(self.episodes) * self.samples_per_episode)

    def _episode(self, ref: EpisodeRef) -> Dict[str, np.ndarray]:
        if ref.path in self.cache:
            return self.cache[ref.path]
        return self._read_episode(ref.path)

    def _choose_start(self, ep: Dict[str, np.ndarray]) -> int:
        length = len(ep["marker"])
        lo = max(self.tactile_vae_window - 1, int(length * self.phase_start_frac))
        hi = min(int(length * self.phase_end_frac), length - self.horizon - 1, length - self.chunk_size - 1)
        if hi <= lo:
            return max(self.tactile_vae_window - 1, min(length - 2, lo))
        candidates = np.arange(lo, hi, dtype=np.int64)
        if self.contact_only and len(candidates) > 0:
            marker_sig = marker_contact_signal(ep["marker"])
            force_mag = np.linalg.norm(ep["force"][:, :3], axis=-1)
            contact = (
                (marker_sig[candidates] >= self.force_ref["marker_contact_threshold"])
                | (force_mag[candidates] >= self.force_ref["force_contact_threshold"])
            )
            if np.any(contact):
                candidates = candidates[contact]
        return int(np.random.choice(candidates))

    @staticmethod
    def _pad_chunk(arr: np.ndarray, start: int, length: int) -> Tuple[np.ndarray, int]:
        end = min(len(arr), start + length)
        chunk = arr[start:end]
        valid = len(chunk)
        if valid <= 0:
            idx = max(0, min(start, len(arr) - 1))
            chunk = arr[idx : idx + 1]
            valid = 1
        if valid < length:
            chunk = np.concatenate([chunk, np.repeat(chunk[-1:], length - valid, axis=0)], axis=0)
        return chunk.astype(np.float32), valid

    def _marker_window(self, marker: np.ndarray, end_t: int) -> np.ndarray:
        frames = []
        for i in range(self.tactile_vae_window):
            ts = max(0, end_t - (self.tactile_vae_window - 1 - i))
            frame = marker[ts]
            frame = (frame - self.norm_stats["marker_offset_mean"]) / self.norm_stats["marker_offset_std"]
            frames.append(frame.astype(np.float32))
        return np.stack(frames, axis=0)

    def _future_marker_windows(self, marker: np.ndarray, start_t: int) -> np.ndarray:
        return np.stack([self._marker_window(marker, min(start_t + h, len(marker) - 1)) for h in range(1, self.horizon + 1)], axis=0)

    def _future_force(self, force: np.ndarray, start_t: int) -> np.ndarray:
        frames = []
        for h in range(1, self.horizon + 1):
            frames.append(force[min(start_t + h, len(force) - 1)])
        return np.stack(frames, axis=0).astype(np.float32)

    def _targets_from_force_marker(self, force_future: np.ndarray, marker_future_raw: np.ndarray, label: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        proxy = force_proxy_np(force_future)
        force_mag = proxy[:, 0]
        force_delta = proxy[:, 2]
        marker_sig = marker_contact_signal(marker_future_raw[:, -1])
        contact = (
            (marker_sig >= self.force_ref["marker_contact_threshold"])
            | (force_mag >= self.force_ref["force_contact_threshold"])
        ).astype(np.float32)

        reason = np.full((self.horizon,), LABEL_TO_REASON[label], dtype=np.int64)
        too_small = force_mag < self.force_ref["force_good_low"]
        too_large = force_mag > self.force_ref["force_good_high"]
        oscillate = force_delta > self.force_ref["force_delta_oscillate"]
        if label.startswith("positive"):
            reason[:] = REASON_TO_ID["good"]
            reason[too_small & (contact > 0.5)] = REASON_TO_ID["too_small"]
            reason[too_large & (contact > 0.5)] = REASON_TO_ID["too_large"]
            reason[oscillate & (contact > 0.5)] = REASON_TO_ID["oscillate"]
        return proxy.astype(np.float32), reason, contact

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        ref = self.episodes[index % len(self.episodes)]
        try:
            ep = self._episode(ref)
        except (OSError, KeyError):
            return self.__getitem__(np.random.randint(len(self.episodes)))

        start = self._choose_start(ep)
        marker_hist = self._marker_window(ep["marker"], start)
        future_marker = self._future_marker_windows(ep["marker"], start)
        future_force = self._future_force(ep["force"], start)
        raw_future_marker = np.stack([
            np.stack([ep["marker"][max(0, min(start + h, len(ep["marker"]) - 1) - (self.tactile_vae_window - 1 - i))]
                      for i in range(self.tactile_vae_window)], axis=0)
            for h in range(1, self.horizon + 1)
        ], axis=0).astype(np.float32)
        force_proxy, force_band, contact = self._targets_from_force_marker(future_force, raw_future_marker, ref.label)

        qpos = ep["qpos"][start]
        if self.use_state_trajectory:
            action_raw, valid = self._pad_chunk(ep["qpos"], start + 1, self.chunk_size)
            qpos_norm, action_norm = self.norm(qpos=qpos, action=action_raw, action_as_qpos=True)
        else:
            action_raw, valid = self._pad_chunk(ep["action"], start, self.chunk_size)
            qpos_norm, action_norm = self.norm(qpos=qpos, action=action_raw)
        is_pad = np.zeros((self.chunk_size,), dtype=bool)
        is_pad[valid:] = True

        force_proxy_norm = force_proxy.copy()
        force_proxy_norm[:, 0] = (force_proxy[:, 0] - self.force_ref["force_mag_center"]) / self.force_ref["force_mag_sigma"]
        force_proxy_norm[:, 1] = (force_proxy[:, 1] - self.norm_stats["force_mean"][2]) / self.norm_stats["force_std"][2]
        force_proxy_norm[:, 2] = force_proxy[:, 2] / self.force_ref["force_delta_oscillate"]
        force_proxy_norm[:, 3] = force_proxy[:, 3] / self.force_ref["force_delta_oscillate"]
        force_proxy_norm[:, 4] = force_proxy[:, 4] / max(float(np.linalg.norm(self.norm_stats["force_std"][3:6])), 1e-4)
        force_proxy_norm[:, 5] = force_proxy[:, 5] / self.force_ref["force_delta_oscillate"]

        return {
            "marker_hist": torch.from_numpy(marker_hist).float(),
            "qpos": torch.from_numpy(qpos_norm).float(),
            "action": torch.from_numpy(action_norm).float(),
            "is_pad": torch.from_numpy(is_pad),
            "future_marker": torch.from_numpy(future_marker).float(),
            "future_force_proxy": torch.from_numpy(force_proxy_norm).float(),
            "future_force_band": torch.from_numpy(force_band).long(),
            "future_contact": torch.from_numpy(contact).float(),
            "label_id": torch.tensor(LABEL_TO_REASON[ref.label], dtype=torch.long),
        }


class ForceAwareMultiStepForesightModel(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        foresight_layers: int = 3,
        foresight_nheads: int = 8,
        foresight_dim_feedforward: int = 2048,
        dropout: float = 0.1,
        tactile_vae_ckpt: Optional[str] = None,
        tactile_vae_latent_dim: int = 16,
        predict_horizon: int = 16,
        tactile_vae_window: int = 8,
        force_proxy_dim: int = 6,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.predict_horizon = predict_horizon
        self.tactile_vae_latent_dim = tactile_vae_latent_dim
        self.n_tactile_spatial = 9
        self.tactile_vae = TactileVAE(latent_dim=tactile_vae_latent_dim, temporal_window=tactile_vae_window)
        if tactile_vae_ckpt and os.path.exists(tactile_vae_ckpt):
            ckpt = torch.load(tactile_vae_ckpt, map_location="cpu")
            state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            self.tactile_vae.load_state_dict(state)
            print(f"Loaded TactileVAE from: {tactile_vae_ckpt}")
        self.tactile_vae.requires_grad_(False)

        self.tac_latent_proj = nn.Linear(tactile_vae_latent_dim, hidden_dim)
        self.tac_spatial_pos_embed = nn.Parameter(torch.randn(9, 1, hidden_dim) * 0.02)

        self.foresight = MultiStepSpatialForesightTransformer(
            d_model=hidden_dim,
            action_dim=action_dim,
            num_layers=foresight_layers,
            nhead=foresight_nheads,
            dim_feedforward=foresight_dim_feedforward,
            dropout=dropout,
            latent_dim=tactile_vae_latent_dim,
            n_tactile_spatial=self.n_tactile_spatial,
            predict_horizon=predict_horizon,
            max_history=1,
            state_dim=state_dim,
        )
        self.force_proxy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, force_proxy_dim),
        )
        self.force_band_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 4),
        )
        self.contact_gate_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def encode_marker(self, marker_hist: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            z_t, _ = self.tactile_vae.encode_single_frame(marker_hist)
        bsz = z_t.shape[0]
        z_current = z_t.reshape(bsz, -1)
        z_tokens = z_t.reshape(bsz, self.tactile_vae_latent_dim, 9).permute(2, 0, 1)
        t_tokens = self.tac_latent_proj(z_tokens) + self.tac_spatial_pos_embed
        return t_tokens, z_current

    def encode_future_marker(self, future_marker: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, horizon = future_marker.shape[:2]
        z_gt = []
        with torch.no_grad():
            for h in range(horizon):
                z_h, _ = self.tactile_vae.encode_single_frame(future_marker[:, h])
                z_gt.append(z_h.reshape(bsz, -1))
        return torch.stack(z_gt, dim=1), future_marker[:, :, -1]

    def decode_latent_sequence(self, z_seq: torch.Tensor) -> torch.Tensor:
        bsz, horizon, _ = z_seq.shape
        channels = self.tactile_vae_latent_dim
        z_spatial = z_seq.reshape(bsz * horizon, channels, 3, 3)
        marker = self.tactile_vae.decoder(z_spatial)
        return marker.reshape(bsz, horizon, 9, 9, 2)

    def forward(self, marker_hist: torch.Tensor, qpos: torch.Tensor, action: torch.Tensor, future_marker: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        t_tokens, z_current = self.encode_marker(marker_hist)
        empty_v = torch.empty(0, marker_hist.shape[0], self.hidden_dim, device=marker_hist.device, dtype=marker_hist.dtype)
        t_hat, _, embed = self.foresight(empty_v, t_tokens, action, n_v=0, proprio=qpos)
        out = {
            "z_pred": t_hat,
            "z_current": z_current,
            "force_proxy_pred": self.force_proxy_head(embed),
            "force_band_logits": self.force_band_head(embed),
            "contact_logits": self.contact_gate_head(embed).squeeze(-1),
        }
        if future_marker is not None:
            z_gt, marker_gt = self.encode_future_marker(future_marker)
            out["z_gt"] = z_gt
            out["marker_gt"] = marker_gt
            out["marker_pred"] = self.decode_latent_sequence(t_hat)
        return out


def force_aware_loss(out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], model: ForceAwareMultiStepForesightModel, weights: Dict[str, float]) -> Tuple[torch.Tensor, Dict[str, float]]:
    z_pred = out["z_pred"]
    z_gt = out["z_gt"]
    marker_pred = out["marker_pred"]
    marker_gt = out["marker_gt"]
    force_proxy_gt = batch["future_force_proxy"]
    force_band_gt = batch["future_force_band"]
    contact_gt = batch["future_contact"]

    latent_seq = F.smooth_l1_loss(z_pred, z_gt)
    latent_final = F.smooth_l1_loss(z_pred[:, -1], z_gt[:, -1])
    marker = F.smooth_l1_loss(marker_pred, marker_gt)
    if z_pred.size(1) > 1:
        delta = F.smooth_l1_loss(z_pred[:, 1:] - z_pred[:, :-1], z_gt[:, 1:] - z_gt[:, :-1])
        force_smooth = F.smooth_l1_loss(
            out["force_proxy_pred"][:, 1:, :3] - out["force_proxy_pred"][:, :-1, :3],
            force_proxy_gt[:, 1:, :3] - force_proxy_gt[:, :-1, :3],
        )
    else:
        delta = torch.zeros((), device=z_pred.device)
        force_smooth = torch.zeros((), device=z_pred.device)

    force_proxy = F.smooth_l1_loss(out["force_proxy_pred"], force_proxy_gt)
    band = F.cross_entropy(out["force_band_logits"].reshape(-1, 4), force_band_gt.reshape(-1))
    contact = F.binary_cross_entropy_with_logits(out["contact_logits"], contact_gt)
    total = (
        latent_seq
        + weights["final"] * latent_final
        + weights["marker"] * marker
        + weights["delta"] * delta
        + weights["force"] * force_proxy
        + weights["band"] * band
        + weights["contact"] * contact
        + weights["smooth"] * force_smooth
    )
    metrics = {
        "total": float(total.detach().cpu()),
        "latent_seq": float(latent_seq.detach().cpu()),
        "latent_final": float(latent_final.detach().cpu()),
        "marker": float(marker.detach().cpu()),
        "delta": float(delta.detach().cpu()),
        "force": float(force_proxy.detach().cpu()),
        "band": float(band.detach().cpu()),
        "contact": float(contact.detach().cpu()),
        "force_smooth": float(force_smooth.detach().cpu()),
    }
    return total, metrics


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def aggregate(dst: Dict[str, float], metrics: Dict[str, float], n: int) -> None:
    for key, value in metrics.items():
        dst[key] = dst.get(key, 0.0) + float(value) * n


def average(src: Dict[str, float], n: int) -> Dict[str, float]:
    return {k: v / max(n, 1) for k, v in src.items()}


def classification_metrics(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> Dict[str, float]:
    pred = logits.argmax(dim=-1).reshape(-1)
    lab = labels.reshape(-1)
    acc = (pred == lab).float().mean().item()
    recalls = []
    for cls in range(num_classes):
        mask = lab == cls
        if mask.any():
            recalls.append((pred[mask] == cls).float().mean().item())
    return {"acc": acc, "balanced_acc": float(np.mean(recalls)) if recalls else 0.0}


def eval_extra(model: ForceAwareMultiStepForesightModel, loader: DataLoader, device: torch.device, max_batches: int = 0) -> Dict[str, float]:
    model.eval()
    force_abs, force_count = 0.0, 0
    contact_correct, contact_count = 0.0, 0
    band_logits, band_labels = [], []
    with torch.no_grad():
        for idx, raw_batch in enumerate(loader):
            if max_batches > 0 and idx >= max_batches:
                break
            batch = to_device(raw_batch, device)
            out = model(batch["marker_hist"], batch["qpos"], batch["action"], future_marker=batch["future_marker"])
            force_abs += torch.abs(out["force_proxy_pred"] - batch["future_force_proxy"]).sum().item()
            force_count += int(np.prod(batch["future_force_proxy"].shape))
            contact_pred = torch.sigmoid(out["contact_logits"]) >= 0.5
            contact_correct += (contact_pred == (batch["future_contact"] >= 0.5)).float().sum().item()
            contact_count += int(np.prod(batch["future_contact"].shape))
            band_logits.append(out["force_band_logits"].detach().cpu())
            band_labels.append(batch["future_force_band"].detach().cpu())
    if band_logits:
        cls = classification_metrics(torch.cat(band_logits, dim=0), torch.cat(band_labels, dim=0), 4)
    else:
        cls = {"acc": 0.0, "balanced_acc": 0.0}
    return {
        "force_proxy_mae": force_abs / max(force_count, 1),
        "contact_acc": contact_correct / max(contact_count, 1),
        "band_acc": cls["acc"],
        "band_balanced_acc": cls["balanced_acc"],
    }


def plot_history(history: Dict[str, List[float]], path: str) -> None:
    keys = ["total", "latent_seq", "marker", "delta", "force", "band", "contact"]
    fig, axes = plt.subplots(1, len(keys), figsize=(4.5 * len(keys), 4))
    for ax, key in zip(axes, keys):
        tr = f"train_{key}"
        va = f"val_{key}"
        if tr in history:
            ax.plot(history[tr], label="train")
        if va in history:
            ax.plot(history[va], label="val")
        ax.set_title(key)
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def split_episodes(episodes: List[EpisodeRef], train_ratio: float, seed: int) -> Tuple[List[EpisodeRef], List[EpisodeRef]]:
    rng = np.random.default_rng(seed)
    train, val = [], []
    for label in sorted(set(e.label for e in episodes)):
        group = [e for e in episodes if e.label == label]
        order = rng.permutation(len(group))
        n_train = max(1, int(len(group) * train_ratio))
        if n_train >= len(group):
            n_train = len(group) - 1
        for i, idx in enumerate(order):
            (train if i < n_train else val).append(group[int(idx)])
    return train, val


def run(config: Dict) -> None:
    seed = int(config.get("seed", 42))
    set_seed(seed)
    gpu = int(config.get("gpu", 0))
    if gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset_roots = parse_dataset_roots(config)
    episodes = scan_labeled_episodes(dataset_roots)
    meta = infer_meta(episodes[0].path, config)
    marker_key = f"observations/tac/{meta['tac_side']}/marker_offset"
    norm_stats = compute_norm_stats(
        episodes,
        meta["proprio_key"],
        meta["action_key"],
        marker_key,
        meta["force_key"],
        max_episodes=int(config.get("norm_max_episodes", 200)),
    )
    stats_path = config.get("tactile_vae_stats")
    if stats_path and os.path.exists(stats_path):
        with open(stats_path) as f:
            vae_stats = json.load(f)
        norm_stats["marker_offset_mean"] = np.asarray(vae_stats["mean"], dtype=np.float32)
        norm_stats["marker_offset_std"] = np.asarray(vae_stats["std"], dtype=np.float32)
        print(f"Loaded TactileVAE marker stats from {stats_path}")
    force_ref = build_force_reference(episodes, meta, config)
    print(f"Force reference: {force_ref}")

    save_root = Path(config.get("save_dir", "/home/chenshuai/Project/output/foresight_ckpt"))
    name = str(config.get("name", "latent_foresight_board_forceaware_multistep16"))
    ckpt_dir = save_root / name
    if ckpt_dir.exists():
        i = 0
        while (save_root / f"{name}_{i}").exists():
            i += 1
        ckpt_dir = save_root / f"{name}_{i}"
    ckpt_dir.mkdir(parents=True, exist_ok=False)

    train_eps, val_eps = split_episodes(episodes, float(config.get("train_ratio", 0.9)), seed)
    print(f"Train episodes: {len(train_eps)}, Val episodes: {len(val_eps)}")

    dataset_kwargs = dict(
        norm_stats=norm_stats,
        meta=meta,
        force_ref=force_ref,
        chunk_size=int(config.get("chunk_size", 16)),
        horizon=int(config.get("predict_horizon", 16)),
        tactile_vae_window=int(config.get("tactile_vae_window", 8)),
        samples_per_episode=int(config.get("samples_per_episode", 30)),
        phase_start_frac=float(config.get("phase_start_frac", 0.20)),
        phase_end_frac=float(config.get("phase_end_frac", 0.95)),
        contact_only=bool(config.get("contact_only", True)),
        use_state_trajectory=bool(config.get("use_state_trajectory", True)),
        preload=bool(config.get("preload", False)),
    )
    train_set = ForceAwareForesightDataset(train_eps, **dataset_kwargs)
    val_set = ForceAwareForesightDataset(val_eps, **dataset_kwargs)
    batch_size = int(config.get("batch_size", 16))
    num_workers = int(config.get("num_workers", 0))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())

    action_dim = meta["state_dim"] if dataset_kwargs["use_state_trajectory"] else meta["action_dim"]
    model = ForceAwareMultiStepForesightModel(
        state_dim=meta["state_dim"],
        action_dim=action_dim,
        hidden_dim=int(config.get("hidden_dim", 512)),
        foresight_layers=int(config.get("foresight_layers", 3)),
        foresight_nheads=int(config.get("foresight_nheads", 8)),
        foresight_dim_feedforward=int(config.get("foresight_dim_feedforward", 2048)),
        dropout=float(config.get("dropout", 0.1)),
        tactile_vae_ckpt=config.get("tactile_vae_ckpt"),
        tactile_vae_latent_dim=int(config.get("tactile_vae_latent_dim", 16)),
        predict_horizon=int(config.get("predict_horizon", 16)),
        tactile_vae_window=int(config.get("tactile_vae_window", 8)),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M / {total_params / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(config.get("lr", 4e-5)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    weights = {
        "final": float(config.get("w_final", 1.0)),
        "marker": float(config.get("w_marker", 0.3)),
        "delta": float(config.get("w_delta", 0.5)),
        "force": float(config.get("w_force", 0.5)),
        "band": float(config.get("w_band", 0.5)),
        "contact": float(config.get("w_contact", 0.2)),
        "smooth": float(config.get("w_smooth", 0.1)),
    }

    saved_config = {
        **config,
        "dataset_roots": dataset_roots,
        "meta": meta,
        "num_episodes": len(episodes),
        "train_episodes": len(train_eps),
        "val_episodes": len(val_eps),
        "force_ref": force_ref,
        "loss_weights": weights,
        "norm_stats": {k: v.tolist() if hasattr(v, "tolist") else v for k, v in norm_stats.items()},
    }
    with open(ckpt_dir / "args.json", "w") as f:
        json.dump(saved_config, f, indent=2)
    with open(ckpt_dir / "dataset_stats.pkl", "wb") as f:
        pickle.dump({"norm_stats": norm_stats, "force_ref": force_ref, "meta": meta}, f)

    history: Dict[str, List[float]] = {}
    best_val = float("inf")
    best_epoch = -1
    epochs = int(config.get("num_epochs", 100))
    log_interval = int(config.get("log_interval", 5))
    save_interval = int(config.get("save_interval", 25))
    max_train_batches = int(config.get("max_train_batches", 0))
    max_val_batches = int(config.get("max_val_batches", 0))

    for epoch in tqdm(range(epochs), desc="force-aware foresight"):
        model.train()
        model.tactile_vae.eval()
        train_sum: Dict[str, float] = {}
        train_count = 0
        for batch_idx, raw_batch in enumerate(train_loader):
            if max_train_batches > 0 and batch_idx >= max_train_batches:
                break
            batch = to_device(raw_batch, device)
            out = model(batch["marker_hist"], batch["qpos"], batch["action"], future_marker=batch["future_marker"])
            loss, metrics = force_aware_loss(out, batch, model, weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(config.get("grad_clip", 1.0)))
            optimizer.step()
            n = batch["action"].shape[0]
            aggregate(train_sum, metrics, n)
            train_count += n
        train_metrics = average(train_sum, train_count)

        model.eval()
        val_sum: Dict[str, float] = {}
        val_count = 0
        with torch.no_grad():
            for batch_idx, raw_batch in enumerate(val_loader):
                if max_val_batches > 0 and batch_idx >= max_val_batches:
                    break
                batch = to_device(raw_batch, device)
                out = model(batch["marker_hist"], batch["qpos"], batch["action"], future_marker=batch["future_marker"])
                _, metrics = force_aware_loss(out, batch, model, weights)
                n = batch["action"].shape[0]
                aggregate(val_sum, metrics, n)
                val_count += n
        val_metrics = average(val_sum, val_count)

        for key in sorted(set(train_metrics) | set(val_metrics)):
            history.setdefault(f"train_{key}", []).append(train_metrics.get(key, 0.0))
            history.setdefault(f"val_{key}", []).append(val_metrics.get(key, 0.0))

        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            best_epoch = epoch
            torch.save({"model_state_dict": model.state_dict(), "config": saved_config, "epoch": epoch, "val": val_metrics}, ckpt_dir / "foresight_force_best.ckpt")

        if epoch % save_interval == 0:
            torch.save({"model_state_dict": model.state_dict(), "config": saved_config, "epoch": epoch, "val": val_metrics}, ckpt_dir / f"foresight_force_epoch_{epoch}.ckpt")

        if epoch % log_interval == 0 or epoch == epochs - 1:
            extra = eval_extra(model, val_loader, device, max_batches=max_val_batches)
            print(
                f"Epoch {epoch}: train={train_metrics['total']:.4f} "
                f"val={val_metrics['total']:.4f} "
                f"force={val_metrics['force']:.4f} band={val_metrics['band']:.4f} "
                f"contact={val_metrics['contact']:.4f} "
                f"band_bacc={extra['band_balanced_acc']:.3f} contact_acc={extra['contact_acc']:.3f} "
                f"best=epoch {best_epoch}, {best_val:.4f}"
            )
            plot_history(history, str(ckpt_dir / "pretrain_force_loss.png"))
            with open(ckpt_dir / "pretrain_force_history.pkl", "wb") as f:
                pickle.dump(history, f)

    final_extra = eval_extra(model, val_loader, device, max_batches=max_val_batches)
    torch.save({"model_state_dict": model.state_dict(), "config": saved_config, "epoch": epochs - 1, "val_extra": final_extra}, ckpt_dir / "foresight_force_last.ckpt")
    with open(ckpt_dir / "pretrain_force_history.pkl", "wb") as f:
        pickle.dump(history, f)
    plot_history(history, str(ckpt_dir / "pretrain_force_loss.png"))

    summary = {
        "ckpt_dir": str(ckpt_dir),
        "best_epoch": best_epoch,
        "best_val_total": best_val,
        "final_val_extra": final_extra,
        "force_ref": force_ref,
        "purpose": "force-aware multi-step Foresight for board TacQualityEnergy gradient guidance",
    }
    with open(ckpt_dir / "force_aware_foresight_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)
    run(config)


if __name__ == "__main__":
    main()
