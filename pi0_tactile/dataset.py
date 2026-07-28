"""
Dataset adapter: loads our HDF5 episodes and returns Pi0-compatible observation format.

Each sample contains:
  - images: dict of camera views as (C, H, W) tensors in [-1, 1]
  - state: qpos (action_dim,) normalized to [-1, 1]
  - marker_offset: (T_hist, 9, 9, 2) tactile sequence
  - future_marker_offset: (T_hist, 9, 9, 2) for foresight GT
  - actions: (action_horizon, action_dim) normalized to [-1, 1]
  - lang_tokens / lang_masks: tokenized prompt
"""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass

import h5py
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

from pi0_tactile.prompt import PromptTokenizer


CAMERA_ALIASES = {
    "base_0_rgb": "global",
    "left_wrist_0_rgb": "wrist",
    "right_wrist_0_rgb": "right_wrist",
}


def _fit_stat_dim(arr: np.ndarray, dim: int, pad_value: float) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    if arr.shape[0] >= dim:
        return arr[:dim]
    pad = np.full(dim - arr.shape[0], float(pad_value), dtype=np.float32)
    return np.concatenate([arr, pad], axis=0)


@dataclass
class Pi0Observation:
    """Pi0-compatible observation structure."""
    images: dict  # {cam_name: (B, C, H, W)}
    image_masks: dict  # {cam_name: (B,) bool}
    state: torch.Tensor  # (B, state_dim)
    tokenized_prompt: torch.Tensor  # (B, max_token_len)
    tokenized_prompt_mask: torch.Tensor  # (B, max_token_len)
    marker_offset: torch.Tensor  # (B, T_hist, 9, 9, 2)
    future_marker_offset: torch.Tensor  # (B, T_hist, 9, 9, 2)

    def to(self, device):
        return Pi0Observation(
            images={k: v.to(device) for k, v in self.images.items()},
            image_masks={k: v.to(device) for k, v in self.image_masks.items()},
            state=self.state.to(device),
            tokenized_prompt=self.tokenized_prompt.to(device),
            tokenized_prompt_mask=self.tokenized_prompt_mask.to(device),
            marker_offset=self.marker_offset.to(device),
            future_marker_offset=self.future_marker_offset.to(device),
        )


class Pi0TactileDataset(Dataset):
    """
    Episodic dataset that loads HDF5 data in Pi0 observation format.
    """

    def __init__(
        self,
        episode_ids: list,
        dataset_dir: str,
        norm_stats: dict,
        action_horizon: int = 20,
        model_action_dim: int = 7,
        robot_action_dim: int = 7,
        tac_history: int = 8,
        foresight_horizon: int = 10,
        image_size: tuple = (224, 224),
        camera_names: list[str] | None = None,
        max_token_len: int = 48,
        fixed_prompt: str = "grasp the object with tactile feedback",
        tokenizer_backend: str = "ascii",
        pi05: bool = False,
        marker_mean: list = None,
        marker_std: list = None,
        action_key: str = "actions/joint_abs",
        proprio_key: str = "observations/proprio_joint",
        tactile_key: str = "observations/tac/left/marker_offset",
        preload: bool = True,
        samples_per_episode: int = 10,
    ):
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.norm_stats = norm_stats
        self.action_horizon = action_horizon
        self.model_action_dim = model_action_dim
        self.robot_action_dim = robot_action_dim
        self.tac_history = tac_history
        self.foresight_horizon = foresight_horizon
        self.image_size = image_size
        self.max_token_len = max_token_len
        self.camera_names = list(camera_names or ["base_0_rgb", "left_wrist_0_rgb"])
        self.hdf5_camera_names = [CAMERA_ALIASES.get(c, c) for c in self.camera_names]
        self.fixed_prompt = fixed_prompt
        self.pi05 = bool(pi05)
        self.action_key = action_key
        self.proprio_key = proprio_key
        self.tactile_key = tactile_key
        self.preload = bool(preload)
        self.samples_per_episode = int(samples_per_episode)

        # Marker normalization
        self.marker_mean = np.array(marker_mean or [0.572, -1.786], dtype=np.float32)
        self.marker_std = np.array(marker_std or [1.596, 3.845], dtype=np.float32)

        # Action/qpos normalization stats
        self.action_min = _fit_stat_dim(norm_stats["action_min"], robot_action_dim, 0.0)
        self.action_max = _fit_stat_dim(norm_stats["action_max"], robot_action_dim, 1.0)
        self.qpos_min = _fit_stat_dim(norm_stats["qpos_min"], robot_action_dim, 0.0)
        self.qpos_max = _fit_stat_dim(norm_stats["qpos_max"], robot_action_dim, 1.0)

        # Image transforms: resize to 224x224, normalize to [-1, 1]
        self.img_transform = T.Compose([
            T.Resize(image_size),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # → [-1, 1]
        ])

        self.prompt_tokenizer = PromptTokenizer(
            max_token_len=max_token_len,
            pi05=self.pi05,
            backend=tokenizer_backend,
        )
        self.prompt_tokens = None
        self.prompt_mask = None
        if not self.pi05:
            self.prompt_tokens, self.prompt_mask = self.prompt_tokenizer.tokenize_torch(
                fixed_prompt
            )

        # Preload episodes into memory for speed
        self.cache = {}
        self.valid_episode_ids = []
        if self.preload:
            self._preload()
        else:
            self.valid_episode_ids = [
                ep_id for ep_id in self.episode_ids
                if os.path.exists(os.path.join(self.dataset_dir, f"episode_{ep_id}.hdf5"))
            ]

    def _preload(self):
        """Preload all episodes into memory."""
        print(f"[Pi0TactileDataset] Preloading {len(self.episode_ids)} episodes...")
        for ep_id in self.episode_ids:
            path = os.path.join(self.dataset_dir, f"episode_{ep_id}.hdf5")
            if not os.path.exists(path):
                continue
            with h5py.File(path, "r") as f:
                images = {}
                for out_name, h5_name in zip(self.camera_names, self.hdf5_camera_names):
                    h5_path = f"observations/images/{h5_name}"
                    if h5_path not in f:
                        raise KeyError(f"Missing camera dataset {h5_path} in {path}")
                    images[out_name] = f[h5_path][:]
                self.cache[ep_id] = {
                    "action": f[self.action_key][:, : self.robot_action_dim].astype(np.float32),
                    "qpos": f[self.proprio_key][:, : self.robot_action_dim].astype(np.float32),
                    "images": images,
                    "marker_offset": f[self.tactile_key][:].astype(np.float32),
                }
                self.valid_episode_ids.append(ep_id)
        print(f"[Pi0TactileDataset] Loaded {len(self.cache)} episodes")

    def __len__(self):
        return len(self.valid_episode_ids) * max(1, self.samples_per_episode)

    def _load_episode(self, ep_id: int) -> dict:
        if self.preload:
            return self.cache[ep_id]
        path = os.path.join(self.dataset_dir, f"episode_{ep_id}.hdf5")
        with h5py.File(path, "r") as f:
            images = {}
            for out_name, h5_name in zip(self.camera_names, self.hdf5_camera_names):
                images[out_name] = f[f"observations/images/{h5_name}"][:]
            return {
                "action": f[self.action_key][:, : self.robot_action_dim].astype(np.float32),
                "qpos": f[self.proprio_key][:, : self.robot_action_dim].astype(np.float32),
                "images": images,
                "marker_offset": f[self.tactile_key][:].astype(np.float32),
            }

    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        """Normalize action to [-1, 1] using min-max."""
        return (action - self.action_min) / (self.action_max - self.action_min + 1e-8) * 2 - 1

    def _normalize_qpos(self, qpos: np.ndarray) -> np.ndarray:
        """Normalize qpos to [-1, 1] using min-max."""
        return (qpos - self.qpos_min) / (self.qpos_max - self.qpos_min + 1e-8) * 2 - 1

    def _normalize_marker(self, marker: np.ndarray) -> np.ndarray:
        """Normalize marker_offset per channel."""
        return (marker - self.marker_mean) / (self.marker_std + 1e-8)

    def _process_image(self, img: np.ndarray) -> torch.Tensor:
        """Convert uint8 HWC image to CHW float tensor in [-1, 1]."""
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        t = torch.from_numpy(img).permute(2, 0, 1).float()  # (C, H, W)
        return self.img_transform(t)

    def _get_marker_window(self, marker_data: np.ndarray, ts: int) -> np.ndarray:
        """Get T_hist frames of marker_offset ending at ts."""
        T = self.tac_history
        ep_len = marker_data.shape[0]
        frames = []
        for i in range(T):
            idx = max(0, ts - (T - 1 - i))
            idx = min(idx, ep_len - 1)
            frames.append(marker_data[idx])
        return np.stack(frames, axis=0)  # (T_hist, 9, 9, 2)

    def __getitem__(self, index):
        if not self.valid_episode_ids:
            raise RuntimeError("No valid HDF5 episodes loaded")
        ep_id = self.valid_episode_ids[index % len(self.valid_episode_ids)]

        ep = self._load_episode(ep_id)
        ep_len = ep["action"].shape[0]

        # Random start time
        max_start = max(1, ep_len - self.action_horizon)
        start_ts = np.random.randint(0, max_start)

        # === State (qpos) ===
        qpos = self._normalize_qpos(ep["qpos"][start_ts])

        # === Images ===
        images = {
            cam_name: self._process_image(ep["images"][cam_name][start_ts])
            for cam_name in self.camera_names
        }

        # === Tactile (current) ===
        marker_current = self._get_marker_window(ep["marker_offset"], start_ts)
        marker_current = self._normalize_marker(marker_current)

        # === Tactile (future, for foresight GT) ===
        future_ts = min(start_ts + self.foresight_horizon, ep_len - 1)
        marker_future = self._get_marker_window(ep["marker_offset"], future_ts)
        marker_future = self._normalize_marker(marker_future)

        # === Actions ===
        action_end = min(start_ts + self.action_horizon, ep_len)
        action_chunk = ep["action"][start_ts:action_end]
        # Pad if shorter than action_horizon
        if len(action_chunk) < self.action_horizon:
            pad = np.tile(action_chunk[-1:], (self.action_horizon - len(action_chunk), 1))
            action_chunk = np.concatenate([action_chunk, pad], axis=0)
        action_chunk = self._normalize_action(action_chunk)

        if self.pi05:
            prompt_state = qpos
            if self.model_action_dim > prompt_state.shape[-1]:
                prompt_state = np.concatenate([
                    prompt_state,
                    np.zeros(self.model_action_dim - prompt_state.shape[-1], dtype=np.float32),
                ])
            prompt_tokens, prompt_mask = self.prompt_tokenizer.tokenize_torch(
                self.fixed_prompt,
                state=prompt_state,
            )
        else:
            prompt_tokens = self.prompt_tokens.clone()
            prompt_mask = self.prompt_mask.clone()

        return {
            "images": images,
            "image_masks": {cam_name: True for cam_name in self.camera_names},
            "state": torch.from_numpy(qpos).float(),
            "tokenized_prompt": prompt_tokens,
            "tokenized_prompt_mask": prompt_mask,
            "marker_offset": torch.from_numpy(marker_current).float(),
            "future_marker_offset": torch.from_numpy(marker_future).float(),
            "actions": torch.from_numpy(action_chunk).float(),
        }


def collate_pi0_batch(batch: list[dict]) -> tuple:
    """Custom collate function for Pi0TactileDataset."""
    B = len(batch)

    # Stack images
    images = {
        cam: torch.stack([b["images"][cam] for b in batch])
        for cam in batch[0]["images"].keys()
    }
    image_masks = {
        cam: torch.tensor([b["image_masks"][cam] for b in batch], dtype=torch.bool)
        for cam in batch[0]["image_masks"].keys()
    }

    state = torch.stack([b["state"] for b in batch])
    prompt = torch.stack([b["tokenized_prompt"] for b in batch])
    prompt_mask = torch.stack([b["tokenized_prompt_mask"] for b in batch])
    marker_offset = torch.stack([b["marker_offset"] for b in batch])
    future_marker = torch.stack([b["future_marker_offset"] for b in batch])
    actions = torch.stack([b["actions"] for b in batch])

    observation = Pi0Observation(
        images=images,
        image_masks=image_masks,
        state=state,
        tokenized_prompt=prompt,
        tokenized_prompt_mask=prompt_mask,
        marker_offset=marker_offset,
        future_marker_offset=future_marker,
    )

    return observation, actions


def build_datasets(
    dataset_dir: str,
    config,
    seed: int = 1,
    val_ratio: float = 0.2,
):
    """Build train/val datasets from episode directory."""
    # Find all episodes
    episode_files = sorted([
        f for f in os.listdir(dataset_dir)
        if f.startswith("episode_") and f.endswith(".hdf5")
    ])
    episode_ids = [int(f.split("_")[1].split(".")[0]) for f in episode_files]

    # Train/val split
    rng = np.random.RandomState(seed)
    rng.shuffle(episode_ids)
    n_val = int(len(episode_ids) * val_ratio)
    val_ids = episode_ids[:n_val]
    train_ids = episode_ids[n_val:]

    # Load norm stats
    stats_path = os.path.join(dataset_dir, "dataset_stats.pkl")
    with open(stats_path, "rb") as f:
        norm_stats = pickle.load(f)

    train_ds = Pi0TactileDataset(
        episode_ids=train_ids,
        dataset_dir=dataset_dir,
        norm_stats=norm_stats,
        action_horizon=config.action_horizon,
        model_action_dim=config.action_dim,
        robot_action_dim=config.robot_action_dim,
        tac_history=config.tac_history,
        foresight_horizon=config.foresight_horizon,
        image_size=config.image_size,
        camera_names=config.camera_names,
        max_token_len=config.max_token_len,
        fixed_prompt=config.fixed_prompt,
        tokenizer_backend=config.tokenizer_backend,
        pi05=config.pi05,
        marker_mean=config.marker_mean,
        marker_std=config.marker_std,
        action_key=config.action_key,
        proprio_key=config.proprio_key,
        tactile_key=config.tactile_key,
        preload=config.preload_dataset,
        samples_per_episode=config.samples_per_episode,
    )

    val_ds = Pi0TactileDataset(
        episode_ids=val_ids,
        dataset_dir=dataset_dir,
        norm_stats=norm_stats,
        action_horizon=config.action_horizon,
        model_action_dim=config.action_dim,
        robot_action_dim=config.robot_action_dim,
        tac_history=config.tac_history,
        foresight_horizon=config.foresight_horizon,
        image_size=config.image_size,
        camera_names=config.camera_names,
        max_token_len=config.max_token_len,
        fixed_prompt=config.fixed_prompt,
        tokenizer_backend=config.tokenizer_backend,
        pi05=config.pi05,
        marker_mean=config.marker_mean,
        marker_std=config.marker_std,
        action_key=config.action_key,
        proprio_key=config.proprio_key,
        tactile_key=config.tactile_key,
        preload=config.preload_dataset,
        samples_per_episode=config.samples_per_episode,
    )

    return train_ds, val_ds, norm_stats
