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
        tac_history: int = 8,
        foresight_horizon: int = 10,
        image_size: tuple = (224, 224),
        max_token_len: int = 48,
        fixed_prompt: str = "grasp the object with tactile feedback",
        marker_mean: list = None,
        marker_std: list = None,
    ):
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.norm_stats = norm_stats
        self.action_horizon = action_horizon
        self.tac_history = tac_history
        self.foresight_horizon = foresight_horizon
        self.image_size = image_size
        self.max_token_len = max_token_len

        # Marker normalization
        self.marker_mean = np.array(marker_mean or [0.572, -1.786], dtype=np.float32)
        self.marker_std = np.array(marker_std or [1.596, 3.845], dtype=np.float32)

        # Action/qpos normalization stats
        self.action_min = np.array(norm_stats["action_min"], dtype=np.float32)
        self.action_max = np.array(norm_stats["action_max"], dtype=np.float32)
        self.qpos_min = np.array(norm_stats["qpos_min"], dtype=np.float32)
        self.qpos_max = np.array(norm_stats["qpos_max"], dtype=np.float32)

        # Image transforms: resize to 224x224, normalize to [-1, 1]
        self.img_transform = T.Compose([
            T.Resize(image_size),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # → [-1, 1]
        ])

        # Tokenize fixed prompt (simple integer encoding, will be replaced by
        # proper SentencePiece tokenizer when pi0 weights are loaded)
        self._tokenize_prompt(fixed_prompt)

        # Preload episodes into memory for speed
        self.cache = {}
        self._preload()

    def _tokenize_prompt(self, prompt: str):
        """Simple prompt tokenization placeholder.
        In full pipeline, use PaliGemma's SentencePiece tokenizer.
        """
        # Placeholder: encode as ASCII bytes, pad to max_token_len
        tokens = [ord(c) for c in prompt[:self.max_token_len]]
        pad_len = self.max_token_len - len(tokens)
        self.prompt_tokens = torch.tensor(
            tokens + [0] * pad_len, dtype=torch.int32
        )
        self.prompt_mask = torch.tensor(
            [True] * len(tokens) + [False] * pad_len, dtype=torch.bool
        )

    def _preload(self):
        """Preload all episodes into memory."""
        print(f"[Pi0TactileDataset] Preloading {len(self.episode_ids)} episodes...")
        for ep_id in self.episode_ids:
            path = os.path.join(self.dataset_dir, f"episode_{ep_id}.hdf5")
            if not os.path.exists(path):
                continue
            with h5py.File(path, "r") as f:
                self.cache[ep_id] = {
                    "action": f["actions/joint_abs"][:].astype(np.float32),
                    "qpos": f["observations/proprio_joint"][:].astype(np.float32),
                    "global": f["observations/images/global"][:],
                    "wrist": f["observations/images/wrist"][:],
                    "marker_offset": f["observations/tac/left/marker_offset"][:].astype(np.float32),
                }
        print(f"[Pi0TactileDataset] Loaded {len(self.cache)} episodes")

    def __len__(self):
        return len(self.episode_ids) * 10  # oversample for variety

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
        ep_id = self.episode_ids[index % len(self.episode_ids)]
        if ep_id not in self.cache:
            ep_id = list(self.cache.keys())[0]

        ep = self.cache[ep_id]
        ep_len = ep["action"].shape[0]

        # Random start time
        max_start = max(1, ep_len - self.action_horizon)
        start_ts = np.random.randint(0, max_start)

        # === State (qpos) ===
        qpos = self._normalize_qpos(ep["qpos"][start_ts])

        # === Images ===
        global_img = self._process_image(ep["global"][start_ts])
        wrist_img = self._process_image(ep["wrist"][start_ts])

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

        return {
            "images": {
                "base_0_rgb": global_img,
                "left_wrist_0_rgb": wrist_img,
            },
            "image_masks": {
                "base_0_rgb": True,
                "left_wrist_0_rgb": True,
            },
            "state": torch.from_numpy(qpos).float(),
            "tokenized_prompt": self.prompt_tokens.clone(),
            "tokenized_prompt_mask": self.prompt_mask.clone(),
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
        tac_history=config.tac_history,
        foresight_horizon=config.foresight_horizon,
        image_size=config.image_size,
        max_token_len=config.max_token_len,
        fixed_prompt=config.fixed_prompt,
        marker_mean=config.marker_mean,
        marker_std=config.marker_std,
    )

    val_ds = Pi0TactileDataset(
        episode_ids=val_ids,
        dataset_dir=dataset_dir,
        norm_stats=norm_stats,
        action_horizon=config.action_horizon,
        tac_history=config.tac_history,
        foresight_horizon=config.foresight_horizon,
        image_size=config.image_size,
        max_token_len=config.max_token_len,
        fixed_prompt=config.fixed_prompt,
        marker_mean=config.marker_mean,
        marker_std=config.marker_std,
    )

    return train_ds, val_ds, norm_stats
