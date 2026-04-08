"""
ForesightEpisodicDataset V2: 在 V1 基础上额外返回过去 k 帧历史数据,
用于时序 ForesightTransformer 的输入。
"""

import os
import numpy as np
import torch
import h5py
from torchvision import transforms

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import NormalizeSeparate


class ForesightEpisodicDataset(torch.utils.data.Dataset):
    """
    返回: (all_cam_images, qpos, action, is_pad, future_cam_images, history_cam_images)
    - future_cam_images: t+h 时刻的所有相机图像
    - history_cam_images: 过去 k 帧 [t-(k-1), ..., t-1, t] 的相机数据
      结构: per camera list of k 帧 stacked tensor
      history_cam_images[cam_idx] = (k, C, H, W) 或 marker mode (k, 9, 9, 2)
    """

    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats,
                 chunk_size, foresight_horizon=8, image_size=None,
                 proprio_key="qpos", action_key="action",
                 tac_side="left", tac_img_key="img",
                 tactile_mode="image", history_len=1):
        super().__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.chunk_size = chunk_size
        self.horizon = foresight_horizon
        self.proprio_key = proprio_key
        self.action_key = action_key
        self.tac_side = tac_side
        self.tac_img_key = tac_img_key
        self.image_size = image_size
        self.tactile_mode = tactile_mode  # "image" or "marker"
        self.history_len = history_len    # k: number of past frames (including current)

        self.action_qpos_normalize = NormalizeSeparate(norm_stats)

        self.image_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

        # marker_offset normalization (None if image mode or stats not available)
        if 'marker_offset_mean' in norm_stats and tactile_mode == 'marker':
            self.mo_mean = torch.tensor(norm_stats['marker_offset_mean'], dtype=torch.float32)  # (2,)
            self.mo_std = torch.tensor(norm_stats['marker_offset_std'], dtype=torch.float32)    # (2,)
        else:
            self.mo_mean = None
            self.mo_std = None

        # initialize image_size
        self.__getitem__(0)

    def __len__(self):
        return len(self.episode_ids)

    def _load_cam_images(self, root, cam_name, ts):
        """加载单个相机在时刻 ts 的图像/触觉数据。
        Returns:
            tactile_mode="image":  (C, H, W) image tensor for all cameras
            tactile_mode="marker": (9, 9, 2) marker_offset tensor for gelsight,
                                   (C, H, W) image tensor for others
        """
        if cam_name == 'gelsight':
            if self.tactile_mode == 'marker':
                # Load marker_offset (9, 9, 2) instead of image
                mo_path = f'observations/tac/{self.tac_side}/marker_offset'
                if mo_path in root:
                    data = root[mo_path][ts]  # (9, 9, 2)
                    data = torch.tensor(data, dtype=torch.float32)
                else:
                    data = torch.zeros(9, 9, 2, dtype=torch.float32)
                # normalize per x/y channel
                if self.mo_mean is not None:
                    data = (data - self.mo_mean) / self.mo_std  # broadcast (2,) to (9,9,2)
                return data

            # tactile_mode="image": original logic
            tac_path = f'observations/tac/{self.tac_side}/{self.tac_img_key}'
            if tac_path in root:
                data = root[tac_path][ts]
                data = torch.tensor(data, dtype=torch.float32) / 255.0
                data = torch.einsum('h w c -> c h w', data)
                data = self.image_normalize(data)
                return data
            elif 'observations/gelsight/depth_strain_image' in root:
                data = root['observations/gelsight/depth_strain_image'][ts]
                data = torch.tensor(data, dtype=torch.float32)
                data = torch.einsum('h w c -> c h w', data)
                return data
            else:
                raise KeyError(f"No tactile data found at '{tac_path}' or legacy path")

        elif cam_name == 'blank':
            return torch.zeros(3, self.image_size[0], self.image_size[1])

        else:
            image = root[f'/observations/images/{cam_name}'][ts]
            if self.image_size is None:
                self.image_size = (image.shape[0], image.shape[1])
            image = torch.tensor(image, dtype=torch.float32) / 255.0
            image = torch.einsum('h w c -> c h w', image)
            image = self.image_normalize(image)
            return image

    def __getitem__(self, index):
        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')

        with h5py.File(dataset_path, 'r') as root:
            action_dataset = root[f'/{self.action_key}']
            episode_len = action_dataset.shape[0]

            start_ts = np.random.choice(episode_len)
            future_ts = min(start_ts + self.horizon, episode_len - 1)

            # qpos
            qpos = root[f'/observations/{self.proprio_key}'][start_ts]

            # infer image_size if needed
            if self.image_size is None:
                if 'image_height' in root.attrs:
                    self.image_size = (root.attrs['image_height'], root.attrs['image_width'])
                else:
                    first_cam = [c for c in self.camera_names if c not in ('gelsight', 'blank')][0]
                    img_shape = root[f'/observations/images/{first_cam}'].shape
                    self.image_size = (img_shape[1], img_shape[2])

            # 当前时刻图像
            all_cam_images = []
            for cam_name in self.camera_names:
                all_cam_images.append(self._load_cam_images(root, cam_name, start_ts))

            # t+h 时刻图像 (多帧: t+1 到 t+H 的 gelsight; 单帧 t+H 的 vision)
            future_cam_images = []
            for cam_name in self.camera_names:
                if cam_name == 'gelsight' and self.tactile_mode == 'marker':
                    # 多帧 GT: (H, 9, 9, 2) for t+1 to t+H
                    future_frames = []
                    for h in range(1, self.horizon + 1):
                        ft = min(start_ts + h, episode_len - 1)
                        future_frames.append(self._load_cam_images(root, cam_name, ft))
                    future_cam_images.append(torch.stack(future_frames))  # (H, 9, 9, 2)
                else:
                    # Vision: 只返回最后一帧 t+H
                    future_cam_images.append(self._load_cam_images(root, cam_name, future_ts))

            # 历史 k 帧: [t-(k-1), ..., t-1, t], clamp to 0
            # Per camera: stack k frames into (k, ...) tensor
            history_cam_images = []
            for cam_name in self.camera_names:
                frames = []
                for i in range(self.history_len):
                    hist_ts = max(0, start_ts - (self.history_len - 1 - i))
                    frames.append(self._load_cam_images(root, cam_name, hist_ts))
                history_cam_images.append(torch.stack(frames))  # (k, C, H, W) or (k, 9, 9, 2)

            # action chunk
            action_len = min(episode_len - start_ts, self.chunk_size)
            action = action_dataset[start_ts:start_ts + action_len]

        # normalize
        qpos, action = self.action_qpos_normalize(qpos=qpos, action=action)

        # pad action
        padded_action = np.zeros([self.chunk_size, action.shape[1]], dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(self.chunk_size)
        is_pad[action_len:] = 1

        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        return all_cam_images, qpos_data, action_data, is_pad, future_cam_images, history_cam_images
