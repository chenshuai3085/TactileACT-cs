"""
ForesightEpisodicDataset: 在 EpisodicDataset 基础上额外返回 t+h 时刻的图像,
用于前瞻预测的监督信号。
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
    返回: (all_cam_images, qpos, action, is_pad, future_cam_images)
    其中 future_cam_images 是 t+h 时刻的所有相机图像 (与 all_cam_images 结构相同)。
    """

    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats,
                 chunk_size, foresight_horizon=8, image_size=None,
                 proprio_key="qpos", action_key="action",
                 tac_side="left", tac_img_key="img"):
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

        self.action_qpos_normalize = NormalizeSeparate(norm_stats)

        self.image_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

        # initialize image_size
        self.__getitem__(0)

    def __len__(self):
        return len(self.episode_ids)

    def _load_cam_images(self, root, cam_name, ts):
        """加载单个相机在时刻 ts 的图像, 返回 tensor (C,H,W)"""
        if cam_name == 'gelsight':
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

            # t+h 时刻图像
            future_cam_images = []
            for cam_name in self.camera_names:
                future_cam_images.append(self._load_cam_images(root, cam_name, future_ts))

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

        return all_cam_images, qpos_data, action_data, is_pad, future_cam_images
