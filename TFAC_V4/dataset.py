"""
ForesightEpisodicDataset V4: Same as V3 dataset, adapted for TFAC_V4 imports.
Marker mode is always used in V4 (spatial tokenization requires marker_offset).

Returns: (all_cam_images, qpos, action, is_pad, future_cam_images, history_cam_images)
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
    Same as V3's ForesightEpisodicDataset.
    V4 always uses tactile_mode='marker'.
    """

    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats,
                 chunk_size, foresight_horizon=8, image_size=None,
                 proprio_key="qpos", action_key="action",
                 tac_side="left", tac_img_key="img",
                 tactile_mode="marker", history_len=1,
                 multi_frame_vision=False, preload=True,
                 contrastive_vision_indices=None):
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
        self.tactile_mode = tactile_mode
        self.history_len = history_len
        self.multi_frame_vision = multi_frame_vision

        if contrastive_vision_indices is not None:
            self.contrastive_vision_indices = contrastive_vision_indices
        elif multi_frame_vision and foresight_horizon > 1:
            self.contrastive_vision_indices = [foresight_horizon // 2 - 1, foresight_horizon - 1]
        else:
            self.contrastive_vision_indices = None

        self.action_qpos_normalize = NormalizeSeparate(norm_stats)

        self.image_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

        if 'marker_offset_mean' in norm_stats and tactile_mode == 'marker':
            self.mo_mean = torch.tensor(norm_stats['marker_offset_mean'], dtype=torch.float32)
            self.mo_std = torch.tensor(norm_stats['marker_offset_std'], dtype=torch.float32)
        else:
            self.mo_mean = None
            self.mo_std = None

        self.cache = {}
        if preload:
            self._preload_all()
        else:
            self.__getitem__(0)

    def _preload_all(self):
        from tqdm import tqdm
        print(f"Preloading {len(self.episode_ids)} episodes into memory...")
        total_mb = 0
        for ep_id in tqdm(self.episode_ids, desc="Preload"):
            path = os.path.join(self.dataset_dir, f'episode_{ep_id}.hdf5')
            ep_data = {}
            with h5py.File(path, 'r') as root:
                ep_data['action'] = root[f'/{self.action_key}'][()]
                ep_data['qpos'] = root[f'/observations/{self.proprio_key}'][()]

                for cam_name in self.camera_names:
                    if cam_name == 'gelsight':
                        if self.tactile_mode == 'marker':
                            mo_path = f'observations/tac/{self.tac_side}/marker_offset'
                            if mo_path in root:
                                ep_data['tac_marker'] = root[mo_path][()]
                            else:
                                T = ep_data['action'].shape[0]
                                ep_data['tac_marker'] = np.zeros((T, 9, 9, 2), dtype=np.float32)
                        else:
                            tac_path = f'observations/tac/{self.tac_side}/{self.tac_img_key}'
                            if tac_path in root:
                                ep_data['tac_img'] = root[tac_path][()]
                            elif 'observations/gelsight/depth_strain_image' in root:
                                ep_data['tac_img'] = root['observations/gelsight/depth_strain_image'][()]
                    elif cam_name != 'blank':
                        ep_data[f'img_{cam_name}'] = root[f'/observations/images/{cam_name}'][()]

                if self.image_size is None:
                    if 'image_height' in root.attrs:
                        self.image_size = (root.attrs['image_height'], root.attrs['image_width'])
                    else:
                        first_cam = [c for c in self.camera_names if c not in ('gelsight', 'blank')][0]
                        self.image_size = (ep_data[f'img_{first_cam}'].shape[1],
                                           ep_data[f'img_{first_cam}'].shape[2])

            for v in ep_data.values():
                total_mb += v.nbytes / 1024**2
            self.cache[ep_id] = ep_data

        print(f"Preload done: {total_mb:.0f} MB in memory")

    def __len__(self):
        return len(self.episode_ids) * 30

    def _process_cam(self, cam_name, raw_data, ts):
        if cam_name == 'gelsight':
            if self.tactile_mode == 'marker':
                data = torch.tensor(raw_data, dtype=torch.float32)
                if self.mo_mean is not None:
                    data = (data - self.mo_mean) / self.mo_std
                return data
            else:
                data = torch.tensor(raw_data, dtype=torch.float32) / 255.0
                data = data.permute(2, 0, 1)
                data = self.image_normalize(data)
                return data
        elif cam_name == 'blank':
            return torch.zeros(3, self.image_size[0], self.image_size[1])
        else:
            image = torch.tensor(raw_data, dtype=torch.float32) / 255.0
            image = image.permute(2, 0, 1)
            image = self.image_normalize(image)
            return image

    def _get_cam_raw(self, ep_data, cam_name, ts):
        if cam_name == 'gelsight':
            if self.tactile_mode == 'marker':
                return ep_data['tac_marker'][ts]
            else:
                return ep_data['tac_img'][ts]
        elif cam_name == 'blank':
            return None
        else:
            return ep_data[f'img_{cam_name}'][ts]

    def __getitem__(self, index):
        episode_id = self.episode_ids[index % len(self.episode_ids)]

        if episode_id in self.cache:
            ep_data = self.cache[episode_id]
            episode_len = ep_data['action'].shape[0]

            start_ts = np.random.choice(episode_len)
            future_ts = min(start_ts + self.horizon, episode_len - 1)

            qpos = ep_data['qpos'][start_ts]

            all_cam_images = []
            for cam_name in self.camera_names:
                raw = self._get_cam_raw(ep_data, cam_name, start_ts)
                all_cam_images.append(self._process_cam(cam_name, raw, start_ts))

            future_cam_images = []
            for cam_name in self.camera_names:
                if cam_name == 'gelsight' and self.tactile_mode == 'marker':
                    future_frames = []
                    for h in range(1, self.horizon + 1):
                        ft = min(start_ts + h, episode_len - 1)
                        raw = self._get_cam_raw(ep_data, cam_name, ft)
                        future_frames.append(self._process_cam(cam_name, raw, ft))
                    future_cam_images.append(torch.stack(future_frames))
                elif self.multi_frame_vision and self.contrastive_vision_indices is not None:
                    future_frames = []
                    for idx in self.contrastive_vision_indices:
                        h = idx + 1
                        ft = min(start_ts + h, episode_len - 1)
                        raw = self._get_cam_raw(ep_data, cam_name, ft)
                        future_frames.append(self._process_cam(cam_name, raw, ft))
                    future_cam_images.append(torch.stack(future_frames))
                elif self.multi_frame_vision:
                    future_frames = []
                    for h in range(1, self.horizon + 1):
                        ft = min(start_ts + h, episode_len - 1)
                        raw = self._get_cam_raw(ep_data, cam_name, ft)
                        future_frames.append(self._process_cam(cam_name, raw, ft))
                    future_cam_images.append(torch.stack(future_frames))
                else:
                    raw = self._get_cam_raw(ep_data, cam_name, future_ts)
                    future_cam_images.append(self._process_cam(cam_name, raw, future_ts))

            history_cam_images = []
            for cam_name in self.camera_names:
                frames = []
                for i in range(self.history_len):
                    hist_ts = max(0, start_ts - (self.history_len - 1 - i))
                    raw = self._get_cam_raw(ep_data, cam_name, hist_ts)
                    frames.append(self._process_cam(cam_name, raw, hist_ts))
                history_cam_images.append(torch.stack(frames))

            action_len = min(episode_len - start_ts, self.chunk_size)
            action = ep_data['action'][start_ts:start_ts + action_len]

        else:
            dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
            with h5py.File(dataset_path, 'r') as root:
                action_dataset = root[f'/{self.action_key}']
                episode_len = action_dataset.shape[0]

                start_ts = np.random.choice(episode_len)
                future_ts = min(start_ts + self.horizon, episode_len - 1)

                qpos = root[f'/observations/{self.proprio_key}'][start_ts]

                if self.image_size is None:
                    if 'image_height' in root.attrs:
                        self.image_size = (root.attrs['image_height'], root.attrs['image_width'])
                    else:
                        first_cam = [c for c in self.camera_names if c not in ('gelsight', 'blank')][0]
                        img_shape = root[f'/observations/images/{first_cam}'].shape
                        self.image_size = (img_shape[1], img_shape[2])

                all_cam_images = []
                for cam_name in self.camera_names:
                    all_cam_images.append(self._load_cam_images(root, cam_name, start_ts))

                future_cam_images = []
                for cam_name in self.camera_names:
                    if cam_name == 'gelsight' and self.tactile_mode == 'marker':
                        future_frames = []
                        for h in range(1, self.horizon + 1):
                            ft = min(start_ts + h, episode_len - 1)
                            future_frames.append(self._load_cam_images(root, cam_name, ft))
                        future_cam_images.append(torch.stack(future_frames))
                    elif self.multi_frame_vision and self.contrastive_vision_indices is not None:
                        future_frames = []
                        for idx in self.contrastive_vision_indices:
                            h = idx + 1
                            ft = min(start_ts + h, episode_len - 1)
                            future_frames.append(self._load_cam_images(root, cam_name, ft))
                        future_cam_images.append(torch.stack(future_frames))
                    elif self.multi_frame_vision:
                        future_frames = []
                        for h in range(1, self.horizon + 1):
                            ft = min(start_ts + h, episode_len - 1)
                            future_frames.append(self._load_cam_images(root, cam_name, ft))
                        future_cam_images.append(torch.stack(future_frames))
                    else:
                        future_cam_images.append(self._load_cam_images(root, cam_name, future_ts))

                history_cam_images = []
                for cam_name in self.camera_names:
                    frames = []
                    for i in range(self.history_len):
                        hist_ts = max(0, start_ts - (self.history_len - 1 - i))
                        frames.append(self._load_cam_images(root, cam_name, hist_ts))
                    history_cam_images.append(torch.stack(frames))

                action_len = min(episode_len - start_ts, self.chunk_size)
                action = action_dataset[start_ts:start_ts + action_len]

        qpos, action = self.action_qpos_normalize(qpos=qpos, action=action)

        padded_action = np.zeros([self.chunk_size, action.shape[1]], dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(self.chunk_size)
        is_pad[action_len:] = 1

        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        return all_cam_images, qpos_data, action_data, is_pad, future_cam_images, history_cam_images

    def _load_cam_images(self, root, cam_name, ts):
        """HDF5 fallback path (preload=False)."""
        if cam_name == 'gelsight':
            if self.tactile_mode == 'marker':
                mo_path = f'observations/tac/{self.tac_side}/marker_offset'
                if mo_path in root:
                    data = root[mo_path][ts]
                    data = torch.tensor(data, dtype=torch.float32)
                else:
                    data = torch.zeros(9, 9, 2, dtype=torch.float32)
                if self.mo_mean is not None:
                    data = (data - self.mo_mean) / self.mo_std
                return data

            tac_path = f'observations/tac/{self.tac_side}/{self.tac_img_key}'
            if tac_path in root:
                data = root[tac_path][ts]
                data = torch.tensor(data, dtype=torch.float32) / 255.0
                data = data.permute(2, 0, 1)
                data = self.image_normalize(data)
                return data
            elif 'observations/gelsight/depth_strain_image' in root:
                data = root['observations/gelsight/depth_strain_image'][ts]
                data = torch.tensor(data, dtype=torch.float32)
                data = data.permute(2, 0, 1)
                return data
            else:
                raise KeyError(f"No tactile data at '{tac_path}'")

        elif cam_name == 'blank':
            return torch.zeros(3, self.image_size[0], self.image_size[1])

        else:
            image = root[f'/observations/images/{cam_name}'][ts]
            if self.image_size is None:
                self.image_size = (image.shape[0], image.shape[1])
            image = torch.tensor(image, dtype=torch.float32) / 255.0
            image = image.permute(2, 0, 1)
            image = self.image_normalize(image)
            return image
