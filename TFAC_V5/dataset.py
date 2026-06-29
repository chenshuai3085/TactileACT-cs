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
                 tactile_mode="image", history_len=1,
                 multi_frame_vision=False, preload=True,
                 contrastive_vision_indices=None,
                 tactile_vae_window=0,
                 use_state_trajectory=False,
                 future_offset=1,
                 action_offset=None,
                 temporal_stride=1):
        super().__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.chunk_size = chunk_size
        self.horizon = foresight_horizon
        self.proprio_key = proprio_key
        self.action_key = action_key
        self.use_state_trajectory = use_state_trajectory
        self.future_offset = int(future_offset)
        if self.future_offset < 0:
            raise ValueError("future_offset must be >= 0")
        if action_offset is None:
            # Backward compatible defaults:
            # - state trajectory conditions on qpos[t+1:...] for old foresight.
            # - raw action conditions on action[t:...] for old foresight.
            self.action_offset = self.future_offset if use_state_trajectory else 0
        else:
            self.action_offset = int(action_offset)
        if self.action_offset < 0:
            raise ValueError("action_offset must be >= 0")
        self.temporal_stride = int(temporal_stride)
        if self.temporal_stride < 1:
            raise ValueError(f"temporal_stride must be >= 1, got {temporal_stride}")
        self.tac_side = tac_side
        self.tac_img_key = tac_img_key
        self.image_size = image_size
        self.tactile_mode = tactile_mode  # "image" or "marker"
        self.history_len = history_len    # k: number of past frames (including current)
        self.multi_frame_vision = multi_frame_vision  # 视觉相机也返回多帧 future (per-frame contrastive)
        self.tactile_vae_window = tactile_vae_window  # T: TactileVAE temporal window (0=disabled)

        # 视觉 future 只加载指定帧用于对比学习 (默认: H//2 和 H, 即中间+结尾)
        # 触觉 future 仍加载全部 H 帧 (foresight_tac loss 需要)
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

        # marker_offset normalization (None if image mode or stats not available)
        if 'marker_offset_mean' in norm_stats and tactile_mode == 'marker':
            self.mo_mean = torch.tensor(norm_stats['marker_offset_mean'], dtype=torch.float32)  # (2,)
            self.mo_std = torch.tensor(norm_stats['marker_offset_std'], dtype=torch.float32)    # (2,)
        else:
            self.mo_mean = None
            self.mo_std = None

        # --- 预加载所有 episode 数据到内存 ---
        self.cache = {}  # episode_id → dict of numpy arrays
        self.has_image_camera = any(c not in ('gelsight', 'blank') for c in self.camera_names)

        if preload:
            self._preload_all()
        elif self.has_image_camera:
            # initialize image_size
            self.__getitem__(0)

    def _preload_all(self):
        """预加载所有 episode 数据到内存，避免训练时反复打开 hdf5。"""
        from tqdm import tqdm
        print(f"Preloading {len(self.episode_ids)} episodes into memory...")
        total_mb = 0
        skipped = []
        for ep_id in tqdm(self.episode_ids, desc="Preload"):
            if isinstance(ep_id, str) and os.path.isabs(ep_id):
                path = ep_id
            else:
                path = os.path.join(self.dataset_dir, f'episode_{ep_id}.hdf5')
            ep_data = {}
            try:
                with h5py.File(path, 'r') as root:
                    # action & qpos
                    ep_data['action'] = root[f'/{self.action_key}'][()]
                    ep_data['qpos'] = root[f'/observations/{self.proprio_key}'][()]

                    # 视觉相机图像
                    for cam_name in self.camera_names:
                        if cam_name == 'gelsight':
                            if self.tactile_mode == 'marker':
                                mo_path = f'observations/tac/{self.tac_side}/marker_offset'
                                if mo_path in root:
                                    ep_data[f'tac_marker'] = root[mo_path][()]  # (T, 9, 9, 2)
                                else:
                                    T = ep_data['action'].shape[0]
                                    ep_data[f'tac_marker'] = np.zeros((T, 9, 9, 2), dtype=np.float32)
                            else:
                                tac_path = f'observations/tac/{self.tac_side}/{self.tac_img_key}'
                                if tac_path in root:
                                    ep_data[f'tac_img'] = root[tac_path][()]  # (T, H, W, 3)
                                elif 'observations/gelsight/depth_strain_image' in root:
                                    ep_data[f'tac_img'] = root['observations/gelsight/depth_strain_image'][()]
                        elif cam_name != 'blank':
                            ep_data[f'img_{cam_name}'] = root[f'/observations/images/{cam_name}'][()]  # (T, H, W, 3)

                    # infer image_size
                    if self.image_size is None and self.has_image_camera:
                        if 'image_height' in root.attrs:
                            self.image_size = (root.attrs['image_height'], root.attrs['image_width'])
                        else:
                            first_cam = [c for c in self.camera_names if c not in ('gelsight', 'blank')][0]
                            self.image_size = (ep_data[f'img_{first_cam}'].shape[1],
                                               ep_data[f'img_{first_cam}'].shape[2])

            except (OSError, KeyError) as e:
                skipped.append(path)
                continue

            for v in ep_data.values():
                total_mb += v.nbytes / 1024**2
            self.cache[ep_id] = ep_data

        if skipped:
            print(f"WARNING: Skipped {len(skipped)} corrupt episodes:")
            for s in skipped:
                print(f"  {s}")
            self.episode_ids = [eid for eid in self.episode_ids if eid in self.cache]
        print(f"Preload done: {len(self.cache)} episodes, {total_mb:.0f} MB in memory")

    def __len__(self):
        # 每 epoch 每个 episode 采 30 帧, 1000 epoch ≈ 30k steps (bs=256)
        # 对齐 ACT 原版训练量级, 避免 *300 导致单 epoch 过长
        return len(self.episode_ids) * 30

    def _process_cam(self, cam_name, raw_data, ts):
        """将 numpy 数据转为归一化后的 tensor。
        Args:
            cam_name: 相机名
            raw_data: numpy array (单帧, 已索引 ts)
            ts: 时间步 (仅 blank 相机需要)
        Returns:
            tactile_mode="marker" + gelsight: (9, 9, 2) tensor
            其他: (C, H, W) tensor
        """
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
        """从 cache 中获取相机原始 numpy 数据 (单帧)。"""
        if cam_name == 'gelsight':
            if self.tactile_mode == 'marker':
                return ep_data['tac_marker'][ts]
            else:
                return ep_data['tac_img'][ts]
        elif cam_name == 'blank':
            return None  # blank 由 _process_cam 处理
        else:
            return ep_data[f'img_{cam_name}'][ts]

    def _load_cam_images(self, root, cam_name, ts):
        """加载单个相机在时刻 ts 的图像/触觉数据 (hdf5 回退路径)。
        仅在 preload=False 时使用。
        """
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
                raise KeyError(f"No tactile data found at '{tac_path}' or legacy path")

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

    def _required_span(self, offset, length):
        return int(offset) + (int(length) - 1) * self.temporal_stride + 1

    def _chunk_indices(self, start_ts, offset, length, episode_len):
        first = int(start_ts) + int(offset)
        if first >= episode_len:
            return np.array([episode_len - 1], dtype=np.int64)
        max_count = ((episode_len - 1 - first) // self.temporal_stride) + 1
        count = max(1, min(int(length), int(max_count)))
        return first + np.arange(count, dtype=np.int64) * self.temporal_stride

    def __getitem__(self, index):
        try:
            return self._getitem_impl(index)
        except (OSError, KeyError):
            return self._getitem_impl(np.random.randint(len(self)))

    def _getitem_impl(self, index):
        episode_id = self.episode_ids[index % len(self.episode_ids)]

        # --- 从 cache 读取 (preload 模式) ---
        if episode_id in self.cache:
            ep_data = self.cache[episode_id]
            episode_len = ep_data['action'].shape[0]

            min_required = max(
                self._required_span(self.future_offset, self.horizon),
                self._required_span(self.action_offset, self.chunk_size),
            )
            max_start = max(0, episode_len - min_required)
            start_ts = np.random.randint(max_start + 1)
            future_ts = min(
                start_ts + self.future_offset + (self.horizon - 1) * self.temporal_stride,
                episode_len - 1,
            )

            qpos = ep_data['qpos'][start_ts]

            # 当前时刻图像
            all_cam_images = []
            for cam_name in self.camera_names:
                if cam_name == 'gelsight' and self.tactile_mode == 'marker' and self.tactile_vae_window > 0:
                    T = self.tactile_vae_window
                    frames = []
                    for i in range(T):
                        hist_ts = max(0, start_ts - (T - 1 - i))
                        raw = self._get_cam_raw(ep_data, cam_name, hist_ts)
                        frames.append(self._process_cam(cam_name, raw, hist_ts))
                    all_cam_images.append(torch.stack(frames))  # (T, 9, 9, 2)
                else:
                    raw = self._get_cam_raw(ep_data, cam_name, start_ts)
                    all_cam_images.append(self._process_cam(cam_name, raw, start_ts))

            # future 图像
            future_cam_images = []
            for cam_name in self.camera_names:
                if cam_name == 'gelsight' and self.tactile_mode == 'marker':
                    # 触觉: 全部 H 帧 (foresight_tac loss 需要)
                    future_frames = []
                    for i in range(self.horizon):
                        ft = min(
                            start_ts + self.future_offset + i * self.temporal_stride,
                            episode_len - 1,
                        )
                        if self.tactile_vae_window > 0:
                            T = self.tactile_vae_window
                            window_frames = []
                            for i in range(T):
                                wt = max(0, ft - (T - 1 - i))
                                raw = self._get_cam_raw(ep_data, cam_name, wt)
                                window_frames.append(self._process_cam(cam_name, raw, wt))
                            future_frames.append(torch.stack(window_frames))  # (T, 9, 9, 2)
                        else:
                            raw = self._get_cam_raw(ep_data, cam_name, ft)
                            future_frames.append(self._process_cam(cam_name, raw, ft))
                    future_cam_images.append(torch.stack(future_frames))
                elif self.multi_frame_vision and self.contrastive_vision_indices is not None:
                    # 视觉: 只加载指定帧 (对比学习用, 大幅减少 backbone 开销)
                    future_frames = []
                    for idx in self.contrastive_vision_indices:
                        ft = min(
                            start_ts + self.future_offset + idx * self.temporal_stride,
                            episode_len - 1,
                        )
                        raw = self._get_cam_raw(ep_data, cam_name, ft)
                        future_frames.append(self._process_cam(cam_name, raw, ft))
                    future_cam_images.append(torch.stack(future_frames))
                elif self.multi_frame_vision:
                    future_frames = []
                    for i in range(self.horizon):
                        ft = min(
                            start_ts + self.future_offset + i * self.temporal_stride,
                            episode_len - 1,
                        )
                        raw = self._get_cam_raw(ep_data, cam_name, ft)
                        future_frames.append(self._process_cam(cam_name, raw, ft))
                    future_cam_images.append(torch.stack(future_frames))
                else:
                    raw = self._get_cam_raw(ep_data, cam_name, future_ts)
                    future_cam_images.append(self._process_cam(cam_name, raw, future_ts))

            # 历史 k 帧
            history_cam_images = []
            for cam_name in self.camera_names:
                frames = []
                for i in range(self.history_len):
                    hist_ts = max(
                        0,
                        start_ts - (self.history_len - 1 - i) * self.temporal_stride,
                    )
                    raw = self._get_cam_raw(ep_data, cam_name, hist_ts)
                    frames.append(self._process_cam(cam_name, raw, hist_ts))
                history_cam_images.append(torch.stack(frames))

            # action chunk (or state trajectory if use_state_trajectory)
            if self.use_state_trajectory:
                # state[t+offset:t+offset+chunk] — future state trajectory
                # as foresight conditioning. offset=1 preserves the old path;
                # offset=6 matches the DP action_offset=6 experiment.
                action_indices = self._chunk_indices(
                    start_ts, self.action_offset, self.chunk_size, episode_len)
                action = ep_data['qpos'][action_indices]
                action_len = len(action_indices)
            else:
                action_indices = self._chunk_indices(
                    start_ts, self.action_offset, self.chunk_size, episode_len)
                action = ep_data['action'][action_indices]
                action_len = len(action_indices)

        # --- 回退: 从 hdf5 读取 (preload=False) ---
        else:
            if isinstance(episode_id, str) and os.path.isabs(episode_id):
                dataset_path = episode_id
            else:
                dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
            with h5py.File(dataset_path, 'r') as root:
                action_dataset = root[f'/{self.action_key}']
                episode_len = action_dataset.shape[0]

                min_required = max(
                    self._required_span(self.future_offset, self.horizon),
                    self._required_span(self.action_offset, self.chunk_size),
                )
                max_start = max(0, episode_len - min_required)
                start_ts = np.random.randint(max_start + 1)
                future_ts = min(
                    start_ts + self.future_offset + (self.horizon - 1) * self.temporal_stride,
                    episode_len - 1,
                )

                qpos = root[f'/observations/{self.proprio_key}'][start_ts]

                if self.image_size is None and self.has_image_camera:
                    if 'image_height' in root.attrs:
                        self.image_size = (root.attrs['image_height'], root.attrs['image_width'])
                    else:
                        first_cam = [c for c in self.camera_names if c not in ('gelsight', 'blank')][0]
                        img_shape = root[f'/observations/images/{first_cam}'].shape
                        self.image_size = (img_shape[1], img_shape[2])

                all_cam_images = []
                for cam_name in self.camera_names:
                    if cam_name == 'gelsight' and self.tactile_mode == 'marker' and self.tactile_vae_window > 0:
                        T = self.tactile_vae_window
                        frames = []
                        for i in range(T):
                            hist_ts = max(0, start_ts - (T - 1 - i))
                            frames.append(self._load_cam_images(root, cam_name, hist_ts))
                        all_cam_images.append(torch.stack(frames))
                    else:
                        all_cam_images.append(self._load_cam_images(root, cam_name, start_ts))

                future_cam_images = []
                for cam_name in self.camera_names:
                    if cam_name == 'gelsight' and self.tactile_mode == 'marker':
                        future_frames = []
                        for i in range(self.horizon):
                            ft = min(
                                start_ts + self.future_offset + i * self.temporal_stride,
                                episode_len - 1,
                            )
                            if self.tactile_vae_window > 0:
                                T = self.tactile_vae_window
                                window_frames = []
                                for i in range(T):
                                    wt = max(0, ft - (T - 1 - i))
                                    window_frames.append(self._load_cam_images(root, cam_name, wt))
                                future_frames.append(torch.stack(window_frames))
                            else:
                                future_frames.append(self._load_cam_images(root, cam_name, ft))
                        future_cam_images.append(torch.stack(future_frames))
                    elif self.multi_frame_vision and self.contrastive_vision_indices is not None:
                        future_frames = []
                        for idx in self.contrastive_vision_indices:
                            ft = min(
                                start_ts + self.future_offset + idx * self.temporal_stride,
                                episode_len - 1,
                            )
                            future_frames.append(self._load_cam_images(root, cam_name, ft))
                        future_cam_images.append(torch.stack(future_frames))
                    elif self.multi_frame_vision:
                        future_frames = []
                        for i in range(self.horizon):
                            ft = min(
                                start_ts + self.future_offset + i * self.temporal_stride,
                                episode_len - 1,
                            )
                            future_frames.append(self._load_cam_images(root, cam_name, ft))
                        future_cam_images.append(torch.stack(future_frames))
                    else:
                        future_cam_images.append(self._load_cam_images(root, cam_name, future_ts))

                history_cam_images = []
                for cam_name in self.camera_names:
                    frames = []
                    for i in range(self.history_len):
                        hist_ts = max(
                            0,
                            start_ts - (self.history_len - 1 - i) * self.temporal_stride,
                        )
                        frames.append(self._load_cam_images(root, cam_name, hist_ts))
                    history_cam_images.append(torch.stack(frames))

                if self.use_state_trajectory:
                    qpos_dataset = root[f'/observations/{self.proprio_key}']
                    action_indices = self._chunk_indices(
                        start_ts, self.action_offset, self.chunk_size, episode_len)
                    action = qpos_dataset[action_indices]
                    action_len = len(action_indices)
                else:
                    action_indices = self._chunk_indices(
                        start_ts, self.action_offset, self.chunk_size, episode_len)
                    action = action_dataset[action_indices]
                    action_len = len(action_indices)

        # normalize
        if self.use_state_trajectory:
            qpos, action = self.action_qpos_normalize(qpos=qpos, action=action,
                                                       action_as_qpos=True)
        else:
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
