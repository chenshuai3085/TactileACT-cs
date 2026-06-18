"""
Train Diffusion Policy + Frozen TactileVAE (concat baseline).

Based on train_dp_official.py, adds frozen TactileVAE encoding of
8-frame marker_offset history, concatenated to obs_cond.

Vision: same as train_dp_official.py
  - ResNet18 + AdaptiveAvgPool → 512 dim/camera (GroupNorm, no BN)
  - Independent encoder per camera (deepcopy)
  - Resize(240,320) + RandomCrop(216,288) / CenterCrop(216,288)

Tactile: Frozen TactileVAE
  - Input: 8-frame marker_offset history from left sensor (B, 8, 9, 9, 2)
  - Normalized internally: mean=[0.2102, -0.6422], std=[1.6805, 3.6717]
  - TactileVAE.encode_single_frame → (B, latent_dim, 3, 3) → flatten → 144 dim
  - Completely frozen (requires_grad=False)

obs_cond = [vis_feat(512*n_cams) | tac_feat(144) | qpos(7)] * obs_horizon
         = (512*2 + 144 + 7) * 2 = 2350

Training: same as official (EMA, TopK(5), full data, no val split)
"""
import os, sys, json, argparse, copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from tqdm import tqdm
import h5py
from torchvision import transforms

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'TFAC_V5'))
from utils import set_seed
from network import ConditionalUnet1D, get_resnet, replace_bn_with_gn
from tactile_vae import TactileVAE, build_tactile_vae


# ==================== EMA ====================
class EMAModel:
    def __init__(self, model, inv_gamma=1.0, power=0.75, max_value=0.9999):
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}
        self.inv_gamma = inv_gamma
        self.power = power
        self.max_value = max_value
        self.step_count = 0

    def get_decay(self, step):
        return min(1 - (1 + step / self.inv_gamma) ** (-self.power), self.max_value)

    @torch.no_grad()
    def update(self, model):
        self.step_count += 1
        decay = self.get_decay(self.step_count)
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(decay).add_(v, alpha=1 - decay)

    def apply_to(self, model):
        model.load_state_dict(self.shadow)

    def state_dict(self):
        return self.shadow


# ==================== Vision Encoder (same as official) ====================
class OfficialVisionEncoder(nn.Module):
    """Per-camera ResNet18 + AdaptiveAvgPool → 512 dim, GroupNorm."""
    def __init__(self, camera_names):
        super().__init__()
        self.camera_names = camera_names
        base = get_resnet('resnet18')
        replace_bn_with_gn(base, features_per_group=16)
        self.encoders = nn.ModuleDict()
        for cam in camera_names:
            self.encoders[cam] = copy.deepcopy(base)
        self.feat_dim = 512

    def forward(self, images_dict):
        features = []
        for cam in self.camera_names:
            feat = self.encoders[cam](images_dict[cam])
            features.append(feat)
        return torch.cat(features, dim=-1)


# ==================== Frozen TactileVAE Encoder ====================
class FrozenTactileVAEEncoder(nn.Module):
    """
    Frozen TactileVAE: encode 8-frame marker_offset → flat latent vector.
    Input: (B, 8, 9, 9, 2) raw marker_offset (normalized internally)
    Output: (B, latent_dim * 3 * 3) = (B, 144) for latent_dim=16
    """
    DEFAULT_TAC_MEAN = np.array([0.2102, -0.6422], dtype=np.float32)
    DEFAULT_TAC_STD = np.array([1.6805, 3.6717], dtype=np.float32)

    def __init__(self, vae_checkpoint_path, latent_dim=16, temporal_window=8):
        super().__init__()
        self.vae = build_tactile_vae(latent_dim=latent_dim, temporal_window=temporal_window)

        tac_mean = self.DEFAULT_TAC_MEAN
        tac_std = self.DEFAULT_TAC_STD
        if vae_checkpoint_path and os.path.exists(vae_checkpoint_path):
            ckpt = torch.load(vae_checkpoint_path, map_location='cpu')
            if 'model_state_dict' in ckpt:
                self.vae.load_state_dict(ckpt['model_state_dict'])
            else:
                self.vae.load_state_dict(ckpt)
            print(f"Loaded TactileVAE: {vae_checkpoint_path}")

            norm_stats = ckpt.get('norm_stats') if isinstance(ckpt, dict) else None
            if norm_stats is not None and 'mean' in norm_stats and 'std' in norm_stats:
                tac_mean = np.array(norm_stats['mean'], dtype=np.float32)
                tac_std = np.array(norm_stats['std'], dtype=np.float32)
                print(f"Loaded TactileVAE norm_stats: mean={tac_mean.tolist()}, std={tac_std.tolist()}")
            else:
                print(f"WARNING: TactileVAE checkpoint has no norm_stats; using default mean/std.")
        else:
            print(f"WARNING: TactileVAE checkpoint not found: {vae_checkpoint_path}")

        self.vae.eval()
        self.vae.requires_grad_(False)

        self.feat_dim = latent_dim * 3 * 3  # 16 * 9 = 144
        self.register_buffer('tac_mean', torch.tensor(tac_mean))
        self.register_buffer('tac_std', torch.tensor(tac_std))

    def normalize_marker(self, marker_offset):
        return (marker_offset - self.tac_mean) / self.tac_std

    @torch.no_grad()
    def forward(self, marker_seq):
        """
        Args:
            marker_seq: (B, T, 9, 9, 2) raw marker_offset
        Returns:
            (B, feat_dim) flattened latent
        """
        marker_norm = self.normalize_marker(marker_seq)
        z_last, _ = self.vae.encode_single_frame(marker_norm)
        return z_last.flatten(1)


# ==================== Dataset ====================
class DPTacConcatDataset(torch.utils.data.Dataset):
    """Sliding-window dataset for DP + tactile concat training.

    By default this keeps the original fast path: all resized images are
    preloaded into RAM.  For larger board datasets, lazy_images=True keeps only
    qpos/action/marker in RAM and reads the needed image frames from HDF5 in
    __getitem__, avoiding RAM/swap exhaustion.
    """

    def __init__(self, episode_entries, camera_names, norm_stats,
                 pred_horizon, obs_horizon=2, tac_history=8,
                 proprio_key="proprio_joint", action_key="actions/joint_abs",
                 tac_side="left",
                 resize_shape=(240, 320), crop_shape=(216, 288), is_train=True,
                 lazy_images=False, image_cache_dir=None, max_train_windows=None, seed=0):
        self.camera_names = camera_names
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.tac_history = tac_history
        self.tac_side = tac_side
        self.lazy_images = lazy_images
        self.image_cache_dir = image_cache_dir
        self.use_image_cache = image_cache_dir is not None
        self.proprio_key = proprio_key
        self.action_key = action_key
        self.max_train_windows = max_train_windows
        self.seed = seed

        self.action_min = np.array(norm_stats["action_min"], dtype=np.float32)
        self.action_max = np.array(norm_stats["action_max"], dtype=np.float32)
        self.qpos_min = np.array(norm_stats["qpos_min"], dtype=np.float32)
        self.qpos_max = np.array(norm_stats["qpos_max"], dtype=np.float32)

        self.resize_transform = transforms.Resize(resize_shape)
        if is_train:
            self.crop_transform = transforms.RandomCrop(crop_shape)
        else:
            self.crop_transform = transforms.CenterCrop(crop_shape)
        self.image_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Preload low-dimensional data.  Images are preloaded only in the
        # original fast path; lazy mode reads image frames on demand.
        self.episodes = []
        n_skipped = 0
        desc = "Indexing episodes" if (lazy_images or self.use_image_cache) else "Preloading episodes"
        for ds_dir, ep_id in tqdm(episode_entries, desc=desc):
            path = os.path.join(ds_dir, f'episode_{ep_id}.hdf5')
            try:
                ep_data = self._preload_episode(path, camera_names, proprio_key, action_key)
                self.episodes.append(ep_data)
            except Exception:
                n_skipped += 1

        total_frames = sum(ep['qpos'].shape[0] for ep in self.episodes)
        if n_skipped:
            tag = "cache-index" if self.use_image_cache else ("lazy-index" if lazy_images else "preload")
            print(f"  [{tag}] skipped {n_skipped} corrupted episodes")

        self.indices = []
        for ep_idx, ep in enumerate(self.episodes):
            ep_len = ep['qpos'].shape[0]
            for start_ts in range(max(1, ep_len - pred_horizon + 1)):
                self.indices.append((ep_idx, start_ts))
        if max_train_windows is not None and len(self.indices) > max_train_windows:
            rng = np.random.default_rng(seed)
            chosen = np.sort(rng.choice(len(self.indices), max_train_windows, replace=False))
            self.indices = [self.indices[i] for i in chosen]

        mode = "cached-images" if self.use_image_cache else ("lazy-images" if lazy_images else "preload-images")
        print(f"  DPTacConcatDataset ({mode}): {len(self.episodes)} episodes, "
              f"{total_frames} total frames, {len(self.indices)} windows "
              f"({'train' if is_train else 'val'})")

    def _preload_episode(self, path, camera_names, proprio_key, action_key):
        with h5py.File(path, 'r') as f:
            qpos = f[f'observations/{proprio_key}'][()].astype(np.float32)
            action = f[f'{action_key}'][()].astype(np.float32)
            images = None
            if not self.lazy_images and not self.use_image_cache:
                images = {}
                for cam in camera_names:
                    raw = f[f'observations/images/{cam}'][()]  # (T, H, W, 3) uint8
                    # Pre-resize + normalize at load time, store as fp16 to save RAM
                    # Shape: (T, 3, resize_H, resize_W) fp16
                    T = raw.shape[0]
                    imgs_resized = []
                    for i in range(T):
                        imgs_resized.append(self._resize_normalize_raw_image(raw[i]).half())
                    images[cam] = torch.stack(imgs_resized)  # (T, 3, 240, 320) fp16
            # Load marker_offset for left sensor
            marker = f[f'observations/tac/{self.tac_side}/marker_offset'][()].astype(np.float32)
        cache_paths = None
        if self.use_image_cache:
            cache_paths = {
                cam: self._cache_path(path, cam)
                for cam in camera_names
            }
            missing = [p for p in cache_paths.values() if not os.path.exists(p)]
            if missing:
                raise FileNotFoundError(f"Missing image cache for {path}: {missing[:2]}")
        return {
            'path': path,
            'qpos': qpos,
            'action': action,
            'images': images,
            'marker': marker,
            'cache_paths': cache_paths,
            'cache_arrays': None,
        }

    def __len__(self):
        return len(self.indices)

    def _minmax_norm(self, x, xmin, xmax):
        return (x - xmin) / (xmax - xmin + 1e-8) * 2 - 1

    def _process_image(self, img_fp16):
        """Images are already resized+normalized at preload. Only crop here."""
        return self.crop_transform(img_fp16.float())

    def _resize_normalize_raw_image(self, raw_img):
        img_t = torch.from_numpy(np.asarray(raw_img)).float().div_(255.0).permute(2, 0, 1)
        img_t = self.resize_transform(img_t)
        return self.image_normalize(img_t)

    def _load_lazy_images(self, ep, obs_indices):
        images = {cam: [] for cam in self.camera_names}
        with h5py.File(ep['path'], 'r') as f:
            for t in obs_indices:
                for cam in self.camera_names:
                    raw = f[f'observations/images/{cam}'][t]
                    images[cam].append(self.crop_transform(self._resize_normalize_raw_image(raw)))
        return images

    def _cache_path(self, episode_path, cam):
        ep_name = os.path.splitext(os.path.basename(episode_path))[0]
        parent = os.path.basename(os.path.dirname(episode_path))
        return os.path.join(self.image_cache_dir, parent, f"{ep_name}_{cam}_rnorm.npy")

    def _load_cached_images(self, ep, obs_indices):
        images = {cam: [] for cam in self.camera_names}
        if ep.get('cache_arrays') is None:
            ep['cache_arrays'] = {
                cam: np.load(ep['cache_paths'][cam], mmap_mode='r')
                for cam in self.camera_names
            }
        for cam in self.camera_names:
            arr = ep['cache_arrays'][cam]
            for t in obs_indices:
                img = torch.from_numpy(np.array(arr[t], copy=True)).float()
                images[cam].append(self.crop_transform(img))
        return images

    def _get_marker_history(self, ep, t):
        """Get tac_history frames of marker_offset ending at time t."""
        ep_len = ep['marker'].shape[0]
        frames = []
        for k in range(self.tac_history):
            idx = t - self.tac_history + 1 + k
            idx = max(0, min(idx, ep_len - 1))
            frames.append(ep['marker'][idx])
        return np.stack(frames, axis=0)  # (tac_history, 9, 9, 2)

    def __getitem__(self, index):
        ep_idx, start_ts = self.indices[index]
        ep = self.episodes[ep_idx]
        ep_len = ep['qpos'].shape[0]

        obs_indices = [max(0, start_ts - self.obs_horizon + 1 + k)
                       for k in range(self.obs_horizon)]

        all_images = {cam: [] for cam in self.camera_names}
        all_qpos = []
        all_marker_hist = []

        if self.use_image_cache:
            all_images = self._load_cached_images(ep, obs_indices)
        elif self.lazy_images:
            all_images = self._load_lazy_images(ep, obs_indices)

        for t in obs_indices:
            all_qpos.append(ep['qpos'][t])
            if not self.lazy_images and not self.use_image_cache:
                for cam in self.camera_names:
                    all_images[cam].append(self._process_image(ep['images'][cam][t]))
            all_marker_hist.append(self._get_marker_history(ep, t))

        action_end = min(start_ts + self.pred_horizon, ep_len)
        action = ep['action'][start_ts:action_end]
        if action.shape[0] < self.pred_horizon:
            pad = np.tile(action[-1:], (self.pred_horizon - action.shape[0], 1))
            action = np.concatenate([action, pad], axis=0)

        images_per_cam = {cam: torch.stack(imgs) for cam, imgs in all_images.items()}
        qpos = np.stack(all_qpos)
        marker_hist = np.stack(all_marker_hist, axis=0)  # (obs_horizon, tac_history, 9, 9, 2)

        return {
            'images': images_per_cam,
            'marker_hist': torch.from_numpy(marker_hist),
            'qpos': torch.from_numpy(self._minmax_norm(qpos, self.qpos_min, self.qpos_max)),
            'action': torch.from_numpy(self._minmax_norm(action, self.action_min, self.action_max)),
        }


def dp_tac_concat_collate(batch):
    camera_names = list(batch[0]['images'].keys())
    images = {}
    for cam in camera_names:
        images[cam] = torch.stack([b['images'][cam] for b in batch])
    return {
        'images': images,
        'marker_hist': torch.stack([b['marker_hist'] for b in batch]),
        'qpos': torch.stack([b['qpos'] for b in batch]),
        'action': torch.stack([b['action'] for b in batch]),
    }


def get_minmax_stats(dataset_dirs, proprio_key, action_key):
    all_qpos, all_action = [], []
    for ds_dir in dataset_dirs:
        episode_files = sorted([f for f in os.listdir(ds_dir)
                                if f.startswith('episode_') and f.endswith('.hdf5')])
        for ef in tqdm(episode_files, desc=f"Min/max stats ({os.path.basename(ds_dir)})"):
            try:
                with h5py.File(os.path.join(ds_dir, ef), 'r') as root:
                    all_qpos.append(root[f'/observations/{proprio_key}'][()])
                    all_action.append(root[f'/{action_key}'][()])
            except Exception:
                pass
    all_qpos = np.concatenate(all_qpos, axis=0)
    all_action = np.concatenate(all_action, axis=0)
    return {
        "qpos_min": all_qpos.min(axis=0),
        "qpos_max": all_qpos.max(axis=0),
        "action_min": all_action.min(axis=0),
        "action_max": all_action.max(axis=0),
    }


def build_image_cache(dataset_dirs, camera_names, image_cache_dir, resize_shape):
    """Build per-episode resized+normalized image cache as float16 .npy files."""
    os.makedirs(image_cache_dir, exist_ok=True)
    resize_transform = transforms.Resize(resize_shape)
    mean_np = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
    std_np = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
    image_normalize = transforms.Normalize(
        mean=mean_np.reshape(3).tolist(), std=std_np.reshape(3).tolist())
    built = 0
    skipped = 0
    reused = 0
    for ds_dir in dataset_dirs:
        episode_files = sorted([f for f in os.listdir(ds_dir)
                                if f.startswith('episode_') and f.endswith('.hdf5')])
        parent = os.path.basename(ds_dir)
        out_dir = os.path.join(image_cache_dir, parent)
        os.makedirs(out_dir, exist_ok=True)
        for ef in tqdm(episode_files, desc=f"Image cache ({parent})"):
            ep_name = os.path.splitext(ef)[0]
            ep_path = os.path.join(ds_dir, ef)
            try:
                with h5py.File(ep_path, 'r') as f:
                    for cam in camera_names:
                        out_path = os.path.join(out_dir, f"{ep_name}_{cam}_rnorm.npy")
                        if os.path.exists(out_path):
                            reused += 1
                            continue
                        raw = f[f'observations/images/{cam}']
                        T = raw.shape[0]
                        arr = np.lib.format.open_memmap(
                            out_path, mode='w+', dtype=np.float16,
                            shape=(T, 3, resize_shape[0], resize_shape[1]))
                        if raw.shape[1] == resize_shape[0] and raw.shape[2] == resize_shape[1]:
                            imgs = raw[()].astype(np.float32) / 255.0
                            imgs = np.transpose(imgs, (0, 3, 1, 2))
                            imgs = (imgs - mean_np) / std_np
                            arr[:] = imgs.astype(np.float16)
                        else:
                            for i in range(T):
                                img = torch.from_numpy(raw[i]).float().div_(255.0).permute(2, 0, 1)
                                img = resize_transform(img)
                                img = image_normalize(img)
                                arr[i] = img.numpy().astype(np.float16)
                        arr.flush()
                        built += 1
            except Exception as exc:
                print(f"  [image-cache] skip {ep_path}: {exc}")
                skipped += 1
    return {"built": built, "reused": reused, "skipped": skipped, "cache_dir": image_cache_dir}


# ==================== Training ====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Dataset directory (comma-separated for multiple)')
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--camera_names', type=str, default='global,wrist')
    parser.add_argument('--proprio_key', type=str, default='proprio_joint')
    parser.add_argument('--action_key', type=str, default='actions/joint_abs')
    parser.add_argument('--tac_side', type=str, default='left')
    parser.add_argument('--tac_history', type=int, default=8)
    parser.add_argument('--vae_checkpoint', type=str,
                        default='/sharedata/chenshuai/ckpt/TactileVae/best_tactile_vae.pt')
    parser.add_argument('--vae_latent_dim', type=int, default=16)
    parser.add_argument('--pred_horizon', type=int, default=16)
    parser.add_argument('--obs_horizon', type=int, default=2)
    parser.add_argument('--n_action_steps', type=int, default=8)
    parser.add_argument('--resize_shape', type=str, default='240,320')
    parser.add_argument('--crop_shape', type=str, default='216,288')
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--num_train_timesteps', type=int, default=100)
    parser.add_argument('--num_inference_steps', type=int, default=100)
    parser.add_argument('--diffusion_step_embed_dim', type=int, default=128)
    parser.add_argument('--down_dims', type=str, default='512,1024,2048')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--no_ema', action='store_true', default=False)
    parser.add_argument('--save_freq', type=int, default=50)
    parser.add_argument('--latest_freq', type=int, default=1,
                        help='Save dp_latest.pth every N epochs. Default 1 preserves the original every-epoch behavior. Set 0 to disable.')
    parser.add_argument('--topk_k', type=int, default=5,
                        help='Number of train-loss top-k checkpoints to retain. Set 0 to disable.')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU ids, comma-separated (e.g., 0,1,2,3)')
    parser.add_argument('--lazy_images', action='store_true', default=False,
                        help='Read image frames from HDF5 on demand instead of preloading all resized images into RAM.')
    parser.add_argument('--image_cache_dir', type=str, default=None,
                        help='Directory with resized+normalized image .npy caches. Avoids RAM preload and repeated HDF5 image transforms.')
    parser.add_argument('--build_image_cache', action='store_true', default=False,
                        help='Build image_cache_dir before training. Requires --image_cache_dir.')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='DataLoader workers. Defaults to 0 for lazy/cache images, otherwise 8 for multi-GPU or 4 for single GPU.')
    parser.add_argument('--max_train_windows', type=int, default=None,
                        help='Optional random subset of sliding windows for quick smoke/debug training.')
    parser.add_argument('--max_val_windows', type=int, default=None,
                        help='Optional random subset of validation windows for quick smoke/debug validation.')
    parser.add_argument('--val_ratio', type=float, default=0.0,
                        help='Episode-level validation split ratio. If >0, dp_best.pth is selected by val loss.')
    parser.add_argument('--val_interval', type=int, default=5,
                        help='Run validation every N epochs when val_ratio > 0.')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Print batch-level training progress every N batches. Set 0 to disable.')
    parser.add_argument('--max_steps_per_epoch', type=int, default=None,
                        help='Optional cap on train batches per epoch for debug or quick subset training.')
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    gpu_ids = [int(x) for x in args.gpu.split(',')]
    device = torch.device(f'cuda:{gpu_ids[0]}')
    use_multi_gpu = len(gpu_ids) > 1
    if use_multi_gpu:
        print(f"Using DataParallel on GPUs: {gpu_ids}")
    camera_names = args.camera_names.split(',')
    down_dims = [int(x) for x in args.down_dims.split(',')]
    resize_shape = tuple(int(x) for x in args.resize_shape.split(','))
    crop_shape = tuple(int(x) for x in args.crop_shape.split(','))

    dataset_dirs = [d.strip() for d in args.dataset_dir.split(',')]
    for d in dataset_dirs:
        assert os.path.isdir(d), f"Dataset dir not found: {d}"
    print(f"Dataset dirs: {dataset_dirs}")

    norm_stats = get_minmax_stats(dataset_dirs, args.proprio_key, args.action_key)

    if args.build_image_cache:
        assert args.image_cache_dir, "--build_image_cache requires --image_cache_dir"
        cache_result = build_image_cache(dataset_dirs, camera_names, args.image_cache_dir, resize_shape)
        print(f"Image cache result: {cache_result}")

    all_entries = []
    for ds_dir in dataset_dirs:
        episode_files = sorted([f for f in os.listdir(ds_dir)
                                if f.startswith('episode_') and f.endswith('.hdf5')])
        for ef in episode_files:
            ep_id = int(ef.split('_')[1].split('.')[0])
            all_entries.append((ds_dir, ep_id))
    print(f"Total episodes: {len(all_entries)}")

    rng = np.random.default_rng(args.seed)
    shuffled_entries = list(all_entries)
    rng.shuffle(shuffled_entries)
    if args.val_ratio > 0 and len(shuffled_entries) > 1:
        n_val = max(1, int(round(len(shuffled_entries) * args.val_ratio)))
        n_val = min(n_val, len(shuffled_entries) - 1)
        val_entries = shuffled_entries[:n_val]
        train_entries = shuffled_entries[n_val:]
    else:
        val_entries = []
        train_entries = shuffled_entries

    train_dataset = DPTacConcatDataset(
        train_entries, camera_names, norm_stats,
        args.pred_horizon, args.obs_horizon, args.tac_history,
        args.proprio_key, args.action_key, args.tac_side,
        resize_shape=resize_shape, crop_shape=crop_shape, is_train=True,
        lazy_images=args.lazy_images, image_cache_dir=args.image_cache_dir,
        max_train_windows=args.max_train_windows,
        seed=args.seed,
    )

    if args.num_workers is not None:
        num_workers = args.num_workers
    else:
        num_workers = 0 if (args.lazy_images or args.image_cache_dir) else (8 if use_multi_gpu else 4)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=True,
                              collate_fn=dp_tac_concat_collate)
    val_loader = None
    if val_entries:
        val_dataset = DPTacConcatDataset(
            val_entries, camera_names, norm_stats,
            args.pred_horizon, args.obs_horizon, args.tac_history,
            args.proprio_key, args.action_key, args.tac_side,
            resize_shape=resize_shape, crop_shape=crop_shape, is_train=False,
            lazy_images=args.lazy_images, image_cache_dir=args.image_cache_dir,
            max_train_windows=args.max_val_windows,
            seed=args.seed,
        )
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=num_workers, pin_memory=True,
                                collate_fn=dp_tac_concat_collate)

    # Models
    action_dim = norm_stats['action_min'].shape[0]
    qpos_dim = norm_stats['qpos_min'].shape[0]

    vision_encoder = OfficialVisionEncoder(camera_names).to(device)
    vis_feat_dim = vision_encoder.feat_dim  # 512 per camera
    n_cams = len(camera_names)

    tac_encoder = FrozenTactileVAEEncoder(
        args.vae_checkpoint, latent_dim=args.vae_latent_dim,
        temporal_window=args.tac_history,
    ).to(device)
    tac_feat_dim = tac_encoder.feat_dim  # 144

    global_cond_dim = (vis_feat_dim * n_cams + tac_feat_dim + qpos_dim) * args.obs_horizon

    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=global_cond_dim,
        diffusion_step_embed_dim=args.diffusion_step_embed_dim,
        down_dims=down_dims,
        kernel_size=5,
    ).to(device)

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=args.num_train_timesteps,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon',
    )

    # Wrap with DataParallel for multi-GPU (all models)
    if use_multi_gpu:
        vision_encoder = nn.DataParallel(vision_encoder, device_ids=gpu_ids)
        noise_pred_net = nn.DataParallel(noise_pred_net, device_ids=gpu_ids)
        tac_encoder = nn.DataParallel(tac_encoder, device_ids=gpu_ids)

    use_ema = not args.no_ema
    # EMA operates on the underlying module (unwrapped)
    vis_module = vision_encoder.module if use_multi_gpu else vision_encoder
    net_module = noise_pred_net.module if use_multi_gpu else noise_pred_net
    ema_vis = EMAModel(vis_module) if use_ema else None
    ema_net = EMAModel(net_module) if use_ema else None

    # TactileVAE is frozen — only optimize vision_encoder + noise_pred_net
    all_params = list(net_module.parameters()) + list(vis_module.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=args.lr,
                                  betas=(0.95, 0.999), weight_decay=args.weight_decay)

    total_steps = len(train_loader) * args.epochs
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))
    lr_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Save config
    config = vars(args)
    config['dataset_dirs'] = dataset_dirs
    config['resize_shape'] = list(resize_shape)
    config['crop_shape'] = list(crop_shape)
    config['action_dim'] = int(action_dim)
    config['global_cond_dim'] = int(global_cond_dim)
    config['vis_feat_dim'] = int(vis_feat_dim)
    config['tac_feat_dim'] = int(tac_feat_dim)
    config['down_dims'] = down_dims
    config['use_ema'] = use_ema
    config['n_train'] = len(train_entries)
    config['n_val'] = len(val_entries)
    config['variant'] = 'tactile_vae_frozen'
    config['gpu_ids'] = gpu_ids
    config['num_workers_resolved'] = num_workers
    config['image_loading_mode'] = 'cached' if args.image_cache_dir else ('lazy' if args.lazy_images else 'preload')
    ns = {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in norm_stats.items()}
    config['norm_stats'] = ns
    with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # Verify frozen
    frozen_params = sum(1 for p in tac_encoder.parameters() if not p.requires_grad)
    total_tac_params = sum(1 for p in tac_encoder.parameters())
    print(f"=== DP + Frozen TactileVAE (concat) ===")
    print(f"GPUs: {gpu_ids} ({'DataParallel' if use_multi_gpu else 'single'})")
    print(f"action_dim={action_dim}, global_cond_dim={global_cond_dim}")
    print(f"vis_feat_dim={vis_feat_dim}/camera, cameras={camera_names}")
    print(f"tac_feat_dim={tac_feat_dim} (frozen: {frozen_params}/{total_tac_params} params)")
    print(f"pred_horizon={args.pred_horizon}, obs_horizon={args.obs_horizon}")
    print(f"resize={resize_shape}, crop={crop_shape}")
    print(f"EMA={use_ema}, tac_history={args.tac_history}, tac_side={args.tac_side}")
    print(f"Train: {len(train_entries)} eps ({len(train_dataset)} windows)")
    if val_loader is not None:
        print(f"Val: {len(val_entries)} eps ({len(val_dataset)} windows), interval={args.val_interval}")
    print(f"Batch: {args.batch_size}, Epochs: {args.epochs}")

    # TopK checkpoint manager
    topk_train_losses = {}

    train_losses = []
    val_losses = []
    best_metric = float('inf')
    best_metric_name = 'val_loss' if val_loader is not None else 'train_loss'
    global_step = 0

    def _run_validation():
        vision_encoder.eval()
        noise_pred_net.eval()
        losses = []
        with torch.no_grad():
            for batch in val_loader:
                B = batch['qpos'].shape[0]
                qpos = batch['qpos'].to(device)
                action = batch['action'].to(device)
                marker_hist = batch['marker_hist'].to(device)

                obs_feats = []
                for t in range(args.obs_horizon):
                    imgs_t = {cam: batch['images'][cam][:, t].to(device) for cam in camera_names}
                    vf = vision_encoder(imgs_t)
                    marker_t = marker_hist[:, t]
                    tf = tac_encoder(marker_t)
                    obs_feats.append(torch.cat([vf, tf, qpos[:, t]], dim=-1))
                obs_cond = torch.cat(obs_feats, dim=-1)

                noise = torch.randn_like(action)
                ts = torch.randint(0, args.num_train_timesteps, (B,), device=device).long()
                noisy = noise_scheduler.add_noise(action, noise, ts)
                pred = noise_pred_net(noisy, ts, global_cond=obs_cond)
                losses.append(nn.functional.mse_loss(pred, noise).item())
        return float(np.mean(losses))

    def _save_ckpt(path, epoch, train_loss, val_loss=None, include_optimizer=False):
        sd = {
            'epoch': epoch,
            'global_step': global_step,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_metric_name': best_metric_name,
            'best_metric': best_metric,
            'noise_pred_net': net_module.state_dict(),
            'vision_encoder': vis_module.state_dict(),
            'config': config,
            'norm_stats': ns,
        }
        if ema_vis:
            sd['ema_vis'] = ema_vis.state_dict()
            sd['ema_net'] = ema_net.state_dict()
        if include_optimizer:
            sd['optimizer'] = optimizer.state_dict()
            sd['lr_sched'] = lr_sched.state_dict()
        torch.save(sd, path)

    for epoch in range(args.epochs):
        vision_encoder.train()
        noise_pred_net.train()
        # tac_encoder stays in eval (frozen)
        ep_losses = []

        for batch_idx, batch in enumerate(train_loader):
            if args.max_steps_per_epoch is not None and batch_idx >= args.max_steps_per_epoch:
                break
            B = batch['qpos'].shape[0]
            qpos = batch['qpos'].to(device)                   # (B, obs_horizon, 7)
            action = batch['action'].to(device)               # (B, pred_horizon, 7)
            marker_hist = batch['marker_hist'].to(device)     # (B, obs_horizon, tac_history, 9, 9, 2)

            obs_feats = []
            for t in range(args.obs_horizon):
                # Vision
                imgs_t = {cam: batch['images'][cam][:, t].to(device) for cam in camera_names}
                vf = vision_encoder(imgs_t)  # (B, 512*n_cams)

                # Tactile (frozen)
                marker_t = marker_hist[:, t]  # (B, tac_history, 9, 9, 2)
                tf = tac_encoder(marker_t)    # (B, 144)

                # Concat: [vision | tactile | qpos]
                obs_feats.append(torch.cat([vf, tf, qpos[:, t]], dim=-1))

            obs_cond = torch.cat(obs_feats, dim=-1)  # (B, global_cond_dim)

            # Diffusion training
            noise = torch.randn_like(action)
            ts = torch.randint(0, args.num_train_timesteps, (B,), device=device).long()
            noisy = noise_scheduler.add_noise(action, noise, ts)

            pred = noise_pred_net(noisy, ts, global_cond=obs_cond)
            loss = nn.functional.mse_loss(pred, noise)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()
            lr_sched.step()
            global_step += 1

            if ema_vis:
                ema_vis.update(vis_module)
                ema_net.update(net_module)

            ep_losses.append(loss.item())
            if args.log_interval and (batch_idx == 0 or (batch_idx + 1) % args.log_interval == 0):
                print(f"  Ep {epoch+1} batch {batch_idx+1}/{len(train_loader)} "
                      f"loss={loss.item():.6f} lr={optimizer.param_groups[0]['lr']:.2e}",
                      flush=True)

        train_loss = np.mean(ep_losses)
        train_losses.append(train_loss)
        val_loss = None
        should_validate = val_loader is not None and (
            epoch == 0 or (epoch + 1) % max(1, args.val_interval) == 0 or (epoch + 1) == args.epochs
        )
        if should_validate:
            val_loss = _run_validation()
            val_losses.append((epoch + 1, val_loss))
            vision_encoder.train()
            noise_pred_net.train()

        # TopK checkpoint saving
        if args.topk_k > 0:
            if len(topk_train_losses) < args.topk_k:
                ckpt_path = os.path.join(args.save_dir, f'dp_topk_ep{epoch+1}_loss{train_loss:.4f}.pth')
                _save_ckpt(ckpt_path, epoch, train_loss, val_loss=val_loss)
                topk_train_losses[ckpt_path] = train_loss
            else:
                worst_path = max(topk_train_losses, key=topk_train_losses.get)
                if train_loss < topk_train_losses[worst_path]:
                    if os.path.exists(worst_path):
                        os.remove(worst_path)
                    del topk_train_losses[worst_path]
                    ckpt_path = os.path.join(args.save_dir, f'dp_topk_ep{epoch+1}_loss{train_loss:.4f}.pth')
                    _save_ckpt(ckpt_path, epoch, train_loss, val_loss=val_loss)
                    topk_train_losses[ckpt_path] = train_loss

        current_metric = val_loss if val_loss is not None else (float('inf') if val_loader is not None else train_loss)
        improved = current_metric < best_metric
        if improved:
            best_metric = current_metric
            _save_ckpt(os.path.join(args.save_dir, 'dp_best.pth'), epoch, train_loss,
                       val_loss=val_loss, include_optimizer=False)

        should_save_latest = args.latest_freq > 0 and (
            epoch == 0 or (epoch + 1) % args.latest_freq == 0 or (epoch + 1) == args.epochs
        )
        if should_save_latest:
            _save_ckpt(os.path.join(args.save_dir, 'dp_latest.pth'), epoch, train_loss,
                       val_loss=val_loss, include_optimizer=True)

        val_msg = f" | val={val_loss:.6f}" if val_loss is not None else ""
        topk_msg = f"{min(topk_train_losses.values()):.6f}" if topk_train_losses else "disabled"
        best_msg = f"{best_metric_name}={best_metric:.6f}" if np.isfinite(best_metric) else f"{best_metric_name}=pending"
        print(f"Ep {epoch+1}/{args.epochs} | train={train_loss:.6f} | "
              f"topk_train_best={topk_msg}{val_msg} | "
              f"best={best_msg}{' *' if improved else ''} | "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

        if (epoch + 1) % args.save_freq == 0:
            _save_ckpt(os.path.join(args.save_dir, f'dp_epoch{epoch+1}.pth'), epoch,
                       train_loss, val_loss=val_loss, include_optimizer=False)

    _save_ckpt(os.path.join(args.save_dir, 'dp_final.pth'), args.epochs - 1,
               train_losses[-1], val_loss=val_loss, include_optimizer=False)

    np.save(os.path.join(args.save_dir, 'train_losses.npy'), train_losses)
    np.save(os.path.join(args.save_dir, 'val_losses.npy'), np.array(val_losses, dtype=np.float32))
    metrics = {
        'train_losses': [float(x) for x in train_losses],
        'val_losses': [{'epoch': int(e), 'val_loss': float(v)} for e, v in val_losses],
        'best_metric_name': best_metric_name,
        'best_metric': float(best_metric),
    }
    with open(os.path.join(args.save_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)


if __name__ == '__main__':
    main()
