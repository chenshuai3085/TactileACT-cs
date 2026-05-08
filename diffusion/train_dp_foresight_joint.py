"""
Train Diffusion Policy + Foresight Joint (auxiliary foresight loss).

Based on train_dp_tac_concat.py, adds a Foresight auxiliary loss:
  1. Standard DP loss: MSE(noise_pred, noise)
  2. Predict x0 from noise prediction (clean action estimate)
  3. Feed x0 to pretrained Foresight → predict future tactile latent
  4. L1 loss between predicted and GT future tactile latent
  5. Gradient flows back to noise_pred_net, encouraging actions that
     lead to good tactile outcomes

Key differences from train_dp_tac_concat.py:
  - Loads pretrained LatentForesightPretrainModel (backbone+VAE frozen)
  - Dataset returns future_marker (marker_offset at t+foresight_horizon)
  - Training loop computes foresight auxiliary loss (only for low-t samples)
  - Optimizer includes Foresight trainable params
"""
import os, sys, json, argparse, copy, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from tqdm import tqdm
import h5py
from torchvision import transforms

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)

# Import TFAC_V5 modules FIRST (before diffusion/ goes on path)
# to avoid utils.py conflict (diffusion/utils.py vs TFAC_V5/utils.py)
_TFAC_V5 = os.path.join(ROOT, 'TFAC_V5')
sys.path.insert(0, _TFAC_V5)
from pretrain_latent_foresight import LatentForesightPretrainModel
from tactile_vae import TactileVAE, build_tactile_vae

# Now add diffusion/ and import its modules
_DIFFUSION = os.path.join(ROOT, 'diffusion')
if _DIFFUSION not in sys.path:
    sys.path.insert(0, _DIFFUSION)
from utils import set_seed
from network import ConditionalUnet1D, get_resnet, replace_bn_with_gn


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


# ==================== Vision Encoder ====================
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
    TAC_MEAN = np.array([0.2102, -0.6422], dtype=np.float32)
    TAC_STD = np.array([1.6805, 3.6717], dtype=np.float32)

    def __init__(self, vae_checkpoint_path, latent_dim=16, temporal_window=8):
        super().__init__()
        self.vae = build_tactile_vae(latent_dim=latent_dim, temporal_window=temporal_window)

        if vae_checkpoint_path and os.path.exists(vae_checkpoint_path):
            ckpt = torch.load(vae_checkpoint_path, map_location='cpu')
            if 'model_state_dict' in ckpt:
                self.vae.load_state_dict(ckpt['model_state_dict'])
            else:
                self.vae.load_state_dict(ckpt)
            print(f"Loaded TactileVAE: {vae_checkpoint_path}")
        else:
            print(f"WARNING: TactileVAE checkpoint not found: {vae_checkpoint_path}")

        self.vae.eval()
        self.vae.requires_grad_(False)

        self.feat_dim = latent_dim * 3 * 3  # 16 * 9 = 144
        self.register_buffer('tac_mean', torch.tensor(self.TAC_MEAN))
        self.register_buffer('tac_std', torch.tensor(self.TAC_STD))

    def normalize_marker(self, marker_offset):
        return (marker_offset - self.tac_mean) / self.tac_std

    @torch.no_grad()
    def forward(self, marker_seq):
        marker_norm = self.normalize_marker(marker_seq)
        z_last, _ = self.vae.encode_single_frame(marker_norm)
        return z_last.flatten(1)


# ==================== Dataset ====================
class DPForesightJointDataset(torch.utils.data.Dataset):
    """Same as DPTacConcatDataset but also returns future_marker for foresight GT."""

    def __init__(self, episode_entries, camera_names, norm_stats,
                 pred_horizon, obs_horizon=2, tac_history=8,
                 proprio_key="proprio_joint", action_key="actions/joint_abs",
                 tac_side="left", foresight_horizon=10,
                 resize_shape=(240, 320), crop_shape=(216, 288), is_train=True):
        self.camera_names = camera_names
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.tac_history = tac_history
        self.tac_side = tac_side
        self.foresight_horizon = foresight_horizon

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

        self.episodes = []
        n_skipped = 0
        for ds_dir, ep_id in tqdm(episode_entries, desc="Preloading episodes"):
            path = os.path.join(ds_dir, f'episode_{ep_id}.hdf5')
            try:
                ep_data = self._preload_episode(path, camera_names, proprio_key, action_key)
                self.episodes.append(ep_data)
            except OSError:
                n_skipped += 1

        total_frames = sum(ep['qpos'].shape[0] for ep in self.episodes)
        if n_skipped:
            print(f"  [preload] skipped {n_skipped} corrupted episodes")

        self.indices = []
        for ep_idx, ep in enumerate(self.episodes):
            ep_len = ep['qpos'].shape[0]
            for start_ts in range(max(1, ep_len - pred_horizon + 1)):
                self.indices.append((ep_idx, start_ts))

        print(f"  DPForesightJointDataset: {len(self.episodes)} episodes, "
              f"{total_frames} total frames, {len(self.indices)} windows "
              f"({'train' if is_train else 'val'})")

    def _preload_episode(self, path, camera_names, proprio_key, action_key):
        with h5py.File(path, 'r') as f:
            qpos = f[f'observations/{proprio_key}'][()].astype(np.float32)
            action = f[f'{action_key}'][()].astype(np.float32)
            images = {}
            for cam in camera_names:
                raw = f[f'observations/images/{cam}'][()]
                T = raw.shape[0]
                imgs_resized = []
                for i in range(T):
                    img_t = torch.from_numpy(raw[i]).float().div_(255.0).permute(2, 0, 1)
                    img_t = self.resize_transform(img_t)
                    img_t = self.image_normalize(img_t)
                    imgs_resized.append(img_t.half())
                images[cam] = torch.stack(imgs_resized)
            marker = f[f'observations/tac/{self.tac_side}/marker_offset'][()].astype(np.float32)
        return {'qpos': qpos, 'action': action, 'images': images, 'marker': marker}

    def __len__(self):
        return len(self.indices)

    def _minmax_norm(self, x, xmin, xmax):
        return (x - xmin) / (xmax - xmin + 1e-8) * 2 - 1

    def _process_image(self, img_fp16):
        return self.crop_transform(img_fp16.float())

    def _get_marker_history(self, ep, t):
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

        for t in obs_indices:
            all_qpos.append(ep['qpos'][t])
            for cam in self.camera_names:
                all_images[cam].append(self._process_image(ep['images'][cam][t]))
            all_marker_hist.append(self._get_marker_history(ep, t))

        action_end = min(start_ts + self.pred_horizon, ep_len)
        action = ep['action'][start_ts:action_end]
        if action.shape[0] < self.pred_horizon:
            pad = np.tile(action[-1:], (self.pred_horizon - action.shape[0], 1))
            action = np.concatenate([action, pad], axis=0)

        # Future marker for foresight GT
        future_t = min(start_ts + self.foresight_horizon, ep_len - 1)
        future_marker = self._get_marker_history(ep, future_t)  # (tac_history, 9, 9, 2)

        # Raw qpos (unnormalized) for foresight
        qpos_raw = ep['qpos'][obs_indices[-1]]  # last obs frame

        images_per_cam = {cam: torch.stack(imgs) for cam, imgs in all_images.items()}
        qpos = np.stack(all_qpos)
        marker_hist = np.stack(all_marker_hist, axis=0)

        return {
            'images': images_per_cam,
            'marker_hist': torch.from_numpy(marker_hist),
            'qpos': torch.from_numpy(self._minmax_norm(qpos, self.qpos_min, self.qpos_max)),
            'action': torch.from_numpy(self._minmax_norm(action, self.action_min, self.action_max)),
            'future_marker': torch.from_numpy(future_marker),  # (8, 9, 9, 2) raw
            'qpos_raw': torch.from_numpy(qpos_raw),  # (7,) raw for foresight
        }


def collate_fn(batch):
    camera_names = list(batch[0]['images'].keys())
    images = {}
    for cam in camera_names:
        images[cam] = torch.stack([b['images'][cam] for b in batch])
    return {
        'images': images,
        'marker_hist': torch.stack([b['marker_hist'] for b in batch]),
        'qpos': torch.stack([b['qpos'] for b in batch]),
        'action': torch.stack([b['action'] for b in batch]),
        'future_marker': torch.stack([b['future_marker'] for b in batch]),
        'qpos_raw': torch.stack([b['qpos_raw'] for b in batch]),
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
            except OSError:
                pass
    all_qpos = np.concatenate(all_qpos, axis=0)
    all_action = np.concatenate(all_action, axis=0)
    return {
        "qpos_min": all_qpos.min(axis=0),
        "qpos_max": all_qpos.max(axis=0),
        "action_min": all_action.min(axis=0),
        "action_max": all_action.max(axis=0),
    }


# ==================== Foresight Loading ====================
def load_foresight_model(ckpt_path, foresight_dir, device):
    """Load pretrained LatentForesightPretrainModel."""
    if foresight_dir is None:
        foresight_dir = os.path.dirname(ckpt_path)

    args_path = os.path.join(foresight_dir, "args.json")
    with open(args_path) as f:
        config = json.load(f)

    camera_names = config['camera_names']
    cam_backbone_mapping = {cam: 0 for cam in camera_names}

    model = LatentForesightPretrainModel(
        camera_names=camera_names,
        cam_backbone_mapping=cam_backbone_mapping,
        hidden_dim=config['hidden_dim'],
        state_dim=config['state_dim'],
        foresight_layers=config.get('foresight_layers', 3),
        foresight_nheads=config.get('foresight_nheads', 8),
        foresight_dim_feedforward=config.get('foresight_dim_feedforward', 2048),
        dropout=config.get('dropout', 0.1),
        tactile_mode=config.get('tactile_mode', 'marker'),
        max_history=config.get('max_history', 8),
        predict_horizon=config.get('predict_horizon', 1),
        tactile_vae_ckpt=config.get('tactile_vae_ckpt'),
        tactile_vae_latent_dim=config.get('tactile_vae_latent_dim', 16),
        use_delta_pred=config.get('use_delta_pred', False),
        residual_prediction=config.get('residual_prediction', False),
    ).to(device)

    state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    n_loaded = len(state_dict) - len(unexpected)
    print(f"[foresight] Loaded: {n_loaded} keys, "
          f"{len(missing)} missing (backbone/VAE), {len(unexpected)} unexpected")

    # Freeze backbone + TactileVAE, keep foresight trainable parts
    model.backbone.requires_grad_(False)
    model.tactile_vae.requires_grad_(False)

    model.train()
    return model, config


def load_foresight_norm_stats(foresight_dir, device):
    """Load foresight mean/std normalization stats."""
    stats_path = os.path.join(foresight_dir, "dataset_stats.pkl")
    if os.path.exists(stats_path):
        with open(stats_path, 'rb') as f:
            ns = pickle.load(f)
        print(f"[foresight] norm_stats from dataset_stats.pkl")
    else:
        args_path = os.path.join(foresight_dir, "args.json")
        with open(args_path) as f:
            ns = json.load(f)
        print(f"[foresight] norm_stats from args.json")

    return {
        'qpos_mean': torch.tensor(ns['qpos_mean'], dtype=torch.float32, device=device),
        'qpos_std': torch.tensor(ns['qpos_std'], dtype=torch.float32, device=device),
        'action_mean': torch.tensor(ns['action_mean'], dtype=torch.float32, device=device),
        'action_std': torch.tensor(ns['action_std'], dtype=torch.float32, device=device),
    }


# ==================== x0 prediction ====================
def predict_x0(noisy, noise_pred, timesteps, alphas_cumprod):
    """Predict clean action x0 from noisy action and noise prediction."""
    alpha_prod = alphas_cumprod[timesteps].view(-1, 1, 1)
    x0 = (noisy - torch.sqrt(1 - alpha_prod) * noise_pred) / torch.sqrt(alpha_prod)
    return x0.clamp(-1, 1)


# ==================== Training ====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--camera_names', type=str, default='global,wrist')
    parser.add_argument('--proprio_key', type=str, default='proprio_joint')
    parser.add_argument('--action_key', type=str, default='actions/joint_abs')
    parser.add_argument('--tac_side', type=str, default='left')
    parser.add_argument('--tac_history', type=int, default=8)
    parser.add_argument('--vae_checkpoint', type=str,
                        default='/home/chenshuai/Project/output/tactile_vae_full/best_tactile_vae.pt')
    parser.add_argument('--vae_latent_dim', type=int, default=16)
    parser.add_argument('--pred_horizon', type=int, default=16)
    parser.add_argument('--obs_horizon', type=int, default=2)
    parser.add_argument('--n_action_steps', type=int, default=8)
    parser.add_argument('--resize_shape', type=str, default='240,320')
    parser.add_argument('--crop_shape', type=str, default='216,288')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=128)
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
    parser.add_argument('--gpu', type=str, default='0')
    # Foresight joint training params
    parser.add_argument('--foresight_ckpt', type=str, required=True)
    parser.add_argument('--foresight_dir', type=str, default=None)
    parser.add_argument('--lambda_foresight', type=float, default=0.1)
    parser.add_argument('--foresight_horizon', type=int, default=10)
    parser.add_argument('--foresight_warmup_epochs', type=int, default=10)
    parser.add_argument('--foresight_t_threshold', type=int, default=50,
                        help='Only compute foresight loss when diffusion t < threshold')
    parser.add_argument('--foresight_lr', type=float, default=None,
                        help='Separate LR for foresight params (default: same as --lr)')
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

    all_entries = []
    for ds_dir in dataset_dirs:
        episode_files = sorted([f for f in os.listdir(ds_dir)
                                if f.startswith('episode_') and f.endswith('.hdf5')])
        for ef in episode_files:
            ep_id = int(ef.split('_')[1].split('.')[0])
            all_entries.append((ds_dir, ep_id))
    print(f"Total episodes: {len(all_entries)}")

    train_dataset = DPForesightJointDataset(
        all_entries, camera_names, norm_stats,
        args.pred_horizon, args.obs_horizon, args.tac_history,
        args.proprio_key, args.action_key, args.tac_side,
        foresight_horizon=args.foresight_horizon,
        resize_shape=resize_shape, crop_shape=crop_shape, is_train=True,
    )

    num_workers = 8 if use_multi_gpu else 4
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=True,
                              collate_fn=collate_fn)

    # DP Models
    action_dim = norm_stats['action_min'].shape[0]
    qpos_dim = norm_stats['qpos_min'].shape[0]

    vision_encoder = OfficialVisionEncoder(camera_names).to(device)
    vis_feat_dim = vision_encoder.feat_dim
    n_cams = len(camera_names)

    tac_encoder = FrozenTactileVAEEncoder(
        args.vae_checkpoint, latent_dim=args.vae_latent_dim,
        temporal_window=args.tac_history,
    ).to(device)
    tac_feat_dim = tac_encoder.feat_dim

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

    # Foresight Model
    foresight_dir = args.foresight_dir or os.path.dirname(args.foresight_ckpt)
    foresight, fs_config = load_foresight_model(args.foresight_ckpt, foresight_dir, device)
    fs_norm = load_foresight_norm_stats(foresight_dir, device)
    fs_chunk_size = fs_config.get('chunk_size', 10)

    # Collect foresight trainable params
    fs_trainable_params = []
    for name, param in foresight.named_parameters():
        if param.requires_grad:
            fs_trainable_params.append(param)
    n_fs_trainable = sum(p.numel() for p in fs_trainable_params)
    print(f"[foresight] trainable params: {n_fs_trainable:,}")

    # DataParallel
    if use_multi_gpu:
        vision_encoder = nn.DataParallel(vision_encoder, device_ids=gpu_ids)
        noise_pred_net = nn.DataParallel(noise_pred_net, device_ids=gpu_ids)
        tac_encoder = nn.DataParallel(tac_encoder, device_ids=gpu_ids)

    use_ema = not args.no_ema
    vis_module = vision_encoder.module if use_multi_gpu else vision_encoder
    net_module = noise_pred_net.module if use_multi_gpu else noise_pred_net
    ema_vis = EMAModel(vis_module) if use_ema else None
    ema_net = EMAModel(net_module) if use_ema else None

    # Optimizer: DP params + Foresight trainable params
    dp_params = list(net_module.parameters()) + list(vis_module.parameters())
    fs_lr = args.foresight_lr or args.lr
    optimizer = torch.optim.AdamW([
        {'params': dp_params, 'lr': args.lr},
        {'params': fs_trainable_params, 'lr': fs_lr},
    ], betas=(0.95, 0.999), weight_decay=args.weight_decay)

    total_steps = len(train_loader) * args.epochs
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))
    lr_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, [lr_lambda, lr_lambda])

    # Action normalization conversion tensors
    action_min_t = torch.tensor(norm_stats['action_min'], dtype=torch.float32, device=device)
    action_max_t = torch.tensor(norm_stats['action_max'], dtype=torch.float32, device=device)
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)

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
    config['n_train'] = len(all_entries)
    config['n_val'] = 0
    config['variant'] = 'tactile_vae_frozen'
    config['gpu_ids'] = gpu_ids
    config['fs_chunk_size'] = fs_chunk_size
    config['n_fs_trainable_params'] = n_fs_trainable
    ns = {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in norm_stats.items()}
    config['norm_stats'] = ns
    with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print(f"=== DP + Foresight Joint Training ===")
    print(f"GPUs: {gpu_ids} ({'DataParallel' if use_multi_gpu else 'single'})")
    print(f"action_dim={action_dim}, global_cond_dim={global_cond_dim}")
    print(f"vis_feat_dim={vis_feat_dim}/camera, cameras={camera_names}")
    print(f"tac_feat_dim={tac_feat_dim} (frozen VAE)")
    print(f"pred_horizon={args.pred_horizon}, obs_horizon={args.obs_horizon}")
    print(f"foresight: λ={args.lambda_foresight}, horizon={args.foresight_horizon}, "
          f"chunk={fs_chunk_size}, warmup={args.foresight_warmup_epochs}ep, "
          f"t_threshold={args.foresight_t_threshold}")
    print(f"Train: {len(all_entries)} eps ({len(train_dataset)} windows)")
    print(f"Batch: {args.batch_size}, Epochs: {args.epochs}")

    # TopK checkpoint
    topk_train_losses = {}
    topk_k = 5

    train_losses = []
    fs_losses_log = []
    global_step = 0

    for epoch in range(args.epochs):
        vision_encoder.train()
        noise_pred_net.train()
        foresight.train()
        ep_losses = []
        ep_fs_losses = []

        for batch in train_loader:
            B = batch['qpos'].shape[0]
            qpos = batch['qpos'].to(device)
            action = batch['action'].to(device)
            marker_hist = batch['marker_hist'].to(device)
            future_marker = batch['future_marker'].to(device)  # (B, 8, 9, 9, 2)
            qpos_raw = batch['qpos_raw'].to(device)  # (B, 7)

            # --- DP obs_cond ---
            obs_feats = []
            # Save last-frame image for foresight
            last_frame_imgs = {}
            for t in range(args.obs_horizon):
                imgs_t = {cam: batch['images'][cam][:, t].to(device) for cam in camera_names}
                vf = vision_encoder(imgs_t)
                marker_t = marker_hist[:, t]
                tf = tac_encoder(marker_t)
                obs_feats.append(torch.cat([vf, tf, qpos[:, t]], dim=-1))
                if t == args.obs_horizon - 1:
                    last_frame_imgs = imgs_t

            obs_cond = torch.cat(obs_feats, dim=-1)

            # --- Standard DP loss ---
            noise = torch.randn_like(action)
            ts = torch.randint(0, args.num_train_timesteps, (B,), device=device).long()
            noisy = noise_scheduler.add_noise(action, noise, ts)
            pred = noise_pred_net(noisy, ts, global_cond=obs_cond)
            loss_dp = F.mse_loss(pred, noise)

            # --- Foresight auxiliary loss ---
            loss_fs = torch.tensor(0.0, device=device)
            if epoch >= args.foresight_warmup_epochs:
                low_t_mask = ts < args.foresight_t_threshold
                n_low_t = low_t_mask.sum().item()

                if n_low_t > 0:
                    # Predict x0 for low-t samples
                    x0_pred = predict_x0(
                        noisy[low_t_mask], pred[low_t_mask],
                        ts[low_t_mask], alphas_cumprod
                    )  # (n_low_t, pred_horizon, 7) in [-1, 1]

                    # Convert DP min-max → raw → Foresight mean/std
                    x0_raw = (x0_pred + 1) / 2 * (action_max_t - action_min_t) + action_min_t
                    x0_fs = (x0_raw - fs_norm['action_mean']) / fs_norm['action_std']
                    x0_fs_chunk = x0_fs[:, :fs_chunk_size, :]  # (n_low_t, chunk_size, 7)

                    # Prepare foresight inputs: [global_img, wrist_img, marker_window]
                    fs_images = []
                    for cam in camera_names:
                        fs_images.append(last_frame_imgs[cam][low_t_mask])
                    # marker_window from last obs frame
                    fs_marker = marker_hist[low_t_mask, -1]  # (n_low_t, 8, 9, 9, 2)
                    fs_images.append(fs_marker)

                    # qpos for foresight (mean/std normalized)
                    qpos_fs = (qpos_raw[low_t_mask] - fs_norm['qpos_mean']) / fs_norm['qpos_std']

                    # Future marker as GT (pass through foresight to get z_gt)
                    fs_cam_names = fs_config.get('camera_names', ['global', 'wrist', 'gelsight'])
                    fs_future_images = [None] * len(fs_cam_names)
                    for idx, cam_name in enumerate(fs_cam_names):
                        if cam_name == 'gelsight':
                            # future_marker: (n_low_t, 8, 9, 9, 2) → need (n_low_t, 1, 8, 9, 9, 2)
                            # for predict_horizon=1
                            fm = future_marker[low_t_mask].unsqueeze(1)  # (n_low_t, 1, 8, 9, 9, 2)
                            fs_future_images[idx] = fm
                            break

                    # Foresight forward
                    z_pred, z_gt, _, _, _, _ = foresight(
                        fs_images, x0_fs_chunk, future_images=fs_future_images, qpos=qpos_fs
                    )

                    if z_gt is not None:
                        loss_fs = F.l1_loss(z_pred, z_gt)

            # Total loss
            loss = loss_dp + args.lambda_foresight * loss_fs

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(dp_params + fs_trainable_params, 1.0)
            optimizer.step()
            lr_sched.step()
            global_step += 1

            if ema_vis:
                ema_vis.update(vis_module)
                ema_net.update(net_module)

            ep_losses.append(loss_dp.item())
            ep_fs_losses.append(loss_fs.item())

        train_loss = np.mean(ep_losses)
        fs_loss = np.mean(ep_fs_losses)
        train_losses.append(train_loss)
        fs_losses_log.append(fs_loss)

        # TopK checkpoint saving
        def _save_ckpt(path):
            sd = {'epoch': epoch, 'train_loss': train_loss, 'fs_loss': fs_loss}
            if ema_vis:
                sd['ema_vis'] = ema_vis.state_dict()
                sd['ema_net'] = ema_net.state_dict()
            sd['vision_encoder'] = vis_module.state_dict()
            sd['noise_pred_net'] = net_module.state_dict()
            sd['foresight'] = foresight.state_dict()
            torch.save(sd, path)

        if len(topk_train_losses) < topk_k:
            ckpt_path = os.path.join(args.save_dir,
                                     f'dp_topk_ep{epoch+1}_loss{train_loss:.4f}.pth')
            _save_ckpt(ckpt_path)
            topk_train_losses[ckpt_path] = train_loss
        else:
            worst_path = max(topk_train_losses, key=topk_train_losses.get)
            if train_loss < topk_train_losses[worst_path]:
                if os.path.exists(worst_path):
                    os.remove(worst_path)
                del topk_train_losses[worst_path]
                ckpt_path = os.path.join(args.save_dir,
                                         f'dp_topk_ep{epoch+1}_loss{train_loss:.4f}.pth')
                _save_ckpt(ckpt_path)
                topk_train_losses[ckpt_path] = train_loss

        print(f"Ep {epoch+1}/{args.epochs} | dp={train_loss:.6f} | "
              f"fs={fs_loss:.4f} | best={min(topk_train_losses.values()):.6f} | "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

        if (epoch + 1) % args.save_freq == 0:
            sd = {'noise_pred_net': net_module.state_dict(),
                  'vision_encoder': vis_module.state_dict(),
                  'foresight': foresight.state_dict(),
                  'epoch': epoch}
            if ema_vis:
                sd['ema_vis'] = ema_vis.state_dict()
                sd['ema_net'] = ema_net.state_dict()
            torch.save(sd, os.path.join(args.save_dir, f'dp_epoch{epoch+1}.pth'))

    # Final save
    sd = {'noise_pred_net': net_module.state_dict(),
          'vision_encoder': vis_module.state_dict(),
          'foresight': foresight.state_dict(),
          'epoch': args.epochs - 1}
    if ema_vis:
        sd['ema_vis'] = ema_vis.state_dict()
        sd['ema_net'] = ema_net.state_dict()
    torch.save(sd, os.path.join(args.save_dir, 'dp_final.pth'))

    np.save(os.path.join(args.save_dir, 'train_losses.npy'), train_losses)
    np.save(os.path.join(args.save_dir, 'fs_losses.npy'), fs_losses_log)


if __name__ == '__main__':
    main()
