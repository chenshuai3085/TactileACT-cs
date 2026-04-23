"""
Train Diffusion Policy + Frozen TactileVAE (Plan C).

Vision: Official-aligned (same as train_dp_official.py)
  - ResNet18 random init + GroupNorm + SpatialSoftmax → 1024 dim/camera
  - Independent encoder per camera (deepcopy, no sharing)
  - End-to-end trained from scratch

Tactile: Frozen TactileVAE
  - Input: 8-frame marker_offset history (B, 8, 9, 9, 2)
  - Normalized: (x - mean) / std, mean=[0.21, -0.64], std=[1.68, 3.67]
  - TactileVAE.encode_single_frame → (B, 16, 3, 3) → flatten → 144 dim
  - Completely frozen (requires_grad=False)

Training aligned with official DP:
  - EMA, DDPM 100 steps, squaredcos_cap_v2
  - diffusion_step_embed_dim=128
  - LR warmup + cosine decay
  - AdamW betas=(0.95, 0.999)
  - obs_horizon=2, min-max to [-1,1]

Cameras: global + wrist (no gelsight image)
"""
import os, sys, json, argparse, copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from tqdm import tqdm
import h5py
import torchvision
from torchvision import transforms

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'TFAC_V5'))
from utils import set_seed
from network import ConditionalUnet1D, replace_bn_with_gn
from tactile_vae import TactileVAE, build_tactile_vae


# ==================== SpatialSoftmax ====================
class SpatialSoftmax(nn.Module):
    """
    Spatial Softmax pooling (Levine et al. 2016, used in official DP).
    Input: (B, C, H, W) feature map
    Output: (B, C*2) — per-channel expected (x, y) coordinates
    """
    def __init__(self, h, w, num_channels):
        super().__init__()
        pos_x = torch.linspace(0.0, 1.0, w)
        pos_y = torch.linspace(0.0, 1.0, h)
        self.register_buffer('pos_x', pos_x.reshape(1, 1, 1, w))
        self.register_buffer('pos_y', pos_y.reshape(1, 1, h, 1))
        self.num_channels = num_channels

    def forward(self, x):
        B, C, H, W = x.shape
        attention = x.reshape(B, C, H * W)
        attention = torch.softmax(attention, dim=-1)
        attention = attention.reshape(B, C, H, W)
        expected_x = (attention * self.pos_x).sum(dim=(2, 3))
        expected_y = (attention * self.pos_y).sum(dim=(2, 3))
        return torch.cat([expected_x, expected_y], dim=-1)


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


# ==================== Vision Encoder (Official-aligned) ====================
def _make_resnet18_backbone():
    """Create ResNet18 backbone: random init, GroupNorm, remove avgpool+fc."""
    resnet = torchvision.models.resnet18()
    backbone = nn.Sequential(*list(resnet.children())[:-2])
    replace_bn_with_gn(backbone, features_per_group=16)
    return backbone


class OfficialVisionEncoder(nn.Module):
    """
    Official DP vision encoder — same as train_dp_official.py.
    Per-camera independent ResNet18 + SpatialSoftmax → 1024 dim/camera.
    """
    def __init__(self, camera_names, input_h=480, input_w=640):
        super().__init__()
        self.camera_names = camera_names

        base_backbone = _make_resnet18_backbone()
        self.encoders = nn.ModuleDict()
        for cam in camera_names:
            self.encoders[cam] = copy.deepcopy(base_backbone)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_h, input_w)
            feat = base_backbone(dummy)
            _, C, H, W = feat.shape

        self.spatial_softmax = SpatialSoftmax(H, W, C)
        self.feat_dim = C * 2  # 1024

    def forward(self, images_dict):
        features = []
        for cam in self.camera_names:
            feat_map = self.encoders[cam](images_dict[cam])
            feat = self.spatial_softmax(feat_map)
            features.append(feat)
        return torch.cat(features, dim=-1)


# ==================== Frozen TactileVAE Encoder ====================
class FrozenTactileVAEEncoder(nn.Module):
    """
    Frozen TactileVAE: encode 8-frame marker_offset → flat latent vector.
    Input: (B, 8, 9, 9, 2) normalized marker_offset
    Output: (B, latent_dim * 3 * 3) = (B, 72) for latent_dim=8
    """
    TAC_MEAN = np.array([0.2102, -0.6422], dtype=np.float32)
    TAC_STD = np.array([1.6805, 3.6717], dtype=np.float32)

    def __init__(self, vae_checkpoint_path, latent_dim=8, temporal_window=8):
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

        self.feat_dim = latent_dim * 3 * 3  # 8 * 3 * 3 = 72
        self.temporal_window = temporal_window

        self.register_buffer('tac_mean', torch.tensor(self.TAC_MEAN))
        self.register_buffer('tac_std', torch.tensor(self.TAC_STD))

    def normalize_marker(self, marker_offset):
        """Normalize marker_offset: (x - mean) / std."""
        return (marker_offset - self.tac_mean) / self.tac_std

    @torch.no_grad()
    def forward(self, marker_seq):
        """
        Args:
            marker_seq: (B, T, 9, 9, 2) raw marker_offset (NOT normalized)
        Returns:
            (B, feat_dim) flattened latent
        """
        marker_norm = self.normalize_marker(marker_seq)
        z_last, _ = self.vae.encode_single_frame(marker_norm)  # (B, C, 3, 3)
        return z_last.flatten(1)  # (B, C*3*3)


# ==================== Dataset ====================
class DPTacVAEDataset(torch.utils.data.Dataset):
    """
    Sliding-window exhaustive sampling.
    Loads global+wrist images + 8-frame marker_offset history.
    """

    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats,
                 pred_horizon, obs_horizon=2, tac_history=8,
                 proprio_key="proprio_joint", action_key="actions/joint_abs",
                 tac_side="left"):
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.tac_history = tac_history
        self.proprio_key = proprio_key
        self.action_key = action_key
        self.tac_side = tac_side

        self.action_min = np.array(norm_stats["action_min"])
        self.action_max = np.array(norm_stats["action_max"])
        self.qpos_min = np.array(norm_stats["qpos_min"])
        self.qpos_max = np.array(norm_stats["qpos_max"])

        self.image_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.indices = []
        for ep_id in episode_ids:
            path = os.path.join(dataset_dir, f'episode_{ep_id}.hdf5')
            with h5py.File(path, 'r') as f:
                ep_len = f[f'observations/{proprio_key}'].shape[0]
            max_start = max(1, ep_len - pred_horizon)
            for t in range(max_start):
                self.indices.append((ep_id, t))
        print(f"  DPTacVAEDataset: {len(episode_ids)} episodes, "
              f"{len(self.indices)} samples (sliding window)")

    def __len__(self):
        return len(self.indices)

    def _minmax_norm(self, x, xmin, xmax):
        return (x - xmin) / (xmax - xmin + 1e-8) * 2 - 1

    def _load_image(self, f, cam, t):
        img = f[f'observations/images/{cam}'][t]
        img_t = torch.tensor(img, dtype=torch.float32) / 255.0
        img_t = img_t.permute(2, 0, 1)
        return self.image_normalize(img_t)

    def _load_marker_history(self, f, t, ep_len):
        """Load tac_history frames of marker_offset ending at time t."""
        marker_key = f'observations/tac/{self.tac_side}/marker_offset'
        frames = []
        for k in range(self.tac_history):
            idx = max(0, t - self.tac_history + 1 + k)
            frames.append(f[marker_key][idx])
        return np.stack(frames, axis=0)  # (tac_history, 9, 9, 2)

    def __getitem__(self, index):
        ep_id, start_ts = self.indices[index]
        path = os.path.join(self.dataset_dir, f'episode_{ep_id}.hdf5')

        with h5py.File(path, 'r') as f:
            ep_len = f[f'observations/{self.proprio_key}'].shape[0]

            obs_indices = []
            for k in range(self.obs_horizon):
                t = max(0, start_ts - self.obs_horizon + 1 + k)
                obs_indices.append(t)

            all_qpos = []
            all_images = {cam: [] for cam in self.camera_names}
            all_marker_hist = []

            for t in obs_indices:
                all_qpos.append(f[f'observations/{self.proprio_key}'][t])
                for cam in self.camera_names:
                    all_images[cam].append(self._load_image(f, cam, t))
                all_marker_hist.append(self._load_marker_history(f, t, ep_len))

            action_end = min(start_ts + self.pred_horizon, ep_len)
            action = f[f'/{self.action_key}'][start_ts:action_end]

        action_len = action.shape[0]
        if action_len < self.pred_horizon:
            pad = np.tile(action[-1:], (self.pred_horizon - action_len, 1))
            action = np.concatenate([action, pad], axis=0)

        images_per_cam = {cam: torch.stack(imgs) for cam, imgs in all_images.items()}

        # marker_hist: (obs_horizon, tac_history, 9, 9, 2)
        marker_hist = np.stack(all_marker_hist, axis=0)

        qpos = np.stack(all_qpos)
        qpos_norm = self._minmax_norm(qpos, self.qpos_min, self.qpos_max)
        action_norm = self._minmax_norm(action, self.action_min, self.action_max)

        return {
            'images': images_per_cam,
            'marker_hist': torch.tensor(marker_hist, dtype=torch.float32),
            'qpos': torch.tensor(qpos_norm, dtype=torch.float32),
            'action': torch.tensor(action_norm, dtype=torch.float32),
        }


def dp_tac_vae_collate(batch):
    """Custom collate: stack dict-of-tensors for images."""
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


def get_minmax_stats(dataset_dir, proprio_key, action_key):
    episode_files = sorted([f for f in os.listdir(dataset_dir)
                            if f.startswith('episode_') and f.endswith('.hdf5')])
    all_qpos, all_action = [], []
    for ef in tqdm(episode_files, desc="Min/max stats"):
        with h5py.File(os.path.join(dataset_dir, ef), 'r') as root:
            all_qpos.append(root[f'/observations/{proprio_key}'][()])
            all_action.append(root[f'/{action_key}'][()])
    all_qpos = np.concatenate(all_qpos, axis=0)
    all_action = np.concatenate(all_action, axis=0)
    return {
        "qpos_min": all_qpos.min(axis=0),
        "qpos_max": all_qpos.max(axis=0),
        "action_min": all_action.min(axis=0),
        "action_max": all_action.max(axis=0),
    }


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
    parser.add_argument('--pred_horizon', type=int, default=20)
    parser.add_argument('--obs_horizon', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--num_train_timesteps', type=int, default=100)
    parser.add_argument('--num_inference_steps', type=int, default=100)
    parser.add_argument('--diffusion_step_embed_dim', type=int, default=128)
    parser.add_argument('--down_dims', type=str, default='256,512,1024')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--no_ema', action='store_true', default=False)
    parser.add_argument('--save_freq', type=int, default=500)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')
    camera_names = args.camera_names.split(',')
    down_dims = [int(x) for x in args.down_dims.split(',')]

    norm_stats = get_minmax_stats(args.dataset_dir, args.proprio_key, args.action_key)

    episode_files = sorted([f for f in os.listdir(args.dataset_dir)
                            if f.startswith('episode_') and f.endswith('.hdf5')])
    episode_ids = [int(f.split('_')[1].split('.')[0]) for f in episode_files]

    np.random.seed(args.seed)
    shuffled = np.random.permutation(episode_ids)
    split = int(0.8 * len(episode_ids))
    train_ids = shuffled[:split].tolist()
    val_ids = shuffled[split:].tolist()

    train_dataset = DPTacVAEDataset(train_ids, args.dataset_dir, camera_names,
                                     norm_stats, args.pred_horizon, args.obs_horizon,
                                     args.tac_history, args.proprio_key, args.action_key,
                                     args.tac_side)
    val_dataset = DPTacVAEDataset(val_ids, args.dataset_dir, camera_names,
                                   norm_stats, args.pred_horizon, args.obs_horizon,
                                   args.tac_history, args.proprio_key, args.action_key,
                                   args.tac_side)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True,
                              persistent_workers=True, collate_fn=dp_tac_vae_collate)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=2, pin_memory=True,
                            persistent_workers=True, collate_fn=dp_tac_vae_collate)

    # models
    action_dim = norm_stats['action_min'].shape[0]
    qpos_dim = norm_stats['qpos_min'].shape[0]

    vision_encoder = OfficialVisionEncoder(camera_names).to(device)
    vis_feat_dim = vision_encoder.feat_dim  # 1024 per camera
    n_cams = len(camera_names)

    tac_encoder = FrozenTactileVAEEncoder(
        args.vae_checkpoint, latent_dim=args.vae_latent_dim,
        temporal_window=args.tac_history
    ).to(device)
    tac_feat_dim = tac_encoder.feat_dim  # 72

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

    use_ema = not args.no_ema
    ema_vis = EMAModel(vision_encoder) if use_ema else None
    ema_net = EMAModel(noise_pred_net) if use_ema else None

    # TactileVAE is frozen — only optimize vision_encoder + noise_pred_net
    all_params = list(noise_pred_net.parameters()) + list(vision_encoder.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=args.lr,
                                  betas=(0.95, 0.999), weight_decay=args.weight_decay)

    total_steps = len(train_loader) * args.epochs
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))
    lr_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # save config
    config = vars(args)
    config['action_dim'] = int(action_dim)
    config['global_cond_dim'] = int(global_cond_dim)
    config['vis_feat_dim'] = int(vis_feat_dim)
    config['tac_feat_dim'] = int(tac_feat_dim)
    config['down_dims'] = down_dims
    config['use_ema'] = use_ema
    config['n_train'] = len(train_ids)
    config['n_val'] = len(val_ids)
    config['variant'] = 'tactile_vae_frozen'
    ns = {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in norm_stats.items()}
    config['norm_stats'] = ns
    with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # verify TactileVAE is frozen
    frozen_params = sum(1 for p in tac_encoder.parameters() if not p.requires_grad)
    total_tac_params = sum(1 for p in tac_encoder.parameters())
    assert frozen_params == total_tac_params, "TactileVAE must be fully frozen!"

    print(f"=== DP + Frozen TactileVAE ===")
    print(f"action_dim={action_dim}, global_cond_dim={global_cond_dim}")
    print(f"vis_feat_dim={vis_feat_dim}/camera, tac_feat_dim={tac_feat_dim}")
    print(f"cameras={camera_names}, tac_history={args.tac_history}")
    print(f"pred_horizon={args.pred_horizon}, obs_horizon={args.obs_horizon}")
    print(f"EMA={use_ema}, inference_steps={args.num_inference_steps}")
    print(f"TactileVAE frozen: {frozen_params}/{total_tac_params} params")
    print(f"Train: {len(train_ids)} eps, Val: {len(val_ids)} eps")
    print(f"Batch: {args.batch_size}, Epochs: {args.epochs}")

    best_val = float('inf')
    train_losses, val_losses = [], []
    global_step = 0

    for epoch in range(args.epochs):
        vision_encoder.train()
        noise_pred_net.train()
        tac_encoder.eval()
        ep_losses = []

        for batch in train_loader:
            B = batch['qpos'].shape[0]
            qpos = batch['qpos'].to(device)
            action = batch['action'].to(device)
            marker_hist = batch['marker_hist'].to(device)  # (B, obs_horizon, tac_history, 9, 9, 2)

            obs_feats = []
            for t in range(args.obs_horizon):
                imgs_t = {cam: batch['images'][cam][:, t].to(device) for cam in camera_names}
                vf = vision_encoder(imgs_t)
                tf = tac_encoder(marker_hist[:, t])  # (B, tac_feat_dim)
                obs_feats.append(torch.cat([vf, tf, qpos[:, t]], dim=-1))
            obs_cond = torch.cat(obs_feats, dim=-1)

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
                ema_vis.update(vision_encoder)
                ema_net.update(noise_pred_net)

            ep_losses.append(loss.item())

        train_loss = np.mean(ep_losses)
        train_losses.append(train_loss)

        # validation
        if (epoch + 1) % 10 == 0 or epoch == 0:
            if ema_vis:
                orig_vis = copy.deepcopy(vision_encoder.state_dict())
                orig_net = copy.deepcopy(noise_pred_net.state_dict())
                ema_vis.apply_to(vision_encoder)
                ema_net.apply_to(noise_pred_net)

            vision_encoder.eval()
            noise_pred_net.eval()
            v_losses = []
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
                        tf = tac_encoder(marker_hist[:, t])
                        obs_feats.append(torch.cat([vf, tf, qpos[:, t]], dim=-1))
                    obs_cond = torch.cat(obs_feats, dim=-1)

                    noise = torch.randn_like(action)
                    ts = torch.randint(0, args.num_train_timesteps, (B,), device=device).long()
                    noisy = noise_scheduler.add_noise(action, noise, ts)
                    pred = noise_pred_net(noisy, ts, global_cond=obs_cond)
                    v_losses.append(nn.functional.mse_loss(pred, noise).item())

            val_loss = np.mean(v_losses)
            val_losses.append(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                sd = {'noise_pred_net': noise_pred_net.state_dict(),
                      'vision_encoder': vision_encoder.state_dict(),
                      'epoch': epoch, 'val_loss': val_loss}
                if ema_vis:
                    sd['ema_vis'] = ema_vis.state_dict()
                    sd['ema_net'] = ema_net.state_dict()
                torch.save(sd, os.path.join(args.save_dir, 'dp_best.pth'))

            if ema_vis:
                vision_encoder.load_state_dict(orig_vis)
                noise_pred_net.load_state_dict(orig_net)

            print(f"Ep {epoch+1}/{args.epochs} | train={train_loss:.6f} | "
                  f"val={val_loss:.6f} | best={best_val:.6f} | "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}")

        if (epoch + 1) % args.save_freq == 0:
            sd = {'noise_pred_net': noise_pred_net.state_dict(),
                  'vision_encoder': vision_encoder.state_dict(), 'epoch': epoch}
            if ema_vis:
                sd['ema_vis'] = ema_vis.state_dict()
                sd['ema_net'] = ema_net.state_dict()
            torch.save(sd, os.path.join(args.save_dir, f'dp_epoch{epoch+1}.pth'))

    sd = {'noise_pred_net': noise_pred_net.state_dict(),
          'vision_encoder': vision_encoder.state_dict(), 'epoch': args.epochs - 1}
    if ema_vis:
        sd['ema_vis'] = ema_vis.state_dict()
        sd['ema_net'] = ema_net.state_dict()
    torch.save(sd, os.path.join(args.save_dir, 'dp_final.pth'))

    np.save(os.path.join(args.save_dir, 'train_losses.npy'), train_losses)
    np.save(os.path.join(args.save_dir, 'val_losses.npy'), val_losses)
    print(f"\nDone! Best val: {best_val:.6f}")


if __name__ == '__main__':
    main()
