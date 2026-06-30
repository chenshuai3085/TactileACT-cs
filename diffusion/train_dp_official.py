"""
Train Diffusion Policy — Official-aligned baseline (no tactile).

Aligned with official DP real-robot hybrid config (Chi et al. RSS 2023):
  - ResNet18 + AdaptiveAvgPool → 512 dim per camera (GroupNorm, no BN)
  - Independent encoder per camera (deepcopy, no weight sharing)
  - Resize(240,320) + RandomCrop(216,288) / CenterCrop(216,288)
  - EMA for inference (power=0.75)
  - DDPM 100 steps, squaredcos_cap_v2, epsilon prediction
  - pred_horizon=16, n_action_steps=8
  - Exhaustive sliding-window sampling (every valid window visited once per epoch)
  - Optional temporal_stride for faster policies: obs/action labels are sampled every N frames
  - LR warmup 500 steps + cosine decay
  - AdamW betas=(0.95, 0.999), weight_decay=1e-6
  - obs_horizon=2
  - min-max normalization to [-1,1]

Cameras: global + wrist only (no tactile)
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


# ==================== Vision Encoder (Official-aligned) ====================
class OfficialVisionEncoder(nn.Module):
    """
    Official DP vision encoder: per-camera ResNet18 + AdaptiveAvgPool → 512 dim.
    GroupNorm replaces BatchNorm for EMA compatibility.
    """
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
            features.append(self.encoders[cam](images_dict[cam]))
        return torch.cat(features, dim=-1)


# ==================== Dataset ====================
class DPOfficialDataset(torch.utils.data.Dataset):
    """Preload all data into RAM, exhaustive sliding-window sampling.

    Every valid (episode, start_ts) window is visited exactly once per epoch.
    DataLoader shuffle=True handles randomization.
    """

    def __init__(self, episode_entries, camera_names, norm_stats,
                 pred_horizon, obs_horizon=2,
                 proprio_key="proprio_joint", action_key="actions/joint_abs",
                 resize_shape=(240, 320), crop_shape=(216, 288), is_train=True,
                 temporal_stride=1):
        self.camera_names = camera_names
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.temporal_stride = int(temporal_stride)
        if self.temporal_stride < 1:
            raise ValueError(f"temporal_stride must be >= 1, got {temporal_stride}")

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
            required_future_span = (pred_horizon - 1) * self.temporal_stride + 1
            for start_ts in range(max(1, ep_len - required_future_span + 1)):
                self.indices.append((ep_idx, start_ts))

        print(f"  DPOfficialDataset: {len(self.episodes)} episodes, "
              f"{total_frames} total frames, {len(self.indices)} windows "
              f"({'train' if is_train else 'val'}), temporal_stride={self.temporal_stride}")

    def _preload_episode(self, path, camera_names, proprio_key, action_key):
        with h5py.File(path, 'r') as f:
            qpos = f[f'observations/{proprio_key}'][()].astype(np.float32)
            action = f[f'{action_key}'][()].astype(np.float32)
            images = {}
            for cam in camera_names:
                images[cam] = f[f'observations/images/{cam}'][()]
        return {'qpos': qpos, 'action': action, 'images': images}

    def __len__(self):
        return len(self.indices)

    def _minmax_norm(self, x, xmin, xmax):
        return (x - xmin) / (xmax - xmin + 1e-8) * 2 - 1

    def _process_image(self, img_uint8):
        img_t = torch.from_numpy(img_uint8).float().div_(255.0).permute(2, 0, 1)
        img_t = self.resize_transform(img_t)
        img_t = self.crop_transform(img_t)
        return self.image_normalize(img_t)

    def __getitem__(self, index):
        ep_idx, start_ts = self.indices[index]
        ep = self.episodes[ep_idx]
        ep_len = ep['qpos'].shape[0]

        obs_indices = [max(0, start_ts - (self.obs_horizon - 1 - k) * self.temporal_stride)
                       for k in range(self.obs_horizon)]

        all_images = {cam: [] for cam in self.camera_names}
        all_qpos = []
        for t in obs_indices:
            all_qpos.append(ep['qpos'][t])
            for cam in self.camera_names:
                all_images[cam].append(self._process_image(ep['images'][cam][t]))

        action_indices = [
            min(start_ts + k * self.temporal_stride, ep_len - 1)
            for k in range(self.pred_horizon)
        ]
        action = ep['action'][action_indices]
        if action.shape[0] < self.pred_horizon:
            pad = np.tile(action[-1:], (self.pred_horizon - action.shape[0], 1))
            action = np.concatenate([action, pad], axis=0)

        images_per_cam = {cam: torch.stack(imgs) for cam, imgs in all_images.items()}
        qpos = np.stack(all_qpos)

        return {
            'images': images_per_cam,
            'qpos': torch.from_numpy(self._minmax_norm(qpos, self.qpos_min, self.qpos_max)),
            'action': torch.from_numpy(self._minmax_norm(action, self.action_min, self.action_max)),
        }


def dp_official_collate(batch):
    """Custom collate: stack dict-of-tensors for images."""
    camera_names = list(batch[0]['images'].keys())
    images = {}
    for cam in camera_names:
        images[cam] = torch.stack([b['images'][cam] for b in batch])
    return {
        'images': images,
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


# ==================== Training ====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Dataset directory (comma-separated for multiple)')
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--camera_names', type=str, default='global,wrist')
    parser.add_argument('--proprio_key', type=str, default='proprio_joint')
    parser.add_argument('--action_key', type=str, default='actions/joint_abs')
    parser.add_argument('--pred_horizon', type=int, default=16)
    parser.add_argument('--obs_horizon', type=int, default=2)
    parser.add_argument('--n_action_steps', type=int, default=8,
                        help='Steps to execute at inference (saved to config.json)')
    parser.add_argument('--temporal_stride', type=int, default=1,
                        help='Frame stride used for observation history and future action labels. '
                             '1 preserves dense official DP; 3 trains obs at t-3,t and actions t,t+3,...')
    parser.add_argument('--resize_shape', type=str, default='240,320',
                        help='Resize images to (H,W) before crop')
    parser.add_argument('--crop_shape', type=str, default='216,288',
                        help='Random/center crop to (H,W)')
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
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--val_ratio', type=float, default=0.0,
                        help='Episode-level validation ratio. 0 preserves train-only behavior.')
    parser.add_argument('--val_interval', type=int, default=5,
                        help='Run validation every N epochs when val_ratio > 0.')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')
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

    val_ratio = float(args.val_ratio)
    if val_ratio < 0 or val_ratio >= 1:
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")
    if val_ratio > 0 and len(all_entries) > 1:
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(len(all_entries)).tolist()
        n_val = max(1, int(round(len(all_entries) * val_ratio)))
        n_val = min(n_val, len(all_entries) - 1)
        val_ids = set(perm[:n_val])
        train_entries = [entry for i, entry in enumerate(all_entries) if i not in val_ids]
        val_entries = [entry for i, entry in enumerate(all_entries) if i in val_ids]
    else:
        train_entries = all_entries
        val_entries = []
    print(f"Train episodes: {len(train_entries)}, Val episodes: {len(val_entries)}")

    train_dataset = DPOfficialDataset(train_entries, camera_names,
                                       norm_stats, args.pred_horizon, args.obs_horizon,
                                       args.proprio_key, args.action_key,
                                       resize_shape=resize_shape,
                                       crop_shape=crop_shape, is_train=True,
                                       temporal_stride=args.temporal_stride)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=True,
                              collate_fn=dp_official_collate)
    if val_entries:
        val_dataset = DPOfficialDataset(val_entries, camera_names,
                                        norm_stats, args.pred_horizon, args.obs_horizon,
                                        args.proprio_key, args.action_key,
                                        resize_shape=resize_shape,
                                        crop_shape=crop_shape, is_train=False,
                                        temporal_stride=args.temporal_stride)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=0, pin_memory=True,
                                collate_fn=dp_official_collate)
    else:
        val_dataset = None
        val_loader = None

    # model
    action_dim = norm_stats['action_min'].shape[0]
    qpos_dim = norm_stats['qpos_min'].shape[0]

    vision_encoder = OfficialVisionEncoder(camera_names).to(device)
    vis_feat_dim = vision_encoder.feat_dim  # 512 per camera
    n_cams = len(camera_names)
    global_cond_dim = (vis_feat_dim * n_cams + qpos_dim) * args.obs_horizon

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
    config['dataset_dirs'] = dataset_dirs
    config['resize_shape'] = list(resize_shape)
    config['crop_shape'] = list(crop_shape)
    config['action_dim'] = int(action_dim)
    config['global_cond_dim'] = int(global_cond_dim)
    config['vis_feat_dim'] = int(vis_feat_dim)
    config['down_dims'] = down_dims
    config['use_ema'] = use_ema
    config['n_train'] = len(train_entries)
    config['n_val'] = len(val_entries)
    config['variant'] = 'official_no_tactile'
    ns = {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in norm_stats.items()}
    config['norm_stats'] = ns
    with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print(f"=== DP Official Baseline (no tactile) ===")
    print(f"action_dim={action_dim}, global_cond_dim={global_cond_dim}")
    print(f"vis_feat_dim={vis_feat_dim}/camera, cameras={camera_names}")
    print(f"pred_horizon={args.pred_horizon}, obs_horizon={args.obs_horizon}, "
          f"n_action_steps={args.n_action_steps}, temporal_stride={args.temporal_stride}")
    print(f"resize={resize_shape}, crop={crop_shape}")
    print(f"EMA={use_ema}, inference_steps={args.num_inference_steps}")
    print(f"Train: {len(train_entries)} eps ({len(train_dataset)} windows)")
    if val_dataset is not None:
        print(f"Val: {len(val_entries)} eps ({len(val_dataset)} windows), "
              f"val_interval={args.val_interval}")
    print(f"Batch: {args.batch_size}, Epochs: {args.epochs}")

    # TopK checkpoint manager (keep top-5 by train_loss, like official)
    topk_train_losses = {}  # {path: train_loss}
    topk_k = 5

    train_losses = []
    val_losses = []
    best_metric = float('inf')
    best_metric_name = 'val_loss' if val_loader is not None else 'train_loss'
    global_step = 0

    @torch.no_grad()
    def evaluate(loader):
        vision_encoder.eval()
        noise_pred_net.eval()
        losses = []
        for batch in loader:
            B = batch['qpos'].shape[0]
            qpos = batch['qpos'].to(device)
            action = batch['action'].to(device)

            obs_feats = []
            for t in range(args.obs_horizon):
                imgs_t = {cam: batch['images'][cam][:, t].to(device) for cam in camera_names}
                vf = vision_encoder(imgs_t)
                obs_feats.append(torch.cat([vf, qpos[:, t]], dim=-1))
            obs_cond = torch.cat(obs_feats, dim=-1)

            noise = torch.randn_like(action)
            ts = torch.randint(0, args.num_train_timesteps, (B,), device=device).long()
            noisy = noise_scheduler.add_noise(action, noise, ts)

            pred = noise_pred_net(noisy, ts, global_cond=obs_cond)
            loss = nn.functional.mse_loss(pred, noise)
            losses.append(loss.item())
        return float(np.mean(losses))

    for epoch in range(args.epochs):
        vision_encoder.train()
        noise_pred_net.train()
        ep_losses = []

        for batch in train_loader:
            B = batch['qpos'].shape[0]
            qpos = batch['qpos'].to(device)
            action = batch['action'].to(device)

            obs_feats = []
            for t in range(args.obs_horizon):
                imgs_t = {cam: batch['images'][cam][:, t].to(device) for cam in camera_names}
                vf = vision_encoder(imgs_t)
                obs_feats.append(torch.cat([vf, qpos[:, t]], dim=-1))
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
        val_loss = None
        should_val = (
            val_loader is not None
            and (epoch == 0 or (epoch + 1) % max(1, args.val_interval) == 0 or epoch == args.epochs - 1)
        )
        if should_val:
            val_loss = evaluate(val_loader)
            val_losses.append({'epoch': epoch, 'val_loss': val_loss})

        # TopK checkpoint saving (monitor train_loss, keep top-5)
        def _save_ckpt(path, val_loss_for_ckpt=None):
            sd = {'epoch': epoch, 'train_loss': train_loss}
            if val_loss_for_ckpt is not None:
                sd['val_loss'] = val_loss_for_ckpt
            if ema_vis:
                sd['ema_vis'] = ema_vis.state_dict()
                sd['ema_net'] = ema_net.state_dict()
            sd['vision_encoder'] = vision_encoder.state_dict()
            sd['noise_pred_net'] = noise_pred_net.state_dict()
            torch.save(sd, path)

        if len(topk_train_losses) < topk_k:
            ckpt_path = os.path.join(args.save_dir, f'dp_topk_ep{epoch+1}_loss{train_loss:.4f}.pth')
            _save_ckpt(ckpt_path)
            topk_train_losses[ckpt_path] = train_loss
        else:
            worst_path = max(topk_train_losses, key=topk_train_losses.get)
            if train_loss < topk_train_losses[worst_path]:
                if os.path.exists(worst_path):
                    os.remove(worst_path)
                del topk_train_losses[worst_path]
                ckpt_path = os.path.join(args.save_dir, f'dp_topk_ep{epoch+1}_loss{train_loss:.4f}.pth')
                _save_ckpt(ckpt_path)
                topk_train_losses[ckpt_path] = train_loss

        metric = val_loss if val_loss is not None else (train_loss if val_loader is None else None)
        if metric is not None and metric < best_metric:
            best_metric = metric
            _save_ckpt(os.path.join(args.save_dir, 'dp_best.pth'), val_loss)

        msg = (f"Ep {epoch+1}/{args.epochs} | train={train_loss:.6f}")
        if val_loss is not None:
            msg += f" | val={val_loss:.6f}"
        msg += (f" | best_{best_metric_name}={best_metric:.6f} | "
                f"topk_train={min(topk_train_losses.values()):.6f} | "
                f"lr={optimizer.param_groups[0]['lr']:.2e}")
        print(msg)

        if (epoch + 1) % args.save_freq == 0:
            sd = {'noise_pred_net': noise_pred_net.state_dict(),
                  'vision_encoder': vision_encoder.state_dict(), 'epoch': epoch,
                  'train_loss': train_loss}
            if val_loss is not None:
                sd['val_loss'] = val_loss
            if ema_vis:
                sd['ema_vis'] = ema_vis.state_dict()
                sd['ema_net'] = ema_net.state_dict()
            torch.save(sd, os.path.join(args.save_dir, f'dp_epoch{epoch+1}.pth'))

    sd = {'noise_pred_net': noise_pred_net.state_dict(),
          'vision_encoder': vision_encoder.state_dict(), 'epoch': args.epochs - 1,
          'train_loss': train_losses[-1]}
    if ema_vis:
        sd['ema_vis'] = ema_vis.state_dict()
        sd['ema_net'] = ema_net.state_dict()
    torch.save(sd, os.path.join(args.save_dir, 'dp_final.pth'))

    np.save(os.path.join(args.save_dir, 'train_losses.npy'), train_losses)
    if val_losses:
        with open(os.path.join(args.save_dir, 'val_losses.json'), 'w') as f:
            json.dump(val_losses, f, indent=2)
    print(f"\nDone! Best {best_metric_name}: {best_metric:.6f}")
    print(f"Top-{topk_k} checkpoints saved in {args.save_dir}")


if __name__ == '__main__':
    main()
