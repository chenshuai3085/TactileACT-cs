"""
Train Diffusion Policy with joint_abs action space (7D).
Aligned with official DP (Chi et al. RSS 2023).

Key alignment:
  - EMA for inference
  - DDPM 100 steps (full, not DDIM)
  - diffusion_step_embed_dim=128
  - LR warmup + cosine decay
  - AdamW betas=(0.95, 0.999)
  - obs_horizon=2 (2-frame obs conditioning)
  - min-max normalization to [-1,1]

Data: same as ACT (per-episode random sampling, not exhaustive expansion)
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
sys.path.insert(0, os.path.join(ROOT, 'diffusion'))
from network import ConditionalUnet1D

try:
    from clip_pretraining_xiaomi import modified_resnet18
except ImportError:
    from clip_pretraining import modified_resnet18


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
class VisionEncoder(nn.Module):
    def __init__(self, camera_names, clip_vision_path=None, clip_tac_path=None):
        super().__init__()
        self.camera_names = camera_names

        vis_bb = modified_resnet18()
        if clip_vision_path and os.path.exists(clip_vision_path):
            vis_bb.load_state_dict(torch.load(clip_vision_path, map_location='cpu'), strict=False)
            print(f"Loaded CLIP vision: {clip_vision_path}")

        tac_bb = modified_resnet18()
        if clip_tac_path and os.path.exists(clip_tac_path):
            tac_bb.load_state_dict(torch.load(clip_tac_path, map_location='cpu'), strict=False)
            print(f"Loaded CLIP tactile: {clip_tac_path}")

        self.shared_vision = nn.Sequential(vis_bb, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.gelsight_encoder = nn.Sequential(tac_bb, nn.AdaptiveAvgPool2d(1), nn.Flatten())

    def forward(self, images_list):
        features = []
        for i, cam in enumerate(self.camera_names):
            img = images_list[i]
            if cam == 'gelsight':
                features.append(self.gelsight_encoder(img))
            else:
                features.append(self.shared_vision(img))
        return torch.cat(features, dim=-1)


# ==================== Dataset ====================
class DPJointDataset(torch.utils.data.Dataset):
    """Per-episode random sampling (like ACT), with obs_horizon support."""

    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats,
                 pred_horizon, obs_horizon=2,
                 proprio_key="proprio_joint", action_key="actions/joint_abs",
                 tac_side="left", tac_img_key="img"):
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.proprio_key = proprio_key
        self.action_key = action_key
        self.tac_side = tac_side
        self.tac_img_key = tac_img_key

        self.action_min = np.array(norm_stats["action_min"])
        self.action_max = np.array(norm_stats["action_max"])
        self.qpos_min = np.array(norm_stats["qpos_min"])
        self.qpos_max = np.array(norm_stats["qpos_max"])

        self.image_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.episode_ids)

    def _minmax_norm(self, x, xmin, xmax):
        return (x - xmin) / (xmax - xmin + 1e-8) * 2 - 1

    def _load_image(self, f, cam, t):
        if cam == 'gelsight':
            img = f[f'observations/tac/{self.tac_side}/{self.tac_img_key}'][t]
        else:
            img = f[f'observations/images/{cam}'][t]
        img_t = torch.tensor(img, dtype=torch.float32) / 255.0
        img_t = img_t.permute(2, 0, 1)
        return self.image_normalize(img_t)

    def __getitem__(self, index):
        ep_id = self.episode_ids[index]
        path = os.path.join(self.dataset_dir, f'episode_{ep_id}.hdf5')

        with h5py.File(path, 'r') as f:
            ep_len = f[f'observations/{self.proprio_key}'].shape[0]
            max_start = ep_len - self.pred_horizon
            start_ts = np.random.randint(0, max(1, max_start))

            # obs: obs_horizon frames ending at start_ts
            obs_indices = []
            for k in range(self.obs_horizon):
                t = max(0, start_ts - self.obs_horizon + 1 + k)
                obs_indices.append(t)

            # load obs
            all_qpos = []
            all_images = []  # list of list: [cam_idx][obs_step]
            for cam in self.camera_names:
                all_images.append([])

            for t in obs_indices:
                all_qpos.append(f[f'observations/{self.proprio_key}'][t])
                for ci, cam in enumerate(self.camera_names):
                    all_images[ci].append(self._load_image(f, cam, t))

            # action chunk
            action_end = min(start_ts + self.pred_horizon, ep_len)
            action = f[f'/{self.action_key}'][start_ts:action_end]

        # pad action if needed
        action_len = action.shape[0]
        if action_len < self.pred_horizon:
            pad = np.tile(action[-1:], (self.pred_horizon - action_len, 1))
            action = np.concatenate([action, pad], axis=0)

        # stack per camera: (obs_horizon, C, H, W)
        images_per_cam = [torch.stack(imgs) for imgs in all_images]

        qpos = np.stack(all_qpos)
        qpos_norm = self._minmax_norm(qpos, self.qpos_min, self.qpos_max)
        action_norm = self._minmax_norm(action, self.action_min, self.action_max)

        return {
            'images': images_per_cam,
            'qpos': torch.tensor(qpos_norm, dtype=torch.float32),
            'action': torch.tensor(action_norm, dtype=torch.float32),
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
    parser.add_argument('--camera_names', type=str, default='global,wrist,gelsight')
    parser.add_argument('--proprio_key', type=str, default='proprio_joint')
    parser.add_argument('--action_key', type=str, default='actions/joint_abs')
    parser.add_argument('--tac_side', type=str, default='left')
    parser.add_argument('--tac_img_key', type=str, default='img')
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
    parser.add_argument('--clip_vision_path', type=str, default=None)
    parser.add_argument('--clip_tac_path', type=str, default=None)
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

    train_dataset = DPJointDataset(train_ids, args.dataset_dir, camera_names,
                                   norm_stats, args.pred_horizon, args.obs_horizon,
                                   args.proprio_key, args.action_key,
                                   args.tac_side, args.tac_img_key)
    val_dataset = DPJointDataset(val_ids, args.dataset_dir, camera_names,
                                 norm_stats, args.pred_horizon, args.obs_horizon,
                                 args.proprio_key, args.action_key,
                                 args.tac_side, args.tac_img_key)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True,
                              persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=2, pin_memory=True,
                            persistent_workers=True)

    # model
    action_dim = norm_stats['action_min'].shape[0]
    qpos_dim = norm_stats['qpos_min'].shape[0]
    vis_feat = 512
    n_cams = len(camera_names)
    global_cond_dim = (vis_feat * n_cams + qpos_dim) * args.obs_horizon

    vision_encoder = VisionEncoder(camera_names, args.clip_vision_path, args.clip_tac_path).to(device)
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
    config['action_dim'] = int(action_dim)
    config['global_cond_dim'] = int(global_cond_dim)
    config['down_dims'] = down_dims
    config['use_ema'] = use_ema
    config['n_train'] = len(train_ids)
    config['n_val'] = len(val_ids)
    ns = {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in norm_stats.items()}
    config['norm_stats'] = ns
    with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print(f"=== DP Training (official-aligned) ===")
    print(f"action_dim={action_dim}, global_cond_dim={global_cond_dim}, down_dims={down_dims}")
    print(f"pred_horizon={args.pred_horizon}, obs_horizon={args.obs_horizon}")
    print(f"EMA={use_ema}, inference_steps={args.num_inference_steps}")
    print(f"Train: {len(train_ids)} eps, Val: {len(val_ids)} eps")
    print(f"Batch: {args.batch_size}, Epochs: {args.epochs}")

    best_val = float('inf')
    train_losses, val_losses = [], []
    global_step = 0

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
                imgs_t = [batch['images'][ci][:, t].to(device) for ci in range(len(camera_names))]
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

                    obs_feats = []
                    for t in range(args.obs_horizon):
                        imgs_t = [batch['images'][ci][:, t].to(device) for ci in range(len(camera_names))]
                        vf = vision_encoder(imgs_t)
                        obs_feats.append(torch.cat([vf, qpos[:, t]], dim=-1))
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
