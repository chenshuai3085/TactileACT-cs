"""
Train Diffusion Policy — DAgger-style β-decay sampling.

Based on train_dp_official.py but with two data sources:
  - success_dir: expert demonstrations (clean insertion)
  - bounce_dir:  recovery demonstrations (hit-adjust-retry)

β schedule controls sampling ratio:
  β=1.0 → 100% success, 0% bounce
  β=0.6 → 60% success, 40% bounce
  Linear decay from beta_start to beta_end over training epochs.
  Optional warmup: first beta_warmup_epochs use pure success (β=1.0).

Everything else (model, EMA, TopK ckpt, cosine LR) is identical to official DP.
"""
import os, sys, json, argparse, copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
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


# ==================== Vision Encoder ====================
class OfficialVisionEncoder(nn.Module):
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
    """Preload all data into RAM, exhaustive sliding-window sampling."""

    def __init__(self, episode_entries, camera_names, norm_stats,
                 pred_horizon, obs_horizon=2,
                 proprio_key="proprio_joint", action_key="actions/joint_abs",
                 resize_shape=(240, 320), crop_shape=(216, 288), is_train=True):
        self.camera_names = camera_names
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon

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

        print(f"  DPOfficialDataset: {len(self.episodes)} episodes, "
              f"{total_frames} total frames, {len(self.indices)} windows "
              f"({'train' if is_train else 'val'})")

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

        obs_indices = [max(0, start_ts - self.obs_horizon + 1 + k)
                       for k in range(self.obs_horizon)]

        all_images = {cam: [] for cam in self.camera_names}
        all_qpos = []
        for t in obs_indices:
            all_qpos.append(ep['qpos'][t])
            for cam in self.camera_names:
                all_images[cam].append(self._process_image(ep['images'][cam][t]))

        action_end = min(start_ts + self.pred_horizon, ep_len)
        action = ep['action'][start_ts:action_end]
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


def compute_beta(epoch, total_epochs, beta_start, beta_end, warmup_epochs):
    """Compute current β value with optional warmup."""
    if epoch < warmup_epochs:
        return beta_start
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return beta_start + (beta_end - beta_start) * progress


def build_weighted_sampler(n_success, n_bounce, beta, num_samples):
    """Build WeightedRandomSampler for current β.

    success weight = β / n_success   (per-sample, so total mass for success = β)
    bounce weight  = (1-β) / n_bounce (per-sample, so total mass for bounce = 1-β)

    When β=1.0, bounce weights are 0 → only success sampled.
    """
    if beta >= 1.0:
        # Pure success: weight=1 for success, weight=0 for bounce
        weights = [1.0] * n_success + [0.0] * n_bounce
    else:
        w_success = beta / n_success
        w_bounce = (1.0 - beta) / n_bounce
        weights = [w_success] * n_success + [w_bounce] * n_bounce
    return WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)


# ==================== Training ====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--success_dir', type=str, required=True,
                        help='Directory of expert (success) episodes')
    parser.add_argument('--bounce_dir', type=str, required=True,
                        help='Directory of recovery (bounce) episodes')
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--camera_names', type=str, default='global,wrist')
    parser.add_argument('--proprio_key', type=str, default='proprio_joint')
    parser.add_argument('--action_key', type=str, default='actions/joint_abs')
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
    parser.add_argument('--gpu', type=int, default=0)
    # DAgger β schedule
    parser.add_argument('--beta_start', type=float, default=1.0,
                        help='Initial β (1.0 = pure success)')
    parser.add_argument('--beta_end', type=float, default=0.6,
                        help='Final β (0.6 = 60%% success + 40%% bounce)')
    parser.add_argument('--beta_warmup_epochs', type=int, default=50,
                        help='Epochs to keep β=beta_start before decay')
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')
    camera_names = args.camera_names.split(',')
    down_dims = [int(x) for x in args.down_dims.split(',')]
    resize_shape = tuple(int(x) for x in args.resize_shape.split(','))
    crop_shape = tuple(int(x) for x in args.crop_shape.split(','))

    assert os.path.isdir(args.success_dir), f"Success dir not found: {args.success_dir}"
    assert os.path.isdir(args.bounce_dir), f"Bounce dir not found: {args.bounce_dir}"

    # norm stats from both datasets combined
    norm_stats = get_minmax_stats([args.success_dir, args.bounce_dir],
                                  args.proprio_key, args.action_key)

    # build entries
    def _get_entries(ds_dir):
        episode_files = sorted([f for f in os.listdir(ds_dir)
                                if f.startswith('episode_') and f.endswith('.hdf5')])
        return [(ds_dir, int(f.split('_')[1].split('.')[0])) for f in episode_files]

    success_entries = _get_entries(args.success_dir)
    bounce_entries = _get_entries(args.bounce_dir)
    print(f"Success episodes: {len(success_entries)}")
    print(f"Bounce episodes:  {len(bounce_entries)}")

    # build datasets separately
    print("\n--- Loading SUCCESS dataset ---")
    success_dataset = DPOfficialDataset(success_entries, camera_names,
                                         norm_stats, args.pred_horizon, args.obs_horizon,
                                         args.proprio_key, args.action_key,
                                         resize_shape=resize_shape,
                                         crop_shape=crop_shape, is_train=True)
    print("\n--- Loading BOUNCE dataset ---")
    bounce_dataset = DPOfficialDataset(bounce_entries, camera_names,
                                        norm_stats, args.pred_horizon, args.obs_horizon,
                                        args.proprio_key, args.action_key,
                                        resize_shape=resize_shape,
                                        crop_shape=crop_shape, is_train=True)

    n_success = len(success_dataset)
    n_bounce = len(bounce_dataset)
    combined_dataset = ConcatDataset([success_dataset, bounce_dataset])
    # ConcatDataset: indices [0, n_success) → success, [n_success, n_success+n_bounce) → bounce

    # samples per epoch = success windows count (keep epoch length similar to pure-success training)
    samples_per_epoch = n_success

    # model
    action_dim = norm_stats['action_min'].shape[0]
    qpos_dim = norm_stats['qpos_min'].shape[0]

    vision_encoder = OfficialVisionEncoder(camera_names).to(device)
    vis_feat_dim = vision_encoder.feat_dim
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

    steps_per_epoch = (samples_per_epoch + args.batch_size - 1) // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))
    lr_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # save config
    config = vars(args)
    config['resize_shape'] = list(resize_shape)
    config['crop_shape'] = list(crop_shape)
    config['action_dim'] = int(action_dim)
    config['global_cond_dim'] = int(global_cond_dim)
    config['vis_feat_dim'] = int(vis_feat_dim)
    config['down_dims'] = down_dims
    config['use_ema'] = use_ema
    config['n_success_episodes'] = len(success_entries)
    config['n_bounce_episodes'] = len(bounce_entries)
    config['n_success_windows'] = n_success
    config['n_bounce_windows'] = n_bounce
    config['samples_per_epoch'] = samples_per_epoch
    config['variant'] = 'dagger_beta_decay'
    ns = {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in norm_stats.items()}
    config['norm_stats'] = ns
    with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n=== DP DAgger-style β-decay Training ===")
    print(f"action_dim={action_dim}, global_cond_dim={global_cond_dim}")
    print(f"vis_feat_dim={vis_feat_dim}/camera, cameras={camera_names}")
    print(f"pred_horizon={args.pred_horizon}, obs_horizon={args.obs_horizon}")
    print(f"Success: {len(success_entries)} eps ({n_success} windows)")
    print(f"Bounce:  {len(bounce_entries)} eps ({n_bounce} windows)")
    print(f"Samples/epoch: {samples_per_epoch}")
    print(f"β schedule: {args.beta_start} → {args.beta_end} "
          f"(warmup {args.beta_warmup_epochs} eps)")
    print(f"EMA={use_ema}, Batch={args.batch_size}, Epochs={args.epochs}")

    # TopK checkpoint manager
    topk_train_losses = {}
    topk_k = 5

    train_losses = []
    beta_history = []
    global_step = 0

    for epoch in range(args.epochs):
        # compute current β and build sampler
        beta = compute_beta(epoch, args.epochs, args.beta_start,
                            args.beta_end, args.beta_warmup_epochs)
        beta_history.append(beta)

        sampler = build_weighted_sampler(n_success, n_bounce, beta, samples_per_epoch)
        train_loader = DataLoader(combined_dataset, batch_size=args.batch_size,
                                  sampler=sampler, num_workers=0, pin_memory=True,
                                  collate_fn=dp_official_collate)

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

        # TopK checkpoint saving
        def _save_ckpt(path):
            sd = {'epoch': epoch, 'train_loss': train_loss, 'beta': beta}
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

        success_pct = beta * 100
        bounce_pct = (1 - beta) * 100
        print(f"Ep {epoch+1}/{args.epochs} | train={train_loss:.6f} | "
              f"best={min(topk_train_losses.values()):.6f} | "
              f"β={beta:.3f} (S:{success_pct:.0f}%/B:{bounce_pct:.0f}%) | "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

        if (epoch + 1) % args.save_freq == 0:
            sd = {'noise_pred_net': noise_pred_net.state_dict(),
                  'vision_encoder': vision_encoder.state_dict(),
                  'epoch': epoch, 'beta': beta}
            if ema_vis:
                sd['ema_vis'] = ema_vis.state_dict()
                sd['ema_net'] = ema_net.state_dict()
            torch.save(sd, os.path.join(args.save_dir, f'dp_epoch{epoch+1}.pth'))

    # save final
    sd = {'noise_pred_net': noise_pred_net.state_dict(),
          'vision_encoder': vision_encoder.state_dict(), 'epoch': args.epochs - 1}
    if ema_vis:
        sd['ema_vis'] = ema_vis.state_dict()
        sd['ema_net'] = ema_net.state_dict()
    torch.save(sd, os.path.join(args.save_dir, 'dp_final.pth'))

    np.save(os.path.join(args.save_dir, 'train_losses.npy'), train_losses)
    np.save(os.path.join(args.save_dir, 'beta_history.npy'), beta_history)
    print(f"\nDone! Best train_loss: {min(topk_train_losses.values()):.6f}")
    print(f"Top-{topk_k} checkpoints saved in {args.save_dir}")


if __name__ == '__main__':
    main()
