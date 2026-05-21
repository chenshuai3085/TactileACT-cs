"""
Diagnose: Demo-distribution scoring vs current naive scoring.

Compares three scoring methods on K candidates:
1. Current: score = -||z_pred - z_current||  (smaller change = better)
2. Demo global: score = -||z_pred - mu_global|| / sigma_global  (mahalanobis-like)
3. Demo staged: score = -||z_pred - mu_stage|| / sigma_stage  (stage-aware)

Tests whether demo-distribution scoring gives better candidate separation.

Usage:
    python scripts/diagnose_demo_scoring.py --gpu 0
"""
import os, sys, json, pickle, argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import copy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'diffusion'))
sys.path.insert(0, os.path.join(ROOT, 'TFAC_V5'))

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from network import ConditionalUnet1D, get_resnet, replace_bn_with_gn
from pretrain_latent_foresight import LatentForesightPretrainModel
from tactile_vae import build_tactile_vae
import torchvision.transforms as transforms
import h5py


class OfficialVisionEncoder(torch.nn.Module):
    def __init__(self, camera_names):
        super().__init__()
        self.camera_names = camera_names
        base = get_resnet('resnet18')
        replace_bn_with_gn(base, features_per_group=16)
        self.encoders = torch.nn.ModuleDict()
        for cam in camera_names:
            self.encoders[cam] = copy.deepcopy(base)
        self.feat_dim = 512

    def forward(self, images_dict):
        features = []
        for cam in self.camera_names:
            feat = self.encoders[cam](images_dict[cam])
            features.append(feat)
        return torch.cat(features, dim=-1)


class FrozenTactileVAEEncoder(torch.nn.Module):
    TAC_MEAN = np.array([0.2102, -0.6422], dtype=np.float32)
    TAC_STD = np.array([1.6805, 3.6717], dtype=np.float32)

    def __init__(self, vae_checkpoint_path, latent_dim=16, temporal_window=8):
        super().__init__()
        self.vae = build_tactile_vae(latent_dim=latent_dim, temporal_window=temporal_window)
        if vae_checkpoint_path and os.path.exists(vae_checkpoint_path):
            ckpt = torch.load(vae_checkpoint_path, map_location='cpu', weights_only=False)
            if 'model_state_dict' in ckpt:
                self.vae.load_state_dict(ckpt['model_state_dict'])
            else:
                self.vae.load_state_dict(ckpt)
        self.vae.eval()
        self.vae.requires_grad_(False)
        self.feat_dim = latent_dim * 3 * 3
        self.register_buffer('tac_mean', torch.tensor(self.TAC_MEAN))
        self.register_buffer('tac_std', torch.tensor(self.TAC_STD))

    @torch.no_grad()
    def forward(self, marker_seq):
        marker_norm = (marker_seq - self.tac_mean) / self.tac_std
        if marker_norm.dim() == 4:
            marker_norm = marker_norm.unsqueeze(1)
        z_last, _ = self.vae.encode_single_frame(marker_norm)
        return z_last.flatten(1)


def load_foresight(foresight_dir, joint_ckpt_path, device):
    args_path = os.path.join(foresight_dir, 'args.json')
    with open(args_path) as f:
        fs_config = json.load(f)

    camera_names = fs_config.get('camera_names', ['global', 'wrist', 'gelsight'])
    cam_backbone_mapping = {cam: 0 for cam in camera_names}

    model = LatentForesightPretrainModel(
        camera_names=camera_names,
        cam_backbone_mapping=cam_backbone_mapping,
        hidden_dim=fs_config.get('hidden_dim', 512),
        state_dim=fs_config.get('state_dim', 7),
        foresight_layers=fs_config.get('foresight_layers', 3),
        foresight_nheads=fs_config.get('foresight_nheads', 8),
        foresight_dim_feedforward=fs_config.get('foresight_dim_feedforward', 2048),
        dropout=fs_config.get('dropout', 0.1),
        tactile_mode=fs_config.get('tactile_mode', 'marker'),
        max_history=fs_config.get('max_history', 8),
        predict_horizon=fs_config.get('predict_horizon', 1),
        tactile_vae_ckpt=fs_config.get('tactile_vae_ckpt'),
        tactile_vae_latent_dim=fs_config.get('tactile_vae_latent_dim', 16),
        use_delta_pred=fs_config.get('use_delta_pred', False),
        residual_prediction=fs_config.get('residual_prediction', False),
    ).to(device)

    ckpt = torch.load(joint_ckpt_path, map_location=device, weights_only=False)
    if 'foresight' in ckpt:
        model.load_state_dict(ckpt['foresight'], strict=False)

    model.eval()

    stats_path = os.path.join(foresight_dir, 'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    fs_norm = {
        'action_mean': torch.tensor(stats['action_mean'], dtype=torch.float32, device=device),
        'action_std': torch.tensor(stats['action_std'], dtype=torch.float32, device=device),
        'qpos_mean': torch.tensor(stats['qpos_mean'], dtype=torch.float32, device=device),
        'qpos_std': torch.tensor(stats['qpos_std'], dtype=torch.float32, device=device),
    }
    return model, fs_config, fs_norm


def load_demo_stats(stats_path, device):
    with open(stats_path, 'rb') as f:
        data = pickle.load(f)

    n_bins = data['n_bins']
    global_mu = torch.tensor(data['global_mu'], dtype=torch.float32, device=device)
    global_std = torch.tensor(data['global_std'], dtype=torch.float32, device=device)

    stage_mus = []
    stage_stds = []
    for s in data['stage_stats']:
        stage_mus.append(torch.tensor(s['mu'], dtype=torch.float32, device=device))
        stage_stds.append(torch.tensor(s['std'], dtype=torch.float32, device=device))

    return {
        'n_bins': n_bins,
        'global_mu': global_mu,
        'global_std': global_std,
        'stage_mus': stage_mus,
        'stage_stds': stage_stds,
    }


def load_sample_obs(dataset_dir, episode_idx, timestep, camera_names, tac_history, device):
    path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
    resize_tf = transforms.Resize((240, 320))
    crop_tf = transforms.CenterCrop((216, 288))
    img_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    with h5py.File(path, 'r') as f:
        qpos = f['observations/proprio_joint'][()].astype(np.float32)
        marker = f['observations/tac/left/marker_offset'][()].astype(np.float32)
        ep_len = qpos.shape[0]
        images = {}
        for cam in camera_names:
            raw = f[f'observations/images/{cam}'][()]
            images[cam] = raw

    obs_images = {}
    for cam in camera_names:
        img = torch.from_numpy(images[cam][timestep]).float().div_(255.0).permute(2, 0, 1)
        img = resize_tf(img)
        img = crop_tf(img)
        img = img_norm(img)
        obs_images[cam] = img.unsqueeze(0).to(device)

    frames = []
    for k in range(tac_history):
        idx = max(0, timestep - tac_history + 1 + k)
        frames.append(marker[idx])
    marker_hist = np.stack(frames, axis=0)
    marker_hist_t = torch.from_numpy(marker_hist).unsqueeze(0).to(device)

    qpos_t = torch.from_numpy(qpos[timestep]).unsqueeze(0).to(device)
    progress = timestep / (ep_len - 1)

    return obs_images, marker_hist_t, qpos_t, progress, ep_len


@torch.no_grad()
def sample_k_candidates(noise_pred_net, noise_scheduler, obs_cond, action_dim,
                        pred_horizon, K, device):
    noise_scheduler.set_timesteps(100)
    obs_cond_K = obs_cond.expand(K, -1)
    action = torch.randn((K, pred_horizon, action_dim), device=device)
    for t in noise_scheduler.timesteps:
        noise_pred = noise_pred_net(
            action, t.unsqueeze(0).expand(K).to(device), global_cond=obs_cond_K
        )
        action = noise_scheduler.step(noise_pred, t, action).prev_sample
    return action


@torch.no_grad()
def foresight_predict(foresight, fs_config, fs_norm, candidates,
                      obs_images, marker_hist, qpos_raw,
                      action_min_t, action_max_t, device):
    K = candidates.shape[0]
    chunk_size = fs_config.get('chunk_size', 10)

    x0_raw = (candidates + 1) / 2 * (action_max_t - action_min_t) + action_min_t
    x0_fs = (x0_raw - fs_norm['action_mean']) / fs_norm['action_std']
    x0_fs_chunk = x0_fs[:, :chunk_size, :]

    fs_camera_names = fs_config.get('camera_names', ['global', 'wrist', 'gelsight'])
    fs_images = []
    for cam in fs_camera_names:
        if cam == 'gelsight':
            fs_images.append(marker_hist.expand(K, -1, -1, -1, -1))
        elif cam in obs_images:
            fs_images.append(obs_images[cam].expand(K, -1, -1, -1))

    qpos_fs = (qpos_raw - fs_norm['qpos_mean']) / fs_norm['qpos_std']
    qpos_fs = qpos_fs.expand(K, -1)

    z_pred, _, _, z_current, _, _ = foresight(
        fs_images, x0_fs_chunk, future_images=None, qpos=qpos_fs
    )
    if z_pred.dim() == 3:
        z_pred = z_pred[:, -1, :]

    return z_pred, z_current[:1]


def score_naive(z_pred, z_cur):
    """Current method: score = -||z_pred - z_cur||"""
    delta = torch.norm(z_pred - z_cur.expand_as(z_pred), dim=-1)
    return -delta


def score_demo_global(z_pred, demo_stats):
    """Global demo distribution: standardized distance"""
    mu = demo_stats['global_mu']
    std = demo_stats['global_std']
    deviation = ((z_pred - mu) / std).abs().mean(dim=-1)
    return -deviation


def score_demo_staged(z_pred, demo_stats, progress):
    """Stage-aware demo distribution: use the right bin"""
    n_bins = demo_stats['n_bins']
    bin_idx = min(int(progress * n_bins), n_bins - 1)
    mu = demo_stats['stage_mus'][bin_idx]
    std = demo_stats['stage_stds'][bin_idx]
    deviation = ((z_pred - mu) / std).abs().mean(dim=-1)
    return -deviation


def score_demo_staged_mahal(z_pred, demo_stats, progress):
    """Stage-aware with L2 norm (Euclidean in normalized space)"""
    n_bins = demo_stats['n_bins']
    bin_idx = min(int(progress * n_bins), n_bins - 1)
    mu = demo_stats['stage_mus'][bin_idx]
    std = demo_stats['stage_stds'][bin_idx]
    normalized_diff = (z_pred - mu) / std
    dist = torch.norm(normalized_diff, dim=-1)
    return -dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--K', type=int, default=16)
    parser.add_argument('--n_episodes', type=int, default=5)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    K = args.K

    # Paths
    joint_ckpt_dir = '/home/chenshuai/Project/output/ckpt/dp_foresight_joint_0209'
    joint_ckpt_path = os.path.join(joint_ckpt_dir, 'dp_topk_ep130_loss0.0029.pth')
    foresight_dir = '/home/chenshuai/Project/output/foresight_ckpt/latent_foresight_0209'
    dataset_dir = '/home/chenshuai/data/dataset/0209-0210_truncated'
    vae_ckpt = '/home/chenshuai/Project/output/tactile_vae_full/best_tactile_vae.pt'
    demo_stats_path = '/home/chenshuai/Project/output/demo_z_stats.pkl'

    with open(os.path.join(joint_ckpt_dir, 'config.json')) as f:
        config = json.load(f)

    camera_names = config['camera_names']
    if isinstance(camera_names, str):
        camera_names = camera_names.split(',')
    action_dim = config['action_dim']
    pred_horizon = config['pred_horizon']
    obs_horizon = config.get('obs_horizon', 2)
    tac_history = config.get('tac_history', 8)
    num_train_timesteps = config.get('num_train_timesteps', 100)

    ns = config['norm_stats']
    action_min_t = torch.tensor(ns['action_min'], dtype=torch.float32, device=device)
    action_max_t = torch.tensor(ns['action_max'], dtype=torch.float32, device=device)
    qpos_min = np.array(ns['qpos_min'], dtype=np.float32)
    qpos_max = np.array(ns['qpos_max'], dtype=np.float32)

    # Load models
    print("Loading models...")
    vis_cams = [c for c in camera_names if c != 'gelsight']
    vision_encoder = OfficialVisionEncoder(vis_cams).to(device)
    tac_encoder = FrozenTactileVAEEncoder(vae_ckpt, latent_dim=16, temporal_window=tac_history).to(device)
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=config['global_cond_dim'],
        diffusion_step_embed_dim=config.get('diffusion_step_embed_dim', 128),
        down_dims=config['down_dims'],
        kernel_size=5,
    ).to(device)

    ckpt = torch.load(joint_ckpt_path, map_location=device, weights_only=False)
    vision_encoder.load_state_dict(ckpt.get('ema_vis', ckpt.get('vision_encoder')))
    noise_pred_net.load_state_dict(ckpt.get('ema_net', ckpt.get('noise_pred_net')))
    vision_encoder.eval()
    noise_pred_net.eval()
    tac_encoder.eval()

    foresight, fs_config, fs_norm = load_foresight(foresight_dir, joint_ckpt_path, device)

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon',
    )

    # Load demo stats
    demo_stats = load_demo_stats(demo_stats_path, device)
    print(f"Demo stats: {demo_stats['n_bins']} bins")

    # Also load GT future z for comparison (oracle scoring)
    print("\n" + "=" * 70)
    print("SCORING METHOD COMPARISON: K={} candidates".format(K))
    print("=" * 70)

    test_episodes = [0, 5, 10, 30, 50][:args.n_episodes]
    test_timesteps = [30, 80, 130, 180, 230]  # early → late

    all_results = {
        'naive': [], 'demo_global': [], 'demo_staged': [], 'demo_staged_mahal': []
    }

    for ep_idx in test_episodes:
        for ts in test_timesteps:
            try:
                obs_images, marker_hist, qpos_raw, progress, ep_len = load_sample_obs(
                    dataset_dir, ep_idx, ts, vis_cams, tac_history, device
                )
            except Exception as e:
                continue

            # Encode obs
            with torch.no_grad():
                vis_feat = vision_encoder(obs_images)
                tac_feat = tac_encoder(marker_hist)
                qpos_norm = torch.from_numpy(
                    (qpos_raw.cpu().numpy() - qpos_min) / (qpos_max - qpos_min + 1e-8) * 2 - 1
                ).float().to(device)
                obs_cond = torch.cat([vis_feat, tac_feat, qpos_norm], dim=-1)
                if obs_horizon == 2:
                    obs_cond = torch.cat([obs_cond, obs_cond], dim=-1)

            # Sample K candidates
            candidates = sample_k_candidates(
                noise_pred_net, noise_scheduler, obs_cond, action_dim,
                pred_horizon, K, device
            )

            # Foresight predict
            z_pred, z_cur = foresight_predict(
                foresight, fs_config, fs_norm, candidates,
                obs_images, marker_hist, qpos_raw, action_min_t, action_max_t, device
            )

            # Compute all scores
            s_naive = score_naive(z_pred, z_cur)
            s_global = score_demo_global(z_pred, demo_stats)
            s_staged = score_demo_staged(z_pred, demo_stats, progress)
            s_mahal = score_demo_staged_mahal(z_pred, demo_stats, progress)

            # Metrics for each scoring method
            bin_idx = min(int(progress * demo_stats['n_bins']), demo_stats['n_bins'] - 1)

            for name, scores in [('naive', s_naive), ('demo_global', s_global),
                                  ('demo_staged', s_staged), ('demo_staged_mahal', s_mahal)]:
                score_range = (scores.max() - scores.min()).item()
                score_std = scores.std().item()
                score_mean = scores.mean().item()
                # Relative range = range / |mean| (how much spread relative to absolute value)
                rel_range = score_range / (abs(score_mean) + 1e-8)
                all_results[name].append({
                    'range': score_range, 'std': score_std,
                    'mean': score_mean, 'rel_range': rel_range,
                    'ep': ep_idx, 'ts': ts, 'progress': progress, 'bin': bin_idx,
                })

            print(f"\nEp {ep_idx}, ts={ts}, progress={progress:.2f} (bin {bin_idx}):")
            print(f"  {'Method':<20} {'Range':<10} {'Std':<10} {'Mean':<12} {'RelRange%':<10}")
            print(f"  {'-'*60}")
            print(f"  {'Naive(-delta)':<20} {s_naive.max().item()-s_naive.min().item():<10.4f} "
                  f"{s_naive.std().item():<10.4f} {s_naive.mean().item():<12.4f} "
                  f"{(s_naive.max()-s_naive.min()).item()/abs(s_naive.mean().item()+1e-8)*100:<10.2f}")
            print(f"  {'Demo-global':<20} {s_global.max().item()-s_global.min().item():<10.4f} "
                  f"{s_global.std().item():<10.4f} {s_global.mean().item():<12.4f} "
                  f"{(s_global.max()-s_global.min()).item()/abs(s_global.mean().item()+1e-8)*100:<10.2f}")
            print(f"  {'Demo-staged':<20} {s_staged.max().item()-s_staged.min().item():<10.4f} "
                  f"{s_staged.std().item():<10.4f} {s_staged.mean().item():<12.4f} "
                  f"{(s_staged.max()-s_staged.min()).item()/abs(s_staged.mean().item()+1e-8)*100:<10.2f}")
            print(f"  {'Demo-staged-L2':<20} {s_mahal.max().item()-s_mahal.min().item():<10.4f} "
                  f"{s_mahal.std().item():<10.4f} {s_mahal.mean().item():<12.4f} "
                  f"{(s_mahal.max()-s_mahal.min()).item()/abs(s_mahal.mean().item()+1e-8)*100:<10.2f}")

            # Also print: which candidate does each method pick?
            print(f"  Selected: naive={s_naive.argmax().item()}, global={s_global.argmax().item()}, "
                  f"staged={s_staged.argmax().item()}, staged-L2={s_mahal.argmax().item()}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY (averaged across all test points)")
    print("=" * 70)
    print(f"{'Method':<20} {'AvgRange':<10} {'AvgStd':<10} {'AvgRelRange%':<12} {'Agreement'}")
    print("-" * 70)

    for name in ['naive', 'demo_global', 'demo_staged', 'demo_staged_mahal']:
        results = all_results[name]
        if not results:
            continue
        avg_range = np.mean([r['range'] for r in results])
        avg_std = np.mean([r['std'] for r in results])
        avg_rel = np.mean([r['rel_range'] for r in results]) * 100
        print(f"  {name:<20} {avg_range:<10.4f} {avg_std:<10.4f} {avg_rel:<12.2f}%")

    # Cross-method agreement: how often do different methods pick the same candidate?
    print("\n  Method agreement (% of times same best candidate picked):")
    methods = ['naive', 'demo_global', 'demo_staged', 'demo_staged_mahal']
    # We didn't store picks... let's just note it from per-point output above


if __name__ == '__main__':
    main()
