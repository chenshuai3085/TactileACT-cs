"""
Diagnose: Do K diffusion candidates produce different tactile predictions?

Tests:
1. DDPM vs DDIM: action diversity across K candidates
2. Foresight sensitivity: does LTFT predict different z for different actions?
3. Score separation: are CQF scores meaningfully different across K?

Usage:
    python scripts/diagnose_k_diversity.py --gpu 0
"""
import os, sys, json, pickle
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'diffusion'))
sys.path.insert(0, os.path.join(ROOT, 'TFAC_V5'))

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from network import ConditionalUnet1D
from pretrain_latent_foresight import LatentForesightPretrainModel
from tactile_vae import build_tactile_vae
import torchvision.transforms as transforms
import h5py
import copy
import argparse


# ==================== Models ====================
class OfficialVisionEncoder(torch.nn.Module):
    def __init__(self, camera_names):
        super().__init__()
        from network import get_resnet, replace_bn_with_gn
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
            ckpt = torch.load(vae_checkpoint_path, map_location='cpu')
            if 'model_state_dict' in ckpt:
                self.vae.load_state_dict(ckpt['model_state_dict'])
            else:
                self.vae.load_state_dict(ckpt)
        self.vae.eval()
        self.vae.requires_grad_(False)
        self.feat_dim = latent_dim * 3 * 3
        self.register_buffer('tac_mean', torch.tensor(self.TAC_MEAN))
        self.register_buffer('tac_std', torch.tensor(self.TAC_STD))

    def normalize_marker(self, marker_offset):
        return (marker_offset - self.tac_mean) / self.tac_std

    @torch.no_grad()
    def forward(self, marker_seq):
        # marker_seq: (B, T, 9, 9, 2)
        marker_norm = self.normalize_marker(marker_seq)
        if marker_norm.dim() == 4:
            marker_norm = marker_norm.unsqueeze(1)
        z_last, _ = self.vae.encode_single_frame(marker_norm)
        return z_last.flatten(1)  # (B, 144)


def load_foresight(foresight_dir, joint_ckpt_path, device):
    """Load foresight model from joint checkpoint."""
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

    # Load from joint checkpoint
    ckpt = torch.load(joint_ckpt_path, map_location=device)
    if 'foresight' in ckpt:
        missing, unexpected = model.load_state_dict(ckpt['foresight'], strict=False)
        print(f"[foresight] loaded from joint ckpt, {len(missing)} missing (backbone/VAE)")

    model.eval()

    # Load norm stats
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


def load_sample_obs(dataset_dir, episode_idx, timestep, camera_names, obs_horizon, tac_history, device):
    """Load a single observation from dataset for testing."""
    path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')

    resize_tf = transforms.Resize((240, 320))
    crop_tf = transforms.CenterCrop((216, 288))
    img_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    with h5py.File(path, 'r') as f:
        qpos = f['observations/proprio_joint'][()].astype(np.float32)
        marker = f['observations/tac/left/marker_offset'][()].astype(np.float32)
        images = {}
        for cam in camera_names:
            raw = f[f'observations/images/{cam}'][()]
            images[cam] = raw

    # Process obs at timestep
    obs_images = {}
    for cam in camera_names:
        img = torch.from_numpy(images[cam][timestep]).float().div_(255.0).permute(2, 0, 1)
        img = resize_tf(img)
        img = crop_tf(img)
        img = img_norm(img)
        obs_images[cam] = img.unsqueeze(0).to(device)  # (1, 3, H, W)

    # Marker history
    frames = []
    for k in range(tac_history):
        idx = max(0, timestep - tac_history + 1 + k)
        frames.append(marker[idx])
    marker_hist = np.stack(frames, axis=0)  # (8, 9, 9, 2)
    marker_hist_t = torch.from_numpy(marker_hist).unsqueeze(0).to(device)  # (1, 8, 9, 9, 2)

    qpos_t = torch.from_numpy(qpos[timestep]).unsqueeze(0).to(device)  # (1, 7)

    return obs_images, marker_hist_t, qpos_t


@torch.no_grad()
def sample_k_candidates(noise_pred_net, noise_scheduler, obs_cond, action_dim,
                        pred_horizon, K, device, scheduler_type='ddpm',
                        num_inference_steps=None):
    """Sample K candidates using DDPM or DDIM."""
    if num_inference_steps is None:
        num_inference_steps = 100 if scheduler_type == 'ddpm' else 20

    noise_scheduler.set_timesteps(num_inference_steps)
    obs_cond_K = obs_cond.expand(K, -1)
    action = torch.randn((K, pred_horizon, action_dim), device=device)

    for t in noise_scheduler.timesteps:
        noise_pred = noise_pred_net(
            action, t.unsqueeze(0).expand(K).to(device), global_cond=obs_cond_K
        )
        action = noise_scheduler.step(noise_pred, t, action).prev_sample

    return action  # (K, pred_horizon, action_dim)


@torch.no_grad()
def foresight_predict(foresight, fs_config, fs_norm, candidates,
                      obs_images, marker_hist, qpos_raw,
                      action_min_t, action_max_t, device):
    """Run foresight on K candidates, return z_pred for each."""
    K = candidates.shape[0]
    chunk_size = fs_config.get('chunk_size', 10)

    # Convert DP [-1,1] → raw → Foresight mean/std
    x0_raw = (candidates + 1) / 2 * (action_max_t - action_min_t) + action_min_t
    x0_fs = (x0_raw - fs_norm['action_mean']) / fs_norm['action_std']
    x0_fs_chunk = x0_fs[:, :chunk_size, :]

    # Prepare foresight inputs
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

    return z_pred, z_current[:1]  # (K, 144), (1, 144)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--K', type=int, default=16)
    parser.add_argument('--n_tests', type=int, default=5,
                        help='Number of different observations to test')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    K = args.K

    # Paths
    joint_ckpt_dir = '/home/chenshuai/Project/output/ckpt/dp_foresight_joint_0209'
    joint_ckpt_path = os.path.join(joint_ckpt_dir, 'dp_topk_ep130_loss0.0029.pth')
    foresight_dir = '/home/chenshuai/Project/output/foresight_ckpt/latent_foresight_0209'
    dataset_dir = '/home/chenshuai/data/dataset/0209-0210_truncated'
    vae_ckpt = '/home/chenshuai/Project/output/tactile_vae_full/best_tactile_vae.pt'

    # Load config
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
    action_min = np.array(ns['action_min'], dtype=np.float32)
    action_max = np.array(ns['action_max'], dtype=np.float32)
    qpos_min = np.array(ns['qpos_min'], dtype=np.float32)
    qpos_max = np.array(ns['qpos_max'], dtype=np.float32)
    action_min_t = torch.tensor(action_min, dtype=torch.float32, device=device)
    action_max_t = torch.tensor(action_max, dtype=torch.float32, device=device)

    print("=" * 60)
    print("DIAGNOSIS: K-candidate diversity & foresight sensitivity")
    print("=" * 60)
    print(f"K={K}, action_dim={action_dim}, pred_horizon={pred_horizon}")

    # Build models
    print("\n[1/4] Loading DP models...")
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

    # Load DP weights from joint ckpt
    ckpt = torch.load(joint_ckpt_path, map_location=device)
    vis_sd = ckpt.get('ema_vis', ckpt.get('vision_encoder'))
    vision_encoder.load_state_dict(vis_sd)
    net_sd = ckpt.get('ema_net', ckpt.get('noise_pred_net'))
    noise_pred_net.load_state_dict(net_sd)
    print("  DP loaded (EMA)" if 'ema_vis' in ckpt else "  DP loaded (raw)")

    vision_encoder.eval()
    noise_pred_net.eval()
    tac_encoder.eval()

    print("\n[2/4] Loading Foresight model...")
    foresight, fs_config, fs_norm = load_foresight(foresight_dir, joint_ckpt_path, device)

    # Build schedulers
    ddpm_scheduler = DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon',
    )
    ddim_scheduler = DDIMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon',
    )

    # Test on multiple observations
    print(f"\n[3/4] Testing on {args.n_tests} observations...")
    test_episodes = [0, 10, 50, 100, 200][:args.n_tests]
    test_timesteps = [50, 100, 150]  # different phases of episode

    results_ddpm = []
    results_ddim = []

    for ep_idx in test_episodes:
        for ts in test_timesteps:
            print(f"\n--- Episode {ep_idx}, timestep {ts} ---")

            # Load obs
            try:
                obs_images, marker_hist, qpos_raw = load_sample_obs(
                    dataset_dir, ep_idx, ts, vis_cams, obs_horizon, tac_history, device
                )
            except Exception as e:
                print(f"  Skip: {e}")
                continue

            # Encode obs
            with torch.no_grad():
                vis_feat = vision_encoder(obs_images)
                tac_feat = tac_encoder(marker_hist)  # (1, 8, 9, 9, 2)
                qpos_norm = torch.from_numpy(
                    (qpos_raw.cpu().numpy() - qpos_min) / (qpos_max - qpos_min + 1e-8) * 2 - 1
                ).float().to(device)
                obs_cond = torch.cat([vis_feat, tac_feat, qpos_norm], dim=-1)
                # obs_horizon=2: repeat for 2-frame condition
                if obs_horizon == 2:
                    obs_cond = torch.cat([obs_cond, obs_cond], dim=-1)

            # === DDPM sampling ===
            candidates_ddpm = sample_k_candidates(
                noise_pred_net, ddpm_scheduler, obs_cond, action_dim,
                pred_horizon, K, device, 'ddpm', num_inference_steps=100
            )

            # === DDIM sampling ===
            candidates_ddim = sample_k_candidates(
                noise_pred_net, ddim_scheduler, obs_cond, action_dim,
                pred_horizon, K, device, 'ddim', num_inference_steps=20
            )

            # Analyze action diversity
            def analyze_actions(candidates, label):
                # (K, pred_horizon, action_dim)
                # Pairwise L1 distance
                K_ = candidates.shape[0]
                dists = []
                for i in range(K_):
                    for j in range(i+1, K_):
                        d = (candidates[i] - candidates[j]).abs().mean().item()
                        dists.append(d)
                mean_dist = np.mean(dists)
                max_dist = np.max(dists)
                std_across_k = candidates.std(dim=0).mean().item()
                print(f"  [{label}] action diversity: mean_pairwise_L1={mean_dist:.5f}, "
                      f"max={max_dist:.5f}, std_across_K={std_across_k:.5f}")
                return mean_dist, max_dist, std_across_k

            ddpm_stats = analyze_actions(candidates_ddpm, "DDPM")
            ddim_stats = analyze_actions(candidates_ddim, "DDIM")

            # Foresight predictions
            z_pred_ddpm, z_cur = foresight_predict(
                foresight, fs_config, fs_norm, candidates_ddpm,
                obs_images, marker_hist, qpos_raw, action_min_t, action_max_t, device
            )
            z_pred_ddim, _ = foresight_predict(
                foresight, fs_config, fs_norm, candidates_ddim,
                obs_images, marker_hist, qpos_raw, action_min_t, action_max_t, device
            )

            # Analyze z_pred diversity
            def analyze_z_preds(z_preds, z_cur, label):
                # z_preds: (K, 144)
                K_ = z_preds.shape[0]
                # Pairwise distance between predictions
                z_dists = []
                for i in range(K_):
                    for j in range(i+1, K_):
                        d = (z_preds[i] - z_preds[j]).abs().mean().item()
                        z_dists.append(d)
                mean_z_dist = np.mean(z_dists)
                max_z_dist = np.max(z_dists)
                std_z = z_preds.std(dim=0).mean().item()

                # Scores (current method: -||z_pred - z_cur||)
                deltas = torch.norm(z_preds - z_cur.expand_as(z_preds), dim=-1)
                scores = -deltas
                score_range = (scores.max() - scores.min()).item()
                score_std = scores.std().item()

                print(f"  [{label}] z_pred diversity: mean_pairwise_L1={mean_z_dist:.6f}, "
                      f"max={max_z_dist:.6f}, std_across_K={std_z:.6f}")
                print(f"  [{label}] scores: range={score_range:.4f}, std={score_std:.4f}, "
                      f"min={scores.min().item():.4f}, max={scores.max().item():.4f}")
                return mean_z_dist, score_range, score_std

            z_stats_ddpm = analyze_z_preds(z_pred_ddpm, z_cur, "DDPM")
            z_stats_ddim = analyze_z_preds(z_pred_ddim, z_cur, "DDIM")

            results_ddpm.append((*ddpm_stats, *z_stats_ddpm))
            results_ddim.append((*ddim_stats, *z_stats_ddim))

    # === Summary ===
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if results_ddpm:
        r = np.array(results_ddpm)
        print(f"\nDDPM (100 steps):")
        print(f"  Action diversity (mean pairwise L1):  {r[:, 0].mean():.5f} ± {r[:, 0].std():.5f}")
        print(f"  Action std across K:                  {r[:, 2].mean():.5f}")
        print(f"  z_pred diversity (mean pairwise L1):  {r[:, 3].mean():.6f} ± {r[:, 3].std():.6f}")
        print(f"  Score range (max-min):                {r[:, 4].mean():.4f} ± {r[:, 4].std():.4f}")
        print(f"  Score std:                            {r[:, 5].mean():.4f}")

    if results_ddim:
        r = np.array(results_ddim)
        print(f"\nDDIM (20 steps):")
        print(f"  Action diversity (mean pairwise L1):  {r[:, 0].mean():.5f} ± {r[:, 0].std():.5f}")
        print(f"  Action std across K:                  {r[:, 2].mean():.5f}")
        print(f"  z_pred diversity (mean pairwise L1):  {r[:, 3].mean():.6f} ± {r[:, 3].std():.6f}")
        print(f"  Score range (max-min):                {r[:, 4].mean():.4f} ± {r[:, 4].std():.4f}")
        print(f"  Score std:                            {r[:, 5].mean():.4f}")

    # === Sensitivity test: same obs, manually perturbed action ===
    print("\n" + "=" * 60)
    print("SENSITIVITY TEST: same obs, action perturbation → z_pred change?")
    print("=" * 60)

    obs_images, marker_hist, qpos_raw = load_sample_obs(
        dataset_dir, 0, 100, vis_cams, obs_horizon, tac_history, device
    )
    with torch.no_grad():
        vis_feat = vision_encoder(obs_images)
        tac_feat = tac_encoder(marker_hist)  # (1, 8, 9, 9, 2)
        qpos_norm = torch.from_numpy(
            (qpos_raw.cpu().numpy() - qpos_min) / (qpos_max - qpos_min + 1e-8) * 2 - 1
        ).float().to(device)
        obs_cond = torch.cat([vis_feat, tac_feat, qpos_norm], dim=-1)
        if obs_horizon == 2:
            obs_cond = torch.cat([obs_cond, obs_cond], dim=-1)

    # Get one baseline action
    baseline = sample_k_candidates(
        noise_pred_net, ddpm_scheduler, obs_cond, action_dim,
        pred_horizon, 1, device, 'ddpm', 100
    )  # (1, pred_horizon, 7)

    perturbation_scales = [0.01, 0.05, 0.1, 0.2, 0.5]
    print(f"\nBaseline action norm: {baseline.abs().mean().item():.4f}")

    for scale in perturbation_scales:
        perturbed = baseline + torch.randn_like(baseline) * scale
        perturbed = perturbed.clamp(-1, 1)

        combined = torch.cat([baseline, perturbed], dim=0)  # (2, pred_horizon, 7)

        z_preds, z_cur = foresight_predict(
            foresight, fs_config, fs_norm, combined,
            obs_images, marker_hist, qpos_raw, action_min_t, action_max_t, device
        )

        z_diff = (z_preds[0] - z_preds[1]).abs().mean().item()
        action_diff = (combined[0] - combined[1]).abs().mean().item()
        print(f"  perturbation={scale:.2f} → action_diff={action_diff:.4f}, "
              f"z_pred_diff={z_diff:.6f}, ratio={z_diff/action_diff:.4f}")


if __name__ == '__main__':
    main()
