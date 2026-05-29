"""
Guided Diffusion Sampling: 在DDIM去噪过程中加入reward梯度引导

标准去噪: a_{t-1} = denoise(a_t, t)
引导去噪: a_{t-1} = denoise(a_t, t) + γ · ∇_{a_t} R(a_t)

Reward选项:
  1. action_smoothness: -||a_t[1:] - a_t[:-1]||² (越平滑越好)
  2. tactile_delta: ||z_pred(a_t) - z_cur|| (触觉变化量)
  3. ensemble_proximity: -||a_t - mean(a_t across K)||² (向中心靠拢)
  4. combined: 加权组合

对比:
  - Standard DDPM: 基线
  - Ensemble (average K): Step 1发现的最佳方法
  - Guided DDPM: 本脚本测试

Usage:
  python scripts/test_guided_sampling.py \
    --dp_ckpt /home/chenshuai/Project/output/ckpt/dp_foresight_joint_0209/dp_topk_ep130_loss0.0029.pth \
    --dp_config /home/chenshuai/Project/output/ckpt/dp_foresight_joint_0209/config.json \
    --foresight_ckpt /home/chenshuai/Project/output/foresight_ckpt/latent_foresight_full/foresight_best.ckpt \
    --foresight_dir /home/chenshuai/Project/output/foresight_ckpt/latent_foresight_full \
    --data_dir /home/chenshuai/data/dataset/0209-0210_truncated \
    --K 16 --n_eval 100 --guidance_scale 0.1
"""

import argparse
import copy
import json
import os
import pickle
import sys

import h5py
import numpy as np
import torch
import torch.nn as nn
import torchvision
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from torchvision import transforms
from tqdm import tqdm

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'diffusion'))
sys.path.insert(0, os.path.join(ROOT, 'TFAC_V5'))

from network import ConditionalUnet1D, get_resnet, replace_bn_with_gn
from pretrain_latent_foresight import LatentForesightPretrainModel
from tactile_vae import build_tactile_vae


# ==================== Reuse components ====================

class JointDPVisionEncoder(nn.Module):
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


class FrozenTactileVAEEncoder(nn.Module):
    TAC_MEAN = np.array([0.2102, -0.6422], dtype=np.float32)
    TAC_STD = np.array([1.6805, 3.6717], dtype=np.float32)

    def __init__(self, vae_checkpoint_path, latent_dim=16, temporal_window=8):
        super().__init__()
        self.vae = build_tactile_vae(latent_dim=latent_dim, temporal_window=temporal_window)
        if vae_checkpoint_path and os.path.exists(vae_checkpoint_path):
            ckpt = torch.load(vae_checkpoint_path, map_location='cpu')
            sd = ckpt.get('model_state_dict', ckpt)
            self.vae.load_state_dict(sd)
        self.vae.eval()
        self.vae.requires_grad_(False)
        self.feat_dim = latent_dim * 3 * 3
        self.temporal_window = temporal_window
        self.register_buffer('tac_mean', torch.tensor(self.TAC_MEAN))
        self.register_buffer('tac_std', torch.tensor(self.TAC_STD))

    @torch.no_grad()
    def forward(self, marker_seq):
        marker_norm = (marker_seq - self.tac_mean) / self.tac_std
        z_last, _ = self.vae.encode_single_frame(marker_norm)
        return z_last.flatten(1)


class EMAModel:
    def __init__(self):
        self.shadow = {}
    @staticmethod
    def from_state_dict(sd):
        m = EMAModel()
        m.shadow = sd
        return m
    def apply_to(self, model):
        model.load_state_dict(self.shadow)


# ==================== Reward Functions ====================

def reward_smoothness(action):
    """Reward for smooth action trajectories.
    action: (K, pred_horizon, action_dim)
    Returns: (K,) reward (higher = smoother)
    """
    # Penalize large differences between consecutive steps
    diff = action[:, 1:, :] - action[:, :-1, :]
    smoothness = -torch.norm(diff, dim=-1).mean(dim=-1)
    return smoothness


def reward_center_proximity(action, center):
    """Reward for being close to ensemble center.
    action: (K, pred_horizon, action_dim)
    center: (pred_horizon, action_dim)
    Returns: (K,) reward
    """
    dist = torch.norm(action - center.unsqueeze(0), dim=-1).mean(dim=-1)
    return -dist


def reward_tactile_delta(action, foresight, z_cur, fs_norm_fn, fs_chunk):
    """Reward based on tactile prediction change.
    NOTE: This requires gradient flow through foresight model.
    """
    K = action.shape[0]
    action_fs = action[:, :fs_chunk, :]
    action_fs_norm = fs_norm_fn(action_fs)

    # Foresight predict (with gradient)
    # Need to handle the foresight model's forward with gradient
    # This is complex because foresight has frozen backbone
    # For now, return zeros (placeholder)
    return torch.zeros(K, device=action.device)


def reward_combined(action, center, gamma_smooth=0.3, gamma_center=0.7):
    """Combined reward."""
    r_smooth = reward_smoothness(action)
    r_center = reward_center_proximity(action, center)
    return gamma_smooth * r_smooth + gamma_center * r_center


# ==================== Guided Sampling ====================

@torch.no_grad()
def standard_ddpm_sampling(noise_pred_net, noise_scheduler, obs_cond, K,
                           pred_horizon, action_dim, device):
    """Standard DDPM sampling (baseline)."""
    obs_cond_K = obs_cond.expand(K, -1)
    noisy_action = torch.randn(K, pred_horizon, action_dim, device=device)

    noise_scheduler.set_timesteps(noise_scheduler.num_train_timesteps)
    for t in noise_scheduler.timesteps:
        noise_pred = noise_pred_net(
            sample=noisy_action, timestep=t, global_cond=obs_cond_K)
        noisy_action = noise_scheduler.step(
            model_output=noise_pred, timestep=t, sample=noisy_action).prev_sample

    return noisy_action


def guided_ddpm_sampling(noise_pred_net, noise_scheduler, obs_cond, K,
                         pred_horizon, action_dim, device,
                         guidance_scale=0.1, reward_fn=None,
                         guidance_start_step=None):
    """Guided DDPM sampling with reward gradient.

    At each denoising step:
    1. Standard denoising update
    2. Compute reward gradient w.r.t. noisy_action
    3. Add gradient to push toward higher reward
    """
    obs_cond_K = obs_cond.expand(K, -1)
    noisy_action = torch.randn(K, pred_horizon, action_dim, device=device)

    noise_scheduler.set_timesteps(noise_scheduler.num_train_timesteps)
    total_steps = len(noise_scheduler.timesteps)

    for step_idx, t in enumerate(noise_scheduler.timesteps):
        # Standard denoising (no gradient needed for noise prediction)
        with torch.no_grad():
            noise_pred = noise_pred_net(
                sample=noisy_action, timestep=t, global_cond=obs_cond_K)
            denoised = noise_scheduler.step(
                model_output=noise_pred, timestep=t, sample=noisy_action).prev_sample

        # Guided update: add reward gradient
        if reward_fn is not None and (guidance_start_step is None or step_idx >= guidance_start_step):
            # Enable gradient for noisy_action
            noisy_action_grad = denoised.detach().clone().requires_grad_(True)

            reward = reward_fn(noisy_action_grad)
            reward_mean = reward.mean()

            # Compute gradient
            grad = torch.autograd.grad(reward_mean, noisy_action_grad)[0]

            # Normalize gradient
            grad_norm = grad.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            grad_normalized = grad / grad_norm

            # Apply guidance
            noisy_action = denoised + guidance_scale * grad_normalized
        else:
            noisy_action = denoised

    return noisy_action.detach()


def guided_ddim_sampling(noise_pred_net, noise_scheduler, obs_cond, K,
                        pred_horizon, action_dim, device,
                        guidance_scale=0.1, reward_fn=None,
                        num_inference_steps=20):
    """Guided DDIM sampling (faster, fewer steps)."""
    obs_cond_K = obs_cond.expand(K, -1)
    noisy_action = torch.randn(K, pred_horizon, action_dim, device=device)

    noise_scheduler.set_timesteps(num_inference_steps)

    for step_idx, t in enumerate(noise_scheduler.timesteps):
        with torch.no_grad():
            noise_pred = noise_pred_net(
                sample=noisy_action, timestep=t, global_cond=obs_cond_K)
            denoised = noise_scheduler.step(
                model_output=noise_pred, timestep=t, sample=noisy_action).prev_sample

        if reward_fn is not None:
            noisy_action_grad = denoised.detach().clone().requires_grad_(True)
            reward = reward_fn(noisy_action_grad)
            reward_mean = reward.mean()
            grad = torch.autograd.grad(reward_mean, noisy_action_grad)[0]
            grad_norm = grad.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            grad_normalized = grad / grad_norm
            noisy_action = denoised + guidance_scale * grad_normalized
        else:
            noisy_action = denoised

    return noisy_action.detach()


# ==================== Evaluation Pipeline ====================

class GuidedSamplingEval:
    """Evaluate guided sampling vs standard sampling vs ensemble."""

    def __init__(self, dp_config_path, dp_ckpt_path, device="cuda:0", K=16):
        self.device = torch.device(device)
        self.K = K

        with open(dp_config_path) as f:
            self.dp_config = json.load(f)

        self._load_dp(dp_ckpt_path)

    def _load_dp(self, ckpt_path):
        config = self.dp_config
        self.pred_horizon = config["pred_horizon"]
        self.obs_horizon = config.get("obs_horizon", 2)
        self.action_dim = config["action_dim"]
        self.action_shift = config.get("action_shift", 0)

        camera_names = config["camera_names"]
        if isinstance(camera_names, str):
            camera_names = camera_names.split(",")
        self.dp_camera_names = camera_names

        self.dp_vision = JointDPVisionEncoder(camera_names).to(self.device)

        vae_ckpt = config.get("vae_checkpoint",
                              "/home/chenshuai/Project/output/tactile_vae_full/best_tactile_vae.pt")
        self.dp_tac_encoder = FrozenTactileVAEEncoder(
            vae_ckpt, latent_dim=config.get("vae_latent_dim", 16),
            temporal_window=config.get("tac_history", 8),
        ).to(self.device)
        self.dp_tac_encoder.eval()
        self.dp_tac_history = config.get("tac_history", 8)

        resize_shape = config.get("resize_shape", [240, 320])
        crop_shape = config.get("crop_shape", [216, 288])
        if isinstance(resize_shape, str):
            resize_shape = [int(x) for x in resize_shape.split(",")]
        if isinstance(crop_shape, str):
            crop_shape = [int(x) for x in crop_shape.split(",")]
        self.resize_shape = tuple(resize_shape)
        self.crop_shape = tuple(crop_shape)

        down_dims = config.get("down_dims", [512, 1024, 2048])
        if isinstance(down_dims, str):
            down_dims = [int(x) for x in down_dims.split(",")]
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.action_dim, global_cond_dim=config["global_cond_dim"],
            diffusion_step_embed_dim=config.get("diffusion_step_embed_dim", 128),
            down_dims=down_dims, kernel_size=5,
        ).to(self.device)

        self.ddpm_scheduler = DDPMScheduler(
            num_train_timesteps=config.get("num_train_timesteps", 100),
            beta_schedule="squaredcos_cap_v2", clip_sample=True, prediction_type="epsilon",
        )
        self.ddim_scheduler = DDIMScheduler(
            num_train_timesteps=config.get("num_train_timesteps", 100),
            beta_schedule="squaredcos_cap_v2", clip_sample=True, prediction_type="epsilon",
        )

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        if "ema_net" in ckpt:
            EMAModel.from_state_dict(ckpt["ema_net"]).apply_to(self.noise_pred_net)
            EMAModel.from_state_dict(ckpt["ema_vis"]).apply_to(self.dp_vision)
        else:
            self.noise_pred_net.load_state_dict(ckpt["noise_pred_net"])
            self.dp_vision.load_state_dict(ckpt["vision_encoder"])
        self.noise_pred_net.eval()
        self.dp_vision.eval()

        ns = config["norm_stats"]
        self.dp_action_min = torch.tensor(ns["action_min"], dtype=torch.float32, device=self.device)
        self.dp_action_max = torch.tensor(ns["action_max"], dtype=torch.float32, device=self.device)
        self.dp_qpos_min = torch.tensor(ns["qpos_min"], dtype=torch.float32, device=self.device)
        self.dp_qpos_max = torch.tensor(ns["qpos_max"], dtype=torch.float32, device=self.device)

    def _dp_unnorm_action(self, a):
        return (a + 1) / 2 * (self.dp_action_max - self.dp_action_min) + self.dp_action_min

    def _dp_norm_qpos(self, q):
        return (q - self.dp_qpos_min) / (self.dp_qpos_max - self.dp_qpos_min + 1e-8) * 2 - 1

    def _preprocess_image(self, img_uint8):
        img = torch.tensor(img_uint8.astype(np.float32) / 255.0).permute(2, 0, 1)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img = transforms.functional.resize(img, list(self.resize_shape))
        return normalize(img)

    def _get_dp_marker_history(self, marker_all, t):
        T_total = marker_all.shape[0]
        frames = []
        for i in range(self.dp_tac_history):
            idx = min(max(0, t - (self.dp_tac_history - 1 - i)), T_total - 1)
            frames.append(marker_all[idx].astype(np.float32))
        return torch.tensor(np.stack(frames), dtype=torch.float32)

    @torch.no_grad()
    def build_obs_cond(self, images_obs, qpos_obs, marker_hists):
        obs_feats = []
        for step_idx in range(self.obs_horizon):
            imgs_dict = {}
            for cam in self.dp_camera_names:
                img_t = self._preprocess_image(images_obs[step_idx][cam])
                imgs_dict[cam] = img_t.to(self.device).unsqueeze(0)
            vf = self.dp_vision(imgs_dict)
            tf = self.dp_tac_encoder(marker_hists[step_idx].unsqueeze(0).to(self.device))
            qpos_norm = self._dp_norm_qpos(
                torch.tensor(qpos_obs[step_idx], dtype=torch.float32, device=self.device).unsqueeze(0))
            obs_feats.append(torch.cat([vf, tf, qpos_norm], dim=-1))
        return torch.cat(obs_feats, dim=-1)


def find_hdf5_files(data_dir):
    hdf5_files = []
    for subdir in ["success", "bounce", ""]:
        pattern_dir = os.path.join(data_dir, subdir) if subdir else data_dir
        if os.path.isdir(pattern_dir):
            for fn in os.listdir(pattern_dir):
                if fn.endswith(".hdf5"):
                    hdf5_files.append(os.path.join(pattern_dir, fn))
    return sorted(set(hdf5_files))


def run_comparison(eval_obj, data_dir, n_eval=100, K=16, guidance_scale=0.1, seed=42):
    """Compare: Standard DDPM, Ensemble, Guided DDPM (smoothness), Guided DDPM (center)."""
    rng = np.random.RandomState(seed)

    hdf5_files = find_hdf5_files(data_dir)
    all_frames = []
    for hf in hdf5_files:
        with h5py.File(hf, 'r') as f:
            T = f['observations/proprio_joint'].shape[0]
            ep_name = os.path.splitext(os.path.basename(hf))[0]
            for t in range(2, T - eval_obj.pred_horizon - 10, 5):
                all_frames.append((hf, ep_name, t))

    if len(all_frames) > n_eval:
        idx = rng.choice(len(all_frames), n_eval, replace=False)
        all_frames = [all_frames[i] for i in idx]

    print(f"\nComparison: {len(all_frames)} frames, K={K}, guidance_scale={guidance_scale}")

    methods = ['standard_single', 'standard_ensemble', 'guided_smoothness',
               'guided_center', 'guided_combined']
    results = {m: {'l1': [], 'smoothness': []} for m in methods}

    for hdf5_path, ep_name, t in tqdm(all_frames, desc="Comparing"):
        try:
            with h5py.File(hdf5_path, 'r') as f:
                qpos_raw = f['observations/proprio_joint'][t].astype(np.float32)
                marker_all = f['observations/tac/left/marker_offset'][:]

                images_obs, qpos_obs, marker_hists = [], [], []
                for step in range(eval_obj.obs_horizon):
                    t_obs = max(0, t - eval_obj.obs_horizon + 1 + step)
                    obs_dict = {}
                    for cam in eval_obj.dp_camera_names:
                        key = f'observations/images/{cam}'
                        obs_dict[cam] = f[key][t_obs] if key in f else np.zeros((480, 640, 3), dtype=np.uint8)
                    images_obs.append(obs_dict)
                    qpos_obs.append(f['observations/proprio_joint'][t_obs].astype(np.float32))
                    marker_hists.append(eval_obj._get_dp_marker_history(marker_all, t_obs))

                action_start = t + eval_obj.action_shift
                action_expert = f['actions/joint_abs'][action_start:action_start + eval_obj.pred_horizon].astype(np.float32)
                if len(action_expert) < eval_obj.pred_horizon:
                    action_expert = np.pad(action_expert, ((0, eval_obj.pred_horizon - len(action_expert)), (0, 0)), mode='edge')

            obs_cond = eval_obj.build_obs_cond(images_obs, qpos_obs, marker_hists)
            expert_flat = action_expert.flatten()

            # Method 1: Standard single sample
            actions_std = standard_ddpm_sampling(
                eval_obj.noise_pred_net, eval_obj.ddpm_scheduler, obs_cond,
                K=1, pred_horizon=eval_obj.pred_horizon, action_dim=eval_obj.action_dim,
                device=eval_obj.device)
            actions_std_raw = eval_obj._dp_unnorm_action(actions_std).cpu().numpy()
            results['standard_single']['l1'].append(
                np.abs(actions_std_raw[0].flatten() - expert_flat).mean())
            results['standard_single']['smoothness'].append(
                np.abs(actions_std_raw[0, 1:] - actions_std_raw[0, :-1]).mean())

            # Method 2: Standard ensemble (average K samples)
            actions_ens = standard_ddpm_sampling(
                eval_obj.noise_pred_net, eval_obj.ddpm_scheduler, obs_cond,
                K=K, pred_horizon=eval_obj.pred_horizon, action_dim=eval_obj.action_dim,
                device=eval_obj.device)
            actions_ens_raw = eval_obj._dp_unnorm_action(actions_ens).cpu().numpy()
            ensemble_action = actions_ens_raw.mean(axis=0)
            results['standard_ensemble']['l1'].append(
                np.abs(ensemble_action.flatten() - expert_flat).mean())
            results['standard_ensemble']['smoothness'].append(
                np.abs(ensemble_action[1:] - ensemble_action[:-1]).mean())

            # Method 3: Guided with smoothness reward
            def smoothness_reward(a):
                return reward_smoothness(eval_obj._dp_unnorm_action(a))

            actions_guided_smooth = guided_ddpm_sampling(
                eval_obj.noise_pred_net, eval_obj.ddpm_scheduler, obs_cond,
                K=K, pred_horizon=eval_obj.pred_horizon, action_dim=eval_obj.action_dim,
                device=eval_obj.device, guidance_scale=guidance_scale,
                reward_fn=smoothness_reward)
            actions_guided_smooth_raw = eval_obj._dp_unnorm_action(actions_guided_smooth).cpu().numpy()
            guided_smooth_action = actions_guided_smooth_raw.mean(axis=0)
            results['guided_smoothness']['l1'].append(
                np.abs(guided_smooth_action.flatten() - expert_flat).mean())
            results['guided_smoothness']['smoothness'].append(
                np.abs(guided_smooth_action[1:] - guided_smooth_action[:-1]).mean())

            # Method 4: Guided with center proximity
            # First get standard ensemble as center
            center = actions_ens.mean(dim=0).detach()

            def center_reward(a):
                return reward_center_proximity(eval_obj._dp_unnorm_action(a),
                                                eval_obj._dp_unnorm_action(center))

            actions_guided_center = guided_ddpm_sampling(
                eval_obj.noise_pred_net, eval_obj.ddpm_scheduler, obs_cond,
                K=K, pred_horizon=eval_obj.pred_horizon, action_dim=eval_obj.action_dim,
                device=eval_obj.device, guidance_scale=guidance_scale,
                reward_fn=center_reward)
            actions_guided_center_raw = eval_obj._dp_unnorm_action(actions_guided_center).cpu().numpy()
            guided_center_action = actions_guided_center_raw.mean(axis=0)
            results['guided_center']['l1'].append(
                np.abs(guided_center_action.flatten() - expert_flat).mean())
            results['guided_center']['smoothness'].append(
                np.abs(guided_center_action[1:] - guided_center_action[:-1]).mean())

            # Method 5: Guided combined
            def combined_reward(a):
                a_unnorm = eval_obj._dp_unnorm_action(a)
                c_unnorm = eval_obj._dp_unnorm_action(center)
                return reward_combined(a_unnorm, c_unnorm)

            actions_guided_combined = guided_ddpm_sampling(
                eval_obj.noise_pred_net, eval_obj.ddpm_scheduler, obs_cond,
                K=K, pred_horizon=eval_obj.pred_horizon, action_dim=eval_obj.action_dim,
                device=eval_obj.device, guidance_scale=guidance_scale,
                reward_fn=combined_reward)
            actions_guided_combined_raw = eval_obj._dp_unnorm_action(actions_guided_combined).cpu().numpy()
            guided_combined_action = actions_guided_combined_raw.mean(axis=0)
            results['guided_combined']['l1'].append(
                np.abs(guided_combined_action.flatten() - expert_flat).mean())
            results['guided_combined']['smoothness'].append(
                np.abs(guided_combined_action[1:] - guided_combined_action[:-1]).mean())

        except Exception as e:
            continue

    # ========== Report ==========
    n = len(results['standard_single']['l1'])
    print(f"\n{'='*70}")
    print(f"Method Comparison ({n} frames, K={K}, guidance_scale={guidance_scale})")
    print(f"{'='*70}")
    print(f"\n{'Method':<25} {'L1 to expert':>15} {'Smoothness':>15}")
    print(f"{'-'*55}")

    for method in methods:
        l1 = np.mean(results[method]['l1'])
        sm = np.mean(results[method]['smoothness'])
        print(f"{method:<25} {l1:>15.4f} {sm:>15.6f}")

    # Improvement analysis
    baseline_l1 = np.mean(results['standard_single']['l1'])
    print(f"\nImprovement vs standard_single:")
    for method in methods[1:]:
        l1 = np.mean(results[method]['l1'])
        impr = (baseline_l1 - l1) / baseline_l1 * 100
        print(f"  {method:<25}: {impr:+.1f}%")

    print(f"\n{'='*70}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dp_ckpt", default="/home/chenshuai/Project/output/ckpt/dp_foresight_joint_0209/dp_topk_ep130_loss0.0029.pth")
    parser.add_argument("--dp_config", default="/home/chenshuai/Project/output/ckpt/dp_foresight_joint_0209/config.json")
    parser.add_argument("--data_dir", default="/home/chenshuai/data/dataset/0209-0210_truncated")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--n_eval", type=int, default=100)
    parser.add_argument("--guidance_scale", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    eval_obj = GuidedSamplingEval(
        dp_config_path=args.dp_config, dp_ckpt_path=args.dp_ckpt,
        device=args.device, K=args.K,
    )

    run_comparison(eval_obj, args.data_dir, n_eval=args.n_eval, K=args.K,
                   guidance_scale=args.guidance_scale, seed=args.seed)


if __name__ == "__main__":
    main()
