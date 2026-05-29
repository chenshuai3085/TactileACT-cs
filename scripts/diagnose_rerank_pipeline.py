"""
TacScore Pipeline Diagnostic: 候选多样性 + Scorer区分度 + Oracle Match

用Joint DP + Foresight + PhaseAwareScorer跑完整pipeline，输出:
  1. K候选的pairwise L1 (动作多样性)
  2. z_pred的pairwise L1 (触觉预测多样性)
  3. PhaseAwareScorer的score分布和per-frame range
  4. Oracle Match率 (scorer选中的是否和oracle一致)
  5. Score-L1相关性

Usage:
  python scripts/diagnose_rerank_pipeline.py \
    --dp_ckpt /home/chenshuai/Project/output/ckpt/dp_foresight_joint_0209/dp_topk_ep130_loss0.0029.pth \
    --dp_config /home/chenshuai/Project/output/ckpt/dp_foresight_joint_0209/config.json \
    --foresight_ckpt /home/chenshuai/Project/output/foresight_ckpt/latent_foresight_full/foresight_best.ckpt \
    --foresight_dir /home/chenshuai/Project/output/foresight_ckpt/latent_foresight_full \
    --data_dir /home/chenshuai/data/dataset/0209-0210/truncated \
    --K 16 --n_eval 100
"""

import argparse
import copy
import json
import os
import pickle
import sys
import time

import h5py
import numpy as np
import torch
import torch.nn as nn
import torchvision
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torchvision import transforms
from tqdm import tqdm

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'diffusion'))
sys.path.insert(0, os.path.join(ROOT, 'TFAC_V5'))

from network import ConditionalUnet1D, get_resnet, replace_bn_with_gn
from pretrain_latent_foresight import LatentForesightPretrainModel
from tactile_vae import build_tactile_vae


# ==================== Vision Encoder (Joint DP variant) ====================

class JointDPVisionEncoder(nn.Module):
    """Per-camera ResNet18 + AdaptiveAvgPool → 512 dim, GroupNorm.
    Matches train_dp_foresight_joint.py OfficialVisionEncoder."""

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


# ==================== EMA ====================
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


# ==================== PhaseAwareScorer (from serve_dp_foresight_rerank.py) ====================

class PhaseAwareScorer:
    """Multi-dimensional CQV scorer using phase-annotated demo statistics."""

    def __init__(self, phase_stats_path, device):
        self.device = device
        with open(phase_stats_path, 'rb') as f:
            data = pickle.load(f)
        self.phase_stats = data['phase_stats']
        self.phase_tensors = {}
        for pid, stats in self.phase_stats.items():
            if stats is None:
                continue
            self.phase_tensors[pid] = {
                'delta_norm_mu': stats['delta_norm_mu'],
                'delta_norm_std': stats['delta_norm_std'],
                'z_int_norm_p95': stats['z_int_norm_p95'],
                'z_mu': torch.tensor(stats['z_mu'], dtype=torch.float32, device=device),
                'z_std': torch.tensor(stats['z_std'], dtype=torch.float32, device=device),
            }

    def estimate_phase(self, progress):
        if progress < 0.3:
            return 0
        elif progress < 0.6:
            return 1
        elif progress < 0.75:
            return 2
        elif progress < 0.9:
            return 3
        else:
            return 4

    def score_fit(self, z_pred, phase):
        if phase not in self.phase_tensors:
            return torch.zeros(z_pred.shape[0], device=self.device)
        mu = self.phase_tensors[phase]['z_mu']
        std = self.phase_tensors[phase]['z_std']
        return -((z_pred - mu) / std).abs().mean(dim=-1)

    def score_smoothness(self, z_pred, z_cur, phase):
        delta = torch.norm(z_pred - z_cur.expand_as(z_pred), dim=-1)
        if phase not in self.phase_tensors:
            return -delta
        mu = self.phase_tensors[phase]['delta_norm_mu']
        std = self.phase_tensors[phase]['delta_norm_std']
        return -(delta - mu) / (std + 1e-8)

    def score_safety(self, z_pred, phase):
        z_int = z_pred[:, :9]
        z_int_norm = torch.norm(z_int, dim=-1)
        if phase not in self.phase_tensors:
            return torch.zeros(z_pred.shape[0], device=self.device)
        bound = self.phase_tensors[phase]['z_int_norm_p95']
        return -torch.clamp(z_int_norm - bound, min=0)

    def score(self, z_pred, z_cur, phase, mode='fit'):
        if mode == 'fit':
            return self.score_fit(z_pred, phase)
        elif mode == 'smoothness':
            return self.score_smoothness(z_pred, z_cur, phase)
        elif mode == 'naive':
            return -torch.norm(z_pred - z_cur.expand_as(z_pred), dim=-1)
        elif mode == 'multidim':
            s_fit = self.score_fit(z_pred, phase)
            s_smooth = self.score_smoothness(z_pred, z_cur, phase)
            s_safe = self.score_safety(z_pred, phase)
            return 0.7 * s_fit + 0.2 * s_smooth + 0.1 * s_safe
        else:
            return -torch.norm(z_pred - z_cur.expand_as(z_pred), dim=-1)


# ==================== Pipeline Diagnostic ====================

class PipelineDiagnostic:
    """Load Joint DP + Foresight + PhaseAwareScorer, run full pipeline diagnostic."""

    def __init__(self, dp_config_path, dp_ckpt_path, foresight_ckpt_path,
                 foresight_dir, phase_stats_path, device="cuda:0", K=16):
        self.device = torch.device(device)
        self.K = K

        with open(dp_config_path) as f:
            self.dp_config = json.load(f)

        self._load_dp(dp_ckpt_path)
        self._load_foresight(foresight_ckpt_path, foresight_dir)
        self.scorer = PhaseAwareScorer(phase_stats_path, self.device)

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

        # Joint DP uses AdaptiveAvgPool (512 dim/camera)
        self.dp_vision = JointDPVisionEncoder(camera_names).to(self.device)

        vae_ckpt = config.get("vae_checkpoint",
                              "/home/chenshuai/Project/output/tactile_vae_full/best_tactile_vae.pt")
        self.dp_tac_encoder = FrozenTactileVAEEncoder(
            vae_ckpt,
            latent_dim=config.get("vae_latent_dim", 16),
            temporal_window=config.get("tac_history", 8),
        ).to(self.device)
        self.dp_tac_encoder.eval()

        self.dp_tac_history = config.get("tac_history", 8)

        # Image preprocessing for Joint DP (resize + crop)
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
            input_dim=self.action_dim,
            global_cond_dim=config["global_cond_dim"],
            diffusion_step_embed_dim=config.get("diffusion_step_embed_dim", 128),
            down_dims=down_dims,
            kernel_size=5,
        ).to(self.device)

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=config.get("num_train_timesteps", 100),
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        if "ema_net" in ckpt:
            EMAModel.from_state_dict(ckpt["ema_net"]).apply_to(self.noise_pred_net)
            EMAModel.from_state_dict(ckpt["ema_vis"]).apply_to(self.dp_vision)
            print(f"DP loaded (EMA, epoch {ckpt.get('epoch', '?')})")
        else:
            self.noise_pred_net.load_state_dict(ckpt["noise_pred_net"])
            self.dp_vision.load_state_dict(ckpt["vision_encoder"])
            print(f"DP loaded (epoch {ckpt.get('epoch', '?')})")
        self.noise_pred_net.eval()
        self.dp_vision.eval()

        ns = config["norm_stats"]
        self.dp_action_min = torch.tensor(ns["action_min"], dtype=torch.float32, device=self.device)
        self.dp_action_max = torch.tensor(ns["action_max"], dtype=torch.float32, device=self.device)
        self.dp_qpos_min = torch.tensor(ns["qpos_min"], dtype=torch.float32, device=self.device)
        self.dp_qpos_max = torch.tensor(ns["qpos_max"], dtype=torch.float32, device=self.device)

    def _load_foresight(self, ckpt_path, foresight_dir):
        if foresight_dir is None:
            foresight_dir = os.path.dirname(ckpt_path)
        args_path = os.path.join(foresight_dir, "args.json")
        with open(args_path) as f:
            self.foresight_config = json.load(f)
        config = self.foresight_config

        camera_names = config['camera_names']
        cam_backbone_mapping = {cam: 0 for cam in camera_names}

        self.foresight = LatentForesightPretrainModel(
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
        ).to(self.device)

        state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        self.foresight.load_state_dict(state_dict, strict=False)
        self.foresight.eval()

        stats_path = os.path.join(foresight_dir, "dataset_stats.pkl")
        if os.path.exists(stats_path):
            with open(stats_path, 'rb') as f:
                ns = pickle.load(f)
        else:
            ns = config.get('norm_stats', config)

        self.fs_qpos_mean = torch.tensor(ns['qpos_mean'], dtype=torch.float32, device=self.device)
        self.fs_qpos_std = torch.tensor(ns['qpos_std'], dtype=torch.float32, device=self.device)
        self.fs_action_mean = torch.tensor(ns['action_mean'], dtype=torch.float32, device=self.device)
        self.fs_action_std = torch.tensor(ns['action_std'], dtype=torch.float32, device=self.device)

        self.foresight_chunk = config.get('chunk_size', 10)
        self.vae_window = config.get('tactile_vae_window', 8)
        if self.vae_window is None:
            self.vae_window = 8

        print(f"Foresight loaded from {ckpt_path}")

    # ========== Normalization ==========

    def _dp_unnorm_action(self, action_norm):
        return (action_norm + 1) / 2 * (self.dp_action_max - self.dp_action_min) + self.dp_action_min

    def _dp_norm_qpos(self, qpos_raw):
        return (qpos_raw - self.dp_qpos_min) / (self.dp_qpos_max - self.dp_qpos_min + 1e-8) * 2 - 1

    def _fs_norm_action(self, action_raw):
        return (action_raw - self.fs_action_mean) / self.fs_action_std

    def _fs_norm_qpos(self, qpos_raw):
        return (qpos_raw - self.fs_qpos_mean) / self.fs_qpos_std

    # ========== Image preprocessing ==========

    def _preprocess_image(self, img_uint8):
        """Resize + crop + normalize for Joint DP."""
        img = torch.tensor(img_uint8.astype(np.float32) / 255.0).permute(2, 0, 1)
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img = transforms.functional.resize(img, list(self.resize_shape))
        img = normalize(img)
        return img

    def _get_marker_window(self, marker_all, t):
        T_total = marker_all.shape[0]
        frames = []
        for i in range(self.vae_window):
            idx = max(0, t - (self.vae_window - 1 - i))
            idx = min(idx, T_total - 1)
            frames.append(marker_all[idx].astype(np.float32))
        window = np.stack(frames)
        return torch.tensor(window, dtype=torch.float32)

    def _get_dp_marker_history(self, marker_all, t):
        T_total = marker_all.shape[0]
        frames = []
        for i in range(self.dp_tac_history):
            idx = max(0, t - (self.dp_tac_history - 1 - i))
            idx = min(idx, T_total - 1)
            frames.append(marker_all[idx].astype(np.float32))
        return torch.tensor(np.stack(frames), dtype=torch.float32)

    # ========== Core pipeline ==========

    @torch.no_grad()
    def build_obs_cond(self, images_obs, qpos_obs, marker_hists):
        obs_feats = []
        for step_idx in range(self.obs_horizon):
            imgs_dict = {}
            for cam in self.dp_camera_names:
                img_uint8 = images_obs[step_idx][cam]
                img_t = self._preprocess_image(img_uint8)
                imgs_dict[cam] = img_t.to(self.device).unsqueeze(0)
            vf = self.dp_vision(imgs_dict)

            tf = self.dp_tac_encoder(
                marker_hists[step_idx].unsqueeze(0).to(self.device))

            qpos_norm = self._dp_norm_qpos(
                torch.tensor(qpos_obs[step_idx], dtype=torch.float32,
                             device=self.device).unsqueeze(0))

            obs_feats.append(torch.cat([vf, tf, qpos_norm], dim=-1))

        return torch.cat(obs_feats, dim=-1)

    @torch.no_grad()
    def generate_candidates(self, obs_cond, K=None):
        if K is None:
            K = self.K
        obs_cond_K = obs_cond.expand(K, -1)
        noisy_action = torch.randn(
            K, self.pred_horizon, self.action_dim, device=self.device)

        self.noise_scheduler.set_timesteps(
            self.dp_config.get("num_inference_steps", 100))

        for t in self.noise_scheduler.timesteps:
            noise_pred = self.noise_pred_net(
                sample=noisy_action, timestep=t, global_cond=obs_cond_K)
            noisy_action = self.noise_scheduler.step(
                model_output=noise_pred, timestep=t, sample=noisy_action,
            ).prev_sample

        return self._dp_unnorm_action(noisy_action)

    @torch.no_grad()
    def predict_foresight(self, actions_raw, marker_window):
        """Predict future tactile latent for K candidates."""
        K = actions_raw.shape[0]

        marker_win = marker_window.unsqueeze(0).to(self.device)
        z_cur_raw, _ = self.foresight.tactile_vae.encode_single_frame(marker_win)
        z_cur = z_cur_raw.reshape(1, -1).expand(K, -1)

        # Build foresight images (dummy - we only need marker for z_cur)
        # Actually we need real images for the foresight model
        # But for diagnostic purposes, we can use the foresight model's own encoding
        fs_chunk = self.foresight_chunk
        action_fs = actions_raw[:, :fs_chunk, :]
        action_fs_norm = self._fs_norm_action(action_fs)

        # Use z_cur as qpos proxy for foresight (simplified)
        # In full pipeline, we'd pass real images
        return z_cur, action_fs_norm

    @torch.no_grad()
    def score_with_phase_aware(self, z_preds, z_cur, progress):
        """Score candidates using PhaseAwareScorer."""
        phase = self.scorer.estimate_phase(progress)
        scores = self.scorer.score(z_preds, z_cur, phase, mode='fit')
        return scores, phase


# ==================== Diagnostic Runner ====================

def find_hdf5_files(data_dir):
    hdf5_files = []
    for subdir in ["success", "bounce", ""]:
        pattern_dir = os.path.join(data_dir, subdir) if subdir else data_dir
        if os.path.isdir(pattern_dir):
            for fn in os.listdir(pattern_dir):
                if fn.endswith(".hdf5"):
                    hdf5_files.append(os.path.join(pattern_dir, fn))
    return sorted(set(hdf5_files))


def run_diagnostic(diag, data_dir, n_eval=100, seed=42):
    """Run full pipeline diagnostic."""
    rng = np.random.RandomState(seed)
    K = diag.K

    hdf5_files = find_hdf5_files(data_dir)
    if not hdf5_files:
        print(f"No HDF5 files found in {data_dir}")
        return

    # Sample frames
    all_frames = []
    for hf in hdf5_files:
        with h5py.File(hf, 'r') as f:
            T = f['observations/proprio_joint'].shape[0]
            ep_name = os.path.splitext(os.path.basename(hf))[0]
            for t in range(2, T - diag.pred_horizon - 10, 5):
                all_frames.append((hf, ep_name, t))

    if len(all_frames) > n_eval:
        idx = rng.choice(len(all_frames), n_eval, replace=False)
        all_frames = [all_frames[i] for i in idx]

    print(f"\nDiagnostic: {len(all_frames)} frames, K={K}")
    print(f"  DP: {diag.dp_config.get('save_dir', '?')}")
    print(f"  Foresight: {diag.foresight_config.get('save_dir', '?')}")

    # Collect metrics
    pairwise_l1_actions = []
    pairwise_l1_zpred = []
    score_ranges = []
    score_means = []
    score_stds = []
    oracle_match_count = 0
    total_frames = 0
    all_scores = []
    all_l1_to_oracle = []
    phase_counts = {}

    for hdf5_path, ep_name, t in tqdm(all_frames, desc="Diagnostic"):
        try:
            with h5py.File(hdf5_path, 'r') as f:
                qpos_raw = f['observations/proprio_joint'][t].astype(np.float32)
                marker_all = f['observations/tac/left/marker_offset'][:]
                T = marker_all.shape[0]

                # Build obs_cond
                images_obs = []
                qpos_obs = []
                marker_hists = []
                for step in range(diag.obs_horizon):
                    t_obs = max(0, t - diag.obs_horizon + 1 + step)
                    obs_dict = {}
                    for cam in diag.dp_camera_names:
                        key = f'observations/images/{cam}'
                        if key in f:
                            obs_dict[cam] = f[key][t_obs]
                        else:
                            obs_dict[cam] = np.zeros((480, 640, 3), dtype=np.uint8)
                    images_obs.append(obs_dict)
                    qpos_obs.append(f['observations/proprio_joint'][t_obs].astype(np.float32))
                    marker_hists.append(diag._get_dp_marker_history(marker_all, t_obs))

                # Expert action
                action_start = t + diag.action_shift
                action_expert = f['actions/joint_abs'][action_start:action_start + diag.pred_horizon].astype(np.float32)
                if len(action_expert) < diag.pred_horizon:
                    action_expert = np.pad(action_expert,
                                           ((0, diag.pred_horizon - len(action_expert)), (0, 0)), mode='edge')

                # Marker window for foresight
                marker_window = diag._get_marker_window(marker_all, t)

                # Progress estimate
                progress = t / T

            # Generate K candidates
            obs_cond = diag.build_obs_cond(images_obs, qpos_obs, marker_hists)
            actions_raw = diag.generate_candidates(obs_cond, K=K).detach()
            actions_np = actions_raw.cpu().numpy()

            # Compute pairwise L1 between candidates
            pairwise_l1s = []
            for i in range(K):
                for j in range(i + 1, K):
                    l1 = np.abs(actions_np[i].flatten() - actions_np[j].flatten()).mean()
                    pairwise_l1s.append(l1)
            pairwise_l1_actions.append(np.mean(pairwise_l1s))

            # Predict foresight for each candidate
            marker_win = marker_window.unsqueeze(0).to(diag.device)
            z_cur_raw, _ = diag.foresight.tactile_vae.encode_single_frame(marker_win)
            z_cur = z_cur_raw.reshape(1, -1).expand(K, -1)

            fs_chunk = diag.foresight_chunk
            action_fs = actions_raw[:, :fs_chunk, :]
            action_fs_norm = diag._fs_norm_action(action_fs)
            qpos_fs_norm = diag._fs_norm_qpos(
                torch.tensor(qpos_raw, dtype=torch.float32, device=diag.device).unsqueeze(0)).expand(K, -1)

            # Build dummy foresight images (use current frame images)
            foresight_imgs = []
            for cam_name in ['global', 'wrist']:
                with h5py.File(hdf5_path, 'r') as f:
                    key = f'observations/images/{cam_name}'
                    if key in f:
                        img = diag._preprocess_image(f[key][t]).unsqueeze(0).to(diag.device)
                    else:
                        img = torch.zeros(1, 3, *diag.resize_shape, device=diag.device)
                foresight_imgs.append(img)
            marker_win_dev = marker_window.unsqueeze(0).to(diag.device)
            foresight_imgs.append(marker_win_dev)

            fs_images = [img.expand(K, *img.shape[1:]) for img in foresight_imgs]
            t_hat, _, _, _, _, _ = diag.foresight(
                fs_images, action_fs_norm, qpos=qpos_fs_norm)
            z_preds = t_hat.detach()

            # Pairwise L1 on z_pred
            z_pred_np = z_preds.cpu().numpy()
            pairwise_zpred = []
            for i in range(K):
                for j in range(i + 1, K):
                    l1 = np.abs(z_pred_np[i].flatten() - z_pred_np[j].flatten()).mean()
                    pairwise_zpred.append(l1)
            pairwise_l1_zpred.append(np.mean(pairwise_zpred))

            # PhaseAwareScorer scoring
            scores, phase = diag.score_with_phase_aware(z_preds, z_cur, progress)
            scores_np = scores.detach().cpu().numpy()

            score_ranges.append(scores_np.max() - scores_np.min())
            score_means.append(scores_np.mean())
            score_stds.append(scores_np.std())
            all_scores.extend(scores_np.tolist())

            # Oracle: pick candidate closest to expert
            expert_flat = action_expert.flatten()
            l1_to_expert = [np.abs(actions_np[i].flatten() - expert_flat).mean() for i in range(K)]
            oracle_idx = np.argmin(l1_to_expert)
            scorer_best_idx = np.argmax(scores_np)

            if scorer_best_idx == oracle_idx:
                oracle_match_count += 1
            total_frames += 1

            # Store for correlation
            for i in range(K):
                all_l1_to_oracle.append(l1_to_expert[i])

            phase_name = ['approach', 'insertion', 'pre_bounce', 'lift', 'reposition'][phase]
            phase_counts[phase_name] = phase_counts.get(phase_name, 0) + 1

        except Exception as e:
            import traceback
            print(f"Error on {ep_name} t={t}: {e}")
            traceback.print_exc()
            continue

    if total_frames == 0:
        print("No valid frames!")
        return

    # ========== Report ==========
    print(f"\n{'='*70}")
    print(f"TacScore Pipeline Diagnostic Results")
    print(f"{'='*70}")
    print(f"Frames evaluated: {total_frames}")
    print(f"Phase distribution: {phase_counts}")

    print(f"\n--- Candidate Diversity ---")
    print(f"  Action pairwise L1:  {np.mean(pairwise_l1_actions):.4f} +/- {np.std(pairwise_l1_actions):.4f}")
    print(f"  z_pred pairwise L1:  {np.mean(pairwise_l1_zpred):.4f} +/- {np.std(pairwise_l1_zpred):.4f}")
    action_range = np.mean([np.abs(actions_np[i].flatten()).mean() for i in range(K)]) if 'actions_np' in dir() else 0
    print(f"  Action scale (mean |a|): ~[-1, 1] normalized space")

    print(f"\n--- PhaseAwareScorer Score Distribution ---")
    all_scores_arr = np.array(all_scores)
    print(f"  Mean:   {all_scores_arr.mean():.4f} +/- {all_scores_arr.std():.4f}")
    print(f"  Range:  [{all_scores_arr.min():.4f}, {all_scores_arr.max():.4f}]")
    print(f"  Per-frame range: {np.mean(score_ranges):.4f} +/- {np.std(score_ranges):.4f}")
    print(f"  Per-frame std:   {np.mean(score_stds):.4f} +/- {np.std(score_stds):.4f}")

    print(f"\n--- Oracle Match ---")
    print(f"  Oracle Match: {oracle_match_count}/{total_frames} ({oracle_match_count/total_frames*100:.1f}%)")
    print(f"  Random baseline: {1/K*100:.1f}%")

    # Score-L1 correlation
    if len(all_scores) > 0 and len(all_l1_to_oracle) > 0:
        corr = np.corrcoef(np.array(all_scores), -np.array(all_l1_to_oracle))[0, 1]
        print(f"\n--- Score-L1 Correlation ---")
        print(f"  Pearson r (score vs -L1): {corr:.4f}")
        print(f"  (>0 means higher score → lower L1 to expert)")

    # Per-phase breakdown
    print(f"\n--- Per-Phase Score Range ---")
    # This would need per-phase tracking, simplified here
    print(f"  (See phase distribution above)")

    print(f"\n{'='*70}")
    print(f"Key Takeaways:")
    print(f"  1. Action diversity (pairwise L1): {'LOW' if np.mean(pairwise_l1_actions) < 0.02 else 'OK'}")
    print(f"  2. z_pred diversity: {'LOW' if np.mean(pairwise_l1_zpred) < 0.01 else 'OK'}")
    print(f"  3. Score range: {'SATURATED' if np.mean(score_ranges) < 0.1 else 'OK'}")
    print(f"  4. Oracle Match: {'GOOD' if oracle_match_count/total_frames > 0.6 else 'POOR'}")
    print(f"{'='*70}")

    return {
        'pairwise_l1_actions': np.mean(pairwise_l1_actions),
        'pairwise_l1_zpred': np.mean(pairwise_l1_zpred),
        'score_range': np.mean(score_ranges),
        'score_mean': all_scores_arr.mean(),
        'oracle_match': oracle_match_count / total_frames,
        'n_frames': total_frames,
    }


def main():
    parser = argparse.ArgumentParser(description="TacScore Pipeline Diagnostic")
    parser.add_argument("--dp_ckpt", type=str,
                        default="/home/chenshuai/Project/output/ckpt/dp_foresight_joint_0209/dp_topk_ep130_loss0.0029.pth")
    parser.add_argument("--dp_config", type=str,
                        default="/home/chenshuai/Project/output/ckpt/dp_foresight_joint_0209/config.json")
    parser.add_argument("--foresight_ckpt", type=str,
                        default="/home/chenshuai/Project/output/foresight_ckpt/latent_foresight_full/foresight_best.ckpt")
    parser.add_argument("--foresight_dir", type=str,
                        default="/home/chenshuai/Project/output/foresight_ckpt/latent_foresight_full")
    parser.add_argument("--phase_stats", type=str,
                        default="/home/chenshuai/Project/output/phase_scoring_stats.pkl")
    parser.add_argument("--data_dir", type=str,
                        default="/home/chenshuai/data/dataset/0209-0210/truncated")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--n_eval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    diag = PipelineDiagnostic(
        dp_config_path=args.dp_config,
        dp_ckpt_path=args.dp_ckpt,
        foresight_ckpt_path=args.foresight_ckpt,
        foresight_dir=args.foresight_dir,
        phase_stats_path=args.phase_stats,
        device=args.device,
        K=args.K,
    )

    results = run_diagnostic(diag, args.data_dir, n_eval=args.n_eval, seed=args.seed)


if __name__ == "__main__":
    main()
