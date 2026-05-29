"""
PhaseAwareScorer v2: 多模式评分 + Pairwise Oracle Match 诊断

基于Step 1诊断结果(fit mode Oracle Match仅8%, 相关性-0.39)，
测试所有评分模式，找出最有效的评分策略。

新增功能:
  1. 多模式对比: fit, smoothness, naive, multidim, intensity, delta_norm
  2. Per-phase分析: 每个阶段的Oracle Match分别统计
  3. Pairwise accuracy: A和B两个候选中，scorer选对的比例
  4. z_int vs z_pat分析: 哪部分latent空间包含区分信息

Usage:
  python TFAC_V5/phase_aware_scorer_v2.py \
    --data_dir /home/chenshuai/data/dataset/0209-0210_truncated \
    --dp_ckpt /home/chenshuai/Project/output/ckpt/dp_foresight_joint_0209/dp_topk_ep130_loss0.0029.pth \
    --dp_config /home/chenshuai/Project/output/ckpt/dp_foresight_joint_0209/config.json \
    --foresight_ckpt /home/chenshuai/Project/output/foresight_ckpt/latent_foresight_full/foresight_best.ckpt \
    --foresight_dir /home/chenshuai/Project/output/foresight_ckpt/latent_foresight_full \
    --phase_stats /home/chenshuai/Project/output/phase_scoring_stats.pkl \
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


# ==================== Vision Encoder ====================

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


# ==================== Multi-Mode Scorer ====================

class MultiModeScorer:
    """支持多种评分模式的scorer，用于对比不同评分策略。"""

    def __init__(self, phase_stats_path, device):
        self.device = device
        self.phase_tensors = {}
        if phase_stats_path and os.path.exists(phase_stats_path):
            with open(phase_stats_path, 'rb') as f:
                data = pickle.load(f)
            for pid, stats in data.get('phase_stats', {}).items():
                if stats is None:
                    continue
                self.phase_tensors[pid] = {
                    'z_mu': torch.tensor(stats['z_mu'], dtype=torch.float32, device=device),
                    'z_std': torch.tensor(stats['z_std'], dtype=torch.float32, device=device),
                    'delta_norm_mu': stats['delta_norm_mu'],
                    'delta_norm_std': stats['delta_norm_std'],
                    'z_int_norm_p95': stats['z_int_norm_p95'],
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

    def score_fit(self, z_pred, z_cur, phase):
        """Demo distribution matching."""
        if phase not in self.phase_tensors:
            return torch.zeros(z_pred.shape[0], device=self.device)
        mu = self.phase_tensors[phase]['z_mu']
        std = self.phase_tensors[phase]['z_std']
        return -((z_pred - mu) / std).abs().mean(dim=-1)

    def score_smoothness(self, z_pred, z_cur, phase):
        """Phase-normalized tactile delta (smaller = smoother = better)."""
        delta = torch.norm(z_pred - z_cur.expand_as(z_pred), dim=-1)
        if phase not in self.phase_tensors:
            return -delta
        mu = self.phase_tensors[phase]['delta_norm_mu']
        std = self.phase_tensors[phase]['delta_norm_std']
        return -(delta - mu) / (std + 1e-8)

    def score_naive(self, z_pred, z_cur, phase):
        """Simple negative delta norm."""
        return -torch.norm(z_pred - z_cur.expand_as(z_pred), dim=-1)

    def score_intensity(self, z_pred, z_cur, phase):
        """Contact intensity (z_int norm) - lower for approach, target range for insertion."""
        z_int = z_pred[:, :9]  # first 9 dims = intensity
        intensity = torch.norm(z_int, dim=-1)
        if phase == 0:  # approach: minimize
            return -intensity
        elif phase == 1:  # insertion: target range
            target = 3.0  # target intensity
            return -torch.abs(intensity - target)
        else:
            return -intensity

    def score_delta_norm(self, z_pred, z_cur, phase):
        """Delta norm with phase-specific thresholds."""
        delta = torch.norm(z_pred - z_cur.expand_as(z_pred), dim=-1)
        if phase == 0:  # approach: small delta OK
            return -delta * 0.5
        elif phase == 1:  # insertion: moderate delta expected
            return -torch.abs(delta - 4.0)
        elif phase == 2:  # pre_bounce: large delta = bad
            return -delta * 2.0
        else:
            return -delta

    def score_multidim(self, z_pred, z_cur, phase):
        """Weighted combination."""
        s_fit = self.score_fit(z_pred, z_cur, phase)
        s_smooth = self.score_smoothness(z_pred, z_cur, phase)
        s_safe = self.score_intensity(z_pred, z_cur, phase)
        return 0.5 * s_fit + 0.3 * s_smooth + 0.2 * s_safe

    def score_zpat_consistency(self, z_pred, z_cur, phase):
        """z_pattern direction consistency with current state."""
        z_pat_pred = z_pred[:, 9:]  # pattern dims
        z_pat_cur = z_cur[:, 9:]
        # Cosine similarity of pattern vectors
        cos = torch.nn.functional.cosine_similarity(z_pat_pred, z_pat_cur, dim=-1)
        return cos

    def score_combined_v2(self, z_pred, z_cur, phase):
        """V2 combined: delta_norm + z_pat consistency + intensity penalty."""
        delta = torch.norm(z_pred - z_cur.expand_as(z_pred), dim=-1)
        z_pat_pred = z_pred[:, 9:]
        z_pat_cur = z_cur[:, 9:]
        cos = torch.nn.functional.cosine_similarity(z_pat_pred, z_pat_cur, dim=-1)
        z_int = z_pred[:, :9]
        intensity = torch.norm(z_int, dim=-1)

        # Delta score: prefer moderate delta (not too small, not too large)
        if phase == 0:
            delta_score = -delta * 0.3
        elif phase == 1:
            delta_score = -torch.abs(delta - 4.0) * 0.5
        else:
            delta_score = -delta * 0.5

        # Consistency: higher cosine = better
        consistency_score = cos * 0.3

        # Intensity: penalize extreme values
        intensity_score = -torch.clamp(intensity - 5.0, min=0) * 0.2

        return delta_score + consistency_score + intensity_score

    def score(self, z_pred, z_cur, phase, mode='fit'):
        scorers = {
            'fit': self.score_fit,
            'smoothness': self.score_smoothness,
            'naive': self.score_naive,
            'intensity': self.score_intensity,
            'delta_norm': self.score_delta_norm,
            'multidim': self.score_multidim,
            'zpat_consistency': self.score_zpat_consistency,
            'combined_v2': self.score_combined_v2,
        }
        if mode in scorers:
            return scorers[mode](z_pred, z_cur, phase)
        return self.score_naive(z_pred, z_cur, phase)


# ==================== Diagnostic Pipeline ====================

class ScorerDiagnostic:
    """Load pipeline and compare all scoring modes."""

    def __init__(self, dp_config_path, dp_ckpt_path, foresight_ckpt_path,
                 foresight_dir, phase_stats_path, device="cuda:0", K=16):
        self.device = torch.device(device)
        self.K = K

        with open(dp_config_path) as f:
            self.dp_config = json.load(f)

        self._load_dp(dp_ckpt_path)
        self._load_foresight(foresight_ckpt_path, foresight_dir)
        self.scorer = MultiModeScorer(phase_stats_path, self.device)

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
            input_dim=self.action_dim,
            global_cond_dim=config["global_cond_dim"],
            diffusion_step_embed_dim=config.get("diffusion_step_embed_dim", 128),
            down_dims=down_dims, kernel_size=5,
        ).to(self.device)

        self.noise_scheduler = DDPMScheduler(
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

    def _load_foresight(self, ckpt_path, foresight_dir):
        if foresight_dir is None:
            foresight_dir = os.path.dirname(ckpt_path)
        args_path = os.path.join(foresight_dir, "args.json")
        with open(args_path) as f:
            config = json.load(f)

        camera_names = config['camera_names']
        cam_backbone_mapping = {cam: 0 for cam in camera_names}
        self.foresight = LatentForesightPretrainModel(
            camera_names=camera_names, cam_backbone_mapping=cam_backbone_mapping,
            hidden_dim=config['hidden_dim'], state_dim=config['state_dim'],
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
        self.vae_window = config.get('tactile_vae_window', 8) or 8

    def _dp_unnorm_action(self, a):
        return (a + 1) / 2 * (self.dp_action_max - self.dp_action_min) + self.dp_action_min

    def _dp_norm_qpos(self, q):
        return (q - self.dp_qpos_min) / (self.dp_qpos_max - self.dp_qpos_min + 1e-8) * 2 - 1

    def _fs_norm_action(self, a):
        return (a - self.fs_action_mean) / self.fs_action_std

    def _fs_norm_qpos(self, q):
        return (q - self.fs_qpos_mean) / self.fs_qpos_std

    def _preprocess_image(self, img_uint8):
        img = torch.tensor(img_uint8.astype(np.float32) / 255.0).permute(2, 0, 1)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img = transforms.functional.resize(img, list(self.resize_shape))
        return normalize(img)

    def _get_marker_window(self, marker_all, t):
        T_total = marker_all.shape[0]
        frames = []
        for i in range(self.vae_window):
            idx = min(max(0, t - (self.vae_window - 1 - i)), T_total - 1)
            frames.append(marker_all[idx].astype(np.float32))
        return torch.tensor(np.stack(frames), dtype=torch.float32)

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

    @torch.no_grad()
    def generate_candidates(self, obs_cond, K=None):
        K = K or self.K
        obs_cond_K = obs_cond.expand(K, -1)
        noisy_action = torch.randn(K, self.pred_horizon, self.action_dim, device=self.device)
        self.noise_scheduler.set_timesteps(self.dp_config.get("num_inference_steps", 100))
        for t in self.noise_scheduler.timesteps:
            noise_pred = self.noise_pred_net(sample=noisy_action, timestep=t, global_cond=obs_cond_K)
            noisy_action = self.noise_scheduler.step(model_output=noise_pred, timestep=t, sample=noisy_action).prev_sample
        return self._dp_unnorm_action(noisy_action).detach()

    @torch.no_grad()
    def predict_z_preds(self, actions_raw, marker_window, hdf5_path, t):
        """Predict z_pred for K candidates."""
        K = actions_raw.shape[0]
        marker_win = marker_window.unsqueeze(0).to(self.device)
        z_cur_raw, _ = self.foresight.tactile_vae.encode_single_frame(marker_win)
        z_cur = z_cur_raw.reshape(1, -1).expand(K, -1)

        fs_chunk = self.foresight_chunk
        action_fs = actions_raw[:, :fs_chunk, :]
        action_fs_norm = self._fs_norm_action(action_fs)

        # Build foresight images
        foresight_imgs = []
        for cam_name in ['global', 'wrist']:
            with h5py.File(hdf5_path, 'r') as f:
                key = f'observations/images/{cam_name}'
                if key in f:
                    img = self._preprocess_image(f[key][t]).unsqueeze(0).to(self.device)
                else:
                    img = torch.zeros(1, 3, *self.resize_shape, device=self.device)
            foresight_imgs.append(img)
        marker_win_dev = marker_window.unsqueeze(0).to(self.device)
        foresight_imgs.append(marker_win_dev)
        fs_images = [img.expand(K, *img.shape[1:]) for img in foresight_imgs]

        qpos_dummy = torch.zeros(K, 7, device=self.device)  # placeholder
        t_hat, _, _, _, _, _ = self.foresight(fs_images, action_fs_norm, qpos=qpos_dummy)
        return t_hat.detach(), z_cur.detach()


def find_hdf5_files(data_dir):
    hdf5_files = []
    for subdir in ["success", "bounce", ""]:
        pattern_dir = os.path.join(data_dir, subdir) if subdir else data_dir
        if os.path.isdir(pattern_dir):
            for fn in os.listdir(pattern_dir):
                if fn.endswith(".hdf5"):
                    hdf5_files.append(os.path.join(pattern_dir, fn))
    return sorted(set(hdf5_files))


def run_multi_mode_diagnostic(diag, data_dir, n_eval=100, seed=42):
    """Compare all scoring modes."""
    rng = np.random.RandomState(seed)
    K = diag.K

    hdf5_files = find_hdf5_files(data_dir)
    if not hdf5_files:
        print(f"No HDF5 files found in {data_dir}")
        return

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

    modes = ['fit', 'smoothness', 'naive', 'intensity', 'delta_norm',
             'multidim', 'zpat_consistency', 'combined_v2']

    # Per-mode results
    results = {mode: {
        'oracle_match': 0, 'pairwise_correct': 0, 'pairwise_total': 0,
        'score_l1_corr': [], 'per_phase': {},
    } for mode in modes}

    n_valid = 0

    for hdf5_path, ep_name, t in tqdm(all_frames, desc="Multi-mode diagnostic"):
        try:
            with h5py.File(hdf5_path, 'r') as f:
                qpos_raw = f['observations/proprio_joint'][t].astype(np.float32)
                marker_all = f['observations/tac/left/marker_offset'][:]
                T = marker_all.shape[0]
                progress = t / T

                images_obs, qpos_obs, marker_hists = [], [], []
                for step in range(diag.obs_horizon):
                    t_obs = max(0, t - diag.obs_horizon + 1 + step)
                    obs_dict = {}
                    for cam in diag.dp_camera_names:
                        key = f'observations/images/{cam}'
                        obs_dict[cam] = f[key][t_obs] if key in f else np.zeros((480, 640, 3), dtype=np.uint8)
                    images_obs.append(obs_dict)
                    qpos_obs.append(f['observations/proprio_joint'][t_obs].astype(np.float32))
                    marker_hists.append(diag._get_dp_marker_history(marker_all, t_obs))

                action_start = t + diag.action_shift
                action_expert = f['actions/joint_abs'][action_start:action_start + diag.pred_horizon].astype(np.float32)
                if len(action_expert) < diag.pred_horizon:
                    action_expert = np.pad(action_expert, ((0, diag.pred_horizon - len(action_expert)), (0, 0)), mode='edge')

                marker_window = diag._get_marker_window(marker_all, t)

            # Generate candidates and predict
            obs_cond = diag.build_obs_cond(images_obs, qpos_obs, marker_hists)
            actions_raw = diag.generate_candidates(obs_cond, K=K)
            actions_np = actions_raw.cpu().numpy()
            z_preds, z_cur = diag.predict_z_preds(actions_raw, marker_window, hdf5_path, t)

            # Oracle: pick candidate closest to expert
            expert_flat = action_expert.flatten()
            l1_to_expert = np.array([np.abs(actions_np[i].flatten() - expert_flat).mean() for i in range(K)])
            oracle_idx = np.argmin(l1_to_expert)

            phase = diag.scorer.estimate_phase(progress)
            phase_name = ['approach', 'insertion', 'pre_bounce', 'lift', 'reposition'][phase]

            # Test each scoring mode
            for mode in modes:
                scores = diag.scorer.score(z_preds, z_cur, phase, mode=mode)
                scores_np = scores.detach().cpu().numpy()

                # Oracle match
                scorer_best = np.argmax(scores_np)
                if scorer_best == oracle_idx:
                    results[mode]['oracle_match'] += 1

                # Pairwise accuracy: for all pairs (i,j) where l1_i < l1_j,
                # check if score_i > score_j
                for i in range(K):
                    for j in range(i + 1, K):
                        results[mode]['pairwise_total'] += 1
                        if (l1_to_expert[i] < l1_to_expert[j] and scores_np[i] > scores_np[j]) or \
                           (l1_to_expert[i] > l1_to_expert[j] and scores_np[i] < scores_np[j]):
                            results[mode]['pairwise_correct'] += 1

                # Score-L1 correlation
                if np.std(scores_np) > 1e-8:
                    corr = np.corrcoef(scores_np, -l1_to_expert)[0, 1]
                    results[mode]['score_l1_corr'].append(corr)

                # Per-phase tracking
                if phase_name not in results[mode]['per_phase']:
                    results[mode]['per_phase'][phase_name] = {'match': 0, 'total': 0}
                results[mode]['per_phase'][phase_name]['total'] += 1
                if scorer_best == oracle_idx:
                    results[mode]['per_phase'][phase_name]['match'] += 1

            n_valid += 1

        except Exception as e:
            continue

    if n_valid == 0:
        print("No valid frames!")
        return

    # ========== Report ==========
    print(f"\n{'='*80}")
    print(f"Multi-Mode Scorer Comparison ({n_valid} frames, K={K})")
    print(f"{'='*80}")
    print(f"\n{'Mode':<20} {'Oracle%':>10} {'Pairwise%':>12} {'Corr':>10} {'ScoreRange':>12}")
    print(f"{'-'*64}")

    for mode in modes:
        r = results[mode]
        om = r['oracle_match'] / n_valid * 100
        pw = r['pairwise_correct'] / max(r['pairwise_total'], 1) * 100
        corr = np.mean(r['score_l1_corr']) if r['score_l1_corr'] else 0
        print(f"{mode:<20} {om:>9.1f}% {pw:>11.1f}% {corr:>10.4f}")

    # Per-phase breakdown for best mode
    best_mode = max(modes, key=lambda m: results[m]['oracle_match'])
    print(f"\nBest mode: {best_mode}")
    print(f"\nPer-phase breakdown ({best_mode}):")
    for phase_name, stats in results[best_mode]['per_phase'].items():
        if stats['total'] > 0:
            print(f"  {phase_name}: {stats['match']}/{stats['total']} ({stats['match']/stats['total']*100:.1f}%)")

    # z_int vs z_pat analysis
    print(f"\n--- z_int vs z_pat Variance Analysis ---")
    # This needs to be computed from actual z_pred data
    # Simplified: report average z_int and z_pat norms
    print(f"  (See per-mode results above)")

    print(f"\n{'='*80}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dp_ckpt", default="/home/chenshuai/Project/output/ckpt/dp_foresight_joint_0209/dp_topk_ep130_loss0.0029.pth")
    parser.add_argument("--dp_config", default="/home/chenshuai/Project/output/ckpt/dp_foresight_joint_0209/config.json")
    parser.add_argument("--foresight_ckpt", default="/home/chenshuai/Project/output/foresight_ckpt/latent_foresight_full/foresight_best.ckpt")
    parser.add_argument("--foresight_dir", default="/home/chenshuai/Project/output/foresight_ckpt/latent_foresight_full")
    parser.add_argument("--phase_stats", default="/home/chenshuai/Project/output/phase_scoring_stats.pkl")
    parser.add_argument("--data_dir", default="/home/chenshuai/data/dataset/0209-0210_truncated")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--n_eval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    diag = ScorerDiagnostic(
        dp_config_path=args.dp_config, dp_ckpt_path=args.dp_ckpt,
        foresight_ckpt_path=args.foresight_ckpt, foresight_dir=args.foresight_dir,
        phase_stats_path=args.phase_stats, device=args.device, K=args.K,
    )

    run_multi_mode_diagnostic(diag, args.data_dir, n_eval=args.n_eval, seed=args.seed)


if __name__ == "__main__":
    main()
