"""
Pairwise Scorer: 用DP候选数据训练pairwise对比scorer

基于Step 2诊断结果(所有评分模式≈随机)，训练一个learned scorer:
  输入: (candidate_A_features, candidate_B_features)
  输出: P(A better than B)

训练数据: DP生成K=16候选, 用L1-to-expert做label
特征: z_pred + action_chunk + qpos

同时测试ensemble baseline (平均所有候选)

Usage:
  python TFAC_V5/train_pairwise_scorer.py \
    --data_dir /home/chenshuai/data/dataset/0209-0210_truncated \
    --dp_ckpt /home/chenshuai/Project/output/ckpt/dp_foresight_joint_0209/dp_topk_ep130_loss0.0029.pth \
    --dp_config /home/chenshuai/Project/output/ckpt/dp_foresight_joint_0209/config.json \
    --foresight_ckpt /home/chenshuai/Project/output/foresight_ckpt/latent_foresight_full/foresight_best.ckpt \
    --foresight_dir /home/chenshuai/Project/output/foresight_ckpt/latent_foresight_full \
    --phase_stats /home/chenshuai/Project/output/phase_scoring_stats.pkl \
    --n_gen 500 --K 16 --epochs 30
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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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


# ==================== Reuse from phase_aware_scorer_v2.py ====================
# (Vision encoder, TactileVAE, EMA, pipeline classes - same as before)

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


# ==================== Pairwise Scorer Model ====================

class PairwiseScorer(nn.Module):
    """Pairwise comparison scorer.
    Input: features of candidate A and B
    Output: P(A better than B) via sigmoid(score_A - score_B)
    """

    def __init__(self, feat_dim=156, hidden=128):
        super().__init__()
        # Shared feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, 1),  # scalar score
        )

    def forward(self, feat_a, feat_b):
        """Returns probability that A is better than B."""
        score_a = self.encoder(feat_a).squeeze(-1)
        score_b = self.encoder(feat_b).squeeze(-1)
        return score_a, score_b, torch.sigmoid(score_a - score_b)

    def predict_score(self, feat):
        """Pointwise score for ranking."""
        return self.encoder(feat).squeeze(-1)


# ==================== Pairwise Dataset ====================

class PairwiseDataset(Dataset):
    """Generate (A, B, label) pairs from DP candidate groups.
    label = 1 if A is closer to expert than B, else 0.
    """

    def __init__(self, data_path):
        data = torch.load(data_path)
        self.groups = data['groups']  # list of {z_preds, actions, l1_to_expert, qpos}

        # Generate all pairs
        self.pairs = []
        for g_idx, g in enumerate(self.groups):
            K = len(g['l1_to_expert'])
            for i in range(K):
                for j in range(i + 1, K):
                    label = 1.0 if g['l1_to_expert'][i] < g['l1_to_expert'][j] else 0.0
                    self.pairs.append((g_idx, i, j, label))

        print(f"PairwiseDataset: {len(self.groups)} groups, {len(self.pairs)} pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        g_idx, i, j, label = self.pairs[idx]
        g = self.groups[g_idx]

        feat_a = np.concatenate([g['z_preds'][i], g['actions'][i].flatten()[:12]])  # 144+12=156
        feat_b = np.concatenate([g['z_preds'][j], g['actions'][j].flatten()[:12]])

        return {
            'feat_a': torch.tensor(feat_a, dtype=torch.float32),
            'feat_b': torch.tensor(feat_b, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.float32),
        }


# ==================== Data Generation ====================

class DataGenerator:
    """Generate DP candidate groups with foresight predictions and L1 labels."""

    def __init__(self, dp_config_path, dp_ckpt_path, foresight_ckpt_path,
                 foresight_dir, device="cuda:0", K=16):
        self.device = torch.device(device)
        self.K = K

        with open(dp_config_path) as f:
            self.dp_config = json.load(f)

        self._load_dp(dp_ckpt_path)
        self._load_foresight(foresight_ckpt_path, foresight_dir)

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
            dropout=config.get('dropout', 0.1), tactile_mode=config.get('tactile_mode', 'marker'),
            max_history=config.get('max_history', 8), predict_horizon=config.get('predict_horizon', 1),
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
        K = actions_raw.shape[0]
        marker_win = marker_window.unsqueeze(0).to(self.device)
        z_cur_raw, _ = self.foresight.tactile_vae.encode_single_frame(marker_win)
        z_cur = z_cur_raw.reshape(1, -1).expand(K, -1)

        fs_chunk = self.foresight_chunk
        action_fs = actions_raw[:, :fs_chunk, :]
        action_fs_norm = self._fs_norm_action(action_fs)

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

        qpos_dummy = torch.zeros(K, 7, device=self.device)
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


# ==================== Generate Training Data ====================

def generate_pairwise_data(gen, data_dir, n_gen=500, save_path=None):
    """Generate DP candidate groups for pairwise scorer training."""
    rng = np.random.RandomState(42)
    hdf5_files = find_hdf5_files(data_dir)

    all_frames = []
    for hf in hdf5_files:
        with h5py.File(hf, 'r') as f:
            T = f['observations/proprio_joint'].shape[0]
            ep_name = os.path.splitext(os.path.basename(hf))[0]
            for t in range(2, T - gen.pred_horizon - 10, 5):
                all_frames.append((hf, ep_name, t))

    if len(all_frames) > n_gen:
        idx = rng.choice(len(all_frames), n_gen, replace=False)
        all_frames = [all_frames[i] for i in idx]

    groups = []
    for hdf5_path, ep_name, t in tqdm(all_frames, desc="Generating data"):
        try:
            with h5py.File(hdf5_path, 'r') as f:
                qpos_raw = f['observations/proprio_joint'][t].astype(np.float32)
                marker_all = f['observations/tac/left/marker_offset'][:]

                images_obs, qpos_obs, marker_hists = [], [], []
                for step in range(gen.obs_horizon):
                    t_obs = max(0, t - gen.obs_horizon + 1 + step)
                    obs_dict = {}
                    for cam in gen.dp_camera_names:
                        key = f'observations/images/{cam}'
                        obs_dict[cam] = f[key][t_obs] if key in f else np.zeros((480, 640, 3), dtype=np.uint8)
                    images_obs.append(obs_dict)
                    qpos_obs.append(f['observations/proprio_joint'][t_obs].astype(np.float32))
                    marker_hists.append(gen._get_dp_marker_history(marker_all, t_obs))

                action_start = t + gen.action_shift
                action_expert = f['actions/joint_abs'][action_start:action_start + gen.pred_horizon].astype(np.float32)
                if len(action_expert) < gen.pred_horizon:
                    action_expert = np.pad(action_expert, ((0, gen.pred_horizon - len(action_expert)), (0, 0)), mode='edge')

                marker_window = gen._get_marker_window(marker_all, t)

            obs_cond = gen.build_obs_cond(images_obs, qpos_obs, marker_hists)
            actions_raw = gen.generate_candidates(obs_cond, K=gen.K)
            actions_np = actions_raw.cpu().numpy()
            z_preds, z_cur = gen.predict_z_preds(actions_raw, marker_window, hdf5_path, t)
            z_preds_np = z_preds.cpu().numpy()

            expert_flat = action_expert.flatten()
            l1_to_expert = [np.abs(actions_np[i].flatten() - expert_flat).mean() for i in range(gen.K)]

            groups.append({
                'z_preds': z_preds_np,
                'actions': actions_np,
                'l1_to_expert': l1_to_expert,
                'qpos': qpos_raw,
            })
        except Exception as e:
            continue

    if save_path:
        torch.save({'groups': groups}, save_path)
        print(f"Saved {len(groups)} groups to {save_path}")

    return groups


# ==================== Training ====================

def train_pairwise_scorer(data_path, epochs=30, lr=1e-3, batch_size=256, device="cuda:0"):
    """Train pairwise scorer."""
    dataset = PairwiseDataset(data_path)

    # Split 80/20
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    model = PairwiseScorer(feat_dim=156, hidden=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for batch in train_loader:
            feat_a = batch['feat_a'].to(device)
            feat_b = batch['feat_b'].to(device)
            label = batch['label'].to(device)

            score_a, score_b, prob = model(feat_a, feat_b)
            loss = F.binary_cross_entropy(prob, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(label)
            pred = (prob > 0.5).float()
            train_correct += (pred == label).sum().item()
            train_total += len(label)

        scheduler.step()

        # Validate
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                feat_a = batch['feat_a'].to(device)
                feat_b = batch['feat_b'].to(device)
                label = batch['label'].to(device)

                _, _, prob = model(feat_a, feat_b)
                pred = (prob > 0.5).float()
                val_correct += (pred == label).sum().item()
                val_total += len(label)

        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(os.path.dirname(data_path), 'pairwise_scorer_best.pt'))

        print(f"Epoch {epoch+1}/{epochs}: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, best={best_val_acc:.4f}")

    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    return model


# ==================== Evaluation ====================

def evaluate_pairwise_scorer(model, gen, data_dir, n_eval=100, device="cuda:0"):
    """Evaluate pairwise scorer vs ensemble baseline."""
    rng = np.random.RandomState(42)
    K = gen.K

    hdf5_files = find_hdf5_files(data_dir)
    all_frames = []
    for hf in hdf5_files:
        with h5py.File(hf, 'r') as f:
            T = f['observations/proprio_joint'].shape[0]
            ep_name = os.path.splitext(os.path.basename(hf))[0]
            for t in range(2, T - gen.pred_horizon - 10, 5):
                all_frames.append((hf, ep_name, t))

    if len(all_frames) > n_eval:
        idx = rng.choice(len(all_frames), n_eval, replace=False)
        all_frames = [all_frames[i] for i in idx]

    model.eval()
    l1_pairwise, l1_ensemble, l1_random, l1_oracle = [], [], [], []

    for hdf5_path, ep_name, t in tqdm(all_frames, desc="Evaluating"):
        try:
            with h5py.File(hdf5_path, 'r') as f:
                qpos_raw = f['observations/proprio_joint'][t].astype(np.float32)
                marker_all = f['observations/tac/left/marker_offset'][:]

                images_obs, qpos_obs, marker_hists = [], [], []
                for step in range(gen.obs_horizon):
                    t_obs = max(0, t - gen.obs_horizon + 1 + step)
                    obs_dict = {}
                    for cam in gen.dp_camera_names:
                        key = f'observations/images/{cam}'
                        obs_dict[cam] = f[key][t_obs] if key in f else np.zeros((480, 640, 3), dtype=np.uint8)
                    images_obs.append(obs_dict)
                    qpos_obs.append(f['observations/proprio_joint'][t_obs].astype(np.float32))
                    marker_hists.append(gen._get_dp_marker_history(marker_all, t_obs))

                action_start = t + gen.action_shift
                action_expert = f['actions/joint_abs'][action_start:action_start + gen.pred_horizon].astype(np.float32)
                if len(action_expert) < gen.pred_horizon:
                    action_expert = np.pad(action_expert, ((0, gen.pred_horizon - len(action_expert)), (0, 0)), mode='edge')

                marker_window = gen._get_marker_window(marker_all, t)

            obs_cond = gen.build_obs_cond(images_obs, qpos_obs, marker_hists)
            actions_raw = gen.generate_candidates(obs_cond, K=K)
            actions_np = actions_raw.cpu().numpy()
            z_preds, z_cur = gen.predict_z_preds(actions_raw, marker_window, hdf5_path, t)
            z_preds_np = z_preds.cpu().numpy()

            expert_flat = action_expert.flatten()
            l1_to_expert = np.array([np.abs(actions_np[i].flatten() - expert_flat).mean() for i in range(K)])

            # Pairwise scorer ranking
            feats = []
            for i in range(K):
                feat = np.concatenate([z_preds_np[i], actions_np[i].flatten()[:12]])
                feats.append(torch.tensor(feat, dtype=torch.float32))
            feats = torch.stack(feats).to(device)

            with torch.no_grad():
                scores = model.predict_score(feats).cpu().numpy()

            best_idx = np.argmax(scores)
            l1_pairwise.append(l1_to_expert[best_idx])

            # Ensemble: average all candidates
            ensemble_action = actions_np.mean(axis=0)
            l1_ensemble.append(np.abs(ensemble_action.flatten() - expert_flat).mean())

            # Random
            random_idx = rng.randint(K)
            l1_random.append(l1_to_expert[random_idx])

            # Oracle
            l1_oracle.append(l1_to_expert.min())

        except Exception as e:
            continue

    n = len(l1_pairwise)
    print(f"\n{'='*60}")
    print(f"Evaluation Results ({n} frames, K={K})")
    print(f"{'='*60}")
    print(f"  Oracle (best candidate): {np.mean(l1_oracle):.4f}")
    print(f"  Pairwise Scorer:         {np.mean(l1_pairwise):.4f}")
    print(f"  Ensemble (average):      {np.mean(l1_ensemble):.4f}")
    print(f"  Random:                  {np.mean(l1_random):.4f}")

    impr_vs_random = (np.mean(l1_random) - np.mean(l1_pairwise)) / np.mean(l1_random) * 100
    impr_vs_ensemble = (np.mean(l1_ensemble) - np.mean(l1_pairwise)) / np.mean(l1_ensemble) * 100
    print(f"\n  Pairwise vs Random:    {impr_vs_random:+.1f}%")
    print(f"  Pairwise vs Ensemble:  {impr_vs_ensemble:+.1f}%")
    print(f"  Pairwise wins rate:    {sum(1 for i in range(n) if l1_pairwise[i] < l1_random[i])/n*100:.1f}%")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dp_ckpt", default="/home/chenshuai/Project/output/ckpt/dp_foresight_joint_0209/dp_topk_ep130_loss0.0029.pth")
    parser.add_argument("--dp_config", default="/home/chenshuai/Project/output/ckpt/dp_foresight_joint_0209/config.json")
    parser.add_argument("--foresight_ckpt", default="/home/chenshuai/Project/output/foresight_ckpt/latent_foresight_full/foresight_best.ckpt")
    parser.add_argument("--foresight_dir", default="/home/chenshuai/Project/output/foresight_ckpt/latent_foresight_full")
    parser.add_argument("--data_dir", default="/home/chenshuai/data/dataset/0209-0210_truncated")
    parser.add_argument("--save_dir", default="/home/chenshuai/Project/output/pairwise_scorer")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--n_gen", type=int, default=500)
    parser.add_argument("--n_eval", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    data_path = os.path.join(args.save_dir, 'pairwise_data.pt')

    # Step 1: Generate training data (if not exists)
    if not os.path.exists(data_path):
        print("=== Step 1: Generating pairwise training data ===")
        gen = DataGenerator(
            dp_config_path=args.dp_config, dp_ckpt_path=args.dp_ckpt,
            foresight_ckpt_path=args.foresight_ckpt, foresight_dir=args.foresight_dir,
            device=args.device, K=args.K,
        )
        generate_pairwise_data(gen, args.data_dir, n_gen=args.n_gen, save_path=data_path)

    # Step 2: Train pairwise scorer
    print("\n=== Step 2: Training pairwise scorer ===")
    model = train_pairwise_scorer(data_path, epochs=args.epochs, lr=args.lr,
                                   batch_size=args.batch_size, device=args.device)

    # Step 3: Evaluate
    print("\n=== Step 3: Evaluation ===")
    gen = DataGenerator(
        dp_config_path=args.dp_config, dp_ckpt_path=args.dp_ckpt,
        foresight_ckpt_path=args.foresight_ckpt, foresight_dir=args.foresight_dir,
        device=args.device, K=args.K,
    )

    # Load best model
    best_path = os.path.join(args.save_dir, 'pairwise_scorer_best.pt')
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path))

    evaluate_pairwise_scorer(model, gen, args.data_dir, n_eval=args.n_eval, device=args.device)


if __name__ == "__main__":
    main()
