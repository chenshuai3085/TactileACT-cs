"""
TFAC_V5 Stage 1: MDN Latent Foresight 预训练脚本。

与 pretrain_latent_foresight.py 的区别:
  - 预测头从单点回归改为 Mixture Density Network (MDN)
  - 输出 K 个高斯模式 {(μ_k, σ_k, π_k)}, K=2 by default
  - Loss 从 L1 改为 NLL = -log Σ_k π_k · N(z_gt | μ_k, σ_k)
  - 额外跟踪 best-mode L1 指标 (可与 baseline L1 直接对比)

动机:
  bounce 插入任务中, 同一观测状态可能对应两种未来触觉 (平滑插入 vs bounce 接触)。
  L1 回归会平均两个模式, 产生无效的中间值预测。
  MDN 保持多模态输出, 每个模式独立学习, 不存在模态平均问题。
"""

import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import math

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils import set_seed
from TFAC_V5.dataset import ForesightEpisodicDataset
from TFAC_V5.foresight_transformer import ForesightTransformer
from TFAC_V5.tactile_vae import TactileVAE
from detr.models.backbone import Backbone, Joiner, PositionEmbeddingSine


class MDNHead(nn.Module):
    """
    Mixture Density Network head: 将 ForesightTransformer 的隐层输出映射为
    K 个高斯分量的参数 {(μ_k, σ_k, π_k)}。

    μ_k: (B, K, D) — 每个分量的均值
    σ_k: (B, K, D) — 每个分量的标准差 (对角协方差)
    π_k: (B, K) — 混合权重 (softmax 归一化)
    """

    def __init__(self, input_dim, output_dim, num_modes=2, hidden_dim=256,
                 sigma_min=0.01, sigma_max=10.0):
        super().__init__()
        self.num_modes = num_modes
        self.output_dim = output_dim
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.mu_head = nn.Linear(hidden_dim, num_modes * output_dim)
        self.log_sigma_head = nn.Linear(hidden_dim, num_modes * output_dim)
        self.pi_head = nn.Linear(hidden_dim, num_modes)

    def forward(self, x):
        """
        Args:
            x: (B, input_dim) — ForesightTransformer 输出
        Returns:
            mu:    (B, K, D) — 各分量均值
            sigma: (B, K, D) — 各分量标准差
            pi:    (B, K) — 混合权重 (已 softmax)
        """
        B = x.size(0)
        K = self.num_modes
        D = self.output_dim

        h = self.shared(x)  # (B, hidden_dim)

        mu = self.mu_head(h).view(B, K, D)

        log_sigma = self.log_sigma_head(h).view(B, K, D)
        sigma = log_sigma.exp().clamp(min=self.sigma_min, max=self.sigma_max)

        pi = F.softmax(self.pi_head(h), dim=-1)  # (B, K)

        return mu, sigma, pi


def mdn_nll_loss(mu, sigma, pi, z_gt):
    """
    MDN 负对数似然损失。

    -log p(z_gt) = -log Σ_k π_k · N(z_gt | μ_k, σ_k)

    使用 logsumexp 保证数值稳定性。

    Args:
        mu:    (B, K, D)
        sigma: (B, K, D)
        pi:    (B, K)
        z_gt:  (B, D)
    Returns:
        nll: scalar — batch 平均 NLL
    """
    B, K, D = mu.shape
    z_gt = z_gt.unsqueeze(1).expand_as(mu)  # (B, K, D)

    # log N(z | μ, σ) = -0.5 * [D*log(2π) + Σ_d log(σ_d²) + Σ_d ((z_d - μ_d)/σ_d)²]
    log_normal = -0.5 * (
        D * math.log(2 * math.pi)
        + 2 * sigma.log().sum(dim=-1)  # Σ_d log(σ_d²) = 2 * Σ_d log(σ_d)
        + ((z_gt - mu) / sigma).pow(2).sum(dim=-1)  # Σ_d ((z - μ)/σ)²
    )  # (B, K)

    log_pi = pi.log()  # (B, K)

    # logsumexp for numerical stability
    log_likelihood = torch.logsumexp(log_pi + log_normal, dim=-1)  # (B,)

    nll = -log_likelihood.mean()
    return nll


def best_mode_l1(mu, pi, z_gt):
    """
    Best-mode L1: 选择概率最高的模式, 计算其 μ 与 GT 的 L1 距离。
    用于与 baseline L1 直接对比。

    Args:
        mu: (B, K, D)
        pi: (B, K)
        z_gt: (B, D)
    Returns:
        l1: scalar
    """
    best_k = pi.argmax(dim=-1)  # (B,)
    B = mu.size(0)
    mu_best = mu[torch.arange(B, device=mu.device), best_k]  # (B, D)
    return F.l1_loss(mu_best, z_gt)


def oracle_mode_l1(mu, z_gt):
    """
    Oracle-mode L1: 选择离 GT 最近的模式 (cheating metric, 衡量 MDN 覆盖能力)。

    Args:
        mu: (B, K, D)
        z_gt: (B, D)
    Returns:
        l1: scalar
    """
    z_gt_exp = z_gt.unsqueeze(1).expand_as(mu)  # (B, K, D)
    per_mode_l1 = (mu - z_gt_exp).abs().mean(dim=-1)  # (B, K)
    best_l1 = per_mode_l1.min(dim=-1).values  # (B,)
    return best_l1.mean()


class MDNLatentForesightPretrainModel(nn.Module):
    """
    Stage 1 预训练模型 (MDN 版): ForesightTransformer + MDN Head。

    ForesightTransformer 输出 (B, 144) 的 latent representation,
    MDN Head 将其映射为 K 个高斯分量 {(μ, σ, π)}。
    """

    def __init__(self, camera_names, cam_backbone_mapping,
                 hidden_dim=512, state_dim=7,
                 foresight_layers=3, foresight_nheads=8,
                 foresight_dim_feedforward=2048, dropout=0.1,
                 tactile_mode="marker",
                 max_history=8,
                 foresight_change_weight=False,
                 predict_horizon=1,
                 tactile_vae_ckpt=None,
                 tactile_vae_latent_dim=16,
                 # MDN params
                 mdn_num_modes=2,
                 mdn_hidden_dim=256,
                 mdn_sigma_min=0.01,
                 mdn_sigma_max=10.0,
                 **kwargs):
        super().__init__()

        self.camera_names = camera_names
        self.cam_backbone_mapping = cam_backbone_mapping
        self.hidden_dim = hidden_dim
        self.tactile_mode = tactile_mode
        self.foresight_change_weight = foresight_change_weight
        self.predict_horizon = predict_horizon
        self.tactile_vae_latent_dim = tactile_vae_latent_dim
        self.mdn_num_modes = mdn_num_modes

        # --- Vision backbone (frozen, ImageNet pretrained ResNet18) ---
        N_steps = hidden_dim // 2
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
        backbone = Backbone(name='resnet18', train_backbone=False,
                            return_interm_layers=False, dilation=False)
        backbone_model = Joiner(backbone, position_embedding)
        backbone_model.num_channels = backbone.num_channels
        self.backbone = nn.ModuleList([backbone_model])
        self.backbone.requires_grad_(False)

        self.input_proj = nn.Conv2d(backbone_model.num_channels, hidden_dim, kernel_size=1)

        # --- TactileVAE (frozen) ---
        self.tactile_vae = TactileVAE(latent_dim=tactile_vae_latent_dim, temporal_window=8)
        if tactile_vae_ckpt and os.path.exists(tactile_vae_ckpt):
            ckpt = torch.load(tactile_vae_ckpt, map_location='cpu')
            if 'model_state_dict' in ckpt:
                self.tactile_vae.load_state_dict(ckpt['model_state_dict'])
            else:
                self.tactile_vae.load_state_dict(ckpt)
            print(f"Loaded TactileVAE from: {tactile_vae_ckpt}")
        self.tactile_vae.requires_grad_(False)

        # --- Latent → token projection (trainable) ---
        self.tac_latent_proj = nn.Linear(tactile_vae_latent_dim, hidden_dim)
        self.tac_spatial_pos_embed = nn.Parameter(
            torch.randn(9, 1, hidden_dim) * 0.02)

        # --- Camera-level embedding (trainable) ---
        n_cams = len([c for c in camera_names if c not in ('gelsight', 'blank')])
        self.cam_embed = nn.Parameter(torch.randn(n_cams, 1, 1, hidden_dim) * 0.02)

        # --- ForesightTransformer (trainable) ---
        tactile_out_dim = tactile_vae_latent_dim * 3 * 3  # 16*3*3 = 144
        self.foresight = ForesightTransformer(
            d_model=hidden_dim, action_dim=state_dim,
            num_layers=foresight_layers, nhead=foresight_nheads,
            dim_feedforward=foresight_dim_feedforward, dropout=dropout,
            tactile_out_dim=tactile_out_dim,
            tactile_decoder_type="linear",
            max_history=max_history,
            predict_horizon=predict_horizon,
            n_tactile_spatial=9)

        # --- MDN Head (trainable): ForesightTransformer output → K Gaussian modes ---
        self.mdn_head = MDNHead(
            input_dim=tactile_out_dim,  # 144
            output_dim=tactile_out_dim,  # 144
            num_modes=mdn_num_modes,
            hidden_dim=mdn_hidden_dim,
            sigma_min=mdn_sigma_min,
            sigma_max=mdn_sigma_max,
        )

    def _encode_images(self, images):
        """编码单帧所有相机图像 → tokens."""
        all_cam_features = []
        n_vision = 0
        n_tactile = 0
        vis_cam_idx = 0
        z_current_flat = None

        for cam_id, cam_name in enumerate(self.camera_names):
            if cam_name == 'gelsight' and self.tactile_mode == 'marker':
                with torch.no_grad():
                    z_t, _ = self.tactile_vae.encode_single_frame(images[cam_id])

                B = z_t.size(0)
                C = self.tactile_vae_latent_dim
                z_current_flat = z_t.reshape(B, -1)
                z_flat = z_t.reshape(B, C, 9)
                z_flat = z_flat.permute(2, 0, 1)
                proj = self.tac_latent_proj(z_flat)
                proj = proj + self.tac_spatial_pos_embed
                n_tactile += 9
            else:
                features, pos = self.backbone[self.cam_backbone_mapping[cam_name]](images[cam_id])
                features = features[0]
                pos = pos[0]
                proj = self.input_proj(features).flatten(2)
                pos_flat = pos.flatten(2)

                n_tokens = proj.size(2)
                if cam_name == 'gelsight':
                    n_tactile += n_tokens
                else:
                    n_vision += n_tokens

                proj = proj.permute(2, 0, 1)
                pos_flat = pos_flat.permute(2, 0, 1)
                proj = proj + pos_flat
                if cam_name not in ('gelsight', 'blank'):
                    proj = proj + self.cam_embed[vis_cam_idx]
                    vis_cam_idx += 1

            all_cam_features.append(proj)

        src = torch.cat(all_cam_features, dim=0)
        return src, n_vision, n_tactile, z_current_flat

    def forward(self, images, actions, future_images=None, qpos=None):
        """
        Returns:
            mu:         (B, K, 144) — K 个分量的均值
            sigma:      (B, K, 144) — K 个分量的标准差
            pi:         (B, K) — 混合权重
            z_gt:       (B, 144) — GT latent (t+H)
            future_raw: (B, 9, 9, 2) — raw last future frame
            z_current:  (B, 144) — current frame latent
            t_hat_raw:  (B, 144) — ForesightTransformer 原始输出 (用于 obs aux loss)
        """
        src, n_vision, n_tactile, z_current = self._encode_images(images)

        v_tokens = src[:n_vision]
        t_tokens = src[n_vision:]

        t_hat_raw, v_hat_future, t_embed = self.foresight(
            v_tokens, t_tokens, actions, n_vision, proprio=qpos)

        # MDN head: t_hat_raw (B, 144) → (μ, σ, π)
        mu, sigma, pi = self.mdn_head(t_hat_raw)

        # GT: encode future frames through frozen TactileVAE
        z_gt = None
        future_raw_last = None
        if future_images is not None:
            for cam_id, cam_name in enumerate(self.camera_names):
                if cam_name == 'gelsight' and self.tactile_mode == 'marker':
                    future_marker = future_images[cam_id]
                    B = future_marker.shape[0]
                    H = future_marker.shape[1]

                    with torch.no_grad():
                        if future_marker.dim() == 6:
                            z_gt_list = []
                            for h in range(H):
                                z_h, _ = self.tactile_vae.encode_single_frame(
                                    future_marker[:, h])
                                z_gt_list.append(z_h.reshape(B, -1))
                            z_gt = torch.stack(z_gt_list, dim=1)
                            future_raw_last = future_marker[:, -1, -1]
                        else:
                            z_gt_list = []
                            for h in range(H):
                                z_h, _ = self.tactile_vae.encode_single_frame(
                                    future_marker[:, h:h+1].unsqueeze(1) if future_marker[:, h].dim() == 3
                                    else future_marker[:, h].unsqueeze(1))
                                z_gt_list.append(z_h.reshape(B, -1))
                            z_gt = torch.stack(z_gt_list, dim=1)
                            future_raw_last = future_marker[:, -1]
                    break

        # predict_horizon=1: take last frame GT
        if self.predict_horizon == 1 and z_gt is not None:
            z_gt = z_gt[:, -1]

        return mu, sigma, pi, z_gt, future_raw_last, z_current, t_hat_raw


def compute_mdn_loss(mu, sigma, pi, z_gt, model=None, future_raw=None, t_hat_raw=None):
    """
    MDN loss = NLL + obs auxiliary。

    Returns:
        total:       scalar — total loss
        nll:         float — NLL loss value
        best_l1:     float — best-mode L1 (最高概率模式)
        oracle_l1:   float — oracle-mode L1 (最近模式, cheating metric)
        obs_loss:    float — obs-space auxiliary loss
    """
    # Primary: MDN NLL
    nll = mdn_nll_loss(mu, sigma, pi, z_gt)

    # Metrics (不参与梯度)
    with torch.no_grad():
        bl1 = best_mode_l1(mu, pi, z_gt).item()
        ol1 = oracle_mode_l1(mu, z_gt).item()

    # Obs-space auxiliary loss (使用最高概率模式的 μ 解码)
    loss_obs = torch.tensor(0.0, device=mu.device)
    if model is not None and future_raw is not None and t_hat_raw is not None:
        with torch.no_grad():
            C = model.tactile_vae_latent_dim
            # 用最高概率模式
            best_k = pi.argmax(dim=-1)  # (B,)
            B = mu.size(0)
            mu_best = mu[torch.arange(B, device=mu.device), best_k]  # (B, 144)
            z_hat_spatial = mu_best.reshape(-1, C, 3, 3)
            marker_hat = model.tactile_vae.decoder(z_hat_spatial)

        if future_raw.dim() == 4:
            marker_gt = future_raw
        else:
            marker_gt = future_raw[:, -1]
        loss_obs = F.smooth_l1_loss(marker_hat.detach(), marker_gt)

    total = nll + 0.3 * loss_obs

    return total, nll.item(), bl1, ol1, loss_obs.item()


# ====== Utility functions (same as base version) ======

def _scan_episode_paths(dataset_dir):
    import glob
    direct = sorted(glob.glob(os.path.join(dataset_dir, 'episode_*.hdf5')))
    if direct:
        return direct
    sub_paths = []
    for sub in sorted(os.listdir(dataset_dir)):
        sub_dir = os.path.join(dataset_dir, sub)
        if os.path.isdir(sub_dir):
            sub_paths.extend(sorted(glob.glob(os.path.join(sub_dir, 'episode_*.hdf5'))))
    return sub_paths


def _infer_meta_from_path(episode_path, config_overrides=None):
    import h5py
    overrides = config_overrides or {}
    with h5py.File(episode_path, 'r') as root:
        if 'camera_names' in overrides and overrides['camera_names']:
            camera_names = overrides['camera_names']
        else:
            camera_names = sorted(root['observations/images'].keys()) if 'observations/images' in root else []
            if ('observations/tac' in root or 'observations/gelsight' in root) and 'gelsight' not in camera_names:
                camera_names.append('gelsight')

        proprio_key = overrides.get('proprio_key', None)
        if not proprio_key:
            for c in ['qpos', 'proprio_joint', 'proprio_eef']:
                if f'observations/{c}' in root:
                    proprio_key = c
                    break
            if not proprio_key:
                proprio_key = 'qpos'

        state_dim = overrides.get('state_dim', None)
        if not state_dim:
            pp = f'observations/{proprio_key}'
            state_dim = root[pp].shape[-1] if pp in root else 7

        action_key = overrides.get('action_key', None)
        if not action_key:
            action_key = 'actions/joint_abs' if 'actions/joint_abs' in root else 'action'

        tac_side = overrides.get('tac_side', 'left')
        tac_img_key = overrides.get('tac_img_key', 'img')

    return {
        'camera_names': camera_names,
        'state_dim': state_dim,
        'proprio_key': proprio_key,
        'action_key': action_key,
        'tac_side': tac_side,
        'tac_img_key': tac_img_key,
    }


def _compute_norm_stats(episode_paths, proprio_key, action_key, tactile_mode, tac_side,
                        max_episodes=50):
    import h5py
    all_qpos, all_action, all_mo = [], [], []
    sample_paths = episode_paths if len(episode_paths) <= max_episodes else \
        [episode_paths[i] for i in np.linspace(0, len(episode_paths)-1, max_episodes, dtype=int)]

    for p in sample_paths:
        with h5py.File(p, 'r') as root:
            all_qpos.append(root[f'observations/{proprio_key}'][()])
            all_action.append(root[f'{action_key}'][()])
            if tactile_mode == 'marker':
                mo_path = f'observations/tac/{tac_side}/marker_offset'
                if mo_path in root:
                    all_mo.append(root[mo_path][()])

    qpos = np.concatenate(all_qpos, axis=0)
    action = np.concatenate(all_action, axis=0)
    stats = {
        'qpos_mean': qpos.mean(axis=0).astype(np.float32),
        'qpos_std': qpos.std(axis=0).astype(np.float32),
        'action_mean': action.mean(axis=0).astype(np.float32),
        'action_std': action.std(axis=0).astype(np.float32),
    }
    stats['qpos_std'] = np.clip(stats['qpos_std'], 1e-4, None)
    stats['action_std'] = np.clip(stats['action_std'], 1e-4, None)

    if all_mo:
        mo = np.concatenate(all_mo, axis=0)
        stats['marker_offset_mean'] = mo.mean(axis=(0, 1, 2)).astype(np.float32)
        stats['marker_offset_std'] = np.clip(mo.std(axis=(0, 1, 2)), 1e-4, None).astype(np.float32)

    return stats


# ====== Main training loop ======

def main(args):
    save_dir = args['save_dir']
    model_name = args['name']
    batch_size = args['batch_size']
    num_epochs = args['num_epochs']
    chunk_size = args['chunk_size']
    seed = args['seed']
    gpu = args.get('gpu', -1)

    ckpt_dir = os.path.join(save_dir, model_name)
    dataset_dir = args.get('dataset_dir', os.path.join(save_dir, 'data'))
    assert os.path.exists(dataset_dir), f'{dataset_dir} does not exist.'

    episode_paths = _scan_episode_paths(dataset_dir)
    assert len(episode_paths) > 0, f'No episode_*.hdf5 found in {dataset_dir}'
    print(f"Found {len(episode_paths)} episodes in {dataset_dir}")

    meta_data = _infer_meta_from_path(episode_paths[0], config_overrides=args)
    meta_data['num_episodes'] = len(episode_paths)
    camera_names = meta_data['camera_names']
    state_dim = meta_data['state_dim']
    proprio_key = meta_data['proprio_key']
    action_key = meta_data['action_key']
    tac_side = meta_data['tac_side']
    tac_img_key = meta_data['tac_img_key']
    print(f"Meta: cameras={camera_names}, state_dim={state_dim}, "
          f"proprio={proprio_key}, action={action_key}")

    tactile_mode = args.get('tactile_mode', 'marker')

    norm_stats = _compute_norm_stats(episode_paths, proprio_key, action_key,
                                      tactile_mode, tac_side)

    vae_stats_path = args.get('tactile_vae_stats', None)
    if vae_stats_path and os.path.exists(vae_stats_path):
        with open(vae_stats_path) as f:
            vae_stats = json.load(f)
        norm_stats['marker_offset_mean'] = np.array(vae_stats['mean'], dtype=np.float32)
        norm_stats['marker_offset_std'] = np.array(vae_stats['std'], dtype=np.float32)
        print(f"Loaded TactileVAE norm stats: mean={vae_stats['mean']}, std={vae_stats['std']}")

    args['norm_stats'] = {k: v.tolist() if hasattr(v, 'tolist') else v
                          for k, v in norm_stats.items()}

    set_seed(seed)
    if gpu != -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    # --- MDN params ---
    mdn_num_modes = args.get('mdn_num_modes', 2)
    mdn_hidden_dim = args.get('mdn_hidden_dim', 256)
    mdn_sigma_min = args.get('mdn_sigma_min', 0.01)
    mdn_sigma_max = args.get('mdn_sigma_max', 10.0)

    # --- Build model ---
    cam_backbone_mapping = {cam_name: 0 for cam_name in camera_names}

    model = MDNLatentForesightPretrainModel(
        camera_names=camera_names,
        cam_backbone_mapping=cam_backbone_mapping,
        hidden_dim=args['hidden_dim'],
        state_dim=state_dim,
        foresight_layers=args.get('foresight_layers', 3),
        foresight_nheads=args.get('foresight_nheads', 8),
        foresight_dim_feedforward=args.get('foresight_dim_feedforward', 2048),
        dropout=args.get('dropout', 0.1),
        tactile_mode=tactile_mode,
        max_history=args.get('max_history', 8),
        foresight_change_weight=args.get('foresight_change_weight', False),
        predict_horizon=args.get('predict_horizon', 1),
        tactile_vae_ckpt=args.get('tactile_vae_ckpt', None),
        tactile_vae_latent_dim=args.get('tactile_vae_latent_dim', 16),
        mdn_num_modes=mdn_num_modes,
        mdn_hidden_dim=mdn_hidden_dim,
        mdn_sigma_min=mdn_sigma_min,
        mdn_sigma_max=mdn_sigma_max,
    )
    model.cuda()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"MDN Latent Foresight: {trainable/1e6:.2f}M trainable / {total/1e6:.2f}M total")
    print(f"MDN modes: K={mdn_num_modes}, hidden={mdn_hidden_dim}")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.get('lr', 4e-5),
        weight_decay=args.get('weight_decay', 1e-4),
    )

    # --- Checkpoint dir ---
    if os.path.exists(ckpt_dir):
        n = 0
        while os.path.exists(ckpt_dir + f'_{n}'):
            n += 1
        print(f'Warning: {ckpt_dir} exists. Using {ckpt_dir}_{n}')
        ckpt_dir = ckpt_dir + f'_{n}'
    os.makedirs(ckpt_dir)

    combo_dict = {**args, **meta_data}
    with open(os.path.join(ckpt_dir, 'args.json'), 'w') as f:
        json.dump(combo_dict, f, indent=4)

    # --- Dataset ---
    foresight_horizon = args.get('foresight_horizon', 10)
    history_len = args.get('history_len', 1)
    tactile_vae_window = args.get('tactile_vae_window', 8)
    train_ratio = 0.9
    shuffled_paths = np.random.permutation(episode_paths).tolist()
    n_train = int(train_ratio * len(shuffled_paths))
    train_paths = shuffled_paths[:n_train]
    val_paths = shuffled_paths[n_train:]
    print(f"Train: {len(train_paths)}, Val: {len(val_paths)}")

    stats_path = os.path.join(ckpt_dir, 'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(norm_stats, f)

    train_dataset = ForesightEpisodicDataset(
        train_paths, dataset_dir, camera_names, norm_stats,
        chunk_size=chunk_size, foresight_horizon=foresight_horizon,
        proprio_key=proprio_key, action_key=action_key,
        tac_side=tac_side, tac_img_key=tac_img_key,
        tactile_mode=tactile_mode, history_len=history_len,
        tactile_vae_window=tactile_vae_window, preload=True)
    val_dataset = ForesightEpisodicDataset(
        val_paths, dataset_dir, camera_names, norm_stats,
        chunk_size=chunk_size, foresight_horizon=foresight_horizon,
        proprio_key=proprio_key, action_key=action_key,
        tac_side=tac_side, tac_img_key=tac_img_key,
        tactile_mode=tactile_mode, history_len=history_len,
        tactile_vae_window=tactile_vae_window, preload=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)

    # --- Training loop ---
    best_val_loss = float('inf')
    best_epoch = 0

    train_losses = []
    val_losses = []
    train_nll_losses = []
    val_nll_losses = []
    train_best_l1 = []
    val_best_l1 = []
    train_oracle_l1 = []
    val_oracle_l1 = []
    train_obs_losses = []
    val_obs_losses = []

    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        # ---- Train ----
        model.train()
        model.backbone.eval()
        model.tactile_vae.eval()
        epoch_loss = 0.0
        epoch_nll = 0.0
        epoch_bl1 = 0.0
        epoch_ol1 = 0.0
        epoch_obs = 0.0
        n_train_samples = 0

        for batch in train_loader:
            all_cam_images, qpos_data, action_data, is_pad, future_cam_images, history_cam_images = batch

            images = [img.cuda() for img in all_cam_images]
            qpos = qpos_data.cuda()
            actions = action_data.cuda()
            future_images = [img.cuda() for img in future_cam_images]

            mu, sigma, pi, z_gt, future_raw, z_current, t_hat_raw = model(
                images, actions, future_images=future_images, qpos=qpos)

            loss, nll_val, bl1_val, ol1_val, obs_val = compute_mdn_loss(
                mu, sigma, pi, z_gt, model=model,
                future_raw=future_raw, t_hat_raw=t_hat_raw)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            bs = actions.size(0)
            epoch_loss += loss.item() * bs
            epoch_nll += nll_val * bs
            epoch_bl1 += bl1_val * bs
            epoch_ol1 += ol1_val * bs
            epoch_obs += obs_val * bs
            n_train_samples += bs

        avg_train = epoch_loss / n_train_samples
        avg_train_nll = epoch_nll / n_train_samples
        avg_train_bl1 = epoch_bl1 / n_train_samples
        avg_train_ol1 = epoch_ol1 / n_train_samples
        avg_train_obs = epoch_obs / n_train_samples
        train_losses.append(avg_train)
        train_nll_losses.append(avg_train_nll)
        train_best_l1.append(avg_train_bl1)
        train_oracle_l1.append(avg_train_ol1)
        train_obs_losses.append(avg_train_obs)

        # ---- Validate ----
        model.eval()
        epoch_loss = 0.0
        epoch_nll = 0.0
        epoch_bl1 = 0.0
        epoch_ol1 = 0.0
        epoch_obs = 0.0
        n_val_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                all_cam_images, qpos_data, action_data, is_pad, future_cam_images, history_cam_images = batch
                images = [img.cuda() for img in all_cam_images]
                qpos = qpos_data.cuda()
                actions = action_data.cuda()
                future_images = [img.cuda() for img in future_cam_images]

                mu, sigma, pi, z_gt, future_raw, z_current, t_hat_raw = model(
                    images, actions, future_images=future_images, qpos=qpos)

                loss, nll_val, bl1_val, ol1_val, obs_val = compute_mdn_loss(
                    mu, sigma, pi, z_gt, model=model,
                    future_raw=future_raw, t_hat_raw=t_hat_raw)

                bs = actions.size(0)
                epoch_loss += loss.item() * bs
                epoch_nll += nll_val * bs
                epoch_bl1 += bl1_val * bs
                epoch_ol1 += ol1_val * bs
                epoch_obs += obs_val * bs
                n_val_samples += bs

        avg_val = epoch_loss / n_val_samples
        avg_val_nll = epoch_nll / n_val_samples
        avg_val_bl1 = epoch_bl1 / n_val_samples
        avg_val_ol1 = epoch_ol1 / n_val_samples
        avg_val_obs = epoch_obs / n_val_samples
        val_losses.append(avg_val)
        val_nll_losses.append(avg_val_nll)
        val_best_l1.append(avg_val_bl1)
        val_oracle_l1.append(avg_val_ol1)
        val_obs_losses.append(avg_val_obs)

        # ---- Logging ----
        is_best = avg_val < best_val_loss
        if is_best:
            best_val_loss = avg_val
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(ckpt_dir, 'foresight_best.ckpt'))

        pbar.set_postfix(
            trn=f"{avg_train:.4f}", val=f"{avg_val:.4f}",
            nll=f"{avg_val_nll:.4f}", bl1=f"{avg_val_bl1:.4f}",
            ol1=f"{avg_val_ol1:.4f}", best=f"{best_val_loss:.4f}")

        if epoch % 10 == 0:
            msg = (f"\nEpoch {epoch}: train={avg_train:.4f} (nll={avg_train_nll:.4f}, bl1={avg_train_bl1:.4f}, ol1={avg_train_ol1:.4f})"
                   f"  val={avg_val:.4f} (nll={avg_val_nll:.4f}, bl1={avg_val_bl1:.4f}, ol1={avg_val_ol1:.4f})"
                   f"  best: epoch {best_epoch}, {best_val_loss:.4f}")
            print(msg)

        if epoch % 100 == 0:
            torch.save(model.state_dict(),
                       os.path.join(ckpt_dir, f'foresight_epoch_{epoch}.ckpt'))

        if epoch % 10 == 0:
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))

            axes[0, 0].plot(train_losses, label='train', alpha=0.7)
            axes[0, 0].plot(val_losses, label='val', alpha=0.7)
            axes[0, 0].set_title('Total Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].legend()

            axes[0, 1].plot(train_nll_losses, label='train', alpha=0.7)
            axes[0, 1].plot(val_nll_losses, label='val', alpha=0.7)
            axes[0, 1].set_title('NLL Loss')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].legend()

            axes[0, 2].plot(train_obs_losses, label='train', alpha=0.7)
            axes[0, 2].plot(val_obs_losses, label='val', alpha=0.7)
            axes[0, 2].set_title('Obs-Space Aux Loss')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].legend()

            axes[1, 0].plot(train_best_l1, label='train', alpha=0.7)
            axes[1, 0].plot(val_best_l1, label='val', alpha=0.7)
            axes[1, 0].set_title('Best-Mode L1 (vs baseline)')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].legend()

            axes[1, 1].plot(train_oracle_l1, label='train', alpha=0.7)
            axes[1, 1].plot(val_oracle_l1, label='val', alpha=0.7)
            axes[1, 1].set_title('Oracle-Mode L1 (cheating)')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].legend()

            # Mode usage histogram (from last batch)
            if pi is not None:
                with torch.no_grad():
                    mode_usage = pi.argmax(dim=-1).cpu().numpy()
                axes[1, 2].hist(mode_usage, bins=np.arange(mdn_num_modes + 1) - 0.5,
                                rwidth=0.8, color='steelblue', edgecolor='black')
                axes[1, 2].set_title('Mode Selection (last batch)')
                axes[1, 2].set_xlabel('Mode index')
                axes[1, 2].set_ylabel('Count')
                axes[1, 2].set_xticks(range(mdn_num_modes))

            plt.suptitle(f'MDN Latent Foresight (K={mdn_num_modes}, best: epoch {best_epoch}, {best_val_loss:.4f})')
            plt.tight_layout()
            plt.savefig(os.path.join(ckpt_dir, 'pretrain_loss.png'), dpi=100)
            plt.close()

    # ---- Final save ----
    torch.save(model.state_dict(), os.path.join(ckpt_dir, 'foresight_last.ckpt'))

    history = {
        'train': train_losses, 'val': val_losses,
        'train_nll': train_nll_losses, 'val_nll': val_nll_losses,
        'train_best_l1': train_best_l1, 'val_best_l1': val_best_l1,
        'train_oracle_l1': train_oracle_l1, 'val_oracle_l1': val_oracle_l1,
        'train_obs': train_obs_losses, 'val_obs': val_obs_losses,
    }
    with open(os.path.join(ckpt_dir, 'pretrain_history.pkl'), 'wb') as f:
        pickle.dump(history, f)

    # Final plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].plot(train_losses, label='train', alpha=0.7)
    axes[0, 0].plot(val_losses, label='val', alpha=0.7)
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend()

    axes[0, 1].plot(train_nll_losses, label='train', alpha=0.7)
    axes[0, 1].plot(val_nll_losses, label='val', alpha=0.7)
    axes[0, 1].set_title('NLL Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend()

    axes[0, 2].plot(train_obs_losses, label='train', alpha=0.7)
    axes[0, 2].plot(val_obs_losses, label='val', alpha=0.7)
    axes[0, 2].set_title('Obs-Space Aux Loss')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].legend()

    axes[1, 0].plot(train_best_l1, label='train', alpha=0.7)
    axes[1, 0].plot(val_best_l1, label='val', alpha=0.7)
    axes[1, 0].set_title('Best-Mode L1 (vs baseline)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].legend()

    axes[1, 1].plot(train_oracle_l1, label='train', alpha=0.7)
    axes[1, 1].plot(val_oracle_l1, label='val', alpha=0.7)
    axes[1, 1].set_title('Oracle-Mode L1 (cheating)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].legend()

    axes[1, 2].text(0.5, 0.5, f'K={mdn_num_modes}\nBest epoch: {best_epoch}\nBest val: {best_val_loss:.4f}',
                    ha='center', va='center', fontsize=14, transform=axes[1, 2].transAxes)
    axes[1, 2].set_title('Summary')

    plt.suptitle(f'MDN Latent Foresight (K={mdn_num_modes}, best: epoch {best_epoch}, {best_val_loss:.4f})')
    plt.tight_layout()
    plt.savefig(os.path.join(ckpt_dir, 'pretrain_loss.png'), dpi=100)
    plt.close()

    print(f"\nDone! Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")
    print(f"Checkpoints saved to: {ckpt_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    cli = parser.parse_args()

    with open(cli.config, 'r') as f:
        config = json.load(f)
    main(config)
