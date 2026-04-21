"""
TFAC_V5 Stage 1: Latent Foresight 预训练脚本。

训练 ForesightTransformer 在 TactileVAE latent 空间 (16×3×3=144-dim) 做预测。
Vision backbone (ImageNet ResNet18) 和 TactileVAE 均冻结, 用 GT action 作为条件。

训练模块:
  - input_proj (Conv2d): backbone feature → hidden_dim
  - tac_latent_proj (Linear): VAE latent channel (16) → hidden_dim (512)
  - tac_spatial_pos_embed: 9 spatial position embeddings
  - foresight (ForesightTransformer): 预测未来 latent

冻结模块:
  - vision backbone (ImageNet pretrained ResNet18)
  - TactileVAE (预训练好的 encoder + decoder)
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

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils import set_seed
from TFAC_V5.dataset import ForesightEpisodicDataset
from TFAC_V5.foresight_transformer import ForesightTransformer
from TFAC_V5.tactile_vae import TactileVAE
from detr.models.backbone import Backbone, Joiner, PositionEmbeddingSine


class LatentForesightPretrainModel(nn.Module):
    """
    Stage 1 预训练模型: ForesightTransformer 在 TactileVAE latent 空间预测。

    触觉编码: TactileVAE encoder → z_t (B, 16, 3, 3) → 9 spatial tokens (9, B, 512)
    预测目标: VAE latent (B, 144) 而非 raw marker_offset (B, 162)
    """

    def __init__(self, camera_names, cam_backbone_mapping,
                 hidden_dim=512, state_dim=7,
                 foresight_layers=3, foresight_nheads=8,
                 foresight_dim_feedforward=2048, dropout=0.1,
                 tactile_mode="marker",
                 max_history=8,
                 foresight_change_weight=False,
                 predict_horizon=1,
                 # TactileVAE params
                 tactile_vae_ckpt=None,
                 tactile_vae_latent_dim=16,
                 **kwargs):
        super().__init__()

        self.camera_names = camera_names
        self.cam_backbone_mapping = cam_backbone_mapping
        self.hidden_dim = hidden_dim
        self.tactile_mode = tactile_mode
        self.foresight_change_weight = foresight_change_weight
        self.predict_horizon = predict_horizon
        self.tactile_vae_latent_dim = tactile_vae_latent_dim

        # --- Vision backbone (frozen, ImageNet pretrained ResNet18) ---
        N_steps = hidden_dim // 2
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
        backbone = Backbone(name='resnet18', train_backbone=False,
                            return_interm_layers=False, dilation=False)
        backbone_model = Joiner(backbone, position_embedding)
        backbone_model.num_channels = backbone.num_channels
        self.backbone = nn.ModuleList([backbone_model])
        self.backbone.requires_grad_(False)

        # input_proj (trainable)
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

        # --- G-TCL (optional, controlled by config) ---
        self.use_gtcl = kwargs.get('use_gtcl', False)
        if self.use_gtcl:
            from TFAC_V5.foresight_transformer import GatedTemporalContrastive
            self.gtcl = GatedTemporalContrastive(
                latent_dim=tactile_out_dim,
                proj_dim=kwargs.get('gtcl_proj_dim', 64),
                temperature=kwargs.get('gtcl_temperature', 0.07),
                epsilon=kwargs.get('gtcl_epsilon', 2.5),
            )

    def _encode_images(self, images):
        """编码单帧所有相机图像 → tokens (with spatial + camera pos encoding)."""
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
                z_current_flat = z_t.reshape(B, -1)  # (B, 144) for G-TCL gating
                z_flat = z_t.reshape(B, C, 9)   # (B, 16, 9)
                z_flat = z_flat.permute(2, 0, 1)  # (9, B, 16)
                proj = self.tac_latent_proj(z_flat)  # (9, B, 512)
                proj = proj + self.tac_spatial_pos_embed  # spatial pos for tactile
                n_tactile += 9
            else:
                features, pos = self.backbone[self.cam_backbone_mapping[cam_name]](images[cam_id])
                features = features[0]
                pos = pos[0]
                proj = self.input_proj(features).flatten(2)  # (B, D, N)
                pos_flat = pos.flatten(2)  # (B, D, N)

                n_tokens = proj.size(2)
                if cam_name == 'gelsight':
                    n_tactile += n_tokens
                else:
                    n_vision += n_tokens

                proj = proj.permute(2, 0, 1)  # (N, B, D)
                pos_flat = pos_flat.permute(2, 0, 1)  # (N, B, D)

                # spatial position encoding
                proj = proj + pos_flat
                # camera-level embedding (区分 global vs wrist)
                if cam_name not in ('gelsight', 'blank'):
                    proj = proj + self.cam_embed[vis_cam_idx]  # (1, 1, D) broadcast
                    vis_cam_idx += 1

            all_cam_features.append(proj)

        src = torch.cat(all_cam_features, dim=0)
        return src, n_vision, n_tactile, z_current_flat

    def forward(self, images, actions, future_images=None, qpos=None):
        """
        Returns:
            t_hat:      (B, 144) — predicted latent
            z_gt:       (B, 144) — GT latent from frozen TactileVAE (t+H)
            future_raw: (B, 9, 9, 2) — raw last future frame
            z_current:  (B, 144) or None — current frame latent (for G-TCL)
            z_gt_prev:  (B, 144) or None — t+H-1 frame latent (for G-TCL hard neg)
        """
        # Encode current frame
        src, n_vision, n_tactile, z_current = self._encode_images(images)

        # Foresight: predict future tactile latent
        v_tokens = src[:n_vision]
        t_tokens = src[n_vision:]

        t_hat_raw, v_hat_future, t_embed = self.foresight(
            v_tokens, t_tokens, actions, n_vision, proprio=qpos)
        # t_hat_raw: (B, 144) for predict_horizon=1

        # GT: encode future frames through frozen TactileVAE
        z_gt = None
        future_raw_last = None
        if future_images is not None:
            for cam_id, cam_name in enumerate(self.camera_names):
                if cam_name == 'gelsight' and self.tactile_mode == 'marker':
                    future_marker = future_images[cam_id]
                    # future_marker: (B, H, T=8, 9, 9, 2) or (B, H, 9, 9, 2)
                    B = future_marker.shape[0]
                    H = future_marker.shape[1]

                    with torch.no_grad():
                        if future_marker.dim() == 6:
                            # VAE window mode: (B, H, T=8, 9, 9, 2)
                            z_gt_list = []
                            for h in range(H):
                                z_h, _ = self.tactile_vae.encode_single_frame(
                                    future_marker[:, h])  # (B, T=8, 9, 9, 2) → (B, 16, 3, 3)
                                z_gt_list.append(z_h.reshape(B, -1))  # (B, 144)
                            z_gt = torch.stack(z_gt_list, dim=1)  # (B, H, 144)
                            # Raw last frame for obs-space auxiliary loss
                            future_raw_last = future_marker[:, -1, -1]  # (B, 9, 9, 2) last future, last window frame
                        else:
                            # Legacy mode: (B, H, 9, 9, 2) — encode with T=1 (not ideal)
                            z_gt_list = []
                            for h in range(H):
                                z_h, _ = self.tactile_vae.encode_single_frame(
                                    future_marker[:, h:h+1].unsqueeze(1) if future_marker[:, h].dim() == 3
                                    else future_marker[:, h].unsqueeze(1))
                                z_gt_list.append(z_h.reshape(B, -1))
                            z_gt = torch.stack(z_gt_list, dim=1)
                            future_raw_last = future_marker[:, -1]
                    break

        # predict_horizon=1: take last frame GT, save prev for G-TCL hard negative
        z_gt_prev = None
        if self.predict_horizon == 1 and z_gt is not None:
            if z_gt.shape[1] >= 2:
                z_gt_prev = z_gt[:, -2]  # (B, 144) — t+H-1 frame (hard negative)
            z_gt = z_gt[:, -1]  # (B, 144) — t+H frame (positive/GT)

        return t_hat_raw, z_gt, future_raw_last, z_current, z_gt_prev


def compute_loss(t_hat, z_gt, model=None, future_raw=None,
                 foresight_change_weight=False, z_current=None,
                 z_gt_prev=None, gtcl_module=None, gtcl_weight=0.1,
                 delta_weighted=False, delta_alpha=2.0):
    """
    Primary: L1 in latent space (latent 值域小, L1 对小误差更敏感).
    Auxiliary: SmoothL1 in obs space (marker 值域大, 需要对离群值鲁棒).
    Optional: G-TCL gated temporal contrastive loss.
    Optional: Δφ-weighted L1 (delta_weighted=True).
    """
    if delta_weighted and z_current is not None:
        with torch.no_grad():
            delta_phi = (z_gt - z_current).norm(dim=-1)  # (B,)
            weights = 1.0 + delta_alpha * delta_phi / (delta_phi.mean() + 1e-8)  # (B,)
        per_sample_l1 = (t_hat - z_gt).abs().mean(dim=-1)  # (B,)
        loss_latent = (weights * per_sample_l1).mean()
    else:
        loss_latent = F.l1_loss(t_hat, z_gt)

    # Obs-space auxiliary loss
    loss_obs = torch.tensor(0.0, device=t_hat.device)
    if model is not None and future_raw is not None:
        with torch.no_grad():
            C = model.tactile_vae_latent_dim
            if t_hat.dim() == 2:
                z_hat_spatial = t_hat.reshape(-1, C, 3, 3)
            else:
                z_hat_spatial = t_hat[:, -1].reshape(-1, C, 3, 3)
            marker_hat = model.tactile_vae.decoder(z_hat_spatial)  # (B, 9, 9, 2)

        if future_raw.dim() == 4:  # (B, 9, 9, 2)
            marker_gt = future_raw
        else:  # (B, H, 9, 9, 2)
            marker_gt = future_raw[:, -1]
        loss_obs = F.smooth_l1_loss(marker_hat.detach(), marker_gt)

    total = loss_latent + 0.3 * loss_obs

    # G-TCL (optional)
    gtcl_loss_val = 0.0
    frac_dynamic = 0.0
    if gtcl_module is not None and z_current is not None and z_gt_prev is not None:
        gtcl_loss, frac_dynamic = gtcl_module(t_hat, z_gt, z_gt_prev, z_current)
        total = total + gtcl_weight * gtcl_loss
        gtcl_loss_val = gtcl_loss.item()

    return total, loss_latent.item(), loss_obs.item(), gtcl_loss_val, frac_dynamic


def _scan_episode_paths(dataset_dir):
    """扫描 dataset_dir 下所有 episode_*.hdf5, 支持子目录 (success/, bounce/ 等)."""
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
    """从单个 hdf5 文件推断 meta 信息."""
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
    """从部分 episodes 计算归一化统计量 (采样加速)."""
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
        mo = np.concatenate(all_mo, axis=0)  # (N, 9, 9, 2)
        stats['marker_offset_mean'] = mo.mean(axis=(0, 1, 2)).astype(np.float32)  # (2,)
        stats['marker_offset_std'] = np.clip(mo.std(axis=(0, 1, 2)), 1e-4, None).astype(np.float32)

    return stats


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

    # Scan all episode paths (supports sub-directories like success/, bounce/)
    episode_paths = _scan_episode_paths(dataset_dir)
    assert len(episode_paths) > 0, f'No episode_*.hdf5 found in {dataset_dir} (or sub-dirs)'
    print(f"Found {len(episode_paths)} episodes in {dataset_dir}")

    # Infer meta from first episode
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

    # Compute norm stats from sampled episodes
    norm_stats = _compute_norm_stats(episode_paths, proprio_key, action_key,
                                      tactile_mode, tac_side)

    # Override marker_offset normalization with TactileVAE stats
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

    # --- Build model ---
    cam_backbone_mapping = {cam_name: 0 for cam_name in camera_names}

    model = LatentForesightPretrainModel(
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
        use_gtcl=args.get('use_gtcl', False),
        gtcl_epsilon=args.get('gtcl_epsilon', 2.5),
        gtcl_proj_dim=args.get('gtcl_proj_dim', 64),
        gtcl_temperature=args.get('gtcl_temperature', 0.07),
    )
    model.cuda()

    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Latent Foresight pretrain: {trainable/1e6:.2f}M trainable / {total/1e6:.2f}M total")

    # --- Optimizer (only trainable params) ---
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
    use_gtcl = args.get('use_gtcl', False)
    gtcl_weight = args.get('gtcl_weight', 0.1)
    delta_weighted = args.get('delta_weighted', False)
    delta_alpha = args.get('delta_alpha', 2.0)

    best_val_loss = float('inf')
    best_epoch = 0
    train_losses = []
    val_losses = []
    train_latent_losses = []
    val_latent_losses = []
    train_obs_losses = []
    val_obs_losses = []
    train_gtcl_losses = []
    val_gtcl_losses = []

    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        # ---- Train ----
        model.train()
        model.backbone.eval()
        model.tactile_vae.eval()
        epoch_loss = 0.0
        epoch_latent = 0.0
        epoch_obs = 0.0
        epoch_gtcl = 0.0
        epoch_frac_dyn = 0.0
        n_train = 0

        for batch in train_loader:
            all_cam_images, qpos_data, action_data, is_pad, future_cam_images, history_cam_images = batch

            images = [img.cuda() for img in all_cam_images]
            qpos = qpos_data.cuda()
            actions = action_data.cuda()
            future_images = [img.cuda() for img in future_cam_images]

            t_hat, z_gt, future_raw, z_current, z_gt_prev = model(
                images, actions, future_images=future_images, qpos=qpos)

            loss, lat_loss, obs_loss, gtcl_loss, frac_dyn = compute_loss(
                t_hat, z_gt, model=model, future_raw=future_raw,
                z_current=z_current, z_gt_prev=z_gt_prev,
                gtcl_module=model.gtcl if use_gtcl else None,
                gtcl_weight=gtcl_weight,
                delta_weighted=delta_weighted, delta_alpha=delta_alpha)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            bs = actions.size(0)
            epoch_loss += loss.item() * bs
            epoch_latent += lat_loss * bs
            epoch_obs += obs_loss * bs
            epoch_gtcl += gtcl_loss * bs
            epoch_frac_dyn += frac_dyn * bs
            n_train += bs

        avg_train = epoch_loss / n_train
        avg_train_lat = epoch_latent / n_train
        avg_train_obs = epoch_obs / n_train
        avg_train_gtcl = epoch_gtcl / n_train
        avg_train_frac_dyn = epoch_frac_dyn / n_train
        train_losses.append(avg_train)
        train_latent_losses.append(avg_train_lat)
        train_obs_losses.append(avg_train_obs)
        train_gtcl_losses.append(avg_train_gtcl)

        # ---- Validate ----
        model.eval()
        epoch_loss = 0.0
        epoch_latent = 0.0
        epoch_obs = 0.0
        epoch_gtcl = 0.0
        n_val = 0

        with torch.no_grad():
            for batch in val_loader:
                all_cam_images, qpos_data, action_data, is_pad, future_cam_images, history_cam_images = batch
                images = [img.cuda() for img in all_cam_images]
                qpos = qpos_data.cuda()
                actions = action_data.cuda()
                future_images = [img.cuda() for img in future_cam_images]

                t_hat, z_gt, future_raw, z_current, z_gt_prev = model(
                    images, actions, future_images=future_images, qpos=qpos)

                loss, lat_loss, obs_loss, gtcl_loss, frac_dyn = compute_loss(
                    t_hat, z_gt, model=model, future_raw=future_raw,
                    z_current=z_current, z_gt_prev=z_gt_prev,
                    gtcl_module=model.gtcl if use_gtcl else None,
                    gtcl_weight=gtcl_weight,
                    delta_weighted=delta_weighted, delta_alpha=delta_alpha)

                bs = actions.size(0)
                epoch_loss += loss.item() * bs
                epoch_latent += lat_loss * bs
                epoch_obs += obs_loss * bs
                epoch_gtcl += gtcl_loss * bs
                n_val += bs

        avg_val = epoch_loss / n_val
        avg_val_lat = epoch_latent / n_val
        avg_val_obs = epoch_obs / n_val
        avg_val_gtcl = epoch_gtcl / n_val
        val_losses.append(avg_val)
        val_latent_losses.append(avg_val_lat)
        val_obs_losses.append(avg_val_obs)
        val_gtcl_losses.append(avg_val_gtcl)

        # ---- Logging ----
        is_best = avg_val < best_val_loss
        if is_best:
            best_val_loss = avg_val
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(ckpt_dir, 'foresight_best.ckpt'))

        postfix = dict(trn=f"{avg_train:.4f}", val=f"{avg_val:.4f}",
                        lat=f"{avg_val_lat:.4f}", best=f"{best_val_loss:.4f}")
        if use_gtcl:
            postfix['gtcl'] = f"{avg_val_gtcl:.4f}"
            postfix['dyn'] = f"{avg_train_frac_dyn:.1%}"
        if delta_weighted:
            postfix['dw'] = 'on'
        pbar.set_postfix(**postfix)

        if epoch % 10 == 0:
            msg = (f"\nEpoch {epoch}: train={avg_train:.4f} (lat={avg_train_lat:.4f}, obs={avg_train_obs:.4f}"
                   f"{f', gtcl={avg_train_gtcl:.4f}, dyn={avg_train_frac_dyn:.1%}' if use_gtcl else ''})"
                   f"  val={avg_val:.4f} (lat={avg_val_lat:.4f}, obs={avg_val_obs:.4f}"
                   f"{f', gtcl={avg_val_gtcl:.4f}' if use_gtcl else ''})"
                   f"  best: epoch {best_epoch}, {best_val_loss:.4f}")
            print(msg)

        if epoch % 100 == 0:
            torch.save(model.state_dict(),
                       os.path.join(ckpt_dir, f'foresight_epoch_{epoch}.ckpt'))

        if epoch % 10 == 0:
            n_panels = 4 if use_gtcl else 3
            fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))

            axes[0].plot(train_losses, label='train', alpha=0.7)
            axes[0].plot(val_losses, label='val', alpha=0.7)
            axes[0].set_title('Total Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].legend()

            axes[1].plot(train_latent_losses, label='train', alpha=0.7)
            axes[1].plot(val_latent_losses, label='val', alpha=0.7)
            axes[1].set_title('Latent Loss (L1)')
            axes[1].set_xlabel('Epoch')
            axes[1].legend()

            axes[2].plot(train_obs_losses, label='train', alpha=0.7)
            axes[2].plot(val_obs_losses, label='val', alpha=0.7)
            axes[2].set_title('Obs-Space Aux Loss')
            axes[2].set_xlabel('Epoch')
            axes[2].legend()

            if use_gtcl:
                axes[3].plot(train_gtcl_losses, label='train', alpha=0.7)
                axes[3].plot(val_gtcl_losses, label='val', alpha=0.7)
                axes[3].set_title('G-TCL Loss')
                axes[3].set_xlabel('Epoch')
                axes[3].legend()

            plt.suptitle(f'Latent Foresight Pretrain (best: epoch {best_epoch}, {best_val_loss:.4f})')
            plt.tight_layout()
            plt.savefig(os.path.join(ckpt_dir, 'pretrain_loss.png'), dpi=100)
            plt.close()

    # ---- Final save ----
    torch.save(model.state_dict(), os.path.join(ckpt_dir, 'foresight_last.ckpt'))

    history = {
        'train': train_losses, 'val': val_losses,
        'train_latent': train_latent_losses, 'val_latent': val_latent_losses,
        'train_obs': train_obs_losses, 'val_obs': val_obs_losses,
    }
    if use_gtcl:
        history['train_gtcl'] = train_gtcl_losses
        history['val_gtcl'] = val_gtcl_losses
    with open(os.path.join(ckpt_dir, 'pretrain_history.pkl'), 'wb') as f:
        pickle.dump(history, f)

    # Final plot
    n_panels = 4 if use_gtcl else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    axes[0].plot(train_losses, label='train', alpha=0.7)
    axes[0].plot(val_losses, label='val', alpha=0.7)
    axes[0].set_title('Total Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()

    axes[1].plot(train_latent_losses, label='train', alpha=0.7)
    axes[1].plot(val_latent_losses, label='val', alpha=0.7)
    axes[1].set_title('Latent Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()

    axes[2].plot(train_obs_losses, label='train', alpha=0.7)
    axes[2].plot(val_obs_losses, label='val', alpha=0.7)
    axes[2].set_title('Obs-Space Aux Loss')
    axes[2].set_xlabel('Epoch')
    axes[2].legend()

    if use_gtcl:
        axes[3].plot(train_gtcl_losses, label='train', alpha=0.7)
        axes[3].plot(val_gtcl_losses, label='val', alpha=0.7)
        axes[3].set_title('G-TCL Loss')
        axes[3].set_xlabel('Epoch')
        axes[3].legend()

    plt.suptitle(f'Latent Foresight Pretrain (best: epoch {best_epoch}, {best_val_loss:.4f})')
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
