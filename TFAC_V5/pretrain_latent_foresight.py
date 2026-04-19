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

from utils import get_norm_stats, set_seed, load_meta_data
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
                 tactile_vae_latent_dim=16):
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

        # --- ForesightTransformer (trainable) ---
        tactile_out_dim = tactile_vae_latent_dim * 3 * 3  # 16*3*3 = 144
        self.foresight = ForesightTransformer(
            d_model=hidden_dim, action_dim=state_dim,
            num_layers=foresight_layers, nhead=foresight_nheads,
            dim_feedforward=foresight_dim_feedforward, dropout=dropout,
            tactile_out_dim=tactile_out_dim,
            tactile_decoder_type="linear",
            max_history=max_history,
            predict_horizon=predict_horizon)

    def _encode_images(self, images):
        """编码单帧所有相机图像 → tokens."""
        all_cam_features = []
        all_cam_pos = []
        n_vision = 0
        n_tactile = 0

        for cam_id, cam_name in enumerate(self.camera_names):
            if cam_name == 'gelsight' and self.tactile_mode == 'marker':
                # TactileVAE encode: (B, T=8, 9, 9, 2) → (B, 16, 3, 3)
                with torch.no_grad():
                    z_t, _ = self.tactile_vae.encode_single_frame(images[cam_id])

                B = z_t.size(0)
                C = self.tactile_vae_latent_dim
                z_flat = z_t.reshape(B, C, 9)   # (B, 16, 9)
                z_flat = z_flat.permute(2, 0, 1)  # (9, B, 16)
                proj = self.tac_latent_proj(z_flat)  # (9, B, 512)
                pos_flat = self.tac_spatial_pos_embed  # (9, 1, 512)
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
                pos_flat = pos_flat.permute(2, 0, 1)

            all_cam_features.append(proj)
            all_cam_pos.append(pos_flat)

        src = torch.cat(all_cam_features, dim=0)
        pos = torch.cat(all_cam_pos, dim=0)
        return src, pos, n_vision, n_tactile

    def forward(self, images, actions, future_images=None, qpos=None):
        """
        Returns:
            t_hat: (B, 144) — predicted latent
            z_gt:  (B, 144) — GT latent from frozen TactileVAE
        """
        # Encode current frame
        src, pos, n_vision, n_tactile = self._encode_images(images)

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

        # predict_horizon=1: take last frame GT
        if self.predict_horizon == 1 and z_gt is not None:
            z_gt = z_gt[:, -1]  # (B, 144)

        return t_hat_raw, z_gt, future_raw_last


def compute_loss(t_hat, z_gt, model=None, future_raw=None,
                 foresight_change_weight=False, z_current=None):
    """
    Primary: SmoothL1 in latent space.
    Auxiliary: decode z_hat back to marker space via frozen VAE decoder.
    """
    loss_latent = F.smooth_l1_loss(t_hat, z_gt)

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

    # Change weighting (optional)
    total = loss_latent + 0.3 * loss_obs
    return total, loss_latent.item(), loss_obs.item()


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

    meta_data = load_meta_data(dataset_dir, save_dir=save_dir, config_overrides=args)
    num_episodes = meta_data['num_episodes']
    camera_names = meta_data['camera_names']
    state_dim = meta_data['state_dim']
    proprio_key = meta_data['proprio_key']
    action_key = meta_data['action_key']
    tac_side = meta_data['tac_side']
    tac_img_key = meta_data['tac_img_key']

    tactile_mode = args.get('tactile_mode', 'marker')

    norm_stats = get_norm_stats(dataset_dir, num_episodes, chunk_size=0,
                                proprio_key=proprio_key, action_key=action_key,
                                tactile_mode=tactile_mode, tac_side=tac_side)

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
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    stats_path = os.path.join(ckpt_dir, 'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(norm_stats, f)

    train_dataset = ForesightEpisodicDataset(
        train_indices, dataset_dir, camera_names, norm_stats,
        chunk_size=chunk_size, foresight_horizon=foresight_horizon,
        proprio_key=proprio_key, action_key=action_key,
        tac_side=tac_side, tac_img_key=tac_img_key,
        tactile_mode=tactile_mode, history_len=history_len,
        tactile_vae_window=tactile_vae_window)
    val_dataset = ForesightEpisodicDataset(
        val_indices, dataset_dir, camera_names, norm_stats,
        chunk_size=chunk_size, foresight_horizon=foresight_horizon,
        proprio_key=proprio_key, action_key=action_key,
        tac_side=tac_side, tac_img_key=tac_img_key,
        tactile_mode=tactile_mode, history_len=history_len,
        tactile_vae_window=tactile_vae_window)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    # --- Training loop ---
    best_val_loss = float('inf')
    best_epoch = 0
    train_losses = []
    val_losses = []
    train_latent_losses = []
    val_latent_losses = []
    train_obs_losses = []
    val_obs_losses = []

    for epoch in tqdm(range(num_epochs)):
        # ---- Train ----
        model.train()
        model.backbone.eval()
        model.tactile_vae.eval()
        epoch_loss = 0.0
        epoch_latent = 0.0
        epoch_obs = 0.0
        n_train = 0

        for batch in train_loader:
            all_cam_images, qpos_data, action_data, is_pad, future_cam_images, history_cam_images = batch

            images = [img.cuda() for img in all_cam_images]
            qpos = qpos_data.cuda()
            actions = action_data.cuda()
            future_images = [img.cuda() for img in future_cam_images]

            t_hat, z_gt, future_raw = model(images, actions,
                                             future_images=future_images,
                                             qpos=qpos)

            loss, lat_loss, obs_loss = compute_loss(
                t_hat, z_gt, model=model, future_raw=future_raw)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            bs = actions.size(0)
            epoch_loss += loss.item() * bs
            epoch_latent += lat_loss * bs
            epoch_obs += obs_loss * bs
            n_train += bs

        avg_train = epoch_loss / n_train
        avg_train_lat = epoch_latent / n_train
        avg_train_obs = epoch_obs / n_train
        train_losses.append(avg_train)
        train_latent_losses.append(avg_train_lat)
        train_obs_losses.append(avg_train_obs)

        # ---- Validate ----
        model.eval()
        epoch_loss = 0.0
        epoch_latent = 0.0
        epoch_obs = 0.0
        n_val = 0

        with torch.no_grad():
            for batch in val_loader:
                all_cam_images, qpos_data, action_data, is_pad, future_cam_images, history_cam_images = batch
                images = [img.cuda() for img in all_cam_images]
                qpos = qpos_data.cuda()
                actions = action_data.cuda()
                future_images = [img.cuda() for img in future_cam_images]

                t_hat, z_gt, future_raw = model(images, actions,
                                                 future_images=future_images,
                                                 qpos=qpos)

                loss, lat_loss, obs_loss = compute_loss(
                    t_hat, z_gt, model=model, future_raw=future_raw)

                bs = actions.size(0)
                epoch_loss += loss.item() * bs
                epoch_latent += lat_loss * bs
                epoch_obs += obs_loss * bs
                n_val += bs

        avg_val = epoch_loss / n_val
        avg_val_lat = epoch_latent / n_val
        avg_val_obs = epoch_obs / n_val
        val_losses.append(avg_val)
        val_latent_losses.append(avg_val_lat)
        val_obs_losses.append(avg_val_obs)

        # ---- Logging ----
        is_best = avg_val < best_val_loss
        if is_best:
            best_val_loss = avg_val
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(ckpt_dir, 'foresight_best.ckpt'))

        if epoch % 10 == 0:
            print(f"\nEpoch {epoch}: train={avg_train:.4f} (lat={avg_train_lat:.4f}, obs={avg_train_obs:.4f})"
                  f"  val={avg_val:.4f} (lat={avg_val_lat:.4f}, obs={avg_val_obs:.4f})"
                  f"  best: epoch {best_epoch}, {best_val_loss:.4f}")

        if epoch % 100 == 0:
            torch.save(model.state_dict(),
                       os.path.join(ckpt_dir, f'foresight_epoch_{epoch}.ckpt'))

            # Plot
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            axes[0].plot(train_losses, label='train', alpha=0.7)
            axes[0].plot(val_losses, label='val', alpha=0.7)
            axes[0].set_title('Total Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].legend()

            axes[1].plot(train_latent_losses, label='train', alpha=0.7)
            axes[1].plot(val_latent_losses, label='val', alpha=0.7)
            axes[1].set_title('Latent Loss (SmoothL1)')
            axes[1].set_xlabel('Epoch')
            axes[1].legend()

            axes[2].plot(train_obs_losses, label='train', alpha=0.7)
            axes[2].plot(val_obs_losses, label='val', alpha=0.7)
            axes[2].set_title('Obs-Space Aux Loss')
            axes[2].set_xlabel('Epoch')
            axes[2].legend()

            plt.suptitle(f'Latent Foresight Pretrain (best: epoch {best_epoch}, {best_val_loss:.4f})')
            plt.tight_layout()
            plt.savefig(os.path.join(ckpt_dir, 'pretrain_loss.png'), dpi=100)
            plt.close()

    # ---- Final save ----
    torch.save(model.state_dict(), os.path.join(ckpt_dir, 'foresight_last.ckpt'))

    with open(os.path.join(ckpt_dir, 'pretrain_history.pkl'), 'wb') as f:
        pickle.dump({
            'train': train_losses, 'val': val_losses,
            'train_latent': train_latent_losses, 'val_latent': val_latent_losses,
            'train_obs': train_obs_losses, 'val_obs': val_obs_losses,
        }, f)

    # Final plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
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
