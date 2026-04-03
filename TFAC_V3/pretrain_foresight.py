"""
TFAC_V3 Stage 1: Foresight 预训练脚本。
只训练 foresight 通路: backbone(input_proj) + MarkerEncoder + ForesightTransformer + SpatialTactileDecoder
Vision backbone (CLIP) 冻结, 用 GT action 作为条件。
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

from utils import get_norm_stats, set_seed
from TFAC_V3.dataset import ForesightEpisodicDataset
from TFAC_V3.foresight_transformer import ForesightTransformer
from TFAC_V3.marker_encoder import build_marker_encoder


class ForesightPretrainModel(nn.Module):
    """
    Stage 1 预训练模型: 只包含 foresight 通路。

    输入: 历史 k 帧 images + GT action → 预测 t+H marker_offset

    训练的模块:
      - input_proj (Conv2d): backbone feature → hidden_dim
      - marker_encoder (PointNet): marker_offset → embedding
      - marker_pos_embed: marker token position embedding
      - foresight (ForesightTransformer): 预测未来触觉

    冻结的模块:
      - vision backbone (CLIP ResNet18)
    """

    def __init__(self, backbone, camera_names, cam_backbone_mapping,
                 hidden_dim=512, state_dim=7,
                 # foresight params
                 foresight_layers=3, foresight_nheads=8,
                 foresight_dim_feedforward=2048, dropout=0.1,
                 tactile_mode="marker", marker_encoder_type="pointnet",
                 foresight_tac_decoder="spatial", spatial_tac_dec_layers=3,
                 max_history=8,
                 foresight_change_weight=False):
        super().__init__()

        self.camera_names = camera_names
        self.cam_backbone_mapping = cam_backbone_mapping
        self.hidden_dim = hidden_dim
        self.tactile_mode = tactile_mode
        self.foresight_change_weight = foresight_change_weight

        # Vision backbone (frozen)
        self.backbone = nn.ModuleList([backbone])
        self.backbone.requires_grad_(False)

        # input_proj (trainable)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)

        # Marker encoder (trainable)
        self.marker_encoder = None
        if tactile_mode == "marker":
            self.marker_encoder = build_marker_encoder(marker_encoder_type, hidden_dim=hidden_dim)
            self.marker_pos_embed = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # Foresight (trainable)
        tactile_out_dim = 9 * 9 * 2 if tactile_mode == "marker" else hidden_dim
        self.foresight = ForesightTransformer(
            d_model=hidden_dim, action_dim=state_dim,
            num_layers=foresight_layers, nhead=foresight_nheads,
            dim_feedforward=foresight_dim_feedforward, dropout=dropout,
            tactile_out_dim=tactile_out_dim,
            tactile_decoder_type=foresight_tac_decoder,
            spatial_tac_dec_layers=spatial_tac_dec_layers,
            max_history=max_history)

    def _encode_images(self, images):
        """编码单帧所有相机图像 → tokens."""
        all_cam_features = []
        all_cam_pos = []
        n_vision = 0
        n_tactile = 0

        for cam_id, cam_name in enumerate(self.camera_names):
            if cam_name == 'gelsight' and self.tactile_mode == 'marker':
                marker_feat = self.marker_encoder(images[cam_id])  # (B, D)
                proj = marker_feat.unsqueeze(0)  # (1, B, D)
                pos_flat = self.marker_pos_embed  # (1, 1, D)
                n_tactile += 1
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

            all_cam_features.append(proj)
            all_cam_pos.append(pos_flat)

        src = torch.cat(all_cam_features, dim=0)  # (N_total, B, D)
        pos = torch.cat(all_cam_pos, dim=0)
        return src, pos, n_vision, n_tactile

    def _encode_history(self, history_images):
        """编码历史 k 帧, 返回 (k, N_total, B, D)."""
        num_cams = len(self.camera_names)
        k = history_images[0].shape[1]  # (B, k, C, H, W)

        results = []
        with torch.no_grad():
            for t in range(k):
                frame_images = [history_images[cam_idx][:, t] for cam_idx in range(num_cams)]
                src, pos, n_v, n_t = self._encode_images(frame_images)
                results.append(src)

        return torch.stack(results)  # (k, N_total, B, D)

    def forward(self, images, actions, history_images=None, future_images=None):
        """
        Args:
            images: list of (B, C, H, W) per camera — 当前帧
            actions: (B, chunk_size, action_dim) — GT action
            history_images: list of (B, k, C, H, W) per camera — 历史帧
            future_images: list of (B, ...) per camera — t+H 的 GT
        Returns:
            t_hat: (B, 9, 9, 2) — 预测的 marker_offset
            t_gt: (B, 9, 9, 2) — GT marker_offset
        """
        bs = images[0].size(0)

        # Encode current frame
        src, pos, n_vision, n_tactile = self._encode_images(images)

        # Foresight prediction
        if history_images is not None and history_images[0].shape[1] > 1:
            hist_src = self._encode_history(history_images)
            # Replace last frame with current (preserves gradients for input_proj/marker_encoder)
            hist_src = torch.cat([hist_src[:-1], src.unsqueeze(0)], dim=0)
            v_tokens = hist_src[:, :n_vision]  # (k, N_v, B, D)
            t_tokens = hist_src[:, n_vision:]  # (k, N_t, B, D)
        else:
            v_tokens = src[:n_vision]   # (N_v, B, D)
            t_tokens = src[n_vision:]   # (N_t, B, D)

        t_hat_raw, v_hat_future = self.foresight(
            v_tokens, t_tokens, actions, n_vision)

        # Reshape prediction
        if self.tactile_mode == "marker":
            t_hat = t_hat_raw.view(bs, 9, 9, 2)
        else:
            t_hat = t_hat_raw

        # GT future tactile
        t_gt = None
        if future_images is not None:
            for cam_id, cam_name in enumerate(self.camera_names):
                if cam_name == 'gelsight' and self.tactile_mode == 'marker':
                    t_gt = future_images[cam_id]  # (B, 9, 9, 2)
                    break

        return t_hat, t_gt


def compute_loss(t_hat, t_gt, foresight_change_weight=False, t_current=None):
    """
    计算 foresight tactile loss (smooth_L1)。
    可选: foresight_change_weight 根据触觉变化量加权。
    """
    t_hat_flat = t_hat.view(t_hat.size(0), -1)  # (B, 162)
    t_gt_flat = t_gt.view(t_gt.size(0), -1)     # (B, 162)

    per_sample = F.smooth_l1_loss(t_hat_flat, t_gt_flat, reduction='none').mean(dim=1)  # (B,)

    if foresight_change_weight and t_current is not None:
        t_cur_flat = t_current.view(t_current.size(0), -1)
        delta = (t_gt_flat - t_cur_flat).abs().mean(dim=1)  # (B,)
        weight = 1.0 + delta
        weight = weight / weight.mean()
        loss = (weight * per_sample).mean()
    else:
        loss = per_sample.mean()

    return loss


def main(args):
    save_dir = args['save_dir']
    model_name = args['name']
    batch_size = args['batch_size']
    num_epochs = args['num_epochs']
    chunk_size = args['chunk_size']
    seed = args['seed']
    gpu = args['gpu']

    ckpt_dir = os.path.join(save_dir, model_name)
    dataset_dir = os.path.join(save_dir, 'data')
    assert os.path.exists(save_dir), f'{save_dir} does not exist.'

    with open(os.path.join(save_dir, 'meta_data.json'), 'r') as f:
        meta_data = json.load(f)

    num_episodes = meta_data['num_episodes']
    camera_names = meta_data['camera_names']
    state_dim = meta_data['state_dim']
    proprio_key = meta_data.get('proprio_key', 'qpos')
    action_key = meta_data.get('action_key', 'action')
    tac_side = meta_data.get('tac_side', 'left')
    tac_img_key = meta_data.get('tac_img_key', 'img')

    tactile_mode = args.get('tactile_mode', 'marker')

    norm_stats = get_norm_stats(dataset_dir, num_episodes, chunk_size=0,
                                proprio_key=proprio_key, action_key=action_key,
                                tactile_mode=tactile_mode, tac_side=tac_side)
    args['norm_stats'] = {k: v.tolist() for k, v in norm_stats.items()}

    set_seed(seed)
    if gpu != -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    # --- Load vision backbone (frozen) ---
    if args['backbone'] == "clip_backbone":
        try:
            from clip_pretraining_xiaomi import modified_resnet18
        except ImportError:
            from clip_pretraining import modified_resnet18
        vision_model = modified_resnet18()
        if args.get('vision_backbone_path', 'none') != 'none':
            vision_model.load_state_dict(torch.load(args['vision_backbone_path']))
        cam_backbone_mapping = {cam_name: 0 for cam_name in camera_names}
    else:
        raise ValueError(f"Unsupported backbone: {args['backbone']}")

    # --- Load CLIP-pretrained tactile encoder ---
    pretrain_tac_path = args.get('pretrain_tactile_encoder_path', None)

    # --- Build pretrain model ---
    model = ForesightPretrainModel(
        backbone=vision_model,
        camera_names=camera_names,
        cam_backbone_mapping=cam_backbone_mapping,
        hidden_dim=args['hidden_dim'],
        state_dim=state_dim,
        foresight_layers=args.get('foresight_layers', 3),
        foresight_nheads=args.get('foresight_nheads', 8),
        foresight_dim_feedforward=args.get('foresight_dim_feedforward', 2048),
        dropout=args.get('dropout', 0.1),
        tactile_mode=tactile_mode,
        marker_encoder_type=args.get('marker_encoder_type', 'pointnet'),
        foresight_tac_decoder=args.get('foresight_tac_decoder', 'spatial'),
        spatial_tac_dec_layers=args.get('spatial_tac_dec_layers', 3),
        max_history=args.get('max_history', 8),
        foresight_change_weight=args.get('foresight_change_weight', False),
    )
    # Load CLIP-pretrained tactile encoder (PointNet) if available
    if pretrain_tac_path and os.path.exists(pretrain_tac_path):
        clip_tac_state = torch.load(pretrain_tac_path)
        # CLIP PointNetEncoder → model.marker_encoder (same architecture)
        missing, unexpected = model.marker_encoder.load_state_dict(clip_tac_state, strict=False)
        print(f"Loaded CLIP tactile encoder from: {pretrain_tac_path}")
        if missing:
            print(f"  Missing keys: {missing}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected}")
    elif pretrain_tac_path:
        print(f"WARNING: pretrain_tactile_encoder_path not found: {pretrain_tac_path}")

    model.cuda()

    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Foresight pretrain: {trainable/1e6:.2f}M trainable / {total/1e6:.2f}M total")

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
    history_len = args.get('history_len', 3)
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
        tactile_mode=tactile_mode, history_len=history_len)
    val_dataset = ForesightEpisodicDataset(
        val_indices, dataset_dir, camera_names, norm_stats,
        chunk_size=chunk_size, foresight_horizon=foresight_horizon,
        proprio_key=proprio_key, action_key=action_key,
        tac_side=tac_side, tac_img_key=tac_img_key,
        tactile_mode=tactile_mode, history_len=history_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    # --- Training loop ---
    foresight_change_weight = args.get('foresight_change_weight', False)
    best_val_loss = float('inf')
    best_epoch = 0
    train_losses = []
    val_losses = []

    for epoch in tqdm(range(num_epochs)):
        # ---- Train ----
        model.train()
        model.backbone.eval()  # keep backbone in eval mode (BN, dropout)
        epoch_train_loss = 0.0
        n_train = 0

        for batch in train_loader:
            # Unpack: (all_cam_images, qpos, action, is_pad, future_cam_images, history_cam_images)
            all_cam_images, qpos_data, action_data, is_pad, future_cam_images, history_cam_images = batch

            # Move to GPU
            images = [img.cuda() for img in all_cam_images]
            actions = action_data.cuda()
            future_images = [img.cuda() for img in future_cam_images]
            history_imgs = [img.cuda() for img in history_cam_images]

            # Current tactile for change_weight
            t_current = None
            if foresight_change_weight:
                for cam_id, cam_name in enumerate(camera_names):
                    if cam_name == 'gelsight' and tactile_mode == 'marker':
                        t_current = images[cam_id]  # (B, 9, 9, 2)
                        break

            # Forward
            t_hat, t_gt = model(images, actions,
                                history_images=history_imgs,
                                future_images=future_images)

            loss = compute_loss(t_hat, t_gt,
                                foresight_change_weight=foresight_change_weight,
                                t_current=t_current)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_train_loss += loss.item() * actions.size(0)
            n_train += actions.size(0)

        avg_train_loss = epoch_train_loss / n_train
        train_losses.append(avg_train_loss)

        # ---- Validate ----
        model.eval()
        epoch_val_loss = 0.0
        n_val = 0

        with torch.no_grad():
            for batch in val_loader:
                all_cam_images, qpos_data, action_data, is_pad, future_cam_images, history_cam_images = batch
                images = [img.cuda() for img in all_cam_images]
                actions = action_data.cuda()
                future_images = [img.cuda() for img in future_cam_images]
                history_imgs = [img.cuda() for img in history_cam_images]

                t_current = None
                if foresight_change_weight:
                    for cam_id, cam_name in enumerate(camera_names):
                        if cam_name == 'gelsight' and tactile_mode == 'marker':
                            t_current = images[cam_id]
                            break

                t_hat, t_gt = model(images, actions,
                                    history_images=history_imgs,
                                    future_images=future_images)
                loss = compute_loss(t_hat, t_gt,
                                    foresight_change_weight=foresight_change_weight,
                                    t_current=t_current)

                epoch_val_loss += loss.item() * actions.size(0)
                n_val += actions.size(0)

        avg_val_loss = epoch_val_loss / n_val
        val_losses.append(avg_val_loss)

        # ---- Logging ----
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(ckpt_dir, 'foresight_best.ckpt'))

        if epoch % 10 == 0:
            print(f"\nEpoch {epoch}: train={avg_train_loss:.4f}, val={avg_val_loss:.4f}"
                  f" (best: epoch {best_epoch}, {best_val_loss:.4f})")

        # Save checkpoint every 100 epochs
        if epoch % 100 == 0:
            torch.save(model.state_dict(),
                       os.path.join(ckpt_dir, f'foresight_epoch_{epoch}.ckpt'))

            # Plot
            plt.figure(figsize=(8, 5))
            plt.plot(train_losses, label='train', alpha=0.7)
            plt.plot(val_losses, label='val', alpha=0.7)
            plt.xlabel('Epoch')
            plt.ylabel('Foresight Tac Loss')
            plt.title('Foresight Pretraining')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(ckpt_dir, 'pretrain_loss.png'), dpi=100)
            plt.close()

    # ---- Final save ----
    torch.save(model.state_dict(), os.path.join(ckpt_dir, 'foresight_last.ckpt'))

    with open(os.path.join(ckpt_dir, 'pretrain_history.pkl'), 'wb') as f:
        pickle.dump({'train': train_losses, 'val': val_losses}, f)

    # Final plot
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='train', alpha=0.7)
    plt.plot(val_losses, label='val', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Foresight Tac Loss')
    plt.title(f'Foresight Pretraining (best: epoch {best_epoch}, {best_val_loss:.4f})')
    plt.legend()
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
