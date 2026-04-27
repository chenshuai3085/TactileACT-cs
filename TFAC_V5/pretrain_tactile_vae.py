"""
TactileVAE 预训练脚本

从 HDF5 数据集中提取连续触觉序列，训练 TactileVAE 学习结构化的触觉 latent space。
预训练完成后冻结 encoder，供下游 ForesightTransformer 在 latent space 中预测。

用法:
    python TFAC_V5/pretrain_tactile_vae.py \
        --data_dirs /home/chenshuai/data/dataset/0414 /home/chenshuai/data/dataset/0401 \
        --output_dir /home/chenshuai/Project/output/tactile_vae \
        --epochs 200 --batch_size 128 --lr 1e-4

    # 使用单个数据集
    python TFAC_V5/pretrain_tactile_vae.py \
        --data_dirs /home/chenshuai/data/dataset/0414 \
        --output_dir /home/chenshuai/Project/output/tactile_vae
"""

import os
import sys
import argparse
import time
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import h5py

import matplotlib
matplotlib.use('Agg')  # 无头模式，服务器不需要 display
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from tactile_vae import TactileVAE, build_tactile_vae


# =============================================================================
# Dataset: 从 HDF5 中提取连续触觉序列
# =============================================================================

class TactileSequenceDataset(Dataset):
    """
    从 HDF5 数据集中提取连续的 marker_offset 序列，用于 TactileVAE 预训练。

    每个样本是一个长度为 T 的连续触觉帧序列 (T, 9, 9, 2)。
    同时使用左右两个触觉传感器的数据（数据量翻倍）。

    支持可选的逐通道归一化。
    """

    def __init__(self, data_dirs, temporal_window=8, stride=4,
                 sides=('left', 'right'), normalize=True):
        """
        Args:
            data_dirs: list of dataset directories containing episode_*.hdf5
            temporal_window: T, 序列长度
            stride: 采样步长（每隔 stride 帧取一个起点），控制数据量
            sides: 使用哪些触觉传感器 ('left', 'right')
            normalize: 是否做逐通道归一化
        """
        super().__init__()
        self.temporal_window = temporal_window
        self.normalize = normalize

        # 收集所有触觉数据
        print("Loading tactile data for TactileVAE pretraining...")
        self.sequences = []  # list of (episode_data_ref, start_idx, side)
        self.all_data = []   # list of numpy arrays, each (T_ep, 9, 9, 2)

        total_samples = 0
        n_skipped = 0
        for data_dir in data_dirs:
            if not os.path.isdir(data_dir):
                print(f"  WARNING: directory not found, skipping: {data_dir}")
                continue
            files = sorted([f for f in os.listdir(data_dir) if f.endswith('.hdf5')])
            for fname in files:
                fpath = os.path.join(data_dir, fname)
                try:
                    with h5py.File(fpath, 'r') as h5:
                        for side in sides:
                            key = f'observations/tac/{side}/marker_offset'
                            if key not in h5:
                                continue
                            data = h5[key][()].astype(np.float32)  # (T_ep, 9, 9, 2)
                            T_ep = data.shape[0]

                            if T_ep < temporal_window:
                                continue

                            data_idx = len(self.all_data)
                            self.all_data.append(data)

                            # 生成所有合法的起始位置
                            for start in range(0, T_ep - temporal_window + 1, stride):
                                self.sequences.append((data_idx, start))
                                total_samples += 1
                except (OSError, KeyError) as e:
                    n_skipped += 1
                    if n_skipped <= 5:
                        print(f"  WARNING: skipping corrupt file: {fpath} ({e})")

        if n_skipped > 5:
            print(f"  WARNING: {n_skipped} files skipped in total (showing first 5)")
        elif n_skipped > 0:
            print(f"  WARNING: {n_skipped} files skipped due to errors")
        print(f"  Loaded {len(self.all_data)} tactile streams")
        print(f"  Total samples (window={temporal_window}, stride={stride}): {total_samples}")

        # 计算归一化统计量
        if normalize:
            all_cat = np.concatenate([d.reshape(-1, 2) for d in self.all_data], axis=0)
            self.mean = all_cat.mean(axis=0).astype(np.float32)  # (2,)
            self.std = all_cat.std(axis=0).astype(np.float32) + 1e-6
            print(f"  Normalization: mean={self.mean}, std={self.std}")
        else:
            self.mean = np.zeros(2, dtype=np.float32)
            self.std = np.ones(2, dtype=np.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        data_idx, start = self.sequences[idx]
        seq = self.all_data[data_idx][start:start + self.temporal_window].copy()  # (T, 9, 9, 2)

        if self.normalize:
            seq = (seq - self.mean) / self.std

        return torch.from_numpy(seq)  # (T, 9, 9, 2)


# =============================================================================
# Quiver Plot 可视化: GT vs Recon 向量场对比
# =============================================================================

def compute_direction_metrics(gt, recon):
    """
    计算方向性指标，用于监控重建质量。

    Args:
        gt:    (9, 9, 2) numpy array
        recon: (9, 9, 2) numpy array
    Returns:
        dict with:
          - cosine_sim: 平均余弦相似度 (1.0=完美, -1.0=完全反向)
          - angular_error_deg: 平均角度误差 (度)
          - magnitude_ratio: |recon| / |gt| 的中位数 (1.0=长度一致)
    """
    gt_flat = gt.reshape(-1, 2)
    recon_flat = recon.reshape(-1, 2)

    gt_norm = np.linalg.norm(gt_flat, axis=1, keepdims=True) + 1e-8
    recon_norm = np.linalg.norm(recon_flat, axis=1, keepdims=True) + 1e-8

    # 余弦相似度
    cos_sim = (gt_flat * recon_flat).sum(axis=1) / (gt_norm.squeeze() * recon_norm.squeeze())
    cos_sim = np.clip(cos_sim, -1.0, 1.0)

    # 角度误差
    angular_error = np.arccos(cos_sim) * 180.0 / np.pi  # degrees

    # 过滤掉 GT 接近零的点 (静止区域，方向无意义)
    gt_mag = gt_norm.squeeze()
    active_mask = gt_mag > 0.05 * gt_mag.max()  # 只看有明显位移的点

    if active_mask.sum() < 3:
        # 几乎没有运动，跳过方向指标
        return {
            'cosine_sim': 0.0,
            'angular_error_deg': 0.0,
            'magnitude_ratio': 1.0,
            'active_ratio': 0.0,
        }

    # 幅值比
    mag_ratio = recon_norm.squeeze()[active_mask] / gt_mag[active_mask]

    return {
        'cosine_sim': float(cos_sim[active_mask].mean()),
        'angular_error_deg': float(angular_error[active_mask].mean()),
        'magnitude_ratio': float(np.median(mag_ratio)),
        'active_ratio': float(active_mask.mean()),
    }


def visualize_quiver(gt_batch, recon_batch, epoch, output_dir,
                     mean=None, std=None, n_samples=4):
    """
    绘制 GT vs Recon 的 9×9 Quiver Plot (向量场对比图)。

    每个样本一行: 左=GT, 中=Recon, 右=Error (差值向量)
    箭头颜色编码幅值大小，便于看长度是否一致。

    Args:
        gt_batch:   (B, T', 9, 9, 2) tensor
        recon_batch: (B, T', 9, 9, 2) tensor
        epoch: 当前 epoch
        output_dir: 保存路径
        mean, std: 归一化参数 (用于反归一化显示真实尺度)
        n_samples: 展示几个样本
    """
    B = gt_batch.shape[0]
    n_samples = min(n_samples, B)

    # 取中间时间帧
    T_prime = gt_batch.shape[1]
    t_mid = T_prime // 2

    gt_np = gt_batch[:n_samples, t_mid].cpu().numpy()       # (n, 9, 9, 2)
    recon_np = recon_batch[:n_samples, t_mid].cpu().numpy()  # (n, 9, 9, 2)

    # 反归一化 (如果需要)
    if mean is not None and std is not None:
        gt_np = gt_np * std + mean
        recon_np = recon_np * std + mean

    # 9×9 网格坐标
    x = np.arange(9)
    y = np.arange(9)
    X, Y = np.meshgrid(x, y)

    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 4 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    for i in range(n_samples):
        gt_i = gt_np[i]       # (9, 9, 2)
        recon_i = recon_np[i]  # (9, 9, 2)
        err_i = recon_i - gt_i

        # 方向指标
        metrics = compute_direction_metrics(gt_i, recon_i)

        # 统一箭头缩放: 用 GT 的最大幅值
        gt_mag = np.sqrt(gt_i[..., 0]**2 + gt_i[..., 1]**2)
        recon_mag = np.sqrt(recon_i[..., 0]**2 + recon_i[..., 1]**2)
        err_mag = np.sqrt(err_i[..., 0]**2 + err_i[..., 1]**2)
        max_mag = max(gt_mag.max(), 1e-6)

        # --- GT ---
        ax = axes[i, 0]
        q = ax.quiver(X, Y, gt_i[..., 0], gt_i[..., 1],
                       gt_mag, cmap='hot', scale=max_mag * 12,
                       scale_units='width', width=0.015)
        ax.set_title(f'GT (sample {i})', fontsize=11)
        ax.set_xlim(-0.5, 8.5)
        ax.set_ylim(-0.5, 8.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        plt.colorbar(q, ax=ax, fraction=0.046, pad=0.04)

        # --- Recon ---
        ax = axes[i, 1]
        q = ax.quiver(X, Y, recon_i[..., 0], recon_i[..., 1],
                       recon_mag, cmap='hot', scale=max_mag * 12,
                       scale_units='width', width=0.015)
        ax.set_title(f'Recon  cos={metrics["cosine_sim"]:.3f}  '
                     f'ang={metrics["angular_error_deg"]:.1f}°  '
                     f'mag={metrics["magnitude_ratio"]:.2f}x',
                     fontsize=10)
        ax.set_xlim(-0.5, 8.5)
        ax.set_ylim(-0.5, 8.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        plt.colorbar(q, ax=ax, fraction=0.046, pad=0.04)

        # --- Error (用 GT 同一 colorbar 范围, 方便对比) ---
        ax = axes[i, 2]
        q = ax.quiver(X, Y, err_i[..., 0], err_i[..., 1],
                       err_mag, cmap='cool', scale=max_mag * 12,
                       scale_units='width', width=0.015,
                       clim=[0, max_mag])
        ax.set_title(f'Error (max={err_mag.max():.3f} / GT_max={max_mag:.3f})',
                     fontsize=10)
        ax.set_xlim(-0.5, 8.5)
        ax.set_ylim(-0.5, 8.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        plt.colorbar(q, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f'Epoch {epoch} | Quiver Plot: GT vs Recon (t={t_mid}/{T_prime})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    save_path = os.path.join(vis_dir, f'quiver_epoch{epoch:04d}.png')
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)

    return save_path, metrics


# =============================================================================
# 训练循环
# =============================================================================

def train_one_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_dir = 0.0
    n_batches = 0

    for batch in dataloader:
        x = batch.to(device)  # (B, T, 9, 9, 2)

        # Forward
        recon, mu, logvar = model(x)

        # Align GT with temporally downsampled latent
        gt = TactileVAE.align_gt(x, temporal_stride=2)

        # Loss
        loss, recon_loss, kl_loss, dir_loss = model.loss(recon, gt, mu, logvar)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        total_dir += dir_loss.item()
        n_batches += 1

    return {
        'loss': total_loss / n_batches,
        'recon': total_recon / n_batches,
        'kl': total_kl / n_batches,
        'dir': total_dir / n_batches,
    }


@torch.no_grad()
def validate(model, dataloader, device):
    """
    验证并返回方向性指标 + 一组样本用于可视化。

    Returns:
        metrics: dict with loss, recon, kl, cosine_sim, angular_error_deg
        vis_data: (gt_batch, recon_batch) 用于 quiver plot, 或 None
    """
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_cos = 0.0
    total_ang = 0.0
    n_batches = 0
    vis_gt = None
    vis_recon = None

    for batch in dataloader:
        x = batch.to(device)
        recon, mu, logvar = model(x)
        gt = TactileVAE.align_gt(x, temporal_stride=2)
        loss, recon_loss, kl_loss, dir_loss = model.loss(recon, gt, mu, logvar)

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()

        # 方向指标 (在第一个 batch 上算，避免每 batch 都算太慢)
        if n_batches == 0:
            vis_gt = gt.detach()
            vis_recon = recon.detach()
            # 计算 batch 平均方向指标
            B = gt.shape[0]
            T_mid = gt.shape[1] // 2
            cos_list, ang_list = [], []
            for b in range(min(B, 16)):  # 最多看 16 个样本
                m = compute_direction_metrics(
                    gt[b, T_mid].cpu().numpy(),
                    recon[b, T_mid].cpu().numpy()
                )
                if m['active_ratio'] > 0:
                    cos_list.append(m['cosine_sim'])
                    ang_list.append(m['angular_error_deg'])
            total_cos = np.mean(cos_list) if cos_list else 0.0
            total_ang = np.mean(ang_list) if ang_list else 0.0

        n_batches += 1

    return {
        'loss': total_loss / n_batches,
        'recon': total_recon / n_batches,
        'kl': total_kl / n_batches,
        'cosine_sim': float(total_cos),
        'angular_error_deg': float(total_ang),
    }, (vis_gt, vis_recon)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='TactileVAE Pretraining')
    # Data
    parser.add_argument('--data_dirs', type=str, nargs='+', required=True,
                        help='Dataset directories containing episode_*.hdf5')
    parser.add_argument('--output_dir', type=str,
                        default='/home/chenshuai/Project/output/tactile_vae',
                        help='Output directory for checkpoints')
    parser.add_argument('--sides', type=str, nargs='+', default=['left', 'right'],
                        help='Tactile sensor sides to use')

    # Model
    parser.add_argument('--latent_dim', type=int, default=8,
                        help='Latent channels C (total latent = C * 3 * 3)')
    parser.add_argument('--temporal_window', type=int, default=8,
                        help='Input sequence length T')
    parser.add_argument('--num_freqs', type=int, default=4,
                        help='Fourier positional encoding levels for INR')
    parser.add_argument('--inr_hidden', type=int, default=64,
                        help='INR MLP hidden dim')
    parser.add_argument('--kl_weight', type=float, default=1e-6,
                        help='KL divergence loss weight')
    parser.add_argument('--direction_weight', type=float, default=0.2,
                        help='Cosine direction loss weight')

    # Training
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--sample_stride', type=int, default=4,
                        help='Stride for sampling sequences from episodes')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Validation data ratio')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_every', type=int, default=50,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--normalize', action='store_true', default=True,
                        help='Normalize marker_offset per channel')
    parser.add_argument('--no_normalize', dest='normalize', action='store_false')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--vis_every', type=int, default=10,
                        help='Save quiver plot every N epochs')
    parser.add_argument('--vis_samples', type=int, default=4,
                        help='Number of samples in quiver plot')

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Save config
    config = vars(args)
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {args.output_dir}/config.json")

    # =========================================================================
    # Dataset
    # =========================================================================
    full_dataset = TactileSequenceDataset(
        data_dirs=args.data_dirs,
        temporal_window=args.temporal_window,
        stride=args.sample_stride,
        sides=args.sides,
        normalize=args.normalize,
    )

    # Train/val split
    n_total = len(full_dataset)
    n_val = max(1, int(n_total * args.val_ratio))
    n_train = n_total - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )
    print(f"Train: {n_train}, Val: {n_val}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # =========================================================================
    # Model
    # =========================================================================
    model = TactileVAE(
        latent_dim=args.latent_dim,
        temporal_window=args.temporal_window,
        num_freqs=args.num_freqs,
        inr_hidden=args.inr_hidden,
        kl_weight=args.kl_weight,
        direction_weight=args.direction_weight,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    # Optimizer & scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # Save normalization stats (needed when loading pretrained model)
    norm_stats = {
        'mean': full_dataset.mean.tolist(),
        'std': full_dataset.std.tolist(),
    }
    with open(os.path.join(args.output_dir, 'tactile_norm_stats.json'), 'w') as f:
        json.dump(norm_stats, f, indent=2)

    # =========================================================================
    # Training
    # =========================================================================
    best_val_loss = float('inf')
    history = []

    print(f"\n{'='*60}")
    print(f"Starting TactileVAE pretraining")
    print(f"  latent_dim={args.latent_dim}, total_latent={args.latent_dim*9}/frame")
    print(f"  temporal_window={args.temporal_window}")
    print(f"  epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, epoch)

        # Validate (returns metrics + vis data)
        val_metrics, vis_data = validate(model, val_loader, device)

        # Step scheduler
        scheduler.step()

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]['lr']

        # Log
        log = {
            'epoch': epoch,
            'lr': lr_now,
            'train_loss': train_metrics['loss'],
            'train_recon': train_metrics['recon'],
            'train_kl': train_metrics['kl'],
            'train_dir': train_metrics['dir'],
            'val_loss': val_metrics['loss'],
            'val_recon': val_metrics['recon'],
            'val_kl': val_metrics['kl'],
            'val_cosine_sim': val_metrics['cosine_sim'],
            'val_angular_error': val_metrics['angular_error_deg'],
            'time': elapsed,
        }
        history.append(log)

        # 实时写入 history (每 epoch 都写, 方便训练中画图)
        history_path = os.path.join(args.output_dir, 'train_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        # Print (含方向指标)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"[Epoch {epoch:3d}/{args.epochs}] "
                  f"train: {train_metrics['loss']:.6f} "
                  f"(recon={train_metrics['recon']:.6f} dir={train_metrics['dir']:.4f}) | "
                  f"val: {val_metrics['loss']:.6f} "
                  f"(recon={val_metrics['recon']:.6f}) | "
                  f"cos={val_metrics['cosine_sim']:.3f} "
                  f"ang={val_metrics['angular_error_deg']:.1f}° | "
                  f"lr={lr_now:.2e} | {elapsed:.1f}s")

        # Quiver Plot 可视化
        if epoch == 1 or epoch % args.vis_every == 0 or epoch == args.epochs:
            if vis_data[0] is not None:
                # 准备反归一化参数
                vis_mean = full_dataset.mean if full_dataset.normalize else None
                vis_std = full_dataset.std if full_dataset.normalize else None
                vis_path, _ = visualize_quiver(
                    vis_data[0], vis_data[1],
                    epoch=epoch, output_dir=args.output_dir,
                    mean=vis_mean, std=vis_std,
                    n_samples=args.vis_samples,
                )
                if epoch == 1 or epoch % (args.vis_every * 5) == 0:
                    print(f"  → Quiver plot saved: {vis_path}")

        # Save best
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            save_path = os.path.join(args.output_dir, 'best_tactile_vae.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'config': config,
                'norm_stats': norm_stats,
            }, save_path)

        # Save periodic checkpoint
        if epoch % args.save_every == 0:
            ckpt_path = os.path.join(args.output_dir, f'tactile_vae_epoch{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'config': config,
                'norm_stats': norm_stats,
            }, ckpt_path)

    # Save final model
    final_path = os.path.join(args.output_dir, 'final_tactile_vae.pt')
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'config': config,
        'norm_stats': norm_stats,
    }, final_path)

    # Save training history
    with open(os.path.join(args.output_dir, 'train_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # 最终方向指标
    final_cos = history[-1]['val_cosine_sim']
    final_ang = history[-1]['val_angular_error']

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Best val loss:     {best_val_loss:.6f}")
    print(f"  Final cosine sim:  {final_cos:.3f}")
    print(f"  Final angular err: {final_ang:.1f}°")
    print(f"  Best model:        {os.path.join(args.output_dir, 'best_tactile_vae.pt')}")
    print(f"  Final model:       {final_path}")
    print(f"  Visualizations:    {os.path.join(args.output_dir, 'visualizations/')}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
