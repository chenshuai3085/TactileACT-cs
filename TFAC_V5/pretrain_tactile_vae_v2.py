"""
TactileVAE v2 预训练脚本

基于 v1 预训练脚本改造，适配 v2 的新架构：
  - Cross-Attention Decoder (替代 INR)
  - Temporal Attention Pooling (替代 take-last-frame)
  - Intensity-Pattern Disentangled Latent

输出单帧重建 (temporal attention pooling 聚合后)，GT 使用最后一帧。

用法:
    python TFAC_V5/pretrain_tactile_vae_v2.py \
        --data_dirs /media/chenshuai/czy/pih_dataset/260112/success \
                    /media/chenshuai/czy/pih_dataset/260113/success \
        --output_dir /home/chenshuai/Project/output/tactile_vae_v2 \
        --epochs 200 --batch_size 512 --lr 1e-4

    # 使用 v1 相同的全量数据
    python TFAC_V5/pretrain_tactile_vae_v2.py \
        --data_dirs /media/chenshuai/czy/pih_dataset/260112/success \
                    /media/chenshuai/czy/pih_dataset/260113/success \
                    /media/chenshuai/czy/pih_dataset/260114/success \
                    /media/chenshuai/czy/pih_dataset/260115/success \
        --output_dir /home/chenshuai/Project/output/tactile_vae_v2 \
        --epochs 200 --batch_size 512 --lr 1e-4 --sides left
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
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from tactile_vae_v2 import TactileVAEv2, build_tactile_vae_v2


# =============================================================================
# Dataset: 从 HDF5 中提取连续触觉序列 (复用 v1 的 dataset)
# =============================================================================

class TactileSequenceDataset(Dataset):
    """
    从 HDF5 数据集中提取连续的 marker_offset 序列，用于 TactileVAE 预训练。
    每个样本是一个长度为 T 的连续触觉帧序列 (T, 9, 9, 2)。
    """

    def __init__(self, data_dirs, temporal_window=8, stride=4,
                 sides=('left',), normalize=True):
        super().__init__()
        self.temporal_window = temporal_window
        self.normalize = normalize

        print("Loading tactile data for TactileVAE v2 pretraining...")
        self.sequences = []
        self.all_data = []

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
                            data = h5[key][()].astype(np.float32)
                            T_ep = data.shape[0]

                            if T_ep < temporal_window:
                                continue

                            data_idx = len(self.all_data)
                            self.all_data.append(data)

                            for start in range(0, T_ep - temporal_window + 1, stride):
                                self.sequences.append((data_idx, start))
                                total_samples += 1
                except (OSError, KeyError) as e:
                    n_skipped += 1
                    if n_skipped <= 5:
                        print(f"  WARNING: skipping corrupt file: {fpath} ({e})")

        if n_skipped > 5:
            print(f"  WARNING: {n_skipped} files skipped in total")
        elif n_skipped > 0:
            print(f"  WARNING: {n_skipped} files skipped due to errors")
        print(f"  Loaded {len(self.all_data)} tactile streams")
        print(f"  Total samples (window={temporal_window}, stride={stride}): {total_samples}")

        if normalize:
            all_cat = np.concatenate([d.reshape(-1, 2) for d in self.all_data], axis=0)
            self.mean = all_cat.mean(axis=0).astype(np.float32)
            self.std = all_cat.std(axis=0).astype(np.float32) + 1e-6
            print(f"  Normalization: mean={self.mean}, std={self.std}")
        else:
            self.mean = np.zeros(2, dtype=np.float32)
            self.std = np.ones(2, dtype=np.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        data_idx, start = self.sequences[idx]
        seq = self.all_data[data_idx][start:start + self.temporal_window].copy()

        if self.normalize:
            seq = (seq - self.mean) / self.std

        return torch.from_numpy(seq)


# =============================================================================
# Quiver Plot 可视化
# =============================================================================

def compute_direction_metrics(gt, recon):
    """计算方向性指标: cosine_sim, angular_error, magnitude_ratio."""
    gt_flat = gt.reshape(-1, 2)
    recon_flat = recon.reshape(-1, 2)

    gt_norm = np.linalg.norm(gt_flat, axis=1, keepdims=True) + 1e-8
    recon_norm = np.linalg.norm(recon_flat, axis=1, keepdims=True) + 1e-8

    cos_sim = (gt_flat * recon_flat).sum(axis=1) / (gt_norm.squeeze() * recon_norm.squeeze())
    cos_sim = np.clip(cos_sim, -1.0, 1.0)

    angular_error = np.arccos(cos_sim) * 180.0 / np.pi

    gt_mag = gt_norm.squeeze()
    active_mask = gt_mag > 0.05 * gt_mag.max()

    if active_mask.sum() < 3:
        return {
            'cosine_sim': 0.0,
            'angular_error_deg': 0.0,
            'magnitude_ratio': 1.0,
            'active_ratio': 0.0,
        }

    mag_ratio = recon_norm.squeeze()[active_mask] / gt_mag[active_mask]

    return {
        'cosine_sim': float(cos_sim[active_mask].mean()),
        'angular_error_deg': float(angular_error[active_mask].mean()),
        'magnitude_ratio': float(np.median(mag_ratio)),
        'active_ratio': float(active_mask.mean()),
    }


def visualize_quiver_v2(gt_batch, recon_batch, epoch, output_dir,
                        mean=None, std=None, n_samples=4):
    """
    绘制 GT vs Recon 的 9×9 Quiver Plot。
    v2 输出单帧，gt_batch 和 recon_batch 都是 (B, 9, 9, 2)。
    """
    B = gt_batch.shape[0]
    n_samples = min(n_samples, B)

    gt_np = gt_batch[:n_samples].cpu().numpy()
    recon_np = recon_batch[:n_samples].cpu().numpy()

    if mean is not None and std is not None:
        gt_np = gt_np * std + mean
        recon_np = recon_np * std + mean

    x = np.arange(9)
    y = np.arange(9)
    X, Y = np.meshgrid(x, y)

    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 4 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    for i in range(n_samples):
        gt_i = gt_np[i]
        recon_i = recon_np[i]
        err_i = recon_i - gt_i

        metrics = compute_direction_metrics(gt_i, recon_i)

        gt_mag = np.sqrt(gt_i[..., 0]**2 + gt_i[..., 1]**2)
        recon_mag = np.sqrt(recon_i[..., 0]**2 + recon_i[..., 1]**2)
        err_mag = np.sqrt(err_i[..., 0]**2 + err_i[..., 1]**2)
        max_mag = max(gt_mag.max(), 1e-6)

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

    fig.suptitle(f'Epoch {epoch} | TactileVAE v2: GT vs Recon (last frame)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    save_path = os.path.join(vis_dir, f'quiver_epoch{epoch:04d}.png')
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)

    return save_path, metrics


def visualize_temporal_attention(attn_weights, epoch, output_dir, n_samples=8):
    """
    可视化 temporal attention 分布，看模型关注哪些时间帧。
    attn_weights: (B, 1, T') tensor
    """
    B = attn_weights.shape[0]
    n_samples = min(n_samples, B)
    T_prime = attn_weights.shape[2]

    attn_np = attn_weights[:n_samples, 0].cpu().numpy()  # (n, T')

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    im = ax.imshow(attn_np, aspect='auto', cmap='Blues', vmin=0)
    ax.set_xlabel('Temporal Frame (encoded)')
    ax.set_ylabel('Sample')
    ax.set_title(f'Epoch {epoch} | Temporal Attention Weights')
    ax.set_xticks(range(T_prime))
    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    save_path = os.path.join(vis_dir, f'temporal_attn_epoch{epoch:04d}.png')
    fig.savefig(save_path, dpi=100)
    plt.close(fig)
    return save_path


# =============================================================================
# 训练循环
# =============================================================================

def train_one_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_losses = {'total': 0, 'recon': 0, 'kl': 0, 'direction': 0,
                    'intensity': 0, 'rank': 0}
    n_batches = 0

    for batch in dataloader:
        x = batch.to(device)  # (B, T, 9, 9, 2)

        recon, mu, logvar, z_agg, attn_weights = model(x)

        gt = x[:, -1]  # (B, 9, 9, 2) 最后一帧作为重建目标

        total_loss, losses = model.loss(recon, gt, mu, logvar, z_agg)

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for k, v in losses.items():
            total_losses[k] += v.item()
        n_batches += 1

    return {k: v / n_batches for k, v in total_losses.items()}


@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    total_losses = {'total': 0, 'recon': 0, 'kl': 0, 'direction': 0,
                    'intensity': 0, 'rank': 0}
    n_batches = 0
    vis_gt = None
    vis_recon = None
    vis_attn = None
    cos_list, ang_list = [], []

    for batch in dataloader:
        x = batch.to(device)
        recon, mu, logvar, z_agg, attn_weights = model(x)
        gt = x[:, -1]
        total_loss, losses = model.loss(recon, gt, mu, logvar, z_agg)

        for k, v in losses.items():
            total_losses[k] += v.item()

        if n_batches == 0:
            vis_gt = gt.detach()
            vis_recon = recon.detach()
            vis_attn = attn_weights.detach()

            B = gt.shape[0]
            for b in range(min(B, 16)):
                m = compute_direction_metrics(
                    gt[b].cpu().numpy(),
                    recon[b].cpu().numpy()
                )
                if m['active_ratio'] > 0:
                    cos_list.append(m['cosine_sim'])
                    ang_list.append(m['angular_error_deg'])

        n_batches += 1

    metrics = {k: v / n_batches for k, v in total_losses.items()}
    metrics['cosine_sim'] = float(np.mean(cos_list)) if cos_list else 0.0
    metrics['angular_error_deg'] = float(np.mean(ang_list)) if ang_list else 0.0

    return metrics, (vis_gt, vis_recon, vis_attn)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='TactileVAE v2 Pretraining')
    # Data
    parser.add_argument('--data_dirs', type=str, nargs='+', required=True,
                        help='Dataset directories containing episode_*.hdf5')
    parser.add_argument('--output_dir', type=str,
                        default='/home/chenshuai/Project/output/tactile_vae_v2',
                        help='Output directory for checkpoints')
    parser.add_argument('--sides', type=str, nargs='+', default=['left'],
                        help='Tactile sensor sides to use')

    # Model
    parser.add_argument('--latent_dim', type=int, default=16,
                        help='Latent channels C (total latent = C * 3 * 3)')
    parser.add_argument('--temporal_window', type=int, default=8,
                        help='Input sequence length T')
    parser.add_argument('--decoder_hidden', type=int, default=128,
                        help='Cross-attention decoder hidden dim')
    parser.add_argument('--decoder_heads', type=int, default=4,
                        help='Cross-attention decoder heads')
    parser.add_argument('--decoder_layers', type=int, default=2,
                        help='Cross-attention decoder layers')
    parser.add_argument('--kl_weight', type=float, default=1e-6,
                        help='KL divergence loss weight')
    parser.add_argument('--direction_weight', type=float, default=0.2,
                        help='Cosine direction loss weight')
    parser.add_argument('--intensity_weight', type=float, default=0.1,
                        help='Intensity supervision loss weight')
    parser.add_argument('--rank_weight', type=float, default=0.05,
                        help='Pairwise ranking loss weight')

    # Training
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=512)
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

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    config = vars(args)
    config['model_version'] = 'v2'
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
    model = TactileVAEv2(
        latent_dim=args.latent_dim,
        temporal_window=args.temporal_window,
        decoder_hidden=args.decoder_hidden,
        decoder_heads=args.decoder_heads,
        decoder_layers=args.decoder_layers,
        kl_weight=args.kl_weight,
        direction_weight=args.direction_weight,
        intensity_weight=args.intensity_weight,
        rank_weight=args.rank_weight,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # Save normalization stats
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
    print(f"Starting TactileVAE v2 pretraining")
    print(f"  latent_dim={args.latent_dim}, total_latent={args.latent_dim*9}")
    print(f"  temporal_window={args.temporal_window}")
    print(f"  decoder: hidden={args.decoder_hidden}, heads={args.decoder_heads}, layers={args.decoder_layers}")
    print(f"  loss weights: kl={args.kl_weight}, dir={args.direction_weight}, "
          f"intensity={args.intensity_weight}, rank={args.rank_weight}")
    print(f"  epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_metrics, vis_data = validate(model, val_loader, device)

        scheduler.step()

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]['lr']

        log = {
            'epoch': epoch,
            'lr': lr_now,
            'train_loss': train_metrics['total'],
            'train_recon': train_metrics['recon'],
            'train_kl': train_metrics['kl'],
            'train_dir': train_metrics['direction'],
            'train_intensity': train_metrics['intensity'],
            'train_rank': train_metrics['rank'],
            'val_loss': val_metrics['total'],
            'val_recon': val_metrics['recon'],
            'val_kl': val_metrics['kl'],
            'val_intensity': val_metrics['intensity'],
            'val_rank': val_metrics['rank'],
            'val_cosine_sim': val_metrics['cosine_sim'],
            'val_angular_error': val_metrics['angular_error_deg'],
            'time': elapsed,
        }
        history.append(log)

        history_path = os.path.join(args.output_dir, 'train_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"[Epoch {epoch:3d}/{args.epochs}] "
                  f"train: {train_metrics['total']:.6f} "
                  f"(recon={train_metrics['recon']:.6f} "
                  f"int={train_metrics['intensity']:.4f} "
                  f"rank={train_metrics['rank']:.4f}) | "
                  f"val: {val_metrics['total']:.6f} "
                  f"(recon={val_metrics['recon']:.6f}) | "
                  f"cos={val_metrics['cosine_sim']:.3f} "
                  f"ang={val_metrics['angular_error_deg']:.1f}° | "
                  f"lr={lr_now:.2e} | {elapsed:.1f}s")

        # Quiver Plot
        if epoch == 1 or epoch % args.vis_every == 0 or epoch == args.epochs:
            if vis_data[0] is not None:
                vis_mean = full_dataset.mean if full_dataset.normalize else None
                vis_std = full_dataset.std if full_dataset.normalize else None
                vis_path, _ = visualize_quiver_v2(
                    vis_data[0], vis_data[1],
                    epoch=epoch, output_dir=args.output_dir,
                    mean=vis_mean, std=vis_std,
                    n_samples=args.vis_samples,
                )
                if epoch == 1 or epoch % (args.vis_every * 5) == 0:
                    print(f"  -> Quiver plot saved: {vis_path}")

            # Temporal attention visualization
            if vis_data[2] is not None:
                attn_path = visualize_temporal_attention(
                    vis_data[2], epoch=epoch, output_dir=args.output_dir
                )
                if epoch == 1 or epoch % (args.vis_every * 5) == 0:
                    print(f"  -> Temporal attention saved: {attn_path}")

        # Save best
        if val_metrics['total'] < best_val_loss:
            best_val_loss = val_metrics['total']
            save_path = os.path.join(args.output_dir, 'best_tactile_vae_v2.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'config': config,
                'norm_stats': norm_stats,
            }, save_path)

        # Periodic checkpoint
        if epoch % args.save_every == 0:
            ckpt_path = os.path.join(args.output_dir, f'tactile_vae_v2_epoch{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['total'],
                'config': config,
                'norm_stats': norm_stats,
            }, ckpt_path)

    # Save final model
    final_path = os.path.join(args.output_dir, 'final_tactile_vae_v2.pt')
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'config': config,
        'norm_stats': norm_stats,
    }, final_path)

    with open(os.path.join(args.output_dir, 'train_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    final_cos = history[-1]['val_cosine_sim']
    final_ang = history[-1]['val_angular_error']

    print(f"\n{'='*60}")
    print(f"TactileVAE v2 Training complete!")
    print(f"  Best val loss:     {best_val_loss:.6f}")
    print(f"  Final cosine sim:  {final_cos:.3f}")
    print(f"  Final angular err: {final_ang:.1f}°")
    print(f"  Best model:        {os.path.join(args.output_dir, 'best_tactile_vae_v2.pt')}")
    print(f"  Final model:       {final_path}")
    print(f"  Visualizations:    {os.path.join(args.output_dir, 'visualizations/')}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
