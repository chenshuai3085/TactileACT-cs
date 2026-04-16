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
        for data_dir in data_dirs:
            files = sorted([f for f in os.listdir(data_dir) if f.endswith('.hdf5')])
            for fname in files:
                fpath = os.path.join(data_dir, fname)
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
# 训练循环
# =============================================================================

def train_one_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    n_batches = 0

    for batch in dataloader:
        x = batch.to(device)  # (B, T, 9, 9, 2)

        # Forward
        recon, mu, logvar = model(x)

        # Align GT with temporally downsampled latent
        gt = TactileVAE.align_gt(x, temporal_stride=2)

        # Loss
        loss, recon_loss, kl_loss = model.loss(recon, gt, mu, logvar)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        n_batches += 1

    return {
        'loss': total_loss / n_batches,
        'recon': total_recon / n_batches,
        'kl': total_kl / n_batches,
    }


@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    n_batches = 0

    for batch in dataloader:
        x = batch.to(device)
        recon, mu, logvar = model(x)
        gt = TactileVAE.align_gt(x, temporal_stride=2)
        loss, recon_loss, kl_loss = model.loss(recon, gt, mu, logvar)

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        n_batches += 1

    return {
        'loss': total_loss / n_batches,
        'recon': total_recon / n_batches,
        'kl': total_kl / n_batches,
    }


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

        # Validate
        val_metrics = validate(model, val_loader, device)

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
            'val_loss': val_metrics['loss'],
            'val_recon': val_metrics['recon'],
            'val_kl': val_metrics['kl'],
            'time': elapsed,
        }
        history.append(log)

        # Print
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"[Epoch {epoch:3d}/{args.epochs}] "
                  f"train: {train_metrics['loss']:.6f} "
                  f"(recon={train_metrics['recon']:.6f} kl={train_metrics['kl']:.4f}) | "
                  f"val: {val_metrics['loss']:.6f} "
                  f"(recon={val_metrics['recon']:.6f}) | "
                  f"lr={lr_now:.2e} | {elapsed:.1f}s")

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

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Best val loss: {best_val_loss:.6f}")
    print(f"  Best model:    {os.path.join(args.output_dir, 'best_tactile_vae.pt')}")
    print(f"  Final model:   {final_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
