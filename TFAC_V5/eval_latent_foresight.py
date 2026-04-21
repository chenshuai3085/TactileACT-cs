"""
Latent Foresight 预测可视化: 从 best ckpt 推理, quiver plot 对比 GT vs Predicted marker_offset。

用法:
    python TFAC_V5/eval_latent_foresight.py \
        --ckpt_dir /home/chenshuai/data/xiaomi_act/latent_foresight_pretrain_1
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import json
import pickle
import argparse
import matplotlib.pyplot as plt
import h5py

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from TFAC_V5.pretrain_latent_foresight import (
    LatentForesightPretrainModel, _scan_episode_paths, _infer_meta_from_path
)
from TFAC_V5.dataset import ForesightEpisodicDataset
from TFAC_V5.tactile_vae import TactileVAE


def denormalize_marker(marker_norm, mean, std):
    """反归一化 marker_offset: (9,9,2) normalized → raw."""
    return marker_norm * std + mean


def plot_quiver_comparison(gt_raw, pred_raw, sample_idx, save_path=None):
    """
    绘制 GT vs Predicted marker_offset 的 quiver 对比图。
    gt_raw, pred_raw: (9, 9, 2) 原始 marker_offset (未归一化)
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    y, x = np.mgrid[0:9, 0:9]

    # GT
    ax = axes[0]
    ax.quiver(x, y, gt_raw[:, :, 0], -gt_raw[:, :, 1],
              scale=30, color='blue', alpha=0.8)
    ax.set_title('GT marker_offset')
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(8.5, -0.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Predicted
    ax = axes[1]
    ax.quiver(x, y, pred_raw[:, :, 0], -pred_raw[:, :, 1],
              scale=30, color='red', alpha=0.8)
    ax.set_title('Predicted marker_offset')
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(8.5, -0.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Overlay
    ax = axes[2]
    ax.quiver(x, y, gt_raw[:, :, 0], -gt_raw[:, :, 1],
              scale=30, color='blue', alpha=0.6, label='GT')
    ax.quiver(x, y, pred_raw[:, :, 0], -pred_raw[:, :, 1],
              scale=30, color='red', alpha=0.6, label='Pred')
    ax.set_title('Overlay (blue=GT, red=Pred)')
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(8.5, -0.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Error metrics
    l1_err = np.abs(gt_raw - pred_raw).mean()
    cos_sim = np.sum(gt_raw * pred_raw) / (
        np.linalg.norm(gt_raw) * np.linalg.norm(pred_raw) + 1e-8)

    plt.suptitle(f'Sample {sample_idx} | L1={l1_err:.4f} | Cosine={cos_sim:.4f}')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, required=True)
    parser.add_argument('--n_samples', type=int, default=12,
                        help='Number of samples to visualize')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    ckpt_dir = args.ckpt_dir
    n_samples = args.n_samples

    # Load config
    with open(os.path.join(ckpt_dir, 'args.json')) as f:
        config = json.load(f)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Setup
    camera_names = config['camera_names']
    norm_stats = config['norm_stats']
    mo_mean = np.array(norm_stats['marker_offset_mean'], dtype=np.float32)
    mo_std = np.array(norm_stats['marker_offset_std'], dtype=np.float32)

    # Norm stats: convert lists to numpy
    for k in norm_stats:
        if isinstance(norm_stats[k], list):
            norm_stats[k] = np.array(norm_stats[k], dtype=np.float32)

    # Scan episodes
    dataset_dir = config['dataset_dir']
    episode_paths = _scan_episode_paths(dataset_dir)
    print(f"Found {len(episode_paths)} episodes")

    # Use a small val split (same seed as training)
    np.random.seed(config.get('seed', 1))
    shuffled = np.random.permutation(episode_paths).tolist()
    n_train = int(0.9 * len(shuffled))
    val_paths = shuffled[n_train:]
    print(f"Using {len(val_paths)} val episodes for visualization")

    # Build dataset
    val_dataset = ForesightEpisodicDataset(
        val_paths, dataset_dir, camera_names, norm_stats,
        chunk_size=config['chunk_size'],
        foresight_horizon=config.get('foresight_horizon', 10),
        proprio_key=config['proprio_key'],
        action_key=config['action_key'],
        tac_side=config.get('tac_side', 'left'),
        tac_img_key=config.get('tac_img_key', 'img'),
        tactile_mode=config.get('tactile_mode', 'marker'),
        history_len=config.get('history_len', 1),
        tactile_vae_window=config.get('tactile_vae_window', 8),
        preload=True)

    # Build model
    cam_backbone_mapping = {c: 0 for c in camera_names}
    model = LatentForesightPretrainModel(
        camera_names=camera_names,
        cam_backbone_mapping=cam_backbone_mapping,
        hidden_dim=config.get('hidden_dim', 512),
        state_dim=config.get('state_dim', 7),
        foresight_layers=config.get('foresight_layers', 3),
        foresight_nheads=config.get('foresight_nheads', 8),
        foresight_dim_feedforward=config.get('foresight_dim_feedforward', 2048),
        dropout=config.get('dropout', 0.1),
        tactile_mode=config.get('tactile_mode', 'marker'),
        predict_horizon=config.get('predict_horizon', 1),
        tactile_vae_ckpt=config.get('tactile_vae_ckpt'),
        tactile_vae_latent_dim=config.get('tactile_vae_latent_dim', 16),
    ).cuda()

    # Load best checkpoint
    ckpt_path = os.path.join(ckpt_dir, 'foresight_best.ckpt')
    state = torch.load(ckpt_path, map_location='cuda')
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path}")

    # Visualization directory
    vis_dir = os.path.join(ckpt_dir, 'vis_marker')
    os.makedirs(vis_dir, exist_ok=True)

    # Collect samples and run inference
    all_l1 = []
    all_cos = []

    latent_dim = config.get('tactile_vae_latent_dim', 16)

    for i in range(n_samples):
        idx = np.random.randint(len(val_dataset))
        all_cam_images, qpos_data, action_data, is_pad, future_cam_images, history_cam_images = val_dataset[idx]

        # To GPU
        images = [img.unsqueeze(0).cuda() for img in all_cam_images]
        qpos = qpos_data.unsqueeze(0).cuda()
        actions = action_data.unsqueeze(0).cuda()
        future_images = [img.unsqueeze(0).cuda() for img in future_cam_images]

        with torch.no_grad():
            t_hat, z_gt, future_raw, _, _, _ = model(images, actions,
                                                    future_images=future_images,
                                                    qpos=qpos)

            # Decode predicted latent → marker_offset via frozen VAE decoder
            z_hat_spatial = t_hat.reshape(-1, latent_dim, 3, 3)  # (1, 16, 3, 3)
            marker_pred_norm = model.tactile_vae.decoder(z_hat_spatial)  # (1, 9, 9, 2)
            marker_pred_norm = marker_pred_norm[0].cpu().numpy()  # (9, 9, 2)

            # Decode GT latent → marker_offset
            z_gt_spatial = z_gt.reshape(-1, latent_dim, 3, 3)
            marker_gt_norm = model.tactile_vae.decoder(z_gt_spatial)
            marker_gt_norm = marker_gt_norm[0].cpu().numpy()

            # Also get raw GT from future_raw (direct from dataset, normalized)
            if future_raw is not None:
                marker_gt_direct = future_raw[0].cpu().numpy()  # (9, 9, 2) normalized
            else:
                marker_gt_direct = marker_gt_norm

        # Denormalize
        pred_raw = denormalize_marker(marker_pred_norm, mo_mean, mo_std)
        gt_raw = denormalize_marker(marker_gt_norm, mo_mean, mo_std)
        gt_direct_raw = denormalize_marker(marker_gt_direct, mo_mean, mo_std)

        # Metrics
        l1 = np.abs(pred_raw - gt_raw).mean()
        cos = np.sum(pred_raw * gt_raw) / (
            np.linalg.norm(pred_raw) * np.linalg.norm(gt_raw) + 1e-8)
        all_l1.append(l1)
        all_cos.append(cos)

        # Plot: Pred vs GT (decoded from latent)
        plot_quiver_comparison(gt_raw, pred_raw, i,
                               save_path=os.path.join(vis_dir, f'sample_{i:02d}_latent.png'))

        # Plot: Pred vs GT direct (raw from dataset, shows VAE reconstruction error too)
        plot_quiver_comparison(gt_direct_raw, pred_raw, i,
                               save_path=os.path.join(vis_dir, f'sample_{i:02d}_raw.png'))

    # Summary plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(range(n_samples), all_l1, color='steelblue', alpha=0.8)
    axes[0].axhline(np.mean(all_l1), color='red', linestyle='--',
                    label=f'mean={np.mean(all_l1):.4f}')
    axes[0].set_title('Per-sample L1 Error (raw marker space)')
    axes[0].set_xlabel('Sample')
    axes[0].legend()

    axes[1].bar(range(n_samples), all_cos, color='coral', alpha=0.8)
    axes[1].axhline(np.mean(all_cos), color='red', linestyle='--',
                    label=f'mean={np.mean(all_cos):.4f}')
    axes[1].set_title('Per-sample Cosine Similarity')
    axes[1].set_xlabel('Sample')
    axes[1].legend()

    plt.suptitle(f'Latent Foresight Eval ({n_samples} val samples)\n'
                 f'Mean L1={np.mean(all_l1):.4f}, Mean Cos={np.mean(all_cos):.4f}')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'summary.png'), dpi=120)
    plt.close()

    print(f"\nResults: L1={np.mean(all_l1):.4f} ± {np.std(all_l1):.4f}, "
          f"Cosine={np.mean(all_cos):.4f} ± {np.std(all_cos):.4f}")
    print(f"Visualizations saved to: {vis_dir}")


if __name__ == '__main__':
    main()
