"""
TactileVAE 完整 Episode 重建评估。

对每一帧 t，取其前 T=8 帧窗口 → VAE encode → decode → 与 GT 对比。
输出逐帧 quiver plot (GT / Recon / Error)、方向指标曲线、统计摘要。

用法:
    python TFAC_V5/eval_tactile_vae_episode.py \
        --vae_ckpt /home/chenshuai/Project/output/tactile_vae_full/best_tactile_vae.pt \
        --vae_stats /home/chenshuai/Project/output/tactile_vae_full/tactile_norm_stats.json \
        --episode /home/chenshuai/data/dataset/260407/bounce/episode_179.hdf5 \
        --step 1
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
import json
import argparse
import h5py
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from TFAC_V5.tactile_vae import TactileVAE, build_tactile_vae


def compute_direction_metrics(gt, recon):
    """
    方向性指标 (from pretrain_tactile_vae.py)。
    gt, recon: (9, 9, 2) numpy
    """
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


def plot_frame(gt_raw, recon_raw, frame_t, metrics, save_path):
    """GT / Recon / Error 三列 quiver plot"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    x = np.arange(9)
    X, Y = np.meshgrid(x, x)

    gt_mag = np.sqrt(gt_raw[..., 0]**2 + gt_raw[..., 1]**2)
    recon_mag = np.sqrt(recon_raw[..., 0]**2 + recon_raw[..., 1]**2)
    err = recon_raw - gt_raw
    err_mag = np.sqrt(err[..., 0]**2 + err[..., 1]**2)
    max_mag = max(gt_mag.max(), 1e-6)

    # GT
    ax = axes[0]
    q = ax.quiver(X, Y, gt_raw[..., 0], gt_raw[..., 1],
                  gt_mag, cmap='hot', scale=max_mag * 12,
                  scale_units='width', width=0.015)
    ax.set_title(f'GT (frame {frame_t})')
    ax.set_xlim(-0.5, 8.5); ax.set_ylim(-0.5, 8.5)
    ax.set_aspect('equal'); ax.invert_yaxis()
    plt.colorbar(q, ax=ax, fraction=0.046, pad=0.04)

    # Recon
    ax = axes[1]
    q = ax.quiver(X, Y, recon_raw[..., 0], recon_raw[..., 1],
                  recon_mag, cmap='hot', scale=max_mag * 12,
                  scale_units='width', width=0.015)
    ax.set_title(f'Recon  cos={metrics["cosine_sim"]:.3f}  '
                 f'ang={metrics["angular_error_deg"]:.1f}°  '
                 f'mag={metrics["magnitude_ratio"]:.2f}x')
    ax.set_xlim(-0.5, 8.5); ax.set_ylim(-0.5, 8.5)
    ax.set_aspect('equal'); ax.invert_yaxis()
    plt.colorbar(q, ax=ax, fraction=0.046, pad=0.04)

    # Error
    ax = axes[2]
    q = ax.quiver(X, Y, err[..., 0], err[..., 1],
                  err_mag, cmap='cool', scale=max_mag * 12,
                  scale_units='width', width=0.015,
                  clim=[0, max_mag])
    ax.set_title(f'Error (max={err_mag.max():.3f} / GT_max={max_mag:.3f})')
    ax.set_xlim(-0.5, 8.5); ax.set_ylim(-0.5, 8.5)
    ax.set_aspect('equal'); ax.invert_yaxis()
    plt.colorbar(q, ax=ax, fraction=0.046, pad=0.04)

    l1 = np.abs(gt_raw - recon_raw).mean()
    plt.suptitle(f't={frame_t} | L1={l1:.4f} | cos={metrics["cosine_sim"]:.3f} | '
                 f'ang={metrics["angular_error_deg"]:.1f}° | active={metrics["active_ratio"]:.2f}')
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vae_ckpt', type=str, required=True,
                        help='TactileVAE checkpoint path')
    parser.add_argument('--vae_stats', type=str, required=True,
                        help='TactileVAE norm stats JSON path')
    parser.add_argument('--episode', type=str, required=True,
                        help='Episode HDF5 path')
    parser.add_argument('--step', type=int, default=1,
                        help='Evaluate every N frames')
    parser.add_argument('--latent_dim', type=int, default=16,
                        help='TactileVAE latent channel dim')
    parser.add_argument('--temporal_window', type=int, default=8,
                        help='VAE temporal input window T')
    parser.add_argument('--tac_side', type=str, default='left',
                        help='Tactile sensor side')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: next to ckpt)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    T = args.temporal_window

    # --- Load norm stats ---
    with open(args.vae_stats) as f:
        stats = json.load(f)
    mo_mean = np.array(stats['mean'], dtype=np.float32)
    mo_std = np.array(stats['std'], dtype=np.float32)
    print(f"Norm stats: mean={mo_mean}, std={mo_std}")

    # --- Build VAE ---
    vae = build_tactile_vae(latent_dim=args.latent_dim,
                            temporal_window=T).to(device)
    state = torch.load(args.vae_ckpt, map_location=device)
    if 'model_state_dict' in state:
        vae.load_state_dict(state['model_state_dict'], strict=True)
    else:
        vae.load_state_dict(state, strict=True)
    vae.eval()
    print(f"Loaded VAE: {args.vae_ckpt}")

    # --- Load episode ---
    with h5py.File(args.episode, 'r') as f:
        marker_all = f[f'observations/tac/{args.tac_side}/marker_offset'][()]
    ep_len = marker_all.shape[0]
    ep_name = os.path.splitext(os.path.basename(args.episode))[0]
    print(f"Episode: {ep_name}, length={ep_len}")

    # --- Output dir ---
    if args.output_dir is None:
        ckpt_parent = os.path.dirname(args.vae_ckpt)
        vis_dir = os.path.join(ckpt_parent, f'vae_eval_{ep_name}')
    else:
        vis_dir = args.output_dir
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(os.path.join(vis_dir, 'frames'), exist_ok=True)

    # --- Evaluation loop ---
    # For each frame t, build a T-frame window ending at t, encode→decode,
    # compare the LAST latent frame's reconstruction with GT at frame t.
    # (encode_single_frame returns last frame latent, decoder reconstructs it)
    #
    # VAE temporal stride=2: input T=8 → latent T'=4 (frames at input idx 1,3,5,7)
    # The last latent frame corresponds to input index T-1=7, i.e., the current frame t.
    # So we compare decoder(z_last) with normalized marker_all[t].

    all_l1 = []
    all_mse = []
    all_cos = []
    all_ang = []
    all_mag = []
    all_active = []
    all_t = []

    # Also track raw L1 (denormalized)
    all_l1_raw = []

    eval_frames = list(range(0, ep_len, args.step))
    print(f"Evaluating {len(eval_frames)} frames (step={args.step})")

    for t in tqdm(eval_frames, desc="VAE Recon Eval"):
        # Build T-frame window ending at t
        window_frames = []
        for i in range(T):
            idx = max(0, t - (T - 1 - i))
            mo = marker_all[idx].astype(np.float32)
            mo_norm = (mo - mo_mean) / mo_std
            window_frames.append(torch.tensor(mo_norm, dtype=torch.float32))
        window = torch.stack(window_frames).unsqueeze(0).to(device)  # (1, T, 9, 9, 2)

        # GT (normalized) for current frame
        gt_norm = (marker_all[t].astype(np.float32) - mo_mean) / mo_std

        with torch.no_grad():
            # Full forward: get all T' reconstructed frames
            recon_all, mu, logvar = vae(window)  # recon_all: (1, T'=4, 9, 9, 2)
            # Last reconstructed frame corresponds to current frame t
            recon_last = recon_all[0, -1].cpu().numpy()  # (9, 9, 2)

        # Metrics in normalized space
        l1_norm = np.abs(recon_last - gt_norm).mean()
        mse_norm = ((recon_last - gt_norm) ** 2).mean()

        # Denormalize for raw-space metrics and visualization
        gt_raw = gt_norm * mo_std + mo_mean
        recon_raw = recon_last * mo_std + mo_mean
        l1_raw = np.abs(recon_raw - gt_raw).mean()

        # Direction metrics (on raw space, more interpretable)
        metrics = compute_direction_metrics(gt_raw, recon_raw)

        all_l1.append(l1_norm)
        all_mse.append(mse_norm)
        all_l1_raw.append(l1_raw)
        all_cos.append(metrics['cosine_sim'])
        all_ang.append(metrics['angular_error_deg'])
        all_mag.append(metrics['magnitude_ratio'])
        all_active.append(metrics['active_ratio'])
        all_t.append(t)

        # Per-frame quiver plot
        plot_frame(gt_raw, recon_raw, t, metrics,
                   os.path.join(vis_dir, 'frames', f'frame_{t:04d}.png'))

    # --- Summary statistics ---
    all_l1 = np.array(all_l1)
    all_mse = np.array(all_mse)
    all_l1_raw = np.array(all_l1_raw)
    all_cos = np.array(all_cos)
    all_ang = np.array(all_ang)
    all_mag = np.array(all_mag)
    all_active = np.array(all_active)
    all_t = np.array(all_t)

    # Filter: only frames with active_ratio > 0 for direction metrics
    active_mask = all_active > 0

    print("\n" + "=" * 70)
    print(f"TactileVAE Reconstruction Evaluation: {ep_name}")
    print("=" * 70)
    print(f"  Frames evaluated:   {len(eval_frames)}")
    print(f"  Active frames:      {active_mask.sum()} ({active_mask.mean()*100:.1f}%)")
    print(f"  --- Normalized space ---")
    print(f"  L1 (norm):   {all_l1.mean():.4f} +/- {all_l1.std():.4f}")
    print(f"  MSE (norm):  {all_mse.mean():.6f} +/- {all_mse.std():.6f}")
    print(f"  --- Raw space ---")
    print(f"  L1 (raw):    {all_l1_raw.mean():.4f} +/- {all_l1_raw.std():.4f}")
    print(f"  --- Direction metrics (active frames only) ---")
    if active_mask.sum() > 0:
        print(f"  Cosine sim:  {all_cos[active_mask].mean():.4f} +/- {all_cos[active_mask].std():.4f}")
        print(f"  Angular err: {all_ang[active_mask].mean():.2f}° +/- {all_ang[active_mask].std():.2f}°")
        print(f"  Mag ratio:   {all_mag[active_mask].mean():.4f} +/- {all_mag[active_mask].std():.4f}")
    print(f"  --- Worst/Best frames ---")
    worst_idx = np.argmax(all_l1_raw)
    best_idx = np.argmin(all_l1_raw)
    print(f"  Worst L1: t={all_t[worst_idx]}, L1_raw={all_l1_raw[worst_idx]:.4f}")
    print(f"  Best  L1: t={all_t[best_idx]}, L1_raw={all_l1_raw[best_idx]:.4f}")
    print("=" * 70)

    # --- Summary curves (4 panels) ---
    fig, axes = plt.subplots(4, 1, figsize=(16, 16), sharex=True)

    # Panel 1: L1 raw
    ax = axes[0]
    ax.plot(all_t, all_l1_raw, 'b-', alpha=0.7, linewidth=1)
    ax.axhline(all_l1_raw.mean(), color='red', linestyle='--',
               label=f'mean={all_l1_raw.mean():.4f}')
    ax.fill_between(all_t,
                    all_l1_raw.mean() - all_l1_raw.std(),
                    all_l1_raw.mean() + all_l1_raw.std(),
                    color='red', alpha=0.1)
    ax.set_ylabel('L1 Error (raw)')
    ax.set_title(f'TactileVAE Reconstruction: {ep_name} (latent_dim={args.latent_dim})')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel 2: Cosine similarity
    ax = axes[1]
    ax.plot(all_t, all_cos, 'g-', alpha=0.7, linewidth=1)
    if active_mask.sum() > 0:
        ax.axhline(all_cos[active_mask].mean(), color='darkgreen', linestyle='--',
                   label=f'mean(active)={all_cos[active_mask].mean():.4f}')
    ax.set_ylabel('Cosine Similarity')
    ax.set_ylim(-0.1, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel 3: Angular error
    ax = axes[2]
    ax.plot(all_t, all_ang, 'r-', alpha=0.7, linewidth=1)
    if active_mask.sum() > 0:
        ax.axhline(all_ang[active_mask].mean(), color='darkred', linestyle='--',
                   label=f'mean(active)={all_ang[active_mask].mean():.1f}°')
    ax.set_ylabel('Angular Error (°)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel 4: Magnitude ratio + Active ratio
    ax = axes[3]
    ax.plot(all_t, all_mag, 'purple', alpha=0.7, linewidth=1, label='Mag ratio')
    ax.axhline(1.0, color='gray', linestyle=':', alpha=0.5)
    ax2 = ax.twinx()
    ax2.fill_between(all_t, 0, all_active, alpha=0.15, color='orange', label='Active ratio')
    ax2.set_ylabel('Active Ratio', color='orange')
    ax2.set_ylim(0, 1.1)
    ax.set_ylabel('Magnitude Ratio')
    ax.set_xlabel('Frame t')
    ax.legend(loc='upper left', fontsize=10)
    ax2.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    summary_path = os.path.join(vis_dir, 'episode_summary.png')
    plt.savefig(summary_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSummary plot: {summary_path}")

    # --- Save metrics to JSON ---
    results = {
        'episode': args.episode,
        'ep_name': ep_name,
        'vae_ckpt': args.vae_ckpt,
        'latent_dim': args.latent_dim,
        'temporal_window': T,
        'num_frames': len(eval_frames),
        'step': args.step,
        'metrics': {
            'l1_norm_mean': float(all_l1.mean()),
            'l1_norm_std': float(all_l1.std()),
            'mse_norm_mean': float(all_mse.mean()),
            'mse_norm_std': float(all_mse.std()),
            'l1_raw_mean': float(all_l1_raw.mean()),
            'l1_raw_std': float(all_l1_raw.std()),
            'cosine_sim_mean': float(all_cos[active_mask].mean()) if active_mask.sum() > 0 else 0.0,
            'cosine_sim_std': float(all_cos[active_mask].std()) if active_mask.sum() > 0 else 0.0,
            'angular_error_mean': float(all_ang[active_mask].mean()) if active_mask.sum() > 0 else 0.0,
            'angular_error_std': float(all_ang[active_mask].std()) if active_mask.sum() > 0 else 0.0,
            'magnitude_ratio_mean': float(all_mag[active_mask].mean()) if active_mask.sum() > 0 else 0.0,
            'magnitude_ratio_std': float(all_mag[active_mask].std()) if active_mask.sum() > 0 else 0.0,
            'active_frame_ratio': float(active_mask.mean()),
        },
        'worst_frame': {'t': int(all_t[worst_idx]), 'l1_raw': float(all_l1_raw[worst_idx])},
        'best_frame': {'t': int(all_t[best_idx]), 'l1_raw': float(all_l1_raw[best_idx])},
    }
    json_path = os.path.join(vis_dir, 'eval_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results JSON: {json_path}")
    print(f"Frame plots: {os.path.join(vis_dir, 'frames/')}")


if __name__ == '__main__':
    main()
