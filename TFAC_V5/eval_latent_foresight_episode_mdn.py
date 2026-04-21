"""
MDN 版逐帧 episode 可视化: 对每一帧 t, 预测 t+H 的 K 个触觉模式,
每个模式单独绘制 quiver plot + 混合权重, 并与 GT 对比。

用法:
    python TFAC_V5/eval_latent_foresight_episode_mdn.py \
        --ckpt_dir /home/chenshuai/data/xiaomi_act/latent_foresight_pretrain_mdn \
        --episode /home/chenshuai/data/dataset/260407/bounce/episode_100.hdf5
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import json
import argparse
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from TFAC_V5.pretrain_latent_foresight_mdn import MDNLatentForesightPretrainModel
from TFAC_V5.tactile_vae import TactileVAE


def denormalize_marker(marker_norm, mean, std):
    return marker_norm * std + mean


def plot_mdn_quiver_frame(gt_raw, mode_raws, mode_weights, frame_t, future_t,
                          best_l1, oracle_l1, save_path):
    """
    绘制 MDN 多模式 quiver plot。

    Layout: GT | Mode 0 (π=X.XX) | Mode 1 (π=X.XX) | Best vs GT overlay
    """
    K = len(mode_raws)
    n_cols = 2 + K  # GT + K modes + overlay
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))
    y, x = np.mgrid[0:9, 0:9]

    mag_gt = np.sqrt(gt_raw[:, :, 0]**2 + gt_raw[:, :, 1]**2)
    max_mag = mag_gt.max()
    for mr in mode_raws:
        mag = np.sqrt(mr[:, :, 0]**2 + mr[:, :, 1]**2)
        max_mag = max(max_mag, mag.max())
    max_mag = max(max_mag, 1e-6)
    scale = max(30, max_mag * 5)

    # GT
    ax = axes[0]
    ax.quiver(x, y, gt_raw[:, :, 0], -gt_raw[:, :, 1],
              scale=scale, color='blue', alpha=0.8)
    ax.set_title(f'GT (frame {future_t})')
    ax.set_xlim(-0.5, 8.5); ax.set_ylim(8.5, -0.5)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

    # Each mode
    colors = ['red', 'green', 'orange', 'purple']
    best_idx = np.argmax(mode_weights)
    for k in range(K):
        ax = axes[1 + k]
        c = colors[k % len(colors)]
        ax.quiver(x, y, mode_raws[k][:, :, 0], -mode_raws[k][:, :, 1],
                  scale=scale, color=c, alpha=0.8)
        marker = ' *' if k == best_idx else ''
        ax.set_title(f'Mode {k} (π={mode_weights[k]:.3f}){marker}')
        ax.set_xlim(-0.5, 8.5); ax.set_ylim(8.5, -0.5)
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

    # Overlay: best mode vs GT
    ax = axes[-1]
    ax.quiver(x, y, gt_raw[:, :, 0], -gt_raw[:, :, 1],
              scale=scale, color='blue', alpha=0.5, label='GT')
    ax.quiver(x, y, mode_raws[best_idx][:, :, 0], -mode_raws[best_idx][:, :, 1],
              scale=scale, color=colors[best_idx % len(colors)], alpha=0.5,
              label=f'Best (Mode {best_idx})')
    ax.set_title('Best Mode vs GT')
    ax.set_xlim(-0.5, 8.5); ax.set_ylim(8.5, -0.5)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    plt.suptitle(f't={frame_t} → t+H={future_t} | Best-L1={best_l1:.4f} | Oracle-L1={oracle_l1:.4f}')
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, required=True)
    parser.add_argument('--episode', type=str, required=True)
    parser.add_argument('--step', type=int, default=5)
    args = parser.parse_args()

    ckpt_dir = args.ckpt_dir
    episode_path = args.episode

    with open(os.path.join(ckpt_dir, 'args.json')) as f:
        config = json.load(f)

    norm_stats = config['norm_stats']
    mo_mean = np.array(norm_stats['marker_offset_mean'], dtype=np.float32)
    mo_std = np.array(norm_stats['marker_offset_std'], dtype=np.float32)

    camera_names = config['camera_names']
    chunk_size = config['chunk_size']
    horizon = config.get('foresight_horizon', 10)
    tactile_vae_window = config.get('tactile_vae_window', 8)
    latent_dim = config.get('tactile_vae_latent_dim', 16)
    proprio_key = config['proprio_key']
    action_key = config['action_key']
    tac_side = config.get('tac_side', 'left')
    mdn_num_modes = config.get('mdn_num_modes', 2)

    # Build model
    cam_backbone_mapping = {c: 0 for c in camera_names}
    model = MDNLatentForesightPretrainModel(
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
        tactile_vae_latent_dim=latent_dim,
        mdn_num_modes=mdn_num_modes,
        mdn_hidden_dim=config.get('mdn_hidden_dim', 256),
    ).cuda()

    ckpt_path = os.path.join(ckpt_dir, 'foresight_best.ckpt')
    state = torch.load(ckpt_path, map_location='cuda')
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"Loaded: {ckpt_path}")

    # ImageNet normalization
    img_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    img_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    # Load episode
    with h5py.File(episode_path, 'r') as f:
        ep_len = f[action_key].shape[0]
        actions_all = f[action_key][()]
        qpos_all = f[f'observations/{proprio_key}'][()]
        marker_all = f[f'observations/tac/{tac_side}/marker_offset'][()]

        vis_cams = {}
        for cam in camera_names:
            if cam not in ('gelsight', 'blank'):
                vis_cams[cam] = f[f'observations/images/{cam}'][()]

    print(f"Episode: {episode_path}, length={ep_len}")

    # Normalize
    qpos_mean = np.array(norm_stats['qpos_mean'], dtype=np.float32)
    qpos_std = np.array(norm_stats['qpos_std'], dtype=np.float32)
    action_mean = np.array(norm_stats['action_mean'], dtype=np.float32)
    action_std = np.array(norm_stats['action_std'], dtype=np.float32)

    # Output dir
    ep_name = os.path.splitext(os.path.basename(episode_path))[0]
    vis_dir = os.path.join(ckpt_dir, f'vis_episode_{ep_name}')
    os.makedirs(vis_dir, exist_ok=True)

    all_best_l1 = []
    all_oracle_l1 = []
    all_mode_weights = []
    all_t = []

    eval_frames = list(range(0, ep_len - horizon, args.step))
    print(f"Evaluating {len(eval_frames)} frames (step={args.step})")

    for t in tqdm(eval_frames, desc="Frames"):
        future_t = min(t + horizon, ep_len - 1)

        # --- Prepare inputs ---
        images_list = []
        for cam in camera_names:
            if cam == 'gelsight':
                T = tactile_vae_window
                tac_frames = []
                for i in range(T):
                    ht = max(0, t - (T - 1 - i))
                    mo = marker_all[ht].astype(np.float32)
                    mo = (mo - mo_mean) / mo_std
                    tac_frames.append(torch.tensor(mo, dtype=torch.float32))
                tac_window = torch.stack(tac_frames)
                images_list.append(tac_window.unsqueeze(0).cuda())
            elif cam == 'blank':
                images_list.append(torch.zeros(1, 3, 480, 640).cuda())
            else:
                img = vis_cams[cam][t].astype(np.float32) / 255.0
                img = torch.tensor(img).permute(2, 0, 1)
                img = (img - img_mean) / img_std
                images_list.append(img.unsqueeze(0).cuda())

        qpos_norm = (qpos_all[t] - qpos_mean) / qpos_std
        qpos_t = torch.tensor(qpos_norm, dtype=torch.float32).unsqueeze(0).cuda()

        act_end = min(t + chunk_size, ep_len)
        act_raw = actions_all[t:act_end]
        act_norm = (act_raw - action_mean) / action_std
        padded = np.zeros((chunk_size, 7), dtype=np.float32)
        padded[:act_raw.shape[0]] = act_norm
        actions_t = torch.tensor(padded, dtype=torch.float32).unsqueeze(0).cuda()

        # Future tactile for GT
        future_images_list = []
        for cam in camera_names:
            if cam == 'gelsight':
                T = tactile_vae_window
                future_windows = []
                for h in range(1, horizon + 1):
                    ft = min(t + h, ep_len - 1)
                    window_frames = []
                    for i in range(T):
                        wt = max(0, ft - (T - 1 - i))
                        mo = marker_all[wt].astype(np.float32)
                        mo = (mo - mo_mean) / mo_std
                        window_frames.append(torch.tensor(mo, dtype=torch.float32))
                    future_windows.append(torch.stack(window_frames))
                stacked = torch.stack(future_windows)
                future_images_list.append(stacked.unsqueeze(0).cuda())
            elif cam == 'blank':
                future_images_list.append(torch.zeros(1, horizon, 3, 480, 640).cuda())
            else:
                future_frames = []
                for h in range(1, horizon + 1):
                    ft = min(t + h, ep_len - 1)
                    img = vis_cams[cam][ft].astype(np.float32) / 255.0
                    img = torch.tensor(img).permute(2, 0, 1)
                    img = (img - img_mean) / img_std
                    future_frames.append(img)
                future_images_list.append(torch.stack(future_frames).unsqueeze(0).cuda())

        # --- Inference ---
        with torch.no_grad():
            mu, sigma, pi, z_gt, _, _, _ = model(
                images_list, actions_t, future_images=future_images_list, qpos=qpos_t)

            # Decode GT
            z_gt_spatial = z_gt.reshape(-1, latent_dim, 3, 3)
            marker_gt_norm = model.tactile_vae.decoder(z_gt_spatial)[0].cpu().numpy()

            # Decode each mode
            mode_markers = []
            pi_np = pi[0].cpu().numpy()
            for k in range(mdn_num_modes):
                z_k = mu[0, k]  # (144,)
                z_k_spatial = z_k.reshape(latent_dim, 3, 3).unsqueeze(0)
                marker_k_norm = model.tactile_vae.decoder(z_k_spatial)[0].cpu().numpy()
                mode_markers.append(denormalize_marker(marker_k_norm, mo_mean, mo_std))

        gt_raw = denormalize_marker(marker_gt_norm, mo_mean, mo_std)

        # Compute metrics
        best_k = np.argmax(pi_np)
        best_l1 = np.abs(mode_markers[best_k] - gt_raw).mean()
        oracle_l1s = [np.abs(m - gt_raw).mean() for m in mode_markers]
        orc_l1 = min(oracle_l1s)

        all_best_l1.append(best_l1)
        all_oracle_l1.append(orc_l1)
        all_mode_weights.append(pi_np.copy())
        all_t.append(t)

        # Save quiver plot
        plot_mdn_quiver_frame(gt_raw, mode_markers, pi_np, t, future_t,
                              best_l1, orc_l1,
                              os.path.join(vis_dir, f'frame_{t:04d}.png'))

    # --- Summary curves ---
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    axes[0].plot(all_t, all_best_l1, 'b-o', markersize=3, alpha=0.7, label='Best-mode L1')
    axes[0].plot(all_t, all_oracle_l1, 'g-s', markersize=3, alpha=0.7, label='Oracle-mode L1')
    axes[0].axhline(np.mean(all_best_l1), color='blue', linestyle='--',
                    label=f'Best mean={np.mean(all_best_l1):.4f}')
    axes[0].axhline(np.mean(all_oracle_l1), color='green', linestyle='--',
                    label=f'Oracle mean={np.mean(all_oracle_l1):.4f}')
    axes[0].set_ylabel('L1 Error (raw marker)')
    axes[0].set_title(f'Episode: {ep_name} | MDN K={mdn_num_modes} | Foresight H={horizon}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Mode weight distribution over time
    weights_arr = np.array(all_mode_weights)  # (T, K)
    for k in range(mdn_num_modes):
        axes[1].plot(all_t, weights_arr[:, k], '-o', markersize=2, alpha=0.7,
                    label=f'Mode {k}')
    axes[1].set_ylabel('Mixture Weight π')
    axes[1].set_title('Mode Selection Over Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # L1 gap between best and oracle (shows mode separation benefit)
    gap = np.array(all_best_l1) - np.array(all_oracle_l1)
    axes[2].plot(all_t, gap, 'r-o', markersize=3, alpha=0.7)
    axes[2].axhline(0, color='black', linestyle='-', alpha=0.3)
    axes[2].set_ylabel('Best - Oracle L1 Gap')
    axes[2].set_xlabel('Frame t')
    axes[2].set_title(f'Mode Selection Gap (mean={np.mean(gap):.4f})')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'episode_summary.png'), dpi=120)
    plt.close()

    print(f"\nEpisode {ep_name}:")
    print(f"  Best-mode L1: {np.mean(all_best_l1):.4f} +/- {np.std(all_best_l1):.4f}")
    print(f"  Oracle-mode L1: {np.mean(all_oracle_l1):.4f} +/- {np.std(all_oracle_l1):.4f}")
    print(f"  Gap (best-oracle): {np.mean(gap):.4f}")
    print(f"  Worst frame: t={all_t[np.argmax(all_best_l1)]}, Best-L1={max(all_best_l1):.4f}")
    print(f"  Best frame: t={all_t[np.argmin(all_best_l1)]}, Best-L1={min(all_best_l1):.4f}")
    print(f"  Saved {len(eval_frames)} frames to: {vis_dir}")


if __name__ == '__main__':
    main()
