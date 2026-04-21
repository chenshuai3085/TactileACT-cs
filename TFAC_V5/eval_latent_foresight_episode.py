"""
完整 episode 逐帧可视化: 对每一帧 t, 用当前观测 + GT action 预测 t+H 的触觉,
与 GT t+H 帧对比。输出逐帧 quiver plot + 整条 episode 的 L1/cosine 曲线。

用法:
    python TFAC_V5/eval_latent_foresight_episode.py \
        --ckpt_dir /home/chenshuai/data/xiaomi_act/latent_foresight_pretrain_1 \
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

from TFAC_V5.pretrain_latent_foresight import LatentForesightPretrainModel
from TFAC_V5.tactile_vae import TactileVAE


def denormalize_marker(marker_norm, mean, std):
    return marker_norm * std + mean


def plot_quiver_frame(gt_raw, pred_raw, frame_t, future_t, l1, cos, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    y, x = np.mgrid[0:9, 0:9]

    mag_gt = np.sqrt(gt_raw[:, :, 0]**2 + gt_raw[:, :, 1]**2)
    mag_pred = np.sqrt(pred_raw[:, :, 0]**2 + pred_raw[:, :, 1]**2)
    max_mag = max(mag_gt.max(), mag_pred.max(), 1e-6)
    scale = max(30, max_mag * 5)

    ax = axes[0]
    ax.quiver(x, y, gt_raw[:, :, 0], -gt_raw[:, :, 1],
              scale=scale, color='blue', alpha=0.8)
    ax.set_title(f'GT (frame {future_t})')
    ax.set_xlim(-0.5, 8.5); ax.set_ylim(8.5, -0.5)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.quiver(x, y, pred_raw[:, :, 0], -pred_raw[:, :, 1],
              scale=scale, color='red', alpha=0.8)
    ax.set_title(f'Predicted (from frame {frame_t})')
    ax.set_xlim(-0.5, 8.5); ax.set_ylim(8.5, -0.5)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.quiver(x, y, gt_raw[:, :, 0], -gt_raw[:, :, 1],
              scale=scale, color='blue', alpha=0.5, label='GT')
    ax.quiver(x, y, pred_raw[:, :, 0], -pred_raw[:, :, 1],
              scale=scale, color='red', alpha=0.5, label='Pred')
    ax.set_title('Overlay')
    ax.set_xlim(-0.5, 8.5); ax.set_ylim(8.5, -0.5)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    plt.suptitle(f't={frame_t} → t+H={future_t} | L1={l1:.4f} | Cos={cos:.4f}')
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, required=True)
    parser.add_argument('--episode', type=str, required=True)
    parser.add_argument('--step', type=int, default=5,
                        help='Visualize every N frames')
    args = parser.parse_args()

    ckpt_dir = args.ckpt_dir
    episode_path = args.episode

    with open(os.path.join(ckpt_dir, 'args.json')) as f:
        config = json.load(f)

    norm_stats = config['norm_stats']
    mo_mean = np.array(norm_stats['marker_offset_mean'], dtype=np.float32)
    mo_std = np.array(norm_stats['marker_offset_std'], dtype=np.float32)
    mo_mean_t = torch.tensor(mo_mean, dtype=torch.float32)
    mo_std_t = torch.tensor(mo_std, dtype=torch.float32)

    camera_names = config['camera_names']
    chunk_size = config['chunk_size']
    horizon = config.get('foresight_horizon', 10)
    tactile_vae_window = config.get('tactile_vae_window', 8)
    latent_dim = config.get('tactile_vae_latent_dim', 16)
    proprio_key = config['proprio_key']
    action_key = config['action_key']
    tac_side = config.get('tac_side', 'left')

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
        tactile_vae_latent_dim=latent_dim,
    ).cuda()

    ckpt_path = os.path.join(ckpt_dir, 'foresight_best.ckpt')
    state = torch.load(ckpt_path, map_location='cuda')
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"Loaded: {ckpt_path}")

    # ImageNet normalization
    img_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    img_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    # Load episode data
    with h5py.File(episode_path, 'r') as f:
        ep_len = f[action_key].shape[0]
        actions_all = f[action_key][()]           # (T, 7)
        qpos_all = f[f'observations/{proprio_key}'][()]  # (T, 7)
        marker_all = f[f'observations/tac/{tac_side}/marker_offset'][()]  # (T, 9, 9, 2)

        vis_cams = {}
        for cam in camera_names:
            if cam not in ('gelsight', 'blank'):
                vis_cams[cam] = f[f'observations/images/{cam}'][()]  # (T, H, W, 3)

    print(f"Episode: {episode_path}, length={ep_len}")

    # Normalize qpos/action
    qpos_mean = np.array(norm_stats['qpos_mean'], dtype=np.float32)
    qpos_std = np.array(norm_stats['qpos_std'], dtype=np.float32)
    action_mean = np.array(norm_stats['action_mean'], dtype=np.float32)
    action_std = np.array(norm_stats['action_std'], dtype=np.float32)

    # Output dir
    ep_name = os.path.splitext(os.path.basename(episode_path))[0]
    vis_dir = os.path.join(ckpt_dir, f'vis_episode_{ep_name}')
    os.makedirs(vis_dir, exist_ok=True)

    all_l1 = []
    all_cos = []
    all_t = []

    eval_frames = list(range(0, ep_len - horizon, args.step))
    print(f"Evaluating {len(eval_frames)} frames (step={args.step})")

    for t in tqdm(eval_frames, desc="Frames"):
        future_t = min(t + horizon, ep_len - 1)

        # --- Prepare inputs ---
        # Vision images: normalize
        images_list = []
        for cam in camera_names:
            if cam == 'gelsight':
                # Tactile: gather T=8 window for current frame t
                T = tactile_vae_window
                tac_frames = []
                for i in range(T):
                    ht = max(0, t - (T - 1 - i))
                    mo = marker_all[ht].astype(np.float32)
                    mo = (mo - mo_mean) / mo_std
                    tac_frames.append(torch.tensor(mo, dtype=torch.float32))
                tac_window = torch.stack(tac_frames)  # (T=8, 9, 9, 2)
                images_list.append(tac_window.unsqueeze(0).cuda())
            elif cam == 'blank':
                images_list.append(torch.zeros(1, 3, 480, 640).cuda())
            else:
                img = vis_cams[cam][t].astype(np.float32) / 255.0
                img = torch.tensor(img).permute(2, 0, 1)  # (3, H, W)
                img = (img - img_mean) / img_std
                images_list.append(img.unsqueeze(0).cuda())

        # Qpos
        qpos_norm = (qpos_all[t] - qpos_mean) / qpos_std
        qpos_t = torch.tensor(qpos_norm, dtype=torch.float32).unsqueeze(0).cuda()

        # Action chunk: t → t+chunk_size
        act_end = min(t + chunk_size, ep_len)
        act_raw = actions_all[t:act_end]
        act_norm = (act_raw - action_mean) / action_std
        padded = np.zeros((chunk_size, 7), dtype=np.float32)
        padded[:act_raw.shape[0]] = act_norm
        actions_t = torch.tensor(padded, dtype=torch.float32).unsqueeze(0).cuda()

        # Future tactile: gather T=8 window for future_t
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
                    future_windows.append(torch.stack(window_frames))  # (T, 9, 9, 2)
                stacked = torch.stack(future_windows)  # (H, T, 9, 9, 2)
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
            t_hat, z_gt, _, _, _ = model(images_list, actions_t,
                                        future_images=future_images_list,
                                        qpos=qpos_t)

            # Decode to marker space
            z_hat_spatial = t_hat.reshape(-1, latent_dim, 3, 3)
            marker_pred_norm = model.tactile_vae.decoder(z_hat_spatial)[0].cpu().numpy()

            z_gt_spatial = z_gt.reshape(-1, latent_dim, 3, 3)
            marker_gt_norm = model.tactile_vae.decoder(z_gt_spatial)[0].cpu().numpy()

        pred_raw = denormalize_marker(marker_pred_norm, mo_mean, mo_std)
        gt_raw = denormalize_marker(marker_gt_norm, mo_mean, mo_std)

        l1 = np.abs(pred_raw - gt_raw).mean()
        gt_norm_val = np.linalg.norm(gt_raw)
        cos = np.sum(pred_raw * gt_raw) / (
            np.linalg.norm(pred_raw) * gt_norm_val + 1e-8) if gt_norm_val > 1e-6 else 0.0

        all_l1.append(l1)
        all_cos.append(cos)
        all_t.append(t)

        # Save quiver plot
        plot_quiver_frame(gt_raw, pred_raw, t, future_t, l1, cos,
                          os.path.join(vis_dir, f'frame_{t:04d}.png'))

    # --- Summary curves ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(all_t, all_l1, 'b-o', markersize=3, alpha=0.7)
    axes[0].axhline(np.mean(all_l1), color='red', linestyle='--',
                    label=f'mean={np.mean(all_l1):.4f}')
    axes[0].set_ylabel('L1 Error (raw marker)')
    axes[0].set_title(f'Episode: {ep_name} | Foresight H={horizon}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(all_t, all_cos, 'r-o', markersize=3, alpha=0.7)
    axes[1].axhline(np.mean(all_cos), color='blue', linestyle='--',
                    label=f'mean={np.mean(all_cos):.4f}')
    axes[1].set_ylabel('Cosine Similarity')
    axes[1].set_xlabel('Frame t')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'episode_summary.png'), dpi=120)
    plt.close()

    print(f"\nEpisode {ep_name}: L1={np.mean(all_l1):.4f} ± {np.std(all_l1):.4f}, "
          f"Cos={np.mean(all_cos):.4f} ± {np.std(all_cos):.4f}")
    print(f"Worst frame: t={all_t[np.argmax(all_l1)]}, L1={max(all_l1):.4f}")
    print(f"Best frame: t={all_t[np.argmin(all_l1)]}, L1={min(all_l1):.4f}")
    print(f"Saved {len(eval_frames)} frames to: {vis_dir}")


if __name__ == '__main__':
    main()
