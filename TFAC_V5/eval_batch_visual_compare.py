"""
批量可视化重建对比：对相同的 10 条 bounce episode，4 个方法逐帧生成
GT vs Pred quiver plot 对比图（每帧一行 5 列：GT + 4 个方法）。

与 eval_batch_compare.py 使用完全相同的 episode 选择 (seed=42)。

用法:
    python TFAC_V5/eval_batch_visual_compare.py \
        --bounce_dir /home/chenshuai/data/dataset/260407/bounce \
        --num_episodes 10 \
        --step 5
"""

import torch
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
from TFAC_V5.pretrain_latent_foresight import LatentForesightPretrainModel


METHODS = {
    'Baseline': '/home/chenshuai/data/xiaomi_act/latent_foresight_pretrain_1',
    'DW': '/home/chenshuai/data/xiaomi_act/latent_foresight_pretrain_dw',
    'Delta-Pred': '/home/chenshuai/data/xiaomi_act/latent_foresight_pretrain_dp',
    'Residual': '/home/chenshuai/data/xiaomi_act/latent_foresight_pretrain_residual',
}


def denormalize_marker(marker_norm, mean, std):
    return marker_norm * std + mean


def load_model(ckpt_dir):
    with open(os.path.join(ckpt_dir, 'args.json')) as f:
        config = json.load(f)
    camera_names = config['camera_names']
    latent_dim = config.get('tactile_vae_latent_dim', 16)
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
        residual_prediction=config.get('residual_prediction', False),
    ).cuda()
    ckpt_path = os.path.join(ckpt_dir, 'foresight_best.ckpt')
    state = torch.load(ckpt_path, map_location='cuda')
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, config


def predict_one_frame(model, config, marker_all, vis_cams, actions_all, qpos_all, t):
    """Predict future tactile for frame t, return pred_raw and gt_raw in marker space."""
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
    residual_prediction = config.get('residual_prediction', False)
    ep_len = marker_all.shape[0]

    qpos_mean = np.array(norm_stats['qpos_mean'], dtype=np.float32)
    qpos_std = np.array(norm_stats['qpos_std'], dtype=np.float32)
    action_mean = np.array(norm_stats['action_mean'], dtype=np.float32)
    action_std = np.array(norm_stats['action_std'], dtype=np.float32)
    img_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    img_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    # Current images
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
            images_list.append(torch.stack(tac_frames).unsqueeze(0).cuda())
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

    # Future tactile windows
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
            future_images_list.append(torch.stack(future_windows).unsqueeze(0).cuda())
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

    with torch.no_grad():
        t_hat, z_gt, _, z_current, _, _ = model(
            images_list, actions_t,
            future_images=future_images_list, qpos=qpos_t)

        if residual_prediction and z_current is not None:
            z_hat_abs = z_current + t_hat
        else:
            z_hat_abs = t_hat

        z_hat_spatial = z_hat_abs.reshape(-1, latent_dim, 3, 3)
        marker_pred_norm = model.tactile_vae.decoder(z_hat_spatial)[0].cpu().numpy()

        z_gt_spatial = z_gt.reshape(-1, latent_dim, 3, 3)
        marker_gt_norm = model.tactile_vae.decoder(z_gt_spatial)[0].cpu().numpy()

    pred_raw = denormalize_marker(marker_pred_norm, mo_mean, mo_std)
    gt_raw = denormalize_marker(marker_gt_norm, mo_mean, mo_std)
    return pred_raw, gt_raw


def plot_comparison_frame(gt_raw, preds_dict, frame_t, future_t, save_path):
    """Plot GT + all methods' predictions in one row."""
    n_methods = len(preds_dict)
    fig, axes = plt.subplots(1, 1 + n_methods, figsize=(5 * (1 + n_methods), 4.5))
    y, x = np.mgrid[0:9, 0:9]

    # Compute unified scale
    all_mag = [np.sqrt(gt_raw[..., 0]**2 + gt_raw[..., 1]**2).max()]
    for pred in preds_dict.values():
        all_mag.append(np.sqrt(pred[..., 0]**2 + pred[..., 1]**2).max())
    max_mag = max(max(all_mag), 1e-6)
    scale = max(30, max_mag * 5)

    # GT
    ax = axes[0]
    gt_mag = np.sqrt(gt_raw[..., 0]**2 + gt_raw[..., 1]**2)
    ax.quiver(x, y, gt_raw[..., 0], -gt_raw[..., 1],
              scale=scale, color='blue', alpha=0.8)
    ax.set_title(f'GT (t+H={future_t})', fontsize=10, fontweight='bold')
    ax.set_xlim(-0.5, 8.5); ax.set_ylim(8.5, -0.5)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

    # Each method
    colors = ['#C44E52', '#55A868', '#8172B2', '#CCB974']
    for i, (method_name, pred_raw) in enumerate(preds_dict.items()):
        ax = axes[1 + i]
        l1 = np.abs(pred_raw - gt_raw).mean()
        pred_flat = pred_raw.reshape(-1, 2)
        gt_flat = gt_raw.reshape(-1, 2)
        dot = (pred_flat * gt_flat).sum()
        cos = dot / (np.linalg.norm(pred_flat) * np.linalg.norm(gt_flat) + 1e-8)

        ax.quiver(x, y, pred_raw[..., 0], -pred_raw[..., 1],
                  scale=scale, color=colors[i % len(colors)], alpha=0.8)
        ax.set_title(f'{method_name}\nL1={l1:.3f} cos={cos:.3f}', fontsize=9)
        ax.set_xlim(-0.5, 8.5); ax.set_ylim(8.5, -0.5)
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

    plt.suptitle(f't={frame_t} → t+H={future_t}', fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bounce_dir', type=str,
                        default='/home/chenshuai/data/dataset/260407/bounce')
    parser.add_argument('--num_episodes', type=int, default=10)
    parser.add_argument('--step', type=int, default=5)
    parser.add_argument('--output_dir', type=str,
                        default='/home/chenshuai/data/xiaomi_act/foresight_comparison')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Same episode selection as eval_batch_compare.py
    all_episodes = sorted([f for f in os.listdir(args.bounce_dir) if f.endswith('.hdf5')])
    np.random.seed(args.seed)
    selected = np.random.choice(len(all_episodes), size=min(args.num_episodes, len(all_episodes)),
                                replace=False)
    selected_episodes = [all_episodes[i] for i in sorted(selected)]
    print(f"Selected {len(selected_episodes)} episodes (same as batch compare)")

    # Load all models
    models = {}
    configs = {}
    for method_name, ckpt_dir in METHODS.items():
        if not os.path.exists(os.path.join(ckpt_dir, 'foresight_best.ckpt')):
            print(f"[SKIP] {method_name}")
            continue
        print(f"Loading {method_name}...")
        models[method_name], configs[method_name] = load_model(ckpt_dir)
    print(f"Loaded {len(models)} models")

    # Use first method's config for episode data loading
    ref_config = list(configs.values())[0]
    action_key = ref_config['action_key']
    proprio_key = ref_config['proprio_key']
    tac_side = ref_config.get('tac_side', 'left')
    camera_names = ref_config['camera_names']
    horizon = ref_config.get('foresight_horizon', 10)

    vis_base = os.path.join(args.output_dir, 'visual_compare')
    os.makedirs(vis_base, exist_ok=True)

    for ep_file in selected_episodes:
        ep_path = os.path.join(args.bounce_dir, ep_file)
        ep_name = ep_file.replace('.hdf5', '')
        ep_dir = os.path.join(vis_base, ep_name)
        os.makedirs(ep_dir, exist_ok=True)

        # Load episode data once
        with h5py.File(ep_path, 'r') as f:
            ep_len = f[action_key].shape[0]
            actions_all = f[action_key][()]
            qpos_all = f[f'observations/{proprio_key}'][()]
            marker_all = f[f'observations/tac/{tac_side}/marker_offset'][()]
            vis_cams = {}
            for cam in camera_names:
                if cam not in ('gelsight', 'blank'):
                    vis_cams[cam] = f[f'observations/images/{cam}'][()]

        eval_frames = list(range(0, ep_len - horizon, args.step))
        print(f"\n{ep_name}: {len(eval_frames)} frames")

        for t in tqdm(eval_frames, desc=ep_name):
            future_t = min(t + horizon, ep_len - 1)
            preds = {}
            gt_raw = None

            for method_name in models:
                pred_raw, gt_raw_m = predict_one_frame(
                    models[method_name], configs[method_name],
                    marker_all, vis_cams, actions_all, qpos_all, t)
                preds[method_name] = pred_raw
                if gt_raw is None:
                    gt_raw = gt_raw_m

            plot_comparison_frame(
                gt_raw, preds, t, future_t,
                os.path.join(ep_dir, f'frame_{t:04d}.png'))

        print(f"  Saved to: {ep_dir}")

    # Cleanup
    for m in models.values():
        del m
    torch.cuda.empty_cache()

    print(f"\nAll visual comparisons saved to: {vis_base}")


if __name__ == '__main__':
    main()
