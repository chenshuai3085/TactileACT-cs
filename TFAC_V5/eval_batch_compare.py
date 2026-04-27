"""
批量评估 + 跨方法对比：对多条 bounce episode 用统一设置评估所有 Foresight 方法，
输出汇总 CSV + 对比表 + 对比柱状图。

不画逐帧 quiver plot，只计算指标，确保效率。

用法:
    python TFAC_V5/eval_batch_compare.py \
        --bounce_dir /home/chenshuai/data/dataset/260407/bounce \
        --num_episodes 10 \
        --step 5
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
import csv

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


def compute_direction_metrics(gt, recon):
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
        return {'cosine_sim': 0.0, 'angular_error_deg': 0.0,
                'magnitude_ratio': 1.0, 'active_ratio': 0.0}
    mag_ratio = recon_norm.squeeze()[active_mask] / gt_mag[active_mask]
    return {
        'cosine_sim': float(cos_sim[active_mask].mean()),
        'angular_error_deg': float(angular_error[active_mask].mean()),
        'magnitude_ratio': float(np.median(mag_ratio)),
        'active_ratio': float(active_mask.mean()),
    }


def load_model(ckpt_dir):
    """Load model + config from checkpoint directory."""
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


def eval_one_episode(model, config, episode_path, step=5):
    """Evaluate one episode, return per-frame metrics."""
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

    qpos_mean = np.array(norm_stats['qpos_mean'], dtype=np.float32)
    qpos_std = np.array(norm_stats['qpos_std'], dtype=np.float32)
    action_mean = np.array(norm_stats['action_mean'], dtype=np.float32)
    action_std = np.array(norm_stats['action_std'], dtype=np.float32)

    img_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    img_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    with h5py.File(episode_path, 'r') as f:
        ep_len = f[action_key].shape[0]
        actions_all = f[action_key][()]
        qpos_all = f[f'observations/{proprio_key}'][()]
        marker_all = f[f'observations/tac/{tac_side}/marker_offset'][()]
        vis_cams = {}
        for cam in camera_names:
            if cam not in ('gelsight', 'blank'):
                vis_cams[cam] = f[f'observations/images/{cam}'][()]

    results = []
    eval_frames = list(range(0, ep_len - horizon, step))

    for t in eval_frames:
        future_t = min(t + horizon, ep_len - 1)

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
                tac_window = torch.stack(tac_frames)
                images_list.append(tac_window.unsqueeze(0).cuda())
            elif cam == 'blank':
                images_list.append(torch.zeros(1, 3, 480, 640).cuda())
            else:
                img = vis_cams[cam][t].astype(np.float32) / 255.0
                img = torch.tensor(img).permute(2, 0, 1)
                img = (img - img_mean) / img_std
                images_list.append(img.unsqueeze(0).cuda())

        # Qpos
        qpos_norm = (qpos_all[t] - qpos_mean) / qpos_std
        qpos_t = torch.tensor(qpos_norm, dtype=torch.float32).unsqueeze(0).cuda()

        # Action chunk
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

        # Inference
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

        l1 = np.abs(pred_raw - gt_raw).mean()
        dir_metrics = compute_direction_metrics(gt_raw, pred_raw)

        # Compute delta_phi (physical change magnitude)
        if z_current is not None:
            z_cur_spatial = z_current.reshape(-1, latent_dim, 3, 3)
            marker_cur_norm = model.tactile_vae.decoder(z_cur_spatial)[0].cpu().numpy()
            cur_raw = denormalize_marker(marker_cur_norm, mo_mean, mo_std)
            delta_phi = np.abs(gt_raw - cur_raw).mean()
        else:
            delta_phi = 0.0

        results.append({
            'frame': t,
            'l1': l1,
            'cosine_sim': dir_metrics['cosine_sim'],
            'angular_error': dir_metrics['angular_error_deg'],
            'mag_ratio': dir_metrics['magnitude_ratio'],
            'delta_phi': delta_phi,
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bounce_dir', type=str,
                        default='/home/chenshuai/data/dataset/260407/bounce')
    parser.add_argument('--num_episodes', type=int, default=10,
                        help='Number of bounce episodes to evaluate')
    parser.add_argument('--step', type=int, default=5,
                        help='Evaluate every N frames per episode')
    parser.add_argument('--output_dir', type=str,
                        default='/home/chenshuai/data/xiaomi_act/foresight_comparison')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Select episodes deterministically
    all_episodes = sorted([f for f in os.listdir(args.bounce_dir) if f.endswith('.hdf5')])
    np.random.seed(args.seed)
    selected = np.random.choice(len(all_episodes), size=min(args.num_episodes, len(all_episodes)),
                                replace=False)
    selected_episodes = [all_episodes[i] for i in sorted(selected)]
    print(f"Selected {len(selected_episodes)} bounce episodes: "
          f"{[e.replace('.hdf5','').replace('episode_','ep') for e in selected_episodes]}")

    # Evaluate each method
    all_method_results = {}

    for method_name, ckpt_dir in METHODS.items():
        if not os.path.exists(os.path.join(ckpt_dir, 'foresight_best.ckpt')):
            print(f"\n[SKIP] {method_name}: no best checkpoint")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating: {method_name} ({ckpt_dir})")
        print(f"{'='*60}")

        model, config = load_model(ckpt_dir)
        method_results = []

        for ep_file in tqdm(selected_episodes, desc=method_name):
            ep_path = os.path.join(args.bounce_dir, ep_file)
            ep_name = ep_file.replace('.hdf5', '')
            ep_results = eval_one_episode(model, config, ep_path, step=args.step)

            for r in ep_results:
                r['episode'] = ep_name
                r['method'] = method_name
            method_results.extend(ep_results)

        all_method_results[method_name] = method_results
        del model
        torch.cuda.empty_cache()

    # --- Aggregate and compare ---
    print(f"\n{'='*70}")
    print(f"COMPARISON RESULTS ({len(selected_episodes)} bounce episodes, step={args.step})")
    print(f"{'='*70}")

    summary_rows = []
    for method_name in METHODS:
        if method_name not in all_method_results:
            continue
        results = all_method_results[method_name]
        l1s = [r['l1'] for r in results]
        coss = [r['cosine_sim'] for r in results]
        angs = [r['angular_error'] for r in results]
        mags = [r['mag_ratio'] for r in results]

        # Per-episode averages
        ep_l1s = {}
        for r in results:
            ep = r['episode']
            if ep not in ep_l1s:
                ep_l1s[ep] = []
            ep_l1s[ep].append(r['l1'])
        ep_means = [np.mean(v) for v in ep_l1s.values()]

        row = {
            'method': method_name,
            'n_frames': len(results),
            'l1_mean': np.mean(l1s),
            'l1_std': np.std(l1s),
            'l1_median': np.median(l1s),
            'cos_mean': np.mean(coss),
            'cos_std': np.std(coss),
            'ang_mean': np.mean(angs),
            'ang_std': np.std(angs),
            'mag_mean': np.mean(mags),
            'mag_std': np.std(mags),
            'ep_l1_mean': np.mean(ep_means),
            'ep_l1_std': np.std(ep_means),
        }
        summary_rows.append(row)

        print(f"\n  {method_name}:")
        print(f"    L1 (raw):      {row['l1_mean']:.4f} +/- {row['l1_std']:.4f}  (median={row['l1_median']:.4f})")
        print(f"    Cosine sim:    {row['cos_mean']:.4f} +/- {row['cos_std']:.4f}")
        print(f"    Angular err:   {row['ang_mean']:.2f}° +/- {row['ang_std']:.2f}°")
        print(f"    Mag ratio:     {row['mag_mean']:.4f} +/- {row['mag_std']:.4f}")
        print(f"    Per-ep L1:     {row['ep_l1_mean']:.4f} +/- {row['ep_l1_std']:.4f}")

    # --- Save CSV ---
    csv_path = os.path.join(args.output_dir, 'comparison_summary.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSummary CSV: {csv_path}")

    # Save per-frame CSV
    detail_path = os.path.join(args.output_dir, 'comparison_detail.csv')
    all_detail = []
    for results in all_method_results.values():
        all_detail.extend(results)
    if all_detail:
        with open(detail_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_detail[0].keys())
            writer.writeheader()
            writer.writerows(all_detail)
        print(f"Detail CSV: {detail_path}")

    # --- Comparison bar chart ---
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    methods = [r['method'] for r in summary_rows]
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2'][:len(methods)]

    # L1
    ax = axes[0]
    vals = [r['l1_mean'] for r in summary_rows]
    errs = [r['l1_std'] for r in summary_rows]
    bars = ax.bar(methods, vals, yerr=errs, capsize=5, color=colors, alpha=0.8)
    ax.set_ylabel('L1 Error (raw)')
    ax.set_title('L1 Error')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{v:.4f}', ha='center', va='bottom', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # Cosine
    ax = axes[1]
    vals = [r['cos_mean'] for r in summary_rows]
    errs = [r['cos_std'] for r in summary_rows]
    bars = ax.bar(methods, vals, yerr=errs, capsize=5, color=colors, alpha=0.8)
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('Cosine Similarity')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{v:.4f}', ha='center', va='bottom', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # Angular error
    ax = axes[2]
    vals = [r['ang_mean'] for r in summary_rows]
    errs = [r['ang_std'] for r in summary_rows]
    bars = ax.bar(methods, vals, yerr=errs, capsize=5, color=colors, alpha=0.8)
    ax.set_ylabel('Angular Error (°)')
    ax.set_title('Angular Error')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{v:.1f}°', ha='center', va='bottom', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # Magnitude ratio
    ax = axes[3]
    vals = [r['mag_mean'] for r in summary_rows]
    errs = [r['mag_std'] for r in summary_rows]
    bars = ax.bar(methods, vals, yerr=errs, capsize=5, color=colors, alpha=0.8)
    ax.axhline(1.0, color='gray', linestyle=':', alpha=0.5)
    ax.set_ylabel('Magnitude Ratio')
    ax.set_title('Magnitude Ratio (1.0=ideal)')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.suptitle(f'Foresight Method Comparison ({len(selected_episodes)} bounce episodes, step={args.step})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(args.output_dir, 'comparison_chart.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Chart: {fig_path}")

    # --- Per-episode L1 comparison ---
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(selected_episodes))
    width = 0.8 / len(summary_rows)
    for i, (method_name, results) in enumerate(all_method_results.items()):
        ep_l1s = {}
        for r in results:
            ep = r['episode']
            if ep not in ep_l1s:
                ep_l1s[ep] = []
            ep_l1s[ep].append(r['l1'])
        ep_means = [np.mean(ep_l1s.get(e.replace('.hdf5',''), [0])) for e in selected_episodes]
        ax.bar(x + i * width, ep_means, width, label=method_name, alpha=0.8, color=colors[i])

    ax.set_xlabel('Episode')
    ax.set_ylabel('Mean L1 Error (raw)')
    ax.set_title('Per-Episode L1 Comparison')
    ep_labels = [e.replace('.hdf5','').replace('episode_','') for e in selected_episodes]
    ax.set_xticks(x + width * (len(summary_rows) - 1) / 2)
    ax.set_xticklabels(ep_labels, rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig_path2 = os.path.join(args.output_dir, 'per_episode_l1.png')
    plt.savefig(fig_path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Per-episode chart: {fig_path2}")

    print(f"\nAll results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
