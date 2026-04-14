"""
Per-sample correlation analysis: foresight_tac vs l1_final.
Proves "worse tactile prediction but better action" is not statistical coincidence.

Usage:
    python TFAC/eval_correlation.py --ckpt_dir /path/to/ckpt_dir [--num_samples 500]
"""

import argparse
import json
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from TFAC_V1.dataset import ForesightEpisodicDataset
from TFAC_V1.eval_foresight import build_policy_from_args
from utils import set_seed, load_meta_data


def main():
    parser = argparse.ArgumentParser(description='Per-sample correlation: foresight_tac vs l1_final')
    parser.add_argument('--ckpt_dir', type=str, required=True)
    parser.add_argument('--ckpt_name', type=str, default='policy_best.ckpt')
    parser.add_argument('--num_samples', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    cli = parser.parse_args()

    set_seed(cli.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load config
    with open(os.path.join(cli.ckpt_dir, 'args.json')) as f:
        args = json.load(f)

    with open(os.path.join(cli.ckpt_dir, 'dataset_stats.pkl'), 'rb') as f:
        norm_stats = pickle.load(f)

    # Build dataset (validation split)
    save_dir = args['save_dir']
    dataset_dir = args.get('dataset_dir') or os.path.join(save_dir, 'data')
    meta_data = load_meta_data(dataset_dir, save_dir=save_dir, config_overrides=args)
    num_episodes = meta_data['num_episodes']
    camera_names = meta_data['camera_names']
    chunk_size = args['chunk_size']
    foresight_horizon = args.get('foresight_horizon', 8)
    tactile_mode = args.get('tactile_mode', 'image')

    if tactile_mode != 'marker':
        print('This script requires tactile_mode=marker (needs GT tactile for comparison).')
        return

    np.random.seed(args.get('seed', 1))
    shuffled_indices = np.random.permutation(num_episodes)
    val_indices = shuffled_indices[int(0.8 * num_episodes):]

    val_dataset = ForesightEpisodicDataset(
        val_indices, dataset_dir, camera_names, norm_stats,
        chunk_size=chunk_size, foresight_horizon=foresight_horizon,
        proprio_key=meta_data['proprio_key'],
        action_key=meta_data['action_key'],
        tac_side=meta_data['tac_side'],
        tac_img_key=meta_data['tac_img_key'],
        tactile_mode=tactile_mode,
    )

    # Build & load model
    ckpt_path = os.path.join(cli.ckpt_dir, cli.ckpt_name)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        ckpt = ckpt['model']

    merged_args = {**args, **meta_data}
    if 'spatial_tac_dec_layers' not in merged_args:
        fc_key = 'model.foresight.tactile_out.fc.weight'
        if fc_key in ckpt:
            fc_out = ckpt[fc_key].shape[0]
            merged_args['spatial_tac_dec_layers'] = 3 if fc_out == 2304 else 2

    policy = build_policy_from_args(merged_args)
    result = policy.load_state_dict(ckpt, strict=False)
    if result.missing_keys:
        print(f'Warning missing keys: {result.missing_keys}')
    policy.to(device)
    policy.eval()
    print(f'Loaded {ckpt_path}')

    # Collect per-sample losses
    N = min(cli.num_samples, len(val_dataset))
    np.random.seed(cli.seed)
    sample_indices = np.random.choice(len(val_dataset), size=N, replace=False)

    per_sample_foresight_tac = []
    per_sample_l1_final = []
    per_sample_l1_draft = []
    per_sample_contact_mag = []

    print(f'\nCollecting {N} samples...')
    with torch.inference_mode():
        for idx in sample_indices:
            all_cam_images, qpos_data, action_data, is_pad, future_cam_images = val_dataset[idx]

            qpos = qpos_data.unsqueeze(0).to(device)
            images = [img.unsqueeze(0).to(device) for img in all_cam_images]
            actions = action_data.unsqueeze(0).to(device)
            is_pad_t = is_pad.unsqueeze(0).to(device)
            future_images = [img.unsqueeze(0).to(device) for img in future_cam_images]

            (a1, a2, t_hat, v_hat, v_gt, t_gt, t_hat_enc, t_cur,
             (mu, logvar)) = policy.model(
                qpos, images, actions, is_pad_t, future_images,
                use_predicted_future=True)

            # Per-sample foresight_tac loss (smooth L1 on raw marker_offset)
            if t_hat is not None and t_gt is not None:
                foresight_loss = F.smooth_l1_loss(t_hat, t_gt).item()
                per_sample_foresight_tac.append(foresight_loss)

                # Contact magnitude
                mag = torch.sqrt((t_gt[0] ** 2).sum(dim=-1)).mean().item()
                per_sample_contact_mag.append(mag)

            # Per-sample l1_final (action L1 on non-padded steps)
            target = actions[0]  # (chunk_size, action_dim)
            pad_mask = is_pad[0] if is_pad.dim() > 1 else is_pad  # (chunk_size,)
            valid = ~pad_mask.bool() if pad_mask.dtype != torch.bool else ~pad_mask

            if valid.any():
                l1_final = F.l1_loss(a2[0][valid.to(device)], target[valid.to(device)]).item()
                l1_draft = F.l1_loss(a1[0][valid.to(device)], target[valid.to(device)]).item()
            else:
                l1_final = F.l1_loss(a2[0], target.to(device)).item()
                l1_draft = F.l1_loss(a1[0], target.to(device)).item()

            per_sample_l1_final.append(l1_final)
            per_sample_l1_draft.append(l1_draft)

    foresight_arr = np.array(per_sample_foresight_tac)
    l1_final_arr = np.array(per_sample_l1_final)
    l1_draft_arr = np.array(per_sample_l1_draft)
    contact_arr = np.array(per_sample_contact_mag)

    # Correlation analysis
    pearson_r, pearson_p = stats.pearsonr(foresight_arr, l1_final_arr)
    spearman_r, spearman_p = stats.spearmanr(foresight_arr, l1_final_arr)

    pearson_draft_r, pearson_draft_p = stats.pearsonr(foresight_arr, l1_draft_arr)

    model_name = os.path.basename(cli.ckpt_dir)
    print(f'\n=== Correlation Analysis ({model_name}, {N} samples) ===')
    print(f'foresight_tac vs l1_final:')
    print(f'  Pearson:  r={pearson_r:.4f}, p={pearson_p:.2e}')
    print(f'  Spearman: r={spearman_r:.4f}, p={spearman_p:.2e}')
    print(f'foresight_tac vs l1_draft:')
    print(f'  Pearson:  r={pearson_draft_r:.4f}, p={pearson_draft_p:.2e}')
    print(f'\nMean foresight_tac: {foresight_arr.mean():.4f} +/- {foresight_arr.std():.4f}')
    print(f'Mean l1_final:      {l1_final_arr.mean():.4f} +/- {l1_final_arr.std():.4f}')
    print(f'Mean l1_draft:      {l1_draft_arr.mean():.4f} +/- {l1_draft_arr.std():.4f}')

    # --- Plot ---
    out_dir = os.path.join(cli.ckpt_dir, 'correlation_eval')
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Left: foresight_tac vs l1_final scatter
    ax = axes[0]
    sc = ax.scatter(foresight_arr, l1_final_arr, c=contact_arr, cmap='coolwarm',
                    s=15, alpha=0.6)
    plt.colorbar(sc, ax=ax, label='Contact magnitude')
    # Fit line
    z = np.polyfit(foresight_arr, l1_final_arr, 1)
    p = np.poly1d(z)
    x_line = np.linspace(foresight_arr.min(), foresight_arr.max(), 100)
    ax.plot(x_line, p(x_line), 'r--', alpha=0.7)
    ax.set_xlabel('foresight_tac (Smooth L1)', fontsize=11)
    ax.set_ylabel('l1_final (Action L1)', fontsize=11)
    ax.set_title(f'Pearson r={pearson_r:.3f} (p={pearson_p:.1e})\n'
                 f'Spearman r={spearman_r:.3f} (p={spearman_p:.1e})', fontsize=11)

    # Middle: foresight_tac vs l1_draft scatter
    ax2 = axes[1]
    ax2.scatter(foresight_arr, l1_draft_arr, c=contact_arr, cmap='coolwarm',
                s=15, alpha=0.6)
    z2 = np.polyfit(foresight_arr, l1_draft_arr, 1)
    p2 = np.poly1d(z2)
    ax2.plot(x_line, p2(x_line), 'r--', alpha=0.7)
    ax2.set_xlabel('foresight_tac (Smooth L1)', fontsize=11)
    ax2.set_ylabel('l1_draft (Action L1)', fontsize=11)
    ax2.set_title(f'foresight_tac vs l1_draft\n'
                  f'Pearson r={pearson_draft_r:.3f} (p={pearson_draft_p:.1e})', fontsize=11)

    # Right: improvement from draft to final vs foresight_tac
    improvement = l1_draft_arr - l1_final_arr  # positive = A2 better than A1
    imp_r, imp_p = stats.pearsonr(foresight_arr, improvement)
    ax3 = axes[2]
    ax3.scatter(foresight_arr, improvement, c=contact_arr, cmap='coolwarm',
                s=15, alpha=0.6)
    ax3.axhline(0, color='gray', linestyle='--', alpha=0.5)
    z3 = np.polyfit(foresight_arr, improvement, 1)
    p3 = np.poly1d(z3)
    ax3.plot(x_line, p3(x_line), 'r--', alpha=0.7)
    ax3.set_xlabel('foresight_tac (Smooth L1)', fontsize=11)
    ax3.set_ylabel('l1_draft - l1_final (A2 improvement)', fontsize=11)
    ax3.set_title(f'Foresight quality vs A2 improvement\n'
                  f'Pearson r={imp_r:.3f} (p={imp_p:.1e})', fontsize=11)

    fig.suptitle(f'Per-sample Correlation: {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(out_dir, 'correlation_scatter.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nSaved: {save_path}')


if __name__ == '__main__':
    main()
