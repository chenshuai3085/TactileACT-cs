"""
评估对比学习效果: 相似度矩阵 + 检索准确率。

Usage:
    python TFAC/eval_contrastive.py --ckpt_dir /path/to/ckpt_dir [--num_samples 64]
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from TFAC_V1.dataset import ForesightEpisodicDataset
from TFAC_V1.eval_foresight import build_policy_from_args
from utils import set_seed


def main():
    parser = argparse.ArgumentParser(description='Evaluate contrastive alignment')
    parser.add_argument('--ckpt_dir', type=str, required=True)
    parser.add_argument('--ckpt_name', type=str, default='policy_best.ckpt')
    parser.add_argument('--num_samples', type=int, default=64,
                        help='Number of samples for similarity matrix')
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
    with open(os.path.join(save_dir, 'meta_data.json')) as f:
        meta_data = json.load(f)

    num_episodes = meta_data['num_episodes']
    camera_names = meta_data['camera_names']
    chunk_size = args['chunk_size']
    foresight_horizon = args.get('foresight_horizon', 8)
    tactile_mode = args.get('tactile_mode', 'image')

    np.random.seed(args.get('seed', 1))
    shuffled_indices = np.random.permutation(num_episodes)
    val_indices = shuffled_indices[int(0.8 * num_episodes):]

    val_dataset = ForesightEpisodicDataset(
        val_indices, os.path.join(save_dir, 'data'), camera_names, norm_stats,
        chunk_size=chunk_size, foresight_horizon=foresight_horizon,
        proprio_key=meta_data.get('proprio_key', 'qpos'),
        action_key=meta_data.get('action_key', 'action'),
        tac_side=meta_data.get('tac_side', 'left'),
        tac_img_key=meta_data.get('tac_img_key', 'img'),
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
            print(f'Auto-detected spatial_tac_dec_layers={merged_args["spatial_tac_dec_layers"]}')

    policy = build_policy_from_args(merged_args)
    result = policy.load_state_dict(ckpt, strict=False)
    if result.missing_keys:
        print(f'Warning missing keys: {result.missing_keys}')
    policy.to(device)
    policy.eval()
    print(f'Loaded {ckpt_path}')

    # Collect features
    N = min(cli.num_samples, len(val_dataset))
    np.random.seed(cli.seed)
    sample_indices = np.random.choice(len(val_dataset), size=N, replace=False)

    all_v_gt = []      # GT vision features
    all_t_hat_enc = []  # predicted tactile features (encoded)
    all_v_hat = []      # predicted vision features
    all_t_gt_enc = []   # GT tactile features (encoded)

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

            all_v_gt.append(v_gt[0].cpu())
            all_t_hat_enc.append(t_hat_enc[0].cpu())
            all_v_hat.append(v_hat[0].cpu())

            # Encode GT tactile for comparison
            if tactile_mode == "marker" and t_gt is not None:
                with torch.no_grad():
                    t_gt_enc = policy.model.marker_encoder(t_gt)
                all_t_gt_enc.append(t_gt_enc[0].cpu())

    all_v_gt = torch.stack(all_v_gt)           # (N, D)
    all_t_hat_enc = torch.stack(all_t_hat_enc) # (N, D)
    all_v_hat = torch.stack(all_v_hat)         # (N, D)

    # Use the model's contrastive projection heads
    contrastive = policy.model.contrastive
    contrastive.eval()

    with torch.inference_mode():
        # Project through learned heads
        v_proj = F.normalize(contrastive.v_proj(all_v_gt.to(device)), dim=-1).cpu()
        t_proj = F.normalize(contrastive.t_proj(all_t_hat_enc.to(device)), dim=-1).cpu()

    # --- 1. Similarity matrix ---
    sim_matrix = (v_proj @ t_proj.T).numpy()  # (N, N)

    # --- 2. Retrieval accuracy ---
    # v→t: given v_gt, find the matching t_hat
    v2t_ranks = []
    for i in range(N):
        sims = sim_matrix[i]
        rank = (sims > sims[i]).sum()  # how many negatives scored higher
        v2t_ranks.append(rank)
    v2t_ranks = np.array(v2t_ranks)

    # t→v: given t_hat, find the matching v_gt
    t2v_ranks = []
    for i in range(N):
        sims = sim_matrix[:, i]
        rank = (sims > sims[i]).sum()
        t2v_ranks.append(rank)
    t2v_ranks = np.array(t2v_ranks)

    # Metrics
    r1_v2t = (v2t_ranks == 0).mean() * 100   # Recall@1
    r5_v2t = (v2t_ranks < 5).mean() * 100    # Recall@5
    r1_t2v = (t2v_ranks == 0).mean() * 100
    r5_t2v = (t2v_ranks < 5).mean() * 100

    print(f'\n=== Contrastive Alignment Evaluation ({N} samples) ===')
    print(f'V→T Retrieval:  R@1={r1_v2t:.1f}%  R@5={r5_v2t:.1f}%  '
          f'MedRank={np.median(v2t_ranks):.0f}')
    print(f'T→V Retrieval:  R@1={r1_t2v:.1f}%  R@5={r5_t2v:.1f}%  '
          f'MedRank={np.median(t2v_ranks):.0f}')
    print(f'Diagonal mean sim: {np.diag(sim_matrix).mean():.3f}')
    print(f'Off-diagonal mean sim: {(sim_matrix.sum() - np.trace(sim_matrix)) / (N*N - N):.3f}')

    # --- Plot ---
    out_dir = os.path.join(cli.ckpt_dir, 'contrastive_eval')
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Left: similarity matrix
    ax = axes[0]
    im = ax.imshow(sim_matrix, cmap='viridis', vmin=-0.3, vmax=1.0)
    ax.set_xlabel('Predicted Tactile (t_hat)')
    ax.set_ylabel('GT Vision (v_gt)')
    ax.set_title(f'Similarity Matrix\ndiag={np.diag(sim_matrix).mean():.3f}, '
                 f'off-diag={(sim_matrix.sum() - np.trace(sim_matrix)) / (N*N - N):.3f}')
    plt.colorbar(im, ax=ax)

    # Middle: diagonal vs off-diagonal histogram
    ax2 = axes[1]
    diag_vals = np.diag(sim_matrix)
    off_diag = sim_matrix[~np.eye(N, dtype=bool)]
    ax2.hist(off_diag, bins=50, alpha=0.6, label=f'off-diagonal (n={len(off_diag)})',
             color='gray', density=True)
    ax2.hist(diag_vals, bins=20, alpha=0.8, label=f'diagonal (n={N})',
             color='green', density=True)
    ax2.set_xlabel('Cosine Similarity')
    ax2.set_ylabel('Density')
    ax2.set_title('Diagonal vs Off-diagonal Distribution')
    ax2.legend()
    ax2.axvline(diag_vals.mean(), color='green', linestyle='--', alpha=0.7)
    ax2.axvline(off_diag.mean(), color='gray', linestyle='--', alpha=0.7)

    # Right: retrieval rank distribution
    ax3 = axes[2]
    ax3.hist(v2t_ranks, bins=range(N+1), alpha=0.6, label='V→T', color='blue')
    ax3.hist(t2v_ranks, bins=range(N+1), alpha=0.6, label='T→V', color='red')
    ax3.set_xlabel('Rank (0=correct)')
    ax3.set_ylabel('Count')
    ax3.set_title(f'Retrieval Rank Distribution\n'
                  f'V→T R@1={r1_v2t:.0f}% R@5={r5_v2t:.0f}% | '
                  f'T→V R@1={r1_t2v:.0f}% R@5={r5_t2v:.0f}%')
    ax3.legend()

    plt.tight_layout()
    save_path = os.path.join(out_dir, 'contrastive_eval.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nSaved to: {save_path}')


if __name__ == '__main__':
    main()
