"""
t-SNE visualization in the contrastive projection space (128-d).
Projects embeddings through contrastive heads before t-SNE, like pretrain CLIP t-SNE.

Usage:
    python TFAC/eval_tsne.py --ckpt_dir /path/to/ckpt_dir [--num_samples 200]
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
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from TFAC.dataset import ForesightEpisodicDataset
from TFAC.eval_foresight import build_policy_from_args
from utils import set_seed


def main():
    parser = argparse.ArgumentParser(description='t-SNE in contrastive projection space')
    parser.add_argument('--ckpt_dir', type=str, required=True)
    parser.add_argument('--ckpt_name', type=str, default='policy_best.ckpt')
    parser.add_argument('--num_samples', type=int, default=200,
                        help='Number of samples (each __getitem__ = 1 random timestep from 1 episode)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--perplexity', type=float, default=30)
    cli = parser.parse_args()

    set_seed(cli.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load config
    with open(os.path.join(cli.ckpt_dir, 'args.json')) as f:
        args = json.load(f)

    with open(os.path.join(cli.ckpt_dir, 'dataset_stats.pkl'), 'rb') as f:
        norm_stats = pickle.load(f)

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

    policy = build_policy_from_args(merged_args)
    result = policy.load_state_dict(ckpt, strict=False)
    if result.missing_keys:
        print(f'Warning missing keys: {result.missing_keys}')
    policy.to(device)
    policy.eval()
    print(f'Loaded {ckpt_path}')

    model_name = os.path.basename(cli.ckpt_dir)
    out_dir = os.path.join(cli.ckpt_dir, 'tsne_eval')
    os.makedirs(out_dir, exist_ok=True)

    contrastive = policy.model.contrastive

    # --- Collect embeddings ---
    # Dataset __len__ = num_val_episodes, __getitem__ randomly picks a timestep.
    # We sample repeatedly to get diverse timesteps.
    N = min(cli.num_samples, len(val_dataset) * 3)  # can sample more than episodes
    np.random.seed(cli.seed)

    all_v_proj = []       # GT vision in projection space (128-d)
    all_t_hat_proj = []   # Pred tactile in projection space
    all_t_gt_proj = []    # GT tactile in projection space
    all_v_raw = []        # GT vision raw (512-d)
    all_t_hat_raw = []    # Pred tactile raw (512-d)
    all_t_gt_raw = []     # GT tactile raw (512-d)

    print(f'\nCollecting {N} samples...')
    with torch.inference_mode():
        for i in range(N):
            idx = np.random.randint(len(val_dataset))
            data = val_dataset[idx]
            all_cam_images, qpos_data, action_data, is_pad, future_cam_images = data

            qpos = qpos_data.unsqueeze(0).to(device)
            images = [img.unsqueeze(0).to(device) for img in all_cam_images]
            actions = action_data.unsqueeze(0).to(device)
            is_pad_t = is_pad.unsqueeze(0).to(device)
            future_images = [img.unsqueeze(0).to(device) for img in future_cam_images]

            (a1, a2, t_hat, v_hat, v_gt, t_gt, t_hat_enc, t_cur,
             (mu, logvar)) = policy.model(
                qpos, images, actions, is_pad_t, future_images,
                use_predicted_future=True)

            # Raw embeddings (512-d)
            all_v_raw.append(v_gt[0].cpu().numpy())
            all_t_hat_raw.append(t_hat_enc[0].cpu().numpy())

            # Project through contrastive heads → 128-d
            v_proj = F.normalize(contrastive.v_proj(v_gt), dim=-1)
            t_hat_proj = F.normalize(contrastive.t_proj(t_hat_enc), dim=-1)
            all_v_proj.append(v_proj[0].cpu().numpy())
            all_t_hat_proj.append(t_hat_proj[0].cpu().numpy())

            # GT tactile
            if tactile_mode == "marker" and t_gt is not None:
                t_gt_enc = policy.model.marker_encoder(t_gt)
                all_t_gt_raw.append(t_gt_enc[0].cpu().numpy())
                t_gt_proj = F.normalize(contrastive.t_proj(t_gt_enc), dim=-1)
                all_t_gt_proj.append(t_gt_proj[0].cpu().numpy())

    all_v_proj = np.stack(all_v_proj)
    all_t_hat_proj = np.stack(all_t_hat_proj)
    all_t_gt_proj = np.stack(all_t_gt_proj)
    all_v_raw = np.stack(all_v_raw)
    all_t_hat_raw = np.stack(all_t_hat_raw)
    all_t_gt_raw = np.stack(all_t_gt_raw)
    N_actual = len(all_v_proj)

    # --- 1. t-SNE in PROJECTION space (128-d) ---
    combined_proj = np.vstack([all_v_proj, all_t_hat_proj, all_t_gt_proj])
    print(f'\nRunning t-SNE in projection space ({combined_proj.shape[0]} points, {combined_proj.shape[1]}-d)...')
    perp = min(cli.perplexity, len(combined_proj) // 4)
    tsne = TSNE(n_components=2, perplexity=perp, random_state=cli.seed,
                n_iter=1000, learning_rate='auto', init='pca')
    coords = tsne.fit_transform(combined_proj)

    v_c = coords[:N_actual]
    t_hat_c = coords[N_actual:2*N_actual]
    t_gt_c = coords[2*N_actual:]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: all three modalities
    ax = axes[0]
    ax.scatter(t_gt_c[:, 0], t_gt_c[:, 1], c='#2ca02c', marker='s', s=20,
               alpha=0.5, label='GT Tactile')
    ax.scatter(v_c[:, 0], v_c[:, 1], c='#1f77b4', marker='o', s=20,
               alpha=0.5, label='GT Vision')
    ax.scatter(t_hat_c[:, 0], t_hat_c[:, 1], c='#d62728', marker='^', s=20,
               alpha=0.5, label='Pred Tactile')
    ax.legend(fontsize=11)
    ax.set_title('Projection Space (128-d, after contrastive heads)', fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])

    # Right: GT Vision vs Pred Tactile only, with connecting lines
    ax2 = axes[1]
    ax2.scatter(v_c[:, 0], v_c[:, 1], c='#1f77b4', marker='o', s=25,
                alpha=0.5, label='GT Vision')
    ax2.scatter(t_hat_c[:, 0], t_hat_c[:, 1], c='#d62728', marker='^', s=25,
                alpha=0.5, label='Pred Tactile')
    # Connect same-sample pairs
    for i in range(0, N_actual, max(1, N_actual // 30)):
        ax2.plot([v_c[i, 0], t_hat_c[i, 0]], [v_c[i, 1], t_hat_c[i, 1]],
                 'gray', alpha=0.3, linewidth=0.5)
    ax2.legend(fontsize=11)
    ax2.set_title('GT Vision ↔ Pred Tactile alignment', fontsize=12)
    ax2.set_xticks([]); ax2.set_yticks([])

    fig.suptitle(f't-SNE: {model_name} (Projection Space)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, 'tsne_proj_space.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')

    # --- 2. Similarity matrix (random 64 samples) ---
    n_sim = min(64, N_actual)
    sim_v = all_v_proj[:n_sim]
    sim_t = all_t_hat_proj[:n_sim]
    sim_v_norm = sim_v / (np.linalg.norm(sim_v, axis=1, keepdims=True) + 1e-8)
    sim_t_norm = sim_t / (np.linalg.norm(sim_t, axis=1, keepdims=True) + 1e-8)
    sim_matrix = sim_v_norm @ sim_t_norm.T

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes2[0]
    im = ax.imshow(sim_matrix, cmap='viridis', vmin=-0.3, vmax=1.0)
    ax.set_xlabel('Pred Tactile')
    ax.set_ylabel('GT Vision')
    diag_mean = np.diag(sim_matrix).mean()
    off_diag = (sim_matrix.sum() - np.trace(sim_matrix)) / (n_sim * n_sim - n_sim)
    ax.set_title(f'V-T Similarity (proj space)\ndiag={diag_mean:.3f}, off-diag={off_diag:.3f}')
    plt.colorbar(im, ax=ax)

    ax2 = axes2[1]
    diag_vals = np.diag(sim_matrix)
    off_diag_vals = sim_matrix[~np.eye(n_sim, dtype=bool)]
    ax2.hist(off_diag_vals, bins=50, alpha=0.6, label=f'off-diag (mean={off_diag_vals.mean():.3f})',
             color='gray', density=True)
    ax2.hist(diag_vals, bins=20, alpha=0.8, label=f'diagonal (mean={diag_vals.mean():.3f})',
             color='green', density=True)
    ax2.set_xlabel('Cosine Similarity')
    ax2.set_ylabel('Density')
    ax2.set_title('Diagonal vs Off-diagonal')
    ax2.legend()

    fig2.suptitle(f'{model_name} (Projection Space)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    path2 = os.path.join(out_dir, 'sim_matrix_proj.png')
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path2}')

    # --- 3. Pairwise distances ---
    d_pred_vis = np.diag(cdist(all_t_hat_proj, all_v_proj, 'cosine'))
    d_pred_gt = np.diag(cdist(all_t_hat_proj, all_t_gt_proj, 'cosine'))
    d_vis_gt = np.diag(cdist(all_v_proj, all_t_gt_proj, 'cosine'))

    # Also in raw space for comparison
    d_pred_gt_raw = np.diag(cdist(all_t_hat_raw, all_t_gt_raw, 'cosine'))
    d_pred_vis_raw = np.diag(cdist(all_t_hat_raw, all_v_raw, 'cosine'))

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.hist(d_pred_vis, bins=30, alpha=0.6,
             label=f'Pred Tac ↔ GT Vis (mean={d_pred_vis.mean():.3f})', color='#d62728')
    ax3.hist(d_pred_gt, bins=30, alpha=0.6,
             label=f'Pred Tac ↔ GT Tac (mean={d_pred_gt.mean():.3f})', color='#2ca02c')
    ax3.hist(d_vis_gt, bins=30, alpha=0.6,
             label=f'GT Vis ↔ GT Tac (mean={d_vis_gt.mean():.3f})', color='#1f77b4')
    ax3.set_xlabel('Cosine Distance (projection space)', fontsize=12)
    ax3.set_ylabel('Count', fontsize=12)
    ax3.set_title(f'Pairwise Distances in Projection Space ({model_name})', fontsize=13)
    ax3.legend(fontsize=10)
    plt.tight_layout()
    path3 = os.path.join(out_dir, 'pairwise_distances_proj.png')
    plt.savefig(path3, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path3}')

    # Print summary
    print(f'\n=== Projection Space Distances ({model_name}) ===')
    print(f'Pred Tac ↔ GT Vision:  mean={d_pred_vis.mean():.4f} +/- {d_pred_vis.std():.4f}')
    print(f'Pred Tac ↔ GT Tactile: mean={d_pred_gt.mean():.4f} +/- {d_pred_gt.std():.4f}')
    print(f'GT Vision ↔ GT Tactile: mean={d_vis_gt.mean():.4f} +/- {d_vis_gt.std():.4f}')
    print(f'\n=== Raw Space Distances ({model_name}) ===')
    print(f'Pred Tac ↔ GT Vision (raw):  mean={d_pred_vis_raw.mean():.4f}')
    print(f'Pred Tac ↔ GT Tactile (raw): mean={d_pred_gt_raw.mean():.4f}')


if __name__ == '__main__':
    main()
