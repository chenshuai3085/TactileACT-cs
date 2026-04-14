"""
Linear probe: evaluate semantic quality of predicted tactile embeddings.
Freeze embeddings, train a logistic regression to classify contact state.

Usage:
    python TFAC/eval_probe.py --ckpt_dir /path/to/ckpt_dir [--num_samples 500]
"""

import argparse
import json
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from TFAC_V1.dataset import ForesightEpisodicDataset
from TFAC_V1.eval_foresight import build_policy_from_args
from utils import set_seed, load_meta_data


def main():
    parser = argparse.ArgumentParser(description='Linear probe on predicted tactile embeddings')
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

    # Collect embeddings
    N = min(cli.num_samples, len(val_dataset))
    np.random.seed(cli.seed)
    sample_indices = np.random.choice(len(val_dataset), size=N, replace=False)

    all_t_hat_enc = []   # predicted tactile embeddings (512-d)
    all_t_gt_enc = []    # GT tactile embeddings (512-d)
    all_v_gt = []        # GT vision embeddings (512-d)
    all_contact_mag = [] # for labeling

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

            all_t_hat_enc.append(t_hat_enc[0].cpu().numpy())
            all_v_gt.append(v_gt[0].cpu().numpy())

            if tactile_mode == "marker" and t_gt is not None:
                t_gt_enc = policy.model.marker_encoder(t_gt)
                all_t_gt_enc.append(t_gt_enc[0].cpu().numpy())
                mag = torch.sqrt((t_gt[0] ** 2).sum(dim=-1)).mean().item()
                all_contact_mag.append(mag)

    X_pred = np.stack(all_t_hat_enc)    # (N, 512)
    X_gt_tac = np.stack(all_t_gt_enc)   # (N, 512)
    X_vis = np.stack(all_v_gt)          # (N, 512)
    contact_mag = np.array(all_contact_mag)

    # Binary contact label: above median = contact
    thresh = np.median(contact_mag)
    y = (contact_mag > thresh).astype(int)
    print(f'Contact threshold (median): {thresh:.4f}')
    print(f'Label distribution: contact={y.sum()}, no_contact={(1-y).sum()}')

    model_name = os.path.basename(cli.ckpt_dir)
    out_dir = os.path.join(cli.ckpt_dir, 'probe_eval')
    os.makedirs(out_dir, exist_ok=True)

    # --- Linear probe on three types of embeddings ---
    results = {}
    for name, X in [
        ('Pred Tactile', X_pred),
        ('GT Tactile', X_gt_tac),
        ('GT Vision', X_vis),
    ]:
        clf = LogisticRegression(max_iter=1000, random_state=cli.seed, solver='lbfgs')
        scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
        # Also get AUC
        auc_scores = cross_val_score(clf, X, y, cv=5, scoring='roc_auc')

        results[name] = {
            'acc_mean': scores.mean(),
            'acc_std': scores.std(),
            'auc_mean': auc_scores.mean(),
            'auc_std': auc_scores.std(),
        }
        print(f'\n{name}:')
        print(f'  5-fold CV Accuracy: {scores.mean():.3f} +/- {scores.std():.3f}')
        print(f'  5-fold CV AUC:      {auc_scores.mean():.3f} +/- {auc_scores.std():.3f}')

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(results.keys())
    accs = [results[n]['acc_mean'] for n in names]
    acc_stds = [results[n]['acc_std'] for n in names]
    aucs = [results[n]['auc_mean'] for n in names]
    auc_stds = [results[n]['auc_std'] for n in names]

    x = np.arange(len(names))
    w = 0.35
    bars1 = ax.bar(x - w/2, accs, w, yerr=acc_stds, label='Accuracy', capsize=4,
                   color=['#d62728', '#2ca02c', '#1f77b4'])
    bars2 = ax.bar(x + w/2, aucs, w, yerr=auc_stds, label='AUC', capsize=4,
                   color=['#ff9896', '#98df8a', '#aec7e8'])

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Linear Probe: Contact Classification\n{model_name}', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0.4, 1.05)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='chance')

    # Add value labels
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.3f}',
                ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.3f}',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(out_dir, 'probe_results.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nSaved: {save_path}')

    # Print summary
    print(f'\n=== Linear Probe Summary ({model_name}) ===')
    for name in names:
        r = results[name]
        print(f'{name:15s}: Acc={r["acc_mean"]:.3f}+/-{r["acc_std"]:.3f}  '
              f'AUC={r["auc_mean"]:.3f}+/-{r["auc_std"]:.3f}')


if __name__ == '__main__':
    main()
