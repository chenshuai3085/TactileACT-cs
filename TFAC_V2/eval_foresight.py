"""
评估 foresight 触觉预测质量。
可视化 GT vs Predicted marker_offset (箭头图), 计算误差统计。

Usage:
    python TFAC/eval_foresight.py --ckpt_dir /path/to/ckpt_dir [--num_samples 8]
"""

import argparse
import json
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from TFAC_V2.dataset import ForesightEpisodicDataset
from TFAC_V2.tfac_policy import TFACPolicy

from utils import get_norm_stats, set_seed


FREEZE_TACTILE = True


def build_policy_from_args(args):
    """Rebuild policy from saved args.json (same logic as train.py)."""
    camera_names = args['camera_names']
    state_dim = args['state_dim']
    tactile_mode = args.get('tactile_mode', 'image')

    pretrained_backbones = None
    camera_backbone_mapping = None

    if args.get('backbone') == 'clip_backbone':
        try:
            from clip_pretraining_xiaomi import modified_resnet18
        except ImportError:
            from clip_pretraining import modified_resnet18
        vision_model = modified_resnet18()
        camera_backbone_mapping = {c: 0 for c in camera_names}

        if tactile_mode == 'image':
            gelsight_model = modified_resnet18()
            camera_backbone_mapping['gelsight'] = 1
            if FREEZE_TACTILE:
                gelsight_model.requires_grad_(False)
            pretrained_backbones = [vision_model, gelsight_model]
        else:
            camera_backbone_mapping['gelsight'] = 0
            pretrained_backbones = [vision_model]

    return TFACPolicy(
        state_dim=state_dim,
        hidden_dim=args.get('hidden_dim', 512),
        position_embedding_type=args.get('position_embedding', 'sine'),
        lr_backbone=args.get('lr_backbone', 1e-5),
        masks=args.get('masks', False),
        backbone_type=args.get('backbone', 'resnet18'),
        dilation=args.get('dilation', False),
        dropout=args.get('dropout', 0.1),
        nheads=args.get('nheads', 8),
        dim_feedforward=args.get('dim_feedforward', 2048),
        num_enc_layers=args.get('enc_layers', 4),
        num_dec_layers=args.get('dec_layers', 7),
        pre_norm=args.get('pre_norm', False),
        num_queries=args.get('chunk_size', 20),
        camera_names=camera_names,
        z_dimension=args.get('z_dimension', 32),
        lr=args.get('lr', 1e-5),
        weight_decay=args.get('weight_decay', 1e-4),
        kl_weight=args.get('kl_weight', 10),
        pretrained_backbones=pretrained_backbones,
        cam_backbone_mapping=camera_backbone_mapping,
        foresight_layers=args.get('foresight_layers', 2),
        foresight_nheads=args.get('foresight_nheads', 4),
        foresight_dim_feedforward=args.get('foresight_dim_feedforward', 2048),
        proj_dim=args.get('proj_dim', 128),
        contrastive_temperature=args.get('contrastive_temperature', 0.07),
        curriculum_ratio=args.get('curriculum_ratio', 0.75),
        lambda_draft=args.get('lambda_draft', 0.5),
        lambda_foresight=args.get('lambda_foresight', 1.0),
        lambda_foresight_vis=args.get('lambda_foresight_vis', 0.3),
        lambda_contrastive=args.get('lambda_contrastive', 0.1),
        num_dec_layers_draft=args.get('dec_layers_draft', None),
        foresight_change_weight=args.get('foresight_change_weight', False),
        tactile_mode=args.get('tactile_mode', 'image'),
        marker_encoder_type=args.get('marker_encoder_type', 'conv2d'),
        fusion_mode=args.get('fusion_mode', 'gate'),
        foresight_tac_decoder=args.get('foresight_tac_decoder', 'linear'),
        spatial_tac_dec_layers=args.get('spatial_tac_dec_layers', 3),
        a2_init=args.get('a2_init', 'zero'),
    )


def plot_quiver_comparison(gt, pred, sample_idx, save_path=None, title=None):
    """
    画 GT vs Predicted marker_offset 箭头图。
    gt, pred: (9, 9, 2) numpy arrays
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Grid positions
    yi, xi = np.meshgrid(np.arange(9), np.arange(9), indexing='ij')
    # Flip y for display (row 0 at top)
    yi_plot = 8 - yi

    for ax, data, label in [
        (axes[0], gt, 'GT future'),
        (axes[1], pred, 'Predicted future'),
        (axes[2], pred - gt, 'Error (pred - GT)'),
    ]:
        u = data[:, :, 0]  # x displacement
        v = data[:, :, 1]  # y displacement
        magnitude = np.sqrt(u**2 + v**2)

        if label == 'Error (pred - GT)':
            color = magnitude
            cmap = 'Reds'
        else:
            color = magnitude
            cmap = 'viridis'

        q = ax.quiver(xi, yi_plot, u, v, color, cmap=cmap, scale=60, width=0.008)
        ax.set_xlim(-0.5, 8.5)
        ax.set_ylim(-0.5, 8.5)
        ax.set_aspect('equal')
        ax.set_title(f'{label}\nmax={magnitude.max():.1f}, mean={magnitude.mean():.1f}')
        ax.grid(True, alpha=0.3)
        plt.colorbar(q, ax=ax, label='magnitude (px)')

    if title is None:
        title = f'Sample {sample_idx} — marker_offset (9x9 grid)'
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  Saved: {save_path}')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate foresight tactile prediction')
    parser.add_argument('--ckpt_dir', type=str, required=True)
    parser.add_argument('--ckpt_name', type=str, default='policy_best.ckpt')
    parser.add_argument('--num_samples', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    cli = parser.parse_args()

    set_seed(cli.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load config
    with open(os.path.join(cli.ckpt_dir, 'args.json')) as f:
        args = json.load(f)

    tactile_mode = args.get('tactile_mode', 'image')
    if tactile_mode != 'marker':
        print('This script is designed for tactile_mode=marker. '
              'Image mode foresight predicts embeddings, not visualizable as arrows.')
        return

    # Load norm stats
    with open(os.path.join(cli.ckpt_dir, 'dataset_stats.pkl'), 'rb') as f:
        norm_stats = pickle.load(f)

    # marker_offset denormalization (if stats available)
    mo_mean = norm_stats.get('marker_offset_mean', None)
    mo_std = norm_stats.get('marker_offset_std', None)
    if mo_mean is not None:
        mo_mean = np.array(mo_mean)  # (2,)
        mo_std = np.array(mo_std)    # (2,)
        print(f'marker_offset norm stats: mean={mo_mean}, std={mo_std}')
    else:
        print('No marker_offset norm stats found — assuming raw values')

    # Build dataset (validation split)
    save_dir = args['save_dir']
    dataset_dir = os.path.join(save_dir, 'data')
    with open(os.path.join(save_dir, 'meta_data.json')) as f:
        meta_data = json.load(f)

    num_episodes = meta_data['num_episodes']
    camera_names = meta_data['camera_names']
    chunk_size = args['chunk_size']
    foresight_horizon = args.get('foresight_horizon', 8)

    np.random.seed(args.get('seed', 1))
    shuffled_indices = np.random.permutation(num_episodes)
    val_indices = shuffled_indices[int(0.8 * num_episodes):]

    val_dataset = ForesightEpisodicDataset(
        val_indices, dataset_dir, camera_names, norm_stats,
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

    # Auto-detect SpatialTactileDecoder layers from checkpoint weights
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
    if result.unexpected_keys:
        print(f'Warning unexpected keys: {result.unexpected_keys}')
    policy.to(device)
    policy.eval()
    print(f'Loaded {ckpt_path}')

    # Output dir
    out_dir = os.path.join(cli.ckpt_dir, 'foresight_eval')
    os.makedirs(out_dir, exist_ok=True)

    # Run evaluation
    all_mse = []
    all_rmse_per_marker = []
    gelsight_idx = camera_names.index('gelsight')

    np.random.seed(cli.seed)
    sample_indices = np.random.choice(len(val_dataset), size=min(cli.num_samples, len(val_dataset)), replace=False)

    print(f'\nEvaluating {len(sample_indices)} samples...\n')

    with torch.inference_mode():
        for i, idx in enumerate(sample_indices):
            all_cam_images, qpos_data, action_data, is_pad, future_cam_images, history_cam_images = val_dataset[idx]

            # To device, add batch dim
            qpos = qpos_data.unsqueeze(0).to(device)
            images = [img.unsqueeze(0).to(device) for img in all_cam_images]
            actions = action_data.unsqueeze(0).to(device)
            is_pad_t = is_pad.unsqueeze(0).to(device)
            future_images = [img.unsqueeze(0).to(device) for img in future_cam_images]

            # Forward (training mode to get foresight outputs)
            (a1, a2, t_hat, v_hat, v_gt, t_gt, t_hat_enc, t_cur, (mu, logvar)) = \
                policy.model(qpos, images, actions, is_pad_t, future_images,
                             use_predicted_future=True)

            # t_hat: (1, 9, 9, 2), t_gt: (1, 9, 9, 2)
            t_hat_np = t_hat[0].cpu().numpy()  # (9, 9, 2)
            t_gt_np = t_gt[0].cpu().numpy()    # (9, 9, 2)

            # Denormalize for visualization and metrics in pixel space
            if mo_mean is not None:
                t_hat_np = t_hat_np * mo_std + mo_mean
                t_gt_np = t_gt_np * mo_std + mo_mean

            # Per-sample metrics
            mse = np.mean((t_hat_np - t_gt_np) ** 2)
            rmse_per_marker = np.sqrt(np.sum((t_hat_np - t_gt_np) ** 2, axis=-1))  # (9, 9)
            gt_magnitude = np.sqrt(np.sum(t_gt_np ** 2, axis=-1))  # (9, 9)

            all_mse.append(mse)
            all_rmse_per_marker.append(rmse_per_marker)

            print(f'Sample {i} (dataset idx {idx}):')
            print(f'  MSE={mse:.3f}, RMSE={np.sqrt(mse):.3f}')
            print(f'  GT magnitude: mean={gt_magnitude.mean():.2f}, max={gt_magnitude.max():.2f}')
            print(f'  Error per marker: mean={rmse_per_marker.mean():.2f}, '
                  f'max={rmse_per_marker.max():.2f}, median={np.median(rmse_per_marker):.2f}')

            # Visualize
            plot_quiver_comparison(
                t_gt_np, t_hat_np, i,
                save_path=os.path.join(out_dir, f'foresight_sample_{i}.png'))

    # Aggregate statistics
    all_mse = np.array(all_mse)
    all_rmse = np.stack(all_rmse_per_marker)  # (N, 9, 9)

    print(f'\n=== Aggregate over {len(all_mse)} samples ===')
    print(f'MSE:  mean={all_mse.mean():.3f}, std={all_mse.std():.3f}, '
          f'min={all_mse.min():.3f}, max={all_mse.max():.3f}')
    print(f'RMSE per marker: mean={all_rmse.mean():.3f}, '
          f'median={np.median(all_rmse):.3f}, max={all_rmse.max():.3f}')

    # Plot error distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(all_mse, bins=20, edgecolor='black')
    axes[0].set_xlabel('Per-sample MSE')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'MSE distribution (mean={all_mse.mean():.3f})')
    axes[0].axvline(all_mse.mean(), color='red', linestyle='--', label=f'mean={all_mse.mean():.3f}')
    axes[0].legend()

    # Spatial error heatmap (average RMSE per grid position)
    spatial_rmse = all_rmse.mean(axis=0)  # (9, 9)
    im = axes[1].imshow(spatial_rmse, cmap='hot', interpolation='nearest')
    axes[1].set_title('Average RMSE per grid position')
    axes[1].set_xlabel('j')
    axes[1].set_ylabel('i')
    plt.colorbar(im, ax=axes[1], label='RMSE (pixels)')

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'error_statistics.png'), dpi=150, bbox_inches='tight')
    print(f'\nSaved to {out_dir}/')


if __name__ == '__main__':
    main()
