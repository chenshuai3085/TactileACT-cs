"""
Foresight 序列动画可视化: 对整个 episode 逐时刻预测, 生成 GIF 动画。

Usage:
    python TFAC/eval_foresight_sequence.py --ckpt_dir /home/chenshuai/data/xiaomi_act/tfac_v5_token+a1_refine_3 --ckpt_name /home/chenshuai//data//xiaomi_act//tfac_v5_token+a1_refine_3/policy_epoch_1700_seed_1.ckpt --episode_id 178 --dataset_dir /home/chenshuai/data/dataset/0401/"""


import argparse
import json
import os 
import pickle
import sys

import h5py
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from TFAC_V3.eval_foresight import build_policy_from_args, plot_quiver_comparison
from utils import NormalizeSeparate, set_seed


IMG_NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])


def load_timestep(root, camera_names, ts, tac_side, tac_img_key,
                  tactile_mode, mo_mean, mo_std):
    """Load all camera data for a single timestep."""
    images = []
    for cam_name in camera_names:
        if cam_name == 'gelsight' and tactile_mode == 'marker':
            mo_path = f'observations/tac/{tac_side}/marker_offset'
            if mo_path in root:
                data = root[mo_path][ts]  # (9, 9, 2)
                data = torch.tensor(data, dtype=torch.float32)
            else:
                data = torch.zeros(9, 9, 2, dtype=torch.float32)
            if mo_mean is not None:
                data = (data - mo_mean) / mo_std
            images.append(data)
        elif cam_name == 'gelsight':
            tac_path = f'observations/tac/{tac_side}/{tac_img_key}'
            if tac_path in root:
                data = root[tac_path][ts]
                data = torch.tensor(data, dtype=torch.float32) / 255.0
                data = torch.einsum('h w c -> c h w', data)
                data = IMG_NORM(data)
            else:
                data = torch.zeros(3, 480, 640, dtype=torch.float32)
            images.append(data)
        else:
            img = root[f'/observations/images/{cam_name}'][ts]
            img = torch.tensor(img, dtype=torch.float32) / 255.0
            img = torch.einsum('h w c -> c h w', img)
            img = IMG_NORM(img)
            images.append(img)
    return images


def main():
    parser = argparse.ArgumentParser(description='Foresight sequence animation')
    parser.add_argument('--ckpt_dir', type=str, required=True)
    parser.add_argument('--ckpt_name', type=str, default='policy_best.ckpt')
    parser.add_argument('--episode_id', type=int, required=True)
    parser.add_argument('--output_format', type=str, default='gif', choices=['gif', 'mp4'])
    parser.add_argument('--fps', type=int, default=5)
    parser.add_argument('--dataset_dir', type=str, default=None,
                        help='Override dataset directory (default: use save_dir/data from args.json)')
    parser.add_argument('--seed', type=int, default=42)
    cli = parser.parse_args()

    set_seed(cli.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load config
    with open(os.path.join(cli.ckpt_dir, 'args.json')) as f:
        args = json.load(f)

    tactile_mode = args.get('tactile_mode', 'image')
    if tactile_mode != 'marker':
        print('This script is designed for tactile_mode=marker.')
        return

    # Load norm stats
    with open(os.path.join(cli.ckpt_dir, 'dataset_stats.pkl'), 'rb') as f:
        norm_stats = pickle.load(f)

    mo_mean = norm_stats.get('marker_offset_mean', None)
    mo_std = norm_stats.get('marker_offset_std', None)
    if mo_mean is not None:
        mo_mean = torch.tensor(mo_mean, dtype=torch.float32)
        mo_std = torch.tensor(mo_std, dtype=torch.float32)
        mo_mean_np = mo_mean.numpy()
        mo_std_np = mo_std.numpy()
        print(f'marker_offset norm: mean={mo_mean_np}, std={mo_std_np}')
    else:
        mo_mean_np = mo_std_np = None
        print('No marker_offset norm stats — using raw values')

    normalizer = NormalizeSeparate(norm_stats)

    # Load metadata
    save_dir = args['save_dir']
    with open(os.path.join(save_dir, 'meta_data.json')) as f:
        meta_data = json.load(f)
    if cli.dataset_dir:
        dataset_dir = cli.dataset_dir
    else:
        dataset_dir = os.path.join(save_dir, 'data')

    camera_names = meta_data['camera_names']
    chunk_size = args['chunk_size']
    foresight_horizon = args.get('foresight_horizon', 8)
    proprio_key = meta_data.get('proprio_key', 'qpos')
    action_key = meta_data.get('action_key', 'action')
    tac_side = meta_data.get('tac_side', 'left')
    tac_img_key = meta_data.get('tac_img_key', 'img')

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
            # 2-layer: 128*3*3=1152, 3-layer: 256*3*3=2304
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

    # Output dirs
    foresight_dir = os.path.join(cli.ckpt_dir, 'foresight_eval')
    gate_dir = os.path.join(cli.ckpt_dir, 'gate_eval')
    frames_dir = os.path.join(foresight_dir, f'episode_{cli.episode_id}_frames')
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(gate_dir, exist_ok=True)

    # Open episode HDF5
    ep_path = os.path.join(dataset_dir, f'episode_{cli.episode_id}.hdf5')
    if not os.path.exists(ep_path):
        print(f'Episode file not found: {ep_path}')
        return

    frame_paths = []
    all_mse = []
    all_errors = []  # list of (9,9,2) error arrays for per-grid stats

    # Gate weight tracking (only for gate fusion mode)
    has_gate = hasattr(policy.model, 'gated_fusion')
    gate_history = []  # list of (g_mem, g_a1, g_fut)

    # Cross-attention tracking (all fusion modes)
    fusion_mode = args.get('fusion_mode', 'gate')
    policy.model.enable_attn_hooks()
    attn_history = []  # list of (t, attn_weights_np)
    print(f'Attention hooks enabled (fusion_mode={fusion_mode})')

    with h5py.File(ep_path, 'r') as root:
        actions_all = root[f'/{action_key}'][()]  # (T, action_dim)
        qpos_all = root[f'/observations/{proprio_key}'][()]  # (T, state_dim)
        episode_len = actions_all.shape[0]

        print(f'\nEpisode {cli.episode_id}: {episode_len} timesteps, '
              f'foresight_horizon={foresight_horizon}')
        print(f'Will generate {episode_len - foresight_horizon} frames\n')

        with torch.inference_mode():
            for t in tqdm(range(episode_len - foresight_horizon),
                          desc='Generating frames'):
                future_t = t + foresight_horizon

                # Load current and future images
                curr_images = load_timestep(
                    root, camera_names, t, tac_side, tac_img_key,
                    tactile_mode, mo_mean, mo_std)
                fut_images = load_timestep(
                    root, camera_names, future_t, tac_side, tac_img_key,
                    tactile_mode, mo_mean, mo_std)

                # qpos and action chunk
                qpos = qpos_all[t]
                action_len = min(episode_len - t, chunk_size)
                action = actions_all[t:t + action_len]

                # Normalize qpos and action
                qpos, action = normalizer(qpos=qpos, action=action)

                # Pad action
                padded_action = np.zeros([chunk_size, action.shape[1]],
                                         dtype=np.float32)
                padded_action[:action_len] = action
                is_pad = np.zeros(chunk_size)
                is_pad[action_len:] = 1

                # To tensors, add batch dim, to device
                qpos_t = torch.from_numpy(qpos).float().unsqueeze(0).to(device)
                action_t = torch.from_numpy(padded_action).float().unsqueeze(0).to(device)
                is_pad_t = torch.from_numpy(is_pad).bool().unsqueeze(0).to(device)
                images_t = [img.unsqueeze(0).to(device) for img in curr_images]
                fut_images_t = [img.unsqueeze(0).to(device) for img in fut_images]

                # Forward
                (a1, a2, t_hat, v_hat, v_gt, t_gt, t_hat_enc, t_cur,
                 (mu, logvar)) = policy.model(
                    qpos_t, images_t, action_t, is_pad_t, fut_images_t,
                    use_predicted_future=True)

                # Record gate weights
                if has_gate:
                    gm, ga, gf = policy.model.gated_fusion._last_gate_means
                    gate_history.append((gm, ga, gf))

                # Record cross-attention weights
                attn_w = policy.model._attn_weights.get('decoder2_cross')
                if attn_w is not None:
                    attn_history.append((t, attn_w.numpy()))

                # To numpy
                t_hat_np = t_hat[0].cpu().numpy()  # (9, 9, 2)
                t_gt_np = t_gt[0].cpu().numpy()    # (9, 9, 2)

                # Denormalize
                if mo_mean_np is not None:
                    t_hat_np = t_hat_np * mo_std_np + mo_mean_np
                    t_gt_np = t_gt_np * mo_std_np + mo_mean_np

                # Metrics
                error = t_hat_np - t_gt_np  # (9,9,2)
                mse = np.mean(error ** 2)
                all_mse.append(mse)
                all_errors.append(error)

                # Plot
                frame_path = os.path.join(frames_dir, f'frame_{t:04d}.png')
                plot_quiver_comparison(
                    t_gt_np, t_hat_np, t,
                    save_path=frame_path,
                    title=f'Episode {cli.episode_id}, t={t} → t+{foresight_horizon}={future_t}  '
                          f'MSE={mse:.2f}')
                frame_paths.append(frame_path)

    # Assemble animation (normalize frame sizes — tight_layout causes ±1px jitter)
    print(f'\nAssembling {len(frame_paths)} frames into {cli.output_format}...')
    target_size = None
    images_for_anim = []
    for p in frame_paths:
        pil = Image.open(p).convert('RGB')
        if target_size is None:
            target_size = pil.size
        elif pil.size != target_size:
            pil = pil.resize(target_size, Image.LANCZOS)
        images_for_anim.append(np.array(pil))

    anim_filename = f'episode_{cli.episode_id}_sequence.{cli.output_format}'
    anim_path = os.path.join(foresight_dir, anim_filename)

    if cli.output_format == 'gif':
        imageio.mimsave(anim_path, images_for_anim, fps=cli.fps, loop=0)
    else:
        imageio.mimsave(anim_path, images_for_anim, fps=cli.fps)

    # Summary
    all_mse = np.array(all_mse)
    all_errors = np.array(all_errors)  # (N, 9, 9, 2)
    print(f'\n=== Episode {cli.episode_id} Summary ===')
    print(f'Frames: {len(all_mse)}')
    print(f'MSE: mean={all_mse.mean():.3f}, std={all_mse.std():.3f}, '
          f'min={all_mse.min():.3f}, max={all_mse.max():.3f}')
    # Mean absolute error (magnitude)
    mae_per_point = np.sqrt((all_errors ** 2).sum(axis=-1))  # (N,9,9)
    print(f'MAE (magnitude): mean={mae_per_point.mean():.3f}, '
          f'std={mae_per_point.std():.3f}')
    print(f'\nFrames saved to: {frames_dir}/')
    print(f'Animation saved to: {anim_path}')

    # --- Error statistics plot ---
    fig_stat, axes_stat = plt.subplots(1, 3, figsize=(20, 6))

    # Left: MSE distribution histogram
    ax = axes_stat[0]
    ax.hist(all_mse, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(all_mse.mean(), color='red', linestyle='--',
               label=f'mean={all_mse.mean():.3f}')
    ax.set_xlabel('Per-sample MSE')
    ax.set_ylabel('Count')
    ax.set_title(f'MSE distribution (mean={all_mse.mean():.3f})')
    ax.legend()

    # Middle: RMSE per grid position (9x9)
    ax2 = axes_stat[1]
    rmse_grid = np.sqrt((all_errors ** 2).mean(axis=(0, 3)))  # (9,9)
    im = ax2.imshow(rmse_grid, cmap='hot', interpolation='nearest')
    ax2.set_xlabel('j')
    ax2.set_ylabel('i')
    ax2.set_title('Average RMSE per grid position')
    plt.colorbar(im, ax=ax2, label='RMSE (pixels)')

    # Right: MSE over time
    ax3 = axes_stat[2]
    ax3.plot(all_mse, color='red', alpha=0.8)
    ax3.set_xlabel('Timestep')
    ax3.set_ylabel('MSE')
    ax3.set_title(f'MSE over time (episode {cli.episode_id})')
    ax3.grid(True, alpha=0.3)

    fig_stat.tight_layout()
    stat_path = os.path.join(foresight_dir, f'episode_{cli.episode_id}_error_statistics.png')
    fig_stat.savefig(stat_path, dpi=150, bbox_inches='tight')
    plt.close(fig_stat)
    print(f'Error statistics saved to: {stat_path}')

    # Gate weight plot
    if gate_history:
        gate_arr = np.array(gate_history)  # (T, 3)
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        # Top: gate weights over time
        ax = axes[0]
        ax.plot(gate_arr[:, 0], label='memory', alpha=0.8)
        ax.plot(gate_arr[:, 1], label='a1_draft', alpha=0.8)
        ax.plot(gate_arr[:, 2], label='future_tac', alpha=0.8)
        ax.set_ylabel('Gate Weight')
        ax.set_title(f'Episode {cli.episode_id} — Gate Weights over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Bottom: MSE over time
        ax2 = axes[1]
        ax2.plot(all_mse, color='red', alpha=0.8, label='foresight MSE')
        ax2.set_xlabel('Timestep')
        ax2.set_ylabel('MSE')
        ax2.set_title('Foresight Tactile MSE over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # X axis: minor ticks every 10 steps, no labels
        for a in axes:
            a.xaxis.set_major_locator(plt.MultipleLocator(10))
            a.tick_params(axis='x', labelbottom=False)
        axes[1].tick_params(axis='x', labelbottom=True)

        plt.tight_layout()
        gate_path = os.path.join(gate_dir,
                                 f'episode_{cli.episode_id}_gate_weights.png')
        plt.savefig(gate_path, dpi=150)
        plt.close()
        print(f'Gate weights plot saved to: {gate_path}')

        # Print gate weight stats
        print(f'\nGate weight averages:')
        print(f'  memory:     {gate_arr[:, 0].mean():.3f} ± {gate_arr[:, 0].std():.3f}')
        print(f'  a1_draft:   {gate_arr[:, 1].mean():.3f} ± {gate_arr[:, 1].std():.3f}')
        print(f'  future_tac: {gate_arr[:, 2].mean():.3f} ± {gate_arr[:, 2].std():.3f}')


    # Cross-attention plot (all fusion modes)
    if attn_history:
        attn_dir = os.path.join(cli.ckpt_dir, 'attention_eval')
        os.makedirs(attn_dir, exist_ok=True)

        timesteps_attn = [item[0] for item in attn_history]
        # Average across action queries → (memory_len,) per step
        attn_avg_per_step = [aw[0].mean(axis=0) for _, aw in attn_history]
        attn_matrix = np.array(attn_avg_per_step)  # (T, memory_len)
        memory_len = attn_matrix.shape[1]

        fig, axes = plt.subplots(3, 1, figsize=(16, 14))

        # Top: full attention heatmap
        ax = axes[0]
        im = ax.imshow(attn_matrix.T, aspect='auto', cmap='hot',
                       interpolation='nearest')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Memory Token Index')
        ax.set_title(f'Episode {cli.episode_id} — Decoder₂ Cross-Attention Map')
        ax.axhline(y=1.5, color='cyan', linestyle='--', alpha=0.5, label='latent|proprio')
        if fusion_mode == "token":
            ax.axhline(y=memory_len - 1.5, color='lime', linestyle='--',
                       alpha=0.7, label='foresight token')
        ax.legend(loc='upper right', fontsize=8)
        plt.colorbar(im, ax=ax, label='attention weight')

        # Middle: key token attention over time
        # Memory layout: [latent, proprio, vision_tokens..., tactile_token, (foresight_token)]
        ax2 = axes[1]
        n_prefix = 2  # latent + proprio
        # tactile token is the last one before foresight (token mode) or the very last (gate/ltd)
        if fusion_mode == "token":
            tac_idx = memory_len - 2   # second to last
            foresight_idx = memory_len - 1
            vision_end = tac_idx       # vision tokens: [2, tac_idx)
        else:
            tac_idx = memory_len - 1   # last token
            foresight_idx = None
            vision_end = tac_idx       # vision tokens: [2, tac_idx)

        # Vision sum (excluding latent, proprio, tactile, foresight)
        if vision_end > n_prefix:
            vision_sum = attn_matrix[:, n_prefix:vision_end].sum(axis=1)
            ax2.plot(timesteps_attn, vision_sum, color='blue', alpha=0.6,
                     label=f'vision sum ({vision_end - n_prefix} tokens)')
        # Tactile token (当前触觉)
        ax2.plot(timesteps_attn, attn_matrix[:, tac_idx], color='purple', linewidth=2,
                 label='tactile (current)')
        # Foresight token (only token mode)
        if foresight_idx is not None:
            ax2.plot(timesteps_attn, attn_matrix[:, foresight_idx], color='green',
                     linewidth=2, label='foresight token')
        # latent and proprio
        ax2.plot(timesteps_attn, attn_matrix[:, 0], color='orange', alpha=0.6,
                 label='latent z')
        ax2.plot(timesteps_attn, attn_matrix[:, 1], color='red', alpha=0.6,
                 label='proprio')
        ax2.set_xlabel('Timestep')
        ax2.set_ylabel('Attention Weight')
        ax2.set_title('Key Token Attention over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Bottom: foresight MSE for comparison
        ax3 = axes[2]
        ax3.plot(all_mse, color='red', alpha=0.8, label='foresight MSE')
        ax3.set_xlabel('Timestep')
        ax3.set_ylabel('MSE')
        ax3.set_title('Foresight Tactile MSE over Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        attn_path = os.path.join(attn_dir,
                                 f'episode_{cli.episode_id}_cross_attention.png')
        plt.savefig(attn_path, dpi=150)
        plt.close()
        print(f'Cross-attention plot saved to: {attn_path}')

        # Save raw data
        np.savez(os.path.join(attn_dir,
                              f'episode_{cli.episode_id}_cross_attention.npz'),
                 timesteps=np.array(timesteps_attn),
                 attn_matrix=attn_matrix)

    policy.model.disable_attn_hooks()


if __name__ == '__main__':
    main()
