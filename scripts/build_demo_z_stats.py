"""
Build demonstration z_future statistics by episode progress stage.

For each episode, encode future tactile (at t + foresight_horizon) using TactileVAE,
group by progress bin, compute per-bin mean and std.

Output: demo_z_stats.pkl containing:
  - 'stage_stats': list of dicts, each with 'mu' (144,), 'std' (144,), 'count' int
  - 'n_bins': int
  - 'global_mu': (144,)
  - 'global_std': (144,)

Usage:
    python scripts/build_demo_z_stats.py --gpu 0
"""
import os, sys, pickle, argparse
import numpy as np
import torch
from tqdm import tqdm
import h5py

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'TFAC_V5'))

from tactile_vae import build_tactile_vae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str,
                        default='/home/chenshuai/data/dataset/0209-0210_truncated')
    parser.add_argument('--vae_ckpt', type=str,
                        default='/home/chenshuai/Project/output/tactile_vae_full/best_tactile_vae.pt')
    parser.add_argument('--foresight_horizon', type=int, default=10)
    parser.add_argument('--tac_history', type=int, default=8)
    parser.add_argument('--n_bins', type=int, default=10)
    parser.add_argument('--latent_dim', type=int, default=16)
    parser.add_argument('--output', type=str,
                        default='/home/chenshuai/Project/output/demo_z_stats.pkl')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # Load VAE
    print("Loading TactileVAE...")
    vae = build_tactile_vae(latent_dim=args.latent_dim, temporal_window=args.tac_history)
    ckpt = torch.load(args.vae_ckpt, map_location='cpu', weights_only=False)
    if 'model_state_dict' in ckpt:
        vae.load_state_dict(ckpt['model_state_dict'])
    else:
        vae.load_state_dict(ckpt)
    vae = vae.to(device).eval()

    TAC_MEAN = np.array([0.2102, -0.6422], dtype=np.float32)
    TAC_STD = np.array([1.6805, 3.6717], dtype=np.float32)

    # Collect all episodes
    episode_files = sorted([f for f in os.listdir(args.dataset_dir)
                           if f.startswith('episode_') and f.endswith('.hdf5')])
    print(f"Found {len(episode_files)} episodes")

    # For each episode, encode z_future at every valid timestep
    # Group by progress bin
    bin_collections = [[] for _ in range(args.n_bins)]
    all_z = []

    for ef in tqdm(episode_files, desc="Encoding episodes"):
        path = os.path.join(args.dataset_dir, ef)
        try:
            with h5py.File(path, 'r') as f:
                marker = f['observations/tac/left/marker_offset'][()].astype(np.float32)
        except OSError:
            continue

        ep_len = marker.shape[0]
        if ep_len < args.tac_history + args.foresight_horizon:
            continue

        # Encode z_future for each valid timestep
        # Future time = t + foresight_horizon
        # Need marker_window at future time: [t+fh-W+1 : t+fh+1]
        valid_start = args.tac_history - 1
        valid_end = ep_len - args.foresight_horizon

        if valid_end <= valid_start:
            continue

        # Batch encode: collect all future marker windows
        batch_windows = []
        batch_progress = []

        for t in range(valid_start, valid_end):
            future_t = t + args.foresight_horizon
            # marker window ending at future_t
            window_start = max(0, future_t - args.tac_history + 1)
            window = marker[window_start:future_t + 1]
            if window.shape[0] < args.tac_history:
                pad = np.tile(window[:1], (args.tac_history - window.shape[0], 1, 1, 1))
                window = np.concatenate([pad, window], axis=0)

            batch_windows.append(window)
            progress = t / (ep_len - 1)  # 0~1
            batch_progress.append(progress)

        if not batch_windows:
            continue

        # Encode in batches
        batch_size = 128
        z_all_ep = []
        for i in range(0, len(batch_windows), batch_size):
            batch = np.stack(batch_windows[i:i+batch_size])  # (B, 8, 9, 9, 2)
            batch_norm = (batch - TAC_MEAN) / TAC_STD
            batch_t = torch.from_numpy(batch_norm).float().to(device)

            with torch.no_grad():
                z_last, _ = vae.encode_single_frame(batch_t)  # (B, 16, 3, 3)
                z_flat = z_last.flatten(1).cpu().numpy()  # (B, 144)
            z_all_ep.append(z_flat)

        z_all_ep = np.concatenate(z_all_ep, axis=0)  # (N_valid, 144)
        all_z.append(z_all_ep)

        # Assign to bins
        for idx, prog in enumerate(batch_progress):
            bin_idx = min(int(prog * args.n_bins), args.n_bins - 1)
            bin_collections[bin_idx].append(z_all_ep[idx])

    # Compute statistics
    all_z_concat = np.concatenate(all_z, axis=0)
    global_mu = all_z_concat.mean(axis=0)
    global_std = all_z_concat.std(axis=0) + 1e-8

    stage_stats = []
    print(f"\n{'Bin':<5} {'Count':<8} {'mu_norm':<12} {'std_mean':<12}")
    print("-" * 40)
    for i in range(args.n_bins):
        if bin_collections[i]:
            arr = np.stack(bin_collections[i])
            mu = arr.mean(axis=0)
            std = arr.std(axis=0) + 1e-8
            count = len(bin_collections[i])
            stage_stats.append({'mu': mu, 'std': std, 'count': count})
            print(f"{i:<5} {count:<8} {np.linalg.norm(mu):<12.4f} {std.mean():<12.6f}")
        else:
            stage_stats.append({'mu': global_mu, 'std': global_std, 'count': 0})
            print(f"{i:<5} {'EMPTY':<8}")

    # Save
    output = {
        'stage_stats': stage_stats,
        'n_bins': args.n_bins,
        'global_mu': global_mu,
        'global_std': global_std,
        'foresight_horizon': args.foresight_horizon,
        'tac_history': args.tac_history,
        'total_samples': len(all_z_concat),
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump(output, f)
    print(f"\nSaved to {args.output}")
    print(f"Total samples: {len(all_z_concat)}, bins: {args.n_bins}")


if __name__ == '__main__':
    main()
