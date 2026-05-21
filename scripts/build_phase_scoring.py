"""
Build phase-aware multi-dimensional scoring statistics.

For each annotated phase (approach, insertion, pre_bounce, lift, reposition),
compute tactile latent statistics that define "what good contact looks like".

Multi-dimensional scoring:
  1. Phase-fit: z_pred should match the demonstration distribution for current phase
  2. Safety: z_int should not exceed safe bounds (avoid collision/excessive force)
  3. Smoothness: z_delta should be within typical transition range (no sudden jumps)
  4. Consistency: z_int spatial pattern should be symmetric/uniform (task-dependent)

Output: phase_scoring_stats.pkl

Usage:
    python scripts/build_phase_scoring.py --gpu 0
"""
import os, sys, json, pickle, argparse
import numpy as np
import torch
from tqdm import tqdm
import h5py

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'TFAC_V5'))
from tactile_vae import build_tactile_vae


PHASE_NAMES = {0: 'approach', 1: 'insertion', 2: 'pre_bounce', 3: 'lift', 4: 'reposition'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str,
                        default='/home/chenshuai/data/dataset/0209-0210_truncated')
    parser.add_argument('--vae_ckpt', type=str,
                        default='/home/chenshuai/Project/output/tactile_vae_full/best_tactile_vae.pt')
    parser.add_argument('--foresight_horizon', type=int, default=10)
    parser.add_argument('--tac_history', type=int, default=8)
    parser.add_argument('--latent_dim', type=int, default=16)
    parser.add_argument('--output', type=str,
                        default='/home/chenshuai/Project/output/phase_scoring_stats.pkl')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # Load VAE
    print("Loading TactileVAE...")
    vae = build_tactile_vae(latent_dim=args.latent_dim, temporal_window=args.tac_history)
    ckpt = torch.load(args.vae_ckpt, map_location='cpu', weights_only=False)
    sd = ckpt.get('model_state_dict', ckpt)
    vae.load_state_dict(sd)
    vae = vae.to(device).eval()

    TAC_MEAN = np.array([0.2102, -0.6422], dtype=np.float32)
    TAC_STD = np.array([1.6805, 3.6717], dtype=np.float32)

    # Load annotations
    ann_path = os.path.join(args.dataset_dir, 'annotations.pkl')
    with open(ann_path, 'rb') as f:
        annotations = pickle.load(f)
    print(f"Loaded annotations: {len(annotations)} episodes")

    # Collect z per phase + z_delta per phase
    phase_z_collections = {i: [] for i in range(5)}       # z_future by phase
    phase_z_cur_collections = {i: [] for i in range(5)}   # z_current by phase
    phase_delta_collections = {i: [] for i in range(5)}   # z_future - z_current by phase
    phase_intensity_collections = {i: [] for i in range(5)}  # ||marker|| at future by phase

    episode_files = sorted([f for f in os.listdir(args.dataset_dir)
                           if f.startswith('episode_') and f.endswith('.hdf5')])

    for ef in tqdm(episode_files, desc="Processing"):
        ep_key = ef.replace('.hdf5', '')
        if ep_key not in annotations:
            continue

        ann = annotations[ep_key]
        labels = ann['labels']
        ep_type = ann['type']

        path = os.path.join(args.dataset_dir, ef)
        try:
            with h5py.File(path, 'r') as f:
                marker = f['observations/tac/left/marker_offset'][()].astype(np.float32)
        except OSError:
            continue

        ep_len = marker.shape[0]
        if ep_len < args.tac_history + args.foresight_horizon:
            continue

        # Encode z at each valid timestep
        valid_start = args.tac_history - 1
        valid_end = ep_len - args.foresight_horizon

        if valid_end <= valid_start:
            continue

        # Batch encode current and future
        cur_windows = []
        fut_windows = []
        cur_phases = []
        fut_phases = []
        intensities = []

        for t in range(valid_start, valid_end):
            future_t = t + args.foresight_horizon
            cur_phase = labels[t] if t < len(labels) else 0
            fut_phase = labels[future_t] if future_t < len(labels) else labels[-1]

            # Current window
            cs = max(0, t - args.tac_history + 1)
            cw = marker[cs:t+1]
            if cw.shape[0] < args.tac_history:
                pad = np.tile(cw[:1], (args.tac_history - cw.shape[0], 1, 1, 1))
                cw = np.concatenate([pad, cw], axis=0)
            cur_windows.append(cw)

            # Future window
            fs = max(0, future_t - args.tac_history + 1)
            fw = marker[fs:future_t+1]
            if fw.shape[0] < args.tac_history:
                pad = np.tile(fw[:1], (args.tac_history - fw.shape[0], 1, 1, 1))
                fw = np.concatenate([pad, fw], axis=0)
            fut_windows.append(fw)

            cur_phases.append(cur_phase)
            fut_phases.append(fut_phase)

            # Raw intensity at future (force proxy)
            intensity = np.linalg.norm(marker[future_t], axis=-1).mean()
            intensities.append(intensity)

        if not cur_windows:
            continue

        # Batch encode
        batch_size = 128
        z_curs = []
        z_futs = []

        for i in range(0, len(cur_windows), batch_size):
            # Current
            batch_c = np.stack(cur_windows[i:i+batch_size])
            batch_c_norm = (batch_c - TAC_MEAN) / TAC_STD
            with torch.no_grad():
                zc, _ = vae.encode_single_frame(
                    torch.from_numpy(batch_c_norm).float().to(device))
                z_curs.append(zc.flatten(1).cpu().numpy())

            # Future
            batch_f = np.stack(fut_windows[i:i+batch_size])
            batch_f_norm = (batch_f - TAC_MEAN) / TAC_STD
            with torch.no_grad():
                zf, _ = vae.encode_single_frame(
                    torch.from_numpy(batch_f_norm).float().to(device))
                z_futs.append(zf.flatten(1).cpu().numpy())

        z_curs = np.concatenate(z_curs, axis=0)  # (N, 144)
        z_futs = np.concatenate(z_futs, axis=0)   # (N, 144)
        z_deltas = z_futs - z_curs                # (N, 144)

        # Assign to phases (use CURRENT phase as the key - "what phase am I in now?")
        for idx in range(len(cur_phases)):
            phase = cur_phases[idx]
            phase_z_collections[phase].append(z_futs[idx])
            phase_z_cur_collections[phase].append(z_curs[idx])
            phase_delta_collections[phase].append(z_deltas[idx])
            phase_intensity_collections[phase].append(intensities[idx])

    # Compute statistics
    print("\n" + "="*60)
    print("PHASE SCORING STATISTICS")
    print("="*60)

    phase_stats = {}
    for phase_id in range(5):
        name = PHASE_NAMES[phase_id]
        z_arr = np.array(phase_z_collections[phase_id]) if phase_z_collections[phase_id] else None
        delta_arr = np.array(phase_delta_collections[phase_id]) if phase_delta_collections[phase_id] else None
        int_arr = np.array(phase_intensity_collections[phase_id]) if phase_intensity_collections[phase_id] else None

        if z_arr is None or len(z_arr) < 10:
            print(f"\n  Phase {phase_id} ({name}): INSUFFICIENT DATA")
            phase_stats[phase_id] = None
            continue

        # z_future statistics
        z_mu = z_arr.mean(axis=0)
        z_std = z_arr.std(axis=0) + 1e-8

        # z_delta statistics (how much change is typical)
        delta_mu = delta_arr.mean(axis=0)
        delta_std = delta_arr.std(axis=0) + 1e-8
        delta_norm_mu = np.linalg.norm(delta_arr, axis=1).mean()
        delta_norm_std = np.linalg.norm(delta_arr, axis=1).std()

        # Intensity (force) statistics
        int_mu = int_arr.mean()
        int_std = int_arr.std()
        int_p95 = np.percentile(int_arr, 95)  # safety bound
        int_p99 = np.percentile(int_arr, 99)

        # z_int (first 9 dims) statistics
        z_int_arr = z_arr[:, :9]
        z_int_mu = z_int_arr.mean(axis=0)
        z_int_std = z_int_arr.std(axis=0) + 1e-8
        z_int_norm_mu = np.linalg.norm(z_int_arr, axis=1).mean()
        z_int_norm_std = np.linalg.norm(z_int_arr, axis=1).std()
        z_int_norm_p95 = np.percentile(np.linalg.norm(z_int_arr, axis=1), 95)

        # Symmetry: std across spatial positions (lower = more uniform)
        # z_int is (9,) = 3x3 spatial, check uniformity
        z_int_3x3 = z_int_arr.reshape(-1, 3, 3)  # (N, 3, 3)
        uniformity = z_int_3x3.std(axis=(1, 2)).mean()  # avg spatial std

        phase_stats[phase_id] = {
            'name': name,
            'count': len(z_arr),
            'z_mu': z_mu,
            'z_std': z_std,
            'delta_mu': delta_mu,
            'delta_std': delta_std,
            'delta_norm_mu': float(delta_norm_mu),
            'delta_norm_std': float(delta_norm_std),
            'intensity_mu': float(int_mu),
            'intensity_std': float(int_std),
            'intensity_p95': float(int_p95),
            'intensity_p99': float(int_p99),
            'z_int_mu': z_int_mu,
            'z_int_std': z_int_std,
            'z_int_norm_mu': float(z_int_norm_mu),
            'z_int_norm_std': float(z_int_norm_std),
            'z_int_norm_p95': float(z_int_norm_p95),
            'uniformity': float(uniformity),
        }

        print(f"\n  Phase {phase_id} ({name}): {len(z_arr)} samples")
        print(f"    z_future norm: {np.linalg.norm(z_mu):.2f}")
        print(f"    delta norm: mean={delta_norm_mu:.4f}, std={delta_norm_std:.4f}")
        print(f"    intensity: mean={int_mu:.4f}, std={int_std:.4f}, p95={int_p95:.4f}")
        print(f"    z_int norm: mean={z_int_norm_mu:.4f}, p95={z_int_norm_p95:.4f}")
        print(f"    uniformity (spatial std): {uniformity:.4f}")

    # Key comparison: approach vs insertion vs pre_bounce
    print(f"\n{'='*60}")
    print("KEY PHASE COMPARISON (what differentiates good from bad)")
    print(f"{'='*60}")
    print(f"{'Metric':<25} {'approach':<12} {'insertion':<12} {'pre_bounce':<12}")
    print("-"*60)
    for metric in ['delta_norm_mu', 'intensity_mu', 'intensity_p95', 'z_int_norm_mu', 'uniformity']:
        vals = []
        for pid in [0, 1, 2]:
            if phase_stats[pid]:
                vals.append(f"{phase_stats[pid][metric]:.4f}")
            else:
                vals.append("N/A")
        print(f"  {metric:<25} {vals[0]:<12} {vals[1]:<12} {vals[2]:<12}")

    print(f"\n  Key insight:")
    if phase_stats[1] and phase_stats[2]:
        ins_int = phase_stats[1]['intensity_mu']
        pb_int = phase_stats[2]['intensity_mu']
        ins_delta = phase_stats[1]['delta_norm_mu']
        pb_delta = phase_stats[2]['delta_norm_mu']
        print(f"    insertion vs pre_bounce intensity: {ins_int:.4f} vs {pb_int:.4f} "
              f"({'insertion lower' if ins_int < pb_int else 'pre_bounce lower'})")
        print(f"    insertion vs pre_bounce delta: {ins_delta:.4f} vs {pb_delta:.4f} "
              f"({'insertion smoother' if ins_delta < pb_delta else 'pre_bounce smoother'})")

    # Save
    output = {
        'phase_stats': phase_stats,
        'phase_names': PHASE_NAMES,
        'foresight_horizon': args.foresight_horizon,
        'tac_history': args.tac_history,
        'dataset_dir': args.dataset_dir,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump(output, f)
    print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
