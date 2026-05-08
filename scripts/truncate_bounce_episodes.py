"""
Truncate bounce episodes: keep only [last_lift_start - N, end].

For each bounce episode, finds the last lift event and keeps data from
(last_lift_start - pre_frames) to the end. Success episodes are copied as-is.

Output: new directory with re-indexed episode_0.hdf5, episode_1.hdf5, ...

Usage:
    python scripts/truncate_bounce_episodes.py \
        --data_dir /home/chenshuai/data/dataset/0414 \
        --output_dir /home/chenshuai/data/dataset/0414_truncated \
        --pre_frames 10
"""
import argparse
import os
import pickle
import shutil

import h5py
import numpy as np
from tqdm import tqdm


def truncate_episode(src_path, dst_path, start_idx):
    """Copy HDF5 data from start_idx to end into a new file."""
    with h5py.File(src_path, 'r') as src, h5py.File(dst_path, 'w') as dst:
        def copy_truncated(name, obj):
            if isinstance(obj, h5py.Dataset):
                data = obj[start_idx:]
                dst.create_dataset(name, data=data)
            elif isinstance(obj, h5py.Group):
                if name not in dst:
                    dst.create_group(name)
        src.visititems(copy_truncated)


def copy_episode(src_path, dst_path):
    """Copy HDF5 file as-is."""
    shutil.copy2(src_path, dst_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--pre_frames', type=int, default=10,
                        help='Keep N frames before last lift start')
    parser.add_argument('--skip_episodes', type=str, default='',
                        help='Comma-separated episode names to skip (e.g. episode_269,episode_282)')
    args = parser.parse_args()

    ann_path = os.path.join(args.data_dir, 'annotations.pkl')
    if not os.path.exists(ann_path):
        print(f"Error: {ann_path} not found. Run annotate_episodes.py first.")
        return

    with open(ann_path, 'rb') as f:
        ann = pickle.load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    skip_set = set()
    if args.skip_episodes:
        skip_set = set(args.skip_episodes.split(','))
        print(f"  Skipping episodes: {skip_set}")

    success_eps = []
    bounce_eps = []
    for k, v in ann.items():
        if k == '_meta':
            continue
        if k in skip_set:
            continue
        if v['type'] == 'success':
            success_eps.append(k)
        else:
            bounce_eps.append(k)

    print(f"Source: {args.data_dir}")
    print(f"  Success: {len(success_eps)}, Bounce: {len(bounce_eps)}")
    print(f"  Pre-frames: {args.pre_frames}")
    print(f"Output: {args.output_dir}")

    ep_idx = 0
    truncated_lengths = []

    # Copy success episodes
    print(f"\nCopying {len(success_eps)} success episodes...")
    for ep_name in tqdm(success_eps):
        src_path = os.path.join(args.data_dir, f'{ep_name}.hdf5')
        if not os.path.exists(src_path):
            continue
        dst_path = os.path.join(args.output_dir, f'episode_{ep_idx}.hdf5')
        copy_episode(src_path, dst_path)
        ep_idx += 1

    # Truncate bounce episodes
    print(f"\nTruncating {len(bounce_eps)} bounce episodes...")
    n_skipped = 0
    for ep_name in tqdm(bounce_eps):
        src_path = os.path.join(args.data_dir, f'{ep_name}.hdf5')
        if not os.path.exists(src_path):
            continue

        ep_ann = ann[ep_name]
        lifts = ep_ann['lifts']
        if not lifts:
            n_skipped += 1
            continue

        last_lift_start = lifts[-1][0]
        start_idx = max(0, last_lift_start - args.pre_frames)

        # Check remaining length is reasonable
        with h5py.File(src_path, 'r') as f:
            T = f['observations/proprio_joint'].shape[0]
        remaining = T - start_idx
        if remaining < 30:
            n_skipped += 1
            continue

        dst_path = os.path.join(args.output_dir, f'episode_{ep_idx}.hdf5')
        truncate_episode(src_path, dst_path, start_idx)
        truncated_lengths.append(remaining)
        ep_idx += 1

    print(f"\n=== Summary ===")
    print(f"Total output episodes: {ep_idx}")
    print(f"  Success (copied): {len(success_eps)}")
    print(f"  Bounce (truncated): {len(truncated_lengths)}")
    print(f"  Skipped: {n_skipped}")
    if truncated_lengths:
        print(f"  Truncated lengths: min={min(truncated_lengths)}, "
              f"max={max(truncated_lengths)}, mean={np.mean(truncated_lengths):.0f}")


if __name__ == '__main__':
    main()
