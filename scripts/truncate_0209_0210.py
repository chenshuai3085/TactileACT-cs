"""
Truncate 0209-0210 bounce episodes using GT episode numbering rule.

GT rule:
  Q1 (0-79):   even=success, odd=bounce
  Q2 (80-159): even=success, odd=bounce
  Q3 (160-239): all success
  Q4 (240-319): all bounce

For bounce episodes: keep [last_lift_start - pre_frames, end].
If no lift detected: skip the episode.

Usage:
    python scripts/truncate_0209_0210.py \
        --data_dir /home/chenshuai/data/dataset/0209-0210 \
        --output_dir /home/chenshuai/data/dataset/0209-0210_truncated \
        --pre_frames 10
"""
import argparse
import os
import shutil
import sys

import h5py
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.annotate_episodes import detect_lifts


def is_bounce_gt(ep_idx):
    """Ground-truth bounce label by episode index."""
    if 0 <= ep_idx <= 79:
        return ep_idx % 2 == 1
    elif 80 <= ep_idx <= 159:
        return ep_idx % 2 == 1
    elif 160 <= ep_idx <= 239:
        return False
    elif 240 <= ep_idx <= 319:
        return True
    return False


def truncate_episode(src_path, dst_path, start_idx):
    """Copy HDF5 data from start_idx to end into a new file."""
    with h5py.File(src_path, 'r') as src, h5py.File(dst_path, 'w') as dst:
        def copy_truncated(name, obj):
            if isinstance(obj, h5py.Dataset):
                dst.create_dataset(name, data=obj[start_idx:])
            elif isinstance(obj, h5py.Group):
                if name not in dst:
                    dst.create_group(name)
        src.visititems(copy_truncated)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--pre_frames', type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    ep_idx_out = 0
    n_success_copied = 0
    n_bounce_truncated = 0
    n_bounce_skipped = 0
    truncated_lengths = []

    for i in tqdm(range(320), desc="Processing"):
        src_path = os.path.join(args.data_dir, f'episode_{i}.hdf5')
        if not os.path.exists(src_path):
            continue

        dst_path = os.path.join(args.output_dir, f'episode_{ep_idx_out}.hdf5')

        if not is_bounce_gt(i):
            # Success: copy as-is
            shutil.copy2(src_path, dst_path)
            n_success_copied += 1
            ep_idx_out += 1
        else:
            # Bounce: find last lift, truncate
            with h5py.File(src_path, 'r') as f:
                z = f['observations/proprio_eef'][:, 2]
                T = z.shape[0]

            lifts = detect_lifts(z, z_max=None)
            if not lifts:
                n_bounce_skipped += 1
                continue

            last_lift_start = lifts[-1][0]
            start_idx = max(0, last_lift_start - args.pre_frames)
            remaining = T - start_idx

            if remaining < 30:
                n_bounce_skipped += 1
                continue

            truncate_episode(src_path, dst_path, start_idx)
            truncated_lengths.append(remaining)
            n_bounce_truncated += 1
            ep_idx_out += 1

    print(f"\n=== Summary ===")
    print(f"Total output: {ep_idx_out} episodes")
    print(f"  Success (copied): {n_success_copied}")
    print(f"  Bounce (truncated): {n_bounce_truncated}")
    print(f"  Bounce (skipped, no lift): {n_bounce_skipped}")
    if truncated_lengths:
        print(f"  Truncated lengths: min={min(truncated_lengths)}, "
              f"max={max(truncated_lengths)}, mean={np.mean(truncated_lengths):.0f}")


if __name__ == '__main__':
    main()
