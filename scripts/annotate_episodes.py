#!/usr/bin/env python3
"""
Episode annotation script — Z-axis based bounce detection.

Generates frame-level labels for CQF training:
  0 = approach (descending toward socket)
  1 = insertion (final descent, CQF positive sample)
  2 = pre_bounce (15 frames before lift, CQF core negative sample)
  3 = lift (z ascending after collision)
  4 = reposition (stabilizing after lift, before next descent)

Usage:
  python scripts/annotate_episodes.py                        # all known datasets
  python scripts/annotate_episodes.py --data_dir /path/to/ds  # single dataset
  python scripts/annotate_episodes.py --pre_bounce_frames 10   # custom window
"""

import argparse
import glob
import os
import pickle
from datetime import date

import h5py
import numpy as np

# ── constants ──────────────────────────────────────────────────────────────────
DATA_ROOT = "/home/chenshuai/data/dataset"

SPLIT_DATASETS = ["260309", "260310", "260401", "260402", "260403", "260407"]
MERGED_DATASETS = [
    "0414", "260401_0402", "260408_0409", "260417",
    "0209-0210", "0331", "260309_0310",
]

THRESHOLD_Z_VEL = 0.0002
MIN_LIFT_FRAMES = 5
SMOOTH_WINDOW = 5
Z_MAX_FOR_LIFT = 0.180  # only detect lifts when z < 180mm (bounce zone)


def smooth(arr, win=SMOOTH_WINDOW):
    kernel = np.ones(win) / win
    return np.convolve(arr, kernel, mode="same")


def detect_lifts(z, threshold=THRESHOLD_Z_VEL, min_frames=MIN_LIFT_FRAMES,
                  z_max=Z_MAX_FOR_LIFT):
    """Return list of (start, end, rise_mm) for each lift segment.

    Args:
        z_max: only detect lifts starting below this Z value (meters).
               Filters out false positives from velocity fluctuations at high Z.
               Set to None to disable Z range filtering.
    """
    T = len(z)
    z_vel = np.zeros(T)
    z_vel[1:] = z[1:] - z[:-1]
    z_vel_smooth = smooth(z_vel)

    in_lift = False
    lift_start = 0
    lifts = []

    for i in range(T):
        if not in_lift and z_vel_smooth[i] > threshold:
            # Only start a lift if Z is in the bounce zone
            if z_max is not None and z[i] > z_max:
                continue
            in_lift = True
            lift_start = i
        elif in_lift and z_vel_smooth[i] <= threshold:
            in_lift = False
            if i - lift_start >= min_frames:
                rise_mm = (z[min(i, T - 1)] - z[lift_start]) * 1000
                lifts.append((lift_start, i, rise_mm))

    if in_lift and T - lift_start >= min_frames:
        rise_mm = (z[T - 1] - z[lift_start]) * 1000
        lifts.append((lift_start, T, rise_mm))

    return lifts


def find_last_descent_start(z):
    """Find where the final sustained descent begins (for success episodes)."""
    T = len(z)
    z_vel = np.zeros(T)
    z_vel[1:] = z[1:] - z[:-1]
    z_vel_smooth = smooth(z_vel)

    last_neg_start = T // 2
    in_descent = False
    for i in range(T - 1, -1, -1):
        if z_vel_smooth[i] < -1e-5:
            if not in_descent:
                in_descent = True
            last_neg_start = i
        else:
            if in_descent:
                break

    return max(last_neg_start, 1)


def annotate_episode(z, pre_bounce_frames=15, z_max=Z_MAX_FOR_LIFT):
    """Return (labels, lifts, episode_type)."""
    T = len(z)
    lifts = detect_lifts(z, z_max=z_max)
    labels = np.zeros(T, dtype=np.int32)

    if len(lifts) == 0:
        # success: no lifts detected
        descent_start = find_last_descent_start(z)
        labels[:descent_start] = 0   # approach
        labels[descent_start:] = 1   # insertion
        return labels, lifts, "success"

    # bounce episode — label lift and reposition segments first
    for start, end, _ in lifts:
        labels[start:end] = 3  # lift

    # pre_bounce: N frames before each lift
    for start, end, _ in lifts:
        pb_start = max(0, start - pre_bounce_frames)
        for j in range(pb_start, start):
            if labels[j] == 0:
                labels[j] = 2  # pre_bounce

    # reposition: from lift end to next descent
    z_vel = np.zeros(T)
    z_vel[1:] = z[1:] - z[:-1]
    z_vel_smooth = smooth(z_vel)

    for idx, (start, end, _) in enumerate(lifts):
        repo_end = end
        for j in range(end, T):
            if z_vel_smooth[j] < -1e-5:
                break
            repo_end = j + 1
        if idx < len(lifts) - 1:
            repo_end = min(repo_end, lifts[idx + 1][0])
        labels[end:repo_end] = 4  # reposition

    # insertion: after last lift's reposition, where z is descending
    last_lift_end = lifts[-1][1]
    repo_end_last = last_lift_end
    for j in range(last_lift_end, T):
        if z_vel_smooth[j] < -1e-5:
            break
        repo_end_last = j + 1
    labels[repo_end_last:] = 1  # insertion

    # everything still 0 is approach
    return labels, lifts, "bounce"


def process_directory(data_dir, pre_bounce_frames=15, z_max=Z_MAX_FOR_LIFT):
    """Process all episodes in a directory (flat or with success/bounce subdirs)."""
    annotations = {}
    episode_files = []

    # check for success/bounce subdirs
    success_dir = os.path.join(data_dir, "success")
    bounce_dir = os.path.join(data_dir, "bounce")

    if os.path.isdir(success_dir) or os.path.isdir(bounce_dir):
        for subdir in [success_dir, bounce_dir]:
            if os.path.isdir(subdir):
                files = sorted(glob.glob(os.path.join(subdir, "episode_*.hdf5")))
                episode_files.extend(files)
    else:
        episode_files = sorted(glob.glob(os.path.join(data_dir, "episode_*.hdf5")))

    if not episode_files:
        print(f"  [SKIP] No episodes found in {data_dir}")
        return None

    stats = {
        "total": 0, "success": 0, "bounce": 0,
        "label_frames": {i: 0 for i in range(5)},
        "lift_counts": [],
    }

    for fpath in episode_files:
        ep_name = os.path.splitext(os.path.basename(fpath))[0]
        try:
            with h5py.File(fpath, "r") as f:
                eef = f["observations/proprio_eef"][:]
        except Exception as e:
            print(f"  [WARN] {ep_name}: {e}")
            continue

        z = eef[:, 2]
        labels, lifts, ep_type = annotate_episode(z, pre_bounce_frames, z_max=z_max)

        annotations[ep_name] = {
            "labels": labels,
            "lifts": lifts,
            "z": z.astype(np.float32),
            "type": ep_type,
        }

        stats["total"] += 1
        stats[ep_type] += 1
        stats["lift_counts"].append(len(lifts))
        for lbl in range(5):
            stats["label_frames"][lbl] += int(np.sum(labels == lbl))

    annotations["_meta"] = {
        "version": 2,
        "threshold_z_vel": THRESHOLD_Z_VEL,
        "min_lift_frames": MIN_LIFT_FRAMES,
        "pre_bounce_frames": pre_bounce_frames,
        "z_max_for_lift": z_max,
        "smooth_window": SMOOTH_WINDOW,
        "created": str(date.today()),
    }

    out_path = os.path.join(data_dir, "annotations.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(annotations, f)

    return stats


def print_stats(name, stats):
    label_names = {0: "approach", 1: "insertion", 2: "pre_bounce", 3: "lift", 4: "reposition"}
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Episodes: {stats['total']}  (success={stats['success']}, bounce={stats['bounce']})")
    print(f"  Frame counts per label:")
    total_frames = sum(stats["label_frames"].values())
    for lbl in range(5):
        cnt = stats["label_frames"][lbl]
        pct = 100 * cnt / total_frames if total_frames > 0 else 0
        print(f"    {lbl} ({label_names[lbl]:>12}): {cnt:>8} frames  ({pct:5.1f}%)")

    if stats["bounce"] > 0:
        lc = [c for c in stats["lift_counts"] if c > 0]
        if lc:
            from collections import Counter
            dist = Counter(lc)
            print(f"  Lift count distribution (bounce eps): {dict(sorted(dist.items()))}")


def main():
    parser = argparse.ArgumentParser(description="Z-axis bounce annotation")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Single data directory to process")
    parser.add_argument("--data_root", type=str, default=DATA_ROOT)
    parser.add_argument("--pre_bounce_frames", type=int, default=15)
    parser.add_argument("--z_max", type=float, default=Z_MAX_FOR_LIFT,
                        help="Only detect lifts when Z < this value (meters). "
                             "Default=0.180 (180mm). Set to 0 to disable.")
    args = parser.parse_args()

    # z_max=0 means disabled
    if args.z_max <= 0:
        args.z_max = None

    if args.data_dir:
        dirs = [args.data_dir]
    else:
        dirs = []
        for name in SPLIT_DATASETS + MERGED_DATASETS:
            d = os.path.join(args.data_root, name)
            if os.path.isdir(d):
                dirs.append(d)

    print(f"Processing {len(dirs)} dataset(s) with pre_bounce_frames={args.pre_bounce_frames}")

    global_stats = {
        "total": 0, "success": 0, "bounce": 0,
        "label_frames": {i: 0 for i in range(5)},
        "lift_counts": [],
    }

    for data_dir in dirs:
        name = os.path.basename(data_dir)
        print(f"\nProcessing {name}...")
        stats = process_directory(data_dir, args.pre_bounce_frames, z_max=args.z_max)
        if stats is None:
            continue

        print_stats(name, stats)

        global_stats["total"] += stats["total"]
        global_stats["success"] += stats["success"]
        global_stats["bounce"] += stats["bounce"]
        for lbl in range(5):
            global_stats["label_frames"][lbl] += stats["label_frames"][lbl]
        global_stats["lift_counts"].extend(stats["lift_counts"])

    print_stats("GLOBAL SUMMARY", global_stats)

    total_pos = global_stats["label_frames"][1]
    total_neg = global_stats["label_frames"][2]
    print(f"\n  CQF training samples:")
    print(f"    Positive (insertion):   {total_pos} frames")
    print(f"    Negative (pre_bounce):  {total_neg} frames")
    print(f"    Ratio pos:neg = 1:{total_neg/total_pos:.1f}" if total_pos > 0 else "")
    print(f"\nDone. annotations.pkl saved in each dataset directory.")


if __name__ == "__main__":
    main()
