"""Episode-level held-out evaluation for TactileVAE reconstruction.

The training script reports a random window-level validation loss.  That is
useful for monitoring optimization, but it can share episode context with the
training set.  This script builds a deterministic episode-level holdout split
and evaluates reconstruction on complete unseen episodes.
"""

import argparse
import glob
import json
import os
import random
import sys
from collections import defaultdict

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from tactile_vae import TactileVAE


BOARD_DATA_DIRS = [
    "/home/chenshuai/data/dataset/260609/wipe_pos_straight_z124_125_150_20260609",
    "/home/chenshuai/data/dataset/260609/z_too_high",
    "/home/chenshuai/data/dataset/260610/z_too_low",
    "/home/chenshuai/data/dataset/260610/z_too_oscillate",
]


def label_from_path(path):
    parent = os.path.basename(os.path.dirname(path))
    if parent.startswith("wipe_pos"):
        return "positive_straight"
    return parent


def collect_episode_split(data_dirs, test_ratio, seed):
    by_label = defaultdict(list)
    for data_dir in data_dirs:
        for path in sorted(glob.glob(os.path.join(data_dir, "episode_*.hdf5"))):
            by_label[label_from_path(path)].append(path)

    rng = random.Random(seed)
    train, test = [], []
    split_summary = {}
    for label, paths in sorted(by_label.items()):
        paths = list(paths)
        rng.shuffle(paths)
        n_test = max(1, int(round(len(paths) * test_ratio)))
        label_test = sorted(paths[:n_test])
        label_train = sorted(paths[n_test:])
        train.extend(label_train)
        test.extend(label_test)
        split_summary[label] = {
            "train_episodes": len(label_train),
            "test_episodes": len(label_test),
            "test_files": label_test,
        }
    return sorted(train), sorted(test), split_summary


class EpisodeWindowDataset(Dataset):
    def __init__(self, episode_paths, temporal_window, stride, side, mean, std):
        self.temporal_window = temporal_window
        self.mean = mean.reshape(1, 1, 1, 2).astype(np.float32)
        self.std = std.reshape(1, 1, 1, 2).astype(np.float32)
        self.streams = []
        self.windows = []

        for path in episode_paths:
            with h5py.File(path, "r") as h5:
                key = f"observations/tac/{side}/marker_offset"
                if key not in h5:
                    continue
                data = h5[key][()].astype(np.float32)
            if data.shape[0] < temporal_window:
                continue
            stream_idx = len(self.streams)
            self.streams.append((path, data))
            for start in range(0, data.shape[0] - temporal_window + 1, stride):
                self.windows.append((stream_idx, start))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        stream_idx, start = self.windows[idx]
        path, data = self.streams[stream_idx]
        seq = data[start:start + self.temporal_window].copy()
        seq_norm = (seq - self.mean) / self.std
        return {
            "x": torch.from_numpy(seq_norm.astype(np.float32)),
            "raw": torch.from_numpy(seq.astype(np.float32)),
            "label": label_from_path(path),
        }


def compute_active_direction_metrics(gt, pred):
    gt_flat = gt.reshape(-1, 2)
    pred_flat = pred.reshape(-1, 2)
    gt_mag = np.linalg.norm(gt_flat, axis=1)
    pred_mag = np.linalg.norm(pred_flat, axis=1)
    mask = gt_mag > max(1e-6, 0.05 * float(gt_mag.max()))
    if mask.sum() < 3:
        return 0.0, 0.0
    dot = (gt_flat[mask] * pred_flat[mask]).sum(axis=1)
    cos = dot / ((gt_mag[mask] + 1e-8) * (pred_mag[mask] + 1e-8))
    cos = np.clip(cos, -1.0, 1.0)
    ang = np.degrees(np.arccos(cos))
    return float(cos.mean()), float(ang.mean())


def draw_marker(ax, marker, title, vmax, cmap="viridis", quiver=True):
    mag = np.linalg.norm(marker, axis=-1)
    im = ax.imshow(mag, cmap=cmap, vmin=0, vmax=max(vmax, 1e-6))
    if quiver:
        yy, xx = np.mgrid[0:9, 0:9]
        ax.quiver(xx, yy, marker[..., 0], marker[..., 1], color="white",
                  angles="xy", scale_units="xy", scale=max(vmax * 3.5, 1e-6),
                  width=0.006)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def save_examples(examples, out_dir):
    vis_dir = os.path.join(out_dir, "episode_split_examples")
    os.makedirs(vis_dir, exist_ok=True)
    for i, item in enumerate(examples):
        gt = item["gt"]
        pred = item["pred"]
        err = pred - gt
        vmax = max(float(np.linalg.norm(gt, axis=-1).max()),
                   float(np.linalg.norm(pred, axis=-1).max()), 1e-6)
        evmax = max(float(np.linalg.norm(err, axis=-1).max()), 1e-6)
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
        draw_marker(axes[0], gt, f"GT {item['label']}", vmax)
        draw_marker(axes[1], pred, "Recon", vmax)
        draw_marker(axes[2], err, f"Error L2={item['vec_l2']:.3f}", evmax,
                    cmap="magma", quiver=False)
        fig.suptitle(f"held-out example {i} | {item['label']}", fontsize=12)
        fig.tight_layout()
        out_path = os.path.join(vis_dir, f"heldout_example_{i:02d}.png")
        fig.savefig(out_path, dpi=130)
        plt.close(fig)


def evaluate(args):
    os.makedirs(args.out_dir, exist_ok=True)
    train_eps, test_eps, split_summary = collect_episode_split(
        args.data_dirs, args.test_ratio, args.seed)

    with open(args.stats_json) as f:
        stats = json.load(f)
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)

    dataset = EpisodeWindowDataset(
        test_eps, args.temporal_window, args.sample_stride, args.side, mean, std)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    config = ckpt.get("config", {})
    model = TactileVAE(
        latent_dim=int(config.get("latent_dim", args.latent_dim)),
        temporal_window=int(config.get("temporal_window", args.temporal_window)),
        num_freqs=int(config.get("num_freqs", 4)),
        inr_hidden=int(config.get("inr_hidden", 64)),
        kl_weight=float(config.get("kl_weight", 1e-6)),
        direction_weight=float(config.get("direction_weight", 0.2)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    totals = defaultdict(float)
    counts = defaultdict(int)
    examples = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            labels = batch["label"]
            recon, mu, logvar = model(x)
            gt = TactileVAE.align_gt(x, temporal_stride=2)
            loss, recon_loss, kl_loss, dir_loss = model.loss(recon, gt, mu, logvar)

            recon_np = recon.detach().cpu().numpy()
            gt_np = gt.detach().cpu().numpy()
            pred_raw = recon_np * std.reshape(1, 1, 1, 1, 2) + mean.reshape(1, 1, 1, 1, 2)
            gt_raw = gt_np * std.reshape(1, 1, 1, 1, 2) + mean.reshape(1, 1, 1, 1, 2)
            err = pred_raw - gt_raw
            vec_l2 = np.linalg.norm(err, axis=-1).mean(axis=(1, 2, 3))
            raw_mae = np.abs(err).mean(axis=(1, 2, 3, 4))
            norm_mae = np.abs(recon_np - gt_np).mean(axis=(1, 2, 3, 4))

            batch_size = len(labels)
            totals["all_loss"] += float(loss.item()) * batch_size
            totals["all_recon"] += float(recon_loss.item()) * batch_size
            totals["all_kl"] += float(kl_loss.item()) * batch_size
            totals["all_dir"] += float(dir_loss.item()) * batch_size

            for i, label in enumerate(labels):
                totals["all_raw_vec_l2"] += float(vec_l2[i])
                totals["all_raw_mae"] += float(raw_mae[i])
                totals["all_norm_mae"] += float(norm_mae[i])
                counts["all"] += 1

                key = str(label)
                t_mid = pred_raw.shape[1] // 2
                cos, ang = compute_active_direction_metrics(
                    gt_raw[i, t_mid], pred_raw[i, t_mid])
                totals["all_cos"] += cos
                totals["all_ang"] += ang
                totals[f"{key}_raw_vec_l2"] += float(vec_l2[i])
                totals[f"{key}_raw_mae"] += float(raw_mae[i])
                totals[f"{key}_norm_mae"] += float(norm_mae[i])
                totals[f"{key}_cos"] += cos
                totals[f"{key}_ang"] += ang
                counts[key] += 1

                if len(examples) < args.num_examples:
                    examples.append({
                        "label": key,
                        "gt": gt_raw[i, t_mid],
                        "pred": pred_raw[i, t_mid],
                        "vec_l2": float(vec_l2[i]),
                        "cos": cos,
                        "ang": ang,
                    })

    all_count = max(counts["all"], 1)
    metrics = {
        "ckpt": args.ckpt,
        "stats_json": args.stats_json,
        "test_windows": counts["all"],
        "test_episodes": len(test_eps),
        "train_episodes_for_split_only": len(train_eps),
        "sample_stride": args.sample_stride,
        "temporal_window": args.temporal_window,
        "split": split_summary,
        "overall": {
            "loss": totals["all_loss"] / all_count,
            "recon": totals["all_recon"] / all_count,
            "kl": totals["all_kl"] / all_count,
            "dir": totals["all_dir"] / all_count,
            "raw_vec_l2": totals["all_raw_vec_l2"] / all_count,
            "raw_mae": totals["all_raw_mae"] / all_count,
            "norm_mae": totals["all_norm_mae"] / all_count,
            "mid_frame_cos": totals["all_cos"] / all_count,
            "mid_frame_ang_deg": totals["all_ang"] / all_count,
        },
        "by_label": {},
    }
    for key in sorted(k for k in counts if k != "all"):
        n = max(counts[key], 1)
        metrics["by_label"][key] = {
            "windows": counts[key],
            "raw_vec_l2": totals[f"{key}_raw_vec_l2"] / n,
            "raw_mae": totals[f"{key}_raw_mae"] / n,
            "norm_mae": totals[f"{key}_norm_mae"] / n,
            "mid_frame_cos": totals[f"{key}_cos"] / n,
            "mid_frame_ang_deg": totals[f"{key}_ang"] / n,
        }

    with open(os.path.join(args.out_dir, "episode_split_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    save_examples(examples, args.out_dir)
    print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="/home/chenshuai/Project/output/tactile_vae_board_260609_260610_left_tw8_ld16_s2_e150/best_tactile_vae.pt")
    parser.add_argument("--stats_json", default="/home/chenshuai/Project/output/tactile_vae_board_260609_260610_left_tw8_ld16_s2_e150/tactile_norm_stats.json")
    parser.add_argument("--out_dir", default="/home/chenshuai/Project/output/tactile_vae_board_260609_260610_left_tw8_ld16_s2_e150/episode_split_eval")
    parser.add_argument("--data_dirs", nargs="+", default=BOARD_DATA_DIRS)
    parser.add_argument("--side", default="left")
    parser.add_argument("--temporal_window", type=int, default=8)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--sample_stride", type=int, default=4)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_examples", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
