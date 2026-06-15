#!/usr/bin/env python3
"""Visualize actual inputs used by diffusion/train_dp_tac_concat.py.

The script reuses DPTacConcatDataset, so processed image panels match the
training resize/crop/normalize path. It also saves raw reference frames and a
JSON file with tensor shapes/ranges for sanity checking.
"""

import argparse
import json
import os
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diffusion"))

from diffusion.train_dp_tac_concat import (  # noqa: E402
    DPTacConcatDataset,
    get_minmax_stats,
)


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


def parse_shape(text):
    values = [int(x) for x in text.split(",")]
    if len(values) != 2:
        raise ValueError(f"Expected H,W shape, got: {text}")
    return tuple(values)


def episode_id(path):
    return int(os.path.basename(path).split("_")[1].split(".")[0])


def list_episode_entries(dataset_dirs):
    entries = []
    for ds_dir in dataset_dirs:
        files = sorted(
            [f for f in os.listdir(ds_dir) if f.startswith("episode_") and f.endswith(".hdf5")],
            key=lambda x: int(x.split("_")[1].split(".")[0]),
        )
        entries.extend((ds_dir, episode_id(f)) for f in files)
    return entries


def find_contact_window(ds_dir, tac_side, tac_history, pred_horizon):
    """Pick the episode/timestep with the largest smoothed marker magnitude."""
    best = None
    files = sorted(
        [f for f in os.listdir(ds_dir) if f.startswith("episode_") and f.endswith(".hdf5")],
        key=lambda x: int(x.split("_")[1].split(".")[0]),
    )
    for ef in files:
        path = os.path.join(ds_dir, ef)
        try:
            with h5py.File(path, "r") as f:
                marker = f[f"observations/tac/{tac_side}/marker_offset"][()].astype(np.float32)
                ep_len = marker.shape[0]
                mag = np.linalg.norm(marker, axis=-1).mean(axis=(1, 2))
                kernel = np.ones(max(1, tac_history), dtype=np.float32) / max(1, tac_history)
                smooth = np.convolve(mag, kernel, mode="same")
                valid_end = max(1, ep_len - pred_horizon)
                t = int(np.argmax(smooth[:valid_end]))
                score = float(smooth[t])
        except Exception as exc:
            print(f"[skip] {path}: {exc}")
            continue
        if best is None or score > best["score"]:
            best = {"ds_dir": ds_dir, "ep_id": episode_id(ef), "start_ts": t, "score": score}
    if best is None:
        raise RuntimeError(f"No valid episode found in {ds_dir}")
    return best


def image_to_display(img_chw):
    arr = img_chw.detach().cpu().numpy().astype(np.float32)
    arr = arr * IMAGENET_STD + IMAGENET_MEAN
    arr = np.clip(arr, 0.0, 1.0)
    return np.moveaxis(arr, 0, -1)


def raw_image(path, cam, t):
    with h5py.File(path, "r") as f:
        return f[f"observations/images/{cam}"][t]


def class_label(ds_dir):
    name = os.path.basename(ds_dir)
    if "wipe_pos" in name:
        return "positive_straight"
    return name


def plot_sample(sample, meta, out_path, raw_frames):
    camera_names = list(sample["images"].keys())
    obs_horizon = sample["qpos"].shape[0]
    tac_history = sample["marker_hist"].shape[1]
    pred_horizon = sample["action"].shape[0]

    rows = 5
    cols = max(len(camera_names) * obs_horizon, tac_history, 4)
    fig = plt.figure(figsize=(2.4 * cols, 12))
    grid = fig.add_gridspec(rows, cols, height_ratios=[1.1, 1.1, 1.2, 1.0, 1.0])

    for cam_idx, cam in enumerate(camera_names):
        for obs_idx in range(obs_horizon):
            col = cam_idx * obs_horizon + obs_idx
            ax = fig.add_subplot(grid[0, col])
            ax.imshow(raw_frames[cam][obs_idx])
            ax.set_title(f"raw {cam} obs{obs_idx}")
            ax.axis("off")

            ax = fig.add_subplot(grid[1, col])
            ax.imshow(image_to_display(sample["images"][cam][obs_idx]))
            shape = tuple(sample["images"][cam][obs_idx].shape)
            ax.set_title(f"train {cam} {shape}")
            ax.axis("off")

    marker_hist = sample["marker_hist"][-1].numpy()
    mags = np.linalg.norm(marker_hist, axis=-1)
    vmax = max(float(mags.max()), 1e-6)
    for i in range(tac_history):
        ax = fig.add_subplot(grid[2, i])
        im = ax.imshow(mags[i], vmin=0.0, vmax=vmax, cmap="magma")
        ax.set_title(f"marker t-{tac_history - 1 - i}")
        ax.axis("off")
    fig.colorbar(im, ax=[fig.axes[-1]], fraction=0.035, pad=0.01)

    ax = fig.add_subplot(grid[3, :])
    qpos = sample["qpos"].numpy()
    for j in range(qpos.shape[-1]):
        ax.plot(np.arange(obs_horizon), qpos[:, j], marker="o", label=f"q{j}")
    ax.set_title(f"normalized qpos input shape={tuple(sample['qpos'].shape)}")
    ax.set_xlabel("obs step")
    ax.set_ylim(-1.15, 1.15)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=min(7, qpos.shape[-1]), fontsize=8)

    ax = fig.add_subplot(grid[4, :])
    action = sample["action"].numpy()
    for j in range(action.shape[-1]):
        ax.plot(np.arange(pred_horizon), action[:, j], label=f"a{j}")
    ax.set_title(f"normalized action target horizon shape={tuple(sample['action'].shape)}")
    ax.set_xlabel("future action step")
    ax.set_ylim(-1.15, 1.15)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=min(7, action.shape[-1]), fontsize=8)

    fig.suptitle(
        f"{meta['label']} | episode_{meta['ep_id']} | start_ts={meta['start_ts']} | "
        f"contact_score={meta['contact_score']:.4f}",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def tensor_summary(x):
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True, help="Comma-separated dataset dirs.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--camera_names", default="global,wrist")
    parser.add_argument("--proprio_key", default="proprio_joint")
    parser.add_argument("--action_key", default="actions/joint_abs")
    parser.add_argument("--tac_side", default="left")
    parser.add_argument("--tac_history", type=int, default=8)
    parser.add_argument("--pred_horizon", type=int, default=16)
    parser.add_argument("--obs_horizon", type=int, default=2)
    parser.add_argument("--resize_shape", default="240,320")
    parser.add_argument("--crop_shape", default="216,288")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    dataset_dirs = [d.strip() for d in args.dataset_dir.split(",") if d.strip()]
    camera_names = [x.strip() for x in args.camera_names.split(",") if x.strip()]
    resize_shape = parse_shape(args.resize_shape)
    crop_shape = parse_shape(args.crop_shape)

    norm_stats = get_minmax_stats(dataset_dirs, args.proprio_key, args.action_key)
    all_entries = list_episode_entries(dataset_dirs)
    dataset = DPTacConcatDataset(
        all_entries,
        camera_names,
        norm_stats,
        pred_horizon=args.pred_horizon,
        obs_horizon=args.obs_horizon,
        tac_history=args.tac_history,
        proprio_key=args.proprio_key,
        action_key=args.action_key,
        tac_side=args.tac_side,
        resize_shape=resize_shape,
        crop_shape=crop_shape,
        is_train=False,
        lazy_images=True,
        image_cache_dir=None,
        max_train_windows=None,
        seed=args.seed,
    )

    index_lookup = {(ep["path"], start_ts): idx for idx, (ep_i, start_ts) in enumerate(dataset.indices)
                    for ep in [dataset.episodes[ep_i]]}

    summary = {
        "dataset_dirs": dataset_dirs,
        "camera_names": camera_names,
        "resize_shape": list(resize_shape),
        "crop_shape": list(crop_shape),
        "pred_horizon": args.pred_horizon,
        "obs_horizon": args.obs_horizon,
        "tac_history": args.tac_history,
        "total_episodes": len(all_entries),
        "total_windows": len(dataset),
        "samples": [],
        "norm_stats": {k: np.asarray(v).tolist() for k, v in norm_stats.items()},
    }

    for ds_dir in dataset_dirs:
        selected = find_contact_window(ds_dir, args.tac_side, args.tac_history, args.pred_horizon)
        ep_path = os.path.join(ds_dir, f"episode_{selected['ep_id']}.hdf5")
        start_ts = selected["start_ts"]
        idx = index_lookup[(ep_path, start_ts)]
        sample = dataset[idx]
        obs_indices = [max(0, start_ts - args.obs_horizon + 1 + k)
                       for k in range(args.obs_horizon)]
        raw_frames = {
            cam: [raw_image(ep_path, cam, t) for t in obs_indices]
            for cam in camera_names
        }
        label = class_label(ds_dir)
        meta = {
            "label": label,
            "dataset_dir": ds_dir,
            "episode_path": ep_path,
            "ep_id": selected["ep_id"],
            "start_ts": start_ts,
            "obs_indices": obs_indices,
            "contact_score": selected["score"],
        }
        out_png = os.path.join(args.output_dir, f"{label}_episode_{selected['ep_id']}_ts{start_ts}_dp_input.png")
        plot_sample(sample, meta, out_png, raw_frames)
        meta["figure"] = out_png
        meta["raw_image_shapes"] = {
            cam: [list(frame.shape) for frame in frames]
            for cam, frames in raw_frames.items()
        }
        meta["train_tensor_shapes"] = {
            "images": {cam: list(sample["images"][cam].shape) for cam in camera_names},
            "marker_hist": list(sample["marker_hist"].shape),
            "qpos": list(sample["qpos"].shape),
            "action": list(sample["action"].shape),
        }
        meta["tensor_summaries"] = {
            "marker_hist": tensor_summary(sample["marker_hist"]),
            "qpos": tensor_summary(sample["qpos"]),
            "action": tensor_summary(sample["action"]),
        }
        for cam in camera_names:
            meta["tensor_summaries"][f"image_{cam}"] = tensor_summary(sample["images"][cam])
        summary["samples"].append(meta)
        print(f"Saved {out_png}")

    summary_path = os.path.join(args.output_dir, "dp_input_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
