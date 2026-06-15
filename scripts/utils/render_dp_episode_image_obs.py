#!/usr/bin/env python3
"""Render full-episode image observations exactly as DP training consumes them."""

import argparse
import json
import os

import cv2
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


def parse_shape(text):
    values = [int(x) for x in text.split(",")]
    if len(values) != 2:
        raise ValueError(f"Expected H,W but got {text}")
    return tuple(values)


def process_image(raw, resize_transform, crop_transform, image_normalize):
    img_t = torch.from_numpy(np.asarray(raw)).float().div_(255.0).permute(2, 0, 1)
    img_t = resize_transform(img_t)
    img_t = image_normalize(img_t)
    img_t = crop_transform(img_t)
    return img_t


def image_to_display(img_chw):
    arr = img_chw.detach().cpu().numpy().astype(np.float32)
    arr = arr * IMAGENET_STD + IMAGENET_MEAN
    arr = np.clip(arr, 0.0, 1.0)
    return np.moveaxis(arr, 0, -1)


def fig_to_bgr(fig):
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    rgb = rgba[:, :, :3]
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def make_frame(raw_by_cam, proc_by_cam, camera_names, t, obs_indices, out_shape):
    obs_horizon = len(obs_indices)
    fig, axes = plt.subplots(
        2,
        len(camera_names) * obs_horizon,
        figsize=(3.2 * len(camera_names) * obs_horizon, 5.4),
    )
    if axes.ndim == 1:
        axes = axes.reshape(2, -1)

    for cam_i, cam in enumerate(camera_names):
        for obs_i, obs_t in enumerate(obs_indices):
            col = cam_i * obs_horizon + obs_i
            axes[0, col].imshow(raw_by_cam[cam][obs_i])
            axes[0, col].set_title(f"raw {cam}\nobs{obs_i}=t{obs_t}")
            axes[0, col].axis("off")

            axes[1, col].imshow(image_to_display(proc_by_cam[cam][obs_i]))
            axes[1, col].set_title(f"train {cam}\n{tuple(proc_by_cam[cam][obs_i].shape)}")
            axes[1, col].axis("off")

    fig.suptitle(
        f"DP image observation input | current t={t} | obs_indices={obs_indices} | "
        f"train image shape per obs/cam={out_shape}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    frame = fig_to_bgr(fig)
    plt.close(fig)
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--camera_names", default="global,wrist")
    parser.add_argument("--obs_horizon", type=int, default=2)
    parser.add_argument("--resize_shape", default="240,320")
    parser.add_argument("--crop_shape", default="216,288")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--snapshot_count", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    camera_names = [x.strip() for x in args.camera_names.split(",") if x.strip()]
    resize_shape = parse_shape(args.resize_shape)
    crop_shape = parse_shape(args.crop_shape)

    resize_transform = transforms.Resize(resize_shape)
    crop_transform = transforms.CenterCrop(crop_shape)
    image_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    with h5py.File(args.episode, "r") as f:
        lengths = [f[f"observations/images/{cam}"].shape[0] for cam in camera_names]
        T = min(lengths)
        raw_shapes = {
            cam: list(f[f"observations/images/{cam}"].shape)
            for cam in camera_names
        }

    timesteps = list(range(0, T, max(1, args.stride)))
    if args.max_frames is not None:
        timesteps = timesteps[: args.max_frames]
    if not timesteps:
        raise RuntimeError("No timesteps selected")

    video_path = os.path.join(args.output_dir, "episode_dp_image_obs.mp4")
    writer = None
    snapshot_indices = set(np.linspace(0, len(timesteps) - 1, min(args.snapshot_count, len(timesteps)), dtype=int).tolist())
    snapshots = []

    train_image_shape = [3, crop_shape[0], crop_shape[1]]
    with h5py.File(args.episode, "r") as f:
        for frame_i, t in enumerate(timesteps):
            obs_indices = [
                max(0, t - args.obs_horizon + 1 + k)
                for k in range(args.obs_horizon)
            ]
            raw_by_cam = {cam: [] for cam in camera_names}
            proc_by_cam = {cam: [] for cam in camera_names}
            for cam in camera_names:
                ds = f[f"observations/images/{cam}"]
                for obs_t in obs_indices:
                    raw = ds[obs_t]
                    raw_by_cam[cam].append(raw)
                    proc_by_cam[cam].append(process_image(raw, resize_transform, crop_transform, image_normalize))

            frame = make_frame(raw_by_cam, proc_by_cam, camera_names, t, obs_indices, train_image_shape)
            if writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(video_path, fourcc, args.fps, (w, h))
                if not writer.isOpened():
                    raise RuntimeError(f"Failed to open video writer: {video_path}")
            writer.write(frame)

            if frame_i in snapshot_indices:
                snap_path = os.path.join(args.output_dir, f"snapshot_t{t:05d}.png")
                cv2.imwrite(snap_path, frame)
                snapshots.append(snap_path)

    if writer is not None:
        writer.release()

    summary = {
        "episode": args.episode,
        "video": video_path,
        "snapshots": snapshots,
        "camera_names": camera_names,
        "episode_length": int(T),
        "rendered_frames": len(timesteps),
        "stride": args.stride,
        "fps": args.fps,
        "obs_horizon": args.obs_horizon,
        "raw_image_shapes": raw_shapes,
        "resize_shape": list(resize_shape),
        "crop_shape": list(crop_shape),
        "train_image_shape_per_obs_camera": train_image_shape,
        "train_batch_image_shape_per_camera": [args.obs_horizon] + train_image_shape,
    }
    summary_path = os.path.join(args.output_dir, "episode_dp_image_obs_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
