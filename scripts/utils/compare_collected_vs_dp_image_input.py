#!/usr/bin/env python3
"""Compare collected raw images with DP processed image inputs."""

import argparse
import json
import os

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


def parse_shape(text):
    vals = [int(x) for x in text.split(",")]
    if len(vals) != 2:
        raise ValueError(f"Expected H,W, got {text}")
    return tuple(vals)


def to_processed(raw, resize, crop, normalize):
    img = torch.from_numpy(np.asarray(raw)).float().div_(255.0).permute(2, 0, 1)
    resized = resize(img)
    normed = normalize(resized)
    cropped = crop(normed)
    return resized, cropped


def chw_to_rgb01(chw):
    arr = chw.detach().cpu().numpy().astype(np.float32)
    if arr.min() < -0.1 or arr.max() > 1.1:
        arr = arr * STD + MEAN
    arr = np.clip(arr, 0, 1)
    return np.moveaxis(arr, 0, -1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--camera_names", default="global,wrist")
    parser.add_argument("--timesteps", default="0,115,230,345,465,580,695,815")
    parser.add_argument("--resize_shape", default="240,320")
    parser.add_argument("--crop_shape", default="216,288")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cameras = [x.strip() for x in args.camera_names.split(",") if x.strip()]
    requested_ts = [int(x) for x in args.timesteps.split(",") if x.strip()]
    resize_shape = parse_shape(args.resize_shape)
    crop_shape = parse_shape(args.crop_shape)

    resize = transforms.Resize(resize_shape)
    crop = transforms.CenterCrop(crop_shape)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    with h5py.File(args.episode, "r") as f:
        T = min(f[f"observations/images/{cam}"].shape[0] for cam in cameras)
        timesteps = [min(max(0, t), T - 1) for t in requested_ts]
        raw_shapes = {cam: list(f[f"observations/images/{cam}"].shape) for cam in cameras}

        fig, axes = plt.subplots(
            len(timesteps) * len(cameras),
            3,
            figsize=(12, 3.0 * len(timesteps) * len(cameras)),
        )
        if axes.ndim == 1:
            axes = axes.reshape(1, -1)

        row = 0
        for t in timesteps:
            for cam in cameras:
                raw = f[f"observations/images/{cam}"][t]
                resized, cropped = to_processed(raw, resize, crop, normalize)

                axes[row, 0].imshow(raw)
                axes[row, 0].set_title(f"collected raw {cam} t={t}\nHWC={raw.shape}, dtype={raw.dtype}")
                axes[row, 0].axis("off")

                axes[row, 1].imshow(chw_to_rgb01(resized))
                axes[row, 1].set_title(f"after Resize\nCHW={tuple(resized.shape)}")
                axes[row, 1].axis("off")

                axes[row, 2].imshow(chw_to_rgb01(cropped))
                axes[row, 2].set_title(f"DP train input after Normalize+CenterCrop\nCHW={tuple(cropped.shape)}")
                axes[row, 2].axis("off")
                row += 1

    fig.suptitle("Collected image vs DP image input", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out_png = os.path.join(args.output_dir, "collected_vs_dp_image_input.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    summary = {
        "episode": args.episode,
        "figure": out_png,
        "camera_names": cameras,
        "episode_length": int(T),
        "timesteps": timesteps,
        "raw_collected_shapes": raw_shapes,
        "resize_shape_hw": list(resize_shape),
        "dp_crop_shape_hw": list(crop_shape),
        "dp_train_image_shape_per_obs_camera_chw": [3, crop_shape[0], crop_shape[1]],
        "dp_train_batch_shape_per_camera_with_obs_horizon_2": [2, 3, crop_shape[0], crop_shape[1]],
    }
    out_json = os.path.join(args.output_dir, "collected_vs_dp_image_input_summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
