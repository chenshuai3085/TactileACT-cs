#!/usr/bin/env python3
"""Build a side-by-side camera and tactile marker video from an episode HDF5."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render global camera frames beside a 9x9 tactile marker field."
    )
    parser.add_argument("--input", required=True, help="Input episode HDF5 file.")
    parser.add_argument("--output", required=True, help="Output mp4 path.")
    parser.add_argument(
        "--camera-key",
        default="observations/images/global",
        help="HDF5 dataset for camera frames.",
    )
    parser.add_argument(
        "--marker-key",
        default="observations/tac/right/marker_offset",
        help="HDF5 dataset for marker_offset, shape T x 9 x 9 x 2.",
    )
    parser.add_argument("--fps", type=float, default=20.0, help="Output video FPS.")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride.")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means all frames.")
    parser.add_argument(
        "--raw-marker",
        action="store_true",
        help="Draw raw marker_offset instead of marker_offset relative to frame 0.",
    )
    parser.add_argument(
        "--title",
        default="V8J board wiping: global camera + tactile marker",
        help="Title rendered in the output video.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Output frame width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Output frame height.",
    )
    return parser.parse_args()


def resize_letterbox(img: np.ndarray, width: int, height: int, fill: int = 245) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(width / w, height / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), fill, dtype=np.uint8)
    x0 = (width - new_w) // 2
    y0 = (height - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def draw_label(img: np.ndarray, text: str, org: tuple[int, int], scale: float = 0.58) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (30, 38, 50), 2, cv2.LINE_AA)


def render_marker_panel(
    marker: np.ndarray,
    panel_size: tuple[int, int],
    arrow_scale: float,
    max_mag: float,
) -> np.ndarray:
    width, height = panel_size
    panel = np.full((height, width, 3), 248, dtype=np.uint8)

    mag = np.linalg.norm(marker, axis=-1)
    heat = np.clip(mag / max(max_mag, 1e-6), 0.0, 1.0)
    heat_img = cv2.resize((heat * 255).astype(np.uint8), (width, height), interpolation=cv2.INTER_CUBIC)
    heat_color = cv2.applyColorMap(heat_img, cv2.COLORMAP_TURBO)
    panel = cv2.addWeighted(panel, 0.62, heat_color, 0.38, 0)

    margin = int(min(width, height) * 0.11)
    xs = np.linspace(margin, width - margin, marker.shape[1])
    ys = np.linspace(margin, height - margin, marker.shape[0])

    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            dx, dy = marker[iy, ix]
            start = (int(round(x)), int(round(y)))
            end = (int(round(x + dx * arrow_scale)), int(round(y + dy * arrow_scale)))
            cv2.circle(panel, start, 4, (28, 35, 48), -1, cv2.LINE_AA)
            cv2.arrowedLine(panel, start, end, (0, 45, 120), 2, cv2.LINE_AA, tipLength=0.30)

    cv2.rectangle(panel, (0, 0), (width - 1, height - 1), (210, 216, 224), 2)
    return panel


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.stride < 1:
        raise ValueError("--stride must be >= 1")

    with h5py.File(input_path, "r") as f:
        camera = f[args.camera_key][:]
        marker = f[args.marker_key][:]

    total = min(len(camera), len(marker))
    frame_indices = np.arange(0, total, args.stride)
    if args.max_frames > 0:
        frame_indices = frame_indices[: args.max_frames]
    if len(frame_indices) == 0:
        raise ValueError("No frames selected.")

    marker_vis = marker.astype(np.float32)
    if not args.raw_marker:
        marker_vis = marker_vis - marker_vis[0:1]

    selected_marker = marker_vis[frame_indices]
    mags = np.linalg.norm(selected_marker, axis=-1)
    max_mag = float(np.percentile(mags, 98))
    if max_mag < 1e-6:
        max_mag = float(max(np.max(mags), 1.0))

    canvas_w, canvas_h = args.width, args.height
    top_h = 64
    side_margin = 36
    gap = 24
    panel_w = (canvas_w - side_margin * 2 - gap) // 2
    panel_h = canvas_h - top_h - 58
    camera_h = int(panel_w / 4 * 3)
    camera_h = min(camera_h, panel_h)
    marker_size = min(panel_w, panel_h)
    arrow_scale = (marker_size * 0.085) / max_mag

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, args.fps, (canvas_w, canvas_h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")

    for out_i, src_i in enumerate(frame_indices):
        canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
        draw_label(canvas, args.title, (side_margin, 40), scale=0.72)
        draw_label(canvas, f"frame {int(src_i):04d}/{total - 1:04d}", (canvas_w - 220, 40), scale=0.50)

        cam = camera[src_i]
        if cam.ndim != 3 or cam.shape[2] != 3:
            raise ValueError(f"Camera frame must be HxWx3, got {cam.shape}")
        cam_bgr = cv2.cvtColor(cam, cv2.COLOR_RGB2BGR)
        cam_panel = resize_letterbox(cam_bgr, panel_w, camera_h, fill=245)
        cam_x = side_margin
        cam_y = top_h + (panel_h - camera_h) // 2
        canvas[cam_y : cam_y + camera_h, cam_x : cam_x + panel_w] = cam_panel
        cv2.rectangle(canvas, (cam_x, cam_y), (cam_x + panel_w - 1, cam_y + camera_h - 1), (210, 216, 224), 2)
        draw_label(canvas, "global camera", (cam_x, top_h + 24), scale=0.56)

        marker_panel = render_marker_panel(
            selected_marker[out_i],
            (marker_size, marker_size),
            arrow_scale=arrow_scale,
            max_mag=max_mag,
        )
        marker_x = side_margin + panel_w + gap + (panel_w - marker_size) // 2
        marker_y = top_h + (panel_h - marker_size) // 2
        canvas[marker_y : marker_y + marker_size, marker_x : marker_x + marker_size] = marker_panel
        mean_mag = float(np.mean(mags[out_i]))
        draw_label(canvas, "right tactile marker displacement", (side_margin + panel_w + gap, top_h + 24), scale=0.56)
        draw_label(canvas, f"mean |d|={mean_mag:.4f}", (side_margin + panel_w + gap, canvas_h - 24), scale=0.50)

        writer.write(canvas)

    writer.release()
    print(f"wrote {output_path} ({len(frame_indices)} frames, fps={args.fps:g})")


if __name__ == "__main__":
    main()
