#!/usr/bin/env python3
"""Render a dynamic XYZ force curve video for one replay trace."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-npz", required=True)
    parser.add_argument("--trace-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--force-key", default="ft")
    parser.add_argument("--component", choices=["xyz", "z"], default="xyz")
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="Centered moving-average window in steps. 1 disables smoothing.",
    )
    parser.add_argument("--phone-video", default="")
    parser.add_argument("--phone-speed", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--title", default="End-effector force XYZ")
    parser.add_argument("--preview-frame", default="")
    return parser.parse_args()


def read_video_meta(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if frames <= 0 or fps <= 0:
        raise RuntimeError(f"Invalid video metadata for {path}: frames={frames}, fps={fps}")
    return {"frames": frames, "fps": fps, "duration": frames / fps}


def load_trace_elapsed(path: Path, expected_len: int) -> np.ndarray:
    wall_times = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "wall_time" not in (reader.fieldnames or []):
            raise ValueError(f"{path} does not contain a wall_time column")
        for row in reader:
            wall_times.append(float(row["wall_time"]))
    if len(wall_times) != expected_len:
        raise ValueError(f"Trace rows={len(wall_times)} but force rows={expected_len}")
    elapsed = np.asarray(wall_times, dtype=np.float64)
    elapsed -= elapsed[0]
    return elapsed


def draw_text(img: np.ndarray, text: str, xy: tuple[int, int], scale: float = 0.62) -> None:
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (35, 42, 54), 2, cv2.LINE_AA)


def to_plot_points(
    values: np.ndarray,
    x0: int,
    y0: int,
    w: int,
    h: int,
    ymin: float,
    ymax: float,
) -> np.ndarray:
    n = len(values)
    xs = x0 + np.linspace(0, w, n)
    ys = y0 + h - (values - ymin) / max(ymax - ymin, 1e-9) * h
    return np.stack([xs, ys], axis=1).round().astype(np.int32)


def draw_polyline(img: np.ndarray, pts: np.ndarray, color: tuple[int, int, int], thickness: int) -> None:
    if len(pts) >= 2:
        cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, color, thickness, cv2.LINE_AA)


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    if window % 2 == 0:
        window += 1
    pad = window // 2
    kernel = np.ones(window, dtype=np.float32) / float(window)
    if values.ndim == 1:
        padded = np.pad(values, (pad, pad), mode="edge")
        return np.convolve(padded, kernel, mode="valid")
    channels = []
    for i in range(values.shape[1]):
        padded = np.pad(values[:, i], (pad, pad), mode="edge")
        channels.append(np.convolve(padded, kernel, mode="valid"))
    return np.stack(channels, axis=1)


def make_frame(
    force_xyz: np.ndarray,
    elapsed: np.ndarray,
    current_time: float,
    frame_idx: int,
    total_frames: int,
    width: int,
    height: int,
    title: str,
    component: str,
    smooth_window: int,
) -> np.ndarray:
    n = len(force_xyz)
    step_idx = int(np.searchsorted(elapsed, current_time, side="right") - 1)
    step_idx = max(0, min(step_idx, n - 1))

    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    margin_l, margin_r, margin_t, margin_b = 92, 44, 116, 82
    plot_x, plot_y = margin_l, margin_t
    plot_w, plot_h = width - margin_l - margin_r, height - margin_t - margin_b

    if component == "z":
        raw_vals = force_xyz[:, 2:3]
        labels = ["Fz"]
        colors = [(214, 85, 50)]
        pale = [(244, 205, 195)]
    else:
        raw_vals = force_xyz[:, :3]
        labels = ["Fx", "Fy", "Fz"]
        colors = [(30, 92, 220), (34, 150, 82), (214, 85, 50)]
        pale = [(200, 214, 246), (204, 232, 214), (244, 205, 195)]

    vals = moving_average(raw_vals, smooth_window)
    range_vals = np.concatenate([raw_vals.reshape(-1), vals.reshape(-1)])
    ymin = float(np.percentile(range_vals, 1.0))
    ymax = float(np.percentile(range_vals, 99.0))
    pad = max((ymax - ymin) * 0.12, 1.0)
    ymin -= pad
    ymax += pad

    draw_text(canvas, title, (36, 48), scale=0.82)
    draw_text(
        canvas,
        f"step {step_idx + 1:03d}/{n:03d}   trace t={current_time:.2f}/{elapsed[-1]:.2f}s   frame {frame_idx + 1:04d}/{total_frames:04d}",
        (36, 82),
        scale=0.52,
    )

    cv2.rectangle(canvas, (plot_x, plot_y), (plot_x + plot_w, plot_y + plot_h), (210, 216, 224), 2)
    for frac in np.linspace(0.0, 1.0, 6):
        y = int(round(plot_y + plot_h * frac))
        cv2.line(canvas, (plot_x, y), (plot_x + plot_w, y), (232, 236, 241), 1, cv2.LINE_AA)
    for frac in np.linspace(0.0, 1.0, 9):
        x = int(round(plot_x + plot_w * frac))
        cv2.line(canvas, (x, plot_y), (x, plot_y + plot_h), (240, 243, 247), 1, cv2.LINE_AA)

    for i in range(vals.shape[1]):
        raw_pts = to_plot_points(raw_vals[:, i], plot_x, plot_y, plot_w, plot_h, ymin, ymax)
        smooth_pts = to_plot_points(vals[:, i], plot_x, plot_y, plot_w, plot_h, ymin, ymax)
        draw_polyline(canvas, raw_pts, pale[i], 1)
        draw_polyline(canvas, smooth_pts[: step_idx + 1], colors[i], 3)

    cur_x = int(round(plot_x + plot_w * step_idx / max(n - 1, 1)))
    cv2.line(canvas, (cur_x, plot_y), (cur_x, plot_y + plot_h), (35, 42, 54), 2, cv2.LINE_AA)

    for frac, value in zip([0.0, 0.5, 1.0], [ymax, (ymin + ymax) * 0.5, ymin]):
        y = int(round(plot_y + plot_h * frac))
        draw_text(canvas, f"{value:6.1f}", (16, y + 5), scale=0.44)
    draw_text(canvas, "force (N)", (plot_x, plot_y - 8), scale=0.48)
    draw_text(canvas, "step", (plot_x + plot_w - 34, height - 28), scale=0.52)
    draw_text(canvas, "0", (plot_x - 4, height - 28), scale=0.48)
    draw_text(canvas, str(n - 1), (plot_x + plot_w - 68, height - 28), scale=0.48)

    legend_x = plot_x + 12
    legend_y = plot_y + 28
    for i, label in enumerate(labels):
        x = legend_x + i * 118
        cv2.line(canvas, (x, legend_y), (x + 36, legend_y), colors[i], 4, cv2.LINE_AA)
        if smooth_window > 1:
            text = f"{label}={vals[step_idx, i]:.2f}  raw={raw_vals[step_idx, i]:.2f}"
        else:
            text = f"{label}={vals[step_idx, i]:.2f}"
        draw_text(canvas, text, (x + 46, legend_y + 6), scale=0.50)

    if smooth_window > 1:
        draw_text(canvas, f"moving average: {smooth_window} steps", (plot_x + plot_w - 330, plot_y - 8), scale=0.48)

    return canvas


def render(args: argparse.Namespace) -> dict:
    trace_npz = Path(args.trace_npz)
    trace_csv = Path(args.trace_csv)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    data = np.load(trace_npz)
    if args.force_key not in data.files:
        raise KeyError(f"{args.force_key} not found in {trace_npz}; available={data.files}")
    force = data[args.force_key].astype(np.float32)
    if force.ndim != 2 or force.shape[1] < 3:
        raise ValueError(f"Expected force shape T x >=3, got {force.shape}")
    force_xyz = force[:, :3]
    elapsed = load_trace_elapsed(trace_csv, len(force_xyz))

    if args.phone_video:
        phone_meta = read_video_meta(Path(args.phone_video))
        output_duration = phone_meta["duration"] / args.phone_speed
    else:
        output_duration = float(elapsed[-1]) / args.phone_speed
    output_frames = max(2, int(round(output_duration * args.fps)))
    output_duration = output_frames / args.fps

    writer = cv2.VideoWriter(
        str(out),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (args.width, args.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {out}")

    preview = None
    for i in tqdm(range(output_frames), desc=out.name):
        progress = i / max(output_frames - 1, 1)
        current_time = progress * float(elapsed[-1])
        frame = make_frame(
            force_xyz,
            elapsed,
            current_time,
            i,
            output_frames,
            args.width,
            args.height,
            args.title,
            args.component,
            args.smooth_window,
        )
        writer.write(frame)
        if i == output_frames // 2:
            preview = frame.copy()
    writer.release()

    preview_path = None
    if args.preview_frame:
        preview_path = Path(args.preview_frame)
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        if preview is not None:
            cv2.imwrite(str(preview_path), preview)

    summary = {
        "trace_npz": str(trace_npz),
        "trace_csv": str(trace_csv),
        "force_key": args.force_key,
        "component": args.component,
        "smooth_window": int(args.smooth_window),
        "steps": int(len(force_xyz)),
        "trace_duration_s": float(elapsed[-1]),
        "output": str(out),
        "preview_frame": str(preview_path) if preview_path else None,
        "output_frames": int(output_frames),
        "output_fps": float(args.fps),
        "output_duration_s": float(output_duration),
        "phone_video": args.phone_video or None,
        "phone_speed": float(args.phone_speed),
    }
    out.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    print(json.dumps(render(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
