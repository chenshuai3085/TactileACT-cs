#!/usr/bin/env python3
"""Compose a phone operation video with GT and predicted tactile marker videos."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from render_trial_marker_deformed_grid_videos import (
    draw_label,
    marker_vmax,
    render_deformed_grid_panel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phone-video", required=True)
    parser.add_argument("--marker-npz", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--phone-speed", type=float, required=True)
    parser.add_argument(
        "--trace-csv",
        default="",
        help=(
            "Optional replay trace.csv. When provided, marker frames are sampled by "
            "the recorded wall_time axis, preserving every execution pause."
        ),
    )
    parser.add_argument(
        "--marker-sampling",
        choices=["hold", "linear"],
        default="hold",
        help="Sampling mode when --trace-csv is used. hold preserves pause segments.",
    )
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--phone-width", type=int, default=1120)
    parser.add_argument("--phone-height", type=int, default=840)
    parser.add_argument("--marker-size", type=int, default=430)
    parser.add_argument("--title", default="Board wiping: real operation + tactile foresight")
    parser.add_argument(
        "--preview-frame",
        default="",
        help="Optional path for saving one preview frame from the composed video.",
    )
    return parser.parse_args()


def read_video_meta(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frames <= 0 or fps <= 0:
        raise RuntimeError(f"Invalid video metadata for {path}: frames={frames}, fps={fps}")
    return {"frames": frames, "fps": fps, "width": width, "height": height}


def load_trace_elapsed(path: Path, expected_len: int) -> np.ndarray:
    wall_times = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "wall_time" not in (reader.fieldnames or []):
            raise ValueError(f"{path} does not contain a wall_time column")
        for row in reader:
            wall_times.append(float(row["wall_time"]))
    if not wall_times:
        raise ValueError(f"No rows found in {path}")
    elapsed = np.asarray(wall_times, dtype=np.float64)
    elapsed -= elapsed[0]
    if len(elapsed) != expected_len:
        raise ValueError(
            f"Trace length mismatch: {path} has {len(elapsed)} rows, marker has {expected_len} frames"
        )
    if np.any(np.diff(elapsed) < 0):
        raise ValueError(f"Trace wall_time is not monotonic in {path}")
    return elapsed


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


def sample_marker(markers: np.ndarray, pos: float) -> np.ndarray:
    if len(markers) == 1:
        return markers[0]
    lo = int(np.floor(pos))
    hi = min(lo + 1, len(markers) - 1)
    alpha = float(pos - lo)
    return markers[lo] * (1.0 - alpha) + markers[hi] * alpha


def sample_marker_from_trace(
    markers: np.ndarray,
    elapsed: np.ndarray,
    t: float,
    sampling: str,
) -> tuple[np.ndarray, float, int]:
    if len(markers) == 1:
        return markers[0], 0.0, 0

    if sampling == "linear":
        if t <= elapsed[0]:
            return markers[0], 0.0, 0
        if t >= elapsed[-1]:
            last = len(markers) - 1
            return markers[last], float(last), last
        hi = int(np.searchsorted(elapsed, t, side="right"))
        lo = max(0, hi - 1)
        hi = min(hi, len(markers) - 1)
        denom = max(float(elapsed[hi] - elapsed[lo]), 1e-9)
        alpha = float((t - elapsed[lo]) / denom)
        marker = markers[lo] * (1.0 - alpha) + markers[hi] * alpha
        return marker, lo + alpha, lo

    idx = int(np.searchsorted(elapsed, t, side="right") - 1)
    idx = max(0, min(idx, len(markers) - 1))
    return markers[idx], float(idx), idx


def read_monotonic_frame(cap: cv2.VideoCapture, target_idx: int, state: dict) -> np.ndarray:
    """Read frames forward until target_idx and return that frame.

    OpenCV random seeking is very slow on long phone videos, while this workload
    always samples source frames in increasing order.
    """
    if target_idx < state["idx"]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        state["idx"] = target_idx - 1
        state["frame"] = None

    while state["idx"] < target_idx:
        ok, frame = cap.read()
        if not ok:
            if state["frame"] is not None:
                return state["frame"]
            raise RuntimeError(f"Could not read phone frame {target_idx}")
        state["idx"] += 1
        state["frame"] = frame

    if state["frame"] is None:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Could not read initial phone frame {target_idx}")
        state["idx"] += 1
        state["frame"] = frame
    return state["frame"]


def draw_panel_label(canvas: np.ndarray, text: str, x: int, y: int) -> None:
    draw_label(canvas, text, (x, y), scale=0.62)


def compose(args: argparse.Namespace) -> dict:
    phone_path = Path(args.phone_video)
    marker_path = Path(args.marker_npz)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    phone_meta = read_video_meta(phone_path)
    phone_cap = cv2.VideoCapture(str(phone_path))
    marker_data = np.load(marker_path)
    gt = marker_data["gt"].astype(np.float32)
    pred = marker_data["pred"].astype(np.float32)
    if gt.shape != pred.shape or gt.ndim != 4 or gt.shape[-3:] != (9, 9, 2):
        raise ValueError(f"Expected gt/pred marker shape T x 9 x 9 x 2, got {gt.shape} and {pred.shape}")

    phone_duration = phone_meta["frames"] / phone_meta["fps"]
    output_duration = phone_duration / args.phone_speed
    output_frames = max(2, int(round(output_duration * args.fps)))
    output_duration = output_frames / args.fps
    trace_elapsed = None
    trace_duration = None
    if args.trace_csv:
        trace_elapsed = load_trace_elapsed(Path(args.trace_csv), len(gt))
        trace_duration = float(trace_elapsed[-1])

    vmax = marker_vmax(gt, pred)
    displacement_scale = (args.marker_size * 0.085) / max(vmax, 1e-6)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, args.fps, (args.width, args.height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer: {output_path}")

    margin = 36
    top_h = 82
    phone_x = margin
    phone_y = top_h + 56
    right_x = phone_x + args.phone_width + 44
    marker_gt_y = top_h + 32
    marker_pred_y = marker_gt_y + args.marker_size + 72

    phone_state = {"idx": -1, "frame": None}
    preview_frame = None
    for out_i in tqdm(range(output_frames), desc=output_path.name):
        denom = max(1, output_frames - 1)
        progress = out_i / denom
        phone_idx = int(round(progress * (phone_meta["frames"] - 1)))
        if trace_elapsed is None:
            marker_pos = progress * (len(gt) - 1)
            marker_step = int(round(marker_pos))
            marker_time = None
        else:
            marker_time = progress * trace_duration
            marker_pos = None
            marker_step = None

        phone_frame = read_monotonic_frame(phone_cap, phone_idx, phone_state)

        if trace_elapsed is None:
            gt_marker = sample_marker(gt, marker_pos)
            pred_marker = sample_marker(pred, marker_pos)
        else:
            gt_marker, marker_pos, marker_step = sample_marker_from_trace(
                gt,
                trace_elapsed,
                marker_time,
                args.marker_sampling,
            )
            pred_marker, _, _ = sample_marker_from_trace(
                pred,
                trace_elapsed,
                marker_time,
                args.marker_sampling,
            )

        canvas = np.full((args.height, args.width, 3), 255, dtype=np.uint8)
        draw_label(canvas, args.title, (margin, 44), scale=0.82)
        draw_label(
            canvas,
            f"phone {args.phone_speed:g}x trace-aligned   frame {out_i + 1:04d}/{output_frames:04d}",
            (args.width - 470, 44),
            scale=0.52,
        )

        phone_panel = resize_letterbox(phone_frame, args.phone_width, args.phone_height, fill=245)
        canvas[phone_y : phone_y + args.phone_height, phone_x : phone_x + args.phone_width] = phone_panel
        cv2.rectangle(
            canvas,
            (phone_x, phone_y),
            (phone_x + args.phone_width - 1, phone_y + args.phone_height - 1),
            (210, 216, 224),
            2,
        )
        draw_panel_label(canvas, "real operation video", phone_x, phone_y - 18)

        gt_panel = render_deformed_grid_panel(
            gt_marker,
            (args.marker_size, args.marker_size),
            displacement_scale=displacement_scale,
            max_mag=vmax,
        )
        pred_panel = render_deformed_grid_panel(
            pred_marker,
            (args.marker_size, args.marker_size),
            displacement_scale=displacement_scale,
            max_mag=vmax,
        )
        canvas[marker_gt_y : marker_gt_y + args.marker_size, right_x : right_x + args.marker_size] = gt_panel
        canvas[
            marker_pred_y : marker_pred_y + args.marker_size,
            right_x : right_x + args.marker_size,
        ] = pred_panel
        draw_panel_label(canvas, "GT tactile marker", right_x, marker_gt_y - 18)
        draw_panel_label(canvas, "foresight predicted marker", right_x, marker_pred_y - 18)
        err = float(np.linalg.norm(pred_marker - gt_marker, axis=-1).mean())
        if marker_time is None:
            marker_text = f"marker step={marker_pos:.1f}/{len(gt) - 1}"
        else:
            marker_text = (
                f"marker step={marker_step + 1:03d}/{len(gt):03d}   "
                f"trace t={marker_time:.2f}/{trace_duration:.2f}s"
            )
        draw_label(canvas, f"{marker_text}   L2={err:.3f}", (right_x, args.height - 38), scale=0.50)

        writer.write(canvas)
        if out_i == output_frames // 2:
            preview_frame = canvas.copy()

    writer.release()
    phone_cap.release()

    preview_path = None
    if args.preview_frame:
        preview_path = Path(args.preview_frame)
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        if preview_frame is not None:
            cv2.imwrite(str(preview_path), preview_frame)

    summary = {
        "phone_video": str(phone_path),
        "marker_npz": str(marker_path),
        "output": str(output_path),
        "preview_frame": str(preview_path) if preview_path else None,
        "phone_source_frames": int(phone_meta["frames"]),
        "phone_source_fps": float(phone_meta["fps"]),
        "phone_source_duration_s": float(phone_duration),
        "phone_speed": float(args.phone_speed),
        "marker_frames": int(len(gt)),
        "trace_csv": str(Path(args.trace_csv)) if args.trace_csv else None,
        "trace_duration_s": trace_duration,
        "marker_sampling": args.marker_sampling if args.trace_csv else "linear_step",
        "output_frames": int(output_frames),
        "output_fps": float(args.fps),
        "output_duration_s": float(output_duration),
    }
    summary_path = output_path.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    print(json.dumps(compose(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
