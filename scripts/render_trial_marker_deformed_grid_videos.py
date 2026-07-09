#!/usr/bin/env python3
"""Render complete-trial marker predictions as deformed-grid videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import cv2
import numpy as np
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-render saved foresight marker_prediction npz files using the "
            "deformed-grid marker style."
        )
    )
    parser.add_argument(
        "--input-root",
        default="outputs/foresight_retrain_20260708/trial_marker_videos",
        help="Root containing */*/marker_prediction_tplusXX.npz files.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/foresight_retrain_20260708/trial_marker_videos_deformed_grid",
        help="Output root for deformed-grid mp4 files.",
    )
    parser.add_argument("--pred-step", type=int, default=1, help="Prediction step suffix.")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--panel-size", type=int, default=620)
    parser.add_argument("--header-height", type=int, default=70)
    parser.add_argument("--footer-height", type=int, default=42)
    parser.add_argument("--side-margin", type=int, default=30)
    parser.add_argument("--gap", type=int, default=24)
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Render only the first N npz files. 0 means all files.",
    )
    return parser.parse_args()


def draw_label(img: np.ndarray, text: str, org: tuple[int, int], scale: float = 0.58) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (30, 38, 50), 2, cv2.LINE_AA)


def draw_poly_grid(
    panel: np.ndarray,
    points: np.ndarray,
    color: tuple[int, int, int],
    thickness: int,
    line_type: int = cv2.LINE_AA,
) -> None:
    rows, cols = points.shape[:2]
    for iy in range(rows):
        for ix in range(cols - 1):
            p0 = tuple(np.rint(points[iy, ix]).astype(int))
            p1 = tuple(np.rint(points[iy, ix + 1]).astype(int))
            cv2.line(panel, p0, p1, color, thickness, line_type)
    for iy in range(rows - 1):
        for ix in range(cols):
            p0 = tuple(np.rint(points[iy, ix]).astype(int))
            p1 = tuple(np.rint(points[iy + 1, ix]).astype(int))
            cv2.line(panel, p0, p1, color, thickness, line_type)


def render_deformed_grid_panel(
    marker: np.ndarray,
    panel_size: tuple[int, int],
    displacement_scale: float,
    max_mag: float,
) -> np.ndarray:
    width, height = panel_size
    panel = np.full((height, width, 3), 252, dtype=np.uint8)

    margin = int(min(width, height) * 0.13)
    xs = np.linspace(margin, width - margin, marker.shape[1])
    ys = np.linspace(margin, height - margin, marker.shape[0])
    base_x, base_y = np.meshgrid(xs, ys)
    base = np.stack([base_x, base_y], axis=-1).astype(np.float32)
    displaced = base + marker.astype(np.float32) * displacement_scale

    mag = np.linalg.norm(marker, axis=-1)
    heat = np.clip(mag / max(max_mag, 1e-6), 0.0, 1.0)
    heat_img = cv2.resize((heat * 255).astype(np.uint8), (width, height), interpolation=cv2.INTER_CUBIC)
    heat_color = cv2.applyColorMap(heat_img, cv2.COLORMAP_TURBO)
    panel = cv2.addWeighted(panel, 0.86, heat_color, 0.14, 0)

    draw_poly_grid(panel, base, (214, 220, 228), 1)
    for iy in range(marker.shape[0]):
        for ix in range(marker.shape[1]):
            start = tuple(np.rint(base[iy, ix]).astype(int))
            end = tuple(np.rint(displaced[iy, ix]).astype(int))
            cv2.arrowedLine(panel, start, end, (110, 116, 128), 1, cv2.LINE_AA, tipLength=0.22)

    overlay = panel.copy()
    draw_poly_grid(overlay, displaced, (18, 125, 160), 3)
    panel = cv2.addWeighted(panel, 0.72, overlay, 0.28, 0)
    draw_poly_grid(panel, displaced, (10, 104, 140), 2)

    for iy in range(marker.shape[0]):
        for ix in range(marker.shape[1]):
            base_pt = tuple(np.rint(base[iy, ix]).astype(int))
            cur_pt = tuple(np.rint(displaced[iy, ix]).astype(int))
            color_idx = int(np.clip(heat[iy, ix] * 255, 0, 255))
            color = cv2.applyColorMap(np.array([[color_idx]], dtype=np.uint8), cv2.COLORMAP_TURBO)[0, 0]
            color_tuple = tuple(int(v) for v in color)
            cv2.circle(panel, base_pt, 2, (160, 166, 176), -1, cv2.LINE_AA)
            cv2.circle(panel, cur_pt, 6, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(panel, cur_pt, 4, color_tuple, -1, cv2.LINE_AA)

    cv2.rectangle(panel, (0, 0), (width - 1, height - 1), (210, 216, 224), 2)
    return panel


def ensure_even(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    pad_h = h % 2
    pad_w = w % 2
    if pad_h == 0 and pad_w == 0:
        return frame
    return cv2.copyMakeBorder(frame, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(255, 255, 255))


def discover_npz(input_root: Path, pred_step: int) -> list[Path]:
    pattern = f"*/{''}*/marker_prediction_tplus{pred_step:02d}.npz"
    return sorted(input_root.glob(pattern))


def marker_vmax(*arrays: np.ndarray) -> float:
    mags = []
    for arr in arrays:
        if arr.size:
            mags.append(np.linalg.norm(arr.astype(np.float32), axis=-1).reshape(-1))
    if not mags:
        return 1.0
    merged = np.concatenate(mags)
    vmax = float(np.percentile(merged, 98.0))
    if vmax < 1e-6:
        vmax = float(max(np.max(merged), 1.0))
    return vmax


def metric_text(marker: np.ndarray, err: float | None = None) -> str:
    mean_mag = float(np.linalg.norm(marker, axis=-1).mean())
    if err is None:
        return f"mean |d|={mean_mag:.3f}"
    return f"mean |d|={mean_mag:.3f}   marker L2={err:.3f}"


def make_single_frame(
    marker: np.ndarray,
    label: str,
    timestep: int,
    frame_idx: int,
    total: int,
    panel_size: int,
    header_h: int,
    footer_h: int,
    margin: int,
    displacement_scale: float,
    vmax: float,
    err: float | None = None,
) -> np.ndarray:
    width = panel_size + margin * 2
    height = header_h + panel_size + footer_h
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    draw_label(canvas, label, (margin, 42), scale=0.72)
    draw_label(canvas, f"frame {frame_idx + 1:04d}/{total:04d}   t={timestep}", (width - 300, 42), scale=0.48)
    panel = render_deformed_grid_panel(
        marker,
        (panel_size, panel_size),
        displacement_scale=displacement_scale,
        max_mag=vmax,
    )
    canvas[header_h : header_h + panel_size, margin : margin + panel_size] = panel
    draw_label(canvas, metric_text(marker, err), (margin, height - 16), scale=0.48)
    return ensure_even(canvas)


def make_pair_frame(
    gt: np.ndarray,
    pred: np.ndarray,
    timestep: int,
    frame_idx: int,
    total: int,
    panel_size: int,
    header_h: int,
    footer_h: int,
    margin: int,
    gap: int,
    displacement_scale: float,
    vmax: float,
    err: float,
) -> np.ndarray:
    width = panel_size * 2 + margin * 2 + gap
    height = header_h + panel_size + footer_h
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    draw_label(canvas, "GT marker", (margin, 42), scale=0.72)
    right_x = margin + panel_size + gap
    draw_label(canvas, "Foresight pred marker", (right_x, 42), scale=0.72)
    draw_label(canvas, f"frame {frame_idx + 1:04d}/{total:04d}   t={timestep}", (width - 300, 42), scale=0.48)

    gt_panel = render_deformed_grid_panel(
        gt,
        (panel_size, panel_size),
        displacement_scale=displacement_scale,
        max_mag=vmax,
    )
    pred_panel = render_deformed_grid_panel(
        pred,
        (panel_size, panel_size),
        displacement_scale=displacement_scale,
        max_mag=vmax,
    )
    canvas[header_h : header_h + panel_size, margin : margin + panel_size] = gt_panel
    canvas[header_h : header_h + panel_size, right_x : right_x + panel_size] = pred_panel
    draw_label(canvas, metric_text(gt), (margin, height - 16), scale=0.48)
    draw_label(canvas, metric_text(pred, err), (right_x, height - 16), scale=0.48)
    return ensure_even(canvas)


def write_video(path: Path, frames: Iterable[np.ndarray], fps: float) -> Dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    iterator = iter(frames)
    first = next(iterator)
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")
    writer.write(first)
    count = 1
    for frame in iterator:
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        writer.write(frame)
        count += 1
    writer.release()
    return {"video": str(path), "frames": int(count), "width": int(w), "height": int(h), "fps": float(fps)}


def render_one(npz_path: Path, input_root: Path, output_root: Path, args: argparse.Namespace) -> dict:
    data = np.load(npz_path)
    gt = data["gt"].astype(np.float32)
    pred = data["pred"].astype(np.float32)
    target_ts = data["target_ts"].astype(np.int64) if "target_ts" in data.files else np.arange(len(gt), dtype=np.int64)
    err_l2 = data["err_l2"].astype(np.float32) if "err_l2" in data.files else np.linalg.norm(pred - gt, axis=-1).mean(axis=(1, 2))
    if gt.shape != pred.shape:
        raise ValueError(f"GT/pred shape mismatch in {npz_path}: {gt.shape} vs {pred.shape}")
    if gt.ndim != 4 or gt.shape[-3:] != (9, 9, 2):
        raise ValueError(f"Expected marker shape T x 9 x 9 x 2 in {npz_path}, got {gt.shape}")

    rel_dir = npz_path.parent.relative_to(input_root)
    out_dir = output_root / rel_dir
    total = len(gt)
    vmax = marker_vmax(gt, pred)
    displacement_scale = (args.panel_size * 0.085) / max(vmax, 1e-6)
    suffix = f"tplus{args.pred_step:02d}"

    def gt_frames():
        for i in tqdm(range(total), desc=f"{rel_dir} gt", leave=False):
            yield make_single_frame(
                gt[i],
                "GT marker deformed grid",
                int(target_ts[i]),
                i,
                total,
                args.panel_size,
                args.header_height,
                args.footer_height,
                args.side_margin,
                displacement_scale,
                vmax,
            )

    def pred_frames():
        for i in tqdm(range(total), desc=f"{rel_dir} pred", leave=False):
            yield make_single_frame(
                pred[i],
                "Foresight pred marker deformed grid",
                int(target_ts[i]),
                i,
                total,
                args.panel_size,
                args.header_height,
                args.footer_height,
                args.side_margin,
                displacement_scale,
                vmax,
                float(err_l2[i]),
            )

    def pair_frames():
        for i in tqdm(range(total), desc=f"{rel_dir} pair", leave=False):
            yield make_pair_frame(
                gt[i],
                pred[i],
                int(target_ts[i]),
                i,
                total,
                args.panel_size,
                args.header_height,
                args.footer_height,
                args.side_margin,
                args.gap,
                displacement_scale,
                vmax,
                float(err_l2[i]),
            )

    videos = {
        "gt": write_video(out_dir / f"gt_marker_deformed_grid_{suffix}.mp4", gt_frames(), args.fps),
        "pred": write_video(out_dir / f"foresight_pred_marker_deformed_grid_{suffix}.mp4", pred_frames(), args.fps),
        "gt_vs_pred": write_video(out_dir / f"gt_vs_pred_marker_deformed_grid_{suffix}.mp4", pair_frames(), args.fps),
    }
    summary = {
        "source_npz": str(npz_path),
        "task": rel_dir.parts[0] if len(rel_dir.parts) >= 1 else "",
        "trial": rel_dir.parts[1] if len(rel_dir.parts) >= 2 else str(rel_dir),
        "frames": int(total),
        "target_t_start": int(target_ts[0]) if total else None,
        "target_t_end": int(target_ts[-1]) if total else None,
        "marker_vmax_p98": float(vmax),
        "mean_marker_l2": float(np.mean(err_l2)) if total else None,
        "max_marker_l2": float(np.max(err_l2)) if total else None,
        "videos": videos,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "deformed_grid_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    npz_files = discover_npz(input_root, args.pred_step)
    if args.max_files > 0:
        npz_files = npz_files[: args.max_files]
    if not npz_files:
        raise FileNotFoundError(
            f"No marker_prediction_tplus{args.pred_step:02d}.npz files under {input_root}"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for npz_path in tqdm(npz_files, desc="trials"):
        summaries.append(render_one(npz_path, input_root, output_root, args))

    summary_path = output_root / f"all_trial_deformed_grid_tplus{args.pred_step:02d}_summary.json"
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "trials": len(summaries)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
