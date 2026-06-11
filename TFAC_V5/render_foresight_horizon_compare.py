"""Render cached multi-horizon foresight videos from episode predictions.

This script does not run the model. It reads episode_video_predictions.npz,
then renders:
1. one full-episode GT/Pred video for each requested future step;
2. one combined full-episode video with all requested future steps.
"""

import argparse
import glob
import json
import os

import cv2
import numpy as np
from tqdm import tqdm


def marker_magnitude(marker):
    return np.linalg.norm(marker, axis=-1)


def make_even(frame):
    h, w = frame.shape[:2]
    out_w = w + (w % 2)
    out_h = h + (h % 2)
    if (out_w, out_h) == (w, h):
        return frame
    return cv2.copyMakeBorder(frame, 0, out_h - h, 0, out_w - w, cv2.BORDER_CONSTANT)


def put_text(img, text, org, scale=0.55, color=(245, 245, 245), thickness=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thickness, cv2.LINE_AA)


def draw_marker_panel(marker, title, vmax, panel_size=150, label_h=24, draw_quiver=True):
    mag = marker_magnitude(marker)
    norm = np.clip(mag / max(vmax, 1e-6), 0, 1)
    gray = (norm * 255).astype(np.uint8)
    heat = cv2.applyColorMap(cv2.resize(gray, (panel_size, panel_size),
                                        interpolation=cv2.INTER_NEAREST),
                             cv2.COLORMAP_VIRIDIS)
    if draw_quiver:
        cell = panel_size / marker.shape[0]
        max_vec = max(float(marker_magnitude(marker).max()), 1e-6)
        arrow_scale = cell * 0.40 / max_vec
        for y in range(marker.shape[0]):
            for x in range(marker.shape[1]):
                dx, dy = marker[y, x]
                cx = int((x + 0.5) * cell)
                cy = int((marker.shape[0] - y - 0.5) * cell)
                ex = int(cx + dx * arrow_scale)
                ey = int(cy - dy * arrow_scale)
                cv2.arrowedLine(
                    heat, (cx, cy), (ex, ey), (255, 255, 255),
                    1, cv2.LINE_AA, tipLength=0.25)

    panel = np.full((panel_size + label_h, panel_size, 3), 22, dtype=np.uint8)
    panel[label_h:, :] = heat
    put_text(panel, title, (6, 17), scale=0.47)
    return panel


def make_horizon_frame(gt, pred, step, timestep, vmax, fps_label, panel_size):
    gt_panel = draw_marker_panel(gt, f"GT t+{step}", vmax, panel_size=panel_size)
    pred_panel = draw_marker_panel(pred, f"Pred t+{step}", vmax, panel_size=panel_size)
    gap = np.full((gt_panel.shape[0], 12, 3), 18, dtype=np.uint8)
    body = np.concatenate([gt_panel, gap, pred_panel], axis=1)
    header = np.full((34, body.shape[1], 3), 18, dtype=np.uint8)
    put_text(header, f"episode t={timestep} | future step t+{step} | {fps_label}",
             (8, 22), scale=0.56)
    return np.concatenate([header, body], axis=0)


def make_combined_frame(gt_seq, pred_seq, steps, timestep, vmax, panel_size):
    blocks = []
    for step in steps:
        h = step - 1
        gt_panel = draw_marker_panel(gt_seq[h], f"GT +{step}", vmax, panel_size=panel_size)
        pred_panel = draw_marker_panel(pred_seq[h], f"Pred +{step}", vmax, panel_size=panel_size)
        gap = np.full((gt_panel.shape[0], 8, 3), 18, dtype=np.uint8)
        block_body = np.concatenate([gt_panel, gap, pred_panel], axis=1)
        header = np.full((26, block_body.shape[1], 3), 18, dtype=np.uint8)
        put_text(header, f"t+{step}", (8, 18), scale=0.50)
        blocks.append(np.concatenate([header, block_body], axis=0))

    row_gap = np.full((16, blocks[0].shape[1] * 4 + 12 * 3, 3), 18, dtype=np.uint8)
    rows = []
    for row in range(2):
        row_blocks = blocks[row * 4:(row + 1) * 4]
        col_gap = np.full((row_blocks[0].shape[0], 12, 3), 18, dtype=np.uint8)
        rows.append(np.concatenate(
            [row_blocks[0], col_gap, row_blocks[1], col_gap, row_blocks[2], col_gap, row_blocks[3]],
            axis=1))
    body = np.concatenate([rows[0], row_gap, rows[1]], axis=0)
    header = np.full((36, body.shape[1], 3), 18, dtype=np.uint8)
    put_text(header, f"all horizons GT/Pred | episode t={timestep}", (8, 24), scale=0.62)
    return np.concatenate([header, body], axis=0)


def write_video(path, frames, fps):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    first = make_even(next(frames))
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    writer.write(first)
    count = 1
    for frame in frames:
        frame = make_even(frame)
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h))
        writer.write(frame)
        count += 1
    writer.release()
    return count, w, h


def load_predictions(prediction_npz):
    data = np.load(prediction_npz)
    return {
        "timesteps": data["timesteps"],
        "gt_sequence": data["gt_sequence"],
        "pred_sequence": data["pred_sequence"],
    }


def render_one_prediction(prediction_npz, out_dir, steps, fps, panel_size, combined_panel_size):
    data = load_predictions(prediction_npz)
    timesteps = data["timesteps"]
    gt_seq = data["gt_sequence"]
    pred_seq = data["pred_sequence"]

    max_step = max(steps)
    if gt_seq.shape[1] < max_step or pred_seq.shape[1] < max_step:
        raise ValueError(f"Need horizon >= {max_step}, got gt={gt_seq.shape}, pred={pred_seq.shape}")

    selected_gt = gt_seq[:, [s - 1 for s in steps]]
    selected_pred = pred_seq[:, [s - 1 for s in steps]]
    vmax = max(float(marker_magnitude(selected_gt).max()),
               float(marker_magnitude(selected_pred).max()), 1e-6)

    metrics = {
        "prediction_npz": prediction_npz,
        "out_dir": out_dir,
        "steps": steps,
        "fps": fps,
        "num_frames": int(len(timesteps)),
        "vmax": vmax,
        "single_horizon_videos": [],
    }

    single_dir = os.path.join(out_dir, "single_horizon_gt_pred")
    for step in steps:
        out_video = os.path.join(single_dir, f"horizon_tplus{step:02d}_gt_pred.mp4")

        def frame_iter(step=step):
            h = step - 1
            desc = f"{os.path.basename(os.path.dirname(prediction_npz))} t+{step}"
            for i in tqdm(range(len(timesteps)), desc=desc):
                yield make_horizon_frame(
                    gt_seq[i, h], pred_seq[i, h], step, int(timesteps[i]),
                    vmax, f"{fps} fps", panel_size)

        count, width, height = write_video(out_video, frame_iter(), fps)
        err = marker_magnitude(pred_seq[:, step - 1] - gt_seq[:, step - 1]).mean(axis=(1, 2))
        metrics["single_horizon_videos"].append({
            "step": step,
            "video": out_video,
            "frames": count,
            "width": width,
            "height": height,
            "mean_marker_l2": float(err.mean()),
            "max_marker_l2": float(err.max()),
        })

    combined_video = os.path.join(out_dir, "combined_horizons_tplus02_to_tplus16_gt_pred_grid.mp4")

    def combined_iter():
        desc = f"{os.path.basename(os.path.dirname(prediction_npz))} combined"
        for i in tqdm(range(len(timesteps)), desc=desc):
            yield make_combined_frame(
                gt_seq[i], pred_seq[i], steps, int(timesteps[i]),
                vmax, combined_panel_size)

    count, width, height = write_video(combined_video, combined_iter(), fps)
    metrics["combined_video"] = {
        "video": combined_video,
        "frames": count,
        "width": width,
        "height": height,
    }
    metrics["mean_step_marker_l2"] = [
        float(marker_magnitude(pred_seq[:, s - 1] - gt_seq[:, s - 1]).mean())
        for s in steps
    ]

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "horizon_compare_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def find_prediction_npzs(input_dir):
    pattern = os.path.join(input_dir, "*", "episode_video_predictions.npz")
    return sorted(glob.glob(pattern))


def parse_steps(value):
    return [int(x) for x in value.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default=None)
    parser.add_argument("--prediction_npz", default=None)
    parser.add_argument("--out_subdir", default="horizon_compare_gt_pred")
    parser.add_argument("--steps", default="2,4,6,8,10,12,14,16")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--panel_size", type=int, default=220)
    parser.add_argument("--combined_panel_size", type=int, default=110)
    args = parser.parse_args()

    steps = parse_steps(args.steps)
    if args.prediction_npz:
        prediction_npzs = [args.prediction_npz]
    elif args.input_dir:
        prediction_npzs = find_prediction_npzs(args.input_dir)
    else:
        raise ValueError("Provide --input_dir or --prediction_npz")
    if not prediction_npzs:
        raise FileNotFoundError("No episode_video_predictions.npz found")

    all_metrics = []
    for prediction_npz in prediction_npzs:
        parent = os.path.dirname(prediction_npz)
        out_dir = os.path.join(parent, args.out_subdir)
        all_metrics.append(render_one_prediction(
            prediction_npz, out_dir, steps, args.fps,
            args.panel_size, args.combined_panel_size))

    print(json.dumps(all_metrics, indent=2))


if __name__ == "__main__":
    main()
