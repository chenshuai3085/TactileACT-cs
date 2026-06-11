"""Render full-episode and future-process videos for multi-step tactile foresight.

Each video frame corresponds to a real episode timestep t and compares:
current marker, GT marker at t+target_step, predicted marker at t+target_step,
and the prediction error map.

The script can also render future-process videos.  For a fixed current timestep
t, these videos play the model's predicted future sequence t+1 ... t+H against
the GT future sequence.
"""

import argparse
import glob
import json
import os
import sys

import cv2
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from TFAC_V5.pretrain_latent_foresight_multistep import scan_episode_paths
from TFAC_V5.visualize_foresight_multistep import build_model, load_run


def as_np_stats(norm_stats):
    out = {}
    for key, value in norm_stats.items():
        out[key] = np.asarray(value, dtype=np.float32) if isinstance(value, list) else value
    return out


def choose_val_episode(args):
    episode_paths = []
    for dataset_dir in args["dataset_dirs"]:
        episode_paths.extend(scan_episode_paths(dataset_dir))
    rng = np.random.RandomState(int(args.get("seed", 42)))
    shuffled = rng.permutation(episode_paths).tolist()
    n_train = max(1, int(len(shuffled) * float(args.get("train_ratio", 0.9))))
    val_paths = shuffled[n_train:]
    if not val_paths:
        val_paths = shuffled[-1:]

    scored = []
    tac_side = args.get("tac_side", "left")
    for path in val_paths:
        with h5py.File(path, "r") as root:
            marker = root[f"observations/tac/{tac_side}/marker_offset"][()]
        mag = np.linalg.norm(marker, axis=-1).mean(axis=(1, 2))
        scored.append((float(mag.std()), float(mag.mean()), path))
    scored.sort(reverse=True)
    return scored[0][2], val_paths


def load_episode(path, args):
    proprio_key = args.get("proprio_key", "proprio_joint")
    action_key = args.get("action_key", "actions/joint_abs")
    tac_side = args.get("tac_side", "left")
    with h5py.File(path, "r") as root:
        qpos = root[f"observations/{proprio_key}"][()].astype(np.float32)
        actions = root[action_key][()].astype(np.float32)
        marker = root[f"observations/tac/{tac_side}/marker_offset"][()].astype(np.float32)
    return qpos, actions, marker


def normalize_marker(marker, stats):
    mean = stats["marker_offset_mean"].reshape(1, 1, 1, 2)
    std = stats["marker_offset_std"].reshape(1, 1, 1, 2)
    return (marker - mean) / std


def marker_window(marker_norm, t, window):
    frames = []
    for i in range(window):
        idx = max(0, t - (window - 1 - i))
        frames.append(marker_norm[idx])
    return np.stack(frames, axis=0).astype(np.float32)


def future_marker_windows(marker_norm, t, horizon, window, episode_len):
    frames = []
    for h in range(1, horizon + 1):
        ft = min(t + h, episode_len - 1)
        frames.append(marker_window(marker_norm, ft, window))
    return np.stack(frames, axis=0).astype(np.float32)


def normalize_qpos(qpos, stats):
    return (qpos - stats["qpos_mean"]) / stats["qpos_std"]


def normalize_action(actions, stats):
    return (actions - stats["action_mean"]) / stats["action_std"]


def build_condition_chunk(qpos, actions, t, args, stats):
    chunk_size = int(args.get("chunk_size", 16))
    use_state_traj = bool(args.get("use_state_trajectory", True))
    if use_state_traj:
        raw = qpos[t + 1:min(t + 1 + chunk_size, len(qpos))]
        raw = normalize_qpos(raw, stats)
        dim = qpos.shape[-1]
    else:
        raw = actions[t:min(t + chunk_size, len(actions))]
        raw = normalize_action(raw, stats)
        dim = actions.shape[-1]

    padded = np.zeros((chunk_size, dim), dtype=np.float32)
    if len(raw) > 0:
        padded[:len(raw)] = raw.astype(np.float32)
    return padded


def marker_magnitude(marker):
    return np.linalg.norm(marker, axis=-1)


def run_episode_prediction(model, qpos, actions, marker, args, stats, device, stride):
    horizon = int(args.get("predict_horizon", args.get("foresight_horizon", 16)))
    target_step = horizon
    window = int(args.get("tactile_vae_window", 8))
    marker_norm = normalize_marker(marker, stats)
    max_t = len(marker) - target_step
    timesteps = np.arange(0, max_t, stride, dtype=np.int32)

    pred_target = []
    gt_target = []
    pred_sequence = []
    gt_sequence = []
    current = []
    latent_mae = []
    all_step_marker_l2 = []

    model.eval()
    with torch.no_grad():
        for t in tqdm(timesteps, desc="Episode foresight"):
            curr_win = marker_window(marker_norm, int(t), window)
            future_wins = future_marker_windows(marker_norm, int(t), horizon, window, len(marker))
            action_chunk = build_condition_chunk(qpos, actions, int(t), args, stats)
            qpos_norm = normalize_qpos(qpos[int(t)], stats)

            images = [torch.from_numpy(curr_win).unsqueeze(0).to(device)]
            future_images = [torch.from_numpy(future_wins).unsqueeze(0).to(device)]
            action_t = torch.from_numpy(action_chunk).unsqueeze(0).to(device)
            qpos_t = torch.from_numpy(qpos_norm.astype(np.float32)).unsqueeze(0).to(device)

            t_hat, z_gt, _, _ = model(
                images, action_t, future_images=future_images, qpos=qpos_t)
            marker_hat = model.decode_latent_sequence(t_hat)[0].cpu().numpy()
            marker_hat_raw = (
                marker_hat * stats["marker_offset_std"].reshape(1, 1, 1, 2)
                + stats["marker_offset_mean"].reshape(1, 1, 1, 2)
            )

            gt_seq = []
            for h in range(1, horizon + 1):
                gt_seq.append(marker[min(int(t) + h, len(marker) - 1)])
            gt_seq = np.stack(gt_seq, axis=0)
            step_l2 = np.linalg.norm(marker_hat_raw - gt_seq, axis=-1).mean(axis=(1, 2))

            pred_target.append(marker_hat_raw[target_step - 1])
            gt_target.append(marker[int(t) + target_step])
            pred_sequence.append(marker_hat_raw)
            gt_sequence.append(gt_seq)
            current.append(marker[int(t)])
            latent_mae.append(float(torch.abs(t_hat - z_gt).mean().item()))
            all_step_marker_l2.append(step_l2)

    return {
        "timesteps": timesteps,
        "current": np.stack(current, axis=0),
        "gt_target": np.stack(gt_target, axis=0),
        "pred_target": np.stack(pred_target, axis=0),
        "gt_sequence": np.stack(gt_sequence, axis=0),
        "pred_sequence": np.stack(pred_sequence, axis=0),
        "latent_mae": np.asarray(latent_mae, dtype=np.float32),
        "all_step_marker_l2": np.stack(all_step_marker_l2, axis=0).astype(np.float32),
        "target_step": target_step,
        "horizon": horizon,
    }


def draw_marker(ax, marker, title, vmax, cmap="viridis", draw_quiver=True):
    mag = marker_magnitude(marker)
    im = ax.imshow(mag, cmap=cmap, vmin=0, vmax=vmax, origin="lower")
    if draw_quiver:
        yy, xx = np.mgrid[0:9, 0:9]
        ax.quiver(
            xx, yy, marker[..., 0], marker[..., 1],
            color="white", angles="xy", scale_units="xy", scale=1.0,
            width=0.004, alpha=0.85,
        )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def render_video(results, episode_path, out_video, out_dir, fps):
    os.makedirs(out_dir, exist_ok=True)
    current = results["current"]
    gt = results["gt_target"]
    pred = results["pred_target"]
    err_vec = pred - gt
    err_mag = marker_magnitude(err_vec)
    gt_mag = marker_magnitude(gt).mean(axis=(1, 2))
    pred_mag = marker_magnitude(pred).mean(axis=(1, 2))
    curr_mag = marker_magnitude(current).mean(axis=(1, 2))
    frame_l2 = err_mag.mean(axis=(1, 2))
    timesteps = results["timesteps"]
    target_step = int(results["target_step"])

    vmax = max(float(marker_magnitude(gt).max()), float(marker_magnitude(pred).max()), 1e-6)
    curr_vmax = max(float(marker_magnitude(current).max()), vmax)
    err_vmax = max(float(err_mag.max()), 1e-6)
    n = len(timesteps)
    ep_label = os.path.relpath(episode_path, "/home/chenshuai/data/dataset")

    writer = None
    preview_indices = {0, n // 2, n - 1}
    metrics_xlim = (int(timesteps[0]), int(timesteps[-1]))

    for i in tqdm(range(n), desc="Render video"):
        fig = plt.figure(figsize=(16, 8.8))
        gs = fig.add_gridspec(2, 4, height_ratios=[3.2, 1.3])
        axes = [fig.add_subplot(gs[0, j]) for j in range(4)]
        ax_curve = fig.add_subplot(gs[1, :])

        draw_marker(axes[0], current[i], f"Current marker t={timesteps[i]}", curr_vmax)
        draw_marker(axes[1], gt[i], f"GT marker t+{target_step}", vmax)
        draw_marker(axes[2], pred[i], f"Pred marker t+{target_step}", vmax)
        draw_marker(axes[3], err_vec[i], f"Error |Pred-GT|={frame_l2[i]:.3f}", err_vmax,
                    cmap="magma", draw_quiver=False)

        ax_curve.plot(timesteps, curr_mag, label="current mean |marker|", color="#4C78A8")
        ax_curve.plot(timesteps, gt_mag, label=f"GT t+{target_step}", color="#54A24B")
        ax_curve.plot(timesteps, pred_mag, label=f"Pred t+{target_step}", color="#F58518")
        ax_curve.plot(timesteps, frame_l2, label="mean vector error", color="#E45756")
        ax_curve.axvline(timesteps[i], color="black", linestyle="--", linewidth=1.1)
        ax_curve.set_xlim(metrics_xlim)
        ymax = max(float(gt_mag.max()), float(pred_mag.max()), float(curr_mag.max()), float(frame_l2.max())) * 1.08
        ax_curve.set_ylim(0, ymax)
        ax_curve.set_xlabel("episode timestep")
        ax_curve.grid(True, alpha=0.25)
        ax_curve.legend(loc="upper right", ncol=4, fontsize=9)

        fig.suptitle(
            f"Multi-step Foresight Episode Video | {ep_label} | "
            f"t={timesteps[i]} -> t+{target_step} | error={frame_l2[i]:.3f}",
            fontsize=13,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        rgb = np.asarray(canvas.buffer_rgba())[..., :3]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]
        if writer is None:
            if w % 2:
                w += 1
            if h % 2:
                h += 1
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_video, fourcc, fps, (w, h))
        if (bgr.shape[1], bgr.shape[0]) != (w, h):
            bgr = cv2.resize(bgr, (w, h))
        writer.write(bgr)

        if i in preview_indices:
            preview_path = os.path.join(out_dir, f"preview_frame_{i:04d}_t{timesteps[i]:04d}.png")
            cv2.imwrite(preview_path, bgr)
        plt.close(fig)

    if writer is not None:
        writer.release()

    summary = {
        "episode_path": episode_path,
        "video": out_video,
        "num_video_frames": int(n),
        "fps": int(fps),
        "stride": int(timesteps[1] - timesteps[0]) if len(timesteps) > 1 else 1,
        "target_step": target_step,
        "mean_target_marker_l2": float(frame_l2.mean()),
        "std_target_marker_l2": float(frame_l2.std()),
        "max_target_marker_l2": float(frame_l2.max()),
        "mean_latent_mae_all_horizon": float(results["latent_mae"].mean()),
        "mean_all_step_marker_l2": results["all_step_marker_l2"].mean(axis=0).tolist(),
        "gt_target_mag_mean": float(gt_mag.mean()),
        "pred_target_mag_mean": float(pred_mag.mean()),
    }
    with open(os.path.join(out_dir, "episode_video_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    np.savez_compressed(
        os.path.join(out_dir, "episode_video_predictions.npz"),
        timesteps=timesteps,
        current=current,
        gt_target=gt,
        pred_target=pred,
        gt_sequence=results["gt_sequence"],
        pred_sequence=results["pred_sequence"],
        latent_mae=results["latent_mae"],
        all_step_marker_l2=results["all_step_marker_l2"],
    )
    return summary


def render_future_process_videos(results, episode_path, out_dir, fps, max_videos):
    """For selected current timesteps, render GT/Pred/Error for t+1...t+H."""
    os.makedirs(out_dir, exist_ok=True)
    timesteps = results["timesteps"]
    current = results["current"]
    gt_seq = results["gt_sequence"]
    pred_seq = results["pred_sequence"]
    step_l2 = results["all_step_marker_l2"]
    horizon = int(results["horizon"])
    ep_label = os.path.relpath(episode_path, "/home/chenshuai/data/dataset")

    gt_mag_all = marker_magnitude(gt_seq)
    pred_mag_all = marker_magnitude(pred_seq)
    err_mag_all = marker_magnitude(pred_seq - gt_seq)
    total_error = step_l2.mean(axis=1)

    candidate_indices = [
        0,
        int(np.argmax(total_error)),
        int(np.argmax(marker_magnitude(current).mean(axis=(1, 2)))),
        len(timesteps) // 2,
        len(timesteps) - 1,
    ]
    selected = []
    for idx in candidate_indices:
        idx = int(np.clip(idx, 0, len(timesteps) - 1))
        if idx not in selected:
            selected.append(idx)
        if len(selected) >= max_videos:
            break

    outputs = []
    for idx in selected:
        t = int(timesteps[idx])
        video_path = os.path.join(out_dir, f"future_process_t{t:04d}_h{horizon}.mp4")
        writer = None
        vmax = max(float(gt_mag_all[idx].max()), float(pred_mag_all[idx].max()), 1e-6)
        curr_vmax = max(float(marker_magnitude(current[idx]).max()), vmax)
        err_vmax = max(float(err_mag_all[idx].max()), 1e-6)

        for h in tqdm(range(horizon), desc=f"Future process t={t}"):
            fig = plt.figure(figsize=(16, 8.8))
            gs = fig.add_gridspec(2, 4, height_ratios=[3.2, 1.3])
            axes = [fig.add_subplot(gs[0, j]) for j in range(4)]
            ax_curve = fig.add_subplot(gs[1, :])

            draw_marker(axes[0], current[idx], f"Current marker t={t}", curr_vmax)
            draw_marker(axes[1], gt_seq[idx, h], f"GT marker t+{h + 1}", vmax)
            draw_marker(axes[2], pred_seq[idx, h], f"Pred marker t+{h + 1}", vmax)
            draw_marker(
                axes[3],
                pred_seq[idx, h] - gt_seq[idx, h],
                f"Error |Pred-GT|={step_l2[idx, h]:.3f}",
                err_vmax,
                cmap="magma",
                draw_quiver=False,
            )

            steps = np.arange(1, horizon + 1)
            ax_curve.plot(
                steps, gt_mag_all[idx].mean(axis=(1, 2)),
                marker="o", label="GT mean |marker|", color="#54A24B")
            ax_curve.plot(
                steps, pred_mag_all[idx].mean(axis=(1, 2)),
                marker="o", label="Pred mean |marker|", color="#F58518")
            ax_curve.plot(
                steps, step_l2[idx],
                marker="o", label="mean vector error", color="#E45756")
            ax_curve.axvline(h + 1, color="black", linestyle="--", linewidth=1.1)
            ax_curve.set_xlim(1, horizon)
            ymax = max(
                float(gt_mag_all[idx].mean(axis=(1, 2)).max()),
                float(pred_mag_all[idx].mean(axis=(1, 2)).max()),
                float(step_l2[idx].max()),
            ) * 1.08
            ax_curve.set_ylim(0, ymax)
            ax_curve.set_xlabel("future step")
            ax_curve.grid(True, alpha=0.25)
            ax_curve.legend(loc="upper right", ncol=3, fontsize=9)

            fig.suptitle(
                f"Multi-step Future Process | {ep_label} | "
                f"current t={t}, showing t+{h + 1}/{horizon}",
                fontsize=13,
            )
            fig.tight_layout(rect=[0, 0, 1, 0.95])

            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            rgb = np.asarray(canvas.buffer_rgba())[..., :3]
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            height, width = bgr.shape[:2]
            if writer is None:
                if width % 2:
                    width += 1
                if height % 2:
                    height += 1
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            if (bgr.shape[1], bgr.shape[0]) != (width, height):
                bgr = cv2.resize(bgr, (width, height))
            writer.write(bgr)

            if h in {0, horizon // 2, horizon - 1}:
                preview_path = os.path.join(
                    out_dir, f"future_process_t{t:04d}_step{h + 1:02d}.png")
                cv2.imwrite(preview_path, bgr)
            plt.close(fig)

        if writer is not None:
            writer.release()
        outputs.append({
            "timestep": t,
            "video": video_path,
            "mean_future_marker_l2": float(step_l2[idx].mean()),
            "max_future_marker_l2": float(step_l2[idx].max()),
        })

    with open(os.path.join(out_dir, "future_process_metrics.json"), "w") as f:
        json.dump(outputs, f, indent=2)
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_dir",
        default="/home/chenshuai/Project/output/foresight_ckpt/"
                "latent_foresight_board_260609_260610_multistep16_marker_only_e100_bs16_preload",
    )
    parser.add_argument("--episode_path", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--future_fps", type=int, default=3)
    parser.add_argument("--future_process_videos", type=int, default=3)
    parser.add_argument("--skip_episode_video", action="store_true")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args_cli = parser.parse_args()

    args, _, ckpt_path = load_run(args_cli.ckpt_dir)
    stats = as_np_stats(args["norm_stats"])
    device = torch.device(
        args_cli.device if torch.cuda.is_available() and args_cli.device == "cuda" else "cpu")
    model = build_model(args, ckpt_path, device)

    episode_path = args_cli.episode_path
    val_episode_count = None
    if episode_path is None:
        episode_path, val_paths = choose_val_episode(args)
        val_episode_count = len(val_paths)
    if not os.path.exists(episode_path):
        raise FileNotFoundError(episode_path)

    qpos, actions, marker = load_episode(episode_path, args)
    ep_name = os.path.splitext(os.path.basename(episode_path))[0]
    ep_parent = os.path.basename(os.path.dirname(episode_path))
    out_dir = args_cli.out_dir or os.path.join(
        args_cli.ckpt_dir, "episode_video", f"{ep_parent}_{ep_name}_stride{args_cli.stride}")
    os.makedirs(out_dir, exist_ok=True)
    out_video = os.path.join(out_dir, f"{ep_parent}_{ep_name}_tplus16.mp4")

    results = run_episode_prediction(
        model, qpos, actions, marker, args, stats, device, max(1, args_cli.stride))
    summary = None
    if not args_cli.skip_episode_video:
        summary = render_video(results, episode_path, out_video, out_dir, args_cli.fps)
    if val_episode_count is not None:
        if summary is not None:
            summary["selected_from_val_episode_count"] = val_episode_count
            with open(os.path.join(out_dir, "episode_video_metrics.json"), "w") as f:
                json.dump(summary, f, indent=2)

    future_outputs = []
    if args_cli.future_process_videos > 0:
        future_dir = os.path.join(out_dir, "future_process")
        future_outputs = render_future_process_videos(
            results, episode_path, future_dir, args_cli.future_fps,
            args_cli.future_process_videos)

    print(json.dumps({
        "episode_video": summary,
        "future_process_videos": future_outputs,
    }, indent=2))
    if summary is not None:
        print(f"Saved video to: {out_video}")


if __name__ == "__main__":
    main()
