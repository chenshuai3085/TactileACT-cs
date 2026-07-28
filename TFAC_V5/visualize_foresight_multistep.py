"""Visualize multi-step tactile foresight training and reconstructions."""

import argparse
import json
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from TFAC_V5.dataset import ForesightEpisodicDataset
from TFAC_V5.pretrain_latent_foresight_multistep import (
    MultiStepLatentForesightModel,
    scan_episode_paths,
    to_device,
)


def _as_numpy_stats(norm_stats):
    out = {}
    for key, value in norm_stats.items():
        if isinstance(value, list):
            out[key] = np.asarray(value, dtype=np.float32)
        else:
            out[key] = value
    return out


def load_run(ckpt_dir):
    args_path = os.path.join(ckpt_dir, "args.json")
    history_path = os.path.join(ckpt_dir, "pretrain_history.pkl")
    ckpt_path = os.path.join(ckpt_dir, "foresight_best.ckpt")
    if not os.path.exists(args_path):
        raise FileNotFoundError(args_path)
    if not os.path.exists(history_path):
        raise FileNotFoundError(history_path)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(ckpt_path)

    with open(args_path) as f:
        args = json.load(f)
    with open(history_path, "rb") as f:
        history = pickle.load(f)
    return args, history, ckpt_path


def plot_loss(history, out_path):
    names = ["total", "latent", "latent_seq", "latent_final", "marker", "delta"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()
    best_epoch = int(np.argmin(history["val_total"]))

    for ax, name in zip(axes, names):
        train = history.get(f"train_{name}")
        val = history.get(f"val_{name}")
        if train is not None:
            ax.plot(train, label="train", linewidth=1.8)
        if val is not None:
            ax.plot(val, label="val", linewidth=1.8)
        ax.axvline(best_epoch, color="black", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_title(name)
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.25)
        ax.legend()

    fig.suptitle(
        f"Multi-step Foresight Loss Curves | best epoch={best_epoch}, "
        f"best val_total={history['val_total'][best_epoch]:.6f}",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_val_dataset(args):
    dataset_dirs = args["dataset_dirs"]
    episode_paths = []
    for dataset_dir in dataset_dirs:
        episode_paths.extend(scan_episode_paths(dataset_dir))
    if not episode_paths:
        raise RuntimeError("No episode files found")

    seed = int(args.get("seed", 42))
    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(episode_paths).tolist()
    train_ratio = float(args.get("train_ratio", 0.9))
    n_train = max(1, int(len(shuffled) * train_ratio))
    val_paths = shuffled[n_train:]
    if not val_paths:
        val_paths = shuffled[-1:]

    norm_stats = _as_numpy_stats(args["norm_stats"])
    dataset_root = os.path.dirname(dataset_dirs[0])
    dataset = ForesightEpisodicDataset(
        val_paths,
        dataset_root,
        args["camera_names"],
        norm_stats,
        chunk_size=int(args.get("chunk_size", 16)),
        foresight_horizon=int(args.get("foresight_horizon", 16)),
        proprio_key=args.get("proprio_key", "proprio_joint"),
        action_key=args.get("action_key", "actions/joint_abs"),
        tac_side=args.get("tac_side", "left"),
        tac_img_key=args.get("tac_img_key", "img"),
        tactile_mode=args.get("tactile_mode", "marker"),
        history_len=int(args.get("history_len", 1)),
        tactile_vae_window=int(args.get("tactile_vae_window", 8)),
        preload=True,
        use_state_trajectory=bool(args.get("use_state_trajectory", False)),
    )
    return dataset, val_paths, norm_stats


def build_model(args, ckpt_path, device):
    camera_names = args["camera_names"]
    cam_backbone_mapping = {name: 0 for name in camera_names}
    model = MultiStepLatentForesightModel(
        camera_names=camera_names,
        cam_backbone_mapping=cam_backbone_mapping,
        hidden_dim=int(args.get("hidden_dim", 512)),
        state_dim=int(args.get("state_dim", 7)),
        foresight_layers=int(args.get("foresight_layers", 3)),
        foresight_nheads=int(args.get("foresight_nheads", 8)),
        foresight_dim_feedforward=int(args.get("foresight_dim_feedforward", 2048)),
        dropout=float(args.get("dropout", 0.1)),
        tactile_mode=args.get("tactile_mode", "marker"),
        max_history=int(args.get("max_history", 8)),
        predict_horizon=int(args.get("predict_horizon", args.get("foresight_horizon", 16))),
        tactile_vae_ckpt=args.get("tactile_vae_ckpt"),
        tactile_vae_latent_dim=int(args.get("tactile_vae_latent_dim", 16)),
        tactile_vae_window=int(args.get("tactile_vae_window", 8)),
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def denorm_marker(marker, norm_stats):
    mean = norm_stats.get("marker_offset_mean")
    std = norm_stats.get("marker_offset_std")
    if mean is None or std is None:
        return marker
    mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 1, 1, 2)
    std = np.asarray(std, dtype=np.float32).reshape(1, 1, 1, 1, 2)
    return marker * std + mean


def marker_mag(marker):
    return np.linalg.norm(marker, axis=-1)


def collect_predictions(model, loader, device, max_batches):
    pred_markers, gt_markers, pred_latents, gt_latents = [], [], [], []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break
            images, qpos, actions, future_images = to_device(batch, device)
            t_hat, z_gt, marker_gt, _ = model(
                images, actions, future_images=future_images, qpos=qpos)
            marker_hat = model.decode_latent_sequence(t_hat)
            pred_markers.append(marker_hat.detach().cpu().numpy())
            gt_markers.append(marker_gt.detach().cpu().numpy())
            pred_latents.append(t_hat.detach().cpu().numpy())
            gt_latents.append(z_gt.detach().cpu().numpy())

    return {
        "pred_marker_norm": np.concatenate(pred_markers, axis=0),
        "gt_marker_norm": np.concatenate(gt_markers, axis=0),
        "pred_latent": np.concatenate(pred_latents, axis=0),
        "gt_latent": np.concatenate(gt_latents, axis=0),
    }


def compute_metrics(arrays, norm_stats):
    pred_marker = arrays["pred_marker_norm"]
    gt_marker = arrays["gt_marker_norm"]
    pred_marker_raw = denorm_marker(pred_marker, norm_stats)
    gt_marker_raw = denorm_marker(gt_marker, norm_stats)
    pred_latent = arrays["pred_latent"]
    gt_latent = arrays["gt_latent"]

    marker_abs = np.abs(pred_marker - gt_marker)
    marker_vec_l2 = np.linalg.norm(pred_marker - gt_marker, axis=-1)
    marker_raw_vec_l2 = np.linalg.norm(pred_marker_raw - gt_marker_raw, axis=-1)
    latent_abs = np.abs(pred_latent - gt_latent)

    metrics = {
        "num_windows": int(pred_marker.shape[0]),
        "horizon": int(pred_marker.shape[1]),
        "marker_norm_mae": float(marker_abs.mean()),
        "marker_norm_vec_l2": float(marker_vec_l2.mean()),
        "marker_raw_vec_l2": float(marker_raw_vec_l2.mean()),
        "latent_mae": float(latent_abs.mean()),
        "marker_norm_mae_by_step": marker_abs.mean(axis=(0, 2, 3, 4)).tolist(),
        "marker_norm_vec_l2_by_step": marker_vec_l2.mean(axis=(0, 2, 3)).tolist(),
        "marker_raw_vec_l2_by_step": marker_raw_vec_l2.mean(axis=(0, 2, 3)).tolist(),
        "latent_mae_by_step": latent_abs.mean(axis=(0, 2)).tolist(),
        "gt_raw_mag_by_step": marker_mag(gt_marker_raw).mean(axis=(0, 2, 3)).tolist(),
        "pred_raw_mag_by_step": marker_mag(pred_marker_raw).mean(axis=(0, 2, 3)).tolist(),
    }
    return metrics, pred_marker_raw, gt_marker_raw


def plot_sequence_metrics(metrics, out_path):
    steps = np.arange(1, metrics["horizon"] + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(steps, metrics["latent_mae_by_step"], marker="o")
    axes[0].set_title("Latent MAE by Future Step")
    axes[0].set_xlabel("future step")
    axes[0].set_ylabel("MAE")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(steps, metrics["marker_norm_vec_l2_by_step"], marker="o", label="normalized")
    axes[1].plot(steps, metrics["marker_raw_vec_l2_by_step"], marker="o", label="denorm")
    axes[1].set_title("Marker Vector Error by Future Step")
    axes[1].set_xlabel("future step")
    axes[1].set_ylabel("mean vector L2")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    axes[2].plot(steps, metrics["gt_raw_mag_by_step"], marker="o", label="GT")
    axes[2].plot(steps, metrics["pred_raw_mag_by_step"], marker="o", label="Pred")
    axes[2].set_title("Mean Marker Magnitude")
    axes[2].set_xlabel("future step")
    axes[2].set_ylabel("mean |offset|")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend()

    fig.suptitle(f"Val prediction summary over {metrics['num_windows']} sampled windows", y=1.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_sample_marker(pred_raw, gt_raw, sample_idx, out_path, steps=None):
    if steps is None:
        horizon = pred_raw.shape[1]
        steps = np.unique(np.linspace(0, horizon - 1, min(6, horizon), dtype=int)).tolist()

    pred = pred_raw[sample_idx]
    gt = gt_raw[sample_idx]
    err = np.linalg.norm(pred - gt, axis=-1)
    gt_mag = marker_mag(gt)
    pred_mag = marker_mag(pred)
    vmax = max(float(gt_mag[:, :, :].max()), float(pred_mag[:, :, :].max()), 1e-6)
    err_vmax = max(float(err.max()), 1e-6)

    n_rows = len(steps)
    fig, axes = plt.subplots(n_rows, 3, figsize=(9.5, 2.6 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]
    yy, xx = np.mgrid[0:9, 0:9]

    for r, step in enumerate(steps):
        for c, (title, field, mag, cmap, vmax_i) in enumerate([
            ("GT", gt[step], gt_mag[step], "viridis", vmax),
            ("Pred", pred[step], pred_mag[step], "viridis", vmax),
            ("|Pred-GT|", None, err[step], "magma", err_vmax),
        ]):
            ax = axes[r, c]
            im = ax.imshow(mag, cmap=cmap, vmin=0, vmax=vmax_i, origin="lower")
            if field is not None:
                ax.quiver(
                    xx, yy, field[..., 0], field[..., 1],
                    color="white", angles="xy", scale_units="xy", scale=1.0,
                    width=0.004, alpha=0.85,
                )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"t+{step + 1} {title}")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    fig.suptitle(f"Future marker reconstruction sample {sample_idx}", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_dir",
        default="/home/chenshuai/Project/output/foresight_ckpt/"
                "latent_foresight_board_260609_260610_multistep16_marker_only_e100_bs16_preload",
    )
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_batches", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args_cli = parser.parse_args()

    ckpt_dir = args_cli.ckpt_dir
    out_dir = args_cli.out_dir or os.path.join(ckpt_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)

    args, history, ckpt_path = load_run(ckpt_dir)
    plot_loss(history, os.path.join(out_dir, "loss_curves_detailed.png"))

    device = torch.device(args_cli.device if torch.cuda.is_available() and args_cli.device == "cuda" else "cpu")
    val_dataset, val_paths, norm_stats = build_val_dataset(args)
    loader = DataLoader(val_dataset, batch_size=args_cli.batch_size, shuffle=False, num_workers=0)
    model = build_model(args, ckpt_path, device)

    arrays = collect_predictions(model, loader, device, args_cli.max_batches)
    metrics, pred_raw, gt_raw = compute_metrics(arrays, norm_stats)
    metrics["val_episode_count"] = len(val_paths)
    metrics["ckpt_dir"] = ckpt_dir
    metrics["checkpoint"] = ckpt_path

    with open(os.path.join(out_dir, "val_prediction_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.savez_compressed(
        os.path.join(out_dir, "val_prediction_examples.npz"),
        pred_marker_raw=pred_raw,
        gt_marker_raw=gt_raw,
        pred_marker_norm=arrays["pred_marker_norm"],
        gt_marker_norm=arrays["gt_marker_norm"],
        pred_latent=arrays["pred_latent"],
        gt_latent=arrays["gt_latent"],
    )

    plot_sequence_metrics(metrics, os.path.join(out_dir, "val_sequence_metrics.png"))
    for sample_idx in range(min(args_cli.num_samples, pred_raw.shape[0])):
        plot_sample_marker(
            pred_raw,
            gt_raw,
            sample_idx,
            os.path.join(out_dir, f"sample_{sample_idx:02d}_marker_gt_pred_error.png"),
        )

    print(f"Saved figures to: {out_dir}")
    print(json.dumps({
        "num_windows": metrics["num_windows"],
        "marker_norm_mae": metrics["marker_norm_mae"],
        "marker_norm_vec_l2": metrics["marker_norm_vec_l2"],
        "marker_raw_vec_l2": metrics["marker_raw_vec_l2"],
        "latent_mae": metrics["latent_mae"],
    }, indent=2))


if __name__ == "__main__":
    main()
