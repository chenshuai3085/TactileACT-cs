#!/usr/bin/env python3
"""Evaluate V1/V2 action-conditioned foresight on identical held-out windows."""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from TFAC_V5.dataset import ForesightEpisodicDataset
from TFAC_V5 import pretrain_latent_foresight_multistep as trainer
from TFAC_V5.tactile_vae_v2 import TactileVAEv2


V1_DIR = Path(
    "/home/chenshuai/Project/output/foresight_ckpt/"
    "latent_foresight_board_260609_260610_multistep16_boardvae_marker_only_"
    "action_cond_e100_bs16_preload"
)
V2_DIR = Path(
    "/home/chenshuai/Project/output/foresight_ckpt/"
    "v2_board_action_conditioned_h16_e100_20260820"
)
V2_CONFIG = Path(
    "/home/chenshuai/Project/output/"
    "tactile_vae_v2_board_260609_260610_left_tw8_ld16_s2_e150/"
    "foresight_v2_formal.json"
)


class V2MultiStepLatentForesightModel(trainer.MultiStepLatentForesightModel):
    def __init__(self, *args, tactile_vae_ckpt=None, tactile_vae_latent_dim=16,
                 tactile_vae_window=8, **kwargs):
        super().__init__(
            *args,
            tactile_vae_ckpt=None,
            tactile_vae_latent_dim=tactile_vae_latent_dim,
            tactile_vae_window=tactile_vae_window,
            **kwargs,
        )
        self.tactile_vae = TactileVAEv2(
            latent_dim=tactile_vae_latent_dim,
            temporal_window=int(tactile_vae_window),
        )
        checkpoint = torch.load(tactile_vae_ckpt, map_location="cpu", weights_only=False)
        self.tactile_vae.load_state_dict(checkpoint["model_state_dict"])
        self.tactile_vae.requires_grad_(False)


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_paths(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def load_stats(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def build_model(version, config, checkpoint_path, device):
    cls = trainer.MultiStepLatentForesightModel if version == "v1" else V2MultiStepLatentForesightModel
    camera_names = config.get("camera_names", ["gelsight"])
    model = cls(
        camera_names=camera_names,
        cam_backbone_mapping={name: 0 for name in camera_names},
        hidden_dim=int(config.get("hidden_dim", 512)),
        state_dim=int(config.get("state_dim", 7)),
        foresight_layers=int(config.get("foresight_layers", 3)),
        foresight_nheads=int(config.get("foresight_nheads", 8)),
        foresight_dim_feedforward=int(config.get("foresight_dim_feedforward", 2048)),
        dropout=float(config.get("dropout", 0.1)),
        tactile_mode=config.get("tactile_mode", "marker"),
        max_history=int(config.get("max_history", 8)),
        predict_horizon=int(config.get("predict_horizon", 16)),
        tactile_vae_ckpt=config["tactile_vae_ckpt"],
        tactile_vae_latent_dim=int(config.get("tactile_vae_latent_dim", 16)),
        tactile_vae_window=int(config.get("tactile_vae_window", 8)),
    )
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval().requires_grad_(False)


def build_dataset(config, stats, val_paths):
    return ForesightEpisodicDataset(
        val_paths,
        os.path.dirname(val_paths[0]),
        config.get("camera_names", ["gelsight"]),
        stats,
        chunk_size=int(config.get("chunk_size", 16)),
        foresight_horizon=int(config.get("foresight_horizon", 16)),
        proprio_key=config.get("proprio_key", "proprio_joint"),
        action_key=config.get("action_key", "actions/joint_abs"),
        tac_side=config.get("tac_side", "left"),
        tac_img_key=config.get("tac_img_key", "img"),
        tactile_mode="marker",
        history_len=int(config.get("history_len", 1)),
        tactile_vae_window=int(config.get("tactile_vae_window", 8)),
        preload=True,
        use_state_trajectory=bool(config.get("use_state_trajectory", False)),
        future_offset=int(config.get("future_offset", 1)),
        action_offset=int(config.get("action_offset", 0)),
        temporal_stride=int(config.get("temporal_stride", 1)),
        exhaustive_windows=True,
    )


def to_device(batch, device):
    images, qpos, actions, _, future_images, _ = batch
    return (
        [value.to(device, non_blocking=True) for value in images],
        qpos.to(device, non_blocking=True),
        actions.to(device, non_blocking=True),
        [value.to(device, non_blocking=True) for value in future_images],
    )


def evaluate(version, model_dir, config, device, batch_size, max_windows):
    stats = load_stats(model_dir / "dataset_stats.pkl")
    val_paths = load_paths(model_dir / "val_episode_paths.txt")
    dataset = build_dataset(config, stats, val_paths)
    if max_windows and max_windows < len(dataset):
        dataset = Subset(dataset, range(max_windows))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = build_model(version, config, model_dir / "foresight_best.ckpt", device)

    horizon = int(config.get("predict_horizon", 16))
    sums = {name: np.zeros(horizon, np.float64) for name in (
        "latent_mae", "latent_cosine", "marker_mae", "marker_rmse",
        "marker_cosine",
    )}
    counts = np.zeros(horizon, np.int64)
    mean = torch.as_tensor(stats["marker_offset_mean"], device=device).view(1, 1, 1, 1, 2)
    std = torch.as_tensor(stats["marker_offset_std"], device=device).view(1, 1, 1, 1, 2)

    with torch.inference_mode():
        for batch in loader:
            images, qpos, actions, future_images = to_device(batch, device)
            pred, target, marker_norm, _ = model(
                images, actions, future_images=future_images, qpos=qpos)
            decoded_norm = model.decode_latent_sequence(pred)
            decoded = decoded_norm * std + mean
            marker = marker_norm * std + mean

            latent_mae = (pred - target).abs().mean(dim=-1)
            latent_cos = F.cosine_similarity(pred, target, dim=-1)
            marker_mae = (decoded - marker).abs().mean(dim=(2, 3, 4))
            marker_rmse = (decoded - marker).square().mean(dim=(2, 3, 4)).sqrt()
            pred_vec = decoded.flatten(2, 3)
            gt_vec = marker.flatten(2, 3)
            vector_cos = F.cosine_similarity(pred_vec, gt_vec, dim=-1)
            active = gt_vec.norm(dim=-1) > 1e-4
            marker_cos = (vector_cos * active).sum(dim=-1) / active.sum(dim=-1).clamp(min=1)

            values = {
                "latent_mae": latent_mae,
                "latent_cosine": latent_cos,
                "marker_mae": marker_mae,
                "marker_rmse": marker_rmse,
                "marker_cosine": marker_cos,
            }
            batch_n = pred.size(0)
            counts[:pred.size(1)] += batch_n
            for name, value in values.items():
                sums[name][:value.size(1)] += value.double().sum(dim=0).cpu().numpy()

    metrics = {name: (value / counts).tolist() for name, value in sums.items()}
    return {
        "version": version,
        "checkpoint": str(model_dir / "foresight_best.ckpt"),
        "heldout_episodes": len(val_paths),
        "windows": int(counts[0]),
        "metrics_by_horizon": metrics,
        "mean_over_horizon": {name: float(np.mean(value)) for name, value in metrics.items()},
    }


def write_report(results, output_dir, model_keys):
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    lines = [
        "# V1/V2 Action-Conditioned Foresight Evaluation", "",
        f"Held-out episodes: {results['contract']['heldout_episodes']}; "
        f"identical exhaustive windows: {results['contract']['windows']}.", "",
        "| Model | Raw marker MAE | Raw marker RMSE | Marker cosine | Latent MAE | Latent cosine |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "v1": "V1",
        "v2": "Full V2",
        "v2_without_intensity_ranking": "V2 without intensity/ranking",
    }
    for key in model_keys:
        row = results[key]["mean_over_horizon"]
        lines.append(
            f"| {labels[key]} | {row['marker_mae']:.6f} | {row['marker_rmse']:.6f} | "
            f"{row['marker_cosine']:.6f} | {row['latent_mae']:.6f} | {row['latent_cosine']:.6f} |"
        )
    lines.extend([
        "", "## Per-horizon raw marker MAE", "",
        "| Horizon | " + " | ".join(labels[key] for key in model_keys) + " |",
        "|---:|" + "---:|" * len(model_keys),
    ])
    horizon = len(results[model_keys[0]]["metrics_by_horizon"]["marker_mae"])
    for index in range(horizon):
        values = [results[key]["metrics_by_horizon"]["marker_mae"][index] for key in model_keys]
        lines.append(f"| {index + 1} | " + " | ".join(f"{value:.6f}" for value in values) + " |")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    horizons = np.arange(1, horizon + 1)
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.1))
    colors = {
        "v1": "#4978A8",
        "v2": "#D36B4A",
        "v2_without_intensity_ranking": "#4F8A65",
    }
    for key in model_keys:
        axes[0].plot(horizons, results[key]["metrics_by_horizon"]["marker_mae"],
                     marker="o", ms=3, lw=1.8, color=colors[key], label=labels[key])
        axes[1].plot(horizons, results[key]["metrics_by_horizon"]["marker_cosine"],
                     marker="o", ms=3, lw=1.8, color=colors[key], label=labels[key])
    axes[0].set_ylabel("Raw marker MAE")
    axes[1].set_ylabel("Marker cosine")
    for axis in axes:
        axis.set_xlabel("Prediction horizon")
        axis.set_xticks([1, 4, 8, 12, 16])
        axis.grid(alpha=0.22, linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "foresight_horizon_comparison.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / "foresight_horizon_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--v2-ablated-dir", type=Path)
    parser.add_argument("--v2-ablated-config", type=Path)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    v1_config = load_json(V1_DIR / "args.json")
    v2_config = load_json(V2_CONFIG)
    v1 = evaluate("v1", V1_DIR, v1_config, device, args.batch_size, args.max_windows)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    v2 = evaluate("v2", V2_DIR, v2_config, device, args.batch_size, args.max_windows)
    if v1["windows"] != v2["windows"]:
        raise RuntimeError(f"Window mismatch: V1={v1['windows']}, V2={v2['windows']}")
    results = {
        "contract": {
            "heldout_episodes": v1["heldout_episodes"],
            "windows": v1["windows"],
            "future_offset": 1,
            "action_offset": 0,
            "horizon": 16,
            "primary_cross_model_metric": "decoded raw-marker error",
        },
        "v1": v1,
        "v2": v2,
    }
    model_keys = ["v1", "v2"]
    if args.v2_ablated_dir or args.v2_ablated_config:
        if not (args.v2_ablated_dir and args.v2_ablated_config):
            raise ValueError("Both --v2-ablated-dir and --v2-ablated-config are required")
        ablated_config = load_json(args.v2_ablated_config)
        ablated = evaluate(
            "v2", args.v2_ablated_dir, ablated_config, device,
            args.batch_size, args.max_windows,
        )
        if ablated["windows"] != v1["windows"]:
            raise RuntimeError(
                f"Window mismatch: reference={v1['windows']}, ablated={ablated['windows']}"
            )
        results["v2_without_intensity_ranking"] = ablated
        model_keys.append("v2_without_intensity_ranking")
    write_report(results, args.output_dir, model_keys)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
