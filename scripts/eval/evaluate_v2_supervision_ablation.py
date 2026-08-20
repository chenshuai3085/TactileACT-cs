#!/usr/bin/env python3
"""Compare full V2 against the same architecture without intensity/rank losses."""

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, Dataset, Subset


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from TFAC_V5.tactile_vae_v2 import TactileVAEv2


DATA_DIRS = [
    Path("/media/chenshuai/EXTERNAL_USB/pih_dataset/260609/wipe_pos_straight_z124_125_150_20260609"),
    Path("/media/chenshuai/EXTERNAL_USB/pih_dataset/260609/z_too_high"),
    Path("/media/chenshuai/EXTERNAL_USB/pih_dataset/260610/z_too_low"),
    Path("/media/chenshuai/EXTERNAL_USB/pih_dataset/260610/z_too_oscillate"),
]
SPLIT_DIR = Path(
    "/home/chenshuai/Project/output/foresight_ckpt/"
    "latent_foresight_board_260609_260610_multistep16_boardvae_marker_only_"
    "action_cond_e100_bs16_preload"
)


class MarkerForceWindows(Dataset):
    def __init__(self, mean, std, window=8, stride=2):
        self.mean = np.asarray(mean, np.float32)
        self.std = np.asarray(std, np.float32)
        self.marker, self.force, self.windows = [], [], []
        for directory in DATA_DIRS:
            for path in sorted(directory.glob("episode_*.hdf5")):
                with h5py.File(path, "r") as handle:
                    marker = handle["observations/tac/left/marker_offset"][()].astype(np.float32)
                    force = handle["observations/tac/left/force6d"][()].astype(np.float32)
                if len(marker) < window or len(marker) != len(force):
                    continue
                stream = len(self.marker)
                self.marker.append(marker)
                self.force.append(force)
                self.windows.extend((stream, start) for start in range(0, len(marker) - window + 1, stride))
        self.window = window

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        stream, start = self.windows[index]
        raw = self.marker[stream][start:start + self.window]
        normalized = (raw - self.mean) / self.std
        force = self.force[stream][start + self.window - 1, :3]
        return torch.from_numpy(normalized), torch.from_numpy(raw[-1]), torch.from_numpy(force)


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = TactileVAEv2(latent_dim=16, temporal_window=8)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval().requires_grad_(False), checkpoint


def pair_accuracy(score, target, seed=42, pairs=500000):
    rng = np.random.default_rng(seed)
    left = rng.integers(0, len(score), pairs)
    right = rng.integers(0, len(score), pairs)
    delta = target[left] - target[right]
    valid = (left != right) & (np.abs(delta) > 1e-3)
    return float(np.mean(np.sign(score[left][valid] - score[right][valid]) == np.sign(delta[valid])))


def extract_window_metrics(model, loader, mean, std, device):
    recon_error, cosine, intensity, targets, attention = [], [], [], [], []
    mean_t = torch.as_tensor(mean, device=device).view(1, 1, 1, 2)
    std_t = torch.as_tensor(std, device=device).view(1, 1, 1, 2)
    with torch.inference_mode():
        for normalized, raw, force in loader:
            normalized = normalized.to(device, non_blocking=True)
            raw = raw.to(device, non_blocking=True)
            recon, mu, _, latent, weights = model(normalized)
            recon_raw = recon * std_t + mean_t
            recon_error.append((recon_raw - raw).abs().mean(dim=(1, 2, 3)).cpu().numpy())
            gt_vec = raw.flatten(1, 2)
            pred_vec = recon_raw.flatten(1, 2)
            active = gt_vec.norm(dim=-1) > 1e-4
            vec_cos = F.cosine_similarity(pred_vec, gt_vec, dim=-1)
            cosine.append(((vec_cos * active).sum(-1) / active.sum(-1).clamp(min=1)).cpu().numpy())
            intensity.append(latent[:, :1].mean(dim=(1, 2, 3)).cpu().numpy())
            normalized_last = normalized[:, -1]
            targets.append(np.column_stack([
                normalized_last.norm(dim=-1).mean(dim=(1, 2)).cpu().numpy(),
                raw.norm(dim=-1).mean(dim=(1, 2)).cpu().numpy(),
                force[:, 2].numpy(),
                force.norm(dim=-1).numpy(),
            ]))
            attention.append(weights.squeeze(1).cpu().numpy())
    error = np.concatenate(recon_error)
    score = np.concatenate(intensity)
    target = np.concatenate(targets)
    direct = {}
    names = ("normalized_marker_activity", "raw_marker_activity", "fz", "force_xyz")
    for index, name in enumerate(names):
        direct[name] = {
            "pearson": float(pearsonr(score, target[:, index]).statistic),
            "spearman": float(spearmanr(score, target[:, index]).statistic),
            "pairwise_accuracy": pair_accuracy(score, target[:, index]),
        }
    return {
        "raw_marker_mae": float(error.mean()),
        "raw_marker_mae_median": float(np.median(error)),
        "marker_cosine": float(np.concatenate(cosine).mean()),
        "direct_intensity_channel": direct,
        "mean_temporal_attention": np.concatenate(attention).mean(axis=0).tolist(),
        "per_window_raw_mae": error,
    }


def encode_episode(model, marker, mean, std, device, batch_size=512):
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    normalized = ((marker.astype(np.float32) - mean) / std).astype(np.float32)
    windows = np.lib.stride_tricks.sliding_window_view(normalized, 8, axis=0)
    windows = windows.transpose(0, 4, 1, 2, 3).copy()
    assert windows.dtype == np.float32 and windows.shape[1:] == (8, 9, 9, 2)
    output = []
    with torch.inference_mode():
        for start in range(0, len(windows), batch_size):
            batch = torch.from_numpy(windows[start:start + batch_size]).to(device)
            mu, _ = model.encode(batch)
            latent, _ = model.temporal_aggregate(mu)
            output.append(latent.flatten(1).cpu().numpy())
    return np.concatenate(output)


def read_split(name):
    return [Path(line) for line in (SPLIT_DIR / name).read_text().splitlines() if line]


def fit_latent_stats(model, paths, mean, std, device):
    rng = np.random.default_rng(42)
    samples = []
    for path in paths:
        with h5py.File(path, "r") as handle:
            marker = handle["observations/tac/left/marker_offset"][()].astype(np.float32)
        latent = encode_episode(model, marker, mean, std, device)
        if len(latent) > 64:
            latent = latent[rng.choice(len(latent), 64, replace=False)]
        samples.append(latent)
    values = np.concatenate(samples).astype(np.float64)
    return values.mean(axis=0), np.maximum(values.std(axis=0), 1e-5)


def latent_smoothness(model, mean, std, device):
    train_paths = read_split("train_episode_paths.txt")
    paths = read_split("val_episode_paths.txt")
    latent_mean, latent_std = fit_latent_stats(model, train_paths, mean, std, device)
    first, second, relative = [], [], []
    for path in paths:
        with h5py.File(path, "r") as handle:
            marker = handle["observations/tac/left/marker_offset"][()].astype(np.float32)
        latent = encode_episode(model, marker, mean, std, device)
        latent = (latent - latent_mean) / latent_std
        dz = np.diff(latent, axis=0)
        ddz = np.diff(latent, n=2, axis=0)
        first.append(np.linalg.norm(dz, axis=1) / np.sqrt(latent.shape[1]))
        second.append(np.linalg.norm(ddz, axis=1) / np.sqrt(latent.shape[1]))
        relative.append(np.linalg.norm(ddz, axis=1) /
                        (np.linalg.norm(dz[:-1], axis=1) + np.linalg.norm(dz[1:], axis=1) + 1e-8))
    return {
        "whitened_step_rms": float(np.concatenate(first).mean()),
        "whitened_curvature_rms": float(np.concatenate(second).mean()),
        "relative_curvature": float(np.concatenate(relative).mean()),
        "heldout_episodes": len(paths),
    }


def plot_summary(result, output_dir):
    labels = ["Without supervision", "Full V2"]
    keys = ["without_intensity_ranking", "full_v2"]
    colors = ["#4F8A65", "#D36B4A"]
    marker_order = [
        result[key]["direct_intensity_channel"]["normalized_marker_activity"]["pairwise_accuracy"]
        for key in keys
    ]
    fz_order = [
        result[key]["direct_intensity_channel"]["fz"]["pairwise_accuracy"]
        for key in keys
    ]
    marker_mae = [result[key]["raw_marker_mae"] for key in keys]
    smoothness = [result[key]["latent_smoothness"]["whitened_step_rms"] for key in keys]

    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.0))
    x = np.arange(2)
    width = 0.34
    axes[0].bar(x - width / 2, marker_order, width, color="#4978A8", label="Marker activity")
    axes[0].bar(x + width / 2, fz_order, width, color="#C8A24A", label="Fz")
    axes[0].axhline(0.5, color="#777777", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("Pair-order accuracy")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].bar(x, marker_mae, color=colors, width=0.55)
    axes[1].set_ylabel("Raw marker MAE")
    axes[2].bar(x, smoothness, color=colors, width=0.55)
    axes[2].set_ylabel("Whitened latent step RMS")
    for axis in axes:
        axis.set_xticks(x, labels, rotation=12, ha="right")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(output_dir / "supervision_ablation_summary.png", dpi=240, bbox_inches="tight")
    fig.savefig(output_dir / "supervision_ablation_summary.pdf", bbox_inches="tight")
    plt.close(fig)


def serializable(metrics):
    return {key: value for key, value in metrics.items() if key != "per_window_raw_mae"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--without-supervision", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full, full_ckpt = load_model(args.full, device)
    ablated, ablated_ckpt = load_model(args.without_supervision, device)
    full_stats = full_ckpt["norm_stats"]
    ablated_stats = ablated_ckpt["norm_stats"]
    if not (np.allclose(full_stats["mean"], ablated_stats["mean"]) and
            np.allclose(full_stats["std"], ablated_stats["std"])):
        raise ValueError("Normalization statistics differ")

    dataset = MarkerForceWindows(full_stats["mean"], full_stats["std"])
    n_val = max(1, int(len(dataset) * 0.1))
    permutation = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(42)).tolist()
    val_indices = permutation[len(dataset) - n_val:]
    loader = DataLoader(Subset(dataset, val_indices), batch_size=512, shuffle=False, num_workers=4)
    full_metrics = extract_window_metrics(full, loader, full_stats["mean"], full_stats["std"], device)
    ablated_metrics = extract_window_metrics(ablated, loader, full_stats["mean"], full_stats["std"], device)
    full_metrics["latent_smoothness"] = latent_smoothness(
        full, np.asarray(full_stats["mean"]), np.asarray(full_stats["std"]), device)
    ablated_metrics["latent_smoothness"] = latent_smoothness(
        ablated, np.asarray(full_stats["mean"]), np.asarray(full_stats["std"]), device)

    full_error = full_metrics["per_window_raw_mae"]
    ablated_error = ablated_metrics["per_window_raw_mae"]
    result = {
        "contract": {
            "all_windows": len(dataset),
            "validation_windows": len(val_indices),
            "seed": 42,
            "same_architecture": True,
            "only_removed_losses": ["intensity", "ranking"],
        },
        "full_v2": serializable(full_metrics),
        "without_intensity_ranking": serializable(ablated_metrics),
        "paired": {
            "full_lower_reconstruction_mae_fraction": float(np.mean(full_error < ablated_error)),
            "full_minus_ablated_raw_mae": float(np.mean(full_error - ablated_error)),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        "# TactileVAE V2 Intensity/Ranking Supervision Ablation", "",
        "| Model | Raw marker MAE | Marker cosine | Intensity Spearman | Intensity pair accuracy | Fz Spearman | Fz pair accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, key in (("Without intensity/ranking", "without_intensity_ranking"), ("Full V2", "full_v2")):
        values = result[key]
        normalized = values["direct_intensity_channel"]["normalized_marker_activity"]
        fz = values["direct_intensity_channel"]["fz"]
        lines.append(
            f"| {label} | {values['raw_marker_mae']:.6f} | {values['marker_cosine']:.6f} | "
            f"{normalized['spearman']:.4f} | {normalized['pairwise_accuracy']:.2%} | "
            f"{fz['spearman']:.4f} | {fz['pairwise_accuracy']:.2%} |"
        )
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plot_summary(result, args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
