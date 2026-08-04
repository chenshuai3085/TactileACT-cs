#!/usr/bin/env python3
"""Render held-out multi-task, multi-step tactile forecasts from legacy qpos models."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import h5py
import matplotlib
import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TFAC_V5.visualize_foresight_episode_video import (  # noqa: E402
    as_np_stats,
    build_condition_chunk,
    future_marker_windows,
    marker_window,
    normalize_marker,
    normalize_qpos,
)
from TFAC_V5.visualize_foresight_multistep import build_model, load_run  # noqa: E402


TASKS = [
    {
        "name": "Board wiping",
        "short": "board",
        "held_out_episode": (
            "/media/chenshuai/EXTERNAL_USB/pih_dataset/260625_v8j_caheiban/"
            "peg_in_hole_0625/episode_3.hdf5"
        ),
        "episode_candidates": [
            "/media/chenshuai/EXTERNAL_USB/pih_dataset/260625_v8j_caheiban/"
            "peg_in_hole_0625/episode_3.hdf5",
        ],
        "checkpoint": (
            "/home/chenshuai/Project/output/foresight_ckpt/"
            "latent_foresight_board_caheiban_260609_260610_260625_"
            "nostride_h16_future1_exhaustive_e10_bs32_0"
        ),
    },
    {
        "name": "Vase wiping",
        "short": "vase",
        "held_out_episode": (
            "/media/chenshuai/czy_data22/pih_dataset/260630_v8j_huaping/"
            "peg_in_hole_0630/episode_4.hdf5"
        ),
        "episode_candidates": [
            "/media/chenshuai/czy_data22/pih_dataset/260630_v8j_huaping/"
            "peg_in_hole_0630/episode_4.hdf5",
            "/media/chenshuai/EXTERNAL_USB/pih_dataset/260629_v8j_card/"
            "260630_v8j_huaping/peg_in_hole_0630/episode_4.hdf5",
        ],
        "checkpoint": (
            "/home/chenshuai/Project/output/foresight_ckpt/"
            "latent_foresight_huaping_260630_nostride_h16_future1_"
            "exhaustive_resume_best_e10_bs64"
        ),
    },
    {
        "name": "Card swiping",
        "short": "card",
        "held_out_episode": (
            "/media/chenshuai/czy_data22/pih_dataset/260707_v8j_card/"
            "peg_in_hole_0707/success/episode_0.hdf5"
        ),
        "episode_candidates": [
            "/media/chenshuai/czy_data22/pih_dataset/260707_v8j_card/"
            "peg_in_hole_0707/success/episode_0.hdf5",
            "/media/chenshuai/SANDISK ELE/\u6570\u636e\u91c7\u96c6/260707_v8j_card/"
            "peg_in_hole_0707/success/episode_0.hdf5",
        ],
        "checkpoint": (
            "/home/chenshuai/Project/output/foresight_ckpt/"
            "latent_foresight_card_260707_nostride_h16_future1_exhaustive_e10_bs32"
        ),
    },
]
HORIZON_STEPS = (1, 4, 8, 12, 16)
OUTPUT_BASE = ROOT / "paper/figures/foretac_multitask_qpos_preview"


def configure_reproducibility(seed: int = 0) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def load_episode(path: str, tac_side: str) -> Dict[str, np.ndarray]:
    with h5py.File(path, "r") as root:
        return {
            "qpos": root["observations/proprio_joint"][()].astype(np.float32),
            "marker": root[f"observations/tac/{tac_side}/marker_offset"][()].astype(np.float32),
            "global": root["observations/images/global"][()],
            "wrist": root["observations/images/wrist"][()],
        }


def predict_at(
    model: torch.nn.Module,
    data: Dict[str, np.ndarray],
    args: Dict[str, Any],
    stats: Dict[str, np.ndarray],
    t: int,
    device: torch.device,
) -> Dict[str, Any]:
    horizon = int(args.get("predict_horizon", 16))
    window = int(args.get("tactile_vae_window", 8))
    marker_norm = normalize_marker(data["marker"], stats)
    current_window = marker_window(marker_norm, t, window)
    future_windows = future_marker_windows(marker_norm, t, horizon, window, len(data["marker"]))
    condition = build_condition_chunk(data["qpos"], None, t, args, stats)
    qpos = normalize_qpos(data["qpos"][t], stats).astype(np.float32)

    images = [torch.from_numpy(current_window).unsqueeze(0).to(device)]
    future_images = [torch.from_numpy(future_windows).unsqueeze(0).to(device)]
    condition_t = torch.from_numpy(condition).unsqueeze(0).to(device)
    qpos_t = torch.from_numpy(qpos).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_latent, gt_latent, _, _ = model(
            images, condition_t, future_images=future_images, qpos=qpos_t
        )
        pred_norm = model.decode_latent_sequence(pred_latent)[0].cpu().numpy()

    mean = stats["marker_offset_mean"].reshape(1, 1, 1, 2)
    std = stats["marker_offset_std"].reshape(1, 1, 1, 2)
    pred = pred_norm * std + mean
    gt = np.stack([data["marker"][t + h] for h in range(1, horizon + 1)], axis=0)
    error = np.linalg.norm(pred - gt, axis=-1).mean(axis=(1, 2))
    contact = float(np.linalg.norm(data["marker"][t], axis=-1).mean())
    dynamics = float(np.linalg.norm(gt[-1] - data["marker"][t], axis=-1).mean())
    latent_mae = float(torch.abs(pred_latent - gt_latent).mean().cpu())
    return {
        "t": int(t),
        "current": data["marker"][t],
        "pred": pred,
        "gt": gt,
        "error": error,
        "mean_error": float(error.mean()),
        "final_error": float(error[-1]),
        "contact": contact,
        "dynamics": dynamics,
        "latent_mae": latent_mae,
        "rgb_global": data["global"][t],
        "rgb_wrist": data["wrist"][t],
    }


def choose_result(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    contact_cut = float(np.quantile([x["contact"] for x in candidates], 0.55))
    dynamics_cut = float(np.quantile([x["dynamics"] for x in candidates], 0.40))
    eligible = [
        x for x in candidates
        if x["contact"] >= contact_cut and x["dynamics"] >= dynamics_cut
    ]
    return min(eligible or candidates, key=lambda x: x["mean_error"])


def resolve_episode(task: Dict[str, Any]) -> str:
    for candidate in task["episode_candidates"]:
        if Path(candidate).is_file():
            return candidate
    raise FileNotFoundError(
        f"No mounted copy found for held-out episode: {task['held_out_episode']}"
    )


def collect_task(task: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    args, _, checkpoint = load_run(task["checkpoint"])
    if not bool(args.get("use_state_trajectory", False)):
        raise RuntimeError(f"Expected legacy qpos conditioning: {task['checkpoint']}")
    if int(args.get("predict_horizon", 0)) != 16:
        raise RuntimeError(f"Expected a 16-step predictor: {task['checkpoint']}")
    val_paths = Path(task["checkpoint"], "val_episode_paths.txt").read_text().splitlines()
    if task["held_out_episode"] not in val_paths:
        raise RuntimeError(f"Selected episode is not held out: {task['held_out_episode']}")
    episode = resolve_episode(task)

    model = build_model(args, checkpoint, device)
    stats = as_np_stats(args["norm_stats"])
    data = load_episode(episode, args.get("tac_side", "left"))
    horizon = int(args["predict_horizon"])
    lo = max(8, int(0.08 * len(data["marker"])))
    hi = min(len(data["marker"]) - horizon - 1, int(0.92 * len(data["marker"])))
    candidates = [
        predict_at(model, data, args, stats, t, device)
        for t in range(lo, hi, 6)
    ]
    selected = choose_result(candidates)
    selected.update({
        "task": task["name"],
        "short": task["short"],
        "episode": episode,
        "held_out_episode": task["held_out_episode"],
        "checkpoint_dir": task["checkpoint"],
        "checkpoint": checkpoint,
        "candidate_summary": [
            {k: row[k] for k in ("t", "mean_error", "final_error", "contact", "dynamics", "latent_mae")}
            for row in candidates
        ],
    })
    del model
    torch.cuda.empty_cache()
    return selected


def draw_rgb(ax: plt.Axes, global_rgb: np.ndarray, wrist_rgb: np.ndarray) -> None:
    def brighten(image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32)
        lo, hi = np.percentile(image, [2.0, 98.5])
        image = np.clip((image - lo) / max(float(hi - lo), 1.0), 0.0, 1.0)
        return np.clip(np.power(image, 0.72) * 255.0, 0, 255).astype(np.uint8)

    global_rgb, wrist_rgb = brighten(global_rgb), brighten(wrist_rgb)
    gap = np.full((global_rgb.shape[0], 4, 3), 255, dtype=np.uint8)
    ax.imshow(np.concatenate([global_rgb, gap, wrist_rgb], axis=1))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#CBD5E1")
        spine.set_linewidth(0.65)


def draw_marker(ax: plt.Axes, marker: np.ndarray, norm: Normalize) -> object:
    rows, cols = marker.shape[:2]
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    magnitude = np.linalg.norm(marker, axis=-1)
    image = ax.imshow(
        magnitude, origin="lower", cmap="viridis", norm=norm,
        interpolation="nearest", extent=(-0.5, cols - 0.5, -0.5, rows - 0.5),
    )
    ax.quiver(
        x, y, marker[..., 0], marker[..., 1], color="white",
        angles="xy", scale_units="xy", scale=10.0, width=0.010,
        headwidth=3.7, headlength=4.5, alpha=0.90,
    )
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(-0.5, rows - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#CBD5E1")
        spine.set_linewidth(0.55)
    return image


def make_figure(results: List[Dict[str, Any]], output_base: Path) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7.2,
        "axes.titlesize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(7.16, 6.35), dpi=260, facecolor="white")
    gs = GridSpec(
        6, 10, figure=fig,
        width_ratios=[1.56, 0.78, 0.13, 0.78, 0.78, 0.78, 0.78, 0.78, 1.25, 0.035],
        height_ratios=[1, 1, 1, 1, 1, 1],
        hspace=0.12, wspace=0.15,
    )

    for task_idx, item in enumerate(results):
        row_gt, row_pred = task_idx * 2, task_idx * 2 + 1
        arrays = [item["current"], item["gt"], item["pred"]]
        values = np.concatenate([np.linalg.norm(x, axis=-1).reshape(-1) for x in arrays])
        norm = Normalize(vmin=0.0, vmax=max(1.0, float(np.quantile(values, 0.99))))

        rgb_ax = fig.add_subplot(gs[row_gt:row_pred + 1, 0])
        draw_rgb(rgb_ax, item["rgb_global"], item["rgb_wrist"])
        rgb_ax.set_title(
            f"{item['task']}\n$t={item['t']}$",
            loc="left", fontweight="semibold", pad=3,
        )

        current_ax = fig.add_subplot(gs[row_gt:row_pred + 1, 1])
        last_image = draw_marker(current_ax, item["current"], norm)
        if task_idx == 0:
            current_ax.set_title("Current", pad=3)

        label_gt = fig.add_subplot(gs[row_gt, 2])
        label_pred = fig.add_subplot(gs[row_pred, 2])
        for label_ax, label, color in (
            (label_gt, "GT", "#1F2937"),
            (label_pred, "Pred", "#B4233A"),
        ):
            label_ax.axis("off")
            label_ax.text(
                0.5, 0.5, label, rotation=90, ha="center", va="center",
                fontsize=7, fontweight="semibold", color=color,
            )

        for col_idx, horizon in enumerate(HORIZON_STEPS, start=3):
            gt_ax = fig.add_subplot(gs[row_gt, col_idx])
            pred_ax = fig.add_subplot(gs[row_pred, col_idx])
            last_image = draw_marker(gt_ax, item["gt"][horizon - 1], norm)
            draw_marker(pred_ax, item["pred"][horizon - 1], norm)
            if task_idx == 0:
                gt_ax.set_title(f"$H={horizon}$", pad=3)

        curve_ax = fig.add_subplot(gs[row_gt:row_pred + 1, 8])
        steps = np.arange(1, 17)
        curve_ax.plot(steps, item["error"], color="#D1495B", linewidth=1.45)
        curve_ax.scatter(
            list(HORIZON_STEPS), item["error"][np.asarray(HORIZON_STEPS) - 1],
            color="#D1495B", s=12, edgecolor="white", linewidth=0.4, zorder=3,
        )
        curve_ax.fill_between(steps, item["error"], color="#D1495B", alpha=0.09)
        curve_ax.set_xlim(1, 16)
        curve_ax.set_ylim(bottom=0)
        curve_ax.set_xticks([1, 4, 8, 12, 16])
        curve_ax.tick_params(labelsize=6.2, length=2)
        curve_ax.tick_params(axis="y", labelleft=False, left=False)
        curve_ax.grid(axis="y", color="#E2E8F0", linewidth=0.55)
        curve_ax.spines[["top", "right"]].set_visible(False)
        curve_ax.spines[["left", "bottom"]].set_color("#CBD5E1")
        if task_idx == 0:
            curve_ax.set_title("Prediction error", pad=4)
        curve_ax.set_xlabel("Future step", fontsize=6.5, labelpad=2)
        curve_ax.text(
            0.02, 0.98, f"Mean {item['mean_error']:.3f} px", transform=curve_ax.transAxes,
            ha="left", va="top", fontsize=6.2, color="#64748B",
        )

        cax = fig.add_subplot(gs[row_gt:row_pred + 1, 9])
        colorbar = fig.colorbar(last_image, cax=cax)
        colorbar.ax.tick_params(labelsize=5.8, length=1.5)
        colorbar.outline.set_linewidth(0.5)

    fig.text(
        0.5, 0.008,
        "Multi-step tactile forecasts on held-out real-robot episodes. Color and arrows encode marker displacement magnitude and direction.",
        ha="center", va="bottom", fontsize=6.8, color="#475569",
    )
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), bbox_inches="tight", pad_inches=0.05)
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def serializable(item: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in item.items() if not isinstance(v, np.ndarray)}


def main() -> None:
    configure_reproducibility()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this visualization run")
    device = torch.device("cuda")
    results = [collect_task(task, device) for task in TASKS]
    make_figure(results, OUTPUT_BASE)

    np.savez_compressed(
        OUTPUT_BASE.with_suffix(".npz"),
        task=np.asarray([x["short"] for x in results]),
        timestep=np.asarray([x["t"] for x in results], dtype=np.int64),
        current=np.stack([x["current"] for x in results]),
        gt=np.stack([x["gt"] for x in results]),
        pred=np.stack([x["pred"] for x in results]),
        error=np.stack([x["error"] for x in results]),
    )
    OUTPUT_BASE.with_suffix(".json").write_text(
        json.dumps({
            "scope": "Held-out qpos-conditioned multi-step foresight diagnostic.",
            "selection": (
                "Lowest mean 16-step marker error among sampled frames above the 55th contact "
                "and 40th future-dynamics percentiles."
            ),
            "horizon_steps_shown": list(HORIZON_STEPS),
            "results": [serializable(x) for x in results],
        }, indent=2),
        encoding="utf-8",
    )
    fields = ["task", "episode", "checkpoint", "t", "mean_error", "final_error", "contact", "dynamics", "latent_mae"]
    with OUTPUT_BASE.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows([{k: x[k] for k in fields} for x in results])
    print(OUTPUT_BASE.with_suffix(".png"))
    print(json.dumps([
        {k: x[k] for k in ("task", "episode", "t", "mean_error", "final_error", "contact", "dynamics")}
        for x in results
    ], indent=2))


if __name__ == "__main__":
    main()
