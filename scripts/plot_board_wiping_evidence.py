#!/usr/bin/env python3
"""Build statistically grounded board-wiping and guidance figures.

The rollout panels use recorded robot/tactile logs and an offline scorer trace.
The diagnostic panels use paired pre/post updates from the same sampler state;
held-out windows are used only for the action-space gradient map.
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TRIAL = ROOT / "outputs/multitask_replay_records/caheiban_260609/20260708_180826_port8781_episode_6"
VIDEO = ROOT / "tmp_vtm_ProjectPage/static/videos/board_real.mp4"
SCORE_CSV = ROOT / "outputs/foresight_retrain_20260708/board_quality_scores/caheiban_260609/20260708_180826_port8781_episode_6/board_quality_score_curve.csv"
DENSE_CSV = ROOT / "outputs/board_stride3_guidance_gradient_vis/ddpm_gradient_steps.csv"
GRAD_NPZ = ROOT / "outputs/board_stride3_val_guidance_debug_20260714_n128/val_action_gradient_tensors.npz"
OUT = ROOT / "paper/figures"

BLUE = "#2878B5"
GREEN = "#2A9D5B"
ORANGE = "#E68632"
RED = "#D84A4A"
INK = "#25313C"
MUTED = "#687581"
GRID = "#DDE3E8"


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 8.8,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.2,
            "axes.edgecolor": "#88939D",
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 400,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_numeric_csv(path: Path) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    return {
        key: np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        for key in rows[0]
        if key not in {"wall_time_iso"}
        and all(row.get(key, "") not in {"", "nan", "NaN"} for row in rows)
    }


def smooth(values: np.ndarray, window: int = 11) -> np.ndarray:
    if window <= 1:
        return values.copy()
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def read_video_frames(progress: list[float]) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {VIDEO}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for value in progress:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(value * (count - 1))))
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Cannot read frame at progress={value}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        # Keep the robot and board while removing unused room margins.
        frame = frame[int(0.12 * h) : int(0.93 * h), int(0.12 * w) : int(0.90 * w)]
        frames.append(frame)
    cap.release()
    return frames


def load_rollout() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    trace = read_numeric_csv(TRIAL / "trace.csv")
    baseline = np.nanmedian(trace["left_fy"][:20])
    normal_force = smooth(np.maximum(-(trace["left_fy"] - baseline), 0.0), 13)
    progress = np.linspace(0.0, 100.0, len(normal_force))
    score = read_numeric_csv(SCORE_CSV)["score"]
    score = smooth(score, 9)
    if len(score) != len(progress):
        score = np.interp(progress, np.linspace(0, 100, len(score)), score)
    return progress, normal_force, score


def load_updates() -> dict[str, np.ndarray]:
    rows = [
        row
        for row in csv.DictReader(DENSE_CSV.open(encoding="utf-8"))
        if row["mode"] == "guided"
    ]
    pre = np.asarray([float(row["pre_expert_margin"]) for row in rows])
    post = np.asarray([float(row["post_expert_margin"]) for row in rows])
    update = np.asarray([float(row["applied_update_norm"]) for row in rows])
    grad = np.asarray([float(row["grad_norm"]) for row in rows])
    return {"pre": pre, "post": post, "delta": post - pre, "update": update, "grad": grad}


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.08, 1.06, label, transform=ax.transAxes, fontsize=10, fontweight="bold", color=INK)


def rollout_frames(fig: plt.Figure, spec, frame_progress: list[float]) -> None:
    sub = spec.subgridspec(1, 5, width_ratios=[0.52, 1, 1, 1, 1], wspace=0.055)
    stages = ["Approach", "Initial contact", "Stable wiping", "Completion"]
    frames = read_video_frames(frame_progress)
    label_ax = fig.add_subplot(sub[0, 0])
    label_ax.axis("off")
    label_ax.text(0.02, 0.62, "(a)", fontsize=10, fontweight="bold", color=INK)
    label_ax.text(0.02, 0.39, "Recorded\nexecution", fontsize=8.4, fontweight="bold", color=INK, linespacing=1.15)
    for idx, (stage, frame) in enumerate(zip(stages, frames), start=1):
        ax = fig.add_subplot(sub[0, idx])
        ax.imshow(frame)
        ax.set_title(stage, pad=3.5, color=INK, fontweight="semibold")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#CAD2D9")
            spine.set_linewidth(0.8)


def rollout_signals(fig: plt.Figure, spec, progress: np.ndarray, force: np.ndarray, score: np.ndarray, stage_progress: list[float]) -> None:
    sub = spec.subgridspec(1, 2, wspace=0.28)
    ax_force = fig.add_subplot(sub[0, 0])
    panel_label(ax_force, "(b)")
    ax_force.plot(progress, force, color=RED, linewidth=1.45, label="Normal contact force")
    ax_force.fill_between(progress, 0, force, color=RED, alpha=0.08, linewidth=0)
    for p in stage_progress:
        ax_force.axvline(100 * p, color=GRID, linewidth=0.7, zorder=0)
    ax_force.set(xlabel="Task progress (%)", ylabel="Normal contact force (N)", xlim=(0, 100), ylim=(0, max(14, np.ceil(force.max()))))
    ax_force.grid(axis="y", color=GRID, linewidth=0.55)
    ax_force.legend(frameon=False, loc="upper right")

    ax_score = fig.add_subplot(sub[0, 1])
    panel_label(ax_score, "(c)")
    ax_score.plot(progress, score, color=BLUE, linewidth=1.45, label="Offline contact-quality score")
    ax_score.fill_between(progress, 0, score, color=BLUE, alpha=0.08, linewidth=0)
    for p in stage_progress:
        ax_score.axvline(100 * p, color=GRID, linewidth=0.7, zorder=0)
    ax_score.set(xlabel="Task progress (%)", ylabel="Contact-quality score", xlim=(0, 100), ylim=(0, 1.0))
    ax_score.grid(axis="y", color=GRID, linewidth=0.55)
    ax_score.legend(frameon=False, loc="lower right")


def diagnostic_panels(fig: plt.Figure, spec, updates: dict[str, np.ndarray]) -> None:
    sub = spec.subgridspec(1, 3, width_ratios=[1.05, 0.85, 1.25], wspace=0.35)

    ax_delta = fig.add_subplot(sub[0, 0])
    panel_label(ax_delta, "(d)")
    delta = updates["delta"]
    clipped = np.clip(delta, np.percentile(delta, 1), np.percentile(delta, 99))
    parts = ax_delta.violinplot(clipped, positions=[0], widths=0.68, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor(GREEN)
        body.set_edgecolor(GREEN)
        body.set_alpha(0.23)
    rng = np.random.default_rng(12)
    pick = rng.choice(len(clipped), min(100, len(clipped)), replace=False)
    ax_delta.scatter(rng.normal(0, 0.055, len(pick)), clipped[pick], s=7, color=GREEN, alpha=0.36, edgecolors="none")
    ax_delta.axhline(0, color=MUTED, linestyle="--", linewidth=0.8)
    ax_delta.scatter([0], [np.median(delta)], marker="D", s=24, color=INK, zorder=5, label="Median")
    improve = 100 * np.mean(delta > 0)
    ax_delta.text(0.97, 0.96, f"{improve:.1f}% improve\n$n={len(delta)}$ paired updates", transform=ax_delta.transAxes, ha="right", va="top", color=INK, fontsize=7.4,
                  bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.5))
    ax_delta.text(0.03, 0.04, "1st--99th percentile shown", transform=ax_delta.transAxes, ha="left", va="bottom", color=MUTED, fontsize=6.5)
    ax_delta.set(xlim=(-0.55, 0.55), xticks=[0], xticklabels=["Same-state\nrefinement"], ylabel=r"Score change  $S_{after}-S_{before}$")
    ax_delta.grid(axis="y", color=GRID, linewidth=0.55)

    ax_stats = fig.add_subplot(sub[0, 1])
    panel_label(ax_stats, "(e)")
    finite = 100 * np.mean(np.isfinite(updates["grad"]))
    bounded = 100 * np.mean(updates["update"] <= 0.003001)
    vals = [finite, improve, bounded]
    bars = ax_stats.bar([0, 1, 2], vals, width=0.62, color=[BLUE, GREEN, ORANGE], alpha=0.9)
    for bar, val in zip(bars, vals):
        ax_stats.text(bar.get_x() + bar.get_width() / 2, val + 2.0, f"{val:.1f}%", ha="center", va="bottom", fontsize=7.2, color=INK)
    ax_stats.set(ylim=(0, 112), ylabel="Fraction of updates (%)", xticks=[0, 1, 2], xticklabels=["Finite\ngrad.", "Score\ngain", "Bounded\nupdate"])
    ax_stats.tick_params(axis="x", labelsize=6.8)
    ax_stats.grid(axis="y", color=GRID, linewidth=0.55)

    ax_heat = fig.add_subplot(sub[0, 2])
    panel_label(ax_heat, "(f)")
    grad = np.load(GRAD_NPZ)["lambda_grad_score"]
    mean_abs = np.mean(np.abs(grad), axis=0).T
    image = ax_heat.imshow(mean_abs, aspect="auto", origin="lower", cmap="YlGnBu", interpolation="nearest")
    ax_heat.set(xlabel="Action horizon", ylabel="Joint dimension", xticks=[0, 3, 7, 11, 15], xticklabels=[1, 4, 8, 12, 16], yticks=np.arange(7), yticklabels=[f"j{i}" for i in range(7)])
    cbar = fig.colorbar(image, ax=ax_heat, fraction=0.048, pad=0.035)
    cbar.set_label(r"Mean $|\lambda\,\partial S/\partial a|$", fontsize=7.2)
    cbar.ax.tick_params(labelsize=6.7)
    ax_heat.text(0.98, 0.96, "$n=128$ windows", transform=ax_heat.transAxes, ha="right", va="top", color=INK, fontsize=7.1, fontweight="bold",
                 bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=1.2))


def build_composite() -> None:
    progress, force, score = load_rollout()
    updates = load_updates()
    stage_progress = [0.03, 0.18, 0.55, 0.90]
    fig = plt.figure(figsize=(7.15, 5.65), facecolor="white")
    grid = fig.add_gridspec(3, 1, height_ratios=[0.88, 1.02, 1.30], hspace=0.31, left=0.07, right=0.985, top=0.965, bottom=0.08)
    rollout_frames(fig, grid[0], stage_progress)
    rollout_signals(fig, grid[1], progress, force, score, stage_progress)
    diagnostic_panels(fig, grid[2], updates)
    fig.savefig(OUT / "board_wiping_guidance_evidence.png", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT / "board_wiping_guidance_evidence.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def build_diagnostics() -> None:
    updates = load_updates()
    fig = plt.figure(figsize=(10.8, 3.25), facecolor="white")
    spec = fig.add_gridspec(1, 1, left=0.07, right=0.985, top=0.91, bottom=0.18)
    diagnostic_panels(fig, spec[0], updates)
    fig.savefig(OUT / "board_guidance_diagnostics_revised.png", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(OUT / "board_guidance_diagnostics_revised.pdf", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> None:
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    build_composite()
    build_diagnostics()
    print(OUT / "board_wiping_guidance_evidence.png")
    print(OUT / "board_guidance_diagnostics_revised.png")


if __name__ == "__main__":
    main()
