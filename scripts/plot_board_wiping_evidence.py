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
GRAD_NPZ = ROOT / "outputs/board_stride3_val_guidance_debug_20260812_n1024/val_action_gradient_tensors.npz"
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


def load_rollout() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    trace = read_numeric_csv(TRIAL / "trace.csv")
    steps = np.arange(len(trace["left_fx"]), dtype=np.float64)
    baseline = slice(0, min(20, len(steps)))
    fx = smooth(trace["left_fx"] - np.nanmedian(trace["left_fx"][baseline]), 13)
    # Contact loading is aligned with the sensor's negative y-axis; report its
    # positive magnitude so increasing contact has the conventional sign.
    fy = smooth(-(trace["left_fy"] - np.nanmedian(trace["left_fy"][baseline])), 13)
    fz = smooth(trace["left_fz"] - np.nanmedian(trace["left_fz"][baseline]), 13)
    score = read_numeric_csv(SCORE_CSV)["score"]
    score = smooth(score, 9)
    if len(score) != len(steps):
        score = np.interp(steps, np.linspace(0, len(steps) - 1, len(score)), score)
    return steps, fx, fy, fz, score


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


def panel_label(ax: plt.Axes, label: str, *, outside: bool = False) -> None:
    y = 1.065 if outside else 0.97
    ax.text(0.01, y, label, transform=ax.transAxes, fontsize=9.2,
            fontweight="bold", color=INK, ha="left",
            va="bottom" if outside else "top", clip_on=False)


def signal_panels(fig: plt.Figure, spec, steps: np.ndarray, fx: np.ndarray, fy: np.ndarray, fz: np.ndarray, score: np.ndarray) -> None:
    sub = spec.subgridspec(2, 1, height_ratios=[1.25, 1.0], hspace=0.08)
    ax_force = fig.add_subplot(sub[0, 0])
    ax_score = fig.add_subplot(sub[1, 0], sharex=ax_force)
    panel_label(ax_force, "(a)")

    phase_edges = [0, 135, 205, 742, len(steps) - 1]
    phase_names = ["Approach", "Contact", "Wiping", "Release"]
    phase_colors = ["#F3F5F7", "#FFF4DF", "#EAF6EF", "#F1F0F7"]
    for idx, (left, right) in enumerate(zip(phase_edges[:-1], phase_edges[1:])):
        for ax in (ax_force, ax_score):
            ax.axvspan(left, right, color=phase_colors[idx], alpha=0.72, linewidth=0, zorder=0)
        ax_force.text((left + right) / 2, 1.025, phase_names[idx], transform=ax_force.get_xaxis_transform(),
                      ha="center", va="bottom", fontsize=7.2, color=INK, fontweight="semibold")

    ax_force.plot(steps, fx, color=BLUE, linewidth=1.0, label=r"$F_x$")
    ax_force.plot(steps, fy, color=ORANGE, linewidth=1.0, label=r"$F_y$ (contact load)")
    ax_force.plot(steps, fz, color=GREEN, linewidth=1.25, label=r"$F_z$")
    ax_force.axhline(0, color=MUTED, linewidth=0.6, alpha=0.65)
    force_lim = np.ceil(max(np.max(np.abs(fx)), np.max(np.abs(fy)), np.max(np.abs(fz))) + 0.5)
    ax_force.set(ylabel="Force (N)", xlim=(0, len(steps) - 1), ylim=(-force_lim, force_lim))
    ax_force.legend(frameon=False, loc="upper right", ncol=3, handlelength=1.6, columnspacing=1.0)
    ax_force.grid(axis="y", color=GRID, linewidth=0.55)
    ax_force.tick_params(labelbottom=False)

    ax_score.plot(steps, score, color="#376F9E", linewidth=1.35, label="Contact-quality score")
    ax_score.fill_between(steps, 0, score, color="#77A9CF", alpha=0.10, linewidth=0)
    ax_score.set(xlabel="Timestep", ylabel="Quality score", xlim=(0, int(steps[-1])), ylim=(0, 1.0))
    ax_score.legend(frameon=False, loc="lower right")
    ax_score.grid(axis="y", color=GRID, linewidth=0.55)


def diagnostic_panels(fig: plt.Figure, spec, updates: dict[str, np.ndarray], labels=("(b)", "(c)", "(d)")) -> None:
    sub = spec.subgridspec(1, 3, width_ratios=[1.05, 0.85, 1.25], wspace=0.35)

    ax_delta = fig.add_subplot(sub[0, 0])
    panel_label(ax_delta, labels[0], outside=True)
    delta = updates["delta"]
    parts = ax_delta.violinplot(delta, positions=[0], widths=0.68, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor(GREEN)
        body.set_edgecolor(GREEN)
        body.set_alpha(0.23)
    rng = np.random.default_rng(12)
    ax_delta.scatter(rng.normal(0, 0.055, len(delta)), delta, s=6.5,
                     color=GREEN, alpha=0.32, edgecolors="none")
    ax_delta.axhline(0, color=MUTED, linestyle="--", linewidth=0.8)
    ax_delta.scatter([0], [np.median(delta)], marker="D", s=24, color=INK, zorder=5, label="Median")
    improve = 100 * np.mean(delta > 0)
    ax_delta.text(0.97, 0.96, f"{improve:.1f}% improve\n$n={len(delta)}$ paired updates", transform=ax_delta.transAxes, ha="right", va="top", color=INK, fontsize=7.4,
                  bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.5))
    ax_delta.text(0.03, 0.04, "All 305 updates shown", transform=ax_delta.transAxes,
                  ha="left", va="bottom", color=MUTED, fontsize=6.5)
    ax_delta.set_yscale("symlog", linthresh=0.02, linscale=0.8)
    ax_delta.set(xlim=(-0.55, 0.55), ylim=(-10, 3), xticks=[0],
                 xticklabels=["Paired guidance\nupdate"],
                 ylabel=r"$\Delta$ expert margin (logit units)")
    ax_delta.grid(axis="y", color=GRID, linewidth=0.55)

    ax_stats = fig.add_subplot(sub[0, 1])
    panel_label(ax_stats, labels[1], outside=True)
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
    panel_label(ax_heat, labels[2], outside=True)
    grad = np.load(GRAD_NPZ)["lambda_grad_score"]
    mean_abs = np.mean(np.abs(grad), axis=0).T
    image = ax_heat.imshow(mean_abs, aspect="auto", origin="lower", cmap="YlGnBu", interpolation="nearest")
    ax_heat.set(xlabel="Action horizon", ylabel="Joint dimension", xticks=[0, 3, 7, 11, 15], xticklabels=[1, 4, 8, 12, 16], yticks=np.arange(7), yticklabels=[f"j{i}" for i in range(7)])
    cbar = fig.colorbar(image, ax=ax_heat, fraction=0.048, pad=0.035)
    cbar.set_label(r"Mean $|\lambda\,\partial S/\partial a|$", fontsize=7.2)
    cbar.ax.tick_params(labelsize=6.7)
    ax_heat.text(0.98, 0.96, "$n=1{,}024$ windows", transform=ax_heat.transAxes, ha="right", va="top", color=INK, fontsize=7.1, fontweight="bold",
                 bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=1.2))


def build_composite() -> None:
    steps, fx, fy, fz, score = load_rollout()
    updates = load_updates()
    fig = plt.figure(figsize=(7.15, 5.15), facecolor="white")
    grid = fig.add_gridspec(2, 1, height_ratios=[1.45, 1.0], hspace=0.38, left=0.075, right=0.985, top=0.95, bottom=0.085)
    signal_panels(fig, grid[0], steps, fx, fy, fz, score)
    diagnostic_panels(fig, grid[1], updates)
    fig.savefig(OUT / "board_wiping_guidance_evidence.png", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT / "board_wiping_guidance_evidence.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def build_diagnostics() -> None:
    updates = load_updates()
    fig = plt.figure(figsize=(10.8, 3.25), facecolor="white")
    spec = fig.add_gridspec(1, 1, left=0.07, right=0.985, top=0.86, bottom=0.18)
    diagnostic_panels(fig, spec[0], updates, labels=("(a)", "(b)", "(c)"))
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
