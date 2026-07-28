from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


OUT_DIR = Path(__file__).resolve().parent / "figures"


COLORS = {
    "obs": "#2F5E8E",
    "policy": "#546A7B",
    "foresight": "#D9792D",
    "score": "#2F8F5B",
    "guide": "#B23A48",
    "exec": "#3A3A3A",
    "light": "#F7F8FA",
    "edge": "#263238",
    "muted": "#667085",
}


def rounded_box(ax, xy, width, height, fc, title, body, ec="#263238", lw=1.2):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.055",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height * 0.68,
        title,
        ha="center",
        va="center",
        fontsize=10.5,
        weight="bold",
        color="white",
        zorder=4,
    )
    ax.text(
        x + width / 2,
        y + height * 0.34,
        body,
        ha="center",
        va="center",
        fontsize=8.2,
        color="white",
        linespacing=1.25,
        zorder=4,
    )
    return patch


def arrow(ax, start, end, color="#263238", lw=1.35, style="-", rad=0.0, mutation_scale=11):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=lw,
        linestyle=style,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        zorder=2,
    )
    ax.add_patch(arr)
    return arr


def small_tactile_strip(ax, x, y, w, h):
    for i, label in enumerate(["z+1", "z+4", "z+8", "z+16"]):
        xi = x + i * (w * 0.24)
        rect = Rectangle((xi, y), w * 0.18, h, facecolor="#FFF3E6", edgecolor="#D9792D", lw=0.8, zorder=4)
        ax.add_patch(rect)
        for gx in range(3):
            for gy in range(3):
                ax.plot(
                    xi + 0.025 + gx * (w * 0.045),
                    y + 0.025 + gy * (h * 0.26),
                    marker="o",
                    markersize=1.5,
                    color="#D9792D",
                    zorder=5,
                )
        ax.text(xi + w * 0.09, y - 0.055, label, ha="center", va="top", fontsize=6.5, color=COLORS["muted"])


def draw():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12.2, 4.25))
    ax.set_xlim(0, 12.2)
    ax.set_ylim(0, 4.25)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Main inference chain.
    y = 2.35
    bw, bh = 1.62, 0.86
    xs = [0.45, 2.35, 4.25, 6.15, 8.05, 9.95]
    rounded_box(ax, (xs[0], y), bw, bh, COLORS["obs"], "Observation", "RGB views\nqpos + tactile history")
    rounded_box(ax, (xs[1], y), bw, bh, COLORS["policy"], "Base Policy", "DP / RDT / pi0.5\ncandidate chunk a")
    rounded_box(ax, (xs[2], y), bw, bh, COLORS["foresight"], "Tactile Foresight", "action-conditioned\nfuture tactile latents")
    rounded_box(ax, (xs[3], y), bw, bh, COLORS["score"], "Contact Energy", "expert margin vs.\nfailure modes")
    rounded_box(ax, (xs[4], y), bw, bh, COLORS["guide"], "Trust-region Guide", "bounded late-step\naction update")
    rounded_box(ax, (xs[5], y), bw, bh, COLORS["exec"], "Robot Execution", "refined action\nchunk")

    for i in range(len(xs) - 1):
        arrow(ax, (xs[i] + bw + 0.05, y + bh / 2), (xs[i + 1] - 0.08, y + bh / 2))

    ax.text(3.16, y + bh + 0.22, "behavior prior", ha="center", fontsize=8, color=COLORS["muted"])
    ax.text(5.05, y + bh + 0.22, "predicted contact consequence", ha="center", fontsize=8, color=COLORS["muted"])
    ax.text(7.02, y + bh + 0.22, "contact-quality score", ha="center", fontsize=8, color=COLORS["muted"])
    ax.text(8.96, y + bh + 0.22, "small correction", ha="center", fontsize=8, color=COLORS["muted"])

    # Gradient loop from scorer/guidance back to denoising action sample.
    arrow(
        ax,
        (8.05, y + 0.12),
        (3.05, y + 0.12),
        color=COLORS["guide"],
        lw=1.4,
        style=(0, (4, 3)),
        rad=-0.30,
        mutation_scale=12,
    )
    ax.text(
        5.55,
        1.92,
        "backpropagate d score / d action in late denoising",
        ha="center",
        va="center",
        fontsize=8.2,
        color=COLORS["guide"],
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.92),
        zorder=6,
    )

    # Future tactile strip.
    small_tactile_strip(ax, 4.42, 1.35, 1.25, 0.34)
    ax.text(5.04, 1.14, "decoded future marker fields", ha="center", fontsize=7.2, color=COLORS["muted"])

    # Failure classes.
    fail_box = FancyBboxPatch(
        (6.28, 1.15),
        1.36,
        0.52,
        boxstyle="round,pad=0.015,rounding_size=0.04",
        linewidth=0.8,
        edgecolor="#2F8F5B",
        facecolor="#EAF6EF",
        zorder=3,
    )
    ax.add_patch(fail_box)
    ax.text(
        6.96,
        1.41,
        "failure modes:\nloss / overload / unstable / slip / jam",
        ha="center",
        va="center",
        fontsize=6.9,
        color="#1F6B43",
        linespacing=1.15,
        zorder=4,
    )

    # Training supervision lane.
    lane = FancyBboxPatch(
        (0.45, 0.25),
        10.95,
        0.58,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=0.9,
        edgecolor="#D0D5DD",
        facecolor="#F7F8FA",
        zorder=1,
    )
    ax.add_patch(lane)
    ax.text(0.72, 0.54, "Training signals", ha="left", va="center", fontsize=8.5, weight="bold", color=COLORS["edge"])
    ax.text(
        3.05,
        0.54,
        "demonstration action chunks",
        ha="center",
        va="center",
        fontsize=7.5,
        color=COLORS["muted"],
    )
    ax.text(
        4.75,
        0.54,
        "future TacVAE latent targets",
        ha="center",
        va="center",
        fontsize=7.5,
        color=COLORS["muted"],
    )
    ax.text(
        7.35,
        0.54,
        "semantic contact labels",
        ha="center",
        va="center",
        fontsize=7.5,
        color=COLORS["muted"],
    )
    ax.text(
        9.9,
        0.54,
        "trust-region constraints",
        ha="center",
        va="center",
        fontsize=7.5,
        color=COLORS["muted"],
    )

    # Light training links.
    arrow(ax, (3.05, 0.83), (3.16, y - 0.08), color="#A0A7B2", lw=0.8, style=(0, (2, 3)), mutation_scale=8)
    arrow(ax, (4.75, 0.83), (5.06, y - 0.08), color="#A0A7B2", lw=0.8, style=(0, (2, 3)), mutation_scale=8)
    arrow(ax, (7.35, 0.83), (6.96, y - 0.08), color="#A0A7B2", lw=0.8, style=(0, (2, 3)), mutation_scale=8)
    arrow(ax, (9.9, 0.83), (8.86, y - 0.08), color="#A0A7B2", lw=0.8, style=(0, (2, 3)), mutation_scale=8)

    ax.text(
        0.45,
        4.02,
        "ForeTac overview: predict tactile consequences, score contact quality, and guide action locally",
        ha="left",
        va="top",
        fontsize=12.5,
        weight="bold",
        color=COLORS["edge"],
    )

    fig.tight_layout(pad=0.2)
    fig.savefig(OUT_DIR / "foretac_overview.png", dpi=300, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(OUT_DIR / "foretac_overview.pdf", bbox_inches="tight", pad_inches=0.06)


if __name__ == "__main__":
    draw()
