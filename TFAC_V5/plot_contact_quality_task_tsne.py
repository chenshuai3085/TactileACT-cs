#!/usr/bin/env python3
"""Render a compact contact-quality t-SNE with task marker shapes.

The script consumes the frozen feature archive produced by
visualize_five_task_tacvae_tsne.py. It does not resample data or alter the
cached t-SNE coordinates, so the publication plot remains directly comparable
with the original audit figures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


STATES = [
    "No contact",
    "Stable contact proxy",
    "Contact dropout / insufficient pressure proxy",
    "Excessive pressure proxy",
    "Slip / oscillation proxy",
    "Jamming / bounce proxy",
]
STATE_DISPLAY = {
    "No contact": "No contact",
    "Stable contact proxy": "Stable contact",
    "Contact dropout / insufficient pressure proxy": "Insufficient pressure",
    "Excessive pressure proxy": "Excessive pressure",
    "Slip / oscillation proxy": "Slip / oscillation",
    "Jamming / bounce proxy": "Jamming / bounce",
}
STATE_COLORS = {
    "No contact": "#737B86",
    "Stable contact proxy": "#15936F",
    "Contact dropout / insufficient pressure proxy": "#E5A019",
    "Excessive pressure proxy": "#E05252",
    "Slip / oscillation proxy": "#3478C8",
    "Jamming / bounce proxy": "#7952B3",
}
TASKS = ["board", "vase", "card", "chip", "socket"]
TASK_DISPLAY = {
    "board": "Board wiping",
    "vase": "Vase wiping",
    "card": "Card swiping",
    "chip": "Chip grasping",
    "socket": "Socket insertion",
}
TASK_MARKERS = {"board": "o", "vase": "s", "card": "X", "chip": "v", "socket": "^"}


def _text_array(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind == "S":
        return np.char.decode(values, "utf-8")
    return values.astype(str)


def load_archive(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"latent", "tsne", "labels", "tasks", "episodes"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"Feature archive is missing keys: {sorted(missing)}")
        latent = np.asarray(archive["latent"], dtype=np.float32)
        coords = np.asarray(archive["tsne"], dtype=np.float32)
        labels = _text_array(np.asarray(archive["labels"]))
        tasks = _text_array(np.asarray(archive["tasks"]))
        episodes = _text_array(np.asarray(archive["episodes"]))

    n = len(coords)
    if coords.shape != (n, 2) or latent.shape != (n, 45):
        raise ValueError(f"Unexpected feature shapes: latent={latent.shape}, tsne={coords.shape}")
    if not (len(labels) == len(tasks) == len(episodes) == n):
        raise ValueError("Feature arrays have inconsistent sample counts")
    if not np.isfinite(coords).all() or not np.isfinite(latent).all():
        raise ValueError("Feature archive contains non-finite values")
    unknown_states = sorted(set(labels).difference(STATES))
    unknown_tasks = sorted(set(tasks).difference(TASKS))
    if unknown_states or unknown_tasks:
        raise ValueError(f"Unknown labels: states={unknown_states}, tasks={unknown_tasks}")
    return coords, labels, tasks, episodes


def contingency(labels: np.ndarray, tasks: np.ndarray) -> dict[str, dict[str, int]]:
    return {
        state: {task: int(np.sum((labels == state) & (tasks == task))) for task in TASKS}
        for state in STATES
    }


def plot(coords: np.ndarray, labels: np.ndarray, tasks: np.ndarray, output: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axis = plt.subplots(figsize=(10.8, 7.5))
    fig.subplots_adjust(left=0.07, right=0.76, bottom=0.09, top=0.90)
    axis.set_facecolor("#FAFBFC")

    # Draw the dominant Board samples first so rarer task shapes remain visible.
    for state in STATES:
        for task in TASKS:
            mask = (labels == state) & (tasks == task)
            if not mask.any():
                continue
            axis.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=30 if task != "card" else 34,
                color=STATE_COLORS[state],
                marker=TASK_MARKERS[task],
                alpha=0.76,
                linewidths=0.35,
                edgecolors="white",
                rasterized=True,
                zorder=2 + TASKS.index(task),
            )

    axis.margins(x=0.025, y=0.035)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_xlabel("t-SNE dimension 1", fontsize=9.5)
    axis.set_ylabel("t-SNE dimension 2", fontsize=9.5)
    axis.set_title("TacVAE contact-quality representation", fontsize=14, pad=12)
    axis.text(
        0.0,
        1.006,
        "Color: contact state   |   Marker: task   |   3,000 held-out tactile windows",
        transform=axis.transAxes,
        color="#56606D",
        fontsize=9.2,
        va="bottom",
    )
    for spine in axis.spines.values():
        spine.set_color("#C2C8D0")
        spine.set_linewidth(0.8)

    state_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=STATE_COLORS[state],
            markeredgecolor="white",
            markeredgewidth=0.4,
            markersize=7.8,
            label=STATE_DISPLAY[state],
        )
        for state in STATES
    ]
    state_legend = axis.legend(
        handles=state_handles,
        title="Contact state (proxy)",
        loc="upper left",
        bbox_to_anchor=(1.015, 1.0),
        borderaxespad=0,
        frameon=False,
        labelspacing=0.72,
        handletextpad=0.65,
        fontsize=9.1,
        title_fontsize=9.7,
    )
    axis.add_artist(state_legend)

    task_handles = [
        Line2D(
            [0],
            [0],
            marker=TASK_MARKERS[task],
            linestyle="none",
            markerfacecolor="#343B45",
            markeredgecolor="white",
            markeredgewidth=0.4,
            markersize=7.8,
            label=TASK_DISPLAY[task],
        )
        for task in TASKS
    ]
    axis.legend(
        handles=task_handles,
        title="Task",
        loc="upper left",
        bbox_to_anchor=(1.015, 0.53),
        borderaxespad=0,
        frameon=False,
        labelspacing=0.72,
        handletextpad=0.65,
        fontsize=9.1,
        title_fontsize=9.7,
    )
    fig.text(
        0.765,
        0.095,
        "Proxy labels have incomplete\ntask-state coverage.",
        fontsize=8.3,
        color="#69727E",
        ha="left",
        va="bottom",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("outputs/five_task_tacvae_tsne_20260805/contact_quality_proxy_features.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/figures/tacvae_contact_quality_task_tsne.png"),
    )
    args = parser.parse_args()

    coords, labels, tasks, episodes = load_archive(args.features)
    plot(coords, labels, tasks, args.output)
    metadata = {
        "source_features": str(args.features),
        "source_sha256": hashlib.sha256(args.features.read_bytes()).hexdigest(),
        "coordinates": "cached t-SNE coordinates; no resampling or re-embedding",
        "samples": int(len(coords)),
        "episodes": int(len(set(episodes.tolist()))),
        "state_counts": dict(Counter(labels.tolist())),
        "task_counts": dict(Counter(tasks.tolist())),
        "state_by_task": contingency(labels, tasks),
        "output": str(args.output),
        "caveat": "Proxy labels have incomplete task-state coverage and are task-confounded.",
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
