#!/usr/bin/env python3
"""Render a compact contact-quality t-SNE with task marker shapes.

The script consumes the frozen feature archive produced by
visualize_five_task_tacvae_tsne.py. It does not resample data. By default it
renders the original cached perplexity-50 coordinates; ``--reembed`` enables
the label-agnostic sensitivity embedding.
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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


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
    "board": "Board",
    "vase": "Vase",
    "card": "Card",
    "chip": "Chip",
    "socket": "Socket",
}
TASK_MARKERS = {"board": "o", "vase": "s", "card": "X", "chip": "v", "socket": "^"}


def _text_array(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind == "S":
        return np.char.decode(values, "utf-8")
    return values.astype(str)


def load_archive(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    return latent, coords, labels, tasks, episodes


def embed(
    latent: np.ndarray,
    perplexity: float,
    early_exaggeration: float,
    iterations: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    standardized = StandardScaler().fit_transform(latent)
    pca = PCA(n_components=min(30, latent.shape[1]), random_state=seed)
    pca_features = pca.fit_transform(standardized)
    coords = TSNE(
        n_components=2,
        perplexity=min(perplexity, (len(latent) - 1) / 3),
        early_exaggeration=early_exaggeration,
        init="pca",
        learning_rate="auto",
        n_iter=iterations,
        random_state=seed,
    ).fit_transform(pca_features)
    return coords.astype(np.float32), pca_features.astype(np.float32)


def neighbor_purity(coords: np.ndarray, values: np.ndarray, k: int = 10) -> float:
    model = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    indices = model.kneighbors(coords, return_distance=False)[:, 1:]
    return float(np.mean(values[indices] == values[:, None]))


def grid_occupancy(coords: np.ndarray, bins: int = 30) -> float:
    normalized = (coords - coords.min(axis=0)) / np.maximum(np.ptp(coords, axis=0), 1e-8)
    cells = np.minimum((normalized * bins).astype(np.int64), bins - 1)
    return float(len(set(map(tuple, cells.tolist()))) / (bins * bins))


def contract_islands(coords: np.ndarray, seed: int = 42, clusters: int = 28, factor: float = 0.82) -> np.ndarray:
    """Contract unsupervised t-SNE island centroids for a presentation-only view."""
    if not 0 < factor <= 1:
        raise ValueError("factor must be in (0, 1]")
    n_clusters = min(clusters, max(2, len(coords) // 20))
    model = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed)
    assignments = model.fit_predict(coords)
    global_center = coords.mean(axis=0)
    contracted_centers = global_center + factor * (model.cluster_centers_ - global_center)
    return contracted_centers[assignments] + factor * (coords - model.cluster_centers_[assignments])


def contingency(labels: np.ndarray, tasks: np.ndarray) -> dict[str, dict[str, int]]:
    return {
        state: {task: int(np.sum((labels == state) & (tasks == task))) for task in TASKS}
        for state in STATES
    }


def plot(coords: np.ndarray, labels: np.ndarray, tasks: np.ndarray, output: Path) -> int:
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
    fig, axis = plt.subplots(figsize=(8.6, 7.2))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.085, top=0.90)
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
                s=28 if task != "card" else 32,
                color=STATE_COLORS[state],
                marker=TASK_MARKERS[task],
                alpha=0.82,
                linewidths=0.3,
                edgecolors="white",
                rasterized=True,
                zorder=2 + TASKS.index(task),
            )

    axis.margins(x=0.012, y=0.015)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_xlabel("t-SNE dimension 1", fontsize=9.5)
    axis.set_ylabel("t-SNE dimension 2", fontsize=9.5)
    axis.set_title("TacVAE contact-quality representation", fontsize=13, pad=11)
    axis.text(
        0.0,
        1.006,
        f"Color: contact state   |   Marker: task   |   {len(coords):,} held-out tactile windows",
        transform=axis.transAxes,
        color="#56606D",
        fontsize=8.7,
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
            markersize=6.5,
            label=STATE_DISPLAY[state],
        )
        for state in STATES
    ]
    task_handles = [
        Line2D(
            [0],
            [0],
            marker=TASK_MARKERS[task],
            linestyle="none",
            markerfacecolor="#343B45",
            markeredgecolor="white",
            markeredgewidth=0.4,
            markersize=6.5,
            label=TASK_DISPLAY[task],
        )
        for task in TASKS
    ]
    section_handle = Line2D([0], [0], linestyle="none", marker=None, color="none")
    handles = [section_handle, *state_handles, section_handle, *task_handles]
    legend = axis.legend(
        handles=handles,
        labels=[
            "CONTACT STATE (PROXY)",
            *[STATE_DISPLAY[state] for state in STATES],
            "TASK",
            *[TASK_DISPLAY[task] for task in TASKS],
        ],
        loc="lower right",
        bbox_to_anchor=(0.988, 0.018),
        ncol=2,
        columnspacing=1.0,
        borderpad=0.58,
        labelspacing=0.43,
        handlelength=0.9,
        handletextpad=0.38,
        fontsize=7.1,
        frameon=True,
        fancybox=False,
        framealpha=0.94,
        facecolor="white",
        edgecolor="#D3D9E1",
    )
    legend.get_frame().set_linewidth(0.75)
    legend_texts = legend.get_texts()
    legend_texts[0].set_fontweight("bold")
    legend_texts[len(STATES) + 1].set_fontweight("bold")

    fig.canvas.draw()
    legend_bbox = legend.get_window_extent(fig.canvas.get_renderer()).transformed(axis.transData.inverted())
    legend_overlap = int(
        np.sum(
            (coords[:, 0] >= legend_bbox.x0)
            & (coords[:, 0] <= legend_bbox.x1)
            & (coords[:, 1] >= legend_bbox.y0)
            & (coords[:, 1] <= legend_bbox.y1)
        )
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return legend_overlap


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
    parser.add_argument("--perplexity", type=float, default=150.0)
    parser.add_argument("--early-exaggeration", type=float, default=6.0)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--presentation-output",
        type=Path,
        default=Path("paper/figures/tacvae_contact_quality_task_tsne_presentation.png"),
    )
    parser.add_argument("--chip-only-no-contact", action="store_true")
    parser.add_argument("--contract-islands", action="store_true")
    parser.add_argument(
        "--reembed",
        action="store_true",
        help="Recompute the sensitivity embedding instead of using cached p=50 coordinates.",
    )
    args = parser.parse_args()

    latent, cached_coords, labels, tasks, episodes = load_archive(args.features)
    use_cached_coordinates = not args.reembed
    if use_cached_coordinates:
        coords = cached_coords
        pca_features = StandardScaler().fit_transform(latent)
        coordinate_protocol = "cached original coordinates"
    else:
        coords, pca_features = embed(
            latent,
            perplexity=args.perplexity,
            early_exaggeration=args.early_exaggeration,
            iterations=args.iterations,
            seed=args.seed,
        )
        coordinate_protocol = (
            f"StandardScaler -> PCA-30 -> t-SNE(perplexity={args.perplexity:g}, "
            f"early_exaggeration={args.early_exaggeration:g}, seed={args.seed})"
        )
    legend_overlap = plot(coords, labels, tasks, args.output)
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        tsne=coords,
        labels=labels,
        tasks=tasks,
        episodes=episodes,
    )
    metadata = {
        "source_features": str(args.features),
        "source_sha256": hashlib.sha256(args.features.read_bytes()).hexdigest(),
        "coordinates": coordinate_protocol,
        "sampling": "same fixed held-out samples; no resampling",
        "perplexity": None if use_cached_coordinates else args.perplexity,
        "early_exaggeration": None if use_cached_coordinates else args.early_exaggeration,
        "iterations": None if use_cached_coordinates else args.iterations,
        "seed": args.seed,
        "samples": int(len(coords)),
        "episodes": int(len(set(episodes.tolist()))),
        "state_counts": dict(Counter(labels.tolist())),
        "task_counts": dict(Counter(tasks.tolist())),
        "state_by_task": contingency(labels, tasks),
        "grid_occupancy_30x30": grid_occupancy(coords),
        "trustworthiness_k10": float(trustworthiness(pca_features, coords, n_neighbors=10)),
        "tsne_neighbor_purity_k10": {
            "state": neighbor_purity(coords, labels),
            "task": neighbor_purity(coords, tasks),
        },
        "point_centers_under_legend": legend_overlap,
        "output": str(args.output),
        "caveat": "Proxy labels have incomplete task-state coverage and are task-confounded.",
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if args.chip_only_no_contact or args.contract_islands:
        display_mask = np.ones(len(coords), dtype=bool)
        if args.chip_only_no_contact:
            display_mask &= (labels != "No contact") | (tasks == "chip")
        display_coords = coords[display_mask]
        display_labels = labels[display_mask]
        display_tasks = tasks[display_mask]
        if args.contract_islands:
            display_coords = contract_islands(display_coords, seed=args.seed)
        presentation_overlap = plot(
            display_coords,
            display_labels,
            display_tasks,
            args.presentation_output,
        )
        np.savez_compressed(
            args.presentation_output.with_suffix(".npz"),
            tsne=display_coords,
            labels=display_labels,
            tasks=display_tasks,
            episodes=episodes[display_mask],
        )
        presentation_metadata = {
            "source_features": str(args.features),
            "source_coordinates": coordinate_protocol,
            "sampling": "same fixed held-out samples, with no-contact display restricted to Chip",
            "display_filter": "No contact retains only task=chip; all other states unchanged",
            "island_contract": "unsupervised KMeans centroid contraction, factor=0.82, n_clusters=28"
            if args.contract_islands
            else "none",
            "samples": int(len(display_coords)),
            "state_counts": dict(Counter(display_labels.tolist())),
            "task_counts": dict(Counter(display_tasks.tolist())),
            "point_centers_under_legend": presentation_overlap,
            "presentation_only": True,
            "caveat": "Coordinates are visually contracted and must not be used for geometric claims.",
        }
        args.presentation_output.with_suffix(".json").write_text(
            json.dumps(presentation_metadata, indent=2), encoding="utf-8"
        )
        print(json.dumps(presentation_metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
