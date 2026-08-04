#!/usr/bin/env python3
"""Audit the five-task TacVAE representation on the untouched test split.

Two analyses are produced from frozen 45D encoder means:

1. A task-balanced contact-phase audit shared by all five tasks.
2. A six-state contact-quality proxy audit based on explicit collection
   conditions. Proxy labels are kept separate from ground-truth claims.

Every coloring of an analysis reuses exactly the same samples and t-SNE
coordinates. Sampling is episode-capped to limit temporal autocorrelation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import balanced_accuracy_score, f1_score, silhouette_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TFAC_V5.tac_quality_energy.audit_board_contact_windows import detect_contact  # noqa: E402
from TFAC_V5.tactile_vae import TactileVAE  # noqa: E402
from TFAC_V5.train_multitask_tacvae_5task import DEFAULT_TASKS, load_manifest  # noqa: E402


TASKS = list(DEFAULT_TASKS)
PHASES = ["Pre-contact", "Contact onset", "Sustained contact"]
QUALITY_STATES = [
    "No contact",
    "Stable contact proxy",
    "Insufficient pressure proxy",
    "Excessive pressure proxy",
    "Oscillation proxy",
    "Jamming / bounce proxy",
]

TASK_MARKERS = {"board": "o", "vase": "s", "card": "X", "chip": "v", "socket": "^"}
TASK_COLORS = {
    "board": "#277DA1",
    "vase": "#43AA8B",
    "card": "#F8961E",
    "chip": "#9C4DCC",
    "socket": "#D1495B",
}
PHASE_COLORS = {
    "Pre-contact": "#7A7F87",
    "Contact onset": "#E6A117",
    "Sustained contact": "#178A55",
}
QUALITY_COLORS = {
    "No contact": "#7A7F87",
    "Stable contact proxy": "#178A55",
    "Insufficient pressure proxy": "#E6A117",
    "Excessive pressure proxy": "#D64545",
    "Oscillation proxy": "#3676C8",
    "Jamming / bounce proxy": "#7B4AB5",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_episode(path: str, side: str) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as h5:
        marker = h5[f"observations/tac/{side}/marker_offset"][()].astype(np.float32)
        force_key = f"observations/tac/{side}/force6d"
        force = h5[force_key][()].astype(np.float32) if force_key in h5 else h5["ft"][()].astype(np.float32)
    n = min(len(marker), len(force))
    return marker[:n], force[:n]


def detect(marker: np.ndarray, force: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    return detect_contact(
        marker,
        force,
        smooth_radius=args.smooth_radius,
        min_contact_len=args.min_contact_len,
        marker_quantile=0.65,
        force_quantile=0.65,
        marker_scale=6.0,
        force_scale=6.0,
        marker_floor=0.05,
        force_floor=0.05,
    )


def choose(values: Iterable[int], cap: int, rng: np.random.Generator) -> list[int]:
    unique = np.asarray(sorted(set(int(value) for value in values)), dtype=np.int64)
    if len(unique) <= cap:
        return unique.tolist()
    return sorted(rng.choice(unique, size=cap, replace=False).astype(int).tolist())


def make_row(
    record: Any,
    marker: np.ndarray,
    force: np.ndarray,
    det: dict[str, Any],
    end: int,
    label: str,
    analysis: str,
    label_source: str,
    window: int,
) -> dict[str, Any]:
    start = end - window + 1
    sequence = marker[start : end + 1]
    marker_mag = np.linalg.norm(sequence, axis=-1).mean(axis=(1, 2))
    force_mag = np.linalg.norm(force[start : end + 1, :3], axis=1)
    return {
        "path": record.path,
        "episode": record.episode_id,
        "source": record.episode_id.split("/", 1)[0],
        "task": record.task,
        "condition": record.condition,
        "analysis": analysis,
        "label": label,
        "label_source": label_source,
        "window_start": int(start),
        "window_end": int(end),
        "marker_mag_mean": float(marker_mag.mean()),
        "marker_dynamic_mean": float(np.abs(np.diff(marker_mag)).mean()),
        "force_mag_mean": float(force_mag.mean()),
        "contact_detection_ok": bool(det["detection_ok"]),
        "sequence": sequence.astype(np.float32),
    }


def phase_candidates(
    record: Any,
    marker: np.ndarray,
    force: np.ndarray,
    det: dict[str, Any],
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    n = len(marker)
    start = int(det["contact_start"])
    end = int(det["contact_end"])
    window = args.window

    pre = []
    for window_end in range(window - 1, max(window - 1, start - window), args.phase_stride):
        lo = window_end - window + 1
        marker_low = np.all(det["marker_mag"][lo : window_end + 1] < det["marker_threshold"])
        force_low = np.all(det["force_mag"][lo : window_end + 1] < det["force_threshold"])
        if marker_low and force_low:
            pre.append(window_end)

    onset_start = max(window - 1, start - args.onset_pre_frames)
    onset_end = min(n, end, start + args.onset_post_frames)
    onset = range(onset_start, onset_end, args.phase_stride)

    contact_len = max(0, end - start)
    sustained_start = start + max(window, int(0.25 * contact_len))
    sustained_end = start + int(0.75 * contact_len)
    sustained = range(max(window - 1, sustained_start + window - 1), sustained_end, args.phase_stride)

    rows = []
    for label, indices in zip(PHASES, (pre, onset, sustained)):
        for window_end in choose(indices, args.max_per_episode_phase, rng):
            rows.append(
                make_row(record, marker, force, det, window_end, label, "phase", "signal-derived", window)
            )
    return rows


def quality_condition(record: Any) -> str | None:
    condition = record.condition.lower()
    if record.task == "board":
        mapping = {
            "normal": QUALITY_STATES[1],
            "success": QUALITY_STATES[1],
            "unlabeled_positive": QUALITY_STATES[1],
            "too_high": QUALITY_STATES[2],
            "too_low": QUALITY_STATES[3],
            "oscillate": QUALITY_STATES[4],
        }
        return mapping.get(condition)
    if record.task == "vase":
        return QUALITY_STATES[1] if condition == "normal" else None
    if record.task == "card":
        if condition in {"success", "replay_positive", "second_insert"}:
            return QUALITY_STATES[1]
        if condition in {"bounce", "horizontal_bounce", "alignment_bounce", "collision"}:
            return QUALITY_STATES[5]
        return None
    if record.task == "chip":
        return QUALITY_STATES[1] if condition == "unlabeled_grasp" else None
    if record.task == "socket":
        if "mixed" in condition:
            return None
        if "bounce" in condition:
            return QUALITY_STATES[5]
        if "success" in condition:
            return QUALITY_STATES[1]
    return None


def quality_candidates(
    record: Any,
    marker: np.ndarray,
    force: np.ndarray,
    det: dict[str, Any],
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    window = args.window
    contact_start = int(det["contact_start"])
    contact_end = int(det["contact_end"])
    rows = []

    pre = []
    for window_end in range(window - 1, max(window - 1, contact_start - window), args.quality_stride):
        lo = window_end - window + 1
        marker_low = np.all(det["marker_mag"][lo : window_end + 1] < det["marker_threshold"])
        force_low = np.all(det["force_mag"][lo : window_end + 1] < det["force_threshold"])
        if marker_low and force_low:
            pre.append(window_end)
    for window_end in choose(pre, args.max_per_episode_quality, rng):
        rows.append(
            make_row(
                record,
                marker,
                force,
                det,
                window_end,
                QUALITY_STATES[0],
                "quality",
                "conservative-pre-contact-proxy",
                window,
            )
        )

    label = quality_condition(record)
    if label is None:
        return rows
    central_start = contact_start
    central_end = contact_end
    contact_indices = range(
        max(window - 1, central_start + window - 1), central_end, args.quality_stride
    )
    for window_end in choose(contact_indices, args.max_per_episode_quality, rng):
        rows.append(
            make_row(
                record,
                marker,
                force,
                det,
                window_end,
                label,
                "quality",
                f"condition:{record.condition}+detected-contact-proxy",
                window,
            )
        )
    return rows


def balance_strata(
    rows: list[dict[str, Any]], fields: tuple[str, ...], target: int, seed: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rng = np.random.default_rng(seed)
    buckets: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(str(row[field]) for field in fields)].append(row)
    counts = {"|".join(key): len(value) for key, value in sorted(buckets.items())}
    missing = []
    expected = [tuple([task, phase]) for task in TASKS for phase in PHASES] if fields == ("task", "label") else []
    for key in expected:
        if key not in buckets:
            missing.append("|".join(key))
    if missing:
        raise RuntimeError(f"Missing sampling strata: {missing}")
    if expected:
        target = min(target, min(len(buckets[key]) for key in expected))
        selected_keys = expected
    else:
        selected_keys = list(buckets)
    selected = []
    for key in selected_keys:
        values = buckets[key]
        take = min(target, len(values))
        indices = rng.choice(len(values), size=take, replace=False)
        selected.extend(values[int(index)] for index in indices)
    return selected, counts


def balance_quality(
    rows: list[dict[str, Any]], target: int, seed: int, min_per_present_task: int = 30
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rng = np.random.default_rng(seed)
    buckets = {label: [row for row in rows if row["label"] == label] for label in QUALITY_STATES}
    counts = {label: len(values) for label, values in buckets.items()}
    missing = [label for label, values in buckets.items() if not values]
    if missing:
        raise RuntimeError(f"Missing quality states: {missing}")
    take = min(target, min(counts.values()))
    selected = []
    for label in QUALITY_STATES:
        values = buckets[label]
        chosen: set[int] = set()
        by_task: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(values):
            by_task[row["task"]].append(index)
        present_tasks = [task for task in TASKS if by_task[task]]
        quota = min(min_per_present_task, take // max(1, len(present_tasks)))
        for task in present_tasks:
            task_indices = by_task[task]
            task_take = min(quota, len(task_indices))
            chosen.update(rng.choice(task_indices, size=task_take, replace=False).astype(int).tolist())
        remaining = [index for index in range(len(values)) if index not in chosen]
        fill = take - len(chosen)
        if fill:
            chosen.update(rng.choice(remaining, size=fill, replace=False).astype(int).tolist())
        selected.extend(values[index] for index in sorted(chosen))
    return selected, counts


def load_model(args: argparse.Namespace, expected_manifest_hash: str) -> tuple[Any, np.ndarray, np.ndarray, torch.device]:
    device = torch.device(args.device)
    # This is a locally produced trusted checkpoint; weights_only cannot load
    # the saved NumPy RNG state that supports exact training resume.
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("manifest_hash") != expected_manifest_hash:
        raise RuntimeError(
            f"Checkpoint/manifest mismatch: {checkpoint.get('manifest_hash')} != {expected_manifest_hash}"
        )
    signature = checkpoint["signature"]
    model = TactileVAE(
        latent_dim=int(signature["latent_channels"]),
        temporal_window=int(signature["temporal_window"]),
        num_freqs=4,
        inr_hidden=int(signature["inr_hidden"]),
        kl_weight=float(signature["kl_weight"]),
        direction_weight=0.0,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    stats = checkpoint["norm_stats"]
    mean = np.asarray(stats["mean"], dtype=np.float32).reshape(1, 1, 1, 2)
    std = np.asarray(stats["std"], dtype=np.float32).reshape(1, 1, 1, 2)
    return model, mean, std, device


def encode(rows: list[dict[str, Any]], args: argparse.Namespace, manifest_hash: str) -> np.ndarray:
    model, mean, std, device = load_model(args, manifest_hash)
    sequences = np.stack([row["sequence"] for row in rows]).astype(np.float32)
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(sequences), args.batch_size):
            normalized = (sequences[start : start + args.batch_size] - mean) / std
            _, mu_last = model.encode_single_frame(torch.from_numpy(normalized).to(device))
            chunks.append(mu_last.flatten(1).cpu().numpy())
    latent = np.concatenate(chunks).astype(np.float32)
    if latent.shape != (len(rows), 45) or not np.isfinite(latent).all():
        raise RuntimeError(f"Invalid latent array: shape={latent.shape}, finite={np.isfinite(latent).all()}")
    return latent


def embed(latent: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, float]:
    standardized = StandardScaler().fit_transform(latent)
    pca_dim = min(args.pca_dim, standardized.shape[1], len(standardized) - 1)
    pca = PCA(n_components=pca_dim, random_state=args.seed)
    pca_features = pca.fit_transform(standardized)
    coords = TSNE(
        n_components=2,
        perplexity=min(args.perplexity, (len(latent) - 1) / 3),
        init="pca",
        learning_rate="auto",
        n_iter=args.tsne_iterations,
        random_state=args.seed,
    ).fit_transform(pca_features)
    return coords.astype(np.float32), pca_features.astype(np.float32), float(pca.explained_variance_ratio_.sum())


def style_axis(axis: plt.Axes) -> None:
    axis.set_xlabel("t-SNE dimension 1", fontsize=9)
    axis.set_ylabel("t-SNE dimension 2", fontsize=9)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color("#B8BEC7")
        spine.set_linewidth(0.8)


def legend_handles(names: list[str], colors: dict[str, str], counts: Counter[str]) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=colors[name],
            markeredgecolor="white",
            markersize=7,
            label=f"{name} (n={counts[name]})",
        )
        for name in names
    ]


def plot_dual(
    rows: list[dict[str, Any]],
    coords: np.ndarray,
    label_order: list[str],
    label_colors: dict[str, str],
    title: str,
    output: Path,
) -> None:
    labels = np.asarray([row["label"] for row in rows])
    tasks = np.asarray([row["task"] for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(15.4, 6.6), constrained_layout=True)
    for label in label_order:
        for task in TASKS:
            mask = (labels == label) & (tasks == task)
            if mask.any():
                axes[0].scatter(
                    coords[mask, 0],
                    coords[mask, 1],
                    s=22,
                    c=label_colors[label],
                    marker=TASK_MARKERS[task],
                    alpha=0.62,
                    linewidths=0.25,
                    edgecolors="white",
                    rasterized=True,
                )
    axes[0].set_title("(a) Color = contact label, marker = task", loc="left", fontsize=11, fontweight="bold")
    axes[0].legend(
        handles=legend_handles(label_order, label_colors, Counter(labels.tolist())),
        loc="best",
        fontsize=7.7,
        framealpha=0.94,
        title="Contact label",
        title_fontsize=8.5,
    )
    style_axis(axes[0])

    for task in TASKS:
        mask = tasks == task
        axes[1].scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=22,
            c=TASK_COLORS[task],
            marker=TASK_MARKERS[task],
            alpha=0.62,
            linewidths=0.25,
            edgecolors="white",
            rasterized=True,
        )
    task_counts = Counter(tasks.tolist())
    handles = [
        Line2D(
            [0],
            [0],
            marker=TASK_MARKERS[task],
            linestyle="none",
            markerfacecolor=TASK_COLORS[task],
            markeredgecolor="white",
            markersize=7,
            label=f"{task.title()} (n={task_counts[task]})",
        )
        for task in TASKS
    ]
    axes[1].set_title("(b) Same coordinates, color = task", loc="left", fontsize=11, fontweight="bold")
    axes[1].legend(handles=handles, loc="best", fontsize=8, framealpha=0.94, title="Task/domain")
    style_axis(axes[1])
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.patch.set_facecolor("white")
    fig.savefig(output, dpi=260, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_facets(
    rows: list[dict[str, Any]],
    coords: np.ndarray,
    label_order: list[str],
    label_colors: dict[str, str],
    title: str,
    output: Path,
) -> None:
    labels = np.asarray([row["label"] for row in rows])
    tasks = np.asarray([row["task"] for row in rows])
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.6), constrained_layout=True)
    for axis, task in zip(axes.flat, TASKS):
        task_mask = tasks == task
        for label in label_order:
            mask = task_mask & (labels == label)
            if mask.any():
                axis.scatter(
                    coords[mask, 0],
                    coords[mask, 1],
                    s=24,
                    c=label_colors[label],
                    marker=TASK_MARKERS[task],
                    alpha=0.66,
                    linewidths=0.25,
                    edgecolors="white",
                    rasterized=True,
                )
        axis.set_title(f"{task.title()} (n={int(task_mask.sum())})", fontsize=11, fontweight="bold")
        style_axis(axis)
    axes.flat[-1].axis("off")
    counts = Counter(labels.tolist())
    axes.flat[-1].legend(
        handles=legend_handles(label_order, label_colors, counts),
        loc="center",
        fontsize=8.2,
        framealpha=0.94,
        title="Contact label",
    )
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.savefig(output, dpi=260, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def grouped_probe(latent: np.ndarray, rows: list[dict[str, Any]], field: str, seed: int) -> dict[str, Any]:
    names = sorted({str(row[field]) for row in rows})
    y = np.asarray([names.index(str(row[field])) for row in rows], dtype=np.int64)
    groups = np.asarray([row["episode"] for row in rows])
    groups_per_class = [len(set(groups[y == index])) for index in range(len(names))]
    n_splits = min(5, min(groups_per_class))
    if n_splits < 2:
        return {"status": "insufficient episode groups", "classes": names}
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    balanced_accuracy = []
    macro_f1 = []
    for train_index, test_index in splitter.split(latent, y, groups):
        if len(np.unique(y[train_index])) != len(names) or len(np.unique(y[test_index])) != len(names):
            continue
        scaler = StandardScaler().fit(latent[train_index])
        classifier = LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=3000, random_state=seed
        ).fit(scaler.transform(latent[train_index]), y[train_index])
        prediction = classifier.predict(scaler.transform(latent[test_index]))
        balanced_accuracy.append(float(balanced_accuracy_score(y[test_index], prediction)))
        macro_f1.append(float(f1_score(y[test_index], prediction, average="macro", zero_division=0)))
    return {
        "protocol": f"{n_splits}-fold StratifiedGroupKFold by episode",
        "classes": names,
        "valid_folds": len(balanced_accuracy),
        "balanced_accuracy_mean": float(np.mean(balanced_accuracy)) if balanced_accuracy else None,
        "balanced_accuracy_std": float(np.std(balanced_accuracy)) if balanced_accuracy else None,
        "macro_f1_mean": float(np.mean(macro_f1)) if macro_f1 else None,
        "macro_f1_std": float(np.std(macro_f1)) if macro_f1 else None,
    }


def cramers_v(rows: list[dict[str, Any]]) -> float:
    labels = sorted({row["label"] for row in rows})
    table = np.asarray(
        [[sum(row["label"] == label and row["task"] == task for row in rows) for task in TASKS] for label in labels],
        dtype=np.float64,
    )
    total = table.sum()
    expected = table.sum(axis=1, keepdims=True) @ table.sum(axis=0, keepdims=True) / total
    valid = expected > 0
    chi2 = float(np.sum(np.square(table - expected)[valid] / expected[valid]))
    return math.sqrt(chi2 / total / max(1, min(table.shape[0] - 1, table.shape[1] - 1)))


def neighbor_purity(features: np.ndarray, rows: list[dict[str, Any]], field: str, k: int = 10) -> dict[str, float]:
    values = np.asarray([row[field] for row in rows])
    neighbors = NearestNeighbors(n_neighbors=min(k + 1, len(features))).fit(features)
    indices = neighbors.kneighbors(features, return_distance=False)[:, 1:]
    observed = float(np.mean(values[indices] == values[:, None]))
    counts = np.asarray(list(Counter(values.tolist()).values()), dtype=np.float64)
    random_baseline = float(np.sum(counts * (counts - 1)) / (len(values) * max(1, len(values) - 1)))
    return {"k": int(indices.shape[1]), "observed": observed, "random_baseline": random_baseline}


def analyze(
    name: str,
    rows: list[dict[str, Any]],
    latent: np.ndarray,
    label_order: list[str],
    label_colors: dict[str, str],
    title: str,
    args: argparse.Namespace,
    out_dir: Path,
) -> dict[str, Any]:
    coords, pca_features, pca_variance = embed(latent, args)
    plot_dual(rows, coords, label_order, label_colors, title, out_dir / f"{name}_dual.png")
    plot_facets(rows, coords, label_order, label_colors, title, out_dir / f"{name}_task_facets.png")
    labels = np.asarray([label_order.index(row["label"]) for row in rows])
    task_ids = np.asarray([TASKS.index(row["task"]) for row in rows])
    standardized = StandardScaler().fit_transform(latent)
    state_task_counts = {
        label: {task: sum(row["label"] == label and row["task"] == task for row in rows) for task in TASKS}
        for label in label_order
    }
    np.savez_compressed(
        out_dir / f"{name}_features.npz",
        latent=latent,
        pca=pca_features,
        tsne=coords,
        labels=np.asarray([row["label"] for row in rows]),
        tasks=np.asarray([row["task"] for row in rows]),
        episodes=np.asarray([row["episode"] for row in rows]),
    )
    return {
        "n_samples": len(rows),
        "episodes": len({row["episode"] for row in rows}),
        "label_counts": dict(Counter(row["label"] for row in rows)),
        "task_counts": dict(Counter(row["task"] for row in rows)),
        "label_by_task": state_task_counts,
        "label_task_cramers_v": cramers_v(rows),
        "silhouette_45d_by_label": float(silhouette_score(standardized, labels)),
        "silhouette_45d_by_task": float(silhouette_score(standardized, task_ids)),
        "silhouette_tsne_by_label": float(silhouette_score(coords, labels)),
        "silhouette_tsne_by_task": float(silhouette_score(coords, task_ids)),
        "pca_explained_variance_sum": pca_variance,
        "episode_grouped_label_probe": grouped_probe(latent, rows, "label", args.seed),
        "episode_grouped_task_probe": grouped_probe(latent, rows, "task", args.seed),
        "neighbor_purity_45d": {
            field: neighbor_purity(standardized, rows, field) for field in ("label", "task", "source", "episode")
        },
        "neighbor_purity_tsne": {
            field: neighbor_purity(coords, rows, field) for field in ("label", "task", "source", "episode")
        },
        "dual_figure": str(out_dir / f"{name}_dual.png"),
        "facet_figure": str(out_dir / f"{name}_task_facets.png"),
    }


def write_manifest(rows: list[dict[str, Any]], output: Path) -> None:
    fields = [
        "analysis",
        "path",
        "episode",
        "source",
        "task",
        "condition",
        "window_start",
        "window_end",
        "label",
        "label_source",
        "marker_mag_mean",
        "marker_dynamic_mean",
        "force_mag_mean",
        "contact_detection_ok",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="outputs/multitask_tacvae_5task_20260803/manifest.jsonl")
    parser.add_argument("--checkpoint", default="outputs/multitask_tacvae_5task_20260804/train/best_macro.pt")
    parser.add_argument("--out-dir", default="outputs/five_task_tacvae_tsne_20260805")
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--side", default="left", choices=("left", "right"))
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--phase-stride", type=int, default=2)
    parser.add_argument("--quality-stride", type=int, default=2)
    parser.add_argument("--onset-pre-frames", type=int, default=16)
    parser.add_argument("--onset-post-frames", type=int, default=112)
    parser.add_argument("--max-per-episode-phase", type=int, default=80)
    parser.add_argument("--max-per-episode-quality", type=int, default=128)
    parser.add_argument("--samples-per-task-phase", type=int, default=160)
    parser.add_argument("--samples-per-quality", type=int, default=500)
    parser.add_argument("--smooth-radius", type=int, default=4)
    parser.add_argument("--min-contact-len", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--pca-dim", type=int, default=30)
    parser.add_argument("--perplexity", type=float, default=50.0)
    parser.add_argument("--tsne-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.window != 8:
        raise ValueError("The trained TacVAE requires window=8")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    set_seed(args.seed)
    records, manifest_hash = load_manifest(Path(args.manifest), DEFAULT_TASKS)
    records = [record for record in records if record.split == args.split]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    phase_pool = []
    quality_pool = []
    detection_counts = Counter()
    for index, record in enumerate(records, start=1):
        marker, force = load_episode(record.path, args.side)
        det = detect(marker, force, args)
        detection_counts[f"{record.task}:{'ok' if det['detection_ok'] else 'fallback'}"] += 1
        phase_pool.extend(phase_candidates(record, marker, force, det, args, rng))
        quality_pool.extend(quality_candidates(record, marker, force, det, args, rng))
        if index % 100 == 0:
            print(f"sampled {index}/{len(records)} episodes", flush=True)

    phase_rows, phase_candidates_by_stratum = balance_strata(
        phase_pool, ("task", "label"), args.samples_per_task_phase, args.seed
    )
    quality_rows, quality_candidate_counts = balance_quality(
        quality_pool, args.samples_per_quality, args.seed
    )
    all_rows = phase_rows + quality_rows
    latent = encode(all_rows, args, manifest_hash)
    phase_latent = latent[: len(phase_rows)]
    quality_latent = latent[len(phase_rows) :]
    write_manifest(all_rows, out_dir / "heldout_samples_manifest.csv")

    summary = {
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": int(
            torch.load(args.checkpoint, map_location="cpu", weights_only=False)["completed_epoch"]
        ),
        "manifest": args.manifest,
        "manifest_hash": manifest_hash,
        "split": args.split,
        "representation": "deterministic encoder mu_last, 5x3x3 flattened to 45D",
        "sampling_args": vars(args),
        "contact_detection_counts": dict(detection_counts),
        "phase_candidate_counts": phase_candidates_by_stratum,
        "quality_candidate_counts": quality_candidate_counts,
        "phase": analyze(
            "contact_phase",
            phase_rows,
            phase_latent,
            PHASES,
            PHASE_COLORS,
            "Five-task TacVAE on untouched test episodes: contact phase",
            args,
            out_dir,
        ),
        "quality_proxy": analyze(
            "contact_quality_proxy",
            quality_rows,
            quality_latent,
            QUALITY_STATES,
            QUALITY_COLORS,
            "Five-task TacVAE on untouched test episodes: contact-quality proxies",
            args,
            out_dir,
        ),
        "caveats": [
            "Only the test split is used; it was not used for training or checkpoint selection.",
            "Pre-contact is a conservative marker-and-force rule, not manual ground truth.",
            "Stable, insufficient, excessive, oscillation, and jamming/bounce are collection-condition plus detected-contact proxies.",
            "Insufficient, excessive, and oscillation are available only for Board; quality labels are therefore task-confounded.",
            "Chip has only four test episodes and no crush/slip annotation.",
            "Phase and quality windows use stride 2; adjacent windows overlap by six of eight frames.",
            "t-SNE is descriptive; 45D silhouette and episode-grouped probes are reported separately.",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
