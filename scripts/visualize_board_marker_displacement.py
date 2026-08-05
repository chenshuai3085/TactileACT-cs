#!/usr/bin/env python3
"""Render Fig. 3-style tactile marker displacement fields for board wiping."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper" / "figures" / "board_marker_displacement"


@dataclass(frozen=True)
class PanelSpec:
    key: str
    title: str
    path: str
    frame: int
    reference_frame: int = 50


NORMAL = "/media/chenshuai/EXTERNAL_USB/pih_dataset/260609/wipe_pos_straight_z124_125_150_20260609/episode_24.hdf5"

SEQUENCE = [
    PanelSpec("approach", "Approach / no contact", NORMAL, 50),
    PanelSpec("initial_contact", "Initial contact", NORMAL, 164),
    PanelSpec("stable_wipe", "Stable wiping", NORMAL, 475),
    PanelSpec("peak_shear", "Peak shear", NORMAL, 588),
]

MODES = [
    PanelSpec(
        "insufficient_pressure",
        "Insufficient pressure",
        "/media/chenshuai/EXTERNAL_USB/pih_dataset/260609/z_too_high/episode_30.hdf5",
        637,
    ),
    PanelSpec("stable_contact", "Stable wiping", NORMAL, 475),
    PanelSpec(
        "excessive_pressure",
        "Excessive pressure",
        "/media/chenshuai/EXTERNAL_USB/pih_dataset/260610/z_too_low/episode_19.hdf5",
        530,
    ),
    PanelSpec(
        "oscillation",
        "Slip / oscillation proxy",
        "/media/chenshuai/EXTERNAL_USB/pih_dataset/260610/z_too_oscillate/episode_21.hdf5",
        642,
    ),
]


def load_frame(spec: PanelSpec, side: str = "left") -> dict[str, np.ndarray | float]:
    with h5py.File(spec.path, "r") as h5:
        image = h5[f"observations/tac/{side}/img"][spec.frame].astype(np.uint8)
        reference_image = h5[f"observations/tac/{side}/img"][spec.reference_frame].astype(np.uint8)
        marker = h5[f"observations/tac/{side}/marker_offset"]
        displacement = marker[spec.frame].astype(np.float32) - marker[spec.reference_frame].astype(np.float32)
        force = h5[f"observations/tac/{side}/force6d"][spec.frame].astype(np.float32)
        if spec.frame > 0:
            temporal_delta = marker[spec.frame].astype(np.float32) - marker[spec.frame - 1].astype(np.float32)
        else:
            temporal_delta = np.zeros_like(displacement)
    return {
        "image": image,
        "reference_image": reference_image,
        "displacement": displacement,
        "force": force,
        "temporal_delta": temporal_delta,
    }


def detect_reference_centers(image: np.ndarray) -> tuple[np.ndarray, int]:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    best: tuple[int, np.ndarray, int] | None = None
    for threshold in range(45, 111, 5):
        mask = (gray < threshold).astype(np.uint8)
        count, _, stats, centers = cv2.connectedComponentsWithStats(mask, connectivity=8)
        points = np.asarray(
            [
                centers[index]
                for index in range(1, count)
                if 18 <= stats[index, cv2.CC_STAT_AREA] <= 160
                and 5 < centers[index, 0] < image.shape[1] - 5
                and 5 < centers[index, 1] < image.shape[0] - 15
            ],
            dtype=np.float32,
        )
        distance = abs(len(points) - 81)
        if best is None or distance < best[0]:
            best = (distance, points, threshold)
        if len(points) == 81:
            break
    if best is None or len(best[1]) != 81:
        found = 0 if best is None else len(best[1])
        raise RuntimeError(f"Expected 81 marker centers, found {found}")
    points = best[1][np.argsort(best[1][:, 1])].reshape(9, 9, 2)
    points = np.stack([row[np.argsort(row[:, 0])] for row in points])
    return points, best[2]


def draw_overlay(axis: plt.Axes, payload: dict[str, np.ndarray | float], centers: np.ndarray) -> None:
    image = np.asarray(payload["image"])
    displacement = np.asarray(payload["displacement"])
    axis.imshow(image)
    axis.imshow(np.zeros(image.shape[:2], dtype=np.float32), cmap="gray", vmin=0, vmax=1, alpha=0.08)
    axis.scatter(
        centers[..., 0],
        centers[..., 1],
        s=7,
        facecolors="none",
        edgecolors="white",
        linewidths=0.45,
        alpha=0.62,
        zorder=3,
    )
    axis.quiver(
        centers[..., 0],
        centers[..., 1],
        displacement[..., 0],
        displacement[..., 1],
        color="#FFD43B",
        edgecolor="#332600",
        linewidth=0.38,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.0065,
        headwidth=3.8,
        headlength=4.8,
        headaxislength=4.2,
        pivot="tail",
        zorder=4,
    )
    axis.set_xlim(0, image.shape[1])
    axis.set_ylim(image.shape[0], 0)
    axis.set_aspect("equal")
    axis.axis("off")


def metrics(payload: dict[str, np.ndarray | float]) -> dict[str, float]:
    displacement = np.asarray(payload["displacement"])
    temporal_delta = np.asarray(payload["temporal_delta"])
    force = np.asarray(payload["force"])
    magnitude = np.linalg.norm(displacement, axis=-1)
    return {
        "mean_displacement_px": float(magnitude.mean()),
        "max_displacement_px": float(magnitude.max()),
        "mean_dx_px": float(displacement[..., 0].mean()),
        "mean_dy_px": float(displacement[..., 1].mean()),
        "frame_to_frame_change_px": float(np.linalg.norm(temporal_delta, axis=-1).mean()),
        "force_xyz_norm": float(np.linalg.norm(force[:3])),
    }


def render_individual(spec: PanelSpec, payload: dict[str, np.ndarray | float], centers: np.ndarray, prefix: str) -> str:
    output = OUT_DIR / "panels" / f"{prefix}_{spec.key}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(4.0, 4.0), dpi=400)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    draw_overlay(axis, payload, centers)
    fig.savefig(output, dpi=400, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return str(output.relative_to(ROOT))


def render_composite(specs: list[PanelSpec], name: str, heading: str) -> tuple[list[dict[str, object]], Path]:
    payloads = [load_frame(spec) for spec in specs]
    centers_and_thresholds = [detect_reference_centers(np.asarray(payload["reference_image"])) for payload in payloads]
    fig, axes = plt.subplots(1, len(specs), figsize=(14.0, 3.7), constrained_layout=True)
    records = []
    for index, (axis, spec, payload, center_result) in enumerate(
        zip(axes, specs, payloads, centers_and_thresholds), start=1
    ):
        centers, threshold = center_result
        draw_overlay(axis, payload, centers)
        panel_metrics = metrics(payload)
        axis.set_title(
            f"({chr(96 + index)}) {spec.title}\n"
            f"mean {panel_metrics['mean_displacement_px']:.2f} px | max {panel_metrics['max_displacement_px']:.2f} px",
            fontsize=10,
            fontweight="bold",
            pad=8,
        )
        individual = render_individual(spec, payload, centers, name)
        records.append(
            {
                "key": spec.key,
                "title": spec.title,
                "episode": spec.path,
                "frame": spec.frame,
                "reference_frame": spec.reference_frame,
                "marker_detection_threshold": threshold,
                "metrics": panel_metrics,
                "individual_png": individual,
            }
        )
    fig.suptitle(heading, fontsize=14, fontweight="bold")
    output = OUT_DIR / f"{name}.png"
    fig.savefig(output, dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return records, output


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sequence_records, sequence_output = render_composite(
        SEQUENCE,
        "board_marker_wiping_sequence",
        "Marker displacement during a board-wiping trajectory",
    )
    mode_records, mode_output = render_composite(
        MODES,
        "board_marker_contact_modes",
        "Marker displacement under different board-contact modes",
    )
    metadata = {
        "reference": "Xue et al. (2025), Reactive Diffusion Policy, Fig. 3 visual style",
        "representation": "left tactile 9x9 marker displacement relative to a no-contact reference frame",
        "arrow_scale": "1 image pixel per marker displacement pixel; no vector magnification",
        "arrow_color": "yellow",
        "sequence_figure": str(sequence_output.relative_to(ROOT)),
        "mode_figure": str(mode_output.relative_to(ROOT)),
        "sequence": sequence_records,
        "modes": mode_records,
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
