#!/usr/bin/env python3
"""Export synchronized potato-chip observations and Figure 2 visual assets."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EPISODE = Path(
    "/home/chenshuai/Desktop/260709_v8j_jiashupian/jiashupian_0709/episode_1.hdf5"
)
EXECUTION_VIDEO = ROOT / "tmp_vtm_ProjectPage/static/videos/chip_real_raw.mp4"
OUTPUT = ROOT / "paper/figures/figure2_chip_assets"

CAMERA_FRAMES = [110, 130, 160, 200, 280, 360]
MARKER_FRAMES = [
    0, 23, 46, 70, 93, 117, 140, 163, 187, 210,
    234, 257, 281, 304, 327, 351, 374, 398, 421, 445,
]
EXECUTION_TIMES = [4.0, 6.0, 10.0, 12.0]
REFERENCE_FRAMES = 20
TACTILE_SIDE = "left"
ARROW_COLOR = (59, 212, 255)  # BGR: RDP-style yellow.
DISPLAY_ARROW_SCALE = 45.0


def ensure_rgb(image: np.ndarray) -> np.ndarray:
    """HDF5 camera streams are stored as RGB."""
    return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR)


def tactile_bgr(image: np.ndarray) -> np.ndarray:
    """This dataset's tactile stream is stored in OpenCV BGR order."""
    return image.astype(np.uint8).copy()


def detect_marker_centers(reference_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.0)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=13,
        param1=60,
        param2=9,
        minRadius=3,
        maxRadius=7,
    )
    if circles is None or len(circles[0]) < 70:
        found = 0 if circles is None else len(circles[0])
        raise RuntimeError(f"Too few reference marker circles: {found}")
    points = circles[0, :, :2].astype(np.float32)

    def cluster_axis(values: np.ndarray) -> np.ndarray:
        data = values.reshape(-1, 1).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.01)
        _, _, centers = cv2.kmeans(data, 9, None, criteria, 20, cv2.KMEANS_PP_CENTERS)
        return np.sort(centers[:, 0])

    xs = cluster_axis(points[:, 0])
    ys = cluster_axis(points[:, 1])
    if np.any(np.diff(xs) < 10) or np.any(np.diff(ys) < 10):
        raise RuntimeError("Detected marker grid axes are not consistently separated")
    grid_x, grid_y = np.meshgrid(xs, ys)
    return np.stack([grid_x, grid_y], axis=-1).astype(np.float32)


def draw_marker_overlay(
    tactile: np.ndarray,
    centers: np.ndarray,
    displacement: np.ndarray,
    arrow_scale: float,
) -> np.ndarray:
    panel = tactile.copy()
    for row in range(9):
        for col in range(9):
            start = tuple(np.rint(centers[row, col]).astype(int))
            delta = displacement[row, col] * arrow_scale
            end_xy = centers[row, col] + delta
            end_xy[0] = np.clip(end_xy[0], 0, panel.shape[1] - 1)
            end_xy[1] = np.clip(end_xy[1], 0, panel.shape[0] - 1)
            end = tuple(np.rint(end_xy).astype(int))
            if np.linalg.norm(delta) >= 2.0:
                cv2.arrowedLine(panel, start, end, (35, 35, 35), 3, cv2.LINE_AA, tipLength=0.34)
                cv2.arrowedLine(panel, start, end, ARROW_COLOR, 1, cv2.LINE_AA, tipLength=0.34)
    return panel


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 2]):
        raise RuntimeError(f"Failed to write {path}")


def make_contact_sheet(paths: list[Path], output: Path, columns: int, cell: int = 280) -> None:
    rows = (len(paths) + columns - 1) // columns
    sheet = np.full((rows * cell, columns * cell, 3), 255, dtype=np.uint8)
    for index, path in enumerate(paths):
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"Could not read {path}")
        scale = min((cell - 34) / image.shape[1], (cell - 34) / image.shape[0])
        resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        row, col = divmod(index, columns)
        x = col * cell + (cell - resized.shape[1]) // 2
        y = row * cell + 8
        sheet[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
        cv2.putText(
            sheet,
            path.stem,
            (col * cell + 10, (row + 1) * cell - 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (35, 42, 54),
            1,
            cv2.LINE_AA,
        )
    write_image(output, sheet)


def export_execution_frames() -> list[dict[str, object]]:
    capture = cv2.VideoCapture(str(EXECUTION_VIDEO))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {EXECUTION_VIDEO}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    records: list[dict[str, object]] = []
    for seconds in EXECUTION_TIMES:
        frame_index = int(round(seconds * fps))
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Could not read execution frame {frame_index}")
        output = OUTPUT / "execution" / f"chip_execution_t{seconds:04.1f}s_f{frame_index:04d}.png"
        write_image(output, frame)
        clean = frame[72:690, 100:1180]
        clean_output = OUTPUT / "execution_clean" / f"chip_execution_clean_t{seconds:04.1f}s_f{frame_index:04d}.png"
        write_image(clean_output, clean)
        records.append(
            {
                "seconds": seconds,
                "frame": frame_index,
                "file": str(output.relative_to(ROOT)),
                "clean_file": str(clean_output.relative_to(ROOT)),
            }
        )
    capture.release()
    return records


def main() -> None:
    if not EPISODE.exists():
        raise FileNotFoundError(EPISODE)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    camera_outputs: list[Path] = []
    marker_outputs: list[Path] = []

    with h5py.File(EPISODE, "r") as h5:
        global_frames = h5["observations/images/global"]
        wrist_frames = h5["observations/images/wrist"]
        tactile_frames = h5[f"observations/tac/{TACTILE_SIDE}/img"]
        marker_offsets = h5[f"observations/tac/{TACTILE_SIDE}/marker_offset"]
        reference_image = tactile_bgr(np.mean(tactile_frames[:REFERENCE_FRAMES], axis=0).astype(np.uint8))
        reference_marker = np.mean(marker_offsets[:REFERENCE_FRAMES], axis=0).astype(np.float32)
        centers = detect_marker_centers(reference_image)

        selected_displacements = np.asarray(marker_offsets[MARKER_FRAMES], dtype=np.float32) - reference_marker
        global_max = float(np.percentile(np.linalg.norm(selected_displacements, axis=-1), 99))
        arrow_scale = DISPLAY_ARROW_SCALE

        for frame in CAMERA_FRAMES:
            for camera_name, dataset in (("global", global_frames), ("wrist", wrist_frames)):
                output = OUTPUT / "camera_pairs" / f"chip_ep01_f{frame:04d}_{camera_name}.png"
                write_image(output, ensure_rgb(dataset[frame]))
                camera_outputs.append(output)
                records.append(
                    {"kind": camera_name, "episode": 1, "frame": frame, "file": str(output.relative_to(ROOT))}
                )

        for frame, displacement in zip(MARKER_FRAMES, selected_displacements):
            image = tactile_bgr(tactile_frames[frame])
            overlay = draw_marker_overlay(image, centers, displacement, arrow_scale)
            output = OUTPUT / "marker_rdp" / f"chip_ep01_f{frame:04d}_{TACTILE_SIDE}_rdp.png"
            write_image(output, overlay)
            marker_outputs.append(output)
            magnitude = np.linalg.norm(displacement, axis=-1)
            records.append(
                {
                    "kind": "marker_rdp",
                    "episode": 1,
                    "frame": frame,
                    "side": TACTILE_SIDE,
                    "mean_displacement_px": float(magnitude.mean()),
                    "max_displacement_px": float(magnitude.max()),
                    "file": str(output.relative_to(ROOT)),
                }
            )

    execution_records = export_execution_frames()
    make_contact_sheet(camera_outputs, OUTPUT / "camera_pairs_contact_sheet.png", columns=4, cell=300)
    make_contact_sheet(marker_outputs, OUTPUT / "marker_rdp_20_contact_sheet.png", columns=5, cell=270)
    execution_paths = [ROOT / item["file"] for item in execution_records]
    make_contact_sheet(execution_paths, OUTPUT / "execution_contact_sheet.png", columns=4, cell=340)
    clean_execution_paths = [ROOT / item["clean_file"] for item in execution_records]
    make_contact_sheet(clean_execution_paths, OUTPUT / "execution_clean_contact_sheet.png", columns=4, cell=340)

    metadata = {
        "source_episode": str(EPISODE),
        "camera_frames": CAMERA_FRAMES,
        "marker_frames": MARKER_FRAMES,
        "marker_side": TACTILE_SIDE,
        "marker_reference": f"mean of frames 0:{REFERENCE_FRAMES}",
        "marker_arrow_scale": arrow_scale,
        "marker_arrow_scale_policy": "one shared display magnification across all 20 images; not literal pixel scale",
        "marker_selected_p99_magnitude": global_max,
        "execution_video": str(EXECUTION_VIDEO),
        "execution_frames": execution_records,
        "assets": records,
    }
    (OUTPUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    with (OUTPUT / "asset_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["kind", "episode", "frame", "file"])
        writer.writeheader()
        for item in records:
            writer.writerow({key: item.get(key, "") for key in writer.fieldnames})
    print(json.dumps({"output": str(OUTPUT), "files": len(records) + len(execution_records)}, indent=2))


if __name__ == "__main__":
    main()
