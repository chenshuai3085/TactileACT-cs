#!/usr/bin/env python3
"""Review a task video and extract publication-ready phase frames."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont


FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc")


def video_info(path: Path) -> tuple[float, int, int, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if fps <= 0 or frames <= 0:
        raise RuntimeError(f"Invalid video metadata: fps={fps}, frames={frames}")
    return fps, frames, width, height


def read_frame(path: Path, timestamp: float) -> Image.Image:
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Cannot read {path} at {timestamp:.2f}s")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)


def crop_task_frame(image: Image.Image, output_size: tuple[int, int] = (560, 280)) -> Image.Image:
    """Center-crop 16:9 video to the 2:1 ratio used by the paper task grid."""
    target_ratio = output_size[0] / output_size[1]
    source_ratio = image.width / image.height
    if source_ratio < target_ratio:
        crop_h = round(image.width / target_ratio)
        top = (image.height - crop_h) // 2
        image = image.crop((0, top, image.width, top + crop_h))
    else:
        crop_w = round(image.height * target_ratio)
        left = (image.width - crop_w) // 2
        image = image.crop((left, 0, left + crop_w, image.height))
    return image.resize(output_size, Image.Resampling.LANCZOS)


def make_contact_sheet(video: Path, output: Path, samples: int) -> None:
    fps, frame_count, _, _ = video_info(video)
    duration = (frame_count - 1) / fps
    timestamps = [duration * idx / (samples - 1) for idx in range(samples)]
    cell_w, cell_h = 426, 240
    caption_h = 42
    columns = 4
    rows = (samples + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_w, rows * (cell_h + caption_h)), (242, 244, 246))
    draw = ImageDraw.Draw(sheet)
    label_font = ImageFont.truetype(str(FONT), 25)

    for idx, timestamp in enumerate(timestamps):
        frame = read_frame(video, timestamp).resize((cell_w, cell_h), Image.Resampling.LANCZOS)
        col, row = idx % columns, idx // columns
        x, y = col * cell_w, row * (cell_h + caption_h)
        sheet.paste(frame, (x, y))
        draw.text((x + 12, y + cell_h + 5), f"{idx + 1:02d}  {timestamp:05.1f}s", font=label_font, fill=(28, 35, 41))

    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, optimize=True)


def extract_frames(video: Path, output_dir: Path, prefix: str, timestamps: list[float]) -> None:
    if len(timestamps) != 5:
        raise ValueError("Exactly five timestamps are required")
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, timestamp in enumerate(timestamps):
        frame = crop_task_frame(read_frame(video, timestamp))
        frame.save(output_dir / f"{prefix}_{idx}.jpg", quality=97, subsampling=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    review = subparsers.add_parser("review")
    review.add_argument("--video", type=Path, required=True)
    review.add_argument("--output", type=Path, required=True)
    review.add_argument("--samples", type=int, default=12)

    extract = subparsers.add_parser("extract")
    extract.add_argument("--video", type=Path, required=True)
    extract.add_argument("--output-dir", type=Path, required=True)
    extract.add_argument("--prefix", required=True)
    extract.add_argument("--timestamps", type=float, nargs=5, required=True)

    args = parser.parse_args()
    if args.command == "review":
        make_contact_sheet(args.video, args.output, args.samples)
    else:
        extract_frames(args.video, args.output_dir, args.prefix, args.timestamps)


if __name__ == "__main__":
    main()
