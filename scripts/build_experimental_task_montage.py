#!/usr/bin/env python3
"""Build a publication-ready platform and five-task overview figure."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFont


CANVAS = (3600, 1450)
BG = (246, 248, 250)
PANEL = (255, 255, 255)
INK = (25, 32, 38)
MUTED = (91, 103, 112)
RULE = (213, 220, 226)
TEAL = (22, 158, 178)
ORANGE = (235, 145, 38)
SOFT_TEAL = (228, 245, 248)
FONT_REGULAR = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FONT_MEDIUM = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc")

TASKS = [
    ("board", "Board Wiping"),
    ("vase", "Vase Wiping"),
    ("card", "Card Swiping"),
    ("chip", "Chip Grasping"),
    ("socket", "Socket Insertion"),
]
PHASES = ["Initial", "Approach", "Contact", "Manipulation", "Result"]


def font(size: int, medium: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_MEDIUM if medium else FONT_REGULAR), size)


def centered_text(
    draw: ImageDraw.ImageDraw,
    center: tuple[float, float],
    text: str,
    text_font: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int],
) -> None:
    bbox = draw.textbbox((0, 0), text, font=text_font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    draw.text((center[0] - width / 2, center[1] - height / 2 - 2), text, font=text_font, fill=fill)


def fit_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Fit an image into a fixed cell without cropping its task content."""
    target_w, target_h = size
    scale = min(target_w / image.width, target_h / image.height)
    resized = image.resize(
        (round(image.width * scale), round(image.height * scale)),
        Image.Resampling.LANCZOS,
    )
    cell = Image.new("RGB", size, PANEL)
    x = (target_w - resized.width) // 2
    y = (target_h - resized.height) // 2
    cell.paste(resized, (x, y))
    return cell


def number_pin(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    number: int,
    color: tuple[int, int, int],
    radius: int = 21,
) -> None:
    x, y = xy
    draw.ellipse(
        (x - radius - 3, y - radius - 3, x + radius + 3, y + radius + 3),
        fill=(255, 255, 255),
    )
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    centered_text(draw, (x, y), str(number), font(25, medium=True), (255, 255, 255))


def draw_setup_panel(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    setup_path: Path,
    box: tuple[int, int, int, int],
) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=10, fill=PANEL, outline=RULE, width=3)
    draw.rounded_rectangle((x0 + 28, y0 + 24, x0 + 93, y0 + 78), radius=8, fill=SOFT_TEAL)
    centered_text(draw, (x0 + 60, y0 + 51), "(a)", font(31, medium=True), TEAL)
    draw.text((x0 + 112, y0 + 20), "Experimental Setup", font=font(44, medium=True), fill=INK)

    image_x, image_y = x0 + 36, y0 + 116
    image_w = x1 - x0 - 72
    image_h = 780
    source = Image.open(setup_path).convert("RGB")
    if source.size != (4096, 3072):
        raise ValueError(f"Expected a 4096x3072 setup image, got {source.size}")
    # A light horizontal crop creates a taller panel without losing the sensors.
    crop_left, crop_right = 335, 3761
    setup = source.crop((crop_left, 0, crop_right, 3072)).resize(
        (image_w, image_h), Image.Resampling.LANCZOS
    )
    canvas.paste(setup, (image_x, image_y))
    draw.rectangle(
        (image_x, image_y, image_x + image_w, image_y + image_h),
        outline=RULE,
        width=2,
    )

    # Pin positions are normalized against the original 4096x3072 photograph.
    source_points = [(3000, 570), (3300, 790), (2180, 232), (840, 2408)]
    pin_colors = [TEAL, TEAL, TEAL, ORANGE]
    for idx, ((sx, sy), pin_color) in enumerate(zip(source_points, pin_colors), start=1):
        px = image_x + round((sx - crop_left) / (crop_right - crop_left) * image_w)
        py = image_y + round(sy / 3072 * image_h)
        number_pin(draw, (px, py), idx, pin_color)

    legend_y = image_y + image_h + 58
    legend_items = [
        (1, "Robot arm", TEAL),
        (2, "Wrist camera", TEAL),
        (3, "Global camera", TEAL),
        (4, "Workspace", ORANGE),
    ]
    for idx, (number, label, item_color) in enumerate(legend_items):
        row = idx // 2
        col = idx % 2
        lx = x0 + 78 + col * 410
        ly = legend_y + row * 86
        number_pin(draw, (lx, ly), number, item_color, radius=18)
        draw.text((lx + 35, ly - 25), label, font=font(36, medium=True), fill=INK)

    note_y = legend_y + 206
    draw.line((x0 + 58, note_y, x1 - 58, note_y), fill=RULE, width=2)
    centered_text(
        draw,
        ((x0 + x1) / 2, note_y + 48),
        "Global and wrist-view sensing",
        font(32),
        MUTED,
    )


def draw_task_grid(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    sequence_dir: Path,
    replacement_dir: Path | None,
    box: tuple[int, int, int, int],
) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=10, fill=PANEL, outline=RULE, width=3)

    label_w = 420
    left_pad = 24
    right_pad = 24
    phase_h = 92
    row_gap = 12
    frame_gap = 13
    frame_w = (x1 - x0 - left_pad - right_pad - label_w - 4 * frame_gap) // 5
    frame_h = frame_w // 2
    rows_h = 5 * frame_h + 4 * row_gap
    top = y0 + phase_h + (y1 - y0 - phase_h - rows_h) // 2
    frames_x = x0 + left_pad + label_w

    for col, phase in enumerate(PHASES):
        cx = frames_x + col * (frame_w + frame_gap) + frame_w / 2
        centered_text(draw, (cx, y0 + 51), phase, font(42, medium=True), MUTED)
        if col < 4:
            arrow_x = frames_x + (col + 1) * frame_w + col * frame_gap + frame_gap / 2
            centered_text(draw, (arrow_x, y0 + 51), "›", font(42, medium=True), (174, 184, 192))

    for row, (task_key, task_name) in enumerate(TASKS):
        row_y = top + row * (frame_h + row_gap)
        if row % 2 == 0:
            draw.rounded_rectangle(
                (x0 + 12, row_y - 4, x1 - 12, row_y + frame_h + 4),
                radius=5,
                fill=(249, 250, 251),
            )

        badge_x = x0 + 48
        badge_y = row_y + frame_h // 2
        draw.rounded_rectangle(
            (badge_x - 3, badge_y - 31, badge_x + 61, badge_y + 31),
            radius=8,
            fill=SOFT_TEAL,
        )
        centered_text(draw, (badge_x + 29, badge_y), f"({chr(ord('b') + row)})", font(35, medium=True), TEAL)
        task_font = font(42, medium=True)
        while draw.textbbox((0, 0), task_name, font=task_font)[2] > label_w - 140:
            task_font = font(task_font.size - 1, medium=True)
        task_bbox = draw.textbbox((0, 0), task_name, font=task_font)
        task_h = task_bbox[3] - task_bbox[1]
        draw.text((badge_x + 80, badge_y - task_h / 2 - 4), task_name, font=task_font, fill=INK)

        for col in range(5):
            path = sequence_dir / f"{task_key}_{col}.jpg"
            if replacement_dir is not None:
                replacement = replacement_dir / path.name
                if replacement.exists():
                    path = replacement
            if not path.exists():
                raise FileNotFoundError(path)
            source_frame = Image.open(path).convert("RGB")
            if task_key == "socket":
                source_frame = source_frame.crop((40, 20, 520, 260))
                source_frame = ImageEnhance.Brightness(source_frame).enhance(1.14)
                source_frame = ImageEnhance.Contrast(source_frame).enhance(1.05)
            frame = fit_image(source_frame, (frame_w, frame_h))
            frame_x = frames_x + col * (frame_w + frame_gap)
            canvas.paste(frame, (frame_x, row_y))
            draw.rectangle(
                (frame_x, row_y, frame_x + frame_w, row_y + frame_h),
                outline=RULE,
                width=2,
            )


def build(
    setup_path: Path,
    sequence_dir: Path,
    replacement_dir: Path | None,
    output_path: Path,
) -> None:
    canvas = Image.new("RGB", CANVAS, BG)
    draw = ImageDraw.Draw(canvas)
    draw_setup_panel(canvas, draw, setup_path, (48, 48, 990, 1402))
    draw_task_grid(canvas, draw, sequence_dir, replacement_dir, (1032, 48, 3552, 1402))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() in {".jpg", ".jpeg"}:
        canvas.save(output_path, quality=97, subsampling=0)
    else:
        canvas.save(output_path, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup-image", type=Path, required=True)
    parser.add_argument("--task-sequence-dir", type=Path, required=True)
    parser.add_argument("--replacement-sequence-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.setup_image, args.task_sequence_dir, args.replacement_sequence_dir, args.output)


if __name__ == "__main__":
    main()
