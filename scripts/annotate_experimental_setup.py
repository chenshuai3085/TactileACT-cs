#!/usr/bin/env python3
"""Create publication-style annotations for the experimental setup photo."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


FONT_MEDIUM = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc")
SENSOR_COLOR = (32, 184, 207, 255)
SCENE_COLOR = (244, 151, 44, 255)
BOX_COLOR = (18, 24, 29, 218)
TEXT_COLOR = (248, 250, 251, 255)


LABELS = {
    "en": {
        "robot": "Robot arm",
        "wrist": "Wrist camera",
        "global": "Global camera",
        "objects": "Task objects",
        "workspace": "Workspace",
    },
    "zh": {
        "robot": "机械臂",
        "wrist": "腕部相机",
        "global": "全局相机",
        "objects": "操作物体",
        "workspace": "工作区",
    },
}


def rounded_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    number: int,
    color: tuple[int, int, int, int],
    font: ImageFont.FreeTypeFont,
) -> tuple[int, int, int, int]:
    x, y = xy
    badge_r = 23
    pad_x, pad_y = 18, 12
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    width = badge_r * 2 + 12 + text_w + pad_x * 2
    height = max(badge_r * 2 + 8, text_h + pad_y * 2)
    box = (x, y, x + width, y + height)

    draw.rounded_rectangle(
        (box[0] + 4, box[1] + 5, box[2] + 4, box[3] + 5),
        radius=10,
        fill=(0, 0, 0, 70),
    )
    draw.rounded_rectangle(box, radius=10, fill=BOX_COLOR, outline=color, width=2)
    cx, cy = x + pad_x + badge_r, y + height // 2
    draw.ellipse((cx - badge_r, cy - badge_r, cx + badge_r, cy + badge_r), fill=color)
    num_font = ImageFont.truetype(str(FONT_MEDIUM), 29)
    num_bbox = draw.textbbox((0, 0), str(number), font=num_font)
    draw.text(
        (cx - (num_bbox[2] - num_bbox[0]) / 2, cy - (num_bbox[3] - num_bbox[1]) / 2 - 3),
        str(number),
        font=num_font,
        fill=(10, 18, 22, 255),
    )
    draw.text((cx + badge_r + 12, y + (height - text_h) / 2 - 3), text, font=font, fill=TEXT_COLOR)
    return box


def leader(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[int, int]],
    color: tuple[int, int, int, int],
) -> None:
    draw.line(points, fill=(0, 0, 0, 125), width=9, joint="curve")
    draw.line(points, fill=color, width=4, joint="curve")
    tx, ty = points[-1]
    draw.ellipse((tx - 10, ty - 10, tx + 10, ty + 10), fill=color, outline=(255, 255, 255, 240), width=3)


def category_badge(draw: ImageDraw.ImageDraw, xy: tuple[int, int], number: int) -> None:
    """Repeat a compact category marker without adding long crossing leaders."""
    cx, cy = xy
    radius = 18
    draw.ellipse(
        (cx - radius, cy - radius, cx + radius, cy + radius),
        fill=SCENE_COLOR,
        outline=(255, 255, 255, 245),
        width=3,
    )
    font = ImageFont.truetype(str(FONT_MEDIUM), 22)
    text = str(number)
    bbox = draw.textbbox((0, 0), text, font=font)
    draw.text(
        (cx - (bbox[2] - bbox[0]) / 2, cy - (bbox[3] - bbox[1]) / 2 - 2),
        text,
        font=font,
        fill=(10, 18, 22, 255),
    )


def annotate(source: Path, destination: Path, language: str) -> None:
    image = Image.open(source).convert("RGBA")
    if image.size != (4096, 3072):
        raise ValueError(f"Expected a 4096x3072 source image, got {image.size}")
    image = image.resize((2048, 1536), Image.Resampling.LANCZOS)

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.truetype(str(FONT_MEDIUM), 34)
    labels = LABELS[language]

    # A restrained outline defines the actual black manipulation surface.
    workspace = [(5, 313), (1250, 322), (1246, 1348), (5, 1364)]
    draw.line(workspace + [workspace[0]], fill=(0, 0, 0, 120), width=10, joint="curve")
    draw.line(workspace + [workspace[0]], fill=SCENE_COLOR, width=4, joint="curve")

    robot_box = rounded_label(draw, (1430, 34), labels["robot"], 1, SENSOR_COLOR, font)
    wrist_box = rounded_label(draw, (1580, 501), labels["wrist"], 2, SENSOR_COLOR, font)
    global_box = rounded_label(draw, (775, 214), labels["global"], 3, SENSOR_COLOR, font)
    objects_box = rounded_label(draw, (24, 371), labels["objects"], 4, SCENE_COLOR, font)
    workspace_box = rounded_label(draw, (24, 1221), labels["workspace"], 5, SCENE_COLOR, font)

    leader(draw, [(robot_box[0] + 35, robot_box[3]), (1435, 150), (1500, 285)], SENSOR_COLOR)
    leader(draw, [(wrist_box[0], (wrist_box[1] + wrist_box[3]) // 2), (1600, 515), (1650, 395)], SENSOR_COLOR)
    leader(draw, [(global_box[2] - 45, global_box[1]), (1075, 174), (1090, 116)], SENSOR_COLOR)
    leader(draw, [(objects_box[2], (objects_box[1] + objects_box[3]) // 2), (330, 411), (405, 509)], SCENE_COLOR)
    leader(draw, [(workspace_box[2], (workspace_box[1] + workspace_box[3]) // 2), (325, 1251), (420, 1204)], SCENE_COLOR)

    # The task set is distributed across the scene; repeated category markers
    # identify representative groups without a web of crossing leader lines.
    for point in [(590, 132), (655, 980), (1140, 835), (820, 1445)]:
        category_badge(draw, point, 4)

    result = Image.alpha_composite(image, overlay).convert("RGB")
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.save(destination, quality=96, subsampling=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    stem = "experimental_setup_annotated"
    annotate(args.source, args.output_dir / f"{stem}_en.jpg", "en")
    annotate(args.source, args.output_dir / f"{stem}_zh.jpg", "zh")


if __name__ == "__main__":
    main()
