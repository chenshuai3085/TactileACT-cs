#!/usr/bin/env python3
"""Build continuous multi-step tactile-future assets for ForeTac Figure 2."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "paper/figures/foretac_multitask_qpos_preview.npz"
OUT = ROOT / "paper/figures/foretac_figure2_ppt_assets"

INK = "#0F172A"
TEAL = "#0F766E"
BORDER = "#CBD5E1"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = Path(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    )
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def load_board_prediction() -> np.ndarray:
    data = np.load(SOURCE)
    tasks = [str(value) for value in data["task"]]
    board_idx = tasks.index("board")
    prediction = data["pred"][board_idx]
    if prediction.shape != (16, 9, 9, 2):
        raise ValueError(f"Expected continuous (16, 9, 9, 2), got {prediction.shape}")
    return prediction


def render_frame(marker: np.ndarray, norm: Normalize, output: Path) -> Image.Image:
    rows, cols = marker.shape[:2]
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    magnitude = np.linalg.norm(marker, axis=-1)
    fig, ax = plt.subplots(figsize=(1.6, 1.6), dpi=220)
    ax.imshow(magnitude, origin="lower", cmap="viridis", norm=norm,
              interpolation="nearest", extent=(-0.5, cols - 0.5, -0.5, rows - 0.5))
    ax.quiver(x, y, marker[..., 0], marker[..., 1], color="white",
              angles="xy", scale_units="xy", scale=10.0, width=0.010,
              headwidth=3.7, headlength=4.5, alpha=0.90)
    ax.set(xlim=(-0.5, cols - 0.5), ylim=(-0.5, rows - 0.5))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(BORDER)
        spine.set_linewidth(0.9)
    fig.subplots_adjust(0.01, 0.01, 0.99, 0.99)
    fig.savefig(output, dpi=220, facecolor="white")
    plt.close(fig)
    return Image.open(output).convert("RGB")


def render_frames(prediction: np.ndarray) -> list[Image.Image]:
    magnitude = np.linalg.norm(prediction, axis=-1)
    norm = Normalize(vmin=0.0, vmax=float(np.percentile(magnitude, 99.5)))
    frames = []
    for idx, marker in enumerate(prediction, start=1):
        output = OUT / f"foresight_pred_tplus{idx:02d}.png"
        frames.append(render_frame(marker, norm, output))
    return frames


def draw_arrow(draw: ImageDraw.ImageDraw, x0: int, x1: int, y: int) -> None:
    draw.line((x0, y, x1 - 9, y), fill=TEAL, width=4)
    draw.polygon([(x1, y), (x1 - 12, y - 7), (x1 - 12, y + 7)], fill=TEAL)


def build_row(frames: list[Image.Image], indices: list[int], card: int = 216) -> Image.Image:
    gap = 44
    margin_x = 28
    top = 58
    width = 2 * margin_x + len(indices) * card + (len(indices) - 1) * gap
    height = top + card + 20
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    label_font = font(25, bold=True)
    for slot, idx in enumerate(indices):
        x = margin_x + slot * (card + gap)
        frame = frames[idx - 1].resize((card, card), Image.Resampling.LANCZOS)
        canvas.paste(frame, (x, top))
        label = f"t+{idx}"
        box = draw.textbbox((0, 0), label, font=label_font)
        draw.text((x + (card - (box[2] - box[0])) / 2, 14), label,
                  fill=INK, font=label_font)
        if slot < len(indices) - 1:
            draw_arrow(draw, x + card + 10, x + card + gap - 10, top + card // 2)
    return canvas


def build_grid(frames: list[Image.Image]) -> Image.Image:
    card = 170
    gap_x = 20
    gap_y = 58
    margin_x = 26
    margin_y = 20
    label_h = 34
    width = 2 * margin_x + 8 * card + 7 * gap_x
    height = 2 * margin_y + 2 * (label_h + card) + gap_y
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    label_font = font(20, bold=True)
    for idx, frame in enumerate(frames, start=1):
        row = (idx - 1) // 8
        col = (idx - 1) % 8
        x = margin_x + col * (card + gap_x)
        y = margin_y + row * (label_h + card + gap_y)
        label = f"t+{idx}"
        box = draw.textbbox((0, 0), label, font=label_font)
        draw.text((x + (card - (box[2] - box[0])) / 2, y), label,
                  fill=INK, font=label_font)
        canvas.paste(frame.resize((card, card), Image.Resampling.LANCZOS),
                     (x, y + label_h))
        if col < 7:
            draw_arrow(draw, x + card + 4, x + card + gap_x - 3,
                       y + label_h + card // 2)
    return canvas


def save_asset(stem: str, image: Image.Image) -> None:
    image.save(OUT / f"{stem}.png", dpi=(300, 300))
    image.save(OUT / f"{stem}.pdf", resolution=300)
    print(OUT / f"{stem}.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frames = render_frames(load_board_prediction())
    save_asset("foresight_future_continuous_t1_t6", build_row(frames, list(range(1, 7))))
    save_asset("foresight_future_continuous_t1_t16_row", build_row(frames, list(range(1, 17)), card=150))
    save_asset("foresight_future_continuous_t1_t16_grid", build_grid(frames))


if __name__ == "__main__":
    main()
