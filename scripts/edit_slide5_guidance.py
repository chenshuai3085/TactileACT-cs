#!/usr/bin/env python3
"""Add the inference-time guidance panel to slide 5 of the working PPT.

The edit is intentionally limited to slide 5.  The new elements are native
PowerPoint shapes so the panel remains editable after opening the deck.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml import parse_xml
from pptx.oxml.ns import nsdecls, qn
from pptx.util import Inches, Pt


EMU_PER_INCH = 914400


def rgb(value: str) -> RGBColor:
    value = value.lstrip("#")
    return RGBColor(int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))


COLORS = {
    "ink": rgb("263238"),
    "muted": rgb("6B7785"),
    "blue": rgb("5A86C5"),
    "blue_light": rgb("DCEAF8"),
    "blue_panel": rgb("F5F9FE"),
    "orange": rgb("EE822F"),
    "orange_light": rgb("FBE4D0"),
    "red": rgb("F06A5E"),
    "red_light": rgb("FBE0DC"),
    "green": rgb("72B77A"),
    "green_dark": rgb("3D8C58"),
    "green_light": rgb("DDEFE0"),
    "gray": rgb("AAB5C0"),
    "gray_light": rgb("E9EEF3"),
    "white": rgb("FFFFFF"),
}


def _set_alpha(shape, percent: int) -> None:
    """Set fill alpha using DrawingML because python-pptx lacks transparency."""
    if not percent:
        return
    solid = shape._element.spPr.solidFill
    if solid is None:
        return
    color = solid[0]
    alpha = color.find(qn("a:alpha"))
    if alpha is None:
        alpha = parse_xml(f"<a:alpha {nsdecls('a')} val='100000'/>")
        color.append(alpha)
    alpha.set("val", str(max(0, min(100, 100 - percent)) * 1000))


def fill(shape, color: RGBColor, transparency: int = 0) -> None:
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    _set_alpha(shape, transparency)


def line(shape, color: RGBColor, width: float = 1.0, transparency: int = 0) -> None:
    shape.line.color.rgb = color
    shape.line.width = Pt(width)
    if transparency:
        ln = shape._element.spPr.ln
        solid = ln.find(qn("a:solidFill"))
        if solid is None:
            solid = parse_xml(f"<a:solidFill {nsdecls('a')}><a:srgbClr val='FFFFFF'/></a:solidFill>")
            ln.append(solid)
        color_el = solid[0]
        alpha = color_el.find(qn("a:alpha"))
        if alpha is None:
            alpha = parse_xml(f"<a:alpha {nsdecls('a')} val='100000'/>")
            color_el.append(alpha)
        alpha.set("val", str(max(0, min(100, 100 - transparency)) * 1000))


def set_dash(shape, dash: str = "dash") -> None:
    ln = shape._element.spPr.ln
    prst = ln.find(qn("a:prstDash"))
    if prst is None:
        prst = parse_xml(f"<a:prstDash {nsdecls('a')} val='{dash}'/>")
        ln.append(prst)
    else:
        prst.set("val", dash)


def arrow_head(shape, end: bool = True, kind: str = "triangle") -> None:
    ln = shape._element.spPr.ln
    tag = qn("a:tailEnd" if end else "a:headEnd")
    old = ln.find(tag)
    if old is not None:
        ln.remove(old)
    node = parse_xml(f"<a:{'tailEnd' if end else 'headEnd'} {nsdecls('a')} type='{kind}' w='med' len='med'/>")
    ln.append(node)


def connector(slide, x1, y1, x2, y2, color, width=1.25, dashed=False, arrow=True):
    shape = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT,
        Inches(x1), Inches(y1), Inches(x2), Inches(y2)
    )
    line(shape, color, width)
    if dashed:
        set_dash(shape)
    if arrow:
        arrow_head(shape)
    return shape


def curve(slide, x1, y1, x2, y2, color, width=1.5, dashed=False):
    shape = slide.shapes.add_connector(
        MSO_CONNECTOR.CURVE,
        Inches(x1), Inches(y1), Inches(x2), Inches(y2)
    )
    line(shape, color, width)
    if dashed:
        set_dash(shape)
    arrow_head(shape)
    return shape


def box(slide, x, y, w, h, color, fill_color=None, radius=True, width=1.0, transparency=0):
    kind = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    shape = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    fill(shape, fill_color or COLORS["white"], transparency)
    line(shape, color, width)
    return shape


def ellipse(slide, x, y, w, h, color, fill_color=None, width=1.0, transparency=0):
    shape = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(w), Inches(h))
    fill(shape, fill_color or COLORS["white"], transparency)
    line(shape, color, width)
    return shape


def text(slide, value, x, y, w, h, size=9, color=None, bold=False, align=PP_ALIGN.CENTER, font="Arial"):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = shape.text_frame
    tf.clear()
    tf.margin_left = Pt(1)
    tf.margin_right = Pt(1)
    tf.margin_top = Pt(0)
    tf.margin_bottom = Pt(0)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.word_wrap = False
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = value
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color or COLORS["ink"]
    return shape


def remove_old_gradient(slide) -> None:
    """Remove the old bracket/vertical arrow/gradient box near the right edge."""
    for shape in list(slide.shapes):
        x = shape.left / EMU_PER_INCH
        y = shape.top / EMU_PER_INCH
        w = shape.width / EMU_PER_INCH
        h = shape.height / EMU_PER_INCH
        if 9.20 <= x <= 9.80 and 1.45 <= y <= 5.70 and w <= 1.0:
            shape._element.getparent().remove(shape._element)


def fix_existing_labels(slide) -> None:
    for shape in slide.shapes:
        if not hasattr(shape, "text"):
            continue
        value = shape.text.strip()
        if value not in {"Slip", "Crush", "Stable"}:
            continue
        shape.left = Inches(8.24)
        shape.width = Inches(0.92)
        shape.height = Inches(0.30)
        if value == "Slip":
            shape.top = Inches(2.20)
        elif value == "Crush":
            shape.top = Inches(3.86)
        else:
            shape.top = Inches(5.61)
        tf = shape.text_frame
        tf.margin_left = Pt(0)
        tf.margin_right = Pt(0)
        tf.word_wrap = False
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        for p in tf.paragraphs:
            p.alignment = PP_ALIGN.CENTER
            for run in p.runs:
                run.font.name = "Arial"
                run.font.size = Pt(13)
                run.font.color.rgb = COLORS["ink"]


def add_action_tokens(slide, x, y, guided=False):
    colors = [COLORS["gray_light"], COLORS["gray_light"], COLORS["orange_light"], COLORS["orange"], COLORS["orange"]]
    if guided:
        colors = [COLORS["green_light"], COLORS["green_light"], COLORS["green_light"], COLORS["green"], COLORS["green_dark"]]
    widths = [0.34, 0.34, 0.34, 0.34, 0.34]
    for i, w in enumerate(widths):
        fill_color = colors[i]
        border = COLORS["orange"] if not guided and i >= 3 else (COLORS["green_dark"] if guided and i >= 3 else COLORS["gray"])
        box(slide, x + i * 0.43, y, w, 0.27, border, fill_color, radius=True, width=0.8)
        if i < len(widths) - 1:
            connector(slide, x + i * 0.43 + w + 0.015, y + 0.135, x + (i + 1) * 0.43 - 0.015, y + 0.135, COLORS["muted"], width=0.65, arrow=True)
    # Small positional accents make the strip read as an action chunk, not a progress bar.
    for i in range(3):
        ellipse(slide, x + 0.08 + i * 0.09, y + 0.08, 0.035, 0.035, COLORS["white"], COLORS["white"], width=0.2)


def add_quality_field(slide, x, y, w, h):
    """A small local score-field inset; it is deliberately schematic."""
    # Outer trust region and two risk lobes.
    ellipse(slide, x, y, w, h, COLORS["blue"], COLORS["blue_panel"], width=0.8)
    ellipse(slide, x + 0.08, y + 0.10, w - 0.16, h - 0.20, COLORS["red"], COLORS["red_light"], width=0.7, transparency=8)
    ellipse(slide, x + 0.22, y + 0.25, w - 0.45, h - 0.48, COLORS["green"], COLORS["green_light"], width=0.8, transparency=4)
    ellipse(slide, x + 0.38, y + 0.38, w - 0.76, h - 0.74, COLORS["green_dark"], COLORS["green_light"], width=0.7)
    # Current action and guided local update.
    ellipse(slide, x + 0.23, y + 0.52, 0.10, 0.10, COLORS["orange"], COLORS["orange"], width=0.4)
    connector(slide, x + 0.33, y + 0.57, x + 0.82, y + 0.57, COLORS["green_dark"], width=1.5, arrow=True)
    ellipse(slide, x + 0.79, y + 0.52, 0.10, 0.10, COLORS["green_dark"], COLORS["green"], width=0.5)
    text(slide, "local contact-quality field", x - 0.02, y + h + 0.02, w + 0.04, 0.16, size=6.3, color=COLORS["muted"])


def add_guidance_panel(slide):
    # A restrained tinted lane separates the new mechanism from the existing rows.
    panel = box(slide, 10.30, 1.00, 2.83, 5.42, COLORS["blue"], COLORS["blue_panel"], radius=True, width=0.8)
    # Make the panel very light so it does not compete with the tactile heatmaps.
    _set_alpha(panel, 2)

    text(slide, "BASE ACTION SAMPLE", 10.43, 1.12, 2.55, 0.18, size=7.5, color=COLORS["muted"], bold=True)
    text(slide, "x^k", 10.42, 1.31, 0.28, 0.18, size=8, color=COLORS["muted"], align=PP_ALIGN.LEFT)
    add_action_tokens(slide, 10.78, 1.30, guided=False)
    text(slide, "clean estimate", 11.95, 1.08, 0.86, 0.16, size=6.2, color=COLORS["orange"], align=PP_ALIGN.RIGHT)

    text(slide, "LATE DENOISING STEP", 10.43, 1.86, 2.55, 0.18, size=7.5, color=COLORS["muted"], bold=True)
    # Base policy denoising chain.
    node_x = [10.62, 11.12, 11.62, 12.12, 12.62]
    for i, nx in enumerate(node_x):
        if i < len(node_x) - 1:
            connector(slide, nx + 0.09, 2.24, node_x[i + 1] - 0.09, 2.24, COLORS["gray"], width=1.0, arrow=True)
        ellipse(slide, nx, 2.15, 0.18, 0.18, COLORS["gray"], COLORS["white"], width=1.0)
    # Highlight the late step and its bounded trust region.
    ellipse(slide, 11.98, 2.01, 0.46, 0.46, COLORS["green"], COLORS["green_light"], width=1.0, transparency=8)
    ellipse(slide, 12.12, 2.15, 0.18, 0.18, COLORS["green_dark"], COLORS["green"], width=1.0)
    text(slide, "bounded local update", 11.73, 2.51, 1.08, 0.18, size=6.2, color=COLORS["green_dark"])

    # Small local energy/quality field beside the highlighted node.
    add_quality_field(slide, 10.60, 2.82, 1.55, 1.05)

    # The score gradient enters the highlighted late node.  The score node itself
    # sits just outside the lane, where the three outcome gauges converge.
    score = box(slide, 9.38, 3.26, 0.72, 0.48, COLORS["orange"], COLORS["orange_light"], radius=True, width=1.0)
    text(slide, "∇x Sψ", 9.42, 3.30, 0.64, 0.22, size=10, color=COLORS["orange"], bold=True)
    # Three short incoming links from the three existing score gauges.
    gauge_y = [1.70, 3.35, 5.17]
    for gy in gauge_y:
        connector(slide, 9.18, gy, 9.38, 3.50, COLORS["gray"], width=0.8, arrow=False)
    # The guidance must enter the highlighted late node, not the first denoising
    # node.  A direct diagonal keeps the loop legible in the narrow right lane.
    connector(slide, 10.10, 3.50, 12.12, 2.24, COLORS["orange"], width=1.35, dashed=True, arrow=True)
    text(slide, "contact gradient", 10.33, 3.72, 1.05, 0.18, size=6.3, color=COLORS["orange"], align=PP_ALIGN.LEFT)

    connector(slide, 12.64, 3.92, 12.64, 4.43, COLORS["green_dark"], width=1.2, arrow=True)

    # Continue the sample after the local correction.
    text(slide, "GUIDED ACTION", 10.43, 4.20, 2.55, 0.18, size=7.5, color=COLORS["green_dark"], bold=True)
    add_action_tokens(slide, 10.78, 4.48, guided=True)
    text(slide, "x^0", 10.42, 4.52, 0.28, 0.18, size=8, color=COLORS["green_dark"], align=PP_ALIGN.LEFT)
    # Stable endpoint badge.
    ellipse(slide, 12.72, 4.43, 0.30, 0.30, COLORS["green_dark"], COLORS["green_light"], width=1.0)
    text(slide, "✓", 12.72, 4.44, 0.30, 0.27, size=13, color=COLORS["green_dark"], bold=True)
    connector(slide, 12.55, 4.62, 12.70, 4.58, COLORS["green_dark"], width=1.0, arrow=True)


def edit(input_path: Path, output_path: Path) -> None:
    prs = Presentation(str(input_path))
    slide = prs.slides[4]
    remove_old_gradient(slide)
    fix_existing_labels(slide)
    add_guidance_panel(slide)
    prs.save(str(output_path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    edit(args.input, args.output)


if __name__ == "__main__":
    main()
