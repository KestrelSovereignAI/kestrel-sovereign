#!/usr/bin/env python3
"""Create a comparison grid of the full Kestrel ecosystem branding suite."""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

import sys

# Default to v2; pass --v1 to use original ecosystem dir
_v1 = "--v1" in sys.argv
ROOT = Path(__file__).parent.parent / "docs" / "logo-concepts" / ("ecosystem" if _v1 else "ecosystem-v2")

# Grid layout: rows = categories, columns = styles
ENTITY_ROWS = [
    ("Sovereign — The Roost", "sovereign"),
    ("Talon — The Strike", "talon"),
    ("Eye — The Sight", "eye"),
    ("Flight — The Voice", "flight"),
    ("Claws — The Grip", "claws"),
    ("Castle — The Fortress", "castle"),
    ("Falconer — The Handler", "falconer"),
]
STYLE_COLS = ["heraldic", "minimalist", "dramatic"]

RITUAL_ITEMS = [
    ("Morning Signal", "rituals/morning-signal.png"),
    ("The Hunt", "rituals/the-hunt.png"),
    ("Evening Return", "rituals/evening-return.png"),
    ("Learning Loop", "rituals/learning-loop.png"),
]

FAMILY_ITEMS = [
    ("Full Flock", "family/full-flock.png"),
    ("Hierarchy", "family/hierarchy.png"),
]

# Layout settings
THUMB = 400
PAD = 20
LABEL_H = 40
ROW_LABEL_W = 280
SECTION_H = 60
COL_HEADER_H = 50


def load_fonts():
    try:
        label = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        header = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
        section = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 32)
    except OSError:
        label = header = section = ImageFont.load_default()
    return label, header, section


def draw_image_cell(canvas, draw, img_path, x, y, label, font_label):
    """Draw a thumbnail with label at (x, y)."""
    if img_path.exists():
        img = Image.open(img_path)
        img.thumbnail((THUMB, THUMB), Image.LANCZOS)
        x_off = x + (THUMB - img.width) // 2
        y_off = y + (THUMB - img.height) // 2
        canvas.paste(img, (x_off, y_off))
        draw.rectangle([x, y, x + THUMB, y + THUMB], outline="#cccccc", width=1)
    else:
        draw.rectangle([x, y, x + THUMB, y + THUMB], outline="#cccccc", width=1)
        draw.text((x + 10, y + THUMB // 2), "not yet generated", fill="#999", font=font_label)
    # Label below
    bbox = draw.textbbox((0, 0), label, font=font_label)
    tw = bbox[2] - bbox[0]
    draw.text((x + (THUMB - tw) // 2, y + THUMB + 4), label, fill="#333", font=font_label)


def main():
    font_label, font_header, font_section = load_fonts()

    cols = len(STYLE_COLS)
    grid_w = ROW_LABEL_W + cols * (THUMB + PAD) + PAD

    # Calculate total height
    h = PAD
    h += SECTION_H  # "Entity Logos" header
    h += COL_HEADER_H  # column headers
    h += len(ENTITY_ROWS) * (THUMB + LABEL_H + PAD)  # entity rows
    h += SECTION_H  # "Rituals" header
    h += THUMB + LABEL_H + PAD  # ritual row
    h += SECTION_H  # "Family" header
    h += THUMB + LABEL_H + PAD  # family row
    h += PAD

    canvas = Image.new("RGB", (grid_w, h), "white")
    draw = ImageDraw.Draw(canvas)
    y = PAD

    # === Section: Entity Logos ===
    draw.text((PAD, y), "Entity Logos", fill="#1B3A6B", font=font_section)
    y += SECTION_H

    # Column headers
    for ci, style in enumerate(STYLE_COLS):
        x = ROW_LABEL_W + ci * (THUMB + PAD)
        bbox = draw.textbbox((0, 0), style.title(), font=font_header)
        tw = bbox[2] - bbox[0]
        draw.text((x + (THUMB - tw) // 2, y), style.title(), fill="#006D77", font=font_header)
    y += COL_HEADER_H

    # Entity rows
    for row_label, cat in ENTITY_ROWS:
        # Row label (vertically centered)
        bbox = draw.textbbox((0, 0), row_label, font=font_header)
        th = bbox[3] - bbox[1]
        draw.text((PAD, y + (THUMB - th) // 2), row_label, fill="#1B3A6B", font=font_header)
        # Thumbnails
        for ci, style in enumerate(STYLE_COLS):
            x = ROW_LABEL_W + ci * (THUMB + PAD)
            img_path = ROOT / cat / f"{style}.png"
            draw_image_cell(canvas, draw, img_path, x, y, style.title(), font_label)
        y += THUMB + LABEL_H + PAD

    # === Section: Rituals ===
    draw.text((PAD, y), "Rituals", fill="#1B3A6B", font=font_section)
    y += SECTION_H

    for i, (label, rel_path) in enumerate(RITUAL_ITEMS):
        x = ROW_LABEL_W + i * (THUMB + PAD)
        img_path = ROOT / rel_path
        draw_image_cell(canvas, draw, img_path, x, y, label, font_label)
    y += THUMB + LABEL_H + PAD

    # === Section: Family ===
    draw.text((PAD, y), "Family", fill="#1B3A6B", font=font_section)
    y += SECTION_H

    for i, (label, rel_path) in enumerate(FAMILY_ITEMS):
        x = ROW_LABEL_W + i * (THUMB + PAD)
        img_path = ROOT / rel_path
        draw_image_cell(canvas, draw, img_path, x, y, label, font_label)

    # Save
    grid_path = ROOT / "ecosystem-grid.png"
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(grid_path), quality=95)
    print(f"Grid saved to {grid_path}")
    print(f"Size: {canvas.width} x {canvas.height}")


if __name__ == "__main__":
    main()
