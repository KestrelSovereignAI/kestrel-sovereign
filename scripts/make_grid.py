#!/usr/bin/env python3
"""Create a grid of all logo concepts for easy comparison."""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

output_dir = Path(__file__).parent.parent / "docs" / "logo-concepts"

# Collect all images in order
sections = [
    ("V1 Originals", [
        ("Heraldic", output_dir / "kestrel-heraldic.png"),
        ("Minimalist", output_dir / "kestrel-minimalist.png"),
        ("Dramatic", output_dir / "kestrel-dramatic.png"),
        ("Woodcut", output_dir / "kestrel-woodcut.png"),
        ("Modern Tech", output_dir / "kestrel-modern_tech.png"),
        ("Playful", output_dir / "kestrel-playful.png"),
    ]),
    ("V2 Minimalist", [
        ("Upward", output_dir / "v2" / "min-v1-upward.png"),
        ("Negative", output_dir / "v2" / "min-v2-negative.png"),
        ("Circle", output_dir / "v2" / "min-v3-circle.png"),
        ("Merged", output_dir / "v2" / "min-v4-merged.png"),
        ("Launching", output_dir / "v2" / "min-v5-launching.png"),
    ]),
    ("V2 Heraldic", [
        ("Sharp", output_dir / "v2" / "herald-v1-sharp.png"),
        ("Modern", output_dir / "v2" / "herald-v2-modern.png"),
        ("Seal", output_dir / "v2" / "herald-v3-seal.png"),
        ("Crest", output_dir / "v2" / "herald-v4-crest.png"),
        ("Minimal", output_dir / "v2" / "herald-v5-minimal.png"),
    ]),
]

# Grid settings
thumb_size = 400
padding = 20
label_height = 40
section_header_height = 50
cols = 6  # max columns

# Calculate grid dimensions
max_row_items = max(len(items) for _, items in sections)
grid_width = cols * (thumb_size + padding) + padding

total_height = padding
for section_name, items in sections:
    total_height += section_header_height
    rows_needed = (len(items) + cols - 1) // cols
    total_height += rows_needed * (thumb_size + label_height + padding)
total_height += padding

# Create canvas
canvas = Image.new("RGB", (grid_width, total_height), "white")
draw = ImageDraw.Draw(canvas)

# Try to get a decent font
try:
    font_label = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    font_header = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
except OSError:
    font_label = ImageFont.load_default()
    font_header = ImageFont.load_default()

y_cursor = padding

for section_name, items in sections:
    # Section header
    draw.text((padding, y_cursor), section_name, fill="#1a1a2e", font=font_header)
    y_cursor += section_header_height

    for i, (label, path) in enumerate(items):
        col = i % cols
        row = i // cols
        x = padding + col * (thumb_size + padding)
        y = y_cursor + row * (thumb_size + label_height + padding)

        if path.exists():
            img = Image.open(path)
            img.thumbnail((thumb_size, thumb_size), Image.LANCZOS)
            # Center in the thumb_size box
            x_offset = x + (thumb_size - img.width) // 2
            y_offset = y + (thumb_size - img.height) // 2
            canvas.paste(img, (x_offset, y_offset))
            # Light border
            draw.rectangle([x, y, x + thumb_size, y + thumb_size], outline="#cccccc", width=1)
        else:
            draw.rectangle([x, y, x + thumb_size, y + thumb_size], outline="#cccccc", width=1)
            draw.text((x + 10, y + thumb_size // 2), "missing", fill="#999")

        # Label below
        bbox = draw.textbbox((0, 0), label, font=font_label)
        text_w = bbox[2] - bbox[0]
        draw.text((x + (thumb_size - text_w) // 2, y + thumb_size + 4), label, fill="#333", font=font_label)

    rows_in_section = (len(items) + cols - 1) // cols
    y_cursor += rows_in_section * (thumb_size + label_height + padding)

grid_path = output_dir / "all-concepts-grid.png"
canvas.save(str(grid_path), quality=95)
print(f"Grid saved to {grid_path}")
print(f"Size: {canvas.width} x {canvas.height}")
