#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "opencv-python>=4.8",
#   "numpy>=1.26",
#   "Pillow>=10",
#   "scikit-image>=0.22",
#   "fonttools>=4.50",
# ]
# ///
"""Build the Kestrel logo asset bundle (SVG + PNG + ICO) from a source PNG.

Inputs:
  docs/design/launch/sources/kestrel_mark_source.png    (bird + castle, no text)
  docs/design/launch/sources/kestrel_lockup_source.png  (mark + wordmark, reference only)
  docs/design/launch/fonts/Montserrat-VF.ttf            (variable font, 100..900)

Outputs land under docs/design/launch/{mark,lockup}/.

Pipeline stages (run with `--all`, or individually):
  1. Trace mark source -> kestrel_mark.svg (vector, transparent background).
  2. Compose lockup SVG = traced mark + outlined Montserrat-Black wordmark.
  3. Render PNGs at every requested size, both transparent and white-bg.
  4. Bundle mark PNGs into a multi-resolution favicon.ico.
  5. Compare each rendered SVG to its source PNG via SSIM; gate at >= 0.97.
  6. Emit manifest.json with sha256 per artifact.

Run with `uv run scripts/build_logo_assets.py --all` (PEP 723 deps auto-install).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
LAUNCH_DIR = PROJECT_ROOT / "docs/design/launch"
SOURCES_DIR = LAUNCH_DIR / "sources"
FONTS_DIR = LAUNCH_DIR / "fonts"
MARK_DIR = LAUNCH_DIR / "mark"
LOCKUP_DIR = LAUNCH_DIR / "lockup"

MARK_SOURCE = SOURCES_DIR / "kestrel_mark_source.png"
LOCKUP_SOURCE = SOURCES_DIR / "kestrel_lockup_source.png"
FONT_VF = FONTS_DIR / "Montserrat-VF.ttf"

# Size matrices.
MARK_PNG_SIZES = [16, 32, 48, 64, 128, 180, 192, 256, 400, 512, 1024]
LOCKUP_PNG_WIDTHS = [256, 512, 1024, 1280, 2048]
ICO_SIZES = [16, 32, 48, 64]

# SSIM gate. Below this, fail loudly.
SSIM_GATE = 0.97

# Tracing knobs.
UPSCALE_TARGET = 4096      # potrace fidelity scales with input pixels
COLOR_TOLERANCE = 28       # per-channel L2 distance for mask membership
MIN_REGION_AREA = 600      # px^2 at 4096-trace resolution; drops AA noise
POTRACE_FLAGS = ["-i", "-a", "1.34", "-O", "0.5", "-t", "20", "-u", "1"]

# Wordmark layout. Sizes auto-fit to width fractions of the mark.
WORDMARK_GAP_FRAC = 0.04         # gap above wordmark, as fraction of mark height
WORDMARK_LINE_GAP_FRAC = 0.02    # gap between KESTREL and SOVEREIGN AI
WORDMARK_KESTREL_WIDTH_FRAC = 0.72   # KESTREL spans this fraction of mark width
WORDMARK_TAGLINE_WIDTH_FRAC = 0.50   # SOVEREIGN AI spans this fraction
WORDMARK_FONT_WEIGHT = 900       # Black


# ---------------------------------------------------------------------------
# Palette (extracted, then snapped to canonical values from the design spec).
# ---------------------------------------------------------------------------

@dataclass
class PaletteEntry:
    name: str
    rgb: tuple[int, int, int]
    @property
    def hex(self) -> str:
        return "#{:02X}{:02X}{:02X}".format(*self.rgb)


def extract_palette(rgb: np.ndarray, k: int = 6) -> list[PaletteEntry]:
    """K-means cluster the foreground pixels and return the dominant colours."""
    h, w = rgb.shape[:2]
    bg = background_mask(rgb)
    fg_pixels = rgb[bg == 0].reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    _, labels, centers = cv2.kmeans(
        fg_pixels, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )
    counts = np.bincount(labels.flatten(), minlength=k)
    order = np.argsort(-counts)
    out: list[PaletteEntry] = []
    for idx in order:
        rgb_c = tuple(int(c) for c in centers[idx])
        out.append(PaletteEntry(name=f"c{len(out)}", rgb=rgb_c))
    return out


def classify_palette(palette: list[PaletteEntry]) -> dict[str, PaletteEntry]:
    """Assign semantic names to extracted colours by HSV characteristics."""
    named: dict[str, PaletteEntry] = {}
    for entry in palette:
        r, g, b = entry.rgb
        hsv = cv2.cvtColor(np.uint8([[[r, g, b]]]), cv2.COLOR_RGB2HSV)[0, 0]
        h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
        if v < 90 and s > 60:                         # deep, saturated -> navy/silhouette
            key = "silhouette"
        elif s > 100 and 75 <= h <= 110 and v > 130:  # bright cyan/teal accent
            key = "cyan"
        elif v > 220 and s < 40:                       # near-white highlight
            key = "white"
        elif s > 100 and 15 <= h <= 40 and v > 150:    # gold/yellow beak
            key = "gold"
        elif v < 50:                                    # truly black pupil
            key = "pupil"
        else:
            key = f"misc_{len(named)}"
        named.setdefault(key, entry)
    return named


# ---------------------------------------------------------------------------
# Background separation. White-bg pixels reachable from any corner = transparent.
# White pixels strictly inside the mark (eye highlight, etc.) are kept.
# ---------------------------------------------------------------------------

def background_mask(rgb: np.ndarray, tol: int = 6) -> np.ndarray:
    h, w = rgb.shape[:2]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    seeds = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
    for sx, sy in seeds:
        # Only seed where the corner pixel itself is near-white.
        px = rgb[sy, sx]
        if int(px[0]) > 240 and int(px[1]) > 240 and int(px[2]) > 240:
            cv2.floodFill(
                bgr, flood_mask, (sx, sy), (0, 0, 255),
                loDiff=(tol, tol, tol), upDiff=(tol, tol, tol),
                flags=4 | cv2.FLOODFILL_FIXED_RANGE,
            )
    bg = (flood_mask[1:-1, 1:-1] != 0).astype(np.uint8) * 255
    return bg


# ---------------------------------------------------------------------------
# Per-colour binary masks (foreground only).
# ---------------------------------------------------------------------------

def colour_mask(rgb: np.ndarray, target: tuple[int, int, int],
                bg: np.ndarray, tolerance: int = COLOR_TOLERANCE) -> np.ndarray:
    target_arr = np.array(target, dtype=np.float32)
    diff = rgb.astype(np.float32) - target_arr
    distance = np.sqrt(np.sum(diff * diff, axis=2))
    mask = ((distance < tolerance) & (bg == 0)).astype(np.uint8) * 255
    # Morphology to close small holes and drop specks. Larger kernel kills more AA noise.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def drop_small_components(mask: np.ndarray, min_area: int = MIN_REGION_AREA) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), 8
    )
    keep = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[i] = True
    return np.where(keep[labels], 255, 0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Potrace bridge.
# ---------------------------------------------------------------------------

def trace_mask_to_paths(mask: np.ndarray) -> str:
    """Return SVG <path d="..."/> markup for the foreground in `mask`. May be empty."""
    if not mask.any():
        return ""
    with tempfile.TemporaryDirectory() as td:
        bmp_path = Path(td) / "mask.bmp"
        svg_path = Path(td) / "mask.svg"
        Image.fromarray(mask).convert("L").save(bmp_path, format="BMP")
        cmd = ["potrace", str(bmp_path), "-s", "-o", str(svg_path), *POTRACE_FLAGS]
        subprocess.run(cmd, check=True, capture_output=True)
        svg_text = svg_path.read_text()
    # Pull every <path d="..."/> out and concatenate the d attributes.
    import re
    ds = re.findall(r'd="([^"]+)"', svg_text)
    if not ds:
        return ""
    combined = " ".join(ds)
    return combined


# ---------------------------------------------------------------------------
# SVG composition.
# ---------------------------------------------------------------------------

LAYER_ORDER = ["silhouette", "cyan", "white", "gold", "pupil"]


@dataclass
class TracedLayer:
    name: str
    fill: str
    d: str
    fill_rule: str = "evenodd"


def trace_mark(source_png: Path) -> tuple[list[TracedLayer], tuple[int, int], tuple[int, int, int, int]]:
    """Run the full tracing pipeline.

    Returns:
        layers: traced colour layers in z-order.
        (width, height): canvas size at trace resolution.
        content_bbox: (x_min, y_min, x_max, y_max) of the union of all foreground
            masks, in trace-resolution pixels.
    """
    img_bgr = cv2.imread(str(source_png))
    if img_bgr is None:
        raise FileNotFoundError(source_png)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Upscale for potrace fidelity (Lanczos in PIL).
    h, w = img_rgb.shape[:2]
    scale = max(1.0, UPSCALE_TARGET / max(h, w))
    if scale > 1.0:
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        pil = Image.fromarray(img_rgb).resize((new_w, new_h), Image.LANCZOS)
        img_rgb = np.array(pil)

    bg = background_mask(img_rgb)
    palette = extract_palette(img_rgb, k=6)
    named = classify_palette(palette)

    # Print observed palette for debugging.
    print("[palette] extracted:")
    for entry in palette:
        print(f"  {entry.hex}  rgb={entry.rgb}")
    print("[palette] classified:")
    for key in LAYER_ORDER:
        e = named.get(key)
        if e is not None:
            print(f"  {key:11s} -> {e.hex}")

    layers: list[TracedLayer] = []
    union_mask = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
    for layer_name in LAYER_ORDER:
        entry = named.get(layer_name)
        if entry is None:
            continue
        mask = colour_mask(img_rgb, entry.rgb, bg)
        mask = drop_small_components(mask, MIN_REGION_AREA)
        if not mask.any():
            continue
        union_mask |= mask
        d = trace_mask_to_paths(mask)
        if not d:
            continue
        debug_path = MARK_DIR / "_debug" / f"mask_{layer_name}.png"
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask).save(debug_path)
        layers.append(TracedLayer(name=layer_name, fill=entry.hex, d=d))

    ys, xs = np.where(union_mask > 0)
    if len(xs) == 0:
        raise RuntimeError("trace produced an empty content bbox")
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    print(f"[trace] content bbox: x={bbox[0]}..{bbox[2]} y={bbox[1]}..{bbox[3]}")
    return layers, (img_rgb.shape[1], img_rgb.shape[0]), bbox


def render_layers_svg(
    layers: list[TracedLayer],
    width: int,
    height: int,
    *,
    transform: str | None = None,
    title: str = "Kestrel Mark",
) -> str:
    """Build an SVG document from traced layers. potrace emits y-flipped paths so we
    apply a flip transform unless the caller already wraps in a group."""
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" aria-label="{title}">',
        f'  <title>{title}</title>',
    ]
    # potrace's PostScript-style coordinate system has Y up; flip to SVG (Y down).
    g_open = f'  <g transform="translate(0,{height}) scale(1,-1)">'
    if transform:
        g_open = f'  <g transform="{transform}">'
    parts.append(g_open)
    for layer in layers:
        parts.append(
            f'    <path id="{layer.name}" fill="{layer.fill}" '
            f'fill-rule="{layer.fill_rule}" d="{layer.d}"/>'
        )
    parts.append('  </g>')
    parts.append('</svg>')
    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# Wordmark glyph outlining (variable font -> Black instance -> SVG paths).
# ---------------------------------------------------------------------------

def glyph_paths_for_text(text: str, ttf_path: Path, size_units: float, weight: int = 900):
    """Return (path_d, advance_total_units) for `text` rendered in the variable
    font at the given weight. Coordinates are in font em units, baseline at y=0."""
    from fontTools.ttLib import TTFont
    from fontTools.varLib.instancer import instantiateVariableFont
    from fontTools.pens.svgPathPen import SVGPathPen

    font = TTFont(ttf_path)
    if "fvar" in font:
        font = instantiateVariableFont(font, {"wght": weight})

    cmap = font.getBestCmap()
    glyph_set = font.getGlyphSet()
    upem = font["head"].unitsPerEm
    scale = size_units / upem  # font units -> output units

    pen_d_segments: list[str] = []
    x_cursor = 0.0
    hmtx = font["hmtx"]
    for ch in text:
        gid = cmap.get(ord(ch))
        if gid is None:
            x_cursor += upem * 0.25
            continue
        pen = SVGPathPen(glyph_set)
        glyph_set[gid].draw(pen)
        d = pen.getCommands()
        if d:
            # Transform: scale by `scale`, translate to current x cursor, flip Y
            # because font Y is up. Writing this as a transform on a <g> would be
            # cleaner, but composing inline keeps the lockup SVG flat.
            transform = (
                f'translate({x_cursor * scale:.3f},0) '
                f'scale({scale:.5f},{-scale:.5f})'
            )
            pen_d_segments.append(f'<path transform="{transform}" d="{d}"/>')
        adv, _ = hmtx[gid]
        x_cursor += adv

    total_advance = x_cursor * scale
    return pen_d_segments, total_advance


# ---------------------------------------------------------------------------
# Lockup composition.
# ---------------------------------------------------------------------------

def compose_lockup(
    mark_layers: list[TracedLayer],
    mark_width: int,
    mark_height: int,
    content_bbox: tuple[int, int, int, int],
    font_path: Path,
) -> str:
    """Compose a lockup SVG: mark above, "KESTREL" + "SOVEREIGN AI" wordmarks below.

    Both wordmarks are outlined Montserrat Black so the SVG renders identically
    on any host.
    """
    silhouette = next((l for l in mark_layers if l.name == "silhouette"), mark_layers[0])
    title_fill = silhouette.fill
    bx0, by0, bx1, by1 = content_bbox
    bird_height = by1 - by0
    bird_width = bx1 - bx0

    # Probe each wordmark at em=1000 to measure its natural width, then scale to
    # the target fraction of mark width.
    probe_em = 1000.0
    _, line1_probe_w = glyph_paths_for_text("KESTREL", font_path, probe_em)
    _, line2_probe_w = glyph_paths_for_text("SOVEREIGN AI", font_path, probe_em)

    target_w_kestrel = bird_width * WORDMARK_KESTREL_WIDTH_FRAC
    target_w_tagline = bird_width * WORDMARK_TAGLINE_WIDTH_FRAC
    kestrel_em = probe_em * (target_w_kestrel / line1_probe_w)
    tagline_em = probe_em * (target_w_tagline / line2_probe_w)

    line1_glyphs, line1_w = glyph_paths_for_text("KESTREL", font_path, kestrel_em)
    line2_glyphs, line2_w = glyph_paths_for_text("SOVEREIGN AI", font_path, tagline_em)

    # Cap height ~ 0.72 * em for Montserrat.
    cap_kestrel = kestrel_em * 0.72
    cap_tagline = tagline_em * 0.72

    # Compose lockup viewBox: tight crop around content + wordmarks.
    pad_top = bird_height * 0.05
    pad_side = bird_width * 0.04
    gap_above = bird_height * WORDMARK_GAP_FRAC
    gap_between = bird_height * WORDMARK_LINE_GAP_FRAC

    line1_y_baseline = by1 + gap_above + cap_kestrel
    line2_y_baseline = line1_y_baseline + gap_between + cap_tagline
    canvas_top = by0 - pad_top
    canvas_bottom = line2_y_baseline + pad_top
    canvas_left = bx0 - pad_side
    canvas_right = bx1 + pad_side
    canvas_w = canvas_right - canvas_left
    canvas_h = canvas_bottom - canvas_top

    line1_x = bx0 + (bird_width - line1_w) / 2.0
    line2_x = bx0 + (bird_width - line2_w) / 2.0

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="{canvas_left:.1f} {canvas_top:.1f} {canvas_w:.1f} {canvas_h:.1f}" '
        f'width="{canvas_w:.0f}" height="{canvas_h:.0f}" '
        f'role="img" aria-label="Kestrel Sovereign AI">',
        '  <title>Kestrel Sovereign AI</title>',
        f'  <g id="kestrel-mark" transform="translate(0,{mark_height}) scale(1,-1)">',
    ]
    for layer in mark_layers:
        parts.append(
            f'    <path id="{layer.name}" fill="{layer.fill}" '
            f'fill-rule="{layer.fill_rule}" d="{layer.d}"/>'
        )
    parts.append('  </g>')

    parts.append(
        f'  <g id="wordmark-kestrel" fill="{title_fill}" '
        f'transform="translate({line1_x:.2f},{line1_y_baseline:.2f})">'
    )
    parts.extend(f"    {g}" for g in line1_glyphs)
    parts.append('  </g>')

    parts.append(
        f'  <g id="wordmark-tagline" fill="{title_fill}" '
        f'transform="translate({line2_x:.2f},{line2_y_baseline:.2f})">'
    )
    parts.extend(f"    {g}" for g in line2_glyphs)
    parts.append('  </g>')

    parts.append('</svg>')
    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# Rendering: SVG -> PNG via rsvg-convert.
# ---------------------------------------------------------------------------

def rsvg_render(svg_path: Path, png_path: Path, *, width: int | None = None,
                height: int | None = None, background: str | None = None) -> None:
    cmd = ["rsvg-convert", str(svg_path), "-o", str(png_path)]
    if width is not None:
        cmd += ["-w", str(width)]
    if height is not None:
        cmd += ["-h", str(height)]
    if background is not None:
        cmd += ["-b", background]
    subprocess.run(cmd, check=True, capture_output=True)


def make_white_bg_variant(transparent_png: Path, white_png: Path) -> None:
    img = Image.open(transparent_png).convert("RGBA")
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    bg.alpha_composite(img)
    bg.convert("RGB").save(white_png, format="PNG", optimize=True)


# ---------------------------------------------------------------------------
# Build favicon.ico from a set of square PNGs.
# ---------------------------------------------------------------------------

def build_ico(png_paths: list[Path], ico_path: Path) -> None:
    images = [Image.open(p).convert("RGBA") for p in png_paths]
    largest = max(images, key=lambda im: im.size[0])
    sizes = [(im.size[0], im.size[1]) for im in images]
    largest.save(ico_path, format="ICO", sizes=sizes)


# ---------------------------------------------------------------------------
# SSIM gate.
# ---------------------------------------------------------------------------

def ssim_against_source(svg_path: Path, source_png: Path) -> float:
    src = Image.open(source_png).convert("RGB")
    with tempfile.TemporaryDirectory() as td:
        rendered = Path(td) / "rendered.png"
        # Render at source resolution, on white background, so we compare apples to apples.
        rsvg_render(svg_path, rendered, width=src.width, height=src.height,
                    background="white")
        rend = Image.open(rendered).convert("RGB").resize(src.size, Image.LANCZOS)
    a = np.array(src)
    b = np.array(rend)
    # multichannel SSIM
    score = ssim(a, b, channel_axis=2, data_range=255)
    return float(score)


# ---------------------------------------------------------------------------
# Optimisation (svgo) and manifest.
# ---------------------------------------------------------------------------

def svgo_minify(in_path: Path, out_path: Path) -> None:
    if not shutil.which("npx"):
        shutil.copy(in_path, out_path)
        return
    cmd = [
        "npx", "--yes", "svgo", str(in_path), "-o", str(out_path),
        "--multipass",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        # Fall back to copy if svgo blew up.
        sys.stderr.write(f"[svgo] failed, copying unminified: {res.stderr}\n")
        shutil.copy(in_path, out_path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class Manifest:
    generated_at: str
    artifacts: list[dict] = field(default_factory=list)


def write_manifest(artifacts: list[Path], manifest_path: Path) -> None:
    import datetime as dt
    entries = []
    for p in artifacts:
        if not p.exists():
            continue
        entries.append({
            "path": str(p.relative_to(PROJECT_ROOT)),
            "size": p.stat().st_size,
            "sha256": sha256(p),
        })
    manifest = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "artifacts": entries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


# ---------------------------------------------------------------------------
# Top-level driver.
# ---------------------------------------------------------------------------

def stage_trace_mark() -> tuple[Path, Path, list[TracedLayer], tuple[int, int], tuple[int, int, int, int]]:
    print(f"[trace] {MARK_SOURCE.name} -> mark SVG")
    layers, (w, h), bbox = trace_mark(MARK_SOURCE)
    if not layers:
        raise RuntimeError("trace produced zero layers")
    svg = render_layers_svg(layers, w, h, title="Kestrel Mark")
    raw_path = MARK_DIR / "kestrel_mark.svg"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(svg)

    min_path = MARK_DIR / "kestrel_mark.min.svg"
    svgo_minify(raw_path, min_path)
    print(f"[trace] {raw_path.stat().st_size} bytes -> {min_path.stat().st_size} bytes (minified)")
    return raw_path, min_path, layers, (w, h), bbox


def stage_compose_lockup(layers: list[TracedLayer], w: int, h: int,
                         bbox: tuple[int, int, int, int]) -> tuple[Path, Path]:
    print("[compose] lockup = mark + Montserrat Black wordmarks")
    if not FONT_VF.exists():
        raise FileNotFoundError(f"missing font: {FONT_VF}")
    svg = compose_lockup(layers, w, h, bbox, FONT_VF)
    raw_path = LOCKUP_DIR / "kestrel_lockup.svg"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(svg)

    min_path = LOCKUP_DIR / "kestrel_lockup.min.svg"
    svgo_minify(raw_path, min_path)
    print(f"[compose] {raw_path.stat().st_size} bytes -> {min_path.stat().st_size} bytes")
    return raw_path, min_path


def stage_render_pngs(svg_path: Path, out_dir: Path, *, sizes: list[int],
                      square: bool, prefix: str) -> list[Path]:
    """Render PNGs at each requested size. For square=True, sizes are NxN; for False
    we treat sizes as widths and let height auto-compute."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for s in sizes:
        for label, bg in (("transparent", None), ("white", "white")):
            name = f"{prefix}_{s}{'_white' if bg else ''}.png"
            p = out_dir / name
            if square:
                rsvg_render(svg_path, p, width=s, height=s, background=bg)
            else:
                rsvg_render(svg_path, p, width=s, background=bg)
            written.append(p)
    return written


def stage_build_ico(mark_dir: Path) -> Path:
    print("[ico] building favicon.ico")
    pngs = [mark_dir / "png" / f"kestrel_mark_{s}.png" for s in ICO_SIZES]
    for p in pngs:
        if not p.exists():
            raise FileNotFoundError(p)
    ico_path = mark_dir / "kestrel_mark.ico"
    build_ico(pngs, ico_path)
    return ico_path


def stage_ssim(svg_path: Path, source: Path, label: str) -> float:
    score = ssim_against_source(svg_path, source)
    status = "OK" if score >= SSIM_GATE else "FAIL"
    print(f"[ssim] {label}: {score:.4f} (gate {SSIM_GATE:.2f}) [{status}]")
    return score


def main() -> int:
    ap = argparse.ArgumentParser(description="Build Kestrel logo asset bundle.")
    ap.add_argument("--all", action="store_true", help="Run the full pipeline.")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--lockup", action="store_true")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--ico", action="store_true")
    ap.add_argument("--ssim", action="store_true")
    ap.add_argument("--manifest", action="store_true")
    ap.add_argument(
        "--no-gate", action="store_true",
        help="Don't fail on SSIM below threshold (useful for first pass).",
    )
    args = ap.parse_args()

    if not (args.all or args.trace or args.lockup or args.render
            or args.ico or args.ssim or args.manifest):
        ap.print_help()
        return 2

    do_trace = args.all or args.trace
    do_lockup = args.all or args.lockup
    do_render = args.all or args.render
    do_ico = args.all or args.ico
    do_ssim = args.all or args.ssim
    do_manifest = args.all or args.manifest

    layers: list[TracedLayer] = []
    mw = mh = 0
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    mark_svg: Path | None = None
    lockup_svg: Path | None = None

    if do_trace:
        mark_svg, _, layers, (mw, mh), bbox = stage_trace_mark()
    if do_lockup:
        if not layers:
            # We need traced layers + bbox to compose lockup.
            mark_svg, _, layers, (mw, mh), bbox = stage_trace_mark()
        lockup_svg, _ = stage_compose_lockup(layers, mw, mh, bbox)

    rendered_png_paths: list[Path] = []
    if do_render:
        if mark_svg is None:
            mark_svg = MARK_DIR / "kestrel_mark.svg"
        if not mark_svg.exists():
            raise FileNotFoundError(mark_svg)
        print(f"[render] mark PNGs ({len(MARK_PNG_SIZES)} sizes x 2 backgrounds)")
        rendered_png_paths += stage_render_pngs(
            mark_svg, MARK_DIR / "png", sizes=MARK_PNG_SIZES,
            square=True, prefix="kestrel_mark",
        )
        if lockup_svg is None:
            lockup_svg = LOCKUP_DIR / "kestrel_lockup.svg"
        if lockup_svg.exists():
            print(f"[render] lockup PNGs ({len(LOCKUP_PNG_WIDTHS)} widths x 2 backgrounds)")
            rendered_png_paths += stage_render_pngs(
                lockup_svg, LOCKUP_DIR / "png", sizes=LOCKUP_PNG_WIDTHS,
                square=False, prefix="kestrel_lockup",
            )

    if do_ico:
        stage_build_ico(MARK_DIR)

    if do_ssim:
        if mark_svg is None:
            mark_svg = MARK_DIR / "kestrel_mark.svg"
        if mark_svg.exists():
            mark_score = stage_ssim(mark_svg, MARK_SOURCE, "mark")
            # Only the mark is gated. The lockup intentionally re-frames the
            # composition (tighter crop, wordmark scaled to mark width), so it
            # diverges from the source layout by design and SSIM is not meaningful.
            if not args.no_gate and mark_score < SSIM_GATE:
                print(f"[ssim] FAILED gate for mark ({mark_score:.4f} < {SSIM_GATE})",
                      file=sys.stderr)
                return 1

    if do_manifest:
        artifacts: list[Path] = []
        for d in (MARK_DIR, LOCKUP_DIR):
            for p in sorted(d.rglob("*")):
                if p.is_file() and "_debug" not in p.parts:
                    artifacts.append(p)
        manifest_path = LAUNCH_DIR / "manifest.json"
        write_manifest(artifacts, manifest_path)
        print(f"[manifest] {len(artifacts)} artifacts -> {manifest_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
