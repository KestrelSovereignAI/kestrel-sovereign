#!/usr/bin/env python3
"""Smooth jagged SVG paths by resampling and re-fitting bezier curves.

Instead of re-tracing from bitmaps, this works directly on the existing
vector path data: parse -> evaluate -> resample -> smooth -> re-fit beziers.
This preserves shape detail while eliminating integer-coordinate jaggedness.
"""

import re
import numpy as np
from pathlib import Path


def tokenize_path(d: str) -> list:
    """Tokenize SVG path data into commands and numbers."""
    tokens = re.findall(r'[MmCcZzLlHhVvSsQqTtAa]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?', d)
    return tokens


def parse_path(d: str) -> list:
    """Parse SVG path data into list of subpaths.

    Each subpath is a list of (command, points) tuples.
    All coordinates converted to absolute.
    """
    tokens = tokenize_path(d)
    subpaths = []
    current_subpath = []
    cx, cy = 0.0, 0.0  # current point
    sx, sy = 0.0, 0.0  # subpath start
    i = 0

    while i < len(tokens):
        cmd = tokens[i]
        i += 1

        if cmd in ('M', 'm'):
            if current_subpath:
                subpaths.append(current_subpath)
                current_subpath = []

            x = float(tokens[i]); y = float(tokens[i+1]); i += 2
            if cmd == 'm':
                x += cx; y += cy
            cx, cy = x, y
            sx, sy = x, y
            current_subpath.append(('M', [(cx, cy)]))

            # Additional coordinate pairs after M are implicit L commands
            while i < len(tokens) and tokens[i] not in 'MmCcZzLlHhVvSsQqTtAa':
                x = float(tokens[i]); y = float(tokens[i+1]); i += 2
                if cmd == 'm':
                    x += cx; y += cy
                cx, cy = x, y
                current_subpath.append(('L', [(cx, cy)]))

        elif cmd in ('c', 'C'):
            while i + 5 < len(tokens) and tokens[i] not in 'MmCcZzLlHhVvSsQqTtAa':
                x1 = float(tokens[i]); y1 = float(tokens[i+1])
                x2 = float(tokens[i+2]); y2 = float(tokens[i+3])
                x = float(tokens[i+4]); y = float(tokens[i+5])
                i += 6
                if cmd == 'c':
                    x1 += cx; y1 += cy
                    x2 += cx; y2 += cy
                    x += cx; y += cy
                current_subpath.append(('C', [(x1, y1), (x2, y2), (x, y)]))
                cx, cy = x, y

        elif cmd in ('l', 'L'):
            while i + 1 < len(tokens) and tokens[i] not in 'MmCcZzLlHhVvSsQqTtAa':
                x = float(tokens[i]); y = float(tokens[i+1]); i += 2
                if cmd == 'l':
                    x += cx; y += cy
                cx, cy = x, y
                current_subpath.append(('L', [(cx, cy)]))

        elif cmd in ('z', 'Z'):
            current_subpath.append(('Z', []))
            cx, cy = sx, sy

        else:
            # Skip unknown commands
            pass

    if current_subpath:
        subpaths.append(current_subpath)

    return subpaths


def eval_cubic_bezier(p0, p1, p2, p3, num_samples=10):
    """Evaluate cubic bezier curve at num_samples evenly spaced t values."""
    p0, p1, p2, p3 = np.array(p0), np.array(p1), np.array(p2), np.array(p3)
    ts = np.linspace(0, 1, num_samples + 1)[1:]  # skip t=0 (that's the start point)
    points = []
    for t in ts:
        pt = (1-t)**3 * p0 + 3*(1-t)**2*t * p1 + 3*(1-t)*t**2 * p2 + t**3 * p3
        points.append(tuple(pt))
    return points


def subpath_to_polyline(subpath: list, samples_per_segment=8) -> list:
    """Convert a parsed subpath to a polyline (list of (x,y) points)."""
    points = []
    cx, cy = 0.0, 0.0

    for cmd, pts in subpath:
        if cmd == 'M':
            cx, cy = pts[0]
            points.append((cx, cy))
        elif cmd == 'L':
            cx, cy = pts[0]
            points.append((cx, cy))
        elif cmd == 'C':
            p0 = (cx, cy)
            p1, p2, p3 = pts
            sampled = eval_cubic_bezier(p0, p1, p2, p3, samples_per_segment)
            points.extend(sampled)
            cx, cy = p3
        elif cmd == 'Z':
            if points and points[0] != points[-1]:
                points.append(points[0])

    return points


def rdp_simplify(points: list, epsilon: float) -> list:
    """Ramer-Douglas-Peucker polyline simplification."""
    if len(points) <= 2:
        return points

    # Find point with maximum distance from line between first and last
    start = np.array(points[0])
    end = np.array(points[-1])
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)

    if line_len < 1e-10:
        # All points collapse to a single point
        dists = [np.linalg.norm(np.array(p) - start) for p in points]
        max_idx = np.argmax(dists)
        max_dist = dists[max_idx]
    else:
        line_unit = line_vec / line_len
        dists = []
        for p in points:
            v = np.array(p) - start
            proj = np.dot(v, line_unit)
            proj = max(0, min(line_len, proj))
            closest = start + proj * line_unit
            dists.append(np.linalg.norm(np.array(p) - closest))
        max_idx = np.argmax(dists)
        max_dist = dists[max_idx]

    if max_dist > epsilon:
        left = rdp_simplify(points[:max_idx + 1], epsilon)
        right = rdp_simplify(points[max_idx:], epsilon)
        return left[:-1] + right
    else:
        return [points[0], points[-1]]


def fit_cubic_bezier(points: list) -> tuple:
    """Fit a single cubic bezier curve to a sequence of points.

    Uses least-squares fitting with chord-length parameterization.
    Returns (p0, p1, p2, p3) control points.
    """
    pts = np.array(points)
    n = len(pts)
    if n <= 2:
        p0, p3 = pts[0], pts[-1]
        p1 = p0 + (p3 - p0) / 3
        p2 = p0 + 2 * (p3 - p0) / 3
        return tuple(p0), tuple(p1), tuple(p2), tuple(p3)

    # Chord-length parameterization
    dists = np.sqrt(np.sum(np.diff(pts, axis=0)**2, axis=1))
    total = np.sum(dists)
    if total < 1e-10:
        return tuple(pts[0]), tuple(pts[0]), tuple(pts[-1]), tuple(pts[-1])
    t = np.zeros(n)
    t[1:] = np.cumsum(dists) / total

    p0, p3 = pts[0], pts[-1]

    # Bernstein basis matrix for interior control points
    # B(t) = (1-t)^3 * P0 + 3(1-t)^2*t * P1 + 3(1-t)*t^2 * P2 + t^3 * P3
    # Rearrange: B(t) - (1-t)^3*P0 - t^3*P3 = 3(1-t)^2*t * P1 + 3(1-t)*t^2 * P2
    A = np.zeros((n, 2))
    A[:, 0] = 3 * (1-t)**2 * t      # coefficient for P1
    A[:, 1] = 3 * (1-t) * t**2      # coefficient for P2

    # Target: actual points minus P0 and P3 contributions
    rhs = pts - np.outer((1-t)**3, p0) - np.outer(t**3, p3)

    # Solve for P1, P2 (each has x,y components)
    try:
        result_x, _, _, _ = np.linalg.lstsq(A, rhs[:, 0], rcond=None)
        result_y, _, _, _ = np.linalg.lstsq(A, rhs[:, 1], rcond=None)
        p1 = np.array([result_x[0], result_y[0]])
        p2 = np.array([result_x[1], result_y[1]])
    except np.linalg.LinAlgError:
        p1 = p0 + (p3 - p0) / 3
        p2 = p0 + 2 * (p3 - p0) / 3

    return tuple(p0), tuple(p1), tuple(p2), tuple(p3)


def fit_bezier_chain(points: list, max_error: float = 1.5, max_segment_points: int = 30) -> list:
    """Fit a chain of cubic bezier curves to a polyline.

    Splits the polyline into segments and fits a bezier to each.
    Returns list of (p0, p1, p2, p3) tuples.
    """
    if len(points) <= 2:
        return [fit_cubic_bezier(points)]

    beziers = []
    i = 0
    while i < len(points) - 1:
        # Try increasingly longer segments
        best_end = min(i + 3, len(points))
        for end in range(min(i + 3, len(points)), min(i + max_segment_points, len(points)) + 1):
            segment = points[i:end]
            if len(segment) < 2:
                break
            p0, p1, p2, p3 = fit_cubic_bezier(segment)

            # Compute max error
            pts = np.array(segment)
            n = len(pts)
            dists = np.sqrt(np.sum(np.diff(pts, axis=0)**2, axis=1))
            total = np.sum(dists)
            if total < 1e-10:
                best_end = end
                continue
            t = np.zeros(n)
            t[1:] = np.cumsum(dists) / total

            fitted = np.zeros_like(pts)
            for j, tj in enumerate(t):
                fitted[j] = ((1-tj)**3 * np.array(p0) + 3*(1-tj)**2*tj * np.array(p1) +
                            3*(1-tj)*tj**2 * np.array(p2) + tj**3 * np.array(p3))

            errors = np.sqrt(np.sum((pts - fitted)**2, axis=1))
            if np.max(errors) <= max_error:
                best_end = end
            else:
                break

        segment = points[i:best_end]
        beziers.append(fit_cubic_bezier(segment))
        i = best_end - 1  # overlap by one point for continuity

    return beziers


def smooth_polyline(points: list, window: int = 3) -> list:
    """Apply gentle smoothing to a polyline using moving average."""
    if len(points) <= window:
        return points

    pts = np.array(points)
    smoothed = pts.copy()

    # Don't smooth first and last points
    half = window // 2
    for i in range(half, len(pts) - half):
        smoothed[i] = np.mean(pts[i-half:i+half+1], axis=0)

    return [tuple(p) for p in smoothed]


def catmull_rom_to_bezier(p_prev, p0, p1, p_next, alpha=0.5):
    """Convert Catmull-Rom segment to cubic bezier control points.

    Uses centripetal parameterization (alpha=0.5) for smooth curves
    without cusps or overshooting.
    """
    p_prev, p0, p1, p_next = [np.array(p) for p in [p_prev, p0, p1, p_next]]

    d1 = np.linalg.norm(p0 - p_prev) ** alpha
    d2 = np.linalg.norm(p1 - p0) ** alpha
    d3 = np.linalg.norm(p_next - p1) ** alpha

    if d1 < 1e-10: d1 = 1.0
    if d2 < 1e-10: d2 = 1.0
    if d3 < 1e-10: d3 = 1.0

    # Catmull-Rom to Bezier conversion
    b1 = p0 + (p1 - p_prev) / (3 * (d1 + d2)) * d2
    b2 = p1 - (p_next - p0) / (3 * (d2 + d3)) * d2

    return tuple(p0), tuple(b1), tuple(b2), tuple(p1)


def points_to_catmull_rom_beziers(points: list) -> list:
    """Convert a list of points to Catmull-Rom cubic bezier chain.

    Returns list of (p0, p1, p2, p3) control point tuples.
    Catmull-Rom guarantees smooth C1 continuity and no overshooting.
    """
    n = len(points)
    if n < 2:
        return []
    if n == 2:
        return [fit_cubic_bezier(points)]

    beziers = []
    for i in range(n - 1):
        # Get the 4 points needed for Catmull-Rom
        p_prev = points[max(0, i - 1)]
        p0 = points[i]
        p1 = points[i + 1]
        p_next = points[min(n - 1, i + 2)]

        bezier = catmull_rom_to_bezier(p_prev, p0, p1, p_next)
        beziers.append(bezier)

    return beziers


def subpath_to_smooth_svg(subpath: list, rdp_epsilon: float = 0.3,
                          bezier_error: float = 0.6) -> str:
    """Convert a parsed subpath to smoothed SVG path commands."""
    polyline = subpath_to_polyline(subpath, samples_per_segment=12)
    if len(polyline) < 2:
        return ""

    is_closed = (subpath[-1][0] == 'Z')

    # Compute total path length to determine if this is a small feature
    pts_arr = np.array(polyline)
    total_length = np.sum(np.sqrt(np.sum(np.diff(pts_arr, axis=0)**2, axis=1)))

    # For small paths (like the beak), use very tight tolerances
    if total_length < 200:
        eps = rdp_epsilon * 0.3
    else:
        eps = rdp_epsilon

    # Simplify with RDP (no moving average - it shifts endpoints)
    simplified = rdp_simplify(polyline, eps)

    # Use Catmull-Rom for smooth interpolation (no overshooting)
    beziers = points_to_catmull_rom_beziers(simplified)

    # Generate SVG commands
    parts = []
    if beziers:
        p0 = beziers[0][0]
        parts.append(f"M{p0[0]:.1f} {p0[1]:.1f}")
        for p0, p1, p2, p3 in beziers:
            parts.append(f"C{p1[0]:.1f} {p1[1]:.1f} {p2[0]:.1f} {p2[1]:.1f} {p3[0]:.1f} {p3[1]:.1f}")
        if is_closed:
            parts.append("Z")

    return " ".join(parts)


def smooth_svg_path(d: str, rdp_epsilon=0.8, bezier_error=1.5) -> str:
    """Smooth an entire SVG path string."""
    subpaths = parse_path(d)
    smooth_parts = []

    for sp in subpaths:
        smooth_d = subpath_to_smooth_svg(sp, rdp_epsilon=rdp_epsilon,
                                         bezier_error=bezier_error)
        if smooth_d:
            smooth_parts.append(smooth_d)

    return " ".join(smooth_parts)


def process_svg_file(input_path: Path, output_path: Path,
                     rdp_epsilon=0.8, bezier_error=1.5):
    """Read an SVG, smooth all paths, write output."""
    content = input_path.read_text()

    def replace_path(match):
        d = match.group(1)
        smoothed = smooth_svg_path(d, rdp_epsilon=rdp_epsilon, bezier_error=bezier_error)
        return f'd="{smoothed}"'

    # Replace all d="..." path attributes (use \b word boundary to avoid matching id="...")
    output = re.sub(r'\bd="([^"]+)"', replace_path, content)
    output_path.write_text(output)

    orig_size = len(content)
    new_size = len(output)
    print(f"  {input_path.name}: {orig_size:,} -> {new_size:,} bytes ({new_size/orig_size:.1f}x)")


def main():
    repo_root = Path(__file__).resolve().parent.parent
    design_dir = repo_root / "docs" / "design"

    print("Smoothing SVG paths (parse -> resample -> RDP simplify -> bezier refit)...\n")

    # Process both the icon and logo SVGs
    for name in ["kestrel_icon", "kestrel_logo"]:
        src = design_dir / f"{name}.svg"
        dst = design_dir / f"{name}_v2.svg"
        print(f"Processing {src.name}...")
        process_svg_file(src, dst, rdp_epsilon=0.3, bezier_error=0.6)

    # Also create a favicon version from the icon
    icon_v2 = design_dir / "kestrel_icon_v2.svg"
    content = icon_v2.read_text()
    # Update viewBox for favicon (tighter crop with background)
    favicon = content.replace(
        'viewBox="100 0 760 626"',
        'viewBox="80 -10 820 646"'
    )
    # Add background rect if not present
    if '<rect' not in favicon:
        favicon = favicon.replace(
            '<g id="dark_teal">',
            '<rect x="80" y="-10" width="820" height="646" fill="#ffffff" rx="60"/>\n<g id="dark_teal">'
        )
    (design_dir / "kestrel_favicon_v2.svg").write_text(favicon)
    print(f"  Created kestrel_favicon_v2.svg from smoothed icon")

    # Render comparisons
    print("\nRendering comparisons...")
    render_comparisons(design_dir)
    print("\nDone!")


def render_comparisons(design_dir: Path):
    """Render comparison images."""
    import subprocess
    from PIL import Image, ImageDraw, ImageFont

    render_h = 800

    # Render old and new
    for name in ["kestrel_icon", "kestrel_icon_v2"]:
        src = design_dir / f"{name}.svg"
        subprocess.run(["rsvg-convert", "-h", str(render_h), str(src),
                       "-o", f"/tmp/{name}.png"], check=True)

    old = Image.open("/tmp/kestrel_icon.png").convert("RGBA")
    new = Image.open("/tmp/kestrel_icon_v2.png").convert("RGBA")
    ref = Image.open(design_dir / "kestrel_avatar_400x400.png").convert("RGBA")
    ref = ref.resize((int(ref.width * render_h / ref.height), render_h), Image.LANCZOS)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
    except Exception:
        font = ImageFont.load_default()

    gap = 20
    header = 40

    # Full comparison: Ref | Old | New
    total_w = ref.width + gap + old.width + gap + new.width
    canvas = Image.new("RGBA", (total_w, render_h + header), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    x = 0
    for label, img, color in [("Reference PNG", ref, "blue"),
                                ("Old SVG", old, "red"),
                                ("New SVG (smoothed)", new, (0, 128, 0))]:
        canvas.paste(img, (x, header), img)
        draw.text((x + 10, 8), label, fill=color, font=font)
        x += img.width + gap
    canvas.save("/tmp/logo_comparison.png")
    print(f"  Full: /tmp/logo_comparison.png")

    # Zoomed comparisons (head + wing)
    for region, crop_params in [("head", (0.50, 0.02, 0.45, 0.50)),
                                 ("wing", (0.00, 0.15, 0.45, 0.55))]:
        cx_f, cy_f, cw_f, ch_f = crop_params
        crops = {}
        for tag, img in [("old", old), ("new", new)]:
            cx = int(img.width * cx_f)
            cy = int(img.height * cy_f)
            cw = int(img.width * cw_f)
            ch = int(img.height * ch_f)
            crop = img.crop((cx, cy, cx + cw, cy + ch))
            crop = crop.resize((crop.width * 2, crop.height * 2), Image.LANCZOS)
            crops[tag] = crop

        zw = crops["old"].width + gap + crops["new"].width
        zh = max(crops["old"].height, crops["new"].height) + header
        zcanvas = Image.new("RGBA", (zw, zh), (255, 255, 255, 255))
        zd = ImageDraw.Draw(zcanvas)
        zcanvas.paste(crops["old"], (0, header))
        zd.text((10, 8), f"Old {region} (2x)", fill="red", font=font)
        zcanvas.paste(crops["new"], (crops["old"].width + gap, header))
        zd.text((crops["old"].width + gap + 10, 8), f"New {region} (2x)", fill=(0, 128, 0), font=font)
        zcanvas.save(f"/tmp/logo_{region}_comparison.png")
        print(f"  {region.title()}: /tmp/logo_{region}_comparison.png")


if __name__ == "__main__":
    main()
