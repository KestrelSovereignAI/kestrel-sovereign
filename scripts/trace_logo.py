#!/usr/bin/env python3
"""
Trace Kestrel logo PNG to SVG using color segmentation and contour detection.
Generates comparison overlays for iterative refinement.
"""

import re
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
LOGO_PATH = PROJECT_ROOT / "docs/design/KESTREL_LOGO.png"
OUTPUT_DIR = PROJECT_ROOT / "docs/design"


def load_image(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load image and return BGR and RGB versions."""
    img_bgr = cv2.imread(str(path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_bgr, img_rgb


def get_dominant_colors(img_rgb: np.ndarray, n_colors: int = 8) -> list[tuple[int, int, int]]:
    """Extract dominant colors using k-means clustering."""
    pixels = img_rgb.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    _, labels, centers = cv2.kmeans(pixels, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Count pixels per cluster
    unique, counts = np.unique(labels, return_counts=True)
    sorted_idx = np.argsort(-counts)

    colors = []
    for idx in sorted_idx:
        color = tuple(int(c) for c in centers[idx])
        colors.append(color)

    return colors


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    """Convert RGB tuple to hex string."""
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def create_color_mask(img_rgb: np.ndarray, target_color: tuple[int, int, int], tolerance: int = 30) -> np.ndarray:
    """Create binary mask for pixels close to target color."""
    target = np.array(target_color, dtype=np.float32)
    img_float = img_rgb.astype(np.float32)
    diff = img_float - target
    distance = np.sqrt(np.sum(diff ** 2, axis=2))
    mask = (distance < tolerance).astype(np.uint8) * 255
    return mask


def contours_to_svg_paths(contours: list, color_hex: str, min_area: int = 100) -> str:
    """Convert OpenCV contours to SVG path elements."""
    paths = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        # Simplify contour - use smaller epsilon for smoother curves
        epsilon = 0.002 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) < 3:
            continue

        # Build SVG path
        d = f"M{approx[0][0][0]},{approx[0][0][1]}"
        for point in approx[1:]:
            d += f"L{point[0][0]},{point[0][1]}"
        d += "Z"

        paths.append(f'<path d="{d}" fill="{color_hex}"/>')

    return "\n".join(paths)


def trace_color_layer(mask: np.ndarray, color_hex: str, min_area: int = 100) -> str:
    """Trace a binary mask to SVG paths using potrace for smooth curves."""
    # Try potrace first for better curve fitting
    result = trace_with_potrace(mask, color_hex)
    if result:
        return result

    # Fallback to OpenCV contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)

    if not contours:
        return ""

    return contours_to_svg_paths_with_hierarchy(contours, hierarchy, color_hex, min_area)


def trace_with_potrace(mask: np.ndarray, color_hex: str) -> str:
    """Use potrace for high-quality bitmap-to-vector tracing."""
    # Save mask as BMP for potrace (potrace traces black areas)
    with tempfile.NamedTemporaryFile(suffix=".bmp", delete=False) as f_bmp:
        bmp_path = f_bmp.name

    with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as f_svg:
        svg_path = f_svg.name

    try:
        height, width = mask.shape

        # Invert mask: potrace traces black on white
        mask_inv = 255 - mask
        Image.fromarray(mask_inv).save(bmp_path)

        # Run potrace with optimized settings for smooth curves
        # -t: suppress speckles of up to this size (higher = cleaner)
        # -a: corner threshold (higher = smoother, 0-1.34, default 1)
        # -O: optimization tolerance (higher = smoother curves)
        # -n: turn off curve optimization (use -O instead)
        # --unit: set output unit to 1 (pixels)
        # --alphamax: corner smoothness (0=sharp, 1.34=very smooth)
        result = subprocess.run(
            ["potrace", "-s", "--flat",
             "-t", "5", "-a", "1.34", "-O", "0.5",
             "--unit", "1",
             "-o", svg_path, bmp_path],
            check=True,
            capture_output=True,
            text=True
        )

        with open(svg_path, "r") as f:
            svg_content = f.read()

        # Extract path data from potrace output
        import re

        # Potrace uses transform="translate(0,H) scale(S,-S)" where S depends on unit
        # With --unit 1, scale should be (1,-1)

        # Find all path d attributes
        paths = re.findall(r'd="([^"]+)"', svg_content)

        if not paths:
            return ""

        # Use a group with the transform to flip Y axis correctly
        # Potrace outputs with Y=0 at bottom, SVG has Y=0 at top
        result_paths = []
        for path_d in paths:
            result_paths.append(f'<path d="{path_d}" fill="{color_hex}"/>')

        # Wrap in a group with transform to correct coordinate system
        return f'<g transform="translate(0,{height}) scale(1,-1)">\n' + "\n".join(result_paths) + '\n</g>'

    except subprocess.CalledProcessError as e:
        print(f"  Potrace failed: {e.stderr}")
        return ""
    except Exception as e:
        print(f"  Potrace error: {e}")
        return ""
    finally:
        Path(bmp_path).unlink(missing_ok=True)
        Path(svg_path).unlink(missing_ok=True)


def contours_to_svg_paths_with_hierarchy(contours: list, hierarchy: np.ndarray, color_hex: str, min_area: int = 100) -> str:
    """Convert OpenCV contours with hierarchy to SVG paths (supports holes)."""
    if hierarchy is None or len(contours) == 0:
        return contours_to_svg_paths(contours, color_hex, min_area)

    paths = []
    hierarchy = hierarchy[0]  # hierarchy is nested in an extra array

    # Process only top-level contours (parent == -1)
    for i, contour in enumerate(contours):
        # Skip if this contour has a parent (it's a hole, will be handled by parent)
        if hierarchy[i][3] != -1:
            continue

        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        # Build path for outer contour
        d = contour_to_path_d(contour)

        # Add any child contours as holes (reversed winding)
        child_idx = hierarchy[i][2]  # First child
        while child_idx != -1:
            child_contour = contours[child_idx]
            if cv2.contourArea(child_contour) >= min_area // 4:
                # Reverse the child contour to make it a hole
                d += " " + contour_to_path_d(child_contour[::-1])
            child_idx = hierarchy[child_idx][0]  # Next sibling

        paths.append(f'<path d="{d}" fill="{color_hex}" fill-rule="evenodd"/>')

    return "\n".join(paths)


def contour_to_path_d(contour: np.ndarray, epsilon_factor: float = 0.0003) -> str:
    """Convert a single contour to SVG path d attribute with smooth curves."""
    # Simplify contour minimally for smoother output
    epsilon = epsilon_factor * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    if len(approx) < 3:
        return ""

    points = [(float(p[0][0]), float(p[0][1])) for p in approx]
    n = len(points)

    if n < 4:
        # Simple polygon for small shapes
        d = f"M{points[0][0]:.1f},{points[0][1]:.1f}"
        for point in points[1:]:
            d += f" L{point[0]:.1f},{point[1]:.1f}"
        d += " Z"
        return d

    # Use Catmull-Rom to cubic Bezier conversion for smooth curves
    d = f"M{points[0][0]:.1f},{points[0][1]:.1f}"

    for i in range(n):
        p0 = points[(i - 1) % n]
        p1 = points[i]
        p2 = points[(i + 1) % n]
        p3 = points[(i + 2) % n]

        # Catmull-Rom to Bezier control points
        c1x = p1[0] + (p2[0] - p0[0]) / 6
        c1y = p1[1] + (p2[1] - p0[1]) / 6
        c2x = p2[0] - (p3[0] - p1[0]) / 6
        c2y = p2[1] - (p3[1] - p1[1]) / 6

        d += f" C{c1x:.1f},{c1y:.1f} {c2x:.1f},{c2y:.1f} {p2[0]:.1f},{p2[1]:.1f}"

    d += " Z"
    return d


def create_svg(width: int, height: int, paths: list[str], background: str = None, text_elements: str = None) -> str:
    """Create complete SVG document."""
    bg_rect = ""
    if background:
        bg_rect = f'<rect width="{width}" height="{height}" fill="{background}"/>'

    paths_content = "\n".join(p for p in paths if p)

    text_content = text_elements if text_elements else ""

    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">
{bg_rect}
{paths_content}
{text_content}
</svg>'''


def create_comparison_overlay(original_path: Path, svg_path: Path, output_path: Path, crop_height: int = None):
    """Create overlay image comparing original PNG with rendered SVG."""
    # Load original
    original = cv2.imread(str(original_path))
    h, w = original.shape[:2]

    # Crop original if needed
    if crop_height:
        original = original[:crop_height, :]
        h = crop_height

    # Try using rsvg-convert (comes with librsvg)
    rendered = None
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f_png:
        png_path = f_png.name

    try:
        result = subprocess.run(
            ["rsvg-convert", "-w", str(w), "-h", str(h), "-o", png_path, str(svg_path)],
            check=True,
            capture_output=True,
            text=True
        )
        rendered = cv2.imread(png_path)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Note: rsvg-convert not available ({e}), using placeholder")
        rendered = np.ones((h, w, 3), dtype=np.uint8) * 200  # Gray placeholder
    finally:
        Path(png_path).unlink(missing_ok=True)

    # Create side-by-side comparison
    comparison = np.hstack([original, rendered])

    # Add labels
    cv2.putText(comparison, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(comparison, "SVG Rendered", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.imwrite(str(output_path), comparison)
    print(f"Comparison saved to: {output_path}")


def analyze_colors(img_rgb: np.ndarray):
    """Analyze and print color information from the image."""
    print("\n=== Color Analysis ===")
    colors = get_dominant_colors(img_rgb, n_colors=10)
    for i, color in enumerate(colors):
        hex_color = rgb_to_hex(color)
        print(f"  {i+1}. RGB{color} = {hex_color}")
    return colors


def trace_logo(icon_only: bool = False):
    """Main function to trace the logo."""
    print(f"Loading: {LOGO_PATH}")
    img_bgr, img_rgb = load_image(LOGO_PATH)
    height, width = img_bgr.shape[:2]
    print(f"Image size: {width}x{height}")

    # Analyze colors
    colors = analyze_colors(img_rgb)

    # Based on color analysis, the actual colors in the image are:
    # #003f50 - dark teal (bird body, main text)
    # #0a2737 - darker navy (shadows)
    # #0c97c0 - bright teal/cyan (wing highlights, subtitle)
    # #e8c849 - golden yellow (eye)
    # #f9f7f2 - off-white background

    # For both icon and full logo, we only trace the bird portion
    # For full logo, we add text as SVG text elements (not traced)
    # Bird ends at roughly 68% from top
    bird_crop_height = int(height * 0.68)
    img_rgb_work = img_rgb[:bird_crop_height, :]

    if icon_only:
        work_height = bird_crop_height
    else:
        work_height = height  # Full logo height, but bird traced separately

    # Create a mask for "not background" to isolate the bird
    bg_color = (249, 247, 242)
    bg_mask = create_color_mask(img_rgb_work, bg_color, tolerance=20)
    bird_mask = 255 - bg_mask  # Invert to get bird area

    # Clean up bird mask
    kernel = np.ones((3, 3), np.uint8)
    bird_mask = cv2.morphologyEx(bird_mask, cv2.MORPH_CLOSE, kernel)
    bird_mask = cv2.morphologyEx(bird_mask, cv2.MORPH_OPEN, kernel)

    # Save bird mask for debugging
    cv2.imwrite(str(OUTPUT_DIR / "_mask_bird_silhouette.png"), bird_mask)

    # Define color regions to trace (order matters - background first, then details on top)
    color_config = [
        # (target_rgb, hex_color, tolerance, name, min_area)
        ((0, 63, 80), "#003f50", 55, "dark_teal", 30),        # Main dark teal
        ((12, 151, 192), "#0c97c0", 50, "bright_teal", 20),   # Bright teal highlights
        ((183, 194, 198), "#b7c2c6", 35, "silver", 50),       # Silver/light areas on bird
        ((232, 200, 73), "#e8c849", 40, "gold", 5),           # Golden eye
    ]

    print(f"\n=== Tracing {'Icon' if icon_only else 'Full Logo'} ===")
    print(f"Working area: {width}x{work_height}")

    all_paths = []

    for target_rgb, hex_color, tolerance, name, min_area in color_config:
        print(f"Processing {name} ({hex_color})...")

        # Create mask
        mask = create_color_mask(img_rgb_work, target_rgb, tolerance)

        # Mask to bird area only (exclude background pixels)
        mask = cv2.bitwise_and(mask, bird_mask)

        # Clean up mask with morphological operations
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Check if mask has content
        pixel_count = np.sum(mask > 0)
        if pixel_count == 0:
            print(f"  No pixels found for {name}")
            continue

        print(f"  Found {pixel_count} pixels")

        # Save mask for debugging
        mask_path = OUTPUT_DIR / f"_mask_{name}.png"
        cv2.imwrite(str(mask_path), mask)

        # Trace using OpenCV contours
        paths = trace_color_layer(mask, hex_color, min_area)
        if paths:
            all_paths.append(f'<!-- {name} -->\n<g id="{name}">\n{paths}\n</g>')
            print(f"  Added paths for {name}")

    # For full logo, add text elements instead of traced text
    text_elements = None
    if not icon_only:
        # SVG text elements for crisp text rendering
        # Positions based on original image analysis (centered at ~520, KESTREL at ~710, SOVEREIGN AI at ~810)
        text_elements = '''<!-- Text elements for crisp rendering -->
<g id="text">
  <text x="520" y="735" text-anchor="middle" font-family="Arial Black, Impact, Helvetica, sans-serif" font-size="130" font-weight="900" fill="#003f50" letter-spacing="6">KESTREL</text>
  <text x="520" y="840" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="52" font-weight="400" fill="#0c97c0" letter-spacing="16">SOVEREIGN AI</text>
</g>'''

    # Create SVG - both with transparent background
    svg_content = create_svg(width, work_height, all_paths, background=None, text_elements=text_elements)

    # Output filename
    if icon_only:
        output_path = OUTPUT_DIR / "kestrel_icon.svg"
    else:
        output_path = OUTPUT_DIR / "kestrel_logo.svg"

    with open(output_path, "w") as f:
        f.write(svg_content)

    print(f"\nSVG saved to: {output_path}")

    # Create comparison
    comparison_path = OUTPUT_DIR / f"_comparison_{'icon' if icon_only else 'logo'}.png"
    create_comparison_overlay(
        LOGO_PATH,
        output_path,
        comparison_path,
        crop_height=work_height if icon_only else None
    )

    return output_path


if __name__ == "__main__":
    import sys

    # Parse args
    icon_only = "--icon" in sys.argv
    full_only = "--full" in sys.argv

    if icon_only:
        trace_logo(icon_only=True)
    elif full_only:
        trace_logo(icon_only=False)
    else:
        # Generate both
        print("=" * 50)
        print("Generating bird icon (no text)")
        print("=" * 50)
        trace_logo(icon_only=True)

        print("\n" + "=" * 50)
        print("Generating full logo (with text)")
        print("=" * 50)
        trace_logo(icon_only=False)
