"""
Visual Identity Feature - Companion Image Generation

This module re-exports VisualIdentityFeature from the extracted
kestrel-feature-visual package. If the package is not installed,
falls back to the bundled copy under packages/kestrel-feature-visual/.

Install the standalone package:
    pip install kestrel-feature-visual
"""

try:
    from kestrel_feature_visual.feature import VisualIdentityFeature
except ImportError:
    # Fallback: import from the bundled package source in packages/
    import importlib
    import sys
    from pathlib import Path

    _pkg_src = Path(__file__).resolve().parents[3] / "packages" / "kestrel-feature-visual" / "src"
    if _pkg_src.exists() and str(_pkg_src) not in sys.path:
        sys.path.insert(0, str(_pkg_src))

    from kestrel_feature_visual.feature import VisualIdentityFeature  # noqa: F811

__all__ = ["VisualIdentityFeature"]
