"""
Operating Agreement Generator for Wyoming DAO LLCs.

This module re-exports from the extracted kestrel-feature-legal package.
If the package is not installed, falls back to the bundled copy under
packages/kestrel-feature-legal/.

Install the standalone package:
    pip install kestrel-feature-legal
"""

try:
    from kestrel_feature_legal.operating_agreement import generate_operating_agreement
except ImportError:
    import sys
    from pathlib import Path

    _pkg_src = Path(__file__).resolve().parents[2] / "packages" / "kestrel-feature-legal" / "src"
    if _pkg_src.exists() and str(_pkg_src) not in sys.path:
        sys.path.insert(0, str(_pkg_src))

    from kestrel_feature_legal.operating_agreement import generate_operating_agreement  # noqa: F811

__all__ = ["generate_operating_agreement"]
