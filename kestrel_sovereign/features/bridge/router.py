"""
Bridge API Router.

This module re-exports the bridge router from the extracted
kestrel-feature-spawn package. If the package is not installed,
falls back to the bundled copy under kestrel-feature-spawn/.

Install the standalone package:
    pip install kestrel-feature-spawn
"""

try:
    from kestrel_feature_spawn.bridge.router import (
        get_router,
        _get_bridge_feature,
        _build_context_note,
    )
except ImportError:
    import sys
    from pathlib import Path

    _pkg_src = Path(__file__).resolve().parents[3] / "kestrel-feature-spawn" / "src"
    if _pkg_src.exists() and str(_pkg_src) not in sys.path:
        sys.path.insert(0, str(_pkg_src))

    from kestrel_feature_spawn.bridge.router import (  # noqa: F811
        get_router,
        _get_bridge_feature,
        _build_context_note,
    )

__all__ = ["get_router", "_get_bridge_feature", "_build_context_note"]
