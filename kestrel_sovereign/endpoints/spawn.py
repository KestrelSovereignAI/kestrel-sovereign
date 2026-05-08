"""
Spawn panel API endpoints — re-exported from kestrel-feature-spawn.

Install the standalone package:
    pip install kestrel-feature-spawn
"""

try:
    from kestrel_feature_spawn.spawn.endpoints import router
except ImportError:
    import sys
    from pathlib import Path

    # Source-clone fallback: ``parents[2]`` lands at the repo root
    # now that this module lives at ``kestrel_sovereign/endpoints/``.
    # Codex review v3 on PR #1097 caught the off-by-one.
    _pkg_src = Path(__file__).resolve().parents[2] / "kestrel-feature-spawn" / "src"
    if _pkg_src.exists() and str(_pkg_src) not in sys.path:
        sys.path.insert(0, str(_pkg_src))

    from kestrel_feature_spawn.spawn.endpoints import router  # noqa: F811

__all__ = ["router"]
