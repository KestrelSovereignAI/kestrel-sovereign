"""Aggregate host-feature UI contributions into a host-scoped console surface.

Host features return the SDK :class:`~kestrel_sdk.features.ui.UIContributions`
shape (``static_dir`` + ``modules`` + ``capability`` + ``css``) from
``get_ui_contributions()``. Unlike per-agent contributions (served under
``/features/{name}/static`` and listed at ``GET /api/ui/contributions``), host
contributions are a **distinct host-scoped surface**:

* static assets mount at ``/host/features/{slug}/static`` (host root, no agent
  prefix), and
* the manifest is served at ``GET /api/host/ui/contributions``.

Phase 1 provides the surface/mount only; panel *content* comes from host
features in later phases (issue #2293, Q2). Security mirrors the per-agent path:
only same-origin, non-traversal module paths are emitted.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from kestrel_sdk.features.host_base import HostFeature
from kestrel_sdk.features.ui import UIContributions

logger = logging.getLogger(__name__)


def host_feature_slug(feature: HostFeature) -> str:
    """URL-safe single path segment for a host feature's static mount."""
    raw = getattr(feature, "name", None) or type(feature).__name__
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", str(raw)).strip("-").lower()
    return slug or "host-feature"


def _is_remote(path: str) -> bool:
    if not isinstance(path, str) or not path:
        return True
    return "://" in path or path.startswith("//")


def _has_traversal(path: str) -> bool:
    return any(part == ".." for part in re.split(r"[\\/]+", path))


def _resolve_asset_url(
    path: str,
    mount_path: Optional[str],
    *,
    feature: str,
    kind: str,
) -> Optional[str]:
    if _is_remote(path):
        logger.warning("Rejecting remote host UI %s %r from %s", kind, path, feature)
        return None
    if _has_traversal(path):
        logger.warning("Rejecting traversal host UI %s %r from %s", kind, path, feature)
        return None
    if mount_path is None:
        if not path.startswith("/"):
            logger.warning(
                "Host feature %s %s %r without a static_dir must be root-relative",
                feature, kind, path,
            )
            return None
        return path
    if path.startswith("/"):
        logger.warning(
            "Host feature %s %s %r must be relative to its static mount",
            feature, kind, path,
        )
        return None
    return f"{mount_path}/{path}"


def _contributions(feature: HostFeature) -> Optional[UIContributions]:
    getter = getattr(feature, "get_ui_contributions", None)
    if not callable(getter):
        return None
    try:
        contrib = getter()
    except Exception as exc:  # noqa: BLE001 - one bad feature must not break boot
        logger.warning("Host feature %s get_ui_contributions() failed: %s", feature, exc)
        return None
    if contrib is None:
        return None
    if not isinstance(contrib, UIContributions):
        logger.warning(
            "Host feature %s get_ui_contributions() returned %r, expected UIContributions",
            feature, type(contrib).__name__,
        )
        return None
    return contrib


def host_feature_static_mounts(features: List[HostFeature]) -> List[Tuple[str, str]]:
    """Return ``(mount_path, directory)`` pairs to mount for host-feature UI."""
    mounts: List[Tuple[str, str]] = []
    seen: set = set()
    for feature in features:
        contrib = _contributions(feature)
        if contrib is None or not contrib.static_dir:
            continue
        directory = Path(contrib.static_dir).expanduser()
        if not directory.is_dir():
            logger.warning(
                "Host feature %s declared static_dir %r which is not a directory; skipping",
                host_feature_slug(feature), contrib.static_dir,
            )
            continue
        mount_path = f"/host/features/{host_feature_slug(feature)}/static"
        if mount_path in seen:
            continue
        seen.add(mount_path)
        mounts.append((mount_path, str(directory.resolve())))
    return mounts


def compute_host_ui_manifest(features: List[HostFeature]) -> List[Dict[str, Any]]:
    """Build the validated host-scoped UI manifest.

    Each entry: ``{feature, capability, modules: [url, ...], css: [url, ...]}``. Served at
    ``GET /api/host/ui/contributions`` for the console to ``import()`` in order.
    """
    manifest: List[Dict[str, Any]] = []
    for feature in features:
        contrib = _contributions(feature)
        if contrib is None:
            continue
        slug = host_feature_slug(feature)
        mount_path = f"/host/features/{slug}/static" if contrib.static_dir else None
        modules = [
            url
            for m in contrib.modules
            if (
                url := _resolve_asset_url(
                    m, mount_path, feature=slug, kind="module"
                )
            )
        ]
        if not modules:
            continue
        css = [
            url
            for path in contrib.css
            if (
                url := _resolve_asset_url(
                    path, mount_path, feature=slug, kind="css"
                )
            )
        ]
        manifest.append(
            {
                "feature": slug,
                "capability": contrib.capability,
                "modules": modules,
                "css": css,
            }
        )
    return manifest


__all__ = [
    "host_feature_slug",
    "host_feature_static_mounts",
    "compute_host_ui_manifest",
]
