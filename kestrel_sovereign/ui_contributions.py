"""Merge out-of-tree feature UI assets into a boot manifest (#2043, epic #2038).

A feature declares the frontend assets it ships by returning a
:class:`~kestrel_sovereign.features.base.UIContributions` from
``get_ui_contributions()``. This module turns the declarations from all
**enabled** features into:

* a list of ``(mount_path, directory)`` pairs the server mounts so a
  pip-installed, out-of-tree feature can serve its own JS/CSS without the
  assets living in core ``static/`` (see :func:`feature_static_mounts`), and
* the JSON manifest served at ``GET /api/ui/contributions`` that the frontend
  boot loader reads to dynamically ``import()`` each feature's modules in
  declared order (see :func:`compute_ui_manifest`).

Security (must not be hand-waved — see the ticket): a contribution may only
reference **same-origin** asset paths. Remote / absolute URLs (scheme-bearing
``https://…`` or protocol-relative ``//host/…``) and directory-traversal
(``..``) escapes are rejected here, server-side, so a manifest can never point
the browser at a cross-origin module. ``installed = trusted``: a feature's UI
JS runs with the same DOM/session access as any core script, which is no
greater privilege than the arbitrary Python a pip install already granted.

Bundled bridge: a small number of features still ship their JS inside core
``static/`` while their Python lives out-of-tree (voice today — its assets are
slated to move to the package "later", per the ticket). Those are declared in
:data:`BUNDLED_UI_CONTRIBUTIONS` and gated on the same capability set, so
``app.js`` no longer imports them directly and a disabled feature contributes
nothing. When a feature's JS moves into its own package the bundle entry is
deleted and the package declares ``get_ui_contributions()`` like any other.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from kestrel_sovereign.feature_registry import get_package_for_feature
from kestrel_sovereign.features.base import UIContributions
from kestrel_sovereign.ui_capabilities import compute_feature_capabilities

logger = logging.getLogger(__name__)


# Core-bundled UI contributions: features whose JS still lives in core static/
# but whose Python is an out-of-tree package. Keyed by the capability that gates
# them (ticket 03 / #2041). This is an explicit, temporary bridge — see module
# docstring. Module/css paths are root-relative (no static_dir; core already
# serves them under /js).
BUNDLED_UI_CONTRIBUTIONS: List[Dict[str, Any]] = [
    {
        "feature": "voice",
        "capability": "voice",
        "modules": ["/js/voice/boot.js"],
        "css": [],
    },
]


def _feature_mount_name(feature_key: str) -> str:
    """URL-safe single path segment for ``/features/{name}/static``."""
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", feature_key).strip("-").lower()
    return slug or "feature"


def _is_remote(path: str) -> bool:
    """True for a cross-origin / remote URL the manifest must never carry."""
    if not isinstance(path, str) or not path:
        return True
    return "://" in path or path.startswith("//")


def _has_traversal(path: str) -> bool:
    return any(part == ".." for part in re.split(r"[\\/]+", path))


def _resolve_asset_url(
    path: str, mount_path: Optional[str], *, feature: str, kind: str
) -> Optional[str]:
    """Resolve one declared asset path to a same-origin URL, or None if invalid.

    ``mount_path`` is the ``/features/{name}/static`` prefix when the feature
    declared a ``static_dir`` (paths are relative to it); ``None`` means the
    feature serves no directory and the path must already be root-relative.
    """
    if _is_remote(path):
        logger.warning(
            "Rejecting remote %s URL %r from feature %s: cross-origin assets are not allowed",
            kind, path, feature,
        )
        return None
    if _has_traversal(path):
        logger.warning(
            "Rejecting %s path %r from feature %s: directory traversal is not allowed",
            kind, path, feature,
        )
        return None

    if mount_path is None:
        # No static_dir: the path must be a root-relative same-origin path the
        # host already serves (e.g. core-bundled /js/...).
        if not path.startswith("/"):
            logger.warning(
                "Rejecting %s path %r from feature %s: a contribution without a "
                "static_dir must use a root-relative ('/'-prefixed) path",
                kind, path, feature,
            )
            return None
        return path

    # static_dir present: the path is relative to the feature's mount. A leading
    # slash would escape the mount, so reject it.
    if path.startswith("/"):
        logger.warning(
            "Rejecting %s path %r from feature %s: paths are relative to the "
            "feature's static mount and must not be absolute",
            kind, path, feature,
        )
        return None
    return f"{mount_path}/{path}"


def _feature_items(agent, *, include_disabled: bool = False) -> List[Tuple[str, UIContributions]]:
    """Yield ``(feature_key, contributions)`` for features that declare UI.

    A feature present in ``agent.features`` and not runtime-disabled is enabled.
    When ``include_disabled`` is True, runtime-disabled features are included too
    — used by the server to mount asset dirs for ALL features that declare a
    ``static_dir`` so a feature enabled at runtime serves its JS without a restart
    (the manifest at ``GET /api/ui/contributions`` still lists only enabled ones).
    """
    features = getattr(agent, "features", {}) or {}
    items: List[Tuple[str, UIContributions]] = []
    for name, feature in features.items():
        if not include_disabled and not getattr(feature, "enabled", True):
            continue
        getter = getattr(feature, "get_ui_contributions", None)
        if not callable(getter):
            # SDK-only Feature subclasses predate this hook; nothing to contribute.
            continue
        try:
            contrib = getter()
        except Exception as exc:  # noqa: BLE001 - one bad feature must not break boot
            logger.warning("Feature %s get_ui_contributions() failed: %s", name, exc)
            continue
        if contrib is None:
            continue
        if not isinstance(contrib, UIContributions):
            logger.warning(
                "Feature %s get_ui_contributions() returned %r, expected UIContributions",
                name, type(contrib).__name__,
            )
            continue
        items.append((name, contrib))
    return items


def _default_capability(feature_key: str, contrib: UIContributions) -> str:
    """Resolve the capability this contribution gates on.

    Explicit ``contrib.capability`` wins; otherwise default to the feature's
    registry name (keeps it in sync with the #2041 capability set), falling back
    to the feature class key.
    """
    if contrib.capability:
        return contrib.capability
    pkg = get_package_for_feature(feature_key)
    if pkg is not None and getattr(pkg, "name", None):
        return pkg.name
    return feature_key


def feature_static_mounts(agent, *, include_disabled: bool = False) -> List[Tuple[str, str]]:
    """Return ``(mount_path, directory)`` pairs for the server to mount.

    One per feature that declares an existing ``static_dir``. The server mounts
    each at ``mount_path`` (``/features/{name}/static``) so the feature's own
    assets are served same-origin.

    The server passes ``include_disabled=True`` so a feature that starts DISABLED
    still gets its asset dir mounted at startup. Otherwise enabling it from the
    Feature Store would surface its contribution in the manifest while its
    ``static_dir`` was never mounted — the dynamic ``import()`` would 404 and the
    tab would never appear until a restart (the runtime-enable 404, #2048).
    """
    mounts: List[Tuple[str, str]] = []
    seen: set = set()
    for feature_key, contrib in _feature_items(agent, include_disabled=include_disabled):
        if not contrib.static_dir:
            continue
        directory = Path(contrib.static_dir).expanduser()
        if not directory.is_dir():
            logger.warning(
                "Feature %s declared static_dir %r which is not a directory; skipping",
                feature_key, contrib.static_dir,
            )
            continue
        mount_path = f"/features/{_feature_mount_name(feature_key)}/static"
        if mount_path in seen:
            continue
        seen.add(mount_path)
        mounts.append((mount_path, str(directory.resolve())))
    return mounts


def compute_ui_manifest(agent) -> List[Dict[str, Any]]:
    """Build the enabled-filtered, validated UI contribution manifest.

    Each entry: ``{feature, capability, modules: [url, ...], css: [url, ...]}``.
    Returned to the frontend from ``GET /api/ui/contributions``; the boot loader
    ``import()``s each ``modules`` entry in order.
    """
    manifest: List[Dict[str, Any]] = []

    for feature_key, contrib in _feature_items(agent):
        mount_path = (
            f"/features/{_feature_mount_name(feature_key)}/static"
            if contrib.static_dir
            else None
        )
        modules = [
            url
            for m in contrib.modules
            if (url := _resolve_asset_url(m, mount_path, feature=feature_key, kind="module"))
        ]
        if not modules:
            # No loadable entry module → nothing to contribute.
            continue
        css = [
            url
            for c in contrib.css
            if (url := _resolve_asset_url(c, mount_path, feature=feature_key, kind="css"))
        ]
        manifest.append({
            "feature": feature_key,
            "capability": _default_capability(feature_key, contrib),
            "modules": modules,
            "css": css,
        })

    # Core-bundled bridge entries, gated on the live capability set so a disabled
    # feature contributes nothing.
    capabilities = compute_feature_capabilities(agent)
    for entry in BUNDLED_UI_CONTRIBUTIONS:
        capability = entry.get("capability") or entry.get("feature")
        if not capabilities.get(capability, False):
            continue
        modules = [
            url
            for m in entry.get("modules", [])
            if (url := _resolve_asset_url(m, None, feature=entry["feature"], kind="module"))
        ]
        if not modules:
            continue
        css = [
            url
            for c in entry.get("css", [])
            if (url := _resolve_asset_url(c, None, feature=entry["feature"], kind="css"))
        ]
        manifest.append({
            "feature": entry["feature"],
            "capability": capability,
            "modules": modules,
            "css": css,
        })

    return manifest
