"""Discover host-scoped features (SDK ``HostFeature``) for the multi-agent host.

Host features are the fleet-scoped sibling of per-agent ``Feature``s. Where a
per-agent ``Feature`` is discovered from the ``kestrel_sovereign.features``
entry-point group and mounted under an agent prefix behind a ``get_agent``
dependency, a **host feature** is discovered here from a distinct
``kestrel_sovereign.host_features`` entry-point group and mounted at the host
root with no agent binding (see :mod:`kestrel_sovereign.host_features.runtime`).

This mirrors :func:`kestrel_sovereign.features.discover_entrypoint_feature_classes`
but for the host scope, and layers on the host manifest
(``.kestrel-host-features.toml``): a host feature may be marked host-scoped and
enabled/disabled there (issue #2293, Q for UI + enablement).
"""

from __future__ import annotations

import importlib.metadata
import logging
from pathlib import Path
from typing import Dict, List, Optional, Type

from kestrel_sdk.features.host_base import HostFeature

logger = logging.getLogger(__name__)

#: Entry-point group host-feature packages register under::
#:
#:     [project.entry-points."kestrel_sovereign.host_features"]
#:     FleetObservabilityHostFeature = "kestrel_claws.host:FleetObservabilityHostFeature"
HOST_FEATURE_ENTRY_POINT_GROUP = "kestrel_sovereign.host_features"

#: Default host manifest filename (shared with ``kestrel feature sync``).
HOST_MANIFEST_FILENAME = ".kestrel-host-features.toml"


def discover_host_feature_classes() -> Dict[str, Type[HostFeature]]:
    """Discover ``HostFeature`` subclasses registered via entry points.

    Returns a mapping of class name → class. Mirrors per-agent feature
    discovery: an entry that does not resolve to a concrete ``HostFeature``
    subclass is logged and skipped rather than aborting discovery.
    """
    classes: Dict[str, Type[HostFeature]] = {}
    try:
        eps = importlib.metadata.entry_points()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read host-feature entry_points: %s", exc)
        return classes

    # Python 3.12+ returns SelectableGroups; older returns a dict.
    if hasattr(eps, "select"):
        host_eps = eps.select(group=HOST_FEATURE_ENTRY_POINT_GROUP)
    else:  # pragma: no cover - legacy Python
        host_eps = eps.get(HOST_FEATURE_ENTRY_POINT_GROUP, [])

    for ep in host_eps:
        try:
            cls = ep.load()
        except Exception as exc:  # noqa: BLE001 - one bad package must not break the host
            logger.warning("Failed to load host feature entry point %r: %s", ep.name, exc)
            continue
        if not (
            isinstance(cls, type)
            and issubclass(cls, HostFeature)
            and cls is not HostFeature
        ):
            logger.warning(
                "Host-feature entry point %r does not point to a HostFeature subclass; skipping",
                ep.name,
            )
            continue
        classes[cls.__name__] = cls
        logger.info("Discovered host feature: %s from %s", cls.__name__, ep.value)

    return classes


def _feature_slug(cls: Type[HostFeature]) -> str:
    """Stable slug for a host-feature class (its ``name`` attribute)."""
    return getattr(cls, "name", None) or cls.__name__


def read_host_scoped_manifest(manifest_path: Optional[Path] = None) -> Dict[str, bool]:
    """Read host-scoped enablement from the host manifest.

    Returns a mapping ``{slug: enabled}`` for every manifest ``[[feature]]``
    entry marked host-scoped (``host_scoped = true`` or ``scope = "host"``).
    An entry may set ``enabled = false`` to keep a host feature installed but
    disabled at host scope; enablement defaults to ``True``.

    A missing or malformed manifest yields an empty mapping (all discovered
    host features stay enabled by default — see :func:`instantiate_host_features`).
    """
    path = manifest_path or (Path.cwd() / HOST_MANIFEST_FILENAME)
    if not path.is_file():
        return {}
    try:
        try:
            import tomllib  # Python 3.11+
        except ModuleNotFoundError:  # pragma: no cover
            import tomli as tomllib  # type: ignore
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except Exception as exc:  # noqa: BLE001 - never let a bad manifest break the host
        logger.warning("Failed to read host manifest %s: %s", path, exc)
        return {}

    entries = data.get("feature", [])
    if not isinstance(entries, list):
        return {}

    result: Dict[str, bool] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not name:
            continue
        host_scoped = bool(entry.get("host_scoped")) or entry.get("scope") == "host"
        if not host_scoped:
            continue
        result[str(name)] = bool(entry.get("enabled", True))
    return result


def instantiate_host_features(
    classes: Optional[Dict[str, Type[HostFeature]]] = None,
    *,
    manifest_path: Optional[Path] = None,
) -> List[HostFeature]:
    """Instantiate the enabled discovered host features.

    Discovery gates on installation (entry points); the host manifest gates on
    *enablement*. A discovered host feature is instantiated unless the manifest
    explicitly disables it (``enabled = false`` on its host-scoped entry). When
    the manifest contains no host-scoped entries, all discovered host features
    are enabled (back-compat / zero-config).
    """
    if classes is None:
        classes = discover_host_feature_classes()

    manifest = read_host_scoped_manifest(manifest_path)

    features: List[HostFeature] = []
    for cls in classes.values():
        slug = _feature_slug(cls)
        if manifest.get(slug, manifest.get(cls.__name__, True)) is False:
            logger.info("Host feature %s disabled via host manifest; skipping", slug)
            continue
        try:
            features.append(cls())
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to instantiate host feature %s: %s", slug, exc)
    return features


__all__ = [
    "HOST_FEATURE_ENTRY_POINT_GROUP",
    "HOST_MANIFEST_FILENAME",
    "discover_host_feature_classes",
    "read_host_scoped_manifest",
    "instantiate_host_features",
]
