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

A manifest may also state what happens to a host feature it does **not** name,
via ``[host_features] default_enabled`` (issue #3099). Without that key —
including when there is no manifest at all — an unnamed host feature is
enabled, which is what a fresh install wants. A manifest that sets it ``false``
declares the opposite: only what this manifest names runs, so installing host
feature number seven cannot start it without an edit here that says so.
"""

from __future__ import annotations

import importlib.metadata
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Type

from kestrel_sdk.features.host_base import HostFeature

logger = logging.getLogger(__name__)

#: Entry-point group host-feature packages register under::
#:
#:     [project.entry-points."kestrel_sovereign.host_features"]
#:     FleetObservabilityHostFeature = "kestrel_claws.host:FleetObservabilityHostFeature"
HOST_FEATURE_ENTRY_POINT_GROUP = "kestrel_sovereign.host_features"

#: Default host manifest filename (shared with ``kestrel feature sync``).
HOST_MANIFEST_FILENAME = ".kestrel-host-features.toml"

#: Manifest table carrying host-scope policy that is not about one feature.
#: ``kestrel feature sync`` reads only ``[[feature]]``, so this table is inert
#: to the installer and read solely here.
HOST_SCOPE_TABLE = "host_features"

#: Key inside :data:`HOST_SCOPE_TABLE` answering "what about a host feature this
#: manifest never names?".
DEFAULT_ENABLED_KEY = "default_enabled"


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


@dataclass(frozen=True)
class HostScopedManifest:
    """The host-scope enablement a manifest declares.

    ``features`` maps a host-scoped ``[[feature]]`` entry's name to its
    ``enabled`` value. ``default_enabled`` answers the question that mapping
    cannot: what becomes of a discovered host feature the manifest never names.

    Both halves are needed because only one of them can track a set that grows.
    A manifest listing six features by name says nothing about the seventh
    someone installs tomorrow; ``default_enabled = false`` does, permanently
    (issue #3099).
    """

    default_enabled: bool = True
    features: Mapping[str, bool] = field(default_factory=dict)

    def is_enabled(self, *names: str) -> bool:
        """Whether a host feature runs, given the names it may be declared under.

        ``names`` is tried in order — a class's slug before its class name — so
        that an explicit entry under either spelling beats the default. A
        feature named by none of them inherits :attr:`default_enabled`.
        """
        for name in names:
            if name in self.features:
                return self.features[name]
        return self.default_enabled


def read_host_scoped_manifest(
    manifest_path: Optional[Path] = None,
) -> HostScopedManifest:
    """Read host-scoped enablement from the host manifest.

    Collects every manifest ``[[feature]]`` entry marked host-scoped
    (``host_scoped = true`` or ``scope = "host"``). An entry may set
    ``enabled = false`` to keep a host feature installed but disabled at host
    scope; a host-scoped entry's enablement defaults to ``True``.

    ``[host_features] default_enabled`` sets the answer for a host feature no
    entry names. It defaults to ``True``, so a missing or malformed manifest
    yields the permissive zero-config policy: everything discovered runs (see
    :func:`instantiate_host_features`).
    """
    path = manifest_path or (Path.cwd() / HOST_MANIFEST_FILENAME)
    if not path.is_file():
        return HostScopedManifest()
    try:
        try:
            import tomllib  # Python 3.11+
        except ModuleNotFoundError:  # pragma: no cover
            import tomli as tomllib  # type: ignore
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except Exception as exc:  # noqa: BLE001 - never let a bad manifest break the host
        logger.warning("Failed to read host manifest %s: %s", path, exc)
        return HostScopedManifest()

    default_enabled = _read_default_enabled(data, path)

    entries = data.get("feature", [])
    if not isinstance(entries, list):
        return HostScopedManifest(default_enabled=default_enabled)

    features: Dict[str, bool] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not name:
            continue
        host_scoped = bool(entry.get("host_scoped")) or entry.get("scope") == "host"
        if not host_scoped:
            continue
        features[str(name)] = bool(entry.get("enabled", True))
    return HostScopedManifest(default_enabled=default_enabled, features=features)


def _read_default_enabled(data: dict, path: Path) -> bool:
    """Read ``[host_features] default_enabled``, or ``True`` if it is absent.

    Only a real TOML boolean is honoured. ``bool("false")`` is ``True``, so
    coercing a string here would read a typo as the *opposite* of what it says
    — and in the permissive direction, which is the failure mode this key
    exists to remove. A malformed value is reported and the documented default
    stands, matching how this reader already treats a manifest it cannot parse.
    """
    table = data.get(HOST_SCOPE_TABLE)
    if not isinstance(table, dict) or DEFAULT_ENABLED_KEY not in table:
        return True
    value = table[DEFAULT_ENABLED_KEY]
    if isinstance(value, bool):
        return value
    logger.warning(
        "Host manifest %s: [%s] %s must be a boolean, got %r; "
        "host features not named by this manifest stay enabled",
        path,
        HOST_SCOPE_TABLE,
        DEFAULT_ENABLED_KEY,
        value,
    )
    return True


def instantiate_host_features(
    classes: Optional[Dict[str, Type[HostFeature]]] = None,
    *,
    manifest_path: Optional[Path] = None,
) -> List[HostFeature]:
    """Instantiate the enabled discovered host features.

    Discovery gates on installation (entry points); the host manifest gates on
    *enablement*. A discovered host feature is instantiated unless the manifest
    disables it — either explicitly (``enabled = false`` on its host-scoped
    entry) or by declaring ``[host_features] default_enabled = false`` and not
    naming it. When the manifest says neither, all discovered host features are
    enabled (back-compat / zero-config, and the answer for no manifest at all).
    """
    if classes is None:
        classes = discover_host_feature_classes()

    manifest = read_host_scoped_manifest(manifest_path)

    features: List[HostFeature] = []
    for cls in classes.values():
        slug = _feature_slug(cls)
        if not manifest.is_enabled(slug, cls.__name__):
            logger.info("Host feature %s disabled via host manifest; skipping", slug)
            continue
        try:
            features.append(cls())
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to instantiate host feature %s: %s", slug, exc)
    return features


__all__ = [
    "DEFAULT_ENABLED_KEY",
    "HOST_FEATURE_ENTRY_POINT_GROUP",
    "HOST_MANIFEST_FILENAME",
    "HOST_SCOPE_TABLE",
    "HostScopedManifest",
    "discover_host_feature_classes",
    "read_host_scoped_manifest",
    "instantiate_host_features",
]
