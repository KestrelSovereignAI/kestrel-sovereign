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
enabled/disabled there (issue #2293, Q for UI + enablement), and the manifest
may set the answer for host features it does not name at all (issue #3099).
"""

from __future__ import annotations

import importlib.metadata
import logging
from dataclasses import dataclass, field
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

#: Manifest table carrying host-scope policy that belongs to no single
#: ``[[feature]]`` entry.
HOST_SCOPE_TABLE = "host_features"

#: Key in :data:`HOST_SCOPE_TABLE` deciding enablement for a discovered host
#: feature the manifest never names.
HOST_SCOPE_DEFAULT_KEY = "default_enabled"


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


def default_host_manifest_path() -> Path:
    """Where this process reads the host manifest from.

    Resolved from the project dir (``KESTREL_HOME`` / marker walk-up /
    ``~/.kestrel``), **not** ``Path.cwd()``. A service launched under
    ``KESTREL_HOME``, systemd, cron, or a direct path may have a CWD that
    misses the real manifest, and reading from CWD there lets a host-disabled
    feature still mount (issue #2293 P2).

    This is the one seam a caller overrides to name a different manifest for
    the whole process.
    """
    from kestrel_sovereign.paths import project_dir

    return project_dir() / HOST_MANIFEST_FILENAME


@dataclass(frozen=True)
class HostScopedManifest:
    """Host-scope enablement policy read from the host manifest.

    ``enablement`` maps the name of each host-scoped ``[[feature]]`` entry to
    its ``enabled`` value. ``default_enabled`` answers for every discovered
    host feature the manifest does not name.
    """

    default_enabled: bool = True
    enablement: Dict[str, bool] = field(default_factory=dict)

    def is_enabled(self, *names: str) -> bool:
        """Is the host feature known by any of *names* enabled?

        The first name the manifest actually names decides. A host feature the
        manifest is silent about falls to :attr:`default_enabled`.
        """
        for name in names:
            if name in self.enablement:
                return self.enablement[name]
        return self.default_enabled


def read_host_manifest(manifest_path: Optional[Path] = None) -> HostScopedManifest:
    """Read the host-scope enablement policy from the host manifest.

    Per-entry enablement comes from every ``[[feature]]`` entry marked
    host-scoped (``host_scoped = true`` or ``scope = "host"``); such an entry
    may set ``enabled = false`` to keep a host feature installed but disabled
    at host scope.

    A missing or malformed manifest yields the permissive policy: every
    discovered host feature stays enabled. That is deliberate for a fresh
    install, where enabling nothing would be worse than enabling everything.

    A manifest that wants the opposite says so::

        [host_features]
        default_enabled = false

    which flips *absence* to disabled, so a host feature installed later does
    not start merely because nothing names it (issue #3099).
    ``default_enabled`` must be a TOML boolean; a present but non-boolean
    value is a manifest error and is read as ``false``, because honouring an
    unreadable restriction as "no restriction" is the silent widening this key
    exists to prevent.
    """
    path = manifest_path or default_host_manifest_path()
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

    default_enabled = True
    scope = data.get(HOST_SCOPE_TABLE)
    if isinstance(scope, dict) and HOST_SCOPE_DEFAULT_KEY in scope:
        declared = scope[HOST_SCOPE_DEFAULT_KEY]
        if isinstance(declared, bool):
            default_enabled = declared
        else:
            logger.warning(
                "Host manifest %s: [%s] %s must be a boolean, got %r; "
                "reading it as false rather than ignoring the restriction",
                path,
                HOST_SCOPE_TABLE,
                HOST_SCOPE_DEFAULT_KEY,
                declared,
            )
            default_enabled = False

    entries = data.get("feature", [])
    if not isinstance(entries, list):
        entries = []

    enablement: Dict[str, bool] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not name:
            continue
        host_scoped = bool(entry.get("host_scoped")) or entry.get("scope") == "host"
        if not host_scoped:
            continue
        enablement[str(name)] = bool(entry.get("enabled", True))
    return HostScopedManifest(
        default_enabled=default_enabled, enablement=enablement
    )


def read_host_scoped_manifest(manifest_path: Optional[Path] = None) -> Dict[str, bool]:
    """Host-scoped ``{name: enabled}`` entries from the host manifest.

    The per-entry half of :func:`read_host_manifest`, for callers that only
    ask about host features the manifest names. It cannot express
    ``default_enabled`` — read the whole policy when the answer for an unnamed
    host feature matters.
    """
    return dict(read_host_manifest(manifest_path).enablement)


def instantiate_host_features(
    classes: Optional[Dict[str, Type[HostFeature]]] = None,
    *,
    manifest_path: Optional[Path] = None,
) -> List[HostFeature]:
    """Instantiate the enabled discovered host features.

    Discovery gates on installation (entry points); the host manifest gates on
    *enablement*. A discovered host feature is instantiated unless the manifest
    says otherwise — either an explicit ``enabled = false`` on its host-scoped
    entry, or a manifest-level ``default_enabled = false`` that it is not
    named by. A manifest with neither enables every discovered host feature
    (back-compat / zero-config), as does no manifest at all.

    ``manifest_path`` defaults to :func:`default_host_manifest_path`.
    """
    if classes is None:
        classes = discover_host_feature_classes()

    manifest = read_host_manifest(manifest_path)

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
    "HOST_FEATURE_ENTRY_POINT_GROUP",
    "HOST_MANIFEST_FILENAME",
    "HOST_SCOPE_TABLE",
    "HOST_SCOPE_DEFAULT_KEY",
    "HostScopedManifest",
    "default_host_manifest_path",
    "discover_host_feature_classes",
    "read_host_manifest",
    "read_host_scoped_manifest",
    "instantiate_host_features",
]
