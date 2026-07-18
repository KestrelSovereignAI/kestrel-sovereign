"""Host-scoped feature runtime for the multi-agent host (issue #2293, Phase 1).

The multi-agent host discovers and mounts **host-scoped features** — the SDK
:class:`~kestrel_sdk.features.host_base.HostFeature` contract — in parallel to
per-agent features, adding a host scope without touching per-agent behavior:

* :func:`discover_host_feature_classes` / :func:`instantiate_host_features` —
  discovery via the ``kestrel_sovereign.host_features`` entry-point group,
  filtered by the host manifest (``.kestrel-host-features.toml``).
* :func:`build_host_context` — the fleet-scoped ``HostContext`` (host backend +
  fleet tenant-scoped entities session factory).
* :func:`mount_host_feature_routers` — mount each router at the host root
  (``/api/<name>/…``, no agent dependency → 200, not 503).
* :func:`mount_host_feature_ui` — a distinct host-scoped console UI surface.
* :func:`start_host_features` / :func:`stop_host_features` — host lifecycle.
"""

from __future__ import annotations

from .context import (
    FLEET_TENANT_ID,
    FleetSessionFactory,
    SovereignHostContext,
    build_host_context,
)
from .discovery import (
    HOST_FEATURE_ENTRY_POINT_GROUP,
    HOST_MANIFEST_FILENAME,
    discover_host_feature_classes,
    instantiate_host_features,
    read_host_scoped_manifest,
)
from .runtime import (
    HOST_FEATURE_STATIC_ASSET_RE,
    host_feature_path_prefixes,
    is_host_feature_path,
    mount_host_feature_routers,
    mount_host_feature_ui,
    start_host_features,
    stop_host_features,
    unmount_host_features,
)
from .storage import (
    HOST_DB_PATH_ENV,
    HOST_FEATURE_DB_FILENAME,
    HostStorageError,
    host_database_path,
    prepare_host_database,
)
from .ui import compute_host_ui_manifest, host_feature_static_mounts

__all__ = [
    "FLEET_TENANT_ID",
    "FleetSessionFactory",
    "SovereignHostContext",
    "build_host_context",
    "HOST_FEATURE_ENTRY_POINT_GROUP",
    "HOST_MANIFEST_FILENAME",
    "discover_host_feature_classes",
    "instantiate_host_features",
    "read_host_scoped_manifest",
    "HOST_FEATURE_STATIC_ASSET_RE",
    "host_feature_path_prefixes",
    "is_host_feature_path",
    "mount_host_feature_routers",
    "mount_host_feature_ui",
    "start_host_features",
    "stop_host_features",
    "unmount_host_features",
    "HOST_DB_PATH_ENV",
    "HOST_FEATURE_DB_FILENAME",
    "HostStorageError",
    "host_database_path",
    "prepare_host_database",
    "compute_host_ui_manifest",
    "host_feature_static_mounts",
]
