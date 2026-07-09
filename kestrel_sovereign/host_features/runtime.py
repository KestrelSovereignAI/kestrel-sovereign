"""Mount host-feature routers/UI and drive their host-scoped lifecycle.

This is the framework wiring the SDK contract deliberately leaves out: given a
list of :class:`~kestrel_sdk.features.host_base.HostFeature` instances, mount
each ``get_router()`` at the **host root** (no agent prefix, no ``get_agent``
dependency — so host paths return 200, not the 503 "Agent not initialized" that
un-prefixed agent paths hit), aggregate their UI at a host-scoped surface, and
run ``on_host_start`` / ``on_host_stop`` at host lifecycle boundaries.
"""

from __future__ import annotations

import logging
import re
from typing import Any, List

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from kestrel_sdk.features.host_base import HostFeature

from .ui import compute_host_ui_manifest, host_feature_static_mounts

logger = logging.getLogger(__name__)

#: Auth-exemption regex for host-feature UI static assets. Header-less browser
#: mechanisms (``<link href>`` / ``import()``) can't attach the API-key header,
#: so their static tree bypasses API-key auth — the ``/static/`` segment keeps
#: any host-feature *API* route protected. Mirrors ``FEATURE_STATIC_ASSET_RE``.
HOST_FEATURE_STATIC_ASSET_RE = re.compile(r"^/host/features/[^/]+/static/")


def _route_prefix(path: str) -> str:
    """Static path prefix identifying a host-feature route.

    ``/api/fleet/runs/{id}`` → ``/api/fleet`` (first two segments for ``/api/*``
    routers, which mount at ``/api/<name>/…``); ``/fleet/x`` → ``/fleet``. Used
    to scope CSRF enforcement to host-feature paths only, so per-agent proxy
    paths (``/api/agents/{id}/…``) are untouched.
    """
    segments = [s for s in path.split("/") if s]
    if not segments:
        return "/"
    if segments[0] == "api" and len(segments) >= 2:
        return "/api/" + segments[1]
    return "/" + segments[0]


def host_feature_path_prefixes(app: FastAPI) -> set:
    """Return the set of host-feature route prefixes recorded at mount time."""
    return set(getattr(app.state, "_host_feature_prefixes", set()) or set())


def is_host_feature_path(app: FastAPI, path: str) -> bool:
    """Whether ``path`` belongs to a mounted host-feature router."""
    prefix = _route_prefix(path)
    return prefix in host_feature_path_prefixes(app)


def mount_host_feature_routers(app: FastAPI, features: List[HostFeature]) -> List[Any]:
    """Mount each host feature's router at the host root; return added routes.

    A host feature serving ``/api/<name>/…`` mounts with NO prefix and NO
    ``get_agent`` dependency, so those paths resolve at host scope and return
    200 (given valid host auth), never 503.
    """
    added: List[Any] = []
    prefixes: set = set()
    mounted_names: List[str] = []
    for feature in features:
        getter = getattr(feature, "get_router", None)
        if not callable(getter):
            continue
        try:
            router = getter()
        except Exception as exc:  # noqa: BLE001 - one bad feature must not break the host
            logger.warning("Host feature %s get_router() failed: %s", feature, exc)
            continue
        if router is None:
            continue
        before = len(app.routes)
        try:
            app.include_router(router)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to mount host feature router %s: %s", feature, exc)
            continue
        new_routes = app.routes[before:]
        added.extend(new_routes)
        for route in new_routes:
            route_path = getattr(route, "path", None)
            if route_path:
                prefixes.add(_route_prefix(route_path))
        mounted_names.append(getattr(feature, "name", type(feature).__name__))

    app.state._host_feature_routes = added
    app.state._host_feature_prefixes = prefixes
    if mounted_names:
        logger.info("Mounted host feature routers: %s", ", ".join(mounted_names))
    return added


def mount_host_feature_ui(app: FastAPI, features: List[HostFeature]) -> List[Any]:
    """Mount host-feature static dirs and record the host UI manifest.

    Static assets mount at ``/host/features/{slug}/static``; the manifest is
    stashed on ``app.state.host_ui_manifest`` for ``GET
    /api/host/ui/contributions`` to serve.
    """
    added: List[Any] = []
    for mount_path, directory in host_feature_static_mounts(features):
        try:
            app.mount(
                mount_path,
                StaticFiles(directory=directory),
                name=f"host-feature-ui:{mount_path}",
            )
            added.append(app.routes[-1])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to mount host feature UI at %s: %s", mount_path, exc)

    app.state._host_feature_ui_mounts = added
    app.state.host_ui_manifest = compute_host_ui_manifest(features)
    if added:
        logger.info("Mounted host feature UI dirs: %s", ", ".join(m for m, _ in host_feature_static_mounts(features)))
    return added


def unmount_host_features(app: FastAPI) -> None:
    """Remove host-feature routes/mounts (idempotent; safe across restarts)."""
    for attr in ("_host_feature_routes", "_host_feature_ui_mounts"):
        routes = getattr(app.state, attr, None) or []
        for route in routes:
            try:
                app.routes.remove(route)
            except ValueError:
                pass
        setattr(app.state, attr, [])
    app.state._host_feature_prefixes = set()
    app.state.host_ui_manifest = []


async def start_host_features(features: List[HostFeature], ctx: Any) -> None:
    """Run ``on_host_start`` for every host feature (isolating failures)."""
    for feature in features:
        try:
            await feature.on_host_start(ctx)
        except Exception as exc:  # noqa: BLE001 - one feature's start must not abort the host
            logger.warning("Host feature %s on_host_start failed: %s", feature, exc)


async def stop_host_features(features: List[HostFeature], ctx: Any) -> None:
    """Run ``on_host_stop`` for every host feature (isolating failures)."""
    for feature in features:
        try:
            await feature.on_host_stop(ctx)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Host feature %s on_host_stop failed: %s", feature, exc)


__all__ = [
    "HOST_FEATURE_STATIC_ASSET_RE",
    "mount_host_feature_routers",
    "mount_host_feature_ui",
    "unmount_host_features",
    "start_host_features",
    "stop_host_features",
    "host_feature_path_prefixes",
    "is_host_feature_path",
]
