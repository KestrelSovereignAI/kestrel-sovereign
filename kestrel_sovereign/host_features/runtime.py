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

from kestrel_sovereign.features.contribution_runtime import (
    FeatureContributionRuntime,
    FeatureContributionRuntimeError,
)
from kestrel_sovereign.operator import OperatorRuntimeRegistry
from kestrel_sovereign.signals import SourceRegistry
from kestrel_sovereign.waits import WaitRegistry

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


def _host_contribution_runtime(ctx: Any) -> FeatureContributionRuntime:
    runtime = getattr(ctx, "feature_contribution_runtime", None)
    if runtime is not None:
        return runtime
    runtime = FeatureContributionRuntime(
        operator_registry=OperatorRuntimeRegistry(),
        wait_registry=WaitRegistry(),
        source_registry=SourceRegistry(),
    )
    # Retain the exact registries on Sovereign's extensible HostContext so the
    # stop boundary receives the same lifecycle capabilities.
    ctx.feature_contribution_runtime = runtime
    ctx.operator_registry = runtime.operator_registry
    ctx.wait_registry = runtime.wait_registry
    ctx.signal_registry = runtime.source_registry
    ctx.permission_defaults_registry = runtime.permission_defaults_registry
    ctx.setup_step_registry = runtime.setup_step_registry
    return runtime


async def start_host_features(
    features: List[HostFeature], ctx: Any
) -> List[HostFeature]:
    """Start features after validating the complete contribution transition.

    An ordinary feature start failure remains isolated, but its declarative
    registrations are exactly reversed before a later feature is started.
    Contribution contract and owner conflicts raise before any mutation.
    Typed collection failures likewise remain fatal at this host boundary;
    their sanitized wrapper crosses intact with the original failure chained.
    """
    runtime = _host_contribution_runtime(ctx)
    transition = runtime.prepare_transition(features)
    for rejection in transition.rejected:
        # Loud, and never silent: a host feature that did not load must not be
        # mistaken for one that loaded and did nothing (issue #2951).
        logger.error(
            "Host feature %s did not load — %s",
            rejection.feature_name,
            rejection.reason,
        )
    # Merged, not overwritten. Repeated calls against one context are
    # supported (`previously_started`), and a plain assignment made health stop
    # reporting feature A the moment a later call started feature B cleanly.
    # A prior rejection is superseded only for a feature THIS call carried —
    # its verdict now comes from this transition (#2951).
    retried = {id(feature) for feature in features}
    ctx.rejected_host_feature_contributions = tuple(
        rejection
        for rejection in getattr(ctx, "rejected_host_feature_contributions", ()) or ()
        if id(rejection.feature) not in retried
    ) + tuple(transition.rejected)
    previously_started = tuple(getattr(ctx, "started_host_features", ()))
    started: List[HostFeature] = []
    for feature, prepared_item in transition.activatable(features):
        try:
            runtime.activate(prepared_item)
        except Exception as exc:
            # A declarative commit failure is not an optional imperative
            # lifecycle failure. Reverse the already-started prefix and reject
            # the transition so callers cannot serve a partially-active set.
            await _rollback_started_host_features(started, ctx, runtime)
            ctx.started_host_features = previously_started
            raise FeatureContributionRuntimeError(
                f"host feature {feature!r} contribution activation failed"
            ) from exc
        try:
            await feature.on_host_start(ctx)
        except Exception as exc:  # noqa: BLE001 - isolate a reversible failure
            logger.warning("Host feature %s on_host_start failed: %s", feature, exc)
            try:
                runtime.deactivate(feature)
            except Exception:  # noqa: BLE001 - unsafe rollback cannot be hidden
                logger.exception(
                    "Host feature %s contribution rollback failed", feature
                )
                raise
            try:
                await feature.on_host_stop(ctx)
            except Exception:  # noqa: BLE001 - partial imperative cleanup
                logger.exception(
                    "Host feature %s partial-start cleanup failed", feature
                )
            continue
        started.append(feature)
    try:
        runtime.setup_step_registry.ordered()
    except Exception:
        # A failed member may have been the target of another member's hard
        # setup ordering constraint. That leaves no valid partial host set, so
        # reverse every successfully started member before reporting failure.
        await _rollback_started_host_features(started, ctx, runtime)
        ctx.started_host_features = previously_started
        raise
    ctx.started_host_features = (*previously_started, *started)
    return started


async def _rollback_started_host_features(
    features: List[HostFeature],
    ctx: Any,
    runtime: FeatureContributionRuntime,
) -> None:
    """Best-effort cleanup of a rejected prospective host transition."""

    for feature in reversed(features):
        try:
            await feature.on_host_stop(ctx)
        except Exception:  # noqa: BLE001 - continue exact declarative cleanup
            logger.exception("Host feature %s rollback stop failed", feature)
        try:
            runtime.deactivate(feature)
        except Exception:  # noqa: BLE001 - continue cleaning remaining owners
            logger.exception(
                "Host feature %s declarative rollback failed", feature
            )


async def stop_host_features(features: List[HostFeature], ctx: Any) -> None:
    """Stop active host features and remove their exact contribution objects."""
    runtime = getattr(ctx, "feature_contribution_runtime", None)
    requested = tuple(features)
    for feature in reversed(requested):
        if runtime is not None and not runtime.is_active(feature):
            continue
        try:
            await feature.on_host_stop(ctx)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Host feature %s on_host_stop failed: %s", feature, exc)
        finally:
            if runtime is not None:
                try:
                    runtime.deactivate(feature)
                except Exception as exc:  # noqa: BLE001 - continue stopping peers
                    logger.warning(
                        "Host feature %s contribution teardown failed: %s",
                        feature,
                        exc,
                    )
    removed_ids = {id(feature) for feature in requested}
    ctx.started_host_features = tuple(
        feature
        for feature in getattr(ctx, "started_host_features", ())
        if id(feature) not in removed_ids
    )


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
