#!/usr/bin/env python3
"""
A FastAPI server to expose Kestrel agent functionality as a service.
"""
import argparse
import asyncio
import logging
import os
import re
import secrets
from collections.abc import Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute, APIWebSocketRoute
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from starlette.routing import (
    Match,
    Route as StarletteRoute,
    WebSocketRoute as StarletteWebSocketRoute,
)
from kestrel_sovereign.main import get_agent_did_async
from kestrel_sovereign.kestrel_agent import (
    KestrelAgent,
    await_agent_shutdown_completion,
    await_lifecycle_task_completion,
)
from kestrel_sovereign.lifecycle_checks import verify_identity_isolation
from kestrel_sovereign.llm.service import LLMService
from dotenv import load_dotenv
from slowapi.errors import RateLimitExceeded
from kestrel_sovereign.rate_limit import limiter
from kestrel_sovereign.security.bootstrap_access import is_bootstrap_host_allowed
from kestrel_sovereign.api_errors import (
    api_error_response,
    api_unhandled_exception_handler,
    register_api_error_handlers,
)

from kestrel_sovereign.kestrel_config.constants import SHUTDOWN_TIMEOUT
from kestrel_sovereign.telemetry import setup_tracing

# Load environment variables from .env file.
# override=False: Don't clobber env vars already set by ProcessManager
# (e.g., KESTREL_DB_PATH is set per-agent in multi_agent mode). With
# override=False the FIRST source to define a key wins, so order is
# custody-critical: the resolved project home must be consulted BEFORE the
# current directory (#2468). A stray source-checkout ``.env`` in CWD must never
# shadow the KESTREL_DATA_KEY the agent's identity was encrypted under, or the
# server would boot with a key that cannot decrypt its own memory.
#
# Resolution order (first defined wins):
#   1. <project_dir>/.env — the resolved project home (KESTREL_HOME, marker
#      walk-up, or ~/.kestrel for pip-installed users). This is the home whose
#      persisted KESTREL_DATA_KEY the identity was encrypted under; authoritative.
#   2. CWD/.env — the directory the operator happened to launch from; only fills
#      in keys the home did not define.
#   3. <package-dir>/.env — legacy: source clones where someone dropped a .env
#      next to the package source.
# python-dotenv silently no-ops on missing files, so all three calls are safe.
try:
    from kestrel_sovereign.paths import project_dir as _resolve_project_dir
    load_dotenv(_resolve_project_dir() / ".env", override=False)
except Exception:
    # Resolver should never raise, but a .env load is best-effort: if
    # this somehow blows up we want the server to keep booting.
    pass
load_dotenv(Path.cwd() / ".env", override=False)
load_dotenv(Path(__file__).parent / ".env", override=False)

from kestrel_sovereign.logging_config import (
    setup_logging,
    correlation_id_var,
    session_id_var,
    agent_name_var,
    resolve_correlation_id,
)

setup_logging()
logger = logging.getLogger(__name__)

# Security Configuration
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
security = HTTPBearer(auto_error=False)

# Auth-exempt feature UI static assets, matched precisely to the mount shape
# (/features/{slug}/static/…). Anchored + single-segment so a feature *API*
# route that merely contains a later "static" segment (e.g.
# /features/foo/api/static/secret) stays protected.
#
# The optional /api/agents/{id} prefix matters: in multi-agent host mode
# server:app itself owns the agents (app.state.agent_manager) and the browser
# hits /api/agents/{id}/features/{slug}/static/…. auth_middleware runs BEFORE
# MultiAgentAgentRoutingMiddleware strips that prefix (the strip middleware is
# innermost — added earliest — so it runs LAST), so auth sees the UN-stripped
# path here and the exemption must match it. In the separate-subprocess host.py
# topology the host strips the prefix and forwards /features/… to the agent, so
# the un-prefixed form is matched instead. This regex is identical to host.py's
# FEATURE_STATIC_ASSET_RE and covers both.
FEATURE_STATIC_ASSET_RE = re.compile(r"^(?:/api/agents/[^/]+)?/features/[^/]+/static/")

# Webhook dispatch paths that bypass API-key auth. Webhooks authenticate
# themselves (HMAC signature / bearer token) — an external caller (Stripe,
# GitHub, …) has no host API key. Matches the bare ``/webhooks/{name}`` form AND
# the per-agent ``/api/agents/{id}/webhooks/{name}`` form so an external, keyless
# caller can address a SPECIFIC agent's webhook on a heterogeneous multi-agent
# host (#2091/P3). Only webhook *dispatch* is exempted — webhook management is
# command-only (``!webhooks``), so no HTTP admin route is opened. (Moved here
# from the retired ``kestrel_sovereign.host`` when the legacy proxy host was
# consolidated onto ``server:app`` — issue #2382.)
WEBHOOK_PATH_RE = re.compile(r"^(?:/api/agents/[^/]+)?/webhooks/")

# Paths where API key query parameter auth is allowed
# (EventSource/SSE can't send headers, so these endpoints need query param auth).
# Both the canonical /api/agent/* paths and the deprecated /agent/* paths are
# allowed during the back-compat window (#871).
SSE_PATHS = {
    "/api/agent/notifications/sse",
    "/api/agent/stream",
    "/agent/notifications/sse",
    "/agent/stream",
}


def resolve_multi_agent_path(env: dict | os._Environ) -> Path:
    """Compute the multi_agent.toml path the lifespan should load (#868).

    Centralised so unit tests can exercise the real decision logic
    instead of reimplementing it locally — the bug class this guards
    against (a demo run silently mounting live agents) is exactly the
    kind of thing where test-and-prod-drift is dangerous.

    Decision matrix:

    ============================  =====================================
     Inputs                        Result
    ============================  =====================================
     KESTREL_MULTI_AGENT_CONFIG    Honour it verbatim (operator opted in
       set                          to a specific path)
     KESTREL_DEMO_SERVER=1 + no    Refuse to auto-mount the project-root
       explicit config + the       ``multi_agent.toml``.  Returns a path that
       default ``multi_agent.toml``    does not exist so the lifespan skips
       exists at the project       multi-agent setup.  This is the
       root                        guard that would have stopped the
                                   #867 wipe.
     anything else                 Use the default path (the lifespan's
                                   ``.exists()`` check handles missing
                                   files; production behaviour preserved).
    ============================  =====================================

    Args:
        env: Mapping of environment variables.  Pass ``os.environ`` in
            production; tests pass a plain ``dict``.

    Returns:
        :class:`pathlib.Path` the lifespan should attempt to load.  The
        caller still uses ``.exists()`` to decide whether to enter
        multi-agent mode.
    """
    multi_agent_path = Path(env.get("KESTREL_MULTI_AGENT_CONFIG", "multi_agent.toml"))
    demo_server_env = env.get("KESTREL_DEMO_SERVER", "").lower() in (
        "1", "true", "yes",
    )
    multi_agent_explicit = "KESTREL_MULTI_AGENT_CONFIG" in env
    if demo_server_env and not multi_agent_explicit and multi_agent_path.exists():
        logger.warning(
            "[demo-server] KESTREL_DEMO_SERVER=1 with no explicit "
            "KESTREL_MULTI_AGENT_CONFIG — refusing to auto-mount %s.  "
            "A demo server must not silently load live agents.  Pass "
            "KESTREL_MULTI_AGENT_CONFIG=<path> explicitly to opt in.",
            multi_agent_path,
        )
        return Path("/dev/null/multi_agent-disabled")
    return multi_agent_path


def _set_startup_error(app: FastAPI, error: Optional[Exception]) -> None:
    """Persist startup failure state for diagnostics and health endpoints."""
    app.state.startup_error = str(error) if error else None


def _mandatory_feature_failure_record(
    agent_name: str,
    error: Exception,
) -> Optional[dict]:
    """Return a health-safe record for a typed sovereignty boot failure."""
    from kestrel_sovereign.features import MandatoryFeatureReadinessError

    if not isinstance(error, MandatoryFeatureReadinessError):
        return None
    return {
        "agent": agent_name,
        "feature": error.feature_name,
        "stage": error.stage,
        "error": str(error),
    }


def _identity_readiness_failure_record(
    agent_name: str,
    error: Exception,
) -> Optional[dict]:
    """Return a health-safe record for a blocked identity root of trust."""
    from kestrel_sovereign.identity.runtime_identity import (
        IdentityReadinessError,
    )

    if not isinstance(error, IdentityReadinessError):
        return None
    return {
        "agent": agent_name,
        "state": "blocked",
        "stage": error.stage,
        "failure": error.failure,
        "error_code": error.error_code,
        "cause_type": error.cause_type,
        "error": str(error),
    }


def _scheduler_readiness_failure_record(
    scope: str,
    error: BaseException,
    *,
    agent_name: Optional[str] = None,
) -> dict:
    """Return a non-sensitive scheduler readiness failure record.

    Scheduler identity/database errors often carry a local path or connection
    string.  They belong in logs, not in health output.  The authenticated
    endpoint receives the affected configured agent (when known), a stable
    operator action code, and exception class; public readiness receives only
    the aggregate 503 contract.
    """
    record = {
        "scope": scope,
        "state": "unavailable",
        "error_code": f"scheduler_{scope}_unavailable",
        "cause_type": type(error).__name__,
    }
    if agent_name is not None:
        record["agent"] = agent_name
    return record


def _latch_scheduler_readiness_failure(
    app: FastAPI,
    scope: str,
    error: BaseException,
    *,
    agent_name: Optional[str] = None,
) -> None:
    """Latch a scheduler safety outage until a new host lifecycle starts."""
    record = _scheduler_readiness_failure_record(
        scope,
        error,
        agent_name=agent_name,
    )
    failures = list(getattr(app.state, "scheduler_readiness_failures", ()))
    is_new = record not in failures
    if is_new:
        failures.append(record)
        app.state.scheduler_readiness_failures = failures
        logger.error(
            "Scheduler readiness failure (scope=%s, agent=%s, cause=%s)",
            scope,
            agent_name,
            type(error).__name__,
            exc_info=(type(error), error, error.__traceback__),
        )


def _is_enabled_scheduler_feature(name: object, feature: object) -> bool:
    """Return whether a feature-map entry is an active scheduler feature."""

    return (
        name == "SchedulerFeature"
        or type(feature).__name__ == "SchedulerFeature"
    ) and bool(getattr(feature, "enabled", True))


def _latch_active_scheduler_runner_failures(
    app: FastAPI,
    agent,
    manager,
) -> None:
    """Promote live runner safety latches into the host readiness contract.

    A shared-PostgreSQL host owns a fleet runner, but each loaded
    ``SchedulerFeature`` may also have a scoped runner while the host is
    coming up or while a deployment topology changes.  Those runners cannot
    receive the app-owned callback at construction time.  Poll their explicit
    ``readiness_failure`` latch at the health boundary and persist only a
    sanitized record; a scheduler protocol/database outage must not look
    healthy merely because a sibling agent remains routable.
    """

    candidates: list[tuple[str, object]] = []
    if agent is not None:
        candidates.append(("default", agent))
    if manager is not None:
        try:
            managed = manager.list_agents()
        except Exception:  # pragma: no cover - health must not crash
            logger.warning("Unable to inspect managed scheduler readiness", exc_info=True)
            managed = {}
        if isinstance(managed, dict):
            candidates.extend(
                (name, candidate)
                for name, candidate in managed.items()
                if isinstance(name, str)
            )

    for agent_name, candidate in candidates:
        try:
            features = getattr(candidate, "features", None)
        except Exception:  # pragma: no cover - health must not crash
            logger.warning(
                "Unable to inspect scheduler features for %s",
                agent_name,
                exc_info=True,
            )
            continue
        if not isinstance(features, dict):
            continue
        for name, feature in features.items():
            try:
                if not _is_enabled_scheduler_feature(name, feature):
                    continue
                runner = getattr(feature, "_runner", None)
                failure = getattr(runner, "readiness_failure", None)
            except Exception:  # pragma: no cover - health must not crash
                logger.warning(
                    "Unable to inspect scheduler runner for %s",
                    agent_name,
                    exc_info=True,
                )
                continue
            if isinstance(failure, BaseException):
                _latch_scheduler_readiness_failure(
                    app,
                    "runtime",
                    failure,
                    agent_name=agent_name,
                )

    host_runner = getattr(app.state, "host_scheduler_runner", None)
    host_failure = getattr(host_runner, "readiness_failure", None)
    if isinstance(host_failure, BaseException):
        _latch_scheduler_readiness_failure(app, "protocol", host_failure)


def _active_scheduler_workers_available(app: FastAPI, agent, manager) -> bool:
    """Return false when an enabled scheduler lacks its topology's live worker."""

    runners = []
    host_worker_required = False
    candidates = [agent] if agent is not None else []
    if manager is not None:
        try:
            candidates.extend(manager.list_agents().values())
        except Exception:  # pragma: no cover - public health must not crash
            return False
    for candidate in candidates:
        try:
            features = getattr(candidate, "features", None)
        except Exception:  # pragma: no cover - public health must not crash
            logger.warning(
                "Unable to inspect scheduler worker availability",
                exc_info=True,
            )
            return False
        if not isinstance(features, dict):
            continue
        for name, feature in features.items():
            try:
                if not _is_enabled_scheduler_feature(name, feature):
                    continue
                runner = getattr(feature, "_runner", None)
                if runner is None:
                    if getattr(feature, "_polling_managed_by_host", False) is True:
                        host_worker_required = True
                        continue
                    return False
                if not hasattr(runner, "worker_available"):
                    return False
                runners.append(runner)
            except Exception:  # pragma: no cover - public health must not crash
                logger.warning(
                    "Unable to inspect scheduler worker availability",
                    exc_info=True,
                )
                return False
    try:
        host_runner = getattr(app.state, "host_scheduler_runner", None)
        if host_runner is None:
            if host_worker_required:
                return False
        else:
            if not hasattr(host_runner, "worker_available"):
                return False
            runners.append(host_runner)
        return all(
            getattr(runner, "worker_available", False) for runner in runners
        )
    except Exception:  # pragma: no cover - public health must not crash
        logger.warning(
            "Unable to inspect scheduler worker availability",
            exc_info=True,
        )
        return False


def _constitution_safe_mode_record(agent_name: str, agent) -> Optional[dict]:
    """Return a controlled operator-readiness record for a restricted agent."""
    safe_mode = getattr(agent, "_safe_mode", False) is True
    audit_pending = getattr(agent, "_constitution_audit_pending", False) is True
    if not safe_mode and not audit_pending:
        return None
    record = {
        "agent": agent_name,
        "state": "safe_mode" if safe_mode else "audit_pending",
        "error_code": (
            "constitution_safe_mode"
            if safe_mode
            else "constitution_audit_pending"
        ),
    }
    entered_at = getattr(agent, "_safe_mode_entered_at", None)
    if isinstance(entered_at, datetime):
        record["entered_at"] = entered_at.isoformat()
    if audit_pending and not safe_mode:
        record["failure"] = "startup_audit_required"
    elif getattr(agent, "_constitution_state_load_error", None):
        record["failure"] = "state_unavailable"
    else:
        record["failure"] = "integrity_restriction"
    return record


def _constitution_safe_mode_records(agent, manager) -> list[dict]:
    """Collect restricted agents without leaking stored integrity reasons."""
    records: list[dict] = []
    seen: set[int] = set()
    if agent is not None:
        seen.add(id(agent))
        record = _constitution_safe_mode_record(
            getattr(agent, "_agent_name", None) or "default", agent
        )
        if record is not None:
            records.append(record)
    if manager is not None:
        for name, managed_agent in manager.list_agents().items():
            if id(managed_agent) in seen:
                continue
            seen.add(id(managed_agent))
            record = _constitution_safe_mode_record(name, managed_agent)
            if record is not None:
                records.append(record)
    return records


def _oauth_required() -> bool:
    """Return whether OAuth is the required auth mode.

    Set KESTREL_REQUIRE_OAUTH=true in Cloud Run deploy scripts to force
    Google sign-in. When false (default, local dev), API key bootstrap
    is available and the frontend won't redirect to OAuth.

    This is the single source of truth for auth mode.
    """
    return os.environ.get("KESTREL_REQUIRE_OAUTH", "").lower() in {
        "1", "true", "yes", "on"
    }


def _bootstrap_key_enabled() -> bool:
    """Localhost API-key bootstrap is available when OAuth is not required."""
    return not _oauth_required()


def get_api_key():
    """Get or generate the API key."""
    api_key = os.environ.get("KESTREL_API_KEY")
    if not api_key:
        generated_key = secrets.token_urlsafe(32)
        os.environ["KESTREL_API_KEY"] = generated_key
        logger.warning("⚠️  NO KESTREL_API_KEY SET. A temporary key has been generated.")
        logger.warning("Please set KESTREL_API_KEY in your environment for persistence.")
        return generated_key
    # Strip surrounding quotes (Docker --env-file includes them literally)
    if len(api_key) >= 2 and api_key[0] == api_key[-1] and api_key[0] in ('"', "'"):
        api_key = api_key[1:-1]
    return api_key


async def verify_api_key(
    request: Request,
    api_key_header: Optional[str] = Security(api_key_header),
    token: Optional[HTTPAuthorizationCredentials] = Security(security)
):
    """Verify the API key from Header or Bearer token or query parameter.

    Note: This dependency is primarily used for OpenAPI documentation.
    The actual auth is handled by auth_middleware which supports query params for SSE.
    """
    if request.url.path == "/health":
        return True
    if SERVE_UI and request.url.path.startswith("/static"):
        return True

    expected_key = get_api_key()

    if api_key_header and secrets.compare_digest(api_key_header, expected_key):
        return True
    if token and secrets.compare_digest(token.credentials, expected_key):
        return True

    # Support query parameter auth for SSE endpoints only (EventSource can't send headers)
    # Restricted to SSE_PATHS to avoid leaking keys in URL logs on other endpoints
    api_key_query = request.query_params.get("api_key")
    if api_key_query and request.url.path in SSE_PATHS:
        if secrets.compare_digest(api_key_query, expected_key):
            return True

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API Key",
        headers={"WWW-Authenticate": "Bearer"},
    )


def _is_webhook_receiver(receiver) -> bool:
    """Duck-typed webhook-receiver check (keeps core decoupled from the class)."""
    return (
        receiver is not None
        and hasattr(receiver, "handle_webhook")
        and hasattr(receiver, "webhooks")
    )


def _resolve_route_agent(scope, mount_agent):
    """Resolve which agent owns the request hitting a gated feature route.

    Prefers the per-request agent the multi-agent routing middleware attached
    to ``scope["state"]`` (so ``/api/agents/{name}/...`` resolves to the RIGHT
    agent), and falls back to the agent captured when the route was mounted
    (single-agent mode, where no per-request agent is set).
    """
    state = scope.get("state") or {}
    return state.get("agent") or mount_agent


def _feature_router_route_selector(router, route_index: int) -> tuple:
    """Describe one child route without retaining its instance-bound callable.

    ``FastAPI.include_router`` copies a route's endpoint at mount time.  That
    is fine for module-level handlers, but is wrong for a per-agent feature
    whose route closes over ``self``: a route first mounted for Alice would
    otherwise call Alice's feature when a request is addressed to Bob.  Keep
    only structural coordinates here and resolve Bob's current child route at
    request time.
    """

    routes = tuple(getattr(router, "routes", ()) or ())
    if route_index < 0 or route_index >= len(routes):
        raise IndexError("feature router child route index is out of range")
    route = routes[route_index]
    return (
        getattr(router, "prefix", ""),
        route_index,
        type(route).__name__,
        getattr(route, "path", None),
        tuple(sorted(getattr(route, "methods", ()) or ())),
    )


def _current_feature_router_route(feature, selector: tuple):
    """Resolve the request target's current matching child route, or ``None``.

    A replacement feature is eligible only if it still exposes exactly the
    router shape that was mounted.  Returning ``None`` on removal/reload shape
    drift is intentionally fail-closed: the old captured callable must never
    serve a request merely because a similarly named feature was once loaded.
    """

    get_router = getattr(feature, "get_router", None)
    if not callable(get_router):
        return None
    try:
        router = get_router()
    except Exception:  # noqa: BLE001 - optional feature boundary
        logger.warning("Failed to resolve current router from feature", exc_info=True)
        return None
    if router is None:
        return None
    prefix, index, route_type, path, methods = selector
    routes = tuple(getattr(router, "routes", ()) or ())
    if index >= len(routes):
        return None
    current = routes[index]
    if (
        getattr(router, "prefix", "") != prefix
        or type(current).__name__ != route_type
        or getattr(current, "path", None) != path
        or tuple(sorted(getattr(current, "methods", ()) or ())) != methods
        or not callable(getattr(current, "app", None))
    ):
        return None
    return current


async def _feature_route_gone_response(scope, receive, send) -> None:
    """Emit the protocol-correct no-route response for a reload race."""

    if scope.get("type") == "websocket":
        await send({"type": "websocket.close", "code": 1008})
        return
    await Response(status_code=404)(scope, receive, send)


def _feature_route_host_dependencies(route, initial_current) -> tuple:
    """Keep dependencies added by the app while a feature route hot-reloads.

    ``include_router`` concatenates the app/router dependencies before the
    source route's own dependencies. Strip that initial source suffix so a
    request can use the selected feature's current router dependencies without
    retaining a stale sibling feature's closure.
    """

    mounted = tuple(getattr(route, "dependencies", ()) or ())
    initial = tuple(getattr(initial_current, "dependencies", ()) or ())
    if not initial or len(mounted) < len(initial):
        return mounted
    if all(left == right for left, right in zip(mounted[-len(initial):], initial)):
        return mounted[:-len(initial)]
    # An unusual custom router did not preserve FastAPI's normal suffix
    # identity. Keep the app-bound dependencies rather than dropping a host
    # authorization dependency.
    return mounted


def _app_bound_current_feature_route_app(
    app: FastAPI,
    mounted_route,
    current_route,
    host_dependencies: tuple,
):
    """Build a current-feature route with FastAPI's app-owned execution seam.

    The mounted copy carries host dependencies and the app's
    ``dependency_overrides`` provider; the current child supplies the
    request-target feature's endpoint and per-router dependencies. Rebuilding
    the tiny route wrapper per request deliberately avoids retaining a stale
    bound feature across a reload or a multi-agent request.
    """

    dependencies = [
        *host_dependencies,
        *(getattr(current_route, "dependencies", ()) or ()),
    ]
    if isinstance(mounted_route, APIRoute) and isinstance(current_route, APIRoute):
        rebound = type(mounted_route)(
            path=mounted_route.path,
            endpoint=current_route.endpoint,
            response_model=current_route.response_model,
            status_code=current_route.status_code,
            tags=current_route.tags,
            dependencies=dependencies,
            summary=current_route.summary,
            description=current_route.description,
            response_description=current_route.response_description,
            responses=current_route.responses,
            deprecated=current_route.deprecated,
            methods=mounted_route.methods,
            operation_id=current_route.operation_id,
            response_model_include=current_route.response_model_include,
            response_model_exclude=current_route.response_model_exclude,
            response_model_by_alias=current_route.response_model_by_alias,
            response_model_exclude_unset=current_route.response_model_exclude_unset,
            response_model_exclude_defaults=current_route.response_model_exclude_defaults,
            response_model_exclude_none=current_route.response_model_exclude_none,
            include_in_schema=mounted_route.include_in_schema,
            response_class=current_route.response_class,
            name=current_route.name,
            dependency_overrides_provider=app,
            callbacks=current_route.callbacks,
            openapi_extra=current_route.openapi_extra,
            generate_unique_id_function=current_route.generate_unique_id_function,
            strict_content_type=current_route.strict_content_type,
        )
        return rebound.app
    if isinstance(mounted_route, APIWebSocketRoute) and isinstance(
        current_route, APIWebSocketRoute
    ):
        rebound = APIWebSocketRoute(
            path=mounted_route.path,
            endpoint=current_route.endpoint,
            name=current_route.name,
            dependencies=dependencies,
            dependency_overrides_provider=app,
        )
        return rebound.app
    if isinstance(mounted_route, StarletteRoute) and isinstance(
        current_route, StarletteRoute
    ):
        rebound = StarletteRoute(
            path=mounted_route.path,
            endpoint=current_route.endpoint,
            methods=mounted_route.methods,
            name=current_route.name,
            include_in_schema=mounted_route.include_in_schema,
        )
        return rebound.app
    if isinstance(mounted_route, StarletteWebSocketRoute) and isinstance(
        current_route, StarletteWebSocketRoute
    ):
        rebound = StarletteWebSocketRoute(
            path=mounted_route.path,
            endpoint=current_route.endpoint,
            name=current_route.name,
        )
        return rebound.app
    return None


def _gate_feature_route(
    app: FastAPI,
    route,
    feature_name: str,
    mount_agent,
    selector: tuple,
    initial_current,
) -> None:
    """Wrap ``route`` so it only serves while its owning feature is live-enabled.

    Feature routers are mounted ONCE at startup, but features can be
    soft-disabled or removed at runtime. Rather than unmount/remount (which
    risks duplicate routes on re-enable and drift from the enable/disable
    lifecycle), the route stays physically mounted and this gate rejects the
    request at match time when the owning feature is currently disabled or
    gone (kestrel-sovereign#2522). Re-enabling the feature flips the gate back
    on with no route churn. Only the feature-contributed routes are gated;
    core/server routes are never touched.

    The gate lives at ``route.matches()`` rather than ``route.app``: Starlette
    otherwise produces a 405 for a wrong-method request before it calls the
    app, and a WebSocket route would receive an invalid HTTP ``JSONResponse``.
    Returning :class:`starlette.routing.Match.NONE` removes a disabled route
    from both HTTP and WebSocket matching before either protocol is handled.
    """
    original_matches = route.matches
    host_dependencies = _feature_route_host_dependencies(route, initial_current)

    def _gated_matches(scope):
        agent = _resolve_route_agent(scope, mount_agent)
        # Dynamic feature routes stay physically mounted so a live feature can
        # be disabled/re-enabled without route churn.  When their mount owner
        # has been DELETEd, however, an unprefixed request must not keep using
        # that stale captured object while a scheduler races to cold-wake it.
        # Request-scoped routes still work for another currently managed agent
        # that exposes the same feature.
        manager = getattr(app.state, "agent_manager", None)
        if manager is not None and agent is mount_agent:
            managed = manager.list_agents()
            managed_agents = (
                managed.values() if hasattr(managed, "values") else (managed or ())
            )
            if not any(current is mount_agent for current in managed_agents):
                return Match.NONE, {}
        features = getattr(agent, "features", None) or {}
        feature = features.get(feature_name) if hasattr(features, "get") else None
        if (
            feature is None
            or not bool(getattr(feature, "enabled", True))
            or _current_feature_router_route(feature, selector) is None
        ):
            return Match.NONE, {}
        return original_matches(scope)

    route.matches = _gated_matches
    # Do not call the copied ``route.app`` directly: it retains the first
    # feature's bound endpoint. Do not call ``current.app`` either: that source
    # child bypasses the app's dependency overrides and host dependencies.
    # Rebind the current endpoint through a fresh app-owned FastAPI route.
    async def _dispatch_current_feature_route(scope, receive, send):
        agent = _resolve_route_agent(scope, mount_agent)
        features = getattr(agent, "features", None) or {}
        feature = features.get(feature_name) if hasattr(features, "get") else None
        if feature is None or not bool(getattr(feature, "enabled", True)):
            await _feature_route_gone_response(scope, receive, send)
            return
        current = _current_feature_router_route(feature, selector)
        if current is None:
            await _feature_route_gone_response(scope, receive, send)
            return
        current_app = _app_bound_current_feature_route_app(
            app,
            route,
            current,
            host_dependencies,
        )
        if current_app is None:
            await _feature_route_gone_response(scope, receive, send)
            return
        await current_app(scope, receive, send)

    route.app = _dispatch_current_feature_route


def _feature_router_signature(feature_name: str, router) -> tuple:
    """Return a stable de-duplication key for one feature router.

    Cold agent registration can happen after the startup mount pass.  Router
    objects are freshly constructed by many features, so object identity is
    not sufficient to avoid duplicate FastAPI routes; use the exposed route
    shape instead.  The feature name keeps intentionally distinct features
    from being conflated merely because they happen to share a path.
    """
    routes = tuple(
        (
            type(route).__name__,
            getattr(route, "path", None),
            tuple(sorted(getattr(route, "methods", ()) or ())),
        )
        for route in getattr(router, "routes", ())
    )
    return (feature_name, getattr(router, "prefix", ""), routes)


def _mount_feature_routers(app: FastAPI, *, agents=None) -> None:
    """Mount routers contributed by discovered features.

    After agent initialization, iterate over all registered features and
    call ``feature.get_router()``. If a feature returns a router, include
    it in the FastAPI app. This allows feature packages (voice, spawn,
    observability, etc.) to contribute HTTP endpoints dynamically.

    Each contributed route is wrapped by :func:`_gate_feature_route` so that
    disabling or removing the owning feature at runtime makes its routes 404
    without unmounting them — the Bridge router, for instance, stops serving
    the moment its feature is disabled and resumes on re-enable, with no
    duplicate routes (kestrel-sovereign#2522).

    Tracks the number of routes added so they can be removed on shutdown
    via ``_unmount_feature_routers``.
    """
    routes_before = len(app.routes)
    mounted = []
    mounted_keys = set(getattr(app.state, "_feature_router_keys", ()))

    def _collect_routers_from_agent(agent) -> None:
        features = getattr(agent, "features", {})
        if not features:
            return
        for name, feature in features.items():
            # Webhook receivers are NOT mounted per-feature here — they are
            # served by ONE shared, live /webhooks/{name} dispatch router
            # (below) whose receiver set is re-scanned per request and filtered
            # to enabled features, so a disabled feature's webhooks stop
            # dispatching without a stale startup snapshot (#2089/#2522).
            if _is_webhook_receiver(getattr(feature, "receiver", None)):
                continue
            routes_before_include: Optional[int] = None
            try:
                router = feature.get_router()
                if router is None:
                    continue
                router_key = _feature_router_signature(name, router)
                if router_key in mounted_keys:
                    continue
                routes_before_include = len(app.routes)
                app.include_router(router)
                # Gate exactly the routes this feature just contributed so a
                # runtime disable/remove makes them 404 (never a core route).
                selectors = tuple(
                    (_feature_router_route_selector(router, index), candidate)
                    for index, candidate in enumerate(
                        tuple(getattr(router, "routes", ()) or ())
                    )
                    if hasattr(candidate, "app")
                )
                added = tuple(
                    candidate
                    for candidate in app.routes[routes_before_include:]
                    if hasattr(candidate, "app")
                )
                if len(added) != len(selectors):
                    raise RuntimeError(
                        "Feature router mount did not preserve child-route shape"
                    )
                for added_route, (selector, initial_current) in zip(added, selectors):
                    _gate_feature_route(
                        app, added_route, name, agent, selector, initial_current
                    )
                mounted_keys.add(router_key)
                mounted.append(name)
            except Exception as exc:
                if routes_before_include is not None:
                    # ``include_router`` can copy ordinary routes before it
                    # reaches a child it cannot preserve (for example a
                    # mounted sub-application). Treat validation as an atomic
                    # mount: never leave those partial, ungated routes in the
                    # app or let a later retry add duplicates.
                    del app.routes[routes_before_include:]
                logger.warning("Failed to mount router from feature %s: %s", name, exc)

    if agents is None:
        # Single-agent mode
        agent = getattr(app.state, "agent", None)
        if agent is not None:
            _collect_routers_from_agent(agent)

        # Multi-agent mode — mount routers from all loaded agents
        manager = getattr(app.state, "agent_manager", None)
        if manager is not None:
            for agent_name in manager.list_agents():
                agent = manager.get_agent(agent_name)
                if agent is not None:
                    _collect_routers_from_agent(agent)
    else:
        for agent in agents:
            if agent is not None:
                _collect_routers_from_agent(agent)

    # Mount one cross-agent webhook dispatch router with a LIVE, scope-aware
    # receiver provider. It re-scans the current agents' ENABLED features on
    # every request (not a startup-captured list), so a webhook stops
    # dispatching the instant its feature is disabled and resumes when
    # re-enabled — the exact "startup-captured receivers still process requests
    # even when disabled" bug (#2522). The provider also honours the
    # request-scoped target agent: an agent-prefixed
    # /api/agents/{name}/webhooks/{name} request sees ONLY that agent's enabled
    # receivers (so it can't dispatch to another agent's identically-named
    # webhook), while the unprefixed /webhooks/{name} form aggregates across
    # every agent (#2522). Mounted when at least one enabled webhook receiver
    # exists at startup; the provider itself stays live thereafter.
    if _live_webhook_receivers(app) and not getattr(
        app.state, "_feature_webhook_dispatch_mounted", False
    ):
        try:
            from kestrel_sovereign.features.webhooks.receiver import (
                build_webhook_dispatch_router,
            )

            app.include_router(
                build_webhook_dispatch_router(
                    lambda agent=None: _live_webhook_receivers(app, agent)
                )
            )
            app.state._feature_webhook_dispatch_mounted = True
            mounted.append("webhooks")
        except Exception as exc:
            logger.warning("Failed to mount webhook dispatch router: %s", exc)

    # Retain the concrete route objects, rather than just a trailing count.
    # Lifespan restarts and test harnesses can call this helper more than once
    # before an outer teardown runs.  A single count would be overwritten by
    # the later mount and leave the earlier routes behind; object ownership
    # lets ``_unmount_feature_routers`` remove every batch it owns.
    added_routes = list(app.routes[routes_before:])
    tracked_routes = list(getattr(app.state, "_feature_routes", ()))
    tracked_routes.extend(added_routes)
    app.state._feature_routes = tracked_routes
    app.state._feature_router_keys = mounted_keys
    # Kept for compatibility with any external diagnostics that surface the
    # old state field.  Teardown uses ``_feature_routes`` as its authority.
    app.state._feature_route_count = len(tracked_routes)

    # FastAPI memoizes the generated schema. A scheduler cold wake can mount a
    # feature router after a client has already fetched /openapi.json, so leave
    # no stale schema that omits the newly reachable API surface.
    if added_routes:
        app.openapi_schema = None

    if mounted:
        logger.info("Dynamically mounted routers from features: %s", ", ".join(mounted))


def _iter_current_agents(app: FastAPI):
    """Yield every agent currently loaded on the app (single- or multi-agent)."""
    agent = getattr(app.state, "agent", None)
    if agent is not None:
        yield agent
    manager = getattr(app.state, "agent_manager", None)
    if manager is not None:
        for agent_name in manager.list_agents():
            resolved = manager.get_agent(agent_name)
            if resolved is not None:
                yield resolved


def _agent_webhook_receivers(agent) -> list:
    """Enabled webhook receivers contributed by a SINGLE agent's features.

    Deduplicated by identity; a disabled/removed feature's receiver is dropped.
    """
    receivers: list = []
    features = getattr(agent, "features", {}) or {}
    for feature in features.values():
        if not bool(getattr(feature, "enabled", True)):
            continue
        receiver = getattr(feature, "receiver", None)
        if _is_webhook_receiver(receiver) and receiver not in receivers:
            receivers.append(receiver)
    return receivers


def _live_webhook_receivers(app: FastAPI, agent=None) -> list:
    """Collect webhook receivers from currently ENABLED features (live).

    Re-scanned on every webhook request by the dispatch router, so a
    disabled/removed feature's receiver is dropped and a re-enabled feature's
    receiver reappears — no stale startup closure.

    Request/scope-aware (kestrel-sovereign#2522): when ``agent`` is provided —
    the target agent the multi-agent routing middleware resolved for an
    agent-prefixed ``/api/agents/{name}/webhooks/{name}`` request — ONLY that
    agent's enabled receivers are returned, so an agent-prefixed request can
    never dispatch to another agent's webhook (e.g. A's feature disabled while
    B enables the same webhook name). When ``agent`` is ``None`` (the
    unprefixed ``/webhooks/{name}`` form, or single-agent mode) the aggregate
    of every current agent's enabled receivers is returned. Deduplicated by
    identity because one receiver can be reached through multiple agents.
    """
    if agent is not None:
        return _agent_webhook_receivers(agent)

    receivers: list = []
    for current in _iter_current_agents(app):
        for receiver in _agent_webhook_receivers(current):
            if receiver not in receivers:
                receivers.append(receiver)
    return receivers


def _unmount_feature_routers(app: FastAPI) -> None:
    """Remove dynamically-mounted feature routes added by ``_mount_feature_routers``.

    This prevents route accumulation when the app lifespan restarts
    (e.g. across TestClient sessions in the same pytest process).
    """
    tracked_routes = tuple(getattr(app.state, "_feature_routes", ()))
    if tracked_routes:
        tracked_ids = {id(route) for route in tracked_routes}
        app.routes[:] = [
            route for route in app.routes if id(route) not in tracked_ids
        ]
        logger.info(
            "Removed %d dynamically-mounted feature routes", len(tracked_routes)
        )
        # The cached schema can still describe these endpoints after their
        # owning feature routes are gone. Rebuild it on the next OpenAPI read.
        app.openapi_schema = None
    app.state._feature_routes = []
    app.state._feature_route_count = 0
    app.state._feature_router_keys = set()
    app.state._feature_webhook_dispatch_mounted = False


def _mount_feature_ui_assets(app: FastAPI, *, agents=None) -> None:
    """Mount per-feature static asset directories (#2043).

    A feature that returns a ``UIContributions`` with a ``static_dir`` from
    ``get_ui_contributions()`` gets that directory served at
    ``/features/{name}/static/`` — so a pip-installed, out-of-tree feature can
    ship its own frontend JS/CSS without the assets living in core ``static/``.
    The manifest served at ``GET /api/ui/contributions`` references these mounts.

    Mounted BEFORE ``_mount_feature_routers`` so the feature routers remain the
    trailing block its index-based ``_unmount`` deletes; these mounts are removed
    by object identity in ``_unmount_feature_ui_assets``.

    NOT gated on ``SERVE_UI`` (unlike the core ``/static`` SPA mount): a feature's
    assets are served by the agent that owns them even in multi-agent host mode,
    where the agent runs with ``KESTREL_SERVE_UI=false`` and the host proxies the
    browser's pinned ``/api/agents/{id}/features/{slug}/static/...`` import to it.
    Gating the mount on ``SERVE_UI`` would 404 that import on every host agent —
    the auth exemption alone (see ``auth_middleware``) is not enough (#2048).
    """
    from kestrel_sovereign.ui_contributions import feature_static_mounts

    seen: set = set()
    mounted_paths = set(getattr(app.state, "_feature_ui_mount_paths", ()))
    pending: list = []

    def _collect(agent) -> None:
        # include_disabled=True: mount every feature that declares a static_dir,
        # even one that starts disabled, so enabling it at runtime from the
        # Feature Store serves its JS without a restart (the runtime-enable 404,
        # #2048). The manifest at GET /api/ui/contributions still lists only
        # enabled features, so a disabled feature's mount is dormant until enabled.
        for mount_path, directory in feature_static_mounts(agent, include_disabled=True):
            if mount_path in seen or mount_path in mounted_paths:
                continue
            seen.add(mount_path)
            pending.append((mount_path, directory))

    if agents is None:
        agent = getattr(app.state, "agent", None)
        if agent is not None:
            _collect(agent)
        manager = getattr(app.state, "agent_manager", None)
        if manager is not None:
            for agent_name in manager.list_agents():
                a = manager.get_agent(agent_name)
                if a is not None:
                    _collect(a)
    else:
        for agent in agents:
            if agent is not None:
                _collect(agent)

    added = []
    for mount_path, directory in pending:
        try:
            app.mount(
                mount_path,
                StaticFiles(directory=directory),
                name=f"feature-ui:{mount_path}",
            )
            added.append(app.routes[-1])
            mounted_paths.add(mount_path)
        except Exception as exc:  # noqa: BLE001 - never block startup on one feature
            logger.warning("Failed to mount feature UI assets at %s: %s", mount_path, exc)

    existing = list(getattr(app.state, "_feature_ui_mounts", ()))
    existing.extend(added)
    app.state._feature_ui_mounts = existing
    app.state._feature_ui_mount_paths = mounted_paths
    if added:
        logger.info(
            "Mounted feature UI asset dirs: %s",
            ", ".join(m for m, _ in pending),
        )


def _unmount_feature_ui_assets(app: FastAPI) -> None:
    """Remove feature static mounts added by ``_mount_feature_ui_assets``.

    Removes by route object identity (not trailing index) so it is robust to
    other dynamic routes added after it within the same lifespan.
    """
    added = getattr(app.state, "_feature_ui_mounts", None)
    if not added:
        return
    for route in added:
        try:
            app.routes.remove(route)
        except ValueError:
            pass
    app.state._feature_ui_mounts = []
    app.state._feature_ui_mount_paths = set()


def _active_local_peer_host_url(app: FastAPI) -> Optional[str]:
    """Return the local peer URL from the host's effective listen settings."""

    config = getattr(app.state, "multi_agent_config", None)
    host = getattr(config, "host", None)
    port = getattr(host, "port", None)
    if isinstance(port, int) and 0 < port <= 65535:
        # A wildcard listener is reachable through loopback from the local
        # peer adapter. Do not advertise 0.0.0.0/:: as an HTTP client target.
        bind = getattr(host, "bind", None)
        hostname = (
            bind
            if isinstance(bind, str) and bind not in {"", "0.0.0.0", "::"}
            else "localhost"
        )
        if ":" in hostname and not hostname.startswith("["):
            hostname = f"[{hostname}]"
        return f"http://{hostname}:{port}"
    explicit_url = os.environ.get("KESTREL_HOST_URL")
    return explicit_url.rstrip("/") if explicit_url else None


def _hosted_peer_directory_context(app: FastAPI, agent) -> tuple[object, object]:
    """Return the effective directory pair for one hosted agent's A2A policy.

    ``PeersFeature`` owns the local-host compatibility adapter after feature
    initialization. Its startup snapshot can predate generated API-key auth,
    explicit multi-agent config discovery, or a platform ``PORT`` override.
    Rebuild only that local adapter from the active host state before freezing
    the hosted policy. An injected scoped router remains authoritative.

    Agents without a Peers feature retain the explicit construction-injection
    seam. A present Peers feature is authoritative: a missing or malformed
    live context is returned as ``(None, None)`` and makes its hosted policy
    reject inbound delivery instead of silently falling back to stale attrs.
    """

    features = getattr(agent, "features", None)
    peer_feature = (
        features.get("PeersFeature")
        if hasattr(features, "get")
        else None
    )
    if peer_feature is None and features is not None:
        values = features.values() if hasattr(features, "values") else features
        peer_feature = next(
            (
                feature
                for feature in values
                if feature.__class__.__name__ == "PeersFeature"
            ),
            None,
        )
    if peer_feature is not None:
        context = getattr(peer_feature, "hosted_peer_directory_context", None)
        if not callable(context):
            return None, None
        resolved = context()
        if resolved is not None:
            router, requester = resolved
            from kestrel_sovereign.features.peers.directory import (
                LocalHostPeerDirectory,
            )

            # Never replace an explicitly injected user-scoped router with
            # global local-host discovery. Only the compatibility adapter is
            # rebuilt from the active host config/auth state.
            if not isinstance(router, LocalHostPeerDirectory):
                return router, requester

        host_url = _active_local_peer_host_url(app)
        if host_url is None:
            return None, None
        refresh = getattr(peer_feature, "refresh_local_host_peer_directory", None)
        if not callable(refresh):
            return None, None
        refreshed = refresh(host_url=host_url, api_key=get_api_key())
        return refreshed if refreshed is not None else (None, None)
    return (
        getattr(agent, "peer_directory_router", None),
        getattr(agent, "peer_requester", None),
    )


async def _onboard_host_registered_agent(app: FastAPI, manager, name: str, agent) -> None:
    """Apply host-owned integration to every newly registered agent.

    Initial fleet startup and scheduler cold wakes share ``AgentManager``'s
    registration seam.  Keeping this work in the app (rather than the
    scheduler) makes a dynamically loaded tenant indistinguishable from an
    autostart tenant for same-host A2A verification and feature HTTP/UI
    exposure.  Failure is intentionally propagated to the manager, which
    unpublishes the partially onboarded agent instead of dispatching work to
    an incompletely integrated tenant.
    """
    from kestrel_sovereign.a2a.did_registry import install_a2a_did_resolver
    from kestrel_sovereign.a2a.inbound_authorization import (
        install_a2a_inbound_sender_authorizer,
        mark_a2a_inbound_scoped_policy,
    )

    federated = os.environ.get("KESTREL_A2A_FEDERATED_DID", "").lower() in (
        "1", "true", "yes",
    )
    # Every agent reaching this hook is hosted by AgentManager, even when its
    # PeersFeature uses the local adapter and therefore has no injected
    # router/requester fields on KestrelAgent. Record hosted policy
    # unconditionally and retain the manager that owns the atomic A2A
    # topology/commit lease. Standalone agents never pass this hook.
    mark_a2a_inbound_scoped_policy(agent, required=True)
    agent._a2a_host_manager = manager
    # A cold wake receives its own verification resolver and explicit inbound
    # authorization seam. Existing agents retain their recipient-scoped
    # objects, whose live manager views see this new peer without replacing
    # their authorization context.
    install_a2a_did_resolver(
        manager,
        recipient=agent,
        federated_fallback=federated,
    )
    install_a2a_inbound_sender_authorizer(manager, recipient=agent)
    peer_router, peer_requester = _hosted_peer_directory_context(app, agent)
    manager.install_a2a_hosted_policy(
        agent,
        resolver=agent.a2a_did_resolver,
        authorizer=agent.a2a_inbound_sender_authorizer,
        router=peer_router,
        requester=peer_requester,
    )
    _mount_feature_ui_assets(app, agents=(agent,))
    _mount_feature_routers(app, agents=(agent,))

    # The host-level demo classification is a live fleet property.  Refresh it
    # on dynamic registration rather than leaving a cold-woken demo tenant
    # classified from the startup snapshot.
    from kestrel_sovereign.security.demo_isolation import classify_server_mode

    app.state.demo_mode = classify_server_mode(manager.list_agents())
    logger.info("Completed host onboarding for dynamically registered agent %r", name)


def _host_config_mapping(config) -> dict:
    """Read-only host config mapping handed to host features via HostContext.

    Deliberately minimal: enough for a host feature to learn the host's
    bind/port and agent roster, plus only the operator-selected mappings under
    ``[host.features.<name>]``. This avoids coupling extensions to the full
    config object or exposing agent definitions and unrelated host settings.
    Carries only the tenant resolver (below) when no multi-agent config is
    available (e.g. a single-agent boot) — host features are host-scoped and
    must still mount. (Moved from the retired ``kestrel_sovereign.host`` — issue
    #2382.)

    Always injects the identity→tenant resolver (issue #2444) under
    ``observability_tenant_resolver`` — the seam the fleet observability host
    feature consumes in ``on_host_start`` to stamp each request's ``tenant_id``.
    It is present even on a single-agent boot (config ``None``) so the store is
    tenant-scoped regardless of deployment shape; zero-config resolves every
    request to one stable default personal tenant (INV-SOLO).
    """
    from copy import deepcopy

    from kestrel_sovereign.security.tenant_resolver import (
        HOST_CONFIG_KEY as _TENANT_RESOLVER_KEY,
        build_tenant_resolver,
    )

    mapping: dict = {}
    if config is None:
        return {_TENANT_RESOLVER_KEY: build_tenant_resolver()}
    try:
        feature_config = getattr(config.host, "features", {})
        if isinstance(feature_config, dict):
            mapping.update(deepcopy(feature_config))
        mapping.update(
            {
                "host_bind": config.host.bind,
                "host_port": config.host.port,
                "agents": list(config.agents.keys()),
            }
        )
    except Exception:  # noqa: BLE001
        pass
    # Security-owned runtime values are written last. Besides the typed
    # HostConfig validator, this protects duck-typed/alternate config objects
    # from shadowing the identity→tenant resolver.
    mapping[_TENANT_RESOLVER_KEY] = build_tenant_resolver()
    return mapping


_SERVER_HOST_ENV = "KESTREL_SERVER_HOST"
_SERVER_PORT_ENV = "PORT"


def _apply_platform_host_port(config, env) -> None:
    """Keep host metadata aligned with the effective listen address.

    The packaged module CLI writes its resolved host and port to the environment
    before Uvicorn starts. Cloud Run and Azure Container Apps inject ``PORT``
    while their entrypoints pass the matching value to Uvicorn. Host features
    therefore receive the effective socket address through ``HostContext``
    rather than stale values from ``multi_agent.toml``.
    """
    runtime_host = env.get(_SERVER_HOST_ENV)
    if runtime_host is not None:
        config.host.bind = runtime_host

    platform_port = env.get(_SERVER_PORT_ENV)
    if platform_port is not None:
        config.host.port = int(platform_port)


# Total budget for Phoenix to become reachable after the host starts (#2589).
# The host is already serving by the time this runs; this only bounds how long
# the background task keeps (re)trying before it gives up and leaves /phoenix
# returning 503 until Phoenix recovers.
_PHOENIX_STARTUP_BUDGET_SECONDS = 180.0


async def _start_phoenix_in_executor(app, supervisor) -> bool:
    """Run one Phoenix ``start`` call without losing its worker on cancellation.

    Cancelling ``asyncio.to_thread`` only cancels its awaiting coroutine; the
    executor worker can still launch the subprocess afterwards.  Keep the task
    on the app until it is terminal so shutdown can join the actual work before
    it calls ``supervisor.stop``.
    """
    start_task = asyncio.create_task(
        asyncio.to_thread(supervisor.start, wait_for_health=False),
        name="phoenix_supervisor_start",
    )
    app.state.phoenix_start_task = start_task
    try:
        return await asyncio.shield(start_task)
    finally:
        # On cancellation, the shield leaves the executor task running.  It is
        # then owned by ``_shutdown_phoenix``.  Only clear a terminal task; a
        # later restart never overwrites a still-running start operation.
        if (
            start_task.done()
            and getattr(app.state, "phoenix_start_task", None) is start_task
        ):
            app.state.phoenix_start_task = None


async def _supervise_phoenix_startup(app, supervisor) -> None:
    """Bring Phoenix up in the background so host boot never blocks on it (#2589).

    Runs ``supervisor.start`` (adopt-or-reap + spawn) off the event loop, then
    polls reachability. It only re-invokes ``start`` if the child process died
    outright — a still-starting child (creating its SQLite schema on first boot)
    reports ``running`` and is left alone. Adopt-or-reap makes re-invocation
    safe: a healthy orphan is adopted rather than duplicated.

    A terminal failure is NOT silent (#2690): when ``start`` returns ``False``
    (custody, port-ownership, or spawn failure — it already logged the specific
    cause at ERROR), when the retry budget expires without reachability, or on an
    unexpected error, this records ``app.state.phoenix_disabled_reason`` and logs
    one ERROR with an operator action. ``/health/detailed`` then surfaces
    'tracing disabled: <reason>' instead of ``disabled_reason: null``. The reason
    is cleared automatically once Phoenix becomes reachable (either here on a
    slow start, or by ``_phoenix_tracing_status`` when it probes the port). The
    supervisor reference is left on ``app.state.phoenix`` so a slow child can
    still recover through reachability gating and shutdown can always reap it.
    """
    import asyncio
    import time as _time

    def _disable(reason: str, action: str) -> None:
        app.state.phoenix_disabled_reason = reason
        logger.error(
            "Phoenix tracing DISABLED — %s. Operator action: %s; no traces are "
            "recorded until then.",
            reason,
            action,
        )

    try:
        # Initial adopt-or-reap + spawn, off the event loop (subprocess launch +
        # a short reachability probe).
        started = await _start_phoenix_in_executor(app, supervisor)
        if not started:
            _disable(
                "supervised start failed (custody, port ownership, or spawn — "
                "see the ERROR logged above for the specific cause)",
                "resolve the cause above (e.g. stop any process holding the "
                "Phoenix port) and restart Kestrel",
            )
            return
        deadline = _time.monotonic() + _PHOENIX_STARTUP_BUDGET_SECONDS
        while _time.monotonic() < deadline:
            if await supervisor.is_reachable():
                logger.info("Phoenix trace backend is reachable.")
                # Recovered — clear any earlier transient disabled reason.
                app.state.phoenix_disabled_reason = None
                return
            if not supervisor.running:
                # The child exited before binding — try a fresh (re)start.
                logger.info("Phoenix child not running yet — restarting.")
                restarted = await _start_phoenix_in_executor(app, supervisor)
                if not restarted:
                    _disable(
                        "supervised restart failed (see the ERROR logged above "
                        "for the specific cause)",
                        "resolve the cause above and restart Kestrel",
                    )
                    return
            await asyncio.sleep(2.0)
        _disable(
            "Phoenix did not become reachable within "
            f"{_PHOENIX_STARTUP_BUDGET_SECONDS:.0f}s",
            "inspect phoenix.log in the trace working directory, then restart "
            "Kestrel",
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - Phoenix must never crash the host
        _disable(
            f"background supervision error: {exc}",
            "check the host logs and restart Kestrel",
        )


async def _shutdown_single_agent(agent: KestrelAgent) -> None:
    """Bound user-visible shutdown while retaining ownership of its tail."""
    cancelled = False
    try:
        await asyncio.wait_for(agent.shutdown(), timeout=SHUTDOWN_TIMEOUT)
        logger.info("Agent shutdown complete.")
    except asyncio.TimeoutError:
        logger.warning(
            "Agent shutdown timed out (%ss); waiting for durable cleanup",
            SHUTDOWN_TIMEOUT,
        )
    except asyncio.CancelledError:
        cancelled = True
        logger.debug("Agent shutdown cancelled")
    except Exception as e:
        logger.warning(f"Error during agent shutdown: {e}")
    cancelled = await await_agent_shutdown_completion(agent) or cancelled
    if cancelled:
        raise asyncio.CancelledError()


async def _shutdown_phoenix(app: FastAPI) -> bool:
    """Release all server-owned Phoenix work before lifespan teardown returns.

    The supervisor task and its proxy client are server-owned resources.  A
    cancelled agent shutdown must not strand either one: join the task and the
    async close despite repeated cancellation, then always terminate the child
    even when closing the proxy client fails.  The return value lets the caller
    re-raise cancellation only after this teardown is complete.
    """
    cancelled = False

    phoenix_task = getattr(app.state, "phoenix_task", None)
    if phoenix_task is not None:
        if not phoenix_task.done():
            phoenix_task.cancel()
        try:
            join_cancelled, failure = await await_lifecycle_task_completion(
                phoenix_task
            )
            cancelled = cancelled or join_cancelled
            if failure is not None and not isinstance(
                failure, asyncio.CancelledError
            ):
                logger.warning(
                    "Phoenix supervision task failed during shutdown: %s",
                    failure,
                    exc_info=(type(failure), failure, failure.__traceback__),
                )
        finally:
            app.state.phoenix_task = None

    # ``phoenix_task`` can finish immediately when it is cancelled while its
    # shielded executor-backed ``start`` is still running.  Never cancel this
    # task: cancelling it would still leave the worker thread alive.  Join it
    # first, then stop the supervisor so a late worker cannot launch Phoenix
    # after teardown has returned.
    phoenix_start_task = getattr(app.state, "phoenix_start_task", None)
    if phoenix_start_task is not None:
        try:
            join_cancelled, failure = await await_lifecycle_task_completion(
                phoenix_start_task
            )
            cancelled = cancelled or join_cancelled
            if failure is not None and not isinstance(
                failure, asyncio.CancelledError
            ):
                logger.warning(
                    "Phoenix startup worker failed during shutdown: %s",
                    failure,
                    exc_info=(type(failure), failure, failure.__traceback__),
                )
        finally:
            if getattr(app.state, "phoenix_start_task", None) is phoenix_start_task:
                app.state.phoenix_start_task = None

    phoenix = getattr(app.state, "phoenix", None)
    if phoenix is None:
        return cancelled

    try:
        close_task = asyncio.create_task(phoenix.aclose())
        close_cancelled, close_failure = await await_lifecycle_task_completion(
            close_task
        )
        cancelled = cancelled or close_cancelled
        if close_failure is not None:
            logger.warning(
                "Error closing Phoenix proxy client: %s",
                close_failure,
                exc_info=(
                    type(close_failure),
                    close_failure,
                    close_failure.__traceback__,
                ),
            )
    except Exception as exc:  # noqa: BLE001 - stop still must reap the child
        logger.warning("Error closing Phoenix proxy client: %s", exc)
    finally:
        try:
            phoenix.stop()
        except Exception as exc:  # noqa: BLE001 - complete state teardown
            logger.warning("Error stopping Phoenix: %s", exc)
        app.state.phoenix = None

    return cancelled


async def _shutdown_host_features(app: FastAPI) -> None:
    """Release host-scoped resources without leaving later resources behind."""
    from kestrel_sovereign import host_features as _hf

    host_features = getattr(app.state, "host_features", []) or []
    host_context = getattr(app.state, "host_context", None)
    try:
        if host_features and host_context is not None:
            await _hf.stop_host_features(host_features, host_context)
    except Exception as exc:  # noqa: BLE001 - preserve the existing best effort
        logger.warning("Host feature shutdown failed: %s", exc)
    finally:
        # Router/UI state must not outlive a failed feature shutdown.  Each
        # following cleanup is in a ``finally`` so one bad unmount cannot leave
        # the host session factory or database live.
        session_factory = (
            getattr(host_context, "session_factory", None)
            if host_context is not None
            else None
        )
        try:
            _hf.unmount_host_features(app)
        finally:
            try:
                _unmount_feature_routers(app)
            finally:
                try:
                    _unmount_feature_ui_assets(app)
                finally:
                    try:
                        if session_factory is not None:
                            await session_factory.close()
                    except Exception as exc:  # noqa: BLE001 - close host DB too
                        logger.warning(
                            "Host feature session-factory shutdown failed: %s", exc
                        )
                    finally:
                        host_db = (
                            getattr(host_context, "db", None)
                            if host_context is not None
                            else None
                        )
                        if host_db is not None and hasattr(host_db, "close"):
                            try:
                                await host_db.close()
                            except Exception as exc:  # noqa: BLE001 - terminal cleanup
                                logger.warning(
                                    "Host feature database shutdown failed: %s", exc
                                )


def _uses_shared_postgres_scheduler() -> bool:
    """Whether this host can safely poll the fleet's shared schedule table."""
    return (
        os.environ.get("KESTREL_DB_BACKEND", "sqlite").lower() == "postgres"
        and bool(os.environ.get("KESTREL_DATABASE_URL"))
    )


async def _prepare_shared_postgres_scheduler_protocol(app: FastAPI, manager, config) -> None:
    """Seed one shared scheduler protocol epoch before agent runners start.

    Hosted ``SchedulerFeature`` instances do not create agent-scoped polling
    runners, but their post-load hooks still seed defaults concurrently. Do one
    non-polling host bootstrap first: discover every configured local DID
    without initializing it, establish the database-global provenance marker,
    and write every DID's rollout row in the same protocol pass.

    This deliberately closes its temporary storage before agent initialization.
    It is a migration/preflight owner, not a second scheduler replica; starting
    a polling host runner here could race a concurrent autostart cold load.
    ``_start_host_scheduler`` creates the long-lived executor only after the
    configured fleet has completed its initialization phase.
    """
    if not _uses_shared_postgres_scheduler():
        return

    # Mark every subsequently loaded AgentManager tenant before feature
    # initialization. The long-lived host runner below is the sole poller and
    # owns the manager's live authority/lifecycle gates; an agent-scoped runner
    # would retain stale execution authority across remove_agent().
    configure_host_polling = getattr(
        manager, "set_scheduler_polling_managed_by_host", None
    )
    if callable(configure_host_polling):
        configure_host_polling(True)

    from kestrel_sovereign.features.scheduler.feature import SchedulerFeature
    from kestrel_sovereign.features.scheduler.runner import (
        AgentManagerHostedSchedulerExecutor,
        SchedulerRunner,
    )
    from kestrel_sovereign.storage.async_storage import AsyncStorage

    # Do this before ``load_from_config``.  The manager's lookup is explicitly
    # read-only and omits an unincepted/unresolved DID rather than creating a
    # blank identity DB or withholding healthy peers from the shared fleet.
    agent_configs = await manager.local_agent_configs_by_did(config)
    for name, error in getattr(manager, "cold_scheduler_identity_failures", ()):
        _latch_scheduler_readiness_failure(
            app,
            "identity",
            error,
            agent_name=name,
        )
    if not agent_configs:
        logger.info("Shared scheduler protocol bootstrap skipped: no local agents")
        return

    storage = AsyncStorage(
        backend="postgres",
        dsn=os.environ["KESTREL_DATABASE_URL"],
    )
    preflight_failure: BaseException | None = None
    try:
        await storage.initialize()
        if storage.db is None:
            raise RuntimeError(
                "shared PostgreSQL scheduler protocol storage did not initialize"
            )

        def _on_scheduler_protocol_failure(error: BaseException) -> None:
            _latch_scheduler_readiness_failure(app, "protocol", error)

        live_authority = getattr(manager, "is_scheduler_agent_authorized", None)
        live_scope = getattr(manager, "scheduler_authorized_agent_ids", None)
        runner = SchedulerRunner(
            db=storage.db,
            agent_id=None,
            executor=AgentManagerHostedSchedulerExecutor(manager, agent_configs),
            authorized_agent_ids=agent_configs.keys(),
            authorized_agent_ids_provider=(
                live_scope if callable(live_scope) else None
            ),
            is_agent_authorized=(
                live_authority if callable(live_authority) else None
            ),
            on_protocol_failure=_on_scheduler_protocol_failure,
            misfire_grace_seconds=SchedulerFeature._load_misfire_grace_seconds(),
            max_concurrent_tasks=SchedulerFeature._load_max_concurrent_tasks(),
            lease_seconds=SchedulerFeature._load_lease_seconds(),
        )
        # This is intentionally ``_ensure_tables`` rather than ``start``.  It
        # establishes the durable schema/provenance/rollout state but never
        # polls or dispatches while autostart agents are being initialized.
        await runner._ensure_tables()
    except BaseException as error:
        preflight_failure = error
        if not isinstance(error, asyncio.CancelledError):
            _latch_scheduler_readiness_failure(app, "protocol", error)
        raise
    finally:
        # A transient bootstrap pool is invisible to ordinary server teardown,
        # so this helper remains its cancellation-safe lifecycle owner until it
        # is terminal.  Do not allow a repeated cancellation to strand it.
        close_task = asyncio.create_task(
            storage.close(), name="host_scheduler_protocol_preflight_close"
        )
        close_cancelled, close_failure = await await_lifecycle_task_completion(
            close_task
        )
        if close_failure is not None:
            if not isinstance(close_failure, asyncio.CancelledError):
                logger.warning(
                    "Shared scheduler protocol preflight storage close failed: %s",
                    close_failure,
                    exc_info=(
                        type(close_failure),
                        close_failure,
                        close_failure.__traceback__,
                    ),
                )
            if preflight_failure is None:
                if not isinstance(close_failure, asyncio.CancelledError):
                    _latch_scheduler_readiness_failure(
                        app, "protocol", close_failure
                    )
                raise close_failure
        if close_cancelled and preflight_failure is None:
            raise asyncio.CancelledError()


async def _start_host_scheduler(app: FastAPI, manager, config) -> None:
    """Start the shared-PostgreSQL scheduler that can wake cold local agents.

    Agent-scoped runners remain the standalone executor implementation outside
    this topology. In a shared PostgreSQL fleet, this is the sole poller: it
    claims rows without an ``agent_id`` scope and dispatches through
    ``AgentManager``; therefore an occurrence for ``autostart=false`` can load
    its target only after the claim succeeds. The dedicated storage connection
    belongs to the host lifecycle, not to any agent that the scheduler might
    wake or later shut down.
    """
    app.state.host_scheduler_runner = None
    app.state.host_scheduler_storage = None
    app.state.scheduler_cold_agent_failures = []
    # The schema-only preflight runs before parallel agent initialization and
    # may already have latched an unresolved DID or protocol failure.  A later
    # transition into the long-lived host runner must not make that safety
    # evidence disappear from readiness.
    if not hasattr(app.state, "scheduler_readiness_failures"):
        app.state.scheduler_readiness_failures = []
    if not _uses_shared_postgres_scheduler():
        return

    from kestrel_sovereign.features.scheduler.feature import SchedulerFeature
    from kestrel_sovereign.features.scheduler.runner import (
        AgentManagerHostedSchedulerExecutor,
        SchedulerRunner,
    )
    from kestrel_sovereign.storage.async_storage import AsyncStorage

    # Build this before the first poll and from *all* local configuration,
    # including cold (autostart=false) agents.  The manager resolves cold DIDs
    # from their local identity store without loading the agent itself.
    agent_configs = await manager.local_agent_configs_by_did(config)
    cold_identity_failures = getattr(
        manager, "cold_scheduler_identity_failures", []
    )
    for name, error in cold_identity_failures:
        _latch_scheduler_readiness_failure(
            app,
            "identity",
            error,
            agent_name=name,
        )
    # Kept as a named compatibility field for authenticated diagnostics; its
    # entries are the same redacted readiness records, never exception text.
    app.state.scheduler_cold_agent_failures = list(
        getattr(app.state, "scheduler_readiness_failures", ())
    )
    for failure in app.state.scheduler_cold_agent_failures:
        logger.error(
            "Shared scheduler cannot wake configured agent %r (cause=%s)",
            failure.get("agent"),
            failure["cause_type"],
        )
    storage = AsyncStorage(
        backend="postgres",
        dsn=os.environ["KESTREL_DATABASE_URL"],
    )
    # Publish ownership before the first await.  A cancellation or initialization
    # failure can then use the ordinary host teardown path to close this
    # connection rather than leaving an invisible pool behind.
    app.state.host_scheduler_storage = storage
    runner = None
    try:
        await storage.initialize()
        if storage.db is None:
            raise RuntimeError("shared PostgreSQL scheduler storage did not initialize")
        def _on_scheduler_protocol_failure(error: BaseException) -> None:
            _latch_scheduler_readiness_failure(app, "protocol", error)

        live_authority = getattr(manager, "is_scheduler_agent_authorized", None)
        live_scope = getattr(manager, "scheduler_authorized_agent_ids", None)
        runner = SchedulerRunner(
            db=storage.db,
            agent_id=None,
            executor=AgentManagerHostedSchedulerExecutor(manager, agent_configs),
            authorized_agent_ids=agent_configs.keys(),
            authorized_agent_ids_provider=(
                live_scope if callable(live_scope) else None
            ),
            is_agent_authorized=(
                live_authority if callable(live_authority) else None
            ),
            on_protocol_failure=_on_scheduler_protocol_failure,
            misfire_grace_seconds=SchedulerFeature._load_misfire_grace_seconds(),
            max_concurrent_tasks=SchedulerFeature._load_max_concurrent_tasks(),
            lease_seconds=SchedulerFeature._load_lease_seconds(),
        )
        # ``runner.start`` runs schema migration / rollout fencing and can fail.
        # Publish it first so cleanup also reaches a partially-started runner.
        app.state.host_scheduler_runner = runner
        await runner.start()

        async def _register_runtime_scheduler_tenant(
            name: str,
            agent_id: str,
            _config,
        ):
            # The manager has already published live config authority, but has
            # not yet exposed this DID to the host runner's execution scope.
            # Prepare its rollout row under the same database-global bootstrap
            # serialization as process startup, before SchedulerFeature
            # post-load can seed built-ins.
            tenant_runner = SchedulerRunner(
                db=storage.db,
                agent_id=None,
                executor=AgentManagerHostedSchedulerExecutor(manager),
                authorized_agent_ids=(agent_id,),
                on_protocol_failure=_on_scheduler_protocol_failure,
                misfire_grace_seconds=(
                    SchedulerFeature._load_misfire_grace_seconds()
                ),
                max_concurrent_tasks=(
                    SchedulerFeature._load_max_concurrent_tasks()
                ),
                lease_seconds=SchedulerFeature._load_lease_seconds(),
            )
            registration = await tenant_runner.prepare_tenant_registration()

            async def _rollback_runtime_scheduler_tenant() -> None:
                await tenant_runner.rollback_tenant_registration(registration)

            # The manager retains this callable only until app-owned
            # onboarding commits. Carry the private nonce through that seam so
            # SchedulerFeature can stamp schedules seeded during the pending
            # registration and rollback can identify only those rows.
            _rollback_runtime_scheduler_tenant.scheduler_registration_nonce = (
                registration.registration_nonce
            )

            logger.info(
                "Prepared shared scheduler protocol for runtime agent %r",
                name,
            )
            return _rollback_runtime_scheduler_tenant

        configure_registration = getattr(
            manager, "set_scheduler_tenant_registration_hook", None
        )
        if callable(configure_registration):
            configure_registration(_register_runtime_scheduler_tenant)
    except BaseException as startup_failure:
        cleanup = asyncio.create_task(
            _shutdown_host_scheduler(app),
            name="host_scheduler_startup_cleanup",
        )
        cleanup_cancelled, cleanup_failure = await await_lifecycle_task_completion(
            cleanup
        )
        if cleanup_failure is not None and not isinstance(
            cleanup_failure, asyncio.CancelledError
        ):
            logger.warning(
                "Host scheduler startup cleanup failed after %s: %s",
                type(startup_failure).__name__,
                cleanup_failure,
                exc_info=(
                    type(cleanup_failure),
                    cleanup_failure,
                    cleanup_failure.__traceback__,
                ),
            )
        if cleanup_cancelled and not isinstance(
            startup_failure, asyncio.CancelledError
        ):
            raise asyncio.CancelledError() from startup_failure
        raise

    logger.info(
        "Started shared PostgreSQL scheduler for %d local agent(s)",
        len(agent_configs),
    )


async def _shutdown_host_scheduler(app: FastAPI) -> None:
    """Stop the host-owned scheduler before agent storage is released."""
    runner = getattr(app.state, "host_scheduler_runner", None)
    storage = getattr(app.state, "host_scheduler_storage", None)
    manager = getattr(app.state, "agent_manager", None)
    if manager is not None:
        configure_registration = getattr(
            manager, "set_scheduler_tenant_registration_hook", None
        )
        if callable(configure_registration):
            configure_registration(None)
    try:
        if runner is not None:
            await runner.stop()
    finally:
        try:
            if storage is not None:
                await storage.close()
        finally:
            app.state.host_scheduler_runner = None
            app.state.host_scheduler_storage = None


async def _shutdown_server_agents(app: FastAPI) -> None:
    """Run the agent-owned shutdown phase for either server topology."""
    manager = getattr(app.state, "agent_manager", None)
    cleanup_manager = getattr(app.state, "startup_cleanup_agent_manager", None)
    managers = []
    for candidate in (manager, cleanup_manager):
        if candidate is not None and all(candidate is not known for known in managers):
            managers.append(candidate)
    if managers:
        for owned_manager in managers:
            await owned_manager.shutdown_all()
            if getattr(app.state, "agent_manager", None) is owned_manager:
                app.state.agent_manager = None
            if (
                getattr(app.state, "startup_cleanup_agent_manager", None)
                is owned_manager
            ):
                app.state.startup_cleanup_agent_manager = None
        logger.info("All agents shutdown complete.")
        return

    agent = getattr(app.state, "agent", None)
    if agent is not None:
        await _shutdown_single_agent(agent)


async def _rollback_startup_agent_manager(manager) -> bool:
    """Drain a partially-started multi-agent manager before dropping it.

    Host scheduler startup happens after agents have been loaded.  Its failure
    must therefore not make the manager unreachable while those agents still
    own dispatchers or storage.  Keep the cleanup task alive through repeated
    cancellation and return whether the startup caller was cancelled.
    """
    rollback = asyncio.create_task(
        manager.shutdown_all(), name="server_startup:rollback_agents"
    )
    cancelled, failure = await await_lifecycle_task_completion(rollback)
    if failure is not None and not isinstance(failure, asyncio.CancelledError):
        logger.warning(
            "Multi-agent startup rollback did not fully shut down loaded agents: %s",
            failure,
            exc_info=(type(failure), failure, failure.__traceback__),
        )
    elif isinstance(failure, asyncio.CancelledError):
        logger.warning(
            "Multi-agent startup rollback was cancelled after its cleanup task "
            "reached a cancelled terminal state"
        )
    return cancelled


async def _run_lifespan_shutdown_phase(
    name: str,
    operation,
) -> tuple[bool, BaseException | None, object | None]:
    """Own one teardown phase through repeated cancellation.

    Each phase receives a distinct task.  A cancellation of the lifespan task
    therefore records cancellation but cannot stop the next independently
    owned phase from releasing its resources.
    """
    phase_task = asyncio.create_task(operation(), name=f"server_shutdown:{name}")
    cancelled, failure = await await_lifecycle_task_completion(phase_task)
    if failure is not None and not isinstance(failure, asyncio.CancelledError):
        logger.warning(
            "Server shutdown phase %r failed: %s",
            name,
            failure,
            exc_info=(type(failure), failure, failure.__traceback__),
        )
    if isinstance(failure, asyncio.CancelledError):
        cancelled = True
    result = None
    if failure is None:
        result = phase_task.result()
    return cancelled, failure, result


async def _shutdown_server_resources(app: FastAPI) -> tuple[bool, BaseException | None]:
    """Drain every server-owned teardown phase before reporting an outcome."""
    logger.info("Server shutting down...")
    cancelled = False
    first_failure: BaseException | None = None

    for name, operation in (
        ("host-scheduler", lambda: _shutdown_host_scheduler(app)),
        # A shared PostgreSQL runner can still complete a cold wake's
        # registration/onboarding path while stop() drains owned work. Drain
        # it before unmounting host and feature surfaces, otherwise that late
        # onboarding can remount routes or UI after their only teardown pass.
        ("host-features", lambda: _shutdown_host_features(app)),
        ("agents", lambda: _shutdown_server_agents(app)),
        ("phoenix", lambda: _shutdown_phoenix(app)),
    ):
        phase_cancelled, failure, result = await _run_lifespan_shutdown_phase(
            name, operation
        )
        cancelled = cancelled or phase_cancelled
        if result is True:
            # ``_shutdown_phoenix`` records cancellation that it observed while
            # joining its app-owned work.
            cancelled = True
        if (
            failure is not None
            and not isinstance(failure, asyncio.CancelledError)
            and first_failure is None
        ):
            first_failure = failure

    return cancelled, first_failure


@asynccontextmanager
async def _lifespan_teardown_owner(app: FastAPI):
    """Make teardown own startup, the ``yield``, and all cancellation paths."""
    body_failure: BaseException | None = None
    try:
        yield
    except BaseException as exc:
        body_failure = exc
        raise
    finally:
        cancelled, teardown_failure = await _shutdown_server_resources(app)
        # Preserve the original failure from startup/request lifespan body.  On
        # a normal exit, however, cleanup must make a terminal agent failure
        # visible only after Phoenix and every other phase have been drained.
        if body_failure is None:
            if cancelled:
                raise asyncio.CancelledError()
            if teardown_failure is not None:
                raise teardown_failure


@asynccontextmanager
async def _lifespan_startup(app: FastAPI):
    """Initialize server resources; outer lifespan ownership handles teardown."""
    logger.info("Server starting up...")
    _set_startup_error(app, None)
    app.state.mandatory_feature_failures = []
    app.state.identity_readiness_failures = []
    app.state.scheduler_cold_agent_failures = []
    app.state.scheduler_readiness_failures = []
    # On a failed rollback this private owner remains reachable only to
    # teardown. It must never become the public routing manager.
    app.state.startup_cleanup_agent_manager = None

    # --- Host-supervised Phoenix trace backend (issue #2570) ---
    # Launch Phoenix BEFORE agents so the OTLP endpoint is on os.environ when
    # in-process agents initialize and when subprocess agents inherit the env
    # (INV-SOLO zero-config wiring). Fully degrades when arize-phoenix is not
    # installed or KESTREL_PHOENIX_ENABLED=0 — the host is unaffected.
    app.state.phoenix = None
    app.state.phoenix_task = None
    # A Phoenix ``start`` call runs in an executor.  Its task is distinct from
    # the supervisor coroutine because cancelling the latter cannot stop an
    # already-running thread; shutdown joins this task before calling stop().
    app.state.phoenix_start_task = None
    # Reason tracing is off, surfaced on the authenticated /health/detailed so a
    # silent-off boot is visible to operators (#2690). ``None`` == not disabled.
    app.state.phoenix_disabled_reason = None
    try:
        from kestrel_sovereign.phoenix_supervisor import (
            PhoenixStorageError,
            PhoenixSupervisor,
            autowire_otel_project,
            autowire_otlp_endpoint,
            should_supervise_phoenix,
        )

        if should_supervise_phoenix():
            supervisor = PhoenixSupervisor()
            try:
                # Custody-aware reconcile BEFORE prepare_storage (#2690): reap a
                # leaked legacy-store (or unidentified) Phoenix holding the port
                # so the legacy migration then sees a *stopped* store. Without
                # this, a live legacy Phoenix is adopted yet custody refuses to
                # migrate it, and tracing goes silently off for the whole boot.
                await asyncio.to_thread(supervisor.reconcile_storage_conflict)
                # Privacy gate (#2609): establish/migrate private trace custody
                # before advertising a local OTLP collector to the host or
                # agents. A failure leaves app.state.phoenix unset and the
                # endpoint unwired, so tracing is clearly disabled rather than
                # writing insecurely.
                await asyncio.to_thread(supervisor.prepare_storage)
            except PhoenixStorageError as exc:
                # Never a silent-off boot (#2690): log at ERROR once with a
                # clear operator action and record the reason for /health.
                reason = f"private trace-store custody could not be established: {exc}"
                logger.error(
                    "Phoenix tracing DISABLED — %s. Operator action: resolve the "
                    "storage conflict reported above (stop any stale Phoenix, then "
                    "restart Kestrel); no traces are recorded until then.",
                    reason,
                )
                app.state.phoenix = None
                app.state.phoenix_disabled_reason = reason
            else:
                # Track the supervisor immediately so /phoenix + the mint can
                # gate on reachability (503 until Phoenix answers). Zero-config
                # (INV-SOLO): default the OTLP endpoint to the local Phoenix
                # collector for this process AND for env inherited by spawned
                # agents, unless the operator already set it.
                app.state.phoenix = supervisor
                endpoint = autowire_otlp_endpoint(os.environ)
                if endpoint:
                    logger.info(
                        "OTEL_EXPORTER_OTLP_ENDPOINT auto-set to local Phoenix (%s)",
                        endpoint,
                    )
                # Group host + spawned-agent traces under a single Phoenix
                # project (obs#32). Mutates os.environ so subprocess agents
                # inherit it; the SDK's tracing bootstrap (>= 0.30.2) reads
                # KESTREL_OTEL_PROJECT.
                project = autowire_otel_project(os.environ)
                if project:
                    logger.info(
                        "KESTREL_OTEL_PROJECT auto-set to '%s' (fleet-scoped traces)",
                        project,
                    )
                # Non-blocking first boot (#2589): Phoenix's first start (SQLite
                # schema creation) can exceed the launcher's health window. Bring
                # it up in the BACKGROUND (adopt-or-reap + retry) so the host's
                # own /health never waits on Phoenix; /phoenix returns 503 until
                # ready.
                app.state.phoenix_task = asyncio.create_task(
                    _supervise_phoenix_startup(app, supervisor)
                )
        else:
            from kestrel_sovereign.phoenix_supervisor import (
                _running_under_pytest,
                phoenix_available,
            )

            if _running_under_pytest():
                logger.info(
                    "Phoenix supervision suppressed under pytest (installed=%s) — "
                    "/phoenix returns 503.",
                    phoenix_available(),
                )
            else:
                logger.info(
                    "Phoenix trace backend disabled (installed=%s) — /phoenix returns 503.",
                    phoenix_available(),
                )
    except Exception as exc:  # noqa: BLE001 - Phoenix must never block startup
        # Never a silent-off boot (#2690): ERROR once with an operator action,
        # and surface the reason on /health/detailed.
        reason = f"supervision setup failed: {exc}"
        logger.error(
            "Phoenix tracing DISABLED — %s. Operator action: check the host logs "
            "above and restart Kestrel; no traces are recorded until then.",
            reason,
        )
        app.state.phoenix = None
        app.state.phoenix_disabled_reason = reason

    # Detect multi-agent mode
    multi_agent_env = os.environ.get("KESTREL_MULTI_AGENT", "").lower() in ("1", "true", "yes")
    multi_agent_path = resolve_multi_agent_path(os.environ)

    if multi_agent_env or multi_agent_path.exists():
        # --- Multi-agent mode ---
        manager = None
        try:
            from kestrel_sovereign.multi_agent.agent_manager import AgentManager
            from kestrel_sovereign.multi_agent.config import MultiAgentConfig

            config = MultiAgentConfig.load(
                str(multi_agent_path) if multi_agent_path.exists() else None,
                auto_discover_fallback=True,
            )
            _apply_platform_host_port(config, os.environ)
            manager = AgentManager(base_data_dir=Path.cwd())
            app.state.agent_manager = manager
            # Persistence context for runtime agent creation (#2358): when the
            # deployment is DRIVEN BY a multi_agent.toml, a UI-created agent
            # must be appended there or it vanishes on restart (startup loads
            # the file whenever it exists). Auto-discovered deployments need no
            # write — their agents re-discover from agent_data/ on boot.
            app.state.multi_agent_config = config
            app.state.multi_agent_config_path = (
                multi_agent_path if multi_agent_path.exists() else None
            )
            app.state.agent = None  # No single default agent
            # Registration is the one path shared by autostart, runtime
            # creation, spawning, and scheduler cold wakes.  Install the
            # app-owned onboarding hook before the first initializer publishes
            # so every tenant receives A2A + feature integration.
            manager.set_agent_registration_hook(
                lambda name, agent: _onboard_host_registered_agent(
                    app, manager, name, agent
                )
            )
            # Seed the database-global scheduler provenance and every local
            # DID's durable protocol row before concurrent agent
            # initialization and post-load default seeding.
            await _prepare_shared_postgres_scheduler_protocol(app, manager, config)
            loaded = await manager.load_from_config(config)
            logger.info(f"Multi-agent mode: {loaded} agent(s) loaded")

            # Fleet-idleness (#F235) is wired at the AgentManager's single
            # registration point, so every agent — startup or dynamically
            # woken — gets the co-hosted-agents provider and no tenant can
            # bypass the whole-host-restart gate.

            # Lifecycle hardening (#377): surface per-agent init failures
            # — without this, a multi-agent host whose providers all failed
            # would report healthy startup while every agent was mute.
            init_failures = manager.init_failures
            app.state.mandatory_feature_failures = [
                record
                for name, exc in init_failures
                if (
                    record := _mandatory_feature_failure_record(name, exc)
                ) is not None
            ]
            app.state.identity_readiness_failures = [
                record
                for name, exc in init_failures
                if (
                    record := _identity_readiness_failure_record(name, exc)
                ) is not None
            ]
            if init_failures and loaded == 0:
                # Every configured agent failed to initialize. Treat the
                # whole host as broken so /health reports it.
                _set_startup_error(
                    app,
                    RuntimeError(
                        "Multi-agent startup: no agents initialized. "
                        f"{len(init_failures)} failures — "
                        + "; ".join(
                            f"{name}: {type(exc).__name__}: {exc}"
                            for name, exc in init_failures[:5]
                        )
                    ),
                )
            elif init_failures:
                # Partial failure: some agents up, some not. Log loudly so
                # operators see the gap. Not a startup error (the host can
                # still serve the agents that did come up), but the partial
                # state must be visible in logs.
                for name, exc in init_failures:
                    logger.error(
                        f"Multi-agent partial failure: agent '{name}' failed "
                        f"to initialize — {type(exc).__name__}: {exc}"
                    )

            # A shared PostgreSQL database can carry schedules for agents that
            # are intentionally cold at host boot.  Start the fleet-owned
            # runner only in that topology; SQLite installations retain the
            # longstanding agent-scoped runner and never share schedule rows.
            await _start_host_scheduler(app, manager, config)
        except Exception as e:
            rollback_cancelled = False
            rollback_complete = manager is None
            if manager is not None:
                rollback_cancelled = await _rollback_startup_agent_manager(manager)
                rollback_complete = not manager.list_agents()
                if not rollback_complete:
                    # Preserve the manager for the outer lifespan teardown to
                    # retry, but never leave it on the public routing surface:
                    # a startup error dominates readiness and these agents may
                    # still be draining their own cleanup.
                    app.state.startup_cleanup_agent_manager = manager
                    app.state.agent_manager = None
            identity_record = _identity_readiness_failure_record("default", e)
            if identity_record is not None:
                logger.error(
                    "Error during multi-agent startup: %s "
                    "(code=%s, cause_type=%s)",
                    e,
                    identity_record["error_code"],
                    identity_record["cause_type"],
                )
                app.state.identity_readiness_failures = [identity_record]
            else:
                logger.error(
                    f"Error during multi-agent startup: {e}",
                    exc_info=True,
                )
            if rollback_complete:
                app.state.agent_manager = None
            app.state.agent = None
            _set_startup_error(app, e)
            if rollback_cancelled:
                raise asyncio.CancelledError()
    else:
        # --- Single-agent mode (original behavior) ---
        app.state.agent_manager = None
        llm_service = None
        try:
            db_backend = os.environ.get("KESTREL_DB_BACKEND", "sqlite")
            database_url = os.environ.get("KESTREL_DATABASE_URL")

            if db_backend.lower() == "postgres" and database_url:
                logger.info("Using PostgreSQL backend for Kestrel")
                storage_dir = os.environ.get("KESTREL_DB_PATH", os.getcwd())
                db_path = os.path.join(storage_dir, "kestrel_prime.db")
                agent_did = await get_agent_did_async(
                    storage_dir,
                    db_backend="postgres",
                    database_url=database_url,
                )
                verify_identity_isolation(agent_did)
                llm_service = LLMService()
                app.state.agent = KestrelAgent(
                    did=agent_did,
                    storage_path=db_path,
                    llm_service=llm_service,
                    database_url=database_url,
                    db_backend="postgres",
                )
            else:
                storage_dir = os.environ.get("KESTREL_DB_PATH", os.getcwd())
                db_path = os.path.join(storage_dir, "kestrel_prime.db")
                agent_did = await get_agent_did_async(storage_dir)
                verify_identity_isolation(agent_did)
                llm_service = LLMService()
                app.state.agent = KestrelAgent(
                    did=agent_did,
                    storage_path=db_path,
                    llm_service=llm_service,
                )
                logger.info(f"Using SQLite backend for Kestrel: {db_path}")

            # Lifecycle hardening: provider availability (#377) is verified
            # inside KestrelAgent.initialize so every boot path — including
            # the multi-agent AgentManager path above — gets the same check.
            await app.state.agent.initialize()
            logger.info(f"Kestrel Agent initialized and ready (backend: {db_backend})")
        except Exception as e:
            app.state.agent = None
            mandatory_record = _mandatory_feature_failure_record("default", e)
            identity_record = _identity_readiness_failure_record("default", e)
            if identity_record is not None:
                logger.error(
                    "Error during startup: %s (code=%s, cause_type=%s)",
                    e,
                    identity_record["error_code"],
                    identity_record["cause_type"],
                )
                app.state.identity_readiness_failures = [identity_record]
                if llm_service is not None:
                    try:
                        await llm_service.close()
                    except Exception as close_error:  # noqa: BLE001
                        logger.warning(
                            "Could not close LLM service after blocked identity "
                            "startup (cause_type=%s)",
                            type(close_error).__name__,
                        )
            else:
                logger.error(f"Error during startup: {e}", exc_info=True)
            if mandatory_record is not None:
                app.state.mandatory_feature_failures = [mandatory_record]
            _set_startup_error(app, e)

    # Per-feature static asset mounts (#2043). Done BEFORE router mounting so the
    # feature routers stay the trailing block _unmount_feature_routers deletes.
    _mount_feature_ui_assets(app)

    # Dynamic router mounting: features contribute routers via get_router()
    _mount_feature_routers(app)

    # Server-side demo-mode classification (#766). Done after agents are
    # loaded so the rail knows whether to treat destructive ops as safe.
    from kestrel_sovereign.security.demo_isolation import classify_server_mode
    if getattr(app.state, "agent_manager", None):
        loaded = app.state.agent_manager.list_agents()
        app.state.demo_mode = classify_server_mode(loaded)
    elif getattr(app.state, "agent", None):
        app.state.demo_mode = classify_server_mode(
            {"_default": app.state.agent}
        )
    else:
        app.state.demo_mode = False
    if app.state.demo_mode:
        logger.info(
            "[demo-mode] this server is restricted to demo-scoped agents — "
            "destructive ops on live agents will be refused"
        )
    else:
        logger.info(
            "[demo-mode] live server — destructive ops on live agents "
            "require the X-Kestrel-Allow-Destructive header"
        )

    # --- Host-scoped features (issue #2293, consolidated onto server:app in
    # #2382) ---
    # Discover + mount host features at the host root (no agent prefix, no
    # get_agent dependency), aggregate their host-scoped UI, build the fleet
    # HostContext, and run their host lifecycle. Mounted UNCONDITIONALLY after
    # agent setup — host features are host-scoped and independent of single- vs
    # multi-agent mode. Reversible imperative failures remain isolated; an
    # invalid complete contribution set fails startup before mounted state is
    # changed.
    from kestrel_sovereign import host_features as _hf
    from kestrel_sdk.features import ContributionContractError
    from kestrel_sovereign.features.contribution_runtime import (
        FeatureContributionRuntimeError,
    )
    from kestrel_sovereign.paths import project_dir as _host_project_dir

    if not hasattr(app.state, "host_features"):
        app.state.host_features = []
    if not hasattr(app.state, "host_context"):
        app.state.host_context = None
    if not hasattr(app.state, "host_ui_manifest"):
        app.state.host_ui_manifest = []
    replacing_host_state = bool(app.state.host_features)
    candidate_ctx = None
    candidate_started = []
    try:
        # Resolve the host manifest from the resolved PROJECT_DIR (KESTREL_HOME /
        # marker walk-up / ~/.kestrel), NOT Path.cwd(). A service launched under
        # KESTREL_HOME, systemd, cron, or a direct path may have a CWD that
        # misses the real manifest — reading from CWD there would let a
        # host-disabled feature still mount (issue #2293 P2).
        features = _hf.instantiate_host_features(
            manifest_path=_host_project_dir() / _hf.HOST_MANIFEST_FILENAME,
        )
        if features:
            host_cfg = getattr(app.state, "multi_agent_config", None)
            ctx = await _hf.build_host_context(
                config=_host_config_mapping(host_cfg)
            )
            candidate_ctx = ctx
            # Validate and activate the complete prospective contribution set
            # before changing any already-valid mounted host surface.
            started_features = await _hf.start_host_features(features, ctx)
            # Backward-compatible with host integrations that still return
            # ``None`` from the lifecycle hook: the canonical runtime returns
            # the exact successfully-started set.
            if started_features is None:
                started_features = features
            candidate_started = list(started_features)
            if replacing_host_state:
                _hf.unmount_host_features(app)
            _hf.mount_host_feature_routers(app, candidate_started)
            _hf.mount_host_feature_ui(app, candidate_started)
            app.state.host_features = list(started_features)
            app.state.host_context = ctx
            runtime = getattr(ctx, "feature_contribution_runtime", None)
            if runtime is not None:
                app.state.host_operator_registry = runtime.operator_registry
                app.state.host_wait_registry = runtime.wait_registry
                app.state.host_signal_registry = runtime.source_registry
                app.state.host_permission_defaults_registry = (
                    runtime.permission_defaults_registry
                )
                app.state.host_setup_step_registry = runtime.setup_step_registry
            logger.info("Host features initialized: %d", len(started_features))
    except (ContributionContractError, FeatureContributionRuntimeError):
        # Complete prospective-set rejection is a startup failure, not an
        # optional-feature warning. No candidate was mounted and prior valid
        # state remains visible.
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("Host feature initialization failed: %s", exc)
        if candidate_ctx is not None and candidate_started:
            await _hf.stop_host_features(candidate_started, candidate_ctx)

    # Initialize OpenTelemetry tracing (no-op if packages not installed)
    setup_tracing(app)

    yield


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and cancellation-safe teardown for the server."""
    async with _lifespan_teardown_owner(app):
        async with _lifespan_startup(app):
            yield


app = FastAPI(lifespan=lifespan)
register_api_error_handlers(app)


# ---------------------------------------------------------------------------
# ASGI-level multi_agent routing for /api/agents/{name}/...
#
# Why this AND `agent_routing_middleware` below: FastAPI's
# @app.middleware("http") only fires on HTTP scope. WebSocket upgrades
# (e.g. /api/agents/Nellie/voice/chat) bypass it entirely, so the
# downstream WS handler can't resolve the agent and 4503's. This
# class-based ASGI middleware sees both http and websocket scopes and
# does the same prefix-strip + agent-resolve for either. The HTTP-only
# version downstream is kept as a safety net in case middleware order
# matters for some flow we haven't enumerated.
# ---------------------------------------------------------------------------


_AGENT_PATH_RE_ASGI = re.compile(r"^/api/agents/([^/]+)/(.+)$")


def _agent_not_found_response(
    agent_name: str,
    *,
    correlation_id: str | None = None,
) -> Response:
    """Return the shared public error contract for an unknown host agent."""
    message = f"Agent '{agent_name}' not found"
    return api_error_response(
        status_code=404,
        code="agent_not_found",
        message=message,
        correlation_id=correlation_id,
    )


class MultiAgentAgentRoutingMiddleware:
    """Strip /api/agents/{name}/ prefix + attach the resolved agent to scope.

    Works for both HTTP and WebSocket. For 404 (unknown agent) HTTP
    requests we synthesize a JSON 404; for WebSocket we close with
    code=4404 — close codes 4xxx are the application-defined range.
    """

    def __init__(self, asgi_app):
        self.app = asgi_app

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            return await self.app(scope, receive, send)

        agent_manager = getattr(app.state, "agent_manager", None)
        if agent_manager is None:
            return await self.app(scope, receive, send)

        path = scope.get("path", "")
        match = _AGENT_PATH_RE_ASGI.match(path)
        if not match:
            return await self.app(scope, receive, send)

        agent_name = match.group(1)
        agent = agent_manager.get_agent(agent_name)
        if agent is None:
            if scope["type"] == "http":
                response = _agent_not_found_response(
                    agent_name,
                    correlation_id=scope.get("state", {}).get("correlation_id"),
                )
                return await response(scope, receive, send)
            # WebSocket: accept then close with a clear reason — browsers see
            # the close code in the onclose event.
            await send({"type": "websocket.close", "code": 4404, "reason": "agent not found"})
            return

        # Mutate scope so downstream routes match the prefix-stripped path
        # and the handler can find the agent on `request.state` /
        # `websocket.state`. Starlette wires scope["state"] → both.
        scope["path"] = "/" + match.group(2)
        scope["raw_path"] = scope["path"].encode("utf-8")
        scope.setdefault("state", {})["agent"] = agent

        await self.app(scope, receive, send)


app.add_middleware(MultiAgentAgentRoutingMiddleware)


# Rate limiting
app.state.limiter = limiter


def canonical_rate_limit_exceeded_handler(
    request: Request,
    exc: RateLimitExceeded,
) -> Response:
    """Return a typed 429 while retaining SlowAPI's rate-limit headers."""
    response = api_error_response(
        status_code=429,
        code="rate_limit_exceeded",
        message=f"Rate limit exceeded: {exc.detail}",
        correlation_id=getattr(request.state, "correlation_id", None),
    )
    return request.app.state.limiter._inject_headers(
        response,
        request.state.view_rate_limit,
    )


app.add_exception_handler(RateLimitExceeded, canonical_rate_limit_exceeded_handler)


class CanonicalCORSMiddleware(CORSMiddleware):
    """Keep rejected preflights on the public API error contract.

    Starlette answers preflight requests inside ``CORSMiddleware`` without
    entering the inner request-context middleware.  Its stock rejection is a
    plain-text 400 with no support ID, so normalize only that response here
    while retaining the CORS headers Starlette already calculated.
    """

    def preflight_response(self, request_headers):
        response = super().preflight_response(request_headers)
        if response.status_code < 400:
            return response

        headers = {
            key: value
            for key, value in response.headers.items()
            if key.lower() not in {"content-length", "content-type"}
        }
        return api_error_response(
            status_code=400,
            code="cors_preflight_rejected",
            message="CORS preflight request rejected.",
            headers=headers,
            correlation_id=resolve_correlation_id(
                request_headers.get("X-Correlation-ID"),
                use_context=False,
            ),
        )

# Mount static files (disabled when running behind Kestrel Host)
SERVE_UI = os.environ.get("KESTREL_SERVE_UI", "true").lower() == "true"
# server.py now lives inside the package (kestrel_sovereign/server.py), so
# the static dir is a sibling — not under another kestrel_sovereign/ subdir.
STATIC_DIR = Path(__file__).parent / "static"
if SERVE_UI:
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    app.mount("/js", StaticFiles(directory=str(STATIC_DIR / "js")), name="js")
    app.mount("/shared", StaticFiles(directory=str(STATIC_DIR / "shared")), name="shared")
    app.mount("/utils", StaticFiles(directory=str(STATIC_DIR / "utils")), name="utils")

# Include core routers (always present, not feature-gated)
from kestrel_sovereign.endpoints import (
    agent_router,
    conversations_router,
    memories_router,
    sovereignty_router,
    database_router,
    models_router,
    commands_router,
    files_router,
    security_router,
    saved_items_router,
    metrics_router,
    observability_router,
    features_router,
    ui_router,
    github_router,
)
from kestrel_sovereign.endpoints.rasa_shim import router as rasa_shim_router

from kestrel_sovereign.endpoints.auth_oauth import router as auth_oauth_router, register_oauth, oauth
app.include_router(auth_oauth_router)
register_oauth(app)

# Canonical mount under /api/* (see #871). The deprecated /agent/* prefix
# is rewritten to /api/agent/* by a middleware below — we don't double-mount
# the router because that would defeat OpenAPI / route-inventory tooling.
app.include_router(agent_router)
app.include_router(conversations_router)
app.include_router(memories_router)
app.include_router(sovereignty_router)
app.include_router(database_router)
app.include_router(models_router)
app.include_router(commands_router)
app.include_router(files_router)
app.include_router(security_router)
app.include_router(saved_items_router)
app.include_router(metrics_router)
app.include_router(observability_router)
app.include_router(features_router)
app.include_router(ui_router)
app.include_router(rasa_shim_router)
app.include_router(github_router)


# Regex for multi-agent path routing: /api/agents/{name}/{remaining_path}
_AGENT_PATH_RE = re.compile(r"^/api/agents/([^/]+)/(.+)$")

# #871 — first-hit dedupe state for the deprecated /agent/* prefix shim.
# The middleware itself is registered at the end of this file so it runs
# OUTERMOST (Starlette runs middleware in reverse registration order); this
# is critical so the path rewrite happens BEFORE auth sees the request.
_DEPRECATED_AGENT_PREFIX_SEEN: set[tuple[str, str]] = set()


async def logging_context_middleware(request: Request, call_next):
    """Set request-scoped logging context (correlation ID, session ID, agent name).

    This middleware is registered explicitly after ``auth_middleware`` below.
    Starlette processes middleware in reverse order of addition, so correlation
    context exists before authentication and is cleaned up after the response.
    """
    # Preserve caller correlation only when it is safe for response headers,
    # logs, and support UI. Invalid/untrusted values receive a fresh ID.
    cid = resolve_correlation_id(
        request.headers.get("X-Correlation-ID"),
        use_context=False,
    )
    token_cid = correlation_id_var.set(cid)
    request.state.correlation_id = cid

    # Session ID from query params or headers (if available)
    sid = request.query_params.get("session_id") or request.headers.get("X-Session-ID")
    token_sid = session_id_var.set(sid) if sid else None

    # Agent name from app state (if initialized)
    agent = getattr(request.app.state, "agent", None)
    aname = getattr(agent, "name", None) if agent else None
    token_aname = agent_name_var.set(aname) if aname else None

    try:
        response = await call_next(request)
        response.headers["X-Correlation-ID"] = cid
        return response
    finally:
        correlation_id_var.reset(token_cid)
        if token_sid is not None:
            session_id_var.reset(token_sid)
        if token_aname is not None:
            agent_name_var.reset(token_aname)


@app.middleware("http")
async def request_metrics_middleware(request: Request, call_next):
    """Record Prometheus request count and latency metrics."""
    from kestrel_sdk.metrics import PROMETHEUS_AVAILABLE, REQUEST_COUNT, REQUEST_DURATION
    if not PROMETHEUS_AVAILABLE:
        return await call_next(request)

    import time as _time
    start = _time.monotonic()
    response = await call_next(request)
    duration = _time.monotonic() - start

    # Normalize path to avoid unbounded cardinality from path params
    path = request.url.path
    method = request.method
    status = str(response.status_code)

    REQUEST_COUNT.labels(method=method, path=path, status=status).inc()
    REQUEST_DURATION.labels(method=method, path=path).observe(duration)
    return response


@app.middleware("http")
async def static_cache_control(request: Request, call_next):
    """Prevent browser caching of JS/CSS files during development."""
    response = await call_next(request)
    path = request.url.path
    if path.endswith((".js", ".css")):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response


@app.middleware("http")
async def agent_routing_middleware(request: Request, call_next):
    """Route /api/agents/{name}/... requests to the correct in-process agent.

    In multi-agent mode:
    1. Extracts agent name from path prefix /api/agents/{name}/...
    2. Looks up agent in AgentManager
    3. Sets request.state.agent to the resolved agent
    4. Rewrites request path to strip the prefix

    In single-agent mode, this middleware is a no-op.
    """
    agent_manager = getattr(request.app.state, 'agent_manager', None)
    if agent_manager is None:
        return await call_next(request)

    path = request.url.path
    match = _AGENT_PATH_RE.match(path)
    if match:
        agent_name = match.group(1)
        remaining_path = "/" + match.group(2)

        agent = agent_manager.get_agent(agent_name)
        if agent is None:
            return _agent_not_found_response(
                agent_name,
                correlation_id=getattr(request.state, "correlation_id", None),
            )

        request.state.agent = agent
        # Rewrite path so existing routers see the original endpoint path
        request.scope["path"] = remaining_path

    return await call_next(request)


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Global authentication middleware.

    Accepts authentication via:
    1. API key (X-API-Key header, Bearer token, or query param) — for programmatic access
    2. OAuth session cookie — for browser access via Google sign-in
    """
    # Only the minimal readiness probe is public. Detailed health exposes
    # provider/model routing, disk capacity, agent state, and backend failure
    # messages, so it must pass the same API-key/OAuth checks as operator APIs
    # (#137, #2611).
    public_paths = [
        "/health",
        "/favicon.ico",
        "/api/auth/key",
        "/metrics",
        "/webhooks/github-app",
    ]
    auth_paths = ["/auth/login", "/auth/callback", "/auth/logout", "/auth/token"]
    # `/api/github/` was previously here — that exempted the GitHub-API
    # proxy at server.py:498 from auth, which let any unauthenticated
    # caller spend the server's GITHUB_TOKEN. Removed per the post-launch
    # review (2026-05-06). The proxy now requires API key or OAuth session
    # like every other /api/ endpoint.
    static_prefixes = ["/static", "/js/", "/shared/", "/utils/", "/api/ui/"]

    if request.url.path in public_paths or request.url.path in auth_paths:
        return await call_next(request)
    # Webhooks authenticate themselves (HMAC, bearer, etc.) — bypass API key auth.
    # Matches the bare /webhooks/{name} AND the per-agent
    # /api/agents/{id}/webhooks/{name} form (WEBHOOK_PATH_RE) so an external,
    # keyless caller can reach a specific agent's webhook (#2091/P3, #2382).
    if WEBHOOK_PATH_RE.match(request.url.path):
        return await call_next(request)
    if request.method == "OPTIONS":
        return await call_next(request)
    if SERVE_UI and any(request.url.path.startswith(p) for p in static_prefixes):
        return await call_next(request)
    # Out-of-tree feature UI assets are mounted at /features/{name}/static/ (#2043)
    # and loaded by raw browser mechanisms (<link href=...> / await import(mod))
    # that can't attach the X-API-Key header. Bypass auth for those static mounts,
    # matched narrowly as /features/{slug}/static/ so /features/ API routes (if any)
    # stay protected.
    #
    # NOT gated on SERVE_UI (unlike the core /static exemption above): the feature
    # asset mounts are installed unconditionally by _mount_feature_ui_assets(), so
    # they exist even on a host-managed agent (which runs with KESTREL_SERVE_UI=
    # false). In multi-agent host mode the browser's pinned import() reaches the
    # agent via the host's /api/agents/{id}/... proxy; the host bypasses its own
    # auth for these paths (FEATURE_STATIC_ASSET_RE) and forwards no key, so a
    # SERVE_UI gate here would 401 the header-less import() on every host agent.
    if FEATURE_STATIC_ASSET_RE.match(request.url.path):
        return await call_next(request)
    # Host-feature UI static assets (host-scoped surface, #2293/#2382) are loaded
    # header-less too — bypass API-key auth, matched narrowly so host-feature
    # *API* routes stay protected.
    from kestrel_sovereign.host_features import HOST_FEATURE_STATIC_ASSET_RE
    if HOST_FEATURE_STATIC_ASSET_RE.match(request.url.path):
        return await call_next(request)

    # Phoenix dynamic-import chunks request root-absolute /assets/… (outside the
    # /phoenix cookie scope). The handler is a pure 307 redirect back under
    # /phoenix — no data served — so it is safe header-less; the redirect target
    # still enforces the embed-cookie auth at the proxy. GET/HEAD only.
    if request.method in ("GET", "HEAD") and request.url.path.startswith("/assets/"):
        return await call_next(request)

    # Phoenix reverse-proxy embed auth (issue #2570). The console mints a
    # short-lived, signed, HttpOnly cookie (scoped to /phoenix) via
    # POST /api/host/phoenix/session — which itself requires standard host auth —
    # then loads the iframe. The browser can attach that cookie but not the
    # X-API-Key header. Accept it here IN ADDITION to the standard checks below,
    # and ONLY for /phoenix paths, so it never widens auth for any other route.
    # A logged-in console (session cookie / API key) can also reach /phoenix via
    # the standard path, so absence of the embed cookie just falls through.
    if request.url.path == "/phoenix" or request.url.path.startswith("/phoenix/"):
        from kestrel_sovereign.phoenix_supervisor import verify_embed_cookie
        from kestrel_sovereign.auth import CallerContext, AuthMethod

        if verify_embed_cookie(request, _SESSION_SECRET):
            request.state.caller = CallerContext.sovereign(AuthMethod.API_KEY)
            return await call_next(request)

    # Credential evaluation only happens inside this try. Dispatch of an
    # authenticated (or deliberately unauthenticated) request stays OUTSIDE
    # it, so an unhandled downstream application fault keeps FastAPI's
    # 500 semantics instead of masquerading as a 401 (#2490).
    from kestrel_sovereign.auth import CallerContext, AuthMethod

    caller = None
    unauthenticated_root_dispatch = False
    try:
        expected_key = get_api_key()

        # Check X-API-Key header
        api_key_header = request.headers.get(API_KEY_NAME)
        if api_key_header and secrets.compare_digest(api_key_header, expected_key):
            caller = CallerContext.sovereign(AuthMethod.API_KEY)

        # Check Bearer token (API key OR JWT)
        if caller is None:
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header[7:]
                # First try: API key match
                if secrets.compare_digest(token, expected_key):
                    caller = CallerContext.sovereign(AuthMethod.API_KEY)
                else:
                    # Second try: JWT token
                    try:
                        from kestrel_sovereign.endpoints.auth_oauth import _verify_jwt
                        jwt_payload = _verify_jwt(token)
                        if jwt_payload:
                            caller = CallerContext.authenticated(
                                identity=jwt_payload.get("sub", "unknown"),
                                auth_method=AuthMethod.JWT,
                            )
                    except Exception:
                        pass

        # Check query parameter for SSE endpoints only (EventSource can't send headers)
        # Restricted to SSE_PATHS to avoid leaking keys in URL logs on other endpoints.
        # Use scope["path"] rather than request.url.path: the deprecated_agent_prefix_compat
        # middleware rewrites scope["path"] before auth runs, but request.url caches the
        # original path if it was accessed earlier in the same middleware call chain.
        if caller is None:
            api_key_query = request.query_params.get("api_key")
            _scope_path = request.scope.get("path", request.url.path)
            # Browser-rendered images (an <img> can't set headers) need the same
            # query-param auth as SSE: the channel pairing-QR endpoint is fetched by
            # `<img src=...>` from the chat. Matched narrowly by suffix so the
            # query-key path doesn't widen to arbitrary endpoints (#1825).
            _query_key_ok = any(
                _scope_path == p or _scope_path.endswith(p) for p in SSE_PATHS
            ) or _scope_path.endswith("/link-qr.png")
            if api_key_query and _query_key_ok:
                if secrets.compare_digest(api_key_query, expected_key):
                    caller = CallerContext.sovereign(AuthMethod.API_KEY)

        # Check OAuth session cookie
        if caller is None:
            user_email = request.session.get("user_email") if hasattr(request, "session") else None
            if user_email:
                # CSRF: a cookie-authed state-changing request to a host-feature
                # endpoint must present a matching double-submit token. API-key /
                # bearer callers (handled above) are exempt — not CSRF-susceptible.
                # Scoped to host-feature paths so per-agent routes stay unchanged
                # (#2293/#2382).
                csrf_error = _enforce_host_csrf(request)
                if csrf_error is not None:
                    return csrf_error
                caller = CallerContext.authenticated(
                    identity=user_email,
                    auth_method=AuthMethod.OAUTH_SESSION,
                )

        # No valid auth — for the root page in a browser:
        if caller is None and request.url.path == "/" and SERVE_UI:
            accept = request.headers.get("accept", "")
            if "text/html" in accept:
                if _oauth_required() and "google" in oauth._clients:
                    return RedirectResponse(url="/auth/login", status_code=302)
                unauthenticated_root_dispatch = True

    except Exception as exc:
        logger.error(f"Auth error: {exc}")
        return api_error_response(
            status_code=401,
            code="authentication_failed",
            message="Authentication failed",
        )

    if caller is not None:
        request.state.caller = caller
        return await call_next(request)
    if unauthenticated_root_dispatch:
        return await call_next(request)
    return api_error_response(
        status_code=401,
        code="authentication_required",
        message="Invalid or missing API Key",
    )


# Register correlation/logging context AFTER auth so it executes OUTSIDE auth.
# Middleware-returned 401s therefore share the same request correlation ID as
# route/handler errors and receive the response header decoration as well.
app.middleware("http")(logging_context_middleware)


# Session middleware must be added AFTER auth_middleware so it's outermost
# (Starlette processes middleware in reverse order of addition)
from starlette.middleware.sessions import SessionMiddleware


def _get_session_secret() -> str:
    """Return the session signing secret.

    Priority:
        1. KESTREL_SESSION_SECRET env var (explicit session secret)
        2. KESTREL_API_KEY env var (shared API key as fallback)
        3. Random ephemeral secret (sessions won't survive restarts)

    Note: Starlette's SessionMiddleware already sets the httponly flag on
    session cookies, so JavaScript cannot access them.
    """
    secret = os.environ.get("KESTREL_SESSION_SECRET") or os.environ.get("KESTREL_API_KEY")
    if not secret:
        secret = secrets.token_urlsafe(32)
        logger.warning(
            "No KESTREL_SESSION_SECRET set — using random ephemeral secret "
            "(sessions won't survive restarts)"
        )
    return secret


# Captured once so the Phoenix embed cookie (issue #2570) is signed and verified
# with the SAME secret the session cookie uses — recomputing would return a fresh
# random value on the ephemeral-secret path and break verification.
_SESSION_SECRET = _get_session_secret()

app.add_middleware(
    SessionMiddleware,
    secret_key=_SESSION_SECRET,
    session_cookie="kestrel_session",
    max_age=7 * 24 * 3600,  # 7 days
    same_site="lax",
    https_only=os.environ.get("KESTREL_ENV", "development") == "production",
)


# FastAPI installs ServerErrorMiddleware outside all user middleware. Without
# this inner boundary, its 500 response bypasses CORSMiddleware and a
# cross-origin console cannot read either the canonical envelope or support ID.
# Register this after SessionMiddleware and before CORSMiddleware so CORS wraps
# every response produced here while this boundary still covers session, auth,
# routing, and endpoint failures.
@app.middleware("http")
async def canonical_unhandled_error_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as exc:
        return await api_unhandled_exception_handler(request, exc)


# CORS middleware — added last so it runs outermost (before auth/session).
# Origins from KESTREL_CORS_ORIGINS (comma-separated) or built-in defaults;
# build_cors_origins() rejects a wildcard since credentialed CORS is on.
from kestrel_sovereign.config import build_cors_origins
CORS_ORIGINS = build_cors_origins()
app.add_middleware(
    CanonicalCORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=[
        "Content-Type",
        "Authorization",
        "X-API-Key",
        "X-Correlation-ID",
        "X-CSRF-Token",
        "X-Kestrel-Allow-Destructive",
    ],
    expose_headers=[
        "X-Correlation-ID",
        "X-Request-ID",
        "X-Session-ID",
    ],
)


# #871 — Registered LAST so it wraps everything else. Starlette runs
# middleware in reverse registration order, so this is the OUTERMOST
# middleware: it sees /agent/* before auth, rewrites the scope to the
# canonical /api/agent/*, dispatches the rest of the stack, and decorates
# the response with RFC 8594 Deprecation/Sunset/Link headers. Drop this
# shim alongside the back-compat support window.
@app.middleware("http")
async def deprecated_agent_prefix_compat(request: Request, call_next):
    path = request.url.path
    if path.startswith("/agent/") or path == "/agent":
        rewritten = "/api" + path  # /agent/foo -> /api/agent/foo
        client = request.headers.get("user-agent", "?")
        key = (path, client)
        if key not in _DEPRECATED_AGENT_PREFIX_SEEN:
            _DEPRECATED_AGENT_PREFIX_SEEN.add(key)
            logger.warning(
                "deprecated /agent/* prefix used: path=%s ua=%s — migrate to %s (#871)",
                path,
                client,
                rewritten,
            )
        request.scope["path"] = rewritten
        request.scope["raw_path"] = rewritten.encode("utf-8")
        response = await call_next(request)
        response.headers["Deprecation"] = "true"
        response.headers["Sunset"] = "next-release"
        response.headers["Link"] = f'<{rewritten}>; rel="successor-version"'
        return response
    return await call_next(request)


# Registered after every other middleware so it is the outermost ASGI layer.
# Application/auth code sees the original query string; the scope is redacted
# only as ``http.response.start`` crosses back to Uvicorn's access logger.
from kestrel_sovereign.security.access_log import (
    SensitiveQueryStringRedactionMiddleware,
)

app.add_middleware(SensitiveQueryStringRedactionMiddleware)


if SERVE_UI:
    @app.get("/", response_class=HTMLResponse)
    async def read_root(request: Request):
        """Serve the main web UI."""
        try:
            with open(STATIC_DIR / "index.html", encoding="utf-8") as f:
                html = f.read()
        except FileNotFoundError:
            logger.error(f"{STATIC_DIR / 'index.html'} not found.")
            raise HTTPException(status_code=404, detail="Index file not found.")

        # #2041: seed window.KESTREL_UI_CONFIG.featureCapabilities before the
        # module scripts load so the capability set is known before app.js runs
        # feature registrations / nav gating. Only possible when a single agent
        # is resolvable here; in multi-agent host mode no agent is selected at
        # render, so seed `multiAgentHost` instead — app.js then skips the
        # un-prefixed boot fetches of /api/ui/capabilities and
        # /api/ui/contributions (which would 503 "Agent not initialized") and
        # lets selectAgent() run both once routing is pinned (#2048).
        agent = getattr(request.state, "agent", None) or getattr(
            request.app.state, "agent", None
        )
        if "</head>" in html:
            script = None
            if agent is not None:
                from kestrel_sovereign.ui_capabilities import render_ui_config_script

                script = render_ui_config_script(agent)
            elif getattr(request.app.state, "agent_manager", None) is not None:
                from kestrel_sovereign.ui_capabilities import (
                    render_multi_agent_host_config_script,
                )

                script = render_multi_agent_host_config_script()
            if script:
                html = html.replace("</head>", f"{script}\n</head>", 1)

        return HTMLResponse(content=html, status_code=200)


@app.get("/api/auth/key")
@limiter.limit("5/minute")
async def get_bootstrap_key(request: Request):
    """Return API key for initial frontend setup (localhost / Docker gateway only)."""
    if not _bootstrap_key_enabled():
        raise HTTPException(status_code=404, detail="API key bootstrap endpoint is disabled")

    # Narrowed from the whole 172.16.0.0/12 bridge range to loopback + the
    # Docker gateway (+ explicit KESTREL_BOOTSTRAP_ALLOWED_HOSTS) so a sibling
    # container can't fetch sovereign credentials (#1724).
    client_host = request.client.host if request.client else None
    if not is_bootstrap_host_allowed(client_host):
        logger.warning(f"Auth key request from non-allowed host: {client_host}")
        raise HTTPException(status_code=403, detail="API key bootstrap only accessible from localhost")

    response = {
        "key": get_api_key(),
        "header": API_KEY_NAME,
        "usage": "Include as 'X-API-Key' header or 'Authorization: Bearer <key>'"
    }
    # The public half is exposed only to the explicitly isolated Kite test
    # process.  Ordinary bootstrap callers never learn about the evidence
    # seam, and the signing module refuses to initialize without its private
    # test-home trust anchor.
    agent = getattr(app.state, "agent", None)
    if agent is None:
        manager = getattr(app.state, "agent_manager", None)
        getter = getattr(manager, "get_agent", None)
        agent = getter("kite") if callable(getter) else None
    if (
        bool(getattr(agent, "is_test_instance", False))
        and os.environ.get("KESTREL_KITE_RELEASE_EVIDENCE", "").strip().lower()
        in {"1", "true", "yes", "on"}
    ):
        from kestrel_sovereign.knowledge.kite_evidence_signing import kite_evidence_public_key

        response["kite_evidence_public_key"] = kite_evidence_public_key()
    return response


@app.get("/health")
def health_check(request: Request):
    """Return a stable, public-safe readiness probe for load balancers.

    This endpoint deliberately exposes only aggregate readiness and whether a
    runtime initialized. Operator diagnostics belong on the authenticated
    ``/health/detailed`` endpoint; never add agent names, provider/model data,
    failure records, reachability, exception strings, paths, or capacities to
    this response.
    """
    startup_error = getattr(request.app.state, "startup_error", None)
    agent = getattr(request.state, 'agent', None) or getattr(request.app.state, 'agent', None)
    mandatory_failures = getattr(
        request.app.state,
        "mandatory_feature_failures",
        [],
    )
    identity_failures = getattr(
        request.app.state,
        "identity_readiness_failures",
        [],
    )
    manager = getattr(request.app.state, 'agent_manager', None)
    # A failed startup may retain an internal manager solely so teardown can
    # finish closing its agents. It is not routeable and cannot make readiness
    # pass merely because it still has loaded entries.
    if startup_error:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "agent_initialized": False},
        )
    _latch_active_scheduler_runner_failures(request.app, agent, manager)
    scheduler_workers_available = _active_scheduler_workers_available(
        request.app, agent, manager
    )
    scheduler_failures = getattr(
        request.app.state,
        "scheduler_readiness_failures",
        [],
    )
    constitution_safe_mode = _constitution_safe_mode_records(agent, manager)
    any_initialized = bool(agent) or bool(manager and manager.list_agents())
    if (
        mandatory_failures
        or identity_failures
        or constitution_safe_mode
        or scheduler_failures
        or not scheduler_workers_available
    ):
        return JSONResponse(
            status_code=503,
            content={
                # Do not reveal whether the cause is identity, a mandatory
                # feature, safe mode, or a pending constitutional audit.
                "status": "unhealthy",
                "agent_initialized": any_initialized,
            },
        )
    if agent:
        return {"status": "ok", "agent_initialized": True}
    # In multi-agent mode, check if any agents are loaded
    if any_initialized:
        return {
            "status": "ok",
            "agent_initialized": True,
        }
    return JSONResponse(
        status_code=503,
        content={
            "status": "unhealthy" if startup_error else "degraded",
            "agent_initialized": False,
        },
    )


async def _phoenix_tracing_status(app) -> dict:
    """Phoenix tracing status for /health/detailed (#2690).

    Surfaces whether the trace backend is enabled, whether it is currently
    reachable, and — critically — *why* it is off when supervision setup failed
    (``phoenix_disabled_reason``). This makes a silent-off boot visible to
    operators instead of only manifesting as a mint 503 and dead embeds. Never
    raises: a probe error degrades to ``reachable: False``.
    """
    supervisor = getattr(app.state, "phoenix", None)
    disabled_reason = getattr(app.state, "phoenix_disabled_reason", None)
    reachable = False
    if supervisor is not None:
        try:
            reachable = await supervisor.is_reachable()
        except Exception:  # noqa: BLE001 - health must never crash on a probe
            reachable = False
        if reachable and disabled_reason is not None:
            # Phoenix recovered after a transient startup stall (e.g. a slow
            # first-boot schema build that outran the background budget) — clear
            # the now-stale disabled reason so health reflects live state (#2690).
            app.state.phoenix_disabled_reason = None
            disabled_reason = None
    return {
        "enabled": supervisor is not None,
        "reachable": reachable,
        "disabled_reason": disabled_reason,
    }


async def _agent_detailed_health(agent) -> dict:
    """Compute the detailed health result for a single agent.

    Prefers the agent's ``HealthFeature.get_latest()`` (its cached liveness
    result); falls back to running the check suite directly when the feature is
    absent. The returned dict always carries at least ``status`` and ``checks``.
    Shared by the single-agent and multi-agent (``agent_manager``) branches of
    ``/health/detailed`` so a managed fleet is evaluated with the same logic as
    a lone default agent (#2698).
    """
    features = getattr(agent, 'features', {})
    health_feature = None
    for feat in features.values() if isinstance(features, dict) else features:
        if feat.__class__.__name__ == "HealthFeature":
            health_feature = feat
            break

    if health_feature:
        result = await health_feature.get_latest()
        if isinstance(result, dict):
            return result
        return {"status": "unhealthy", "checks": []}

    # Fallback: run checks directly without the feature. Shares HealthFeature's
    # list rather than repeating it — the two copies had already drifted, and a
    # check missing here reports `healthy` for a state the other calls a
    # warning. HealthFeature is not mandatory, so this path is reachable by
    # design.
    from kestrel_sovereign.features.health.checks import (
        derive_overall_status,
        run_standard_checks,
    )
    from kestrel_sovereign.features.storage_access import (
        resolve_feature_database,
    )

    # HealthFeature uses the feature-internal database resolver during its
    # own initialization.  The fallback must honor that same boundary: a real
    # agent's public ``storage.db`` is a privacy-governed view whose
    # ``backend`` property is intentionally unavailable, while the database
    # health check needs backend lifecycle state to diagnose retained workers.
    db = resolve_feature_database(agent)

    checks = await run_standard_checks(agent, db)
    overall = derive_overall_status(checks)
    return {"status": overall, "checks": checks}


def _roll_up_fleet_status(agent_statuses: list) -> str:
    """Three-state rollup over a managed fleet's per-agent statuses (#2698).

    - ``healthy`` only when every managed agent is healthy.
    - ``unhealthy`` only when zero managed agents are healthy.
    - ``degraded`` otherwise (some healthy, some not — a partial outage).

    This mirrors the single-agent path's warn/fail -> degraded/unhealthy
    semantics so a partial fleet outage is surfaced at the top level rather than
    masked by one healthy peer.
    """
    if not agent_statuses:
        return "unhealthy"
    healthy = sum(1 for status in agent_statuses if status == "healthy")
    if healthy == len(agent_statuses):
        return "healthy"
    if healthy == 0:
        return "unhealthy"
    return "degraded"


@app.get("/health/detailed")
async def health_detailed(request: Request):
    """Authenticated operator diagnostics using the HealthFeature.

    Returns individual check results for database, LLM service,
    memory system, disk space, and context budget. Global auth middleware
    requires an API key, JWT, or OAuth session before this handler runs.
    """
    agent = getattr(request.state, 'agent', None) or getattr(request.app.state, 'agent', None)
    manager = getattr(request.app.state, "agent_manager", None)
    # Tracing status is surfaced on every branch so an off/unreachable trace
    # backend (and its disabled reason) is visible regardless of agent health.
    tracing = await _phoenix_tracing_status(request.app)
    if getattr(request.app.state, "startup_error", None):
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": "Server startup failed",
                "checks": [],
                "tracing": tracing,
            },
        )
    _latch_active_scheduler_runner_failures(request.app, agent, manager)
    safe_mode_records = _constitution_safe_mode_records(agent, manager)
    if safe_mode_records:
        return JSONResponse(
            status_code=503,
            content={
                "status": "restricted",
                "constitution_safe_mode": safe_mode_records,
                "checks": [],
                "tracing": tracing,
            },
        )
    scheduler_workers_available = _active_scheduler_workers_available(
        request.app, agent, manager
    )
    scheduler_failures = getattr(
        request.app.state, "scheduler_readiness_failures", []
    )
    if scheduler_failures or not scheduler_workers_available:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": "Scheduler unavailable",
                "scheduler_readiness_failures": scheduler_failures,
                "checks": [],
                "tracing": tracing,
            },
        )
    if agent:
        result = await _agent_detailed_health(agent)
        if isinstance(result, dict):
            result.setdefault("tracing", tracing)
        return result

    # No singleton default agent (multi-agent deployments set app.state.agent to
    # None). Resolve health from the live fleet rather than reporting a false
    # total outage while managed agents serve traffic (#2698).
    managed = manager.list_agents() if manager is not None else {}
    if managed:
        import asyncio

        names = list(managed.keys())
        results = await asyncio.gather(
            *(_agent_detailed_health(a) for a in managed.values()),
            return_exceptions=True,
        )
        breakdown: dict = {}
        for name, res in zip(names, results):
            if isinstance(res, BaseException):
                breakdown[name] = {
                    "status": "unhealthy",
                    "checks": [],
                    "error": str(res),
                }
            else:
                breakdown[name] = {
                    "status": res.get("status", "unhealthy"),
                    "checks": res.get("checks", []),
                }
        overall = _roll_up_fleet_status(
            [entry["status"] for entry in breakdown.values()]
        )
        return {
            "status": overall,
            "agents": breakdown,
            "checks": [],
            "tracing": tracing,
            "scheduler_cold_agent_failures": getattr(
                request.app.state, "scheduler_cold_agent_failures", []
            ),
        }

    return {
        "status": "unhealthy",
        "error": "No agent available",
        "checks": [],
        "tracing": tracing,
    }


def _enforce_host_csrf(request: Request):
    """Enforce double-submit CSRF on cookie-authed state-changing host-feature
    requests. Returns a 403 ``JSONResponse`` on failure, else ``None``.

    Only host-feature router paths are checked so per-agent routes and the
    existing management endpoints keep their current behavior (#2293/#2382).
    """
    from kestrel_sovereign.security.csrf import CSRFError, enforce_csrf
    from kestrel_sovereign.host_features import is_host_feature_path

    if request.method not in {"POST", "PUT", "PATCH", "DELETE"}:
        return None
    # Auth runs before the multi-agent routing middleware strips
    # ``/api/agents/{name}``.  Normalize that public alias here so a
    # cookie-authenticated request cannot bypass host-feature CSRF simply by
    # spelling the same route through an agent prefix (#2382 review).
    path = request.scope.get("path", request.url.path)
    match = _AGENT_PATH_RE.match(path)
    if match:
        path = "/" + match.group(2)
    if not is_host_feature_path(request.app, path):
        return None
    try:
        enforce_csrf(request, authed_via_cookie=True)
    except CSRFError as exc:
        return api_error_response(
            status_code=403,
            code="csrf_failed",
            message=exc.detail,
        )
    return None


@app.get("/api/host/ui/contributions")
async def host_ui_contributions(request: Request):
    """Host-scoped UI manifest (distinct from per-agent /api/ui/contributions).

    Lists the ES modules each host feature contributes so the console can load
    a host-scoped panel surface. Empty when no host feature ships UI (#2293).
    """
    manifest = getattr(request.app.state, "host_ui_manifest", []) or []
    return {"contributions": manifest}


@app.get("/api/host/csrf")
async def host_csrf_token(request: Request):
    """Issue (or refresh) the double-submit CSRF cookie and return its token.

    The console reads the token from this response (or the cookie) and echoes it
    in the ``X-CSRF-Token`` header on state-changing host-feature requests.
    """
    from kestrel_sovereign.security.csrf import (
        CSRF_COOKIE_NAME,
        generate_csrf_token,
        issue_csrf_cookie,
    )

    token = request.cookies.get(CSRF_COOKIE_NAME) or generate_csrf_token()
    response = JSONResponse(content={"csrf_token": token})
    issue_csrf_cookie(response, token)
    return response


# ---------------------------------------------------------------------------
# Phoenix trace backend: embed-session mint + same-origin reverse proxy (#2570)
# ---------------------------------------------------------------------------

_PHOENIX_PROXY_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]


def _phoenix_supervisor(request: Request):
    """Return the supervised Phoenix, or ``None`` when not enabled/running."""
    return getattr(request.app.state, "phoenix", None)


@app.post("/api/host/phoenix/session")
async def phoenix_embed_session(request: Request):
    """Mint the short-lived embed cookie for the Phoenix iframe (issue #2570).

    Requires standard host auth (``X-API-Key`` or session) — already enforced by
    ``auth_middleware`` before this handler runs. Returns an HttpOnly,
    SameSite=Lax cookie scoped to ``/phoenix`` and never puts a credential in a
    URL. The console calls this with headers, then loads ``/phoenix/``.
    """
    from kestrel_sovereign.phoenix_supervisor import (
        EMBED_COOKIE_PATH,
        EMBED_TTL_SECONDS,
        issue_embed_cookie,
    )

    supervisor = _phoenix_supervisor(request)
    # Health is reachability, not child liveness (#2589): probe the Phoenix port
    # so the mint tracks whatever is actually serving it, not a possibly-zombie
    # tracked child. 503 until Phoenix answers (e.g. during first-boot schema
    # creation).
    if supervisor is None or not await supervisor.is_reachable():
        return _phoenix_not_enabled_json()

    caller = getattr(request.state, "caller", None)
    identity = getattr(caller, "identity", None) or "sovereign"
    secure = os.environ.get("KESTREL_ENV", "development") == "production"

    response = JSONResponse(
        content={
            "ok": True,
            "embed_path": EMBED_COOKIE_PATH + "/",
            "expires_in": EMBED_TTL_SECONDS,
        }
    )
    issue_embed_cookie(response, _SESSION_SECRET, identity=identity, secure=secure)
    return response


def _phoenix_not_enabled_json() -> JSONResponse:
    from kestrel_sovereign.phoenix_supervisor import _phoenix_disabled_response

    return _phoenix_disabled_response()


@app.api_route("/phoenix", methods=_PHOENIX_PROXY_METHODS)
@app.api_route("/phoenix/{path:path}", methods=_PHOENIX_PROXY_METHODS)
async def phoenix_proxy(request: Request, path: str = ""):
    """Same-origin authenticated reverse proxy to the local Phoenix UI (#2570).

    Auth (session cookie, API key, or the minted embed cookie) is enforced by
    ``auth_middleware`` before this runs. Streams the response through so the UI
    functions end-to-end. Returns a clear 503 when Phoenix is not supervised.
    """
    from kestrel_sovereign.phoenix_supervisor import proxy_to_phoenix

    supervisor = _phoenix_supervisor(request)
    return await proxy_to_phoenix(request, supervisor, path)


@app.get("/assets/{path:path}")
@app.head("/assets/{path:path}")
async def phoenix_assets_redirect(request: Request, path: str):
    """Redirect Phoenix's root-absolute dynamic-import chunks into ``/phoenix``.

    Phoenix's Vite bundle honours the root path in its static HTML tags
    (``/phoenix/assets/...``) but *dynamically imported* chunks resolve from the
    build-time base — root-absolute ``/assets/...`` — landing outside the
    ``/phoenix`` cookie scope (observed: ``vendor-shiki``/``rolldown-runtime``
    → 401, broken lazy views). A 307 sends the browser back under ``/phoenix``
    where the embed cookie applies and the authenticated proxy serves the real
    chunk. No data is served here; auth stays enforced at the proxy.
    """
    if _phoenix_supervisor(request) is None:
        raise HTTPException(status_code=404)
    return RedirectResponse(url=f"/phoenix/assets/{path}", status_code=307)


def _server_host(value: str) -> str:
    """Return a non-empty Uvicorn host value for argparse."""
    host = value.strip()
    if not host:
        raise argparse.ArgumentTypeError("host must not be empty")
    return host


def _server_port(value: str) -> int:
    """Return a valid TCP port for argparse."""
    try:
        port = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("port must be an integer") from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def _server_argument_parser(
    environ: Mapping[str, str] | None = None,
) -> argparse.ArgumentParser:
    """Build the direct-server CLI with explicit environment precedence."""
    from kestrel_sovereign.multi_agent.config import (
        DEFAULT_HOST_BIND,
        DEFAULT_HOST_PORT,
    )

    env = os.environ if environ is None else environ
    parser = argparse.ArgumentParser(
        prog="python -m kestrel_sovereign.server",
        description="Run the Kestrel Sovereign ASGI server.",
    )
    parser.add_argument(
        "--host",
        type=_server_host,
        default=env.get(_SERVER_HOST_ENV, DEFAULT_HOST_BIND),
        metavar="HOST",
        help=(
            "interface to bind (precedence: --host, KESTREL_SERVER_HOST, "
            f"{DEFAULT_HOST_BIND})"
        ),
    )
    parser.add_argument(
        "--port",
        type=_server_port,
        default=env.get(_SERVER_PORT_ENV, str(DEFAULT_HOST_PORT)),
        metavar="PORT",
        help=(
            "TCP port to bind (precedence: --port, PORT, "
            f"{DEFAULT_HOST_PORT})"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the packaged server entry point.

    Resolved values are mirrored into the environment so the application
    lifespan and Uvicorn observe the same effective host/port. ``argparse``
    rejects unsupported or invalid arguments before any socket is opened.
    """
    args = _server_argument_parser().parse_args(argv)
    os.environ[_SERVER_HOST_ENV] = args.host
    os.environ[_SERVER_PORT_ENV] = str(args.port)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
