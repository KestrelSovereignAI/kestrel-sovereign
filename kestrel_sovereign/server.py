#!/usr/bin/env python3
"""
A FastAPI server to expose Kestrel agent functionality as a service.
"""
import argparse
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
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from kestrel_sovereign.main import get_agent_did_async
from kestrel_sovereign.kestrel_agent import KestrelAgent
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


def _mount_feature_routers(app: FastAPI) -> None:
    """Mount routers contributed by discovered features.

    After agent initialization, iterate over all registered features and
    call ``feature.get_router()``. If a feature returns a router, include
    it in the FastAPI app. This allows feature packages (voice, spawn,
    observability, etc.) to contribute HTTP endpoints dynamically —
    disabling a feature cleanly removes its routes.

    Tracks the number of routes added so they can be removed on shutdown
    via ``_unmount_feature_routers``.
    """
    routes_before = len(app.routes)
    mounted = []
    # Webhook receivers are collected across every agent and served by ONE
    # shared /webhooks/{name} dispatch router. Each WebhookFeature's own
    # get_router() produces an identical /webhooks/{name} catch-all, so
    # mounting them per-agent would let the first-mounted agent shadow all the
    # others — every webhook owned by a later agent would 404 (issue #2089).
    webhook_receivers = []

    def _is_webhook_receiver(receiver) -> bool:
        # Duck-typed so core server.py stays decoupled from the feature class.
        return (
            receiver is not None
            and hasattr(receiver, "handle_webhook")
            and hasattr(receiver, "webhooks")
        )

    def _collect_routers_from_agent(agent) -> None:
        features = getattr(agent, "features", {})
        if not features:
            return
        for name, feature in features.items():
            receiver = getattr(feature, "receiver", None)
            if _is_webhook_receiver(receiver):
                if receiver not in webhook_receivers:
                    webhook_receivers.append(receiver)
                continue
            try:
                router = feature.get_router()
                if router is not None:
                    app.include_router(router)
                    mounted.append(name)
            except Exception as exc:
                logger.warning("Failed to mount router from feature %s: %s", name, exc)

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

    # Mount one cross-agent webhook dispatch router for all collected receivers.
    if webhook_receivers:
        try:
            from kestrel_sovereign.features.webhooks.receiver import (
                build_webhook_dispatch_router,
            )

            app.include_router(
                build_webhook_dispatch_router(lambda: list(webhook_receivers))
            )
            mounted.append("webhooks")
        except Exception as exc:
            logger.warning("Failed to mount webhook dispatch router: %s", exc)

    # Record how many routes were added so shutdown can remove them
    app.state._feature_route_count = len(app.routes) - routes_before

    if mounted:
        logger.info("Dynamically mounted routers from features: %s", ", ".join(mounted))


def _unmount_feature_routers(app: FastAPI) -> None:
    """Remove dynamically-mounted feature routes added by ``_mount_feature_routers``.

    This prevents route accumulation when the app lifespan restarts
    (e.g. across TestClient sessions in the same pytest process).
    """
    count = getattr(app.state, "_feature_route_count", 0)
    if count > 0:
        del app.routes[-count:]
        app.state._feature_route_count = 0
        logger.info("Removed %d dynamically-mounted feature routes", count)


def _mount_feature_ui_assets(app: FastAPI) -> None:
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
    pending: list = []

    def _collect(agent) -> None:
        # include_disabled=True: mount every feature that declares a static_dir,
        # even one that starts disabled, so enabling it at runtime from the
        # Feature Store serves its JS without a restart (the runtime-enable 404,
        # #2048). The manifest at GET /api/ui/contributions still lists only
        # enabled features, so a disabled feature's mount is dormant until enabled.
        for mount_path, directory in feature_static_mounts(agent, include_disabled=True):
            if mount_path in seen:
                continue
            seen.add(mount_path)
            pending.append((mount_path, directory))

    agent = getattr(app.state, "agent", None)
    if agent is not None:
        _collect(agent)
    manager = getattr(app.state, "agent_manager", None)
    if manager is not None:
        for agent_name in manager.list_agents():
            a = manager.get_agent(agent_name)
            if a is not None:
                _collect(a)

    added = []
    for mount_path, directory in pending:
        try:
            app.mount(
                mount_path,
                StaticFiles(directory=directory),
                name=f"feature-ui:{mount_path}",
            )
            added.append(app.routes[-1])
        except Exception as exc:  # noqa: BLE001 - never block startup on one feature
            logger.warning("Failed to mount feature UI assets at %s: %s", mount_path, exc)

    app.state._feature_ui_mounts = added
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


def _host_config_mapping(config) -> dict:
    """Read-only host config mapping handed to host features via HostContext.

    Deliberately minimal: enough for a host feature to learn the host's
    bind/port and agent roster without coupling to the full config object
    shape. Carries only the tenant resolver (below) when no multi-agent config
    is available (e.g. a single-agent boot) — host features are host-scoped and
    must still mount. (Moved from the retired ``kestrel_sovereign.host`` — issue
    #2382.)

    Always injects the identity→tenant resolver (issue #2444) under
    ``observability_tenant_resolver`` — the seam the fleet observability host
    feature consumes in ``on_host_start`` to stamp each request's ``tenant_id``.
    It is present even on a single-agent boot (config ``None``) so the store is
    tenant-scoped regardless of deployment shape; zero-config resolves every
    request to one stable default personal tenant (INV-SOLO).
    """
    from kestrel_sovereign.security.tenant_resolver import (
        HOST_CONFIG_KEY as _TENANT_RESOLVER_KEY,
        build_tenant_resolver,
    )

    mapping: dict = {_TENANT_RESOLVER_KEY: build_tenant_resolver()}
    if config is None:
        return mapping
    try:
        mapping.update(
            {
                "host_bind": config.host.bind,
                "host_port": config.host.port,
                "agents": list(config.agents.keys()),
            }
        )
    except Exception:  # noqa: BLE001
        pass
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


async def _supervise_phoenix_startup(supervisor) -> None:
    """Bring Phoenix up in the background so host boot never blocks on it (#2589).

    Runs ``supervisor.start`` (adopt-or-reap + spawn) off the event loop, then
    polls reachability. It only re-invokes ``start`` if the child process died
    outright — a still-starting child (creating its SQLite schema on first boot)
    reports ``running`` and is left alone. Adopt-or-reap makes re-invocation
    safe: a healthy orphan is adopted rather than duplicated.
    """
    import asyncio
    import time as _time

    try:
        # Initial adopt-or-reap + spawn, off the event loop (subprocess launch +
        # a short reachability probe).
        await asyncio.to_thread(supervisor.start, wait_for_health=False)
        deadline = _time.monotonic() + _PHOENIX_STARTUP_BUDGET_SECONDS
        while _time.monotonic() < deadline:
            if await supervisor.is_reachable():
                logger.info("Phoenix trace backend is reachable.")
                return
            if not supervisor.running:
                # The child exited before binding — try a fresh (re)start.
                logger.info("Phoenix child not running yet — restarting.")
                await asyncio.to_thread(supervisor.start, wait_for_health=False)
            await asyncio.sleep(2.0)
        logger.warning(
            "Phoenix did not become reachable within %.0fs — /phoenix returns "
            "503 until it recovers.",
            _PHOENIX_STARTUP_BUDGET_SECONDS,
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - Phoenix must never crash the host
        logger.warning("Phoenix background supervision error: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the application's lifespan."""
    import asyncio
    logger.info("Server starting up...")
    _set_startup_error(app, None)
    app.state.mandatory_feature_failures = []
    app.state.identity_readiness_failures = []

    # --- Host-supervised Phoenix trace backend (issue #2570) ---
    # Launch Phoenix BEFORE agents so the OTLP endpoint is on os.environ when
    # in-process agents initialize and when subprocess agents inherit the env
    # (INV-SOLO zero-config wiring). Fully degrades when arize-phoenix is not
    # installed or KESTREL_PHOENIX_ENABLED=0 — the host is unaffected.
    app.state.phoenix = None
    app.state.phoenix_task = None
    try:
        from kestrel_sovereign.phoenix_supervisor import (
            PhoenixSupervisor,
            autowire_otel_project,
            autowire_otlp_endpoint,
            should_supervise_phoenix,
        )

        if should_supervise_phoenix():
            supervisor = PhoenixSupervisor()
            # Privacy gate (#2609): establish/migrate private trace custody
            # before advertising a local OTLP collector to the host or agents.
            # A failure leaves app.state.phoenix unset and the endpoint unwired,
            # so tracing is clearly disabled rather than writing insecurely.
            await asyncio.to_thread(supervisor.prepare_storage)
            # Track the supervisor immediately so /phoenix + the mint can gate on
            # reachability (503 until Phoenix answers). Zero-config (INV-SOLO):
            # default the OTLP endpoint to the local Phoenix collector for this
            # process AND for env inherited by spawned agents, unless the operator
            # already set it.
            app.state.phoenix = supervisor
            endpoint = autowire_otlp_endpoint(os.environ)
            if endpoint:
                logger.info(
                    "OTEL_EXPORTER_OTLP_ENDPOINT auto-set to local Phoenix (%s)",
                    endpoint,
                )
            # Group host + spawned-agent traces under a single Phoenix project
            # (obs#32). Mutates os.environ so subprocess agents inherit it; the
            # SDK's tracing bootstrap (>= 0.30.2) reads KESTREL_OTEL_PROJECT.
            project = autowire_otel_project(os.environ)
            if project:
                logger.info(
                    "KESTREL_OTEL_PROJECT auto-set to '%s' (fleet-scoped traces)",
                    project,
                )
            # Non-blocking first boot (#2589): Phoenix's first start (SQLite
            # schema creation) can exceed the launcher's health window. Bring it
            # up in the BACKGROUND (adopt-or-reap + retry) so the host's own
            # /health never waits on Phoenix; /phoenix returns 503 until ready.
            app.state.phoenix_task = asyncio.create_task(
                _supervise_phoenix_startup(supervisor)
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
        logger.warning("Phoenix supervision setup failed: %s", exc)
        app.state.phoenix = None

    # Detect multi-agent mode
    multi_agent_env = os.environ.get("KESTREL_MULTI_AGENT", "").lower() in ("1", "true", "yes")
    multi_agent_path = resolve_multi_agent_path(os.environ)

    if multi_agent_env or multi_agent_path.exists():
        # --- Multi-agent mode ---
        try:
            from kestrel_sovereign.multi_agent.agent_manager import AgentManager
            from kestrel_sovereign.multi_agent.config import MultiAgentConfig

            config = MultiAgentConfig.load(
                str(multi_agent_path) if multi_agent_path.exists() else None,
                auto_discover_fallback=True,
            )
            _apply_platform_host_port(config, os.environ)
            manager = AgentManager(base_data_dir=Path.cwd())
            loaded = await manager.load_from_config(config)
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
            logger.info(f"Multi-agent mode: {loaded} agent(s) loaded")

            # A2A sender verification (#1705): the host holds every agent, so it
            # builds the same-host DID registry and injects the resolver into
            # each agent (consumed by /tasks/send verification, #1673). Local
            # same-host resolution by default; federated did:web opt-in via
            # KESTREL_A2A_FEDERATED_DID.
            try:
                from kestrel_sovereign.a2a.did_registry import install_a2a_did_resolver

                federated = os.environ.get("KESTREL_A2A_FEDERATED_DID", "").lower() in (
                    "1", "true", "yes",
                )
                install_a2a_did_resolver(manager, federated_fallback=federated)
            except Exception as exc:  # noqa: BLE001 - never block startup on this
                logger.warning("Could not install A2A DID resolver: %s", exc)
            # Fleet-idleness (#F235) is wired at the AgentManager's single agent
            # registration point (agent_manager._load_one), so every agent —
            # startup or spawned — gets the co-hosted-agents provider and no
            # dynamically-added agent can bypass the whole-host-restart gate.

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
        except Exception as e:
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
            app.state.agent_manager = None
            app.state.agent = None
            _set_startup_error(app, e)
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
    # multi-agent mode. Isolated in try/except so a host-feature failure never
    # blocks the host from serving agents.
    from kestrel_sovereign import host_features as _hf
    from kestrel_sovereign.paths import project_dir as _host_project_dir

    app.state.host_features = []
    app.state.host_context = None
    app.state.host_ui_manifest = []
    try:
        # Resolve the host manifest from the resolved PROJECT_DIR (KESTREL_HOME /
        # marker walk-up / ~/.kestrel), NOT Path.cwd(). A service launched under
        # KESTREL_HOME, systemd, cron, or a direct path may have a CWD that
        # misses the real manifest — reading from CWD there would let a
        # host-disabled feature still mount (issue #2293 P2).
        features = _hf.instantiate_host_features(
            manifest_path=_host_project_dir() / _hf.HOST_MANIFEST_FILENAME,
        )
        app.state.host_features = features
        if features:
            host_cfg = getattr(app.state, "multi_agent_config", None)
            ctx = await _hf.build_host_context(
                config=_host_config_mapping(host_cfg)
            )
            app.state.host_context = ctx
            _hf.mount_host_feature_routers(app, features)
            _hf.mount_host_feature_ui(app, features)
            await _hf.start_host_features(features, ctx)
            logger.info("Host features initialized: %d", len(features))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Host feature initialization failed: %s", exc)

    # Initialize OpenTelemetry tracing (no-op if packages not installed)
    setup_tracing(app)

    yield

    # Shutdown
    logger.info("Server shutting down...")
    # Stop host features first (mirror of startup), then agents.
    try:
        _host_features = getattr(app.state, "host_features", []) or []
        _host_ctx = getattr(app.state, "host_context", None)
        if _host_features and _host_ctx is not None:
            await _hf.stop_host_features(_host_features, _host_ctx)
        _hf.unmount_host_features(app)
        # Close the host backend / session factory if opened.
        _sf = getattr(_host_ctx, "session_factory", None) if _host_ctx is not None else None
        if _sf is not None:
            await _sf.close()
        _hdb = getattr(_host_ctx, "db", None) if _host_ctx is not None else None
        if _hdb is not None and hasattr(_hdb, "close"):
            await _hdb.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Host feature shutdown failed: %s", exc)
    _unmount_feature_routers(app)
    _unmount_feature_ui_assets(app)
    if getattr(app.state, 'agent_manager', None):
        await app.state.agent_manager.shutdown_all()
        logger.info("All agents shutdown complete.")
    elif getattr(app.state, 'agent', None):
        try:
            await asyncio.wait_for(app.state.agent.shutdown(), timeout=SHUTDOWN_TIMEOUT)
            logger.info("Agent shutdown complete.")
        except asyncio.TimeoutError:
            logger.warning("Agent shutdown timed out (5s)")
        except asyncio.CancelledError:
            logger.debug("Agent shutdown cancelled")
        except Exception as e:
            logger.warning(f"Error during agent shutdown: {e}")

    # Stop the supervised Phoenix subprocess (mirror of startup).
    _phoenix_task = getattr(app.state, "phoenix_task", None)
    if _phoenix_task is not None:
        _phoenix_task.cancel()
        try:
            await _phoenix_task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
        app.state.phoenix_task = None
    _phoenix = getattr(app.state, "phoenix", None)
    if _phoenix is not None:
        try:
            await _phoenix.aclose()
            _phoenix.stop()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error stopping Phoenix: %s", exc)
        app.state.phoenix = None


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

    return {
        "key": get_api_key(),
        "header": API_KEY_NAME,
        "usage": "Include as 'X-API-Key' header or 'Authorization: Bearer <key>'"
    }


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
    constitution_safe_mode = _constitution_safe_mode_records(agent, manager)
    any_initialized = bool(agent) or bool(manager and manager.list_agents())
    if mandatory_failures or identity_failures or constitution_safe_mode:
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
        return {"status": "ok", "agent_initialized": True}
    return JSONResponse(
        status_code=503,
        content={
            "status": "unhealthy" if startup_error else "degraded",
            "agent_initialized": False,
        },
    )


@app.get("/health/detailed")
async def health_detailed(request: Request):
    """Authenticated operator diagnostics using the HealthFeature.

    Returns individual check results for database, LLM service,
    memory system, disk space, and context budget. Global auth middleware
    requires an API key, JWT, or OAuth session before this handler runs.
    """
    agent = getattr(request.state, 'agent', None) or getattr(request.app.state, 'agent', None)
    manager = getattr(request.app.state, "agent_manager", None)
    safe_mode_records = _constitution_safe_mode_records(agent, manager)
    if safe_mode_records:
        return JSONResponse(
            status_code=503,
            content={
                "status": "restricted",
                "constitution_safe_mode": safe_mode_records,
                "checks": [],
            },
        )
    if not agent:
        return {"status": "unhealthy", "error": "No agent available", "checks": []}

    features = getattr(agent, 'features', {})
    health_feature = None
    for feat in features.values() if isinstance(features, dict) else features:
        if feat.__class__.__name__ == "HealthFeature":
            health_feature = feat
            break

    if not health_feature:
        # Fallback: run checks directly without the feature
        from kestrel_sovereign.features.health.checks import (
            check_bootstrap_state, check_context_budget, check_database,
            check_disk_space, check_llm_service, check_memory_system,
        )
        db = None
        if hasattr(agent, 'storage') and agent.storage:
            db = getattr(agent.storage, 'db', None)

        checks = [
            await check_database(db),
            await check_llm_service(agent),
            await check_memory_system(agent),
            await check_disk_space(),
            await check_context_budget(agent),
            await check_bootstrap_state(agent),
        ]
        statuses = [c.get("status") for c in checks]
        if "fail" in statuses:
            overall = "unhealthy"
        elif "warn" in statuses:
            overall = "degraded"
        else:
            overall = "healthy"
        return {"status": overall, "checks": checks}

    return await health_feature.get_latest()


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
