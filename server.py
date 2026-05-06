#!/usr/bin/env python3
"""
A FastAPI server to expose Kestrel agent functionality as a service.
"""
import ipaddress
import os
import secrets
from typing import Optional
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from contextlib import asynccontextmanager
import logging
from main import get_agent_did_async
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.llm.service import LLMService
from dotenv import load_dotenv
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from kestrel_sovereign.rate_limit import limiter

import re

from kestrel_sovereign.kestrel_config.constants import SHUTDOWN_TIMEOUT
from kestrel_sovereign.telemetry import setup_tracing

# Load environment variables from .env file
# override=False: Don't clobber env vars already set by ProcessManager
# (e.g., KESTREL_DB_PATH is set per-agent in multi_agent mode)
load_dotenv(Path(__file__).parent / ".env", override=False)

from kestrel_sovereign.logging_config import (
    setup_logging,
    correlation_id_var,
    session_id_var,
    agent_name_var,
    get_correlation_id,
)

setup_logging()
logger = logging.getLogger(__name__)

# Security Configuration
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
security = HTTPBearer(auto_error=False)

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

    def _collect_routers_from_agent(agent) -> None:
        features = getattr(agent, "features", {})
        if not features:
            return
        for name, feature in features.items():
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the application's lifespan."""
    import asyncio
    logger.info("Server starting up...")
    _set_startup_error(app, None)

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
            manager = AgentManager(base_data_dir=Path.cwd())
            loaded = await manager.load_from_config(config)
            app.state.agent_manager = manager
            app.state.agent = None  # No single default agent
            logger.info(f"Multi-agent mode: {loaded} agent(s) loaded")
        except Exception as e:
            logger.error(f"Error during multi-agent startup: {e}", exc_info=True)
            app.state.agent_manager = None
            app.state.agent = None
            _set_startup_error(app, e)
    else:
        # --- Single-agent mode (original behavior) ---
        app.state.agent_manager = None
        try:
            db_backend = os.environ.get("KESTREL_DB_BACKEND", "sqlite")
            database_url = os.environ.get("KESTREL_DATABASE_URL")

            if db_backend.lower() == "postgres" and database_url:
                logger.info("Using PostgreSQL backend for Kestrel")
                storage_dir = os.environ.get("KESTREL_DB_PATH", os.getcwd())
                db_path = os.path.join(storage_dir, "kestrel_prime.db")
                agent_did = await get_agent_did_async(storage_dir)
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
                llm_service = LLMService()
                app.state.agent = KestrelAgent(
                    did=agent_did,
                    storage_path=db_path,
                    llm_service=llm_service,
                )
                logger.info(f"Using SQLite backend for Kestrel: {db_path}")

            await app.state.agent.initialize()
            logger.info(f"Kestrel Agent initialized and ready (backend: {db_backend})")
        except Exception as e:
            logger.error(f"Error during startup: {e}", exc_info=True)
            app.state.agent = None
            _set_startup_error(app, e)

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

    # Initialize OpenTelemetry tracing (no-op if packages not installed)
    setup_tracing(app)

    yield

    # Shutdown
    logger.info("Server shutting down...")
    _unmount_feature_routers(app)
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


app = FastAPI(lifespan=lifespan)


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
                from starlette.responses import JSONResponse as _JR
                response = _JR(
                    status_code=404,
                    content={"detail": f"Agent '{agent_name}' not found"},
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
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Mount static files (disabled when running behind Kestrel Host)
SERVE_UI = os.environ.get("KESTREL_SERVE_UI", "true").lower() == "true"
STATIC_DIR = Path(__file__).parent / "kestrel_sovereign" / "static"
if SERVE_UI:
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    app.mount("/js", StaticFiles(directory=str(STATIC_DIR / "js")), name="js")
    app.mount("/shared", StaticFiles(directory=str(STATIC_DIR / "shared")), name="shared")
    app.mount("/utils", StaticFiles(directory=str(STATIC_DIR / "utils")), name="utils")

# Include core routers (always present, not feature-gated)
from endpoints import (
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
    features_router,
    ui_router,
)
from endpoints.rasa_shim import router as rasa_shim_router

from endpoints.auth_oauth import router as auth_oauth_router, register_oauth, oauth
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
app.include_router(features_router)
app.include_router(ui_router)
app.include_router(rasa_shim_router)


# --- GitHub API Proxy (for Portfolio Dashboard) ---

@app.get("/api/github/{path:path}")
async def github_proxy(path: str, request: Request):
    """Proxy GitHub API requests using server-side GITHUB_TOKEN."""
    import httpx

    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        env_path = Path(__file__).parent / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("GITHUB_TOKEN="):
                    token = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not token:
        return JSONResponse({"error": "No GITHUB_TOKEN configured"}, status_code=503)

    gh_url = f"https://api.github.com/{path}"
    if request.url.query:
        gh_url += f"?{request.url.query}"

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(
                gh_url,
                headers={
                    "Authorization": f"token {token}",
                    "Accept": "application/vnd.github.v3+json",
                    "User-Agent": "kestrel-host",
                },
                timeout=httpx.Timeout(connect=5.0, read=15.0, write=5.0, pool=5.0),
            )
            return JSONResponse(content=resp.json(), status_code=resp.status_code)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=502)


# Regex for multi-agent path routing: /api/agents/{name}/{remaining_path}
_AGENT_PATH_RE = re.compile(r"^/api/agents/([^/]+)/(.+)$")

# #871 — first-hit dedupe state for the deprecated /agent/* prefix shim.
# The middleware itself is registered at the end of this file so it runs
# OUTERMOST (Starlette runs middleware in reverse registration order); this
# is critical so the path rewrite happens BEFORE auth sees the request.
_DEPRECATED_AGENT_PREFIX_SEEN: set[tuple[str, str]] = set()


@app.middleware("http")
async def logging_context_middleware(request: Request, call_next):
    """Set request-scoped logging context (correlation ID, session ID, agent name).

    Starlette processes middleware in reverse order of addition, so this
    runs early (before auth/routing) and cleans up after the response.
    """
    # Correlation ID: prefer incoming header, else generate
    cid = request.headers.get("X-Correlation-ID") or get_correlation_id()
    token_cid = correlation_id_var.set(cid)

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
    from kestrel_sovereign.metrics import PROMETHEUS_AVAILABLE, REQUEST_COUNT, REQUEST_DURATION
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
            return JSONResponse(
                status_code=404,
                content={"detail": f"Agent '{agent_name}' not found"},
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
    public_paths = ["/health", "/health/detailed", "/favicon.ico", "/api/auth/key", "/metrics", "/webhooks/github-app"]
    auth_paths = ["/auth/login", "/auth/callback", "/auth/logout", "/auth/token"]
    # `/api/github/` was previously here — that exempted the GitHub-API
    # proxy at server.py:498 from auth, which let any unauthenticated
    # caller spend the server's GITHUB_TOKEN. Removed per the post-launch
    # review (2026-05-06). The proxy now requires API key or OAuth session
    # like every other /api/ endpoint.
    static_prefixes = ["/static", "/js/", "/shared/", "/utils/", "/api/ui/"]

    if request.url.path in public_paths or request.url.path in auth_paths:
        return await call_next(request)
    # Webhooks authenticate themselves (HMAC, bearer, etc.) — bypass API key auth
    if request.url.path.startswith("/webhooks/"):
        return await call_next(request)
    if request.method == "OPTIONS":
        return await call_next(request)
    if SERVE_UI and any(request.url.path.startswith(p) for p in static_prefixes):
        return await call_next(request)

    try:
        expected_key = get_api_key()

        from kestrel_sovereign.auth import CallerContext, AuthMethod

        # Check X-API-Key header
        api_key_header = request.headers.get(API_KEY_NAME)
        if api_key_header and secrets.compare_digest(api_key_header, expected_key):
            request.state.caller = CallerContext.sovereign(AuthMethod.API_KEY)
            return await call_next(request)

        # Check Bearer token (API key OR JWT)
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]
            # First try: API key match
            if secrets.compare_digest(token, expected_key):
                request.state.caller = CallerContext.sovereign(AuthMethod.API_KEY)
                return await call_next(request)
            # Second try: JWT token
            try:
                from endpoints.auth_oauth import _verify_jwt
                jwt_payload = _verify_jwt(token)
                if jwt_payload:
                    request.state.caller = CallerContext.authenticated(
                        identity=jwt_payload.get("sub", "unknown"),
                        auth_method=AuthMethod.JWT,
                    )
                    return await call_next(request)
            except Exception:
                pass

        # Check query parameter for SSE endpoints only (EventSource can't send headers)
        # Restricted to SSE_PATHS to avoid leaking keys in URL logs on other endpoints.
        # Use scope["path"] rather than request.url.path: the deprecated_agent_prefix_compat
        # middleware rewrites scope["path"] before auth runs, but request.url caches the
        # original path if it was accessed earlier in the same middleware call chain.
        api_key_query = request.query_params.get("api_key")
        _scope_path = request.scope.get("path", request.url.path)
        if api_key_query and any(_scope_path == p or _scope_path.endswith(p) for p in SSE_PATHS):
            if secrets.compare_digest(api_key_query, expected_key):
                request.state.caller = CallerContext.sovereign(AuthMethod.API_KEY)
                return await call_next(request)

        # Check OAuth session cookie
        user_email = request.session.get("user_email") if hasattr(request, "session") else None
        if user_email:
            request.state.caller = CallerContext.authenticated(
                identity=user_email,
                auth_method=AuthMethod.OAUTH_SESSION,
            )
            return await call_next(request)

        # No valid auth — for the root page in a browser:
        if request.url.path == "/" and SERVE_UI:
            accept = request.headers.get("accept", "")
            if "text/html" in accept:
                if _oauth_required() and "google" in oauth._clients:
                    return RedirectResponse(url="/auth/login", status_code=302)
                else:
                    return await call_next(request)

        return JSONResponse(content={"detail": "Invalid or missing API Key"}, status_code=401)

    except Exception as exc:
        logger.error(f"Auth error: {exc}")
        return JSONResponse(content={"detail": "Authentication failed"}, status_code=401)


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


app.add_middleware(
    SessionMiddleware,
    secret_key=_get_session_secret(),
    session_cookie="kestrel_session",
    max_age=7 * 24 * 3600,  # 7 days
    same_site="lax",
    https_only=os.environ.get("KESTREL_ENV", "development") == "production",
)

# CORS middleware — added last so it runs outermost (before auth/session).
# Override defaults via KESTREL_CORS_ORIGINS (comma-separated).
_DEFAULT_CORS_ORIGINS = [
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8888",
    "http://127.0.0.1:8888",
    "https://kestrelsovereignai.github.io",
]
_cors_env = os.environ.get("KESTREL_CORS_ORIGINS", "")
CORS_ORIGINS = [o.strip() for o in _cors_env.split(",") if o.strip()] if _cors_env else _DEFAULT_CORS_ORIGINS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
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


if SERVE_UI:
    @app.get("/", response_class=HTMLResponse)
    async def read_root(request: Request):
        """Serve the main web UI."""
        try:
            with open(STATIC_DIR / "index.html", encoding="utf-8") as f:
                return HTMLResponse(content=f.read(), status_code=200)
        except FileNotFoundError:
            logger.error(f"{STATIC_DIR / 'index.html'} not found.")
            raise HTTPException(status_code=404, detail="Index file not found.")


def _is_docker_network(host: str) -> bool:
    """Check if the host IP is within Docker's internal network range (172.16.0.0/12)."""
    try:
        return ipaddress.ip_address(host) in ipaddress.ip_network("172.16.0.0/12")
    except ValueError:
        return False


@app.get("/api/auth/key")
@limiter.limit("5/minute")
async def get_bootstrap_key(request: Request):
    """Return API key for initial frontend setup (localhost only)."""
    if not _bootstrap_key_enabled():
        raise HTTPException(status_code=404, detail="API key bootstrap endpoint is disabled")

    client_host = request.client.host if request.client else None
    allowed_hosts = {"127.0.0.1", "localhost", "::1", "172.17.0.1"}
    is_docker_internal = client_host and _is_docker_network(client_host)

    if client_host not in allowed_hosts and not is_docker_internal:
        logger.warning(f"Auth key request from non-local host: {client_host}")
        raise HTTPException(status_code=403, detail="API key bootstrap only accessible from localhost")

    return {
        "key": get_api_key(),
        "header": API_KEY_NAME,
        "usage": "Include as 'X-API-Key' header or 'Authorization: Bearer <key>'"
    }


@app.get("/health")
def health_check(request: Request):
    """A simple health check endpoint."""
    startup_error = getattr(request.app.state, "startup_error", None)
    agent = getattr(request.state, 'agent', None) or getattr(request.app.state, 'agent', None)
    if agent:
        return {"status": "ok", "agent_initialized": True}
    # In multi-agent mode, check if any agents are loaded
    manager = getattr(request.app.state, 'agent_manager', None)
    if manager and manager.list_agents():
        return {"status": "ok", "agent_initialized": True}
    payload = {"status": "unhealthy" if startup_error else "degraded", "agent_initialized": False}
    if startup_error:
        payload["error"] = startup_error
    return JSONResponse(status_code=503, content=payload)


@app.get("/health/detailed")
async def health_detailed(request: Request):
    """Detailed liveness check using the HealthFeature.

    Returns individual check results for database, LLM service,
    memory system, disk space, and context budget.
    """
    agent = getattr(request.state, 'agent', None) or getattr(request.app.state, 'agent', None)
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
            check_database, check_llm_service, check_memory_system,
            check_disk_space, check_context_budget,
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


# Stripe Crypto On-Ramp webhook endpoint
_stripe_webhook_handler = None


def get_stripe_webhook_handler():
    """Lazy initialization of Stripe webhook handler.

    Phase 6 of #889 wires `handler.on_deposit_complete` to the
    dispatcher path: completed deposits resolve the owning agent (by
    `OnRampSession.agent_did` in multi-agent mode, or app.state.agent
    for single-agent) and invoke `agent.on_stripe_deposit_complete`,
    which builds an UNTRUSTED COGNITION signal and `enqueue_signal`s
    it. Without this wire, the `agent.on_stripe_deposit_complete`
    method exists but is never called by real Stripe events
    (caught in #906 review P1).
    """
    global _stripe_webhook_handler
    if _stripe_webhook_handler is None:
        try:
            from kestrel_feature_wallet.onramp import StripeOnRamp, StripeWebhookHandler
            onramp = StripeOnRamp()
            _stripe_webhook_handler = StripeWebhookHandler(onramp)
            _stripe_webhook_handler.on_deposit_complete = _on_stripe_deposit_complete
            logger.info("Stripe webhook handler initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Stripe webhook handler: {e}")
    return _stripe_webhook_handler


async def _on_stripe_deposit_complete(session) -> None:
    """Route a completed Stripe on-ramp deposit to the owning agent.

    Resolution order:
    1. Multi-agent: AgentManager scan for an agent whose .did matches
       `session.agent_did`. Webhooks are global (Stripe sends to one
       URL) so the session's recorded owner is the routing key.
    2. Single-agent: app.state.agent fallback.

    Failures are logged and swallowed — Stripe's record-of-truth is
    the on-ramp DB update that already ran inside the handler before
    this callback fires. Never break the webhook's success path on a
    dispatcher hiccup.
    """
    target_agent = None
    target_did = getattr(session, "agent_did", None)

    manager = getattr(app.state, "agent_manager", None)
    if manager is not None and target_did:
        try:
            for _name, agent in manager.list_agents().items():
                if getattr(agent, "did", None) == target_did:
                    target_agent = agent
                    break
        except Exception as e:
            logger.warning(
                "AgentManager lookup failed for stripe deposit %s: %s",
                getattr(session, "session_id", "<unknown>"), e,
            )

    if target_agent is None:
        target_agent = getattr(app.state, "agent", None)

    if target_agent is None or not hasattr(
        target_agent, "on_stripe_deposit_complete"
    ):
        logger.warning(
            "Stripe deposit %s acknowledged but no agent could be "
            "resolved to wake (looked for did=%r); cognition signal "
            "will not fire",
            getattr(session, "session_id", "<unknown>"),
            target_did,
        )
        return

    try:
        await target_agent.on_stripe_deposit_complete(session)
    except Exception as e:
        logger.error(
            "Failed to dispatch stripe deposit signal for session %s: %s",
            getattr(session, "session_id", "<unknown>"), e,
            exc_info=True,
        )


@app.post("/webhooks/stripe/crypto")
async def stripe_crypto_webhook(request: Request):
    """
    Handle Stripe Crypto On-Ramp webhook events.

    Stripe sends events when:
    - On-ramp session status changes
    - Purchase completes successfully
    - Purchase fails

    Security: Validates webhook signature using STRIPE_WEBHOOK_SECRET.
    """
    handler = get_stripe_webhook_handler()
    if not handler:
        logger.error("Stripe webhook handler not available")
        return JSONResponse(
            content={"error": "Webhook handler not configured"},
            status_code=503
        )

    try:
        payload = await request.body()
        signature = request.headers.get("Stripe-Signature")

        result = await handler.handle_webhook(payload, signature)

        if result.success:
            return {"status": "received", "message": result.message}
        else:
            logger.error(f"Webhook processing failed: {result.error}")
            return JSONResponse(
                content={"error": result.error},
                status_code=400
            )
    except Exception as e:
        logger.error(f"Webhook error: {e}", exc_info=True)
        return JSONResponse(
            content={"error": "Internal server error"},
            status_code=500
        )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8888))
    uvicorn.run(app, host="0.0.0.0", port=port)
