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

# Load environment variables from .env file
# override=False: Don't clobber env vars already set by ProcessManager
# (e.g., KESTREL_DB_PATH is set per-agent in rookery mode)
load_dotenv(Path(__file__).parent / ".env", override=False)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security Configuration
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
security = HTTPBearer(auto_error=False)

# Paths where API key query parameter auth is allowed
# (EventSource/SSE can't send headers, so these endpoints need query param auth)
SSE_PATHS = {"/agent/notifications/sse", "/agent/stream"}


def _set_startup_error(app: FastAPI, error: Optional[Exception]) -> None:
    """Persist startup failure state for diagnostics and health endpoints."""
    app.state.startup_error = str(error) if error else None


def _bootstrap_key_enabled() -> bool:
    """Return whether localhost API-key bootstrap is explicitly enabled."""
    return os.environ.get("KESTREL_ENABLE_API_KEY_BOOTSTRAP", "").lower() in {
        "1", "true", "yes", "on"
    }


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the application's lifespan."""
    import asyncio
    logger.info("Server starting up...")
    _set_startup_error(app, None)

    # Detect multi-agent mode
    multi_agent_env = os.environ.get("KESTREL_MULTI_AGENT", "").lower() in ("1", "true", "yes")
    rookery_path = Path(os.environ.get("KESTREL_ROOKERY_CONFIG", "rookery.toml"))

    if multi_agent_env or rookery_path.exists():
        # --- Multi-agent mode ---
        try:
            from kestrel_sovereign.rookery.agent_manager import AgentManager
            from kestrel_sovereign.rookery.config import RookeryConfig

            config = RookeryConfig.load(
                str(rookery_path) if rookery_path.exists() else None,
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

    yield

    # Shutdown
    logger.info("Server shutting down...")
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

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Mount static files (disabled when running behind Kestrel Host)
SERVE_UI = os.environ.get("KESTREL_SERVE_UI", "true").lower() == "true"
if SERVE_UI:
    app.mount("/static", StaticFiles(directory="static"), name="static")
    app.mount("/js", StaticFiles(directory="static/js"), name="js")
    app.mount("/shared", StaticFiles(directory="static/shared"), name="shared")
    app.mount("/utils", StaticFiles(directory="static/utils"), name="utils")

# Include routers
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
    observability_router,
    saved_items_router,
)

from endpoints.auth_oauth import router as auth_oauth_router, register_oauth, oauth
app.include_router(auth_oauth_router)
register_oauth(app)

app.include_router(agent_router)
app.include_router(conversations_router)
app.include_router(memories_router)
app.include_router(sovereignty_router)
app.include_router(database_router)
app.include_router(models_router)
app.include_router(commands_router)
app.include_router(files_router)
app.include_router(security_router)
app.include_router(observability_router)
app.include_router(saved_items_router)


# Regex for multi-agent path routing: /api/agents/{name}/{remaining_path}
_AGENT_PATH_RE = re.compile(r"^/api/agents/([^/]+)/(.+)$")


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
    public_paths = ["/health", "/health/detailed", "/favicon.ico", "/webhooks/stripe/crypto"]
    if _bootstrap_key_enabled():
        public_paths.append("/api/auth/key")
    auth_paths = ["/auth/login", "/auth/callback", "/auth/logout"]
    static_prefixes = ["/static", "/js/", "/shared/", "/utils/"]

    if request.url.path in public_paths or request.url.path in auth_paths:
        return await call_next(request)
    if SERVE_UI and any(request.url.path.startswith(p) for p in static_prefixes):
        return await call_next(request)

    try:
        expected_key = get_api_key()

        # Check X-API-Key header
        api_key_header = request.headers.get(API_KEY_NAME)
        if api_key_header and secrets.compare_digest(api_key_header, expected_key):
            return await call_next(request)

        # Check Bearer token
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]
            if secrets.compare_digest(token, expected_key):
                return await call_next(request)

        # Check query parameter for SSE endpoints only (EventSource can't send headers)
        # Restricted to SSE_PATHS to avoid leaking keys in URL logs on other endpoints
        api_key_query = request.query_params.get("api_key")
        if api_key_query and any(request.url.path.endswith(p) for p in SSE_PATHS):
            if secrets.compare_digest(api_key_query, expected_key):
                return await call_next(request)

        # Check OAuth session cookie
        user_email = request.session.get("user_email") if hasattr(request, "session") else None
        if user_email:
            return await call_next(request)

        # No valid auth — for the root page in a browser:
        if request.url.path == "/" and SERVE_UI:
            accept = request.headers.get("accept", "")
            if "text/html" in accept:
                # KESTREL_REQUIRE_OAUTH: explicitly opt in to OAuth login.
                # Set this in Cloud Run deploy scripts to force Google
                # sign-in for browser access. Without it, the UI is
                # served directly using API key auth.
                if os.environ.get("KESTREL_REQUIRE_OAUTH") and "google" in oauth._clients:
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


if SERVE_UI:
    @app.get("/", response_class=HTMLResponse)
    async def read_root(request: Request):
        """Serve the main web UI."""
        try:
            with open("static/index.html", encoding="utf-8") as f:
                return HTMLResponse(content=f.read(), status_code=200)
        except FileNotFoundError:
            logger.error("static/index.html not found.")
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
    """Detailed health check using the HeartbeatFeature.

    Returns individual check results for database, LLM service,
    memory system, disk space, and context budget.
    """
    agent = getattr(request.state, 'agent', None) or getattr(request.app.state, 'agent', None)
    if not agent:
        return {"status": "unhealthy", "error": "No agent available", "checks": []}

    # Find the HeartbeatFeature among the agent's features
    features = getattr(agent, 'features', {})
    heartbeat_feature = None
    for feat in features.values() if isinstance(features, dict) else features:
        if feat.__class__.__name__ == "HeartbeatFeature":
            heartbeat_feature = feat
            break

    if not heartbeat_feature:
        # Fallback: run checks directly without the feature
        from kestrel_sovereign.features.heartbeat.checks import (
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

    result = await heartbeat_feature.get_latest_heartbeat()
    return result


# Stripe Crypto On-Ramp webhook endpoint
_stripe_webhook_handler = None


def get_stripe_webhook_handler():
    """Lazy initialization of Stripe webhook handler."""
    global _stripe_webhook_handler
    if _stripe_webhook_handler is None:
        try:
            from kestrel_sovereign.features.wallet.onramp import StripeOnRamp, StripeWebhookHandler
            onramp = StripeOnRamp()
            _stripe_webhook_handler = StripeWebhookHandler(onramp)
            logger.info("Stripe webhook handler initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Stripe webhook handler: {e}")
    return _stripe_webhook_handler


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
