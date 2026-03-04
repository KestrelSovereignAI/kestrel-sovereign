#!/usr/bin/env python3
"""
A FastAPI server to expose Kestrel agent functionality as a service.
"""
import os
import secrets
from typing import Optional
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from contextlib import asynccontextmanager
import logging
from main import get_agent_did_async
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.llm.service import LLMService
from dotenv import load_dotenv
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

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

    if api_key_header and api_key_header == expected_key:
        return True
    if token and token.credentials == expected_key:
        return True

    # Support query parameter auth for SSE endpoints (EventSource can't send headers)
    api_key_query = request.query_params.get("api_key")
    if api_key_query and api_key_query == expected_key:
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
    try:
        # Check database backend configuration
        db_backend = os.environ.get("KESTREL_DB_BACKEND", "sqlite")
        database_url = os.environ.get("KESTREL_DATABASE_URL")

        if db_backend.lower() == "postgres" and database_url:
            # PostgreSQL mode - for cloud deployments
            logger.info("Using PostgreSQL backend for Kestrel")
            storage_dir = os.environ.get("KESTREL_DB_PATH", os.getcwd())
            db_path = os.path.join(storage_dir, "kestrel_prime.db")  # For SQLite fallback stores
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
            # SQLite mode (default) - for local deployments
            storage_dir = os.environ.get("KESTREL_DB_PATH", os.getcwd())
            db_path = os.path.join(storage_dir, "kestrel_prime.db")
            agent_did = await get_agent_did_async(storage_dir)
            llm_service = LLMService()
            app.state.agent = KestrelAgent(
                did=agent_did,
                storage_path=db_path,
                llm_service=llm_service
            )
            logger.info(f"Using SQLite backend for Kestrel: {db_path}")

        await app.state.agent.initialize()
        logger.info(f"Kestrel Agent initialized and ready (backend: {db_backend})")
    except Exception as e:
        logger.error(f"Error during startup: {e}", exc_info=True)
    yield
    logger.info("Server shutting down...")
    if hasattr(app.state, 'agent') and app.state.agent:
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
limiter = Limiter(key_func=get_remote_address)
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

from endpoints.auth_oauth import router as auth_oauth_router, register_oauth
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


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Global authentication middleware.

    Accepts authentication via:
    1. API key (X-API-Key header, Bearer token, or query param) — for programmatic access
    2. OAuth session cookie — for browser access via Google sign-in
    """
    public_paths = ["/health", "/api/auth/key", "/api/models", "/api/model/current", "/api/identity", "/api/commands", "/favicon.ico", "/webhooks/stripe/crypto"]
    auth_paths = ["/auth/login", "/auth/callback", "/auth/logout"]
    static_prefixes = ["/static", "/api/files/", "/js/", "/shared/", "/utils/"]

    if request.url.path in public_paths or request.url.path in auth_paths:
        return await call_next(request)
    if SERVE_UI and any(request.url.path.startswith(p) for p in static_prefixes):
        return await call_next(request)

    try:
        expected_key = get_api_key()

        # Check X-API-Key header
        api_key_header = request.headers.get(API_KEY_NAME)
        if api_key_header and api_key_header == expected_key:
            return await call_next(request)

        # Check Bearer token
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]
            if token == expected_key:
                return await call_next(request)

        # Check query parameter (for SSE endpoints - EventSource can't send headers)
        api_key_query = request.query_params.get("api_key")
        if api_key_query and api_key_query == expected_key:
            return await call_next(request)

        # Check OAuth session cookie
        user_email = request.session.get("user_email") if hasattr(request, "session") else None
        if user_email:
            return await call_next(request)

        # No valid auth — for the root page, redirect browsers to login
        if request.url.path == "/" and SERVE_UI:
            accept = request.headers.get("accept", "")
            if "text/html" in accept:
                return RedirectResponse(url="/auth/login", status_code=302)

        return JSONResponse(content={"detail": "Invalid or missing API Key"}, status_code=401)

    except Exception as exc:
        logger.error(f"Auth error: {exc}")
        return JSONResponse(content={"detail": "Authentication failed"}, status_code=401)


# Session middleware must be added AFTER auth_middleware so it's outermost
# (Starlette processes middleware in reverse order of addition)
from starlette.middleware.sessions import SessionMiddleware
app.add_middleware(
    SessionMiddleware,
    secret_key=os.environ.get("KESTREL_SESSION_SECRET") or os.environ.get("KESTREL_API_KEY", "kestrel-dev-session-key"),
    session_cookie="kestrel_session",
    max_age=7 * 24 * 3600,  # 7 days
    same_site="lax",
    https_only=os.environ.get("KESTREL_ENV", "development") == "production",
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


@app.get("/api/auth/key")
async def get_bootstrap_key(request: Request):
    """Return API key for initial frontend setup (localhost only)."""
    client_host = request.client.host if request.client else None
    allowed_hosts = {"127.0.0.1", "localhost", "::1", "172.17.0.1"}
    is_docker_internal = client_host and client_host.startswith("172.")

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
    if hasattr(request.app.state, 'agent') and request.app.state.agent:
        return {"status": "ok", "agent_initialized": True}
    return {"status": "ok", "agent_initialized": False}


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
