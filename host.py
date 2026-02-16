#!/usr/bin/env python3
"""
Kestrel Host - Thin FastAPI proxy + static file server.

The host is NOT an agent. It has no DID, no memory, no LLM.
It is pure infrastructure that:
1. Serves the Kestrel UI (static files)
2. Aggregates agent discovery (A2A cards)
3. Proxies API requests to the correct agent process

Architecture:
    Browser → localhost:8888 (host.py)
                  ├── /static/*              → serves UI files directly
                  ├── /api/agents            → aggregates A2A cards from all agents
                  └── /api/agents/{id}/*     → proxies to agent on its port

Key principle: No agent is privileged. The host treats all agents as equal peers.
"""

import os
import secrets
import logging
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from kestrel_sovereign.rookery.config import (
    RookeryConfig,
    LocalAgentConfig,
    RemoteAgentConfig,
)
from kestrel_sovereign.rookery.proxy import (
    proxy_request_streaming,
    get_agent_base_url,
)

load_dotenv(Path(__file__).parent / ".env", override=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security Configuration
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
security = HTTPBearer(auto_error=False)


def get_api_key() -> str:
    """Get or generate the host API key."""
    api_key = os.environ.get("KESTREL_HOST_API_KEY") or os.environ.get("KESTREL_API_KEY")
    if not api_key:
        generated_key = secrets.token_urlsafe(32)
        os.environ["KESTREL_API_KEY"] = generated_key
        logger.warning("No KESTREL_HOST_API_KEY or KESTREL_API_KEY set. Generated a temporary key.")
        return generated_key
    return api_key


def load_rookery_config() -> RookeryConfig:
    """Load rookery configuration from file or auto-discover."""
    config_path = os.environ.get("KESTREL_ROOKERY_CONFIG")
    if config_path:
        return RookeryConfig.load(config_path)
    return RookeryConfig.load()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the host application lifecycle."""
    logger.info("Kestrel Host starting up...")

    config = load_rookery_config()
    app.state.rookery_config = config
    app.state.http_client = httpx.AsyncClient()

    agent_count = len(config.agents)
    local_count = len(config.get_local_agents())
    remote_count = len(config.get_remote_agents())
    logger.info(
        f"Rookery loaded: {agent_count} agents "
        f"({local_count} local, {remote_count} remote)"
    )
    for name, agent_cfg in config.agents.items():
        if isinstance(agent_cfg, LocalAgentConfig):
            logger.info(f"  Agent '{name}': localhost:{agent_cfg.port}")
        elif isinstance(agent_cfg, RemoteAgentConfig):
            logger.info(f"  Agent '{name}': {agent_cfg.url}")

    yield

    logger.info("Kestrel Host shutting down...")
    await app.state.http_client.aclose()


app = FastAPI(title="Kestrel Host", lifespan=lifespan)

# Mount static files
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    app.mount("/js", StaticFiles(directory=str(STATIC_DIR / "js")), name="js")
    app.mount("/shared", StaticFiles(directory=str(STATIC_DIR / "shared")), name="shared")
    app.mount("/utils", StaticFiles(directory=str(STATIC_DIR / "utils")), name="utils")


# --- Auth Middleware ---

@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Global authentication middleware (same pattern as server.py)."""
    public_paths = {"/health", "/", "/favicon.ico", "/api/auth/key"}
    static_prefixes = ("/static", "/js/", "/shared/", "/utils/")

    if request.url.path in public_paths or any(
        request.url.path.startswith(p) for p in static_prefixes
    ):
        return await call_next(request)

    expected_key = get_api_key()

    # Check X-API-Key header
    header_key = request.headers.get(API_KEY_NAME)
    if header_key and header_key == expected_key:
        return await call_next(request)

    # Check Bearer token
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header[7:]
        if token == expected_key:
            return await call_next(request)

    # Check query parameter (for SSE endpoints)
    api_key_query = request.query_params.get("api_key")
    if api_key_query and api_key_query == expected_key:
        return await call_next(request)

    return JSONResponse(
        content={"detail": "Invalid or missing API Key"},
        status_code=401,
    )


# --- Routes ---


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve the main web UI."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return HTMLResponse(content=index_path.read_text(), status_code=200)


@app.get("/api/auth/key")
async def get_bootstrap_key(request: Request):
    """Return API key for initial frontend setup (localhost only)."""
    client_host = request.client.host if request.client else None
    allowed_hosts = {"127.0.0.1", "localhost", "::1", "172.17.0.1"}
    is_docker_internal = client_host and client_host.startswith("172.")

    if client_host not in allowed_hosts and not is_docker_internal:
        raise HTTPException(
            status_code=403,
            detail="API key bootstrap only accessible from localhost",
        )

    return {
        "key": get_api_key(),
        "header": API_KEY_NAME,
        "usage": "Include as 'X-API-Key' header or 'Authorization: Bearer <key>'",
    }


@app.get("/health")
async def health_check(request: Request):
    """Host health check with agent statuses."""
    config: RookeryConfig = request.app.state.rookery_config
    client: httpx.AsyncClient = request.app.state.http_client

    agent_statuses = {}
    for name, agent_cfg in config.agents.items():
        base_url = get_agent_base_url(agent_cfg)
        try:
            resp = await client.get(
                f"{base_url}/health",
                timeout=httpx.Timeout(connect=2.0, read=5.0, write=5.0, pool=2.0),
            )
            if resp.status_code == 200:
                agent_statuses[name] = {"status": "online", "url": base_url}
            else:
                agent_statuses[name] = {
                    "status": "unhealthy",
                    "url": base_url,
                    "http_status": resp.status_code,
                }
        except (httpx.ConnectError, httpx.TimeoutException):
            agent_statuses[name] = {"status": "offline", "url": base_url}

    return {
        "status": "ok",
        "role": "host",
        "agents": agent_statuses,
    }


@app.get("/api/agents")
async def list_agents(request: Request):
    """Aggregate A2A cards from all registered agents.

    For each agent:
    1. Ping /health to check status
    2. If online, fetch A2A card from /api/agents
    3. Return aggregated list with online/offline status
    """
    config: RookeryConfig = request.app.state.rookery_config
    client: httpx.AsyncClient = request.app.state.http_client

    agents = []
    for name, agent_cfg in config.agents.items():
        base_url = get_agent_base_url(agent_cfg)
        agent_entry = {
            "name": name,
            "url": base_url,
            "status": "offline",
            "type": "local" if isinstance(agent_cfg, LocalAgentConfig) else "remote",
        }

        try:
            # Check health first
            health_resp = await client.get(
                f"{base_url}/health",
                timeout=httpx.Timeout(connect=2.0, read=5.0, write=5.0, pool=2.0),
            )
            if health_resp.status_code != 200:
                agents.append(agent_entry)
                continue

            agent_entry["status"] = "online"

            # Fetch A2A card
            card_resp = await client.get(
                f"{base_url}/api/agents",
                timeout=httpx.Timeout(connect=2.0, read=5.0, write=5.0, pool=2.0),
                headers=build_auth_headers(request),
            )
            if card_resp.status_code == 200:
                card_data = card_resp.json()
                # The agent's /api/agents returns {"agents": [...]}
                agent_cards = card_data.get("agents", [])
                if agent_cards:
                    # Merge the first card's data into our entry
                    card = agent_cards[0]
                    agent_entry["card"] = card
                    # Use DID from card if available
                    if "id" in card:
                        agent_entry["id"] = card["id"]

        except (httpx.ConnectError, httpx.TimeoutException):
            pass  # Already set to offline

        agents.append(agent_entry)

    return {"agents": agents}


@app.api_route(
    "/api/agents/{agent_id}/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
)
async def proxy_to_agent(request: Request, agent_id: str, path: str):
    """Proxy requests to the correct agent by alias.

    Route pattern: /api/agents/{agent_id}/conversations → agent:{port}/api/conversations
    """
    config: RookeryConfig = request.app.state.rookery_config
    client: httpx.AsyncClient = request.app.state.http_client

    # Rewrite path: the agent expects /api/... paths
    agent_path = f"api/{path}" if not path.startswith("api/") else path

    return await proxy_request_streaming(
        request=request,
        agent_id=agent_id,
        path=agent_path,
        config=config,
        client=client,
    )


def build_auth_headers(request: Request) -> dict[str, str]:
    """Extract auth headers from the incoming request for forwarding."""
    headers = {}
    api_key = request.headers.get(API_KEY_NAME)
    if api_key:
        headers[API_KEY_NAME] = api_key
    auth = request.headers.get("Authorization")
    if auth:
        headers["Authorization"] = auth
    return headers


if __name__ == "__main__":
    import uvicorn

    config = load_rookery_config()
    port = config.host.port
    bind = config.host.bind
    logger.info(f"Starting Kestrel Host on {bind}:{port}")
    uvicorn.run(app, host=bind, port=port)
