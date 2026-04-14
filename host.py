#!/usr/bin/env python3
"""
Kestrel Host - Thin FastAPI proxy + static file server + process manager.

The host is NOT an agent. It has no DID, no memory, no LLM.
It is pure infrastructure that:
1. Serves the Kestrel UI (static files)
2. Aggregates agent discovery (A2A cards)
3. Proxies API requests to the correct agent process
4. Manages agent process lifecycle (start/stop/status/logs)

Architecture:
    Browser → localhost:8888 (host.py)
                  ├── /static/*                     → serves UI files directly
                  ├── /api/agents                   → aggregates A2A cards from all agents
                  ├── /api/agents/{id}/start        → start an agent process
                  ├── /api/agents/{id}/stop         → stop an agent process
                  ├── /api/agents/{id}/status       → agent process status
                  ├── /api/agents/{id}/logs         → tail agent logs
                  └── /api/agents/{id}/*            → proxies to agent on its port

Key principle: No agent is privileged. The host treats all agents as equal peers.
"""

import ipaddress
import os
import secrets
import logging
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
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
from kestrel_sovereign.rookery.process_manager import ProcessManager

load_dotenv(Path(__file__).parent / ".env", override=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security Configuration
API_KEY_NAME = "X-API-Key"

# Paths (suffixes) where API key query parameter auth is allowed.
# EventSource/SSE connections cannot send custom headers, so these endpoints
# need query param auth. Restricted to avoid leaking keys in URL logs.
# The host proxies SSE requests under /api/agents/{id}/..., so we match
# path suffixes rather than exact paths.
SSE_PATH_SUFFIXES = ("/agent/notifications/sse", "/agent/stream")

# Project directory (where host.py lives)
PROJECT_DIR = Path(__file__).parent.resolve()


def get_api_key() -> str:
    """Get or generate the host API key, persisting it to .env if newly generated."""
    api_key = os.environ.get("KESTREL_HOST_API_KEY") or os.environ.get("KESTREL_API_KEY")
    if not api_key:
        generated_key = secrets.token_urlsafe(32)
        os.environ["KESTREL_API_KEY"] = generated_key
        # Persist to .env so it survives restarts and won't cause browser 401s
        env_path = PROJECT_DIR / ".env"
        try:
            with open(env_path, "a", encoding="utf-8") as f:
                f.write(f"\nKESTREL_API_KEY={generated_key}\n")
            logger.info(f"KESTREL_API_KEY generated and saved to .env")
        except OSError as e:
            logger.warning(f"Could not persist KESTREL_API_KEY to .env: {e}")
            logger.warning(f"Add this to your .env manually: KESTREL_API_KEY={generated_key}")
        return generated_key
    # Strip surrounding quotes (Docker --env-file includes them literally)
    if len(api_key) >= 2 and api_key[0] == api_key[-1] and api_key[0] in ('"', "'"):
        api_key = api_key[1:-1]
    return api_key


def load_rookery_config() -> RookeryConfig:
    """Load rookery configuration from file or auto-discover.

    When running on Cloud Run or Azure Container Apps, the platform injects
    a PORT env var. Override the host port to match so the container binds
    to the correct port.
    """
    config_path = os.environ.get("KESTREL_ROOKERY_CONFIG")
    if config_path:
        config = RookeryConfig.load(config_path)
    else:
        config = RookeryConfig.load()

    # Cloud Run / Azure Container Apps override: bind to platform-assigned port
    cloud_port = os.environ.get("PORT")
    if cloud_port:
        config.host.port = int(cloud_port)

    return config


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the host application lifecycle.

    On startup: load config, create ProcessManager, register agents,
    optionally autostart agents.
    On shutdown: stop all managed agents, close HTTP client.
    """
    logger.info("Kestrel Host starting up...")

    config = load_rookery_config()
    app.state.rookery_config = config
    app.state.http_client = httpx.AsyncClient()

    # Create process manager
    pm = ProcessManager(PROJECT_DIR)
    app.state.process_manager = pm

    agent_count = len(config.agents)
    local_count = len(config.get_local_agents())
    remote_count = len(config.get_remote_agents())
    logger.info(
        f"Rookery loaded: {agent_count} agents "
        f"({local_count} local, {remote_count} remote)"
    )

    # Register all local agents with the process manager
    for name, agent_cfg in config.agents.items():
        if isinstance(agent_cfg, LocalAgentConfig):
            pm.register_agent(name, agent_cfg)
            logger.info(f"  Agent '{name}': localhost:{agent_cfg.port}")
        elif isinstance(agent_cfg, RemoteAgentConfig):
            logger.info(f"  Agent '{name}': {agent_cfg.url}")

    # Autostart agents if KESTREL_HOST_AUTOSTART is set (or not explicitly disabled)
    autostart_disabled = os.environ.get("KESTREL_HOST_AUTOSTART", "").lower() in ("0", "false", "no")
    if not autostart_disabled:
        autostart_agents = config.get_autostart_agents()
        if autostart_agents:
            logger.info(f"Autostarting {len(autostart_agents)} agents...")
            started = pm.start_autostart_agents(config)
            for name, ap in started.items():
                if pm.wait_for_health(ap.port, timeout=30):
                    logger.info(f"  Agent '{name}' started on :{ap.port}")
                else:
                    logger.warning(f"  Agent '{name}' failed health check on :{ap.port}")

    yield

    logger.info("Kestrel Host shutting down...")
    pm.stop_all()
    await app.state.http_client.aclose()


app = FastAPI(title="Kestrel Host", lifespan=lifespan)

# Mount static files
STATIC_DIR = Path(__file__).parent / "kestrel_sovereign" / "static"
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
    static_prefixes = ("/static", "/js/", "/shared/", "/utils/", "/api/github/")

    if request.url.path in public_paths or any(
        request.url.path.startswith(p) for p in static_prefixes
    ):
        return await call_next(request)
    # Webhooks authenticate themselves (HMAC, bearer, etc.) — bypass API key auth
    if request.url.path.startswith("/webhooks/"):
        return await call_next(request)
    if request.method == "OPTIONS":
        return await call_next(request)

    expected_key = get_api_key()

    # Check X-API-Key header
    header_key = request.headers.get(API_KEY_NAME)
    if header_key and secrets.compare_digest(header_key, expected_key):
        return await call_next(request)

    # Check Bearer token
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header[7:]
        if secrets.compare_digest(token, expected_key):
            return await call_next(request)

    # Check query parameter for SSE endpoints only (EventSource can't send headers)
    # Restricted to SSE paths to avoid leaking keys in URL logs on other endpoints
    api_key_query = request.query_params.get("api_key")
    if api_key_query and any(request.url.path.endswith(s) for s in SSE_PATH_SUFFIXES):
        if secrets.compare_digest(api_key_query, expected_key):
            return await call_next(request)

    return JSONResponse(
        content={"detail": "Invalid or missing API Key"},
        status_code=401,
    )


# CORS middleware — added after auth so it runs outermost (before auth).
# Uses same defaults as server.py. Override via KESTREL_CORS_ORIGINS env var.
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


# --- Routes ---


def _is_docker_network(host: str) -> bool:
    """Check if the host IP is within Docker's internal network range (172.16.0.0/12)."""
    try:
        return ipaddress.ip_address(host) in ipaddress.ip_network("172.16.0.0/12")
    except ValueError:
        return False


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
    is_docker_internal = client_host and _is_docker_network(client_host)

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


# --- GitHub API Proxy (for dashboard) ---

@app.get("/api/github/{path:path}")
async def github_proxy(path: str, request: Request):
    """Proxy GitHub API requests using server-side token."""
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        # Try .env file
        env_path = Path(__file__).parent / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("GITHUB_TOKEN="):
                    token = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not token:
        return JSONResponse({"error": "No GITHUB_TOKEN configured on server"}, status_code=503)

    gh_url = f"https://api.github.com/{path}"
    if request.url.query:
        gh_url += f"?{request.url.query}"

    client: httpx.AsyncClient = request.app.state.http_client
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
        return JSONResponse(
            content=resp.json(),
            status_code=resp.status_code,
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=502)


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


# --- Process Management Endpoints ---


@app.post("/api/agents/{agent_id}/start")
async def start_agent(request: Request, agent_id: str):
    """Start an agent process.

    Only works for local agents configured in rookery.toml.
    """
    config: RookeryConfig = request.app.state.rookery_config
    pm: ProcessManager = request.app.state.process_manager

    local_agents = config.get_local_agents()
    if agent_id not in local_agents:
        raise HTTPException(
            status_code=404,
            detail=f"Agent '{agent_id}' not found or not a local agent",
        )

    agent_cfg = local_agents[agent_id]
    try:
        ap = pm.start_agent(agent_id, agent_cfg, config.host.bind, config.host.port)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))

    # Wait briefly for the agent to become healthy
    healthy = pm.wait_for_health(ap.port, timeout=15)

    return {
        "agent_id": agent_id,
        "port": ap.port,
        "pid": ap.pid,
        "status": "running" if healthy else "starting",
        "healthy": healthy,
    }


@app.post("/api/agents/{agent_id}/stop")
async def stop_agent(request: Request, agent_id: str):
    """Stop an agent process."""
    config: RookeryConfig = request.app.state.rookery_config
    pm: ProcessManager = request.app.state.process_manager

    local_agents = config.get_local_agents()
    if agent_id not in local_agents:
        raise HTTPException(
            status_code=404,
            detail=f"Agent '{agent_id}' not found or not a local agent",
        )

    pm.stop_agent(agent_id)
    return {"agent_id": agent_id, "status": "stopped"}


@app.get("/api/agents/{agent_id}/status")
async def agent_process_status(request: Request, agent_id: str):
    """Get process status for an agent."""
    config: RookeryConfig = request.app.state.rookery_config
    pm: ProcessManager = request.app.state.process_manager

    local_agents = config.get_local_agents()
    if agent_id not in local_agents:
        raise HTTPException(
            status_code=404,
            detail=f"Agent '{agent_id}' not found or not a local agent",
        )

    return pm.get_agent_status(agent_id)


@app.get("/api/agents/{agent_id}/logs")
async def agent_logs(request: Request, agent_id: str, lines: int = 50):
    """Get recent log output for an agent.

    Query params:
        lines: Number of lines to return (default 50, max 1000)
    """
    config: RookeryConfig = request.app.state.rookery_config
    pm: ProcessManager = request.app.state.process_manager

    local_agents = config.get_local_agents()
    if agent_id not in local_agents:
        raise HTTPException(
            status_code=404,
            detail=f"Agent '{agent_id}' not found or not a local agent",
        )

    lines = min(lines, 1000)
    log_text = pm.read_logs(agent_id, lines=lines)
    if log_text is None:
        raise HTTPException(
            status_code=404,
            detail=f"No log file found for agent '{agent_id}'",
        )

    return PlainTextResponse(content=log_text)


# --- Rasa Webhook Proxy (for RemoteCares RCS integration) ---
# RCS calls /webhooks/rest/webhook directly; forward to the first agent.


@app.post("/webhooks/rest/webhook")
async def rasa_webhook_proxy(request: Request):
    """Forward Rasa webhook requests to the first configured agent."""
    config: RookeryConfig = request.app.state.rookery_config
    client: httpx.AsyncClient = request.app.state.http_client
    first_agent = next(iter(config.agents))
    return await proxy_request_streaming(
        request=request,
        agent_id=first_agent,
        path="webhooks/rest/webhook",
        config=config,
        client=client,
    )


# --- GitHub App Webhook Proxy ---
# Forward GitHub App webhooks to the first agent (which has GitHubAppFeature)

@app.post("/webhooks/github-app")
async def github_app_webhook_proxy(request: Request):
    """Forward GitHub App webhook to the first configured agent."""
    import sys, json as _json
    # Cloud Run captures stdout as structured logs
    print(_json.dumps({"severity": "WARNING", "message": "HOST: /webhooks/github-app received"}), flush=True)
    config: RookeryConfig = request.app.state.rookery_config
    client: httpx.AsyncClient = request.app.state.http_client
    first_agent = next(iter(config.agents))
    print(_json.dumps({"severity": "WARNING", "message": f"HOST: proxying to agent={first_agent}"}), flush=True)
    try:
        result = await proxy_request_streaming(
            request=request,
            agent_id=first_agent,
            path="webhooks/github-app",
            config=config,
            client=client,
        )
        print(_json.dumps({"severity": "WARNING", "message": f"HOST: proxy returned status={result.status_code}"}), flush=True)
        return result
    except Exception as e:
        print(_json.dumps({"severity": "ERROR", "message": f"HOST: proxy ERROR: {e}"}), flush=True)
        raise


# --- Proxy Route (must be AFTER specific routes to avoid path conflicts) ---


@app.api_route(
    "/api/agents/{agent_id}/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
)
async def proxy_to_agent(request: Request, agent_id: str, path: str):
    """Proxy requests to the correct agent by alias.

    Path is forwarded as-is to the agent. The caller includes the full path:
        /api/agents/claw/api/conversations  → agent:8801/api/conversations
        /api/agents/claw/v1/chat/completions → agent:8801/v1/chat/completions
        /api/agents/claw/health             → agent:8801/health
    """
    config: RookeryConfig = request.app.state.rookery_config
    client: httpx.AsyncClient = request.app.state.http_client

    return await proxy_request_streaming(
        request=request,
        agent_id=agent_id,
        path=path,
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
