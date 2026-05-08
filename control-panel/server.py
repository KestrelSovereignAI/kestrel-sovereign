#!/usr/bin/env python3
"""
Kestrel Control Panel - Multi-agent orchestrator for power users.

Discovers agents in agent_data/, manages their lifecycle, provides dashboard.
"""

import json
import logging
import os
import secrets
import shutil
import signal
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
KESTREL_ROOT = Path(__file__).parent.parent
AGENT_DATA_DIR = KESTREL_ROOT / "agent_data"
BASE_PORT = 8900  # Control panel agents start here
CONTROL_PANEL_PORT = 8899

app = FastAPI(title="Kestrel Control Panel")

# Track running agents: {agent_id: {"port": int, "pid": int}}
running_agents: dict[str, dict] = {}


# --- Authentication ---
API_KEY_NAME = "X-API-Key"


def get_api_key() -> str:
    """Get the API key from environment, falling back to KESTREL_API_KEY.

    Checks KESTREL_CONTROL_API_KEY first (control-panel-specific),
    then KESTREL_API_KEY (shared with main server).
    If neither is set, generates a temporary key and logs a warning.
    """
    api_key = os.environ.get("KESTREL_CONTROL_API_KEY") or os.environ.get("KESTREL_API_KEY")
    if not api_key:
        generated_key = secrets.token_urlsafe(32)
        os.environ["KESTREL_CONTROL_API_KEY"] = generated_key
        logger.warning("No KESTREL_CONTROL_API_KEY or KESTREL_API_KEY set. A temporary key has been generated.")
        logger.warning("Set KESTREL_CONTROL_API_KEY in your environment for persistence.")
        return generated_key
    # Strip surrounding quotes (Docker --env-file includes them literally)
    if len(api_key) >= 2 and api_key[0] == api_key[-1] and api_key[0] in ('"', "'"):
        api_key = api_key[1:-1]
    return api_key


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Authenticate all /api/ requests via API key.

    Accepts:
      - X-API-Key header
      - Authorization: Bearer <key>

    Public paths (no auth required):
      - / (dashboard HTML)
      - /health
    """
    # Public paths — no auth required
    if request.url.path in ("/", "/health"):
        return await call_next(request)

    # Only protect /api/ endpoints
    if not request.url.path.startswith("/api/"):
        return await call_next(request)

    expected_key = get_api_key()

    # Check X-API-Key header
    header_key = request.headers.get(API_KEY_NAME)
    if header_key and secrets.compare_digest(header_key, expected_key):
        return await call_next(request)

    # Check Authorization: Bearer <key>
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header[7:]
        if secrets.compare_digest(token, expected_key):
            return await call_next(request)

    return JSONResponse(
        content={"detail": "Invalid or missing API Key"},
        status_code=401,
    )


class AgentInfo(BaseModel):
    id: str
    name: str
    did: str
    path: str
    is_test: bool
    test_cycle: Optional[str]
    port: Optional[int] = None
    pid: Optional[int] = None
    running: bool = False


def read_agent_from_db(db_path: Path, agent_id: str, agent_path: Path) -> Optional[AgentInfo]:
    """Read agent info from a kestrel_prime.db file."""
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Get agent identity from graph_nodes table
        cursor.execute("""
            SELECT node_id, label, properties FROM graph_nodes 
            WHERE node_type = 'agent' LIMIT 1
        """)
        row = cursor.fetchone()
        conn.close()
        
        if row:
            did, label, properties_json = row
            metadata = json.loads(properties_json) if properties_json else {}
            name = label  # label is the agent name
            
            is_test = metadata.get("is_test_instance", False)
            test_cycle = metadata.get("test_cycle_id")
            
            # Check if running
            running = agent_id in running_agents
            port = running_agents.get(agent_id, {}).get("port")
            pid = running_agents.get(agent_id, {}).get("pid")
            
            return AgentInfo(
                id=agent_id,
                name=name or agent_id,
                did=did,
                path=str(agent_path),
                is_test=is_test,
                test_cycle=test_cycle,
                port=port,
                pid=pid,
                running=running
            )
    except Exception as e:
        print(f"Error reading {db_path}: {e}")
    
    return None


def discover_agents() -> list[AgentInfo]:
    """Scan agent_data/ for valid agents."""
    agents = []
    
    if not AGENT_DATA_DIR.exists():
        return agents
    
    # Check for "default" agent (flat in agent_data/)
    default_db = AGENT_DATA_DIR / "kestrel_prime.db"
    if default_db.exists():
        agent = read_agent_from_db(default_db, "default", AGENT_DATA_DIR)
        if agent:
            agents.append(agent)
    
    # Check subdirectories for additional agents
    for entry in AGENT_DATA_DIR.iterdir():
        if not entry.is_dir():
            continue
        
        # Skip known non-agent directories
        if entry.name in ("trusted_agents", "__pycache__"):
            continue
        
        db_path = entry / "kestrel_prime.db"
        if not db_path.exists():
            continue
        
        agent = read_agent_from_db(db_path, entry.name, entry)
        if agent:
            agents.append(agent)
    
    return agents


def get_next_port() -> int:
    """Get next available port for an agent."""
    used_ports = {info["port"] for info in running_agents.values()}
    port = BASE_PORT
    while port in used_ports:
        port += 1
    return port


@app.get("/")
async def index():
    """Serve the dashboard."""
    return FileResponse(Path(__file__).parent / "index.html")


@app.get("/api/agents")
async def list_agents():
    """List all discovered agents."""
    return discover_agents()


@app.post("/api/agents/{agent_id}/start")
async def start_agent(agent_id: str):
    """Start an agent on an available port."""
    if agent_id in running_agents:
        return {"status": "already_running", **running_agents[agent_id]}
    
    # Find the agent
    agents = discover_agents()
    agent = next((a for a in agents if a.id == agent_id), None)
    
    if not agent:
        raise HTTPException(404, f"Agent {agent_id} not found")
    
    port = get_next_port()
    log_file = KESTREL_ROOT / "control-panel" / f"{agent_id}.log"
    
    # Create agent-specific .env file
    agent_env_file = KESTREL_ROOT / "control-panel" / f".env.{agent_id}"
    base_env_file = KESTREL_ROOT / ".env"
    
    # Copy base .env and override KESTREL_DB_PATH
    env_content = base_env_file.read_text() if base_env_file.exists() else ""
    
    # Replace or add KESTREL_DB_PATH
    lines = env_content.split("\n")
    new_lines = []
    found = False
    for line in lines:
        if line.strip().startswith("KESTREL_DB_PATH="):
            new_lines.append(f"KESTREL_DB_PATH={agent.path}")
            found = True
        elif line.strip().startswith("# KESTREL_DB_PATH"):
            # Skip commented lines about db path
            continue
        else:
            new_lines.append(line)
    if not found:
        new_lines.insert(0, f"KESTREL_DB_PATH={agent.path}")
    
    agent_env_file.write_text("\n".join(new_lines))
    
    # Create isolated run directory with symlinks to Kestrel + our .env
    run_dir = KESTREL_ROOT / "control-panel" / f"run_{agent_id}"
    run_dir.mkdir(exist_ok=True)
    
    # Symlink everything from KESTREL_ROOT except .env
    for item in KESTREL_ROOT.iterdir():
        if item.name in (".env", "control-panel", ".git", "__pycache__", ".venv"):
            continue
        link = run_dir / item.name
        if not link.exists():
            link.symlink_to(item)
    
    # Copy our agent-specific .env
    (run_dir / ".env").unlink(missing_ok=True)
    shutil.copy(agent_env_file, run_dir / ".env")
    
    # Start from the isolated run directory
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "kestrel_sovereign.server:app", "--host", "0.0.0.0", "--port", str(port)],
        cwd=str(run_dir),
        stdout=open(log_file, "w"),
        stderr=subprocess.STDOUT,
        start_new_session=True
    )
    
    running_agents[agent_id] = {
        "port": port,
        "pid": process.pid,
        "log": str(log_file),
        "run_dir": str(run_dir)
    }
    
    return {
        "status": "started",
        "port": port,
        "pid": process.pid,
        "url": f"http://localhost:{port}"
    }


@app.post("/api/agents/{agent_id}/stop")
async def stop_agent(agent_id: str):
    """Stop a running agent."""
    if agent_id not in running_agents:
        raise HTTPException(400, f"Agent {agent_id} is not running")
    
    info = running_agents[agent_id]
    
    try:
        os.killpg(os.getpgid(info["pid"]), signal.SIGTERM)
    except ProcessLookupError:
        pass  # Already dead
    
    del running_agents[agent_id]
    
    return {"status": "stopped", "agent_id": agent_id}


@app.get("/api/agents/{agent_id}/logs")
async def get_logs(agent_id: str, lines: int = 100):
    """Get recent logs for an agent."""
    log_file = KESTREL_ROOT / "control-panel" / f"{agent_id}.log"
    
    if not log_file.exists():
        return {"logs": ""}
    
    with open(log_file) as f:
        all_lines = f.readlines()
        return {"logs": "".join(all_lines[-lines:])}


@app.get("/health")
async def health_check():
    """Public health check endpoint."""
    return {"status": "ok", "service": "kestrel-control-panel"}


if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("CONTROL_PANEL_HOST", "127.0.0.1")
    print(f"Kestrel Control Panel starting on http://{host}:{CONTROL_PANEL_PORT}")
    print(f"   Scanning agents in: {AGENT_DATA_DIR}")
    uvicorn.run(app, host=host, port=CONTROL_PANEL_PORT)
