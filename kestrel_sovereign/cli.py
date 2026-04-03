"""
Unified Kestrel CLI for host and agent management.

This is the single entry point for managing the Kestrel Host and all agents.
It replaces start_kestrel.sh / stop_kestrel.sh and subsumes main.py's
interactive chat into `kestrel shell <name>`.

Commands:
    kestrel start                  # start all agents in-process (default)
    kestrel start --subprocess     # start host + agents as separate processes
    kestrel start <name>           # start just one agent (subprocess)
    kestrel stop                   # stop everything (agents first, then host)
    kestrel stop <name>            # stop just one agent
    kestrel status                 # table: host + all agents with ports, PIDs, status
    kestrel logs <name>            # tail agent logs (or "host" for host logs)
    kestrel list                   # list rookery agents, ports, data dirs
    kestrel create <name>          # inception: generate DID, create agent folder, add to rookery.toml
    kestrel shell <name>           # interactive CLI chat (what main.py does today)
    kestrel health                 # run health check
    kestrel config <agent_dir>     # show/edit agent config
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from kestrel_sovereign import __version__
from kestrel_sovereign.rookery.config import (
    RookeryConfig,
    LocalAgentConfig,
    ROOKERY_CONFIG_FILENAME,
    DEFAULT_AGENT_START_PORT,
)
from kestrel_sovereign.rookery.process_manager import ProcessManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_project_dir() -> Path:
    """Get the project root directory (where server.py lives)."""
    return Path(__file__).parent.parent.resolve()


def _format_uptime(pid: int) -> str:
    """Get uptime string for a running process."""
    try:
        import psutil
        p = psutil.Process(pid)
        elapsed = time.time() - p.create_time()
        hours, remainder = divmod(int(elapsed), 3600)
        minutes, _ = divmod(remainder, 60)
        if hours > 0:
            return f"{hours}h {minutes}m"
        return f"{minutes}m"
    except Exception:
        return "-"


# ---------------------------------------------------------------------------
# Host PID / log file paths
# ---------------------------------------------------------------------------

def _host_pid_file(project_dir: Optional[Path] = None) -> Path:
    """PID file for the host process."""
    if project_dir is None:
        project_dir = _get_project_dir()
    logs_dir = project_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir / ".host.pid"


def _host_log_file(project_dir: Optional[Path] = None) -> Path:
    """Log file for the host process."""
    if project_dir is None:
        project_dir = _get_project_dir()
    logs_dir = project_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir / "host.log"


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def cmd_start(args) -> int:
    """Start host and/or agents."""
    project_dir = _get_project_dir()
    rookery = RookeryConfig.load(project_dir / ROOKERY_CONFIG_FILENAME)
    pm = ProcessManager(project_dir)

    if args.name:
        # Start a single agent by name
        local_agents = rookery.get_local_agents()
        if args.name not in local_agents:
            print(f"Agent '{args.name}' not found in rookery config")
            print(f"Available agents: {', '.join(local_agents.keys()) or '(none)'}")
            return 1

        agent_cfg = local_agents[args.name]
        print(f"   Starting {args.name} on :{agent_cfg.port}...", end="", flush=True)
        try:
            pm.start_agent(args.name, agent_cfg, rookery.host.bind, standalone=True)
        except RuntimeError as e:
            print(f"          \u274c")
            print(f"   {e}")
            return 1

        if pm.wait_for_health(agent_cfg.port, timeout=30):
            print("          \u2705")
        else:
            print("          \u274c")
            return 1
        return 0

    if getattr(args, "subprocess", False):
        return _start_subprocess_mode(project_dir, rookery, pm)
    return _start_inprocess_mode(project_dir, rookery, pm)


def _start_inprocess_mode(project_dir: Path, rookery, pm: ProcessManager) -> int:
    """Start all agents in a single server process (default mode)."""
    autostart = rookery.get_autostart_agents()
    manual = {
        name: cfg for name, cfg in rookery.get_local_agents().items()
        if not cfg.autostart
    }

    print("\U0001F985 Kestrel Rookery starting (in-process)...")
    print(f"   URL:      http://localhost:{rookery.host.port}")

    if autostart or manual:
        print("   Agents:")
        for name, cfg in autostart.items():
            resolved = (project_dir / cfg.data_dir).resolve()
            print(f"     {name:12} {resolved}/       autostart")
        for name, cfg in manual.items():
            resolved = (project_dir / cfg.data_dir).resolve()
            print(f"     {name:12} {resolved}/       manual")
    print()

    host_pid_file = _host_pid_file(project_dir)
    existing_host = pm.read_pid(host_pid_file)
    if existing_host and pm.is_process_running(existing_host):
        print(f"   Server already running (PID: {existing_host})")
        return 0

    if existing_host:
        pm.clear_pid(host_pid_file)

    if pm.is_port_in_use(rookery.host.port):
        print(f"   Port {rookery.host.port} already in use")
        return 1

    env = pm._load_env()
    env["PORT"] = str(rookery.host.port)
    env["KESTREL_MULTI_AGENT"] = "true"
    env["KESTREL_SERVE_UI"] = "true"

    log_file = _host_log_file(project_dir)
    cmd = [sys.executable, "-m", "uvicorn", "server:app",
           "--host", rookery.host.bind, "--port", str(rookery.host.port)]

    print(f"   Starting server on :{rookery.host.port}...", end="", flush=True)
    pm._spawn(cmd, env, log_file, host_pid_file)

    if pm.wait_for_health(rookery.host.port, timeout=30):
        print("          \u2705")
    else:
        print("          \u274c")
        print(f"   Check log: {log_file}")
        return 1

    print(f"\n\U0001F985 Rookery ready: http://localhost:{rookery.host.port}")
    return 0


def _start_subprocess_mode(project_dir: Path, rookery, pm: ProcessManager) -> int:
    """Start host + separate agent processes (legacy --subprocess mode)."""
    # Start the full rookery (host + autostart agents)
    print("\U0001F985 Kestrel Rookery starting (subprocess)...")
    print(f"   Host:     http://localhost:{rookery.host.port}")

    autostart = rookery.get_autostart_agents()
    manual = {
        name: cfg for name, cfg in rookery.get_local_agents().items()
        if not cfg.autostart
    }

    if autostart or manual:
        print("   Agents:")
        for name, cfg in autostart.items():
            resolved = (project_dir / cfg.data_dir).resolve()
            print(f"     {name:12} port {cfg.port}  {resolved}/       autostart")
        for name, cfg in manual.items():
            resolved = (project_dir / cfg.data_dir).resolve()
            print(f"     {name:12} port {cfg.port}  {resolved}/       manual (skipped)")
    print()

    # Start host
    host_pid_file = _host_pid_file(project_dir)
    existing_host = pm.read_pid(host_pid_file)
    if existing_host and pm.is_process_running(existing_host):
        print(f"   Host already running (PID: {existing_host})")
    else:
        if existing_host:
            pm.clear_pid(host_pid_file)

        if pm.is_port_in_use(rookery.host.port):
            print(f"   Host port {rookery.host.port} already in use")
            return 1

        env = pm._load_env()
        env["PORT"] = str(rookery.host.port)
        # Host is NOT an agent — no DB path, no KESTREL_SERVE_UI

        log_file = _host_log_file(project_dir)
        cmd = [sys.executable, "-m", "uvicorn", "host:app",
               "--host", rookery.host.bind, "--port", str(rookery.host.port)]

        print(f"   Starting host on :{rookery.host.port}...", end="", flush=True)
        pm._spawn(cmd, env, log_file, host_pid_file)

        if pm.wait_for_health(rookery.host.port, timeout=30):
            print("          \u2705")
        else:
            print("          \u274c")
            print(f"   Check log: {log_file}")
            return 1

    # Start autostart agents
    for name, cfg in autostart.items():
        print(f"   Starting {name} on :{cfg.port}...", end="", flush=True)
        try:
            pm.start_agent(name, cfg, rookery.host.bind)
        except RuntimeError as e:
            print(f"          \u274c")
            print(f"   {e}")
            continue

        if pm.wait_for_health(cfg.port, timeout=30):
            print("          \u2705")
        else:
            print("          \u274c")

    print(f"\n\U0001F985 Rookery ready: http://localhost:{rookery.host.port}")
    return 0


def cmd_stop(args) -> int:
    """Stop host and/or agents."""
    project_dir = _get_project_dir()
    rookery = RookeryConfig.load(project_dir / ROOKERY_CONFIG_FILENAME)
    pm = ProcessManager(project_dir)

    if args.name:
        # Stop a single agent
        local_agents = rookery.get_local_agents()
        if args.name not in local_agents:
            print(f"Agent '{args.name}' not found in rookery config")
            return 1

        agent_cfg = local_agents[args.name]
        pm.register_agent(args.name, agent_cfg)
        ap = pm._agents.get(args.name)
        if ap and ap.pid:
            print(f"   Stopping {args.name} (PID: {ap.pid})...")
            pm.stop_agent(args.name)
            print(f"   {args.name} stopped")
        return 0

    # Stop everything: agents first, then host
    print("\U0001F6D1 Stopping Kestrel Rookery...")

    for name, cfg in rookery.get_local_agents().items():
        pm.register_agent(name, cfg)
        ap = pm._agents.get(name)
        if ap and ap.pid:
            print(f"   Stopping {name} (PID: {ap.pid})...")
            pm.stop_agent(name)
            print(f"   {name} stopped")

    # Stop host
    host_pid_file = _host_pid_file(project_dir)
    host_pid = pm.read_pid(host_pid_file)
    if host_pid and pm.is_process_running(host_pid):
        print(f"   Stopping host (PID: {host_pid})...")
        pm.kill_process(host_pid, force=False)
        for _ in range(10):
            if not pm.is_process_running(host_pid):
                break
            time.sleep(0.5)
        if pm.is_process_running(host_pid):
            pm.kill_process(host_pid, force=True)
            time.sleep(0.5)
        pm.clear_pid(host_pid_file)
        print("   host stopped")

    print("\u2705 Rookery stopped")
    return 0


def cmd_status(args) -> int:
    """Show status of host and all agents."""
    project_dir = _get_project_dir()
    rookery = RookeryConfig.load(project_dir / ROOKERY_CONFIG_FILENAME)

    # Host/server status
    host_pid = ProcessManager.read_pid(_host_pid_file(project_dir))
    host_running = host_pid is not None and ProcessManager.is_process_running(host_pid)
    host_pid_str = str(host_pid) if host_running else "-"
    host_uptime = _format_uptime(host_pid) if host_running else "-"

    # Detect mode: check if any agent has its own PID file (subprocess mode)
    local_agents = rookery.get_local_agents()
    any_agent_pid = any(
        ProcessManager.read_pid(
            ProcessManager.agent_pid_file((project_dir / cfg.data_dir).resolve())
        ) is not None
        for cfg in local_agents.values()
    )

    if any_agent_pid:
        # Subprocess mode: show per-agent PID status
        print(f"  {'NAME':12} {'PORT':>6}   {'STATUS':10} {'PID':>7}   {'UPTIME':>8}")
        host_status = "online" if host_running else "offline"
        print(f"  {'host':12} {rookery.host.port:>6}   {host_status:10} {host_pid_str:>7}   {host_uptime:>8}")

        for name, cfg in local_agents.items():
            resolved_dir = (project_dir / cfg.data_dir).resolve()
            pid = ProcessManager.read_pid(ProcessManager.agent_pid_file(resolved_dir))
            running = pid is not None and ProcessManager.is_process_running(pid)
            status_str = "online" if running else "offline"
            pid_str = str(pid) if running else "-"
            uptime = _format_uptime(pid) if running else "-"
            print(f"  {name:12} {cfg.port:>6}   {status_str:10} {pid_str:>7}   {uptime:>8}")
    else:
        # In-process mode: query server API for agent status
        print(f"  {'NAME':12} {'STATUS':10} {'PID':>7}   {'UPTIME':>8}")
        server_status = "online" if host_running else "offline"
        print(f"  {'server':12} {server_status:10} {host_pid_str:>7}   {host_uptime:>8}")

        if host_running:
            agents = _query_agents_api(rookery.host.port)
            if agents is not None:
                for agent_info in agents:
                    name = agent_info.get("name", "?")
                    print(f"  {name:12} {'in-process':10} {host_pid_str:>7}   {host_uptime:>8}")
            else:
                # API unavailable, show config-based list
                for name in local_agents:
                    print(f"  {name:12} {'in-process':10} {host_pid_str:>7}   {host_uptime:>8}")
        else:
            for name in local_agents:
                print(f"  {name:12} {'offline':10} {'-':>7}   {'-':>8}")

    return 0


def _query_agents_api(port: int):
    """Query the running server's /api/agents endpoint. Returns list or None."""
    try:
        import urllib.request
        import json
        url = f"http://localhost:{port}/api/agents"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
            return data.get("agents", [])
    except Exception:
        return None


def cmd_logs(args) -> int:
    """Tail agent or host logs."""
    project_dir = _get_project_dir()

    if args.name == "host" or args.name == "server":
        log_file = _host_log_file(project_dir)
    else:
        rookery = RookeryConfig.load(project_dir / ROOKERY_CONFIG_FILENAME)
        local_agents = rookery.get_local_agents()
        if args.name not in local_agents:
            print(f"Agent '{args.name}' not found in rookery config")
            print("Use 'host' for host/server logs")
            return 1

        agent_cfg = local_agents[args.name]
        resolved_dir = (project_dir / agent_cfg.data_dir).resolve()
        log_file = ProcessManager.agent_log_file(resolved_dir)

        # In-process mode: agent has no separate log, fall back to server log
        if not log_file.exists():
            log_file = _host_log_file(project_dir)

    if not log_file.exists():
        print(f"No log file found: {log_file}")
        return 1

    # Build tail command
    tail_args = ["tail"]
    tail_args.extend(["-n", str(args.lines)])
    if args.follow:
        tail_args.append("-f")
    tail_args.append(str(log_file))

    try:
        return subprocess.call(tail_args)
    except KeyboardInterrupt:
        return 0


def cmd_list(args) -> int:
    """List all agents in rookery."""
    project_dir = _get_project_dir()
    rookery = RookeryConfig.load(project_dir / ROOKERY_CONFIG_FILENAME)

    local_agents = rookery.get_local_agents()
    remote_agents = rookery.get_remote_agents()

    if not local_agents and not remote_agents:
        print("No agents configured")
        print("Create one: kestrel create <name>")
        return 0

    if local_agents:
        print(f"  {'NAME':12} {'PORT':>6}   {'DATA DIR':30} {'AUTOSTART'}")
        for name, cfg in local_agents.items():
            autostart_str = "autostart" if cfg.autostart else "manual"
            print(f"  {name:12} {cfg.port:>6}   {str(cfg.data_dir):30} {autostart_str}")

    if remote_agents:
        print(f"\n  {'NAME':12} {'URL'}")
        for name, cfg in remote_agents.items():
            print(f"  {name:12} {cfg.url}")

    return 0


def cmd_create(args) -> int:
    """Create a new agent via inception."""
    project_dir = _get_project_dir()
    name = args.name

    # Determine data directory
    agent_data_dir = project_dir / "agent_data" / name

    if agent_data_dir.exists() and (agent_data_dir / "kestrel_prime.db").exists():
        print(f"Agent '{name}' already exists at {agent_data_dir}")
        return 1

    print(f"\U0001F985 Creating new Kestrel agent: {name}")

    # Run inception service
    cmd = [
        sys.executable, "-m", "kestrel_sovereign.inception_service",
        "--output-dir", str(agent_data_dir),
        "--name", name,
    ]

    result = subprocess.run(cmd, cwd=project_dir)
    if result.returncode != 0:
        print("Inception failed")
        return 1

    # Determine next available port
    rookery_path = project_dir / ROOKERY_CONFIG_FILENAME
    rookery = RookeryConfig.load(rookery_path)
    port = args.port
    if port is None:
        used_ports = {rookery.host.port}
        for cfg in rookery.get_local_agents().values():
            used_ports.add(cfg.port)
        # Start from the max existing agent port + 1 or default, whichever is higher
        max_agent_port = max(
            (cfg.port for cfg in rookery.get_local_agents().values()),
            default=DEFAULT_AGENT_START_PORT - 1,
        )
        port = max(DEFAULT_AGENT_START_PORT, max_agent_port + 1)
        while port in used_ports:
            port += 1

    # Add agent to rookery config
    rookery.agents[name] = LocalAgentConfig(
        data_dir=Path("agent_data") / name,
        port=port,
        autostart=True,
    )

    # Save rookery config
    rookery.save(rookery_path)

    # Print the DID from the agent's database
    did_str = _get_agent_did(agent_data_dir)

    print(f"   DID: {did_str or '(unknown)'}")
    print(f"   Data dir: agent_data/{name}/")
    print(f"   Port: {port} (next available)")
    print(f"   Added to {ROOKERY_CONFIG_FILENAME}")
    print(f"\u2705 Agent created. Start with: kestrel start {name}")
    return 0


def _get_agent_did(agent_dir: Path) -> Optional[str]:
    """Read agent DID from the database without starting the full agent."""
    db_path = agent_dir / "kestrel_prime.db"
    if not db_path.exists():
        return None
    try:
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute(
            "SELECT node_id FROM nodes WHERE node_type = 'agent' LIMIT 1"
        )
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None
    except Exception:
        return None


def cmd_shell(args) -> int:
    """Interactive CLI chat (absorbs main.py)."""
    import asyncio

    project_dir = _get_project_dir()
    rookery = RookeryConfig.load(project_dir / ROOKERY_CONFIG_FILENAME)
    local_agents = rookery.get_local_agents()

    if args.name not in local_agents:
        print(f"Agent '{args.name}' not found in rookery config")
        print(f"Available agents: {', '.join(local_agents.keys()) or '(none)'}")
        return 1

    agent_cfg = local_agents[args.name]
    agent_dir = (project_dir / agent_cfg.data_dir).resolve()

    if not (agent_dir / "kestrel_prime.db").exists():
        print(f"Agent database not found at {agent_dir}")
        print(f"Create the agent first: kestrel create {args.name}")
        return 1

    # Run the interactive shell using the main module logic
    return asyncio.run(_run_shell(agent_dir, args))


async def _run_shell(agent_dir: Path, args) -> int:
    """Run the interactive chat shell for an agent."""
    from kestrel_sovereign.storage import AsyncStorage
    from kestrel_sovereign.security.encryption import DecryptionError
    from kestrel_sovereign.kestrel_agent import KestrelAgent
    from kestrel_sovereign.llm.service import LLMService
    from kestrel_sovereign.kestrel_config.constants import SHUTDOWN_TIMEOUT
    import logging

    logger = logging.getLogger(__name__)

    # Load agent DID
    db_path = agent_dir / "kestrel_prime.db"
    storage = AsyncStorage(str(db_path))
    await storage.initialize()
    try:
        agent_nodes = await storage.get_nodes_by_type("agent")
        if not agent_nodes:
            print("No agent found in the database. Run inception service first.")
            return 1
        agent_did = agent_nodes[0].node_id
    finally:
        await storage.close()

    # Initialize agent
    llm_service = LLMService()
    agent = KestrelAgent(
        did=agent_did,
        storage_path=str(db_path),
        llm_service=llm_service,
    )
    await agent.initialize()

    # Load extension if requested
    if hasattr(args, 'app') and args.app:
        if args.app == 'elderly':
            from kestrel_sovereign.extensions.elderly_extension import ElderlyExtension
            agent.extension = ElderlyExtension(agent)
            agent.app_context = args.app
            print(f"   Extension Loaded: {args.app}")

    print("\u2705 Kestrel Agent Initialized.")
    print(f"   DID: {agent.agent_id}")
    print(f"   Memory: {agent_dir}")

    decryption_error_count = 0
    MAX_DECRYPTION_ERRORS = 3

    try:
        while True:
            user_input = input("\n> ")
            if user_input.lower() == '!quit':
                break
            try:
                response = await agent.process_input(user_input)
                decryption_error_count = 0
                print(f"\nKestrel: {response}")
            except DecryptionError:
                decryption_error_count += 1
                print(f"\n\U0001f510 DECRYPTION ERROR: Cannot read encrypted data.")
                print(f"   This usually means KESTREL_DATA_KEY is incorrect or missing.")
                print(f"   Error count: {decryption_error_count}/{MAX_DECRYPTION_ERRORS}")

                if decryption_error_count >= MAX_DECRYPTION_ERRORS:
                    print("\n\u26a0\ufe0f  Too many decryption errors. Entering safe mode.")
                    print("   The agent cannot access encrypted memories.")
                    print("   Please verify KESTREL_DATA_KEY and restart.")
                    print("   Use !quit to exit.")
                    if hasattr(agent, '_safe_mode'):
                        agent._safe_mode = True

    except KeyboardInterrupt:
        print("\nDeactivating agent...")
    finally:
        try:
            import asyncio
            await asyncio.wait_for(agent.shutdown(), timeout=SHUTDOWN_TIMEOUT)
            print("Agent deactivated.")
        except asyncio.TimeoutError:
            print(f"Agent shutdown timed out ({SHUTDOWN_TIMEOUT}s), forcing exit.")
        except asyncio.CancelledError:
            print("Agent shutdown cancelled.")
        except Exception:
            print("Agent deactivated (with errors).")

    return 0


def cmd_health(args) -> int:
    """Run health check."""
    from kestrel_sovereign.health_check import run_health_check
    run_health_check()
    return 0


# ---------------------------------------------------------------------------
# Feature CLI commands
# ---------------------------------------------------------------------------

def _load_kestrel_toml(project_dir: Path) -> dict:
    """Load kestrel.toml from project dir. Returns empty dict if not found."""
    import toml
    toml_path = project_dir / "kestrel.toml"
    if not toml_path.exists():
        return {}
    try:
        return toml.load(toml_path)
    except Exception:
        return {}


def _save_kestrel_toml(project_dir: Path, data: dict) -> None:
    """Save data to kestrel.toml in project dir, preserving existing content."""
    import toml
    toml_path = project_dir / "kestrel.toml"

    # Load existing content and merge
    existing = _load_kestrel_toml(project_dir)
    existing.update(data)

    with open(toml_path, "w", encoding="utf-8") as f:
        toml.dump(existing, f)


def _get_toml_disabled_features(project_dir: Path) -> list:
    """Read disabled features list from kestrel.toml [features] section."""
    data = _load_kestrel_toml(project_dir)
    return data.get("features", {}).get("disabled", [])


def _set_toml_disabled_features(project_dir: Path, disabled: list) -> None:
    """Write disabled features list to kestrel.toml [features] section."""
    data = _load_kestrel_toml(project_dir)
    if "features" not in data:
        data["features"] = {}
    data["features"]["disabled"] = sorted(set(disabled))
    _save_kestrel_toml(project_dir, data)


def _resolve_feature_name(name: str, registry: dict) -> Optional[str]:
    """
    Resolve a user-provided name to a registry package name.

    Accepts: package name ("cloud"), feature class name ("RunPodFeature"),
    or human-friendly name (case-insensitive match).
    """
    # Direct package name match
    if name in registry:
        return name

    # Case-insensitive package name match
    lower = name.lower()
    for pkg_name in registry:
        if pkg_name.lower() == lower:
            return pkg_name

    # Feature class name match
    for pkg_name, info in registry.items():
        if name in info.features:
            return pkg_name

    return None


def _status_icon(status) -> str:
    """Return a status icon for display."""
    from kestrel_sovereign.feature_registry import FeatureStatus
    return {
        FeatureStatus.ENABLED: "\u2713",
        FeatureStatus.INSTALLED: "\u2713",
        FeatureStatus.DISABLED: "\u2717",
        FeatureStatus.AVAILABLE: "\u25cb",
    }.get(status, "?")


def cmd_feature(args) -> int:
    """Dispatch feature subcommands."""
    feature_commands = {
        "list": cmd_feature_list,
        "install": cmd_feature_install,
        "enable": cmd_feature_enable,
        "disable": cmd_feature_disable,
        "info": cmd_feature_info,
        "scaffold": cmd_feature_scaffold,
        "skills": cmd_feature_skills,
    }

    handler = feature_commands.get(args.feature_command)
    if handler is None:
        print("Usage: kestrel feature {list|install|enable|disable|info|scaffold|skills}")
        return 1
    return handler(args)


def cmd_feature_list(args) -> int:
    """List all features with installed/enabled status."""
    from kestrel_sovereign.feature_registry import (
        get_registry,
        FeatureStatus,
    )

    project_dir = _get_project_dir()
    toml_disabled = set(_get_toml_disabled_features(project_dir))

    registry = get_registry()

    # Override status for features disabled via kestrel.toml
    for info in registry.values():
        feature_classes = set(info.features)
        if feature_classes & toml_disabled:
            info.status = FeatureStatus.DISABLED

    # Separate installed/core from available-to-install
    installed = {}
    available = {}
    for name, info in sorted(registry.items()):
        if info.status in (FeatureStatus.ENABLED, FeatureStatus.INSTALLED, FeatureStatus.DISABLED):
            installed[name] = info
        else:
            available[name] = info

    # Print installed features
    if installed:
        print()
        print(f"  {'INSTALLED':<40} {'STATUS'}")
        print(f"  {'─' * 50}")
        for name, info in installed.items():
            icon = _status_icon(info.status)
            status_str = info.status.value
            label = info.description if len(info.description) <= 36 else info.description[:33] + "..."
            core_tag = " (core)" if info.core else ""
            print(f"  {icon} {name:<38}{core_tag:>7} {status_str}")

    # Print available features
    if available:
        print()
        print(f"  {'AVAILABLE TO INSTALL':<40} {'PACKAGE'}")
        print(f"  {'─' * 50}")
        for name, info in available.items():
            print(f"  \u25cb {name:<38} {info.package}")

    if not installed and not available:
        print("  No features found in registry")

    print()
    return 0


def cmd_feature_install(args) -> int:
    """Install a feature package via pip."""
    from kestrel_sovereign.feature_registry import load_registry

    registry = load_registry()
    pkg_name = _resolve_feature_name(args.name, registry)

    if pkg_name is None:
        print(f"Unknown feature: {args.name}")
        print("Run 'kestrel feature list' to see available features")
        return 1

    info = registry[pkg_name]

    if info.core:
        print(f"Feature '{pkg_name}' is a core feature (already included in kestrel-sovereign)")
        return 0

    package = info.package
    print(f"Installing {package}...")

    cmd = [sys.executable, "-m", "pip", "install", package]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        # Try git URL as fallback
        if info.git:
            print(f"pip install failed, trying git: {info.git}")
            cmd = [sys.executable, "-m", "pip", "install", f"git+{info.git}"]
            result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"Installed {package}")
        return 0
    else:
        print(f"Failed to install {package}")
        if result.stderr:
            # Show last few lines of error
            lines = result.stderr.strip().split("\n")
            for line in lines[-5:]:
                print(f"  {line}")
        return 1


def cmd_feature_enable(args) -> int:
    """Enable a feature (remove from disabled list in kestrel.toml)."""
    from kestrel_sovereign.feature_registry import load_registry

    registry = load_registry()
    pkg_name = _resolve_feature_name(args.name, registry)

    if pkg_name is None:
        print(f"Unknown feature: {args.name}")
        return 1

    info = registry[pkg_name]
    project_dir = _get_project_dir()
    disabled = _get_toml_disabled_features(project_dir)

    # Remove all feature classes for this package from disabled list
    removed = []
    new_disabled = []
    for feat in disabled:
        if feat in info.features:
            removed.append(feat)
        else:
            new_disabled.append(feat)

    if not removed:
        print(f"Feature '{pkg_name}' is not disabled")
        return 0

    _set_toml_disabled_features(project_dir, new_disabled)
    print(f"Enabled {pkg_name} (removed {', '.join(removed)} from disabled list)")
    print("Restart the agent for changes to take effect")
    return 0


def cmd_feature_disable(args) -> int:
    """Disable a feature (add to disabled list in kestrel.toml)."""
    from kestrel_sovereign.feature_registry import load_registry

    registry = load_registry()
    pkg_name = _resolve_feature_name(args.name, registry)

    if pkg_name is None:
        print(f"Unknown feature: {args.name}")
        return 1

    info = registry[pkg_name]
    project_dir = _get_project_dir()
    disabled = _get_toml_disabled_features(project_dir)

    # Add all feature classes for this package to disabled list
    added = []
    for feat in info.features:
        if feat not in disabled:
            disabled.append(feat)
            added.append(feat)

    if not added:
        print(f"Feature '{pkg_name}' is already disabled")
        return 0

    _set_toml_disabled_features(project_dir, disabled)
    print(f"Disabled {pkg_name} (added {', '.join(added)} to disabled list)")
    print("Restart the agent for changes to take effect")
    return 0


def cmd_feature_info(args) -> int:
    """Show detailed info about a feature package."""
    from kestrel_sovereign.feature_registry import get_registry, FeatureStatus

    project_dir = _get_project_dir()
    toml_disabled = set(_get_toml_disabled_features(project_dir))

    registry = get_registry()

    # Override status for kestrel.toml disabled
    for info in registry.values():
        if set(info.features) & toml_disabled:
            info.status = FeatureStatus.DISABLED

    pkg_name = _resolve_feature_name(args.name, registry)
    if pkg_name is None:
        print(f"Unknown feature: {args.name}")
        return 1

    info = registry[pkg_name]

    print()
    print(f"  {pkg_name}")
    print(f"  {'─' * 40}")
    print(f"  Description:  {info.description}")
    print(f"  Package:      {info.package}")
    print(f"  Status:       {info.status.value}")
    print(f"  Core:         {'yes' if info.core else 'no'}")
    print(f"  Git:          {info.git}")

    if info.tags:
        print(f"  Tags:         {', '.join(info.tags)}")

    if info.features:
        print(f"  Features:     {', '.join(info.features)}")

    if info.skills:
        print(f"\n  Skills:")
        for skill in info.skills:
            tags_str = f" [{', '.join(skill.tags)}]" if skill.tags else ""
            print(f"    {skill.name:<24} {skill.description}{tags_str}")

    print()
    return 0


def cmd_feature_scaffold(args) -> int:
    """Generate a feature package project template."""
    name = args.name.lower().replace("-", "_").replace(" ", "_")
    pkg_name = f"kestrel_feature_{name}"
    dir_name = f"kestrel-feature-{name.replace('_', '-')}"

    scaffold_dir = Path(dir_name)
    if scaffold_dir.exists():
        print(f"Directory '{dir_name}' already exists")
        return 1

    # Create directory structure
    src_dir = scaffold_dir / pkg_name
    tests_dir = scaffold_dir / "tests"

    src_dir.mkdir(parents=True)
    tests_dir.mkdir(parents=True)

    # Feature class name
    class_name = "".join(word.capitalize() for word in name.split("_")) + "Feature"

    # __init__.py
    (src_dir / "__init__.py").write_text(
        f'"""Kestrel Feature: {name}"""\n'
        f"from .feature import {class_name}\n\n"
        f'__all__ = ["{class_name}"]\n'
    )

    # feature.py
    (src_dir / "feature.py").write_text(
        f'"""{class_name} — TODO: describe your feature."""\n\n'
        f"from kestrel_sovereign.features.base import Feature\n\n\n"
        f"class {class_name}(Feature):\n"
        f'    """TODO: describe what this feature does."""\n\n'
        f"    @property\n"
        f"    def name(self) -> str:\n"
        f'        return "{class_name}"\n\n'
        f"    @property\n"
        f"    def description(self) -> str:\n"
        f'        return "TODO: add description"\n\n'
        f"    async def initialize(self) -> None:\n"
        f"        pass\n\n"
        f"    async def shutdown(self) -> None:\n"
        f"        pass\n"
    )

    # pyproject.toml
    (scaffold_dir / "pyproject.toml").write_text(
        f'[build-system]\n'
        f'requires = ["setuptools>=64", "wheel"]\n'
        f'build-backend = "setuptools.backends._legacy:_Backend"\n\n'
        f'[project]\n'
        f'name = "kestrel-feature-{name.replace("_", "-")}"\n'
        f'version = "0.1.0"\n'
        f'description = "Kestrel feature: {name}"\n'
        f'requires-python = ">=3.11"\n'
        f'dependencies = ["kestrel-sovereign"]\n\n'
        f'[project.entry-points."kestrel_sovereign.features"]\n'
        f'{class_name} = "{pkg_name}.feature:{class_name}"\n'
    )

    # tests/__init__.py
    (tests_dir / "__init__.py").write_text("")

    # tests/test_feature.py
    (tests_dir / f"test_{name}.py").write_text(
        f'"""Tests for {class_name}."""\n\n'
        f"from {pkg_name}.feature import {class_name}\n\n\n"
        f"def test_feature_name():\n"
        f"    # TODO: instantiate with a mock agent\n"
        f"    pass\n"
    )

    print(f"Scaffolded feature package: {dir_name}/")
    print(f"  {pkg_name}/")
    print(f"    __init__.py")
    print(f"    feature.py          <- implement {class_name} here")
    print(f"  tests/")
    print(f"    test_{name}.py")
    print(f"  pyproject.toml        <- entry_point registered")
    print()
    print(f"Install in dev mode:  pip install -e ./{dir_name}")
    return 0


def cmd_feature_skills(args) -> int:
    """List skills for a specific feature package."""
    from kestrel_sovereign.feature_registry import get_skills_for_package, load_registry

    registry = load_registry()
    pkg_name = _resolve_feature_name(args.name, registry)

    if pkg_name is None:
        print(f"Unknown feature: {args.name}")
        return 1

    skills = get_skills_for_package(pkg_name)
    if not skills:
        print(f"No skills declared for '{pkg_name}'")
        return 0

    print()
    print(f"  Skills for {pkg_name}:")
    print(f"  {'─' * 50}")
    for skill in skills:
        tags_str = f" [{', '.join(skill.tags)}]" if skill.tags else ""
        print(f"  {skill.name:<24} {skill.description}{tags_str}")
    print()
    return 0


# ---------------------------------------------------------------------------
# Skills CLI commands
# ---------------------------------------------------------------------------

def cmd_skills(args) -> int:
    """Dispatch skills subcommands."""
    skills_commands = {
        "search": cmd_skills_search,
    }

    handler = skills_commands.get(args.skills_command)
    if handler is None:
        print("Usage: kestrel skills {search}")
        return 1
    return handler(args)


def cmd_skills_search(args) -> int:
    """Search all skills by name or tag."""
    from kestrel_sovereign.feature_registry import load_registry

    query = args.query.lower()
    registry = load_registry()

    matches = []
    for pkg_name, info in registry.items():
        for skill in info.skills:
            if (query in skill.name.lower()
                    or query in skill.description.lower()
                    or query in skill.category.lower()
                    or any(query in t.lower() for t in skill.tags)):
                matches.append((pkg_name, skill))

    if not matches:
        print(f"No skills matching '{args.query}'")
        return 0

    print()
    print(f"  {'SKILL':<24} {'PACKAGE':<16} {'DESCRIPTION'}")
    print(f"  {'─' * 60}")
    for pkg_name, skill in matches:
        print(f"  {skill.name:<24} {pkg_name:<16} {skill.description}")
    print()
    return 0


def cmd_config(args) -> int:
    """Show or edit agent config."""
    from kestrel_sovereign.agent_config import AgentConfig, find_agent_dir

    agent_dir = find_agent_dir(args.agent_dir)
    if not agent_dir:
        print("No agent directory found")
        return 1

    config = AgentConfig.from_directory(agent_dir)

    if args.set_port:
        config.port = args.set_port
        config.save()
        print(f"Port set to {args.set_port}")

    if args.set_name:
        config.name = args.set_name
        config.save()
        print(f"Name set to {args.set_name}")

    if args.init:
        config.save()
        print(f"Config created: {config.config_file}")

    # Show current config
    print(f"\n{config.name}")
    print(f"   Directory: {config.agent_dir}")
    print(f"   Port:      {config.port}")
    print(f"   Host:      {config.host}")
    print(f"   Config:    {config.config_file}")
    print(f"   Exists:    {'Yes' if config.config_file.exists() else 'No'}")

    return 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the unified CLI."""
    parser = argparse.ArgumentParser(
        prog="kestrel",
        description="Kestrel Sovereign Agent Manager",
    )
    parser.add_argument(
        "--version", action="version", version=f"kestrel {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command")

    # kestrel start [name] [--subprocess]
    start_p = subparsers.add_parser("start", help="Start host and/or agents")
    start_p.add_argument("name", nargs="?", help="Agent name (omit for all)")
    start_p.add_argument(
        "--subprocess", action="store_true",
        help="Run each agent as a separate process (legacy mode)",
    )

    # kestrel stop [name]
    stop_p = subparsers.add_parser("stop", help="Stop host and/or agents")
    stop_p.add_argument("name", nargs="?", help="Agent name (omit for all)")

    # kestrel status
    subparsers.add_parser("status", help="Show status of host and agents")

    # kestrel logs <name>
    logs_p = subparsers.add_parser("logs", help="Tail agent or host logs")
    logs_p.add_argument("name", help="Agent name or 'host'")
    logs_p.add_argument("-n", "--lines", type=int, default=50)
    logs_p.add_argument("-f", "--follow", action="store_true")

    # kestrel list
    subparsers.add_parser("list", help="List all agents in rookery")

    # kestrel create <name>
    create_p = subparsers.add_parser("create", help="Create a new agent")
    create_p.add_argument("name", help="Agent name")
    create_p.add_argument("--port", type=int, help="Override port assignment")

    # kestrel shell <name>
    shell_p = subparsers.add_parser("shell", help="Interactive CLI chat")
    shell_p.add_argument("name", help="Agent name")
    shell_p.add_argument(
        "--app", type=str, default=None, choices=["elderly"],
        help="Load an application extension",
    )

    # kestrel health
    subparsers.add_parser("health", help="Run health check")

    # kestrel config <agent_dir>
    config_p = subparsers.add_parser("config", help="Show/edit agent config")
    config_p.add_argument("agent_dir", nargs="?", help="Agent directory")
    config_p.add_argument("--init", action="store_true", help="Create kestrel.toml")
    config_p.add_argument("--set-port", type=int, help="Set port")
    config_p.add_argument("--set-name", type=str, help="Set name")

    # kestrel feature {list|install|enable|disable|info|scaffold|skills}
    feature_p = subparsers.add_parser("feature", help="Manage features")
    feature_sub = feature_p.add_subparsers(dest="feature_command")

    feature_sub.add_parser("list", help="List features with status")

    feat_install = feature_sub.add_parser("install", help="Install a feature package")
    feat_install.add_argument("name", help="Feature name (e.g. cloud, voice)")

    feat_enable = feature_sub.add_parser("enable", help="Enable a disabled feature")
    feat_enable.add_argument("name", help="Feature or package name")

    feat_disable = feature_sub.add_parser("disable", help="Disable a feature")
    feat_disable.add_argument("name", help="Feature or package name")

    feat_info = feature_sub.add_parser("info", help="Show feature details")
    feat_info.add_argument("name", help="Feature or package name")

    feat_scaffold = feature_sub.add_parser("scaffold", help="Generate feature project template")
    feat_scaffold.add_argument("name", help="Feature name (e.g. myfeature)")

    feat_skills = feature_sub.add_parser("skills", help="List skills in a feature")
    feat_skills.add_argument("name", help="Feature or package name")

    # kestrel skills {search}
    skills_p = subparsers.add_parser("skills", help="Search and manage skills")
    skills_sub = skills_p.add_subparsers(dest="skills_command")

    skills_search = skills_sub.add_parser("search", help="Search skills by name/tag")
    skills_search.add_argument("query", help="Search query")

    return parser


def main() -> int:
    """Main entry point for the kestrel CLI."""
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    commands = {
        "start": cmd_start,
        "stop": cmd_stop,
        "status": cmd_status,
        "logs": cmd_logs,
        "list": cmd_list,
        "create": cmd_create,
        "shell": cmd_shell,
        "health": cmd_health,
        "config": cmd_config,
        "feature": cmd_feature,
        "skills": cmd_skills,
    }

    handler = commands.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
