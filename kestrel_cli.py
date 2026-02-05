#!/usr/bin/env python3
"""
Kestrel Sovereign Agent CLI

Cross-platform command-line interface for managing Kestrel agents.
Works on Windows, macOS, and Linux without requiring shell scripts.

Usage:
    uv run kestrel start ./agent_data/claw     # Start agent (reads port from config)
    uv run kestrel stop ./agent_data/claw      # Stop agent
    uv run kestrel status                       # Show all running agents
    uv run kestrel list                         # List available agents
    uv run kestrel health                       # Run health check
    uv run kestrel create                       # Create a new agent
    uv run kestrel chat ./agent_data/claw      # Start CLI chat

Each agent directory can have a kestrel.toml config:
    [agent]
    name = "Claw"
    port = 8888
"""

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from kestrel_sovereign.agent_config import (
    AgentConfig,
    find_agent_dir,
    list_agents,
    DEFAULT_PORT,
)


def get_script_dir() -> Path:
    """Get the directory containing this script."""
    return Path(__file__).parent.resolve()


def is_port_in_use(port: int) -> bool:
    """Check if a port is in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def is_process_running(pid: int) -> bool:
    """Check if a process with the given PID is running."""
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(0x1000, False, pid)
            if handle:
                kernel32.CloseHandle(handle)
                return True
            return False
        except Exception:
            return False
    else:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


def kill_process(pid: int, force: bool = False) -> bool:
    """Kill a process by PID. Returns True if successful."""
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                         capture_output=True, check=True)
        else:
            sig = signal.SIGKILL if force else signal.SIGTERM
            os.kill(pid, sig)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def wait_for_health(port: int, timeout: int = 30) -> bool:
    """Wait for server health endpoint to respond."""
    import urllib.request
    import urllib.error

    url = f"http://localhost:{port}/health"
    start = time.time()

    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
            pass
        time.sleep(0.5)

    return False


def cmd_start(args):
    """Start the Kestrel server for an agent."""
    script_dir = get_script_dir()

    # Find agent directory
    agent_dir = find_agent_dir(args.agent_dir)
    if not agent_dir:
        print("❌ No agent directory found")
        print("   Specify one: uv run kestrel start ./agent_data/claw")
        print("   Or create:   uv run kestrel create")
        return 1

    # Load config
    config = AgentConfig.from_directory(agent_dir)

    # CLI args override config
    port = args.port or config.port

    # Check if already running via PID file
    existing_pid = config.get_pid()
    if existing_pid and is_process_running(existing_pid):
        print(f"❌ {config.name} already running (PID: {existing_pid})")
        print(f"   Stop with: uv run kestrel stop {agent_dir}")
        return 1
    elif existing_pid:
        print("⚠️  Stale PID file found, removing...")
        config.clear_pid()

    # Check if port is in use
    if is_port_in_use(port):
        print(f"❌ Port {port} is already in use")
        print("   Change port in kestrel.toml or use --port")
        return 1

    print(f"🦅 Starting {config.name}...")
    print(f"   Agent: {agent_dir}")
    print(f"   Port:  {port}")
    print(f"   Log:   {config.log_file}")

    # Load .env if exists
    env = os.environ.copy()
    env_file = script_dir / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                env[key.strip()] = value.strip().strip('"').strip("'")

    # Set KESTREL_DB_PATH
    env["KESTREL_DB_PATH"] = str(agent_dir)
    env["PORT"] = str(port)

    # Create log file directory
    config.log_file.parent.mkdir(parents=True, exist_ok=True)

    # Start server
    with open(config.log_file, "w") as log:
        if sys.platform == "win32":
            process = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "server:app",
                 "--host", config.host, "--port", str(port)],
                cwd=script_dir,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            )
        else:
            # Use sys.executable to run uvicorn as a module
            # This works both when running directly and through uv
            process = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "server:app",
                 "--host", config.host, "--port", str(port)],
                cwd=script_dir,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True
            )

    # Save PID to agent directory
    config.save_pid(process.pid)

    # Wait for server to start
    print("   Waiting for server...")
    if wait_for_health(port, timeout=30):
        print(f"\n✅ {config.name} started!")
        print(f"   URL: http://localhost:{port}")
        print(f"   PID: {process.pid}")
        print(f"\n   Stop: uv run kestrel stop {agent_dir}")
        return 0
    else:
        print(f"\n⚠️  Server started but health check timed out")
        print(f"   Check log: {config.log_file}")
        return 1


def cmd_stop(args):
    """Stop the Kestrel server for an agent."""
    # Find agent directory
    agent_dir = find_agent_dir(args.agent_dir)
    if not agent_dir:
        print("❌ No agent directory found")
        print("   Specify one: uv run kestrel stop ./agent_data/claw")
        return 1

    config = AgentConfig.from_directory(agent_dir)
    pid = config.get_pid()

    if pid is None:
        print(f"⚠️  No PID file for {config.name}")
        if is_port_in_use(config.port):
            print(f"   But port {config.port} is in use by another process")
        return 0

    print(f"🛑 Stopping {config.name} (PID: {pid})...")

    if not is_process_running(pid):
        print(f"⚠️  Process {pid} not found (already stopped?)")
        config.clear_pid()
        return 0

    # Try graceful shutdown
    kill_process(pid, force=False)

    # Wait for graceful shutdown
    for _ in range(10):
        if not is_process_running(pid):
            break
        time.sleep(0.5)

    # Force kill if still running
    if is_process_running(pid):
        print("⚠️  Graceful shutdown failed, forcing...")
        kill_process(pid, force=True)
        time.sleep(0.5)

    config.clear_pid()
    print(f"✅ {config.name} stopped")
    return 0


def cmd_status(args):
    """Check status of agents."""
    print("🦅 Kestrel Agent Status")
    print("=" * 50)

    # If specific agent given, show that
    if args.agent_dir:
        agent_dir = find_agent_dir(args.agent_dir)
        if agent_dir:
            config = AgentConfig.from_directory(agent_dir)
            _print_agent_status(config)
            return 0
        else:
            print(f"❌ Agent not found: {args.agent_dir}")
            return 1

    # Otherwise show all known agents
    agents = list_agents("./agent_data")

    # Also check common locations
    for loc in ["./my_agent", "."]:
        path = Path(loc).resolve()
        if (path / "kestrel_prime.db").exists():
            config = AgentConfig.from_directory(path)
            if config.agent_dir not in [a.agent_dir for a in agents]:
                agents.append(config)

    if not agents:
        print("   No agents found")
        print("\n   Create one: uv run kestrel create")
        return 0

    for config in agents:
        _print_agent_status(config)
        print()

    return 0


def _print_agent_status(config: AgentConfig):
    """Print status for a single agent."""
    pid = config.get_pid()
    running = pid and is_process_running(pid)
    port_in_use = is_port_in_use(config.port)

    status = "🟢 Running" if running else "⚪ Stopped"
    print(f"\n{config.name} ({config.agent_dir.name})")
    print(f"   Status: {status}")
    print(f"   Port:   {config.port} {'(in use)' if port_in_use else '(available)'}")

    if running:
        print(f"   PID:    {pid}")
        print(f"   URL:    http://localhost:{config.port}")

        # Health check
        import urllib.request
        import urllib.error
        try:
            with urllib.request.urlopen(f"http://localhost:{config.port}/health", timeout=2) as resp:
                if resp.status == 200:
                    print(f"   Health: ✅ OK")
                else:
                    print(f"   Health: ⚠️  Status {resp.status}")
        except Exception:
            print(f"   Health: ❌ Not responding")


def cmd_list(args):
    """List all available agents."""
    print("🦅 Available Agents")
    print("=" * 50)

    agents = list_agents("./agent_data")

    # Also check common locations
    for loc in ["./my_agent"]:
        path = Path(loc).resolve()
        if (path / "kestrel_prime.db").exists():
            config = AgentConfig.from_directory(path)
            agents.append(config)

    if not agents:
        print("   No agents found")
        print("\n   Create one: uv run kestrel create")
        return 0

    for config in agents:
        pid = config.get_pid()
        running = pid and is_process_running(pid)
        status = "🟢" if running else "⚪"
        has_config = "✓" if config.config_file.exists() else "-"
        print(f"   {status} {config.name:20} {config.agent_dir.name:15} port:{config.port:5} config:{has_config}")

    print(f"\n   Total: {len(agents)} agent(s)")
    return 0


def cmd_health(args):
    """Run health check."""
    from kestrel_sovereign.health_check import run_health_check
    run_health_check()
    return 0


def cmd_create(args):
    """Create a new agent."""
    output_dir = args.output_dir or "./my_agent"
    output_path = Path(output_dir).resolve()

    print(f"🦅 Creating new Kestrel agent...")
    print(f"   Output: {output_path}")

    # Run inception service
    cmd = [
        sys.executable, "-m", "kestrel_sovereign.inception_service",
        "--output-dir", str(output_path)
    ]

    if args.name:
        cmd.extend(["--name", args.name])

    if args.test:
        cmd.append("--test")

    result = subprocess.run(cmd, cwd=get_script_dir())

    if result.returncode == 0:
        # Create default config
        config = AgentConfig.from_directory(output_path)
        if args.name:
            config.name = args.name
        if args.port:
            config.port = args.port
        config.save()

        print(f"\n✅ Agent created!")
        print(f"   Config: {config.config_file}")
        print(f"\n   Start:  uv run kestrel start {output_path}")
        print(f"   Chat:   uv run kestrel chat {output_path}")

    return result.returncode


def cmd_chat(args):
    """Start CLI chat with an agent."""
    agent_dir = find_agent_dir(args.agent_dir)

    if not agent_dir:
        print("❌ No agent directory found")
        print("   Specify one: uv run kestrel chat ./agent_data/claw")
        print("   Or create:   uv run kestrel create")
        return 1

    config = AgentConfig.from_directory(agent_dir)
    print(f"🦅 Starting chat with {config.name}...")

    result = subprocess.run(
        [sys.executable, "main.py", str(agent_dir)],
        cwd=get_script_dir()
    )

    return result.returncode


def cmd_config(args):
    """Show or edit agent config."""
    agent_dir = find_agent_dir(args.agent_dir)

    if not agent_dir:
        print("❌ No agent directory found")
        return 1

    config = AgentConfig.from_directory(agent_dir)

    if args.set_port:
        config.port = args.set_port
        config.save()
        print(f"✅ Port set to {args.set_port}")

    if args.set_name:
        config.name = args.set_name
        config.save()
        print(f"✅ Name set to {args.set_name}")

    if args.init:
        config.save()
        print(f"✅ Config created: {config.config_file}")

    # Show current config
    print(f"\n{config.name}")
    print(f"   Directory: {config.agent_dir}")
    print(f"   Port:      {config.port}")
    print(f"   Host:      {config.host}")
    print(f"   Config:    {config.config_file}")
    print(f"   Exists:    {'Yes' if config.config_file.exists() else 'No'}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Kestrel Sovereign Agent CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run kestrel start ./agent_data/claw   # Start agent
  uv run kestrel stop ./agent_data/claw    # Stop agent
  uv run kestrel status                     # Show all agents
  uv run kestrel list                       # List available agents
  uv run kestrel create --name MyAgent      # Create new agent
  uv run kestrel chat ./agent_data/claw    # CLI chat

Agent Config (kestrel.toml):
  Each agent can have a config file in its directory.
  Use 'kestrel config --init' to create one.
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # start
    p_start = subparsers.add_parser("start", help="Start an agent server")
    p_start.add_argument("agent_dir", nargs="?", help="Agent directory")
    p_start.add_argument("--port", type=int, help="Override port from config")
    p_start.set_defaults(func=cmd_start)

    # stop
    p_stop = subparsers.add_parser("stop", help="Stop an agent server")
    p_stop.add_argument("agent_dir", nargs="?", help="Agent directory")
    p_stop.set_defaults(func=cmd_stop)

    # status
    p_status = subparsers.add_parser("status", help="Show agent status")
    p_status.add_argument("agent_dir", nargs="?", help="Agent directory (optional)")
    p_status.set_defaults(func=cmd_status)

    # list
    p_list = subparsers.add_parser("list", help="List available agents")
    p_list.set_defaults(func=cmd_list)

    # health
    p_health = subparsers.add_parser("health", help="Run health check")
    p_health.set_defaults(func=cmd_health)

    # create
    p_create = subparsers.add_parser("create", help="Create a new agent")
    p_create.add_argument("--output-dir", type=str, help="Output directory (default: ./my_agent)")
    p_create.add_argument("--name", type=str, help="Agent name")
    p_create.add_argument("--port", type=int, help="Default port")
    p_create.add_argument("--test", action="store_true", help="Create as test instance")
    p_create.set_defaults(func=cmd_create)

    # chat
    p_chat = subparsers.add_parser("chat", help="Start CLI chat")
    p_chat.add_argument("agent_dir", nargs="?", help="Agent directory")
    p_chat.set_defaults(func=cmd_chat)

    # config
    p_config = subparsers.add_parser("config", help="Show/edit agent config")
    p_config.add_argument("agent_dir", nargs="?", help="Agent directory")
    p_config.add_argument("--init", action="store_true", help="Create kestrel.toml")
    p_config.add_argument("--set-port", type=int, help="Set port")
    p_config.add_argument("--set-name", type=str, help="Set name")
    p_config.set_defaults(func=cmd_config)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
