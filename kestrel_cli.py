#!/usr/bin/env python3
"""
Kestrel Sovereign Agent CLI

Cross-platform command-line interface for managing Kestrel agents.
Works on Windows, macOS, and Linux without requiring shell scripts.

Usage:
    uv run python kestrel_cli.py start          # Start the server
    uv run python kestrel_cli.py stop           # Stop the server
    uv run python kestrel_cli.py status         # Check server status
    uv run python kestrel_cli.py health         # Run health check
    uv run python kestrel_cli.py create         # Create a new agent
    uv run python kestrel_cli.py chat           # Start CLI chat
"""

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

# Constants
DEFAULT_PORT = 8888
PID_FILE = ".kestrel.pid"
LOG_DIR = "logs"
LOG_FILE = "logs/kestrel.log"


def get_script_dir() -> Path:
    """Get the directory containing this script."""
    return Path(__file__).parent.resolve()


def is_port_in_use(port: int) -> bool:
    """Check if a port is in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def get_pid_from_file(script_dir: Path) -> int | None:
    """Read PID from file, return None if not exists or invalid."""
    pid_path = script_dir / PID_FILE
    if not pid_path.exists():
        return None
    try:
        pid = int(pid_path.read_text().strip())
        return pid
    except (ValueError, OSError):
        return None


def is_process_running(pid: int) -> bool:
    """Check if a process with the given PID is running."""
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
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
    """Start the Kestrel server."""
    script_dir = get_script_dir()
    pid_path = script_dir / PID_FILE
    log_dir = script_dir / LOG_DIR
    log_file = script_dir / LOG_FILE
    port = args.port or DEFAULT_PORT
    
    # Create logs directory
    log_dir.mkdir(exist_ok=True)
    
    # Check if already running via PID file
    existing_pid = get_pid_from_file(script_dir)
    if existing_pid and is_process_running(existing_pid):
        print(f"❌ Kestrel server already running (PID: {existing_pid})")
        print("   Use: uv run python kestrel_cli.py stop")
        return 1
    elif existing_pid:
        print("⚠️  Stale PID file found, removing...")
        pid_path.unlink(missing_ok=True)
    
    # Check if port is in use
    if is_port_in_use(port):
        print(f"❌ Port {port} is already in use")
        return 1
    
    print("🦅 Starting Kestrel Sovereign Agent...")
    print(f"   Port: {port}")
    print(f"   Logs: {log_file}")
    
    # Load .env if exists
    env = os.environ.copy()
    env_file = script_dir / ".env"
    if env_file.exists():
        print(f"   Loading: .env")
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                env[key.strip()] = value.strip().strip('"').strip("'")
    
    # Set KESTREL_DB_PATH if not set
    if "KESTREL_DB_PATH" not in env and args.agent_dir:
        env["KESTREL_DB_PATH"] = str(args.agent_dir)
    
    # Start server
    with open(log_file, "w") as log:
        if sys.platform == "win32":
            # Windows: use subprocess with CREATE_NEW_PROCESS_GROUP
            process = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "server:app", 
                 "--host", "0.0.0.0", "--port", str(port)],
                cwd=script_dir,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            )
        else:
            # Unix: use uv run
            process = subprocess.Popen(
                ["uv", "run", "uvicorn", "server:app",
                 "--host", "0.0.0.0", "--port", str(port)],
                cwd=script_dir,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True
            )
    
    # Save PID
    pid_path.write_text(str(process.pid))
    
    # Wait for server to start
    print("   Waiting for server to start...")
    if wait_for_health(port, timeout=30):
        print(f"\n✅ Kestrel server started successfully!")
        print(f"   URL: http://localhost:{port}")
        print(f"   PID: {process.pid}")
        print(f"\n   Stop with: uv run python kestrel_cli.py stop")
        return 0
    else:
        print(f"\n⚠️  Server started but health check timed out")
        print(f"   Check logs: {log_file}")
        return 1


def cmd_stop(args):
    """Stop the Kestrel server."""
    script_dir = get_script_dir()
    pid_path = script_dir / PID_FILE
    port = args.port or DEFAULT_PORT
    
    pid = get_pid_from_file(script_dir)
    
    if pid is None:
        print("⚠️  No PID file found - server may not be running")
        if is_port_in_use(port):
            print(f"   But port {port} is in use by another process")
        return 0
    
    print(f"🛑 Stopping Kestrel server (PID: {pid})...")
    
    if not is_process_running(pid):
        print(f"⚠️  Process {pid} not found (already stopped?)")
        pid_path.unlink(missing_ok=True)
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
    
    pid_path.unlink(missing_ok=True)
    print("✅ Kestrel server stopped")
    return 0


def cmd_status(args):
    """Check server status."""
    script_dir = get_script_dir()
    port = args.port or DEFAULT_PORT
    
    pid = get_pid_from_file(script_dir)
    
    print("🦅 Kestrel Server Status")
    print("=" * 30)
    
    if pid and is_process_running(pid):
        print(f"   Process: Running (PID: {pid})")
    elif pid:
        print(f"   Process: Stopped (stale PID file: {pid})")
    else:
        print("   Process: Not running")
    
    if is_port_in_use(port):
        print(f"   Port {port}: In use")
        # Try health check
        import urllib.request
        import urllib.error
        try:
            with urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2) as resp:
                if resp.status == 200:
                    print(f"   Health: ✅ OK")
                else:
                    print(f"   Health: ⚠️  Status {resp.status}")
        except Exception as e:
            print(f"   Health: ❌ Not responding")
    else:
        print(f"   Port {port}: Available")
    
    return 0


def cmd_health(args):
    """Run health check."""
    from kestrel_sovereign.health_check import run_health_check
    run_health_check()
    return 0


def cmd_create(args):
    """Create a new agent."""
    import asyncio
    
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
        print(f"\n✅ Agent created successfully!")
        print(f"\n   Start server: KESTREL_DB_PATH={output_path} uv run python server.py")
        print(f"   Or CLI chat:  uv run python main.py {output_path}")
    
    return result.returncode


def cmd_chat(args):
    """Start CLI chat with an agent."""
    agent_dir = args.agent_dir
    
    if not agent_dir:
        # Look for common locations
        for candidate in ["./my_agent", "./agent_data/claw", "."]:
            db_file = Path(candidate) / "kestrel_prime.db"
            if db_file.exists():
                agent_dir = candidate
                break
    
    if not agent_dir:
        print("❌ No agent directory specified and none found")
        print("   Create one with: uv run python kestrel_cli.py create")
        return 1
    
    agent_path = Path(agent_dir).resolve()
    db_file = agent_path / "kestrel_prime.db"
    
    if not db_file.exists():
        print(f"❌ No agent database found at: {db_file}")
        return 1
    
    print(f"🦅 Starting chat with agent in: {agent_path}")
    
    result = subprocess.run(
        [sys.executable, "main.py", str(agent_path)],
        cwd=get_script_dir()
    )
    
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Kestrel Sovereign Agent CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run kestrel health         # Check prerequisites
  uv run kestrel create         # Create a new agent
  uv run kestrel start          # Start the server
  uv run kestrel status         # Check if running
  uv run kestrel stop           # Stop the server
  uv run kestrel chat           # CLI chat interface
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # start
    p_start = subparsers.add_parser("start", help="Start the Kestrel server")
    p_start.add_argument("--port", type=int, help=f"Port to listen on (default: {DEFAULT_PORT})")
    p_start.add_argument("--agent-dir", type=str, help="Agent directory (default: from KESTREL_DB_PATH)")
    p_start.set_defaults(func=cmd_start)
    
    # stop
    p_stop = subparsers.add_parser("stop", help="Stop the Kestrel server")
    p_stop.add_argument("--port", type=int, help=f"Port to check (default: {DEFAULT_PORT})")
    p_stop.set_defaults(func=cmd_stop)
    
    # status
    p_status = subparsers.add_parser("status", help="Check server status")
    p_status.add_argument("--port", type=int, help=f"Port to check (default: {DEFAULT_PORT})")
    p_status.set_defaults(func=cmd_status)
    
    # health
    p_health = subparsers.add_parser("health", help="Run health check")
    p_health.set_defaults(func=cmd_health)
    
    # create
    p_create = subparsers.add_parser("create", help="Create a new agent")
    p_create.add_argument("--output-dir", type=str, help="Output directory (default: ./my_agent)")
    p_create.add_argument("--name", type=str, help="Agent name")
    p_create.add_argument("--test", action="store_true", help="Create as test instance")
    p_create.set_defaults(func=cmd_create)
    
    # chat
    p_chat = subparsers.add_parser("chat", help="Start CLI chat")
    p_chat.add_argument("agent_dir", nargs="?", help="Agent directory")
    p_chat.set_defaults(func=cmd_chat)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
