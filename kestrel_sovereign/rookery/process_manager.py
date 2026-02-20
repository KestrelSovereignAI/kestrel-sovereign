"""
Agent Process Manager - Starts, stops, and monitors agent subprocesses.

The ProcessManager is used by both the CLI (`kestrel start/stop`) and the
host (`host.py` lifespan + API endpoints) to manage agent processes.

Each agent runs as a separate uvicorn process on its own port:
    KESTREL_DB_PATH=agent_data/claw PORT=8801 uvicorn server:app

Remote agents (url-only in config) are ignored — they're not managed by
this host.
"""

import logging
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from kestrel_sovereign.rookery.config import (
    LocalAgentConfig,
    RookeryConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class AgentProcess:
    """State for a single managed agent process."""

    name: str
    port: int
    data_dir: Path  # Resolved absolute path
    pid: Optional[int] = None
    pid_file: Optional[Path] = None
    log_file: Optional[Path] = None


class ProcessManager:
    """Manages agent subprocesses for a Kestrel Host.

    Usage:
        pm = ProcessManager(project_dir=Path("/path/to/kestrel"))
        pm.start_all(rookery_config)
        ...
        pm.stop_all()
    """

    def __init__(self, project_dir: Path):
        self.project_dir = project_dir.resolve()
        self._agents: dict[str, AgentProcess] = {}

    @property
    def agents(self) -> dict[str, AgentProcess]:
        """Registered agent processes (keyed by name)."""
        return dict(self._agents)

    # ------------------------------------------------------------------
    # Low-level helpers (static so CLI can use them without a PM instance)
    # ------------------------------------------------------------------

    @staticmethod
    def is_port_in_use(port: int) -> bool:
        """Check if a TCP port is already bound."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("localhost", port)) == 0

    @staticmethod
    def is_process_running(pid: int) -> bool:
        """Check if a process with the given PID is alive."""
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

    @staticmethod
    def read_pid(pid_file: Path) -> Optional[int]:
        """Read PID from file, return None if missing or invalid."""
        if not pid_file.exists():
            return None
        try:
            return int(pid_file.read_text().strip())
        except (ValueError, OSError):
            return None

    @staticmethod
    def write_pid(pid_file: Path, pid: int) -> None:
        """Write PID to file."""
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        pid_file.write_text(str(pid))

    @staticmethod
    def clear_pid(pid_file: Path) -> None:
        """Remove PID file."""
        pid_file.unlink(missing_ok=True)

    @staticmethod
    def agent_pid_file(agent_dir: Path) -> Path:
        """PID file path for an agent process."""
        return agent_dir / "agent.pid"

    @staticmethod
    def agent_log_file(agent_dir: Path) -> Path:
        """Log file path for an agent process."""
        return agent_dir / "agent.log"

    @staticmethod
    def kill_process(pid: int, force: bool = False) -> bool:
        """Send signal to a process. Returns True if signal sent."""
        try:
            if sys.platform == "win32":
                subprocess.run(
                    ["taskkill", "/F", "/PID", str(pid)],
                    capture_output=True,
                    check=True,
                )
            else:
                sig = signal.SIGKILL if force else signal.SIGTERM
                os.kill(pid, sig)
            return True
        except (OSError, subprocess.CalledProcessError):
            return False

    @staticmethod
    def wait_for_health(port: int, timeout: int = 30) -> bool:
        """Wait for a server's /health endpoint to respond 200."""
        import urllib.request
        import urllib.error

        url = f"http://localhost:{port}/health"
        start = time.time()

        while time.time() - start < timeout:
            try:
                with urllib.request.urlopen(url, timeout=2) as resp:
                    if resp.status == 200:
                        return True
            except (urllib.error.URLError, urllib.error.HTTPError,
                    TimeoutError, OSError):
                pass
            time.sleep(0.5)

        return False

    # ------------------------------------------------------------------
    # Process spawning
    # ------------------------------------------------------------------

    def _load_env(self) -> dict:
        """Load .env file into an env dict copy."""
        from dotenv import dotenv_values

        env = os.environ.copy()
        env_file = self.project_dir / ".env"
        if env_file.exists():
            env.update({
                k: v for k, v in dotenv_values(env_file).items()
                if v is not None
            })
        return env

    def _spawn(
        self,
        cmd: list[str],
        env: dict,
        log_file: Path,
        pid_file: Path,
    ) -> int:
        """Spawn a background process. Returns PID."""
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a") as log:
            kwargs = dict(
                cwd=self.project_dir,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            if sys.platform == "win32":
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                kwargs["start_new_session"] = True
            process = subprocess.Popen(cmd, **kwargs)

        self.write_pid(pid_file, process.pid)
        return process.pid

    # ------------------------------------------------------------------
    # Agent start / stop
    # ------------------------------------------------------------------

    def start_agent(
        self,
        name: str,
        config: LocalAgentConfig,
        host_bind: str = "0.0.0.0",
    ) -> AgentProcess:
        """Start a single agent process.

        Args:
            name: Agent name (from rookery config).
            config: The agent's LocalAgentConfig.
            host_bind: Interface to bind to.

        Returns:
            AgentProcess with pid set on success.

        Raises:
            RuntimeError: If agent can't be started (port in use, already
                          running, validation error).
        """
        resolved_dir = (self.project_dir / config.data_dir).resolve()
        pid_file = self.agent_pid_file(resolved_dir)
        log_file = self.agent_log_file(resolved_dir)

        # Already running?
        existing_pid = self.read_pid(pid_file)
        if existing_pid and self.is_process_running(existing_pid):
            ap = AgentProcess(
                name=name,
                port=config.port,
                data_dir=resolved_dir,
                pid=existing_pid,
                pid_file=pid_file,
                log_file=log_file,
            )
            self._agents[name] = ap
            return ap

        if existing_pid:
            self.clear_pid(pid_file)

        # Port check
        if self.is_port_in_use(config.port):
            raise RuntimeError(
                f"Agent '{name}': port {config.port} already in use"
            )

        # Runtime validation
        errors = config.validate_runtime(base_dir=self.project_dir)
        if errors:
            raise RuntimeError(
                f"Agent '{name}' validation failed: {errors[0]}"
            )

        # Build env
        env = self._load_env()
        env["KESTREL_DB_PATH"] = str(resolved_dir)
        env["PORT"] = str(config.port)
        env["KESTREL_SERVE_UI"] = "false"

        cmd = [
            sys.executable, "-m", "uvicorn", "server:app",
            "--host", host_bind, "--port", str(config.port),
        ]

        pid = self._spawn(cmd, env, log_file, pid_file)
        logger.info(f"Started agent '{name}' on :{config.port} (PID {pid})")

        ap = AgentProcess(
            name=name,
            port=config.port,
            data_dir=resolved_dir,
            pid=pid,
            pid_file=pid_file,
            log_file=log_file,
        )
        self._agents[name] = ap
        return ap

    def stop_agent(self, name: str, timeout: float = 5.0) -> bool:
        """Stop a single agent process.

        Args:
            name: Agent name.
            timeout: Seconds to wait for graceful shutdown before SIGKILL.

        Returns:
            True if agent was stopped (or wasn't running).
        """
        ap = self._agents.get(name)
        if ap is None:
            return True

        if ap.pid is None or not self.is_process_running(ap.pid):
            if ap.pid_file:
                self.clear_pid(ap.pid_file)
            ap.pid = None
            return True

        logger.info(f"Stopping agent '{name}' (PID {ap.pid})...")
        self.kill_process(ap.pid, force=False)

        # Wait for graceful shutdown
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self.is_process_running(ap.pid):
                break
            time.sleep(0.25)

        # Force kill if still running
        if self.is_process_running(ap.pid):
            logger.warning(f"Agent '{name}' didn't stop gracefully, sending SIGKILL")
            self.kill_process(ap.pid, force=True)
            time.sleep(0.5)

        if ap.pid_file:
            self.clear_pid(ap.pid_file)
        ap.pid = None
        logger.info(f"Agent '{name}' stopped")
        return True

    def start_autostart_agents(
        self,
        config: RookeryConfig,
    ) -> dict[str, AgentProcess]:
        """Start all agents with autostart=True.

        Args:
            config: Rookery configuration.

        Returns:
            Dict of started agent processes.
        """
        started = {}
        for name, agent_cfg in config.get_autostart_agents().items():
            try:
                ap = self.start_agent(name, agent_cfg, config.host.bind)
                started[name] = ap
            except RuntimeError as e:
                logger.error(f"Failed to start agent '{name}': {e}")
        return started

    def stop_all(self, timeout: float = 5.0) -> None:
        """Stop all managed agent processes.

        Args:
            timeout: Seconds to wait per agent for graceful shutdown.
        """
        for name in list(self._agents):
            self.stop_agent(name, timeout=timeout)

    def get_agent_status(self, name: str) -> dict:
        """Get status info for a single agent.

        Returns:
            Dict with name, port, pid, status, data_dir, log_file.
        """
        ap = self._agents.get(name)
        if ap is None:
            return {"name": name, "status": "unknown"}

        running = ap.pid is not None and self.is_process_running(ap.pid)
        return {
            "name": name,
            "port": ap.port,
            "pid": ap.pid if running else None,
            "status": "running" if running else "stopped",
            "data_dir": str(ap.data_dir),
            "log_file": str(ap.log_file) if ap.log_file else None,
        }

    def get_all_status(self) -> dict[str, dict]:
        """Get status info for all managed agents."""
        return {name: self.get_agent_status(name) for name in self._agents}

    def read_logs(self, name: str, lines: int = 50) -> Optional[str]:
        """Read the last N lines of an agent's log file.

        Args:
            name: Agent name.
            lines: Number of lines to read from the end.

        Returns:
            Log text, or None if no log file exists.
        """
        ap = self._agents.get(name)
        if ap is None or ap.log_file is None:
            return None

        if not ap.log_file.exists():
            return None

        # Read all lines and return the last N
        try:
            all_lines = ap.log_file.read_text().splitlines()
            return "\n".join(all_lines[-lines:])
        except OSError:
            return None

    def register_agent(
        self,
        name: str,
        config: LocalAgentConfig,
    ) -> AgentProcess:
        """Register an agent without starting it (for status tracking).

        Checks if it's already running from a previous session via PID file.
        """
        resolved_dir = (self.project_dir / config.data_dir).resolve()
        pid_file = self.agent_pid_file(resolved_dir)
        log_file = self.agent_log_file(resolved_dir)

        existing_pid = self.read_pid(pid_file)
        pid = None
        if existing_pid and self.is_process_running(existing_pid):
            pid = existing_pid

        ap = AgentProcess(
            name=name,
            port=config.port,
            data_dir=resolved_dir,
            pid=pid,
            pid_file=pid_file,
            log_file=log_file,
        )
        self._agents[name] = ap
        return ap
