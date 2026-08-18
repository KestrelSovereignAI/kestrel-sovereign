"""
Agent Process Manager - Starts, stops, and monitors agent subprocesses.

The ProcessManager is used by both the CLI (`kestrel start/stop`) and the
host (`host.py` lifespan + API endpoints) to manage agent processes.

Each agent runs as a separate uvicorn process on its own port:
    KESTREL_DB_PATH=agent_data/<agent> PORT=8801 uvicorn kestrel_sovereign.server:app

Note that ``KESTREL_DB_PATH`` scopes an agent only in this separate-process
mode. The in-process host runs every agent in ONE process, so per-agent state
must be bound by parameter there rather than by environment (see #2769).

Remote agents (url-only in config) are ignored — they're not managed by
this host.
"""

import json
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from kestrel_sovereign.multi_agent.config import (
    LocalAgentConfig,
    MultiAgentConfig,
)
from kestrel_sovereign.config import (
    SEMANTIC_CAPABILITIES_CONFIGURED_ENV,
    SEMANTIC_CAPABILITIES_CONFIG_ENV,
    SEMANTIC_INFERENCE_CONFIG_ENV,
    SEMANTIC_MAINTENANCE_CONFIG_ENV,
    SEMANTIC_MAINTENANCE_CONFIGURED_ENV,
)

# NOTE: ``IDENTITY_EXPORT_DIR_ENV`` is imported lazily inside ``start_agent``
# (its only use). Pulling it in at module scope would force ``identity``'s
# package ``__init__`` to eager-load the identity exporter → the whole agent
# cognition/LLM stack, so importing this module's lightweight cross-platform
# port/process helpers (e.g. ``find_pids_on_port``, used by phoenix custody
# discovery that must never raise) would transitively depend on that entire
# import graph resolving cleanly. See #2690.

logger = logging.getLogger(__name__)

# Feature discovery, provider catalog loading, and embedding-route validation
# can legitimately take longer than the old 30-second lifecycle window. Keep
# one default for every CLI start/restart/update path; callers may still pass a
# shorter or longer operator-selected timeout explicitly.
DEFAULT_STARTUP_HEALTH_TIMEOUT_SECONDS = 120.0


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
        pm.start_all(multi_agent_config)
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
    def is_port_in_use(port: int, host: str = "0.0.0.0") -> bool:
        """Whether a server of ours could NOT bind ``host:port`` right now.

        Every caller is asking one operational question — can the server take
        this address — so the probe performs the same bind the server will,
        under the same options, at the same address.

        Not ``connect_ex``, which asks "is something accepting connections"
        and is wrong in the direction that matters: a listener whose accept
        backlog is full REFUSES new connections while still owning the port.
        Measured — such a listener probes True, then False after a single
        unaccepted connection, while ``bind`` keeps reporting it taken.

        ``SO_REUSEADDR`` is set because uvicorn's asyncio server sets it
        (verified on the live loop), and without it a port left in
        ``TIME_WAIT`` by a clean shutdown reads as occupied. Kestrel would
        then stop the service and refuse to start it again — `restart` and
        `update` both abort on a failed stop — over a port uvicorn could have
        bound immediately. Measured: TIME_WAIT binds fine with the option, and
        a live listener still fails with ``EADDRINUSE``, so nothing is lost.

        ``host`` matters and must be the address the server is configured to
        bind. Measured both ways: a listener on ``0.0.0.0`` is invisible to a
        probe of ``127.0.0.1``, and a listener on ``127.0.0.1`` is invisible
        to a probe of ``0.0.0.0``. Probing where the server will actually bind
        makes this succeed exactly when the server's own bind would.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, port))
            except OSError:
                return True
        return False

    @staticmethod
    def find_pids_on_port(port: int) -> list[int]:
        """Return PIDs of processes currently listening on `port`.

        Used by `kestrel stop` to reap orphans — servers whose PID file was
        lost or never written, which would otherwise block a subsequent start.
        Returns an empty list on any failure (tool missing, parse error).
        """
        try:
            import psutil
            pids: set[int] = set()
            for c in psutil.net_connections(kind="inet"):
                if (
                    c.status == psutil.CONN_LISTEN
                    and c.laddr
                    and c.laddr.port == port
                    and c.pid is not None
                ):
                    pids.add(c.pid)
            return sorted(pids)
        except (ImportError, Exception):
            pass

        if sys.platform == "win32":
            try:
                out = subprocess.run(
                    ["netstat", "-ano", "-p", "TCP"],
                    capture_output=True, text=True, timeout=5,
                )
                pids: set[int] = set()
                for line in out.stdout.splitlines():
                    parts = line.split()
                    if len(parts) >= 5 and parts[-1].isdigit() and f":{port}" in parts[1] and parts[3].upper() == "LISTENING":
                        pids.add(int(parts[-1]))
                return sorted(pids)
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                return []
        try:
            out = subprocess.run(
                ["lsof", "-iTCP:%d" % port, "-sTCP:LISTEN", "-t", "-Pn"],
                capture_output=True, text=True, timeout=5,
            )
            return sorted({int(p) for p in out.stdout.split() if p.isdigit()})
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return []

    @staticmethod
    def is_process_running(pid: int) -> bool:
        """Check if a process with the given PID is alive and can still run.

        A zombie is excluded. It has already exited, holds no port and can
        never run again, but ``os.kill(pid, 0)`` succeeds for it — so treating
        PID existence as liveness makes a completed stop look like a failure
        for as long as nobody reaps the child. ``_reap_if_child`` handles the
        children this process parented; this covers the ones it did not, such
        as a container PID 1 that does not reap.
        """
        try:
            import psutil

            try:
                if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                    return False
            except psutil.NoSuchProcess:
                return False
            except psutil.AccessDenied:
                # Visible but not inspectable: it exists, which is the only
                # claim being made here.
                return True
        except ImportError:
            pass  # Fall through to the syscall probe below.

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
    def wait_for_health(
        port: int,
        timeout: float = DEFAULT_STARTUP_HEALTH_TIMEOUT_SECONDS,
    ) -> bool:
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
        """Load .env file into an env dict copy.

        The body moved to ``paths.spawned_agent_env`` so ``kestrel doctor`` can
        ask this launcher which database a spawned agent will open instead of
        reimplementing the precedence and quietly disagreeing with it (#2892).
        """
        from kestrel_sovereign.paths import spawned_agent_env

        return spawned_agent_env(self.project_dir)

    def _spawn_detached(
        self,
        cmd: list[str],
        env: dict,
        log_file: Path,
        pid_file: Path,
    ) -> int:
        """Spawn a process whose stdout/stderr go DIRECTLY to ``log_file``
        via inherited file descriptors. No pump thread.

        Use this when the parent (the launcher, e.g. ``kestrel start``)
        EXITS immediately after spawning — the in-process host pattern
        for local dev. The pipe+pump model in ``_spawn`` requires the
        parent to keep running so its daemon pump thread can survive;
        when the launcher exits, that thread dies and the child's
        future stdout writes hit a closed pipe → EPIPE → silently
        swallowed. Runtime log lines (INFO from request handlers,
        ERROR + traceback from exception paths) get LOST.

        With direct fd redirection, the kernel writes the child's
        stdout/stderr straight to ``log_file``. The parent's file
        handle gets closed once Popen has duped it into the child;
        when the parent exits, the child's inherited fd remains valid
        and writes continue without interruption.

        Forces ``PYTHONUNBUFFERED=1`` (see ``_spawn`` for the
        block-buffering rationale).
        """
        log_file.parent.mkdir(parents=True, exist_ok=True)
        child_env = dict(env)
        child_env.setdefault("PYTHONUNBUFFERED", "1")

        # Open the file with the OS-level append + create flags so two
        # spawns can't truncate each other and the file exists before
        # we hand the fd to the child. ``buffering=0`` because we hand
        # the raw fd to Popen — Python's own buffering layer doesn't
        # matter here.
        log_fd = os.open(
            log_file,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o644,
        )
        try:
            kwargs = dict(
                cwd=self.project_dir,
                env=child_env,
                stdout=log_fd,
                stderr=subprocess.STDOUT,
                close_fds=True,
            )
            if sys.platform == "win32":
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                kwargs["start_new_session"] = True
            process = subprocess.Popen(cmd, **kwargs)
        finally:
            # Popen has duped our fd into the child's stdout; close
            # our reference so the parent can exit cleanly. The child's
            # inherited copy is what keeps writes flowing.
            os.close(log_fd)

        self.write_pid(pid_file, process.pid)
        return process.pid

    def _spawn(
        self,
        cmd: list[str],
        env: dict,
        log_file: Path,
        pid_file: Path,
        agent_name: Optional[str] = None,
    ) -> int:
        """Spawn a background process. Returns PID.

        Subprocess stdout + stderr are tee'd to two destinations by a
        background daemon thread:
          1. ``log_file`` on disk (preserves the per-agent debug log inside
             the container — same behavior as before).
          2. The parent process's stdout, prefixed with ``[agent:<name>] ``.
             This is what Cloud Run / Cloud Logging captures, so
             ``logger.info`` calls inside agents become visible alongside
             host logs. See issue #812.

        The daemon thread exits naturally when the subprocess closes its
        stdout (i.e. on exit). It uses ``daemon=True`` so a stuck pump can
        never block the host from shutting down.
        """
        log_file.parent.mkdir(parents=True, exist_ok=True)
        # Force unbuffered stdout in the CHILD process so the pump sees
        # runtime log lines (single-line WARNINGs, exception tracebacks)
        # immediately. Without this, Python detects its stdout isn't a
        # TTY and switches to BLOCK-buffered mode (~4 KiB), which means
        # sparse runtime INFO/WARNING/ERROR lines sit in the child's
        # libc buffer until either the buffer fills or the child exits.
        # On a long-running uvicorn host that bufsize is never met, so
        # host.log appears to "stop" after the chatty startup phase
        # fills the buffer and runtime errors silently vanish. The
        # parent's pump already has ``bufsize=1`` for line-buffered
        # reads on its side — that handles the pipe READ; this env var
        # handles the WRITE.
        child_env = dict(env)
        child_env.setdefault("PYTHONUNBUFFERED", "1")
        kwargs = dict(
            cwd=self.project_dir,
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,  # line-buffered (text mode)
        )
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        process = subprocess.Popen(cmd, **kwargs)

        # Tee the pipe to file + parent stdout in a daemon thread.
        prefix = f"[agent:{agent_name}] " if agent_name else "[agent] "
        thread = threading.Thread(
            target=self._pump_stdout,
            args=(process, log_file, prefix),
            daemon=True,
            name=f"stdout-pump-{agent_name or process.pid}",
        )
        thread.start()

        self.write_pid(pid_file, process.pid)
        return process.pid

    @staticmethod
    def _pump_stdout(
        process: "subprocess.Popen[str]",
        log_file: Path,
        prefix: str,
    ) -> None:
        """Tee ``process.stdout`` to ``log_file`` and the parent's stdout.

        Runs on its own daemon thread until the subprocess closes its pipe.
        Catches and logs any exception so a misbehaving agent's output can
        never crash the pump (which would silently break logging on disk
        going forward).

        Per-line failures on either sink are caught so one broken sink
        (e.g. disk full) cannot starve the other. Each sink emits a single
        warning the first time it fails, then stays silent — otherwise a
        persistent failure would either spam the log or vanish entirely.
        """
        tag = prefix.strip()
        log_warned = False
        stdout_warned = False
        try:
            with open(log_file, "a", encoding="utf-8") as log:
                assert process.stdout is not None  # text=True + PIPE ⇒ TextIO
                for line in process.stdout:
                    try:
                        log.write(line)
                        log.flush()
                    except Exception as exc:
                        if not log_warned:
                            logger.warning(
                                "stdout pump %s: file sink failed (suppressing further warnings): %s",
                                tag, exc,
                            )
                            log_warned = True
                    try:
                        sys.stdout.write(prefix + line)
                        sys.stdout.flush()
                    except Exception as exc:
                        if not stdout_warned:
                            logger.warning(
                                "stdout pump %s: parent stdout sink failed (suppressing further warnings): %s",
                                tag, exc,
                            )
                            stdout_warned = True
        except Exception as exc:
            logger.warning("stdout pump for %s ended unexpectedly: %s", tag, exc)

    # ------------------------------------------------------------------
    # Agent start / stop
    # ------------------------------------------------------------------

    def start_agent(
        self,
        name: str,
        config: LocalAgentConfig,
        host_bind: str = "0.0.0.0",
        host_port: int = 8888,
        standalone: bool = False,
    ) -> AgentProcess:
        """Start a single agent process.

        Args:
            name: Agent name (from multi_agent config).
            config: The agent's LocalAgentConfig.
            host_bind: Interface to bind to.
            host_port: Host port (for inter-agent communication).
            standalone: If True, agent serves its own UI (solo mode).

        Returns:
            AgentProcess with pid set on success.

        Raises:
            RuntimeError: If agent can't be started (port in use, already
                          running, validation error).
        """
        resolved_dir = config.resolve_data_dir(self.project_dir)
        identity_export_dir = config.resolve_identity_export_dir(self.project_dir)
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
        if self.is_port_in_use(config.port, host_bind):
            raise RuntimeError(
                f"Agent '{name}': port {config.port} already in use"
            )

        # Runtime validation
        errors = config.validate_runtime(base_dir=self.project_dir)
        if errors:
            raise RuntimeError(
                f"Agent '{name}' validation failed: {errors[0]}"
            )

        # Imported lazily (not at module scope) so the lightweight port/process
        # helpers stay importable without the agent stack — see the module-top
        # note and #2690.
        from kestrel_sovereign.identity.protected_export import (
            IDENTITY_EXPORT_DIR_ENV,
        )

        # Build env
        env = self._load_env()
        env["KESTREL_DB_PATH"] = str(resolved_dir)
        # A parent-process KESTREL_DATA_DIR is not a per-agent setting. Carry
        # the resolved custody root in a dedicated child-only variable so
        # unrelated KESTREL_DATA_DIR consumers retain their existing meaning.
        env[IDENTITY_EXPORT_DIR_ENV] = str(identity_export_dir or resolved_dir)
        env["PORT"] = str(config.port)
        env["KESTREL_SERVE_UI"] = "true" if standalone else "false"
        env["KESTREL_HOST_URL"] = f"http://localhost:{host_port}"
        # Unlike an in-process AgentManager, this process handoff cannot pass
        # constructor arguments. Carry the exact per-agent config rather than
        # asking the child to discover a project-global profile. Remove an
        # inherited value when this agent has no configured profile so one
        # sibling's approval cannot cross a process/tenant boundary.
        if config.semantic_inference is None:
            env.pop(SEMANTIC_INFERENCE_CONFIG_ENV, None)
        else:
            try:
                env[SEMANTIC_INFERENCE_CONFIG_ENV] = json.dumps(
                    config.semantic_inference,
                    sort_keys=True,
                )
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Agent '{name}' has a non-serializable semantic inference profile"
                ) from exc
        if config.semantic_maintenance is None:
            env.pop(SEMANTIC_MAINTENANCE_CONFIG_ENV, None)
            env.pop(SEMANTIC_MAINTENANCE_CONFIGURED_ENV, None)
        else:
            try:
                env[SEMANTIC_MAINTENANCE_CONFIG_ENV] = json.dumps(
                    config.semantic_maintenance,
                    sort_keys=True,
                )
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Agent '{name}' has a non-serializable semantic maintenance budget"
                ) from exc
            env[SEMANTIC_MAINTENANCE_CONFIGURED_ENV] = "1"
        if config.semantic_capabilities is None:
            env.pop(SEMANTIC_CAPABILITIES_CONFIG_ENV, None)
            env.pop(SEMANTIC_CAPABILITIES_CONFIGURED_ENV, None)
        else:
            try:
                env[SEMANTIC_CAPABILITIES_CONFIG_ENV] = json.dumps(
                    config.semantic_capabilities,
                    sort_keys=True,
                )
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Agent '{name}' has a non-serializable semantic capability selection"
                ) from exc
            env[SEMANTIC_CAPABILITIES_CONFIGURED_ENV] = "1"

        # Force child agents into single-agent mode. Without this,
        # server.py detects multi_agent.toml in the CWD and enters multi-
        # agent mode, leaving app.state.agent = None → 503 on all endpoints.
        env["KESTREL_MULTI_AGENT_CONFIG"] = "__none__"

        # Per-agent data key: KESTREL_DATA_KEY_CLAW overrides KESTREL_DATA_KEY.
        # The resolution lives in ``paths.spawned_agent_data_key`` so an offline
        # tool can ask which key this agent really starts with instead of
        # reading the pre-override value out of the environment (#2892).
        from kestrel_sovereign.paths import spawned_agent_data_key

        agent_key_var = f"KESTREL_DATA_KEY_{name.upper()}"
        effective_key = spawned_agent_data_key(env, name)
        if effective_key and effective_key != env.get("KESTREL_DATA_KEY"):
            env["KESTREL_DATA_KEY"] = effective_key
            logger.info(f"Agent '{name}' using per-agent data key from {agent_key_var}")

        cmd = [
            sys.executable, "-m", "uvicorn", "kestrel_sovereign.server:app",
            "--host", host_bind, "--port", str(config.port),
        ]

        pid = self._spawn(cmd, env, log_file, pid_file, agent_name=name)
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
            True only once the agent is confirmed gone (or was never running).
            False if it is still running after SIGKILL.

        A stop is reported on the strength of the postcondition, never on the
        strength of having sent a signal: ``kill_process`` cannot signal a
        process owned by another user at all, and a delivered signal is not
        proof it was honoured. On failure the PID file is deliberately left in
        place — it is the only record of a process that outlived the stop, and
        clearing it would strand a live agent with nothing pointing at it.

        Note:
            ``is_process_running`` reports False for a live process owned by
            another user (#2995), so True here is only as strong as that probe.
            Callers that know the agent's port should confirm the port was
            released as well.
        """
        ap = self._agents.get(name)
        if ap is None:
            return True

        # Reap before asking anything about the process. An agent that exited
        # on its own is a zombie until this process — its parent, via
        # ``subprocess.Popen`` — collects it, and a stop that returns early
        # because the child is already gone must still not leave that corpse
        # behind for the lifetime of the host.
        if ap.pid is not None:
            self._reap_if_child(ap.pid)

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

        # Reap before judging. ``start_agent`` spawns agents with
        # ``subprocess.Popen``, so this process is their parent, and a killed
        # child that nobody waits on stays a ZOMBIE — for which
        # ``os.kill(pid, 0)`` succeeds. Without this, a stop that worked
        # perfectly would report failure, and keep reporting it until the
        # parent happened to exit. Measured: state 'Z', probe says alive.
        self._reap_if_child(ap.pid)

        if self.is_process_running(ap.pid):
            logger.error(
                f"Agent '{name}' (PID {ap.pid}) is still running after SIGKILL; "
                "leaving the PID file in place"
            )
            return False

        if ap.pid_file:
            self.clear_pid(ap.pid_file)
        ap.pid = None
        logger.info(f"Agent '{name}' stopped")
        return True

    def start_autostart_agents(
        self,
        config: MultiAgentConfig,
    ) -> dict[str, AgentProcess]:
        """Start all agents with autostart=True.

        Args:
            config: MultiAgent configuration.

        Returns:
            Dict of started agent processes.
        """
        started = {}
        for name, agent_cfg in config.get_autostart_agents().items():
            try:
                ap = self.start_agent(name, agent_cfg, config.host.bind, config.host.port)
                started[name] = ap
            except RuntimeError as e:
                logger.error(f"Failed to start agent '{name}': {e}")
        return started

    @staticmethod
    def _reap_if_child(pid: int) -> None:
        """Collect `pid`'s exit status if this process is its parent.

        Best effort by design. ``ChildProcessError`` means it is not our
        child — an agent started by an earlier CLI invocation, say — and
        there is nothing to reap; the kernel's own init already handles the
        orphan case. ``waitpid`` is unavailable on Windows, where a killed
        process leaves no zombie to begin with.
        """
        if sys.platform == "win32":
            return
        try:
            os.waitpid(pid, os.WNOHANG)
        except (ChildProcessError, OSError):
            pass

    def stop_all(self, timeout: float = 5.0) -> list[str]:
        """Stop all managed agent processes.

        Args:
            timeout: Seconds to wait per agent for graceful shutdown.

        Returns:
            The names of agents still running after SIGKILL, in the order they
            were attempted. Empty when every agent stopped. Returned rather
            than discarded so a caller cannot report a clean shutdown over the
            top of an agent that outlived it.
        """
        still_running = []
        for name in list(self._agents):
            if not self.stop_agent(name, timeout=timeout):
                still_running.append(name)
        return still_running

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
