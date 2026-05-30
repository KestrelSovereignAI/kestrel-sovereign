"""
Unit tests for the Kestrel ProcessManager.

Tests process lifecycle, agent registration, status tracking,
and log reading — all without spawning real subprocesses.
"""

import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import toml

from kestrel_sovereign.multi_agent.config import (
    MultiAgentConfig,
    LocalAgentConfig,
    RemoteAgentConfig,
)
from kestrel_sovereign.multi_agent.process_manager import ProcessManager, AgentProcess


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def project_dir(tmp_path):
    """Create a project directory with agent data directories."""
    # Create agent directories
    claw_dir = tmp_path / "agent_data" / "claw"
    claw_dir.mkdir(parents=True)
    (claw_dir / "kestrel_prime.db").touch()

    testbot_dir = tmp_path / "agent_data" / "testbot"
    testbot_dir.mkdir(parents=True)
    (testbot_dir / "kestrel_prime.db").touch()

    # Create .env file
    (tmp_path / ".env").write_text("KESTREL_API_KEY=test-key\n")

    return tmp_path


@pytest.fixture
def multi_agent_config():
    """Create a multi_agent config with local and remote agents."""
    return MultiAgentConfig(
        agents={
            "claw": LocalAgentConfig(
                data_dir=Path("agent_data/claw"), port=8801, autostart=True,
            ),
            "testbot": LocalAgentConfig(
                data_dir=Path("agent_data/testbot"), port=8802, autostart=False,
            ),
            "remote": RemoteAgentConfig(url="https://example.com"),
        }
    )


@pytest.fixture
def pm(project_dir):
    """Create a ProcessManager for the temp project directory."""
    return ProcessManager(project_dir)


# -----------------------------------------------------------------------
# Constructor tests
# -----------------------------------------------------------------------

class TestProcessManagerInit:
    """Test ProcessManager initialization."""

    def test_init_resolves_project_dir(self, tmp_path):
        """Project dir is stored as resolved absolute path."""
        pm = ProcessManager(tmp_path)
        assert pm.project_dir.is_absolute()
        assert pm.project_dir == tmp_path.resolve()

    def test_init_empty_agents(self, tmp_path):
        """Fresh ProcessManager has no agents."""
        pm = ProcessManager(tmp_path)
        assert pm.agents == {}


# -----------------------------------------------------------------------
# Static helper tests
# -----------------------------------------------------------------------

class TestStaticHelpers:
    """Test static utility methods."""

    def test_is_port_in_use_unused(self):
        """Port 0 (special) should not be in use."""
        assert ProcessManager.is_port_in_use(0) is False

    def test_is_process_running_self(self):
        """Current process should be detected as running."""
        assert ProcessManager.is_process_running(os.getpid()) is True

    def test_is_process_running_dead(self):
        """Non-existent PID should not be running."""
        assert ProcessManager.is_process_running(999999) is False

    def test_read_write_clear_pid(self, tmp_path):
        """PID file round-trip: write, read, clear."""
        pid_file = tmp_path / "test.pid"

        # Not exists
        assert ProcessManager.read_pid(pid_file) is None

        # Write and read
        ProcessManager.write_pid(pid_file, 42)
        assert ProcessManager.read_pid(pid_file) == 42

        # Clear
        ProcessManager.clear_pid(pid_file)
        assert not pid_file.exists()
        assert ProcessManager.read_pid(pid_file) is None

    def test_clear_pid_nonexistent(self, tmp_path):
        """Clearing a non-existent PID file should not raise."""
        ProcessManager.clear_pid(tmp_path / "nope.pid")

    def test_read_pid_invalid_content(self, tmp_path):
        """Invalid PID file content returns None."""
        pid_file = tmp_path / "bad.pid"
        pid_file.write_text("garbage")
        assert ProcessManager.read_pid(pid_file) is None

    def test_agent_pid_file_path(self, tmp_path):
        """Agent PID file is <agent_dir>/agent.pid."""
        assert ProcessManager.agent_pid_file(tmp_path) == tmp_path / "agent.pid"

    def test_agent_log_file_path(self, tmp_path):
        """Agent log file is <agent_dir>/agent.log."""
        assert ProcessManager.agent_log_file(tmp_path) == tmp_path / "agent.log"


# -----------------------------------------------------------------------
# Register agent tests
# -----------------------------------------------------------------------

class TestRegisterAgent:
    """Test agent registration (without starting)."""

    def test_register_agent(self, pm, project_dir):
        """Registering an agent creates an AgentProcess entry."""
        cfg = LocalAgentConfig(
            data_dir=Path("agent_data/claw"), port=8801,
        )
        ap = pm.register_agent("claw", cfg)

        assert ap.name == "claw"
        assert ap.port == 8801
        assert ap.data_dir == (project_dir / "agent_data" / "claw").resolve()
        assert ap.pid is None  # Not running, no PID file
        assert "claw" in pm.agents

    def test_register_agent_detects_existing_process(self, pm, project_dir):
        """If a PID file exists with a running PID, register picks it up."""
        agent_dir = (project_dir / "agent_data" / "claw").resolve()
        pid_file = ProcessManager.agent_pid_file(agent_dir)

        # Write our own PID (which is running)
        ProcessManager.write_pid(pid_file, os.getpid())

        cfg = LocalAgentConfig(
            data_dir=Path("agent_data/claw"), port=8801,
        )
        ap = pm.register_agent("claw", cfg)

        assert ap.pid == os.getpid()

    def test_register_agent_ignores_stale_pid(self, pm, project_dir):
        """If PID file has a dead PID, register returns pid=None."""
        agent_dir = (project_dir / "agent_data" / "claw").resolve()
        pid_file = ProcessManager.agent_pid_file(agent_dir)

        # Write a dead PID
        ProcessManager.write_pid(pid_file, 999999)

        cfg = LocalAgentConfig(
            data_dir=Path("agent_data/claw"), port=8801,
        )
        ap = pm.register_agent("claw", cfg)

        assert ap.pid is None


# -----------------------------------------------------------------------
# Start agent tests
# -----------------------------------------------------------------------

class TestStartAgent:
    """Test starting agent processes (mocked subprocess)."""

    def test_start_agent_spawns_process(self, pm, project_dir):
        """Start spawns a subprocess and records PID."""
        cfg = LocalAgentConfig(
            data_dir=Path("agent_data/claw"), port=8801,
        )

        mock_process = MagicMock()
        mock_process.pid = 12345

        with patch("subprocess.Popen", return_value=mock_process) as mock_popen:
            ap = pm.start_agent("claw", cfg, host_bind="127.0.0.1")

        assert ap.pid == 12345
        assert "claw" in pm.agents
        # Verify command includes the in-package ASGI module reference.
        # `kestrel_sovereign.server:app` (not `server:app`) is what survives a
        # pip install — the wheel doesn't ship a top-level server.py.
        cmd = mock_popen.call_args[0][0]
        assert "kestrel_sovereign.server:app" in cmd
        assert "--port" in cmd
        assert "8801" in cmd

    def test_start_agent_sets_env_vars(self, pm, project_dir):
        """Start sets KESTREL_DB_PATH, PORT, KESTREL_SERVE_UI in env."""
        cfg = LocalAgentConfig(
            data_dir=Path("agent_data/claw"), port=8801,
        )

        mock_process = MagicMock()
        mock_process.pid = 12345

        with patch("subprocess.Popen", return_value=mock_process) as mock_popen:
            pm.start_agent("claw", cfg)

        env = mock_popen.call_args[1]["env"]
        assert "KESTREL_DB_PATH" in env
        assert env["PORT"] == "8801"
        assert env["KESTREL_SERVE_UI"] == "false"

    def test_start_agent_port_in_use_raises(self, pm, project_dir):
        """Start raises RuntimeError if port is already in use."""
        cfg = LocalAgentConfig(
            data_dir=Path("agent_data/claw"), port=8801,
        )

        with patch.object(ProcessManager, "is_port_in_use", return_value=True):
            with pytest.raises(RuntimeError, match="port 8801 already in use"):
                pm.start_agent("claw", cfg)

    def test_start_agent_validation_failure_raises(self, pm, project_dir):
        """Start raises RuntimeError if data_dir validation fails."""
        cfg = LocalAgentConfig(
            data_dir=Path("agent_data/nonexistent"), port=8803,
        )

        with pytest.raises(RuntimeError, match="validation failed"):
            pm.start_agent("bad", cfg)

    def test_start_agent_already_running_returns_existing(self, pm, project_dir):
        """Start returns existing AgentProcess if already running."""
        cfg = LocalAgentConfig(
            data_dir=Path("agent_data/claw"), port=8801,
        )

        # Pre-write a PID file with our own PID (running)
        agent_dir = (project_dir / "agent_data" / "claw").resolve()
        pid_file = ProcessManager.agent_pid_file(agent_dir)
        ProcessManager.write_pid(pid_file, os.getpid())

        # Should not spawn a new process
        with patch("subprocess.Popen") as mock_popen:
            ap = pm.start_agent("claw", cfg)

        mock_popen.assert_not_called()
        assert ap.pid == os.getpid()

    def test_start_agent_writes_pid_file(self, pm, project_dir):
        """Start writes PID to the agent's pid file."""
        cfg = LocalAgentConfig(
            data_dir=Path("agent_data/claw"), port=8801,
        )

        mock_process = MagicMock()
        mock_process.pid = 54321

        with patch("subprocess.Popen", return_value=mock_process):
            pm.start_agent("claw", cfg)

        agent_dir = (project_dir / "agent_data" / "claw").resolve()
        pid_file = ProcessManager.agent_pid_file(agent_dir)
        assert ProcessManager.read_pid(pid_file) == 54321


# -----------------------------------------------------------------------
# Stop agent tests
# -----------------------------------------------------------------------

class TestStopAgent:
    """Test stopping agent processes."""

    def test_stop_unregistered_agent_returns_true(self, pm):
        """Stopping an unregistered agent is a no-op (returns True)."""
        assert pm.stop_agent("nonexistent") is True

    def test_stop_agent_not_running_returns_true(self, pm, project_dir):
        """Stopping a registered but non-running agent returns True."""
        cfg = LocalAgentConfig(
            data_dir=Path("agent_data/claw"), port=8801,
        )
        pm.register_agent("claw", cfg)
        assert pm.stop_agent("claw") is True

    def test_stop_agent_sends_sigterm(self, pm, project_dir):
        """Stopping a running agent sends SIGTERM first."""
        cfg = LocalAgentConfig(
            data_dir=Path("agent_data/claw"), port=8801,
        )
        ap = pm.register_agent("claw", cfg)
        ap.pid = 99999  # Fake running PID

        # Call sequence: (1) line 303 check → True, (2) line 315 loop → False (break),
        # (3) line 320 force-kill check → False (skip)
        with patch.object(ProcessManager, "is_process_running", side_effect=[True, False, False]), \
             patch.object(ProcessManager, "kill_process") as mock_kill, \
             patch("time.sleep"):
            pm.stop_agent("claw")

        mock_kill.assert_called_once_with(99999, force=False)

    def test_stop_agent_escalates_to_sigkill(self, pm, project_dir):
        """If agent doesn't stop gracefully, SIGKILL is sent."""
        cfg = LocalAgentConfig(
            data_dir=Path("agent_data/claw"), port=8801,
        )
        ap = pm.register_agent("claw", cfg)
        ap.pid = 99999

        # Process always reports as running (ignores SIGTERM)
        with patch.object(ProcessManager, "is_process_running", return_value=True), \
             patch.object(ProcessManager, "kill_process") as mock_kill, \
             patch("time.sleep"):
            pm.stop_agent("claw", timeout=0.01)

        # Should have been called at least twice: SIGTERM then SIGKILL
        assert mock_kill.call_count >= 2
        # First call: graceful (force=False)
        first_call = mock_kill.call_args_list[0]
        assert first_call == ((99999,), {"force": False})
        # Last call: forced (force=True)
        last_call = mock_kill.call_args_list[-1]
        assert last_call == ((99999,), {"force": True})

    def test_stop_agent_clears_pid(self, pm, project_dir):
        """Stopping an agent clears its PID file."""
        cfg = LocalAgentConfig(
            data_dir=Path("agent_data/claw"), port=8801,
        )
        ap = pm.register_agent("claw", cfg)
        ap.pid = 99999

        # Write PID file
        ProcessManager.write_pid(ap.pid_file, 99999)

        with patch.object(ProcessManager, "is_process_running", return_value=False):
            pm.stop_agent("claw")

        assert not ap.pid_file.exists()


# -----------------------------------------------------------------------
# Start/stop all tests
# -----------------------------------------------------------------------

class TestStartStopAll:
    """Test bulk start/stop operations."""

    def test_start_autostart_agents(self, pm, project_dir, multi_agent_config):
        """start_autostart_agents starts only autostart=True agents."""
        mock_process = MagicMock()
        mock_process.pid = 11111

        with patch("subprocess.Popen", return_value=mock_process):
            started = pm.start_autostart_agents(multi_agent_config)

        # Only "claw" has autostart=True
        assert "claw" in started
        assert "testbot" not in started

    def test_start_autostart_handles_errors(self, pm, project_dir):
        """start_autostart_agents logs errors but continues."""
        config = MultiAgentConfig(
            agents={
                "bad": LocalAgentConfig(
                    data_dir=Path("agent_data/nonexistent"), port=9901, autostart=True,
                ),
                "claw": LocalAgentConfig(
                    data_dir=Path("agent_data/claw"), port=8801, autostart=True,
                ),
            }
        )

        mock_process = MagicMock()
        mock_process.pid = 22222

        with patch("subprocess.Popen", return_value=mock_process):
            started = pm.start_autostart_agents(config)

        # "bad" fails validation, "claw" succeeds
        assert "bad" not in started
        assert "claw" in started

    def test_stop_all(self, pm, project_dir):
        """stop_all stops all registered agents."""
        cfg1 = LocalAgentConfig(
            data_dir=Path("agent_data/claw"), port=8801,
        )
        cfg2 = LocalAgentConfig(
            data_dir=Path("agent_data/testbot"), port=8802,
        )
        pm.register_agent("claw", cfg1)
        pm.register_agent("testbot", cfg2)

        with patch.object(pm, "stop_agent") as mock_stop:
            pm.stop_all()

        assert mock_stop.call_count == 2


# -----------------------------------------------------------------------
# Status tests
# -----------------------------------------------------------------------

class TestAgentStatus:
    """Test status reporting."""

    def test_status_unknown_agent(self, pm):
        """Unknown agent returns status='unknown'."""
        result = pm.get_agent_status("nonexistent")
        assert result["status"] == "unknown"

    def test_status_stopped_agent(self, pm, project_dir):
        """Registered but not running agent returns status='stopped'."""
        cfg = LocalAgentConfig(
            data_dir=Path("agent_data/claw"), port=8801,
        )
        pm.register_agent("claw", cfg)

        result = pm.get_agent_status("claw")
        assert result["status"] == "stopped"
        assert result["pid"] is None
        assert result["port"] == 8801

    def test_status_running_agent(self, pm, project_dir):
        """Running agent returns status='running' with PID."""
        cfg = LocalAgentConfig(
            data_dir=Path("agent_data/claw"), port=8801,
        )
        ap = pm.register_agent("claw", cfg)
        ap.pid = os.getpid()  # Use our PID so it appears running

        result = pm.get_agent_status("claw")
        assert result["status"] == "running"
        assert result["pid"] == os.getpid()

    def test_get_all_status(self, pm, project_dir):
        """get_all_status returns status for all registered agents."""
        cfg1 = LocalAgentConfig(
            data_dir=Path("agent_data/claw"), port=8801,
        )
        cfg2 = LocalAgentConfig(
            data_dir=Path("agent_data/testbot"), port=8802,
        )
        pm.register_agent("claw", cfg1)
        pm.register_agent("testbot", cfg2)

        result = pm.get_all_status()
        assert "claw" in result
        assert "testbot" in result
        assert result["claw"]["port"] == 8801
        assert result["testbot"]["port"] == 8802


# -----------------------------------------------------------------------
# Log reading tests
# -----------------------------------------------------------------------

class TestReadLogs:
    """Test log file reading."""

    def test_read_logs_no_agent(self, pm):
        """read_logs for unknown agent returns None."""
        assert pm.read_logs("nonexistent") is None

    def test_read_logs_no_file(self, pm, project_dir):
        """read_logs returns None when log file doesn't exist."""
        cfg = LocalAgentConfig(
            data_dir=Path("agent_data/claw"), port=8801,
        )
        pm.register_agent("claw", cfg)
        assert pm.read_logs("claw") is None

    def test_read_logs_with_content(self, pm, project_dir):
        """read_logs returns last N lines of log file."""
        cfg = LocalAgentConfig(
            data_dir=Path("agent_data/claw"), port=8801,
        )
        ap = pm.register_agent("claw", cfg)

        # Write some log lines
        log_lines = [f"log line {i}" for i in range(100)]
        ap.log_file.write_text("\n".join(log_lines))

        result = pm.read_logs("claw", lines=10)
        lines = result.split("\n")
        assert len(lines) == 10
        assert "log line 99" in lines[-1]

    def test_read_logs_fewer_than_requested(self, pm, project_dir):
        """read_logs returns all lines when file has fewer than requested."""
        cfg = LocalAgentConfig(
            data_dir=Path("agent_data/claw"), port=8801,
        )
        ap = pm.register_agent("claw", cfg)

        ap.log_file.write_text("line 1\nline 2\nline 3")

        result = pm.read_logs("claw", lines=100)
        assert result.count("\n") == 2  # 3 lines, 2 newlines


# -----------------------------------------------------------------------
# AgentProcess dataclass tests
# -----------------------------------------------------------------------

class TestAgentProcess:
    """Test AgentProcess dataclass."""

    def test_defaults(self):
        """AgentProcess has sensible defaults."""
        ap = AgentProcess(name="test", port=8801, data_dir=Path("/tmp/test"))
        assert ap.pid is None
        assert ap.pid_file is None
        assert ap.log_file is None

    def test_all_fields(self, tmp_path):
        """AgentProcess stores all fields."""
        ap = AgentProcess(
            name="claw",
            port=8801,
            data_dir=tmp_path,
            pid=12345,
            pid_file=tmp_path / "agent.pid",
            log_file=tmp_path / "agent.log",
        )
        assert ap.name == "claw"
        assert ap.port == 8801
        assert ap.pid == 12345


# -----------------------------------------------------------------------
# Stdout pump (issue #812 — agent subprocess output → host stdout)
# -----------------------------------------------------------------------


class TestPumpStdout:
    """Verify the tee daemon thread mirrors subprocess output to file + stdout."""

    def test_pump_writes_to_log_and_stdout(self, tmp_path, capfd):
        """A real ``echo`` subprocess: every output line lands in the log file
        AND in the parent's stdout (prefixed)."""
        import subprocess
        import sys
        log_file = tmp_path / "agent.log"
        marker = "PUMP_TEST_MARKER_42"
        process = subprocess.Popen(
            [sys.executable, "-c", f"print('{marker}'); print('second-line')"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        # Run pump synchronously (not the daemon thread) so we can assert deterministically
        ProcessManager._pump_stdout(process, log_file, "[agent:test] ")
        process.wait(timeout=5)

        # File contains both lines verbatim
        log_text = log_file.read_text(encoding="utf-8")
        assert marker in log_text
        assert "second-line" in log_text

        # Parent stdout (captured by capfd) shows both lines, prefixed
        captured = capfd.readouterr()
        assert f"[agent:test] {marker}" in captured.out
        assert "[agent:test] second-line" in captured.out

    def test_pump_survives_log_write_failure(self, tmp_path, capfd):
        """If the log file path can't be opened, the pump warns and exits;
        it doesn't propagate the exception (which would silently kill the
        background thread for all future output)."""
        import subprocess
        import sys
        log_file = tmp_path / "does" / "not" / "exist" / "agent.log"  # parent dirs missing
        process = subprocess.Popen(
            [sys.executable, "-c", "print('payload')"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
        )
        # Should not raise — pump catches the open() failure and logs a warning
        ProcessManager._pump_stdout(process, log_file, "[agent:bad] ")
        process.wait(timeout=5)


class TestSpawnUnbuffered:
    """``ProcessManager._spawn`` must force ``PYTHONUNBUFFERED=1`` in the
    child env so the parent's pump thread sees runtime log lines
    immediately. Without this, the host's Python child detects its
    stdout is a pipe (not a TTY) and switches to block-buffered
    mode (~4 KiB), which holds sparse runtime WARN/ERROR lines until
    the buffer fills — on a long-running uvicorn that buffer never
    fills, so host.log appears to "freeze" after the chatty startup
    and runtime errors silently vanish. This is the root cause of
    the observability blackout that hid Emma's memory_feature
    failures."""

    def test_spawn_sets_pythonunbuffered_when_unset(self, pm, tmp_path):
        """The default env passed to _spawn has no PYTHONUNBUFFERED.
        The spawn helper must inject it so the child Python is
        line-flush instead of 4KiB-block-flush."""
        import subprocess
        captured = {}

        class _FakeProcess:
            pid = 12345
            stdout = None

            def wait(self, timeout=None):
                return 0

        def _fake_popen(cmd, **kwargs):
            captured["env"] = kwargs.get("env")
            return _FakeProcess()

        with patch.object(subprocess, "Popen", side_effect=_fake_popen):
            # Also patch the pump thread so we don't try to read from None.
            with patch("threading.Thread"):
                pm._spawn(
                    cmd=["python", "-c", "pass"],
                    env={"PATH": "/usr/bin"},  # NO PYTHONUNBUFFERED set
                    log_file=tmp_path / "spawn.log",
                    pid_file=tmp_path / "spawn.pid",
                    agent_name="probe",
                )

        env = captured["env"]
        assert env is not None
        assert env.get("PYTHONUNBUFFERED") == "1", (
            f"_spawn must force PYTHONUNBUFFERED=1 in the child env. "
            f"Without it, runtime log lines block-buffer in the child's "
            f"stdout and never reach host.log after the startup phase. "
            f"Got PYTHONUNBUFFERED={env.get('PYTHONUNBUFFERED')!r}."
        )

    def test_spawn_does_not_mutate_caller_env(self, pm, tmp_path):
        """The caller's env dict must not gain a PYTHONUNBUFFERED key
        as a side effect — that would surprise callers passing their
        own carefully-curated env (e.g. integration tests, CI pipes
        verifying buffered behavior). The spawn helper must copy."""
        import subprocess

        class _FakeProcess:
            pid = 12345
            stdout = None

            def wait(self, timeout=None):
                return 0

        caller_env = {"PATH": "/usr/bin"}
        with patch.object(subprocess, "Popen", return_value=_FakeProcess()):
            with patch("threading.Thread"):
                pm._spawn(
                    cmd=["python", "-c", "pass"],
                    env=caller_env,
                    log_file=tmp_path / "spawn.log",
                    pid_file=tmp_path / "spawn.pid",
                )

        assert "PYTHONUNBUFFERED" not in caller_env, (
            "_spawn must copy the caller's env before mutating; "
            "mutating in place would leak the override back into a "
            "shared dict held by tests or higher-level config code."
        )

    def test_spawn_respects_caller_pythonunbuffered_override(self, pm, tmp_path):
        """If the caller explicitly sets PYTHONUNBUFFERED themselves
        (e.g. PYTHONUNBUFFERED=0 to debug buffer behavior), don't
        overwrite it. ``setdefault`` semantics preserve caller intent."""
        import subprocess
        captured = {}

        class _FakeProcess:
            pid = 12345
            stdout = None

            def wait(self, timeout=None):
                return 0

        def _fake_popen(cmd, **kwargs):
            captured["env"] = kwargs.get("env")
            return _FakeProcess()

        with patch.object(subprocess, "Popen", side_effect=_fake_popen):
            with patch("threading.Thread"):
                pm._spawn(
                    cmd=["python", "-c", "pass"],
                    env={"PATH": "/usr/bin", "PYTHONUNBUFFERED": "0"},
                    log_file=tmp_path / "spawn.log",
                    pid_file=tmp_path / "spawn.pid",
                )

        assert captured["env"]["PYTHONUNBUFFERED"] == "0", (
            "An explicit caller override must survive — _spawn uses "
            "setdefault, not assignment."
        )
