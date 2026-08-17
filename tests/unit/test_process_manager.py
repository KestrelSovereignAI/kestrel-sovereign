"""
Unit tests for the Kestrel ProcessManager.

Tests process lifecycle, agent registration, status tracking,
and log reading — all without spawning real subprocesses.
"""

import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from kestrel_sovereign.multi_agent.config import (
    MultiAgentConfig,
    LocalAgentConfig,
    RemoteAgentConfig,
)
from kestrel_sovereign.multi_agent.process_manager import ProcessManager, AgentProcess
from kestrel_sovereign.config import (
    SEMANTIC_CAPABILITIES_CONFIGURED_ENV,
    SEMANTIC_CAPABILITIES_CONFIG_ENV,
    SEMANTIC_INFERENCE_CONFIG_ENV,
)


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
        assert pid_file.exists()
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

    def test_start_agent_sets_agent_bound_env_vars(self, pm, project_dir):
        """Child storage and export roots both bind to this agent."""
        cfg = LocalAgentConfig(
            data_dir=Path("agent_data/claw"), port=8801,
        )

        mock_process = MagicMock()
        mock_process.pid = 12345

        with patch("subprocess.Popen", return_value=mock_process) as mock_popen:
            pm.start_agent("claw", cfg)

        env = mock_popen.call_args[1]["env"]
        expected_root = (project_dir / "agent_data" / "claw").resolve()
        assert env["KESTREL_DB_PATH"] == str(expected_root)
        assert env["KESTREL_IDENTITY_EXPORT_DIR"] == str(expected_root)
        assert "KESTREL_DATA_DIR" not in env
        assert env["PORT"] == "8801"
        assert env["KESTREL_SERVE_UI"] == "false"

    def test_start_agent_passes_per_agent_semantic_inference_profile(
        self,
        pm,
        project_dir,
    ):
        cfg = LocalAgentConfig(
            data_dir=Path("agent_data/claw"),
            port=8801,
            semantic_inference={
                "enabled": True,
                "rdfs_version": "1.0.0",
                "ontology": {
                    "namespace": "kestrel-test",
                    "version": "1",
                    "content_digest": "sha256:test",
                    "compatibility_profile": "semantic-kb-v1",
                },
            },
        )
        mock_process = MagicMock(pid=12345)

        with patch("subprocess.Popen", return_value=mock_process) as mock_popen:
            pm.start_agent("claw", cfg)

        env = mock_popen.call_args.kwargs["env"]
        assert json.loads(env[SEMANTIC_INFERENCE_CONFIG_ENV]) == cfg.semantic_inference

    def test_start_agent_passes_exact_per_agent_semantic_capabilities(
        self,
        pm,
        project_dir,
    ):
        selection = {
            "mode": "experimental",
            "rdf12": {"capability": "rdf-profile:rdf12-test", "version": "0.1.0"},
            "sparql12": {"capability": "query-profile:sparql12-test", "version": "0.1.0"},
            "shacl12": {"capability": "validation-profile:shacl12-test", "version": "0.1.0"},
            "shape_set": {"identifier": "test-shapes", "version": "0.1.0"},
        }
        cfg = LocalAgentConfig(
            data_dir=Path("agent_data/claw"),
            port=8801,
            semantic_capabilities=selection,
        )
        mock_process = MagicMock(pid=12345)

        with patch("subprocess.Popen", return_value=mock_process) as mock_popen:
            pm.start_agent("claw", cfg)

        env = mock_popen.call_args.kwargs["env"]
        assert json.loads(env[SEMANTIC_CAPABILITIES_CONFIG_ENV]) == selection
        assert env[SEMANTIC_CAPABILITIES_CONFIGURED_ENV] == "1"

    def test_start_agent_resolves_explicit_export_override_below_data_dir(
        self,
        pm,
        project_dir,
    ):
        cfg = LocalAgentConfig(
            data_dir=Path("agent_data/claw"),
            identity_export_dir=Path("continuity"),
            port=8801,
        )
        mock_process = MagicMock(pid=12345)

        with patch("subprocess.Popen", return_value=mock_process) as mock_popen:
            pm.start_agent("claw", cfg)

        env = mock_popen.call_args[1]["env"]
        assert env["KESTREL_IDENTITY_EXPORT_DIR"] == str(
            (project_dir / "agent_data" / "claw" / "continuity").resolve()
        )

    def test_two_children_do_not_inherit_one_shared_parent_export_root(
        self,
        pm,
        project_dir,
    ):
        configs = {
            "claw": LocalAgentConfig(
                data_dir=Path("agent_data/claw"),
                port=8801,
            ),
            "testbot": LocalAgentConfig(
                data_dir=Path("agent_data/testbot"),
                port=8802,
            ),
        }
        processes = [MagicMock(pid=12345), MagicMock(pid=12346)]

        with (
            patch.object(
                pm,
                "_load_env",
                return_value={"KESTREL_DATA_DIR": "/shared-parent-root"},
            ),
            patch("subprocess.Popen", side_effect=processes) as mock_popen,
        ):
            for name, config in configs.items():
                pm.start_agent(name, config)

        export_roots = [
            call.kwargs["env"]["KESTREL_IDENTITY_EXPORT_DIR"]
            for call in mock_popen.call_args_list
        ]
        assert export_roots == [
            str((project_dir / "agent_data" / name).resolve())
            for name in configs
        ]
        assert len(set(export_roots)) == 2
        assert {
            call.kwargs["env"]["KESTREL_DATA_DIR"]
            for call in mock_popen.call_args_list
        } == {"/shared-parent-root"}

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


class TestSpawnDetached:
    """``_spawn_detached`` hands the log file's fd straight to the
    child via Popen ``stdout=fd, stderr=STDOUT``, then closes the
    parent's reference. The child inherits the duplicated fd and
    keeps writing to the file even after the launcher exits — which
    is exactly what ``kestrel start`` needs since it's fire-and-exit.

    The existing ``_spawn`` (pipe+pump) writes are LOST after the
    launcher exits because the daemon pump thread dies with its
    parent and the child's stdout pipe gets EPIPE on subsequent
    writes. This bug hid runtime errors after every restart."""

    def test_spawn_detached_passes_log_fd_as_stdout(self, pm, tmp_path):
        """Popen must receive an integer fd for stdout (not PIPE, not
        a file object) so the kernel handles writes directly. ``PIPE``
        was the buggy model — it requires a parent-side reader."""
        import subprocess
        captured = {}

        class _FakeProcess:
            pid = 67890
            stdout = None

            def wait(self, timeout=None):
                return 0

        def _fake_popen(cmd, **kwargs):
            captured["stdout"] = kwargs.get("stdout")
            captured["stderr"] = kwargs.get("stderr")
            captured["env"] = kwargs.get("env")
            return _FakeProcess()

        with patch.object(subprocess, "Popen", side_effect=_fake_popen):
            pm._spawn_detached(
                cmd=["python", "-c", "pass"],
                env={"PATH": "/usr/bin"},
                log_file=tmp_path / "detached.log",
                pid_file=tmp_path / "detached.pid",
            )

        # An OS file descriptor is an int.
        assert isinstance(captured["stdout"], int), (
            f"_spawn_detached must pass an int fd to Popen so the "
            f"kernel writes directly to the log file. Got "
            f"{type(captured['stdout']).__name__}={captured['stdout']!r}."
        )
        assert captured["stderr"] == subprocess.STDOUT, (
            "stderr must merge into stdout so tracebacks land in the "
            "same log file the rest of the runtime writes to."
        )
        # And PYTHONUNBUFFERED is still set (same rationale as _spawn).
        assert captured["env"]["PYTHONUNBUFFERED"] == "1"

    def test_spawn_detached_writes_to_log_file_after_parent_exits(
        self, pm, tmp_path,
    ):
        """End-to-end: spawn a real subprocess that writes a line,
        let our process method return (simulating launcher exit by
        completing the call), then verify the line is on disk.

        Before the fix, the same scenario with ``_spawn`` would race:
        the pump thread is a daemon and dies if the parent exits
        before it consumes the line. With detached fd redirection,
        the child writes the line directly — no thread needed,
        nothing to race."""
        import sys
        import time

        log_file = tmp_path / "real-detached.log"
        pid_file = tmp_path / "real-detached.pid"
        marker = "post-launcher-exit-marker-9c8b2"

        # The child writes a marker then sleeps so we can read while
        # it's still running (mimicking a long-lived uvicorn host).
        cmd = [
            sys.executable, "-c",
            f"import sys, time; "
            f"sys.stdout.write({marker!r}+'\\n'); "
            f"sys.stdout.flush(); "
            f"time.sleep(5)",
        ]
        pid = pm._spawn_detached(
            cmd=cmd, env={"PATH": "/usr/bin"},
            log_file=log_file, pid_file=pid_file,
        )

        # Poll briefly for the marker to land. Don't sleep forever —
        # 2s is plenty for a single print to make it through the
        # kernel write path.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if log_file.exists() and marker in log_file.read_text(
                encoding="utf-8",
            ):
                break
            time.sleep(0.05)
        try:
            assert log_file.exists()
            content = log_file.read_text(encoding="utf-8")
            assert marker in content, (
                f"Detached spawn must persist child stdout to {log_file} "
                f"without a pump thread. Got content: {content!r}"
            )
        finally:
            # Clean up the still-running test child.
            try:
                import os
                import signal
                os.kill(pid, signal.SIGTERM)
            except (ProcessLookupError, OSError):
                pass

    def test_spawn_detached_appends_does_not_truncate(self, pm, tmp_path):
        """Repeated launches must append to the same log file, not
        truncate. Operators rely on host.log accumulating across
        restarts so they can correlate errors across reboots."""
        import subprocess

        log_file = tmp_path / "append.log"
        log_file.write_text("PRIOR-RESTART-CONTENT\n", encoding="utf-8")

        class _FakeProcess:
            pid = 11111
            stdout = None

            def wait(self, timeout=None):
                return 0

        with patch.object(subprocess, "Popen", return_value=_FakeProcess()):
            pm._spawn_detached(
                cmd=["python", "-c", "pass"],
                env={"PATH": "/usr/bin"},
                log_file=log_file,
                pid_file=tmp_path / "append.pid",
            )

        # The file should still carry its prior content; the OS append
        # flag we opened with doesn't truncate.
        content = log_file.read_text(encoding="utf-8")
        assert "PRIOR-RESTART-CONTENT" in content, (
            "Detached spawn must open with O_APPEND so prior content "
            "(another agent's pre-restart logs) is preserved."
        )


# ---------------------------------------------------------------------------
# #2987: a PID outlives the process it names, so signalling one read from a
# file can reach a stranger. These mock the process description rather than
# assume anything about the host's real processes — a test that depends on
# who owns PID 1 fails as root and proves nothing about the implementation.
# ---------------------------------------------------------------------------

def test_kill_process_refuses_a_pid_that_is_provably_not_kestrel():
    """A reused PID must not be signalled, and force must not override it.

    `.host.pid` and `agent.pid` survive an unclean exit — OOM, `kill -9`, a
    host reboot — after which the OS may hand that number to anything.
    `os.kill(pid, 0)` cannot detect it: it proves a PID is allocated, not
    whose it is.
    """
    with patch.object(
        ProcessManager, "describe_process", return_value="/usr/bin/postgres -D /data",
    ), patch("os.kill") as mock_kill:
        assert ProcessManager.kill_process(4321) is False
        assert ProcessManager.kill_process(4321, force=True) is False
    mock_kill.assert_not_called()


def test_kill_process_signals_a_recognisable_kestrel_process():
    """The real host must stay stoppable — a refusal that locks out a
    legitimate stop is its own outage."""
    cmdline = (
        "/opt/kestrel/.venv/bin/python3 -m uvicorn kestrel_sovereign.server:app "
        "--host 0.0.0.0 --port 8888"
    )
    with patch.object(ProcessManager, "describe_process", return_value=cmdline), \
         patch("os.kill") as mock_kill:
        assert ProcessManager.kill_process(4321) is True
    mock_kill.assert_called_once()


def test_kill_process_proceeds_when_identity_is_undeterminable():
    """Undeterminable is NOT the same as "not Kestrel".

    No `ps`, an unsupported platform, or a process that has already exited all
    yield None. Refusing there would make hosts unstoppable on those systems,
    so the guarantee is "never knowingly signal a stranger" — not "only ever
    signal a proven Kestrel".
    """
    with patch.object(ProcessManager, "describe_process", return_value=None), \
         patch("os.kill") as mock_kill:
        assert ProcessManager.kill_process(4321) is True
    mock_kill.assert_called_once()


def test_process_identity_distinguishes_unknown_from_foreign():
    """Three states, because two of them must behave differently."""
    with patch.object(ProcessManager, "describe_process", return_value=None):
        assert ProcessManager.process_is_recognisably_kestrel(1) is None
    with patch.object(ProcessManager, "describe_process", return_value="/sbin/launchd"):
        assert ProcessManager.process_is_recognisably_kestrel(1) is False
    with patch.object(ProcessManager, "describe_process", return_value="kestrel start"):
        assert ProcessManager.process_is_recognisably_kestrel(1) is True


def test_describe_process_returns_none_for_a_pid_that_does_not_exist():
    """Driven against the real `ps`, so the None contract is not just mocked."""
    assert ProcessManager.describe_process(2**22) is None


@pytest.mark.parametrize("cmdline,expected,why", [
    (
        "/opt/kestrel/.venv/bin/python3 -m uvicorn kestrel_sovereign.server:app "
        "--host 0.0.0.0 --port 8888",
        True,
        "the host and per-agent subprocesses both name this target",
    ),
    ("/usr/local/bin/kestrel start", True, "installed console script"),
    ("python -m kestrel_sovereign.cli status", True, "module invocation"),
    (
        "python -m uvicorn other_app:app --port 9000",
        False,
        "an unrelated ASGI service is NOT ours — a bare 'uvicorn' marker would "
        "have handed this stranger to SIGKILL",
    ),
    (
        "/opt/kestrel-tools/bin/postgres -D /data",
        False,
        "living under a kestrel-named path does not make a process ours",
    ),
    ("/sbin/launchd", False, "init is emphatically not ours"),
])
def test_kestrel_identification_is_specific_not_substring(cmdline, expected, why):
    """Identity must key on the Kestrel server/CLI target, not loose fragments.

    Both loose markers were real holes: "uvicorn" matches any ASGI app, and
    "kestrel" matches anything merely installed under a kestrel-named
    directory — including the venv path of a completely unrelated tool.
    """
    with patch.object(ProcessManager, "describe_process", return_value=cmdline):
        assert ProcessManager.process_is_recognisably_kestrel(1) is expected, why


def test_a_refusal_does_not_copy_the_foreign_process_arguments_into_our_logs():
    """A stranger's argv may carry that stranger's credentials.

    The refusal has to identify the process well enough to act on without
    persisting someone else's secrets into Kestrel's logs.
    """
    secretive = "/usr/bin/psql --password=hunter2 -h db.internal"
    with patch.object(ProcessManager, "describe_process", return_value=secretive):
        assert ProcessManager.process_program_name(9) == "psql"

        with patch("os.kill") as mock_kill:
            assert ProcessManager.kill_process(9, force=True) is False
        mock_kill.assert_not_called()
