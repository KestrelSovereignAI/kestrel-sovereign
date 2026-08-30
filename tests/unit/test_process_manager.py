"""
Unit tests for the Kestrel ProcessManager.

Tests process lifecycle, agent registration, status tracking,
and log reading — all without spawning real subprocesses.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from kestrel_sovereign.multi_agent.config import (
    MultiAgentConfig,
    LocalAgentConfig,
    RemoteAgentConfig,
)
from kestrel_sovereign.multi_agent.process_manager import (
    ProcessManager,
    AgentProcess,
    PidStatus,
)
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
        """PID file round-trip: write, read, clear.

        Written for a process that actually exists. A PID file names a running
        process, and ``read_pid`` deliberately withholds a number that names
        nothing — so a made-up integer would exercise the stale path rather
        than the round-trip this is about (#2995).
        """
        pid_file = tmp_path / "test.pid"

        # Not exists
        assert ProcessManager.read_pid(pid_file) is None

        # Write and read
        ProcessManager.write_pid(pid_file, os.getpid())
        assert pid_file.exists()
        assert ProcessManager.read_pid(pid_file) == os.getpid()

        # Clear
        ProcessManager.clear_pid(pid_file)
        assert not pid_file.exists()
        assert ProcessManager.read_pid(pid_file) is None

    def test_a_pid_naming_no_live_process_is_not_handed_back(self, tmp_path):
        """A number that names nothing must not reach a caller.

        Every caller of ``read_pid`` went on to probe or signal what it
        returned, so handing back a PID whose process is gone is how a stale
        file got treated as a running agent — and, after reuse, how an
        unrelated process got signalled (#2987).
        """
        pid_file = tmp_path / "dead.pid"
        dead = subprocess.Popen([sys.executable, "-c", "pass"])
        dead.wait()

        ProcessManager.write_pid(pid_file, dead.pid)
        record = ProcessManager.read_pid_record(pid_file)

        assert record.status is PidStatus.STALE
        assert record.is_running is False
        assert ProcessManager.read_pid(pid_file) is None
        # The file itself is left alone: deciding to remove a record is the
        # caller's call, not the reader's.
        assert pid_file.exists()

    def test_a_recycled_pid_is_not_the_process_that_was_recorded(self, tmp_path):
        """The case no command-line match can catch.

        Two Kestrel checkouts on one machine are identical by argv, and a
        reused number is identical by number. The start instant is the only
        thing that differs, and it is why it is recorded.
        """
        pid_file = tmp_path / "recycled.pid"
        ProcessManager.write_pid(pid_file, os.getpid())

        payload = json.loads(pid_file.read_text())
        assert "started_at" in payload, "the identity field must be recorded"
        # Same number, but the recorded instant belongs to an earlier process.
        payload["started_at"] -= 500
        pid_file.write_text(json.dumps(payload))

        record = ProcessManager.read_pid_record(pid_file)
        assert record.status is PidStatus.STALE
        assert ProcessManager.read_pid(pid_file) is None

    def test_a_legacy_bare_integer_is_undecidable_but_still_counts_as_running(
        self, tmp_path
    ):
        """Files written before #2995 record a number and nothing else.

        Something IS running under it, so calling it stopped would wave a
        guard straight past a live agent; but nothing proves it is ours, so
        calling it LIVE would license signalling it. Undecidable is the honest
        answer, and it is not the same as either.
        """
        pid_file = tmp_path / "legacy.pid"
        pid_file.write_text(str(os.getpid()))

        record = ProcessManager.read_pid_record(pid_file)
        assert record.status is PidStatus.UNDECIDABLE
        assert record.is_running is True
        assert ProcessManager.read_pid(pid_file) == os.getpid()

    def test_a_signal_is_withheld_when_the_pid_changed_hands(self):
        """The gap between reading a PID file and signalling is the hazard.

        After an OOM, a `kill -9` or a reboot the number is free for the OS to
        reuse, and the old path signalled whatever now held it — so
        `kestrel shutdown --force` could SIGKILL an unrelated process (#2987).
        """
        victim = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        time.sleep(0.4)
        try:
            actual = ProcessManager.process_start_time(victim.pid)
            assert actual is not None

            # The identity of a process that held this number earlier.
            sent = ProcessManager.kill_process(
                victim.pid, force=True, started_at=actual - 500
            )
            time.sleep(0.3)
            assert sent is False
            assert ProcessManager.is_process_running(victim.pid), (
                "a process that was never ours was signalled anyway"
            )

            # With the identity it actually has, the signal goes through.
            assert ProcessManager.kill_process(
                victim.pid, force=True, started_at=actual
            ) is True
        finally:
            victim.kill()
            victim.wait()

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="PID 1 is a POSIX convention; Windows uses 0 and 4 and has no PID 1",
    )
    def test_a_process_owned_by_another_user_is_not_reported_dead(self):
        """`os.kill(pid, 0)` raises PermissionError for another user's process.

        Every OSError mapped to False, so a live process read as stopped.
        PID 1 is the deterministic POSIX case: always running, never ours.
        """
        assert ProcessManager.is_process_running(1) is True

    def test_an_unreaped_zombie_is_not_a_live_record(self, tmp_path):
        """`create_time()` still answers for a zombie, but it has exited.

        Classifying it LIVE would put the record and ``is_process_running`` in
        direct disagreement about the same PID, and the status and guard paths
        trust the record — so an exited agent would read as online and block
        maintenance until somebody reaped it.
        """
        if sys.platform == "win32":
            pytest.skip("zombies are a POSIX process state")
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        time.sleep(0.3)
        pid_file = tmp_path / "zombie.pid"
        ProcessManager.write_pid(pid_file, child.pid)

        child.kill()
        time.sleep(0.5)
        # Deliberately not reaped: that is the condition under test.
        try:
            record = ProcessManager.read_pid_record(pid_file)
            assert record.status is PidStatus.STALE
            assert record.is_running is False
            assert ProcessManager.is_process_running(child.pid) is False
        finally:
            child.wait()

    def test_a_file_that_names_no_pid_is_not_reported_running(self, tmp_path):
        """Unreadable is not the same claim as undecidable.

        Undecidable names a PID that IS alive; an empty or malformed file
        establishes no process at all. Reporting the latter as running showed
        an online agent with a PID of ``None``, forced subprocess-mode output,
        and made the guard refuse while ``read_pid`` returned nothing.
        """
        for content in ("", "   ", "not-a-pid", '{"no_pid": true}'):
            pid_file = tmp_path / "bad.pid"
            pid_file.write_text(content)
            record = ProcessManager.read_pid_record(pid_file)
            assert record.status is PidStatus.UNREADABLE, content
            assert record.is_running is False, content
            assert record.pid is None, content
            assert record.needs_cleanup is False, (
                "an unreadable file may name a live host; deleting it would "
                "destroy the only record of it"
            )

    def test_the_recorded_start_time_is_what_reaches_the_caller(self, tmp_path):
        """The record must carry the instant from the FILE.

        A caller that looked the value up again would read whatever holds the
        number now, so a PID reused between the read and the kill would be
        validated against itself and signalled — defeating the check.
        """
        pid_file = tmp_path / "live.pid"
        ProcessManager.write_pid(pid_file, os.getpid())
        recorded = json.loads(pid_file.read_text())["started_at"]

        record = ProcessManager.read_pid_record(pid_file)
        assert record.status is PidStatus.LIVE
        assert record.started_at == pytest.approx(recorded)

    def test_a_process_we_cannot_inspect_is_undecidable_not_stale(self, tmp_path):
        """"Opaque" and "gone" must not collapse into the same answer.

        psutil raises ``AccessDenied`` for a process owned by another account
        and for anything under a Linux ``hidepid`` mount. Reading that as "no
        such process" makes the record STALE, after which start clears it,
        status shows the host offline, and the encryption backfill mutates the
        database it is serving — every one of those fails open.

        ``AccessDenied`` is raised through a patched ``psutil.Process`` rather
        than produced for real: this test runs unprivileged on a machine whose
        kernel lets it read PID 1, so the condition is not reachable here. The
        exception is the one psutil actually raises, and the branch under test
        is the handler for it.
        """
        import psutil

        pid_file = tmp_path / "opaque.pid"
        ProcessManager.write_pid(pid_file, os.getpid())

        class _Denied:
            def __init__(self, pid):
                pass

            def create_time(self):
                raise psutil.AccessDenied(pid=1)

            def status(self):
                raise psutil.AccessDenied(pid=1)

        with patch.object(psutil, "Process", _Denied):
            record = ProcessManager.read_pid_record(pid_file)

        assert record.status is PidStatus.UNDECIDABLE
        assert record.is_running is True, (
            "an inaccessible process must fail closed — a guard that waves "
            "through a database another process may hold costs more than a "
            "refusal the operator can clear"
        )
        assert record.needs_cleanup is False, (
            "clearing the record of a process we merely cannot inspect "
            "destroys the only pointer to a possibly-live host"
        )

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
        # The recorded PID, read straight off the file: 54321 is a mock and
        # names no process, so ``read_pid`` correctly withholds it.
        record = ProcessManager.read_pid_record(pid_file)
        assert record.pid == 54321
        # The identity that makes the record verifiable is recorded with it.
        payload = json.loads(pid_file.read_text())
        assert payload["pid"] == 54321
        assert payload["root"] == str(pm.project_dir)
        assert payload["port"] == cfg.port


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

        # Call sequence: (1) initial check → True, (2) graceful-wait loop → False
        # (break), (3) force-kill check → False (skip), (4) the post-stop
        # verification that decides the return value → False (confirmed gone).
        with patch.object(ProcessManager, "is_process_running",
                          side_effect=[True, False, False, False]), \
             patch.object(ProcessManager, "kill_process") as mock_kill, \
             patch("time.sleep"):
            pm.stop_agent("claw")

        # The identity is carried into the signal so a PID that changed
        # hands since registration is not signalled (#2995).
        mock_kill.assert_called_once_with(99999, force=False, started_at=None)

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
        assert first_call == ((99999,), {"force": False, "started_at": None})
        # Last call: forced (force=True)
        last_call = mock_kill.call_args_list[-1]
        assert last_call == ((99999,), {"force": True, "started_at": None})

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

    def test_stop_agent_that_survives_sigkill_reports_failure(self, pm, project_dir):
        """An agent still running after SIGKILL must not be reported stopped."""
        cfg = LocalAgentConfig(data_dir=Path("agent_data/claw"), port=8801)
        ap = pm.register_agent("claw", cfg)
        ap.pid = 99999
        ProcessManager.write_pid(ap.pid_file, 99999)

        # The process ignores every signal it is sent.
        with patch.object(ProcessManager, "is_process_running", return_value=True), \
             patch.object(ProcessManager, "kill_process"), \
             patch("time.sleep"):
            result = pm.stop_agent("claw", timeout=0.01)

        assert result is False
        # The PID file is the only record of a process that outlived the stop;
        # clearing it would strand a live agent with nothing pointing at it.
        assert ap.pid_file.exists()
        assert ap.pid == 99999

    def test_terminate_all_names_the_agents_that_survived(self, pm, project_dir):
        """terminate_all reports which agents outlived it rather than dropping it."""
        for name, port in (("claw", 8801), ("testbot", 8802)):
            ap = pm.register_agent(name, LocalAgentConfig(
                data_dir=Path(f"agent_data/{name}"), port=port,
            ))
            ap.pid = 99999 if name == "claw" else 99998

        # 'claw' ignores signals; 'testbot' dies on the first check.
        def _running(pid):
            return pid == 99999

        with patch.object(ProcessManager, "is_process_running", side_effect=_running), \
             patch.object(ProcessManager, "kill_process"), \
             patch("time.sleep"):
            survivors = pm.terminate_all(timeout=0.01)

        assert survivors == ["claw"]

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="zombies are a POSIX process state; Windows leaves none to reap",
    )
    def test_a_zombie_child_counts_as_stopped_and_is_reaped(self, pm, project_dir):
        """A killed child this process parented is a zombie until waited on.

        It has already exited, holds no port and can never run again, yet the
        raw ``os.kill(pid, 0)`` syscall still succeeds for it. Treating that as
        liveness made a perfectly successful stop report failure — and keep
        reporting it until the parent exited. ``start_agent`` uses
        ``subprocess.Popen``, so this process really is the parent; a real
        zombie is used because a mocked probe cannot exhibit what is being
        fixed.

        Two layers are asserted, because each covers the other's gap: the
        probe rejects zombie status (which also covers children this process
        did not parent, e.g. under a non-reaping container PID 1), and the
        stop reaps its own children so they do not accumulate.
        """
        import subprocess
        import sys as _sys

        child = subprocess.Popen(
            [_sys.executable, "-c", "import time; time.sleep(300)"]
        )
        child.kill()
        time.sleep(0.5)
        # Deliberately not reaped here — that is the condition under test.

        def _raw_syscall_says_alive() -> bool:
            try:
                os.kill(child.pid, 0)
                return True
            except OSError:
                return False

        assert _raw_syscall_says_alive(), "setup must produce an unreaped zombie"
        assert ProcessManager.is_process_running(child.pid) is False, (
            "the liveness probe must not count a zombie as running"
        )

        cfg = LocalAgentConfig(data_dir=Path("agent_data/claw"), port=8801)
        ap = pm.register_agent("claw", cfg)
        ap.pid = child.pid
        ProcessManager.write_pid(ap.pid_file, child.pid)

        try:
            result = pm.stop_agent("claw", timeout=0.01)
            # Observed BEFORE the cleanup below. Waiting first would reap the
            # child itself and the assertion would be checking this test's own
            # housekeeping rather than the stop's.
            reaped_by_stop = not _raw_syscall_says_alive()
        finally:
            try:
                child.wait(timeout=5)
            except Exception:
                pass

        assert result is True
        assert not ap.pid_file.exists()
        assert reaped_by_stop, (
            "the stop left its own child unreaped; zombies accumulate for the "
            "lifetime of the parent"
        )


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

    def test_terminate_all(self, pm, project_dir):
        """terminate_all terminates all registered agent processes."""
        cfg1 = LocalAgentConfig(
            data_dir=Path("agent_data/claw"), port=8801,
        )
        cfg2 = LocalAgentConfig(
            data_dir=Path("agent_data/testbot"), port=8802,
        )
        pm.register_agent("claw", cfg1)
        pm.register_agent("testbot", cfg2)

        with patch.object(pm, "stop_agent") as mock_stop:
            pm.terminate_all()

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
