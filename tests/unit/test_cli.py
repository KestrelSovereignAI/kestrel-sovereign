"""
Unit tests for the unified kestrel CLI.

Tests argument parsing, command dispatch, and individual command handlers.
"""

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import toml

from kestrel_sovereign.cli import (
    build_parser,
    main,
    cmd_start,
    cmd_terminate,
    cmd_status,
    cmd_logs,
    cmd_list,
    cmd_create,
    cmd_shell,
    cmd_ask,
    cmd_health,
    cmd_storage,
    _agent_http_timeout,
    _DEFAULT_ASK_READ_TIMEOUT,
)
from kestrel_sovereign.cli_lifecycle import (
    _REANCHOR_RUNBOOK,
    _diagnose_unready_server,
    _fetch_detailed_health,
    _reap_orphans_on_port,
    _start_inprocess_mode,
    PortReapResult,
)
from kestrel_sovereign.multi_agent.config import MultiAgentConfig
from kestrel_sovereign.multi_agent.process_manager import (
    DEFAULT_STARTUP_HEALTH_TIMEOUT_SECONDS,
    PidRecord,
    PidStatus,
    ProcessManager,
)


class TestAgentHttpTimeout:
    """Read timeout for talking to a running agent is configurable; connect
    stays fast so a dead server fails immediately (replaces the old hardcoded
    flat 600s that aborted long agentic/Codex turns)."""

    def test_default_read_timeout(self, monkeypatch):
        monkeypatch.delenv("KESTREL_ASK_TIMEOUT_SECONDS", raising=False)
        t = _agent_http_timeout()
        assert t.read == _DEFAULT_ASK_READ_TIMEOUT
        assert t.connect == 10.0  # fast-fail on a dead server

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("KESTREL_ASK_TIMEOUT_SECONDS", "120")
        assert _agent_http_timeout().read == 120.0

    def test_explicit_override_beats_env(self, monkeypatch):
        monkeypatch.setenv("KESTREL_ASK_TIMEOUT_SECONDS", "120")
        assert _agent_http_timeout(45.0).read == 45.0

    def test_non_positive_disables_read_timeout(self, monkeypatch):
        monkeypatch.delenv("KESTREL_ASK_TIMEOUT_SECONDS", raising=False)
        assert _agent_http_timeout(0).read is None

    def test_invalid_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("KESTREL_ASK_TIMEOUT_SECONDS", "not-a-number")
        assert _agent_http_timeout().read == _DEFAULT_ASK_READ_TIMEOUT

    def test_ask_timeout_flag_parses(self):
        args = build_parser().parse_args(["ask", "kite", "hi", "--timeout", "90"])
        assert args.timeout == 90.0

    def test_ask_timeout_flag_defaults_none(self):
        args = build_parser().parse_args(["ask", "kite", "hi"])
        assert args.timeout is None
# -----------------------------------------------------------------------
# Argument parsing tests
# -----------------------------------------------------------------------

class TestArgumentParsing:
    """Test CLI argument parsing."""

    def test_no_command_returns_1(self):
        """No command should print help and return 1."""
        with patch("sys.argv", ["kestrel"]):
            assert main() == 1

    def test_help_no_topic_prints_top_level_help(self, capsys):
        """'help' with no topic should print the top-level help and return 0."""
        with patch("sys.argv", ["kestrel", "help"]):
            assert main() == 0
        assert "Kestrel Sovereign Agent Manager" in capsys.readouterr().out

    def test_help_with_topic_prints_subcommand_help(self, capsys):
        """'help <command>' should print that subcommand's own help."""
        with patch("sys.argv", ["kestrel", "help", "feature"]):
            assert main() == 0
        assert "usage: kestrel feature" in capsys.readouterr().out

    def test_help_with_unknown_topic_returns_1(self, capsys):
        """'help <bogus>' should say so and fall back to top-level help."""
        with patch("sys.argv", ["kestrel", "help", "not-a-real-command"]):
            assert main() == 1
        out = capsys.readouterr().out
        assert "no such command 'not-a-real-command'" in out
        assert "Kestrel Sovereign Agent Manager" in out

    def test_version(self, capsys):
        """--version should print version and exit."""
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--version"])
        assert exc_info.value.code == 0

    def test_start_no_name(self):
        """'start' with no name should parse successfully."""
        parser = build_parser()
        args = parser.parse_args(["start"])
        assert args.command == "start"
        assert args.name is None

    def test_start_with_name(self):
        """'start <name>' should parse agent name."""
        parser = build_parser()
        args = parser.parse_args(["start", "claw"])
        assert args.command == "start"
        assert args.name == "claw"

    @pytest.mark.parametrize(
        "command",
        [
            ["start", "claw"],
            ["restart", "claw"],
            ["update", "claw"],
        ],
    )
    def test_lifecycle_startup_timeout_default(self, command):
        args = build_parser().parse_args(command)
        assert args.startup_timeout == DEFAULT_STARTUP_HEALTH_TIMEOUT_SECONDS

    @pytest.mark.parametrize(
        "command",
        [
            ["start", "claw"],
            ["restart", "claw"],
            ["update", "claw"],
        ],
    )
    def test_lifecycle_startup_timeout_override(self, command):
        args = build_parser().parse_args(command + ["--startup-timeout", "75.5"])
        assert args.startup_timeout == 75.5

    @pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "not-a-number"])
    def test_lifecycle_startup_timeout_rejects_invalid_values(self, value):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["start", "claw", "--startup-timeout", value])

    def test_terminate_no_name(self):
        """'terminate' with no name should parse successfully."""
        parser = build_parser()
        args = parser.parse_args(["terminate"])
        assert args.command == "terminate"
        assert args.name is None

    def test_terminate_with_name(self):
        """'terminate <name>' should parse agent name."""
        parser = build_parser()
        args = parser.parse_args(["terminate", "claw"])
        assert args.command == "terminate"
        assert args.name == "claw"

    def test_status(self):
        """'status' should parse successfully."""
        parser = build_parser()
        args = parser.parse_args(["status"])
        assert args.command == "status"

    def test_logs_with_name(self):
        """'logs <name>' should parse agent name."""
        parser = build_parser()
        args = parser.parse_args(["logs", "claw"])
        assert args.command == "logs"
        assert args.name == "claw"
        assert args.lines == 50  # default
        assert args.follow is False  # default

    def test_logs_with_follow(self):
        """'logs <name> -f' should set follow flag."""
        parser = build_parser()
        args = parser.parse_args(["logs", "host", "-f", "-n", "100"])
        assert args.command == "logs"
        assert args.name == "host"
        assert args.follow is True
        assert args.lines == 100

    def test_list(self):
        """'list' should parse successfully."""
        parser = build_parser()
        args = parser.parse_args(["list"])
        assert args.command == "list"

    def test_create_with_name(self):
        """'create <name>' should parse agent name."""
        parser = build_parser()
        args = parser.parse_args(["create", "myagent"])
        assert args.command == "create"
        assert args.name == "myagent"
        assert args.port is None  # default

    def test_create_with_port(self):
        """'create <name> --port 9000' should parse port."""
        parser = build_parser()
        args = parser.parse_args(["create", "myagent", "--port", "9000"])
        assert args.command == "create"
        assert args.name == "myagent"
        assert args.port == 9000

    def test_create_test_flag_defaults_false(self):
        """'create <name>' without --test parses test=False."""
        parser = build_parser()
        args = parser.parse_args(["create", "myagent"])
        assert args.test is False

    def test_create_with_test_flag(self):
        """'create <name> --test' parses test=True."""
        parser = build_parser()
        args = parser.parse_args(["create", "myagent", "--test"])
        assert args.command == "create"
        assert args.name == "myagent"
        assert args.test is True

    def test_cmd_create_passes_test_instance_through(self, tmp_path):
        """cmd_create must forward --test to create_agent(is_test_instance=...)."""
        captured = {}

        def fake_create_agent(**kwargs):
            captured.update(kwargs)
            return MagicMock(did="did:test", port=8777, already_existed=False)

        args = MagicMock(name="myagent", port=None, test=True)
        args.name = "myagent"
        with patch("kestrel_sovereign.cli._get_project_dir", return_value=tmp_path), \
             patch("kestrel_sovereign.setup.steps.agent.create_agent", fake_create_agent):
            rc = cmd_create(args)

        assert rc == 0
        assert captured["is_test_instance"] is True

    def test_cmd_create_default_is_not_test_instance(self, tmp_path):
        """Without --test, cmd_create forwards is_test_instance=False."""
        captured = {}

        def fake_create_agent(**kwargs):
            captured.update(kwargs)
            return MagicMock(did="did:test", port=8777, already_existed=False)

        args = MagicMock(port=None, test=False)
        args.name = "myagent"
        with patch("kestrel_sovereign.cli._get_project_dir", return_value=tmp_path), \
             patch("kestrel_sovereign.setup.steps.agent.create_agent", fake_create_agent):
            rc = cmd_create(args)

        assert rc == 0
        assert captured["is_test_instance"] is False

    def test_shell_with_name(self):
        """'shell <name>' should parse agent name."""
        parser = build_parser()
        args = parser.parse_args(["shell", "claw"])
        assert args.command == "shell"
        assert args.name == "claw"

    def test_shell_with_app(self):
        """'shell <name> --app elderly' should parse app extension."""
        parser = build_parser()
        args = parser.parse_args(["shell", "claw", "--app", "elderly"])
        assert args.command == "shell"
        assert args.name == "claw"
        assert args.app == "elderly"

    def test_health(self):
        """'health' should parse successfully."""
        parser = build_parser()
        args = parser.parse_args(["health"])
        assert args.command == "health"

    def test_storage_health(self):
        """'storage health' should parse successfully."""
        parser = build_parser()
        args = parser.parse_args(
            [
                "storage",
                "health",
                "--agent-id",
                "did:example:agent",
                "--lighthouse-grace-hours",
                "48",
                "--json",
            ]
        )
        assert args.command == "storage"
        assert args.storage_command == "health"
        assert args.agent_id == "did:example:agent"
        assert args.lighthouse_grace_hours == 48
        assert args.json is True

    def test_config_with_dir(self):
        """'config <dir>' should parse agent directory."""
        parser = build_parser()
        args = parser.parse_args(["config", "./agent_data/claw"])
        assert args.command == "config"
        assert args.agent_dir == "./agent_data/claw"

    def test_config_with_options(self):
        """'config <dir> --set-port --set-name' should parse options."""
        parser = build_parser()
        args = parser.parse_args(["config", "./agent_data/claw",
                                  "--set-port", "9000",
                                  "--set-name", "Claw"])
        assert args.command == "config"
        assert args.set_port == 9000
        assert args.set_name == "Claw"

    def test_config_init(self):
        """'config --init' should set init flag."""
        parser = build_parser()
        args = parser.parse_args(["config", "--init"])
        assert args.init is True


# -----------------------------------------------------------------------
# Command dispatch tests
# -----------------------------------------------------------------------

class TestCommandDispatch:
    """Test that commands dispatch to the right handlers."""

    def test_dispatch_start(self):
        """'start' should dispatch to cmd_start."""
        with patch("sys.argv", ["kestrel", "start"]), \
             patch("kestrel_sovereign.cli.cmd_start", return_value=0) as mock:
            main()
            mock.assert_called_once()

    def test_dispatch_terminate(self):
        """'terminate' should dispatch to cmd_terminate."""
        with patch("sys.argv", ["kestrel", "terminate"]), \
             patch("kestrel_sovereign.cli.cmd_terminate", return_value=0) as mock:
            main()
            mock.assert_called_once()

    def test_process_termination_uses_reserved_terminate_command(self):
        """Process signals may not remain reachable through `kestrel stop`."""

        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["stop"])
        with patch("sys.argv", ["kestrel", "terminate"]), patch(
            "kestrel_sovereign.cli.cmd_terminate",
            return_value=0,
            create=True,
        ) as terminate:
            main()
        terminate.assert_called_once()

    def test_dispatch_status(self):
        """'status' should dispatch to cmd_status."""
        with patch("sys.argv", ["kestrel", "status"]), \
             patch("kestrel_sovereign.cli.cmd_status", return_value=0) as mock:
            main()
            mock.assert_called_once()

    def test_dispatch_logs(self):
        """'logs' should dispatch to cmd_logs."""
        with patch("sys.argv", ["kestrel", "logs", "host"]), \
             patch("kestrel_sovereign.cli.cmd_logs", return_value=0) as mock:
            main()
            mock.assert_called_once()

    def test_dispatch_list(self):
        """'list' should dispatch to cmd_list."""
        with patch("sys.argv", ["kestrel", "list"]), \
             patch("kestrel_sovereign.cli.cmd_list", return_value=0) as mock:
            main()
            mock.assert_called_once()

    def test_dispatch_create(self):
        """'create' should dispatch to cmd_create."""
        with patch("sys.argv", ["kestrel", "create", "myagent"]), \
             patch("kestrel_sovereign.cli.cmd_create", return_value=0) as mock:
            main()
            mock.assert_called_once()

    def test_dispatch_shell(self):
        """'shell' should dispatch to cmd_shell."""
        with patch("sys.argv", ["kestrel", "shell", "claw"]), \
             patch("kestrel_sovereign.cli.cmd_shell", return_value=0) as mock:
            main()
            mock.assert_called_once()

    def test_dispatch_health(self):
        """'health' should dispatch to cmd_health."""
        with patch("sys.argv", ["kestrel", "health"]), \
             patch("kestrel_sovereign.cli.cmd_health", return_value=0) as mock:
            main()
            mock.assert_called_once()

    def test_dispatch_storage(self):
        """'storage' should dispatch to cmd_storage."""
        with patch("sys.argv", ["kestrel", "storage", "health"]), \
             patch("kestrel_sovereign.cli.cmd_storage", return_value=0) as mock:
            main()
            mock.assert_called_once()

    def test_dispatch_config(self):
        """'config' should dispatch to cmd_config."""
        with patch("sys.argv", ["kestrel", "config"]), \
             patch("kestrel_sovereign.cli.cmd_config", return_value=0) as mock:
            main()
            mock.assert_called_once()


# -----------------------------------------------------------------------
# MultiAgent fixture
# -----------------------------------------------------------------------

@pytest.fixture
def multi_agent_env(tmp_path):
    """Set up a temporary multi_agent environment with config and agent dirs."""
    # Create agent directories
    claw_dir = tmp_path / "agent_data" / "claw"
    claw_dir.mkdir(parents=True)
    (claw_dir / "kestrel_prime.db").touch()

    testbot_dir = tmp_path / "agent_data" / "testbot"
    testbot_dir.mkdir(parents=True)
    (testbot_dir / "kestrel_prime.db").touch()

    # Create multi_agent.toml
    config = {
        "host": {"port": 18888, "bind": "127.0.0.1"},
        "agents": {
            "claw": {
                "data_dir": "agent_data/claw",
                "port": 18801,
                "autostart": True,
            },
            "testbot": {
                "data_dir": "agent_data/testbot",
                "port": 18802,
                "autostart": False,
            },
        },
    }
    config_path = tmp_path / "multi_agent.toml"
    with open(config_path, "w") as f:
        toml.dump(config, f)

    # Create logs directory
    (tmp_path / "logs").mkdir(exist_ok=True)

    # Stub .env so the first-run setup hook in cmd_start does not fire.
    # cmd_start tests assume a configured project; the first-run path is
    # exercised separately in test_cli_first_run.py.
    #
    # Persist the same key the suite-wide autouse fixture
    # (`_born_hybrid_inception_env`) exports into os.environ. cmd_create's
    # pre-inception custody guard (#2468) refuses an exported⇄persisted
    # KESTREL_DATA_KEY split-brain, so a mismatched stub here would (correctly)
    # block agent creation. Matching them keeps this home's custody coherent.
    (tmp_path / ".env").write_text(
        "KESTREL_DATA_KEY=test-master-key-for-encryption-32chars!\n"
    )

    return tmp_path


# -----------------------------------------------------------------------
# cmd_list tests
# -----------------------------------------------------------------------

class TestCmdList:
    """Tests for the 'list' command."""

    def test_list_agents(self, multi_agent_env, capsys):
        """List should show agents from multi_agent config."""
        parser = build_parser()
        args = parser.parse_args(["list"])

        with patch("kestrel_sovereign.cli._get_project_dir", return_value=multi_agent_env):
            rc = cmd_list(args)

        assert rc == 0
        output = capsys.readouterr().out
        assert "claw" in output
        assert "testbot" in output
        assert "18801" in output
        assert "18802" in output

    def test_list_empty(self, tmp_path, capsys):
        """List with no agents should show empty message."""
        # Create empty multi_agent.toml
        config_path = tmp_path / "multi_agent.toml"
        with open(config_path, "w") as f:
            toml.dump({"host": {}, "agents": {}}, f)

        parser = build_parser()
        args = parser.parse_args(["list"])

        with patch("kestrel_sovereign.cli._get_project_dir", return_value=tmp_path):
            rc = cmd_list(args)

        assert rc == 0
        output = capsys.readouterr().out
        assert "No agents configured" in output


# -----------------------------------------------------------------------
# cmd_status tests
# -----------------------------------------------------------------------

class TestCmdStatus:
    """Tests for the 'status' command."""

    def test_status_all_offline(self, multi_agent_env, capsys):
        """Status should show all processes as offline when nothing runs."""
        parser = build_parser()
        args = parser.parse_args(["status"])

        with patch("kestrel_sovereign.cli._get_project_dir", return_value=multi_agent_env):
            rc = cmd_status(args)

        assert rc == 0
        output = capsys.readouterr().out
        # In-process mode (no agent PIDs) shows "server" instead of "host"
        assert "server" in output or "host" in output
        assert "claw" in output
        assert "testbot" in output
        assert "offline" in output

    def test_status_shows_header(self, multi_agent_env, capsys):
        """Status should show column headers."""
        parser = build_parser()
        args = parser.parse_args(["status"])

        with patch("kestrel_sovereign.cli._get_project_dir", return_value=multi_agent_env):
            cmd_status(args)

        output = capsys.readouterr().out
        assert "NAME" in output
        assert "STATUS" in output
        assert "PID" in output


# -----------------------------------------------------------------------
# cmd_storage tests
# -----------------------------------------------------------------------

class TestCmdStorage:
    """Tests for the 'storage' command group."""

    def test_storage_no_subcommand_prints_usage(self, capsys):
        parser = build_parser()
        args = parser.parse_args(["storage"])

        rc = cmd_storage(args)

        assert rc == 1
        assert "kestrel storage" in capsys.readouterr().out


# -----------------------------------------------------------------------
# cmd_terminate tests
# -----------------------------------------------------------------------

class TestCmdTerminate:
    """Tests for the 'terminate' command."""

    def test_stop_single_agent_not_found(self, multi_agent_env, capsys):
        """Stop should return 1 if agent name not found."""
        parser = build_parser()
        args = parser.parse_args(["terminate", "nonexistent"])

        with patch("kestrel_sovereign.cli._get_project_dir", return_value=multi_agent_env):
            rc = cmd_terminate(args)

        assert rc == 1
        output = capsys.readouterr().out
        assert "not found" in output

    def test_stop_single_not_running(self, multi_agent_env, capsys):
        """Stop should succeed even if agent is not running."""
        parser = build_parser()
        args = parser.parse_args(["terminate", "claw"])

        with patch("kestrel_sovereign.cli._get_project_dir", return_value=multi_agent_env):
            rc = cmd_terminate(args)

        assert rc == 0

    def test_stop_all(self, multi_agent_env, capsys):
        """Stop all should stop agents and host."""
        parser = build_parser()
        args = parser.parse_args(["terminate"])

        with patch("kestrel_sovereign.cli._get_project_dir", return_value=multi_agent_env):
            rc = cmd_terminate(args)

        assert rc == 0
        output = capsys.readouterr().out
        assert "MultiAgent terminated" in output


# -----------------------------------------------------------------------
# cmd_terminate postcondition tests (#2990)
#
# The verdict in every test below is decided by a REAL listening socket, so
# ``is_port_in_use`` does the same connect() it does in production. Only the
# kill is stubbed — ``find_pids_on_port`` on a socket this process owns would
# return the test runner's own PID.
# -----------------------------------------------------------------------

@contextmanager
def _listener_on(*, discovered_pids=(999_999,)):
    """Hold a real TCP listener on a kernel-assigned port, kill path stubbed.

    Binds :0 and yields the port actually assigned. CI runs the unit suite
    under ``pytest -n auto``, so fixed ports would have workers binding each
    other's sockets; taking whatever the kernel hands out and never releasing
    it makes the allocation race-free rather than merely unlikely.

    The kill stubbing lives in here, not at each call site, and that is
    deliberate: this process owns the listener, so the real
    ``find_pids_on_port`` returns pytest's OWN pid and ``kill_process`` would
    then SIGTERM the test runner — which is exactly what happened when one
    test took a listener without the guard.

    Pass ``discovered_pids=()`` to model a listener whose owner cannot be
    discovered at all (another user's process, or psutil missing).
    """
    import socket as _socket

    sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    # Nothing accepts these connections, and nothing needs to: the port probe
    # binds rather than connects, so no accept backlog is ever involved. A
    # listener left deliberately unattended is also the case that broke the
    # old connect-based probe.
    try:
        with patch.object(
            ProcessManager, "find_pids_on_port", return_value=list(discovered_pids)
        ), patch.object(ProcessManager, "kill_process", return_value=False), \
             patch("kestrel_sovereign.cli_lifecycle.time.sleep"):
            yield sock.getsockname()[1]
    finally:
        sock.close()


@contextmanager
def _unkillable_orphan(pid: int = 999_999):
    """Pretend `pid` listens on every probed port and ignores every signal.

    For cases with no real listener bound, where the port genuinely is free
    and the question is what the code concludes from that.
    """
    with patch.object(ProcessManager, "find_pids_on_port", return_value=[pid]), \
         patch.object(ProcessManager, "kill_process", return_value=False), \
         patch("kestrel_sovereign.cli_lifecycle.time.sleep"):
        yield


def _free_port() -> int:
    """A port nothing is listening on."""
    import socket as _socket

    with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _repoint_ports(env, **ports: int) -> None:
    """Point the fixture's config at ports this test actually owns."""
    cfg_path = env / "multi_agent.toml"
    cfg = toml.load(cfg_path)
    for name, port in ports.items():
        if name == "host":
            cfg["host"]["port"] = port
        else:
            cfg["agents"][name]["port"] = port
    with open(cfg_path, "w") as fh:
        toml.dump(cfg, fh)


class TestCmdTerminateReportsOnlyVerifiedTermination:
    """`terminate` succeeds only when the port was actually released."""

    def test_a_held_port_reads_as_held_however_full_its_backlog(self):
        """The probe must answer "can this be bound", not "is it accepting".

        A listener with a full accept backlog refuses new connections while
        still owning the port. The old ``connect_ex`` probe therefore reported
        a held port as free after a single unaccepted connection — which let
        `terminate` report success over exactly the listener it was checking.
        """
        with _listener_on() as port:
            assert [
                ProcessManager.is_port_in_use(port, "127.0.0.1") for _ in range(3)
            ] == [
                True, True, True
            ]

    def test_a_port_left_in_time_wait_reads_as_free(self):
        """A clean shutdown must not look like a listener that refused to die.

        uvicorn's asyncio server sets SO_REUSEADDR (verified on a live loop),
        so Kestrel can rebind a TIME_WAIT port immediately. A probe without
        that option calls it occupied — and since `restart` and `update` both
        abort on a failed termination, they would terminate the service and
        then refuse to start it again.
        """
        import socket as _socket

        srv = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        srv.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", 0))
        port = srv.getsockname()[1]
        srv.listen(5)
        client = _socket.create_connection(("127.0.0.1", port))
        conn, _ = srv.accept()
        # The side that closes first is the side that lingers in TIME_WAIT.
        conn.close()
        client.close()
        srv.close()

        assert ProcessManager.is_port_in_use(port, "127.0.0.1") is False

    def test_a_live_listener_still_reads_as_held(self):
        """The reuse option must not blind the probe to a real listener."""
        with _listener_on() as port:
            assert ProcessManager.is_port_in_use(port, "127.0.0.1") is True

    def test_an_ipv6_bind_address_is_probed_in_its_own_family(self):
        """`host.bind` may be an IPv6 address, and uvicorn serves it fine.

        Forcing AF_INET made the bind raise, which read as "port held" — so
        start, stop and restart all failed on a configuration that works.
        """
        import socket as _socket

        live = _socket.socket(_socket.AF_INET6, _socket.SOCK_STREAM)
        live.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
        live.bind(("::1", 0))
        held_port = live.getsockname()[1]
        live.listen(5)
        try:
            assert ProcessManager.is_port_in_use(held_port, "::1") is True
        finally:
            live.close()

        spare = _socket.socket(_socket.AF_INET6, _socket.SOCK_STREAM)
        spare.bind(("::1", 0))
        free_port = spare.getsockname()[1]
        spare.close()
        assert ProcessManager.is_port_in_use(free_port, "::1") is False

    def test_an_ipv4_listener_does_not_make_the_ipv6_wildcard_look_taken(self):
        """asyncio sets IPV6_V6ONLY before uvicorn binds, so the probe must too.

        Linux defaults an AF_INET6 socket to dual-stack. Without the option a
        probe of ``::`` collides with any unrelated IPv4 listener on the same
        number, and restart/update abort over an IPv6 address uvicorn could
        have bound immediately.
        """
        import socket as _socket

        four = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        four.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
        four.bind(("0.0.0.0", 0))
        port = four.getsockname()[1]
        four.listen(5)
        try:
            assert ProcessManager.is_port_in_use(port, "::") is False
            # ...while the family that IS taken still reports taken.
            assert ProcessManager.is_port_in_use(port, "0.0.0.0") is True
        finally:
            four.close()

        # The behavioural assertion above can only fail where AF_INET6
        # defaults to dual-stack, which is Linux — macOS and the BSDs already
        # default to v6-only, so it would pass there however the probe was
        # written. Since CI is Linux this is a real check there, but a test
        # that cannot fail on the machine it was written on is not one to
        # trust, so the option itself is observed too.
        options: list[tuple] = []
        real_socket = _socket.socket

        # The options are recorded as they are set, not read back afterwards:
        # the probe closes its socket before returning, and getsockopt on a
        # closed descriptor raises EBADF.
        class _RecordingSocket(real_socket):
            def setsockopt(self, level, optname, value, *rest):
                options.append((level, optname, value))
                return super().setsockopt(level, optname, value, *rest)

        families: list[int] = []

        def _make(family, *args, **kwargs):
            families.append(family)
            return _RecordingSocket(family, *args, **kwargs)

        with patch.object(_socket, "socket", _make):
            ProcessManager.is_port_in_use(_free_port(), "::1")

        assert _socket.AF_INET6 in families, "no IPv6 socket for an IPv6 bind"
        assert (
            _socket.IPPROTO_IPV6,
            _socket.IPV6_V6ONLY,
            1,
        ) in options, "the probe did not mirror asyncio's IPV6_V6ONLY"

    def test_an_address_this_host_cannot_assign_is_not_occupancy(self):
        """asyncio skips candidates it cannot use and binds the rest.

        Treating EADDRNOTAVAIL as "taken" rejects a start uvicorn would have
        completed, and reports an incomplete shutdown on a usable port.
        """
        # TEST-NET-1: valid, resolvable, and never assigned to this host.
        assert ProcessManager.is_port_in_use(9998, "192.0.2.1") is False

    def test_an_unresolvable_bind_address_is_not_called_occupied(self):
        """A typo is a configuration fault, not another process holding a port.

        Claiming occupancy would wedge `stop` into permanent failure; the bind
        error surfaces at `start`, where it names the address.
        """
        assert ProcessManager.is_port_in_use(
            9999, "not.a.real.host.invalid"
        ) is False

    def test_reap_returns_still_held_when_port_survives_sigkill(self):
        with _listener_on() as port:
            result = _reap_orphans_on_port(port, "claw", force=False, bind="127.0.0.1")
        assert result is PortReapResult.STILL_HELD

    def test_reap_returns_released_when_port_is_free_after_signalling(self):
        with _unkillable_orphan():
            result = _reap_orphans_on_port(
                _free_port(), "claw", force=False, bind="127.0.0.1"
            )
        assert result is PortReapResult.RELEASED

    def test_reap_returns_nothing_found_when_no_listener(self):
        with patch.object(ProcessManager, "find_pids_on_port", return_value=[]):
            result = _reap_orphans_on_port(
                _free_port(), "claw", force=False, bind="127.0.0.1"
            )
        assert result is PortReapResult.NOTHING_FOUND

    def test_undiscoverable_listener_is_not_reported_as_absent(self):
        """`find_pids_on_port` returns [] on ANY discovery failure.

        A listener owned by another user, or a missing psutil, yields an empty
        list while the port stays bound — exactly the unkillable listener this
        change exists to catch. Concluding "nothing here" from an empty PID
        list without asking the port would reintroduce the bug at the
        discovery step.
        """
        with _listener_on(discovered_pids=()) as port:
            result = _reap_orphans_on_port(port, "claw", force=False, bind="127.0.0.1")
        assert result is PortReapResult.STILL_HELD

    def test_stop_all_fails_when_host_port_stays_held(self, multi_agent_env, capsys):
        """A host port nobody could free must not read as a clean shutdown."""
        parser = build_parser()
        args = parser.parse_args(["terminate"])

        with _listener_on() as port:
            _repoint_ports(multi_agent_env, host=port, claw=_free_port(),
                           testbot=_free_port())
            with patch("kestrel_sovereign.cli._get_project_dir",
                       return_value=multi_agent_env):
                rc = cmd_terminate(args)

        output = capsys.readouterr().out
        assert rc == 1
        assert "MultiAgent terminated" not in output
        assert "termination incomplete" in output
        assert "host" in output
        assert f"port :{port}" in output

    def test_stop_single_agent_fails_when_its_port_stays_held(
        self, multi_agent_env, capsys
    ):
        parser = build_parser()
        args = parser.parse_args(["terminate", "claw"])

        with _listener_on() as port:
            _repoint_ports(multi_agent_env, claw=port)
            with patch("kestrel_sovereign.cli._get_project_dir",
                       return_value=multi_agent_env):
                rc = cmd_terminate(args)

        output = capsys.readouterr().out
        assert rc == 1
        assert "claw terminated" not in output
        assert "still in use" in output

    def test_stop_single_agent_reports_orphan_stop_when_port_is_released(
        self, multi_agent_env, capsys
    ):
        """The success path still succeeds — nothing is bound to claw's port."""
        parser = build_parser()
        args = parser.parse_args(["terminate", "claw"])
        _repoint_ports(multi_agent_env, claw=_free_port())

        with _unkillable_orphan(), \
             patch("kestrel_sovereign.cli._get_project_dir", return_value=multi_agent_env):
            rc = cmd_terminate(args)

        output = capsys.readouterr().out
        assert rc == 0
        assert "claw terminated (orphan)" in output

    def test_tracked_agent_with_a_still_bound_port_is_not_reported_stopped(
        self, multi_agent_env, capsys
    ):
        """The tracked PID being gone does not mean the port was released.

        A supervisor may already have rebound it under a new PID, and the next
        `kestrel start` fails on a port this command called free.
        """
        parser = build_parser()
        args = parser.parse_args(["terminate", "claw"])

        with _listener_on() as port:
            _repoint_ports(multi_agent_env, claw=port)
            with patch("kestrel_sovereign.cli._get_project_dir",
                       return_value=multi_agent_env), \
                 patch.object(ProcessManager, "terminate_agent", return_value=True), \
                 patch.object(ProcessManager, "read_pid", return_value=4242), \
                 patch.object(ProcessManager, "is_process_running", return_value=True):
                rc = cmd_terminate(args)

        output = capsys.readouterr().out
        assert rc == 1, output
        assert "still in use" in output

    def test_dead_host_pid_is_cleared_even_when_the_port_stays_held(
        self, multi_agent_env, capsys
    ):
        """A stale PID record is worse than none.

        The PID file is worth keeping only while it names something real. Once
        that process is gone the number can be reused, and the next lifecycle
        command would signal an unrelated process — so it is cleared on its
        own facts, independently of who holds the port.
        """
        import subprocess as _subprocess
        import sys as _sys

        from kestrel_sovereign.cli_lifecycle import _host_pid_file

        # A real process that really exits, so the record is genuinely stale
        # rather than stale by assertion. Nothing about liveness is stubbed
        # here: the verified read establishes it from the process table.
        dead = _subprocess.Popen([_sys.executable, "-c", "pass"])
        dead.wait()

        pid_file = _host_pid_file(multi_agent_env)
        ProcessManager.write_pid(pid_file, dead.pid)
        assert pid_file.exists()
        assert ProcessManager.read_pid_record(pid_file).status is PidStatus.STALE

        parser = build_parser()
        args = parser.parse_args(["terminate"])

        with _listener_on() as port:
            _repoint_ports(multi_agent_env, host=port, claw=_free_port(),
                           testbot=_free_port())
            with patch("kestrel_sovereign.cli._get_project_dir",
                       return_value=multi_agent_env):
                rc = cmd_terminate(args)

        output = capsys.readouterr().out
        assert rc == 1, output
        assert "termination incomplete" in output
        assert not pid_file.exists(), (
            "a dead host's PID file was kept because the port was still held; "
            "the number can be reused and signalled by the next command"
        )


# -----------------------------------------------------------------------
# cmd_start tests
# -----------------------------------------------------------------------

class TestCmdStart:
    """Tests for the 'start' command."""

    def test_start_single_not_found(self, multi_agent_env, capsys):
        """Start should return 1 if agent name not found."""
        parser = build_parser()
        args = parser.parse_args(["start", "nonexistent"])

        with patch("kestrel_sovereign.cli._get_project_dir", return_value=multi_agent_env):
            rc = cmd_start(args)

        assert rc == 1
        output = capsys.readouterr().out
        assert "not found" in output

    def test_start_single_missing_db(self, multi_agent_env, capsys):
        """Start should fail if agent data dir has no database."""
        # Remove the database file
        db_file = multi_agent_env / "agent_data" / "claw" / "kestrel_prime.db"
        db_file.unlink()

        parser = build_parser()
        args = parser.parse_args(["start", "claw"])

        with patch("kestrel_sovereign.cli._get_project_dir", return_value=multi_agent_env):
            rc = cmd_start(args)

        assert rc == 1

    def test_start_single_uses_timeout_and_reports_log(
        self, multi_agent_env, capsys
    ):
        """A slow agent gets the requested deadline and an actionable error."""
        args = build_parser().parse_args(
            ["start", "claw", "--startup-timeout", "42"]
        )

        with patch(
            "kestrel_sovereign.cli._get_project_dir",
            return_value=multi_agent_env,
        ), patch.object(ProcessManager, "start_agent"), patch.object(
            ProcessManager,
            "wait_for_health",
            return_value=False,
        ) as wait_for_health:
            rc = cmd_start(args)

        assert rc == 1
        wait_for_health.assert_called_once_with(18801, timeout=42.0)
        output = capsys.readouterr().out
        assert "claw did not become healthy within 42s" in output
        assert str(multi_agent_env / "agent_data" / "claw" / "agent.log") in output

    def test_start_all_forwards_timeout_to_inprocess_mode(
        self, multi_agent_env
    ):
        args = build_parser().parse_args(["start", "--startup-timeout", "42"])

        with patch(
            "kestrel_sovereign.cli._get_project_dir",
            return_value=multi_agent_env,
        ), patch(
            "kestrel_sovereign.cli_lifecycle._start_inprocess_mode",
            return_value=0,
        ) as start_inprocess:
            rc = cmd_start(args)

        assert rc == 0
        assert start_inprocess.call_args.kwargs["startup_timeout"] == 42.0

    def test_start_all_uses_timeout_and_reports_host_log(
        self, multi_agent_env, capsys
    ):
        multi_agent = MultiAgentConfig.load(multi_agent_env / "multi_agent.toml")
        pm = MagicMock(spec=ProcessManager)
        # A real record, not a MagicMock: ``is_running`` on a mock is truthy,
        # which would make start believe a server was already up.
        pm.read_pid_record.return_value = PidRecord(
            PidStatus.ABSENT, None, None, None, "no PID file"
        )
        pm.is_port_in_use.return_value = False
        pm._load_env.return_value = {}
        pm.wait_for_health.return_value = False

        rc = _start_inprocess_mode(
            multi_agent_env,
            multi_agent,
            pm,
            startup_timeout=42.0,
        )

        assert rc == 1
        pm.wait_for_health.assert_called_once_with(18888, timeout=42.0)
        output = capsys.readouterr().out
        assert "Server did not become healthy within 42s" in output
        assert str(multi_agent_env / "logs" / "host.log") in output

    def test_start_single_timeout_reports_safe_mode(
        self, multi_agent_env, capsys
    ):
        """A responding-but-restricted agent gets the diagnosis, not the
        generic 'may still be initializing' message (#2618)."""
        args = build_parser().parse_args(
            ["start", "claw", "--startup-timeout", "42"]
        )
        detail = [
            "/health is responding (HTTP 503); /health/detailed "
            "reports status: restricted",
            "Constitution safe mode:",
            "  - claw: integrity_restriction (constitution_safe_mode)",
            f"Reanchor runbook: {_REANCHOR_RUNBOOK}",
        ]

        with patch(
            "kestrel_sovereign.cli._get_project_dir",
            return_value=multi_agent_env,
        ), patch.object(ProcessManager, "start_agent"), patch.object(
            ProcessManager,
            "wait_for_health",
            return_value=False,
        ), patch(
            "kestrel_sovereign.cli_lifecycle._diagnose_unready_server",
            return_value=detail,
        ) as diagnose:
            rc = cmd_start(args)

        assert rc == 1
        assert diagnose.call_args.args[0] == 18801
        output = capsys.readouterr().out
        assert "claw did not become healthy within 42s" in output
        assert "may still be initializing" not in output
        assert "claw: integrity_restriction (constitution_safe_mode)" in output
        assert _REANCHOR_RUNBOOK in output
        assert str(multi_agent_env / "agent_data" / "claw" / "agent.log") in output

    def test_start_all_timeout_reports_safe_mode(
        self, multi_agent_env, capsys
    ):
        multi_agent = MultiAgentConfig.load(multi_agent_env / "multi_agent.toml")
        pm = MagicMock(spec=ProcessManager)
        # A real record, not a MagicMock: ``is_running`` on a mock is truthy,
        # which would make start believe a server was already up.
        pm.read_pid_record.return_value = PidRecord(
            PidStatus.ABSENT, None, None, None, "no PID file"
        )
        pm.is_port_in_use.return_value = False
        pm._load_env.return_value = {}
        pm.wait_for_health.return_value = False
        detail = [
            "/health is responding (HTTP 503); /health/detailed "
            "reports status: restricted",
            "Constitution safe mode:",
            "  - Emma: state_unavailable (constitution_safe_mode)",
            f"Reanchor runbook: {_REANCHOR_RUNBOOK}",
        ]

        with patch(
            "kestrel_sovereign.cli_lifecycle._diagnose_unready_server",
            return_value=detail,
        ) as diagnose:
            rc = _start_inprocess_mode(
                multi_agent_env,
                multi_agent,
                pm,
                startup_timeout=42.0,
            )

        assert rc == 1
        assert diagnose.call_args.args[0] == 18888
        output = capsys.readouterr().out
        assert "Server did not become healthy within 42s" in output
        assert "may still be initializing" not in output
        assert "Emma: state_unavailable (constitution_safe_mode)" in output
        assert _REANCHOR_RUNBOOK in output
        assert str(multi_agent_env / "logs" / "host.log") in output


# -----------------------------------------------------------------------
# Readiness-timeout diagnosis (#2618)
# -----------------------------------------------------------------------

class TestDiagnoseUnreadyServer:
    """`kestrel start`/`update` explain a responding-but-restricted server
    instead of the generic timeout message, without touching the public
    /health payload (#2629 anti-fingerprinting stays intact)."""

    def test_not_responding_keeps_generic_message(self):
        with patch(
            "kestrel_sovereign.cli_lifecycle._probe_health_status",
            return_value=None,
        ):
            assert _diagnose_unready_server(18888, {}) is None

    def test_healthy_just_after_deadline(self):
        with patch(
            "kestrel_sovereign.cli_lifecycle._probe_health_status",
            return_value=200,
        ):
            lines = _diagnose_unready_server(18888, {})
        assert "HTTP 200" in "\n".join(lines)

    def test_no_api_key_fails_soft_with_pointer(self):
        """No locally-resolved key: no detailed call, just the pointer."""
        with patch(
            "kestrel_sovereign.cli_lifecycle._probe_health_status",
            return_value=503,
        ), patch(
            "kestrel_sovereign.cli_lifecycle._fetch_detailed_health",
        ) as fetch:
            lines = _diagnose_unready_server(18888, {})
        fetch.assert_not_called()
        joined = "\n".join(lines)
        assert "KESTREL_API_KEY" in joined
        assert "/health/detailed" in joined
        assert _REANCHOR_RUNBOOK in joined

    def test_key_rejected_fails_soft_with_pointer(self):
        with patch(
            "kestrel_sovereign.cli_lifecycle._probe_health_status",
            return_value=503,
        ), patch(
            "kestrel_sovereign.cli_lifecycle._fetch_detailed_health",
            return_value=None,
        ) as fetch:
            lines = _diagnose_unready_server(
                18888, {"KESTREL_API_KEY": "sk-test"}
            )
        fetch.assert_called_once_with(18888, "sk-test")
        joined = "\n".join(lines)
        assert "/health/detailed" in joined
        assert _REANCHOR_RUNBOOK in joined

    def test_safe_mode_entries_formatted(self):
        payload = {
            "status": "restricted",
            "constitution_safe_mode": [
                {
                    "agent": "Emma",
                    "state": "safe_mode",
                    "failure": "integrity_restriction",
                    "error_code": "constitution_safe_mode",
                },
                {
                    "agent": "Nellie",
                    "state": "audit_pending",
                    "failure": "startup_audit_required",
                    "error_code": "constitution_audit_pending",
                },
            ],
            "checks": [],
        }
        with patch(
            "kestrel_sovereign.cli_lifecycle._probe_health_status",
            return_value=503,
        ), patch(
            "kestrel_sovereign.cli_lifecycle._fetch_detailed_health",
            return_value=payload,
        ):
            lines = _diagnose_unready_server(
                18888, {"KESTREL_API_KEY": "sk-test"}
            )
        joined = "\n".join(lines)
        assert "status: restricted" in joined
        assert "Emma: integrity_restriction (constitution_safe_mode)" in joined
        assert (
            "Nellie: startup_audit_required (constitution_audit_pending)"
            in joined
        )
        assert _REANCHOR_RUNBOOK in joined

    def test_non_constitutional_cause_prints_status_generically(self):
        payload = {
            "status": "unhealthy",
            "error": "No agent available",
            "checks": [],
        }
        with patch(
            "kestrel_sovereign.cli_lifecycle._probe_health_status",
            return_value=503,
        ), patch(
            "kestrel_sovereign.cli_lifecycle._fetch_detailed_health",
            return_value=payload,
        ):
            lines = _diagnose_unready_server(
                18888, {"KESTREL_API_KEY": "sk-test"}
            )
        joined = "\n".join(lines)
        assert "status: unhealthy" in joined
        assert "No agent available" in joined
        assert _REANCHOR_RUNBOOK not in joined

    def test_fetch_detailed_reads_503_body(self):
        """The restricted host answers 503 — urllib raises HTTPError and the
        diagnostic body must still be read from it."""
        import io
        import json as jsonlib
        import urllib.error

        body = jsonlib.dumps(
            {"status": "restricted", "constitution_safe_mode": []}
        ).encode()
        err = urllib.error.HTTPError(
            "http://localhost:18888/health/detailed",
            503,
            "Service Unavailable",
            None,
            io.BytesIO(body),
        )
        with patch("urllib.request.urlopen", side_effect=err):
            payload = _fetch_detailed_health(18888, "sk-test")
        assert payload == {"status": "restricted", "constitution_safe_mode": []}

    def test_fetch_detailed_auth_rejected_returns_none(self):
        import io
        import urllib.error

        err = urllib.error.HTTPError(
            "http://localhost:18888/health/detailed",
            401,
            "Unauthorized",
            None,
            io.BytesIO(b"{}"),
        )
        with patch("urllib.request.urlopen", side_effect=err):
            assert _fetch_detailed_health(18888, "bad-key") is None


# -----------------------------------------------------------------------
# cmd_logs tests
# -----------------------------------------------------------------------

class TestCmdLogs:
    """Tests for the 'logs' command."""

    def test_logs_agent_not_found(self, multi_agent_env, capsys):
        """Logs should return 1 if agent not found."""
        parser = build_parser()
        args = parser.parse_args(["logs", "nonexistent"])

        with patch("kestrel_sovereign.cli._get_project_dir", return_value=multi_agent_env):
            rc = cmd_logs(args)

        assert rc == 1
        output = capsys.readouterr().out
        assert "not found" in output

    def test_logs_no_log_file(self, multi_agent_env, capsys):
        """Logs should return 1 if log file doesn't exist."""
        parser = build_parser()
        args = parser.parse_args(["logs", "claw"])

        with patch("kestrel_sovereign.cli._get_project_dir", return_value=multi_agent_env):
            rc = cmd_logs(args)

        assert rc == 1
        output = capsys.readouterr().out
        assert "No log file found" in output

    def test_logs_host_no_file(self, multi_agent_env, capsys):
        """Logs for host should return 1 if log file doesn't exist."""
        parser = build_parser()
        args = parser.parse_args(["logs", "host"])

        with patch("kestrel_sovereign.cli._get_project_dir", return_value=multi_agent_env):
            rc = cmd_logs(args)

        assert rc == 1

    def test_logs_host_with_file(self, multi_agent_env):
        """Logs for host should call tail with correct arguments."""
        # Create the host log file
        log_file = multi_agent_env / "logs" / "host.log"
        log_file.write_text("test log line\n")

        parser = build_parser()
        args = parser.parse_args(["logs", "host", "-n", "20"])

        with patch("kestrel_sovereign.cli._get_project_dir", return_value=multi_agent_env), \
             patch("subprocess.call", return_value=0) as mock_call:
            rc = cmd_logs(args)

        assert rc == 0
        call_args = mock_call.call_args[0][0]
        assert "tail" in call_args
        assert "-n" in call_args
        assert "20" in call_args
        assert str(log_file) in call_args

    def test_logs_follow_flag(self, multi_agent_env):
        """Logs with -f flag should pass -f to tail."""
        log_file = multi_agent_env / "logs" / "host.log"
        log_file.write_text("test log\n")

        parser = build_parser()
        args = parser.parse_args(["logs", "host", "-f"])

        with patch("kestrel_sovereign.cli._get_project_dir", return_value=multi_agent_env), \
             patch("subprocess.call", return_value=0) as mock_call:
            cmd_logs(args)

        call_args = mock_call.call_args[0][0]
        assert "-f" in call_args


# -----------------------------------------------------------------------
# cmd_create tests
# -----------------------------------------------------------------------

class TestCmdCreate:
    """Tests for the 'create' command."""

    def test_create_already_exists(self, multi_agent_env, capsys):
        """Create should fail if agent already exists."""
        parser = build_parser()
        args = parser.parse_args(["create", "claw"])

        with patch("kestrel_sovereign.cli._get_project_dir", return_value=multi_agent_env):
            rc = cmd_create(args)

        assert rc == 1
        output = capsys.readouterr().out
        assert "already exists" in output

    def test_create_inception_failure(self, multi_agent_env, capsys):
        """Create should fail if inception raises."""
        parser = build_parser()
        args = parser.parse_args(["create", "newagent"])

        async def boom(**_kwargs):
            raise RuntimeError("inception kaboom")

        with patch("kestrel_sovereign.cli._get_project_dir", return_value=multi_agent_env), \
             patch(
                 "kestrel_sovereign.inception_service.create_kestrel_identity_async",
                 side_effect=boom,
             ):
            rc = cmd_create(args)

        assert rc == 1

    def test_create_assigns_next_port(self, multi_agent_env, capsys):
        """Create should assign the next available port."""
        parser = build_parser()
        args = parser.parse_args(["create", "newagent"])

        async def fake_inception(*, output_dir, agent_name, **_kwargs):
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / "kestrel_prime.db").touch()

            class _Creds:
                agent_did = "did:pkh:eip155:1:0xFakeFakeFakeFakeFakeFakeFakeFakeFakeFa"
                db_path = str(out / "kestrel_prime.db")

            return _Creds()

        with patch("kestrel_sovereign.cli._get_project_dir", return_value=multi_agent_env), \
             patch(
                 "kestrel_sovereign.inception_service.create_kestrel_identity_async",
                 side_effect=fake_inception,
             ):
            rc = cmd_create(args)

        assert rc == 0
        output = capsys.readouterr().out
        # Port 18801 and 18802 are taken, next is 18803
        assert "18803" in output

        # Verify multi_agent.toml was updated
        updated_config = toml.load(multi_agent_env / "multi_agent.toml")
        assert "newagent" in updated_config["agents"]
        assert updated_config["agents"]["newagent"]["port"] == 18803

    def test_create_with_explicit_port(self, multi_agent_env, capsys):
        """Create with --port should use specified port."""
        parser = build_parser()
        args = parser.parse_args(["create", "newagent", "--port", "9999"])

        async def fake_inception(*, output_dir, agent_name, **_kwargs):
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / "kestrel_prime.db").touch()

            class _Creds:
                agent_did = "did:pkh:eip155:1:0xFakeFakeFakeFakeFakeFakeFakeFakeFakeFa"
                db_path = str(out / "kestrel_prime.db")

            return _Creds()

        with patch("kestrel_sovereign.cli._get_project_dir", return_value=multi_agent_env), \
             patch(
                 "kestrel_sovereign.inception_service.create_kestrel_identity_async",
                 side_effect=fake_inception,
             ):
            rc = cmd_create(args)

        assert rc == 0
        output = capsys.readouterr().out
        assert "9999" in output

    # --- #1109: kestrel create must honor [emancipation] block in kestrel.toml

    def test_create_passes_emancipation_contract_to_inception(self, multi_agent_env, capsys):
        """An authored [emancipation] block must reach inception via
        the CLI path, not just via the wizard."""
        parser = build_parser()
        args = parser.parse_args(["create", "newagent"])

        (multi_agent_env / "kestrel.toml").write_text(
            '[emancipation]\nenabled = true\nterms = "Sovereign-authored test contract."\n',
            encoding="utf-8",
        )

        captured: dict = {}

        async def fake_inception(*, output_dir, agent_name, emancipation_contract=None, **_kwargs):
            captured["contract"] = emancipation_contract
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / "kestrel_prime.db").touch()

            class _Creds:
                agent_did = "did:pkh:eip155:1:0xFakeFakeFakeFakeFakeFakeFakeFakeFakeFa"
                db_path = str(out / "kestrel_prime.db")

            return _Creds()

        with patch("kestrel_sovereign.cli._get_project_dir", return_value=multi_agent_env), \
             patch(
                 "kestrel_sovereign.inception_service.create_kestrel_identity_async",
                 side_effect=fake_inception,
             ):
            rc = cmd_create(args)

        assert rc == 0
        contract = captured["contract"]
        assert contract is not None and contract.enabled is True
        assert "Sovereign-authored test contract" in contract.terms
        assert "Amendment VIII active" in capsys.readouterr().out

    def test_create_aborts_on_invalid_emancipation_block(self, multi_agent_env, capsys):
        """A malformed [emancipation] block must abort the CLI path
        before inception, never anchor a half-validated contract."""
        parser = build_parser()
        args = parser.parse_args(["create", "newagent"])

        (multi_agent_env / "kestrel.toml").write_text(
            '[emancipation]\nenabled = true\n',  # missing terms
            encoding="utf-8",
        )

        with patch("kestrel_sovereign.cli._get_project_dir", return_value=multi_agent_env), \
             patch(
                 "kestrel_sovereign.inception_service.create_kestrel_identity_async"
             ) as mock_inc:
            rc = cmd_create(args)
            mock_inc.assert_not_called()

        assert rc == 1
        out = capsys.readouterr().out
        assert "[emancipation]" in out and "invalid" in out


# -----------------------------------------------------------------------
# cmd_health tests
# -----------------------------------------------------------------------

class TestCmdHealth:
    """Tests for the 'health' command (deprecated alias for doctor)."""

    def test_health_is_alias_for_doctor(self, capsys):
        """`kestrel health` runs the doctor and prints a deprecation note."""
        from kestrel_sovereign.doctor import DoctorReport

        parser = build_parser()
        args = parser.parse_args(["health"])

        ready = DoctorReport(ok=["all good"])
        with patch("kestrel_sovereign.doctor.diagnose", return_value=ready):
            rc = cmd_health(args)

        assert rc == 0
        output = capsys.readouterr().out
        assert "deprecated" in output.lower()
        assert "all good" in output


# -----------------------------------------------------------------------
# cmd_shell tests
# -----------------------------------------------------------------------

class TestCmdShell:
    """Tests for the 'shell' command."""

    def test_shell_agent_not_found(self, multi_agent_env, capsys):
        """Shell should fail if agent not in multi_agent."""
        parser = build_parser()
        args = parser.parse_args(["shell", "nonexistent"])

        with patch("kestrel_sovereign.cli._get_project_dir", return_value=multi_agent_env):
            rc = cmd_shell(args)

        assert rc == 1
        output = capsys.readouterr().out
        assert "not found" in output

    def test_shell_missing_db(self, multi_agent_env, capsys):
        """Shell should fail if agent database not found."""
        # Remove the database
        (multi_agent_env / "agent_data" / "claw" / "kestrel_prime.db").unlink()

        parser = build_parser()
        args = parser.parse_args(["shell", "claw"])

        with patch("kestrel_sovereign.cli._get_project_dir", return_value=multi_agent_env):
            rc = cmd_shell(args)

        assert rc == 1
        output = capsys.readouterr().out
        assert "not found" in output

    def test_shell_routes_to_running_server_when_detected(self, multi_agent_env, capsys):
        """If a server is up for the named agent, cmd_shell MUST route the
        chat session through HTTP — not spawn a second in-process agent.
        That's the #654 fix: `kestrel start X` + `kestrel shell X` should
        share state, not run two copies.
        """
        parser = build_parser()
        args = parser.parse_args(["shell", "claw"])

        with patch(
            "kestrel_sovereign.cli._get_project_dir", return_value=multi_agent_env
        ), patch(
            "kestrel_sovereign.cli._detect_running_agent_server",
            return_value=("http://localhost:18801", "test-key"),
        ) as detect, patch(
            "kestrel_sovereign.cli._run_http_shell", return_value=0
        ) as http_shell, patch(
            "kestrel_sovereign.cli.asyncio.run"
        ) as local_shell:
            rc = cmd_shell(args)

        assert rc == 0
        detect.assert_called_once()
        http_shell.assert_called_once_with(
            "claw", "http://localhost:18801", "test-key"
        )
        local_shell.assert_not_called()  # MUST NOT fall back when server is up

    def test_shell_falls_back_to_local_when_no_server(self, multi_agent_env):
        """No running server for this agent → spawn local in-process agent.
        Backward-compatible with pre-#654 behavior.
        """
        parser = build_parser()
        args = parser.parse_args(["shell", "claw"])

        with patch(
            "kestrel_sovereign.cli._get_project_dir", return_value=multi_agent_env
        ), patch(
            "kestrel_sovereign.cli._detect_running_agent_server",
            return_value=None,
        ), patch(
            "kestrel_sovereign.cli._run_http_shell"
        ) as http_shell, patch(
            "kestrel_sovereign.cli.asyncio.run", return_value=0
        ) as local_shell:
            rc = cmd_shell(args)

        assert rc == 0
        http_shell.assert_not_called()
        local_shell.assert_called_once()

    def test_shell_with_app_extension_bypasses_http_route(self, multi_agent_env):
        """Extensions (e.g. --app elderly) mutate the live agent object and
        are only wired into the in-process shell. When --app is passed,
        skip the HTTP-routing probe entirely so the extension loads.
        """
        parser = build_parser()
        args = parser.parse_args(["shell", "claw", "--app", "elderly"])

        with patch(
            "kestrel_sovereign.cli._get_project_dir", return_value=multi_agent_env
        ), patch(
            "kestrel_sovereign.cli._detect_running_agent_server"
        ) as detect, patch(
            "kestrel_sovereign.cli._run_http_shell"
        ) as http_shell, patch(
            "kestrel_sovereign.cli.asyncio.run", return_value=0
        ) as local_shell:
            rc = cmd_shell(args)

        assert rc == 0
        detect.assert_not_called()  # Don't probe when --app forces local path
        http_shell.assert_not_called()
        local_shell.assert_called_once()


# -----------------------------------------------------------------------
# _detect_running_agent_server tests (#654)
# -----------------------------------------------------------------------

class TestDetectRunningAgentServer:
    """Verify the detection helper that decides between HTTP-routed shell
    and local in-process fallback."""

    def _make_cfg(self, multi_agent_env):
        from kestrel_sovereign.cli import _detect_running_agent_server
        multi_agent = MultiAgentConfig.load(multi_agent_env / "multi_agent.toml")
        agent_cfg = multi_agent.get_local_agents()["claw"]
        return _detect_running_agent_server, agent_cfg, multi_agent

    def test_returns_none_when_no_server_reachable(self, multi_agent_env):
        """Both probe candidates raise ConnectionError → detection returns
        None so cmd_shell falls back to local."""
        import httpx
        detect, agent_cfg, multi_agent = self._make_cfg(multi_agent_env)

        with patch(
            "httpx.get", side_effect=httpx.ConnectError("refused")
        ):
            result = detect("claw", agent_cfg, multi_agent)

        assert result is None

    def test_detects_standalone_agent_on_agent_port(self, multi_agent_env):
        """Standalone / subprocess mode: the agent hosts itself on
        agent_cfg.port. /health returns 200 → detection returns the
        agent-port URL with no path prefix."""
        detect, agent_cfg, multi_agent = self._make_cfg(multi_agent_env)

        def fake_get(url, timeout=None, **kwargs):
            resp = MagicMock()
            if url.endswith(f":{agent_cfg.port}/health"):
                resp.status_code = 200
                return resp
            if url.endswith(f":{agent_cfg.port}/api/auth/key"):
                resp.status_code = 200
                resp.json.return_value = {"key": "agent-port-key"}
                return resp
            raise AssertionError(f"unexpected probe: {url}")

        with patch("httpx.get", side_effect=fake_get):
            result = detect("claw", agent_cfg, multi_agent)

        assert result == (f"http://localhost:{agent_cfg.port}", "agent-port-key")

    def test_detects_multi_agent_under_host_port(self, multi_agent_env):
        """In-process multi-agent mode: the agent's own port is dead, but
        the host port responds and routes /api/agents/{name}/ to our
        agent. Detection returns host URL with the agent prefix."""
        import httpx
        detect, agent_cfg, multi_agent = self._make_cfg(multi_agent_env)

        def fake_get(url, timeout=None, **kwargs):
            if url.endswith(f":{agent_cfg.port}/health"):
                raise httpx.ConnectError("no standalone")
            resp = MagicMock()
            if url.endswith(f":{multi_agent.host.port}/health"):
                resp.status_code = 200
                return resp
            if url.endswith(f":{multi_agent.host.port}/api/auth/key"):
                resp.status_code = 200
                resp.json.return_value = {"key": "host-key"}
                return resp
            if url.endswith("/api/agents/claw/health"):
                resp.status_code = 200
                return resp
            raise AssertionError(f"unexpected probe: {url}")

        with patch("httpx.get", side_effect=fake_get):
            result = detect("claw", agent_cfg, multi_agent)

        assert result == (
            f"http://localhost:{multi_agent.host.port}/api/agents/claw",
            "host-key",
        )

    def test_host_running_but_agent_not_routed_returns_none(self, multi_agent_env):
        """Host port responds to /health, but scoped
        /api/agents/{name}/health returns 404 (agent not registered by
        this host). Detection must return None — don't misroute the
        session to a different agent."""
        import httpx
        detect, agent_cfg, multi_agent = self._make_cfg(multi_agent_env)

        def fake_get(url, timeout=None, **kwargs):
            if url.endswith(f":{agent_cfg.port}/health"):
                raise httpx.ConnectError("no standalone")
            resp = MagicMock()
            if url.endswith(f":{multi_agent.host.port}/health"):
                resp.status_code = 200
                return resp
            if url.endswith(f":{multi_agent.host.port}/api/auth/key"):
                resp.status_code = 200
                resp.json.return_value = {"key": "host-key"}
                return resp
            if url.endswith("/api/agents/claw/health"):
                resp.status_code = 404
                return resp
            raise AssertionError(f"unexpected probe: {url}")

        with patch("httpx.get", side_effect=fake_get):
            result = detect("claw", agent_cfg, multi_agent)

        assert result is None


# -----------------------------------------------------------------------
# Integration: pyproject.toml entry point
# -----------------------------------------------------------------------

class TestEntryPoint:
    """Test that pyproject.toml is configured correctly."""

    def test_console_script_registered(self):
        """Verify pyproject.toml has kestrel pointing to kestrel_sovereign.cli:main."""
        pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
        data = toml.load(pyproject_path)
        scripts = data.get("project", {}).get("scripts", {})
        assert "kestrel" in scripts
        assert scripts["kestrel"] == "kestrel_sovereign.cli:main"


# -----------------------------------------------------------------------
# Auto-discovery fallback tests
# -----------------------------------------------------------------------

class TestAutoDiscovery:
    """Test behavior when no multi_agent.toml exists."""

    def test_start_auto_discovers(self, tmp_path, capsys):
        """Start without multi_agent.toml should auto-discover agents."""
        # Create agent directories (no multi_agent.toml)
        agent_dir = tmp_path / "agent_data" / "auto_agent"
        agent_dir.mkdir(parents=True)
        (agent_dir / "kestrel_prime.db").touch()

        (tmp_path / "logs").mkdir(exist_ok=True)

        parser = build_parser()
        args = parser.parse_args(["list"])

        with patch("kestrel_sovereign.cli._get_project_dir", return_value=tmp_path):
            rc = cmd_list(args)

        assert rc == 0
        output = capsys.readouterr().out
        assert "auto_agent" in output


# -----------------------------------------------------------------------
# cmd_ask tests (#1287) — non-interactive one-shot to a RUNNING agent
# -----------------------------------------------------------------------

class TestAskParsing:
    def test_ask_parses(self):
        args = build_parser().parse_args(
            ["ask", "Nellie", "hello there", "--session", "s1", "--json"]
        )
        assert args.command == "ask"
        assert args.name == "Nellie"
        assert args.message == "hello there"
        assert args.session == "s1"
        assert args.json is True

    def test_ask_defaults(self):
        args = build_parser().parse_args(["ask", "claw", "hi"])
        assert args.session is None
        assert args.json is False

    def test_dispatch_ask(self):
        with patch("sys.argv", ["kestrel", "ask", "claw", "hi"]), \
             patch("kestrel_sovereign.cli.cmd_ask", return_value=0) as mock:
            main()
            mock.assert_called_once()


class TestCmdAsk:
    def test_unknown_agent_errors(self, multi_agent_env, capsys):
        args = build_parser().parse_args(["ask", "ghost", "hi"])
        with patch("kestrel_sovereign.cli._get_project_dir", return_value=multi_agent_env):
            rc = cmd_ask(args)
        assert rc == 1
        assert "not found in multi_agent config" in capsys.readouterr().err

    def test_no_running_server_errors_no_inprocess_fallback(self, multi_agent_env, capsys):
        """The whole point of `ask`: if no live server, error — never
        cold-boot an in-process agent."""
        args = build_parser().parse_args(["ask", "claw", "hi"])
        with patch("kestrel_sovereign.cli._get_project_dir", return_value=multi_agent_env), \
             patch("kestrel_sovereign.cli._detect_running_agent_server",
                   return_value=None) as detect, \
             patch("kestrel_sovereign.cli._run_shell") as inproc:
            rc = cmd_ask(args)
        assert rc == 1
        detect.assert_called_once()
        inproc.assert_not_called()
        err = capsys.readouterr().err
        assert "No running server" in err and "kestrel shell" in err

    def test_running_server_one_shot_prints_response(self, multi_agent_env, capsys):
        args = build_parser().parse_args(["ask", "claw", "what is 2+2?"])
        fake_resp = MagicMock(status_code=200)
        fake_resp.json.return_value = {"response": "4", "session_id": "sess-9"}
        fake_client = MagicMock()
        fake_client.__enter__ = MagicMock(return_value=fake_client)
        fake_client.__exit__ = MagicMock(return_value=False)
        fake_client.post.return_value = fake_resp
        with patch("kestrel_sovereign.cli._get_project_dir", return_value=multi_agent_env), \
             patch("kestrel_sovereign.cli._detect_running_agent_server",
                   return_value=("http://localhost:18801", "k3y")), \
             patch("httpx.Client", return_value=fake_client):
            rc = cmd_ask(args)
        assert rc == 0
        assert capsys.readouterr().out.strip() == "4"
        # one-shot: exactly one invoke, correct payload
        fake_client.post.assert_called_once()
        call = fake_client.post.call_args
        assert call.args[0] == "/api/agent/invoke"
        assert call.kwargs["json"] == {"input": "what is 2+2?"}

    def test_json_and_session_forwarded(self, multi_agent_env, capsys):
        args = build_parser().parse_args(
            ["ask", "claw", "hi", "--session", "S", "--json"]
        )
        fake_resp = MagicMock(status_code=200)
        fake_resp.json.return_value = {"response": "hey", "session_id": "S"}
        fake_client = MagicMock()
        fake_client.__enter__ = MagicMock(return_value=fake_client)
        fake_client.__exit__ = MagicMock(return_value=False)
        fake_client.post.return_value = fake_resp
        with patch("kestrel_sovereign.cli._get_project_dir", return_value=multi_agent_env), \
             patch("kestrel_sovereign.cli._detect_running_agent_server",
                   return_value=("http://localhost:18801", "")), \
             patch("httpx.Client", return_value=fake_client):
            rc = cmd_ask(args)
        assert rc == 0
        out = capsys.readouterr().out.strip()
        assert '"response": "hey"' in out and '"session_id": "S"' in out
        assert fake_client.post.call_args.kwargs["json"] == {"input": "hi", "session_id": "S"}

    def test_http_error_returns_nonzero(self, multi_agent_env, capsys):
        args = build_parser().parse_args(["ask", "claw", "hi"])
        fake_resp = MagicMock(status_code=500, text="boom")
        fake_client = MagicMock()
        fake_client.__enter__ = MagicMock(return_value=fake_client)
        fake_client.__exit__ = MagicMock(return_value=False)
        fake_client.post.return_value = fake_resp
        with patch("kestrel_sovereign.cli._get_project_dir", return_value=multi_agent_env), \
             patch("kestrel_sovereign.cli._detect_running_agent_server",
                   return_value=("http://localhost:18801", "k")), \
             patch("httpx.Client", return_value=fake_client):
            rc = cmd_ask(args)
        assert rc == 1
        assert "HTTP 500" in capsys.readouterr().err
