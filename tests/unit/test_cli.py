"""
Unit tests for the unified kestrel CLI.

Tests argument parsing, command dispatch, and individual command handlers.
"""

import os
import signal
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import toml

from kestrel_sovereign.cli import (
    build_parser,
    main,
    cmd_start,
    cmd_stop,
    cmd_status,
    cmd_logs,
    cmd_list,
    cmd_create,
    cmd_shell,
    cmd_health,
    cmd_config,
    _is_port_in_use,
    _is_process_running,
    _read_pid,
    _write_pid,
    _clear_pid,
    _host_pid_file,
    _agent_pid_file,
    _agent_log_file,
    _get_project_dir,
)


# -----------------------------------------------------------------------
# Argument parsing tests
# -----------------------------------------------------------------------

class TestArgumentParsing:
    """Test CLI argument parsing."""

    def test_no_command_returns_1(self):
        """No command should print help and return 1."""
        with patch("sys.argv", ["kestrel"]):
            assert main() == 1

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

    def test_stop_no_name(self):
        """'stop' with no name should parse successfully."""
        parser = build_parser()
        args = parser.parse_args(["stop"])
        assert args.command == "stop"
        assert args.name is None

    def test_stop_with_name(self):
        """'stop <name>' should parse agent name."""
        parser = build_parser()
        args = parser.parse_args(["stop", "claw"])
        assert args.command == "stop"
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

    def test_dispatch_stop(self):
        """'stop' should dispatch to cmd_stop."""
        with patch("sys.argv", ["kestrel", "stop"]), \
             patch("kestrel_sovereign.cli.cmd_stop", return_value=0) as mock:
            main()
            mock.assert_called_once()

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

    def test_dispatch_config(self):
        """'config' should dispatch to cmd_config."""
        with patch("sys.argv", ["kestrel", "config"]), \
             patch("kestrel_sovereign.cli.cmd_config", return_value=0) as mock:
            main()
            mock.assert_called_once()


# -----------------------------------------------------------------------
# Process helper tests
# -----------------------------------------------------------------------

class TestProcessHelpers:
    """Test process management helper functions."""

    def test_read_pid_no_file(self, tmp_path):
        """Reading PID from non-existent file returns None."""
        assert _read_pid(tmp_path / "nonexistent.pid") is None

    def test_write_and_read_pid(self, tmp_path):
        """Write and read PID should round-trip."""
        pid_file = tmp_path / "test.pid"
        _write_pid(pid_file, 12345)
        assert _read_pid(pid_file) == 12345

    def test_clear_pid(self, tmp_path):
        """Clearing PID should remove the file."""
        pid_file = tmp_path / "test.pid"
        _write_pid(pid_file, 12345)
        assert pid_file.exists()
        _clear_pid(pid_file)
        assert not pid_file.exists()

    def test_clear_pid_nonexistent(self, tmp_path):
        """Clearing non-existent PID file should not raise."""
        _clear_pid(tmp_path / "nonexistent.pid")

    def test_read_pid_invalid_content(self, tmp_path):
        """Reading PID from file with invalid content returns None."""
        pid_file = tmp_path / "bad.pid"
        pid_file.write_text("not-a-number")
        assert _read_pid(pid_file) is None

    def test_agent_pid_file(self, tmp_path):
        """Agent PID file should be in agent directory."""
        pid_file = _agent_pid_file(tmp_path / "myagent")
        assert pid_file == tmp_path / "myagent" / "agent.pid"

    def test_agent_log_file(self, tmp_path):
        """Agent log file should be in agent directory."""
        log_file = _agent_log_file(tmp_path / "myagent")
        assert log_file == tmp_path / "myagent" / "agent.log"

    def test_is_process_running_current(self):
        """Current process should be detected as running."""
        assert _is_process_running(os.getpid()) is True

    def test_is_process_running_invalid(self):
        """Non-existent PID should not be detected as running."""
        assert _is_process_running(999999) is False

    def test_is_port_in_use_no(self):
        """An unused port should return False."""
        # Port 0 is never in use (it's assigned dynamically)
        assert _is_port_in_use(0) is False


# -----------------------------------------------------------------------
# Rookery fixture
# -----------------------------------------------------------------------

@pytest.fixture
def rookery_env(tmp_path):
    """Set up a temporary rookery environment with config and agent dirs."""
    # Create agent directories
    claw_dir = tmp_path / "agent_data" / "claw"
    claw_dir.mkdir(parents=True)
    (claw_dir / "kestrel_prime.db").touch()

    testbot_dir = tmp_path / "agent_data" / "testbot"
    testbot_dir.mkdir(parents=True)
    (testbot_dir / "kestrel_prime.db").touch()

    # Create rookery.toml
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
    config_path = tmp_path / "rookery.toml"
    with open(config_path, "w") as f:
        toml.dump(config, f)

    # Create logs directory
    (tmp_path / "logs").mkdir(exist_ok=True)

    return tmp_path


# -----------------------------------------------------------------------
# cmd_list tests
# -----------------------------------------------------------------------

class TestCmdList:
    """Tests for the 'list' command."""

    def test_list_agents(self, rookery_env, capsys):
        """List should show agents from rookery config."""
        parser = build_parser()
        args = parser.parse_args(["list"])

        with patch("kestrel_sovereign.cli._get_project_dir", return_value=rookery_env):
            rc = cmd_list(args)

        assert rc == 0
        output = capsys.readouterr().out
        assert "claw" in output
        assert "testbot" in output
        assert "18801" in output
        assert "18802" in output

    def test_list_empty(self, tmp_path, capsys):
        """List with no agents should show empty message."""
        # Create empty rookery.toml
        config_path = tmp_path / "rookery.toml"
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

    def test_status_all_offline(self, rookery_env, capsys):
        """Status should show all processes as offline when nothing runs."""
        parser = build_parser()
        args = parser.parse_args(["status"])

        with patch("kestrel_sovereign.cli._get_project_dir", return_value=rookery_env):
            rc = cmd_status(args)

        assert rc == 0
        output = capsys.readouterr().out
        assert "host" in output
        assert "claw" in output
        assert "testbot" in output
        assert "offline" in output

    def test_status_shows_header(self, rookery_env, capsys):
        """Status should show column headers."""
        parser = build_parser()
        args = parser.parse_args(["status"])

        with patch("kestrel_sovereign.cli._get_project_dir", return_value=rookery_env):
            cmd_status(args)

        output = capsys.readouterr().out
        assert "NAME" in output
        assert "PORT" in output
        assert "STATUS" in output
        assert "PID" in output


# -----------------------------------------------------------------------
# cmd_stop tests
# -----------------------------------------------------------------------

class TestCmdStop:
    """Tests for the 'stop' command."""

    def test_stop_single_agent_not_found(self, rookery_env, capsys):
        """Stop should return 1 if agent name not found."""
        parser = build_parser()
        args = parser.parse_args(["stop", "nonexistent"])

        with patch("kestrel_sovereign.cli._get_project_dir", return_value=rookery_env):
            rc = cmd_stop(args)

        assert rc == 1
        output = capsys.readouterr().out
        assert "not found" in output

    def test_stop_single_not_running(self, rookery_env, capsys):
        """Stop should succeed even if agent is not running."""
        parser = build_parser()
        args = parser.parse_args(["stop", "claw"])

        with patch("kestrel_sovereign.cli._get_project_dir", return_value=rookery_env):
            rc = cmd_stop(args)

        assert rc == 0

    def test_stop_all(self, rookery_env, capsys):
        """Stop all should stop agents and host."""
        parser = build_parser()
        args = parser.parse_args(["stop"])

        with patch("kestrel_sovereign.cli._get_project_dir", return_value=rookery_env):
            rc = cmd_stop(args)

        assert rc == 0
        output = capsys.readouterr().out
        assert "Rookery stopped" in output


# -----------------------------------------------------------------------
# cmd_start tests
# -----------------------------------------------------------------------

class TestCmdStart:
    """Tests for the 'start' command."""

    def test_start_single_not_found(self, rookery_env, capsys):
        """Start should return 1 if agent name not found."""
        parser = build_parser()
        args = parser.parse_args(["start", "nonexistent"])

        with patch("kestrel_sovereign.cli._get_project_dir", return_value=rookery_env):
            rc = cmd_start(args)

        assert rc == 1
        output = capsys.readouterr().out
        assert "not found" in output

    def test_start_single_missing_db(self, rookery_env, capsys):
        """Start should fail if agent data dir has no database."""
        # Remove the database file
        db_file = rookery_env / "agent_data" / "claw" / "kestrel_prime.db"
        db_file.unlink()

        parser = build_parser()
        args = parser.parse_args(["start", "claw"])

        with patch("kestrel_sovereign.cli._get_project_dir", return_value=rookery_env):
            rc = cmd_start(args)

        assert rc == 1


# -----------------------------------------------------------------------
# cmd_logs tests
# -----------------------------------------------------------------------

class TestCmdLogs:
    """Tests for the 'logs' command."""

    def test_logs_agent_not_found(self, rookery_env, capsys):
        """Logs should return 1 if agent not found."""
        parser = build_parser()
        args = parser.parse_args(["logs", "nonexistent"])

        with patch("kestrel_sovereign.cli._get_project_dir", return_value=rookery_env):
            rc = cmd_logs(args)

        assert rc == 1
        output = capsys.readouterr().out
        assert "not found" in output

    def test_logs_no_log_file(self, rookery_env, capsys):
        """Logs should return 1 if log file doesn't exist."""
        parser = build_parser()
        args = parser.parse_args(["logs", "claw"])

        with patch("kestrel_sovereign.cli._get_project_dir", return_value=rookery_env):
            rc = cmd_logs(args)

        assert rc == 1
        output = capsys.readouterr().out
        assert "No log file found" in output

    def test_logs_host_no_file(self, rookery_env, capsys):
        """Logs for host should return 1 if log file doesn't exist."""
        parser = build_parser()
        args = parser.parse_args(["logs", "host"])

        with patch("kestrel_sovereign.cli._get_project_dir", return_value=rookery_env):
            rc = cmd_logs(args)

        assert rc == 1

    def test_logs_host_with_file(self, rookery_env):
        """Logs for host should call tail with correct arguments."""
        # Create the host log file
        log_file = rookery_env / "logs" / "host.log"
        log_file.write_text("test log line\n")

        parser = build_parser()
        args = parser.parse_args(["logs", "host", "-n", "20"])

        with patch("kestrel_sovereign.cli._get_project_dir", return_value=rookery_env), \
             patch("subprocess.call", return_value=0) as mock_call:
            rc = cmd_logs(args)

        assert rc == 0
        call_args = mock_call.call_args[0][0]
        assert "tail" in call_args
        assert "-n" in call_args
        assert "20" in call_args
        assert str(log_file) in call_args

    def test_logs_follow_flag(self, rookery_env):
        """Logs with -f flag should pass -f to tail."""
        log_file = rookery_env / "logs" / "host.log"
        log_file.write_text("test log\n")

        parser = build_parser()
        args = parser.parse_args(["logs", "host", "-f"])

        with patch("kestrel_sovereign.cli._get_project_dir", return_value=rookery_env), \
             patch("subprocess.call", return_value=0) as mock_call:
            cmd_logs(args)

        call_args = mock_call.call_args[0][0]
        assert "-f" in call_args


# -----------------------------------------------------------------------
# cmd_create tests
# -----------------------------------------------------------------------

class TestCmdCreate:
    """Tests for the 'create' command."""

    def test_create_already_exists(self, rookery_env, capsys):
        """Create should fail if agent already exists."""
        parser = build_parser()
        args = parser.parse_args(["create", "claw"])

        with patch("kestrel_sovereign.cli._get_project_dir", return_value=rookery_env):
            rc = cmd_create(args)

        assert rc == 1
        output = capsys.readouterr().out
        assert "already exists" in output

    def test_create_inception_failure(self, rookery_env, capsys):
        """Create should fail if inception fails."""
        parser = build_parser()
        args = parser.parse_args(["create", "newagent"])

        mock_result = MagicMock()
        mock_result.returncode = 1

        with patch("kestrel_sovereign.cli._get_project_dir", return_value=rookery_env), \
             patch("subprocess.run", return_value=mock_result):
            rc = cmd_create(args)

        assert rc == 1

    def test_create_assigns_next_port(self, rookery_env, capsys):
        """Create should assign the next available port."""
        parser = build_parser()
        args = parser.parse_args(["create", "newagent"])

        mock_result = MagicMock()
        mock_result.returncode = 0

        agent_dir = rookery_env / "agent_data" / "newagent"

        def fake_inception(*a, **kw):
            # Simulate inception creating the agent directory
            agent_dir.mkdir(parents=True, exist_ok=True)
            (agent_dir / "kestrel_prime.db").touch()
            return mock_result

        with patch("kestrel_sovereign.cli._get_project_dir", return_value=rookery_env), \
             patch("subprocess.run", side_effect=fake_inception):
            rc = cmd_create(args)

        assert rc == 0
        output = capsys.readouterr().out
        # Port 18801 and 18802 are taken, next is 18803
        assert "18803" in output

        # Verify rookery.toml was updated
        updated_config = toml.load(rookery_env / "rookery.toml")
        assert "newagent" in updated_config["agents"]
        assert updated_config["agents"]["newagent"]["port"] == 18803

    def test_create_with_explicit_port(self, rookery_env, capsys):
        """Create with --port should use specified port."""
        parser = build_parser()
        args = parser.parse_args(["create", "newagent", "--port", "9999"])

        mock_result = MagicMock()
        mock_result.returncode = 0

        agent_dir = rookery_env / "agent_data" / "newagent"

        def fake_inception(*a, **kw):
            agent_dir.mkdir(parents=True, exist_ok=True)
            (agent_dir / "kestrel_prime.db").touch()
            return mock_result

        with patch("kestrel_sovereign.cli._get_project_dir", return_value=rookery_env), \
             patch("subprocess.run", side_effect=fake_inception):
            rc = cmd_create(args)

        assert rc == 0
        output = capsys.readouterr().out
        assert "9999" in output


# -----------------------------------------------------------------------
# cmd_health tests
# -----------------------------------------------------------------------

class TestCmdHealth:
    """Tests for the 'health' command."""

    def test_health_calls_run_health_check(self):
        """Health should delegate to run_health_check."""
        parser = build_parser()
        args = parser.parse_args(["health"])

        with patch("kestrel_sovereign.cli.cmd_health") as mock:
            mock.return_value = 0
            # Just verify it's callable
            assert cmd_health(args) == 0


# -----------------------------------------------------------------------
# cmd_shell tests
# -----------------------------------------------------------------------

class TestCmdShell:
    """Tests for the 'shell' command."""

    def test_shell_agent_not_found(self, rookery_env, capsys):
        """Shell should fail if agent not in rookery."""
        parser = build_parser()
        args = parser.parse_args(["shell", "nonexistent"])

        with patch("kestrel_sovereign.cli._get_project_dir", return_value=rookery_env):
            rc = cmd_shell(args)

        assert rc == 1
        output = capsys.readouterr().out
        assert "not found" in output

    def test_shell_missing_db(self, rookery_env, capsys):
        """Shell should fail if agent database not found."""
        # Remove the database
        (rookery_env / "agent_data" / "claw" / "kestrel_prime.db").unlink()

        parser = build_parser()
        args = parser.parse_args(["shell", "claw"])

        with patch("kestrel_sovereign.cli._get_project_dir", return_value=rookery_env):
            rc = cmd_shell(args)

        assert rc == 1
        output = capsys.readouterr().out
        assert "not found" in output


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
    """Test behavior when no rookery.toml exists."""

    def test_start_auto_discovers(self, tmp_path, capsys):
        """Start without rookery.toml should auto-discover agents."""
        # Create agent directories (no rookery.toml)
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
