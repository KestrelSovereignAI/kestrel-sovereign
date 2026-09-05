"""Focused output contracts for host and named-agent start forms."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from kestrel_sovereign.cli import build_parser, cmd_start
from kestrel_sovereign.cli_lifecycle import _start_inprocess_mode
from kestrel_sovereign.multi_agent.config import (
    DEFAULT_AGENT_START_PORT,
    DEFAULT_HOST_PORT,
    DEFAULT_HOST_BIND,
    LocalAgentConfig,
    MultiAgentConfig,
)
from kestrel_sovereign.multi_agent.process_manager import (
    DEFAULT_STARTUP_HEALTH_TIMEOUT_SECONDS,
    PidRecord,
    PidStatus,
    ProcessManager,
)
from kestrel_sovereign.setup.steps.agent import DEFAULT_QUICKSTART_AGENT_NAME


def _quickstart_config(tmp_path):
    config = MultiAgentConfig()
    config.agents[DEFAULT_QUICKSTART_AGENT_NAME] = LocalAgentConfig(
        data_dir=tmp_path / "agent_data" / DEFAULT_QUICKSTART_AGENT_NAME,
        port=DEFAULT_AGENT_START_PORT,
        autostart=True,
    )
    return config


def test_host_start_output_uses_configured_host_port(tmp_path, capsys):
    config = _quickstart_config(tmp_path)
    process_manager = MagicMock(spec=ProcessManager)
    # A real record, not a MagicMock: ``is_running`` on a mock is truthy, so
    # start would take the already-running branch and never probe health.
    process_manager.read_pid_record.return_value = PidRecord(
        PidStatus.ABSENT, None, None, None, "no PID file"
    )
    process_manager.is_port_in_use.return_value = False
    process_manager._load_env.return_value = {"KESTREL_API_KEY": "host-test-key"}
    process_manager.wait_for_health.return_value = True

    result = _start_inprocess_mode(tmp_path, config, process_manager)

    assert result == 0
    process_manager.wait_for_health.assert_called_once_with(
        DEFAULT_HOST_PORT,
        timeout=DEFAULT_STARTUP_HEALTH_TIMEOUT_SECONDS,
    )
    output = capsys.readouterr().out
    assert f"URL:      http://localhost:{DEFAULT_HOST_PORT}" in output
    assert f"Starting server on :{DEFAULT_HOST_PORT}" in output
    assert f"MultiAgent ready: http://localhost:{DEFAULT_HOST_PORT}" in output
    assert f"http://localhost:{DEFAULT_HOST_PORT}/#key=host-test-key" in output


def test_host_start_rejects_missing_out_of_band_api_key(tmp_path, capsys):
    config = _quickstart_config(tmp_path)
    process_manager = MagicMock(spec=ProcessManager)
    process_manager.read_pid_record.return_value = PidRecord(
        PidStatus.ABSENT, None, None, None, "no PID file"
    )
    process_manager._load_env.return_value = {}

    result = _start_inprocess_mode(tmp_path, config, process_manager)

    assert result == 1
    process_manager._spawn_detached.assert_not_called()
    output = capsys.readouterr().out
    assert "KESTREL_API_KEY" in output
    assert "multi-agent host" in output.lower()
    assert "kestrel setup keys" in output
    assert "--steps" not in output


def test_oauth_only_host_start_still_requires_sovereign_api_key(tmp_path, capsys):
    config = _quickstart_config(tmp_path)
    process_manager = MagicMock(spec=ProcessManager)
    process_manager.read_pid_record.return_value = PidRecord(
        PidStatus.ABSENT, None, None, None, "no PID file"
    )
    process_manager._load_env.return_value = {
        "KESTREL_REQUIRE_OAUTH": "true",
        "GOOGLE_CLIENT_ID": "operator-client-id",
        "GOOGLE_CLIENT_SECRET": "operator-client-secret",
        "KESTREL_ALLOWED_EMAILS": "operator@example.com",
    }

    result = _start_inprocess_mode(tmp_path, config, process_manager)

    assert result == 1
    process_manager._spawn_detached.assert_not_called()
    output = capsys.readouterr().out
    assert "stable KESTREL_API_KEY" in output
    assert "kestrel setup keys" in output


def test_oauth_flag_without_credentials_does_not_start_keyless_host(
    tmp_path, capsys
):
    config = _quickstart_config(tmp_path)
    process_manager = MagicMock(spec=ProcessManager)
    process_manager.read_pid_record.return_value = PidRecord(
        PidStatus.ABSENT, None, None, None, "no PID file"
    )
    process_manager._load_env.return_value = {"KESTREL_REQUIRE_OAUTH": "true"}

    result = _start_inprocess_mode(tmp_path, config, process_manager)

    assert result == 1
    process_manager._spawn_detached.assert_not_called()
    output = capsys.readouterr().out
    assert "stable KESTREL_API_KEY" in output


def test_already_running_host_ignores_credentials_for_a_future_launch(
    tmp_path, capsys
):
    """An idempotent start reports the live PID even after project-key rotation."""

    config = _quickstart_config(tmp_path)
    process_manager = MagicMock(spec=ProcessManager)
    process_manager.read_pid_record.return_value = PidRecord(
        PidStatus.LIVE,
        4242,
        str(tmp_path),
        DEFAULT_HOST_PORT,
        "verified live host",
        123.0,
    )
    # Empty configuration would reject a new API-key fleet launch. It must not
    # reinterpret the authority of the already-running process.
    process_manager._load_env.return_value = {}

    result = _start_inprocess_mode(tmp_path, config, process_manager)

    assert result == 0
    process_manager._load_env.assert_not_called()
    process_manager._spawn_detached.assert_not_called()
    assert "Server already running (PID: 4242)" in capsys.readouterr().out


def test_named_start_output_uses_assigned_agent_port(tmp_path, capsys):
    config = _quickstart_config(tmp_path)
    config.save(tmp_path / "multi_agent.toml")
    args = build_parser().parse_args(
        ["start", DEFAULT_QUICKSTART_AGENT_NAME]
    )

    with patch(
        "kestrel_sovereign.cli._get_project_dir",
        return_value=tmp_path,
    ), patch(
        "kestrel_sovereign.cli._maybe_first_run_setup",
        return_value=None,
    ), patch.object(
        ProcessManager,
        "start_agent",
    ) as start_agent, patch.object(
        ProcessManager,
        "wait_for_health",
        return_value=True,
    ):
        result = cmd_start(args)

    assert result == 0
    start_agent.assert_called_once()
    assert start_agent.call_args.args[:3] == (
        DEFAULT_QUICKSTART_AGENT_NAME,
        config.agents[DEFAULT_QUICKSTART_AGENT_NAME],
        DEFAULT_HOST_BIND,
    )
    assert start_agent.call_args.kwargs == {"standalone": True}
    assert (
        f"Starting {DEFAULT_QUICKSTART_AGENT_NAME} "
        f"on :{DEFAULT_AGENT_START_PORT}"
        in capsys.readouterr().out
    )
