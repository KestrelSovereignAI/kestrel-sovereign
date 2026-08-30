"""CLI contract for cooperative Stop and separate process shutdown (#3160)."""

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx

from kestrel_sovereign.cli import build_parser
from kestrel_sovereign import cli_stop
from kestrel_sovereign.cli_stop import cmd_stop


def _args(*, name="Emma", all_agents=False, reason=None):
    return SimpleNamespace(name=name, all=all_agents, reason=reason)


def _response(*, status=200, payload):
    response = MagicMock(status_code=status, text="")
    response.json.return_value = payload
    return response


def _outcome(disposition="stopped", *, receipt="receipt-1", agent="did:emma"):
    return {
        "scope": "agent",
        "requested_target": agent,
        "resolved_target": agent,
        "agent_id": agent,
        "disposition": disposition,
        "correlation_id": "cli:fixed",
        "receipt_id": receipt,
    }


def test_parser_separates_cooperative_stop_from_process_shutdown():
    stop = build_parser().parse_args(["stop", "Emma", "--reason", "andon"])
    assert stop.command == "stop"
    assert stop.name == "Emma"
    assert stop.reason == "andon"
    assert stop.all is False
    assert not hasattr(stop, "force")

    fleet = build_parser().parse_args(["stop", "--all"])
    assert fleet.all is True
    shutdown = build_parser().parse_args(["shutdown", "Emma", "--force"])
    assert shutdown.command == "shutdown"
    assert shutdown.name == "Emma"
    assert shutdown.force is True


def test_stop_requires_exactly_one_agent_or_all(capsys):
    assert cmd_stop(_args(name=None)) == 2
    assert cmd_stop(_args(name="Emma", all_agents=True)) == 2
    assert "exactly one" in capsys.readouterr().out


def test_host_stop_resolution_uses_host_door_and_local_sovereign_key(tmp_path):
    from kestrel_sovereign import cli

    config = SimpleNamespace(host=SimpleNamespace(port=8888))
    config.get_local_agents = lambda: {}
    config.get_remote_agents = lambda: {}
    with (
        patch.object(cli, "_get_project_dir", return_value=tmp_path),
        patch.object(cli.MultiAgentConfig, "load", return_value=config),
        patch(
            "kestrel_sovereign.cli_stop.spawned_agent_env",
            return_value={"KESTREL_API_KEY": "sovereign-secret"},
        ),
    ):
        assert cli_stop._stop_endpoint(
            _args(name=None, all_agents=True)
        ) == ("http://localhost:8888/api/host/stop", "sovereign-secret")


def test_named_stop_resolution_delegates_agent_routing_to_live_http_probe(tmp_path):
    from kestrel_sovereign import cli

    agent_config = SimpleNamespace(port=8801)
    config = SimpleNamespace(host=SimpleNamespace(port=8888))
    config.get_local_agents = lambda: {"Emma": agent_config}
    config.get_remote_agents = lambda: {}
    with (
        patch.object(cli, "_get_project_dir", return_value=tmp_path),
        patch.object(cli.MultiAgentConfig, "load", return_value=config),
        patch.object(
            cli,
            "_detect_running_agent_server",
            return_value=("http://localhost:8888/api/agents/Emma", "key"),
        ) as detect,
    ):
        resolved = cli_stop._stop_endpoint(_args())
    assert resolved == (
        "http://localhost:8888/api/agents/Emma/api/agent/stop",
        "key",
    )
    detect.assert_called_once_with("Emma", agent_config, config)


def test_cooperative_stop_module_has_no_process_mutation_door():
    source = inspect.getsource(cli_stop)
    assert "stop_agent(" not in source
    assert "kill_process(" not in source
    assert "terminate_agent(" not in source
    assert "cmd_shutdown(" not in source


def test_named_stop_posts_only_intent_and_prints_receipted_outcome(capsys):
    response = _response(
        payload={"success": True, "stop_outcomes": [_outcome()]},
    )
    with (
        patch(
            "kestrel_sovereign.cli_stop._stop_endpoint",
            return_value=("http://host/api/agent/stop", "secret"),
        ),
        patch(
            "kestrel_sovereign.cli_stop._operation_id",
            return_value="cli:fixed",
        ),
        patch("httpx.post", return_value=response) as post,
    ):
        assert cmd_stop(_args(reason="andon")) == 0

    post.assert_called_once()
    assert post.call_args.args == ("http://host/api/agent/stop",)
    assert post.call_args.kwargs["json"] == {
        "correlation_id": "cli:fixed",
        "reason": "andon",
    }
    assert post.call_args.kwargs["headers"] == {"X-API-Key": "secret"}
    output = capsys.readouterr().out
    assert "did:emma" in output
    assert "stopped" in output
    assert "receipt-1" in output


def test_refused_agent_error_prints_typed_outcome_and_fails(capsys):
    refused = _outcome("refused", receipt="receipt-refused")
    response = _response(
        status=503,
        payload={"error": {"details": [refused]}},
    )
    with (
        patch(
            "kestrel_sovereign.cli_stop._stop_endpoint",
            return_value=("http://host/api/agent/stop", "secret"),
        ),
        patch("httpx.post", return_value=response),
    ):
        assert cmd_stop(_args()) == 1

    output = capsys.readouterr().out
    assert "did:emma" in output
    assert "refused" in output
    assert "receipt-refused" in output


def test_stop_all_preserves_partial_outcomes_and_nonzero_exit(capsys):
    response = _response(
        payload={
            "success": False,
            "state": "partial",
            "stop_outcomes": [
                _outcome(agent="did:alpha"),
                _outcome(
                    "unreachable",
                    receipt="receipt-2",
                    agent="did:beta",
                ),
            ],
        },
    )
    with (
        patch(
            "kestrel_sovereign.cli_stop._stop_endpoint",
            return_value=("http://host/api/host/stop", "secret"),
        ),
        patch("httpx.post", return_value=response),
    ):
        assert cmd_stop(_args(name=None, all_agents=True)) == 1

    output = capsys.readouterr().out
    assert "did:alpha" in output and "stopped" in output
    assert "did:beta" in output and "unreachable" in output


def test_transport_failure_is_indeterminate(capsys):
    request = httpx.Request("POST", "http://host/api/agent/stop")
    with (
        patch(
            "kestrel_sovereign.cli_stop._stop_endpoint",
            return_value=("http://host/api/agent/stop", "secret"),
        ),
        patch(
            "httpx.post",
            side_effect=httpx.ConnectError("offline", request=request),
        ),
    ):
        assert cmd_stop(_args()) == 1
    assert "indeterminate" in capsys.readouterr().out


def test_success_without_a_durable_receipt_is_nonzero(capsys):
    response = _response(
        payload={
            "success": True,
            "stop_outcomes": [_outcome(receipt=None)],
        },
    )
    with (
        patch(
            "kestrel_sovereign.cli_stop._stop_endpoint",
            return_value=("http://host/api/agent/stop", "secret"),
        ),
        patch("httpx.post", return_value=response),
    ):
        assert cmd_stop(_args()) == 1
    assert "stopped" in capsys.readouterr().out


def test_restart_uses_process_shutdown_not_cooperative_stop():
    from kestrel_sovereign import cli

    args = SimpleNamespace(name=None, force=False, startup_timeout=30)
    with (
        patch.object(cli, "cmd_shutdown", return_value=0) as shutdown,
        patch.object(cli, "cmd_start", return_value=0) as start,
        patch.object(cli, "cmd_stop", side_effect=AssertionError("cooperative")),
    ):
        assert cli.cmd_restart(args) == 0
    shutdown.assert_called_once_with(args)
    start.assert_called_once_with(args)
