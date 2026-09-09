"""CLI contract for cooperative Stop and separate process termination (#3160)."""

import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest

from kestrel_sovereign import cli_stop
from kestrel_sovereign.cli import build_parser
from kestrel_sovereign.cli_stop import cmd_stop


def _args(*, name="Emma", all_agents=False, reason=None):
    return SimpleNamespace(name=name, all=all_agents, reason=reason)


def _response(*, status=200, payload):
    response = MagicMock(status_code=status, text="")
    response.json.return_value = payload
    return response


def _outcome(
    disposition="stopped",
    *,
    receipt="receipt-1",
    agent="did:emma",
    scope="agent",
):
    return {
        "scope": scope,
        "requested_target": agent if scope == "agent" else None,
        "resolved_target": agent,
        "agent_id": agent,
        "disposition": disposition,
        "correlation_id": "cli:fixed",
        "receipt_id": receipt,
    }


def _attestation(*, port=8888):
    project = Path("/project")
    return cli_stop._LocalProcessAttestation(
        project_root=project,
        pid_file=project / "logs" / ".host.pid",
        pid=123,
        port=port,
        started_at=100.0,
    )


def _endpoint(
    url="http://host/api/agent/stop",
    *,
    key="secret",
    expected_agent_id="did:emma",
    port=8888,
):
    return cli_stop._StopEndpoint(
        url=url,
        api_key=key,
        attestation=_attestation(port=port),
        expected_agent_id=expected_agent_id,
    )


def test_parser_separates_cooperative_stop_from_process_termination():
    stop = build_parser().parse_args(["stop", "Emma", "--reason", "andon"])
    assert stop.command == "stop"
    assert stop.name == "Emma"
    assert stop.reason == "andon"
    assert stop.all is False
    assert not hasattr(stop, "force")

    fleet = build_parser().parse_args(["stop", "--all"])
    assert fleet.all is True
    termination = build_parser().parse_args(["terminate", "Emma", "--force"])
    assert termination.command == "terminate"
    assert termination.name == "Emma"
    assert termination.force is True


def test_stop_requires_exactly_one_agent_or_all(capsys):
    assert cmd_stop(_args(name=None)) == 2
    assert cmd_stop(_args(name="Emma", all_agents=True)) == 2
    assert "exactly one" in capsys.readouterr().out


def test_unreachable_fleet_stop_names_all_agents_not_none(capsys):
    with patch(
        "kestrel_sovereign.cli_stop._stop_endpoint",
        return_value=None,
    ):
        assert cmd_stop(_args(name=None, all_agents=True)) == 1

    output = capsys.readouterr().out
    assert "all agents" in output
    assert "None" not in output


def test_local_request_attests_the_connected_socket_before_sending_credentials():
    connection = MagicMock()
    response = MagicMock(status=200)
    response.read.return_value = b'{"ok": true}'
    connection.getresponse.return_value = response
    with (
        patch("http.client.HTTPConnection", return_value=connection),
        patch.object(cli_stop, "_attestation_is_current", return_value=True),
        patch.object(
            cli_stop,
            "_connected_socket_is_owned_by",
            return_value=True,
        ) as connected_owner,
    ):
        result = cli_stop._local_request(
            "GET",
            "http://127.0.0.1:8888/probe",
            attestation=_attestation(),
            headers={"X-API-Key": "secret"},
        )

    connection.connect.assert_called_once_with()
    connected_owner.assert_called_once_with(connection, _attestation())
    connection.request.assert_called_once_with(
        "GET",
        "/probe",
        body=None,
        headers={"X-API-Key": "secret"},
    )
    assert result.json() == {"ok": True}


def test_local_request_never_sends_when_connected_socket_owner_is_unproven():
    connection = MagicMock()
    with (
        patch("http.client.HTTPConnection", return_value=connection),
        patch.object(cli_stop, "_attestation_is_current", return_value=True),
        patch.object(
            cli_stop,
            "_connected_socket_is_owned_by",
            return_value=False,
        ),
        pytest.raises(cli_stop._LocalRequestError, match="does not belong"),
    ):
        cli_stop._local_request(
            "POST",
            "http://127.0.0.1:8888/api/agent/stop",
            attestation=_attestation(),
            headers={"X-API-Key": "secret"},
        )
    connection.request.assert_not_called()


def test_connected_socket_proof_binds_exact_flow_to_attested_server_pid():
    import psutil

    connection = MagicMock()
    connection.sock.getsockname.return_value = ("127.0.0.1", 54321)
    connection.sock.getpeername.return_value = ("127.0.0.1", 8888)
    server_flow = SimpleNamespace(
        status=psutil.CONN_ESTABLISHED,
        laddr=("127.0.0.1", 8888),
        raddr=("127.0.0.1", 54321),
        pid=123,
    )
    unrelated = SimpleNamespace(
        status=psutil.CONN_ESTABLISHED,
        laddr=("127.0.0.1", 9999),
        raddr=("127.0.0.1", 54321),
        pid=999,
    )
    with patch("psutil.net_connections", return_value=[unrelated, server_flow]):
        assert cli_stop._connected_socket_is_owned_by(connection, _attestation())

    server_flow.pid = None
    with patch("psutil.net_connections", return_value=[server_flow]):
        assert not cli_stop._connected_socket_is_owned_by(connection, _attestation())


def test_listener_proof_rejects_an_unknown_co_listener():
    import psutil

    owned = SimpleNamespace(
        status=psutil.CONN_LISTEN,
        laddr=("127.0.0.1", 8888),
        pid=123,
    )
    unknown = SimpleNamespace(
        status=psutil.CONN_LISTEN,
        laddr=("0.0.0.0", 8888),
        pid=None,
    )
    with patch("psutil.net_connections", return_value=[owned, unknown]):
        assert not cli_stop._exclusive_listener_is_owned_by(8888, 123)


def test_attestation_requires_matching_live_project_process_and_listener(tmp_path):
    from kestrel_sovereign.multi_agent.process_manager import PidStatus

    pid_file = tmp_path / "logs" / ".host.pid"
    record = SimpleNamespace(
        status=PidStatus.LIVE,
        pid=123,
        root=str(tmp_path),
        port=8888,
        started_at=100.0,
    )
    with (
        patch.object(cli_stop.ProcessManager, "read_pid_record", return_value=record),
        patch.object(cli_stop, "_exclusive_listener_is_owned_by", return_value=True),
    ):
        attestation = cli_stop._attested_local_process(
            tmp_path,
            pid_file=pid_file,
            port=8888,
        )

    assert attestation == cli_stop._LocalProcessAttestation(
        project_root=tmp_path.resolve(),
        pid_file=pid_file,
        pid=123,
        port=8888,
        started_at=100.0,
    )


def test_attestation_rejects_listener_owned_by_another_process(tmp_path):
    from kestrel_sovereign.multi_agent.process_manager import PidStatus

    record = SimpleNamespace(
        status=PidStatus.LIVE,
        pid=123,
        root=str(tmp_path),
        port=8888,
        started_at=100.0,
    )
    with (
        patch.object(cli_stop.ProcessManager, "read_pid_record", return_value=record),
        patch.object(cli_stop, "_exclusive_listener_is_owned_by", return_value=False),
    ):
        assert cli_stop._attested_local_process(
            tmp_path,
            pid_file=tmp_path / ".host.pid",
            port=8888,
        ) is None


def test_host_stop_resolution_probes_live_host_for_accepted_sovereign_key(tmp_path):
    from kestrel_sovereign import cli

    config = SimpleNamespace(host=SimpleNamespace(port=8888))
    config.get_local_agents = lambda: {}
    config.get_remote_agents = lambda: {}
    with (
        patch.object(cli, "_get_project_dir", return_value=tmp_path),
        patch.object(cli.MultiAgentConfig, "load", return_value=config),
        patch.object(
            cli,
            "_operator_api_keys",
            return_value=("exported-secret", "file-secret"),
        ),
        patch(
            "kestrel_sovereign.cli_stop._attested_local_process",
            return_value=_attestation(),
        ),
        patch(
            "kestrel_sovereign.cli_stop._host_operator_key",
            return_value="exported-secret",
        ) as detect,
    ):
        resolved = cli_stop._stop_endpoint(_args(name=None, all_agents=True))
    assert resolved == _endpoint(
        "http://127.0.0.1:8888/api/host/stop",
        key="exported-secret",
        expected_agent_id=None,
    )
    detect.assert_called_once_with(
        _attestation(),
        ("exported-secret", "file-secret"),
    )


def test_host_operator_key_uses_authenticated_read_only_probe():
    with (
        patch(
            "kestrel_sovereign.cli_stop._attestation_is_current",
            return_value=True,
        ),
        patch("kestrel_sovereign.cli_stop._local_request") as request,
    ):
        request.side_effect = [
            _response(status=401, payload={}),
            _response(status=200, payload={"can_stop": True}),
        ]

        assert cli_stop._host_operator_key(
            _attestation(),
            ("stale", "accepted"),
        ) == "accepted"

    assert [call.args[:2] for call in request.call_args_list] == [
        ("GET", "http://127.0.0.1:8888/api/host/stop/status"),
        ("GET", "http://127.0.0.1:8888/api/host/stop/status"),
    ]
    assert request.call_args_list[0].kwargs["headers"] == {"X-API-Key": "stale"}
    assert request.call_args_list[1].kwargs["headers"] == {
        "X-API-Key": "accepted"
    }


def test_operator_key_is_not_sent_after_process_attestation_changes():
    with (
        patch(
            "kestrel_sovereign.cli_stop._attestation_is_current",
            return_value=False,
        ),
        patch(
            "kestrel_sovereign.cli_stop._local_request",
            side_effect=AssertionError("secret crossed an unattested socket"),
        ),
    ):
        assert cli_stop._host_operator_key(_attestation(), ("secret",)) is None


def test_agent_probe_requires_the_expected_durable_identity():
    with (
        patch(
            "kestrel_sovereign.cli_stop._attestation_is_current",
            return_value=True,
        ),
        patch(
            "kestrel_sovereign.cli_stop._local_request",
            return_value=_response(
                status=200,
                payload={"agent_id": "did:other", "status": "healthy"},
            ),
        ),
    ):
        assert cli_stop._agent_operator_key(
            _attestation(),
            "http://127.0.0.1:8888/api/agent/info",
            ("secret",),
            expected_agent_id="did:emma",
        ) is None


def test_named_stop_resolution_delegates_agent_routing_to_live_http_probe(tmp_path):
    from kestrel_sovereign import cli

    data_dir = tmp_path / "agents" / "emma"
    agent_config = SimpleNamespace(
        port=8801,
        resolve_data_dir=lambda _project: data_dir,
    )
    config = SimpleNamespace(host=SimpleNamespace(port=8888))
    config.get_local_agents = lambda: {"Emma": agent_config}
    config.get_remote_agents = lambda: {}
    with (
        patch.object(cli, "_get_project_dir", return_value=tmp_path),
        patch.object(cli.MultiAgentConfig, "load", return_value=config),
        patch.object(cli, "_operator_api_keys", return_value=("operator-key",)),
        patch(
            "kestrel_sovereign.cli_stop.read_anchor_agent_did_sync",
            return_value="did:emma",
        ),
        patch(
            "kestrel_sovereign.cli_stop._attested_local_process",
            side_effect=[_attestation(), None],
        ),
        patch(
            "kestrel_sovereign.cli_stop._agent_operator_key",
            return_value="key",
        ) as detect,
    ):
        resolved = cli_stop._stop_endpoint(_args())
    assert resolved == _endpoint(
        "http://127.0.0.1:8888/api/agents/Emma/api/agent/stop",
        key="key",
    )
    detect.assert_called_once_with(
        _attestation(),
        "http://127.0.0.1:8888/api/agents/Emma/api/agent/info",
        ("operator-key",),
        expected_agent_id="did:emma",
    )


def test_remote_stop_never_sends_the_local_sovereign_key(tmp_path):
    from kestrel_sovereign import cli

    remote = SimpleNamespace(url="https://peer.example")
    config = SimpleNamespace(host=SimpleNamespace(port=8888))
    config.get_local_agents = lambda: {}
    config.get_remote_agents = lambda: {"Peer": remote}
    with (
        patch.object(cli, "_get_project_dir", return_value=tmp_path),
        patch.object(cli.MultiAgentConfig, "load", return_value=config),
        patch(
            "kestrel_sovereign.cli_stop._host_operator_key",
            side_effect=AssertionError("local sovereign key disclosure"),
        ),
    ):
        assert cli_stop._stop_endpoint(_args(name="Peer")) is None


def test_cooperative_stop_module_has_no_process_mutation_door():
    source = inspect.getsource(cli_stop)
    assert "stop_agent(" not in source
    assert "kill_process(" not in source
    assert "terminate_agent(" not in source
    assert "cmd_terminate(" not in source


def test_named_stop_posts_only_intent_and_prints_receipted_outcome(capsys):
    response = _response(
        payload={"success": True, "stop_outcomes": [_outcome()]},
    )
    with (
        patch(
            "kestrel_sovereign.cli_stop._stop_endpoint",
            return_value=_endpoint(),
        ),
        patch(
            "kestrel_sovereign.cli_stop._operation_id",
            return_value="cli:fixed",
        ),
        patch(
            "kestrel_sovereign.cli_stop._attestation_is_current",
            return_value=True,
        ),
        patch(
            "kestrel_sovereign.cli_stop._local_request",
            return_value=response,
        ) as post,
    ):
        assert cmd_stop(_args(reason="andon")) == 0

    post.assert_called_once()
    assert post.call_args.args == ("POST", "http://host/api/agent/stop")
    assert post.call_args.kwargs["json"] == {
        "correlation_id": "cli:fixed",
        "reason": "andon",
    }
    assert post.call_args.kwargs["headers"] == {"X-API-Key": "secret"}
    output = capsys.readouterr().out
    assert "did:emma" in output
    assert "stopped" in output
    assert "receipt-1" in output


def test_successful_fleet_stop_requires_complete_host_evidence(capsys):
    outcomes = [
        _outcome(agent="did:alpha", scope="host"),
        _outcome(agent="did:beta", scope="host"),
    ]
    response = _response(
        payload={
            "success": True,
            "state": "confirmed",
            "target_count": 2,
            "confirmed_count": 2,
            "unconfirmed_count": 0,
            "correlation_id": "cli:fixed",
            "stop_outcomes": outcomes,
        },
    )
    with (
        patch(
            "kestrel_sovereign.cli_stop._stop_endpoint",
            return_value=_endpoint(
                "http://host/api/host/stop",
                expected_agent_id=None,
            ),
        ),
        patch(
            "kestrel_sovereign.cli_stop._operation_id",
            return_value="cli:fixed",
        ),
        patch(
            "kestrel_sovereign.cli_stop._attestation_is_current",
            return_value=True,
        ),
        patch(
            "kestrel_sovereign.cli_stop._local_request",
            return_value=response,
        ),
    ):
        assert cmd_stop(_args(name=None, all_agents=True)) == 0

    output = capsys.readouterr().out
    assert "did:alpha" in output and "did:beta" in output


@pytest.mark.parametrize(
    ("mutation", "value"),
    [
        ("state", "partial"),
        ("confirmed_count", True),
        ("unconfirmed_count", 0.0),
    ],
)
def test_fleet_success_rejects_noncanonical_envelope_fields(
    mutation,
    value,
    capsys,
):
    payload = {
        "success": True,
        "state": "confirmed",
        "target_count": 1,
        "confirmed_count": 1,
        "unconfirmed_count": 0,
        "correlation_id": "cli:fixed",
        "stop_outcomes": [_outcome(agent="did:alpha", scope="host")],
    }
    payload[mutation] = value
    response = _response(payload=payload)
    with (
        patch("kestrel_sovereign.cli_stop._stop_endpoint", return_value=_endpoint(
            "http://host/api/host/stop", expected_agent_id=None
        )),
        patch("kestrel_sovereign.cli_stop._operation_id", return_value="cli:fixed"),
        patch("kestrel_sovereign.cli_stop._attestation_is_current", return_value=True),
        patch("kestrel_sovereign.cli_stop._local_request", return_value=response),
    ):
        assert cmd_stop(_args(name=None, all_agents=True)) == 1
    assert "inconsistent" in capsys.readouterr().out


def test_fleet_success_rejects_duplicate_targets_and_mixed_receipts(capsys):
    first = _outcome(agent="did:alpha", scope="host")
    duplicate = dict(first, receipt_id="receipt-2")
    response = _response(
        payload={
            "success": True,
            "state": "confirmed",
            "target_count": 2,
            "confirmed_count": 2,
            "unconfirmed_count": 0,
            "correlation_id": "cli:fixed",
            "stop_outcomes": [first, duplicate],
        }
    )
    with (
        patch("kestrel_sovereign.cli_stop._stop_endpoint", return_value=_endpoint(
            "http://host/api/host/stop", expected_agent_id=None
        )),
        patch("kestrel_sovereign.cli_stop._operation_id", return_value="cli:fixed"),
        patch("kestrel_sovereign.cli_stop._attestation_is_current", return_value=True),
        patch("kestrel_sovereign.cli_stop._local_request", return_value=response),
    ):
        assert cmd_stop(_args(name=None, all_agents=True)) == 1
    assert "inconsistent" in capsys.readouterr().out


def test_success_rejects_whitespace_only_typed_evidence(capsys):
    response = _response(
        payload={
            "success": True,
            "stop_outcomes": [
                dict(
                    _outcome(),
                    agent_id=" ",
                    resolved_target=" ",
                    receipt_id=" ",
                )
            ],
        }
    )
    with (
        patch("kestrel_sovereign.cli_stop._stop_endpoint", return_value=_endpoint()),
        patch("kestrel_sovereign.cli_stop._operation_id", return_value="cli:fixed"),
        patch("kestrel_sovereign.cli_stop._attestation_is_current", return_value=True),
        patch("kestrel_sovereign.cli_stop._local_request", return_value=response),
    ):
        assert cmd_stop(_args()) == 1
    assert "inconsistent" in capsys.readouterr().out


def test_success_with_malformed_or_mismatched_outcomes_fails_closed(capsys):
    response = _response(
        payload={
            "success": True,
            "stop_outcomes": [_outcome(), "not-an-outcome"],
        },
    )
    with (
        patch(
            "kestrel_sovereign.cli_stop._stop_endpoint",
            return_value=_endpoint(),
        ),
        patch(
            "kestrel_sovereign.cli_stop._operation_id",
            return_value="cli:fixed",
        ),
        patch(
            "kestrel_sovereign.cli_stop._attestation_is_current",
            return_value=True,
        ),
        patch(
            "kestrel_sovereign.cli_stop._local_request",
            return_value=response,
        ),
    ):
        assert cmd_stop(_args()) == 1

    assert "inconsistent" in capsys.readouterr().out


def test_changed_process_attestation_prevents_credential_dispatch(capsys):
    with (
        patch(
            "kestrel_sovereign.cli_stop._stop_endpoint",
            return_value=_endpoint(),
        ),
        patch(
            "kestrel_sovereign.cli_stop._attestation_is_current",
            return_value=False,
        ),
        patch(
            "kestrel_sovereign.cli_stop._local_request",
            side_effect=AssertionError("credential crossed an untrusted socket"),
        ),
    ):
        assert cmd_stop(_args()) == 1

    assert "identity changed" in capsys.readouterr().out


def test_refused_agent_error_prints_typed_outcome_and_fails(capsys):
    refused = _outcome("refused", receipt="receipt-refused")
    response = _response(
        status=503,
        payload={"error": {"details": [refused]}},
    )
    with (
        patch(
            "kestrel_sovereign.cli_stop._stop_endpoint",
            return_value=_endpoint(),
        ),
        patch(
            "kestrel_sovereign.cli_stop._attestation_is_current",
            return_value=True,
        ),
        patch("kestrel_sovereign.cli_stop._local_request", return_value=response),
    ):
        assert cmd_stop(_args()) == 1

    output = capsys.readouterr().out
    assert "did:emma" in output
    assert "refused" in output
    assert "receipt-refused" in output


def test_malformed_error_details_use_indeterminate_path_without_traceback(capsys):
    response = _response(
        status=422,
        payload={"error": {"details": ["malformed"]}},
    )
    with (
        patch("kestrel_sovereign.cli_stop._stop_endpoint", return_value=_endpoint()),
        patch("kestrel_sovereign.cli_stop._attestation_is_current", return_value=True),
        patch("kestrel_sovereign.cli_stop._local_request", return_value=response),
    ):
        assert cmd_stop(_args()) == 1
    assert "indeterminate" in capsys.readouterr().out


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
            return_value=_endpoint(
                "http://host/api/host/stop",
                expected_agent_id=None,
            ),
        ),
        patch(
            "kestrel_sovereign.cli_stop._attestation_is_current",
            return_value=True,
        ),
        patch("kestrel_sovereign.cli_stop._local_request", return_value=response),
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
            return_value=_endpoint(),
        ),
        patch(
            "kestrel_sovereign.cli_stop._attestation_is_current",
            return_value=True,
        ),
        patch(
            "kestrel_sovereign.cli_stop._local_request",
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
            return_value=_endpoint(),
        ),
        patch(
            "kestrel_sovereign.cli_stop._attestation_is_current",
            return_value=True,
        ),
        patch("kestrel_sovereign.cli_stop._local_request", return_value=response),
    ):
        assert cmd_stop(_args()) == 1
    assert "stopped" in capsys.readouterr().out


def test_restart_uses_process_termination_not_cooperative_stop():
    from kestrel_sovereign import cli

    args = SimpleNamespace(name=None, force=False, startup_timeout=30)
    with (
        patch.object(cli, "cmd_terminate", return_value=0) as terminate,
        patch.object(cli, "cmd_start", return_value=0) as start,
        patch.object(cli, "cmd_stop", side_effect=AssertionError("cooperative")),
    ):
        assert cli.cmd_restart(args) == 0
    terminate.assert_called_once_with(args)
    start.assert_called_once_with(args)
