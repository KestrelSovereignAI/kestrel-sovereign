"""CLI contract for cooperative Stop and separate process termination (#3160)."""

import inspect
import socket
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


def _attestation(*, port=8888, bind_host="0.0.0.0", connect_host="127.0.0.1"):
    project = Path("/project")
    return cli_stop._LocalProcessAttestation(
        project_root=project,
        pid_file=project / "logs" / ".host.pid",
        pid=123,
        port=port,
        started_at=100.0,
        bind_host=bind_host,
        connect_host=connect_host,
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


def _all_targets(endpoint=None, *, unreachable=()):
    targets = () if endpoint is None else (
        cli_stop._AllStopTarget(endpoint, True, "host fleet"),
    )
    return targets, tuple(unreachable)


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
        "kestrel_sovereign.cli_stop._all_stop_targets",
        return_value=_all_targets(unreachable=("host fleet",)),
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


@pytest.mark.parametrize(
    ("bind_host", "connect_host", "origin"),
    [
        ("0.0.0.0", "127.0.0.1", "http://127.0.0.1:8888"),
        ("::", "::1", "http://[::1]:8888"),
        ("192.0.2.10", "192.0.2.10", "http://192.0.2.10:8888"),
    ],
)
def test_attested_origin_honors_supported_bind_addresses(
    bind_host,
    connect_host,
    origin,
):
    attestation = _attestation(
        bind_host=bind_host,
        connect_host=connect_host,
    )
    assert cli_stop._connect_host(bind_host) == connect_host
    assert cli_stop._origin(attestation) == origin


def test_connected_socket_proof_binds_exact_flow_to_attested_server_pid():
    import psutil

    connection = MagicMock()
    connection.sock.getsockname.return_value = ("127.0.0.1", 54321)
    connection.sock.getpeername.return_value = ("127.0.0.1", 8888)
    server_flow = SimpleNamespace(
        status=psutil.CONN_ESTABLISHED,
        laddr=("127.0.0.1", 8888),
        raddr=("127.0.0.1", 54321),
    )
    unrelated = SimpleNamespace(
        status=psutil.CONN_ESTABLISHED,
        laddr=("127.0.0.1", 9999),
        raddr=("127.0.0.1", 54321),
    )
    with patch.object(
        cli_stop,
        "_process_connections",
        return_value=[unrelated, server_flow],
    ):
        assert cli_stop._connected_socket_is_owned_by(connection, _attestation())

    with (
        patch.object(
            cli_stop,
            "_process_connections",
            side_effect=[[], [server_flow]],
        ) as inspect_process,
        patch("time.sleep"),
    ):
        assert cli_stop._connected_socket_is_owned_by(connection, _attestation())
    assert inspect_process.call_count == 2


@pytest.mark.parametrize(
    ("connect_host", "resolved_host", "peer_host"),
    [
        ("localhost", "127.0.0.1", "127.0.0.1"),
        ("0:0:0:0:0:0:0:1", "::1", "::1"),
    ],
)
def test_connected_socket_proof_accepts_resolved_canonical_peer(
    connect_host,
    resolved_host,
    peer_host,
):
    import psutil

    connection = MagicMock()
    connection.sock.getsockname.return_value = (peer_host, 54321)
    connection.sock.getpeername.return_value = (peer_host, 8888)
    server_flow = SimpleNamespace(
        status=psutil.CONN_ESTABLISHED,
        laddr=(peer_host, 8888),
        raddr=(peer_host, 54321),
    )
    attestation = _attestation(
        bind_host=connect_host,
        connect_host=connect_host,
    )
    addrinfo = [
        (
            socket.AF_INET6 if ":" in resolved_host else socket.AF_INET,
            socket.SOCK_STREAM,
            socket.IPPROTO_TCP,
            "",
            (resolved_host, 8888),
        )
    ]
    with (
        patch("socket.getaddrinfo", return_value=addrinfo),
        patch.object(
            cli_stop,
            "_process_connections",
            return_value=[server_flow],
        ),
    ):
        assert cli_stop._connected_socket_is_owned_by(connection, attestation)

def test_connected_socket_proof_honors_ipv6_bind():
    import psutil

    connection = MagicMock()
    connection.sock.getsockname.return_value = ("::1", 54321, 0, 0)
    connection.sock.getpeername.return_value = ("::1", 8888, 0, 0)
    server_flow = SimpleNamespace(
        status=psutil.CONN_ESTABLISHED,
        laddr=("::1", 8888),
        raddr=("::1", 54321),
    )
    attestation = _attestation(bind_host="::", connect_host="::1")
    with patch.object(
        cli_stop,
        "_process_connections",
        return_value=[server_flow],
    ):
        assert cli_stop._connected_socket_is_owned_by(connection, attestation)


def test_listener_proof_uses_the_recorded_process_not_root_only_system_query():
    import psutil

    owned = SimpleNamespace(
        status=psutil.CONN_LISTEN,
        laddr=("127.0.0.1", 8888),
    )
    process = MagicMock()
    process.net_connections.return_value = [owned]
    with (
        patch("psutil.Process", return_value=process) as process_cls,
        patch(
            "psutil.net_connections",
            side_effect=AssertionError("root-only system query used"),
        ),
    ):
        assert cli_stop._listener_is_owned_by(8888, 123)
    process_cls.assert_called_once_with(123)
    process.net_connections.assert_called_once_with(kind="tcp")


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
        patch.object(cli_stop, "_listener_is_owned_by", return_value=True),
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
        bind_host="127.0.0.1",
        connect_host="127.0.0.1",
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
        patch.object(cli_stop, "_listener_is_owned_by", return_value=False),
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
    config.get_remote_agents = dict
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


def test_named_stop_quotes_the_agent_as_one_url_path_segment(tmp_path):
    from kestrel_sovereign import cli

    data_dir = tmp_path / "agents" / "emma bird"
    agent_config = SimpleNamespace(
        port=8801,
        resolve_data_dir=lambda _project: data_dir,
    )
    config = SimpleNamespace(host=SimpleNamespace(port=8888))
    config.get_local_agents = lambda: {"Emma bird/\N{SNOWMAN}": agent_config}
    config.get_remote_agents = lambda: {}
    with (
        patch.object(cli, "_get_project_dir", return_value=tmp_path),
        patch.object(cli.MultiAgentConfig, "load", return_value=config),
        patch.object(cli, "_operator_api_keys", return_value=("operator-key",)),
        patch.object(
            cli_stop,
            "read_anchor_agent_did_sync",
            return_value="did:emma",
        ),
        patch.object(
            cli_stop,
            "_attested_local_process",
            side_effect=[_attestation(), None],
        ),
        patch.object(
            cli_stop,
            "_agent_operator_key",
            return_value="key",
        ) as detect,
    ):
        resolved = cli_stop._stop_endpoint(
            _args(name="Emma bird/\N{SNOWMAN}")
        )

    encoded = "Emma%20bird%2F%E2%98%83"
    assert resolved is not None
    assert resolved.url.endswith(f"/api/agents/{encoded}/api/agent/stop")
    assert detect.call_args.args[1].endswith(
        f"/api/agents/{encoded}/api/agent/info"
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


def test_stop_all_resolution_includes_each_live_standalone_process(tmp_path):
    from kestrel_sovereign import cli

    alpha_dir = tmp_path / "agents" / "alpha"
    beta_dir = tmp_path / "agents" / "beta"
    alpha = SimpleNamespace(
        port=8801,
        resolve_data_dir=lambda _project: alpha_dir,
    )
    beta = SimpleNamespace(
        port=8802,
        resolve_data_dir=lambda _project: beta_dir,
    )
    config = SimpleNamespace(host=SimpleNamespace(port=8888, bind="0.0.0.0"))
    config.get_local_agents = lambda: {"Alpha": alpha, "Beta": beta}
    alpha_pid_file = cli_stop.ProcessManager.agent_pid_file(alpha_dir)
    alpha_attestation = cli_stop._LocalProcessAttestation(
        project_root=tmp_path,
        pid_file=alpha_pid_file,
        pid=101,
        port=8801,
        started_at=100.0,
        bind_host="0.0.0.0",
        connect_host="127.0.0.1",
    )
    alpha_endpoint = cli_stop._StopEndpoint(
        url="http://127.0.0.1:8801/api/agent/stop",
        api_key="alpha-key",
        attestation=alpha_attestation,
        expected_agent_id="did:alpha",
    )
    absent = SimpleNamespace(is_running=False)
    live = SimpleNamespace(is_running=True)

    def resolve(args):
        return alpha_endpoint if args.name == "Alpha" else None

    with (
        patch.object(cli, "_get_project_dir", return_value=tmp_path),
        patch.object(cli.MultiAgentConfig, "load", return_value=config),
        patch.object(
            cli_stop.ProcessManager,
            "read_pid_record",
            side_effect=[absent, live, absent],
        ),
        patch.object(cli_stop, "_stop_endpoint", side_effect=resolve),
    ):
        targets, unreachable = cli_stop._all_stop_targets(
            _args(name=None, all_agents=True)
        )

    assert targets == (
        cli_stop._AllStopTarget(alpha_endpoint, False, "Alpha"),
    )
    assert unreachable == ()


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
        "expected_agent_id": "did:emma",
    }
    assert post.call_args.kwargs["headers"] == {"X-API-Key": "secret"}
    output = capsys.readouterr().out
    assert "did:emma" in output
    assert "stopped" in output
    assert "receipt-1" in output


def test_stop_all_fans_out_to_host_and_live_standalone_processes(capsys):
    host = _endpoint(
        "http://host/api/host/stop",
        expected_agent_id=None,
    )
    standalone = _endpoint(
        "http://standalone/api/agent/stop",
        expected_agent_id="did:beta",
        port=8802,
    )
    host_payload = {
        "success": True,
        "state": "confirmed",
        "target_count": 1,
        "confirmed_count": 1,
        "unconfirmed_count": 0,
        "correlation_id": "cli:host",
        "stop_outcomes": [dict(
            _outcome(agent="did:alpha", scope="host"),
            correlation_id="cli:host",
        )],
    }
    standalone_payload = {
        "success": True,
        "stop_outcomes": [dict(
            _outcome(agent="did:beta"),
            correlation_id="cli:beta",
        )],
    }
    with (
        patch(
            "kestrel_sovereign.cli_stop._all_stop_targets",
            return_value=(
                (
                    cli_stop._AllStopTarget(host, True, "host fleet"),
                    cli_stop._AllStopTarget(standalone, False, "Beta"),
                ),
                (),
            ),
        ),
        patch(
            "kestrel_sovereign.cli_stop._operation_id",
            side_effect=["cli:host", "cli:beta"],
        ),
        patch(
            "kestrel_sovereign.cli_stop._attestation_is_current",
            return_value=True,
        ),
        patch(
            "kestrel_sovereign.cli_stop._local_request",
            side_effect=[
                _response(payload=host_payload),
                _response(payload=standalone_payload),
            ],
        ) as post,
    ):
        assert cmd_stop(_args(name=None, all_agents=True)) == 0

    assert post.call_count == 2
    assert post.call_args_list[1].kwargs["json"]["expected_agent_id"] == "did:beta"
    output = capsys.readouterr().out
    assert "did:alpha" in output and "did:beta" in output


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
            "kestrel_sovereign.cli_stop._all_stop_targets",
            return_value=_all_targets(_endpoint(
                "http://host/api/host/stop", expected_agent_id=None
            )),
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
        patch("kestrel_sovereign.cli_stop._all_stop_targets", return_value=_all_targets(
            _endpoint("http://host/api/host/stop", expected_agent_id=None)
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
        patch("kestrel_sovereign.cli_stop._all_stop_targets", return_value=_all_targets(
            _endpoint("http://host/api/host/stop", expected_agent_id=None)
        )),
        patch("kestrel_sovereign.cli_stop._operation_id", return_value="cli:fixed"),
        patch("kestrel_sovereign.cli_stop._attestation_is_current", return_value=True),
        patch("kestrel_sovereign.cli_stop._local_request", return_value=response),
    ):
        assert cmd_stop(_args(name=None, all_agents=True)) == 1
    assert "inconsistent" in capsys.readouterr().out


def test_fleet_success_rejects_cross_wired_target_identities(capsys):
    first = _outcome(agent="did:alpha", scope="host")
    second = _outcome(agent="did:beta", scope="host")
    first["resolved_target"] = "did:beta"
    second["resolved_target"] = "did:alpha"
    response = _response(payload={
        "success": True,
        "state": "confirmed",
        "target_count": 2,
        "confirmed_count": 2,
        "unconfirmed_count": 0,
        "correlation_id": "cli:fixed",
        "stop_outcomes": [first, second],
    })
    with (
        patch(
            "kestrel_sovereign.cli_stop._all_stop_targets",
            return_value=_all_targets(_endpoint(
                "http://host/api/host/stop", expected_agent_id=None
            )),
        ),
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
            "kestrel_sovereign.cli_stop._all_stop_targets",
            return_value=_all_targets(_endpoint(
                "http://host/api/host/stop", expected_agent_id=None
            )),
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
