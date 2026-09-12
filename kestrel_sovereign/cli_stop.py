"""Cooperative ``kestrel stop`` commands.

Stop is an andon cord for in-flight cognition.  It calls the authenticated
agent/host Stop APIs and never enters the process lifecycle manager; process
teardown is the separate ``kestrel terminate`` command.
"""

from __future__ import annotations

import http.client
import ipaddress
import json as json_module
import os
import socket
import stat
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlsplit

from kestrel_sovereign.identity.local_anchor import (
    AgentDIDLookupMode,
    read_anchor_agent_did_sync,
)
from kestrel_sovereign.multi_agent.config import MULTI_AGENT_CONFIG_FILENAME
from kestrel_sovereign.multi_agent.process_manager import PidStatus, ProcessManager
from kestrel_sovereign.multi_agent.route_name import encode_agent_route_name

_CONFIRMED_DISPOSITIONS = frozenset({"stopped", "already_complete"})


@dataclass(frozen=True, slots=True)
class _LocalProcessAttestation:
    """OS-backed identity of the one local process allowed to receive a key."""

    project_root: Path
    pid_file: Path
    pid: int
    port: int
    started_at: float
    bind_host: str
    connect_host: str


@dataclass(frozen=True, slots=True)
class _StopEndpoint:
    url: str
    api_key: str
    attestation: _LocalProcessAttestation
    expected_agent_id: str | None = None


@dataclass(frozen=True, slots=True)
class _AllStopTarget:
    endpoint: _StopEndpoint
    all_agents: bool
    label: str


def _operation_id() -> str:
    return f"cli:{uuid.uuid4()}"


@dataclass(frozen=True, slots=True)
class _LocalResponse:
    status_code: int
    body: bytes

    @property
    def text(self) -> str:
        return self.body.decode("utf-8", errors="replace")

    def json(self) -> Any:
        return json_module.loads(self.body)


class _LocalRequestError(RuntimeError):
    """The attested local control connection could not complete safely."""


def _address_parts(address: object) -> tuple[str, int] | None:
    try:
        host = getattr(address, "ip", None)
        port = getattr(address, "port", None)
        if host is None or port is None:
            host = address[0]  # type: ignore[index]
            port = address[1]  # type: ignore[index]
        host = str(host)
        port = int(port)
    except (AttributeError, IndexError, TypeError, ValueError):
        return None
    return host, port


def _canonical_ip(host: str) -> str:
    """Canonicalize a numeric socket address without discarding its family."""

    bare_host = host.split("%", 1)[0]
    try:
        return ipaddress.ip_address(bare_host).compressed
    except ValueError:
        return host.casefold()


def _resolved_tcp_addresses(host: str, port: int) -> frozenset[tuple[str, int]]:
    """Resolve a configured host to the numeric peers a socket may report."""

    try:
        addresses = socket.getaddrinfo(
            host,
            port,
            family=socket.AF_UNSPEC,
            type=socket.SOCK_STREAM,
            proto=socket.IPPROTO_TCP,
        )
    except OSError:
        return frozenset()
    return frozenset(
        (_canonical_ip(str(sockaddr[0])), int(sockaddr[1]))
        for _family, _type, _proto, _canonname, sockaddr in addresses
    )


def _canonical_socket_address(address: object) -> tuple[str, int] | None:
    parts = _address_parts(address)
    if parts is None:
        return None
    return _canonical_ip(parts[0]), parts[1]


def _connect_host(bind_host: str) -> str:
    """Select an address accepted by a server's configured bind."""

    bind_host = str(bind_host).strip()
    if not bind_host:
        raise ValueError("server bind host must be concrete")
    if bind_host == "0.0.0.0":
        return "127.0.0.1"
    if bind_host == "::":
        return "::1"
    return bind_host


def _origin(attestation: _LocalProcessAttestation) -> str:
    host = attestation.connect_host
    rendered = f"[{host}]" if ":" in host and not host.startswith("[") else host
    return f"http://{rendered}:{attestation.port}"


def _process_connections(pid: int) -> list[Any]:
    """Inspect one recorded process without macOS's root-only fleet query."""

    import psutil

    process = psutil.Process(pid)
    return list(process.net_connections(kind="tcp"))


def _listener_is_owned_by(port: int, pid: int) -> bool:
    """Fail closed unless the recorded process owns the configured listener."""

    try:
        import psutil

        listeners = []
        for connection in _process_connections(pid):
            local = _address_parts(connection.laddr)
            if (
                connection.status == psutil.CONN_LISTEN
                and local is not None
                and local[1] == port
            ):
                listeners.append(connection)
    except Exception:  # noqa: BLE001 - inability to prove ownership is refusal
        return False
    return bool(listeners)


def _connected_socket_is_owned_by(
    connection: http.client.HTTPConnection,
    attestation: _LocalProcessAttestation,
) -> bool:
    """Bind the exact connected TCP flow to the recorded server process."""

    sock = connection.sock
    if sock is None:
        return False
    try:
        client = _address_parts(sock.getsockname())
        server = _address_parts(sock.getpeername())
    except OSError:
        return False
    expected_servers = _resolved_tcp_addresses(
        attestation.connect_host,
        attestation.port,
    )
    canonical_server = _canonical_socket_address(server)
    if client is None or canonical_server not in expected_servers:
        return False

    # The handshake can complete just before the server event loop accepts it.
    # No HTTP bytes (and therefore no credential) are sent during this bounded
    # wait for the exact server-side socket to acquire a process owner.
    for _ in range(20):
        try:
            import psutil

            peers = []
            for candidate in _process_connections(attestation.pid):
                if (
                    candidate.status == psutil.CONN_ESTABLISHED
                    and _canonical_socket_address(candidate.laddr)
                    == canonical_server
                    and _canonical_socket_address(candidate.raddr)
                    == _canonical_socket_address(client)
                ):
                    peers.append(candidate)
        except Exception:  # noqa: BLE001 - inability to prove ownership is refusal
            return False
        if peers:
            return True
        time.sleep(0.005)
    return False


def _local_request(
    method: str,
    url: str,
    *,
    attestation: _LocalProcessAttestation,
    json: Any = None,
    headers: dict[str, str] | None = None,
    timeout: float = 2.0,
) -> _LocalResponse:
    """Send credentials only over an exact, process-attested TCP connection."""

    parsed = urlsplit(url)
    if (
        parsed.scheme != "http"
        or parsed.hostname is None
        or parsed.hostname.casefold() != attestation.connect_host.casefold()
        or parsed.port != attestation.port
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise _LocalRequestError("local control URL is not attested loopback")
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    request_headers = dict(headers or {})
    body = None
    if json is not None:
        body = json_module.dumps(json).encode("utf-8")
        request_headers.setdefault("Content-Type", "application/json")

    connection = http.client.HTTPConnection(
        attestation.connect_host,
        attestation.port,
        timeout=timeout,
    )
    try:
        connection.connect()
        if (
            not _attestation_is_current(attestation)
            or not _connected_socket_is_owned_by(connection, attestation)
        ):
            raise _LocalRequestError(
                "local control connection does not belong to the attested process"
            )
        connection.request(method, path, body=body, headers=request_headers)
        response = connection.getresponse()
        return _LocalResponse(response.status, response.read())
    except (TimeoutError, OSError, http.client.HTTPException) as error:
        raise _LocalRequestError(str(error)) from error
    finally:
        connection.close()


def _attested_local_process(
    project_root: Path,
    *,
    pid_file: Path,
    port: int,
    bind_host: str = "127.0.0.1",
) -> _LocalProcessAttestation | None:
    """Bind a configured port to this project's live, recorded process."""

    project_root = project_root.resolve()
    record = ProcessManager.read_pid_record(pid_file)
    if (
        record.status is not PidStatus.LIVE
        or record.pid is None
        or record.root is None
        or record.port != port
        or record.started_at is None
    ):
        return None
    try:
        recorded_root = Path(record.root).resolve()
    except (OSError, RuntimeError):
        return None
    if recorded_root != project_root:
        return None
    try:
        connect_host = _connect_host(bind_host)
    except ValueError:
        return None
    # A PID file establishes process identity, not socket ownership. Refuse if
    # the configured listener cannot be attributed to that exact process.
    if not _listener_is_owned_by(port, record.pid):
        return None
    return _LocalProcessAttestation(
        project_root=project_root,
        pid_file=pid_file,
        pid=record.pid,
        port=port,
        started_at=record.started_at,
        bind_host=bind_host,
        connect_host=connect_host,
    )


def _attestation_is_current(attestation: _LocalProcessAttestation) -> bool:
    current = _attested_local_process(
        attestation.project_root,
        pid_file=attestation.pid_file,
        port=attestation.port,
        bind_host=attestation.bind_host,
    )
    return current == attestation


def _host_operator_key(
    attestation: _LocalProcessAttestation,
    candidates: tuple[str, ...],
) -> str | None:
    """Identify the credential accepted by the live host without mutating it."""

    import httpx

    probe_url = f"{_origin(attestation)}/api/host/stop/status"
    for candidate in candidates:
        if not _attestation_is_current(attestation):
            return None
        try:
            response = _local_request(
                "GET",
                probe_url,
                attestation=attestation,
                headers={"X-API-Key": candidate},
                timeout=2.0,
            )
        except (httpx.RequestError, _LocalRequestError):
            continue
        if response.status_code != 200:
            continue
        try:
            payload = response.json()
        except ValueError:
            continue
        if isinstance(payload, dict) and payload.get("can_stop") is True:
            return candidate
    return None


def _agent_operator_key(
    attestation: _LocalProcessAttestation,
    info_url: str,
    candidates: tuple[str, ...],
    *,
    expected_agent_id: str,
) -> str | None:
    """Choose a key only when the authenticated route proves the target DID."""

    import httpx

    for candidate in candidates:
        if not candidate or not _attestation_is_current(attestation):
            return None
        try:
            response = _local_request(
                "GET",
                info_url,
                attestation=attestation,
                headers={"X-API-Key": candidate},
                timeout=2.0,
            )
        except (httpx.RequestError, _LocalRequestError):
            continue
        if response.status_code != 200:
            continue
        try:
            payload = response.json()
        except ValueError:
            continue
        if (
            isinstance(payload, dict)
            and payload.get("agent_id") == expected_agent_id
        ):
            return candidate
    return None


def _standalone_bootstrap_key(
    attestation: _LocalProcessAttestation,
) -> str | None:
    """Reuse or read the key belonging to this exact standalone process."""

    import httpx

    if not _attestation_is_current(attestation):
        return None
    cached = _read_bootstrap_key_cache(attestation)
    if cached is not None:
        return cached
    try:
        response = _local_request(
            "GET",
            f"{_origin(attestation)}/api/auth/key",
            attestation=attestation,
            timeout=2.0,
        )
    except (httpx.RequestError, _LocalRequestError):
        return None
    if response.status_code != 200 or not _attestation_is_current(attestation):
        return None
    try:
        payload = response.json()
    except ValueError:
        return None
    key = payload.get("key") if isinstance(payload, dict) else None
    if not isinstance(key, str) or not key:
        return None
    return key if _write_bootstrap_key_cache(attestation, key) else None


def _bootstrap_key_cache_path(attestation: _LocalProcessAttestation) -> Path:
    return attestation.pid_file.with_name(f"{attestation.pid_file.name}.stop-key.json")


def _read_bootstrap_key_cache(
    attestation: _LocalProcessAttestation,
) -> str | None:
    """Read a private cache only when it is bound to the attested process."""

    path = _bootstrap_key_cache_path(attestation)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError:
        return None
    try:
        metadata = os.fstat(fd)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_mode & 0o077
            or metadata.st_uid != os.getuid()
        ):
            return None
        with os.fdopen(fd, encoding="utf-8") as stream:
            fd = -1
            encoded = stream.read(16_385)
        if len(encoded) > 16_384:
            return None
        payload = json_module.loads(encoded)
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    finally:
        if fd >= 0:
            os.close(fd)
    if not isinstance(payload, dict):
        return None
    key = payload.get("key")
    if (
        payload.get("pid") != attestation.pid
        or payload.get("started_at") != attestation.started_at
        or not isinstance(key, str)
        or not key
    ):
        return None
    return key


def _write_bootstrap_key_cache(
    attestation: _LocalProcessAttestation,
    key: str,
) -> bool:
    """Atomically preserve a process-bound credential with owner-only mode."""

    path = _bootstrap_key_cache_path(attestation)
    temporary: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(temporary, flags, 0o600)
        try:
            payload = json_module.dumps(
                {
                    "pid": attestation.pid,
                    "started_at": attestation.started_at,
                    "key": key,
                },
                separators=(",", ":"),
            ).encode("utf-8")
            os.write(fd, payload)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(temporary, path)
        return True
    except OSError:
        # Reuse is load-bearing because bootstrap is deliberately rate-limited.
        # Refuse this key instead of silently entering a later-call failure mode.
        try:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        except OSError:
            pass
        return False


def _stop_endpoint(args) -> _StopEndpoint | None:
    """Resolve one host-owned HTTP Stop door and its local operator key."""

    from kestrel_sovereign import cli

    project_dir = cli._get_project_dir()
    config = cli.MultiAgentConfig.load(
        project_dir / MULTI_AGENT_CONFIG_FILENAME
    )
    operator_keys = cli._operator_api_keys(project_dir)
    host_attestation = _attested_local_process(
        project_dir,
        pid_file=project_dir / "logs" / ".host.pid",
        port=config.host.port,
        bind_host=getattr(config.host, "bind", "0.0.0.0"),
    )
    if args.all:
        if host_attestation is None:
            return None
        operator_key = _host_operator_key(
            host_attestation,
            operator_keys,
        )
        if operator_key is None:
            return None
        return _StopEndpoint(
            url=f"{_origin(host_attestation)}/api/host/stop",
            api_key=operator_key,
            attestation=host_attestation,
        )

    local = config.get_local_agents().get(args.name)
    if local is not None:
        data_dir = local.resolve_data_dir(project_dir)
        try:
            expected_agent_id = read_anchor_agent_did_sync(
                str(data_dir),
                mode=AgentDIDLookupMode.INSPECTION,
            )
        except (OSError, RuntimeError, ValueError):
            return None
        if not isinstance(expected_agent_id, str) or not expected_agent_id.strip():
            return None

        standalone_attestation = _attested_local_process(
            project_dir,
            pid_file=ProcessManager.agent_pid_file(data_dir),
            port=local.port,
            bind_host=getattr(config.host, "bind", "0.0.0.0"),
        )
        if standalone_attestation is not None:
            bootstrap_key = _standalone_bootstrap_key(standalone_attestation)
            candidates = tuple(
                dict.fromkeys(
                    key for key in (bootstrap_key, *operator_keys) if key
                )
            )
            origin = _origin(standalone_attestation)
            key = _agent_operator_key(
                standalone_attestation,
                f"{origin}/api/agent/info",
                candidates,
                expected_agent_id=expected_agent_id,
            )
            if key is not None:
                return _StopEndpoint(
                    url=f"{origin}/api/agent/stop",
                    api_key=key,
                    attestation=standalone_attestation,
                    expected_agent_id=expected_agent_id,
                )

        if host_attestation is not None:
            agent_segment = encode_agent_route_name(args.name)
            origin = (
                f"{_origin(host_attestation)}/api/agent-routes/{agent_segment}"
            )
            key = _agent_operator_key(
                host_attestation,
                f"{origin}/api/agent/info",
                operator_keys,
                expected_agent_id=expected_agent_id,
            )
            if key is not None:
                return _StopEndpoint(
                    url=f"{origin}/api/agent/stop",
                    api_key=key,
                    attestation=host_attestation,
                    expected_agent_id=expected_agent_id,
                )
        return None

    # Remote registrations carry routing only, not a remote sovereign
    # credential. Never send this host's KESTREL_API_KEY to an arbitrary
    # configured URL. A future remote operator-auth contract can add a
    # separately scoped credential; until then this CLI door fails closed.
    if args.name in config.get_remote_agents():
        return None
    return None


def _all_stop_targets(args) -> tuple[tuple[_AllStopTarget, ...], tuple[str, ...]]:
    """Resolve every live local control process participating in ``--all``."""

    from kestrel_sovereign import cli

    project_dir = cli._get_project_dir()
    config = cli.MultiAgentConfig.load(
        project_dir / MULTI_AGENT_CONFIG_FILENAME
    )
    targets: list[_AllStopTarget] = []
    unreachable: list[str] = []

    host_pid_file = project_dir / "logs" / ".host.pid"
    host_record = ProcessManager.read_pid_record(host_pid_file)
    host_endpoint = _stop_endpoint(args)
    if host_endpoint is not None:
        targets.append(_AllStopTarget(host_endpoint, True, "host fleet"))
    elif host_record.is_running:
        unreachable.append("host fleet")

    for name, local in sorted(
        config.get_local_agents().items(),
        key=lambda item: (item[0].casefold(), item[0]),
    ):
        data_dir = local.resolve_data_dir(project_dir)
        pid_file = ProcessManager.agent_pid_file(data_dir)
        record = ProcessManager.read_pid_record(pid_file)
        if not record.is_running:
            continue
        endpoint = _stop_endpoint(
            SimpleNamespace(name=name, all=False, reason=args.reason)
        )
        if endpoint is None or endpoint.attestation.pid_file != pid_file:
            unreachable.append(name)
            continue
        targets.append(_AllStopTarget(endpoint, False, name))

    return tuple(targets), tuple(unreachable)


def _error_message(response: Any) -> str:
    try:
        payload = response.json()
    except ValueError:
        return str(getattr(response, "text", "") or "request failed")[:300]
    if not isinstance(payload, dict):
        return "request failed"
    error = payload.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or error.get("code") or "request failed")
    detail = payload.get("detail")
    if isinstance(detail, dict):
        return str(detail.get("message") or detail.get("code") or "request failed")
    return str(detail or payload.get("message") or "request failed")


def _response_outcomes(payload: Any) -> list[Any]:
    if not isinstance(payload, dict):
        return []
    direct = payload.get("stop_outcomes")
    if isinstance(direct, list):
        return list(direct)
    candidates = [
        (payload.get("error") or {}).get("details")
        if isinstance(payload.get("error"), dict)
        else None,
        payload.get("details"),
        payload.get("detail"),
    ]
    for candidate in candidates:
        if isinstance(candidate, list):
            return list(candidate)
        if isinstance(candidate, dict) and candidate.get("disposition"):
            return [candidate]
    return []


def _print_outcome(outcome: dict[str, Any]) -> bool:
    disposition = str(outcome.get("disposition") or "indeterminate")
    agent_id = str(outcome.get("agent_id") or "unknown-agent")
    requested = str(outcome.get("requested_target") or "current work")
    detail = outcome.get("detail")
    receipt_id = outcome.get("receipt_id")
    suffix = f" — {detail}" if isinstance(detail, str) and detail else ""
    receipt = f" [receipt {receipt_id}]" if receipt_id else ""
    print(f"{agent_id} {requested}: {disposition}{receipt}{suffix}")
    return disposition in _CONFIRMED_DISPOSITIONS and bool(receipt_id)


def _valid_success_outcomes(
    payload: dict[str, Any],
    *,
    operation_id: str,
    all_agents: bool,
    expected_agent_id: str | None,
) -> list[dict[str, Any]] | None:
    """Strictly validate evidence before the CLI may return success."""

    direct = payload.get("stop_outcomes")
    if payload.get("success") is not True or not isinstance(direct, list) or not direct:
        return None
    if not all(isinstance(item, dict) for item in direct):
        return None
    outcomes: list[dict[str, Any]] = list(direct)
    if all_agents:
        count_fields = (
            payload.get("target_count"),
            payload.get("confirmed_count"),
            payload.get("unconfirmed_count"),
        )
        if (
            not all(type(value) is int for value in count_fields)
            or payload["target_count"] != len(outcomes)
            or payload.get("confirmed_count") != len(outcomes)
            or payload.get("unconfirmed_count") != 0
            or payload.get("correlation_id") != operation_id
            or payload.get("state") != "confirmed"
        ):
            return None
    elif expected_agent_id is None or len(outcomes) != 1:
        return None
    agent_ids: set[str] = set()
    resolved_targets: set[str] = set()
    receipt_ids: set[str] = set()
    for outcome in outcomes:
        required_text = (
            outcome.get("agent_id"),
            outcome.get("resolved_target"),
            outcome.get("receipt_id"),
        )
        if (
            not all(isinstance(value, str) and value.strip() for value in required_text)
            or outcome.get("correlation_id") != operation_id
            or outcome.get("disposition") not in _CONFIRMED_DISPOSITIONS
            or outcome.get("scope") != ("host" if all_agents else "agent")
        ):
            return None
        if all_agents and outcome.get("requested_target") is not None:
            return None
        if all_agents and outcome["agent_id"] != outcome["resolved_target"]:
            return None
        if all_agents:
            agent_ids.add(outcome["agent_id"])
            resolved_targets.add(outcome["resolved_target"])
            receipt_ids.add(outcome["receipt_id"])
        if not all_agents and (
            outcome.get("requested_target") != expected_agent_id
            or outcome["agent_id"] != expected_agent_id
            or outcome["resolved_target"] != expected_agent_id
        ):
            return None
    if all_agents and (
        len(agent_ids) != len(outcomes)
        or len(resolved_targets) != len(outcomes)
        or len(receipt_ids) != 1
    ):
        return None
    return outcomes


def _stop_one(
    resolved: _StopEndpoint,
    *,
    reason: str | None,
    all_agents: bool,
    label: str,
) -> int:
    """Submit and verify one process-bound cooperative Stop operation."""

    url, api_key = resolved.url, resolved.api_key
    if not api_key:
        print("Stop requires a locally configured KESTREL_API_KEY.")
        return 1

    import httpx

    operation_id = _operation_id()
    body = {"correlation_id": operation_id}
    if reason is not None:
        body["reason"] = reason
    if resolved.expected_agent_id is not None:
        body["expected_agent_id"] = resolved.expected_agent_id
    if not _attestation_is_current(resolved.attestation):
        print("Stop target identity changed before dispatch; outcome is indeterminate.")
        return 1
    try:
        response = _local_request(
            "POST",
            url,
            attestation=resolved.attestation,
            json=body,
            headers={"X-API-Key": api_key},
            timeout=60.0,
        )
    except (httpx.RequestError, _LocalRequestError) as error:
        print(f"{label}: indeterminate — Stop request failed: {error}")
        return 1

    if response.status_code != 200:
        try:
            error_payload = response.json()
        except ValueError:
            error_payload = None
        error_outcomes = _response_outcomes(error_payload)
        if error_outcomes and all(
            isinstance(outcome, dict) and outcome.get("disposition")
            for outcome in error_outcomes
        ):
            for outcome in error_outcomes:
                _print_outcome(outcome)
            return 1
        print(
            f"{label}: indeterminate — HTTP {response.status_code}: "
            f"{_error_message(response)}"
        )
        return 1
    try:
        payload = response.json()
    except ValueError:
        print("Stop response was not valid JSON; outcome is indeterminate.")
        return 1
    if not isinstance(payload, dict):
        print("Stop response had an invalid shape; outcome is indeterminate.")
        return 1
    outcomes = _response_outcomes(payload)
    if not outcomes:
        print("Stop returned no typed outcomes; outcome is indeterminate.")
        return 1

    verified = _valid_success_outcomes(
        payload,
        operation_id=operation_id,
        all_agents=all_agents,
        expected_agent_id=resolved.expected_agent_id,
    )
    if verified is None:
        for outcome in outcomes:
            if isinstance(outcome, dict):
                _print_outcome(outcome)
            else:
                print("unknown-agent current work: indeterminate — invalid outcome")
        print("Stop response evidence was inconsistent; outcome is indeterminate.")
        return 1
    confirmed = True
    for outcome in verified:
        if not isinstance(outcome, dict):
            print("unknown-agent current work: indeterminate — invalid outcome")
            confirmed = False
            continue
        confirmed = _print_outcome(outcome) and confirmed
    return 0 if confirmed else 1


def cmd_stop(args) -> int:
    """Cooperatively stop one named agent or every live local control process."""

    if bool(args.all) == bool(args.name):
        print("Choose exactly one Stop target: an agent name or --all.")
        return 2

    if args.all:
        targets, unreachable = _all_stop_targets(args)
        if not targets and not unreachable:
            print("Stop target all agents is not configured or is unreachable.")
            return 1
        print("Stopping all agents cooperatively:")
        result = 0
        for target in targets:
            result = max(
                result,
                _stop_one(
                    target.endpoint,
                    reason=args.reason,
                    all_agents=target.all_agents,
                    label=target.label,
                ),
            )
        for label in unreachable:
            print(f"{label}: unreachable — local Stop authority could not be proven")
            result = 1
        return result

    resolved = _stop_endpoint(args)
    if resolved is None:
        print(f"Stop target '{args.name}' is not configured or is unreachable.")
        return 1
    return _stop_one(
        resolved,
        reason=args.reason,
        all_agents=False,
        label=args.name,
    )


def add_stop_subparser(subparsers) -> None:
    parser = subparsers.add_parser(
        "stop",
        help="Cooperatively stop in-flight agent work (never processes)",
    )
    parser.add_argument("name", nargs="?", help="One configured agent name")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Stop in-flight work across all host-owned agents",
    )
    parser.add_argument(
        "--reason",
        help="Operator reason recorded in the durable Stop receipt",
    )


__all__ = ["add_stop_subparser", "cmd_stop"]
