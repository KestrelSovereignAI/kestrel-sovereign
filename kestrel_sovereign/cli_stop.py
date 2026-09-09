"""Cooperative ``kestrel stop`` commands.

Stop is an andon cord for in-flight cognition.  It calls the authenticated
agent/host Stop APIs and never enters the process lifecycle manager; process
teardown is the separate ``kestrel terminate`` command.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kestrel_sovereign.identity.local_anchor import (
    AgentDIDLookupMode,
    read_anchor_agent_did_sync,
)
from kestrel_sovereign.multi_agent.config import MULTI_AGENT_CONFIG_FILENAME
from kestrel_sovereign.multi_agent.process_manager import PidStatus, ProcessManager

_CONFIRMED_DISPOSITIONS = frozenset({"stopped", "already_complete"})


@dataclass(frozen=True, slots=True)
class _LocalProcessAttestation:
    """OS-backed identity of the one local process allowed to receive a key."""

    project_root: Path
    pid_file: Path
    pid: int
    port: int


@dataclass(frozen=True, slots=True)
class _StopEndpoint:
    url: str
    api_key: str
    attestation: _LocalProcessAttestation
    expected_agent_id: str | None = None


def _operation_id() -> str:
    return f"cli:{uuid.uuid4()}"


def _local_request(method: str, url: str, **kwargs):
    """Send local control traffic without honoring ambient proxy variables."""

    import httpx

    with httpx.Client(trust_env=False) as client:
        return client.request(method, url, **kwargs)


def _attested_local_process(
    project_root: Path,
    *,
    pid_file: Path,
    port: int,
) -> _LocalProcessAttestation | None:
    """Bind a configured port to this project's live, recorded process."""

    project_root = project_root.resolve()
    record = ProcessManager.read_pid_record(pid_file)
    if (
        record.status is not PidStatus.LIVE
        or record.pid is None
        or record.root is None
        or record.port != port
    ):
        return None
    try:
        recorded_root = Path(record.root).resolve()
    except (OSError, RuntimeError):
        return None
    if recorded_root != project_root:
        return None
    # A PID file establishes process identity, not socket ownership. Refuse if
    # the configured listener cannot be attributed exclusively to that process.
    if set(ProcessManager.find_pids_on_port(port)) != {record.pid}:
        return None
    return _LocalProcessAttestation(
        project_root=project_root,
        pid_file=pid_file,
        pid=record.pid,
        port=port,
    )


def _attestation_is_current(attestation: _LocalProcessAttestation) -> bool:
    current = _attested_local_process(
        attestation.project_root,
        pid_file=attestation.pid_file,
        port=attestation.port,
    )
    return current == attestation


def _host_operator_key(
    attestation: _LocalProcessAttestation,
    candidates: tuple[str, ...],
) -> str | None:
    """Identify the credential accepted by the live host without mutating it."""

    import httpx

    probe_url = f"http://127.0.0.1:{attestation.port}/api/host/stop/status"
    for candidate in candidates:
        if not _attestation_is_current(attestation):
            return None
        try:
            response = _local_request(
                "GET",
                probe_url,
                headers={"X-API-Key": candidate},
                timeout=2.0,
            )
        except httpx.RequestError:
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
                headers={"X-API-Key": candidate},
                timeout=2.0,
            )
        except httpx.RequestError:
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
    """Read a standalone bootstrap key only from the attested local process."""

    import httpx

    if not _attestation_is_current(attestation):
        return None
    try:
        response = _local_request(
            "GET",
            f"http://127.0.0.1:{attestation.port}/api/auth/key",
            timeout=2.0,
        )
    except httpx.RequestError:
        return None
    if response.status_code != 200 or not _attestation_is_current(attestation):
        return None
    try:
        payload = response.json()
    except ValueError:
        return None
    key = payload.get("key") if isinstance(payload, dict) else None
    return key if isinstance(key, str) and key else None


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
            url=f"http://127.0.0.1:{config.host.port}/api/host/stop",
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
        )
        if standalone_attestation is not None:
            bootstrap_key = _standalone_bootstrap_key(standalone_attestation)
            candidates = tuple(
                dict.fromkeys(
                    key for key in (bootstrap_key, *operator_keys) if key
                )
            )
            origin = f"http://127.0.0.1:{local.port}"
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
            origin = (
                f"http://127.0.0.1:{config.host.port}"
                f"/api/agents/{args.name}"
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
        if (
            not isinstance(payload.get("target_count"), int)
            or isinstance(payload.get("target_count"), bool)
            or payload["target_count"] != len(outcomes)
            or payload.get("confirmed_count") != len(outcomes)
            or payload.get("unconfirmed_count") != 0
            or payload.get("correlation_id") != operation_id
        ):
            return None
    elif expected_agent_id is None or len(outcomes) != 1:
        return None
    for outcome in outcomes:
        required_text = (
            outcome.get("agent_id"),
            outcome.get("resolved_target"),
            outcome.get("receipt_id"),
        )
        if (
            not all(isinstance(value, str) and value for value in required_text)
            or outcome.get("correlation_id") != operation_id
            or outcome.get("disposition") not in _CONFIRMED_DISPOSITIONS
            or outcome.get("scope") != ("host" if all_agents else "agent")
        ):
            return None
        if all_agents and outcome.get("requested_target") is not None:
            return None
        if not all_agents and (
            outcome.get("requested_target") != expected_agent_id
            or outcome["agent_id"] != expected_agent_id
            or outcome["resolved_target"] != expected_agent_id
        ):
            return None
    return outcomes


def cmd_stop(args) -> int:
    """Cooperatively stop one named agent or all host-owned in-flight work."""

    if bool(args.all) == bool(args.name):
        print("Choose exactly one Stop target: an agent name or --all.")
        return 2

    resolved = _stop_endpoint(args)
    if resolved is None:
        target = "all agents" if args.all else f"'{args.name}'"
        print(f"Stop target {target} is not configured or is unreachable.")
        return 1
    url, api_key = resolved.url, resolved.api_key
    if not api_key:
        print("Stop requires a locally configured KESTREL_API_KEY.")
        return 1

    import httpx

    operation_id = _operation_id()
    body = {"correlation_id": operation_id}
    if args.reason is not None:
        body["reason"] = args.reason
    if not _attestation_is_current(resolved.attestation):
        print("Stop target identity changed before dispatch; outcome is indeterminate.")
        return 1
    try:
        response = _local_request(
            "POST",
            url,
            json=body,
            headers={"X-API-Key": api_key},
            timeout=httpx.Timeout(60.0, connect=5.0),
        )
    except httpx.RequestError as error:
        target = "all agents" if args.all else args.name
        print(f"{target}: indeterminate — Stop request failed: {error}")
        return 1

    if response.status_code != 200:
        try:
            error_payload = response.json()
        except ValueError:
            error_payload = None
        error_outcomes = _response_outcomes(error_payload)
        if error_outcomes:
            for outcome in error_outcomes:
                _print_outcome(outcome)
            return 1
        target = "all agents" if args.all else args.name
        print(
            f"{target}: indeterminate — HTTP {response.status_code}: "
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
        all_agents=bool(args.all),
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
