"""Cooperative ``kestrel stop`` commands.

Stop is an andon cord for in-flight cognition.  It calls the authenticated
agent/host Stop APIs and never enters the process lifecycle manager; process
teardown is the separate ``kestrel shutdown`` command.
"""

from __future__ import annotations

import uuid
from typing import Any

from kestrel_sovereign.multi_agent.config import MULTI_AGENT_CONFIG_FILENAME
from kestrel_sovereign.paths import spawned_agent_env


_CONFIRMED_DISPOSITIONS = frozenset({"stopped", "already_complete"})


def _operation_id() -> str:
    return f"cli:{uuid.uuid4()}"


def _local_api_key(project_dir) -> str:
    return str(
        spawned_agent_env(project_dir).get("KESTREL_API_KEY") or ""
    ).strip()


def _stop_endpoint(args) -> tuple[str, str] | None:
    """Resolve one host-owned HTTP Stop door and its local operator key."""

    from kestrel_sovereign import cli

    project_dir = cli._get_project_dir()
    config = cli.MultiAgentConfig.load(
        project_dir / MULTI_AGENT_CONFIG_FILENAME
    )
    if args.all:
        return (
            f"http://localhost:{config.host.port}/api/host/stop",
            _local_api_key(project_dir),
        )

    local = config.get_local_agents().get(args.name)
    if local is not None:
        resolved = cli._detect_running_agent_server(args.name, local, config)
        if resolved is None:
            return None
        base_url, discovered_key = resolved
        return f"{base_url}/api/agent/stop", (
            discovered_key or _local_api_key(project_dir)
        )

    remote = config.get_remote_agents().get(args.name)
    if remote is not None:
        return (
            f"{remote.url.rstrip('/')}/api/agent/stop",
            _local_api_key(project_dir),
        )
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


def _response_outcomes(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    direct = payload.get("stop_outcomes")
    if isinstance(direct, list):
        return [item for item in direct if isinstance(item, dict)]
    candidates = [
        (payload.get("error") or {}).get("details")
        if isinstance(payload.get("error"), dict)
        else None,
        payload.get("details"),
        payload.get("detail"),
    ]
    for candidate in candidates:
        if isinstance(candidate, list):
            return [item for item in candidate if isinstance(item, dict)]
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


def cmd_stop(args) -> int:
    """Cooperatively stop one named agent or all host-owned in-flight work."""

    if bool(args.all) == bool(args.name):
        print("Choose exactly one Stop target: an agent name or --all.")
        return 2

    resolved = _stop_endpoint(args)
    if resolved is None:
        print(f"Stop target '{args.name}' is not configured or is unreachable.")
        return 1
    url, api_key = resolved
    if not api_key:
        print("Stop requires a locally configured KESTREL_API_KEY.")
        return 1

    import httpx

    operation_id = _operation_id()
    body = {"correlation_id": operation_id}
    if args.reason is not None:
        body["reason"] = args.reason
    try:
        response = httpx.post(
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

    confirmed = True
    for outcome in outcomes:
        if not isinstance(outcome, dict):
            print("unknown-agent current work: indeterminate — invalid outcome")
            confirmed = False
            continue
        confirmed = _print_outcome(outcome) and confirmed
    return 0 if confirmed and payload.get("success") is True else 1


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
