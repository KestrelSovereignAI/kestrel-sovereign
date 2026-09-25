"""Sovereign host door for cooperative, receipt-gated Stop fan-out."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator

from kestrel_sovereign.api_errors import ApiHTTPException
from kestrel_sovereign.endpoints.agent_helpers import (
    caller_is_sovereign,
    sovereign_actor_id,
)
from kestrel_sovereign.endpoints.receipt_feed import (
    RECEIPT_FEED_SCHEMA_VERSION,
    bounded_filter_text,
    decode_cursor,
    disclosable_identity,
    encode_cursor,
    parse_time_window,
    resolve_page_size,
)
from kestrel_sovereign.rate_limit import (
    stop_admission_rate_limit,
)
from kestrel_sovereign.stop import (
    MAX_STOP_CORRELATION_ID_BYTES,
    CooperativeStopTarget,
    PeerStopCircuitError,
    PeerStopCircuitStore,
    StopCleanupRegistry,
    StopReceiptError,
    StopReceiptRecord,
    UnavailableStopReceiptStore,
    execute_fleet_stop,
    fleet_in_flight_count,
)
from kestrel_sovereign.stop.circuit import (
    MAX_CIRCUIT_EVENT_PAGE,
    MAX_CIRCUIT_REASON_LENGTH,
    MAX_CIRCUIT_TARGET_LENGTH,
)
from kestrel_sovereign.stop.runtime_target import build_runtime_stop_target

router = APIRouter(prefix="/api/host", tags=["host"])


class HostStopBody(BaseModel):
    """Host Stop intent; target identity is deliberately absent."""

    model_config = ConfigDict(extra="forbid")

    reason: Annotated[str | None, Field(min_length=1, max_length=1024)] = None
    correlation_id: Annotated[
        str | None,
        Field(min_length=1, max_length=MAX_STOP_CORRELATION_ID_BYTES),
    ] = None

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("reason must contain non-whitespace text")
        return value

    @field_validator("correlation_id")
    @classmethod
    def validate_correlation_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value.strip():
            raise ValueError("correlation_id must contain non-whitespace text")
        try:
            encoded = value.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError("correlation_id must be valid Unicode text") from error
        if len(encoded) > MAX_STOP_CORRELATION_ID_BYTES:
            raise ValueError("correlation_id exceeds its UTF-8 byte limit")
        return value


class PeerStopCircuitResetBody(BaseModel):
    """Sovereign reset of one target's peer Stop circuit (#3170)."""

    model_config = ConfigDict(extra="forbid")

    target: Annotated[str, Field(min_length=1, max_length=MAX_CIRCUIT_TARGET_LENGTH)]
    reason: Annotated[str, Field(min_length=1, max_length=MAX_CIRCUIT_REASON_LENGTH)]

    @field_validator("target", "reason")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must contain non-whitespace text")
        return value


def _peer_stop_circuit(request: Request) -> PeerStopCircuitStore | None:
    circuit = getattr(request.app.state, "peer_stop_circuit", None)
    return circuit if isinstance(circuit, PeerStopCircuitStore) else None


async def _peer_stop_circuit_status(request: Request) -> dict:
    """The open peer Stop circuits a sovereign must see (#3170).

    An unreadable breaker is reported as unavailable, never as "no circuit is
    open": the console would otherwise hide the one warning it exists for.
    """

    circuit = _peer_stop_circuit(request)
    if circuit is None:
        return {"available": False, "open": []}
    try:
        open_circuits = await circuit.open_circuits()
    except PeerStopCircuitError:
        return {"available": False, "open": []}
    return {
        "available": True,
        "threshold": circuit.policy.threshold,
        "window_seconds": circuit.policy.window_seconds,
        "open": [entry.to_dict() for entry in open_circuits],
    }


def _caller_can_stop_host(request: Request) -> bool:
    return caller_is_sovereign(request)


def _host_agents(request: Request) -> tuple[tuple[str, object], ...]:
    manager = getattr(request.app.state, "agent_manager", None)
    if manager is not None:
        list_agents = getattr(manager, "list_agents", None)
        if not callable(list_agents):
            raise RuntimeError("host agent inventory is unavailable")
        listed = list_agents()
        if not isinstance(listed, dict):
            raise TypeError("host agent inventory has an invalid type")
        candidates = tuple(
            sorted(listed.items(), key=lambda item: (item[0].casefold(), item[0]))
        )
    else:
        agent = getattr(request.app.state, "agent", None)
        candidates = (("local", agent),) if agent is not None else ()

    resolved: list[tuple[str, object]] = []
    for _name, candidate in candidates:
        agent_id = getattr(candidate, "agent_id", None)
        if not isinstance(agent_id, str) or not agent_id.strip():
            raise RuntimeError("host agent inventory contains no trusted identity")
        resolved.append((agent_id, candidate))
    return tuple(resolved)


def _host_targets(request: Request) -> tuple[CooperativeStopTarget, ...]:
    """Resolve the one live runtime snapshot shared by status and Stop."""

    distributed_registry = getattr(
        request.app.state,
        "distributed_invocation_registry",
        None,
    )
    return tuple(
        build_runtime_stop_target(
            candidate,
            agent_id=agent_id,
            distributed_registry=distributed_registry,
        )
        for agent_id, candidate in _host_agents(request)
    )


@router.get("/stop/status")
async def host_stop_status(request: Request, response: Response):
    """Expose caller authority and authoritative live-agent work count."""

    try:
        targets = _host_targets(request)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ApiHTTPException(
            status_code=503,
            code="host_stop_inventory_unavailable",
            message="Host Stop target inventory is unavailable.",
        ) from error
    response.headers["Cache-Control"] = "private, no-store"
    response.headers["Vary"] = "Authorization, Cookie, X-API-Key"
    distributed_registry = getattr(
        request.app.state,
        "distributed_invocation_registry",
        None,
    )
    durable_has_work = None
    if distributed_registry is not None:
        durable_has_work = getattr(
            distributed_registry,
            "agent_has_unsettled_work",
            None,
        )
        if not callable(durable_has_work):
            raise ApiHTTPException(
                status_code=503,
                code="host_stop_inventory_unavailable",
                message="Host Stop target inventory is unavailable.",
            )

    try:
        in_flight_count = await fleet_in_flight_count(
            targets,
            durable_has_work=durable_has_work,
        )
    except Exception as error:
        raise ApiHTTPException(
            status_code=503,
            code="host_stop_inventory_unavailable",
            message="Host Stop target inventory is unavailable.",
        ) from error
    can_stop = _caller_can_stop_host(request)
    payload = {
        "can_stop": can_stop,
        "in_flight_count": in_flight_count,
    }
    if can_stop:
        # Target DIDs and counts are host control-plane evidence: only the
        # sovereign, who alone may reset a circuit, is shown them.
        payload["peer_stop_circuit"] = await _peer_stop_circuit_status(request)
    return payload


@router.post("/stop/circuit/reset")
async def reset_peer_stop_circuit(
    request: Request,
    response: Response,
    body: PeerStopCircuitResetBody,
):
    """Close one target's peer Stop circuit and discard its count.

    Sovereign-only and receipted: the ``reset`` event names the actor and the
    reason.  It never touches operator Stop, and it neither sets nor releases
    a Hold.
    """

    actor_id = sovereign_actor_id(request)
    response.headers["Cache-Control"] = "private, no-store"
    circuit = _peer_stop_circuit(request)
    if circuit is None:
        raise ApiHTTPException(
            status_code=503,
            code="peer_stop_circuit_unavailable",
            message="Peer Stop circuit breaker is unavailable.",
        )
    try:
        event = await circuit.reset(
            body.target, actor_id=actor_id, reason=body.reason
        )
    except PeerStopCircuitError as error:
        raise ApiHTTPException(
            status_code=503,
            code="peer_stop_circuit_unavailable",
            message="Peer Stop circuit breaker is unavailable.",
        ) from error
    return {"event": event.to_dict()}


@router.get("/stop/circuit/events")
async def peer_stop_circuit_events(
    request: Request,
    response: Response,
    target: str | None = None,
    limit: str | None = None,
):
    """Receipted peer Stop circuit transitions, newest first; sovereign-only."""

    sovereign_actor_id(request)
    response.headers["Cache-Control"] = "private, no-store"
    response.headers["Vary"] = "Authorization, Cookie, X-API-Key"
    target_filter = bounded_filter_text(
        target, "target", max_length=MAX_CIRCUIT_TARGET_LENGTH
    )
    page_size = min(resolve_page_size(limit), MAX_CIRCUIT_EVENT_PAGE)
    circuit = _peer_stop_circuit(request)
    if circuit is None:
        raise ApiHTTPException(
            status_code=503,
            code="peer_stop_circuit_unavailable",
            message="Peer Stop circuit breaker is unavailable.",
        )
    try:
        events = await circuit.list_events(
            target_agent_id=target_filter, limit=page_size
        )
    except PeerStopCircuitError as error:
        raise ApiHTTPException(
            status_code=503,
            code="peer_stop_circuit_unavailable",
            message="Peer Stop circuit breaker is unavailable.",
        ) from error
    return {"events": [event.to_dict() for event in events]}


@router.post("/stop")
@stop_admission_rate_limit
async def stop_host(
    request: Request,
    response: Response,
    body: HostStopBody | None = None,
):
    """Cooperatively stop every currently loaded agent; never stop a process."""

    actor_id = sovereign_actor_id(request)
    try:
        targets = _host_targets(request)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ApiHTTPException(
            status_code=503,
            code="host_stop_inventory_unavailable",
            message="Host Stop target inventory is unavailable.",
        ) from error

    cleanup_registry = getattr(request.app.state, "stop_cleanup_registry", None)
    if cleanup_registry is None:
        cleanup_registry = StopCleanupRegistry()
        request.app.state.stop_cleanup_registry = cleanup_registry
    elif not isinstance(cleanup_registry, StopCleanupRegistry):
        raise ApiHTTPException(
            status_code=503,
            code="host_stop_cleanup_unavailable",
            message="Host Stop cleanup ownership is unavailable.",
        )

    return await execute_fleet_stop(
        targets,
        actor_id=actor_id,
        cleanup_registry=cleanup_registry,
        receipt_store=(
            getattr(request.app.state, "stop_receipt_store", None)
            or UnavailableStopReceiptStore()
        ),
        reason=body.reason if body is not None else None,
        correlation_id=(
            body.correlation_id if body is not None else None
        ),
    )


# Filter bounds: an agent DID of any method, and a W3C trace id (32 hex).
MAX_STOP_AGENT_FILTER_LENGTH = 512
MAX_STOP_TRACE_FILTER_LENGTH = 32


def _stop_receipt_payload(record: StopReceiptRecord) -> dict:
    """Project one immutable receipt for the observability read surface.

    The header answers actor / scope / reason / cascade / when / which agent;
    the ordered outcomes answer what actually happened to each target. What it
    deliberately does NOT do is present a blinded digest as an identity: a
    pre-#3159 agent-scope row recorded no agent DID at all, and it reads back
    as ``null`` — "agent not recorded" — rather than being recovered by
    re-hashing the live inventory, which would guess an identity the receipt
    never held.
    """

    return {
        "feed_seq": record.feed_seq,
        "receipt_id": record.receipt_id,
        "scope": record.scope,
        "door": record.door,
        "actor_id": record.actor_id,
        "target_agent_id": disclosable_identity(record.target_agent_id),
        "reason": record.reason,
        "cascade": record.cascade,
        "occurred_at": record.occurred_at,
        "trace_id": record.trace_id,
        "span_id": record.span_id,
        "outcomes": [
            {
                "ordinal": outcome.ordinal,
                "disposition": outcome.disposition,
                "detail": outcome.detail,
                "agent_id": disclosable_identity(outcome.agent_id),
                "resolved_target": disclosable_identity(outcome.resolved_target),
            }
            for outcome in record.outcomes
        ],
    }


@router.get("/stop/receipts")
async def host_stop_receipts(
    request: Request,
    response: Response,
    since: str | None = None,
    until: str | None = None,
    agent_id: str | None = None,
    trace_id: str | None = None,
    cursor: str | None = None,
    limit: str | None = None,
):
    """Read the durable Stop evidence an observability view renders from.

    Sovereign-only, the same gate as ``POST /api/host/stop``: these rows carry
    operator-written reasons for acts performed on other agents, and the actor
    who performed them.

    Spans are joined to THESE receipts, never the reverse — the receipt is the
    only authority for "stopped", and nothing copies its reason or actor onto
    a span (#3159 R1).
    """

    # Authority first, before any filter is interpreted, so a refusal cannot
    # become a probe for which agents or traces exist. Every filter is taken as
    # raw text for the same reason: a constrained ``Query`` annotation would be
    # validated by FastAPI BEFORE this line runs (see ``receipt_feed``).
    sovereign_actor_id(request)
    response.headers["Cache-Control"] = "private, no-store"
    response.headers["Vary"] = "Authorization, Cookie, X-API-Key"

    window_start, window_end = parse_time_window(since, until)
    agent_filter = bounded_filter_text(
        agent_id, "agent_id", max_length=MAX_STOP_AGENT_FILTER_LENGTH
    )
    trace_filter = bounded_filter_text(
        trace_id, "trace_id", max_length=MAX_STOP_TRACE_FILTER_LENGTH
    )
    after = decode_cursor(cursor)
    page_size = resolve_page_size(limit)

    store = getattr(request.app.state, "stop_receipt_store", None)
    if store is None:
        store = UnavailableStopReceiptStore()
    try:
        page = await store.list_receipts(
            since=window_start,
            until=window_end,
            agent_id=agent_filter,
            trace_id=trace_filter,
            after=after,
            limit=page_size,
        )
    except StopReceiptError as error:
        # An unreadable history is reported as unreadable. An empty list here
        # would tell a console that nothing was ever stopped.
        raise ApiHTTPException(
            status_code=503,
            code="stop_receipts_unavailable",
            message="Durable Stop evidence is unavailable.",
        ) from error
    return {
        "schema_version": RECEIPT_FEED_SCHEMA_VERSION,
        "receipts": [
            _stop_receipt_payload(record) for record in page.receipts
        ],
        "next_cursor": encode_cursor(page.next_key),
    }


__all__ = [
    "HostStopBody",
    "PeerStopCircuitResetBody",
    "host_stop_receipts",
    "host_stop_status",
    "peer_stop_circuit_events",
    "reset_peer_stop_circuit",
    "router",
    "stop_host",
]
