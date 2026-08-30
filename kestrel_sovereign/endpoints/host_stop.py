"""Sovereign host door for cooperative, receipt-gated Stop fan-out."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

from kestrel_sovereign.api_errors import ApiHTTPException
from kestrel_sovereign.auth import CallerContext
from kestrel_sovereign.stop import (
    CancellationAuthority,
    MAX_STOP_CORRELATION_ID_BYTES,
    StopCleanupRegistry,
    StopDisposition,
    StopRequest,
    StopScope,
    UnavailableStopReceiptStore,
)
from kestrel_sovereign.stop.runtime_target import build_runtime_stop_target
from kestrel_sovereign.telemetry import current_trace_identity


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


def _sovereign_actor(request: Request) -> str:
    caller = getattr(request.state, "caller", None)
    if not isinstance(caller, CallerContext) or not caller.is_sovereign:
        raise ApiHTTPException(
            status_code=403,
            code="sovereign_authority_required",
            message="Host Stop requires sovereign authority.",
        )
    identity = caller.identity
    return identity if isinstance(identity, str) and identity.strip() else "api_key"


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


@router.post("/stop")
async def stop_host(
    request: Request,
    body: HostStopBody | None = None,
):
    """Cooperatively stop every currently loaded agent; never stop a process."""

    actor_id = _sovereign_actor(request)
    distributed_registry = getattr(
        request.app.state,
        "distributed_invocation_registry",
        None,
    )
    try:
        candidates = _host_agents(request)
        targets = tuple(
            build_runtime_stop_target(
                candidate,
                agent_id=agent_id,
                distributed_registry=distributed_registry,
            )
            for agent_id, candidate in candidates
        )
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

    authority = CancellationAuthority(
        lambda: targets,
        cleanup_registry=cleanup_registry,
        receipt_store=(
            getattr(request.app.state, "stop_receipt_store", None)
            or UnavailableStopReceiptStore()
        ),
    )
    trace_id, span_id = current_trace_identity()
    stop_request = StopRequest(
        scope=StopScope.HOST,
        actor_id=actor_id,
        reason=body.reason if body is not None else None,
        trace_id=trace_id,
        span_id=span_id,
        **(
            {"correlation_id": body.correlation_id}
            if body is not None and body.correlation_id is not None
            else {}
        ),
    )
    outcomes = await authority.stop(stop_request)
    confirmed = tuple(
        outcome
        for outcome in outcomes
        if outcome.disposition
        in {StopDisposition.STOPPED, StopDisposition.ALREADY_COMPLETE}
    )
    unconfirmed = tuple(
        outcome
        for outcome in outcomes
        if outcome.disposition
        in {StopDisposition.REFUSED, StopDisposition.UNREACHABLE}
    )
    if not targets:
        state = "empty"
    elif confirmed and unconfirmed:
        state = "partial"
    elif unconfirmed:
        state = "unconfirmed"
    else:
        state = "confirmed"
    return {
        "success": bool(targets) and not unconfirmed,
        "state": state,
        "target_count": len(targets),
        "confirmed_count": len(confirmed),
        "unconfirmed_count": len(unconfirmed),
        "correlation_id": stop_request.correlation_id,
        "stop_outcomes": [outcome.to_dict() for outcome in outcomes],
    }


__all__ = ["HostStopBody", "router", "stop_host"]
