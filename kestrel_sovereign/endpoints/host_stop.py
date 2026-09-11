"""Sovereign host door for cooperative, receipt-gated Stop fan-out."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator

from kestrel_sovereign.api_errors import ApiHTTPException
from kestrel_sovereign.endpoints.agent_helpers import caller_is_sovereign, get_caller
from kestrel_sovereign.rate_limit import (
    stop_admission_rate_limit,
)
from kestrel_sovereign.stop import (
    MAX_STOP_CORRELATION_ID_BYTES,
    CooperativeStopTarget,
    StopCleanupRegistry,
    UnavailableStopReceiptStore,
    execute_fleet_stop,
    fleet_in_flight_count,
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


def _sovereign_actor(request: Request) -> str:
    if not caller_is_sovereign(request):
        raise ApiHTTPException(
            status_code=403,
            code="sovereign_authority_required",
            message="Host Stop requires sovereign authority.",
        )
    identity = get_caller(request).identity
    return identity if isinstance(identity, str) and identity.strip() else "api_key"


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
    return {
        "can_stop": _caller_can_stop_host(request),
        "in_flight_count": in_flight_count,
    }


@router.post("/stop")
@stop_admission_rate_limit
async def stop_host(
    request: Request,
    response: Response,
    body: HostStopBody | None = None,
):
    """Cooperatively stop every currently loaded agent; never stop a process."""

    actor_id = _sovereign_actor(request)
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


__all__ = ["HostStopBody", "host_stop_status", "router", "stop_host"]
