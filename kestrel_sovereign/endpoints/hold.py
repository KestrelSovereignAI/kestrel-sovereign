"""Sovereign host door for durable Hold latches and their receipts.

Hold is state; Stop is an event. This door therefore reads and mutates the
persisted latches every worker observes after a restart, and it never cancels
work — that is :mod:`kestrel_sovereign.endpoints.host_stop`, a deliberately
separate door for a deliberately different type. A console gesture that means
"stop and hold" performs two requests here and there, and keeps both receipts.

The store owns idempotency (``operation_id``) and the release compare-and-set
(``expected_hold_receipt_id``); this module resolves authority and the target,
and projects the store's typed result onto the wire.
"""

from __future__ import annotations

from typing import Annotated, Any, Optional

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator

from kestrel_sovereign.api_errors import ApiHTTPException
from kestrel_sovereign.endpoints.agent_helpers import (
    caller_is_sovereign,
    sovereign_actor_id,
)
from kestrel_sovereign.features.storage_access import (
    AgentIdentityUnavailable,
    resolve_scoped_agent_did,
)
from kestrel_sovereign.hold import (
    EffectiveHoldState,
    HoldCorruptStateError,
    HoldIdempotencyConflict,
    HoldMutation,
    HoldScope,
    HoldState,
    HoldStateError,
    hold_latch_payload,
    hold_receipt_payload,
    require_context_hold_store,
)
from kestrel_sovereign.rate_limit import hold_admission_rate_limit

router = APIRouter(prefix="/api/host", tags=["host"])

MAX_HOLD_REASON_LENGTH = 1024
MAX_HOLD_OPERATION_ID_LENGTH = 256
MAX_HOLD_RECEIPT_ID_LENGTH = 256
MAX_HOLD_TARGET_ID_LENGTH = 512


def _non_blank(value: str | None, field: str) -> str | None:
    if value is None:
        return None
    if not value.strip():
        raise ValueError(f"{field} must contain non-whitespace text")
    return value


class HoldBody(BaseModel):
    """Set one latch. The host scope carries no caller-chosen target."""

    model_config = ConfigDict(extra="forbid")

    scope: HoldScope
    target_id: Annotated[
        str | None, Field(min_length=1, max_length=MAX_HOLD_TARGET_ID_LENGTH)
    ] = None
    reason: Annotated[str, Field(min_length=1, max_length=MAX_HOLD_REASON_LENGTH)]
    operation_id: Annotated[
        str, Field(min_length=1, max_length=MAX_HOLD_OPERATION_ID_LENGTH)
    ]

    @field_validator("target_id", "reason", "operation_id")
    @classmethod
    def validate_text(cls, value: str | None, info) -> str | None:
        return _non_blank(value, info.field_name)


class HoldReleaseBody(HoldBody):
    """Release exactly the observed latch; a stale release is refused."""

    expected_hold_receipt_id: Annotated[
        str, Field(min_length=1, max_length=MAX_HOLD_RECEIPT_ID_LENGTH)
    ]

    @field_validator("expected_hold_receipt_id")
    @classmethod
    def validate_receipt_id(cls, value: str) -> str:
        return _non_blank(value, "expected_hold_receipt_id")


def _hold_store(request: Request) -> Any:
    """Return the host's durable Hold store, or refuse the request.

    A console that cannot read the latch must be told so, never shown an
    agent as un-held because the store was missing.
    """

    context = getattr(request.app.state, "host_context", None)
    if context is None:
        raise ApiHTTPException(
            status_code=503,
            code="hold_state_unavailable",
            message="Durable Hold state is unavailable.",
        )
    try:
        return require_context_hold_store(context)
    except HoldStateError as error:
        raise ApiHTTPException(
            status_code=503,
            code="hold_state_unavailable",
            message="Durable Hold state is unavailable.",
        ) from error


def _hold_agent_targets(request: Request) -> tuple[str, ...]:
    """The DIDs this host can latch, in the identity enforcement reads.

    ``hold/enforcement.py`` scopes the turn-start latch through the same
    guard, so a latch set here addresses exactly the row that refuses that
    agent's next turn. Resolving a second identity (a display name, an
    ``agent_id`` alias) would write a latch nothing ever reads.
    """

    manager = getattr(request.app.state, "agent_manager", None)
    if manager is not None:
        list_agents = getattr(manager, "list_agents", None)
        if not callable(list_agents):
            raise RuntimeError("host agent inventory is unavailable")
        listed = list_agents()
        if not isinstance(listed, dict):
            raise TypeError("host agent inventory has an invalid type")
        candidates = tuple(
            agent for _name, agent in sorted(
                listed.items(), key=lambda item: (item[0].casefold(), item[0])
            )
        )
    else:
        agent = getattr(request.app.state, "agent", None)
        candidates = (agent,) if agent is not None else ()

    targets: list[str] = []
    for candidate in candidates:
        try:
            did = resolve_scoped_agent_did(candidate)
        except AgentIdentityUnavailable as error:
            raise RuntimeError(
                "host agent inventory contains no trusted identity"
            ) from error
        if did not in targets:
            targets.append(did)
    return tuple(targets)


def _inventory(request: Request) -> tuple[str, ...]:
    try:
        return _hold_agent_targets(request)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ApiHTTPException(
            status_code=503,
            code="hold_inventory_unavailable",
            message="Host Hold target inventory is unavailable.",
        ) from error


def _resolve_target(request: Request, scope: HoldScope, target_id: str | None) -> Optional[str]:
    """Bind an agent latch to a live agent, never to an unread string."""

    if scope is HoldScope.HOST:
        # The store owns the fixed host target; a foreign one is a 400 there.
        return target_id
    if target_id is None:
        raise ApiHTTPException(
            status_code=400,
            code="hold_target_required",
            message="An agent Hold requires the target agent's identity.",
        )
    if target_id not in _inventory(request):
        raise ApiHTTPException(
            status_code=404,
            code="hold_target_unknown",
            message="No agent with that identity is hosted here.",
        )
    return target_id


def _mutation_payload(mutation: HoldMutation) -> dict[str, Any]:
    return {
        "receipt": hold_receipt_payload(mutation.receipt),
        "current": hold_latch_payload(mutation.current),
    }


def _refuse_store_failure(error: Exception) -> ApiHTTPException:
    if isinstance(error, HoldIdempotencyConflict):
        return ApiHTTPException(
            status_code=409,
            code="hold_operation_conflict",
            message="That operation id already recorded a different Hold mutation.",
        )
    if isinstance(error, HoldCorruptStateError):
        return ApiHTTPException(
            status_code=503,
            code="hold_state_corrupt",
            message="Durable Hold state could not be interpreted safely.",
        )
    if isinstance(error, HoldStateError):
        return ApiHTTPException(
            status_code=503,
            code="hold_state_unavailable",
            message="Durable Hold state is unavailable.",
        )
    return ApiHTTPException(
        status_code=400,
        code="hold_request_invalid",
        message=str(error),
    )


def _agent_entry(agent_id: str, host: Optional[HoldState], agent: Optional[HoldState]):
    # Compose through the runtime dataclass rather than an `or` here: "held"
    # is an authority rule, and the console must not own a second copy of it.
    effective = EffectiveHoldState(host=host, agent=agent)
    return {
        "agent_id": agent_id,
        "held": effective.held,
        "sources": [source.value for source in effective.sources],
        "agent_hold": hold_latch_payload(agent),
    }


@router.get("/hold")
async def host_hold_state(request: Request, response: Response):
    """Expose caller authority and every latch this host's cards render."""

    response.headers["Cache-Control"] = "private, no-store"
    response.headers["Vary"] = "Authorization, Cookie, X-API-Key"
    store = _hold_store(request)
    targets = _inventory(request)
    try:
        host = await store.get_hold(HoldScope.HOST)
        agents = [
            _agent_entry(
                agent_id,
                host,
                await store.get_hold(HoldScope.AGENT, agent_id),
            )
            for agent_id in targets
        ]
    except Exception as error:
        raise _refuse_store_failure(error) from error
    return {
        "can_hold": caller_is_sovereign(request),
        "host_hold": hold_latch_payload(host),
        "agents": agents,
    }


@router.post("/hold")
@hold_admission_rate_limit
async def set_host_hold(request: Request, response: Response, body: HoldBody):
    """Latch an agent's (or the host's) willingness to begin a turn."""

    # Authority first: the target resolution below distinguishes a hosted
    # agent from an unknown one, so refusing after it would be a probe.
    actor_id = sovereign_actor_id(request)
    store = _hold_store(request)
    target_id = _resolve_target(request, body.scope, body.target_id)
    try:
        mutation = await store.set_hold(
            scope=body.scope,
            target_id=target_id,
            actor_id=actor_id,
            reason=body.reason,
            operation_id=body.operation_id,
        )
    except Exception as error:
        raise _refuse_store_failure(error) from error
    return _mutation_payload(mutation)


@router.post("/hold/release")
@hold_admission_rate_limit
async def release_host_hold(
    request: Request, response: Response, body: HoldReleaseBody
):
    """Release exactly the latch the caller observed, and nothing else."""

    actor_id = sovereign_actor_id(request)
    store = _hold_store(request)
    target_id = _resolve_target(request, body.scope, body.target_id)
    try:
        mutation = await store.release_hold(
            scope=body.scope,
            target_id=target_id,
            actor_id=actor_id,
            reason=body.reason,
            operation_id=body.operation_id,
            expected_hold_receipt_id=body.expected_hold_receipt_id,
        )
    except Exception as error:
        raise _refuse_store_failure(error) from error
    return _mutation_payload(mutation)


__all__ = [
    "HoldBody",
    "HoldReleaseBody",
    "host_hold_state",
    "release_host_hold",
    "router",
    "set_host_hold",
]
