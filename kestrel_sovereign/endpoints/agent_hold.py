"""Sovereign host doors for durable agent-scoped Hold and Resume."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

from kestrel_sovereign.api_errors import ApiHTTPException
from kestrel_sovereign.hold import (
    EffectiveHoldState,
    HoldCorruptStateError,
    HoldDisposition,
    HoldIdempotencyConflict,
    HoldMutation,
    HoldReceipt,
    HoldScope,
    HoldState,
    HoldStateError,
)

from .host_authority import require_sovereign_actor


router = APIRouter(prefix="/api/host/holds/agents", tags=["host"])

_MAX_OPERATION_ID_LENGTH = 256
_MAX_RECEIPT_ID_LENGTH = 256


class AgentHoldBody(BaseModel):
    """One agent Hold intent bound to the observed routing-name/DID pair."""

    model_config = ConfigDict(extra="forbid")

    target_agent_id: Annotated[str, Field(min_length=1, max_length=1024)]
    reason: Annotated[str, Field(min_length=1, max_length=1024)]
    operation_id: Annotated[
        str,
        Field(min_length=1, max_length=_MAX_OPERATION_ID_LENGTH),
    ]

    @field_validator("target_agent_id", "reason", "operation_id")
    @classmethod
    def require_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must contain non-whitespace text")
        return value


class AgentReleaseHoldBody(AgentHoldBody):
    """Release intent carrying the exact latch authority the user observed."""

    expected_hold_receipt_id: Annotated[
        str,
        Field(min_length=1, max_length=_MAX_RECEIPT_ID_LENGTH),
    ]

    @field_validator("expected_hold_receipt_id")
    @classmethod
    def require_receipt_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must contain non-whitespace text")
        return value


def hold_state_payload(state: HoldState | None) -> dict[str, Any] | None:
    if state is None:
        return None
    return {
        "scope": state.scope.value,
        "target_id": state.target_id,
        "reason": state.reason,
        "actor_id": state.actor_id,
        "set_at": state.set_at,
        "hold_receipt_id": state.hold_receipt_id,
        "revision": state.revision,
    }


def effective_hold_payload(state: EffectiveHoldState) -> dict[str, Any]:
    return {
        "available": True,
        "held": state.held,
        "sources": [scope.value for scope in state.sources],
        "host": hold_state_payload(state.host),
        "agent": hold_state_payload(state.agent),
    }


def unavailable_hold_payload() -> dict[str, Any]:
    """Represent unknown authority as unknown, never as an unheld claim."""

    return {
        "available": False,
        "held": None,
        "sources": [],
        "host": None,
        "agent": None,
        "error": "hold_state_unavailable",
    }


def hold_receipt_payload(receipt: HoldReceipt) -> dict[str, Any]:
    return {
        "receipt_id": receipt.receipt_id,
        "operation_id": receipt.operation_id,
        "action": receipt.action.value,
        "disposition": receipt.disposition.value,
        "scope": receipt.scope.value,
        "target_id": receipt.target_id,
        "reason": receipt.reason,
        "actor_id": receipt.actor_id,
        "occurred_at": receipt.occurred_at,
        "expected_hold_receipt_id": receipt.expected_hold_receipt_id,
        "prior_hold_receipt_id": receipt.prior_hold_receipt_id,
        "resulting_hold_receipt_id": receipt.resulting_hold_receipt_id,
    }


def hold_mutation_payload(mutation: HoldMutation) -> dict[str, Any]:
    return {
        "success": mutation.receipt.disposition
        is not HoldDisposition.REFUSED_STALE,
        "receipt": hold_receipt_payload(mutation.receipt),
        "current": hold_state_payload(mutation.current),
    }


def _hold_store(request: Request):
    host_context = getattr(request.app.state, "host_context", None)
    store = getattr(host_context, "hold_store", None)
    if store is None:
        raise ApiHTTPException(
            status_code=503,
            code="hold_state_unavailable",
            message="Durable Hold state is unavailable.",
        )
    return store


def _resolve_agent_id(
    request: Request,
    *,
    agent_name: str,
    expected_agent_id: str,
) -> str:
    manager = getattr(request.app.state, "agent_manager", None)
    get_agent = getattr(manager, "get_agent", None)
    agent = get_agent(agent_name) if callable(get_agent) else None
    if agent is None:
        raise ApiHTTPException(
            status_code=404,
            code="agent_not_found",
            message="The requested agent is not loaded on this host.",
        )
    agent_id = getattr(agent, "agent_id", None)
    if not isinstance(agent_id, str) or not agent_id.strip():
        raise ApiHTTPException(
            status_code=503,
            code="agent_identity_unavailable",
            message="The requested agent has no trusted runtime identity.",
        )
    if agent_id != expected_agent_id:
        raise ApiHTTPException(
            status_code=409,
            code="agent_identity_changed",
            message="The agent name now resolves to a different identity; refresh and retry.",
        )
    return agent_id


def _raise_hold_failure(error: HoldStateError) -> None:
    if isinstance(error, HoldIdempotencyConflict):
        raise ApiHTTPException(
            status_code=409,
            code="hold_operation_conflict",
            message="The Hold operation identity was already used for another intent.",
        ) from error
    if isinstance(error, HoldCorruptStateError):
        raise ApiHTTPException(
            status_code=503,
            code="hold_state_corrupt",
            message="Durable Hold state failed its integrity checks.",
        ) from error
    raise ApiHTTPException(
        status_code=503,
        code="hold_state_unavailable",
        message="Durable Hold state is unavailable.",
    ) from error


async def effective_hold_for_agent(request: Request, agent_id: str) -> dict[str, Any]:
    """Read card state without converting storage failure into 'not held'."""

    try:
        store = _hold_store(request)
        return effective_hold_payload(await store.get_effective(agent_id))
    except Exception:
        return unavailable_hold_payload()


@router.post("/{agent_name}")
async def hold_agent(
    request: Request,
    agent_name: str,
    body: AgentHoldBody,
):
    actor_id = require_sovereign_actor(request, action="Agent Hold")
    agent_id = _resolve_agent_id(
        request,
        agent_name=agent_name,
        expected_agent_id=body.target_agent_id,
    )
    store = _hold_store(request)
    try:
        mutation = await store.set_hold(
            scope=HoldScope.AGENT,
            target_id=agent_id,
            actor_id=actor_id,
            reason=body.reason,
            operation_id=body.operation_id,
        )
    except HoldStateError as error:
        _raise_hold_failure(error)
    except Exception as error:
        raise ApiHTTPException(
            status_code=503,
            code="hold_state_unavailable",
            message="Durable Hold state is unavailable.",
        ) from error
    return hold_mutation_payload(mutation)


@router.post("/{agent_name}/release")
async def release_agent_hold(
    request: Request,
    agent_name: str,
    body: AgentReleaseHoldBody,
):
    actor_id = require_sovereign_actor(request, action="Agent Hold release")
    agent_id = _resolve_agent_id(
        request,
        agent_name=agent_name,
        expected_agent_id=body.target_agent_id,
    )
    store = _hold_store(request)
    try:
        mutation = await store.release_hold(
            scope=HoldScope.AGENT,
            target_id=agent_id,
            actor_id=actor_id,
            reason=body.reason,
            operation_id=body.operation_id,
            expected_hold_receipt_id=body.expected_hold_receipt_id,
        )
    except HoldStateError as error:
        _raise_hold_failure(error)
    except Exception as error:
        raise ApiHTTPException(
            status_code=503,
            code="hold_state_unavailable",
            message="Durable Hold state is unavailable.",
        ) from error
    return hold_mutation_payload(mutation)


__all__ = [
    "AgentHoldBody",
    "AgentReleaseHoldBody",
    "effective_hold_for_agent",
    "effective_hold_payload",
    "hold_agent",
    "hold_mutation_payload",
    "hold_receipt_payload",
    "hold_state_payload",
    "release_agent_hold",
    "router",
    "unavailable_hold_payload",
]
