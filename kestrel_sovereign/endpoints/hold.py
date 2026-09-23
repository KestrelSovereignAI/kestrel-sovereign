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
from kestrel_sovereign.endpoints.receipt_feed import (
    RECEIPT_FEED_SCHEMA_VERSION,
    bounded_filter_text,
    decode_cursor,
    encode_cursor,
    parse_time_window,
    resolve_page_size,
)
from kestrel_sovereign.features.storage_access import (
    AgentIdentityUnavailable,
    resolve_scoped_agent_did,
)
from kestrel_sovereign.hold import (
    HOST_HOLD_TARGET,
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
    """Bind an agent latch to a live agent, never to an unread string.

    Every caller-supplied target is answered HERE, so that by the time the
    store is called the only argument it can refuse is one this door got
    wrong. That is what lets the store's exceptions below be read as server
    faults rather than sorted, at the wire, into "probably the caller's".
    """

    if scope is HoldScope.HOST:
        # The host scope has exactly one latch, so there is nothing to name.
        if target_id not in (None, HOST_HOLD_TARGET):
            raise ApiHTTPException(
                status_code=400,
                code="hold_request_invalid",
                message="The host Hold scope takes no caller-chosen target.",
            )
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


# What the store is expected to raise at this door, and nothing else. A
# storage driver's OperationalError, or any other unexpected exception —
# `ValueError` included, since this door validates its own arguments before
# the store sees them — is neither a caller mistake nor something whose text
# belongs on the wire. Those are deliberately absent here: they propagate to
# the server's own handler and become a sanitized 500, the way the sibling
# host Stop door leaves its own mutation path unwrapped.
_EXPECTED_HOLD_STORE_FAILURES = HoldStateError


def _refuse_store_failure(error: HoldStateError) -> ApiHTTPException:
    """Project one EXPECTED store refusal onto the wire.

    Total over ``HoldStateError``: the two subclasses carry their own wire
    meaning, and the base class is the store telling this door its durable
    state cannot be served. No branch here reports a caller error, because by
    this point no argument the caller chose is still unvalidated.
    """

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
    return ApiHTTPException(
        status_code=503,
        code="hold_state_unavailable",
        message="Durable Hold state is unavailable.",
    )


def _active_latches(
    snapshot: tuple[HoldState, ...],
) -> dict[tuple[HoldScope, str], HoldState]:
    """Index one validated Hold snapshot by the key a card is rendered from."""

    return {(latch.scope, latch.target_id): latch for latch in snapshot}


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
    # ONE validated snapshot, not one read per card. Every `get_hold` enters
    # the Hold evidence protocol, takes its cross-process serialization lock,
    # and revalidates the database-wide receipt graph — work that is O(the
    # whole receipt history) and does not shrink because the caller only wants
    # one target. Every open console polls this route every few seconds, so
    # 1 + N of those would contend with real Hold mutations and with the
    # turn-start check for the whole fleet. `read_boot_state` is the store's
    # own "every active latch from one stable snapshot" read: one protocol
    # entry, one locked history validation, every latch this door projects.
    try:
        active = _active_latches(await store.read_boot_state())
    except _EXPECTED_HOLD_STORE_FAILURES as error:
        raise _refuse_store_failure(error) from error
    host = active.get((HoldScope.HOST, HOST_HOLD_TARGET))
    agents = [
        _agent_entry(agent_id, host, active.get((HoldScope.AGENT, agent_id)))
        for agent_id in targets
    ]
    return {
        "can_hold": caller_is_sovereign(request),
        "host_hold": hold_latch_payload(host),
        "agents": agents,
    }


@router.get("/hold/receipts")
async def host_hold_receipts(
    request: Request,
    response: Response,
    since: str | None = None,
    until: str | None = None,
    scope: str | None = None,
    agent_id: str | None = None,
    cursor: str | None = None,
    limit: str | None = None,
):
    """Read the append-only Hold history a held agent's card is explained by.

    ``GET /api/host/hold`` answers "what is latched NOW" and forgets
    everything else: the latch row is blanked on release, so a hold that was
    resumed survives only here. That makes this route the only place a resume
    is legible — it is its own ``release`` receipt, linked by
    ``prior_hold_receipt_id`` to the hold it ended, and it neither rewrites nor
    erases that hold (#3159 R5).

    Sovereign-only, matching ``POST /api/host/hold``. This deliberately does
    not widen ``GET /api/host/hold``'s existing exposure; that surface is its
    own ticket.
    """

    # Authority first, and every filter as raw text: a typed or constrained
    # parameter would be validated by FastAPI BEFORE this line (receipt_feed).
    sovereign_actor_id(request)
    response.headers["Cache-Control"] = "private, no-store"
    response.headers["Vary"] = "Authorization, Cookie, X-API-Key"

    window_start, window_end = parse_time_window(since, until)
    scope_filter: HoldScope | None = None
    if scope is not None:
        try:
            scope_filter = HoldScope(scope)
        except ValueError as error:
            raise ApiHTTPException(
                status_code=400,
                code="receipt_filter_invalid",
                message="scope must be 'host' or 'agent'.",
            ) from error
    agent_filter = bounded_filter_text(
        agent_id, "agent_id", max_length=MAX_HOLD_TARGET_ID_LENGTH
    )
    after = decode_cursor(cursor)
    page_size = resolve_page_size(limit)
    if agent_filter is not None:
        if scope_filter is HoldScope.HOST:
            raise ApiHTTPException(
                status_code=400,
                code="hold_request_invalid",
                message="The host Hold scope takes no caller-chosen target.",
            )
        # Naming an agent means the agent latch, never the host one: a host
        # hold is not that agent's receipt even though it holds that agent.
        scope_filter = HoldScope.AGENT

    store = _hold_store(request)
    try:
        page = await store.list_receipts(
            since=window_start,
            until=window_end,
            scope=scope_filter,
            target_id=agent_filter,
            after=after,
            limit=page_size,
        )
    except _EXPECTED_HOLD_STORE_FAILURES as error:
        raise _refuse_store_failure(error) from error
    return {
        "schema_version": RECEIPT_FEED_SCHEMA_VERSION,
        "receipts": [
            {**hold_receipt_payload(entry.receipt), "feed_seq": entry.feed_seq}
            for entry in page.entries
        ],
        "next_cursor": encode_cursor(page.next_key),
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
    except _EXPECTED_HOLD_STORE_FAILURES as error:
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
    except _EXPECTED_HOLD_STORE_FAILURES as error:
        raise _refuse_store_failure(error) from error
    return _mutation_payload(mutation)


__all__ = [
    "HoldBody",
    "HoldReleaseBody",
    "host_hold_receipts",
    "host_hold_state",
    "release_host_hold",
    "router",
    "set_host_hold",
]
