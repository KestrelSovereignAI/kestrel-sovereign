"""Host-neutral fleet Stop execution for trusted embedding adapters.

Authentication and tenant selection belong to the HTTP host. This module owns
the shared typed Stop execution, receipt, and response contract so an embed
cannot silently fork cooperative cancellation semantics.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
from typing import Any

from kestrel_sovereign.telemetry import current_trace_identity

from .authority import CancellationAuthority, CooperativeStopTarget, StopCleanupRegistry
from .types import StopDisposition, StopRequest, StopScope

DurableWorkReader = Callable[[str], Awaitable[bool]]


async def fleet_in_flight_count(
    targets: Iterable[CooperativeStopTarget],
    *,
    durable_has_work: DurableWorkReader | None = None,
) -> int:
    """Count active agent targets from one trusted inventory snapshot."""

    frozen = tuple(targets)
    if any(not isinstance(target, CooperativeStopTarget) for target in frozen):
        raise TypeError("fleet Stop inventory contains an untyped target")
    if durable_has_work is not None and not callable(durable_has_work):
        raise TypeError("durable fleet work reader must be callable")

    in_flight_count = 0
    for target in frozen:
        active = bool(target.turn_ids)
        if not active and durable_has_work is not None:
            active = await durable_has_work(target.agent_id)
            if not isinstance(active, bool):
                raise TypeError("durable fleet work reader returned a non-boolean")
        in_flight_count += int(active)
    return in_flight_count


async def execute_fleet_stop(
    targets: Iterable[CooperativeStopTarget],
    *,
    actor_id: str,
    cleanup_registry: StopCleanupRegistry,
    receipt_store: Any,
    reason: str | None = None,
    correlation_id: str | None = None,
) -> dict[str, object]:
    """Execute and summarize one receipt-gated host-scope Stop operation."""

    frozen = tuple(targets)
    authority = CancellationAuthority(
        lambda: frozen,
        cleanup_registry=cleanup_registry,
        receipt_store=receipt_store,
    )
    trace_id, span_id = current_trace_identity()
    request_kwargs: dict[str, object] = {}
    if correlation_id is not None:
        request_kwargs["correlation_id"] = correlation_id
    stop_request = StopRequest(
        scope=StopScope.HOST,
        actor_id=actor_id,
        reason=reason,
        trace_id=trace_id,
        span_id=span_id,
        **request_kwargs,
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
    empty_inventory = (
        len(outcomes) == 1
        and outcomes[0].scope is StopScope.HOST
        and outcomes[0].requested_target is None
        and outcomes[0].resolved_target == StopScope.HOST.value
        and outcomes[0].agent_id == StopScope.HOST.value
    )
    target_count = 0 if empty_inventory else len(outcomes)
    if empty_inventory:
        state = "empty"
    elif confirmed and unconfirmed:
        state = "partial"
    elif unconfirmed:
        state = "unconfirmed"
    else:
        state = "confirmed"
    return {
        "success": target_count > 0 and not unconfirmed,
        "state": state,
        "target_count": target_count,
        "confirmed_count": len(confirmed),
        "unconfirmed_count": len(unconfirmed),
        "correlation_id": stop_request.correlation_id,
        "stop_outcomes": [outcome.to_dict() for outcome in outcomes],
    }


__all__ = ["execute_fleet_stop", "fleet_in_flight_count"]
