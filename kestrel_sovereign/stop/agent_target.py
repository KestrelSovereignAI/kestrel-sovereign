"""Agent-local adapter behind every cooperative Stop door.

The HTTP operator route and authenticated peer-signal source deliberately share
this inventory builder.  A door chooses the actor and routing authority; this
module alone snapshots the agent's live work addresses and performs cooperative
cancellation.
"""

from __future__ import annotations

from typing import Any, Optional

from kestrel_sovereign.agent.invocation import validate_invocation_id
from kestrel_sovereign.agent.request_lifecycle import RequestCompletionDisposition

from .authority import CancellationAuthority, CooperativeStopTarget, StopCleanupRegistry
from .types import StopDisposition, StopRequest, StopScope


def agent_stop_identity(agent: Any) -> str:
    """Return the agent's stable Stop address or fail closed."""

    for attribute in ("did", "agent_id"):
        candidate = getattr(agent, attribute, None)
        if isinstance(candidate, str) and candidate.strip():
            return candidate
    raise ValueError("agent Stop requires a stable target identity")


def build_agent_cancellation_authority(
    agent: Any,
    *,
    cleanup_registry: StopCleanupRegistry | None = None,
    include_request_id: Optional[str] = None,
) -> CancellationAuthority:
    """Build the one-agent authority used by local and peer Stop doors."""

    registry = cleanup_registry
    if registry is None:
        registry = getattr(agent, "_stop_cleanup_registry", None)
        if registry is None:
            registry = StopCleanupRegistry()
            setattr(agent, "_stop_cleanup_registry", registry)
        elif not isinstance(registry, StopCleanupRegistry):
            raise TypeError("agent Stop cleanup registry has an invalid type")
    elif not isinstance(registry, StopCleanupRegistry):
        raise TypeError("Stop cleanup registry has an invalid type")

    return CancellationAuthority(
        lambda: (
            build_agent_stop_target(
                agent,
                include_request_id=include_request_id,
            ),
        ),
        cleanup_registry=registry,
    )


def build_agent_stop_target(
    agent: Any,
    *,
    include_request_id: Optional[str],
    target_identity: Optional[str] = None,
) -> CooperativeStopTarget:
    agent_id = (
        agent_stop_identity(agent)
        if target_identity is None
        else target_identity
    )
    if not isinstance(agent_id, str) or not agent_id.strip():
        raise ValueError("agent Stop requires a stable target identity")
    active_request_ids = set(
        getattr(agent, "_active_request_ids", set()) or set()
    )
    abandoned_turns = getattr(agent, "_abandoned_request_generations", None)
    if isinstance(abandoned_turns, dict):
        active_request_ids.update(abandoned_turns)
    current_turn = getattr(agent, "_current_request_id", None)
    if isinstance(current_turn, str) and current_turn:
        active_request_ids.add(current_turn)
    if include_request_id is not None:
        active_request_ids.add(include_request_id)

    turn_request_ids, turn_request_generations = _turn_request_bindings(agent)
    turn_addresses = active_request_ids.union(turn_request_ids)

    async def cancel_request(stop_request: StopRequest) -> StopDisposition:
        cancelled_request_ids: list[Optional[str]] = []
        if stop_request.scope is StopScope.TURN:
            cancel_kwargs: dict[str, object] = {"request_id": stop_request.target}
            if stop_request.request_generation is not None:
                cancel_kwargs["generation"] = stop_request.request_generation
            canceled = agent.cancel_current_request(**cancel_kwargs)
            if canceled:
                cancelled_request_ids.append(stop_request.target)
            else:
                # An exact public request may be between HTTP dispatch and
                # lifecycle registration. Fence it briefly rather than letting
                # that race resurrect work after an acknowledged Stop.
                reserve = getattr(type(agent), "reserve_request_cancellation", None)
                if stop_request.request_generation is None and callable(reserve):
                    reserve(agent, stop_request.target)
        else:
            canceled = False
            for active_request_id in sorted(active_request_ids):
                request_cancelled = agent.cancel_current_request(
                    request_id=active_request_id
                )
                if request_cancelled:
                    cancelled_request_ids.append(active_request_id)
                canceled = request_cancelled or canceled
            if not active_request_ids:
                canceled = agent.cancel_current_request(request_id=None)
                if canceled:
                    cancelled_request_ids.append(None)

        if canceled:
            wait_for_completion = getattr(agent, "wait_for_request_completion", None)
            if not callable(wait_for_completion):
                raise RuntimeError("agent cannot confirm request lifecycle completion")
            abandoned = False
            for cancelled_request_id in cancelled_request_ids:
                wait_kwargs = {}
                if (
                    stop_request.scope is StopScope.TURN
                    and stop_request.request_generation is not None
                ):
                    wait_kwargs["generation"] = stop_request.request_generation
                completion_disposition = await wait_for_completion(
                    cancelled_request_id,
                    **wait_kwargs,
                )
                abandoned = abandoned or (
                    completion_disposition is RequestCompletionDisposition.ABANDONED
                )
            if abandoned:
                return StopDisposition.UNREACHABLE
        return (
            StopDisposition.STOPPED
            if canceled
            else StopDisposition.ALREADY_COMPLETE
        )

    return CooperativeStopTarget(
        target_id=agent_id,
        agent_id=agent_id,
        cancel=cancel_request,
        turn_ids=frozenset(turn_addresses),
        turn_request_ids=turn_request_ids,
        turn_request_generations=turn_request_generations,
    )


def _turn_request_bindings(agent: Any) -> tuple[dict[str, str], dict[str, int]]:
    instance_accessor = vars(agent).get("active_turn_request_bindings")
    if callable(instance_accessor):
        has_binding_accessor = True
        raw_turn_bindings = instance_accessor()
    else:
        class_accessor = getattr(type(agent), "active_turn_request_bindings", None)
        has_binding_accessor = callable(class_accessor)
        raw_turn_bindings = (
            class_accessor(agent) if has_binding_accessor else None
        )

    if has_binding_accessor:
        if not isinstance(raw_turn_bindings, dict):
            raise TypeError("agent turn binding inventory has an invalid type")
        request_ids: dict[str, str] = {}
        generations: dict[str, int] = {}
        for turn_id, binding in raw_turn_bindings.items():
            if (
                not isinstance(turn_id, str)
                or not isinstance(binding, tuple)
                or len(binding) != 2
                or not isinstance(binding[0], str)
            ):
                raise TypeError("agent turn binding inventory is malformed")
            try:
                validate_invocation_id(turn_id)
                validate_invocation_id(binding[0])
            except ValueError as error:
                raise TypeError("agent turn binding inventory is malformed") from error
            request_ids[turn_id] = binding[0]
            if binding[1] is not None:
                if (
                    not isinstance(binding[1], int)
                    or isinstance(binding[1], bool)
                    or binding[1] <= 0
                ):
                    raise TypeError("agent turn generation is malformed")
                generations[turn_id] = binding[1]
        return request_ids, generations

    turn_index_accessor = vars(agent).get("active_turn_request_ids")
    if not callable(turn_index_accessor):
        turn_index_accessor = getattr(type(agent), "active_turn_request_ids", None)
        request_ids = (
            turn_index_accessor(agent) if callable(turn_index_accessor) else {}
        )
    else:
        request_ids = turn_index_accessor()
    if not isinstance(request_ids, dict):
        raise TypeError("agent turn request inventory has an invalid type")
    return request_ids, {}


__all__ = [
    "agent_stop_identity",
    "build_agent_cancellation_authority",
    "build_agent_stop_target",
]
