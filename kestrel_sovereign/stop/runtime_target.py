"""Shared live-agent adapter for cooperative Stop authorities.

This module is the only bridge from typed Stop requests to an agent's
request-lifecycle API. HTTP doors choose scope and trusted identity; they do
not reimplement cancellation, generation fencing, or completion evidence.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from kestrel_sovereign.agent.invocation import validate_invocation_id
from kestrel_sovereign.agent.request_lifecycle import (
    RequestCompletionDisposition,
)

from .authority import CooperativeStopTarget
from .types import StopDisposition, StopRequest, StopScope


def _active_request_snapshot(
    agent: object,
    *,
    explicit_request_id: str | None = None,
) -> frozenset[str]:
    active = set(getattr(agent, "_active_request_ids", set()) or set())
    abandoned = getattr(agent, "_abandoned_request_generations", None)
    if isinstance(abandoned, Mapping):
        active.update(abandoned)
    current = getattr(agent, "_current_request_id", None)
    if isinstance(current, str) and current:
        active.add(current)
    if explicit_request_id is not None:
        active.add(explicit_request_id)
    if any(not isinstance(item, str) or not item for item in active):
        raise TypeError("agent active request inventory is malformed")
    return frozenset(active)


def _turn_request_bindings(
    agent: object,
) -> tuple[dict[str, str], dict[str, int]]:
    instance_accessor = vars(agent).get("active_turn_request_bindings")
    if callable(instance_accessor):
        raw_bindings = instance_accessor()
    else:
        class_accessor = getattr(type(agent), "active_turn_request_bindings", None)
        raw_bindings = class_accessor(agent) if callable(class_accessor) else None

    if raw_bindings is not None:
        if not isinstance(raw_bindings, dict):
            raise TypeError("agent turn binding inventory has an invalid type")
        request_ids: dict[str, str] = {}
        generations: dict[str, int] = {}
        for turn_id, binding in raw_bindings.items():
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
                raise TypeError(
                    "agent turn binding inventory is malformed"
                ) from error
            request_ids[turn_id] = binding[0]
            generation = binding[1]
            if generation is not None:
                if (
                    not isinstance(generation, int)
                    or isinstance(generation, bool)
                    or generation <= 0
                ):
                    raise TypeError("agent turn generation is malformed")
                generations[turn_id] = generation
        return request_ids, generations

    instance_index = vars(agent).get("active_turn_request_ids")
    if callable(instance_index):
        raw_index = instance_index()
    else:
        class_index = getattr(type(agent), "active_turn_request_ids", None)
        raw_index = class_index(agent) if callable(class_index) else {}
    if not isinstance(raw_index, dict) or any(
        not isinstance(turn_id, str)
        or not isinstance(request_id, str)
        for turn_id, request_id in raw_index.items()
    ):
        raise TypeError("agent turn request inventory is malformed")
    try:
        for turn_id, request_id in raw_index.items():
            validate_invocation_id(turn_id)
            validate_invocation_id(request_id)
    except ValueError as error:
        raise TypeError("agent turn request inventory is malformed") from error
    return dict(raw_index), {}


def build_runtime_stop_target(
    agent: object,
    *,
    agent_id: str,
    explicit_request_id: str | None = None,
    explicit_turn_id: str | None = None,
    distributed_registry: Any | None = None,
    resolve_turn_addresses: bool = True,
) -> CooperativeStopTarget:
    """Snapshot one agent and bind its typed cooperative cancellation action."""

    if not isinstance(agent_id, str) or not agent_id.strip():
        raise ValueError("runtime Stop target requires a concrete agent identity")
    if explicit_request_id is not None and (
        not isinstance(explicit_request_id, str) or not explicit_request_id
    ):
        raise ValueError("explicit Stop request identity must be concrete")
    if explicit_turn_id is not None and (
        not isinstance(explicit_turn_id, str) or not explicit_turn_id
    ):
        raise ValueError("explicit Stop turn identity must be concrete")
    if not isinstance(resolve_turn_addresses, bool):
        raise TypeError("resolve_turn_addresses must be a boolean")

    active_at_resolution = _active_request_snapshot(
        agent,
        explicit_request_id=explicit_request_id,
    )
    if resolve_turn_addresses:
        turn_request_ids, turn_request_generations = _turn_request_bindings(agent)
        turn_addresses = active_at_resolution.union(turn_request_ids)
        if explicit_turn_id is not None and distributed_registry is not None:
            turn_addresses = turn_addresses.union((explicit_turn_id,))
    else:
        turn_request_ids = {}
        turn_request_generations = {}
        turn_addresses = frozenset()

    async def cancel(stop_request: StopRequest) -> StopDisposition:
        distributed_ticket = None
        if distributed_registry is not None:
            if stop_request.scope is StopScope.TURN:
                is_public_turn = (
                    stop_request.target_is_turn_id
                    or stop_request.turn_id != stop_request.target
                )
                if is_public_turn:
                    distributed_ticket = (
                        await distributed_registry.request_public_turn(
                            agent_id,
                            stop_request.turn_id,
                        )
                    )
                else:
                    distributed_ticket = await distributed_registry.request_turn(
                        agent_id,
                        stop_request.target,
                    )
            else:
                distributed_ticket = await distributed_registry.request_agent(
                    agent_id
                )

        cancel_current = getattr(agent, "cancel_current_request", None)
        if not callable(cancel_current):
            raise RuntimeError("agent has no cooperative request cancellation seam")
        cancelled_requests: list[tuple[str | None, int | None]] = []
        if stop_request.scope is StopScope.TURN:
            public_turn_is_remote = (
                stop_request.target_is_turn_id
                and stop_request.turn_id == stop_request.target
            )
            if public_turn_is_remote:
                cancelled = False
            else:
                cancel_kwargs: dict[str, object] = {
                    "request_id": stop_request.target
                }
                if stop_request.request_generation is not None:
                    cancel_kwargs["generation"] = (
                        stop_request.request_generation
                    )
                cancelled = bool(cancel_current(**cancel_kwargs))
            if cancelled:
                cancelled_requests.append(
                    (stop_request.target, stop_request.request_generation)
                )
            else:
                reserve = getattr(type(agent), "reserve_request_cancellation", None)
                if (
                    not public_turn_is_remote
                    and stop_request.request_generation is None
                    and callable(reserve)
                ):
                    reserve(agent, stop_request.target)
        else:
            cancelled = False
            cancel_local_ticket = getattr(
                type(distributed_registry),
                "cancel_local_ticket",
                None,
            )
            if distributed_ticket is not None and callable(cancel_local_ticket):
                ticketed_local = cancel_local_ticket(
                    distributed_registry,
                    distributed_ticket,
                )
                cancelled_requests.extend(ticketed_local)
                cancelled = bool(ticketed_local)
            else:
                # A local host re-reads at cancellation linearization so work
                # admitted during receipt preflight cannot escape. A
                # compatibility registry without UUID mapping uses the
                # original durable snapshot and never widens after it.
                turns_to_cancel = (
                    active_at_resolution.union(_active_request_snapshot(agent))
                    if distributed_ticket is None
                    else active_at_resolution
                )
                for request_id in sorted(turns_to_cancel):
                    request_cancelled = bool(
                        cancel_current(request_id=request_id)
                    )
                    if request_cancelled:
                        cancelled_requests.append((request_id, None))
                    cancelled = request_cancelled or cancelled
                if not turns_to_cancel:
                    cancelled = bool(cancel_current(request_id=None))
                    if cancelled:
                        cancelled_requests.append((None, None))

        if cancelled:
            wait_for_completion = getattr(agent, "wait_for_request_completion", None)
            if not callable(wait_for_completion):
                raise RuntimeError("agent cannot confirm request lifecycle completion")
            abandoned = False
            for request_id, generation in cancelled_requests:
                wait_kwargs: dict[str, object] = {}
                if generation is not None:
                    wait_kwargs["generation"] = generation
                completion = await wait_for_completion(request_id, **wait_kwargs)
                abandoned = abandoned or (
                    completion is RequestCompletionDisposition.ABANDONED
                )
            if abandoned:
                return StopDisposition.UNREACHABLE

        distributed_disposition = StopDisposition.ALREADY_COMPLETE
        if distributed_ticket is not None:
            distributed_disposition = await distributed_registry.wait_for_stop(
                distributed_ticket
            )
            if distributed_disposition is StopDisposition.UNREACHABLE:
                return StopDisposition.UNREACHABLE
        if cancelled or distributed_disposition is StopDisposition.STOPPED:
            return StopDisposition.STOPPED
        return StopDisposition.ALREADY_COMPLETE

    return CooperativeStopTarget(
        target_id=agent_id,
        agent_id=agent_id,
        cancel=cancel,
        turn_ids=frozenset(turn_addresses),
        turn_request_ids=turn_request_ids,
        turn_request_generations=turn_request_generations,
        resolves_public_turns_durably=distributed_registry is not None,
    )


__all__ = ["build_runtime_stop_target"]
