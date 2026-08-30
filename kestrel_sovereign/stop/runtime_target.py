"""Shared live-agent adapter for cooperative Stop authorities.

This module is the only bridge from typed Stop requests to an agent's
request-lifecycle API. HTTP doors choose scope and trusted identity; they do
not reimplement cancellation, generation fencing, or completion evidence.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

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
                or not turn_id.strip()
                or not isinstance(binding, tuple)
                or len(binding) != 2
                or not isinstance(binding[0], str)
                or not binding[0].strip()
            ):
                raise TypeError("agent turn binding inventory is malformed")
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
        or not turn_id.strip()
        or not isinstance(request_id, str)
        or not request_id.strip()
        for turn_id, request_id in raw_index.items()
    ):
        raise TypeError("agent turn request inventory is malformed")
    return dict(raw_index), {}


def build_runtime_stop_target(
    agent: object,
    *,
    agent_id: str,
    explicit_request_id: str | None = None,
    distributed_registry: Any | None = None,
) -> CooperativeStopTarget:
    """Snapshot one agent and bind its typed cooperative cancellation action."""

    if not isinstance(agent_id, str) or not agent_id.strip():
        raise ValueError("runtime Stop target requires a concrete agent identity")
    if explicit_request_id is not None and (
        not isinstance(explicit_request_id, str) or not explicit_request_id
    ):
        raise ValueError("explicit Stop request identity must be concrete")

    active_at_resolution = _active_request_snapshot(
        agent,
        explicit_request_id=explicit_request_id,
    )
    turn_request_ids, turn_request_generations = _turn_request_bindings(agent)
    turn_addresses = active_at_resolution.union(turn_request_ids)

    async def cancel(stop_request: StopRequest) -> StopDisposition:
        distributed_ticket = None
        if distributed_registry is not None:
            if stop_request.scope is StopScope.TURN:
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
        cancelled_request_ids: list[str | None] = []
        if stop_request.scope is StopScope.TURN:
            cancel_kwargs: dict[str, object] = {"request_id": stop_request.target}
            if stop_request.request_generation is not None:
                cancel_kwargs["generation"] = stop_request.request_generation
            cancelled = bool(cancel_current(**cancel_kwargs))
            if cancelled:
                cancelled_request_ids.append(stop_request.target)
            else:
                reserve = getattr(type(agent), "reserve_request_cancellation", None)
                if stop_request.request_generation is None and callable(reserve):
                    reserve(agent, stop_request.target)
        else:
            # Receipt preflight and claim I/O happen after the inventory
            # snapshot. Re-read at cancellation linearization so an admitted
            # turn cannot outlive an agent- or host-wide STOPPED receipt.
            turns_to_cancel = active_at_resolution.union(
                _active_request_snapshot(agent)
            )
            cancelled = False
            for request_id in sorted(turns_to_cancel):
                request_cancelled = bool(cancel_current(request_id=request_id))
                if request_cancelled:
                    cancelled_request_ids.append(request_id)
                cancelled = request_cancelled or cancelled
            if not turns_to_cancel:
                cancelled = bool(cancel_current(request_id=None))
                if cancelled:
                    cancelled_request_ids.append(None)

        if cancelled:
            wait_for_completion = getattr(agent, "wait_for_request_completion", None)
            if not callable(wait_for_completion):
                raise RuntimeError("agent cannot confirm request lifecycle completion")
            abandoned = False
            for request_id in cancelled_request_ids:
                wait_kwargs: dict[str, object] = {}
                if (
                    stop_request.scope is StopScope.TURN
                    and stop_request.request_generation is not None
                ):
                    wait_kwargs["generation"] = stop_request.request_generation
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
    )


__all__ = ["build_runtime_stop_target"]
