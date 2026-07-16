"""Cancellation-safe ownership of internal asyncio tasks.

Callers that create cleanup, launch, or persistence tasks remain responsible
for reaching a terminal outcome even when their own task is cancelled. This
module provides that host-wide ownership rule without depending on subprocess,
Talon, compute, or training concepts.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar, cast


_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class OwnedTaskOutcome(Generic[_T]):
    """Terminal result of an owned task plus caller cancellation state."""

    result: _T | None
    error: BaseException | None
    cancellation: asyncio.CancelledError | None


async def await_owned_task(
    task: asyncio.Task[_T],
    pending_cancellation: asyncio.CancelledError | None = None,
) -> OwnedTaskOutcome[_T]:
    """Retrieve ``task`` despite repeated cancellation of its caller.

    The shielded waiter carries only a successful event result, never the
    owned task's exception. The owned result is retrieved exactly once after
    terminalization, avoiding orphaned-task and shield-future warnings.
    """

    completion = asyncio.Event()

    def mark_complete(_task: asyncio.Task[_T]) -> None:
        completion.set()

    task.add_done_callback(mark_complete)
    completion_task = asyncio.create_task(completion.wait())
    try:
        while not task.done():
            try:
                await asyncio.shield(completion_task)
            except asyncio.CancelledError as error:
                if pending_cancellation is None:
                    pending_cancellation = error
    finally:
        task.remove_done_callback(mark_complete)
        if not completion_task.done():
            completion_task.cancel()

        # Do not await between terminalization and result retrieval. A further
        # Task.cancel() in that window could interrupt the bookkeeping that
        # prevents an unobserved completion-task exception.
        def retrieve_completion(waiter: asyncio.Task[bool]) -> None:
            try:
                waiter.result()
            except asyncio.CancelledError:
                pass

        if completion_task.done():
            retrieve_completion(completion_task)
        else:
            completion_task.add_done_callback(retrieve_completion)

    try:
        return OwnedTaskOutcome(task.result(), None, pending_cancellation)
    except BaseException as error:
        return OwnedTaskOutcome(None, error, pending_cancellation)


def raise_owned_outcome(
    outcome: OwnedTaskOutcome[_T],
    *,
    operation: str,
) -> _T:
    """Apply caller-cancellation precedence to a retrieved owned outcome."""

    if outcome.cancellation is not None:
        if outcome.error is not None:
            outcome.cancellation.add_note(f"{operation} also failed: {outcome.error}")
            raise outcome.cancellation from outcome.error
        raise outcome.cancellation
    if outcome.error is not None:
        raise outcome.error
    return cast(_T, outcome.result)


async def run_blocking_operation(
    operation: Callable[..., _T],
    *args: object,
) -> _T:
    """Finish a blocking operation before propagating caller cancellation."""

    operation_task = asyncio.create_task(asyncio.to_thread(operation, *args))
    outcome = await await_owned_task(operation_task)
    return raise_owned_outcome(outcome, operation="owned blocking operation")


__all__ = [
    "OwnedTaskOutcome",
    "await_owned_task",
    "raise_owned_outcome",
    "run_blocking_operation",
]
