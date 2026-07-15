"""Cancellation-safe ownership of small internal asyncio tasks."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar, cast


logger = logging.getLogger(__name__)

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

    A separate success-only completion waiter avoids copying the owned task's
    exception into a shield future.  The owned task result is retrieved exactly
    once after terminalization, so neither task nor shield warnings are orphaned.
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
                continue
    finally:
        task.remove_done_callback(mark_complete)
        if not completion_task.done():
            completion_task.cancel()

        # There must be no await between owned-task terminalization and result
        # retrieval: another Task.cancel() in that window could otherwise
        # interrupt the very bookkeeping that prevents orphan warnings.
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
            logger.error(
                "%s failed while preserving caller cancellation: %s",
                operation,
                outcome.error,
                exc_info=(
                    type(outcome.error),
                    outcome.error,
                    outcome.error.__traceback__,
                ),
            )
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
    return raise_owned_outcome(outcome, operation="Owned blocking operation")
