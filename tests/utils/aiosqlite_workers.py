"""Failure injection helpers for aiosqlite worker-lifecycle tests."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager, suppress
from typing import Any
from unittest.mock import patch

import aiosqlite


def aiosqlite_worker(connection: aiosqlite.Connection) -> threading.Thread:
    """Return the worker for either supported aiosqlite threading model."""
    return getattr(connection, "_thread", connection)


@contextmanager
def delay_aiosqlite_worker_exit(
    release_worker: threading.Event,
    worker_exit_delayed: threading.Event,
    *,
    should_delay: Callable[[threading.Thread], bool] | None = None,
) -> Iterator[list[threading.Thread]]:
    """Delay aiosqlite's final worker return on 0.21 and 0.22+.

    ``Connection.close()`` acknowledges its stop sentinel before this final
    return, which reproduces the event-loop teardown race without mocking the
    connection lifecycle itself.
    """
    workers: list[threading.Thread] = []

    def _delay_current_worker(worker: threading.Thread) -> None:
        if should_delay is not None and not should_delay(worker):
            return
        workers.append(worker)
        worker_exit_delayed.set()
        release_worker.wait()

    worker_target = getattr(aiosqlite.core, "_connection_worker_thread", None)

    if worker_target is not None:

        def delayed_worker(*args, **kwargs):
            try:
                return worker_target(*args, **kwargs)
            finally:
                _delay_current_worker(threading.current_thread())

        patcher = patch.object(
            aiosqlite.core, "_connection_worker_thread", delayed_worker,
        )
    else:
        original_run = aiosqlite.Connection.run

        def delayed_run(connection, *args, **kwargs):
            try:
                return original_run(connection, *args, **kwargs)
            finally:
                _delay_current_worker(connection)

        patcher = patch.object(aiosqlite.Connection, "run", delayed_run)

    with patcher:
        yield workers


async def wait_until_aiosqlite_worker_exit_is_delayed(
    worker_exit_delayed: threading.Event,
) -> None:
    """Yield until a failure-injected worker has reached its final return."""
    while not worker_exit_delayed.is_set():
        await asyncio.sleep(0)


async def wait_for_lifecycle_checkpoint(
    checkpoint: Awaitable[Any],
    lifecycle_task: asyncio.Task[Any],
    *,
    description: str,
    harness_timeout: float = 10.0,
    require_live_lifecycle: bool = True,
) -> None:
    """Wait for a checkpoint unless the lifecycle terminates first.

    Lifecycle regressions should fail because the owner completed, failed, or
    cancelled before reaching the expected checkpoint—not because a busy CI
    runner did not schedule two cooperating tasks within a one-second wall
    clock window.  ``harness_timeout`` remains a generous deadlock guard; it
    is not part of the behavior being asserted.

    Pass ``require_live_lifecycle=False`` where the asserted contract is that
    the lifecycle *terminates*—for a deliberately short shutdown drain, the
    owner raising before the checkpoint is the behavior under test, not a
    regression.  The injected worker still sets the checkpoint on its way out,
    so the wait stays bounded by the same guard; imposing the live-owner
    precondition there would turn a legitimate scheduling order into a flake.
    """
    checkpoint_task = asyncio.ensure_future(checkpoint)
    awaited: set[asyncio.Future[Any]] = {checkpoint_task}
    if require_live_lifecycle:
        awaited.add(lifecycle_task)
    try:
        done, _pending = await asyncio.wait(
            awaited,
            timeout=harness_timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )
        # Fail closed when both tasks become ready in the same event-loop
        # turn.  A completed lifecycle cannot prove that it was still live
        # when the checkpoint was reached, so owner termination takes
        # precedence over a simultaneously observed checkpoint.
        if lifecycle_task in done:
            if lifecycle_task.cancelled():
                raise AssertionError(
                    f"Lifecycle cancelled before {description}"
                )
            failure = lifecycle_task.exception()
            if failure is not None:
                raise AssertionError(
                    f"Lifecycle failed before {description}"
                ) from failure
            raise AssertionError(
                f"Lifecycle completed before {description}"
            )

        if checkpoint_task in done:
            await checkpoint_task
            return

        raise AssertionError(
            f"Lifecycle and {description} made no progress within the "
            f"{harness_timeout:g}s test-harness guard"
        )
    finally:
        if not checkpoint_task.done():
            checkpoint_task.cancel()
        with suppress(asyncio.CancelledError):
            await checkpoint_task
