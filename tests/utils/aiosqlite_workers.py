"""Failure injection helpers for aiosqlite worker-lifecycle tests."""

from __future__ import annotations

import asyncio
import threading
from contextlib import contextmanager
from typing import Callable, Iterator
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
