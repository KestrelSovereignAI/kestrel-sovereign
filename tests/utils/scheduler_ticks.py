"""Drive one scheduler poll and wait for the occurrences it admitted.

Since #3465 ``SchedulerRunner._tick`` only admits due occurrences onto
runner-owned tasks and returns; the poll loop never waits for them. A test
that asserts on an occurrence's outcome therefore waits for it explicitly.
"""

from __future__ import annotations

import asyncio
from typing import Any


async def settle_occurrences(runner: Any) -> None:
    """Wait until no occurrence the runner admitted is still in flight.

    Cancelling the waiter cancels those occurrences, as cancelling a tick
    that awaited its batch once did.
    """

    while True:
        pending = [
            occurrence.task
            for occurrence in runner._occurrences.values()
            if not occurrence.task.done()
        ]
        if not pending:
            return
        await asyncio.gather(*pending, return_exceptions=True)


async def tick_and_settle(runner: Any) -> None:
    """Run one poll, then wait for every occurrence still in flight."""

    await runner._tick()
    await settle_occurrences(runner)
