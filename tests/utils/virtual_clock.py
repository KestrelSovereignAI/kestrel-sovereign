"""Event-loop clocks that only move when a test advances them.

Loop deadlines (``call_at``, ``asyncio.timeout``, ``asyncio.wait`` timeouts)
are measured against ``loop.time()``. Pinning that clock lets a test decide
exactly which stage a deadline lands in, instead of racing the wall clock
under parallel load.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from typing import Iterator
from unittest.mock import patch


class VirtualClock:
    """A monotonic clock that moves only through :meth:`advance`."""

    def __init__(self, now: float = 0.0) -> None:
        self._now = now

    def time(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("a virtual clock cannot move backwards")
        self._now += seconds


class VirtualClockLoop(asyncio.SelectorEventLoop):
    """An event loop whose ``time()`` only moves when a test advances it, so
    ``asyncio.timeout`` deadlines are measured against virtual seconds."""

    def __init__(self) -> None:
        super().__init__()
        self._clock = VirtualClock()

    def time(self) -> float:
        return self._clock.time()

    def advance(self, seconds: float) -> None:
        self._clock.advance(seconds)


@contextmanager
def pinned_loop_clock() -> Iterator[VirtualClock]:
    """Freeze the running loop's clock for an ``async`` test.

    The clock starts at the loop's current time. While pinned, a timer comes
    due only once the test advances past it, and that includes timers
    scheduled before the pin. Leaving the block restores the real clock.
    """

    loop = asyncio.get_running_loop()
    clock = VirtualClock(loop.time())
    with patch.object(loop, "time", clock.time):
        yield clock
