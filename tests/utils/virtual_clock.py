"""Event-loop clocks that only move when a test advances them.

Loop deadlines (``call_at``, ``asyncio.timeout``, ``asyncio.wait`` timeouts)
are measured against ``loop.time()``. Pinning that clock lets a test decide
exactly which stage a deadline lands in, instead of racing the wall clock
under parallel load.

A test that moves the loop clock cannot bound its own waits with that clock:
a pinned clock never fires the bound, and a jump fires it at once.
:func:`await_within_hang_guard` and :func:`wait_for_thread_event` measure
their hang guards on a thread instead.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Awaitable, Callable
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Iterator, TypeVar
from unittest.mock import patch

_T = TypeVar("_T")

# Bound for a wait that must finish promptly. Generous, so a GC pause on a
# loaded xdist worker cannot fail a passing test.
HANG_GUARD_SECONDS = 30.0


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


@contextmanager
def pinned_shacl_validation_clock() -> Iterator[VirtualClock]:
    """Freeze the clock SHACL validation measures its wall-time budget on.

    ``GovernedShaclValidationService`` bounds each run by
    ``ShaclValidationLimits.max_wall_time_seconds`` of ``time.monotonic``, so
    a stalled host can exhaust it inside a validation that takes a few
    milliseconds, and the governed write is rejected (#3858). Only that
    module's ``time`` binding is replaced: the event loop and every other
    clock keep real time. A test that asserts what a write stores pins it; a
    test of the budget itself spends it with ``advance``.
    """

    from kestrel_sovereign.knowledge import shacl_validation

    clock = VirtualClock(time.monotonic())
    with patch.object(
        shacl_validation, "time", SimpleNamespace(monotonic=clock.time)
    ):
        yield clock


class AdvanceableClock:
    """A clock that keeps real time and jumps forward on :meth:`advance`."""

    def __init__(self, real_time: Callable[[], float]) -> None:
        self._real_time = real_time
        self._skipped = 0.0

    def time(self) -> float:
        return self._real_time() + self._skipped

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("a loop clock cannot move backwards")
        self._skipped += seconds


@contextmanager
def advanceable_loop_clock() -> Iterator[AdvanceableClock]:
    """Let an ``async`` test jump the running loop's clock forward.

    Unlike :func:`pinned_loop_clock`, time keeps passing, so polls and the
    real work around a deadline run as usual. ``advance`` skips ahead, and
    every timer the skip passes comes due on the loop's next turn: a test
    expires one deadline at the stage it chooses, without racing the wall
    clock.

    A skip expires every loop-clock deadline it passes, an ``asyncio.timeout``
    hang guard included, so guard waits with :func:`await_within_hang_guard`.
    Leaving the block takes the skips back out, so a timer armed after one
    comes due that much later: settle the work before leaving.
    """

    loop = asyncio.get_running_loop()
    clock = AdvanceableClock(loop.time)
    with patch.object(loop, "time", clock.time):
        yield clock


async def await_within_hang_guard(
    awaitable: Awaitable[_T],
    *,
    step: str,
    seconds: float = HANG_GUARD_SECONDS,
) -> _T:
    """Await ``awaitable``; fail naming ``step`` if it hangs.

    The guard runs on a thread, so pinning or advancing the loop clock does
    not move it, and it calls no ``asyncio`` helper a test might instrument.
    When it fires, an awaited task or future is left as it is for the test's
    own cleanup; an awaited coroutine is cancelled.
    """
    loop = asyncio.get_running_loop()
    future = asyncio.ensure_future(awaitable)
    owns_future = future is not awaitable
    settled = loop.create_future()

    def settle(_finished: object = None) -> None:
        if not settled.done():
            settled.set_result(None)

    future.add_done_callback(settle)
    timer = threading.Timer(
        seconds, loop.call_soon_threadsafe, args=(settle,)
    )
    timer.daemon = True
    timer.start()
    try:
        await settled
    except BaseException:
        if owns_future:
            future.cancel()
        raise
    finally:
        timer.cancel()
        future.remove_done_callback(settle)
    if not future.done():
        if owns_future:
            future.cancel()
        raise AssertionError(
            f"{step} did not happen within the {seconds:g}s hang guard"
        )
    return future.result()


async def wait_for_thread_event(
    event: threading.Event,
    *,
    step: str,
    seconds: float = HANG_GUARD_SECONDS,
) -> None:
    """Wait for ``event``, set on another thread; fail naming ``step``."""
    if not await asyncio.to_thread(event.wait, seconds):
        raise AssertionError(
            f"{step} did not happen within the {seconds:g}s hang guard"
        )
