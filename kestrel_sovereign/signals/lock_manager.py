"""Single ordered lock manager for the signal dispatcher.

Per SIGNAL_DISPATCHER.md §Concern 2: any caller acquiring multiple named
locks acquires them in a single global order. `CONVERSATION` is the
highest-order acquisition (always last) and is owned solely by the turn
lifecycle — the dispatcher never pre-acquires it for COGNITION sources.

The order is defined by `lock_sort_key` below: `CONVERSATION` is pinned to
the end via a (rank, name) tuple; everything else lex-orders on its enum
value within the lower rank. We do NOT sort by raw enum value because
"conversation" sorts before "memory"/"scheduler"/"wallet" alphabetically,
which would put CONVERSATION first — directly contradicting the design.

Acquisitions are instrumented (#2770). A turn that stalls while holding
`CONVERSATION` makes an agent look completely dead — every later turn queues
behind a bare `asyncio.Lock` — while its non-turn endpoints keep answering in
milliseconds. Previously that produced no log line at all on either side of the
blocking call: the only INFO record was the turn's own "process_input called",
and `turn_lifecycle` logged begin/end at DEBUG. Diagnosing one incident meant
bisecting the locked region by hand. So both *waiting for* and *holding* a lock
now emit periodic WARNINGs that name the holder and the elapsed time.

These are diagnostics, not enforcement: acquisition still blocks indefinitely
rather than failing. Turns legitimately run for minutes (the Anthropic SDK's
default read timeout alone is 600s, and generation is retried up to 5 times), so
a hard acquire deadline would cancel healthy work. The goal is that the next
stall is legible from one log line instead of a manual bisect.
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Iterable, Optional

from kestrel_sdk.signals import ResourceLock

logger = logging.getLogger(__name__)


# How long to wait before reporting, and then re-reporting, a contended
# acquisition. Contention itself is normal — turns serialize by design — so this
# is set well above a healthy turn's queueing time to stay quiet in normal
# operation.
SLOW_WAIT_WARN_SECONDS = 30.0

# How long a single holder may hold a lock before it is reported, and then
# re-reported. Above the p99 healthy turn but far below the ~600s-per-attempt
# ceiling an LLM call can reach, so a genuine stall surfaces while it is still
# happening rather than only in hindsight.
SLOW_HOLD_WARN_SECONDS = 60.0


@dataclass(frozen=True)
class LockHolder:
    """Who holds a lock, and since when (``time.monotonic``)."""

    label: str
    acquired_at: float

    def held_seconds(self) -> float:
        return time.monotonic() - self.acquired_at


def lock_sort_key(name: ResourceLock) -> tuple[int, str]:
    """Global lock acquisition order.

    `CONVERSATION` is pinned last (rank 1). Every other lock acquires in
    lex order of its enum value (rank 0). Exposed at module scope so tests
    and other components can assert against the canonical ordering.
    """
    if name == ResourceLock.CONVERSATION:
        return (1, "")
    return (0, name.value)


class OrderedLockManager:
    """Acquires a set of named locks in a single sorted pass.

    The single global order (see `lock_sort_key`) is the entire deadlock-
    freedom invariant: with one order, two callers cannot acquire the same
    pair of locks in opposite directions, so the deadlock surface is empty.
    """

    def __init__(self) -> None:
        self._locks: dict[ResourceLock, asyncio.Lock] = {}
        self._registry_lock = asyncio.Lock()
        self._holders: dict[ResourceLock, LockHolder] = {}

    async def _get(self, name: ResourceLock) -> asyncio.Lock:
        # Lazy creation; protected so first-use races don't double-create.
        if name in self._locks:
            return self._locks[name]
        async with self._registry_lock:
            if name not in self._locks:
                self._locks[name] = asyncio.Lock()
            return self._locks[name]

    @asynccontextmanager
    async def acquire(
        self, names: Iterable[ResourceLock], *, label: Optional[str] = None
    ) -> AsyncIterator[None]:
        """Acquire all named locks in canonical order; release in reverse
        on exit. Empty iterable is a no-op (yields immediately).

        ``label`` identifies the acquirer in slow-wait/slow-hold diagnostics —
        pass something a human can act on, e.g. ``"Nellie turn_ab12cd34ef56"``.
        Without it the lock's own name is reported, which tells an operator that
        *something* is stuck but not what (#2770).
        """
        ordered = sorted(set(names), key=lock_sort_key)
        acquired: list[tuple[ResourceLock, asyncio.Lock]] = []
        watchdogs: list[asyncio.Task] = []
        try:
            for name in ordered:
                lock = await self._get(name)
                await self._acquire_one(name, lock, label)
                acquired.append((name, lock))
                watchdogs.append(
                    asyncio.ensure_future(self._warn_while_holding(name))
                )
            yield
        finally:
            # Cancel diagnostics before releasing so a watchdog cannot report a
            # hold that has already ended.
            for watchdog in reversed(watchdogs):
                watchdog.cancel()
            for name, lock in reversed(acquired):
                self._holders.pop(name, None)
                lock.release()

    async def _acquire_one(
        self,
        name: ResourceLock,
        lock: asyncio.Lock,
        label: Optional[str],
    ) -> None:
        """Acquire one lock, reporting the wait if it is contended.

        The watchdog runs as a separate task rather than wrapping the acquire in
        ``wait_for``: cancelling a timed-out ``Lock.acquire`` that has already
        succeeded would drop the lock on the floor. ``asyncio.Lock.acquire``
        already handles its own cancellation correctly, so this keeps the
        acquisition path untouched and bolts observability alongside it.
        """
        if not lock.locked():
            # Uncontended fast path — no task churn on the hot path. A lock
            # taken between this check and the acquire below simply costs a
            # missed wait warning, never correctness.
            await lock.acquire()
            self._holders[name] = LockHolder(label or name.value, time.monotonic())
            return

        watchdog = asyncio.ensure_future(self._warn_while_waiting(name, label))
        try:
            await lock.acquire()
        finally:
            watchdog.cancel()
        self._holders[name] = LockHolder(label or name.value, time.monotonic())

    async def _warn_while_waiting(
        self, name: ResourceLock, label: Optional[str]
    ) -> None:
        """Periodically report a blocked acquisition, naming the holder."""
        waited = 0.0
        while True:
            await asyncio.sleep(SLOW_WAIT_WARN_SECONDS)
            waited += SLOW_WAIT_WARN_SECONDS
            holder = self._holders.get(name)
            if holder is None:
                held = "holder unknown"
            else:
                held = f"held {holder.held_seconds():.0f}s by {holder.label}"
            logger.warning(
                "%s has been waiting %.0fs for the %s lock (%s). The holder is "
                "still running; nothing is being dropped.",
                label or "an acquirer",
                waited,
                name.value,
                held,
            )

    async def _warn_while_holding(self, name: ResourceLock) -> None:
        """Periodically report a long-running holder.

        This is the signal that was missing in #2770: it fires while the stall
        is happening, from the holder's side, so an operator does not have to
        infer a wedge from the absence of logs.
        """
        while True:
            await asyncio.sleep(SLOW_HOLD_WARN_SECONDS)
            holder = self._holders.get(name)
            if holder is None:
                return
            logger.warning(
                "%s has held the %s lock for %.0fs. Every other turn for this "
                "agent is queued behind it.",
                holder.label,
                name.value,
                holder.held_seconds(),
            )

    def is_held(self, name: ResourceLock) -> bool:
        """Best-effort introspection for tests/debug. Not for routing logic."""
        lock = self._locks.get(name)
        return bool(lock and lock.locked())

    def holder(self, name: ResourceLock) -> Optional[LockHolder]:
        """Who currently holds ``name``, for diagnostics and tests.

        Best-effort like :meth:`is_held` — never use it for routing.
        """
        return self._holders.get(name)
