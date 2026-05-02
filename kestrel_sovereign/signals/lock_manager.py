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
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator, Iterable

from kestrel_sdk.signals import ResourceLock


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
        self, names: Iterable[ResourceLock]
    ) -> AsyncIterator[None]:
        """Acquire all named locks in canonical order; release in reverse
        on exit. Empty iterable is a no-op (yields immediately)."""
        ordered = sorted(set(names), key=lock_sort_key)
        acquired: list[asyncio.Lock] = []
        try:
            for name in ordered:
                lock = await self._get(name)
                await lock.acquire()
                acquired.append(lock)
            yield
        finally:
            for lock in reversed(acquired):
                lock.release()

    def is_held(self, name: ResourceLock) -> bool:
        """Best-effort introspection for tests/debug. Not for routing logic."""
        lock = self._locks.get(name)
        return bool(lock and lock.locked())
