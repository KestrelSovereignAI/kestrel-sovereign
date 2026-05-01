"""Single ordered lock manager for the signal dispatcher.

Per SIGNAL_DISPATCHER.md §Concern 2: any caller acquiring multiple named
locks acquires them in lexicographic order. `CONVERSATION` is the
highest-order acquisition (always last) and is owned solely by the turn
lifecycle — the dispatcher never pre-acquires it for COGNITION sources.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator, Iterable

from kestrel_sdk.signals import ResourceLock


class OrderedLockManager:
    """Acquires a set of named locks in a single sorted pass.

    Lex order on lock name is the global invariant. There is no second
    order, so the only deadlock surface (two callers acquiring the same
    pair in different orders) cannot exist.
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
        """Acquire all named locks in lex order; release in reverse on exit.

        Empty iterable is a no-op (yields immediately).
        """
        ordered = sorted(set(names), key=lambda r: r.value)
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
