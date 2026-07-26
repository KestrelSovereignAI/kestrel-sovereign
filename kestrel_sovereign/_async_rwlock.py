"""Small cancellation-safe reader/writer synchronization primitives."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Self


class _DowngradedReaderLease:
    """A reader lease created by atomically downgrading a held writer."""

    def __init__(self, lock: "AsyncReaderWriterLock") -> None:
        self._lock = lock
        self._released = False

    async def __aenter__(self) -> "AsyncReaderWriterLock":
        return self._lock

    async def __aexit__(self, *_args: object) -> None:
        if not self._released:
            self._released = True
            self._lock.release_read()


class AsyncReaderWriterLock:
    """A writer-preferring asynchronous reader/writer lock.

    ``asyncio.Lock`` exposes a synchronous ``release`` method, and a few
    lifecycle callers intentionally retain that ownership shape.  This lock
    preserves that interface for writers while adding :meth:`read` for
    concurrent admission leases.  State changes never await, so cancellation
    cannot leak a reader or writer after ownership has been granted.
    """

    def __init__(self) -> None:
        self._readers = 0
        self._writer = False
        self._waiting_writers = 0
        self._changed = asyncio.Event()

    def locked(self) -> bool:
        """Whether an active reader or writer currently owns the lock."""

        return self._writer or self._readers > 0

    async def acquire(self) -> bool:
        """Acquire the exclusive writer lease.

        Once a writer queues, new readers wait behind it.  That preference is
        necessary for administrative removal and rollout transitions: a busy
        scheduler must not be able to postpone a fence indefinitely.
        """

        self._waiting_writers += 1
        try:
            while True:
                if not self._writer and self._readers == 0:
                    self._writer = True
                    return True
                self._changed.clear()
                if self._writer or self._readers:
                    await self._changed.wait()
        finally:
            self._waiting_writers -= 1
            # Wake readers when a cancelled writer was the last waiter, and
            # wake the next writer whenever ownership was claimed/released.
            self._changed.set()

    def release(self) -> None:
        """Release the exclusive writer lease."""

        if not self._writer:
            raise RuntimeError("Cannot release an un-acquired writer lock")
        self._writer = False
        self._changed.set()

    def downgrade(self) -> _DowngradedReaderLease:
        """Atomically turn a held writer into one admitted reader lease.

        This is used after a cold scheduler wake: an administrative writer
        already queued behind initialization must still drain the execution
        which it could not have prevented. New readers remain blocked behind
        that queued writer until this retained reader lease exits.
        """

        if not self._writer:
            raise RuntimeError("Cannot downgrade an un-acquired writer lock")
        self._writer = False
        self._readers += 1
        self._changed.set()
        return _DowngradedReaderLease(self)

    async def acquire_read(self) -> None:
        """Acquire a shared reader lease, respecting queued writers."""

        while True:
            if not self._writer and self._waiting_writers == 0:
                self._readers += 1
                return
            self._changed.clear()
            if self._writer or self._waiting_writers:
                await self._changed.wait()

    def release_read(self) -> None:
        """Release one shared reader lease."""

        if self._readers == 0:
            raise RuntimeError("Cannot release an un-acquired reader lock")
        self._readers -= 1
        self._changed.set()

    async def __aenter__(self) -> Self:
        await self.acquire()
        return self

    async def __aexit__(self, *_args: object) -> None:
        self.release()

    @asynccontextmanager
    async def read(self) -> AsyncIterator["AsyncReaderWriterLock"]:
        """Hold a shared admission lease until the context exits."""

        await self.acquire_read()
        try:
            yield self
        finally:
            self.release_read()
