"""#4040: a cancelled PostgreSQL close still closes every pool it owns.

Close joins a runtime-owner renewal in flight by taking renewal's lock before
it closes renewal's pool. That wait was cancellable: a close cancelled while a
renewal held the lock raised at once, before any pool closer ran, and left
the renewal pool, the advisory pool and the operational pool all open and
attached. Close now keeps the cancellation, closes every pool once the
renewal has finished, then raises it.

The pools are real asyncpg pools with no minimum size, so none ever opens a
connection; their closers run as they do against a live server. The same
case runs on a real PostgreSQL in
``tests/integration/test_runtime_owner_liveness_pg.py``.
"""

from __future__ import annotations

import asyncio

import asyncpg
import pytest

from kestrel_sovereign.storage.db.postgres import PostgresBackend

pytestmark = [pytest.mark.asyncio, pytest.mark.timeout(30)]

POOLS = ("_runtime_owner_renewal_pool", "_advisory_pool", "_pool")


async def _unconnected_pool():
    async def refuse(*_args, **_kwargs):
        raise AssertionError("this pool must never open a connection")

    return await asyncpg.create_pool(min_size=0, max_size=1, connect=refuse)


async def _until_waiting(lock: asyncio.Lock, waiter: asyncio.Task) -> None:
    """Return once ``waiter`` waits for ``lock``, which another task holds."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 5
    while not getattr(lock, "_waiters", None):
        assert not waiter.done(), "close ended before it waited for the renewal"
        assert loop.time() < deadline, "close never waited for the renewal"
        await asyncio.sleep(0)


async def test_cancelling_close_while_it_joins_a_renewal_still_closes_every_pool():
    backend = PostgresBackend("postgresql://renewal@127.0.0.1:1/never-connected")
    pools = {attribute: await _unconnected_pool() for attribute in POOLS}
    for attribute, pool in pools.items():
        setattr(backend, attribute, pool)
    lock = backend._runtime_owner_renewal_lock()
    # A renewal in flight holds renewal's lock across its write.
    await lock.acquire()
    closing = asyncio.create_task(backend.close())
    try:
        await _until_waiting(lock, closing)
        closing.cancel()
        for _ in range(5):
            await asyncio.sleep(0)
        assert not closing.done(), "a cancelled close stopped joining the renewal"
        assert not any(pool.is_closing() for pool in pools.values()), (
            "close closed a pool under the renewal in flight"
        )
    finally:
        lock.release()

    done, _ = await asyncio.wait({closing}, timeout=10)
    assert closing in done, "close never finished once the renewal had"
    assert closing.cancelled(), "close did not deliver the caller's cancellation"
    for attribute, pool in pools.items():
        assert pool.is_closing(), f"close left {attribute} open"
        assert getattr(backend, attribute) is None, f"close left {attribute} attached"
    assert backend._runtime_owner_renewal_retired is True
    assert not lock.locked()


async def test_a_close_cancelled_repeatedly_still_closes_every_pool():
    """Each cancellation while close waits is kept; none skips a closer."""
    backend = PostgresBackend("postgresql://renewal@127.0.0.1:1/never-connected")
    pools = {attribute: await _unconnected_pool() for attribute in POOLS}
    for attribute, pool in pools.items():
        setattr(backend, attribute, pool)
    lock = backend._runtime_owner_renewal_lock()
    await lock.acquire()
    closing = asyncio.create_task(backend.close())
    try:
        for _ in range(3):
            await _until_waiting(lock, closing)
            closing.cancel()
            await asyncio.sleep(0)
        assert not closing.done()
    finally:
        lock.release()

    done, _ = await asyncio.wait({closing}, timeout=10)
    assert closing in done and closing.cancelled()
    for attribute, pool in pools.items():
        assert pool.is_closing() and getattr(backend, attribute) is None, attribute
