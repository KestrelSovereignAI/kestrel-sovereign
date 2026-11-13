"""#4040: SQLite runtime-owner renewal beside the shared connection and close.

Renewal writes on a connection of its own, so it is the first writer in the
process that does not share the backend's connection. Two consequences:

* A deferred transaction on the shared connection that read, then wrote after
  a renewal committed, failed at once with ``database is locked``: WAL refuses
  to upgrade a stale read snapshot, and the busy timeout does not apply. Every
  ``transaction()`` therefore takes the writer slot at ``BEGIN``.
* Close joins a renewal before retiring its connection. A renewal waiting for
  the writer slot behind another connection held that join for its whole busy
  timeout, longer than the reservation agent shutdown makes for close. It now
  tries in attempts of ``_SQLITE_OWNER_RENEWAL_BUSY_SLICE_S`` and ends, having
  written nothing, once close has retired it; close's connections take their
  deadline after the join.

* Renewal keeps its connection between renewals, so opening it is no longer
  a check of the file it reaches on every renewal. Each renewal checks that the
  database pathname still names the file this backend connected to, before it
  waits for the writer slot and again before it commits.

Each test runs a real file database, with another connection holding the
writer slot where one is needed.
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

import kestrel_sovereign.storage.db.sqlite as sqlite_module
from kestrel_sovereign.signals.durable import DurableSignalStore
from kestrel_sovereign.storage.db import ConnectionError, SQLiteBackend
from tests.utils.aiosqlite_workers import aiosqlite_worker

pytestmark = [pytest.mark.asyncio, pytest.mark.timeout(60)]

AGENT = "did:example:renewal-4040"
OWNER = "dispatcher:renewal-4040"
OWNERS = DurableSignalStore.RUNTIME_OWNERS


@pytest.fixture
async def owner(tmp_path):
    """A connected file backend whose store has registered ``OWNER``."""
    backend = SQLiteBackend(str(tmp_path / "renewal.db"))
    await backend.connect()
    store = DurableSignalStore(backend)
    await store.initialize()
    await store.register_runtime_owner(agent_id=AGENT, owner_id=OWNER)
    await backend.execute("CREATE TABLE ordinary (id INTEGER PRIMARY KEY, value TEXT)")
    await backend.execute("INSERT INTO ordinary (id, value) VALUES (1, 'before')")
    try:
        yield backend, store
    finally:
        if backend.is_connected:
            await backend.close()


def _owner_row(backend):
    """The owner's heartbeat and stop, read on a connection of this test's own."""
    return _owner_row_at(backend.db_path)


def _owner_row_at(path):
    """The owner's heartbeat and stop in the database file at ``path``."""
    with closing(sqlite3.connect(path)) as peer:
        return peer.execute(
            f"SELECT heartbeat_at, stopped_at FROM {OWNERS} "
            "WHERE agent_id = ? AND owner_id = ?",
            (AGENT, OWNER),
        ).fetchone()


def _renew(backend, store, *, sampled=None):
    """Start one renewal; ``sampled`` collects the loop time it samples ``now``."""
    loop = asyncio.get_running_loop()

    def timestamp():
        if sampled is not None:
            sampled.append(loop.time())
        return store.to_timestamp_param(store.now_utc())

    return asyncio.create_task(
        backend.retain_unretired_runtime_owner(
            agent_id=AGENT, owner_id=OWNER, timestamp=timestamp,
        )
    )


class _WriterSlotHolder:
    """Another connection holding SQLite's writer slot until released."""

    def __init__(self, backend):
        self._connection = sqlite3.connect(backend.db_path, isolation_level=None)
        self._connection.execute("BEGIN IMMEDIATE")

    def release(self):
        if self._connection is not None:
            self._connection.execute("ROLLBACK")
            self._connection.close()
            self._connection = None


@pytest.fixture
def writer_slot_holders():
    holders = []

    def hold(backend):
        holder = _WriterSlotHolder(backend)
        holders.append(holder)
        return holder

    yield hold
    for holder in holders:
        holder.release()


async def _until_an_attempt_starts(backend, monkeypatch):
    """Return just as a waiting renewal starts another attempt for the slot.

    The attempt then runs for about ``_SQLITE_OWNER_RENEWAL_BUSY_SLICE_S``
    before it fails on the held slot, so whatever the caller does next
    happens while the renewal holds its lock and waits.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 10
    while getattr(backend, "_runtime_owner_renewal_connection", None) is None:
        assert loop.time() < deadline, "the renewal never opened its connection"
        await asyncio.sleep(0.01)
    connection = backend._runtime_owner_renewal_connection
    execute = connection.execute
    attempt = asyncio.Event()

    def watching_execute(sql, *args, **kwargs):
        if sql == "BEGIN IMMEDIATE":
            attempt.set()
        return execute(sql, *args, **kwargs)

    monkeypatch.setattr(connection, "execute", watching_execute)
    # An attempt already running when the watch began may be near its end.
    for _ in range(2):
        attempt.clear()
        await asyncio.wait_for(attempt.wait(), 10)
    return connection


async def _retired(connections, backend):
    """Each connection's worker has exited or is held as a retained close."""
    for connection in connections:
        assert (
            not aiosqlite_worker(connection).is_alive()
            or connection in backend._retired_connection_closes
        ), "a connection close left neither retired nor retained"


async def test_a_renewal_commit_does_not_fail_a_read_then_write_transaction(
    owner,
):
    """Operator P2-2 on PR #4050: a renewal commit between a transaction's
    read and its write failed the write at once with ``database is locked``.
    The transaction now takes the writer slot at ``BEGIN``, so the renewal
    waits for it instead, and commits once the transaction has."""
    backend, store = owner
    sampled: list[float] = []
    loop = asyncio.get_running_loop()
    renewal = None
    async with backend.transaction():
        assert await backend.fetch_one(
            f"SELECT heartbeat_at FROM {OWNERS} WHERE owner_id = ?", (OWNER,)
        ) is not None
        before = _owner_row(backend)
        renewal = _renew(backend, store, sampled=sampled)
        # A renewal free to commit does so within a few attempts.
        give_up = loop.time() + 10 * sqlite_module._SQLITE_OWNER_RENEWAL_BUSY_SLICE_S
        while not renewal.done() and _owner_row(backend) == before and loop.time() < give_up:
            await asyncio.sleep(0.01)
        await backend.execute("UPDATE ordinary SET value = 'after' WHERE id = 1")
    committed_at = loop.time()

    assert await asyncio.wait_for(renewal, 10) is True
    assert sampled and sampled[-1] >= committed_at, (
        "the renewal sampled its heartbeat before the transaction committed"
    )
    assert _owner_row(backend) != before
    row = await backend.fetch_one("SELECT value FROM ordinary WHERE id = 1")
    assert tuple(row) == ("after",)


async def test_close_does_not_wait_out_a_renewal_waiting_for_the_slot(
    owner, writer_slot_holders, monkeypatch,
):
    """Meridian's P2-1 (a): close joins only a renewal that holds the writer
    slot. One still waiting for it ends at its next attempt having written
    nothing, so close fits the reservation agent shutdown makes for it, and
    its connections' deadline is taken after the join."""
    backend, store = owner
    loop = asyncio.get_running_loop()
    closes: list[tuple[object, float]] = []
    close_connection = sqlite_module._close_aiosqlite_connection

    async def recording_close(connection, **kwargs):
        closes.append((connection, kwargs["deadline"]))
        return await close_connection(connection, **kwargs)

    monkeypatch.setattr(sqlite_module, "_close_aiosqlite_connection", recording_close)
    before = _owner_row(backend)
    holder = writer_slot_holders(backend)
    renewal = _renew(backend, store)
    ended: list[float] = []
    renewal.add_done_callback(lambda _: ended.append(loop.time()))
    renewal_connection = await _until_an_attempt_starts(backend, monkeypatch)
    connections = [backend._connection, renewal_connection]

    started = loop.time()
    async with asyncio.timeout(backend.minimum_close_timeout_s):
        await backend.close()
    # Close waited for the attempt running when it began.
    assert ended[0] - started >= sqlite_module._SQLITE_OWNER_RENEWAL_BUSY_SLICE_S / 2

    with pytest.raises(ConnectionError, match="original backend"):
        await renewal
    for connection in connections:
        assert not aiosqlite_worker(connection).is_alive()
    assert {connection for connection, _ in closes} == set(connections)
    # The join did not consume the budget every connection's close shares.
    budget = sqlite_module._minimum_close_timeout_s()
    for _, deadline in closes:
        assert deadline >= ended[0] + budget - 0.005
    holder.release()
    assert _owner_row(backend) == before


async def test_cancelling_close_during_the_join_still_retires_both_connections(
    owner, writer_slot_holders, monkeypatch,
):
    """Meridian's P2-1 (b): a close cancelled while it joins a renewal keeps
    the cancellation, retires every connection, then raises it."""
    backend, store = owner
    holder = writer_slot_holders(backend)
    renewal = _renew(backend, store)
    renewal_connection = await _until_an_attempt_starts(backend, monkeypatch)
    connections = [backend._connection, renewal_connection]

    closing_task = asyncio.create_task(backend.close())
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 5
    while not getattr(backend, "_runtime_owner_renewal_retired", False):
        assert loop.time() < deadline, "close never began"
        await asyncio.sleep(0)
    await asyncio.sleep(0)
    # Close is waiting for the renewal, which holds its lock mid-attempt.
    assert backend._runtime_owner_renewal_lock().locked()
    assert not renewal.done() and not closing_task.done()
    closing_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing_task

    await _retired(connections, backend)
    assert backend._connection is None
    with pytest.raises(ConnectionError, match="original backend"):
        await renewal
    holder.release()


async def test_the_agent_reserves_the_close_budget_the_backend_asks_for(tmp_path):
    """Meridian's P2-1 (c): close's reservation covers one renewal attempt
    and one connection's close, and the agent reserves exactly that."""
    from kestrel_sovereign.kestrel_agent import _minimum_storage_close_timeout
    from kestrel_sovereign.storage.async_storage import AsyncStorage

    backend = SQLiteBackend(str(tmp_path / "budget.db"))
    assert backend.minimum_close_timeout_s >= (
        sqlite_module._SQLITE_OWNER_RENEWAL_BUSY_SLICE_S
        + sqlite_module.AIOSQLITE_WORKER_SHUTDOWN_TIMEOUT_S
    )
    storage = AsyncStorage(backend=backend)
    assert _minimum_storage_close_timeout(storage) == backend.minimum_close_timeout_s


async def test_a_cancelled_renewal_ends_with_its_cancellation(
    owner, writer_slot_holders, monkeypatch,
):
    """Operator P3 on PR #4050: a renewal cancelled while it waits for the
    slot ended with its connection's close overrun instead of its own
    cancellation. The cancellation wins; the overrun is kept as its cause."""
    backend, store = owner
    holder = writer_slot_holders(backend)
    renewal = _renew(backend, store)
    connection = await _until_an_attempt_starts(backend, monkeypatch)
    # Close it faster than the attempt can end, so its close overruns.
    monkeypatch.setattr(sqlite_module, "AIOSQLITE_WORKER_SHUTDOWN_TIMEOUT_S", 0.001)
    renewal.cancel()
    with pytest.raises(asyncio.CancelledError) as cancelled:
        await renewal
    assert isinstance(cancelled.value.__cause__, sqlite_module.SQLiteCloseOverrunError)
    monkeypatch.undo()
    holder.release()
    await asyncio.to_thread(aiosqlite_worker(connection).join, 10)
    assert not aiosqlite_worker(connection).is_alive()
    assert backend._runtime_owner_renewal_connection is None


async def test_renewal_keeps_its_connection_until_close_retires_it(owner):
    """Operator P3 on PR #4050: renewal writes on one connection across
    renewals, and close retires it."""
    backend, store = owner
    assert await _renew(backend, store) is True
    connection = backend._runtime_owner_renewal_connection
    assert connection is not None
    assert await _renew(backend, store) is True
    assert backend._runtime_owner_renewal_connection is connection

    await backend.close()
    assert backend._runtime_owner_renewal_connection is None
    assert not aiosqlite_worker(connection).is_alive()
    # A renewal after close finds no connection and opens none.
    with pytest.raises(ConnectionError):
        await _renew(backend, store)
    assert getattr(backend, "_runtime_owner_renewal_connection", None) is None


def _replace_database_pathname(backend, directory):
    """Move the connected database aside and put another database at its path.

    The backend's connections still reach the original file, which this
    returns the new path of; its path now names a database it never opened.
    """
    path = Path(backend.db_path)
    retained = directory / "retained-original.db"
    for suffix in ("", "-wal", "-shm"):
        part = Path(str(path) + suffix)
        if part.exists():
            os.rename(part, str(retained) + suffix)
    with closing(sqlite3.connect(path)) as replacement:
        replacement.execute("CREATE TABLE replacement (id INTEGER PRIMARY KEY)")
        replacement.commit()
    return retained


async def test_a_renewal_refuses_a_replaced_database_pathname(owner, tmp_path):
    """Self-review of #4040: renewal keeps its connection between renewals,
    and opening it was the only check of the file it reached. Once the
    database pathname was replaced, a renewal through the kept connection
    still reported the owner renewed, on a file this backend no longer
    reaches by its path. Each renewal now checks the file before it waits
    for the writer slot, so it begins no write there, and retires its
    connection."""
    backend, store = owner
    assert await _renew(backend, store) is True
    connection = backend._runtime_owner_renewal_connection
    assert connection is not None
    retained = _replace_database_pathname(backend, tmp_path)
    before = _owner_row_at(retained)

    sampled: list[float] = []
    with pytest.raises(ConnectionError, match="inode changed"):
        await _renew(backend, store, sampled=sampled)
    assert sampled == [], "the renewal began a write on the replaced database"
    assert _owner_row_at(retained) == before
    assert backend._runtime_owner_renewal_connection is None
    assert not aiosqlite_worker(connection).is_alive()


async def test_a_renewal_does_not_commit_once_its_pathname_is_replaced(owner, tmp_path):
    """Self-review of #4040: the pathname replaced while a renewal holds the
    writer slot, after the check before it. The renewal checks the file
    again before it commits, so its UPDATE is rolled back and the owner's
    heartbeat in the original file is unchanged."""
    backend, store = owner
    assert await _renew(backend, store) is True
    before = _owner_row(backend)
    replaced: list[Path] = []

    def replacing_timestamp():
        # Sampled once the renewal holds the writer slot.
        replaced.append(_replace_database_pathname(backend, tmp_path))
        return store.to_timestamp_param(store.now_utc())

    with pytest.raises(ConnectionError, match="inode changed"):
        await backend.retain_unretired_runtime_owner(
            agent_id=AGENT, owner_id=OWNER, timestamp=replacing_timestamp,
        )
    assert len(replaced) == 1
    assert _owner_row_at(replaced[0]) == before, "the renewal committed on the replaced file"
    assert backend._runtime_owner_renewal_connection is None
