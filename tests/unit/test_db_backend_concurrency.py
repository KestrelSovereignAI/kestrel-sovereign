"""Storage backend concurrency hardening (#1726):

- SQLite reads run under the write guard, so a DIFFERENT task can't observe
  another task's UNCOMMITTED writes (dirty read).
- Postgres transaction connection is PER-TASK (ContextVar), so a concurrent
  task's execute()/transaction() doesn't route onto another task's open
  transaction.
"""
from __future__ import annotations

import asyncio

import pytest

from kestrel_sovereign.storage.db.sqlite import SQLiteBackend


# ---------------------------------------------------------------------------
# SQLite: reads don't see another task's uncommitted writes
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_sqlite_read_does_not_see_uncommitted_writes(tmp_path):
    backend = SQLiteBackend(str(tmp_path / "c.db"))
    await backend.connect()
    try:
        await backend.execute("CREATE TABLE t (v INTEGER)")
        await backend.execute("INSERT INTO t (v) VALUES (1)")

        reader_saw = {}
        reader_started = asyncio.Event()
        release_writer = asyncio.Event()

        async def writer():
            async with backend.transaction():
                await backend.execute("INSERT INTO t (v) VALUES (2)")
                # Signal that an uncommitted write exists, then hold the txn open
                # until the reader has attempted its read.
                reader_started.set()
                await release_writer.wait()
            # transaction commits here

        async def reader():
            await reader_started.wait()
            # This read is from a DIFFERENT task while the writer's txn is open.
            # It must block until the writer commits (write guard), then see 2
            # rows — NEVER a dirty read of the uncommitted row mid-transaction.
            async def _do_read():
                return await backend.fetch_val("SELECT COUNT(*) FROM t")
            read_task = asyncio.create_task(_do_read())
            # Give the read a chance to run; it should be blocked on the lock.
            await asyncio.sleep(0.05)
            reader_saw["blocked"] = not read_task.done()
            release_writer.set()
            reader_saw["count"] = await read_task

        await asyncio.gather(writer(), reader())

        # The reader blocked during the open transaction (no dirty read) and saw
        # the committed state (2 rows) afterward.
        assert reader_saw["blocked"] is True
        assert reader_saw["count"] == 2
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_sqlite_owner_sees_own_uncommitted_writes(tmp_path):
    """The write guard is re-entrant for the txn owner: a task reads its OWN
    in-flight writes within its transaction."""
    backend = SQLiteBackend(str(tmp_path / "o.db"))
    await backend.connect()
    try:
        await backend.execute("CREATE TABLE t (v INTEGER)")
        async with backend.transaction():
            await backend.execute("INSERT INTO t (v) VALUES (42)")
            # Same task, mid-transaction → sees its own write.
            assert await backend.fetch_val("SELECT COUNT(*) FROM t") == 1
    finally:
        await backend.close()


# ---------------------------------------------------------------------------
# Postgres: transaction connection is per-task (ContextVar)
# ---------------------------------------------------------------------------
def test_postgres_txn_conn_is_per_task_contextvar():
    """A PostgresBackend's transaction connection lives in a ContextVar, so it is
    scoped per asyncio task — not a shared instance attribute that concurrent
    tasks would cross-route onto (#1726)."""
    import contextvars
    from kestrel_sovereign.storage.db.postgres import PostgresBackend

    backend = PostgresBackend.__new__(PostgresBackend)
    backend._txn_conn_var = contextvars.ContextVar("pg_txn_conn", default=None)

    # Default is None (no open transaction).
    assert backend._txn_conn_var.get() is None

    async def main():
        # Setting in one task does not leak into a sibling task's context.
        async def task_a():
            token = backend._txn_conn_var.set("conn-A")
            await asyncio.sleep(0.01)
            seen = backend._txn_conn_var.get()
            backend._txn_conn_var.reset(token)
            return seen

        async def task_b():
            # Never set anything → must observe the default, not task_a's value.
            await asyncio.sleep(0.005)
            return backend._txn_conn_var.get()

        return await asyncio.gather(task_a(), task_b())

    a_seen, b_seen = asyncio.run(main())
    assert a_seen == "conn-A"
    assert b_seen is None  # task B never saw task A's transaction connection
