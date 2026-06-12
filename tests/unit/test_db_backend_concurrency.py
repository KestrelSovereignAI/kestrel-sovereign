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
    """A non-owner read during another task's open transaction sees the COMMITTED
    snapshot (no dirty read) and does NOT block (separate read connection)."""
    backend = SQLiteBackend(str(tmp_path / "c.db"))
    await backend.connect()
    try:
        await backend.execute("CREATE TABLE t (v INTEGER)")
        await backend.execute("INSERT INTO t (v) VALUES (1)")  # committed: 1 row

        reader_saw = {}
        reader_started = asyncio.Event()
        release_writer = asyncio.Event()

        async def writer():
            async with backend.transaction():
                await backend.execute("INSERT INTO t (v) VALUES (2)")  # UNcommitted
                reader_started.set()
                await release_writer.wait()
            # commits here

        async def reader():
            await reader_started.wait()
            # Read from a DIFFERENT task while the writer's txn is open: must see
            # the committed snapshot (1), NOT the uncommitted 2nd row, and return
            # promptly (no blocking on the write lock).
            reader_saw["during"] = await asyncio.wait_for(
                backend.fetch_val("SELECT COUNT(*) FROM t"), timeout=2.0
            )
            release_writer.set()

        await asyncio.gather(writer(), reader())

        assert reader_saw["during"] == 1  # no dirty read of the uncommitted row
        assert await backend.fetch_val("SELECT COUNT(*) FROM t") == 2  # committed after
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
def test_postgres_txn_conn_is_per_task_and_not_inherited_by_children():
    """A PostgresBackend's transaction connection is keyed to the OWNING task
    (#1726). A SIBLING task never sees it, and — critically — a CHILD task
    created inside the transaction does NOT inherit it (ContextVars are copied
    into child tasks; the owner-task check prevents cross-routing onto the
    parent's connection)."""
    import contextvars
    from kestrel_sovereign.storage.db.postgres import PostgresBackend

    backend = PostgresBackend.__new__(PostgresBackend)
    backend._txn_conn_var = contextvars.ContextVar("pg_txn_conn", default=None)

    assert backend._current_txn_conn() is None  # no open transaction

    async def main():
        results = {}

        async def owner():
            token = backend._txn_conn_var.set((asyncio.current_task(), "conn-OWNER"))
            try:
                results["owner_sees"] = backend._current_txn_conn()
                # A CHILD task spawned inside the "transaction" inherits the
                # ContextVar but must NOT treat the parent's conn as its own.
                child = asyncio.create_task(_child())
                results["child_sees"] = await child
            finally:
                backend._txn_conn_var.reset(token)

        async def _child():
            return backend._current_txn_conn()

        async def sibling():
            await asyncio.sleep(0.005)
            return backend._current_txn_conn()

        sib = asyncio.create_task(sibling())
        await owner()
        results["sibling_sees"] = await sib
        return results

    r = asyncio.run(main())
    assert r["owner_sees"] == "conn-OWNER"   # owner uses its connection
    assert r["child_sees"] is None           # child does NOT inherit it
    assert r["sibling_sees"] is None         # sibling never saw it
