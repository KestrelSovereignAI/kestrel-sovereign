"""Postgres backend concurrency hardening (#1726).

The Postgres transaction connection is keyed to the OWNING asyncio task, so a
concurrent task's execute()/transaction() doesn't route onto another task's open
transaction — and a child task created inside a transaction does NOT inherit it.

(SQLite read-isolation against dirty reads is tracked separately as a follow-up;
the dedicated-read-connection approach interacted badly with WAL visibility under
the full suite, so it was deferred rather than shipped on the hot read path.)
"""
from __future__ import annotations

import asyncio


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
