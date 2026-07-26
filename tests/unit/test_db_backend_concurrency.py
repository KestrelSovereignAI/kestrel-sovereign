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
from unittest.mock import AsyncMock, Mock, patch

import pytest


def test_postgres_from_pool_keeps_advisory_dsn_outside_operational_pool_state():
    """A wrapped pool needs an explicit, scheduler-only connection source."""
    from kestrel_sovereign.storage.db.postgres import PostgresBackend

    class _Pool:
        def get_max_size(self):
            return 2

    backend = PostgresBackend.from_pool(
        _Pool(),
        advisory_dsn="postgresql://scheduler-test/kestrel",
    )

    # ``_dsn`` remains absent for existing wrapped-pool consumers (including
    # SQLAlchemy factories); only advisory gates use the explicit DSN.
    assert backend._dsn is None
    assert backend._advisory_dsn == "postgresql://scheduler-test/kestrel"
    assert backend._advisory_max_pool_size == 2


@pytest.mark.asyncio
async def test_postgres_from_pool_derives_a_dedicated_advisory_pool_recipe():
    """Pool-only embeddings never borrow the operational pool for gates."""

    from kestrel_sovereign.storage.db import postgres as postgres_module
    from kestrel_sovereign.storage.db.postgres import PostgresBackend

    class _Pool:
        _connect_args = ("postgresql://pool-only/kestrel",)
        _connect_kwargs = {"server_settings": {"application_name": "host"}}
        _connect = staticmethod(lambda *_args, **_kwargs: None)
        _connection_class = object
        _record_class = object

        def get_max_size(self):
            return 2

        acquire = Mock(side_effect=AssertionError("shared pool must not be acquired"))

    shared_pool = _Pool()
    backend = PostgresBackend.from_pool(shared_pool)
    advisory_pool = object()
    with patch.object(
        postgres_module.asyncpg,
        "create_pool",
        AsyncMock(return_value=advisory_pool),
    ) as create_pool:
        assert await backend._ensure_advisory_pool() is advisory_pool

    create_pool.assert_awaited_once_with(
        "postgresql://pool-only/kestrel",
        min_size=0,
        max_size=2,
        server_settings={"application_name": "host"},
        connect=shared_pool._connect,
        connection_class=object,
        record_class=object,
    )
    shared_pool.acquire.assert_not_called()


@pytest.mark.asyncio
async def test_postgres_from_pool_derives_keyword_only_advisory_pool_recipe():
    """asyncpg keyword-only pool settings are a valid dedicated recipe."""

    from kestrel_sovereign.storage.db import postgres as postgres_module
    from kestrel_sovereign.storage.db.postgres import PostgresBackend

    class _Pool:
        # Real ``asyncpg.create_pool(host=..., database=..., user=...)``
        # stores the absent positional DSN as ``(None,)``.
        _connect_args = (None,)
        _connect_kwargs = {
            "host": "postgres.internal",
            "database": "kestrel",
            "user": "scheduler",
            "password": "pool-only-secret",
        }

        def get_max_size(self):
            return 2

        acquire = Mock(side_effect=AssertionError("shared pool must not be acquired"))

    shared_pool = _Pool()
    backend = PostgresBackend.from_pool(shared_pool)
    assert backend._advisory_recipe_available is True
    advisory_pool = object()
    with patch.object(
        postgres_module.asyncpg,
        "create_pool",
        AsyncMock(return_value=advisory_pool),
    ) as create_pool:
        assert await backend._ensure_advisory_pool() is advisory_pool

    create_pool.assert_awaited_once_with(
        None,
        min_size=0,
        max_size=2,
        host="postgres.internal",
        database="kestrel",
        user="scheduler",
        password="pool-only-secret",
    )
    shared_pool.acquire.assert_not_called()


@pytest.mark.asyncio
async def test_postgres_from_pool_rejects_incomplete_advisory_pool_recipe():
    """Factory/class metadata alone cannot authorize a new DB connection."""

    from kestrel_sovereign.storage.db import postgres as postgres_module
    from kestrel_sovereign.storage.db.postgres import PostgresBackend
    from kestrel_sovereign.storage.db.interface import ConnectionError

    class _Pool:
        _connect_args = ()
        _connect_kwargs = {}
        _connect = staticmethod(lambda *_args, **_kwargs: None)
        _connection_class = object
        _record_class = object

        def get_max_size(self):
            return 2

        acquire = Mock(side_effect=AssertionError("shared pool must not be acquired"))

    shared_pool = _Pool()
    backend = PostgresBackend.from_pool(shared_pool)
    assert backend._advisory_recipe_available is False
    with patch.object(postgres_module.asyncpg, "create_pool", AsyncMock()) as create_pool:
        with pytest.raises(ConnectionError, match="valid wrapped-pool recipe"):
            await backend._ensure_advisory_pool()

    create_pool.assert_not_awaited()
    shared_pool.acquire.assert_not_called()


def test_postgres_from_pool_explicit_advisory_dsn_ignores_pool_connect_kwargs():
    """An explicit scheduler DSN does not inherit operational credentials."""

    from kestrel_sovereign.storage.db.postgres import PostgresBackend

    class _Pool:
        _connect_args = ("postgresql://operational/kestrel",)
        _connect_kwargs = {"ssl": "operational-only", "password": "secret"}
        _connection_class = object
        _record_class = object

        def get_max_size(self):
            return 2

    backend = PostgresBackend.from_pool(
        _Pool(),
        advisory_dsn="postgresql://scheduler/kestrel",
        advisory_connect_kwargs={"server_settings": {"search_path": "scheduler"}},
    )

    assert backend._advisory_connect_args == ("postgresql://scheduler/kestrel",)
    assert backend._advisory_connect_kwargs == {
        "connection_class": object,
        "record_class": object,
        "server_settings": {"search_path": "scheduler"},
    }


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


def test_autocommit_write_retries_on_tuple_concurrently_updated():
    """Concurrent upserts of the same row (parallel agent-init writes) can
    surface a transient 'tuple concurrently updated' from Postgres even with
    ON CONFLICT DO UPDATE. The autocommit execute() path retries it instead
    of 500-ing. See kestrel #1805."""
    import contextvars
    from kestrel_sovereign.storage.db.postgres import (
        PostgresBackend,
        _is_concurrent_update_error,
    )

    assert _is_concurrent_update_error(Exception("tuple concurrently updated"))
    assert not _is_concurrent_update_error(Exception("some other error"))

    backend = PostgresBackend.__new__(PostgresBackend)
    backend._txn_conn_var = contextvars.ContextVar("pg_txn_conn", default=None)

    class _FlakyPool:
        def __init__(self, fail_times):
            self.fail_times = fail_times
            self.calls = 0

        async def execute(self, query, *params):
            self.calls += 1
            if self.calls <= self.fail_times:
                raise Exception("tuple concurrently updated")
            return "INSERT 0 1"

    pool = _FlakyPool(fail_times=2)
    backend._ensure_connected = lambda: pool
    backend._convert_query = lambda q: q
    backend._strip_tz = lambda p: p

    async def main():
        return await backend.execute(
            "INSERT OR REPLACE INTO agent_metadata (agent_id, key, value) VALUES (?, ?, ?)",
            ("did:x", "k", "v"),
        )

    rows = asyncio.run(main())
    assert rows == 1
    assert pool.calls == 3  # 2 transient failures + 1 success


def test_autocommit_write_gives_up_after_max_retries():
    """A persistent concurrent-update error eventually raises (bounded retry)."""
    import contextvars
    from kestrel_sovereign.storage.db.postgres import PostgresBackend
    from kestrel_sovereign.storage.db.interface import QueryError

    backend = PostgresBackend.__new__(PostgresBackend)
    backend._txn_conn_var = contextvars.ContextVar("pg_txn_conn", default=None)

    class _AlwaysFailPool:
        def __init__(self):
            self.calls = 0

        async def execute(self, query, *params):
            self.calls += 1
            raise Exception("tuple concurrently updated")

    pool = _AlwaysFailPool()
    backend._ensure_connected = lambda: pool
    backend._convert_query = lambda q: q
    backend._strip_tz = lambda p: p

    async def main():
        await backend.execute("UPDATE agent_metadata SET value = ?", ("v",))

    import pytest
    with pytest.raises(QueryError):
        asyncio.run(main())
    assert pool.calls == 5  # initial + 4 retries


@pytest.mark.asyncio
async def test_postgres_close_cleans_primary_after_advisory_failure_and_retries():
    """A failed advisory close cannot strand its handle or skip primary cleanup."""

    from kestrel_sovereign.storage.db.postgres import PostgresBackend

    class _Pool:
        def __init__(self, *, fail_once: bool = False) -> None:
            self.fail_once = fail_once
            self.close_calls = 0

        async def close(self) -> None:
            self.close_calls += 1
            if self.fail_once and self.close_calls == 1:
                raise RuntimeError("advisory close failed")

    advisory = _Pool(fail_once=True)
    primary = _Pool(fail_once=True)
    backend = PostgresBackend.__new__(PostgresBackend)
    backend._advisory_pool = advisory
    backend._pool = primary
    backend._owns_pool = True

    with pytest.raises(RuntimeError, match="advisory close failed"):
        await backend.close()

    assert advisory.close_calls == 1
    assert primary.close_calls == 1
    assert backend._advisory_pool is advisory
    assert backend._pool is primary

    await backend.close()
    assert advisory.close_calls == 2
    assert primary.close_calls == 2
    assert backend._advisory_pool is None
    assert backend._pool is None


@pytest.mark.asyncio
async def test_postgres_close_releases_external_primary_after_advisory_failure():
    """from_pool never closes its primary pool, even when advisory close fails."""

    from kestrel_sovereign.storage.db.postgres import PostgresBackend

    class _AdvisoryPool:
        async def close(self) -> None:
            raise RuntimeError("advisory close failed")

    class _ExternalPool:
        def __init__(self) -> None:
            self.close_calls = 0

        async def close(self) -> None:
            self.close_calls += 1

    advisory = _AdvisoryPool()
    primary = _ExternalPool()
    backend = PostgresBackend.__new__(PostgresBackend)
    backend._advisory_pool = advisory
    backend._pool = primary
    backend._owns_pool = False

    with pytest.raises(RuntimeError, match="advisory close failed"):
        await backend.close()

    assert backend._advisory_pool is advisory
    assert backend._pool is None
    assert primary.close_calls == 0


@pytest.mark.asyncio
async def test_postgres_close_drains_both_pools_through_repeated_cancellation():
    """Caller cancellation is delayed until owned advisory and primary closes finish."""

    from kestrel_sovereign.storage.db.postgres import PostgresBackend

    class _BlockingPool:
        def __init__(self) -> None:
            self.entered = asyncio.Event()
            self.release = asyncio.Event()
            self.close_calls = 0

        async def close(self) -> None:
            self.close_calls += 1
            self.entered.set()
            await self.release.wait()

    advisory = _BlockingPool()
    primary = _BlockingPool()
    backend = PostgresBackend.__new__(PostgresBackend)
    backend._advisory_pool = advisory
    backend._pool = primary
    backend._owns_pool = True

    closing = asyncio.create_task(backend.close())
    await asyncio.wait_for(advisory.entered.wait(), timeout=1.0)
    closing.cancel()
    await asyncio.sleep(0)
    closing.cancel()
    advisory.release.set()
    await asyncio.wait_for(primary.entered.wait(), timeout=1.0)
    primary.release.set()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(closing, timeout=1.0)

    assert advisory.close_calls == 1
    assert primary.close_calls == 1
    assert backend._advisory_pool is None
    assert backend._pool is None
