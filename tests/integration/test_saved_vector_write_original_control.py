"""Actual PG saved-vector updates in full and genuine partial schemas."""

import os

import asyncpg
import pytest

from kestrel_sovereign.execution_custody import (
    _explicit_control_chain,
    is_execution_control_error,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.db.postgres import PostgresBackend
from kestrel_sovereign.storage.embedding_column import ensure_embedding_vec_column
from kestrel_sovereign.storage.saved_items_store import SavedItemsStore
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_saved_items_original_control import (
    native_recall as native_recall,
)


@pytest.fixture
async def native_vector_store(native_recall, native_pg):
    backend, storage, memory, agent, feature, store = native_recall
    # Independent actual executor configured at construction with the SAME
    # tenant schema and original scope, plus the installed extension namespace.
    # The standard minimal fixture intentionally hides public/vector types.
    schema = native_pg[2]
    pool = await asyncpg.create_pool(
        os.environ["TEST_POSTGRES_URL"],
        min_size=0,
        max_size=2,
        server_settings={"search_path": f'"{schema}",public'},
    )
    vector_backend = PostgresBackend.from_pool(pool)
    vector_backend._execution_custody = backend._execution_custody
    db = AsyncDatabase(vector_backend)
    try:
        yield db, SavedItemsStore(db, agent.did), agent.did
    finally:
        vector_backend._execution_custody = None
        await db.close()
        await pool.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("door", ["combined", "vector-only", "profile-only"])
async def test_actual_saved_vector_partial_updates_keep_original_commit_control(
    native_vector_store, native_pg, monkeypatch, door
):
    db, store, did = native_vector_store
    assert await ensure_embedding_vec_column(db, "saved_items", 3)
    await db.execute(
        "ALTER TABLE saved_items ADD COLUMN IF NOT EXISTS embedding_profile_id TEXT"
    )
    if door == "vector-only":
        await db.execute("ALTER TABLE saved_items DROP COLUMN embedding_profile_id")
    elif door == "profile-only":
        await db.execute("ALTER TABLE saved_items DROP COLUMN embedding_vec")
    await db.execute(
        "INSERT INTO saved_items (id,agent_id,item_type,name,content,content_hash) VALUES (?,?,?,?,?,?)",
        ("local-vector", did, "excerpt", "local", "local", "hash"),
    )
    observed, faulted = [], []
    actual_execute = asyncpg.Connection.execute

    async def original_update(connection, sql, *args, **kwargs):
        result = await actual_execute(connection, sql, *args, **kwargs)
        if "UPDATE saved_items SET embedding" in sql:
            observed.append(sql)
        return result

    monkeypatch.setattr(asyncpg.Connection, "execute", original_update)
    original = OSError("actual native saved-vector COMMIT release ACK lost")
    actual_exit = asyncpg.pool.PoolAcquireContext.__aexit__

    async def lose_ack(checkout, typ, error, tb):
        result = await actual_exit(checkout, typ, error, tb)
        if observed and not faulted and typ is None:
            faulted.append(checkout)
            raise original
        return result

    monkeypatch.setattr(asyncpg.pool.PoolAcquireContext, "__aexit__", lose_ack)
    failure = None
    try:
        await store._write_embedding_vec(
            "local-vector", [1.0, 0.0, 0.0], "local-profile"
        )
    except BaseException as error:
        failure = error
    assert observed and faulted, "did not reach actual native update COMMIT/release"
    observer, schema = native_pg[1:]
    column = "embedding_profile_id" if door == "profile-only" else "embedding_vec"
    assert await observer.fetchval(
        f'SELECT {column} IS NOT NULL FROM "{schema}".saved_items WHERE id=$1',
        "local-vector",
    )
    assert failure is not None and original in tuple(
        _explicit_control_chain(failure)
    ), f"actual {door} committed control became apparent partial success {failure!r}"
    assert is_execution_control_error(failure)


@pytest.mark.asyncio
@pytest.mark.parametrize("door", ["column", "hnsw"])
async def test_actual_first_vector_column_commit_keeps_original_control(
    native_vector_store, native_pg, monkeypatch, door
):
    db, store, did = native_vector_store
    observed, faulted = [], []
    original = OSError("actual native first-vector-column COMMIT release ACK lost")
    actual_execute = asyncpg.Connection.execute

    async def original_ddl(connection, sql, *args, **kwargs):
        result = await actual_execute(connection, sql, *args, **kwargs)
        target = (
            "ALTER TABLE saved_items ADD COLUMN"
            if door == "column"
            else "CREATE INDEX IF NOT EXISTS idx_saved_items_embedding_vec_hnsw"
        )
        if target in sql:
            observed.append(sql)
        return result

    monkeypatch.setattr(asyncpg.Connection, "execute", original_ddl)
    actual_exit = asyncpg.pool.PoolAcquireContext.__aexit__

    async def lose_ack(checkout, typ, error, tb):
        result = await actual_exit(checkout, typ, error, tb)
        if observed and not faulted and typ is None:
            faulted.append(checkout)
            raise original
        return result

    monkeypatch.setattr(asyncpg.pool.PoolAcquireContext, "__aexit__", lose_ack)
    result = failure = None
    try:
        result = await ensure_embedding_vec_column(db, "saved_items", 3)
    except BaseException as error:
        failure = error
    assert observed and faulted
    assert (
        await native_pg[1].fetchval(
            "SELECT COUNT(*) FROM information_schema.columns WHERE table_schema=$1 AND table_name='saved_items' AND column_name='embedding_vec'",
            native_pg[2],
        )
        == 1
    )
    if door == "hnsw":
        assert (
            await native_pg[1].fetchval(
                "SELECT COUNT(*) FROM pg_indexes WHERE schemaname=$1 AND indexname='idx_saved_items_embedding_vec_hnsw'",
                native_pg[2],
            )
            == 1
        )
    assert failure is not None and original in tuple(
        _explicit_control_chain(failure)
    ), f"actual first-column COMMIT control became {result!r}, failure={failure!r}"
    assert is_execution_control_error(failure)
