"""Original cold replay DDL/commit custody on actual isolated databases."""

import asyncio
from contextlib import AsyncExitStack

import pytest

from kestrel_sovereign.a2a.replay_store import SharedReplayNonceStore
from kestrel_sovereign.storage.db.interface import TransactionError


@pytest.mark.parametrize("native_a2a_backend_factory", ["postgres"], indirect=True)
@pytest.mark.parametrize("same_nonce", [False, True])
async def test_cold_replay_stores_share_original_schema_exclusion(
    native_a2a_backend_factory, same_nonce,
):
    async with AsyncExitStack() as resources:
        backends = [native_a2a_backend_factory() for _ in range(3)]
        for backend in backends:
            resources.push_async_callback(backend.close)
            await backend.connect()
        first, second, observer = backends
        stores = [SharedReplayNonceStore(first), SharedReplayNonceStore(second)]
        assert not await observer.table_exists("a2a_replay_nonces")
        created, second_ready, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        second_pid = None

        async def hold_first_schema():
            async with first.transaction():
                reserved = await stores[0].reserve(
                    "did:replay:test", "a" * 32, now_ts=100, ttl_seconds=60,
                )
                created.set()
                await release.wait()
                return reserved

        async def reserve_second():
            nonlocal second_pid
            await created.wait()
            async with second.transaction():
                second_pid = (await second.fetch_one("SELECT pg_backend_pid()"))[0]
                second_ready.set()
                return await stores[1].reserve(
                    "did:replay:test", ("a" if same_nonce else "b") * 32,
                    now_ts=100, ttl_seconds=60,
                )

        async with asyncio.TaskGroup() as owners:
            original = owners.create_task(hold_first_schema())
            contender = owners.create_task(reserve_second())
            try:
                async with asyncio.timeout(5):
                    await second_ready.wait()
                    while True:
                        state = await observer.fetch_one(
                            "SELECT state, wait_event_type, query FROM pg_stat_activity WHERE pid = ?",
                            (second_pid,),
                        )
                        if state is not None and state[:2] == ("active", "Lock"):
                            break
                        await asyncio.sleep(0.001)
                # No table warming, fake DB return, or process-local lock:
                # observe what the actual competing PostgreSQL session awaits.
                assert "pg_advisory_xact_lock" in state[2], (
                    ("cold replay DDL entered PostgreSQL before obtaining the "
                     "original shared native schema lease"), state,
                )
                assert not contender.done()
            finally:
                release.set()
        assert original.result() is True
        assert contender.result() is not same_nonce
        assert all(store._initialized for store in stores)
        assert await observer.fetch_one("SELECT COUNT(*) FROM a2a_replay_nonces") == (
            1 if same_nonce else 2,
        )


async def test_rollback_does_not_cache_uncommitted_replay_schema(
    native_a2a_backend_factory,
):
    backend = native_a2a_backend_factory()
    await backend.connect()
    original = ValueError("real caller rolls back provisional replay DDL")
    store = SharedReplayNonceStore(backend)
    try:
        with pytest.raises(TransactionError) as captured:
            async with backend.transaction():
                await store.ensure_table()
                assert not store._initialized, "provisional DDL was acknowledged before original COMMIT"
                raise original
        assert captured.value.__cause__ is original
        assert not store._initialized
        assert not await backend.table_exists("a2a_replay_nonces")
        await store.ensure_table()
        assert store._initialized
        assert await backend.table_exists("a2a_replay_nonces")
    finally:
        await backend.close()


async def test_original_transaction_can_reserve_again_while_same_store_contender_waits(
    native_a2a_backend_factory,
):
    async with AsyncExitStack() as resources:
        backend, observer = [native_a2a_backend_factory() for _ in range(2)]
        for connection in (backend, observer):
            resources.push_async_callback(connection.close)
            await connection.connect()
        store = SharedReplayNonceStore(backend)
        contender_started = asyncio.Event()
        contender_pid = None

        async def reserve_contender():
            nonlocal contender_pid
            if backend.backend_type == "postgres":
                # Only the pool's other connection is available: the original
                # owner still holds the first until its transaction commits.
                contender_pid = (await backend.fetch_one("SELECT pg_backend_pid()"))[0]
            contender_started.set()
            return await store.reserve(
                "did:replay:test", "b" * 32, now_ts=100, ttl_seconds=60,
            )

        async with asyncio.TaskGroup() as owners:
            async with backend.transaction():
                assert await store.reserve(
                    "did:replay:test", "a" * 32, now_ts=100, ttl_seconds=60,
                )
                assert not store._initialized
                contender = owners.create_task(reserve_contender())
                async with asyncio.timeout(5):
                    await contender_started.wait()
                    if backend.backend_type == "postgres":
                        while True:
                            state = await observer.fetch_one(
                                "SELECT state, wait_event_type, query FROM pg_stat_activity WHERE pid = ?",
                                (contender_pid,),
                            )
                            if state is not None and state[:2] == ("active", "Lock"):
                                break
                            await asyncio.sleep(0.001)
                        assert "pg_advisory_xact_lock" in state[2]
                assert not contender.done()
                # A competing Python lock must not keep this same original
                # DB-transaction owner from making progress before COMMIT.
                async with asyncio.timeout(5):
                    assert await store.reserve(
                        "did:replay:test", "c" * 32, now_ts=100, ttl_seconds=60,
                    )
                assert not store._initialized
            assert store._initialized
        assert contender.result() is True
        assert await observer.fetch_one("SELECT COUNT(*) FROM a2a_replay_nonces") == (3,)


async def test_failed_original_commit_never_acknowledges_replay_initialization(
    native_a2a_backend_factory,
):
    backend = native_a2a_backend_factory()
    await backend.connect()
    store = SharedReplayNonceStore(backend)
    try:
        await backend.execute_script(
            "CREATE TABLE replay_commit_parent (id INTEGER PRIMARY KEY);"
            "CREATE TABLE replay_commit_child (parent_id INTEGER REFERENCES "
            "replay_commit_parent(id) DEFERRABLE INITIALLY DEFERRED)"
        )
        with pytest.raises(TransactionError):
            async with backend.transaction():
                await store.ensure_table()
                await backend.execute("INSERT INTO replay_commit_child VALUES (1)")
                assert not store._initialized
        assert not store._initialized
        assert not await backend.table_exists("a2a_replay_nonces")
        assert await backend.fetch_one("SELECT COUNT(*) FROM replay_commit_child") == (0,)
    finally:
        await backend.close()


async def test_failed_replay_index_creation_preserves_original_schema(
    native_a2a_backend_factory,
):
    backend = native_a2a_backend_factory()
    await backend.connect()
    store = SharedReplayNonceStore(backend)
    try:
        await backend.execute(
            "CREATE TABLE a2a_replay_nonces (sender TEXT, nonce TEXT, seen_at DOUBLE PRECISION)"
        )
        await backend.execute("INSERT INTO a2a_replay_nonces VALUES ('original', 'retained', 1)")
        with pytest.raises(TransactionError):
            await store.ensure_table()
        assert not store._initialized
        assert await backend.fetch_all("SELECT * FROM a2a_replay_nonces") == [
            ("original", "retained", 1),
        ]
    finally:
        await backend.close()
