"""Exact session death and transaction-bound generation fences (#3569).

Uses only TEST_POSTGRES_URL and owns a unique schema; no shared rows are reset.
"""

import asyncio
import os
import uuid

import asyncpg
import pytest

from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError,
    bind_execution_custody,
)
from kestrel_sovereign.storage.db.postgres import PostgresBackend


pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


@pytest.fixture
async def native_pg():
    dsn = os.environ.get("TEST_POSTGRES_URL")
    if not dsn:
        pytest.skip("TEST_POSTGRES_URL required")
    schema = "custody_" + uuid.uuid4().hex
    control = await asyncpg.connect(dsn)
    await control.execute(f'CREATE SCHEMA "{schema}"')
    pool = await asyncpg.create_pool(
        dsn, min_size=0, max_size=2,
        server_settings={"search_path": schema},
    )
    backend = PostgresBackend.from_pool(pool, advisory_max_pool_size=1)
    try:
        await backend.execute("CREATE TABLE authority (id INTEGER PRIMARY KEY, generation INTEGER)")
        await backend.execute("INSERT INTO authority VALUES (1, 1)")
        await backend.execute("CREATE TABLE effects (id INTEGER PRIMARY KEY, value TEXT)")
        yield backend, control, schema
    finally:
        await backend.close()
        await pool.close()
        await control.execute(f'DROP SCHEMA "{schema}" CASCADE')
        await control.close()


class GenerationFence:
    backend_type = "postgres"

    def __init__(self, generation=1, lease=None):
        self.generation = generation
        self.lease = lease

    def require_work(self):
        if self.lease is not None:
            self.lease.require_live()

    async def lock_and_validate(self, connection):
        row = await connection.fetchrow(
            "SELECT generation FROM authority WHERE id=1 FOR SHARE"
        )
        if row is None or row[0] != self.generation:
            raise ExecutionAuthorityError("runtime generation changed")


async def test_exact_advisory_session_loss_irrevocably_denies_other_connections(native_pg):
    backend, control, _ = native_pg
    lost = asyncio.Event()
    observed = []

    def on_loss(lease):
        observed.append(lease)
        lost.set()

    with pytest.raises(Exception, match="authority|advisory|session"):
        async with backend.advisory_locks([(3569, 1)], on_loss=on_loss) as lease:
            lease.require_live()
            with bind_execution_custody(GenerationFence(lease=lease)):
                assert await control.fetchval("SELECT pg_terminate_backend($1)", lease.backend_pid)
                await asyncio.wait_for(lost.wait(), timeout=5)
                with pytest.raises(Exception, match="authority|advisory|session"):
                    await backend.execute("INSERT INTO effects VALUES (1, 'stale')")
                lease.require_live()
    assert len(observed) == 1
    assert await backend.fetch_val("SELECT count(*) FROM effects") == 0
    async with backend.advisory_locks([(3569, 1)]) as replacement:
        replacement.require_live()
        with pytest.raises(Exception, match="authority|advisory|session"):
            lease.require_live()


async def test_normal_retirement_invalidates_handle_without_loss_notification(native_pg):
    backend, _, _ = native_pg
    observed = []
    async with backend.advisory_locks([(3569, 2)], on_loss=observed.append) as lease:
        lease.require_live()
    with pytest.raises(Exception, match="retired|released|authority|advisory"):
        lease.require_live()
    assert observed == []
    async with backend.advisory_locks([]) as empty:
        assert empty is None


@pytest.mark.parametrize("method", ["execute", "execute_many", "fetch_one", "fetch_all", "fetch_val", "execute_script", "transaction"])
async def test_stale_generation_rejects_every_native_sql_surface(native_pg, method):
    backend, _, _ = native_pg
    fence = GenerationFence()
    await backend.execute("UPDATE authority SET generation=2 WHERE id=1")
    with bind_execution_custody(fence):
        with pytest.raises(Exception, match="runtime generation changed"):
            if method == "execute_many":
                await backend.execute_many("INSERT INTO effects VALUES (?, ?)", [(1, "stale")])
            elif method == "execute_script":
                await backend.execute_script("INSERT INTO effects VALUES (1, 'stale');")
            elif method == "transaction":
                async with backend.transaction():
                    await backend.execute("INSERT INTO effects VALUES (1, 'stale')")
            else:
                await getattr(backend, method)("INSERT INTO effects VALUES (1, 'stale') RETURNING id")
    assert await backend.fetch_val("SELECT count(*) FROM effects") == 0


async def test_guarded_mutation_serializes_before_generation_transfer(native_pg):
    backend, control, schema = native_pg
    guard_held = asyncio.Event()
    release = asyncio.Event()

    async def mutate():
        with bind_execution_custody(GenerationFence()):
            async with backend.transaction():
                await backend.execute("INSERT INTO effects VALUES (1, 'authorized')")
                guard_held.set()
                await release.wait()

    mutation = asyncio.create_task(mutate())
    await asyncio.wait_for(guard_held.wait(), timeout=5)
    transfer = asyncio.create_task(control.execute(
        f'UPDATE "{schema}".authority SET generation=2 WHERE id=1'
    ))
    try:
        # PostgreSQL reports the exact blocked lock, not a timing assumption.
        async with asyncio.timeout(5):
            while not await backend.fetch_val(
                "SELECT EXISTS (SELECT 1 FROM pg_stat_activity WHERE pid=? AND wait_event_type='Lock')",
                (control.get_server_pid(),),
            ):
                await asyncio.sleep(0.01)
        assert not transfer.done()
        release.set()
        await mutation
        await transfer
    finally:
        release.set()
        await asyncio.gather(mutation, transfer, return_exceptions=True)
    assert await backend.fetch_val("SELECT generation FROM authority") == 2
    assert await backend.fetch_val("SELECT value FROM effects") == "authorized"


async def test_revocation_inside_transaction_rolls_back_and_sql_cannot_escape(native_pg):
    backend, _, _ = native_pg
    with bind_execution_custody(GenerationFence()) as scope:
        with pytest.raises(Exception, match="revoked while awaiting"):
            async with backend.transaction():
                await backend.execute("INSERT INTO effects VALUES (1, 'stale')")
                scope.revoke("revoked while awaiting")
    assert await backend.fetch_val("SELECT count(*) FROM effects") == 0
    with bind_execution_custody(GenerationFence()):
        with pytest.raises(Exception, match="transaction|Transaction"):
            await backend.execute_script("COMMIT; INSERT INTO effects VALUES (1, 'escape');")
    assert await backend.fetch_val("SELECT count(*) FROM effects") == 0


async def test_alternate_sqlite_path_cannot_silently_drop_postgres_fence(tmp_path):
    from kestrel_sovereign.storage.db.sqlite import SQLiteBackend

    backend = SQLiteBackend(str(tmp_path / "custody.db"))
    await backend.connect()
    try:
        with bind_execution_custody(GenerationFence()):
            with pytest.raises(ExecutionAuthorityError, match="backend"):
                await backend.execute("CREATE TABLE forbidden (id INTEGER)")
    finally:
        await backend.close()
