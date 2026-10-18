"""Review66: original PG validator provenance and finite health COMMIT custody."""
import asyncio
from types import SimpleNamespace

import asyncpg
import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import (
    ExecutionCustody, ProcessRuntimeExecutionFence, bind_execution_custody,
    execution_commit_outcome, execution_denial_scopes,
)
from kestrel_sovereign.features.health.feature import HealthFeature
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.stop import DistributedInvocationRegistry, DistributedInvocationStore
from tests.integration.test_execution_authority_postgres import (
    GenerationFence, native_pg as native_pg,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


async def test_native_validator_wait_denies_only_original_stop_caller(native_pg):
    backend, control, schema = native_pg
    db = AsyncDatabase(backend)
    store = DistributedInvocationStore(db)
    await store.ensure_schema()
    registry = DistributedInvocationRegistry(store)
    entered = asyncio.Event()

    class OriginalGeneration(GenerationFence):
        async def lock_and_validate(self, connection):
            entered.set()
            await super().lock_and_validate(connection)

    transaction = control.transaction()
    await transaction.start()
    caller = None
    try:
        await control.execute(f'UPDATE "{schema}".authority SET generation=2 WHERE id=1')
        with bind_execution_custody(OriginalGeneration()) as original:
            caller = create_owned_task(registry.agent_has_unsettled_work("tenant-a"))
            await asyncio.wait_for(entered.wait(), 2)
            assert not caller.done(), "native generation check must wait on the actual authority row"
            await transaction.commit()
            transaction = None
            outcome = await await_owned_task(caller)
            assert execution_denial_scopes(outcome.error) == (original,), "native denial lost caller provenance"
            assert execution_commit_outcome(outcome.error) is None
        assert registry._terminal_control is None, "one caller retired the shared Stop owner"
        assert registry.owner_lifecycle_status == "healthy"
        assert await registry.agent_has_unsettled_work("tenant-b") is False
    finally:
        if transaction is not None:
            await transaction.rollback()
        if caller is not None:
            await await_owned_task(caller)
        await await_owned_task(create_owned_task(registry.close()))


@pytest.mark.parametrize("cancel_shutdown", [False, True])
async def test_health_disable_joins_original_native_commit_ack(native_pg, monkeypatch, cancel_shutdown):
    backend, control, schema = native_pg
    db = AsyncDatabase(backend)
    await db.execute("CREATE TABLE health_log(id TEXT, agent_id TEXT, status TEXT, checks_json TEXT, overall_healthy INTEGER, created_at TEXT)")
    runtime = ExecutionCustody(ProcessRuntimeExecutionFence("healthy-original-agent"))
    backend._execution_custody = runtime
    agent = SimpleNamespace(did="healthy-original-agent", _execution_custody=runtime)
    feature = HealthFeature(agent)
    feature._db, feature._agent_id = db, agent.did
    feature._background_task = None
    feature._running = False
    feature._health_persist_tasks = set()
    feature._health_persist_control = None
    feature._in_memory_history = []
    entered, release = asyncio.Event(), asyncio.Event()
    original_exit = asyncpg.transaction.Transaction.__aexit__
    commits = []

    async def held_ack(transaction, error_type, error, traceback):
        result = await original_exit(transaction, error_type, error, traceback)
        if error_type is None and not transaction._nested and asyncio.current_task().get_name().startswith("health-log-persist:"):
            commits.append(transaction)
            entered.set()
            await release.wait()
        return result

    monkeypatch.setattr(asyncpg.transaction.Transaction, "__aexit__", held_ack)
    checking = create_owned_task(feature.run_once())
    stopping = None
    try:
        await asyncio.wait_for(entered.wait(), 3)
        assert await control.fetchval(f'SELECT COUNT(*) FROM "{schema}".health_log') == 1
        stopping = create_owned_task(feature.shutdown())
        if cancel_shutdown:
            await asyncio.sleep(0)
            stopping.cancel()
        done, _ = await asyncio.wait({stopping}, timeout=.05)
        assert not done, "Health disable interrupted an already committed native acknowledgement"
        release.set()
        await await_owned_task(checking)
        outcome = await await_owned_task(stopping)
        assert execution_commit_outcome(outcome.error) is None
        assert feature._health_persist_control is None
        assert not feature._health_persist_tasks
        assert len(commits) == 1
        runtime.require_work()
        await db.execute("INSERT INTO effects VALUES (67, 'agent remains healthy after feature disable')")
    finally:
        release.set()
        await await_owned_task(checking)
        if stopping is not None:
            await await_owned_task(stopping)
        monkeypatch.undo()
        backend._execution_custody = None
