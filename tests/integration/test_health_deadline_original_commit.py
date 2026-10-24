"""The actual health-query COMMIT survives its own expired observer deadline."""

from tests.shared.native_shutdown_budget import shutdown_fixture_agent

import asyncio

import asyncpg
import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import (
    ExecutionCustody,
    _explicit_control_chain,
    execution_commit_outcome,
)
from kestrel_sovereign.features.health import checks
from kestrel_sovereign.features.health.feature import HealthFeature
from kestrel_sovereign.server import _agent_health_result
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.async_storage import AsyncStorage
from tests.integration.test_execution_authority_postgres import GenerationFence
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.unit.test_cleanup_task_admission import _partial_native_agent


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["feature", "fallback"])
async def test_real_health_query_deadline_retains_original_committed_cancellation(
    native_pg, tmp_path, monkeypatch, surface
):
    backend, observer, _schema = native_pg
    agent, original_db = await _partial_native_agent(tmp_path, monkeypatch)
    db = AsyncDatabase(backend)
    storage = AsyncStorage(backend=backend)
    storage.db = db
    agent.storage = agent._raw_storage = storage
    scope = backend._execution_custody = ExecutionCustody(GenerationFence())
    feature = HealthFeature(agent)
    await feature.initialize()
    entered, release = asyncio.Event(), asyncio.Event()
    actual_commit = asyncpg.transaction.Transaction.__aexit__
    acknowledgments = []

    async def hold_original_health_query_ack(transaction, error_type, error, traceback):
        result = await actual_commit(transaction, error_type, error, traceback)
        if error is None and not transaction._nested and not acknowledgments:
            acknowledgments.append(transaction)
            entered.set()
            await release.wait()
        return result

    monkeypatch.setattr(
        asyncpg.transaction.Transaction, "__aexit__", hold_original_health_query_ack
    )
    monkeypatch.setattr(checks, "DATABASE_HEALTH_CHECK_TIMEOUT_S", 1.0)
    operation = (
        feature.run_once() if surface == "feature" else _agent_health_result(agent)
    )
    owner = create_owned_task(operation, retain_originals=True)
    try:
        await asyncio.wait_for(entered.wait(), 3)
        async with asyncio.timeout(3):
            while not owner.cancelling():
                await asyncio.sleep(0.01)
        assert not owner.done(), "health deadline abandoned its original native ACK"
        release.set()
        outcome = await await_owned_task(owner)
        assert execution_commit_outcome(outcome.error) == "committed", (
            "health deadline converted original committed cancellation into a diagnostic"
        )
        receipts = [
            receipt
            for error in _explicit_control_chain(outcome.error)
            for receipt in vars(error).get("_kestrel_acknowledged_commits", ())
        ]
        assert receipts and all(
            receipt.acknowledged and receipt.retired for receipt in receipts
        )
        assert await observer.fetchval("SELECT 1") == 1
        scope.require_work()
        assert not agent._runtime_publication_ready.is_set()
    finally:
        release.set()
        await await_owned_task(owner)
        backend._execution_custody = None
        await feature.shutdown()
        await shutdown_fixture_agent(agent)
        await original_db.close()
