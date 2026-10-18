"""Lifecycle consumers join actual PostgreSQL retirement before native control."""
import asyncio

import pytest

from kestrel_sovereign._async_ownership import await_owned_task
from kestrel_sovereign.multi_agent.agent_manager import AgentManager
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.async_storage import AsyncStorage
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.unit.test_execution_custody_review29 import native_owner
from tests.unit.test_execution_custody_review32 import receipt
from tests.utils.original_failure import OriginalFailure
from kestrel_sovereign.execution_custody import execution_commit_outcome, is_execution_control_error

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


@pytest.mark.parametrize("surface", ["server", "unregistered", "reaper", "fleet"])
@pytest.mark.parametrize("kind", ["unknown", "cancel_unknown", "group"])
async def test_native_postgres_lifecycle_consumers_join_before_original_receipt(
    native_pg, monkeypatch, tmp_path, surface, kind,
):
    backend, observer, schema = native_pg
    agent, old_db = await native_owner(tmp_path)
    await old_db.close()
    db = AsyncDatabase(backend)
    storage = AsyncStorage(backend=backend)
    storage.db = db
    agent.storage = agent._raw_storage = storage
    release, closed = asyncio.Event(), asyncio.Event()
    failure = receipt(kind)
    original = OriginalFailure.capture(failure)
    original_outcome = execution_commit_outcome(failure)
    continuation = waiter = shutdown = None

    async def tail():
        await release.wait()
        # The real executor remains available to its retained original owner.
        await db.execute("INSERT INTO effects VALUES (34, 'retained native tail')")
        await storage.close()
        closed.set()

    async def failed_shutdown():
        raise failure

    monkeypatch.setattr(agent, "shutdown", failed_shutdown)
    try:
        continuation = asyncio.create_task(tail())
        agent._durable_shutdown_continuation = continuation
        if surface == "server":
            from kestrel_sovereign.server import _shutdown_single_agent
            waiter = asyncio.create_task(_shutdown_single_agent(agent))
        elif surface == "unregistered":
            waiter = asyncio.create_task(AgentManager._shutdown_unregistered_agent("native", agent))
        elif surface == "fleet":
            manager = AgentManager(base_data_dir=tmp_path)
            manager._agents["native"] = agent
            manager._agent_names[agent.agent_id] = "native"
            waiter = asyncio.create_task(manager.shutdown_all())
        else:
            shutdown = asyncio.create_task(agent.shutdown())
            waiter = agent.handoff_shutdown_to_reaper(shutdown)
        done, _ = await asyncio.wait({waiter}, timeout=0.1)
        assert not done, "native control escaped before PostgreSQL retirement"
        assert not closed.is_set()
        release.set()
        outcome = await await_owned_task(waiter)
        original.assert_retained(outcome.error)
        assert is_execution_control_error(outcome.error)
        assert execution_commit_outcome(outcome.error) == original_outcome
        assert closed.is_set()
        assert await observer.fetchval(f'SELECT COUNT(*) FROM "{schema}".effects WHERE id=34') == 1
    finally:
        release.set()
        for task in (shutdown, continuation, waiter):
            if task is not None:
                await await_owned_task(task)
        await storage.close()
