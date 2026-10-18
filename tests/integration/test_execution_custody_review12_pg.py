"""Actual native cold-init resident liveness, without providers or new pools."""

import asyncio

import pytest

from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError, bind_execution_custody, require_execution_work,
)
from kestrel_sovereign.signals import SignalDispatcher, SignalLogStore, SourceRegistry, OrderedLockManager
from tests.integration.test_execution_authority_postgres import native_pg as _native_pg, GenerationFence
from tests.unit.test_execution_custody_review10 import runtime_owner
from tests.unit.test_durable_signal_delivery import _Agent

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]
native_pg = _native_pg


async def test_standalone_pg_cold_init_keeps_original_runtime_not_occurrence(native_pg, monkeypatch):
    from kestrel_sovereign import server
    backend, _, _ = native_pg
    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.setenv("KESTREL_DATABASE_URL", "postgresql://owned-native-probe")
    agent = runtime_owner()
    agent.did = "did:test:actual-cold-native"
    template = _Agent(agent.did)
    agent._durable_persistence_gate = template._durable_persistence_gate
    agent._execution_custody = server._standalone_runtime_execution_custody("cold", agent.did, None)
    backend._execution_custody = agent._execution_custody
    store = SignalLogStore(backend)
    dispatcher = SignalDispatcher(agent=agent, registry=SourceRegistry(), lock_manager=OrderedLockManager(), store=store)
    try:
        with bind_execution_custody(GenerationFence()) as occurrence:
            await store.initialize()
            await dispatcher.initialize_durable_delivery()
            assert dispatcher._runtime_owner_heartbeat_timer is not None
        with pytest.raises(ExecutionAuthorityError):
            occurrence.require_work()
        agent._runtime_publication_ready.set()
        dispatcher._runtime_owner_heartbeat_timer.cancel()
        dispatcher._start_runtime_owner_heartbeat()
        task = dispatcher._runtime_owner_heartbeat_task
        assert task is not None
        await task
        assert dispatcher._runtime_owner_heartbeat_timer is not None
        row = await backend.fetch_one(
            "SELECT stopped_at, heartbeat_at FROM durable_signal_runtime_owners WHERE agent_id=? AND owner_id=?",
            (agent.did, dispatcher._durable_delivery_owner),
        )
        assert row is not None and row[0] is None
        require_execution_work(agent)
        agent._execution_custody.fence.retire()
        with pytest.raises(ExecutionAuthorityError, match="process runtime retired"):
            await backend.execute("INSERT INTO effects VALUES (1, 'denied')")
    finally:
        await dispatcher._stop_runtime_owner_heartbeat()
        tasks = tuple(agent._background_tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
