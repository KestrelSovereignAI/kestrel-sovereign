"""Lifecycle consumers must own native retirement and original receipts."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from tests.utils.original_failure import OriginalFailure

from kestrel_sovereign._async_ownership import await_owned_task
from kestrel_sovereign.features.spawn.feature import SpawnFeature
from kestrel_sovereign.llm.lifecycle import (
    drain_unpublished_llm_owners, retire_unpublished_llm,
)
from kestrel_sovereign.multi_agent.agent_manager import AgentManager
from kestrel_sovereign.storage.async_database import AsyncDatabase
from tests.unit.test_execution_custody_review22 import stub_single_startup
from tests.unit.test_execution_custody_review29 import native_owner
from tests.unit.test_execution_custody_review30 import service_owner
from tests.unit.test_execution_custody_review32 import receipt


async def native_close_owner(monkeypatch):
    from kestrel_sovereign.llm import service as service_module
    service = service_owner()
    client = httpx.AsyncClient()
    entered, release = asyncio.Event(), asyncio.Event()
    original = client.aclose

    async def delayed_close():
        entered.set()
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                pass
        await original()

    monkeypatch.setattr(client, "aclose", delayed_close)
    service.providers = [{"adapter": client}]
    monkeypatch.setattr(service_module, "CLIENT_CLOSE_TIMEOUT", 0.01)
    return service, client, entered, release


@pytest.mark.asyncio
async def test_actual_host_drain_joins_unpublished_native_client(monkeypatch):
    service, client, entered, release = await native_close_owner(monkeypatch)
    owners = {}
    drain = None
    try:
        with pytest.raises(RuntimeError, match="incomplete"):
            await retire_unpublished_llm(owners, "unborn", service, ())
        assert entered.is_set() and not client.is_closed
        drain = asyncio.create_task(drain_unpublished_llm_owners(owners))
        done, _ = await asyncio.wait({drain}, timeout=0.4)
        assert not done, "host drain returned before actual native close joined"
        assert len(owners) == 1
        release.set()
        await drain
        assert client.is_closed and not owners
    finally:
        release.set()
        if drain is not None:
            await await_owned_task(drain)
        await service.wait_for_close_completion()


@pytest.mark.asyncio
async def test_host_drain_upgrades_an_already_running_startup_close(monkeypatch):
    service, client, entered, release = await native_close_owner(monkeypatch)
    owners = {}
    startup = drain = None
    try:
        startup = asyncio.create_task(retire_unpublished_llm(owners, "unborn", service, ()))
        await asyncio.wait_for(entered.wait(), 5)
        drain = asyncio.create_task(drain_unpublished_llm_owners(owners))
        done, _ = await asyncio.wait({drain}, timeout=0.4)
        assert not done, "host drain adopted the bounded startup join instead of physical ownership"
        assert not client.is_closed
        release.set()
        await drain
        await await_owned_task(startup)
        assert client.is_closed and not owners
    finally:
        release.set()
        for task in (startup, drain):
            if task is not None:
                await await_owned_task(task)
        await service.wait_for_close_completion()


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["server", "unregistered", "reaper", "fleet"])
@pytest.mark.parametrize("kind", ["unknown", "cancel_unknown", "group"])
async def test_actual_lifecycle_caller_joins_native_tail_before_control(
    monkeypatch, tmp_path, surface, kind,
):
    agent, db = await native_owner(tmp_path)
    release, closed = asyncio.Event(), asyncio.Event()
    failure = receipt(kind)
    source = OriginalFailure.capture(failure)

    async def native_tail():
        await release.wait()
        await agent.storage.close()
        closed.set()

    continuation = asyncio.create_task(native_tail())
    agent._durable_shutdown_continuation = continuation

    async def shutdown_receipt():
        raise failure

    monkeypatch.setattr(agent, "shutdown", shutdown_receipt)
    if surface == "server":
        from kestrel_sovereign.server import _shutdown_single_agent
        waiter = asyncio.create_task(_shutdown_single_agent(agent))
    elif surface == "unregistered":
        waiter = asyncio.create_task(AgentManager._shutdown_unregistered_agent("unborn", agent))
    elif surface == "fleet":
        manager = AgentManager(base_data_dir=tmp_path)
        manager._agents["native"] = agent
        manager._agent_names[agent.agent_id] = "native"
        waiter = asyncio.create_task(manager.shutdown_all())
    else:
        shutdown = asyncio.create_task(agent.shutdown())
        waiter = agent.handoff_shutdown_to_reaper(shutdown)
    try:
        done, _ = await asyncio.wait({waiter}, timeout=0.1)
        assert not done, "control receipt bypassed mandatory native lifecycle join"
        assert db.backend._connection is not None
        release.set()
        outcome = await await_owned_task(waiter)
        source.assert_retained(outcome.error)
        assert closed.is_set() and db.backend._connection is None
    finally:
        release.set()
        await await_owned_task(waiter)
        await await_owned_task(continuation)


@pytest.mark.asyncio
async def test_actual_failed_agent_birth_is_not_provider_only_cleanup(monkeypatch, tmp_path):
    server, app = stub_single_startup(monkeypatch, tmp_path, backend="sqlite")
    agent, db = await native_owner(tmp_path)
    service, client, entered, release_close = await native_close_owner(monkeypatch)
    agent.llm_service = service
    release_agent = asyncio.Event()
    failure = receipt("cancel_unknown")

    async def native_tail():
        await release_agent.wait()
        await service.close()
        await agent.storage.close()

    async def failed_initialize():
        agent._durable_shutdown_continuation = asyncio.create_task(native_tail())
        raise failure

    monkeypatch.setattr(agent, "initialize", failed_initialize)
    monkeypatch.setattr(server, "LLMService", lambda: service)
    monkeypatch.setattr(server, "KestrelAgent", lambda **kwargs: agent)
    waiter = None
    try:
        with pytest.raises(BaseException) as caught:
            async with server._lifespan_startup(app):
                pytest.fail("failed birth published")
        assert caught.value is failure
        assert not entered.is_set(), "provider closed beneath failed Agent's native tail"
        assert app.state.agent is None
        assert app.state.startup_cleanup_agent is agent
        waiter = asyncio.create_task(server._shutdown_server_agents(app))
        done, _ = await asyncio.wait({waiter}, timeout=0.1)
        assert not done
        release_close.set()
        release_agent.set()
        outcome = await await_owned_task(waiter)
        assert outcome.error is failure
        assert client.is_closed and db.backend._connection is None
        assert app.state.startup_cleanup_agent is agent
    finally:
        release_close.set()
        release_agent.set()
        if waiter is not None:
            await await_owned_task(waiter)
        if agent._durable_shutdown_continuation is not None:
            await await_owned_task(agent._durable_shutdown_continuation)
        await service.wait_for_close_completion()
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["unknown", "cancel_unknown", "group"])
async def test_actual_manager_quarantine_retains_native_terminal_receipt(tmp_path, kind):
    manager = AgentManager(base_data_dir=tmp_path)
    failure = receipt(kind)
    source = OriginalFailure.capture(failure)
    db = await AsyncDatabase.sqlite(str(tmp_path / "native.db"))

    async def retire_then_report():
        await db.close()
        raise failure

    task = asyncio.create_task(retire_then_report())
    manager._retain_quarantined_cleanup(name="native", agent_id="did:test:native", task=task)
    await await_owned_task(task)
    await asyncio.sleep(0)
    assert not manager._quarantined_shutdown_reapers
    with pytest.raises(BaseException) as caught:
        await manager.drain_quarantined_shutdowns()
    source.assert_retained(caught.value)


@pytest.mark.asyncio
async def test_actual_agent_adopts_failed_host_factory_native_owner(tmp_path, monkeypatch):
    from kestrel_sovereign.execution_custody import (
        ExecutionCustody, ProcessRuntimeExecutionFence, bind_execution_runtime,
        execution_commit_outcome,
    )
    from kestrel_sovereign.services.process_host_database import _ProcessHostBackend
    from kestrel_sovereign.storage.async_database import DatabaseInitializationCleanupError

    path = tmp_path / "agent_data"
    path.mkdir()

    async def no_schema(_db):
        return None

    seed = await AsyncDatabase.sqlite(str(path / "host.db"), schema_initializer=no_schema)
    await seed.close()
    agent, db = await native_owner(tmp_path)
    agent.storage_path = str(path / "agent.db")
    agent._execution_custody = ExecutionCustody(ProcessRuntimeExecutionFence(agent.did))
    commit_entered = asyncio.Event()
    original_connect, original_close = _ProcessHostBackend.connect, AsyncDatabase.close
    databases = []

    async def connected(backend):
        await original_connect(backend)
        original_commit = backend._connection.commit

        async def commit_then_lost_receipt():
            await original_commit()
            commit_entered.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(backend._connection, "commit", commit_then_lost_receipt)

    async def failed_factory_close(database):
        databases.append(database)
        raise OSError("native factory retirement failed")

    monkeypatch.setattr(_ProcessHostBackend, "connect", connected)
    monkeypatch.setattr(AsyncDatabase, "close", failed_factory_close)
    task = None
    try:
        with bind_execution_runtime(agent):
            task = asyncio.create_task(agent._resolve_host_db())
        await asyncio.wait_for(commit_entered.wait(), 5)
        task.cancel()
        outcome = await await_owned_task(task)
        failure = outcome.error
        assert isinstance(failure, DatabaseInitializationCleanupError)
        assert execution_commit_outcome(failure) == "unknown"
        assert len(databases) == 1
        host_db = databases[0]
        assert vars(agent).get("_resolved_local_host_db") is host_db, "Agent discarded factory's live retirement owner"
        assert "host_key_database" in agent._resource_custody().held
        monkeypatch.setattr(AsyncDatabase, "close", original_close)
        with pytest.raises(BaseException) as caught:
            await agent.shutdown()
        assert execution_commit_outcome(caught.value) == "unknown"
        with pytest.raises(BaseException) as caught:
            await agent.wait_for_shutdown_completion()
        assert execution_commit_outcome(caught.value) == "unknown"
        assert host_db.backend._connection is None
        assert db.backend._connection is None
    finally:
        if task is not None:
            await await_owned_task(task)
        monkeypatch.setattr(AsyncDatabase, "close", original_close)
        for database in databases:
            await original_close(database)
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["unknown", "cancel_unknown", "group", "authority"])
async def test_actual_delegate_result_repeatedly_preserves_child_control(monkeypatch, tmp_path, kind):
    parent, db = await native_owner(tmp_path)
    feature = SpawnFeature(parent)
    await feature.initialize()
    failure = receipt(kind)

    async def child_effect(_task):
        await db.execute("INSERT INTO effects VALUES (1)")
        raise failure

    child = SimpleNamespace(agent_id="did:test:child", process_input=child_effect)
    manager = SimpleNamespace(
        get_agent=lambda name: child,
        get_authoritative_spawn_relations=AsyncMock(return_value={"did:test:child": (parent.agent_id, "child")}),
    )
    monkeypatch.setattr(feature, "_get_ready_agent_manager", AsyncMock(return_value=manager))
    try:
        await feature.delegate_task("child", "native work")
        await await_owned_task(feature._child_tasks["child"])
        await asyncio.sleep(0)
        assert await db.fetchval("SELECT COUNT(*) FROM effects") == 1
        for _ in range(2):
            with pytest.raises(BaseException) as caught:
                await feature.get_child_result("child")
            assert caught.value is failure
        with pytest.raises(BaseException) as caught:
            await feature.shutdown()
        assert caught.value is failure
    finally:
        for task in tuple(feature._child_tasks.values()):
            await await_owned_task(task)
        await db.close()
