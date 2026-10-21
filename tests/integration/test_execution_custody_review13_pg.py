"""Native idle dispatcher retirement through the real Kestrel shutdown tail."""

import asyncio
from types import SimpleNamespace

import pytest

from kestrel_sovereign.execution_custody import ExecutionAuthorityError
from kestrel_sovereign.signals import (
    OrderedLockManager, SignalDispatcher, SignalLogStore, SourceRegistry,
)
from tests.integration.test_execution_authority_postgres import native_pg as _native_pg
from tests.unit.test_agent_boot_phases import _make_agent
from tests.unit.test_durable_signal_delivery import _Agent
from tests.integration.test_execution_authority_postgres import GenerationFence
from kestrel_sovereign.execution_custody import ExecutionCustody
from kestrel_sovereign.signals.durable import DurableSignalStore
from kestrel_sovereign.signals.sources.channels import DURABLE_COGNITION_CONSUMER_ID

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]
native_pg = _native_pg


async def test_idle_standalone_native_shutdown_releases_original_owner_and_storage(native_pg, monkeypatch, tmp_path):
    from kestrel_sovereign import server

    backend, controller, schema = native_pg
    store = SignalLogStore(backend)
    await store.initialize()
    agent = _make_agent(tmp_path)
    service = agent.llm_service
    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.setenv("KESTREL_DATABASE_URL", "postgresql://owned-native-probe")
    scope = server._standalone_runtime_execution_custody("idle", agent.did, None)
    agent._execution_custody = scope
    backend._execution_custody = scope
    agent._durable_persistence_gate = _Agent(agent.did)._durable_persistence_gate
    dispatcher = SignalDispatcher(
        agent=agent, registry=SourceRegistry(), lock_manager=OrderedLockManager(), store=store,
    )
    agent.dispatcher = dispatcher
    agent.storage = SimpleNamespace(close=backend.close)
    agent.features = {}
    agent.llm_service = None
    agent.task_manager = None
    agent.memory_system = None
    agent._sync_service = None
    try:
        await dispatcher.initialize_durable_delivery()
        await agent.shutdown()
        continuation = agent._durable_shutdown_continuation
        if continuation is not None:
            await continuation
        assert backend._pool is None, "retired runtime retained its native storage"
        stopped = await controller.fetchval(
            f'SELECT stopped_at IS NOT NULL FROM "{schema}".durable_signal_runtime_owners '
            "WHERE agent_id=$1 AND owner_id=$2",
            agent.did, dispatcher._durable_delivery_owner,
        )
        assert stopped is True
        with pytest.raises(ExecutionAuthorityError, match="retired"):
            scope.require_work()
    finally:
        await dispatcher._stop_runtime_owner_heartbeat()
        continuation = agent._durable_shutdown_continuation
        if continuation is not None:
            if not continuation.done():
                continuation.cancel()
            await asyncio.gather(continuation, return_exceptions=True)
        tasks = tuple(agent._background_tasks)
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await service.close()


async def _seed_owned_deliveries(backend, controller, schema):
    await DurableSignalStore(backend).initialize()
    await controller.execute(f'SET search_path TO "{schema}"')
    await controller.execute(
        "INSERT INTO durable_signal_runtime_owners (agent_id,owner_id,heartbeat_at,stopped_at) "
        "VALUES ('original','dispatcher:original',NOW()-INTERVAL '1 hour',NULL), "
        "('original','dispatcher:successor',NOW()-INTERVAL '1 hour',NULL), "
        "('other','dispatcher:other',NOW()-INTERVAL '1 hour',NULL)"
    )
    await controller.execute(
        "INSERT INTO durable_signal_consumers (agent_id,consumer_id,source,max_attempts,lease_seconds,active) "
        "VALUES ('original',$1,'channel.message',0,10,TRUE), ('other',$1,'channel.message',0,10,TRUE)",
        DURABLE_COGNITION_CONSUMER_ID,
    )
    for index, (agent_id, owner, status) in enumerate([
        ("original", "dispatcher:original", "initial_reserved"),
        ("original", "dispatcher:original", "leased"),
        ("original", "dispatcher:successor", "initial_reserved"),
        ("other", "dispatcher:other", "initial_reserved"),
    ]):
        await controller.execute(
            "INSERT INTO durable_signal_events "
            "(event_id,agent_id,target_agent,source,kind,mode,visibility,payload,urgency,"
            "causation_chain,arrived_at,retention_until,source_sequence) "
            "VALUES ($1,$2,$2,'channel.message','message','cognition','internal','{}',"
            "'normal','[]',NOW(),NOW()+INTERVAL '1 day',$3)", f"event{index}", agent_id, index+1,
        )
        await controller.execute(
            "INSERT INTO durable_signal_deliveries "
            "(delivery_id,event_id,agent_id,consumer_id,status,max_attempts,lease_owner,lease_token) "
            "VALUES ($1,$2,$3,$4,$5,0,$6,$7)",
            f"delivery{index}", f"event{index}", agent_id, DURABLE_COGNITION_CONSUMER_ID,
            status, owner, f"token{index}",
        )


@pytest.mark.parametrize("stop", [False, True])
async def test_retired_native_owner_cleanup_is_exact_and_does_not_admit_work(native_pg, stop):
    backend, controller, schema = native_pg
    await _seed_owned_deliveries(backend, controller, schema)
    store = DurableSignalStore(backend)
    scope = ExecutionCustody(GenerationFence())
    backend._execution_custody = scope
    scope.revoke("original generation retired")
    with pytest.raises(ExecutionAuthorityError):
        await backend.execute("INSERT INTO effects VALUES (1,'denied')")
    assert await store.release_initial_reservations(agent_id="original", owner_id="dispatcher:missing") == 0
    assert await store.release_initial_reservations(agent_id="original", owner_id="dispatcher:original", mark_owner_stopped=stop) == 1
    rows = await controller.fetch("SELECT delivery_id,status,lease_owner FROM durable_signal_deliveries ORDER BY delivery_id")
    assert [row["status"] for row in rows] == ["retry", "leased", "initial_reserved", "initial_reserved"]
    assert rows[1]["lease_owner"] == "dispatcher:original"
    assert rows[2]["lease_owner"] == "dispatcher:successor"
    assert await controller.fetchval("SELECT stopped_at IS NOT NULL FROM durable_signal_runtime_owners WHERE owner_id='dispatcher:original'") is stop
    assert await controller.fetchval("SELECT count(*) FROM durable_signal_runtime_owners") == 3
    assert await controller.fetchval("SELECT heartbeat_at < NOW()-INTERVAL '30 minutes' FROM durable_signal_runtime_owners WHERE owner_id='dispatcher:successor'")
    with pytest.raises(ExecutionAuthorityError):
        await backend.fetch_val("SELECT 1")
    assert await controller.fetchval("SELECT count(*) FROM effects") == 0


@pytest.mark.parametrize("index", [0, 1])
async def test_retired_initial_handoff_compensation_keeps_exact_token_cas(native_pg, index):
    backend, controller, schema = native_pg
    await _seed_owned_deliveries(backend, controller, schema)
    store = DurableSignalStore(backend)
    backend._execution_custody = ExecutionCustody(GenerationFence())
    backend._execution_custody.revoke("original generation retired")
    arguments = dict(agent_id="original", consumer_id=DURABLE_COGNITION_CONSUMER_ID,
                     delivery_id=f"delivery{index}", owner_id="dispatcher:original", reservation_token=f"token{index}")
    assert not await store.abandon_initial_reservation(**(arguments | {"reservation_token": "replacement-token"}))
    assert not await store.abandon_initial_reservation(**(arguments | {"delivery_id": "delivery2"}))
    assert await store.abandon_initial_reservation(**arguments)
    assert not await store.abandon_initial_reservation(**arguments)
    assert await controller.fetchval("SELECT status FROM durable_signal_deliveries WHERE delivery_id=$1", f"delivery{index}") == "retry"
    assert await controller.fetchval("SELECT status FROM durable_signal_deliveries WHERE delivery_id='delivery2'") == "initial_reserved"
    with pytest.raises(ExecutionAuthorityError):
        await backend.execute("INSERT INTO effects VALUES (1,'denied')")


async def test_retained_native_shutdown_heartbeat_and_rearm_use_cleanup_only(native_pg):
    backend, controller, schema = native_pg
    await _seed_owned_deliveries(backend, controller, schema)
    agent = _Agent("original")
    scope = ExecutionCustody(GenerationFence())
    backend._execution_custody = scope
    agent._execution_custody = scope
    dispatcher = SignalDispatcher(agent=agent, registry=SourceRegistry(), lock_manager=OrderedLockManager(), store=SignalLogStore(backend))
    dispatcher._durable_delivery_owner = "dispatcher:original"
    await dispatcher.initialize_durable_delivery()
    dispatcher._durable_shutdown = True
    dispatcher._durable_shutdown_owner_fenced = True
    scope.revoke("original runtime retired")
    # A denied resident context must not be mistaken for new work admission.
    class DeniedResident(type(agent)):
        def _runtime_owner_context(self):
            scope.require_work()
    agent.__class__ = DeniedResident
    try:
        await dispatcher._heartbeat_runtime_owner()
        dispatcher._schedule_runtime_owner_heartbeat()
        assert dispatcher._runtime_owner_heartbeat_timer is not None
        assert await controller.fetchval("SELECT heartbeat_at > NOW()-INTERVAL '1 minute' FROM durable_signal_runtime_owners WHERE owner_id='dispatcher:original'")
        assert await controller.fetchval("SELECT heartbeat_at < NOW()-INTERVAL '30 minutes' FROM durable_signal_runtime_owners WHERE owner_id='dispatcher:successor'")
        assert await controller.fetchval("SELECT status FROM durable_signal_deliveries WHERE delivery_id='delivery1'") == "leased"
        with pytest.raises(ExecutionAuthorityError):
            await backend.execute("INSERT INTO effects VALUES (1,'denied')")
    finally:
        await dispatcher._stop_runtime_owner_heartbeat()


async def test_actual_native_retained_shutdown_stops_owner_only_after_join(native_pg):
    backend, controller, schema = native_pg
    await _seed_owned_deliveries(backend, controller, schema)
    agent = _Agent("original")
    scope = ExecutionCustody(GenerationFence())
    backend._execution_custody = scope
    agent._execution_custody = scope
    dispatcher = SignalDispatcher(agent=agent, registry=SourceRegistry(), lock_manager=OrderedLockManager(), store=SignalLogStore(backend))
    dispatcher._durable_delivery_owner = "dispatcher:original"
    await dispatcher.initialize_durable_delivery()
    dispatcher._durable_shutdown = True
    started, release, joined = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def original_route():
        started.set()
        try:
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    continue
        finally:
            joined.set()

    task = asyncio.create_task(original_route(), name="owned-native-retained-probe")
    dispatcher._retained_durable_cognition_tasks.add(task)
    task.add_done_callback(dispatcher._retained_durable_cognition_tasks.discard)
    try:
        await started.wait()
        scope.revoke("original runtime retired")
        await dispatcher._activate_retained_durable_shutdown_fence()
        assert dispatcher.durable_shutdown_owner_fenced
        assert not task.done()
        assert await controller.fetchval("SELECT stopped_at IS NULL FROM durable_signal_runtime_owners WHERE owner_id='dispatcher:original'")
        assert await controller.fetchval("SELECT status FROM durable_signal_deliveries WHERE delivery_id='delivery0'") == "retry"
        with pytest.raises(ExecutionAuthorityError):
            await backend.execute("INSERT INTO effects VALUES (1,'denied')")
        release.set()
        await task
        await dispatcher._fenced_durable_shutdown_completion
        assert joined.is_set()
        assert not dispatcher.durable_shutdown_owner_fenced
        assert await controller.fetchval("SELECT stopped_at IS NOT NULL FROM durable_signal_runtime_owners WHERE owner_id='dispatcher:original'")
        assert await controller.fetchval("SELECT stopped_at IS NULL FROM durable_signal_runtime_owners WHERE owner_id='dispatcher:successor'")
    finally:
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        completion = dispatcher._fenced_durable_shutdown_completion
        if completion is not None:
            await asyncio.gather(completion, return_exceptions=True)
        await dispatcher._stop_runtime_owner_heartbeat()
