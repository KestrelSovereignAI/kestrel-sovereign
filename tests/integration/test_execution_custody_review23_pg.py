"""Native production startup rollback and resident boot-scope retirement."""

import asyncio
from types import SimpleNamespace

import pytest

from kestrel_sovereign.agent.boot import BootPhase
from kestrel_sovereign.agent.custody import ReleaseOutcome
from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError, ExecutionCustody, ProcessRuntimeExecutionFence,
    bind_execution_custody, current_execution_custody,
)
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.async_storage import AsyncStorage
from tests.integration.test_execution_authority_postgres import native_pg as _native_pg
from tests.unit.test_execution_custody_review22 import stub_single_startup

native_pg = _native_pg
pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


@pytest.mark.parametrize("failure", ["error", "cancel"])
async def test_actual_server_initialize_denies_native_child_during_boot_rollback(native_pg, monkeypatch, tmp_path, failure):
    from kestrel_sovereign.llm.service import LLMService
    backend, control, schema = native_pg
    await control.execute(f'SET search_path TO "{schema}"')
    server, app = stub_single_startup(monkeypatch, tmp_path)
    monkeypatch.setattr(LLMService, "_load_from_disk_cache", lambda self: False)
    undo_entered, release_child, finish_undo = asyncio.Event(), asyncio.Event(), asyncio.Event()
    children, candidates = [], []

    async def acquire(agent, ctx):
        native = agent._build_shared_pool_postgres_backend()
        agent._raw_storage = agent.storage = AsyncStorage(backend=native, agent_id=agent.did)
        agent._raw_storage.db = AsyncDatabase(native)

        async def child():
            await release_child.wait()
            await native.execute("INSERT INTO effects VALUES (94, 'boot rollback escape')")

        children.append(asyncio.create_task(child()))

        async def undo():
            undo_entered.set()
            release_child.set()
            await finish_undo.wait()
            return ReleaseOutcome.RELEASED

        ctx.on_rollback("pause-before-native-close", undo)

    async def fail(ctx):
        if failure == "cancel":
            raise asyncio.CancelledError("late boot")
        raise ValueError("late boot")

    def construct(**kwargs):
        agent = KestrelAgent(pg_pool=backend.operational_pool, **kwargs)
        agent._boot_phases = lambda: [BootPhase("native", lambda ctx: acquire(agent, ctx)), BootPhase("failure", fail)]
        candidates.append(agent)
        return agent

    monkeypatch.setattr(server, "KestrelAgent", construct)

    async def startup():
        async with server._lifespan_startup(app):
            assert app.state.agent is None

    task = asyncio.create_task(startup())
    try:
        # This rendezvous includes cold provider discovery/Agent construction,
        # not the native rollback's safety deadline. Allow ordinary parallel
        # gates to finish setup; all authority/order assertions below remain.
        await asyncio.wait_for(undo_entered.wait(), 10)
        result = await asyncio.gather(*children, return_exceptions=True)
        assert isinstance(result[0], ExecutionAuthorityError), "native write escaped during real initializer rollback"
        assert await control.fetchval("SELECT COUNT(*) FROM effects") == 0
        assert not task.done(), "probe must observe rollback before server exception handling"
    finally:
        release_child.set()
        finish_undo.set()
        await asyncio.gather(task, *children, return_exceptions=True)
        for agent in candidates:
            await agent.llm_service.close()
            await agent._raw_storage.close()


async def test_actual_pg_storage_phase_ignores_unused_existing_host_db(native_pg, monkeypatch, tmp_path):
    from kestrel_sovereign.llm.service import LLMService
    backend, _, _ = native_pg
    server, app = stub_single_startup(monkeypatch, tmp_path)
    monkeypatch.setattr(LLMService, "_load_from_disk_cache", lambda self: False)
    path = tmp_path / "agent_data"
    path.mkdir()
    host = await AsyncDatabase.sqlite(str(path / "host.db"))
    await host.close()
    candidates = []

    def construct(**kwargs):
        kwargs["storage_path"] = str(path / "agent.db")
        agent = KestrelAgent(pg_pool=backend.operational_pool, **kwargs)
        agent._boot_phases = lambda: [BootPhase("storage_privacy", agent._boot_phase_storage_privacy)]
        candidates.append(agent)
        return agent

    monkeypatch.setattr(server, "KestrelAgent", construct)
    try:
        async with server._lifespan_startup(app):
            assert app.state.agent is candidates[0], "unneeded host.db blocked real PG storage/provider-finalization phase"
            assert vars(candidates[0]).get("_resolved_local_host_db") is None
    finally:
        for agent in candidates:
            await agent.llm_service.close()
            if agent._raw_storage is not None:
                await agent._raw_storage.close()


@pytest.mark.parametrize("feature_name", ["delivery", "health"])
async def test_native_feature_worker_survives_only_boot_occurrence(native_pg, monkeypatch, feature_name):
    from kestrel_sovereign.features.delivery.feature import DeliveryFeature
    from kestrel_sovereign.features.health.feature import HealthFeature
    from tests.unit.test_execution_custody_review10 import runtime_owner
    backend, _, _ = native_pg
    owner = runtime_owner()
    owner.did = "did:test:resident"
    owner._execution_custody = ExecutionCustody(ProcessRuntimeExecutionFence(owner.did))
    # Native boundary uses the same retained generation as the actual task
    # handoff. This independent runtime cannot acquire a fresh occurrence.
    backend._execution_custody = owner._execution_custody
    db = AsyncDatabase(backend)
    owner._raw_storage = owner.storage = SimpleNamespace(db=db)
    feature = DeliveryFeature(owner) if feature_name == "delivery" else HealthFeature(owner)
    observed, proceed = asyncio.Event(), asyncio.Event()
    failures = []

    async def tick():
        await proceed.wait()
        try:
            assert current_execution_custody() == (owner._execution_custody,)
            await db.fetchone("SELECT COUNT(*) FROM effects")
        except BaseException as error:
            failures.append(error)
        finally:
            observed.set()
        return 0 if feature_name == "delivery" else {"status": "healthy"}

    try:
        with bind_execution_custody(ProcessRuntimeExecutionFence("cold-start-occurrence")):
            await feature.initialize()
            if feature_name == "delivery":
                feature._queue.process_pending = tick
            else:
                feature._interval_seconds = 0.001
                feature._run_health = tick
            owner._runtime_publication_ready.set()
        proceed.set()
        await asyncio.wait_for(observed.wait(), 3)
        assert not failures, "actual native worker retained completed cold-start occurrence"
        owner._execution_custody.revoke("runtime retired")
        with pytest.raises(ExecutionAuthorityError):
            await db.fetchone("SELECT COUNT(*) FROM effects")
    finally:
        proceed.set()
        await feature.shutdown()
        await owner._shutdown_background_tasks()
