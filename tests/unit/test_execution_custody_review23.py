"""Real boot rollback, process host store and first-party resident handoff."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from kestrel_sovereign.agent.boot import BootPhase, BootPhaseState
from kestrel_sovereign.agent.custody import ReleaseOutcome, ResourceCustody
from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError, ExecutionCommitOutcomeError, ExecutionCustody, ProcessRuntimeExecutionFence,
    bind_execution_custody, bind_execution_runtime, require_execution_work,
)
from kestrel_sovereign.kestrel_agent import KestrelAgent
from tests.unit.test_execution_custody import Authority
from tests.unit.test_execution_custody_review10 import runtime_owner


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["error", "cancel"])
@pytest.mark.parametrize("local", [True, False])
async def test_real_initialize_retires_only_local_runtime_before_first_rollback(failure, local):
    agent = runtime_owner()
    agent._execution_custody = ExecutionCustody(
        ProcessRuntimeExecutionFence("failed-boot") if local else Authority()
    )
    agent._custody = ResourceCustody()
    agent._boot_state = BootPhaseState.NOT_STARTED
    agent._host_authority_boot_expired = False
    agent._disarm_host_authority_boot_deadline = lambda: None
    agent._release_serving_record = lambda: True
    observed = []

    async def undo():
        if local:
            with pytest.raises(ExecutionAuthorityError, match="retired"):
                require_execution_work(agent)
        else:
            require_execution_work(agent)
        observed.append(True)
        await asyncio.sleep(0)
        return ReleaseOutcome.RELEASED

    async def acquire(ctx):
        ctx.on_rollback("probe", undo)

    error = asyncio.CancelledError("late phase") if failure == "cancel" else ValueError("late phase")

    async def fail(ctx):
        raise error

    agent._boot_phases = lambda: [BootPhase("acquire", acquire), BootPhase("fail", fail)]
    with pytest.raises(type(error)) as caught:
        await KestrelAgent.initialize(agent)
    assert caught.value is error
    assert observed == [True]
    assert agent._boot_state is BootPhaseState.FAILED


@pytest.mark.asyncio
async def test_real_host_store_retains_process_scope_without_allowing_ordinary_sqlite(tmp_path):
    from kestrel_sovereign.services.payer_resolver import open_host_db
    from kestrel_sovereign.storage.async_database import AsyncDatabase
    path = tmp_path / "agent_data"
    path.mkdir()
    seed = await AsyncDatabase.sqlite(str(path / "host.db"))
    await seed.close()
    owner = SimpleNamespace(_execution_custody=ExecutionCustody(ProcessRuntimeExecutionFence("host")))
    db = None
    try:
        with bind_execution_runtime(owner):
            db = await open_host_db(storage_path=path / "agent.db")
            assert await db.fetchone("SELECT COUNT(*) FROM host_service_keys") == (0,)
            with pytest.raises(Exception, match="prohibited|authorized"):
                await db.execute("INSERT INTO graph_nodes (id) VALUES ('not-host-metadata')")
            with pytest.raises(ExecutionAuthorityError, match="requires postgres"):
                await AsyncDatabase.sqlite(":memory:")
        from kestrel_sovereign.storage.sqla.session import make_session_factory
        with pytest.raises(ExecutionAuthorityError, match="unfenced SQLAlchemy"):
            make_session_factory(db)
        # No ambient scope now: the host backend must retain its original one.
        owner._execution_custody.fence.retire()
        with pytest.raises(ExecutionAuthorityError, match="retired"):
            await db.execute("INSERT INTO service_providers (id, name) VALUES ('blocked', 'blocked')")
    finally:
        if db is not None:
            await db.close()
    verify = await AsyncDatabase.sqlite(str(path / "host.db"))
    try:
        assert await verify.fetchone("SELECT COUNT(*) FROM service_providers WHERE id='blocked'") == (0,)
    finally:
        await verify.close()


@pytest.mark.asyncio
async def test_process_host_store_preserves_commit_receipt_and_canonical_close(tmp_path, monkeypatch):
    from kestrel_sovereign.storage.async_database import AsyncDatabase
    path = tmp_path / "agent_data"
    path.mkdir()
    seed = await AsyncDatabase.sqlite(str(path / "host.db"))
    await seed.close()
    owner = runtime_owner()
    owner.storage_path = str(path / "agent.db")
    owner._injected_host_db = None
    owner._execution_custody = ExecutionCustody(ProcessRuntimeExecutionFence("host"))
    owner._custody = ResourceCustody()
    with bind_execution_runtime(owner):
        db = await owner._resolve_host_db()
        assert await owner._resolve_host_db() is db
        original_commit = db.backend._connection.commit

        async def commit_then_retire():
            await original_commit()
            owner._execution_custody.fence.retire()

        monkeypatch.setattr(db.backend._connection, "commit", commit_then_retire)
        with pytest.raises(ExecutionCommitOutcomeError) as caught:
            await db.execute("INSERT INTO service_providers (id,name) VALUES ('receipt','receipt')")
        assert caught.value.commit_outcome == "committed"
    try:
        assert await owner._close_local_host_db() is ReleaseOutcome.RELEASED
        assert vars(owner)["_resolved_local_host_db"] is None
        verify = await AsyncDatabase.sqlite(str(path / "host.db"))
        try:
            assert await verify.fetchone("SELECT COUNT(*) FROM service_providers WHERE id='receipt'") == (1,)
        finally:
            await verify.close()
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_host_store_never_drops_foreign_generation_or_occurrence(tmp_path):
    from kestrel_sovereign.services.payer_resolver import open_host_db
    from kestrel_sovereign.storage.async_database import AsyncDatabase
    path = tmp_path / "agent_data"
    path.mkdir()
    seed = await AsyncDatabase.sqlite(str(path / "host.db"))
    await seed.close()
    with bind_execution_custody(Authority()):
        with pytest.raises(ExecutionAuthorityError, match="native PostgreSQL host store|requires postgres"):
            await open_host_db(storage_path=path / "agent.db")


@pytest.mark.asyncio
async def test_failed_host_store_close_keeps_owned_recovery_evidence():
    from kestrel_sovereign.storage.async_database import DatabaseRetirementError
    owner = runtime_owner()
    owner._custody = ResourceCustody()
    owner._custody.acquire("host_key_database")
    db = SimpleNamespace(close=AsyncMock(side_effect=ValueError("close unavailable")))
    owner._resolved_local_host_db = db
    with pytest.raises(DatabaseRetirementError):
        await owner._custody.release("host_key_database", owner._close_local_host_db)
    assert owner._resolved_local_host_db is db
    assert "host_key_database" in owner._custody.held


@pytest.mark.asyncio
@pytest.mark.parametrize("feature_name", ["delivery", "health"])
async def test_actual_feature_initialize_publishes_resident_not_occurrence(feature_name, monkeypatch):
    from kestrel_sovereign.features.delivery.feature import DeliveryFeature
    from kestrel_sovereign.features.health.feature import HealthFeature
    owner = runtime_owner()
    owner.did = "resident"
    db = SimpleNamespace(execute=AsyncMock(return_value=0), fetchone=AsyncMock(return_value=(0,)), fetchall=AsyncMock(return_value=[]))
    owner._raw_storage = owner.storage = SimpleNamespace(db=db)
    feature = DeliveryFeature(owner) if feature_name == "delivery" else HealthFeature(owner)
    observed, release = asyncio.Event(), asyncio.Event()
    seen = []

    async def loop():
        require_execution_work(owner)
        seen.append(True)
        observed.set()
        await release.wait()

    if feature_name == "delivery":
        from kestrel_sovereign.features.delivery.queue import DeliveryQueue
        monkeypatch.setattr(DeliveryQueue, "_loop", lambda self: loop())
        monkeypatch.setattr(DeliveryQueue, "_ensure_tables", AsyncMock())
        monkeypatch.setattr(DeliveryQueue, "_reclaim_in_flight", AsyncMock(return_value=0))
    else:
        monkeypatch.setattr(feature, "_background_loop", loop)
    try:
        with bind_execution_custody(Authority()) as occurrence:
            await feature.initialize()
            await asyncio.sleep(0)
            assert not seen, "resident escaped READY publication barrier"
            owner._runtime_publication_ready.set()
        with pytest.raises(ExecutionAuthorityError):
            occurrence.require_work()
        await asyncio.wait_for(observed.wait(), 1)
        assert seen == [True], "resident retained the now-finished boot occurrence"
        owner._execution_custody.revoke("runtime retired")
    finally:
        release.set()
        await feature.shutdown()
        await owner._shutdown_background_tasks()
