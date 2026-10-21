"""Shared native Stop custody must not inherit one caller's lifetime loss."""
import asyncio
import pytest
from kestrel_sovereign.execution_custody import (
    ExecutionCustody, ProcessRuntimeExecutionFence,
    bind_execution_custody_snapshot, execution_denial_scopes,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.stop import DistributedInvocationRegistry, DistributedInvocationStore
from tests.unit.test_distributed_stop_invocations import _SelfFencingReplicaAgent


@pytest.mark.asyncio
async def test_retired_caller_does_not_fence_healthy_native_stop_generation(tmp_path):
    db = await AsyncDatabase.sqlite(str(tmp_path / "original-stop.db"))
    store = DistributedInvocationStore(db)
    await store.ensure_schema()
    registry = DistributedInvocationRegistry(store)
    healthy = _SelfFencingReplicaAgent("did:test:healthy-tenant")
    registry.attach(healthy)
    turn = asyncio.create_task(healthy.run_turn("healthy-original-turn"))
    retired = ExecutionCustody(ProcessRuntimeExecutionFence("did:test:retired-tenant"))
    try:
        await asyncio.wait_for(healthy.operation_started.wait(), 2)
        with bind_execution_custody_snapshot((retired,)):
            retired.revoke("original tenant retired")
            with pytest.raises(BaseException) as denied:
                await registry.agent_has_unsettled_work("did:test:retired-tenant")
            assert execution_denial_scopes(denied.value) == (retired,)
        assert registry._terminal_control is None, "tenant-local denial latched onto shared registry"
        assert registry.owner_lifecycle_status == "healthy"
        assert not turn.done(), "unrelated healthy generation was cancelled"
        assert not healthy.is_request_self_fenced("healthy-original-turn")
        assert await db.fetchval("SELECT COUNT(*) FROM stop_active_invocations WHERE agent_id = ?", (healthy.agent_id,)) == 1
        healthy.release_operation.set()
        assert await turn == "healthy-original-turn"
    finally:
        healthy.release_operation.set()
        await asyncio.gather(turn, return_exceptions=True)
        await registry.close()
        await db.close()


@pytest.mark.asyncio
async def test_shared_native_stop_custody_loss_still_fences_original_owner(tmp_path):
    db = await AsyncDatabase.sqlite(str(tmp_path / "original-shared-stop.db"))
    store = DistributedInvocationStore(db)
    await store.ensure_schema()
    original = ExecutionCustody(ProcessRuntimeExecutionFence("original-shared-host"))
    with bind_execution_custody_snapshot((original,)):
        registry = DistributedInvocationRegistry(store)
    try:
        with bind_execution_custody_snapshot((original,)):
            original.revoke("original shared host retired")
            with pytest.raises(BaseException) as denied:
                await registry.agent_has_unsettled_work("did:test:tenant")
        assert execution_denial_scopes(denied.value) == (original,)
        assert registry._terminal_control is denied.value
        assert registry.owner_lifecycle_status == "self_fenced"
        with pytest.raises(BaseException) as repeated:
            await registry.agent_has_unsettled_work("did:test:another-tenant")
        assert repeated.value is denied.value
        with pytest.raises(BaseException) as closed:
            await registry.close()
        assert closed.value is denied.value
    finally:
        await db.close()
