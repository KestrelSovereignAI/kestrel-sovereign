"""Core classification and shared shutdown retention keep actual native sources."""
import asyncio

import asyncpg
import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import (
    ExecutionCustody, _explicit_control_chain, bind_execution_custody_snapshot,
    execution_commit_outcome,
)
from kestrel_sovereign.kestrel_agent import require_feature_lifecycle_integrity
from kestrel_sovereign.storage.db import postgres as postgres_module
from tests.fixtures.sdk_contribution_fixture import SDKFixtureFeature
from tests.integration.test_execution_authority_postgres import GenerationFence, native_pg as native_pg
from tests.integration.test_execution_authority_postgres import native_pg as _cleanup_pg
from tests.integration.test_host_onboarding_original_admission import onboarding_owner as onboarding_owner
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.utils.original_failure import OriginalFailure

cleanup_pg = _cleanup_pg


@pytest.mark.asyncio
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_unregister_shared_retention_does_not_mutate_two_native_sources(
    native_pg, cleanup_pg, onboarding_owner, monkeypatch, source_cancel,
):
    _, agent, database = onboarding_owner
    first = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    second = OriginalFailure.capture(await _original_control(cleanup_pg[0], monkeypatch))
    class Feature(SDKFixtureFeature):
        contribution_prefix = "two-cleanup-sources-209"
        async def on_enable(self):
            raise OSError("foreign activation prose")
        async def on_disable(self):
            await database.execute("INSERT INTO host_rollback_effects VALUES (312)")
            raise first.error
        async def shutdown(self):
            await database.execute("INSERT INTO host_rollback_effects VALUES (313)")
            raise second.error
    owner = create_owned_task(agent._activate_feature_runtime(Feature(agent)), retain_originals=True)
    result = await await_owned_task(owner)
    for source in (first, second):
        source.assert_retained(result.error)
        source.assert_retained(agent._feature_lifecycle_control)
        source.assert_retained(agent._shutdown_control)
    assert await database.fetchval("SELECT count(*) FROM host_rollback_effects WHERE id IN (312,313)") == 2


@pytest.mark.asyncio
@pytest.mark.parametrize('withdraw', [False, True])
async def test_classified_reset_failure_keeps_caller_out_of_both_shared_latches(
    native_pg, onboarding_owner, monkeypatch, withdraw,
):
    backend, observer, schema = native_pg
    _, agent, _database = onboarding_owner
    physical = OriginalFailure.capture(OSError("actual cleanup reset ACK failed"))
    entered, release, resets, withdrawals = asyncio.Event(), asyncio.Event(), [], []
    actual_reset, actual_join = asyncpg.pool.PoolAcquireContext.__aexit__, postgres_module.await_owned_task
    async def reset(checkout, *args):
        result = await actual_reset(checkout, *args)
        if checkout.pool is backend._pool:
            resets.append((checkout, asyncio.current_task()))
            entered.set()
            await release.wait()
            raise physical.error
        return result
    async def join(*args, **kwargs):
        result = await actual_join(*args, **kwargs)
        if result.cancellation is not None:
            withdrawals.append(result.cancellation)
        return result
    monkeypatch.setattr(asyncpg.pool.PoolAcquireContext, "__aexit__", reset)
    monkeypatch.setattr(postgres_module, "await_owned_task", join)
    class Feature(SDKFixtureFeature):
        contribution_prefix = "classified-cleanup-209"
        async def on_enable(self):
            raise OSError("foreign activation prose")
        async def on_disable(self):
            with bind_execution_custody_snapshot((ExecutionCustody(GenerationFence()),)):
                await backend.execute("INSERT INTO effects VALUES (314, 'actual cleanup COMMIT')")
    owner = create_owned_task(agent._activate_feature_runtime(Feature(agent)), retain_originals=True)
    try:
        await asyncio.wait_for(entered.wait(), 5)
        if withdraw:
            owner.cancel("actual classified native cleanup observer withdrew")
            await asyncio.sleep(0)
        release.set()
        result = await await_owned_task(owner)
        physical.assert_retained(result.error)
        assert bool(withdrawals) is withdraw
        assert all(error in tuple(_explicit_control_chain(result.error)) for error in withdrawals)
        for shared in (agent._feature_lifecycle_control, agent._shutdown_control):
            physical.assert_retained(shared)
            assert execution_commit_outcome(shared) == "committed"
            assert all(error not in tuple(_explicit_control_chain(shared)) for error in withdrawals)
        with pytest.raises(BaseException) as later:
            require_feature_lifecycle_integrity(agent)
        assert all(error not in tuple(_explicit_control_chain(later.value)) for error in withdrawals)
        assert len(resets) == 1 and resets[0][1].done()
        assert await observer.fetchval(f'SELECT count(*) FROM "{schema}".effects WHERE id=314') == 1
    finally:
        release.set()
        await await_owned_task(owner)
