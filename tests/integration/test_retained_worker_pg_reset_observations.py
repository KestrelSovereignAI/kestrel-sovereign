"""Actual committed PG reset failure plus worker C remains finite through retry.

Real COMMIT/reset and original Task cancellation, with declared cleanup adapters.
Not signed graph mutation, complete Agent readiness or provider acceptance.
"""
import asyncio

import asyncpg
import pytest

from kestrel_sovereign import _async_ownership as ownership
from kestrel_sovereign.execution_custody import (
    ExecutionCommitOutcomeError, ExecutionCustody, _explicit_control_chain,
    bind_execution_custody_snapshot, execution_cleanup_debts, execution_commit_outcome,
)
from kestrel_sovereign.multi_agent.config import LocalAgentConfig
from kestrel_sovereign.storage.db import postgres as postgres_module
from tests.integration.test_execution_authority_postgres import GenerationFence, native_pg as _native_pg
from tests.integration.test_retained_spawn_worker_terminal_policy import (
    UnexpectedControl, admission_for, onboarding_owner as _onboarding_owner,
    retained_worker as _retained_worker,
)
from tests.utils.original_failure import OriginalFailure

native_pg = _native_pg
onboarding_owner = _onboarding_owner
retained_worker = _retained_worker


@pytest.mark.asyncio
@pytest.mark.parametrize(("phase", "later"), [
    (phase, later) for phase in ("receipt", "roster", "runtime", "witness")
    for later in ("success", "retry-failure")
] + [("receipt", "witness-failure"), ("receipt", "slot-failure")])
async def test_retained_worker_keeps_actual_pg_reset_withdrawal_inside_core_wrapper(
    native_pg, onboarding_owner, retained_worker, monkeypatch, phase, later,
):
    backend, observer, schema = native_pg
    manager, child, db = onboarding_owner
    admission = admission_for(manager, child)
    if phase == "receipt":
        admission.spawn_receipt_graph = object()
    if phase == "roster":
        admission.spawn_startup_config = LocalAgentConfig(data_dir="finite-wrapper", port=8801)
    if phase == "witness" or later == "witness-failure":
        admission.spawn_authority_witness_did = child.did
    entered, release = asyncio.Event(), asyncio.Event()
    resets, observations, originals, calls = [], [], [], []
    physical = OriginalFailure.capture(OSError("actual original native reset acknowledgement failed"))
    later_error = OriginalFailure.capture(UnexpectedControl("actual later restrictive cleanup failed"))
    worker = None
    actual_reset = asyncpg.pool.PoolAcquireContext.__aexit__
    async def reset(checkout, *args):
        result = await actual_reset(checkout, *args)
        if checkout.pool is backend._pool and not resets:
            resets.append((checkout, asyncio.current_task()))
            entered.set()
            await release.wait()
            raise physical.error
        return result
    monkeypatch.setattr(asyncpg.pool.PoolAcquireContext, "__aexit__", reset)
    actual_owned = postgres_module.await_owned_task
    async def observe(task, *args, **kwargs):
        outcome = await actual_owned(task, *args, **kwargs)
        if (asyncio.current_task() is worker and resets and task is resets[0][1]
                and outcome.cancellation is not None):
            observations.append(OriginalFailure.capture(outcome.cancellation))
        return outcome
    monkeypatch.setattr(postgres_module, "await_owned_task", observe)
    async def selected():
        calls.append(phase)
        if len(calls) == 1:
            try:
                with bind_execution_custody_snapshot((ExecutionCustody(GenerationFence()),)):
                    async with backend.transaction():
                        await backend.execute("INSERT INTO effects VALUES (471, 'actual retained worker COMMIT')")
            except BaseException as error:
                originals.append(OriginalFailure.capture(error))
                assert isinstance(error, Exception), "probe did not reach the caught ordinary-class Core carrier"
                assert any(isinstance(item, ExecutionCommitOutcomeError) for item in _explicit_control_chain(error))
                raise
            pytest.fail("actual committed native reset did not report failure")
        await db.execute("INSERT INTO host_rollback_effects VALUES (472)")
        if later == "retry-failure":
            raise later_error.error
        if phase == "receipt":
            admission.spawn_receipt_graph = None
        if phase == "roster":
            admission.spawn_startup_config = admission.spawn_startup_config_path = None
        if phase == "witness":
            admission.spawn_authority_witness_did = admission.spawn_authority_witness_mandate = None
        return False
    async def receipt(*args):
        return await selected()
    async def roster(*args):
        return await selected()
    async def runtime(*args):
        return await selected()
    async def normal_runtime(*args):
        return False
    async def witness(*args):
        if phase == "witness":
            return await selected()
        raise later_error.error
    if phase == "receipt":
        monkeypatch.setattr(manager, "_downgrade_uncommitted_spawn_receipt", receipt)
    if phase == "roster":
        monkeypatch.setattr(manager, "_withdraw_uncommitted_spawn_startup_registration", roster)
    monkeypatch.setattr(manager, "_rollback_uncommitted_spawn_runtime",
                        runtime if phase == "runtime" else normal_runtime)
    if phase == "witness" or later == "witness-failure":
        monkeypatch.setattr(manager, "_withdraw_uncommitted_spawn_authority_witness", witness)
    actual_resolve = manager._resolve_spawn_slot_terminal
    def resolve(current):
        actual_resolve(current)
        if later == "slot-failure":
            raise later_error.error
    monkeypatch.setattr(manager, "_resolve_spawn_slot_terminal", resolve)
    worker = await retained_worker(admission)
    try:
        await asyncio.wait_for(entered.wait(), 5)
        worker.cancel("actual original worker withdrew during pool reset")
        await asyncio.sleep(0)
        release.set()
        result = await ownership.await_owned_task(worker)
        assert len(originals) == len(observations) == len(resets) == 1
        assert calls == [phase, phase], "same restrictive phase did not reach its existing retry exit"
        observations[0].assert_retained(result.error)
        if later != "success":
            later_error.assert_retained(result.error)
        assert execution_commit_outcome(result.error) == "committed"
        assert execution_commit_outcome(manager._quarantined_shutdown_control) == "committed"
        assert observations[0].error not in tuple(_explicit_control_chain(manager._quarantined_shutdown_control))
        assert tuple(_explicit_control_chain(originals[0].error)) == originals[0].chain
        assert tuple(execution_cleanup_debts(originals[0].error)) == originals[0].debts
        for resource, owner in originals[0].debts:
            assert any(resource is retained and owner is retained_owner
                       for retained, retained_owner in execution_cleanup_debts(result.error))
        assert any(getattr(resource, "context_manager", None) is resets[0][0] and task is resets[0][1]
                   for resource, task in execution_cleanup_debts(result.error))
        assert resets[0][1].done() and ownership.owned_task_error(resets[0][1]) is physical.error
        assert await observer.fetchval(f'SELECT count(*) FROM "{schema}".effects WHERE id=471') == 1
        assert manager._pending_spawns == int(later in {"retry-failure", "witness-failure"})
    finally:
        release.set()
        await ownership.await_owned_task(worker)
        for _checkout, task in resets:
            await ownership.await_owned_task(task)
