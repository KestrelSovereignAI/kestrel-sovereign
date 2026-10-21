"""Later native evidence cannot lose an earlier actual finite withdrawal.

Real typed PG denial/COMMIT sources and SQLite effects; declared callbacks are
not whole-Agent, authority/signature, financial or provider acceptance.
"""
import asyncio

import pytest

from kestrel_sovereign import _async_ownership as ownership
from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError, _explicit_control_chain,
    bind_execution_custody, execution_denial_scopes, execution_finite_error,
    execution_native_source_error,
)
from kestrel_sovereign.kestrel_agent import await_lifecycle_task_completion
from kestrel_sovereign.multi_agent.agent_manager import _merge_manager_cleanup_control
from tests.integration.test_execution_authority_postgres import GenerationFence, native_pg as _native_pg
from tests.integration.test_execution_custody_review18_pg import setup_delivery, join_owned
from tests.integration.test_host_onboarding_original_admission import onboarding_owner as _onboarding_owner
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.utils.original_failure import OriginalFailure

native_pg = _native_pg
onboarding_owner = _onboarding_owner


@pytest.mark.asyncio
@pytest.mark.parametrize("writer", ["budget", "reaper"])
@pytest.mark.parametrize("finite", [False, True])
async def test_forwarded_manager_summary_stays_unscoped_after_real_native_denials(
    native_pg, onboarding_owner, monkeypatch, writer, finite,
):
    manager, agent, database = onboarding_owner
    originals, withdrawals = [], []
    for index in range(2):
        with bind_execution_custody(GenerationFence()) as scope:
            scope.revoke("actual typed native work admission denied")
            with pytest.raises(ExecutionAuthorityError) as caught:
                await native_pg[0].execute("INSERT INTO effects VALUES (?, ?)", (501 + index, "must not enter"))
        assert execution_denial_scopes(caught.value) == (scope,)
        originals.append(OriginalFailure.capture(caught.value))
    summary = _merge_manager_cleanup_control(_merge_manager_cleanup_control(None, originals[0].error), originals[1].error)
    assert execution_denial_scopes(summary) is None
    if finite:
        async def caller():
            asyncio.current_task().cancel("actual forwarded-summary observer withdrew")
            try:
                await asyncio.sleep(0)
            except asyncio.CancelledError as error:
                withdrawals.append(error)
                return execution_finite_error(summary, error)
        summary_input = (await await_owned_task(create_owned_task(caller(), retain_originals=True))).result
    else:
        summary_input = summary
    async def operation():
        await database.execute("INSERT INTO host_rollback_effects VALUES (73)")
        raise summary_input
    if writer == "budget":
        task = manager._start_child_budget_release("forwarded-source", release_operation=operation).task
    else:
        task = create_owned_task(operation(), retain_originals=True)
        manager._retain_quarantined_cleanup(name="forwarded-source", agent_id=agent.did, task=task)
    result = await await_owned_task(task)
    await asyncio.sleep(0)
    assert result.error is not None
    retained = manager._quarantined_shutdown_control
    assert execution_denial_scopes(retained) is None, "forwarding relocalized a compacted summary to its selected tenant"
    assert execution_denial_scopes(execution_native_source_error(retained)) is None
    originals[0].assert_retained(retained)
    for source in originals:
        source.assert_retained(source.error)
    assert execution_denial_scopes(summary) is None
    assert all(error not in tuple(_explicit_control_chain(retained)) for error in withdrawals)
    assert await database.fetchval("SELECT count(*) FROM host_rollback_effects") == 1
    assert await native_pg[1].fetchval('SELECT count(*) FROM "' + native_pg[2] + '".effects WHERE id IN (501,502)') == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["writer-later", "writer-latch", "manager-batch", "shutdown-drains"])
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_ordinary_first_join_keeps_current_c_until_later_native_control(
    native_pg, onboarding_owner, monkeypatch, surface, source_cancel,
):
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    ordinary = OriginalFailure.capture(OSError("earlier original ordinary writer"))
    manager, _, database = onboarding_owner
    agent, dispatcher, _, _ = await setup_delivery(native_pg)
    entered, release, withdrawals = asyncio.Event(), asyncio.Event(), []
    actual_join, owner = ownership.await_owned_task, None
    async def observe(task, *args, **kwargs):
        outcome = await actual_join(task, *args, **kwargs)
        if asyncio.current_task() is owner and outcome.cancellation is not None:
            withdrawals.append(outcome.cancellation)
        return outcome
    async def first():
        await database.execute("INSERT INTO host_rollback_effects VALUES (74)")
        entered.set()
        await release.wait()
        raise ordinary.error
    async def later(*, finite_errors=None):
        await database.execute("INSERT INTO host_rollback_effects VALUES (75)")
        raise source.error
    monkeypatch.setattr(ownership, "await_owned_task", observe)
    tasks = [create_owned_task(first(), retain_originals=True)]
    if surface not in ("writer-latch", "shutdown-drains"):
        tasks.append(create_owned_task(later(), retain_originals=True))
    elif surface == "writer-latch":
        dispatcher._retain_durable_writer_control(source.error)
    if surface == "shutdown-drains":
        dispatcher._outcome_log_tasks.add(tasks[0])
        tasks[0].add_done_callback(dispatcher._outcome_log_tasks.discard)
        monkeypatch.setattr(dispatcher, "_drain_post_commit_reservation_repairs", later)
        operation = dispatcher._complete_durable_shutdown()
    else:
        operation = manager._settle_owned_lifecycle_tasks(tasks, description="actual native owner batch") if surface == "manager-batch" else dispatcher._join_outcome_log_tasks(tasks)
    owner = create_owned_task(operation, retain_originals=True)
    try:
        await asyncio.wait_for(entered.wait(), 5)
        owner.cancel("actual earlier ordinary writer observer withdrew")
        await asyncio.sleep(0)
        release.set()
        result = await await_owned_task(owner)
        source.assert_retained(result.error)
        ordinary.assert_retained(result.error)
        assert withdrawals and all(error in tuple(_explicit_control_chain(result.error)) for error in withdrawals)
        assert all(task.done() for task in tasks)
        assert await database.fetchval("SELECT count(*) FROM host_rollback_effects") == (1 if surface == "writer-latch" else 2)
    finally:
        release.set()
        await await_owned_task(owner)
        for task in tasks:
            await await_owned_task(task)
        await join_owned(agent, dispatcher)


@pytest.mark.asyncio
@pytest.mark.parametrize("source_cancel", [False, True])
@pytest.mark.parametrize("repeat", [False, True])
async def test_lifecycle_join_returns_actual_native_failure_and_every_current_c(
    native_pg, onboarding_owner, monkeypatch, source_cancel, repeat,
):
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    _, _, database = onboarding_owner
    entered, release, withdrawals = asyncio.Event(), asyncio.Event(), []
    async def native():
        await database.execute("INSERT INTO host_rollback_effects VALUES (76)")
        entered.set()
        await release.wait()
        raise source.error
    original = create_owned_task(native(), retain_originals=True)
    owner = create_owned_task(await_lifecycle_task_completion(original, withdrawals=withdrawals), retain_originals=True)
    try:
        await asyncio.wait_for(entered.wait(), 5)
        owner.cancel("actual lifecycle join withdrew")
        await asyncio.sleep(0)
        if repeat:
            owner.cancel("actual repeated lifecycle join withdrew")
            await asyncio.sleep(0)
        release.set()
        result = await await_owned_task(owner)
        assert result.error is None and result.result[0] is True
        source.assert_retained(result.result[1])
        assert withdrawals and all(error in tuple(_explicit_control_chain(result.result[1])) for error in withdrawals)
        assert original.done() and await database.fetchval("SELECT count(*) FROM host_rollback_effects") == 1
    finally:
        release.set()
        await await_owned_task(owner)
        await await_owned_task(original)
