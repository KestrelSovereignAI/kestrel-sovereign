"""Actual manager lock ownership and accepted coordinator publication.

Native PG source faults, actual locks and SQLite effects; declared reconciliation
callback is not authority/signature, financial, whole-Agent or provider acceptance.
"""
import asyncio

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task, owned_task_error
from kestrel_sovereign.execution_custody import execution_cleanup_debts
from tests.integration.test_execution_authority_postgres import native_pg as _native_pg
from tests.integration.test_host_onboarding_original_admission import onboarding_owner as _onboarding_owner
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.integration.test_native_completion_vs_observation import observation_factory
from tests.utils.native_admission_fault import NativeAdmissionFault
from tests.utils.original_failure import OriginalFailure

native_pg = _native_pg
onboarding_owner = _onboarding_owner


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["seal", "drain", "reconcile", "identity", "deadline"])
@pytest.mark.parametrize("mode", ["refuse", "publish-before", "publish-after"])
@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_manager_coordinator_keeps_exact_accepted_task_and_balances_locks(
    native_pg, onboarding_owner, monkeypatch, surface, mode, eager, source_cancel,
):
    manager, agent, database = onboarding_owner
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    entries = []
    async def reconcile():
        entries.append(asyncio.current_task())
        await database.execute("INSERT INTO host_rollback_effects VALUES (71)")
    monkeypatch.setattr(manager, "_prune_all_fully_removed_child_tracking", reconcile)
    lock = manager._lock if surface in ("seal", "identity") else manager._quarantined_shutdown_drain_lock
    operation = (manager._set_quarantined_shutdown_handoffs_sealed(True) if surface == "seal" else
                 manager._await_before_spawn_admission_deadline(reconcile,
                     deadline=asyncio.get_running_loop().time()+10, phase="declared coordinator admission", agent=agent)
                 if surface == "deadline" else
                 manager._admit_agent_runtime_offboarding_identity(name="never-delete", agent_id="never-delete", config=None,
                     revoked=None, pending_offboarding=[]) if surface == "identity" else
                 manager._acquire_quarantined_shutdown_drain() if surface == "drain" else
                 manager._reconcile_fully_removed_child_tracking())
    predicate = (lambda op: getattr(op, "cr_code", None) is reconcile.__code__) if surface in ("reconcile", "deadline") else (
        lambda op: getattr(op, "cr_code", None) is asyncio.Lock.acquire.__code__)
    fault = NativeAdmissionFault(predicate, source.error, mode, eager=eager)
    owner = create_owned_task(operation, retain_originals=True)
    try:
        with fault:
            result = await await_owned_task(owner)
        fault.assert_terminal_unstarted()
        source.assert_retained(result.error)
        assert not entries
        assert not manager._lock.locked() and not manager._quarantined_shutdown_drain_lock.locked()
        assert not manager._quarantined_shutdown_handoffs_sealed
        assert await database.fetchval("SELECT count(*) FROM host_rollback_effects") == 0
        assert await native_pg[1].fetchval('SELECT count(*) FROM "' + native_pg[2] + '".effects WHERE id=231') == 1
    finally:
        await await_owned_task(owner)
        for task in fault.accepted:
            await await_owned_task(task)
        for operation in (*fault.operations, *fault.refused):
            # Exact old-source refusal negatives lack the canonical closer.
            # They own only these never-admitted fixture coroutines.
            if getattr(operation, "cr_frame", None) is not None and operation.cr_await is None:
                operation.close()
        # The failing old-source negative can have acquired an unreturned lock.
        # Release only this isolated fixture's exact lock, after joining its Task.
        if lock.locked():
            lock.release()


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["seal", "drain", "identity"])
@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("source_cancel", [False, True])
@pytest.mark.parametrize("unexpected", [False, True], ids=["ordinary-control", "unexpected-control"])
async def test_physical_lock_acquisition_is_released_when_observation_fails(
    native_pg, onboarding_owner, monkeypatch, surface, eager, source_cancel, unexpected,
):
    manager, _, database = onboarding_owner
    original = await _original_control(native_pg[0], monkeypatch, source_cancel)
    if unexpected:
        class UnexpectedObservation(BaseException):
            pass
        failure = UnexpectedObservation("actual non-lifecycle completion observation failed")
        failure.__cause__ = original
        original = failure
    source = OriginalFailure.capture(original)
    lock = manager._lock if surface in ("seal", "identity") else manager._quarantined_shutdown_drain_lock
    actual = lock.acquire
    entered, release, attached = asyncio.Event(), asyncio.Event(), asyncio.Event()
    accepted = []
    async def acquire():
        result = await actual()
        await database.execute("INSERT INTO host_rollback_effects VALUES (72)")
        entered.set()
        await release.wait()
        return result
    monkeypatch.setattr(lock, "acquire", acquire)
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    factory = observation_factory(lambda op: getattr(op, "cr_code", None) is acquire.__code__,
                                  source.error, attached, accepted, previous, eager=eager)
    operation = (manager._set_quarantined_shutdown_handoffs_sealed(True) if surface == "seal" else
                 manager._admit_agent_runtime_offboarding_identity(name="never-delete", agent_id="never-delete", config=None,
                     revoked=None, pending_offboarding=[]) if surface == "identity" else
                 manager._acquire_quarantined_shutdown_drain())
    owner = create_owned_task(operation, retain_originals=True)
    try:
        loop.set_task_factory(factory)
        await asyncio.wait_for(entered.wait(), 5)
        await asyncio.wait_for(attached.wait(), 5)
        assert lock.locked() and not owner.done()
        release.set()
        result = await await_owned_task(owner)
        loop.set_task_factory(previous)
        assert len(accepted) == 1 and accepted[0].done()
        assert owned_task_error(accepted[0]) is None and accepted[0].result() is True
        source.assert_retained(result.error)
        assert not lock.locked(), "physically acquired lock was stranded by its observation failure"
        assert not manager._quarantined_shutdown_handoffs_sealed
        assert not any(resource is lock for resource, _ in execution_cleanup_debts(result.error))
        assert await database.fetchval("SELECT count(*) FROM host_rollback_effects") == 1
    finally:
        release.set()
        loop.set_task_factory(previous)
        await await_owned_task(owner)
        for task in accepted:
            await await_owned_task(task)
        if lock.locked():
            lock.release()
