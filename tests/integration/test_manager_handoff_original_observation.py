"""Physical lock/refund ownership survives original join/publication faults."""
import asyncio

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import execution_cleanup_debts
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_host_onboarding_original_admission import onboarding_owner as onboarding_owner
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.unit.test_host_context_native_cleanup import _actual_operation
from tests.utils.original_failure import OriginalFailure
from tests.utils.native_admission_fault import NativeAdmissionFault


@pytest.mark.asyncio
async def test_handoff_releases_physically_acquired_lock_after_join_observer_failure(
    native_pg, onboarding_owner, monkeypatch,
):
    manager, child, _database = onboarding_owner
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch))
    admission, owns = await manager._admit_agent_operation("HandoffLock", kind="spawn")
    assert owns
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    physical = []

    class FailedRemovalTask(asyncio.Task):
        failed = False
        def remove_done_callback(self, callback):
            removed = super().remove_done_callback(callback)
            if self.done() and not self.failed:
                self.failed = True
                physical.append(manager._lock.locked())
                raise source.error
            return removed

    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        code = getattr(operation, "cr_code", None)
        if code is not None and (code.co_qualname.endswith(".acquire_manager_lock") or code.co_qualname == "Lock.acquire"):
            return FailedRemovalTask(coroutine, loop=loop, **kwargs)
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)

    owner = create_owned_task(manager._handoff_failed_spawn_cleanup(admission, child), retain_originals=True)
    loop.set_task_factory(factory)
    try:
        result = await await_owned_task(owner)
        source.assert_retained(result.error)
        assert physical == [True], "fault did not occur after actual lock acquisition"
        assert not manager._lock.locked(), "successful acquisition leaked through failed observation"
        assert not manager._quarantined_shutdown_reapers
    finally:
        loop.set_task_factory(previous)
        await await_owned_task(owner)
        if manager._lock.locked():
            manager._lock.release()
        await manager._release_agent_operation(admission)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["refuse", "publish-before", "publish-after"])
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_budget_release_inner_admission_joins_original_task(
    native_pg, onboarding_owner, monkeypatch, mode, source_cancel,
):
    manager, _agent, database = onboarding_owner
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    resource = object()
    manager._child_budgets["OriginalBudget"] = resource
    accepted, refused, effects = [], [], []

    async def refund(_name):
        await database.execute("INSERT INTO host_rollback_effects VALUES (340)")
        effects.append(True)
    monkeypatch.setattr(manager, "_release_child_budget", refund)
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()

    class PublicationTask(asyncio.Task):
        failed = False
        def add_done_callback(self, callback, *args, **kwargs):
            if not self.failed and getattr(callback, "__name__", "") == "close_unstarted":
                self.failed = True
                if mode == "publish-after":
                    super().add_done_callback(callback, *args, **kwargs)
                raise source.error
            return super().add_done_callback(callback, *args, **kwargs)

    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        if getattr(operation, "cr_code", None) is refund.__code__:
            if mode == "refuse":
                refused.append(coroutine)
                raise source.error
            task = PublicationTask(coroutine, loop=loop, **kwargs)
            accepted.append(task)
            return task
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)

    owner = create_owned_task(manager._release_child_budget_cancellation_safe("OriginalBudget"), retain_originals=True)
    loop.set_task_factory(factory)
    try:
        result = await await_owned_task(owner)
        source.assert_retained(result.error)
        assert len(refused if mode == "refuse" else accepted) == 1
        assert all(task.done() for task in accepted) and not effects
        assert await database.fetchval("SELECT count(*) FROM host_rollback_effects WHERE id=340") == 0
        expected = None if mode == "refuse" else accepted[0]
        assert any(value is resource and task is expected for value, task in execution_cleanup_debts(result.error))
    finally:
        loop.set_task_factory(previous)
        await await_owned_task(owner)
        for task in accepted:
            await await_owned_task(task)
        for coroutine in refused:
            coroutine.close()
        manager._child_budgets.pop("OriginalBudget", None)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["refuse", "publish-before", "publish-after"])
@pytest.mark.parametrize("eager", [False, True])
async def test_standalone_descendant_boundary_joins_original_before_releasing_manager(
    native_pg, onboarding_owner, monkeypatch, mode, eager,
):
    # Actual manager/partial Agent storage lifetime; this deliberately fails
    # admission before descendant routing, not whole-tree Stop acceptance.
    from kestrel_sovereign.features.spawn.feature import SpawnFeature

    manager, agent, database = onboarding_owner
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch))
    manager._agents["Original"] = agent
    agent._agent_manager = manager
    feature = SpawnFeature(agent)
    feature._agent_manager = manager
    feature._standalone_manager_owned = True
    connection = database.backend._connection
    worker = getattr(connection, "_thread", connection)
    target = type(manager).shutdown_spawn_authority_tree.__code__
    fault = NativeAdmissionFault(lambda operation: getattr(operation, "cr_code", None) is target, source.error, mode, eager=eager)
    owner = create_owned_task(feature._shutdown_after_delegated_tasks(manager, []), retain_originals=True)
    try:
        with fault:
            result = await await_owned_task(owner)
        fault.assert_terminal_unstarted()
        source.assert_retained(result.error)
        assert any(resource is manager and task is fault.task for resource, task in execution_cleanup_debts(result.error))
        assert feature._agent_manager is agent._agent_manager is manager
        assert manager._agents["Original"] is agent and database.backend._connection is connection and worker.is_alive()
    finally:
        await await_owned_task(owner)
        for task in fault.accepted:
            await await_owned_task(task)
        manager._agents.pop("Original", None)
