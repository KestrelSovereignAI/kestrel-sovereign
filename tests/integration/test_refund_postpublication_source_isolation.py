"""Cached publication diagnostics cannot rewrite an actual native refund source.

Refund callback is a declared real SQLite effect, not paid funds acceptance.
"""
import asyncio

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import _explicit_control_chain, execution_cleanup_debts
from tests.integration.test_execution_authority_postgres import native_pg as _native_pg
from tests.integration.test_host_onboarding_original_admission import onboarding_owner as _onboarding_owner
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.unit.test_host_context_native_cleanup import _actual_operation
from tests.utils.original_failure import OriginalFailure

native_pg, onboarding_owner = _native_pg, _onboarding_owner


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["observer-before", "observer-after", "gate-then-name"])
@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_cached_refund_diagnostics_preserve_native_source_and_pending_admission_debt(
    native_pg, onboarding_owner, monkeypatch, mode, eager, source_cancel,
):
    manager, _agent, db = onboarding_owner
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    resource = object()
    manager._child_budgets["PublishedBudget"] = resource
    publication, naming = RuntimeError("actual refund observer publication failed"), RuntimeError("actual refund diagnostic naming failed")
    accepted, cached_before_name, effects = [], [], []
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()

    async def refund():
        await db.execute("INSERT INTO host_rollback_effects VALUES (354)")
        effects.append(True)
        raise source.error

    class PublicationTask(asyncio.Task):
        def add_done_callback(self, callback, *args, **kwargs):
            name = getattr(callback, "__name__", "")
            if mode == "gate-then-name" and name == "close_unstarted":
                raise source.error
            if mode.startswith("observer-") and name == "observe_budget_release_completion":
                if mode == "observer-after":
                    super().add_done_callback(callback, *args, **kwargs)
                raise publication
            return super().add_done_callback(callback, *args, **kwargs)

        def set_name(self, name):
            if mode == "gate-then-name" and str(name).startswith("agent_budget_release:PublishedBudget"):
                assert not self.done(), "naming fault must hit the pending accepted gate"
                cached_before_name.append(getattr(self, "_kestrel_owned_terminal_error", None))
                raise naming
            return super().set_name(name)

    def factory(loop, coroutine, **kwargs):
        if getattr(_actual_operation(coroutine), "cr_code", None) is refund.__code__:
            task = PublicationTask(coroutine, loop=loop, eager_start=eager, **kwargs)
            accepted.append(task)
            return task
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)

    record = None
    try:
        loop.set_task_factory(factory)
        async with manager._lock:
            record = manager._start_child_budget_release("PublishedBudget", release_operation=refund)
            assert record.task is accepted[0]
        result = await await_owned_task(create_owned_task(
            manager._await_child_budget_release_cancellation_safe(record.task), retain_originals=True,
        ))
        loop.set_task_factory(previous)
        source.assert_retained(result.error)
        chain = tuple(_explicit_control_chain(result.error))
        assert record.task.done() and len(accepted) == 1
        if mode == "gate-then-name":
            assert naming in chain and len(cached_before_name) == 1
            assert cached_before_name[0] in chain
            assert any(value is resource and task is record.task for value, task in execution_cleanup_debts(result.error))
            assert effects == []
        else:
            assert publication in chain and effects == [True]
        assert await db.fetchval("SELECT count(*) FROM host_rollback_effects WHERE id=354") == (0 if mode == "gate-then-name" else 1)
    finally:
        loop.set_task_factory(previous)
        if record is not None:
            await await_owned_task(record.task)
        for task in accepted:
            await await_owned_task(task)
        manager._child_budgets.pop("PublishedBudget", None)
