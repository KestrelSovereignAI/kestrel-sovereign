"""Each rollback phase owns its actual accepted or refused recipe."""
import asyncio
import inspect

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import execution_cleanup_debts
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_host_onboarding_original_admission import onboarding_owner as onboarding_owner
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.unit.test_host_context_native_cleanup import _actual_operation
from tests.utils.original_failure import OriginalFailure

PHASES = (
    "_downgrade_uncommitted_spawn_receipt",
    "_withdraw_uncommitted_spawn_startup_registration",
    "_rollback_uncommitted_spawn_runtime",
    "_withdraw_uncommitted_spawn_authority_witness",
)


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", PHASES)
@pytest.mark.parametrize("mode", ["refuse", "publish-before", "publish-after"])
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_original_spawn_rollback_phase_owns_admission(
    native_pg, onboarding_owner, monkeypatch, phase, mode, source_cancel,
):
    manager, child, database = onboarding_owner
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    admission, owns = await manager._admit_agent_operation("PhaseRollback", kind="spawn")
    assert owns
    started, accepted, refused, handoffs = [], [], [], []

    async def phase_body(label):
        started.append(label)
        await database.execute("INSERT INTO host_rollback_effects VALUES (?)", (321 + PHASES.index(label),))
        return False

    for label in PHASES:
        monkeypatch.setattr(manager, label, lambda *_args, _label=label, **_kwargs: phase_body(_label))

    async def handoff(*_args, **_kwargs):
        handoffs.append(admission)
        return False

    monkeypatch.setattr(manager, "_handoff_failed_spawn_cleanup", handoff)
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
        frame = getattr(operation, "cr_frame", None)
        if frame is not None and frame.f_locals.get("label") == phase:
            if mode == "refuse":
                refused.extend((coroutine, operation))
                raise source.error
            task = PublicationTask(coroutine, loop=loop, **kwargs)
            accepted.append(task)
            return task
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)

    owner = create_owned_task(manager._rollback_uncommitted_spawn(admission, child), retain_originals=True)
    loop.set_task_factory(factory)
    try:
        result = await await_owned_task(owner)
        source.assert_retained(result.error)
        assert started == list(PHASES[:PHASES.index(phase)])
        assert await database.fetchval("SELECT count(*) FROM host_rollback_effects") == PHASES.index(phase)
        assert handoffs == [admission] and admission.rollback_incomplete
        assert all(task.done() for task in accepted)
        assert all(inspect.getcoroutinestate(recipe) == inspect.CORO_CLOSED for recipe in refused)
        expected = None if mode == "refuse" else accepted[0]
        assert any(resource is admission and task is expected for resource, task in execution_cleanup_debts(result.error))
    finally:
        loop.set_task_factory(previous)
        await await_owned_task(owner)
        for task in accepted:
            await await_owned_task(task)
        for recipe in refused:
            recipe.close()
        await manager._release_agent_operation(admission)
