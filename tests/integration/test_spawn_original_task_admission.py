"""Both spawn entries protect the original name across Task admission refusal."""
import asyncio
import inspect

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.spawn.mandate import SpawnMandate
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_host_onboarding_original_admission import onboarding_owner as onboarding_owner
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.unit.test_host_context_native_cleanup import _actual_operation
from tests.utils.original_failure import OriginalFailure


@pytest.mark.asyncio
@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("fault", ["refuse", "publish-before", "publish-after"])
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_spawn_entry_closes_recipe_joins_accepted_owner_and_retires_name(
    native_pg, onboarding_owner, monkeypatch, direct, fault, source_cancel,
):
    manager, parent, _database = onboarding_owner
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    manager._agents["Parent"] = parent
    manager._agent_names[parent.agent_id] = "Parent"
    accepted, refused, fired, bodies = [], [], [], []
    actual_direct = manager._do_spawn
    async def body(*args, **kwargs):
        bodies.append(True)
        raise RuntimeError("declared spawn body must remain unstarted after refusal")
    monkeypatch.setattr(manager, "_do_spawn", body)
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    class PublicationTask(asyncio.Task):
        def add_done_callback(self, callback, *args, **kwargs):
            if not fired and getattr(callback, "__name__", "") == "close_unstarted":
                fired.append(True)
                if fault == "publish-after":
                    super().add_done_callback(callback, *args, **kwargs)
                raise source.error
            return super().add_done_callback(callback, *args, **kwargs)
    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        code = getattr(operation, "cr_code", None)
        if code is not None and code.co_name == ("run_direct_spawn" if direct else "_run_admitted_spawn"):
            if fault == "refuse":
                fired.append(True)
                refused.append((coroutine, operation))
                raise source.error
            task = PublicationTask(coroutine, loop=loop, **kwargs)
            accepted.append(task)
            return task
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)
    mandate = SpawnMandate(parent_did=parent.agent_id)
    operation = actual_direct("AdmittedChild", parent, mandate) if direct else manager.spawn_agent("AdmittedChild", parent, mandate)
    owner = create_owned_task(operation, retain_originals=True)
    loop.set_task_factory(factory)
    try:
        result = await await_owned_task(owner)
        source.assert_retained(result.error)
        assert fired and not bodies
        assert all(task.done() for task in accepted)
        assert all(inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED for pair in refused for coro in pair)
        assert not manager._agent_operations and manager._pending_spawns == 0
        admission, created = await manager._admit_agent_operation("AdmittedChild", kind="load")
        assert created, "failed Task admission permanently reserved its original name"
        await manager._release_agent_operation(admission)
    finally:
        loop.set_task_factory(previous)
        await await_owned_task(owner)
        for task in accepted:
            await await_owned_task(task)
        for pair in refused:
            for coroutine in pair:
                coroutine.close()
