"""Advisory retirement distinguishes no Task from failed Task publication."""
import asyncio

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import execution_cleanup_debts
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_execution_authority_postgres import native_pg as _work_pg
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.unit.test_host_context_native_cleanup import _actual_operation
from tests.utils.original_failure import OriginalFailure
from kestrel_sovereign.storage.db.postgres import TerminalAdvisoryBarrier

work_pg = _work_pg


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["refuse", "publish-before", "publish-after"])
@pytest.mark.parametrize("source_cancel", [False, True])
@pytest.mark.parametrize("terminal", [False, True])
async def test_advisory_closer_retains_original_accepted_task(
    native_pg, work_pg, monkeypatch, mode, source_cancel, terminal,
):
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    backend, observer, _schema = work_pg
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    sessions, accepted, refused = [], [], []

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
        if sessions and frame is not None and frame.f_locals.get("self") is sessions[0][1] and getattr(operation.cr_code, "co_qualname", "").endswith("Connection.close"):
            if mode == "refuse":
                refused.append(coroutine)
                raise source.error
            task = PublicationTask(coroutine, loop=loop, **kwargs)
            accepted.append(task)
            return task
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)

    async def invoke():
        keys = TerminalAdvisoryBarrier(((210, 338),)) if terminal else [(210, 338)]
        async with backend.advisory_locks(keys) as lease:
            sessions.append((lease._connection, lease._connection._con))
            raise OSError("actual original advisory operation failed")

    owner = create_owned_task(invoke(), retain_originals=True)
    loop.set_task_factory(factory)
    try:
        result = await await_owned_task(owner)
        source.assert_retained(result.error)
        assert len(refused if mode == "refuse" else accepted) == 1
        assert all(task.done() for task in accepted)
        exact_owner = None if mode == "refuse" else accepted[0]
        assert any(getattr(resource, "context_manager", None) is sessions[0][0] and task is exact_owner
                   for resource, task in execution_cleanup_debts(result.error))
        async with asyncio.timeout(3):
            while not await observer.fetchval("SELECT pg_try_advisory_lock(210,338)"):
                await asyncio.sleep(.01)
        assert await observer.fetchval("SELECT pg_advisory_unlock(210,338)")
    finally:
        loop.set_task_factory(previous)
        await await_owned_task(owner)
        for task in accepted:
            await await_owned_task(task)
        await observer.execute("SELECT pg_advisory_unlock(210,338)")
