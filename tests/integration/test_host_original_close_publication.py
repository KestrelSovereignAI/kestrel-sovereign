"""Actual independent SQLite resources retain their original host close owners."""
import asyncio

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import execution_cleanup_debts
from kestrel_sovereign.host_features.context import _close_partial_host_resources
from kestrel_sovereign.storage.db.sqlite import SQLiteBackend
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.unit.test_host_context_native_cleanup import _actual_operation
from tests.utils.original_failure import OriginalFailure


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["refuse", "publish-before", "publish-after"])
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_host_closer_publication_retains_exact_task_and_continues_independent_close(
    native_pg, monkeypatch, mode, source_cancel,
):
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    first, second = SQLiteBackend(":memory:"), SQLiteBackend(":memory:")
    await first.connect()
    await second.connect()
    first_connection, second_connection = first._connection, second._connection
    first_worker = getattr(first_connection, "_thread", first_connection)
    second_worker = getattr(second_connection, "_thread", second_connection)
    owners, accepted, refused = {}, [], []
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
        if frame is not None and frame.f_locals.get("self") is first and getattr(operation, "cr_code", None) is SQLiteBackend.close.__code__:
            if mode == "refuse":
                refused.append(coroutine)
                raise source.error
            task = PublicationTask(coroutine, loop=loop, **kwargs)
            accepted.append(task)
            return task
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)

    owner = create_owned_task(_close_partial_host_resources(
        None, None, first, second, owners=owners,
    ), retain_originals=True)
    loop.set_task_factory(factory)
    try:
        result = await await_owned_task(owner)
        source.assert_retained(result.error)
        expected = None if mode == "refuse" else accepted[0]
        assert all(task.done() for task in accepted)
        assert owners[id(first)].task is expected and not owners[id(first)].closed
        assert any(resource is first and task is expected for resource, task in execution_cleanup_debts(result.error))
        assert first._connection is first_connection and first_worker.is_alive()
        assert second._connection is None and not second_worker.is_alive()
        assert owners[id(second)].closed
    finally:
        loop.set_task_factory(previous)
        await await_owned_task(owner)
        for task in accepted:
            await await_owned_task(task)
        for coroutine in refused:
            coroutine.close()
        await first.close()
        await second.close()
