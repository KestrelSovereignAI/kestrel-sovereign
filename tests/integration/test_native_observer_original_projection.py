"""Native inventory and iterator observers preserve independent original graphs."""
import asyncio

import pytest

from kestrel_sovereign._async_ownership import OwnedAsyncIterator, await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import execution_cleanup_debts
from kestrel_sovereign.native_retirement import NativeExecutionObserver, bind_native_execution_observer
from kestrel_sovereign.storage.async_database import AsyncDatabase
from tests.integration.test_completion_observer_consumers import _publication_task_type
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.unit.test_host_context_native_cleanup import _actual_operation
from tests.utils.original_failure import OriginalFailure


@pytest.mark.asyncio
async def test_inventory_adds_open_pg_body_debt_without_mutating_original_pg_control(native_pg, tmp_path, monkeypatch):
    inventory = NativeExecutionObserver()
    with bind_native_execution_observer(inventory):
        await _original_control(native_pg[0], monkeypatch)
    # Read the ACTUAL backend-observed source, not an arbitrary foreign error
    # attribute or a manually assigned inventory receipt.
    source = OriginalFailure.capture(inventory._error)
    assert source.error is not None and not inventory._resources
    backend = native_pg[0]
    entered, release = asyncio.Event(), asyncio.Event()
    async def body():
        with bind_native_execution_observer(inventory):
            async with backend.transaction():
                await backend.execute("INSERT INTO effects VALUES (97, 'actual independent held native body')")
                entered.set()
                await release.wait()
    owner = create_owned_task(body(), retain_originals=True)
    try:
        await asyncio.wait_for(entered.wait(), 5)
        result = await inventory.join_accepted()
        source.assert_retained(result.error)
        assert not owner.done()
        assert any(getattr(resource, "body_owner", None) is owner and task is None
                   for resource, task in execution_cleanup_debts(result.error))
    finally:
        release.set()
        await await_owned_task(owner)


@pytest.mark.asyncio
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_completed_iterator_watcher_keeps_independent_native_cancellation(native_pg, tmp_path, monkeypatch, source_cancel):
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    database = await AsyncDatabase.sqlite(str(tmp_path / "watcher-native.db"))
    await database.execute("CREATE TABLE source_rows (id INTEGER)")
    async def produce():
        await database.execute("INSERT INTO source_rows VALUES (1)")
        return
        yield None
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    task_type = _publication_task_type(source.error, remove=True, after=True,
        enabled=lambda: asyncio.current_task().get_name().startswith("owned_async_iterator_cancel:"))
    accepted = []
    def factory(loop, coroutine, **kwargs):
        code = getattr(_actual_operation(coroutine), "cr_code", None)
        if code is not None and code.co_qualname == "OwnedAsyncIterator.__init__.<locals>.run_owned":
            task = task_type(coroutine, loop=loop, **kwargs)
            accepted.append(task)
            return task
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)
    loop.set_task_factory(factory)
    stream = OwnedAsyncIterator(produce, operation="actual-native-watcher-observation")
    try:
        result = await await_owned_task(create_owned_task(anext(stream), retain_originals=True))
        source.assert_retained(result.error)
        from kestrel_sovereign._async_ownership import owned_task_error
        assert owned_task_error(stream._owner) is None, "original producer did not physically succeed"
        assert accepted and all(task.done() for task in accepted)
        assert stream._cancellation_owner.done()
        assert await database.fetchval("SELECT count(*) FROM source_rows") == 1
    finally:
        loop.set_task_factory(previous)
        await await_owned_task(create_owned_task(stream.aclose(), retain_originals=True))
        await database.close()
