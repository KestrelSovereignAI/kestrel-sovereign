"""Original real SQLite worker stays owned when its inner closer cannot publish."""
import asyncio

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import execution_cleanup_debts
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.db.sqlite import _close_aiosqlite_connection
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.unit.test_host_context_native_cleanup import _actual_operation
from tests.utils.original_failure import OriginalFailure


@pytest.mark.asyncio
@pytest.mark.parametrize("accepted", [False, True])
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_sqlite_inner_close_keeps_actual_connection_and_original_owner(
    native_pg, tmp_path, monkeypatch, accepted, source_cancel,
):
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    database = await AsyncDatabase.sqlite(str(tmp_path / "inner-close-admission.db"))
    backend, connection = database.backend, database.backend._connection
    worker = getattr(connection, "_thread", connection)
    await database.execute("CREATE TABLE durable_effects (id INTEGER)")
    await database.execute("INSERT INTO durable_effects VALUES (1)")
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    owners, recipes, fired = [], [], []
    class PublicationTask(asyncio.Task):
        def add_done_callback(self, callback, *args, **kwargs):
            if not fired and getattr(callback, "__name__", "") == "close_unstarted":
                fired.append(True)
                super().add_done_callback(callback, *args, **kwargs)
                raise source.error
            return super().add_done_callback(callback, *args, **kwargs)
    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        code = getattr(operation, "cr_code", None)
        if code is not None and code.co_qualname == "Connection.close":
            recipes.append(operation)
            if not accepted:
                fired.append(True)
                raise source.error
            task = PublicationTask(coroutine, loop=loop, **kwargs)
            owners.append(task)
            return task
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)
    owner = None
    try:
        loop.set_task_factory(factory)
        owner = create_owned_task(backend.close(), retain_originals=True)
        result = await await_owned_task(owner)
        loop.set_task_factory(previous)
        for task in owners:
            await await_owned_task(task)
        assert fired and len(recipes) == 1 and recipes[0].cr_frame is None
        source.assert_retained(result.error)
        assert any(resource is connection and task is (owners[0] if accepted else None)
                   for resource, task in execution_cleanup_debts(result.error))
        retained = backend._retired_connection_closes[connection]
        assert retained.close_task is (owners[0] if accepted else None)
        assert worker.is_alive() and backend.connection_retirement_pending
        repeated = await await_owned_task(create_owned_task(backend.close(), retain_originals=True))
        source.assert_retained(repeated.error)
        assert len(recipes) == 1, "native-refused original closer was replaced"
        # SQLAlchemy and other coordinating callers can probe the SAME helper
        # with no explicit Task. Its existing record must supply that owner or
        # preserve its explicit absence, never allocate a replacement close.
        helper_probe = await await_owned_task(create_owned_task(_close_aiosqlite_connection(
            connection, retained_closes=backend._retired_connection_closes), retain_originals=True))
        source.assert_retained(helper_probe.error)
        assert worker.is_alive() and backend._retired_connection_closes[connection] is retained
    finally:
        loop.set_task_factory(previous)
        if owner is not None:
            await await_owned_task(owner)
        for task in owners:
            await await_owned_task(task)
        # The original attempted closer never entered Connection.close().
        # Operator reconciliation, not a product retry or a fabricated ACK.
        await connection.close()
        await asyncio.to_thread(worker.join, 5)
        assert not worker.is_alive()
        await await_owned_task(create_owned_task(backend.close(), retain_originals=True))
