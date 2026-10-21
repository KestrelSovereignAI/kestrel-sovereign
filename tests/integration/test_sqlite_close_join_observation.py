"""Actual cancelled-write drain and worker close retain join control/local C."""
import asyncio
import sqlite3

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import _explicit_control_chain
from kestrel_sovereign.storage.async_database import AsyncDatabase
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.unit.test_host_context_native_cleanup import _actual_operation
from tests.utils.original_failure import OriginalFailure


@pytest.mark.asyncio
@pytest.mark.parametrize("remove", [False, True])
async def test_sqlite_close_keeps_original_drain_observation_and_actual_withdrawal(native_pg, tmp_path, monkeypatch, remove):
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch))
    path = tmp_path / "original-drain-observer.db"
    database = await AsyncDatabase.sqlite(str(path))
    backend, conn = database.backend, database.backend._connection
    worker = getattr(conn, "_thread", conn)
    await database.execute("CREATE TABLE observations (id INTEGER PRIMARY KEY)")
    await conn.execute("BEGIN")
    await conn.execute("INSERT INTO observations VALUES (1)")
    entered, release, waiting = asyncio.Event(), asyncio.Event(), asyncio.Event()
    actual_rollback, actual_wait = conn.rollback, asyncio.wait
    withdrawals, accepted = [], []
    async def rollback():
        await actual_rollback()
        entered.set()
        await release.wait()
    monkeypatch.setattr(conn, "rollback", rollback)
    drain = None
    async def wait(owners, *args, **kwargs):
        observes_drain = drain is not None and any(task is drain for task in owners)
        if observes_drain:
            waiting.set()
        try:
            return await actual_wait(owners, *args, **kwargs)
        except asyncio.CancelledError as error:
            if observes_drain:
                withdrawals.append(error)
            raise
    monkeypatch.setattr(asyncio, "wait", wait)
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    class PublicationTask(asyncio.Task):
        failed = False
        def add_done_callback(self, callback, *args, **kwargs):
            if not remove and not self.failed and getattr(callback, "__name__", "") == "mark_complete":
                self.failed = True
                super().add_done_callback(callback, *args, **kwargs)
                raise source.error
            return super().add_done_callback(callback, *args, **kwargs)
        def remove_done_callback(self, callback):
            result = super().remove_done_callback(callback)
            if remove and not self.failed and getattr(callback, "__name__", "") == "mark_complete":
                self.failed = True
                raise source.error
            return result
    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        code = getattr(operation, "cr_code", None)
        if code is not None and code.co_qualname == "SQLiteBackend._drain_cancelled_write":
            task = PublicationTask(coroutine, loop=loop, **kwargs)
            accepted.append(task)
            return task
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)
    owner = None
    loop.set_task_factory(factory)
    try:
        backend._handoff_cancelled_write(conn)
        drain = backend._cancelled_write_drain
        await asyncio.wait_for(entered.wait(), 5)
        owner = create_owned_task(backend.close(), retain_originals=True)
        await asyncio.wait_for(waiting.wait(), 5)
        owner.cancel("actual SQLite close observer withdrew")
        outcome = await await_owned_task(owner)
        source.assert_retained(outcome.error)
        assert withdrawals and all(error in tuple(_explicit_control_chain(outcome.error)) for error in withdrawals), "SQLite close lost actual caller withdrawal behind join control"
        assert accepted == [drain] and drain.done() and backend._connection is None and not worker.is_alive()
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as observer:
            assert observer.execute("SELECT count(*) FROM observations").fetchone()[0] == 0
    finally:
        loop.set_task_factory(previous)
        release.set()
        if owner is not None:
            await await_owned_task(owner)
        if drain is not None:
            await await_owned_task(drain)
        await backend.close()
