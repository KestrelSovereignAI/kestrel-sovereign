"""Close timeout and the earlier native drain retain every delivered caller C."""
import asyncio
import sqlite3
from contextlib import closing

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import _explicit_control_chain
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.db import sqlite as sqlite_module
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.utils.original_failure import OriginalFailure


@pytest.mark.asyncio
async def test_close_deadline_keeps_actual_withdrawal_and_same_retained_owner(native_pg, tmp_path, monkeypatch):
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch))
    database = await AsyncDatabase.sqlite(str(tmp_path / "close-deadline.db"))
    backend, connection = database.backend, database.backend._connection
    worker = getattr(connection, "_thread", connection)
    entered, release, closers, withdrawals = asyncio.Event(), asyncio.Event(), [], []
    actual_close, actual_wait = connection.close, asyncio.wait
    async def close():
        closers.append(asyncio.current_task())
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()
        await actual_close()
    async def wait(*args, **kwargs):
        try:
            return await actual_wait(*args, **kwargs)
        except asyncio.CancelledError as error:
            if asyncio.current_task() is owner:
                withdrawals.append(error)
            raise
    monkeypatch.setattr(connection, "close", close)
    monkeypatch.setattr(asyncio, "wait", wait)
    monkeypatch.setattr(sqlite_module, "AIOSQLITE_WORKER_SHUTDOWN_TIMEOUT_S", 0.15)
    backend._cancelled_write_drain_error = source.error
    owner = create_owned_task(backend.close(), retain_originals=True)
    try:
        await asyncio.wait_for(entered.wait(), 5)
        owner.cancel("actual close waiter C before original deadline")
        result = await await_owned_task(owner)
        source.assert_retained(result.error)
        assert withdrawals and all(error in tuple(_explicit_control_chain(result.error)) for error in withdrawals)
        assert len(closers) == 1 and worker.is_alive()
        retained = backend._retired_connection_closes[connection]
        assert retained.close_task is closers[0] and backend.connection_retirement_pending
    finally:
        release.set()
        await await_owned_task(owner)
        for task in closers:
            await await_owned_task(task)
        retained = backend._retired_connection_closes.get(connection)
        if retained is not None and retained.retirement_task is not None:
            await await_owned_task(retained.retirement_task)
        await sqlite_module._wait_for_aiosqlite_worker_shutdown(connection, timeout_s=1)
        assert not worker.is_alive()


@pytest.mark.asyncio
async def test_close_drain_keeps_two_distinct_actual_observer_withdrawals(native_pg, tmp_path, monkeypatch):
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch))
    await _assert_close_drain_withdrawals(tmp_path, monkeypatch, source=source)


@pytest.mark.asyncio
async def test_close_drain_observer_uses_original_join_and_ignores_foreign_borrower(tmp_path, monkeypatch):
    """Healthy native SQLite control; the separate PG source remains CI-only."""
    await _assert_close_drain_withdrawals(tmp_path, monkeypatch, foreign_borrower=True)


async def _assert_close_drain_withdrawals(tmp_path, monkeypatch, *, source=None, foreign_borrower=False):
    path = tmp_path / "drain-two-withdrawals.db"
    database = await AsyncDatabase.sqlite(str(path))
    backend, connection = database.backend, database.backend._connection
    worker = getattr(connection, "_thread", connection)
    await database.execute("CREATE TABLE effects (id INTEGER)")
    await connection.execute("BEGIN")
    await connection.execute("INSERT INTO effects VALUES (1)")
    entered, release, waiting, unwinding, joining = (asyncio.Event() for _ in range(5))
    actual_rollback = connection.rollback
    withdrawals, owner, borrowed = [], None, None
    from tests.shared.owned_task_observation import observe_bounded_asyncio_wait, observe_owned_task
    async def rollback():
        await actual_rollback()
        entered.set()
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                unwinding.set()
    def entered_wait(tasks):
        if asyncio.current_task() is owner and drain in tasks:
            waiting.set()
    def withdrew_wait(tasks, error):
        if asyncio.current_task() is owner and drain in tasks:
            withdrawals.append(error)
    def entered_join(task):
        if asyncio.current_task() is owner and task is drain:
            joining.set()
    def joined(task, outcome):
        if asyncio.current_task() is owner and task is drain and outcome.cancellation is not None:
            withdrawals.append(outcome.cancellation)
    monkeypatch.setattr(connection, "rollback", rollback)
    observe_bounded_asyncio_wait(monkeypatch, on_enter=entered_wait, on_withdrawal=withdrew_wait)
    observe_owned_task(monkeypatch, on_enter=entered_join, on_outcome=joined)
    backend._handoff_cancelled_write(connection)
    drain = backend._cancelled_write_drain
    try:
        await asyncio.wait_for(entered.wait(), 5)
        if source is not None:
            backend._cancelled_write_drain_error = source.error
        owner = create_owned_task(backend.close(), retain_originals=True)
        await asyncio.wait_for(waiting.wait(), 5)
        if foreign_borrower:
            borrower_entered = asyncio.Event()
            async def borrow():
                borrower_entered.set()
                await asyncio.wait({drain})
            borrowed = create_owned_task(borrow(), retain_originals=True)
            await asyncio.wait_for(borrower_entered.wait(), 5)
            borrowed.cancel('foreign drain observer withdrawal')
            observed = await await_owned_task(borrowed)
            assert isinstance(observed.error, asyncio.CancelledError)
            assert withdrawals == [] and not drain.cancelling()
        owner.cancel("actual earlier drain waiter C1")
        await asyncio.wait_for(unwinding.wait(), 5)
        await asyncio.wait_for(joining.wait(), 5)
        owner.cancel("actual canonical drain join C2")
        release.set()
        result = await await_owned_task(owner)
        if source is not None:
            source.assert_retained(result.error)
        assert len(withdrawals) == 2 and withdrawals[0] is not withdrawals[1]
        assert withdrawals[0].args == ('actual earlier drain waiter C1',)
        assert withdrawals[1].args == ('actual canonical drain join C2',)
        assert all(error in tuple(_explicit_control_chain(result.error)) for error in withdrawals)
        assert drain.done() and backend._connection is None and not worker.is_alive()
        with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as observer:
            assert observer.execute("SELECT count(*) FROM effects").fetchone()[0] == 0
    finally:
        release.set()
        if borrowed is not None:
            if not borrowed.done():
                borrowed.cancel('retire foreign drain observer')
            await await_owned_task(borrowed)
        if owner is not None:
            await await_owned_task(owner)
        await await_owned_task(drain)
        await backend.close()
