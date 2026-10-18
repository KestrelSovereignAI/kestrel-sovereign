"""Two actual close/worker-wait withdrawals survive one finite native failure."""
import asyncio

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import _explicit_control_chain
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.db import sqlite as sqlite_module
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.utils.original_failure import OriginalFailure


@pytest.mark.asyncio
async def test_sqlite_keeps_actual_close_wait_and_worker_wait_withdrawals(native_pg, tmp_path, monkeypatch):
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch))
    database = await AsyncDatabase.sqlite(str(tmp_path / "distinct-wait-withdrawals.db"))
    backend, connection = database.backend, database.backend._connection
    worker = getattr(connection, "_thread", connection)
    close_entered, close_release = asyncio.Event(), asyncio.Event()
    worker_entered, worker_release = asyncio.Event(), asyncio.Event()
    actual_close, actual_wait, actual_worker_wait = connection.close, asyncio.wait, sqlite_module._wait_for_aiosqlite_worker_shutdown
    withdrawals, worker_observations = [], []
    async def held_close():
        await actual_close()
        close_entered.set()
        await close_release.wait()
    async def wait(*args, **kwargs):
        try:
            return await actual_wait(*args, **kwargs)
        except asyncio.CancelledError as error:
            withdrawals.append(error)
            raise
    async def worker_wait(*args, **kwargs):
        result = await actual_worker_wait(*args, **kwargs)
        worker_observations.append(not worker.is_alive())
        if len(worker_observations) == 1:
            worker_entered.set()
            try:
                await worker_release.wait()
            except asyncio.CancelledError as error:
                withdrawals.append(error)
                raise
        return result
    monkeypatch.setattr(connection, "close", held_close)
    monkeypatch.setattr(asyncio, "wait", wait)
    monkeypatch.setattr(sqlite_module, "_wait_for_aiosqlite_worker_shutdown", worker_wait)
    backend._cancelled_write_drain_error = source.error
    owner = create_owned_task(backend.close(), retain_originals=True)
    try:
        await asyncio.wait_for(close_entered.wait(), 5)
        owner.cancel("actual close-task waiter C1")
        await asyncio.sleep(0)
        close_release.set()
        await asyncio.wait_for(worker_entered.wait(), 5)
        owner.cancel("actual worker waiter C2")
        result = await await_owned_task(owner)
        source.assert_retained(result.error)
        assert len(withdrawals) == 2 and withdrawals[0] is not withdrawals[1]
        assert all(error in tuple(_explicit_control_chain(result.error)) for error in withdrawals)
        assert worker_observations == [True, True]
        assert backend._connection is None and not worker.is_alive()
    finally:
        close_release.set()
        worker_release.set()
        await await_owned_task(owner)
        await backend.close()
