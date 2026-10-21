"""Stop startup reports the actual cleanup withdrawal after native opening.

Real PG source and actual SQLite connection/worker close, no live host mutation.
"""
import asyncio

import pytest
from fastapi import FastAPI
from unittest.mock import AsyncMock

from kestrel_sovereign import server, _async_ownership as ownership
from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import _explicit_control_chain
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.stop import StopReceiptStore
from tests.integration.test_execution_authority_postgres import native_pg as _native_pg
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.utils.original_failure import OriginalFailure

native_pg = _native_pg


@pytest.mark.asyncio
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_stop_startup_native_opening_retains_actual_successful_cleanup_withdrawal(
    native_pg, monkeypatch, tmp_path, source_cancel,
):
    from kestrel_sovereign.host_features import storage as host_storage
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    path = tmp_path / "owned-stop.db"
    database = await AsyncDatabase.sqlite(str(path))
    connection = database.backend._connection
    worker = getattr(connection, "_thread", connection)
    actual_close, actual_join = database.close, ownership.await_owned_task
    entered, release, withdrawals = asyncio.Event(), asyncio.Event(), []
    owner = None
    async def schema(self):
        await database.execute("CREATE TABLE actual_stop_fixture (id INTEGER)")
        raise source.error
    async def close():
        entered.set()
        await release.wait()
        await actual_close()
    async def observe(task, *args, **kwargs):
        result = await actual_join(task, *args, **kwargs)
        if asyncio.current_task() is owner and result.cancellation is not None:
            withdrawals.append(result.cancellation)
        return result
    monkeypatch.setenv("KESTREL_DB_BACKEND", "sqlite")
    monkeypatch.setattr(host_storage, "prepare_host_database", lambda: path)
    monkeypatch.setattr(host_storage, "validate_sqlite_family_private", lambda path: None)
    monkeypatch.setattr(AsyncDatabase, "sqlite", AsyncMock(return_value=database))
    monkeypatch.setattr(StopReceiptStore, "ensure_schema", schema)
    monkeypatch.setattr(database, "close", close)
    monkeypatch.setattr(ownership, "await_owned_task", observe)
    owner = create_owned_task(server._initialize_stop_receipts(FastAPI()), retain_originals=True)
    try:
        await asyncio.wait_for(entered.wait(), 5)
        owner.cancel("actual Stop startup cleanup observer withdrew")
        await asyncio.sleep(0)
        assert database.backend._connection is connection and worker.is_alive() and not owner.done()
        release.set()
        result = await await_owned_task(owner)
        source.assert_retained(result.error)
        assert withdrawals and all(error in tuple(_explicit_control_chain(result.error)) for error in withdrawals)
        assert database.backend._connection is None and not worker.is_alive()
    finally:
        release.set()
        await await_owned_task(owner)
        await actual_close()
