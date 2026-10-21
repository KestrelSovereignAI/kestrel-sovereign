"""A real stopped SQLite worker must not mutate an earlier native receipt."""
import asyncio

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import _explicit_control_chain
from kestrel_sovereign.storage.async_database import AsyncDatabase
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_execution_authority_postgres import native_pg as _cleanup_pg
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.utils.original_failure import OriginalFailure

cleanup_pg = _cleanup_pg


@pytest.mark.asyncio
@pytest.mark.parametrize("source_cancel", [False, True])
@pytest.mark.parametrize("native_cleanup", [False, True])
async def test_sqlite_intermediate_close_keeps_original_source_graph(
    native_pg, cleanup_pg, tmp_path, monkeypatch, source_cancel, native_cleanup,
):
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    database = await AsyncDatabase.sqlite(str(tmp_path / "intermediate-source.db"))
    backend, conn = database.backend, database.backend._connection
    worker = getattr(conn, "_thread", conn)
    actual_close, physical = conn.close, []
    cleanup = OriginalFailure.capture(await _original_control(cleanup_pg[0], monkeypatch)) if native_cleanup else None
    independent = cleanup.error if cleanup else OSError("independent failure after actual SQLite close")
    async def close():
        physical.append(asyncio.current_task())
        await actual_close()
        raise independent
    try:
        backend._cancelled_write_drain_error = source.error
        with monkeypatch.context() as patch:
            patch.setattr(conn, "close", close)
            outcome = await await_owned_task(create_owned_task(backend.close(), retain_originals=True))
        assert len(physical) == 1 and physical[0].done()
        assert backend._connection is None and not worker.is_alive()
        source.assert_retained(outcome.error)
        if cleanup:
            cleanup.assert_retained(outcome.error)
        assert independent in tuple(_explicit_control_chain(outcome.error))
    finally:
        await backend.close()
