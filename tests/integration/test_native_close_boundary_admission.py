"""Original native close Tasks survive publication at actual lifecycle boundaries.

The stream fixture owns a real SQLite worker but does not claim provider/SSE
network acceptance. The preflight case exercises its raw-close boundary, not
the complete role-validation protocol. No Agent READY or paid inference.
"""
import asyncio
import os

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.endpoints.closing_streaming_response import ClosingStreamingResponse
from kestrel_sovereign.execution_custody import close_execution_stream, execution_cleanup_debts
from kestrel_sovereign.hold.state import _close_postgres_preflight_backends
from kestrel_sovereign.llm.cancellation import await_or_cancelled
from kestrel_sovereign.storage.async_database import AsyncDatabase, _close_owned_database
from kestrel_sovereign.storage.db.postgres import PostgresBackend
from tests.integration.test_execution_authority_postgres import native_pg as _native_pg
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.utils.native_admission_fault import NativeAdmissionFault
from tests.utils.original_failure import OriginalFailure

native_pg = _native_pg


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["refuse", "publish-before", "publish-after"])
@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("source_cancel", [False, True])
@pytest.mark.parametrize("boundary", ["database-owner", "db-dispose", "db-backend", "postgres-pool", "preflight", "response", "forwarder", "abandoned-result"])
async def test_native_boundary_keeps_exact_accepted_closer(
    native_pg, tmp_path, monkeypatch, mode, eager, source_cancel, boundary,
):
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    db = await AsyncDatabase.sqlite(str(tmp_path / "native-boundary.db"))
    connection = db.backend._connection
    worker = getattr(connection, "_thread", connection)
    pg = None

    class Stream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

        async def aclose(self):
            await db.close()

    stream = Stream()
    resource = db
    if boundary == "postgres-pool":
        pg = PostgresBackend(dsn=os.environ["TEST_POSTGRES_URL"], min_pool_size=0, max_pool_size=1)
        await pg.connect()
        assert await pg.fetch_val("SELECT 1") == 1
        resource = pg._pool
        operation = pg.close()
        target = resource.close.__code__
        selected_self = resource
    elif boundary == "preflight":
        operation = _close_postgres_preflight_backends(db.backend)
        resource, target, selected_self = db.backend, type(db.backend).close.__code__, db.backend
    elif boundary == "database-owner":
        operation = _close_owned_database(db)
        target, selected_self = AsyncDatabase.close.__code__, db
    elif boundary in {"db-dispose", "db-backend"}:
        operation = db.close()
        target = (AsyncDatabase.dispose_cached_sqla_factory.__code__ if boundary == "db-dispose" else type(db.backend).close.__code__)
        selected_self = db if boundary == "db-dispose" else db.backend
    elif boundary == "forwarder":
        operation = close_execution_stream(object(), stream)
        resource, target, selected_self = stream, Stream.aclose.__code__, stream
    elif boundary == "response":
        async def send(_message):
            return None

        async def receive():
            return {"type": "http.disconnect"}

        response = ClosingStreamingResponse(stream)
        operation = response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
        resource, target, selected_self = stream, Stream.aclose.__code__, stream
    else:
        # The real await helper receives an already-acquired native resource.
        # A pre-cancelled request must close that exact abandoned result.
        acquired = asyncio.get_running_loop().create_future()
        acquired.set_result(stream)
        operation = await_or_cancelled(acquired, lambda: True, abandoned_result_cleanup=lambda value: value.aclose())
        resource, target, selected_self = stream, Stream.aclose.__code__, stream

    def selected(coroutine):
        frame = getattr(coroutine, "cr_frame", None)
        return getattr(coroutine, "cr_code", None) is target and frame is not None and frame.f_locals.get("self") is selected_self

    fault = NativeAdmissionFault(selected, source.error, mode, eager=eager)
    owner = create_owned_task(operation, retain_originals=True)
    try:
        with fault:
            result = await await_owned_task(owner)
        fault.assert_terminal_unstarted()
        # The preflight helper returns its joined errors for its real caller.
        terminal = result.result[0] if boundary == "preflight" and result.error is None else result.error
        source.assert_retained(terminal)
        assert any(value is resource and task is fault.task for value, task in execution_cleanup_debts(terminal))
        if boundary == "db-dispose":
            assert db.backend._connection is None and not worker.is_alive(), "independent primary close skipped"
        elif boundary == "postgres-pool":
            assert pg._pool is resource and not resource._closed
            before = tuple(fault.accepted)
            repeated = await await_owned_task(create_owned_task(pg.close(), retain_originals=True))
            source.assert_retained(repeated.error)
            assert tuple(fault.accepted) == before and pg._pool is resource
        else:
            assert db.backend._connection is connection and worker.is_alive()
    finally:
        await await_owned_task(owner)
        for task in fault.accepted:
            await await_owned_task(task)
        if pg is not None:
            # Explicit fixture recovery of the retained native pool, not a
            # production retry or a synthetic successful close receipt.
            pool = pg._pool
            if pool is not None:
                await pool.close()
                pg._pool = None
        await db.close()
