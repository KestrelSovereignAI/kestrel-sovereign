"""Actual server phases preserve native body control and unstarted resources."""

import asyncio
import inspect

import pytest
from fastapi import FastAPI

from kestrel_sovereign.execution_custody import (
    ExecutionCustody,
    execution_cleanup_debts,
    execution_commit_outcome,
    is_execution_control_error,
)
from kestrel_sovereign.host_features import build_host_context, close_host_context
from kestrel_sovereign.storage.async_database import AsyncDatabase
from tests.integration.test_execution_authority_postgres import GenerationFence
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_execution_custody_review56_pg import lose_native_ack
from tests.unit.test_host_context_native_cleanup import _actual_operation


@pytest.mark.asyncio
@pytest.mark.parametrize("unknown", [False, True])
async def test_refused_server_phase_keeps_actual_resources_and_drains_later_phases(
    native_pg, tmp_path, monkeypatch, unknown
):
    from kestrel_sovereign.server import _lifespan_teardown_owner

    backend, observer, schema = native_pg
    monkeypatch.setenv("KESTREL_DB_BACKEND", "sqlite")
    monkeypatch.setenv("KESTREL_HOLD_BACKEND", "sqlite")
    context = await build_host_context(db_path=str(tmp_path / "original-host.db"))
    receipts = await AsyncDatabase.sqlite(str(tmp_path / "independent-receipts.db"))
    await receipts.execute("CREATE TABLE original_receipts (id INTEGER)")
    await receipts.execute("INSERT INTO original_receipts VALUES (1)")
    host_connection, receipt_connection = (
        context.db.backend._connection,
        receipts.backend._connection,
    )
    host_worker, receipt_worker = (
        getattr(connection, "_thread", connection)
        for connection in (host_connection, receipt_connection)
    )
    app = FastAPI()
    app.state.host_context = context
    app.state.host_features = []
    app.state.stop_receipt_db = receipts
    loop, previous, refused = (
        asyncio.get_running_loop(),
        asyncio.get_running_loop().get_task_factory(),
        [],
    )
    allocation_error = RuntimeError("original host-feature phase allocation refused")

    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        code = getattr(operation, "cr_code", None)
        if (
            not refused
            and code is not None
            and code.co_qualname == "_shutdown_host_features"
        ):
            refused.append((coroutine, operation))
            raise allocation_error
        return (
            previous(loop, coroutine, **kwargs)
            if previous
            else asyncio.Task(coroutine, loop=loop, **kwargs)
        )

    native_ack = []
    if unknown:
        backend._execution_custody = ExecutionCustody(GenerationFence())
        native_ack = lose_native_ack(
            monkeypatch,
            boundary="commit",
            original=OSError("actual original body COMMIT ACK lost"),
        )
    caught = None
    try:
        loop.set_task_factory(factory)
        try:
            async with _lifespan_teardown_owner(app):
                await backend.execute(
                    "INSERT INTO effects VALUES (?, ?)", (129, "original lifespan body")
                )
        except BaseException as error:
            caught = error
        loop.set_task_factory(previous)
        assert refused and caught is not None
        assert is_execution_control_error(caught), (
            "phase refusal lost actual unstarted native retirement debt"
        )
        if unknown:
            assert (
                native_ack == [True] and execution_commit_outcome(caught) == "unknown"
            ), "phase refusal replaced original native COMMIT control"
        assert (
            await observer.fetchval(
                f'SELECT COUNT(*) FROM "{schema}".effects WHERE id=129'
            )
            == 1
        )
        assert any(
            resource is context.db or resource is context.db.backend
            for resource, _ in execution_cleanup_debts(caught)
        )
        assert app.state.host_context is context and host_worker.is_alive()
        assert (
            app.state.stop_receipt_db is None
            and receipts.backend._connection is None
            and not receipt_worker.is_alive()
        ), "unadmitted phase skipped an independent original database closer"
        assert all(
            inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED
            for pair in refused
            for coro in pair
        ), "refused phase coroutine leaked"
    finally:
        loop.set_task_factory(previous)
        backend._execution_custody = None
        for pair in refused:
            for coroutine in pair:
                coroutine.close()
        await receipts.close()
        await close_host_context(context)
