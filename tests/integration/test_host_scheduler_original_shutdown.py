"""Original Scheduler control survives independent native storage cleanup."""

import asyncio
import inspect

import pytest
from fastapi import FastAPI

from kestrel_sovereign.execution_custody import (
    ExecutionCustody,
    execution_commit_outcome,
    is_execution_control_error,
)
from kestrel_sovereign.features.scheduler.runner import (
    AgentManagerHostedSchedulerExecutor,
    SchedulerRunner,
)
from kestrel_sovereign.multi_agent.agent_manager import AgentManager
from kestrel_sovereign.storage.async_database import AsyncDatabase
from tests.integration.test_execution_authority_postgres import GenerationFence
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_execution_custody_review56_pg import lose_native_ack
from tests.unit.test_host_context_native_cleanup import _actual_operation


@pytest.mark.asyncio
@pytest.mark.parametrize("refuse_storage", [False, True])
async def test_original_scheduler_stop_and_failed_storage_close_preserve_both_owners(
    native_pg, tmp_path, monkeypatch, refuse_storage
):
    from kestrel_sovereign.server import _shutdown_host_scheduler

    backend, observer, schema = native_pg
    db, failures, signalled = AsyncDatabase(backend), [], asyncio.Event()

    def failed(error):
        failures.append(error)
        signalled.set()

    manager = AgentManager(base_data_dir=tmp_path / "empty-actual-manager")
    runner = SchedulerRunner(
        db=db,
        agent_id=None,
        executor=AgentManagerHostedSchedulerExecutor(manager),
        authorized_agent_ids=("did:test:original-f33",),
        on_protocol_failure=failed,
    )
    app = FastAPI()
    app.state.host_scheduler_runner = runner
    app.state.host_scheduler_storage = db
    loop, previous, refused = (
        asyncio.get_running_loop(),
        asyncio.get_running_loop().get_task_factory(),
        [],
    )
    close_error = RuntimeError("original native scheduler pool closer refused")

    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        code, frame = (
            getattr(operation, "cr_code", None),
            getattr(operation, "cr_frame", None),
        )
        if (
            refuse_storage
            and not refused
            and code is not None
            and code.co_qualname == "PostgresBackend.close"
            and frame.f_locals.get("self") is backend
        ):
            refused.append((coroutine, operation))
            raise close_error
        return (
            previous(loop, coroutine, **kwargs)
            if previous
            else asyncio.Task(coroutine, loop=loop, **kwargs)
        )

    try:
        await runner.start()
        backend._execution_custody = ExecutionCustody(GenerationFence())
        receipts = lose_native_ack(
            monkeypatch,
            boundary="commit",
            original=OSError("actual original scheduler COMMIT ACK lost"),
        )
        await asyncio.wait_for(signalled.wait(), 5)
        assert receipts == [True] and any(
            execution_commit_outcome(error) == "unknown" for error in failures
        ), "actual scheduler did not retain a native COMMIT failure"
        loop.set_task_factory(factory)
        caught = None
        try:
            await _shutdown_host_scheduler(app)
        except BaseException as error:
            caught = error
        loop.set_task_factory(previous)
        assert bool(refused) is refuse_storage and caught is not None
        assert execution_commit_outcome(caught) == "unknown", (
            "storage retirement replaced actual original Scheduler COMMIT control"
        )
        assert app.state.host_scheduler_runner is runner, (
            "failed original Scheduler owner was discarded"
        )
        if refuse_storage:
            assert app.state.host_scheduler_storage is db
        else:
            assert app.state.host_scheduler_storage is None and not backend.is_connected
        assert all(
            inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED
            for pair in refused
            for coro in pair
        )
    finally:
        loop.set_task_factory(previous)
        monkeypatch.undo()
        backend._execution_custody = None
        for pair in refused:
            for coroutine in pair:
                coroutine.close()
        try:
            await runner.stop()
        except BaseException as error:
            if not is_execution_control_error(error):
                raise
        await backend.close()
