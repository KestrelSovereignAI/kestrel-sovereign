"""Actual native cancellation CAS/readback must retain its FIRST receipt."""

import asyncio
import inspect
from uuid import uuid4

import asyncpg
import pytest

from kestrel_sovereign.a2a.stores.unified import (
    ObservabilityStore,
    SessionService,
    TaskStore,
)
from kestrel_sovereign.a2a.task_manager import TaskManager
from kestrel_sovereign.a2a.types import Message, TaskSendParams, TextPart
from kestrel_sovereign.execution_custody import (
    ExecutionCustody,
    ProcessRuntimeExecutionFence,
    execution_commit_outcome,
)
from kestrel_sovereign.storage.db.postgres import PostgresBackend
from tests.integration.test_execution_authority_postgres import GenerationFence
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.unit.test_execution_custody_review43_native import carries


@pytest.mark.asyncio
@pytest.mark.parametrize("later", ["callback", "projection", "allocation"])
async def test_public_cancel_native_unknown_survives_real_reconciliation_and_later_control(
    native_pg, monkeypatch, later
):
    backend, observer, schema = native_pg
    projection_backend = PostgresBackend.from_pool(
        backend._pool, execution_custody=ExecutionCustody(GenerationFence())
    )
    task_store, sessions, observations = (
        TaskStore(backend),
        SessionService(projection_backend),
        ObservabilityStore(backend),
    )
    for store in (task_store, sessions, observations):
        await store.initialize()
    manager = TaskManager(task_store, sessions, observations)
    task = await manager.create_task(
        TaskSendParams(
            id=uuid4().hex,
            sessionId=uuid4().hex,
            message=Message(
                role="user", parts=[TextPart(text="native cancellation receipt")]
            ),
        ),
        agent_name="did:test:original-cancel",
    )
    first, second = (
        OSError("original native cancellation COMMIT ACK lost"),
        OSError("original native projection COMMIT ACK lost"),
    )
    native_exit, commits, callbacks = asyncpg.transaction.Transaction.__aexit__, [], []
    loop, previous, refused = (
        asyncio.get_running_loop(),
        asyncio.get_running_loop().get_task_factory(),
        [],
    )
    foreign = ExecutionCustody(
        ProcessRuntimeExecutionFence("different callback admission")
    )
    foreign.revoke("foreign callback admission retired")

    def callback(_task):
        callbacks.append(_task.id)
        foreign.require_work()

    if later == "callback":
        manager._on_task_cancelled = callback

    async def lose_ack(transaction, error_type, error, traceback):
        result = await native_exit(transaction, error_type, error, traceback)
        if error is None and not transaction._nested:
            commits.append(transaction)
            if len(commits) == 1:
                raise first
            if later == "projection" and len(commits) == 2:
                raise second
        return result

    def refuse_projection(loop, coroutine, **kwargs):
        frame = getattr(coroutine, "cr_frame", None)
        inner = frame.f_locals.get("operation") if frame is not None else None
        original = inner if inspect.iscoroutine(inner) else coroutine
        code = getattr(original, "cr_code", None)
        if (
            code is not None
            and code.co_qualname == "TaskManager._project_status_transition"
        ):
            refused.append((coroutine, original))
            raise RuntimeError("original A2A projection Task refused")
        return (
            previous(loop, coroutine, **kwargs)
            if previous
            else asyncio.Task(coroutine, loop=loop, **kwargs)
        )

    try:
        with monkeypatch.context() as fault:
            fault.setattr(asyncpg.transaction.Transaction, "__aexit__", lose_ack)
            if later == "allocation":
                loop.set_task_factory(refuse_projection)
            with pytest.raises(BaseException) as caught:
                await manager.cancel_task(
                    task.id,
                    reason="original cancellation",
                    agent_name="did:test:original-cancel",
                )
        assert execution_commit_outcome(caught.value) == "unknown"
        assert carries(caught.value, first), (
            "later callback/projection erased the original actual CAS COMMIT failure"
        )
        row = await observer.fetchrow(
            f'SELECT status,cancel_operation_id,canceled_by FROM "{schema}".a2a_tasks WHERE id=$1',
            task.id,
        )
        assert row["status"] == "canceled" and row["cancel_operation_id"]
        assert row["canceled_by"] == "did:test:original-cancel"
        if later == "callback":
            assert callbacks == [task.id], (
                "exact actor/operation readback never reached the original callback"
            )
        elif later == "projection":
            assert len(commits) >= 2 and carries(caught.value, second)
        else:
            assert len(refused) == 1
            assert all(
                inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED
                for pair in refused
                for coro in pair
            ), "unadmitted original A2A projection coroutine leaked"
    finally:
        loop.set_task_factory(previous)
        for pair in refused:
            for coroutine in pair:
                coroutine.close()
        await manager.close()
