"""Watcher refusal must join its original real native COMMIT-ACK owner.

The actual scheduled callback runs an original blocking worker with its copied
custody. Its private PostgreSQL event loop uses the real guarded backend and
native transaction exit. No Agent startup, provider or READY is simulated.
"""

import asyncio
import inspect
import os
import threading
from contextvars import ContextVar
from uuid import uuid4

import asyncpg
import pytest

from kestrel_sovereign._async_ownership import (
    await_owned_task,
    create_owned_task,
    run_blocking_operation,
)
from kestrel_sovereign.execution_custody import (
    ExecutionCustody,
    ProcessRuntimeExecutionFence,
    bind_execution_custody_snapshot,
    execution_commit_outcome,
    is_execution_control_error,
)
from kestrel_sovereign.features.scheduler import runner as scheduler_module
from kestrel_sovereign.features.scheduler.runner import SchedulerRunner
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.db.postgres import PostgresBackend
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_scheduler_postgres_claims import (
    _activate_protocol_for_test_agent,
)
from tests.unit.test_execution_custody_review43_native import carries
from tests.unit.test_scheduler_occurrence_liveness import _seed_due


@pytest.mark.asyncio
@pytest.mark.skipif(
    not hasattr(asyncio, "eager_task_factory"),
    reason="native eager Tasks require Python 3.12+",
)
async def test_refused_lease_watcher_joins_original_native_effect(
    native_pg, monkeypatch
):
    backend, observer, schema = native_pg
    db = AsyncDatabase(backend)
    did = "did:web:watcher-owner-" + uuid4().hex
    runtime = ExecutionCustody(ProcessRuntimeExecutionFence("original-watcher-host"))
    effects, parents, states, workers, refusals = [], [], [], [], []
    refused, committed, release = asyncio.Event(), threading.Event(), threading.Event()
    original = RuntimeError("original Scheduler lease watcher Task refused")
    native_failure = OSError("actual original scheduled COMMIT ACK lost")
    # Test routing metadata only; this flag confers no authority.
    worker_commit = ContextVar("actual-original-worker-commit", default=False)
    native_exit = asyncpg.transaction.Transaction.__aexit__
    loop, previous = (
        asyncio.get_running_loop(),
        asyncio.get_running_loop().get_task_factory(),
    )
    tick = None

    async def withheld_native_ack(transaction, error_type, error, tb):
        result = await native_exit(transaction, error_type, error, tb)
        if worker_commit.get() and error_type is None and not transaction._nested:
            committed.set()
            assert await asyncio.to_thread(release.wait, 10), (
                "original test ACK hold expired"
            )
            raise native_failure
        return result

    async def native_effect():
        pool = await asyncpg.create_pool(
            os.environ["TEST_POSTGRES_URL"],
            min_size=1,
            max_size=1,
            server_settings={"search_path": schema},
        )
        private = PostgresBackend.from_pool(pool)
        token = worker_commit.set(True)
        try:
            await private.execute(
                "UPDATE effects SET value = ? WHERE id = ?", ("original committed", 53)
            )
        finally:
            worker_commit.reset(token)
            await private.close()
            await pool.close()

    async def executor(_name, _args):
        effects.append(asyncio.current_task())
        states.append(scheduler_module._rollout_renewal_state.get())
        return await run_blocking_operation(lambda: asyncio.run(native_effect()))

    def original_operation(coroutine):
        frame = getattr(coroutine, "cr_frame", None)
        inner = frame.f_locals.get("operation") if frame is not None else None
        return inner if inspect.iscoroutine(inner) else coroutine

    def factory(loop, coroutine, **kwargs):
        operation = original_operation(coroutine)
        code = getattr(operation, "cr_code", None)
        frame = getattr(operation, "cr_frame", None)
        if (
            states
            and code is not None
            and code.co_qualname == "Event.wait"
            and frame is not None
            and frame.f_locals.get("self") is states[0].lost
        ):
            # The native worker/event loop can progress while this bounded
            # synchronous observer waits. The flag is set ONLY AFTER real
            # PostgreSQL COMMIT returned, before its original ACK can retire.
            assert committed.wait(5), "original native effect never reached COMMIT"
            refusals.append((coroutine, operation))
            refused.set()
            raise original
        is_effect = code is not None and code.co_qualname.endswith(
            "._run_dispatch_while_lease_live.<locals>.invoke_dispatch"
        )
        is_worker = code is not None and code.co_qualname == "to_thread"
        if is_effect or is_worker:
            if is_effect:
                parents.append(asyncio.current_task())
            kwargs["eager_start"] = True
            task = asyncio.Task(coroutine, loop=loop, **kwargs)
            if is_worker:
                workers.append(task)
            return task
        return (
            previous(loop, coroutine, **kwargs)
            if previous
            else asyncio.Task(coroutine, loop=loop, **kwargs)
        )

    await _activate_protocol_for_test_agent(db, did, monkeypatch)
    runner = SchedulerRunner(db, did, executor, lease_seconds=20)
    await runner._ensure_tables()
    await db.execute("INSERT INTO effects VALUES (53, 'before original')")
    await _seed_due(db, uuid4().hex, "original-watcher-effect", agent_id=did)
    monkeypatch.setattr(
        asyncpg.transaction.Transaction, "__aexit__", withheld_native_ack
    )
    try:
        loop.set_task_factory(factory)
        with bind_execution_custody_snapshot((runtime,)):
            tick = create_owned_task(runner._tick())
        await asyncio.wait_for(refused.wait(), 7)
        loop.set_task_factory(previous)
        await await_owned_task(tick)
        assert committed.is_set() and len(parents) == 2
        assert len(effects) == len(states) == len(workers) == 1
        assert (
            await observer.fetchval(f'SELECT value FROM "{schema}".effects WHERE id=53')
            == "original committed"
        )
        assert not effects[0].done() and not workers[0].done()
        await asyncio.wait({parents[-1]}, timeout=0.1)
        # Assert BEFORE releasing the actual ACK owner or fixture cleanup.
        assert all(not task.done() for task in parents), (
            "watcher refusal let original occurrence retire before its accepted native ACK owner"
        )
        release.set()
        parents_outcomes = [
            await asyncio.wait_for(await_owned_task(task), 5) for task in parents
        ]
        effect = await asyncio.wait_for(await_owned_task(effects[0]), 5)
        assert all(task.done() for task in [*parents, *effects, *workers])
        assert execution_commit_outcome(effect.error) == "unknown"
        assert execution_commit_outcome(parents_outcomes[-1].error) == "unknown"
        assert carries(parents_outcomes[-1].error, original)
        assert carries(parents_outcomes[-1].error, native_failure)
        assert runner.readiness_failure is not None
        assert execution_commit_outcome(runner.readiness_failure) == "unknown"
        assert all(
            inspect.getcoroutinestate(op) == inspect.CORO_CLOSED
            for pair in refusals
            for op in pair
        )
    finally:
        loop.set_task_factory(previous)
        release.set()
        if tick is not None:
            await await_owned_task(tick)
        for task in {*parents, *effects, *workers}:
            await await_owned_task(task)
        for pair in refusals:
            for operation in pair:
                operation.close()
        try:
            await runner.stop()
        except BaseException as error:
            if not is_execution_control_error(error):
                raise
