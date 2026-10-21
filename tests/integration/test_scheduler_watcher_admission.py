"""Gated watcher admission and reachable original native COMMIT-ACK retirement.

Synchronous watcher refusal cannot follow eager payload execution: canonical
Task admission gates that payload. Test that ordering separately from actual
post-start cancellation/lease loss of a PostgreSQL blocking worker. No Agent
startup, provider, READY, or native transaction completion is simulated.
"""

import asyncio
import inspect
import os
import threading
from contextlib import asynccontextmanager
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


def _original_operation(coroutine):
    frame = getattr(coroutine, "cr_frame", None)
    inner = frame.f_locals.get("operation") if frame is not None else None
    return inner if inspect.iscoroutine(inner) else coroutine


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["preparation", "dispatch"])
@pytest.mark.parametrize(
    "eager",
    [False, pytest.param(True, marks=pytest.mark.skipif(
        not hasattr(asyncio, "eager_task_factory"),
        reason="native eager Tasks require Python 3.12+",
    ))],
)
async def test_refused_lease_watcher_retires_original_gated_effect(
    native_pg, monkeypatch, phase, eager
):
    backend, observer, schema = native_pg
    db = AsyncDatabase(backend)
    did = "did:web:gated-watcher-" + uuid4().hex
    runtime = ExecutionCustody(ProcessRuntimeExecutionFence("gated-watcher-host"))
    effects, parents, entered, refusals, notifications = [], [], [], [], []
    refused = asyncio.Event()
    original = RuntimeError("original Scheduler lease watcher Task refused")
    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()
    tick = None

    async def executor(_name, _args):
        entered.append(asyncio.current_task())
        await db.execute("UPDATE effects SET value = 'must not run' WHERE id = 53")

    class Prepared:
        @asynccontextmanager
        async def prepare_scheduled(self, execution):
            await executor(execution.task_name, execution.args)
            yield lambda: executor(execution.task_name, execution.args)

    def factory(loop, coroutine, **kwargs):
        operation = _original_operation(coroutine)
        code = getattr(operation, "cr_code", None)
        frame = getattr(operation, "cr_frame", None)
        state = scheduler_module._rollout_renewal_state.get()
        if (
            state is not None
            and code is not None
            and code.co_qualname == "Event.wait"
            and frame is not None
            and frame.f_locals.get("self") is state.lost
        ):
            # Identify the ORIGINAL watcher before the gated executor can run;
            # executor-populated metadata is deliberately unavailable here.
            refusals.append((coroutine, operation))
            refused.set()
            raise original
        is_effect = code is not None and code.co_qualname.endswith(
            "._run_dispatch_while_lease_live.<locals>.invoke_dispatch"
        )
        if is_effect:
            parents.append(asyncio.current_task())
            if eager:
                kwargs["eager_start"] = True
            task = asyncio.Task(coroutine, loop=loop, **kwargs)
            effects.append((task, coroutine, operation))
            return task
        return (
            previous(loop, coroutine, **kwargs)
            if previous
            else asyncio.Task(coroutine, loop=loop, **kwargs)
        )

    def on_failure(error):
        notifications.append((error, tuple(task.done() for task, _, _ in effects)))

    await _activate_protocol_for_test_agent(db, did, monkeypatch)
    runner = SchedulerRunner(
        db, did, Prepared() if phase == "preparation" else executor,
        lease_seconds=20, on_protocol_failure=on_failure,
    )
    await runner._ensure_tables()
    await db.execute("INSERT INTO effects VALUES (53, 'before original')")
    await _seed_due(db, uuid4().hex, "gated-watcher-effect", agent_id=did)
    try:
        loop.set_task_factory(factory)
        with bind_execution_custody_snapshot((runtime,)):
            tick = create_owned_task(runner._tick())
        await asyncio.wait_for(refused.wait(), 7)
        loop.set_task_factory(previous)
        await await_owned_task(tick)
        for parent in parents:
            await asyncio.wait_for(await_owned_task(parent), 5)
        assert len(effects) == len(parents) == len(refusals) == 1
        assert not entered, "watcher refusal admitted a gated native effect"
        assert all(task.done() for task, _, _ in effects)
        assert all(
            inspect.getcoroutinestate(operation) == inspect.CORO_CLOSED
            for _, wrapper, payload in effects
            for operation in (wrapper, payload)
        )
        assert all(
            inspect.getcoroutinestate(operation) == inspect.CORO_CLOSED
            for pair in refusals for operation in pair
        )
        assert carries(runner.readiness_failure, original)
        assert notifications and all(
            carries(error, original) and completed == (True,)
            for error, completed in notifications
        ), "scheduler published failure before joining the original gated effect"
        assert await observer.fetchval(
            f'SELECT value FROM "{schema}".effects WHERE id=53'
        ) == "before original"
        assert await observer.fetchval(
            f'SELECT status FROM "{schema}".task_execution_log WHERE agent_id=$1', did
        ) == ("executing" if phase == "preparation" else "claimed")
    finally:
        loop.set_task_factory(previous)
        if tick is not None:
            await await_owned_task(tick)
        for parent in parents:
            await await_owned_task(parent)
        for task, _, _ in effects:
            await await_owned_task(task)
        try:
            await runner.stop()
        except BaseException as error:
            if error is not original and not is_execution_control_error(error):
                raise


@pytest.mark.asyncio
@pytest.mark.parametrize("lost_ack", [False, True])
@pytest.mark.parametrize("phase", ["preparation", "dispatch"])
@pytest.mark.parametrize("control", ["cancel", "lease_loss"])
async def test_original_native_effect_retirement_after_commit(
    native_pg, monkeypatch, lost_ack, phase, control
):
    backend, observer, schema = native_pg
    db = AsyncDatabase(backend)
    did = "did:web:watcher-owner-" + uuid4().hex
    runtime = ExecutionCustody(ProcessRuntimeExecutionFence("original-watcher-host"))
    effects, parents, states, workers = [], [], [], []
    committed, release = asyncio.Event(), threading.Event()
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
            loop.call_soon_threadsafe(committed.set)
            assert await asyncio.to_thread(release.wait, 10), (
                "original test ACK hold expired"
            )
            if lost_ack:
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

    class Prepared:
        @asynccontextmanager
        async def prepare_scheduled(self, execution):
            await executor(execution.task_name, execution.args)

            async def dispatch():
                pytest.fail("withdrawn preparation admitted a new dispatch")

            yield dispatch

    def factory(loop, coroutine, **kwargs):
        operation = _original_operation(coroutine)
        code = getattr(operation, "cr_code", None)
        is_effect = code is not None and code.co_qualname.endswith(
            "._run_dispatch_while_lease_live.<locals>.invoke_dispatch"
        )
        is_worker = code is not None and code.co_qualname == "to_thread"
        if is_effect or is_worker:
            if is_effect:
                parents.append(asyncio.current_task())
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
    runner = SchedulerRunner(
        db, did, Prepared() if phase == "preparation" else executor, lease_seconds=20
    )
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
        await asyncio.wait_for(committed.wait(), 7)
        loop.set_task_factory(previous)
        await await_owned_task(tick)
        assert len(parents) == (
            1 if phase == "preparation" else 2
        )
        assert len(effects) == len(states) == len(workers) == 1, (
            "the scheduled callback must retain its original blocking worker Task"
        )
        assert (
            await observer.fetchval(f'SELECT value FROM "{schema}".effects WHERE id=53')
            == "original committed"
        )
        assert not effects[0].done() and not workers[0].done()
        if control == "cancel":
            parents[-1].cancel("original scheduler observer withdrew after COMMIT")
        else:
            states[0].mark_lost()
        await asyncio.wait({parents[-1]}, timeout=0.1)
        # Assert BEFORE releasing the actual ACK owner or fixture cleanup.
        assert all(not task.done() for task in parents), (
            "original occurrence retired before its accepted native ACK owner"
        )
        assert not effects[0].done() and not workers[0].done()
        release.set()
        parents_outcomes = [
            await asyncio.wait_for(await_owned_task(task), 5) for task in parents
        ]
        effect = await asyncio.wait_for(await_owned_task(effects[0]), 5)
        assert all(task.done() for task in [*parents, *effects, *workers])
        if lost_ack:
            assert execution_commit_outcome(effect.error) == "unknown"
            if phase == "dispatch":
                assert execution_commit_outcome(parents_outcomes[-1].error) == "unknown"
            assert carries(runner.readiness_failure, native_failure)
            assert execution_commit_outcome(runner.readiness_failure) == "unknown"
        else:
            assert effect.cancellation is not None or effect.error is not None
        assert (
            await observer.fetchval(
                f'SELECT status FROM "{schema}".task_execution_log WHERE agent_id=$1',
                did,
            )
            == "executing"
        )
    finally:
        loop.set_task_factory(previous)
        release.set()
        if tick is not None:
            await await_owned_task(tick)
        for task in {*parents, *effects, *workers}:
            await await_owned_task(task)
        try:
            await runner.stop()
        except BaseException as error:
            if not is_execution_control_error(error):
                raise
