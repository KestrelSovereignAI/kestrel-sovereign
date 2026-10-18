"""Original Agent-owned Hold resources close without an extra collector Task."""

import asyncio
import inspect

import pytest

from kestrel_sovereign.execution_custody import (
    execution_cleanup_debts,
    is_execution_control_error,
)
from kestrel_sovereign.hold import build_bound_host_context
from kestrel_sovereign.host_features import close_host_context
from tests.unit.test_cleanup_task_admission import _partial_native_agent
from tests.unit.test_host_context_native_cleanup import _actual_operation


@pytest.mark.asyncio
@pytest.mark.parametrize("refusal", ["outer", "native"])
async def test_agent_shutdown_retains_original_standalone_context(
    tmp_path, monkeypatch, refusal
):
    monkeypatch.setenv("KESTREL_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("KESTREL_HOST_DB_PATH", str(tmp_path / "original-host.db"))
    monkeypatch.setenv("KESTREL_DB_BACKEND", "sqlite")
    monkeypatch.setenv("KESTREL_HOLD_BACKEND", "sqlite")
    agent, db = await _partial_native_agent(tmp_path, monkeypatch)
    context = await build_bound_host_context(
        agent, agent_data_root=tmp_path / "agent-root"
    )
    agent._standalone_hold_context = context
    connection = context.hold_db.backend._connection
    worker = getattr(connection, "_thread", connection)
    loop, previous, refused = (
        asyncio.get_running_loop(),
        asyncio.get_running_loop().get_task_factory(),
        [],
    )
    original = RuntimeError("original standalone context closer refused")
    selected = (
        "close_bound_host_context" if refusal == "outer" else "SQLiteBackend.close"
    )

    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        frame, code = (
            getattr(operation, "cr_frame", None),
            getattr(operation, "cr_code", None),
        )
        backend = frame.f_locals.get("self") if frame else None
        if (
            not refused
            and code is not None
            and code.co_qualname == selected
            and (refusal == "outer" or backend is context.hold_db.backend)
        ):
            refused.append((coroutine, operation))
            raise original
        return (
            previous(loop, coroutine, **kwargs)
            if previous
            else asyncio.Task(coroutine, loop=loop, **kwargs)
        )

    try:
        loop.set_task_factory(factory)
        try:
            cancelled, failure = await agent._close_standalone_hold_context()
        except BaseException as error:
            cancelled, failure = False, error
        loop.set_task_factory(previous)
        if refusal == "outer":
            assert not refused, (
                "Agent created an unnecessary all-resource collector Task"
            )
            assert not cancelled and failure is None
            assert agent._standalone_hold_context is None
            assert not worker.is_alive() and context.hold_db.backend._connection is None
        else:
            assert (
                refused and failure is not None and is_execution_control_error(failure)
            )
            assert agent._standalone_hold_context is context and worker.is_alive()
            assert any(
                resource is context.hold_db.backend or resource is context.hold_db
                for resource, _ in execution_cleanup_debts(failure)
            )
            _, repeated = await agent._close_standalone_hold_context()
            assert repeated is failure and len(refused) == 1
        assert all(
            inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED
            for pair in refused
            for coro in pair
        )
        assert not agent._runtime_publication_ready.is_set()
    finally:
        loop.set_task_factory(previous)
        for pair in refused:
            for coroutine in pair:
                coroutine.close()
        await context.hold_db.close()
        try:
            await close_host_context(context)
        except BaseException as error:
            if not is_execution_control_error(error):
                raise
        agent._standalone_hold_context = None
        await agent.shutdown()
        await db.close()


@pytest.mark.asyncio
async def test_agent_original_context_close_joins_actual_ack_through_withdrawal(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("KESTREL_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("KESTREL_HOST_DB_PATH", str(tmp_path / "original-host.db"))
    monkeypatch.setenv("KESTREL_DB_BACKEND", "sqlite")
    monkeypatch.setenv("KESTREL_HOLD_BACKEND", "sqlite")
    agent, db = await _partial_native_agent(tmp_path, monkeypatch)
    context = await build_bound_host_context(
        agent, agent_data_root=tmp_path / "agent-root"
    )
    agent._standalone_hold_context = context
    connection = context.hold_db.backend._connection
    worker = getattr(connection, "_thread", connection)
    native_close = connection.close
    entered, release = asyncio.Event(), asyncio.Event()

    async def held_ack():
        await native_close()
        entered.set()
        await release.wait()

    monkeypatch.setattr(connection, "close", held_ack)
    shutdown = asyncio.create_task(agent._close_standalone_hold_context())
    try:
        await asyncio.wait_for(entered.wait(), 5)
        shutdown.cancel("first original caller withdrawal")
        await asyncio.sleep(0)
        shutdown.cancel("second original caller withdrawal")
        await asyncio.sleep(0)
        assert not shutdown.done() and agent._standalone_hold_context is context
        release.set()
        cancelled, failure = await asyncio.wait_for(shutdown, 5)
        assert cancelled and failure is None
        assert agent._standalone_hold_context is None and not worker.is_alive()
        assert not agent._runtime_publication_ready.is_set()
    finally:
        release.set()
        if not shutdown.done():
            await shutdown
        await context.hold_db.close()
        await close_host_context(context)
        agent._standalone_hold_context = None
        await agent.shutdown()
        await db.close()
