"""Original shutdown dependencies remain live; independent native owners retire."""

import asyncio
import inspect
from contextlib import asynccontextmanager

import pytest
from tests.utils.original_failure import OriginalFailure
from fastapi import FastAPI

from kestrel_sovereign._async_ownership import await_owned_task
from kestrel_sovereign.execution_custody import (
    _explicit_control_chain,
    execution_cleanup_debts,
    is_execution_control_error,
)
from kestrel_sovereign.hold import build_bound_host_context
from kestrel_sovereign.host_features import close_host_context
from kestrel_sovereign.phoenix_supervisor import PhoenixSupervisor
from kestrel_sovereign.storage.async_database import AsyncDatabase
from tests.unit.test_cleanup_task_admission import _partial_native_agent
from tests.unit.test_host_context_native_cleanup import _actual_operation


@asynccontextmanager
async def native_phoenix_proxy(tmp_path):
    peers, closed = set(), asyncio.Event()

    async def peer(reader, writer):
        task = asyncio.current_task()
        peers.add(task)
        try:
            await reader.readuntil(b"\r\n\r\n")
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
            await writer.drain()
            await reader.read()
        finally:
            writer.close()
            await writer.wait_closed()
            closed.set()
            peers.discard(task)

    listener = await asyncio.start_server(peer, "127.0.0.1", 0)
    supervisor = PhoenixSupervisor(
        working_dir=tmp_path / "private-never-started-phoenix"
    )
    assert supervisor.process is None and supervisor._adopted_pid is None
    client = supervisor._proxy_client()
    try:
        assert (
            await client.get(
                f"http://127.0.0.1:{listener.sockets[0].getsockname()[1]}/"
            )
        ).text == "ok"
        assert not closed.is_set()
        yield supervisor, client, closed
    finally:
        # No Phoenix process is started, adopted or terminated by this fixture.
        await client.aclose()
        listener.close()
        await listener.wait_closed()
        for task in tuple(peers):
            await await_owned_task(task)


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["agents", "host-scheduler"])
async def test_refused_prerequisite_keeps_actual_hold_dependencies_while_independent_proxy_closes(
    tmp_path, monkeypatch, phase
):
    from kestrel_sovereign.server import _lifespan_teardown_owner

    monkeypatch.setenv("KESTREL_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("KESTREL_HOST_DB_PATH", str(tmp_path / "original-host.db"))
    monkeypatch.setenv("KESTREL_DB_BACKEND", "sqlite")
    monkeypatch.setenv("KESTREL_HOLD_BACKEND", "sqlite")
    agent, db = await _partial_native_agent(tmp_path, monkeypatch)
    context = await build_bound_host_context(
        agent, agent_data_root=tmp_path / "agent-root"
    )
    await context.hold_store.get_effective(agent.did)
    connection = context.hold_db.backend._connection
    worker = getattr(connection, "_thread", connection)
    receipts = await AsyncDatabase.sqlite(str(tmp_path / "dependent-receipts.db"))
    receipt_connection = receipts.backend._connection
    app = FastAPI()
    app.state.agent = agent
    app.state.host_context = context
    app.state.host_features = []
    app.state.stop_receipt_db = receipts
    runner, scheduler_db = None, None
    if phase == "host-scheduler":
        from kestrel_sovereign.features.scheduler.runner import (
            AgentManagerHostedSchedulerExecutor,
            SchedulerRunner,
        )
        from kestrel_sovereign.multi_agent.agent_manager import AgentManager

        scheduler_db = await AsyncDatabase.sqlite(
            str(tmp_path / "original-scheduler.db")
        )
        manager = AgentManager(base_data_dir=tmp_path / "empty-hosted-fleet")
        runner = SchedulerRunner(
            db=scheduler_db,
            agent_id=None,
            executor=AgentManagerHostedSchedulerExecutor(manager),
            authorized_agent_ids=(agent.did,),
        )
        # Real protocol preparation and polling Tasks; no schedules, cold
        # agent resolution, cognition, substituted executor or READY result.
        app.state.host_scheduler_runner = runner
        app.state.host_scheduler_storage = scheduler_db
    loop, previous, refused = (
        asyncio.get_running_loop(),
        asyncio.get_running_loop().get_task_factory(),
        [],
    )
    refusal = RuntimeError("original Agent shutdown phase admission refused")

    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        code = getattr(operation, "cr_code", None)
        target = (
            "_shutdown_server_agents"
            if phase == "agents"
            else "_shutdown_host_scheduler"
        )
        if not refused and code is not None and code.co_qualname == target:
            refused.append((coroutine, operation))
            raise refusal
        return (
            previous(loop, coroutine, **kwargs)
            if previous
            else asyncio.Task(coroutine, loop=loop, **kwargs)
        )

    try:
        if runner is not None:
            await runner.start()
            assert runner._running and not runner._task.done()
        async with native_phoenix_proxy(tmp_path) as (supervisor, client, closed):
            app.state.phoenix = supervisor
            loop.set_task_factory(factory)
            with pytest.raises(BaseException) as caught:
                async with _lifespan_teardown_owner(app):
                    pass
            loop.set_task_factory(previous)
            assert refusal in tuple(_explicit_control_chain(caught.value)) and is_execution_control_error(caught.value)
            assert tuple(_explicit_control_chain(refusal)) == (refusal,)
            assert app.state.agent is agent and db.backend._connection is not None
            if runner is not None:
                assert app.state.host_scheduler_runner is runner and runner._running
                assert (
                    not runner._task.done()
                    and scheduler_db.backend._connection is not None
                )
            assert (
                context.hold_db.backend._connection is connection and worker.is_alive()
            ), (
                "refused Agent shutdown nevertheless closed its live original Hold dependency"
            )
            await context.hold_store.get_effective(agent.did)
            assert (
                receipts.backend._connection is receipt_connection
                and app.state.stop_receipt_db is receipts
            )
            assert any(
                resource is context.hold_db or resource is context.hold_db.backend
                for resource, _ in execution_cleanup_debts(caught.value)
            )
            await asyncio.wait_for(closed.wait(), 2)
            assert client.is_closed and app.state.phoenix is None, (
                "blocked dependent phases skipped an independent original proxy close"
            )
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
        if runner is not None:
            await runner.stop()
            await scheduler_db.close()
        await agent.shutdown()
        await db.close()
        await receipts.close()
        await close_host_context(context)


@pytest.mark.asyncio
async def test_refused_inner_phoenix_close_keeps_actual_proxy_and_debt(tmp_path):
    from kestrel_sovereign.server import _lifespan_teardown_owner

    loop, previous, refused = (
        asyncio.get_running_loop(),
        asyncio.get_running_loop().get_task_factory(),
        [],
    )
    refusal = RuntimeError("original Phoenix proxy closer admission refused")
    source = OriginalFailure.capture(refusal)

    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        code = getattr(operation, "cr_code", None)
        if (
            not refused
            and code is not None
            and code.co_qualname == "PhoenixSupervisor.aclose"
        ):
            refused.append((coroutine, operation))
            raise refusal
        return (
            previous(loop, coroutine, **kwargs)
            if previous
            else asyncio.Task(coroutine, loop=loop, **kwargs)
        )

    async with native_phoenix_proxy(tmp_path) as (supervisor, client, closed):
        app = FastAPI()
        app.state.phoenix = supervisor
        caught = None
        try:
            loop.set_task_factory(factory)
            try:
                async with _lifespan_teardown_owner(app):
                    pass
            except BaseException as error:
                caught = error
            loop.set_task_factory(previous)
            assert (
                refused and is_execution_control_error(caught)
            ), "inner proxy Task refusal was only logged"
            source.assert_retained(caught)
            assert app.state.phoenix is supervisor and supervisor._client is client
            assert not client.is_closed and not closed.is_set()
            assert any(
                resource is client or resource is supervisor
                for resource, _ in execution_cleanup_debts(caught)
            )
            assert all(
                inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED
                for pair in refused
                for coro in pair
            )
            with pytest.raises(BaseException) as repeated:
                async with _lifespan_teardown_owner(app):
                    pass
            assert repeated.value is caught and len(refused) == 1
        finally:
            loop.set_task_factory(previous)
            for pair in refused:
                for coroutine in pair:
                    coroutine.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_actual_proxy_close_failure_keeps_original_resource_and_owner(
    tmp_path, monkeypatch, cancel
):
    from kestrel_sovereign.server import _lifespan_teardown_owner

    async with native_phoenix_proxy(tmp_path) as (supervisor, client, closed):
        app = FastAPI()
        app.state.phoenix = supervisor
        native_close, calls = client.aclose, []
        original = (
            asyncio.CancelledError("original native proxy close ACK lost")
            if cancel
            else OSError("original native proxy close ACK lost")
        )
        snapshot = OriginalFailure.capture(original)

        async def lose_ack():
            await native_close()
            calls.append(True)
            raise original

        with monkeypatch.context() as fault:
            fault.setattr(client, "aclose", lose_ack)
            with pytest.raises(BaseException) as caught:
                async with _lifespan_teardown_owner(app):
                    pass
            snapshot.assert_retained(caught.value)
            assert is_execution_control_error(caught.value)
            await asyncio.wait_for(closed.wait(), 2)
            assert client.is_closed and supervisor._client is client
            assert app.state.phoenix is supervisor
            owner = app.state.phoenix_proxy_close_task
            assert owner.done() and any(
                resource is client and task is owner
                for resource, task in execution_cleanup_debts(caught.value)
            )
            with pytest.raises(BaseException) as repeated:
                async with _lifespan_teardown_owner(app):
                    pass
            snapshot.assert_retained(repeated.value)
            assert repeated.value is caught.value and calls == [True]
            assert app.state.phoenix_proxy_close_task is owner


@pytest.mark.asyncio
async def test_actual_proxy_close_joins_original_ack_through_caller_withdrawal(
    tmp_path, monkeypatch
):
    from kestrel_sovereign.server import _shutdown_phoenix

    async with native_phoenix_proxy(tmp_path) as (supervisor, client, closed):
        app = FastAPI()
        app.state.phoenix = supervisor
        native_close = client.aclose
        entered, release = asyncio.Event(), asyncio.Event()

        async def held_ack():
            await native_close()
            entered.set()
            await release.wait()

        shutdown = None
        with monkeypatch.context() as fault:
            fault.setattr(client, "aclose", held_ack)
            shutdown = asyncio.create_task(_shutdown_phoenix(app))
            try:
                await asyncio.wait_for(entered.wait(), 2)
                owner = app.state.phoenix_proxy_close_task
                shutdown.cancel("first original observer withdrawal")
                await asyncio.sleep(0)
                shutdown.cancel("second original observer withdrawal")
                await asyncio.sleep(0)
                assert not shutdown.done() and not owner.done()
                assert owner.cancelling() == 0
                release.set()
                assert await asyncio.wait_for(shutdown, 2)
                await asyncio.wait_for(closed.wait(), 2)
                assert owner.done() and client.is_closed
                assert supervisor._client is None and app.state.phoenix is None
            finally:
                release.set()
                if shutdown is not None:
                    await await_owned_task(shutdown)
