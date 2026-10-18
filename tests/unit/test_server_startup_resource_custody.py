"""New server lifespans cannot overwrite actual retained native resources."""

import asyncio

import pytest
from tests.utils.original_failure import OriginalFailure
import pytest_asyncio
from fastapi import FastAPI

from kestrel_sovereign._async_ownership import await_owned_task
from kestrel_sovereign.execution_custody import (
    _explicit_control_chain,
    execution_cleanup_debts,
    is_execution_control_error,
)
from kestrel_sovereign.hold import build_bound_host_context
from kestrel_sovereign.host_features import close_host_context
from kestrel_sovereign.storage.async_database import AsyncDatabase
from tests.unit.test_cleanup_task_admission import _partial_native_agent
from tests.unit.test_host_context_native_cleanup import _actual_operation
from tests.unit.test_server_original_shutdown_dependencies import native_phoenix_proxy


@pytest_asyncio.fixture
async def original_server(tmp_path, monkeypatch):
    monkeypatch.setenv("KESTREL_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("KESTREL_HOST_DB_PATH", str(tmp_path / "actual-host.db"))
    monkeypatch.setenv("KESTREL_DB_BACKEND", "sqlite")
    monkeypatch.setenv("KESTREL_HOLD_BACKEND", "sqlite")
    agent, db = await _partial_native_agent(tmp_path, monkeypatch)
    context = await build_bound_host_context(
        agent, agent_data_root=tmp_path / "agent-root"
    )
    receipts = await AsyncDatabase.sqlite(str(tmp_path / "original-receipts.db"))
    app = FastAPI()
    app.state.agent = agent
    app.state.host_context = context
    app.state.host_features = []
    app.state.stop_receipt_db = receipts
    try:
        yield app, agent, db, context, receipts
    finally:
        await agent.shutdown()
        await db.close()
        # A refused original facade closer can retain its control after this
        # explicit physical fixture cleanup. Assertions always precede it.
        await receipts.backend.close()
        try:
            await receipts.close()
        except BaseException as error:
            if not is_execution_control_error(error):
                raise
        await close_host_context(context)


@pytest.mark.asyncio
async def test_original_same_app_startup_cannot_overwrite_retained_native_owners(
    original_server, monkeypatch
):
    from kestrel_sovereign import server

    app, agent, db, context, receipts = original_server
    host_connection = context.hold_db.backend._connection
    host_worker = getattr(host_connection, "_thread", host_connection)
    loop, previous, refused = (
        asyncio.get_running_loop(),
        asyncio.get_running_loop().get_task_factory(),
        [],
    )
    refusal = RuntimeError("original Agent phase refused")

    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        code = getattr(operation, "cr_code", None)
        if (
            not refused
            and code is not None
            and code.co_qualname == "_shutdown_server_agents"
        ):
            refused.append((coroutine, operation))
            raise refusal
        return (
            previous(loop, coroutine, **kwargs)
            if previous
            else asyncio.Task(coroutine, loop=loop, **kwargs)
        )

    started, release = asyncio.Event(), asyncio.Event()
    original_initialize = server._initialize_stop_receipts

    async def hold_before_original_receipt_initialization(app):
        # Observe entry BEFORE original acquisition, never substitute its
        # result or publish fake initialization/genesis/READY.
        started.set()
        await release.wait()
        return await original_initialize(app)

    async def restart():
        async with server.lifespan(app):
            raise AssertionError("retained resources reached a new serving lifespan")

    restart_task, observing = None, None
    try:
        loop.set_task_factory(factory)
        with pytest.raises(BaseException) as caught:
            async with server._lifespan_teardown_owner(app):
                pass
        assert is_execution_control_error(caught.value)
        assert refusal in tuple(_explicit_control_chain(caught.value))
        assert tuple(_explicit_control_chain(refusal)) == (refusal,)
        loop.set_task_factory(previous)
        assert (
            app.state.agent is agent
            and app.state.host_context is context
            and app.state.stop_receipt_db is receipts
        )
        assert host_worker.is_alive()
        monkeypatch.setattr(
            server,
            "_initialize_stop_receipts",
            hold_before_original_receipt_initialization,
        )
        restart_task = asyncio.create_task(restart())
        observing = asyncio.create_task(started.wait())
        await asyncio.wait(
            {restart_task, observing}, timeout=5, return_when=asyncio.FIRST_COMPLETED
        )
        assert restart_task.done() and not started.is_set(), (
            "new startup entered before original native owners retired"
        )
        with pytest.raises(RuntimeError, match="retire"):
            await restart_task
        assert (
            app.state.host_context is context
            and app.state.stop_receipt_db is receipts
            and host_worker.is_alive()
        )
        assert not agent._runtime_publication_ready.is_set()
    finally:
        loop.set_task_factory(previous)
        for task in (restart_task, observing):
            if task is not None:
                if not task.done():
                    task.cancel()
                await await_owned_task(task)
        for pair in refused:
            for coroutine in pair:
                coroutine.close()


@pytest.mark.asyncio
async def test_cached_original_phoenix_close_cannot_retire_replacement(
    tmp_path, monkeypatch
):
    from kestrel_sovereign import server

    async with (
        native_phoenix_proxy(tmp_path / "original") as (
            original_supervisor,
            original_client,
            original_closed,
        ),
        native_phoenix_proxy(tmp_path / "replacement") as (
            replacement,
            replacement_client,
            replacement_closed,
        ),
    ):
        app = FastAPI()
        app.state.phoenix = original_supervisor
        original_close = type(original_client).aclose
        failure = OSError("original physical Phoenix close ACK lost")
        snapshot = OriginalFailure.capture(failure)

        async def close_then_lose_original_ack(client):
            result = await original_close(client)
            if client is original_client:
                raise failure
            return result

        with monkeypatch.context() as patch:
            patch.setattr(type(original_client), "aclose", close_then_lose_original_ack)
            with pytest.raises(BaseException) as first:
                await server._shutdown_phoenix(app)
        snapshot.assert_retained(first.value)
        await asyncio.wait_for(original_closed.wait(), 2)
        original_task = app.state.phoenix_proxy_close_task
        assert original_task.done() and original_client.is_closed
        assert app.state.phoenix_proxy_close_supervisor is original_supervisor
        # Exercise defensive generation binding even if an external caller
        # bypasses the public lifespan preflight and installs another owner.
        app.state.phoenix = replacement
        cached = app.state.phoenix_proxy_close_error
        cached_source = OriginalFailure.capture(cached)
        with pytest.raises(BaseException) as repeated:
            await server._shutdown_phoenix(app)
        assert is_execution_control_error(repeated.value)
        debts = execution_cleanup_debts(repeated.value)
        assert all(
            any(resource is supervisor for resource, _ in debts)
            for supervisor in (original_supervisor, replacement)
        ), "replacement and prior supervisor must retain distinct native custody"
        assert app.state.phoenix is replacement
        assert app.state.phoenix_proxy_close_task is original_task
        assert app.state.phoenix_proxy_close_supervisor is original_supervisor
        assert not replacement_client.is_closed and not replacement_closed.is_set()
        for _ in range(20):
            with pytest.raises(BaseException) as later:
                await server._shutdown_phoenix(app)
            snapshot.assert_retained(later.value)
            cached_source.assert_retained(app.state.phoenix_proxy_close_error)
            assert app.state.phoenix_proxy_close_error is cached
            assert app.state.phoenix_proxy_close_task is original_task
        assert not replacement_client.is_closed and not replacement_closed.is_set()
        with pytest.raises(RuntimeError, match="retire"):
            server._require_retired_server_resources(app)


@pytest.mark.asyncio
async def test_positive_original_retirement_allows_next_preflight(original_server):
    from kestrel_sovereign import server

    app, agent, db, context, receipts = original_server
    connections = (
        db.backend._connection,
        context.hold_db.backend._connection,
        receipts.backend._connection,
    )
    workers = tuple(
        getattr(connection, "_thread", connection) for connection in connections
    )
    async with server._lifespan_teardown_owner(app):
        pass
    assert all(not worker.is_alive() for worker in workers)
    assert (
        app.state.agent is None
        and app.state.host_context is None
        and app.state.stop_receipt_db is None
    )
    server._require_retired_server_resources(app)
    assert not agent._runtime_publication_ready.is_set()


@pytest.mark.asyncio
async def test_failed_original_stop_database_close_remains_owned(original_server):
    from kestrel_sovereign import server

    app, agent, db, context, receipts = original_server
    connection = receipts.backend._connection
    worker = getattr(connection, "_thread", connection)
    loop, previous, refused = (
        asyncio.get_running_loop(),
        asyncio.get_running_loop().get_task_factory(),
        [],
    )
    original = RuntimeError("original Stop evidence database closer refused")
    source = OriginalFailure.capture(original)

    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        code, frame = (
            getattr(operation, "cr_code", None),
            getattr(operation, "cr_frame", None),
        )
        if (
            not refused
            and code is not None
            and code.co_qualname == "SQLiteBackend.close"
            and frame.f_locals.get("self") is receipts.backend
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
        with pytest.raises(BaseException) as caught:
            await server._shutdown_stop_receipts(app)
        loop.set_task_factory(previous)
        source.assert_retained(caught.value)
        assert is_execution_control_error(caught.value)
        assert any(resource is receipts and task is None
                   for resource, task in execution_cleanup_debts(caught.value))
        assert app.state.stop_receipt_db is receipts and worker.is_alive()
        with pytest.raises(RuntimeError, match="retire"):
            server._require_retired_server_resources(app)
        assert refused
    finally:
        loop.set_task_factory(previous)
        for pair in refused:
            for coroutine in pair:
                coroutine.close()
