"""Actual default ASGI response/native owner boundaries, not full Agent READY."""

import asyncio
import inspect
from contextlib import asynccontextmanager
from contextvars import ContextVar

import asyncpg
import pytest
from starlette.requests import ClientDisconnect

from kestrel_sovereign._async_ownership import (
    OwnedAsyncIterator,
    await_owned_task,
    create_owned_task,
)
from kestrel_sovereign.endpoints.closing_streaming_response import (
    ClosingStreamingResponse,
)
from kestrel_sovereign.execution_custody import (
    ExecutionCustody,
    _explicit_control_chain,
    execution_cleanup_debts,
    execution_commit_outcome,
    is_execution_control_error,
)
from kestrel_sovereign.storage.db.postgres import PostgresPoolAdapter
from tests.integration.test_execution_authority_postgres import GenerationFence
from tests.integration.test_execution_authority_postgres import native_pg as native_pg


def scope(version):
    return {
        "type": "http",
        "method": "GET",
        "path": "/native-default-stream",
        "headers": [],
        "asgi": {"spec_version": version},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("version", ["2.3", "2.4"])
@pytest.mark.parametrize("failure_kind", ["cancel", "oserror"])
async def test_default_asgi_preserves_original_native_control(
    native_pg, monkeypatch, version, failure_kind
):
    backend, observer, schema = native_pg
    backend._execution_custody = ExecutionCustody(GenerationFence())
    entered, release, closed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original_errors, frames = [], []
    context = ContextVar("default native source context", default=None)
    marker = object()
    original_pool = backend._pool
    source_owner = None
    if failure_kind == "cancel":
        native_exit = asyncpg.pool.PoolAcquireContext.__aexit__

        async def hold_actual_release(checkout, typ, error, tb):
            result = await native_exit(checkout, typ, error, tb)
            entered.set()
            await release.wait()
            return result

        monkeypatch.setattr(
            asyncpg.pool.PoolAcquireContext, "__aexit__", hold_actual_release
        )
    else:

        @asynccontextmanager
        async def wrapper(native):
            async with native as connection:
                yield connection
            # Original wrapper retirement observer, AFTER genuine native COMMIT/reset.
            raise OSError("actual native checkout wrapper acknowledgement lost")

        backend._pool = PostgresPoolAdapter(original_pool, wrapper)

    async def source():
        nonlocal source_owner
        source_owner = asyncio.current_task()
        token = context.set(marker)
        try:
            async with backend.transaction():
                await backend.execute(
                    "INSERT INTO effects VALUES (1, 'native response')"
                )
            yield b"actual committed native frame"
        except BaseException as error:
            original_errors.append(error)
            raise
        finally:
            assert context.get() is marker and asyncio.current_task() is source_owner
            context.reset(token)
            closed.set()

    body = OwnedAsyncIterator(source, operation="actual default response native source")
    response = ClosingStreamingResponse(body)
    incoming = asyncio.Queue()

    async def receive():
        return await incoming.get()

    async def send(message):
        frames.append(message)

    response_owner = create_owned_task(response(scope(version), receive, send))
    try:
        if failure_kind == "cancel":
            await asyncio.wait_for(entered.wait(), 5)
            body.owner_task.cancel("original native stream withdrawal")
            await asyncio.sleep(0)
            assert not source_owner.done()
            release.set()
        done, pending = await asyncio.wait({response_owner}, timeout=5)
        assert done and not pending, "ASGI observer failed to settle"
        result = await await_owned_task(response_owner)
        assert len(original_errors) == 1 and closed.is_set()
        original = original_errors[0]
        assert is_execution_control_error(original)
        assert execution_commit_outcome(original) == "committed"
        assert result.error is not None and is_execution_control_error(result.error), (
            "ASGI normalized original native control into EOF/client disconnect"
        )
        assert original in tuple(_explicit_control_chain(result.error))
        assert execution_commit_outcome(result.error) == "committed"
        assert not any(m.get("body") for m in frames)
    finally:
        release.set()
        if not response_owner.done():
            response_owner.cancel("fixture retires its response observer")
        await await_owned_task(response_owner)
        try:
            await body.aclose()
        except BaseException as error:
            assert is_execution_control_error(error)
        await await_owned_task(body._owner)
        await await_owned_task(body.owner_task)
        backend._pool = original_pool
        backend._execution_custody = None


@pytest.mark.asyncio
@pytest.mark.parametrize("version", ["2.3", "2.4"])
@pytest.mark.parametrize("withdrawal", ["send", "caller"])
async def test_default_response_refused_close_retains_actual_native_body(
    native_pg, version, withdrawal
):
    backend, observer, schema = native_pg
    backend._execution_custody = ExecutionCustody(GenerationFence())
    context = ContextVar("actual held transaction source", default=None)
    marker = object()
    closed = asyncio.Event()
    native_connections, native_pids, refused = [], [], []
    original_delivery = OSError("actual response send lost its client")
    original_refusal = RuntimeError("actual response cleanup Task admission refused")
    chunk_seen = asyncio.Event()
    source_release = asyncio.Event()

    async def source():
        owner = asyncio.current_task()
        token = context.set(marker)
        try:
            async with backend.transaction():
                await backend.execute(
                    "INSERT INTO effects VALUES (1, 'not yet committed')"
                )
                native_connections.append(backend._current_txn_conn())
                native_pids.append(await backend.fetch_val("SELECT pg_backend_pid()"))
                yield b"actual original native transaction is still open"
                await source_release.wait()
        finally:
            assert asyncio.current_task() is owner and context.get() is marker
            context.reset(token)
            closed.set()

    body = OwnedAsyncIterator(
        source, operation="default response original held native transaction"
    )
    response = ClosingStreamingResponse(body)
    loop, old_factory = (
        asyncio.get_running_loop(),
        asyncio.get_running_loop().get_task_factory(),
    )

    def refuse(loop, operation, **kwargs):
        operation_body = (
            operation.cr_frame.f_locals.get("operation") if operation.cr_frame else None
        )
        refused.append((operation, operation_body))
        raise original_refusal

    incoming = asyncio.Queue()

    async def receive():
        return await incoming.get()

    async def send(message):
        if message["type"] == "http.response.body" and message.get("body"):
            if withdrawal == "send":
                loop.set_task_factory(refuse)
                raise original_delivery
            chunk_seen.set()

    failure = None
    response_owner = create_owned_task(
        response(scope(version), receive, send), retain_originals=True
    )
    try:
        if withdrawal == "caller":
            await asyncio.wait_for(chunk_seen.wait(), 5)
            await asyncio.sleep(0)
            loop.set_task_factory(refuse)
            response_owner.cancel("actual response caller withdrawal")
        outcome = await await_owned_task(response_owner)
        failure = outcome.error
        assert refused and native_connections and native_pids
        assert not closed.is_set() and not body._owner.done()
        assert native_connections[0].is_in_transaction()
        row = await observer.fetchrow(
            "SELECT state, xact_start IS NOT NULL FROM pg_stat_activity WHERE pid=$1",
            native_pids[0],
        )
        assert row is not None and row[0] == "idle in transaction" and row[1]
        assert failure is not None and is_execution_control_error(failure), (
            "refused close lost the actual outstanding native body obligation"
        )
        assert any(
            resource is body and owner is None
            for resource, owner in execution_cleanup_debts(failure)
        )
        chain = tuple(_explicit_control_chain(failure))
        if withdrawal == "caller":
            assert any(
                isinstance(error, asyncio.CancelledError)
                and error.args == ("actual response caller withdrawal",)
                for error in chain
            )
        elif version == "2.4":
            # Starlette's ordinary transport contract is ClientDisconnect.
            # Its actual original primary is retained, not replaced by refusal.
            assert any(
                isinstance(error, ClientDisconnect)
                and error.__context__ is original_delivery
                for error in chain
            )
        else:
            assert original_delivery in chain
        assert original_refusal in tuple(_explicit_control_chain(failure))
        for wrapper, operation in refused:
            assert inspect.getcoroutinestate(wrapper) == inspect.CORO_CLOSED
            if inspect.iscoroutine(operation):
                assert inspect.getcoroutinestate(operation) == inspect.CORO_CLOSED
    finally:
        loop.set_task_factory(old_factory)
        if not response_owner.done():
            response_owner.cancel("fixture retires its response observer")
        await await_owned_task(response_owner)
        # Fixture-only retirement, AFTER all production outcome/debt assertions.
        for wrapper, operation in refused:
            wrapper.close()
            if inspect.iscoroutine(operation):
                operation.close()
        try:
            await body.aclose()
        except BaseException as error:
            assert is_execution_control_error(error)
        await await_owned_task(body._owner)
        await await_owned_task(body.owner_task)
        backend._execution_custody = None
        assert closed.is_set() and body._owner.done() and body.owner_task.done()
        assert await backend.fetch_val("SELECT count(*) FROM effects") == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("version", ["2.3", "2.4"])
@pytest.mark.parametrize("withdrawal", ["send", "caller"])
async def test_default_response_joins_actual_native_close_before_reporting(
    native_pg, monkeypatch, version, withdrawal
):
    backend, observer, schema = native_pg
    backend._execution_custody = ExecutionCustody(GenerationFence())
    entered, release, chunk_seen, closed = (asyncio.Event() for _ in range(4))
    original_delivery = OSError("original native-close response delivery failed")
    original_ack_error = OSError(
        "actual physical release ACK lost while closing response"
    )
    native_exit = asyncpg.pool.PoolAcquireContext.__aexit__
    source_errors = []
    context = ContextVar("accepted response source owner context", default=None)
    marker = object()

    async def held_native_release(checkout, typ, error, tb):
        await native_exit(checkout, typ, error, tb)
        entered.set()
        await release.wait()
        raise original_ack_error

    async def source():
        original_owner = asyncio.current_task()
        token = context.set(marker)
        try:
            async with backend.transaction():
                await backend.execute(
                    "INSERT INTO effects VALUES (1, 'actual response native close')"
                )
                yield b"actual guarded native transaction"
                await asyncio.Event().wait()
        except BaseException as error:
            source_errors.append(error)
            raise
        finally:
            assert asyncio.current_task() is original_owner and context.get() is marker
            context.reset(token)
            closed.set()

    body = OwnedAsyncIterator(source, operation="accepted response native closer")
    response = ClosingStreamingResponse(body)
    incoming = asyncio.Queue()

    async def receive():
        return await incoming.get()

    async def send(message):
        if message["type"] == "http.response.body" and message.get("body"):
            chunk_seen.set()
            if withdrawal == "send":
                raise original_delivery

    monkeypatch.setattr(
        asyncpg.pool.PoolAcquireContext, "__aexit__", held_native_release
    )
    response_owner = create_owned_task(
        response(scope(version), receive, send), retain_originals=True
    )
    try:
        await asyncio.wait_for(chunk_seen.wait(), 5)
        if withdrawal == "caller":
            response_owner.cancel("original response caller withdrawal")
        await asyncio.wait_for(entered.wait(), 5)
        assert (
            not response_owner.done() and not body._owner.done() and not closed.is_set()
        )
        response_owner.cancel(
            "response observer withdrew while original close ACK pending"
        )
        await asyncio.sleep(0)
        response_owner.cancel("repeated response observer withdrawal")
        await asyncio.sleep(0)
        assert (
            not response_owner.done() and not body._owner.done() and not closed.is_set()
        )
        release.set()
        result = await await_owned_task(response_owner)
        assert result.error is not None and is_execution_control_error(result.error)
        chain = tuple(_explicit_control_chain(result.error))
        assert len(source_errors) == 1 and source_errors[0] in chain
        assert original_ack_error in chain
        expected_withdrawal = (
            "original response caller withdrawal"
            if withdrawal == "caller"
            else "response observer withdrew while original close ACK pending"
        )
        assert any(
            isinstance(error, asyncio.CancelledError)
            and error.args == (expected_withdrawal,)
            for error in chain
        )
        if withdrawal == "send":
            assert original_delivery in chain or any(
                isinstance(error, ClientDisconnect)
                and error.__context__ is original_delivery
                for error in chain
            )
        assert closed.is_set() and body._owner.done() and body.owner_task.done()
        assert (
            await observer.fetchval('SELECT count(*) FROM "' + schema + '".effects')
            == 0
        )
    finally:
        release.set()
        if not response_owner.done():
            response_owner.cancel("fixture retires original response observer")
        await await_owned_task(response_owner)
        try:
            await body.aclose()
        except BaseException as error:
            assert is_execution_control_error(error)
        await await_owned_task(body._owner)
        await await_owned_task(body.owner_task)
        backend._execution_custody = None
