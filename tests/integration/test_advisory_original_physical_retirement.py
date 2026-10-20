"""Advisory retirement waits for the original PostgreSQL session, not local abort.

Only disposable native PostgreSQL sessions are used. The in-flight bounded
pg_sleep is an actual protocol tail, not CPU stress or an external provider.
"""

import asyncio
from uuid import uuid4

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import (
    ExecutionCustody, _explicit_control_chain, execution_commit_outcome,
)
from tests.integration.test_execution_authority_postgres import GenerationFence, native_pg as native_pg
from tests.integration.test_execution_custody_review56_pg import lose_native_ack
from kestrel_sovereign.storage.db.postgres import TerminalAdvisoryBarrier


@pytest.mark.asyncio
@pytest.mark.parametrize(("route", "shared"), [("advisory", False), ("advisory", True), ("polled", False), ("terminal", False)])
@pytest.mark.parametrize("body_kind", ["unknown", "cancel"])
async def test_advisory_error_exit_joins_original_server_session(
    native_pg, monkeypatch, route, shared, body_kind,
):
    backend, control, schema = native_pg
    original = None
    if body_kind == "unknown":
        backend._execution_custody = ExecutionCustody(GenerationFence())
        with monkeypatch.context() as ack:
            committed = lose_native_ack(ack, boundary="commit", original=OSError("actual advisory body COMMIT ACK lost"))
            try:
                await backend.execute("INSERT INTO effects VALUES (188, 'original committed advisory body')")
            except BaseException as error:
                original = error
            assert committed == [True] and execution_commit_outcome(original) == "unknown"
        backend._execution_custody = None

        assert await control.fetchval(f'SELECT count(*) FROM "{schema}".effects WHERE id=188') == 1
    source_graph = tuple(_explicit_control_chain(original))
    key = (3569, uuid4().int % (2**31))
    queries, capabilities = [], []
    entered = asyncio.Event()
    acquired = False
    task = None

    async def run():
        lease = (backend.polled_advisory_lock(key) if route == "polled" else
                 backend.advisory_locks(TerminalAdvisoryBarrier((key,)) if route == "terminal" else [key],
                                        shared=shared, on_loss=lambda _lease: None))
        async with lease as capability:
            connection = (capability._connection if capability is not None else
                          next(holder._proxy for holder in backend._advisory_pool._holders if holder._in_use is not None))
            pid = connection.get_server_pid()
            capabilities.append((connection, pid))
            # Deliberately interrupt an original native protocol tail. The
            # lease owns abort/retirement even if the body cannot join it.
            queries.append(create_owned_task(connection.execute("SELECT pg_sleep(0.5)")))
            async with asyncio.timeout(5):
                while not await control.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM pg_stat_activity WHERE pid=$1 AND query='SELECT pg_sleep(0.5)' AND state='active')",
                    pid,
                ):
                    await asyncio.sleep(0.01)
            entered.set()
            if body_kind == "cancel":
                await asyncio.Event().wait()
            raise original

    try:
        task = create_owned_task(run(), retain_originals=True)
        await asyncio.wait_for(entered.wait(), 5)
        if body_kind == "cancel":
            task.cancel("actual advisory body withdrawal")
        outcome = await await_owned_task(task)
        assert outcome.error is not None
        if body_kind == "unknown":
            assert original in tuple(_explicit_control_chain(outcome.error))
            assert execution_commit_outcome(outcome.error) == "unknown"
            assert tuple(_explicit_control_chain(original)) == source_graph
        else:
            assert isinstance(outcome.error, asyncio.CancelledError)
        acquired = await control.fetchval("SELECT pg_try_advisory_lock($1,$2)", *key)
        assert acquired, "advisory exit returned after local abort but BEFORE original PostgreSQL session released its lock"
        assert all(query.done() for query in queries), "original native protocol tail was not physically joined"
    finally:
        if task is not None:
            await await_owned_task(task)
        for query in queries:
            await await_owned_task(query)
        if acquired:
            await control.fetchval("SELECT pg_advisory_unlock($1,$2)", *key)
        # Exact fixture-owned server PID only. Even a negative test may not
        # leave its native server tail unobserved after the assertion fails.
        for _connection, pid in capabilities:
            async with asyncio.timeout(5):
                while await control.fetchval("SELECT EXISTS(SELECT 1 FROM pg_stat_activity WHERE pid=$1)", pid):
                    await asyncio.sleep(0.01)
        backend._execution_custody = None



@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["refused", "close-ack", "caller-and-close-ack"])
async def test_advisory_failed_closer_retains_original_resource_debt(native_pg, monkeypatch, mode):
    import inspect
    import asyncpg
    from kestrel_sovereign.storage.db import postgres as postgres_module
    from kestrel_sovereign.execution_custody import execution_cleanup_debts, is_execution_control_error
    from tests.unit.test_host_context_native_cleanup import _actual_operation

    backend, control, _schema = native_pg
    backend._execution_custody = ExecutionCustody(GenerationFence())
    original = None
    with monkeypatch.context() as ack:
        committed = lose_native_ack(ack, boundary="commit", original=OSError("actual closer body COMMIT ACK lost"))
        try:
            await backend.execute("INSERT INTO effects VALUES (189, 'actual failed closer body')")
        except BaseException as error:
            original = error
        assert committed == [True] and execution_commit_outcome(original) == "unknown"
    backend._execution_custody = None
    graph = tuple(_explicit_control_chain(original))
    original_debts = tuple(execution_cleanup_debts(original))
    pool = await backend._ensure_advisory_pool()
    entered, release = asyncio.Event(), asyncio.Event()
    actual_close, actual_join = asyncpg.Connection.close, postgres_module.await_owned_task
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    refusal, lost_ack = RuntimeError("actual original advisory closer admission refused"), OSError("actual native advisory close ACK lost")
    refused, withdrawals, close_tasks = [], [], []
    owner = None
    pid = None

    async with pool.acquire() as proxy:
        native = proxy._con
        pid = native.get_server_pid()

        async def observed_close(receiver, *args, **kwargs):
            if receiver is not native:
                return await actual_close(receiver, *args, **kwargs)
            close_tasks.append(asyncio.current_task())
            entered.set()
            await release.wait()
            await actual_close(receiver, *args, **kwargs)
            raise lost_ack

        async def observed_join(task, *args, **kwargs):
            outcome = await actual_join(task, *args, **kwargs)
            if task in close_tasks and outcome.cancellation is not None:
                withdrawals.append(outcome.cancellation)
            return outcome

        def factory(loop, coroutine, **kwargs):
            actual = _actual_operation(coroutine)
            code, frame = getattr(actual, "cr_code", None), getattr(actual, "cr_frame", None)
            if mode == "refused" and code is not None and code.co_qualname.endswith("observed_close") and frame.f_locals.get("receiver") is native:
                refused.append((coroutine, actual))
                raise refusal
            return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)

        try:
            with monkeypatch.context() as patch:
                patch.setattr(asyncpg.Connection, "close", observed_close)
                patch.setattr(postgres_module, "await_owned_task", observed_join)
                loop.set_task_factory(factory)
                owner = create_owned_task(backend._close_failed_advisory_session(proxy, original), retain_originals=True)
                if mode != "refused":
                    await asyncio.wait_for(entered.wait(), 5)
                    if mode == "caller-and-close-ack":
                        owner.cancel("actual advisory retirement observer withdrawal")
                    release.set()
                outcome = await await_owned_task(owner)
                terminal = outcome.result if outcome.error is None else outcome.error
                loop.set_task_factory(previous)
            assert terminal is not None and is_execution_control_error(terminal)
            assert execution_commit_outcome(terminal) == "unknown"
            assert original in tuple(_explicit_control_chain(terminal))
            assert tuple(_explicit_control_chain(original)) == graph
            assert tuple(execution_cleanup_debts(original)) == original_debts
            debts = execution_cleanup_debts(terminal)
            assert any(resource.connection is proxy and resource.native_connection is native and
                       ((task is None) if mode == "refused" else (task in close_tasks and task.done()))
                       for resource, task in debts), "failed advisory closer lost its exact original resource/owner"
            if mode == "refused":
                assert refusal in tuple(_explicit_control_chain(terminal)) and refused
                assert all(inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED for pair in refused for coro in pair)
            else:
                assert lost_ack in tuple(_explicit_control_chain(terminal))
            if mode == "caller-and-close-ack":
                assert len(withdrawals) == 1 and withdrawals[0] in tuple(_explicit_control_chain(terminal))
        finally:
            loop.set_task_factory(previous)
            release.set()
            if owner is not None:
                await await_owned_task(owner)
            for pair in refused:
                for coroutine in pair:
                    coroutine.close()
            await actual_close(native)
    async with asyncio.timeout(5):
        while await control.fetchval("SELECT EXISTS(SELECT 1 FROM pg_stat_activity WHERE pid=$1)", pid):
            await asyncio.sleep(0.01)


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["advisory", "polled", "terminal"])
async def test_unlock_ack_failure_joins_original_native_protocol_tail(native_pg, monkeypatch, route):
    import asyncpg

    backend, control, _schema = native_pg
    key = (3569, uuid4().int % (2**31))
    actual_execute, actual_fetchval = asyncpg.Connection.execute, asyncpg.Connection.fetchval
    ack = OSError("actual original advisory unlock acknowledgement lost")
    sessions, queries, lost = [], [], []
    pid = None
    acquired = False

    async def after_unlock(connection, query, call, *args, **kwargs):
        result = await call(connection, query, *args, **kwargs)
        if sessions and connection is sessions[0] and query.startswith("SELECT pg_advisory_unlock(") and not lost:
            lost.append(True)
            queries.append(create_owned_task(actual_execute(connection, "SELECT pg_sleep(0.5)")))
            async with asyncio.timeout(5):
                while not await control.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM pg_stat_activity WHERE pid=$1 AND query='SELECT pg_sleep(0.5)' AND state='active')",
                    pid,
                ):
                    await asyncio.sleep(0.01)
            raise ack
        return result

    async def observed_execute(connection, query, *args, **kwargs):
        return await after_unlock(connection, query, actual_execute, *args, **kwargs)

    async def observed_fetchval(connection, query, *args, **kwargs):
        return await after_unlock(connection, query, actual_fetchval, *args, **kwargs)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(asyncpg.Connection, "execute", observed_execute)
            patch.setattr(asyncpg.Connection, "fetchval", observed_fetchval)
            lease = (backend.polled_advisory_lock(key) if route == "polled" else
                     backend.advisory_locks(TerminalAdvisoryBarrier((key,)) if route == "terminal" else [key]))
            with pytest.raises(OSError) as caught:
                async with lease as capability:
                    proxy = (capability._connection if capability is not None else
                             next(holder._proxy for holder in backend._advisory_pool._holders if holder._in_use is not None))
                    sessions.append(proxy._con)
                    pid = proxy.get_server_pid()
            assert caught.value is ack and lost == [True]
        assert not await control.fetchval("SELECT EXISTS(SELECT 1 FROM pg_stat_activity WHERE pid=$1)", pid), "unlock failure returned before original server protocol tail retired"
        assert all(query.done() for query in queries)
        acquired = await control.fetchval("SELECT pg_try_advisory_lock($1,$2)", *key)
        assert acquired
    finally:
        for query in queries:
            await await_owned_task(query)
        if acquired:
            await control.fetchval("SELECT pg_advisory_unlock($1,$2)", *key)
        if pid is not None:
            async with asyncio.timeout(5):
                while await control.fetchval("SELECT EXISTS(SELECT 1 FROM pg_stat_activity WHERE pid=$1)", pid):
                    await asyncio.sleep(0.01)
