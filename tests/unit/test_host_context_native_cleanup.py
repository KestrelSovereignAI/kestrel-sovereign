"""Actual host-owned SQLite resources survive allocation/control failures."""

import asyncio
import inspect
import sqlite3
from contextlib import closing

import aiosqlite
import pytest

from kestrel_sovereign.execution_custody import (
    execution_cleanup_debts,
    execution_commit_outcome,
    is_execution_control_error,
)
from kestrel_sovereign.host_features import (
    SovereignHostContext,
    build_host_context,
    close_host_context,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase


def _actual_operation(coroutine):
    frame = getattr(coroutine, "cr_frame", None)
    inner = frame.f_locals.get("operation") if frame is not None else None
    return inner if inspect.iscoroutine(inner) else coroutine


@pytest.mark.asyncio
@pytest.mark.parametrize("refusal", ["native", "collector", "resource"])
async def test_public_context_closer_keeps_original_native_resource_debt(
    tmp_path, refusal
):
    first = await AsyncDatabase.sqlite(str(tmp_path / "first-native.db"))
    second = await AsyncDatabase.sqlite(str(tmp_path / "second-native.db"))
    for db in (first, second):
        await db.execute("CREATE TABLE actual_native_context (id INTEGER)")
        await db.execute("INSERT INTO actual_native_context VALUES (1)")
    first_connection, second_connection = (
        first.backend._connection,
        second.backend._connection,
    )
    workers = [
        getattr(conn, "_thread", conn) for conn in (first_connection, second_connection)
    ]
    context = SovereignHostContext(db=second, hold_db=first, hold_evidence_db=first)
    loop, previous, refused = (
        asyncio.get_running_loop(),
        asyncio.get_running_loop().get_task_factory(),
        [],
    )
    original = RuntimeError("original native host closer allocation refused")

    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        code = getattr(operation, "cr_code", None)
        suffix = {
            "native": "SQLiteBackend.close",
            "collector": "_close_partial_host_resources",
            "resource": "AsyncDatabase.close",
        }[refusal]
        if not refused and code is not None and code.co_qualname == suffix:
            refused.append((coroutine, operation))
            raise original
        return (
            previous(loop, coroutine, **kwargs)
            if previous
            else asyncio.Task(coroutine, loop=loop, **kwargs)
        )

    caught = None
    try:
        loop.set_task_factory(factory)
        try:
            await close_host_context(context)
        except BaseException as error:
            caught = error
        loop.set_task_factory(previous)
        if refusal in {"native", "resource"}:
            assert (
                refused and caught is not None and is_execution_control_error(caught)
            ), "host context consumed actual native retirement debt as a warning"
            assert any(
                resource is first for resource, _ in execution_cleanup_debts(caught)
            )
            assert (
                first.backend._connection is first_connection and workers[0].is_alive()
            ), "failed original native closer must remain owned, not be declared closed"
            assert not workers[1].is_alive() and second.backend._connection is None, (
                "independent native database was not physically closed"
            )
            with pytest.raises(BaseException) as repeated:
                await close_host_context(context)
            assert repeated.value is caught, (
                "retry replaced the original unresolved native retirement owner"
            )
        else:
            assert caught is None and not refused, (
                "host cleanup unnecessarily allocated an all-resource collector Task"
            )
            assert all(not worker.is_alive() for worker in workers)
        assert all(
            inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED
            for pair in refused
            for coro in pair
        )
    finally:
        loop.set_task_factory(previous)
        for pair in refused:
            for coroutine in pair:
                coroutine.close()
        # The test owns these ORIGINAL raw resources, including the closer
        # that was never admitted. Cleanup follows the ownership assertions.
        await first.close()
        await second.close()


@pytest.mark.asyncio
async def test_actual_host_bootstrap_unknown_is_not_a_degraded_context(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("KESTREL_HOLD_BACKEND", "sqlite")
    monkeypatch.setenv("KESTREL_DB_BACKEND", "sqlite")
    path = tmp_path / "actual-bootstrap.db"
    native_commit, commits = aiosqlite.Connection.commit, []
    original = OSError("actual host bootstrap COMMIT ACK lost")

    async def lose_commit_ack(connection):
        result = await native_commit(connection)
        if not commits:
            commits.append(connection)
            raise original
        return result

    context, caught = None, None
    try:
        with monkeypatch.context() as fault:
            fault.setattr(aiosqlite.Connection, "commit", lose_commit_ack)
            try:
                context = await build_host_context(db_path=str(path))
            except BaseException as error:
                caught = error
        assert commits, "fault must reach actual SQLite COMMIT"
        assert caught is not None and execution_commit_outcome(caught) == "unknown", (
            "bootstrap normalized original native UNKNOWN into degraded context text"
        )
        assert context is None
        with closing(sqlite3.connect(path)) as independent:
            assert (
                independent.execute(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
                ).fetchone()[0]
                > 0
            )
    finally:
        if context is not None:
            await close_host_context(context)


@pytest.mark.asyncio
async def test_context_close_keeps_original_native_ack_through_repeated_cancellation(
    tmp_path, monkeypatch
):
    from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task

    first = await AsyncDatabase.sqlite(str(tmp_path / "held-original-close.db"))
    second = await AsyncDatabase.sqlite(str(tmp_path / "independent-close.db"))
    for db in (first, second):
        await db.execute("CREATE TABLE original_context_effect (id INTEGER)")
    context = SovereignHostContext(db=second, hold_db=first)
    connection = first.backend._connection
    worker = getattr(connection, "_thread", connection)
    native_close, entered, release, calls = (
        connection.close,
        asyncio.Event(),
        asyncio.Event(),
        [],
    )
    owner = None
    loop, previous = (
        asyncio.get_running_loop(),
        asyncio.get_running_loop().get_task_factory(),
    )

    async def hold_actual_close_ack():
        calls.append(connection)
        await native_close()
        entered.set()
        await release.wait()

    def forbid_new_close(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        code = getattr(operation, "cr_code", None)
        if code is not None and code.co_qualname == "AsyncDatabase.close":
            raise AssertionError(
                "completed original resource acquired a replacement close Task"
            )
        return (
            previous(loop, coroutine, **kwargs)
            if previous
            else asyncio.Task(coroutine, loop=loop, **kwargs)
        )

    try:
        monkeypatch.setattr(connection, "close", hold_actual_close_ack)
        owner = create_owned_task(close_host_context(context))
        await asyncio.wait_for(entered.wait(), 5)
        original_close = context._native_close_owners[id(first)].task
        owner.cancel("original caller stop")
        await asyncio.sleep(0)
        owner.cancel("repeated original caller stop")
        await asyncio.sleep(0.01)
        assert not owner.done() and not original_close.done(), (
            "host released its actual close acknowledgement owner"
        )
        release.set()
        outcome = await await_owned_task(owner)
        assert isinstance(outcome.error, asyncio.CancelledError)
        assert calls == [connection] and not worker.is_alive()
        assert first.backend._connection is None and second.backend._connection is None
        loop.set_task_factory(forbid_new_close)
        await close_host_context(context)
        assert calls == [connection]
    finally:
        loop.set_task_factory(previous)
        release.set()
        if owner is not None:
            await await_owned_task(owner)
        await first.close()
        await second.close()
