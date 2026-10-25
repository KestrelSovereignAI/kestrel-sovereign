"""Original standalone resources and caller withdrawal survive failed startup."""

import asyncio
import inspect
import sqlite3
from pathlib import Path

import pytest

from kestrel_sovereign.execution_custody import (
    execution_cleanup_debts,
    is_execution_control_error,
)
from kestrel_sovereign.hold.enforcement import initialize_with_bound_hold_context
from kestrel_sovereign.host_features.storage import resolve_host_database_launch_context
from kestrel_sovereign.storage.db import SQLiteBackend
from tests.unit.test_cleanup_task_admission import _partial_native_agent
from tests.utils.aiosqlite_workers import wait_for_lifecycle_checkpoint


@pytest.mark.asyncio
@pytest.mark.parametrize("refusal", ["native", "outer"])
async def test_actual_standalone_startup_preserves_host_close_debt(
    tmp_path, monkeypatch, refusal
):
    monkeypatch.setenv("KESTREL_HOME", str(tmp_path / "host-home"))
    monkeypatch.setenv("KESTREL_HOLD_BACKEND", "sqlite")
    monkeypatch.setenv("KESTREL_DB_BACKEND", "sqlite")
    agent, db = await _partial_native_agent(tmp_path, monkeypatch)
    native_connect = SQLiteBackend.connect
    denied_schema_actions = []

    async def connect_with_actual_schema_denial(backend):
        result = await native_connect(backend)
        if Path(backend.db_path) == Path(agent.storage_path):
            # Fail original initialization in SQLite itself, independently of
            # installed feature registries or incidental Peers availability.
            # This is a real native authorizer, not a substituted initialize
            # method, fake genesis/READY, or fabricated database exception.
            def deny_create_table(action, *_args):
                if action == sqlite3.SQLITE_CREATE_TABLE:
                    denied_schema_actions.append(action)
                    return sqlite3.SQLITE_DENY
                return sqlite3.SQLITE_OK

            connection = backend._connection
            await connection._execute(
                connection._conn.set_authorizer, deny_create_table
            )
        return result

    monkeypatch.setattr(SQLiteBackend, "connect", connect_with_actual_schema_denial)
    host_path = tmp_path / "actual-host.db"
    launch = resolve_host_database_launch_context(
        env={
            "KESTREL_HOST_DB_PATH": str(host_path),
            "KESTREL_HOLD_BACKEND": "sqlite",
            "KESTREL_DB_BACKEND": "sqlite",
            "KESTREL_HOME": str(tmp_path / "host-home"),
        },
        project_root=tmp_path,
    )
    loop, previous = (
        asyncio.get_running_loop(),
        asyncio.get_running_loop().get_task_factory(),
    )
    refused, original = (
        [],
        RuntimeError("actual original host closer Task allocation refused"),
    )

    def factory(loop, coroutine, **kwargs):
        frame = getattr(coroutine, "cr_frame", None)
        operation = frame.f_locals.get("operation") if frame is not None else None
        operation = operation if inspect.iscoroutine(operation) else coroutine
        code, frame = (
            getattr(operation, "cr_code", None),
            getattr(operation, "cr_frame", None),
        )
        backend = frame.f_locals.get("self") if frame is not None else None
        selected = code is not None and (
            (
                refusal == "native"
                and code.co_qualname == "SQLiteBackend.close"
                and Path(backend.db_path) == launch.database_path
            )
            or (refusal == "outer" and code.co_qualname == "close_bound_host_context")
        )
        if not refused and selected:
            refused.append((coroutine, operation, backend))
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
            await initialize_with_bound_hold_context(
                agent, host_database_launch_context=launch
            )
        except BaseException as error:
            caught = error
        loop.set_task_factory(previous)
        assert denied_schema_actions, "original native schema denial was not exercised"
        if refusal == "native":
            assert refused, (
                "actual standalone failure cleanup never reached native host closer"
            )
            backend = refused[0][2]
            assert backend._connection is not None
            worker = getattr(backend._connection, "_thread", backend._connection)
            assert worker.is_alive()
            assert caught is not None and is_execution_control_error(caught), (
                "standalone caller discarded actual host retirement control"
            )
            assert any(
                resource is backend or getattr(resource, "backend", None) is backend
                for resource, _ in execution_cleanup_debts(caught)
            )
        else:
            assert not refused, (
                "standalone cleanup unnecessarily requires a new observer Task"
            )
            assert caught is not None
            assert agent._hold_store._db.backend._connection is None
        assert not agent._runtime_publication_ready.is_set()
    finally:
        loop.set_task_factory(previous)
        for outer, operation, backend in refused:
            outer.close()
            operation.close()
            if backend is not None:
                await backend.close()
        if agent._hold_store is not None:
            await agent._hold_store._db.close()
        await agent.shutdown()
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("startup_contention", [False, True], ids=["normal", "contended"])
async def test_actual_bootstrap_failure_does_not_mask_cleanup_withdrawal(
    tmp_path, monkeypatch, startup_contention,
):
    import aiosqlite

    from kestrel_sovereign._async_ownership import await_owned_task
    from kestrel_sovereign.host_features import build_host_context

    monkeypatch.setenv("KESTREL_DB_BACKEND", "sqlite")
    monkeypatch.setenv("KESTREL_HOLD_BACKEND", "sqlite")
    original_execute, original_close = (
        aiosqlite.Connection.execute,
        aiosqlite.Connection.close,
    )
    faulted, closed = [], []
    entered, release = asyncio.Event(), asyncio.Event()

    async def native_schema_error(connection, query, *args, **kwargs):
        if not faulted and "CREATE TABLE IF NOT EXISTS hold_latches" in query:
            faulted.append(connection)
            return await original_execute(
                connection, "SELECT * FROM nonexistent_bootstrap_fault_table"
            )
        return await original_execute(connection, query, *args, **kwargs)

    async def held_actual_close(connection):
        await original_close(connection)
        closed.append(connection)
        entered.set()
        await release.wait()

    monkeypatch.setattr(aiosqlite.Connection, "execute", native_schema_error)
    monkeypatch.setattr(aiosqlite.Connection, "close", held_actual_close)
    blocker = None
    release_startup = None
    bootstrap = None
    try:
        if startup_contention:
            # Real startup may wait on an existing native writer before reaching
            # the cleanup behavior under test. It has no three-second SLO.
            blocker = sqlite3.connect(tmp_path / "native-bootstrap.db")
            blocker.execute("PRAGMA journal_mode=WAL")
            blocker.execute("CREATE TABLE owned_startup_contention (id INTEGER)")
            blocker.execute("BEGIN IMMEDIATE")
            release_startup = asyncio.get_running_loop().call_later(3.5, blocker.rollback)
        bootstrap = asyncio.create_task(
            build_host_context(db_path=str(tmp_path / "native-bootstrap.db"))
        )
        await wait_for_lifecycle_checkpoint(
            entered.wait(), bootstrap, description="actual failed-bootstrap native close",
        )
        assert faulted and closed
        bootstrap.cancel("caller withdrew after actual failed schema")
        await asyncio.sleep(0)
        assert not bootstrap.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await bootstrap
        assert all(not getattr(conn, "_thread", conn).is_alive() for conn in closed)
    finally:
        release.set()
        if release_startup is not None:
            release_startup.cancel()
        if blocker is not None:
            blocker.rollback()
            blocker.close()
        if bootstrap is not None:
            await await_owned_task(bootstrap)


@pytest.mark.asyncio
async def test_original_unknown_stays_control_during_native_bootstrap_withdrawal(
    tmp_path, monkeypatch
):
    import aiosqlite

    from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
    from kestrel_sovereign.execution_custody import execution_commit_outcome
    from kestrel_sovereign.host_features import build_host_context
    from tests.unit.test_execution_custody_review43_native import carries

    monkeypatch.setenv("KESTREL_DB_BACKEND", "sqlite")
    monkeypatch.setenv("KESTREL_HOLD_BACKEND", "sqlite")
    native_commit, native_close = (
        aiosqlite.Connection.commit,
        aiosqlite.Connection.close,
    )
    commits, closed = [], []
    entered, release = asyncio.Event(), asyncio.Event()
    original = OSError("actual bootstrap COMMIT ACK lost before caller withdrawal")

    async def lose_first_commit_ack(connection):
        await native_commit(connection)
        if not commits:
            commits.append(connection)
            raise original

    async def hold_actual_close_ack(connection):
        await native_close(connection)
        closed.append(connection)
        entered.set()
        await release.wait()

    monkeypatch.setattr(aiosqlite.Connection, "commit", lose_first_commit_ack)
    monkeypatch.setattr(aiosqlite.Connection, "close", hold_actual_close_ack)
    bootstrap = create_owned_task(
        build_host_context(db_path=str(tmp_path / "unknown-bootstrap.db")),
        retain_originals=True,
    )
    try:
        await wait_for_lifecycle_checkpoint(
            entered.wait(), bootstrap, description="original unknown-bootstrap native close",
        )
        for message in (
            "first native bootstrap withdrawal",
            "second native bootstrap withdrawal",
        ):
            bootstrap.cancel(message)
            await asyncio.sleep(0)
            assert not bootstrap.done()
        release.set()
        outcome = await await_owned_task(bootstrap)
        assert execution_commit_outcome(outcome.error) == "unknown"
        assert carries(outcome.error, original)
        assert commits and closed
        assert all(not getattr(conn, "_thread", conn).is_alive() for conn in closed)
    finally:
        release.set()
        await await_owned_task(bootstrap)
