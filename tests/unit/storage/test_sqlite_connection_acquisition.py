"""Join actual SQLite creation before withdrawal, without fabricated close ACK."""

import asyncio
import sqlite3
import threading

import aiosqlite
import pytest

from kestrel_sovereign.execution_custody import execution_evidence_carriers, is_execution_control_error
from kestrel_sovereign.storage.db.interface import ConnectionError
from kestrel_sovereign.storage.db.sqlite import SQLiteBackend
from tests.unit.test_host_context_native_cleanup import _actual_operation
from tests.utils.aiosqlite_workers import aiosqlite_worker
from tests.utils.virtual_clock import HANG_GUARD_SECONDS, await_within_hang_guard, wait_for_thread_event


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["primary", "cold", "snapshot", "backup"])
@pytest.mark.parametrize("repeated", [False, True])
async def test_actual_connection_creation_is_joined_before_caller_withdrawal(tmp_path, monkeypatch, route, repeated):
    path = tmp_path / "original-creation.db"
    backend = SQLiteBackend(str(path))
    await backend.connect()
    if route in {"primary", "cold"}:
        await backend.close()
        backend = SQLiteBackend(str(path), cold_read=route == "cold")
        target = backend._cold_read_uri() if route == "cold" else str(path)
    else:
        target = path.as_uri() + "?mode=rw" if route == "snapshot" else str(tmp_path / "original-backup.db")
    native_connect, async_connect = sqlite3.connect, aiosqlite.connect
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()
    actual, wrappers = [], []
    observed = {"abandoned": False, "test_reconciled": False}

    def original_native_connect(database, *args, **kwargs):
        connection = native_connect(database, *args, **kwargs)
        if database != target:
            return connection
        actual.append(connection)
        entered.set()
        try:
            if not release.wait(HANG_GUARD_SECONDS):
                connection.close()
                raise TimeoutError("Original test SQLite creator was not released")
            if observed["abandoned"]:
                # Negative/mutation cleanup AFTER the failure snapshot, on the
                # actual creating worker. This is not a product retirement ACK.
                assert connection.execute("SELECT 1").fetchone() == (1,)
                connection.close()
                observed["test_reconciled"] = True
            return connection
        finally:
            returned.set()

    def original_async_connect(database, *args, **kwargs):
        connection = async_connect(database, *args, **kwargs)
        if database == target:
            wrappers.append(connection)
        return connection

    async def acquire():
        if route in {"primary", "cold"}:
            return await backend.connect()
        if route == "backup":
            return await backend.backup_to(target)
        return await backend._open_snapshot_read_connection()

    pending = None
    try:
        with monkeypatch.context() as fault:
            fault.setattr(sqlite3, "connect", original_native_connect)
            fault.setattr(aiosqlite, "connect", original_async_connect)
            pending = asyncio.create_task(acquire())
            await wait_for_thread_event(entered, step="the actual original SQLite creator holding its open handle")
            pending.cancel("Caller withdrew during original SQLite creation")
            # Deliver cancellation to the already parked caller before taking
            # the pre-release snapshot; no wall-clock completion guess.
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            if repeated:
                pending.cancel("Same caller withdrew again before creation returned")
                await asyncio.sleep(0)
                await asyncio.sleep(0)
            premature = pending.done()
            observed["abandoned"] = premature
            release.set()
            result = (await await_within_hang_guard(
                asyncio.gather(pending, return_exceptions=True), step="the original SQLite creation and retirement joining"
            ))[0]
            await wait_for_thread_event(returned, step="the exact native creator returning")
            if isinstance(result, aiosqlite.Connection):
                await result.close()  # explicit test recovery of a lost cancellation
            assert not premature, ("Caller escaped while its original SQLite creator still owned the handle", observed)
            assert isinstance(result, asyncio.CancelledError)
            assert len(actual) == len(wrappers) == 1 and not observed["test_reconciled"]
            assert not aiosqlite_worker(wrappers[0]).is_alive()
            # SQL execute checks thread affinity BEFORE closed state. This
            # native property checks the actual C connection's closed pointer
            # without substituting a wrapper flag or another SQL connection.
            with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                _ = actual[0].in_transaction
    finally:
        release.set()
        if pending is not None:
            await asyncio.gather(pending, return_exceptions=True)
        await backend.close()


@pytest.mark.asyncio
async def test_failed_actual_sqlite_creation_joins_its_original_worker(tmp_path, monkeypatch):
    directory = tmp_path / "not-a-database"
    directory.mkdir()
    backend = SQLiteBackend(str(directory))
    original, wrappers = aiosqlite.connect, []

    def observe(*args, **kwargs):
        connection = original(*args, **kwargs)
        wrappers.append(connection)
        return connection

    monkeypatch.setattr(aiosqlite, "connect", observe)
    try:
        with pytest.raises(ConnectionError) as failed:
            await backend.connect()
        assert any(isinstance(error, sqlite3.OperationalError) for error in execution_evidence_carriers(failed.value))
        assert len(wrappers) == 1 and not aiosqlite_worker(wrappers[0]).is_alive()
        assert backend._connection is None
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_refused_original_connection_owner_never_starts_a_sqlite_worker(tmp_path, monkeypatch):
    backend = SQLiteBackend(str(tmp_path / "refused-creation.db"))
    original, wrappers, refused = aiosqlite.connect, [], []
    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()
    refusal = RuntimeError("Original SQLite acquisition Task admission refused")

    def observe(*args, **kwargs):
        connection = original(*args, **kwargs)
        wrappers.append(connection)
        return connection

    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        code = getattr(operation, "cr_code", None)
        if code is not None and code.co_qualname == "SQLiteBackend._acquire_connection.<locals>.connect_original":
            refused.append(operation)
            raise refusal
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)

    monkeypatch.setattr(aiosqlite, "connect", observe)
    try:
        loop.set_task_factory(factory)
        with pytest.raises(ConnectionError) as failed:
            await backend.connect()
        loop.set_task_factory(previous)
        assert refused and is_execution_control_error(failed.value)
        assert any(error is refusal for error in execution_evidence_carriers(failed.value))
        assert len(wrappers) == 1 and aiosqlite_worker(wrappers[0]).ident is None
        assert not (tmp_path / "refused-creation.db").exists()
    finally:
        loop.set_task_factory(previous)
        await backend.close()
