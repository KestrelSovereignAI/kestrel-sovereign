"""Original receipts through direct native retirement, reads and health."""
import asyncio
import sqlite3

import aiosqlite
import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.features import isolated_runtime
from kestrel_sovereign.storage.db import sqlite as sqlite_module
from kestrel_sovereign.storage.db.sqlite import (
    SQLiteBackend,
    _wait_for_aiosqlite_worker_shutdown,
)
from tests.unit.test_execution_custody_review48_native import terminal
from tests.unit.test_execution_custody_review49_native import KINDS, join


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
async def test_direct_late_close_receipt_survives_worker_exit(tmp_path, monkeypatch, kind):
    backend = SQLiteBackend(str(tmp_path / "effect.db"))
    await backend.connect()
    await backend.execute("CREATE TABLE effects(id INTEGER)")
    await backend.execute("INSERT INTO effects VALUES (1)")
    connection = backend._connection
    native_close = connection.close
    release = asyncio.Event()
    original = terminal(kind)
    monkeypatch.setattr(sqlite_module, "AIOSQLITE_WORKER_SHUTDOWN_TIMEOUT_S", 0.03)

    async def native_late_close():
        # A native close can settle after its bounded caller has handed off.
        await await_owned_task(create_owned_task(release.wait()))
        await native_close()
        raise original

    monkeypatch.setattr(connection, "close", native_late_close)
    try:
        first = await join(create_owned_task(backend.close()))
        assert first.error is not None
        records = list(backend._retired_connection_closes.values())
        assert records and backend.connection_retirement_pending
        release.set()
        for record in records:
            await join(record.close_task)
            if record.retirement_task is not None:
                await join(record.retirement_task)
        await _wait_for_aiosqlite_worker_shutdown(connection, timeout_s=1)
        assert not backend.connection_retirement_pending
        assert (await join(create_owned_task(backend.close()))).error is original
        assert (await join(create_owned_task(backend.close()))).error is None
    finally:
        release.set()
        monkeypatch.undo()
        await native_close()
        for record in list(backend._retired_connection_closes.values()):
            await join(record.close_task)
            if record.retirement_task is not None:
                await join(record.retirement_task)
        await _wait_for_aiosqlite_worker_shutdown(connection, timeout_s=1)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
async def test_native_snapshot_body_receipt_wins_over_connection_cleanup(tmp_path, monkeypatch, kind):
    backend = SQLiteBackend(str(tmp_path / "effect.db"))
    await backend.connect()
    await backend.execute("CREATE TABLE effects(id INTEGER)")
    await backend.execute("INSERT INTO effects VALUES (1)")
    entered, release = asyncio.Event(), asyncio.Event()

    async def transaction_owner():
        async with backend.transaction():
            entered.set()
            await release.wait()

    owner = create_owned_task(transaction_owner())
    await asyncio.wait_for(entered.wait(), 2)
    original = terminal(kind)
    native_open = backend._open_snapshot_read_connection
    snapshots = []

    async def original_snapshot():
        connection = await native_open()
        snapshots.append(connection)
        native_close = connection.close

        async def close_then_ordinary_error():
            await native_close()
            raise ValueError("later ordinary snapshot close")

        monkeypatch.setattr(connection, "close", close_then_ordinary_error)
        return connection

    async def native_read_then_receipt():
        async with backend._read_connection() as connection:
            assert connection is not backend._connection
            async with connection.execute("SELECT COUNT(*) FROM effects") as cursor:
                assert (await cursor.fetchone())[0] == 1
            raise original

    monkeypatch.setattr(backend, "_open_snapshot_read_connection", original_snapshot)
    try:
        assert (await join(create_owned_task(native_read_then_receipt()))).error is original
        for connection in snapshots:
            await _wait_for_aiosqlite_worker_shutdown(connection, timeout_s=1)
    finally:
        release.set()
        await join(owner)
        monkeypatch.undo()
        await backend.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["unknown", "committed", "group"])
async def test_native_fetch_receipt_wins_over_later_cursor_close(tmp_path, monkeypatch, kind):
    backend = SQLiteBackend(str(tmp_path / "effect.db"))
    await backend.connect()
    await backend.execute("CREATE TABLE effects(id INTEGER)")
    await backend.execute("INSERT INTO effects VALUES (1)")
    original = terminal(kind)
    fetch, close = aiosqlite.Cursor.fetchone, aiosqlite.Cursor.close
    effects = []

    async def native_fetch(cursor):
        effects.append(await fetch(cursor))
        raise original

    async def native_close(cursor):
        await close(cursor)
        raise ValueError("later ordinary cursor close")

    monkeypatch.setattr(aiosqlite.Cursor, "fetchone", native_fetch)
    monkeypatch.setattr(aiosqlite.Cursor, "close", native_close)
    try:
        outcome = await join(create_owned_task(backend._fetch_on_connection(
            backend._connection, "SELECT COUNT(*) FROM effects", (), one=True)))
        assert outcome.error is original
        assert effects and effects[0][0] == 1
    finally:
        monkeypatch.undo()
        await backend.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", ["pending", "abandoned"])
@pytest.mark.parametrize("kind", KINDS)
async def test_cancelled_native_cursor_control_cannot_be_retried_away(tmp_path, monkeypatch, lane, kind):
    backend = SQLiteBackend(str(tmp_path / "effect.db"))
    await backend.connect()
    await backend.execute("CREATE TABLE effects(id INTEGER)")
    await backend.execute("INSERT INTO effects VALUES (1)")
    connection = backend._connection
    task = create_owned_task(connection.execute("SELECT COUNT(*) FROM effects"))
    cursor = (await join(task)).result
    assert (await cursor.fetchone())[0] == 1
    original = sqlite3.OperationalError("original cursor close interrupted")
    original.__cause__ = terminal(kind)
    native_close = cursor.close
    calls = []

    async def native_close_then_receipt():
        await native_close()
        calls.append(True)
        if len(calls) == 1:
            raise original

    monkeypatch.setattr(cursor, "close", native_close_then_receipt)
    if lane == "pending":
        backend._pending_read_executions.add(task)
    else:
        backend._abandoned_read_cursors.add(cursor)
    try:
        backend._handoff_cancelled_write(connection)
        await join(backend._cancelled_write_drain)
        assert backend._cancelled_write_drain_error is original
        assert calls == [True]
    finally:
        monkeypatch.undo()
        await join(create_owned_task(backend.close()))


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", ["close", "drain"])
@pytest.mark.parametrize("kind", KINDS)
async def test_native_interrupt_control_survives_successful_retirement(tmp_path, monkeypatch, lane, kind):
    backend = SQLiteBackend(str(tmp_path / "effect.db"))
    await backend.connect()
    await backend.execute("CREATE TABLE effects(id INTEGER)")
    await backend.execute("INSERT INTO effects VALUES (1)")
    connection = backend._connection
    interrupt = connection.interrupt
    original = ValueError("original native interrupt acknowledgment")
    original.__cause__ = terminal(kind)
    calls = []

    async def native_interrupt_then_receipt():
        await interrupt()
        calls.append(True)
        raise original

    monkeypatch.setattr(connection, "interrupt", native_interrupt_then_receipt)
    try:
        if lane == "close":
            backend._active_write_operations = 1
            outcome = await join(create_owned_task(backend.close()))
            assert outcome.error is original
            assert backend._connection is None
            await _wait_for_aiosqlite_worker_shutdown(connection, timeout_s=1)
        else:
            backend._handoff_cancelled_write(connection)
            await join(backend._cancelled_write_drain)
            assert backend._cancelled_write_drain_error is original
        assert calls
    finally:
        monkeypatch.undo()
        backend._active_write_operations = 0
        await join(create_owned_task(backend.close()))


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", ["immediate", "timeout"])
@pytest.mark.parametrize("kind", KINDS)
async def test_health_probe_keeps_native_control_before_restart(tmp_path, monkeypatch, lane, kind):
    backend = SQLiteBackend(str(tmp_path / "health.db"))
    await backend.connect()
    await backend.execute("CREATE TABLE effects(id INTEGER)")
    original = terminal(kind)
    owned = []
    monkeypatch.setattr(isolated_runtime, "_HEALTH_PROBE_TIMEOUT", 0.02)
    monkeypatch.setattr(isolated_runtime, "_HEALTH_PROBE_CANCELLATION_GRACE", 0.2)

    async def native_health():
        if lane == "timeout":
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                pass
        await backend.execute("INSERT INTO effects VALUES (1)")
        raise original

    try:
        outcome = await join(create_owned_task(isolated_runtime._await_owned_health_probe(
            native_health(), name="review52-native", on_started=owned.append,
            on_late_task=owned.append)))
        assert outcome.error is original
        assert owned and all(child.done() for child in owned)
        assert await backend.fetch_val("SELECT COUNT(*) FROM effects") == 1
    finally:
        monkeypatch.undo()
        for child in owned:
            if not child.done():
                child.cancel()
        await backend.close()
