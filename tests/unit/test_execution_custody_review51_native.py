"""Native receipt ownership beyond worker exit and repeated cancellation."""
import asyncio
import threading

import aiosqlite
import pytest
from sqlalchemy import event, text

from kestrel_sovereign._async_ownership import create_owned_task
from kestrel_sovereign.execution_custody import (
    _explicit_control_chain,
    ExecutionAuthorityError,
    is_execution_control_error,
)
from tests.utils.original_failure import OriginalFailure
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.db import sqlite as sqlite_module
from kestrel_sovereign.storage.db.sqlite import (
    _aiosqlite_worker_is_alive,
    _wait_for_aiosqlite_worker_shutdown,
)
from kestrel_sovereign.storage.sqla import make_session_factory
from tests.unit.test_execution_custody_review48_native import terminal
from tests.unit.test_execution_custody_review49_native import KINDS, join


def original_receipt(kind):
    if kind == "wrapped":
        error = RuntimeError("original native transport carrier")
        error.__cause__ = terminal("unknown")
        return error
    return terminal(kind)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["cancel_unknown", "group"])
async def test_late_stronger_raw_driver_receipt_is_unseen_after_weaker_publication(tmp_path, monkeypatch, kind):
    db = await AsyncDatabase.sqlite(str(tmp_path / "primary.db"))
    factory = make_session_factory(db)
    connections = [await aiosqlite.connect(str(tmp_path / f"driver-{i}.db")) for i in range(2)]
    weak, original = ExecutionAuthorityError("earlier driver admission loss"), terminal(kind)
    release = asyncio.Event()
    monkeypatch.setattr(sqlite_module, "AIOSQLITE_WORKER_SHUTDOWN_TIMEOUT_S", 0.05)
    for index, connection in enumerate(connections):
        await connection.execute("CREATE TABLE effects(id INTEGER)")
        await connection.execute("INSERT INTO effects VALUES (1)")
        await connection.commit()
        native_close = connection.close

        async def native_close_with_receipt(index=index, native_close=native_close):
            if index == 1:
                await release.wait()
            await native_close()
            raise original if index == 1 else weak

        monkeypatch.setattr(connection, "close", native_close_with_receipt)
    factory._sqlite_connections.extend(connections)
    try:
        assert (await join(create_owned_task(factory.close()))).error is weak
        release.set()
        for record in list(factory._retired_sqlite_closes.values()):
            await join(record.close_task)
            if record.retirement_task is not None:
                await join(record.retirement_task)
        for connection in connections:
            await _wait_for_aiosqlite_worker_shutdown(connection, timeout_s=1)
        assert not factory.retirement_pending
        with pytest.raises(type(original)) as raised:
            factory.finalize_retirement()
        assert raised.value is original
    finally:
        release.set()
        monkeypatch.undo()
        await join(create_owned_task(db.close()))


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", [*KINDS, "wrapped"])
async def test_driver_control_cause_is_not_replaced_by_ordinary_disposal(tmp_path, monkeypatch, kind):
    db = await AsyncDatabase.sqlite(str(tmp_path / "primary.db"))
    factory = make_session_factory(db)
    connection = await aiosqlite.connect(str(tmp_path / "driver.db"))
    await connection.execute("CREATE TABLE effects(id INTEGER)")
    await connection.execute("INSERT INTO effects VALUES (1)")
    await connection.commit()
    factory._sqlite_connections.append(connection)
    original = original_receipt(kind)
    original_cause = original.__cause__
    release, collecting = asyncio.Event(), asyncio.Event()
    native_close = connection.close
    collect = factory._close_live_sqlite_connections

    async def ordinary_disposal(*args, **kwargs):
        raise RuntimeError("ordinary engine disposal failed")

    async def native_close_with_receipt():
        await release.wait()
        await native_close()
        raise original

    async def original_collector(**kwargs):
        collecting.set()
        return await collect(**kwargs)

    monkeypatch.setattr(type(factory.engine), "dispose", ordinary_disposal)
    monkeypatch.setattr(connection, "close", native_close_with_receipt)
    monkeypatch.setattr(factory, "_close_live_sqlite_connections", original_collector)
    owner = create_owned_task(factory.close())
    try:
        await asyncio.wait_for(collecting.wait(), 2)
        release.set()
        outcome = await join(owner)
        assert outcome.error is original
        assert original.__cause__ is original_cause
        assert is_execution_control_error(outcome.error)
        assert not _aiosqlite_worker_is_alive(connection)
    finally:
        release.set()
        await join(owner)
        monkeypatch.undo()
        await join(create_owned_task(db.close()))


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["collector", "finalizer"])
@pytest.mark.parametrize("kind", ["cancel_unknown", "group"])
async def test_raw_close_receipt_survives_worker_liveness_pruning(tmp_path, monkeypatch, route, kind):
    db = await AsyncDatabase.sqlite(str(tmp_path / "primary.db"))
    factory = make_session_factory(db)
    checkouts, checkins = set(), set()
    def checkout(connection, record, proxy):
        checkouts.add(id(record))
    def checkin(connection, record):
        checkins.add(id(record))
    event.listen(factory.engine.sync_engine, "checkout", checkout)
    event.listen(factory.engine.sync_engine, "checkin", checkin)
    session = factory._async_session()
    await session.execute(text("CREATE TABLE effects(id INTEGER)"))
    original_connection = await session.connection()
    connection = factory._sqlite_connections[0]
    await connection.execute("INSERT INTO effects VALUES (1)")
    await connection.commit()
    original = terminal(kind)
    native_close, session_close = connection.close, session.close
    entered, release = asyncio.Event(), asyncio.Event()
    owner = None

    async def native_close_with_receipt():
        await native_close()
        entered.set()
        raise original

    async def delayed_session_close(*args, **kwargs):
        await release.wait()
        await session.invalidate()
        await session_close(*args, **kwargs)

    monkeypatch.setattr(connection, "close", native_close_with_receipt)
    monkeypatch.setattr(session, "close", delayed_session_close)
    try:
        if route == "collector":
            factory._active_sessions.add(session)
            owner = create_owned_task(factory.close())
        else:
            factory._close_started = True
            factory._engine_dispose_task = create_owned_task(factory.engine.dispose())
            factory._begin_sqlite_connection_retirement()
        await asyncio.wait_for(entered.wait(), 2)
        records = list(factory._retired_sqlite_closes.values())
        for record in records:
            await join(record.close_task)
        await _wait_for_aiosqlite_worker_shutdown(connection, timeout_s=1)
        assert not factory.sqlite_connection_retirement_pending
        release.set()
        if route == "collector":
            assert (await join(owner)).error is original
        else:
            with pytest.raises(type(original)) as raised:
                factory.finalize_retirement()
            assert raised.value is original
    finally:
        try:
            release.set()
            if owner is not None:
                await join(owner)
            monkeypatch.undo()
            factory._active_sessions.discard(session)
            # Session.invalidate can drop its own transaction bookkeeping
            # before a fault-injected native closer raises BaseException.
            # Retire the SAME retained public connection after removing that
            # injection; never synthesize a pool check-in or native close ACK.
            await original_connection.invalidate()
            await session.invalidate()
            await session_close()
            await join(create_owned_task(db.close()))
        finally:
            event.remove(factory.engine.sync_engine, "checkout", checkout)
            event.remove(factory.engine.sync_engine, "checkin", checkin)
    assert checkouts and checkouts == checkins, "original checkout event has no matching physical check-in"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
async def test_repeated_cancellation_preserves_close_receipt_and_real_worker_owner(tmp_path, monkeypatch, kind):
    loop = asyncio.get_running_loop()
    native_worker = aiosqlite.core._connection_worker_thread
    worker_stop_seen, release_worker = asyncio.Event(), threading.Event()

    def native_worker_with_held_final_return(*args):
        native_worker(*args)
        loop.call_soon_threadsafe(worker_stop_seen.set)
        assert release_worker.wait(3), "bounded native worker exit gate was not released"

    monkeypatch.setattr(aiosqlite.core, "_connection_worker_thread", native_worker_with_held_final_return)
    db = await AsyncDatabase.sqlite(str(tmp_path / "worker.db"))
    connection = db.backend._connection
    await db.execute("CREATE TABLE effects(id INTEGER)")
    await db.execute("INSERT INTO effects VALUES (1)")
    original, native_close = terminal(kind), connection.close
    original_source = OriginalFailure.capture(original)
    withdrawals = []
    actual_worker_wait = sqlite_module._wait_for_aiosqlite_worker_shutdown
    async def worker_wait(*args, **kwargs):
        try:
            return await actual_worker_wait(*args, **kwargs)
        except asyncio.CancelledError as error:
            withdrawals.append(error)
            raise
    monkeypatch.setattr(sqlite_module, "_wait_for_aiosqlite_worker_shutdown", worker_wait)

    async def close_then_receipt():
        await native_close()
        raise original

    monkeypatch.setattr(connection, "close", close_then_receipt)
    owner = create_owned_task(db.backend.close())
    try:
        await asyncio.wait_for(worker_stop_seen.wait(), 2)
        await asyncio.sleep(0.02)
        assert _aiosqlite_worker_is_alive(connection)
        owner.cancel()
        await asyncio.sleep(0.02)
        owner.cancel()
        outcome = await join(owner)
        original_source.assert_retained(outcome.error)
        assert withdrawals and all(error in tuple(_explicit_control_chain(outcome.error)) for error in withdrawals)
        assert connection in db.backend._retired_connection_closes
    finally:
        release_worker.set()
        await join(owner)
        await _wait_for_aiosqlite_worker_shutdown(connection, timeout_s=1)
        for record in list(db.backend._retired_connection_closes.values()):
            if record.retirement_task is not None:
                await join(record.retirement_task)
        monkeypatch.undo()
        if db.backend._connection is not None:
            await join(create_owned_task(db.backend.close()))
    assert not _aiosqlite_worker_is_alive(connection)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["cancel_unknown", "cancel_committed"])
async def test_shared_native_read_returns_original_child_cancellation_cause(tmp_path, monkeypatch, kind):
    db = await AsyncDatabase.sqlite(str(tmp_path / "read.db"))
    backend, native_execute = db.backend, db.backend._connection.execute
    await db.execute("CREATE TABLE effects(id INTEGER)")
    await db.execute("INSERT INTO effects VALUES (1)")
    original = terminal("cancel_unknown")
    if kind == "cancel_committed":
        original.__cause__ = terminal("committed")
    cause, observed = original.__cause__, []

    async def native_read_then_receipt(query, *args, **kwargs):
        cursor = await native_execute(query, *args, **kwargs)
        if query == "SELECT COUNT(*) FROM effects":
            observed.append((await cursor.fetchone())[0])
            await cursor.close()
            raise original
        return cursor

    monkeypatch.setattr(backend._connection, "execute", native_read_then_receipt)
    try:
        outcome = await join(create_owned_task(db.fetchval("SELECT COUNT(*) FROM effects")))
        assert observed == [1]
        assert outcome.error is original
        assert outcome.error.__cause__ is cause
        assert is_execution_control_error(outcome.error)
    finally:
        monkeypatch.undo()
        if backend._cancelled_write_drain is not None:
            await join(backend._cancelled_write_drain)
        await join(create_owned_task(db.close()))
    assert backend._connection is None
