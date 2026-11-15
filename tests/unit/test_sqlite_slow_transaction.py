"""A slow SQLite transaction names the task that held the writer slot (#4081).

Every SQLite transaction takes the writer slot at ``BEGIN IMMEDIATE`` (#4040),
so one held open on a shared file holds back every other writer, the
Distributed Stop owner's lease renewal included. A wait for the slot says only
that someone held it; the holder's own warning says who.
"""

import asyncio
import logging
import re
import sqlite3
import time

import aiosqlite
import pytest

from kestrel_sovereign.storage.db.sqlite import SQLiteBackend

_SLOW_TRANSACTION_SECONDS = 0.5

_WARNING = re.compile(
    r"Slow SQLite transaction on (?P<file>\S+): task '(?P<task>[^']+)' waited "
    r"(?P<waited>[0-9.]+) s for the writer slot and "
    r"(?:held it (?P<held>[0-9.]+) s until (?P<until>[a-z ]+)"
    r"|(?P<unacquired>did not acquire it[a-z ]*)"
    r"(?:; its rollback ended (?P<rollback>[0-9.]+) s later)?)"
    r"\. Opened at:\n"
)


def _slow_transactions(caplog) -> dict[str, re.Match]:
    found: dict[str, re.Match] = {}
    for record in caplog.records:
        message = record.getMessage()
        if not message.startswith("Slow SQLite transaction"):
            continue
        match = _WARNING.match(message)
        assert match is not None, message
        assert match["task"] not in found, f"two warnings for {match['task']}"
        found[match["task"]] = match
    return found


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_slow_transaction_names_its_holder(tmp_path, caplog):
    caplog.set_level(logging.WARNING, logger="kestrel_sovereign.storage.db.sqlite")
    path = str(tmp_path / "shared-writer.db")
    holding = SQLiteBackend(path)
    waiting = SQLiteBackend(path)
    await holding.connect()
    await waiting.connect()
    try:
        await holding.execute("CREATE TABLE IF NOT EXISTS t (v INTEGER)")
        began = asyncio.Event()

        async def hold() -> None:
            async with holding.transaction():
                await holding.execute("INSERT INTO t VALUES (1)")
                began.set()
                await asyncio.sleep(0.8)

        async def wait_behind() -> None:
            await began.wait()
            async with waiting.transaction():
                await waiting.execute("INSERT INTO t VALUES (2)")

        caplog.clear()
        await asyncio.gather(
            asyncio.create_task(hold(), name="slot-holder"),
            asyncio.create_task(wait_behind(), name="slot-waiter"),
        )

        warnings = _slow_transactions(caplog)
        assert set(warnings) == {"slot-holder", "slot-waiter"}, warnings
        holder, waiter = warnings["slot-holder"], warnings["slot-waiter"]
        assert holder["file"] == "shared-writer.db"
        assert float(holder["held"]) >= _SLOW_TRANSACTION_SECONDS
        assert float(holder["waited"]) < _SLOW_TRANSACTION_SECONDS
        assert holder["until"] == "it was committed"
        # It waited behind the holder, then held the slot only briefly.
        assert float(waiter["waited"]) >= _SLOW_TRANSACTION_SECONDS
        assert float(waiter["held"]) < _SLOW_TRANSACTION_SECONDS / 2
        # The opener's stack ends at the code that opened the transaction.
        for match, opener in ((holder, "hold"), (waiter, "wait_behind")):
            message = match.string
            assert message.endswith(f", in {opener}"), message
            assert "INSERT" not in message and "VALUES" not in message

        caplog.clear()
        async with holding.transaction():
            await holding.execute("INSERT INTO t VALUES (3)")
            await asyncio.sleep(0.1)
        assert _slow_transactions(caplog) == {}
    finally:
        await waiting.close()
        await holding.close()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_a_slow_transaction_that_rolls_back_says_so(tmp_path, caplog):
    caplog.set_level(logging.WARNING, logger="kestrel_sovereign.storage.db.sqlite")
    backend = SQLiteBackend(str(tmp_path / "rolled-back.db"))
    await backend.connect()

    class _Refused(Exception):
        pass

    async def hold_then_fail() -> None:
        async with backend.transaction():
            await asyncio.sleep(_SLOW_TRANSACTION_SECONDS + 0.05)
            raise _Refused("refused")

    try:
        caplog.clear()
        with pytest.raises(Exception, match="refused"):
            await asyncio.create_task(hold_then_fail(), name="failing-holder")
        warnings = _slow_transactions(caplog)
        assert set(warnings) == {"failing-holder"}
        assert warnings["failing-holder"]["until"] == "it was rolled back"
    finally:
        await backend.close()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_the_opener_stack_carries_no_source_lines(tmp_path, caplog):
    """Frame locations only: a caller's source line here is a call whose
    argument is SQL text, and none of it reaches the warning."""

    caplog.set_level(logging.WARNING, logger="kestrel_sovereign.storage.db.sqlite")
    backend = SQLiteBackend(str(tmp_path / "no-source.db"))
    await backend.connect()

    async def write_slowly(statement: str) -> None:
        async with backend.transaction():
            await backend.execute(statement)
            await asyncio.sleep(_SLOW_TRANSACTION_SECONDS + 0.05)

    async def caller() -> None:
        await write_slowly("CREATE TABLE IF NOT EXISTS secret_rows (v INTEGER)")

    try:
        caplog.clear()
        await asyncio.create_task(caller(), name="sql-carrying-caller")
        (match,) = _slow_transactions(caplog).values()
        message = match.string
        assert ", in caller\n" in message, message
        assert message.endswith(", in write_slowly"), message
        assert "secret_rows" not in message and "CREATE" not in message, message
        stack = message.split("Opened at:\n", 1)[1].splitlines()
        assert all(line.startswith('  File "') for line in stack), stack
    finally:
        await backend.close()


def _delay_rollbacks_of(monkeypatch, backend: SQLiteBackend, seconds: float):
    """Make ``backend``'s connection take ``seconds`` before each rollback."""

    original = aiosqlite.Connection.rollback
    started: list[float] = []

    async def delayed(connection):
        if connection is backend._connection:
            started.append(time.monotonic())
            await asyncio.sleep(seconds)
        return await original(connection)

    monkeypatch.setattr(aiosqlite.Connection, "rollback", delayed)
    return started


async def _rollback_drained(backend: SQLiteBackend) -> None:
    async with asyncio.timeout(10):
        while backend._cancelled_write_drain is not None:
            await asyncio.sleep(0.01)


def _file_is_locked(path: str) -> bool:
    probe = sqlite3.connect(path, timeout=0, isolation_level=None)
    try:
        probe.execute("BEGIN IMMEDIATE")
    except sqlite3.OperationalError as error:
        assert "locked" in str(error), error
        return True
    else:
        probe.rollback()
        return False
    finally:
        probe.close()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_a_cancelled_transaction_is_timed_through_its_rollback(
    tmp_path, caplog, monkeypatch
):
    """A transaction cancelled after 0.1 s whose handed-off rollback takes
    0.7 s held the writer slot for 0.8 s, and its warning says so once the
    rollback has ended, naming the task and opener it captured at exit."""

    caplog.set_level(logging.WARNING, logger="kestrel_sovereign.storage.db.sqlite")
    path = str(tmp_path / "cancelled.db")
    backend = SQLiteBackend(path)
    await backend.connect()
    began = asyncio.Event()

    async def open_and_hold() -> None:
        async with backend.transaction():
            await backend.execute("CREATE TABLE IF NOT EXISTS t (v INTEGER)")
            began.set()
            await asyncio.sleep(30)

    try:
        caplog.clear()
        rollbacks = _delay_rollbacks_of(monkeypatch, backend, 0.7)
        holder = asyncio.create_task(open_and_hold(), name="cancelled-holder")
        await began.wait()
        await asyncio.sleep(0.1)
        holder.cancel()
        cancelled_at = time.monotonic()
        with pytest.raises(asyncio.CancelledError):
            await holder
        # Cancellation stays prompt: the rollback is not awaited by its task.
        assert time.monotonic() - cancelled_at < 0.5
        assert _slow_transactions(caplog) == {}
        # The rollback has begun and holds the slot: nothing is reported yet.
        async with asyncio.timeout(5):
            while not rollbacks:
                await asyncio.sleep(0.01)
        assert _file_is_locked(path)
        assert _slow_transactions(caplog) == {}

        await _rollback_drained(backend)
        assert not _file_is_locked(path)

        (match,) = _slow_transactions(caplog).values()
        assert match["task"] == "cancelled-holder"
        assert match["until"] == (
            "the rollback it handed off when it was cancelled ended"
        )
        assert float(match["held"]) >= 0.7, match.string
        assert float(match["waited"]) < _SLOW_TRANSACTION_SECONDS
        assert match.string.endswith(", in open_and_hold"), match.string
    finally:
        await backend.close()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_a_quickly_cancelled_transaction_with_a_quick_rollback_logs_nothing(
    tmp_path, caplog
):
    caplog.set_level(logging.WARNING, logger="kestrel_sovereign.storage.db.sqlite")
    backend = SQLiteBackend(str(tmp_path / "quick-cancel.db"))
    await backend.connect()
    began = asyncio.Event()

    async def open_and_hold() -> None:
        async with backend.transaction():
            began.set()
            await asyncio.sleep(30)

    try:
        caplog.clear()
        holder = asyncio.create_task(open_and_hold(), name="quick-cancel")
        await began.wait()
        await asyncio.sleep(0.1)
        holder.cancel()
        with pytest.raises(asyncio.CancelledError):
            await holder
        await _rollback_drained(backend)
        # The report runs as a done-callback of the rollback; let it run.
        await asyncio.sleep(0)
        assert _slow_transactions(caplog) == {}
    finally:
        await backend.close()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_a_transaction_cancelled_waiting_for_the_slot_names_its_rollback(
    tmp_path, caplog
):
    """Cancelled before BEGIN IMMEDIATE returned, a transaction held no slot it
    knew of: its one warning names the wait and how long its rollback took."""

    caplog.set_level(logging.WARNING, logger="kestrel_sovereign.storage.db.sqlite")
    path = str(tmp_path / "cancelled-waiting.db")
    backend = SQLiteBackend(path)
    await backend.connect()
    holder = sqlite3.connect(path, timeout=10, isolation_level=None)
    entered = asyncio.Event()

    async def wait_for_the_slot() -> None:
        entered.set()
        async with backend.transaction():
            await asyncio.sleep(30)

    try:
        caplog.clear()
        holder.execute("BEGIN IMMEDIATE")
        waiter = asyncio.create_task(wait_for_the_slot(), name="cancelled-waiter")
        await entered.wait()
        await asyncio.sleep(_SLOW_TRANSACTION_SECONDS + 0.1)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        # SQLite's worker may still be waiting in BEGIN until the slot frees.
        await asyncio.sleep(0.1)
        holder.rollback()
        await _rollback_drained(backend)
        await asyncio.sleep(0)

        (match,) = _slow_transactions(caplog).values()
        assert match["task"] == "cancelled-waiter"
        assert match["unacquired"] == "did not acquire it before it was cancelled"
        assert match["rollback"] is not None, match.string
        assert float(match["waited"]) >= _SLOW_TRANSACTION_SECONDS
        assert match.string.endswith(", in wait_for_the_slot"), match.string
    finally:
        holder.close()
        await backend.close()
