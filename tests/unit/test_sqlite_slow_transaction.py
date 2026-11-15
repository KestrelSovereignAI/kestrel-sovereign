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
    r"(?:; its rollback returned (?P<rollback>[0-9.]+) s later)?)"
    r"\. Opened at:\n"
)
# A transaction whose release was never confirmed, logged at any duration.
_UNCONFIRMED = re.compile(
    r"SQLite transaction on (?P<file>\S+) may still hold the writer slot: task "
    r"'(?P<task>[^']+)' waited (?P<waited>[0-9.]+) s for it and "
    r"(?:held it at least (?P<held>[0-9.]+) s; its release"
    r"|did not acquire it before it was cancelled; the release of a slot it "
    r"may have taken then)"
    r" was not confirmed \((?P<failure>\w+)\)\. Opened at:\n"
)


def _transaction_warnings(caplog) -> list[str]:
    """Every warning a transaction logged about the writer slot."""

    return [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith(
            ("Slow SQLite transaction", "SQLite transaction on ")
        )
    ]


def _slow_transactions(caplog) -> dict[str, re.Match]:
    found: dict[str, re.Match] = {}
    for message in _transaction_warnings(caplog):
        if not message.startswith("Slow SQLite transaction"):
            continue
        match = _WARNING.match(message)
        assert match is not None, message
        assert match["task"] not in found, f"two warnings for {match['task']}"
        found[match["task"]] = match
    return found


def _unconfirmed_releases(caplog) -> list[re.Match]:
    found = []
    for message in _transaction_warnings(caplog):
        if message.startswith("Slow SQLite transaction"):
            continue
        match = _UNCONFIRMED.match(message)
        assert match is not None, message
        found.append(match)
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
        assert _transaction_warnings(caplog) == []
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
    0.7 s held the writer slot for 0.8 s, and its warning says so once that
    rollback has returned, naming the task and opener it captured at exit.

    Mutation (finding 1 of the #4081 review): stamp the release at the
    handoff, which reports a 0.1 s hold and logs nothing."""

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
        assert _transaction_warnings(caplog) == []
        # The rollback has begun and holds the slot: nothing is reported yet.
        async with asyncio.timeout(5):
            while not rollbacks:
                await asyncio.sleep(0.01)
        assert _file_is_locked(path)
        assert _transaction_warnings(caplog) == []

        await _rollback_drained(backend)
        assert not _file_is_locked(path)

        (match,) = _slow_transactions(caplog).values()
        assert match["task"] == "cancelled-holder"
        assert match["until"] == (
            "the rollback it handed off when it was cancelled returned"
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
        assert _transaction_warnings(caplog) == []
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


class _Refused(Exception):
    pass


async def _refuse_rollbacks(backend: SQLiteBackend) -> dict[str, bool]:
    """Install an authorizer on ``backend``'s connection that, once its
    switch is on, refuses every ROLLBACK, so SQLite keeps the transaction open
    and the writer slot taken."""

    switch = {"on": False}

    def authorize(action, first, *_):
        if (
            switch["on"]
            and action == sqlite3.SQLITE_TRANSACTION
            and first == "ROLLBACK"
        ):
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    await backend._connection.set_authorizer(authorize)
    return switch


async def _end_refused_transaction(backend: SQLiteBackend, switch) -> None:
    switch["on"] = False
    await backend._connection.rollback()
    await backend._connection.set_authorizer(None)


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_refused_rollback_is_not_reported_as_released(tmp_path, caplog):
    """P7: a transaction that held the slot under 0.5 s, failed, and had its
    ROLLBACK refused still holds the slot. Its one warning says the release
    was not confirmed, whatever the durations, and never says rolled back.

    Mutation: stamp the release in ``finally`` whatever the rollback did, as
    the code did before; the transaction then reports nothing."""

    caplog.set_level(logging.WARNING, logger="kestrel_sovereign.storage.db.sqlite")
    path = str(tmp_path / "refused-rollback.db")
    backend = SQLiteBackend(path)
    await backend.connect()
    switch = await _refuse_rollbacks(backend)
    await backend.execute("CREATE TABLE IF NOT EXISTS t (v INTEGER)")

    async def fail_quickly() -> None:
        async with backend.transaction():
            await backend.execute("INSERT INTO t VALUES (1)")
            switch["on"] = True
            raise _Refused("refused")

    try:
        caplog.clear()
        with pytest.raises(Exception, match="refused"):
            await asyncio.create_task(fail_quickly(), name="refused-rollback")
        # The release did not happen: no other connection can take the slot.
        assert _file_is_locked(path)

        assert len(_transaction_warnings(caplog)) == 1, _transaction_warnings(caplog)
        (match,) = _unconfirmed_releases(caplog)
        message = match.string
        assert "rolled back" not in message and "released" not in message
        assert match["task"] == "refused-rollback"
        assert match["file"] == "refused-rollback.db"
        assert float(match["held"]) < _SLOW_TRANSACTION_SECONDS
        # The rollback's own failure is named, by type only.
        assert match["failure"] == "DatabaseError", message
        assert "not authorized" not in message
        assert message.endswith(", in fail_quickly"), message
    finally:
        await _end_refused_transaction(backend, switch)
        await backend.close()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_failed_handed_off_rollback_is_not_reported_as_released(
    tmp_path, caplog
):
    """P8: a cancelled transaction hands its rollback to the backend, whose
    ROLLBACK is refused. The drain ends normally, having recorded the failure;
    the transaction's one warning says its release was not confirmed.

    Mutation: report when the drain ends without reading whether its ROLLBACK
    returned; the quick transaction then reports nothing."""

    caplog.set_level(logging.WARNING, logger="kestrel_sovereign.storage.db.sqlite")
    path = str(tmp_path / "refused-drain.db")
    backend = SQLiteBackend(path)
    await backend.connect()
    switch = await _refuse_rollbacks(backend)
    await backend.execute("CREATE TABLE IF NOT EXISTS t (v INTEGER)")
    began = asyncio.Event()

    async def open_and_hold() -> None:
        async with backend.transaction():
            await backend.execute("INSERT INTO t VALUES (1)")
            began.set()
            await asyncio.sleep(30)

    try:
        caplog.clear()
        holder = asyncio.create_task(open_and_hold(), name="refused-drain")
        await began.wait()
        switch["on"] = True
        holder.cancel()
        with pytest.raises(asyncio.CancelledError):
            await holder
        await _rollback_drained(backend)
        # The report runs as a done-callback of the drain; let it run.
        await asyncio.sleep(0)
        assert backend._cancelled_write_drain_error is not None
        assert _file_is_locked(path)

        assert len(_transaction_warnings(caplog)) == 1, _transaction_warnings(caplog)
        (match,) = _unconfirmed_releases(caplog)
        assert "rolled back" not in match.string
        assert "released" not in match.string
        assert match["task"] == "refused-drain"
        assert match["failure"] == "DatabaseError", match.string
        assert float(match["held"]) < _SLOW_TRANSACTION_SECONDS
        assert match.string.endswith(", in open_and_hold"), match.string
    finally:
        await _end_refused_transaction(backend, switch)
        await backend.close()


def _gate_close_of(monkeypatch, backend: SQLiteBackend) -> asyncio.Event:
    """Hold ``backend.close()`` before it closes its connection, after it has
    begun closing, until the returned event is set."""

    from kestrel_sovereign.storage.db import sqlite as sqlite_module

    gate = asyncio.Event()
    original = sqlite_module._close_aiosqlite_connection
    primary = backend._connection

    async def gated(connection, **kwargs):
        if connection is primary:
            await gate.wait()
        return await original(connection, **kwargs)

    monkeypatch.setattr(sqlite_module, "_close_aiosqlite_connection", gated)
    return gate


async def _close_until_closing(backend: SQLiteBackend) -> "asyncio.Task[None]":
    closer = asyncio.create_task(backend.close(), name="closer")
    async with asyncio.timeout(5):
        while not backend._closing:
            await asyncio.sleep(0.005)
    return closer


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_cancel_during_close_follows_the_threshold(
    tmp_path, caplog, monkeypatch
):
    """P9: a transaction cancelled while the backend closes leaves its
    rollback to close, which declines the handoff. Close owns that release,
    so a quick one logs nothing: shutdown must not log one warning for every
    transaction it cancels.

    Mutation: report a declined handoff as an unconfirmed release whatever
    the backend was doing; the quick transaction then logs a warning."""

    caplog.set_level(logging.WARNING, logger="kestrel_sovereign.storage.db.sqlite")
    backend = SQLiteBackend(str(tmp_path / "cancel-in-close.db"))
    await backend.connect()
    began = asyncio.Event()

    async def open_and_hold() -> None:
        async with backend.transaction():
            await backend.execute("CREATE TABLE IF NOT EXISTS t (v INTEGER)")
            began.set()
            await asyncio.sleep(30)

    caplog.clear()
    holder = asyncio.create_task(open_and_hold(), name="cancelled-in-close")
    await began.wait()
    gate = _gate_close_of(monkeypatch, backend)
    closer = await _close_until_closing(backend)
    try:
        holder.cancel()
        with pytest.raises(asyncio.CancelledError):
            await holder
        # Close declined the handoff: no drain took the rollback.
        assert backend._cancelled_write_drain is None
    finally:
        gate.set()
        await closer
    assert _transaction_warnings(caplog) == []


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_a_commit_cancelled_during_close_is_left_to_close(
    tmp_path, caplog, monkeypatch
):
    """A transaction that held the slot 0.6 s, then was cancelled in its
    COMMIT while the backend closed, had its rollback declined: its cleanup
    returned without a ROLLBACK and the transaction is still open. It is
    reported as cancelled during close, not as ended by its commit.

    Mutation: take the cleanup's return as the end of the transaction without
    asking whether it is still open; it is then reported as ended."""

    caplog.set_level(logging.WARNING, logger="kestrel_sovereign.storage.db.sqlite")
    backend = SQLiteBackend(str(tmp_path / "commit-in-close.db"))
    await backend.connect()
    in_commit = asyncio.Event()
    never = asyncio.Event()
    original_commit = aiosqlite.Connection.commit

    async def blocked_commit(connection):
        if connection is backend._connection:
            in_commit.set()
            await never.wait()
        return await original_commit(connection)

    async def hold_then_commit() -> None:
        async with backend.transaction():
            await backend.execute("CREATE TABLE IF NOT EXISTS t (v INTEGER)")
            await asyncio.sleep(_SLOW_TRANSACTION_SECONDS + 0.1)

    caplog.clear()
    monkeypatch.setattr(aiosqlite.Connection, "commit", blocked_commit)
    holder = asyncio.create_task(hold_then_commit(), name="commit-in-close")
    async with asyncio.timeout(5):
        await in_commit.wait()
    gate = _gate_close_of(monkeypatch, backend)
    closer = await _close_until_closing(backend)
    try:
        holder.cancel()
        with pytest.raises(Exception, match="outcome is unknown"):
            await holder
        assert backend._cancelled_write_drain is None
    finally:
        gate.set()
        await closer

    assert len(_transaction_warnings(caplog)) == 1, _transaction_warnings(caplog)
    (match,) = _slow_transactions(caplog).values()
    assert match["task"] == "commit-in-close"
    assert match["until"] == "it was cancelled during close", match.string
    assert float(match["held"]) >= _SLOW_TRANSACTION_SECONDS


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_a_failed_commit_is_not_reported_as_a_rollback(
    tmp_path, caplog, monkeypatch
):
    """A COMMIT that raises may have taken effect: the transaction ended,
    rolled back or not, and its warning says only that.

    Mutation: report every rollback after a failure as "rolled back"."""

    caplog.set_level(logging.WARNING, logger="kestrel_sovereign.storage.db.sqlite")
    backend = SQLiteBackend(str(tmp_path / "failed-commit.db"))
    await backend.connect()
    original_commit = aiosqlite.Connection.commit

    async def refused_commit(connection):
        if connection is backend._connection:
            raise sqlite3.OperationalError("disk I/O error")
        return await original_commit(connection)

    async def hold_then_commit() -> None:
        async with backend.transaction():
            await backend.execute("CREATE TABLE IF NOT EXISTS t (v INTEGER)")
            await asyncio.sleep(_SLOW_TRANSACTION_SECONDS + 0.05)

    try:
        caplog.clear()
        monkeypatch.setattr(aiosqlite.Connection, "commit", refused_commit)
        with pytest.raises(Exception, match="outcome is unknown"):
            await asyncio.create_task(hold_then_commit(), name="failed-commit")
        monkeypatch.setattr(aiosqlite.Connection, "commit", original_commit)
        assert not backend._connection.in_transaction

        assert len(_transaction_warnings(caplog)) == 1, _transaction_warnings(caplog)
        (match,) = _slow_transactions(caplog).values()
        assert match["until"] == "it ended after its commit raised", match.string
    finally:
        await backend.close()


@pytest.mark.parametrize("raises", ["original", "rollback", "wrapper"])
@pytest.mark.parametrize(
    "rollback_failure", [sqlite3.DatabaseError, asyncio.CancelledError]
)
def test_an_unconfirmed_release_names_the_rollbacks_own_failure(
    raises, rollback_failure
):
    """The cleanup after a failure raises the rollback's own exception, the
    original failure, or a wrapper it selected; each names the rollback's.

    Mutation: name the rollback's failure only when the original came back."""

    from kestrel_sovereign.storage.db.sqlite import _cleanup_failure

    class _Wrapper(Exception):
        pass

    def cleanup(original: BaseException) -> None:
        try:
            raise rollback_failure()
        except BaseException as failure:
            selected = {
                "original": original,
                "rollback": failure,
                "wrapper": _Wrapper(),
            }[raises]
            if selected is not failure:
                raise selected
            raise

    try:
        raise _Refused("refused")
    except _Refused as original:
        try:
            cleanup(original)
        except BaseException as raised:
            assert isinstance(_cleanup_failure(raised, original), rollback_failure)


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_an_unconfirmed_hold_after_a_commit_is_bounded_by_its_start(
    tmp_path, caplog, monkeypatch
):
    """From the moment its COMMIT is issued a transaction may have ended on
    the worker, seen or not. A COMMIT that raises after 0.6 s, followed by a
    refused ROLLBACK, leaves the release unconfirmed, and the hold it was
    certainly open for ends where the COMMIT began, not where its rollback
    did.

    Mutation: measure the hold to the rollback's start after a commit too."""

    caplog.set_level(logging.WARNING, logger="kestrel_sovereign.storage.db.sqlite")
    path = str(tmp_path / "slow-failed-commit.db")
    backend = SQLiteBackend(path)
    await backend.connect()
    switch = await _refuse_rollbacks(backend)
    original_commit = aiosqlite.Connection.commit

    async def slow_failing_commit(connection):
        if connection is backend._connection:
            await asyncio.sleep(_SLOW_TRANSACTION_SECONDS + 0.1)
            switch["on"] = True
            raise sqlite3.OperationalError("disk I/O error")
        return await original_commit(connection)

    async def commit_slowly() -> None:
        async with backend.transaction():
            await backend.execute("CREATE TABLE IF NOT EXISTS t (v INTEGER)")

    try:
        caplog.clear()
        monkeypatch.setattr(aiosqlite.Connection, "commit", slow_failing_commit)
        # The cleanup keeps the commit's uncertain outcome and the refused
        # rollback together, as a group.
        with pytest.raises(Exception):
            await asyncio.create_task(commit_slowly(), name="commit-then-refused")
        monkeypatch.setattr(aiosqlite.Connection, "commit", original_commit)
        assert _file_is_locked(path)

        assert len(_transaction_warnings(caplog)) == 1, _transaction_warnings(caplog)
        (match,) = _unconfirmed_releases(caplog)
        assert match["failure"] == "DatabaseError", match.string
        assert float(match["held"]) < _SLOW_TRANSACTION_SECONDS, match.string
    finally:
        await _end_refused_transaction(backend, switch)
        await backend.close()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_a_refusal_before_the_commit_is_issued_is_a_rollback(
    tmp_path, caplog, monkeypatch
):
    """The commit step can refuse before it issues a COMMIT. Nothing could
    have committed, so its rolled-back transaction says so.

    Mutation: word every failure in the commit step as a commit that raised."""

    caplog.set_level(logging.WARNING, logger="kestrel_sovereign.storage.db.sqlite")
    backend = SQLiteBackend(str(tmp_path / "refused-before-commit.db"))
    await backend.connect()

    def refuse_commit_scopes():
        raise ConnectionError("refused before COMMIT")

    async def hold_then_commit() -> None:
        async with backend.transaction():
            await backend.execute("CREATE TABLE IF NOT EXISTS t (v INTEGER)")
            await asyncio.sleep(_SLOW_TRANSACTION_SECONDS + 0.05)
            monkeypatch.setattr(backend, "_execution_commit_scopes", refuse_commit_scopes)

    try:
        caplog.clear()
        with pytest.raises(Exception, match="refused before COMMIT"):
            await asyncio.create_task(hold_then_commit(), name="refused-before-commit")
        monkeypatch.undo()
        assert not backend._connection.in_transaction

        assert len(_transaction_warnings(caplog)) == 1, _transaction_warnings(caplog)
        (match,) = _slow_transactions(caplog).values()
        assert match["until"] == "it was rolled back", match.string
    finally:
        await backend.close()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_a_failure_that_ended_the_transaction_bounds_nothing_after_it(
    tmp_path, caplog, monkeypatch
):
    """``INSERT OR ROLLBACK`` that violates a constraint rolls the whole
    transaction back inside SQLite. A body that goes on 0.6 s past it and then
    fails had no transaction open for those 0.6 s, so when its own rollback
    then raises, the hold it is certain of is none of that time.

    Mutation: measure the unconfirmed hold to the rollback's start whether or
    not the transaction was still open then."""

    caplog.set_level(logging.WARNING, logger="kestrel_sovereign.storage.db.sqlite")
    backend = SQLiteBackend(str(tmp_path / "ended-by-its-failure.db"))
    await backend.connect()
    await backend.execute("CREATE TABLE IF NOT EXISTS t (v INTEGER PRIMARY KEY)")
    await backend.execute("INSERT INTO t VALUES (1)")
    original_rollback = aiosqlite.Connection.rollback
    ended_by_sqlite = []

    async def failing_rollback(connection):
        if connection is backend._connection:
            raise sqlite3.OperationalError("disk I/O error")
        return await original_rollback(connection)

    async def fail_after_sqlite_rolled_back() -> None:
        async with backend.transaction():
            with pytest.raises(Exception, match="UNIQUE"):
                await backend.execute("INSERT OR ROLLBACK INTO t VALUES (1)")
            ended_by_sqlite.append(not backend._connection.in_transaction)
            await asyncio.sleep(_SLOW_TRANSACTION_SECONDS + 0.1)
            monkeypatch.setattr(aiosqlite.Connection, "rollback", failing_rollback)
            raise _Refused("refused")

    try:
        caplog.clear()
        with pytest.raises(Exception, match="refused"):
            await asyncio.create_task(
                fail_after_sqlite_rolled_back(), name="ended-by-its-failure"
            )
        monkeypatch.undo()
        assert ended_by_sqlite == [True]

        assert len(_transaction_warnings(caplog)) == 1, _transaction_warnings(caplog)
        (match,) = _unconfirmed_releases(caplog)
        assert match["failure"] == "OperationalError", match.string
        assert float(match["held"]) < _SLOW_TRANSACTION_SECONDS, match.string
    finally:
        await backend.close()
