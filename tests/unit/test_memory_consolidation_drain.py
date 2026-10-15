"""Work a timed-out consolidation pass leaves running is settled before it can overlap (#2978).

The configured deadline cancels the pass, but an aiosqlite statement already
running on the connection's worker thread keeps running. These tests drive a
real ``SQLiteBackend`` whose worker is held inside a user-defined SQL function,
so "still running" is a fact the test controls rather than a timing guess, and
check the contract ``MemorySystem.consolidate`` now keeps: the MEMORY lock is
released only once that work has finished, or once the connection is isolated
so a later pass cannot use it; a durable record tracks the work until it
settles.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from kestrel_sdk.signals import ResourceLock
from kestrel_sdk.tools.result import ToolResultStatus

from kestrel_sovereign.features.health.checks import (
    check_database,
    check_memory_system,
)
from kestrel_sovereign.features.memory.feature import MemoryFeature
from kestrel_sovereign.signals.lock_manager import OrderedLockManager
from kestrel_sovereign.storage import consolidation_drain as drain_module
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.consolidation_drain import (
    DRAIN_RECORD_SUFFIX,
    read_consolidation_drain_record,
)
from kestrel_sovereign.storage.db.interface import ConnectionError, QueryError
from kestrel_sovereign.storage.db.sqlite import SQLiteBackend
from kestrel_sovereign.storage.memory_system import (
    ABANDONED_WORK_FAILED,
    ABANDONED_WORK_FINISHED,
    ABANDONED_WORK_ISOLATED,
    ABANDONED_WORK_NONE,
    ABANDONED_WORK_RETIRED,
    MemoryConsolidationTimeoutError,
    MemorySystem,
)

DEADLINE_SECONDS = 0.05
# The watchdog only rescues a regression from hanging the suite; every test
# releases the worker itself, on an event, long before it fires.
WATCHDOG_SECONDS = 5.0
# Likewise for every event a test awaits: it fails a regression that would
# never set the event, and never orders anything.
HANG_GUARD_SECONDS = 30.0


async def _until(awaitable):
    """Await ``awaitable`` under the hang guard."""
    async with asyncio.timeout(HANG_GUARD_SECONDS):
        return await awaitable


class _BlockedWorker:
    """A real SQLite database whose aiosqlite worker the test holds and frees.

    ``hold(value)`` is a SQL function that parks the worker thread until
    :meth:`release`; ``mark(tag)`` records that a statement carrying ``tag``
    executed. ``log`` is the worker-side order of both, which is the order
    the statements actually ran on the connection.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.backend = SQLiteBackend(str(path))
        self.db = AsyncDatabase(self.backend)
        self.log: list[str] = []
        self.entered = asyncio.Event()
        self.returned = threading.Event()
        self._release = threading.Event()
        self._watchdog = threading.Timer(WATCHDOG_SECONDS, self._release.set)

    async def open(self) -> "_BlockedWorker":
        loop = asyncio.get_running_loop()

        def hold(value):
            self.log.append("abandoned:enter")
            loop.call_soon_threadsafe(self.entered.set)
            self._release.wait()
            self.log.append("abandoned:return")
            self.returned.set()
            return value + 1

        def mark(tag):
            self.log.append(f"{tag}:write")
            return tag

        await self.backend.connect()
        conn = self.backend._ensure_connected()
        await conn.create_function("hold", 1, hold)
        await conn.create_function("mark", 1, mark)
        await self.db.execute("CREATE TABLE memories (value INTEGER NOT NULL)")
        await self.db.execute("INSERT INTO memories (value) VALUES (0)")
        await self.db.execute("CREATE TABLE later_pass (tag TEXT NOT NULL)")
        self._watchdog.start()
        return self

    async def abandoned_write(self) -> None:
        """The write a deadline cancels while the worker is inside it."""
        await self.db.execute("UPDATE memories SET value = hold(value)")

    async def later_write(self, tag: str) -> None:
        await self.db.execute(
            "INSERT INTO later_pass (tag) VALUES (mark(?))", (tag,)
        )

    def release(self) -> None:
        self._release.set()

    @property
    def record_path(self) -> Path:
        return self.path.with_name(self.path.name + DRAIN_RECORD_SUFFIX)

    def record(self) -> dict:
        return json.loads(self.record_path.read_text())

    async def close(self) -> None:
        self._watchdog.cancel()
        self._release.set()
        await self.backend.close()


@pytest.fixture
async def worker(tmp_path):
    blocked = await _BlockedWorker(tmp_path / "agent.db").open()
    try:
        yield blocked
    finally:
        await blocked.close()


class _RecordWatch:
    """Wake the test when the drain record reaches a state, without polling."""

    def __init__(self, monkeypatch) -> None:
        self.states: list[str] = []
        self._events: dict[str, asyncio.Event] = {}
        real = drain_module.write_consolidation_drain_record

        def writer(path, record):
            real(path, record)
            self.states.append(record["state"])
            self.event(record["state"]).set()

        monkeypatch.setattr(
            drain_module, "write_consolidation_drain_record", writer
        )

    def event(self, state: str) -> asyncio.Event:
        return self._events.setdefault(state, asyncio.Event())

    async def reached(self, state: str, *, unless_done: asyncio.Task) -> None:
        """Wait for ``state``; fail rather than hang if ``unless_done`` ends first."""
        waiter = asyncio.ensure_future(self.event(state).wait())
        done, _ = await _until(
            asyncio.wait(
                {waiter, unless_done}, return_when=asyncio.FIRST_COMPLETED
            )
        )
        waiter.cancel()
        assert waiter in done, (
            f"the drain record never reached {state!r} before the pass ended; "
            f"states written: {self.states}"
        )


class _ObservedLocks(OrderedLockManager):
    """An ordinary lock manager that also says when an acquisition is requested."""

    def __init__(self) -> None:
        super().__init__()
        self._requests = 0
        self._requested: dict[int, asyncio.Event] = {}

    def requested(self, count: int) -> asyncio.Event:
        return self._requested.setdefault(count, asyncio.Event())

    @asynccontextmanager
    async def acquire(self, names, *, label=None):
        self._requests += 1
        self.requested(self._requests).set()
        async with super().acquire(names, label=label):
            yield


def _deadline(monkeypatch, seconds: float) -> dict:
    """Configure the consolidation deadline; mutate the result to change it."""
    config = {"memory_consolidation_timeout_seconds": seconds}
    monkeypatch.setattr(
        "kestrel_sovereign.storage.memory_system.load_section",
        lambda _section: config,
    )
    return config


def _tool(worker: _BlockedWorker, locks, passes) -> MemoryFeature:
    """The ``memory_consolidate`` tool over production wiring.

    ``passes`` runs once per consolidation pass, in order.
    """
    calls = iter(passes)

    async def run_consolidation():
        await next(calls)()
        return {"episodes_created": 1}

    memory_system = MemorySystem(
        storage=SimpleNamespace(db=worker.db), agent_id="did:test:drain"
    )
    memory_system.consolidator = SimpleNamespace(
        run_consolidation=run_consolidation
    )
    agent = SimpleNamespace(
        agent_name="Claw",
        memory_system=memory_system,
        dispatcher=SimpleNamespace(lock_manager=locks),
    )
    return MemoryFeature(agent)


async def test_later_pass_takes_memory_only_after_abandoned_write_returned(
    monkeypatch, worker,
):
    """No write of a later pass can land while the abandoned one still runs."""
    watch = _RecordWatch(monkeypatch)
    config = _deadline(monkeypatch, DEADLINE_SECONDS)
    locks = _ObservedLocks()
    later_pass_saw_worker_returned = []

    async def later_pass():
        later_pass_saw_worker_returned.append(worker.returned.is_set())
        await worker.later_write("later")

    feature = _tool(worker, locks, [worker.abandoned_write, later_pass])

    abandoned = asyncio.create_task(feature.memory_consolidate())
    await watch.reached("draining", unless_done=abandoned)
    config["memory_consolidation_timeout_seconds"] = 60
    later = asyncio.create_task(feature.memory_consolidate())
    await _until(locks.requested(2).wait())
    assert locks.is_held(ResourceLock.MEMORY)
    assert not later.done()

    worker.release()
    abandoned_result, later_result = await _until(
        asyncio.gather(abandoned, later)
    )

    assert later_pass_saw_worker_returned == [True]
    assert worker.log == ["abandoned:enter", "abandoned:return", "later:write"]
    assert abandoned_result.status is ToolResultStatus.ERROR
    assert abandoned_result.data["abandoned_work"] == ABANDONED_WORK_FINISHED
    assert later_result.status is ToolResultStatus.OK
    # The abandoned UPDATE was rolled back; the later write committed alone.
    assert await worker.db.fetchall("SELECT value FROM memories") == [(0,)]
    assert await worker.db.fetchall("SELECT tag FROM later_pass") == [("later",)]
    assert worker.record()["state"] == "done"
    assert watch.states == ["draining", "done"]


async def test_wait_that_runs_out_isolates_connection_before_releasing_memory(
    monkeypatch, worker, caplog,
):
    """The bounded wait never releases MEMORY onto a shared connection."""
    from kestrel_sovereign.storage.db import sqlite as sqlite_module

    # The wait is bounded by the drain's own deadline. Shrink it; the worker
    # stays held by the test, so the wait runs out whatever the clock does.
    monkeypatch.setattr(sqlite_module, "_CANCELLED_OPERATION_DRAIN_TIMEOUT_S", 0.2)
    watch = _RecordWatch(monkeypatch)
    config = _deadline(monkeypatch, DEADLINE_SECONDS)
    locks = OrderedLockManager()

    async def later_pass():
        await worker.later_write("later")

    feature = _tool(worker, locks, [worker.abandoned_write, later_pass])

    with caplog.at_level("WARNING", logger="kestrel_sovereign.storage.memory_system"):
        abandoned_result = await feature.memory_consolidate()

    assert not worker.returned.is_set()
    assert not locks.is_held(ResourceLock.MEMORY)
    assert abandoned_result.status is ToolResultStatus.ERROR
    assert abandoned_result.data["abandoned_work"] == ABANDONED_WORK_ISOLATED
    assert "refuses all work until that work finishes" in abandoned_result.error
    assert abandoned_result.data["drain_record"] == str(worker.record_path)
    record = worker.record()
    assert record["state"] == "isolated"
    assert record["isolation"] == "cleanup_deadline"
    assert any(
        entry.levelname == "WARNING" and "still running" in entry.getMessage()
        for entry in caplog.records
    )
    assert worker.backend.write_connection_unavailable

    # The next pass takes MEMORY at once, and the connection refuses it.
    config["memory_consolidation_timeout_seconds"] = 60
    later_result = await feature.memory_consolidate()
    assert later_result.status is ToolResultStatus.ERROR
    assert worker.log == ["abandoned:enter"]

    worker.release()
    await _until(watch.event("done").wait())

    assert worker.log == ["abandoned:enter", "abandoned:return"]
    record = worker.record()
    assert record["state"] == "done"
    assert record["cleanup"] == "finished"
    assert record["isolated_at"] is not None
    assert not worker.backend.write_connection_unavailable
    await worker.later_write("after")
    assert await worker.db.fetchall("SELECT value FROM memories") == [(0,)]
    assert await worker.db.fetchall("SELECT tag FROM later_pass") == [("after",)]


async def test_caller_cancellation_before_deadline_is_unchanged(
    monkeypatch, worker,
):
    """A caller's cancel releases MEMORY at once and settles nothing, as before."""
    _deadline(monkeypatch, 60)
    locks = OrderedLockManager()
    feature = _tool(worker, locks, [worker.abandoned_write])

    call = asyncio.create_task(feature.memory_consolidate())
    await _until(worker.entered.wait())
    assert locks.is_held(ResourceLock.MEMORY)

    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call

    assert not worker.returned.is_set()
    assert not locks.is_held(ResourceLock.MEMORY)
    assert not worker.record_path.exists()
    # The backend's own fence remains; nothing was quarantined.
    assert worker.backend.cancellation_cleanup_pending
    assert not worker.backend.write_connection_quarantined

    worker.release()
    await worker.later_write("after")
    assert await worker.db.fetchall("SELECT value FROM memories") == [(0,)]


async def test_caller_cancelling_the_wait_isolates_before_cancellation_propagates(
    monkeypatch, worker,
):
    watch = _RecordWatch(monkeypatch)
    _deadline(monkeypatch, DEADLINE_SECONDS)
    locks = OrderedLockManager()
    feature = _tool(worker, locks, [worker.abandoned_write])

    call = asyncio.create_task(feature.memory_consolidate())
    await watch.reached("draining", unless_done=call)
    assert locks.is_held(ResourceLock.MEMORY)

    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call

    assert not worker.returned.is_set()
    assert not locks.is_held(ResourceLock.MEMORY)
    assert worker.backend.write_connection_quarantined
    record = worker.record()
    assert record["state"] == "isolated"
    assert record["isolation"] == "caller_cancelled"
    with pytest.raises(ConnectionError, match="isolated until"):
        await worker.later_write("refused")

    worker.release()
    await _until(watch.event("done").wait())

    assert worker.log == ["abandoned:enter", "abandoned:return"]
    assert worker.record()["cleanup"] == "finished"
    assert not worker.backend.write_connection_quarantined
    await worker.later_write("after")
    assert await worker.db.fetchall("SELECT tag FROM later_pass") == [("after",)]


async def test_provider_timeout_is_not_treated_as_the_deadline(
    monkeypatch, worker,
):
    """A provider TimeoutError neither waits for, records, nor isolates work."""
    _deadline(monkeypatch, 60)
    locks = OrderedLockManager()

    # Work some other caller abandoned is pending on the connection.
    other = asyncio.create_task(worker.abandoned_write())
    await _until(worker.entered.wait())
    other.cancel()
    with pytest.raises(asyncio.CancelledError):
        await other
    assert worker.backend.cancellation_cleanup_pending

    async def provider_times_out():
        raise TimeoutError("provider socket timeout")

    feature = _tool(worker, locks, [provider_times_out])
    result = await feature.memory_consolidate()

    assert result.status is ToolResultStatus.ERROR
    assert result.error == "provider socket timeout"
    assert not worker.returned.is_set()
    assert not locks.is_held(ResourceLock.MEMORY)
    assert not worker.record_path.exists()
    assert not worker.backend.write_connection_quarantined
    worker.release()


async def test_deadline_with_no_database_work_running_releases_at_once(
    monkeypatch, worker,
):
    _deadline(monkeypatch, DEADLINE_SECONDS)
    locks = OrderedLockManager()

    async def waits_on_a_provider():
        await asyncio.Event().wait()

    feature = _tool(worker, locks, [waits_on_a_provider])
    result = await feature.memory_consolidate()

    assert result.status is ToolResultStatus.ERROR
    assert result.data["abandoned_work"] == ABANDONED_WORK_NONE
    assert "no database work left running" in result.error
    assert result.data["drain_record"] is None
    assert not worker.record_path.exists()
    assert not locks.is_held(ResourceLock.MEMORY)


async def test_memory_health_reports_the_record_until_work_settles(
    monkeypatch, worker,
):
    watch = _RecordWatch(monkeypatch)
    _deadline(monkeypatch, DEADLINE_SECONDS)
    feature = _tool(worker, OrderedLockManager(), [worker.abandoned_write])
    agent = SimpleNamespace(
        _raw_storage=SimpleNamespace(db=worker.db),
        storage=SimpleNamespace(retriever=object(), consolidator=object()),
    )

    call = asyncio.create_task(feature.memory_consolidate())
    await watch.reached("draining", unless_done=call)
    draining = await check_memory_system(agent)
    database = await check_database(worker.db)

    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call
    isolated = await check_memory_system(agent)
    quarantined = await check_database(worker.db)

    worker.release()
    await _until(watch.event("done").wait())
    settled = await check_memory_system(agent)

    assert draining["status"] == "warn"
    assert "memory lock is held" in draining["message"]
    assert draining["details"]["consolidation_drain"]["state"] == "draining"
    assert database["status"] == "warn"
    assert isolated["status"] == "fail"
    assert "refuses all work" in isolated["message"]
    assert quarantined["status"] == "fail"
    assert "isolated until it finishes" in quarantined["message"]
    assert settled["status"] == "pass"


async def test_unreadable_drain_record_degrades_memory_health(tmp_path):
    database_file = tmp_path / "agent.db"
    backend = SQLiteBackend(str(database_file))
    (tmp_path / ("agent.db" + DRAIN_RECORD_SUFFIX)).write_text("[not, a, record")
    agent = SimpleNamespace(
        _raw_storage=SimpleNamespace(db=AsyncDatabase(backend)),
        storage=SimpleNamespace(retriever=object(), consolidator=object()),
    )

    result = await check_memory_system(agent)

    assert result["status"] == "warn"
    assert "Could not read the memory consolidation drain record" in result["message"]


class _SettlingDatabase:
    """A database whose abandoned work settles as the test says it does."""

    def __init__(self, path: Path, outcome: str) -> None:
        self.backend = SimpleNamespace(backend_type="sqlite", db_path=str(path))
        self.cancellation_cleanup_pending = True
        self._outcome = outcome
        self._callbacks = []
        self.quarantined = False

    def add_cancellation_cleanup_callback(self, callback) -> bool:
        self._callbacks.append(callback)
        return True

    async def await_cancellation_cleanup(self) -> str:
        for callback in self._callbacks:
            callback(self._outcome)
        return self._outcome

    def quarantine_cancellation_cleanup(self) -> bool:
        self.quarantined = True
        return True


@pytest.mark.parametrize(
    ("outcome", "message"),
    [
        (ABANDONED_WORK_FAILED, "refuses all work until it is reconnected"),
        (ABANDONED_WORK_RETIRED, "connection was closed under the work"),
    ],
)
async def test_unclean_settlement_is_recorded_and_reported(
    monkeypatch, tmp_path, outcome, message,
):
    _deadline(monkeypatch, DEADLINE_SECONDS)
    db = _SettlingDatabase(tmp_path / "agent.db", outcome)

    async def run_consolidation():
        await asyncio.Event().wait()

    memory_system = MemorySystem(
        storage=SimpleNamespace(db=db), agent_id="did:test:settled"
    )
    memory_system.consolidator = SimpleNamespace(
        run_consolidation=run_consolidation
    )

    with pytest.raises(MemoryConsolidationTimeoutError) as caught:
        await memory_system.consolidate()

    assert caught.value.abandoned_work == outcome
    assert message in str(caught.value)
    record = read_consolidation_drain_record(
        tmp_path / ("agent.db" + DRAIN_RECORD_SUFFIX)
    )
    assert record["state"] == "done"
    assert record["cleanup"] == outcome
    assert not db.quarantined


async def test_record_write_failure_does_not_replace_the_deadline_outcome(
    monkeypatch, tmp_path, caplog,
):
    _deadline(monkeypatch, DEADLINE_SECONDS)

    def unwritable(path, record):
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr(
        drain_module, "write_consolidation_drain_record", unwritable
    )
    db = _SettlingDatabase(tmp_path / "agent.db", ABANDONED_WORK_FINISHED)

    async def run_consolidation():
        await asyncio.Event().wait()

    memory_system = MemorySystem(
        storage=SimpleNamespace(db=db), agent_id="did:test:unwritable"
    )
    memory_system.consolidator = SimpleNamespace(
        run_consolidation=run_consolidation
    )

    with caplog.at_level("ERROR", logger="kestrel_sovereign.storage.consolidation_drain"):
        with pytest.raises(MemoryConsolidationTimeoutError) as caught:
            await memory_system.consolidate()

    assert caught.value.abandoned_work == ABANDONED_WORK_FINISHED
    assert any(
        "Could not persist the memory consolidation drain record"
        in entry.getMessage()
        for entry in caplog.records
    )


async def test_quarantine_is_a_no_op_without_abandoned_work(tmp_path):
    backend = SQLiteBackend(str(tmp_path / "idle.db"))
    await backend.connect()
    try:
        assert not backend.cancellation_cleanup_pending
        assert backend.quarantine_cancellation_cleanup() is False
        assert backend.add_cancellation_cleanup_callback(lambda _: None) is False
        assert await backend.await_cancellation_cleanup() == "finished"
        assert not backend.write_connection_unavailable
        await backend.execute("CREATE TABLE t (v INTEGER)")
    finally:
        await backend.close()


async def test_quarantined_connection_refuses_reads_but_serves_diagnostics(worker):
    other = asyncio.create_task(worker.abandoned_write())
    await _until(worker.entered.wait())
    other.cancel()
    with pytest.raises(asyncio.CancelledError):
        await other
    settled = []
    assert worker.backend.add_cancellation_cleanup_callback(settled.append)

    assert worker.backend.quarantine_cancellation_cleanup() is True

    with pytest.raises(QueryError, match="isolated until"):
        await worker.backend.fetch_all("SELECT value FROM memories")
    assert await worker.backend.fetch_all_diagnostic(
        "SELECT value FROM memories"
    ) == [(0,)]

    worker.release()
    assert await worker.backend.await_cancellation_cleanup() == "finished"
    assert settled == ["finished"]
    assert not worker.backend.write_connection_unavailable
    assert await worker.backend.fetch_all("SELECT value FROM memories") == [(0,)]


async def test_failed_wait_isolates_before_its_error_propagates(
    monkeypatch, tmp_path,
):
    _deadline(monkeypatch, DEADLINE_SECONDS)
    db = _SettlingDatabase(tmp_path / "agent.db", ABANDONED_WORK_FINISHED)

    async def wait_fails():
        raise RuntimeError("drain wait broke")

    db.await_cancellation_cleanup = wait_fails

    async def run_consolidation():
        await asyncio.Event().wait()

    memory_system = MemorySystem(
        storage=SimpleNamespace(db=db), agent_id="did:test:wait-failed"
    )
    memory_system.consolidator = SimpleNamespace(
        run_consolidation=run_consolidation
    )

    with pytest.raises(RuntimeError, match="drain wait broke"):
        await memory_system.consolidate()

    assert db.quarantined
    record = read_consolidation_drain_record(
        tmp_path / ("agent.db" + DRAIN_RECORD_SUFFIX)
    )
    assert record["state"] == "isolated"
    assert record["isolation"] == "wait_failed"


async def test_cancelled_cleanup_waiter_quarantines_the_connection(worker):
    other = asyncio.create_task(worker.abandoned_write())
    await _until(worker.entered.wait())
    other.cancel()
    with pytest.raises(asyncio.CancelledError):
        await other

    waiter = asyncio.create_task(worker.backend.await_cancellation_cleanup())
    await asyncio.sleep(0)  # one loop turn: the waiter is now inside its wait
    assert not waiter.done()
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    assert worker.backend.write_connection_quarantined
    with pytest.raises(ConnectionError, match="isolated until"):
        await worker.later_write("refused")
    worker.release()
    assert await _until(worker.backend.await_cancellation_cleanup()) == "finished"
    assert not worker.backend.write_connection_quarantined


async def test_quarantine_keeps_a_stronger_failure_latch(worker):
    other = asyncio.create_task(worker.abandoned_write())
    await _until(worker.entered.wait())
    other.cancel()
    with pytest.raises(asyncio.CancelledError):
        await other
    # A second hand-off while the drain is live is an invariant breach the
    # backend latches as needing a reconnect; isolation must not soften it.
    worker.backend._handoff_cancelled_write(worker.backend._ensure_connected())
    assert worker.backend.write_connection_requires_reconnect

    assert worker.backend.quarantine_cancellation_cleanup() is True

    assert worker.backend.write_connection_requires_reconnect
    assert not worker.backend.write_connection_quarantined
    worker.release()
    assert await _until(worker.backend.await_cancellation_cleanup()) == "failed"
    assert worker.backend.write_connection_requires_reconnect


# ---------------------------------------------------------------------------
# A record outlives its owner only as history (#2978 review): a process that
# dies mid-drain never marks its record done, so a record still claiming
# running work is believed only while its owner may be alive.
# ---------------------------------------------------------------------------


def _record_path(database_file: Path) -> Path:
    return database_file.with_name(database_file.name + DRAIN_RECORD_SUFFIX)


def _exited_owner() -> dict:
    """An owner provably gone: this PID, but a process that started an hour earlier.

    A recycled PID necessarily started later than the record naming it was
    written, so a start instant that does not match the live one is the
    proof a crashed predecessor's record carries.
    """
    from kestrel_sovereign.multi_agent.liveness import process_identity

    owner = {**process_identity(), "runtime_id": "a-crashed-predecessor"}
    assert owner["started_at"] is not None, "this host must be able to read start times"
    owner["started_at"] -= 3600.0
    return owner


def _record_left_by(database_file: Path, owner, *, state: str) -> dict:
    """The record a pass in another process left at ``state`` before dying."""
    record = drain_module.new_consolidation_drain_record(
        agent_id="did:test:crashed",
        database=str(database_file),
        connection="the-dead-process-connection",
        deadline_seconds=DEADLINE_SECONDS,
    )
    if state == "isolated":
        record = drain_module.isolated_consolidation_drain_record(
            record, "cleanup_deadline"
        )
    record["owner"] = owner
    return record


def _health_agent(db) -> SimpleNamespace:
    return SimpleNamespace(
        _raw_storage=SimpleNamespace(db=db),
        storage=SimpleNamespace(retriever=object(), consolidator=object()),
    )


async def test_fresh_connection_reconciles_an_isolated_record_its_dead_owner_left(
    monkeypatch, tmp_path, caplog,
):
    """The reported defect: memory health failed forever after a crash mid-drain.

    Before the fix, database health passed on the new connection while memory
    health kept failing with "the database connection refuses all work".
    """
    database_file = tmp_path / "agent.db"
    left = _record_left_by(database_file, _exited_owner(), state="isolated")
    drain_module.write_consolidation_drain_record(_record_path(database_file), left)

    backend = SQLiteBackend(str(database_file))
    await backend.connect()
    try:
        db = AsyncDatabase(backend)
        with caplog.at_level("WARNING", logger=drain_module.__name__):
            database = await check_database(db)
            memory = await check_memory_system(_health_agent(db))
        writes = []
        monkeypatch.setattr(
            drain_module,
            "write_consolidation_drain_record",
            lambda path, record: writes.append(record),
        )
        again = await check_memory_system(_health_agent(db))
    finally:
        await backend.close()

    assert database["status"] == "pass"
    assert memory["status"] == "pass", memory
    on_disk = read_consolidation_drain_record(_record_path(database_file))
    assert memory["details"]["consolidation_drain"] == on_disk
    assert on_disk["state"] == "done"
    assert on_disk["cleanup"] == "owner_retired"
    assert on_disk["reconciled_from"] == "isolated"
    assert "has exited" in on_disk["reconciliation"]
    # When the work ended is unknown; it is not invented.
    assert on_disk["finished_at"] is None
    # The evidence of what happened survives the close-out.
    for field in (
        "pass_id", "agent_id", "timed_out_at", "isolated_at", "isolation",
        "owner", "connection", "deadline_seconds",
    ):
        assert on_disk[field] == left[field], field
    assert any(
        "recorded as settled" in entry.getMessage() for entry in caplog.records
    )
    # Reconciled once: a later read reports the history and writes nothing.
    assert again["status"] == "pass"
    assert writes == []


async def test_memory_system_startup_reconciles_a_draining_record_of_an_exited_process(
    tmp_path,
):
    from kestrel_sovereign.storage.async_storage import AsyncStorage

    async with AsyncStorage(
        str(tmp_path / "kestrel.db"), agent_id="did:test:restart"
    ) as raw:
        path = drain_module.consolidation_drain_record_path(raw.db)
        assert path is not None
        left = _record_left_by(
            Path(raw.db.backend.db_path), _exited_owner(), state="draining"
        )
        drain_module.write_consolidation_drain_record(path, left)

        memory_system = MemorySystem(storage=raw, agent_id="did:test:restart")
        await memory_system.initialize()

        record = read_consolidation_drain_record(path)

    assert record["state"] == "done"
    assert record["cleanup"] == "owner_retired"
    assert record["reconciled_from"] == "draining"
    assert record["pass_id"] == left["pass_id"]


async def test_a_live_other_process_keeps_its_record(tmp_path):
    """Only proof of exit closes a record out; a running owner is reported."""
    from kestrel_sovereign.multi_agent.liveness import process_identity
    from kestrel_sovereign.multi_agent.process_manager import ProcessManager

    parent = os.getppid()
    owner = {
        **process_identity(),
        "pid": parent,
        "started_at": ProcessManager.process_start_time(parent),
        "runtime_id": "the-test-runner-parent",
    }
    database_file = tmp_path / "agent.db"
    left = _record_left_by(database_file, owner, state="isolated")
    drain_module.write_consolidation_drain_record(_record_path(database_file), left)

    memory = await check_memory_system(
        _health_agent(AsyncDatabase(SQLiteBackend(str(database_file))))
    )

    assert memory["status"] == "warn", memory
    assert f"PID {parent!r}" in memory["message"]
    assert "not the one holding it" in memory["message"]
    assert read_consolidation_drain_record(_record_path(database_file)) == left


@pytest.mark.parametrize(
    ("owner", "holder"),
    [
        ({"host": "another-machine"}, "on host 'another-machine'"),
        (None, "an owner the record does not identify"),
    ],
)
async def test_an_owner_that_cannot_be_proven_exited_keeps_its_record(
    tmp_path, owner, holder,
):
    """Another host's PID, or no owner at all, proves nothing either way."""
    database_file = tmp_path / "agent.db"
    left = _record_left_by(
        database_file,
        None if owner is None else {**_exited_owner(), **owner},
        state="isolated",
    )
    drain_module.write_consolidation_drain_record(_record_path(database_file), left)

    memory = await check_memory_system(
        _health_agent(AsyncDatabase(SQLiteBackend(str(database_file))))
    )

    assert memory["status"] == "warn", memory
    assert holder in memory["message"]
    assert "may be deleted" in memory["message"]
    assert read_consolidation_drain_record(_record_path(database_file)) == left


async def test_a_record_of_this_process_no_pass_holds_is_reconciled(tmp_path):
    """Every pass here holding work is known, so an unknown one holds none."""
    database_file = tmp_path / "agent.db"
    left = drain_module.isolated_consolidation_drain_record(
        drain_module.new_consolidation_drain_record(
            agent_id="did:test:orphan",
            database=str(database_file),
            connection=None,
            deadline_seconds=DEADLINE_SECONDS,
        ),
        "caller_cancelled",
    )
    drain_module.write_consolidation_drain_record(_record_path(database_file), left)

    status = drain_module.reconcile_consolidation_drain(
        AsyncDatabase(SQLiteBackend(str(database_file)))
    )

    assert not status.unsettled
    assert status.record["cleanup"] == "owner_retired"
    assert "no pass in the process" in status.record["reconciliation"]
    assert read_consolidation_drain_record(_record_path(database_file)) == status.record


async def test_reconciliation_never_replaces_a_record_that_moved_on(
    monkeypatch, tmp_path,
):
    """A pass that timed out after the stale record was read keeps its record."""
    database_file = tmp_path / "agent.db"
    path = _record_path(database_file)
    stale = _record_left_by(database_file, _exited_owner(), state="isolated")
    fresh = drain_module.new_consolidation_drain_record(
        agent_id="did:test:fresh",
        database=str(database_file),
        connection="a-live-connection",
        deadline_seconds=DEADLINE_SECONDS,
    )
    drain_module.write_consolidation_drain_record(path, fresh)
    real = drain_module.read_consolidation_drain_record
    reads = []

    def read(record_path):
        reads.append(record_path)
        # The first read sees the stale record; the fresh one lands after.
        return dict(stale) if len(reads) == 1 else real(record_path)

    monkeypatch.setattr(drain_module, "read_consolidation_drain_record", read)

    status = drain_module.reconcile_consolidation_drain(
        AsyncDatabase(SQLiteBackend(str(database_file)))
    )

    assert status.record["cleanup"] == "owner_retired"
    assert len(reads) == 2
    assert real(path) == fresh


async def test_a_settlement_a_failed_write_left_unrecorded_is_trusted_and_retried(
    monkeypatch, tmp_path,
):
    """The owning process answers from its pass, not from a stale file."""
    _deadline(monkeypatch, DEADLINE_SECONDS)
    real = drain_module.write_consolidation_drain_record
    refused_states = {"done"}

    def writer(path, record):
        if record["state"] in refused_states:
            raise PermissionError(13, "Permission denied", str(path))
        real(path, record)

    monkeypatch.setattr(drain_module, "write_consolidation_drain_record", writer)
    db = _SettlingDatabase(tmp_path / "agent.db", ABANDONED_WORK_FINISHED)

    async def run_consolidation():
        await asyncio.Event().wait()

    memory_system = MemorySystem(
        storage=SimpleNamespace(db=db), agent_id="did:test:unrecorded"
    )
    memory_system.consolidator = SimpleNamespace(
        run_consolidation=run_consolidation
    )
    with pytest.raises(MemoryConsolidationTimeoutError):
        await memory_system.consolidate()
    path = _record_path(tmp_path / "agent.db")
    assert read_consolidation_drain_record(path)["state"] == "draining"

    still_refused = await check_memory_system(_health_agent(db))
    assert still_refused["status"] == "pass", still_refused
    assert still_refused["details"]["consolidation_drain"]["state"] == "done"
    assert read_consolidation_drain_record(path)["state"] == "draining"

    refused_states.clear()
    retried = await check_memory_system(_health_agent(db))

    assert retried["status"] == "pass"
    record = read_consolidation_drain_record(path)
    assert record["state"] == "done"
    assert record["cleanup"] == ABANDONED_WORK_FINISHED
    assert "reconciliation" not in record


async def test_isolation_fails_only_the_connection_that_holds_the_work(
    monkeypatch, worker,
):
    """The record names its connection generation; another connection is not isolated."""
    watch = _RecordWatch(monkeypatch)
    _deadline(monkeypatch, DEADLINE_SECONDS)
    feature = _tool(worker, OrderedLockManager(), [worker.abandoned_write])

    call = asyncio.create_task(feature.memory_consolidate())
    await watch.reached("draining", unless_done=call)
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call
    assert worker.record()["connection"] == worker.backend.connection_generation

    holder = await check_memory_system(_health_agent(worker.db))
    other = await check_memory_system(
        _health_agent(AsyncDatabase(SQLiteBackend(str(worker.path))))
    )

    worker.release()
    await _until(watch.event("done").wait())

    assert holder["status"] == "fail"
    assert "refuses all work" in holder["message"]
    assert other["status"] == "warn", other
    assert "another connection of this process" in other["message"]
    assert worker.record()["state"] == "done"


async def test_each_connect_has_its_own_generation(tmp_path):
    backend = SQLiteBackend(str(tmp_path / "generations.db"))
    assert backend.connection_generation is None
    await backend.connect()
    first = backend.connection_generation
    await backend.close()
    assert backend.connection_generation is None
    await backend.connect()
    try:
        assert first is not None
        assert backend.connection_generation not in (None, first)
    finally:
        await backend.close()


def test_a_record_that_names_no_process_proves_no_exit(tmp_path):
    from kestrel_sovereign.multi_agent.liveness import recorded_process_exited

    path = tmp_path / "record.json"
    assert recorded_process_exited(path, None) is False
    assert recorded_process_exited(path, ["not", "a", "record"]) is False
    assert recorded_process_exited(path, _exited_owner()) is True
