"""Work a timed-out consolidation pass leaves running is settled before it can overlap (#2978).

The configured deadline cancels the pass, but an aiosqlite statement already
running on the connection's worker thread keeps running. These tests drive a
real ``SQLiteBackend`` whose worker is held inside a user-defined SQL function,
so "still running" is a fact the test controls rather than a timing guess, and
check the contract ``MemorySystem.consolidate`` now keeps: the MEMORY lock is
released only once that work has finished, or once the connection is isolated
so a later pass cannot use it; each timed-out pass keeps its own durable
record of the work until it settles, and no other pass or process ever
writes that record.
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
    DRAIN_RECORD_DIR_SUFFIX,
    LEGACY_DRAIN_RECORD_SUFFIX,
    read_consolidation_drain_record,
)
from kestrel_sovereign.storage.db.interface import ConnectionError, QueryError
from kestrel_sovereign.storage.db import sqlite as sqlite_module
from kestrel_sovereign.storage.db.sqlite import SQLiteBackend
from kestrel_sovereign.storage.memory_system import (
    ABANDONED_WORK_FAILED,
    ABANDONED_WORK_FINISHED,
    ABANDONED_WORK_ISOLATED,
    ABANDONED_WORK_NONE,
    ABANDONED_WORK_RETIRED,
    ABANDONED_WORK_RETIRING,
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


@pytest.fixture(autouse=True)
def _own_passes_per_test(monkeypatch):
    """Each test starts with no passes of this process, and leaves none behind."""
    monkeypatch.setattr(drain_module, "_OWNED_PASSES", {})


async def _until(awaitable):
    """Await ``awaitable`` under the hang guard."""
    async with asyncio.timeout(HANG_GUARD_SECONDS):
        return await awaitable


def _records_dir(database_file: Path) -> Path:
    return database_file.with_name(database_file.name + DRAIN_RECORD_DIR_SUFFIX)


def _record_files(database_file: Path) -> list[Path]:
    directory = _records_dir(database_file)
    if not directory.exists():
        return []
    return sorted(
        path for path in directory.iterdir() if not path.name.startswith(".")
    )


def _only_record(database_file: Path) -> dict:
    [path] = _record_files(database_file)
    return read_consolidation_drain_record(path)


def _records_by_agent(database_file: Path) -> dict:
    records = [read_consolidation_drain_record(p) for p in _record_files(database_file)]
    return {record["agent_id"]: record for record in records}


class _BlockedWorker:
    """A real SQLite database whose aiosqlite worker the test holds and frees.

    ``hold(value)`` is a SQL function that parks the worker thread until
    :meth:`release`; ``mark(tag)`` records that a statement carrying ``tag``
    executed. ``log`` is the worker-side order of both, which is the order
    the statements actually ran on the connection. Several workers may open
    the same database file, each on its own connection and worker thread.
    """

    def __init__(self, path: Path, *, create_schema: bool = True) -> None:
        self.path = path
        self.backend = SQLiteBackend(str(path))
        self.db = AsyncDatabase(self.backend)
        self.log: list[str] = []
        self.entered = asyncio.Event()
        self.returned = threading.Event()
        self._create_schema = create_schema
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
        if self._create_schema:
            await self.db.execute("CREATE TABLE memories (value INTEGER NOT NULL)")
            await self.db.execute("INSERT INTO memories (value) VALUES (0)")
            await self.db.execute("CREATE TABLE later_pass (tag TEXT NOT NULL)")
        self._watchdog.start()
        return self

    async def abandoned_write(self) -> None:
        """The write a deadline cancels while the worker is inside it."""
        await self.db.execute("UPDATE memories SET value = hold(value)")

    async def abandoned_read(self) -> None:
        """A read a deadline cancels while the worker is inside it.

        Readers of one WAL database do not block each other, so several
        workers can each hold one at once.
        """
        await self.db.fetchall("SELECT hold(value) FROM memories")

    async def later_write(self, tag: str) -> None:
        await self.db.execute(
            "INSERT INTO later_pass (tag) VALUES (mark(?))", (tag,)
        )

    def release(self) -> None:
        self._release.set()

    @property
    def records_dir(self) -> Path:
        return _records_dir(self.path)

    def record_path(self) -> Path:
        [path] = _record_files(self.path)
        return path

    def record(self) -> dict:
        return json.loads(self.record_path().read_text())

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
    """Wake the test when a drain record reaches a state, without polling.

    Events are kept per state, and per state of the pass on one connection
    generation, so a test running several passes waits for the right one.
    """

    def __init__(self, monkeypatch) -> None:
        self.states: list[str] = []
        self._events: dict[tuple, asyncio.Event] = {}
        real = drain_module.write_consolidation_drain_record

        def writer(path, record):
            real(path, record)
            self.states.append(record["state"])
            self.event(record["state"]).set()
            self.event(record["state"], record.get("connection")).set()

        monkeypatch.setattr(
            drain_module, "write_consolidation_drain_record", writer
        )

    def event(self, state: str, connection: object = "any") -> asyncio.Event:
        return self._events.setdefault((state, connection), asyncio.Event())

    async def reached(
        self,
        state: str,
        *,
        unless_done: asyncio.Task,
        on: _BlockedWorker | None = None,
    ) -> None:
        """Wait for ``state``; fail rather than hang if ``unless_done`` ends first."""
        event = (
            self.event(state)
            if on is None
            else self.event(state, on.backend.connection_generation)
        )
        waiter = asyncio.ensure_future(event.wait())
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


def _memory_system(db, agent_id: str, passes) -> MemorySystem:
    """A ``MemorySystem`` over ``db``; ``passes`` runs once per pass, in order."""
    calls = iter(passes)

    async def run_consolidation():
        await next(calls)()
        return {"episodes_created": 1}

    memory_system = MemorySystem(storage=SimpleNamespace(db=db), agent_id=agent_id)
    memory_system.consolidator = SimpleNamespace(
        run_consolidation=run_consolidation
    )
    return memory_system


def _tool(worker: _BlockedWorker, locks, passes) -> MemoryFeature:
    """The ``memory_consolidate`` tool over production wiring."""
    agent = SimpleNamespace(
        agent_name="Claw",
        memory_system=_memory_system(worker.db, "did:test:drain", passes),
        dispatcher=SimpleNamespace(lock_manager=locks),
    )
    return MemoryFeature(agent)


def _health_agent(db) -> SimpleNamespace:
    return SimpleNamespace(
        _raw_storage=SimpleNamespace(db=db),
        storage=SimpleNamespace(retriever=object(), consolidator=object()),
    )


async def _memory_health(db) -> dict:
    return await check_memory_system(_health_agent(db))


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
    assert abandoned_result.data["drain_record"] == str(worker.record_path())
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
    assert not worker.records_dir.exists()
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
    assert not worker.records_dir.exists()
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
    assert not worker.records_dir.exists()
    assert not locks.is_held(ResourceLock.MEMORY)


async def test_memory_health_reports_the_record_until_work_settles(
    monkeypatch, worker,
):
    watch = _RecordWatch(monkeypatch)
    _deadline(monkeypatch, DEADLINE_SECONDS)
    feature = _tool(worker, OrderedLockManager(), [worker.abandoned_write])

    call = asyncio.create_task(feature.memory_consolidate())
    await watch.reached("draining", unless_done=call)
    draining = await _memory_health(worker.db)
    database = await check_database(worker.db)

    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call
    isolated = await _memory_health(worker.db)
    quarantined = await check_database(worker.db)

    worker.release()
    await _until(watch.event("done").wait())
    settled = await _memory_health(worker.db)

    assert draining["status"] == "warn"
    assert "memory lock is held" in draining["message"]
    [entry] = draining["details"]["consolidation_drains"]
    assert entry["record"]["state"] == "draining"
    assert entry["holder"] == "this_connection"
    assert database["status"] == "warn"
    assert isolated["status"] == "fail"
    assert "refuses all work" in isolated["message"]
    assert quarantined["status"] == "fail"
    assert "isolated until it finishes" in quarantined["message"]
    assert settled["status"] == "pass"
    assert settled["details"]["consolidation_drain"]["state"] == "done"


@pytest.mark.parametrize(
    ("name", "content"),
    [
        ("0" * 32 + ".json", "[not, a, record"),
        ("0" * 32 + ".json", json.dumps({"pass_id": "1" * 32, "state": "done"})),
    ],
    ids=["not-json", "names-another-pass"],
)
async def test_unreadable_drain_record_degrades_memory_health(tmp_path, name, content):
    database_file = tmp_path / "agent.db"
    _records_dir(database_file).mkdir()
    (_records_dir(database_file) / name).write_text(content)
    # A staging file a crashed write left behind is not a record.
    (_records_dir(database_file) / f".{name}.123.abcd.tmp").write_text("{")

    result = await _memory_health(AsyncDatabase(SQLiteBackend(str(database_file))))

    assert result["status"] == "warn"
    assert "Could not read the memory consolidation drain record" in result["message"]
    [unreadable] = result["details"]["unreadable_drain_records"]
    assert unreadable["record_path"].endswith(name)


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


def _hung_memory_system(db, agent_id: str) -> MemorySystem:
    async def hangs():
        await asyncio.Event().wait()

    return _memory_system(db, agent_id, [hangs])


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

    with pytest.raises(MemoryConsolidationTimeoutError) as caught:
        await _hung_memory_system(db, "did:test:settled").consolidate()

    assert caught.value.abandoned_work == outcome
    assert message in str(caught.value)
    record = _only_record(tmp_path / "agent.db")
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

    with caplog.at_level("ERROR", logger="kestrel_sovereign.storage.consolidation_drain"):
        with pytest.raises(MemoryConsolidationTimeoutError) as caught:
            await _hung_memory_system(db, "did:test:unwritable").consolidate()

    assert caught.value.abandoned_work == ABANDONED_WORK_FINISHED
    # No record was written, so none is named.
    assert caught.value.drain_record is None
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

    with pytest.raises(RuntimeError, match="drain wait broke"):
        await _hung_memory_system(db, "did:test:wait-failed").consolidate()

    assert db.quarantined
    record = _only_record(tmp_path / "agent.db")
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
# Shutdown past its budget (#2978 review). ``close`` cancels the retained drain
# before its rollback runs and keeps the still-running worker for later
# retirement; closing the connection does not stop the statement on that
# worker. The record must stay unsettled, and memory health must not pass,
# until the worker thread has exited.
# ---------------------------------------------------------------------------

#: Shrinks ``close``'s budget. The worker is held by the test, so the close
#: runs out whatever the clock does; this only keeps the test quick.
CLOSE_BUDGET_SECONDS = 0.05


async def _close_past_its_budget(worker: _BlockedWorker):
    """Shut the connection down while its worker is held, so the close runs out.

    Returns the closed aiosqlite connection, whose worker is still running.
    """
    conn = worker.backend._ensure_connected()
    with pytest.raises(ConnectionError, match="did not complete within"):
        await _until(worker.backend.close())
    assert worker.backend.connection_retirement_pending
    assert sqlite_module._aiosqlite_worker_is_alive(conn)
    assert not worker.returned.is_set()
    return conn


async def _assert_retired_only_once_the_worker_exits(worker, watch, conn) -> None:
    """Free the worker; the record settles as retired only after it has exited."""
    worker.release()
    await _until(watch.event("done").wait())

    assert worker.returned.is_set()
    assert not sqlite_module._aiosqlite_worker_is_alive(conn)
    record = worker.record()
    assert record["state"] == "done"
    assert record["cleanup"] == "retired"
    settled = await _memory_health(worker.db)
    assert settled["status"] == "pass", settled
    assert settled["details"]["consolidation_drain"]["cleanup"] == "retired"
    assert not worker.backend.connection_retirement_pending


async def test_shutdown_past_its_budget_keeps_a_waiting_pass_unsettled(
    monkeypatch, worker,
):
    """Shutdown cancelling the drain under a waiting pass does not settle it."""
    monkeypatch.setattr(
        sqlite_module, "AIOSQLITE_WORKER_SHUTDOWN_TIMEOUT_S", CLOSE_BUDGET_SECONDS
    )
    watch = _RecordWatch(monkeypatch)
    _deadline(monkeypatch, DEADLINE_SECONDS)
    locks = OrderedLockManager()
    feature = _tool(worker, locks, [worker.abandoned_write])

    call = asyncio.create_task(feature.memory_consolidate())
    await watch.reached("draining", unless_done=call)
    assert locks.is_held(ResourceLock.MEMORY)

    conn = await _close_past_its_budget(worker)
    result = await _until(call)

    assert result.status is ToolResultStatus.ERROR
    assert result.data["abandoned_work"] == ABANDONED_WORK_RETIRING
    assert "until that connection's worker thread exits" in result.error
    assert not locks.is_held(ResourceLock.MEMORY)
    record = worker.record()
    assert record["state"] == "isolated"
    assert record["isolation"] == "connection_closed"
    assert record["finished_at"] is None
    health = await _memory_health(worker.db)
    assert health["status"] == "fail", health
    assert "cannot be reconnected until it does" in health["message"]
    [entry] = health["details"]["consolidation_drains"]
    assert entry["holder"] == "closed_connection"
    # A later pass cannot reach the abandoned work: the database cannot even
    # reconnect while the worker runs.
    with pytest.raises(ConnectionError, match="still retiring"):
        await worker.backend.connect()
    assert worker.log == ["abandoned:enter"]

    await _assert_retired_only_once_the_worker_exits(worker, watch, conn)
    assert watch.states == ["draining", "isolated", "done"]


async def test_shutdown_past_its_budget_keeps_an_isolated_pass_unsettled(
    monkeypatch, worker,
):
    """The pass already returned isolated; a later shutdown does not settle it."""
    monkeypatch.setattr(sqlite_module, "_CANCELLED_OPERATION_DRAIN_TIMEOUT_S", 0.2)
    monkeypatch.setattr(
        sqlite_module, "AIOSQLITE_WORKER_SHUTDOWN_TIMEOUT_S", CLOSE_BUDGET_SECONDS
    )
    watch = _RecordWatch(monkeypatch)
    _deadline(monkeypatch, DEADLINE_SECONDS)
    feature = _tool(worker, OrderedLockManager(), [worker.abandoned_write])

    result = await feature.memory_consolidate()
    assert result.data["abandoned_work"] == ABANDONED_WORK_ISOLATED

    conn = await _close_past_its_budget(worker)

    record = worker.record()
    assert record["state"] == "isolated"
    assert record["isolation"] == "cleanup_deadline"
    health = await _memory_health(worker.db)
    assert health["status"] == "fail", health
    [entry] = health["details"]["consolidation_drains"]
    assert entry["holder"] == "closed_connection"

    await _assert_retired_only_once_the_worker_exits(worker, watch, conn)
    assert watch.states == ["draining", "isolated", "done"]


async def test_cleanup_callback_waits_for_the_worker_not_the_close(
    monkeypatch, worker,
):
    """The backend reports a cancelled drain's work settled only once its worker exits."""
    monkeypatch.setattr(
        sqlite_module, "AIOSQLITE_WORKER_SHUTDOWN_TIMEOUT_S", CLOSE_BUDGET_SECONDS
    )
    abandoned = asyncio.create_task(worker.abandoned_write())
    await _until(worker.entered.wait())
    abandoned.cancel()
    with pytest.raises(asyncio.CancelledError):
        await abandoned
    conn = worker.backend._ensure_connected()
    settled = asyncio.get_running_loop().create_future()
    outcomes = []

    def on_settled(outcome):
        outcomes.append(
            (outcome, sqlite_module._aiosqlite_worker_is_alive(conn))
        )
        settled.set_result(None)

    assert worker.backend.add_cancellation_cleanup_callback(on_settled)

    await _close_past_its_budget(worker)
    assert outcomes == []

    worker.release()
    await _until(settled)
    assert outcomes == [("retired", False)]


async def test_a_drain_cancelled_without_a_close_leaves_the_connection_to_replace(
    worker,
):
    """No rollback will prove the work returned, so nothing may share the connection."""
    abandoned = asyncio.create_task(worker.abandoned_write())
    await _until(worker.entered.wait())
    abandoned.cancel()
    with pytest.raises(asyncio.CancelledError):
        await abandoned
    conn = worker.backend._ensure_connected()
    outcomes = []
    assert worker.backend.add_cancellation_cleanup_callback(outcomes.append)
    waiter = asyncio.create_task(worker.backend.await_cancellation_cleanup())
    await asyncio.sleep(0)  # one loop turn: the waiter is now inside its wait

    # As the event loop's shutdown does: cancel the backend-owned drain itself.
    worker.backend._cancelled_write_drain.cancel()
    assert await _until(waiter) == "retiring"

    assert worker.backend.write_connection_requires_reconnect
    with pytest.raises(ConnectionError, match="cleanup failed"):
        await worker.later_write("refused")
    database = await check_database(worker.db)
    assert database["status"] == "fail"
    assert "reconnect is required" in database["message"]
    worker.release()
    assert outcomes == []
    assert worker.log == ["abandoned:enter"]

    # ``connect`` is the recovery surface: it retires the connection, whose
    # worker exit is what settles the work, and opens its replacement.
    await _until(worker.backend.connect())
    assert outcomes == ["retired"]
    assert not sqlite_module._aiosqlite_worker_is_alive(conn)
    assert not worker.backend.write_connection_unavailable
    await worker.db.execute("INSERT INTO later_pass (tag) VALUES ('after')")
    assert await worker.db.fetchall("SELECT tag FROM later_pass") == [("after",)]
    assert worker.log == ["abandoned:enter", "abandoned:return"]


# ---------------------------------------------------------------------------
# Every pass owns its own record (#2978 review). Passes timing out on several
# connections to one database, or in several processes, each write only
# their own; one settling must never make another look settled.
# ---------------------------------------------------------------------------


async def _open_workers(database_file: Path, count: int) -> list[_BlockedWorker]:
    workers = [await _BlockedWorker(database_file).open()]
    for _ in range(count - 1):
        workers.append(
            await _BlockedWorker(database_file, create_schema=False).open()
        )
    return workers


async def test_earlier_pass_settling_leaves_a_later_isolated_pass_unsettled(
    monkeypatch, tmp_path,
):
    """The reported defect: an earlier pass's ``done`` erased a later pass's ``isolated``.

    Both passes recorded at one shared path per database, so the first to
    settle overwrote the second's evidence, and memory health passed while
    the second connection still refused all work.
    """
    watch = _RecordWatch(monkeypatch)
    _deadline(monkeypatch, DEADLINE_SECONDS)
    database_file = tmp_path / "agent.db"
    first, second = await _open_workers(database_file, 2)
    try:
        first_pass = asyncio.create_task(
            _memory_system(first.db, "did:test:first", [first.abandoned_read]).consolidate()
        )
        await watch.reached("draining", unless_done=first_pass, on=first)
        second_pass = asyncio.create_task(
            _memory_system(second.db, "did:test:second", [second.abandoned_read]).consolidate()
        )
        await watch.reached("draining", unless_done=second_pass, on=second)
        second_pass.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second_pass
        assert second.backend.write_connection_quarantined

        first.release()
        with pytest.raises(MemoryConsolidationTimeoutError) as caught:
            await _until(first_pass)
        assert caught.value.abandoned_work == ABANDONED_WORK_FINISHED

        records = _records_by_agent(database_file)
        assert records["did:test:first"]["state"] == "done"
        assert records["did:test:second"]["state"] == "isolated"
        assert len(_record_files(database_file)) == 2
        on_second = await _memory_health(second.db)
        on_first = await _memory_health(first.db)
        assert on_second["status"] == "fail", on_second
        assert "refuses all work" in on_second["message"]
        assert on_first["status"] == "warn", on_first
        assert "another connection of this process" in on_first["message"]
        [unsettled] = on_first["details"]["consolidation_drains"]
        assert unsettled["record"]["agent_id"] == "did:test:second"

        second.release()
        await _until(watch.event("done", second.backend.connection_generation).wait())

        assert (await _memory_health(second.db))["status"] == "pass"
        assert (await _memory_health(first.db))["status"] == "pass"
        records = _records_by_agent(database_file)
        assert records["did:test:first"]["cleanup"] == "finished"
        assert records["did:test:second"]["cleanup"] == "finished"
        assert records["did:test:second"]["isolation"] == "caller_cancelled"
    finally:
        await second.close()
        await first.close()


async def test_three_overlapping_passes_pass_health_only_once_the_last_settles(
    monkeypatch, tmp_path,
):
    watch = _RecordWatch(monkeypatch)
    _deadline(monkeypatch, DEADLINE_SECONDS)
    database_file = tmp_path / "agent.db"
    workers = await _open_workers(database_file, 3)
    try:
        passes = []
        for index, blocked in enumerate(workers):
            call = asyncio.create_task(
                _memory_system(
                    blocked.db, f"did:test:pass-{index}", [blocked.abandoned_read]
                ).consolidate()
            )
            await watch.reached("draining", unless_done=call, on=blocked)
            passes.append(call)

        # Settle out of order: the middle pass, then the last, then the first.
        for settled, order in ((1, 1), (2, 2), (0, 3)):
            unsettled_before = [
                index for index, call in enumerate(passes) if not call.done()
            ]
            for blocked in workers:
                health = await _memory_health(blocked.db)
                assert health["status"] == "warn", (order, health)
                assert len(health["details"]["consolidation_drains"]) == len(
                    unsettled_before
                )

            workers[settled].release()
            with pytest.raises(MemoryConsolidationTimeoutError):
                await _until(passes[settled])

        for blocked in workers:
            assert (await _memory_health(blocked.db))["status"] == "pass"
        records = _records_by_agent(database_file)
        assert sorted(records) == [f"did:test:pass-{i}" for i in range(3)]
        assert {record["state"] for record in records.values()} == {"done"}
        assert all(blocked.log == ["abandoned:enter", "abandoned:return"] for blocked in workers)
    finally:
        for blocked in reversed(workers):
            await blocked.close()


async def test_a_pass_whose_record_never_reached_the_disk_still_fails_health(
    monkeypatch, worker,
):
    """A pass of this process is reported from memory when its file is missing."""
    _deadline(monkeypatch, DEADLINE_SECONDS)
    real = drain_module.write_consolidation_drain_record
    refusing = {"draining", "isolated"}
    attempted = {"draining": asyncio.Event(), "isolated": asyncio.Event()}

    def writer(path, record):
        if record["state"] in refusing:
            attempted[record["state"]].set()
            raise PermissionError(13, "Permission denied", str(path))
        real(path, record)

    monkeypatch.setattr(drain_module, "write_consolidation_drain_record", writer)
    call = asyncio.create_task(
        _memory_system(worker.db, "did:test:unwritten", [worker.abandoned_write]).consolidate()
    )
    # The write is attempted just before the pass starts waiting on its
    # drain; when this wakes, the pass is suspended in that wait.
    waiter = asyncio.ensure_future(attempted["draining"].wait())
    done, _ = await _until(
        asyncio.wait({waiter, call}, return_when=asyncio.FIRST_COMPLETED)
    )
    waiter.cancel()
    assert waiter in done
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call
    assert attempted["isolated"].is_set()
    assert _record_files(worker.path) == []

    health = await _memory_health(worker.db)

    assert health["status"] == "fail", health
    assert "refuses all work" in health["message"]

    refusing.clear()
    worker.release()
    assert await _until(worker.backend.await_cancellation_cleanup()) == "finished"
    assert (await _memory_health(worker.db))["status"] == "pass"
    assert worker.record()["state"] == "done"


# ---------------------------------------------------------------------------
# A record left by another process is reported, never rewritten (#2978
# review): only the owning pass writes its record, so one left by a process
# that died mid-drain stays as it was until an operator deletes it.
# ---------------------------------------------------------------------------


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


def _live_other_owner() -> dict:
    """An owner that is running: the process that started this test run."""
    from kestrel_sovereign.multi_agent.liveness import process_identity
    from kestrel_sovereign.multi_agent.process_manager import ProcessManager

    parent = os.getppid()
    return {
        **process_identity(),
        "pid": parent,
        "started_at": ProcessManager.process_start_time(parent),
        "runtime_id": "the-test-runner-parent",
    }


def _left_record(database_file: Path, owner, *, state: str, agent_id: str = "did:test:crashed") -> dict:
    """The record a pass in another process left at ``state``."""
    record = drain_module.new_consolidation_drain_record(
        agent_id=agent_id,
        database=str(database_file),
        connection="the-other-process-connection",
        deadline_seconds=DEADLINE_SECONDS,
    )
    if state == "isolated":
        record = drain_module.isolated_consolidation_drain_record(
            record, "cleanup_deadline"
        )
    if state == "done":
        record = drain_module.done_consolidation_drain_record(record, "finished")
    record["owner"] = owner
    return record


def _leave(database_file: Path, record: dict) -> Path:
    path = _records_dir(database_file) / f"{record['pass_id']}.json"
    drain_module.write_consolidation_drain_record(path, record)
    return path


def _no_writes(monkeypatch) -> list:
    writes = []
    monkeypatch.setattr(
        drain_module,
        "write_consolidation_drain_record",
        lambda path, record: writes.append((path, record)),
    )
    return writes


async def test_a_record_its_exited_owner_left_is_reported_and_never_rewritten(
    monkeypatch, tmp_path,
):
    """A crash mid-drain: health says the work ended, not that it is running.

    Before the first review fix memory health failed forever claiming the
    connection refused all work. It now reports the record as left by an
    exited process, until an operator deletes it, and never rewrites it.
    """
    database_file = tmp_path / "agent.db"
    left = _left_record(database_file, _exited_owner(), state="isolated")
    path = _leave(database_file, left)
    before = path.read_bytes()
    writes = _no_writes(monkeypatch)

    backend = SQLiteBackend(str(database_file))
    await backend.connect()
    try:
        db = AsyncDatabase(backend)
        database = await check_database(db)
        memory = await _memory_health(db)
        again = await _memory_health(db)
    finally:
        await backend.close()

    assert database["status"] == "pass"
    assert memory["status"] == "warn", memory
    assert "has provably exited" in memory["message"]
    assert "work ended with it" in memory["message"]
    assert "delete it" in memory["message"]
    [entry] = memory["details"]["consolidation_drains"]
    assert entry["holder"] == "owner_exited"
    assert entry["record"] == left
    assert again == {**memory, "duration_ms": again["duration_ms"]}
    assert writes == []
    assert path.read_bytes() == before

    path.unlink()
    assert (await _memory_health(AsyncDatabase(SQLiteBackend(str(database_file)))))[
        "status"
    ] == "pass"


async def test_memory_system_startup_reports_records_and_changes_none(
    tmp_path, caplog,
):
    from kestrel_sovereign.storage.async_storage import AsyncStorage

    async with AsyncStorage(
        str(tmp_path / "kestrel.db"), agent_id="did:test:restart"
    ) as raw:
        database_file = Path(raw.db.backend.db_path)
        exited = _leave(
            database_file,
            _left_record(database_file, _exited_owner(), state="draining", agent_id="did:test:exited"),
        )
        live = _leave(
            database_file,
            _left_record(database_file, _live_other_owner(), state="isolated", agent_id="did:test:live"),
        )
        settled = _leave(
            database_file,
            _left_record(database_file, _exited_owner(), state="done", agent_id="did:test:settled"),
        )
        damaged = _records_dir(database_file) / ("b" * 32 + ".json")
        damaged.write_text("[damaged")
        before = {
            path: path.read_bytes() for path in (exited, live, settled, damaged)
        }

        memory_system = MemorySystem(storage=raw, agent_id="did:test:restart")
        with caplog.at_level("WARNING", logger="kestrel_sovereign.storage.memory_system"):
            await memory_system.initialize()

    assert {path: path.read_bytes() for path in before} == before
    startup = [
        entry.getMessage()
        for entry in caplog.records
        if "found at startup" in entry.getMessage()
    ]
    assert len(startup) == 2, startup
    assert any(str(exited) in message and "provably exited" in message for message in startup)
    assert any(str(live) in message and "running or cannot be proven" in message for message in startup)
    assert any(
        "At startup of agent" in entry.getMessage()
        and str(damaged) in entry.getMessage()
        for entry in caplog.records
    )


async def test_a_live_other_process_keeps_its_record(monkeypatch, tmp_path):
    """Only its owner settles a record; a running owner is reported as holding it."""
    database_file = tmp_path / "agent.db"
    owner = _live_other_owner()
    left = _left_record(database_file, owner, state="isolated")
    path = _leave(database_file, left)
    writes = _no_writes(monkeypatch)

    memory = await _memory_health(AsyncDatabase(SQLiteBackend(str(database_file))))

    assert memory["status"] == "warn", memory
    assert f"PID {owner['pid']!r}" in memory["message"]
    assert "not the one holding it" in memory["message"]
    assert writes == []
    assert read_consolidation_drain_record(path) == left


@pytest.mark.parametrize(
    ("owner", "holder"),
    [
        ({"host": "another-machine"}, "on host 'another-machine'"),
        (None, "an owner the record does not identify"),
    ],
)
async def test_an_owner_that_cannot_be_proven_exited_keeps_its_record(
    monkeypatch, tmp_path, owner, holder,
):
    """Another host's PID, or no owner at all, proves nothing either way."""
    database_file = tmp_path / "agent.db"
    left = _left_record(
        database_file,
        None if owner is None else {**_exited_owner(), **owner},
        state="isolated",
    )
    path = _leave(database_file, left)
    writes = _no_writes(monkeypatch)

    memory = await _memory_health(AsyncDatabase(SQLiteBackend(str(database_file))))

    assert memory["status"] == "warn", memory
    assert holder in memory["message"]
    assert "may be deleted" in memory["message"]
    assert writes == []
    assert read_consolidation_drain_record(path) == left


async def test_a_record_of_this_process_no_pass_holds_is_reported(
    monkeypatch, tmp_path,
):
    """Every pass here holding work is known, so an unknown one is reported, not settled."""
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
    path = _leave(database_file, left)
    writes = _no_writes(monkeypatch)

    report = drain_module.reconcile_consolidation_drains(
        AsyncDatabase(SQLiteBackend(str(database_file)))
    )

    [status] = report.unsettled
    assert status.holder == "unaccounted"
    assert "no pass in this process holds it" in status.describe()
    assert writes == []
    assert read_consolidation_drain_record(path) == left


async def test_a_fifo_record_is_refused_without_blocking(tmp_path):
    """A FIFO named as a record must not stall the health check on its open."""
    database_file = tmp_path / "agent.db"
    _records_dir(database_file).mkdir()
    fifo = _records_dir(database_file) / ("a" * 32 + ".json")
    os.mkfifo(fifo)
    db = AsyncDatabase(SQLiteBackend(str(database_file)))
    try:
        # In a thread, so a regression blocks the thread rather than the loop
        # and fails on the hang guard.
        report = await _until(
            asyncio.to_thread(drain_module.reconcile_consolidation_drains, db)
        )
    finally:
        try:
            # Opening the write end releases a reader blocked on the FIFO.
            os.close(os.open(fifo, os.O_WRONLY | os.O_NONBLOCK))
        except OSError:
            pass

    [unreadable] = report.unreadable
    assert unreadable.path == fifo.resolve()
    assert "not a regular file" in unreadable.error
    assert report.statuses == ()


async def test_an_unlistable_record_directory_still_reports_this_process_passes(
    tmp_path,
):
    """A pass whose record cannot be written still fails health on its connection."""
    database_file = tmp_path / "agent.db"
    _records_dir(database_file).write_text("not a directory")
    db = AsyncDatabase(SQLiteBackend(str(database_file)))
    drain = drain_module.ConsolidationDrain(
        db, agent_id="did:test:unlistable", deadline_seconds=DEADLINE_SECONDS
    )
    assert drain.write() is False
    drain.isolate("cleanup_deadline")

    health = await _memory_health(db)

    assert health["status"] == "fail", health
    assert "refuses all work" in health["message"]
    assert "its record could not be written durably" in health["message"]
    assert "(and 1 more; see details)" in health["message"]
    [unsettled] = health["details"]["consolidation_drains"]
    assert unsettled["holder"] == "this_connection"
    [unreadable] = health["details"]["unreadable_drain_records"]
    assert unreadable["record_path"] == str(_records_dir(database_file).resolve())
    report = drain_module.reconcile_consolidation_drains(db)
    assert report.unreadable[0].describe().startswith(
        "Could not list the memory consolidation drain records in"
    )


async def test_a_pass_of_this_process_is_answered_from_memory_not_its_file(tmp_path):
    """Its own file is never read, so a damaged copy neither hides nor doubles it."""
    database_file = tmp_path / "agent.db"
    db = AsyncDatabase(SQLiteBackend(str(database_file)))
    drain = drain_module.ConsolidationDrain(
        db, agent_id="did:test:own", deadline_seconds=DEADLINE_SECONDS
    )
    assert drain.write()
    drain.path.write_text("[damaged")

    health = await _memory_health(db)

    assert health["status"] == "warn", health
    assert "memory lock is held" in health["message"]
    [unsettled] = health["details"]["consolidation_drains"]
    assert unsettled["record"]["pass_id"] == drain.pass_id
    assert health["details"]["unreadable_drain_records"] == []

    drain.path.unlink()
    missing = await _memory_health(db)
    assert missing["status"] == "warn", missing
    assert "its record is missing" in missing["message"]

    drain.settle("finished")
    assert (await _memory_health(db))["status"] == "pass"
    assert _only_record(database_file)["state"] == "done"


async def test_a_forked_child_reports_its_parents_pass_as_another_process(
    monkeypatch, tmp_path,
):
    """After a fork the parent's passes are the parent's, never the child's."""
    monkeypatch.setattr(drain_module, "_RUNTIME_ID", drain_module._RUNTIME_ID)
    database_file = tmp_path / "agent.db"
    db = AsyncDatabase(SQLiteBackend(str(database_file)))
    parent_pass = drain_module.ConsolidationDrain(
        db, agent_id="did:test:parent", deadline_seconds=DEADLINE_SECONDS
    )
    assert parent_pass.write()

    drain_module._forget_parent_passes()
    writes = _no_writes(monkeypatch)
    report = drain_module.reconcile_consolidation_drains(db)

    [status] = report.unsettled
    assert status.holder == "other_process"
    assert status.record["pass_id"] == parent_pass.pass_id
    assert drain_module._OWNED_PASSES == {}
    assert writes == []

    # The child's copy of the pass cannot settle or isolate the parent's
    # record: it is inert.
    parent_pass.settle("retired")
    parent_pass.isolate("caller_cancelled")
    assert parent_pass.write() is False
    assert not parent_pass.settled
    assert parent_pass.record["state"] == "draining"
    assert writes == []
    assert drain_module._OWNED_PASSES == {}


@pytest.mark.parametrize("settled_first", [True, False])
async def test_every_record_is_read_so_a_settled_one_hides_no_other(
    monkeypatch, tmp_path, settled_first,
):
    """Health does not stop at one record, whichever order they list in."""
    database_file = tmp_path / "agent.db"
    settled = _left_record(database_file, _live_other_owner(), state="done", agent_id="did:test:settled")
    running = _left_record(database_file, _live_other_owner(), state="isolated", agent_id="did:test:running")
    low, high = ("0" * 32, "f" * 32)
    settled["pass_id"], running["pass_id"] = (low, high) if settled_first else (high, low)
    _leave(database_file, settled)
    _leave(database_file, running)
    writes = _no_writes(monkeypatch)

    memory = await _memory_health(AsyncDatabase(SQLiteBackend(str(database_file))))

    assert memory["status"] == "warn", memory
    [entry] = memory["details"]["consolidation_drains"]
    assert entry["record"]["agent_id"] == "did:test:running"
    assert writes == []


@pytest.mark.parametrize("state", ["isolated", "done"])
async def test_a_legacy_single_record_is_read_as_one_more_record(
    monkeypatch, tmp_path, state,
):
    """The shared per-database record of this change's earlier shape is not lost."""
    database_file = tmp_path / "agent.db"
    legacy_path = database_file.with_name(database_file.name + LEGACY_DRAIN_RECORD_SUFFIX)
    legacy = _left_record(database_file, _exited_owner(), state=state, agent_id="did:test:legacy")
    legacy_path.write_text(json.dumps(legacy))
    current = _left_record(database_file, _live_other_owner(), state="done", agent_id="did:test:current")
    _leave(database_file, current)
    writes = _no_writes(monkeypatch)

    memory = await _memory_health(AsyncDatabase(SQLiteBackend(str(database_file))))

    if state == "isolated":
        assert memory["status"] == "warn", memory
        [entry] = memory["details"]["consolidation_drains"]
        assert entry["record_path"] == str(legacy_path.resolve())
        assert entry["holder"] == "owner_exited"
    else:
        assert memory["status"] == "pass", memory
    assert writes == []
    assert json.loads(legacy_path.read_text()) == legacy


async def test_a_settlement_a_failed_write_left_unrecorded_is_trusted_and_retried(
    monkeypatch, tmp_path,
):
    """The owning pass answers from memory, says the file lags, and retries it.

    Health does not pass while the durable record still claims running work,
    even though this process knows the work settled.
    """
    _deadline(monkeypatch, DEADLINE_SECONDS)
    real = drain_module.write_consolidation_drain_record
    refused_states = {"done"}

    def writer(path, record):
        if record["state"] in refused_states:
            raise PermissionError(13, "Permission denied", str(path))
        real(path, record)

    monkeypatch.setattr(drain_module, "write_consolidation_drain_record", writer)
    db = _SettlingDatabase(tmp_path / "agent.db", ABANDONED_WORK_FINISHED)

    with pytest.raises(MemoryConsolidationTimeoutError):
        await _hung_memory_system(db, "did:test:unrecorded").consolidate()
    assert _only_record(tmp_path / "agent.db")["state"] == "draining"

    still_refused = await _memory_health(db)
    assert still_refused["status"] == "warn", still_refused
    assert "writing that to its record failed" in still_refused["message"]
    assert "which still says draining" in still_refused["message"]
    [entry] = still_refused["details"]["consolidation_drains"]
    assert entry["holder"] == "settled_unrecorded"
    assert entry["record"]["state"] == "done"
    assert _only_record(tmp_path / "agent.db")["state"] == "draining"

    refused_states.clear()
    retried = await _memory_health(db)

    assert retried["status"] == "pass", retried
    assert retried["details"]["consolidation_drain"]["state"] == "done"
    record = _only_record(tmp_path / "agent.db")
    assert record["state"] == "done"
    assert record["cleanup"] == ABANDONED_WORK_FINISHED
    assert drain_module._OWNED_PASSES == {}


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

    holder = await _memory_health(worker.db)
    other = await _memory_health(AsyncDatabase(SQLiteBackend(str(worker.path))))

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
