"""Regression coverage for dropped ``signal_log`` audit rows (#2660).

Production accumulated 3,323 ``Failed to write signal_log entry`` errors —
``sqlite3.OperationalError: database is locked`` raised from
``SignalLogStore.append`` — across roughly two months before anyone noticed.
The symptom stopped when #2718 routed owned aiosqlite connections through
``_close_aiosqlite_connection``; leaked connections against the same file are
contention that a *per-connection* write lock structurally cannot serialize,
which is why ``_write_guard`` was present throughout and did not help.

Two gaps let that happen and are what this module covers:

1. No test drove concurrent writes against one backend while signal outcomes
   were being appended, so nothing would have caught the loss or would catch a
   recurrence.
2. A failed append only produced a log line. Nothing counted it, so the loss
   was invisible anywhere an operator looks.
"""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import datetime, timezone

import pytest

from kestrel_sdk.signals import (
    RedactionPolicy,
    Signal,
    SignalMode,
    SignalResult,
    SourceRegistration,
    Status,
    Trust,
)
from kestrel_sovereign.features.health.checks import check_signal_audit_log
from kestrel_sovereign.signals import (
    OrderedLockManager,
    SignalDispatcher,
    SignalLogStore,
    SourceRegistry,
)
from kestrel_sovereign.storage.db import SQLiteBackend
from kestrel_sovereign.storage.db import sqlite as sqlite_module



def _registration(name: str = "contention") -> SourceRegistration:
    async def _handler(payload):
        return None

    return SourceRegistration(
        name=name,
        schema=dict,
        default_mode=SignalMode.ACTION,
        allowed_modes=frozenset({SignalMode.ACTION}),
        handler=_handler,
        trust=Trust.TRUSTED,
        log_redaction=RedactionPolicy(summarize=lambda p: "<redacted>"),
        retention_days=7,
    )


def _signal(index: int) -> Signal:
    return Signal(
        source="contention",
        kind="test",
        mode=SignalMode.ACTION,
        payload={"index": index},
        target_agent="agent-test",
    )


def _result(signal: Signal) -> SignalResult:
    return SignalResult(
        signal_id=signal.id,
        status=Status.OK,
        mode=SignalMode.ACTION,
        duration_ms=1,
    )


@pytest.fixture
async def backend(tmp_path):
    be = SQLiteBackend(str(tmp_path / "contention.db"))
    await be.connect()
    yield be
    await be.close()


@pytest.mark.asyncio
async def test_a_foreign_connection_holding_the_write_lock_drops_the_append(
    backend, tmp_path
):
    """Control: prove this harness can actually observe the production failure.

    A first attempt at coverage here drove 40 concurrent appends against one
    backend and asserted the row count. It passed — and it still passed with
    ``_write_guard`` mutated to a no-op, because aiosqlite serializes one
    connection on a single worker thread regardless. That test proved nothing,
    and its green would have been read as protection.

    The real failure needs a *second connection to the same file*, which is
    what the in-process, per-connection write lock structurally cannot reach.
    This test creates that condition deliberately and asserts the append fails,
    establishing that a passing sibling test below means something.
    """
    store = SignalLogStore(backend)
    await store.initialize()
    registration = _registration()

    # Fail fast instead of waiting out the production 30s timeout.
    await backend._connection.execute("PRAGMA busy_timeout=50")

    foreign = sqlite3.connect(str(tmp_path / "contention.db"), timeout=0.05)
    try:
        foreign.execute("BEGIN IMMEDIATE")  # take the file's write lock
        signal = _signal(0)
        with pytest.raises(Exception) as excinfo:
            await store.append(signal, registration, _result(signal))
        assert "locked" in str(excinfo.value).lower()
    finally:
        foreign.rollback()
        foreign.close()

    # And the same append succeeds once the foreign writer releases: the row
    # was lost to contention, not to anything malformed about it.
    await backend._connection.execute("PRAGMA busy_timeout=30000")
    signal = _signal(1)
    await store.append(signal, registration, _result(signal))
    assert await backend.fetch_val("SELECT COUNT(*) FROM signal_log") == 1


@pytest.mark.asyncio
async def test_snapshot_read_connections_are_closed(backend):
    """No owned connection outlives its use (#2718 — the actual fix).

    The snapshot-read path opens a *second* connection to the same file so a
    non-owner task reads committed state during another task's transaction.
    Leaking those is the contention class the write lock cannot serialize, and
    closing them is what stopped 3,323 dropped rows in production.

    Asserting on ``_connection`` is reaching into aiosqlite, and that is the
    point: connection lifetime is exactly what regressed, so the assertion has
    to be on the connection rather than on a symptom that only appears under
    load.
    """
    opened: list = []
    real_connect = sqlite_module.aiosqlite.connect

    def _tracking_connect(*args, **kwargs):
        handle = real_connect(*args, **kwargs)
        opened.append(handle)
        return handle

    sqlite_module.aiosqlite.connect = _tracking_connect
    try:
        await backend.execute(
            "CREATE TABLE IF NOT EXISTS turn_writes (id INTEGER PRIMARY KEY, body TEXT)"
        )

        reader_done = asyncio.Event()
        holder_may_commit = asyncio.Event()

        async def _holder() -> None:
            async with backend.transaction():
                await backend.execute(
                    "INSERT INTO turn_writes (body) VALUES (?)", ("in-txn",)
                )
                await reader_done.wait()

        async def _reader() -> None:
            # A different task than the transaction owner -> snapshot path.
            await asyncio.sleep(0.02)
            await backend.fetch_all("SELECT * FROM turn_writes")
            reader_done.set()

        await asyncio.gather(_holder(), _reader())
        holder_may_commit.set()
    finally:
        sqlite_module.aiosqlite.connect = real_connect

    assert opened, "snapshot read path never opened a connection to track"
    still_open = [c for c in opened if getattr(c, "_connection", None) is not None]
    assert not still_open, (
        f"{len(still_open)} owned aiosqlite connection(s) left open; leaked "
        "connections are the contention the per-connection write lock cannot "
        "serialize"
    )


@pytest.mark.asyncio
async def test_transaction_holder_does_not_starve_signal_log_append(backend):
    """An append issued while another task holds a transaction still lands.

    The write unit is held for a whole BEGIN..COMMIT span (#1675), so an append
    racing a turn's transaction must wait for it and then succeed — not fail
    with "database is locked". This is the narrower shape of the same race.
    """
    store = SignalLogStore(backend)
    await store.initialize()
    registration = _registration()

    await backend.execute(
        "CREATE TABLE IF NOT EXISTS turn_writes (id INTEGER PRIMARY KEY, body TEXT)"
    )

    append_started = asyncio.Event()

    async def _holder() -> None:
        async with backend.transaction():
            await backend.execute(
                "INSERT INTO turn_writes (body) VALUES (?)", ("in-txn",)
            )
            # Let the append task reach the write guard before committing.
            append_started.set()
            await asyncio.sleep(0.05)

    async def _appender() -> None:
        await append_started.wait()
        signal = _signal(0)
        await store.append(signal, registration, _result(signal))

    await asyncio.gather(_holder(), _appender())

    assert await backend.fetch_val("SELECT COUNT(*) FROM signal_log") == 1
    assert await backend.fetch_val("SELECT COUNT(*) FROM turn_writes") == 1


# ---------------------------------------------------------------------------
# The loss must leave a record an operator can see (#2660 criterion 2)
# ---------------------------------------------------------------------------


class _DispatcherStub:
    """Minimal stand-in exposing only the accounting the health check reads."""

    def __init__(self, count, last=None):
        self.log_write_failure_count = count
        self.last_log_write_failure = last


class _FakeAgent:
    """Satisfies DispatcherAgent; collects background tasks so we can drain."""

    def __init__(self):
        self.background_tasks: list = []

    @property
    def did(self) -> str:
        return "agent-test"

    async def process_input(self, prompt: str):
        return "ok"

    def _track_background_task(self, coro, *, name: str):
        task = asyncio.create_task(coro, name=name)
        self.background_tasks.append(task)
        return task


@pytest.mark.asyncio
async def test_dispatch_counts_a_dropped_audit_row_end_to_end(backend):
    """The counter is wired to the real failure, not just to the health check.

    Without this the accounting could be dead code and the stub-driven health
    tests below would still pass — asserting the reporting shape while nothing
    ever incremented it. Drives a genuine dispatch whose store write raises and
    asserts the loss was counted and attributed.
    """
    store = SignalLogStore(backend)
    await store.initialize()

    async def _exploding_append(*args, **kwargs):
        raise RuntimeError("database is locked")

    store.append = _exploding_append  # type: ignore[method-assign]

    agent = _FakeAgent()
    registry = SourceRegistry()
    dispatcher = SignalDispatcher(
        agent=agent,
        registry=registry,
        lock_manager=OrderedLockManager(),
        store=store,
    )
    registry.register(_registration(name="contention"))

    assert dispatcher.log_write_failure_count == 0

    result = await dispatcher.dispatch_signal(_signal(0))
    # The dispatch itself still succeeds — losing the audit row must not fail
    # the work the row describes, which is precisely why it went unnoticed.
    assert result.status == Status.OK

    pending = [t for t in agent.background_tasks if not t.done()]
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)

    assert dispatcher.log_write_failure_count == 1
    last = dispatcher.last_log_write_failure
    assert last is not None
    assert "database is locked" in last.error
    assert last.signal_id


@pytest.mark.asyncio
async def test_health_check_passes_when_no_rows_dropped():
    agent = type("A", (), {"dispatcher": _DispatcherStub(0)})()
    result = await check_signal_audit_log(agent)
    assert result["status"] == "pass"
    assert result["name"] == "signal_audit_log"


@pytest.mark.asyncio
async def test_health_check_warns_and_names_the_lost_signal():
    from kestrel_sovereign.signals import SignalLogWriteFailure

    failure = SignalLogWriteFailure(
        signal_id="sig_abc123",
        error="QueryError: database is locked",
        failed_at=datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc),
    )
    agent = type("A", (), {"dispatcher": _DispatcherStub(7, failure)})()

    result = await check_signal_audit_log(agent)

    assert result["status"] == "warn"
    assert "7" in result["message"]
    # Actionability is the point: an operator must be able to identify WHICH
    # signal lost its audit row, not merely that some number were lost.
    assert result["details"]["dropped"] == 7
    assert result["details"]["last_signal_id"] == "sig_abc123"
    assert "database is locked" in result["details"]["last_error"]


@pytest.mark.asyncio
async def test_health_check_tolerates_agent_without_dispatcher():
    """Agents built without a dispatcher must not fail the health surface."""
    agent = type("A", (), {})()
    result = await check_signal_audit_log(agent)
    assert result["status"] == "pass"
