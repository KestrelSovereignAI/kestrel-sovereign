"""What an SQLite close reports when it overruns its budget (#3460).

Closing the last connection to a WAL database checkpoints the whole -wal
into the main file and fsyncs it, inside the same 1.0s budget as worker
termination. These tests pin what an overrun reports, so a CI failure shows
how much WAL the close had to checkpoint without any extra step.
"""

import asyncio
import logging
import re
import threading
from pathlib import Path

import pytest

import kestrel_sovereign.storage.db.sqlite as sqlite_backend_module
from kestrel_sovereign.storage.db import ConnectionError, SQLiteBackend
from tests.utils.aiosqlite_workers import (
    aiosqlite_worker,
    delay_aiosqlite_worker_exit,
)

# Bound for awaits that must finish promptly. Generous, so a GC pause on a
# loaded xdist worker cannot fail a passing test.
_HANG_GUARD_SECONDS = 30.0

# The patched worker-shutdown window. Close budgets add one 10ms poll to it.
_SHORT_WINDOW_S = 0.05
_SHORT_BUDGET = "budget=0.06s"


class _SlowCloseConnection:
    """A sqlite3 connection whose ``close`` blocks in aiosqlite's worker.

    Stands in for a final checkpoint stalled on disk I/O: the worker is
    genuinely busy inside ``close`` until the test releases it, then the real
    close runs.
    """

    def __init__(self, real, entered: threading.Event, release: threading.Event):
        self._real = real
        self._entered = entered
        self._release = release

    def close(self) -> None:
        self._entered.set()
        self._release.wait(_HANG_GUARD_SECONDS)
        self._real.close()

    def __getattr__(self, name):
        return getattr(self._real, name)


@pytest.fixture
def short_close_window(monkeypatch):
    monkeypatch.setattr(
        sqlite_backend_module,
        "AIOSQLITE_WORKER_SHUTDOWN_TIMEOUT_S",
        _SHORT_WINDOW_S,
    )


async def _connected_wal_backend(path: Path) -> SQLiteBackend:
    """A file-backed backend with committed pages still in its WAL."""
    backend = SQLiteBackend(str(path))
    await backend.connect()
    await backend.execute("CREATE TABLE payload (body BLOB)")
    for _ in range(20):
        await backend.execute("INSERT INTO payload VALUES (randomblob(4000))")
    return backend


def _wal_frames(wal_bytes: int, page_size: int) -> int:
    return (wal_bytes - 32) // (24 + page_size)


async def _page_size(backend: SQLiteBackend) -> int:
    row = await backend.fetch_one("PRAGMA page_size")
    return int(row[0])


def _block_close(backend: SQLiteBackend):
    """Make the backend's next close block in its worker; return the handles."""
    connection = backend._ensure_connected()
    entered = threading.Event()
    release = threading.Event()
    connection._connection = _SlowCloseConnection(
        connection._connection, entered, release
    )
    return connection, entered, release


async def _retire(backend: SQLiteBackend, release: threading.Event, worker) -> None:
    """Let a blocked close finish and wait for its worker to retire."""
    release.set()
    retirement_tasks = [
        retained.retirement_task
        for retained in backend._retired_connection_closes.values()
        if retained.retirement_task is not None
    ]
    async with asyncio.timeout(_HANG_GUARD_SECONDS):
        await asyncio.gather(*retirement_tasks, return_exceptions=True)
    worker.join(timeout=_HANG_GUARD_SECONDS)
    assert not worker.is_alive()
    assert not backend.connection_retirement_pending


def _overrun_warnings(caplog) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if record.name == sqlite_backend_module.__name__
        and record.levelno == logging.WARNING
    ]


@pytest.mark.asyncio
async def test_slow_close_reports_elapsed_budget_and_wal_size(
    tmp_path, short_close_window, caplog
):
    db_path = tmp_path / "slow-close.db"
    backend = await _connected_wal_backend(db_path)
    page_size = await _page_size(backend)
    wal_bytes = Path(f"{db_path}-wal").stat().st_size
    assert wal_bytes > 0
    connection, _entered, release = _block_close(backend)
    worker = aiosqlite_worker(connection)
    try:
        with caplog.at_level(logging.WARNING, logger=sqlite_backend_module.__name__):
            with pytest.raises(
                ConnectionError, match="connection close did not complete"
            ) as raised:
                await backend.close()
    finally:
        await _retire(backend, release, worker)

    message = str(raised.value)
    assert _SHORT_BUDGET in message
    elapsed = float(re.search(r"elapsed=([0-9.]+)s", message).group(1))
    assert elapsed >= _SHORT_WINDOW_S
    assert "database=slow-close.db" in message
    assert str(tmp_path) not in message
    assert "journal_mode=wal" in message
    assert f"wal_bytes={wal_bytes}" in message
    assert f"wal_pages={_wal_frames(wal_bytes, page_size)}" in message
    assert [record.getMessage() for record in _overrun_warnings(caplog)] == [
        message
    ]


@pytest.mark.asyncio
async def test_overrun_interrupts_before_it_logs(
    tmp_path, short_close_window, monkeypatch
):
    """A log handler stalled on the same disk must not delay the interrupt."""
    backend = await _connected_wal_backend(tmp_path / "ordered.db")
    connection, _entered, release = _block_close(backend)
    worker = aiosqlite_worker(connection)
    order = []
    real_interrupt = connection.interrupt

    async def recording_interrupt():
        order.append("interrupt")
        await real_interrupt()

    class RecordingHandler(logging.Handler):
        def emit(self, record):
            if record.levelno == logging.WARNING:
                order.append("warning")

    monkeypatch.setattr(connection, "interrupt", recording_interrupt)
    handler = RecordingHandler()
    module_logger = logging.getLogger(sqlite_backend_module.__name__)
    module_logger.addHandler(handler)
    try:
        with pytest.raises(ConnectionError):
            await backend.close()
    finally:
        module_logger.removeHandler(handler)
        await _retire(backend, release, worker)

    assert order == ["interrupt", "warning"]


@pytest.mark.asyncio
async def test_normal_close_logs_nothing_and_checkpoints(tmp_path, caplog):
    db_path = tmp_path / "normal-close.db"
    backend = await _connected_wal_backend(db_path)
    assert Path(f"{db_path}-wal").stat().st_size > 0
    worker = aiosqlite_worker(backend._ensure_connected())

    with caplog.at_level(logging.DEBUG, logger=sqlite_backend_module.__name__):
        await backend.close()

    assert _overrun_warnings(caplog) == []
    assert not backend.is_connected
    assert not worker.is_alive()
    assert not backend.connection_retirement_pending
    # An idle close is not interrupted, so its final checkpoint ran and
    # removed the WAL, as before.
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "normal-close.db"
    ]


@pytest.mark.asyncio
async def test_failed_wal_read_reports_unknown_and_keeps_close_error(
    tmp_path, short_close_window, monkeypatch, caplog
):
    db_path = tmp_path / "unreadable-wal.db"
    backend = await _connected_wal_backend(db_path)
    connection, _entered, release = _block_close(backend)
    worker = aiosqlite_worker(connection)
    real_stat = Path.stat

    def failing_wal_stat(self, *args, **kwargs):
        if self.name.endswith("-wal"):
            raise PermissionError("injected -wal stat failure")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", failing_wal_stat)
    try:
        with caplog.at_level(logging.WARNING, logger=sqlite_backend_module.__name__):
            with pytest.raises(ConnectionError) as raised:
                await backend.close()
    finally:
        monkeypatch.setattr(Path, "stat", real_stat)
        await _retire(backend, release, worker)

    message = str(raised.value)
    assert message.startswith("SQLite connection close did not complete")
    assert "wal_bytes=unknown wal_pages=unknown" in message
    assert "journal_mode=wal" in message
    assert len(_overrun_warnings(caplog)) == 1


@pytest.mark.asyncio
async def test_non_wal_database_reports_journal_mode_without_wal(
    short_close_window, caplog
):
    backend = SQLiteBackend(":memory:")
    await backend.connect()
    await backend.execute("CREATE TABLE payload (body BLOB)")
    connection, _entered, release = _block_close(backend)
    worker = aiosqlite_worker(connection)
    try:
        with caplog.at_level(logging.WARNING, logger=sqlite_backend_module.__name__):
            with pytest.raises(
                ConnectionError, match="connection close did not complete"
            ) as raised:
                await backend.close()
    finally:
        await _retire(backend, release, worker)

    message = str(raised.value)
    assert "database=:memory:" in message
    assert "journal_mode=memory" in message
    assert "wal_" not in message
    assert _SHORT_BUDGET in message


@pytest.mark.asyncio
async def test_non_wal_database_closes_normally(caplog):
    backend = SQLiteBackend(":memory:")
    await backend.connect()
    assert backend._journal_mode == "memory"

    with caplog.at_level(logging.DEBUG, logger=sqlite_backend_module.__name__):
        await backend.close()

    assert _overrun_warnings(caplog) == []
    assert not backend.is_connected


@pytest.mark.asyncio
async def test_known_non_wal_journal_mode_never_reads_a_stray_wal(tmp_path):
    db_path = tmp_path / "rollback.db"
    db_path.write_bytes(b"")
    Path(f"{db_path}-wal").write_bytes(b"x" * 100)

    diagnostics = sqlite_backend_module._capture_close_diagnostics(
        db_path, "delete"
    )

    assert diagnostics.journal_mode == "delete"
    assert diagnostics.wal is None
    assert "wal_" not in diagnostics.describe(diagnostics.started_at)


@pytest.mark.asyncio
async def test_worker_termination_overrun_reports_wal_read_before_close(
    tmp_path, short_close_window
):
    """The close checkpointed the WAL away; the report keeps its size."""
    db_path = tmp_path / "stuck-worker.db"
    release_worker = threading.Event()
    worker_exit_delayed = threading.Event()
    with delay_aiosqlite_worker_exit(release_worker, worker_exit_delayed):
        backend = await _connected_wal_backend(db_path)
        page_size = await _page_size(backend)
        wal_bytes = Path(f"{db_path}-wal").stat().st_size
        worker = aiosqlite_worker(backend._ensure_connected())
        retirement_tasks = []
        try:
            with pytest.raises(
                ConnectionError, match="worker did not terminate"
            ) as raised:
                await backend.close()
            retirement_tasks = [
                retained.retirement_task
                for retained in backend._retired_connection_closes.values()
            ]
        finally:
            release_worker.set()
            async with asyncio.timeout(_HANG_GUARD_SECONDS):
                await asyncio.gather(*retirement_tasks, return_exceptions=True)
            worker.join(timeout=_HANG_GUARD_SECONDS)

    assert not Path(f"{db_path}-wal").exists()
    message = str(raised.value)
    assert "database=stuck-worker.db" in message
    assert "journal_mode=wal" in message
    assert f"wal_bytes={wal_bytes}" in message
    assert f"wal_pages={_wal_frames(wal_bytes, page_size)}" in message
    assert _SHORT_BUDGET in message


def test_wal_size_counts_frames_with_the_known_page_size(tmp_path):
    db_path = tmp_path / "framed.db"
    # Enough frames that their 24-byte headers add up to more than a page,
    # so a count that ignores them comes out different.
    frames = 30
    Path(f"{db_path}-wal").write_bytes(bytes(32 + frames * (24 + 512)))

    wal = sqlite_backend_module._read_wal_size(db_path, 512)

    assert wal == sqlite_backend_module._WalSize(
        size=32 + frames * (24 + 512), pages=frames
    )


def test_wal_size_without_a_page_size_reports_bytes_only(tmp_path):
    db_path = tmp_path / "unsized.db"
    Path(f"{db_path}-wal").write_bytes(bytes(64))

    wal = sqlite_backend_module._read_wal_size(db_path, None)

    assert wal == sqlite_backend_module._WalSize(size=64, pages=None)


def test_wal_size_never_opens_the_wal(tmp_path, monkeypatch):
    """Every close reads this; only a stat, never a read of the file."""
    db_path = tmp_path / "stat-only.db"
    Path(f"{db_path}-wal").write_bytes(bytes(32 + 2 * (24 + 512)))

    def refuse_open(self, *args, **kwargs):
        raise AssertionError(f"opened {self.name}")

    monkeypatch.setattr(Path, "open", refuse_open)

    assert sqlite_backend_module._read_wal_size(db_path, 512).pages == 2


@pytest.mark.asyncio
async def test_missing_wal_is_reported_absent(tmp_path):
    diagnostics = sqlite_backend_module._capture_close_diagnostics(
        tmp_path / "no-wal.db", "wal"
    )

    assert diagnostics.wal == sqlite_backend_module._WalSize(absent=True)
    assert "wal_bytes=absent wal_pages=absent" in diagnostics.describe(
        diagnostics.started_at
    )


@pytest.mark.asyncio
async def test_snapshot_close_reports_primary_journal_mode(
    tmp_path, monkeypatch
):
    backend = await _connected_wal_backend(tmp_path / "snapshot.db")
    snapshot = await backend._open_snapshot_read_connection()
    entered = threading.Event()
    release = threading.Event()
    snapshot._connection = _SlowCloseConnection(
        snapshot._connection, entered, release
    )
    worker = aiosqlite_worker(snapshot)
    try:
        # Only the deliberately stalled snapshot close has a 60ms budget.
        # Its independently owned primary connection is ordinary teardown,
        # not another injected slow-close operation under that short window.
        with monkeypatch.context() as fault:
            fault.setattr(sqlite_backend_module, "AIOSQLITE_WORKER_SHUTDOWN_TIMEOUT_S", _SHORT_WINDOW_S)
            with pytest.raises(ConnectionError) as raised:
                await sqlite_backend_module._close_aiosqlite_connection(
                    snapshot,
                    retained_closes=backend._retired_connection_closes,
                    diagnostics=backend._close_diagnostics(),
                )
    finally:
        await _retire(backend, release, worker)
        await backend.close()

    message = str(raised.value)
    assert "database=snapshot.db" in message
    assert _SHORT_BUDGET in message
    assert "journal_mode=wal" in message
    assert "wal_bytes=" in message


@pytest.mark.asyncio
async def test_every_backend_close_names_the_database_it_closes(
    tmp_path, monkeypatch
):
    """Snapshot reads, backups and the primary close each pass their file."""
    backend = await _connected_wal_backend(tmp_path / "primary.db")
    reported = []
    real_close = sqlite_backend_module._close_aiosqlite_connection

    async def record_diagnostics(connection, **kwargs):
        reported.append(kwargs.get("diagnostics"))
        await real_close(connection, **kwargs)

    monkeypatch.setattr(
        sqlite_backend_module, "_close_aiosqlite_connection", record_diagnostics
    )
    holding = asyncio.Event()
    release = asyncio.Event()

    async def hold_transaction():
        async with backend.transaction():
            await backend.execute("INSERT INTO payload VALUES (x'00')")
            holding.set()
            await release.wait()

    holder = asyncio.create_task(hold_transaction())
    try:
        async with asyncio.timeout(_HANG_GUARD_SECONDS):
            await holding.wait()
            # Another task's open transaction routes this read to a snapshot
            # connection, which is closed when the read finishes.
            assert await backend.fetch_one("SELECT count(*) FROM payload") == (20,)
    finally:
        release.set()
        await holder
    await backend.backup_to(str(tmp_path / "copy.db"))
    await backend.close()

    assert [
        (diagnostics.database, diagnostics.journal_mode)
        for diagnostics in reported
    ] == [
        ("primary.db", "wal"),
        ("copy.db", "unknown"),
        ("primary.db", "wal"),
    ]


@pytest.mark.asyncio
async def test_session_factory_close_passes_its_database_to_driver_closes(
    tmp_path, monkeypatch
):
    import kestrel_sovereign.storage.sqla.session as session_module
    from sqlalchemy import text

    from kestrel_sovereign.storage.async_database import AsyncDatabase
    from kestrel_sovereign.storage.sqla import make_session_factory

    db = await AsyncDatabase.sqlite(str(tmp_path / "factory.db"))
    captured = []
    handed_off = []
    real_capture = session_module._capture_close_diagnostics

    def record_capture(db_path, journal_mode):
        diagnostics = real_capture(db_path, journal_mode)
        captured.append(diagnostics)
        return diagnostics

    monkeypatch.setattr(
        session_module, "_capture_close_diagnostics", record_capture
    )
    try:
        factory = make_session_factory(db)
        assert factory._sqlite_path == db.backend.db_path
        real_close_live = factory._close_live_sqlite_connections

        async def record_close_live(*, deadline, diagnostics=None):
            handed_off.append(diagnostics)
            await real_close_live(deadline=deadline, diagnostics=diagnostics)

        monkeypatch.setattr(
            factory, "_close_live_sqlite_connections", record_close_live
        )
        async with factory.read_session() as session:
            await session.execute(text("SELECT 1"))
        await factory.close()
        # A retried close reports what the first attempt saw before it began.
        await factory.close()
    finally:
        await db.close()

    assert [diagnostics.database for diagnostics in captured] == ["factory.db"]
    assert len(handed_off) >= 2
    assert all(
        (diagnostics.database, diagnostics.wal)
        == (captured[0].database, captured[0].wal)
        for diagnostics in handed_off
    )
    # ...but each attempt is timed from its own start.
    assert handed_off[1].started_at > handed_off[0].started_at
