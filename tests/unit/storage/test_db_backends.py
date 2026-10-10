"""
Tests for database backend abstraction layer.
"""
import asyncio
import logging
import sqlite3
import threading
import time
import weakref
from contextlib import suppress
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import pytest
import kestrel_sovereign.storage.db.sqlite as sqlite_backend_module
from kestrel_sovereign.storage.db import (
    ConnectionError,
    QueryError,
    SQLiteBackend,
    sqlite_to_postgres,
    postgres_to_sqlite,
    normalize_schema,
)
from tests.utils.aiosqlite_workers import (
    aiosqlite_worker,
    delay_aiosqlite_worker_exit,
    wait_for_lifecycle_checkpoint,
    wait_until_aiosqlite_worker_exit_is_delayed,
)


# Bound for awaits that must finish promptly. Generous, so a GC pause on a
# loaded xdist worker cannot fail a passing test.
_HANG_GUARD_SECONDS = 30.0


async def _read_owns_open_transaction(database):
    return database.owns_open_transaction


def _sqlite_operational_error(
    code: int, name: str, message: str
) -> sqlite3.OperationalError:
    """An OperationalError carrying the result code sqlite3 attaches."""
    exc = sqlite3.OperationalError(message)
    exc.sqlite_errorcode = code
    exc.sqlite_errorname = name
    return exc


def _sqlite_interrupted() -> sqlite3.OperationalError:
    return _sqlite_operational_error(
        sqlite3.SQLITE_INTERRUPT, "SQLITE_INTERRUPT", "interrupted"
    )


class TestPlaceholderConversion:
    """Test SQL placeholder conversion utilities."""
    
    def test_sqlite_to_postgres_simple(self):
        query = "SELECT * FROM users WHERE id = ? AND name = ?"
        result, count = sqlite_to_postgres(query)
        assert result == "SELECT * FROM users WHERE id = $1 AND name = $2"
        assert count == 2
    
    def test_sqlite_to_postgres_no_placeholders(self):
        query = "SELECT * FROM users"
        result, count = sqlite_to_postgres(query)
        assert result == "SELECT * FROM users"
        assert count == 0
    
    def test_sqlite_to_postgres_insert(self):
        query = "INSERT INTO users (name, email) VALUES (?, ?)"
        result, count = sqlite_to_postgres(query)
        assert result == "INSERT INTO users (name, email) VALUES ($1, $2)"
        assert count == 2
    
    def test_sqlite_to_postgres_preserves_strings(self):
        query = "SELECT * FROM users WHERE name = ? AND status = 'active?'"
        result, count = sqlite_to_postgres(query)
        assert result == "SELECT * FROM users WHERE name = $1 AND status = 'active?'"
        assert count == 1
    
    def test_sqlite_to_postgres_double_quotes(self):
        query = 'SELECT * FROM "table?" WHERE id = ?'
        result, count = sqlite_to_postgres(query)
        assert result == 'SELECT * FROM "table?" WHERE id = $1'
        assert count == 1
    
    def test_postgres_to_sqlite_simple(self):
        query = "SELECT * FROM users WHERE id = $1 AND name = $2"
        result = postgres_to_sqlite(query)
        assert result == "SELECT * FROM users WHERE id = ? AND name = ?"
    
    def test_postgres_to_sqlite_high_numbers(self):
        query = "SELECT * FROM t WHERE a=$1 AND b=$10 AND c=$100"
        result = postgres_to_sqlite(query)
        assert result == "SELECT * FROM t WHERE a=? AND b=? AND c=?"


class TestSchemaConversion:
    """Test schema normalization for different backends."""
    
    def test_normalize_for_sqlite_noop(self):
        schema = "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)"
        result = normalize_schema(schema, "sqlite")
        assert result == schema
    
    def test_normalize_for_postgres_autoincrement(self):
        schema = "CREATE TABLE users (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT)"
        result = normalize_schema(schema, "postgres")
        assert "SERIAL PRIMARY KEY" in result
        assert "AUTOINCREMENT" not in result
    
    def test_normalize_for_postgres_real(self):
        schema = "CREATE TABLE data (value REAL)"
        result = normalize_schema(schema, "postgres")
        assert "DOUBLE PRECISION" in result
        assert "REAL" not in result
    
    def test_normalize_for_postgres_blob(self):
        schema = "CREATE TABLE files (content BLOB)"
        result = normalize_schema(schema, "postgres")
        assert "BYTEA" in result
        assert "BLOB" not in result
    
    def test_normalize_for_sqlite_serial(self):
        schema = "CREATE TABLE users (id SERIAL PRIMARY KEY, name TEXT)"
        result = normalize_schema(schema, "sqlite")
        assert "INTEGER" in result
        assert "SERIAL" not in result


class TestSQLiteBackend:
    """Test SQLite backend implementation."""
    
    @pytest.fixture
    async def backend(self, tmp_path):
        """Create a temporary SQLite backend."""
        db_path = str(tmp_path / "test.db")
        backend = SQLiteBackend(db_path)
        await backend.connect()
        yield backend
        await backend.close()
    
    @pytest.mark.asyncio
    async def test_backend_type(self, backend):
        assert backend.backend_type == "sqlite"

    @pytest.mark.asyncio
    async def test_nested_transaction_strategy_is_joined(self, backend):
        assert backend.nested_transaction_strategy == "joined"

    @pytest.mark.asyncio
    async def test_owns_open_transaction_only_in_the_opening_task(self, backend):
        """True exactly where a ``transaction()`` would join an open one; a
        task spawned inside it opens its own instead."""
        assert backend.owns_open_transaction is False
        async with backend.transaction():
            assert backend.owns_open_transaction is True
            child = asyncio.get_running_loop().create_task(
                _read_owns_open_transaction(backend)
            )
            assert await child is False
        assert backend.owns_open_transaction is False

    @pytest.mark.asyncio
    async def test_is_connected(self, backend):
        assert backend.is_connected is True
    
    @pytest.mark.asyncio
    async def test_create_table(self, backend):
        await backend.execute(
            "CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)"
        )
        exists = await backend.table_exists("test")
        assert exists is True
    
    @pytest.mark.asyncio
    async def test_insert_and_fetch(self, backend):
        await backend.execute(
            "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)"
        )
        await backend.execute(
            "INSERT INTO users (id, name) VALUES (?, ?)",
            (1, "Alice")
        )
        
        row = await backend.fetch_one(
            "SELECT * FROM users WHERE id = ?", (1,)
        )
        assert row == (1, "Alice")
    
    @pytest.mark.asyncio
    async def test_fetch_all(self, backend):
        await backend.execute(
            "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)"
        )
        await backend.execute("INSERT INTO users (id, name) VALUES (?, ?)", (1, "Alice"))
        await backend.execute("INSERT INTO users (id, name) VALUES (?, ?)", (2, "Bob"))
        
        rows = await backend.fetch_all("SELECT * FROM users ORDER BY id")
        assert len(rows) == 2
        assert rows[0] == (1, "Alice")
        assert rows[1] == (2, "Bob")
    
    @pytest.mark.asyncio
    async def test_fetch_val(self, backend):
        await backend.execute(
            "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)"
        )
        await backend.execute("INSERT INTO users (id, name) VALUES (?, ?)", (1, "Alice"))
        
        count = await backend.fetch_val("SELECT COUNT(*) FROM users")
        assert count == 1
    
    @pytest.mark.asyncio
    async def test_execute_many(self, backend):
        await backend.execute(
            "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)"
        )
        await backend.execute_many(
            "INSERT INTO users (id, name) VALUES (?, ?)",
            [(1, "Alice"), (2, "Bob"), (3, "Charlie")]
        )
        
        count = await backend.fetch_val("SELECT COUNT(*) FROM users")
        assert count == 3
    
    @pytest.mark.asyncio
    async def test_transaction_commit(self, backend):
        await backend.execute(
            "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)"
        )
        
        async with backend.transaction():
            await backend.execute("INSERT INTO users (id, name) VALUES (?, ?)", (1, "Alice"))
            await backend.execute("INSERT INTO users (id, name) VALUES (?, ?)", (2, "Bob"))
        
        count = await backend.fetch_val("SELECT COUNT(*) FROM users")
        assert count == 2
    
    @pytest.mark.asyncio
    async def test_transaction_rollback(self, backend):
        from kestrel_sovereign.storage.db import TransactionError
        
        await backend.execute(
            "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)"
        )
        
        with pytest.raises(TransactionError):
            async with backend.transaction():
                await backend.execute("INSERT INTO users (id, name) VALUES (?, ?)", (1, "Alice"))
                raise ValueError("Simulated error")
        
        count = await backend.fetch_val("SELECT COUNT(*) FROM users")
        assert count == 0
    
    @pytest.mark.asyncio
    async def test_concurrent_autocommit_writes_all_persist(self, backend):
        """#1675: concurrent autocommit writers on the shared connection must
        each be an atomic write unit — no lost writes."""
        import asyncio

        await backend.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")

        async def write(i):
            await backend.execute("INSERT INTO t (id, v) VALUES (?, ?)", (i, f"v{i}"))

        await asyncio.gather(*(write(i) for i in range(25)))
        count = await backend.fetch_val("SELECT COUNT(*) FROM t")
        assert count == 25

    @pytest.mark.asyncio
    async def test_concurrent_writer_waits_for_open_transaction(self, backend):
        """#1675: a transaction is one atomic write unit — a concurrent
        autocommit writer must wait for it to finish, not interleave into its
        connection-scoped transaction (which a sibling rollback could discard)."""
        import asyncio

        await backend.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
        in_txn = asyncio.Event()
        release = asyncio.Event()

        async def txn_task():
            async with backend.transaction():
                await backend.execute("INSERT INTO t (id, v) VALUES (1, 'a')")
                in_txn.set()
                await release.wait()  # hold the transaction open

        async def writer_task():
            await in_txn.wait()
            await backend.execute("INSERT INTO t (id, v) VALUES (2, 'b')")

        t1 = asyncio.create_task(txn_task())
        t2 = asyncio.create_task(writer_task())

        await in_txn.wait()
        await asyncio.sleep(0.05)  # give the writer a chance to (wrongly) proceed
        # The writer is blocked on the write lock, and this sibling read sees
        # only committed rows rather than the transaction's uncommitted row.
        mid = await backend.fetch_all("SELECT id FROM t ORDER BY id")
        assert mid == []
        assert not t2.done()

        release.set()
        await asyncio.gather(t1, t2)
        final = await backend.fetch_all("SELECT id FROM t ORDER BY id")
        assert final == [(1,), (2,)]

    @pytest.mark.asyncio
    async def test_concurrent_reader_does_not_see_uncommitted_transaction(self, backend):
        """#1745: sibling reads must not observe another task's uncommitted
        rows on SQLite's shared write connection."""
        import asyncio

        await backend.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
        in_txn = asyncio.Event()
        release = asyncio.Event()

        async def txn_task():
            async with backend.transaction():
                await backend.execute("INSERT INTO t (id, v) VALUES (1, 'a')")
                owner_mid = await backend.fetch_all("SELECT id FROM t ORDER BY id")
                in_txn.set()
                await release.wait()
                return owner_mid

        async def reader_task():
            await in_txn.wait()
            return await backend.fetch_all("SELECT id FROM t ORDER BY id")

        t1 = asyncio.create_task(txn_task())
        t2 = asyncio.create_task(reader_task())

        sibling_mid = await t2
        release.set()
        owner_mid = await t1

        assert owner_mid == [(1,)]
        assert sibling_mid == []
        final = await backend.fetch_all("SELECT id FROM t ORDER BY id")
        assert final == [(1,)]

    @pytest.mark.asyncio
    async def test_canceled_transaction_rolls_back_and_frees_lock(self, backend):
        """#1675: a transaction canceled mid-flight (CancelledError, a
        BaseException) must roll back and release the write lock — not leak an
        open transaction the next writer would inherit and commit."""
        import asyncio

        await backend.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)")
        started = asyncio.Event()

        async def txn_task():
            async with backend.transaction():
                await backend.execute("INSERT INTO t (id) VALUES (1)")
                started.set()
                await asyncio.sleep(10)  # cancelled here, before commit

        t = asyncio.create_task(txn_task())
        await started.wait()
        t.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t

        # The canceled transaction's write was rolled back.
        assert await backend.fetch_all("SELECT id FROM t") == []
        # The lock was released, so a fresh writer proceeds.
        await backend.execute("INSERT INTO t (id) VALUES (2)")
        assert await backend.fetch_all("SELECT id FROM t") == [(2,)]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "write_method", ["execute", "execute_many", "execute_script"]
    )
    async def test_worker_blocked_write_cancellation_hands_off_rollback(
        self, backend, write_method,
    ):
        """Every autocommit write API returns cancellation before worker drain."""
        await backend.execute("CREATE TABLE t (value INTEGER NOT NULL)")
        await backend.execute("INSERT INTO t (value) VALUES (0)")
        entered_worker = threading.Event()
        release_worker = threading.Event()
        watchdog = threading.Timer(2.0, release_worker.set)

        def block_in_worker(value):
            entered_worker.set()
            release_worker.wait()
            return value + 1

        conn = backend._ensure_connected()
        await conn.create_function("block_in_worker", 1, block_in_worker)

        if write_method == "execute":
            blocked_write = backend.execute(
                "UPDATE t SET value = block_in_worker(value)"
            )
        elif write_method == "execute_many":
            blocked_write = backend.execute_many(
                "UPDATE t SET value = block_in_worker(value) WHERE ?",
                [(1,)],
            )
        else:
            blocked_write = backend.execute_script(
                "BEGIN; UPDATE t SET value = block_in_worker(value);"
            )

        task = asyncio.create_task(blocked_write)
        following_write = None
        try:
            watchdog.start()
            async with asyncio.timeout(0.5):
                while not entered_worker.is_set():
                    await asyncio.sleep(0.005)

            task.cancel()
            async with asyncio.timeout(0.2):
                with pytest.raises(asyncio.CancelledError):
                    await task

            # A later writer is fenced on the retained rollback rather than
            # entering the shared connection while the worker is still stuck.
            following_write = asyncio.create_task(
                backend.execute("UPDATE t SET value = value + 1")
            )
            await asyncio.sleep(0.02)
            assert not following_write.done()

            release_worker.set()
            async with asyncio.timeout(1.0):
                await following_write
            assert await backend.fetch_all("SELECT value FROM t") == [(1,)]
        finally:
            watchdog.cancel()
            release_worker.set()
            if not task.done():
                task.cancel()
            if following_write is not None and not following_write.done():
                following_write.cancel()
            await asyncio.gather(
                task,
                *([following_write] if following_write is not None else []),
                return_exceptions=True,
            )

    @pytest.mark.asyncio
    async def test_worker_blocked_read_cancellation_hands_off_drain(self, backend):
        """A cancelled fetch returns before its queued cursor close can drain."""
        await backend.execute("CREATE TABLE t (value INTEGER NOT NULL)")
        await backend.execute_many(
            "INSERT INTO t (value) VALUES (?)", [(1,), (2,)]
        )
        entered_worker = threading.Event()
        release_worker = threading.Event()
        watchdog = threading.Timer(2.0, release_worker.set)
        calls = 0

        def block_on_second_row(value):
            nonlocal calls
            calls += 1
            if calls == 2:
                entered_worker.set()
                release_worker.wait()
            return value

        conn = backend._ensure_connected()
        await conn.create_function("block_on_second_row", 1, block_on_second_row)
        blocked_read = asyncio.create_task(
            backend.fetch_all("SELECT block_on_second_row(value) FROM t")
        )
        following_write = None
        try:
            watchdog.start()
            async with asyncio.timeout(0.5):
                while not entered_worker.is_set():
                    await asyncio.sleep(0.005)

            blocked_read.cancel()
            # Do not add an outer timeout here: its second cancellation would
            # make the old inline cursor-close path look bounded. The first
            # cancellation must reach the caller on its own.
            await asyncio.sleep(0.05)
            assert blocked_read.done()
            with pytest.raises(asyncio.CancelledError):
                await blocked_read
            assert backend.write_connection_unavailable

            following_write = asyncio.create_task(
                backend.execute("UPDATE t SET value = value + 10")
            )
            await asyncio.sleep(0.02)
            assert not following_write.done()

            release_worker.set()
            async with asyncio.timeout(1.0):
                await following_write
            assert await backend.fetch_all(
                "SELECT value FROM t ORDER BY value"
            ) == [(11,), (12,)]
        finally:
            watchdog.cancel()
            release_worker.set()
            if not blocked_read.done():
                blocked_read.cancel()
            if following_write is not None and not following_write.done():
                following_write.cancel()
            await asyncio.gather(
                blocked_read,
                *([following_write] if following_write is not None else []),
                return_exceptions=True,
            )

    @pytest.mark.asyncio
    async def test_cancelled_statement_uses_runtime_drain_budget(
        self, backend, monkeypatch
    ):
        """Cleanup interrupts first and does not inherit the shutdown timeout."""
        await backend.execute("CREATE TABLE recoverable (value INTEGER NOT NULL)")
        await backend.execute("INSERT INTO recoverable (value) VALUES (0)")
        entered_worker = threading.Event()
        release_worker = threading.Event()
        watchdog = threading.Timer(2.0, release_worker.set)

        def block_in_worker(value):
            entered_worker.set()
            release_worker.wait()
            return value

        conn = backend._ensure_connected()
        await conn.create_function("block_in_worker", 1, block_in_worker)
        blocked_read = asyncio.create_task(
            backend.fetch_one("SELECT block_in_worker(value) FROM recoverable")
        )
        following_read = None
        following_write = None
        release_handle = None
        try:
            watchdog.start()
            async with asyncio.timeout(0.5):
                while not entered_worker.is_set():
                    await asyncio.sleep(0.005)

            interrupt = AsyncMock(wraps=conn.interrupt)
            with (
                monkeypatch.context() as deadlines,
                patch.object(conn, "interrupt", new=interrupt),
            ):
                # Prove runtime cleanup remains available beyond the much
                # shorter worker-shutdown lifecycle window.
                deadlines.setattr(
                    sqlite_backend_module,
                    "AIOSQLITE_WORKER_SHUTDOWN_TIMEOUT_S",
                    0.02,
                )
                deadlines.setattr(
                    sqlite_backend_module,
                    "_CANCELLED_OPERATION_DRAIN_TIMEOUT_S",
                    0.3,
                )
                blocked_read.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await blocked_read

                following_read = asyncio.create_task(
                    backend.fetch_one("SELECT value FROM recoverable")
                )
                following_write = asyncio.create_task(
                    backend.execute(
                        "UPDATE recoverable SET value = value + 1"
                    )
                )
                release_handle = asyncio.get_running_loop().call_later(
                    0.08, release_worker.set
                )

                async with asyncio.timeout(0.5):
                    read_result, _ = await asyncio.gather(
                        following_read, following_write
                    )

                assert read_result in {(0,), (1,)}
                assert await backend.fetch_one(
                    "SELECT value FROM recoverable"
                ) == (1,)
                assert backend.write_connection_unavailable is False
                assert (
                    backend.write_connection_cleanup_deadline_exceeded
                    is False
                )
                interrupt.assert_awaited_once_with()
        finally:
            watchdog.cancel()
            if release_handle is not None:
                release_handle.cancel()
            release_worker.set()
            for task in (blocked_read, following_read, following_write):
                if task is not None and not task.done():
                    task.cancel()
            await asyncio.gather(
                *(
                    task
                    for task in (blocked_read, following_read, following_write)
                    if task is not None
                ),
                return_exceptions=True,
            )

    @pytest.mark.asyncio
    async def test_cancelled_snapshot_read_interrupts_before_owned_close(
        self, backend, monkeypatch
    ):
        """A long snapshot query cannot trap cancellation in connection close."""
        if backend.db_path == ":memory:":
            pytest.skip("in-memory SQLite has no independent snapshot connection")

        transaction_entered = asyncio.Event()
        release_transaction = asyncio.Event()

        async def hold_transaction():
            async with backend.transaction():
                transaction_entered.set()
                await release_transaction.wait()

        original_open_snapshot = backend._open_snapshot_read_connection
        entered_snapshot_worker = threading.Event()

        async def open_observed_snapshot():
            conn = await original_open_snapshot()
            await conn.set_progress_handler(entered_snapshot_worker.set, 100)
            return conn

        monkeypatch.setattr(
            backend, "_open_snapshot_read_connection", open_observed_snapshot
        )
        transaction = asyncio.create_task(hold_transaction())
        snapshot_read = None
        try:
            await transaction_entered.wait()
            snapshot_read = asyncio.create_task(
                backend.fetch_all(
                    """
                    WITH RECURSIVE count(value) AS (
                        SELECT 1
                        UNION ALL
                        SELECT value + 1 FROM count WHERE value < 100000000
                    )
                    SELECT sum(value) FROM count
                    """
                )
            )
            async with asyncio.timeout(0.5):
                while not entered_snapshot_worker.is_set():
                    await asyncio.sleep(0.005)

            snapshot_read.cancel()
            # A second cancellation would hide an inline-close regression.
            await asyncio.sleep(0.05)
            assert snapshot_read.done()
            with pytest.raises(asyncio.CancelledError):
                await snapshot_read
        finally:
            release_transaction.set()
            if snapshot_read is not None and not snapshot_read.done():
                snapshot_read.cancel()
            await asyncio.gather(
                transaction,
                *([snapshot_read] if snapshot_read is not None else []),
                return_exceptions=True,
            )

    @pytest.mark.asyncio
    async def test_worker_blocked_write_fences_following_transaction(self, backend):
        """A same-continuation transaction cannot overtake retained rollback."""
        await backend.execute("CREATE TABLE t (value INTEGER NOT NULL)")
        await backend.execute("INSERT INTO t (value) VALUES (0)")
        entered_worker = threading.Event()
        release_worker = threading.Event()
        watchdog = threading.Timer(2.0, release_worker.set)

        def block_in_worker(value):
            entered_worker.set()
            release_worker.wait()
            return value + 1

        conn = backend._ensure_connected()
        await conn.create_function("block_in_worker", 1, block_in_worker)
        blocked = asyncio.create_task(
            backend.execute("UPDATE t SET value = block_in_worker(value)")
        )
        release_handle = None
        try:
            watchdog.start()
            async with asyncio.timeout(0.5):
                while not entered_worker.is_set():
                    await asyncio.sleep(0.005)

            blocked.cancel()
            with pytest.raises(asyncio.CancelledError):
                await blocked

            # Continue in this task immediately after cancellation.  Without
            # the transaction fence, BEGIN is queued before the retained
            # rollback and fails with "transaction within a transaction".
            release_handle = asyncio.get_running_loop().call_later(
                0.05, release_worker.set
            )
            async with asyncio.timeout(1.0):
                async with backend.transaction():
                    await backend.execute("UPDATE t SET value = value + 10")

            assert await backend.fetch_all("SELECT value FROM t") == [(10,)]
        finally:
            watchdog.cancel()
            if release_handle is not None:
                release_handle.cancel()
            release_worker.set()
            if not blocked.done():
                blocked.cancel()
            await asyncio.gather(blocked, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_close_cancels_bounded_pending_write_drain(self, tmp_path):
        """Close retires its retained cleanup task before connection teardown."""
        backend = SQLiteBackend(str(tmp_path / "pending-drain.db"))
        await backend.connect()
        never_release = asyncio.Event()
        drain = asyncio.create_task(never_release.wait())
        backend._cancelled_write_drain = drain

        try:
            with patch(
                "kestrel_sovereign.storage.db.sqlite."
                "AIOSQLITE_WORKER_SHUTDOWN_TIMEOUT_S",
                0.01,
            ):
                await backend.close()

            assert drain.cancelled()
            assert backend._cancelled_write_drain is None
            assert backend._cancelled_write_drain_error is None
            assert not backend.is_connected
        finally:
            if not drain.done():
                drain.cancel()
                with suppress(asyncio.CancelledError):
                    await drain
            if backend.is_connected:
                await backend.close()

    @pytest.mark.asyncio
    async def test_close_pending_real_write_drain_fits_declared_budget(
        self, tmp_path
    ):
        """The shutdown reservation covers drain and connection-close phases."""
        backend = SQLiteBackend(str(tmp_path / "pending-real-drain.db"))
        await backend.connect()
        entered_worker = threading.Event()
        release_worker = threading.Event()
        shutdown_window = (
            sqlite_backend_module.AIOSQLITE_WORKER_SHUTDOWN_TIMEOUT_S
        )
        watchdog = threading.Timer(3 * shutdown_window, release_worker.set)

        def block_in_worker(value):
            entered_worker.set()
            release_worker.wait()
            return value

        conn = backend._ensure_connected()
        worker = aiosqlite_worker(conn)
        await conn.create_function("block_in_worker", 1, block_in_worker)
        blocked = asyncio.create_task(
            backend.execute("SELECT block_in_worker(1)")
        )
        release_handle = None
        try:
            watchdog.start()
            async with asyncio.timeout(0.5):
                while not entered_worker.is_set():
                    await asyncio.sleep(0.005)

            blocked.cancel()
            with pytest.raises(asyncio.CancelledError):
                await blocked
            drain = backend._cancelled_write_drain
            assert drain is not None
            assert not drain.done()

            # Let the drain consume most of the shared worker window, then let
            # connection.close() retire the real worker inside the same
            # absolute deadline. No timeout constants are patched: this
            # verifies the public default budget.
            loop = asyncio.get_running_loop()
            release_handle = loop.call_later(
                0.75 * shutdown_window,
                release_worker.set,
            )
            started = loop.time()
            await backend.close()
            elapsed = loop.time() - started

            assert elapsed <= backend.minimum_close_timeout_s
            assert not worker.is_alive()
            assert not backend.is_connected
        finally:
            watchdog.cancel()
            if release_handle is not None:
                release_handle.cancel()
            release_worker.set()
            if not blocked.done():
                blocked.cancel()
            await asyncio.gather(blocked, return_exceptions=True)
            if backend.is_connected:
                await backend.close()
            worker.join(timeout=1.0)

    @pytest.mark.asyncio
    async def test_cancelled_close_retires_pending_drain_and_worker(self, tmp_path):
        """Shutdown cancellation is delivered after retained cleanup closes."""
        backend = SQLiteBackend(str(tmp_path / "cancelled-pending-drain.db"))
        await backend.connect()
        conn = backend._ensure_connected()
        worker = aiosqlite_worker(conn)
        never_release = asyncio.Event()
        drain_started = asyncio.Event()

        async def pending_drain():
            drain_started.set()
            await never_release.wait()

        drain = asyncio.create_task(pending_drain())
        await drain_started.wait()
        backend._cancelled_write_drain = drain
        close_task = asyncio.create_task(backend.close())

        try:
            await asyncio.sleep(0)
            assert not close_task.done()
            close_task.cancel()

            with pytest.raises(asyncio.CancelledError):
                await close_task

            assert drain.cancelled()
            assert not backend.is_connected
            assert not worker.is_alive()
            assert backend._cancelled_write_drain is None
            assert backend._cancelled_write_drain_error is None
        finally:
            if not drain.done():
                drain.cancel()
                with suppress(asyncio.CancelledError):
                    await drain
            if backend.is_connected:
                await backend.close()
            worker.join(timeout=1.0)

    @pytest.mark.asyncio
    async def test_close_preserves_cancellation_while_reaping_timed_out_drain(
        self, tmp_path, monkeypatch
    ):
        """Caller cancellation during drain reaping is delivered after close."""
        monkeypatch.setattr(
            sqlite_backend_module,
            "AIOSQLITE_WORKER_SHUTDOWN_TIMEOUT_S",
            0.02,
        )
        backend = SQLiteBackend(str(tmp_path / "cancel-during-drain-reap.db"))
        await backend.connect()
        worker = aiosqlite_worker(backend._ensure_connected())
        first_drain_cancel = asyncio.Event()
        never_release = asyncio.Event()

        async def stubborn_drain():
            try:
                await never_release.wait()
            except asyncio.CancelledError:
                first_drain_cancel.set()
                await never_release.wait()

        drain = asyncio.create_task(stubborn_drain())
        backend._cancelled_write_drain = drain
        close_task = asyncio.create_task(backend.close())
        try:
            async with asyncio.timeout(0.2):
                await first_drain_cancel.wait()
            await asyncio.sleep(0)
            close_task.cancel()

            with pytest.raises(asyncio.CancelledError):
                await close_task

            assert not backend.is_connected
            assert drain.done()
            assert not worker.is_alive()
        finally:
            never_release.set()
            if not drain.done():
                drain.cancel()
            await asyncio.gather(drain, close_task, return_exceptions=True)
            if backend.is_connected:
                await backend.close()
            worker.join(timeout=1.0)

    @pytest.mark.asyncio
    async def test_cleanup_error_does_not_survive_close_or_connect(self, tmp_path):
        """A failed drain fences writes but leaves committed reads available."""
        backend = SQLiteBackend(str(tmp_path / "drain-recovery.db"))
        await backend.connect()
        await backend.execute("CREATE TABLE durable (value INTEGER NOT NULL)")
        await backend.execute("INSERT INTO durable (value) VALUES (7)")
        backend._cancelled_write_drain_error = RuntimeError("rollback failed")

        # A fresh query-only connection can still expose the committed state to
        # an explicitly diagnostic caller without trusting the shared
        # connection's possibly-abandoned transaction.
        assert await backend.fetch_one_diagnostic(
            "SELECT value FROM durable"
        ) == (7,)
        with pytest.raises(QueryError, match="cancellation cleanup failed"):
            await backend.fetch_one("SELECT value FROM durable")
        with pytest.raises(ConnectionError, match="cancellation cleanup failed"):
            await backend.execute("UPDATE durable SET value = 8")

        await backend.close()
        assert backend._cancelled_write_drain_error is None

        # Exercise connect's recovery while a failed shared connection is
        # still open; an idempotent early return would leave the latch set.
        await backend.connect()
        stale_conn = backend._ensure_connected()
        backend._cancelled_write_drain_error = RuntimeError("stale failure")
        await backend.connect()
        try:
            assert backend._ensure_connected() is not stale_conn
            assert backend._cancelled_write_drain_error is None
            assert await backend.fetch_one("SELECT 1") == (1,)
        finally:
            await backend.close()

    @pytest.mark.asyncio
    async def test_pending_write_drain_does_not_block_file_snapshot_reads(
        self, tmp_path
    ):
        """A wedged shared worker cannot wedge committed diagnostic reads."""
        backend = SQLiteBackend(str(tmp_path / "pending-drain-read.db"))
        await backend.connect()
        await backend.execute("CREATE TABLE durable (value INTEGER NOT NULL)")
        await backend.execute("INSERT INTO durable (value) VALUES (7)")
        entered_worker = threading.Event()
        release_worker = threading.Event()

        def block_in_worker(value):
            entered_worker.set()
            release_worker.wait()
            return value

        conn = backend._ensure_connected()
        await conn.create_function("block_in_worker", 1, block_in_worker)
        blocked = asyncio.create_task(
            backend.execute("SELECT block_in_worker(value) FROM durable")
        )
        try:
            async with asyncio.timeout(0.5):
                while not entered_worker.is_set():
                    await asyncio.sleep(0.005)
            blocked.cancel()
            with pytest.raises(asyncio.CancelledError):
                await blocked
            drain = backend._cancelled_write_drain
            assert drain is not None and not drain.done()

            async with asyncio.timeout(0.2):
                assert await backend.fetch_one_diagnostic(
                    "SELECT value FROM durable"
                ) == (7,)
        finally:
            release_worker.set()
            await asyncio.gather(blocked, return_exceptions=True)
            await backend.close()

    @pytest.mark.asyncio
    async def test_application_read_waits_for_cancelled_commit_drain(
        self, tmp_path
    ):
        """A stale diagnostic snapshot cannot drive a later application write."""
        backend = SQLiteBackend(str(tmp_path / "cancelled-commit-read.db"))
        await backend.connect()
        await backend.execute(
            "CREATE TABLE episodes (source TEXT NOT NULL)"
        )
        conn = backend._ensure_connected()
        commit_entered = threading.Event()
        release_commit = threading.Event()
        real_worker_commit = conn._conn.commit

        async def blocked_commit():
            def commit_in_worker():
                commit_entered.set()
                release_commit.wait()
                real_worker_commit()

            await conn._execute(commit_in_worker)

        timed_out_write = None
        application_read = None
        with patch.object(conn, "commit", new=blocked_commit):
            try:
                timed_out_write = asyncio.create_task(
                    backend.execute(
                        "INSERT INTO episodes (source) VALUES ('messages-1-2')"
                    )
                )
                async with asyncio.timeout(0.5):
                    while not commit_entered.is_set():
                        await asyncio.sleep(0.005)

                timed_out_write.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await timed_out_write
                drain = backend._cancelled_write_drain
                assert drain is not None and not drain.done()

                # Diagnostics remain available and, correctly, expose only the
                # pre-commit snapshot while the worker is blocked.
                assert await backend.fetch_all_diagnostic(
                    "SELECT source FROM episodes"
                ) == []

                # Application reads must not make the same stale decision. The
                # queued commit can still succeed before retained rollback.
                application_read = asyncio.create_task(
                    backend.fetch_all("SELECT source FROM episodes")
                )
                await asyncio.sleep(0.02)
                assert not application_read.done()

                release_commit.set()
                async with asyncio.timeout(1.0):
                    rows = await application_read
                assert rows == [("messages-1-2",)]
                assert backend.write_connection_unavailable is False

                # A consolidation-style read/decide/write continuation now
                # observes coverage and therefore cannot create a duplicate.
                if not rows:
                    await backend.execute(
                        "INSERT INTO episodes (source) VALUES ('messages-1-2')"
                    )
                assert await backend.fetch_val(
                    "SELECT COUNT(*) FROM episodes"
                ) == 1
            finally:
                release_commit.set()
                await asyncio.gather(
                    *(
                        task
                        for task in (timed_out_write, application_read)
                        if task is not None
                    ),
                    return_exceptions=True,
                )
                await backend.close()

    @pytest.mark.asyncio
    async def test_close_interrupts_real_blocked_worker_within_budget(
        self, tmp_path, monkeypatch
    ):
        """Close interrupts a retained real SQLite VM before awaiting close."""
        monkeypatch.setattr(
            sqlite_backend_module,
            "AIOSQLITE_WORKER_SHUTDOWN_TIMEOUT_S",
            0.2,
        )
        backend = SQLiteBackend(str(tmp_path / "interrupt-close.db"))
        await backend.connect()
        conn = backend._ensure_connected()
        worker = aiosqlite_worker(conn)
        entered_worker = threading.Event()
        await conn.set_progress_handler(entered_worker.set, 100)
        blocked = asyncio.create_task(
            backend.fetch_all(
                """
                WITH RECURSIVE count(value) AS (
                    SELECT 1
                    UNION ALL
                    SELECT value + 1 FROM count WHERE value < 100000000
                )
                SELECT sum(value) FROM count
                """
            )
        )
        watchdog = threading.Timer(2.0, conn._conn.interrupt)
        try:
            watchdog.start()
            async with asyncio.timeout(0.5):
                while not entered_worker.is_set():
                    await asyncio.sleep(0.005)

            blocked.cancel()
            with pytest.raises(asyncio.CancelledError):
                await blocked
            drain = backend._cancelled_write_drain
            assert drain is not None and not drain.done()

            started = asyncio.get_running_loop().time()
            await backend.close()
            elapsed = asyncio.get_running_loop().time() - started

            assert elapsed <= backend.minimum_close_timeout_s
            assert not backend.is_connected
            assert not worker.is_alive()
            assert drain.done()
        finally:
            watchdog.cancel()
            if not blocked.done():
                blocked.cancel()
            await asyncio.gather(blocked, return_exceptions=True)
            if backend.is_connected:
                await backend.close()
            worker.join(timeout=1.0)

    @pytest.mark.asyncio
    async def test_close_still_retires_worker_when_interrupt_raises(
        self, tmp_path, monkeypatch
    ):
        """A failed interrupt cannot bypass owned connection retirement."""
        monkeypatch.setattr(
            sqlite_backend_module,
            "AIOSQLITE_WORKER_SHUTDOWN_TIMEOUT_S",
            0.2,
        )
        backend = SQLiteBackend(str(tmp_path / "failed-interrupt-close.db"))
        await backend.connect()
        conn = backend._ensure_connected()
        worker = aiosqlite_worker(conn)
        entered_worker = threading.Event()
        release_worker = threading.Event()
        watchdog = threading.Timer(2.0, release_worker.set)

        def block_in_worker() -> int:
            entered_worker.set()
            release_worker.wait()
            return 1

        await conn.create_function("block_in_worker", 0, block_in_worker)
        blocked_read = asyncio.create_task(
            backend.fetch_one("SELECT block_in_worker()")
        )
        release_handle = None
        try:
            watchdog.start()
            async with asyncio.timeout(0.5):
                while not entered_worker.is_set():
                    await asyncio.sleep(0.005)

            blocked_read.cancel()
            with pytest.raises(asyncio.CancelledError):
                await blocked_read
            drain = backend._cancelled_write_drain
            assert drain is not None and not drain.done()

            release_handle = asyncio.get_running_loop().call_later(
                0.02, release_worker.set
            )
            interrupt = AsyncMock(
                side_effect=ValueError("no active connection")
            )
            with patch.object(conn, "interrupt", new=interrupt):
                await backend.close()

            interrupt.assert_awaited_once_with()
            assert drain.done()
            assert not backend.is_connected
            assert not backend.connection_retirement_pending
            assert not worker.is_alive()
        finally:
            watchdog.cancel()
            if release_handle is not None:
                release_handle.cancel()
            release_worker.set()
            await asyncio.gather(blocked_read, return_exceptions=True)
            if backend.is_connected:
                await backend.close()
            worker.join(timeout=1.0)

    @pytest.mark.asyncio
    async def test_close_does_not_interrupt_overlapping_shared_read(
        self, tmp_path, monkeypatch
    ):
        """Read serialization must not masquerade as active write work."""
        backend = SQLiteBackend(str(tmp_path / "read-overlap-close.db"))
        await backend.connect()
        conn = backend._ensure_connected()
        worker = aiosqlite_worker(conn)
        read_entered = asyncio.Event()
        release_read = asyncio.Event()
        close_entered = asyncio.Event()
        release_close = asyncio.Event()
        original_fetch = backend._fetch_on_connection
        original_close = sqlite_backend_module._close_aiosqlite_connection

        async def held_fetch(*args, **kwargs):
            read_entered.set()
            await release_read.wait()
            return await original_fetch(*args, **kwargs)

        async def held_close(connection, *, retained_closes, deadline=None):
            close_entered.set()
            await release_close.wait()
            await original_close(
                connection,
                retained_closes=retained_closes,
                deadline=deadline,
            )

        monkeypatch.setattr(backend, "_fetch_on_connection", held_fetch)
        read_task = asyncio.create_task(backend.fetch_one("SELECT 1"))
        close_task = None
        try:
            await read_entered.wait()
            assert backend._write_lock.locked()
            assert backend._active_write_operations == 0

            interrupt = AsyncMock(wraps=conn.interrupt)
            with patch.object(conn, "interrupt", new=interrupt), patch.object(
                sqlite_backend_module,
                "_close_aiosqlite_connection",
                new=held_close,
            ):
                close_task = asyncio.create_task(backend.close())
                await close_entered.wait()
                interrupt.assert_not_awaited()

                release_read.set()
                assert await read_task == (1,)
                release_close.set()
                await close_task

            assert not backend.is_connected
            assert not worker.is_alive()
        finally:
            release_read.set()
            release_close.set()
            await asyncio.gather(read_task, return_exceptions=True)
            if close_task is not None:
                await asyncio.gather(close_task, return_exceptions=True)
            if backend.is_connected:
                await backend.close()
            worker.join(timeout=1.0)

    @pytest.mark.asyncio
    async def test_cancelled_drain_becomes_connection_error_not_caller_cancel(
        self, tmp_path
    ):
        """Lifecycle cancellation of cleanup must not cancel unrelated work."""
        backend = SQLiteBackend(str(tmp_path / "cancelled-drain-waiter.db"))
        await backend.connect()
        rollback_started = asyncio.Event()
        conn = backend._ensure_connected()

        async def pending_rollback():
            rollback_started.set()
            await asyncio.Event().wait()

        with patch.object(conn, "rollback", new=pending_rollback):
            backend._handoff_cancelled_write(conn)
            drain = backend._cancelled_write_drain
            assert drain is not None
            await rollback_started.wait()
            waiter = asyncio.create_task(
                backend.execute("CREATE TABLE later (id INTEGER)")
            )
            try:
                await asyncio.sleep(0)
                drain.cancel()
                with pytest.raises(
                    ConnectionError, match="cancellation cleanup failed"
                ):
                    await waiter
                assert not waiter.cancelled()
            finally:
                if not drain.done():
                    drain.cancel()
                await asyncio.gather(drain, waiter, return_exceptions=True)
                await backend.close()

    @pytest.mark.asyncio
    async def test_writer_wait_for_cancelled_drain_has_a_deadline(
        self, tmp_path, monkeypatch
    ):
        """A wedged rollback cannot block every later writer indefinitely."""
        monkeypatch.setattr(
            sqlite_backend_module,
            "_CANCELLED_OPERATION_DRAIN_TIMEOUT_S",
            0.02,
        )
        backend = SQLiteBackend(str(tmp_path / "bounded-drain-wait.db"))
        await backend.connect()
        conn = backend._ensure_connected()
        rollback_started = asyncio.Event()
        release_rollback = asyncio.Event()

        async def pending_rollback():
            rollback_started.set()
            await release_rollback.wait()

        with patch.object(conn, "rollback", new=pending_rollback):
            backend._handoff_cancelled_write(conn)
            drain = backend._cancelled_write_drain
            assert drain is not None
            await rollback_started.wait()
            try:
                async with asyncio.timeout(0.2):
                    with pytest.raises(
                        ConnectionError, match="cleanup is still pending"
                    ):
                        await backend.execute("CREATE TABLE later (id INTEGER)")
                assert backend.write_connection_requires_reconnect is False
                assert not drain.done()
            finally:
                release_rollback.set()
                await asyncio.gather(drain, return_exceptions=True)
                assert backend.write_connection_unavailable is False
                await backend.execute("CREATE TABLE later (id INTEGER)")
                await backend.close()

    @pytest.mark.asyncio
    async def test_drain_retries_rollback_its_own_interrupt_aborted(
        self, backend
    ):
        """An interrupted drain rollback converges instead of latching (#3496)."""
        await backend.execute("CREATE TABLE durable (value INTEGER NOT NULL)")
        conn = backend._ensure_connected()
        real_rollback = conn.rollback
        attempts = 0

        async def rollback_interrupted_once():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise _sqlite_interrupted()
            await real_rollback()

        with patch.object(conn, "rollback", new=rollback_interrupted_once):
            backend._handoff_cancelled_write(conn)
            drain = backend._cancelled_write_drain
            assert drain is not None
            async with asyncio.timeout(_HANG_GUARD_SECONDS):
                await drain

        assert attempts == 2
        assert backend._cancelled_write_drain_error is None
        assert backend.write_connection_unavailable is False
        await backend.execute("INSERT INTO durable (value) VALUES (1)")
        assert await backend.fetch_val("SELECT value FROM durable") == 1

    @pytest.mark.asyncio
    async def test_drain_rollback_failing_otherwise_still_latches(
        self, backend
    ):
        """Only SQLITE_INTERRUPT is retried; any other failure latches."""
        conn = backend._ensure_connected()
        failure = _sqlite_operational_error(
            sqlite3.SQLITE_IOERR, "SQLITE_IOERR", "disk I/O error"
        )
        attempts = 0

        async def failing_rollback():
            nonlocal attempts
            attempts += 1
            raise failure

        with patch.object(conn, "rollback", new=failing_rollback):
            backend._handoff_cancelled_write(conn)
            drain = backend._cancelled_write_drain
            assert drain is not None
            async with asyncio.timeout(_HANG_GUARD_SECONDS):
                await drain

        assert attempts == 1
        assert backend._cancelled_write_drain_error is failure
        assert backend.write_connection_requires_reconnect is True
        with pytest.raises(ConnectionError, match="cancellation cleanup failed"):
            await backend.execute("CREATE TABLE later (id INTEGER)")

    @pytest.mark.asyncio
    async def test_drain_rollback_still_interrupted_at_deadline_latches(
        self, backend, monkeypatch
    ):
        """The retry stays inside the drain deadline, then latches."""
        drain_timeout = 0.2
        monkeypatch.setattr(
            sqlite_backend_module,
            "_CANCELLED_OPERATION_DRAIN_TIMEOUT_S",
            drain_timeout,
        )
        conn = backend._ensure_connected()

        async def always_interrupted():
            raise _sqlite_interrupted()

        with patch.object(conn, "rollback", new=always_interrupted):
            backend._handoff_cancelled_write(conn)
            drain = backend._cancelled_write_drain
            started_at = backend._cancelled_write_drain_started_at
            assert drain is not None and started_at is not None
            async with asyncio.timeout(_HANG_GUARD_SECONDS):
                await drain
            latched_after = time.monotonic() - started_at

        assert latched_after >= drain_timeout
        assert sqlite_backend_module._is_sqlite_interrupt(
            backend._cancelled_write_drain_error
        )
        assert backend.write_connection_requires_reconnect is True
        with pytest.raises(ConnectionError, match="cancellation cleanup failed"):
            await backend.execute("CREATE TABLE later (id INTEGER)")

    @pytest.mark.asyncio
    async def test_drain_does_not_retry_rollback_interrupted_during_close(
        self, tmp_path
    ):
        """Close's interrupt is close's: the drain stops instead of retrying."""
        backend = SQLiteBackend(str(tmp_path / "close-interrupts-drain.db"))
        await backend.connect()
        conn = backend._ensure_connected()
        rollback_started = asyncio.Event()
        release_rollback = asyncio.Event()
        attempts = 0

        async def rollback_interrupted_by_close():
            nonlocal attempts
            attempts += 1
            rollback_started.set()
            await release_rollback.wait()
            raise _sqlite_interrupted()

        close_task = None
        drain = None
        try:
            with patch.object(
                conn, "rollback", new=rollback_interrupted_by_close
            ):
                backend._handoff_cancelled_write(conn)
                drain = backend._cancelled_write_drain
                assert drain is not None
                async with asyncio.timeout(_HANG_GUARD_SECONDS):
                    await rollback_started.wait()
                close_task = asyncio.create_task(backend.close())
                async with asyncio.timeout(_HANG_GUARD_SECONDS):
                    while not backend._closing:
                        await asyncio.sleep(0)
                release_rollback.set()
                async with asyncio.timeout(_HANG_GUARD_SECONDS):
                    await close_task

            assert attempts == 1
            assert drain.done() and not drain.cancelled()
            assert not backend.is_connected
        finally:
            release_rollback.set()
            for task in (drain, close_task):
                if task is not None and not task.done():
                    task.cancel()
            await asyncio.gather(
                *(task for task in (drain, close_task) if task is not None),
                return_exceptions=True,
            )
            if backend.is_connected:
                await backend.close()

    @pytest.mark.asyncio
    async def test_drain_releases_read_cursor_held_by_aiosqlite_worker(
        self, backend, monkeypatch
    ):
        """A transaction read cancelled mid-execute leaves a genuinely
        interrupted ROLLBACK, and the drain still converges (#3496).

        The worker finished the SELECT's first step, so its statement is
        active and only aiosqlite's worker still references its cursor.
        """
        monkeypatch.setattr(
            sqlite_backend_module,
            "_CANCELLED_OPERATION_DRAIN_TIMEOUT_S",
            5.0,
        )
        await backend.execute("CREATE TABLE t (value INTEGER NOT NULL)")
        await backend.execute_many(
            "INSERT INTO t (value) VALUES (?)", [(1,), (2,), (3,)]
        )
        conn = backend._ensure_connected()
        real_execute = conn.execute
        real_rollback = conn.rollback
        executed = asyncio.Event()
        rollbacks: list[str] = []

        async def execute_then_hang(sql, parameters=None):
            if not sql.startswith("SELECT value FROM t"):
                return await real_execute(sql, parameters)
            await real_execute(sql, parameters)
            executed.set()
            await asyncio.Event().wait()

        async def observed_rollback():
            try:
                await real_rollback()
            except sqlite3.OperationalError as exc:
                rollbacks.append(exc.sqlite_errorname)
                raise
            rollbacks.append("ok")

        async def cancelled_transaction():
            async with backend.transaction():
                await backend.execute("INSERT INTO t (value) VALUES (4)")
                await backend.fetch_all("SELECT value FROM t ORDER BY value")

        with (
            patch.object(conn, "execute", new=execute_then_hang),
            patch.object(conn, "rollback", new=observed_rollback),
        ):
            transaction = asyncio.create_task(cancelled_transaction())
            async with asyncio.timeout(_HANG_GUARD_SECONDS):
                await executed.wait()
            transaction.cancel()
            with pytest.raises(asyncio.CancelledError):
                await transaction
            drain = backend._cancelled_write_drain
            assert drain is not None
            async with asyncio.timeout(_HANG_GUARD_SECONDS):
                await drain

        assert rollbacks == ["SQLITE_INTERRUPT", "ok"]
        assert backend._cancelled_write_drain_error is None
        await backend.execute("INSERT INTO t (value) VALUES (5)")
        assert await backend.fetch_all("SELECT value FROM t ORDER BY value") == [
            (1,),
            (2,),
            (3,),
            (5,),
        ]

    @pytest.mark.asyncio
    async def test_drain_closes_cancelled_read_cursor_before_retrying(
        self, backend, monkeypatch
    ):
        """A cancelled read's cursor kept alive by its task's exception is
        closed by the drain, so the interrupted ROLLBACK converges (#3496)."""
        monkeypatch.setattr(
            sqlite_backend_module,
            "_CANCELLED_OPERATION_DRAIN_TIMEOUT_S",
            5.0,
        )
        await backend.execute("CREATE TABLE t (value INTEGER NOT NULL)")
        await backend.execute_many(
            "INSERT INTO t (value) VALUES (?)", [(1,), (2,), (3,)]
        )
        conn = backend._ensure_connected()
        real_rollback = conn.rollback
        fetch_entered = asyncio.Event()
        rollbacks: list[str] = []

        async def hanging_fetchall(cursor):
            fetch_entered.set()
            await asyncio.Event().wait()

        async def observed_rollback():
            try:
                await real_rollback()
            except sqlite3.OperationalError as exc:
                rollbacks.append(exc.sqlite_errorname)
                raise
            rollbacks.append("ok")

        async def cancelled_transaction():
            async with backend.transaction():
                await backend.execute("INSERT INTO t (value) VALUES (4)")
                await backend.fetch_all("SELECT value FROM t ORDER BY value")

        transaction = asyncio.create_task(cancelled_transaction())
        try:
            with (
                patch.object(aiosqlite.Cursor, "fetchall", new=hanging_fetchall),
                patch.object(conn, "rollback", new=observed_rollback),
            ):
                async with asyncio.timeout(_HANG_GUARD_SECONDS):
                    await fetch_entered.wait()
                transaction.cancel()
                drain = None
                async with asyncio.timeout(_HANG_GUARD_SECONDS):
                    while drain is None:
                        await asyncio.sleep(0)
                        drain = backend._cancelled_write_drain
                # The cancelled transaction is deliberately not awaited yet:
                # its exception keeps the abandoned cursor alive.
                assert transaction.done()
                async with asyncio.timeout(_HANG_GUARD_SECONDS):
                    await drain
            with pytest.raises(asyncio.CancelledError):
                await transaction
        finally:
            if not transaction.done():
                transaction.cancel()
            await asyncio.gather(transaction, return_exceptions=True)

        assert rollbacks == ["SQLITE_INTERRUPT", "ok"]
        assert backend._cancelled_write_drain_error is None
        await backend.execute("INSERT INTO t (value) VALUES (5)")
        assert await backend.fetch_all("SELECT value FROM t ORDER BY value") == [
            (1,),
            (2,),
            (3,),
            (5,),
        ]

    @pytest.mark.asyncio
    async def test_cancelled_autocommit_read_does_not_interrupt_next_write(
        self, backend
    ):
        """A ``fetch_one`` cancelled between its ``execute`` and ``fetchone``,
        with no transaction open, does not fail the next write (#3508).

        Outside a transaction the drain's rollback runs no statement, so it
        cannot clear the drain's interrupt while the abandoned statement is
        active. Only closing the abandoned cursor lets the write's BEGIN run.
        """
        await backend.execute("CREATE TABLE t (value INTEGER NOT NULL)")
        await backend.execute_many(
            "INSERT INTO t (value) VALUES (?)", [(1,), (2,), (3,)]
        )
        fetch_entered = asyncio.Event()
        abandoned: list[weakref.ref] = []

        async def hanging_fetchone(cursor):
            abandoned.append(weakref.ref(cursor))
            fetch_entered.set()
            await asyncio.Event().wait()

        read = None
        try:
            with patch.object(aiosqlite.Cursor, "fetchone", new=hanging_fetchone):
                read = asyncio.create_task(
                    backend.fetch_one("SELECT value FROM t ORDER BY value")
                )
                async with asyncio.timeout(_HANG_GUARD_SECONDS):
                    await fetch_entered.wait()
                read.cancel()
                async with asyncio.timeout(_HANG_GUARD_SECONDS):
                    while not read.done():
                        await asyncio.sleep(0)
            assert not backend._in_transaction
            # The cancelled read is deliberately not awaited yet: its
            # exception keeps the abandoned cursor, and its statement, alive.
            async with asyncio.timeout(_HANG_GUARD_SECONDS):
                await backend.execute("INSERT INTO t (value) VALUES (4)")
            assert abandoned[0]() is not None
            with pytest.raises(asyncio.CancelledError):
                await read
        finally:
            if read is not None:
                if not read.done():
                    read.cancel()
                await asyncio.gather(read, return_exceptions=True)

        assert backend._cancelled_write_drain_error is None
        assert await backend.fetch_all("SELECT value FROM t ORDER BY value") == [
            (1,),
            (2,),
            (3,),
            (4,),
        ]

    @pytest.mark.asyncio
    async def test_drain_releases_abandoned_read_outside_transaction(
        self, backend
    ):
        """The issue's aiosqlite-level reproduction, through the drain (#3508)."""
        await backend.execute("CREATE TABLE t (v INTEGER)")
        await backend.execute_many(
            "INSERT INTO t VALUES (?)", [(i,) for i in range(5)]
        )
        conn = backend._ensure_connected()
        # An abandoned read, its cursor still referenced, with rows pending.
        held = await conn.execute("SELECT v FROM t")
        await held.fetchone()
        backend._abandoned_read_cursors.add(held)

        # Interrupt, then a rollback that is a no-op in autocommit.
        backend._handoff_cancelled_write(conn)
        drain = backend._cancelled_write_drain
        assert drain is not None
        async with asyncio.timeout(_HANG_GUARD_SECONDS):
            await drain

        assert backend._cancelled_write_drain_error is None
        assert not backend._abandoned_read_cursors
        # Failed OperationalError('interrupted') before the drain closed it.
        await conn.execute("INSERT INTO t VALUES (100)")
        await conn.commit()
        assert await backend.fetch_val("SELECT COUNT(*) FROM t") == 6
        assert held is not None

    @pytest.mark.asyncio
    async def test_drain_treats_closed_and_collected_cursors_as_released(
        self, backend, caplog
    ):
        """Closing a cursor already closed, or forgetting one already
        collected, is a no-op for the drain (#3508)."""
        await backend.execute("CREATE TABLE t (v INTEGER)")
        await backend.execute_many(
            "INSERT INTO t VALUES (?)", [(i,) for i in range(5)]
        )
        conn = backend._ensure_connected()
        closed = await conn.execute("SELECT v FROM t")
        await closed.fetchone()
        await closed.close()
        collected = await conn.execute("SELECT v FROM t")
        backend._abandoned_read_cursors.add(closed)
        backend._abandoned_read_cursors.add(collected)
        del collected
        assert set(backend._abandoned_read_cursors) == {closed}

        with caplog.at_level(logging.WARNING, logger=sqlite_backend_module.__name__):
            backend._handoff_cancelled_write(conn)
            drain = backend._cancelled_write_drain
            assert drain is not None
            async with asyncio.timeout(_HANG_GUARD_SECONDS):
                await drain

        assert not [
            record
            for record in caplog.records
            if record.name == sqlite_backend_module.__name__
        ]
        assert backend._cancelled_write_drain_error is None
        assert not backend._abandoned_read_cursors
        await backend.execute("INSERT INTO t VALUES (100)")
        assert await backend.fetch_val("SELECT COUNT(*) FROM t") == 6

    @pytest.mark.asyncio
    async def test_drain_retries_failed_cursor_close_before_releasing_writes(
        self, backend, caplog
    ):
        """A failed close is logged, stays recorded, and keeps writes fenced
        until the drain's retry closes it (#3508)."""
        await backend.execute("CREATE TABLE t (v INTEGER)")
        await backend.execute_many(
            "INSERT INTO t VALUES (?)", [(i,) for i in range(5)]
        )
        conn = backend._ensure_connected()
        held = await conn.execute("SELECT v FROM t")
        await held.fetchone()
        backend._abandoned_read_cursors.add(held)
        real_close = held.close
        failure = sqlite3.ProgrammingError("simulated cursor close failure")
        first_close_entered = asyncio.Event()
        fail_first_close = asyncio.Event()
        recorded_at_retry: list[bool] = []
        closes = 0

        async def close_failing_once():
            nonlocal closes
            closes += 1
            if closes == 1:
                first_close_entered.set()
                await fail_first_close.wait()
                raise failure
            recorded_at_retry.append(held in backend._abandoned_read_cursors)
            await real_close()

        write = None
        try:
            with (
                patch.object(held, "close", new=close_failing_once),
                caplog.at_level(
                    logging.WARNING, logger=sqlite_backend_module.__name__
                ),
            ):
                backend._handoff_cancelled_write(conn)
                drain = backend._cancelled_write_drain
                assert drain is not None
                async with asyncio.timeout(_HANG_GUARD_SECONDS):
                    await first_close_entered.wait()
                write = asyncio.create_task(
                    backend.execute("INSERT INTO t VALUES (100)")
                )
                fail_first_close.set()
                async with asyncio.timeout(_HANG_GUARD_SECONDS):
                    await drain
                async with asyncio.timeout(_HANG_GUARD_SECONDS):
                    await write
        finally:
            fail_first_close.set()
            if write is not None:
                if not write.done():
                    write.cancel()
                await asyncio.gather(write, return_exceptions=True)

        assert closes == 2
        assert recorded_at_retry == [True]
        assert [
            record.exc_info[1]
            for record in caplog.records
            if record.name == sqlite_backend_module.__name__
            and record.exc_info is not None
        ] == [failure]
        assert backend._cancelled_write_drain_error is None
        assert not backend._abandoned_read_cursors
        assert await backend.fetch_val("SELECT COUNT(*) FROM t") == 6

    @pytest.mark.asyncio
    async def test_drain_cursor_close_still_failing_at_deadline_latches(
        self, backend, monkeypatch
    ):
        """A close that keeps failing latches at the drain deadline instead
        of retiring the fence as success (#3508)."""
        drain_timeout = 0.2
        monkeypatch.setattr(
            sqlite_backend_module,
            "_CANCELLED_OPERATION_DRAIN_TIMEOUT_S",
            drain_timeout,
        )
        await backend.execute("CREATE TABLE t (v INTEGER)")
        await backend.execute_many(
            "INSERT INTO t VALUES (?)", [(i,) for i in range(5)]
        )
        conn = backend._ensure_connected()
        held = await conn.execute("SELECT v FROM t")
        await held.fetchone()
        backend._abandoned_read_cursors.add(held)
        failure = sqlite3.ProgrammingError("simulated cursor close failure")

        async def always_failing_close():
            raise failure

        try:
            with patch.object(held, "close", new=always_failing_close):
                backend._handoff_cancelled_write(conn)
                drain = backend._cancelled_write_drain
                started_at = backend._cancelled_write_drain_started_at
                assert drain is not None and started_at is not None
                async with asyncio.timeout(_HANG_GUARD_SECONDS):
                    await drain
                latched_after = time.monotonic() - started_at

            assert latched_after >= drain_timeout
            assert backend._cancelled_write_drain_error is failure
            assert backend.write_connection_requires_reconnect is True
            assert set(backend._abandoned_read_cursors) == {held}
            with pytest.raises(
                ConnectionError, match="cancellation cleanup failed"
            ):
                await backend.execute("INSERT INTO t VALUES (100)")
        finally:
            await held.close()

    @pytest.mark.asyncio
    async def test_drain_does_not_retry_cursor_close_failing_during_close(
        self, tmp_path
    ):
        """Once close owns the connection, a failed cursor close ends the
        drain instead of retrying (#3508)."""
        backend = SQLiteBackend(str(tmp_path / "close-during-cursor-close.db"))
        await backend.connect()
        conn = backend._ensure_connected()
        held = await conn.execute("SELECT 1 UNION ALL SELECT 2")
        await held.fetchone()
        backend._abandoned_read_cursors.add(held)
        failure = sqlite3.ProgrammingError("simulated cursor close failure")
        close_entered = asyncio.Event()
        fail_close = asyncio.Event()
        closes = 0

        async def close_failing_when_released():
            nonlocal closes
            closes += 1
            close_entered.set()
            await fail_close.wait()
            raise failure

        close_task = None
        drain = None
        try:
            with patch.object(held, "close", new=close_failing_when_released):
                backend._handoff_cancelled_write(conn)
                drain = backend._cancelled_write_drain
                assert drain is not None
                async with asyncio.timeout(_HANG_GUARD_SECONDS):
                    await close_entered.wait()
                close_task = asyncio.create_task(backend.close())
                async with asyncio.timeout(_HANG_GUARD_SECONDS):
                    while not backend._closing:
                        await asyncio.sleep(0)
                fail_close.set()
                async with asyncio.timeout(_HANG_GUARD_SECONDS):
                    await close_task

            assert closes == 1
            assert drain.done() and not drain.cancelled()
            assert not backend.is_connected
        finally:
            fail_close.set()
            for task in (drain, close_task):
                if task is not None and not task.done():
                    task.cancel()
            await asyncio.gather(
                *(task for task in (drain, close_task) if task is not None),
                return_exceptions=True,
            )
            if backend.is_connected:
                await backend.close()

    @pytest.mark.asyncio
    async def test_close_fences_late_cancellation_handoff(self, tmp_path):
        """Close cannot orphan a rollback task installed after drain sampling."""
        backend = SQLiteBackend(str(tmp_path / "late-close-handoff.db"))
        await backend.connect()
        conn = backend._ensure_connected()
        close_entered = asyncio.Event()
        release_close = asyncio.Event()
        real_close = sqlite_backend_module._close_aiosqlite_connection

        async def delayed_close(connection, *, retained_closes, deadline=None):
            close_entered.set()
            await release_close.wait()
            await real_close(
                connection,
                retained_closes=retained_closes,
                deadline=deadline,
            )

        with patch.object(
            sqlite_backend_module,
            "_close_aiosqlite_connection",
            new=delayed_close,
        ):
            close_task = asyncio.create_task(backend.close())
            await close_entered.wait()
            backend._handoff_cancelled_write(conn)
            assert backend._cancelled_write_drain is None
            release_close.set()
            await close_task

        assert backend._cancelled_write_drain is None
        assert backend._cancelled_write_drain_error is None
        assert not backend.is_connected

    @pytest.mark.asyncio
    async def test_table_not_exists(self, backend):
        exists = await backend.table_exists("nonexistent")
        assert exists is False
    
    @pytest.mark.asyncio
    async def test_memory_database(self):
        backend = SQLiteBackend(":memory:")
        await backend.connect()
        assert backend.is_connected
        await backend.execute("CREATE TABLE test (id INTEGER)")
        await backend.close()
        assert not backend.is_connected

    @pytest.mark.asyncio
    async def test_close_waits_for_delayed_aiosqlite_worker_exit(self, tmp_path):
        """Close must not return after aiosqlite merely acknowledges its stop.

        aiosqlite resolves ``close()`` before its worker thread's target has
        returned.  Delay that final return to reproduce the xdist teardown
        race, then prove SQLiteBackend waits for the actual thread exit.
        """
        release_worker = threading.Event()
        worker_exit_delayed = threading.Event()
        with delay_aiosqlite_worker_exit(release_worker, worker_exit_delayed) as workers:
            backend = SQLiteBackend(str(tmp_path / "delayed-worker.db"))
            await backend.connect()

        connection = backend._connection
        assert connection is not None
        worker = aiosqlite_worker(connection)
        close_task = asyncio.create_task(backend.close())

        try:
            await wait_for_lifecycle_checkpoint(
                wait_until_aiosqlite_worker_exit_is_delayed(worker_exit_delayed),
                close_task,
                description="the SQLite worker exit delay",
            )
            assert workers == [worker]
            assert not close_task.done()

            release_worker.set()
            await close_task
            assert not worker.is_alive()
        finally:
            release_worker.set()
            if not close_task.done():
                await close_task
            worker.join(timeout=1.0)

    @pytest.mark.asyncio
    async def test_cancelled_close_waits_for_delayed_aiosqlite_worker_exit(
        self, tmp_path,
    ):
        """Cancellation propagates only after the owned worker has stopped."""
        release_worker = threading.Event()
        worker_exit_delayed = threading.Event()
        with delay_aiosqlite_worker_exit(
            release_worker, worker_exit_delayed,
        ) as workers:
            backend = SQLiteBackend(str(tmp_path / "cancelled-worker.db"))
            await backend.connect()
            connection = backend._connection
            assert connection is not None
            worker = aiosqlite_worker(connection)
            close_task = asyncio.create_task(backend.close())

            try:
                await wait_for_lifecycle_checkpoint(
                    wait_until_aiosqlite_worker_exit_is_delayed(
                        worker_exit_delayed,
                    ),
                    close_task,
                    description="the cancelled SQLite worker exit delay",
                )
                assert workers == [worker]
                close_task.cancel()
                await asyncio.sleep(0)
                assert not close_task.done()

                release_worker.set()
                with pytest.raises(asyncio.CancelledError):
                    await close_task
                assert not worker.is_alive()
            finally:
                release_worker.set()
                if not close_task.done():
                    with suppress(asyncio.CancelledError):
                        await close_task
                worker.join(timeout=1.0)

        assert not worker.is_alive()

    @pytest.mark.asyncio
    async def test_close_fails_if_aiosqlite_worker_misses_shutdown_deadline(
        self, tmp_path,
    ):
        """A worker that cannot exit must fail close within its bounded wait."""
        release_worker = threading.Event()
        worker_exit_delayed = threading.Event()
        with delay_aiosqlite_worker_exit(
            release_worker, worker_exit_delayed,
        ) as workers, patch(
            "kestrel_sovereign.storage.db.sqlite."
            "AIOSQLITE_WORKER_SHUTDOWN_TIMEOUT_S",
            0.01,
        ):
            backend = SQLiteBackend(str(tmp_path / "stuck-worker.db"))
            await backend.connect()
            connection = backend._connection
            assert connection is not None
            worker = aiosqlite_worker(connection)
            retirement_tasks = []
            try:
                with pytest.raises(ConnectionError, match="worker did not terminate"):
                    await backend.close()
                assert not backend.is_connected
                assert worker.is_alive()
                assert workers == [worker]
                assert backend.connection_retirement_pending
                retirement_tasks = list(
                    retained.retirement_task
                    for retained in backend._retired_connection_closes.values()
                )
            finally:
                release_worker.set()
                await asyncio.gather(
                    *retirement_tasks, return_exceptions=True
                )
                worker.join(timeout=1.0)

        assert not worker.is_alive()
        assert not backend.connection_retirement_pending

    @pytest.mark.asyncio
    async def test_close_retains_python_udf_worker_and_fences_reconnect(
        self, tmp_path, monkeypatch
    ):
        """A close deadline cannot orphan a worker blocked in Python code."""
        monkeypatch.setattr(
            sqlite_backend_module,
            "AIOSQLITE_WORKER_SHUTDOWN_TIMEOUT_S",
            0.02,
        )
        backend = SQLiteBackend(str(tmp_path / "blocked-udf-close.db"))
        await backend.connect()
        connection = backend._ensure_connected()
        worker = aiosqlite_worker(connection)
        entered_worker = threading.Event()
        release_worker = threading.Event()
        watchdog = threading.Timer(2.0, release_worker.set)

        def block_in_worker() -> int:
            entered_worker.set()
            release_worker.wait()
            return 1

        await connection.create_function(
            "block_in_worker", 0, block_in_worker
        )
        blocked_read = asyncio.create_task(
            backend.fetch_one("SELECT block_in_worker()")
        )
        retirement_tasks = []
        try:
            watchdog.start()
            async with asyncio.timeout(0.5):
                while not entered_worker.is_set():
                    await asyncio.sleep(0.005)

            blocked_read.cancel()
            with pytest.raises(asyncio.CancelledError):
                await blocked_read

            started = asyncio.get_running_loop().time()
            with pytest.raises(
                ConnectionError, match="connection close did not complete"
            ):
                await backend.close()
            elapsed = asyncio.get_running_loop().time() - started

            # Keep a generous harness margin around the 50ms production
            # budget so a loaded CI event loop cannot turn scheduler jitter
            # into a lifecycle failure.
            assert elapsed < 0.2
            assert worker.is_alive()
            assert backend.connection_retirement_pending
            retained_close = next(
                iter(backend._retired_connection_closes.values())
            )
            retirement_task = retained_close.retirement_task
            assert retirement_task is not None
            retirement_tasks = [retirement_task]
            assert len(retirement_tasks) == 1
            assert retained_close.close_task is not None
            retirement_task.cancel()
            await asyncio.sleep(0)
            assert retirement_task.cancelled()
            assert backend.connection_retirement_pending

            with pytest.raises(
                ConnectionError, match="previous connection worker"
            ):
                await backend.connect()
            assert backend._connection is None

            release_worker.set()
            async with asyncio.timeout(1.0):
                while worker.is_alive():
                    await asyncio.sleep(0.005)
            assert not backend.connection_retirement_pending
            assert not worker.is_alive()

            await backend.connect()
            replacement = backend._ensure_connected()
            assert replacement is not connection
        finally:
            watchdog.cancel()
            release_worker.set()
            await asyncio.gather(blocked_read, return_exceptions=True)
            await asyncio.gather(
                *retirement_tasks, return_exceptions=True
            )
            if backend.is_connected:
                await backend.close()
            worker.join(timeout=1.0)

    @pytest.mark.asyncio
    async def test_primary_close_reports_retained_snapshot_worker(
        self, tmp_path, monkeypatch
    ):
        """Primary close cannot report success while an auxiliary worker lives."""
        monkeypatch.setattr(
            sqlite_backend_module,
            "AIOSQLITE_WORKER_SHUTDOWN_TIMEOUT_S",
            0.02,
        )
        backend = SQLiteBackend(str(tmp_path / "retained-snapshot.db"))
        await backend.connect()
        primary = backend._ensure_connected()
        primary_worker = aiosqlite_worker(primary)
        snapshot = await backend._open_snapshot_read_connection()
        snapshot_worker = aiosqlite_worker(snapshot)
        entered_worker = threading.Event()
        release_worker = threading.Event()
        watchdog = threading.Timer(2.0, release_worker.set)

        def block_in_worker() -> int:
            entered_worker.set()
            release_worker.wait()
            return 1

        await snapshot.create_function("block_in_worker", 0, block_in_worker)
        blocked_read = asyncio.create_task(
            snapshot.execute("SELECT block_in_worker()")
        )
        retirement_tasks = []
        try:
            watchdog.start()
            async with asyncio.timeout(0.5):
                while not entered_worker.is_set():
                    await asyncio.sleep(0.005)

            blocked_read.cancel()
            with pytest.raises(asyncio.CancelledError):
                await blocked_read

            with pytest.raises(
                ConnectionError, match="connection close did not complete"
            ):
                await sqlite_backend_module._close_aiosqlite_connection(
                    snapshot,
                    retained_closes=backend._retired_connection_closes,
                )
            assert snapshot_worker.is_alive()
            assert backend.connection_retirement_pending
            retirement_tasks = [
                retained.retirement_task
                for retained in backend._retired_connection_closes.values()
                if retained.retirement_task is not None
            ]

            with pytest.raises(
                ConnectionError, match="worker is still retiring"
            ):
                await backend.close()
            assert not backend.is_connected
            assert not primary_worker.is_alive()
            assert snapshot_worker.is_alive()
            assert backend.connection_retirement_pending

            with pytest.raises(
                ConnectionError, match="previous connection worker"
            ):
                await backend.connect()

            release_worker.set()
            await asyncio.gather(*retirement_tasks, return_exceptions=True)
            assert not backend.connection_retirement_pending
            assert not snapshot_worker.is_alive()

            await backend.connect()
        finally:
            watchdog.cancel()
            release_worker.set()
            await asyncio.gather(blocked_read, return_exceptions=True)
            await asyncio.gather(*retirement_tasks, return_exceptions=True)
            if backend.is_connected:
                await backend.close()
            primary_worker.join(timeout=1.0)
            snapshot_worker.join(timeout=1.0)

    @pytest.mark.asyncio
    async def test_repeated_close_cancellation_retains_live_worker(
        self, tmp_path, monkeypatch
    ):
        """A second caller cancellation cannot orphan a blocked worker."""
        monkeypatch.setattr(
            sqlite_backend_module,
            "AIOSQLITE_WORKER_SHUTDOWN_TIMEOUT_S",
            0.2,
        )
        connection = await aiosqlite.connect(
            str(tmp_path / "repeated-close-cancellation.db")
        )
        worker = aiosqlite_worker(connection)
        entered_worker = threading.Event()
        release_worker = threading.Event()
        watchdog = threading.Timer(2.0, release_worker.set)
        first_wait_started = asyncio.Event()
        retry_wait_started = asyncio.Event()
        retained_closes = {}
        real_wait = asyncio.wait
        wait_count = 0

        def block_in_worker() -> int:
            entered_worker.set()
            release_worker.wait()
            return 1

        async def tracked_wait(*args, **kwargs):
            nonlocal wait_count
            wait_count += 1
            if wait_count == 1:
                first_wait_started.set()
            elif wait_count == 2:
                retry_wait_started.set()
            return await real_wait(*args, **kwargs)

        await connection.create_function(
            "block_in_worker", 0, block_in_worker
        )
        blocked_read = asyncio.create_task(
            connection.execute("SELECT block_in_worker()")
        )
        retirement_tasks = []
        close_call = None
        try:
            watchdog.start()
            async with asyncio.timeout(0.5):
                while not entered_worker.is_set():
                    await asyncio.sleep(0.005)

            blocked_read.cancel()
            with pytest.raises(asyncio.CancelledError):
                await blocked_read

            with patch.object(
                sqlite_backend_module.asyncio, "wait", new=tracked_wait
            ):
                close_call = asyncio.create_task(
                    sqlite_backend_module._close_aiosqlite_connection(
                        connection,
                        retained_closes=retained_closes,
                    )
                )
                await first_wait_started.wait()
                close_call.cancel()
                await retry_wait_started.wait()
                close_call.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await close_call

            assert worker.is_alive()
            retained = retained_closes[connection]
            assert retained.close_task is not None
            assert retained.retirement_task is not None
            retirement_tasks = [retained.retirement_task]

            release_worker.set()
            await asyncio.gather(*retirement_tasks, return_exceptions=True)
            assert not worker.is_alive()
        finally:
            watchdog.cancel()
            release_worker.set()
            await asyncio.gather(blocked_read, return_exceptions=True)
            if close_call is not None:
                await asyncio.gather(close_call, return_exceptions=True)
            await asyncio.gather(*retirement_tasks, return_exceptions=True)
            if worker.is_alive():
                await connection.close()
            worker.join(timeout=1.0)

    @pytest.mark.asyncio
    async def test_backup_waits_for_destination_worker_exit(self, tmp_path):
        """A backup is not complete until its owned destination worker exits."""
        backend = SQLiteBackend(str(tmp_path / "source.db"))
        await backend.connect()
        release_worker = threading.Event()
        worker_exit_delayed = threading.Event()

        with delay_aiosqlite_worker_exit(release_worker, worker_exit_delayed) as workers:
            backup_task = asyncio.create_task(backend.backup_to(str(tmp_path / "backup.db")))
            try:
                await wait_for_lifecycle_checkpoint(
                    wait_until_aiosqlite_worker_exit_is_delayed(worker_exit_delayed),
                    backup_task,
                    description="the backup worker exit delay",
                )
                assert len(workers) == 1
                worker = workers[0]
                assert not backup_task.done()

                release_worker.set()
                await backup_task
                assert not worker.is_alive()
            finally:
                release_worker.set()
                if not backup_task.done():
                    await backup_task
                for worker in workers:
                    worker.join(timeout=1.0)
                await backend.close()

    @pytest.mark.asyncio
    async def test_snapshot_read_waits_for_one_shot_worker_exit(self, tmp_path):
        """A sibling read waits for its snapshot connection worker to exit."""
        backend = SQLiteBackend(str(tmp_path / "snapshot.db"))
        await backend.connect()
        await backend.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)")
        release_worker = threading.Event()
        worker_exit_delayed = threading.Event()

        try:
            async with backend.transaction():
                with delay_aiosqlite_worker_exit(
                    release_worker, worker_exit_delayed,
                ) as workers:
                    read_task = asyncio.create_task(backend.fetch_all("SELECT id FROM t"))
                    try:
                        await wait_for_lifecycle_checkpoint(
                            wait_until_aiosqlite_worker_exit_is_delayed(worker_exit_delayed),
                            read_task,
                            description="the snapshot worker exit delay",
                        )
                        assert len(workers) == 1
                        worker = workers[0]
                        assert not read_task.done()

                        release_worker.set()
                        assert await read_task == []
                        assert not worker.is_alive()
                    finally:
                        release_worker.set()
                        if not read_task.done():
                            await read_task
                        for worker in workers:
                            worker.join(timeout=1.0)
        finally:
            await backend.close()

    @pytest.mark.asyncio
    async def test_snapshot_setup_failure_waits_for_one_shot_worker_exit(self, tmp_path):
        """A failed snapshot setup closes and waits for its owned worker."""
        backend = SQLiteBackend(str(tmp_path / "snapshot-setup-failure.db"))
        await backend.connect()
        release_worker = threading.Event()
        worker_exit_delayed = threading.Event()

        try:
            with delay_aiosqlite_worker_exit(
                release_worker, worker_exit_delayed,
            ) as workers, patch.object(
                aiosqlite.Connection,
                "execute",
                AsyncMock(side_effect=RuntimeError("snapshot setup failed")),
            ):
                open_task = asyncio.create_task(backend._open_snapshot_read_connection())
                try:
                    await wait_for_lifecycle_checkpoint(
                        wait_until_aiosqlite_worker_exit_is_delayed(worker_exit_delayed),
                        open_task,
                        description="the failed snapshot worker exit delay",
                    )
                    assert len(workers) == 1
                    worker = workers[0]
                    assert not open_task.done()

                    release_worker.set()
                    with pytest.raises(RuntimeError, match="snapshot setup failed"):
                        await open_task
                    assert not worker.is_alive()
                finally:
                    release_worker.set()
                    if not open_task.done():
                        with suppress(RuntimeError):
                            await open_task
                    for worker in workers:
                        worker.join(timeout=1.0)
        finally:
            await backend.close()


class TestAsyncDatabase:
    """Test the AsyncDatabase facade."""

    @pytest.mark.asyncio
    async def test_column_shape_helpers_delegate_to_backend_safe_probes(self):
        from kestrel_sovereign.storage.async_database import AsyncDatabase

        database = AsyncDatabase(MagicMock())
        database._column_accepts_null = AsyncMock(return_value=True)
        database._column_has_default = AsyncMock(return_value=True)

        assert await database.column_accepts_null("widgets", "note") is True
        assert await database.column_has_default("widgets", "note") is True
        database._column_accepts_null.assert_awaited_once_with("widgets", "note")
        database._column_has_default.assert_awaited_once_with("widgets", "note")

    def test_nested_transaction_strategy_allow_list_and_backend_contracts(self):
        from kestrel_sovereign.storage.async_database import AsyncDatabase
        from kestrel_sovereign.storage.db.postgres import PostgresBackend

        unknown_backend = MagicMock()
        unknown_backend.nested_transaction_strategy = "future-strategy"
        assert AsyncDatabase(unknown_backend).nested_transaction_strategy is None
        assert SQLiteBackend(":memory:").nested_transaction_strategy == "joined"
        assert PostgresBackend(
            "postgresql://test:test@127.0.0.1/test"
        ).nested_transaction_strategy == "savepoint"

    @pytest.mark.asyncio
    async def test_owns_open_transaction_contracts(self, tmp_path):
        """Unknown backends answer ``None`` so callers fail closed; SQLite and
        PostgreSQL answer for the current task only."""
        from kestrel_sovereign.storage.async_database import AsyncDatabase
        from kestrel_sovereign.storage.db.postgres import PostgresBackend

        unknown_backend = MagicMock()
        unknown_backend.owns_open_transaction = MagicMock()
        assert AsyncDatabase(unknown_backend).owns_open_transaction is None
        assert AsyncDatabase(object()).owns_open_transaction is None

        database = await AsyncDatabase.sqlite(str(tmp_path / "owns.db"))
        try:
            assert database.owns_open_transaction is False
            async with database.transaction():
                assert database.owns_open_transaction is True
            assert database.owns_open_transaction is False
        finally:
            await database.close()

        # PostgreSQL keys its open transaction to the task that opened it.
        postgres = PostgresBackend("postgresql://test:test@127.0.0.1/test")
        assert postgres.owns_open_transaction is False
        token = postgres._txn_conn_var.set((asyncio.current_task(), object()))
        try:
            assert postgres.owns_open_transaction is True
            child = asyncio.get_running_loop().create_task(
                _read_owns_open_transaction(postgres)
            )
            assert await child is False
        finally:
            postgres._txn_conn_var.reset(token)
        assert postgres.owns_open_transaction is False

    @pytest.mark.asyncio
    async def test_column_shape_helpers_define_missing_column_semantics(self):
        from kestrel_sovereign.storage.async_database import AsyncDatabase

        database = await AsyncDatabase.sqlite(":memory:")
        try:
            await database.execute(
                "CREATE TABLE widgets (nullable TEXT DEFAULT 'note', "
                "required TEXT NOT NULL)"
            )

            assert await database.column_accepts_null("widgets", "nullable")
            assert not await database.column_accepts_null("widgets", "required")
            assert not await database.column_accepts_null("widgets", "missing")
            assert await database.column_has_default("widgets", "nullable")
            assert not await database.column_has_default("widgets", "required")
            assert not await database.column_has_default("widgets", "missing")
        finally:
            await database.close()

    @pytest.mark.asyncio
    async def test_cancelled_cached_sqla_disposal_still_closes_primary_worker(
        self, tmp_path,
    ):
        """The public close lifecycle cannot skip SQLite after pre-close cancellation."""
        from kestrel_sovereign.storage.async_database import AsyncDatabase
        from kestrel_sovereign.storage.sqla import make_session_factory

        release_worker = threading.Event()
        worker_exit_delayed = threading.Event()
        allow_factory_disposal = asyncio.Event()
        with delay_aiosqlite_worker_exit(
            release_worker, worker_exit_delayed,
        ) as workers:
            db = await AsyncDatabase.sqlite(str(tmp_path / "cached-factory.db"))
            connection = db.backend._connection
            assert connection is not None
            worker = aiosqlite_worker(connection)

            factory = make_session_factory(db)
            factory_close = factory.close
            factory_disposal_started = asyncio.Event()

            async def delayed_factory_close():
                factory_disposal_started.set()
                try:
                    await allow_factory_disposal.wait()
                finally:
                    await factory_close()

            factory.close = delayed_factory_close
            close_task = asyncio.create_task(db.close())

            try:
                await wait_for_lifecycle_checkpoint(
                    factory_disposal_started.wait(),
                    close_task,
                    description="SQLAlchemy factory disposal started",
                )
                close_task.cancel()
                await wait_for_lifecycle_checkpoint(
                    wait_until_aiosqlite_worker_exit_is_delayed(
                        worker_exit_delayed,
                    ),
                    close_task,
                    description="primary SQLite worker exit was delayed",
                )
                assert workers == [worker]
                assert not close_task.done()

                release_worker.set()
                with pytest.raises(asyncio.CancelledError):
                    await close_task
                assert not worker.is_alive()
            finally:
                allow_factory_disposal.set()
                release_worker.set()
                if not close_task.done():
                    with suppress(asyncio.CancelledError):
                        await close_task
                worker.join(timeout=1.0)

        assert not worker.is_alive()

    @pytest.mark.asyncio
    async def test_factory_close_timeout_fails_database_close_after_primary_attempt(
        self, tmp_path,
    ):
        """A timed-out cached factory cannot make ``AsyncDatabase.close`` lie.

        The factory uses a real file-backed SQLAlchemy/aiosqlite connection.
        Its worker is held after the close sentinel so the factory's bounded
        drain raises, while the primary backend still receives its own close
        chance before that failure reaches the caller.
        """
        from sqlalchemy import text

        from kestrel_sovereign.storage.async_database import AsyncDatabase
        from kestrel_sovereign.storage.sqla import make_session_factory

        db = await AsyncDatabase.sqlite(str(tmp_path / "factory-timeout.db"))
        primary_connection = db.backend._connection
        assert primary_connection is not None
        primary_worker = aiosqlite_worker(primary_connection)
        release_worker = threading.Event()
        worker_exit_delayed = threading.Event()
        factory_worker = None
        close_task = None
        retirement_tasks = []
        factory = None

        try:
            with delay_aiosqlite_worker_exit(
                release_worker,
                worker_exit_delayed,
                should_delay=lambda candidate: candidate is factory_worker,
            ) as workers, patch(
                "kestrel_sovereign.storage.db.sqlite."
                "AIOSQLITE_WORKER_SHUTDOWN_TIMEOUT_S",
                0.01,
            ):
                factory = make_session_factory(db)
                async with factory.read_session() as session:
                    await session.execute(text("SELECT 1"))
                factory_worker = aiosqlite_worker(factory._sqlite_connections[0])

                close_task = asyncio.create_task(db.close())
                # With the deliberately tiny lifecycle window, a busy runner
                # may expire either while Connection.close() is still being
                # scheduled or while waiting for the worker's final return.
                # Both are bounded factory-close failures. Wait until the
                # injected worker reaches that final return before inspecting
                # retained ownership, without requiring the already-failing
                # close lifecycle to remain pending until this task runs.
                await wait_for_lifecycle_checkpoint(
                    wait_until_aiosqlite_worker_exit_is_delayed(
                        worker_exit_delayed,
                    ),
                    close_task,
                    description="the timed-out factory worker exit delay",
                    require_live_lifecycle=False,
                )
                with pytest.raises(
                    ConnectionError,
                    match=(
                        "(?:SQLite worker did not terminate|"
                        "SQLite connection close did not complete)"
                    ),
                ):
                    await close_task

                # The factory timeout is observable, but only after the
                # primary backend close was attempted and its worker exited.
                # Join rather than assert on the instant: the worker's exit is
                # a real thread teardown, and on a loaded 2-core runner it had
                # simply not been scheduled yet by the time this line ran. The
                # deadline under test is the FACTORY's, asserted above; this is
                # only an observation bound, so it is generous.
                assert not db.backend.is_connected
                primary_worker.join(timeout=30.0)
                assert not primary_worker.is_alive()
                assert factory_worker.is_alive()
                assert workers == [factory_worker]
                assert db._sovereign_sqla_factory is None
                assert db._sovereign_sqla_retirement_owner is factory
                assert factory.sqlite_connection_retirement_pending
                # A later SQLAlchemy user must see the retained lifecycle
                # owner instead of replacing it with a falsely-safe factory or
                # opening a second worker through an existing factory handle.
                with pytest.raises(
                    ConnectionError, match="replacement factory cannot be created"
                ):
                    make_session_factory(db)
                tracked_connection_count = len(factory._sqlite_connections)
                for session_context in (
                    factory.read_session,
                    factory.write_session,
                ):
                    with pytest.raises(
                        ConnectionError, match="factory is closing or closed"
                    ):
                        async with session_context() as session:
                            await session.execute(text("SELECT 42"))
                    assert (
                        len(factory._sqlite_connections)
                        == tracked_connection_count
                    )
                retirement_tasks = list(
                    retained.retirement_task
                    for retained in factory._retired_sqlite_closes.values()
                )
                assert len(retirement_tasks) == 1
        finally:
            release_worker.set()
            await asyncio.gather(
                *retirement_tasks, return_exceptions=True
            )
            if close_task is not None:
                await asyncio.gather(close_task, return_exceptions=True)
            await db.finalize_retired_sqla_factory()
            if factory_worker is not None:
                factory_worker.join(timeout=1.0)
            primary_worker.join(timeout=1.0)

        assert factory_worker is not None and not factory_worker.is_alive()
        assert not primary_worker.is_alive()
        assert factory is not None
        assert not factory.sqlite_connection_retirement_pending
        assert getattr(db, "_sovereign_sqla_factory", None) is None
        assert getattr(db, "_sovereign_sqla_retirement_owner", None) is None

    @pytest.mark.asyncio
    async def test_sqla_factory_closes_two_workers_under_one_deadline(
        self, tmp_path,
    ):
        """A pooled factory retires every worker inside one reservation."""
        from sqlalchemy import text

        import kestrel_sovereign.storage.sqla.session as session_module
        from kestrel_sovereign.storage.async_database import AsyncDatabase
        from kestrel_sovereign.storage.sqla import make_session_factory

        db = await AsyncDatabase.sqlite(str(tmp_path / "sqla-two-workers.db"))
        release_workers = threading.Event()
        worker_exit_delayed = threading.Event()
        watchdog = threading.Timer(2.0, release_workers.set)
        sessions = []
        factory = None
        factory_workers = []
        retirement_tasks = []
        deadlines = []

        real_close = session_module._close_aiosqlite_connection

        async def record_deadline(
            connection,
            *,
            retained_closes,
            deadline=None,
            close_task=None,
            cancel_close_task_on_timeout=True,
        ):
            deadlines.append(deadline)
            await real_close(
                connection,
                retained_closes=retained_closes,
                deadline=deadline,
                close_task=close_task,
                cancel_close_task_on_timeout=cancel_close_task_on_timeout,
            )

        try:
            watchdog.start()
            with delay_aiosqlite_worker_exit(
                release_workers, worker_exit_delayed,
            ) as workers, patch.object(
                sqlite_backend_module,
                "AIOSQLITE_WORKER_SHUTDOWN_TIMEOUT_S",
                0.2,
            ), patch.object(
                session_module,
                "_close_aiosqlite_connection",
                new=record_deadline,
            ):
                factory = make_session_factory(db)
                first = factory._async_session()
                second = factory._async_session()
                sessions = [first, second]
                # Each live transaction keeps its QueuePool connection checked
                # out, forcing two real aiosqlite workers for the close path.
                await asyncio.gather(
                    first.execute(text("SELECT 1")),
                    second.execute(text("SELECT 2")),
                )
                assert len(factory._sqlite_connections) == 2
                factory_workers = [
                    aiosqlite_worker(connection)
                    for connection in factory._sqlite_connections
                ]

                started = asyncio.get_running_loop().time()
                with pytest.raises(
                    ConnectionError,
                    match=(
                        "(?:SQLite worker did not terminate|"
                        "SQLite connection close did not complete)"
                    ),
                ):
                    await factory.close()
                elapsed = asyncio.get_running_loop().time() - started

                reservation = factory.minimum_close_timeout_s
                # The 100ms margin is event-loop scheduling slack, not a second
                # per-worker window. A sequential implementation takes roughly
                # two full 210ms reservations and fails this bound.
                assert elapsed < reservation + 0.1
                assert len(deadlines) == 2
                assert deadlines[0] is not None
                assert deadlines[0] == deadlines[1]
                assert set(workers) == set(factory_workers)
                assert all(worker.is_alive() for worker in factory_workers)
                assert factory.sqlite_connection_retirement_pending
                assert len(factory._retired_sqlite_closes) == 2
                retirement_tasks = [
                    retained.retirement_task
                    for retained in factory._retired_sqlite_closes.values()
                    if retained.retirement_task is not None
                ]
                assert len(retirement_tasks) == 2
        finally:
            watchdog.cancel()
            release_workers.set()
            await asyncio.gather(
                *retirement_tasks, return_exceptions=True
            )
            for session in sessions:
                with suppress(Exception):
                    await session.close()
            if getattr(db, "_sovereign_sqla_factory", None) is not None:
                await db.dispose_cached_sqla_factory()
            await db.close()
            for worker in factory_workers:
                worker.join(timeout=1.0)

        assert factory is not None
        assert not factory.sqlite_connection_retirement_pending
        assert all(not worker.is_alive() for worker in factory_workers)

    @pytest.mark.asyncio
    async def test_sqla_factory_initial_gather_cancellation_retains_worker(
        self, tmp_path,
    ):
        """Cancellation before close wrappers run cannot orphan their workers."""
        from sqlalchemy import text

        import kestrel_sovereign.storage.sqla.session as session_module
        from kestrel_sovereign.storage.async_database import AsyncDatabase
        from kestrel_sovereign.storage.sqla import make_session_factory

        db = await AsyncDatabase.sqlite(str(tmp_path / "sqla-cancel-handoff.db"))
        factory = make_session_factory(db)
        session = factory._async_session()
        await session.execute(text("SELECT 1"))
        connection = factory._sqlite_connections[0]
        worker = aiosqlite_worker(connection)
        entered_worker = threading.Event()
        release_worker = threading.Event()
        watchdog = threading.Timer(2.0, release_worker.set)
        blocked_read = None
        retirement_tasks = []

        def block_in_worker() -> int:
            entered_worker.set()
            release_worker.wait()
            return 1

        await connection.create_function(
            "block_in_worker", 0, block_in_worker
        )

        real_gather = asyncio.gather
        gather_calls = 0

        def cancel_at_initial_gather(*awaitables, **kwargs):
            nonlocal gather_calls
            gather_calls += 1
            gathered = real_gather(*awaitables, **kwargs)
            if gather_calls == 1:
                current = asyncio.current_task()
                assert current is not None
                current.cancel()
            return gathered

        async def skip_engine_dispose(_engine, close=True):
            return None

        try:
            watchdog.start()
            blocked_read = asyncio.create_task(
                connection.execute("SELECT block_in_worker()")
            )
            async with asyncio.timeout(0.5):
                while not entered_worker.is_set():
                    await asyncio.sleep(0.005)
            blocked_read.cancel()
            with pytest.raises(asyncio.CancelledError):
                await blocked_read

            with patch.object(
                type(factory.engine), "dispose", skip_engine_dispose,
            ), patch.object(
                session_module.asyncio, "gather", new=cancel_at_initial_gather,
            ):
                with pytest.raises(asyncio.CancelledError):
                    await db.dispose_cached_sqla_factory()

            assert gather_calls == 2
            assert worker.is_alive()
            assert db._sovereign_sqla_factory is None
            assert db._sovereign_sqla_retirement_owner is factory
            assert factory.sqlite_connection_retirement_pending
            assert len(factory._retired_sqlite_closes) == 1
            retained = factory._retired_sqlite_closes[connection]
            assert retained.close_task is not None
            assert retained.retirement_task is not None
            retirement_tasks = [retained.retirement_task]

            tracked_connection_count = len(factory._sqlite_connections)
            with pytest.raises(
                ConnectionError, match="factory is closing or closed"
            ):
                async with factory.read_session() as fenced_session:
                    await fenced_session.execute(text("SELECT 42"))
            assert len(factory._sqlite_connections) == tracked_connection_count

            release_worker.set()
            await asyncio.gather(*retirement_tasks, return_exceptions=True)
            assert not worker.is_alive()
            assert not factory.sqlite_connection_retirement_pending
        finally:
            watchdog.cancel()
            release_worker.set()
            if blocked_read is not None:
                await asyncio.gather(blocked_read, return_exceptions=True)
            await asyncio.gather(*retirement_tasks, return_exceptions=True)
            with suppress(Exception):
                await session.close()
            await db.finalize_retired_sqla_factory()
            if db.backend.is_connected:
                await db.close()
            worker.join(timeout=1.0)

        assert not worker.is_alive()

    @pytest.mark.asyncio
    async def test_sqla_factory_hard_abandon_keeps_durable_retirement_owner(
        self, tmp_path,
    ):
        """Kestrel's GeneratorExit abandon path cannot orphan a pooled worker."""
        from sqlalchemy import text

        from kestrel_sovereign.storage.async_database import AsyncDatabase
        from kestrel_sovereign.storage.sqla import make_session_factory

        db = await AsyncDatabase.sqlite(str(tmp_path / "sqla-hard-abandon.db"))
        factory = make_session_factory(db)
        async with factory.read_session() as session:
            await session.execute(text("SELECT 1"))
        connection = factory._sqlite_connections[0]
        worker = aiosqlite_worker(connection)
        entered_worker = threading.Event()
        release_worker = threading.Event()
        watchdog = threading.Timer(2.0, release_worker.set)
        blocked_read = None
        dispose_task = None
        retirement_tasks = []

        def block_in_worker() -> int:
            entered_worker.set()
            release_worker.wait()
            return 1

        await connection.create_function("block_in_worker", 0, block_in_worker)

        try:
            watchdog.start()
            blocked_read = asyncio.create_task(
                connection.execute("SELECT block_in_worker()")
            )
            async with asyncio.timeout(0.5):
                while not entered_worker.is_set():
                    await asyncio.sleep(0.005)
            blocked_read.cancel()
            with pytest.raises(asyncio.CancelledError):
                await blocked_read

            dispose_task = asyncio.create_task(db.dispose_cached_sqla_factory())
            async with asyncio.timeout(0.5):
                while not factory.sqlite_connection_retirement_pending:
                    await asyncio.sleep(0)

            # This is the production shutdown's _abandon sequence: request
            # cooperative cancellation, then hard-close the suspended outer
            # coroutine so GeneratorExit prevents it from running an except
            # path that could accidentally shed lifecycle ownership.
            dispose_task.cancel()
            dispose_task.get_coro().close()
            await asyncio.sleep(0)

            assert db._sovereign_sqla_factory is None
            assert db._sovereign_sqla_retirement_owner is factory
            assert db.connection_retirement_pending
            assert factory.sqlite_connection_retirement_pending
            assert worker.is_alive()
            with pytest.raises(
                ConnectionError, match="replacement factory cannot be created"
            ):
                make_session_factory(db)
            with pytest.raises(
                ConnectionError, match="factory is closing or closed"
            ):
                async with factory.read_session() as stale_session:
                    await stale_session.execute(text("SELECT 42"))

            retirement_tasks = [
                retained.retirement_task
                for retained in factory._retired_sqlite_closes.values()
                if retained.retirement_task is not None
            ]
            assert len(retirement_tasks) == 1

            release_worker.set()
            await asyncio.gather(*retirement_tasks, return_exceptions=True)
            if factory._engine_dispose_task is not None:
                await asyncio.gather(
                    factory._engine_dispose_task, return_exceptions=True
                )
            await asyncio.gather(dispose_task, return_exceptions=True)
            assert not worker.is_alive()
            assert not factory.sqlite_connection_retirement_pending

            await db.finalize_retired_sqla_factory()
            assert db._sovereign_sqla_retirement_owner is None
        finally:
            watchdog.cancel()
            release_worker.set()
            if blocked_read is not None:
                await asyncio.gather(blocked_read, return_exceptions=True)
            await asyncio.gather(*retirement_tasks, return_exceptions=True)
            if dispose_task is not None:
                await asyncio.gather(dispose_task, return_exceptions=True)
            if db.connection_retirement_pending:
                pending = [
                    retained.retirement_task
                    for retained in factory._retired_sqlite_closes.values()
                    if retained.retirement_task is not None
                ]
                await asyncio.gather(*pending, return_exceptions=True)
            await db.finalize_retired_sqla_factory()
            if db.backend.is_connected:
                await db.close()
            worker.join(timeout=1.0)

        assert not worker.is_alive()

    @pytest.mark.asyncio
    async def test_non_sqlite_hard_abandon_fences_reinitialize_and_surfaces_error(
        self, tmp_path,
    ):
        """An owned non-SQLite disposal remains visible after GeneratorExit."""
        from kestrel_sovereign.storage.async_storage import AsyncStorage
        from kestrel_sovereign.storage.sqla import make_session_factory
        from kestrel_sovereign.storage.sqla.session import (
            SovereignSqlaSessionFactory,
        )

        disposal_started = asyncio.Event()
        release_disposal = asyncio.Event()

        class PendingPostgresEngine:
            class dialect:
                name = "postgresql"

            async def dispose(self):
                disposal_started.set()
                await release_disposal.wait()
                raise RuntimeError("late PostgreSQL dispose failure")

        storage = AsyncStorage(str(tmp_path / "postgres-dispose-owner.db"))
        await storage.initialize()
        first_db = storage.db
        assert first_db is not None
        factory = SovereignSqlaSessionFactory(PendingPostgresEngine())
        first_db._sovereign_sqla_factory = factory
        dispose_task = asyncio.create_task(
            storage.dispose_cached_sqla_factory()
        )

        try:
            await asyncio.wait_for(disposal_started.wait(), timeout=0.5)
            assert factory.retirement_pending
            assert not factory.sqlite_connection_retirement_pending
            assert first_db.connection_retirement_pending
            assert first_db._sovereign_sqla_factory is None
            assert first_db._sovereign_sqla_retirement_owner is factory

            # Match Kestrel's bounded shutdown abandonment: cooperative
            # cancellation followed by GeneratorExit injected into the outer
            # suspended coroutine.  The independent disposal stays visible.
            dispose_task.cancel()
            dispose_task.get_coro().close()
            await asyncio.sleep(0)

            assert factory.retirement_pending
            assert first_db.connection_retirement_pending
            with pytest.raises(
                ConnectionError, match="replacement factory cannot be created"
            ):
                make_session_factory(first_db)
            with pytest.raises(
                ConnectionError, match="retirement is still pending"
            ):
                await first_db.finalize_retired_sqla_factory()

            # Closing the facade still gives its primary backend a chance, but
            # cannot report this independent engine lifecycle as complete.
            with pytest.raises(
                ConnectionError,
                match="engine or connection retirement is still pending",
            ):
                await storage.close()
            assert not storage._initialized
            assert not storage._backend.is_connected
            assert storage.db is first_db

            with pytest.raises(
                ConnectionError, match="previous SQLAlchemy engine"
            ):
                await storage.initialize()
            assert storage.db is first_db
            assert not storage._backend.is_connected

            release_disposal.set()
            assert factory._engine_dispose_task is not None
            await asyncio.gather(
                factory._engine_dispose_task, return_exceptions=True
            )
            await asyncio.gather(dispose_task, return_exceptions=True)
            assert not factory.retirement_pending
            assert not first_db.connection_retirement_pending
            assert first_db._sovereign_sqla_retirement_owner is factory

            # The done callback retrieved the late task exception, and the
            # retained owner now surfaces it before replacement is allowed.
            with pytest.raises(
                RuntimeError, match="late PostgreSQL dispose failure"
            ):
                await storage.initialize()
            assert first_db._sovereign_sqla_retirement_owner is None
            assert not storage._backend.is_connected

            await storage.initialize()
            assert storage._initialized
            assert storage.db is not first_db
            assert await storage.db.fetchval("SELECT 42") == 42
        finally:
            release_disposal.set()
            if factory._engine_dispose_task is not None:
                await asyncio.gather(
                    factory._engine_dispose_task, return_exceptions=True
                )
            await asyncio.gather(dispose_task, return_exceptions=True)
            if storage._initialized:
                with suppress(Exception):
                    await storage.close()
            elif storage._backend.is_connected:
                await storage._backend.close()

    @pytest.mark.asyncio
    async def test_closed_sqla_factory_reference_cannot_reopen_worker(
        self, tmp_path,
    ):
        """A cleanly closed factory stays permanently closed to stale callers."""
        from sqlalchemy import text

        from kestrel_sovereign.storage.async_database import AsyncDatabase
        from kestrel_sovereign.storage.sqla import make_session_factory

        db = await AsyncDatabase.sqlite(str(tmp_path / "sqla-stale-factory.db"))
        factory = make_session_factory(db)
        async with factory.read_session() as session:
            await session.execute(text("SELECT 1"))
        tracked_count = len(factory._sqlite_connections)
        worker = aiosqlite_worker(factory._sqlite_connections[0])

        try:
            await db.dispose_cached_sqla_factory()
            assert not worker.is_alive()
            assert db._sovereign_sqla_factory is None
            assert db._sovereign_sqla_retirement_owner is None

            for session_context in (
                factory.read_session,
                factory.write_session,
            ):
                with pytest.raises(
                    ConnectionError, match="factory is closing or closed"
                ):
                    async with session_context() as stale_session:
                        await stale_session.execute(text("SELECT 42"))
                assert len(factory._sqlite_connections) == tracked_count
        finally:
            await db.close()
            worker.join(timeout=1.0)

        assert not worker.is_alive()

    @pytest.mark.asyncio
    async def test_pre_admitted_lazy_session_cannot_connect_after_factory_close(
        self, tmp_path,
    ):
        """A yielded but still-lazy session observes the permanent close fence."""
        from sqlalchemy import text

        from kestrel_sovereign.storage.async_database import AsyncDatabase
        from kestrel_sovereign.storage.sqla import make_session_factory

        db = await AsyncDatabase.sqlite(str(tmp_path / "sqla-lazy-session.db"))
        factory = make_session_factory(db)
        session_context = factory.read_session()
        session = await session_context.__aenter__()

        try:
            # Merely entering an AsyncSession context does not check out a
            # connection.  This is the ordering that previously let a stale
            # session reopen SQLAlchemy's disposed engine.
            assert factory._sqlite_connections == []

            await db.dispose_cached_sqla_factory()
            assert not factory.retirement_pending

            with pytest.raises(
                ConnectionError, match="factory is closing or closed"
            ):
                await session.execute(text("SELECT 42"))
            assert factory._sqlite_connections == []
        finally:
            await session_context.__aexit__(None, None, None)
            await db.close()

    @pytest.mark.asyncio
    async def test_checked_in_blocked_worker_does_not_bypass_factory_deadline(
        self, tmp_path, monkeypatch,
    ):
        """Engine disposal and raw-driver cleanup share one bounded deadline."""
        from sqlalchemy import text

        from kestrel_sovereign.storage.async_database import AsyncDatabase
        from kestrel_sovereign.storage.sqla import make_session_factory

        # Shared close+join deadline; the worker here is held, so the JOIN is
        # the stage that must expire. Too small and a loaded machine cannot
        # finish the CLOSE either, which is why this file's siblings flaked.
        timeout_seconds = 0.5
        monkeypatch.setattr(
            sqlite_backend_module,
            "AIOSQLITE_WORKER_SHUTDOWN_TIMEOUT_S",
            timeout_seconds,
        )
        db = await AsyncDatabase.sqlite(str(tmp_path / "sqla-dispose-deadline.db"))
        factory = make_session_factory(db)
        async with factory.read_session() as session:
            await session.execute(text("SELECT 1"))

        connection = factory._sqlite_connections[0]
        worker = aiosqlite_worker(connection)
        entered_worker = threading.Event()
        release_worker = threading.Event()
        watchdog = threading.Timer(2.0, release_worker.set)
        blocked_read = None
        retirement_tasks = []
        close_task = None

        def block_in_worker() -> int:
            entered_worker.set()
            release_worker.wait()
            return 1

        await connection.create_function(
            "block_in_worker", 0, block_in_worker
        )

        try:
            watchdog.start()
            blocked_read = asyncio.create_task(
                connection.execute("SELECT block_in_worker()")
            )
            async with asyncio.timeout(0.5):
                while not entered_worker.is_set():
                    await asyncio.sleep(0.005)
            blocked_read.cancel()
            with pytest.raises(asyncio.CancelledError):
                await blocked_read

            started = asyncio.get_running_loop().time()
            close_task = asyncio.create_task(
                db.dispose_cached_sqla_factory()
            )
            with pytest.raises(
                ConnectionError,
                match=(
                    "(?:SQLite worker did not terminate|"
                    "SQLite connection close did not complete)"
                ),
            ) as close_error:
                await close_task
            elapsed = asyncio.get_running_loop().time() - started

            assert elapsed >= timeout_seconds
            assert elapsed < 1.5
            assert isinstance(close_error.value.__cause__, ConnectionError)
            assert "engine or session disposal did not complete" in str(
                close_error.value.__cause__
            )
            assert worker.is_alive()
            assert factory._engine_dispose_task is not None
            assert not factory._engine_dispose_task.done()
            assert factory.retirement_pending
            assert factory.sqlite_connection_retirement_pending
            assert db._sovereign_sqla_factory is None
            assert db._sovereign_sqla_retirement_owner is factory
            assert db.connection_retirement_pending

            retirement_tasks = [
                retained.retirement_task
                for retained in factory._retired_sqlite_closes.values()
                if retained.retirement_task is not None
            ]
            assert len(retirement_tasks) == 1

            release_worker.set()
            await asyncio.gather(
                factory._engine_dispose_task,
                *retirement_tasks,
                return_exceptions=True,
            )
            assert not worker.is_alive()
            assert not factory.retirement_pending
            await db.finalize_retired_sqla_factory()
            assert db._sovereign_sqla_retirement_owner is None
        finally:
            watchdog.cancel()
            release_worker.set()
            if blocked_read is not None:
                await asyncio.gather(blocked_read, return_exceptions=True)
            await asyncio.gather(*retirement_tasks, return_exceptions=True)
            if factory._engine_dispose_task is not None:
                await asyncio.gather(
                    factory._engine_dispose_task, return_exceptions=True
                )
            if close_task is not None:
                await asyncio.gather(close_task, return_exceptions=True)
            if db.connection_retirement_pending:
                await db.finalize_retired_sqla_factory()
            if db.backend.is_connected:
                await db.close()
            worker.join(timeout=1.0)

        assert not worker.is_alive()

    @pytest.mark.asyncio
    async def test_factory_dispose_error_preserves_cleanup_timeout_for_checked_out_worker(
        self, tmp_path,
    ):
        """A pre-disposal error cannot strand or hide a checked-out worker error."""
        from sqlalchemy import text

        from kestrel_sovereign.storage.async_database import AsyncDatabase
        from kestrel_sovereign.storage.sqla import make_session_factory

        db = await AsyncDatabase.sqlite(str(tmp_path / "factory-error.db"))
        primary_connection = db.backend._connection
        assert primary_connection is not None
        primary_worker = aiosqlite_worker(primary_connection)
        release_worker = threading.Event()
        worker_exit_delayed = threading.Event()
        factory_worker = None
        close_task = None
        session = None
        retirement_tasks = []
        factory = None

        try:
            with delay_aiosqlite_worker_exit(
                release_worker,
                worker_exit_delayed,
                should_delay=lambda candidate: candidate is factory_worker,
            ) as workers, patch(
                # 0.5s, not tens of milliseconds. This budget is a DEADLINE
                # SHARED by two stages (sqlite.py: deadline = now + this + one
                # poll): first aiosqlite's own close, then the worker join. The
                # worker here is held deliberately, so the join is what must
                # expire -- but at 0.01s a loaded machine could not finish the
                # CLOSE either, and whichever stage lost the race decided the
                # error message and whether the primary worker had exited yet.
                # That is what made this test flaky under `-n auto`. Keep this
                # comfortably above a real close and below the 2.0s watchdog.
                "kestrel_sovereign.storage.db.sqlite."
                "AIOSQLITE_WORKER_SHUTDOWN_TIMEOUT_S",
                0.5,
            ):
                factory = make_session_factory(db)
                # Keep the SQLAlchemy connection checked out.  Engine disposal
                # intentionally leaves this connection alone, so this drives
                # the fallback that explicitly closes tracked raw drivers.
                session = factory._async_session()
                await session.execute(text("SELECT 1"))
                factory_worker = aiosqlite_worker(factory._sqlite_connections[0])

                async def fail_before_dispose(_engine, close=True):
                    raise RuntimeError("injected factory dispose error")

                with patch.object(
                    type(factory.engine), "dispose", fail_before_dispose,
                ):
                    close_task = asyncio.create_task(db.close())

                    # Unlike the other checkpoints, this drain is patched to
                    # 0.01s and db.close() is *asserted* to raise below, so
                    # close_task finishing first is an expected order rather
                    # than a lifecycle regression.  The held worker still sets
                    # the checkpoint on its way out.
                    await wait_for_lifecycle_checkpoint(
                        wait_until_aiosqlite_worker_exit_is_delayed(
                            worker_exit_delayed,
                        ),
                        close_task,
                        description="the failed factory worker exit delay",
                        require_live_lifecycle=False,
                    )
                    assert workers == [factory_worker]
                    assert factory_worker.is_alive()
                    # The fallback sends the close sentinel to the checked-out
                    # driver.  Its bounded drain fails while that worker is
                    # deliberately held, but preserves the original disposal
                    # failure as context instead of hiding either error.
                    with pytest.raises(
                        ConnectionError, match="worker did not terminate",
                    ) as close_error:
                        await close_task
                    assert isinstance(close_error.value.__cause__, RuntimeError)
                    assert "injected factory dispose error" in str(
                        close_error.value.__cause__
                    )

                    assert not db.backend.is_connected
                    assert factory_worker.is_alive()
                    assert not primary_worker.is_alive()
                    assert db._sovereign_sqla_factory is None
                    assert db._sovereign_sqla_retirement_owner is factory
                    retirement_tasks = list(
                        retained.retirement_task
                        for retained in factory._retired_sqlite_closes.values()
                    )
                    assert len(retirement_tasks) == 1
        finally:
            release_worker.set()
            await asyncio.gather(
                *retirement_tasks, return_exceptions=True
            )
            if close_task is not None and not close_task.done():
                with suppress(ConnectionError):
                    await close_task
            if session is not None:
                with suppress(Exception):
                    await session.close()
            await db.finalize_retired_sqla_factory()
            if db.backend.is_connected:
                await db.close()
            if factory_worker is not None:
                factory_worker.join(timeout=1.0)
            primary_worker.join(timeout=1.0)

        assert factory_worker is not None and not factory_worker.is_alive()
        assert not primary_worker.is_alive()

    @pytest.mark.asyncio
    async def test_async_storage_reinitializes_after_factory_close_failure(
        self, tmp_path,
    ):
        """A propagated factory-close failure cannot leave the facade initialized."""
        from sqlalchemy import text

        from kestrel_sovereign.storage.async_storage import AsyncStorage
        from kestrel_sovereign.storage.sqla import make_session_factory

        storage = AsyncStorage(str(tmp_path / "storage-reinitialize.db"))
        await storage.initialize()
        first_db = storage.db
        assert first_db is not None
        factory = make_session_factory(first_db)
        async with factory.read_session() as session:
            await session.execute(text("SELECT 1"))
        factory_worker = aiosqlite_worker(factory._sqlite_connections[0])

        async def fail_before_dispose(_engine, close=True):
            raise RuntimeError("injected factory dispose error")

        try:
            with patch.object(type(factory.engine), "dispose", fail_before_dispose):
                with pytest.raises(RuntimeError, match="injected factory dispose error"):
                    await storage.close()

            assert not storage._initialized
            assert not first_db._initialized
            assert not storage._backend.is_connected
            assert not factory_worker.is_alive()

            await storage.initialize()
            assert storage._initialized
            assert storage.db is not first_db
            assert await storage.db.fetchval("SELECT 1") == 1
        finally:
            if storage._initialized:
                await storage.close()
            factory_worker.join(timeout=1.0)

        assert not factory_worker.is_alive()

    @pytest.mark.asyncio
    async def test_async_storage_reinitialize_waits_for_retained_factory_worker(
        self, tmp_path, monkeypatch,
    ):
        """Reinitialization cannot replace a database that owns a live worker."""
        from sqlalchemy import text

        from kestrel_sovereign.storage.async_storage import AsyncStorage
        from kestrel_sovereign.storage.sqla import make_session_factory

        # See the note above: this is a shared close+join deadline, and the
        # join is the stage this test holds open. 0.05s was small enough that a
        # loaded machine spent the whole budget inside the close and the test
        # timed out instead of asserting. Below the 2.0s watchdog.
        monkeypatch.setattr(
            sqlite_backend_module,
            "AIOSQLITE_WORKER_SHUTDOWN_TIMEOUT_S",
            0.5,
        )
        storage = AsyncStorage(str(tmp_path / "storage-retained-worker.db"))
        await storage.initialize()
        first_db = storage.db
        assert first_db is not None
        factory = make_session_factory(first_db)
        session = factory._async_session()
        await session.execute(text("SELECT 1"))
        connection = factory._sqlite_connections[0]
        worker = aiosqlite_worker(connection)
        entered_worker = threading.Event()
        release_worker = threading.Event()
        watchdog = threading.Timer(2.0, release_worker.set)
        blocked_read = None
        retirement_tasks = []

        def block_in_worker() -> int:
            entered_worker.set()
            release_worker.wait()
            return 1

        await connection.create_function(
            "block_in_worker", 0, block_in_worker
        )

        try:
            watchdog.start()
            blocked_read = asyncio.create_task(
                connection.execute("SELECT block_in_worker()")
            )
            async with asyncio.timeout(0.5):
                while not entered_worker.is_set():
                    await asyncio.sleep(0.005)
            blocked_read.cancel()
            with pytest.raises(asyncio.CancelledError):
                await blocked_read

            with pytest.raises(
                ConnectionError,
                match=(
                    "(?:SQLite worker did not terminate|"
                    "SQLite connection close did not complete)"
                ),
            ):
                await storage.close()

            assert not storage._initialized
            assert storage.db is first_db
            assert first_db._sovereign_sqla_factory is None
            assert first_db._sovereign_sqla_retirement_owner is factory
            assert first_db.connection_retirement_pending
            assert not storage._backend.is_connected
            assert worker.is_alive()
            assert factory.sqlite_connection_retirement_pending
            tracked_connection_count = len(factory._sqlite_connections)
            retirement_tasks = [
                retained.retirement_task
                for retained in factory._retired_sqlite_closes.values()
                if retained.retirement_task is not None
            ]
            assert len(retirement_tasks) == 1

            with pytest.raises(
                ConnectionError, match="previous SQLAlchemy engine"
            ):
                await storage.initialize()
            assert storage.db is first_db
            assert not storage._backend.is_connected
            assert len(factory._sqlite_connections) == tracked_connection_count

            release_worker.set()
            await asyncio.gather(*retirement_tasks, return_exceptions=True)
            assert not worker.is_alive()
            assert not first_db.connection_retirement_pending

            with suppress(Exception):
                await session.close()
            with pytest.raises(
                ConnectionError, match="factory is closing or closed"
            ):
                async with factory.read_session() as stale_session:
                    await stale_session.execute(text("SELECT 42"))
            await storage.initialize()
            assert storage._initialized
            assert storage.db is not first_db
            assert first_db._sovereign_sqla_retirement_owner is None
            assert await storage.db.fetchval("SELECT 42") == 42
            with pytest.raises(
                ConnectionError, match="factory is closing or closed"
            ):
                async with factory.read_session() as stale_session:
                    await stale_session.execute(text("SELECT 42"))
        finally:
            watchdog.cancel()
            release_worker.set()
            if blocked_read is not None:
                await asyncio.gather(blocked_read, return_exceptions=True)
            await asyncio.gather(*retirement_tasks, return_exceptions=True)
            with suppress(Exception):
                await session.close()
            if storage._initialized:
                await storage.close()
            worker.join(timeout=1.0)

        assert not worker.is_alive()

    @pytest.mark.asyncio
    async def test_sqla_factory_close_waits_for_its_aiosqlite_worker(self, tmp_path):
        """A cached file-backed SQLAlchemy engine owns its driver's last turn."""
        from sqlalchemy import text

        from kestrel_sovereign.storage.async_database import AsyncDatabase
        from kestrel_sovereign.storage.sqla import make_session_factory

        db = await AsyncDatabase.sqlite(str(tmp_path / "sqla-worker.db"))
        release_worker = threading.Event()
        worker_exit_delayed = threading.Event()
        worker = None

        try:
            with delay_aiosqlite_worker_exit(
                release_worker,
                worker_exit_delayed,
                should_delay=lambda candidate: candidate is worker,
            ) as workers:
                factory = make_session_factory(db)
                async with factory.read_session() as session:
                    await session.execute(text("SELECT 1"))

                connection = factory._sqlite_connections[0]
                worker = aiosqlite_worker(connection)
                close_task = asyncio.create_task(factory.close())
                try:
                    await wait_for_lifecycle_checkpoint(
                        wait_until_aiosqlite_worker_exit_is_delayed(
                            worker_exit_delayed,
                        ),
                        close_task,
                        description="the SQLAlchemy worker exit delay",
                    )
                    assert workers == [worker]
                    assert not close_task.done()

                    release_worker.set()
                    await close_task
                    assert not worker.is_alive()
                finally:
                    release_worker.set()
                    if not close_task.done():
                        await close_task
                    worker.join(timeout=1.0)
        finally:
            await db.close()

        assert worker is not None and not worker.is_alive()

    @pytest.mark.asyncio
    async def test_cancelled_sqla_factory_close_waits_for_its_worker(self, tmp_path):
        """Cancellation reaches the caller only after the SQLAlchemy worker exits."""
        from sqlalchemy import text

        from kestrel_sovereign.storage.async_database import AsyncDatabase
        from kestrel_sovereign.storage.sqla import make_session_factory

        db = await AsyncDatabase.sqlite(str(tmp_path / "sqla-cancelled-worker.db"))
        release_worker = threading.Event()
        worker_exit_delayed = threading.Event()
        worker = None

        try:
            with delay_aiosqlite_worker_exit(
                release_worker,
                worker_exit_delayed,
                should_delay=lambda candidate: candidate is worker,
            ) as workers:
                factory = make_session_factory(db)
                async with factory.read_session() as session:
                    await session.execute(text("SELECT 1"))

                connection = factory._sqlite_connections[0]
                worker = aiosqlite_worker(connection)
                close_task = asyncio.create_task(factory.close())
                try:
                    await wait_for_lifecycle_checkpoint(
                        wait_until_aiosqlite_worker_exit_is_delayed(
                            worker_exit_delayed,
                        ),
                        close_task,
                        description="the cancelled SQLAlchemy worker exit delay",
                    )
                    assert workers == [worker]
                    close_task.cancel()
                    await asyncio.sleep(0)
                    assert not close_task.done()

                    release_worker.set()
                    with pytest.raises(asyncio.CancelledError):
                        await close_task
                    assert not worker.is_alive()
                finally:
                    release_worker.set()
                    if not close_task.done():
                        with suppress(asyncio.CancelledError):
                            await close_task
                    worker.join(timeout=1.0)
        finally:
            await db.close()

        assert worker is not None and not worker.is_alive()
    
    @pytest.mark.asyncio
    async def test_sqlite_factory(self, tmp_path):
        from kestrel_sovereign.storage.async_database import AsyncDatabase
        
        db_path = str(tmp_path / "test.db")
        db = await AsyncDatabase.sqlite(db_path)
        
        assert db.backend_type == "sqlite"
        
        # Core tables should exist
        assert await db.table_exists("files")
        assert await db.table_exists("conversation_history")
        assert await db.table_exists("wallet_state")
        
        await db.close()
    
    @pytest.mark.asyncio
    async def test_execute_and_fetch(self, tmp_path):
        from kestrel_sovereign.storage.async_database import AsyncDatabase
        
        db_path = str(tmp_path / "test.db")
        db = await AsyncDatabase.sqlite(db_path)
        
        await db.execute(
            "INSERT INTO files (content_hash, original_name) VALUES (?, ?)",
            ("hash123", "test.txt")
        )
        
        row = await db.fetchone(
            "SELECT * FROM files WHERE content_hash = ?",
            ("hash123",)
        )
        assert row is not None
        assert row[0] == "hash123"
        assert row[1] == "test.txt"
        
        await db.close()
    
    @pytest.mark.asyncio
    async def test_context_manager(self, tmp_path):
        from kestrel_sovereign.storage.async_database import AsyncDatabase, SQLiteBackend
        
        db_path = str(tmp_path / "test.db")
        backend = SQLiteBackend(db_path)
        await backend.connect()
        
        async with AsyncDatabase(backend) as db:
            assert await db.table_exists("files")
        
        # Should be closed after context
        assert not backend.is_connected
