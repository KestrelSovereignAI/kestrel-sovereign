"""
Tests for database backend abstraction layer.
"""
import asyncio
import threading
from contextlib import suppress
from unittest.mock import AsyncMock, patch

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

            # Let the drain consume more than the old one-phase reservation,
            # then let connection.close() retire the real worker.  No timeout
            # constants are patched: this verifies the public default budget.
            loop = asyncio.get_running_loop()
            release_handle = loop.call_later(
                1.5 * shutdown_window,
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
    async def test_cleanup_error_does_not_survive_close_or_connect(self, tmp_path):
        """A fresh connection is not poisoned by an old drain failure."""
        backend = SQLiteBackend(str(tmp_path / "drain-recovery.db"))
        await backend.connect()
        backend._cancelled_write_drain_error = RuntimeError("rollback failed")

        with pytest.raises(QueryError, match="cancellation cleanup failed"):
            await backend.fetch_one("SELECT 1")

        await backend.close()
        assert backend._cancelled_write_drain_error is None

        # Exercise connect's reset independently from close's reset.
        backend._cancelled_write_drain_error = RuntimeError("stale failure")
        await backend.connect()
        try:
            assert backend._cancelled_write_drain_error is None
            assert await backend.fetch_one("SELECT 1") == (1,)
        finally:
            await backend.close()

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
            try:
                with pytest.raises(ConnectionError, match="worker did not terminate"):
                    await backend.close()
                assert not backend.is_connected
                assert worker.is_alive()
                assert workers == [worker]
            finally:
                release_worker.set()
                worker.join(timeout=1.0)

        assert not worker.is_alive()

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

                with pytest.raises(ConnectionError, match="worker did not terminate"):
                    await db.close()

                # The factory timeout is observable, but only after the
                # primary backend close was attempted and its worker exited.
                assert not db.backend.is_connected
                assert not primary_worker.is_alive()
                assert factory_worker.is_alive()
                assert workers == [factory_worker]
        finally:
            release_worker.set()
            if factory_worker is not None:
                factory_worker.join(timeout=1.0)
            primary_worker.join(timeout=1.0)

        assert factory_worker is not None and not factory_worker.is_alive()
        assert not primary_worker.is_alive()

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
        finally:
            release_worker.set()
            if close_task is not None and not close_task.done():
                with suppress(ConnectionError):
                    await close_task
            if session is not None:
                with suppress(Exception):
                    await session.close()
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
