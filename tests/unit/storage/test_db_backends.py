"""
Tests for database backend abstraction layer.
"""
import pytest
from kestrel_sovereign.storage.db import (
    DatabaseBackend,
    SQLiteBackend,
    sqlite_to_postgres,
    postgres_to_sqlite,
    normalize_schema,
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
        # The writer is blocked on the write lock (still). This read is from a
        # NON-owner task, which now reads the separate read connection and sees
        # the COMMITTED snapshot — neither the transaction's uncommitted row (1)
        # nor the blocked writer's row (2). That's read-committed isolation
        # (#1726); previously this connection-shared read saw the dirty row (1).
        mid = await backend.fetch_all("SELECT id FROM t ORDER BY id")
        assert mid == []

        release.set()
        await asyncio.gather(t1, t2)
        final = await backend.fetch_all("SELECT id FROM t ORDER BY id")
        assert final == [(1,), (2,)]

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


class TestAsyncDatabase:
    """Test the AsyncDatabase facade."""
    
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
