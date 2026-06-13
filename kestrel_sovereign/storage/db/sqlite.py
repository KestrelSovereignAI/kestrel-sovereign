"""
SQLite Database Backend

Implementation of DatabaseBackend using aiosqlite.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, List, Optional, Sequence

import aiosqlite

from .interface import (
    ConnectionError,
    DatabaseBackend,
    Params,
    QueryError,
    Row,
    TransactionError,
)
from .write_audit import record_write_query, record_write_script

logger = logging.getLogger(__name__)


class SQLiteBackend(DatabaseBackend):
    """
    SQLite database backend using aiosqlite.
    
    Features:
    - File-based or in-memory databases
    - Automatic directory creation
    - WAL mode for better concurrency
    - Foreign key enforcement
    """
    
    def __init__(self, db_path: str):
        """
        Initialize SQLite backend.
        
        Args:
            db_path: Path to SQLite database file, or ":memory:" for in-memory
        """
        self.db_path = db_path
        self._connection: Optional[aiosqlite.Connection] = None
        self._in_transaction = False
        # Serializes write *units* on the single shared connection. aiosqlite
        # serializes individual operations, but NOT the execute->commit/rollback
        # pair: without this, two concurrent autocommit writers share one
        # connection-scoped transaction, so one writer's failed rollback() can
        # discard the other's uncommitted write (#1675). The lock makes each
        # autocommit statement and each explicit transaction an atomic write
        # unit. Re-entrant for the task that owns an open transaction (its own
        # statements must not deadlock on the lock it already holds).
        self._write_lock = asyncio.Lock()
        self._txn_owner: Optional["asyncio.Task"] = None
    
    @property
    def backend_type(self) -> str:
        return "sqlite"
    
    @property
    def is_connected(self) -> bool:
        return self._connection is not None
    
    async def connect(self) -> None:
        """Connect to SQLite database."""
        if self._connection is not None:
            return
        
        try:
            # Create directory if needed (unless in-memory)
            if self.db_path != ":memory:":
                db_dir = Path(self.db_path).parent
                db_dir.mkdir(parents=True, exist_ok=True)
            
            self._connection = await aiosqlite.connect(self.db_path, timeout=30)
            
            # Enable WAL mode for better concurrency
            await self._connection.execute("PRAGMA journal_mode=WAL")
            
            # Allow concurrent writers to wait up to 30s for the lock
            await self._connection.execute("PRAGMA busy_timeout=30000")
            
            # Enable foreign keys
            await self._connection.execute("PRAGMA foreign_keys=ON")
            
            # Row factory to return tuples
            self._connection.row_factory = aiosqlite.Row
            
            logger.debug(f"Connected to SQLite: {self.db_path}")
            
        except Exception as e:
            raise ConnectionError(f"Failed to connect to SQLite: {e}") from e
    
    async def close(self) -> None:
        """Close database connection and wait for background thread to stop."""
        if self._connection is not None:
            conn = self._connection
            try:
                await conn.close()
                logger.debug(f"Closed SQLite connection: {self.db_path}")
            finally:
                self._connection = None
    
    def _ensure_connected(self) -> aiosqlite.Connection:
        """Ensure we have an active connection."""
        if self._connection is None:
            raise ConnectionError("Not connected to database. Call connect() first.")
        return self._connection

    async def _open_snapshot_read_connection(self) -> aiosqlite.Connection:
        """Open a one-shot connection for committed reads during another task's txn."""
        conn = await aiosqlite.connect(self.db_path, timeout=30)
        try:
            await conn.execute("PRAGMA busy_timeout=30000")
            await conn.execute("PRAGMA foreign_keys=ON")
            await conn.execute("PRAGMA query_only=ON")
            conn.row_factory = aiosqlite.Row
            return conn
        except Exception:
            await conn.close()
            raise

    @asynccontextmanager
    async def _read_connection(self) -> AsyncIterator[aiosqlite.Connection]:
        """Return a connection with read-committed semantics for this task.

        The shared aiosqlite connection must be used for normal reads and for
        the transaction owner's reads. A different task reading that shared
        connection while a transaction is open would see connection-local
        uncommitted rows, so it gets a fresh connection and therefore SQLite's
        last committed snapshot.
        """
        conn = self._ensure_connected()
        if (
            self.db_path != ":memory:"
            and self._txn_owner is not None
            and self._txn_owner is not asyncio.current_task()
        ):
            read_conn = await self._open_snapshot_read_connection()
            try:
                yield read_conn
            finally:
                await read_conn.close()
            return

        yield conn
    
    @asynccontextmanager
    async def _write_guard(self) -> AsyncIterator[None]:
        """Hold the connection write lock for one atomic write unit.

        Re-entrant only for the task that owns an open transaction: that task's
        own statements run under the lock it already holds (no deadlock), while
        every other writer — autocommit or a different task — must acquire the
        lock and therefore waits until the in-flight write unit completes.
        """
        if self._txn_owner is not None and self._txn_owner is asyncio.current_task():
            yield
            return
        async with self._write_lock:
            yield

    async def execute(self, query: str, params: Params = ()) -> int:
        """Execute a write query."""
        record_write_query(query)
        conn = self._ensure_connected()
        async with self._write_guard():
            try:
                cursor = await conn.execute(query, params)
                if not self._in_transaction:
                    await conn.commit()
                return cursor.rowcount
            except Exception as e:
                if not self._in_transaction:
                    await conn.rollback()
                raise QueryError(f"Query failed: {e}\nQuery: {query}") from e
            except BaseException:
                # Cancellation (CancelledError is BaseException, not Exception):
                # roll back the partial statement so the next writer to acquire
                # the lock doesn't inherit — and commit — a canceled write, then
                # propagate unchanged.
                if not self._in_transaction:
                    await conn.rollback()
                raise

    async def execute_many(self, query: str, params_list: List[Params]) -> int:
        """Execute query with multiple parameter sets."""
        if not params_list:
            return 0
        record_write_query(query)
        conn = self._ensure_connected()
        async with self._write_guard():
            try:
                cursor = await conn.executemany(query, params_list)
                if not self._in_transaction:
                    await conn.commit()
                return cursor.rowcount
            except Exception as e:
                if not self._in_transaction:
                    await conn.rollback()
                raise QueryError(f"Query failed: {e}\nQuery: {query}") from e
            except BaseException:
                # See execute(): roll back a canceled partial write.
                if not self._in_transaction:
                    await conn.rollback()
                raise

    async def fetch_one(self, query: str, params: Params = ()) -> Optional[Row]:
        """Fetch a single row."""
        record_write_query(query)
        try:
            async with self._read_connection() as conn:
                cursor = await conn.execute(query, params)
                try:
                    row = await cursor.fetchone()
                finally:
                    await cursor.close()
            if row is None:
                return None
            return tuple(row)
        except Exception as e:
            raise QueryError(f"Query failed: {e}\nQuery: {query}") from e
    
    async def fetch_all(self, query: str, params: Params = ()) -> List[Row]:
        """Fetch all rows."""
        record_write_query(query)
        try:
            async with self._read_connection() as conn:
                cursor = await conn.execute(query, params)
                try:
                    rows = await cursor.fetchall()
                finally:
                    await cursor.close()
            return [tuple(row) for row in rows]
        except Exception as e:
            raise QueryError(f"Query failed: {e}\nQuery: {query}") from e
    
    async def fetch_val(self, query: str, params: Params = ()) -> Optional[Any]:
        """Fetch a single value."""
        row = await self.fetch_one(query, params)
        if row is None or len(row) == 0:
            return None
        return row[0]
    
    async def execute_script(self, script: str) -> None:
        """Execute a multi-statement SQL script."""
        record_write_script(script)
        conn = self._ensure_connected()
        async with self._write_guard():
            try:
                await conn.executescript(script)
                if not self._in_transaction:
                    await conn.commit()
            except Exception as e:
                if not self._in_transaction:
                    await conn.rollback()
                raise QueryError(f"Script execution failed: {e}") from e
            except BaseException:
                # See execute(): roll back a canceled partial script.
                if not self._in_transaction:
                    await conn.rollback()
                raise

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        """Transaction context manager."""
        conn = self._ensure_connected()

        if self._in_transaction and self._txn_owner is asyncio.current_task():
            # Nested transaction in the SAME task — just yield (SQLite doesn't
            # support savepoints well). A *different* task starting a
            # transaction falls through and waits on the write lock below.
            yield
            return

        # Hold the write lock for the whole BEGIN..COMMIT/ROLLBACK span so the
        # transaction is one atomic write unit against concurrent writers
        # sharing this connection (#1675).
        async with self._write_lock:
            self._in_transaction = True
            self._txn_owner = asyncio.current_task()
            try:
                await conn.execute("BEGIN")
                yield
                await conn.commit()
            except Exception as e:
                await conn.rollback()
                raise TransactionError(f"Transaction failed: {e}") from e
            except BaseException:
                # Cancellation mid-transaction: roll back so a partial
                # transaction isn't left open for the next writer (which would
                # otherwise commit it), then propagate unchanged.
                await conn.rollback()
                raise
            finally:
                self._in_transaction = False
                self._txn_owner = None
    
    async def table_exists(self, table_name: str) -> bool:
        """Check if a table exists."""
        row = await self.fetch_one(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,)
        )
        return row is not None
