"""
SQLite Database Backend

Implementation of DatabaseBackend using aiosqlite.
"""

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
    
    async def execute(self, query: str, params: Params = ()) -> int:
        """Execute a write query."""
        conn = self._ensure_connected()
        try:
            cursor = await conn.execute(query, params)
            if not self._in_transaction:
                await conn.commit()
            return cursor.rowcount
        except Exception as e:
            if not self._in_transaction:
                await conn.rollback()
            raise QueryError(f"Query failed: {e}\nQuery: {query}") from e
    
    async def execute_many(self, query: str, params_list: List[Params]) -> int:
        """Execute query with multiple parameter sets."""
        conn = self._ensure_connected()
        try:
            cursor = await conn.executemany(query, params_list)
            if not self._in_transaction:
                await conn.commit()
            return cursor.rowcount
        except Exception as e:
            if not self._in_transaction:
                await conn.rollback()
            raise QueryError(f"Query failed: {e}\nQuery: {query}") from e
    
    async def fetch_one(self, query: str, params: Params = ()) -> Optional[Row]:
        """Fetch a single row."""
        conn = self._ensure_connected()
        try:
            cursor = await conn.execute(query, params)
            row = await cursor.fetchone()
            if row is None:
                return None
            return tuple(row)
        except Exception as e:
            raise QueryError(f"Query failed: {e}\nQuery: {query}") from e
    
    async def fetch_all(self, query: str, params: Params = ()) -> List[Row]:
        """Fetch all rows."""
        conn = self._ensure_connected()
        try:
            cursor = await conn.execute(query, params)
            rows = await cursor.fetchall()
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
        conn = self._ensure_connected()
        try:
            await conn.executescript(script)
            await conn.commit()
        except Exception as e:
            await conn.rollback()
            raise QueryError(f"Script execution failed: {e}") from e
    
    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        """Transaction context manager."""
        conn = self._ensure_connected()
        
        if self._in_transaction:
            # Nested transaction - just yield (SQLite doesn't support savepoints well)
            yield
            return
        
        self._in_transaction = True
        try:
            await conn.execute("BEGIN")
            yield
            await conn.commit()
        except Exception as e:
            await conn.rollback()
            raise TransactionError(f"Transaction failed: {e}") from e
        finally:
            self._in_transaction = False
    
    async def table_exists(self, table_name: str) -> bool:
        """Check if a table exists."""
        row = await self.fetch_one(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,)
        )
        return row is not None
