"""
PostgreSQL Database Backend

Implementation of DatabaseBackend using asyncpg.

ADVANCED MODE:
    PostgresBackend is for multi-tenant, high-concurrency, or server deployments.
    For sovereign agents, SQLiteBackend is the recommended default.

    Use cases for PostgresBackend:
        - Multi-tenant SaaS with many agents sharing one database
        - High-concurrency workloads requiring concurrent writers
        - Centralized analytics and admin dashboards
        - Server deployments with connection pooling

    For cloud backup with SQLite: Use SyncService with S3Target or LighthouseTarget
    (see kestrel_sovereign/storage/sync/).

    Constitutional Council Decision (session 9282ed19):
        The council approved SQLite as the DEFAULT while retaining PostgreSQL for
        advanced use cases. See kestrel_sovereign/data/council_sessions/ for the
        full deliberation transcript.
"""

import contextvars
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncIterator, List, Optional, Sequence, Tuple

from .interface import (
    ConnectionError,
    DatabaseBackend,
    Params,
    QueryError,
    Row,
    TransactionError,
)
from .placeholder import sqlite_to_postgres
from .write_audit import record_write_query, record_write_script

logger = logging.getLogger(__name__)

# asyncpg is optional - only required for PostgreSQL usage
try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    asyncpg = None  # type: ignore
    ASYNCPG_AVAILABLE = False


class PostgresBackend(DatabaseBackend):
    """
    PostgreSQL database backend using asyncpg.

    ADVANCED MODE: This backend is for multi-tenant, high-concurrency, or server
    deployments. For sovereign agents, SQLiteBackend is the recommended default.

    Use cases:
        - Multi-tenant SaaS with many agents sharing one database
        - High-concurrency workloads requiring concurrent writers
        - Centralized analytics and admin dashboards
        - Server deployments with connection pooling

    Features:
        - Connection pooling
        - Automatic placeholder conversion (? → $1, $2)
        - Prepared statement caching
        - Transaction support with savepoints
    """

    def __init__(
        self,
        dsn: Optional[str] = None,
        *,
        host: Optional[str] = None,
        port: int = 5432,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        min_pool_size: int = 2,
        max_pool_size: int = 10,
    ):
        """
        Initialize PostgreSQL backend.
        
        Can be initialized with a DSN string or individual parameters.
        
        Args:
            dsn: Connection string (postgresql://user:pass@host:port/db)
            host: Database host
            port: Database port (default 5432)
            database: Database name
            user: Database user
            password: Database password
            min_pool_size: Minimum pool connections
            max_pool_size: Maximum pool connections
        """
        if not ASYNCPG_AVAILABLE:
            raise ImportError(
                "asyncpg is not installed. Install with: pip install asyncpg"
            )
        
        self._dsn = dsn
        self._host = host
        self._port = port
        self._database = database
        self._user = user
        self._password = password
        self._min_pool_size = min_pool_size
        self._max_pool_size = max_pool_size
        
        self._pool: Optional[asyncpg.Pool] = None
        # PER-TASK transaction connection (#1726). Previously a single shared
        # instance attribute, which meant a concurrent task's execute()/fetch()
        # routed onto WHOEVER's transaction was open — cross-contaminating
        # transactions on the very backend built for multi-tenant concurrency. A
        # ContextVar is per-asyncio-task: Task A's open transaction connection is
        # invisible to Task B, so a stray execute() in B uses the pool and a
        # concurrent transaction() in B acquires its own connection.
        self._txn_conn_var: "contextvars.ContextVar[Optional[asyncpg.Connection]]" = (
            contextvars.ContextVar("pg_txn_conn", default=None)
        )
        self._owns_pool = True  # We own pools we create
    
    @classmethod
    def from_pool(cls, pool: "asyncpg.Pool") -> "PostgresBackend":
        """
        Create a PostgresBackend from an existing asyncpg pool.

        This is useful when you want to reuse an existing connection pool
        (e.g., from an app's pg_pool) rather than creating a new one.

        Note: close() will NOT close the pool since we don't own it.

        Args:
            pool: Existing asyncpg.Pool instance

        Returns:
            PostgresBackend wrapping the pool
        """
        instance = cls.__new__(cls)
        instance._dsn = None
        instance._host = None
        instance._port = 5432
        instance._database = None
        instance._user = None
        instance._password = None
        instance._min_pool_size = 2
        instance._max_pool_size = 10
        instance._pool = pool
        instance._txn_conn_var = contextvars.ContextVar("pg_txn_conn", default=None)
        instance._owns_pool = False  # Mark that we don't own the pool
        return instance

    @property
    def backend_type(self) -> str:
        return "postgres"

    @property
    def is_connected(self) -> bool:
        return self._pool is not None
    
    async def connect(self) -> None:
        """Connect to PostgreSQL and create connection pool."""
        if self._pool is not None:
            return
        
        try:
            if self._dsn:
                self._pool = await asyncpg.create_pool(
                    self._dsn,
                    min_size=self._min_pool_size,
                    max_size=self._max_pool_size,
                )
            else:
                self._pool = await asyncpg.create_pool(
                    host=self._host,
                    port=self._port,
                    database=self._database,
                    user=self._user,
                    password=self._password,
                    min_size=self._min_pool_size,
                    max_size=self._max_pool_size,
                )
            
            logger.debug(f"Connected to PostgreSQL pool (size: {self._min_pool_size}-{self._max_pool_size})")
            
        except Exception as e:
            raise ConnectionError(f"Failed to connect to PostgreSQL: {e}") from e
    
    async def close(self) -> None:
        """Close connection pool (only if we own it)."""
        if self._pool is not None:
            if self._owns_pool:
                try:
                    await self._pool.close()
                    logger.debug("Closed PostgreSQL connection pool")
                finally:
                    self._pool = None
            else:
                # We don't own the pool (from from_pool), just release reference
                logger.debug("Released PostgreSQL pool reference (not owned)")
                self._pool = None
    
    def _ensure_connected(self) -> asyncpg.Pool:
        """Ensure we have an active pool."""
        if self._pool is None:
            raise ConnectionError("Not connected to database. Call connect() first.")
        return self._pool
    
    def _convert_query(self, query: str) -> str:
        """Convert SQLite-style ? placeholders to PostgreSQL $N style."""
        converted, _ = sqlite_to_postgres(query)
        return converted

    @staticmethod
    def _strip_tz(params: Params) -> Tuple[Any, ...]:
        """Strip timezone info from datetime params.

        The schema uses TIMESTAMP (naive), not TIMESTAMPTZ. asyncpg
        raises errors when mixing offset-naive and offset-aware datetimes,
        so we strip tzinfo from any aware datetime params.
        """
        return tuple(
            p.replace(tzinfo=None) if isinstance(p, datetime) and p.tzinfo is not None
            else p
            for p in params
        )

    async def execute(self, query: str, params: Params = ()) -> int:
        """Execute a write query."""
        record_write_query(query)
        pool = self._ensure_connected()
        pg_query = self._convert_query(query)
        params = self._strip_tz(params)

        try:
            # Use this task's transaction connection if one is open (#1726).
            txn = self._txn_conn_var.get()
            if txn is not None:
                result = await txn.execute(pg_query, *params)
            else:
                result = await pool.execute(pg_query, *params)
            
            # Parse affected rows from result (e.g., "INSERT 0 1" or "UPDATE 5")
            if result:
                parts = result.split()
                if len(parts) >= 2 and parts[-1].isdigit():
                    return int(parts[-1])
            return 0
            
        except Exception as e:
            raise QueryError(f"Query failed: {e}\nQuery: {pg_query}") from e
    
    async def execute_many(self, query: str, params_list: List[Params]) -> int:
        """Execute query with multiple parameter sets."""
        if not params_list:
            return 0
        record_write_query(query)
        pool = self._ensure_connected()
        pg_query = self._convert_query(query)
        params_list = [self._strip_tz(p) for p in params_list]

        try:
            txn = self._txn_conn_var.get()
            if txn is not None:
                await txn.executemany(pg_query, params_list)
            else:
                async with pool.acquire() as conn:
                    await conn.executemany(pg_query, params_list)
            return len(params_list)  # asyncpg doesn't return affected count
            
        except Exception as e:
            raise QueryError(f"Query failed: {e}\nQuery: {pg_query}") from e
    
    async def fetch_one(self, query: str, params: Params = ()) -> Optional[Row]:
        """Fetch a single row."""
        record_write_query(query)
        pool = self._ensure_connected()
        pg_query = self._convert_query(query)
        params = self._strip_tz(params)

        try:
            txn = self._txn_conn_var.get()
            if txn is not None:
                row = await txn.fetchrow(pg_query, *params)
            else:
                row = await pool.fetchrow(pg_query, *params)
            
            if row is None:
                return None
            return tuple(row.values())
            
        except Exception as e:
            raise QueryError(f"Query failed: {e}\nQuery: {pg_query}") from e
    
    async def fetch_all(self, query: str, params: Params = ()) -> List[Row]:
        """Fetch all rows."""
        record_write_query(query)
        pool = self._ensure_connected()
        pg_query = self._convert_query(query)
        params = self._strip_tz(params)

        try:
            txn = self._txn_conn_var.get()
            if txn is not None:
                rows = await txn.fetch(pg_query, *params)
            else:
                rows = await pool.fetch(pg_query, *params)
            
            return [tuple(row.values()) for row in rows]
            
        except Exception as e:
            raise QueryError(f"Query failed: {e}\nQuery: {pg_query}") from e
    
    async def fetch_val(self, query: str, params: Params = ()) -> Optional[Any]:
        """Fetch a single value."""
        record_write_query(query)
        pool = self._ensure_connected()
        pg_query = self._convert_query(query)
        params = self._strip_tz(params)

        try:
            txn = self._txn_conn_var.get()
            if txn is not None:
                return await txn.fetchval(pg_query, *params)
            else:
                return await pool.fetchval(pg_query, *params)
            
        except Exception as e:
            raise QueryError(f"Query failed: {e}\nQuery: {pg_query}") from e
    
    async def execute_script(self, script: str) -> None:
        """Execute a multi-statement SQL script."""
        record_write_script(script)
        pool = self._ensure_connected()
        
        try:
            txn = self._txn_conn_var.get()
            if txn is not None:
                await txn.execute(script)
            else:
                async with pool.acquire() as conn:
                    await conn.execute(script)
                    
        except Exception as e:
            raise QueryError(f"Script execution failed: {e}") from e
    
    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        """Transaction context manager.

        Nesting is detected PER TASK via the ContextVar (#1726): a nested
        ``transaction()`` within the SAME task reuses that task's connection as a
        savepoint, while a ``transaction()`` in a DIFFERENT concurrent task sees
        no open connection (its ContextVar is the default) and acquires its own —
        so concurrent transactions no longer collide on a shared attribute.
        """
        pool = self._ensure_connected()

        existing = self._txn_conn_var.get()
        if existing is not None:
            # Nested transaction (same task) - use savepoint on this task's conn.
            async with existing.transaction():
                yield
            return

        async with pool.acquire() as conn:
            token = self._txn_conn_var.set(conn)
            try:
                async with conn.transaction():
                    yield
            except Exception as e:
                raise TransactionError(f"Transaction failed: {e}") from e
            finally:
                self._txn_conn_var.reset(token)
    
    async def table_exists(self, table_name: str) -> bool:
        """Check if a table exists."""
        row = await self.fetch_one(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema='public' AND table_name=?",
            (table_name,)
        )
        return row is not None
