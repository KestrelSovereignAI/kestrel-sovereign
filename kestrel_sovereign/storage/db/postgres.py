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

import asyncio
import contextvars
import logging
from collections.abc import Sequence as SequenceABC
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncIterator, List, Optional, Sequence, Tuple

from kestrel_sovereign._async_ownership import await_owned_task

from .interface import (
    ConnectionError,
    DatabaseBackend,
    Params,
    QueryError,
    Row,
    TransactionError,
)
from .placeholder import sqlite_to_postgres
from .timestamp import TimestamptzParameter
from .write_audit import record_write_query, record_write_script

logger = logging.getLogger(__name__)

# Concurrent upserts of the same row (e.g. parallel agent-init writes to the
# composite-PK ``agent_metadata`` rows) can surface a transient
# "tuple concurrently updated" from Postgres even with ON CONFLICT DO UPDATE.
# It clears on retry. We retry only autocommit (non-transaction) writes —
# a single statement inside an explicit transaction can't be replayed in
# isolation once the surrounding txn is poisoned. See kestrel #1805.
_CONCURRENT_UPDATE_MARKER = "tuple concurrently updated"
_CONCURRENT_WRITE_RETRIES = 4
_CONCURRENT_WRITE_BACKOFF_S = 0.02
_ADVISORY_LOCK_POLL_INTERVAL_S = 0.05


def _is_concurrent_update_error(exc: BaseException) -> bool:
    """True for the transient 'tuple concurrently updated' race (#1805)."""
    return _CONCURRENT_UPDATE_MARKER in str(exc).lower()


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
        # Session advisory locks protect scheduler effects for their whole
        # external-work span. Keep a *bounded*, separate pool for those gates:
        # a waiter must not consume the last operational connection needed by
        # an admitted effect's lease renewal/final CAS, and an unbounded direct
        # connection per waiter would merely trade a deadlock for connection
        # exhaustion. Four sessions are enough for distinct-DID concurrency;
        # smaller operational pools keep the same cap.
        self._advisory_max_pool_size = max(1, min(max_pool_size, 4))
        self._advisory_dsn = dsn
        self._advisory_connect_args: tuple[Any, ...] = (dsn,) if dsn else ()
        self._advisory_connect_kwargs: dict[str, Any] = {}
        self._advisory_recipe_available = self._has_advisory_connection_recipe(
            self._advisory_connect_args,
            self._advisory_connect_kwargs,
        )
        self._advisory_pool: Optional[asyncpg.Pool] = None
        self._advisory_pool_lock = asyncio.Lock()
        
        self._pool: Optional[asyncpg.Pool] = None
        # PER-TASK transaction connection (#1726). Previously a single shared
        # instance attribute, which meant a concurrent task's execute()/fetch()
        # routed onto WHOEVER's transaction was open — cross-contaminating
        # transactions on the very backend built for multi-tenant concurrency.
        # The ContextVar stores ``(owner_task, conn)``: a ContextVar is COPIED
        # into child tasks created with asyncio.create_task(), so we must verify
        # the current task IS the owner before treating the connection as ours —
        # otherwise a child task spawned inside a transaction would route onto
        # the parent's uncommitted connection (codex r2).
        self._txn_conn_var: "contextvars.ContextVar" = contextvars.ContextVar(
            "pg_txn_conn", default=None
        )
        self._owns_pool = True  # We own pools we create
    
    @classmethod
    def from_pool(
        cls,
        pool: "asyncpg.Pool",
        *,
        advisory_dsn: Optional[str] = None,
        advisory_connect_kwargs: Optional[dict[str, Any]] = None,
    ) -> "PostgresBackend":
        """
        Create a PostgresBackend from an existing asyncpg pool.

        This is useful when you want to reuse an existing connection pool
        (e.g., from an app's pg_pool) rather than creating a new one.

        Note: close() will NOT close the pool since we don't own it.

        Args:
            pool: Existing asyncpg.Pool instance
            advisory_dsn: Optional explicit DSN for the bounded dedicated
                advisory-lock pool. When omitted, connection construction
                settings are copied from asyncpg's pool construction context.
            advisory_connect_kwargs: Connection options for the dedicated
                advisory-lock pool, such as a non-default ``search_path``.

        Returns:
            PostgresBackend wrapping the pool
        """
        instance = cls.__new__(cls)
        # Preserve the wrapped pool's no-DSN contract for unrelated consumers
        # (for example SQLAlchemy session factories). The scheduler-only
        # dedicated advisory pool has its own explicit connection source.
        instance._dsn = None
        instance._host = None
        instance._port = 5432
        instance._database = None
        instance._user = None
        instance._password = None
        instance._min_pool_size = 2
        instance._max_pool_size = 10
        try:
            operational_max = int(pool.get_max_size())
        except (AttributeError, TypeError, ValueError):
            operational_max = 4
        instance._advisory_max_pool_size = max(1, min(operational_max, 4))
        pool_args, pool_kwargs = cls._advisory_settings_from_pool(pool)
        if advisory_dsn is not None:
            pool_args = (advisory_dsn,)
            # An explicit scheduler connection source must not inherit the
            # wrapped pool's host/password/SSL options. Retain only protocol
            # classes, while callers supply any intended advisory overrides.
            pool_kwargs = {
                key: value
                for key, value in pool_kwargs.items()
                if key in {"connection_class", "record_class"}
            }
        instance._advisory_dsn = advisory_dsn
        instance._advisory_connect_args = pool_args
        instance._advisory_connect_kwargs = {
            **pool_kwargs,
            **dict(advisory_connect_kwargs or {}),
        }
        instance._advisory_recipe_available = cls._has_advisory_connection_recipe(
            instance._advisory_connect_args,
            instance._advisory_connect_kwargs,
        )
        instance._advisory_pool = None
        instance._advisory_pool_lock = asyncio.Lock()
        instance._pool = pool
        instance._txn_conn_var = contextvars.ContextVar("pg_txn_conn", default=None)
        instance._owns_pool = False  # Mark that we don't own the pool
        return instance

    @staticmethod
    def _advisory_settings_from_pool(
        pool: "asyncpg.Pool",
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Copy the independent connection recipe from an asyncpg pool.

        asyncpg deliberately does not offer a public DSN accessor for a live
        pool. It does retain the immutable arguments used to construct that
        pool, which are the only safe pool-only embedding seam for a fresh
        advisory session: borrowing the shared operational pool for a
        long-lived scheduler lock can starve the lease/finalization path.
        Treat malformed third-party doubles as having no derivable recipe so
        the advisory gate still fails closed rather than guessing credentials.
        """

        raw_args = getattr(pool, "_connect_args", ())
        raw_kwargs = getattr(pool, "_connect_kwargs", {})
        if (
            not isinstance(raw_args, SequenceABC)
            or isinstance(raw_args, (str, bytes, bytearray))
            or not isinstance(raw_kwargs, dict)
        ):
            return (), {}
        connect_args = tuple(raw_args)
        kwargs = dict(raw_kwargs)
        connect_factory = getattr(pool, "_connect", None)
        if callable(connect_factory):
            kwargs.setdefault("connect", connect_factory)
        connection_class = getattr(pool, "_connection_class", None)
        if connection_class is not None:
            kwargs.setdefault("connection_class", connection_class)
        record_class = getattr(pool, "_record_class", None)
        if record_class is not None:
            kwargs.setdefault("record_class", record_class)
        return connect_args, kwargs

    @staticmethod
    def _has_advisory_connection_recipe(
        connect_args: tuple[Any, ...],
        connect_kwargs: dict[str, Any],
    ) -> bool:
        """Whether copied asyncpg settings can create an isolated session."""

        # ``Pool`` stores ``connection.connect`` as ``_connect`` when callers
        # do not provide ``connect=``.  That default needs a DSN or complete
        # keyword credentials and must not authorize a fresh connection from
        # ambient configuration.  A distinct factory, however, is itself the
        # caller-provided connection recipe (and may keep its credentials in a
        # closure), so preserve it for the dedicated advisory pool.
        connect_factory = connect_kwargs.get("connect")
        if (
            callable(connect_factory)
            and asyncpg is not None
            and connect_factory is not asyncpg.connection.connect
        ):
            return True
        if connect_args and connect_args[0] is not None:
            return isinstance(connect_args[0], str) and bool(connect_args[0])
        host = connect_kwargs.get("host")
        database = connect_kwargs.get("database")
        if not isinstance(database, str) or not database:
            return False
        if isinstance(host, str):
            return bool(host)
        # asyncpg accepts list/tuple hosts for ordered multi-host failover.
        # Limit copied wrapped-pool recipes to those concrete containers so an
        # arbitrary sequence or ambient asyncpg configuration cannot authorize
        # a dedicated advisory connection.
        return (
            isinstance(host, (list, tuple))
            and bool(host)
            and all(isinstance(candidate, str) and bool(candidate) for candidate in host)
        )

    def _current_txn_conn(self):
        """This task's open transaction connection, or None (#1726).

        The ContextVar holds ``(owner_task, conn)``. A child task inherits a COPY
        of the parent's context, so the entry may belong to a PARENT task — we
        return the connection ONLY when the current task is the owner, so a child
        spawned inside a transaction does NOT route onto the parent's connection.
        """
        entry = self._txn_conn_var.get()
        if entry is None:
            return None
        owner, conn = entry
        if owner is asyncio.current_task():
            return conn
        return None

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
        """Close every owned pool before surfacing a close failure or cancel.

        The advisory pool is a separate owned resource even when the primary
        pool was supplied by ``from_pool``.  Keep each handle until its own
        close task reaches a successful terminal state, so a failure or an
        owned-task cancellation remains retryable instead of stranding a
        dedicated advisory session behind a cleared reference.
        """

        pending_cancellation: asyncio.CancelledError | None = None
        failures: list[BaseException] = []

        advisory_pool = self._advisory_pool
        if advisory_pool is not None:
            outcome = await await_owned_task(
                asyncio.create_task(advisory_pool.close()),
                pending_cancellation,
            )
            pending_cancellation = outcome.cancellation
            if outcome.error is None:
                if self._advisory_pool is advisory_pool:
                    self._advisory_pool = None
            else:
                failures.append(outcome.error)

        primary_pool = self._pool
        if primary_pool is not None:
            if self._owns_pool:
                outcome = await await_owned_task(
                    asyncio.create_task(primary_pool.close()),
                    pending_cancellation,
                )
                pending_cancellation = outcome.cancellation
                if outcome.error is None:
                    if self._pool is primary_pool:
                        self._pool = None
                    logger.debug("Closed PostgreSQL connection pool")
                else:
                    failures.append(outcome.error)
            else:
                # We don't own the pool (from from_pool), just release our
                # reference even when advisory cleanup failed or was cancelled.
                logger.debug("Released PostgreSQL pool reference (not owned)")
                if self._pool is primary_pool:
                    self._pool = None

        if pending_cancellation is not None:
            if failures:
                pending_cancellation.add_note(
                    "PostgreSQL pool cleanup also failed: "
                    + "; ".join(str(error) for error in failures)
                )
                raise pending_cancellation from failures[0]
            raise pending_cancellation
        if failures:
            if len(failures) > 1:
                failures[0].add_note(
                    "Additional PostgreSQL pool cleanup failures: "
                    + "; ".join(str(error) for error in failures[1:])
                )
            raise failures[0]
    
    def _ensure_connected(self) -> asyncpg.Pool:
        """Ensure we have an active pool."""
        if self._pool is None:
            raise ConnectionError("Not connected to database. Call connect() first.")
        return self._pool

    async def _ensure_advisory_pool(self) -> "asyncpg.Pool":
        """Return the bounded pool used only for long advisory-lock gates."""

        # Retain the normal backend lifecycle check: an advisory pool is not a
        # hidden replacement for a closed primary database backend.
        self._ensure_connected()
        if self._advisory_pool is not None:
            return self._advisory_pool
        async with self._advisory_pool_lock:
            if self._advisory_pool is not None:
                return self._advisory_pool
            kwargs = dict(self._advisory_connect_kwargs)
            try:
                if self._advisory_recipe_available:
                    pool = await asyncpg.create_pool(
                        *self._advisory_connect_args,
                        min_size=0,
                        max_size=self._advisory_max_pool_size,
                        **kwargs,
                    )
                else:
                    if not self._host or not self._database:
                        raise ConnectionError(
                            "Dedicated PostgreSQL advisory locks require connection "
                            "parameters, advisory_dsn, or a valid wrapped-pool recipe"
                        )
                    pool = await asyncpg.create_pool(
                        host=self._host,
                        port=self._port,
                        database=self._database,
                        user=self._user,
                        password=self._password,
                        min_size=0,
                        max_size=self._advisory_max_pool_size,
                        **kwargs,
                    )
            except ConnectionError:
                raise
            except Exception as exc:
                raise ConnectionError(
                    f"Failed to create dedicated PostgreSQL advisory-lock pool: {exc}"
                ) from exc
            self._advisory_pool = pool
            return pool
    
    def _convert_query(self, query: str) -> str:
        """Convert SQLite-style ? placeholders to PostgreSQL $N style."""
        converted, _ = sqlite_to_postgres(query)
        return converted

    @staticmethod
    def _strip_tz(params: Params) -> Tuple[Any, ...]:
        """Adapt timestamp parameters without changing their SQL contract.

        Legacy callers bind to genuinely timezone-naive ``TIMESTAMP`` columns,
        where asyncpg requires a naive datetime. Unified stores that declare
        ``TIMESTAMPTZ`` use :class:`TimestamptzParameter` explicitly; preserve
        an aware value there (normalizing to UTC) so its absolute instant does
        not become process-local wall time before asyncpg sees it.
        """
        adapted: list[Any] = []
        for param in params:
            if isinstance(param, TimestamptzParameter):
                value = param.value
                if value.tzinfo is not None and value.utcoffset() is not None:
                    adapted.append(value.astimezone(timezone.utc))
                else:
                    adapted.append(value)
            elif isinstance(param, datetime) and param.tzinfo is not None:
                adapted.append(param.replace(tzinfo=None))
            else:
                adapted.append(param)
        return tuple(adapted)

    async def execute(self, query: str, params: Params = ()) -> int:
        """Execute a write query."""
        record_write_query(query)
        pool = self._ensure_connected()
        pg_query = self._convert_query(query)
        params = self._strip_tz(params)

        # Use this task's transaction connection if one is open (#1726).
        txn = self._current_txn_conn()
        attempt = 0
        while True:
            try:
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
                # Retry the transient concurrent-upsert race on the autocommit
                # path only (a poisoned txn can't be salvaged statement-wise).
                if (
                    txn is None
                    and _is_concurrent_update_error(e)
                    and attempt < _CONCURRENT_WRITE_RETRIES
                ):
                    attempt += 1
                    logger.warning(
                        "Concurrent-update race on write (attempt %d/%d), retrying: %s",
                        attempt, _CONCURRENT_WRITE_RETRIES, e,
                    )
                    await asyncio.sleep(_CONCURRENT_WRITE_BACKOFF_S * attempt)
                    continue
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
            txn = self._current_txn_conn()
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
            txn = self._current_txn_conn()
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
            txn = self._current_txn_conn()
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
            txn = self._current_txn_conn()
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
            txn = self._current_txn_conn()
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

        existing = self._current_txn_conn()
        if existing is not None:
            # Nested transaction (same task) - use savepoint on this task's conn.
            async with existing.transaction():
                yield
            return

        async with pool.acquire() as conn:
            token = self._txn_conn_var.set((asyncio.current_task(), conn))
            try:
                async with conn.transaction():
                    yield
            except Exception as e:
                raise TransactionError(f"Transaction failed: {e}") from e
            finally:
                self._txn_conn_var.reset(token)

    @asynccontextmanager
    async def advisory_locks(
        self, keys: Sequence[Tuple[int, int]], *, shared: bool = False
    ) -> AsyncIterator[None]:
        """Hold multiple session advisory locks on one bounded-pool connection.

        A multi-tenant bootstrap can need to drain effects for more than one
        DID. Acquiring those locks on separate connections risks needless
        connection growth, while using the ordinary query pool can deadlock an
        admitted effect's lease renewal or target writes. Keep every named lock
        on one bounded advisory-pool session instead. ``shared=True`` admits
        concurrent readers and is paired with an exclusive writer using the
        same keys.

        Callers must provide keys in their established global order when more
        than one process can acquire an overlapping set.  Each pair uses
        PostgreSQL's signed two-int advisory-lock form.
        """
        normalized_keys = tuple(keys)
        for namespace, key in normalized_keys:
            if not isinstance(namespace, int) or not isinstance(key, int):
                raise TypeError("PostgreSQL advisory lock keys must be integers")
        if not normalized_keys:
            # A dynamic hosted scheduler can legitimately have no tenants
            # during bootstrap. Do not pin a spare pool connection just to
            # represent that empty exclusion set.
            yield
            return
        lock_function = "pg_advisory_lock_shared" if shared else "pg_advisory_lock"
        unlock_function = (
            "pg_advisory_unlock_shared" if shared else "pg_advisory_unlock"
        )
        advisory_pool = await self._ensure_advisory_pool()
        # Pool acquisition queues outside the operational query pool.  A
        # cancellation while waiting therefore cannot strand an operational
        # connection that an admitted effect still needs.
        async with advisory_pool.acquire() as conn:
            acquired: list[Tuple[int, int]] = []
            try:
                for namespace, key in normalized_keys:
                    await conn.execute(
                        f"SELECT {lock_function}($1, $2)", namespace, key
                    )
                    acquired.append((namespace, key))
                yield
            except BaseException:
                # A cancellation can race PostgreSQL granting a blocking lock
                # after asyncpg has sent the cancellation request. Terminating
                # this dedicated connection is the only unconditional release
                # in that race; the bounded pool discards it and never returns
                # a possibly locked session to another scheduler operation.
                conn.terminate()
                raise
            else:
                try:
                    # ``pg_advisory_unlock`` must run on the same session that
                    # acquired the lock. The normal path returns the clean,
                    # unlocked connection to the bounded advisory pool.
                    for namespace, key in reversed(acquired):
                        await conn.execute(
                            f"SELECT {unlock_function}($1, $2)", namespace, key
                        )
                except BaseException:
                    conn.terminate()
                    raise

    @asynccontextmanager
    async def polled_advisory_lock(
        self, key: Tuple[int, int]
    ) -> AsyncIterator[None]:
        """Hold one session lock without leaving a blocked statement snapshot.

        ``CREATE INDEX CONCURRENTLY`` waits for transactions with older
        snapshots. A second initializer blocked inside ``pg_advisory_lock`` is
        itself such a transaction, which forms a cycle with the first
        initializer's concurrent build. Polling ``pg_try_advisory_lock`` lets
        every unsuccessful statement finish before sleeping in Python, so a
        waiter owns no virtual XID or snapshot while the winner builds.
        """

        namespace, lock_key = key
        if not isinstance(namespace, int) or not isinstance(lock_key, int):
            raise TypeError("PostgreSQL advisory lock keys must be integers")
        advisory_pool = await self._ensure_advisory_pool()
        async with advisory_pool.acquire() as conn:
            acquired = False
            try:
                while not acquired:
                    acquired = bool(
                        await conn.fetchval(
                            "SELECT pg_try_advisory_lock($1, $2)",
                            namespace,
                            lock_key,
                        )
                    )
                    if not acquired:
                        await asyncio.sleep(_ADVISORY_LOCK_POLL_INTERVAL_S)
                yield
            except BaseException:
                conn.terminate()
                raise
            else:
                try:
                    unlocked = await conn.fetchval(
                        "SELECT pg_advisory_unlock($1, $2)",
                        namespace,
                        lock_key,
                    )
                    if not unlocked:
                        raise RuntimeError(
                            "PostgreSQL polled advisory lock disappeared"
                        )
                except BaseException:
                    conn.terminate()
                    raise
    
    async def table_exists(self, table_name: str) -> bool:
        """Check whether an unqualified relation resolves on this connection.

        ``to_regclass`` follows PostgreSQL's active ``search_path``.  That
        preserves the normal production ``public`` behavior while allowing a
        transaction-local schema to be inspected on the connection that owns
        it, rather than accidentally observing an unrelated public table.
        """
        return bool(await self.fetch_val(
            "SELECT to_regclass(?) IS NOT NULL",
            (table_name,),
        ))
