"""
SQLite Database Backend

Implementation of DatabaseBackend using aiosqlite.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, List, Literal, Optional, overload

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


# ``aiosqlite.Connection.close()`` acknowledges its stop sentinel from the
# worker before that thread has necessarily returned.  Do not let the caller
# tear down its event loop in that narrow interval: wait for the worker itself
# and fail explicitly if it remains alive beyond this bounded window.
AIOSQLITE_WORKER_SHUTDOWN_TIMEOUT_S = max(
    float(os.environ.get("KESTREL_AIOSQLITE_WORKER_SHUTDOWN_TIMEOUT_S", "1.0")),
    0.01,
)
_AIOSQLITE_WORKER_SHUTDOWN_POLL_S = 0.01


class _CancelledWriteDrainDeadlineExceeded(RuntimeError):
    """A retained rollback is still running after a later writer's budget."""


def _minimum_close_timeout_s() -> float:
    """Return the outer-guard budget for one aiosqlite connection close.

    The budget covers both ``aiosqlite.Connection.close()`` and final worker
    termination. Reserve one poll interval so an outer guard cannot win a
    scheduling tie with the internal lifecycle deadline.
    """
    return AIOSQLITE_WORKER_SHUTDOWN_TIMEOUT_S + _AIOSQLITE_WORKER_SHUTDOWN_POLL_S


async def _wait_for_aiosqlite_worker_shutdown(
    connection: aiosqlite.Connection,
    *,
    timeout_s: float = AIOSQLITE_WORKER_SHUTDOWN_TIMEOUT_S,
) -> None:
    """Wait until ``connection``'s worker has actually terminated.

    aiosqlite 0.22 keeps the worker on ``_thread``; older supported releases
    subclassed ``Thread`` directly.  Its public close awaitable confirms that
    the stop sentinel ran, which can precede the worker's final return by one
    scheduling turn.  The bounded wait closes that lifecycle gap without
    changing who owns the connection.
    """
    worker = getattr(connection, "_thread", connection)
    is_alive = getattr(worker, "is_alive", None)
    if not callable(is_alive) or not is_alive():
        return

    try:
        async with asyncio.timeout(timeout_s):
            while is_alive():
                await asyncio.sleep(_AIOSQLITE_WORKER_SHUTDOWN_POLL_S)
    except TimeoutError as exc:
        # Avoid a false failure if the worker exits as the timeout fires.
        if not is_alive():
            return
        raise ConnectionError(
            "SQLite worker did not terminate within "
            f"{timeout_s:.2f}s after close"
        ) from exc


async def _close_aiosqlite_connection(connection: aiosqlite.Connection) -> None:
    """Close an owned aiosqlite connection through its full lifecycle.

    Every connection this backend opens owns an aiosqlite worker.  Keeping the
    close and worker-termination wait together prevents a short-lived backup
    or snapshot connection from bypassing the shutdown contract.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _minimum_close_timeout_s()
    close_task = loop.create_task(connection.close())
    pending_cancellation: Optional[asyncio.CancelledError] = None

    try:
        remaining = max(0.0, deadline - loop.time())
        done, _ = await asyncio.wait({close_task}, timeout=remaining)
    except asyncio.CancelledError as exc:
        # Keep the close in its own task: cancelling aiosqlite's public close
        # awaitable can move it into a ``finally`` wait behind the same blocked
        # worker.  Preserve caller cancellation and finish retiring the owned
        # connection inside the original lifecycle budget.
        pending_cancellation = exc
        remaining = max(0.0, deadline - loop.time())
        done, _ = await asyncio.wait({close_task}, timeout=remaining)

    if not done:
        try:
            await connection.interrupt()
        except ValueError as exc:
            if "no active connection" not in str(exc):
                raise
        close_task.cancel()
        raise ConnectionError(
            "SQLite connection close did not complete within "
            f"{_minimum_close_timeout_s():.2f}s"
        )

    close_error: Optional[BaseException] = None
    try:
        close_task.result()
    except BaseException as exc:
        close_error = exc

    remaining = max(0.0, deadline - loop.time())
    try:
        await _wait_for_aiosqlite_worker_shutdown(
            connection,
            timeout_s=remaining,
        )
    except asyncio.CancelledError as exc:
        pending_cancellation = pending_cancellation or exc
        remaining = max(0.0, deadline - loop.time())
        await _wait_for_aiosqlite_worker_shutdown(
            connection,
            timeout_s=remaining,
        )

    if pending_cancellation is not None:
        raise pending_cancellation
    if close_error is not None:
        raise close_error


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
        # Serializes operation *units* on the single shared connection. aiosqlite
        # serializes individual operations, but NOT the execute->commit/rollback
        # pair: without this, two concurrent autocommit writers share one
        # connection-scoped transaction, so one writer's failed rollback() can
        # discard the other's uncommitted write (#1675). The lock makes each
        # autocommit statement and each explicit transaction an atomic write
        # unit. Shared-connection reads also hold it through cursor cleanup so a
        # cancellation drain is queued before another operation can overtake it.
        # Re-entrant for the task that owns an open transaction (its own
        # statements and reads must not deadlock on the lock it already holds).
        self._write_lock = asyncio.Lock()
        self._txn_owner: Optional["asyncio.Task"] = None
        # aiosqlite cannot cancel work already running in its worker thread.
        # When its caller is cancelled, transfer the queued rollback to a
        # retained task and fence later writers on that drain.  This lets an
        # outer deadline release task-owned resources promptly without
        # allowing another writer to commit the abandoned statement (#2907).
        self._cancelled_write_drain: Optional[asyncio.Task[None]] = None
        self._cancelled_write_drain_error: Optional[Exception] = None
        self._closing = False
    
    @property
    def backend_type(self) -> str:
        return "sqlite"
    
    @property
    def is_connected(self) -> bool:
        return self._connection is not None

    @property
    def write_connection_unavailable(self) -> bool:
        """Whether cancellation cleanup currently fences later writes."""
        return (
            self._cancelled_write_drain is not None
            or self._cancelled_write_drain_error is not None
        )

    @property
    def write_connection_requires_reconnect(self) -> bool:
        """Whether failed cleanup keeps writes fenced until reconnect/close."""
        error = self._cancelled_write_drain_error
        return error is not None and not isinstance(
            error, _CancelledWriteDrainDeadlineExceeded
        )

    @property
    def minimum_close_timeout_s(self) -> float:
        """Minimum budget an outer shutdown guard must reserve for ``close``.

        This is intentionally an optional backend extension rather than a
        change to the SDK's generic backend interface: non-SQLite backends do
        not own an aiosqlite worker and therefore have no equivalent lifecycle
        requirement.  A retained cancellation drain and the subsequent
        connection close can each consume one worker-shutdown window.  Keep
        this primary-backend reservation separate from
        :func:`_minimum_close_timeout_s`, which is also used by SQLAlchemy
        factories that have no cancellation-drain phase.
        """
        return (
            2 * AIOSQLITE_WORKER_SHUTDOWN_TIMEOUT_S
            + _AIOSQLITE_WORKER_SHUTDOWN_POLL_S
        )
    
    async def connect(self) -> None:
        """Connect to SQLite database."""
        if self._connection is not None:
            if self._cancelled_write_drain_error is None:
                return
            # A failed cancellation cleanup makes the shared connection
            # untrustworthy. ``connect`` is also the explicit recovery surface:
            # retire that connection fully before opening its replacement.
            await self.close()
        
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

            # A newly opened connection cannot contain work abandoned on the
            # previous connection.  Do not carry a failed cleanup latch across
            # a successful reconnect.
            self._cancelled_write_drain = None
            self._cancelled_write_drain_error = None
            self._closing = False
            
            logger.debug(f"Connected to SQLite: {self.db_path}")
            
        except Exception as e:
            raise ConnectionError(f"Failed to connect to SQLite: {e}") from e
    
    async def close(self) -> None:
        """Close database connection and wait for background thread to stop."""
        conn = self._connection
        if conn is None:
            self._cancelled_write_drain = None
            self._cancelled_write_drain_error = None
            return

        pending_cancellation: Optional[asyncio.CancelledError] = None
        # Fence new cancellation handoffs before sampling the retained drain.
        # A write cancelled from this point onward is retired by connection
        # close itself, so spawning a rollback task would orphan it after the
        # final state reset below.
        self._closing = True
        try:
            drain = self._cancelled_write_drain
            # Cancelling the Python rollback task does not stop a SQLite VM
            # already executing in aiosqlite's worker.  Interrupt the actual
            # connection before waiting for the retained drain so rollback and
            # close can both fit inside the advertised shutdown reservation.
            # Do not interrupt an idle connection: SQLite then leaves its WAL
            # uncheckpointed on close even though the worker terminates.
            if drain is not None or self._write_lock.locked():
                await conn.interrupt()
            if drain is not None and drain is not asyncio.current_task():
                try:
                    async with asyncio.timeout(
                        AIOSQLITE_WORKER_SHUTDOWN_TIMEOUT_S
                    ):
                        await asyncio.shield(drain)
                except (TimeoutError, asyncio.CancelledError) as exc:
                    # The connection close remains responsible for the queued
                    # SQLite work.  Retire the Python task so loop teardown
                    # cannot strand a backend-owned cleanup task.
                    if isinstance(exc, asyncio.CancelledError):
                        pending_cancellation = exc
                    drain.cancel()
                    try:
                        await drain
                    except asyncio.CancelledError as drain_cancelled:
                        # Usually this is the cancellation just sent to the
                        # drain. If the close caller itself is cancelled while
                        # awaiting a stubborn drain, preserve that cancellation
                        # and deliver it after the connection is retired.
                        caller = asyncio.current_task()
                        if caller is not None and caller.cancelling():
                            pending_cancellation = (
                                pending_cancellation or drain_cancelled
                            )

            await _close_aiosqlite_connection(conn)
            logger.debug(f"Closed SQLite connection: {self.db_path}")
        finally:
            self._connection = None
            self._cancelled_write_drain = None
            self._cancelled_write_drain_error = None
            self._closing = False

        if pending_cancellation is not None:
            raise pending_cancellation
    
    def _ensure_connected(self) -> aiosqlite.Connection:
        """Ensure we have an active connection."""
        if self._connection is None:
            raise ConnectionError("Not connected to database. Call connect() first.")
        return self._connection

    async def backup_to(self, dest_path: str) -> None:
        """Copy the live database to ``dest_path`` using SQLite's online backup API.

        Produces a transactionally-consistent snapshot of the running database
        without ever closing the shared connection, so concurrent reads/writes
        keep working. The copy runs in aiosqlite's background thread (not on the
        event loop). We hold the write lock so the snapshot is taken between
        atomic write units rather than mid-write.
        """
        if self.db_path == ":memory:":
            raise ValueError("Cannot back up an in-memory database")
        conn = self._ensure_connected()
        async with self._write_guard():
            dest = await aiosqlite.connect(dest_path, timeout=30)
            try:
                await conn.backup(dest)
            finally:
                await _close_aiosqlite_connection(dest)

    async def _open_snapshot_read_connection(self) -> aiosqlite.Connection:
        """Open a one-shot connection for committed reads during another task's txn."""
        conn = await aiosqlite.connect(self.db_path, timeout=30)
        try:
            await conn.execute("PRAGMA busy_timeout=30000")
            await conn.execute("PRAGMA foreign_keys=ON")
            await conn.execute("PRAGMA query_only=ON")
            conn.row_factory = aiosqlite.Row
            return conn
        except BaseException:
            await _close_aiosqlite_connection(conn)
            raise

    @asynccontextmanager
    async def _read_connection(
        self,
        *,
        diagnostic: bool = False,
    ) -> AsyncIterator[aiosqlite.Connection]:
        """Return a connection with read-committed semantics for this task.

        The shared aiosqlite connection must be used for normal reads and for
        the transaction owner's reads. A different task reading that shared
        connection while a transaction is open would see connection-local
        uncommitted rows, so it gets a fresh connection and therefore SQLite's
        last committed snapshot.
        """
        conn = self._ensure_connected()
        drain = self._cancelled_write_drain
        cleanup_failed = self._cancelled_write_drain_error is not None
        sibling_transaction = (
            self._txn_owner is not None
            and self._txn_owner is not asyncio.current_task()
        )
        diagnostic_bypass = diagnostic and (
            drain is not None or cleanup_failed
        )
        if self.db_path != ":memory:" and (
            sibling_transaction or diagnostic_bypass
        ):
            # A different task must not see another task's uncommitted
            # transaction.  Separately, explicitly diagnostic reads may inspect
            # committed state while cancellation cleanup keeps the application
            # path fenced.  Ordinary reads never take that second bypass:
            # decisions that feed later writes must observe the post-drain state.
            read_conn = await self._open_snapshot_read_connection()
            try:
                yield read_conn
            finally:
                await _close_aiosqlite_connection(read_conn)
            return

        # In-memory databases have no independent committed snapshot. Ordinary
        # file-backed reads also wait here when cleanup is retained so their
        # application decisions cannot be made from a stale snapshot.
        await self._wait_for_cancelled_write_drain()
        cleanup_failed = self._cancelled_write_drain_error is not None
        if cleanup_failed:
            self._raise_cancelled_write_drain_error()

        yield conn
    
    @asynccontextmanager
    async def _write_guard(self) -> AsyncIterator[None]:
        """Hold the shared-connection lock for one atomic operation unit.

        Re-entrant only for the task that owns an open transaction: that task's
        own statements run under the lock it already holds (no deadlock), while
        every other shared-connection operation must acquire the lock and wait
        until the in-flight unit completes. File-backed sibling reads may use
        the independent snapshot path instead.
        """
        if self._txn_owner is not None and self._txn_owner is asyncio.current_task():
            yield
            return

        # A writer may already be queued on ``_write_lock`` when a cancelled
        # owner installs the drain fence.  Check both before and after lock
        # acquisition: the first check avoids occupying the lock for a long
        # drain, and the second closes the handoff race.  asyncio.Lock is fair,
        # so releasing and retrying does not starve earlier waiters.
        while True:
            await self._wait_for_cancelled_write_drain()
            async with self._write_lock:
                if self._cancelled_write_drain is not None:
                    continue
                self._raise_cancelled_write_drain_error()
                yield
                return

    def _raise_cancelled_write_drain_error(self) -> None:
        """Fail closed after a detached rollback could not restore safety."""
        error = self._cancelled_write_drain_error
        if error is not None:
            if isinstance(error, _CancelledWriteDrainDeadlineExceeded):
                raise ConnectionError(
                    "SQLite write connection is unavailable because cancellation "
                    "cleanup is still pending past its deadline"
                ) from error
            raise ConnectionError(
                "SQLite write connection is unavailable because cancellation "
                "cleanup failed"
            ) from error

    async def _wait_for_cancelled_write_drain(self) -> None:
        """Wait for an earlier cancelled write's rollback without owning it.

        A completed drain error is interpreted by the caller: writers fail
        closed, while file-backed readers switch to a fresh query-only
        connection so health and diagnostics remain available.
        """
        drain = self._cancelled_write_drain
        if drain is not None and drain is not asyncio.current_task():
            # A caller cancelled while waiting must not cancel the backend-owned
            # cleanup task.  The retained task remains the sole drain owner.
            try:
                async with asyncio.timeout(
                    AIOSQLITE_WORKER_SHUTDOWN_TIMEOUT_S
                ):
                    await asyncio.shield(drain)
            except TimeoutError:
                # A later writer must never inherit the unbounded worker wait
                # that the cancelled owner escaped. Keep the backend-owned
                # drain alive for close/recovery, but fail the shared write
                # connection closed until that rollback succeeds or reconnect.
                if self._cancelled_write_drain_error is None:
                    self._cancelled_write_drain_error = (
                        _CancelledWriteDrainDeadlineExceeded(
                            "SQLite cancellation cleanup exceeded its deadline"
                        )
                    )
                # The rollback may have completed successfully in the same
                # event-loop turn that delivered our deadline. Its task then
                # had no transient latch to clear, so reconcile that race here.
                if (
                    drain.done()
                    and isinstance(
                        self._cancelled_write_drain_error,
                        _CancelledWriteDrainDeadlineExceeded,
                    )
                ):
                    self._cancelled_write_drain_error = None
                self._raise_cancelled_write_drain_error()
            except asyncio.CancelledError:
                caller = asyncio.current_task()
                if caller is not None and caller.cancelling():
                    raise
                # Backend close (or another lifecycle owner) cancelled the
                # drain, not this caller. Convert that backend failure into the
                # same explicit write-safety latch rather than injecting a
                # spurious cancellation into an unrelated request.
                if self._cancelled_write_drain_error is None or isinstance(
                    self._cancelled_write_drain_error,
                    _CancelledWriteDrainDeadlineExceeded,
                ):
                    self._cancelled_write_drain_error = RuntimeError(
                        "SQLite cancellation cleanup task was cancelled"
                    )
                self._raise_cancelled_write_drain_error()

        self._raise_cancelled_write_drain_error()

    def _handoff_cancelled_write(
        self, conn: aiosqlite.Connection
    ) -> None:
        """Transfer rollback to the backend and return without awaiting it.

        The aiosqlite worker serializes its queue.  If the cancelled statement
        is blocked inside that worker, directly awaiting ``rollback()`` here
        would put the caller's timeout behind the same statement and defeat the
        deadline.  The background drain deliberately does not acquire
        ``_write_lock``; ``_write_guard`` fences all later writers on the
        retained task instead.
        """
        # ``close`` owns every queued operation after it fences handoffs. A
        # stale/cancelled caller must not install a task against a connection
        # that is already closing or no longer belongs to this backend.
        if self._closing or self._connection is not conn:
            return

        drain = self._cancelled_write_drain
        if drain is not None and not drain.done():
            # The shared-connection lock permits only one operation unit or
            # transaction to reach this path, so a second live drain signals an
            # invariant bug.
            self._cancelled_write_drain_error = RuntimeError(
                "overlapping SQLite cancellation drains"
            )
            logger.error("Overlapping SQLite cancellation drains detected")
            return

        self._cancelled_write_drain_error = None
        self._cancelled_write_drain = asyncio.create_task(
            self._drain_cancelled_write(conn),
            name=f"sqlite-cancelled-write-drain:{self.db_path}",
        )

    async def _drain_cancelled_write(
        self, conn: aiosqlite.Connection
    ) -> None:
        """Roll back one cancelled worker operation and retire its fence."""
        this_task = asyncio.current_task()
        try:
            await conn.rollback()
        except Exception as exc:
            self._cancelled_write_drain_error = exc
            logger.exception(
                "SQLite cancellation cleanup failed for %s", self.db_path
            )
        else:
            # The deadline belongs to each waiting writer, not to the
            # backend-owned rollback. A rollback that eventually succeeds
            # restores the shared connection and must remove only that
            # transient marker; genuine cleanup failures remain latched.
            if isinstance(
                self._cancelled_write_drain_error,
                _CancelledWriteDrainDeadlineExceeded,
            ):
                self._cancelled_write_drain_error = None
        finally:
            if self._cancelled_write_drain is this_task:
                self._cancelled_write_drain = None

    async def _rollback_after_failure(
        self, conn: aiosqlite.Connection
    ) -> None:
        """Rollback inline, handing off if cancellation reaches cleanup."""
        try:
            await conn.rollback()
        except asyncio.CancelledError:
            self._handoff_cancelled_write(conn)
            raise

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
            except asyncio.CancelledError:
                if not self._in_transaction:
                    self._handoff_cancelled_write(conn)
                raise
            except Exception as e:
                if not self._in_transaction:
                    await self._rollback_after_failure(conn)
                raise QueryError(f"Query failed: {e}\nQuery: {query}") from e
            except BaseException:
                if not self._in_transaction:
                    await self._rollback_after_failure(conn)
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
            except asyncio.CancelledError:
                if not self._in_transaction:
                    self._handoff_cancelled_write(conn)
                raise
            except Exception as e:
                if not self._in_transaction:
                    await self._rollback_after_failure(conn)
                raise QueryError(f"Query failed: {e}\nQuery: {query}") from e
            except BaseException:
                if not self._in_transaction:
                    await self._rollback_after_failure(conn)
                raise

    @overload
    async def _fetch_on_connection(
        self,
        conn: aiosqlite.Connection,
        query: str,
        params: Params,
        *,
        one: Literal[True],
    ) -> Optional[Row]: ...

    @overload
    async def _fetch_on_connection(
        self,
        conn: aiosqlite.Connection,
        query: str,
        params: Params,
        *,
        one: Literal[False],
    ) -> List[Row]: ...

    async def _fetch_on_connection(
        self,
        conn: aiosqlite.Connection,
        query: str,
        params: Params,
        *,
        one: bool,
    ) -> Optional[Row] | List[Row]:
        """Run a read without making cancelled cursor cleanup an inline drain.

        ``Cursor.close()`` is queued on the aiosqlite worker. If cancellation
        lands while ``execute``/``fetch*`` is stuck in that worker, awaiting
        cursor close from ``finally`` waits behind the same stuck statement and
        defeats the caller's deadline. On the shared connection, hand the queue
        drain to the retained rollback fence used by cancelled writes; snapshot
        connections remain owned by :meth:`_read_connection` and close there.
        """
        cursor: Optional[aiosqlite.Cursor] = None
        cancelled = False
        try:
            cursor = await conn.execute(query, params)
            result = await (cursor.fetchone() if one else cursor.fetchall())
            cursor_to_close = cursor
            cursor = None
            await cursor_to_close.close()
            if one:
                return None if result is None else tuple(result)
            return [tuple(row) for row in result]
        except asyncio.CancelledError:
            cancelled = True
            if conn is self._connection and not self._in_transaction:
                self._handoff_cancelled_write(conn)
            elif conn is not self._connection:
                # A snapshot is private to this read, so it needs no shared
                # transaction fence. Interrupt its current SQLite operation so
                # _read_connection can close the owned worker without waiting
                # for the abandoned query to finish naturally.
                try:
                    await conn.interrupt()
                except Exception:
                    logger.exception(
                        "Failed to interrupt cancelled SQLite snapshot read"
                    )
            raise
        finally:
            # Ordinary failures still close synchronously. Cancellation skips
            # this queue operation: the retained rollback (or enclosing
            # transaction) is the terminal worker marker for the abandoned
            # cursor operation.
            if cursor is not None and not cancelled:
                await cursor.close()

    async def _fetch_one(
        self,
        query: str,
        params: Params,
        *,
        diagnostic: bool,
    ) -> Optional[Row]:
        record_write_query(query)
        try:
            async with self._read_connection(diagnostic=diagnostic) as conn:
                if conn is self._connection:
                    # Serialize the complete shared-connection read unit. This
                    # prevents a writer from queueing between cancellation and
                    # the retained rollback marker. Transaction-owner reads are
                    # re-entrant; sibling transaction reads already selected a
                    # committed snapshot in ``_read_connection``.
                    async with self._write_guard():
                        row = await self._fetch_on_connection(
                            conn, query, params, one=True
                        )
                else:
                    row = await self._fetch_on_connection(
                        conn, query, params, one=True
                    )
            return row
        except Exception as e:
            raise QueryError(f"Query failed: {e}\nQuery: {query}") from e

    async def fetch_one(self, query: str, params: Params = ()) -> Optional[Row]:
        """Fetch a single row after retained cancellation cleanup drains."""
        return await self._fetch_one(query, params, diagnostic=False)

    async def fetch_one_diagnostic(
        self, query: str, params: Params = ()
    ) -> Optional[Row]:
        """Fetch committed state without waiting for retained write cleanup."""
        return await self._fetch_one(query, params, diagnostic=True)

    async def _fetch_all(
        self,
        query: str,
        params: Params,
        *,
        diagnostic: bool,
    ) -> List[Row]:
        record_write_query(query)
        try:
            async with self._read_connection(diagnostic=diagnostic) as conn:
                if conn is self._connection:
                    async with self._write_guard():
                        rows = await self._fetch_on_connection(
                            conn, query, params, one=False
                        )
                else:
                    rows = await self._fetch_on_connection(
                        conn, query, params, one=False
                    )
            return rows
        except Exception as e:
            raise QueryError(f"Query failed: {e}\nQuery: {query}") from e

    async def fetch_all(self, query: str, params: Params = ()) -> List[Row]:
        """Fetch all rows after retained cancellation cleanup drains."""
        return await self._fetch_all(query, params, diagnostic=False)

    async def fetch_all_diagnostic(
        self, query: str, params: Params = ()
    ) -> List[Row]:
        """Fetch committed rows without waiting for retained write cleanup."""
        return await self._fetch_all(query, params, diagnostic=True)
    
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
            except asyncio.CancelledError:
                if not self._in_transaction:
                    self._handoff_cancelled_write(conn)
                raise
            except Exception as e:
                if not self._in_transaction:
                    await self._rollback_after_failure(conn)
                raise QueryError(f"Script execution failed: {e}") from e
            except BaseException:
                if not self._in_transaction:
                    await self._rollback_after_failure(conn)
                raise

    @asynccontextmanager
    async def transaction(self, *, immediate: bool = False) -> AsyncIterator[None]:
        """Transaction context manager.

        ``BEGIN IMMEDIATE`` is reserved for one-time schema migrations that
        must acquire SQLite's writer slot before inspecting a migration marker.
        Normal data mutations retain deferred ``BEGIN`` so read-first flows do
        not unnecessarily block one another.
        """
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
        async with self._write_guard():
            self._in_transaction = True
            self._txn_owner = asyncio.current_task()
            try:
                await conn.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
                yield
                await conn.commit()
            except asyncio.CancelledError:
                self._handoff_cancelled_write(conn)
                raise
            except Exception as e:
                await self._rollback_after_failure(conn)
                raise TransactionError(f"Transaction failed: {e}") from e
            except BaseException:
                await self._rollback_after_failure(conn)
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

    async def table_exists_diagnostic(self, table_name: str) -> bool:
        """Check schema state through the committed diagnostic read path."""
        row = await self.fetch_one_diagnostic(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        )
        return row is not None
