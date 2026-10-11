"""
SQLite Database Backend

Implementation of DatabaseBackend using aiosqlite.
"""

import asyncio
import logging
import os
import sqlite3
import time
import weakref
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Iterator, List, Literal, Optional, overload

import aiosqlite

from kestrel_sovereign.execution_custody import (
    ExecutionCommitOutcomeError,
    _capture_host_identity_anchor_access,
    current_execution_custody,
    execution_commit_outcome,
    execution_terminal_error,
    is_execution_control_error,
    require_execution_backend,
)

from .interface import (
    ConnectionError,
    DatabaseBackend,
    Params,
    QueryError,
    Row,
    TransactionError,
)
from .transaction_control import reject_transaction_control, sqlite_statements
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

# SQLite already treats a lock wait of up to 30 seconds as healthy runtime
# behavior.  Cancellation cleanup is a runtime-availability concern, not a
# worker-shutdown concern, so it must not inherit the much shorter lifecycle
# timeout above.  Interrupting the abandoned statement normally makes this
# handoff immediate; the busy window covers operations (such as a Python UDF
# or stalled VFS call) that cannot observe sqlite3_interrupt() until they
# return control to SQLite.
_SQLITE_BUSY_TIMEOUT_S = 30.0
_CANCELLED_OPERATION_DRAIN_TIMEOUT_S = _SQLITE_BUSY_TIMEOUT_S

# Backoff between attempts of a cancellation-drain rollback that the drain's
# own interrupt aborted. Attempts stop at the drain deadline above.
_CANCELLED_WRITE_ROLLBACK_RETRY_INITIAL_S = 0.01
_CANCELLED_WRITE_ROLLBACK_RETRY_MAX_S = 0.25


class ColdReadUnavailable(RuntimeError):
    """A cold read was requested for a database that is not quiescent.

    Its own type rather than ``ValueError``: callers distinguish "this store
    is busy, ask again when the agent is stopped" from a bad argument, and a
    sibling of a stdlib exception would be caught by handlers that never meant
    to cover this.
    """


class _CancelledWriteDrainDeadlineExceeded(RuntimeError):
    """A retained rollback is still running after a later writer's budget."""


def _is_sqlite_interrupt(exc: Exception) -> bool:
    """Whether ``exc`` is SQLite's ``SQLITE_INTERRUPT`` result, nothing broader."""
    return (
        isinstance(exc, sqlite3.OperationalError)
        and getattr(exc, "sqlite_errorcode", None) == sqlite3.SQLITE_INTERRUPT
    )


def _owned_script_statements(script: str) -> list[str]:
    """Split with SQLite's grammar, retaining the outer transaction owner.

    ``executescript`` implicitly commits before executing anything, even when
    our task owns an open transaction. Use individual complete statements in
    that case. SQLite's completeness parser handles quoted semicolons and
    trigger bodies; this is not a split-on-semicolon SQL parser.
    """
    reject_transaction_control(script, dialect="sqlite")
    return sqlite_statements(script)


@dataclass
class _RetainedAiosqliteClose:
    """Strong ownership of one close until its worker terminates."""

    close_task: asyncio.Task[None]
    retirement_task: Optional[asyncio.Task[None]] = None
    control_error: Optional[BaseException] = None
    control_reported: bool = False


def _capture_retained_close(retained: _RetainedAiosqliteClose) -> None:
    if not retained.close_task.done():
        return
    from kestrel_sovereign._async_ownership import owned_task_error
    error = owned_task_error(retained.close_task)
    if error is not None and is_execution_control_error(error):
        selected = execution_terminal_error(retained.control_error, error)
        if selected is not retained.control_error:
            retained.control_error = selected
            retained.control_reported = False


_RetainedAiosqliteCloses = dict[
    aiosqlite.Connection, _RetainedAiosqliteClose
]


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


def _retain_aiosqlite_close(
    connection: aiosqlite.Connection,
    close_task: asyncio.Task[None],
    retained_closes: _RetainedAiosqliteCloses,
) -> None:
    """Own an in-flight close until its non-daemon worker really exits."""
    existing = retained_closes.get(connection)
    if existing is not None and (
        not existing.close_task.done() or _aiosqlite_worker_is_alive(connection)
    ):
        return
    retained_closes.pop(connection, None)

    retained = _RetainedAiosqliteClose(close_task=close_task)

    async def reap() -> None:
        pending_cancellation: Optional[asyncio.CancelledError] = None
        try:
            await asyncio.shield(retained.close_task)
        except asyncio.CancelledError as exc:
            # The public close task is normally cancelled by the bounded
            # caller below. A cancellation of this reaper must not discard
            # lifecycle ownership while a non-daemon worker is still alive.
            pending_cancellation = exc
        except BaseException:
            # A timeout caller may have returned before this original outcome.
            # Capture it independently of the worker's physical lifetime.
            pass
        _capture_retained_close(retained)

        while _aiosqlite_worker_is_alive(connection):
            try:
                await asyncio.sleep(_AIOSQLITE_WORKER_SHUTDOWN_POLL_S)
            except asyncio.CancelledError as exc:
                pending_cancellation = pending_cancellation or exc

        if pending_cancellation is not None:
            raise pending_cancellation

    loop = asyncio.get_running_loop()
    retirement_task = loop.create_task(
        reap(),
        name=f"aiosqlite-close-retirement:{id(connection)}",
    )
    retained.retirement_task = retirement_task
    retained_closes[connection] = retained
    close_task.add_done_callback(lambda _task: _capture_retained_close(retained))

    def release_ownership(_task: asyncio.Task[None]) -> None:
        # Cancellation before the reaper's first event-loop turn cannot run
        # its cancellation-resistant body. Keep the connection and close task
        # strongly owned in that case; the next lifecycle probe prunes them
        # only after the worker itself has exited.
        if (
            retained_closes.get(connection) is retained
            and not _aiosqlite_worker_is_alive(connection)
            and retained.close_task.done()
        ):
            _capture_retained_close(retained)
            if retained.control_error is None or retained.control_reported:
                retained_closes.pop(connection)
        # Consume reaper cancellation without discarding the native receipt.
        if not _task.cancelled():
            _task.exception()

    retirement_task.add_done_callback(release_ownership)


def _aiosqlite_worker_is_alive(connection: aiosqlite.Connection) -> bool:
    """Return whether either supported aiosqlite worker shape is alive."""
    worker = getattr(connection, "_thread", connection)
    is_alive = getattr(worker, "is_alive", None)
    return bool(callable(is_alive) and is_alive())


def _prune_retained_aiosqlite_closes(
    retained_closes: _RetainedAiosqliteCloses,
    *, preserve_control_outcomes: bool = False,
) -> None:
    """Release lifecycle records only after their workers have exited."""
    for connection, retained in list(retained_closes.items()):
        _capture_retained_close(retained)
        if not _aiosqlite_worker_is_alive(connection) and retained.close_task.done() and (
            not preserve_control_outcomes or retained.control_error is None
            or retained.control_reported
        ):
            retained_closes.pop(connection)


async def _close_aiosqlite_connection(
    connection: aiosqlite.Connection,
    *,
    retained_closes: _RetainedAiosqliteCloses,
    deadline: Optional[float] = None,
    close_task: Optional[asyncio.Task[None]] = None,
    cancel_close_task_on_timeout: bool = True,
) -> None:
    """Close an owned aiosqlite connection through its full lifecycle.

    Every connection this backend opens owns an aiosqlite worker.  Keeping the
    close and worker-termination wait together prevents a short-lived backup
    or snapshot connection from bypassing the shutdown contract.  A caller
    coordinating several connections may pass an already-started ``close_task``
    after installing retained ownership before its first cancellable await.
    SQLAlchemy callers set ``cancel_close_task_on_timeout=False`` because their
    task is the one engine-disposal attempt already closing the same driver;
    starting a second close would race two sentinels against one worker.
    """
    loop = asyncio.get_running_loop()
    if deadline is None:
        deadline = loop.time() + _minimum_close_timeout_s()
    if close_task is None:
        from kestrel_sovereign._async_ownership import create_owned_task
        close_task = create_owned_task(connection.close())

    try:
        await _finish_aiosqlite_connection_close(
            connection,
            close_task=close_task,
            deadline=deadline,
            retained_closes=retained_closes,
            cancel_close_task_on_timeout=cancel_close_task_on_timeout,
        )
    except BaseException as error:
        # Repeated cancellation can interrupt either bounded retry. Never let
        # an exit shed ownership of a still-live non-daemon worker.
        if _aiosqlite_worker_is_alive(connection) or not close_task.done():
            _retain_aiosqlite_close(
                connection, close_task, retained_closes
            )
        from kestrel_sovereign._async_ownership import owned_task_error
        close_error = owned_task_error(close_task) if close_task.done() else None
        terminal = execution_terminal_error(close_error, error)
        retained = retained_closes.get(connection)
        if retained is not None:
            _capture_retained_close(retained)
            if terminal is retained.control_error:
                retained.control_reported = True
        raise terminal


async def _finish_aiosqlite_connection_close(
    connection: aiosqlite.Connection,
    *,
    close_task: asyncio.Task[None],
    deadline: float,
    retained_closes: _RetainedAiosqliteCloses,
    cancel_close_task_on_timeout: bool,
) -> None:
    """Complete one bounded close under the caller's retention guard."""
    loop = asyncio.get_running_loop()
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
        interrupt_error: Optional[Exception] = None
        try:
            await connection.interrupt()
        except ValueError as exc:
            if is_execution_control_error(exc) or "no active connection" not in str(exc):
                interrupt_error = exc
        except Exception as exc:
            interrupt_error = exc
        if cancel_close_task_on_timeout:
            close_task.cancel()
        _retain_aiosqlite_close(
            connection, close_task, retained_closes
        )
        if interrupt_error is not None and is_execution_control_error(interrupt_error):
            raise execution_terminal_error(interrupt_error, pending_cancellation)
        error = ConnectionError(
            "SQLite connection close did not complete within "
            f"{_minimum_close_timeout_s():.2f}s"
        )
        if interrupt_error is not None:
            raise error from interrupt_error
        raise error

    from kestrel_sovereign._async_ownership import owned_task_error
    close_error = owned_task_error(close_task)

    remaining = max(0.0, deadline - loop.time())
    try:
        try:
            await _wait_for_aiosqlite_worker_shutdown(
                connection,
                timeout_s=remaining,
            )
        except ConnectionError as exc:
            _retain_aiosqlite_close(
                connection, close_task, retained_closes
            )
            raise execution_terminal_error(close_error, pending_cancellation, exc)
    except asyncio.CancelledError as exc:
        pending_cancellation = pending_cancellation or exc
        remaining = max(0.0, deadline - loop.time())
        try:
            await _wait_for_aiosqlite_worker_shutdown(
                connection,
                timeout_s=remaining,
            )
        except ConnectionError as exc:
            _retain_aiosqlite_close(
                connection, close_task, retained_closes
            )
            raise execution_terminal_error(close_error, pending_cancellation, exc)

    terminal = execution_terminal_error(close_error, pending_cancellation)
    if terminal is not None:
        raise terminal


class SQLiteBackend(DatabaseBackend):
    """
    SQLite database backend using aiosqlite.
    
    Features:
    - File-based or in-memory databases
    - Automatic directory creation
    - WAL mode for better concurrency
    - Foreign key enforcement
    """
    
    def __init__(self, db_path: str, *, cold_read: bool = False):
        """
        Initialize SQLite backend.

        Args:
            db_path: Path to SQLite database file, or ":memory:" for in-memory
            cold_read: Open the database without disturbing it at all, for
                inspection paths. SQLite serialises writers at the FILE level
                and the per-connection write-unit lock cannot serialise a
                *second* connection to the same file, so an inspection that
                opens read-write can contend with a running agent even though
                it never writes (#2920).

                Not merely ``mode=ro``: a plain read-only connection opening a
                WAL database still CREATES the ``-wal``/``-shm`` sidecars and,
                lacking write access, cannot remove them on close. Those
                leftovers are read as live WAL state by the cold identity
                lookup, so a "read-only" inspection would fabricate evidence
                that the agent is running. This uses the same discipline as
                ``identity.local_anchor``: refuse when sidecars are present,
                then open ``immutable=1``, which creates none.
        """
        self.cold_read = cold_read
        #: Resolved ONCE, and used for the URI, the sidecar checks and the
        #: change marker alike. SQLite places ``-wal``/``-shm`` beside the
        #: RESOLVED target, so a symlinked ``db_path`` would have the sidecar
        #: check looking beside the alias while the connection opens the
        #: target — selecting ``immutable=1`` and silently serving pre-WAL
        #: governance state as current.
        self._resolved_path = (
            None if db_path == ":memory:" else Path(db_path).resolve()
        )
        self._identity_anchor_access = _capture_host_identity_anchor_access(
            self._resolved_path, cold_read=cold_read,
        )
        #: Whether the cold-read connection was opened ``immutable=1`` and is
        #: therefore blind to a WAL that appears while it is open.
        self._cold_read_ignored_wal = False
        #: Committed-state fingerprint taken when an immutable read began.
        self._cold_read_marker: tuple = ()
        self._connection: Optional[aiosqlite.Connection] = None
        self._connected_file_identity: Optional[tuple[int, int]] = None
        self._in_transaction = False
        self._transaction_poisoned = False
        self._savepoint_counter = 0
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
        # ``_write_lock`` also serializes shared-connection reads through
        # cursor cleanup, so lock occupancy alone no longer means that close
        # should interrupt SQLite. Track the write API explicitly to preserve
        # clean WAL checkpointing when shutdown merely overlaps a read.
        self._active_write_operations = 0
        self._txn_owner: Optional["asyncio.Task"] = None
        # aiosqlite cannot cancel work already running in its worker thread.
        # When its caller is cancelled, transfer the queued rollback to a
        # retained task and fence later writers on that drain.  This lets an
        # outer deadline release task-owned resources promptly without
        # allowing another writer to commit the abandoned statement (#2907).
        self._cancelled_write_drain: Optional[asyncio.Task[None]] = None
        self._cancelled_write_drain_started_at: Optional[float] = None
        self._cancelled_write_drain_error: Optional[BaseException] = None
        # A cancelled shared-connection read skips its inline cursor close, so
        # its statement can stay active and keep the drain's interrupt armed.
        # Every drain closes the ones still alive, before it retires its fence
        # and before retrying a rollback such a statement aborted. Weak, so
        # recording a cursor never extends its statement's life.
        self._abandoned_read_cursors: weakref.WeakSet[aiosqlite.Cursor] = (
            weakref.WeakSet()
        )
        # A cancelled execute can return a native cursor after its caller has
        # unwound. Own that return explicitly; resetting the worker's last
        # result cannot release a cursor still held by a queued result callback.
        self._pending_read_executions: set[asyncio.Task] = set()
        self._retired_connection_closes: _RetainedAiosqliteCloses = {}
        self._closing = False
    
    @property
    def backend_type(self) -> str:
        return "sqlite"

    @property
    def nested_transaction_strategy(self) -> str:
        """SQLite currently joins same-task nested transaction scopes."""
        return "joined"

    @property
    def transaction_options(self) -> frozenset[str]:
        """Explicit extensions beyond the SDK's transaction() contract."""
        return frozenset({"immediate", "savepoint"})

    @property
    def owns_open_transaction(self) -> bool:
        """Whether this task still owns the enclosing transaction scope.

        Implicit rollback destroys native write custody, not the caller scope
        or its writer lock. Keep ownership visible until that scope exits so
        higher-level lock ordering cannot treat poisoned custody as top-level.
        SQL operations separately enforce healthy native transaction custody.
        """
        return self._in_transaction and self._txn_owner is asyncio.current_task()

    @property
    def is_connected(self) -> bool:
        return self._connection is not None

    @property
    def db_path(self) -> str:
        """Construction-time resolved target, shared by all native consumers.

        Retargeting a connected backend is not supported. In particular a
        caller cannot rewrite its public path while the connection retains a
        different inode. Consumers opening siblings also verify file custody.
        """
        return ":memory:" if self._resolved_path is None else str(self._resolved_path)

    @property
    def connected_file_identity(self) -> Optional[tuple[int, int]]:
        """The inode retained at connection acquisition, not a fresh path stat."""
        return self._connected_file_identity

    def assert_connected_file_still_valid(self) -> None:
        """Refuse file custody inferred from a replaced pathname or new cwd."""
        if self._connection is None:
            raise ConnectionError("SQLite file custody requires a connected backend")
        if self._resolved_path is None:
            return
        try:
            current = self._resolved_path.lstat()
        except OSError as exc:
            raise ConnectionError("SQLite connected database pathname changed; retain recovery evidence") from exc
        if (current.st_dev, current.st_ino) != self._connected_file_identity:
            raise ConnectionError("SQLite connected database inode changed; retain recovery evidence")

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
    def write_connection_cleanup_deadline_exceeded(self) -> bool:
        """Whether retained cleanup passed the bounded availability window."""
        if isinstance(
            self._cancelled_write_drain_error,
            _CancelledWriteDrainDeadlineExceeded,
        ):
            return True
        drain = self._cancelled_write_drain
        started_at = self._cancelled_write_drain_started_at
        return (
            drain is not None
            and not drain.done()
            and started_at is not None
            and self._cancelled_write_drain_remaining(started_at) <= 0
        )

    @property
    def connection_retirement_pending(self) -> bool:
        """Whether a timed-out close still owns a live aiosqlite worker."""
        _prune_retained_aiosqlite_closes(
            self._retired_connection_closes, preserve_control_outcomes=True
        )
        return any(_aiosqlite_worker_is_alive(connection) or not record.close_task.done()
                   for connection, record in self._retired_connection_closes.items())

    def _raise_if_connection_retirement_pending(self) -> None:
        """Fail close while any backend-owned worker is still retiring."""
        control = None
        for retained in self._retired_connection_closes.values():
            _capture_retained_close(retained)
            if not retained.control_reported:
                control = execution_terminal_error(control, retained.control_error)
        if control is not None:
            for retained in self._retired_connection_closes.values():
                if retained.control_error is control:
                    retained.control_reported = True
            raise control
        if self.connection_retirement_pending:
            raise ConnectionError(
                "SQLite previous connection worker is still retiring after its "
                "close deadline"
            )

    @property
    def minimum_close_timeout_s(self) -> float:
        """Minimum budget an outer shutdown guard must reserve for ``close``.

        This is intentionally an optional backend extension rather than a
        change to the SDK's generic backend interface: non-SQLite backends do
        not own an aiosqlite worker and therefore have no equivalent lifecycle
        requirement.  A retained cancellation drain and the subsequent
        connection close share one absolute worker-shutdown deadline.  This
        keeps the primary-backend reservation equal to the single-connection
        lifecycle contract also used by SQLAlchemy factories, without
        shrinking the rest of whole-agent shutdown.
        """
        return _minimum_close_timeout_s()
    
    def _wal_sidecars(self) -> tuple[Path, ...]:
        """The files SQLite creates beside a WAL database while it is in use."""
        base = self._resolved_path
        return (Path(f"{base}-wal"), Path(f"{base}-shm"))

    def _committed_state_marker(self) -> tuple:
        """A fingerprint of the database's committed state.

        Sidecar presence alone cannot see a writer that opened, committed and
        closed entirely within an immutable read: its final close checkpoints
        and REMOVES the sidecars, so the after-check finds none and passes
        while this connection has been serving pre-write state throughout.

        Measured, because the obvious candidate does not work: SQLite does
        NOT move the header's change counter (bytes 24-27) or its
        version-valid-for number when a WAL commit is checkpointed back into
        the main file — both were observed unchanged across exactly the write
        this must catch. The file's mtime and contents do change, so identity
        and modification time are what get compared. Inode is included because
        a database replaced wholesale is the same event by another route.

        This matters because an immutable reader caches what it has already
        read: after a checkpoint it keeps serving the pages it cached, so a
        second read of the same row returns the pre-write value indefinitely.
        """
        base = self._resolved_path
        if base is None:
            return ()
        try:
            stat = base.stat()
        except OSError:
            return ()
        return (stat.st_ino, stat.st_mtime_ns, stat.st_size)

    def _has_wal_sidecars(self) -> bool:
        return any(sidecar.exists() for sidecar in self._wal_sidecars())

    def _cold_read_uri(self) -> str:
        """Build the cold-read URI, adapted to what is already on disk.

        Neither branch writes, checkpoints, or takes a write lock, so neither
        can contend with a running agent. They differ only in how they handle
        a WAL, and each is wrong in the other's situation:

        ``immutable=1`` promises SQLite the file cannot change. That is what
        stops it creating the ``-wal``/``-shm`` files a plain read-only open
        leaves behind and cannot remove — but it also makes SQLite ignore WAL
        content outright, so using it while a WAL exists would report pre-WAL
        governance state as current.

        A plain ``mode=ro`` reads the WAL truthfully, but on a quiescent
        database it CREATES the sidecars, which the cold identity lookup then
        reads as a live agent — an inspection that plants the evidence.

        So: immutable when there is no WAL to miss, plain read-only when there
        is one, and in both cases the directory is left as it was found.
        """
        # ``as_uri()`` percent-escapes the path. Interpolating a bare path
        # would let a directory named e.g. ``Test?Agent`` end the path early
        # and swallow the flags that make this read cold — SQLite would fall
        # back to read-write-create and open a different file, silently
        # defeating the mechanism.
        self._cold_read_ignored_wal = not self._has_wal_sidecars()
        if self._cold_read_ignored_wal:
            self._cold_read_marker = self._committed_state_marker()
        flags = "mode=ro&immutable=1" if self._cold_read_ignored_wal else "mode=ro"
        return f"{self._resolved_path.as_uri()}?{flags}"

    def warn_if_wal_state_was_stranded(self) -> None:
        """Say so when this read may have kept a WAL from being cleaned up.

        Only the plain read-only branch can do this, and it is not avoidable
        while still reading the WAL truthfully. SQLite lets the LAST connection
        to close checkpoint and remove the sidecars; a read-only connection
        cannot perform that cleanup itself, but it does count as a connection.
        So if the agent exits while this inspection is open, the writer is no
        longer last, the checkpoint never happens, and the sidecars outlive
        both — after which a cold identity lookup refuses the agent and a cold
        wake can be blocked.

        Copying to a snapshot would not fix it: the copy needs a read
        connection on the original too, which shortens the window rather than
        closing it. The race is inherent to reading a live WAL database, so it
        is reported rather than hidden — #2920 was filed because the cost of an
        inspection landed on the agent while the caller saw nothing at all.
        """
        if not self.cold_read or self._cold_read_ignored_wal:
            return
        if self._has_wal_sidecars():
            logger.warning(
                "cold read of %s finished with SQLite WAL sidecars still "
                "present. If the agent exited during this inspection they may "
                "have outlived it, and a cold identity lookup will refuse the "
                "agent until they are cleared. Starting the agent normally "
                "checkpoints and removes them.",
                self.db_path,
            )

    def assert_cold_read_still_valid(self) -> None:
        """Refuse to report on an immutable read that a writer raced.

        Only meaningful for the immutable branch: it was chosen because there
        was no WAL, and an immutable reader ignores one that appears, so a
        writer arriving mid-read would leave this connection reporting stale
        state as current. Detecting it afterwards is the only defence an
        immutable reader has. Callers must invoke this before acting on what
        they read — the storage layer cannot know when a read is complete.
        """
        if not self.cold_read or self.db_path == ":memory:":
            return
        if not self._cold_read_ignored_wal:
            return
        if self._has_wal_sidecars() or (
            self._cold_read_marker
            and self._committed_state_marker() != self._cold_read_marker
        ):
            raise ColdReadUnavailable(
                f"Cannot report on {self.db_path}: the database changed while "
                "it was being read, and this connection was opened immutable "
                "so it has been serving the state from before that change. "
                "Stop the agent and retry."
            )

    async def connect(self) -> None:
        """Connect to SQLite database."""
        self._require_execution_backend()
        self._raise_if_connection_retirement_pending()
        if self._connection is not None:
            if self._cancelled_write_drain_error is None:
                return
            # A failed cancellation cleanup makes the shared connection
            # untrustworthy. ``connect`` is also the explicit recovery surface:
            # retire that connection fully before opening its replacement.
            await self.close()
        
        try:
            # Create directory if needed (unless in-memory). A cold-read
            # opener must not bring the database it is inspecting into
            # existence — "there is nothing here" is an answer it has to be
            # able to give.
            if self.db_path != ":memory:" and not self.cold_read:
                db_dir = self._resolved_path.parent
                db_dir.mkdir(parents=True, exist_ok=True)

            before_identity = None
            if self._resolved_path is not None:
                try:
                    before = self._resolved_path.stat()
                except FileNotFoundError:
                    pass
                else:
                    before_identity = (before.st_dev, before.st_ino)

            if self.cold_read:
                self._connection = await aiosqlite.connect(
                    self._cold_read_uri(),
                    uri=True,
                    timeout=_SQLITE_BUSY_TIMEOUT_S,
                )
            else:
                self._connection = await aiosqlite.connect(
                    str(self._resolved_path) if self._resolved_path is not None else ":memory:",
                    timeout=_SQLITE_BUSY_TIMEOUT_S,
                )

            if self._resolved_path is not None:
                opened = self._resolved_path.stat()
                self._connected_file_identity = (opened.st_dev, opened.st_ino)
                if before_identity is not None and before_identity != self._connected_file_identity:
                    raise ConnectionError("SQLite database inode changed during connection acquisition")

            # Enable WAL mode for better concurrency. Skipped for a cold read:
            # setting the journal mode WRITES to the database header, so this
            # is not merely unnecessary there — it would fail the connection
            # outright, which is the trap in making this path read-only.
            if not self.cold_read:
                await self._connection.execute("PRAGMA journal_mode=WAL")
            
            # Allow concurrent writers to wait up to 30s for the lock
            await self._connection.execute(
                f"PRAGMA busy_timeout={int(_SQLITE_BUSY_TIMEOUT_S * 1000)}"
            )
            
            # Enable foreign keys
            await self._connection.execute("PRAGMA foreign_keys=ON")
            
            # Row factory to return tuples
            self._connection.row_factory = aiosqlite.Row

            # A newly opened connection cannot contain work abandoned on the
            # previous connection.  Do not carry a failed cleanup latch across
            # a successful reconnect.
            self._cancelled_write_drain = None
            self._cancelled_write_drain_started_at = None
            self._cancelled_write_drain_error = None
            self._abandoned_read_cursors.clear()
            self._closing = False
            
            logger.debug(f"Connected to SQLite: {self.db_path}")
            
        except Exception as e:
            raise ConnectionError(f"Failed to connect to SQLite: {e}") from e
    
    async def close(self) -> None:
        """Close database connection and wait for background thread to stop."""
        conn = self._connection
        failure = (
            self._cancelled_write_drain_error
            if self._cancelled_write_drain_error is not None
            and is_execution_control_error(self._cancelled_write_drain_error)
            else None
        )
        if conn is None:
            self._raise_if_connection_retirement_pending()
            self._cancelled_write_drain = None
            self._cancelled_write_drain_started_at = None
            self._cancelled_write_drain_error = None
            if failure is not None:
                raise failure
            return

        pending_cancellation: Optional[asyncio.CancelledError] = None
        loop = asyncio.get_running_loop()
        close_deadline = loop.time() + self.minimum_close_timeout_s
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
            if (
                drain is not None
                or self._in_transaction
                or self._active_write_operations > 0
            ):
                try:
                    await conn.interrupt()
                except asyncio.CancelledError as exc:
                    # Cancellation cannot transfer ownership of the primary
                    # non-daemon worker. Preserve it for delivery only after
                    # the connection has completed its bounded retirement.
                    pending_cancellation = exc
                except ValueError as exc:
                    if is_execution_control_error(exc):
                        failure = execution_terminal_error(failure, exc)
                    if "no active connection" not in str(exc):
                        logger.warning(
                            "Could not interrupt SQLite before close: %s",
                            exc,
                        )
                except Exception as exc:
                    if is_execution_control_error(exc):
                        failure = execution_terminal_error(failure, exc)
                    # Interrupt is an acceleration path. Its failure must not
                    # skip the lifecycle-owned close that follows.
                    logger.warning(
                        "Could not interrupt SQLite before close",
                        exc_info=True,
                    )
                except BaseException as error:
                    failure = execution_terminal_error(failure, error)
            if drain is not None and drain is not asyncio.current_task():
                try:
                    async with asyncio.timeout(
                        min(
                            AIOSQLITE_WORKER_SHUTDOWN_TIMEOUT_S,
                            max(0.0, close_deadline - loop.time()),
                        )
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
                        if is_execution_control_error(drain_cancelled):
                            failure = execution_terminal_error(failure, drain_cancelled)
                    except BaseException as error:
                        failure = execution_terminal_error(failure, error)
                except BaseException as error:
                    failure = execution_terminal_error(failure, error)

            # A grouped drain failure cannot skip the independently owned
            # native close, or erase its original carrier after the join.
            failure = execution_terminal_error(
                failure, self._cancelled_write_drain_error
                if self._cancelled_write_drain_error is not None
                and is_execution_control_error(self._cancelled_write_drain_error)
                else None,
            )
            try:
                await _close_aiosqlite_connection(
                    conn,
                    retained_closes=self._retired_connection_closes,
                    deadline=close_deadline,
                )
                self._raise_if_connection_retirement_pending()
            except BaseException as error:
                failure = execution_terminal_error(failure, error)
            logger.debug(f"Closed SQLite connection: {self.db_path}")
        finally:
            self._connection = None
            self._connected_file_identity = None
            self._cancelled_write_drain = None
            self._cancelled_write_drain_started_at = None
            self._cancelled_write_drain_error = None
            self._abandoned_read_cursors.clear()
            self._closing = False

        terminal = execution_terminal_error(failure, pending_cancellation)
        if terminal is not None:
            raise terminal
    
    def _require_execution_backend(self) -> None:
        require_execution_backend("sqlite", _identity_anchor_access=self._identity_anchor_access)

    def _ensure_connected(self) -> aiosqlite.Connection:
        """Ensure we have an active connection."""
        self._require_execution_backend()
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
            dest = await aiosqlite.connect(
                dest_path, timeout=_SQLITE_BUSY_TIMEOUT_S
            )
            body_error = None
            try:
                await conn.backup(dest)
            except BaseException as error:
                body_error = error
                raise
            finally:
                try:
                    await _close_aiosqlite_connection(
                        dest,
                        retained_closes=self._retired_connection_closes,
                    )
                except BaseException as cleanup_error:
                    raise execution_terminal_error(body_error, cleanup_error)

    async def _open_snapshot_read_connection(self) -> aiosqlite.Connection:
        """Open a one-shot connection for committed reads during another task's txn."""
        self.assert_connected_file_still_valid()
        conn = await aiosqlite.connect(
            self._resolved_path.as_uri() + "?mode=rw",
            uri=True,
            timeout=_SQLITE_BUSY_TIMEOUT_S,
        )
        try:
            self.assert_connected_file_still_valid()
            await conn.execute(
                f"PRAGMA busy_timeout={int(_SQLITE_BUSY_TIMEOUT_S * 1000)}"
            )
            await conn.execute("PRAGMA foreign_keys=ON")
            await conn.execute("PRAGMA query_only=ON")
            conn.row_factory = aiosqlite.Row
            return conn
        except BaseException as error:
            try:
                await _close_aiosqlite_connection(
                    conn,
                    retained_closes=self._retired_connection_closes,
                )
            except BaseException as cleanup_error:
                raise execution_terminal_error(error, cleanup_error)
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
            body_error = None
            try:
                yield read_conn
            except BaseException as error:
                body_error = error
                raise
            finally:
                try:
                    await _close_aiosqlite_connection(
                        read_conn,
                        retained_closes=self._retired_connection_closes,
                    )
                except BaseException as cleanup_error:
                    raise execution_terminal_error(body_error, cleanup_error)
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
            self._require_native_transaction()
            self._require_execution_backend()
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
                self._require_execution_backend()
                yield
                return

    def _require_native_transaction(self) -> None:
        """Implicit SQLite rollback must never turn an owning scope into autocommit."""
        if self._in_transaction and self._txn_owner is asyncio.current_task():
            if self._transaction_poisoned or not self._ensure_connected().in_transaction:
                self._transaction_poisoned = True
                raise TransactionError("SQLite owning transaction was rolled back implicitly; scope is poisoned")

    @contextmanager
    def _write_operation(self) -> Iterator[None]:
        """Mark one write API operation independently of lock occupancy."""
        self._active_write_operations += 1
        try:
            yield
        finally:
            self._active_write_operations -= 1

    def _raise_cancelled_write_drain_error(self) -> None:
        """Fail closed after a detached rollback could not restore safety."""
        error = self._cancelled_write_drain_error
        if error is not None:
            if is_execution_control_error(error):
                raise error
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
            started_at = self._cancelled_write_drain_started_at
            remaining = _CANCELLED_OPERATION_DRAIN_TIMEOUT_S
            if started_at is not None:
                remaining = max(
                    0.0, self._cancelled_write_drain_remaining(started_at)
                )
            try:
                async with asyncio.timeout(remaining):
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
        deadline.  The background drain interrupts the abandoned SQLite VM
        before queueing rollback. It deliberately does not acquire
        ``_write_lock``; ``_write_guard`` fences all later operations on the
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

        started_at = time.monotonic()
        self._cancelled_write_drain_error = None
        self._cancelled_write_drain_started_at = started_at
        self._cancelled_write_drain = asyncio.create_task(
            self._drain_cancelled_write(conn, started_at=started_at),
            name=f"sqlite-cancelled-write-drain:{self.db_path}",
        )

    async def _drain_cancelled_write(
        self, conn: aiosqlite.Connection, *, started_at: float
    ) -> None:
        """Interrupt and roll back one cancelled operation, release the
        statements cancelled reads abandoned, then retire its fence."""
        this_task = asyncio.current_task()
        interrupt_error = None
        try:
            try:
                await conn.interrupt()
            except ValueError as exc:
                if is_execution_control_error(exc):
                    interrupt_error = exc
                if "no active connection" not in str(exc):
                    logger.warning(
                        "Could not interrupt cancelled SQLite operation for %s: %s",
                        self.db_path,
                        exc,
                    )
            except Exception as exc:
                if is_execution_control_error(exc):
                    interrupt_error = exc
                # Rollback is still the terminal marker even when interrupt is
                # unavailable. Preserve the retained fence and let the runtime
                # drain deadline expose a genuinely stuck operation.
                logger.warning(
                    "Could not interrupt cancelled SQLite operation for %s",
                    self.db_path,
                    exc_info=True,
                )
            except BaseException as exc:
                interrupt_error = exc
            await self._rollback_cancelled_write(conn, started_at=started_at)
            if interrupt_error is not None:
                raise interrupt_error
        except BaseException as exc:
            self._cancelled_write_drain_error = execution_terminal_error(
                self._cancelled_write_drain_error, interrupt_error, exc,
            )
            logger.exception(
                "SQLite cancellation cleanup failed for %s", self.db_path
            )
        else:
            # The availability deadline belongs to the handoff, not to the
            # backend-owned rollback. A rollback that eventually succeeds
            # restores the shared connection and must remove only the
            # transient deadline marker; genuine cleanup failures stay latched.
            if isinstance(
                self._cancelled_write_drain_error,
                _CancelledWriteDrainDeadlineExceeded,
            ):
                self._cancelled_write_drain_error = None
        finally:
            if self._cancelled_write_drain is this_task:
                self._cancelled_write_drain = None
                self._cancelled_write_drain_started_at = None

    async def _rollback_cancelled_write(
        self, conn: aiosqlite.Connection, *, started_at: float
    ) -> None:
        """Roll back for the drain and release the statements cancelled reads
        abandoned, retrying until both converge.

        ``sqlite3_interrupt()`` also interrupts every statement started before
        the connection's active-statement count reaches zero, so a ROLLBACK
        queued while the cancelled operation left a statement active fails
        ``interrupted``. That is the drain's own interrupt, not a cleanup
        failure: release the abandoned statements and retry within the drain
        deadline. Any other failure, an interrupt while ``close`` owns the
        connection, or a rollback still interrupted at the deadline raises.

        Outside a transaction the rollback runs no statement, so it cannot
        clear the interrupt while an abandoned statement is active, and the
        next write's BEGIN would fail ``interrupted`` (#3508). The release
        therefore also follows a successful rollback, and the drain converges
        only once every recorded cursor has closed. A close still failing at
        the deadline, or once ``close`` owns the connection, raises.
        """
        retry_delay = _CANCELLED_WRITE_ROLLBACK_RETRY_INITIAL_S
        rolled_back = False
        while True:
            if not rolled_back:
                try:
                    await conn.rollback()
                except sqlite3.OperationalError as exc:
                    if is_execution_control_error(exc):
                        raise
                    if (
                        not _is_sqlite_interrupt(exc)
                        or self._closing
                        or self._cancelled_write_drain_remaining(started_at) <= 0
                    ):
                        raise
                    logger.debug(
                        "SQLite cancellation rollback for %s was interrupted by "
                        "its drain; retrying after releasing abandoned statements",
                        self.db_path,
                    )
                else:
                    rolled_back = True
            close_failure = await self._release_abandoned_statements(
                conn, started_at=started_at
            )
            if rolled_back:
                if close_failure is None:
                    return
                if (
                    self._closing
                    or self._cancelled_write_drain_remaining(started_at) <= 0
                ):
                    raise close_failure
            await asyncio.sleep(
                min(
                    retry_delay,
                    max(0.0, self._cancelled_write_drain_remaining(started_at)),
                )
            )
            retry_delay = min(
                2 * retry_delay, _CANCELLED_WRITE_ROLLBACK_RETRY_MAX_S
            )

    @staticmethod
    def _cancelled_write_drain_remaining(started_at: float) -> float:
        """Seconds left of the drain deadline that began at ``started_at``."""
        return _CANCELLED_OPERATION_DRAIN_TIMEOUT_S - (
            time.monotonic() - started_at
        )

    async def _release_abandoned_statements(
        self, conn: aiosqlite.Connection, *, started_at: float
    ) -> Optional[Exception]:
        """Reset the statements cancelled reads left active on ``conn``.

        Closes the abandoned read cursors still alive. Closing one already
        closed is a no-op, and a collected one has left the weak record. A
        cursor whose close fails stays recorded for the next release to
        retry, and the last such failure is returned; ``None`` means every
        recorded cursor closed. A read cancelled while its ``execute`` was in
        flight never received its cursor, and aiosqlite's worker keeps the
        last result it produced (that cursor) until it completes another
        operation. Opening and closing a cursor completes one, which releases
        it.
        """
        close_failure: Optional[Exception] = None
        from kestrel_sovereign._async_ownership import await_owned_task
        pending = set(self._pending_read_executions)
        if pending:
            # Rollback above is a native queue fence: execute's worker call
            # has returned. A substituted executor may still be suspended
            # after that return. Do not make physical receipt harvesting an
            # unbounded wait; give it the existing drain deadline, then ask
            # only those unfinished read owners to unwind once.
            _, pending = await asyncio.wait(
                pending,
                timeout=max(0.0, self._cancelled_write_drain_remaining(started_at)),
            )
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.sleep(0)
        for task in tuple(self._pending_read_executions):
            if not task.done():
                close_failure = _CancelledWriteDrainDeadlineExceeded(
                    "SQLite native read owner is still pending after its drain deadline"
                )
                continue
            outcome = await await_owned_task(task)
            if outcome.error is not None and is_execution_control_error(outcome.error):
                raise outcome.error
            cursor = outcome.result
            if cursor is not None:
                try:
                    await cursor.close()
                except (sqlite3.Error, ValueError) as exc:
                    if is_execution_control_error(exc):
                        raise
                    close_failure = exc
                    continue
            self._pending_read_executions.discard(task)
        for cursor in list(self._abandoned_read_cursors):
            try:
                await cursor.close()
            except (sqlite3.Error, ValueError) as exc:
                if is_execution_control_error(exc):
                    raise
                close_failure = exc
                logger.warning(
                    "Could not close a cursor a cancelled read abandoned on "
                    "%s; it stays recorded for the drain to retry",
                    self.db_path,
                    exc_info=True,
                )
            else:
                self._abandoned_read_cursors.discard(cursor)
        released = await conn.cursor()
        await released.close()
        return close_failure

    async def _rollback_after_failure(
        self, conn: aiosqlite.Connection, *, error: BaseException | None = None,
    ) -> None:
        """Rollback inline, handing off if cancellation reaches cleanup."""
        try:
            await conn.rollback()
        except asyncio.CancelledError as cleanup_error:
            self._handoff_cancelled_write(conn)
            terminal = execution_terminal_error(error, cleanup_error)
            if terminal is not cleanup_error:
                # The selected error can itself be a legacy QueryError whose
                # explicit cause carries authority/COMMIT evidence. Keep that
                # chain; Python retains this cleanup failure as __context__.
                raise terminal
            raise
        except BaseException as cleanup_error:
            terminal = execution_terminal_error(error, cleanup_error)
            if terminal is not cleanup_error:
                raise terminal
            raise

    def _execution_commit_scopes(self):
        """Every original admission whose native file write this owner holds."""
        scopes = current_execution_custody(self)
        if self._identity_anchor_access is not None:
            scopes = (*self._identity_anchor_access.scopes, *scopes)
        return tuple(dict.fromkeys(scopes))

    async def _commit_write(self, conn: aiosqlite.Connection) -> None:
        """One native COMMIT handoff, including host/local-anchor receipts."""
        scopes = self._execution_commit_scopes()
        if not scopes:
            await conn.commit()
            return
        self._require_execution_backend()
        try:
            await conn.commit()
        except BaseException as error:
            # aiosqlite queues COMMIT on its native worker. A lost await/ACK
            # cannot prove rollback; latch every original admission first.
            outcome = execution_commit_outcome(error) or "unknown"
            for scope in scopes:
                scope.preserve_commit_uncertainty(outcome)
            if execution_commit_outcome(error) is not None:
                raise
            raise ExecutionCommitOutcomeError(outcome) from error
        try:
            self._require_execution_backend()
            for scope in scopes:
                scope.require_work()
        except BaseException as error:
            outcome = execution_commit_outcome(error) or "committed"
            for scope in scopes:
                scope.preserve_commit_uncertainty(outcome)
            if execution_commit_outcome(error) is not None:
                raise
            raise ExecutionCommitOutcomeError(outcome) from error

    async def execute(self, query: str, params: Params = ()) -> int:
        """Execute a write query."""
        if self.owns_open_transaction:
            reject_transaction_control(query, dialect="sqlite")
        record_write_query(query)
        conn = self._ensure_connected()
        async with self._write_guard():
            with self._write_operation():
                try:
                    cursor = await conn.execute(query, params)
                    self._require_execution_backend()
                    if not self._in_transaction:
                        await self._commit_write(conn)
                    return cursor.rowcount
                except asyncio.CancelledError:
                    if not self._in_transaction:
                        self._handoff_cancelled_write(conn)
                    raise
                except Exception as e:
                    if not self._in_transaction:
                        await self._rollback_after_failure(conn, error=e)
                    if is_execution_control_error(e):
                        raise
                    raise QueryError(f"Query failed: {e}\nQuery: {query}") from e
                except BaseException as error:
                    if not self._in_transaction:
                        await self._rollback_after_failure(conn, error=error)
                    raise

    async def execute_many(self, query: str, params_list: List[Params]) -> int:
        """Execute query with multiple parameter sets."""
        if not params_list:
            return 0
        if self.owns_open_transaction:
            reject_transaction_control(query, dialect="sqlite")
        record_write_query(query)
        conn = self._ensure_connected()
        async with self._write_guard():
            with self._write_operation():
                try:
                    cursor = await conn.executemany(query, params_list)
                    self._require_execution_backend()
                    if not self._in_transaction:
                        await self._commit_write(conn)
                    return cursor.rowcount
                except asyncio.CancelledError:
                    if not self._in_transaction:
                        self._handoff_cancelled_write(conn)
                    raise
                except Exception as e:
                    if not self._in_transaction:
                        await self._rollback_after_failure(conn, error=e)
                    if is_execution_control_error(e):
                        raise
                    raise QueryError(f"Query failed: {e}\nQuery: {query}") from e
                except BaseException as error:
                    if not self._in_transaction:
                        await self._rollback_after_failure(conn, error=error)
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
        if self.owns_open_transaction:
            reject_transaction_control(query, dialect="sqlite")
        cancelled = False
        body_error = None
        try:
            if conn is self._connection:
                from kestrel_sovereign._async_ownership import create_owned_task
                execute_task = create_owned_task(conn.execute(query, params))
                self._pending_read_executions.add(execute_task)
                try:
                    # Do not cancel the native cursor-return owner. If this
                    # caller unwinds, the cancellation drain closes that exact
                    # returned statement before permitting another write.
                    cursor = await asyncio.shield(execute_task)
                except asyncio.CancelledError as error:
                    # shield synthesizes a cancellation when its child raises
                    # CancelledError. Recover that child's original receipt;
                    # neither its cause nor an uncertain commit may disappear.
                    from kestrel_sovereign._async_ownership import owned_task_error
                    original = owned_task_error(execute_task) if execute_task.done() else None
                    raise execution_terminal_error(original, error)
                except BaseException:
                    self._pending_read_executions.discard(execute_task)
                    raise
                else:
                    self._pending_read_executions.discard(execute_task)
            else:
                cursor = await conn.execute(query, params)
            result = await (cursor.fetchone() if one else cursor.fetchall())
            # Retain the cursor until its close is acknowledged. Cancellation
            # during queued close must hand this same statement to the drain.
            await cursor.close()
            cursor = None
            if one:
                return None if result is None else tuple(result)
            return [tuple(row) for row in result]
        except asyncio.CancelledError as error:
            body_error = error
            cancelled = True
            if conn is self._connection:
                if cursor is not None:
                    self._abandoned_read_cursors.add(cursor)
                if not self._in_transaction:
                    self._handoff_cancelled_write(conn)
            else:
                # A snapshot is private to this read, so it needs no shared
                # transaction fence. Interrupt its current SQLite operation so
                # _read_connection can close the owned worker without waiting
                # for the abandoned query to finish naturally.
                try:
                    await conn.interrupt()
                except BaseException as cleanup_error:
                    logger.exception(
                        "Failed to interrupt cancelled SQLite snapshot read"
                    )
                    raise execution_terminal_error(error, cleanup_error)
            raise
        except BaseException as error:
            body_error = error
            raise
        finally:
            # Ordinary failures still close synchronously. Cancellation skips
            # this queue operation: the retained rollback (or enclosing
            # transaction) is the terminal worker marker for the abandoned
            # cursor operation.
            if cursor is not None and not cancelled:
                try:
                    await cursor.close()
                except BaseException as cleanup_error:
                    raise execution_terminal_error(body_error, cleanup_error)

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
        if self._execution_commit_scopes() and not self.owns_open_transaction:
            # sqlite3.executescript implicitly commits outside our receipt
            # handoff. Bound host/anchor writes use the SAME owning transaction.
            async with self.transaction():
                for statement in _owned_script_statements(script):
                    await self.execute(statement)
            return
        record_write_script(script)
        conn = self._ensure_connected()
        async with self._write_guard():
            with self._write_operation():
                try:
                    if self._in_transaction:
                        for statement in _owned_script_statements(script):
                            cursor = await conn.execute(statement)
                            await cursor.close()
                    else:
                        await conn.executescript(script)
                    if not self._in_transaction:
                        await self._commit_write(conn)
                except asyncio.CancelledError:
                    if not self._in_transaction:
                        self._handoff_cancelled_write(conn)
                    raise
                except Exception as e:
                    if not self._in_transaction:
                        await self._rollback_after_failure(conn, error=e)
                    if is_execution_control_error(e):
                        raise
                    raise QueryError(f"Script execution failed: {e}") from e
                except BaseException as error:
                    if not self._in_transaction:
                        await self._rollback_after_failure(conn, error=error)
                    raise

    @asynccontextmanager
    async def transaction(self, *, immediate: bool = False, savepoint: bool = False) -> AsyncIterator[None]:
        """Transaction context manager.

        ``BEGIN IMMEDIATE`` is reserved for one-time schema migrations that
        must acquire SQLite's writer slot before inspecting a migration marker.
        Normal data mutations retain deferred ``BEGIN`` so read-first flows do
        not unnecessarily block one another.
        """
        conn = self._ensure_connected()

        if self._in_transaction and self._txn_owner is asyncio.current_task():
            self._require_native_transaction()
            if not savepoint:
                yield
                self._require_native_transaction()
                return
            # Explicit independently rollback-safe unit inside a joined owner.
            # Other callers retain the established joined-scope semantics.
            self._savepoint_counter += 1
            name = f"kestrel_scope_{self._savepoint_counter}"
            await conn.execute(f"SAVEPOINT {name}")
            try:
                yield
                self._require_native_transaction()
                await conn.execute(f"RELEASE SAVEPOINT {name}")
            except BaseException as error:
                if conn.in_transaction and not self._transaction_poisoned:
                    try:
                        await conn.execute(f"ROLLBACK TO SAVEPOINT {name}")
                        await conn.execute(f"RELEASE SAVEPOINT {name}")
                    except BaseException as cleanup_error:
                        self._transaction_poisoned = True
                        raise execution_terminal_error(error, cleanup_error)
                raise
            return

        # Hold the write lock for the whole BEGIN..COMMIT/ROLLBACK span so the
        # transaction is one atomic write unit against concurrent writers
        # sharing this connection (#1675).
        async with self._write_guard():
            with self._write_operation():
                self._in_transaction = True
                self._transaction_poisoned = False
                self._txn_owner = asyncio.current_task()
                try:
                    await conn.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
                    yield
                    self._require_native_transaction()
                    await self._commit_write(conn)
                except asyncio.CancelledError:
                    self._handoff_cancelled_write(conn)
                    raise
                except Exception as e:
                    await self._rollback_after_failure(conn, error=e)
                    if is_execution_control_error(e):
                        raise
                    raise TransactionError(f"Transaction failed: {e}") from e
                except BaseException as error:
                    await self._rollback_after_failure(conn, error=error)
                    raise
                finally:
                    self._in_transaction = False
                    self._transaction_poisoned = False
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
