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
from collections.abc import Callable
from collections.abc import Sequence as SequenceABC
from contextlib import asynccontextmanager, nullcontext
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator, List, Optional, Sequence, Tuple

from kestrel_sovereign._async_ownership import (
    OwnedTaskAdmission,
    await_owned_task,
    create_owned_task,
)
from kestrel_sovereign.execution_custody import (
    AcknowledgedExecutionCommit,
    ExecutionAuthorityError,
    ExecutionCommitOutcomeError,
    ExecutionCustody,
    NativeExecutionRetirement,
    bind_execution_cleanup,
    current_execution_custody,
    execution_commit_outcome,
    execution_denial_scopes,
    execution_error_with_cause,
    execution_terminal_error,
    is_execution_control_error,
    lock_execution_authority,
    require_execution_backend,
    require_execution_work,
    retain_acknowledged_execution_commit,
)
from kestrel_sovereign.native_retirement import (
    _begin_native_resource,
    _native_observation_requested,
    _require_native_observation_for_connection,
    _settle_native_observer_enrollment,
)

from .after_commit import (
    AfterCommitCallback,
    AfterCommitCallbacks,
    run_after_commit,
)
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
from .transaction_control import reject_transaction_control
from .write_audit import record_write_query, record_write_script

logger = logging.getLogger(__name__)


class PostgresSessionLostError(TransactionError):
    """The original session closed before this transaction accepted COMMIT.

    Emitted only after the original native rollback owner has finished. The
    enclosing checkout still owns its exact pool reset and may add retirement
    control before exposing this error. This disposition neither settles an
    accepted COMMIT nor replaces an execution admission.
    """


def _failure_detail(error: BaseException) -> str:
    outcome = execution_commit_outcome(error)
    if outcome is not None:
        return (f"{error}; execution transaction outcome is {outcome}; "
                "the operation may have committed; reconcile durable state before retry")
    return str(error)

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


@dataclass
class _ExecutionCommitState:
    outcome: str | None = None
    scopes: tuple[ExecutionCustody, ...] = ()
    acknowledgments: list[AcknowledgedExecutionCommit] = field(default_factory=list)
    retirement: NativeExecutionRetirement = field(default_factory=NativeExecutionRetirement)
    native_owned: bool = True
    connection: Any = None
    checkout_open: bool = False
    terminal: BaseException | None = None
    body_failure: BaseException | None = None
    native_observation: Any = None

    def remember(self, scopes: tuple[ExecutionCustody, ...]) -> None:
        # A shared checkout can begin unbound and participate in several
        # admitted operations. Retain each admission until physical release.
        self.scopes += tuple(scope for scope in scopes if scope not in self.scopes)

    def retain_acknowledgments(self, error: BaseException) -> BaseException:
        for receipt in self.acknowledgments:
            retain_acknowledged_execution_commit(error, receipt)
        return error


@dataclass(eq=False)
class _NativePostgresRetirement:
    """Exact failed lease/transaction and connection, not authority to retry.

    asyncpg clears a checkout's connection BEFORE attempting pool release.
    Retain the original connection independently of that mutable projection.
    The original proxy's attachment, not a raw connection reference, governs
    whether aborting the still-held lease is safe; it may already be returned.
    """

    context_manager: Any
    connection: Any
    native_connection: Any


class PostgresPoolAdapter:
    """Cooperating caller-context wrapper around an exact native asyncpg pool.

    ``wrapper`` receives a managed native checkout context manager. It must
    enter that manager and yield its SAME connection. Both wrapper halves and
    their ContextVar tokens run in the caller; Core owns only native epilogues.
    Merely exposing an opaque wrapper's underlying pool is not this contract.
    The pool is borrowed, never closed or replaced by the adapter.
    """

    def __init__(self, pool: Any, wrapper: Callable):
        if not isinstance(pool, asyncpg.Pool):
            raise TypeError("PostgresPoolAdapter requires the original native asyncpg pool")
        self.native_pool = pool
        self.wrap_checkout = wrapper

    def __getattr__(self, name):
        return getattr(self.native_pool, name)


@dataclass
class _OperationalSessionLease:
    """One pool connection shared only by a context and its owned children."""

    connection: Any
    lock: asyncio.Lock
    active: bool = True
    commit_state: _ExecutionCommitState = field(default_factory=_ExecutionCommitState)


@dataclass(frozen=True)
class TerminalAdvisoryBarrier:
    """Exact exclusive keys for a physical terminal barrier, NOT work authority.

    A host's fixed terminal API must independently validate original durable
    target/operation permission. This request supplies no executor, connection,
    callback or claim, and cannot initialize a disconnected backend.
    """

    keys: tuple[tuple[int, int], ...]
    backend_type: str = field(default="postgres", init=False)

    def __post_init__(self) -> None:
        if type(self.keys) is not tuple or not self.keys:
            raise ValueError("terminal advisory barrier requires nonempty immutable keys")
        for pair in self.keys:
            if type(pair) is not tuple or len(pair) != 2:
                raise TypeError("terminal advisory keys must be immutable integer pairs")
            if any(type(value) is not int for value in pair):
                raise TypeError("terminal advisory keys must be signed 32-bit integers, not bool")
            if any(not -(2**31) <= value < 2**31 for value in pair):
                raise ValueError("terminal advisory keys must be signed 32-bit integers")

    def require_work(self) -> None:
        raise ExecutionAuthorityError("terminal physical barrier never admits ordinary work")

    async def lock_and_validate(self, connection: Any) -> None:
        self.require_work()


@dataclass(frozen=True)
class _TerminalAdvisoryLifetime:
    """Finite original connected-backend identities across awaited checkout."""

    backend: Any
    pool: Any
    delegate: Any

    @classmethod
    def capture(cls, backend: Any) -> "_TerminalAdvisoryLifetime":
        original = cls(backend, backend._pool, backend._advisory_backend)
        original.require_live()
        return original

    def require_live(self) -> None:
        if (self.pool is None or self.backend._pool is not self.pool
            or self.backend._advisory_backend is not self.delegate
            or (self.delegate is not None and
                (self.delegate._pool is not self.pool or self.delegate._advisory_backend is not None))):
            raise ConnectionError("terminal advisory barrier requires the original connected backend and shared owner")
        native = self.pool.native_pool if isinstance(self.pool, PostgresPoolAdapter) else self.pool
        if asyncpg is None or not isinstance(native, asyncpg.Pool) or native.is_closing():
            raise ConnectionError("terminal advisory barrier requires the original connected backend native pool")

    def require_backend(self, backend: Any) -> None:
        self.require_live()
        if backend is not self.backend and backend is not self.delegate:
            raise ConnectionError("terminal advisory barrier cannot substitute its backend")


class AdvisoryLease:
    """Exact physical-session capability, never revived by reconnection.

    Listeners are registered during acquisition, not by the body after entry.
    PostgreSQL/asyncpg can queue a loss notification; retirement is monotonic
    and a queued callback cannot turn normal completed retirement into loss.
    The consumer must settle its owned work before leaving the lease context.
    """

    def __init__(
        self, connection: Any,
        on_loss: Callable[["AdvisoryLease"], None] | None,
    ) -> None:
        self.backend_pid: int = connection.get_server_pid()
        self._connection = connection
        self._on_loss = on_loss
        self._state = "acquiring"

    @property
    def lost(self) -> bool:
        return self._state == "lost"

    def _connection_terminated(self, _connection: Any) -> None:
        if self._state in ("acquiring", "held"):
            self._state = "lost"
            if self._on_loss is not None:
                self._on_loss(self)

    def require_live(self) -> None:
        self.require_physical()

    def require_physical(self) -> None:
        """Validate this SAME original session, without admitting new work."""
        if self._state in ("acquiring", "held"):
            try:
                closed = self._connection.is_closed()
            except asyncpg.InterfaceError:
                # asyncpg detaches a dead physical session's public pool proxy
                # before its queued termination listener necessarily runs.
                closed = True
            if closed:
                self._connection_terminated(self._connection)
        if self._state not in ("acquiring", "held"):
            raise ExecutionAuthorityError(
                f"PostgreSQL advisory authority session {self.backend_pid} "
                f"is {self._state}"
            )

    def _admit(self) -> None:
        self.require_physical()
        self._state = "held"

    def _retire(self) -> None:
        if self._state != "lost":
            self._state = "retired"

    def _remove_listener(self) -> bool:
        """False means asyncpg already detached this dead session's proxy."""

        try:
            self._connection.remove_termination_listener(self._connection_terminated)
        except asyncpg.InterfaceError:
            # Physical session death clears its listeners and releases its
            # pool holder. There is no live session to unlock/terminate, and
            # probing that invalid proxy must not mask the original loss.
            return False
        return True


class TerminalAdvisoryLease(AdvisoryLease):
    """Physical exclusion only; even a new ordinary fence cannot upgrade it."""

    def __init__(self, connection: Any, on_loss: Callable[[AdvisoryLease], None] | None,
                 *, lifetime: _TerminalAdvisoryLifetime) -> None:
        super().__init__(connection, on_loss)
        self._lifetime = lifetime
        self._loss_scopes = current_execution_custody(lifetime.backend)

    def _connection_terminated(self, connection: Any) -> None:
        # asyncpg dispatches in the transport's creation context, not listener
        # registration context. A reused ordinary session predates this barrier.
        # Keep the original scope union and cleanup-only admission for callbacks
        # and their children; never restore an unrestricted transport context.
        with bind_execution_cleanup(self._lifetime.backend, self._loss_scopes):
            super()._connection_terminated(connection)

    def require_physical(self) -> None:
        self._lifetime.require_live()
        super().require_physical()

    def require_live(self) -> None:
        raise ExecutionAuthorityError("terminal advisory barrier never admits ordinary work")


def is_concurrent_update_error(exc: BaseException) -> bool:
    """True for the transient 'tuple concurrently updated' race (#1805)."""
    return _CONCURRENT_UPDATE_MARKER in str(exc).lower()


# Compatibility for existing internal callers/tests while new transactional
# users consume the public policy helper below.
_is_concurrent_update_error = is_concurrent_update_error


def concurrent_write_retry_delay(
    exc: BaseException,
    retries_done: int,
) -> Optional[float]:
    """Return the shared backoff for a retryable PostgreSQL write conflict."""
    if (
        not is_concurrent_update_error(exc)
        or retries_done >= _CONCURRENT_WRITE_RETRIES
    ):
        return None
    return _CONCURRENT_WRITE_BACKOFF_S * (retries_done + 1)


async def _prove_committable(conn: Any) -> None:
    """Refuse to mint a committed fact for an aborted transaction.

    A statement that fails inside a PostgreSQL transaction aborts it. If the
    caller swallows that failure, COMMIT rolls the transaction back and
    asyncpg reports no error, so ``call_after_commit`` work would run for
    writes that never persisted (#2913). One trivial statement first: on an
    aborted transaction it raises, the transaction rolls back as a failure,
    and no committed receipt or callback is published.
    """
    await conn.execute("SELECT 1")


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
        advisory_max_pool_size: Optional[int] = None,
        execution_custody: ExecutionCustody | None = None,
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
            max_pool_size: Maximum operational pool connections.
            advisory_max_pool_size: Optional advisory-lock pool bound. A
                multi-tenant host sharing this backend should set an explicit
                host-wide bound sized for its advisory-lock workload.
        """
        if not ASYNCPG_AVAILABLE:
            raise ImportError(
                "asyncpg is not installed. Install with: pip install asyncpg"
            )
        
        self._dsn = dsn
        self._execution_custody = execution_custody
        self._host = host
        self._port = port
        self._database = database
        self._user = user
        self._password = password
        self._min_pool_size = min_pool_size
        self._max_pool_size = max_pool_size
        # Session advisory locks can span work performed through the ordinary
        # pool. Keep a *bounded*, separate pool so a waiter cannot consume the
        # last operational connection needed to finish that work; an unbounded
        # direct connection per waiter would merely trade a deadlock for
        # connection exhaustion. Standalone agents retain the historical cap
        # of four; shared hosts explicitly budget this host-wide pool.
        if advisory_max_pool_size is not None and (
            type(advisory_max_pool_size) is not int or advisory_max_pool_size < 1
        ):
            raise ValueError("advisory_max_pool_size must be a positive integer")
        self._advisory_max_pool_size = (
            advisory_max_pool_size
            if advisory_max_pool_size is not None
            else max(1, min(max_pool_size, 4))
        )
        self._advisory_dsn = dsn
        self._advisory_connect_args: tuple[Any, ...] = (dsn,) if dsn else ()
        self._advisory_connect_kwargs: dict[str, Any] = {}
        self._advisory_recipe_available = self._has_advisory_connection_recipe(
            self._advisory_connect_args,
            self._advisory_connect_kwargs,
        )
        self._advisory_pool: Optional[asyncpg.Pool] = None
        self._advisory_pool_lock = asyncio.Lock()
        self._advisory_backend: Optional["PostgresBackend"] = None
        
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
        self._operational_conn_var: "contextvars.ContextVar" = (
            contextvars.ContextVar("pg_operational_conn", default=None)
        )
        self._owns_pool = True  # We own pools we create
    
    @classmethod
    def from_pool(
        cls,
        pool: "asyncpg.Pool | PostgresPoolAdapter",
        *,
        advisory_dsn: Optional[str] = None,
        advisory_connect_kwargs: Optional[dict[str, Any]] = None,
        advisory_backend: Optional["PostgresBackend"] = None,
        advisory_max_pool_size: Optional[int] = None,
        execution_custody: ExecutionCustody | None = None,
    ) -> "PostgresBackend":
        """
        Create a PostgresBackend from an existing asyncpg pool.

        This is useful when you want to reuse an existing connection pool
        (e.g., from an app's pg_pool) rather than creating a new one.

        Note: close() will NOT close the pool since we don't own it.

        Args:
            pool: Existing asyncpg.Pool or cooperating PostgresPoolAdapter.
                Opaque acquire wrappers retain ordinary caller-context
                behavior, but cannot admit execution custody. Compose their
                context behavior around the adapter's managed native lease.
            advisory_dsn: Optional explicit DSN for the bounded dedicated
                advisory-lock pool. When omitted, connection construction
                settings are copied from asyncpg's pool construction context.
            advisory_connect_kwargs: Connection options for the dedicated
                advisory-lock pool, such as a non-default ``search_path``.
            advisory_backend: Trusted host backend whose bounded advisory pool
                this child may share. The delegate must wrap this exact
                operational pool, must own its advisory pool directly, and is
                owned by the caller. It cannot be combined with a connection
                recipe supplied to this child.
            advisory_max_pool_size: Optional explicit bound for a directly
                owned advisory pool. Delegating children inherit the host's
                bound and cannot override it.

        Returns:
            PostgresBackend wrapping the pool
        """
        instance = cls.__new__(cls)
        if advisory_backend is not None:
            if (
                advisory_dsn is not None
                or advisory_connect_kwargs is not None
                or advisory_max_pool_size is not None
            ):
                raise ValueError(
                    "advisory_backend cannot be combined with an advisory "
                    "connection recipe"
                )
            if not isinstance(advisory_backend, cls):
                raise TypeError("advisory_backend must be a PostgresBackend")
            if advisory_backend._pool is not pool:
                raise ValueError(
                    "advisory_backend must wrap the same operational pool"
                )
            if advisory_backend._advisory_backend is not None:
                raise ValueError(
                    "advisory_backend must own its advisory pool directly"
                )
        # Preserve the wrapped pool's no-DSN contract for unrelated consumers
        # (for example SQLAlchemy session factories). The scheduler-only
        # dedicated advisory pool has its own explicit connection source.
        instance._dsn = None
        instance._execution_custody = execution_custody
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
        if advisory_max_pool_size is not None and (
            type(advisory_max_pool_size) is not int or advisory_max_pool_size < 1
        ):
            raise ValueError("advisory_max_pool_size must be a positive integer")
        instance._advisory_max_pool_size = (
            advisory_max_pool_size
            if advisory_max_pool_size is not None
            else max(1, min(operational_max, 4))
        )
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
        instance._advisory_backend = advisory_backend
        instance._pool = pool
        instance._txn_conn_var = contextvars.ContextVar("pg_txn_conn", default=None)
        instance._operational_conn_var = contextvars.ContextVar(
            "pg_operational_conn", default=None
        )
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

    def _transaction_custody_context(self) -> contextvars.ContextVar:
        """Per-backend, owner-task ledger of every admission touching a commit."""

        var = getattr(self, "_txn_custody_var", None)
        if not isinstance(var, contextvars.ContextVar):
            var = contextvars.ContextVar("pg_transaction_execution_custody", default=None)
            self._txn_custody_var = var
        return var

    def _transaction_custody(self) -> list[ExecutionCustody] | None:
        entry = self._transaction_custody_context().get()
        if entry is None or entry[0] is not asyncio.current_task():
            return None
        return entry[1]

    def _remember_transaction_custody(self) -> tuple[ExecutionCustody, ...]:
        current = current_execution_custody(self)
        retained = self._transaction_custody()
        if retained is None:
            return current
        if any(scope not in retained for scope in current):
            # Authority must precede every graph/file lock. Adding an
            # admission after the transaction began would invert that order
            # and could both deadlock and erase a revoked child at commit.
            raise ExecutionAuthorityError(
                "execution admissions cannot be added inside an open transaction"
            )
        return tuple(retained)

    async def _lock_transaction_custody(self, connection: Any) -> None:
        await lock_execution_authority(
            connection, "postgres", self._remember_transaction_custody(),
        )

    def _require_transaction_custody(self) -> None:
        require_execution_backend("postgres", self._remember_transaction_custody())

    def _current_operational_lease(self) -> Optional[_OperationalSessionLease]:
        """Return the live operational lease inherited by this task, if any."""

        operational_var = getattr(self, "_operational_conn_var", None)
        if operational_var is None:
            return None
        lease = operational_var.get()
        if isinstance(lease, _OperationalSessionLease) and lease.active:
            return lease
        return None

    @asynccontextmanager
    async def _operational_query_connection(self) -> AsyncIterator[Any]:
        """Serialize one non-transactional query on the inherited lease."""

        lease = self._current_operational_lease()
        if lease is None:
            yield None
            return
        async with lease.lock:
            # The owning context can start closing while an inherited child is
            # waiting. Never use a connection after that context released it.
            if not lease.active:
                yield None
                return
            yield lease.connection

    @property
    def backend_type(self) -> str:
        return "postgres"

    @property
    def nested_transaction_strategy(self) -> str:
        """PostgreSQL isolates same-task nested transactions with savepoints."""
        return "savepoint"

    @property
    def owns_open_transaction(self) -> bool:
        """Whether the current task has a transaction open here, which a
        ``transaction()`` entered now would join (as a savepoint) rather than
        commit."""
        return self._current_txn_conn() is not None

    def call_after_commit(self, callback: AfterCommitCallback) -> None:
        """Run *callback* once this task's writes so far have committed.

        Inside a transaction this task owns, it runs after that transaction
        commits and never if it rolls back; a savepoint that rolls back
        discards what was registered inside it. With no transaction open the
        writes committed on their own, so it runs now.
        """
        if self.owns_open_transaction:
            self._current_after_commit().register(callback)
        else:
            run_after_commit(callback)

    def _after_commit_context(self) -> "contextvars.ContextVar":
        """``(owner_task, AfterCommitCallbacks)`` for this task's transaction.

        Created on first use rather than in ``__init__``: like
        ``_operational_conn_var``, it must exist on instances built without
        running ``__init__`` (``PoolBackendAdapter`` wraps an existing pool),
        or every transaction on them would fail.
        """
        var = getattr(self, "_after_commit_var", None)
        if var is None:
            var = contextvars.ContextVar("pg_after_commit", default=None)
            self._after_commit_var = var
        return var

    def _current_after_commit(self) -> AfterCommitCallbacks:
        """What this task's open transaction runs once it commits.

        Set with the transaction's connection in :meth:`transaction`, so it
        exists whenever :meth:`_current_txn_conn` does.
        """
        owner, callbacks = self._after_commit_context().get()
        if owner is not asyncio.current_task():
            raise RuntimeError("this task does not own the open transaction")
        return callbacks

    @property
    def is_connected(self) -> bool:
        return self._pool is not None

    @property
    def operational_pool(self) -> "asyncpg.Pool":
        """Return the connected pool for trusted same-process host wiring."""

        return self._ensure_connected()
    
    async def connect(self) -> None:
        """Connect to PostgreSQL and create connection pool."""
        require_execution_backend("postgres", current_execution_custody(self))
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
            raise execution_error_with_cause(ConnectionError(f"Failed to connect to PostgreSQL: {e}"), e)
    
    async def close(self) -> None:
        """Join independent original pool closers, retaining refused owners.

        Never clear a handle until its own accepted closer physically succeeds.
        Task-factory refusal is native retirement debt for that exact pool, not
        an ordinary close failure or permission to skip the independent pool.
        Borrowed primary pools are only detached; their caller retains custody.
        """
        errors: list[BaseException] = []
        for attribute in ("_advisory_pool", "_pool"):
            pool = getattr(self, attribute)
            if pool is None:
                continue
            if attribute == "_pool" and not self._owns_pool:
                if self._pool is pool:
                    self._pool = None
                continue
            errors.append(await self._retire_owned_pool(attribute, pool))
        terminal = execution_terminal_error(
            *errors, retain_originals=True, prefer_cancellation=True,
            isolate_control_originals=True,
        )
        if terminal is not None:
            raise terminal

    async def _retire_owned_pool(self, attribute: str, pool: Any) -> BaseException | None:
        """One exact pool closer for backend exit and refused new allocation."""
        from kestrel_sovereign._async_ownership import (
            owned_task_admission_error,
            owned_task_error,
        )

        owners = vars(self).setdefault("_pool_close_owners", {})
        record = owners.get(attribute)
        if record is None or record[0] is not pool:
            admission = OwnedTaskAdmission()
            record = [pool, None, None]
            owners[attribute] = record
            try:
                record[1] = create_owned_task(
                    pool.close(), retirement_resource=pool, retain_originals=True,
                    admission=admission,
                )
            except BaseException as error:
                record[1] = admission.task
                record[2] = owned_task_admission_error(pool, admission, error)
        task, opening = record[1:]
        if task is None:
            return opening
        outcome = await await_owned_task(task)
        physical = owned_task_error(task)
        if opening is None and physical is None and getattr(self, attribute) is pool:
            setattr(self, attribute, None)
            owners.pop(attribute, None)
        elif opening is None and physical is not None and not is_execution_control_error(physical):
            # Preserve the ordinary joined availability-failure retry contract;
            # refused/unpublished or uncertain owners never take this branch.
            if owners.get(attribute) is record:
                owners.pop(attribute, None)
        return execution_terminal_error(
            opening, outcome.error, outcome.cancellation, retain_originals=True,
            prefer_cancellation=True, isolate_control_originals=True,
        )
    
    def _ensure_connected(self) -> asyncpg.Pool:
        """Ensure we have an active pool."""
        require_execution_backend("postgres", current_execution_custody(self))
        if self._pool is None:
            raise ConnectionError("Not connected to database. Call connect() first.")
        return self._pool

    async def _ensure_advisory_pool(
        self, *, terminal_lifetime: _TerminalAdvisoryLifetime | None = None,
    ) -> "asyncpg.Pool":
        """Return the bounded pool used only for long advisory-lock gates."""

        # Retain the normal backend lifecycle check: an advisory pool is not a
        # hidden replacement for a closed primary database backend.
        if terminal_lifetime is None:
            self._ensure_connected()
        elif type(terminal_lifetime) is not _TerminalAdvisoryLifetime:
            raise TypeError("terminal advisory allocation requires its typed physical barrier")
        else:
            terminal_lifetime.require_backend(self)
        if self._advisory_backend is not None:
            return await self._advisory_backend._ensure_advisory_pool(terminal_lifetime=terminal_lifetime)
        if self._advisory_pool is not None:
            return self._advisory_pool
        async with self._advisory_pool_lock:
            if terminal_lifetime is not None:
                terminal_lifetime.require_backend(self)
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
                raise execution_error_with_cause(ConnectionError(
                    f"Failed to create dedicated PostgreSQL advisory-lock pool: {exc}"
                ), exc)
            self._advisory_pool = pool
            if terminal_lifetime is not None:
                try:
                    terminal_lifetime.require_backend(self)
                except ConnectionError as refusal:
                    cleanup = await self._retire_owned_pool("_advisory_pool", pool)
                    raise execution_terminal_error(
                        refusal, cleanup, retain_originals=True,
                        isolate_control_originals=True,
                    )
            return pool
    
    def _convert_query(self, query: str) -> str:
        """Convert SQLite-style ? placeholders to PostgreSQL $N style."""
        if self._current_txn_conn() is not None or current_execution_custody(self) or _native_observation_requested():
            reject_transaction_control(query, dialect="postgres")
        converted, _ = sqlite_to_postgres(query)
        return converted

    @asynccontextmanager
    async def _execution_query_executor(self, executor: Any, commit_state: _ExecutionCommitState | None = None) -> AsyncIterator[Any]:
        """Fence every native SQL surface on its own transaction/session.

        Unfenced callers retain their normal autocommit path. A fenced read is
        guarded too: SELECT can invoke mutating functions, and stale reads can
        disclose old-owner data. Never classify authority from a SQL verb.
        """

        await _settle_native_observer_enrollment()
        scopes = self._remember_transaction_custody()
        if _native_observation_requested() and executor is not self._pool:
            _require_native_observation_for_connection(executor)
        if not scopes:
            if _native_observation_requested() and executor is self._pool:
                # Observe the real checkout/reset without turning legacy
                # autocommit SQL (including CONCURRENTLY DDL) into a new
                # explicit transaction. Observation grants no work authority
                # and cannot manufacture a known implicit-COMMIT receipt.
                async with self._authority_pool_checkout(executor) as (connection, _state):
                    yield connection
            else:
                yield executor
            return
        require_execution_backend("postgres", scopes)
        if self._current_txn_conn() is not None:
            await lock_execution_authority(executor, "postgres", scopes)
            yield executor
            self._require_transaction_custody()
            return
        if executor is self._pool:
            async with self._authority_pool_checkout(executor) as (connection, state):
                async with self._execution_query_executor(connection, state) as checked:
                    yield checked
            return
        operational = self._current_operational_lease()
        if commit_state is None and operational is not None and executor is operational.connection:
            commit_state = operational.commit_state
        if commit_state is not None and not commit_state.native_owned:
            raise ExecutionAuthorityError("opaque pool checkout cannot admit execution custody; use PostgresPoolAdapter")
        async with self._authority_transaction(executor, scopes, commit_state):
            yield executor

    @asynccontextmanager
    async def _authority_pool_checkout(self, pool: Any) -> AsyncIterator[tuple[Any, _ExecutionCommitState]]:
        async with self._tracked_pool_checkout(pool, admit_work=True) as owned:
            yield owned

    @staticmethod
    def _abort_native_connection(connection: Any, terminal: BaseException) -> None:
        """Dispose only the original dirty session, retaining any abort failure."""
        try:
            if isinstance(connection, asyncpg.pool.PoolConnectionProxy):
                # A detached proxy no longer names a leased connection. Never
                # call through it or terminate a successor acquired by others.
                connection = connection._con
                if connection is None:
                    return
            if not connection.is_closed():
                connection.terminate()
        except BaseException as failure:
            prior = terminal.__cause__
            terminal.__cause__ = (failure if prior is None else
                BaseExceptionGroup("original native abort failure", [prior, failure]))

    @staticmethod
    def _refused_native_owner(
        connection: Any, resource: Any, refusal: BaseException,
        scopes: tuple[ExecutionCustody, ...], body_failure: BaseException | None,
        native_connection: Any, *, owner: asyncio.Task | None = None,
    ) -> BaseException:
        """No created task means no native closer owner, never successful reset.

        The task factory has refused the epilogue before it can start. Abort
        the ORIGINAL native connection synchronously rather than retrying the
        factory, inventing a completed closer or leaking its live transaction.
        Keep the exact resource and absence of an owner as retirement debt.
        """
        debt = ExecutionAuthorityError("original native epilogue ownership refused")
        debt._kestrel_native_retirement_debts = (
            (_NativePostgresRetirement(resource, connection, native_connection), owner),
        )
        terminal = execution_terminal_error(
            body_failure, refusal, debt, retain_originals=True,
            isolate_control_originals=True,
        )
        PostgresBackend._abort_native_connection(connection, terminal)
        for scope in scopes:
            scope.revoke("original native epilogue ownership refused")
        return terminal

    @asynccontextmanager
    async def _tracked_pool_checkout(self, pool: Any, *, admit_work: bool) -> AsyncIterator[tuple[Any, _ExecutionCommitState]]:
        await _settle_native_observer_enrollment()
        state = _ExecutionCommitState()
        state.remember(current_execution_custody(self))
        if isinstance(pool, PostgresPoolAdapter):
            @asynccontextmanager
            async def managed_native():
                try:
                    async with self._native_pool_checkout(pool.native_pool, state, admit_work=admit_work) as (connection, _):
                        yield connection
                except BaseException as error:
                    state.terminal = error
                    raise
            native = managed_native()
            failure = None
            try:
                async with pool.wrap_checkout(native) as connection:
                    if connection is not state.connection or not state.checkout_open:
                        raise ExecutionAuthorityError("pool adapter did not retain its original native checkout")
                    try:
                        yield connection, state
                    except BaseException as error:
                        # Capture BEFORE an arbitrary wrapper can suppress the
                        # cancellation around its yield inside the native lease.
                        state.body_failure = error
                        raise
            except BaseException as error:
                failure = error
            # A wrapper cannot abandon or suppress the registered native lease.
            # This resumes our own generator in the ORIGINAL caller context;
            # only its explicit native epilogue crosses the task boundary.
            if state.checkout_open:
                try:
                    await native.__aexit__(type(failure) if failure is not None else None,
                        failure, failure.__traceback__ if failure is not None else None)
                except BaseException as cleanup:
                    failure = execution_terminal_error(failure, cleanup, retain_originals=True,
                        isolate_control_originals=True)
            failure = execution_terminal_error(state.body_failure, state.terminal, failure, retain_originals=True,
                isolate_control_originals=True)
            if failure is not None:
                # A new wrapper error AFTER successful reset is still an
                # observer of this checkout's already-proven commit. Attach
                # that fact before classifying, then sample ALL admissions.
                failure = state.retain_acknowledgments(failure)
                if state.outcome is not None and not is_execution_control_error(failure):
                    for scope in state.scopes:
                        scope.preserve_commit_uncertainty(state.outcome)
                    raise execution_error_with_cause(ExecutionCommitOutcomeError(state.outcome), failure)
                if admit_work and state.acknowledgments and execution_commit_outcome(failure) == "committed":
                    failure = self._retain_after_commit_denial(failure, state.scopes, state.acknowledgments)
                raise failure
            # Arbitrary wrapper exit may await AFTER native reset validated.
            # Recheck the retained admissions at the final publication boundary.
            if admit_work:
                if state.outcome is not None:
                    self._require_after_authority_commit(state.scopes, state.acknowledgments)
                else:
                    require_execution_backend("postgres", state.scopes)
            return
        if not isinstance(pool, asyncpg.Pool):
            # Preserve ordinary legacy wrapper context semantics. Strong
            # custody cannot be inferred by inspecting an opaque generator.
            state.native_owned = False
            if state.scopes or not admit_work or _native_observation_requested():
                raise ExecutionAuthorityError("opaque pool checkout requires PostgresPoolAdapter for native custody")
            async with pool.acquire() as connection:
                yield connection, state
            return
        async with self._native_pool_checkout(pool, state, admit_work=admit_work) as owned:
            yield owned

    @asynccontextmanager
    async def _native_pool_checkout(self, pool: Any, state: _ExecutionCommitState, *, admit_work: bool) -> AsyncIterator[tuple[Any, _ExecutionCommitState]]:
        checkout = pool.acquire()
        if type(checkout) is not asyncpg.pool.PoolAcquireContext:
            raise ExecutionAuthorityError("native pool acquire is opaque; use PostgresPoolAdapter")
        observation = _begin_native_resource(checkout)
        state.native_observation = observation
        failure = None
        try:
            async with self._native_pool_checkout_body(checkout, state, admit_work=admit_work) as owned:
                yield owned
        except BaseException as error:
            failure = error
            raise
        finally:
            if observation is not None:
                observation.finish(failure)

    @asynccontextmanager
    async def _native_pool_checkout_body(self, checkout: Any, state: _ExecutionCommitState, *, admit_work: bool) -> AsyncIterator[tuple[Any, _ExecutionCommitState]]:
        # asyncpg release/reset is awaited after COMMIT. Carry the exact commit
        # state through that await so a release failure cannot imply rollback.
        body_failure: BaseException | None = None
        connection = await checkout.__aenter__()
        native_connection = connection._con
        observation = state.native_observation
        if observation is not None:
            observation.connection, observation.native_connection = connection, native_connection
        state.connection = connection
        state.checkout_open = True
        try:
            yield connection, state
        except BaseException as error:
            body_failure = error
        finally:
            state.checkout_open = False
        # Only the native reset/return is protected. The checkout body remains
        # in its original task, including every ContextVar token and arbitrary
        # caller await. Cancellation cannot abandon an accepted native reset.
        with observation.phase(checkout) if observation is not None else nullcontext() as phase:
            admission = OwnedTaskAdmission()
            try:
                retirement = create_owned_task(checkout.__aexit__(
                    type(body_failure) if body_failure is not None else None,
                    body_failure,
                    body_failure.__traceback__ if body_failure is not None else None,
                ), name=f"{asyncio.current_task().get_name()}:postgres-original-pool-reset",
                    retain_originals=True, admission=admission)
            except BaseException as refusal:
                accepted = admission.task
                if phase is not None:
                    phase.owner = accepted
                if accepted is not None:
                    joined = await await_owned_task(accepted)
                    refusal = execution_terminal_error(refusal, joined.error, joined.cancellation,
                        retain_originals=True, isolate_control_originals=True)
                terminal = self._refused_native_owner(connection, checkout, refusal, state.scopes, body_failure, native_connection, owner=accepted)
                if phase is not None:
                    phase.record_native_failure(terminal, commit_outcome=state.outcome)
                raise state.retain_acknowledgments(terminal)
            if phase is not None:
                phase.owner = retirement
            state.retirement.owner = retirement
            retired = await await_owned_task(retirement)
            if retired.error is not None:
                if phase is not None:
                    phase.record_native_failure(retired.error, commit_outcome=state.outcome, cancellation=retired.cancellation)
                debt = RuntimeError("original native pool retirement remains unresolved")
                debt._kestrel_native_retirement_debts = (
                    (_NativePostgresRetirement(checkout, connection, native_connection), retirement),
                )
                # Establish private retirement evidence BEFORE either abort or
                # finite composition can decorate an earlier shared receipt.
                terminal = execution_terminal_error(body_failure, retired.error,
                    retired.cancellation, debt, retain_originals=True,
                    isolate_control_originals=True)
                assert terminal is not None
                self._abort_native_connection(connection, terminal)
                for scope in state.scopes:
                    scope.revoke("original native pool retirement failed")
                outcome = execution_commit_outcome(terminal) or state.outcome
                if outcome is not None:
                    for scope in state.scopes:
                        scope.preserve_commit_uncertainty(outcome)
                    if execution_commit_outcome(terminal) is None:
                        raise execution_error_with_cause(ExecutionCommitOutcomeError(outcome), terminal)
                raise terminal
            terminal = execution_terminal_error(body_failure, retired.cancellation,
                retain_originals=True, isolate_control_originals=True)
            if terminal is not None:
                terminal = state.retain_acknowledgments(terminal)
                if (phase is not None and retired.cancellation is not None
                    and state.acknowledgments):
                    phase.record_acknowledged_withdrawal(retired.cancellation,
                                                        state.acknowledgments[-1])
                if admit_work and state.acknowledgments:
                    terminal = self._retain_after_commit_denial(terminal, state.scopes, state.acknowledgments)
                raise terminal
            # Release/reset is an awaited part of the operation, not an
            # authority-free epilogue. Never publish a stale read/RETURNING value
            # or imply rollback when revocation raced a successful COMMIT/reset.
            if not admit_work:
                return
            if state.outcome is not None:
                self._require_after_authority_commit(state.scopes, state.acknowledgments)
            else:
                require_execution_backend("postgres", state.scopes)

    @asynccontextmanager
    async def _authority_transaction(
        self, connection: Any, scopes: tuple[ExecutionCustody, ...],
        commit_state: _ExecutionCommitState | None = None,
    ) -> AsyncIterator[None]:
        _require_native_observation_for_connection(connection)
        state = commit_state if commit_state is not None else _ExecutionCommitState()
        if scopes and not state.native_owned:
            raise ExecutionAuthorityError("opaque pool checkout cannot admit execution custody; use PostgresPoolAdapter")
        try:
            async with self._tracked_native_transaction(connection, scopes, state,
                admit_work=True,
                requires_commit_proof=lambda: bool(scopes) or _native_observation_requested() or (
                    self._current_txn_conn() is not None and self._current_after_commit().pending)):
                await lock_execution_authority(connection, "postgres", scopes)
                yield
                require_execution_backend("postgres", scopes)
                self._require_transaction_custody()
        except BaseException as error:
            if state.acknowledgments and execution_commit_outcome(error) == "committed":
                raise self._retain_after_commit_denial(error, scopes, state.acknowledgments)
            raise
        self._require_after_authority_commit(scopes, state.acknowledgments)

    def _retain_after_commit_denial(
        self, original: BaseException, scopes: tuple[ExecutionCustody, ...],
        acknowledgments: Sequence[AcknowledgedExecutionCommit],
    ) -> BaseException:
        """Cancellation cannot conceal authority loss racing a successful ACK."""
        try:
            self._require_after_authority_commit(scopes, acknowledgments)
        except BaseException as denial:
            if not is_execution_control_error(denial):
                # A validator that cannot complete is not a scoped successful
                # refusal. Preserve its exact cause and fail all original
                # admissions closed, rather than localizing an ordinary error.
                for scope in scopes:
                    scope.preserve_commit_uncertainty("committed")
                failure = ExecutionCommitOutcomeError("committed")
                failure = execution_error_with_cause(failure, denial)
                denial = failure
            return execution_terminal_error(original, denial, retain_originals=True,
                isolate_control_originals=True)
        return original

    @asynccontextmanager
    async def _tracked_native_transaction(
        self, connection: Any, scopes: tuple[ExecutionCustody, ...],
        commit_state: _ExecutionCommitState | None = None,
        *, requires_commit_proof: bool | Callable[[], bool] = True,
        admit_work: bool = False,
    ) -> AsyncIterator[None]:
        """Classify native commit acknowledgement without granting work.

        Ordinary work adds its lock/admission in _authority_transaction; fixed
        cleanup uses only this private physical-outcome boundary.
        """
        # A failure in the body rolls back. Once the body/precommit checks
        # finish, a failed commit acknowledgement must instead be reconciled.
        if commit_state is not None:
            commit_state.remember(scopes)
        transaction = connection.transaction()
        if type(transaction) is not asyncpg.transaction.Transaction:
            raise ExecutionAuthorityError("native transaction exit is opaque; native custody requires asyncpg Transaction")
        top_level = not connection.is_in_transaction()
        needs_proof = requires_commit_proof() if callable(requires_commit_proof) else requires_commit_proof
        if needs_proof and not top_level:
            raise ExecutionAuthorityError("native known-commit custody requires its own top-level transaction")
        if admit_work:
            # Operational mutexes and checkout can suspend after the caller's
            # initial admission. This is the LAST boundary before native BEGIN,
            # not the later in-transaction lock/validator. Fixed cleanup exits
            # deliberately do not acquire a new work admission here.
            _require_native_observation_for_connection(connection)
            require_execution_backend("postgres", scopes)
        await transaction.__aenter__()
        native_connection = connection._con if isinstance(connection, asyncpg.pool.PoolConnectionProxy) else connection
        body_failure: BaseException | None = None
        proven = False
        try:
            yield
            # PostgreSQL accepts COMMIT of an aborted transaction as ROLLBACK,
            # with no asyncpg exception. A successful exit task alone therefore
            # cannot witness a committed receipt. Prove this SAME transaction
            # committable before handing off its native exit, even when there
            # are no after-commit callbacks. This remains cancellable body work.
            if requires_commit_proof() if callable(requires_commit_proof) else requires_commit_proof:
                if not top_level:
                    raise ExecutionAuthorityError("native known-commit custody requires its own top-level transaction")
                await _prove_committable(connection)
                proven = True
            # The proof above is cancellable body work, not COMMIT acceptance.
            # Retirement may happen while it awaits; validate ordinary work
            # again without another await before creating the native exit.
            # Fixed native cleanup deliberately keeps its separate contract.
            if admit_work:
                require_execution_backend("postgres", scopes)
                self._require_transaction_custody()
        except BaseException as error:
            body_failure = error
        committing = body_failure is None
        # Protect the accepted native epilogue, NOT the body above. In
        # particular, external resolvers/gates retain normal cancellation.
        observation = commit_state.native_observation if commit_state is not None else None
        with observation.phase(transaction) if observation is not None else nullcontext() as phase:
            admission = OwnedTaskAdmission()
            try:
                completion = create_owned_task(transaction.__aexit__(
                    type(body_failure) if body_failure is not None else None,
                    body_failure,
                    body_failure.__traceback__ if body_failure is not None else None,
                ), name=f"{asyncio.current_task().get_name()}:postgres-original-transaction-exit",
                    retain_originals=True, admission=admission)
            except BaseException as refusal:
                accepted = admission.task
                if phase is not None:
                    phase.owner = accepted
                if accepted is not None:
                    joined = await await_owned_task(accepted)
                    refusal = execution_terminal_error(refusal, joined.error, joined.cancellation,
                        retain_originals=True, isolate_control_originals=True)
                terminal = self._refused_native_owner(connection, transaction, refusal, scopes, body_failure, native_connection, owner=accepted)
                if phase is not None:
                    phase.record_native_failure(terminal)
                raise terminal
            if phase is not None:
                phase.owner = completion
            if committing and commit_state is not None:
                commit_state.outcome = "unknown"
            completed = await await_owned_task(completion)
            if completed.error is not None:
                if not committing and native_connection.is_closed():
                    # No COMMIT owner was accepted. The original native exit
                    # is joined, and the same session is physically closed;
                    # asyncpg may already have detached its pool proxy. There
                    # is no live rollback owner to invent. Retain every cause
                    # and observer withdrawal; the enclosing checkout must
                    # still finish its own original reset before publication.
                    failure = PostgresSessionLostError(
                        "Original PostgreSQL session closed before COMMIT"
                    )
                    failure.__cause__ = execution_terminal_error(
                        body_failure, completed.error, completed.cancellation,
                        retain_originals=True, isolate_control_originals=True,
                    )
                    raise failure
                if phase is not None:
                    phase.record_native_failure(completed.error, commit_outcome="unknown" if committing else None, cancellation=completed.cancellation)
                if committing:
                    physical = completed.error
                    outcome = execution_commit_outcome(physical) or "unknown"
                    if execution_commit_outcome(physical) is None:
                        classified = ExecutionCommitOutcomeError(outcome)
                        classified = execution_error_with_cause(classified, physical)
                        physical = classified
                    terminal = execution_terminal_error(
                        physical, completed.cancellation, retain_originals=True,
                        isolate_control_originals=True,
                    )
                    for scope in scopes:
                        scope.preserve_commit_uncertainty(outcome)
                    raise terminal
                debt = RuntimeError("original native rollback acknowledgement failed")
                debt.__cause__ = completed.error
                debt._kestrel_native_retirement_debts = (
                    (_NativePostgresRetirement(transaction, connection, native_connection), completion),
                )
                terminal = execution_terminal_error(
                    body_failure, debt, completed.cancellation, retain_originals=True,
                    isolate_control_originals=True,
                )
                for scope in scopes:
                    scope.revoke("original native rollback acknowledgement failed")
                raise terminal
            receipt = (AcknowledgedExecutionCommit(completion, retirement=commit_state.retirement)
                       if committing and proven and commit_state is not None
                       else AcknowledgedExecutionCommit(completion) if committing and proven else None)
            if committing and commit_state is not None:
                commit_state.outcome = "committed" if proven else "unknown"
                # The checkout retains one latest native fact, not every finished
                # task from a long operational session. Earlier escaped receipts
                # still share this SAME original reset record and settle with it.
                commit_state.acknowledgments[:] = [receipt] if receipt is not None else []
                if observation is not None:
                    observation.acknowledgments = tuple(commit_state.acknowledgments)
            terminal = execution_terminal_error(
                body_failure, completed.cancellation, retain_originals=True,
                isolate_control_originals=True,
            )
            if receipt is not None and completed.cancellation is not None and phase is not None:
                phase.record_acknowledged_withdrawal(completed.cancellation, receipt)
            if terminal is not None:
                if receipt is not None:
                    retain_acknowledged_execution_commit(terminal, receipt)
                elif committing:
                    # This is actual observer withdrawal after an unproven
                    # successful exit, NOT a physical source cancellation.
                    raise execution_terminal_error(
                        ExecutionCommitOutcomeError("unknown"), terminal,
                        retain_originals=True, isolate_control_originals=True,
                    )
                raise terminal

    @staticmethod
    def _require_after_authority_commit(
        scopes: tuple[ExecutionCustody, ...],
        acknowledgments: Sequence[AcknowledgedExecutionCommit] = (),
    ) -> None:
        # Commit has completed. If loss raced its acknowledgement, denial here
        # does not mean rollback; callers must reconcile their idempotent write.
        try:
            require_execution_backend("postgres", scopes)
        except ExecutionAuthorityError as exc:
            if acknowledgments:
                denied = execution_denial_scopes(exc)
                if denied is not None:
                    for scope in denied:
                        for receipt in acknowledgments:
                            scope.retain_acknowledged_commit(receipt)
                for receipt in acknowledgments:
                    retain_acknowledged_execution_commit(exc, receipt)
                raise
            for scope in scopes:
                scope.preserve_commit_uncertainty("committed")
            raise execution_error_with_cause(ExecutionCommitOutcomeError("committed"), exc)

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
        await _settle_native_observer_enrollment()
        record_write_query(query)
        pool = self._ensure_connected()
        pg_query = self._convert_query(query)
        params = self._strip_tz(params)

        attempt = 0
        while True:
            try:
                txn = self._current_txn_conn()
                if txn is not None:
                    async with self._execution_query_executor(txn) as checked:
                        result = await checked.execute(pg_query, *params)
                else:
                    async with self._operational_query_connection() as conn:
                        async with self._execution_query_executor(conn or pool) as checked:
                            result = await checked.execute(pg_query, *params)

                # Parse affected rows from result (e.g., "INSERT 0 1" or "UPDATE 5")
                if result:
                    parts = result.split()
                    if len(parts) >= 2 and parts[-1].isdigit():
                        return int(parts[-1])
                return 0

            except Exception as e:
                # Retry the transient concurrent-upsert race on the autocommit
                # path only (a poisoned txn can't be salvaged statement-wise).
                retry_delay = concurrent_write_retry_delay(e, attempt)
                if txn is None and retry_delay is not None:
                    attempt += 1
                    logger.warning(
                        "Concurrent-update race on write (attempt %d/%d), retrying: %s",
                        attempt, _CONCURRENT_WRITE_RETRIES, e,
                    )
                    await asyncio.sleep(retry_delay)
                    continue
                raise execution_error_with_cause(QueryError(f"Query failed: {_failure_detail(e)}\nQuery: {pg_query}"), e)
    
    async def execute_many(self, query: str, params_list: List[Params]) -> int:
        """Execute query with multiple parameter sets."""
        await _settle_native_observer_enrollment()
        if not params_list:
            return 0
        record_write_query(query)
        pool = self._ensure_connected()
        pg_query = self._convert_query(query)
        params_list = [self._strip_tz(p) for p in params_list]

        try:
            txn = self._current_txn_conn()
            if txn is not None:
                async with self._execution_query_executor(txn) as checked:
                    await checked.executemany(pg_query, params_list)
            else:
                async with self._operational_query_connection() as conn:
                    if conn is not None:
                        async with self._execution_query_executor(conn) as checked:
                            await checked.executemany(pg_query, params_list)
                    else:
                        async with self._authority_pool_checkout(pool) as (acquired, state):
                            async with self._execution_query_executor(acquired, state) as checked:
                                await checked.executemany(pg_query, params_list)
            return len(params_list)  # asyncpg doesn't return affected count
            
        except Exception as e:
            raise execution_error_with_cause(QueryError(f"Query failed: {_failure_detail(e)}\nQuery: {pg_query}"), e)
    
    async def fetch_one(self, query: str, params: Params = ()) -> Optional[Row]:
        """Fetch a single row."""
        await _settle_native_observer_enrollment()
        record_write_query(query)
        pool = self._ensure_connected()
        pg_query = self._convert_query(query)
        params = self._strip_tz(params)

        try:
            txn = self._current_txn_conn()
            if txn is not None:
                async with self._execution_query_executor(txn) as checked:
                    row = await checked.fetchrow(pg_query, *params)
            else:
                async with self._operational_query_connection() as conn:
                    async with self._execution_query_executor(conn or pool) as checked:
                        row = await checked.fetchrow(pg_query, *params)
            
            if row is None:
                return None
            return tuple(row.values())
            
        except Exception as e:
            raise execution_error_with_cause(QueryError(f"Query failed: {_failure_detail(e)}\nQuery: {pg_query}"), e)
    
    async def fetch_all(self, query: str, params: Params = ()) -> List[Row]:
        """Fetch all rows."""
        await _settle_native_observer_enrollment()
        record_write_query(query)
        pool = self._ensure_connected()
        pg_query = self._convert_query(query)
        params = self._strip_tz(params)

        try:
            txn = self._current_txn_conn()
            if txn is not None:
                async with self._execution_query_executor(txn) as checked:
                    rows = await checked.fetch(pg_query, *params)
            else:
                async with self._operational_query_connection() as conn:
                    async with self._execution_query_executor(conn or pool) as checked:
                        rows = await checked.fetch(pg_query, *params)
            
            return [tuple(row.values()) for row in rows]
            
        except Exception as e:
            raise execution_error_with_cause(QueryError(f"Query failed: {_failure_detail(e)}\nQuery: {pg_query}"), e)
    
    async def fetch_val(self, query: str, params: Params = ()) -> Optional[Any]:
        """Fetch a single value."""
        await _settle_native_observer_enrollment()
        record_write_query(query)
        pool = self._ensure_connected()
        pg_query = self._convert_query(query)
        params = self._strip_tz(params)

        try:
            txn = self._current_txn_conn()
            if txn is not None:
                async with self._execution_query_executor(txn) as checked:
                    return await checked.fetchval(pg_query, *params)
            async with self._operational_query_connection() as conn:
                async with self._execution_query_executor(conn or pool) as checked:
                    return await checked.fetchval(pg_query, *params)
            
        except Exception as e:
            raise execution_error_with_cause(QueryError(f"Query failed: {_failure_detail(e)}\nQuery: {pg_query}"), e)
    
    async def execute_script(self, script: str) -> None:
        """Execute a multi-statement SQL script."""
        await _settle_native_observer_enrollment()
        if self._current_txn_conn() is not None or current_execution_custody(self) or _native_observation_requested():
            reject_transaction_control(script, dialect="postgres")
        record_write_script(script)
        pool = self._ensure_connected()
        
        try:
            txn = self._current_txn_conn()
            if txn is not None:
                async with self._execution_query_executor(txn) as checked:
                    await checked.execute(script)
            else:
                async with self._operational_query_connection() as conn:
                    if conn is not None:
                        async with self._execution_query_executor(conn) as checked:
                            await checked.execute(script)
                    else:
                        async with self._authority_pool_checkout(pool) as (acquired, state):
                            async with self._execution_query_executor(acquired, state) as checked:
                                await checked.execute(script)
                    
        except Exception as e:
            raise execution_error_with_cause(QueryError(f"Script execution failed: {_failure_detail(e)}"), e)

    @staticmethod
    async def _verify_operational_cluster_identity(
        conn: Any,
        expected_cluster_identity: str | None,
    ) -> None:
        """Bind a checked cluster identity to the session used for later I/O."""

        if expected_cluster_identity is None:
            return
        try:
            actual_cluster_identity = await conn.fetchval(
                "SELECT system_identifier::text "
                "FROM pg_catalog.pg_control_system()"
            )
        except Exception as exc:
            raise execution_error_with_cause(QueryError(
                "PostgreSQL operational session cluster identity could not be "
                "verified"
            ), exc)
        if (
            not isinstance(actual_cluster_identity, str)
            or not actual_cluster_identity.strip()
            or actual_cluster_identity != expected_cluster_identity
        ):
            raise QueryError(
                "PostgreSQL operational session is not on the validated "
                "PostgreSQL cluster"
            )

    @asynccontextmanager
    async def operational_session(
        self,
        *,
        expected_cluster_identity: str | None = None,
    ) -> AsyncIterator[None]:
        """Pin autocommit operations to one checked physical connection.

        This is deliberately not a transaction: Hold's external evidence
        protocol must publish commits incrementally, but a load-balanced DSN
        must not let its cluster-identity probe and custody reads land on
        different servers. Ordinary backend methods route through the pinned
        connection for the owning task for the lifetime of this context.
        """

        await _settle_native_observer_enrollment()
        existing = self._current_txn_conn()
        if existing is not None:
            _require_native_observation_for_connection(existing)
            await self._verify_operational_cluster_identity(
                existing,
                expected_cluster_identity,
            )
            yield
            return

        existing_lease = self._current_operational_lease()
        if existing_lease is not None:
            async with existing_lease.lock:
                if not existing_lease.active:
                    raise QueryError(
                        "PostgreSQL operational session is no longer active"
                    )
                _require_native_observation_for_connection(existing_lease.connection)
                await self._verify_operational_cluster_identity(
                    existing_lease.connection,
                    expected_cluster_identity,
                )
            yield
            return

        pool = self._ensure_connected()
        async with self._authority_pool_checkout(pool) as (conn, commit_state):
            await self._verify_operational_cluster_identity(
                conn,
                expected_cluster_identity,
            )
            lease = _OperationalSessionLease(conn, asyncio.Lock(), commit_state=commit_state)
            token = self._operational_conn_var.set(lease)
            try:
                yield
            finally:
                async with lease.lock:
                    lease.active = False
                self._operational_conn_var.reset(token)

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        """Transaction context manager.

        Nesting is detected PER TASK via the ContextVar (#1726): a nested
        ``transaction()`` within the SAME task reuses that task's connection as a
        savepoint, while a ``transaction()`` in a DIFFERENT concurrent task sees
        no open connection (its ContextVar is the default) and acquires its own —
        so concurrent transactions no longer collide on a shared attribute.
        """
        await _settle_native_observer_enrollment()
        pool = self._ensure_connected()

        existing = self._current_txn_conn()
        if existing is not None:
            # Nested transaction (same task) - use savepoint on this task's conn.
            _require_native_observation_for_connection(existing)
            require_execution_backend("postgres", self._remember_transaction_custody())
            after_commit = self._current_after_commit()
            after_commit.open_savepoint()
            body_failure: BaseException | None = None
            try:
                async with existing.transaction():
                    try:
                        await self._lock_transaction_custody(existing)
                        yield
                        self._require_transaction_custody()
                    except BaseException as error:
                        body_failure = error
                        raise
            except BaseException as cleanup:
                after_commit.roll_back_savepoint()
                raise execution_terminal_error(body_failure, cleanup,
                    retain_originals=True, isolate_control_originals=True)
            after_commit.release_savepoint()
            return

        # Run once the transaction has committed and its connection is back.
        committed: Optional[AfterCommitCallbacks] = None
        operational = self._current_operational_lease()
        if operational is not None:
            async with operational.lock:
                if operational.active:
                    after_commit = AfterCommitCallbacks()
                    token = self._txn_conn_var.set(
                        (asyncio.current_task(), operational.connection)
                    )
                    scopes = list(current_execution_custody(self))
                    custody_var = self._transaction_custody_context()
                    custody_token = custody_var.set((asyncio.current_task(), scopes))
                    after_commit_token = self._after_commit_context().set(
                        (asyncio.current_task(), after_commit)
                    )
                    try:
                        async with self._authority_transaction(operational.connection, tuple(scopes), operational.commit_state):
                            yield
                    except ExecutionCommitOutcomeError:
                        raise
                    except Exception as e:
                        raise execution_error_with_cause(TransactionError(f"Transaction failed: {_failure_detail(e)}"), e)
                    finally:
                        custody_var.reset(custody_token)
                        self._after_commit_context().reset(after_commit_token)
                        self._txn_conn_var.reset(token)
                    committed = after_commit
            if committed is not None:
                committed.run()
                return

        after_commit = AfterCommitCallbacks()
        async with self._authority_pool_checkout(pool) as (conn, commit_state):
            token = self._txn_conn_var.set((asyncio.current_task(), conn))
            scopes = list(current_execution_custody(self))
            custody_var = self._transaction_custody_context()
            custody_token = custody_var.set((asyncio.current_task(), scopes))
            after_commit_token = self._after_commit_context().set(
                (asyncio.current_task(), after_commit)
            )
            try:
                async with self._authority_transaction(conn, tuple(scopes), commit_state):
                    yield
            except ExecutionCommitOutcomeError:
                raise
            except Exception as e:
                raise execution_error_with_cause(TransactionError(f"Transaction failed: {_failure_detail(e)}"), e)
            finally:
                custody_var.reset(custody_token)
                self._after_commit_context().reset(after_commit_token)
                self._txn_conn_var.reset(token)
        after_commit.run()

    async def _close_failed_advisory_session(
        self, connection: Any, original: BaseException,
    ) -> BaseException:
        """Join the original server session before reporting failed release.

        Local ``terminate()`` does not wait for PostgreSQL to process the
        disconnect and release its locks. Native graceful close cancels and
        joins an active query, sends Terminate, and waits for the server EOF.
        A refused or failed closer is explicit retirement debt, never proof
        that the original server-side exclusion has disappeared.
        """
        native = (connection._con if isinstance(connection, asyncpg.pool.PoolConnectionProxy)
                  else connection)
        if native is None:
            # A detached proxy cannot close a successor checkout.
            return original
        scopes = current_execution_custody(self)
        resource = _NativePostgresRetirement(connection, connection, native)
        admission = OwnedTaskAdmission()
        try:
            closer = create_owned_task(
                connection.close(), name="postgres-original-advisory-retirement",
                retirement_resource=resource, retain_originals=True, admission=admission,
            )
        except BaseException as refusal:
            accepted = admission.task
            if accepted is not None:
                joined = await await_owned_task(accepted)
                refusal = execution_terminal_error(
                    refusal, joined.error, joined.cancellation,
                    retain_originals=True, isolate_control_originals=True,
                )
            return self._refused_native_owner(
                connection, connection, refusal, scopes, original, native, owner=accepted,
            )
        outcome = await await_owned_task(closer)
        debt = None
        if outcome.error is not None:
            debt = RuntimeError("original advisory session retirement failed")
            debt.__cause__ = outcome.error
            debt._kestrel_native_retirement_debts = ((resource, closer),)
        terminal = execution_terminal_error(
            original, debt, outcome.error, outcome.cancellation, retain_originals=True,
            isolate_control_originals=True,
        )
        if outcome.error is not None:
            for scope in scopes:
                scope.revoke("original advisory session retirement failed")
        return terminal

    @asynccontextmanager
    async def advisory_locks(
        self,
        keys: Sequence[Tuple[int, int]] | TerminalAdvisoryBarrier,
        *,
        shared: bool = False,
        expected_cluster_identity: Optional[str] = None,
        on_loss: Callable[[AdvisoryLease], None] | None = None,
    ) -> AsyncIterator[AdvisoryLease | TerminalAdvisoryLease | None]:
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
        PostgreSQL's signed two-int advisory-lock form. When
        ``expected_cluster_identity`` is supplied, the exact session that will
        own the locks must prove that immutable ``initdb`` identity first.
        This matters for load-balanced DSNs: the bounded advisory pool is
        intentionally separate from the operational pool, so merely probing
        the latter does not prove which cluster serialized the operation.
        """
        terminal = type(keys) is TerminalAdvisoryBarrier
        if isinstance(keys, TerminalAdvisoryBarrier) and not terminal:
            raise TypeError("terminal advisory barrier must use the exact canonical request type")
        if terminal and shared is not False:
            raise ValueError("terminal advisory barriers must be exclusive")
        request = keys if terminal else None
        normalized_keys = request.keys if terminal else tuple(keys)
        for namespace, key in normalized_keys:
            if not isinstance(namespace, int) or not isinstance(key, int):
                raise TypeError("PostgreSQL advisory lock keys must be integers")
        if not normalized_keys:
            # A dynamic hosted scheduler can legitimately have no tenants
            # during bootstrap. Do not pin a spare pool connection just to
            # represent that empty exclusion set.
            if on_loss is not None:
                raise ValueError("empty advisory keys cannot grant authority custody")
            yield None
            return
        failure = None
        admission = ExecutionCustody(request) if terminal else None
        try:
            # Denial begins BEFORE allocation, listener registration or init
            # callbacks, and remains through original native pool retirement.
            # This request is an always-denying fence, not an acquiring lease.
            with bind_execution_cleanup(self, (admission,)) if terminal else nullcontext():
                async with self._advisory_lock_session(
                    normalized_keys, shared=shared, expected_cluster_identity=expected_cluster_identity,
                    on_loss=on_loss, terminal_lifetime=_TerminalAdvisoryLifetime.capture(self) if terminal else None,
                ) as capability:
                    yield capability
        except BaseException as error:
            failure = error
        finally:
            if admission is not None:
                admission.revoke("terminal advisory barrier retired")
        if terminal:
            # The fixed operation is physically finished, not a replacement
            # admission. Sample ALL original scopes after exit/reset has joined,
            # including when acquisition or the terminal body failed.
            try:
                require_execution_work(self)
            except BaseException as denial:
                failure = execution_terminal_error(
                    failure, denial, retain_originals=True,
                    isolate_control_originals=True,
                )
        if failure is not None:
            raise failure

    @asynccontextmanager
    async def _advisory_lock_session(
        self, normalized_keys: tuple[tuple[int, int], ...], *, shared: bool,
        expected_cluster_identity: str | None,
        on_loss: Callable[[AdvisoryLease], None] | None,
        terminal_lifetime: _TerminalAdvisoryLifetime | None,
    ) -> AsyncIterator[AdvisoryLease]:
        """One native allocator/retirement algorithm for both capability types."""
        lock_function = "pg_advisory_lock_shared" if shared else "pg_advisory_lock"
        unlock_function = (
            "pg_advisory_unlock_shared" if shared else "pg_advisory_unlock"
        )
        advisory_pool = await self._ensure_advisory_pool(terminal_lifetime=terminal_lifetime)
        # Pool acquisition queues outside the operational query pool.  A
        # cancellation while waiting therefore cannot strand an operational
        # connection that an admitted effect still needs.
        async with self._tracked_pool_checkout(advisory_pool, admit_work=False) as (conn, _):
            acquired: list[Tuple[int, int]] = []
            capability = (TerminalAdvisoryLease(conn, on_loss, lifetime=terminal_lifetime)
                          if terminal_lifetime is not None else AdvisoryLease(conn, on_loss))
            conn.add_termination_listener(capability._connection_terminated)
            listener_attached = True
            try:
                capability.require_physical()
                if expected_cluster_identity is not None:
                    actual_cluster_identity = await conn.fetchval(
                        "SELECT system_identifier::text "
                        "FROM pg_catalog.pg_control_system()"
                    )
                    if (
                        not isinstance(actual_cluster_identity, str)
                        or not actual_cluster_identity.strip()
                        or actual_cluster_identity != expected_cluster_identity
                    ):
                        raise QueryError(
                            "PostgreSQL advisory lock session is not on the "
                            "validated PostgreSQL cluster"
                        )
                for namespace, key in normalized_keys:
                    await conn.execute(
                        f"SELECT {lock_function}($1, $2)", namespace, key
                    )
                    acquired.append((namespace, key))
                    capability.require_physical()
                capability._admit()
                yield capability
                capability.require_physical()
            except BaseException as original:
                # A cancellation can race PostgreSQL granting a blocking lock
                # after asyncpg has sent the cancellation request. Terminating
                # this dedicated connection is the only unconditional release
                # in that race; the bounded pool discards it and never returns
                # a possibly locked session to another scheduler operation.
                capability._retire()
                session_attached = capability._remove_listener()
                listener_attached = False
                if session_attached:
                    raise await self._close_failed_advisory_session(conn, original)
                raise
            else:
                capability._retire()
                capability._remove_listener()
                listener_attached = False
                try:
                    # ``pg_advisory_unlock`` must run on the same session that
                    # acquired the lock. The normal path returns the clean,
                    # unlocked connection to the bounded advisory pool.
                    for namespace, key in reversed(acquired):
                        await conn.execute(
                            f"SELECT {unlock_function}($1, $2)", namespace, key
                        )
                except BaseException as original:
                    # asyncpg may already have detached the pooled proxy
                    # after loss. Never replace the original loss/cancellation
                    # with that secondary cleanup error.
                    raise await self._close_failed_advisory_session(conn, original)
            finally:
                capability._retire()
                if listener_attached:
                    capability._remove_listener()

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
            except BaseException as original:
                raise await self._close_failed_advisory_session(conn, original)
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
                except BaseException as original:
                    raise await self._close_failed_advisory_session(conn, original)
    
    async def table_exists(self, table_name: str) -> bool:
        """Check a literal unqualified relation name in the active search path.

        Read the MVCC catalog directly so a fresh statement sees DDL another
        initializer committed while this transaction waited for its lease.
        A previously negative ``to_regclass`` name lookup can remain stale.
        The name is a bound value, as on SQLite; it is not SQL identifier text.
        """
        return bool(await self.fetch_val(
            "SELECT EXISTS ("
            "SELECT 1 FROM pg_catalog.pg_class relation "
            "JOIN pg_catalog.pg_namespace namespace "
            "ON namespace.oid = relation.relnamespace "
            "WHERE relation.relname = ? "
            "AND namespace.nspname = ANY(current_schemas(true)))",
            (table_name,),
        ))

    async def settle_stop_invocation(self, generation_id: str, owner_id: str, disposition: Any) -> None:
        """Only an existing exact Stop generation/owner may terminalize.

        This fixed native operation is available after work-custody denial.
        It neither clears the caller's denial nor exposes a connection or
        generic SQL executor. Ordinary storage/tools remain denied throughout.
        """
        from kestrel_sovereign.stop.invocation import _settle_stop_on_connection

        pool = self._pool
        if pool is None:
            raise ConnectionError("Stop terminal custody requires the original connected backend")
        # A terminalizer must not wedge shutdown waiting for a saturated pool.
        async with asyncio.timeout(5):
            async with self._tracked_pool_checkout(pool, admit_work=False) as (connection, state):
                async with self._tracked_native_transaction(connection, state.scopes, state):
                    await _settle_stop_on_connection(
                        connection, generation_id, owner_id, disposition,
                    )

    async def release_initial_reservations(
        self, *, agent_id: str, owner_id: str, now: datetime,
        mark_owner_stopped: bool,
    ) -> int:
        """Retire one existing runtime owner without reopening ordinary work.

        Fixed cleanup SQL on the original connected backend: no new owner,
        lease, claim, provider cursor or arbitrary executor is exposed. The
        same tenant-wide lock serializes this with canonical recovery.
        """
        from kestrel_sovereign.signals.durable import DurableSignalStore

        if any(type(value) is not str or not value.strip() for value in (agent_id, owner_id)):
            raise ValueError("runtime cleanup requires exact owner identities")
        if type(mark_owner_stopped) is not bool or not isinstance(now, datetime) or now.tzinfo is None:
            raise ValueError("runtime cleanup requires a typed disposition and UTC timestamp")
        pool = self._pool
        if pool is None:
            raise ConnectionError("runtime cleanup requires the original backend")
        async with asyncio.timeout(5):
            async with self._tracked_pool_checkout(pool, admit_work=False) as (connection, state):
                async with self._tracked_native_transaction(connection, state.scopes, state):
                    await connection.fetchval(
                        "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
                        DurableSignalStore.runtime_owner_lock_key(agent_id),
                    )
                    existing = await connection.fetchval(
                        f"SELECT 1 FROM {DurableSignalStore.RUNTIME_OWNERS} "
                        "WHERE agent_id=$1 AND owner_id=$2 FOR UPDATE",
                        agent_id, owner_id,
                    )
                    if existing is None:
                        return 0
                    released = await connection.execute(
                        f"UPDATE {DurableSignalStore.DELIVERIES} "
                        "SET status='retry', lease_owner=NULL, lease_token=NULL, "
                        "lease_expires_at=NULL, next_attempt_at=$3, "
                        "last_error='initial reservation owner stopped before activation', "
                        "updated_at=$3 WHERE agent_id=$1 AND lease_owner=$2 "
                        "AND status='initial_reserved'",
                        agent_id, owner_id, now,
                    )
                    if mark_owner_stopped:
                        await connection.execute(
                            f"UPDATE {DurableSignalStore.RUNTIME_OWNERS} "
                            "SET heartbeat_at=GREATEST(heartbeat_at,$3), "
                            "stopped_at=COALESCE(stopped_at,$3), updated_at=$3 "
                            "WHERE agent_id=$1 AND owner_id=$2",
                            agent_id, owner_id, now,
                        )
                    return int(released.rsplit(" ", 1)[-1])

    async def abandon_initial_reservation(
        self, *, agent_id: str, consumer_id: str, delivery_id: str,
        owner_id: str, reservation_token: str, now: datetime, reason: str,
    ) -> bool:
        """Repair only an existing original raw-handoff owner/token CAS."""
        from kestrel_sovereign.signals.durable import DurableSignalStore

        identities = (agent_id, consumer_id, delivery_id, owner_id, reservation_token)
        if any(type(value) is not str or not value.strip() for value in identities):
            raise ValueError("initial reservation cleanup requires exact identities")
        if not isinstance(reason, str):
            raise ValueError("initial reservation cleanup requires a string reason")
        if not isinstance(now, datetime) or now.tzinfo is None:
            raise ValueError("initial reservation cleanup requires a UTC timestamp")
        pool = self._pool
        if pool is None:
            raise ConnectionError("initial reservation cleanup requires the original backend")
        async with asyncio.timeout(5):
            async with self._tracked_pool_checkout(pool, admit_work=False) as (connection, state):
                async with self._tracked_native_transaction(connection, state.scopes, state):
                    result = await connection.execute(
                        f"UPDATE {DurableSignalStore.DELIVERIES} "
                        "SET status='retry', lease_owner=NULL, lease_token=NULL, "
                        "lease_expires_at=NULL, next_attempt_at=$6, last_error=$7, "
                        "updated_at=$6 WHERE agent_id=$1 AND consumer_id=$2 "
                        "AND delivery_id=$3 AND lease_owner=$4 AND lease_token=$5 "
                        "AND status IN ('initial_reserved','leased')",
                        agent_id, consumer_id, delivery_id, owner_id, reservation_token, now, reason,
                    )
                    return result == "UPDATE 1"

    async def purge_retired_ephemeral_session(self, wrapper, *, reason: str):
        """Run only the original standalone runtime's mandatory privacy purge.

        The fixed operation cannot supply SQL, replace authority, initialize
        storage, return an executor, or clean a borrowed tenant generation.
        Reuse one checkout from the original pool and the existing scoped
        purge algorithms, including lexical/graph/projection safety.
        """
        from kestrel_sovereign.execution_custody import ProcessRuntimeExecutionFence
        from kestrel_sovereign.storage.async_storage import AsyncStorage
        from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage

        from .ephemeral_cleanup import _purge_original_ephemeral_session

        scope = getattr(self, "_execution_custody", None)
        if not isinstance(scope, ExecutionCustody) or not isinstance(scope.fence, ProcessRuntimeExecutionFence) or not scope.fence._retired:
            raise ExecutionAuthorityError("privacy cleanup requires the original retired process runtime")
        if type(wrapper) is not PrivacyEnforcingStorage or type(wrapper._storage) is not AsyncStorage:
            raise ValueError("privacy cleanup requires canonical initialized storage")
        storage = wrapper._storage
        agent_id, since = wrapper.agent_id, wrapper._entered_ephemeral_at
        if storage._backend is not self or not storage._initialized or agent_id != scope.fence.agent_id:
            raise ValueError("privacy cleanup does not match original backend and agent")
        if type(reason) is not str or not reason.strip():
            raise ValueError("privacy cleanup requires its audit reason")
        pool = self._pool
        if pool is None:
            raise ConnectionError("privacy cleanup requires the original connected backend")
        async with asyncio.timeout(5):
            async with self._tracked_pool_checkout(pool, admit_work=False) as (connection, state):
                async with self._tracked_native_transaction(connection, state.scopes, state):
                    report, watermark = await _purge_original_ephemeral_session(
                        self, connection, wrapper, reason=reason, agent_id=agent_id, since=since,
                    )
        # Publish only after native COMMIT and pool release/reset acknowledge.
        if not report.required_sweep_failed:
            wrapper._entered_ephemeral_at = watermark
            wrapper._session_conversations = []
            wrapper._session_files = {}
            wrapper._session_file_metadata = {}
        return report

    async def retain_cognition_cleanup_owner(self, *, agent_id: str, owner_id: str) -> bool:
        """Retain an existing live owner with leased or receipt-pending cognition.

        This cleanup-only metadata operation cannot insert/revive an owner,
        grant a lease, acknowledge ingress, or reopen ordinary native work.
        It serializes with canonical owner recovery using its exact key.
        """
        from kestrel_sovereign.signals.durable import DurableSignalStore
        from kestrel_sovereign.signals.sources.channels import (
            DURABLE_COGNITION_CONSUMER_ID,
        )

        if any(type(value) is not str or not value.strip() for value in (agent_id, owner_id)):
            raise ValueError("cognition cleanup requires exact owner identities")
        if not owner_id.startswith("dispatcher:"):
            raise ValueError("cognition cleanup requires its managed owner")
        pool = self._pool
        if pool is None:
            raise ConnectionError("cognition cleanup requires the original backend")
        async with asyncio.timeout(5):
            async with self._tracked_pool_checkout(pool, admit_work=False) as (connection, state):
                async with self._tracked_native_transaction(connection, state.scopes, state):
                    await connection.fetchval(
                        "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
                        DurableSignalStore.runtime_owner_lock_key(agent_id),
                    )
                    result = await connection.execute(
                        f"UPDATE {DurableSignalStore.RUNTIME_OWNERS} owner "
                        "SET heartbeat_at = GREATEST(heartbeat_at, clock_timestamp()), "
                        "updated_at = clock_timestamp() "
                        "WHERE owner.agent_id = $1 AND owner.owner_id = $2 "
                        "AND owner.stopped_at IS NULL AND EXISTS ("
                        f"SELECT 1 FROM {DurableSignalStore.DELIVERIES} delivery "
                        "WHERE delivery.agent_id = $1 AND delivery.lease_owner = $2 "
                        "AND delivery.consumer_id = $3 "
                        "AND delivery.status IN ('leased', 'retry') "
                        "AND delivery.lease_token IS NOT NULL)",
                        agent_id, owner_id, DURABLE_COGNITION_CONSUMER_ID,
                    )
                    return result == "UPDATE 1"

    async def fail_cognition_delivery(
        self, *, agent_id: str, consumer_id: str, delivery_id: str,
        owner_id: str, lease_token: str, error: str,
    ) -> bool:
        """Terminalize only an existing exact cognition lease after control loss.

        Like Stop settlement, this fixed cleanup operation neither admits work
        nor exposes an executor. The original owner/token CAS cannot acquire a
        new lease, acknowledge a provider cursor, or overwrite a successor.
        """
        from kestrel_sovereign.signals.durable import DurableSignalStore
        from kestrel_sovereign.signals.sources.channels import (
            DURABLE_COGNITION_CONSUMER_ID,
        )

        identities = (agent_id, consumer_id, delivery_id, owner_id, lease_token, error)
        if any(type(value) is not str or not value.strip() for value in identities):
            raise ValueError("cognition terminal custody requires exact identities")
        if consumer_id != DURABLE_COGNITION_CONSUMER_ID or not owner_id.startswith("dispatcher:"):
            raise ValueError("cognition terminal custody requires its managed consumer")
        pool = self._pool
        if pool is None:
            raise ConnectionError("cognition terminal custody requires the original backend")
        async with asyncio.timeout(5):
            async with self._tracked_pool_checkout(pool, admit_work=False) as (connection, state):
                async with self._tracked_native_transaction(connection, state.scopes, state):
                    result = await connection.execute(
                        f"UPDATE {DurableSignalStore.DELIVERIES} "
                        "SET status = 'failed', next_attempt_at = NULL, "
                        "lease_owner = NULL, lease_token = NULL, lease_expires_at = NULL, "
                        "last_error = $1, terminal_at = NOW(), updated_at = NOW() "
                        "WHERE agent_id = $2 AND consumer_id = $3 AND delivery_id = $4 "
                        "AND status IN ('leased', 'retry') "
                        "AND lease_owner = $5 AND lease_token = $6",
                        error, agent_id, consumer_id, delivery_id, owner_id, lease_token,
                    )
                    return result == "UPDATE 1"
