"""Construct async SQLAlchemy session factories that connect to the
same database as a given :class:`AsyncDatabase`.

The factories are duck-typed to match what
``kestrel_sovereign.storage.vector`` expects: an object with an
``engine`` attribute (for dialect dispatch) and ``read_session()`` /
``write_session()`` async-context-managers that yield
``AsyncSession`` instances.

Why a separate factory rather than borrowing AsyncDatabase's pool?
``AsyncDatabase``'s PostgresBackend wraps an ``asyncpg.Pool``;
SQLAlchemy's asyncpg dialect also uses ``asyncpg`` underneath, but
threading a borrowed pool through SQLAlchemy's engine plumbing is
involved. For sovereign-core's first SQLA reads (vector search on
``saved_items``, where query volume is low) a small dedicated pool is
operationally simpler. Pool sharing is a follow-up if it ever
matters.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from functools import wraps
from pathlib import Path
from typing import Any

from sqlalchemy import event
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from kestrel_sovereign._async_ownership import (
    await_owned_task,
    create_owned_task,
    owned_task_error,
)
from kestrel_sovereign.execution_custody import (
    current_execution_custody,
    execution_terminal_error,
    is_execution_control_error,
    refuse_unfenced_executor,
    require_execution_work,
)
from kestrel_sovereign.storage.db.interface import ConnectionError
from kestrel_sovereign.storage.db.sqlite import (
    SQLiteBackend,
    _aiosqlite_worker_is_alive,
    _capture_close_diagnostics,
    _close_aiosqlite_connection,
    _minimum_close_timeout_s,
    _prune_retained_aiosqlite_closes,
    _retain_aiosqlite_close,
    _RetainedAiosqliteCloses,
    _SQLiteCloseDiagnostics,
)


def _guard_factory_session_operation(operation):
    """Reject database work that races or follows factory retirement."""

    @wraps(operation)
    async def guarded(self, *args, **kwargs):
        self._assert_factory_available()
        return await operation(self, *args, **kwargs)

    return guarded


class _LifecycleBoundAsyncSession(AsyncSession):
    """An ``AsyncSession`` whose database operations honor factory close.

    SQLAlchemy sessions acquire connections lazily and disposed engines are
    reusable.  A session yielded before factory shutdown could therefore open
    a replacement connection after the engine had been retired.  Bind the
    session to the factory's permanent admission fence so every operation that
    can acquire or use a connection revalidates at call time.
    """

    def __init__(self, *args, factory_available, **kwargs) -> None:
        self._factory_available = factory_available
        super().__init__(*args, **kwargs)

    def _assert_factory_available(self) -> None:
        self._factory_available()

    execute = _guard_factory_session_operation(AsyncSession.execute)
    scalar = _guard_factory_session_operation(AsyncSession.scalar)
    scalars = _guard_factory_session_operation(AsyncSession.scalars)
    stream = _guard_factory_session_operation(AsyncSession.stream)
    stream_scalars = _guard_factory_session_operation(
        AsyncSession.stream_scalars
    )
    connection = _guard_factory_session_operation(AsyncSession.connection)
    commit = _guard_factory_session_operation(AsyncSession.commit)
    flush = _guard_factory_session_operation(AsyncSession.flush)
    get = _guard_factory_session_operation(AsyncSession.get)
    get_one = _guard_factory_session_operation(AsyncSession.get_one)
    refresh = _guard_factory_session_operation(AsyncSession.refresh)
    delete = _guard_factory_session_operation(AsyncSession.delete)
    merge = _guard_factory_session_operation(AsyncSession.merge)
    run_sync = _guard_factory_session_operation(AsyncSession.run_sync)


class SovereignSqlaSessionFactory:
    """Minimal SQLAlchemy session factory matching the shape the
    sovereign vector backends consume.

    Exposes:

    - ``engine`` — the AsyncEngine; used by the vector factory to
      dispatch on dialect name (``postgresql`` vs ``sqlite``).
    - ``read_session()`` — async context manager yielding an
      ``AsyncSession`` for read-only work. No auto-commit; rely on
      SELECT-only usage.
    - ``write_session()`` — async context manager that commits on
      success and rolls back on exception, matching the feature-pkg
      convention.
    - ``close()`` — disposes of the underlying engine. Callers that
      build a factory in a long-lived service should call this on
      shutdown.

    The shape mirrors what feature pkgs construct internally; sovereign
    just needs ENOUGH of it to drive ``kestrel_sovereign.storage.vector``.
    """

    def __init__(
        self,
        engine: Any,
        *,
        sqlite_custody=None,
        sqlite_path: str | None = None,
        execution_owner=None,
    ) -> None:
        refuse_unfenced_executor(execution_owner)
        self._engine = engine
        self._execution_owner = execution_owner
        self._sqlite_custody = sqlite_custody
        # The SQLite file this engine opens, named by a close that overruns
        # its budget (#3460). ``None`` reports the database as unknown.
        self._sqlite_path = sqlite_path
        self._sqlite_close_diagnostics: _SQLiteCloseDiagnostics | None = None
        # SQLAlchemy's aiosqlite dialect acknowledges ``engine.dispose()``
        # after each driver's ``Connection.close()`` returns, which has the
        # same final-worker-turn gap as our primary SQLite backend.  Track the
        # raw driver connections at the engine boundary so this factory owns
        # their complete lifecycle too.
        self._sqlite_connections: list[Any] = []
        self._retired_sqlite_closes: _RetainedAiosqliteCloses = {}
        self._close_started = False
        self._active_sessions: set[_LifecycleBoundAsyncSession] = set()
        self._session_close_tasks: set[asyncio.Task[None]] = set()
        self._session_close_error: BaseException | None = None
        self._session_close_error_reported = False
        self._engine_dispose_task: asyncio.Task[None] | None = None
        self._engine_dispose_error: BaseException | None = None
        self._engine_dispose_error_reported = False
        self._driver_close_tasks: set[asyncio.Task[None]] = set()
        self._driver_close_error: BaseException | None = None
        self._driver_close_error_reported = False
        self._async_session = async_sessionmaker(
            engine,
            class_=_LifecycleBoundAsyncSession,
            expire_on_commit=False,
            factory_available=self._assert_session_connection_available,
        )
        if getattr(engine.dialect, "name", None) == "sqlite":
            event.listen(
                engine.sync_engine, "connect", self._track_sqlite_connection
            )
        # Guard cached factories, yielded sessions and retained engine-only
        # checkouts. Rollback/close deliberately remain available for retirement.
        for hook in ("do_connect", "connect", "checkout", "before_cursor_execute", "commit"):
            event.listen(engine.sync_engine, hook, self._assert_session_connection_available)

    @property
    def engine(self) -> Any:
        return self._engine

    @property
    def minimum_close_timeout_s(self) -> float:
        """Worker-drain reservation required by a SQLite-backed factory."""
        if getattr(self._engine.dialect, "name", None) != "sqlite":
            return 0.0
        return _minimum_close_timeout_s()

    @property
    def sqlite_connection_retirement_pending(self) -> bool:
        """Whether an in-flight driver close still owns a live worker."""
        _prune_retained_aiosqlite_closes(self._retired_sqlite_closes)
        return bool(self._retired_sqlite_closes)

    @property
    def retirement_pending(self) -> bool:
        """Whether any dialect-neutral engine resource is still retiring."""
        dispose_task = self._engine_dispose_task
        dispose_pending = dispose_task is not None and not dispose_task.done()
        session_pending = any(
            not task.done() for task in self._session_close_tasks
        )
        return (
            dispose_pending
            or session_pending
            or any(not task.done() for task in self._driver_close_tasks)
            or self.sqlite_connection_retirement_pending
        )

    def _capture_driver_close_result(self, task: asyncio.Task[None]) -> None:
        """Retain native receipts independently of a worker's liveness record."""
        self._capture_retirement_result("driver", task)
        self._driver_close_tasks.discard(task)

    def _capture_retirement_result(self, owner: str, task: asyncio.Task[None]) -> None:
        """A stronger late receipt is unseen even after a weaker publication."""
        error = owned_task_error(task)
        if not task.cancelled():
            task.exception()
        field = f"_{owner}_close_error" if owner != "engine" else "_engine_dispose_error"
        previous = getattr(self, field)
        selected = execution_terminal_error(previous, error)
        setattr(self, field, selected)
        if selected is not previous:
            setattr(self, field + "_reported", False)

    def _capture_engine_dispose_result(
        self, task: asyncio.Task[None]
    ) -> None:
        """Retrieve and retain a late engine-disposal failure.

        Kestrel can hard-close the outer shutdown coroutine while the
        independently owned disposal task continues.  Retrieving the task
        outcome here prevents an unobserved-task warning; retaining it lets the
        database lifecycle owner surface a late failure before it permits
        replacement resources.
        """
        self._capture_retirement_result("engine", task)

    def _capture_session_close_result(
        self, task: asyncio.Task[None]
    ) -> None:
        """Retrieve and retain a late active-session close failure."""
        self._capture_retirement_result("session", task)

    def finalize_retirement(self) -> None:
        """Validate completion and surface any previously unseen failure."""
        if self.retirement_pending:
            raise ConnectionError(
                "SQLAlchemy engine or connection retirement is still pending; "
                "the lifecycle owner cannot be detached"
            )

        dispose_task = self._engine_dispose_task
        if dispose_task is not None:
            self._capture_engine_dispose_result(dispose_task)

        for session_task in self._session_close_tasks:
            if session_task.done():
                self._capture_session_close_result(session_task)

        for driver_task in tuple(self._driver_close_tasks):
            if driver_task.done():
                self._capture_driver_close_result(driver_task)

        errors = (
            (
                "session",
                self._session_close_error,
                self._session_close_error_reported,
                "SQLAlchemy session close failed during lifecycle retirement",
            ),
            (
                "engine",
                self._engine_dispose_error,
                self._engine_dispose_error_reported,
                "SQLAlchemy engine disposal failed during lifecycle retirement",
            ),
            ("driver", self._driver_close_error, self._driver_close_error_reported,
             "SQLAlchemy raw driver close failed during lifecycle retirement"),
        )
        unseen = []
        for owner, error, reported, message in errors:
            if error is None or reported:
                continue
            if owner == "session":
                self._session_close_error_reported = True
            elif owner == "engine":
                self._engine_dispose_error_reported = True
            else:
                self._driver_close_error_reported = True
            unseen.append(error)
        terminal = execution_terminal_error(*unseen)
        if terminal is not None:
            raise terminal

    def _track_sqlite_connection(self, dbapi_connection: Any, _record: Any) -> None:
        """Remember one raw aiosqlite connection opened by this engine."""
        connection = getattr(dbapi_connection, "driver_connection", None)
        if connection is not None and connection not in self._sqlite_connections:
            self._sqlite_connections.append(connection)
            if self._close_started:
                self._begin_sqlite_connection_retirement((connection,))

    def _assert_session_connection_available(self, *args, **kwargs) -> None:
        """Permanently fence new work after factory close begins."""
        refuse_unfenced_executor(self._execution_owner)
        if self._close_started:
            raise ConnectionError(
                "SQLAlchemy session factory is closing or closed; "
                "new sessions are unavailable"
            )
        self._assert_sqlite_custody()

    def _assert_sqlite_custody(self, *args, **kwargs) -> None:
        if self._sqlite_custody is not None:
            self._sqlite_custody()

    @asynccontextmanager
    async def read_session(self):
        async with self._session_lifetime(write=False) as session:
            yield session

    @asynccontextmanager
    async def write_session(self):
        async with self._session_lifetime(write=True) as session:
            yield session

    @asynccontextmanager
    async def _session_lifetime(self, *, write):
        """Keep original work/commit control through rollback AND native exit."""
        self._assert_session_connection_available()
        session = self._async_session()
        self._active_sessions.add(session)
        original = None
        try:
            try:
                async with session:
                    try:
                        yield session
                        if write:
                            await session.commit()
                    except BaseException as error:
                        original = error
                        if write:
                            try:
                                await session.rollback()
                            except BaseException as cleanup:
                                original = execution_terminal_error(error, cleanup, retain_originals=True)
                        raise original
            except BaseException as cleanup:
                # AsyncSession.__aexit__ can itself fail after the body raised.
                # Never rely on implicit __context__ for custody provenance.
                raise execution_terminal_error(original, cleanup, retain_originals=True)
        finally:
            self._active_sessions.discard(session)

    async def close(self) -> None:
        """Dispose the engine and drain every SQLite worker it opened."""
        loop = asyncio.get_running_loop()
        diagnostics: _SQLiteCloseDiagnostics | None = None
        if getattr(self._engine.dialect, "name", None) == "sqlite":
            if self._sqlite_close_diagnostics is None:
                # Once, before the first close: a retry must not report a -wal
                # that the first attempt's driver closes have already consumed.
                self._sqlite_close_diagnostics = _capture_close_diagnostics(
                    self._sqlite_path, None
                )
            # Each attempt is timed against its own deadline.
            diagnostics = replace(
                self._sqlite_close_diagnostics, started_at=loop.time()
            )
        deadline = loop.time() + _minimum_close_timeout_s()
        if not self._close_started:
            self._close_started = True
            for session in tuple(self._active_sessions):
                close_task = create_owned_task(
                    session.close(),
                    name=f"sqla-session-close:{id(session)}",
                )
                close_task.add_done_callback(
                    self._capture_session_close_result
                )
                self._session_close_tasks.add(close_task)
        if self._engine_dispose_task is None:
            self._engine_dispose_task = create_owned_task(
                self._engine.dispose(),
                name=f"sqla-engine-dispose:{id(self)}",
            )
            self._engine_dispose_task.add_done_callback(
                self._capture_engine_dispose_result
            )

        # Establish durable ownership and start every raw SQLite close before
        # the first potentially blocking disposal await.  Kestrel's bounded
        # shutdown can hard-close this coroutine with ``GeneratorExit``; the
        # independent retained close/reaper tasks must already own each worker
        # when that happens.
        self._begin_sqlite_connection_retirement()

        lifecycle_error: BaseException | None = None
        try:
            # ``asyncio.wait`` does not transfer caller cancellation to the
            # owned task.  Unlike ``asyncio.shield`` on Python 3.14, it also
            # does not install a proxy that logs a late task exception after
            # hard GeneratorExit abandonment; our done callback retrieves and
            # retains that outcome for lifecycle finalization.
            owned_tasks = {
                self._engine_dispose_task,
                *self._session_close_tasks,
            }
            remaining = max(0.0, deadline - loop.time())
            _, pending = await asyncio.wait(
                owned_tasks,
                timeout=remaining,
            )
            if pending:
                lifecycle_error = ConnectionError(
                    "SQLAlchemy engine or session disposal did not complete "
                    f"within {_minimum_close_timeout_s():.2f}s"
                )
            else:
                self._capture_engine_dispose_result(
                    self._engine_dispose_task
                )
                for session_task in self._session_close_tasks:
                    self._capture_session_close_result(session_task)
                lifecycle_error = execution_terminal_error(
                    self._session_close_error, self._engine_dispose_error,
                )
        except (Exception, asyncio.CancelledError, BaseExceptionGroup) as exc:
            # ``AsyncEngine.dispose`` can fail before it visits all of its
            # connections (and deliberately leaves checked-out connections
            # alone).  The per-driver lifecycle tasks above remain independently
            # owned, wait for this one-shot disposal attempt, and then explicitly
            # close any worker the engine left alive.
            lifecycle_error = exc
            if self._engine_dispose_task.done():
                self._capture_engine_dispose_result(
                    self._engine_dispose_task
                )
                current_task = asyncio.current_task()
                caller_cancelled = (
                    isinstance(exc, asyncio.CancelledError)
                    and current_task is not None
                    and current_task.cancelling() > 0
                )
                if not caller_cancelled:
                    self._engine_dispose_error_reported = True

        cleanup_error: BaseException | None = None
        try:
            await self._close_live_sqlite_connections(
                deadline=deadline, diagnostics=diagnostics
            )
        except (Exception, asyncio.CancelledError, BaseExceptionGroup) as exc:
            cleanup_error = exc

        terminal = execution_terminal_error(
            lifecycle_error, cleanup_error,
            self._session_close_error, self._engine_dispose_error, self._driver_close_error,
        )
        if terminal is not None and is_execution_control_error(terminal):
            if terminal is self._session_close_error:
                self._session_close_error_reported = True
            elif terminal is self._engine_dispose_error:
                self._engine_dispose_error_reported = True
            elif terminal is self._driver_close_error:
                self._driver_close_error_reported = True
            raise terminal

        if cleanup_error is not None:
            # Do not erase a disposal failure when worker cleanup also fails.
            # The cleanup failure is primary because it describes the resource
            # that may still be alive; the disposal failure remains available
            # to callers as its explicit cause.
            if lifecycle_error is not None:
                if lifecycle_error is self._session_close_error:
                    self._session_close_error_reported = True
                elif lifecycle_error is self._engine_dispose_error:
                    self._engine_dispose_error_reported = True
                raise cleanup_error from lifecycle_error
            raise cleanup_error
        if lifecycle_error is not None:
            if lifecycle_error is self._session_close_error:
                self._session_close_error_reported = True
            elif lifecycle_error is self._engine_dispose_error:
                self._engine_dispose_error_reported = True
            raise lifecycle_error

    def _begin_sqlite_connection_retirement(self, connections: Any = None) -> None:
        """Synchronously retain close ownership for every live SQLite worker."""
        if getattr(self._engine.dialect, "name", None) != "sqlite":
            return

        _prune_retained_aiosqlite_closes(self._retired_sqlite_closes)
        candidates = (
            self._sqlite_connections if connections is None else connections
        )
        for connection in candidates:
            if not _aiosqlite_worker_is_alive(connection):
                continue
            if connection in self._retired_sqlite_closes:
                continue

            dispose_task = self._engine_dispose_task
            if dispose_task is None:
                raise RuntimeError(
                    "SQLite retirement started before engine disposal ownership"
                )

            async def close_after_dispose(
                owned_connection: Any = connection,
                owned_dispose_task: asyncio.Task[None] = dispose_task,
            ) -> None:
                # Disposal may be part-way through this driver's own
                # ``close()``. A second ``close()`` started beside it queues a
                # second stop sentinel behind the first; the worker exits on
                # the first and never answers the second, so this task would
                # wait forever (#3858). A cancellation therefore never ends
                # the wait for disposal: it is raised once the driver is
                # retired.
                pending_cancellation: BaseException | None = None
                while not owned_dispose_task.done():
                    try:
                        await asyncio.wait({owned_dispose_task})
                    except asyncio.CancelledError as exc:
                        pending_cancellation = execution_terminal_error(
                            pending_cancellation, exc, retain_originals=True,
                            isolate_control_originals=True,
                        )
                try:
                    owned_task_error(owned_dispose_task)
                except BaseException:
                    # Disposal failure is surfaced by ``close``.  It cannot
                    # transfer ownership of a checked-out/raw driver.
                    pass
                if _aiosqlite_worker_is_alive(owned_connection):
                    await owned_connection.close()
                if pending_cancellation is not None:
                    raise pending_cancellation

            raw_close_task = create_owned_task(
                close_after_dispose(),
                name=f"sqla-aiosqlite-driver-close:{id(connection)}",
            )
            self._driver_close_tasks.add(raw_close_task)
            raw_close_task.add_done_callback(self._capture_driver_close_result)
            _retain_aiosqlite_close(
                connection,
                raw_close_task,
                self._retired_sqlite_closes,
            )

    async def _close_live_sqlite_connections(
        self,
        *,
        deadline: float,
        diagnostics: _SQLiteCloseDiagnostics | None = None,
    ) -> None:
        """Close and drain every live raw SQLite driver this factory owns.

        ``AsyncEngine.dispose()`` normally sends the close sentinel for pooled
        connections, but a failure before that point and checked-out
        connections both remain this factory's responsibility.  The shared
        SQLite helper couples the sentinel with the bounded worker-drain wait,
        so no tracked worker can escape a failed disposal path.
        """
        if not self._sqlite_connections:
            return

        # A QueuePool can own several aiosqlite workers.  The factory advertises
        # one bounded shutdown reservation, so every connection must begin
        # retirement immediately and share one absolute deadline.  Sequential
        # per-connection waits both exceeded that reservation and could leave
        # later workers without retained lifecycle ownership when an outer
        # shutdown guard expired.
        self._begin_sqlite_connection_retirement()
        close_tasks: list[asyncio.Task[None]] = []
        for connection in self._sqlite_connections:
            retained = self._retired_sqlite_closes.get(connection)
            if retained is None:
                continue
            close_tasks.append(
                create_owned_task(
                    _close_aiosqlite_connection(
                        connection,
                        retained_closes=self._retired_sqlite_closes,
                        deadline=deadline,
                        close_task=retained.close_task,
                        cancel_close_task_on_timeout=False,
                        diagnostics=diagnostics,
                    ),
                    name=f"sqla-aiosqlite-close:{id(connection)}",
                )
            )

        failure = None
        for task in close_tasks:
            outcome = await await_owned_task(task)
            failure = execution_terminal_error(
                failure, outcome.error, outcome.cancellation,
            )
        # Engine disposal can retire a worker before its independently owned
        # raw-close task resumes. A dead worker is not that task's terminal
        # outcome. Join the remaining owners within the SAME close budget.
        raw_tasks = set(self._driver_close_tasks)
        if raw_tasks:
            done, pending = await asyncio.wait(
                raw_tasks, timeout=max(0.0, deadline - asyncio.get_running_loop().time()),
            )
            for task in done:
                self._capture_driver_close_result(task)
            if pending:
                failure = execution_terminal_error(failure, ConnectionError(
                    "SQLAlchemy raw driver close ownership is still pending"
                ))
        failure = execution_terminal_error(failure, self._driver_close_error)
        if failure is not None:
            raise failure


# Attribute name used to cache the session factory on the
# ``AsyncDatabase`` so repeated ``make_session_factory(db)`` calls reuse
# one engine + pool instead of leaking a new one per call. Codex
# review (P1) on the saved_items SQLA PR flagged that without this
# the per-request store construction would multiply engines —
# exhausting Postgres connection slots in production.
_CACHE_ATTR = "_sovereign_sqla_factory"
_RETIREMENT_OWNER_ATTR = "_sovereign_sqla_retirement_owner"


def make_session_factory(
    db: Any, *, native_read_fallback: bool = False,
) -> SovereignSqlaSessionFactory | None:
    """Build (or reuse) a SQLAlchemy session factory pointed at the
    same database as ``db``.

    Native read consumers may explicitly select their existing fenced backend
    when execution custody is present. Validate ALL original scopes first and
    return None BEFORE attempting an unsupported SQLAlchemy executor. This is
    capability selection, not recovery from a caught authority denial: it
    creates no engine, connection, replacement owner or renewed permission.
    Default callers retain the strict unfenced-executor refusal.

    Caching: the factory is stashed on ``db`` under
    ``_sovereign_sqla_factory``. Subsequent calls with the same ``db``
    return that cached factory. This binds the factory's lifetime to
    the ``AsyncDatabase``'s lifetime (= app lifetime in typical
    deployments) rather than the caller's lifetime — important
    because ``SavedItemsStore`` is constructed per-request, and one
    pool per request would saturate Postgres slot limits in a hurry.

    ``AsyncDatabase.close()`` disposes the cached factory if attached
    (see the patch in this same PR), so app shutdown still releases
    the engine cleanly.

    Args:
        db: An ``AsyncDatabase`` (or a subclass / mock that exposes
            ``backend_type`` and the backend-specific attributes —
            ``backend.db_path`` for SQLite or ``backend._dsn`` for PG).
        native_read_fallback: The caller already has an original native read
            path. Select it before SQLAlchemy construction when custody is
            present, after validating every original admission.

    Returns:
        A ``SovereignSqlaSessionFactory`` — the same instance for the
        lifetime of ``db``. With ``native_read_fallback=True``, None means
        a valid custody-bound caller must use its existing native backend.
        Revocation, uncertainty and cleanup-only admission still raise.

    Raises:
        NotImplementedError: If ``db`` was constructed via
            ``AsyncDatabase.from_pool(...)`` — that path holds only an
            external asyncpg pool with no recoverable DSN. Callers in
            that situation must construct a session factory explicitly
            from the same DSN they used to build the pool. The
            error message points the way.
        ValueError: If the backend type isn't recognized.
    """
    execution_owner = getattr(db, "backend", None)
    if native_read_fallback:
        require_execution_work(execution_owner)
        if current_execution_custody(execution_owner):
            return None
    refuse_unfenced_executor(execution_owner)
    retirement_owner = getattr(db, _RETIREMENT_OWNER_ATTR, None)
    if isinstance(retirement_owner, SovereignSqlaSessionFactory):
        raise ConnectionError(
            "SQLAlchemy session factory retirement is still owned by this "
            "database; a replacement factory cannot be created"
        )

    cached = getattr(db, _CACHE_ATTR, None)
    # Only treat the cached attribute as a hit if it's the right type —
    # ``MagicMock`` and similar test doubles auto-vivify attributes on
    # ``getattr``, which would otherwise short-circuit this function and
    # return the mock itself instead of a real factory.
    if isinstance(cached, SovereignSqlaSessionFactory):
        cached._assert_session_connection_available()
        return cached

    backend_type = getattr(db, "backend_type", None)

    if backend_type == "sqlite":
        # SQLiteBackend exposes ``db_path``. The aiosqlite dialect uses
        # the same driver under the hood, so two engines against the
        # same file (one bare asyncio, one SQLAlchemy) are safe as long
        # as both honor SQLite's file locking — which they do.
        path = getattr(db.backend, "db_path", None)
        if path is None:
            raise ValueError(
                "SQLite AsyncDatabase backend is missing db_path; "
                "cannot build a SQLAlchemy session factory."
            )
        # ``:memory:`` databases are PER-CONNECTION on SQLite, so two
        # engines would see different empty stores. Refuse rather than
        # silently miss data. Callers wanting in-memory SQLAlchemy
        # access alongside an in-memory AsyncDatabase should pass an
        # explicit shared-file path instead.
        if path == ":memory:":
            raise ValueError(
                "make_session_factory does not support ':memory:' SQLite "
                "databases — the SQLAlchemy engine would see a separate, "
                "empty in-memory store. Use a real file path or construct "
                "the SovereignSqlaSessionFactory directly from a shared "
                "engine."
            )
        custody = None
        if isinstance(db.backend, SQLiteBackend):
            custody = db.backend.assert_connected_file_still_valid
            custody()
            # mode=rw never creates a replacement if the retained path
            # disappears between the custody check and driver acquisition.
            engine = create_async_engine(URL.create("sqlite+aiosqlite", database=Path(path).as_uri(), query={"mode": "rw", "uri": "true"}))
        else:
            engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
        factory = SovereignSqlaSessionFactory(
            engine, sqlite_custody=custody, sqlite_path=path, execution_owner=db.backend
        )
        try:
            setattr(db, _CACHE_ATTR, factory)
        except Exception:
            # ``db`` may be a stubbed namespace in tests that doesn't
            # accept new attributes — caching is best-effort.
            pass
        return factory

    if backend_type == "postgres":
        # PostgresBackend may or may not have a usable DSN: ``.create``
        # constructs from a DSN and stashes it on ``_dsn``;
        # ``.from_pool`` skips that and ``_dsn`` is None. The latter
        # case can't recover the DSN from the asyncpg pool, so we
        # surface a clear error.
        dsn = getattr(db.backend, "_dsn", None)
        if dsn is None:
            raise NotImplementedError(
                "AsyncDatabase was built via from_pool() with no DSN; "
                "cannot derive a SQLAlchemy session factory. Build the "
                "factory directly with create_async_engine() against the "
                "same DSN used for the asyncpg pool."
            )
        # asyncpg DSNs use ``postgresql://`` (or ``postgres://``); the
        # SQLAlchemy asyncpg dialect needs ``postgresql+asyncpg://``.
        # Rewrite the scheme so callers don't have to.
        if dsn.startswith("postgres://"):
            sqla_dsn = "postgresql+asyncpg://" + dsn[len("postgres://"):]
        elif dsn.startswith("postgresql://"):
            sqla_dsn = "postgresql+asyncpg://" + dsn[len("postgresql://"):]
        else:
            sqla_dsn = dsn
        engine = create_async_engine(sqla_dsn)
        factory = SovereignSqlaSessionFactory(engine, execution_owner=db.backend)
        try:
            setattr(db, _CACHE_ATTR, factory)
        except Exception:
            pass
        return factory

    raise ValueError(
        f"Unrecognized AsyncDatabase backend_type={backend_type!r}; "
        "cannot build a SQLAlchemy session factory."
    )
