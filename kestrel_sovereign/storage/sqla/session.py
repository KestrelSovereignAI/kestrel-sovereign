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
from typing import Any

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from kestrel_sovereign.storage.db.interface import ConnectionError
from kestrel_sovereign.storage.db.sqlite import (
    _RetainedAiosqliteCloses,
    _aiosqlite_worker_is_alive,
    _close_aiosqlite_connection,
    _minimum_close_timeout_s,
    _prune_retained_aiosqlite_closes,
    _retain_aiosqlite_close,
)


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

    def __init__(self, engine: Any) -> None:
        self._engine = engine
        self._async_session = async_sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )
        # SQLAlchemy's aiosqlite dialect acknowledges ``engine.dispose()``
        # after each driver's ``Connection.close()`` returns, which has the
        # same final-worker-turn gap as our primary SQLite backend.  Track the
        # raw driver connections at the engine boundary so this factory owns
        # their complete lifecycle too.
        self._sqlite_connections: list[Any] = []
        self._retired_sqlite_closes: _RetainedAiosqliteCloses = {}
        self._close_started = False
        self._engine_dispose_task: asyncio.Task[None] | None = None
        self._engine_dispose_error: BaseException | None = None
        self._engine_dispose_error_reported = False
        if getattr(engine.dialect, "name", None) == "sqlite":
            event.listen(
                engine.sync_engine, "connect", self._track_sqlite_connection
            )

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
        return (
            dispose_task is not None and not dispose_task.done()
        ) or self.sqlite_connection_retirement_pending

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
        if task.cancelled():
            if self._engine_dispose_error is None:
                self._engine_dispose_error = ConnectionError(
                    "SQLAlchemy engine disposal was cancelled before completion"
                )
            return
        error = task.exception()
        if error is not None and self._engine_dispose_error is None:
            self._engine_dispose_error = error

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

        error = self._engine_dispose_error
        if error is None or self._engine_dispose_error_reported:
            return
        self._engine_dispose_error_reported = True
        if isinstance(error, Exception):
            raise error
        raise ConnectionError(
            "SQLAlchemy engine disposal failed during lifecycle retirement"
        ) from error

    def _track_sqlite_connection(self, dbapi_connection: Any, _record: Any) -> None:
        """Remember one raw aiosqlite connection opened by this engine."""
        connection = getattr(dbapi_connection, "driver_connection", None)
        if connection is not None and connection not in self._sqlite_connections:
            self._sqlite_connections.append(connection)
            if self._close_started:
                self._begin_sqlite_connection_retirement((connection,))

    def _assert_session_connection_available(self) -> None:
        """Permanently fence new work after factory close begins."""
        if self._close_started:
            raise ConnectionError(
                "SQLAlchemy session factory is closing or closed; "
                "new sessions are unavailable"
            )

    @asynccontextmanager
    async def read_session(self):
        self._assert_session_connection_available()
        async with self._async_session() as session:
            yield session

    @asynccontextmanager
    async def write_session(self):
        self._assert_session_connection_available()
        async with self._async_session() as session:
            try:
                yield session
                await session.commit()
            except BaseException:
                await session.rollback()
                raise

    async def close(self) -> None:
        """Dispose the engine and drain every SQLite worker it opened."""
        self._close_started = True
        loop = asyncio.get_running_loop()
        deadline = loop.time() + _minimum_close_timeout_s()
        if self._engine_dispose_task is None:
            self._engine_dispose_task = loop.create_task(
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

        dispose_error: BaseException | None = None
        try:
            # ``asyncio.wait`` does not transfer caller cancellation to the
            # owned task.  Unlike ``asyncio.shield`` on Python 3.14, it also
            # does not install a proxy that logs a late task exception after
            # hard GeneratorExit abandonment; our done callback retrieves and
            # retains that outcome for lifecycle finalization.
            await asyncio.wait({self._engine_dispose_task})
            self._engine_dispose_task.result()
        except (Exception, asyncio.CancelledError) as exc:
            # ``AsyncEngine.dispose`` can fail before it visits all of its
            # connections (and deliberately leaves checked-out connections
            # alone).  The per-driver lifecycle tasks above remain independently
            # owned, wait for this one-shot disposal attempt, and then explicitly
            # close any worker the engine left alive.
            dispose_error = exc
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
            await self._close_live_sqlite_connections(deadline=deadline)
        except (Exception, asyncio.CancelledError) as exc:
            cleanup_error = exc

        if cleanup_error is not None:
            # Do not erase a disposal failure when worker cleanup also fails.
            # The cleanup failure is primary because it describes the resource
            # that may still be alive; the disposal failure remains available
            # to callers as its explicit cause.
            if dispose_error is not None:
                raise cleanup_error from dispose_error
            raise cleanup_error
        if dispose_error is not None:
            raise dispose_error

    def _begin_sqlite_connection_retirement(self, connections: Any = None) -> None:
        """Synchronously retain close ownership for every live SQLite worker."""
        if getattr(self._engine.dialect, "name", None) != "sqlite":
            return

        _prune_retained_aiosqlite_closes(self._retired_sqlite_closes)
        loop = asyncio.get_running_loop()
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
                try:
                    await asyncio.wait({owned_dispose_task})
                    owned_dispose_task.result()
                except (Exception, asyncio.CancelledError):
                    # Disposal failure is surfaced by ``close``.  It cannot
                    # transfer ownership of a checked-out/raw driver.
                    pass
                if _aiosqlite_worker_is_alive(owned_connection):
                    await owned_connection.close()

            raw_close_task = loop.create_task(
                close_after_dispose(),
                name=f"sqla-aiosqlite-driver-close:{id(connection)}",
            )
            _retain_aiosqlite_close(
                connection,
                raw_close_task,
                self._retired_sqlite_closes,
            )

    async def _close_live_sqlite_connections(self, *, deadline: float) -> None:
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
        loop = asyncio.get_running_loop()
        close_tasks: list[asyncio.Task[None]] = []
        for connection in self._sqlite_connections:
            retained = self._retired_sqlite_closes.get(connection)
            if retained is None:
                continue
            close_tasks.append(
                loop.create_task(
                    _close_aiosqlite_connection(
                        connection,
                        retained_closes=self._retired_sqlite_closes,
                        deadline=deadline,
                        close_task=retained.close_task,
                    ),
                    name=f"sqla-aiosqlite-close:{id(connection)}",
                )
            )

        if not close_tasks:
            return

        pending_cancellation: asyncio.CancelledError | None = None
        try:
            results = await asyncio.gather(
                *close_tasks, return_exceptions=True
            )
        except asyncio.CancelledError as exc:
            # Preserve the existing cancellation contract: each connection
            # gets the cancellation signal, then the factory waits within the
            # shared lifecycle deadline so the helper can either finish or
            # install retained ownership before cancellation reaches its
            # caller.
            pending_cancellation = exc
            for task in close_tasks:
                task.cancel()
            try:
                results = await asyncio.gather(
                    *close_tasks, return_exceptions=True
                )
            except asyncio.CancelledError:
                # Repeated caller cancellation may end this wait.  Each close
                # helper retains any worker that is still alive in its
                # BaseException guard, so no connection loses ownership.
                for task in close_tasks:
                    task.cancel()
                raise

        if pending_cancellation is not None:
            raise pending_cancellation

        cleanup_errors = [
            result for result in results if isinstance(result, BaseException)
        ]

        if cleanup_errors:
            primary_error = cleanup_errors[0]
            for extra_error in cleanup_errors[1:]:
                primary_error.add_note(
                    "Additional SQLite SQLAlchemy driver close failure: "
                    f"{extra_error!r}"
                )
            raise primary_error


# Attribute name used to cache the session factory on the
# ``AsyncDatabase`` so repeated ``make_session_factory(db)`` calls reuse
# one engine + pool instead of leaking a new one per call. Codex
# review (P1) on the saved_items SQLA PR flagged that without this
# the per-request store construction would multiply engines —
# exhausting Postgres connection slots in production.
_CACHE_ATTR = "_sovereign_sqla_factory"
_RETIREMENT_OWNER_ATTR = "_sovereign_sqla_retirement_owner"


def make_session_factory(db: Any) -> SovereignSqlaSessionFactory:
    """Build (or reuse) a SQLAlchemy session factory pointed at the
    same database as ``db``.

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

    Returns:
        A ``SovereignSqlaSessionFactory`` — the same instance for the
        lifetime of ``db``.

    Raises:
        NotImplementedError: If ``db`` was constructed via
            ``AsyncDatabase.from_pool(...)`` — that path holds only an
            external asyncpg pool with no recoverable DSN. Callers in
            that situation must construct a session factory explicitly
            from the same DSN they used to build the pool. The
            error message points the way.
        ValueError: If the backend type isn't recognized.
    """
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
        engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
        factory = SovereignSqlaSessionFactory(engine)
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
        factory = SovereignSqlaSessionFactory(engine)
        try:
            setattr(db, _CACHE_ATTR, factory)
        except Exception:
            pass
        return factory

    raise ValueError(
        f"Unrecognized AsyncDatabase backend_type={backend_type!r}; "
        "cannot build a SQLAlchemy session factory."
    )
