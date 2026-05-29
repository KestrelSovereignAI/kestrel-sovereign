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

from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
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

    @property
    def engine(self) -> Any:
        return self._engine

    @asynccontextmanager
    async def read_session(self):
        async with self._async_session() as session:
            yield session

    @asynccontextmanager
    async def write_session(self):
        async with self._async_session() as session:
            try:
                yield session
                await session.commit()
            except BaseException:
                await session.rollback()
                raise

    async def close(self) -> None:
        """Dispose the engine. Long-lived services should call this on shutdown."""
        await self._engine.dispose()


# Attribute name used to cache the session factory on the
# ``AsyncDatabase`` so repeated ``make_session_factory(db)`` calls reuse
# one engine + pool instead of leaking a new one per call. Codex
# review (P1) on the saved_items SQLA PR flagged that without this
# the per-request store construction would multiply engines —
# exhausting Postgres connection slots in production.
_CACHE_ATTR = "_sovereign_sqla_factory"


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
