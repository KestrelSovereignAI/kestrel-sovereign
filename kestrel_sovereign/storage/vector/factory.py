"""Dialect-aware factory for vector-search backends.

Picks the best concrete backend for the session factory's engine:

- Postgres → :class:`PgVectorBackend` (fast, pgvector ``<=>`` operator)
- Everything else (SQLite today) → :class:`PurePythonBackend` (slow,
  in-Python cosine)

When the FEAT-8 ``SqliteVecBackend`` lands, this factory grows a third
branch that picks it over the pure-Python fallback whenever the
``sqlite_vec`` package is importable and the dialect is sqlite.

The factory exists so callers can write ``get_vector_backend(sf, spec)``
once and not care about dialect dispatch every time they construct a
search service.
"""

from __future__ import annotations

import logging
from typing import Any

from .spec import VectorTableSpec

logger = logging.getLogger(__name__)


def get_vector_backend(session_factory: Any, spec: VectorTableSpec) -> Any:
    """Return the right ``VectorSearchBackend`` for the engine dialect.

    Args:
        session_factory: An object exposing an ``engine`` attribute whose
            dialect we dispatch on. SQLAlchemy's ``AsyncSessionFactory``
            (the shape feature pkgs use) satisfies this contract.
        spec: ``VectorTableSpec`` describing the target table.

    Returns:
        A backend instance whose ``knn()`` method matches the SDK's
        ``VectorSearchBackend`` Protocol shape.
    """
    dialect = session_factory.engine.dialect.name

    if dialect == "postgresql":
        # Lazy import keeps pgvector off the SQLite-only path. Pgvector
        # the Python pkg is tiny and pure-Python, so this is just hygiene
        # rather than a hard import requirement, but it also means a
        # SqliteVecBackend addition later doesn't have to compete with
        # pgvector's import side-effects.
        from .pg import PgVectorBackend
        logger.info(
            "Vector backend: PgVectorBackend (entity=%s, dim=%d)",
            spec.entity.__name__,
            spec.dimension,
        )
        return PgVectorBackend(session_factory, spec)

    # FEAT-8 hook point: SqliteVecBackend slots in here as
    #   if dialect == "sqlite":
    #       try:
    #           import sqlite_vec  # noqa: F401
    #           from .sqlite_vec import SqliteVecBackend
    #           return SqliteVecBackend(session_factory, spec)
    #       except ImportError:
    #           pass
    # falling through to PurePython when sqlite_vec isn't installed.

    from .python import PurePythonBackend
    logger.info(
        "Vector backend: PurePythonBackend (dialect=%s, entity=%s, dim=%d)",
        dialect, spec.entity.__name__, spec.dimension,
    )
    return PurePythonBackend(session_factory, spec)
