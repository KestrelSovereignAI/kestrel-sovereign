"""Generic vector-search backends for SQLAlchemy-mapped tables.

Public surface:

- :class:`VectorTableSpec` — describes the target table (entity, embedding
  column, dimension, required + optional filter columns, optional
  ``tenant_id_filter_key`` for multi-tenant tables).
- :class:`PgVectorBackend` — fast cosine kNN via the pgvector ``<=>``
  operator. Postgres only.
- :class:`PurePythonBackend` — SQLite-friendly fallback that fetches every
  matching row and computes cosine in numpy. Slow at scale but works
  against any SQLAlchemy dialect.
- :func:`get_vector_backend` — dialect-aware factory; picks the right
  concrete backend.
- :class:`VectorSearchError` — raised on malformed requests (missing
  required filter, wrong embedding length, etc.).

These were lifted out of ``kestrel-feature-story-archive`` so that
story-archive, healthcare, observability, and (in the future)
saved_items_store / memory_manager can all share one implementation.
The earlier story-archive ``PgVectorBackend`` / ``PurePythonBackend``
classes were hardcoded to the ``Event`` entity; the spec parameter is
what generalises them.

See kestrel-sovereign #TBD (issue tracker) for the FEAT-8 sqlite-vec
follow-up that wires a third backend into the factory.
"""

from .exceptions import VectorSearchError
from .factory import get_vector_backend
from .pg import PgVectorBackend
from .python import PurePythonBackend
from .spec import VectorTableSpec

__all__ = [
    "VectorSearchError",
    "VectorTableSpec",
    "PgVectorBackend",
    "PurePythonBackend",
    "get_vector_backend",
]
