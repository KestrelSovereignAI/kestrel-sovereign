"""VectorTableSpec — parameterizes the generic vector-search backends.

The backends in this package (``PgVectorBackend``, ``PurePythonBackend``,
and the FEAT-8 ``SqliteVecBackend`` when it lands) are written against
abstract column references rather than a hard-coded entity. Callers
describe their table with a spec; the backend then issues kNN queries
against it.

This is what lets the same backends serve story-archive's ``story_events``,
healthcare's clinical-note embeddings, and any future SQLAlchemy-mapped
table without each consumer copy-pasting the cosine code.

The spec uses SQLAlchemy ``InstrumentedAttribute`` references rather than
string column names. That gives us:

- Compile-time-ish validation (a typo'd column name fails at import,
  not at first query).
- Real expression-language objects, so the backends can compose them
  into WHERE clauses, ``op('<=>')`` operators, and ``.label(...)`` aliases
  without hand-rolling SQL strings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class VectorTableSpec:
    """Description of a vector-indexed SQLAlchemy table.

    Attributes:
        entity: The SQLAlchemy mapped class (e.g. ``Event``). Used as the
            FROM target of the ``select(...)`` statement.
        id_column: The InstrumentedAttribute identifying the row's
            primary key (returned by ``knn()`` as the result tuple's
            first element).
        embedding_column: The InstrumentedAttribute holding the embedding.
            On PG it should be typed ``Vector(dim)`` (or
            ``PortableVector(dim)``); on SQLite it can be a BLOB. The
            backends introspect the dialect at query time and pick the
            right code path.
        dimension: Embedding dimension (e.g. 1536 for OpenAI ada-002).
            Used by the pgvector backend to construct the
            ``cast(qvec_literal, Vector(dim))`` expression node.
        required_filter_columns: Map of filter-name → InstrumentedAttribute
            for filters every caller MUST supply (e.g.
            ``{"timeline_id": Event.timeline_id, "tenant_id":
            Event.tenant_id}``). The backend raises ``VectorSearchError``
            if any required filter is missing from a ``knn()`` call.
        optional_filter_columns: Map of filter-name → InstrumentedAttribute
            for filters callers MAY supply (e.g.
            ``{"event_type": Event.event_type, "is_sensitive":
            Event.is_sensitive}``). The backend applies these as extra
            WHERE clauses when present.
        tenant_id_filter_key: Name of the filter that should be passed to
            ``TenantContext.use(...)`` to scope the inner read session.
            If ``None``, no TenantContext wrapping happens (e.g. a
            single-tenant store like agent saved_items). When set, this
            key MUST also appear in ``required_filter_columns``.

    Frozen so a spec can be safely shared across goroutines / async
    tasks (the backends store a reference and use it on every query).
    """

    entity: Any
    id_column: Any
    embedding_column: Any
    dimension: int
    required_filter_columns: Dict[str, Any] = field(default_factory=dict)
    optional_filter_columns: Dict[str, Any] = field(default_factory=dict)
    tenant_id_filter_key: str | None = None

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise ValueError(
                f"VectorTableSpec.dimension must be positive, got {self.dimension}"
            )
        if (
            self.tenant_id_filter_key is not None
            and self.tenant_id_filter_key not in self.required_filter_columns
        ):
            # A tenant_id_filter_key that isn't in required_filter_columns
            # would silently never get applied — fail loudly at config
            # time so the consumer fixes the spec.
            raise ValueError(
                f"VectorTableSpec.tenant_id_filter_key={self.tenant_id_filter_key!r} "
                "is not present in required_filter_columns; tenant scoping would "
                "be skipped."
            )
