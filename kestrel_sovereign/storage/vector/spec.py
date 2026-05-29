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

    The spec separates THREE concerns that earlier per-table backends
    conflated:

    1. **What keys must be present in a filter dict** (validation only).
       Set via ``required_filter_keys``.
    2. **Which filter keys map to WHERE clauses** (server-side filtering).
       Set via ``filter_columns``. A key here is applied as
       ``column == filter[key]`` when present in the call.
    3. **Which filter key scopes the session via TenantContext** (the
       session-construction-time loader criterion). Set via
       ``tenant_id_filter_key``. The corresponding value never appears
       in the WHERE clause — it's used to wrap the read session.

    The separation matters because some tables are tenant-scoped purely
    via ``TenantContext`` loader criteria (story-archive's ``Event``)
    while others have no tenant scoping at all (single-agent
    ``saved_items``). Mixing the two roles in one map made the older
    story-archive backends require their callers to also be running
    against a tenant-aware session factory — a hidden coupling.

    Attributes:
        entity: The SQLAlchemy mapped class (e.g. ``Event``). Used as
            the FROM target of the ``select(...)`` statement.
        id_column: The ``InstrumentedAttribute`` identifying the row's
            primary key. Returned as the first element of each result
            tuple from ``knn()``.
        embedding_column: The ``InstrumentedAttribute`` holding the
            embedding. On PG it should be typed ``Vector(dim)`` (or
            ``PortableVector(dim)``); on SQLite it can be a BLOB.
        dimension: Embedding dimension (e.g. 1536 for OpenAI ada-002).
        required_filter_keys: Filter keys every caller MUST supply.
            Missing keys → ``VectorSearchError``. Keys here may also
            appear in ``filter_columns`` (then they're additionally
            applied as WHERE clauses) or ``tenant_id_filter_key`` (then
            their value wraps the session), or neither (just validated).
        filter_columns: Filter keys mapped to ``InstrumentedAttribute``
            references. When a key is present in ``filter`` AND in this
            dict, ``column == filter[key]`` is appended to the WHERE
            clause. Keys can be required (also in
            ``required_filter_keys``) or optional.
        tenant_id_filter_key: Optional key whose value wraps the read
            session in ``TenantContext.use(<that filter value>)``. Must
            also appear in ``required_filter_keys`` so the value is
            guaranteed available.

    Frozen so a spec can be shared safely across async tasks.
    """

    entity: Any
    id_column: Any
    embedding_column: Any
    dimension: int
    required_filter_keys: tuple = field(default_factory=tuple)
    filter_columns: Dict[str, Any] = field(default_factory=dict)
    tenant_id_filter_key: str | None = None

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise ValueError(
                f"VectorTableSpec.dimension must be positive, got {self.dimension}"
            )
        if (
            self.tenant_id_filter_key is not None
            and self.tenant_id_filter_key not in self.required_filter_keys
        ):
            # A tenant_id_filter_key that isn't in required_filter_keys
            # would silently never get applied — fail loudly at config
            # time so the consumer fixes the spec.
            raise ValueError(
                f"VectorTableSpec.tenant_id_filter_key={self.tenant_id_filter_key!r} "
                "is not present in required_filter_keys; tenant scoping would "
                "be skipped."
            )
