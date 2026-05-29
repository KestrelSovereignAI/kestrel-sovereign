"""PgVectorBackend — fast cosine kNN using pgvector's ``<=>`` operator.

Generic over a ``VectorTableSpec`` so the same backend can serve any
SQLAlchemy-mapped table that holds an embedding column.

Requires pgvector to be installed in the target Postgres instance
(``CREATE EXTENSION IF NOT EXISTS vector``). The Python ``pgvector`` pkg
is imported here to get the ``Vector`` type that we cast a query-vector
literal to — the cast is what lets us use the cosine-distance operator
on a column declared as a ``TypeDecorator`` (which doesn't proxy
pgvector's comparator methods).

Tenant scoping: if the spec sets ``tenant_id_filter_key``, the read
session is opened inside ``TenantContext.use(<that filter value>)``. This
matches the multi-tenant pattern feature pkgs already use.
"""

from __future__ import annotations

import logging
import struct
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import String, and_, bindparam, cast, func, literal, select

from .exceptions import VectorSearchError
from .spec import VectorTableSpec

logger = logging.getLogger(__name__)


class PgVectorBackend:
    """Fast vector search using pgvector's ``<=>`` operator.

    Args:
        session_factory: An object with an async ``read_session()``
            context manager that yields an ``AsyncSession``.
        spec: ``VectorTableSpec`` describing the target table.

    Class invariants:
        - Only works against the PostgreSQL dialect with pgvector
          installed. The factory (``get_vector_backend``) is responsible
          for picking this backend only when the dialect supports it.
        - ``query_embedding`` is little-endian float32 packed bytes
          matching ``spec.dimension``. Wrong-length / wrong-dtype inputs
          either raise or return empty results.
    """

    supports_filters = True

    def __init__(self, session_factory: Any, spec: VectorTableSpec) -> None:
        self._sf = session_factory
        self._spec = spec

    async def knn(
        self,
        query_embedding: bytes,
        k: int,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float]]:
        """Find the ``k`` nearest neighbors by cosine similarity.

        Args:
            query_embedding: Packed float32 bytes (little-endian) of length
                ``spec.dimension * 4``. Mismatched lengths raise.
            k: Maximum number of neighbors to return.
            filter: Required-and-optional filter map. Every key in
                ``spec.required_filter_columns`` MUST be present; keys in
                ``spec.optional_filter_columns`` are applied if present.
                Unknown keys are ignored (so callers can pass through
                extra context without breakage).

        Returns:
            List of ``(row_id_str, similarity_float)`` tuples sorted by
            similarity DESC. Similarity is ``1 - cosine_distance``, so
            higher is more similar. The list may be shorter than ``k``
            if fewer rows match the filter or have embeddings.
        """
        # Local import — pgvector is an optional runtime dep (loaded
        # only on the PG path). Importing at module top would force the
        # SQLite-only call sites to have it installed too.
        from pgvector.sqlalchemy import Vector

        # Normalize filter AFTER validation so unfiltered specs (no
        # required_filter_columns) can be called with ``filter=None`` —
        # validation passes, then we work against an empty dict for the
        # rest of the method without tripping on ``None.get``.
        self._require_filters(filter)
        filter = filter or {}

        query_vec = self._unpack(query_embedding)

        # Zero-norm query → cosine is undefined (would yield NaN). Match
        # PurePythonBackend's short-circuit so the two backends are
        # behaviourally identical at the edge.
        if not any(v != 0.0 for v in query_vec):
            logger.warning(
                "PgVectorBackend.knn: query embedding has zero norm — returning []"
            )
            return []

        spec = self._spec
        # Match PurePythonBackend's row-level zero-norm skip via a SQL
        # WHERE clause. pgvector's cosine distance against the all-zero
        # vector is undefined (the operator returns NaN), and an HNSW
        # index scan can surface those rows without filtering them out.
        # Comparing ``vector_norm(embedding) > 0`` is the canonical
        # pgvector way to keep only non-zero vectors in the candidate
        # set; ``isnot(None)`` alone isn't enough because a zero-vector
        # row is non-NULL. (Caught by codex review on the sovereign
        # vector-lift PR.)
        conditions = [
            spec.embedding_column.isnot(None),
            func.vector_norm(spec.embedding_column) > 0,
        ]
        for key, column in spec.filter_columns.items():
            if key in filter:
                conditions.append(column == filter[key])

        # See story-archive's pg.py for the full rationale on why this
        # cast(bindparam → Vector(dim)) shape is needed; in short:
        #   - The column type is a TypeDecorator (PortableVector); pgvector's
        #     comparator methods don't proxy through it.
        #   - type_coerce(list, Vector(dim)) double-binds and crashes
        #     with "expected ndim to be 1".
        #   - literal_column / text don't support .label or arithmetic.
        # The String→Vector cast is the only shape that gives us a real
        # expression node we can compose with ``op('<=>')``.
        qvec_literal = "[" + ",".join(repr(float(v)) for v in query_vec) + "]"
        qvec_expr = cast(
            bindparam("qvec", value=qvec_literal, type_=String),
            Vector(spec.dimension),
        )
        distance = spec.embedding_column.op("<=>")(qvec_expr)
        similarity = (literal(1) - distance).label("similarity")

        stmt = (
            select(spec.id_column, similarity)
            .where(and_(*conditions))
            .order_by(distance)
            .limit(k)
        )

        async with self._open_read_session(filter) as session:
            result = await session.execute(stmt)
            return [(str(row[0]), float(row[1])) for row in result.all()]

    # ----------------------------------------------------------------- helpers

    def _open_read_session(self, filter: Dict[str, Any]):
        """Return an async-context-manager yielding a read session.

        When ``spec.tenant_id_filter_key`` is set we wrap the session in
        ``TenantContext.use(<tenant_id>)`` so the underlying ORM applies
        the same tenant scoping it uses for write paths.
        """
        spec = self._spec
        if spec.tenant_id_filter_key is None:
            return self._sf.read_session()
        tenant_id = filter[spec.tenant_id_filter_key]
        # Imported lazily so a sovereign install that doesn't need
        # multi-tenant scoping (single-agent saved_items, etc.) doesn't
        # have to depend on kestrel_feature_entities.
        from kestrel_feature_entities import TenantContext

        return _TenantScopedSession(self._sf, tenant_id)

    def _require_filters(self, filter: Optional[Dict[str, Any]]) -> None:
        required = set(self._spec.required_filter_keys)
        if filter is None:
            missing = required
        else:
            missing = required - set(filter.keys())
        if missing:
            raise VectorSearchError(
                f"{type(self).__name__}.knn requires filter keys "
                f"{sorted(required)}; missing {sorted(missing)}."
            )

    def _unpack(self, data: bytes) -> List[float]:
        """Unpack packed float32 bytes into a Python list of floats.

        Raises ``VectorSearchError`` if the length doesn't match
        ``spec.dimension * 4`` — silently truncating a wrong-sized
        embedding would produce nonsense distances.
        """
        expected = self._spec.dimension * 4
        if len(data) != expected:
            raise VectorSearchError(
                f"query_embedding has {len(data)} bytes; "
                f"expected {expected} (dim={self._spec.dimension} * 4)."
            )
        return list(struct.unpack(f"<{self._spec.dimension}f", data))

    # Back-compat alias for tests that exercised the older story-archive
    # backends' internals. New callers should use ``_unpack`` directly
    # (or rely on the backend's ``knn()``).
    _deserialize_embedding = _unpack


class _TenantScopedSession:
    """Async-context-manager that wraps a read session in TenantContext.

    Why this exists: the obvious form
        with TenantContext.use(tid):
            async with sf.read_session() as s: ...
    requires the caller to nest both managers, but ``knn()`` returns the
    session from ``_open_read_session()``; the TenantContext has to live
    for the duration of the inner ``async with``. Encapsulating both in
    one async-context-manager keeps the call-site clean.
    """

    def __init__(self, session_factory: Any, tenant_id: Any) -> None:
        self._sf = session_factory
        self._tenant_id = tenant_id
        self._tenant_cm: Any = None
        self._session_cm: Any = None
        self._session: Any = None

    async def __aenter__(self):
        from kestrel_feature_entities import TenantContext

        self._tenant_cm = TenantContext.use(self._tenant_id)
        self._tenant_cm.__enter__()
        # If acquiring the session fails (pool exhaustion, connection
        # refused, etc.), Python will NOT invoke our __aexit__ because
        # the ``async with`` never bound a value. Unwind the tenant
        # context manually so subsequent tenant-filtered queries in
        # this task don't inherit a stale ``TenantContext``. (Caught by
        # codex review on the sovereign vector-lift PR.)
        try:
            self._session_cm = self._sf.read_session()
            self._session = await self._session_cm.__aenter__()
        except BaseException:
            self._tenant_cm.__exit__(None, None, None)
            self._tenant_cm = None
            raise
        return self._session

    async def __aexit__(self, exc_type, exc, tb):
        try:
            return await self._session_cm.__aexit__(exc_type, exc, tb)
        finally:
            if self._tenant_cm is not None:
                self._tenant_cm.__exit__(exc_type, exc, tb)
