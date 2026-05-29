"""PurePythonBackend — SQLite-friendly cosine kNN using numpy.

Fetches every matching row's embedding into Python and computes cosine
similarity in a loop. O(n) per query — fine for small datasets, slow
beyond ~thousands of rows per filter scope. The future ``SqliteVecBackend``
(FEAT-8 in the deferred queue) will replace this on SQLite once trigger
conditions fire.

Generic over a ``VectorTableSpec`` so the same code serves any
SQLAlchemy-mapped table that holds an embedding column. Behavior is
designed to match ``PgVectorBackend`` exactly so backend swaps don't
shift result ordering on the same input.
"""

from __future__ import annotations

import logging
import struct
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import and_, select

from .exceptions import VectorSearchError
from .pg import _TenantScopedSession  # share the multi-tenant wrapper
from .spec import VectorTableSpec

logger = logging.getLogger(__name__)


class PurePythonBackend:
    """Pure-Python cosine similarity. SQLite-friendly, slow at scale."""

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

        Arguments + return shape match :class:`PgVectorBackend.knn`.
        """
        try:
            import numpy as np
        except ImportError as e:
            raise VectorSearchError(
                "numpy is required for PurePythonBackend; install "
                "kestrel-sovereign or add numpy directly."
            ) from e

        # Normalize filter AFTER validation so unfiltered specs (no
        # required_filter_columns) can be called with ``filter=None`` —
        # validation passes, then we work against an empty dict.
        self._require_filters(filter)
        filter = filter or {}

        query_vec = self._unpack(query_embedding)
        query_np = np.array(query_vec, dtype=np.float32)
        query_norm = float(np.linalg.norm(query_np))

        if query_norm == 0.0:
            # Match PgVectorBackend's zero-norm short-circuit so the two
            # backends agree at the edge.
            logger.warning(
                "PurePythonBackend.knn: query embedding has zero norm — returning []"
            )
            return []

        spec = self._spec
        stmt = select(spec.id_column, spec.embedding_column).where(
            and_(
                spec.embedding_column.isnot(None),
                *(column == filter[key] for key, column in spec.required_filter_columns.items()),
                *(column == filter[key] for key, column in spec.optional_filter_columns.items() if key in filter),
            )
        )

        async with self._open_read_session(filter) as session:
            result = await session.execute(stmt)
            rows = result.all()

        if not rows:
            return []

        scored: List[Tuple[str, float]] = []
        for row_id, embedding_data in rows:
            # ``if not embedding_data`` would raise on numpy arrays
            # ("truth value of an array with more than one element is
            # ambiguous"), so check None explicitly. Empty bytes /
            # empty list are filtered downstream when the norm comes
            # out as 0 — they're harmless either way. (Caught by codex
            # review on the sovereign vector-lift PR.)
            if embedding_data is None:
                continue
            try:
                # PortableVector may yield bytes (SQLite manual unpack),
                # a list/tuple (some PG paths), or a numpy.ndarray
                # (pgvector's SQLAlchemy adapter). Accept all three —
                # numpy can wrap bytes itself, so we route them through
                # ``_unpack`` for explicit length validation.
                if isinstance(embedding_data, np.ndarray):
                    row_vec = embedding_data.astype(np.float32, copy=False)
                elif isinstance(embedding_data, (list, tuple)):
                    row_vec = embedding_data
                else:
                    row_vec = self._unpack(embedding_data)
                row_np = np.asarray(row_vec, dtype=np.float32)
                row_norm = float(np.linalg.norm(row_np))
                if row_norm == 0.0:
                    continue
                sim = float(np.dot(query_np, row_np) / (query_norm * row_norm))
                scored.append((str(row_id), sim))
            except Exception as e:
                logger.warning(
                    "PurePythonBackend.knn: failed to score row %s: %s", row_id, e
                )
                continue

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]

    # ----------------------------------------------------------------- helpers

    def _open_read_session(self, filter: Dict[str, Any]):
        """Return the read session, wrapped in TenantContext when spec opts in."""
        if self._spec.tenant_id_filter_key is None:
            return self._sf.read_session()
        return _TenantScopedSession(self._sf, filter[self._spec.tenant_id_filter_key])

    def _require_filters(self, filter: Optional[Dict[str, Any]]) -> None:
        required = set(self._spec.required_filter_columns)
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
        """Same shape as PgVectorBackend._unpack — keep behaviour identical."""
        expected = self._spec.dimension * 4
        if len(data) != expected:
            raise VectorSearchError(
                f"query_embedding has {len(data)} bytes; "
                f"expected {expected} (dim={self._spec.dimension} * 4)."
            )
        return list(struct.unpack(f"<{self._spec.dimension}f", data))
