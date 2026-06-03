"""SQLAlchemy mapping of the ``document_chunks`` table.

The table is created by the raw-SQL ``CORE_SCHEMA`` path in
``async_database.py``; this module just adds an ORM view + spec so
:class:`AsyncRAGStore._search_by_embedding` can hand the table to the
generic vector backends.

Parallel-column design: the legacy ``embedding`` BYTEA / BLOB column
stays as-is for raw ``AsyncDatabase`` IO (the ``chunk_document``
INSERT path). The ORM points at a separate ``embedding_vec`` column
that the Phase-2 migration adds + dual-write keeps in sync. Same
shape we landed for ``saved_items`` in #1454.

RAG-specific details:

- ``chunk_id`` is an autoincrement INTEGER (not a UUID string like
  ``saved_items.id``).
- There's no ``agent_id`` or ``tenant_id`` scoping on RAG — chunks
  are global per-database. The spec accordingly has no required
  filters; ``_search_by_embedding`` calls ``knn(filter=None)``.
- An optional ``file_hash`` filter is exposed in case a future caller
  wants to scope to a single source document.
"""

from __future__ import annotations

from typing import Any, Optional

from sqlalchemy import Integer, Text
from sqlalchemy.orm import Mapped, mapped_column

from ..vector import VectorTableSpec
from .base import SovereignBase
from .types import PortableVector


# Same default as saved_items — Ollama ``nomic-embed-text`` produces
# 768-dim vectors. The Phase-2 migration sniffs the actual dim from
# existing rows on upgrade; this default only matters when the spec is
# constructed without an explicit dim (which production code never
# does; ``AsyncRAGStore._search_via_vector_backend`` always passes
# ``dim=len(query_embedding)``).
DOCUMENT_CHUNK_EMBEDDING_DIM = 768


class DocumentChunk(SovereignBase):
    """ORM view of ``document_chunks``.

    Frozen against the schema in ``CORE_SCHEMA`` (``async_database.py``).
    Adding columns here without a matching migration produces silent
    drift between the ORM and the underlying storage.
    """

    __tablename__ = "document_chunks"

    chunk_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    file_hash: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # ORM-side embedding column. The legacy ``embedding`` BYTEA / BLOB
    # column used by raw IO is NOT in this mapping — see the module
    # docstring for the parallel-column rationale.
    embedding: Mapped[Optional[Any]] = mapped_column(
        "embedding_vec",
        PortableVector(DOCUMENT_CHUNK_EMBEDDING_DIM),
        nullable=True,
    )

    # #1477 — semantic-space identity for this row's embedding.
    # NULL on rows from before this column existed; profile-filtered
    # kNN skips NULL so cross-model recall can't silently mix
    # different coordinate spaces.
    embedding_profile_id: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )


def build_document_chunk_spec(dimension: int) -> VectorTableSpec:
    """Build a ``VectorTableSpec`` for ``document_chunks`` at a given dim.

    Document-chunk embeddings are NOT fixed-dim — they're whatever the
    runtime embedding service produces (nomic-embed-text=768,
    mxbai-embed-large=1024, etc.). ``AsyncRAGStore.search_chunks``
    builds the spec per-query from ``len(query_embedding)``.

    No required filter keys: RAG search is global. ``file_hash`` is
    available as an optional WHERE filter so callers can scope to a
    specific source document if they want.
    """
    if dimension <= 0:
        raise ValueError(
            f"build_document_chunk_spec: dimension must be positive, got {dimension}"
        )
    return VectorTableSpec(
        entity=DocumentChunk,
        id_column=DocumentChunk.chunk_id,
        embedding_column=DocumentChunk.embedding,
        dimension=dimension,
        required_filter_keys=(),
        filter_columns={
            "file_hash": DocumentChunk.file_hash,
            # #1477 — see ``conversation_message.build_conversation_message_spec``.
            "embedding_profile_id": DocumentChunk.embedding_profile_id,
        },
    )
