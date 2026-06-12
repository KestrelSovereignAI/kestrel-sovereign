"""SQLAlchemy mapping of the ``memory_episodes`` table.

The table is created and managed by the raw-SQL ``AsyncDatabase`` /
``CORE_SCHEMA`` path (see ``async_database.py``). This module adds an
ORM view on top so the consolidator's semantic episode recall can hand
the table to the generic vector backends in
``kestrel_sovereign.storage.vector`` — the same path ``saved_items``
uses (see ``saved_item.py``). No new embedding/kNN machinery: episodes
reuse the shared backend, embedding service, and profile-id filtering.

The ``embedding`` column uses :class:`PortableVector` — ``vector(N)``
on Postgres (via pgvector) and ``LargeBinary`` (BLOB) on SQLite. SQLite
(the default backend) drives the ``PurePythonBackend`` cosine path; PG
deployments that haven't run a pgvector migration degrade gracefully to
the keyword fallback in ``MemoryConsolidator.search_episodes`` (#1674 P2).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column

from ..vector import VectorTableSpec
from .base import SovereignBase
from .types import PortableVector


# Default embedding dimension — matches the Ollama ``nomic-embed-text``
# default used elsewhere. The spec is built per-query from the runtime
# embedding length, so this default only matters for the ORM column
# declaration (see ``build_episode_spec``).
EPISODE_EMBEDDING_DIM = 768


class MemoryEpisodeRow(SovereignBase):
    """ORM view of ``memory_episodes``.

    Frozen against the schema in ``CORE_SCHEMA`` (``async_database.py``).
    Changing this entity requires a corresponding schema migration — do
    not add columns here without one. Only the columns the vector path
    needs are mapped; the raw ``INSERT`` / ``from_row`` callers in
    ``memory_consolidator`` own the full row shape.
    """

    __tablename__ = "memory_episodes"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    agent_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    title: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Embedding of the episode's title+summary, written at consolidation
    # time. NULL on episodes created before P2 or when no embedding
    # provider was available — profile-filtered kNN skips NULL rows.
    embedding: Mapped[Optional[Any]] = mapped_column(
        "embedding_vec", PortableVector(EPISODE_EMBEDDING_DIM), nullable=True
    )
    embedding_profile_id: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )

    created_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)


def build_episode_spec(dimension: int) -> VectorTableSpec:
    """Build a ``VectorTableSpec`` for ``memory_episodes`` at a given dim.

    Mirrors ``build_saved_item_spec``: episode embeddings are not
    fixed-dimension (the dim depends on the active embedding model), so
    the spec is built per-query from ``len(query_embedding)``.

    - ``agent_id`` is required and applied as a WHERE clause: every
      recall is scoped to one agent's episodes.
    - ``embedding_profile_id`` is an optional WHERE filter so cross-model
      rows can't sneak into cosine (see ``saved_item``/#1477).
    """
    if dimension <= 0:
        raise ValueError(
            f"build_episode_spec: dimension must be positive, got {dimension}"
        )
    return VectorTableSpec(
        entity=MemoryEpisodeRow,
        id_column=MemoryEpisodeRow.id,
        embedding_column=MemoryEpisodeRow.embedding,
        dimension=dimension,
        required_filter_keys=("agent_id",),
        filter_columns={
            "agent_id": MemoryEpisodeRow.agent_id,
            "embedding_profile_id": MemoryEpisodeRow.embedding_profile_id,
        },
    )
