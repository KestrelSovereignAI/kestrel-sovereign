"""SQLAlchemy mapping of the ``saved_items`` table.

The table is created and managed by the raw-SQL ``AsyncDatabase`` /
``CORE_SCHEMA`` path (see ``async_database.py``). This module adds an
ORM view on top so :func:`SavedItemsStore.search` can hand the table
to the generic vector backends in ``kestrel_sovereign.storage.vector``.

The ``embedding`` column uses :class:`PortableVector` — ``vector(N)``
on Postgres (via pgvector) and ``LargeBinary`` (BLOB) on SQLite. The
Phase-2 PG migration in ``saved_items_pgvector_migration.py`` swaps
existing BYTEA data to ``vector(N)`` on upgrade; SQLite is unchanged.
See kestrel-sovereign #1447 / #1452.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column

from ..vector import VectorTableSpec
from .base import SovereignBase
from .types import PortableVector


# Embedding dimension for ``saved_items``. Matches the default Ollama
# ``nomic-embed-text`` model. The Phase-2 PG migration sniffs existing
# rows to pick the actual dim on upgrade — this default is only used
# when the migration finds no embedded rows (fresh DB).
SAVED_ITEM_EMBEDDING_DIM = 768


class SavedItem(SovereignBase):
    """ORM view of ``saved_items``.

    Frozen against the schema in ``CORE_SCHEMA`` (``async_database.py``).
    Changing this entity requires a corresponding schema migration —
    do not add columns here without one.
    """

    __tablename__ = "saved_items"

    # Primary key is TEXT in the legacy schema — typically a uuid4 string
    # produced by ``SavedItemsStore.save_item``. We keep that shape rather
    # than retrofitting a UUID column type.
    id: Mapped[str] = mapped_column(String, primary_key=True)

    # Scoping: every row belongs to a single agent. ``SavedItemsStore``
    # filters by ``agent_id`` everywhere; the vector backends will too.
    agent_id: Mapped[str] = mapped_column(String, nullable=False, index=True)

    # ``stash`` / ``file`` / ``excerpt`` / ``structured`` etc. Used as
    # an optional filter in search.
    item_type: Mapped[str] = mapped_column(String, nullable=False)

    name: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    ipfs_cid: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    # Embedding column — the ORM uses the parallel ``embedding_vec``
    # SQL column added by the Phase-2 migration (NOT the legacy
    # ``embedding`` BYTEA / BLOB column used by raw ``AsyncDatabase``
    # IO). The two are kept in sync by ``SavedItemsStore.save_item``'s
    # dual-write. This split lets the SQLA + pgvector path land
    # without disturbing the raw ``INSERT`` / ``from_row`` callers
    # that bind/unpack float32 bytes. (Caught by codex review on the
    # Phase 2 PR — an in-place column-type swap would have broken
    # them.)
    embedding: Mapped[Optional[Any]] = mapped_column(
        "embedding_vec", PortableVector(SAVED_ITEM_EMBEDDING_DIM), nullable=True
    )

    # Provenance + categorization. JSON-encoded blobs stored as TEXT —
    # matches the existing ``json.dumps(...)`` writes in
    # ``SavedItemsStore``. Decoding is the caller's responsibility.
    source_type: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    source_ref: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    schema_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    tags: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    item_metadata: Mapped[Optional[str]] = mapped_column(
        "metadata", Text, nullable=True
    )

    created_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    updated_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)


def build_saved_item_spec(dimension: int) -> VectorTableSpec:
    """Build a ``VectorTableSpec`` for ``saved_items`` at a given dim.

    Saved-item embeddings are NOT fixed-dimension: the default Ollama
    model (``nomic-embed-text``) returns 768-dim vectors,
    ``mxbai-embed-large`` returns 1024-dim, OpenAI's ``ada-002`` is
    1536-dim, etc. The store doesn't normalize, so the right
    ``VectorTableSpec.dimension`` is whatever shape the runtime query
    embedding has. ``SavedItemsStore.search`` builds the spec
    per-query from ``len(query_embedding)``.

    - ``agent_id`` is required and applied as a WHERE clause: every
      search is scoped to one agent's stash.
    - ``item_type`` is an optional WHERE filter (search a single type
      of saved item).
    - No ``tenant_id_filter_key``: ``saved_items`` is single-tenant per
      agent, with no separate tenant scoping above that.

    (Caught by codex review: a fixed ``dimension=1536`` made every
    default-model search fall through to the legacy in-Python path on
    a length-check error.)
    """
    if dimension <= 0:
        raise ValueError(
            f"build_saved_item_spec: dimension must be positive, got {dimension}"
        )
    return VectorTableSpec(
        entity=SavedItem,
        id_column=SavedItem.id,
        embedding_column=SavedItem.embedding,
        dimension=dimension,
        required_filter_keys=("agent_id",),
        filter_columns={
            "agent_id": SavedItem.agent_id,
            "item_type": SavedItem.item_type,
        },
    )
