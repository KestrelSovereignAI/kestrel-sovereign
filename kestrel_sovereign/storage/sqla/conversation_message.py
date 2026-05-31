"""SQLAlchemy mapping of the ``conversation_history`` table.

The table is created by the raw-SQL ``CORE_SCHEMA`` path in
``async_database.py``; this module adds an ORM view + spec so the
generic vector backends in
``kestrel_sovereign.storage.vector`` can perform kNN over per-message
embeddings. Consumed by :class:`MemoryRetriever` to swap the legacy
keyword-overlap semantic score for true cosine similarity.

Greenfield embedding column — unlike ``saved_items`` and
``document_chunks``, ``conversation_history`` had no pre-existing
``embedding`` BYTEA column. The Phase-2 migration adds
``embedding_vec`` directly at the configured dimension; there's
nothing to backfill from and no dim to sniff. The
``KESTREL_EMBEDDING_DIM`` env var (or
:data:`CONVERSATION_MESSAGE_EMBEDDING_DIM` default 768) picks the
column width.

Retrieval semantics (consumed by :class:`MemoryRetriever`):

- ``agent_id`` is a required filter — every retrieval is scoped to
  one agent's history.
- ``role`` is an optional filter (typically pinned to
  ``"assistant"`` to exclude user turns from semantic recall — user
  turns are questions, not knowledge).
- ``deleted_at`` is an optional filter. Pass ``None`` to enforce
  ``deleted_at IS NULL`` (SQLAlchemy translates ``column == None``
  to ``IS NULL``); omit it to include soft-deleted rows.

No ``tenant_id_filter_key``: sovereign-core's per-agent
``conversation_history`` is single-tenant by ``agent_id``;
multi-tenant scoping happens above this layer in frinz.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import Integer, Text
from sqlalchemy.orm import Mapped, mapped_column

from ..vector import VectorTableSpec
from .base import SovereignBase
from .types import PortableVector


def resolve_embedding_dim(env: Optional[dict] = None) -> int:
    """Pick the embedding dimension for fresh ``conversation_history`` DBs.

    Unlike ``saved_items`` / ``document_chunks`` the table has no
    legacy embedding column to sniff from, so the migration MUST pick
    a dim up front. ``KESTREL_EMBEDDING_DIM`` lets ops override the
    default before first write (e.g. to switch to ``mxbai-embed-large``
    at 1024 or OpenAI ada-002 at 1536); the fallback matches the
    default Ollama ``nomic-embed-text`` model
    (:class:`~kestrel_sovereign.llm.embedding_service.EmbeddingService.DEFAULT_MODEL`).

    Bad values (non-integer, non-positive) fall back to 768 with a
    warning. Mismatch between the column width and a future writer's
    embedding length will surface as a ``VectorSearchError`` at
    search time — not silently corrupt data.

    Args:
        env: Optional mapping to read from instead of ``os.environ``.
            Lets unit tests exercise every branch without importlib
            tricks (importing this module re-registers
            ``ConversationMessage`` against the declarative base,
            which crashes with ``Table is already defined``).
    """
    source = env if env is not None else os.environ
    raw = source.get("KESTREL_EMBEDDING_DIM")
    if not raw:
        return 768
    try:
        value = int(raw)
    except (TypeError, ValueError):
        import logging
        logging.getLogger(__name__).warning(
            "KESTREL_EMBEDDING_DIM=%r is not an integer; falling back to 768.",
            raw,
        )
        return 768
    if value <= 0:
        import logging
        logging.getLogger(__name__).warning(
            "KESTREL_EMBEDDING_DIM=%d is not positive; falling back to 768.",
            value,
        )
        return 768
    return value


CONVERSATION_MESSAGE_EMBEDDING_DIM = resolve_embedding_dim()


class ConversationMessage(SovereignBase):
    """ORM view of ``conversation_history``.

    Frozen against the schema in ``CORE_SCHEMA`` (``async_database.py``).
    Adding columns here without a matching migration produces silent
    drift between the ORM and the underlying storage.

    Encryption note: ``content`` and ``rendered_content`` are
    Fernet-encrypted ciphertext at rest. This ORM view returns the
    raw ciphertext — callers that need plaintext go through
    :class:`AsyncConversationStore`, which holds the per-agent
    decryption key. The vector backends never touch ``content``;
    they only kNN against the pre-computed ``embedding_vec`` and
    return ``(id, similarity)`` tuples, so encryption is orthogonal
    to vector search.
    """

    __tablename__ = "conversation_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    agent_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    role: Mapped[str] = mapped_column(Text, nullable=False)

    # Ciphertext at rest. The ORM exposes the column so downstream
    # SQLAlchemy tooling (Alembic, introspection) sees the full
    # schema; vector search never reads this column.
    content: Mapped[str] = mapped_column(Text, nullable=False)
    rendered_content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # JSON-encoded TEXT, NOT a JSONB column on PG. Match raw-SQL
    # behaviour — :class:`AsyncConversationStore` writes ``json.dumps``
    # blobs.
    item_metadata: Mapped[Optional[str]] = mapped_column(
        "metadata", Text, nullable=True
    )

    created_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    deleted_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)

    # Parallel-column embedding. Unlike saved_items / document_chunks
    # there's no legacy BYTEA column on conversation_history — the
    # Phase-2 migration creates ``embedding_vec`` fresh at the
    # configured dim. The ORM column key is still "embedding" so call
    # sites stay symmetric across entities (see SavedItem,
    # DocumentChunk).
    embedding: Mapped[Optional[Any]] = mapped_column(
        "embedding_vec",
        PortableVector(CONVERSATION_MESSAGE_EMBEDDING_DIM),
        nullable=True,
    )


def build_conversation_message_spec(dimension: int) -> VectorTableSpec:
    """Build a ``VectorTableSpec`` for ``conversation_history``.

    Filter contract for :class:`MemoryRetriever`:

    - ``agent_id`` (required + WHERE) — every retrieval is
      single-agent. A missing ``agent_id`` raises a
      :class:`~kestrel_sovereign.storage.vector.exceptions.VectorSearchError`
      so a buggy caller can't accidentally pull another agent's
      history.
    - ``role`` (optional WHERE) — pin to ``"assistant"`` to skip
      user turns in semantic recall (they're questions, not
      knowledge — matches the existing retriever's filter).
    - ``deleted_at`` (optional WHERE) — pass ``None`` to enforce
      ``deleted_at IS NULL`` (SQLAlchemy renders ``column == None``
      as ``IS NULL``). Omit to include soft-deleted rows; the
      retriever will always pass ``None``.

    Args:
        dimension: Embedding dimension. The retriever should pass
            ``len(query_embedding)`` so a model switch (768 →
            1024) doesn't require a code change here — the kNN
            backend will raise a clear ``VectorSearchError`` if the
            column width and the runtime embedding disagree.

    Raises:
        ValueError: If ``dimension`` is not positive.
    """
    if dimension <= 0:
        raise ValueError(
            f"build_conversation_message_spec: dimension must be positive, got {dimension}"
        )
    return VectorTableSpec(
        entity=ConversationMessage,
        id_column=ConversationMessage.id,
        embedding_column=ConversationMessage.embedding,
        dimension=dimension,
        required_filter_keys=("agent_id",),
        filter_columns={
            "agent_id": ConversationMessage.agent_id,
            "role": ConversationMessage.role,
            "deleted_at": ConversationMessage.deleted_at,
        },
    )
