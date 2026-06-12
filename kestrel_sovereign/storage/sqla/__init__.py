"""SQLAlchemy infrastructure for sovereign-core storage.

This package is sovereign's bridge between the older raw-SQL
``AsyncDatabase`` storage layer and the SQLAlchemy-based primitives in
``kestrel_sovereign.storage.vector`` (and, in the future, in feature
packages that want to read sovereign-owned tables).

Public surface:

- :class:`SovereignBase` — declarative base for sovereign-core entities.
  Distinct from the per-feature ``EntityBase`` in
  ``kestrel-feature-entities`` so sovereign-core doesn't depend on a
  feature pkg for its own ORM models.
- :class:`SavedItem` — ORM mapping of the existing ``saved_items``
  table; lets the vector backends query the table via SQLAlchemy
  without changing the underlying schema.
- :func:`make_session_factory` — builds an async SQLAlchemy session
  factory that connects to the same database as a given
  ``AsyncDatabase``. Used by ``SavedItemsStore.search()`` to obtain a
  session for the vector backend.

``SavedItem``, ``DocumentChunk``, and ``ConversationMessage`` live here
today; ``AgentMetadata`` and other sovereign-owned tables will follow
as their read paths grow vector / structured-query needs.
"""

from .base import SovereignBase
from .conversation_message import (
    CONVERSATION_MESSAGE_EMBEDDING_DIM,
    ConversationMessage,
    build_conversation_message_spec,
)
from .document_chunk import DocumentChunk, build_document_chunk_spec
from .episode import (
    EPISODE_EMBEDDING_DIM,
    MemoryEpisodeRow,
    build_episode_spec,
)
from .saved_item import SavedItem, build_saved_item_spec
from .session import SovereignSqlaSessionFactory, make_session_factory

__all__ = [
    "SovereignBase",
    "SavedItem",
    "build_saved_item_spec",
    "DocumentChunk",
    "build_document_chunk_spec",
    "MemoryEpisodeRow",
    "build_episode_spec",
    "EPISODE_EMBEDDING_DIM",
    "ConversationMessage",
    "build_conversation_message_spec",
    "CONVERSATION_MESSAGE_EMBEDDING_DIM",
    "SovereignSqlaSessionFactory",
    "make_session_factory",
]
