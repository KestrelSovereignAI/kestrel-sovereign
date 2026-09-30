"""Where a raw-SQL reader takes a stored embedding from, and how to decode it.

Phase 2b of retiring the legacy ``embedding`` column on ``saved_items`` and
``document_chunks`` (#3409, parent #2684). The inventory of readers and
writers is ``docs/architecture/storage/EMBEDDING_COLUMN_RETIREMENT.md``.
Every raw ``AsyncDatabase`` reader of a stored vector on those tables
resolves its column through :func:`stored_embedding_column`.

``embedding_vec`` is the canonical representation. ``kestrel embeddings
reindex`` rewrites it and ``embedding_profile_id`` but never the legacy
column, so a reader of the legacy bytes scores a reindexed row with the
previous model's vector, and misses a row that only reindex embedded.

The legacy column is read only when the table has no ``embedding_vec``
column at all. On PostgreSQL the startup migration defers creating
``vector(N)`` until a legacy row shows the width, so a fresh database holds
its first vectors only in the legacy column until the next boot. Reindex
cannot have run against such a table (it reads and writes ``embedding_vec``),
so those legacy bytes are current. Once the column exists it is the only
source: a row whose ``embedding_vec`` is NULL has no stored vector, whatever
its legacy column holds. ``kestrel embeddings backfill`` repairs such rows.
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:
    from .async_database import AsyncDatabase

logger = logging.getLogger(__name__)

CANONICAL_COLUMN = "embedding_vec"
LEGACY_COLUMN = "embedding"

# Table names are interpolated into SQL, so only these literals are accepted.
_TABLES = frozenset({"saved_items", "document_chunks"})


@dataclass(frozen=True)
class StoredEmbeddingColumn:
    """The column a table's stored embeddings are read from."""

    # ``embedding_vec``, or the legacy ``embedding`` when the table has no
    # ``embedding_vec`` column. Use in predicates such as ``IS NOT NULL``.
    name: str
    # Expression for a SELECT list. Its values decode with
    # :func:`decode_stored_embedding`.
    select: str


async def stored_embedding_column(
    db: "AsyncDatabase", table: str,
) -> StoredEmbeddingColumn:
    """Resolve where *table*'s stored embeddings are read from.

    Asks the catalog rather than letting a SELECT fail to find out: a failed
    statement aborts the enclosing transaction on PostgreSQL.
    """
    if table not in _TABLES:
        raise ValueError(f"no stored embedding column on table {table!r}")
    if not await db.column_exists(table, CANONICAL_COLUMN):
        logger.debug(
            "%s has no %s column yet; reading the legacy %s column.",
            table, CANONICAL_COLUMN, LEGACY_COLUMN,
        )
        return StoredEmbeddingColumn(name=LEGACY_COLUMN, select=LEGACY_COLUMN)
    # pgvector's text output (``[v1,v2,...]``) is the explicit shape: the raw
    # asyncpg pool registers no pgvector codec. SQLite stores float32 bytes.
    select = (
        f"{CANONICAL_COLUMN}::text"
        if getattr(db, "backend_type", None) == "postgres"
        else CANONICAL_COLUMN
    )
    return StoredEmbeddingColumn(name=CANONICAL_COLUMN, select=select)


def decode_stored_embedding(value: Any) -> Optional[List[float]]:
    """Decode a value selected through :func:`stored_embedding_column`.

    Float32 little-endian bytes come from SQLite and from the legacy
    PostgreSQL ``BYTEA``; pgvector text comes from ``embedding_vec::text``.
    Returns ``None`` for NULL, an empty vector, or bytes whose length is not
    a multiple of 4 (unpacking those with ``// 4`` would score noise, #1653).
    """
    if value is None:
        return None
    if isinstance(value, str):
        body = value.strip()
        if not (body.startswith("[") and body.endswith("]")):
            raise ValueError(f"not pgvector text: {body[:40]!r}")
        inner = body[1:-1].strip()
        return [float(part) for part in inner.split(",")] if inner else None
    if isinstance(value, (bytes, bytearray, memoryview)):
        data = bytes(value)
        if len(data) % 4 != 0:
            logger.warning(
                "Skipping embedding: %d bytes not a multiple of 4.", len(data),
            )
            return None
        count = len(data) // 4
        return list(struct.unpack(f"<{count}f", data)) if count else None
    raise TypeError(f"unexpected stored embedding type {type(value).__name__}")
