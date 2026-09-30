"""Where a raw-SQL reader takes a stored embedding from, and how to decode it.

Phase 2b of retiring the legacy ``embedding`` column on ``saved_items`` and
``document_chunks`` (#3409, parent #2684). The inventory of readers and
writers is ``docs/architecture/storage/EMBEDDING_COLUMN_RETIREMENT.md``.
Every raw ``AsyncDatabase`` reader of a stored vector on those tables
resolves its column through :func:`stored_embedding_column`.

Since phase 3 (#3411) writers store vectors in ``embedding_vec`` only, and
create that column with :func:`ensure_embedding_vec_column` when it is absent.

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
its legacy column holds. The next boot copies such rows' legacy vectors
(#3414), and ``kestrel embeddings backfill`` does so on demand.
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


async def ensure_embedding_vec_column(
    db: "AsyncDatabase", table: str, dimension: int,
) -> bool:
    """Create *table*'s ``embedding_vec`` column if it is absent.

    The startup migration sizes PostgreSQL's ``vector(N)`` from a legacy
    row. Writers no longer fill the legacy column (#3411), so on a fresh
    PostgreSQL database no such row ever appears, and the first embedded
    write sizes the column from its own vector instead. SQLite stores bytes
    and gets a ``BLOB``; its startup migration normally created it already.

    It copies no legacy vector. Rows an older release left with one (the
    column is absent because a startup migration rolled back) are copied
    by the next boot (#3414), not by this write.

    Returns whether the column exists afterwards. A failure is logged, not
    raised: the caller's write then fails the same way and is non-fatal.
    The DDL runs inside ``transaction()``, a savepoint when the caller holds
    one on PostgreSQL, so a failure cannot poison the caller's transaction.
    """
    if table not in _TABLES:
        raise ValueError(f"no stored embedding column on table {table!r}")
    # Interpolated into the DDL below.
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
        raise ValueError(f"embedding dimension must be a positive int, got {dimension!r}")
    if await db.column_exists(table, CANONICAL_COLUMN):
        return True

    is_postgres = getattr(db, "backend_type", None) == "postgres"
    try:
        async with db.transaction():
            if is_postgres:
                # The extension must exist before the ALTER names ``vector``.
                await db.execute("CREATE EXTENSION IF NOT EXISTS vector", ())
                # IF NOT EXISTS: a concurrent first write may have added it.
                await db.execute(
                    f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS "
                    f"{CANONICAL_COLUMN} vector({dimension})",
                    (),
                )
            else:
                await db.execute(
                    f"ALTER TABLE {table} ADD COLUMN {CANONICAL_COLUMN} BLOB", ()
                )
    except Exception as exc:
        # Another connection may have added it first (SQLite has no
        # ``ADD COLUMN IF NOT EXISTS``).
        if await db.column_exists(table, CANONICAL_COLUMN):
            return True
        logger.warning(
            "Could not create %s.%s for a %d-dimension embedding: %s. The "
            "vector is not stored.",
            table, CANONICAL_COLUMN, dimension, exc,
        )
        return False
    logger.info(
        "Created %s.%s (%s) for the first embedded write.",
        table, CANONICAL_COLUMN,
        f"vector({dimension})" if is_postgres else "BLOB",
    )

    if is_postgres:
        # Same index the startup migration builds. Separate, so a width
        # pgvector cannot index (HNSW stops at 2000) still keeps the column.
        try:
            async with db.transaction():
                await db.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_{table}_{CANONICAL_COLUMN}_hnsw "
                    f"ON {table} USING hnsw ({CANONICAL_COLUMN} vector_cosine_ops)",
                    (),
                )
        except Exception as exc:
            logger.warning(
                "Could not create the HNSW index on %s.%s: %s. kNN search "
                "still works, without the index.",
                table, CANONICAL_COLUMN, exc,
            )
    return True


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
