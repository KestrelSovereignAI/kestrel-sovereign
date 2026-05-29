"""One-time data migrations for sovereign-core SQLAlchemy entities.

Sovereign-core's schema is managed by raw SQL ``CREATE TABLE IF NOT
EXISTS`` statements in ``async_database.py`` (see ``CORE_SCHEMA``),
not Alembic. Migrations live here as idempotent async functions that
``AsyncDatabase`` runs on startup AFTER the schema-create step.

Phase 2 of #1447: add a parallel ``embedding_vec`` column to
``saved_items``:

- On Postgres: ``vector(N)``, indexed with HNSW for fast cosine kNN.
  The existing ``embedding`` BYTEA column stays — legacy raw IO paths
  in :class:`SavedItemsStore` continue to write/read it unchanged, and
  the new ``save_item`` dual-write keeps the two in sync.
- On SQLite: also ``BLOB``. Same dual-write keeps both columns in
  sync there too. PurePythonBackend can read either; the ORM uses
  ``embedding_vec`` so the code path is the same as PG.

The parallel column lets us flip the vector-backend factory to
``PgVectorBackend`` on PG without rewriting the raw INSERT / SELECT
paths that bind ``embedding`` as float32 bytes. (Caught by codex
review — an in-place ``ALTER COLUMN TYPE`` would have broken
``save_item()`` and ``SavedItem.from_row()`` on PG.)
"""

from __future__ import annotations

import logging
import struct
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..async_database import AsyncDatabase

logger = logging.getLogger(__name__)


# Default embedding dimension if no embedded rows exist yet (fresh DB).
# Matches the default Ollama ``nomic-embed-text`` model. Kept in sync
# with ``saved_item.SAVED_ITEM_EMBEDDING_DIM``.
_DEFAULT_DIM = 768


async def migrate_saved_items_add_embedding_vec(db: "AsyncDatabase") -> None:
    """Add a parallel ``embedding_vec`` column to ``saved_items`` and
    backfill from the existing ``embedding`` BYTEA / BLOB.

    Idempotent: skips cleanly when the column already exists.

    On Postgres:
        1. ``CREATE EXTENSION IF NOT EXISTS vector``.
        2. Sniff the dimension from existing rows (or default to 768).
        3. ``ALTER TABLE ... ADD COLUMN embedding_vec vector(<dim>)`` if
           the column doesn't already exist.
        4. Backfill: for each row with non-null BYTEA ``embedding``,
           unpack to floats, format as pgvector text, set
           ``embedding_vec``.
        5. HNSW index on ``embedding_vec vector_cosine_ops``.

    On SQLite:
        1. ``ALTER TABLE ... ADD COLUMN embedding_vec BLOB`` (no-op via
           ``pragma_table_info`` check if already present).
        2. Backfill: ``UPDATE`` copies bytes from ``embedding`` to
           ``embedding_vec``. (Yes, same data twice — keeps the ORM
           code paths identical across dialects.)

    Runs inside ``db.transaction()`` so any partial failure rolls back
    cleanly. The advertised idempotency depends on this — without a
    transaction, a crash partway through could leave the schema in a
    half-migrated state that later runs misdetect.
    """
    backend_type = getattr(db, "backend_type", None)

    if backend_type == "postgres":
        await _migrate_pg(db)
    elif backend_type == "sqlite":
        await _migrate_sqlite(db)
    # Other dialects: no-op. The ORM column maps to LargeBinary on
    # non-PG dialects so any reasonable backend should work, but we
    # don't try to introspect arbitrary engines.


async def _migrate_pg(db: "AsyncDatabase") -> None:
    """Postgres path — adds vector column + backfills + HNSW index."""

    # Check if embedding_vec already exists; if so, the migration ran
    # in a prior boot.
    rows = await db.fetchall(
        """SELECT 1 FROM information_schema.columns
           WHERE table_name = 'saved_items' AND column_name = 'embedding_vec'""",
        (),
    )
    if rows:
        logger.debug(
            "saved_items.embedding_vec already present — skipping Phase-2 PG migration."
        )
        return

    # Confirm the source table + column exist; otherwise let
    # ``_init_schema`` handle it on the next boot.
    src = await db.fetchall(
        """SELECT udt_name FROM information_schema.columns
           WHERE table_name = 'saved_items' AND column_name = 'embedding'""",
        (),
    )
    if not src:
        logger.debug(
            "saved_items table not yet present — skipping Phase-2 PG migration."
        )
        return

    # Sniff the dimension from existing rows. We INTENTIONALLY don't
    # guess at a default for fresh DBs: an embedding service may be
    # configured for any of nomic-embed-text (768), mxbai-embed-large
    # (1024), or OpenAI ada-002 (1536), and creating ``vector(768)``
    # against a 1536-dim writer would make every subsequent
    # ``_write_embedding_vec`` fail with a dim mismatch. Skip column
    # creation here; the next boot will re-run, sniff the actual dim
    # from saved rows, and create the column at the right size.
    # (Caught by codex review on the Phase 2 PR.)
    sample = await db.fetchall(
        """SELECT octet_length(embedding) FROM saved_items
           WHERE embedding IS NOT NULL LIMIT 1""",
        (),
    )
    if not sample:
        logger.info(
            "saved_items has no embedded rows yet — deferring embedding_vec "
            "column creation until the next boot (we don't guess a dim)."
        )
        return

    byte_len = sample[0][0]
    if byte_len % 4 != 0 or byte_len <= 0:
        logger.error(
            "saved_items.embedding has non-float32 byte length %d. "
            "Refusing to migrate.", byte_len,
        )
        return
    dim = byte_len // 4
    logger.info(
        "Sniffed embedding dimension %d (%d bytes) from existing rows.",
        dim, byte_len,
    )

    expected_bytes = dim * 4

    async with db.transaction():
        # pgvector extension must exist BEFORE ``ALTER TABLE`` references
        # ``vector(N)`` — on a fresh DB without the extension, the ALTER
        # fails with ``type "vector" does not exist``. ``IF NOT EXISTS``
        # so repeat-installs are silent. (Caught by codex review on the
        # Phase 2 PR — the original order had the ALTER first, which
        # broke the migration on fresh Postgres databases.)
        await db.execute("CREATE EXTENSION IF NOT EXISTS vector", ())

        # Add the parallel vector column.
        await db.execute(
            f"ALTER TABLE saved_items ADD COLUMN embedding_vec vector({dim})",
            (),
        )

        # Backfill row-by-row. For typical saved_items volumes this
        # fits in memory comfortably; swap to a server-side cursor if
        # scale ever changes.
        all_rows = await db.fetchall(
            "SELECT id, embedding FROM saved_items WHERE embedding IS NOT NULL",
            (),
        )
        converted = 0
        skipped = 0
        for row_id, embedding_bytes in all_rows:
            if embedding_bytes is None:
                continue
            if len(embedding_bytes) != expected_bytes:
                logger.warning(
                    "Skipping row %s: %d bytes != expected %d (different "
                    "embedding-model dim — will not appear in vector search "
                    "until re-embedded).",
                    row_id, len(embedding_bytes), expected_bytes,
                )
                skipped += 1
                continue
            floats = struct.unpack(f"<{dim}f", bytes(embedding_bytes))
            vec_text = "[" + ",".join(repr(float(v)) for v in floats) + "]"
            await db.execute(
                "UPDATE saved_items SET embedding_vec = $1::vector WHERE id = $2",
                (vec_text, row_id),
            )
            converted += 1

        # HNSW cosine index for ``<=>`` queries.
        await db.execute(
            """CREATE INDEX IF NOT EXISTS idx_saved_items_embedding_vec_hnsw
               ON saved_items USING hnsw (embedding_vec vector_cosine_ops)""",
            (),
        )

    logger.info(
        "saved_items Phase-2 PG migration complete: added embedding_vec "
        "vector(%d), backfilled %d rows, skipped %d mismatched, HNSW index.",
        dim, converted, skipped,
    )


async def _migrate_sqlite(db: "AsyncDatabase") -> None:
    """SQLite path — adds embedding_vec BLOB + copies bytes."""

    # SQLite's ``pragma_table_info`` lets us check column presence
    # idempotently. (``ALTER TABLE ADD COLUMN IF NOT EXISTS`` isn't a
    # thing on SQLite.)
    rows = await db.fetchall(
        "SELECT name FROM pragma_table_info('saved_items') WHERE name = 'embedding_vec'",
        (),
    )
    if rows:
        logger.debug(
            "saved_items.embedding_vec already present — skipping Phase-2 SQLite migration."
        )
        return

    table_exists = await db.fetchall(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='saved_items'",
        (),
    )
    if not table_exists:
        logger.debug(
            "saved_items table not yet present — skipping Phase-2 SQLite migration."
        )
        return

    async with db.transaction():
        await db.execute(
            "ALTER TABLE saved_items ADD COLUMN embedding_vec BLOB", ()
        )
        # Copy existing bytes so the ORM column has data on row 1.
        await db.execute(
            "UPDATE saved_items SET embedding_vec = embedding "
            "WHERE embedding IS NOT NULL",
            (),
        )

    logger.info(
        "saved_items Phase-2 SQLite migration complete: added embedding_vec "
        "BLOB, copied existing embeddings."
    )
