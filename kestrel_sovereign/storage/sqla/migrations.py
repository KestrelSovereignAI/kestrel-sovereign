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

import json
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


# =============================================================================
# document_chunks (kestrel-sovereign #1447 follow-up — same pattern as
# saved_items, applied to AsyncRAGStore.)
# =============================================================================


async def migrate_document_chunks_add_embedding_vec(db: "AsyncDatabase") -> None:
    """Add a parallel ``embedding_vec`` column to ``document_chunks``
    and backfill from the existing ``embedding`` BYTEA / BLOB.

    Same shape as :func:`migrate_saved_items_add_embedding_vec` —
    different table. Idempotent, transactional, defers column creation
    on fresh DBs (no dim guess).
    """
    backend_type = getattr(db, "backend_type", None)
    if backend_type == "postgres":
        await _migrate_pg_table(db, "document_chunks", "chunk_id")
    elif backend_type == "sqlite":
        await _migrate_sqlite_table(db, "document_chunks")
    # Other dialects: no-op.


async def _migrate_pg_table(db: "AsyncDatabase", table: str, id_col: str) -> None:
    """Generic Postgres migration that adds ``embedding_vec`` to ``table``.

    Factored out so the saved_items + document_chunks paths share the
    cast-bytea-to-vector logic. The two callers differ only by table
    name + id column name.
    """
    rows = await db.fetchall(
        f"""SELECT 1 FROM information_schema.columns
            WHERE table_name = '{table}' AND column_name = 'embedding_vec'""",
        (),
    )
    if rows:
        logger.debug(
            "%s.embedding_vec already present — skipping Phase-2 PG migration.",
            table,
        )
        return

    src = await db.fetchall(
        f"""SELECT udt_name FROM information_schema.columns
            WHERE table_name = '{table}' AND column_name = 'embedding'""",
        (),
    )
    if not src:
        logger.debug(
            "%s table not yet present — skipping Phase-2 PG migration.",
            table,
        )
        return

    sample = await db.fetchall(
        f"""SELECT octet_length(embedding) FROM {table}
            WHERE embedding IS NOT NULL LIMIT 1""",
        (),
    )
    if not sample:
        logger.info(
            "%s has no embedded rows yet — deferring embedding_vec column "
            "creation until the next boot.",
            table,
        )
        return

    byte_len = sample[0][0]
    if byte_len % 4 != 0 or byte_len <= 0:
        logger.error(
            "%s.embedding has non-float32 byte length %d. Refusing to migrate.",
            table, byte_len,
        )
        return
    dim = byte_len // 4
    logger.info(
        "Sniffed embedding dimension %d (%d bytes) from existing %s rows.",
        dim, byte_len, table,
    )
    expected_bytes = dim * 4

    async with db.transaction():
        # pgvector extension MUST exist before the ALTER (codex review
        # on #1454).
        await db.execute("CREATE EXTENSION IF NOT EXISTS vector", ())
        await db.execute(
            f"ALTER TABLE {table} ADD COLUMN embedding_vec vector({dim})", ()
        )

        all_rows = await db.fetchall(
            f"SELECT {id_col}, embedding FROM {table} WHERE embedding IS NOT NULL",
            (),
        )
        converted = 0
        skipped = 0
        for row_id, embedding_bytes in all_rows:
            if embedding_bytes is None:
                continue
            if len(embedding_bytes) != expected_bytes:
                logger.warning(
                    "Skipping %s row %s: %d bytes != expected %d.",
                    table, row_id, len(embedding_bytes), expected_bytes,
                )
                skipped += 1
                continue
            floats = struct.unpack(f"<{dim}f", bytes(embedding_bytes))
            vec_text = "[" + ",".join(repr(float(v)) for v in floats) + "]"
            await db.execute(
                f"UPDATE {table} SET embedding_vec = $1::vector WHERE {id_col} = $2",
                (vec_text, row_id),
            )
            converted += 1

        await db.execute(
            f"""CREATE INDEX IF NOT EXISTS idx_{table}_embedding_vec_hnsw
                ON {table} USING hnsw (embedding_vec vector_cosine_ops)""",
            (),
        )

    logger.info(
        "%s Phase-2 PG migration complete: added embedding_vec vector(%d), "
        "backfilled %d rows, skipped %d mismatched, HNSW index.",
        table, dim, converted, skipped,
    )


async def _migrate_sqlite_table(db: "AsyncDatabase", table: str) -> None:
    """Generic SQLite migration that adds ``embedding_vec`` BLOB to ``table``."""
    rows = await db.fetchall(
        f"SELECT name FROM pragma_table_info('{table}') WHERE name = 'embedding_vec'",
        (),
    )
    if rows:
        logger.debug(
            "%s.embedding_vec already present — skipping Phase-2 SQLite migration.",
            table,
        )
        return

    table_exists = await db.fetchall(
        f"SELECT 1 FROM sqlite_master WHERE type='table' AND name='{table}'",
        (),
    )
    if not table_exists:
        logger.debug(
            "%s table not yet present — skipping Phase-2 SQLite migration.",
            table,
        )
        return

    async with db.transaction():
        await db.execute(f"ALTER TABLE {table} ADD COLUMN embedding_vec BLOB", ())
        await db.execute(
            f"UPDATE {table} SET embedding_vec = embedding "
            f"WHERE embedding IS NOT NULL",
            (),
        )

    logger.info(
        "%s Phase-2 SQLite migration complete: added embedding_vec BLOB, "
        "copied existing embeddings.", table,
    )


# =============================================================================
# conversation_history (greenfield — no legacy embedding column to migrate
# from). Adds ``embedding_vec`` at the configured dim plus HNSW on PG.
# This prepares the storage side for MemoryRetriever cosine scoring;
# the current retriever still falls back to keyword/concept overlap
# until the embedding writer/read path is wired through.
# =============================================================================


async def migrate_conversation_history_add_embedding_vec(db: "AsyncDatabase") -> None:
    """Add an ``embedding_vec`` column to ``conversation_history``.

    Greenfield migration — there is NO pre-existing ``embedding``
    column on ``conversation_history`` to copy from, so the dim is
    picked from
    :data:`~kestrel_sovereign.storage.sqla.conversation_message.CONVERSATION_MESSAGE_EMBEDDING_DIM`
    (driven by the ``KESTREL_EMBEDDING_DIM`` env var; default 768 for
    Ollama ``nomic-embed-text``). Operators that switch models AFTER
    rows have been embedded need an explicit re-embedding script;
    this migration won't drop or resize the column. The read path is
    intentionally staged: the column exists before MemoryRetriever
    starts depending on it for cosine semantic scoring.

    Idempotent: skips cleanly if the column already exists. Wrapped
    in a transaction so a partial failure rolls back.
    """
    backend_type = getattr(db, "backend_type", None)
    if backend_type == "postgres":
        await _migrate_pg_greenfield(
            db,
            table="conversation_history",
        )
    elif backend_type == "sqlite":
        await _migrate_sqlite_greenfield(
            db,
            table="conversation_history",
        )
    # Other dialects: no-op.


async def _migrate_pg_greenfield(db: "AsyncDatabase", *, table: str) -> None:
    """Postgres greenfield migration — add ``embedding_vec vector(N)`` +
    HNSW on a table that has no existing embedding column.

    The dim is read lazily from
    ``conversation_message.CONVERSATION_MESSAGE_EMBEDDING_DIM`` rather
    than passed as an argument, so callers don't have to plumb the
    same constant through. Local import avoids a module-import cycle
    with ``sqla/__init__.py``.
    """
    # Local import: ``sqla.__init__`` imports this module at package
    # load, which would otherwise close the cycle.
    from .conversation_message import CONVERSATION_MESSAGE_EMBEDDING_DIM

    dim = CONVERSATION_MESSAGE_EMBEDDING_DIM

    rows = await db.fetchall(
        f"""SELECT 1 FROM information_schema.columns
            WHERE table_name = '{table}' AND column_name = 'embedding_vec'""",
        (),
    )
    if rows:
        logger.debug(
            "%s.embedding_vec already present — skipping greenfield PG migration.",
            table,
        )
        return

    table_exists = await db.fetchall(
        f"""SELECT 1 FROM information_schema.tables
            WHERE table_name = '{table}'""",
        (),
    )
    if not table_exists:
        logger.debug(
            "%s table not yet present — skipping greenfield PG migration.", table,
        )
        return

    async with db.transaction():
        # Extension MUST exist before ``ALTER TABLE`` references
        # ``vector(N)``. Mirrors the lesson from #1454 — original order
        # broke on fresh PG without the extension preloaded.
        await db.execute("CREATE EXTENSION IF NOT EXISTS vector", ())
        await db.execute(
            f"ALTER TABLE {table} ADD COLUMN embedding_vec vector({dim})", (),
        )
        await db.execute(
            f"""CREATE INDEX IF NOT EXISTS idx_{table}_embedding_vec_hnsw
                ON {table} USING hnsw (embedding_vec vector_cosine_ops)""",
            (),
        )

    logger.info(
        "%s greenfield PG migration complete: added embedding_vec vector(%d), "
        "HNSW index. No backfill (greenfield column).", table, dim,
    )


# --- #1477 embedding_profile_id stamping ------------------------------------

async def migrate_add_embedding_profile_id(
    db: "AsyncDatabase", *, table: str
) -> None:
    """Add a nullable ``embedding_profile_id`` column to ``table``.

    Idempotent across both backends and tables — pre-checks the
    column via ``information_schema`` / ``pragma_table_info``.
    Wrapped in ``db.transaction()`` so a partial failure rolls back
    cleanly. Existing rows stay NULL; profile-filtered kNN will skip
    them so a deployment that upgrades into 0.21 sees no false
    positives from un-stamped rows. Operators can backfill with the
    ``kestrel-sovereign embeddings reindex`` subcommand once per
    agent.
    """
    backend_type = getattr(db, "backend_type", None)
    if backend_type == "postgres":
        rows = await db.fetchall(
            f"""SELECT 1 FROM information_schema.columns
                WHERE table_name = '{table}'
                  AND column_name = 'embedding_profile_id'""",
            (),
        )
        if rows:
            logger.debug(
                "%s.embedding_profile_id already present — skipping #1477 PG migration.",
                table,
            )
            return
        table_exists = await db.fetchall(
            f"""SELECT 1 FROM information_schema.tables
                WHERE table_name = '{table}'""",
            (),
        )
        if not table_exists:
            logger.debug(
                "%s table not yet present — skipping #1477 PG migration.", table,
            )
            return
        async with db.transaction():
            await db.execute(
                f"ALTER TABLE {table} ADD COLUMN embedding_profile_id TEXT",
                (),
            )
            # An index on the new column accelerates the profile
            # filter in the kNN WHERE clause; without it pgvector
            # has to scan every row that matches the other filters
            # before applying the cosine sort. Cheap to add on a
            # nullable column.
            await db.execute(
                f"""CREATE INDEX IF NOT EXISTS idx_{table}_embedding_profile_id
                    ON {table}(embedding_profile_id)
                    WHERE embedding_profile_id IS NOT NULL""",
                (),
            )
        logger.info(
            "%s #1477 PG migration complete: added embedding_profile_id TEXT + "
            "partial index.", table,
        )
    elif backend_type == "sqlite":
        rows = await db.fetchall(
            f"SELECT name FROM pragma_table_info('{table}') "
            f"WHERE name = 'embedding_profile_id'",
            (),
        )
        if rows:
            logger.debug(
                "%s.embedding_profile_id already present — skipping #1477 SQLite migration.",
                table,
            )
            return
        table_exists = await db.fetchall(
            f"SELECT 1 FROM sqlite_master WHERE type='table' AND name='{table}'",
            (),
        )
        if not table_exists:
            logger.debug(
                "%s table not yet present — skipping #1477 SQLite migration.",
                table,
            )
            return
        async with db.transaction():
            await db.execute(
                f"ALTER TABLE {table} ADD COLUMN embedding_profile_id TEXT", (),
            )
            await db.execute(
                f"""CREATE INDEX IF NOT EXISTS idx_{table}_embedding_profile_id
                    ON {table}(embedding_profile_id)""",
                (),
            )
        logger.info(
            "%s #1477 SQLite migration complete: added embedding_profile_id TEXT.",
            table,
        )


async def migrate_create_embedding_profiles(db: "AsyncDatabase") -> None:
    """Create the ``embedding_profiles`` registry table (#1477).

    Tiny operator-visibility table — one row per
    ``(provider, model, dim, space_id, normalized)`` seen in the
    deployment. Storage code upserts on every successful write; the
    audit CLI reads it. The kNN filter does NOT join this table — it
    matches against the stamped id directly — so this is purely a
    human-readable mapping.

    Idempotent + transactional.
    """
    backend_type = getattr(db, "backend_type", None)
    if backend_type == "postgres":
        rows = await db.fetchall(
            """SELECT 1 FROM information_schema.tables
               WHERE table_name = 'embedding_profiles'""",
            (),
        )
        if rows:
            logger.debug(
                "embedding_profiles already present — skipping #1477 PG migration."
            )
            return
        async with db.transaction():
            await db.execute(
                """CREATE TABLE embedding_profiles (
                    id          TEXT PRIMARY KEY,
                    provider    TEXT NOT NULL,
                    model       TEXT NOT NULL,
                    dim         INTEGER NOT NULL,
                    space_id    TEXT NOT NULL,
                    normalized  BOOLEAN NOT NULL DEFAULT FALSE,
                    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )""",
                (),
            )
        logger.info("embedding_profiles created (PG, #1477).")
    elif backend_type == "sqlite":
        rows = await db.fetchall(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND "
            "name='embedding_profiles'",
            (),
        )
        if rows:
            logger.debug(
                "embedding_profiles already present — skipping #1477 SQLite migration."
            )
            return
        async with db.transaction():
            await db.execute(
                """CREATE TABLE embedding_profiles (
                    id          TEXT PRIMARY KEY,
                    provider    TEXT NOT NULL,
                    model       TEXT NOT NULL,
                    dim         INTEGER NOT NULL,
                    space_id    TEXT NOT NULL,
                    normalized  INTEGER NOT NULL DEFAULT 0,
                    created_at  TEXT DEFAULT CURRENT_TIMESTAMP
                )""",
                (),
            )
        logger.info("embedding_profiles created (SQLite, #1477).")


async def _migrate_sqlite_greenfield(db: "AsyncDatabase", *, table: str) -> None:
    """SQLite greenfield migration — add ``embedding_vec BLOB`` to a
    table that has no existing embedding column.

    No backfill, no copy. SQLite has no HNSW equivalent; the
    PurePythonBackend reads the column directly and computes cosine
    in Python. FEAT-8's ``SqliteVecBackend`` would add a virtual
    table later; that's not coupled to this migration.
    """
    rows = await db.fetchall(
        f"SELECT name FROM pragma_table_info('{table}') WHERE name = 'embedding_vec'",
        (),
    )
    if rows:
        logger.debug(
            "%s.embedding_vec already present — skipping greenfield SQLite migration.",
            table,
        )
        return

    table_exists = await db.fetchall(
        f"SELECT 1 FROM sqlite_master WHERE type='table' AND name='{table}'",
        (),
    )
    if not table_exists:
        logger.debug(
            "%s table not yet present — skipping greenfield SQLite migration.", table,
        )
        return

    async with db.transaction():
        await db.execute(f"ALTER TABLE {table} ADD COLUMN embedding_vec BLOB", ())

    logger.info(
        "%s greenfield SQLite migration complete: added embedding_vec BLOB. "
        "No backfill (greenfield column).", table,
    )


# =============================================================================
# compress → compact terminology rename (session-context shrinking)
# =============================================================================


async def migrate_compaction_terminology(db: "AsyncDatabase") -> None:
    """One-time data rewrite for the compress → compact terminology
    rename: session-context shrinking is "compaction" (the industry
    term), and the persisted metadata strings move with the code so
    readers never need dual-string compat.

    Rewrites ``conversation_history.metadata`` (plaintext JSON TEXT on
    both dialects — only ``content`` is encrypted at rest, see #1401):

    - ``type: "compression"`` → ``"compaction"``
    - ``type: "hierarchical_compression"`` → ``"hierarchical_compaction"``
    - key ``messages_compressed`` → ``messages_compacted``
    - key ``compressed_at`` → ``compacted_at``
    - ``salvage_reason: "manual-compress"`` → ``"manual-compact"``
    - ``excluded_reason: "Replaced by compression"`` → ``"Replaced by compaction"``

    Legacy ``[COMPRESSED CONTEXT …]`` / ``[HIERARCHICAL COMPRESSION …]``
    ``content`` markers are deliberately left alone: content may be
    encrypted at rest, and no code path parses the marker text — it is
    display prose inside the message body.

    Idempotent by construction: rewritten rows no longer match the
    LIKE filter, so re-running is a no-op. The migration runs on every
    boot (no completion sentinel exists), so the filter is kept tight:
    ``compression"`` matches all three quoted JSON values
    (``"compression"``, ``"hierarchical_compression"``, ``"Replaced by
    compression"``) and ``manual-compress`` matches the salvage reason
    — without fetching ordinary rows whose metadata merely mentions the
    word compress. The key renames (``messages_compressed``,
    ``compressed_at``) only occur on marker rows already matched by the
    type filter. Dialect-neutral: plain ``?`` placeholders, no DDL.
    """
    rows = await db.fetchall(
        "SELECT id, metadata FROM conversation_history "
        "WHERE metadata LIKE '%compression\"%' "
        "   OR metadata LIKE '%manual-compress%'",
        (),
    )
    if not rows:
        return

    rewritten = 0
    async with db.transaction():
        for row_id, raw in rows:
            if not raw:
                continue
            try:
                meta = json.loads(raw)
            except (TypeError, ValueError):
                continue
            if not isinstance(meta, dict):
                continue

            new_meta = dict(meta)
            marker_type = new_meta.get("type")
            if marker_type == "compression":
                new_meta["type"] = "compaction"
            elif marker_type == "hierarchical_compression":
                new_meta["type"] = "hierarchical_compaction"
            if "messages_compressed" in new_meta:
                new_meta["messages_compacted"] = new_meta.pop("messages_compressed")
            if "compressed_at" in new_meta:
                new_meta["compacted_at"] = new_meta.pop("compressed_at")
            if new_meta.get("salvage_reason") == "manual-compress":
                new_meta["salvage_reason"] = "manual-compact"
            if new_meta.get("excluded_reason") == "Replaced by compression":
                new_meta["excluded_reason"] = "Replaced by compaction"

            if new_meta == meta:
                continue
            await db.execute(
                "UPDATE conversation_history SET metadata = ? WHERE id = ?",
                (json.dumps(new_meta), row_id),
            )
            rewritten += 1

    if rewritten:
        logger.info(
            "compaction-terminology migration: rewrote %d "
            "conversation_history metadata row(s) (compression → "
            "compaction).", rewritten,
        )
