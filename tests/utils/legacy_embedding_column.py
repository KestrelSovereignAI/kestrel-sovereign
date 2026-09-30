"""A pre-#3411 schema for tests that need the legacy ``embedding`` column.

Since phase 3 of #2684 no store writes ``saved_items.embedding`` or
``document_chunks.embedding``, and the startup migration drops the column
once no row depends on it, which a freshly booted test database always
satisfies. Tests of the tooling that still reads legacy vectors (the verify
and backfill helper, the readers' fallback, the upgrade itself) put the
column back the way an older release left it, then seed it directly.
"""

from __future__ import annotations

LEGACY_TABLES = ("saved_items", "document_chunks")


async def restore_legacy_embedding_column(db, *tables: str) -> None:
    """Add the legacy ``embedding`` column to *tables* where it is absent.

    *tables* defaults to both tables that carried it. The type is the one
    ``CREATE TABLE`` declares: ``BLOB``, which PostgreSQL spells ``BYTEA``.
    """
    column_type = "BYTEA" if db.backend_type == "postgres" else "BLOB"
    for table in tables or LEGACY_TABLES:
        if not await db.column_exists(table, "embedding"):
            await db.execute(
                f"ALTER TABLE {table} ADD COLUMN embedding {column_type}", ()
            )
