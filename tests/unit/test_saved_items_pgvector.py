"""Tests for Phase 2 of #1447 — ``saved_items.embedding`` BYTEA →
``vector(N)`` on PG.

Covers:

- ``PortableVector`` dialect dispatch: returns ``pgvector.Vector(N)`` on
  PG, ``LargeBinary`` on SQLite.
- ``PortableVector`` bind parameter handling: bytes pass through on
  SQLite, lists pack to float32 bytes, dimension mismatch raises.
- ``SavedItem.embedding`` is wired to ``PortableVector(768)``.
- ``migrate_saved_items_embedding_to_pgvector`` early-exits cleanly on
  SQLite, on already-migrated PG (column is ``vector``), and on a
  missing table; logs a warning and bails on an unexpected column
  type rather than risk data corruption.

The actual happy-path migration (BYTEA → vector(N) on real PG) is
covered by integration tests when a Postgres instance is available;
these unit tests verify the early-exit logic and PortableVector
contract that determine whether the migration runs at all.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sovereign.storage.sqla import SavedItem
from kestrel_sovereign.storage.sqla.migrations import (
    migrate_saved_items_add_embedding_vec,
)
from kestrel_sovereign.storage.sqla.saved_item import SAVED_ITEM_EMBEDDING_DIM
from kestrel_sovereign.storage.sqla.types import PortableVector


# ----------------------------------------------------------------- PortableVector


def test_portable_vector_rejects_non_positive_dimension():
    with pytest.raises(ValueError, match="dimension"):
        PortableVector(0)
    with pytest.raises(ValueError, match="dimension"):
        PortableVector(-1)


def test_portable_vector_uses_largebinary_on_sqlite():
    from sqlalchemy.dialects import sqlite

    pv = PortableVector(768)
    impl = pv.load_dialect_impl(sqlite.dialect())
    # SQLite dialect → LargeBinary descriptor (NOT the pgvector Vector type).
    assert impl.python_type is bytes


def test_portable_vector_uses_pgvector_on_postgresql():
    from sqlalchemy.dialects import postgresql

    pv = PortableVector(768)
    impl = pv.load_dialect_impl(postgresql.dialect())
    # The pgvector Vector type class — confirms we routed to pgvector,
    # not LargeBinary.
    from pgvector.sqlalchemy import Vector
    assert isinstance(impl, Vector)
    assert impl.dim == 768


def test_portable_vector_packs_list_to_bytes_on_sqlite():
    from sqlalchemy.dialects import sqlite

    pv = PortableVector(4)
    out = pv.process_bind_param([1.0, 2.0, 3.0, 4.0], sqlite.dialect())
    assert isinstance(out, bytes)
    assert len(out) == 4 * 4  # 4 floats × 4 bytes each


def test_portable_vector_passes_bytes_through_on_sqlite():
    from sqlalchemy.dialects import sqlite

    pv = PortableVector(4)
    raw = b"\x00" * 16
    out = pv.process_bind_param(raw, sqlite.dialect())
    assert out == raw  # unchanged


def test_portable_vector_unpacks_bytes_to_list_for_pg_path():
    """The PG bind path needs lists for pgvector to consume. If the
    caller hands us legacy AsyncDatabase BYTEA (the case at migration
    time), unpack first.
    """
    import struct

    from sqlalchemy.dialects import postgresql

    pv = PortableVector(4)
    packed = struct.pack("<4f", 0.1, 0.2, 0.3, 0.4)
    out = pv.process_bind_param(packed, postgresql.dialect())
    assert isinstance(out, list)
    assert out == pytest.approx([0.1, 0.2, 0.3, 0.4], rel=1e-6)


def test_portable_vector_raises_on_dimension_mismatch():
    """Wrong-length bytes would silently truncate / wrap in the pgvector
    bind path. Better to raise so the caller fixes the input.
    """
    from sqlalchemy.dialects import postgresql

    pv = PortableVector(4)
    raw = b"\x00" * 8  # 2 floats, not 4
    with pytest.raises(ValueError, match="length"):
        pv.process_bind_param(raw, postgresql.dialect())


def test_portable_vector_none_passes_through():
    """NULL embeddings stay NULL regardless of dialect."""
    from sqlalchemy.dialects import postgresql, sqlite

    pv = PortableVector(4)
    assert pv.process_bind_param(None, postgresql.dialect()) is None
    assert pv.process_bind_param(None, sqlite.dialect()) is None


# ----------------------------------------------------------------- SavedItem wiring


def test_saved_item_embedding_uses_portable_vector_and_correct_sql_column():
    """The ORM points at the parallel ``embedding_vec`` SQL column added
    by the Phase-2 migration (NOT the legacy ``embedding`` BYTEA / BLOB
    column used by raw IO). The split keeps the raw INSERT / from_row
    code paths working unchanged.
    """
    embedding_col = SavedItem.__table__.columns["embedding_vec"]
    assert isinstance(embedding_col.type, PortableVector)
    assert embedding_col.type.dimension == SAVED_ITEM_EMBEDDING_DIM == 768
    # Python attribute name is still ``embedding``.
    assert SavedItem.embedding.expression.name == "embedding_vec"
    # And the legacy ``embedding`` SQL column is NOT in the ORM
    # mapping — it's used only by raw ``AsyncDatabase`` IO.
    assert "embedding" not in SavedItem.__table__.columns


# ----------------------------------------------------------------- migration early-exits


def _fake_db_with_fetchall(
    backend_type: str,
    fetchall_returns: list,
) -> MagicMock:
    """Stub ``AsyncDatabase`` with ``backend_type`` + queued
    ``fetchall`` responses. Includes an async ``transaction()`` context
    manager so the migration's ``async with db.transaction():`` is
    happy without a real DB.
    """
    db = MagicMock()
    db.backend_type = backend_type
    db.fetchall = AsyncMock(side_effect=fetchall_returns)
    db.execute = AsyncMock()

    class _TxCM:
        async def __aenter__(self_inner):
            return self_inner
        async def __aexit__(self_inner, *a):
            return False
    db.transaction = MagicMock(return_value=_TxCM())
    return db


@pytest.mark.asyncio
async def test_migration_is_noop_when_pg_column_already_exists():
    """Already-migrated DB: ``information_schema.columns`` reports
    ``embedding_vec``. Migration must short-circuit without DDL.
    """
    db = _fake_db_with_fetchall("postgres", [[(1,)]])
    await migrate_saved_items_add_embedding_vec(db)
    db.execute.assert_not_called()


@pytest.mark.asyncio
async def test_migration_is_noop_when_table_missing_pg():
    """Fresh DB: ``saved_items`` table doesn't exist yet (init_schema
    hasn't run). Migration must bail rather than fail."""
    db = _fake_db_with_fetchall(
        "postgres",
        [
            [],   # embedding_vec probe → not present
            [],   # source-column probe → no saved_items table
        ],
    )
    await migrate_saved_items_add_embedding_vec(db)
    db.execute.assert_not_called()


@pytest.mark.asyncio
async def test_migration_defers_column_creation_when_no_embedded_rows_pg():
    """Fresh PG DB with the table but NO embedded rows: migration
    deliberately does NOT create the vector column. The embedding-
    service dimension isn't known here (could be 768 / 1024 / 1536),
    and creating ``vector(768)`` against a different-dim writer would
    fail every save. Defer until the next boot, when the migration
    can sniff the actual dim from saved rows. (Caught by codex review
    on the Phase 2 PR.)
    """
    db = _fake_db_with_fetchall(
        "postgres",
        [
            [],            # embedding_vec probe → absent
            [("bytea",)],  # source-column probe → BYTEA exists
            [],            # sniff → no rows
        ],
    )
    await migrate_saved_items_add_embedding_vec(db)
    # NO DDL — defers entirely.
    db.execute.assert_not_called()


@pytest.mark.asyncio
async def test_migration_backfills_existing_rows_and_skips_mismatched_dim_pg():
    """Sniff dim from one row, backfill rows that match, skip rows at
    a different dim, finish with the HNSW index. The legacy
    ``embedding`` column is never touched.
    """
    import struct

    good_emb = struct.pack("<4f", 1.0, 0.0, 0.0, 0.0)
    bad_emb = struct.pack("<2f", 1.0, 0.0)  # different-model dim

    db = _fake_db_with_fetchall(
        "postgres",
        [
            [],                      # embedding_vec absent
            [("bytea",)],            # source-column probe
            [(16,)],                 # sniff: 16 bytes = 4 floats
            [                        # backfill scan
                ("row-a", good_emb),
                ("row-b", bad_emb),
            ],
        ],
    )
    await migrate_saved_items_add_embedding_vec(db)

    calls = [c.args[0] for c in db.execute.call_args_list]
    update_calls = [
        c for c in db.execute.call_args_list
        if c.args[0].startswith("UPDATE saved_items SET embedding_vec")
    ]
    assert len(update_calls) == 1  # only row-a
    vec_text = update_calls[0].args[1][0]
    assert vec_text.startswith("[") and vec_text.endswith("]")
    assert any("ADD COLUMN embedding_vec vector(4)" in q for q in calls)


@pytest.mark.asyncio
async def test_migration_creates_extension_before_alter_table_pg():
    """Regression: ``CREATE EXTENSION IF NOT EXISTS vector`` must run
    BEFORE ``ALTER TABLE ... vector(N)`` references the type. On a fresh
    Postgres database without pgvector installed, the wrong order makes
    the ALTER fail with ``type "vector" does not exist`` and the
    migration never recovers (the startup exception is caught + logged
    but ``embedding_vec`` is never created). (Caught by codex review.)
    """
    db = _fake_db_with_fetchall(
        "postgres",
        [
            [],            # embedding_vec absent
            [("bytea",)],  # source-column probe
            [(16,)],       # sniff: 4 floats → triggers actual DDL
            [],            # backfill scan: empty (no rows after sniff)
        ],
    )
    await migrate_saved_items_add_embedding_vec(db)

    sql_calls = [c.args[0] for c in db.execute.call_args_list]
    create_ext_idx = next(
        (i for i, q in enumerate(sql_calls) if "CREATE EXTENSION" in q), None
    )
    alter_idx = next(
        (i for i, q in enumerate(sql_calls) if "ADD COLUMN embedding_vec vector" in q),
        None,
    )
    assert create_ext_idx is not None, "expected CREATE EXTENSION"
    assert alter_idx is not None, "expected ALTER TABLE ADD COLUMN"
    assert create_ext_idx < alter_idx, (
        f"CREATE EXTENSION at {create_ext_idx} must precede ALTER at {alter_idx}"
    )


@pytest.mark.asyncio
async def test_migration_wraps_in_transaction_pg():
    """Codex-flagged P2: the migration must run inside
    ``db.transaction()`` so partial failures roll back cleanly. Without
    this, a crash mid-migration could leave the schema half-done and
    later boots either skip incorrectly or attempt the conversion
    twice.
    """
    db = _fake_db_with_fetchall(
        "postgres",
        [
            [],            # embedding_vec absent
            [("bytea",)],  # source-column probe
            [(16,)],       # sniff: 4 floats
            [],            # backfill scan empty
        ],
    )
    await migrate_saved_items_add_embedding_vec(db)
    db.transaction.assert_called_once()


# ----------------------------------------------------------------- SQLite path


@pytest.mark.asyncio
async def test_migration_sqlite_adds_column_and_copies_bytes():
    db = _fake_db_with_fetchall(
        "sqlite",
        [
            [],            # pragma_table_info probe → embedding_vec absent
            [(1,)],        # sqlite_master probe → table exists
        ],
    )
    await migrate_saved_items_add_embedding_vec(db)
    calls = [c.args[0] for c in db.execute.call_args_list]
    assert any("ADD COLUMN embedding_vec BLOB" in q for q in calls)
    # Bytes copy is part of the migration so legacy rows are
    # immediately searchable via the ORM column.
    assert any(
        "UPDATE saved_items SET embedding_vec = embedding" in q for q in calls
    )


@pytest.mark.asyncio
async def test_migration_sqlite_skips_when_column_present():
    db = _fake_db_with_fetchall(
        "sqlite",
        [[("embedding_vec",)]],
    )
    await migrate_saved_items_add_embedding_vec(db)
    db.execute.assert_not_called()


@pytest.mark.asyncio
async def test_migration_skips_unknown_dialect():
    db = _fake_db_with_fetchall("mysql", [])
    await migrate_saved_items_add_embedding_vec(db)
    db.execute.assert_not_called()
    db.fetchall.assert_not_called()
