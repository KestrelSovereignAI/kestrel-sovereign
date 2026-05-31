"""Tests for the SQLA + vector-backend path in :class:`AsyncRAGStore`.

Mirrors the saved-items SQLA test surface against the
``document_chunks`` table:

- ``DocumentChunk`` ORM wiring (column types, embedding maps to
  ``embedding_vec``)
- ``build_document_chunk_spec`` validates dimension
- ``migrate_document_chunks_add_embedding_vec`` early-exits cleanly
  on SQLite without a table / on already-migrated PG / on a fresh
  PG without embedded rows; runs the conversion + index when rows
  exist
- ``AsyncRAGStore._search_by_embedding`` falls back to the in-Python
  legacy path when the vector backend errors
- End-to-end SQLite kNN: chunked docs + search returns the right
  ordering

The actual PG happy-path (BYTEA → vector(N) backfill + native
``<=>`` kNN) is exercised by integration tests when a Postgres
container is available; these unit tests cover the early-exit logic
and fallback contract.
"""

from __future__ import annotations

import os
import struct
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.async_rag_store import AsyncRAGStore
from kestrel_sovereign.storage.sqla import (
    DocumentChunk,
    build_document_chunk_spec,
)
from kestrel_sovereign.storage.sqla.document_chunk import (
    DOCUMENT_CHUNK_EMBEDDING_DIM,
)
from kestrel_sovereign.storage.sqla.migrations import (
    migrate_document_chunks_add_embedding_vec,
)
from kestrel_sovereign.storage.sqla.types import PortableVector


# ----------------------------------------------------------------- spec wiring


def test_document_chunk_embedding_maps_to_embedding_vec():
    """ORM points at the parallel ``embedding_vec`` SQL column, NOT the
    legacy ``embedding`` BYTEA / BLOB column. Same split we used for
    saved_items so raw IO paths in ``chunk_document`` don't break.
    """
    col = DocumentChunk.__table__.columns["embedding_vec"]
    assert isinstance(col.type, PortableVector)
    assert col.type.dimension == DOCUMENT_CHUNK_EMBEDDING_DIM == 768
    assert DocumentChunk.embedding.expression.name == "embedding_vec"
    # Legacy ``embedding`` is intentionally absent from the ORM
    # mapping — used only by raw ``AsyncDatabase`` IO.
    assert "embedding" not in DocumentChunk.__table__.columns


def test_build_document_chunk_spec_validates_dim():
    spec = build_document_chunk_spec(dimension=1024)
    assert spec.entity is DocumentChunk
    assert spec.dimension == 1024
    # RAG is global per-DB — no required filter keys.
    assert spec.required_filter_keys == ()
    # ``file_hash`` is exposed as an optional WHERE filter for callers
    # that want to scope to a single source document.
    assert "file_hash" in spec.filter_columns
    assert spec.tenant_id_filter_key is None
    with pytest.raises(ValueError, match="dimension"):
        build_document_chunk_spec(dimension=0)


# ----------------------------------------------------------------- migration


def _fake_db_with_fetchall(backend_type: str, fetchall_returns: list) -> MagicMock:
    """Stub ``AsyncDatabase`` with queued fetchall returns + an async
    ``transaction()`` context manager so the migration runs."""
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
async def test_migration_skips_when_column_present_pg():
    db = _fake_db_with_fetchall("postgres", [[(1,)]])
    await migrate_document_chunks_add_embedding_vec(db)
    db.execute.assert_not_called()


@pytest.mark.asyncio
async def test_migration_skips_when_table_missing_pg():
    db = _fake_db_with_fetchall("postgres", [[], []])
    await migrate_document_chunks_add_embedding_vec(db)
    db.execute.assert_not_called()


@pytest.mark.asyncio
async def test_migration_defers_when_no_embedded_rows_pg():
    """Fresh PG DB with no embedded chunks → defer column creation,
    same as the saved_items behavior. We don't guess a dim."""
    db = _fake_db_with_fetchall(
        "postgres",
        [
            [],            # embedding_vec absent
            [("bytea",)],  # source-column probe
            [],            # sniff → no rows
        ],
    )
    await migrate_document_chunks_add_embedding_vec(db)
    db.execute.assert_not_called()


@pytest.mark.asyncio
async def test_migration_backfills_existing_rows_pg():
    """Sniff dim from one row, backfill rows that match, finish with
    HNSW index. Legacy ``embedding`` column is never touched."""
    good_emb = struct.pack("<4f", 1.0, 0.0, 0.0, 0.0)
    bad_emb = struct.pack("<2f", 1.0, 0.0)  # different-model dim

    db = _fake_db_with_fetchall(
        "postgres",
        [
            [],                          # embedding_vec absent
            [("bytea",)],                # source-column probe
            [(16,)],                     # sniff: 4 floats
            [                            # backfill scan
                (1, good_emb),
                (2, bad_emb),
            ],
        ],
    )
    await migrate_document_chunks_add_embedding_vec(db)
    calls = [c.args[0] for c in db.execute.call_args_list]
    # CREATE EXTENSION runs BEFORE the ALTER (codex review on #1454
    # for saved_items — same constraint applies here).
    ext_idx = next(i for i, q in enumerate(calls) if "CREATE EXTENSION" in q)
    alter_idx = next(i for i, q in enumerate(calls) if "ADD COLUMN embedding_vec vector" in q)
    assert ext_idx < alter_idx

    assert any("ADD COLUMN embedding_vec vector(4)" in q for q in calls)
    assert any(
        "USING hnsw" in q and "vector_cosine_ops" in q for q in calls
    )
    # Backfill count: one good row, one skipped at mismatched dim.
    updates = [c for c in db.execute.call_args_list
               if "UPDATE document_chunks SET embedding_vec" in c.args[0]]
    assert len(updates) == 1


@pytest.mark.asyncio
async def test_migration_sqlite_adds_column_and_copies_bytes():
    db = _fake_db_with_fetchall(
        "sqlite",
        [
            [],            # pragma_table_info probe → embedding_vec absent
            [(1,)],        # sqlite_master probe → table exists
        ],
    )
    await migrate_document_chunks_add_embedding_vec(db)
    calls = [c.args[0] for c in db.execute.call_args_list]
    assert any("ADD COLUMN embedding_vec BLOB" in q for q in calls)
    assert any(
        "UPDATE document_chunks SET embedding_vec = embedding" in q
        for q in calls
    )


@pytest.mark.asyncio
async def test_migration_skips_unknown_dialect():
    db = _fake_db_with_fetchall("mysql", [])
    await migrate_document_chunks_add_embedding_vec(db)
    db.execute.assert_not_called()
    db.fetchall.assert_not_called()


# ----------------------------------------------------------------- search fallback


@pytest.mark.asyncio
async def test_search_via_vector_backend_returns_none_on_failure():
    """An unexpected error inside the vector backend path must NOT
    propagate — the caller falls back to the legacy in-Python search.
    """
    mock_db = MagicMock()
    store = AsyncRAGStore(mock_db)

    import kestrel_sovereign.storage.vector as sov_vector
    real_factory = sov_vector.get_vector_backend

    def _boom(*a, **k):
        raise RuntimeError("simulated backend failure")

    sov_vector.get_vector_backend = _boom
    try:
        result = await store._search_via_vector_backend(
            session_factory=MagicMock(),
            query_embedding=[1.0] + [0.0] * 767,
            limit=5,
            min_score=0.0,
        )
    finally:
        sov_vector.get_vector_backend = real_factory
    assert result is None


@pytest.mark.asyncio
async def test_search_caches_session_factory_unavailable():
    mock_db = MagicMock()
    mock_db.backend_type = "unknown-backend"
    store = AsyncRAGStore(mock_db)
    sf1 = store._get_vector_session_factory()
    assert sf1 is None
    assert store._sqla_factory_unavailable is True
    # Second call doesn't re-attempt construction.
    sf2 = store._get_vector_session_factory()
    assert sf2 is None


# ----------------------------------------------------------------- end-to-end SQLite


@pytest.mark.asyncio
async def test_search_end_to_end_against_real_sqlite():
    """Insert chunks with embeddings, then search via the new vector
    backend through a real ``make_session_factory`` against on-disk
    SQLite. Returns the most similar chunk first.
    """
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "rag_test.db")
        db = await AsyncDatabase.sqlite(db_path)

        try:
            store = AsyncRAGStore(db)

            target = [1.0] + [0.0] * 767
            distractor = [0.0, 1.0] + [0.0] * 766

            # Manual dual-write — production goes through
            # ``chunk_document``'s dual-write path, but that helper
            # also computes embeddings (Ollama-dependent); the test
            # writes both columns directly to stay focused on the
            # search path.
            for fh, vec in [("file-target", target), ("file-distractor", distractor)]:
                packed = struct.pack(f"<768f", *vec)
                cur = await db.execute(
                    "INSERT INTO document_chunks (file_hash, content, embedding) "
                    "VALUES (?, ?, ?)",
                    (fh, f"content for {fh}", packed),
                )
                chunk_id = getattr(cur, "lastrowid", None)
                if chunk_id is None:
                    row = await db.fetchone(
                        "SELECT chunk_id FROM document_chunks WHERE file_hash = ?",
                        (fh,),
                    )
                    chunk_id = row[0]
                await db.execute(
                    "UPDATE document_chunks SET embedding_vec = ? WHERE chunk_id = ?",
                    (packed, chunk_id),
                )
            await db.commit()

            # Stub the store embedding service so the search picks the
            # vector path. Returning ``target`` makes the
            # file-target chunk highest-similarity.
            class _StubEmbed:
                async def aembed(self, _q):
                    return target

            store._get_embedding_service = lambda: _StubEmbed()
            results = await store._search_by_embedding("anything", limit=5)

            assert len(results) == 2
            assert results[0]["file_hash"] == "file-target"
            assert results[0]["score"] > results[1]["score"]
            assert results[0]["source"] == "embedding"
        finally:
            await db.close()
