"""Verify/backfill of ``embedding_vec`` from the legacy ``embedding`` column (#3402).

The SQLite cases run everywhere. The PostgreSQL case runs when
``TEST_POSTGRES_URL`` is set, which the CI unit tier provides.
"""

from __future__ import annotations

import os
import struct
from uuid import uuid4

import pytest
from kestrel_sdk.storage.database.interface import TransactionError

from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.embedding_vec_backfill import (
    EmbeddingVecBackfillError,
    _backfill_value,
    _pgvector_text_to_bytes,
    _VecColumn,
    backfill_embedding_vec,
    verify_embedding_vec,
)


def _pack(values):
    return struct.pack(f"<{len(values)}f", *values)


@pytest.fixture
async def sqlite_db(tmp_path):
    db = await AsyncDatabase.sqlite(str(tmp_path / "backfill.db"))
    try:
        yield db
    finally:
        await db.close()


async def _insert_saved_item(db, item_id, embedding=None, embedding_vec=None):
    await db.execute(
        "INSERT INTO saved_items (id, agent_id, item_type, name, content, "
        "embedding, embedding_vec) VALUES (?, 'did:test:agent', 'stash', ?, 'c', ?, ?)",
        (item_id, item_id, embedding, embedding_vec),
    )


async def _saved_item_columns(db, item_id):
    row = await db.fetchone(
        "SELECT embedding, embedding_vec FROM saved_items WHERE id = ?", (item_id,)
    )
    return tuple(bytes(v) if v is not None else None for v in row)


async def test_backfill_copies_only_missing_rows_and_second_run_changes_nothing(sqlite_db):
    db = sqlite_db
    legacy = _pack([0.1, 0.2, 0.3])
    reindexed = _pack([0.9, 0.8, 0.7])
    await _insert_saved_item(db, "a-legacy-only", embedding=legacy)
    await _insert_saved_item(db, "b-both-agree", embedding=legacy, embedding_vec=legacy)
    await _insert_saved_item(db, "c-both-disagree", embedding=legacy, embedding_vec=reindexed)
    await _insert_saved_item(db, "d-malformed", embedding=b"\x01\x02\x03\x04\x05")
    await _insert_saved_item(db, "e-vec-only", embedding_vec=reindexed)
    await _insert_saved_item(db, "f-neither")
    await _insert_saved_item(db, "g-legacy-only", embedding=reindexed)

    first = await backfill_embedding_vec(db, "saved_items", batch_size=2)

    assert first.embedding_vec_present is True
    assert first.rows_backfilled == 2
    assert first.rows_unbackfillable == 1
    assert first.total_rows == 7
    assert first.rows_with_both == 4
    assert first.rows_missing_embedding_vec == 1
    assert first.rows_embedding_vec_only == 1
    assert first.rows_disagreeing == 1

    assert await _saved_item_columns(db, "a-legacy-only") == (legacy, legacy)
    assert await _saved_item_columns(db, "g-legacy-only") == (reindexed, reindexed)
    # A disagreeing vector (a reindexed row) is reported, never repaired.
    assert await _saved_item_columns(db, "c-both-disagree") == (legacy, reindexed)
    assert await _saved_item_columns(db, "d-malformed") == (b"\x01\x02\x03\x04\x05", None)

    second = await backfill_embedding_vec(db, "saved_items", batch_size=2)

    assert second.rows_backfilled == 0
    assert second.rows_unbackfillable == 1
    assert (
        second.total_rows,
        second.rows_with_both,
        second.rows_missing_embedding_vec,
        second.rows_embedding_vec_only,
        second.rows_disagreeing,
    ) == (7, 4, 1, 1, 1)


async def test_verify_reports_without_writing(sqlite_db):
    db = sqlite_db
    legacy = _pack([1.0, 2.0])
    await _insert_saved_item(db, "legacy-only", embedding=legacy)

    report = await verify_embedding_vec(db, "saved_items")

    assert report.rows_missing_embedding_vec == 1
    assert report.rows_backfilled == 0
    assert report.rows_unbackfillable == 0
    assert await _saved_item_columns(db, "legacy-only") == (legacy, None)


async def test_interrupted_backfill_keeps_committed_batches_and_resumes(
    sqlite_db, monkeypatch
):
    db = sqlite_db
    vectors = {}
    for index in range(5):
        vectors[index] = _pack([float(index), 0.5])
        await db.execute(
            "INSERT INTO document_chunks (file_hash, content, embedding) VALUES (?, ?, ?)",
            ("doc", f"chunk {index}", vectors[index]),
        )

    real_execute = db.execute
    updates = 0

    async def failing_execute(sql, params=()):
        nonlocal updates
        if sql.startswith("UPDATE document_chunks SET embedding_vec"):
            updates += 1
            if updates == 3:
                raise RuntimeError("simulated crash in the second batch")
        return await real_execute(sql, params)

    monkeypatch.setattr(db, "execute", failing_execute)
    with pytest.raises(TransactionError, match="simulated crash"):
        await backfill_embedding_vec(db, "document_chunks", batch_size=2)
    monkeypatch.setattr(db, "execute", real_execute)

    interrupted = await verify_embedding_vec(db, "document_chunks")
    # The first batch committed; the failed second batch rolled back whole.
    assert interrupted.rows_with_both == 2
    assert interrupted.rows_missing_embedding_vec == 3

    resumed = await backfill_embedding_vec(db, "document_chunks", batch_size=2)

    assert resumed.rows_backfilled == 3
    assert resumed.rows_missing_embedding_vec == 0
    assert resumed.rows_with_both == 5
    assert resumed.rows_disagreeing == 0
    rows = await db.fetchall(
        "SELECT embedding, embedding_vec FROM document_chunks ORDER BY chunk_id", ()
    )
    assert [bytes(vec) for _legacy, vec in rows] == [vectors[i] for i in range(5)]

    again = await backfill_embedding_vec(db, "document_chunks", batch_size=2)
    assert again.rows_backfilled == 0


async def test_missing_embedding_vec_column_is_reported_not_created(sqlite_db):
    db = sqlite_db
    await db.execute("ALTER TABLE document_chunks DROP COLUMN embedding_vec", ())
    await db.execute(
        "INSERT INTO document_chunks (file_hash, content, embedding) VALUES (?, ?, ?)",
        ("doc", "chunk", _pack([1.0])),
    )

    report = await backfill_embedding_vec(db, "document_chunks")

    assert report.embedding_vec_present is False
    assert report.total_rows == 1
    assert report.rows_missing_embedding_vec == 1
    assert report.rows_unbackfillable == 1
    assert report.rows_backfilled == 0
    columns = await db.fetchall(
        "SELECT name FROM pragma_table_info('document_chunks') "
        "WHERE name = 'embedding_vec'",
        (),
    )
    assert columns == []


async def test_rejects_unknown_table_and_non_positive_batch(sqlite_db):
    with pytest.raises(ValueError, match="table must be one of"):
        await verify_embedding_vec(sqlite_db, "conversation_history")
    with pytest.raises(ValueError, match="batch_size"):
        await backfill_embedding_vec(sqlite_db, "saved_items", batch_size=0)


async def test_rejects_unsupported_backend():
    class _OtherBackend:
        backend_type = "mysql"

    with pytest.raises(EmbeddingVecBackfillError, match="unsupported"):
        await verify_embedding_vec(_OtherBackend(), "saved_items")


def test_postgres_value_respects_the_declared_vector_width():
    blob = _pack([0.25, -1.5, 3.0])

    assert _backfill_value(blob, _VecColumn(True, 3), True) == "[0.25,-1.5,3.0]"
    assert _backfill_value(blob, _VecColumn(True, None), True) == "[0.25,-1.5,3.0]"
    assert _backfill_value(blob, _VecColumn(True, 4), True) is None
    assert _backfill_value(b"\x00\x00\x00", _VecColumn(True, None), True) is None
    assert _backfill_value(b"", _VecColumn(True, None), False) is None
    assert _backfill_value(blob, _VecColumn(True, None), False) == blob


def test_pgvector_text_repacks_to_the_stored_float32():
    values = [0.1, -0.333333343, 1e-7]
    blob = _pack(values)
    stored = struct.unpack("<3f", blob)
    # pgvector prints the shortest decimal that round-trips each float4.
    text = "[" + ",".join(repr(struct.unpack("<f", struct.pack("<f", v))[0]) for v in stored) + "]"

    assert _pgvector_text_to_bytes(text) == blob
    assert _pgvector_text_to_bytes("[0.1,0.2]") == _pack([0.1, 0.2])
    assert _pgvector_text_to_bytes("[]") == b""


@pytest.fixture
async def postgres_db():
    url = os.environ.get("TEST_POSTGRES_URL")
    if not url:
        pytest.skip("TEST_POSTGRES_URL is not set")
    db = await AsyncDatabase.postgres(url)
    file_hash = f"embedding-vec-backfill-{uuid4()}"
    try:
        yield db, file_hash
    finally:
        try:
            await db.execute(
                "DELETE FROM document_chunks WHERE file_hash = ?", (file_hash,)
            )
        finally:
            await db.close()


async def test_postgres_backfill_is_idempotent(postgres_db):
    db, file_hash = postgres_db
    column = await db.fetchone(
        "SELECT a.atttypmod FROM pg_attribute a "
        "WHERE a.attrelid = to_regclass('document_chunks') "
        "AND a.attname = 'embedding_vec' AND NOT a.attisdropped",
        (),
    )
    created_column = column is None
    if created_column:
        # The startup migration defers the column until a legacy row exists.
        # Report that state, then create the column for the rest of the case.
        absent = await verify_embedding_vec(db, "document_chunks")
        assert absent.embedding_vec_present is False
        assert absent.rows_backfilled == 0
        await db.execute("CREATE EXTENSION IF NOT EXISTS vector", ())
        await db.execute(
            "ALTER TABLE document_chunks ADD COLUMN embedding_vec vector(4)", ()
        )
        dimension = 4
    else:
        dimension = column[0] if column[0] > 0 else 4
    try:
        legacy = [_pack([float(i + 1)] * dimension) for i in range(3)]
        for index, blob in enumerate(legacy):
            await db.execute(
                "INSERT INTO document_chunks (file_hash, content, embedding) "
                "VALUES (?, ?, ?)",
                (file_hash, f"chunk {index}", blob),
            )

        first = await backfill_embedding_vec(db, "document_chunks", batch_size=2)

        assert first.embedding_vec_present is True
        assert first.rows_backfilled >= 3
        assert first.rows_missing_embedding_vec == first.rows_unbackfillable
        rows = await db.fetchall(
            "SELECT embedding, embedding_vec::text FROM document_chunks "
            "WHERE file_hash = ? ORDER BY chunk_id",
            (file_hash,),
        )
        assert [bytes(blob) for blob, _ in rows] == legacy
        assert [_pgvector_text_to_bytes(text) for _, text in rows] == legacy

        second = await backfill_embedding_vec(db, "document_chunks", batch_size=2)

        assert second.rows_backfilled == 0
        assert second.rows_missing_embedding_vec == first.rows_missing_embedding_vec
        assert second.rows_with_both == first.rows_with_both
    finally:
        if created_column:
            await db.execute(
                "DELETE FROM document_chunks WHERE file_hash = ?", (file_hash,)
            )
            await db.execute(
                "ALTER TABLE document_chunks DROP COLUMN IF EXISTS embedding_vec", ()
            )
