"""Verify/backfill of ``embedding_vec`` from the legacy ``embedding`` column (#3402).

The SQLite cases run everywhere. The PostgreSQL case runs when
``TEST_POSTGRES_URL`` is set, which the CI unit tier provides.

A freshly booted database has already lost the legacy column (#3411), so the
fixtures restore it and the rows stand in for data an older release wrote.
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
    backfill_missing_embedding_vec,
    verify_embedding_vec,
)
from tests.utils.legacy_embedding_column import restore_legacy_embedding_column
from tests.utils.postgres_schema import (
    pgvector_schema,
    quoted_search_path,
    with_search_path,
)


def _pack(values):
    return struct.pack(f"<{len(values)}f", *values)


@pytest.fixture
async def retired_sqlite_db(tmp_path):
    """A freshly booted database: the legacy column is already retired."""
    db = await AsyncDatabase.sqlite(str(tmp_path / "backfill.db"))
    try:
        yield db
    finally:
        await db.close()


@pytest.fixture
async def sqlite_db(retired_sqlite_db):
    await restore_legacy_embedding_column(retired_sqlite_db)
    return retired_sqlite_db


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


async def test_non_finite_legacy_embedding_is_unbackfillable_not_fatal(sqlite_db):
    db = sqlite_db
    finite = _pack([0.5, -0.25])
    nan = _pack([0.5, float("nan")])
    await _insert_saved_item(db, "a-nan", embedding=nan)
    await _insert_saved_item(db, "b-finite", embedding=finite)

    first = await backfill_embedding_vec(db, "saved_items")

    assert first.rows_backfilled == 1
    assert first.rows_unbackfillable == 1
    assert first.rows_missing_embedding_vec == 1
    assert first.rows_with_both == 1
    assert await _saved_item_columns(db, "b-finite") == (finite, finite)
    assert await _saved_item_columns(db, "a-nan") == (nan, None)

    second = await backfill_embedding_vec(db, "saved_items")

    assert second.rows_backfilled == 0
    assert second.rows_unbackfillable == 1
    assert second.rows_missing_embedding_vec == 1


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


@pytest.mark.parametrize("write", [False, True])
async def test_retired_table_reports_every_vector_as_embedding_vec_only(
    retired_sqlite_db, write
):
    db = retired_sqlite_db
    assert not await db.column_exists("saved_items", "embedding")
    await db.execute(
        "INSERT INTO saved_items (id, agent_id, item_type, name, content, "
        "embedding_vec) VALUES "
        "('vec', 'did:test:agent', 'stash', 'vec', 'c', ?), "
        "('none', 'did:test:agent', 'stash', 'none', 'c', NULL)",
        (_pack([1.0, 2.0]),),
    )

    run = backfill_embedding_vec if write else verify_embedding_vec
    report = await run(db, "saved_items")

    assert report.embedding_vec_present is True
    assert (
        report.total_rows,
        report.rows_with_both,
        report.rows_missing_embedding_vec,
        report.rows_embedding_vec_only,
        report.rows_disagreeing,
        report.rows_backfilled,
        report.rows_unbackfillable,
    ) == (2, 0, 0, 1, 0, 0, 0)
    assert not await db.column_exists("saved_items", "embedding")


async def test_retired_table_without_embedding_vec_reports_the_column_absent(
    retired_sqlite_db,
):
    db = retired_sqlite_db
    await db.execute("ALTER TABLE document_chunks DROP COLUMN embedding_vec", ())
    await db.execute(
        "INSERT INTO document_chunks (file_hash, content) VALUES ('doc', 'chunk')", ()
    )

    report = await verify_embedding_vec(db, "document_chunks")

    assert report.embedding_vec_present is False
    assert (report.total_rows, report.rows_missing_embedding_vec) == (1, 0)


async def test_backfill_missing_copies_what_backfill_copies_without_reading_pairs(
    sqlite_db, monkeypatch
):
    # The startup sequence runs this on every boot the legacy column
    # survives (#3414), so it must not read the vector pairs the report
    # compares.
    db = sqlite_db
    legacy = _pack([0.1, 0.2, 0.3])
    reindexed = _pack([0.9, 0.8, 0.7])
    await _insert_saved_item(db, "a-legacy-only", embedding=legacy)
    await _insert_saved_item(db, "b-both-disagree", embedding=legacy, embedding_vec=reindexed)
    await _insert_saved_item(db, "c-malformed", embedding=b"\x01\x02\x03\x04\x05")
    await _insert_saved_item(db, "d-vec-only", embedding_vec=reindexed)
    await _insert_saved_item(db, "e-legacy-only", embedding=reindexed)
    reads = []

    def recording(method):
        async def record(sql, params=()):
            reads.append(" ".join(sql.split()))
            return await method(sql, params)

        return record

    for name in ("fetchone", "fetchall"):
        monkeypatch.setattr(db, name, recording(getattr(db, name)))

    assert await backfill_missing_embedding_vec(db, "saved_items", batch_size=1) == (2, 1)

    table_reads = [sql for sql in reads if "FROM saved_items" in sql]
    assert table_reads
    assert all("embedding_vec IS NULL" in sql for sql in table_reads), table_reads
    assert await _saved_item_columns(db, "a-legacy-only") == (legacy, legacy)
    assert await _saved_item_columns(db, "e-legacy-only") == (reindexed, reindexed)
    assert await _saved_item_columns(db, "b-both-disagree") == (legacy, reindexed)
    assert await _saved_item_columns(db, "c-malformed") == (b"\x01\x02\x03\x04\x05", None)
    # Idempotent: the unbackfillable row is counted again, nothing written.
    assert await backfill_missing_embedding_vec(db, "saved_items") == (0, 1)


async def test_backfill_missing_writes_nothing_without_both_columns(sqlite_db):
    db = sqlite_db
    await db.execute("ALTER TABLE document_chunks DROP COLUMN embedding_vec", ())
    await db.execute(
        "INSERT INTO document_chunks (file_hash, content, embedding) VALUES (?, ?, ?)",
        ("doc", "chunk", _pack([1.0])),
    )

    # Creating embedding_vec stays with the startup migration and the first
    # embedded write, which choose its width.
    assert await backfill_missing_embedding_vec(db, "document_chunks") == (0, 0)
    assert not await db.column_exists("document_chunks", "embedding_vec")

    # A retired table has no legacy vector to copy.
    await db.execute("ALTER TABLE saved_items DROP COLUMN embedding", ())
    assert await backfill_missing_embedding_vec(db, "saved_items") == (0, 0)


async def test_rejects_unknown_table_and_non_positive_batch(sqlite_db):
    with pytest.raises(ValueError, match="table must be one of"):
        await verify_embedding_vec(sqlite_db, "conversation_history")
    with pytest.raises(ValueError, match="batch_size"):
        await backfill_embedding_vec(sqlite_db, "saved_items", batch_size=0)
    with pytest.raises(ValueError, match="table must be one of"):
        await backfill_missing_embedding_vec(sqlite_db, "conversation_history")
    with pytest.raises(ValueError, match="batch_size"):
        await backfill_missing_embedding_vec(sqlite_db, "saved_items", batch_size=0)


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


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("is_postgres", [True, False])
def test_non_finite_component_has_no_backfill_value(bad, is_postgres):
    # pgvector rejects ``[nan]``/``[inf]``; SQLite applies the same rule.
    assert _backfill_value(_pack([1.0, bad]), _VecColumn(True, None), is_postgres) is None


def test_pgvector_text_repacks_to_the_stored_float32():
    values = [0.1, -0.333333343, 1e-7]
    blob = _pack(values)
    stored = struct.unpack("<3f", blob)
    # pgvector prints the shortest decimal that round-trips each float4.
    text = "[" + ",".join(repr(struct.unpack("<f", struct.pack("<f", v))[0]) for v in stored) + "]"

    assert _pgvector_text_to_bytes(text) == blob
    assert _pgvector_text_to_bytes("[0.1,0.2]") == _pack([0.1, 0.2])
    assert _pgvector_text_to_bytes("[]") == b""


async def _run_no_schema_ddl(db):
    """Schema initializer for a connection opened after the core schema boot."""


@pytest.fixture
async def postgres_db():
    url = os.environ.get("TEST_POSTGRES_URL")
    if not url:
        pytest.skip("TEST_POSTGRES_URL is not set")
    # Boot the core schema on the worker's own search_path, then reconnect
    # with pgvector's schema named after it. An xdist worker's path names only
    # its own schema, and the extension lives wherever it was first
    # installed, so neither this test's ``vector(4)`` DDL nor the helper's
    # ``::vector`` cast would otherwise resolve the type (#3401).
    boot = await AsyncDatabase.postgres(url)
    try:
        vector_schema = await pgvector_schema(boot)
        (schemas,) = await boot.fetchone("SELECT current_schemas(false)", ())
    finally:
        await boot.close()
    db = await AsyncDatabase.postgres(
        with_search_path(url, quoted_search_path(*schemas, vector_schema)),
        schema_initializer=_run_no_schema_ddl,
    )
    file_hash = f"embedding-vec-backfill-{uuid4()}"
    try:
        # The boot retires the legacy column once no row needs it (#3411).
        await restore_legacy_embedding_column(db, "document_chunks")
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
