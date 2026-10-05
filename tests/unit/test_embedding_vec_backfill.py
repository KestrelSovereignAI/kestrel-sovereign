"""Verify/backfill of ``embedding_vec`` from the legacy ``embedding`` column (#3402).

The SQLite cases run everywhere. The PostgreSQL cases run when
``TEST_POSTGRES_URL`` is set, which the CI unit tier provides, each in a schema
of its own so the table-wide backfill sees only the rows it inserted (#3404).

A freshly booted database has already lost the legacy column (#3411), so the
fixtures restore it and the rows stand in for data an older release wrote.
"""

from __future__ import annotations

import json
import os
import struct
from contextlib import asynccontextmanager
from dataclasses import dataclass

import pytest
from kestrel_sdk.storage.database.interface import TransactionError

from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.embedding_vec_backfill import (
    EmbeddingVecBackfillError,
    EmbeddingVecReport,
    _backfill_value,
    _pgvector_text_to_bytes,
    _VecColumn,
    backfill_embedding_vec,
    backfill_missing_embedding_vec,
    verify_embedding_vec,
)
from tests.utils.legacy_embedding_column import restore_legacy_embedding_column
from tests.utils.postgres_schema import (
    disposable_postgres_schema,
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
    # "f-neither" was in no bucket before #3415.
    assert first.rows_without_any_embedding == 1
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
        second.rows_without_any_embedding,
        second.rows_disagreeing,
    ) == (7, 4, 1, 1, 1, 1)


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
    await db.execute(
        "INSERT INTO document_chunks (file_hash, content) VALUES ('doc', 'never')", ()
    )

    report = await backfill_embedding_vec(db, "document_chunks")

    assert report.embedding_vec_present is False
    assert report.total_rows == 2
    assert report.rows_missing_embedding_vec == 1
    assert report.rows_without_any_embedding == 1
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
        report.rows_without_any_embedding,
        report.rows_disagreeing,
        report.rows_backfilled,
        report.rows_unbackfillable,
    ) == (2, 0, 0, 1, 1, 0, 0, 0)
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
    assert (
        report.total_rows,
        report.rows_missing_embedding_vec,
        report.rows_without_any_embedding,
    ) == (1, 0, 1)


@pytest.mark.parametrize(
    "bucket",
    [
        "rows_with_both",
        "rows_missing_embedding_vec",
        "rows_embedding_vec_only",
        "rows_without_any_embedding",
    ],
)
def test_a_report_whose_buckets_do_not_sum_to_total_rows_is_refused(bucket):
    counts = dict(
        rows_with_both=1,
        rows_missing_embedding_vec=1,
        rows_embedding_vec_only=1,
        rows_without_any_embedding=1,
    )
    fields = dict(
        table="document_chunks",
        embedding_vec_present=True,
        total_rows=4,
        rows_disagreeing=0,
        rows_backfilled=0,
        rows_unbackfillable=0,
    )
    assert EmbeddingVecReport(**fields, **counts).total_rows == 4

    # The count #3415 found missing: rows in no bucket.
    with pytest.raises(EmbeddingVecBackfillError, match="sum to 3, not total_rows 4"):
        EmbeddingVecReport(**fields, **{**counts, bucket: 0})


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
    """Schema initializer for a connection that must not boot its schema."""


@dataclass(frozen=True)
class _PostgresSchema:
    """A booted core schema a case owns, and the URL that selects it."""

    db: AsyncDatabase
    url: str
    vector_schema: str


@asynccontextmanager
async def _own_postgres_schema(url):
    """Boot the core schema into a fresh schema on *url*'s database.

    The backfill walks the whole ``document_chunks`` table, so a case run in
    *url*'s own schema rewrites rows it never inserted whenever
    ``TEST_POSTGRES_URL`` names a reused or shared database (#3404). Booting
    *url* would too: the startup sequence backfills legacy rows (#3414). So
    *url* is opened without DDL, only to create the schema, which is dropped
    on exit.

    The boot's path names only the new schema, so no unqualified name in it
    can reach another schema's table. The connection handed out also names
    pgvector's schema, which can be any schema on the database, for the
    helper's ``::vector`` cast (#3401).
    """
    admin = await AsyncDatabase.postgres(url, schema_initializer=_run_no_schema_ddl)
    try:
        vector_schema = await pgvector_schema(admin)
        async with disposable_postgres_schema(admin, "embedding_vec_backfill") as schema:
            boot = await AsyncDatabase.postgres(with_search_path(url, schema))
            await boot.close()
            schema_url = with_search_path(url, quoted_search_path(schema, vector_schema))
            db = await AsyncDatabase.postgres(
                schema_url, schema_initializer=_run_no_schema_ddl
            )
            try:
                # The boot retires the legacy column once no row needs it (#3411).
                await restore_legacy_embedding_column(db, "document_chunks")
                yield _PostgresSchema(db, schema_url, vector_schema)
            finally:
                await db.close()
    finally:
        await admin.close()


# Parametrized with "postgres" so the #3381 guard fails these cases, rather
# than letting them skip, in a job that provides PostgreSQL.
@pytest.fixture(params=["postgres"])
def postgres_url(request):
    url = os.environ.get("TEST_POSTGRES_URL")
    if not url:
        pytest.skip("TEST_POSTGRES_URL is not set")
    return url


def _report_counts(report):
    return (
        report.total_rows,
        report.rows_with_both,
        report.rows_missing_embedding_vec,
        report.rows_embedding_vec_only,
        report.rows_without_any_embedding,
        report.rows_disagreeing,
        report.rows_backfilled,
        report.rows_unbackfillable,
    )


async def _backfill_case(case: _PostgresSchema) -> None:
    db = case.db
    legacy = [_pack([float(i + 1)] * 4) for i in range(3)]
    for index, blob in enumerate(legacy):
        await db.execute(
            "INSERT INTO document_chunks (file_hash, content, embedding) "
            "VALUES (?, ?, ?)",
            ("doc", f"chunk {index}", blob),
        )
    # A chunk never embedded: no vector in either column (#3415).
    await db.execute(
        "INSERT INTO document_chunks (file_hash, content) VALUES (?, ?)",
        ("doc", "never embedded"),
    )

    # The startup migration defers the column until a legacy row exists, so
    # the fresh schema has none. Report that state, then create the column.
    absent = await verify_embedding_vec(db, "document_chunks")
    assert absent.embedding_vec_present is False
    assert _report_counts(absent) == (4, 0, 3, 0, 1, 0, 0, 3)
    await db.execute(
        "ALTER TABLE document_chunks "
        f'ADD COLUMN embedding_vec "{case.vector_schema}".vector(4)',
        (),
    )

    first = await backfill_embedding_vec(db, "document_chunks", batch_size=2)

    assert first.embedding_vec_present is True
    # The table holds only this case's rows, so the counts are exact.
    assert _report_counts(first) == (4, 3, 0, 0, 1, 0, 3, 0)
    rows = await db.fetchall(
        "SELECT embedding, embedding_vec::text FROM document_chunks "
        "WHERE embedding IS NOT NULL ORDER BY chunk_id",
        (),
    )
    assert [bytes(blob) for blob, _ in rows] == legacy
    assert [_pgvector_text_to_bytes(text) for _, text in rows] == legacy

    second = await backfill_embedding_vec(db, "document_chunks", batch_size=2)

    assert _report_counts(second) == (4, 3, 0, 0, 1, 0, 0, 0)


async def test_postgres_backfill_is_idempotent(postgres_url):
    async with _own_postgres_schema(postgres_url) as case:
        await _backfill_case(case)


async def _document_chunks(db):
    """Every ``document_chunks`` row, keyed by column, so a dropped column shows."""
    rows = await db.fetchall(
        "SELECT row_to_json(chunk)::text FROM document_chunks chunk "
        "ORDER BY chunk_id",
        (),
    )
    return [json.loads(text) for (text,) in rows]


async def test_postgres_case_leaves_a_reused_database_unchanged(postgres_url):
    # TEST_POSTGRES_URL can name a reused database (#3404). Stand one up in a
    # schema of this test's own, holding a legacy-only chunk the case did not
    # insert: exactly the row the backfill repairs.
    async with _own_postgres_schema(postgres_url) as reused:
        await reused.db.execute(
            "ALTER TABLE document_chunks "
            f'ADD COLUMN embedding_vec "{reused.vector_schema}".vector(4)',
            (),
        )
        bystander = _pack([0.5, 1.5, 2.5, 3.5])
        await reused.db.execute(
            "INSERT INTO document_chunks (file_hash, content, embedding) "
            "VALUES (?, ?, ?)",
            ("bystander", "not the case's", bystander),
        )
        before = await _document_chunks(reused.db)
        assert [
            (row["file_hash"], row["embedding"], row["embedding_vec"]) for row in before
        ] == [("bystander", "\\x" + bystander.hex(), None)]

        async with _own_postgres_schema(reused.url) as case:
            await _backfill_case(case)

        assert await _document_chunks(reused.db) == before
