"""Phase 3 of retiring the legacy ``embedding`` column (#3411, parent #2684).

Writers store vectors in ``embedding_vec`` only, creating the column when a
fresh PostgreSQL database has none yet, and the startup migration drops the
legacy column once the verify gate is met: ``embedding_vec`` exists and no
row is missing it, copyable or not. Otherwise it keeps the column and every
byte in it.

Every case runs on SQLite and on PostgreSQL. The upgrade cases boot the real
schema twice, the second boot being the upgrade. Rows are seeded the way an
older release's dual-write left them.
"""

from __future__ import annotations

import logging
import struct
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, List, Optional

import pytest

from kestrel_sovereign.storage import embedding_vec_backfill
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.async_rag_store import AsyncRAGStore, IndexedChunk
from kestrel_sovereign.storage.embedding_column import (
    decode_stored_embedding,
    ensure_embedding_vec_column,
    stored_embedding_column,
)
from kestrel_sovereign.storage.embedding_reindex import EmbeddingReindexer
from kestrel_sovereign.storage.embedding_vec_backfill import (
    EmbeddingVecReport,
    backfill_embedding_vec,
    verify_embedding_vec,
)
from kestrel_sovereign.storage.saved_items_store import SavedItemsStore
from kestrel_sovereign.storage.sqla import migrations
from kestrel_sovereign.storage.sqla.migrations import (
    legacy_embedding_retirement_gate_met,
    migrate_retire_legacy_embedding_column,
)
from tests.utils.legacy_embedding_column import restore_legacy_embedding_column
from tests.utils.postgres_schema import (
    disposable_postgres_schema,
    pgvector_schema,
    postgres_test_url,
    quoted_search_path,
    with_search_path,
)

AGENT = "did:web:test.invalid:embedding-retirement"
DIM = 4
TABLES = ("saved_items", "document_chunks")
ID_COLUMNS = {"saved_items": "id", "document_chunks": "chunk_id"}


class _Model:
    """A deterministic embedding model: each known text maps to one vector."""

    def __init__(self, profile_id: str, vectors: Dict[str, List[float]]):
        self.profile_id = profile_id
        self.vectors = vectors

    async def aembed(self, text: str) -> Optional[List[float]]:
        return self.vectors.get(text)

    async def aembed_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        return [self.vectors.get(text) for text in texts]

    def current_profile_id(self) -> str:
        return self.profile_id

    def describe(self):
        # The profile registry upsert is best-effort; None skips it.
        return None


# "alpha" and "beta" swap coordinates between the models, so a search ranks
# by whichever vector it actually scored.
OLD = _Model(
    "old-model-01",
    {"alpha": [0.0, 0.0, 0.0, 1.0], "beta": [0.0, 0.0, 1.0, 0.0]},
)
NEW = _Model(
    "new-model-02",
    {
        "alpha": [0.0, 0.0, 1.0, 0.0],
        "beta": [0.0, 0.0, 0.0, 1.0],
        "gamma": [0.0, 1.0, 0.0, 0.0],
        # Means "gamma" without containing the word, so LIKE cannot find it.
        "third letter": [0.0, 1.0, 0.0, 0.0],
    },
)


def _pack(values: List[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


@dataclass
class _Deployment:
    """One database, reopened through the real startup boot on demand."""

    backend: str
    boot: Callable[[], Awaitable[AsyncDatabase]]
    # DDL type the startup migration gives ``embedding_vec`` on this backend.
    vector_type: str
    db: Optional[AsyncDatabase] = None

    async def reboot(self) -> AsyncDatabase:
        """Close the database and open it again, running every migration."""
        if self.db is not None:
            await self.db.close()
        self.db = await self.boot()
        return self.db

    async def use_pre_retirement_schema(self) -> None:
        """Put back what a boot of a release before #3411 left.

        SQLite: the legacy column (the boot just dropped it) and
        ``embedding_vec``. PostgreSQL: the boot kept the legacy column and
        deferred ``embedding_vec``; add it as the startup migration would.
        """
        await restore_legacy_embedding_column(self.db)
        for table in TABLES:
            if not await self.db.column_exists(table, "embedding_vec"):
                await self.db.execute(
                    f"ALTER TABLE {table} ADD COLUMN embedding_vec "
                    f"{self.vector_type}",
                    (),
                )

    async def write_legacy(self, table: str, row_id, vector: List[float]) -> None:
        """Store *vector* in the legacy column only, as an older release did."""
        await self.db.execute(
            f"UPDATE {table} SET embedding = ?, embedding_vec = NULL "
            f"WHERE {ID_COLUMNS[table]} = ?",
            (_pack(vector), row_id),
        )

    async def snapshot(self, table: str) -> list:
        """Every row's id, content, both embedding columns and profile id."""
        vec = "embedding_vec::text" if self.backend == "postgres" else "embedding_vec"
        legacy = (
            "embedding"
            if await self.db.column_exists(table, "embedding")
            else "NULL"
        )
        rows = await self.db.fetchall(
            f"SELECT {ID_COLUMNS[table]}, content, {legacy}, {vec}, "
            f"embedding_profile_id FROM {table} ORDER BY {ID_COLUMNS[table]}",
            (),
        )
        return [
            (
                row[0],
                row[1],
                None if row[2] is None else bytes(row[2]),
                decode_stored_embedding(row[3]),
                row[4],
            )
            for row in rows
        ]


async def _leave_schema_unchanged(db):
    """Schema initializer for the connection that only manages schemas."""


@pytest.fixture(params=["sqlite", "postgres"])
async def deployment(request, tmp_path):
    if request.param == "sqlite":
        path = str(tmp_path / "retirement.db")
        dep = _Deployment("sqlite", lambda: AsyncDatabase.sqlite(path), "BLOB")
        await dep.reboot()
        try:
            yield dep
        finally:
            await dep.db.close()
        return

    url = postgres_test_url()
    if not url:
        pytest.skip("TEST_POSTGRES_URL is not set")
    admin = await AsyncDatabase.postgres(
        url, schema_initializer=_leave_schema_unchanged
    )
    try:
        vector_schema = await pgvector_schema(admin)
        async with disposable_postgres_schema(admin, "embedding_retire") as schema:
            # The core schema boots into a schema of this test's own, so its
            # columns can be added and dropped without any other test seeing
            # it. pgvector's schema follows on the path for the ``::vector``
            # casts; the DDL qualifies the type (#3401).
            schema_url = with_search_path(
                url, quoted_search_path(schema, vector_schema)
            )
            dep = _Deployment(
                "postgres",
                lambda: AsyncDatabase.postgres(schema_url),
                f'"{vector_schema}".vector({DIM})',
            )
            db = await dep.reboot()
            try:
                for table in TABLES:
                    # migrate_add_embedding_profile_id probes an unscoped
                    # information_schema, so a same-named table in another
                    # schema can make it skip this one.
                    await db.execute(
                        f"ALTER TABLE {table} "
                        "ADD COLUMN IF NOT EXISTS embedding_profile_id TEXT",
                        (),
                    )
                yield dep
            finally:
                await dep.db.close()
    finally:
        await admin.close()


def _saved_items_store(db, model: Optional[_Model]) -> SavedItemsStore:
    store = SavedItemsStore(db, agent_id=AGENT)
    store._get_embedding_service = lambda: model
    # Route ``search`` through the in-Python reader, which is every bound
    # store's path and reads the resolved stored-embedding column.
    store._sqla_factory_unavailable = True
    return store


async def _rag_store(db, model: Optional[_Model], file_hash: str) -> AsyncRAGStore:
    await db.execute(
        "INSERT INTO files (content_hash, original_name) VALUES (?, ?)",
        (file_hash, "retirement.md"),
    )
    await db.execute(
        "INSERT INTO file_owners (content_hash, agent_id, original_name) "
        "VALUES (?, ?, ?)",
        (file_hash, AGENT, "retirement.md"),
    )
    store = AsyncRAGStore(db, agent_id=AGENT)
    store._get_embedding_service = lambda: model
    return store


async def _chunk_ids(db, file_hash: str) -> Dict[str, int]:
    rows = await db.fetchall(
        "SELECT content, chunk_id FROM document_chunks WHERE file_hash = ?",
        (file_hash,),
    )
    return {content: chunk_id for content, chunk_id in rows}


async def _seed_legacy_rows(dep: _Deployment) -> Dict[str, str]:
    """Save "alpha" and "beta" to both tables as an older release did.

    Returns the saved-item ids by text.
    """
    db = dep.db
    ids = {}
    for text in ("alpha", "beta"):
        item = await _saved_items_store(db, OLD).save_item(
            item_type="excerpt", name=text, content=text
        )
        await dep.write_legacy("saved_items", item.id, OLD.vectors[text])
        ids[text] = item.id
    rag = await _rag_store(db, OLD, "retire-doc")
    await rag.store_precomputed_chunks(
        "retire-doc",
        [IndexedChunk(text, OLD.vectors[text], OLD.profile_id) for text in ("alpha", "beta")],
    )
    for text, chunk_id in (await _chunk_ids(db, "retire-doc")).items():
        await dep.write_legacy("document_chunks", chunk_id, OLD.vectors[text])
    return ids


async def _legacy_values(db, table: str) -> int:
    (count,) = await db.fetchone(
        f"SELECT COUNT(*) FROM {table} WHERE embedding IS NOT NULL", ()
    )
    return int(count)


# ------------------------------------------------------------ the upgrade


async def test_boot_with_the_gate_met_drops_the_legacy_column_and_search_works(
    deployment,
):
    dep = deployment
    await dep.use_pre_retirement_schema()
    ids = await _seed_legacy_rows(dep)
    for table in TABLES:
        report = await backfill_embedding_vec(dep.db, table)
        assert legacy_embedding_retirement_gate_met(report)
    # A reindexed table: its legacy bytes disagree with embedding_vec, which
    # does not block the drop.
    reindexer = EmbeddingReindexer(dep.db, NEW, NEW.profile_id, column_dim=DIM)
    assert (await reindexer.reindex_table("saved_items")).failed == 0
    assert (await verify_embedding_vec(dep.db, "saved_items")).rows_disagreeing == 2
    before = {table: await dep.snapshot(table) for table in TABLES}

    db = await dep.reboot()

    for table in TABLES:
        assert not await db.column_exists(table, "embedding")
        after = await dep.snapshot(table)
        # Every row and its stored vector survive; only the legacy bytes go.
        assert [row[:2] + row[3:] for row in after] == [
            row[:2] + row[3:] for row in before[table]
        ]
        report = await verify_embedding_vec(db, table)
        assert legacy_embedding_retirement_gate_met(report)
        assert report.rows_embedding_vec_only == 2

    items = _saved_items_store(db, NEW)
    results = await items.search("alpha", limit=2)
    assert [r["item"]["id"] for r in results] == [ids["alpha"], ids["beta"]]
    assert results[0]["score"] == pytest.approx(1.0)
    assert (await items.get_by_id(ids["beta"])).embedding == pytest.approx(
        NEW.vectors["beta"]
    )
    chunks = await _rag_store(db, OLD, "retire-check")
    found = await chunks._search_by_embedding("alpha", limit=2)
    assert [r["content"] for r in found] == ["alpha", "beta"]
    assert found[0]["score"] == pytest.approx(1.0)

    # New writes land in embedding_vec and are found.
    gamma = await items.save_item(item_type="excerpt", name="gamma", content="gamma")
    assert gamma.embedding == pytest.approx(NEW.vectors["gamma"])
    assert [r["item"]["id"] for r in await items.search("third letter", limit=1)] == [
        gamma.id
    ]

    # Idempotent: another boot, and a direct call, leave it retired.
    db = await dep.reboot()
    for table in TABLES:
        assert not await db.column_exists(table, "embedding")
        assert await migrate_retire_legacy_embedding_column(db, table) is False


@pytest.mark.parametrize("blocker", ["missing", "unbackfillable"])
async def test_boot_with_the_gate_not_met_keeps_the_column_and_every_byte(
    deployment, blocker, caplog, monkeypatch
):
    dep = deployment
    await dep.use_pre_retirement_schema()
    ids = await _seed_legacy_rows(dep)
    for table in TABLES:
        await backfill_embedding_vec(dep.db, table)
    # One saved item still has its vector only in the legacy column: either
    # never backfilled, or not a vector pgvector can hold.
    vector = (
        [0.5, float("nan"), 0.0, 0.0]
        if blocker == "unbackfillable"
        else [1.0, 0.0, 0.0, 0.0]
    )
    await dep.write_legacy("saved_items", ids["alpha"], vector)
    report = await verify_embedding_vec(dep.db, "saved_items")
    assert report.rows_missing_embedding_vec == 1
    assert report.rows_unbackfillable == (1 if blocker == "unbackfillable" else 0)
    before = {table: await dep.snapshot(table) for table in TABLES}
    if blocker == "missing":
        # The boot copies a copyable legacy vector before the retirement
        # runs (#3414). The retirement must still refuse when that copy
        # fails, and the failure must not stop the boot.
        async def copy_fails(_db, _table, **_kwargs):
            raise RuntimeError("the copy failed")

        monkeypatch.setattr(
            embedding_vec_backfill, "backfill_missing_embedding_vec", copy_fails
        )

    with caplog.at_level(logging.WARNING, logger=migrations.__name__):
        db = await dep.reboot()

    assert await db.column_exists("saved_items", "embedding")
    assert await dep.snapshot("saved_items") == before["saved_items"]
    assert "Keeping the legacy saved_items.embedding column" in caplog.text
    # Each table is judged on its own: document_chunks met the gate.
    assert not await db.column_exists("document_chunks", "embedding")
    assert [row[:2] + row[3:] for row in await dep.snapshot("document_chunks")] == [
        row[:2] + row[3:] for row in before["document_chunks"]
    ]

    if blocker == "missing":
        # The next boot whose copy succeeds retires it, the row's only
        # vector now in embedding_vec.
        monkeypatch.undo()
        db = await dep.reboot()
        assert not await db.column_exists("saved_items", "embedding")
        alpha = next(
            row for row in await dep.snapshot("saved_items") if row[0] == ids["alpha"]
        )
        assert alpha[3] == pytest.approx(vector)


async def test_absent_embedding_vec_keeps_the_legacy_column(deployment):
    dep = deployment
    db = dep.db
    await restore_legacy_embedding_column(db)
    if await db.column_exists("document_chunks", "embedding_vec"):
        await db.execute("ALTER TABLE document_chunks DROP COLUMN embedding_vec", ())
    await db.execute(
        "INSERT INTO document_chunks (file_hash, content, embedding) "
        "VALUES (?, ?, ?)",
        ("absent-doc", "alpha", _pack(OLD.vectors["alpha"])),
    )
    before = await db.fetchall(
        "SELECT chunk_id, content, embedding FROM document_chunks", ()
    )

    assert await migrate_retire_legacy_embedding_column(db, "document_chunks") is False

    assert await db.column_exists("document_chunks", "embedding")
    assert await db.fetchall(
        "SELECT chunk_id, content, embedding FROM document_chunks", ()
    ) == before


@pytest.mark.parametrize(
    "late_vector",
    [[1.0, 0.0, 0.0, 0.0], [0.5, float("nan"), 0.0, 0.0]],
    ids=["copyable", "unbackfillable"],
)
async def test_a_legacy_row_written_between_the_checks_blocks_the_drop(
    deployment, monkeypatch, late_vector, caplog
):
    # The full gate runs unlocked; the drop's transaction then rechecks, under
    # its lock, that no row holds a vector only in the legacy column. A
    # release still dual-writing can land such a row in between, copyable or
    # not; the recheck must see it.
    dep = deployment
    db = dep.db
    await dep.use_pre_retirement_schema()
    real_verify = embedding_vec_backfill.verify_embedding_vec
    calls = 0

    async def verify_then_a_concurrent_legacy_write(target, table):
        nonlocal calls
        calls += 1
        report = await real_verify(target, table)
        item = await _saved_items_store(db, None).save_item(
            item_type="excerpt", name="late", content="late"
        )
        await dep.write_legacy("saved_items", item.id, late_vector)
        return report

    monkeypatch.setattr(
        embedding_vec_backfill, "verify_embedding_vec",
        verify_then_a_concurrent_legacy_write,
    )

    with caplog.at_level(logging.WARNING, logger=migrations.__name__):
        dropped = await migrate_retire_legacy_embedding_column(db, "saved_items")

    assert dropped is False
    # The recheck under the lock is not a second full verify.
    assert calls == 1
    assert "rechecked under the drop's lock" in caplog.text
    assert await db.column_exists("saved_items", "embedding")
    assert await _legacy_values(db, "saved_items") == 1


async def test_the_locked_recheck_reads_no_vector_values(deployment, monkeypatch):
    # The lock blocks every access to the table (and startup with it) for as
    # long as it is held, so the recheck under it is one statement over the
    # columns' NULL state, never verify_embedding_vec's scan of every pair.
    dep = deployment
    db = dep.db
    await dep.use_pre_retirement_schema()
    await _seed_legacy_rows(dep)
    await backfill_embedding_vec(db, "saved_items")
    # Rows the gate does not block on: one with no vector at all, and one
    # with a vector only in embedding_vec.
    await _saved_items_store(db, None).save_item(
        item_type="excerpt", name="none", content="none"
    )
    await _saved_items_store(db, NEW).save_item(
        item_type="excerpt", name="gamma", content="gamma"
    )

    locked = False
    verify_calls_under_lock = []
    statements_under_lock = []
    real_transaction = db.transaction
    real_verify = embedding_vec_backfill.verify_embedding_vec

    @asynccontextmanager
    async def tracked_transaction(**kwargs):
        nonlocal locked
        async with real_transaction(**kwargs):
            locked = True
            try:
                yield
            finally:
                locked = False

    async def tracked_verify(target, table):
        verify_calls_under_lock.append(locked)
        return await real_verify(target, table)

    def recording(method):
        async def record(sql, params=()):
            if locked:
                statements_under_lock.append(" ".join(sql.split()))
            return await method(sql, params)

        return record

    monkeypatch.setattr(db, "transaction", tracked_transaction)
    for name in ("execute", "fetchone", "fetchall", "fetchval"):
        monkeypatch.setattr(db, name, recording(getattr(db, name)))
    monkeypatch.setattr(embedding_vec_backfill, "verify_embedding_vec", tracked_verify)

    assert await migrate_retire_legacy_embedding_column(db, "saved_items") is True

    assert verify_calls_under_lock == [False]
    table_reads = [sql for sql in statements_under_lock if "FROM saved_items" in sql]
    assert table_reads == [
        "SELECT EXISTS (SELECT 1 FROM saved_items "
        "WHERE embedding IS NOT NULL AND embedding_vec IS NULL)"
    ]
    assert statements_under_lock[-1] == "ALTER TABLE saved_items DROP COLUMN embedding"
    assert not await db.column_exists("saved_items", "embedding")


# ---------------------------------------------------------------- writers


async def test_writers_leave_a_remaining_legacy_column_null(deployment):
    dep = deployment
    db = dep.db
    await dep.use_pre_retirement_schema()

    item = await _saved_items_store(db, NEW).save_item(
        item_type="excerpt", name="alpha", content="alpha"
    )
    rag = await _rag_store(db, NEW, "chunked-doc")
    await rag.chunk_document("chunked-doc", "alpha", chunk_size=100)
    # store_precomputed_chunks replaces a file's chunks, so it gets its own.
    rag = await _rag_store(db, NEW, "precomputed-doc")
    await rag.store_precomputed_chunks(
        "precomputed-doc",
        [IndexedChunk("beta", NEW.vectors["beta"], NEW.profile_id)],
    )

    assert item.embedding == pytest.approx(NEW.vectors["alpha"])
    for table, rows in (("saved_items", 1), ("document_chunks", 2)):
        assert await _legacy_values(db, table) == 0
        report = await verify_embedding_vec(db, table)
        assert report.rows_embedding_vec_only == rows
        assert report.rows_missing_embedding_vec == 0


_WRITER_TABLES = {
    "save_item": "saved_items",
    "chunk_document": "document_chunks",
    "store_precomputed_chunks": "document_chunks",
}


@pytest.mark.parametrize("writer", sorted(_WRITER_TABLES))
async def test_first_embedded_write_creates_embedding_vec_at_its_width(
    deployment, writer
):
    # A fresh PostgreSQL database: the startup migration defers the column
    # until a legacy row shows its width, and none ever will.
    dep = deployment
    db = dep.db
    table = _WRITER_TABLES[writer]
    if await db.column_exists(table, "embedding_vec"):
        await db.execute(f"ALTER TABLE {table} DROP COLUMN embedding_vec", ())
    await restore_legacy_embedding_column(db, table)

    if writer == "save_item":
        store = _saved_items_store(db, NEW)
        gamma = await store.save_item(
            item_type="excerpt", name="gamma", content="gamma"
        )
        assert gamma.embedding == pytest.approx(NEW.vectors["gamma"])
        found = await store.search("third letter", limit=1)
        assert [r["item"]["id"] for r in found] == [gamma.id]
    else:
        rag = await _rag_store(db, NEW, "fresh-doc")
        if writer == "chunk_document":
            await rag.chunk_document("fresh-doc", "gamma", chunk_size=100)
        else:
            await rag.store_precomputed_chunks(
                "fresh-doc",
                [IndexedChunk("gamma", NEW.vectors["gamma"], NEW.profile_id)],
            )
        found = await rag._search_by_embedding("third letter", limit=1)
        assert [r["content"] for r in found] == ["gamma"]

    assert (await stored_embedding_column(db, table)).name == "embedding_vec"
    assert await _legacy_values(db, table) == 0
    if dep.backend == "postgres":
        (typmod,) = await db.fetchone(
            "SELECT atttypmod FROM pg_attribute "
            f"WHERE attrelid = to_regclass('{table}') "
            "AND attname = 'embedding_vec' AND NOT attisdropped",
            (),
        )
        assert typmod == DIM

    # The next boot finds every vector in embedding_vec and retires the
    # legacy column.
    db = await dep.reboot()
    assert not await db.column_exists(table, "embedding")


# pgvector's HNSW index takes vectors of at most this many dimensions.
_HNSW_MAX_DIMENSIONS = 2000


def _axis(width: int, index: int) -> List[float]:
    vector = [0.0] * width
    vector[index] = 1.0
    return vector


async def _write_legacy_only(db, table: str, row_id, vector, profile_id) -> None:
    """Store *vector* as an older release did while ``embedding_vec`` was absent."""
    await db.execute(
        f"UPDATE {table} SET embedding = ?, embedding_profile_id = ? "
        f"WHERE {ID_COLUMNS[table]} = ?",
        (_pack(vector), profile_id, row_id),
    )


@pytest.mark.parametrize(
    "width",
    [DIM, _HNSW_MAX_DIMENSIONS + 1],
    ids=["indexable", "too-wide-for-hnsw"],
)
async def test_legacy_vectors_a_first_write_did_not_copy_are_copied_at_the_next_boot(
    deployment, width
):
    # #3414. Legacy rows and no embedding_vec: a startup migration that
    # rolled back leaves this, as on PostgreSQL when its HNSW index refuses
    # a vector wider than 2000. The first embedded write then creates the
    # column without copying those rows, and every later boot found the
    # column present and skipped its copy. The rows stayed out of vector
    # search until an operator ran `kestrel embeddings backfill`.
    dep = deployment
    if width > _HNSW_MAX_DIMENSIONS and dep.backend != "postgres":
        pytest.skip("only PostgreSQL builds an HNSW index on embedding_vec")
    db = dep.db
    model = _Model(
        "legacy-model-03",
        {
            "alpha": _axis(width, width - 1),
            "beta": _axis(width, width - 2),
            "gamma": _axis(width, 0),
            # Means "alpha" without containing the word, so LIKE cannot find it.
            "the last axis": _axis(width, width - 1),
        },
    )
    for table in TABLES:
        if await db.column_exists(table, "embedding_vec"):
            await db.execute(f"ALTER TABLE {table} DROP COLUMN embedding_vec", ())
    await restore_legacy_embedding_column(db)

    ids = {}
    for text in ("alpha", "beta"):
        item = await _saved_items_store(db, None).save_item(
            item_type="excerpt", name=text, content=text
        )
        await _write_legacy_only(
            db, "saved_items", item.id, model.vectors[text], model.profile_id
        )
        ids[text] = item.id
    rag = await _rag_store(db, None, "legacy-doc")
    await rag.store_precomputed_chunks(
        "legacy-doc", [IndexedChunk("alpha"), IndexedChunk("beta")]
    )
    for text, chunk_id in (await _chunk_ids(db, "legacy-doc")).items():
        await _write_legacy_only(
            db, "document_chunks", chunk_id, model.vectors[text], model.profile_id
        )

    if width > _HNSW_MAX_DIMENSIONS:
        # A boot cannot keep the column it sizes from these rows.
        db = await dep.reboot()
        for table in TABLES:
            assert not await db.column_exists(table, "embedding_vec")
            assert await _legacy_values(db, table) == 2

    # The first embedded write creates embedding_vec and copies nothing.
    items = _saved_items_store(db, model)
    gamma = await items.save_item(item_type="excerpt", name="gamma", content="gamma")
    assert gamma.embedding == pytest.approx(model.vectors["gamma"])
    rag = await _rag_store(db, model, "first-write-doc")
    await rag.store_precomputed_chunks(
        "first-write-doc",
        [IndexedChunk("gamma", model.vectors["gamma"], model.profile_id)],
    )
    for table in TABLES:
        report = await verify_embedding_vec(db, table)
        assert report.embedding_vec_present
        assert report.rows_missing_embedding_vec == 2
        assert report.rows_unbackfillable == 0
    found = await items.search("the last axis", limit=3)
    assert ids["alpha"] not in [r["item"]["id"] for r in found]
    found = await rag._search_by_embedding("the last axis", limit=3)
    assert "alpha" not in [r["content"] for r in found]

    db = await dep.reboot()

    for table in TABLES:
        report = await verify_embedding_vec(db, table)
        assert report.rows_missing_embedding_vec == 0
        assert report.rows_embedding_vec_only == 3
        # Every vector copied, so the retirement dropped the legacy column.
        assert not await db.column_exists(table, "embedding")
    items = _saved_items_store(db, model)
    for text in ("alpha", "beta"):
        stored = await items.get_by_id(ids[text])
        assert stored.embedding == pytest.approx(model.vectors[text])
    found = await items.search("the last axis", limit=1)
    assert [r["item"]["id"] for r in found] == [ids["alpha"]]
    assert found[0]["score"] == pytest.approx(1.0)
    chunks = await _rag_store(db, model, "search-doc")
    found = await chunks._search_by_embedding("the last axis", limit=1)
    assert [r["content"] for r in found] == ["alpha"]
    assert found[0]["score"] == pytest.approx(1.0)


async def test_a_failed_embedding_vec_write_stores_no_vector(deployment):
    dep = deployment
    db = dep.db
    await dep.use_pre_retirement_schema()
    store = _saved_items_store(db, NEW)

    async def write_fails(item_id, _embedding, profile_id=None):
        await db.execute(
            "UPDATE saved_items SET embedding_profile_id = ? WHERE id = ?",
            (profile_id, item_id),
        )

    store._write_embedding_vec = write_fails

    alpha = await store.save_item(item_type="excerpt", name="alpha", content="alpha")

    # save_item reports the stored vector, so has_embedding is False, and no
    # legacy copy is left to backfill from.
    assert alpha.embedding is None
    report = await verify_embedding_vec(db, "saved_items")
    assert (report.total_rows, report.rows_missing_embedding_vec) == (1, 0)
    assert await _legacy_values(db, "saved_items") == 0


def _assert_buckets_partition(report: EmbeddingVecReport) -> None:
    assert (
        report.rows_with_both
        + report.rows_missing_embedding_vec
        + report.rows_embedding_vec_only
        + report.rows_without_any_embedding
    ) == report.total_rows


async def test_a_chunk_stored_without_a_vector_is_reported_and_reindex_embeds_it(
    deployment, caplog
):
    # #3415. A constitution reanchor re-indexed RAG with no embedding
    # service, leaving 47 chunks per agent with no vector in either column:
    # invisible to vector search, and in no bucket of `verify`.
    dep = deployment
    db = dep.db
    await dep.use_pre_retirement_schema()
    rag = await _rag_store(db, NEW, "embedded-doc")
    await rag.chunk_document("embedded-doc", "gamma", chunk_size=100)
    rag = await _rag_store(db, None, "unembedded-doc")
    with caplog.at_level(logging.WARNING, logger=AsyncRAGStore.__module__):
        stored = await rag.chunk_document(
            "unembedded-doc", "third letter", chunk_size=100
        )

    assert stored == 1
    assert (
        "Stored 1 of 1 chunks of unembedded-doc without an embedding "
        "(no embedding service resolved)"
    ) in caplog.text
    assert "`kestrel embeddings reindex --yes`" in caplog.text
    report = await verify_embedding_vec(db, "document_chunks")
    assert (
        report.total_rows,
        report.rows_embedding_vec_only,
        report.rows_without_any_embedding,
    ) == (2, 1, 1)
    _assert_buckets_partition(report)
    # A row with no vector has nothing to copy or to lose.
    assert legacy_embedding_retirement_gate_met(report)

    db = await dep.reboot()

    assert not await db.column_exists("document_chunks", "embedding")
    report = await verify_embedding_vec(db, "document_chunks")
    assert (
        report.total_rows,
        report.rows_embedding_vec_only,
        report.rows_without_any_embedding,
    ) == (2, 1, 1)
    _assert_buckets_partition(report)
    # "third letter" means "gamma", but has no vector to be found by.
    search = await _rag_store(db, NEW, "search-doc")
    found = await search._search_by_embedding("gamma", limit=2)
    assert [r["content"] for r in found] == ["gamma"]

    stats = await EmbeddingReindexer(
        db, NEW, NEW.profile_id, column_dim=DIM
    ).reindex_table("document_chunks")

    assert (stats.scanned, stats.reembedded, stats.failed) == (1, 1, 0)
    report = await verify_embedding_vec(db, "document_chunks")
    assert (
        report.total_rows,
        report.rows_embedding_vec_only,
        report.rows_without_any_embedding,
    ) == (2, 2, 0)
    found = await search._search_by_embedding("gamma", limit=2)
    assert sorted(r["content"] for r in found) == ["gamma", "third letter"]


# ------------------------------------------------------ gate and guards


def _report(**overrides) -> EmbeddingVecReport:
    fields = dict(
        table="saved_items",
        embedding_vec_present=True,
        rows_with_both=1,
        rows_missing_embedding_vec=0,
        rows_embedding_vec_only=2,
        rows_without_any_embedding=0,
        rows_disagreeing=1,
        rows_backfilled=0,
        rows_unbackfillable=0,
    )
    fields.update(overrides)
    # The buckets partition the table (#3415).
    fields["total_rows"] = (
        fields["rows_with_both"]
        + fields["rows_missing_embedding_vec"]
        + fields["rows_embedding_vec_only"]
        + fields["rows_without_any_embedding"]
    )
    return EmbeddingVecReport(**fields)


def test_retirement_gate_is_stricter_than_the_phase_2_gate():
    assert legacy_embedding_retirement_gate_met(_report())
    assert not legacy_embedding_retirement_gate_met(
        _report(embedding_vec_present=False, rows_embedding_vec_only=0)
    )
    assert not legacy_embedding_retirement_gate_met(
        _report(rows_missing_embedding_vec=1)
    )
    # The phase-2 gate passes this (missing == unbackfillable); dropping the
    # column would destroy the row's only vector.
    assert not legacy_embedding_retirement_gate_met(
        _report(rows_missing_embedding_vec=1, rows_unbackfillable=1)
    )
    # A row never embedded has no vector to destroy (#3415).
    assert legacy_embedding_retirement_gate_met(_report(rows_without_any_embedding=5))


async def test_old_sqlite_keeps_the_legacy_column(tmp_path, monkeypatch, caplog):
    db = await AsyncDatabase.sqlite(str(tmp_path / "old.db"))
    try:
        await restore_legacy_embedding_column(db)

        async def sqlite_3_34(_db):
            return (3, 34, 1)

        monkeypatch.setattr(migrations, "_sqlite_version", sqlite_3_34)
        with caplog.at_level(logging.WARNING, logger=migrations.__name__):
            dropped = await migrate_retire_legacy_embedding_column(db, "saved_items")

        assert dropped is False
        assert await db.column_exists("saved_items", "embedding")
        assert "SQLite 3.34.1 cannot drop a column" in caplog.text
    finally:
        await db.close()


async def test_retirement_refuses_other_tables(tmp_path):
    db = await AsyncDatabase.sqlite(str(tmp_path / "other.db"))
    try:
        with pytest.raises(ValueError, match="table must be one of"):
            await migrate_retire_legacy_embedding_column(db, "conversation_history")
    finally:
        await db.close()


class _BrokenDdl:
    """Just the surface ``ensure_embedding_vec_column`` uses; DDL fails."""

    backend_type = "sqlite"

    async def column_exists(self, table, column):
        return False

    def transaction(self):
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def cm():
            yield

        return cm()

    async def execute(self, sql, params=()):
        raise RuntimeError("disk is read-only")


async def test_ensure_embedding_vec_column_reports_a_failure_without_raising(caplog):
    with caplog.at_level(logging.WARNING):
        assert await ensure_embedding_vec_column(_BrokenDdl(), "saved_items", 4) is False
    assert "The vector is not stored" in caplog.text


@pytest.mark.parametrize("dimension", [0, -1, 1.5, True, "4"])
async def test_ensure_embedding_vec_column_refuses_a_bad_dimension(dimension):
    with pytest.raises(ValueError, match="dimension"):
        await ensure_embedding_vec_column(_BrokenDdl(), "saved_items", dimension)


async def test_ensure_embedding_vec_column_refuses_other_tables():
    with pytest.raises(ValueError, match="conversation_history"):
        await ensure_embedding_vec_column(_BrokenDdl(), "conversation_history", 4)
