"""Stored-embedding readers use ``embedding_vec`` (#3409, parent #2684).

``kestrel embeddings reindex`` rewrites ``embedding_vec`` and
``embedding_profile_id`` but never the legacy ``embedding`` column, so after a
reindex the two disagree. Every raw-SQL reader must hydrate, score and count
``embedding_vec``. The legacy column is read only while a table has no
``embedding_vec`` column at all: a fresh PostgreSQL database, before the
startup migration sizes ``vector(N)`` from a legacy row.

Every case runs on SQLite and on PostgreSQL. Both legs start with the column
absent and add it the way a deployment gets it: the startup migration creates
the column, then ``kestrel embeddings backfill`` copies the legacy vectors.
The reindexed rows are produced by the real ``EmbeddingReindexer``.

Since #3411 no store writes the legacy column, so legacy vectors are seeded
the way an older release's dual-write left them (:meth:`write_legacy`).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from kestrel_sovereign.features.save.feature import SaveFeature
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.async_rag_store import AsyncRAGStore, IndexedChunk
from kestrel_sovereign.storage.embedding_column import (
    decode_stored_embedding,
    stored_embedding_column,
)
from kestrel_sovereign.storage.embedding_reindex import EmbeddingReindexer
from kestrel_sovereign.storage.embedding_vec_backfill import (
    backfill_embedding_vec,
    verify_embedding_vec,
)
from kestrel_sovereign.storage.saved_items_store import SavedItem, SavedItemsStore
from tests.utils.legacy_embedding_column import restore_legacy_embedding_column
from tests.utils.postgres_schema import (
    disposable_postgres_schema,
    pgvector_schema,
    postgres_test_url,
    quoted_search_path,
    with_search_path,
)

AGENT = "did:web:test.invalid:embedding-readers"
DIM = 4
TABLES = ("saved_items", "document_chunks")


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


# The two models place "alpha" and "beta" in swapped coordinates. A query for
# "alpha" under the new model therefore ranks "alpha" first when scored
# against ``embedding_vec`` and "beta" first when scored against the legacy
# bytes the old model left behind.
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
        # A query that means "gamma" without containing the word, so the
        # text (LIKE) fallback cannot find it.
        "third letter": [0.0, 1.0, 0.0, 0.0],
    },
)


@dataclass
class _Deployment:
    db: AsyncDatabase
    # DDL type the startup migration gives ``embedding_vec`` on this backend.
    vector_type: str

    async def create_embedding_vec(self, table: str) -> None:
        """Add the column as the startup migration does, then pass the gate."""
        await self.db.execute(
            f"ALTER TABLE {table} ADD COLUMN embedding_vec {self.vector_type}", ()
        )
        report = await backfill_embedding_vec(self.db, table)
        assert report.embedding_vec_present
        assert report.rows_missing_embedding_vec == report.rows_unbackfillable == 0

    async def write_legacy(self, table: str, row_id, model: _Model, text: str) -> None:
        """Give a row the vector an older release's dual-write left behind.

        Before #3411 the insert wrote the legacy column, and while
        ``embedding_vec`` was absent the write fell back to stamping only the
        profile id beside it.
        """
        id_col = "id" if table == "saved_items" else "chunk_id"
        vector = model.vectors[text]
        await self.db.execute(
            f"UPDATE {table} SET embedding = ?, embedding_profile_id = ? "
            f"WHERE {id_col} = ?",
            (struct.pack(f"<{len(vector)}f", *vector), model.profile_id, row_id),
        )

    async def reindex(self, table: str, model: _Model) -> None:
        reindexer = EmbeddingReindexer(
            self.db, model, model.profile_id, column_dim=DIM
        )
        stats = await reindexer.reindex_table(table)
        assert stats.failed == 0


async def _leave_schema_unchanged(db):
    """Schema initializer for the connection that only manages schemas."""


@pytest.fixture(params=["sqlite", "postgres"])
async def deployment(request, tmp_path):
    if request.param == "sqlite":
        db = await AsyncDatabase.sqlite(str(tmp_path / "readers.db"))
        try:
            # SQLite's startup migration always adds the column, and then
            # retires the legacy one (#3411). Reverse both so this leg starts
            # where a PostgreSQL database an older release wrote does.
            await restore_legacy_embedding_column(db)
            for table in TABLES:
                await db.execute(
                    f"ALTER TABLE {table} DROP COLUMN embedding_vec", ()
                )
            yield _Deployment(db, "BLOB")
        finally:
            await db.close()
        return

    url = postgres_test_url()
    if not url:
        pytest.skip("TEST_POSTGRES_URL is not set")
    admin = await AsyncDatabase.postgres(
        url, schema_initializer=_leave_schema_unchanged
    )
    try:
        vector_schema = await pgvector_schema(admin)
        async with disposable_postgres_schema(admin, "embedding_readers") as schema:
            # The core schema boots into a schema of this test's own, where
            # the startup migration leaves ``embedding_vec`` absent (no legacy
            # row shows its width yet), and adding it is invisible to every
            # other test. pgvector's schema follows on the path for the
            # stores' ``::vector`` casts; the DDL below qualifies it (#3401).
            db = await AsyncDatabase.postgres(
                with_search_path(url, quoted_search_path(schema, vector_schema))
            )
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
                yield _Deployment(db, f'"{vector_schema}".vector({DIM})')
            finally:
                await db.close()
    finally:
        await admin.close()


def _saved_items_store(db, model: Optional[_Model]) -> SavedItemsStore:
    store = SavedItemsStore(db, agent_id=AGENT)
    store._get_embedding_service = lambda: model
    # Route ``search`` through the in-Python reader under test rather than the
    # SQLAlchemy vector backend, which already ranks by ``embedding_vec``.
    store._sqla_factory_unavailable = True
    return store


def _save_feature(store: SavedItemsStore) -> SaveFeature:
    feature = SaveFeature(agent=MagicMock())
    feature._db = store.db
    feature._saved_items_store = store
    feature.agent_id = AGENT
    return feature


async def _rag_store(db, model: Optional[_Model]) -> tuple:
    file_hash = f"readers-{model.profile_id if model else 'none'}"
    await db.execute(
        "INSERT INTO files (content_hash, original_name) VALUES (?, ?)",
        (file_hash, "readers.md"),
    )
    await db.execute(
        "INSERT INTO file_owners (content_hash, agent_id, original_name) "
        "VALUES (?, ?, ?)",
        (file_hash, AGENT, "readers.md"),
    )
    store = AsyncRAGStore(db, agent_id=AGENT)
    store._get_embedding_service = lambda: model
    return store, file_hash


async def _save_legacy(deployment: _Deployment, text: str, model: _Model) -> SavedItem:
    """Save *text* as an older release did, its vector in the legacy column."""
    item = await _saved_items_store(deployment.db, None).save_item(
        item_type="excerpt", name=text, content=text
    )
    await deployment.write_legacy("saved_items", item.id, model, text)
    return item


# --------------------------------------------------------------- saved_items


async def test_saved_item_search_scores_embedding_vec_after_reindex(deployment):
    db = deployment.db
    alpha = await _save_legacy(deployment, "alpha", OLD)
    beta = await _save_legacy(deployment, "beta", OLD)
    await deployment.create_embedding_vec("saved_items")
    await deployment.reindex("saved_items", NEW)
    store = _saved_items_store(db, NEW)

    # Reindex rewrote embedding_vec and left the old model's legacy bytes.
    assert (await verify_embedding_vec(db, "saved_items")).rows_disagreeing == 2

    results = await store.search("alpha", limit=2)

    assert [r["item"]["id"] for r in results] == [alpha.id, beta.id]
    assert [r["score"] for r in results] == [pytest.approx(1.0), pytest.approx(0.0)]


async def test_saved_item_hydration_and_stats_read_embedding_vec(deployment):
    db = deployment.db
    store = _saved_items_store(db, OLD)
    alpha = await _save_legacy(deployment, "alpha", OLD)
    await deployment.create_embedding_vec("saved_items")
    await deployment.reindex("saved_items", NEW)

    for item in (
        await store.get_by_id(alpha.id),
        await store.get_by_content_hash(alpha.content_hash),
        (await store.list_by_content_hash(alpha.content_hash))[0],
        (await store.list_items())[0],
        (await store.list_items(item_type="excerpt"))[0],
    ):
        assert item.embedding == pytest.approx(NEW.vectors["alpha"])

    # A row that only reindex embedded has embedding_vec and no legacy value.
    unembedded = _saved_items_store(db, None)
    gamma = await unembedded.save_item(item_type="excerpt", name="gamma", content="gamma")
    await deployment.reindex("saved_items", NEW)
    assert (await verify_embedding_vec(db, "saved_items")).rows_embedding_vec_only == 1

    assert (await store.get_by_id(gamma.id)).embedding == pytest.approx(
        NEW.vectors["gamma"]
    )
    assert (await store.get_stats())["with_embedding"] == 2


async def test_has_embedding_is_true_for_a_row_with_only_embedding_vec(deployment):
    db = deployment.db
    await deployment.create_embedding_vec("saved_items")
    store = _saved_items_store(db, None)
    feature = _save_feature(store)

    first = await feature.save_item(name="gamma", content="gamma")
    assert first.data["has_embedding"] is False

    await deployment.reindex("saved_items", NEW)
    assert (await verify_embedding_vec(db, "saved_items")).rows_embedding_vec_only == 1

    # Saving the same record again returns the existing row.
    again = await feature.save_item(name="gamma", content="gamma")
    assert again.data["saved_item_id"] == first.data["saved_item_id"]
    assert again.data["has_embedding"] is True


async def test_saved_item_search_finds_a_row_with_only_embedding_vec(deployment):
    db = deployment.db
    await deployment.create_embedding_vec("saved_items")
    gamma = await _saved_items_store(db, None).save_item(
        item_type="excerpt", name="gamma", content="gamma"
    )
    await deployment.reindex("saved_items", NEW)
    assert (await verify_embedding_vec(db, "saved_items")).rows_embedding_vec_only == 1

    results = await _saved_items_store(db, NEW).search("third letter", limit=5)

    assert [(r["item"]["id"], r["score"]) for r in results] == [
        (gamma.id, pytest.approx(1.0))
    ]


async def test_legacy_only_saved_item_has_no_vector_once_embedding_vec_exists(
    deployment,
):
    db = deployment.db
    await deployment.create_embedding_vec("saved_items")
    store = _saved_items_store(db, OLD)
    # An older release's dual-write whose embedding_vec UPDATE failed.
    alpha = await _save_legacy(deployment, "alpha", OLD)

    assert (await verify_embedding_vec(db, "saved_items")).rows_missing_embedding_vec == 1
    assert (await store.get_by_id(alpha.id)).embedding is None
    assert (await store.get_stats())["with_embedding"] == 0
    # "beta" does not match the row's text, so the LIKE fallback finds
    # nothing. Scoring the legacy bytes would return the row.
    assert await store.search("beta", limit=5) == []


async def test_saved_item_readers_use_legacy_column_while_embedding_vec_is_absent(
    deployment,
):
    # A PostgreSQL database an older release wrote before its first
    # embedding_vec existed holds its vectors only in the legacy column until
    # the next boot creates embedding_vec.
    db = deployment.db
    store = _saved_items_store(db, OLD)

    alpha = await _save_legacy(deployment, "alpha", OLD)
    beta = await _save_legacy(deployment, "beta", OLD)

    assert (await store.get_by_id(alpha.id)).embedding == pytest.approx(
        OLD.vectors["alpha"]
    )
    assert (await store.get_by_id(beta.id)).embedding == pytest.approx(
        OLD.vectors["beta"]
    )
    assert (await store.get_stats())["with_embedding"] == 2
    results = await store.search("alpha", limit=2)
    assert [r["item"]["id"] for r in results] == [alpha.id, beta.id]
    assert results[0]["score"] == pytest.approx(1.0)


# ------------------------------------------------------------ document_chunks


async def _seed_chunks(
    deployment: _Deployment, store: AsyncRAGStore, file_hash: str, model: _Model
) -> None:
    """Store "alpha" and "beta" as an older release did (legacy vectors)."""
    written = await store.store_precomputed_chunks(
        file_hash, [IndexedChunk(text) for text in ("alpha", "beta")]
    )
    assert written == 2
    rows = await deployment.db.fetchall(
        "SELECT chunk_id, content FROM document_chunks WHERE file_hash = ?",
        (file_hash,),
    )
    for chunk_id, text in rows:
        await deployment.write_legacy("document_chunks", chunk_id, model, text)


async def test_rag_search_scores_embedding_vec_after_reindex(deployment):
    db = deployment.db
    store, file_hash = await _rag_store(db, OLD)
    await _seed_chunks(deployment, store, file_hash, OLD)
    await deployment.create_embedding_vec("document_chunks")
    await deployment.reindex("document_chunks", NEW)
    store._get_embedding_service = lambda: NEW

    assert (await verify_embedding_vec(db, "document_chunks")).rows_disagreeing == 2

    results = await store._search_by_embedding("alpha", limit=2)

    assert [r["content"] for r in results] == ["alpha", "beta"]
    assert [r["score"] for r in results] == [pytest.approx(1.0), pytest.approx(0.0)]


async def test_read_indexed_chunks_pairs_the_profile_with_embedding_vec(deployment):
    db = deployment.db
    store, file_hash = await _rag_store(db, OLD)
    await _seed_chunks(deployment, store, file_hash, OLD)
    await deployment.create_embedding_vec("document_chunks")
    await deployment.reindex("document_chunks", NEW)

    chunks = await store.read_indexed_chunks(file_hash)

    assert [c.content for c in chunks] == ["alpha", "beta"]
    assert [c.profile_id for c in chunks] == [NEW.profile_id] * 2
    assert chunks[0].embedding == pytest.approx(NEW.vectors["alpha"])
    assert chunks[1].embedding == pytest.approx(NEW.vectors["beta"])


async def test_rag_search_finds_a_chunk_with_only_embedding_vec(deployment):
    db = deployment.db
    store, file_hash = await _rag_store(db, NEW)
    await deployment.create_embedding_vec("document_chunks")
    assert await store.store_precomputed_chunks(file_hash, [IndexedChunk("gamma")]) == 1
    await deployment.reindex("document_chunks", NEW)
    assert (
        await verify_embedding_vec(db, "document_chunks")
    ).rows_embedding_vec_only == 1

    results = await store._search_by_embedding("third letter", limit=5)

    assert [(r["content"], r["score"]) for r in results] == [
        ("gamma", pytest.approx(1.0))
    ]


async def test_legacy_only_chunk_has_no_vector_once_embedding_vec_exists(deployment):
    db = deployment.db
    store, file_hash = await _rag_store(db, OLD)
    await deployment.create_embedding_vec("document_chunks")
    # An older release's dual-write whose embedding_vec UPDATE failed.
    await _seed_chunks(deployment, store, file_hash, OLD)

    assert (
        await verify_embedding_vec(db, "document_chunks")
    ).rows_missing_embedding_vec == 2
    assert [c.embedding for c in await store.read_indexed_chunks(file_hash)] == [[], []]
    assert await store._search_by_embedding("alpha", limit=2) == []


async def test_rag_readers_use_legacy_column_while_embedding_vec_is_absent(
    deployment,
):
    db = deployment.db
    store, file_hash = await _rag_store(db, OLD)
    await _seed_chunks(deployment, store, file_hash, OLD)

    chunks = await store.read_indexed_chunks(file_hash)
    assert chunks[0].embedding == pytest.approx(OLD.vectors["alpha"])
    assert chunks[1].embedding == pytest.approx(OLD.vectors["beta"])

    results = await store._search_by_embedding("alpha", limit=2)
    assert [r["content"] for r in results] == ["alpha", "beta"]
    assert results[0]["score"] == pytest.approx(1.0)


# ------------------------------------------------------------- column + decode


class _Catalog:
    """Just the catalog surface ``stored_embedding_column`` reads."""

    def __init__(self, backend_type: str, has_embedding_vec: bool):
        self.backend_type = backend_type
        self._has_embedding_vec = has_embedding_vec

    async def column_exists(self, table: str, column: str) -> bool:
        return column == "embedding_vec" and self._has_embedding_vec


@pytest.mark.parametrize(
    "backend_type, present, expected",
    [
        ("postgres", True, ("embedding_vec", "embedding_vec::text")),
        ("sqlite", True, ("embedding_vec", "embedding_vec")),
        ("postgres", False, ("embedding", "embedding")),
        ("sqlite", False, ("embedding", "embedding")),
    ],
)
async def test_stored_embedding_column_prefers_embedding_vec(
    backend_type, present, expected
):
    column = await stored_embedding_column(
        _Catalog(backend_type, present), "document_chunks"
    )

    assert (column.name, column.select) == expected


async def test_stored_embedding_column_refuses_other_tables():
    with pytest.raises(ValueError, match="conversation_history"):
        await stored_embedding_column(
            _Catalog("sqlite", True), "conversation_history"
        )


def test_decode_stored_embedding_reads_both_backend_shapes():
    # SQLite and the legacy PostgreSQL BYTEA: float32 little-endian bytes.
    assert decode_stored_embedding(struct.pack("<2f", 0.5, -1.0)) == [0.5, -1.0]
    assert decode_stored_embedding(memoryview(struct.pack("<f", 2.0))) == [2.0]
    # ``embedding_vec::text`` on PostgreSQL.
    assert decode_stored_embedding("[0.5,-1,1e-07]") == pytest.approx(
        [0.5, -1.0, 1e-07]
    )
    for empty in (None, b"", "[]"):
        assert decode_stored_embedding(empty) is None


def test_decode_stored_embedding_refuses_unknown_shapes():
    with pytest.raises(ValueError, match="pgvector"):
        decode_stored_embedding("0.5,-1")
    with pytest.raises(TypeError, match="list"):
        decode_stored_embedding([0.5, -1.0])
