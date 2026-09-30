"""The embedding columns as the real stores write them (#3402, #3411).

Since #3411 ``SavedItemsStore.save_item`` writes the vector to
``embedding_vec`` only. These cases drive that real writer on a freshly
booted database, whose legacy ``embedding`` column is already retired, and
on one that still carries it, and check that the verify/backfill helper
agrees with it and repairs a legacy-only row an older release left behind.

The PostgreSQL path of the helper is covered by
``tests/unit/test_embedding_vec_backfill.py``, which the CI unit
tier runs against PostgreSQL.
"""

from __future__ import annotations

import struct

import pytest

from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.embedding_vec_backfill import (
    backfill_embedding_vec,
    verify_embedding_vec,
)
from kestrel_sovereign.storage.saved_items_store import SavedItemsStore
from tests.utils.legacy_embedding_column import restore_legacy_embedding_column

_VECTORS = {
    "apples": [1.0, 0.0, 0.0],
    "zebras": [0.0, 1.0, 0.0],
}


class _FixedEmbeddingService:
    """Deterministic embeddings keyed by the text's first word."""

    async def aembed(self, text):
        return _VECTORS[text.split()[0]]

    async def aembed_query(self, text, instruction=None):
        return _VECTORS[text.split()[0]]


@pytest.fixture
async def store(tmp_path):
    db = await AsyncDatabase.sqlite(str(tmp_path / "vector-storage.db"))
    saved = SavedItemsStore(db, agent_id="did:test:vector-storage")
    saved._get_embedding_service = lambda: _FixedEmbeddingService()
    try:
        yield saved
    finally:
        await db.close()


async def _save_all(store):
    return {
        word: await store.save_item(
            item_type="stash", name=word, content=f"{word} notes"
        )
        for word in _VECTORS
    }


async def test_writes_land_in_embedding_vec_on_a_retired_table(store):
    assert not await store.db.column_exists("saved_items", "embedding")

    items = await _save_all(store)

    report = await verify_embedding_vec(store.db, "saved_items")
    assert report.total_rows == 2
    assert report.rows_embedding_vec_only == 2
    assert report.rows_missing_embedding_vec == 0
    for word, item in items.items():
        assert item.embedding == _VECTORS[word]
        assert (await store.get_by_id(item.id)).embedding == _VECTORS[word]


async def test_the_legacy_column_is_left_null_while_it_still_exists(store):
    await restore_legacy_embedding_column(store.db, "saved_items")

    await _save_all(store)

    report = await verify_embedding_vec(store.db, "saved_items")
    assert report.total_rows == 2
    assert report.rows_embedding_vec_only == 2
    assert report.rows_with_both == 0
    assert report.rows_missing_embedding_vec == 0
    (legacy_values,) = await store.db.fetchone(
        "SELECT COUNT(*) FROM saved_items WHERE embedding IS NOT NULL", ()
    )
    assert legacy_values == 0


async def test_backfill_repairs_a_legacy_only_row_from_an_older_release(store):
    await restore_legacy_embedding_column(store.db, "saved_items")
    items = await _save_all(store)
    apples = items["apples"].id
    # What an older release's dual-write left when its embedding_vec UPDATE
    # failed: the vector in the legacy column only.
    legacy = struct.pack("<3f", *_VECTORS["apples"])
    await store.db.execute(
        "UPDATE saved_items SET embedding = ?, embedding_vec = NULL WHERE id = ?",
        (legacy, apples),
    )

    before = await verify_embedding_vec(store.db, "saved_items")
    assert before.rows_missing_embedding_vec == 1
    # Readers take embedding_vec only (#3409); the row has no vector yet.
    assert (await store.get_by_id(apples)).embedding is None

    repaired = await backfill_embedding_vec(store.db, "saved_items")
    again = await backfill_embedding_vec(store.db, "saved_items")

    assert repaired.rows_backfilled == 1
    assert repaired.rows_missing_embedding_vec == 0
    assert repaired.rows_with_both == 1
    assert repaired.rows_disagreeing == 0
    assert again.rows_backfilled == 0
    (vec,) = await store.db.fetchone(
        "SELECT embedding_vec FROM saved_items WHERE id = ?", (apples,)
    )
    assert bytes(vec) == legacy
    assert (await store.get_by_id(apples)).embedding == _VECTORS["apples"]
