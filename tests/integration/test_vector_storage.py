"""The two embedding columns as the real stores write them (#3402).

``SavedItemsStore.save_item`` inserts the legacy ``embedding`` value and
then dual-writes ``embedding_vec`` in a separate, non-fatal UPDATE. These
cases drive that real writer, then check that the verify/backfill helper
agrees with it and repairs a row whose dual-write was lost.

The PostgreSQL path of the helper is covered by
``tests/unit/test_embedding_vec_backfill.py``, which the CI unit
tier runs against PostgreSQL.
"""

from __future__ import annotations

import pytest

from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.embedding_vec_backfill import (
    backfill_embedding_vec,
    verify_embedding_vec,
)
from kestrel_sovereign.storage.saved_items_store import SavedItemsStore

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


async def test_dual_write_leaves_both_columns_in_agreement(store):
    for word in _VECTORS:
        await store.save_item(item_type="stash", name=word, content=f"{word} notes")

    report = await verify_embedding_vec(store.db, "saved_items")

    assert report.total_rows == 2
    assert report.rows_with_both == 2
    assert report.rows_missing_embedding_vec == 0
    assert report.rows_disagreeing == 0


async def test_backfill_repairs_a_lost_dual_write(store):
    items = {}
    for word in _VECTORS:
        items[word] = await store.save_item(
            item_type="stash", name=word, content=f"{word} notes"
        )
    # The dual-write is a separate UPDATE whose failure is only logged.
    await store.db.execute(
        "UPDATE saved_items SET embedding_vec = NULL WHERE id = ?",
        (items["apples"].id,),
    )

    before = await verify_embedding_vec(store.db, "saved_items")
    repaired = await backfill_embedding_vec(store.db, "saved_items")
    again = await backfill_embedding_vec(store.db, "saved_items")

    assert before.rows_missing_embedding_vec == 1
    assert repaired.rows_backfilled == 1
    assert repaired.rows_missing_embedding_vec == 0
    assert repaired.rows_with_both == 2
    assert repaired.rows_disagreeing == 0
    assert again.rows_backfilled == 0
    legacy, vec = await store.db.fetchone(
        "SELECT embedding, embedding_vec FROM saved_items WHERE id = ?",
        (items["apples"].id,),
    )
    assert bytes(vec) == bytes(legacy)
    assert (await store.get_by_id(items["apples"].id)).embedding == _VECTORS["apples"]
