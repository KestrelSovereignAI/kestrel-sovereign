"""Characterization seams for saved-item content encryption issue #2677."""

import pytest

from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.saved_items_store import SavedItemsStore


@pytest.mark.asyncio
async def test_current_writer_persists_saved_item_content_as_plaintext(tmp_path):
    """Pin the pre-encryption behavior that Child C must intentionally replace."""
    db_path = tmp_path / "saved-items.db"
    db = await AsyncDatabase.sqlite(str(db_path))
    sentinel = "saved-item-plaintext-sentinel-2677"
    try:
        store = SavedItemsStore(
            db,
            agent_id="did:test:plaintext-characterization",
        )
        item = await store.save_item(
            item_type="stash",
            name="Encryption characterization",
            content=sentinel,
            compute_embedding=False,
            deduplicate=False,
        )

        raw_row = await db.fetchone(
            "SELECT content FROM saved_items WHERE id = ?",
            (item.id,),
        )
        columns = await db.fetchall("PRAGMA table_info(saved_items)")

        assert raw_row == (sentinel,)
        assert "content_ciphertext" not in {column[1] for column in columns}
    finally:
        await db.close()

    # This is an at-rest characterization, not merely an assertion against the
    # store's returned object: after SQLite closes/checkpoints, the sentinel is
    # visible in the database file to an offline reader.
    assert sentinel.encode("utf-8") in db_path.read_bytes()
