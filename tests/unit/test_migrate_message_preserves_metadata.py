"""Read-time re-encryption must preserve row metadata (#2064).

The opportunistic ``_migrate_message`` re-encrypts a row that was written
under the global key once a per-agent key becomes available. It MUST keep
the row's existing metadata (``session_id``, ``sent_form``,
``excluded_from_context``, ``privacy_mode``, …) and only add/update ``enc``
and ``key_version`` — mirroring the metadata-preserving invariant documented
for the one-shot backfill in ``security/encryption_backfill.py``.

Regression: the prior implementation built a fresh ``{enc, key_version}``
dict, silently wiping all other metadata on a default-on read path.
"""
import json
import tempfile
from pathlib import Path

import pytest

from kestrel_sovereign.storage.async_conversation_store import (
    AsyncConversationStore,
    CURRENT_KEY_VERSION,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.encryption import encrypt_string


@pytest.fixture(autouse=True)
def set_test_key(monkeypatch):
    """A master key so both global and per-agent fernets exist."""
    monkeypatch.setenv("KESTREL_DATA_KEY", "test-encryption-key-for-unit-tests")


@pytest.fixture
async def store():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        db = await AsyncDatabase.sqlite(str(db_path))
        store = AsyncConversationStore(db, agent_id="did:test:agent-2064")
        # Both keys must be present for a global->agent migration to trigger.
        assert store._global_fernet is not None
        assert store._agent_fernet is not None
        yield store
        await db.close()


async def _insert_global_key_row(store, *, role, content, meta):
    """Insert a row encrypted under the GLOBAL key (key_version 0)."""
    enc_content, was_encrypted = encrypt_string(content, store._global_fernet)
    assert was_encrypted
    row_meta = {**meta, "enc": True, "key_version": 0}
    await store.db.execute_commit(
        "INSERT INTO conversation_history "
        "(agent_id, role, content, rendered_content, metadata, created_at) "
        "VALUES (?, ?, ?, NULL, ?, datetime('now'))",
        (store.agent_id, role, enc_content, json.dumps(row_meta)),
    )
    row = await store.db.fetchone(
        "SELECT id FROM conversation_history WHERE agent_id = ? ORDER BY id DESC LIMIT 1",
        (store.agent_id,),
    )
    return row[0]


async def _raw_meta(store, row_id):
    row = await store.db.fetchone(
        "SELECT metadata FROM conversation_history WHERE id = ?", (row_id,)
    )
    return json.loads(row[0]) if row[0] else None


@pytest.mark.asyncio
async def test_migration_preserves_metadata_fields(store):
    """A global-key row carrying rich metadata keeps every field after
    the read-time migration; only enc/key_version are updated."""
    row_id = await _insert_global_key_row(
        store,
        role="user",
        content="hello sovereign",
        meta={
            "session_id": "sess-2064",
            "sent_form": True,
            "privacy_mode": "NORMAL",
            "custom_field": "keep-me",
        },
    )

    # Trigger the opportunistic migration via a normal read.
    history = await store.get_conversation_history(limit=10)
    assert any(r["id"] == row_id for r in history)

    meta = await _raw_meta(store, row_id)
    # Pre-existing fields survive.
    assert meta["session_id"] == "sess-2064"
    assert meta["sent_form"] is True
    assert meta["privacy_mode"] == "NORMAL"
    assert meta["custom_field"] == "keep-me"
    # enc/key_version are added/updated by the migration.
    assert meta["enc"] is True
    assert meta["key_version"] == CURRENT_KEY_VERSION


@pytest.mark.asyncio
async def test_migrated_excluded_row_stays_excluded(store):
    """An ``excluded_from_context: true`` row that gets migrated must keep
    the flag and remain excluded from ``get_conversation_history``."""
    row_id = await _insert_global_key_row(
        store,
        role="assistant",
        content="compacted summary content",
        meta={"session_id": "sess-2064", "excluded_from_context": True},
    )

    # get_conversation_history skips excluded rows before migration, so
    # migrate directly with the row's metadata in scope (the call-site
    # contract), then confirm the flag survives.
    await store._migrate_message(row_id, "compacted summary content",
                                 await _raw_meta(store, row_id))

    meta = await _raw_meta(store, row_id)
    assert meta["excluded_from_context"] is True
    assert meta["session_id"] == "sess-2064"
    assert meta["key_version"] == CURRENT_KEY_VERSION

    history = await store.get_conversation_history(limit=10)
    assert all(r["id"] != row_id for r in history)


@pytest.mark.asyncio
async def test_migration_with_no_metadata_still_sets_enc(store):
    """A row with no extra metadata still gets enc/key_version stamped
    (meta=None must not raise)."""
    row_id = await _insert_global_key_row(
        store, role="user", content="bare row", meta={},
    )
    await store.get_conversation_history(limit=10)

    meta = await _raw_meta(store, row_id)
    assert meta["enc"] is True
    assert meta["key_version"] == CURRENT_KEY_VERSION
