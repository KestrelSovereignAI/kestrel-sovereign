"""#2958: every writer of ``conversation_history`` stamps the derived column.

``add_conversation`` is the loud write path and has its own coverage in
``test_session_id_column_migration.py``. These are the quiet ones — a salvage
marker and a backup restore — and they are the ones that would rot silently.
Both write the table with hand-spelled SQL rather than through the store's
insert helper, so neither picks the column up for free, and neither produces a
visible symptom when it stops: the row is present and correct in metadata, and
only the index is wrong.

The claim in each case is the same equality the migration asserts, so a row
these paths write is indistinguishable from a row the backfill lifted:

    session_id == column_session_id(metadata)

Insertion is not the only way ``metadata.session_id`` moves, though. Two APIs
take a caller-chosen key set and so can rewrite it afterwards, and they answer
differently: ``update_message_metadata`` carries the column along (proved
against both engines in
``tests/integration/test_session_id_column_backend_parity.py``), while
``atomic_increment_metadata_counter`` declines — the last test here.
"""

from __future__ import annotations

import json

import pytest

from kestrel_sovereign.agent.salvage import SalvageReason, salvage_messages
from kestrel_sovereign.storage import AsyncStorage
from kestrel_sovereign.storage.async_conversation_store import AsyncConversationStore
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.session_id_column import column_session_id

AGENT = "did:test:session-column-writers"
UUID_A = "3c9d6b71-45ae-4d0f-9f2b-000000000001"


async def _rows(db):
    """``(metadata, session_id)`` per row, in insertion order.

    Keyed on order rather than content because the content column is
    ciphertext whenever ``KESTREL_DATA_KEY`` is set — a ``WHERE content = ?``
    would match nothing there and pass or fail for the wrong reason.
    """
    return await db.fetchall(
        "SELECT metadata, session_id FROM conversation_history ORDER BY id", ()
    )


@pytest.mark.asyncio
async def test_a_salvage_marker_is_filed_in_the_session_it_salvaged(tmp_path):
    """The marker is live history, not bookkeeping.

    It carries ``session_id`` in its metadata precisely so cross-session reads
    do not surface another session's span (#713). A marker whose column is NULL
    would be exactly the row a session-scoped index query misses — the salvage
    that hides the messages it replaced.
    """
    db = await AsyncDatabase.sqlite(str(tmp_path / "salvage.db"))
    try:
        store = AsyncConversationStore(db, agent_id=AGENT)
        await store.add_conversation("user", "long turn", session_id=UUID_A)
        original = await db.fetchone(
            "SELECT id FROM conversation_history WHERE content IS NOT NULL "
            "ORDER BY id LIMIT 1"
        )

        result = await salvage_messages(
            conv_store=store,
            original_messages=[{"id": int(original[0])}],
            reason=SalvageReason.AUTO_PRUNE_PRETRIM,
            model="test-model",
            session_id=UUID_A,
            token_estimate=42,
        )

        metadata, session_id = await db.fetchone(
            "SELECT metadata, session_id FROM conversation_history WHERE id = ?",
            (result.salvage_id,),
        )
        assert json.loads(metadata)["session_id"] == UUID_A
        assert session_id == UUID_A
        assert session_id == column_session_id(metadata)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_a_salvage_marker_outside_the_contract_stamps_null_not_garbage(tmp_path):
    """Salvage does not get its own rule.

    ``session_id`` reaches salvage from the caller, so it can be an id the
    column may not hold. It must land NULL rather than be rewritten — the
    marker's metadata is what grouping reads, and Phase A changes no reader.
    """
    db = await AsyncDatabase.sqlite(str(tmp_path / "salvage-unstampable.db"))
    try:
        store = AsyncConversationStore(db, agent_id=AGENT)
        await store.add_conversation("user", "long turn", session_id="did:x:1")
        original = await db.fetchone(
            "SELECT id FROM conversation_history ORDER BY id LIMIT 1"
        )

        result = await salvage_messages(
            conv_store=store,
            original_messages=[{"id": int(original[0])}],
            reason=SalvageReason.AUTO_PRUNE_PRETRIM,
            model="test-model",
            session_id="did:x:1",
            token_estimate=42,
        )

        metadata, session_id = await db.fetchone(
            "SELECT metadata, session_id FROM conversation_history WHERE id = ?",
            (result.salvage_id,),
        )
        assert json.loads(metadata)["session_id"] == "did:x:1"
        assert session_id is None
    finally:
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["counter_field", "timestamp_field"])
async def test_the_counter_api_declines_the_session_key_rather_than_desyncing(
    tmp_path, field
):
    """The second door onto ``metadata.session_id``, and it says no.

    Both of this method's field names come from the caller, so it *can* be
    pointed at the indexed key — and what it would write there is a counter or
    an ISO timestamp, neither of which is a session identity. Teaching it to
    keep the column in step would make that call succeed quietly; refusing
    leaves the caller's bug where the caller can see it.

    The row must be untouched afterwards. A refusal that had already written
    half of what it was asked for would be worse than the desync.
    """
    db = await AsyncDatabase.sqlite(str(tmp_path / "counter.db"))
    try:
        store = AsyncConversationStore(db, agent_id=AGENT)
        await store.add_conversation("user", "turn", session_id=UUID_A)
        message_id = int((await db.fetchone(
            "SELECT id FROM conversation_history ORDER BY id LIMIT 1"
        ))[0])

        with pytest.raises(ValueError, match="session identity"):
            await store.atomic_increment_metadata_counter(
                message_id, **{
                    "counter_field": "access_count",
                    field: "session_id",
                }
            )

        metadata, session_id = await db.fetchone(
            "SELECT metadata, session_id FROM conversation_history WHERE id = ?",
            (message_id,),
        )
        assert json.loads(metadata)["session_id"] == UUID_A
        assert "access_count" not in json.loads(metadata)
        assert session_id == UUID_A

        # The ordinary call it exists for is unaffected.
        assert await store.atomic_increment_metadata_counter(
            message_id, "access_count", "last_accessed"
        )
        metadata, session_id = await db.fetchone(
            "SELECT metadata, session_id FROM conversation_history WHERE id = ?",
            (message_id,),
        )
        assert json.loads(metadata)["access_count"] == 1
        assert session_id == UUID_A
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_a_restore_rederives_the_column_from_the_backup_metadata(tmp_path):
    """A backup older than the column still restores with it populated.

    The restore SELECT reads ``metadata`` and never the source's ``session_id``
    — deliberately, since a backup taken before #2958 has no such column to
    read. So the column is re-derived on the way in, and a pre-column backup
    lands in exactly the shape a post-column one does.
    """
    source = AsyncStorage(str(tmp_path / "source.db"), agent_id=AGENT)
    await source.initialize()
    try:
        await source.add_conversation("user", "carried", session_id=UUID_A)
        await source.add_conversation("user", "uncarried", session_id="did:x:1")
        blob = await source.create_backup_blob()
    finally:
        await source.close()

    target = AsyncStorage(str(tmp_path / "target.db"), agent_id=AGENT)
    await target.initialize()
    try:
        stats = await target.restore_from_backup_blob(blob)
        assert stats["messages_restored"] == 2

        rows = await _rows(target.db)
        assert [row[1] for row in rows] == [UUID_A, None]
        assert [json.loads(row[0])["session_id"] for row in rows] == [
            UUID_A, "did:x:1",
        ]
        for metadata, session_id in rows:
            assert session_id == column_session_id(metadata)
    finally:
        await target.close()
