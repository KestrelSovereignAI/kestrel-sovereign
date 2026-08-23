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

import io
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
async def test_a_restore_writes_one_timestamp_spelling_whatever_the_backup_held(
    tmp_path,
):
    """The restore reads a FILE, so the column's CHECK cannot police it (#3009).

    Every other path into ``conversation_history.created_at`` writes through
    ``AsyncDatabase``, where the constraint applies. This one opens the backup's
    SQLite file directly with ``aiosqlite`` and copies rows out of it, so it is
    handed whatever text an older kestrel, an import, or a hand-edited row put
    there — and it used to pass that straight through on SQLite, because
    converting was only ever done to satisfy asyncpg.

    The backup is therefore built by hand from the pre-#3009 table shape, which
    is what a backup taken before this change actually contains. Producing it
    by rewriting a live database is no longer possible, and would not be the
    same thing if it were.

    The spellings are ones ``julianday`` and the parser DISAGREE about — a
    ``T``, a ``Z``, an offset — plus one nothing can date at all. The first
    three must come back in the single form the readers date without any
    fallback; the last must take a neighbour's stamp, be counted, and not
    arrive as text the CHECK would refuse.
    """
    import tarfile

    from kestrel_sovereign.storage.session_grouping import coerce_session_timestamp
    from tests.utils.legacy_conversation_history import write_legacy_history

    backup_db = str(tmp_path / "kestrel.db")
    write_legacy_history(
        backup_db,
        [
            (AGENT, "user", "one", None, None, "2026-01-02T03:04:05"),
            (AGENT, "user", "two", None, None, "2026-01-02T05:04:05Z"),
            (AGENT, "user", "three", None, None, "2026-01-02 07:04:05+01:00"),
            (AGENT, "user", "four", None, None, "an older kestrel's idea"),
        ],
    )
    blob = AsyncStorage._tar_gzip_paths([(backup_db, "kestrel.db")])
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        assert tar.getnames() == ["kestrel.db"], (
            "the hand-built blob does not have the shape the restore looks for"
        )

    target = AsyncStorage(str(tmp_path / "odd-target.db"), agent_id=AGENT)
    await target.initialize()
    try:
        stats = await target.restore_from_backup_blob(blob)
        assert stats["messages_restored"] == 4
        assert stats.get("messages_with_unreadable_created_at", 0) == 1, (
            "the row nothing could date was not reported to the caller"
        )

        restored = [
            row[0] for row in await target.db.fetchall(
                "SELECT created_at FROM conversation_history WHERE agent_id = ? "
                "ORDER BY id", (AGENT,),
            )
        ]
        for value in restored:
            assert coerce_session_timestamp(value) is not None, (
                f"restored created_at {value!r} is a value no reader can date"
            )
            # One spelling, and it is the one CURRENT_TIMESTAMP produces.
            assert len(value) == 19 and value[10] == " ", (
                f"restored created_at {value!r} is not the canonical form"
            )
        # The offset is APPLIED, not discarded: 07:04:05+01:00 is 06:04:05 UTC.
        assert "2026-01-02 06:04:05" in restored, restored
        # ...and it lands in the right PLACE, which is #3049. New ids are
        # assigned in the order this SELECT returns and
        # `get_conversation_history()` sorts by id, so that ordering IS the
        # restored transcript's reading order. Ordering by the source's raw
        # text put the 07:04+01:00 row first — a space sorts before `T` — so
        # the transcript came back 06:04, 03:04, 05:04. Normalising the stamp
        # before ordering is not a new decision about what a restore means: it
        # is the order `canonical_order` already defines, applied to a file
        # that has earned none of the guarantees the live column's CHECK gives.
        assert restored == sorted(restored), (
            f"the restored transcript is not in chronological order: {restored}"
        )
        # The undatable row sorts FIRST and takes its neighbour's stamp, which
        # is the same answer `canonical_order` gives — undatable means
        # earliest, always — rather than whatever its raw text happened to do.
        assert restored[0] == "2026-01-02 03:04:05", restored
    finally:
        await target.close()


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
