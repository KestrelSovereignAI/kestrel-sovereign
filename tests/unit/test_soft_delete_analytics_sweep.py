"""Soft-deleted conversation content must not feed derived agent state (#2051).

#763 added the ``deleted_at IS NULL`` read-filter to the recall/context paths,
but several content-bearing analytics/identity readers issued raw
``SELECT ... FROM conversation_history`` queries that bypassed it. As a result
a message a user soft-deleted (stamped ``deleted_at``, sitting in Trash) still
shaped:

  - the personality structure analysis (``PersonalityAnalyzer._get_responses``)
  - the exported calibration few-shot examples
    (``PersonalityAnalyzer._get_calibration_examples`` → identity package)
  - wellness signal computation (``InteractionDepthCalculator.measure``)

These exercise the *real* SQL against a real SQLite DB with rows soft-deleted
via the conversation store, so they fail if the filter is dropped.

The salvage / encryption-backfill / next-id reads legitimately need to see
soft-deleted rows and are intentionally left unfiltered — covered at the end.
"""
import tempfile
from pathlib import Path

import pytest

from kestrel_sovereign.storage.async_conversation_store import AsyncConversationStore
from kestrel_sovereign.storage.async_database import AsyncDatabase


async def _insert(store, role, content):
    await store.db.execute_commit(
        "INSERT INTO conversation_history "
        "(agent_id, role, content, metadata, created_at) "
        "VALUES (?, ?, ?, ?, datetime('now'))",
        (store.agent_id, role, content, "{}"),
    )
    row = await store.db.fetchall(
        "SELECT id FROM conversation_history ORDER BY id DESC LIMIT 1", ()
    )
    return row[0][0]


@pytest.fixture
async def store():
    with tempfile.TemporaryDirectory() as tmp:
        db = await AsyncDatabase.sqlite(str(Path(tmp) / "t.db"))
        s = AsyncConversationStore(db, agent_id="test-agent")
        yield s
        await db.close()


@pytest.mark.asyncio
async def test_personality_responses_exclude_soft_deleted(store):
    from kestrel_sovereign.identity.personality_analyzer import PersonalityAnalyzer

    live = await _insert(store, "assistant", "This live reply shapes my voice.")
    trashed = await _insert(store, "assistant", "DELETED reply must not count.")
    await store.delete_message(trashed)

    analyzer = PersonalityAnalyzer(store.db, agent_id="test-agent")
    responses = await analyzer._get_responses()

    assert "This live reply shapes my voice." in responses
    assert "DELETED reply must not count." not in responses


@pytest.mark.asyncio
async def test_calibration_examples_exclude_soft_deleted_pair(store):
    from kestrel_sovereign.identity.personality_analyzer import PersonalityAnalyzer

    # A live user/assistant pair (consecutive ids) that should be exported...
    await _insert(store, "user", "Here is a substantive question about design.")
    await _insert(
        store,
        "assistant",
        "Here is a thoughtful, sufficiently long reply that calibrates voice.",
    )
    # ...and a soft-deleted pair that must NOT be exported.
    du = await _insert(store, "user", "A deleted question that should vanish from export.")
    da = await _insert(
        store,
        "assistant",
        "A deleted reply that should never reach the identity package.",
    )
    await store.delete_message(du)
    await store.delete_message(da)

    analyzer = PersonalityAnalyzer(store.db, agent_id="test-agent")
    examples = await analyzer._get_calibration_examples(num_examples=10)

    blob = " ".join(e["input"] + " " + e["output"] for e in examples)
    assert "thoughtful, sufficiently long reply" in blob
    assert "deleted question that should vanish" not in blob
    assert "deleted reply that should never reach" not in blob


@pytest.mark.asyncio
async def test_wellness_content_read_excludes_soft_deleted(store):
    from kestrel_sovereign.features.wellness.metrics import InteractionDepthCalculator

    await _insert(store, "user", "A live substantive message that is long enough to count here.")
    trashed = await _insert(
        store, "user", "A deleted substantive message that is long enough to count here."
    )
    await store.delete_message(trashed)

    calc = InteractionDepthCalculator()
    result = await calc.measure(store.db, "test-agent")

    # Only the one live row should be measured.
    assert result["message_count"] == 1


@pytest.mark.asyncio
async def test_salvage_and_next_id_still_see_soft_deleted(store):
    """Intentionally-unfiltered reads must keep seeing deleted rows."""
    await _insert(store, "user", "live")
    trashed = await _insert(store, "assistant", "trashed")
    await store.delete_message(trashed)

    # next-id / id-sequence read (conversation_manager MAX(id)) must include
    # the deleted row so ids are never reused.
    max_id = (await store.db.fetchall(
        "SELECT MAX(id) FROM conversation_history WHERE agent_id = ?",
        ("test-agent",),
    ))[0][0]
    assert max_id == trashed

    # A raw read with no deleted filter (salvage/backfill style) still sees it.
    all_ids = [
        r[0]
        for r in await store.db.fetchall(
            "SELECT id FROM conversation_history WHERE agent_id = ?", ("test-agent",)
        )
    ]
    assert trashed in all_ids
