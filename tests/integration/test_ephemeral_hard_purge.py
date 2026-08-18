"""Integration tests for the EPHEMERAL hard-purge defense-in-depth (#767).

Exercises the storage-layer purge against a real SQLite database so the
JSON-path predicate, transactional edge cleanup, and per-agent scoping
are all proven end-to-end. The kestrel_agent transition wiring is
covered separately in unit tests with mocks.

Updated for #867: leaks are seeded AFTER the wrapper transitions into
EPHEMERAL so the entered_ephemeral_at watermark is older than the leak's
``created_at``.  That mirrors real production timing — a leak happens
while the agent is in EPHEMERAL, not before.  Pre-EPHEMERAL data is now
explicitly out of scope for the leak-purge (see
``test_ephemeral_purge_scoped.py`` for the regression suite).
"""
from __future__ import annotations

import asyncio
import json

import pytest

from kestrel_sovereign.privacy import PrivacyMode
from kestrel_sovereign.storage import AsyncStorage
from kestrel_sovereign.storage.privacy_wrapper import (
    REQUIRED_CONTENT_STORES,
    PrivacyEnforcingStorage,
    PurgeOutcome,
)


AGENT_ID = "did:test:ephemeral-purge"
OTHER_AGENT_ID = "did:test:other-agent"


def _now_iso_utc() -> str:
    """Match SQLite's datetime('now') format used by add_node properties."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


@pytest.mark.asyncio
async def test_purge_clean_ephemeral_session_destroys_nothing(tmp_path):
    """Happy path — EPHEMERAL session that wrote nothing has nothing
    to clean up. Purge returns zero leaks, no warning fires."""
    db_path = tmp_path / "kestrel.db"
    async with AsyncStorage(str(db_path), agent_id=AGENT_ID) as storage:
        wrapper = PrivacyEnforcingStorage(storage, PrivacyMode.EPHEMERAL)

        result = await wrapper.purge_ephemeral_session(reason="test")

        assert result == {
            "conversation_history": 0,
            "graph_nodes": 0,
            "channel_messages": 0,
            "session_projection": 0,
        }


@pytest.mark.asyncio
async def test_purge_destroys_conversation_history_leak(tmp_path):
    """If a row somehow reached conversation_history while EPHEMERAL was
    in effect, the hard-purge scrubs it AND reports the count so the
    caller can audit the leak.

    Updated for #867: the wrapper enters EPHEMERAL FIRST so the watermark
    is captured, then the leak is seeded so its ``created_at`` is past
    the watermark.  This is the real production timing — a leak happens
    *during* the EPHEMERAL stint, not before.
    """
    db_path = tmp_path / "kestrel.db"
    async with AsyncStorage(str(db_path), agent_id=AGENT_ID) as storage:
        wrapper = PrivacyEnforcingStorage(storage, PrivacyMode.EPHEMERAL)
        # Per-second watermark — sleep past the boundary so leaks land
        # strictly after the watermark.
        await asyncio.sleep(1.05)
        # Simulate a leak — write directly through the underlying store,
        # bypassing the privacy wrapper that would otherwise reject this.
        await storage.conversation.add_conversation("user", "leaked turn 1")
        await storage.conversation.add_conversation("assistant", "leaked turn 2")

        result = await wrapper.purge_ephemeral_session(reason="test")

        assert result["conversation_history"] == 2
        # Live and trash are both empty after purge — hard-delete, no recovery.
        live = await storage.conversation.get_full_history_with_ids()
        trash = await storage.conversation.get_full_history_with_ids(only_deleted=True)
        assert live == []
        assert trash == []


@pytest.mark.asyncio
async def test_purge_erases_the_change_ledger_when_no_history_survives(tmp_path):
    """"Leave no trace" reaches the #2959 change ledger.

    A database trigger bumps that ledger on every write to
    ``conversation_history``, and a trigger cannot see privacy mode — so a
    purely EPHEMERAL agent that leaked one turn is left named by a row counting
    it, after the sweep that erased the turn itself. Content-free, but the
    contract here is not "no content".
    """
    db_path = tmp_path / "kestrel.db"
    async with AsyncStorage(str(db_path), agent_id=AGENT_ID) as storage:
        wrapper = PrivacyEnforcingStorage(storage, PrivacyMode.EPHEMERAL)
        await asyncio.sleep(1.05)
        await storage.conversation.add_conversation("user", "leaked turn")

        assert await storage.db.fetchval(
            "SELECT COUNT(*) FROM conversation_history_changes WHERE agent_id = ?",
            (AGENT_ID,),
        ) == 1, "the trigger must have stamped the ledger, or this proves nothing"

        await wrapper.purge_ephemeral_session(reason="test")

        assert await storage.db.fetchval(
            "SELECT COUNT(*) FROM conversation_history_changes WHERE agent_id = ?",
            (AGENT_ID,),
        ) == 0, "the ledger still names the EPHEMERAL agent after its purge"


@pytest.mark.asyncio
async def test_purge_keeps_the_ledger_when_legitimate_history_survives(tmp_path):
    """The scoped-purge contract wins over tidiness.

    An agent with pre-EPHEMERAL history is already recorded by that history, so
    its ledger row names nothing the database does not already say, and the
    counter is a change token with no meaning of its own. Deleting it would
    touch state authored before entry — which the scoped purge forbids — to
    erase a trace that is not one.
    """
    db_path = tmp_path / "kestrel.db"
    async with AsyncStorage(str(db_path), agent_id=AGENT_ID) as storage:
        await storage.conversation.add_conversation("user", "legitimate, pre-EPHEMERAL")
        # Past the per-second watermark boundary BEFORE entering, so the row
        # above is strictly older than the EPHEMERAL entry and the scoped purge
        # must leave it alone. Without this the "legitimate" row lands in the
        # same second as entry, is swept as in-window, and the case silently
        # tests the leak path instead.
        await asyncio.sleep(1.05)
        wrapper = PrivacyEnforcingStorage(storage, PrivacyMode.EPHEMERAL)
        await asyncio.sleep(1.05)

        result = await wrapper.purge_ephemeral_session(reason="test")

        assert result["session_projection"] == 0
        assert await storage.db.fetchval(
            "SELECT COUNT(*) FROM conversation_history_changes WHERE agent_id = ?",
            (AGENT_ID,),
        ) == 1, "a clean stint deleted state authored before EPHEMERAL entry"


@pytest.mark.asyncio
async def test_a_purged_pre_entry_ledger_does_not_report_a_leak(tmp_path):
    """A clean stint must not be audited as a leak because of a change token.

    An agent that hard-purged its NORMAL history BEFORE entering EPHEMERAL has
    no surviving history and a legitimate ledger row left by the trigger. The
    sweep cannot tell that row from one this stint created — provenance would
    have to be captured at entry, and both entry paths are synchronous, so
    there is no read to capture it with.

    What it CAN do is not certify content it does not hold. The ledger is a
    monotonic cache-invalidation token; removing it destroys no record and the
    projection re-derives. So the sweep is reported and is not required, and a
    clean stint is never audited as a leak on account of it.
    """
    db_path = tmp_path / "kestrel.db"
    async with AsyncStorage(str(db_path), agent_id=AGENT_ID) as storage:
        await storage.conversation.add_conversation("user", "normal-mode turn")
        # Hard-purge it while still NORMAL: history is now empty, but the
        # trigger's ledger row survives and predates EPHEMERAL entirely.
        await storage.conversation.purge_all_since("1970-01-01", reason="normal-purge")
        assert await storage.db.fetchval(
            "SELECT COUNT(*) FROM conversation_history_changes WHERE agent_id = ?",
            (AGENT_ID,),
        ) >= 1, "fixture needs a pre-entry ledger row to test anything"

        await asyncio.sleep(1.05)
        wrapper = PrivacyEnforcingStorage(storage, PrivacyMode.EPHEMERAL)
        await asyncio.sleep(1.05)

        report = await wrapper.purge_ephemeral_session(reason="test")

        assert not report.required_sweep_failed, (
            "a stint that wrote nothing was audited as a failed content sweep"
        )
        assert report["conversation_history"] == 0, "no content leaked"
        assert "session_projection" not in REQUIRED_CONTENT_STORES, (
            "a change token must not certify content: requiring it turns "
            "removing a pre-entry counter into a reported leak"
        )


@pytest.mark.asyncio
async def test_a_retry_reaches_the_same_verdict_as_the_first_attempt(tmp_path):
    """The condition must be durable, not a count of what this attempt deleted.

    If the ledger sweep is the only one that fails, the attempt has already
    destroyed the leaked history — so a second attempt sweeps zero content rows,
    not because nothing leaked but because the evidence was removed by the pass
    that failed to finish. "Does any history survive" asks the database, gets
    the same answer both times, and is the reason this needs no orphan probe or
    leak flag.
    """
    db_path = tmp_path / "kestrel.db"
    async with AsyncStorage(str(db_path), agent_id=AGENT_ID) as storage:
        wrapper = PrivacyEnforcingStorage(storage, PrivacyMode.EPHEMERAL)
        await asyncio.sleep(1.05)
        await storage.conversation.add_conversation("user", "leaked turn")

        # The first attempt, stopping after the history sweep.
        await storage.conversation.purge_all_since("1970-01-01", reason="partial")
        assert await storage.db.fetchval(
            "SELECT COUNT(*) FROM conversation_history_changes WHERE agent_id = ?",
            (AGENT_ID,),
        ) >= 1, "fixture must leave the ledger standing to test anything"

        result = await wrapper.purge_ephemeral_session(reason="retry")

        assert result["conversation_history"] == 0, "the retry finds no history left"
        assert await storage.db.fetchval(
            "SELECT COUNT(*) FROM conversation_history_changes WHERE agent_id = ?",
            (AGENT_ID,),
        ) == 0, (
            "the retry read zero deleted rows as 'nothing leaked' and left the "
            "ledger naming the EPHEMERAL agent"
        )


@pytest.mark.asyncio
async def test_purge_destroys_soft_deleted_leak_too(tmp_path):
    """A leaked row that's been soft-deleted still counts — EPHEMERAL
    contract is "no trace at all," not "no live trace." Purge wipes
    both deleted_at IS NULL and deleted_at IS NOT NULL rows."""
    db_path = tmp_path / "kestrel.db"
    async with AsyncStorage(str(db_path), agent_id=AGENT_ID) as storage:
        wrapper = PrivacyEnforcingStorage(storage, PrivacyMode.EPHEMERAL)
        await asyncio.sleep(1.05)
        await storage.conversation.add_conversation("user", "leaked")
        rows = await storage.conversation.get_full_history_with_ids()
        # Soft-delete the leaked row to simulate the "leaked then user
        # tried to delete" path. Both must vanish on EPHEMERAL purge.
        await storage.conversation.delete_message(rows[0]["id"])
        in_trash = await storage.conversation.get_full_history_with_ids(only_deleted=True)
        assert len(in_trash) == 1

        result = await wrapper.purge_ephemeral_session(reason="test")

        assert result["conversation_history"] == 1
        gone = await storage.conversation.get_full_history_with_ids(
            include_deleted=True
        )
        assert gone == []


@pytest.mark.asyncio
async def test_purge_destroys_leaked_graph_nodes(tmp_path):
    """Graph nodes the EPHEMERAL agent shouldn't have written get the
    same hard-delete treatment, with edges scrubbed in the same
    transaction.

    Updated for #867: leak nodes carry an in-window ``created_at``
    matching real production node properties (``add_node`` callers
    stamp it).  Pre-stint nodes are now correctly preserved by the
    scoped purge — that's the safety improvement on the table.
    """
    from kestrel_sovereign.storage.async_graph_store import AsyncGraphStore, GraphNode
    db_path = tmp_path / "kestrel.db"
    async with AsyncStorage(str(db_path), agent_id=AGENT_ID) as storage:
        wrapper = PrivacyEnforcingStorage(storage, PrivacyMode.EPHEMERAL)
        await asyncio.sleep(1.05)
        leak_ts = _now_iso_utc()
        # Two nodes for this agent (the leak), one node for another agent
        # (must NOT be touched), and an edge between two of the leaked nodes.
        await storage.graph.add_node(GraphNode(
            node_id="leak-1", node_type="memory", label="memory-leak-1",
            properties={"agent_id": AGENT_ID, "created_at": leak_ts},
        ))
        await storage.graph.add_node(GraphNode(
            node_id="leak-2", node_type="memory", label="memory-leak-2",
            properties={"agent_id": AGENT_ID, "created_at": leak_ts},
        ))
        other_graph = AsyncGraphStore(storage.db, agent_id=OTHER_AGENT_ID)
        await other_graph.add_node(GraphNode(
            node_id="other-1", node_type="memory", label="other",
            properties={"agent_id": OTHER_AGENT_ID, "created_at": leak_ts},
        ))
        await storage.graph.add_edge("leak-1", "leak-2", "follows")

        result = await wrapper.purge_ephemeral_session(reason="test")

        assert result["graph_nodes"] == 2

        # Leak agent's nodes gone
        assert await storage.graph.get_node("leak-1") is None
        assert await storage.graph.get_node("leak-2") is None

        # Other agent's node untouched — per-agent scoping is preserved
        assert await storage.graph.get_node("other-1") is None
        survivor = await other_graph.get_node("other-1")
        assert survivor is not None
        assert survivor.label == "other"

        # Edges between leaked nodes are gone too
        edges = await storage.graph.get_edges("leak-1")
        assert edges == []


@pytest.mark.asyncio
async def test_purge_destroys_leaked_channel_messages(tmp_path):
    """A leaked channel_messages row (channels feature #2096 / F112) gets
    the same scoped hard-delete treatment, reported in the breakdown, and
    per-agent scoping is preserved.

    The channels feature owns the table, so the test creates it directly
    and seeds an in-window leak plus another agent's row that must survive.
    """
    db_path = tmp_path / "kestrel.db"
    async with AsyncStorage(str(db_path), agent_id=AGENT_ID) as storage:
        await storage.db.execute_commit(
            """CREATE TABLE IF NOT EXISTS channel_messages (
                   id TEXT PRIMARY KEY,
                   agent_id TEXT NOT NULL,
                   channel_type TEXT NOT NULL,
                   direction TEXT NOT NULL,
                   sender TEXT NOT NULL,
                   recipient TEXT NOT NULL,
                   content TEXT NOT NULL,
                   status TEXT NOT NULL DEFAULT 'success',
                   metadata TEXT,
                   created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
               )"""
        )

        wrapper = PrivacyEnforcingStorage(storage, PrivacyMode.EPHEMERAL)
        await asyncio.sleep(1.05)
        leak_ts = _now_iso_utc()

        async def _insert(mid, agent, ts):
            await storage.db.execute_commit(
                "INSERT INTO channel_messages "
                "(id, agent_id, channel_type, direction, sender, recipient, "
                " content, status, metadata, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (mid, agent, "telegram", "inbound", "alice", "bot",
                 "leaked channel text", "received", None, ts),
            )

        await _insert("leak-msg-1", AGENT_ID, leak_ts)
        await _insert("leak-msg-2", AGENT_ID, leak_ts)
        await _insert("other-msg", OTHER_AGENT_ID, leak_ts)

        result = await wrapper.purge_ephemeral_session(reason="test")

        assert result["channel_messages"] == 2

        rows = await storage.db.fetchall(
            "SELECT id FROM channel_messages ORDER BY id"
        )
        surviving = {r[0] for r in rows}
        # Only the other agent's row survives; per-agent scoping preserved.
        assert surviving == {"other-msg"}


@pytest.mark.asyncio
async def test_purge_channel_messages_preserves_same_day_preephemeral_row(tmp_path):
    """Regression (#2102 follow-up): the watermark is space-format
    (``datetime('now')`` → ``YYYY-MM-DD HH:MM:SS``) but the channels feature
    writes ``created_at`` as ``message.timestamp.isoformat()`` (``T`` separator,
    microseconds, offset). A raw lexical ``created_at >= since`` treats ANY
    same-UTC-day ISO row as ``>=`` the watermark (``'T'`` 0x54 > ``' '`` 0x20),
    so a NORMAL channel message authored hours BEFORE the brief EPHEMERAL stint
    was wrongly purged — silent data loss.

    Seeds the real production timestamp shapes around a controlled watermark and
    asserts the pre-watermark NORMAL row survives while the in-window leaks (in
    BOTH formats) are destroyed.
    """
    db_path = tmp_path / "kestrel.db"
    async with AsyncStorage(str(db_path), agent_id=AGENT_ID) as storage:
        await storage.db.execute_commit(
            """CREATE TABLE IF NOT EXISTS channel_messages (
                   id TEXT PRIMARY KEY,
                   agent_id TEXT NOT NULL,
                   channel_type TEXT NOT NULL,
                   direction TEXT NOT NULL,
                   sender TEXT NOT NULL,
                   recipient TEXT NOT NULL,
                   content TEXT NOT NULL,
                   status TEXT NOT NULL DEFAULT 'success',
                   metadata TEXT,
                   created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
               )"""
        )

        async def _insert(mid, ts):
            await storage.db.execute_commit(
                "INSERT INTO channel_messages "
                "(id, agent_id, channel_type, direction, sender, recipient, "
                " content, status, metadata, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (mid, AGENT_ID, "telegram", "inbound", "alice", "bot",
                 "text", "received", None, ts),
            )

        # NORMAL history authored at 09:30 (ISO-T shape, as the channels
        # feature really writes it) — hours before the ephemeral stint.
        await _insert("normal-0930", "2026-07-01T09:30:00.123456+00:00")
        # In-window leaks at 13:15 (ISO-T) and 14:00 (space form) — both purge.
        await _insert("leak-iso-1315", "2026-07-01T13:15:00.000000+00:00")
        await _insert("leak-space-1400", "2026-07-01 14:00:00")

        wrapper = PrivacyEnforcingStorage(storage, PrivacyMode.EPHEMERAL)
        # Control the watermark: entered EPHEMERAL at 12:00 (space form, the
        # real _now_iso() shape). 09:30 is BEFORE it; 13:15/14:00 are after.
        wrapper._entered_ephemeral_at = "2026-07-01 12:00:00"

        result = await wrapper.purge_ephemeral_session(reason="test")

        assert result["channel_messages"] == 2  # only the two leaks
        rows = await storage.db.fetchall(
            "SELECT id FROM channel_messages ORDER BY id"
        )
        assert {r[0] for r in rows} == {"normal-0930"}


@pytest.mark.asyncio
async def test_purge_channel_messages_purges_unparseable_timestamp_failsafe(tmp_path):
    """A leaked row with a malformed created_at must NOT survive the privacy
    sweep — parse failure fail-safes to purge, not keep."""
    db_path = tmp_path / "kestrel.db"
    async with AsyncStorage(str(db_path), agent_id=AGENT_ID) as storage:
        await storage.db.execute_commit(
            """CREATE TABLE IF NOT EXISTS channel_messages (
                   id TEXT PRIMARY KEY, agent_id TEXT NOT NULL,
                   channel_type TEXT NOT NULL, direction TEXT NOT NULL,
                   sender TEXT NOT NULL, recipient TEXT NOT NULL,
                   content TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'success',
                   metadata TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
               )"""
        )
        for mid, ts in [
            ("normal-early", "2026-07-01T09:30:00.000000+00:00"),
            ("leak-malformed", "0000-00-00 00:00:00"),
            ("leak-garbage", "not-a-timestamp"),
        ]:
            await storage.db.execute_commit(
                "INSERT INTO channel_messages (id, agent_id, channel_type, "
                "direction, sender, recipient, content, status, metadata, "
                "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (mid, AGENT_ID, "telegram", "inbound", "a", "b", "x",
                 "received", None, ts),
            )
        wrapper = PrivacyEnforcingStorage(storage, PrivacyMode.EPHEMERAL)
        wrapper._entered_ephemeral_at = "2026-07-01 12:00:00"
        result = await wrapper.purge_ephemeral_session(reason="test")
        # Both malformed leaks purged; the parseable pre-watermark row survives.
        assert result["channel_messages"] == 2
        rows = await storage.db.fetchall("SELECT id FROM channel_messages")
        assert {r[0] for r in rows} == {"normal-early"}


@pytest.mark.asyncio
async def test_purge_channel_messages_batches_beyond_bind_limit(tmp_path):
    """A high-volume leak (> SQLite's 999 bind-param ceiling) must fully purge —
    the DELETE is batched so it never raises on placeholder count and gets
    swallowed as channel_messages: 0."""
    db_path = tmp_path / "kestrel.db"
    async with AsyncStorage(str(db_path), agent_id=AGENT_ID) as storage:
        await storage.db.execute_commit(
            """CREATE TABLE IF NOT EXISTS channel_messages (
                   id TEXT PRIMARY KEY, agent_id TEXT NOT NULL,
                   channel_type TEXT NOT NULL, direction TEXT NOT NULL,
                   sender TEXT NOT NULL, recipient TEXT NOT NULL,
                   content TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'success',
                   metadata TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
               )"""
        )
        n = 1500  # exceeds the 999 default and the _DELETE_ID_BATCH of 500
        for i in range(n):
            await storage.db.execute_commit(
                "INSERT INTO channel_messages (id, agent_id, channel_type, "
                "direction, sender, recipient, content, status, metadata, "
                "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (f"leak-{i:05d}", AGENT_ID, "telegram", "inbound", "a", "b",
                 "x", "received", None, "2026-07-01T13:00:00.000000+00:00"),
            )
        wrapper = PrivacyEnforcingStorage(storage, PrivacyMode.EPHEMERAL)
        wrapper._entered_ephemeral_at = "2026-07-01 12:00:00"
        result = await wrapper.purge_ephemeral_session(reason="test")
        assert result["channel_messages"] == n
        rows = await storage.db.fetchall("SELECT COUNT(*) FROM channel_messages")
        assert rows[0][0] == 0


@pytest.mark.asyncio
async def test_purge_channel_messages_tolerates_missing_table(tmp_path):
    """When the channels feature never loaded, its table doesn't exist —
    the purge must report 0 rather than raising."""
    db_path = tmp_path / "kestrel.db"
    async with AsyncStorage(str(db_path), agent_id=AGENT_ID) as storage:
        wrapper = PrivacyEnforcingStorage(storage, PrivacyMode.EPHEMERAL)
        await asyncio.sleep(1.05)
        result = await wrapper.purge_ephemeral_session(reason="test")
        assert result["channel_messages"] == 0


@pytest.mark.asyncio
async def test_purge_does_not_touch_other_agents_data(tmp_path):
    """Per-agent scoping check at the conversation_history layer.

    AGENT_ID's purge must not affect OTHER_AGENT_ID's conversations,
    even when they share a database file. This guards against the
    multi_agent scenario where multiple agents live in the same DB and
    one of them is briefly in EPHEMERAL mode.
    """
    db_path = tmp_path / "kestrel.db"
    async with AsyncStorage(str(db_path), agent_id=OTHER_AGENT_ID) as other_storage:
        await other_storage.conversation.add_conversation("user", "preserved 1")
        await other_storage.conversation.add_conversation("assistant", "preserved 2")

    async with AsyncStorage(str(db_path), agent_id=AGENT_ID) as storage:
        wrapper = PrivacyEnforcingStorage(storage, PrivacyMode.EPHEMERAL)
        await asyncio.sleep(1.05)
        # Leak some conversation data on the ephemeral agent only
        await storage.conversation.add_conversation("user", "leaked")

        result = await wrapper.purge_ephemeral_session(reason="test")
        assert result["conversation_history"] == 1

    # Reopen as the other agent — its data must survive
    async with AsyncStorage(str(db_path), agent_id=OTHER_AGENT_ID) as other_storage:
        rows = await other_storage.conversation.get_full_history_with_ids()
        assert len(rows) == 2
        assert rows[0]["content"] == "preserved 1"


@pytest.mark.asyncio
async def test_purge_warns_and_returns_breakdown_on_leak(tmp_path, caplog):
    """When the privacy layer leaks, the storage-layer purge logs a
    WARNING with the row count. The caller (kestrel_agent) is then
    responsible for writing the audit entry; the wrapper just makes
    the leak visible in the application log.
    """
    import logging as _logging
    db_path = tmp_path / "kestrel.db"
    async with AsyncStorage(str(db_path), agent_id=AGENT_ID) as storage:
        wrapper = PrivacyEnforcingStorage(storage, PrivacyMode.EPHEMERAL)
        await asyncio.sleep(1.05)
        await storage.conversation.add_conversation("user", "leaked")

        with caplog.at_level(_logging.WARNING, logger="kestrel_sovereign.storage.privacy_wrapper"):
            result = await wrapper.purge_ephemeral_session(reason="test")

        assert result["conversation_history"] == 1
        # The WARNING goes through the privacy_wrapper logger
        assert any(
            "EPHEMERAL session leaked" in rec.message
            for rec in caplog.records
        ), f"Expected leak warning, got {[r.message for r in caplog.records]}"


@pytest.mark.asyncio
async def test_purge_clears_isolated_session_buffer(tmp_path):
    """Belt-and-braces — the in-memory ISOLATED buffer is also cleared.

    If the wrapper somehow retained session-local conversations from a
    prior ISOLATED stint and the agent then flipped through EPHEMERAL
    on the way to NORMAL, the buffer should not leak forward."""
    db_path = tmp_path / "kestrel.db"
    async with AsyncStorage(str(db_path), agent_id=AGENT_ID) as storage:
        wrapper = PrivacyEnforcingStorage(storage, PrivacyMode.EPHEMERAL)
        # Manually seed the buffer the way ISOLATED mode would.
        wrapper._session_conversations.append({
            "role": "user", "content": "stale", "metadata": {},
        })
        wrapper._session_files["foo"] = b"bar"

        await wrapper.purge_ephemeral_session(reason="test")

        assert wrapper._session_conversations == []
        assert wrapper._session_files == {}


@pytest.mark.asyncio
async def test_purge_with_empty_agent_id_is_a_safe_noop(tmp_path):
    """Defensive path: if the wrapper has no agent_id (early bootstrap,
    multi-tenant share-of-DB scenario), purge must NOT issue a
    DELETE without a WHERE clause that would scrub everyone's data.
    """
    db_path = tmp_path / "kestrel.db"
    async with AsyncStorage(str(db_path), agent_id="some-agent") as storage:
        await storage.conversation.add_conversation("user", "owned-by-some-agent")

        # Wrap with an explicitly-empty agent_id facade by overriding
        # the cached agent_id on the wrapper.
        wrapper = PrivacyEnforcingStorage(storage, PrivacyMode.EPHEMERAL)
        # Force the wrapper to think it has no agent
        storage.agent_id = ""

        result = await wrapper.purge_ephemeral_session(reason="test")
        assert result == {
            "conversation_history": 0,
            "graph_nodes": 0,
            "channel_messages": 0,
        }
        # A safe no-op is SKIPPED, never FAILED — it must not block mode exit.
        assert result.required_sweep_failed is False


@pytest.mark.parametrize("backend_method, failed_store", [
    ("purge_conversations_since", "conversation_history"),
    ("purge_agent_graph_nodes", "graph_nodes"),
    ("purge_channel_messages_since", "channel_messages"),
])
@pytest.mark.asyncio
async def test_backend_failure_is_distinguishable_from_clean_zero(
    tmp_path, monkeypatch, backend_method, failed_store
):
    """#2673: an injected failure in EACH purge backend must be distinguishable
    from a clean zero. The flat count reads 0 (row count unknown) exactly like a
    clean sweep, but the structured outcome is FAILED and required_sweep_failed
    trips — so a caller can tell "could not check/delete" from "nothing found".
    """
    db_path = tmp_path / "kestrel.db"
    async with AsyncStorage(str(db_path), agent_id=AGENT_ID) as storage:
        wrapper = PrivacyEnforcingStorage(storage, PrivacyMode.EPHEMERAL)
        await asyncio.sleep(1.05)

        async def _boom(*args, **kwargs):
            raise RuntimeError("simulated backend failure")

        # Inject a failure into just this one backend sweep.
        monkeypatch.setattr(storage, backend_method, _boom)

        report = await wrapper.purge_ephemeral_session(reason="test-fail")

    # Flat mapping: the failed store reads 0 (unknown), same shape a clean zero
    # would produce — so the flat count alone cannot distinguish them.
    assert report[failed_store] == 0
    # The structured outcome DOES distinguish: FAILED vs a genuine CLEAN zero.
    failed = report.store_results[failed_store]
    assert failed.outcome is PurgeOutcome.FAILED
    assert failed.error is not None
    # The other required stores swept cleanly — a genuine CLEAN zero.
    for store in ("conversation_history", "graph_nodes", "channel_messages"):
        if store == failed_store:
            continue
        other = report.store_results[store]
        assert other.outcome is PurgeOutcome.CLEAN and other.rows == 0
    # The required-failure signal trips off the FAILED sweep, not the counts.
    assert report.required_sweep_failed is True


@pytest.mark.asyncio
async def test_all_sweeps_attempted_even_when_one_fails(tmp_path, monkeypatch):
    """#2673: a failure in one sweep must not skip the others. The graph sweep
    still runs and purges its leaked node even though the conversation sweep
    raised — every independent sweep is attempted and its outcome retained.
    """
    from kestrel_sovereign.storage.async_graph_store import GraphNode

    db_path = tmp_path / "kestrel.db"
    async with AsyncStorage(str(db_path), agent_id=AGENT_ID) as storage:
        wrapper = PrivacyEnforcingStorage(storage, PrivacyMode.EPHEMERAL)
        await asyncio.sleep(1.05)
        leak_ts = _now_iso_utc()
        await storage.graph.add_node(GraphNode(
            node_id="leak-node", node_type="memory", label="leaked",
            properties={"agent_id": AGENT_ID, "created_at": leak_ts},
        ))

        async def _boom(*args, **kwargs):
            raise RuntimeError("conversation sweep down")

        monkeypatch.setattr(storage, "purge_conversations_since", _boom)

        report = await wrapper.purge_ephemeral_session(reason="test-partial")

        # Conversation sweep failed...
        assert report.store_results["conversation_history"].outcome is PurgeOutcome.FAILED
        # ...but the graph sweep still ran and purged the leaked node.
        assert report.store_results["graph_nodes"].outcome is PurgeOutcome.PURGED
        assert report["graph_nodes"] == 1
        assert await storage.graph.get_node("leak-node") is None
        assert report.required_sweep_failed is True


@pytest.mark.asyncio
async def test_watermark_preserved_when_required_sweep_fails(tmp_path, monkeypatch):
    """#2673: a required sweep failure must NOT clear the entered-ephemeral
    watermark. The caller keeps the agent in EPHEMERAL (fail closed), so the
    watermark must survive for a retry to re-scope to the same stint rather than
    hitting the #867 no-watermark refusal.
    """
    db_path = tmp_path / "kestrel.db"
    async with AsyncStorage(str(db_path), agent_id=AGENT_ID) as storage:
        wrapper = PrivacyEnforcingStorage(storage, PrivacyMode.EPHEMERAL)
        await asyncio.sleep(1.05)
        watermark = wrapper._entered_ephemeral_at
        assert watermark is not None

        async def _boom(*args, **kwargs):
            raise RuntimeError("down")

        monkeypatch.setattr(storage, "purge_conversations_since", _boom)

        report = await wrapper.purge_ephemeral_session(reason="test-fail")
        assert report.required_sweep_failed is True
        # Watermark NOT cleared — a retry can still scope to the same stint.
        assert wrapper._entered_ephemeral_at == watermark


@pytest.mark.asyncio
async def test_clean_sweep_clears_watermark(tmp_path):
    """#2673 counterpart: a fully clean sweep DOES retire the watermark (the
    stint is certified complete), preserving the pre-existing semantics."""
    db_path = tmp_path / "kestrel.db"
    async with AsyncStorage(str(db_path), agent_id=AGENT_ID) as storage:
        wrapper = PrivacyEnforcingStorage(storage, PrivacyMode.EPHEMERAL)
        await asyncio.sleep(1.05)
        assert wrapper._entered_ephemeral_at is not None

        report = await wrapper.purge_ephemeral_session(reason="test-clean")

        assert report.required_sweep_failed is False
        assert wrapper._entered_ephemeral_at is None


@pytest.mark.asyncio
async def test_purge_leaves_no_watermark_that_can_match_a_restarted_ledger(tmp_path):
    """A stamp must not outlive the ledger incarnation it was read from.

    ``is_stale()`` compares a stored stamp to the change ledger for EQUALITY,
    which is sound only while that counter rises. Erasing the ledger row on its
    own breaks it: the trigger's next write is an INSERT of ``1``, so the
    counter restarts, and a watermark left behind at stamp N matches again as
    soon as N further row events happen — immediately when N is 1, which is the
    case a leaked single turn produces. The projection then reports itself
    CURRENT while describing history that was purged.

    So the purge erases every table the projection owns, together. This test
    fails if the sweep is narrowed back to the ledger alone.
    """
    from kestrel_sovereign.storage.conversation_sessions import (
        ConversationSessionProjection,
    )

    db_path = tmp_path / "kestrel.db"
    async with AsyncStorage(str(db_path), agent_id=AGENT_ID) as storage:
        wrapper = PrivacyEnforcingStorage(storage, PrivacyMode.EPHEMERAL)
        await asyncio.sleep(1.05)
        await storage.conversation.add_conversation("user", "leaked turn")

        projection = ConversationSessionProjection(storage.db, AGENT_ID)
        await projection.repair()
        assert not await projection.is_stale(), (
            "the projection must start CURRENT, or the assertion below passes "
            "for the wrong reason"
        )
        stamp = (await projection.accounted()).stamp
        assert stamp == 1, (
            f"this test needs the N=1 collision and the stamp is {stamp}; "
            "the arrangement changed, not the guard"
        )
        assert await projection.list(), "nothing was projected to go stale"

        await wrapper.purge_ephemeral_session(reason="test")

        # One more row event brings the restarted ledger back to 1 — the value
        # the pre-purge stamp recorded.
        await storage.conversation.add_conversation("user", "after the purge")
        assert await projection.observed_changes() == stamp, (
            "the collision this guards against did not occur, so a pass here "
            "would prove nothing"
        )

        assert await projection.is_stale(), (
            "the projection reports itself CURRENT after a purge, while its "
            "rows describe history that no longer exists"
        )
