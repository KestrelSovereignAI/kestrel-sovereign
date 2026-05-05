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
from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage


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

        assert result == {"conversation_history": 0, "graph_nodes": 0}


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
    from kestrel_sovereign.storage.async_graph_store import GraphNode
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
        await storage.graph.add_node(GraphNode(
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
        survivor = await storage.graph.get_node("other-1")
        assert survivor is not None
        assert survivor.label == "other"

        # Edges between leaked nodes are gone too
        edges = await storage.graph.get_edges("leak-1")
        assert edges == []


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
        assert result == {"conversation_history": 0, "graph_nodes": 0}
