"""Integration tests for the scoped EPHEMERAL leak-purge (#867).

The 2026-04-26 wipe happened because ``purge_ephemeral_session`` called
the *unscoped* ``purge_all_conversations``: flipping a long-lived agent
into EPHEMERAL for thirty seconds then back out destroyed every row the
agent had ever authored.  This file is the regression suite for the
scoping fix.

The tests use a real SQLite database so the ``created_at >= ?`` predicate
is exercised end-to-end.  The ticket's acceptance criteria map onto the
test names — each bullet has a row.
"""
from __future__ import annotations

import asyncio

import pytest

from kestrel_sovereign.privacy import PrivacyMode
from kestrel_sovereign.storage import AsyncStorage
from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage


AGENT_ID = "did:test:ephemeral-scoped"


@pytest.mark.asyncio
async def test_no_writes_during_ephemeral_destroys_nothing(tmp_path):
    """AC: flip in/out of EPHEMERAL with no writes → 0 rows lost.

    The motivating wipe: Meridian had 169 NORMAL rows, was flipped into
    EPHEMERAL by a demo (no writes during the flip), and the exit purge
    destroyed all 169.  After the scoping fix, the same flip leaves
    every preexisting row in place because none of them were authored
    on or after the EPHEMERAL-entry timestamp.
    """
    db_path = tmp_path / "kestrel.db"
    async with AsyncStorage(str(db_path), agent_id=AGENT_ID) as storage:
        # Seed three NORMAL rows BEFORE entering EPHEMERAL.
        await storage.conversation.add_conversation("user", "preexisting #1")
        await storage.conversation.add_conversation("assistant", "preexisting #2")
        await storage.conversation.add_conversation("user", "preexisting #3")

        # The watermark resolution is one second (matches SQLite's
        # ``datetime('now')`` format).  Sleep past the second boundary so
        # the seeded rows are demonstrably authored before the watermark
        # — this mirrors real usage where a mode flip happens minutes or
        # hours after the data was written.
        await asyncio.sleep(1.05)

        wrapper = PrivacyEnforcingStorage(storage, PrivacyMode.NORMAL)
        wrapper.set_privacy_mode(PrivacyMode.EPHEMERAL)

        # No writes during the EPHEMERAL stint.

        result = await wrapper.purge_ephemeral_session(reason="test-no-writes")

    assert result == {"conversation_history": 0, "graph_nodes": 0}, (
        "Scoped purge must not touch rows authored before EPHEMERAL entry"
    )

    # Re-open and confirm the preexisting rows survived.
    async with AsyncStorage(str(db_path), agent_id=AGENT_ID) as storage:
        rows = await storage.conversation.get_full_history_with_ids()
        assert len(rows) == 3


@pytest.mark.asyncio
async def test_only_in_window_rows_are_purged(tmp_path):
    """AC: write 3 messages during EPHEMERAL, flip back → only those 3 rows are purged.

    Pre-existing NORMAL rows survive; rows authored *during* the
    EPHEMERAL stint are destroyed because they're real privacy-layer
    leaks (the wrapper was supposed to drop them).
    """
    db_path = tmp_path / "kestrel.db"
    async with AsyncStorage(str(db_path), agent_id=AGENT_ID) as storage:
        # Two NORMAL rows authored before the EPHEMERAL stint.
        await storage.conversation.add_conversation("user", "before-eph #1")
        await storage.conversation.add_conversation("assistant", "before-eph #2")

        # Watermark resolution is one second — sleep past the boundary so
        # the seeded rows are demonstrably pre-stint.
        await asyncio.sleep(1.05)

        wrapper = PrivacyEnforcingStorage(storage, PrivacyMode.NORMAL)
        wrapper.set_privacy_mode(PrivacyMode.EPHEMERAL)

        # Three rows that "leaked" through the wrapper while EPHEMERAL.
        # We bypass the wrapper for the test because the wrapper would
        # reject the writes — the leak we're modeling is a hypothetical
        # wrapper bug or direct backend write.
        await storage.conversation.add_conversation("user", "in-eph leak #1")
        await storage.conversation.add_conversation("assistant", "in-eph leak #2")
        await storage.conversation.add_conversation("user", "in-eph leak #3")

        result = await wrapper.purge_ephemeral_session(reason="test-leaks")

    assert result["conversation_history"] == 3, (
        f"Expected exactly the 3 in-window leaks to be purged, got "
        f"{result['conversation_history']}"
    )

    # Surviving rows must be exactly the two preexisting ones.
    async with AsyncStorage(str(db_path), agent_id=AGENT_ID) as storage:
        rows = await storage.conversation.get_full_history_with_ids()
        assert len(rows) == 2, (
            f"Preexisting NORMAL rows must survive the scoped purge: "
            f"{[r.get('content') for r in rows]}"
        )


@pytest.mark.asyncio
async def test_purge_without_watermark_refuses(tmp_path):
    """AC: missing entered_ephemeral_at watermark must NOT fall through to
    a wide DELETE.

    This is the regression rail for the original bug: when the wrapper
    has no watermark (e.g. it was constructed already in EPHEMERAL and
    never observed the entry), the legacy code called ``purge_all`` and
    wiped everything.  The fixed code refuses to purge.
    """
    db_path = tmp_path / "kestrel.db"
    async with AsyncStorage(str(db_path), agent_id=AGENT_ID) as storage:
        await storage.conversation.add_conversation("user", "preexisting")

        # Constructed in EPHEMERAL — watermark is captured at construction.
        wrapper = PrivacyEnforcingStorage(storage, PrivacyMode.EPHEMERAL)
        # Manually wipe the watermark to simulate the missing-watermark
        # corner case (e.g. wrapper rebuild mid-stint).
        wrapper._entered_ephemeral_at = None

        result = await wrapper.purge_ephemeral_session(
            reason="test-missing-watermark"
        )

    assert result == {"conversation_history": 0, "graph_nodes": 0}, (
        "Missing watermark must not trigger an unbounded DELETE"
    )

    async with AsyncStorage(str(db_path), agent_id=AGENT_ID) as storage:
        rows = await storage.conversation.get_full_history_with_ids()
        assert len(rows) == 1


@pytest.mark.asyncio
async def test_watermark_refreshes_on_re_entry(tmp_path):
    """The watermark is captured every time EPHEMERAL is entered, not just
    the first time.  Stale watermarks from a prior stint can't survive a
    return to NORMAL → second EPHEMERAL stint.
    """
    db_path = tmp_path / "kestrel.db"
    async with AsyncStorage(str(db_path), agent_id=AGENT_ID) as storage:
        wrapper = PrivacyEnforcingStorage(storage, PrivacyMode.NORMAL)

        wrapper.set_privacy_mode(PrivacyMode.EPHEMERAL)
        first_stamp = wrapper._entered_ephemeral_at
        assert first_stamp is not None

        # Exit → watermark cleared by the purge.
        await wrapper.purge_ephemeral_session(reason="exit-1")
        wrapper.set_privacy_mode(PrivacyMode.NORMAL)
        assert wrapper._entered_ephemeral_at is None

        # Re-enter EPHEMERAL after the next-second boundary so the new
        # watermark is strictly later than the first (per-second resolution).
        await asyncio.sleep(1.05)
        wrapper.set_privacy_mode(PrivacyMode.EPHEMERAL)
        second_stamp = wrapper._entered_ephemeral_at
        assert second_stamp is not None
        assert second_stamp > first_stamp, (
            "Watermark must refresh on re-entry to EPHEMERAL"
        )


@pytest.mark.asyncio
async def test_graph_nodes_with_iso_timestamps_are_scoped_correctly(tmp_path):
    """Regression for the format-mismatch bug found in PR review.

    Production graph_nodes store ``properties.created_at`` as
    ``YYYY-MM-DDTHH:MM:SS+00:00`` (ISO with T separator and offset) per
    the ``async_graph_store`` module docstring — produced by
    ``datetime.now(timezone.utc).isoformat()``.  The leak-purge watermark
    is space-format ``YYYY-MM-DD HH:MM:SS``.  A naive lex compare puts
    ISO strings strictly after space strings (T > space in ASCII), so
    every same-day pre-EPHEMERAL graph node would be incorrectly purged
    against a space-format watermark.

    The fix normalises ``properties.created_at`` server-side before
    comparing — this test seeds real ISO-format timestamps and proves
    that pre-stint nodes survive while in-stint leaks are destroyed.
    """
    from datetime import datetime, timedelta, timezone
    from kestrel_sovereign.storage.async_graph_store import GraphNode

    db_path = tmp_path / "kestrel.db"
    async with AsyncStorage(str(db_path), agent_id=AGENT_ID) as storage:
        # Pre-stint: a node authored 10 minutes ago in real ISO format.
        old_ts = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
        await storage.graph.add_node(GraphNode(
            node_id="pre-stint", node_type="memory", label="from-NORMAL",
            properties={"agent_id": AGENT_ID, "created_at": old_ts},
        ))

        wrapper = PrivacyEnforcingStorage(storage, PrivacyMode.NORMAL)
        wrapper.set_privacy_mode(PrivacyMode.EPHEMERAL)

        await asyncio.sleep(1.05)

        # In-stint leak: ISO timestamp captured AFTER the watermark.
        leak_ts = datetime.now(timezone.utc).isoformat()
        await storage.graph.add_node(GraphNode(
            node_id="in-stint-leak", node_type="memory", label="leaked",
            properties={"agent_id": AGENT_ID, "created_at": leak_ts},
        ))

        result = await wrapper.purge_ephemeral_session(reason="test-iso")

    assert result["graph_nodes"] == 1, (
        f"Only the in-stint leak should be purged from graph_nodes; got "
        f"{result['graph_nodes']}.  Pre-stint ISO-format nodes must "
        f"normalise correctly against the space-format watermark."
    )

    # Confirm the pre-stint node is still in the DB.
    async with AsyncStorage(str(db_path), agent_id=AGENT_ID) as storage:
        survivor = await storage.graph.get_node("pre-stint")
        assert survivor is not None, (
            "Pre-stint NORMAL graph node was destroyed — this is the "
            "T-vs-space lex bug the fix exists to prevent."
        )
        gone = await storage.graph.get_node("in-stint-leak")
        assert gone is None


@pytest.mark.asyncio
async def test_graph_nodes_without_created_at_are_skipped_with_warning(tmp_path, caplog):
    """Conservative coverage: nodes without ``created_at`` can't be proven
    in-window, so they survive — but the operator gets a WARNING with
    the count so the missing provenance can be investigated."""
    import logging as _logging
    from kestrel_sovereign.storage.async_graph_store import GraphNode

    db_path = tmp_path / "kestrel.db"
    async with AsyncStorage(str(db_path), agent_id=AGENT_ID) as storage:
        # Node with no created_at (matches some legacy writers).
        await storage.graph.add_node(GraphNode(
            node_id="legacy", node_type="memory", label="no-timestamp",
            properties={"agent_id": AGENT_ID},
        ))

        wrapper = PrivacyEnforcingStorage(storage, PrivacyMode.NORMAL)
        wrapper.set_privacy_mode(PrivacyMode.EPHEMERAL)
        await asyncio.sleep(1.05)

        with caplog.at_level(_logging.WARNING, logger="kestrel_sovereign.storage.async_graph_store"):
            result = await wrapper.purge_ephemeral_session(reason="test-untimed")

    assert result["graph_nodes"] == 0, (
        "Untimestamped nodes are preserved by the scoped purge — data "
        "preservation wins over leak coverage when provenance is missing."
    )
    assert any(
        "have no properties.created_at" in rec.message for rec in caplog.records
    ), (
        "Operators must see a WARNING listing the count of skipped nodes."
    )

    async with AsyncStorage(str(db_path), agent_id=AGENT_ID) as storage:
        survivor = await storage.graph.get_node("legacy")
        assert survivor is not None


@pytest.mark.asyncio
async def test_other_agents_data_untouched(tmp_path):
    """Per-agent scoping (a holdover from #767) still holds — the scoped
    purge must not touch rows owned by other agents that share the DB.
    """
    db_path = tmp_path / "kestrel.db"
    other = "did:test:other-agent"

    async with AsyncStorage(str(db_path), agent_id=other) as other_store:
        await other_store.conversation.add_conversation("user", "other agent's row")

    async with AsyncStorage(str(db_path), agent_id=AGENT_ID) as storage:
        wrapper = PrivacyEnforcingStorage(storage, PrivacyMode.NORMAL)
        wrapper.set_privacy_mode(PrivacyMode.EPHEMERAL)
        # Watermark is per-second; sleep past the boundary before writing
        # the in-window leak so the test's intent is unambiguous.
        await asyncio.sleep(1.05)
        await storage.conversation.add_conversation("user", "leak from this agent")
        await wrapper.purge_ephemeral_session(reason="test-cross-agent")

    async with AsyncStorage(str(db_path), agent_id=other) as other_store:
        rows = await other_store.conversation.get_full_history_with_ids()
        assert len(rows) == 1, (
            "Other agent's row must not be touched by this agent's "
            "EPHEMERAL leak-purge"
        )
