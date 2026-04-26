"""Integration tests for the retention-purge primitive (#764).

Hits a real SQLite database so the SQL predicate, LIMIT subquery, and
``deleted_at`` boundary semantics are exercised end-to-end. The
janitor scheduling layer is unit-tested separately in
``tests/unit/test_retention_janitor.py``.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from kestrel_sovereign.storage import AsyncStorage


AGENT_ID = "did:test:retention"
OTHER_AGENT_ID = "did:test:other"


def _stamp(store, row_id: int, when: datetime) -> None:
    """Helper: forcibly set ``deleted_at`` on a row to simulate aged trash.

    The purge primitive only cares about timestamp order, so we don't
    need a real elapsed clock — we just write a back-dated string the
    SQL ``<`` comparator will sort correctly.
    """
    iso = when.replace(tzinfo=None).isoformat(sep=" ")
    return store.db.execute_commit(
        "UPDATE conversation_history SET deleted_at = ? WHERE id = ?",
        (iso, row_id),
    )


@pytest.mark.asyncio
async def test_purge_trash_older_than_destroys_aged_rows(tmp_path):
    db = tmp_path / "kestrel.db"
    async with AsyncStorage(str(db), agent_id=AGENT_ID) as storage:
        await storage.conversation.add_conversation("user", "old trash")
        rows = await storage.conversation.get_full_history_with_ids()
        await storage.conversation.delete_message(rows[0]["id"])
        # Back-date the deleted_at to 60 days ago
        long_ago = datetime.now(timezone.utc) - timedelta(days=60)
        await _stamp(storage.conversation, rows[0]["id"], long_ago)

        cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).replace(tzinfo=None).isoformat(sep=" ")
        purged = await storage.purge_trash_older_than(cutoff)

        assert purged == 1
        gone = await storage.conversation.get_full_history_with_ids(include_deleted=True)
        assert gone == []


@pytest.mark.asyncio
async def test_purge_does_not_touch_rows_within_window(tmp_path):
    """A row whose deleted_at is more recent than the cutoff stays in
    Trash. The retention janitor must not be aggressive about edges —
    if the user's policy says 30 days, day-29 rows survive."""
    db = tmp_path / "kestrel.db"
    async with AsyncStorage(str(db), agent_id=AGENT_ID) as storage:
        await storage.conversation.add_conversation("user", "recent trash")
        rows = await storage.conversation.get_full_history_with_ids()
        await storage.conversation.delete_message(rows[0]["id"])
        five_days_ago = datetime.now(timezone.utc) - timedelta(days=5)
        await _stamp(storage.conversation, rows[0]["id"], five_days_ago)

        cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).replace(tzinfo=None).isoformat(sep=" ")
        purged = await storage.purge_trash_older_than(cutoff)

        assert purged == 0
        # Row still recoverable from trash
        in_trash = await storage.conversation.get_full_history_with_ids(only_deleted=True)
        assert len(in_trash) == 1


@pytest.mark.asyncio
async def test_purge_never_touches_live_rows(tmp_path):
    """deleted_at IS NULL is the safety rail. A live row that's been
    around for 10 years still doesn't get purged — the janitor's
    contract is "age out trash," not "compact history."
    """
    db = tmp_path / "kestrel.db"
    async with AsyncStorage(str(db), agent_id=AGENT_ID) as storage:
        # Live row, no deleted_at
        await storage.conversation.add_conversation("user", "long-lived live row")
        await storage.conversation.add_conversation("assistant", "another live one")

        # Cutoff in the future — would sweep everything if the rail
        # weren't checking deleted_at IS NOT NULL.
        far_future = (datetime.now(timezone.utc) + timedelta(days=365)).replace(tzinfo=None).isoformat(sep=" ")
        purged = await storage.purge_trash_older_than(far_future)

        assert purged == 0
        live = await storage.conversation.get_full_history_with_ids()
        assert len(live) == 2


@pytest.mark.asyncio
async def test_per_agent_scoping(tmp_path):
    """Purging AGENT_ID's old trash must not touch OTHER_AGENT_ID's data,
    even when they share the same database file (rookery scenario)."""
    db = tmp_path / "kestrel.db"

    long_ago = datetime.now(timezone.utc) - timedelta(days=60)
    cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).replace(tzinfo=None).isoformat(sep=" ")

    async with AsyncStorage(str(db), agent_id=OTHER_AGENT_ID) as other:
        await other.conversation.add_conversation("user", "other trash")
        rows = await other.conversation.get_full_history_with_ids()
        await other.conversation.delete_message(rows[0]["id"])
        await _stamp(other.conversation, rows[0]["id"], long_ago)

    async with AsyncStorage(str(db), agent_id=AGENT_ID) as storage:
        await storage.conversation.add_conversation("user", "my trash")
        rows = await storage.conversation.get_full_history_with_ids()
        await storage.conversation.delete_message(rows[0]["id"])
        await _stamp(storage.conversation, rows[0]["id"], long_ago)

        purged = await storage.purge_trash_older_than(cutoff)
        assert purged == 1

    # Other agent's old trash is untouched
    async with AsyncStorage(str(db), agent_id=OTHER_AGENT_ID) as other:
        in_trash = await other.conversation.get_full_history_with_ids(only_deleted=True)
        assert len(in_trash) == 1


@pytest.mark.asyncio
async def test_max_rows_caps_a_single_sweep(tmp_path):
    """Per-sweep cap protects the writer thread from a stall when an
    agent suddenly has a huge backlog. The remainder drains on the
    next tick."""
    db = tmp_path / "kestrel.db"
    async with AsyncStorage(str(db), agent_id=AGENT_ID) as storage:
        long_ago = datetime.now(timezone.utc) - timedelta(days=60)
        for i in range(10):
            await storage.conversation.add_conversation("user", f"trash {i}")
        rows = await storage.conversation.get_full_history_with_ids()
        for r in rows:
            await storage.conversation.delete_message(r["id"])
            await _stamp(storage.conversation, r["id"], long_ago)

        cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).replace(tzinfo=None).isoformat(sep=" ")

        # Cap at 3 — only 3 rows go in this sweep
        first = await storage.purge_trash_older_than(cutoff, max_rows=3)
        assert first == 3
        remaining = await storage.conversation.get_full_history_with_ids(only_deleted=True)
        assert len(remaining) == 7

        # Next sweep gets the rest
        second = await storage.purge_trash_older_than(cutoff, max_rows=100)
        assert second == 7
        remaining = await storage.conversation.get_full_history_with_ids(only_deleted=True)
        assert remaining == []


@pytest.mark.asyncio
async def test_idempotent_when_nothing_to_purge(tmp_path):
    """Empty Trash → zero rows. The runner calls this every tick on
    every agent; it must be cheap on the no-op path."""
    db = tmp_path / "kestrel.db"
    async with AsyncStorage(str(db), agent_id=AGENT_ID) as storage:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).replace(tzinfo=None).isoformat(sep=" ")
        purged = await storage.purge_trash_older_than(cutoff)
        assert purged == 0


@pytest.mark.asyncio
async def test_max_rows_zero_or_negative_is_a_safe_noop(tmp_path):
    """Defensive guard — a misconfigured cap of 0 must NOT scrub
    everything. It does nothing instead.
    """
    db = tmp_path / "kestrel.db"
    async with AsyncStorage(str(db), agent_id=AGENT_ID) as storage:
        await storage.conversation.add_conversation("user", "trash")
        rows = await storage.conversation.get_full_history_with_ids()
        await storage.conversation.delete_message(rows[0]["id"])
        long_ago = datetime.now(timezone.utc) - timedelta(days=60)
        await _stamp(storage.conversation, rows[0]["id"], long_ago)

        cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).replace(tzinfo=None).isoformat(sep=" ")
        purged = await storage.purge_trash_older_than(cutoff, max_rows=0)
        assert purged == 0
        purged = await storage.purge_trash_older_than(cutoff, max_rows=-5)
        assert purged == 0
        # Trash row still there
        in_trash = await storage.conversation.get_full_history_with_ids(only_deleted=True)
        assert len(in_trash) == 1


@pytest.mark.asyncio
async def test_privacy_wrapper_exposes_purge_trash_older_than(tmp_path):
    """Regression — the cron handler reads
    ``agent.storage.purge_trash_older_than`` where ``agent.storage``
    is the ``PrivacyEnforcingStorage`` wrapper. Smoke testing caught
    the missing delegator on the wrapper (the task skipped silently
    on every tick). This test pins the wrapper's surface so the
    delegator can't be deleted without breaking a test.
    """
    from kestrel_sovereign.privacy import PrivacyMode
    from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage

    db = tmp_path / "kestrel.db"
    async with AsyncStorage(str(db), agent_id=AGENT_ID) as underlying:
        wrapper = PrivacyEnforcingStorage(underlying, PrivacyMode.NORMAL)

        assert hasattr(wrapper, "purge_trash_older_than"), (
            "privacy wrapper must expose purge_trash_older_than so the "
            "trash_retention cron handler can find it"
        )

        # Seed an aged soft-deleted row
        await underlying.conversation.add_conversation("user", "aged")
        rows = await underlying.conversation.get_full_history_with_ids()
        await underlying.conversation.delete_message(rows[0]["id"])
        long_ago = datetime.now(timezone.utc) - timedelta(days=60)
        await _stamp(underlying.conversation, rows[0]["id"], long_ago)

        cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).replace(tzinfo=None).isoformat(sep=" ")
        purged = await wrapper.purge_trash_older_than(cutoff)
        assert purged == 1


@pytest.mark.asyncio
async def test_privacy_wrapper_purge_trash_works_in_ephemeral_mode(tmp_path):
    """Aging out already-soft-deleted rows from a prior NORMAL stint
    must work even when the agent is currently in EPHEMERAL mode —
    EPHEMERAL gates new persistent writes, not retention sweeps.
    """
    from kestrel_sovereign.privacy import PrivacyMode
    from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage

    db = tmp_path / "kestrel.db"
    async with AsyncStorage(str(db), agent_id=AGENT_ID) as underlying:
        # Seed in NORMAL state, then flip the wrapper to EPHEMERAL.
        await underlying.conversation.add_conversation("user", "soft-deleted in NORMAL")
        rows = await underlying.conversation.get_full_history_with_ids()
        await underlying.conversation.delete_message(rows[0]["id"])
        long_ago = datetime.now(timezone.utc) - timedelta(days=60)
        await _stamp(underlying.conversation, rows[0]["id"], long_ago)

        wrapper = PrivacyEnforcingStorage(underlying, PrivacyMode.EPHEMERAL)
        cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).replace(tzinfo=None).isoformat(sep=" ")
        purged = await wrapper.purge_trash_older_than(cutoff)
        assert purged == 1
