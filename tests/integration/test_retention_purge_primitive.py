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
    even when they share the same database file (multi_agent scenario)."""
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


# ---------------------------------------------------------------------------
# Forgetting deletion tier — purge_decayed_episodes (#1674)
#
# Unlike the trash rail above, episode deletion is decay-driven, not age-driven:
# eligibility = (importance-scaled Ebbinghaus strength < delete_threshold) AND
# (age > grace_days). A high-importance episode decays slower and outlives a
# throwaway one of the same age; age alone never deletes anything.
# ---------------------------------------------------------------------------

from kestrel_sovereign.storage.async_graph_store import GraphNode


async def _add_episode(
    storage, episode_id: str, created_at: datetime, *,
    importance: float = 0.5, agent_id=AGENT_ID,
):
    """Insert a memory_episodes row + its paired KG node (node_id == episode id),
    matching what memory_consolidator writes."""
    # Match the consolidator: datetime.now(tz=utc).isoformat() -> "...T...+00:00".
    iso = created_at.isoformat()
    await storage.db.execute(
        """INSERT INTO memory_episodes
           (id, agent_id, title, summary, created_at, importance)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (episode_id, agent_id, f"title-{episode_id}", "summary", iso, importance),
    )
    await storage.add_node(GraphNode(
        node_id=episode_id,
        node_type="episode",
        label=f"title-{episode_id}",
        properties={"source": "consolidator"},
    ))


@pytest.mark.asyncio
async def test_purge_decayed_removes_faded_episode_and_paired_nodes(tmp_path):
    db = tmp_path / "kestrel.db"
    async with AsyncStorage(str(db), agent_id=AGENT_ID) as storage:
        now = datetime.now(timezone.utc)
        # Low-importance + very old → strength ≈ 0.5^(400/36) ≪ 0.02 → eligible.
        await _add_episode(storage, "faded", now - timedelta(days=400), importance=0.1)
        # Recent → within grace, never eligible.
        await _add_episode(storage, "recent", now - timedelta(days=10), importance=0.1)
        # An edge touching the faded node must be scrubbed (no orphan).
        await storage.graph.add_edge("faded", "recent", "followed_by", {})

        purged = await storage.purge_decayed_episodes(
            delete_threshold=0.02, grace_days=90,
        )

        assert purged == 1
        remaining = await storage.db.fetchall("SELECT id FROM memory_episodes")
        assert {r[0] for r in remaining} == {"recent"}
        # Paired KG node gone; recent node survives.
        assert await storage.get_node("faded") is None
        assert await storage.get_node("recent") is not None
        # The edge touching the purged node was scrubbed (no orphan edge).
        assert await storage.graph.get_edges("recent") == []


@pytest.mark.asyncio
async def test_purge_decayed_is_importance_aware_not_age_based(tmp_path):
    """The core of #1674: two episodes of the SAME age, different importance —
    the load-bearing one survives, the throwaway one is forgotten. An age-based
    sweep would (wrongly) delete both."""
    db = tmp_path / "kestrel.db"
    async with AsyncStorage(str(db), agent_id=AGENT_ID) as storage:
        now = datetime.now(timezone.utc)
        # Same age (400d). importance 0.1 → eff half-life 36d → strength ≪ 0.02.
        # importance 1.0 → eff half-life 90d → strength ≈ 0.046 > 0.02.
        await _add_episode(storage, "throwaway", now - timedelta(days=400), importance=0.1)
        await _add_episode(storage, "load-bearing", now - timedelta(days=400), importance=1.0)

        purged = await storage.purge_decayed_episodes(
            delete_threshold=0.02, grace_days=90,
        )

        assert purged == 1
        remaining = {r[0] for r in await storage.db.fetchall("SELECT id FROM memory_episodes")}
        assert remaining == {"load-bearing"}  # high-importance episode outlives same-age throwaway


@pytest.mark.asyncio
async def test_purge_decayed_grace_window_is_an_independent_gate(tmp_path):
    """Even a fully-faded episode (strength ≈ 0) is kept while it is younger
    than grace_days — there is always a minimum lifetime. Shrinking grace below
    the age then makes the same episode eligible, proving grace gates on its own.
    Uses a 1-day half-life so decay is unambiguous well before normal windows."""
    db = tmp_path / "kestrel.db"
    async with AsyncStorage(str(db), agent_id=AGENT_ID) as storage:
        now = datetime.now(timezone.utc)
        # 30 days old, half_life 1d → strength ≈ 0.5^25 ≈ 3e-8 ≪ 0.02.
        await _add_episode(storage, "faded-but-young", now - timedelta(days=30), importance=0.1)

        # grace 90 > age 30 → kept despite near-zero strength.
        kept = await storage.purge_decayed_episodes(
            delete_threshold=0.02, grace_days=90, half_life_days=1,
        )
        assert kept == 0
        assert len(await storage.db.fetchall("SELECT id FROM memory_episodes")) == 1

        # grace 20 < age 30 AND strength < threshold → now eligible.
        purged = await storage.purge_decayed_episodes(
            delete_threshold=0.02, grace_days=20, half_life_days=1,
        )
        assert purged == 1
        assert await storage.db.fetchall("SELECT id FROM memory_episodes") == []


@pytest.mark.asyncio
async def test_purge_decayed_caps_a_single_sweep(tmp_path):
    db = tmp_path / "kestrel.db"
    async with AsyncStorage(str(db), agent_id=AGENT_ID) as storage:
        now = datetime.now(timezone.utc)
        for i in range(5):
            await _add_episode(storage, f"faded-{i}", now - timedelta(days=300), importance=0.1)
        await _add_episode(storage, "recent", now - timedelta(days=10), importance=0.1)

        # Cap at 2 — only the two oldest eligible rows go this sweep.
        purged = await storage.purge_decayed_episodes(
            delete_threshold=0.02, grace_days=90, max_rows=2,
        )
        assert purged == 2
        remaining = await storage.db.fetchall("SELECT id FROM memory_episodes")
        assert len(remaining) == 4  # 5 faded - 2 purged + 1 recent
        assert "recent" in {r[0] for r in remaining}


@pytest.mark.asyncio
async def test_purge_decayed_scopes_to_agent(tmp_path):
    db = tmp_path / "kestrel.db"
    async with AsyncStorage(str(db), agent_id=AGENT_ID) as storage:
        now = datetime.now(timezone.utc)
        await _add_episode(storage, "mine", now - timedelta(days=400), importance=0.1)
        await _add_episode(storage, "other", now - timedelta(days=400), importance=0.1,
                           agent_id=OTHER_AGENT_ID)

        purged = await storage.purge_decayed_episodes(
            delete_threshold=0.02, grace_days=90,
        )

        assert purged == 1  # only this agent's episode
        remaining = {r[0] for r in await storage.db.fetchall("SELECT id FROM memory_episodes")}
        assert remaining == {"other"}


@pytest.mark.asyncio
async def test_purge_decayed_keeps_row_when_node_delete_fails(tmp_path):
    """Regression (#1674): if a paired KG node delete fails, the episode row
    must NOT be deleted — otherwise we create the orphan node the ordering is
    meant to prevent. The episode is retried on the next sweep."""
    db = tmp_path / "kestrel.db"
    async with AsyncStorage(str(db), agent_id=AGENT_ID) as storage:
        now = datetime.now(timezone.utc)
        await _add_episode(storage, "good", now - timedelta(days=400), importance=0.1)
        await _add_episode(storage, "bad", now - timedelta(days=400), importance=0.1)

        real_delete = storage.graph.delete_node

        async def flaky(node_id):
            if node_id == "bad":
                raise RuntimeError("graph store unavailable")
            return await real_delete(node_id)

        storage.graph.delete_node = flaky

        purged = await storage.purge_decayed_episodes(
            delete_threshold=0.02, grace_days=90,
        )

        assert purged == 1  # only "good" removed
        remaining = {r[0] for r in await storage.db.fetchall("SELECT id FROM memory_episodes")}
        assert remaining == {"bad"}  # row kept — its node could not be deleted
        assert await storage.get_node("bad") is not None  # node not orphaned


@pytest.mark.asyncio
async def test_purge_decayed_non_positive_cap_purges_nothing(tmp_path):
    """Regression (#1674): max_rows<=0 must purge nothing, not fall through to
    an unbounded sweep."""
    db = tmp_path / "kestrel.db"
    async with AsyncStorage(str(db), agent_id=AGENT_ID) as storage:
        now = datetime.now(timezone.utc)
        await _add_episode(storage, "faded", now - timedelta(days=400), importance=0.1)

        assert await storage.purge_decayed_episodes(
            delete_threshold=0.02, grace_days=90, max_rows=0) == 0
        assert await storage.purge_decayed_episodes(
            delete_threshold=0.02, grace_days=90, max_rows=-1) == 0
        # The episode (and its node) survive.
        assert len(await storage.db.fetchall("SELECT id FROM memory_episodes")) == 1
        assert await storage.get_node("faded") is not None


@pytest.mark.asyncio
async def test_privacy_wrapper_exposes_purge_decayed_episodes(tmp_path):
    """The consolidation pass reads ``agent.storage.purge_decayed_episodes``
    where ``agent.storage`` is the ``PrivacyEnforcingStorage`` wrapper. Pin the
    delegator so it can't be dropped without breaking a test."""
    from kestrel_sovereign.privacy import PrivacyMode
    from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage

    db = tmp_path / "kestrel.db"
    async with AsyncStorage(str(db), agent_id=AGENT_ID) as underlying:
        wrapper = PrivacyEnforcingStorage(underlying, PrivacyMode.NORMAL)
        assert hasattr(wrapper, "purge_decayed_episodes"), (
            "privacy wrapper must expose purge_decayed_episodes so the "
            "consolidation forgetting tier can find it"
        )

        now = datetime.now(timezone.utc)
        await _add_episode(underlying, "faded", now - timedelta(days=400), importance=0.1)
        purged = await wrapper.purge_decayed_episodes(
            delete_threshold=0.02, grace_days=90,
        )
        assert purged == 1
