"""CRUD tests for ``WaitSignalStore`` (Wave 2 of #1860).

The store is the durable dedup/delivery ledger the generic wait reconciler
uses — one row per ``(agent_id, kind, handle)`` it has observed. It's the
generic successor to the per-job ``last_signaled_status`` + ``pending_signal_*``
fields talon_monitor stashed inside ``jobs.json``. Each test pins one of the
reconciler's use cases against a real SQLite backend.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from kestrel_sovereign.storage.async_wait_signal_store import (
    WaitSignalStore,
)


@pytest.fixture
def make_store(tmp_path, sqlite_database_factory):
    async def create(agent_id="did:test:agent"):
        db = await sqlite_database_factory(tmp_path / "wait_store.db")
        return WaitSignalStore(db, agent_id=agent_id)

    return create


@pytest.mark.asyncio
async def test_get_missing_returns_none(make_store):
    store = await make_store()
    assert await store.get("talon", "job-1") is None


@pytest.mark.asyncio
async def test_record_pending_then_get(make_store):
    store = await make_store()
    now = datetime.now(timezone.utc)
    await store.record_pending(
        "talon", "job-1",
        signal_id="sig-abc", target="done", attempts=1, attempt_at=now,
    )
    row = await store.get("talon", "job-1")
    assert row is not None
    assert row.kind == "talon"
    assert row.handle == "job-1"
    assert row.pending_signal_id == "sig-abc"
    assert row.pending_signaled_target == "done"
    assert row.last_delivery_attempts == 1
    # Not yet confirmed → no signaled outcome locked.
    assert row.last_signaled_outcome is None


@pytest.mark.asyncio
async def test_record_pending_preserves_signaled_outcome(make_store):
    """An upsert from record_pending must NOT clobber a previously-locked
    last_signaled_outcome (the row may carry a confirmed delivery for an
    earlier transition)."""
    store = await make_store()
    # First confirm a delivery (locks last_signaled_outcome).
    await store.record_pending(
        "talon", "job-1", signal_id="s1", target="failed", attempts=1,
    )
    await store.record_delivery(
        "talon", "job-1",
        delivery_status="ok", signaled_outcome="failed",
    )
    # Now a fresh transition enqueues again — record_pending must keep
    # the prior signaled outcome around (dedup is the reconciler's call,
    # not the store's).
    await store.record_pending(
        "talon", "job-1", signal_id="s2", target="done", attempts=2,
    )
    row = await store.get("talon", "job-1")
    assert row.last_signaled_outcome == "failed"
    assert row.pending_signal_id == "s2"
    assert row.last_delivery_attempts == 2


@pytest.mark.asyncio
async def test_record_delivery_locks_outcome_and_clears_pending(make_store):
    store = await make_store()
    await store.record_pending(
        "talon", "job-1", signal_id="s1", target="done", attempts=1,
    )
    await store.record_delivery(
        "talon", "job-1",
        delivery_status="ok", signaled_outcome="done",
    )
    row = await store.get("talon", "job-1")
    assert row.last_signaled_outcome == "done"
    assert row.last_delivery_status == "ok"
    # All three pending fields cleared.
    assert row.pending_signal_id is None
    assert row.pending_signaled_target is None
    assert row.pending_signal_enqueued_at is None


@pytest.mark.asyncio
async def test_record_delivery_soft_fail_does_not_lock_outcome(make_store):
    """Soft-fail: omit signaled_outcome so the next tick re-detects +
    retries. Pending is still cleared (the harvest is done)."""
    store = await make_store()
    await store.record_pending(
        "talon", "job-1", signal_id="s1", target="done", attempts=1,
    )
    await store.record_delivery(
        "talon", "job-1",
        delivery_status="dropped_quiet_hours",
        delivery_error="inside quiet window",
    )
    row = await store.get("talon", "job-1")
    assert row.last_signaled_outcome is None
    assert row.last_delivery_status == "dropped_quiet_hours"
    assert row.last_delivery_error == "inside quiet window"
    assert row.pending_signal_id is None


@pytest.mark.asyncio
async def test_list_pending_filters_to_unharvested(make_store):
    store = await make_store()
    await store.record_pending(
        "talon", "job-1", signal_id="s1", target="done", attempts=1,
    )
    await store.record_pending(
        "task", "task-2", signal_id="s2", target="done", attempts=1,
    )
    # Harvest one of them.
    await store.record_delivery(
        "talon", "job-1", delivery_status="ok", signaled_outcome="done",
    )
    pending = await store.list_pending()
    keys = {(p.kind, p.handle) for p in pending}
    assert keys == {("task", "task-2")}


@pytest.mark.asyncio
async def test_clear_pending_nulls_only_pending_fields(make_store):
    store = await make_store()
    await store.record_pending(
        "talon", "job-1", signal_id="s1", target="done", attempts=3,
    )
    await store.clear_pending("talon", "job-1")
    row = await store.get("talon", "job-1")
    assert row.pending_signal_id is None
    assert row.pending_signaled_target is None
    assert row.pending_signal_enqueued_at is None
    # Attempt accounting is preserved.
    assert row.last_delivery_attempts == 3
    # And it drops out of the harvest set.
    assert await store.list_pending() == []


@pytest.mark.asyncio
async def test_start_watch_creates_row(make_store):
    store = await make_store()
    await store.start_watch("task", "task-1")
    row = await store.get("task", "task-1")
    assert row is not None
    assert row.watching == 1
    # Fresh insert zeros the counters.
    assert row.last_delivery_attempts == 0
    assert row.last_signaled_outcome is None
    assert row.pending_signal_id is None


@pytest.mark.asyncio
async def test_start_watch_preserves_existing_fields(make_store):
    """A watch on an existing row must NOT clobber its delivery/pending
    state — start_watch only flips watching=1."""
    store = await make_store()
    await store.record_pending(
        "task", "task-1", signal_id="s1", target="done", attempts=2,
    )
    await store.start_watch("task", "task-1")
    row = await store.get("task", "task-1")
    assert row.watching == 1
    assert row.pending_signal_id == "s1"
    assert row.pending_signaled_target == "done"
    assert row.last_delivery_attempts == 2


@pytest.mark.asyncio
async def test_stop_watch_clears_flag(make_store):
    store = await make_store()
    await store.start_watch("task", "task-1")
    await store.stop_watch("task", "task-1")
    row = await store.get("task", "task-1")
    assert row.watching == 0


@pytest.mark.asyncio
async def test_list_watched_only_active_unsignaled(make_store):
    store = await make_store()
    await store.start_watch("task", "active-1")
    await store.start_watch("talon", "active-2")
    # A stopped watch.
    await store.start_watch("task", "stopped")
    await store.stop_watch("task", "stopped")
    # A watched-but-already-signaled handle drops out (terminal delivered).
    await store.start_watch("task", "done-already")
    await store.record_delivery(
        "task", "done-already", delivery_status="ok", signaled_outcome="done",
    )

    watched = await store.list_watched()
    keys = {(w.kind, w.handle) for w in watched}
    assert keys == {("task", "active-1"), ("talon", "active-2")}


@pytest.mark.asyncio
async def test_watching_column_round_trip(make_store):
    """The watching column survives a write→read round trip on the dataclass."""
    store = await make_store()
    await store.start_watch("task", "task-1")
    row = await store.get("task", "task-1")
    assert isinstance(row.watching, int)
    assert row.watching == 1
    await store.stop_watch("task", "task-1")
    assert (await store.get("task", "task-1")).watching == 0


@pytest.mark.asyncio
async def test_watch_isolation_between_agents(tmp_path, sqlite_database_factory):
    db = await sqlite_database_factory(tmp_path / "shared.db")
    store_a = WaitSignalStore(db, agent_id="did:agent:A")
    store_b = WaitSignalStore(db, agent_id="did:agent:B")
    await store_a.start_watch("task", "task-1")
    assert await store_b.list_watched() == []
    assert len(await store_a.list_watched()) == 1


@pytest.mark.asyncio
async def test_seed_signaled_is_insert_or_ignore(make_store):
    """seed_signaled inserts a confirmed-signaled row only if absent, and
    never clobbers an existing reconciler-managed row (codex Wave 2 P2)."""
    store = await make_store()
    assert await store.seed_signaled("talon", "job-1", "done") is True
    assert (await store.get("talon", "job-1")).last_signaled_outcome == "done"
    # Re-seeding the same handle is a no-op (returns False, value unchanged).
    assert await store.seed_signaled("talon", "job-1", "failed") is False
    assert (await store.get("talon", "job-1")).last_signaled_outcome == "done"


@pytest.mark.asyncio
async def test_seed_does_not_clobber_managed_row(make_store):
    """A later legacy seed must not overwrite a row the reconciler already
    manages (e.g. an in-flight pending row)."""
    store = await make_store()
    await store.record_pending(
        "talon", "job-1", signal_id="s1", target="done", attempts=1,
    )
    assert await store.seed_signaled("talon", "job-1", "failed") is False
    row = await store.get("talon", "job-1")
    assert row.pending_signal_id == "s1"
    assert row.last_signaled_outcome is None


@pytest.mark.asyncio
async def test_agent_id_isolation(tmp_path, sqlite_database_factory):
    """A shared backend must not leak rows between agents (the codex P1
    isolation contract carried over from PendingA2AQuestionStore)."""
    db = await sqlite_database_factory(tmp_path / "shared.db")
    store_a = WaitSignalStore(db, agent_id="did:agent:A")
    store_b = WaitSignalStore(db, agent_id="did:agent:B")
    await store_a.record_pending(
        "talon", "job-1", signal_id="s1", target="done", attempts=1,
    )
    # B sees nothing for the same (kind, handle).
    assert await store_b.get("talon", "job-1") is None
    assert await store_b.list_pending() == []
    # A still sees its own row.
    assert await store_a.get("talon", "job-1") is not None
