"""CRUD tests for ``WaitSignalStore`` (Wave 2 of #1860).

The store is the durable dedup/delivery ledger the generic wait reconciler
uses — one row per ``(agent_id, kind, handle)`` it has observed. It's the
generic successor to the per-job ``last_signaled_status`` + ``pending_signal_*``
fields talon_monitor stashed inside ``jobs.json``. Each test pins one of the
reconciler's use cases against a real SQLite backend.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from kestrel_sovereign.storage.async_wait_signal_store import (
    REKEY_IDENTITY_ADOPTED,
    REKEY_RECLASSIFIED,
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
    state — start_watch flips watching=1 and records the wake it is armed
    over, here the one in flight."""
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
    assert row.last_signaled_outcome is None
    assert row.watch_baseline == "done"


@pytest.mark.asyncio
async def test_start_watch_rearms_a_watch_that_already_fired(make_store):
    """#3399: a watch that fired used to stay retired forever while the
    ``wait`` tool still acknowledged ``watching: true``. Registering again
    re-arms it over the event already delivered."""
    store = await make_store()
    await store.start_watch("ci", "o/r#1")
    await store.record_delivery(
        "ci", "o/r#1", delivery_status="ok_queued",
        signaled_outcome="event:run-1",
    )
    assert await store.list_watched() == [], "the first wake spent the watch"

    await store.start_watch("ci", "o/r#1")

    [row] = await store.list_watched()
    assert (row.kind, row.handle) == ("ci", "o/r#1")
    assert row.watch_baseline == "event:run-1"
    assert row.last_signaled_outcome == "event:run-1", (
        "re-arming must not forget what was delivered — the reconciler "
        "dedups the re-armed watch against it"
    )


@pytest.mark.asyncio
async def test_a_rearmed_watch_retires_on_its_next_delivery(make_store):
    store = await make_store()
    await store.record_delivery(
        "ci", "o/r#1", delivery_status="ok_queued",
        signaled_outcome="event:run-1",
    )
    await store.start_watch("ci", "o/r#1")
    # Re-registering while still live is idempotent.
    await store.start_watch("ci", "o/r#1")
    assert len(await store.list_watched()) == 1

    await store.record_delivery(
        "ci", "o/r#1", delivery_status="ok_queued",
        signaled_outcome="event:run-2",
    )

    assert await store.list_watched() == []
    row = await store.get("ci", "o/r#1")
    assert row.watching == 0, "the watch fired, so it is disarmed"
    assert row.watch_baseline == "event:run-1"


@pytest.mark.asyncio
async def test_rearming_inside_the_wake_turn_arms_over_the_wake_in_flight(
    make_store,
):
    """The agent re-runs the job and re-registers from INSIDE the wake that
    announced the failure, before the reconciler harvests it. Arming over the
    last delivered token would let that harvest disarm the new watch — the
    original bug in a different shape."""
    store = await make_store()
    await store.start_watch("ci", "o/r#1")
    await store.record_pending(
        "ci", "o/r#1", signal_id="s1", target="event:run-1", attempts=1,
    )

    await store.start_watch("ci", "o/r#1")
    assert (await store.get("ci", "o/r#1")).watch_baseline == "event:run-1"

    await store.record_delivery(
        "ci", "o/r#1", delivery_status="ok_queued",
        signaled_outcome="event:run-1",
    )
    [row] = await store.list_watched()
    assert row.watching == 1, "the wake it was armed over does not spend it"


@pytest.mark.asyncio
async def test_a_soft_failed_wake_keeps_a_rearmed_watch_polled(make_store):
    """A soft fail leaves ``last_signaled_outcome`` where it was. The watch
    must stay in the poll set so the undelivered wake is retried."""
    store = await make_store()
    await store.record_delivery(
        "ci", "o/r#1", delivery_status="ok_queued",
        signaled_outcome="event:run-1",
    )
    await store.start_watch("ci", "o/r#1")
    await store.record_pending(
        "ci", "o/r#1", signal_id="s2", target="event:run-2", attempts=1,
    )
    await store.record_delivery(
        "ci", "o/r#1", delivery_status="rate_limited",
    )

    [row] = await store.list_watched()
    assert row.last_signaled_outcome == "event:run-1"


@pytest.mark.asyncio
async def test_adopt_signaled_token_rekeys_and_carries_the_watch(make_store):
    """Re-keying a pre-identity delivery to its identity must keep a watch
    armed over the old token live, carry the delivered transition's attempt
    target with it, and is a compare-and-set on it."""
    store = await make_store()
    await store.record_pending(
        "ci", "o/r#1", signal_id="sig-1", target="failed", attempts=1,
    )
    await store.record_delivery(
        "ci", "o/r#1", delivery_status="ok_queued", signaled_outcome="failed",
    )
    await store.start_watch("ci", "o/r#1")

    assert await store.adopt_signaled_token(
        "ci", "o/r#1", previous="failed", token="event:checks@abc:1",
        reason=REKEY_IDENTITY_ADOPTED,
    ) is True

    [row] = await store.list_watched()
    assert row.last_signaled_outcome == "event:checks@abc:1"
    assert row.watch_baseline == "event:checks@abc:1"
    assert row.attempts_signaled_target == "event:checks@abc:1", (
        "left on the old string, a later event would match it as a retry"
    )

    # Stale ``previous``: a delivery recorded since the read wins.
    assert await store.adopt_signaled_token(
        "ci", "o/r#1", previous="failed", token="event:other",
        reason=REKEY_IDENTITY_ADOPTED,
    ) is False
    assert (await store.get("ci", "o/r#1")).last_signaled_outcome == (
        "event:checks@abc:1"
    )
    # Only the re-key that happened is audited.
    [rekey] = await store.list_rekeys("ci", "o/r#1")
    assert (rekey.previous_token, rekey.token, rekey.reason) == (
        "failed", "event:checks@abc:1", REKEY_IDENTITY_ADOPTED,
    )


@pytest.mark.asyncio
async def test_adopt_leaves_a_spent_watch_spent(make_store):
    """A watch that already fired (baseline is not the re-keyed token) must
    not be revived by the re-key."""
    store = await make_store()
    await store.start_watch("ci", "o/r#1")
    await store.record_delivery(
        "ci", "o/r#1", delivery_status="ok_queued", signaled_outcome="failed",
    )

    await store.adopt_signaled_token(
        "ci", "o/r#1", previous="failed", token="event:checks@abc:1",
        reason=REKEY_IDENTITY_ADOPTED,
    )

    assert await store.list_watched() == []
    assert (await store.get("ci", "o/r#1")).watch_baseline is None


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


@pytest.mark.asyncio
async def test_attempt_provenance_columns_migrate_onto_a_legacy_table(
    tmp_path, sqlite_database_factory,
):
    """``wait_signal_state`` is CREATE TABLE IF NOT EXISTS, so a database that
    predates #3105 never receives the two attempt-provenance columns from the
    schema — only the idempotent ALTERs put them there. This builds the real
    pre-#3105 table, including a row mid-retry, and asserts the migration lands
    without disturbing it.

    The legacy row is the case worth pinning: its three recorded attempts
    belonged to some transition the old schema could not name, so it migrates
    to an empty ``attempts_signaled_target``. The next detect therefore reads
    as a NEW transition and restarts the counter — a bounded, one-time loss of
    cap history in the direction of delivering a wake rather than suppressing
    one, which is the safe direction."""
    import sqlite3

    db_path = tmp_path / "legacy_wait_state.db"
    legacy = sqlite3.connect(db_path)
    legacy.executescript(
        """
        CREATE TABLE wait_signal_state (
            agent_id TEXT NOT NULL DEFAULT '',
            kind TEXT NOT NULL,
            handle TEXT NOT NULL,
            last_signaled_outcome TEXT,
            last_delivery_status TEXT,
            last_surface_status TEXT,
            last_delivery_error TEXT,
            last_delivery_attempts INTEGER NOT NULL DEFAULT 0,
            last_delivery_attempt_at TIMESTAMP,
            pending_signal_id TEXT,
            pending_signaled_target TEXT,
            pending_signal_enqueued_at TIMESTAMP,
            watching INTEGER NOT NULL DEFAULT 0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (agent_id, kind, handle)
        );
        """
    )
    legacy.execute(
        "INSERT INTO wait_signal_state "
        "(agent_id, kind, handle, last_delivery_attempts, last_delivery_attempt_at) "
        "VALUES ('did:legacy', 'talon', 'h-old', 3, '2026-08-01 00:00:00')"
    )
    legacy.commit()
    columns_before = {r[1] for r in legacy.execute(
        "PRAGMA table_info(wait_signal_state)"
    )}
    legacy.close()
    assert "attempts_signaled_target" not in columns_before
    assert "last_attempt_started_at" not in columns_before

    database = await sqlite_database_factory(db_path)
    rows = await database.fetchall(
        "SELECT last_delivery_attempts, attempts_signaled_target, "
        "last_attempt_started_at FROM wait_signal_state WHERE handle = ?",
        ("h-old",),
    )

    assert len(rows) == 1, "the migration must not drop or duplicate the row"
    attempts, target, started_at = rows[0][0], rows[0][1], rows[0][2]
    assert attempts == 3, "an existing attempt count survives the ALTER"
    assert target == "", "legacy attempts name no transition, so they name none"
    assert started_at is None, "no dispatch time was ever recorded for it"


@pytest.mark.asyncio
async def test_watch_baseline_migrates_onto_a_legacy_table(
    tmp_path, sqlite_database_factory,
):
    """``watch_baseline`` only reaches an existing database through the ALTER
    (#3399). A watch that fired before the upgrade reads NULL and stays
    retired until it is registered again, so the upgrade replays nothing; a
    watch that never fired stays live."""
    import sqlite3

    db_path = tmp_path / "legacy_wait_state.db"
    legacy = sqlite3.connect(db_path)
    legacy.executescript(
        """
        CREATE TABLE wait_signal_state (
            agent_id TEXT NOT NULL DEFAULT '',
            kind TEXT NOT NULL,
            handle TEXT NOT NULL,
            last_signaled_outcome TEXT,
            last_delivery_status TEXT,
            last_surface_status TEXT,
            last_delivery_error TEXT,
            last_delivery_attempts INTEGER NOT NULL DEFAULT 0,
            last_delivery_attempt_at TIMESTAMP,
            attempts_signaled_target TEXT NOT NULL DEFAULT '',
            last_attempt_started_at TIMESTAMP,
            delivery_deferred_until TIMESTAMP,
            delivery_deferrals INTEGER NOT NULL DEFAULT 0,
            pending_signal_id TEXT,
            pending_signaled_target TEXT,
            pending_signal_enqueued_at TIMESTAMP,
            watching INTEGER NOT NULL DEFAULT 0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (agent_id, kind, handle)
        );
        INSERT INTO wait_signal_state
            (agent_id, kind, handle, last_signaled_outcome, watching)
        VALUES ('did:legacy', 'ci', 'o/r#1', 'failed', 1);
        INSERT INTO wait_signal_state (agent_id, kind, handle, watching)
        VALUES ('did:legacy', 'ci', 'o/r#2', 1);
        """
    )
    legacy.commit()
    legacy.close()

    database = await sqlite_database_factory(db_path)
    store = WaitSignalStore(database, agent_id="did:legacy")

    assert (await store.get("ci", "o/r#1")).watch_baseline is None
    assert [w.handle for w in await store.list_watched()] == ["o/r#2"]

    await store.start_watch("ci", "o/r#1")
    assert {w.handle for w in await store.list_watched()} == {"o/r#1", "o/r#2"}
    assert (await store.get("ci", "o/r#1")).watch_baseline == "failed"


# ---------------------------------------------------------------------------
# Last successful delivery and the re-key audit (#3390)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_only_a_successful_delivery_dates_the_handle(make_store):
    """``last_delivered_at`` is the one column a later transition's attempts
    never rewrite. Hard-fail and retry-cap locks also lock a token, but they
    are not deliveries, so they must not date one. A delivery is dated by
    the wake's own time, not by the harvest that recorded it."""
    store = await make_store()
    await store.record_pending(
        "talon", "job-1", signal_id="s1", target="done:complete", attempts=1,
    )
    await store.record_delivery(
        "talon", "job-1", delivery_status="dropped_quiet_hours",
    )
    assert (await store.get("talon", "job-1")).last_delivered_at is None

    await store.record_delivery(
        "talon", "job-1", delivery_status="dropped_validation",
        signaled_outcome="done:complete",
    )
    assert (await store.get("talon", "job-1")).last_delivered_at is None

    delivered_at = datetime(2026, 8, 24, 12, 0, tzinfo=timezone.utc)
    harvested_at = delivered_at + timedelta(minutes=5)
    await store.record_delivery(
        "talon", "job-1", delivery_status="ok_queued",
        signaled_outcome="done:complete", attempt_at=harvested_at,
        delivered_at=delivered_at,
    )
    row = await store.get("talon", "job-1")
    assert row.delivered_at_utc() == delivered_at, "the wake's time, not the harvest's"
    assert datetime.fromisoformat(row.last_delivery_attempt_at) == (
        harvested_at.replace(tzinfo=None)
    )

    # A later transition's dispatch and failures leave it alone...
    await store.record_pending(
        "talon", "job-1", signal_id="s2", target="failed:failed", attempts=1,
    )
    await store.record_delivery(
        "talon", "job-1", delivery_status="dropped_rate_limit",
    )
    # ...and so does a re-key of the delivered token.
    await store.adopt_signaled_token(
        "talon", "job-1", previous="done:complete", token="partial:complete",
        reason=REKEY_RECLASSIFIED,
    )
    assert (await store.get("talon", "job-1")).delivered_at_utc() == delivered_at


@pytest.mark.asyncio
async def test_a_successful_delivery_must_lock_its_token(make_store):
    store = await make_store()
    with pytest.raises(ValueError, match="signaled_outcome"):
        await store.record_delivery(
            "talon", "job-1", delivery_status="ok_queued",
            delivered_at=datetime.now(timezone.utc),
        )
    assert await store.get("talon", "job-1") is None


@pytest.mark.asyncio
async def test_a_rekey_and_its_audit_row_land_together(make_store):
    """The re-key changes delivered state without waking anyone, so its
    audit row is the only record of it. Both are one transaction: an audit
    write that fails leaves the delivered token where it was."""
    store = await make_store()
    await store.record_delivery(
        "talon", "job-1", delivery_status="ok_queued",
        signaled_outcome="done:complete",
        delivered_at=datetime.now(timezone.utc),
    )
    await store._db.execute(
        "ALTER TABLE wait_signal_rekeys RENAME TO wait_signal_rekeys_moved"
    )

    with pytest.raises(Exception):
        await store.adopt_signaled_token(
            "talon", "job-1", previous="done:complete",
            token="partial:complete", reason=REKEY_RECLASSIFIED,
        )

    assert (await store.get("talon", "job-1")).last_signaled_outcome == (
        "done:complete"
    ), "an unaudited re-key must not survive"


@pytest.mark.asyncio
async def test_rekey_audit_rows_are_scoped_to_their_agent(
    tmp_path, sqlite_database_factory,
):
    db = await sqlite_database_factory(tmp_path / "shared.db")
    store_a = WaitSignalStore(db, agent_id="did:test:a")
    store_b = WaitSignalStore(db, agent_id="did:test:b")
    for store in (store_a, store_b):
        await store.record_delivery(
            "talon", "job-1", delivery_status="ok_queued",
            signaled_outcome="done:complete",
            delivered_at=datetime.now(timezone.utc),
        )
    await store_a.adopt_signaled_token(
        "talon", "job-1", previous="done:complete", token="partial:complete",
        reason=REKEY_RECLASSIFIED,
    )

    assert [r.reason for r in await store_a.list_rekeys()] == [REKEY_RECLASSIFIED]
    assert await store_b.list_rekeys() == []
    assert (await store_b.get("talon", "job-1")).last_signaled_outcome == (
        "done:complete"
    )


@pytest.mark.asyncio
async def test_last_delivered_at_is_backfilled_only_for_delivered_rows(
    tmp_path, sqlite_database_factory,
):
    """A pre-#3390 database gets ``last_delivered_at`` from the migration,
    filled from ``last_attempt_started_at`` — the delivered wake's dispatch —
    only where that still dates a successful delivery: the locked token was
    delivered (a persisted ``ok``/``coalesced`` status, composed or bare) and
    nothing has been dispatched or retried since.

    Never from ``last_delivery_attempt_at``. That is the harvest, a tick
    after the wake, and an event that finished in between would read as older
    than the wake and be announced as a replay (#3390 review). So a row with
    no dispatch time (delivered before #3105), one with a wake pending, and
    one mid-retry, hard-failed, or never delivered all stay NULL, which never
    labels a wake a replay."""
    import sqlite3

    db_path = tmp_path / "legacy_wait_state.db"
    legacy = sqlite3.connect(db_path)
    legacy.executescript(
        """
        CREATE TABLE wait_signal_state (
            agent_id TEXT NOT NULL DEFAULT '',
            kind TEXT NOT NULL,
            handle TEXT NOT NULL,
            last_signaled_outcome TEXT,
            last_delivery_status TEXT,
            last_surface_status TEXT,
            last_delivery_error TEXT,
            last_delivery_attempts INTEGER NOT NULL DEFAULT 0,
            last_delivery_attempt_at TIMESTAMP,
            attempts_signaled_target TEXT NOT NULL DEFAULT '',
            last_attempt_started_at TIMESTAMP,
            delivery_deferred_until TIMESTAMP,
            delivery_deferrals INTEGER NOT NULL DEFAULT 0,
            pending_signal_id TEXT,
            pending_signaled_target TEXT,
            pending_signal_enqueued_at TIMESTAMP,
            watching INTEGER NOT NULL DEFAULT 0,
            watch_baseline TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (agent_id, kind, handle)
        );
        INSERT INTO wait_signal_state (agent_id, kind, handle,
            last_signaled_outcome, last_delivery_status,
            last_attempt_started_at, last_delivery_attempt_at,
            pending_signal_id)
        VALUES
            ('did:legacy', 'talon', 'queued', 'done:complete', 'ok_queued',
             '2026-08-24 10:00:00', '2026-08-24 10:05:00', NULL),
            ('did:legacy', 'talon', 'bare', 'failed:failed', 'ok',
             '2026-08-25 10:00:00', '2026-08-25 10:05:00', NULL),
            ('did:legacy', 'talon', 'coalesced', 'done:complete',
             'coalesced_unbound', '2026-08-26 10:00:00',
             '2026-08-26 10:05:00', NULL),
            ('did:legacy', 'talon', 'undated', 'done:complete', 'ok_queued',
             NULL, '2026-08-23 10:05:00', NULL),
            ('did:legacy', 'talon', 'pending', 'done:complete', 'ok_queued',
             '2026-08-30 10:00:00', '2026-08-29 10:05:00', 'sig-next'),
            ('did:legacy', 'talon', 'hard', 'done:complete',
             'dropped_validation', '2026-08-27 10:00:00',
             '2026-08-27 10:05:00', NULL),
            ('did:legacy', 'talon', 'retrying', 'done:complete',
             'dropped_rate_limit', '2026-08-28 10:00:00',
             '2026-08-28 10:05:00', NULL),
            ('did:legacy', 'talon', 'never', NULL, 'dropped_quiet_hours',
             '2026-08-29 10:00:00', '2026-08-29 10:05:00', NULL);
        """
    )
    legacy.commit()
    legacy.close()

    database = await sqlite_database_factory(db_path)
    store = WaitSignalStore(database, agent_id="did:legacy")

    handles = (
        "queued", "bare", "coalesced", "undated", "pending", "hard",
        "retrying", "never",
    )
    backfilled = {
        handle: (await store.get("talon", handle)).last_delivered_at
        for handle in handles
    }
    assert backfilled == {
        "queued": "2026-08-24 10:00:00",
        "bare": "2026-08-25 10:00:00",
        "coalesced": "2026-08-26 10:00:00",
        "undated": None,
        "pending": None,
        "hard": None,
        "retrying": None,
        "never": None,
    }
    assert await store.list_rekeys() == [], "the audit table exists, empty"


def test_the_rekey_audit_table_is_not_created_by_the_bare_core_schema():
    """The bare ``CREATE TABLE IF NOT EXISTS`` loop is safe in sequence, not
    in parallel: two PostgreSQL initializers on the first post-upgrade boot
    can both pass the catalogue probe and one dies on ``pg_class``'s unique
    index (#3390 review). The table and its index are created by
    ``_ensure_wait_signal_rekeys`` instead."""
    from kestrel_sovereign.storage.async_database import core_schema_sql

    for backend in ("sqlite", "postgres"):
        assert "wait_signal_rekeys" not in core_schema_sql(backend)


@pytest.mark.asyncio
async def test_concurrent_initializers_create_the_rekey_audit_table_once(
    tmp_path, sqlite_database_factory,
):
    """A post-upgrade request burst must not race the audit table's creation:
    probe, migration lock, re-probe, then ``ensure_index``."""
    import asyncio
    from unittest.mock import patch

    from kestrel_sovereign.storage.async_database import (
        _WAIT_SIGNAL_REKEYS_INDEX,
    )

    db = await sqlite_database_factory(tmp_path / "rekey-schema-race.db")
    await db.execute("DROP TABLE wait_signal_rekeys")
    creates: list[str] = []
    indexes: list[tuple] = []
    real_execute = db.execute
    real_ensure_index = db.ensure_index

    async def recording_execute(sql, params=()):
        if sql.lstrip().startswith("CREATE TABLE") and "wait_signal_rekeys" in sql:
            creates.append(sql)
            # Let every contender reach the pre-lock probe; the lock and the
            # re-probe decide whether a second CREATE is attempted.
            await asyncio.sleep(0)
        return await real_execute(sql, params)

    async def recording_ensure_index(*args, **kwargs):
        indexes.append(args)
        return await real_ensure_index(*args, **kwargs)

    with (
        patch.object(db, "execute", recording_execute),
        patch.object(db, "ensure_index", recording_ensure_index),
    ):
        await asyncio.gather(*(db._ensure_wait_signal_rekeys() for _ in range(4)))

    assert len(creates) == 1
    assert indexes and all(args == _WAIT_SIGNAL_REKEYS_INDEX for args in indexes)
    assert await db.table_exists("wait_signal_rekeys")
    store = WaitSignalStore(db, agent_id="did:test:agent")
    await store.record_delivery(
        "talon", "job-1", delivery_status="ok_queued",
        signaled_outcome="done:complete",
        delivered_at=datetime.now(timezone.utc),
    )
    assert await store.adopt_signaled_token(
        "talon", "job-1", previous="done:complete", token="partial:complete",
        reason=REKEY_RECLASSIFIED,
    )
    assert len(await store.list_rekeys()) == 1
