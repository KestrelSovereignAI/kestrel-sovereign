"""CRUD tests for ``PendingA2AQuestionStore`` (#1444 step 2).

The store is sender-side state recording in-flight ``send_a2a_question``
calls. The cognition resumption design uses it for three things:
prompt assembly, restart replay, and hourly expiry sweeps. Each test
pins one of those use cases against a real SQLite backend.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.async_pending_a2a_question_store import (
    RECEIVER_SETTLEMENT_CONFIRMED,
    RECEIVER_SETTLEMENT_LATE_ANSWER,
    RECEIVER_SETTLEMENT_LEASE,
    RECEIVER_SETTLEMENT_PENDING,
    PendingA2AQuestionStore,
)


async def _make_store(tmp_path, agent_id="did:test:sender"):
    db_path = str(tmp_path / "store.db")
    db = await AsyncDatabase.sqlite(db_path)
    return PendingA2AQuestionStore(db, agent_id=agent_id)


@pytest.mark.asyncio
async def test_insert_and_get(tmp_path):
    store = await _make_store(tmp_path)
    deadline = datetime.now(timezone.utc) + timedelta(minutes=10)
    await store.insert(
        task_id="task-001",
        recipient="Meridian",
        recipient_agent_id="did:test:meridian",
        original_question="What is 2+2?",
        origin_turn_id="turn-xyz",
        origin_session_id="sess-abc",
        deadline=deadline,
    )
    row = await store.get("task-001")
    assert row is not None
    assert row.task_id == "task-001"
    assert row.recipient == "Meridian"
    assert row.recipient_agent_id == "did:test:meridian"
    assert row.original_question == "What is 2+2?"
    assert row.origin_turn_id == "turn-xyz"
    assert row.origin_session_id == "sess-abc"
    assert row.status == "WAITING"
    assert row.resolved_at is None
    waiting = await store.list_waiting()
    assert waiting[0].recipient_agent_id == "did:test:meridian"


@pytest.mark.asyncio
async def test_insert_is_idempotent_on_duplicate_task_id(tmp_path):
    """Crash-restart-during-write must not double-insert. INSERT OR IGNORE
    on the task_id PK silently drops the duplicate."""
    store = await _make_store(tmp_path)
    deadline = datetime.now(timezone.utc) + timedelta(minutes=10)
    await store.insert(
        task_id="task-002", recipient="Meridian",
        original_question="A", origin_turn_id=None, origin_session_id=None,
        deadline=deadline,
    )
    # Second insert under same task_id with DIFFERENT content must not
    # overwrite — operator intent is "first write wins" for correlation.
    await store.insert(
        task_id="task-002", recipient="Meridian",
        original_question="DIFFERENT QUESTION",
        origin_turn_id=None, origin_session_id=None,
        deadline=deadline,
    )
    row = await store.get("task-002")
    assert row.original_question == "A", (
        "Duplicate task_id insert must not silently overwrite the original "
        "question — that would corrupt the correlation table."
    )


@pytest.mark.asyncio
async def test_mark_resolved_transitions_waiting_only(tmp_path):
    store = await _make_store(tmp_path)
    deadline = datetime.now(timezone.utc) + timedelta(minutes=10)
    await store.insert(
        task_id="task-003", recipient="Meridian",
        original_question="x", origin_turn_id=None, origin_session_id=None,
        deadline=deadline,
    )

    assert await store.mark_resolved("task-003") is True
    row = await store.get("task-003")
    assert row.status == "RESOLVED"
    assert row.resolved_at is not None

    # Second mark_resolved is benign — subscription racing startup-replay
    # is a known race and both shouldn't fire signals.
    assert await store.mark_resolved("task-003") is False


@pytest.mark.asyncio
async def test_mark_resolved_returns_false_for_unknown_task(tmp_path):
    store = await _make_store(tmp_path)
    assert await store.mark_resolved("does-not-exist") is False


@pytest.mark.asyncio
async def test_mark_waiting_for_retry_restores_terminal_row(tmp_path):
    store = await _make_store(tmp_path)
    deadline = datetime.now(timezone.utc) + timedelta(minutes=10)
    await store.insert(
        task_id="task-retry", recipient="Meridian",
        original_question="x", origin_turn_id=None, origin_session_id=None,
        deadline=deadline,
    )

    assert await store.mark_resolved("task-retry") is True
    assert await store.mark_waiting_for_retry("task-retry") is True

    row = await store.get("task-retry")
    assert row.status == "WAITING"
    assert row.resolved_at is None


@pytest.mark.asyncio
async def test_mark_waiting_for_retry_preserves_terminal_payload(tmp_path):
    store = await _make_store(tmp_path)
    deadline = datetime.now(timezone.utc) + timedelta(minutes=10)
    await store.insert(
        task_id="task-retry-payload", recipient="Meridian",
        original_question="x", origin_turn_id=None, origin_session_id=None,
        deadline=deadline,
    )

    assert await store.mark_resolved("task-retry-payload") is True
    assert await store.mark_waiting_for_retry(
        "task-retry-payload",
        state="completed",
        reply_text="real answer",
    ) is True

    row = await store.get("task-retry-payload")
    assert row.status == "WAITING"
    assert row.retry_state == "completed"
    assert row.retry_reply_text == "real answer"


@pytest.mark.asyncio
async def test_list_waiting_excludes_terminal_rows(tmp_path):
    store = await _make_store(tmp_path)
    deadline = datetime.now(timezone.utc) + timedelta(minutes=10)
    for tid in ("task-A", "task-B", "task-C"):
        await store.insert(
            task_id=tid, recipient="Meridian",
            original_question=tid, origin_turn_id=None,
            origin_session_id=None, deadline=deadline,
        )
    await store.mark_resolved("task-B")

    waiting = await store.list_waiting()
    task_ids = {w.task_id for w in waiting}
    assert task_ids == {"task-A", "task-C"}, (
        f"list_waiting must filter out RESOLVED rows. Got {task_ids}."
    )


@pytest.mark.asyncio
async def test_list_waiting_past_deadline_for_expiry_sweep(tmp_path):
    """The hourly sweep input set. WAITING rows whose deadline is in the
    past land in this list; everyone else is excluded."""
    store = await _make_store(tmp_path)
    past = datetime.now(timezone.utc) - timedelta(hours=2)
    future = datetime.now(timezone.utc) + timedelta(hours=2)
    await store.insert(
        task_id="expired-1", recipient="Meridian",
        original_question="?", origin_turn_id=None, origin_session_id=None,
        deadline=past,
    )
    await store.insert(
        task_id="fresh-1", recipient="Meridian",
        original_question="?", origin_turn_id=None, origin_session_id=None,
        deadline=future,
    )
    await store.insert(
        task_id="already-resolved", recipient="Meridian",
        original_question="?", origin_turn_id=None, origin_session_id=None,
        deadline=past,  # past, but resolved already
    )
    await store.mark_resolved("already-resolved")

    expired = await store.list_waiting_past_deadline()
    task_ids = {e.task_id for e in expired}
    assert task_ids == {"expired-1"}, (
        f"Expired sweep must include past-deadline WAITING rows only "
        f"(not fresh ones, not already-resolved ones). Got {task_ids}."
    )


@pytest.mark.asyncio
async def test_two_agents_on_shared_backend_do_not_see_each_others_rows(tmp_path):
    """Codex round 1 P1 on PR #1453: a shared backend (Postgres multi-
    agent deployment) must NOT let agent A's startup-replay walk agent
    B's WAITING rows. Each store is scoped to its own ``agent_id`` so
    list_waiting / mark_resolved / mark_expired only see this agent's
    rows."""
    db_path = str(tmp_path / "shared.db")
    db = await AsyncDatabase.sqlite(db_path)
    store_a = PendingA2AQuestionStore(db, agent_id="did:test:A")
    store_b = PendingA2AQuestionStore(db, agent_id="did:test:B")
    deadline = datetime.now(timezone.utc) + timedelta(minutes=10)

    # Agent A's row + agent B's row, same task_id (a legitimate
    # collision now that task_id is no longer globally unique).
    await store_a.insert(
        task_id="t-shared", recipient="Meridian",
        original_question="A's question",
        origin_turn_id=None, origin_session_id=None,
        deadline=deadline,
    )
    await store_b.insert(
        task_id="t-shared", recipient="Meridian",
        original_question="B's question",
        origin_turn_id=None, origin_session_id=None,
        deadline=deadline,
    )

    # Each agent sees only its own row.
    rows_a = await store_a.list_waiting()
    rows_b = await store_b.list_waiting()
    assert len(rows_a) == 1 and rows_a[0].original_question == "A's question"
    assert len(rows_b) == 1 and rows_b[0].original_question == "B's question"

    # B's get() never returns A's row.
    a_row_seen_by_b = await store_b.get("t-shared")
    assert a_row_seen_by_b is not None
    assert a_row_seen_by_b.original_question == "B's question", (
        "Cross-agent get() would surface another agent's question "
        "text — that's the root of the codex round-1 P1 misroute bug."
    )

    # A's mark_resolved cannot touch B's row.
    assert await store_a.mark_resolved("t-shared") is True
    b_row_after = await store_b.get("t-shared")
    assert b_row_after.status == "WAITING", (
        "Agent A marking task_id 't-shared' RESOLVED must NOT mark "
        "agent B's same-id row RESOLVED — that would silently lose "
        "B's resumption signal."
    )

    # A's mark_resolved a second time is False (own row already terminal).
    assert await store_a.mark_resolved("t-shared") is False
    # But B can still mark THEIR own row resolved.
    assert await store_b.mark_resolved("t-shared") is True


@pytest.mark.asyncio
async def test_mark_resolved_is_durable_across_reopen(tmp_path):
    """Codex round 4 P2 on PR #1453: terminal status transitions MUST
    be durably committed on SQLite. The prior implementation used
    ``fetchall(UPDATE ... RETURNING ...)`` which surfaced the row to
    the current connection but never committed — restarting the agent
    resurrected RESOLVED rows as WAITING and the startup-replay sweep
    re-fired the resumption signal as a duplicate cognition wake."""
    db_path = str(tmp_path / "durable.db")
    db1 = await AsyncDatabase.sqlite(db_path)
    store1 = PendingA2AQuestionStore(db1, agent_id="did:test:durable")
    await store1.insert(
        task_id="t-durable", recipient="Meridian",
        recipient_agent_id="did:test:meridian",
        original_question="q", origin_turn_id=None, origin_session_id=None,
        deadline=datetime.now(timezone.utc) + timedelta(minutes=10),
    )
    assert await store1.mark_resolved("t-durable") is True
    assert (await store1.get("t-durable")).status == "RESOLVED"
    await db1.close()

    # Reopen the SAME file. If mark_resolved didn't commit, the row
    # will look WAITING again — startup-replay would then re-spawn a
    # supervisor and the eventual terminal would fire a DUPLICATE
    # a2a.question_answered signal.
    db2 = await AsyncDatabase.sqlite(db_path)
    store2 = PendingA2AQuestionStore(db2, agent_id="did:test:durable")
    row = await store2.get("t-durable")
    assert row is not None
    assert row.status == "RESOLVED", (
        "After agent restart the resolved row must STILL be RESOLVED "
        "— if it reverted to WAITING the startup-replay sweep would "
        "double-fire the resumption signal (codex round 4 P2)."
    )
    assert row.resolved_at is not None
    assert row.recipient_agent_id == "did:test:meridian"
    await db2.close()


@pytest.mark.asyncio
async def test_mark_expired_is_durable_across_reopen(tmp_path):
    """Same durability contract as ``mark_resolved`` — terminal
    EXPIRED rows must survive a restart so the hourly sweep doesn't
    walk them again on the next boot."""
    db_path = str(tmp_path / "durable_expired.db")
    db1 = await AsyncDatabase.sqlite(db_path)
    store1 = PendingA2AQuestionStore(db1, agent_id="did:test:durable")
    await store1.insert(
        task_id="t-exp-durable", recipient="Meridian",
        original_question="q", origin_turn_id=None, origin_session_id=None,
        deadline=datetime.now(timezone.utc) - timedelta(hours=2),
    )
    assert await store1.mark_expired("t-exp-durable") is True
    await db1.close()

    db2 = await AsyncDatabase.sqlite(db_path)
    store2 = PendingA2AQuestionStore(db2, agent_id="did:test:durable")
    row = await store2.get("t-exp-durable")
    assert row.status == "EXPIRED", (
        "After restart, an EXPIRED row must NOT revert to WAITING — "
        "the hourly sweep would otherwise re-fire the synthetic "
        "expired signal as a duplicate cognition wake."
    )
    await db2.close()


@pytest.mark.asyncio
async def test_insert_and_sweep_bind_datetime_not_string(tmp_path):
    """Codex round 6 P2 on PR #1453: Postgres TIMESTAMP columns reject
    string binds — asyncpg expects naive ``datetime`` objects. The
    store previously called ``.isoformat()`` on the deadline before
    binding, which worked on SQLite (TEXT-typed column accepts
    anything) but broke ``send_a2a_question`` for Postgres-backed
    agents AFTER the task had already been POSTed (no pending row =
    no resumption signal). Pin that ``insert`` and
    ``list_waiting_past_deadline`` pass real datetime values
    through.

    We patch the underlying ``execute``/``fetchall`` to capture the
    bound params and assert their types. The functional integration
    is already covered by the surrounding CRUD tests."""
    db_path = str(tmp_path / "type_check.db")
    db = await AsyncDatabase.sqlite(db_path)
    store = PendingA2AQuestionStore(db, agent_id="did:test:bind")

    captured_execute: list = []
    captured_fetchall: list = []
    real_execute = db.execute
    real_fetchall = db.fetchall

    async def fake_execute(sql, params=()):
        captured_execute.append((sql, params))
        return await real_execute(sql, params)

    async def fake_fetchall(sql, params=()):
        captured_fetchall.append((sql, params))
        return await real_fetchall(sql, params)

    db.execute = fake_execute
    db.fetchall = fake_fetchall

    deadline = datetime.now(timezone.utc) - timedelta(hours=1)
    await store.insert(
        task_id="t-bind", recipient="Meridian",
        original_question="q", origin_turn_id=None,
        origin_session_id=None, deadline=deadline,
    )
    await store.list_waiting_past_deadline()

    # Find the INSERT and SELECT calls.
    insert_calls = [
        params for sql, params in captured_execute
        if "INSERT" in sql.upper()
    ]
    sweep_calls = [
        params for sql, params in captured_fetchall
        if "deadline <" in sql
    ]
    assert insert_calls, "Insert path was not exercised."
    assert sweep_calls, "Sweep path was not exercised."

    # Insert binds positional (agent_id, task_id, recipient, stable identity,
    # msg, turn, session, deadline). Deadline is the 8th element.
    insert_params = insert_calls[0]
    assert isinstance(insert_params[7], datetime), (
        f"Insert must bind a datetime to the TIMESTAMP column, not a "
        f"string. Got {type(insert_params[7]).__name__}={insert_params[7]!r}. "
        f"Postgres rejects strings here (codex round 6 P2)."
    )
    assert insert_params[7].tzinfo is None, (
        "Bind must be a NAIVE datetime — Postgres TIMESTAMP (without "
        "tz) rejects tz-aware values."
    )

    sweep_params = sweep_calls[0]
    # list_waiting_past_deadline binds (agent_id, ts).
    assert isinstance(sweep_params[1], datetime), (
        f"Sweep must bind a datetime, not a string. Got "
        f"{type(sweep_params[1]).__name__}."
    )
    assert sweep_params[1].tzinfo is None


@pytest.mark.asyncio
async def test_mark_expired_terminal_transition(tmp_path):
    store = await _make_store(tmp_path)
    past = datetime.now(timezone.utc) - timedelta(hours=2)
    await store.insert(
        task_id="task-exp", recipient="Meridian",
        original_question="x", origin_turn_id=None, origin_session_id=None,
        deadline=past,
    )
    assert await store.mark_expired("task-exp") is True
    row = await store.get("task-exp")
    assert row.status == "EXPIRED"
    assert row.resolved_at is not None

    # Already-expired second call is benign.
    assert await store.mark_expired("task-exp") is False


# ---------------------------------------------------------------------------
# Receiver settlement (#3440)
# ---------------------------------------------------------------------------


async def _insert_past(store, task_id):
    await store.insert(
        task_id=task_id, recipient="Meridian",
        original_question="q", origin_turn_id=None, origin_session_id=None,
        deadline=datetime.now(timezone.utc) - timedelta(hours=2),
    )


@pytest.mark.asyncio
async def test_expiry_records_whether_the_recipient_is_settled(tmp_path):
    store = await _make_store(tmp_path)
    await _insert_past(store, "settled")
    await _insert_past(store, "unsettled")
    await _insert_past(store, "answered")

    assert await store.mark_expired("settled", receiver_settled=True) is True
    assert await store.mark_expired("unsettled") is True
    assert await store.mark_resolved("answered") is True

    assert (await store.get("settled")).receiver_settlement == (
        RECEIVER_SETTLEMENT_CONFIRMED
    )
    assert (await store.get("unsettled")).receiver_settlement == (
        RECEIVER_SETTLEMENT_PENDING
    )
    assert (await store.get("answered")).receiver_settlement is None

    awaiting = await store.list_expired_awaiting_receiver()
    assert [row.task_id for row in awaiting] == ["unsettled"]


@pytest.mark.asyncio
async def test_re_expiry_after_a_failed_wake_keeps_its_settlement(tmp_path):
    """An expired wake that was not delivered restores the row to WAITING and
    the retry re-expires it without settling again. A confirmed recipient
    must stay confirmed; an unconfirmed one stays owed."""
    store = await _make_store(tmp_path)
    await _insert_past(store, "confirmed")
    await _insert_past(store, "owed")
    await store.mark_expired("confirmed", receiver_settled=True)
    await store.mark_expired("owed")

    for task_id in ("confirmed", "owed"):
        assert await store.mark_waiting_for_retry(
            task_id, state="expired", reply_text="",
        ) is True
        assert await store.mark_expired(task_id) is True

    assert (await store.get("confirmed")).receiver_settlement == (
        RECEIVER_SETTLEMENT_CONFIRMED
    )
    assert (await store.get("owed")).receiver_settlement == (
        RECEIVER_SETTLEMENT_PENDING
    )


@pytest.mark.asyncio
async def test_unsettled_rows_have_no_age_cutoff(tmp_path):
    """The recipient keeps an undelivered submission for as long as it stays
    undelivered, so an unsettled row is never aged out of settlement."""
    store = await _make_store(tmp_path)
    await _insert_past(store, "ancient")
    await store.mark_expired("ancient")
    await store._db.execute(
        "UPDATE pending_a2a_questions SET resolved_at = ? WHERE task_id = ?",
        (datetime(2025, 1, 1), "ancient"),
    )

    assert [
        row.task_id for row in await store.list_expired_awaiting_receiver()
    ] == ["ancient"]


@pytest.mark.asyncio
async def test_unsettled_expiry_holds_the_lease_until_it_lapses(tmp_path):
    """The expiry's own ``expired`` wake reaches the asker before any pass
    can relay a late answer."""
    store = await _make_store(tmp_path)
    expired_at = datetime.now(timezone.utc)
    await _insert_past(store, "owed")
    await _insert_past(store, "settled")
    await store.mark_expired("owed", now=expired_at)
    await store.mark_expired("settled", receiver_settled=True, now=expired_at)

    assert await store.claim_receiver_settlement("owed", now=expired_at) is None
    lapsed = expired_at + RECEIVER_SETTLEMENT_LEASE
    claimed = await store.claim_receiver_settlement("owed", now=lapsed)
    assert claimed is not None
    assert claimed.task_id == "owed"
    assert claimed.receiver_settlement == RECEIVER_SETTLEMENT_PENDING
    # A confirmed row is never claimed.
    assert await store.claim_receiver_settlement("settled", now=lapsed) is None


@pytest.mark.asyncio
async def test_settlement_claim_is_single_winner_until_it_lapses(tmp_path):
    store = await _make_store(tmp_path)
    expired_at = datetime.now(timezone.utc) - 2 * RECEIVER_SETTLEMENT_LEASE
    await _insert_past(store, "row")
    await store.mark_expired("row", now=expired_at)
    now = datetime.now(timezone.utc)

    assert await store.claim_receiver_settlement("row", now=now) is not None
    assert await store.claim_receiver_settlement("row", now=now) is None
    # A pass that died holding the lease is resumed once it lapses.
    assert await store.claim_receiver_settlement(
        "row", now=now + RECEIVER_SETTLEMENT_LEASE,
    ) is not None


@pytest.mark.asyncio
async def test_renewal_extends_an_unsettled_lease_until_settled(tmp_path):
    """Whoever is waiting on a wake for the row keeps its lease; once the row
    is settled there is nothing left to renew."""
    store = await _make_store(tmp_path)
    t0 = datetime.now(timezone.utc)
    await _insert_past(store, "row")
    await store.mark_expired("row", now=t0)

    renewed_at = t0 + RECEIVER_SETTLEMENT_LEASE - timedelta(minutes=1)
    assert await store.renew_receiver_settlement_lease("row", now=renewed_at)
    # Past the original lease, still held.
    assert await store.claim_receiver_settlement(
        "row", now=t0 + RECEIVER_SETTLEMENT_LEASE,
    ) is None
    # A renewal never shortens a lease, and an unsettled row is still worth
    # holding when it is already leased for longer.
    assert await store.renew_receiver_settlement_lease("row", now=t0)
    assert await store.claim_receiver_settlement(
        "row", now=t0 + RECEIVER_SETTLEMENT_LEASE,
    ) is None
    assert await store.claim_receiver_settlement(
        "row", now=renewed_at + RECEIVER_SETTLEMENT_LEASE,
    ) is not None

    assert await store.confirm_receiver_withdrawn("row")
    assert not await store.renew_receiver_settlement_lease(
        "row", now=renewed_at + 2 * RECEIVER_SETTLEMENT_LEASE,
    )


@pytest.mark.asyncio
async def test_unsettled_rows_are_listed_least_recently_leased_first(tmp_path):
    """A row a pass just attempted goes to the back of the next pass."""
    store = await _make_store(tmp_path)
    t0 = datetime.now(timezone.utc)
    for task_id in ("a", "b"):
        await _insert_past(store, task_id)
        await store.mark_expired(task_id, now=t0)
    await store.claim_receiver_settlement(
        "a", now=t0 + 2 * RECEIVER_SETTLEMENT_LEASE,
    )

    assert [r.task_id for r in await store.list_expired_awaiting_receiver()] == [
        "b", "a",
    ]


@pytest.mark.asyncio
async def test_late_answer_is_held_on_the_row_until_delivered(tmp_path):
    store = await _make_store(tmp_path)
    await _insert_past(store, "late")
    await store.mark_expired("late")

    assert await store.record_late_answer(
        "late", state="completed", reply_text="the numbers",
    ) is True
    # One recording wins; a racing pass cannot overwrite the held answer.
    assert await store.record_late_answer(
        "late", state="failed", reply_text="other",
    ) is False
    held = await store.get("late")
    assert held.receiver_settlement == RECEIVER_SETTLEMENT_LATE_ANSWER
    assert held.receiver_late_state == "completed"
    assert held.receiver_late_reply_text == "the numbers"
    # Still owed: it stays in the settlement set, and a withdrawal cannot
    # retire it.
    assert [r.task_id for r in await store.list_expired_awaiting_receiver()] == [
        "late"
    ]
    assert await store.confirm_receiver_withdrawn("late") is False

    assert await store.confirm_late_answer_delivered("late") is True
    assert await store.confirm_late_answer_delivered("late") is False
    confirmed = await store.get("late")
    assert confirmed.receiver_settlement == RECEIVER_SETTLEMENT_CONFIRMED
    assert confirmed.receiver_late_state is None
    assert confirmed.receiver_late_reply_text is None
    assert await store.list_expired_awaiting_receiver() == []


@pytest.mark.asyncio
async def test_empty_late_reply_is_held_as_empty_not_missing(tmp_path):
    store = await _make_store(tmp_path)
    await _insert_past(store, "terse")
    await store.mark_expired("terse")

    await store.record_late_answer("terse", state="failed", reply_text="")

    held = await store.get("terse")
    assert held.receiver_late_state == "failed"
    assert held.receiver_late_reply_text == ""


@pytest.mark.asyncio
async def test_withdrawal_confirms_a_pending_row_once(tmp_path):
    store = await _make_store(tmp_path)
    await _insert_past(store, "withdrawn")
    await store.mark_expired("withdrawn")

    assert await store.confirm_receiver_withdrawn("withdrawn") is True
    assert await store.confirm_receiver_withdrawn("withdrawn") is False
    assert await store.record_late_answer(
        "withdrawn", state="completed", reply_text="too late",
    ) is False
    assert (await store.get("withdrawn")).receiver_settlement == (
        RECEIVER_SETTLEMENT_CONFIRMED
    )


@pytest.mark.asyncio
async def test_held_answer_survives_a_restored_expiry(tmp_path):
    """A failed ``expired`` wake restores its row to WAITING; the late answer
    held on it must still be owed after the retry re-expires it."""
    store = await _make_store(tmp_path)
    await _insert_past(store, "late")
    await store.mark_expired("late")
    await store.record_late_answer("late", state="completed", reply_text="42")

    assert await store.mark_waiting_for_retry(
        "late", state="expired", reply_text="",
    ) is True
    assert await store.mark_expired("late") is True

    row = await store.get("late")
    assert row.receiver_settlement == RECEIVER_SETTLEMENT_LATE_ANSWER
    assert (row.receiver_late_state, row.receiver_late_reply_text) == (
        "completed", "42",
    )


@pytest.mark.asyncio
async def test_settlement_ignores_rows_that_are_not_expired(tmp_path):
    store = await _make_store(tmp_path)
    await _insert_past(store, "waiting")
    later = datetime.now(timezone.utc) + 2 * RECEIVER_SETTLEMENT_LEASE

    assert await store.claim_receiver_settlement("waiting", now=later) is None
    assert await store.confirm_receiver_withdrawn("waiting") is False
    assert await store.record_late_answer(
        "waiting", state="completed", reply_text="x",
    ) is False
    assert (await store.get("waiting")).receiver_settlement is None


@pytest.mark.asyncio
async def test_settlement_rows_are_scoped_to_their_agent(tmp_path):
    db = await AsyncDatabase.sqlite(str(tmp_path / "shared-settlement.db"))
    emma = PendingA2AQuestionStore(db, agent_id="did:test:emma")
    other = PendingA2AQuestionStore(db, agent_id="did:test:other")
    await _insert_past(emma, "shared-id")
    await emma.mark_expired("shared-id")
    later = datetime.now(timezone.utc) + 2 * RECEIVER_SETTLEMENT_LEASE

    assert await other.list_expired_awaiting_receiver() == []
    assert await other.claim_receiver_settlement("shared-id", now=later) is None
    assert await other.confirm_receiver_withdrawn("shared-id") is False
    assert await other.record_late_answer(
        "shared-id", state="completed", reply_text="x",
    ) is False
    assert len(await emma.list_expired_awaiting_receiver()) == 1
    await db.close()
