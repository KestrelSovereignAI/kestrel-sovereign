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
    RECEIVER_SETTLEMENT_PENDING,
    QUESTION_CLAIM_LEASE,
    PendingA2AQuestionStore,
)

# What a supervisor claims a row with to wake the asker with its answer.
_ANSWER = {"state": "completed", "reply_text": "4", "claim": "supervisor"}


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

    assert await store.mark_resolved("task-003", **_ANSWER) is True
    row = await store.get("task-003")
    assert row.status == "RESOLVED"
    assert row.resolved_at is not None
    # The answer is owed to the asker until its delivery is confirmed.
    assert (row.retry_state, row.retry_reply_text) == ("completed", "4")

    # Second mark_resolved is benign — subscription racing startup-replay
    # is a known race and both shouldn't fire signals.
    assert await store.mark_resolved("task-003", **_ANSWER) is False


@pytest.mark.asyncio
async def test_mark_resolved_returns_false_for_unknown_task(tmp_path):
    store = await _make_store(tmp_path)
    assert await store.mark_resolved("does-not-exist", **_ANSWER) is False


@pytest.mark.asyncio
async def test_mark_waiting_for_retry_restores_terminal_row(tmp_path):
    store = await _make_store(tmp_path)
    deadline = datetime.now(timezone.utc) + timedelta(minutes=10)
    await store.insert(
        task_id="task-retry", recipient="Meridian",
        original_question="x", origin_turn_id=None, origin_session_id=None,
        deadline=deadline,
    )

    assert await store.mark_resolved("task-retry", **_ANSWER) is True
    assert await store.mark_waiting_for_retry(
        "task-retry", claim="supervisor",
    ) is True

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

    assert await store.mark_resolved("task-retry-payload", **_ANSWER) is True
    assert await store.mark_waiting_for_retry(
        "task-retry-payload",
        claim="supervisor",
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
    await store.mark_resolved("task-B", **_ANSWER)

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
    await store.mark_resolved("already-resolved", **_ANSWER)

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
    assert await store_a.mark_resolved("t-shared", **_ANSWER) is True
    b_row_after = await store_b.get("t-shared")
    assert b_row_after.status == "WAITING", (
        "Agent A marking task_id 't-shared' RESOLVED must NOT mark "
        "agent B's same-id row RESOLVED — that would silently lose "
        "B's resumption signal."
    )

    # A's mark_resolved a second time is False (own row already terminal).
    assert await store_a.mark_resolved("t-shared", **_ANSWER) is False
    # But B can still mark THEIR own row resolved.
    assert await store_b.mark_resolved("t-shared", **_ANSWER) is True


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
    assert await store1.mark_resolved("t-durable", **_ANSWER) is True
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
    assert await store1.mark_expired("t-exp-durable", claim="expiry") is True
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
    assert await store.mark_expired("task-exp", claim="expiry") is True
    row = await store.get("task-exp")
    assert row.status == "EXPIRED"
    assert row.resolved_at is not None
    # The ``expired`` wake is owed to the asker until it is delivered.
    assert row.retry_state == "expired"

    # Already-expired second call is benign.
    assert await store.mark_expired("task-exp", claim="other") is False


# ---------------------------------------------------------------------------
# The wake owed to the asker (#3440)
# ---------------------------------------------------------------------------


async def _insert_past(store, task_id):
    await store.insert(
        task_id=task_id, recipient="Meridian",
        original_question="q", origin_turn_id=None, origin_session_id=None,
        deadline=datetime.now(timezone.utc) - timedelta(hours=2),
    )


async def _expire(
    store, task_id, *, claim="expiry", now=None, receiver_settled=False,
    delivered=True,
):
    """Expire ``task_id`` under ``claim`` and, unless ``delivered`` is False,
    confirm its ``expired`` wake was delivered."""
    await _insert_past(store, task_id)
    assert await store.mark_expired(
        task_id, claim=claim, receiver_settled=receiver_settled, now=now,
    )
    if delivered:
        assert await store.confirm_wake_delivered(task_id, claim=claim)


@pytest.mark.asyncio
async def test_a_claimed_wake_stays_owed_until_its_delivery_is_confirmed(
    tmp_path,
):
    store = await _make_store(tmp_path)
    await _insert_past(store, "answered")
    await _insert_past(store, "withdrawn")
    await store.mark_resolved("answered", **_ANSWER)
    await store.mark_expired("withdrawn", claim="expiry", receiver_settled=True)

    # Only the claim that took the row out of WAITING retires its wake.
    assert not await store.confirm_wake_delivered("answered", claim="other")
    assert (await store.get("answered")).retry_state == "completed"
    assert await store.confirm_wake_delivered("answered", claim="supervisor")
    assert not await store.confirm_wake_delivered("answered", claim="supervisor")
    assert await store.confirm_wake_delivered("withdrawn", claim="expiry")

    for task_id, status in (("answered", "RESOLVED"), ("withdrawn", "EXPIRED")):
        row = await store.get(task_id)
        assert row.status == status
        assert (row.retry_state, row.retry_reply_text) == (None, None)
    # Nothing is left to hold either row for.
    assert not await store.renew_claim_lease("answered", claim="supervisor")
    assert not await store.renew_claim_lease("withdrawn", claim="expiry")


@pytest.mark.asyncio
async def test_a_wake_in_flight_keeps_its_row_until_its_claim_lapses(tmp_path):
    """A pass that dies between claiming a row and confirming its wake leaves
    the wake owed. Once its lease lapses, the row goes back to WAITING
    carrying that wake, to be delivered again."""
    store = await _make_store(tmp_path)
    t0 = datetime.now(timezone.utc)
    await _insert_past(store, "answered")
    await _insert_past(store, "withdrawn")
    await _insert_past(store, "unsettled")
    await _insert_past(store, "delivered")
    await store.mark_resolved("answered", **_ANSWER, now=t0)
    await store.mark_expired(
        "withdrawn", claim="expiry", receiver_settled=True, now=t0,
    )
    await store.mark_expired("unsettled", claim="expiry", now=t0)
    await store.mark_resolved("delivered", **_ANSWER, now=t0)
    await store.confirm_wake_delivered("delivered", claim="supervisor")

    # Still leased: the wake may still be in flight.
    assert await store.reclaim_abandoned_wakes(
        now=t0 + QUESTION_CLAIM_LEASE / 2,
    ) == []
    assert (await store.get("answered")).status == "RESOLVED"

    reclaimed = await store.reclaim_abandoned_wakes(now=t0 + QUESTION_CLAIM_LEASE)

    assert {row.task_id: (row.status, row.retry_state) for row in reclaimed} == {
        "answered": ("WAITING", "completed"),
        "withdrawn": ("WAITING", "expired"),
        "unsettled": ("WAITING", "expired"),
    }
    assert (await store.get("answered")).retry_reply_text == "4"
    # An expired row keeps the settlement it recorded.
    assert (await store.get("withdrawn")).receiver_settlement == (
        RECEIVER_SETTLEMENT_CONFIRMED
    )
    assert (await store.get("unsettled")).receiver_settlement == (
        RECEIVER_SETTLEMENT_PENDING
    )
    assert (await store.get("delivered")).status == "RESOLVED"
    # The dead pass's claim is gone: it can neither confirm nor restore.
    assert not await store.confirm_wake_delivered("answered", claim="supervisor")
    assert not await store.mark_waiting_for_retry(
        "answered", claim="supervisor", state="completed", reply_text="4",
    )
    # Reclaimed once.
    assert await store.reclaim_abandoned_wakes(
        now=t0 + 2 * QUESTION_CLAIM_LEASE,
    ) == []


@pytest.mark.asyncio
async def test_a_wake_renewed_while_in_flight_is_not_reclaimed(tmp_path):
    store = await _make_store(tmp_path)
    t0 = datetime.now(timezone.utc)
    await _insert_past(store, "answered")
    await store.mark_resolved("answered", **_ANSWER, now=t0)

    renewed_at = t0 + QUESTION_CLAIM_LEASE - timedelta(minutes=1)
    assert await store.renew_claim_lease(
        "answered", claim="supervisor", now=renewed_at,
    )
    assert await store.reclaim_abandoned_wakes(
        now=t0 + QUESTION_CLAIM_LEASE,
    ) == []
    assert not await store.renew_claim_lease("answered", claim="other")


@pytest.mark.asyncio
async def test_only_the_claim_holder_restores_its_row(tmp_path):
    """A pass whose wake was reclaimed and re-claimed by another must not
    pull the row back to WAITING from under the new holder."""
    store = await _make_store(tmp_path)
    t0 = datetime.now(timezone.utc) - 4 * QUESTION_CLAIM_LEASE
    await _insert_past(store, "answered")
    await store.mark_resolved("answered", **_ANSWER, now=t0)
    await store.reclaim_abandoned_wakes(now=t0 + QUESTION_CLAIM_LEASE)
    assert await store.mark_resolved(
        "answered", state="completed", reply_text="4", claim="retry",
    )

    assert not await store.mark_waiting_for_retry(
        "answered", claim="supervisor", state="completed", reply_text="4",
    )
    assert (await store.get("answered")).status == "RESOLVED"
    assert await store.mark_waiting_for_retry(
        "answered", claim="retry", state="completed", reply_text="4",
    )
    assert (await store.get("answered")).status == "WAITING"


@pytest.mark.asyncio
async def test_owed_wakes_are_scoped_to_their_agent(tmp_path):
    db = await AsyncDatabase.sqlite(str(tmp_path / "shared-wakes.db"))
    emma = PendingA2AQuestionStore(db, agent_id="did:test:emma")
    other = PendingA2AQuestionStore(db, agent_id="did:test:other")
    t0 = datetime.now(timezone.utc) - 4 * QUESTION_CLAIM_LEASE
    await _insert_past(emma, "shared-id")
    await emma.mark_resolved("shared-id", **_ANSWER, now=t0)

    assert await other.reclaim_abandoned_wakes() == []
    assert not await other.confirm_wake_delivered("shared-id", claim="supervisor")
    assert (await emma.get("shared-id")).retry_state == "completed"
    assert [row.task_id for row in await emma.reclaim_abandoned_wakes()] == [
        "shared-id"
    ]
    await db.close()


# ---------------------------------------------------------------------------
# Receiver settlement (#3440)
# ---------------------------------------------------------------------------


async def _expired_and_claimed(store, task_id):
    """An unsettled expiry whose ``expired`` wake was delivered and whose own
    lease has lapsed, claimed by a pass."""
    t0 = datetime.now(timezone.utc) - 2 * QUESTION_CLAIM_LEASE
    await _expire(store, task_id, now=t0)
    claim = await store.claim_receiver_settlement(task_id)
    assert claim is not None
    return claim.token


@pytest.mark.asyncio
async def test_expiry_records_whether_the_recipient_is_settled(tmp_path):
    store = await _make_store(tmp_path)
    await _expire(store, "settled", receiver_settled=True)
    await _expire(store, "unsettled")
    await _insert_past(store, "answered")
    assert await store.mark_resolved("answered", **_ANSWER) is True

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
async def test_no_settlement_while_the_expired_wake_is_owed(tmp_path):
    """The asker hears ``expired`` before any late answer: settlement does not
    start on a row whose ``expired`` wake is not yet confirmed delivered,
    even once that wake's claim lapsed."""
    store = await _make_store(tmp_path)
    t0 = datetime.now(timezone.utc) - 4 * QUESTION_CLAIM_LEASE
    await _expire(store, "owed", now=t0, delivered=False)

    assert await store.list_expired_awaiting_receiver() == []
    assert await store.claim_receiver_settlement("owed") is None

    assert await store.confirm_wake_delivered("owed", claim="expiry")
    assert [r.task_id for r in await store.list_expired_awaiting_receiver()] == [
        "owed"
    ]
    assert await store.claim_receiver_settlement("owed") is not None


@pytest.mark.asyncio
async def test_re_expiry_after_a_failed_wake_keeps_its_settlement(tmp_path):
    """An expired wake that was not delivered restores the row to WAITING and
    the retry re-expires it without settling again. A confirmed recipient
    must stay confirmed; an unconfirmed one stays owed."""
    store = await _make_store(tmp_path)
    await _expire(store, "confirmed", receiver_settled=True, delivered=False)
    await _expire(store, "owed", delivered=False)

    for task_id in ("confirmed", "owed"):
        assert await store.mark_waiting_for_retry(
            task_id, claim="expiry", state="expired", reply_text="",
        ) is True
        assert await store.mark_expired(task_id, claim="retry") is True

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
    await _expire(store, "ancient")
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
    can relay a late answer: its claim outlasts its delivery."""
    store = await _make_store(tmp_path)
    expired_at = datetime.now(timezone.utc)
    await _expire(store, "owed", now=expired_at)
    await _expire(store, "settled", receiver_settled=True, now=expired_at)

    assert await store.claim_receiver_settlement("owed", now=expired_at) is None
    lapsed = expired_at + QUESTION_CLAIM_LEASE
    claim = await store.claim_receiver_settlement("owed", now=lapsed)
    assert claim is not None
    assert claim.question.task_id == "owed"
    assert claim.question.receiver_settlement == RECEIVER_SETTLEMENT_PENDING
    assert claim.token not in ("", "expiry")
    # A confirmed row is never claimed.
    assert await store.claim_receiver_settlement("settled", now=lapsed) is None


@pytest.mark.asyncio
async def test_settlement_claim_is_single_winner_until_it_lapses(tmp_path):
    store = await _make_store(tmp_path)
    expired_at = datetime.now(timezone.utc) - 2 * QUESTION_CLAIM_LEASE
    await _expire(store, "row", now=expired_at)
    now = datetime.now(timezone.utc)

    first = await store.claim_receiver_settlement("row", now=now)
    assert first is not None
    assert await store.claim_receiver_settlement("row", now=now) is None
    # A pass that died holding the claim is resumed once its lease lapses,
    # under a claim of its own.
    second = await store.claim_receiver_settlement(
        "row", now=now + QUESTION_CLAIM_LEASE,
    )
    assert second is not None
    assert second.token != first.token


@pytest.mark.asyncio
async def test_renewal_extends_an_unsettled_lease_until_settled(tmp_path):
    """Whoever is waiting on a wake for the row keeps its lease; once the row
    is settled there is nothing left to renew."""
    store = await _make_store(tmp_path)
    t0 = datetime.now(timezone.utc)
    await _expire(store, "row", now=t0)

    renewed_at = t0 + QUESTION_CLAIM_LEASE - timedelta(minutes=1)
    assert await store.renew_claim_lease(
        "row", claim="expiry", now=renewed_at,
    )
    # Past the original lease, still held.
    assert await store.claim_receiver_settlement(
        "row", now=t0 + QUESTION_CLAIM_LEASE,
    ) is None
    # A renewal never shortens a lease, and the claim is still held when the
    # row is already leased for longer.
    assert await store.renew_claim_lease(
        "row", claim="expiry", now=t0,
    )
    assert await store.claim_receiver_settlement(
        "row", now=t0 + QUESTION_CLAIM_LEASE,
    ) is None
    claim = await store.claim_receiver_settlement(
        "row", now=renewed_at + QUESTION_CLAIM_LEASE,
    )
    assert claim is not None

    assert await store.confirm_receiver_withdrawn("row", claim=claim.token)
    assert not await store.renew_claim_lease(
        "row", claim=claim.token, now=renewed_at + 2 * QUESTION_CLAIM_LEASE,
    )


@pytest.mark.asyncio
async def test_a_replaced_claim_is_fenced_out_of_every_mutation(tmp_path):
    """A pass stalls past its lease and another pass claims the row. When the
    stalled pass resumes it can neither keep the row, nor record, confirm or
    stretch the new claimant's lease (#3440)."""
    store = await _make_store(tmp_path)
    t0 = datetime.now(timezone.utc) - 4 * QUESTION_CLAIM_LEASE
    await _expire(store, "row", now=t0)
    stalled = await store.claim_receiver_settlement(
        "row", now=t0 + QUESTION_CLAIM_LEASE,
    )
    takeover_at = t0 + 2 * QUESTION_CLAIM_LEASE
    current = await store.claim_receiver_settlement("row", now=takeover_at)
    assert stalled is not None and current is not None

    # Later than the takeover, so a renewal that took would push the new
    # claimant's lease out.
    assert not await store.renew_claim_lease(
        "row", claim=stalled.token, now=takeover_at + QUESTION_CLAIM_LEASE / 2,
    )
    assert not await store.record_late_answer(
        "row", claim=stalled.token, state="completed", reply_text="stale",
    )
    assert not await store.confirm_receiver_withdrawn("row", claim=stalled.token)
    # The stalled pass's renewal did not stretch the new claimant's lease:
    # it lapses on the new claimant's own schedule.
    assert await store.claim_receiver_settlement(
        "row", now=takeover_at + QUESTION_CLAIM_LEASE,
    ) is not None
    assert (await store.get("row")).receiver_settlement == (
        RECEIVER_SETTLEMENT_PENDING
    )


@pytest.mark.asyncio
async def test_a_replaced_claim_cannot_retire_a_held_answer(tmp_path):
    """The new claimant recorded the answer and is relaying it. The stalled
    pass must not confirm that relay delivered."""
    store = await _make_store(tmp_path)
    t0 = datetime.now(timezone.utc) - 4 * QUESTION_CLAIM_LEASE
    await _expire(store, "row", now=t0)
    stalled = await store.claim_receiver_settlement(
        "row", now=t0 + QUESTION_CLAIM_LEASE,
    )
    current = await store.claim_receiver_settlement(
        "row", now=t0 + 2 * QUESTION_CLAIM_LEASE,
    )
    assert await store.record_late_answer(
        "row", claim=current.token, state="completed", reply_text="42",
    )

    assert not await store.renew_claim_lease(
        "row", claim=stalled.token,
    )
    assert not await store.confirm_late_answer_delivered(
        "row", claim=stalled.token,
    )
    assert (await store.get("row")).receiver_settlement == (
        RECEIVER_SETTLEMENT_LATE_ANSWER
    )
    assert await store.confirm_late_answer_delivered("row", claim=current.token)


@pytest.mark.asyncio
async def test_a_claim_past_its_lease_holds_until_another_replaces_it(tmp_path):
    """The lease only decides when a row may be claimed away. A holder whose
    renewals were delayed past it, with nobody having claimed the row since,
    still holds it, and renewing puts the lease back."""
    store = await _make_store(tmp_path)
    t0 = datetime.now(timezone.utc) - 4 * QUESTION_CLAIM_LEASE
    await _expire(store, "row", now=t0)
    late = t0 + 2 * QUESTION_CLAIM_LEASE

    assert await store.renew_claim_lease(
        "row", claim="expiry", now=late,
    )
    assert await store.claim_receiver_settlement("row", now=late) is None


@pytest.mark.asyncio
async def test_unsettled_rows_are_listed_least_recently_leased_first(tmp_path):
    """A row a pass just attempted goes to the back of the next pass."""
    store = await _make_store(tmp_path)
    t0 = datetime.now(timezone.utc)
    for task_id in ("a", "b"):
        await _expire(store, task_id, now=t0)
    await store.claim_receiver_settlement(
        "a", now=t0 + 2 * QUESTION_CLAIM_LEASE,
    )

    assert [r.task_id for r in await store.list_expired_awaiting_receiver()] == [
        "b", "a",
    ]


@pytest.mark.asyncio
async def test_late_answer_is_held_on_the_row_until_delivered(tmp_path):
    store = await _make_store(tmp_path)
    claim = await _expired_and_claimed(store, "late")

    assert await store.record_late_answer(
        "late", claim=claim, state="completed", reply_text="the numbers",
    ) is True
    # One recording wins; a racing pass cannot overwrite the held answer.
    assert await store.record_late_answer(
        "late", claim=claim, state="failed", reply_text="other",
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
    assert await store.confirm_receiver_withdrawn("late", claim=claim) is False

    assert await store.confirm_late_answer_delivered("late", claim=claim) is True
    assert await store.confirm_late_answer_delivered("late", claim=claim) is False
    confirmed = await store.get("late")
    assert confirmed.receiver_settlement == RECEIVER_SETTLEMENT_CONFIRMED
    assert confirmed.receiver_late_state is None
    assert confirmed.receiver_late_reply_text is None
    assert await store.list_expired_awaiting_receiver() == []


@pytest.mark.asyncio
async def test_empty_late_reply_is_held_as_empty_not_missing(tmp_path):
    store = await _make_store(tmp_path)
    claim = await _expired_and_claimed(store, "terse")

    await store.record_late_answer(
        "terse", claim=claim, state="failed", reply_text="",
    )

    held = await store.get("terse")
    assert held.receiver_late_state == "failed"
    assert held.receiver_late_reply_text == ""


@pytest.mark.asyncio
async def test_withdrawal_confirms_a_pending_row_once(tmp_path):
    store = await _make_store(tmp_path)
    claim = await _expired_and_claimed(store, "withdrawn")

    assert await store.confirm_receiver_withdrawn("withdrawn", claim=claim) is True
    assert await store.confirm_receiver_withdrawn("withdrawn", claim=claim) is False
    assert await store.record_late_answer(
        "withdrawn", claim=claim, state="completed", reply_text="too late",
    ) is False
    assert (await store.get("withdrawn")).receiver_settlement == (
        RECEIVER_SETTLEMENT_CONFIRMED
    )


@pytest.mark.asyncio
async def test_failed_expired_wake_is_restored_under_its_own_claim(tmp_path):
    """An ``expired`` wake whose delivery failed was never confirmed, so it is
    still owed under the claim that expired the row. That claim, and only it,
    returns the row to WAITING with the wake; no settlement claim can, and
    once a wake is confirmed delivered there is nothing left to restore."""
    store = await _make_store(tmp_path)
    t0 = datetime.now(timezone.utc) - 4 * QUESTION_CLAIM_LEASE
    await _expire(store, "row", claim="expiry", now=t0, delivered=False)
    # The lapsed lease still lets no settlement pass at the row.
    assert await store.claim_receiver_settlement("row") is None

    assert not await store.mark_waiting_for_retry(
        "row", claim="other", state="expired", reply_text="",
    )
    assert await store.mark_waiting_for_retry(
        "row", claim="expiry", state="expired", reply_text="",
    )
    row = await store.get("row")
    assert (row.status, row.retry_state, row.receiver_settlement) == (
        "WAITING", "expired", RECEIVER_SETTLEMENT_PENDING,
    )
    # Restored once: the claim is gone with it.
    assert not await store.mark_waiting_for_retry(
        "row", claim="expiry", state="expired", reply_text="",
    )

    # The retry expires it again under a claim of its own, and is delivered.
    assert await store.mark_expired("row", claim="retry", now=t0)
    assert await store.confirm_wake_delivered("row", claim="retry")
    # A delivered wake is not owed, so no claim can restore it.
    assert not await store.mark_waiting_for_retry(
        "row", claim="retry", state="expired", reply_text="",
    )
    row = await store.get("row")
    assert (row.status, row.retry_state) == ("EXPIRED", None)
    settlement = await store.claim_receiver_settlement("row")
    assert settlement is not None
    assert not await store.mark_waiting_for_retry(
        "row", claim=settlement.token, state="expired", reply_text="",
    )
    assert (await store.get("row")).status == "EXPIRED"


@pytest.mark.asyncio
async def test_held_answer_is_recovered_by_a_lease_takeover(tmp_path):
    """A pass records the recipient's late answer and stalls before relaying
    it. Once its lease lapses another pass claims the row and finds the held
    answer. The stalled pass then can neither renew (the check it makes before
    it enqueues the relay), confirm, nor record again; the new holder can."""
    store = await _make_store(tmp_path)
    t0 = datetime.now(timezone.utc) - 4 * QUESTION_CLAIM_LEASE
    await _expire(store, "row", now=t0)
    stalled = await store.claim_receiver_settlement(
        "row", now=t0 + QUESTION_CLAIM_LEASE,
    )
    assert stalled is not None
    assert await store.record_late_answer(
        "row", claim=stalled.token, state="completed", reply_text="42",
    )
    # Inside the stalled pass's lease nobody takes the row.
    assert await store.claim_receiver_settlement(
        "row", now=t0 + QUESTION_CLAIM_LEASE * 1.5,
    ) is None

    takeover = await store.claim_receiver_settlement(
        "row", now=t0 + 2 * QUESTION_CLAIM_LEASE,
    )

    assert takeover is not None and takeover.token != stalled.token
    held = takeover.question
    assert held.receiver_settlement == RECEIVER_SETTLEMENT_LATE_ANSWER
    assert (held.receiver_late_state, held.receiver_late_reply_text) == (
        "completed", "42",
    )
    assert not await store.renew_claim_lease("row", claim=stalled.token)
    assert not await store.confirm_late_answer_delivered(
        "row", claim=stalled.token,
    )
    assert not await store.record_late_answer(
        "row", claim=stalled.token, state="failed", reply_text="stale",
    )
    assert (await store.get("row")).receiver_late_reply_text == "42"

    assert await store.renew_claim_lease("row", claim=takeover.token)
    assert await store.confirm_late_answer_delivered(
        "row", claim=takeover.token,
    )
    assert (await store.get("row")).receiver_settlement == (
        RECEIVER_SETTLEMENT_CONFIRMED
    )


@pytest.mark.asyncio
async def test_settlement_ignores_rows_that_are_not_expired(tmp_path):
    store = await _make_store(tmp_path)
    await _insert_past(store, "waiting")
    later = datetime.now(timezone.utc) + 2 * QUESTION_CLAIM_LEASE

    assert await store.claim_receiver_settlement("waiting", now=later) is None
    assert await store.confirm_receiver_withdrawn("waiting", claim="x") is False
    assert await store.record_late_answer(
        "waiting", claim="x", state="completed", reply_text="x",
    ) is False
    assert (await store.get("waiting")).receiver_settlement is None


@pytest.mark.asyncio
async def test_settlement_rows_are_scoped_to_their_agent(tmp_path):
    db = await AsyncDatabase.sqlite(str(tmp_path / "shared-settlement.db"))
    emma = PendingA2AQuestionStore(db, agent_id="did:test:emma")
    other = PendingA2AQuestionStore(db, agent_id="did:test:other")
    await _expire(emma, "shared-id", claim="emma")
    later = datetime.now(timezone.utc) + 2 * QUESTION_CLAIM_LEASE

    assert await other.list_expired_awaiting_receiver() == []
    assert await other.claim_receiver_settlement("shared-id", now=later) is None
    # Not even with Emma's own claim.
    assert not await other.renew_claim_lease(
        "shared-id", claim="emma",
    )
    assert await other.confirm_receiver_withdrawn(
        "shared-id", claim="emma",
    ) is False
    assert await other.record_late_answer(
        "shared-id", claim="emma", state="completed", reply_text="x",
    ) is False
    assert len(await emma.list_expired_awaiting_receiver()) == 1
    await db.close()


# The pending_a2a_questions table as releases before #3440 created it.
_PRE_3440_PENDING_A2A_QUESTIONS = """
CREATE TABLE pending_a2a_questions (
    agent_id TEXT NOT NULL DEFAULT '',
    task_id TEXT NOT NULL,
    recipient TEXT NOT NULL,
    recipient_agent_id TEXT,
    original_question TEXT NOT NULL,
    origin_turn_id TEXT,
    origin_session_id TEXT,
    deadline TIMESTAMP NOT NULL,
    status TEXT NOT NULL DEFAULT 'WAITING' CHECK (
        status IN ('WAITING', 'RESOLVED', 'EXPIRED')
    ),
    retry_state TEXT,
    retry_reply_text TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP,
    PRIMARY KEY (agent_id, task_id)
)
"""


def _write_pre_3440_questions(db_path, rows) -> None:
    """A database a release before #3440 left behind: ``rows`` are
    ``(agent_id, task_id, status)``, all past their deadline."""
    import sqlite3
    from contextlib import closing

    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(_PRE_3440_PENDING_A2A_QUESTIONS)
        conn.executemany(
            "INSERT INTO pending_a2a_questions "
            "(agent_id, task_id, recipient, recipient_agent_id, "
            " original_question, origin_session_id, deadline, status) "
            "VALUES (?, ?, 'Nellie', 'did:test:nellie', 'Live numbers?', "
            "        'session', '2026-10-02 19:33:00', ?)",
            rows,
        )
        conn.commit()


@pytest.mark.asyncio
async def test_questions_expired_before_settlement_existed_are_adopted(tmp_path):
    """The 2026-10-02 questions expired on a release with no receiver
    settlement. After the upgrade their recipients' tasks must still be
    settled, so every such row is owed (#3440)."""
    db_path = str(tmp_path / "pre-3440.db")
    _write_pre_3440_questions(db_path, [
        ("did:test:emma", "238234052a234bee", "EXPIRED"),
        ("did:test:emma", "bca4db9cbf0442e6", "EXPIRED"),
        ("did:test:emma", "answered", "RESOLVED"),
        ("did:test:emma", "waiting", "WAITING"),
        ("did:test:other", "theirs", "EXPIRED"),
    ])

    db = await AsyncDatabase.sqlite(db_path)
    emma = PendingA2AQuestionStore(db, agent_id="did:test:emma")
    other = PendingA2AQuestionStore(db, agent_id="did:test:other")

    owed = await emma.list_expired_awaiting_receiver()
    assert [row.task_id for row in owed] == [
        "238234052a234bee", "bca4db9cbf0442e6",
    ]
    assert {row.receiver_settlement for row in owed} == {
        RECEIVER_SETTLEMENT_PENDING
    }
    assert [row.task_id for row in await other.list_expired_awaiting_receiver()] == [
        "theirs"
    ]
    assert (await emma.get("answered")).receiver_settlement is None
    assert (await emma.get("waiting")).receiver_settlement is None
    # Never leased, so the first pass can claim them.
    assert await emma.claim_receiver_settlement("238234052a234bee") is not None
    await db.close()


@pytest.mark.asyncio
async def test_adoption_never_reopens_a_settled_row(tmp_path):
    """Adoption runs at every initialization, so it must touch only rows a
    release before #3440 wrote: never one whose settlement is recorded."""
    db_path = str(tmp_path / "pre-3440.db")
    _write_pre_3440_questions(db_path, [
        ("did:test:emma", "confirmed", "EXPIRED"),
        ("did:test:emma", "late", "EXPIRED"),
    ])
    db = await AsyncDatabase.sqlite(db_path)
    store = PendingA2AQuestionStore(db, agent_id="did:test:emma")
    confirmed = await store.claim_receiver_settlement("confirmed")
    assert await store.confirm_receiver_withdrawn(
        "confirmed", claim=confirmed.token,
    )
    late = await store.claim_receiver_settlement("late")
    assert await store.record_late_answer(
        "late", claim=late.token, state="completed", reply_text="42",
    )
    await db.close()

    db = await AsyncDatabase.sqlite(db_path)
    store = PendingA2AQuestionStore(db, agent_id="did:test:emma")

    assert (await store.get("confirmed")).receiver_settlement == (
        RECEIVER_SETTLEMENT_CONFIRMED
    )
    row = await store.get("late")
    assert row.receiver_settlement == RECEIVER_SETTLEMENT_LATE_ANSWER
    assert row.receiver_late_reply_text == "42"
    # The late answer's claim survived the restart too.
    assert await store.claim_receiver_settlement("late") is None
    await db.close()
