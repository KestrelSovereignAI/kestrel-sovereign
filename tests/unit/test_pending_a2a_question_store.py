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
    PendingA2AQuestionStore,
)


async def _make_store(tmp_path):
    db_path = str(tmp_path / "store.db")
    db = await AsyncDatabase.sqlite(db_path)
    return PendingA2AQuestionStore(db)


@pytest.mark.asyncio
async def test_insert_and_get(tmp_path):
    store = await _make_store(tmp_path)
    deadline = datetime.now(timezone.utc) + timedelta(minutes=10)
    await store.insert(
        task_id="task-001",
        recipient="Meridian",
        original_question="What is 2+2?",
        origin_turn_id="turn-xyz",
        origin_session_id="sess-abc",
        deadline=deadline,
    )
    row = await store.get("task-001")
    assert row is not None
    assert row.task_id == "task-001"
    assert row.recipient == "Meridian"
    assert row.original_question == "What is 2+2?"
    assert row.origin_turn_id == "turn-xyz"
    assert row.origin_session_id == "sess-abc"
    assert row.status == "WAITING"
    assert row.resolved_at is None


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
