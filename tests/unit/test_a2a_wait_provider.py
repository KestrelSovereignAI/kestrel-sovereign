"""Tests for the ``a2a:`` Waitable provider (#2729).

An outbound A2A task/question dispatched to a peer lives in the peer's task
store, not this agent's, so it must be watched with the ``a2a:`` provider —
which routes status checks through the sender-side outbound audit row and the
peer proxy — NOT the local ``task:`` provider. These tests cover the two
surfaces #2729 relies on: ownership validation (``owns_handle``) and terminal
classification (``poll``).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from kestrel_sdk.tools import Outcome
from kestrel_sdk.tools.result import ToolResult

from kestrel_sovereign.a2a.outbound_store import (
    ensure_a2a_outbound_tasks_table,
    record_outbound_dispatch,
    update_outbound_terminal_state,
)
from kestrel_sovereign.features.peers.wait_provider import A2AWaitable

AGENT_ID = "did:test:sender"


class _StubPeers:
    """Minimal PeersFeature stand-in for the a2a provider."""

    def __init__(self, db, *, states=None, raises=False):
        self._db = db
        self._outbound_route_store_ready = True
        self.agent = SimpleNamespace(did=AGENT_ID)
        self._own_name = "sender"
        self._states = states or {}
        self._raises = raises
        self.fetch_calls = []

    async def get_peer_task_result(self, recipient, task_id):
        self.fetch_calls.append((recipient, task_id))
        if self._raises:
            raise RuntimeError("peer proxy exploded")
        state = self._states.get(task_id)
        if state is None:
            return ToolResult.failed(
                "peer unreachable",
                data={"recipient": recipient, "task_id": task_id},
            )
        return ToolResult.ok(
            "fetched",
            data={"recipient": recipient, "task_id": task_id, "state": state},
        )


@pytest.fixture
async def db(tmp_path, sqlite_database_factory):
    database = await sqlite_database_factory(tmp_path / "agent.db")
    await ensure_a2a_outbound_tasks_table(database)
    return database


async def _record(db, task_id, recipient="Nellie"):
    await record_outbound_dispatch(
        db,
        agent_id=AGENT_ID,
        task_id=task_id,
        recipient=recipient,
        verb="task",
        session_id="sess-1",
        dispatch_tool="send_a2a_task",
    )


# ---------------------------------------------------------------------------
# owns_handle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_owns_handle_true_for_recorded_outbound(db):
    await _record(db, "task-abc")
    provider = A2AWaitable(_StubPeers(db))
    assert await provider.owns_handle("task-abc") is True


@pytest.mark.asyncio
async def test_owns_handle_false_for_unknown_id(db):
    provider = A2AWaitable(_StubPeers(db))
    # No outbound row was ever recorded for this id — it is NOT ours.
    assert await provider.owns_handle("never-sent") is False


@pytest.mark.asyncio
async def test_owns_handle_none_when_store_not_ready(db):
    feature = _StubPeers(db)
    feature._outbound_route_store_ready = False
    provider = A2AWaitable(feature)
    # Unverifiable → None so the caller fails open (doesn't block the watch).
    assert await provider.owns_handle("task-abc") is None


@pytest.mark.asyncio
async def test_owns_handle_none_when_no_db():
    feature = _StubPeers(None)
    feature._db = None
    provider = A2AWaitable(feature)
    assert await provider.owns_handle("task-abc") is None


@pytest.mark.asyncio
async def test_owns_handle_false_for_question_verb(db):
    """#2729 P2: a2a: is for outbound TASKS. A question already resumes via
    its own a2a.question_answered rail, so watching it here (which would emit
    a SECOND wait.complete wake) is rejected at registration."""
    await record_outbound_dispatch(
        db, agent_id=AGENT_ID, task_id="q-1", recipient="Nellie",
        verb="question", session_id="s1", dispatch_tool="send_a2a_question",
    )
    provider = A2AWaitable(_StubPeers(db))
    assert await provider.owns_handle("q-1") is False


@pytest.mark.asyncio
async def test_owns_handle_false_for_message_verb(db):
    """A fire-and-forget message has no terminal lifecycle → not watchable."""
    await record_outbound_dispatch(
        db, agent_id=AGENT_ID, task_id="m-1", recipient="Nellie",
        verb="message", session_id="s1", dispatch_tool="send_a2a_message",
    )
    provider = A2AWaitable(_StubPeers(db))
    assert await provider.owns_handle("m-1") is False


# ---------------------------------------------------------------------------
# poll classification
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_poll_completed_is_done(db):
    await _record(db, "task-done")
    provider = A2AWaitable(_StubPeers(db, states={"task-done": "completed"}))
    status = await provider.poll("task-done")
    assert status.outcome is Outcome.DONE
    assert status.data["state"] == "completed"
    assert status.data["recipient"] == "Nellie"


@pytest.mark.asyncio
async def test_poll_failed_is_failed(db):
    await _record(db, "task-fail")
    provider = A2AWaitable(_StubPeers(db, states={"task-fail": "failed"}))
    status = await provider.poll("task-fail")
    assert status.outcome is Outcome.FAILED


@pytest.mark.asyncio
async def test_poll_submitted_is_pending(db):
    await _record(db, "task-pending")
    provider = A2AWaitable(_StubPeers(db, states={"task-pending": "submitted"}))
    status = await provider.poll("task-pending")
    assert status.outcome is Outcome.PENDING


@pytest.mark.asyncio
async def test_poll_unreachable_peer_stays_pending_not_failed(db):
    """The core #2729 invariant: a momentarily-unreachable peer must NOT be
    converted into a terminal failure — the wait stays armed."""
    await _record(db, "task-live")
    # No state configured → get_peer_task_result returns ToolResult.failed.
    provider = A2AWaitable(_StubPeers(db, states={}))
    status = await provider.poll("task-live")
    assert status.outcome is Outcome.PENDING


@pytest.mark.asyncio
async def test_poll_peer_fetch_raises_stays_pending(db):
    await _record(db, "task-live")
    provider = A2AWaitable(_StubPeers(db, raises=True))
    status = await provider.poll("task-live")
    assert status.outcome is Outcome.PENDING


@pytest.mark.asyncio
async def test_poll_uses_stamped_terminal_without_peer_roundtrip(db):
    """A terminal state already stamped on the outbound audit row is
    authoritative — no peer round-trip is made."""
    await _record(db, "task-stamped")
    await update_outbound_terminal_state(
        db, agent_id=AGENT_ID, task_id="task-stamped", terminal_state="completed"
    )
    feature = _StubPeers(db, states={"task-stamped": "submitted"})
    provider = A2AWaitable(feature)
    status = await provider.poll("task-stamped")
    assert status.outcome is Outcome.DONE
    # Terminal row short-circuited the fetch.
    assert feature.fetch_calls == []


@pytest.mark.asyncio
async def test_poll_question_stays_pending_defense_in_depth(db):
    """#2729 P2 defense-in-depth: even if a question watch slipped past
    registration, poll must NOT emit a terminal — the a2a.question_answered
    rail owns that resumption. A terminal-stamped question row stays PENDING."""
    await record_outbound_dispatch(
        db, agent_id=AGENT_ID, task_id="q-terminal", recipient="Nellie",
        verb="question", session_id="s1", dispatch_tool="send_a2a_question",
    )
    await update_outbound_terminal_state(
        db, agent_id=AGENT_ID, task_id="q-terminal", terminal_state="completed"
    )
    feature = _StubPeers(db, states={"q-terminal": "completed"})
    provider = A2AWaitable(feature)
    status = await provider.poll("q-terminal")
    assert status.outcome is Outcome.PENDING
    assert status.data["verb"] == "question"
    # Never round-tripped the peer for a non-task row.
    assert feature.fetch_calls == []


@pytest.mark.asyncio
async def test_poll_dispatch_failed_is_failed(db):
    """A transport-level dispatch failure stamped at send time is terminal."""
    await record_outbound_dispatch(
        db,
        agent_id=AGENT_ID,
        task_id="task-dispatchfail",
        recipient="Nellie",
        recipient_agent_id="did:peer:nellie",
        verb="task",
        session_id="sess-1",
        dispatch_tool="send_a2a_task",
        error="peer 503",
    )
    provider = A2AWaitable(_StubPeers(db))
    status = await provider.poll("task-dispatchfail")
    assert status.outcome is Outcome.FAILED
    assert status.data["state"] == "dispatch_failed"
