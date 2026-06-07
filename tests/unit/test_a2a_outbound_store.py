"""Tests for the sender-side outbound A2A audit store (#1576).

Pins:

* Schema creation is idempotent.
* Every outbound dispatch from PeersFeature writes one audit row with
  the assertion Emma pinned: task_id, recipient, verb, dispatch_tool,
  created_at, terminal/error state when known.
* Transport failures (connect / 503 / 404 / raise_for_status) DO
  write a row, with ``error`` populated and ``terminal_state =
  "dispatch_failed"``.
* ``get_peer_task_result`` stamps terminal_state on the audit row
  when the peer reports a terminal lifecycle state.
* ``list_outbound_a2a_tasks`` returns the rows for the agent's own
  introspection surface.
* Recipient-filter and limit-clamp behave as documented.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from kestrel_sdk.tools.result import ToolResultStatus
from kestrel_sovereign.a2a.outbound_store import (
    OutboundTask,
    ensure_a2a_outbound_tasks_table,
    list_outbound_tasks,
    record_outbound_dispatch,
    update_outbound_terminal_state,
)
from kestrel_sovereign.features.peers.feature import PeersFeature
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.db import SQLiteBackend


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


async def _backend(tmp_path):
    raw = SQLiteBackend(str(tmp_path / "outbound-test.db"))
    await raw.connect()
    db = AsyncDatabase(raw)
    await ensure_a2a_outbound_tasks_table(db)
    return db


def _agent_with_db(db, name="emma"):
    """Mimic ``resolve_feature_database`` resolution path."""
    raw_storage = SimpleNamespace(db=db)
    agent = SimpleNamespace(
        _agent_name=name,
        _raw_storage=raw_storage,
        storage=None,
        _provide_causation_chain=None,
        _track_background_task=lambda c, *, name="": c.close() or MagicMock(),
        _get_current_turn_id=MagicMock(return_value=None),
        pending_a2a_questions=MagicMock(
            insert=AsyncMock(return_value=None),
            mark_resolved=AsyncMock(return_value=True),
        ),
        dispatcher=MagicMock(enqueue_signal=AsyncMock()),
    )
    return agent


async def _make_feature(tmp_path, name="emma"):
    db = await _backend(tmp_path)
    agent = _agent_with_db(db, name)
    feature = PeersFeature(agent)
    await feature.initialize()
    # initialize() doesn't set _host_url if no env var; force it.
    feature._host_url = "http://multi_agent"
    feature._api_key = ""
    feature._own_name = name
    return feature, db


def _mock_post_response(task_id="t1", session_id="s1", state="submitted"):
    response = MagicMock(status_code=200)
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "id": task_id,
        "sessionId": session_id,
        "status": {"state": state},
    }
    return response


def _async_client_with(post_resp=None, get_resp=None):
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.__aexit__.return_value = False
    if post_resp is not None:
        client.post.return_value = post_resp
    if get_resp is not None:
        client.get.return_value = get_resp
    return client


# ---------------------------------------------------------------------------
# Pure store contracts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ensure_table_idempotent(tmp_path):
    db = await _backend(tmp_path)
    await ensure_a2a_outbound_tasks_table(db)  # second call is a no-op


@pytest.mark.asyncio
async def test_record_outbound_dispatch_returns_full_row(tmp_path):
    db = await _backend(tmp_path)
    row = await record_outbound_dispatch(
        db,
        agent_id='emma',
        task_id="abc123",
        recipient="claw",
        verb="task",
        session_id="s1",
        dispatch_tool="send_a2a_task",
        skill_id="soul_alignment",
        message="please align SOUL.md",
    )
    assert isinstance(row, OutboundTask)
    assert row.task_id == "abc123"
    assert row.recipient == "claw"
    assert row.verb == "task"
    assert row.dispatch_tool == "send_a2a_task"
    assert row.skill_id == "soul_alignment"
    assert row.message_summary == "please align SOUL.md"
    assert row.terminal_state is None
    assert row.error is None


@pytest.mark.asyncio
async def test_record_with_error_stamps_dispatch_failed_terminal(tmp_path):
    db = await _backend(tmp_path)
    row = await record_outbound_dispatch(
        db,
        agent_id='emma',
        task_id="failedtask",
        recipient="claw",
        verb="task",
        session_id="s",
        dispatch_tool="send_a2a_task",
        error="connect_error:claw",
    )
    assert row.error == "connect_error:claw"
    assert row.terminal_state == "dispatch_failed"
    assert row.terminal_at is not None


@pytest.mark.asyncio
async def test_message_summary_truncates_at_200_chars(tmp_path):
    db = await _backend(tmp_path)
    long_msg = "x" * 5000
    row = await record_outbound_dispatch(
        db,
        agent_id='emma',
        task_id="t1", recipient="c", verb="task",
        session_id="s", dispatch_tool="send_a2a_task",
        message=long_msg,
    )
    assert row.message_summary is not None
    assert len(row.message_summary) <= 200
    assert row.message_summary.endswith("...")


@pytest.mark.asyncio
async def test_update_terminal_state_idempotent(tmp_path):
    db = await _backend(tmp_path)
    await record_outbound_dispatch(
        db,
        agent_id='emma', task_id="t1", recipient="c", verb="task",
        session_id="s", dispatch_tool="send_a2a_task",
    )
    updated = await update_outbound_terminal_state(
        db,
        agent_id='emma', task_id="t1", terminal_state="completed",
    )
    assert updated >= 1
    # Second update is a no-op because WHERE terminal_state IS NULL.
    again = await update_outbound_terminal_state(
        db,
        agent_id='emma', task_id="t1", terminal_state="failed",
    )
    assert again == 0
    rows = await list_outbound_tasks(db,         agent_id='emma',
limit=10)
    assert rows[0].terminal_state == "completed"


@pytest.mark.asyncio
async def test_list_outbound_filters_by_recipient(tmp_path):
    db = await _backend(tmp_path)
    for recipient in ("claw", "nellie", "claw", "meridian"):
        await record_outbound_dispatch(
            db,
        agent_id='emma', task_id=f"t-{recipient}", recipient=recipient,
            verb="task", session_id="s", dispatch_tool="send_a2a_task",
        )
    rows_claw = await list_outbound_tasks(db,         agent_id='emma',
recipient="claw")
    assert len(rows_claw) == 2
    assert all(r.recipient == "claw" for r in rows_claw)
    rows_all = await list_outbound_tasks(db, agent_id='emma')
    assert len(rows_all) == 4


@pytest.mark.asyncio
async def test_list_outbound_clamps_limit(tmp_path):
    db = await _backend(tmp_path)
    await record_outbound_dispatch(
        db,
        agent_id='emma', task_id="t", recipient="c", verb="task",
        session_id="s", dispatch_tool="send_a2a_task",
    )
    # Negative → clamped to 1; doesn't raise.
    rows = await list_outbound_tasks(db,         agent_id='emma',
limit=-5)
    assert len(rows) == 1


# ---------------------------------------------------------------------------
# PeersFeature wiring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_send_a2a_task_writes_outbound_audit_row(tmp_path):
    """Emma's pinned acceptance assertion: every outbound A2A dispatch
    writes a sender-side audit row with task_id, recipient, verb,
    created_at, dispatch_tool."""
    feature, db = await _make_feature(tmp_path)
    post_resp = _mock_post_response(task_id="taskA", session_id="sA")
    client = _async_client_with(post_resp=post_resp)
    with patch(
        "kestrel_sovereign.features.peers.feature.httpx.AsyncClient",
        return_value=client,
    ):
        result = await feature.send_a2a_task("claw", "align SOUL.md")
    assert result.status is ToolResultStatus.OK

    rows = await list_outbound_tasks(db, agent_id='emma')
    assert len(rows) == 1
    row = rows[0]
    # The audit row records the LOCALLY-generated task_id we sent on
    # the wire (which the peer echoes back in production). It equals
    # the task_id surfaced to the caller, even if a test mock returns
    # a different id from .json(): _post_a2a_task's local id is what
    # the peer is told and what the audit row pins.
    assert row.task_id == result.data["task_id"]
    assert row.recipient == "claw"
    assert row.verb == "task"
    assert row.dispatch_tool == "send_a2a_task"
    assert row.error is None
    assert row.terminal_state is None  # not yet fetched


@pytest.mark.asyncio
async def test_send_a2a_message_writes_outbound_audit_row(tmp_path):
    feature, db = await _make_feature(tmp_path)
    post_resp = _mock_post_response(task_id="msgA")
    client = _async_client_with(post_resp=post_resp)
    with patch(
        "kestrel_sovereign.features.peers.feature.httpx.AsyncClient",
        return_value=client,
    ):
        await feature.send_a2a_message("nellie", "FYI shipped PR 42")
    rows = await list_outbound_tasks(db, agent_id='emma')
    assert len(rows) == 1
    assert rows[0].verb == "message"
    assert rows[0].dispatch_tool == "send_a2a_message"


@pytest.mark.asyncio
async def test_send_a2a_question_writes_outbound_audit_row(tmp_path):
    feature, db = await _make_feature(tmp_path)
    post_resp = _mock_post_response(task_id="qA", state="working")
    client = _async_client_with(post_resp=post_resp)
    with patch(
        "kestrel_sovereign.features.peers.feature.httpx.AsyncClient",
        return_value=client,
    ):
        await feature.send_a2a_question("meridian", "what's the schema?")
    rows = await list_outbound_tasks(db, agent_id='emma')
    assert len(rows) == 1
    assert rows[0].verb == "question"
    assert rows[0].dispatch_tool == "send_a2a_question"


@pytest.mark.asyncio
async def test_dispatch_transport_failure_writes_audit_row_with_error(
    tmp_path,
):
    """A peer that's unreachable / 5xx must still leave a sender-side
    audit row so the agent can see attempted dispatches that didn't
    land. Emma's pinned 'terminal/error state when known' demand."""
    feature, db = await _make_feature(tmp_path)
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.__aexit__.return_value = False
    client.post.side_effect = httpx.ConnectError("peer down")
    with patch(
        "kestrel_sovereign.features.peers.feature.httpx.AsyncClient",
        return_value=client,
    ):
        result = await feature.send_a2a_task("claw", "x")
    assert result.status is ToolResultStatus.ERROR
    rows = await list_outbound_tasks(db, agent_id='emma')
    assert len(rows) == 1
    assert rows[0].terminal_state == "dispatch_failed"
    assert "connect_error" in (rows[0].error or "")


@pytest.mark.asyncio
async def test_get_peer_task_result_stamps_terminal_state(tmp_path):
    """When the agent fetches a peer's result and the peer reports a
    terminal lifecycle state, the sender-side audit row's
    terminal_state is updated. Closes the loop on Emma's pinned
    assertion."""
    feature, db = await _make_feature(tmp_path)

    # First: dispatch. Use the LOCAL task_id (what got audited).
    post_resp = _mock_post_response(task_id="loopA", state="submitted")
    client = _async_client_with(post_resp=post_resp)
    with patch(
        "kestrel_sovereign.features.peers.feature.httpx.AsyncClient",
        return_value=client,
    ):
        result = await feature.send_a2a_task("claw", "do work")
    local_id = result.data["task_id"]

    rows = await list_outbound_tasks(db, agent_id='emma')
    assert rows[0].terminal_state is None

    # Then: fetch a completed result from the peer using the same
    # local id (production: peer echoes; here: we just use it).
    get_resp = MagicMock(status_code=200)
    get_resp.json.return_value = {
        "id": local_id,
        "status": {
            "state": "completed",
            "message": {"role": "agent",
                        "parts": [{"type": "text", "text": "done"}]},
        },
    }
    client = _async_client_with(get_resp=get_resp)
    with patch(
        "kestrel_sovereign.features.peers.feature.httpx.AsyncClient",
        return_value=client,
    ):
        await feature.get_peer_task_result("claw", local_id)

    rows = await list_outbound_tasks(db, agent_id='emma')
    assert rows[0].terminal_state == "completed"


@pytest.mark.asyncio
async def test_get_peer_task_result_does_not_stamp_non_terminal_state(
    tmp_path,
):
    """Interim states (working, submitted) must NOT be stamped — the
    row stays NULL until a terminal fetch lands. Otherwise we'd
    mislead introspection about what is and isn't settled."""
    feature, db = await _make_feature(tmp_path)
    post_resp = _mock_post_response(task_id="wipA", state="submitted")
    client = _async_client_with(post_resp=post_resp)
    with patch(
        "kestrel_sovereign.features.peers.feature.httpx.AsyncClient",
        return_value=client,
    ):
        result = await feature.send_a2a_task("claw", "wip")
    local_id = result.data["task_id"]

    get_resp = MagicMock(status_code=200)
    get_resp.json.return_value = {
        "id": local_id,
        "status": {"state": "working"},
    }
    client = _async_client_with(get_resp=get_resp)
    with patch(
        "kestrel_sovereign.features.peers.feature.httpx.AsyncClient",
        return_value=client,
    ):
        await feature.get_peer_task_result("claw", local_id)

    rows = await list_outbound_tasks(db, agent_id='emma')
    assert rows[0].terminal_state is None


@pytest.mark.asyncio
async def test_fire_question_answered_signal_stamps_audit(tmp_path):
    """Codex review #1576 round 2 P1: ``_fire_question_answered_signal``
    is the single chokepoint for the supervisor's terminal-state
    observation (SSE terminal, deadline expiry, hourly sweep, startup
    replay). EVERY terminal state seen there must stamp the outbound
    audit row, not just states learned via ``get_peer_task_result``."""
    feature, db = await _make_feature(tmp_path)

    # Seed an outbound row by dispatching a question.
    post_resp = _mock_post_response(task_id="qSSE", state="submitted")
    client = _async_client_with(post_resp=post_resp)
    with patch(
        "kestrel_sovereign.features.peers.feature.httpx.AsyncClient",
        return_value=client,
    ):
        result = await feature.send_a2a_question("nellie", "what?")
    local_id = result.data["task_id"]
    assert (await list_outbound_tasks(db, agent_id='emma'))[0].terminal_state is None

    # Fire the terminal signal directly (simulates SSE terminal frame).
    await feature._fire_question_answered_signal(
        task_id=local_id,
        recipient="nellie",
        original_question="what?",
        sess_id="s",
        state="completed",
        reply_text="done",
        causation_chain=None,
    )
    assert (await list_outbound_tasks(db, agent_id='emma'))[0].terminal_state == "completed"


@pytest.mark.asyncio
async def test_fire_question_answered_signal_stamps_expired_state(tmp_path):
    """Same chokepoint, expired path: a question that times out at the
    deadline fires the synthetic ``state='expired'`` signal — the
    audit row must record that too."""
    feature, db = await _make_feature(tmp_path)
    post_resp = _mock_post_response(task_id="qExp", state="working")
    client = _async_client_with(post_resp=post_resp)
    with patch(
        "kestrel_sovereign.features.peers.feature.httpx.AsyncClient",
        return_value=client,
    ):
        result = await feature.send_a2a_question("nellie", "slow?")
    local_id = result.data["task_id"]

    await feature._fire_question_answered_signal(
        task_id=local_id,
        recipient="nellie",
        original_question="slow?",
        sess_id="s",
        state="expired",
        reply_text="",
        causation_chain=None,
    )
    assert (await list_outbound_tasks(db, agent_id='emma'))[0].terminal_state == "expired"


@pytest.mark.asyncio
async def test_list_outbound_a2a_tasks_tool_returns_rows(tmp_path):
    feature, db = await _make_feature(tmp_path)
    # Seed two dispatches; capture the locally-generated task_ids.
    seeded = {}
    for tid_hint, rec in (("a", "claw"), ("b", "nellie")):
        post_resp = _mock_post_response(task_id=tid_hint)
        client = _async_client_with(post_resp=post_resp)
        with patch(
            "kestrel_sovereign.features.peers.feature.httpx.AsyncClient",
            return_value=client,
        ):
            r = await feature.send_a2a_task(rec, f"msg-{tid_hint}")
            seeded[rec] = r.data["task_id"]

    result = await feature.list_outbound_a2a_tasks()
    assert result.status is ToolResultStatus.OK
    assert result.data["count"] == 2
    task_ids = {r["task_id"] for r in result.data["rows"]}
    assert task_ids == set(seeded.values())

    # Recipient filter.
    only_claw = await feature.list_outbound_a2a_tasks(recipient="claw")
    assert only_claw.data["count"] == 1
    assert only_claw.data["rows"][0]["recipient"] == "claw"


@pytest.mark.asyncio
async def test_agent_id_isolates_rows_in_shared_backend(tmp_path):
    """Codex review #1576 round 3 P1: in a shared-backend deployment
    (multiple agents writing to one Postgres), the audit table must
    not leak rows across agents. Recording for agent A and listing
    for agent B returns nothing."""
    db = await _backend(tmp_path)
    await record_outbound_dispatch(
        db, agent_id="emma", task_id="t-emma",
        recipient="claw", verb="task", session_id="s",
        dispatch_tool="send_a2a_task",
    )
    await record_outbound_dispatch(
        db, agent_id="meridian", task_id="t-meridian",
        recipient="claw", verb="task", session_id="s",
        dispatch_tool="send_a2a_task",
    )
    emma_rows = await list_outbound_tasks(db, agent_id="emma")
    meridian_rows = await list_outbound_tasks(db, agent_id="meridian")
    other_rows = await list_outbound_tasks(db, agent_id="nobody")
    assert {r.task_id for r in emma_rows} == {"t-emma"}
    assert {r.task_id for r in meridian_rows} == {"t-meridian"}
    assert other_rows == []


@pytest.mark.asyncio
async def test_agent_id_isolates_terminal_stamp(tmp_path):
    """A terminal-state update by agent A must not silently overwrite
    a same-task_id row owned by agent B, even on a collision."""
    db = await _backend(tmp_path)
    await record_outbound_dispatch(
        db, agent_id="emma", task_id="collision",
        recipient="claw", verb="task", session_id="s",
        dispatch_tool="send_a2a_task",
    )
    await record_outbound_dispatch(
        db, agent_id="meridian", task_id="collision",
        recipient="claw", verb="task", session_id="s",
        dispatch_tool="send_a2a_task",
    )
    # Emma stamps completed; Meridian's same-task_id row stays NULL.
    updated = await update_outbound_terminal_state(
        db, agent_id="emma", task_id="collision",
        terminal_state="completed",
    )
    assert updated >= 1
    emma_row = (await list_outbound_tasks(db, agent_id="emma"))[0]
    meridian_row = (await list_outbound_tasks(db, agent_id="meridian"))[0]
    assert emma_row.terminal_state == "completed"
    assert meridian_row.terminal_state is None


@pytest.mark.asyncio
async def test_dispatch_without_db_does_not_break(tmp_path):
    """An agent without ``_raw_storage.db`` (early-init agent, test
    stub) must still dispatch — the audit write is best-effort."""
    agent = SimpleNamespace(
        _agent_name="emma",
        _raw_storage=None,
        storage=None,
        _provide_causation_chain=None,
        _track_background_task=lambda c, *, name="": c.close() or MagicMock(),
        _get_current_turn_id=MagicMock(return_value=None),
        pending_a2a_questions=MagicMock(
            insert=AsyncMock(return_value=None),
            mark_resolved=AsyncMock(return_value=True),
        ),
        dispatcher=MagicMock(enqueue_signal=AsyncMock()),
    )
    feature = PeersFeature(agent)
    await feature.initialize()
    feature._host_url = "http://multi_agent"
    feature._api_key = ""
    feature._own_name = "emma"
    assert feature._db is None

    post_resp = _mock_post_response(task_id="nodbA")
    client = _async_client_with(post_resp=post_resp)
    with patch(
        "kestrel_sovereign.features.peers.feature.httpx.AsyncClient",
        return_value=client,
    ):
        result = await feature.send_a2a_task("claw", "no db here")
    assert result.status is ToolResultStatus.OK
