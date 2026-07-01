"""``check_task_status``, ``get_task_result`` and ``list_my_tasks`` must
surface the SENDER'S request text — not just the (often empty) agent
reply slot.

#1433: Meridian's cognition turn called ``check_task_status`` on a fresh
inbound A2A task, saw ``message: None`` (because ``status.message`` is
the agent's reply, set only after ``respond_to_a2a_task``), and
concluded the body was null. The actual question text sat in
``task.history[0]`` the whole time. Conflating those two fields meant
every inbox-poll loop hallucinated "null payload" for legitimate
inbound requests.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from kestrel_sovereign.a2a.types import (
    Message,
    Task,
    TaskState,
    TaskStatus,
    TextPart,
)
from kestrel_sovereign.features.tasks.feature import TaskFeature


def _make_feature_with_inbox_task(
    request_text: str = "PING-1428",
    sender: str = "Emma",
    state: TaskState = TaskState.SUBMITTED,
    reply_text: str | None = None,
):
    user_message = Message(role="user", parts=[TextPart(text=request_text)])
    history = [user_message]

    status_message = None
    if reply_text is not None:
        status_message = Message(
            role="agent", parts=[TextPart(text=reply_text)]
        )

    task = Task(
        id="task-1433",
        sessionId="sess-1",
        status=TaskStatus(state=state, message=status_message),
        history=history,
        metadata={"sender": sender, "a2a_verb": "question"},
    )

    async def get_task(task_id):
        return task if task_id == task.id else None

    feature = TaskFeature(agent=None)
    feature.task_manager = MagicMock()
    feature.task_manager.get_task = get_task
    return feature, task


@pytest.mark.asyncio
async def test_check_task_status_returns_request_content_for_submitted_task():
    """Inbound SUBMITTED task: the receiver must see what was ASKED."""
    feature, task = _make_feature_with_inbox_task(
        request_text="PING-1428", sender="Emma",
    )
    result = await feature.check_task_status(task_id=task.id)
    assert result.status.value == "ok", result
    data = result.data or {}
    assert data["request_content"] == "PING-1428", (
        f"Expected request_content='PING-1428' (the sender's actual question "
        f"text from history[0]), got {data.get('request_content')!r}. See "
        "#1433 — without this, the receiver hallucinates 'null body' for "
        "every inbound task and never answers them."
    )
    assert data["sender"] == "Emma"
    assert data["status"] == "submitted"
    # status.message slot is still None until respond_to_a2a_task runs
    assert data["message"] is None
    # Confirmation text must include the request so the LLM sees it even
    # if it only reads the confirmation line.
    assert "PING-1428" in result.confirmation, result.confirmation
    assert "Emma" in result.confirmation


@pytest.mark.asyncio
async def test_check_task_status_returns_both_request_and_reply_for_completed_task():
    """Terminal COMPLETED task: receiver/sender both see request AND reply."""
    feature, task = _make_feature_with_inbox_task(
        request_text="ping?",
        sender="Emma",
        state=TaskState.COMPLETED,
        reply_text="pong",
    )
    result = await feature.check_task_status(task_id=task.id)
    assert result.status.value == "ok"
    data = result.data or {}
    assert data["request_content"] == "ping?"
    assert data["message"] == "pong"
    # Confirmation surfaces both
    assert "ping?" in result.confirmation
    assert "pong" in result.confirmation


@pytest.mark.asyncio
async def test_list_my_tasks_includes_request_content_per_row():
    """Inbox listing must show what each task is ASKING, not just the
    state. Otherwise the LLM has to call check_task_status per row just
    to learn what the inbox contains."""
    user_message = Message(
        role="user", parts=[TextPart(text="please reply with PONG")]
    )
    task_a = Task(
        id="task-A",
        sessionId="sess-A",
        status=TaskStatus(state=TaskState.SUBMITTED),
        history=[user_message],
        metadata={"sender": "Emma", "a2a_verb": "question"},
    )
    task_b = Task(
        id="task-B",
        sessionId="sess-B",
        status=TaskStatus(state=TaskState.SUBMITTED),
        history=[Message(role="user", parts=[TextPart(text="urgent: status")])],
        metadata={"sender": "Claw", "a2a_verb": "question"},
    )

    async def get_pending_tasks(limit=10):
        return [task_a, task_b][:limit]

    feature = TaskFeature(agent=None)
    feature.task_manager = MagicMock()
    feature.task_manager.get_pending_tasks = get_pending_tasks

    result = await feature.list_my_tasks(limit=10)
    assert result.status.value == "ok"
    rows = (result.data or {}).get("tasks") or []
    assert len(rows) == 2
    row_by_id = {row["task_id"]: row for row in rows}
    assert row_by_id["task-A"]["request_content"] == "please reply with PONG"
    assert row_by_id["task-A"]["sender"] == "Emma"
    assert row_by_id["task-B"]["request_content"] == "urgent: status"
    assert row_by_id["task-B"]["sender"] == "Claw"


@pytest.mark.asyncio
async def test_list_my_tasks_status_filter_queries_full_table_not_pending():
    """A ``status`` filter must query the full task table, not the pending
    (SUBMITTED-only) inbox.

    Regression for #1946: ``list_my_tasks`` advertised filters for all
    TaskState values but always called ``get_pending_tasks()``, whose store
    query is ``WHERE status = 'submitted'``. So ``status="completed"`` (and
    working/failed/canceled) returned empty in production even when matching
    tasks existed. The fix routes a status filter through ``list_tasks`` so the
    DB filters by that state. Here ``get_pending_tasks`` returns nothing (as
    the real store would for non-submitted states); only ``list_tasks``
    surfaces the completed task.
    """
    completed_task = Task(
        id="task-done",
        sessionId="sess-done",
        status=TaskStatus(
            state=TaskState.COMPLETED,
            message=Message(role="agent", parts=[TextPart(text="PONG")]),
        ),
        history=[Message(role="user", parts=[TextPart(text="ping")])],
        metadata={"sender": "Emma"},
    )

    async def get_pending_tasks(limit=10):
        # The real store returns ONLY submitted tasks here — no completed ones.
        return []

    captured = {}

    async def list_tasks(status=None, limit=100, session_id=None, user_id=None):
        captured["status"] = status
        captured["limit"] = limit
        return [completed_task] if status == TaskState.COMPLETED else []

    feature = TaskFeature(agent=None)
    feature.task_manager = MagicMock()
    feature.task_manager.get_pending_tasks = get_pending_tasks
    feature.task_manager.list_tasks = list_tasks

    result = await feature.list_my_tasks(status="completed", limit=5)
    assert result.status.value == "ok", result

    # It must have queried list_tasks with the COMPLETED state, NOT relied on
    # the pending inbox.
    assert captured.get("status") == TaskState.COMPLETED
    assert captured.get("limit") == 5

    rows = (result.data or {}).get("tasks") or []
    assert len(rows) == 1
    assert rows[0]["task_id"] == "task-done"
    assert rows[0]["status"] == "completed"


@pytest.mark.asyncio
async def test_list_my_tasks_invalid_status_rejected():
    """An unknown status is rejected up-front (case-insensitive validation)."""
    feature = TaskFeature(agent=None)
    feature.task_manager = MagicMock()

    result = await feature.list_my_tasks(status="bogus")
    assert result.status.value == "error"
    assert "Invalid status" in (result.error or "")


@pytest.mark.asyncio
async def test_list_my_tasks_type_filter_overfetches_then_truncates():
    """With a ``task_type`` filter, ``limit`` must bound MATCHING tasks, not
    pre-filter rows.

    Codex P2 on #1946: the store ``LIMIT`` was applied by status BEFORE the
    Python-side ``task_type`` filter, so a page of non-matching tasks could hide
    matches just beyond it — ``list_my_tasks(status="completed",
    task_type="foo", limit=1)`` could return empty even though a ``foo`` task
    existed. The fix over-fetches (bounded) before the metadata filter and
    truncates to ``limit`` after. Here only the 3rd row matches ``task_type``;
    with ``limit=1`` a naive pre-filter limit would fetch just the 1st row and
    miss it."""

    def _task(i, ttype):
        return Task(
            id=f"task-{i}",
            sessionId=f"sess-{i}",
            status=TaskStatus(
                state=TaskState.COMPLETED,
                message=Message(role="agent", parts=[TextPart(text="done")]),
            ),
            history=[Message(role="user", parts=[TextPart(text="do it")])],
            metadata={"task_type": ttype},
        )

    rows = [_task(0, "other"), _task(1, "other"), _task(2, "foo")]
    captured = {}

    async def list_tasks(status=None, limit=100, session_id=None, user_id=None):
        captured["limit"] = limit
        # The store honours whatever limit it is given; return up to that many.
        return rows[:limit]

    feature = TaskFeature(agent=None)
    feature.task_manager = MagicMock()
    feature.task_manager.list_tasks = list_tasks

    result = await feature.list_my_tasks(status="completed", task_type="foo", limit=1)
    assert result.status.value == "ok", result

    # It must have over-fetched past the caller's limit=1 so the matching row
    # (index 2) was in the fetched window.
    assert captured["limit"] > 1

    out = (result.data or {}).get("tasks") or []
    # Exactly the one matching task, truncated to limit.
    assert len(out) == 1
    assert out[0]["task_id"] == "task-2"
    assert out[0]["task_type"] == "foo"


@pytest.mark.asyncio
async def test_get_task_result_includes_request_content_for_completed_task():
    feature, task = _make_feature_with_inbox_task(
        request_text="please ack",
        state=TaskState.COMPLETED,
        reply_text="acknowledged",
    )
    result = await feature.get_task_result(task_id=task.id)
    assert result.status.value == "ok"
    data = result.data or {}
    assert data["request_content"] == "please ack"
    assert data["message"] == "acknowledged"
    assert data["sender"] == "Emma"


@pytest.mark.asyncio
async def test_check_task_status_surfaces_sender_attached_artifacts():
    """Recipient retrieval (#1525): a SENDER-attached artifact (text body
    + a structured-data reference) must be readable by the recipient via
    ``check_task_status``. The text part is surfaced as ``text`` and the
    data part as ``data``, with ordering and metadata preserved."""
    from kestrel_sovereign.a2a.types import Artifact, DataPart

    user_message = Message(role="user", parts=[TextPart(text="orchestrate")])
    task = Task(
        id="task-1525",
        sessionId="sess-1",
        status=TaskStatus(state=TaskState.SUBMITTED),
        history=[user_message],
        metadata={"sender": "Emma", "a2a_verb": "task"},
        artifacts=[
            Artifact(
                name="plan",
                description="proactive operating model",
                parts=[TextPart(text="step one"), TextPart(text=" step two")],
                index=0,
            ),
            Artifact(
                name="references",
                parts=[DataPart(data={"ref_type": "memory", "id": "m1"})],
                index=0,
                metadata={"kind": "reference"},
            ),
        ],
    )

    async def get_task(task_id):
        return task if task_id == task.id else None

    feature = TaskFeature(agent=None)
    feature.task_manager = MagicMock()
    feature.task_manager.get_task = get_task

    result = await feature.check_task_status(task_id=task.id)
    assert result.status.value == "ok", result
    arts = (result.data or {}).get("artifacts") or []
    assert [a["name"] for a in arts] == ["plan", "references"]
    # Text parts concatenated in order so a chunked body reassembles.
    assert arts[0]["text"] == "step one step two"
    assert arts[0]["description"] == "proactive operating model"
    # Structured reference survives as data, not stringified.
    assert arts[1]["data"] == {"ref_type": "memory", "id": "m1"}
