"""Direct contracts for ``TaskFeature.respond_to_a2a_task``.

The receiver-side completion tool: agent A sent a task to agent B via
``send_a2a_question`` / ``send_a2a_message`` / ``send_a2a_task``;
B's cognition turn fires; B calls this tool to transition the task to
a terminal state with reply text in ``status.message.parts[0].text``.
Without it the sender's polling sits forever in WORKING and the sync
Q&A round-trip times out (the bug discovered after PR #1380 shipped
the verbs but before the receiver side could close the loop).
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sdk.tools.result import ToolResultStatus
from kestrel_sovereign.a2a.types import (
    Message,
    Task,
    TaskState,
    TaskStatus,
    TextPart,
)
from kestrel_sovereign.features.tasks.feature import TaskFeature


def _make_feature(initial_state: TaskState = TaskState.SUBMITTED):
    """Return a feature with a stub task_manager that holds one task
    in the given starting state. Mirrors the surface the real feature
    interacts with — get_task + update_status — without needing a
    real SQLite TaskStore."""
    task = Task(
        id="task-1",
        sessionId="sess-1",
        status=TaskStatus(state=initial_state),
        history=[],
        metadata={"sender": "emma", "a2a_verb": "question"},
    )

    state = {"task": task}

    async def get_task(task_id):
        return state["task"] if state["task"].id == task_id else None

    async def update_status(
        task_id, new_state, message=None, agent_name=None,
    ):
        assert state["task"].id == task_id
        # Validate the transition matches the receiver's expected
        # chain. SUBMITTED → WORKING, WORKING → COMPLETED/FAILED/CANCELED.
        prior = state["task"].status.state
        state["task"].status = TaskStatus(state=new_state, message=message)
        if message is not None and state["task"].history is not None:
            state["task"].history.append(message)
        state["transitions"] = state.get("transitions", []) + [
            (prior.value, new_state.value)
        ]
        return state["task"]

    task_manager = MagicMock()
    task_manager.get_task = AsyncMock(side_effect=get_task)
    task_manager.update_status = AsyncMock(side_effect=update_status)

    agent = SimpleNamespace(did="did:test:receiver")
    feature = TaskFeature(agent)
    feature.task_manager = task_manager
    return feature, state


@pytest.mark.asyncio
async def test_completes_submitted_task_chains_through_working():
    """A2A state machine forbids SUBMITTED → COMPLETED directly. The
    tool chains SUBMITTED → WORKING → COMPLETED automatically so the
    receiver's prompt doesn't have to know about the intermediate."""
    feature, state = _make_feature(initial_state=TaskState.SUBMITTED)
    result = await feature.respond_to_a2a_task(
        task_id="task-1",
        content="Yes, three open PRs",
        state="completed",
    )
    assert result.status is ToolResultStatus.OK
    # Final terminal state.
    assert state["task"].status.state == TaskState.COMPLETED
    # Reply text attached to status.message — the canonical A2A spot
    # the sender's send_a2a_question extracts from.
    parts = state["task"].status.message.parts
    assert len(parts) == 1
    assert parts[0].text == "Yes, three open PRs"
    # Both transitions actually fired (not just the second).
    assert state["transitions"] == [
        ("submitted", "working"),
        ("working", "completed"),
    ]


@pytest.mark.asyncio
async def test_completes_working_task_directly():
    """WORKING → COMPLETED is a direct legal transition; no chain."""
    feature, state = _make_feature(initial_state=TaskState.WORKING)
    result = await feature.respond_to_a2a_task(
        task_id="task-1", content="done", state="completed",
    )
    assert result.status is ToolResultStatus.OK
    assert state["task"].status.state == TaskState.COMPLETED
    assert state["transitions"] == [("working", "completed")]


@pytest.mark.asyncio
async def test_failed_state_supported():
    """FAILED is a valid terminal — receiver couldn't fulfill the
    request; sender's polling sees ToolResult.failed with the error
    text in the answer field."""
    feature, state = _make_feature(initial_state=TaskState.SUBMITTED)
    result = await feature.respond_to_a2a_task(
        task_id="task-1",
        content="permission denied: I don't have access to that resource",
        state="failed",
    )
    assert result.status is ToolResultStatus.OK
    assert state["task"].status.state == TaskState.FAILED
    assert (
        state["task"].status.message.parts[0].text
        == "permission denied: I don't have access to that resource"
    )


@pytest.mark.asyncio
async def test_canceled_state_supported():
    """CANCELED = receiver declines the task entirely."""
    feature, state = _make_feature(initial_state=TaskState.SUBMITTED)
    result = await feature.respond_to_a2a_task(
        task_id="task-1", content="declining: out of scope", state="canceled",
    )
    assert result.status is ToolResultStatus.OK
    assert state["task"].status.state == TaskState.CANCELED


@pytest.mark.asyncio
async def test_invalid_state_rejected():
    """Only terminal states (completed/failed/canceled) accepted —
    no transitioning to WORKING or INPUT_REQUIRED via this tool."""
    feature, _ = _make_feature()
    result = await feature.respond_to_a2a_task(
        task_id="task-1", content="meta", state="working",
    )
    assert result.status is ToolResultStatus.ERROR
    assert "terminal" in result.error.lower()


@pytest.mark.asyncio
async def test_unknown_state_string_rejected():
    feature, _ = _make_feature()
    result = await feature.respond_to_a2a_task(
        task_id="task-1", content="x", state="bogus",
    )
    assert result.status is ToolResultStatus.ERROR
    assert "invalid state" in result.error.lower()


@pytest.mark.asyncio
async def test_task_not_found_returns_error():
    """The receiver tries to respond to a task_id that isn't in their
    store. Common case: typo, stale id, or task that was already
    purged. Better to surface this clearly than to crash."""
    feature, _ = _make_feature()
    result = await feature.respond_to_a2a_task(
        task_id="nonexistent", content="x", state="completed",
    )
    assert result.status is ToolResultStatus.ERROR
    assert "not found" in result.error.lower()


@pytest.mark.asyncio
async def test_already_terminal_task_rejected():
    """If the task is already COMPLETED/FAILED/CANCELED, refuse —
    the receiver can't double-respond, and the sender already has
    their answer."""
    feature, _ = _make_feature(initial_state=TaskState.COMPLETED)
    result = await feature.respond_to_a2a_task(
        task_id="task-1", content="late reply", state="completed",
    )
    assert result.status is ToolResultStatus.ERROR
    assert "already terminal" in result.error.lower()


@pytest.mark.asyncio
async def test_task_manager_unavailable():
    """Feature instantiated without a task_manager (e.g. agent
    without A2A configured). Tool returns a clean failure rather
    than crashing on attribute access."""
    feature, _ = _make_feature()
    feature.task_manager = None
    result = await feature.respond_to_a2a_task(
        task_id="task-1", content="x", state="completed",
    )
    assert result.status is ToolResultStatus.ERROR
    assert "task manager not available" in result.error.lower()
