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
async def test_update_status_value_error_becomes_tool_failure():
    """Codex review (non-blocking) on PR #1387: explicit race-contract
    coverage. Between this tool's ``get_task`` and ``update_status``,
    another caller could mutate the task. ``update_status`` re-reads
    and re-validates the transition; if it now violates VALID_TRANSITIONS
    it raises ValueError. The tool must convert that into a ToolResult
    failure (not let it propagate) — matches the behavior of every
    other ``@tool``-decorated method that interacts with task_manager."""
    feature, _ = _make_feature(initial_state=TaskState.WORKING)
    # Simulate the race by having update_status raise ValueError on the
    # terminal transition. (Real cause would be another caller having
    # already moved the task to a terminal state in the gap.)
    feature.task_manager.update_status = AsyncMock(
        side_effect=ValueError(
            "Invalid state transition: TaskState.WORKING -> TaskState.COMPLETED. "
            "Valid transitions: set()"
        ),
    )
    result = await feature.respond_to_a2a_task(
        task_id="task-1", content="answer", state="completed",
    )
    assert result.status is ToolResultStatus.ERROR
    assert "invalid state transition" in result.error.lower()


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


# ---------------------------------------------------------------------------
# attach_artifact_to_a2a_task — long-reply path via Artifact chunking
# ---------------------------------------------------------------------------

def _make_feature_with_artifact_tracking(initial_state: TaskState = TaskState.WORKING):
    """Same shape as ``_make_feature`` but also tracks ``add_artifact``
    calls so the tests can verify chunked-reply ordering and metadata."""
    task = Task(
        id="task-art",
        sessionId="sess-art",
        status=TaskStatus(state=initial_state),
        history=[],
        artifacts=[],
        metadata={"sender": "emma", "a2a_verb": "question"},
    )
    state = {"task": task, "added": []}

    async def get_task(task_id):
        return state["task"] if state["task"].id == task_id else None

    async def add_artifact(task_id, artifact, agent_name=None):
        assert state["task"].id == task_id
        state["added"].append(artifact)
        if state["task"].artifacts is None:
            state["task"].artifacts = []
        state["task"].artifacts.append(artifact)
        return state["task"]

    task_manager = MagicMock()
    task_manager.get_task = AsyncMock(side_effect=get_task)
    task_manager.add_artifact = AsyncMock(side_effect=add_artifact)

    agent = SimpleNamespace(did="did:test:receiver")
    feature = TaskFeature(agent)
    feature.task_manager = task_manager
    return feature, state


@pytest.mark.asyncio
async def test_attach_artifact_single_chunk_short_reply():
    """The happy path for a ~5K reply that still fits in one segment.
    One call, ``last_chunk=True``, body shows up as a single Artifact
    with ``index=0`` and ``lastChunk=True``."""
    feature, state = _make_feature_with_artifact_tracking()
    result = await feature.attach_artifact_to_a2a_task(
        task_id="task-art", name="reply_body",
        content="A" * 5000, index=0, last_chunk=True,
    )
    assert result.status is ToolResultStatus.OK
    assert result.data["content_chars"] == 5000
    assert len(state["added"]) == 1
    art = state["added"][0]
    assert art.name == "reply_body"
    assert art.index == 0
    assert art.lastChunk is True
    assert art.parts[0].text == "A" * 5000


@pytest.mark.asyncio
async def test_attach_artifact_chunked_multi_segment_order_preserved():
    """A long reply chunked into 3 segments. Each call independently
    records the right index + last_chunk flag so the sender's
    ``get_peer_task_result`` can reassemble in index order with the
    final-segment marker."""
    feature, state = _make_feature_with_artifact_tracking()
    for i, (chunk, last) in enumerate([
        ("X" * 9000, False),
        ("Y" * 9000, False),
        ("Z" * 3000, True),
    ]):
        result = await feature.attach_artifact_to_a2a_task(
            task_id="task-art", name="reply_body",
            content=chunk, index=i, last_chunk=last,
        )
        assert result.status is ToolResultStatus.OK
        assert result.data["index"] == i
        assert result.data["last_chunk"] is last

    assert len(state["added"]) == 3
    assert [a.index for a in state["added"]] == [0, 1, 2]
    assert [a.lastChunk for a in state["added"]] == [False, False, True]
    assert state["added"][0].parts[0].text == "X" * 9000
    assert state["added"][2].parts[0].text == "Z" * 3000


@pytest.mark.asyncio
async def test_attach_artifact_returns_failed_when_task_manager_missing():
    """Standalone agent with no task manager — graceful fail rather
    than AttributeError so the LLM's error-handling path can react."""
    feature, _ = _make_feature_with_artifact_tracking()
    feature.task_manager = None
    result = await feature.attach_artifact_to_a2a_task(
        task_id="task-art", name="reply_body",
        content="x", index=0, last_chunk=True,
    )
    assert result.status is ToolResultStatus.ERROR
    assert "task manager" in result.error.lower()


@pytest.mark.asyncio
async def test_attach_artifact_surfaces_task_not_found():
    """``task_manager.add_artifact`` raises ValueError for unknown
    task ids — should surface as a clean failed ToolResult, not a
    raw exception."""
    feature, _ = _make_feature_with_artifact_tracking()
    feature.task_manager.add_artifact = AsyncMock(
        side_effect=ValueError("Task not found: task-art"),
    )
    result = await feature.attach_artifact_to_a2a_task(
        task_id="task-art", name="reply_body",
        content="x", index=0, last_chunk=True,
    )
    assert result.status is ToolResultStatus.ERROR
    assert "task-art" in result.error
