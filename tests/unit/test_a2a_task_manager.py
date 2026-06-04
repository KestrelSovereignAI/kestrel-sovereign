"""
Unit tests for A2A TaskManager and TaskWorker.

Tests task lifecycle management and background processing.
"""

import asyncio
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sovereign.a2a.types import (
    Task,
    TaskState,
    TaskStatus,
    Message,
    TextPart,
    Artifact,
    TaskSendParams,
)
from kestrel_sovereign.a2a.task_manager import TaskManager, create_task_manager
from kestrel_sovereign.a2a.agent_card import AgentCapabilities, AgentCard, AgentSkill
from kestrel_sovereign.a2a.task_worker import (
    TaskWorker,
    TaskHandler,
    TaskResult,
    SimpleTaskHandler,
    LLMTaskHandler,
)
from kestrel_sovereign.a2a.stores import (
    SQLiteTaskStore,
    SQLiteSessionService,
    SQLiteObservabilityStore,
    SQLiteMemoryService,
    SQLiteFeedbackStore,
)


import pytest_asyncio


@pytest.fixture
def db_path():
    """Create a temporary database file."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    try:
        os.unlink(path)
    except PermissionError:
        import time
        time.sleep(0.1)
        try:
            os.unlink(path)
        except PermissionError:
            pass  # Windows file locking; temp dir cleanup handles it


# Registry to track task managers for cleanup
_managers_to_close = []


@pytest_asyncio.fixture(autouse=True)
async def _async_cleanup_managers():
    """Async cleanup for task managers after each test."""
    yield
    for manager in _managers_to_close:
        try:
            # Close all the underlying stores
            if hasattr(manager, 'task_store'):
                await manager.task_store.close()
            if hasattr(manager, 'session_service'):
                await manager.session_service.close()
            if hasattr(manager, 'observability_store'):
                await manager.observability_store.close()
            if hasattr(manager, 'memory_service'):
                await manager.memory_service.close()
            if hasattr(manager, 'feedback_store'):
                await manager.feedback_store.close()
        except Exception:
            pass
    _managers_to_close.clear()


def track_manager(manager):
    """Register a task manager for cleanup after the test."""
    _managers_to_close.append(manager)
    return manager


@pytest_asyncio.fixture
async def task_manager(db_path):
    """Create an initialized TaskManager."""
    manager = await create_task_manager(db_path)
    track_manager(manager)
    yield manager


# =============================================================================
# TaskManager Tests
# =============================================================================

class TestTaskManager:
    """Tests for TaskManager."""

    @pytest.mark.asyncio
    async def test_create_task_manager(self, db_path):
        """Test creating a TaskManager with factory function."""
        manager = track_manager(await create_task_manager(db_path))
        assert manager is not None
        assert manager.task_store is not None
        assert manager.session_service is not None
        assert manager.observability_store is not None

    @pytest.mark.asyncio
    async def test_async_execute_skill_task_is_owned_and_cancelled_on_close(self):
        """Background skill execution must be cancelled before stores close."""
        call_order = []

        task_store = MagicMock()

        async def save_task(task):
            call_order.append(f"save:{task.status.state.value}")

        async def close_task_store():
            call_order.append("close:task_store")

        task_store.save = AsyncMock(side_effect=save_task)
        task_store.close = AsyncMock(side_effect=close_task_store)
        session_service = MagicMock()
        session_service.close = AsyncMock()
        observability_store = MagicMock()
        observability_store.close = AsyncMock()

        manager = TaskManager(
            task_store=task_store,
            session_service=session_service,
            observability_store=observability_store,
        )

        class BlockingHandler:
            name = "blocking-handler"

            def __init__(self):
                self.started = asyncio.Event()

            async def handle_task(self, task):
                self.started.set()
                await asyncio.Event().wait()

            def get_skill_for_command(self, command):
                return None

        handler = BlockingHandler()
        manager.register_agent(
            AgentCard(
                name="agent",
                url="/agents/agent",
                version="1.0.0",
                capabilities=AgentCapabilities(),
                skills=[
                    AgentSkill(
                        id="slow_skill",
                        name="slow_skill",
                        description="Slow skill",
                    )
                ],
            ),
            handler,
        )

        task = await manager.execute_skill(
            agent_id="agent",
            skill_id="slow_skill",
            args={},
            sync=False,
        )
        await handler.started.wait()

        assert len(manager._execution_tasks) == 1

        execution_task = next(iter(manager._execution_tasks))

        await manager.close()

        assert execution_task.done()
        assert execution_task.cancelled()
        assert manager._execution_tasks == set()
        assert call_order == [
            "save:submitted",
            "save:canceled",
            "close:task_store",
        ]

    @pytest.mark.asyncio
    async def test_create_task(self, task_manager):
        """Test creating a task."""
        params = TaskSendParams(
            message=Message(
                role="user",
                parts=[TextPart(text="Hello, agent!")],
            ),
            metadata={"source": "test"},
        )

        task = await task_manager.create_task(params, agent_name="test-agent")

        assert task is not None
        assert task.id == params.id
        assert task.sessionId == params.sessionId
        assert task.status.state == TaskState.SUBMITTED
        assert len(task.history) == 1

    @pytest.mark.asyncio
    async def test_create_task_persists_sender_artifacts(self, task_manager):
        """Send-side: artifacts attached at create time are persisted on
        the task at SUBMITTED so the recipient can retrieve them from the
        store before producing any response. Covers recipient retrieval
        and ordering/metadata preservation (#1525)."""
        from kestrel_sovereign.a2a.types import DataPart

        params = TaskSendParams(
            message=Message(role="user", parts=[TextPart(text="Orchestrate this plan")]),
        )
        sender_artifacts = [
            Artifact(
                name="plan",
                description="proactive operating model",
                parts=[TextPart(text="step one")],
                index=0,
                metadata={"origin": "saved_item"},
            ),
            Artifact(
                name="references",
                parts=[DataPart(data={"ref_type": "memory", "id": "m1"})],
                index=0,
                metadata={"kind": "reference"},
            ),
        ]

        created = await task_manager.create_task(
            params, agent_name="test-agent", artifacts=sender_artifacts,
        )
        assert created.artifacts is not None
        assert len(created.artifacts) == 2

        # Recipient retrieval: round-trip through the store.
        fetched = await task_manager.task_store.get(params.id)
        assert fetched is not None
        assert fetched.artifacts is not None
        assert len(fetched.artifacts) == 2
        # Ordering preserved: plan first, references second.
        assert fetched.artifacts[0].name == "plan"
        assert fetched.artifacts[0].metadata == {"origin": "saved_item"}
        assert fetched.artifacts[0].parts[0].text == "step one"
        # Structured metadata (DataPart) survives, not only raw text.
        assert fetched.artifacts[1].name == "references"
        assert fetched.artifacts[1].parts[0].data == {"ref_type": "memory", "id": "m1"}

    @pytest.mark.asyncio
    async def test_create_task_without_artifacts_leaves_none(self, task_manager):
        """No send-side artifacts → task.artifacts stays None (existing
        responder-side attach flow remains the only writer)."""
        params = TaskSendParams(
            message=Message(role="user", parts=[TextPart(text="hi")]),
        )
        task = await task_manager.create_task(params, agent_name="test-agent")
        assert task.artifacts is None

    @pytest.mark.asyncio
    async def test_update_status_valid_transition(self, task_manager):
        """Test valid state transitions."""
        params = TaskSendParams(
            message=Message(role="user", parts=[TextPart(text="Test")]),
        )
        task = await task_manager.create_task(params, agent_name="test-agent")

        # SUBMITTED -> WORKING
        updated = await task_manager.update_status(
            task.id,
            TaskState.WORKING,
            agent_name="test-agent",
        )
        assert updated.status.state == TaskState.WORKING

        # WORKING -> COMPLETED
        updated = await task_manager.update_status(
            task.id,
            TaskState.COMPLETED,
            message=Message(role="agent", parts=[TextPart(text="Done")]),
            agent_name="test-agent",
        )
        assert updated.status.state == TaskState.COMPLETED

    @pytest.mark.asyncio
    async def test_update_status_invalid_transition(self, task_manager):
        """Test invalid state transitions raise error."""
        params = TaskSendParams(
            message=Message(role="user", parts=[TextPart(text="Test")]),
        )
        task = await task_manager.create_task(params, agent_name="test-agent")

        # SUBMITTED -> COMPLETED (invalid, must go through WORKING)
        with pytest.raises(ValueError, match="Invalid state transition"):
            await task_manager.update_status(
                task.id,
                TaskState.COMPLETED,
                agent_name="test-agent",
            )

    @pytest.mark.asyncio
    async def test_complete_task(self, task_manager):
        """Test completing a task with response."""
        params = TaskSendParams(
            message=Message(role="user", parts=[TextPart(text="What is 2+2?")]),
        )
        task = await task_manager.create_task(params, agent_name="test-agent")

        # Move to working first
        await task_manager.update_status(task.id, TaskState.WORKING, agent_name="test-agent")

        # Complete
        completed = await task_manager.complete_task(
            task.id,
            response="The answer is 4.",
            agent_name="test-agent",
        )

        assert completed.status.state == TaskState.COMPLETED
        assert len(completed.history) == 2  # User message + agent response

    @pytest.mark.asyncio
    async def test_fail_task(self, task_manager):
        """Test marking a task as failed."""
        params = TaskSendParams(
            message=Message(role="user", parts=[TextPart(text="Test")]),
        )
        task = await task_manager.create_task(params, agent_name="test-agent")

        # Move to working
        await task_manager.update_status(task.id, TaskState.WORKING, agent_name="test-agent")

        # Fail
        failed = await task_manager.fail_task(
            task.id,
            error="Something went wrong",
            agent_name="test-agent",
        )

        assert failed.status.state == TaskState.FAILED

    @pytest.mark.asyncio
    async def test_cancel_task(self, task_manager):
        """Test canceling a task."""
        params = TaskSendParams(
            message=Message(role="user", parts=[TextPart(text="Test")]),
        )
        task = await task_manager.create_task(params, agent_name="test-agent")

        canceled = await task_manager.cancel_task(
            task.id,
            reason="User requested cancellation",
            agent_name="test-agent",
        )

        assert canceled.status.state == TaskState.CANCELED

    @pytest.mark.asyncio
    async def test_add_artifact(self, task_manager):
        """Test adding artifacts to a task."""
        params = TaskSendParams(
            message=Message(role="user", parts=[TextPart(text="Generate report")]),
        )
        task = await task_manager.create_task(params, agent_name="test-agent")

        artifact = Artifact(
            name="report.txt",
            description="Generated report",
            parts=[TextPart(text="Report content here")],
        )

        updated = await task_manager.add_artifact(task.id, artifact, agent_name="test-agent")

        assert updated.artifacts is not None
        assert len(updated.artifacts) == 1
        assert updated.artifacts[0].name == "report.txt"

    @pytest.mark.asyncio
    async def test_get_pending_tasks(self, task_manager):
        """Test getting pending tasks."""
        # Create multiple tasks
        for i in range(3):
            params = TaskSendParams(
                message=Message(role="user", parts=[TextPart(text=f"Task {i}")]),
            )
            await task_manager.create_task(params, agent_name="test-agent")

        pending = await task_manager.get_pending_tasks()
        assert len(pending) == 3

    @pytest.mark.asyncio
    async def test_get_session_tasks(self, task_manager):
        """Test getting tasks for a specific session."""
        session_id = "session-test-123"

        for i in range(2):
            params = TaskSendParams(
                sessionId=session_id,
                message=Message(role="user", parts=[TextPart(text=f"Task {i}")]),
            )
            await task_manager.create_task(params, agent_name="test-agent")

        # Create task in different session
        params = TaskSendParams(
            sessionId="other-session",
            message=Message(role="user", parts=[TextPart(text="Other task")]),
        )
        await task_manager.create_task(params, agent_name="test-agent")

        session_tasks = await task_manager.get_session_tasks(session_id)
        assert len(session_tasks) == 2


class TestOnTaskSubmittedCallback:
    """The ``on_task_submitted`` callback fires inside ``create_task``
    AFTER the task is persisted and BEFORE the SSE notify path. This is
    the missing piece behind every "I sent it, did you get it?" thread
    (#645 / Emma↔Meridian): without the callback, a peer-submitted task
    sat SUBMITTED in the store with no autonomous trigger. The agent's
    handler (KestrelAgent._on_task_submitted) bridges the callback into
    the signal/dispatcher system so the cognition loop wakes."""

    @pytest.mark.asyncio
    async def test_callback_fires_on_create_task(self, db_path):
        """The callback receives the task as its single argument."""
        from kestrel_sovereign.a2a.task_manager import create_task_manager

        received = []
        manager = await create_task_manager(db_path)
        manager._on_task_submitted = lambda t: received.append(t)
        track_manager(manager)

        params = TaskSendParams(
            message=Message(role="user", parts=[TextPart(text="hello")]),
            metadata={"sender": "peer-agent"},
        )
        task = await manager.create_task(params, agent_name="test-agent")

        assert len(received) == 1, "callback must fire exactly once"
        assert received[0].id == task.id
        assert received[0].status.state == TaskState.SUBMITTED, (
            "callback fires while task is still SUBMITTED, before any "
            "status transition"
        )

    @pytest.mark.asyncio
    async def test_callback_exception_does_not_break_create_task(self, db_path):
        """A failing on_task_submitted callback must NOT roll back the
        task creation. The task is the source of truth; the callback
        is best-effort (matches `_on_task_complete` posture)."""
        from kestrel_sovereign.a2a.task_manager import create_task_manager

        def boom(_t):
            raise RuntimeError("dispatcher down")

        manager = await create_task_manager(db_path)
        manager._on_task_submitted = boom
        track_manager(manager)

        params = TaskSendParams(
            message=Message(role="user", parts=[TextPart(text="hi")]),
        )
        # MUST NOT raise — task creation completes even when the
        # signal-emit hook fails.
        task = await manager.create_task(params, agent_name="test-agent")
        assert task is not None
        # And the task IS persisted.
        loaded = await manager.task_store.get(task.id)
        assert loaded is not None

    @pytest.mark.asyncio
    async def test_no_callback_set_is_fine(self, db_path):
        """Backward-compat: when no callback is wired (legacy code, test
        fixtures), create_task behaves exactly as before."""
        from kestrel_sovereign.a2a.task_manager import create_task_manager

        manager = await create_task_manager(db_path)
        assert manager._on_task_submitted is None
        track_manager(manager)

        params = TaskSendParams(
            message=Message(role="user", parts=[TextPart(text="hi")]),
        )
        task = await manager.create_task(params, agent_name="test-agent")
        assert task is not None


class TestA2ATaskSubmittedSignalSource:
    """The signal-source registration must declare the right semantics
    so the dispatcher routes correctly. Mirrors the assertions on the
    `a2a.task_complete` registration."""

    def test_registration_shape(self):
        from kestrel_sovereign.signals.sources.a2a_task_submitted import (
            SOURCE_NAME,
            build_a2a_task_submitted_registration,
        )
        from kestrel_sdk.signals import SignalMode, Trust

        reg = build_a2a_task_submitted_registration()
        assert reg.name == SOURCE_NAME == "a2a.task_submitted"
        assert reg.default_mode == SignalMode.COGNITION
        assert reg.trust == Trust.TRUSTED
        assert reg.allow_self_loops is False, (
            "A→B→A would loop without bound; cycle detection requires "
            "self-loop block at the source registration"
        )

    def test_signal_builder_payload(self):
        from kestrel_sovereign.signals.sources.a2a_task_submitted import (
            SOURCE_NAME,
            build_signal_for_submitted_task,
        )

        # Duck-type a Task (the helper doesn't require a real pydantic
        # object, matching `build_signal_for_completed_task`).
        class _FakeTask:
            id = "task-123"
            sessionId = "sess-abc"
            metadata = {"skill": "workflow.assign", "sender": "emma"}

        sig = build_signal_for_submitted_task(
            _FakeTask(), target_agent="meridian-did", sender="emma",
        )
        assert sig.source == SOURCE_NAME
        assert sig.target_agent == "meridian-did"
        assert sig.payload["task_id"] == "task-123"
        assert sig.payload["sender"] == "emma"
        assert sig.payload["skill_id"] == "workflow.assign"
        # Dedupe by task_id so idempotency retries collapse to one wake.
        assert sig.dedupe_key == "task-123"

    def test_schema_rejects_missing_keys(self):
        from kestrel_sovereign.signals.sources.a2a_task_submitted import (
            build_a2a_task_submitted_registration,
        )
        reg = build_a2a_task_submitted_registration()
        # Missing all required keys.
        with pytest.raises(ValueError, match="missing required key"):
            reg.schema({})

    def test_signal_surfaces_a2a_verb_and_reply_expected(self):
        """Codex P2 on PR #1380: receiver-side verb discrimination
        must not depend on inferring from skill_id alone. The signal
        payload must carry the sender's stated verb so the cognition
        prompt can frame the response appropriately."""
        from kestrel_sovereign.signals.sources.a2a_task_submitted import (
            build_signal_for_submitted_task,
        )

        class _FakeTaskQuestion:
            id = "task-q"
            sessionId = "sess-q"
            metadata = {
                "sender": "emma",
                "a2a_verb": "question",
                "reply_expected": True,
            }

        sig = build_signal_for_submitted_task(
            _FakeTaskQuestion(), target_agent="meridian-did", sender="emma",
        )
        assert sig.payload["a2a_verb"] == "question"
        assert sig.payload["reply_expected"] is True

        # And the schema validator accepts the enriched payload.
        from kestrel_sovereign.signals.sources.a2a_task_submitted import (
            build_a2a_task_submitted_registration,
        )
        reg = build_a2a_task_submitted_registration()
        reg.schema(sig.payload)  # must not raise

    def test_signal_surfaces_request_content_from_history(self):
        """#1433: the cognition prompt needs the sender's actual question
        text inline so the receiver doesn't have to call check_task_status
        first. Extracted from ``task.history[0].parts[0].text``."""
        from kestrel_sovereign.signals.sources.a2a_task_submitted import (
            build_a2a_task_submitted_registration,
            build_signal_for_submitted_task,
        )

        class _FakePart:
            def __init__(self, text):
                self.text = text

        class _FakeMessage:
            def __init__(self, text):
                self.parts = [_FakePart(text)]

        class _FakeTask:
            id = "task-1433"
            sessionId = "sess-1433"
            metadata = {"sender": "emma", "a2a_verb": "question", "reply_expected": True}
            history = [_FakeMessage("What is 2+2?")]

        sig = build_signal_for_submitted_task(
            _FakeTask(), target_agent="meridian-did", sender="emma",
        )
        assert sig.payload["request_content"] == "What is 2+2?", (
            "The signal-source must extract the sender's question text "
            "from history[0] and put it in payload['request_content'] so "
            "the prompt template can render it inline. Without this, the "
            "receiver's cognition turn only sees the task envelope and "
            "hallucinates 'null body' — see #1433."
        )
        # Schema accepts the new field.
        reg = build_a2a_task_submitted_registration()
        reg.schema(sig.payload)

    def test_schema_injects_default_request_content_for_legacy_payloads(self):
        """Codex review #1433 P2: any legacy caller that builds a signal
        payload without ``request_content`` (test fixtures, external
        integrations, prior code paths) must NOT KeyError at template
        render time. The schema injects an empty-string default so the
        prompt's ``{payload[request_content]}`` placeholder always
        resolves and the cognition turn fires."""
        from kestrel_sovereign.signals.sources.a2a_task_submitted import (
            build_a2a_task_submitted_registration,
        )
        reg = build_a2a_task_submitted_registration()
        legacy_payload = {
            "task_id": "task-legacy",
            "session_id": "sess-legacy",
            "sender": "emma",
        }
        out = reg.schema(legacy_payload)
        assert out["request_content"] == "", (
            "Schema must inject an empty-string default for legacy "
            "payloads so the prompt template's {payload[request_content]} "
            "placeholder always resolves. Otherwise cognition turns "
            "spawned from these signals fail silently with KeyError."
        )
        # The schema must inject defaults for EVERY field the prompt
        # template indexes — not just request_content. The first iteration
        # of this test manually padded `a2a_verb`/`skill_id`/`reply_expected`
        # which masked codex round 2 P2: legacy payloads still KeyError'd
        # at render time. Use ONLY the schema's output here so the prompt
        # template's full set of placeholders is exercised.
        assert out["a2a_verb"] == ""
        assert out["skill_id"] == ""
        assert out["reply_expected"] is False

        from kestrel_sovereign.signals.sources.a2a_task_submitted import (
            PROMPT_TEMPLATE,
        )
        template_text = PROMPT_TEMPLATE.read_text()
        template_text.format(
            source="a2a.task_submitted",
            target_agent="meridian-did",
            arrived_at="2026-05-28T20:00:00Z",
            urgency="normal",
            payload=out,
        )

    def test_signal_request_content_empty_when_history_absent(self):
        """A task with no history (edge case from legacy code paths)
        produces an empty ``request_content`` rather than KeyError-ing.
        The prompt template surface guards visually for empty bodies."""
        from kestrel_sovereign.signals.sources.a2a_task_submitted import (
            build_signal_for_submitted_task,
        )

        class _FakeTask:
            id = "task-empty"
            sessionId = "sess-empty"
            metadata = {"sender": "emma"}
            history = []

        sig = build_signal_for_submitted_task(
            _FakeTask(), target_agent="meridian-did", sender="emma",
        )
        assert sig.payload["request_content"] == ""

    def test_signal_empty_a2a_verb_when_metadata_missing(self):
        """Legacy / non-PeersFeature task creators leave a2a_verb
        empty. Signal still builds; payload has empty strings rather
        than missing keys (so downstream consumers can rely on the
        keys being present)."""
        from kestrel_sovereign.signals.sources.a2a_task_submitted import (
            build_signal_for_submitted_task,
        )

        class _FakeLegacyTask:
            id = "task-legacy"
            sessionId = "sess-legacy"
            metadata = {}

        sig = build_signal_for_submitted_task(
            _FakeLegacyTask(), target_agent="x", sender="",
        )
        assert sig.payload["a2a_verb"] == ""
        assert sig.payload["reply_expected"] is False

    def test_signal_rehydrates_causation_chain_from_metadata(self):
        """Codex P1 on PR #1366: without rehydration, A→B→A task-
        submission ping-pong bypasses cycle detection — every inbound
        task starts fresh at depth 1. Inbound signal MUST deserialize
        ``task.metadata["causation_chain"]`` the way the complete-
        direction signal does (`a2a.py:build_signal_for_completed_task`).
        """
        from kestrel_sovereign.signals.sources.a2a_task_submitted import (
            build_signal_for_submitted_task,
        )

        # Construct metadata in the same serialized shape
        # ``serialize_chain_for_metadata`` produces (list of dicts).
        causation_chain = [
            {
                "agent_id": "did:agent:emma",
                "source": "channel.message",
                "signal_id": "sig-1",
                "turn_id": "turn-1",
                "depth": 1,
                "emitted_at": "2026-05-23T10:00:00+00:00",
            },
            {
                "agent_id": "did:agent:meridian",
                "source": "a2a.task_submitted",
                "signal_id": "sig-2",
                "turn_id": "turn-2",
                "depth": 2,
                "emitted_at": "2026-05-23T10:00:05+00:00",
            },
        ]

        class _FakeTask:
            id = "task-789"
            sessionId = "sess-abc"
            metadata = {
                "skill": "workflow.assign",
                "sender": "emma",
                "causation_chain": causation_chain,
            }

        sig = build_signal_for_submitted_task(
            _FakeTask(), target_agent="emma-did", sender="meridian",
        )
        # Chain must be present and have both upstream frames so the
        # dispatcher's cycle detector sees the prior emma→meridian
        # hop and rejects emma→meridian→emma at depth 2.
        assert len(sig.causation_chain) == 2, (
            "rehydrated causation chain must carry every upstream "
            f"frame; got: {sig.causation_chain!r}"
        )
        assert sig.causation_chain[0].agent_id == "did:agent:emma"
        assert sig.causation_chain[1].agent_id == "did:agent:meridian"

    def test_signal_empty_chain_when_metadata_missing(self):
        """When the upstream task was created without dispatcher
        context (e.g. local self-spawn, or pre-causation-chain code),
        the signal carries an empty chain — not a crash."""
        from kestrel_sovereign.signals.sources.a2a_task_submitted import (
            build_signal_for_submitted_task,
        )

        class _FakeTask:
            id = "task-noscope"
            sessionId = "sess"
            metadata = {}  # no causation_chain key

        sig = build_signal_for_submitted_task(
            _FakeTask(), target_agent="x-did", sender="",
        )
        assert sig.causation_chain == [], (
            "empty metadata must yield an empty chain (not None, not "
            "raise) so dispatcher signal validation passes"
        )


# =============================================================================
# TaskWorker Tests
# =============================================================================

class TestTaskWorker:
    """Tests for TaskWorker and handlers."""

    @pytest.mark.asyncio
    async def test_simple_task_handler(self, task_manager):
        """Test SimpleTaskHandler."""

        def handle_task(task: Task) -> TaskResult:
            return TaskResult(
                success=True,
                response="Handled successfully",
            )

        handler = SimpleTaskHandler(handler_fn=handle_task)

        task = Task(
            id="test-task",
            status=TaskStatus(state=TaskState.WORKING),
            history=[Message(role="user", parts=[TextPart(text="Test")])],
        )

        result = await handler.handle(task)
        assert result.success is True
        assert result.response == "Handled successfully"

    @pytest.mark.asyncio
    async def test_simple_handler_with_filter(self, task_manager):
        """Test SimpleTaskHandler with can_handle filter."""

        def handle_task(task: Task) -> TaskResult:
            return TaskResult(success=True, response="Done")

        def can_handle(task: Task) -> bool:
            # Only handle tasks with "process" in metadata
            return task.metadata is not None and "process" in str(task.metadata)

        handler = SimpleTaskHandler(handler_fn=handle_task, can_handle_fn=can_handle)

        task_yes = Task(
            id="task-yes",
            status=TaskStatus(state=TaskState.WORKING),
            metadata={"type": "process"},
        )

        task_no = Task(
            id="task-no",
            status=TaskStatus(state=TaskState.WORKING),
            metadata={"type": "other"},
        )

        assert handler.can_handle(task_yes) is True
        assert handler.can_handle(task_no) is False

    @pytest.mark.asyncio
    async def test_llm_task_handler(self, task_manager):
        """Test LLMTaskHandler."""

        async def mock_llm(prompt: str, history: list[Message]) -> str:
            return f"Response to: {prompt}"

        handler = LLMTaskHandler(llm_fn=mock_llm)

        task = Task(
            id="llm-task",
            status=TaskStatus(state=TaskState.WORKING),
            history=[Message(role="user", parts=[TextPart(text="Hello")])],
        )

        assert handler.can_handle(task) is True

        result = await handler.handle(task)
        assert result.success is True
        assert "Response to: Hello" in result.response

    @pytest.mark.asyncio
    async def test_task_worker_register_handler(self, task_manager):
        """Test registering handlers with TaskWorker."""
        worker = TaskWorker(
            task_manager=task_manager,
            agent_name="test-worker",
        )

        handler = SimpleTaskHandler(
            handler_fn=lambda t: TaskResult(success=True, response="Done")
        )

        worker.register_handler(handler)
        assert len(worker._handlers) == 1

    @pytest.mark.asyncio
    async def test_task_result_needs_input(self):
        """Test TaskResult with needs_input flag."""
        result = TaskResult(
            success=False,
            needs_input=True,
            input_prompt="Please provide more details.",
        )

        assert result.needs_input is True
        assert result.input_prompt == "Please provide more details."

    @pytest.mark.asyncio
    async def test_task_result_with_artifacts(self):
        """Test TaskResult with artifacts."""
        artifact = Artifact(
            name="output.json",
            parts=[TextPart(text='{"result": "data"}')],
        )

        result = TaskResult(
            success=True,
            response="Generated output",
            artifacts=[artifact],
        )

        assert result.artifacts is not None
        assert len(result.artifacts) == 1
        assert result.artifacts[0].name == "output.json"


# =============================================================================
# Integration Tests (TaskManager + TaskWorker)
# =============================================================================

class TestTaskManagerWorkerIntegration:
    """Integration tests for TaskManager and TaskWorker working together."""

    @pytest.mark.asyncio
    async def test_full_task_workflow(self, task_manager):
        """Test complete task workflow: create -> process -> complete."""
        # Create task
        params = TaskSendParams(
            message=Message(role="user", parts=[TextPart(text="Process this data")]),
        )
        task = await task_manager.create_task(params, agent_name="test-agent")
        assert task.status.state == TaskState.SUBMITTED

        # Worker picks up task and starts processing
        await task_manager.update_status(task.id, TaskState.WORKING, agent_name="test-agent")
        working_task = await task_manager.get_task(task.id)
        assert working_task.status.state == TaskState.WORKING

        # Worker completes task
        completed_task = await task_manager.complete_task(
            task.id,
            response="Data processed successfully",
            agent_name="test-agent",
        )
        assert completed_task.status.state == TaskState.COMPLETED

        # Verify final state
        final_task = await task_manager.get_task(task.id)
        assert final_task.status.state == TaskState.COMPLETED
        assert len(final_task.history) == 2

    @pytest.mark.asyncio
    async def test_task_input_required_flow(self, task_manager):
        """Test task flow with INPUT_REQUIRED state."""
        params = TaskSendParams(
            message=Message(role="user", parts=[TextPart(text="Do something vague")]),
        )
        task = await task_manager.create_task(params, agent_name="test-agent")

        # Worker needs more input
        await task_manager.update_status(task.id, TaskState.WORKING, agent_name="test-agent")
        await task_manager.update_status(
            task.id,
            TaskState.INPUT_REQUIRED,
            message=Message(role="agent", parts=[TextPart(text="What specifically?")]),
            agent_name="test-agent",
        )

        task = await task_manager.get_task(task.id)
        assert task.status.state == TaskState.INPUT_REQUIRED

        # User provides input (simulated by resuming)
        await task_manager.update_status(task.id, TaskState.WORKING, agent_name="test-agent")

        # Now complete
        await task_manager.complete_task(task.id, response="Done", agent_name="test-agent")

        final_task = await task_manager.get_task(task.id)
        assert final_task.status.state == TaskState.COMPLETED


# =============================================================================
# Subscribe — late-subscriber close (#1444 codex round 1 P2)
# =============================================================================

class TestSubscribeLateTerminal:
    """A subscriber that connects AFTER the task is already terminal
    must receive the snapshot frame and immediately see the stream
    close — without this fix, the connection stayed alive emitting
    keepalives until the client timed out (codex round 1 P2 on PR
    #1453)."""

    @pytest.mark.asyncio
    async def test_already_completed_task_closes_after_snapshot(self):
        task = Task(
            id="t-late-completed",
            sessionId="sess",
            status=TaskStatus(
                state=TaskState.COMPLETED,
                message=Message(
                    role="agent",
                    parts=[TextPart(type="text", text="done")],
                ),
            ),
        )
        store = MagicMock()
        store.get = AsyncMock(return_value=task)
        store.close = AsyncMock()
        session_service = MagicMock()
        session_service.close = AsyncMock()
        observability_store = MagicMock()
        observability_store.close = AsyncMock()
        manager = track_manager(TaskManager(
            task_store=store,
            session_service=session_service,
            observability_store=observability_store,
        ))

        # asyncio.wait_for around the iteration so a regression where
        # the loop stays open emitting keepalives fails fast instead
        # of hanging the test forever.
        frames = []
        async def _collect():
            async for ev in manager.subscribe("t-late-completed"):
                frames.append(ev)
        await asyncio.wait_for(_collect(), timeout=2.0)

        assert len(frames) == 1, (
            f"Late subscriber to an already-terminal task must receive "
            f"the snapshot frame and then have the stream close — got "
            f"{len(frames)} frame(s), which means the keepalive loop "
            f"was entered. {[(f.get('event'), f.get('final')) for f in frames]}"
        )
        assert frames[0]["event"] == "status"
        assert frames[0]["final"] is True, (
            "Snapshot frame for an already-terminal task must carry "
            "top-level ``final=True`` so SSE bridges (e.g. the "
            "endpoints/agent.py subscribe handler) close the stream "
            "cleanly without parsing JSON data."
        )

    @pytest.mark.asyncio
    async def test_already_failed_task_closes_after_snapshot(self):
        """FAILED is also terminal — same close behavior as COMPLETED."""
        task = Task(
            id="t-late-failed",
            sessionId="sess",
            status=TaskStatus(state=TaskState.FAILED),
        )
        store = MagicMock()
        store.get = AsyncMock(return_value=task)
        store.close = AsyncMock()
        session_service = MagicMock()
        session_service.close = AsyncMock()
        observability_store = MagicMock()
        observability_store.close = AsyncMock()
        manager = track_manager(TaskManager(
            task_store=store,
            session_service=session_service,
            observability_store=observability_store,
        ))

        frames = []
        async def _collect():
            async for ev in manager.subscribe("t-late-failed"):
                frames.append(ev)
        await asyncio.wait_for(_collect(), timeout=2.0)

        assert len(frames) == 1
        assert frames[0]["final"] is True
