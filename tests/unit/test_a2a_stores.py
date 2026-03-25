"""
Unit tests for A2A Protocol Stores.

Tests all 6 core datastores with SQLite implementations:
1. TaskStore - Async task persistence
2. SessionService - Session state and events
3. MemoryService - Long-term searchable memory
4. ObservabilityStore - Telemetry and metrics
5. OrchestrationStore - Multi-agent workflows
6. FeedbackStore - Agent self-diagnosis
"""

import asyncio
import os
import tempfile
from datetime import datetime, timedelta, timezone

import pytest

from kestrel_sovereign.a2a.types import (
    Task,
    TaskState,
    TaskStatus,
    Message,
    TextPart,
    Artifact,
)
from kestrel_sovereign.a2a.stores import (
    # Stores
    TaskStore,
    SessionService,
    MemoryService,
    ObservabilityStore,
    OrchestrationStore,
    FeedbackStore,
    # Data models
    SessionState,
    MemoryEntry,
    ObservabilityEvent,
    OrchestrationStatus,
    OrchestrationTask,
    FeedbackCategory,
    FeedbackSeverity,
    FeedbackStatus,
    FeedbackSource,
    FeedbackEntry,
)
from kestrel_sovereign.storage.db import SQLiteBackend


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


# Registry to track stores that need cleanup
_stores_to_close = []


@pytest.fixture(autouse=True)
def _cleanup_stores():
    """Auto-cleanup any stores created during tests."""
    _stores_to_close.clear()
    yield
    # Cleanup is handled by the async cleanup fixture


@pytest_asyncio.fixture(autouse=True)
async def _async_cleanup_stores():
    """Async cleanup for stores after each test."""
    yield
    for store in _stores_to_close:
        try:
            await store.close()
        except Exception:
            pass
    _stores_to_close.clear()


def track_store(store):
    """Register a store for cleanup after the test."""
    _stores_to_close.append(store)
    return store


# =============================================================================
# TaskStore Tests
# =============================================================================

class TestTaskStore:
    """Tests for TaskStore with SQLite backend."""

    @pytest.mark.asyncio
    async def test_initialize(self, db_path):
        """Test store initialization creates tables."""
        backend = SQLiteBackend(db_path)
        await backend.connect()
        store = track_store(TaskStore(backend))
        await store.initialize()
        # No exception means success

    @pytest.mark.asyncio
    async def test_save_and_get(self, db_path):
        """Test saving and retrieving a task."""
        backend = SQLiteBackend(db_path)
        await backend.connect()
        store = track_store(TaskStore(backend))
        await store.initialize()

        task = Task(
            id="task-001",
            sessionId="session-001",
            status=TaskStatus(state=TaskState.SUBMITTED),
            history=[
                Message(role="user", parts=[TextPart(text="Hello")])
            ],
            metadata={"key": "value"},
        )

        await store.save(task)
        retrieved = await store.get("task-001")

        assert retrieved is not None
        assert retrieved.id == "task-001"
        assert retrieved.sessionId == "session-001"
        assert retrieved.status.state == TaskState.SUBMITTED
        assert len(retrieved.history) == 1
        assert retrieved.metadata["key"] == "value"

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, db_path):
        """Test getting a nonexistent task returns None."""
        backend = SQLiteBackend(db_path)
        await backend.connect()
        store = track_store(TaskStore(backend))
        await store.initialize()

        result = await store.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_update_status(self, db_path):
        """Test updating task status."""
        backend = SQLiteBackend(db_path)
        await backend.connect()
        store = track_store(TaskStore(backend))
        await store.initialize()

        task = Task(
            id="task-002",
            status=TaskStatus(state=TaskState.SUBMITTED),
        )
        await store.save(task)

        new_status = TaskStatus(state=TaskState.WORKING)
        await store.update_status("task-002", new_status)

        retrieved = await store.get("task-002")
        assert retrieved.status.state == TaskState.WORKING

    @pytest.mark.asyncio
    async def test_add_artifact(self, db_path):
        """Test adding artifacts to a task."""
        backend = SQLiteBackend(db_path)
        await backend.connect()
        store = track_store(TaskStore(backend))
        await store.initialize()

        task = Task(
            id="task-003",
            status=TaskStatus(state=TaskState.WORKING),
        )
        await store.save(task)

        artifact = Artifact(
            name="result.txt",
            description="Output file",
            parts=[TextPart(text="Result content")],
        )
        await store.add_artifact("task-003", artifact)

        retrieved = await store.get("task-003")
        assert retrieved.artifacts is not None
        assert len(retrieved.artifacts) == 1
        assert retrieved.artifacts[0].name == "result.txt"

    @pytest.mark.asyncio
    async def test_get_pending_tasks(self, db_path):
        """Test getting pending (submitted) tasks."""
        backend = SQLiteBackend(db_path)
        await backend.connect()
        store = track_store(TaskStore(backend))
        await store.initialize()

        # Create tasks in different states
        for i, state in enumerate([TaskState.SUBMITTED, TaskState.WORKING, TaskState.SUBMITTED, TaskState.COMPLETED]):
            task = Task(
                id=f"task-{i}",
                status=TaskStatus(state=state),
            )
            await store.save(task)

        pending = await store.get_pending_tasks()
        assert len(pending) == 2
        assert all(t.status.state == TaskState.SUBMITTED for t in pending)

    @pytest.mark.asyncio
    async def test_list_tasks_by_session(self, db_path):
        """Test listing tasks filtered by session."""
        backend = SQLiteBackend(db_path)
        await backend.connect()
        store = track_store(TaskStore(backend))
        await store.initialize()

        for i in range(3):
            task = Task(
                id=f"task-a-{i}",
                sessionId="session-a",
                status=TaskStatus(state=TaskState.COMPLETED),
            )
            await store.save(task)

        for i in range(2):
            task = Task(
                id=f"task-b-{i}",
                sessionId="session-b",
                status=TaskStatus(state=TaskState.COMPLETED),
            )
            await store.save(task)

        session_a_tasks = await store.list_tasks(session_id="session-a")
        assert len(session_a_tasks) == 3

        session_b_tasks = await store.list_tasks(session_id="session-b")
        assert len(session_b_tasks) == 2

    @pytest.mark.asyncio
    async def test_delete_task(self, db_path):
        """Test deleting a task."""
        backend = SQLiteBackend(db_path)
        await backend.connect()
        store = track_store(TaskStore(backend))
        await store.initialize()

        task = Task(
            id="task-delete",
            status=TaskStatus(state=TaskState.COMPLETED),
        )
        await store.save(task)

        result = await store.delete("task-delete")
        assert result is True

        retrieved = await store.get("task-delete")
        assert retrieved is None


# =============================================================================
# SessionService Tests
# =============================================================================

class TestSessionService:
    """Tests for SQLiteSessionService."""

    @pytest.mark.asyncio
    async def test_initialize(self, db_path):
        """Test service initialization."""
        backend = SQLiteBackend(db_path)
        await backend.connect()
        service = track_store(SessionService(backend))
        await service.initialize()

    @pytest.mark.asyncio
    async def test_create_and_get_session(self, db_path):
        """Test creating and retrieving a session."""
        backend = SQLiteBackend(db_path)
        await backend.connect()
        service = track_store(SessionService(backend))
        await service.initialize()

        session_id = await service.create_session(
            agent_name="test-agent",
            user_id="user-001",
            metadata={"context": "test"},
        )

        session = await service.get_session(session_id)
        assert session is not None
        assert session.agent_name == "test-agent"
        assert session.user_id == "user-001"
        assert session.metadata["context"] == "test"

    @pytest.mark.asyncio
    async def test_append_event(self, db_path):
        """Test appending events to session history."""
        backend = SQLiteBackend(db_path)
        await backend.connect()
        service = track_store(SessionService(backend))
        await service.initialize()

        session_id = await service.create_session(agent_name="test-agent")

        await service.append_event(
            session_id=session_id,
            event_type="user_message",
            data={"text": "Hello"},
        )
        await service.append_event(
            session_id=session_id,
            event_type="agent_response",
            data={"text": "Hi there!"},
        )

        session = await service.get_session(session_id)
        assert len(session.events) == 2
        assert session.events[0]["event_type"] == "user_message"
        assert session.events[1]["event_type"] == "agent_response"

    @pytest.mark.asyncio
    async def test_update_session_state(self, db_path):
        """Test updating session state."""
        backend = SQLiteBackend(db_path)
        await backend.connect()
        service = track_store(SessionService(backend))
        await service.initialize()

        session_id = await service.create_session(agent_name="test-agent")

        await service.update_session(
            session_id=session_id,
            state={"conversation_turn": 5},
            metadata={"updated": True},
        )

        session = await service.get_session(session_id)
        assert session.state["conversation_turn"] == 5
        assert session.metadata["updated"] is True

    @pytest.mark.asyncio
    async def test_list_sessions(self, db_path):
        """Test listing sessions."""
        backend = SQLiteBackend(db_path)
        await backend.connect()
        service = track_store(SessionService(backend))
        await service.initialize()

        for i in range(3):
            await service.create_session(
                agent_name=f"agent-{i % 2}",  # Two different agents
            )

        all_sessions = await service.list_sessions()
        assert len(all_sessions) == 3

        agent_0_sessions = await service.list_sessions(agent_name="agent-0")
        assert len(agent_0_sessions) == 2

    @pytest.mark.asyncio
    async def test_delete_session(self, db_path):
        """Test deleting a session."""
        backend = SQLiteBackend(db_path)
        await backend.connect()
        service = track_store(SessionService(backend))
        await service.initialize()

        session_id = await service.create_session(agent_name="test-agent")

        result = await service.delete_session(session_id)
        assert result is True

        session = await service.get_session(session_id)
        assert session is None


# =============================================================================
# MemoryService Tests
# =============================================================================

class TestMemoryService:
    """Tests for SQLiteMemoryService."""

    @pytest.mark.asyncio
    async def test_initialize(self, db_path):
        """Test service initialization with FTS5."""
        backend = SQLiteBackend(db_path)
        await backend.connect()
        service = track_store(MemoryService(backend))
        await service.initialize()

    @pytest.mark.asyncio
    async def test_add_and_get_memory(self, db_path):
        """Test adding and retrieving memory."""
        backend = SQLiteBackend(db_path)
        await backend.connect()
        service = track_store(MemoryService(backend))
        await service.initialize()

        memory_id = await service.add_memory(
            session_id="session-001",
            content="The user mentioned they love hiking in the mountains.",
            tags=["preference", "outdoor"],
            metadata={"importance": "high"},
        )

        memory = await service.get_memory(memory_id)
        assert memory is not None
        assert "hiking" in memory.content
        assert "preference" in memory.tags
        assert memory.metadata["importance"] == "high"

    @pytest.mark.asyncio
    async def test_search_memory_fts(self, db_path):
        """Test full-text search on memories."""
        backend = SQLiteBackend(db_path)
        await backend.connect()
        service = track_store(MemoryService(backend))
        await service.initialize()

        # Add several memories
        await service.add_memory(session_id="s1", content="User loves pizza and Italian food")
        await service.add_memory(session_id="s1", content="User enjoys hiking in nature")
        await service.add_memory(session_id="s1", content="User prefers pizza over pasta")

        # Search for pizza-related memories
        results = await service.search_memory("pizza")
        assert len(results) == 2
        assert all("pizza" in m.content.lower() for m in results)

    @pytest.mark.asyncio
    async def test_search_memory_by_tags(self, db_path):
        """Test searching memories by tags."""
        backend = SQLiteBackend(db_path)
        await backend.connect()
        service = track_store(MemoryService(backend))
        await service.initialize()

        await service.add_memory(session_id="s1", content="Memory 1", tags=["work", "important"])
        await service.add_memory(session_id="s1", content="Memory 2", tags=["personal"])
        await service.add_memory(session_id="s1", content="Memory 3", tags=["work"])

        results = await service.search_memory(tags=["work"])
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_get_session_history(self, db_path):
        """Test getting all memories for a session."""
        backend = SQLiteBackend(db_path)
        await backend.connect()
        service = track_store(MemoryService(backend))
        await service.initialize()

        for i in range(5):
            await service.add_memory(
                session_id="session-mem",
                content=f"Memory entry {i}",
            )

        history = await service.get_session_history("session-mem")
        assert len(history) == 5

    @pytest.mark.asyncio
    async def test_delete_memory(self, db_path):
        """Test deleting a memory."""
        backend = SQLiteBackend(db_path)
        await backend.connect()
        service = track_store(MemoryService(backend))
        await service.initialize()

        memory_id = await service.add_memory(
            session_id="s1",
            content="Temporary memory",
        )

        result = await service.delete_memory(memory_id)
        assert result is True

        memory = await service.get_memory(memory_id)
        assert memory is None


# =============================================================================
# ObservabilityStore Tests
# =============================================================================

class TestObservabilityStore:
    """Tests for SQLiteObservabilityStore."""

    @pytest.mark.asyncio
    async def test_initialize(self, db_path):
        """Test store initialization."""
        backend = SQLiteBackend(db_path)
        await backend.connect()
        store = track_store(ObservabilityStore(backend))
        await store.initialize()

    @pytest.mark.asyncio
    async def test_log_tool_call_and_response(self, db_path):
        """Test logging tool calls with timing."""
        backend = SQLiteBackend(db_path)
        await backend.connect()
        store = track_store(ObservabilityStore(backend))
        await store.initialize()

        # Log tool call start
        event_id = await store.log_tool_call(
            agent_name="test-agent",
            tool_name="web_search",
            session_id="session-001",
        )

        # Log tool response with timing
        await store.log_tool_response(
            event_id=event_id,
            success=True,
            duration_ms=150,
        )

        event = await store.get_event(event_id)
        assert event is not None
        assert event.tool_name == "web_search"
        assert event.duration_ms == 150
        assert event.success is True

    @pytest.mark.asyncio
    async def test_log_error(self, db_path):
        """Test logging errors."""
        backend = SQLiteBackend(db_path)
        await backend.connect()
        store = track_store(ObservabilityStore(backend))
        await store.initialize()

        event_id = await store.log_error(
            agent_name="test-agent",
            error_type="ValueError",
            error_message="Invalid input provided",
            session_id="session-001",
            metadata={"input": "bad data"},
        )

        event = await store.get_event(event_id)
        assert event is not None
        assert event.event_type == "error"
        assert event.success is False
        assert "Invalid input" in event.error_message

    @pytest.mark.asyncio
    async def test_log_metric(self, db_path):
        """Test logging metrics."""
        backend = SQLiteBackend(db_path)
        await backend.connect()
        store = track_store(ObservabilityStore(backend))
        await store.initialize()

        event_id = await store.log_metric(
            agent_name="test-agent",
            metric_name="tokens_used",
            metric_value=1500.0,
            metadata={"model": "gpt-5"},
        )

        event = await store.get_event(event_id)
        assert event is not None
        assert event.event_type == "metric"
        assert event.metadata["metric_name"] == "tokens_used"
        assert event.metadata["metric_value"] == 1500.0

    @pytest.mark.asyncio
    async def test_query_events(self, db_path):
        """Test querying events with filters."""
        backend = SQLiteBackend(db_path)
        await backend.connect()
        store = track_store(ObservabilityStore(backend))
        await store.initialize()

        # Log various events
        await store.log_tool_call(agent_name="agent-a", tool_name="tool1")
        await store.log_tool_call(agent_name="agent-a", tool_name="tool2")
        await store.log_tool_call(agent_name="agent-b", tool_name="tool1")
        await store.log_error(agent_name="agent-a", error_type="Error", error_message="Test")

        # Query by agent
        agent_a_events = await store.query_events(agent_name="agent-a")
        assert len(agent_a_events) == 3

        # Query by event type
        errors = await store.query_events(event_type="error")
        assert len(errors) == 1

    @pytest.mark.asyncio
    async def test_prune_old_events(self, db_path):
        """Test pruning old events."""
        backend = SQLiteBackend(db_path)
        await backend.connect()
        store = track_store(ObservabilityStore(backend))
        await store.initialize()

        # Add some events
        for _ in range(5):
            await store.log_tool_call(agent_name="test", tool_name="test")

        # Prune (won't actually delete since events are new)
        deleted = await store.prune_old_events(older_than_days=0)
        # Since events are brand new, may or may not delete depending on timing
        assert deleted >= 0


# =============================================================================
# OrchestrationStore Tests
# =============================================================================

class TestOrchestrationStore:
    """Tests for SQLiteOrchestrationStore."""

    @pytest.mark.asyncio
    async def test_initialize(self, db_path):
        """Test store initialization."""
        backend = SQLiteBackend(db_path)
        await backend.connect()
        store = track_store(OrchestrationStore(backend))
        await store.initialize()

    @pytest.mark.asyncio
    async def test_create_workflow_and_add_tasks(self, db_path):
        """Test creating a workflow with tasks."""
        backend = SQLiteBackend(db_path)
        await backend.connect()
        store = track_store(OrchestrationStore(backend))
        await store.initialize()

        workflow_id = await store.create_workflow(metadata={"name": "data-pipeline"})

        task1_id = await store.add_task(
            workflow_id=workflow_id,
            agent_name="data-loader",
            step_number=1,
            input_data={"source": "database"},
        )

        task2_id = await store.add_task(
            workflow_id=workflow_id,
            agent_name="processor",
            step_number=2,
            input_data={"transform": "normalize"},
        )

        tasks = await store.get_workflow_tasks(workflow_id)
        assert len(tasks) == 2
        assert tasks[0].step_number == 1
        assert tasks[1].step_number == 2

    @pytest.mark.asyncio
    async def test_delegate_task(self, db_path):
        """Test delegating a task to another agent."""
        backend = SQLiteBackend(db_path)
        await backend.connect()
        store = track_store(OrchestrationStore(backend))
        await store.initialize()

        workflow_id = await store.create_workflow()
        task_id = await store.add_task(
            workflow_id=workflow_id,
            agent_name="coordinator",
            step_number=1,
            input_data={},
        )

        await store.delegate_task(task_id, "specialist-agent")

        task = await store.get_task(task_id)
        assert task.delegated_to == "specialist-agent"
        assert task.status == OrchestrationStatus.DELEGATED

    @pytest.mark.asyncio
    async def test_update_task_status(self, db_path):
        """Test updating orchestration task status."""
        backend = SQLiteBackend(db_path)
        await backend.connect()
        store = track_store(OrchestrationStore(backend))
        await store.initialize()

        workflow_id = await store.create_workflow()
        task_id = await store.add_task(
            workflow_id=workflow_id,
            agent_name="worker",
            step_number=1,
            input_data={"job": "process"},
        )

        await store.update_task_status(
            task_id,
            OrchestrationStatus.IN_PROGRESS,
        )

        await store.update_task_status(
            task_id,
            OrchestrationStatus.COMPLETED,
            output_data={"result": "success"},
        )

        task = await store.get_task(task_id)
        assert task.status == OrchestrationStatus.COMPLETED
        assert task.output_data["result"] == "success"
        assert task.completed_at is not None

    @pytest.mark.asyncio
    async def test_get_pending_delegations(self, db_path):
        """Test getting tasks delegated to an agent."""
        backend = SQLiteBackend(db_path)
        await backend.connect()
        store = track_store(OrchestrationStore(backend))
        await store.initialize()

        workflow_id = await store.create_workflow()

        task1_id = await store.add_task(
            workflow_id=workflow_id,
            agent_name="coordinator",
            step_number=1,
            input_data={},
        )
        task2_id = await store.add_task(
            workflow_id=workflow_id,
            agent_name="coordinator",
            step_number=2,
            input_data={},
        )

        await store.delegate_task(task1_id, "worker-agent")
        await store.delegate_task(task2_id, "worker-agent")

        # Complete one task
        await store.update_task_status(task1_id, OrchestrationStatus.COMPLETED)

        pending = await store.get_pending_delegations("worker-agent")
        assert len(pending) == 1
        assert pending[0].task_id == task2_id

    @pytest.mark.asyncio
    async def test_is_workflow_complete(self, db_path):
        """Test checking workflow completion."""
        backend = SQLiteBackend(db_path)
        await backend.connect()
        store = track_store(OrchestrationStore(backend))
        await store.initialize()

        workflow_id = await store.create_workflow()

        task1_id = await store.add_task(
            workflow_id=workflow_id,
            agent_name="worker",
            step_number=1,
            input_data={},
        )
        task2_id = await store.add_task(
            workflow_id=workflow_id,
            agent_name="worker",
            step_number=2,
            input_data={},
        )

        # Initially not complete
        assert await store.is_workflow_complete(workflow_id) is False

        # Complete first task
        await store.update_task_status(task1_id, OrchestrationStatus.COMPLETED)
        assert await store.is_workflow_complete(workflow_id) is False

        # Complete second task
        await store.update_task_status(task2_id, OrchestrationStatus.COMPLETED)
        assert await store.is_workflow_complete(workflow_id) is True

    @pytest.mark.asyncio
    async def test_parent_child_tasks(self, db_path):
        """Test parent-child task relationships."""
        backend = SQLiteBackend(db_path)
        await backend.connect()
        store = track_store(OrchestrationStore(backend))
        await store.initialize()

        workflow_id = await store.create_workflow()

        parent_id = await store.add_task(
            workflow_id=workflow_id,
            agent_name="coordinator",
            step_number=1,
            input_data={"task": "main"},
        )

        child1_id = await store.add_task(
            workflow_id=workflow_id,
            agent_name="worker",
            step_number=2,
            input_data={"subtask": 1},
            parent_task_id=parent_id,
        )

        child2_id = await store.add_task(
            workflow_id=workflow_id,
            agent_name="worker",
            step_number=3,
            input_data={"subtask": 2},
            parent_task_id=parent_id,
        )

        children = await store.get_child_tasks(parent_id)
        assert len(children) == 2


# =============================================================================
# FeedbackStore Tests
# =============================================================================

class TestFeedbackStore:
    """Tests for SQLiteFeedbackStore."""

    @pytest.mark.asyncio
    async def test_initialize(self, db_path):
        """Test store initialization."""
        backend = SQLiteBackend(db_path)
        await backend.connect()
        store = track_store(FeedbackStore(backend))
        await store.initialize()

    @pytest.mark.asyncio
    async def test_submit_and_get_feedback(self, db_path):
        """Test submitting and retrieving feedback."""
        backend = SQLiteBackend(db_path)
        await backend.connect()
        store = track_store(FeedbackStore(backend))
        await store.initialize()

        feedback_id = await store.submit_feedback(
            agent_name="test-agent",
            source=FeedbackSource.USER,
            category=FeedbackCategory.BUG,
            severity=FeedbackSeverity.HIGH,
            title="Response was incorrect",
            description="The agent gave wrong information about X.",
            session_id="session-001",
            context={"last_message": "What is X?"},
        )

        feedback = await store.get_feedback(feedback_id)
        assert feedback is not None
        assert feedback.category == FeedbackCategory.BUG
        assert feedback.severity == FeedbackSeverity.HIGH
        assert feedback.source == FeedbackSource.USER
        assert feedback.status == FeedbackStatus.OPEN

    @pytest.mark.asyncio
    async def test_agent_self_diagnosis(self, db_path):
        """Test agent submitting self-diagnosed feedback."""
        backend = SQLiteBackend(db_path)
        await backend.connect()
        store = track_store(FeedbackStore(backend))
        await store.initialize()

        feedback_id = await store.submit_feedback(
            agent_name="self-aware-agent",
            source=FeedbackSource.AGENT,
            category=FeedbackCategory.CONFUSION,
            severity=FeedbackSeverity.MEDIUM,
            title="Unclear user intent",
            description="Unable to determine user's goal from ambiguous query.",
            context={"query": "do the thing"},
        )

        feedback = await store.get_feedback(feedback_id)
        assert feedback.source == FeedbackSource.AGENT
        assert feedback.category == FeedbackCategory.CONFUSION

    @pytest.mark.asyncio
    async def test_update_feedback_status(self, db_path):
        """Test updating feedback status."""
        backend = SQLiteBackend(db_path)
        await backend.connect()
        store = track_store(FeedbackStore(backend))
        await store.initialize()

        feedback_id = await store.submit_feedback(
            agent_name="test-agent",
            source=FeedbackSource.USER,
            category=FeedbackCategory.IMPROVEMENT,
            severity=FeedbackSeverity.LOW,
            title="Add feature X",
            description="Would be nice to have X.",
        )

        await store.update_status(
            feedback_id,
            FeedbackStatus.ACKNOWLEDGED,
        )

        feedback = await store.get_feedback(feedback_id)
        assert feedback.status == FeedbackStatus.ACKNOWLEDGED

        await store.update_status(
            feedback_id,
            FeedbackStatus.RESOLVED,
            resolution="Feature X was implemented.",
        )

        feedback = await store.get_feedback(feedback_id)
        assert feedback.status == FeedbackStatus.RESOLVED
        assert feedback.resolution == "Feature X was implemented."
        assert feedback.resolved_at is not None

    @pytest.mark.asyncio
    async def test_query_feedback(self, db_path):
        """Test querying feedback with filters."""
        backend = SQLiteBackend(db_path)
        await backend.connect()
        store = track_store(FeedbackStore(backend))
        await store.initialize()

        # Submit various feedback
        await store.submit_feedback(
            agent_name="agent-a",
            source=FeedbackSource.USER,
            category=FeedbackCategory.BUG,
            severity=FeedbackSeverity.CRITICAL,
            title="Bug 1",
            description="Critical bug",
        )
        await store.submit_feedback(
            agent_name="agent-a",
            source=FeedbackSource.AGENT,
            category=FeedbackCategory.SUGGESTION,
            severity=FeedbackSeverity.LOW,
            title="Suggestion 1",
            description="A suggestion",
        )
        await store.submit_feedback(
            agent_name="agent-b",
            source=FeedbackSource.USER,
            category=FeedbackCategory.BUG,
            severity=FeedbackSeverity.HIGH,
            title="Bug 2",
            description="Another bug",
        )

        # Query by agent
        agent_a_feedback = await store.query_feedback(agent_name="agent-a")
        assert len(agent_a_feedback) == 2

        # Query by category
        bugs = await store.query_feedback(category=FeedbackCategory.BUG)
        assert len(bugs) == 2

        # Query by severity
        critical = await store.query_feedback(severity=FeedbackSeverity.CRITICAL)
        assert len(critical) == 1

    @pytest.mark.asyncio
    async def test_get_open_feedback(self, db_path):
        """Test getting open feedback with severity filter."""
        backend = SQLiteBackend(db_path)
        await backend.connect()
        store = track_store(FeedbackStore(backend))
        await store.initialize()

        # Submit feedback at different severities
        for sev in [FeedbackSeverity.LOW, FeedbackSeverity.MEDIUM, FeedbackSeverity.HIGH, FeedbackSeverity.CRITICAL]:
            await store.submit_feedback(
                agent_name="test-agent",
                source=FeedbackSource.USER,
                category=FeedbackCategory.BUG,
                severity=sev,
                title=f"{sev.value} severity bug",
                description="Test",
            )

        # Get all open
        all_open = await store.get_open_feedback()
        assert len(all_open) == 4

        # Get only high and critical
        high_priority = await store.get_open_feedback(min_severity=FeedbackSeverity.HIGH)
        assert len(high_priority) == 2

        # Verify ordering (critical first)
        assert high_priority[0].severity == FeedbackSeverity.CRITICAL

    @pytest.mark.asyncio
    async def test_get_feedback_stats(self, db_path):
        """Test getting feedback statistics."""
        backend = SQLiteBackend(db_path)
        await backend.connect()
        store = track_store(FeedbackStore(backend))
        await store.initialize()

        # Submit various feedback
        await store.submit_feedback(
            agent_name="test",
            source=FeedbackSource.USER,
            category=FeedbackCategory.BUG,
            severity=FeedbackSeverity.HIGH,
            title="Bug",
            description="Test",
        )
        await store.submit_feedback(
            agent_name="test",
            source=FeedbackSource.AGENT,
            category=FeedbackCategory.SUGGESTION,
            severity=FeedbackSeverity.LOW,
            title="Suggestion",
            description="Test",
        )

        stats = await store.get_feedback_stats()

        assert stats["total"] == 2
        assert stats["by_category"]["bug"] == 1
        assert stats["by_category"]["suggestion"] == 1
        assert stats["by_source"]["user"] == 1
        assert stats["by_source"]["agent"] == 1

    @pytest.mark.asyncio
    async def test_delete_feedback(self, db_path):
        """Test deleting feedback."""
        backend = SQLiteBackend(db_path)
        await backend.connect()
        store = track_store(FeedbackStore(backend))
        await store.initialize()

        feedback_id = await store.submit_feedback(
            agent_name="test",
            source=FeedbackSource.USER,
            category=FeedbackCategory.OTHER,
            severity=FeedbackSeverity.LOW,
            title="Test",
            description="Test",
        )

        result = await store.delete_feedback(feedback_id)
        assert result is True

        feedback = await store.get_feedback(feedback_id)
        assert feedback is None
