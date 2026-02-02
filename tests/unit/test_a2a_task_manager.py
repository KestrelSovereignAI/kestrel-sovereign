"""
Unit tests for A2A TaskManager and TaskWorker.

Tests task lifecycle management and background processing.
"""

import asyncio
import os
import tempfile

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
    os.unlink(path)


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
