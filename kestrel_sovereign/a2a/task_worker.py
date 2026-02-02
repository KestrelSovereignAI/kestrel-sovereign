"""
TaskWorker - Background Task Processing for A2A Protocol.

Provides background processing capabilities:
- Poll for pending tasks
- Execute task handlers
- Manage concurrent task processing
- Handle retries and failures

The worker pattern allows stateless HTTP endpoints while
tasks are processed asynchronously.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Optional

from kestrel_sovereign.a2a.types import Task, TaskState, Message, TextPart, Artifact
from kestrel_sovereign.a2a.task_manager import TaskManager

logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    """Result of task execution."""
    success: bool
    response: Optional[str] = None
    error: Optional[str] = None
    artifacts: Optional[list[Artifact]] = None
    needs_input: bool = False
    input_prompt: Optional[str] = None


class TaskHandler(ABC):
    """
    Abstract base class for task handlers.

    Implement this class to define how tasks are processed.
    """

    @abstractmethod
    async def handle(self, task: Task) -> TaskResult:
        """
        Process a task and return the result.

        Args:
            task: The task to process

        Returns:
            TaskResult indicating success/failure and response
        """
        pass

    @abstractmethod
    def can_handle(self, task: Task) -> bool:
        """
        Check if this handler can process the given task.

        Args:
            task: The task to check

        Returns:
            True if this handler can process the task
        """
        pass


class TaskWorker:
    """
    Background worker for processing tasks.

    Polls the TaskManager for pending tasks and dispatches them
    to registered handlers.
    """

    def __init__(
        self,
        task_manager: TaskManager,
        agent_name: str,
        poll_interval: float = 1.0,
        max_concurrent: int = 5,
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ):
        """
        Initialize the task worker.

        Args:
            task_manager: TaskManager instance for task operations
            agent_name: Name of the agent running this worker
            poll_interval: Seconds between polling for new tasks
            max_concurrent: Maximum concurrent task processing
            max_retries: Maximum retry attempts for failed tasks
            retry_delay: Seconds to wait before retrying
        """
        self.task_manager = task_manager
        self.agent_name = agent_name
        self.poll_interval = poll_interval
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self._handlers: list[TaskHandler] = []
        self._running = False
        self._tasks: set[asyncio.Task] = set()
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._retry_counts: dict[str, int] = {}

    def register_handler(self, handler: TaskHandler) -> None:
        """Register a task handler."""
        self._handlers.append(handler)
        logger.info(f"Registered handler: {handler.__class__.__name__}")

    async def start(self) -> None:
        """Start the worker loop."""
        if self._running:
            logger.warning("Worker already running")
            return

        self._running = True
        self._semaphore = asyncio.Semaphore(self.max_concurrent)

        logger.info(
            f"Starting TaskWorker for {self.agent_name} "
            f"(poll_interval={self.poll_interval}s, max_concurrent={self.max_concurrent})"
        )

        while self._running:
            try:
                await self._poll_and_process()
            except Exception as e:
                logger.error(f"Error in worker loop: {e}", exc_info=True)

            await asyncio.sleep(self.poll_interval)

    async def stop(self, timeout: float = 30.0) -> None:
        """
        Stop the worker gracefully.

        Args:
            timeout: Maximum seconds to wait for in-progress tasks
        """
        self._running = False

        if self._tasks:
            logger.info(f"Waiting for {len(self._tasks)} tasks to complete...")
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._tasks, return_exceptions=True),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                logger.warning(f"Timeout waiting for tasks, canceling {len(self._tasks)} tasks")
                for task in self._tasks:
                    task.cancel()

        logger.info("TaskWorker stopped")

    async def _poll_and_process(self) -> None:
        """Poll for pending tasks and dispatch them."""
        # Get pending tasks
        pending = await self.task_manager.get_pending_tasks(limit=self.max_concurrent)

        for task in pending:
            # Find handler
            handler = self._find_handler(task)
            if not handler:
                logger.warning(f"No handler found for task {task.id}")
                continue

            # Check if we can start a new task
            if len(self._tasks) >= self.max_concurrent:
                break

            # Start processing
            asyncio_task = asyncio.create_task(
                self._process_task(task, handler)
            )
            self._tasks.add(asyncio_task)
            asyncio_task.add_done_callback(self._tasks.discard)

    def _find_handler(self, task: Task) -> Optional[TaskHandler]:
        """Find a handler that can process the task."""
        for handler in self._handlers:
            if handler.can_handle(task):
                return handler
        return None

    async def _process_task(self, task: Task, handler: TaskHandler) -> None:
        """Process a single task with the given handler."""
        async with self._semaphore:
            task_id = task.id
            start_time = time.time()

            try:
                # Mark as working
                await self.task_manager.update_status(
                    task_id=task_id,
                    new_state=TaskState.WORKING,
                    agent_name=self.agent_name,
                )

                # Execute handler
                result = await handler.handle(task)
                duration_ms = int((time.time() - start_time) * 1000)

                # Log to observability
                await self.task_manager.observability_store.log_tool_response(
                    event_id=f"{task_id}_handler",
                    success=result.success,
                    duration_ms=duration_ms,
                    error_message=result.error,
                )

                # Handle result
                if result.needs_input:
                    # Request input from user
                    message = Message(
                        role="agent",
                        parts=[TextPart(text=result.input_prompt or "Please provide more information.")]
                    )
                    await self.task_manager.update_status(
                        task_id=task_id,
                        new_state=TaskState.INPUT_REQUIRED,
                        message=message,
                        agent_name=self.agent_name,
                    )
                elif result.success:
                    # Complete successfully
                    await self.task_manager.complete_task(
                        task_id=task_id,
                        response=result.response or "Task completed.",
                        agent_name=self.agent_name,
                        artifacts=result.artifacts,
                    )
                    # Clear retry count on success
                    self._retry_counts.pop(task_id, None)
                else:
                    # Handle failure with retry
                    await self._handle_failure(task, result.error or "Unknown error")

            except Exception as e:
                logger.error(f"Exception processing task {task_id}: {e}", exc_info=True)
                await self._handle_failure(task, str(e))

    async def _handle_failure(self, task: Task, error: str) -> None:
        """Handle task failure with retry logic."""
        task_id = task.id
        retry_count = self._retry_counts.get(task_id, 0)

        if retry_count < self.max_retries:
            # Schedule retry
            self._retry_counts[task_id] = retry_count + 1
            logger.warning(
                f"Task {task_id} failed (attempt {retry_count + 1}/{self.max_retries}): {error}. "
                f"Retrying in {self.retry_delay}s..."
            )

            # Reset to submitted for retry (will be picked up on next poll)
            await self.task_manager.update_status(
                task_id=task_id,
                new_state=TaskState.SUBMITTED,
                agent_name=self.agent_name,
            )

            await asyncio.sleep(self.retry_delay)
        else:
            # Max retries exceeded, mark as failed
            logger.error(f"Task {task_id} failed after {self.max_retries} attempts: {error}")
            await self.task_manager.fail_task(
                task_id=task_id,
                error=f"Failed after {self.max_retries} attempts: {error}",
                agent_name=self.agent_name,
            )
            self._retry_counts.pop(task_id, None)


class SimpleTaskHandler(TaskHandler):
    """
    Simple task handler that uses a callback function.

    Convenient for quick handler implementations.
    """

    def __init__(
        self,
        handler_fn: Callable[[Task], TaskResult],
        can_handle_fn: Optional[Callable[[Task], bool]] = None,
    ):
        """
        Initialize with handler functions.

        Args:
            handler_fn: Function to process tasks (can be async or sync)
            can_handle_fn: Optional function to check if handler applies
        """
        self._handler_fn = handler_fn
        self._can_handle_fn = can_handle_fn or (lambda t: True)

    async def handle(self, task: Task) -> TaskResult:
        """Execute the handler function."""
        result = self._handler_fn(task)
        if asyncio.iscoroutine(result):
            return await result
        return result

    def can_handle(self, task: Task) -> bool:
        """Check using the can_handle function."""
        return self._can_handle_fn(task)


class LLMTaskHandler(TaskHandler):
    """
    Task handler that processes tasks using an LLM.

    This is the default handler for conversation-based tasks.
    """

    def __init__(
        self,
        llm_fn: Callable[[str, list[Message]], str],
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize with LLM function.

        Args:
            llm_fn: Async function that takes (prompt, history) and returns response
            system_prompt: Optional system prompt to prepend
        """
        self._llm_fn = llm_fn
        self._system_prompt = system_prompt

    async def handle(self, task: Task) -> TaskResult:
        """Process task using LLM."""
        try:
            # Extract last user message
            if not task.history:
                return TaskResult(success=False, error="No message history")

            last_message = task.history[-1]
            if last_message.role != "user":
                return TaskResult(success=False, error="Last message is not from user")

            # Get prompt text
            prompt_parts = []
            for part in last_message.parts:
                if hasattr(part, 'text'):
                    prompt_parts.append(part.text)
            prompt = "\n".join(prompt_parts)

            # Call LLM
            response = self._llm_fn(prompt, task.history)
            if asyncio.iscoroutine(response):
                response = await response

            return TaskResult(success=True, response=response)

        except Exception as e:
            return TaskResult(success=False, error=str(e))

    def can_handle(self, task: Task) -> bool:
        """Can handle any task with message history."""
        return task.history is not None and len(task.history) > 0


# Factory function
async def create_task_worker(
    task_manager: TaskManager,
    agent_name: str,
    handlers: Optional[list[TaskHandler]] = None,
    **kwargs,
) -> TaskWorker:
    """
    Create and configure a TaskWorker.

    Args:
        task_manager: TaskManager instance
        agent_name: Name of the agent
        handlers: Optional list of handlers to register
        **kwargs: Additional TaskWorker arguments

    Returns:
        Configured TaskWorker (not started)
    """
    worker = TaskWorker(
        task_manager=task_manager,
        agent_name=agent_name,
        **kwargs,
    )

    if handlers:
        for handler in handlers:
            worker.register_handler(handler)

    return worker
