"""
TaskManager - Task Lifecycle Management for A2A Protocol.

Provides high-level task management operations:
- Create/update/query tasks
- State transition validation
- SSE event generation for real-time updates
- Integration with all 6 datastores

This is the primary API for working with A2A tasks.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, AsyncGenerator, Callable, Optional

from kestrel_sovereign.kestrel_config.constants import HTTP_TIMEOUT_DEFAULT

from kestrel_sovereign.a2a.types import (
    Task,
    TaskState,
    TaskStatus,
    TaskSendParams,
    Message,
    TextPart,
    DataPart,
    Artifact,
    TaskStatusUpdateEvent,
    TaskArtifactUpdateEvent,
)
from kestrel_sovereign.a2a.agent_card import AgentCard, AgentSkill
from kestrel_sovereign.a2a.stores import (
    TaskStore,
    SessionService,
    MemoryService,
    ObservabilityStore,
    FeedbackStore,
)
from typing import Protocol, runtime_checkable, TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from kestrel_sovereign.hooks import HooksManager, HookEvent, HookInput, PermissionDecision


@runtime_checkable
class TaskHandler(Protocol):
    """Protocol for A2A task handling."""
    async def handle_task(self, task: Task) -> Task:
        """Handle an A2A task and return the updated task."""
        ...

    def get_skill_for_command(self, command: str) -> Optional[str]:
        """Find the skill that handles a given command prefix."""
        ...

logger = logging.getLogger(__name__)


# Valid state transitions
VALID_TRANSITIONS: dict[TaskState, set[TaskState]] = {
    TaskState.SUBMITTED: {TaskState.WORKING, TaskState.CANCELED, TaskState.FAILED},
    TaskState.WORKING: {TaskState.INPUT_REQUIRED, TaskState.COMPLETED, TaskState.CANCELED, TaskState.FAILED},
    TaskState.INPUT_REQUIRED: {TaskState.WORKING, TaskState.CANCELED, TaskState.FAILED},
    TaskState.COMPLETED: set(),  # Terminal state
    TaskState.CANCELED: set(),  # Terminal state
    TaskState.FAILED: set(),  # Terminal state
    TaskState.UNKNOWN: {TaskState.SUBMITTED},  # Can only become submitted
}


class TaskManager:
    """
    High-level task lifecycle management.

    Coordinates between TaskStore, SessionService, and ObservabilityStore
    to provide a unified API for task operations.
    """

    def __init__(
        self,
        task_store: TaskStore,
        session_service: SessionService,
        observability_store: ObservabilityStore,
        memory_service: Optional[MemoryService] = None,
        feedback_store: Optional[FeedbackStore] = None,
        hooks_manager: Optional["HooksManager"] = None,
        on_task_complete: Optional[Callable[[Task], None]] = None,
    ):
        self.task_store = task_store
        self.session_service = session_service
        self.observability_store = observability_store
        self.memory_service = memory_service
        self.feedback_store = feedback_store
        self.hooks_manager = hooks_manager

        # Callback for task completion notifications (for chat notifications)
        self._on_task_complete = on_task_complete

        # Event subscribers for SSE streaming
        self._subscribers: dict[str, list[asyncio.Queue]] = {}

        # Agent registry: agent_id -> (AgentCard, TaskHandler)
        self._agents: dict[str, tuple[AgentCard, TaskHandler]] = {}

        # Skill -> agent mapping for fast lookup
        self._skill_to_agent: dict[str, str] = {}

        # Command prefix -> (agent_id, skill_id) mapping
        self._command_to_skill: dict[str, tuple[str, str]] = {}

    async def initialize(self) -> None:
        """Initialize all stores."""
        await self.task_store.initialize()
        await self.session_service.initialize()
        await self.observability_store.initialize()
        if self.memory_service:
            await self.memory_service.initialize()
        if self.feedback_store:
            await self.feedback_store.initialize()
        logger.info("TaskManager initialized with all stores")

    async def close(self) -> None:
        """Close all stores and release resources.

        This must be called during shutdown to prevent thread leaks from
        aiosqlite connections.
        """
        # Close all stores in reverse order of initialization
        if self.feedback_store:
            try:
                await self.feedback_store.close()
            except Exception as e:
                logger.debug(f"Error closing feedback_store: {e}")

        if self.memory_service:
            try:
                await self.memory_service.close()
            except Exception as e:
                logger.debug(f"Error closing memory_service: {e}")

        try:
            await self.observability_store.close()
        except Exception as e:
            logger.debug(f"Error closing observability_store: {e}")

        try:
            await self.session_service.close()
        except Exception as e:
            logger.debug(f"Error closing session_service: {e}")

        try:
            await self.task_store.close()
        except Exception as e:
            logger.debug(f"Error closing task_store: {e}")

        logger.info("TaskManager closed all stores")

    # =========================================================================
    # Agent Registry (Feature/A2A Unification)
    # =========================================================================

    def register_agent(
        self,
        agent_card: AgentCard,
        handler: TaskHandler,
        command_prefixes: Optional[dict[str, str]] = None,
    ) -> None:
        """
        Register a Feature/Agent with its AgentCard and TaskHandler.

        Args:
            agent_card: The agent's capability description
            handler: The TaskHandler that processes tasks for this agent
            command_prefixes: Optional mapping of command prefixes to skill IDs
                             (e.g., {"!list-models": "list_models"})
        """
        agent_id = agent_card.name
        self._agents[agent_id] = (agent_card, handler)

        # Index skills for lookup
        for skill in agent_card.skills:
            skill_key = f"{agent_id}.{skill.id}"
            self._skill_to_agent[skill_key] = agent_id

            # Also register just by skill name for convenience
            if skill.id not in self._skill_to_agent:
                self._skill_to_agent[skill.id] = agent_id

        # Register command prefixes if provided
        if command_prefixes:
            for prefix, skill_id in command_prefixes.items():
                self._command_to_skill[prefix] = (agent_id, skill_id)

        logger.info(f"Registered agent: {agent_id} with {len(agent_card.skills)} skills")

    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent."""
        if agent_id not in self._agents:
            return

        agent_card, _ = self._agents[agent_id]

        # Remove skill mappings
        for skill in agent_card.skills:
            skill_key = f"{agent_id}.{skill.id}"
            self._skill_to_agent.pop(skill_key, None)
            if self._skill_to_agent.get(skill.id) == agent_id:
                self._skill_to_agent.pop(skill.id, None)

        # Remove command mappings
        to_remove = [k for k, v in self._command_to_skill.items() if v[0] == agent_id]
        for k in to_remove:
            del self._command_to_skill[k]

        del self._agents[agent_id]
        logger.info(f"Unregistered agent: {agent_id}")

    def get_agent_cards(self) -> list[AgentCard]:
        """Get all registered agent cards."""
        return [card for card, _ in self._agents.values()]

    def get_agent_for_skill(self, skill_id: str) -> Optional[str]:
        """Find which agent handles a skill."""
        return self._skill_to_agent.get(skill_id)

    def get_agent_for_command(self, command: str) -> Optional[tuple[str, str]]:
        """
        Find which agent/skill handles a command.

        Args:
            command: Command string like "!list-models"

        Returns:
            Tuple of (agent_id, skill_id) or None if not found
        """
        # Check exact command prefix matches
        parts = command.strip().split()
        if parts:
            cmd = parts[0].lower()
            if cmd in self._command_to_skill:
                return self._command_to_skill[cmd]

        # Fallback: ask each handler
        for agent_id, (_, handler) in self._agents.items():
            skill_id = handler.get_skill_for_command(command)
            if skill_id:
                return (agent_id, skill_id)

        return None

    async def execute_skill(
        self,
        agent_id: str,
        skill_id: str,
        args: dict[str, Any],
        sync: bool = True,
        session_id: Optional[str] = None,
    ) -> Task:
        """
        Execute a skill on an agent.

        This is the unified entry point for both command execution and
        A2A task routing.

        Args:
            agent_id: ID of the agent to execute
            skill_id: ID of the skill to invoke
            args: Arguments to pass to the skill
            sync: If True, wait for completion and return result
            session_id: Optional session ID for tracking

        Returns:
            The completed (or in-progress) Task
        """
        if agent_id not in self._agents:
            raise ValueError(f"Unknown agent: {agent_id}")

        _, handler = self._agents[agent_id]

        # Create task
        task_id = uuid4().hex
        session_id = session_id or uuid4().hex

        # Execute PRE_TOOL_USE hooks if hooks_manager is configured
        if self.hooks_manager:
            from kestrel_sovereign.hooks import HookEvent, HookInput, PermissionDecision

            hook_input = HookInput(
                session_id=session_id,
                hook_event_name=HookEvent.PRE_TOOL_USE.value,
                tool_name=skill_id,
                tool_input=args,
                feature_name=agent_id,
            )

            hook_output = await self.hooks_manager.execute_hooks(
                HookEvent.PRE_TOOL_USE,
                hook_input
            )

            # Handle hook decision
            if hook_output.permission_decision == PermissionDecision.DENY:
                # Create a denied task
                task = Task(
                    id=task_id,
                    sessionId=session_id,
                    status=TaskStatus(
                        state=TaskState.FAILED,
                        message=Message(
                            role="agent",
                            parts=[TextPart(text=f"Permission denied: {hook_output.permission_reason or 'Blocked by security policy'}")]
                        )
                    ),
                    metadata={"skill": skill_id, "args": args, "agent_id": agent_id, "denied": True},
                    history=[],
                )
                await self.task_store.save(task)
                return task

            # Note: SecurityHook handles ASK internally by blocking until
            # the user responds via the approval queue, so hooks only ever
            # return ALLOW or DENY here.  There is no need for an ASK branch.

            # Update args if hook modified them
            if hook_output.updated_input:
                args = hook_output.updated_input

        task = Task(
            id=task_id,
            sessionId=session_id,
            status=TaskStatus(state=TaskState.SUBMITTED),
            metadata={"skill": skill_id, "args": args, "agent_id": agent_id},
            history=[
                Message(
                    role="user",
                    parts=[TextPart(text=f"Execute {skill_id} on {agent_id}")]
                )
            ],
        )

        # Save initial task state
        await self.task_store.save(task)

        if sync:
            # Execute synchronously
            task = await handler.handle_task(task)
            await self.task_store.save(task)

            # Execute POST_TOOL_USE hooks
            if self.hooks_manager:
                from kestrel_sovereign.hooks import HookEvent, HookInput
                import time

                post_hook_input = HookInput(
                    session_id=session_id,
                    hook_event_name=HookEvent.POST_TOOL_USE.value,
                    tool_name=skill_id,
                    tool_input=args,
                    feature_name=agent_id,
                    tool_response={"task_id": task.id, "state": task.status.state.value},
                )
                # Post hooks run in parallel (non-blocking)
                await self.hooks_manager.execute_hooks_parallel(
                    HookEvent.POST_TOOL_USE,
                    post_hook_input
                )

            return task
        else:
            # Return immediately, execute in background
            asyncio.create_task(self._execute_async(handler, task))
            return task

    async def _execute_async(self, handler: TaskHandler, task: Task) -> None:
        """Execute a task asynchronously and update the store."""
        try:
            task = await handler.handle_task(task)
            await self.task_store.save(task)
            await self._notify_status_update(task, final=True)
        except Exception as e:
            logger.error(f"Async task execution failed: {e}")
            task.status = TaskStatus(
                state=TaskState.FAILED,
                message=Message(role="agent", parts=[TextPart(text=str(e))])
            )
            await self.task_store.save(task)
            await self._notify_status_update(task, final=True)

    async def execute_command(self, user_input: str) -> Optional[dict]:
        """
        Execute a command by routing to the appropriate agent/skill.

        This replaces ToolRegistry.execute_command().

        Args:
            user_input: Raw user input string (e.g., "!list-models")

        Returns:
            Result dict with success/error, or None if command not found
        """
        result = self.get_agent_for_command(user_input)
        if not result:
            return None

        agent_id, skill_id = result

        # Parse command arguments using the tool's parser if available
        args = {}
        _, handler = self._agents.get(agent_id, (None, None))
        if handler:
            # Try to get the tool from the handler to use its parse_command_args
            tool = handler._get_tool_by_name(skill_id) if hasattr(handler, '_get_tool_by_name') else None
            if tool and hasattr(tool, 'parse_command_args'):
                args = tool.parse_command_args(user_input)
            else:
                args = self._parse_command_args(user_input)
        else:
            args = self._parse_command_args(user_input)

        try:
            task = await self.execute_skill(
                agent_id=agent_id,
                skill_id=skill_id,
                args=args,
                sync=True,
            )

            # Extract result from task artifacts
            if task.status.state == TaskState.COMPLETED and task.artifacts:
                artifact = task.artifacts[0]
                if artifact.parts:
                    part = artifact.parts[0]
                    if hasattr(part, 'data'):
                        return part.data
                    elif hasattr(part, 'text'):
                        return {"success": True, "result": part.text}

            if task.status.state == TaskState.FAILED:
                error_msg = "Task failed"
                if task.status.message and task.status.message.parts:
                    for p in task.status.message.parts:
                        if hasattr(p, 'text'):
                            error_msg = p.text
                            break
                return {"success": False, "error": error_msg}

            return {"success": True, "result": "Task completed"}

        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return {"success": False, "error": str(e)}

    def _parse_command_args(self, user_input: str) -> dict[str, Any]:
        """Parse command arguments from user input."""
        parts = user_input.strip().split()
        if len(parts) <= 1:
            return {}

        # Simple arg parsing: positional args become numbered keys
        args = {}
        positional_idx = 0

        for part in parts[1:]:
            if "=" in part:
                # Key=value format
                key, value = part.split("=", 1)
                args[key.strip("-")] = value
            elif part.startswith("--"):
                # Flag
                args[part[2:]] = True
            else:
                # Positional
                args[f"arg{positional_idx}"] = part
                positional_idx += 1

        return args

    async def create_task(
        self,
        params: TaskSendParams,
        agent_name: str,
    ) -> Task:
        """
        Create a new task from send parameters.

        Args:
            params: Task creation parameters including message
            agent_name: Name of the agent handling this task

        Returns:
            The created Task object
        """
        # Create or get session
        session = await self.session_service.get_session(params.sessionId)
        if not session:
            await self.session_service.create_session(
                session_id=params.sessionId,
                agent_name=agent_name,
                metadata=params.metadata or {}
            )

        # Append user message to session
        await self.session_service.append_event(
            session_id=params.sessionId,
            event_type="user_message",
            data={
                "role": params.message.role,
                "parts": [p.model_dump() for p in params.message.parts],
            }
        )

        # Create task
        task = Task(
            id=params.id,
            sessionId=params.sessionId,
            status=TaskStatus(state=TaskState.SUBMITTED),
            history=[params.message],
            metadata=params.metadata,
        )

        # Save to store
        await self.task_store.save(task)

        # Log to observability
        await self.observability_store.log_tool_call(
            agent_name=agent_name,
            tool_name="task_create",
            session_id=params.sessionId,
            metadata={"task_id": task.id}
        )

        # Notify subscribers
        await self._notify_status_update(task, final=False)

        logger.info(f"Task created: {task.id} in session {params.sessionId}")
        return task

    async def update_status(
        self,
        task_id: str,
        new_state: TaskState,
        message: Optional[Message] = None,
        agent_name: Optional[str] = None,
    ) -> Task:
        """
        Update task status with state transition validation.

        Args:
            task_id: ID of the task to update
            new_state: New state to transition to
            message: Optional status message
            agent_name: Agent performing the update (for observability)

        Returns:
            Updated Task object

        Raises:
            ValueError: If state transition is invalid
        """
        task = await self.task_store.get(task_id)
        if not task:
            raise ValueError(f"Task not found: {task_id}")

        # Validate state transition
        current_state = task.status.state
        if new_state not in VALID_TRANSITIONS.get(current_state, set()):
            raise ValueError(
                f"Invalid state transition: {current_state} -> {new_state}. "
                f"Valid transitions: {VALID_TRANSITIONS.get(current_state, set())}"
            )

        # Update status
        task.status = TaskStatus(
            state=new_state,
            message=message,
        )

        # Add message to history if provided
        if message and task.history is not None:
            task.history.append(message)

        # Save updated task (use save() to persist both status and history)
        await self.task_store.save(task)

        # Log to session
        if task.sessionId:
            await self.session_service.append_event(
                session_id=task.sessionId,
                event_type="status_update",
                data={
                    "task_id": task_id,
                    "old_state": current_state.value,
                    "new_state": new_state.value,
                }
            )

        # Log to observability
        if agent_name:
            await self.observability_store.log_agent_response(
                agent_name=agent_name,
                duration_ms=0,  # No timing for status updates
                session_id=task.sessionId,
                metadata={
                    "task_id": task_id,
                    "state_transition": f"{current_state} -> {new_state}"
                }
            )

        # Determine if this is a final state
        is_final = new_state in (TaskState.COMPLETED, TaskState.CANCELED, TaskState.FAILED)

        # Notify subscribers
        await self._notify_status_update(task, final=is_final)

        # If completed, optionally save to memory
        if is_final and self.memory_service and task.sessionId:
            await self._save_to_memory(task)

        logger.info(f"Task {task_id} transitioned: {current_state} -> {new_state}")
        return task

    async def add_artifact(
        self,
        task_id: str,
        artifact: Artifact,
        agent_name: Optional[str] = None,
    ) -> Task:
        """
        Add an artifact to a task.

        Args:
            task_id: ID of the task
            artifact: Artifact to add
            agent_name: Agent producing the artifact (for observability)

        Returns:
            Updated Task object
        """
        task = await self.task_store.get(task_id)
        if not task:
            raise ValueError(f"Task not found: {task_id}")

        # Add artifact
        await self.task_store.add_artifact(task_id, artifact)

        # Refresh task
        task = await self.task_store.get(task_id)

        # Notify subscribers
        await self._notify_artifact_update(task_id, artifact)

        logger.info(f"Artifact added to task {task_id}: {artifact.name}")
        return task

    async def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        return await self.task_store.get(task_id)

    async def get_session_tasks(
        self,
        session_id: str,
        limit: int = 100,
    ) -> list[Task]:
        """Get all tasks in a session."""
        return await self.task_store.list_tasks(session_id=session_id, limit=limit)

    async def get_pending_tasks(self, limit: int = 100) -> list[Task]:
        """Get all pending (submitted) tasks ready for processing."""
        return await self.task_store.get_pending_tasks(limit=limit)

    async def cancel_task(
        self,
        task_id: str,
        reason: Optional[str] = None,
        agent_name: Optional[str] = None,
    ) -> Task:
        """
        Cancel a task.

        Args:
            task_id: ID of the task to cancel
            reason: Optional cancellation reason
            agent_name: Agent performing the cancellation

        Returns:
            Updated Task object
        """
        message = None
        if reason:
            message = Message(
                role="agent",
                parts=[TextPart(text=f"Task canceled: {reason}")]
            )

        return await self.update_status(
            task_id=task_id,
            new_state=TaskState.CANCELED,
            message=message,
            agent_name=agent_name,
        )

    async def fail_task(
        self,
        task_id: str,
        error: str,
        agent_name: Optional[str] = None,
    ) -> Task:
        """
        Mark a task as failed.

        Args:
            task_id: ID of the task
            error: Error message
            agent_name: Agent reporting the failure

        Returns:
            Updated Task object
        """
        message = Message(
            role="agent",
            parts=[TextPart(text=f"Task failed: {error}")]
        )

        # Log error to observability
        if agent_name:
            task = await self.task_store.get(task_id)
            await self.observability_store.log_error(
                agent_name=agent_name,
                error_type="task_failure",
                error_message=error,
                session_id=task.sessionId if task else None,
                metadata={"task_id": task_id}
            )

        return await self.update_status(
            task_id=task_id,
            new_state=TaskState.FAILED,
            message=message,
            agent_name=agent_name,
        )

    async def complete_task(
        self,
        task_id: str,
        response: str,
        agent_name: Optional[str] = None,
        artifacts: Optional[list[Artifact]] = None,
    ) -> Task:
        """
        Complete a task with a response.

        Args:
            task_id: ID of the task
            response: Agent's response text
            agent_name: Agent completing the task
            artifacts: Optional artifacts to attach

        Returns:
            Updated Task object
        """
        # Add artifacts if provided
        if artifacts:
            for artifact in artifacts:
                await self.add_artifact(task_id, artifact, agent_name)

        message = Message(
            role="agent",
            parts=[TextPart(text=response)]
        )

        return await self.update_status(
            task_id=task_id,
            new_state=TaskState.COMPLETED,
            message=message,
            agent_name=agent_name,
        )

    # =========================================================================
    # SSE Streaming
    # =========================================================================

    async def subscribe(self, task_id: str) -> AsyncGenerator[dict, None]:
        """
        Subscribe to task updates via SSE.

        Args:
            task_id: ID of the task to subscribe to

        Yields:
            SSE event dictionaries with 'event' and 'data' keys
        """
        queue: asyncio.Queue = asyncio.Queue()

        # Register subscriber
        if task_id not in self._subscribers:
            self._subscribers[task_id] = []
        self._subscribers[task_id].append(queue)

        try:
            # Send current state first
            task = await self.task_store.get(task_id)
            if task:
                yield {
                    "event": "status",
                    "data": TaskStatusUpdateEvent(
                        id=task_id,
                        status=task.status,
                        final=task.status.state in (TaskState.COMPLETED, TaskState.CANCELED, TaskState.FAILED),
                    ).model_dump_json()
                }

            # Stream updates
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=HTTP_TIMEOUT_DEFAULT)
                    yield event

                    # Stop if final event
                    if event.get("final"):
                        break
                except asyncio.TimeoutError:
                    # Send keepalive
                    yield {"event": "keepalive", "data": ""}

        finally:
            # Unregister subscriber
            if task_id in self._subscribers:
                self._subscribers[task_id].remove(queue)
                if not self._subscribers[task_id]:
                    del self._subscribers[task_id]

    async def _notify_status_update(self, task: Task, final: bool) -> None:
        """Notify all subscribers of a status update."""
        # Call the global task completion callback if task is in terminal state
        if final and self._on_task_complete:
            try:
                self._on_task_complete(task)
            except Exception as e:
                logger.error(f"Task completion callback error: {e}")

        # Notify SSE subscribers
        if task.id not in self._subscribers:
            return

        event = {
            "event": "status",
            "data": TaskStatusUpdateEvent(
                id=task.id,
                status=task.status,
                final=final,
            ).model_dump_json(),
            "final": final,
        }

        for queue in self._subscribers[task.id]:
            await queue.put(event)

    async def _notify_artifact_update(self, task_id: str, artifact: Artifact) -> None:
        """Notify all subscribers of a new artifact."""
        if task_id not in self._subscribers:
            return

        event = {
            "event": "artifact",
            "data": TaskArtifactUpdateEvent(
                id=task_id,
                artifact=artifact,
            ).model_dump_json(),
        }

        for queue in self._subscribers[task_id]:
            await queue.put(event)

    # =========================================================================
    # Memory Integration
    # =========================================================================

    async def _save_to_memory(self, task: Task) -> None:
        """Save completed task conversation to long-term memory."""
        if not self.memory_service or not task.sessionId:
            return

        # Build content from history
        if task.history:
            content_parts = []
            for msg in task.history:
                role = msg.role
                for part in msg.parts:
                    if hasattr(part, 'text'):
                        content_parts.append(f"{role}: {part.text}")

            content = "\n".join(content_parts)

            await self.memory_service.add_memory(
                session_id=task.sessionId,
                content=content,
                tags=["task", task.status.state.value],
                metadata={
                    "task_id": task.id,
                    "completed_at": datetime.now().isoformat(),
                }
            )

    # =========================================================================
    # Cleanup
    # =========================================================================

    async def cleanup_old_tasks(
        self,
        older_than_days: int = 30,
    ) -> int:
        """
        Clean up old completed/failed/canceled tasks.

        Returns:
            Number of tasks deleted

        Note: Task cleanup requires timestamp-based deletion in task_store.
        Currently returns 0 as the underlying store does not support cleanup.
        """
        logger.warning(
            f"cleanup_old_tasks called for tasks older than {older_than_days} days, "
            "but cleanup is not yet implemented in task_store"
        )
        # Return 0 - no tasks deleted until cleanup is implemented
        return 0


# Factory function for easy creation
async def create_task_manager(
    db_path: str,
    include_memory: bool = True,
    include_feedback: bool = True,
) -> TaskManager:
    """
    Factory function to create a TaskManager with SQLite stores.

    Args:
        db_path: Path to SQLite database
        include_memory: Whether to include MemoryService
        include_feedback: Whether to include FeedbackStore

    Returns:
        Initialized TaskManager
    """
    from kestrel_sovereign.a2a.stores import (
        SQLiteTaskStore,
        SQLiteSessionService,
        SQLiteObservabilityStore,
        SQLiteMemoryService,
        SQLiteFeedbackStore,
    )

    task_store = SQLiteTaskStore(db_path)
    session_service = SQLiteSessionService(db_path)
    observability_store = SQLiteObservabilityStore(db_path)

    memory_service = SQLiteMemoryService(db_path) if include_memory else None
    feedback_store = SQLiteFeedbackStore(db_path) if include_feedback else None

    manager = TaskManager(
        task_store=task_store,
        session_service=session_service,
        observability_store=observability_store,
        memory_service=memory_service,
        feedback_store=feedback_store,
    )

    await manager.initialize()
    return manager
