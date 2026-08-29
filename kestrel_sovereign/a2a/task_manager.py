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
from typing import Any, AsyncGenerator, Callable, Coroutine, Optional

from kestrel_sovereign.kestrel_config.constants import HTTP_TIMEOUT_DEFAULT

from kestrel_sovereign.a2a.types import (
    Task,
    TaskState,
    TaskStatus,
    TaskSendParams,
    Message,
    TextPart,
    Artifact,
    TaskStatusUpdateEvent,
    TaskArtifactUpdateEvent,
)
from kestrel_sovereign.a2a.agent_card import AgentCard
from kestrel_sovereign.a2a.stores import (
    TaskStore,
    SessionService,
    MemoryService,
    ObservabilityStore,
    FeedbackStore,
)
from kestrel_sovereign.a2a.stores.unified.task_store import (
    TaskCancellationSnapshot,
    TaskMutationAuthorizationError,
    without_reserved_cancellation_receipt,
)
from typing import Protocol, runtime_checkable, TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from kestrel_sovereign.hooks import HooksManager


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


class TaskCancellationAuthorizationError(PermissionError):
    """The caller has no durable creator/recipient authority for a task."""


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
        on_task_submitted: Optional[Callable[[Task], None]] = None,
        causation_chain_provider: Optional[Callable[[], Optional[list]]] = None,
        host_agent_id: Optional[str] = None,
        on_task_cancelled: Optional[Callable[[Task], None]] = None,
        on_task_cancellation_started: Optional[
            Callable[[str, str], Optional[Callable[[], None]]]
        ] = None,
    ):
        self.task_store = task_store
        self.session_service = session_service
        self.observability_store = observability_store
        self.memory_service = memory_service
        self.feedback_store = feedback_store
        self.hooks_manager = hooks_manager
        if host_agent_id is not None and (
            not isinstance(host_agent_id, str) or not host_agent_id.strip()
        ):
            raise ValueError("host_agent_id must be a concrete durable identity")
        self.host_agent_id = host_agent_id

        # Callback for task completion notifications (for chat notifications)
        self._on_task_complete = on_task_complete

        # Callback for task creation. Mirrors `on_task_complete`; fires
        # synchronously from ``create_task`` immediately after the task
        # is persisted, BEFORE the SSE notify_status_update. Agents set
        # this in ``initialize()`` to bridge an inbound task into the
        # signal/dispatcher system — the cognition equivalent of the
        # ``channels.feature.py`` inbound-message wake. Without it, a
        # task created by a peer agent sits SUBMITTED in the store with
        # nobody acting on it until the next user-driven chat turn.
        self._on_task_submitted = on_task_submitted
        # Durable cancellation callback.  The owning agent uses this to cancel
        # an already-queued ``a2a.task_submitted`` cognition delivery after the
        # task row has atomically reached CANCELED.
        self._on_task_cancelled = on_task_cancelled
        # Process-local intent is announced before the shared-store await so a
        # live execution monitor cannot observe the committed row and cancel
        # the recipient's own decline before the post-commit callback runs.
        # The callback returns a rollback closure for refused/failed attempts.
        self._on_task_cancellation_started = on_task_cancellation_started

        # Callback returning the in-flight cognition turn's causation
        # chain (already serialized as list of dicts) or None when no
        # signal-driven turn is active. Used by `create_task` to attach
        # the chain to outbound A2A tasks so the receiving side
        # reconstructs the lineage and the dispatcher's cycle detection
        # can fire on real A→B→A loops. Without this, completion-driven
        # cognition would restart at depth 1 every iteration and the
        # only loop bound would be the per-source rate limit
        # (#905 review P1).
        self._causation_chain_provider = causation_chain_provider

        # Event subscribers for SSE streaming
        self._subscribers: dict[str, list[asyncio.Queue]] = {}

        # Agent registry: agent_id -> (AgentCard, TaskHandler)
        self._agents: dict[str, tuple[AgentCard, TaskHandler]] = {}

        # Skill -> agent mapping for fast lookup
        self._skill_to_agent: dict[str, str] = {}

        # Command prefix -> (agent_id, skill_id) mapping
        self._command_to_skill: dict[str, tuple[str, str]] = {}

        # Background skill executions started by execute_skill(sync=False).
        self._execution_tasks: set[asyncio.Task[None]] = set()
        self._execution_authorities: dict[
            asyncio.Task[None], tuple[str, str]
        ] = {}

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
        await self.drain_execution_tasks(cancel=True)

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
        authority_agent_id = self.host_agent_id or agent_id
        hook_feature_name = getattr(handler, "name", None) or agent_id

        # Create task
        task_id = uuid4().hex
        session_id = session_id or uuid4().hex

        # Execute PRE_TOOL_USE hooks if hooks_manager is configured
        if self.hooks_manager:
            from kestrel_sdk.hooks.base import HookEvent, HookInput, PermissionDecision

            hook_input = HookInput(
                session_id=session_id,
                hook_event_name=HookEvent.PRE_TOOL_USE.value,
                tool_name=skill_id,
                tool_input=args,
                feature_name=hook_feature_name,
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
                await self.task_store.save(
                    task,
                    creator_agent_id=authority_agent_id,
                    recipient_agent_id=authority_agent_id,
                )
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
        await self.task_store.save(
            task,
            creator_agent_id=authority_agent_id,
            recipient_agent_id=authority_agent_id,
        )

        if sync:
            # Execute synchronously with transaction safety
            task = await handler.handle_task(task)
            if task.status.state is TaskState.CANCELED:
                task = await self._persist_handler_cancellation(
                    task,
                    authority_agent_id=authority_agent_id,
                )
            else:
                saved: Optional[bool] = None
                try:
                    async with self.task_store._backend.transaction():
                        saved = await self.task_store.save_recipient_lifecycle(
                            task,
                            recipient_agent_id=authority_agent_id,
                        )
                except Exception as save_err:
                    logger.error(
                        f"Failed to save completed task {task.id}: {save_err}. "
                        "Retrying outside transaction..."
                    )
                    try:
                        saved = await self.task_store.save_recipient_lifecycle(
                            task,
                            recipient_agent_id=authority_agent_id,
                        )
                    except Exception as retry_err:
                        logger.critical(
                            f"Task {task.id} completed but save failed permanently: {retry_err}. "
                            f"Result lost for skill={skill_id}, agent={agent_id}"
                        )
                if saved is False:
                    task = await self.task_store.get(task.id) or task

            # Execute POST_TOOL_USE hooks
            if self.hooks_manager:
                from kestrel_sdk.hooks.base import HookEvent, HookInput

                post_hook_input = HookInput(
                    session_id=session_id,
                    hook_event_name=HookEvent.POST_TOOL_USE.value,
                    tool_name=skill_id,
                    tool_input=args,
                    feature_name=hook_feature_name,
                    tool_response={"task_id": task.id, "state": task.status.state.value},
                )
                # Post hooks run in parallel (non-blocking)
                await self.hooks_manager.execute_hooks_parallel(
                    HookEvent.POST_TOOL_USE,
                    post_hook_input
                )

            return task
        else:
            self._track_execution_task(
                self._execute_async(handler, task, authority_agent_id),
                task.id,
                authority_agent_id,
            )
            return task

    def _track_execution_task(
        self,
        coro: Coroutine[Any, Any, None],
        task_id: str,
        authority_agent_id: str,
    ) -> asyncio.Task[None]:
        """Own background task execution so close() can cancel and await it."""
        task = asyncio.create_task(coro, name=f"a2a-task-{task_id}")
        self._execution_tasks.add(task)
        self._execution_authorities[task] = (task_id, authority_agent_id)

        def _discard(done: asyncio.Task[None]) -> None:
            self._execution_tasks.discard(done)
            self._execution_authorities.pop(done, None)

        task.add_done_callback(_discard)
        return task

    async def drain_execution_tasks(self, *, cancel: bool = False) -> None:
        """Wait for tracked background executions, optionally cancelling them."""
        tasks = set(self._execution_tasks)
        if not tasks:
            return

        if cancel:
            caller_cancelled = False
            for task in tasks:
                binding = self._execution_authorities.get(task)
                if binding is not None:
                    task_id, authority_agent_id = binding
                    try:
                        await self.cancel_task(
                            task_id,
                            reason="Task canceled during shutdown",
                            agent_name=authority_agent_id,
                        )
                    except ValueError:
                        pass
                    except asyncio.CancelledError:
                        caller_cancelled = True
                    except Exception:
                        # Shutdown must still stop the in-memory worker when its
                        # durable receipt cannot be persisted. The backend error
                        # is observable in logs, but cannot strand the worker or
                        # prevent the remaining stores from closing.
                        logger.exception(
                            "Could not persist shutdown cancellation for task %s",
                            task_id,
                        )
                    finally:
                        task.cancel()
                else:
                    task.cancel()

        await asyncio.gather(*tasks, return_exceptions=True)
        self._execution_tasks.difference_update(tasks)
        if cancel and caller_cancelled:
            raise asyncio.CancelledError()

    @staticmethod
    def _cancellation_reason(task: Task) -> str:
        message = task.status.message
        if message is not None:
            for part in message.parts:
                text = getattr(part, "text", None)
                if isinstance(text, str) and text:
                    return text
        return "Task canceled"

    async def _persist_handler_cancellation(
        self,
        task: Task,
        *,
        authority_agent_id: str,
    ) -> Task:
        """Commit handler cancellation or return an earlier terminal winner."""

        try:
            return await self.cancel_task(
                task.id,
                reason=self._cancellation_reason(task),
                agent_name=authority_agent_id,
                task_payload=task,
            )
        except ValueError:
            current = await self.task_store.get(task.id)
            if current is not None and current.status.state in {
                TaskState.COMPLETED,
                TaskState.FAILED,
                TaskState.CANCELED,
            }:
                return current
            raise

    async def _execute_async(
        self,
        handler: TaskHandler,
        task: Task,
        authority_agent_id: str,
    ) -> None:
        """Execute a task asynchronously and update the store."""
        try:
            task = await handler.handle_task(task)
            task, owns_notification = await self._persist_execution_outcome(
                task, authority_agent_id=authority_agent_id
            )
            if owns_notification:
                await self._notify_status_update(task, final=True)
        except asyncio.CancelledError:
            try:
                await self.cancel_task(
                    task.id,
                    reason="Task canceled during shutdown",
                    agent_name=authority_agent_id,
                )
            except ValueError:
                # An independently authorized cancellation or other terminal
                # result won the race. Never narrate over that durable state.
                pass
            raise
        except Exception as e:
            logger.error(f"Async task execution failed: {e}")
            task.status = TaskStatus(
                state=TaskState.FAILED,
                message=Message(role="agent", parts=[TextPart(text=str(e))])
            )
            task, owns_notification = await self._persist_execution_outcome(
                task, authority_agent_id=authority_agent_id
            )
            if owns_notification:
                await self._notify_status_update(task, final=True)

    async def _persist_execution_outcome(
        self,
        task: Task,
        *,
        authority_agent_id: str,
    ) -> tuple[Task, bool]:
        """Persist a worker result and report ownership of terminal notification."""

        if task.status.state is TaskState.CANCELED:
            return (
                await self._persist_handler_cancellation(
                    task,
                    authority_agent_id=authority_agent_id,
                ),
                False,
            )

        saved = await self.task_store.save_recipient_lifecycle(
            task,
            recipient_agent_id=authority_agent_id,
        )
        if saved is False:
            # Another terminal writer won its CAS and owns the corresponding
            # completion signal. Returning its durable state must not emit the
            # same terminal event a second time.
            return (await self.task_store.get(task.id) or task), False
        return task, True

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
        artifacts: Optional[list[Artifact]] = None,
        *,
        creator_agent_id: Optional[str] = None,
    ) -> Task:
        """
        Create a new task from send parameters.

        Args:
            params: Task creation parameters including message
            agent_name: Name of the agent handling this task
            artifacts: Optional artifacts/references the SENDER attached
                at task-creation time (send-side handoff payload). These
                are persisted on the task at SUBMITTED so the recipient
                can retrieve them before producing any response — the
                send-side mirror of the responder-side
                ``add_artifact`` flow.
            creator_agent_id: Trusted identity of the assigning agent.  This
                must come from verified envelope or host-attested provenance,
                never from request metadata.  Local tasks default to
                ``agent_name`` as both creator and recipient.

        Returns:
            The created Task object
        """
        # Attach the in-flight turn's causation chain to outbound task
        # metadata if a provider is registered (KestrelAgent wires this
        # in initialize()). The receiving side reconstructs the chain
        # via signals.sources.a2a._deserialize_chain when the task
        # terminates. Empty/missing chains are not stored to avoid
        # bloating metadata with empty lists.
        outbound_metadata = dict(params.metadata) if params.metadata else {}
        if self._causation_chain_provider is not None:
            try:
                chain = self._causation_chain_provider()
            except Exception as e:
                logger.warning(
                    "causation_chain_provider raised; proceeding "
                    "without chain metadata: %s", e,
                )
                chain = None
            if chain:
                outbound_metadata["causation_chain"] = chain

        # Create task. Sender-supplied artifacts (if any) are attached
        # at creation so the recipient can retrieve the handoff payload
        # the moment the task surfaces, without a second round-trip.
        task = Task(
            id=params.id,
            sessionId=params.sessionId,
            status=TaskStatus(state=TaskState.SUBMITTED),
            history=[params.message],
            metadata=without_reserved_cancellation_receipt(outbound_metadata),
            artifacts=list(artifacts) if artifacts else None,
        )

        # Save to store
        if not isinstance(agent_name, str) or not agent_name.strip():
            raise ValueError("Task authority requires concrete agent identities")
        recipient_agent_id = agent_name
        if creator_agent_id is None:
            creator_id = recipient_agent_id
        elif isinstance(creator_agent_id, str) and creator_agent_id.strip():
            creator_id = creator_agent_id
        else:
            raise ValueError("Task authority requires concrete agent identities")
        await self.task_store.save(
            task,
            creator_agent_id=creator_id,
            recipient_agent_id=recipient_agent_id,
        )

        # The insert-only task reservation is deliberately first: a duplicate
        # caller-supplied ID must be rejected before it can append another user
        # message to the existing session. Session and observability projections
        # are best-effort after the durable task commit; their failure cannot
        # turn an accepted task into a 500 that a retry experiences as a 409.
        try:
            session = await self.session_service.get_session(params.sessionId)
            if not session:
                await self.session_service.create_session(
                    session_id=params.sessionId,
                    agent_name=agent_name,
                    metadata=params.metadata or {},
                )
            await self.session_service.append_event(
                session_id=params.sessionId,
                event_type="user_message",
                data={
                    "role": params.message.role,
                    "parts": [p.model_dump() for p in params.message.parts],
                },
            )
        except Exception:
            logger.warning(
                "Task %s was created but its session projection failed",
                task.id,
                exc_info=True,
            )

        try:
            await self.observability_store.log_tool_call(
                agent_name=agent_name,
                tool_name="task_create",
                session_id=params.sessionId,
                metadata={"task_id": task.id},
            )
        except Exception:
            logger.warning(
                "Task %s was created but observability logging failed",
                task.id,
                exc_info=True,
            )

        # Inbound-task callback: agent bridges this into the signal
        # dispatcher so the cognition loop wakes up and acts on the new
        # task. Mirrors `_on_task_complete` for the complete-direction
        # signal; without this hook, a peer-submitted task sits
        # SUBMITTED with no one processing it (the Emma/Meridian
        # symptom). Synchronous callback; agent-side handler dispatches
        # the actual async enqueue via background-task tracking
        # (see KestrelAgent._on_task_submitted).
        if self._on_task_submitted is not None:
            try:
                self._on_task_submitted(task)
            except Exception as e:
                logger.warning(
                    "on_task_submitted callback failed for %s: %s",
                    task.id, e, exc_info=True,
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
        *,
        recipient_agent_id: str,
    ) -> Task:
        """
        Update task status with state transition validation.

        Args:
            task_id: ID of the task to update
            new_state: New state to transition to
            message: Optional status message
            agent_name: Agent performing the update (for observability only)
            recipient_agent_id: Trusted durable recipient performing the write

        Returns:
            Updated Task object

        Raises:
            ValueError: If state transition is invalid
        """
        if new_state is TaskState.CANCELED:
            raise TaskCancellationAuthorizationError(
                "CANCELED is an authorized transition; use cancel_task"
            )

        task = await self.task_store.get_for_recipient(
            task_id,
            recipient_agent_id,
        )
        if not task:
            raise TaskMutationAuthorizationError(
                f"Task mutation was not authorized or task was not found: {task_id}"
            )

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
        saved = await self.task_store.save_recipient_lifecycle(
            task,
            recipient_agent_id=recipient_agent_id,
        )
        if saved is False:
            persisted = await self.task_store.get_for_recipient(
                task_id,
                recipient_agent_id,
            )
            state = persisted.status.state if persisted else TaskState.UNKNOWN
            raise ValueError(
                f"Invalid state transition: task is already {state}"
            )

        # Determine if this is a final state
        is_final = new_state in (TaskState.COMPLETED, TaskState.CANCELED, TaskState.FAILED)
        await self._project_status_transition(
            task,
            old_state=current_state.value,
            new_state=new_state,
            agent_name=agent_name,
            is_final=is_final,
        )

        logger.info(f"Task {task_id} transitioned: {current_state} -> {new_state}")
        return task

    async def add_artifact(
        self,
        task_id: str,
        artifact: Artifact,
        agent_name: Optional[str] = None,
        *,
        recipient_agent_id: str,
    ) -> Task:
        """
        Add an artifact to a task.

        Args:
            task_id: ID of the task
            artifact: Artifact to add
            agent_name: Agent producing the artifact (for observability only)
            recipient_agent_id: Trusted durable recipient performing the write

        Returns:
            Updated Task object
        """
        task = await self.task_store.get_for_recipient(
            task_id,
            recipient_agent_id,
        )
        if not task:
            raise TaskMutationAuthorizationError(
                f"Task mutation was not authorized or task was not found: {task_id}"
            )

        # Add artifact
        await self.task_store.add_artifact(
            task_id,
            artifact,
            recipient_agent_id=recipient_agent_id,
        )

        # Refresh task
        task = await self.task_store.get_for_recipient(
            task_id,
            recipient_agent_id,
        )

        # Notify subscribers
        await self._notify_artifact_update(task_id, artifact)

        logger.info(f"Artifact added to task {task_id}: {artifact.name}")
        return task

    async def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        return await self.task_store.get(task_id)

    async def get_task_cancellation_snapshot(
        self,
        task_id: str,
    ) -> Optional[TaskCancellationSnapshot]:
        """Read the minimal durable state used to withdraw live cognition."""

        return await self.task_store.get_cancellation_snapshot(task_id)

    async def is_task_recipient(self, task_id: str, agent_id: str) -> bool:
        """Whether this manager's durable task row delegates execution to agent."""

        return await self.task_store.is_task_recipient(task_id, agent_id)

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

    async def list_tasks(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        status: Optional[TaskState] = None,
        limit: int = 100,
    ) -> list[Task]:
        """List tasks with optional filters, across ALL states.

        Unlike ``get_pending_tasks`` (which only returns SUBMITTED tasks),
        this passes through to the store's full-table query so callers can
        filter by any ``TaskState`` (completed, failed, working, canceled).
        """
        return await self.task_store.list_tasks(
            session_id=session_id,
            user_id=user_id,
            status=status,
            limit=limit,
        )

    async def cancel_task(
        self,
        task_id: str,
        reason: Optional[str] = None,
        agent_name: Optional[str] = None,
        recipient_agent_id: Optional[str] = None,
        task_payload: Optional[Task] = None,
    ) -> Task:
        """
        Cancel a task.

        Args:
            task_id: ID of the task to cancel
            reason: Optional cancellation reason
            agent_name: Durable DID of the agent performing the cancellation.
                Display names and causation metadata are not authority.
            recipient_agent_id: Optional durable DID of the recipient through
                which a peer cancellation was routed. When present, it joins
                the atomic authorization predicate so another manager sharing
                the same task table cannot mutate this row.
            task_payload: Optional canceled handler result whose artifacts,
                history, and non-authority metadata must commit atomically with
                the authorized terminal transition.

        Returns:
            Updated Task object
        """
        if not isinstance(agent_name, str) or not agent_name:
            raise TaskCancellationAuthorizationError(
                "Task cancellation requires a concrete agent identity"
            )

        cancel_kwargs = {
            "actor_agent_id": agent_name,
            "reason": reason,
        }
        if recipient_agent_id is not None:
            cancel_kwargs["expected_recipient_agent_id"] = recipient_agent_id
        if task_payload is not None:
            cancel_kwargs["task_payload"] = task_payload
        rollback_local_intent = None
        if self._on_task_cancellation_started is not None:
            rollback_local_intent = self._on_task_cancellation_started(
                task_id,
                agent_name,
            )
        try:
            task = await self.task_store.cancel_if_authorized(
                task_id,
                **cancel_kwargs,
            )
        except BaseException:
            if rollback_local_intent is not None:
                rollback_local_intent()
            raise
        if task is None:
            try:
                current = await self.task_store.get(task_id)
            except BaseException:
                if rollback_local_intent is not None:
                    rollback_local_intent()
                raise
            if current is None:
                if rollback_local_intent is not None:
                    rollback_local_intent()
                raise ValueError(f"Task not found: {task_id}")
            receipt = (current.metadata or {}).get("cancellation_receipt") or {}
            if (
                current.status.state is TaskState.CANCELED
                and receipt.get("actor_agent_id") == agent_name
                and (
                    recipient_agent_id is None
                    or await self.task_store.is_task_recipient(
                        task_id,
                        recipient_agent_id,
                    )
                )
            ):
                # The first atomic cancellation may have committed while its
                # transport response was lost. Return that exact durable
                # receipt without replaying notifications/projections. Keep
                # the just-acquired local execution exemption: this retry is
                # still the same authorized recipient decline and its monitor
                # must not cancel the response after observing CANCELED.
                return current
            if rollback_local_intent is not None:
                rollback_local_intent()
            if current.status.state not in {
                TaskState.SUBMITTED,
                TaskState.WORKING,
                TaskState.INPUT_REQUIRED,
            }:
                raise ValueError(
                    f"Invalid state transition: {current.status.state} -> "
                    f"{TaskState.CANCELED}"
                )
            raise TaskCancellationAuthorizationError(
                f"Agent {agent_name!r} is not authorized to cancel task {task_id!r}"
            )

        receipt = (task.metadata or {}).get("cancellation_receipt") or {}
        previous_state = receipt.get("status_before")
        if self._on_task_cancelled is not None:
            try:
                self._on_task_cancelled(task)
            except Exception as exc:
                logger.warning(
                    "on_task_cancelled callback failed for %s: %s",
                    task.id,
                    exc,
                    exc_info=True,
                )
        await self._project_status_transition(
            task,
            old_state=str(previous_state or TaskState.UNKNOWN.value),
            new_state=TaskState.CANCELED,
            agent_name=agent_name,
            is_final=True,
            reason=reason,
        )

        logger.info(
            "Task %s canceled by %s: %s -> %s",
            task_id,
            agent_name,
            previous_state,
            TaskState.CANCELED.value,
        )
        return task

    async def fail_task(
        self,
        task_id: str,
        error: str,
        agent_name: Optional[str] = None,
        *,
        recipient_agent_id: str,
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
            recipient_agent_id=recipient_agent_id,
        )

    async def complete_task(
        self,
        task_id: str,
        response: str,
        agent_name: Optional[str] = None,
        artifacts: Optional[list[Artifact]] = None,
        *,
        recipient_agent_id: str,
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
                await self.add_artifact(
                    task_id,
                    artifact,
                    agent_name,
                    recipient_agent_id=recipient_agent_id,
                )

        message = Message(
            role="agent",
            parts=[TextPart(text=response)]
        )

        return await self.update_status(
            task_id=task_id,
            new_state=TaskState.COMPLETED,
            message=message,
            agent_name=agent_name,
            recipient_agent_id=recipient_agent_id,
        )

    async def _project_status_transition(
        self,
        task: Task,
        *,
        old_state: str,
        new_state: TaskState,
        agent_name: Optional[str],
        is_final: bool,
        reason: Optional[str] = None,
    ) -> None:
        """Best-effort projections after the durable task row has committed.

        Session history, observability, SSE, and memory are projections of the
        canonical task row.  None may turn a committed transition into an
        apparent failure that a retry then experiences as an invalid state.
        """

        event_data: dict[str, Any] = {
            "task_id": task.id,
            "old_state": old_state,
            "new_state": new_state.value,
        }
        if new_state is TaskState.CANCELED:
            event_data.update(
                actor_agent_id=agent_name,
                reason=reason,
            )
        if task.sessionId:
            try:
                await self.session_service.append_event(
                    session_id=task.sessionId,
                    event_type="status_update",
                    data=event_data,
                )
            except Exception:
                logger.warning(
                    "Task %s transitioned durably but its session projection failed",
                    task.id,
                    exc_info=True,
                )

        if agent_name:
            metadata: dict[str, Any] = {
                "task_id": task.id,
                "state_transition": f"{old_state} -> {new_state.value}",
            }
            if new_state is TaskState.CANCELED:
                metadata["cancellation_reason"] = reason
            try:
                await self.observability_store.log_agent_response(
                    agent_name=agent_name,
                    duration_ms=0,
                    session_id=task.sessionId,
                    metadata=metadata,
                )
            except Exception:
                logger.warning(
                    "Task %s transitioned durably but observability logging failed",
                    task.id,
                    exc_info=True,
                )

        try:
            await self._notify_status_update(task, final=is_final)
        except Exception:
            logger.warning(
                "Task %s transitioned durably but subscriber notification failed",
                task.id,
                exc_info=True,
            )
        if is_final and self.memory_service and task.sessionId:
            try:
                await self._save_to_memory(task)
            except Exception:
                logger.warning(
                    "Task %s transitioned durably but memory recording failed",
                    task.id,
                    exc_info=True,
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
            # Send current state first. If the task is already terminal
            # (late subscriber), yield with top-level ``final`` so the
            # endpoint loop breaks cleanly — without this, the SSE
            # stream stayed alive emitting keepalives until the client
            # timed out (codex round 1 P2 on PR #1453). Match the same
            # envelope shape ``_notify_status_update`` uses for the
            # live-update path so subscribers can rely on one contract.
            task = await self.task_store.get(task_id)
            if task:
                terminal = task.status.state in (
                    TaskState.COMPLETED,
                    TaskState.CANCELED,
                    TaskState.FAILED,
                )
                yield {
                    "event": "status",
                    "data": TaskStatusUpdateEvent(
                        id=task_id,
                        status=task.status,
                        final=terminal,
                    ).model_dump_json(),
                    "final": terminal,
                }
                if terminal:
                    # Don't enter the keepalive loop — the subscriber
                    # already has the terminal frame.
                    return

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
        """
        return await self.task_store.cleanup_old(older_than_days=older_than_days)


# Factory function for easy creation
async def create_task_manager(
    db_path: str,
    include_memory: bool = True,
    include_feedback: bool = True,
    host_agent_id: Optional[str] = None,
) -> TaskManager:
    """
    Factory function to create a TaskManager with SQLite stores.

    Args:
        db_path: Path to SQLite database
        include_memory: Whether to include MemoryService
        include_feedback: Whether to include FeedbackStore
        host_agent_id: Durable DID that owns in-process feature tasks

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
        host_agent_id=host_agent_id,
    )

    await manager.initialize()
    return manager
