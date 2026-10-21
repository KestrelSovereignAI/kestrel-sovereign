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
from collections.abc import AsyncGenerator, Callable, Coroutine, Sequence
from datetime import UTC, datetime
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
)
from uuid import uuid4

from a2a.helpers.proto_helpers import new_task_from_user_message, new_text_message
from a2a.types import (
    AgentSkill,
    Artifact,
    Message,
    Role,
    SendMessageRequest,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)
from a2a.utils.errors import InvalidParamsError
from a2a.utils.proto_utils import validate_proto_required_fields
from google.protobuf.json_format import MessageToDict

from kestrel_sovereign._async_ownership import (
    await_owned_task,
    create_owned_task,
    raise_owned_outcome,
)
from kestrel_sovereign.a2a.extensions import (
    causation_metadata,
    correlated_answer_metadata,
    parse_message_metadata,
    parse_question_correlation,
)
from kestrel_sovereign.a2a.lifecycle import (
    STREAM_STOP_STATES,
    TERMINAL_STATES,
    task_state_label,
)
from kestrel_sovereign.a2a.projections import embedded_json_value
from kestrel_sovereign.a2a.scope import NativeRecipientScope
from kestrel_sovereign.a2a.skill_handler import TaskHandler
from kestrel_sovereign.a2a.stores import (
    FeedbackStore,
    MemoryService,
    ObservabilityStore,
    SessionService,
    TaskStore,
)
from kestrel_sovereign.a2a.stores.unified.context_store import (
    ContextOwnershipError,
    ContextStore,
)
from kestrel_sovereign.a2a.stores.unified.outbound_store import OutboundStore
from kestrel_sovereign.a2a.stores.unified.question_store import NativeQuestionStore
from kestrel_sovereign.a2a.stores.unified.task_store import (
    NativeTaskSnapshot,
    TaskCancellationSnapshot,
    TaskMutationAuthorizationError,
    without_reserved_cancellation_receipt,
)
from kestrel_sovereign.execution_custody import (
    execution_terminal_error,
    is_execution_control_error,
)
from kestrel_sovereign.storage.db.interface import DatabaseBackend

if TYPE_CHECKING:
    from kestrel_sovereign.hooks import HooksManager


logger = logging.getLogger(__name__)


class TaskCancellationAuthorizationError(PermissionError):
    """The caller has no durable creator/recipient authority for a task."""


class OutboundActorScopeError(PermissionError):
    """The requested sender does not own this manager's primary outbound ledger."""


# Only a recipient may advance a live task; cancellation has its own atomic
# creator/recipient authority predicate. Interrupted tasks can resume or end.
_LIVE_STATES = frozenset(
    {
        TaskState.TASK_STATE_SUBMITTED,
        TaskState.TASK_STATE_WORKING,
        TaskState.TASK_STATE_INPUT_REQUIRED,
        TaskState.TASK_STATE_AUTH_REQUIRED,
    }
)
VALID_TRANSITIONS: dict[int, frozenset[int]] = {
    state: (_LIVE_STATES | TERMINAL_STATES) - {state} for state in _LIVE_STATES
}


class TaskManager:
    """
    High-level task lifecycle management.

    Coordinates between TaskStore, SessionService, and ObservabilityStore
    to provide a unified API for task operations.
    """

    @classmethod
    def from_backend(
        cls,
        backend: DatabaseBackend,
        *,
        scope: NativeRecipientScope,
        owns_backend: bool,
        include_memory: bool = True,
        include_feedback: bool = True,
        hooks_manager: Optional["HooksManager"] = None,
        on_task_complete: Callable[[NativeTaskSnapshot], None] | None = None,
        on_task_submitted: Callable[[NativeTaskSnapshot], None] | None = None,
        causation_chain_provider: Callable[[], list | None] | None = None,
        on_task_cancelled: Callable[[Task], None] | None = None,
        on_task_cancellation_started: Callable[[str, str], Callable[[], None] | None]
        | None = None,
    ) -> "TaskManager":
        """Build all native stores on one backend with explicit retirement policy.

        Construction performs no I/O. Register boot rollback before awaiting
        initialize; a host passes its existing backend with owns_backend=False.
        """
        return cls(
            task_store=TaskStore(backend, scope=scope, owns_backend=owns_backend),
            context_store=ContextStore(backend, owns_backend=owns_backend),
            session_service=SessionService(backend, owns_backend=owns_backend),
            observability_store=ObservabilityStore(backend, owns_backend=owns_backend),
            memory_service=MemoryService(backend, owns_backend=owns_backend)
            if include_memory
            else None,
            feedback_store=FeedbackStore(backend, owns_backend=owns_backend)
            if include_feedback
            else None,
            hooks_manager=hooks_manager,
            on_task_complete=on_task_complete,
            on_task_submitted=on_task_submitted,
            causation_chain_provider=causation_chain_provider,
            on_task_cancelled=on_task_cancelled,
            on_task_cancellation_started=on_task_cancellation_started,
        )

    def __init__(
        self,
        task_store: TaskStore,
        session_service: SessionService,
        observability_store: ObservabilityStore,
        memory_service: MemoryService | None = None,
        feedback_store: FeedbackStore | None = None,
        hooks_manager: Optional["HooksManager"] = None,
        on_task_complete: Callable[[NativeTaskSnapshot], None] | None = None,
        on_task_submitted: Callable[[NativeTaskSnapshot], None] | None = None,
        causation_chain_provider: Callable[[], list | None] | None = None,
        on_task_cancelled: Callable[[Task], None] | None = None,
        on_task_cancellation_started: Callable[[str, str], Callable[[], None] | None]
        | None = None,
        *,
        context_store: ContextStore,
    ):
        if context_store.backend is not task_store.backend:
            raise ValueError(
                "Task and context admission require the same database backend"
            )
        self.context_store = context_store
        self.task_store = task_store
        self.session_service = session_service
        self.observability_store = observability_store
        self.memory_service = memory_service
        self.feedback_store = feedback_store
        self.hooks_manager = hooks_manager
        self.outbound_store = OutboundStore(
            task_store.backend, sender_scope=task_store.scope, owns_backend=False
        )
        self.question_store = NativeQuestionStore(self.outbound_store)

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

        # Feature providers expose skills; they do not mint peer Agent Cards.
        self._providers: dict[str, tuple[tuple[AgentSkill, ...], TaskHandler]] = {}

        # Skill -> agent mapping for fast lookup
        self._skill_to_agent: dict[str, str] = {}

        # Command prefix -> (agent_id, skill_id) mapping
        self._command_to_skill: dict[str, tuple[str, str]] = {}

        # Background skill executions started by execute_skill(sync=False).
        self._execution_tasks: set[asyncio.Task[None]] = set()
        self._execution_authorities: dict[asyncio.Task[None], tuple[str, str]] = {}
        # Retain the first infrastructure failure or stronger custody control
        # after a child retires. Joining workers cannot reconcile a failed or
        # uncertain durable write, and must not erase its original error.
        self._execution_control: BaseException | None = None

    @property
    def did(self) -> str:
        """This manager's owner, derived from its one server-owned recipient scope."""
        return self.task_store.scope.recipient_id

    def outbound_for_actor(self, actor_id: str) -> OutboundStore:
        """Return the one outbound ledger bound to this actor and primary store."""
        if (
            not isinstance(actor_id, str)
            or not actor_id.strip()
            or not isinstance(self.task_store, TaskStore)
            or not isinstance(self.outbound_store, OutboundStore)
            or self.did != actor_id
            or self.outbound_store.sender_scope != self.task_store.scope
            or self.outbound_store.backend is not self.task_store.backend
            or not isinstance(self.question_store, NativeQuestionStore)
            or self.question_store.outbound_store is not self.outbound_store
            or self.question_store.backend is not self.task_store.backend
        ):
            raise OutboundActorScopeError(
                "Outbound ledger has a different local actor, tenant or primary backend"
            )
        return self.outbound_store

    def questions_for_actor(self, actor_id: str) -> NativeQuestionStore:
        self.outbound_for_actor(actor_id)
        return self.question_store

    async def initialize(self) -> None:
        """Initialize all stores after the read-only stored-protocol gate."""
        from kestrel_sovereign.a2a.native_schema import (
            NativeStoreSchema,
            native_schema_initialization,
            native_store_schema_ready,
            publish_native_store_schema,
        )

        components = [
            (
                NativeStoreSchema.CORE,
                (
                    self.outbound_store,
                    self.question_store,
                    self.task_store,
                    self.context_store,
                    self.session_service,
                    self.observability_store,
                ),
            ),
        ]
        if self.memory_service is not None:
            components.append((NativeStoreSchema.MEMORY, (self.memory_service,)))
        if self.feedback_store is not None:
            components.append((NativeStoreSchema.FEEDBACK, (self.feedback_store,)))
        backend = self.task_store.backend
        async with native_schema_initialization(backend):
            for component, stores in components:
                if await native_store_schema_ready(backend, component):
                    continue
                for store in stores:
                    await store.initialize()
                await publish_native_store_schema(backend, component)
        logger.info("TaskManager initialized with all stores")

    async def close(self) -> None:
        """Close all stores and release resources.

        This must be called during shutdown to prevent thread leaks from
        aiosqlite connections. Every store is closed even when another fails,
        and the method returns only when all of them closed (#3558). The
        stores stay on the manager, so a later call retries every store whose
        close failed; a store that already closed is closed again harmlessly.

        Raises:
            Exception: The first store close that failed, with a note naming
                every other store that failed to close.
            asyncio.CancelledError: The caller was cancelled. The stores not
                yet closed stay open for a later call.
        """
        failures: list[tuple[str, BaseException]] = []
        try:
            await self.drain_execution_tasks(cancel=True)
        except BaseException as error:
            if not isinstance(error, Exception) and not is_execution_control_error(
                error
            ):
                raise
            failures.append(("execution_tasks", error))

        # Close all stores in reverse order of initialization
        stores = (
            ("feedback_store", self.feedback_store),
            ("memory_service", self.memory_service),
            ("observability_store", self.observability_store),
            ("session_service", self.session_service),
            ("context_store", self.context_store),
            ("task_store", self.task_store),
            ("question_store", self.question_store),
            ("outbound_store", self.outbound_store),
        )
        for name, store in stores:
            if store is None:
                continue
            try:
                outcome = await await_owned_task(create_owned_task(store.close()))
                error = execution_terminal_error(outcome.error, outcome.cancellation)
                if error is not None:
                    raise error
            except BaseException as e:
                if not isinstance(
                    e, (Exception, asyncio.CancelledError)
                ) and not is_execution_control_error(e):
                    raise
                logger.warning("Error closing %s: %s", name, e, exc_info=True)
                failures.append((name, e))

        if failures:
            first = execution_terminal_error(*(error for _, error in failures))
            first_name = next(name for name, error in failures if error is first)
            first.add_note(f"TaskManager could not close {first_name}")
            for name, error in failures:
                if error is not first:
                    first.add_note(
                        f"TaskManager could not close {name} either: {error}"
                    )
            raise first

        logger.info("TaskManager closed all stores")

    # =========================================================================
    # Agent Registry (Feature/A2A Unification)
    # =========================================================================

    def register_provider(
        self,
        provider_id: str,
        skills: Sequence[AgentSkill],
        handler: TaskHandler,
        command_prefixes: dict[str, str] | None = None,
    ) -> None:
        """Index a feature's native skills without impersonating a peer agent."""
        if not isinstance(provider_id, str) or not provider_id.strip():
            raise ValueError("A task provider requires a concrete registration ID")
        if provider_id in self._providers:
            raise ValueError(f"Task provider is already registered: {provider_id}")
        copied = []
        for skill in skills:
            if not isinstance(skill, AgentSkill):
                raise TypeError("Task providers require official A2A skills")
            validate_proto_required_fields(skill)
            if skill.id in self._skill_to_agent or any(
                item.id == skill.id for item in copied
            ):
                raise ValueError(f"Task skill is already registered: {skill.id}")
            owned = AgentSkill()
            owned.CopyFrom(skill)
            copied.append(owned)
        prefixes = dict(command_prefixes or {})
        skill_ids = {item.id for item in copied}
        for prefix, skill_id in prefixes.items():
            if prefix in self._command_to_skill or skill_id not in skill_ids:
                raise ValueError(f"Invalid or occupied task command: {prefix}")
        self._providers[provider_id] = (tuple(copied), handler)
        for skill in copied:
            self._skill_to_agent[f"{provider_id}.{skill.id}"] = provider_id
            self._skill_to_agent[skill.id] = provider_id
        for prefix, skill_id in prefixes.items():
            self._command_to_skill[prefix] = (provider_id, skill_id)

    def unregister_provider(self, provider_id: str) -> None:
        """Retire a feature's skills and command routes together."""
        registered = self._providers.pop(provider_id, None)
        if registered is None:
            return
        for skill in registered[0]:
            self._skill_to_agent.pop(f"{provider_id}.{skill.id}", None)
            self._skill_to_agent.pop(skill.id, None)
        for command, registration in list(self._command_to_skill.items()):
            if registration[0] == provider_id:
                del self._command_to_skill[command]

    def get_agent_skills(self) -> list[AgentSkill]:
        """Return independent native skill values for the host's one Agent Card."""
        return [
            skill for skills in self.get_provider_skills().values() for skill in skills
        ]

    def get_provider_skills(self) -> dict[str, list[AgentSkill]]:
        """Expose the registered native catalog without leaking mutable storage."""
        return {
            provider_id: [
                AgentSkill.FromString(skill.SerializeToString()) for skill in skills
            ]
            for provider_id, (skills, _) in self._providers.items()
        }

    def get_commands(self) -> dict[str, str]:
        """Describe exactly the command prefixes admitted by registration."""
        skills = {skill.id: skill.description for skill in self.get_agent_skills()}
        return {
            prefix: skills[skill_id]
            for prefix, (_, skill_id) in self._command_to_skill.items()
        }

    def get_agent_for_skill(self, skill_id: str) -> str | None:
        """Find which agent handles a skill."""
        return self._skill_to_agent.get(skill_id)

    def get_agent_for_command(self, command: str) -> tuple[str, str] | None:
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
        for agent_id, (_, handler) in self._providers.items():
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
        session_id: str | None = None,
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
        if agent_id not in self._providers:
            raise ValueError(f"Unknown agent: {agent_id}")

        _, handler = self._providers[agent_id]
        authority_agent_id = self.did
        hook_feature_name = getattr(handler, "name", None) or agent_id

        session_id = session_id or uuid4().hex
        denied_reason = None
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
                HookEvent.PRE_TOOL_USE, hook_input
            )
            if hook_output.permission_decision == PermissionDecision.DENY:
                denied_reason = (
                    hook_output.permission_reason or "Blocked by security policy"
                )
            elif hook_output.updated_input is not None:
                args = hook_output.updated_input
        metadata = {"skill": skill_id, "args": args, "provider_id": agent_id}
        if self._causation_chain_provider is not None:
            chain = self._causation_chain_provider()
            if chain is not None:
                metadata.update(causation_metadata(chain))
        request = SendMessageRequest(
            tenant=self.task_store.scope.tenant,
            message=new_text_message(
                f"Execute {skill_id} on {agent_id}",
                role=Role.ROLE_USER,
            ),
            metadata=metadata,
        )
        task = await self.create_task(
            request,
            creator_agent_id=authority_agent_id,
            notify_submitted=False,
            cognition_wake_eligible=not sync,
            input_wake_eligible=False,
        )
        baseline = Task.FromString(task.SerializeToString())
        if denied_reason is not None:
            task.metadata.update({"denied": True})
            task.status.CopyFrom(
                TaskStatus(
                    state=TaskState.TASK_STATE_REJECTED,
                    message=new_text_message(
                        f"Permission denied: {denied_reason}",
                        task_id=task.id,
                        context_id=task.context_id,
                    ),
                )
            )
            task, _ = await self._persist_execution_outcome(
                task,
                baseline=baseline,
                authority_agent_id=authority_agent_id,
                expected_state=TaskState.TASK_STATE_SUBMITTED,
            )
            return task

        if sync:
            # Execute synchronously with transaction safety
            expected_state = task.status.state
            task = await handler.handle_task(task, arguments=args)
            # The canonical outcome writer owns CAS and uncertain-commit
            # reconciliation. A persistence failure must reach the caller; an
            # extra transaction/retry cannot manufacture a completed receipt.
            task, _ = await self._persist_execution_outcome(
                task,
                baseline=baseline,
                authority_agent_id=authority_agent_id,
                expected_state=expected_state,
            )
            # Execute POST_TOOL_USE hooks
            if self.hooks_manager:
                from kestrel_sdk.hooks.base import HookEvent, HookInput

                post_hook_input = HookInput(
                    session_id=session_id,
                    hook_event_name=HookEvent.POST_TOOL_USE.value,
                    tool_name=skill_id,
                    tool_input=args,
                    feature_name=hook_feature_name,
                    tool_response={
                        "task_id": task.id,
                        "state": task_state_label(task.status.state),
                    },
                )
                # Post hooks run in parallel (non-blocking)
                await self.hooks_manager.execute_hooks_parallel(
                    HookEvent.POST_TOOL_USE, post_hook_input
                )

            return task
        else:
            self._track_execution_task(
                self._execute_async(
                    handler,
                    Task.FromString(task.SerializeToString()),
                    authority_agent_id,
                    arguments=args,
                ),
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
        task = create_owned_task(coro, name=f"a2a-task-{task_id}")
        self._execution_tasks.add(task)
        self._execution_authorities[task] = (task_id, authority_agent_id)

        def _discard(done: asyncio.Task[None]) -> None:
            from kestrel_sovereign._async_ownership import owned_task_error

            error = owned_task_error(done)
            if error is not None:
                self._remember_execution_error(error)
                if not isinstance(
                    error, asyncio.CancelledError
                ) or is_execution_control_error(error):
                    logger.error(
                        "Background A2A task %s failed outside its handler",
                        task_id,
                        exc_info=(type(error), error, error.__traceback__),
                    )
            self._execution_tasks.discard(done)
            self._execution_authorities.pop(done, None)

        task.add_done_callback(_discard)
        return task

    def _remember_execution_error(self, error: BaseException) -> None:
        # A normal cooperative worker cancellation is settled by cancel_task.
        # Storage failures and native custody receipts remain lifecycle errors.
        if not isinstance(error, asyncio.CancelledError) or is_execution_control_error(
            error
        ):
            self._execution_control = execution_terminal_error(
                self._execution_control, error
            )

    async def drain_execution_tasks(self, *, cancel: bool = False) -> None:
        """Wait for tracked background executions, optionally cancelling them."""
        tasks = set(self._execution_tasks)
        terminal_errors: list[BaseException] = []
        if cancel:
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
                    except ValueError as error:
                        if is_execution_control_error(error):
                            self._remember_execution_error(error)
                    except asyncio.CancelledError as cancellation:
                        self._remember_execution_error(cancellation)
                        terminal_errors.append(cancellation)
                    except BaseException as error:
                        self._remember_execution_error(error)
                        if not isinstance(
                            error, Exception
                        ) and not is_execution_control_error(error):
                            terminal_errors.append(error)
                        # Shutdown must still stop the in-memory worker when its
                        # durable receipt cannot be persisted. The backend error
                        # is retained for the lifecycle caller, but cannot
                        # strand the worker or skip closing the remaining stores.
                        logger.exception(
                            "Could not persist shutdown cancellation for task %s",
                            task_id,
                        )
                    finally:
                        task.cancel()
                else:
                    task.cancel()

        for task in tasks:
            result = await await_owned_task(task)
            if result.error is not None:
                self._remember_execution_error(result.error)
            if result.cancellation is not None:
                terminal_errors.append(result.cancellation)
        self._execution_tasks.difference_update(tasks)
        terminal = execution_terminal_error(
            self._execution_control,
            *terminal_errors,
        )
        if terminal is not None:
            raise terminal

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
        baseline: Task,
        authority_agent_id: str,
    ) -> Task:
        """Commit handler cancellation or return an earlier terminal winner."""

        try:
            return await self.cancel_task(
                task.id,
                reason=self._cancellation_reason(task),
                agent_name=authority_agent_id,
                task_payload=task,
                task_baseline=baseline,
            )
        except ValueError as cancellation_error:
            if is_execution_control_error(cancellation_error):
                raise
            current = await self.task_store.get_for_recipient(
                task.id,
                authority_agent_id,
            )
            if current is not None and current.status.state in {
                TaskState.TASK_STATE_COMPLETED,
                TaskState.TASK_STATE_FAILED,
                TaskState.TASK_STATE_CANCELED,
                TaskState.TASK_STATE_REJECTED,
            }:
                return current
            raise

    async def _execute_async(
        self,
        handler: TaskHandler,
        task: Task,
        authority_agent_id: str,
        *,
        arguments: dict[str, Any] | None = None,
    ) -> None:
        """Execute a task asynchronously and update the store."""
        expected_state = task.status.state
        baseline = Task.FromString(task.SerializeToString())
        try:
            try:
                task = await handler.handle_task(task, arguments=arguments)
            except Exception as error:
                if is_execution_control_error(error):
                    raise
                logger.exception("Async task handler failed for %s", task.id)
                task.status.CopyFrom(
                    TaskStatus(
                        state=TaskState.TASK_STATE_FAILED,
                        message=new_text_message(
                            str(error), task_id=task.id, context_id=task.context_id
                        ),
                    )
                )
            # Persistence and notification are infrastructure boundaries. Their
            # failure cannot rewrite a successful handler result as FAILED or
            # initiate a second terminal write. The tracked owner retains it.
            task, owns_notification = await self._persist_execution_outcome(
                task,
                baseline=baseline,
                authority_agent_id=authority_agent_id,
                expected_state=expected_state,
            )
            if owns_notification:
                await self._notify_status_update(
                    task,
                    final=task.status.state in TERMINAL_STATES,
                )
        except asyncio.CancelledError as cancellation:
            if is_execution_control_error(cancellation):
                raise
            try:
                await self.cancel_task(
                    task.id,
                    reason="Task canceled during shutdown",
                    agent_name=authority_agent_id,
                )
            except ValueError as cancellation_error:
                if is_execution_control_error(cancellation_error):
                    raise
                # An independently authorized cancellation or other terminal
                # result won the race. Never narrate over that durable state.
            raise

    async def _persist_execution_outcome(
        self,
        task: Task,
        *,
        baseline: Task,
        authority_agent_id: str,
        expected_state: TaskState,
    ) -> tuple[Task, bool]:
        """Persist a worker result and report ownership of terminal notification."""

        if task.status.state == TaskState.TASK_STATE_CANCELED:
            return (
                await self._persist_handler_cancellation(
                    task,
                    baseline=baseline,
                    authority_agent_id=authority_agent_id,
                ),
                False,
            )

        terminal_operation_id = (
            uuid4().hex
            if task.status.state in (TERMINAL_STATES - {TaskState.TASK_STATE_CANCELED})
            else None
        )
        try:
            saved = await self._save_recipient_execution_result(
                task,
                baseline=baseline,
                authority_agent_id=authority_agent_id,
                expected_state=expected_state,
                terminal_operation_id=terminal_operation_id,
            )
        except Exception as save_error:
            # Authority/uncertain-COMMIT control is not an ordinary transport
            # failure. A matching terminal token cannot authorize a retired
            # execution to claim a completion wake or start new readback work.
            if is_execution_control_error(save_error):
                raise
            # PostgreSQL can commit and then lose the COMMIT acknowledgement.
            # Reconcile the recipient-scoped canonical row before treating the
            # exception as a failed write: the private per-attempt token proves
            # this execution still owns the completion wake without comparing
            # model values that JSON persistence may normalize. A different
            # token belongs to another writer and must not be narrated over.
            # Cancellation remains outside this Exception-only recovery so task
            # cancellation semantics are never swallowed.
            if terminal_operation_id is None:
                raise
            try:
                committed_operation_id = (
                    await self.task_store.get_recipient_terminal_operation_id(
                        task.id,
                        authority_agent_id,
                    )
                )
            except asyncio.CancelledError:
                raise
            except Exception as reconciliation_error:
                if is_execution_control_error(reconciliation_error):
                    raise
                raise execution_terminal_error(
                    save_error, reconciliation_error, retain_originals=True
                )
            if committed_operation_id != terminal_operation_id:
                raise

            # A matching token proves commit ownership, not the merged payload.
            # Without a canonical read, retain the infrastructure failure;
            # publishing the attempted snapshot could omit committed work.
            try:
                current = await self.task_store.get_for_recipient(
                    task.id,
                    authority_agent_id,
                )
            except asyncio.CancelledError:
                raise
            except Exception as reconciliation_error:
                if is_execution_control_error(reconciliation_error):
                    raise
                raise execution_terminal_error(
                    save_error, reconciliation_error, retain_originals=True
                )
            if current is None:
                raise execution_terminal_error(
                    save_error,
                    TaskMutationAuthorizationError(
                        f"Committed task could not be read by its recipient: {task.id}"
                    ),
                    retain_originals=True,
                )
            return current, True
        attempted_states: set[TaskState] = set()
        while saved is None:
            # A terminal writer owns the durable result and its notification.
            # A remaining live state may instead be progress published by this
            # handler while it ran. Reconcile only a transition that is valid
            # from that exact current state, and retain the CAS on every retry.
            current = await self.task_store.get_for_recipient(
                task.id,
                authority_agent_id,
            )
            if current is None:
                raise TaskMutationAuthorizationError(
                    f"Task outcome was not authorized or task was not found: {task.id}"
                )
            current_state = current.status.state
            if current_state in {
                TaskState.TASK_STATE_COMPLETED,
                TaskState.TASK_STATE_FAILED,
                TaskState.TASK_STATE_CANCELED,
                TaskState.TASK_STATE_REJECTED,
            }:
                return current, False
            if current_state == task.status.state:
                return current, False
            if (
                task.status.state not in VALID_TRANSITIONS.get(current_state, set())
                or current_state in attempted_states
            ):
                return current, False
            attempted_states.add(current_state)
            saved = await self._save_recipient_execution_result(
                task,
                baseline=baseline,
                authority_agent_id=authority_agent_id,
                expected_state=current_state,
                terminal_operation_id=terminal_operation_id,
            )
        return saved, True

    def _refuse_command_authored_self_followup(
        self, skill_id: str, user_input: str
    ) -> dict | None:
        """Refuse a chat-typed ``self_followup`` schedule (#3112 P1).

        Returns a failure dict when the command would create a
        ``self_followup`` row, else ``None``.

        Deliberately does NOT reuse ``parse_command_args``: that binder is
        strictly positional, so a JSON ``args_json`` containing spaces shreds
        across later parameters (#3118) and ``task_name`` cannot be trusted to
        land in ``task_name``. This scans the raw text for the task name
        instead, which is the one token whose presence is unambiguous no matter
        where the positional binder puts it -- but matched as an exact TOKEN
        rather than a substring, because over-matching was not in fact free:
        it refused unrelated schedules that merely mentioned the name inside
        another argument's value. Under-matching would admit the row this
        guard exists to keep out.
        """
        from kestrel_sovereign.signals.sources.self_followup import (
            TASK_NAME as SELF_FOLLOWUP_TASK_NAME,
        )

        if not skill_id.startswith("schedule_"):
            return None
        # Exact-TOKEN, not substring (#3112 gate-2 P2). A bare substring scan
        # refused unrelated schedules that merely mention the name inside
        # another value -- e.g.
        #   !schedule add @hourly github_pr_watch {"repo":"owner/self_followup"}
        # Tokenizing keeps the #3118 robustness the substring scan was chosen
        # for -- we still never trust the positional binder to put task_name in
        # task_name -- while dropping the over-refusal: to create the row, the
        # name must bind to the task_name PARAMETER, which means it has to
        # appear as its own whitespace-delimited token (or as an explicit
        # task_name=<name>). Embedded in a larger JSON token it cannot bind,
        # so refusing it protected nothing and cost a legitimate schedule.
        tokens = user_input.split()
        if not any(
            token == SELF_FOLLOWUP_TASK_NAME
            or token == f"task_name={SELF_FOLLOWUP_TASK_NAME}"
            for token in tokens
        ):
            return None
        return {
            "success": False,
            "error": (
                f"'{SELF_FOLLOWUP_TASK_NAME}' cannot be scheduled from a chat "
                "command. It carries THIS agent's own intention across its own "
                "turn boundary and its signal source is registered TRUSTED on "
                "exactly that ground; text typed at the command surface is "
                "author-by-a-human, so accepting it would wake a full "
                "cognition turn at a trust level its registration does not "
                "describe. Ask the agent in chat to schedule its own "
                "follow-up instead."
            ),
            "refused": "command_authored_self_followup",
        }

    async def _save_recipient_execution_result(
        self,
        task: Task,
        *,
        baseline: Task,
        authority_agent_id: str,
        expected_state: TaskState,
        terminal_operation_id: str | None,
    ) -> Task | None:
        """Persist a handler result without mistaking live progress for a winner."""

        if task.status.state in (TERMINAL_STATES - {TaskState.TASK_STATE_CANCELED}):
            if terminal_operation_id is None:
                raise ValueError("Terminal execution outcome requires an operation ID")
            return await self.task_store.save_recipient_terminal_outcome(
                task,
                baseline=baseline,
                recipient_agent_id=authority_agent_id,
                operation_id=terminal_operation_id,
            )
        return await self.task_store.save_recipient_lifecycle(
            task,
            baseline=baseline,
            recipient_agent_id=authority_agent_id,
            expected_state=expected_state,
        )

    async def execute_command(self, user_input: str) -> dict | None:
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

        # #3112 P1: reaching THIS function is the only authorship fact core
        # has. Both callers (`endpoints/agent.py`, `command_handler.py`) are
        # chat command surfaces: the text was typed by a human, never composed
        # by the model. Everything downstream — `execute_skill`, the scheduler
        # guards — is shared with A2A routing and the in-turn tool loop, so by
        # the time args reach `_prepare_self_followup_args` that distinction is
        # gone and every guard there is reduced to inferring it. The chain of
        # proxies (session-truthiness -> turn-ownership) ran out here.
        #
        # `self_followup` exists so the agent can carry ITS OWN intention
        # across ITS OWN turn boundary at Trust.TRUSTED. A human asking for
        # later work is a different request with a different trust story, and
        # `SourceRegistration.trust` is static per source so it cannot be
        # downgraded per row. Refuse the pairing outright rather than accept
        # it at a trust level the registration comment does not describe.
        refusal = self._refuse_command_authored_self_followup(skill_id, user_input)
        if refusal is not None:
            return refusal

        # Parse command arguments using the tool's parser if available
        args = {}
        _, handler = self._providers.get(agent_id, (None, None))
        if handler:
            # Try to get the tool from the handler to use its parse_command_args
            tool = (
                handler._get_tool_by_name(skill_id)
                if hasattr(handler, "_get_tool_by_name")
                else None
            )
            if tool and hasattr(tool, "parse_command_args"):
                args = tool.parse_command_args(user_input)
            else:
                args = self._parse_command_args(user_input)
        else:
            args = self._parse_command_args(user_input)

        # Mark the tool as typed for the duration of its call, so what it
        # writes is attributed to the caller rather than to the agent (#2652).
        from kestrel_sovereign.llm.provider_policy import command_authored_scope

        try:
            with command_authored_scope(skill_id):
                task = await self.execute_skill(
                    agent_id=agent_id,
                    skill_id=skill_id,
                    args=args,
                    sync=True,
                )

            # Extract result from task artifacts
            if task.status.state == TaskState.TASK_STATE_COMPLETED and task.artifacts:
                artifact = task.artifacts[0]
                if artifact.parts:
                    part = artifact.parts[0]
                    recognized, value = embedded_json_value(part)
                    if recognized:
                        return value
                    if part.HasField("text"):
                        return {"success": True, "result": part.text}

            if task.status.state in {
                TaskState.TASK_STATE_FAILED,
                TaskState.TASK_STATE_REJECTED,
            }:
                error_msg = "Task failed"
                if task.status.message and task.status.message.parts:
                    for p in task.status.message.parts:
                        if p.HasField("text"):
                            error_msg = p.text
                            break
                return {"success": False, "error": error_msg}

            return {"success": True, "result": "Task completed"}

        except Exception as e:
            if is_execution_control_error(e):
                raise
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

    def session_id_for_context(self, context_id: str) -> str:
        """Namespace an opaque peer context away from host chat session IDs."""
        return self.task_store.scope.session_id_for_context(context_id)

    def _announce_submitted_input(self, candidate: NativeTaskSnapshot) -> None:
        """Publish a committed input snapshot through the canonical owner hook."""
        candidate.require_recipient(self.did)
        if (
            not candidate.cognition_wake_eligible
            or not candidate.input_wake_eligible
            or self._on_task_submitted is None
        ):
            return
        try:
            self._on_task_submitted(candidate)
        except Exception as error:
            if is_execution_control_error(error):
                raise
            logger.warning(
                "Task input admission callback failed for %s",
                candidate.task.id,
                exc_info=True,
            )

    async def continue_task(
        self, params: SendMessageRequest, *, creator_agent_id: str
    ) -> Task | None:
        """Admit a follow-up on the same creator-owned task and native context."""
        if not isinstance(params, SendMessageRequest):
            raise TypeError("Task continuation requires an official SendMessageRequest")
        validate_proto_required_fields(params)
        parse_message_metadata(MessageToDict(params.metadata))
        candidate, appended = await self.task_store.append_creator_input(
            params, creator_agent_id=creator_agent_id
        )
        if candidate is None:
            return None
        if appended:
            self._announce_submitted_input(candidate)
            try:
                await self._notify_status_update(candidate.task, final=False)
            except Exception as error:
                if is_execution_control_error(error):
                    raise
                logger.warning(
                    "Admitted task input subscriber projection failed for %s",
                    candidate.task.id,
                    exc_info=True,
                )
        return candidate.task

    async def create_task(
        self,
        params: SendMessageRequest,
        *,
        creator_agent_id: str,
        notify_submitted: bool = True,
        cognition_wake_eligible: bool = True,
        input_wake_eligible: bool = True,
    ) -> Task:
        """Admit a new native task after sender and recipient authorization.

        The caller supplies a verified creator, never metadata-derived identity.
        A continuation must first authorize its existing task through the native
        handler; this new-task path does not honor peer-chosen task IDs.
        """
        if not isinstance(params, SendMessageRequest):
            raise TypeError(
                "Task admission requires an official A2A SendMessageRequest"
            )
        validate_proto_required_fields(params)
        metadata = MessageToDict(params.metadata)
        try:
            parse_message_metadata(metadata)
        except ValueError as error:
            raise InvalidParamsError(message=str(error)) from error
        if not isinstance(creator_agent_id, str) or not creator_agent_id.strip():
            raise ValueError("Task admission requires a verified creator identity")
        if params.tenant != self.task_store.scope.tenant:
            raise InvalidParamsError(
                message="A2A tenant does not match the selected recipient"
            )
        if params.message.task_id:
            raise InvalidParamsError(
                message="New task admission cannot select an existing task ID"
            )
        if params.message.role != Role.ROLE_USER:
            raise InvalidParamsError(message="Task admission requires a user message")
        task = new_task_from_user_message(params.message)
        task.metadata.update(
            without_reserved_cancellation_receipt(MessageToDict(params.metadata))
        )
        # Reserve ownership and task in ONE backend transaction. A failed insert
        # cannot strand a context reservation or authorize another creator.
        transaction = self.task_store.write_transaction()
        ownership_refusal = None
        admitted_snapshot = None
        async with transaction:
            try:
                await self.context_store.reserve(
                    task.context_id,
                    self.task_store.scope,
                    creator_agent_id=creator_agent_id,
                )
            except ContextOwnershipError as error:
                # A conflicting reservation changes no row. Raise the domain
                # refusal outside the backend's infrastructure-error wrapper.
                if is_execution_control_error(error):
                    raise
                ownership_refusal = error
            else:
                await self.task_store.create(
                    task,
                    creator_agent_id=creator_agent_id,
                    recipient_agent_id=self.task_store.scope.recipient_id,
                    cognition_wake_eligible=cognition_wake_eligible,
                    input_wake_eligible=input_wake_eligible,
                )
                await self.task_store.record_initial_input(
                    task,
                    params,
                    creator_agent_id=creator_agent_id,
                )
                admitted_snapshot = await self.task_store.get_task_snapshot(
                    task.id,
                    recipient_agent_id=self.did,
                )
                if admitted_snapshot is None:
                    raise RuntimeError("New task has no canonical input snapshot")
        if ownership_refusal is not None:
            raise ownership_refusal
        projection_id = self.session_id_for_context(task.context_id)
        # Admission is complete only when the durable reservation can wake its
        # recipient.  Fire the synchronous bridge immediately after commit,
        # before any cancellable session/observability await.  The bridge owns
        # its asynchronous delivery through the agent's background-task
        # tracker, so cancellation of this transport coroutine cannot strand a
        # durably SUBMITTED row with no cognition wake.
        if notify_submitted:
            self._announce_submitted_input(admitted_snapshot)

        # The insert-only task reservation is deliberately first: a duplicate
        # caller-supplied ID must be rejected before it can append another user
        # message to the existing session. Session and observability projections
        # are best-effort after the durable task commit; their failure cannot
        # turn an accepted task into a 500 that a retry experiences as a 409.
        try:
            session = await self.session_service.get_session(projection_id)
            if not session:
                await self.session_service.create_session(
                    session_id=projection_id,
                    agent_name=self.did,
                    metadata={"a2a_context_id": task.context_id},
                )
            await self.session_service.append_event(
                session_id=projection_id,
                event_type="user_message",
                data=MessageToDict(params.message),
            )
        except Exception as execution_error:
            if is_execution_control_error(execution_error):
                raise
            logger.warning(
                "Task %s was created but its session projection failed",
                task.id,
                exc_info=True,
            )

        try:
            await self.observability_store.log_tool_call(
                agent_name=self.did,
                tool_name="task_create",
                session_id=projection_id,
                metadata={"task_id": task.id},
            )
        except Exception as execution_error:
            if is_execution_control_error(execution_error):
                raise
            logger.warning(
                "Task %s was created but observability logging failed",
                task.id,
                exc_info=True,
            )

        # A recipient wake can race these best-effort projections and move the
        # canonical row before create_task returns. Never publish an older
        # SUBMITTED snapshot after a terminal transition already committed.
        canonical_read_succeeded = False
        canonical_read = asyncio.create_task(
            self.task_store._get_unscoped(task.id),
            name=f"a2a-task-create-readback:{task.id}",
        )
        readback_outcome = await await_owned_task(canonical_read)
        if readback_outcome.cancellation is not None:
            raise_owned_outcome(
                readback_outcome,
                operation="A2A task creation canonical readback",
            )
        try:
            canonical = raise_owned_outcome(
                readback_outcome,
                operation="A2A task creation canonical readback",
            )
            canonical_read_succeeded = canonical is not None
        except (Exception, asyncio.CancelledError) as readback_error:
            if is_execution_control_error(readback_error):
                raise
            canonical = None
            logger.warning(
                "Task %s was accepted but its canonical readback failed",
                task.id,
                exc_info=True,
            )
        if canonical is None:
            # The insert-only durable reservation and recipient wake already
            # committed. Return the accepted snapshot so a transient read
            # outage cannot make the sender abandon work the recipient owns.
            # Suppress the SUBMITTED SSE projection because a raced recipient
            # may already have moved the unreadable canonical row terminal.
            canonical = task
        if (
            canonical_read_succeeded
            and canonical.status.state == TaskState.TASK_STATE_SUBMITTED
        ):
            await self._notify_status_update(canonical, final=False)

        logger.info(
            "Native A2A task created: %s in context %s", task.id, task.context_id
        )
        return canonical

    async def update_status(
        self,
        task_id: str,
        new_state: TaskState,
        message: Message | None = None,
        agent_name: str | None = None,
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
        if new_state == TaskState.TASK_STATE_CANCELED:
            raise TaskCancellationAuthorizationError(
                "CANCELED is an authorized transition; use cancel_task"
            )

        task = await self.task_store.get_for_recipient(
            task_id,
            recipient_agent_id,
        )
        if task is None:
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
        baseline = Task.FromString(task.SerializeToString())
        if new_state in TERMINAL_STATES and message is not None:
            reply_metadata = MessageToDict(message.metadata)
            answer_metadata = correlated_answer_metadata(
                MessageToDict(task.metadata), reply_metadata
            )
            if (
                parse_question_correlation(answer_metadata) is not None
                and message.role != Role.ROLE_AGENT
            ):
                raise InvalidParamsError(
                    message="A question answer must have the agent role"
                )
            if answer_metadata != reply_metadata:
                message = Message.FromString(message.SerializeToString())
                message.metadata.update(answer_metadata)
        task.status.CopyFrom(TaskStatus(state=new_state, message=message))

        # Add message to history if provided
        if message is not None:
            task.history.append(message)

        # Save updated task (use save() to persist both status and history)
        saved = await self.task_store.save_recipient_lifecycle(
            task,
            baseline=baseline,
            recipient_agent_id=recipient_agent_id,
            expected_state=current_state,
        )
        if saved is None:
            persisted = await self.task_store.get_for_recipient(
                task_id,
                recipient_agent_id,
            )
            state = (
                persisted.status.state
                if persisted is not None
                else TaskState.TASK_STATE_UNSPECIFIED
            )
            raise ValueError(f"Invalid state transition: task is already {state}")
        task = saved

        # Determine if this is a final state
        is_final = new_state in TERMINAL_STATES
        await self._finish_status_projection(
            task,
            old_state=task_state_label(current_state),
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
        agent_name: str | None = None,
        *,
        recipient_agent_id: str,
        append: bool = False,
        last_chunk: bool = True,
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
        if not isinstance(append, bool) or not isinstance(last_chunk, bool):
            raise TypeError("Artifact append and last_chunk must be booleans")
        task = await self.task_store.get_for_recipient(
            task_id,
            recipient_agent_id,
        )
        if task is None:
            raise TaskMutationAuthorizationError(
                f"Task mutation was not authorized or task was not found: {task_id}"
            )

        # Add artifact
        await self.task_store.add_artifact(
            task_id,
            artifact,
            recipient_agent_id=recipient_agent_id,
            append=append,
        )

        # Refresh task
        task = await self.task_store.get_for_recipient(
            task_id,
            recipient_agent_id,
        )

        # Notify subscribers
        await self._notify_artifact_update(
            task, artifact, append=append, last_chunk=last_chunk
        )

        logger.info(f"Artifact added to task {task_id}: {artifact.name}")
        return task

    async def get_task_for_recipient(
        self,
        task_id: str,
        recipient_agent_id: str,
    ) -> Task | None:
        """Get one inbox task through its durable recipient principal."""
        return await self.task_store.get_for_recipient(
            task_id,
            recipient_agent_id,
        )

    async def get_task_snapshot(
        self,
        task_id: str,
        *,
        recipient_agent_id: str,
    ) -> NativeTaskSnapshot | None:
        """Recover task payload and admission ownership through one scoped read."""
        return await self.task_store.get_task_snapshot(
            task_id,
            recipient_agent_id=recipient_agent_id,
        )

    async def get_task_for_creator(
        self,
        task_id: str,
        creator_agent_id: str,
        *,
        recipient_agent_id: str,
    ) -> Task | None:
        """Get one outbound result through its creator and routed recipient."""
        return await self.task_store.get_for_creator(
            task_id,
            creator_agent_id,
            recipient_agent_id=recipient_agent_id,
        )

    async def get_task_cancellation_snapshot(
        self,
        task_id: str,
        *,
        recipient_agent_id: str,
    ) -> TaskCancellationSnapshot | None:
        """Read the minimal durable state used to withdraw live cognition."""

        return await self.task_store.get_cancellation_snapshot(
            task_id,
            recipient_agent_id=recipient_agent_id,
        )

    async def is_task_recipient(self, task_id: str, agent_id: str) -> bool:
        """Whether this manager's durable task row delegates execution to agent."""

        return await self.task_store.is_task_recipient(task_id, agent_id)

    async def get_context_tasks(
        self,
        context_id: str,
        *,
        recipient_agent_id: str,
        limit: int = 100,
    ) -> list[Task]:
        """Get all tasks in a session."""
        return await self.task_store.list_tasks(
            recipient_agent_id=recipient_agent_id,
            context_id=context_id,
            limit=limit,
        )

    async def get_pending_tasks(
        self,
        *,
        recipient_agent_id: str,
        limit: int = 100,
    ) -> list[Task]:
        """Get all pending (submitted) tasks ready for processing."""
        return await self.task_store.get_pending_tasks(
            recipient_agent_id=recipient_agent_id,
            limit=limit,
        )

    async def list_tasks(
        self,
        *,
        recipient_agent_id: str,
        context_id: str | None = None,
        user_id: str | None = None,
        status: TaskState | None = None,
        limit: int = 100,
    ) -> list[Task]:
        """List tasks with optional filters, across ALL states.

        Unlike ``get_pending_tasks`` (which only returns SUBMITTED tasks),
        this passes through to the store's full-table query so callers can
        filter by any ``TaskState`` (completed, failed, working, canceled).
        """
        return await self.task_store.list_tasks(
            recipient_agent_id=recipient_agent_id,
            context_id=context_id,
            user_id=user_id,
            status=status,
            limit=limit,
        )

    async def list_task_snapshots(
        self,
        *,
        recipient_agent_id: str,
        context_id: str | None = None,
        user_id: str | None = None,
        status: TaskState | None = None,
        limit: int = 100,
    ) -> list[NativeTaskSnapshot]:
        """List native task values with the same durable ownership as wakes."""
        return await self.task_store.list_task_snapshots(
            recipient_agent_id=recipient_agent_id,
            context_id=context_id,
            user_id=user_id,
            status=status,
            limit=limit,
        )

    async def list_cognition_wake_candidates(
        self,
        *,
        recipient_agent_id: str,
        live_changed_since: datetime,
        terminal_changed_since: datetime,
    ) -> list[NativeTaskSnapshot]:
        """Return task rows that can reconstruct a missing one-shot wake."""

        return await self.task_store.list_cognition_wake_candidates(
            recipient_agent_id=recipient_agent_id,
            live_changed_since=live_changed_since,
            terminal_changed_since=terminal_changed_since,
        )

    async def cancel_task(
        self,
        task_id: str,
        reason: str | None = None,
        agent_name: str | None = None,
        recipient_agent_id: str | None = None,
        task_payload: Task | None = None,
        task_baseline: Task | None = None,
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
        operation_id = uuid4().hex
        cancel_kwargs["operation_id"] = operation_id
        if recipient_agent_id is not None:
            cancel_kwargs["expected_recipient_agent_id"] = recipient_agent_id
        if task_payload is not None:
            cancel_kwargs["task_payload"] = task_payload
            cancel_kwargs["task_baseline"] = task_baseline
        elif task_baseline is not None:
            raise ValueError("Cancellation baseline requires a handler payload")
        rollback_local_intent = None
        if self._on_task_cancellation_started is not None:
            rollback_local_intent = self._on_task_cancellation_started(
                task_id,
                agent_name,
            )
        store_failure: BaseException | None = None

        async def is_this_actor_receipt(candidate: Task) -> bool:
            receipt = (
                MessageToDict(candidate.metadata).get("cancellation_receipt") or {}
            )
            return (
                candidate.status.state == TaskState.TASK_STATE_CANCELED
                and receipt.get("actor_agent_id") == agent_name
                and (
                    recipient_agent_id is None
                    or await self.task_store.is_task_recipient(
                        task_id,
                        recipient_agent_id,
                    )
                )
            )

        async def reconciliation_read() -> Task | None:
            """Read only through the authority predicate used by this route."""

            if recipient_agent_id is not None:
                if agent_name == recipient_agent_id:
                    return await self.task_store.get_for_recipient(
                        task_id,
                        recipient_agent_id,
                    )
                return await self.task_store.get_for_creator(
                    task_id,
                    agent_name,
                    recipient_agent_id=recipient_agent_id,
                )
            return await self.task_store.get_for_principal(
                task_id,
                agent_name,
            )

        try:
            task = await self.task_store.cancel_if_authorized(
                task_id,
                **cancel_kwargs,
            )
        except BaseException as error:
            # PostgreSQL can commit and then lose the COMMIT acknowledgement.
            # Re-read before declaring this a pre-commit failure: the task that
            # owns the durable receipt must still cancel its queued wake and
            # finish session/SSE/memory/completion projections.  Preserve the
            # original store outcome only after that reconciliation completes.
            store_failure = error
            try:
                current = await reconciliation_read()
                committed_here = (
                    current is not None
                    and await is_this_actor_receipt(current)
                    and await self.task_store.get_cancellation_operation_id(task_id)
                    == operation_id
                )
            except BaseException as reconciliation_failure:  # noqa: BLE001 - preserve both native outcomes before re-raising
                if rollback_local_intent is not None:
                    rollback_local_intent()
                raise execution_terminal_error(error, reconciliation_failure)
            if not committed_here:
                if rollback_local_intent is not None:
                    rollback_local_intent()
                raise
            task = current
        if task is None:
            try:
                current = await reconciliation_read()
            except BaseException:
                if rollback_local_intent is not None:
                    rollback_local_intent()
                raise
            if current is None:
                if rollback_local_intent is not None:
                    rollback_local_intent()
                raise TaskCancellationAuthorizationError(
                    "Task cancellation was not authorized or task was not found: "
                    f"{task_id}"
                )
            if await is_this_actor_receipt(current):
                # The first atomic cancellation owns every derived projection.
                # Its cancellation-safe projection join completes those side
                # effects before propagating a lost transport response, so a
                # retry returns only the canonical receipt and must not append
                # duplicate memory/session rows or completion wakes.
                # Keep the just-acquired local execution exemption: this retry
                # is still the same authorized recipient decline and its
                # monitor must not cancel the response after seeing CANCELED.
                return current
            if rollback_local_intent is not None:
                rollback_local_intent()
            if current.status.state not in {
                TaskState.TASK_STATE_SUBMITTED,
                TaskState.TASK_STATE_WORKING,
                TaskState.TASK_STATE_INPUT_REQUIRED,
                TaskState.TASK_STATE_AUTH_REQUIRED,
            }:
                raise ValueError(
                    f"Invalid state transition: {current.status.state} -> "
                    f"{TaskState.TASK_STATE_CANCELED}"
                )
            raise TaskCancellationAuthorizationError(
                f"Agent {agent_name!r} is not authorized to cancel task {task_id!r}"
            )

        try:
            receipt = MessageToDict(task.metadata).get("cancellation_receipt") or {}
            previous_state = receipt.get("status_before")
            if self._on_task_cancelled is not None:
                try:
                    self._on_task_cancelled(task)
                except Exception as exc:
                    if is_execution_control_error(exc):
                        raise
                    logger.warning(
                        "on_task_cancelled callback failed for %s: %s",
                        task.id,
                        exc,
                        exc_info=True,
                    )
            await self._finish_status_projection(
                task,
                old_state=str(previous_state or "unspecified"),
                new_state=TaskState.TASK_STATE_CANCELED,
                agent_name=agent_name,
                is_final=True,
                reason=reason,
            )
        except BaseException as projection_failure:  # noqa: BLE001 - preserve the original commit receipt before re-raising
            # Exact actor/operation readback proves which CAS committed, not
            # that its UNKNOWN transport outcome may be discarded. Retain the
            # FIRST native receipt through every later callback/projection.
            raise execution_terminal_error(
                store_failure, projection_failure, retain_originals=True
            )

        logger.info(
            "Task %s canceled by %s: %s -> %s",
            task_id,
            agent_name,
            previous_state,
            task_state_label(TaskState.TASK_STATE_CANCELED),
        )
        if store_failure is not None:
            raise store_failure
        return task

    async def _finish_status_projection(
        self,
        task: Task,
        *,
        old_state: str,
        new_state: TaskState,
        agent_name: str | None,
        is_final: bool,
        reason: str | None = None,
    ) -> None:
        """Join post-commit projections before propagating caller cancellation."""

        projection = create_owned_task(
            self._project_status_transition(
                task,
                old_state=old_state,
                new_state=new_state,
                agent_name=agent_name,
                is_final=is_final,
                reason=reason,
            ),
            name="a2a-task-status-projection",
        )
        outcome = await await_owned_task(projection)
        raise_owned_outcome(
            outcome,
            operation="A2A task status projection",
        )

    async def fail_task(
        self,
        task_id: str,
        error: str,
        agent_name: str | None = None,
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
        message = new_text_message(f"Task failed: {error}")

        task = await self.update_status(
            task_id=task_id,
            new_state=TaskState.TASK_STATE_FAILED,
            message=message,
            agent_name=agent_name,
            recipient_agent_id=recipient_agent_id,
        )

        # Failure telemetry is a projection of the recipient-authorized durable
        # transition.  Emitting it before update_status would let a rejected
        # caller write attacker-authored data tied to another task's session.
        if agent_name:
            try:
                await self.observability_store.log_error(
                    agent_name=self.did,
                    error_type="task_failure",
                    error_message=error,
                    session_id=self.session_id_for_context(task.context_id),
                    metadata={"task_id": task_id, "reported_by": agent_name},
                )
            except Exception as execution_error:
                if is_execution_control_error(execution_error):
                    raise
                logger.warning(
                    "Task %s failed durably but failure observability logging failed",
                    task_id,
                    exc_info=True,
                )

        return task

    async def complete_task(
        self,
        task_id: str,
        response: str,
        agent_name: str | None = None,
        artifacts: list[Artifact] | None = None,
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

        message = new_text_message(response)

        return await self.update_status(
            task_id=task_id,
            new_state=TaskState.TASK_STATE_COMPLETED,
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
        agent_name: str | None,
        is_final: bool,
        reason: str | None = None,
    ) -> None:
        """Best-effort projections after the durable task row has committed.

        Session history, observability, SSE, and memory are projections of the
        canonical task row.  None may turn a committed transition into an
        apparent failure that a retry then experiences as an invalid state.
        """

        event_data: dict[str, Any] = {
            "task_id": task.id,
            "old_state": old_state,
            "new_state": task_state_label(new_state),
        }
        if new_state == TaskState.TASK_STATE_CANCELED:
            event_data.update(
                actor_agent_id=agent_name,
                reason=reason,
            )
        if task.context_id:
            try:
                await self.session_service.append_event(
                    session_id=self.session_id_for_context(task.context_id),
                    event_type="status_update",
                    data=event_data,
                )
            except Exception as execution_error:
                if is_execution_control_error(execution_error):
                    raise
                logger.warning(
                    "Task %s transitioned durably but its session projection failed",
                    task.id,
                    exc_info=True,
                )

        if agent_name:
            metadata: dict[str, Any] = {
                "task_id": task.id,
                "state_transition": f"{old_state} -> {task_state_label(new_state)}",
                "reported_by": agent_name,
            }
            if new_state == TaskState.TASK_STATE_CANCELED:
                metadata["cancellation_reason"] = reason
            try:
                await self.observability_store.log_agent_response(
                    agent_name=self.did,
                    duration_ms=0,
                    session_id=self.session_id_for_context(task.context_id),
                    metadata=metadata,
                )
            except Exception as execution_error:
                if is_execution_control_error(execution_error):
                    raise
                logger.warning(
                    "Task %s transitioned durably but observability logging failed",
                    task.id,
                    exc_info=True,
                )

        try:
            await self._notify_status_update(task, final=is_final)
        except Exception as execution_error:
            if is_execution_control_error(execution_error):
                raise
            logger.warning(
                "Task %s transitioned durably but subscriber notification failed",
                task.id,
                exc_info=True,
            )
        if is_final and self.memory_service and task.context_id:
            try:
                await self._save_to_memory(task)
            except Exception as execution_error:
                if is_execution_control_error(execution_error):
                    raise
                logger.warning(
                    "Task %s transitioned durably but memory recording failed",
                    task.id,
                    exc_info=True,
                )

    # =========================================================================
    # SSE Streaming
    # =========================================================================

    async def subscribe(
        self,
        task_id: str,
        *,
        creator_agent_id: str | None = None,
        recipient_agent_id: str | None = None,
    ) -> AsyncGenerator[Task | TaskStatusUpdateEvent | TaskArtifactUpdateEvent, None]:
        """Yield a native task snapshot followed by native SDK event values."""
        if recipient_agent_id is None:
            raise ValueError("Task subscription requires the routed recipient identity")

        async def read_authorized():
            if creator_agent_id is not None:
                return await self.task_store.get_for_creator(
                    task_id,
                    creator_agent_id,
                    recipient_agent_id=recipient_agent_id,
                )
            return await self.task_store.get_for_recipient(task_id, recipient_agent_id)

        if await read_authorized() is None:
            return
        queue: asyncio.Queue[TaskStatusUpdateEvent | TaskArtifactUpdateEvent] = (
            asyncio.Queue()
        )
        self._subscribers.setdefault(task_id, []).append(queue)
        try:
            # Registration precedes the second read so a racing transition is
            # either in this snapshot or in the queue, never lost between them.
            task = await read_authorized()
            if task is None:
                return
            yield task
            if task.status.state in STREAM_STOP_STATES:
                return
            while True:
                event = await queue.get()
                yield event
                if (
                    isinstance(event, TaskStatusUpdateEvent)
                    and event.status.state in STREAM_STOP_STATES
                ):
                    return
        finally:
            self._subscribers[task_id].remove(queue)
            if not self._subscribers[task_id]:
                del self._subscribers[task_id]

    async def _notify_status_update(self, task: Task, final: bool) -> None:
        """Publish the durable native status; no private final flag crosses wire."""
        if final and self._on_task_complete is not None:
            candidate = await self.task_store.get_task_snapshot(
                task.id,
                recipient_agent_id=self.did,
            )
            if candidate is None:
                raise RuntimeError(
                    "Committed task completion has no authoritative wake snapshot"
                )
            if candidate.cognition_wake_eligible:
                try:
                    self._on_task_complete(candidate)
                except Exception as error:
                    if is_execution_control_error(error):
                        raise
                    logger.exception("Task completion callback failed for %s", task.id)
        # A handler can return artifacts in its final Task without calling the
        # incremental artifact API. Publish those durable values before the
        # status that closes the stream. Replacements use the standard ID rule
        # so a client that already saw chunks does not append them twice.
        for artifact in task.artifacts:
            await self._notify_artifact_update(
                task, artifact, append=False, last_chunk=final
            )
        event = TaskStatusUpdateEvent(
            task_id=task.id, context_id=task.context_id, status=task.status
        )
        for queue in self._subscribers.get(task.id, ()):
            await queue.put(event)

    async def _notify_artifact_update(
        self,
        task: Task,
        artifact: Artifact,
        *,
        append: bool = False,
        last_chunk: bool = True,
    ) -> None:
        """Publish a native artifact update in the task's context."""
        event = TaskArtifactUpdateEvent(
            task_id=task.id,
            context_id=task.context_id,
            artifact=artifact,
            append=append,
            last_chunk=last_chunk,
        )
        for queue in self._subscribers.get(task.id, ()):
            await queue.put(event)

    # =========================================================================
    # Memory Integration
    # =========================================================================

    async def _save_to_memory(self, task: Task) -> None:
        """Save completed task conversation to long-term memory."""
        if not self.memory_service or not task.context_id:
            return

        # Build content from history
        if task.history:
            content_parts = []
            for msg in task.history:
                role = Role.Name(msg.role)
                for part in msg.parts:
                    if part.HasField("text"):
                        content_parts.append(f"{role}: {part.text}")

            content = "\n".join(content_parts)

            await self.memory_service.add_memory(
                session_id=self.session_id_for_context(task.context_id),
                content=content,
                tags=["task", task_state_label(task.status.state)],
                metadata={
                    "task_id": task.id,
                    "completed_at": datetime.now(UTC).isoformat(),
                },
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


async def create_task_manager(
    db_path: str,
    *,
    scope: NativeRecipientScope,
    include_memory: bool = True,
    include_feedback: bool = True,
) -> TaskManager:
    """Create native task admission on one owned SQLite connection."""
    from kestrel_sovereign.storage.db.sqlite import SQLiteBackend

    backend = SQLiteBackend(db_path)
    await backend.connect()
    manager = TaskManager.from_backend(
        backend,
        scope=scope,
        owns_backend=True,
        include_memory=include_memory,
        include_feedback=include_feedback,
    )
    try:
        await manager.initialize()
    except BaseException:
        await backend.close()
        raise
    return manager
