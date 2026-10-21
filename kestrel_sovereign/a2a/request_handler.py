"""Official SDK operations over Kestrel's canonical native task lifecycle."""

import asyncio
from collections.abc import AsyncGenerator

from a2a.server.context import ServerCallContext
from a2a.server.events.event_queue import Event
from a2a.server.request_handlers import RequestHandler
from a2a.types import (
    AgentCard,
    CancelTaskRequest,
    DeleteTaskPushNotificationConfigRequest,
    GetExtendedAgentCardRequest,
    GetTaskPushNotificationConfigRequest,
    GetTaskRequest,
    ListTaskPushNotificationConfigsRequest,
    ListTaskPushNotificationConfigsResponse,
    ListTasksRequest,
    ListTasksResponse,
    SendMessageRequest,
    SubscribeToTaskRequest,
    Task,
    TaskPushNotificationConfig,
)
from a2a.utils.errors import (
    ContentTypeNotSupportedError,
    InvalidAgentResponseError,
    InvalidParamsError,
    PushNotificationNotSupportedError,
    TaskNotCancelableError,
    TaskNotFoundError,
    UnsupportedOperationError,
)
from a2a.utils.proto_utils import validate_proto_required_fields
from a2a.utils.task import apply_history_length, validate_history_length
from google.protobuf.json_format import MessageToDict
from starlette.exceptions import HTTPException

from kestrel_sovereign._async_ownership import gather_owned
from kestrel_sovereign.a2a.extensions import (
    parse_message_metadata,
    require_metadata_extensions,
)
from kestrel_sovereign.a2a.lifecycle import TERMINAL_STATES
from kestrel_sovereign.a2a.media import validate_input_parts
from kestrel_sovereign.a2a.native_context import wait_native_disconnect
from kestrel_sovereign.a2a.native_security import (
    AuthorizedNativeRequest,
    NativeDIDRequestGuard,
)
from kestrel_sovereign.a2a.stores.unified.context_store import ContextOwnershipError
from kestrel_sovereign.a2a.task_manager import (
    TaskCancellationAuthorizationError,
    TaskManager,
)
from kestrel_sovereign.execution_custody import (
    close_native_execution_stream,
    is_execution_control_error,
    owned_execution_stream,
)


class NativeA2ARequestHandler(RequestHandler):
    """Both SDK HTTP bindings share one recipient, guard and task manager.

    Incoming messages start independently tracked tasks in their native context.
    A caller-supplied task ID always refers to an existing accessible task;
    task creation never accepts a client-selected identifier.
    """

    def __init__(
        self, manager: TaskManager, guard: NativeDIDRequestGuard, card: AgentCard
    ):
        if manager.task_store.scope != guard.scope:
            raise ValueError(
                "Native handler and task manager require the same recipient scope"
            )
        if not isinstance(card, AgentCard):
            raise TypeError("Native handler requires an official AgentCard")
        validate_proto_required_fields(card)
        self.manager = manager
        self.guard = guard
        self._card = AgentCard.FromString(card.SerializeToString())
        extensions = self._card.capabilities.extensions
        if (
            len({item.uri for item in extensions}) != len(extensions)
            or {item.uri for item in extensions if item.required}
            != guard.required_extensions
        ):
            raise ValueError(
                "Native Agent Card extensions must match the request guard"
            )
        if (
            self._card.capabilities.push_notifications
            or self._card.capabilities.extended_agent_card
        ):
            raise ValueError(
                "Native handler does not provide push notifications or an extended card"
            )

    @property
    def agent_card(self) -> AgentCard:
        """Return a copy of the card whose capabilities this handler enforces."""
        return AgentCard.FromString(self._card.SerializeToString())

    async def _task(self, task_id: str, principal_id: str) -> Task:
        task = await self.manager.task_store.get_for_creator(
            task_id,
            principal_id,
            recipient_agent_id=self.guard.scope.recipient_id,
        )
        if task is None:
            raise TaskNotFoundError()
        return task

    async def on_get_task(
        self, params: GetTaskRequest, context: ServerCallContext
    ) -> Task:
        async with self.guard.admit(params, "GetTask", context) as admitted:
            validate_history_length(admitted.request)
            task = await self._task(admitted.request.id, admitted.principal_id)
            return apply_history_length(task, admitted.request)

    async def on_list_tasks(
        self, params: ListTasksRequest, context: ServerCallContext
    ) -> ListTasksResponse:
        async with self.guard.admit(params, "ListTasks", context) as admitted:
            return await self.manager.task_store.list_for_creator(
                admitted.request,
                creator_agent_id=admitted.principal_id,
                recipient_agent_id=self.guard.scope.recipient_id,
            )

    async def on_cancel_task(
        self, params: CancelTaskRequest, context: ServerCallContext
    ) -> Task:
        async with self.guard.admit(params, "CancelTask", context) as admitted:
            task = await self._task(admitted.request.id, admitted.principal_id)
            if task.status.state in TERMINAL_STATES:
                raise TaskNotCancelableError()
            try:
                return await self.manager.cancel_task(
                    task.id,
                    agent_name=admitted.principal_id,
                    recipient_agent_id=self.guard.scope.recipient_id,
                )
            except TaskCancellationAuthorizationError as error:
                raise TaskNotFoundError() from error
            except ValueError as error:
                # Another terminal transition may win after the authorized read.
                if is_execution_control_error(error):
                    raise
                current = await self._task(task.id, admitted.principal_id)
                if current.status.state in TERMINAL_STATES:
                    raise TaskNotCancelableError() from error
                raise

    async def _submit(
        self, admitted: AuthorizedNativeRequest[SendMessageRequest]
    ) -> Task:
        params, principal_id = admitted.request, admitted.principal_id
        validate_history_length(params.configuration)
        if params.configuration.HasField("task_push_notification_config"):
            raise PushNotificationNotSupportedError()
        if params.message.task_id:
            existing = await self._task(params.message.task_id, principal_id)
            if (
                params.message.context_id
                and params.message.context_id != existing.context_id
            ):
                raise InvalidParamsError(
                    message="Task and message context identifiers disagree"
                )
            raise UnsupportedOperationError(
                message="This task does not accept additional messages"
            )
        try:
            metadata = MessageToDict(params.metadata)
            require_metadata_extensions(metadata, admitted.activated_extensions)
            hint = parse_message_metadata(metadata).skill_hint
        except ValueError as error:
            raise InvalidParamsError(message=str(error)) from error
        modes = self._card.default_input_modes
        if hint is not None:
            selected = next(
                (
                    skill
                    for skill in self.manager.get_agent_skills()
                    if skill.id == hint
                ),
                None,
            )
            if selected is None:
                raise InvalidParamsError(
                    message="Recipient does not advertise the suggested skill"
                )
        validate_input_parts(params.message.parts, modes)
        output_modes = params.configuration.accepted_output_modes
        if output_modes and not set(output_modes).intersection(
            self._card.default_output_modes
        ):
            raise ContentTypeNotSupportedError()
        try:
            return await self.manager.create_task(params, creator_agent_id=principal_id)
        except ContextOwnershipError as error:
            if is_execution_control_error(error):
                raise
            raise HTTPException(
                status_code=403, detail="A2A context is not available to this principal"
            ) from error

    async def _open_stream(self, task_id: str, principal_id: str):
        stream = self.manager.subscribe(
            task_id,
            creator_agent_id=principal_id,
            recipient_agent_id=self.guard.scope.recipient_id,
        )
        try:
            initial = await anext(stream)
        except StopAsyncIteration as error:
            await close_native_execution_stream(stream, self)
            raise TaskNotFoundError() from error
        except BaseException:
            await close_native_execution_stream(stream, self)
            raise
        if not isinstance(initial, Task):
            await close_native_execution_stream(stream, self)
            raise InvalidAgentResponseError(
                message="The initial native stream event must be a Task"
            )
        return stream, initial

    async def _events(
        self, stream, initial: Task, admitted: AuthorizedNativeRequest
    ) -> AsyncGenerator[Event]:
        async with owned_execution_stream(self, stream) as events:
            event = initial
            while True:
                async with self.guard.hold_authority(admitted):
                    await self._task(initial.id, admitted.principal_id)
                yield event
                try:
                    event = await anext(events)
                except StopAsyncIteration:
                    return

    async def on_message_send(
        self, params: SendMessageRequest, context: ServerCallContext
    ) -> Task:
        async with self.guard.admit(params, "SendMessage", context) as admitted:
            task = await self._submit(admitted)
            if admitted.request.configuration.return_immediately:
                return apply_history_length(task, admitted.request.configuration)
            stream, initial = await self._open_stream(task.id, admitted.principal_id)
        async with owned_execution_stream(
            self, self._events(stream, initial, admitted)
        ) as events:

            async def wait_events():
                async for _ in events:
                    pass
                return True

            completed, _ = await gather_owned(
                (
                    factory()
                    for factory in (
                        wait_events,
                        lambda: wait_native_disconnect(context),
                    )
                ),
                return_when=asyncio.FIRST_COMPLETED,
                retain_originals=True,
                name_prefix="native-a2a-blocking-response",
            )
            if completed is not True:
                raise HTTPException(
                    status_code=499, detail="A2A HTTP client disconnected"
                )
        async with self.guard.hold_authority(admitted):
            task = await self._task(task.id, admitted.principal_id)
            return apply_history_length(task, admitted.request.configuration)

    async def on_message_send_stream(
        self, params: SendMessageRequest, context: ServerCallContext
    ) -> AsyncGenerator[Event]:
        async with self.guard.admit(
            params, "SendStreamingMessage", context
        ) as admitted:
            if not self._card.capabilities.streaming:
                raise UnsupportedOperationError(
                    message="Streaming is not advertised by this agent"
                )
            task = await self._submit(admitted)
            stream, initial = await self._open_stream(task.id, admitted.principal_id)
        async with owned_execution_stream(
            self, self._events(stream, initial, admitted)
        ) as events:
            async for event in events:
                yield event

    async def on_subscribe_to_task(
        self, params: SubscribeToTaskRequest, context: ServerCallContext
    ) -> AsyncGenerator[Event]:
        async with self.guard.admit(params, "SubscribeToTask", context) as admitted:
            if not self._card.capabilities.streaming:
                raise UnsupportedOperationError(
                    message="Streaming is not advertised by this agent"
                )
            stream, initial = await self._open_stream(
                admitted.request.id, admitted.principal_id
            )
            if initial.status.state in TERMINAL_STATES:
                await close_native_execution_stream(stream, self)
                raise UnsupportedOperationError(
                    message="Cannot subscribe to a terminal task"
                )
        async with owned_execution_stream(
            self, self._events(stream, initial, admitted)
        ) as events:
            async for event in events:
                yield event

    async def _unsupported_push(
        self, params, context: ServerCallContext, operation: str
    ):
        async with self.guard.admit(params, operation, context):
            raise PushNotificationNotSupportedError()

    async def on_create_task_push_notification_config(
        self, params: TaskPushNotificationConfig, context: ServerCallContext
    ) -> TaskPushNotificationConfig:
        return await self._unsupported_push(
            params, context, "CreateTaskPushNotificationConfig"
        )

    async def on_get_task_push_notification_config(
        self, params: GetTaskPushNotificationConfigRequest, context: ServerCallContext
    ) -> TaskPushNotificationConfig:
        return await self._unsupported_push(
            params, context, "GetTaskPushNotificationConfig"
        )

    async def on_list_task_push_notification_configs(
        self, params: ListTaskPushNotificationConfigsRequest, context: ServerCallContext
    ) -> ListTaskPushNotificationConfigsResponse:
        return await self._unsupported_push(
            params, context, "ListTaskPushNotificationConfigs"
        )

    async def on_delete_task_push_notification_config(
        self,
        params: DeleteTaskPushNotificationConfigRequest,
        context: ServerCallContext,
    ) -> None:
        await self._unsupported_push(
            params, context, "DeleteTaskPushNotificationConfig"
        )

    async def on_get_extended_agent_card(
        self, params: GetExtendedAgentCardRequest, context: ServerCallContext
    ) -> AgentCard:
        async with self.guard.admit(params, "GetExtendedAgentCard", context):
            raise UnsupportedOperationError(
                message="An extended Agent Card is not advertised by this agent"
            )
