"""Official SDK operations over Kestrel's canonical native task lifecycle."""

import asyncio
import base64
import binascii
import json
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, contextmanager
from uuid import uuid4

from kestrel_sdk.native_a2a.server.context import ServerCallContext
from kestrel_sdk.native_a2a.server.events.event_queue import Event
from kestrel_sdk.native_a2a.server.request_handlers import RequestHandler
from kestrel_sdk.native_a2a.types import (
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
    Message,
    Part,
    Role,
    SendMessageRequest,
    SubscribeToTaskRequest,
    Task,
    TaskPushNotificationConfig,
)
from kestrel_sdk.native_a2a.utils.constants import DEFAULT_LIST_TASKS_PAGE_SIZE
from kestrel_sdk.native_a2a.utils.errors import (
    ContentTypeNotSupportedError,
    InternalError,
    InvalidAgentResponseError,
    InvalidParamsError,
    PushNotificationNotSupportedError,
    TaskNotCancelableError,
    TaskNotFoundError,
    UnsupportedOperationError,
)
from kestrel_sdk.native_a2a.utils.proto_utils import validate_proto_required_fields
from kestrel_sdk.native_a2a.utils.task import (
    apply_history_length,
    validate_history_length,
    validate_page_size,
)
from google.protobuf.json_format import MessageToDict
from kestrel_sdk.security.exceptions import SSRFError
from starlette.exceptions import HTTPException

from kestrel_sovereign._async_ownership import gather_owned
from kestrel_sovereign.a2a.extensions import (
    PEER_STOP_URI,
    parse_message_metadata,
    require_metadata_extensions,
)
from kestrel_sovereign.a2a.lifecycle import TERMINAL_STATES
from kestrel_sovereign.a2a.media import (
    encode_inline_json,
    skill_input_modes,
    validate_input_parts,
)
from kestrel_sovereign.a2a.native_context import wait_native_disconnect
from kestrel_sovereign.a2a.native_security import (
    AuthorizedNativeRequest,
    NativeDIDRequestGuard,
)
from kestrel_sovereign.a2a.push_service import PushServiceUnavailable
from kestrel_sovereign.a2a.push_worker import NativePushWorker, PushRetirementPending
from kestrel_sovereign.a2a.stores.unified.context_store import ContextOwnershipError
from kestrel_sovereign.a2a.stores.unified.push_config_store import (
    PushConfigConflict,
    PushTaskNotFound,
)
from kestrel_sovereign.a2a.stores.unified.task_store import (
    TaskInputConflictError,
    TaskInputUnsupportedError,
)
from kestrel_sovereign.a2a.task_manager import (
    TaskCancellationAuthorizationError,
    TaskManager,
)
from kestrel_sovereign.execution_custody import (
    close_native_execution_stream,
    is_execution_control_error,
    owned_execution_stream,
)


@contextmanager
def _push_errors():
    """Translate callback domain refusals, retaining original execution control."""
    try:
        yield
    except (PushConfigConflict, SSRFError, PushServiceUnavailable) as error:
        if is_execution_control_error(error):
            raise
        if isinstance(error, PushServiceUnavailable):
            raise InternalError(
                message="Native push delivery service is unavailable"
            ) from error
        if isinstance(error, PushTaskNotFound):
            raise TaskNotFoundError() from error
        if isinstance(error, PushRetirementPending):
            raise InternalError(
                message="Push deletion is waiting for original HTTP retirement"
            ) from error
        raise InvalidParamsError(
            message="Push notification configuration was not accepted"
        ) from error


def _push_cursor(scope, principal_id, task_id, config_id):
    return (
        base64.urlsafe_b64encode(
            json.dumps(
                [scope.recipient_id, scope.tenant, principal_id, task_id, config_id],
                separators=(",", ":"),
            ).encode()
        )
        .decode()
        .rstrip("=")
    )


def _push_after(token, scope, principal_id, task_id):
    if not token:
        return ""
    try:
        decoded = json.loads(
            base64.b64decode(
                token + "=" * (-len(token) % 4), altchars=b"-_", validate=True
            )
        )
    except (ValueError, UnicodeError, binascii.Error) as error:
        raise InvalidParamsError(
            message="Invalid push configuration page token"
        ) from error
    if (
        not isinstance(decoded, list)
        or len(decoded) != 5
        or decoded[:4] != [scope.recipient_id, scope.tenant, principal_id, task_id]
        or not isinstance(decoded[4], str)
        or not decoded[4]
    ):
        raise InvalidParamsError(
            message="Push configuration page token belongs to another query"
        )
    return decoded[4]


class NativeA2ARequestHandler(RequestHandler):
    """Both SDK HTTP bindings share one recipient, guard and task manager.

    Incoming messages start independently tracked tasks in their native context.
    A caller-supplied task ID always refers to an existing accessible task;
    task creation never accepts a client-selected identifier.
    """

    def __init__(
        self,
        manager: TaskManager,
        guard: NativeDIDRequestGuard,
        card: AgentCard,
        *,
        peer_stop_target: object | None = None,
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
        self._peer_stop_target = peer_stop_target
        self._card = AgentCard.FromString(card.SerializeToString())
        extensions = self._card.capabilities.extensions
        if (PEER_STOP_URI in {item.uri for item in extensions}) != (
            peer_stop_target is not None
        ) or (
            peer_stop_target is not None
            and getattr(peer_stop_target, "task_manager", None) is not manager
        ):
            raise ValueError(
                "Peer Stop must bind the advertised recipient's task manager"
            )
        if peer_stop_target is not None:
            from kestrel_sovereign.signals.sources.peer_stop import (
                peer_stop_target_identity,
            )

            if peer_stop_target_identity(peer_stop_target) != guard.scope.recipient_id:
                raise ValueError("Peer Stop target differs from its native recipient")
        if (
            len({item.uri for item in extensions}) != len(extensions)
            or {item.uri for item in extensions if item.required}
            != guard.required_extensions
        ):
            raise ValueError(
                "Native Agent Card extensions must match the request guard"
            )
        if self._card.capabilities.extended_agent_card:
            raise ValueError("Native handler does not provide an extended card")
        self._push_service = manager.push_service
        if self._card.capabilities.push_notifications != (
            self._push_service is not None
        ):
            raise ValueError(
                "Native push capability must match its bound managed service"
            )
        if self._push_service is not None:
            self._require_push()

    def _require_push(self) -> NativePushWorker:
        if self._push_service is None:
            raise PushNotificationNotSupportedError()
        if self.manager.push_service is not self._push_service:
            raise RuntimeError("Native handler lost its original push service binding")
        worker = self._push_service.require_ready()
        if worker.outbox.configurations.task_store is not self.manager.task_store:
            raise RuntimeError("Native handler lost its canonical push task store")
        return worker

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
    ) -> Task | Message:
        params, principal_id = admitted.request, admitted.principal_id
        validate_history_length(params.configuration)
        if params.configuration.HasField("task_push_notification_config"):
            with _push_errors():
                self._require_push()
        try:
            metadata = MessageToDict(params.metadata)
            require_metadata_extensions(metadata, admitted.activated_extensions)
            claims = parse_message_metadata(metadata)
            modes = skill_input_modes(
                self._card.default_input_modes,
                self.manager.get_agent_skills(),
                claims.skill_hint,
            )
        except ValueError as error:
            raise InvalidParamsError(message=str(error)) from error
        validate_input_parts(params.message.parts, modes)
        output_modes = params.configuration.accepted_output_modes
        if output_modes and not set(output_modes).intersection(
            self._card.default_output_modes
        ):
            raise ContentTypeNotSupportedError()
        if claims.peer_stop is not None:
            if params.configuration.HasField("task_push_notification_config"):
                raise InvalidParamsError(
                    message="Peer Stop returns a Message and cannot register task push"
                )
            if self._peer_stop_target is None:
                raise UnsupportedOperationError(message="Peer Stop is not advertised")
            if (
                params.message.task_id
                or params.message.role != Role.ROLE_USER
                or len(params.message.parts) != 1
                or params.message.parts[0].WhichOneof("content") != "text"
                or params.message.parts[0].text != (claims.peer_stop["reason"] or "")
            ):
                raise InvalidParamsError(
                    message="Peer Stop requires its own user message and reason"
                )
            if output_modes:
                validate_input_parts(
                    [Part(raw=b"{}", media_type="application/json")], output_modes
                )
            from kestrel_sovereign.signals.sources.peer_stop import dispatch_peer_stop

            async with self.guard.hold_authority(admitted):
                receipt = await dispatch_peer_stop(
                    self._peer_stop_target,
                    actor_id=principal_id,
                    intent=claims.peer_stop,
                    causation_chain=claims.causation_chain,
                )
            return Message(
                message_id=uuid4().hex,
                context_id=params.message.context_id or uuid4().hex,
                role=Role.ROLE_AGENT,
                parts=[
                    Part(raw=encode_inline_json(receipt), media_type="application/json")
                ],
            )
        try:
            if params.message.task_id:
                task = await self.manager.continue_task(
                    params,
                    creator_agent_id=principal_id,
                    admission_fence=lambda: self.guard.hold_authority(admitted),
                )
                if task is None:
                    raise TaskNotFoundError()
                return task
            return await self.manager.create_task(
                params,
                creator_agent_id=principal_id,
                admission_fence=lambda: self.guard.hold_authority(admitted),
            )
        except TaskInputUnsupportedError as error:
            raise UnsupportedOperationError(message=str(error)) from error
        except TaskInputConflictError as error:
            raise InvalidParamsError(message=str(error)) from error
        except (PushConfigConflict, SSRFError, PushServiceUnavailable):
            with _push_errors():
                raise
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
    ) -> Task | Message:
        async with self.guard.admit(params, "SendMessage", context) as admitted:
            pass
        task = await self._submit(admitted)
        async with self.guard.hold_authority(admitted):
            if isinstance(task, Message):
                return task
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
        async with self.guard.hold_authority(admitted):
            if isinstance(task, Message):
                yield task
                return
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

    @asynccontextmanager
    async def _admit_push(self, params, operation, context):
        async with self.guard.admit(params, operation, context) as admitted:
            with _push_errors():
                yield admitted, self._require_push()

    async def on_create_task_push_notification_config(
        self, params: TaskPushNotificationConfig, context: ServerCallContext
    ) -> TaskPushNotificationConfig:
        async with self._admit_push(
            params, "CreateTaskPushNotificationConfig", context
        ) as (admitted, worker):
            await self._task(admitted.request.task_id, admitted.principal_id)
            config = TaskPushNotificationConfig.FromString(
                admitted.request.SerializeToString()
            )
        with _push_errors():
            prepared = await worker.outbox.prepare(config)
            async with self.guard.hold_authority(admitted):
                if self._require_push() is not worker:
                    raise RuntimeError("Push admission lost its original worker")
                created = await worker.outbox.register_prepared(
                    prepared, creator_id=admitted.principal_id
                )
            return created.config

    async def on_get_task_push_notification_config(
        self, params: GetTaskPushNotificationConfigRequest, context: ServerCallContext
    ) -> TaskPushNotificationConfig:
        async with self._admit_push(
            params, "GetTaskPushNotificationConfig", context
        ) as (admitted, worker):
            config = await worker.outbox.configurations.get(
                admitted.request.task_id,
                admitted.request.id,
                creator_id=admitted.principal_id,
            )
            if config is None:
                raise TaskNotFoundError()
            return config.config

    async def on_list_task_push_notification_configs(
        self, params: ListTaskPushNotificationConfigsRequest, context: ServerCallContext
    ) -> ListTaskPushNotificationConfigsResponse:
        async with self._admit_push(
            params, "ListTaskPushNotificationConfigs", context
        ) as (admitted, worker):
            params = admitted.request
            size = params.page_size or DEFAULT_LIST_TASKS_PAGE_SIZE
            validate_page_size(size)
            after = _push_after(
                params.page_token,
                self.guard.scope,
                admitted.principal_id,
                params.task_id,
            )
            configs = await worker.outbox.configurations.list(
                params.task_id,
                creator_id=admitted.principal_id,
                limit=size + 1,
                after=after,
            )
            return ListTaskPushNotificationConfigsResponse(
                configs=[config.config for config in configs[:size]],
                next_page_token=(
                    _push_cursor(
                        self.guard.scope,
                        admitted.principal_id,
                        params.task_id,
                        configs[size - 1].config.id,
                    )
                    if len(configs) > size
                    else ""
                ),
            )

    async def on_delete_task_push_notification_config(
        self,
        params: DeleteTaskPushNotificationConfigRequest,
        context: ServerCallContext,
    ) -> None:
        async with self._admit_push(
            params, "DeleteTaskPushNotificationConfig", context
        ) as (admitted, worker):
            await worker.delete(
                admitted.request.task_id,
                admitted.request.id,
                creator_id=admitted.principal_id,
            )

    async def on_get_extended_agent_card(
        self, params: GetExtendedAgentCardRequest, context: ServerCallContext
    ) -> AgentCard:
        async with self.guard.admit(params, "GetExtendedAgentCard", context):
            raise UnsupportedOperationError(
                message="An extended Agent Card is not advertised by this agent"
            )
