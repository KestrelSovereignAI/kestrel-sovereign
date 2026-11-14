"""Official SDK operations over Kestrel's canonical native task lifecycle."""

import asyncio
import base64
import binascii
import json
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager, contextmanager
from typing import NoReturn
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
    A2AError,
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
from kestrel_sdk.native_a2a.utils.source_ownership import SourceWithdrawal, WakeableSource
from kestrel_sdk.native_a2a.utils.stream_refusal import StreamRefusal
from kestrel_sdk.native_a2a.utils.task import (
    apply_history_length,
    validate_history_length,
    validate_page_size,
)
from google.protobuf.json_format import MessageToDict
from kestrel_sdk.security.exceptions import SSRFError
from starlette.exceptions import HTTPException

from kestrel_sovereign._async_ownership import OriginalContextExit, gather_owned
from kestrel_sovereign.a2a.extensions import (
    NativeMessageMetadata,
    PEER_STOP_URI,
    parse_message_metadata,
    require_metadata_extensions,
)
from kestrel_sovereign.a2a.lifecycle import TERMINAL_STATES
from kestrel_sovereign.a2a.native_input import prepare_native_task_input
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
    ExecutionCommitOutcomeError,
    execution_commit_outcome,
    execution_observation_error,
    is_execution_control_error,
    owned_execution_stream,
)


def _push_protocol_error(
    error: PushConfigConflict | SSRFError | PushServiceUnavailable,
) -> A2AError:
    """One callback-domain mapping; original execution control is never data."""
    if is_execution_control_error(error):
        raise error
    if isinstance(error, PushServiceUnavailable):
        return InternalError(message="Native push delivery service is unavailable")
    if isinstance(error, PushTaskNotFound):
        return TaskNotFoundError()
    if isinstance(error, PushRetirementPending):
        return InternalError(
            message="Push deletion is waiting for original HTTP retirement"
        )
    return InvalidParamsError(
        message="Push notification configuration was not accepted"
    )


@contextmanager
def _push_errors():
    """Translate callback domain refusals, retaining original execution control."""
    try:
        yield
    except (PushConfigConflict, SSRFError, PushServiceUnavailable) as error:
        raise _push_protocol_error(error) from error


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
    except (ValueError, UnicodeError, binascii.Error, RecursionError) as error:
        if is_execution_control_error(error):
            raise
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

    def _submission_claims(
        self, admitted: AuthorizedNativeRequest[SendMessageRequest]
    ) -> NativeMessageMetadata | StreamRefusal:
        """Validate protocol input without an async/resource lifetime boundary.

        Only these pure validators produce logical refusals here. Exceptions
        from manager I/O, original close or authority exit are not intercepted
        by a catch-all around the streaming source.
        """
        params = admitted.request
        try:
            validate_history_length(params.configuration)
            metadata = MessageToDict(params.metadata)
            require_metadata_extensions(metadata, admitted.activated_extensions)
            claims = parse_message_metadata(metadata)
            if claims.peer_stop is None:
                prepare_native_task_input(params)
            modes = skill_input_modes(
                self._card.default_input_modes,
                self.manager.get_agent_skills(),
                claims.skill_hint,
            )
            validate_input_parts(params.message.parts, modes)
        except (InvalidParamsError, ContentTypeNotSupportedError) as error:
            if is_execution_control_error(error):
                raise
            return StreamRefusal(error)
        except ValueError as error:
            if is_execution_control_error(error):
                raise
            refusal = InvalidParamsError(message=str(error))
            refusal.__cause__ = error
            return StreamRefusal(refusal)
        output_modes = params.configuration.accepted_output_modes
        if output_modes and not set(output_modes).intersection(
            self._card.default_output_modes
        ):
            return StreamRefusal(ContentTypeNotSupportedError())
        if claims.peer_stop is not None:
            if params.configuration.HasField("task_push_notification_config"):
                return StreamRefusal(
                    InvalidParamsError(
                        message="Peer Stop returns a Message and cannot register task push"
                    )
                )
            if self._peer_stop_target is None:
                return StreamRefusal(
                    UnsupportedOperationError(message="Peer Stop is not advertised")
                )
            if (
                params.message.task_id
                or params.message.role != Role.ROLE_USER
                or len(params.message.parts) != 1
                or params.message.parts[0].WhichOneof("content") != "text"
                or params.message.parts[0].text != (claims.peer_stop["reason"] or "")
            ):
                return StreamRefusal(
                    InvalidParamsError(
                        message="Peer Stop requires its own user message and reason"
                    )
                )
            if output_modes:
                try:
                    validate_input_parts(
                        [Part(raw=b"{}", media_type="application/json")], output_modes
                    )
                except (InvalidParamsError, ContentTypeNotSupportedError) as error:
                    if is_execution_control_error(error):
                        raise
                    return StreamRefusal(error)
        return claims

    async def _submit(
        self, admitted: AuthorizedNativeRequest[SendMessageRequest]
    ) -> Task | Message | StreamRefusal:
        params, principal_id = admitted.request, admitted.principal_id
        if params.configuration.HasField("task_push_notification_config"):
            if self._push_service is None:
                return StreamRefusal(PushNotificationNotSupportedError())
            try:
                self._require_push()
            except (PushConfigConflict, SSRFError, PushServiceUnavailable) as error:
                refusal = _push_protocol_error(error)
                refusal.__cause__ = error
                return StreamRefusal(refusal)
        claims = self._submission_claims(admitted)
        if isinstance(claims, StreamRefusal):
            return claims
        if claims.peer_stop is not None:
            from kestrel_sovereign.signals.sources.peer_stop import dispatch_peer_stop

            async with self.guard.hold_authority(admitted, refusal_values=True) as refused:
                if isinstance(refused, StreamRefusal):
                    return refused
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

        def admission_fence() -> OriginalContextExit[StreamRefusal | None]:
            return OriginalContextExit(self.guard.hold_authority(admitted, refusal_values=True))

        try:
            if params.message.task_id:
                task = await self.manager.continue_task(
                    params,
                    creator_agent_id=principal_id,
                    admission_fence=admission_fence,
                    refusal_values=True,
                )
                if task is None:
                    return StreamRefusal(TaskNotFoundError())
                return task
            return await self.manager.create_task(
                params,
                creator_agent_id=principal_id,
                admission_fence=admission_fence,
                refusal_values=True,
            )
        except TaskInputUnsupportedError as error:
            if is_execution_control_error(error):
                raise
            refusal = UnsupportedOperationError(message=str(error))
            refusal.__cause__ = error
            return StreamRefusal(refusal)
        except TaskInputConflictError as error:
            if is_execution_control_error(error):
                raise
            refusal = InvalidParamsError(message=str(error))
            refusal.__cause__ = error
            return StreamRefusal(refusal)
        except (PushConfigConflict, SSRFError, PushServiceUnavailable) as error:
            refusal = _push_protocol_error(error)
            refusal.__cause__ = error
            return StreamRefusal(refusal)
        except ContextOwnershipError as error:
            if is_execution_control_error(error):
                raise
            refusal = HTTPException(
                status_code=403, detail="A2A context is not available to this principal"
            )
            refusal.__cause__ = error
            return StreamRefusal(refusal)

    @asynccontextmanager
    async def _open_stream(
        self, task_id: str, principal_id: str, *, withdrawal: SourceWithdrawal | None = None
    ) -> AsyncIterator[tuple[AsyncIterator[Event], Task] | StreamRefusal]:
        withdrawal = SourceWithdrawal() if withdrawal is None else withdrawal
        original = self.manager.subscribe(
            task_id,
            creator_agent_id=principal_id,
            recipient_agent_id=self.guard.scope.recipient_id,
            withdrawal=withdrawal,
        )
        # Ownership precedes the FIRST read, not just later event forwarding.
        # The caller registers this scope before leaving its acquisition lease,
        # so an actual lease-exit failure still joins this same source owner.
        async with owned_execution_stream(self, original, withdrawal=withdrawal) as stream:
            try:
                initial = await anext(stream)
            except StopAsyncIteration as error:
                if is_execution_control_error(error):
                    raise
                refusal = TaskNotFoundError()
                refusal.__cause__ = error
            else:
                if not isinstance(initial, Task):
                    raise InvalidAgentResponseError(
                        message="The initial native stream event must be a Task"
                    )
                yield stream, initial
                return
        # A not-found decision is data only AFTER actual original retirement.
        yield StreamRefusal(refusal)

    async def _events(
        self, stream, initial: Task, admitted: AuthorizedNativeRequest,
        *, response_result: Task | None = None,
    ) -> AsyncGenerator[Event]:
        # Borrow the already-owned subscription. _open_stream retains its
        # original owner across acquisition, every read, and physical close.
        event = initial
        while True:
            authority = (
                self.guard.hold_authority(admitted)
                if response_result is None
                else self._response_authority(admitted, response_result)
            )
            async with authority:
                await self._task(initial.id, admitted.principal_id)
            yield event
            try:
                event = await anext(stream)
            except StopAsyncIteration:
                return

    @staticmethod
    def _raise_response_failure(result: Task | Message, error: BaseException) -> NoReturn:
        # A returned Task proves canonical durable admission. A taskless
        # control Message may describe another effect, not a Task transaction;
        # retain that uncertainty instead of manufacturing a commit receipt.
        outcome = "committed" if isinstance(result, Task) else "unknown"
        raise execution_observation_error(ExecutionCommitOutcomeError(outcome), error)

    @asynccontextmanager
    async def _response_authority(self, admitted: AuthorizedNativeRequest, result: Task | Message):
        """A later denial cannot retroactively turn an effect into refusal."""
        async with self.guard.hold_authority(admitted, refusal_values=True) as refused:
            if isinstance(refused, StreamRefusal):
                self._raise_response_failure(result, refused.error)
            yield

    async def _send_task_events(
        self, task: Task, admitted: AuthorizedNativeRequest, withdrawal: SourceWithdrawal
    ) -> AsyncGenerator[Event]:
        """One post-admission subscription path for unary and streamed sends."""
        async with AsyncExitStack() as streams:
            async with self._response_authority(admitted, task):
                opened = await streams.enter_async_context(
                    self._open_stream(task.id, admitted.principal_id, withdrawal=withdrawal)
                )
            if not isinstance(opened, StreamRefusal):
                stream, initial = opened
                async with owned_execution_stream(
                    self, self._events(stream, initial, admitted, response_result=task),
                    withdrawal=withdrawal,
                ) as events:
                    async for event in events:
                        yield event
                return
        self._raise_response_failure(task, opened.error)

    async def on_message_send(
        self, params: SendMessageRequest, context: ServerCallContext
    ) -> Task | Message:
        async with self.guard.admit(params, "SendMessage", context) as admitted:
            pass
        task = await self._submit(admitted)
        if isinstance(task, StreamRefusal):
            raise task.error
        # Bind outcome to the whole post-admission response, not selected
        # rechecks. Physical source/retirement errors can precede the explicit
        # disconnect branch; they must retain BOTH their original evidence and
        # the already-committed operation. Raw caller cancellation stays raw.
        try:
            return await self._message_response(task, admitted, context)
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            if execution_commit_outcome(error) is not None:
                raise
            self._raise_response_failure(task, error)

    async def _message_response(
        self, task: Task | Message, admitted: AuthorizedNativeRequest[SendMessageRequest], context: ServerCallContext
    ) -> Task | Message:
        if isinstance(task, Message) or admitted.request.configuration.return_immediately:
            async with self._response_authority(admitted, task):
                if isinstance(task, Message):
                    return task
                return apply_history_length(task, admitted.request.configuration)

        withdrawal = SourceWithdrawal()

        # Race ONLY demand from the canonical native source kernel. Stopping
        # that observer wakes known idle waits; it never cancels the original
        # source or an already-running finally. This same lifetime scope joins
        # physical retirement before a blocking response can leave.
        async with owned_execution_stream(
            self, self._send_task_events(task, admitted, withdrawal), withdrawal=withdrawal
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
                self._raise_response_failure(
                    task, HTTPException(status_code=499, detail="A2A HTTP client disconnected")
                )
        async with self._response_authority(admitted, task):
            task = await self._task(task.id, admitted.principal_id)
            return apply_history_length(task, admitted.request.configuration)

    def on_message_send_stream(
        self, params: SendMessageRequest, context: ServerCallContext
    ) -> WakeableSource[Event | StreamRefusal]:
        withdrawal = SourceWithdrawal()
        return WakeableSource(self._message_send_stream(params, context, withdrawal), withdrawal)

    async def _message_send_stream(
        self, params: SendMessageRequest, context: ServerCallContext, withdrawal: SourceWithdrawal
    ) -> AsyncGenerator[Event | StreamRefusal]:
        async with self.guard.admit(
            params, "SendStreamingMessage", context, refusal_values=True
        ) as admitted:
            streaming = (
                self._card.capabilities.streaming
                if not isinstance(admitted, StreamRefusal)
                else False
            )
        if isinstance(admitted, StreamRefusal):
            yield admitted
            return
        if not streaming:
            yield StreamRefusal(
                UnsupportedOperationError(
                    message="Streaming is not advertised by this agent"
                )
            )
            return
        task = await self._submit(admitted)
        if isinstance(task, StreamRefusal):
            yield task
            return
        if isinstance(task, Message):
            # Revalidate authority, but do not suspend the source while the
            # lifecycle reader lease is still held.
            async with self._response_authority(admitted, task):
                pass
            yield task
            return
        async with owned_execution_stream(
            self, self._send_task_events(task, admitted, withdrawal), withdrawal=withdrawal
        ) as events:
            async for event in events:
                yield event

    def on_subscribe_to_task(
        self, params: SubscribeToTaskRequest, context: ServerCallContext
    ) -> WakeableSource[Event | StreamRefusal]:
        withdrawal = SourceWithdrawal()
        return WakeableSource(self._subscribe_to_task(params, context, withdrawal), withdrawal)

    async def _subscribe_to_task(
        self, params: SubscribeToTaskRequest, context: ServerCallContext, withdrawal: SourceWithdrawal
    ) -> AsyncGenerator[Event | StreamRefusal]:
        async with AsyncExitStack() as streams:
            async with self.guard.admit(
                params, "SubscribeToTask", context, refusal_values=True
            ) as admitted:
                if isinstance(admitted, StreamRefusal):
                    opened = admitted
                elif not self._card.capabilities.streaming:
                    opened = StreamRefusal(
                        UnsupportedOperationError(
                            message="Streaming is not advertised by this agent"
                        )
                    )
                else:
                    opened = await streams.enter_async_context(
                        self._open_stream(
                            admitted.request.id, admitted.principal_id, withdrawal=withdrawal
                        )
                    )
                    if not isinstance(opened, StreamRefusal):
                        stream, initial = opened
                        if initial.status.state in TERMINAL_STATES:
                            opened = StreamRefusal(
                                UnsupportedOperationError(
                                    message="Cannot subscribe to a terminal task"
                                )
                            )
            if not isinstance(opened, StreamRefusal):
                stream, initial = opened
                async with owned_execution_stream(
                    self, self._events(stream, initial, admitted), withdrawal=withdrawal
                ) as events:
                    async for event in events:
                        yield event
                return
        # Only explicit semantic decisions become values. An exception from
        # actual original close or authority exit never reaches this yield.
        yield opened

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
