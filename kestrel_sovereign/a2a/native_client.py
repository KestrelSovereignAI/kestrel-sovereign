"""Official SDK peer operations with immutable sender-side dispatch evidence."""

import sys
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from uuid import uuid4

import httpx
from a2a.client import ClientConfig, ClientFactory
from a2a.client.client import ClientCallContext
from a2a.client.interceptors import AfterArgs, BeforeArgs, ClientCallInterceptor
from a2a.extensions.common import HTTP_EXTENSION_HEADER
from a2a.types import (
    AgentCard,
    AgentInterface,
    CancelTaskRequest,
    GetTaskRequest,
    ListTasksRequest,
    Message,
    SendMessageRequest,
    StreamResponse,
    SubscribeToTaskRequest,
    Task,
    TaskState,
)
from a2a.utils.constants import VERSION_HEADER
from a2a.utils.proto_utils import validate_proto_required_fields
from google.protobuf.json_format import MessageToDict, ParseError
from google.protobuf.message import Message as ProtoMessage

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.a2a.extensions import (
    NATIVE_EXTENSION_URIS,
    SIGNED_SENDER_URI,
    parse_message_metadata,
    require_metadata_extensions,
)
from kestrel_sovereign.a2a.lifecycle import STREAM_STOP_STATES
from kestrel_sovereign.a2a.media import validate_input_parts
from kestrel_sovereign.a2a.native_card import (
    REQUIRED_NATIVE_EXTENSIONS,
    verify_native_agent_card,
)
from kestrel_sovereign.a2a.native_sender import (
    SENDER_PROOF_HEADER,
    NativeProofPolicy,
    NativeSenderProofError,
    native_request_digest,
    sign_native_request,
)
from kestrel_sovereign.a2a.scope import NativeRecipientScope
from kestrel_sovereign.a2a.stores.unified.outbound_store import (
    DispatchState,
    OutboundDispatch,
    OutboundDispatchConflict,
    OutboundStore,
)
from kestrel_sovereign.a2a.stores.unified.question_store import NativeQuestionStore
from kestrel_sovereign.execution_custody import (
    bind_native_http_execution_admission,
    execution_terminal_error,
    execution_work_operation,
    execution_work_stream,
    is_execution_control_error,
    native_execution_response_owner,
    owned_execution_stream,
)
from kestrel_sovereign.identity.hybrid_keypair import HybridKeypair
from kestrel_sovereign.identity.runtime_identity import AgentIdentity
from kestrel_sovereign.security.signature_verification import verification_method_kid

# Interceptors receive Python client-method names. Proofs bind the official
# service operation; the verifier also checks its exact protobuf input type.
_SDK_OPERATIONS = {
    "send_message": "SendMessage",
    "send_message_streaming": "SendStreamingMessage",
    "get_task": "GetTask",
    "list_tasks": "ListTasks",
    "cancel_task": "CancelTask",
    "subscribe": "SubscribeToTask",
    "create_task_push_notification_config": "CreateTaskPushNotificationConfig",
    "get_task_push_notification_config": "GetTaskPushNotificationConfig",
    "list_task_push_notification_configs": "ListTaskPushNotificationConfigs",
    "delete_task_push_notification_config": "DeleteTaskPushNotificationConfig",
    "get_extended_agent_card": "GetExtendedAgentCard",
}

_REQUEST_BINDING_KEY = "kestrel.native_request_binding"


@dataclass(frozen=True)
class _NativeRequestBinding:
    operation: str
    digest: str


@dataclass(frozen=True)
class NativeSenderInterceptor(ClientCallInterceptor):
    """Sign each SDK request after host verification of the selected peer.

    The caller supplies its authorized signing identity and the recipient
    scope from the verified Agent Card. This hook grants no routing permission,
    resolves no peer itself, and never retries a transport operation.
    """

    keypair: HybridKeypair = field(repr=False)
    sender: str
    recipient: NativeRecipientScope
    extensions: frozenset[str]
    policy: NativeProofPolicy
    classical_kid: str
    pq_kid: str
    clock: Callable[[], float]
    nonce: Callable[[], str]

    @classmethod
    def for_identity(
        cls,
        identity: AgentIdentity,
        *,
        recipient: NativeRecipientScope,
        extensions: frozenset[str],
        policy: NativeProofPolicy,
        clock: Callable[[], float],
        nonce: Callable[[], str],
    ):
        """Select both published key IDs by verified key material, never order."""
        keys = identity.signing_key_ids()
        pair = identity.hybrid_keypair
        return cls(
            keypair=pair,
            sender=identity.signing_did,
            recipient=recipient,
            extensions=extensions,
            policy=policy,
            classical_kid=verification_method_kid(keys[pair.classical.suite_id]),
            pq_kid=verification_method_kid(keys[pair.pq.suite_id]),
            clock=clock,
            nonce=nonce,
        )

    async def before(self, args: BeforeArgs) -> None:
        if not isinstance(args.input, ProtoMessage):
            raise TypeError("Native signing requires an official protobuf request")
        tenant = args.input.DESCRIPTOR.fields_by_name.get("tenant")
        if tenant is None or getattr(args.input, tenant.name) != self.recipient.tenant:
            raise NativeSenderProofError(
                "Request tenant differs from the selected Agent Card interface"
            )
        operation = _SDK_OPERATIONS.get(args.method)
        if operation is None:
            raise NativeSenderProofError("Unsupported SDK operation for native signing")
        original = args.context
        expected = (
            original.state.get(_REQUEST_BINDING_KEY) if original is not None else None
        )
        if expected is not None and (
            not isinstance(expected, _NativeRequestBinding)
            or expected.operation != operation
            or expected.digest != native_request_digest(args.input, operation)
        ):
            raise NativeSenderProofError(
                "SDK request differs from its retained native intent"
            )
        context = ClientCallContext(
            state=dict(original.state) if original is not None else {},
            timeout=original.timeout if original is not None else None,
            service_parameters=dict(original.service_parameters or {})
            if original is not None
            else {},
        )
        context.service_parameters[HTTP_EXTENSION_HEADER] = ",".join(
            sorted(self.extensions)
        )
        context.service_parameters[SENDER_PROOF_HEADER] = sign_native_request(
            self.keypair,
            args.input,
            sender=self.sender,
            recipient=self.recipient.recipient_id,
            operation=operation,
            timestamp=int(self.clock()),
            nonce=self.nonce(),
            extensions=self.extensions,
            classical_kid=self.classical_kid,
            pq_kid=self.pq_kid,
            policy=self.policy,
        )
        args.context = context

    async def after(self, args: AfterArgs) -> None:
        """The SDK owns response parsing and errors; the proof adds no retry."""


class NativePeerProtocolError(ValueError):
    """The independently trusted peer did not provide the native contract."""


@dataclass(frozen=True)
class NativeDispatchReceipt:
    dispatch: OutboundDispatch
    response: Task | Message


class NativePeerSession:
    """A host-authorized peer session over a borrowed, governed HTTP client.

    The host must supply its origin-pinned transport (or an explicit in-process
    transport) and reauthorization callback. A verified card grants no access.
    The caller owns HTTP retirement; the SDK owns protocol encoding and parsing.
    No draft payload, client-selected task ID, retry, or ID rekey is involved.
    """

    def __init__(
        self,
        *,
        card: AgentCard,
        signing_did: str,
        trusted_document: dict,
        recipient: NativeRecipientScope,
        interface: AgentInterface,
        signer: NativeSenderInterceptor,
        http_client: httpx.AsyncClient,
        outbound_store: OutboundStore,
        authorize: Callable[[], Awaitable[None]],
        question_store: NativeQuestionStore | None = None,
    ):
        snapshot = AgentCard.FromString(card.SerializeToString())
        validate_proto_required_fields(snapshot)
        verify_native_agent_card(
            snapshot, signing_did=signing_did, trusted_document=trusted_document
        )
        extensions = {item.uri: item for item in snapshot.capabilities.extensions}
        supported = frozenset(NATIVE_EXTENSION_URIS)
        versions = [
            MessageToDict(item.params).get("version")
            for uri, item in extensions.items()
            if uri in signer.extensions
        ]
        if (
            len(extensions) != len(snapshot.capabilities.extensions)
            or not REQUIRED_NATIVE_EXTENSIONS <= extensions.keys()
            or any(
                item.required and uri not in supported
                for uri, item in extensions.items()
            )
            or any(not extensions[uri].required for uri in REQUIRED_NATIVE_EXTENSIONS)
            or any(isinstance(version, bool) or version != 1 for version in versions)
        ):
            raise NativePeerProtocolError(
                "Peer does not declare the supported native extensions"
            )
        declaration = MessageToDict(extensions[SIGNED_SENDER_URI].params)
        if (
            declaration.get("recipient") != recipient.recipient_id
            or declaration.get("signing_did") != signing_did
            or declaration.get("header") != SENDER_PROOF_HEADER
        ):
            raise NativePeerProtocolError(
                "Signed Agent Card differs from the host-selected peer"
            )
        # ClientFactory selects an interface for a protocol, not for a tenant.
        # Refuse ambiguity instead of altering a signed card to influence it.
        candidates = [
            item
            for item in snapshot.supported_interfaces
            if item.protocol_binding == interface.protocol_binding
        ]
        if (
            interface.protocol_binding not in {"JSONRPC", "HTTP+JSON"}
            or len(candidates) != 1
            or candidates[0] != interface
            or interface.protocol_version != "1.0"
            or interface.tenant != recipient.tenant
            or signer.recipient != recipient
            or not REQUIRED_NATIVE_EXTENSIONS <= signer.extensions
            or not signer.extensions <= extensions.keys()
        ):
            raise NativePeerProtocolError(
                "Peer interface or activated extensions are not the selected native contract"
            )
        self._card_bytes = snapshot.SerializeToString()
        self._activated_extensions = signer.extensions
        self._http = http_client
        self._require_http()
        self._scope = recipient
        self._signing_did = signing_did
        self._interface = AgentInterface.FromString(interface.SerializeToString())
        self._store = outbound_store
        if question_store is not None and (
            not isinstance(question_store, NativeQuestionStore)
            or question_store.outbound_store is not outbound_store
        ):
            raise NativePeerProtocolError(
                "Native question intent must use this exact outbound ledger"
            )
        self._questions = question_store
        self._authorize = authorize
        self._signer = signer
        self._client = self._new_sdk_client(streaming=False)

    def _new_sdk_client(self, *, streaming: bool):
        return ClientFactory(
            ClientConfig(
                httpx_client=self._http,
                streaming=streaming,
                polling=False,
                supported_protocol_bindings=[self._interface.protocol_binding],
            )
        ).create(self.agent_card, interceptors=[self._signer])

    def _require_http(self) -> None:
        safe_headers = {
            "accept",
            "accept-encoding",
            "connection",
            "user-agent",
            "a2a-version",
        }
        if (
            self._http.follow_redirects
            or self._http.auth is not None
            or self._http.cookies
            or set(self._http.headers) - safe_headers
            or self._http.headers.get(VERSION_HEADER) not in {None, "1.0"}
        ):
            raise NativePeerProtocolError(
                "Peer HTTP client must exclude redirects and ambient credentials"
            )

    @property
    def recipient_scope(self) -> NativeRecipientScope:
        """The verified, host-selected recipient and interface tenant."""
        return self._scope

    @asynccontextmanager
    async def _network(self, *, before_dispatch=None):
        with bind_native_http_execution_admission(
            self._http, before_dispatch or self._admit
        ):
            async with native_execution_response_owner(self._http):
                try:
                    yield
                except (ParseError, ValueError, TypeError) as error:
                    # The SDK's protobuf/JSON-RPC decoders can reject a peer
                    # body without producing an A2AError. Keep that wire
                    # failure distinct from storage and execution control.
                    if is_execution_control_error(error) or isinstance(
                        error,
                        (
                            NativePeerProtocolError,
                            NativeSenderProofError,
                            OutboundDispatchConflict,
                        ),
                    ):
                        raise
                    raise NativePeerProtocolError(
                        "Official A2A SDK could not decode the peer response"
                    ) from error

    @property
    def agent_card(self) -> AgentCard:
        return AgentCard.FromString(self._card_bytes)

    @property
    def activated_extensions(self) -> frozenset[str]:
        return self._activated_extensions

    async def _admit(self, dispatch: OutboundDispatch | None = None) -> None:
        if dispatch is not None:
            self._require_binding(dispatch)
            if await self._store.get(dispatch.dispatch_id) != dispatch:
                raise OutboundDispatchConflict(
                    "Retained dispatch is not the current committed ledger row; cannot be submitted again"
                )
        await self._authorize()
        self._require_http()

    def _require_binding(self, dispatch: OutboundDispatch) -> None:
        if (
            dispatch.recipient_id != self._scope.recipient_id
            or dispatch.tenant != self._scope.tenant
            or dispatch.signing_did != self._signing_did
            or dispatch.interface_url != self._interface.url
            or dispatch.protocol_binding != self._interface.protocol_binding
        ):
            raise OutboundDispatchConflict(
                "Retained dispatch belongs to another peer binding"
            )

    def _validate_message(self, wire: SendMessageRequest, verb: str):
        card = self.agent_card
        metadata = MessageToDict(wire.metadata)
        try:
            require_metadata_extensions(metadata, self._activated_extensions)
            claims = parse_message_metadata(metadata)
            correlation = claims.correlation
        except ValueError as error:
            raise NativePeerProtocolError(str(error)) from error
        if (correlation is not None) != (verb == "question"):
            raise NativePeerProtocolError(
                "Native question creation requires its correlation contract"
            )
        hint = claims.skill_hint
        modes = card.default_input_modes
        if hint is not None:
            selected = next((skill for skill in card.skills if skill.id == hint), None)
            if selected is None:
                raise NativePeerProtocolError(
                    "Peer does not advertise the suggested skill"
                )
        validate_input_parts(wire.message.parts, modes)
        return claims

    @staticmethod
    def _request_context(
        operation: str, request: ProtoMessage, *, digest: str | None = None
    ):
        return ClientCallContext(
            state={
                _REQUEST_BINDING_KEY: _NativeRequestBinding(
                    operation,
                    native_request_digest(request, operation)
                    if digest is None
                    else digest,
                ),
            }
        )

    @execution_work_operation
    async def reserve(
        self, request: SendMessageRequest, *, dispatch_tool: str, verb: str
    ) -> tuple[OutboundDispatch, SendMessageRequest]:
        await self._admit()
        wire = SendMessageRequest.FromString(request.SerializeToString())
        # The SDK touches configuration before interception. Make its presence
        # explicit before hashing, so the reservation binds exactly what is sent.
        wire.configuration.SetInParent()
        self._validate_message(wire, verb)
        dispatch = await self._store.reserve(
            uuid4().hex,
            wire,
            recipient=self._scope,
            signing_did=self._signing_did,
            interface=self._interface,
            dispatch_tool=dispatch_tool,
            verb=verb,
        )
        return dispatch, wire

    @execution_work_operation
    async def send(
        self, dispatch: OutboundDispatch, request: SendMessageRequest
    ) -> NativeDispatchReceipt:
        with self._store.active_submission(dispatch):
            return await self._send_once(dispatch, request)

    async def _send_once(
        self, dispatch: OutboundDispatch, request: SendMessageRequest
    ) -> NativeDispatchReceipt:
        self._require_binding(dispatch)
        wire = SendMessageRequest.FromString(request.SerializeToString())
        self._validate_message(wire, dispatch.verb)
        if native_request_digest(wire, "SendMessage") != dispatch.request_digest:
            raise OutboundDispatchConflict(
                "Outbound request differs from its reservation"
            )
        await self._admit(dispatch)
        if dispatch.verb == "question":
            if self._questions is None:
                raise NativePeerProtocolError(
                    "Native question dispatch requires its primary correlation ledger"
                )
            await self._questions.require_intent(dispatch, wire)
        submitting = await self._store.begin_submission(
            dispatch, operation_id=uuid4().hex
        )
        if submitting is None:
            raise OutboundDispatchConflict(
                "Dispatch cannot be submitted again; reconcile delivery with peer reads"
            )
        dispatch_claimed = False
        may_have_sent = False

        async def authorize_original_send():
            nonlocal dispatch_claimed, may_have_sent
            if dispatch_claimed:
                raise OutboundDispatchConflict(
                    "The SDK cannot replay the original native send"
                )
            dispatch_claimed = True
            await self._admit()
            if dispatch.verb == "question":
                await self._questions.require_intent(submitting, wire)
            # No await separates this observation from the original native
            # dispatcher. False proves this attempt never admitted peer HTTP;
            # True is only evidence that delivery may have occurred.
            may_have_sent = True

        receipt = None
        try:
            response = None
            async with (
                self._network(before_dispatch=authorize_original_send),
                owned_execution_stream(
                    self,
                    self._client.send_message(
                        wire,
                        context=self._request_context(
                            "SendMessage", wire, digest=dispatch.request_digest
                        ),
                    ),
                ) as events,
            ):
                async for event in events:
                    kind = event.WhichOneof("payload")
                    if response is not None or kind not in {"task", "message"}:
                        raise NativePeerProtocolError(
                            "Non-streaming peer response must contain one Task or Message"
                        )
                    value = getattr(event, kind)
                    response = type(value).FromString(value.SerializeToString())
            if response is None:
                raise NativePeerProtocolError("Peer returned no Task or Message")
            await self._admit()
            receipt = await self._acknowledge(submitting, response)
            if receipt is None:
                raise OutboundDispatchConflict(
                    "Peer receipt lost its dispatch compare-and-swap"
                )
            if (
                not wire.configuration.return_immediately
                and isinstance(response, Task)
                and response.status.state not in STREAM_STOP_STATES
            ):
                raise NativePeerProtocolError(
                    "Peer returned an active Task before its blocking send completed"
                )
            return NativeDispatchReceipt(receipt, response)
        finally:
            if receipt is None:
                await self._retain_failed_mutation(
                    (
                        self._store.mark_unknown
                        if may_have_sent
                        else self._store.record_no_delivery
                    )(submitting, operation_id=uuid4().hex),
                    original=sys.exc_info()[1],
                )

    async def _acknowledge(self, dispatch, response):
        if dispatch.verb == "question":
            if self._questions is None:
                raise NativePeerProtocolError(
                    "Native question receipt requires its correlation ledger"
                )
            return await self._questions.acknowledge(
                dispatch, response, operation_id=uuid4().hex
            )
        return await self._store.acknowledge(
            dispatch, response, operation_id=uuid4().hex
        )

    @execution_work_operation
    async def get_task(self, dispatch: OutboundDispatch) -> NativeDispatchReceipt:
        self._require_binding(dispatch)
        if not dispatch.remote_task_id:
            raise OutboundDispatchConflict(
                "Dispatch has no acknowledged remote task; reconcile first"
            )
        await self._admit(dispatch)
        request = GetTaskRequest(id=dispatch.remote_task_id, tenant=self._scope.tenant)
        async with self._network():
            result = await self._client.get_task(
                request,
                context=self._request_context("GetTask", request),
            )
        await self._admit()
        if (
            result.id != dispatch.remote_task_id
            or result.context_id != dispatch.remote_context_id
        ):
            raise NativePeerProtocolError(
                "Peer read changed the retained task or context"
            )
        observed = await self._acknowledge(dispatch, result)
        if observed is None:
            raise OutboundDispatchConflict("Retained dispatch changed during task read")
        return NativeDispatchReceipt(observed, result)

    @execution_work_operation
    async def cancel_task(self, dispatch: OutboundDispatch) -> NativeDispatchReceipt:
        self._require_binding(dispatch)
        if not dispatch.remote_task_id:
            raise OutboundDispatchConflict(
                "Dispatch has no acknowledged remote task to cancel"
            )
        await self._admit(dispatch)
        request = CancelTaskRequest(
            id=dispatch.remote_task_id, tenant=self._scope.tenant
        )
        submitting = await self._store.begin_cancellation(
            dispatch, request, operation_id=uuid4().hex
        )
        if submitting is None:
            raise OutboundDispatchConflict("Cancellation lost its dispatch claim")
        dispatch_claimed = False
        may_have_sent = False
        receipt = None

        async def authorize_cancel():
            nonlocal dispatch_claimed, may_have_sent
            if dispatch_claimed:
                raise OutboundDispatchConflict(
                    "The SDK cannot replay native cancellation"
                )
            dispatch_claimed = True
            await self._admit(submitting)
            may_have_sent = True

        try:
            async with self._network(before_dispatch=authorize_cancel):
                result = await self._client.cancel_task(
                    request,
                    context=self._request_context(
                        "CancelTask", request, digest=submitting.cancel_request_digest
                    ),
                )
            await self._admit()
            if (
                result.id != dispatch.remote_task_id
                or result.context_id != dispatch.remote_context_id
                or result.status.state != TaskState.TASK_STATE_CANCELED
            ):
                raise NativePeerProtocolError(
                    "Peer cancellation returned a different task, context or outcome"
                )
            receipt = await self._acknowledge(submitting, result)
            if receipt is None:
                raise OutboundDispatchConflict(
                    "Cancellation receipt lost its dispatch claim"
                )
            return NativeDispatchReceipt(receipt, result)
        finally:
            if receipt is None:
                await self._retain_failed_mutation(
                    self._store.retain_cancellation_failure(
                        submitting,
                        may_have_sent=may_have_sent,
                        operation_id=uuid4().hex,
                    ),
                    original=sys.exc_info()[1],
                )

    @staticmethod
    async def _retain_failed_mutation(
        settle, *, original: BaseException | None
    ) -> None:
        """Retain delivery uncertainty without replacing the original control error."""
        outcome = await await_owned_task(
            create_owned_task(
                settle, name="a2a-retain-mutation-outcome", retain_originals=True
            )
        )
        terminal = execution_terminal_error(
            original, outcome.error, outcome.cancellation, retain_originals=True
        )
        if terminal is not None and terminal is not original:
            raise terminal

    @execution_work_operation
    async def reconcile(self, dispatch: OutboundDispatch) -> OutboundDispatch | None:
        """Search authorized history; never resubmit an uncertain original."""
        self._require_binding(dispatch)
        await self._admit(dispatch)
        if self._store.submission_active(dispatch.dispatch_id):
            raise OutboundDispatchConflict(
                "Original submission is still active; reconciliation must wait"
            )
        if dispatch.state is DispatchState.SUBMITTING:
            dispatch = await self._store.mark_unknown(
                dispatch, operation_id=uuid4().hex
            )
            if dispatch is None:
                raise OutboundDispatchConflict(
                    "Submission evidence changed during reconciliation"
                )
        if dispatch.state is not DispatchState.UNKNOWN:
            raise OutboundDispatchConflict("Only uncertain delivery can be reconciled")
        candidates = []
        token = ""
        seen_tokens = set()
        while True:
            await self._admit()
            request = ListTasksRequest(
                tenant=self._scope.tenant,
                context_id=dispatch.requested_context_id,
                page_token=token,
            )
            async with self._network():
                result = await self._client.list_tasks(
                    request,
                    context=self._request_context("ListTasks", request),
                )
            await self._admit()
            candidates.extend(result.tasks)
            token = result.next_page_token
            if not token:
                break
            if token in seen_tokens:
                raise NativePeerProtocolError(
                    "Peer repeated a task-list pagination token"
                )
            seen_tokens.add(token)
        if dispatch.verb != "question":
            return await self._store.reconcile(
                dispatch, candidates, operation_id=uuid4().hex
            )
        if self._questions is None:
            raise NativePeerProtocolError(
                "Native question reconciliation requires its correlation ledger"
            )
        response = self._store.reconciliation_candidate(dispatch, candidates)
        if response is None:
            return None
        return await self._acknowledge(dispatch, response)

    @execution_work_stream
    async def subscribe(self, dispatch: OutboundDispatch):
        """Consume official SDK events, retaining task identity and original custody."""
        await self._admit(dispatch)
        if not dispatch.remote_task_id or not self.agent_card.capabilities.streaming:
            raise NativePeerProtocolError(
                "Native subscription requires an acknowledged task and advertised streaming"
            )
        request = SubscribeToTaskRequest(
            id=dispatch.remote_task_id, tenant=self._scope.tenant
        )
        client = self._new_sdk_client(streaming=True)
        try:
            async with (
                self._network(),
                owned_execution_stream(
                    self,
                    client.subscribe(
                        request,
                        context=self._request_context("SubscribeToTask", request),
                    ),
                ) as events,
            ):
                async for event in events:
                    await self._admit()
                    kind = event.WhichOneof("payload")
                    if kind not in {"task", "status_update", "artifact_update"}:
                        raise NativePeerProtocolError(
                            "Native task subscription returned an unsupported event"
                        )
                    value = getattr(event, kind)
                    validate_proto_required_fields(value)
                    task_id = value.id if kind == "task" else value.task_id
                    if (
                        task_id != dispatch.remote_task_id
                        or value.context_id != dispatch.remote_context_id
                    ):
                        raise NativePeerProtocolError(
                            "Native subscription changed the retained task or context"
                        )
                    yield StreamResponse.FromString(event.SerializeToString())
        finally:
            await self._retain_failed_mutation(
                client.close(), original=sys.exc_info()[1]
            )

    async def close(self) -> None:
        """Retire the SDK facade; its injected HTTP client remains host-owned."""
        await self._client.close()
