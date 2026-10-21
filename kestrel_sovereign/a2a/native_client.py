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
    Task,
)
from a2a.utils.constants import VERSION_HEADER
from a2a.utils.proto_utils import validate_proto_required_fields
from google.protobuf.json_format import MessageToDict
from google.protobuf.message import Message as ProtoMessage

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.a2a.extensions import (
    CAUSATION_URI,
    QUESTION_CORRELATION_URI,
    SIGNED_SENDER_URI,
)
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
from kestrel_sovereign.execution_custody import (
    bind_native_http_execution_admission,
    execution_terminal_error,
    execution_work_operation,
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
    ):
        snapshot = AgentCard.FromString(card.SerializeToString())
        validate_proto_required_fields(snapshot)
        verify_native_agent_card(
            snapshot, signing_did=signing_did, trusted_document=trusted_document
        )
        extensions = {item.uri: item for item in snapshot.capabilities.extensions}
        supported = REQUIRED_NATIVE_EXTENSIONS | {
            CAUSATION_URI,
            QUESTION_CORRELATION_URI,
        }
        versions = [
            MessageToDict(item.params).get("version")
            for uri, item in extensions.items()
            if uri in REQUIRED_NATIVE_EXTENSIONS
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
        self._http = http_client
        self._require_http()
        self._scope = recipient
        self._signing_did = signing_did
        self._interface = AgentInterface.FromString(interface.SerializeToString())
        self._store = outbound_store
        self._authorize = authorize
        self._client = ClientFactory(
            ClientConfig(
                httpx_client=http_client,
                streaming=False,
                polling=False,
                supported_protocol_bindings=[interface.protocol_binding],
            )
        ).create(snapshot, interceptors=[signer])

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
    async def _network(self):
        with bind_native_http_execution_admission(self._http, self._admit):
            async with native_execution_response_owner(self._http):
                yield

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

    @execution_work_operation
    async def reserve(
        self, request: SendMessageRequest, *, dispatch_tool: str, verb: str
    ) -> tuple[OutboundDispatch, SendMessageRequest]:
        await self._admit()
        wire = SendMessageRequest.FromString(request.SerializeToString())
        # The SDK touches configuration before interception. Make its presence
        # explicit before hashing, so the reservation binds exactly what is sent.
        wire.configuration.SetInParent()
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
        self._require_binding(dispatch)
        wire = SendMessageRequest.FromString(request.SerializeToString())
        if native_request_digest(wire, "SendMessage") != dispatch.request_digest:
            raise OutboundDispatchConflict(
                "Outbound request differs from its reservation"
            )
        await self._admit(dispatch)
        submitting = await self._store.begin_submission(
            dispatch, operation_id=uuid4().hex
        )
        if submitting is None:
            raise OutboundDispatchConflict(
                "Dispatch cannot be submitted again; reconcile delivery with peer reads"
            )
        receipt = None
        try:
            response = None
            async with (
                self._network(),
                owned_execution_stream(self, self._client.send_message(wire)) as events,
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
            receipt = await self._store.acknowledge(
                submitting, response, operation_id=uuid4().hex
            )
            if receipt is None:
                raise OutboundDispatchConflict(
                    "Peer receipt lost its dispatch compare-and-swap"
                )
            return NativeDispatchReceipt(receipt, response)
        finally:
            if receipt is None:
                original = sys.exc_info()[1]
                outcome = await await_owned_task(
                    create_owned_task(
                        self._store.mark_unknown(submitting, operation_id=uuid4().hex),
                        name="a2a-retain-unknown-delivery",
                        retain_originals=True,
                    )
                )
                terminal = execution_terminal_error(
                    original, outcome.error, outcome.cancellation, retain_originals=True
                )
                if terminal is not None and terminal is not original:
                    raise terminal

    @execution_work_operation
    async def get_task(self, dispatch: OutboundDispatch) -> Task:
        self._require_binding(dispatch)
        if not dispatch.remote_task_id:
            raise OutboundDispatchConflict(
                "Dispatch has no acknowledged remote task; reconcile first"
            )
        await self._admit(dispatch)
        async with self._network():
            result = await self._client.get_task(
                GetTaskRequest(
                    id=dispatch.remote_task_id,
                    tenant=self._scope.tenant,
                )
            )
        await self._admit()
        if (
            result.id != dispatch.remote_task_id
            or result.context_id != dispatch.remote_context_id
        ):
            raise NativePeerProtocolError(
                "Peer read changed the retained task or context"
            )
        return result

    @execution_work_operation
    async def cancel_task(self, dispatch: OutboundDispatch) -> Task:
        self._require_binding(dispatch)
        if not dispatch.remote_task_id:
            raise OutboundDispatchConflict(
                "Dispatch has no acknowledged remote task to cancel"
            )
        await self._admit(dispatch)
        async with self._network():
            result = await self._client.cancel_task(
                CancelTaskRequest(
                    id=dispatch.remote_task_id,
                    tenant=self._scope.tenant,
                )
            )
        await self._admit()
        if (
            result.id != dispatch.remote_task_id
            or result.context_id != dispatch.remote_context_id
        ):
            raise NativePeerProtocolError(
                "Peer cancellation changed the retained task or context"
            )
        return result

    @execution_work_operation
    async def reconcile(self, dispatch: OutboundDispatch) -> OutboundDispatch | None:
        """Search authorized history; never resubmit an uncertain original."""
        self._require_binding(dispatch)
        await self._admit(dispatch)
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
            async with self._network():
                result = await self._client.list_tasks(
                    ListTasksRequest(
                        tenant=self._scope.tenant,
                        context_id=dispatch.requested_context_id,
                        page_token=token,
                    )
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
        return await self._store.reconcile(
            dispatch, candidates, operation_id=uuid4().hex
        )

    async def close(self) -> None:
        """Retire the SDK facade; its injected HTTP client remains host-owned."""
        await self._client.close()
