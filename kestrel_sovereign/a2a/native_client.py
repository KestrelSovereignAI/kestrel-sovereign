"""Signed-sender extension hook for the official SDK client."""

from collections.abc import Callable
from dataclasses import dataclass, field

from a2a.client.client import ClientCallContext
from a2a.client.interceptors import AfterArgs, BeforeArgs, ClientCallInterceptor
from a2a.extensions.common import HTTP_EXTENSION_HEADER
from google.protobuf.message import Message as ProtoMessage

from kestrel_sovereign.a2a.native_sender import (
    SENDER_PROOF_HEADER,
    NativeProofPolicy,
    NativeSenderProofError,
    sign_native_request,
)
from kestrel_sovereign.a2a.scope import NativeRecipientScope
from kestrel_sovereign.identity.hybrid_keypair import HybridKeypair

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
