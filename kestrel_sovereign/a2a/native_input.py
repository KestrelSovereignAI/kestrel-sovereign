"""Canonical native request digests and callback-free task input."""

import hashlib
from dataclasses import dataclass, field

import rfc8785
from kestrel_sdk.native_a2a.types import SendMessageRequest, TaskPushNotificationConfig
from kestrel_sdk.native_a2a.utils.errors import InvalidParamsError
from kestrel_sdk.native_a2a.utils.proto_utils import validate_proto_required_fields
from google.protobuf.json_format import MessageToDict, MessageToJson
from google.protobuf.message import Message as ProtoMessage


class NativeSenderProofError(ValueError):
    """The native sender proof cannot authenticate this request."""


def native_request_digest(params: ProtoMessage, operation: str) -> str:
    """Hash the complete native request for its actual SDK service operation."""
    if not isinstance(params, ProtoMessage):
        raise TypeError("Native A2A proof requires an official protobuf request")
    methods = SendMessageRequest.DESCRIPTOR.file.services_by_name[
        "A2AService"
    ].methods_by_name
    method = methods.get(operation)
    if method is None or params.DESCRIPTOR is not method.input_type:
        raise NativeSenderProofError(
            "A2A operation does not match its native request type"
        )
    validate_proto_required_fields(params)
    return hashlib.sha256(rfc8785.dumps(MessageToDict(params))).hexdigest()


@dataclass(frozen=True)
class NativeTaskInput:
    """An immutable execution projection and its original signed-request digest.

    The callback is transient secret material for the configuration service.
    Only ``request_json`` and the digests belong in task-input storage. No
    private protocol model is introduced: both projections remain official
    SDK protobuf values, returned as independent copies.
    """

    _execution_bytes: bytes = field(repr=False)
    wire_request_digest: str
    _callback_bytes: bytes | None = field(repr=False)

    @property
    def request(self) -> SendMessageRequest:
        return SendMessageRequest.FromString(self._execution_bytes)

    @property
    def request_json(self) -> str:
        return MessageToJson(self.request)

    @property
    def execution_digest(self) -> str:
        return hashlib.sha256(self._execution_bytes).hexdigest()

    @property
    def push_configuration(self) -> TaskPushNotificationConfig | None:
        if self._callback_bytes is None:
            return None
        return TaskPushNotificationConfig.FromString(self._callback_bytes)


def prepare_native_task_input(request: SendMessageRequest) -> NativeTaskInput:
    """Retain the complete digest before removing callback credentials from work.

    Call after native request verification. Callback URL/authentication policy
    and encrypted registration remain the configuration service's concerns;
    this pure function never performs I/O or authorizes an external request.
    """
    if not isinstance(request, SendMessageRequest):
        raise TypeError("Task input requires an official SendMessageRequest")
    execution = SendMessageRequest.FromString(request.SerializeToString())
    wire_digest = native_request_digest(execution, "SendMessage")
    # The executor requires nonempty text. Apply that business constraint to
    # both first inputs and continuations before callback preparation or writes.
    if any(part.HasField("text") and not part.text for part in execution.message.parts):
        raise InvalidParamsError(message="Native task input text cannot be empty")
    callback = None
    if execution.configuration.HasField("task_push_notification_config"):
        config = execution.configuration.task_push_notification_config
        if config.task_id:
            raise InvalidParamsError(
                message="Inline push configuration must leave taskId empty"
            )
        if config.tenant and config.tenant != execution.tenant:
            raise InvalidParamsError(
                message="Inline push configuration belongs to another tenant"
            )
        copied = TaskPushNotificationConfig.FromString(config.SerializeToString())
        copied.tenant = execution.tenant
        callback = copied.SerializeToString(deterministic=True)
        execution.configuration.ClearField("task_push_notification_config")
    return NativeTaskInput(
        execution.SerializeToString(deterministic=True), wire_digest, callback
    )
