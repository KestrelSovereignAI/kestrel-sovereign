"""
A2A Protocol Types for Kestrel.

Re-exports from kestrel_sdk.a2a.types for backward compatibility.
Feature packages should import from kestrel_sdk.a2a.types directly.
"""

# Re-export everything from kestrel_sdk
from kestrel_sdk.a2a.types import (  # noqa: F401
    JSONRPCMessage,
    JSONRPCRequest,
    JSONRPCError,
    JSONRPCResponse,
    TaskState,
    TextPart,
    FileContent,
    FilePart,
    DataPart,
    Part,
    Message,
    TaskStatus,
    Artifact,
    Task,
    TaskStatusUpdateEvent,
    TaskArtifactUpdateEvent,
    TaskIdParams,
    TaskQueryParams,
    PushNotificationConfig,
    TaskSendParams,
    JSONParseError,
    InvalidRequestError,
    MethodNotFoundError,
    InvalidParamsError,
    InternalError,
    TaskNotFoundError,
    TaskNotCancelableError,
    UnsupportedOperationError,
)
