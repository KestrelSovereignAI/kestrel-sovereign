"""
A2A Protocol Types for Kestrel.

Core types for the Agent-to-Agent protocol, compatible with:
- https://a2a-protocol.org/latest/specification/
- Google ADK agent patterns

These types are provider-agnostic - they work with SQLite (sovereign Kestrel)
or PostgreSQL (multi-tenant).
"""

import datetime as dt_module
from enum import Enum
from typing import Annotated, Any, Literal, Self
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    model_validator,
)


# =============================================================================
# JSON-RPC Base Classes
# =============================================================================

class JSONRPCMessage(BaseModel):
    """Base JSON-RPC 2.0 message."""
    jsonrpc: Literal['2.0'] = '2.0'
    id: int | str | None = Field(default_factory=lambda: uuid4().hex)


class JSONRPCRequest(JSONRPCMessage):
    """JSON-RPC request message."""
    method: str
    params: dict[str, Any] | None = None


class JSONRPCError(BaseModel):
    """JSON-RPC error object."""
    code: int
    message: str
    data: Any | None = None


class JSONRPCResponse(JSONRPCMessage):
    """JSON-RPC response message."""
    result: Any | None = None
    error: JSONRPCError | None = None


# =============================================================================
# Task States
# =============================================================================

class TaskState(str, Enum):
    """Task lifecycle states."""
    SUBMITTED = 'submitted'
    WORKING = 'working'
    INPUT_REQUIRED = 'input-required'
    COMPLETED = 'completed'
    CANCELED = 'canceled'
    FAILED = 'failed'
    UNKNOWN = 'unknown'


# =============================================================================
# Message Parts (Text, File, Data)
# =============================================================================

class TextPart(BaseModel):
    """Text content in a message."""
    type: Literal['text'] = 'text'
    text: str
    metadata: dict[str, Any] | None = None


class FileContent(BaseModel):
    """File content - either bytes (base64) or URI."""
    name: str | None = None
    mimeType: str | None = None
    bytes: str | None = None  # base64 encoded
    uri: str | None = None

    @model_validator(mode='after')
    def check_content(self) -> Self:
        if not (self.bytes or self.uri):
            raise ValueError(
                "Either 'bytes' or 'uri' must be present in the file data"
            )
        if self.bytes and self.uri:
            raise ValueError(
                "Only one of 'bytes' or 'uri' can be present in the file data"
            )
        return self


class FilePart(BaseModel):
    """File content in a message."""
    type: Literal['file'] = 'file'
    file: FileContent
    metadata: dict[str, Any] | None = None


class DataPart(BaseModel):
    """Structured data in a message."""
    type: Literal['data'] = 'data'
    data: dict[str, Any]
    metadata: dict[str, Any] | None = None


# Union type for message parts with discriminator
Part = Annotated[TextPart | FilePart | DataPart, Field(discriminator='type')]


# =============================================================================
# Messages
# =============================================================================

class Message(BaseModel):
    """A message in a conversation."""
    role: Literal['user', 'agent']
    parts: list[Part]
    metadata: dict[str, Any] | None = None


# =============================================================================
# Task Types
# =============================================================================

class TaskStatus(BaseModel):
    """Current status of a task."""
    state: TaskState
    message: Message | None = None
    timestamp: dt_module.datetime = Field(
        default_factory=lambda: dt_module.datetime.now(dt_module.timezone.utc)
    )

    @field_serializer('timestamp')
    def serialize_dt_to_isoformat(self, dt: dt_module.datetime, _info):
        return dt.isoformat()


class Artifact(BaseModel):
    """Output artifact from a completed task."""
    name: str | None = None
    description: str | None = None
    parts: list[Part]
    metadata: dict[str, Any] | None = None
    index: int = 0
    append: bool | None = None
    lastChunk: bool | None = None


class Task(BaseModel):
    """A task in the A2A system."""
    id: str
    sessionId: str | None = None
    status: TaskStatus
    artifacts: list[Artifact] | None = None
    history: list[Message] | None = None
    metadata: dict[str, Any] | None = None


# =============================================================================
# Task Events (for SSE streaming)
# =============================================================================

class TaskStatusUpdateEvent(BaseModel):
    """Event for task status changes."""
    id: str
    status: TaskStatus
    final: bool = False
    metadata: dict[str, Any] | None = None


class TaskArtifactUpdateEvent(BaseModel):
    """Event for new task artifacts."""
    id: str
    artifact: Artifact
    metadata: dict[str, Any] | None = None


# =============================================================================
# Request/Response Parameters
# =============================================================================

class TaskIdParams(BaseModel):
    """Parameters for task ID operations."""
    id: str
    metadata: dict[str, Any] | None = None


class TaskQueryParams(TaskIdParams):
    """Parameters for task queries."""
    historyLength: int | None = None


class PushNotificationConfig(BaseModel):
    """Configuration for push notifications."""
    url: str
    token: str | None = None


class TaskSendParams(BaseModel):
    """Parameters for sending/creating a task."""
    id: str = Field(default_factory=lambda: uuid4().hex)
    sessionId: str = Field(default_factory=lambda: uuid4().hex)
    message: Message
    acceptedOutputModes: list[str] | None = None
    pushNotification: PushNotificationConfig | None = None
    historyLength: int | None = None
    metadata: dict[str, Any] | None = None


# =============================================================================
# JSON-RPC Error Types
# =============================================================================

class JSONParseError(JSONRPCError):
    """Invalid JSON payload."""
    code: int = -32700
    message: str = 'Invalid JSON payload'


class InvalidRequestError(JSONRPCError):
    """Request payload validation error."""
    code: int = -32600
    message: str = 'Request payload validation error'


class MethodNotFoundError(JSONRPCError):
    """Method not found."""
    code: int = -32601
    message: str = 'Method not found'


class InvalidParamsError(JSONRPCError):
    """Invalid parameters."""
    code: int = -32602
    message: str = 'Invalid parameters'


class InternalError(JSONRPCError):
    """Internal error."""
    code: int = -32603
    message: str = 'Internal error'


# A2A Specific Errors

class TaskNotFoundError(JSONRPCError):
    """Task not found."""
    code: int = -32001
    message: str = 'Task not found'


class TaskNotCancelableError(JSONRPCError):
    """Task cannot be canceled."""
    code: int = -32002
    message: str = 'Task cannot be canceled'


class UnsupportedOperationError(JSONRPCError):
    """Operation not supported."""
    code: int = -32004
    message: str = 'This operation is not supported'
