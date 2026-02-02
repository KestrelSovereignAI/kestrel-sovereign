"""
A2A (Agent-to-Agent) Protocol Implementation for Kestrel.

This module provides full A2A protocol compliance with:
- 6 Core Datastores (Task, Session, Memory, Observability, Orchestration, Feedback)
- Async task management with SSE streaming
- Agent Card for capability discovery
- JSON-RPC compatible message types
- TaskManager for high-level task operations
- TaskWorker for background processing

"""

from kestrel_sovereign.a2a.types import (
    # Enums
    TaskState,
    # Parts
    TextPart,
    FilePart,
    DataPart,
    FileContent,
    Part,
    # Messages
    Message,
    # Task types
    Task,
    TaskStatus,
    Artifact,
    TaskStatusUpdateEvent,
    TaskArtifactUpdateEvent,
    # Request/Response params
    TaskSendParams,
    TaskIdParams,
    TaskQueryParams,
    # JSON-RPC
    JSONRPCMessage,
    JSONRPCRequest,
    JSONRPCResponse,
    JSONRPCError,
)

from kestrel_sovereign.a2a.agent_card import (
    AgentCard,
    AgentSkill,
    AgentCapabilities,
    AgentProvider,
    AgentAuthentication,
)

from kestrel_sovereign.a2a.task_manager import (
    TaskManager,
    create_task_manager,
)

from kestrel_sovereign.a2a.task_worker import (
    TaskWorker,
    TaskHandler,
    TaskResult,
    SimpleTaskHandler,
    LLMTaskHandler,
    create_task_worker,
)

__all__ = [
    # Enums
    "TaskState",
    # Parts
    "TextPart",
    "FilePart",
    "DataPart",
    "FileContent",
    "Part",
    # Messages
    "Message",
    # Task types
    "Task",
    "TaskStatus",
    "Artifact",
    "TaskStatusUpdateEvent",
    "TaskArtifactUpdateEvent",
    # Request/Response params
    "TaskSendParams",
    "TaskIdParams",
    "TaskQueryParams",
    # JSON-RPC
    "JSONRPCMessage",
    "JSONRPCRequest",
    "JSONRPCResponse",
    "JSONRPCError",
    # Agent Card
    "AgentCard",
    "AgentSkill",
    "AgentCapabilities",
    "AgentProvider",
    "AgentAuthentication",
    # Task Manager
    "TaskManager",
    "create_task_manager",
    # Task Worker
    "TaskWorker",
    "TaskHandler",
    "TaskResult",
    "SimpleTaskHandler",
    "LLMTaskHandler",
    "create_task_worker",
]
