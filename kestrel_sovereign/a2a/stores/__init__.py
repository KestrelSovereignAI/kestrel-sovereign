"""
A2A Datastores - Unified Backend-Agnostic Stores.

The stores use the DatabaseBackend abstraction from kestrel_sovereign.storage.db,
allowing the same store code to work with both SQLite and PostgreSQL.

Usage:
    from kestrel_sovereign.a2a.stores import TaskStore, SessionService, MemoryService, ...
    from kestrel_sovereign.storage.db import SQLiteBackend, PostgresBackend
    from kestrel_sovereign.a2a.scope import NativeRecipientScope

    backend = SQLiteBackend("/path/to/db.sqlite")
    await backend.connect()
    task_store = TaskStore(backend, scope=NativeRecipientScope("did:example:recipient"))

Stores:
1. TaskStore - Async task persistence and lifecycle
2. SessionService - Session state and event history (Google ADK pattern)
3. MemoryService - Long-term searchable memory
4. ObservabilityStore - Telemetry, metrics, error tracking
5. OrchestrationStore - Multi-agent workflow coordination
6. FeedbackStore - Agent self-diagnosis and user feedback
"""

# Import unified stores directly (no "Unified" prefix needed)
from kestrel_sovereign.a2a.stores.unified import (
    TaskStore,
    SessionService,
    MemoryService,
    ObservabilityStore,
    OrchestrationStore,
    FeedbackStore,
)

# Import data models
from kestrel_sovereign.a2a.stores.unified.session_service import SessionState
from kestrel_sovereign.a2a.stores.unified.memory_service import MemoryEntry
from kestrel_sovereign.a2a.stores.unified.observability_store import ObservabilityEvent
from kestrel_sovereign.a2a.stores.unified.orchestration_store import (
    OrchestrationStatus,
    OrchestrationTask,
)
from kestrel_sovereign.a2a.stores.unified.feedback_store import (
    FeedbackCategory,
    FeedbackSeverity,
    FeedbackStatus,
    FeedbackSource,
    FeedbackEntry,
)

__all__ = [
    # Unified stores
    "TaskStore",
    "SessionService",
    "MemoryService",
    "ObservabilityStore",
    "OrchestrationStore",
    "FeedbackStore",
    # Data models
    "SessionState",
    "MemoryEntry",
    "ObservabilityEvent",
    "OrchestrationStatus",
    "OrchestrationTask",
    "FeedbackCategory",
    "FeedbackSeverity",
    "FeedbackStatus",
    "FeedbackSource",
    "FeedbackEntry",
]
