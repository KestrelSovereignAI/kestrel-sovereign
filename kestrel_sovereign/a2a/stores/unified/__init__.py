"""
Unified A2A Stores - Backend-Agnostic Implementations

These stores work with both SQLite and PostgreSQL through the DatabaseBackend
abstraction. No more duplicated code!

Usage:
    from kestrel_sovereign.storage.db import SQLiteBackend, PostgresBackend
    from kestrel_sovereign.a2a.stores.unified import TaskStore, SessionService

    # SQLite (for sovereign Kestrel agents)
    backend = SQLiteBackend("/path/to/db.sqlite")
    await backend.connect()

    # PostgreSQL (for multi-tenant deployment)
    backend = PostgresBackend(dsn="postgresql://...")
    await backend.connect()

    # Same store works with either backend!
    task_store = TaskStore(backend)
    await task_store.initialize()
"""

from kestrel_sovereign.a2a.stores.unified.task_store import TaskStore
from kestrel_sovereign.a2a.stores.unified.session_service import SessionService
from kestrel_sovereign.a2a.stores.unified.memory_service import MemoryService
from kestrel_sovereign.a2a.stores.unified.observability_store import (
    ObservabilityStore,
    ObservabilityEvent,
    LLMCallEvent,
)
from kestrel_sovereign.a2a.stores.unified.orchestration_store import OrchestrationStore
from kestrel_sovereign.a2a.stores.unified.feedback_store import FeedbackStore

__all__ = [
    "TaskStore",
    "SessionService",
    "MemoryService",
    "ObservabilityStore",
    "ObservabilityEvent",
    "LLMCallEvent",
    "OrchestrationStore",
    "FeedbackStore",
]
