"""
TaskStore - Async Task Persistence.

This module provides SQLite-backed A2A task storage using the unified
backend-agnostic store implementation with SQLiteBackend.

For new code, use the unified store directly:
    from kestrel_sovereign.storage.db import SQLiteBackend
    from kestrel_sovereign.a2a.stores.unified import TaskStore

    backend = SQLiteBackend(db_path)
    await backend.connect()
    task_store = TaskStore(backend)

The SQLiteTaskStore class here is a thin wrapper for backward compatibility
with existing Kestrel code that uses db_path strings directly.
"""

import logging

from kestrel_sovereign.storage.db import SQLiteBackend

# Import the unified TaskStore (backend-agnostic implementation)
from kestrel_sovereign.a2a.stores.unified import TaskStore as UnifiedTaskStore

# Re-export types for backward compatibility
from kestrel_sovereign.a2a.types import Task, TaskStatus, TaskState, Artifact, Message

logger = logging.getLogger(__name__)


class SQLiteTaskStore(UnifiedTaskStore):
    """
    SQLite-backed task store for sovereign Kestrel agents.

    This is a thin wrapper around the unified TaskStore that accepts
    a db_path string instead of a DatabaseBackend instance.
    """

    def __init__(self, db_path: str):
        """
        Initialize with SQLite database path.

        Args:
            db_path: Path to SQLite database file
        """
        self._db_path = db_path
        backend = SQLiteBackend(db_path)
        super().__init__(backend)

    async def initialize(self) -> None:
        """Initialize store - connects backend and creates tables."""
        # Ensure backend is connected before initializing tables
        if not self._backend.is_connected:
            await self._backend.connect()
        await super().initialize()


# Backward compatibility alias - existing code uses TaskStore directly
TaskStore = SQLiteTaskStore

__all__ = [
    "TaskStore",
    "SQLiteTaskStore",
    # Re-exported types
    "Task",
    "TaskStatus",
    "TaskState",
    "Artifact",
    "Message",
]
