"""
OrchestrationStore - Multi-Agent Workflow Coordination.

This module provides SQLite-backed A2A workflow orchestration using the unified
backend-agnostic store implementation with SQLiteBackend.

For new code, use the unified store directly:
    from kestrel_sovereign.storage.db import SQLiteBackend
    from kestrel_sovereign.a2a.stores.unified import OrchestrationStore

    backend = SQLiteBackend(db_path)
    await backend.connect()
    orch_store = OrchestrationStore(backend)

The SQLiteOrchestrationStore class here is a thin wrapper for backward
compatibility with existing Kestrel code that uses db_path strings directly.
"""

import logging

from kestrel_sovereign.storage.db import SQLiteBackend

# Import the unified OrchestrationStore (backend-agnostic implementation)
from kestrel_sovereign.a2a.stores.unified import OrchestrationStore as UnifiedOrchestrationStore

# Re-export models for backward compatibility
from kestrel_sovereign.a2a.stores.unified.orchestration_store import OrchestrationTask, OrchestrationStatus

logger = logging.getLogger(__name__)


class SQLiteOrchestrationStore(UnifiedOrchestrationStore):
    """
    SQLite-backed orchestration store for multi-agent workflows.

    This is a thin wrapper around the unified OrchestrationStore that accepts
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
        if not self._backend.is_connected:
            await self._backend.connect()
        await super().initialize()


# Backward compatibility alias
OrchestrationStore = SQLiteOrchestrationStore

__all__ = [
    "OrchestrationStore",
    "SQLiteOrchestrationStore",
    "OrchestrationTask",
    "OrchestrationStatus",
]
