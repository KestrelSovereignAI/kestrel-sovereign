"""
MemoryService - Long-term Searchable Memory.

This module provides SQLite-backed A2A memory storage using the unified
backend-agnostic store implementation with SQLiteBackend.

For new code, use the unified store directly:
    from kestrel_sovereign.storage.db import SQLiteBackend
    from kestrel_sovereign.a2a.stores.unified import MemoryService

    backend = SQLiteBackend(db_path)
    await backend.connect()
    memory_service = MemoryService(backend)

The SQLiteMemoryService class here is a thin wrapper for backward compatibility
with existing Kestrel code that uses db_path strings directly.
"""

import logging

from kestrel_sovereign.storage.db import SQLiteBackend

# Import the unified MemoryService (backend-agnostic implementation)
from kestrel_sovereign.a2a.stores.unified import MemoryService as UnifiedMemoryService

# Re-export the MemoryEntry model for backward compatibility
from kestrel_sovereign.a2a.stores.unified.memory_service import MemoryEntry

logger = logging.getLogger(__name__)


class SQLiteMemoryService(UnifiedMemoryService):
    """
    SQLite-backed memory with FTS5 for full-text search.

    This is a thin wrapper around the unified MemoryService that accepts
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
MemoryService = SQLiteMemoryService

__all__ = [
    "MemoryService",
    "SQLiteMemoryService",
    "MemoryEntry",
]
