"""
SessionService - Session State and Event History.

This module provides SQLite-backed A2A session management using the unified
backend-agnostic store implementation with SQLiteBackend.

For new code, use the unified store directly:
    from kestrel_sovereign.storage.db import SQLiteBackend
    from kestrel_sovereign.a2a.stores.unified import SessionService

    backend = SQLiteBackend(db_path)
    await backend.connect()
    session_service = SessionService(backend)

The SQLiteSessionService class here is a thin wrapper for backward compatibility
with existing Kestrel code that uses db_path strings directly.
"""

import logging

from kestrel_sovereign.storage.db import SQLiteBackend

# Import the unified SessionService (backend-agnostic implementation)
from kestrel_sovereign.a2a.stores.unified import SessionService as UnifiedSessionService

# Re-export the SessionState model for backward compatibility
from kestrel_sovereign.a2a.stores.unified.session_service import SessionState

logger = logging.getLogger(__name__)


class SQLiteSessionService(UnifiedSessionService):
    """
    SQLite-backed session service for sovereign Kestrel agents.

    This is a thin wrapper around the unified SessionService that accepts
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
SessionService = SQLiteSessionService

__all__ = [
    "SessionService",
    "SQLiteSessionService",
    "SessionState",
]
