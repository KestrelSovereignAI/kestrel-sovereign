"""
FeedbackStore - Agent Self-Diagnosis and User Feedback.

This module provides SQLite-backed A2A feedback storage using the unified
backend-agnostic store implementation with SQLiteBackend.

For new code, use the unified store directly:
    from kestrel_sovereign.storage.db import SQLiteBackend
    from kestrel_sovereign.a2a.stores.unified import FeedbackStore

    backend = SQLiteBackend(db_path)
    await backend.connect()
    feedback_store = FeedbackStore(backend)

The SQLiteFeedbackStore class here is a thin wrapper for backward
compatibility with existing Kestrel code that uses db_path strings directly.
"""

import logging

from kestrel_sovereign.storage.db import SQLiteBackend

# Import the unified FeedbackStore (backend-agnostic implementation)
from kestrel_sovereign.a2a.stores.unified import FeedbackStore as UnifiedFeedbackStore

# Re-export models for backward compatibility
from kestrel_sovereign.a2a.stores.unified.feedback_store import (
    FeedbackEntry,
    FeedbackCategory,
    FeedbackSeverity,
    FeedbackStatus,
    FeedbackSource,
)

logger = logging.getLogger(__name__)


class SQLiteFeedbackStore(UnifiedFeedbackStore):
    """
    SQLite-backed feedback store for agent self-diagnosis.

    This is a thin wrapper around the unified FeedbackStore that accepts
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
FeedbackStore = SQLiteFeedbackStore

__all__ = [
    "FeedbackStore",
    "SQLiteFeedbackStore",
    "FeedbackEntry",
    "FeedbackCategory",
    "FeedbackSeverity",
    "FeedbackStatus",
    "FeedbackSource",
]
