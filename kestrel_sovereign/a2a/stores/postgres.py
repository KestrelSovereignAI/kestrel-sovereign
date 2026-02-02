"""
PostgreSQL A2A Stores for Multi-Tenant Deployment.

DEPRECATION NOTICE (Issue #2):
    PostgreSQL stores are deprecated in favor of SQLite-first architecture
    with optional cloud sync. This module remains available for existing
    multi-tenant deployments but will be removed in a future release.

    Migration path:
    - Use SQLite stores directly: from kestrel_sovereign.a2a.stores import TaskStore
    - Use SyncService for cloud backup: from kestrel_sovereign.storage.sync import SyncService
    - For multi-tenant aggregation, use PostgreSQL as a sync target only

    See: kestrel_sovereign/storage/sync/
    See: feedback/2026-01-12_postgres_migration_plan.md

This module provides PostgreSQL-backed A2A stores using the unified
backend-agnostic store implementations with PostgresBackend.

For new code, use SQLite stores with sync:
    from kestrel_sovereign.storage.db import SQLiteBackend
    from kestrel_sovereign.a2a.stores.unified import TaskStore
    from kestrel_sovereign.storage.sync import SyncService, S3Target

    backend = SQLiteBackend("/path/to/agent.db")
    await backend.connect()
    task_store = TaskStore(backend)

    # Optional: sync to cloud
    sync = SyncService("/path/to/agent.db")
    sync.add_target(S3Target(bucket="my-bucket"))
    await sync.start()

The Postgres*Store classes here are thin wrappers for backward compatibility
with existing code that uses asyncpg.Pool directly.
"""

import logging
import warnings
from typing import Any, Optional

import asyncpg

# Import the PostgresBackend class directly (not the lazy loader function)
from kestrel_sovereign.storage.db.postgres import PostgresBackend

# Import unified stores - these work with any DatabaseBackend
from kestrel_sovereign.a2a.stores.unified import (
    TaskStore as UnifiedTaskStore,
    SessionService as UnifiedSessionService,
    MemoryService as UnifiedMemoryService,
    ObservabilityStore as UnifiedObservabilityStore,
    OrchestrationStore as UnifiedOrchestrationStore,
    FeedbackStore as UnifiedFeedbackStore,
)

# Re-export data models for backward compatibility
from kestrel_sovereign.a2a.stores.unified.session_service import SessionState
from kestrel_sovereign.a2a.stores.unified.memory_service import MemoryEntry
from kestrel_sovereign.a2a.stores.unified.observability_store import ObservabilityEvent, LLMCallEvent
from kestrel_sovereign.a2a.stores.unified.orchestration_store import OrchestrationTask, OrchestrationStatus
from kestrel_sovereign.a2a.stores.unified.feedback_store import (
    FeedbackEntry,
    FeedbackCategory,
    FeedbackSeverity,
    FeedbackStatus,
    FeedbackSource,
)

logger = logging.getLogger(__name__)


class PoolBackendAdapter(PostgresBackend):
    """
    Adapter that wraps an existing asyncpg.Pool as a PostgresBackend.

    This allows using the unified stores with an existing pool from
    external app_state.pg_pool without creating a new connection pool.
    """

    def __init__(self, pool: asyncpg.Pool):
        """
        Initialize with an existing asyncpg connection pool.

        Args:
            pool: asyncpg connection pool from external pg_pool
        """
        # Don't call super().__init__() - we're wrapping an existing pool
        # Set the pool directly (PostgresBackend checks self._pool)
        self._pool = pool
        self._transaction_conn = None  # Required by PostgresBackend

    async def connect(self) -> None:
        """No-op since pool is already connected."""
        pass

    async def close(self) -> None:
        """No-op - don't close the shared pool."""
        pass


# =============================================================================
# Backward-Compatible Wrapper Classes
# =============================================================================
# These classes wrap the unified stores with the pool adapter, providing
# the same interface as the original Postgres*Store classes.

class PostgresTaskStore(UnifiedTaskStore):
    """PostgreSQL-backed task store for multi-tenant deployment.

    DEPRECATED: Use SQLite stores with SyncService instead.
    """

    def __init__(self, pool: asyncpg.Pool):
        """
        Initialize with asyncpg connection pool.

        Args:
            pool: asyncpg connection pool from external pg_pool
        """
        warnings.warn(
            "PostgresTaskStore is deprecated. Use TaskStore with SQLiteBackend and SyncService. "
            "See kestrel_sovereign/storage/sync/ and Issue #2.",
            DeprecationWarning,
            stacklevel=2,
        )
        backend = PoolBackendAdapter(pool)
        super().__init__(backend)


class PostgresSessionService(UnifiedSessionService):
    """PostgreSQL-backed session service for multi-tenant deployment.

    DEPRECATED: Use SQLite stores with SyncService instead.
    """

    def __init__(self, pool: asyncpg.Pool):
        """
        Initialize with asyncpg connection pool.

        Args:
            pool: asyncpg connection pool from external pg_pool
        """
        warnings.warn(
            "PostgresSessionService is deprecated. Use SessionService with SQLiteBackend. "
            "See kestrel_sovereign/storage/sync/ and Issue #2.",
            DeprecationWarning,
            stacklevel=2,
        )
        backend = PoolBackendAdapter(pool)
        super().__init__(backend)


class PostgresMemoryService(UnifiedMemoryService):
    """PostgreSQL-backed memory with full-text search via tsvector/GIN.

    DEPRECATED: Use SQLite stores with SyncService instead.
    """

    def __init__(self, pool: asyncpg.Pool):
        """
        Initialize with asyncpg connection pool.

        Args:
            pool: asyncpg connection pool from external pg_pool
        """
        warnings.warn(
            "PostgresMemoryService is deprecated. Use MemoryService with SQLiteBackend. "
            "See kestrel_sovereign/storage/sync/ and Issue #2.",
            DeprecationWarning,
            stacklevel=2,
        )
        backend = PoolBackendAdapter(pool)
        super().__init__(backend)


class PostgresObservabilityStore(UnifiedObservabilityStore):
    """PostgreSQL-backed observability store.

    DEPRECATED: Use SQLite stores with SyncService instead.
    """

    def __init__(self, pool: asyncpg.Pool):
        """
        Initialize with asyncpg connection pool.

        Args:
            pool: asyncpg connection pool from external pg_pool
        """
        warnings.warn(
            "PostgresObservabilityStore is deprecated. Use ObservabilityStore with SQLiteBackend. "
            "See kestrel_sovereign/storage/sync/ and Issue #2.",
            DeprecationWarning,
            stacklevel=2,
        )
        backend = PoolBackendAdapter(pool)
        super().__init__(backend)


class PostgresOrchestrationStore(UnifiedOrchestrationStore):
    """PostgreSQL-backed orchestration store for multi-agent workflows.

    DEPRECATED: Use SQLite stores with SyncService instead.
    """

    def __init__(self, pool: asyncpg.Pool):
        """
        Initialize with asyncpg connection pool.

        Args:
            pool: asyncpg connection pool from external pg_pool
        """
        warnings.warn(
            "PostgresOrchestrationStore is deprecated. Use OrchestrationStore with SQLiteBackend. "
            "See kestrel_sovereign/storage/sync/ and Issue #2.",
            DeprecationWarning,
            stacklevel=2,
        )
        backend = PoolBackendAdapter(pool)
        super().__init__(backend)


class PostgresFeedbackStore(UnifiedFeedbackStore):
    """PostgreSQL-backed feedback store for agent self-diagnosis.

    DEPRECATED: Use SQLite stores with SyncService instead.
    """

    def __init__(self, pool: asyncpg.Pool):
        """
        Initialize with asyncpg connection pool.

        Args:
            pool: asyncpg connection pool from external pg_pool
        """
        warnings.warn(
            "PostgresFeedbackStore is deprecated. Use FeedbackStore with SQLiteBackend. "
            "See kestrel_sovereign/storage/sync/ and Issue #2.",
            DeprecationWarning,
            stacklevel=2,
        )
        backend = PoolBackendAdapter(pool)
        super().__init__(backend)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Wrapper classes for backward compatibility
    "PostgresTaskStore",
    "PostgresSessionService",
    "PostgresMemoryService",
    "PostgresObservabilityStore",
    "PostgresOrchestrationStore",
    "PostgresFeedbackStore",
    # Data models
    "SessionState",
    "MemoryEntry",
    "ObservabilityEvent",
    "LLMCallEvent",
    "OrchestrationTask",
    "OrchestrationStatus",
    "FeedbackEntry",
    "FeedbackCategory",
    "FeedbackSeverity",
    "FeedbackStatus",
    "FeedbackSource",
    # Adapter for direct pool usage
    "PoolBackendAdapter",
]
