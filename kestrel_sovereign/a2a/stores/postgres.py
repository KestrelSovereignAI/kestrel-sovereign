"""
PostgreSQL A2A Stores.

These wrap the unified, backend-agnostic A2A stores with a PostgresBackend
so they run against an existing asyncpg.Pool from app_state.pg_pool without
opening a second pool.

    from kestrel_sovereign.a2a.stores.postgres import PostgresTaskStore
    task_store = PostgresTaskStore(app_state.pg_pool)

Backend choice is independent of deployment tier. SQLite is the zero-config
default (a portable single-file sovereign agent); PostgreSQL is always
available as an option and can back a single-user agent just as well as a
multi-tenant deployment (one shared database, per-agent isolation, server-
grade concurrency). Multi-tenant SaaS like Frinz uses PostgreSQL because it
needs a shared pool — not because PostgreSQL is multi-tenant-only. Both
backends are first-class; pick by operational preference, and use the sync
layer (kestrel_sovereign.storage.sync) for cloud replication of either.
"""

import logging
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
# PostgreSQL Store Wrappers
# =============================================================================
# These classes wrap the unified stores with the pool adapter, exposing the
# same interface against an existing asyncpg.Pool from app_state.pg_pool.

class PostgresTaskStore(UnifiedTaskStore):
    """PostgreSQL-backed task store for multi-tenant deployment."""

    def __init__(self, pool: asyncpg.Pool):
        """
        Initialize with asyncpg connection pool.

        Args:
            pool: asyncpg connection pool from app_state.pg_pool
        """
        backend = PoolBackendAdapter(pool)
        super().__init__(backend)


class PostgresSessionService(UnifiedSessionService):
    """PostgreSQL-backed session service for multi-tenant deployment."""

    def __init__(self, pool: asyncpg.Pool):
        """
        Initialize with asyncpg connection pool.

        Args:
            pool: asyncpg connection pool from app_state.pg_pool
        """
        backend = PoolBackendAdapter(pool)
        super().__init__(backend)


class PostgresMemoryService(UnifiedMemoryService):
    """PostgreSQL-backed memory with full-text search via tsvector/GIN."""

    def __init__(self, pool: asyncpg.Pool):
        """
        Initialize with asyncpg connection pool.

        Args:
            pool: asyncpg connection pool from app_state.pg_pool
        """
        backend = PoolBackendAdapter(pool)
        super().__init__(backend)


class PostgresObservabilityStore(UnifiedObservabilityStore):
    """PostgreSQL-backed observability store."""

    def __init__(self, pool: asyncpg.Pool):
        """
        Initialize with asyncpg connection pool.

        Args:
            pool: asyncpg connection pool from app_state.pg_pool
        """
        backend = PoolBackendAdapter(pool)
        super().__init__(backend)


class PostgresOrchestrationStore(UnifiedOrchestrationStore):
    """PostgreSQL-backed orchestration store for multi-agent workflows."""

    def __init__(self, pool: asyncpg.Pool):
        """
        Initialize with asyncpg connection pool.

        Args:
            pool: asyncpg connection pool from app_state.pg_pool
        """
        backend = PoolBackendAdapter(pool)
        super().__init__(backend)


class PostgresFeedbackStore(UnifiedFeedbackStore):
    """PostgreSQL-backed feedback store for agent self-diagnosis."""

    def __init__(self, pool: asyncpg.Pool):
        """
        Initialize with asyncpg connection pool.

        Args:
            pool: asyncpg connection pool from app_state.pg_pool
        """
        backend = PoolBackendAdapter(pool)
        super().__init__(backend)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # PostgreSQL store wrappers
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
