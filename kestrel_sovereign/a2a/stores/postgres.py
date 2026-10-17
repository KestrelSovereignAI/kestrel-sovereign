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

import asyncpg
from kestrel_sovereign.execution_custody import ExecutionCustody

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

    def __init__(self, pool: asyncpg.Pool, *, execution_custody: ExecutionCustody | None = None):
        """
        Initialize with an existing asyncpg connection pool.

        Args:
            pool: asyncpg connection pool from external pg_pool
        """
        # Canonical native initialization; no new pool or authority is minted.
        self.__dict__.update(PostgresBackend.from_pool(
            pool, execution_custody=execution_custody,
        ).__dict__)

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

class _SharedPostgresStore:
    def __init__(
        self, pool: asyncpg.Pool | None = None, *,
        backend: PostgresBackend | None = None,
        execution_custody: ExecutionCustody | None = None,
    ):
        if backend is not None:
            if pool is not None or execution_custody is not None:
                raise ValueError("provide either a native backend or a pool/custody")
            if not isinstance(backend, PostgresBackend):
                raise TypeError("A2A PostgreSQL stores require a native PostgresBackend")
        elif pool is not None:
            backend = PoolBackendAdapter(pool, execution_custody=execution_custody)
        else:
            raise ValueError("A2A PostgreSQL stores require a backend or pool")
        super().__init__(backend)

    async def close(self) -> None:
        # Storage/host owns the backend; primary storage closes it last.
        pass


class PostgresTaskStore(_SharedPostgresStore, UnifiedTaskStore):
    """PostgreSQL-backed task store for multi-tenant deployment."""


class PostgresSessionService(_SharedPostgresStore, UnifiedSessionService):
    """PostgreSQL-backed session service for multi-tenant deployment."""


class PostgresMemoryService(_SharedPostgresStore, UnifiedMemoryService):
    """PostgreSQL-backed memory with full-text search via tsvector/GIN."""


class PostgresObservabilityStore(_SharedPostgresStore, UnifiedObservabilityStore):
    """PostgreSQL-backed observability store."""


class PostgresOrchestrationStore(_SharedPostgresStore, UnifiedOrchestrationStore):
    """PostgreSQL-backed orchestration store for multi-agent workflows."""


class PostgresFeedbackStore(_SharedPostgresStore, UnifiedFeedbackStore):
    """PostgreSQL-backed feedback store for agent self-diagnosis."""


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
