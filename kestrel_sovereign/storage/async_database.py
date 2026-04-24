"""
Async Database for Kestrel Storage.

Unified async database interface supporting SQLite and PostgreSQL.
All queries use SQLite-style ? placeholders - automatically converted for PostgreSQL.
"""
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple

from .db import DatabaseBackend, SQLiteBackend, get_backend, normalize_schema

logger = logging.getLogger(__name__)


# Core schema - written in SQLite style, converted for PostgreSQL
CORE_SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    content_hash TEXT PRIMARY KEY,
    original_name TEXT NOT NULL,
    content BLOB,
    metadata TEXT
);

CREATE TABLE IF NOT EXISTS graph_nodes (
    node_id TEXT PRIMARY KEY,
    node_type TEXT NOT NULL,
    label TEXT NOT NULL,
    properties TEXT
);

CREATE INDEX IF NOT EXISTS idx_graph_nodes_type ON graph_nodes(node_type);
CREATE INDEX IF NOT EXISTS idx_graph_nodes_type_label ON graph_nodes(node_type, label);

CREATE TABLE IF NOT EXISTS graph_edges (
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    label TEXT NOT NULL,
    properties TEXT,
    PRIMARY KEY (source_id, target_id, label)
);

CREATE INDEX IF NOT EXISTS idx_graph_edges_target ON graph_edges(target_id, label);
CREATE INDEX IF NOT EXISTS idx_graph_edges_label ON graph_edges(label);

CREATE TABLE IF NOT EXISTS document_chunks (
    chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_hash TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding BLOB
);

CREATE TABLE IF NOT EXISTS conversation_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT NOT NULL DEFAULT '',
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_conversation_agent_id ON conversation_history(agent_id);

-- User-assigned conversation titles (issue #716).  Decoupled from
-- conversation_history so renames are a single-row upsert instead of a
-- metadata-JSON edit on an encrypted message.  Nullable name means
-- "cleared — use the computed display title instead."
CREATE TABLE IF NOT EXISTS conversation_titles (
    agent_id   TEXT NOT NULL,
    session_id TEXT NOT NULL,
    name       TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (agent_id, session_id)
);

CREATE INDEX IF NOT EXISTS idx_conversation_titles_agent
    ON conversation_titles(agent_id);

CREATE TABLE IF NOT EXISTS model_usage (
    model_id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    last_used TIMESTAMP NOT NULL,
    use_count INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS wallet_state (
    agent_id TEXT PRIMARY KEY,
    main_balance TEXT NOT NULL,
    audit_balance TEXT NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS wallet_transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT NOT NULL,
    transaction_type TEXT NOT NULL,
    currency TEXT NOT NULL DEFAULT 'FIL',
    amount TEXT NOT NULL,
    memo TEXT,
    new_balance TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS temporal_patterns (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    pattern_type TEXT NOT NULL,
    description TEXT NOT NULL,
    trigger_conditions TEXT,
    confidence REAL DEFAULT 0.0,
    observations INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_temporal_patterns_agent ON temporal_patterns(agent_id);

CREATE TABLE IF NOT EXISTS memory_episodes (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    title TEXT NOT NULL,
    summary TEXT,
    timespan_start TIMESTAMP,
    timespan_end TIMESTAMP,
    key_message_ids TEXT,
    emotional_arc TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_memory_episodes_agent ON memory_episodes(agent_id);

CREATE TABLE IF NOT EXISTS reflection_insights (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    session_id TEXT,
    type TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    evidence TEXT,
    confidence REAL DEFAULT 0.5,
    actionable INTEGER DEFAULT 0,
    suggested_action TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_reflection_insights_agent ON reflection_insights(agent_id);
CREATE INDEX IF NOT EXISTS idx_reflection_insights_session ON reflection_insights(session_id);

CREATE TABLE IF NOT EXISTS reflection_sessions (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    trigger TEXT NOT NULL,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    interactions_analyzed INTEGER DEFAULT 0,
    episodes_analyzed INTEGER DEFAULT 0,
    insights_generated INTEGER DEFAULT 0,
    improvements_proposed INTEGER DEFAULT 0,
    improvements_approved INTEGER DEFAULT 0,
    error TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_reflection_sessions_agent ON reflection_sessions(agent_id);

CREATE TABLE IF NOT EXISTS improvement_proposals (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    insight_id TEXT,
    title TEXT NOT NULL,
    description TEXT,
    change_type TEXT NOT NULL,
    proposed_change TEXT NOT NULL,
    requires_approval INTEGER DEFAULT 1,
    approved INTEGER DEFAULT 0,
    rejection_reason TEXT,
    approved_at TIMESTAMP,
    approved_by TEXT,
    applied_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_improvement_proposals_agent ON improvement_proposals(agent_id);

CREATE TABLE IF NOT EXISTS behavior_rules (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    proposal_id TEXT NOT NULL,
    trigger_condition TEXT NOT NULL,
    action TEXT NOT NULL,
    change_type TEXT NOT NULL,
    active INTEGER DEFAULT 1,
    priority INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    deactivated_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_behavior_rules_agent ON behavior_rules(agent_id);
CREATE INDEX IF NOT EXISTS idx_behavior_rules_active ON behavior_rules(agent_id, active);

-- Service API Key Management (BYOK + Managed)
CREATE TABLE IF NOT EXISTS service_providers (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    supports_sub_accounts INTEGER DEFAULT 0,
    referral_program_url TEXT,
    api_docs_url TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS user_service_keys (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    provider_id TEXT NOT NULL,
    key_mode TEXT NOT NULL CHECK (key_mode IN ('byok', 'managed')),
    encrypted_key BLOB NOT NULL,
    key_hash TEXT NOT NULL,
    quota_limit INTEGER,
    quota_used INTEGER DEFAULT 0,
    referral_code TEXT,
    is_active INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    UNIQUE(user_id, provider_id)
);

CREATE INDEX IF NOT EXISTS idx_user_service_keys_user ON user_service_keys(user_id);
CREATE INDEX IF NOT EXISTS idx_user_service_keys_provider ON user_service_keys(provider_id);

CREATE TABLE IF NOT EXISTS agent_service_keys (
    id TEXT PRIMARY KEY,
    agent_did TEXT NOT NULL,
    provider_id TEXT NOT NULL,
    encrypted_key TEXT NOT NULL,
    key_hash TEXT NOT NULL,
    quota_limit INTEGER,
    quota_used INTEGER DEFAULT 0,
    is_active INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(agent_did, provider_id)
);

CREATE INDEX IF NOT EXISTS idx_agent_service_keys_agent ON agent_service_keys(agent_did);
CREATE INDEX IF NOT EXISTS idx_agent_service_keys_provider ON agent_service_keys(provider_id);

CREATE TABLE IF NOT EXISTS service_key_usage (
    id TEXT PRIMARY KEY,
    key_id TEXT NOT NULL,
    key_type TEXT NOT NULL CHECK (key_type IN ('user', 'agent', 'companion')),
    provider_id TEXT NOT NULL,
    operation TEXT NOT NULL,
    units_consumed INTEGER DEFAULT 1,
    cost_estimate_usd REAL,
    companion_id TEXT,
    request_metadata TEXT,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_service_key_usage_key ON service_key_usage(key_id);
CREATE INDEX IF NOT EXISTS idx_service_key_usage_recorded ON service_key_usage(recorded_at);

CREATE TABLE IF NOT EXISTS agent_metadata (
    agent_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (agent_id, key)
);

CREATE INDEX IF NOT EXISTS idx_agent_metadata_agent ON agent_metadata(agent_id);

-- Bootstrap Config: Per-agent configuration for bootstrap file loading convention
CREATE TABLE IF NOT EXISTS bootstrap_config (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    file_name TEXT NOT NULL,
    file_path TEXT DEFAULT '',
    enabled INTEGER DEFAULT 1,
    priority INTEGER DEFAULT 100,
    max_size_bytes INTEGER DEFAULT 10240,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_bootstrap_config_agent ON bootstrap_config(agent_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_bootstrap_config_agent_file ON bootstrap_config(agent_id, file_name);

-- Saved Items: Unified storage for stashes, files, excerpts, and structured items
CREATE TABLE IF NOT EXISTS saved_items (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    item_type TEXT NOT NULL,
    name TEXT NOT NULL,
    summary TEXT,
    content TEXT NOT NULL,
    content_hash TEXT,
    ipfs_cid TEXT,
    embedding BLOB,
    source_type TEXT,
    source_ref TEXT,
    schema_id TEXT,
    tags TEXT,
    metadata TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_saved_items_agent ON saved_items(agent_id);
CREATE INDEX IF NOT EXISTS idx_saved_items_type ON saved_items(agent_id, item_type);
CREATE INDEX IF NOT EXISTS idx_saved_items_hash ON saved_items(content_hash);
"""

# Backend-specific JSON-path indexes on graph_nodes properties.
# These cannot go through normalize_schema because the JSON extraction
# syntax differs fundamentally between SQLite and PostgreSQL.
_SQLITE_JSON_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_graph_nodes_agent
  ON graph_nodes(node_type, json_extract(properties, '$.agent_id'));
CREATE INDEX IF NOT EXISTS idx_graph_nodes_action_status
  ON graph_nodes(json_extract(properties, '$.status'))
  WHERE node_type = 'action_item';
CREATE INDEX IF NOT EXISTS idx_graph_nodes_action_created
  ON graph_nodes(json_extract(properties, '$.created_at'))
  WHERE node_type = 'action_item';
"""

_POSTGRES_JSON_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_graph_nodes_agent
  ON graph_nodes(node_type, (properties::jsonb->>'agent_id'));
CREATE INDEX IF NOT EXISTS idx_graph_nodes_action_status
  ON graph_nodes((properties::jsonb->>'status'))
  WHERE node_type = 'action_item';
CREATE INDEX IF NOT EXISTS idx_graph_nodes_action_created
  ON graph_nodes((properties::jsonb->>'created_at'))
  WHERE node_type = 'action_item';
CREATE INDEX IF NOT EXISTS idx_graph_nodes_properties_gin
  ON graph_nodes USING GIN ((properties::jsonb));
"""


class AsyncDatabase:
    """
    Async database manager supporting SQLite and PostgreSQL.
    
    Initialize with either:
    - A DatabaseBackend instance
    - A config dict (creates appropriate backend)
    """
    
    def __init__(self, backend: DatabaseBackend):
        """
        Initialize with a database backend.
        
        Args:
            backend: DatabaseBackend instance (SQLiteBackend or PostgresBackend)
        """
        self._backend = backend
        self._initialized = False
    
    @classmethod
    async def create(cls, config: Optional[Dict[str, Any]] = None) -> "AsyncDatabase":
        """
        Factory method to create and connect a database.
        
        Args:
            config: Configuration dict with 'backend' key ('sqlite' or 'postgres')
                   and backend-specific options
        
        Returns:
            Connected AsyncDatabase instance
        """
        backend = await get_backend(config)
        db = cls(backend)
        await db._init_schema()
        db._initialized = True
        return db
    
    @classmethod
    async def sqlite(cls, db_path: str) -> "AsyncDatabase":
        """Create SQLite database at given path."""
        backend = SQLiteBackend(db_path)
        await backend.connect()
        db = cls(backend)
        await db._init_schema()
        db._initialized = True
        logger.info(f"SQLite database connected: {db_path}")
        return db
    
    @classmethod
    async def postgres(cls, dsn: str) -> "AsyncDatabase":
        """Create PostgreSQL database with given DSN."""
        from .db.postgres import PostgresBackend
        backend = PostgresBackend(dsn=dsn)
        await backend.connect()
        db = cls(backend)
        await db._init_schema()
        db._initialized = True
        logger.info("PostgreSQL database connected")
        return db

    @classmethod
    async def from_pool(cls, pool) -> "AsyncDatabase":
        """
        Create AsyncDatabase from an existing asyncpg connection pool.

        This is useful for multi-tenant deployments where
        a shared pool is managed by the application server.

        Note: close() will NOT close the underlying pool.

        Args:
            pool: asyncpg.Pool instance

        Returns:
            AsyncDatabase wrapping the pool
        """
        from .db.postgres import PostgresBackend
        backend = PostgresBackend.from_pool(pool)
        db = cls(backend)
        await db._init_schema()
        db._initialized = True
        logger.info("PostgreSQL database created from existing pool")
        return db
    
    @property
    def backend(self) -> DatabaseBackend:
        """Get the underlying database backend."""
        return self._backend
    
    @property
    def backend_type(self) -> str:
        """Get backend type: 'sqlite' or 'postgres'."""
        return self._backend.backend_type
    
    async def _init_schema(self) -> None:
        """Create database tables if they don't exist."""
        schema = normalize_schema(CORE_SCHEMA, self.backend_type)

        # Execute each statement separately for PostgreSQL compatibility
        for statement in schema.split(';'):
            statement = statement.strip()
            if statement:
                await self._backend.execute(statement)

        # JSON-path indexes use backend-specific syntax that cannot be
        # normalised by simple regex, so we pick the right DDL block here.
        json_indexes = (
            _POSTGRES_JSON_INDEXES if self.backend_type == "postgres"
            else _SQLITE_JSON_INDEXES
        )
        for statement in json_indexes.split(';'):
            statement = statement.strip()
            if statement:
                await self._backend.execute(statement)

        logger.debug(f"Database schema initialized ({self.backend_type})")
    
    # ─────────────────────────────────────────────────────────────────
    # Query methods - delegate to backend
    # ─────────────────────────────────────────────────────────────────
    
    async def execute(self, sql: str, params: tuple = ()) -> int:
        """Execute a write query. Returns rows affected."""
        return await self._backend.execute(sql, params)
    
    async def execute_commit(self, sql: str, params: tuple = ()) -> int:
        """Execute a write query (commit is automatic). Returns rows affected."""
        # Backend handles commits automatically, this is for API compatibility
        return await self._backend.execute(sql, params)
    
    async def execute_many(self, sql: str, params_list: List[tuple]) -> int:
        """Execute query with multiple parameter sets."""
        return await self._backend.execute_many(sql, params_list)
    
    async def fetchone(self, sql: str, params: tuple = ()) -> Optional[Tuple[Any, ...]]:
        """Fetch a single row."""
        return await self._backend.fetch_one(sql, params)
    
    async def fetchall(self, sql: str, params: tuple = ()) -> List[Tuple[Any, ...]]:
        """Fetch all rows."""
        return await self._backend.fetch_all(sql, params)
    
    async def fetchval(self, sql: str, params: tuple = ()) -> Optional[Any]:
        """Fetch a single value."""
        return await self._backend.fetch_val(sql, params)
    
    @asynccontextmanager
    async def transaction(self):
        """Transaction context manager with automatic rollback on error."""
        async with self._backend.transaction():
            yield
    
    async def table_exists(self, table_name: str) -> bool:
        """Check if a table exists."""
        return await self._backend.table_exists(table_name)
    
    async def commit(self) -> None:
        """Commit transaction (no-op, commits are automatic in new backend)."""
        # Backend handles commits automatically per-query outside transactions
        pass
    
    async def close(self) -> None:
        """Close the database connection."""
        await self._backend.close()
        self._initialized = False
        logger.debug("Database connection closed")
    
    async def __aenter__(self):
        if not self._initialized:
            await self._init_schema()
            self._initialized = True
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
