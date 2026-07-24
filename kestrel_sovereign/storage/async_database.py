"""
Async Database for Kestrel Storage.

Unified async database interface supporting SQLite and PostgreSQL.
All queries use SQLite-style ? placeholders - automatically converted for PostgreSQL.
"""
import asyncio
import hashlib
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple

from .db import DatabaseBackend, SQLiteBackend, get_backend, normalize_schema

logger = logging.getLogger(__name__)


_BACKFILL_LOCK_DOMAIN = b"kestrel:schema-backfill-lock:v1\0"


def _backfill_lock_id(name: str) -> int:
    """Stable signed-64-bit PostgreSQL advisory-lock key for a one-time backfill.

    Used to serialize the first post-upgrade run across concurrent initializers
    so a request burst doesn't stampede the heavy migration. Hash collisions
    only serialize unrelated backfills; they cannot affect correctness.
    """
    payload = _BACKFILL_LOCK_DOMAIN + name.encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=True)


# Core schema - written in SQLite style, converted for PostgreSQL
CORE_SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    content_hash TEXT PRIMARY KEY,
    original_name TEXT NOT NULL,
    content BLOB,
    metadata TEXT
);

-- A content-addressed blob can be referenced by more than one agent, but the
-- physical ``files`` row is not itself a tenant boundary.  Keep the reference
-- metadata beside the authoritative owner witness so a tenant-bound reader
-- never learns another tenant's filename or metadata through deduplication.
CREATE TABLE IF NOT EXISTS file_owners (
    content_hash TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    original_name TEXT NOT NULL,
    metadata TEXT,
    PRIMARY KEY (content_hash, agent_id)
);

CREATE INDEX IF NOT EXISTS idx_file_owners_agent
    ON file_owners(agent_id, content_hash);

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

-- Authoritative tenant ownership for graph rows.  Ownership is kept outside
-- properties JSON because graph nodes such as content-addressed constitutions
-- can legitimately be shared by multiple agents, while an edge has its own
-- disclosure boundary independent of either endpoint (#2649).
CREATE TABLE IF NOT EXISTS graph_node_owners (
    node_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    PRIMARY KEY (node_id, agent_id)
);

CREATE INDEX IF NOT EXISTS idx_graph_node_owners_agent
    ON graph_node_owners(agent_id, node_id);

CREATE TABLE IF NOT EXISTS graph_edge_owners (
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    label TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    PRIMARY KEY (source_id, target_id, label, agent_id)
);

CREATE INDEX IF NOT EXISTS idx_graph_edge_owners_agent
    ON graph_edge_owners(agent_id, source_id, target_id, label);

CREATE TABLE IF NOT EXISTS document_chunks (
    chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_hash TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding BLOB
);

-- Chunk text is caller-supplied and is therefore a separate tenant
-- capability from the content-addressed file bytes. Two agents may own the
-- same file without sharing annotations or derived chunk content.
CREATE TABLE IF NOT EXISTS document_chunk_owners (
    chunk_id INTEGER NOT NULL,
    agent_id TEXT NOT NULL,
    PRIMARY KEY (chunk_id, agent_id)
);

CREATE INDEX IF NOT EXISTS idx_document_chunk_owners_agent
    ON document_chunk_owners(agent_id, chunk_id);

-- One-time data-migration markers. Expensive backfills (e.g. the #2649
-- ownership ledgers) record a row here once they succeed so they are not
-- re-run on every _init_schema()/from_pool() — repeated multi-join scans on a
-- populated database caused lock contention and statement timeouts.
CREATE TABLE IF NOT EXISTS schema_backfills (
    name TEXT PRIMARY KEY,
    completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS conversation_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT NOT NULL DEFAULT '',
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    rendered_content TEXT DEFAULT NULL,
    model TEXT DEFAULT NULL,
    provider TEXT DEFAULT NULL,
    metadata TEXT,
    lexical_index_id TEXT DEFAULT NULL,
    lexical_index_version TEXT DEFAULT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP DEFAULT NULL,
    archived_at TIMESTAMP DEFAULT NULL
);

CREATE TABLE IF NOT EXISTS conversation_lexical_tokens (
    agent_id TEXT NOT NULL,
    lexical_index_id TEXT NOT NULL,
    token_hash TEXT NOT NULL,
    PRIMARY KEY (agent_id, lexical_index_id, token_hash)
);

CREATE INDEX IF NOT EXISTS idx_conversation_lexical_token_lookup
    ON conversation_lexical_tokens(agent_id, token_hash);

CREATE INDEX IF NOT EXISTS idx_conversation_agent_id ON conversation_history(agent_id);
CREATE INDEX IF NOT EXISTS idx_conversation_agent_created_at
    ON conversation_history(agent_id, created_at DESC);
-- idx_conversation_deleted_at lives in _init_schema (after the
-- soft-delete migration runs). Legacy DBs predate the column and
-- would crash here. See #795.

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
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    importance REAL DEFAULT 0.5,
    access_count INTEGER DEFAULT 0,
    embedding_vec BLOB,
    embedding_profile_id TEXT
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

-- Zero-knowledge user BYOK credentials. The platform stores only ciphertext
-- plus per-row KDF salt and AEAD nonce. Decryption requires a per-request user
-- passphrase and never falls back to KESTREL_DATA_KEY or any host/user master.
CREATE TABLE IF NOT EXISTS user_byok_service_keys (
    id TEXT PRIMARY KEY,
    agent_did TEXT NOT NULL,
    provider_id TEXT NOT NULL,
    encrypted_key TEXT NOT NULL,
    key_salt TEXT NOT NULL,
    key_nonce TEXT NOT NULL,
    key_hash TEXT NOT NULL,
    kdf TEXT NOT NULL DEFAULT 'PBKDF2-SHA256',
    kdf_iterations INTEGER NOT NULL DEFAULT 600000,
    is_active INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(agent_did, provider_id)
);

CREATE INDEX IF NOT EXISTS idx_user_byok_keys_agent ON user_byok_service_keys(agent_did);
CREATE INDEX IF NOT EXISTS idx_user_byok_keys_provider ON user_byok_service_keys(provider_id);

-- Host (operator) master credentials for the HOST_MASTER_PROVISIONED
-- payer-policy path. Single host per deployment. Sponsor and
-- user-master variants are modeled separately if/when needed. See
-- kestrel_sovereign.security.host_key_storage.HostKeyStorage.
CREATE TABLE IF NOT EXISTS host_service_keys (
    id TEXT PRIMARY KEY,
    provider_id TEXT NOT NULL UNIQUE,
    encrypted_key TEXT NOT NULL,
    key_hash TEXT NOT NULL,
    is_active INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_host_service_keys_provider ON host_service_keys(provider_id);

-- Per-user master credentials for the USER_MASTER_PROVISIONED payer-policy
-- path: a named user's master account funds an agent, and the resolver mints
-- a per-agent child against it (same shape as host_service_keys, but scoped
-- per master_did = the funding user's DID). See
-- kestrel_sovereign.security.user_master_key_storage.UserMasterKeyStorage.
CREATE TABLE IF NOT EXISTS user_master_service_keys (
    id TEXT PRIMARY KEY,
    master_did TEXT NOT NULL,
    provider_id TEXT NOT NULL,
    encrypted_key TEXT NOT NULL,
    key_hash TEXT NOT NULL,
    is_active INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(master_did, provider_id)
);

CREATE INDEX IF NOT EXISTS idx_user_master_keys_did ON user_master_service_keys(master_did);
CREATE INDEX IF NOT EXISTS idx_user_master_keys_provider ON user_master_service_keys(provider_id);

-- Per-sponsor master credentials for the SPONSOR payer-policy path: a named
-- third party (e.g. an org) funds agents, and the resolver mints a per-agent
-- child against the sponsor's master (same shape as user_master_service_keys,
-- scoped per master_did = the sponsor's DID). See
-- kestrel_sovereign.security.sponsor_key_storage.SponsorKeyStorage.
CREATE TABLE IF NOT EXISTS sponsor_master_service_keys (
    id TEXT PRIMARY KEY,
    master_did TEXT NOT NULL,
    provider_id TEXT NOT NULL,
    encrypted_key TEXT NOT NULL,
    key_hash TEXT NOT NULL,
    is_active INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(master_did, provider_id)
);

CREATE INDEX IF NOT EXISTS idx_sponsor_master_keys_did ON sponsor_master_service_keys(master_did);
CREATE INDEX IF NOT EXISTS idx_sponsor_master_keys_provider ON sponsor_master_service_keys(provider_id);

-- Sponsor -> beneficiary (agent) roster: which sponsor funds which agent. A
-- policy builder consults this to set PayerSpec(kind=SPONSOR, master_did=...)
-- for an enrolled agent. One funding sponsor per agent (per-agent model).
-- Authority/consent for enrollment is a product concern, not enforced here.
-- (Schema comments must not contain a semicolon: _init_schema splits on it.)
CREATE TABLE IF NOT EXISTS sponsor_beneficiaries (
    sponsor_did TEXT NOT NULL,
    agent_did TEXT NOT NULL,
    is_active INTEGER DEFAULT 1,
    enrolled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (agent_did)
);

CREATE INDEX IF NOT EXISTS idx_sponsor_beneficiaries_sponsor ON sponsor_beneficiaries(sponsor_did);

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

-- Private resources attached to a local agent identity record. Resource
-- contents are encrypted before storage. Public metadata must only contain
-- safe pointers/hashes/provenance and never the private body.
CREATE TABLE IF NOT EXISTS agent_identity_resources (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    resource_id TEXT NOT NULL,
    resource_type TEXT NOT NULL,
    version INTEGER NOT NULL,
    is_current INTEGER DEFAULT 0,
    content_ciphertext BLOB NOT NULL,
    content_hash TEXT NOT NULL,
    content_bytes INTEGER NOT NULL,
    encryption TEXT NOT NULL,
    provenance TEXT,
    public_metadata TEXT,
    anchoring_metadata TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(agent_id, resource_type, version)
);

CREATE INDEX IF NOT EXISTS idx_agent_identity_resources_agent
    ON agent_identity_resources(agent_id, resource_type);
CREATE INDEX IF NOT EXISTS idx_agent_identity_resources_current
    ON agent_identity_resources(agent_id, resource_type, is_current);
CREATE INDEX IF NOT EXISTS idx_agent_identity_resources_resource
    ON agent_identity_resources(resource_id);

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

-- ============================================================================
-- A2A async-question correlation (#1444)
--
-- Sender-side record of in-flight ``send_a2a_question`` calls. When the
-- sender's ``PeersFeature._post_a2a_task`` POSTs a question to a peer, it
-- writes a row here AND spawns a tracked SSE subscription. When the
-- subscription handler sees the task reach a terminal state, it fires a
-- local ``a2a.question_answered`` signal that wakes the sender's cognition
-- loop with the reply inline.
--
-- This is NOT a "suspended turn" table — the asking turn ends cleanly the
-- moment the tool returns. The row exists only for:
--   1. Prompt assembly (the resumed turn's prompt cites the original
--      question text by task_id so the LLM has full context)
--   2. Restart replay (on boot, scan ``status='WAITING'`` rows and re-poll
--      the recipient — catches terminal events that fired while the sender
--      was down)
--   3. Two-questions-in-flight disambiguation (resumed signal carries the
--      task_id key)
--   4. Hourly expiry sweep (rows past ``deadline`` get a synthetic
--      ``state='expired'`` signal so the resumed prompt has a clean branch)
-- ============================================================================
-- ``agent_id`` scopes rows to the OWNING agent so a shared backend
-- (e.g. Postgres in a multi-agent deployment) does NOT let agent A's
-- startup-replay walk agent B's WAITING rows and mis-route the
-- ``a2a.question_answered`` signal to A's local dispatcher with B's
-- question content. The PK is composite (agent_id, task_id) since
-- ``task_id`` is only unique within an agent's own counter.
-- (Codex round 1 P1 on PR #1453.)
CREATE TABLE IF NOT EXISTS pending_a2a_questions (
    agent_id TEXT NOT NULL DEFAULT '',
    task_id TEXT NOT NULL,
    recipient TEXT NOT NULL,
    original_question TEXT NOT NULL,
    origin_turn_id TEXT,
    origin_session_id TEXT,
    deadline TIMESTAMP NOT NULL,
    status TEXT NOT NULL DEFAULT 'WAITING' CHECK (
        status IN ('WAITING', 'RESOLVED', 'EXPIRED')
    ),
    retry_state TEXT,
    retry_reply_text TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP,
    PRIMARY KEY (agent_id, task_id)
);

CREATE INDEX IF NOT EXISTS idx_pending_a2a_questions_sweep
    ON pending_a2a_questions(agent_id, status, deadline);

-- ============================================================================
-- wait_signal_state — durable dedup/delivery ledger for the generic wait
-- reconciler (Wave 2 of #1860). One row per (agent_id, kind, handle) the
-- reconciler has observed. Mirrors what talon_monitor used to stash inside
-- jobs.json (last_signaled_status plus the pending_signal_* fields), but
-- generically for EVERY MonitorableWaitable provider.
--
--   - last_signaled_outcome  application-level dedup: the terminal Outcome
--                            value we have already delivered a signal for, so
--                            the next tick does not re-fire the same transition
--   - last_delivery_*        diagnostics plus retry accounting (attempts caps
--                            the soft-fail retry loop, MAX_DELIVERY_ATTEMPTS)
--   - pending_signal_*       the two-phase harvest set: a signal we enqueued
--                            but have not yet confirmed delivered, cleared on
--                            harvest (record_delivery) so a restart that lost
--                            the in-memory task re-detects and retries
--   - watching               explicit watched-handle flag: the agent called
--                            wait(target, mode="signal") to register a watch on
--                            this (kind, handle). The reconciler polls watched
--                            handles via provider.poll() even when the provider
--                            is poll-only (not MonitorableWaitable), so EVERY
--                            async waitable is wakeable without auto-waking all
--                            tasks (which would self-wake on inbound work)
--
-- ``agent_id`` scopes rows to the OWNING agent for shared-backend isolation,
-- exactly like pending_a2a_questions above.
CREATE TABLE IF NOT EXISTS wait_signal_state (
    agent_id TEXT NOT NULL DEFAULT '',
    kind TEXT NOT NULL,
    handle TEXT NOT NULL,
    last_signaled_outcome TEXT,
    last_delivery_status TEXT,
    last_delivery_error TEXT,
    last_delivery_attempts INTEGER NOT NULL DEFAULT 0,
    last_delivery_attempt_at TIMESTAMP,
    pending_signal_id TEXT,
    pending_signaled_target TEXT,
    pending_signal_enqueued_at TIMESTAMP,
    watching INTEGER NOT NULL DEFAULT 0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (agent_id, kind, handle)
);
CREATE INDEX IF NOT EXISTS idx_wait_signal_state_pending
    ON wait_signal_state(agent_id, pending_signal_id);
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
CREATE INDEX IF NOT EXISTS idx_graph_nodes_todo_status
  ON graph_nodes(json_extract(properties, '$.status'))
  WHERE node_type = 'todo_item';
CREATE INDEX IF NOT EXISTS idx_graph_nodes_todo_scope
  ON graph_nodes(json_extract(properties, '$.scope'))
  WHERE node_type = 'todo_item';
CREATE INDEX IF NOT EXISTS idx_graph_nodes_todo_created
  ON graph_nodes(json_extract(properties, '$.created_at'))
  WHERE node_type = 'todo_item';
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
CREATE INDEX IF NOT EXISTS idx_graph_nodes_todo_status
  ON graph_nodes((properties::jsonb->>'status'))
  WHERE node_type = 'todo_item';
CREATE INDEX IF NOT EXISTS idx_graph_nodes_todo_scope
  ON graph_nodes((properties::jsonb->>'scope'))
  WHERE node_type = 'todo_item';
CREATE INDEX IF NOT EXISTS idx_graph_nodes_todo_created
  ON graph_nodes((properties::jsonb->>'created_at'))
  WHERE node_type = 'todo_item';
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

        # #2649: legacy graph/file/chunk rows predate the authoritative
        # ownership ledgers above. These backfills prove ownership only from
        # existing agent tags/roots/edges, leaving ambiguous rows unowned so
        # explorer reads fail closed. They are ONE-TIME migrations for legacy
        # data — new rows record ownership at write time (async_graph_store /
        # async_file_store / async_rag_store) — but each does heavy multi-join
        # INSERT...SELECT scans. _init_schema runs on every from_pool(), which
        # frinz calls per request, so running them unconditionally made
        # concurrent inits contend on the ownership tables and time out
        # (statement timeout / hung companion creation). Gate behind a
        # persistent marker so they run at most once; after that _init_schema
        # stays cheap. The steady-state fast path skips without taking any lock;
        # only the first post-upgrade run enters the serialized runner below.
        if not await self._backfill_completed("ownership_2649"):
            await self._run_ownership_backfills_once("ownership_2649")

        # Soft-delete migration (#763): add deleted_at to conversation_history
        # for databases created before soft-delete shipped. Idempotent — does
        # nothing on fresh databases (column already in CREATE TABLE). The
        # dependent index is created only after we verify the column is
        # present, otherwise legacy DBs blow up at boot (#795).
        await self._migrate_add_column(
            "conversation_history", "deleted_at", "TIMESTAMP DEFAULT NULL"
        )
        # Archive state (#2149): add archived_at to conversation_history,
        # mirroring the soft-delete deleted_at column. Idempotent — no-op on
        # fresh databases (column already in CREATE TABLE). The dependent
        # index is created below only after verifying the column is present.
        await self._migrate_add_column(
            "conversation_history", "archived_at", "TIMESTAMP DEFAULT NULL"
        )
        # Canonical/transport split (#1402): add rendered_content to hold the
        # byte-stable replay form (memories + RAG baked in) separately from
        # the canonical raw user turn in `content`. Legacy rows have a NULL
        # rendered_content; lazy migration in AsyncConversationStore splits
        # them on first read.
        await self._migrate_add_column(
            "conversation_history", "rendered_content", "TEXT DEFAULT NULL"
        )
        # #1370: assistant messages now stamp the exact resolved
        # route/model that produced the text. Nullable by design:
        # legacy rows predate the signal, restore/import rows may not
        # carry it, and UI/API consumers must tolerate missing values.
        await self._migrate_add_column(
            "conversation_history", "model", "TEXT DEFAULT NULL"
        )
        await self._migrate_add_column(
            "conversation_history", "provider", "TEXT DEFAULT NULL"
        )
        # #1710 follow-up: if a pending A2A question observes a terminal
        # answer but fails to enqueue the local resumption signal, keep the
        # terminal payload on the WAITING row so replay/sweeps can retry the
        # actual answer rather than degrade it to an empty expiry.
        await self._migrate_add_column(
            "pending_a2a_questions", "retry_state", "TEXT DEFAULT NULL"
        )
        await self._migrate_add_column(
            "pending_a2a_questions", "retry_reply_text", "TEXT DEFAULT NULL"
        )
        # Decay-aware forgetting (#1674): memory_episodes need a decay signal
        # before they can be deleted by importance rather than raw age. Stamp
        # the consolidator's avg-importance per episode; legacy episodes default
        # to 0.5 (neutral) so they decay on the median half-life until rewritten.
        await self._migrate_add_column(
            "memory_episodes", "importance", "REAL DEFAULT 0.5"
        )
        # Relevance-based episode recall + access tracking (#1674 P2): episodes
        # carry an embedding of their title+summary (reusing the shared vector
        # backend, same path as saved_items) so genuinely-relevant past episodes
        # can resurface; access_count is the rehearsal signal fed into decay so
        # consulted episodes resist the deletion tier. Legacy episodes get NULL
        # embeddings (recall falls back to keyword) and access_count 0.
        await self._migrate_add_column(
            "memory_episodes", "access_count", "INTEGER DEFAULT 0"
        )
        # embedding_vec is BLOB on SQLite, BYTEA on Postgres (a native pgvector
        # column is a later phase, mirroring saved_items #1447; the SQLite
        # PurePythonBackend and the keyword fallback both work today).
        _episode_blob_type = (
            "BYTEA" if self.backend_type == "postgres" else "BLOB"
        )
        await self._migrate_add_column(
            "memory_episodes", "embedding_vec", _episode_blob_type
        )
        await self._migrate_add_column(
            "memory_episodes", "embedding_profile_id", "TEXT DEFAULT NULL"
        )
        # Managed service-key storage: agent_service_keys gained key_hash +
        # quota columns after its initial release, but the table is created
        # with CREATE TABLE IF NOT EXISTS, so databases predating those columns
        # never received them. store_key then fails its INSERT ("column
        # key_hash of relation agent_service_keys does not exist"), and because
        # per-user provider-key injection is fail-open, callers silently fall
        # back to the shared platform key — disabling per-user metering/caps
        # with no hard error. Migrate the post-initial columns idempotently.
        # key_hash is nullable here (not NOT NULL as in the CREATE TABLE) so the
        # ALTER also succeeds on tables that already hold legacy rows; store_key
        # always supplies it on new inserts.
        await self._migrate_add_column(
            "agent_service_keys", "key_hash", "TEXT"
        )
        await self._migrate_add_column(
            "agent_service_keys", "quota_limit", "INTEGER"
        )
        await self._migrate_add_column(
            "agent_service_keys", "quota_used", "INTEGER DEFAULT 0"
        )
        await self._migrate_add_column(
            "agent_service_keys", "is_active", "INTEGER DEFAULT 1"
        )
        if await self._column_exists("conversation_history", "deleted_at"):
            await self._backend.execute(
                "CREATE INDEX IF NOT EXISTS idx_conversation_deleted_at "
                "ON conversation_history(agent_id, deleted_at)"
            )
        else:
            logger.error(
                "Skipping idx_conversation_deleted_at: column missing after "
                "migration. This indicates a migration failure — see "
                "preceding logs."
            )
        if await self._column_exists("conversation_history", "archived_at"):
            await self._backend.execute(
                "CREATE INDEX IF NOT EXISTS idx_conversation_archived_at "
                "ON conversation_history(agent_id, archived_at)"
            )
        else:
            logger.error(
                "Skipping idx_conversation_archived_at: column missing after "
                "migration. This indicates a migration failure — see "
                "preceding logs."
            )

        # Phase 2 of #1447: add a parallel ``saved_items.embedding_vec``
        # column for the SQLA + pgvector code path. On PG it's
        # ``vector(N)`` indexed with HNSW; on SQLite it's BLOB. The
        # legacy ``embedding`` BYTEA / BLOB column stays — raw IO in
        # SavedItemsStore continues to use it, and ``save_item``'s
        # dual-write keeps both in sync.
        # Idempotent: skips cleanly if the column already exists.
        # Wrapped in ``db.transaction()`` internally so any partial
        # failure rolls back. See sqla/migrations.py for details.
        try:
            from .sqla.migrations import migrate_saved_items_add_embedding_vec
            await migrate_saved_items_add_embedding_vec(self)
        except Exception as e:
            # Migration failure is non-fatal for startup — saved-items
            # search falls back to the legacy in-Python path via
            # SavedItemsStore's existing fallback chain. Log loudly so
            # operators see it on the next boot.
            logger.error(
                "Phase-2 saved_items embedding_vec migration failed: %s. "
                "Falling back to BYTEA + PurePythonBackend until next "
                "boot.", e, exc_info=True,
            )

        # Same Phase-2 treatment for ``document_chunks`` (the
        # AsyncRAGStore backing table). Idempotent + transactional +
        # dim-sniffed, just like the saved_items migration above.
        # Independent try/except so a failure on one table doesn't
        # block the other from migrating.
        try:
            from .sqla.migrations import migrate_document_chunks_add_embedding_vec
            await migrate_document_chunks_add_embedding_vec(self)
        except Exception as e:
            logger.error(
                "Phase-2 document_chunks embedding_vec migration failed: %s. "
                "AsyncRAGStore search falls back to in-Python cosine until "
                "next boot.", e, exc_info=True,
            )

        # Greenfield ``embedding_vec`` column on ``conversation_history``
        # for the SQLAlchemy/vector path that will back
        # ``MemoryRetriever`` cosine scoring. No legacy embedding
        # column exists to migrate from; the dim is taken from
        # ``KESTREL_EMBEDDING_DIM`` (default 768 for Ollama
        # ``nomic-embed-text``). Idempotent + transactional; failure
        # is non-fatal because the current retriever still falls back
        # to keyword/concept overlap.
        try:
            from .sqla.migrations import (
                migrate_conversation_history_add_embedding_vec,
            )
            await migrate_conversation_history_add_embedding_vec(self)
        except Exception as e:
            logger.error(
                "conversation_history embedding_vec migration failed: %s. "
                "MemoryRetriever falls back to keyword-overlap semantic "
                "scoring until next boot.", e, exc_info=True,
            )

        # #1477 — add ``embedding_profile_id`` column to all three
        # embedded tables + create the ``embedding_profiles``
        # registry. Each migration is idempotent + transactional
        # internally; failures here are non-fatal because the kNN
        # filter treats a missing column as "no profile id to
        # match" (column missing → SELECT errors out → caller
        # falls through to keyword search). Operators see clear
        # logs on the next boot.
        try:
            from .sqla.migrations import (
                migrate_add_embedding_profile_id,
                migrate_create_embedding_profiles,
                migrate_embedding_profiles_add_parity,
            )
            await migrate_create_embedding_profiles(self)
            await migrate_embedding_profiles_add_parity(self)
            for table in (
                "conversation_history",
                "saved_items",
                "document_chunks",
            ):
                await migrate_add_embedding_profile_id(self, table=table)
        except Exception as e:
            logger.error(
                "embedding_profile_id migration failed: %s. "
                "Storage writes will leave the column NULL; profile-"
                "filtered kNN reads will skip those rows. Operators "
                "should re-run ``kestrel-sovereign embeddings reindex`` "
                "once the underlying cause is fixed.", e, exc_info=True,
            )

        # #2339 — keyed blind-token index for exact recall over encrypted
        # conversation rows. Missing coverage remains on the full-scan fallback,
        # so migration failure degrades performance rather than correctness.
        try:
            from .sqla.migrations import migrate_conversation_lexical_index
            await migrate_conversation_lexical_index(self)
        except Exception as e:
            logger.error(
                "conversation lexical-index migration failed: %s. Exact "
                "legacy recall will keep using the decrypt-scan bridge until "
                "the next successful migration.", e, exc_info=True,
            )

        # compress → compact terminology rename: rewrite persisted
        # session-compaction metadata strings (marker ``type`` values,
        # ``salvage_reason``, key renames) so readers never need
        # dual-string compat. Idempotent + transactional; non-fatal
        # because untouched legacy rows only degrade transcript
        # annotations and consolidator marker-state checks, not boot.
        try:
            from .sqla.migrations import migrate_compaction_terminology
            await migrate_compaction_terminology(self)
        except Exception as e:
            logger.error(
                "compaction-terminology migration failed: %s. Legacy "
                "'compression' metadata rows remain until next boot.",
                e, exc_info=True,
            )

        # #2012: relink conversation messages whose session_id was stored as a
        # bare integer (the list endpoint's old row-id key, echoed back by the
        # UI) to the canonical UUID on the session's new_session marker — in
        # both conversation_history and conversation_titles. Idempotent +
        # transactional; non-fatal because un-relinked rows only degrade the
        # web message pane on refresh, not boot.
        try:
            from .sqla.migrations import migrate_canonical_session_ids
            await migrate_canonical_session_ids(self)
        except Exception as e:
            logger.error(
                "canonical-session-id migration (#2012) failed: %s. Legacy "
                "integer-keyed messages remain split until next boot.",
                e, exc_info=True,
            )

        logger.debug(f"Database schema initialized ({self.backend_type})")

    async def _backfill_graph_ownership(self) -> None:
        """Backfill authoritative graph ownership without guessing.

        Existing databases used ``properties.agent_id`` for most private
        nodes, but message nodes and agent roots were known exceptions.  Edge
        ownership was previously inferred from endpoint membership, which is
        unsafe for cross-tenant links and loses valid edges to shared nodes.
        This migration records only mechanically provable ownership.
        """

        def insert_ignore(table: str, columns: str, select_sql: str) -> str:
            if self.backend_type == "postgres":
                return (
                    f"INSERT INTO {table} ({columns}) {select_sql} "
                    "ON CONFLICT DO NOTHING"
                )
            return f"INSERT OR IGNORE INTO {table} ({columns}) {select_sql}"

        node_agent = (
            "(properties::jsonb->>'agent_id')"
            if self.backend_type == "postgres"
            else "json_extract(properties, '$.agent_id')"
        )
        root_constitution_hash = (
            "(root.properties::jsonb->>'constitution_hash')"
            if self.backend_type == "postgres"
            else "json_extract(root.properties, '$.constitution_hash')"
        )

        async with self.transaction():
            # Explicit legacy tags remain authoritative.
            await self.execute(
                insert_ignore(
                    "graph_node_owners",
                    "node_id, agent_id",
                    "SELECT node_id, " + node_agent + " FROM graph_nodes "
                    "WHERE " + node_agent + " IS NOT NULL "
                    "AND " + node_agent + " <> ''",
                )
            )

            # The DID-keyed agent row is its own canonical owner.
            await self.execute(
                insert_ignore(
                    "graph_node_owners",
                    "node_id, agent_id",
                    "SELECT node_id, node_id FROM graph_nodes "
                    "WHERE node_type = 'agent' AND node_id <> ''",
                )
            )

            # These writers encode the complete agent DID between a stable
            # prefix and the next colon.  Compare by exact prefix length rather
            # than LIKE so hostile '%'/'_' identifiers cannot act as wildcards.
            for prefix in ("message:", "concept:", "action:", "decision:", "graduation:"):
                prefix_len = len(prefix)
                await self.execute(
                    insert_ignore(
                        "graph_node_owners",
                        "node_id, agent_id",
                        "SELECT n.node_id, roots.agent_id "
                        "FROM graph_nodes n "
                        "JOIN graph_node_owners roots ON roots.node_id = roots.agent_id "
                        "WHERE substr(n.node_id, 1, ? + length(roots.agent_id) + 1) "
                        "= ? || roots.agent_id || ':'",
                    ),
                    (prefix_len, prefix),
                )

            # The pre-v2 identity importer namespaced relationship and skill
            # nodes as ``{agent_id[:20]}_<raw-id>``.  The prefix alone is not
            # ownership proof because many did:pkh identities share those
            # first 20 characters.  Admit it only when exactly one canonical
            # agent root references the node; a collision referenced by two
            # roots deliberately remains unowned.
            await self.execute(
                insert_ignore(
                    "graph_node_owners",
                    "node_id, agent_id",
                    "SELECT n.node_id, roots.agent_id "
                    "FROM graph_nodes n "
                    "JOIN graph_edges e ON e.target_id = n.node_id "
                    "JOIN graph_node_owners roots "
                    "  ON roots.node_id = e.source_id "
                    " AND roots.agent_id = e.source_id "
                    "WHERE n.node_type IN ('user', 'skill') "
                    "AND substr(n.node_id, 1, length(substr(roots.agent_id, 1, 20)) + 1) "
                    "    = substr(roots.agent_id, 1, 20) || '_' "
                    "AND NOT EXISTS ("
                    "  SELECT 1 FROM graph_edges other_e "
                    "  JOIN graph_node_owners other_roots "
                    "    ON other_roots.node_id = other_e.source_id "
                    "   AND other_roots.agent_id = other_e.source_id "
                    "  WHERE other_e.target_id = n.node_id "
                    "    AND other_roots.agent_id <> roots.agent_id"
                    ")",
                )
            )

            # Legacy migration records were not namespaced.  A migrated_via
            # edge from exactly one self-owned agent root is the durable proof
            # of ownership; duplicate/colliding references fail closed.
            await self.execute(
                insert_ignore(
                    "graph_node_owners",
                    "node_id, agent_id",
                    "SELECT n.node_id, roots.agent_id "
                    "FROM graph_nodes n "
                    "JOIN graph_edges e "
                    "  ON e.target_id = n.node_id AND e.label = 'migrated_via' "
                    "JOIN graph_node_owners roots "
                    "  ON roots.node_id = e.source_id "
                    " AND roots.agent_id = e.source_id "
                    "WHERE n.node_type = 'migration_record' "
                    "AND NOT EXISTS ("
                    "  SELECT 1 FROM graph_edges other_e "
                    "  JOIN graph_node_owners other_roots "
                    "    ON other_roots.node_id = other_e.source_id "
                    "   AND other_roots.agent_id = other_e.source_id "
                    "  WHERE other_e.target_id = n.node_id "
                    "    AND other_e.label = 'migrated_via' "
                    "    AND other_roots.agent_id <> roots.agent_id"
                    ")",
                )
            )

            await self.execute(
                insert_ignore(
                    "graph_node_owners",
                    "node_id, agent_id",
                    "SELECT n.node_id, roots.agent_id "
                    "FROM graph_nodes n "
                    "JOIN graph_node_owners roots ON roots.node_id = roots.agent_id "
                    "WHERE n.node_id = roots.agent_id || '#soul'",
                )
            )

            # Consolidated episode graph nodes use the same stable id as the
            # authoritative memory_episodes row, whose physical agent_id
            # column predates the graph ownership ledger.
            await self.execute(
                insert_ignore(
                    "graph_node_owners",
                    "node_id, agent_id",
                    "SELECT n.node_id, episodes.agent_id "
                    "FROM graph_nodes n "
                    "JOIN memory_episodes episodes ON episodes.id = n.node_id "
                    "WHERE n.node_type = 'episode' "
                    "AND episodes.agent_id <> ''",
                )
            )

            # A legacy constitution node is shared content, but each root can
            # prove its own ownership reference by matching the target to the
            # hash persisted on that exact root.  This is stronger than
            # trusting a governed_by label on an arbitrary edge.
            await self.execute(
                insert_ignore(
                    "graph_node_owners",
                    "node_id, agent_id",
                    "SELECT target.node_id, roots.agent_id "
                    "FROM graph_edges e "
                    "JOIN graph_nodes root ON root.node_id = e.source_id "
                    "JOIN graph_nodes target ON target.node_id = e.target_id "
                    "JOIN graph_node_owners roots "
                    "  ON roots.node_id = root.node_id "
                    " AND roots.agent_id = root.node_id "
                    "WHERE e.label = 'governed_by' "
                    "AND root.node_type = 'agent' "
                    "AND target.node_type = 'document' "
                    "AND " + root_constitution_hash + " = e.target_id",
                )
            )

            # A spawned child authors its outbound lineage witness even when
            # the parent node is absent from the child's private database or is
            # owned only by the parent in a shared database. The canonical
            # agent-root source proves which child owns this intentional
            # cross-agent edge; endpoint co-ownership is not required.
            await self.execute(
                insert_ignore(
                    "graph_edge_owners",
                    "source_id, target_id, label, agent_id",
                    "SELECT e.source_id, e.target_id, e.label, roots.agent_id "
                    "FROM graph_edges e "
                    "JOIN graph_node_owners roots "
                    "  ON roots.node_id = e.source_id "
                    " AND roots.agent_id = e.source_id "
                    "WHERE e.label = 'spawned_by'",
                )
            )

            # An ownerless legacy edge can be assigned only when its endpoints
            # have exactly one common owner.  Existing edge ownership is
            # authoritative and independent of endpoint ownership: rerunning
            # this migration must never grant a newly-added common node owner
            # access to another tenant's already-owned edge payload.
            await self.execute(
                insert_ignore(
                    "graph_edge_owners",
                    "source_id, target_id, label, agent_id",
                    "SELECT e.source_id, e.target_id, e.label, MIN(src.agent_id) "
                    "FROM graph_edges e "
                    "JOIN graph_node_owners src ON src.node_id = e.source_id "
                    "JOIN graph_node_owners dst "
                    "  ON dst.node_id = e.target_id "
                    " AND dst.agent_id = src.agent_id "
                    "WHERE NOT EXISTS ("
                    "  SELECT 1 FROM graph_edge_owners existing_owner "
                    "  WHERE existing_owner.source_id = e.source_id "
                    "    AND existing_owner.target_id = e.target_id "
                    "    AND existing_owner.label = e.label"
                    ") "
                    "GROUP BY e.source_id, e.target_id, e.label "
                    "HAVING COUNT(DISTINCT src.agent_id) = 1",
                )
            )

    async def _backfill_file_ownership(self) -> None:
        """Backfill only legacy files anchored by a canonical agent root.

        Generic blobs historically had no authoritative owner.  Metadata JSON
        is intentionally not trusted for migration because it was not an
        isolation boundary and may be caller-controlled. An orphan graph node
        is insufficient too: a tenant could name it after a known foreign
        content hash. Only a canonical constitution/avatar pointer on a
        self-owned agent root, with its matching relationship and typed target,
        is accepted; ambiguous roots fail closed.
        """

        root_constitution_hash = (
            "(root.properties::jsonb->>'constitution_hash')"
            if self.backend_type == "postgres"
            else "json_extract(root.properties, '$.constitution_hash')"
        )
        root_avatar_hash = (
            "(root.properties::jsonb->>'avatar_hash')"
            if self.backend_type == "postgres"
            else "json_extract(root.properties, '$.avatar_hash')"
        )
        target_avatar_hash = (
            "COALESCE(target.properties::jsonb->>'hash', e.target_id)"
            if self.backend_type == "postgres"
            else "COALESCE(json_extract(target.properties, '$.hash'), e.target_id)"
        )
        proven_owners = f"""
            SELECT DISTINCT e.target_id AS content_hash,
                   roots.agent_id AS agent_id,
                   'constitution' AS reference_kind
            FROM graph_edges e
            JOIN graph_nodes root ON root.node_id = e.source_id
            JOIN graph_nodes target ON target.node_id = e.target_id
            JOIN graph_node_owners roots
              ON roots.node_id = root.node_id
             AND roots.agent_id = root.node_id
            WHERE e.label = 'governed_by'
              AND root.node_type = 'agent'
              AND target.node_type = 'document'
              AND target.label = 'KESTREL_CONSTITUTION'
              AND {root_constitution_hash} = e.target_id
            UNION ALL
            SELECT avatar_witnesses.content_hash,
                   MIN(avatar_witnesses.agent_id) AS agent_id,
                   'avatar' AS reference_kind
            FROM (
                SELECT {target_avatar_hash} AS content_hash,
                       roots.agent_id AS agent_id
                FROM graph_edges e
                JOIN graph_nodes root ON root.node_id = e.source_id
                JOIN graph_nodes target ON target.node_id = e.target_id
                JOIN graph_node_owners roots
                  ON roots.node_id = root.node_id
                 AND roots.agent_id = root.node_id
                WHERE e.label = 'has_avatar'
                  AND root.node_type = 'agent'
                  AND target.node_type = 'avatar'
                  AND {root_avatar_hash} = {target_avatar_hash}
            ) avatar_witnesses
            GROUP BY avatar_witnesses.content_hash
            HAVING COUNT(DISTINCT avatar_witnesses.agent_id) = 1
        """

        if self.backend_type == "postgres":
            sql = f"""
                INSERT INTO file_owners
                    (content_hash, agent_id, original_name, metadata)
                SELECT f.content_hash, owners.agent_id,
                       CASE WHEN owners.reference_kind = 'constitution'
                            THEN 'KESTREL_CONSTITUTION.md'
                            ELSE f.original_name END,
                       CASE WHEN owners.reference_kind = 'constitution'
                            THEN '{{}}'
                            ELSE f.metadata END
                FROM files f
                JOIN ({proven_owners}) owners
                  ON owners.content_hash = f.content_hash
                WHERE NOT EXISTS (
                      SELECT 1 FROM file_owners existing_owner
                      WHERE existing_owner.content_hash = f.content_hash
                        AND existing_owner.agent_id = owners.agent_id
                  )
                ON CONFLICT DO NOTHING
            """
        else:
            sql = f"""
                INSERT OR IGNORE INTO file_owners
                    (content_hash, agent_id, original_name, metadata)
                SELECT f.content_hash, owners.agent_id,
                       CASE WHEN owners.reference_kind = 'constitution'
                            THEN 'KESTREL_CONSTITUTION.md'
                            ELSE f.original_name END,
                       CASE WHEN owners.reference_kind = 'constitution'
                            THEN '{{}}'
                            ELSE f.metadata END
                FROM files f
                JOIN ({proven_owners}) owners
                  ON owners.content_hash = f.content_hash
                WHERE NOT EXISTS (
                      SELECT 1 FROM file_owners existing_owner
                      WHERE existing_owner.content_hash = f.content_hash
                        AND existing_owner.agent_id = owners.agent_id
                  )
            """
        await self.execute(sql)

    async def _backfill_document_chunk_ownership(self) -> None:
        """Assign legacy chunks only from an unambiguous file capability.

        Resolve single-owner files FIRST (aggregate ``file_owners`` by
        content_hash), then join to chunks. Grouping after the chunk×owner
        join instead — the original shape — explodes on a content_hash owned
        by many agents (e.g. a shared default/constitution blob): 26k chunks ×
        1.4k owners is tens of millions of rows just to discard them via
        ``HAVING COUNT(DISTINCT ...) = 1``. Pre-aggregating keeps the driving
        set at one row per single-owner file. Semantically identical: a chunk
        is owned iff its file has exactly one distinct owner.
        """

        insert = (
            "INSERT INTO document_chunk_owners (chunk_id, agent_id)"
            if self.backend_type == "postgres"
            else "INSERT OR IGNORE INTO document_chunk_owners (chunk_id, agent_id)"
        )
        conflict = " ON CONFLICT DO NOTHING" if self.backend_type == "postgres" else ""
        sql = f"""
            {insert}
            SELECT chunks.chunk_id, single_owner.agent_id
            FROM document_chunks chunks
            JOIN (
                SELECT content_hash, MIN(agent_id) AS agent_id
                FROM file_owners
                GROUP BY content_hash
                HAVING COUNT(DISTINCT agent_id) = 1
            ) single_owner
              ON single_owner.content_hash = chunks.file_hash
            WHERE NOT EXISTS (
                SELECT 1 FROM document_chunk_owners existing_owner
                WHERE existing_owner.chunk_id = chunks.chunk_id
            ){conflict}
        """
        await self.execute(sql)

    async def _backfill_completed(self, name: str) -> bool:
        """True if the named one-time backfill has already recorded success.

        Backed by the ``schema_backfills`` marker table (created in
        CORE_SCHEMA, so it exists before this is called). Lets ``_init_schema``
        skip expensive one-time migrations on the vast majority of
        ``from_pool()`` calls instead of re-scanning every time.
        """
        row = await self._backend.fetch_one(
            "SELECT 1 FROM schema_backfills WHERE name = ?", (name,)
        )
        return row is not None

    async def _mark_backfill_completed(self, name: str) -> None:
        """Record that the named one-time backfill has completed. Idempotent."""
        if self.backend_type == "postgres":
            await self._backend.execute(
                "INSERT INTO schema_backfills (name) VALUES (?) "
                "ON CONFLICT DO NOTHING",
                (name,),
            )
        else:
            await self._backend.execute(
                "INSERT OR IGNORE INTO schema_backfills (name) VALUES (?)",
                (name,),
            )

    async def _run_ownership_backfills_once(self, name: str) -> None:
        """Run the #2649 ownership backfills exactly once, serialized.

        ``_init_schema`` runs on every ``from_pool()`` (frinz calls it per
        request), so a post-upgrade request burst would otherwise have every
        initializer observe no marker and run the heavy scans concurrently —
        the lock contention/timeout this fix targets. Take a transaction-scoped
        advisory lock (Postgres) so exactly one initializer performs the
        migration while the rest wait, then re-check the marker under the lock
        and skip. The whole migration + marker commit atomically, so an
        interrupted upgrade leaves no marker and retries on the next boot.
        SQLite needs no advisory lock — its single writer already serializes.
        """
        async with self.transaction():
            if self.backend_type == "postgres":
                await self._backend.execute(
                    "SELECT pg_advisory_xact_lock(?)", (_backfill_lock_id(name),)
                )
            else:
                # SQLite transactions BEGIN deferred, so two initializers could
                # both read a missing marker and the second would raise
                # "database is locked" when it upgrades mid-backfill. Promote to
                # the single write slot with a no-op write BEFORE the marker
                # read (busy_timeout makes the loser wait, then it skips) —
                # mirroring the lexical-cleanup serialization.
                await self._backend.execute(
                    "DELETE FROM schema_backfills WHERE 0"
                )
            # Double-checked: a concurrent initializer may have completed the
            # migration between the fast-path check and acquiring the lock.
            if await self._backfill_completed(name):
                return
            await self._backfill_graph_ownership()
            await self._backfill_file_ownership()
            await self._backfill_document_chunk_ownership()
            await self._mark_backfill_completed(name)

    async def _migrate_add_column(
        self, table: str, column: str, col_def: str
    ) -> None:
        """Add a column to an existing table if it isn't already present.

        Idempotent across both backends — Postgres uses ``ADD COLUMN IF
        NOT EXISTS``; SQLite checks ``pragma_table_info`` first. Real
        failures (locked DB, disk full, syntactically-bad ``col_def``)
        propagate to the caller; previously they were swallowed at debug
        level, which let dependent index creation crash boot with a
        misleading error (#795).
        """
        if self.backend_type == "postgres":
            await self._backend.execute(
                f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {col_def}"
            )
            return

        row = await self._backend.fetch_one(
            f"SELECT COUNT(*) FROM pragma_table_info('{table}') "
            f"WHERE name='{column}'"
        )
        if row and row[0] == 0:
            await self._backend.execute(
                f"ALTER TABLE {table} ADD COLUMN {column} {col_def}"
            )
            logger.info(f"Migrated {table}: added {column} column")

    async def _column_exists(self, table: str, column: str) -> bool:
        """Verify a column exists on a table after a migration ran."""
        if self.backend_type == "postgres":
            row = await self._backend.fetch_one(
                "SELECT COUNT(*) FROM information_schema.columns "
                "WHERE table_name=? AND column_name=?",
                (table, column),
            )
        else:
            row = await self._backend.fetch_one(
                f"SELECT COUNT(*) FROM pragma_table_info('{table}') "
                f"WHERE name='{column}'"
            )
        return bool(row and row[0] > 0)
    
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

    async def execute_script(self, script: str) -> None:
        """Execute a multi-statement SQL script (table creation, migrations).

        Pass-through to the underlying DatabaseBackend. Feature stores
        resolved via :func:`resolve_feature_database` get an AsyncDatabase
        (not the raw backend) and rely on this method for CREATE TABLE
        IF NOT EXISTS blocks. Without it, every feature with a schema
        bootstrap (e.g. kestrel-feature-healthcare's FhirResourceStore)
        raises AttributeError at agent init and the whole agent fails to
        load.
        """
        await self._backend.execute_script(script)
    
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
    
    async def dispose_cached_sqla_factory(self) -> None:
        """Dispose the optional cached SQLAlchemy engine.

        The SQLAlchemy engine is an independent resource from the primary
        :class:`DatabaseBackend` connection.  Keep this phase separately
        callable so shutdown orchestration can bound engine disposal without
        stealing the primary SQLite connection's worker-drain reservation.
        Clearing the cache *before* the await makes a timed-out disposal a
        one-shot attempt: a later backend close must not repeat the same
        unbounded pre-close work and starve the primary connection.
        """
        # If the SQLA helper ever cached a session factory on us
        # (``kestrel_sovereign.storage.sqla.make_session_factory``),
        # dispose its engine first so the underlying connection pool
        # is released cleanly before the AsyncDatabase backend
        # shuts down. The cache is best-effort and may be absent in
        # tests / fresh DBs — guard with ``getattr``.
        sqla_factory = getattr(self, "_sovereign_sqla_factory", None)
        self._sovereign_sqla_factory = None
        if sqla_factory is not None:
            try:
                await sqla_factory.close()
            except Exception as e:
                logger.warning(
                    "Failed to close cached SQLAlchemy session factory: %s", e
                )

    @property
    def minimum_sqla_factory_close_timeout_s(self) -> float:
        """Return the cached SQLAlchemy factory's optional close reservation."""
        factory = getattr(self, "_sovereign_sqla_factory", None)
        value = getattr(factory, "minimum_close_timeout_s", 0.0)
        return value if isinstance(value, (int, float)) and value > 0 else 0.0

    async def close(self) -> None:
        """Close the SQLAlchemy factory and primary backend connection.

        A cancellation during optional SQLAlchemy disposal must not skip the
        primary backend close.  In particular, SQLite owns an aiosqlite worker
        whose shutdown has to run before the caller tears down its event loop.
        Propagate cancellation only after that owned backend lifecycle has had
        its chance to complete.
        """
        cancelled = False
        try:
            await self.dispose_cached_sqla_factory()
        except asyncio.CancelledError:
            cancelled = True

        await self._backend.close()
        self._initialized = False
        logger.debug("Database connection closed")
        if cancelled:
            raise asyncio.CancelledError()
    
    async def __aenter__(self):
        if not self._initialized:
            await self._init_schema()
            self._initialized = True
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
