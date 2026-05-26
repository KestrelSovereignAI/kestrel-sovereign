"""
Privacy-Enforcing Storage Wrapper for Kestrel.

This module provides a storage wrapper that enforces privacy mode restrictions
at the storage layer itself, preventing data leakage by design.

The wrapper intercepts all storage operations and:
1. EPHEMERAL mode: Blocks all write operations (raises PrivacyViolationError)
2. ISOLATED mode: Redirects to in-memory session storage
3. ANONYMOUS mode: Applies PII scrubbing before storage
4. NORMAL/PUBLIC mode: Passes through to underlying storage

This is a defense-in-depth measure - even if application code forgets to
check privacy mode, the storage layer will enforce it.
"""

import json
import logging
import warnings
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING, Union
from enum import Enum
from dataclasses import dataclass

from kestrel_sovereign.privacy import (
    PrivacyMode,
    PrivacyConfig,
    get_privacy_preset,
    privacy_mode_to_config,
    privacy_config_to_mode,
)
from kestrel_sovereign.storage.conversation_ids import coerce_persistent_message_id

# Lazy import to avoid circular dependency with features.privacy
# Note: This global cache is shared across all instances and async contexts.
# The anonymize_text function is stateless and thread-safe, so this is acceptable.
_anonymize_text = None

def get_anonymize_text():
    """Lazy-load the anonymize_text function to avoid circular imports."""
    global _anonymize_text
    if _anonymize_text is None:
        from kestrel_sovereign.features.privacy.pii_detector import anonymize_text
        _anonymize_text = anonymize_text
    return _anonymize_text

logger = logging.getLogger(__name__)


class PrivacyViolationError(Exception):
    """Raised when a storage operation violates the current privacy mode."""
    pass


class OperationType(Enum):
    """Types of storage operations for permission checking."""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"


@dataclass
class PrivacyPolicy:
    """Defines what operations are allowed in each privacy mode."""
    allow_persistent_write: bool
    allow_persistent_read: bool
    require_anonymization: bool
    use_session_storage: bool
    allow_cloud_backup: bool
    
    @staticmethod
    def for_mode(mode: Union[PrivacyMode, PrivacyConfig, str]) -> "PrivacyPolicy":
        """Get the policy for a given privacy mode or config."""
        # Convert to PrivacyConfig
        if isinstance(mode, PrivacyConfig):
            config = mode
        elif isinstance(mode, PrivacyMode):
            config = privacy_mode_to_config(mode)
        elif isinstance(mode, str):
            config = get_privacy_preset(mode)
        else:
            raise TypeError(f"Expected PrivacyMode, PrivacyConfig, or str, got {type(mode)}")
        
        # Build policy from config flags
        return PrivacyPolicy(
            allow_persistent_write=config.allows_persistent_storage(),
            allow_persistent_read=True,  # Reading existing data is always allowed
            require_anonymization=config.requires_anonymization(),
            use_session_storage=config.uses_temp_storage(),
            allow_cloud_backup=config.allows_persistent_storage() and not config.is_ephemeral()
        )
    
    @staticmethod
    def from_config(config: PrivacyConfig) -> "PrivacyPolicy":
        """Build policy directly from PrivacyConfig."""
        return PrivacyPolicy(
            allow_persistent_write=config.allows_persistent_storage(),
            allow_persistent_read=True,
            require_anonymization=config.requires_anonymization(),
            use_session_storage=config.uses_temp_storage(),
            allow_cloud_backup=config.allows_persistent_storage()
        )


class PrivacyEnforcingStorage:
    """
    A storage wrapper that enforces privacy mode at the storage layer.
    
    This is a decorator pattern that wraps AsyncStorage and
    intercepts all operations to enforce privacy policies.
    
    Usage:
        async with AsyncStorage(db_path) as storage:
            privacy_storage = PrivacyEnforcingStorage(storage, PrivacyMode.ANONYMOUS)
            await privacy_storage.add_conversation("user", "Hello")
    """
    
    def __init__(self, underlying_storage, privacy_mode: Union[PrivacyMode, PrivacyConfig, str] = PrivacyMode.NORMAL):
        """
        Initialize the privacy-enforcing wrapper.
        
        Args:
            underlying_storage: The real AsyncStorage instance to wrap
            privacy_mode: Initial privacy mode (PrivacyMode, PrivacyConfig, or preset name)
        """
        self._storage = underlying_storage
        self._privacy_config = self._to_config(privacy_mode)
        self._policy = PrivacyPolicy.from_config(self._privacy_config)
        self._session_conversations: List[Dict] = []
        self._session_files: Dict[str, bytes] = {}
        # ISO-8601 timestamp recorded at the moment the wrapper transitions
        # INTO EPHEMERAL.  The leak-purge (#867) uses this to scope its
        # DELETE so that flipping a long-lived agent into EPHEMERAL for 30
        # seconds doesn't destroy the months of NORMAL history that
        # preceded it — only rows authored on/after this timestamp can be
        # leaks.  None when the agent was never in EPHEMERAL during this
        # process; refreshed each time we re-enter EPHEMERAL.
        self._entered_ephemeral_at: Optional[str] = None
        if self._privacy_config.is_ephemeral():
            self._entered_ephemeral_at = self._now_iso()
        logger.info(f"PrivacyEnforcingStorage initialized with config: storage={self._privacy_config.storage}, llm={self._privacy_config.llm_location}")

    @staticmethod
    def _now_iso() -> str:
        """Watermark format used to scope the EPHEMERAL leak-purge.

        Matches SQLite's ``datetime('now')`` shape (``YYYY-MM-DD HH:MM:SS``,
        UTC, no offset, no microseconds) so a lexicographic ``>=`` compares
        cleanly against the values stored in ``conversation_history.created_at``.
        ISO-8601 with a ``T`` separator and microseconds compares as
        strictly greater than every value the DB writes — that's the bug
        the original implementation hit, where ``2026-04-26T13:24:05.5``
        (watermark) was lexicographically *higher* than
        ``2026-04-26 13:24:06`` (row), so no rows ever matched.
        """
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    @property
    def agent_id(self) -> str:
        """Get the agent_id from underlying storage for multi-tenant isolation."""
        return getattr(self._storage, 'agent_id', '')
    
    def _to_config(self, mode: Union[PrivacyMode, PrivacyConfig, str]) -> PrivacyConfig:
        """Convert various mode representations to PrivacyConfig."""
        if isinstance(mode, PrivacyConfig):
            return mode
        elif isinstance(mode, PrivacyMode):
            return privacy_mode_to_config(mode)
        elif isinstance(mode, str):
            return get_privacy_preset(mode)
        else:
            raise TypeError(f"Expected PrivacyMode, PrivacyConfig, or str, got {type(mode)}")

    @property
    def privacy_mode(self) -> PrivacyMode:
        """Backward compatibility: return PrivacyMode enum."""
        return privacy_config_to_mode(self._privacy_config)
    
    @property
    def privacy_config(self) -> PrivacyConfig:
        """Get the current privacy configuration."""
        return self._privacy_config
    
    # Keep _privacy_mode for backward compatibility in error messages
    @property
    def _privacy_mode(self) -> PrivacyMode:
        return self.privacy_mode
    
    def set_privacy_mode(self, mode: Union[PrivacyMode, PrivacyConfig, str]) -> None:
        """
        Change the privacy mode.

        Note: Changing to a more restrictive mode does NOT delete existing data.
        It only affects future operations.

        Records ``_entered_ephemeral_at`` on every transition INTO
        EPHEMERAL so the leak-purge (#867) can scope its DELETE to rows
        authored *during* the EPHEMERAL stint.  Stale watermarks from a
        prior EPHEMERAL stint are overwritten on re-entry; the watermark
        is preserved across the EPHEMERAL→exit transition because the
        purge needs to read it before clearing.
        """
        old_config = self._privacy_config
        new_config = self._to_config(mode)
        was_ephemeral = old_config.is_ephemeral()
        is_ephemeral = new_config.is_ephemeral()
        if is_ephemeral and not was_ephemeral:
            self._entered_ephemeral_at = self._now_iso()
        self._privacy_config = new_config
        self._policy = PrivacyPolicy.from_config(self._privacy_config)
        logger.info(f"Privacy config changed: storage={old_config.storage}->{self._privacy_config.storage}, llm={old_config.llm_location}->{self._privacy_config.llm_location}")

    async def purge_ephemeral_session(
        self, reason: str = "ephemeral-session-close"
    ) -> Dict[str, int]:
        """Hard-purge any data the EPHEMERAL agent may have leaked (#767).

        EPHEMERAL is the strongest privacy guarantee Kestrel offers —
        the contract is "leave no trace." If a write somehow reached
        ``conversation_history`` or ``graph_nodes`` despite the privacy
        layer rejecting persistent writes, this method is the safety
        net that scrubs it.

        Soft-delete (#763) is for *user delete intent* on data the user
        knew was being persisted. EPHEMERAL is the inverse — the user
        explicitly chose "don't persist." Honor that contract by never
        letting EPHEMERAL data live in trash. We bypass the soft-delete
        code path entirely and call the ``purge_*`` primitives.

        If this method actually finds rows, it WARNs and writes a
        security_audit_log entry — that's a bug in the privacy layer,
        and the audit trail is the only way the operator finds out.

        The session-local in-memory buffer (``_session_conversations``)
        is also cleared as a belt-and-braces measure, in case the
        agent flips out of EPHEMERAL while ISOLATED-style buffering
        accumulated something.

        Args:
            reason: Audit reason. Defaults to ``ephemeral-session-close``.

        Returns:
            Dict of ``{table_name: rows_destroyed}`` so callers can log.
            Zero is the happy path; non-zero is a leak.
        """
        agent_id = self.agent_id
        if not agent_id:
            logger.debug("purge_ephemeral_session: no agent_id, skipping")
            return {"conversation_history": 0, "graph_nodes": 0}

        result: Dict[str, int] = {}

        # Belt-and-braces: clear in-memory ISOLATED buffer too. No row
        # count needed — the buffer never persisted.
        self._session_conversations = []
        self._session_files = {}

        # Scoped to ``_entered_ephemeral_at`` (#867) so the DELETE only
        # touches rows authored *during* the EPHEMERAL stint.  Without
        # this watermark, flipping a long-lived agent into EPHEMERAL for
        # a few seconds and back out destroyed every preexisting NORMAL
        # row — that's the wipe that prompted the scoping fix.  When the
        # watermark is missing (e.g. the agent was already EPHEMERAL
        # before the wrapper was constructed and we never observed the
        # transition), the scoped purge primitives refuse to delete
        # anything rather than fall back to the unbounded behaviour.
        since = self._entered_ephemeral_at
        if not since:
            logger.warning(
                "purge_ephemeral_session: no entered_ephemeral_at watermark "
                "(agent=%s, reason=%s) — refusing to purge to avoid the "
                "wipe-on-shutdown bug fixed in #867",
                agent_id, reason,
            )
            return {"conversation_history": 0, "graph_nodes": 0}

        try:
            convs = await self._storage.purge_conversations_since(
                since, reason=reason,
            )
        except Exception as e:
            logger.warning(
                "purge_ephemeral_session: conversation purge failed: %s", e
            )
            convs = 0
        result["conversation_history"] = convs

        try:
            nodes = await self._storage.purge_agent_graph_nodes(
                since_iso=since,
            )
        except Exception as e:
            logger.warning(
                "purge_ephemeral_session: graph_nodes purge failed: %s", e
            )
            nodes = 0
        result["graph_nodes"] = nodes

        leaked = sum(result.values())
        if leaked > 0:
            logger.warning(
                "[privacy] WARNING: EPHEMERAL session leaked %d row(s) "
                "into persistent storage (agent=%s, since=%s, breakdown=%s); "
                "hard-purged with reason=%s",
                leaked, agent_id, since, result, reason,
            )
        else:
            logger.debug(
                "purge_ephemeral_session: clean (no leaks) for agent %s "
                "since %s",
                agent_id, since,
            )

        # Clear the watermark — the EPHEMERAL stint is over.  Re-entering
        # EPHEMERAL refreshes it via :meth:`set_privacy_mode`.
        self._entered_ephemeral_at = None

        # Audit-log emission is the caller's responsibility — the agent
        # has natural access to its SecurityFeature; the storage wrapper
        # doesn't and shouldn't try to reach back through layers.
        return result
    
    async def _check_write_permission(self, operation_name: str) -> None:
        """Check if write operations are allowed in current mode."""
        if not self._policy.allow_persistent_write and not self._policy.use_session_storage:
            raise PrivacyViolationError(
                f"Operation '{operation_name}' blocked: persistent writes are disabled in "
                f"current privacy config (storage={self._privacy_config.storage})"
            )
    
    def _anonymize_if_required(self, content: str) -> str:
        """Anonymize content if required by current policy."""
        if self._policy.require_anonymization:
            anonymize_text = get_anonymize_text()
            return anonymize_text(content)
        return content
    
    # === Conversation Storage (Privacy-Sensitive) ===
    
    async def add_conversation(self, role: str, content: str, metadata: Optional[Dict] = None,
                               session_id: Optional[str] = None,
                               rendered_content: Optional[str] = None) -> None:
        """
        Add a conversation entry, respecting privacy mode.

        - EPHEMERAL: Raises PrivacyViolationError (use in-memory buffer instead)
        - ISOLATED: Stores in session-local list
        - ANONYMOUS: Anonymizes content before storing
        - NORMAL/PUBLIC: Stores as-is

        Args:
            rendered_content: Write-once transport bytes for byte-stable
                cache replay (#1402); anonymized identically to ``content``
                so the redacted bytes match what was actually sent.
        """
        if self._privacy_config.is_ephemeral():
            raise PrivacyViolationError(
                "Cannot store conversations in ephemeral mode. "
                "Use EphemeralSession for in-memory buffering."
            )

        processed_content = self._anonymize_if_required(content)
        processed_rendered = (
            self._anonymize_if_required(rendered_content)
            if rendered_content is not None else None
        )

        if metadata is None:
            metadata = {}
        metadata["privacy_mode"] = self._privacy_mode.value

        if self._policy.use_session_storage:
            # Store in session-local list (ISOLATED mode). Session-local
            # buffer is in-memory and never replayed for cache hits, so
            # the rendered form is intentionally dropped here.
            self._session_conversations.append({
                "role": role,
                "content": processed_content,
                "metadata": metadata,
                "session_id": session_id
            })
            logger.debug(f"Conversation stored in session ({len(self._session_conversations)} total)")
        else:
            # Store in persistent storage
            await self._storage.add_conversation(
                role, processed_content, metadata, session_id,
                rendered_content=processed_rendered,
            )
    
    async def resolve_session_id(self, provided: Optional[str]) -> Optional[str]:
        """Surface the effective session_id to the caller.

        EPHEMERAL: no persistence, return whatever was provided (or None).
        ISOLATED: session-local; the in-memory buffer doesn't expose
        time-gap heuristics, so an explicit value passes through and
        ``None`` stays ``None``.
        NORMAL/PUBLIC: delegate to the persistent store, which applies
        the 30-min-gap heuristic.
        """
        if self._privacy_config.is_ephemeral() or self._policy.use_session_storage:
            return provided
        return await self._storage.resolve_session_id(provided)

    async def get_conversation_history(
        self, limit: int = 100, session_id: str = None
    ) -> List[Dict]:
        """
        Get conversation history respecting privacy mode.

        Args:
            limit: Maximum number of messages to return
            session_id: If provided, get messages from this session only

        - ISOLATED: Returns session-local history only (ignores session_id)
        - Others: Returns from persistent storage
        """
        if self._policy.use_session_storage:
            return self._session_conversations[-limit:]
        return await self._storage.get_conversation_history(limit, session_id=session_id)
    
    def clear_session(self) -> int:
        """
        Clear session-local storage (for ISOLATED mode).
        
        Returns:
            Number of items cleared
        """
        count = len(self._session_conversations)
        self._session_conversations.clear()
        self._session_files.clear()
        logger.info(f"Session cleared: {count} conversations")
        return count
    
    async def save_session_to_persistent(self) -> int:
        """
        Save session-local storage to persistent storage (promote ISOLATED to NORMAL).
        
        Returns:
            Number of items saved
        """
        count = 0
        for conv in self._session_conversations:
            await self._storage.add_conversation(
                conv["role"],
                conv["content"],
                conv.get("metadata"),
                conv.get("session_id")
            )
            count += 1
        self._session_conversations.clear()
        logger.info(f"Session saved to persistent storage: {count} conversations")
        return count
    
    # === File Storage (Privacy-Sensitive) ===
    
    async def store_file(self, content: bytes, original_name: str, metadata: Optional[Dict] = None) -> str:
        """
        Store a file, respecting privacy mode.
        
        - EPHEMERAL: Raises PrivacyViolationError
        - ISOLATED: Stores in session-local dict
        - Others: Stores in persistent storage
        """
        await self._check_write_permission("store_file")
        
        if self._policy.use_session_storage:
            # Generate hash for session storage
            import hashlib
            content_hash = hashlib.sha256(content).hexdigest()
            self._session_files[content_hash] = content
            logger.debug(f"File stored in session: {original_name} ({content_hash[:8]}...)")
            return content_hash
        
        return await self._storage.store_file(content, original_name, metadata)
    
    async def retrieve_file(self, content_hash: str) -> bytes:
        """
        Retrieve a file, checking session storage first in ISOLATED mode.
        """
        if self._policy.use_session_storage and content_hash in self._session_files:
            return self._session_files[content_hash]
        return await self._storage.retrieve_file(content_hash)
    
    # === Graph Storage (Pass-through - structural, not PII-sensitive) ===

    async def add_node(self, node) -> None:
        """
        Add a graph node.

        Graph operations are allowed even in EPHEMERAL mode because they are
        structural metadata (agent nodes, edges) rather than user content.
        The privacy protection focuses on conversation history and file storage.
        """
        # Graph operations bypass privacy check - they're structural, not PII
        await self._storage.add_node(node)
    
    async def get_node(self, node_id: str):
        """Get a graph node."""
        return await self._storage.get_node(node_id)
    
    async def add_edge(self, source_id: str, target_id: str, label: str, properties: Optional[Dict] = None):
        """Add a graph edge. Structural, not PII-sensitive."""
        await self._storage.add_edge(source_id, target_id, label, properties)

    async def delete_node(self, node_id: str) -> None:
        """Delete a graph node and its edges. Structural operation."""
        await self._storage.delete_node(node_id)

    async def get_edges_from(self, node_id: str) -> List:
        """Get outgoing edges from a node."""
        return await self._storage.get_edges_from(node_id)

    async def get_edges_to(self, node_id: str) -> List:
        """Get incoming edges to a node."""
        return await self._storage.get_edges_to(node_id)

    # === RAG Storage ===
    
    async def chunk_document(self, content_hash: str) -> int:
        """Chunk a document for RAG. Respects privacy mode."""
        await self._check_write_permission("chunk_document")
        return await self._storage.chunk_document(content_hash)
    
    async def search_chunks(self, query: str, limit: int = 5) -> List[Dict]:
        """Search document chunks. Read-only, always allowed."""
        return await self._storage.search_chunks(query, limit)
    
    # === Backup Operations (Privacy-Sensitive) ===
    
    async def create_backup_blob(self, include_db: bool = True) -> bytes:
        """
        Create a backup blob.
        
        - EPHEMERAL/ISOLATED: Raises PrivacyViolationError
        - Others: Creates backup
        """
        if not self._policy.allow_cloud_backup:
            raise PrivacyViolationError(
                f"Backups are disabled in current privacy config (storage={self._privacy_config.storage})"
            )
        return await self._storage.create_backup_blob(include_db)
    
    async def restore_from_backup_blob(self, backup_blob: bytes) -> Dict:
        """Restore from a backup blob."""
        await self._check_write_permission("restore_from_backup_blob")
        return await self._storage.restore_from_backup_blob(backup_blob)
    
    async def record_backup_artifact(self, agent_id: str, result: Any) -> str:
        """Record a backup artifact in the graph store."""
        await self._check_write_permission("record_backup_artifact")
        return await self._storage.record_backup_artifact(agent_id, result)
    
    async def get_nodes_by_type(self, node_type: str) -> List:
        """Get all nodes of a specific type."""
        return await self._storage.get_nodes_by_type(node_type)
    
    async def get_file_metadata(self, content_hash: str) -> Optional[Dict]:
        """Get file metadata."""
        return await self._storage.get_file_metadata(content_hash)
    
    async def search_case_law(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search case law (constitutional RAG)."""
        return await self._storage.search_case_law(query, top_k)
    
    @property
    def encryption_enabled(self) -> bool:
        """Check if conversation encryption at rest is enabled.

        This provides a safe way to check encryption status without
        accessing the conversation store directly.
        """
        conv_store = getattr(self._storage, 'conversation', None)
        if conv_store and hasattr(conv_store, 'encryption_enabled'):
            return conv_store.encryption_enabled
        return False

    # === Privacy-Aware Query Methods ===
    #
    # These methods provide privacy-respecting access to the database for
    # operations that were previously done via direct storage.db access.
    # Use these instead of accessing .db, .conversation, or .files directly.

    async def query_conversations(
        self, agent_id: str, limit: int = 50
    ) -> List[Tuple]:
        """
        Query conversation history rows respecting privacy mode.

        In EPHEMERAL mode, returns an empty list (no persistent data exposed).
        In ISOLATED mode, returns session-local conversations as tuple rows.
        In other modes, queries the persistent database.

        Returns rows as tuples: (id, role, content, metadata, created_at)
        """
        if self._privacy_config.is_ephemeral():
            logger.debug("query_conversations blocked: ephemeral mode returns no data")
            return []

        if self._policy.use_session_storage:
            # Return session conversations formatted as tuple rows
            rows = []
            for i, conv in enumerate(self._session_conversations):
                rows.append((
                    i,  # synthetic id
                    conv.get("role", ""),
                    conv.get("content", ""),
                    json.dumps(conv.get("metadata", {})) if conv.get("metadata") else None,
                    conv.get("created_at", None),
                ))
            return rows

        return await self._storage.db.fetchall("""
            SELECT id, role, content, metadata, created_at
            FROM conversation_history
            WHERE agent_id = ? AND deleted_at IS NULL
            ORDER BY created_at DESC
        """, (agent_id,))

    async def query_conversation_start(
        self, message_id: str, agent_id: str
    ) -> Optional[Tuple]:
        """
        Get the created_at timestamp for a specific message, respecting privacy.

        In EPHEMERAL mode, returns None.
        In ISOLATED mode, returns from session storage.
        Otherwise queries the persistent database.

        Returns a single-element tuple (created_at,) or None.
        """
        if self._privacy_config.is_ephemeral():
            return None

        if self._policy.use_session_storage:
            try:
                idx = int(message_id)
                if 0 <= idx < len(self._session_conversations):
                    return (self._session_conversations[idx].get("created_at"),)
            except (ValueError, IndexError):
                pass
            return None

        row_id = coerce_persistent_message_id(message_id)
        if row_id is None:
            return None

        # Filter out soft-deleted anchors so detail-view requests for
        # trashed sessions return 404 from the higher layer instead of
        # silently loading their content.
        return await self._storage.db.fetchone(
            "SELECT created_at FROM conversation_history "
            "WHERE id = ? AND agent_id = ? AND deleted_at IS NULL",
            (row_id, agent_id)
        )

    async def query_conversation_messages(
        self, agent_id: str, start_time: Any, limit: int = 100
    ) -> List[Tuple]:
        """
        Get conversation messages starting from a given time, respecting privacy.

        Returns rows as tuples: (id, role, content, metadata, created_at)
        """
        if self._privacy_config.is_ephemeral():
            return []

        if self._policy.use_session_storage:
            rows = []
            for i, conv in enumerate(self._session_conversations):
                rows.append((
                    i,
                    conv.get("role", ""),
                    conv.get("content", ""),
                    json.dumps(conv.get("metadata", {})) if conv.get("metadata") else None,
                    conv.get("created_at", None),
                ))
            return rows[:limit]

        return await self._storage.db.fetchall("""
            SELECT id, role, content, metadata, created_at
            FROM conversation_history
            WHERE agent_id = ? AND created_at >= ? AND deleted_at IS NULL
            ORDER BY created_at ASC
            LIMIT ?
        """, (agent_id, start_time, limit))

    async def query_last_conversation_row(
        self, agent_id: str
    ) -> Optional[Tuple]:
        """
        Get the most recent conversation row for an agent, respecting privacy.

        Returns a tuple (id, created_at) or None.
        """
        if self._privacy_config.is_ephemeral():
            return None

        if self._policy.use_session_storage:
            if self._session_conversations:
                idx = len(self._session_conversations) - 1
                return (idx, self._session_conversations[idx].get("created_at"))
            return None

        return await self._storage.db.fetchone("""
            SELECT id, created_at FROM conversation_history
            WHERE agent_id = ? AND deleted_at IS NULL
            ORDER BY id DESC LIMIT 1
        """, (agent_id,))

    async def delete_conversation_message(
        self, message_id: int, agent_id: str
    ) -> bool:
        """
        Soft-delete a conversation message by ID, respecting privacy mode (#763).

        In EPHEMERAL mode, raises PrivacyViolationError (nothing to delete).
        In ISOLATED mode, removes from in-memory session storage (which has
        no soft/hard distinction — the row never persisted).
        Otherwise stamps ``deleted_at`` on the persistent row so it can be
        restored from Trash. The matching memory_pin is hard-deleted to
        preserve the sovereign invariant that pins cannot block, delay, or
        resurrect erased content (#750).

        Returns True if a row was soft-deleted, False if not found or
        already in trash.
        """
        if self._privacy_config.is_ephemeral():
            raise PrivacyViolationError(
                "Cannot delete conversations in ephemeral mode (no persistent data)."
            )

        if self._policy.use_session_storage:
            try:
                idx = int(message_id)
                if 0 <= idx < len(self._session_conversations):
                    self._session_conversations.pop(idx)
                    return True
            except (ValueError, IndexError):
                pass
            return False

        row_id = coerce_persistent_message_id(message_id)
        if row_id is None:
            return False

        await self._check_write_permission("delete_conversation_message")
        deleted = await self._storage.delete_message(row_id)

        # Sovereign override: pins cannot point into Trash. Hard-delete
        # the matching pin so the user can't navigate from a pin into a
        # soft-deleted message. If the user later restores the message,
        # they can re-pin it explicitly.
        if deleted:
            await self._delete_pin_for_message(row_id, agent_id)

        return deleted

    async def _delete_pin_for_message(
        self, row_id: int, agent_id: str
    ) -> None:
        """Best-effort drop of any pin pointing at this message id.

        Tolerates a missing ``memory_pins`` table (see
        ``_delete_orphaned_pins`` for rationale).
        """
        try:
            await self._storage.db.execute_commit(
                "DELETE FROM memory_pins WHERE message_id = ? AND agent_id = ?",
                (row_id, agent_id)
            )
        except Exception as e:
            logger.debug(
                "Pin cleanup skipped (memory_pins likely absent): %s", e
            )

    async def restore_conversation_message(
        self, message_id: int, agent_id: str
    ) -> bool:
        """Clear deleted_at on a soft-deleted message (#763 / #765).

        EPHEMERAL has nothing to restore (raises). ISOLATED has no
        persistent state, so restore is a no-op (returns False).
        Otherwise delegates to the conversation store.
        """
        if self._privacy_config.is_ephemeral():
            raise PrivacyViolationError(
                "Cannot restore conversations in ephemeral mode (no persistent data)."
            )
        if self._policy.use_session_storage:
            return False

        row_id = coerce_persistent_message_id(message_id)
        if row_id is None:
            return False

        await self._check_write_permission("restore_conversation_message")
        return await self._storage.restore_message(row_id)

    async def restore_conversation_session(
        self, session_id: str, agent_id: str
    ) -> int:
        """Clear deleted_at on every soft-deleted message in a session.

        EPHEMERAL raises (no persistent data). ISOLATED returns 0 (the
        in-memory list has no Trash distinction). Otherwise delegates to
        the conversation store.
        """
        if self._privacy_config.is_ephemeral():
            raise PrivacyViolationError(
                "Cannot restore conversations in ephemeral mode (no persistent data)."
            )
        if self._policy.use_session_storage:
            return 0

        await self._check_write_permission("restore_conversation_session")
        return await self._storage.restore_conversation_session(session_id)

    async def purge_conversation_message(
        self, message_id: int, agent_id: str, reason: str = "user-initiated"
    ) -> bool:
        """Hard-delete a single message (#763).

        Permanent — bypasses Trash. EPHEMERAL raises (nothing to purge).
        ISOLATED falls through to the soft-delete path because the row
        never persisted in the first place.
        """
        if self._privacy_config.is_ephemeral():
            raise PrivacyViolationError(
                "Cannot purge conversations in ephemeral mode (no persistent data)."
            )

        if self._policy.use_session_storage:
            return await self.delete_conversation_message(message_id, agent_id)

        row_id = coerce_persistent_message_id(message_id)
        if row_id is None:
            return False

        await self._check_write_permission("purge_conversation_message")
        purged = await self._storage.purge_message(row_id, reason=reason)

        if purged:
            await self._delete_pin_for_message(row_id, agent_id)

        return purged

    async def purge_conversation_session(
        self, session_id: str, agent_id: str, reason: str = "user-initiated"
    ) -> int:
        """Hard-delete every message in a session (#763).

        Wipes both live and soft-deleted rows. EPHEMERAL raises.
        ISOLATED falls through to the soft-delete equivalent.
        """
        if self._privacy_config.is_ephemeral():
            raise PrivacyViolationError(
                "Cannot purge conversations in ephemeral mode (no persistent data)."
            )

        if self._policy.use_session_storage:
            return await self.delete_conversation_session(session_id, agent_id)

        await self._check_write_permission("purge_conversation_session")
        purged = await self._storage.purge_conversation_session(
            session_id, reason=reason
        )

        if purged:
            await self._delete_orphaned_pins(agent_id)

        return purged

    async def purge_trash_older_than(
        self,
        cutoff_iso: str,
        *,
        max_rows: int = 10_000,
        reason: str = "retention-janitor",
    ) -> int:
        """Retention-janitor primitive — wrapper delegator (#764).

        The privacy wrapper has to expose this method because the cron
        handler reads ``agent.storage.purge_trash_older_than`` and
        ``agent.storage`` is the wrapper, not the raw facade. Smoke
        testing caught the omission — the task skipped silently with
        "storage facade missing purge_trash_older_than" on every tick.

        No privacy gating needed: the rail only purges rows that were
        already soft-deleted (``deleted_at IS NOT NULL``). Live data is
        never touched. Even in EPHEMERAL mode, where the wrapper
        rejects new persistent writes, aging out already-trashed rows
        from a prior NORMAL stint is the right thing to do.
        """
        return await self._storage.purge_trash_older_than(
            cutoff_iso, max_rows=max_rows, reason=reason,
        )

    async def delete_conversation_session(
        self, session_id: str, agent_id: str
    ) -> int:
        """
        Delete an entire conversation session by ID, respecting privacy mode.

        In EPHEMERAL mode, raises PrivacyViolationError (no persistent data).
        In ISOLATED mode, filters the in-memory session conversations.
        Otherwise delegates to the underlying storage which removes every
        message belonging to the session (metadata-based OR time-gap-based
        resolution — see AsyncConversationStore.delete_conversation_session).

        Returns the number of messages removed (0 when the session didn't
        exist or was already empty).
        """
        if self._privacy_config.is_ephemeral():
            raise PrivacyViolationError(
                "Cannot delete conversations in ephemeral mode (no persistent data)."
            )

        if self._policy.use_session_storage:
            # In ISOLATED mode conversations live in an in-memory list
            # without durable session grouping.  The practical match for
            # "delete this session" is "clear the in-memory backlog."
            removed = len(self._session_conversations)
            self._session_conversations.clear()
            return removed

        await self._check_write_permission("delete_conversation_session")
        count = await self._storage.delete_conversation_session(session_id)

        # Sovereign override: clean up any memory pins that pointed at
        # messages we just soft-deleted. Subquery filters on
        # ``deleted_at IS NULL`` so pins on rows that just moved into
        # Trash are caught here — without that filter the subquery
        # would still find the trashed rows and the NOT IN would skip
        # them, leaving dangling pins (#763 regression).
        if count > 0:
            await self._delete_orphaned_pins(agent_id)

        return count

    async def _delete_orphaned_pins(self, agent_id: str) -> None:
        """Drop pins whose message is no longer live (deleted or purged).

        Tolerates a missing ``memory_pins`` table — the table is created
        by the memory_agency feature, which may not be loaded in slim
        startup paths or constrained tests. Production runs always have
        it; the guard exists so pin cleanup never blocks a legitimate
        delete in those edge cases.
        """
        try:
            await self._storage.db.execute_commit(
                "DELETE FROM memory_pins "
                "WHERE agent_id = ? AND message_id NOT IN "
                "(SELECT id FROM conversation_history "
                " WHERE agent_id = ? AND deleted_at IS NULL)",
                (agent_id, agent_id),
            )
        except Exception as e:
            logger.debug(
                "Pin cleanup skipped (memory_pins likely absent): %s", e
            )

    async def list_trashed_conversations(
        self, limit: int = 200
    ) -> List[Dict[str, Any]]:
        """List soft-deleted messages for the Trash UI (#763 / #765).

        Returns rows where ``deleted_at IS NOT NULL`` for this agent,
        sorted most-recently-trashed first. EPHEMERAL and ISOLATED modes
        return an empty list — neither has a persistent Trash store.
        """
        if self._privacy_config.is_ephemeral():
            return []
        if self._policy.use_session_storage:
            return []

        history = await self._storage.conversation.get_full_history_with_ids(
            include_excluded=True,
            include_stashed=True,
            only_deleted=True,
        )
        history.sort(key=lambda m: m.get("deleted_at") or "", reverse=True)
        return history[:limit]

    async def set_conversation_name(
        self, session_id: str, name: Optional[str]
    ) -> Optional[str]:
        """Upsert a user-chosen display name for a session (issue #716).

        EPHEMERAL raises (no durable data); ISOLATED has no persistent
        store so the wrapper echoes the normalized value without writing;
        NORMAL delegates to the conversation store.
        """
        if self._privacy_config.is_ephemeral():
            raise PrivacyViolationError(
                "Cannot rename conversations in ephemeral mode (no persistent data)."
            )
        if self._policy.use_session_storage:
            if name is None:
                return None
            trimmed = name.strip()
            return trimmed or None

        await self._check_write_permission("set_conversation_name")
        return await self._storage.set_conversation_name(session_id, name)

    async def get_conversation_name(self, session_id: str) -> Optional[str]:
        """Read the user-assigned display name for a session."""
        if self._privacy_config.is_ephemeral():
            return None
        if self._policy.use_session_storage:
            return None
        return await self._storage.get_conversation_name(session_id)

    async def get_conversation_names(self) -> Dict[str, str]:
        """Bulk read of user-assigned conversation names for this agent."""
        if self._privacy_config.is_ephemeral():
            return {}
        if self._policy.use_session_storage:
            return {}
        return await self._storage.get_conversation_names()

    # === Pass-through properties (with deprecation warnings) ===
    #
    # These properties expose the underlying storage objects directly,
    # which bypasses privacy mode enforcement. They are deprecated and
    # callers should migrate to the privacy-aware methods above.
    # They remain for backward compatibility with internal agent code.

    def _warn_direct_access(self, property_name: str) -> None:
        """Log a deprecation warning when a raw storage property is accessed."""
        warnings.warn(
            f"Direct access to PrivacyEnforcingStorage.{property_name} bypasses "
            f"privacy enforcement. Use privacy-aware methods instead "
            f"(e.g., query_conversations, get_conversation_history).",
            DeprecationWarning,
            stacklevel=3,
        )
        logger.warning(
            f"Privacy bypass: direct access to .{property_name} property "
            f"(current mode: {self._privacy_mode.value})"
        )

    @property
    def db(self):
        """Access to underlying database. DEPRECATED: bypasses privacy enforcement."""
        self._warn_direct_access("db")
        return self._storage.db

    @property
    def db_path(self) -> str:
        """Get the database file path from underlying storage."""
        return self._storage.db_path

    @property
    def graph_store(self):
        """Access to graph store."""
        return self._storage.graph

    @property
    def graph(self):
        """Access to graph store (alias for graph_store)."""
        return self._storage.graph

    @property
    def conversation(self):
        """Access to conversation store. DEPRECATED: bypasses privacy enforcement."""
        self._warn_direct_access("conversation")
        return self._storage.conversation

    @property
    def files(self):
        """Access to file store. DEPRECATED: bypasses privacy enforcement."""
        self._warn_direct_access("files")
        return self._storage.files

    @property
    def rag(self):
        """Access to RAG store."""
        return self._storage.rag
    
    async def close(self):
        """Close the underlying storage."""
        await self._storage.close()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()



def wrap_storage_with_privacy(storage, privacy_mode: Union[PrivacyMode, PrivacyConfig, str]) -> PrivacyEnforcingStorage:
    """
    Factory function to wrap a Storage instance with privacy enforcement.
    
    Args:
        storage: The Storage instance to wrap
        privacy_mode: Initial privacy mode (PrivacyMode, PrivacyConfig, or preset name)
        
    Returns:
        PrivacyEnforcingStorage wrapper
    """
    return PrivacyEnforcingStorage(storage, privacy_mode)
