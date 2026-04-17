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

from kestrel_sovereign.privacy import PrivacyMode, PrivacyConfig, get_privacy_preset
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
            config = mode.to_config()
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
        logger.info(f"PrivacyEnforcingStorage initialized with config: storage={self._privacy_config.storage}, llm={self._privacy_config.llm_location}")

    @property
    def agent_id(self) -> str:
        """Get the agent_id from underlying storage for multi-tenant isolation."""
        return getattr(self._storage, 'agent_id', '')
    
    def _to_config(self, mode: Union[PrivacyMode, PrivacyConfig, str]) -> PrivacyConfig:
        """Convert various mode representations to PrivacyConfig."""
        if isinstance(mode, PrivacyConfig):
            return mode
        elif isinstance(mode, PrivacyMode):
            return mode.to_config()
        elif isinstance(mode, str):
            return get_privacy_preset(mode)
        else:
            raise TypeError(f"Expected PrivacyMode, PrivacyConfig, or str, got {type(mode)}")
    
    @property
    def privacy_mode(self) -> PrivacyMode:
        """Backward compatibility: return PrivacyMode enum."""
        return PrivacyMode.from_config(self._privacy_config)
    
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
        """
        old_config = self._privacy_config
        self._privacy_config = self._to_config(mode)
        self._policy = PrivacyPolicy.from_config(self._privacy_config)
        logger.info(f"Privacy config changed: storage={old_config.storage}->{self._privacy_config.storage}, llm={old_config.llm_location}->{self._privacy_config.llm_location}")
    
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
    
    async def add_conversation(self, role: str, content: str, metadata: Optional[Dict] = None, session_id: Optional[str] = None) -> None:
        """
        Add a conversation entry, respecting privacy mode.
        
        - EPHEMERAL: Raises PrivacyViolationError (use in-memory buffer instead)
        - ISOLATED: Stores in session-local list
        - ANONYMOUS: Anonymizes content before storing
        - NORMAL/PUBLIC: Stores as-is
        """
        if self._privacy_config.is_ephemeral():
            raise PrivacyViolationError(
                "Cannot store conversations in ephemeral mode. "
                "Use EphemeralSession for in-memory buffering."
            )
        
        processed_content = self._anonymize_if_required(content)
        
        if metadata is None:
            metadata = {}
        metadata["privacy_mode"] = self._privacy_mode.value
        
        if self._policy.use_session_storage:
            # Store in session-local list (ISOLATED mode)
            self._session_conversations.append({
                "role": role,
                "content": processed_content,
                "metadata": metadata,
                "session_id": session_id
            })
            logger.debug(f"Conversation stored in session ({len(self._session_conversations)} total)")
        else:
            # Store in persistent storage
            await self._storage.add_conversation(role, processed_content, metadata, session_id)
    
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
            WHERE agent_id = ?
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

        return await self._storage.db.fetchone(
            "SELECT created_at FROM conversation_history WHERE id = ? AND agent_id = ?",
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
            WHERE agent_id = ? AND created_at >= ?
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
            WHERE agent_id = ?
            ORDER BY id DESC LIMIT 1
        """, (agent_id,))

    async def delete_conversation_message(
        self, message_id: int, agent_id: str
    ) -> bool:
        """
        Delete a conversation message by ID, respecting privacy mode.

        In EPHEMERAL mode, raises PrivacyViolationError (nothing to delete).
        In ISOLATED mode, removes from session storage.
        Otherwise deletes from persistent database.

        Returns True if a message was deleted, False if not found.
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
        result = await self._storage.db.execute_commit(
            "DELETE FROM conversation_history WHERE id = ? AND agent_id = ?",
            (row_id, agent_id)
        )
        deleted = result.rowcount > 0 if hasattr(result, 'rowcount') else True

        # Sovereign override: clean up any pins on this message.
        # Pins CANNOT block, delay, or resurrect erased content.
        if deleted:
            await self._storage.db.execute_commit(
                "DELETE FROM memory_pins WHERE message_id = ? AND agent_id = ?",
                (row_id, agent_id)
            )

        return deleted

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
