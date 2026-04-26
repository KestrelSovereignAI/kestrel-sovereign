"""
Async Storage - Unified async storage interface for Kestrel.

This module provides a fully async storage facade that composes
all storage components (files, graph, conversation, RAG).

Supports both SQLite (local) and PostgreSQL (cloud) backends via
environment variable KESTREL_DB_BACKEND.
"""
import io
import os
import logging
import tarfile
import tempfile
import shutil
from datetime import datetime, UTC
from typing import Dict, Optional, List, Any, Union

from .async_database import AsyncDatabase
from .async_file_store import AsyncFileStore
from .async_conversation_store import AsyncConversationStore
from .async_graph_store import AsyncGraphStore, GraphNode, Edge
from .async_rag_store import AsyncRAGStore
from .db import DatabaseBackend, SQLiteBackend, create_backend

logger = logging.getLogger(__name__)


def get_default_agent_data_dir() -> str:
    """Get the default agent data directory."""
    return os.environ.get("KESTREL_DB_PATH", os.path.join(os.getcwd(), "agent_data"))


class AsyncStorage:
    """
    Unified async storage interface for Kestrel.

    Provides async access to all storage components:
    - File storage (content-addressable with optional encryption)
    - Conversation history
    - Knowledge graph (nodes and edges)
    - RAG document chunks

    Supports both SQLite and PostgreSQL backends:
    - SQLite (default): Pass db_path string or set KESTREL_DB_PATH
    - PostgreSQL: Pass backend="postgres" or set KESTREL_DB_BACKEND=postgres

    Usage:
        # SQLite (default)
        async with AsyncStorage("/path/to/db") as storage:
            hash = await storage.store_file(b"content", "file.txt")
            content = await storage.retrieve_file(hash)

        # PostgreSQL
        async with AsyncStorage(backend="postgres", dsn="postgresql://...") as storage:
            hash = await storage.store_file(b"content", "file.txt")
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        *,
        backend: Optional[Union[str, DatabaseBackend]] = None,
        dsn: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        agent_id: str = "",
    ):
        """
        Initialize AsyncStorage.

        Args:
            db_path: Path to SQLite database file (for SQLite backend)
            backend: Either 'sqlite', 'postgres', or a DatabaseBackend instance
            dsn: PostgreSQL connection string (for postgres backend)
            config: Full configuration dict (overrides other args)
            agent_id: Agent/companion ID for multi-tenant isolation
        """
        self._backend: Optional[DatabaseBackend] = None
        self.db_path: Optional[str] = None
        self.agent_id = agent_id

        # If backend is already a DatabaseBackend instance, use it directly
        if isinstance(backend, DatabaseBackend):
            self._backend = backend
            if hasattr(backend, 'db_path'):
                self.db_path = backend.db_path
        elif config is not None:
            # Use config dict
            self._backend = create_backend(config)
            self.db_path = config.get('db_path')
        elif backend == "postgres" or os.getenv("KESTREL_DB_BACKEND", "").lower() == "postgres":
            # PostgreSQL mode
            pg_dsn = dsn or os.getenv("KESTREL_DATABASE_URL")
            if pg_dsn:
                self._backend = create_backend({"backend": "postgres", "dsn": pg_dsn})
            else:
                # Fall back to individual env vars
                self._backend = create_backend({"backend": "postgres"})
        else:
            # Default: SQLite mode
            if db_path is None:
                agent_data_dir = get_default_agent_data_dir()
                db_path = os.path.join(agent_data_dir, "kestrel_prime.db")
                os.makedirs(agent_data_dir, exist_ok=True)

            self.db_path = db_path
            self._backend = SQLiteBackend(db_path)

        self.db: Optional[AsyncDatabase] = None
        self.files: Optional[AsyncFileStore] = None
        self.conversation: Optional[AsyncConversationStore] = None
        self.graph: Optional[AsyncGraphStore] = None
        self.rag: Optional[AsyncRAGStore] = None
        self._initialized = False

    @classmethod
    def from_backend(cls, backend: DatabaseBackend) -> "AsyncStorage":
        """Create AsyncStorage from an existing DatabaseBackend."""
        return cls(backend=backend)

    @classmethod
    async def create_sqlite(cls, db_path: str) -> "AsyncStorage":
        """Factory method to create and initialize SQLite-backed storage."""
        storage = cls(db_path=db_path)
        await storage.initialize()
        return storage

    @classmethod
    async def create_postgres(cls, dsn: str) -> "AsyncStorage":
        """Factory method to create and initialize PostgreSQL-backed storage."""
        storage = cls(backend="postgres", dsn=dsn)
        await storage.initialize()
        return storage

    @property
    def backend_type(self) -> str:
        """Get the backend type: 'sqlite' or 'postgres'."""
        return self._backend.backend_type if self._backend else "unknown"

    def _now_sql(self) -> str:
        """Get SQL expression for current timestamp based on backend type."""
        if self.backend_type == "postgres":
            return "NOW()"
        return "datetime('now')"

    async def initialize(self) -> None:
        """Initialize the storage (connect to database)."""
        if not self._initialized:
            await self._backend.connect()
            self.db = AsyncDatabase(self._backend)
            await self.db._init_schema()
            self.db._initialized = True
            self.files = AsyncFileStore(self.db)
            self.conversation = AsyncConversationStore(self.db, agent_id=self.agent_id)
            self.graph = AsyncGraphStore(self.db)
            self.rag = AsyncRAGStore(self.db)
            self._initialized = True
            logger.info(f"AsyncStorage initialized ({self.backend_type}): {self.db_path or 'PostgreSQL'}")
    
    async def close(self) -> None:
        """Close the storage connection."""
        if self.db:
            await self.db.close()
        self._initialized = False
    
    async def __aenter__(self):
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    # --- File Operations ---
    
    async def store_file(self, content: bytes, original_name: str, 
                         metadata: Optional[Dict] = None) -> str:
        """Store a file and return its content hash."""
        if not self._initialized:
            await self.initialize()
        return await self.files.store_file(content, original_name, metadata)
    
    async def retrieve_file(self, content_hash: str) -> Optional[bytes]:
        """Retrieve a file by its content hash."""
        if not self._initialized:
            await self.initialize()
        return await self.files.retrieve_file(content_hash)
    
    async def get_file_metadata(self, content_hash: str) -> Optional[Dict[str, Any]]:
        """Get file metadata."""
        if not self._initialized:
            await self.initialize()
        return await self.files.get_file_metadata(content_hash)
    
    # --- Conversation Operations ---
    
    async def add_conversation(self, role: str, content: str,
                               metadata: Optional[Dict] = None,
                               session_id: Optional[str] = None) -> None:
        """Add a conversation message.

        Args:
            role: Message role (user, assistant, system)
            content: Message content
            metadata: Optional metadata dict
            session_id: If provided, link this message to a specific session.
                       This allows resuming old conversations beyond the 30-min gap.
        """
        if not self._initialized:
            await self.initialize()
        await self.conversation.add_conversation(role, content, metadata, session_id)
    
    async def get_conversation_history(
        self, limit: int = 100, session_id: str = None
    ) -> List[Dict[str, Any]]:
        """Get recent conversation history.

        Args:
            limit: Maximum number of messages to return
            session_id: If provided, get messages from this session only
        """
        if not self._initialized:
            await self.initialize()
        return await self.conversation.get_conversation_history(limit, session_id=session_id)
    
    async def search_conversation(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search conversation history."""
        if not self._initialized:
            await self.initialize()
        return await self.conversation.search_history(query, limit)

    # --- Conversation-session management (issues #715 / #716) ---

    async def delete_conversation_session(self, session_id: str) -> int:
        """Soft-delete every message belonging to a conversation session.

        Delegator onto ``AsyncConversationStore.delete_conversation_session``.
        Exists on the facade because ``PrivacyEnforcingStorage`` calls
        through ``storage.<method>`` rather than
        ``storage.conversation.<method>`` — without this wrapper the
        privacy-aware path gets ``AttributeError`` and the endpoint
        returns 500 (observed on live Meridian DELETE /api/conversations/
        {id} calls before this fix).

        Stamps ``deleted_at`` (#763); use ``purge_conversation_session``
        for permanent removal.
        """
        if not self._initialized:
            await self.initialize()
        return await self.conversation.delete_conversation_session(session_id)

    async def delete_message(self, message_id: int) -> bool:
        """Soft-delete a single message — facade delegator (#763)."""
        if not self._initialized:
            await self.initialize()
        return await self.conversation.delete_message(message_id)

    async def restore_message(self, message_id: int) -> bool:
        """Restore a soft-deleted message — facade delegator (#763)."""
        if not self._initialized:
            await self.initialize()
        return await self.conversation.restore_message(message_id)

    async def restore_conversation_session(self, session_id: str) -> int:
        """Restore a soft-deleted session — facade delegator (#763)."""
        if not self._initialized:
            await self.initialize()
        return await self.conversation.restore_conversation_session(session_id)

    async def purge_message(
        self, message_id: int, reason: str = "user-initiated"
    ) -> bool:
        """Hard-delete a single message — facade delegator (#763)."""
        if not self._initialized:
            await self.initialize()
        return await self.conversation.purge_message(message_id, reason=reason)

    async def purge_conversation_session(
        self, session_id: str, reason: str = "user-initiated"
    ) -> int:
        """Hard-delete an entire session — facade delegator (#763)."""
        if not self._initialized:
            await self.initialize()
        return await self.conversation.purge_conversation_session(
            session_id, reason=reason
        )

    async def purge_all_conversations(
        self, reason: str = "administrative"
    ) -> int:
        """Hard-delete every row for this agent — facade delegator (#763)."""
        if not self._initialized:
            await self.initialize()
        return await self.conversation.purge_all(reason=reason)

    async def set_conversation_name(
        self, session_id: str, name: Optional[str]
    ) -> Optional[str]:
        """Upsert / clear a user-assigned display name for a session.

        Delegator onto ``AsyncConversationStore.set_conversation_name``.
        Same rationale as ``delete_conversation_session`` above — the
        privacy wrapper calls ``self._storage.set_conversation_name`` and
        needs the method to exist at the facade layer.
        """
        if not self._initialized:
            await self.initialize()
        return await self.conversation.set_conversation_name(session_id, name)

    async def get_conversation_name(self, session_id: str) -> Optional[str]:
        """Read the user-assigned display name for a session."""
        if not self._initialized:
            await self.initialize()
        return await self.conversation.get_conversation_name(session_id)

    async def get_conversation_names(self) -> Dict[str, str]:
        """Bulk read of user-assigned conversation names for this agent."""
        if not self._initialized:
            await self.initialize()
        return await self.conversation.get_conversation_names()
    
    # --- Graph Operations ---
    
    async def add_node(self, node: GraphNode) -> None:
        """Add a node to the knowledge graph."""
        if not self._initialized:
            await self.initialize()
        await self.graph.add_node(node)
    
    async def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get a node by ID."""
        if not self._initialized:
            await self.initialize()
        return await self.graph.get_node(node_id)
    
    async def get_nodes_by_type(self, node_type: str) -> List[GraphNode]:
        """Get all nodes of a specific type."""
        if not self._initialized:
            await self.initialize()
        return await self.graph.get_nodes_by_type(node_type)
    
    async def add_edge(self, source_id: str, target_id: str, label: str,
                       properties: Optional[Dict] = None) -> None:
        """Add an edge between nodes."""
        if not self._initialized:
            await self.initialize()
        await self.graph.add_edge(source_id, target_id, label, properties)

    async def delete_node(self, node_id: str) -> None:
        """Delete a node and its edges from the knowledge graph."""
        if not self._initialized:
            await self.initialize()
        await self.graph.delete_node(node_id)

    async def get_edges_from(self, node_id: str) -> List[Edge]:
        """Get outgoing edges from a node."""
        if not self._initialized:
            await self.initialize()
        return await self.graph.get_edges(node_id, direction="out")

    async def get_edges_to(self, node_id: str) -> List[Edge]:
        """Get incoming edges to a node."""
        if not self._initialized:
            await self.initialize()
        return await self.graph.get_edges(node_id, direction="in")

    # --- RAG Operations ---
    
    async def chunk_document(self, content_hash: str) -> int:
        """Chunk a stored document for RAG."""
        if not self._initialized:
            await self.initialize()
        content_bytes = await self.retrieve_file(content_hash)
        if content_bytes:
            content_str = content_bytes.decode('utf-8')
            return await self.rag.chunk_document(content_hash, content_str)
        return 0
    
    async def search_chunks(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search document chunks."""
        if not self._initialized:
            await self.initialize()
        return await self.rag.search_chunks(query, limit)
    
    # --- Case Law / Audit Search ---
    
    async def search_case_law(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Orchestrates a semantic search for relevant "case law" (past audit failures).
        """
        if not self._initialized:
            await self.initialize()
        failures = await self.conversation.get_all_audit_failures()
        return await self.rag.search_case_law(query, failures, top_k)
    
    # --- Backup/Restore Operations ---
    
    async def create_backup_blob(self, include_db: bool = True) -> bytes:
        """
        Creates a gzipped tar archive of selected artifacts and returns its bytes.
        Currently includes only the SQLite DB when include_db is True.

        Note: This temporarily closes the database connection to ensure
        a consistent backup, then reopens it.

        IMPORTANT: SQLite WAL mode stores data in a separate -wal file until checkpoint.
        We must checkpoint before copying the main DB file to ensure all data is included.
        """
        if self.db_path == ":memory:":
            raise ValueError("Cannot backup an in-memory database")

        # Checkpoint WAL before closing - this ensures all data is written to main DB file
        # Without this, the backup would be missing any uncommitted WAL data.
        # Use fetchone so aiosqlite consumes the cursor fully — execute() then
        # auto-commit would raise "cannot commit transaction - SQL statements
        # in progress" because the PRAGMA's result cursor is still open.
        was_initialized = self._initialized
        if was_initialized and self.db:
            await self.db.fetchone("PRAGMA wal_checkpoint(TRUNCATE)")
            await self.close()
        
        try:
            buffer = io.BytesIO()
            with tarfile.open(fileobj=buffer, mode='w:gz') as tar:
                if include_db:
                    tar.add(self.db_path, arcname='kestrel.db')
            buffer.seek(0)
            return buffer.read()
        finally:
            # Reopen if it was open
            if was_initialized:
                await self.initialize()
    
    async def restore_from_backup_blob(self, backup_blob: bytes) -> Dict[str, Any]:
        """
        Restores from a backup blob created by create_backup_blob.
        Extracts the database and conversation history.
        
        Args:
            backup_blob: The gzipped tar archive bytes
            
        Returns:
            Dict with restoration statistics
        """
        if not self._initialized:
            await self.initialize()
            
        stats = {"messages_restored": 0}
        
        try:
            # Try to open as gzipped tar first
            buffer = io.BytesIO(backup_blob)
            with tarfile.open(fileobj=buffer, mode='r:gz') as tar:
                await self._restore_from_tar(tar, stats)
        except tarfile.ReadError as e:
            logger.warning(f"Backup blob is not a valid gzipped tar file: {e}")
            # Try uncompressed tar
            try:
                buffer = io.BytesIO(backup_blob)
                with tarfile.open(fileobj=buffer, mode='r') as tar:
                    await self._restore_from_tar(tar, stats)
            except tarfile.ReadError:
                logger.warning("Backup blob is not a tar file at all, skipping restoration")
        
        return stats
    
    async def _restore_from_tar(self, tar: tarfile.TarFile, stats: Dict[str, Any]) -> None:
        """Helper to restore from an opened tar archive."""
        temp_dir = tempfile.mkdtemp()
        try:
            tar.extractall(temp_dir, filter='data')
            
            backup_db_path = os.path.join(temp_dir, 'kestrel.db')
            if os.path.exists(backup_db_path):
                # Use async database to read from backup
                import aiosqlite
                async with aiosqlite.connect(backup_db_path) as backup_conn:
                    cursor = await backup_conn.execute(
                        "SELECT role, content, metadata FROM conversation_history"
                    )
                    conversations = await cursor.fetchall()

                    for role, content, metadata_json in conversations:
                        # Insert directly into current database with agent_id
                        await self.db.execute_commit(
                            f"INSERT INTO conversation_history (agent_id, role, content, metadata, created_at) VALUES (?, ?, ?, ?, {self._now_sql()})",
                            (self.agent_id, role, content, metadata_json)
                        )
                        stats["messages_restored"] += 1
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    async def record_backup_artifact(self, agent_id: str, result: Any) -> str:
        """
        Records a backup artifact in the graph and links it to the agent.
        Expects a result compatible with FilecoinAdapter.StorageResult.
        Returns the backup node_id.
        """
        if not self._initialized:
            await self.initialize()
            
        properties = {
            "storage_tier": getattr(result, 'storage_tier', None).value if getattr(result, 'storage_tier', None) else None,
            "ipfs_cid": getattr(result, 'ipfs_cid', None),
            "filecoin_deal_id": getattr(result, 'filecoin_deal_id', None),
            "encrypted": getattr(result, 'encrypted', False),
            "encryption_key_hash": getattr(result, 'encryption_key_hash', None),
            "created_at": datetime.now(UTC).isoformat(),
        }

        backup_node = GraphNode(
            node_id=getattr(result, 'content_hash'),
            node_type="backup_artifact",
            label="Backup Artifact",
            properties=properties
        )
        await self.add_node(backup_node)
        await self.add_edge(agent_id, backup_node.node_id, "backup")
        return backup_node.node_id
