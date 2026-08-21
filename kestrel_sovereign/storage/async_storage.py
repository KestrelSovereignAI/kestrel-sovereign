"""
Async Storage - Unified async storage interface for Kestrel.

This module provides a fully async storage facade that composes
all storage components (files, graph, conversation, RAG).

Supports both SQLite (local) and PostgreSQL (cloud) backends via
environment variable KESTREL_DB_BACKEND.
"""
import asyncio
import io
import json
import os
import logging
import tarfile
import tempfile
import shutil
import hashlib
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, UTC
from typing import Dict, Optional, List, Any, Union

from .async_database import AsyncDatabase
from .async_file_store import AsyncFileStore
from .async_conversation_store import AsyncConversationStore, _rows_affected
from .destructive_audit import DestructiveAuditLog, audit_db_path_for
from .async_graph_store import AsyncGraphStore, GraphNode, Edge, NodeSwapResult
from .async_assertion_store import (
    AsyncAssertionStore,
    _AssertionTenantCapability,
    _create_agent_bound_assertion_store,
)
from .async_rag_store import AsyncRAGStore
from .agent_resource_store import (
    AgentResourceStore,
    AgentResourceVersion,
    SOUL_MARKDOWN_RESOURCE_TYPE,
)
from .semantic_binding import SemanticAssertionBinding
from .conversation_created_at import created_at_bind, fill_undatable
from .session_id_column import column_session_id
from kestrel_sovereign.knowledge import Visibility
from .db import ConnectionError, DatabaseBackend, SQLiteBackend, create_backend

logger = logging.getLogger(__name__)

# Bind-parameter ceiling for a single ``id IN (...)`` DELETE. SQLite's default
# SQLITE_MAX_VARIABLE_NUMBER is 999 on older builds; stay well under it (and it
# is safe for Postgres' far larger limit) so a high-volume EPHEMERAL purge never
# raises on the placeholder count and silently leaves leaked rows behind.
_DELETE_ID_BATCH = 500

# Timestamp formats written by SQLite ``datetime('now')`` / ``CURRENT_TIMESTAMP``
# (space separator). ISO-8601 (``T`` separator, optional offset/microseconds) is
# handled by ``datetime.fromisoformat`` directly.
_SQLITE_TS_FORMATS = ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S")


def _parse_utc_datetime(value) -> Optional[datetime]:
    """Parse a stored timestamp (SQLite space-form or ISO-8601) to aware UTC.

    Returns ``None`` when the value is empty or unparseable so callers can make
    an explicit fail-safe decision rather than silently mis-sorting a raw string.
    """
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        s = str(value).strip()
        if not s:
            return None
        dt = None
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError:
            for fmt in _SQLITE_TS_FORMATS:
                try:
                    dt = datetime.strptime(s, fmt)
                    break
                except ValueError:
                    continue
        if dt is None:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


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
        llm_service: Optional[Any] = None,
        _assertion_tenant_capability: Optional[_AssertionTenantCapability] = None,
        semantic_capabilities=None,
        _artifact_clock=None,
        cold_read: bool = False,
    ):
        """
        Initialize AsyncStorage.

        Args:
            db_path: Path to SQLite database file (for SQLite backend)
            backend: Either 'sqlite', 'postgres', or a DatabaseBackend instance
            dsn: PostgreSQL connection string (for postgres backend)
            config: Full configuration dict (overrides other args)
            agent_id: Agent/companion ID for multi-tenant isolation
            llm_service: Agent-scoped LLM service used for provider-backed embeddings
        """
        self._backend: Optional[DatabaseBackend] = None
        self.db_path: Optional[str] = None
        self.agent_id = agent_id
        self.llm_service = llm_service
        # Semantic-vector provider authority is captured at host construction;
        # mutating the public chat-service attribute later cannot substitute a
        # feature-supplied callable or relabel a remote route as local.
        self.__semantic_vector_llm_service = llm_service
        if _artifact_clock is not None and not callable(_artifact_clock):
            raise TypeError("_artifact_clock must be callable")
        self._artifact_clock = _artifact_clock or (lambda: datetime.now(UTC))
        # A semantic runtime is agent-owned, not a caller-selected option on
        # individual maintenance/corpus requests.  Keep its RDF/SPARQL codec
        # alive with storage so the pins used by live sleep work are the pins
        # that were accepted during boot.
        from kestrel_sovereign.knowledge.capabilities import SemanticRuntimeCapabilities

        if semantic_capabilities is not None and not isinstance(
            semantic_capabilities, SemanticRuntimeCapabilities
        ):
            raise TypeError("semantic_capabilities must be SemanticRuntimeCapabilities")
        self.semantic_capabilities = (
            semantic_capabilities or SemanticRuntimeCapabilities.stable()
        )
        try:
            self.semantic_capabilities.validate()
            self._semantic_rdf_codec = self.semantic_capabilities.create_rdf_codec()
        except ValueError as exc:
            # This is the line an operator actually reads when agent boot
            # fails; leaving the cause in the traceback alone is what made a
            # CRLF-smudged checkout look like an unexplained total failure.
            raise ValueError(
                f"semantic runtime capability is unavailable: {exc}"
            ) from exc
        # If backend is already a DatabaseBackend instance, use it directly
        if isinstance(backend, DatabaseBackend):
            self._backend = backend
            if hasattr(backend, 'db_path'):
                self.db_path = backend.db_path
        elif config is not None:
            # Use config dict
            self._backend = create_backend(config)
            self.db_path = config.get('db_path')
        elif backend == "postgres" or (
            backend is None
            and os.getenv("KESTREL_DB_BACKEND", "").lower() == "postgres"
        ):
            # PostgreSQL mode
            pg_dsn = dsn or os.getenv("KESTREL_DATABASE_URL")
            if pg_dsn:
                self._backend = create_backend(
                    {"backend": "postgres", "dsn": pg_dsn, "cold_read": cold_read}
                )
            else:
                # Fall back to individual env vars
                self._backend = create_backend(
                    {"backend": "postgres", "cold_read": cold_read}
                )
        else:
            # Default or explicit SQLite mode.  An explicit backend is an
            # ownership boundary: multi-agent hosts use local SQLite identity
            # stores while their runtime data lives in PostgreSQL, so the
            # environment default must not redirect those reads.
            if db_path is None:
                agent_data_dir = get_default_agent_data_dir()
                db_path = os.path.join(agent_data_dir, "kestrel_prime.db")
                # A cold read must be able to answer "there is nothing here"
                # without bringing the directory into existence to say so. The
                # backend already refuses to mkdir; the facade was creating it
                # first and making that refusal moot.
                if not cold_read:
                    os.makedirs(agent_data_dir, exist_ok=True)

            self.db_path = db_path
            # SQLite serialises writers at the FILE level, so an inspection
            # that opens read-write can contend with a running agent even
            # though it never writes (#2920). A cold read is refused by SQLite
            # itself, not merely by convention.
            self._backend = SQLiteBackend(db_path, cold_read=cold_read)

        if _assertion_tenant_capability is not None:
            if type(_assertion_tenant_capability) is not _AssertionTenantCapability:
                raise TypeError(
                    "assertion tenant capability must be issued by the storage tenant resolver"
                )
            if _assertion_tenant_capability.tenant_id != agent_id:
                raise ValueError("assertion tenant capability does not match AsyncStorage.agent_id")
            self._assertion_tenant_capability = _assertion_tenant_capability
        else:
            self._assertion_tenant_capability = None

        # The mode the caller actually asked for, from whichever channel they
        # used. The ``config=`` path carries ``cold_read`` in the dict while
        # this keyword keeps its default.
        requested_cold_read = bool(
            config.get("cold_read", cold_read) if config is not None else cold_read
        )
        # Prefer what the backend reports, so the facade and the connection
        # cannot disagree — a facade that thinks it is writable runs
        # migrations against an immutable connection. Backends that do not
        # carry the flag at all (PostgreSQL, where a second connection is
        # ordinary and there is no file lock to contend for) fall back to the
        # request, so an inspection is never silently upgraded to a writer.
        self.cold_read = bool(
            getattr(self._backend, "cold_read", requested_cold_read)
        )
        self.db: Optional[AsyncDatabase] = None
        self.files: Optional[AsyncFileStore] = None
        self.conversation: Optional[AsyncConversationStore] = None
        self.destructive_audit: Optional[DestructiveAuditLog] = None
        self.graph: Optional[AsyncGraphStore] = None
        # Canonical semantic facts are intentionally separate from the property
        # graph.  ``graph`` remains an application-resource/projection seam;
        # ``assertions`` owns assertion lifecycle and provenance.
        # Assertion scope is capability-bearing.  Keep it private so exposing
        # this facade's database cannot be combined with a public constructor
        # to forge another tenant's semantic authority.
        self._assertions: Optional[AsyncAssertionStore] = None
        self.rag: Optional[AsyncRAGStore] = None
        self.agent_resources: Optional[AgentResourceStore] = None
        self._initialized = False

    @classmethod
    def from_backend(
        cls,
        backend: DatabaseBackend,
        *,
        agent_id: str = "",
        _assertion_tenant_capability: Optional[_AssertionTenantCapability] = None,
    ) -> "AsyncStorage":
        """Create storage from a shared backend without minting tenant authority.

        ``agent_id`` continues to scope the pre-existing file, graph, RAG, and
        conversation stores.  Canonical assertions additionally require the
        opaque capability issued by the host's authenticated tenant resolver;
        a backend handle plus a caller-selected string is deliberately
        insufficient.
        """
        return cls(
            backend=backend,
            agent_id=agent_id,
            _assertion_tenant_capability=_assertion_tenant_capability,
        )

    @classmethod
    async def create_sqlite(cls, db_path: str, *, agent_id: str = "") -> "AsyncStorage":
        """Create SQLite storage without granting semantic tenant authority.

        Assertion authority is issued only by the authenticated agent boot
        resolver and then injected through the private capability seam.
        """
        storage = cls(db_path=db_path, agent_id=agent_id)
        await storage.initialize()
        return storage

    @classmethod
    async def create_postgres(cls, dsn: str, *, agent_id: str = "") -> "AsyncStorage":
        """Create PostgreSQL storage without granting semantic tenant authority."""
        storage = cls(backend="postgres", dsn=dsn, agent_id=agent_id)
        await storage.initialize()
        return storage

    @property
    def backend_type(self) -> str:
        """Get the backend type: 'sqlite' or 'postgres'."""
        return self._backend.backend_type if self._backend else "unknown"

    @property
    def minimum_close_timeout_s(self) -> float:
        """Minimum time an outer guard must reserve for storage close.

        Backends may opt into this narrow shutdown contract when their close
        lifecycle owns resources that outlive a public close acknowledgement
        (SQLite's aiosqlite worker is one example).  Backends without that
        requirement retain the prior zero-extra-budget behavior.
        """
        value = getattr(self._backend, "minimum_close_timeout_s", 0.0)
        return value if isinstance(value, (int, float)) and value > 0 else 0.0

    @property
    def minimum_sqla_factory_close_timeout_s(self) -> float:
        """Reservation for the optional cached SQLAlchemy pre-close phase."""
        if self.db is None:
            return 0.0
        return self.db.minimum_sqla_factory_close_timeout_s

    @property
    def minimum_potential_sqla_factory_close_timeout_s(self) -> float:
        """Reservation if feature shutdown creates a SQLite factory late."""
        if self.db is None:
            return 0.0
        return self.db.minimum_potential_sqla_factory_close_timeout_s

    def _now_sql(self) -> str:
        """Get SQL expression for current timestamp based on backend type."""
        if self.backend_type == "postgres":
            return "NOW()"
        return "datetime('now')"

    async def initialize(self) -> None:
        """Initialize the storage (connect to database)."""
        if not self._initialized:
            previous_db = self.db
            if previous_db is not None:
                if previous_db.connection_retirement_pending is True:
                    raise ConnectionError(
                        "Cannot initialize storage while a previous SQLAlchemy "
                        "engine or database connection is still retiring"
                    )
                # A failed close retains its database-level lifecycle owner.
                # Finalize and detach it before replacing ``self.db`` so the
                # old factory cannot become an unowned stale worker source.
                await previous_db.finalize_retired_sqla_factory()
            await self._backend.connect()
            self.db = AsyncDatabase(self._backend)
            # A cold read inspects a database it must not alter, so it runs
            # neither of the two write-oriented steps a normal open performs:
            # ``_init_schema`` issues migrations and DDL, and the destructive
            # audit log CREATES a second database file beside the one being
            # inspected. On a current schema those would fail against an
            # immutable connection; the point is that they must not be
            # attempted at all. A cold reader reads what is already there, and
            # records nothing — it performs no destructive operation to audit.
            if not self.cold_read:
                await self.db._init_schema()
            self.db._initialized = True
            if (
                not self.cold_read
                and self.backend_type == "sqlite"
                and self.db_path
                and self.db_path != ":memory:"
            ):
                self.destructive_audit = DestructiveAuditLog(audit_db_path_for(self.db_path))
                await self.destructive_audit.initialize()
            self.files = AsyncFileStore(self.db, agent_id=self.agent_id)
            self.conversation = AsyncConversationStore(
                self.db,
                agent_id=self.agent_id,
                llm_service=self.llm_service,
                destructive_audit=self.destructive_audit,
            )
            self.graph = AsyncGraphStore(self.db, agent_id=self.agent_id)
            if self._assertion_tenant_capability is not None:
                self._assertions = _create_agent_bound_assertion_store(
                    self.db,
                    tenant_capability=self._assertion_tenant_capability,
                    artifact_clock=self._artifact_clock,
                )
                self._assertions._bind_semantic_recall_derivative_revoker(  # noqa: SLF001 - storage composition seam
                    self._withdraw_semantic_recall_derivatives
                )
            self.rag = AsyncRAGStore(
                self.db,
                llm_service=self.llm_service,
                agent_id=self.agent_id,
            )
            self.agent_resources = (
                AgentResourceStore(self.db, self.agent_id)
                if self.agent_id else None
            )
            self._initialized = True
            logger.info(f"AsyncStorage initialized ({self.backend_type}): {self.db_path or 'PostgreSQL'}")
    
    async def close(self) -> None:
        """Close the storage connection."""
        try:
            if self.db:
                await self.db.close()
        finally:
            # ``AsyncDatabase.close`` closes the primary backend before it
            # reports a cached SQLAlchemy-factory failure.  Mark the facade
            # closed, but preserve ``self.db`` as the lifecycle owner: a later
            # initialize is fenced while that database retains any live worker
            # and may replace it only after retirement completes.
            self._initialized = False

    async def dispose_cached_sqla_factory(self) -> None:
        """Dispose the optional SQLAlchemy engine before backend close.

        This is intentionally separate from :meth:`close` for whole-agent
        shutdown: its bounded pre-close phase cannot consume the reservation
        that SQLite needs to drain its primary aiosqlite worker.  Ordinary
        callers still get both phases by calling ``close()`` alone.
        """
        if self.db:
            await self.db.dispose_cached_sqla_factory()
    
    async def __aenter__(self):
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        # Validate the cold read only if the block succeeded. Raising here over
        # an exception already in flight would replace the real failure with a
        # complaint about the connection it happened on. Placed in ``__aexit__``
        # rather than at the end of each caller's block because these blocks
        # return early in several branches, and a check the caller can skip by
        # returning is not a check.
        if exc_type is None:
            self.assert_cold_read_still_valid()
        # Reported even when the block failed: the operator needs to know the
        # store was left with WAL state regardless of why the read ended.
        checker = getattr(self._backend, "warn_if_wal_state_was_stranded", None)
        if checker is not None:
            checker()

    def assert_cold_read_still_valid(self) -> None:
        """Refuse to act on a cold read that a writer raced. No-op otherwise.

        Delegated to the backend, which is the only layer that knows whether
        this connection was opened blind to a WAL.
        """
        checker = getattr(self._backend, "assert_cold_read_still_valid", None)
        if checker is not None:
            checker()

    @asynccontextmanager
    async def transaction(self):
        """Run the enclosed storage operations as one atomic write unit.

        Delegates to the backend's transaction context manager: every write
        inside the ``async with`` block commits together or rolls back
        together. Without this, each facade call auto-commits individually,
        so a failure mid-sequence leaves partially-applied state. Both
        backends are re-entrant for the SAME asyncio task, so nested
        ``transaction()`` scopes (e.g. a helper opening its own around a
        caller's outer one) join the outer transaction.
        """
        if not self._initialized:
            await self.initialize()
        async with self.db.transaction():
            yield

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
                               session_id: Optional[str] = None,
                               rendered_content: Optional[str] = None,
                               model: Optional[str] = None,
                               provider: Optional[str] = None) -> None:
        """Add a conversation message.

        Args:
            role: Message role (user, assistant, system)
            content: Canonical raw message content.
            metadata: Optional metadata dict
            session_id: If provided, link this message to a specific session.
                       This allows resuming old conversations beyond the 30-min gap.
            rendered_content: Write-once transport bytes for byte-stable
                cache replay (#1402); see AsyncConversationStore.add_conversation.
            model: Concrete model that produced this message.
            provider: Resolved provider route that produced this message.
        """
        if not self._initialized:
            await self.initialize()
        persisted_metadata = dict(metadata) if metadata else {}
        dependencies = self._semantic_recall_dependencies_for_persistence(
            persisted_metadata
        )
        # Provider embedding and lexical indexing may perform slow I/O.  Do
        # that work before the assertion lifecycle fence; only the exact
        # liveness check and final INSERT may hold tenant serialization.
        prepared = await self.conversation._prepare_conversation_write(  # noqa: SLF001 - coordinated persistence seam
            role,
            content,
            persisted_metadata,
            session_id,
            rendered_content,
            model,
            provider,
        )
        # Token-first lexical work happens before the canonical fence.  Keep
        # its exact handle even if the prepared object later clears it: an
        # INSERT exception inside the fence rolls back in-fence cleanup, so
        # the caller must retry cleanup after the transaction has exited.
        prepared_lexical_index_id = prepared.lexical_index_id

        async def persist() -> None:
            await self.conversation._persist_prepared_conversation(  # noqa: SLF001 - coordinated persistence seam
                prepared
            )

        async def exclude_and_persist() -> None:
            self._exclude_stale_semantic_recall_derivative(prepared.metadata)
            await self.conversation._exclude_prepared_conversation_from_retrieval(  # noqa: SLF001 - coordinated persistence seam
                prepared
            )
            await persist()

        # Only semantic-recall derivatives carry the lineage field.  Normal
        # conversation writes keep their historical no-lock path.
        if "semantic_recall_dependencies" not in persisted_metadata:
            await persist()
            return

        if dependencies is None:
            await exclude_and_persist()
            return

        if not dependencies:
            # An explicitly empty lineage is not a semantic derivative and is
            # produced by the normal no-recall path.  Preserve that behavior.
            await persist()
            return

        try:
            assertions = self._assertion_store()
        except RuntimeError:
            # A partial/unbound storage bootstrap has no canonical authority
            # with which to validate a claimed semantic lineage.  Persisting it
            # visibly would let a caller create an unrevocable derivative.
            await exclude_and_persist()
            return

        try:
            async with assertions._semantic_recall_persistence_fence(
                dependencies
            ) as visible:
                if not visible:
                    await exclude_and_persist()
                    return
                await persist()
        except Exception:
            if prepared_lexical_index_id is not None:
                await self.conversation._discard_lexical_tokens(  # noqa: SLF001 - post-rollback token cleanup
                    prepared_lexical_index_id
                )
            raise

    @staticmethod
    def _semantic_recall_dependencies_for_persistence(
        metadata: Dict[str, Any],
    ) -> tuple[tuple[str, str], ...] | None:
        """Return exact recalled identities or ``None`` for malformed lineage.

        The marker is supplied only by the trusted context-plan projection.  A
        malformed non-empty marker is never silently treated as ordinary
        metadata: it would bypass the canonical liveness fence and make a
        durable derivative unrevocable.  Empty lineage remains the normal
        semantic-recall-disabled/no-result representation.
        """
        if "semantic_recall_dependencies" not in metadata:
            return ()
        raw = metadata.get("semantic_recall_dependencies")
        if not isinstance(raw, list):
            return None
        dependencies: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for item in raw:
            if not isinstance(item, dict):
                return None
            assertion_id = item.get("assertion_id")
            revision_id = item.get("revision_id")
            if (
                not isinstance(assertion_id, str)
                or not assertion_id
                or not isinstance(revision_id, str)
                or not revision_id
            ):
                return None
            dependency = (assertion_id, revision_id)
            if dependency not in seen:
                seen.add(dependency)
                dependencies.append(dependency)
        return tuple(dependencies)

    @staticmethod
    def _exclude_stale_semantic_recall_derivative(
        metadata: Dict[str, Any],
    ) -> None:
        """Make an unverifiable semantic derivative permanently non-contextual."""
        metadata["excluded_from_context"] = True
        metadata.setdefault(
            "excluded_reason", "semantic_assertion_not_current"
        )
    
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

    async def get_session_message_rows(
        self, session_id: str, limit: int = 100
    ) -> List[tuple]:
        """Raw rows for a session via the canonical dual-scheme resolver —
        facade delegator (#2012).

        Returns the 8-tuples ``_get_session_messages`` yields
        ``(id, role, content, metadata, created_at, rendered_content, model,
        provider)``. The privacy wrapper normalizes/decrypts; the endpoints
        format. Exists on the facade because ``PrivacyEnforcingStorage`` wraps
        this ``AsyncStorage``, not the underlying conversation store.
        """
        if not self._initialized:
            await self.initialize()
        return await self.conversation.get_session_message_rows(session_id, limit)

    async def resolve_session_id(self, provided: Optional[str]) -> Optional[str]:
        """Resolve the effective session_id for an incoming turn — facade delegator.

        Exists on the facade because ``PrivacyEnforcingStorage`` (and the agent
        stream/invoke endpoints) call ``storage.resolve_session_id`` rather than
        ``storage.conversation.resolve_session_id``. Without this wrapper the
        call hits ``AttributeError`` and is swallowed into a ``None`` fallback
        (#1599), so every turn derives a *new* implicit session (fragmented
        history) and the chat pane never learns its durable session id — it
        then requests ``/conversations/undefined``.
        """
        if not self._initialized:
            await self.initialize()
        return await self.conversation.resolve_session_id(provided)

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

    async def list_conversation_sessions(
        self, limit: int = 50, include_trashed: bool = False
    ) -> List[Dict[str, Any]]:
        """List session summaries for navigation — facade delegator (#2019)."""
        if not self._initialized:
            await self.initialize()
        return await self.conversation.list_conversation_sessions(
            limit=limit, include_trashed=include_trashed
        )

    async def count_session_messages(
        self, session_id: str, deleted_filter: str = "all"
    ) -> int:
        """Count a session's messages via the resolver — facade delegator (#2019)."""
        if not self._initialized:
            await self.initialize()
        return await self.conversation.count_session_messages(
            session_id, deleted_filter=deleted_filter
        )

    async def message_belongs_to_session(
        self, message_id: Any, session_id: str
    ) -> bool:
        """Whether a message resolves within a session — facade delegator (#2022)."""
        if not self._initialized:
            await self.initialize()
        return await self.conversation.message_belongs_to_session(
            message_id, session_id
        )

    async def find_messages_matching(
        self, content_pattern: str, session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Find messages matching a pattern — facade delegator (#2019)."""
        if not self._initialized:
            await self.initialize()
        return await self.conversation.find_messages_matching(
            content_pattern, session_id=session_id
        )

    async def delete_messages_matching(
        self, content_pattern: str, session_id: Optional[str] = None
    ) -> int:
        """Soft-delete messages matching a pattern — facade delegator (#2019)."""
        if not self._initialized:
            await self.initialize()
        return await self.conversation.delete_messages_matching(
            content_pattern, session_id=session_id
        )

    async def delete_message(self, message_id: int) -> bool:
        """Soft-delete a single message — facade delegator (#763)."""
        if not self._initialized:
            await self.initialize()
        return await self.conversation.delete_message(message_id)

    async def _exclude_semantic_recall_dependencies(
        self,
        *,
        assertion_ids=(),
        revision_ids=(),
    ) -> tuple[int, ...]:
        """Exclude exact assertion-derived conversation artifacts.

        Private on purpose: only the governed assertion lifecycle may call
        this companion operation, never an arbitrary user-supplied selector.
        """
        if not self._initialized:
            await self.initialize()
        return await self.conversation.exclude_semantic_recall_dependencies(
            assertion_ids=assertion_ids,
            revision_ids=revision_ids,
        )

    async def _scrub_semantic_recall_dependencies(
        self,
        *,
        assertion_ids=(),
        revision_ids=(),
    ) -> int:
        """Drop physically erased identities from excluded artifacts."""
        if not self._initialized:
            await self.initialize()
        return await self.conversation.scrub_semantic_recall_dependencies(
            assertion_ids=assertion_ids,
            revision_ids=revision_ids,
        )

    async def _exclude_memory_episodes_for_key_message_ids(
        self, message_ids: tuple[int, ...],
    ) -> tuple[str, ...]:
        """Exclude episodes whose exact key-message identity intersects IDs.

        Episode summaries are derivatives of conversation artifacts.  They
        cannot remain prompt-visible once an input artifact is excluded, even
        if the summary happens not to repeat the fact verbatim.
        """
        if not self._initialized:
            await self.initialize()
        requested_ids = {str(message_id) for message_id in message_ids}
        if not requested_ids:
            return ()
        rows = await self.db.fetchall(
            "SELECT id, key_message_ids FROM memory_episodes "
            "WHERE agent_id = ? AND COALESCE(excluded_from_context, 0) = 0",
            (self.agent_id,),
        )
        excluded: list[str] = []
        for episode_id, raw_key_ids in rows:
            try:
                key_ids = json.loads(raw_key_ids) if isinstance(raw_key_ids, str) else raw_key_ids
            except (TypeError, ValueError):
                continue
            if not isinstance(key_ids, list) or not requested_ids.intersection(
                str(key_id) for key_id in key_ids
            ):
                continue
            result = await self.db.execute_commit(
                "UPDATE memory_episodes SET excluded_from_context = 1 "
                "WHERE id = ? AND agent_id = ? "
                "AND COALESCE(excluded_from_context, 0) = 0",
                (episode_id, self.agent_id),
            )
            if _rows_affected(result) > 0:
                excluded.append(str(episode_id))
        return tuple(excluded)

    async def _withdraw_semantic_recall_derivatives(
        self,
        *,
        assertion_ids=(),
        revision_ids=(),
        physically_erased: bool,
    ) -> None:
        """Apply one canonical lifecycle withdrawal to exact context artifacts.

        ``AsyncAssertionStore`` calls this private companion while its tenant
        mutation remains open.  The conversation/episode updates therefore
        share that transaction on both SQLite and PostgreSQL.  Assertion
        lifecycle code passes only canonical IDs; this facade deliberately
        never interprets message content as a deletion selector.
        """
        message_ids = await self._exclude_semantic_recall_dependencies(
            assertion_ids=assertion_ids,
            revision_ids=revision_ids,
        )
        if message_ids:
            await self._exclude_memory_episodes_for_key_message_ids(message_ids)
        if physically_erased:
            await self._scrub_semantic_recall_dependencies(
                assertion_ids=assertion_ids,
                revision_ids=revision_ids,
            )

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

    async def archive_conversation_session(self, session_id: str) -> int:
        """Archive an entire session — facade delegator (#2149)."""
        if not self._initialized:
            await self.initialize()
        return await self.conversation.archive_conversation_session(session_id)

    async def unarchive_conversation_session(self, session_id: str) -> int:
        """Unarchive an entire session — facade delegator (#2149)."""
        if not self._initialized:
            await self.initialize()
        return await self.conversation.unarchive_conversation_session(session_id)

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

    async def purge_conversations_since(
        self, since_iso: str, *, reason: str = "ephemeral-leak",
    ) -> int:
        """Scoped variant for the EPHEMERAL leak-purge (#867).

        Only destroys rows whose ``created_at >= since_iso``.  ``since_iso``
        is the timestamp the agent entered EPHEMERAL — anything authored
        before that is preexisting NORMAL data the user explicitly wanted
        persisted, and must not be touched.
        """
        if not self._initialized:
            await self.initialize()
        return await self.conversation.purge_all_since(since_iso, reason=reason)

    async def purge_session_projection(
        self, *, reason: str = "ephemeral-leak",
    ) -> int:
        """Erase this agent's #2959 projection state (EPHEMERAL).

        The CACHE unconditionally; the LEDGER only when no live history
        survives. One transaction, and the condition on the ledger is the whole
        design:

        * It is durable, not a count of what THIS attempt deleted. A retry after
          a partial failure asks the same question and gets the same answer,
          where delete counts would read zero and call the residue clean.
        * When it holds, an EMPTY projection is not merely safe to leave behind
          — it is the correct value. A projection describes live history, and
          there is none, so erasing it writes the truth rather than guessing at
          it. Nothing here has to reason about which rows leaked.

        What accrues without anyone asking is the change ledger, because a
        database trigger bumps it on every write to ``conversation_history`` and
        a trigger cannot see privacy mode. A purely EPHEMERAL agent that leaked
        one turn would otherwise leave a row naming it behind after the sweep
        that erased the turn.

        **Why the ledger cannot go alone** (round-6 review). ``is_stale()``
        answers by comparing a stored stamp to the ledger for equality, which is
        sound only while the ledger is monotonic — the claim
        ``ConversationSessionProjection.observed_changes()`` makes in its own
        docstring. Deleting the row breaks that: the trigger's next write is an
        INSERT of ``1``, so the counter restarts. A projection repaired at stamp
        N, purged, and then written to N more times reports itself CURRENT while
        describing history that no longer exists — immediately when N is 1. The
        stamp is meaningful only against the ledger incarnation it was read
        from, so the two are erased together or neither is.

        An earlier revision of this sweep did reach all three tables and needed
        a leak-detection condition, an orphan probe and cross-table atomicity to
        do it safely. That machinery was for the *scoped* case, where some
        history survives and the sweep must separate what leaked from what did
        not. None of it is needed here: this runs only when nothing survives, so
        the answer for every row is the same one.

        An agent whose legitimate pre-EPHEMERAL history survives keeps its
        LEDGER row: it names nothing the database does not already say, and
        deleting it would breach the scoped-purge contract, which forbids
        touching anything authored before entry. Its projection rows go
        regardless — they are derived, not authored, and the leaked session is
        described in them.
        """
        if not self._initialized:
            await self.initialize()
        from .async_conversation_store import _rows_affected
        from .conversation_sessions import projection_tables

        ledger = "conversation_history_changes"
        tables = [
            table for table, _ddl in projection_tables()
            if await self.db.table_exists(table)
        ]
        if not tables:
            return 0
        # Watermark FIRST, ledger LAST, sessions in between. Two orderings have
        # to hold at once and they do not conflict:
        #
        #   * The ledger last is the correctness guarantee (below).
        #   * The watermark first matches the order a REPAIR takes them in —
        #     `_claim()` locks the watermark row and only then upserts session
        #     rows. Taking them the other way round here is the classic ABBA
        #     deadlock: each holds the row the other is waiting for, and
        #     PostgreSQL resolves it by aborting one. Aborting this sweep would
        #     leave projection rows standing after an EPHEMERAL exit that
        #     reported success, because this store is not a required one.
        #
        # The ledger goes last, and that ordering is the guarantee — not the
        # transaction. The state to avoid is a watermark standing beside a
        # missing ledger, because that is the one where a restarted counter can
        # match a stale stamp. Deleting the ledger after everything derived from
        # it makes that state unreachable at every point in the sequence, on any
        # backend, at any isolation level: while the ledger is still there the
        # watermark is either present and consistent or already gone, and once
        # it is gone there is no stamp left to fool. ``projection_tables()``
        # happens to order it last too, for an unrelated reason — the triggers
        # reference it — so it is re-established here rather than inherited from
        # a coincidence a future reordering could quietly take away.
        watermark = "conversation_session_watermarks"
        tables = (
            [t for t in tables if t == watermark]
            + [t for t in tables if t not in (watermark, ledger)]
            + [t for t in tables if t == ledger]
        )

        purged = 0
        # The transaction is what makes the sweep all-or-nothing, and on SQLite
        # (BEGIN IMMEDIATE) it also holds the writer slot across the test. It is
        # not load-bearing for the invariant above: under PostgreSQL READ
        # COMMITTED a row committed after this SELECT is still invisible to it,
        # so a concurrent write can make the survival test stale. What that
        # costs is bounded — the projection is erased for an agent that now has
        # history, which the next repair rebuilds, and the ledger is
        # content-free — and it is the delete ORDER, not the isolation level,
        # that keeps a stale stamp from ever reading as current.
        async with self.db.transaction(immediate=True):
            # The repair exclusion, taken the way a repair takes it. Ordering
            # the deletes below to match a repair's write order is necessary but
            # not sufficient: on PostgreSQL a DELETE matching no rows locks
            # nothing, so a first-time repair can insert the watermark this
            # sweep has already passed over.
            from .conversation_sessions import ConversationSessionProjection

            claim_created_watermark = await ConversationSessionProjection(
                self.db, self.agent_id
            ).claim_exclusion()
            survives = await self.db.fetchval(
                "SELECT 1 FROM conversation_history WHERE agent_id = ? LIMIT 1",
                (self.agent_id,),
            )
            # The CACHE always goes. It is rebuildable from
            # `conversation_history`, so clearing it destroys no record — while
            # leaving it keeps the leaked session's id, timestamps, counts and
            # message pointer standing after a sweep that reported success. An
            # earlier revision skipped everything whenever any history survived,
            # reading the scoped-purge contract as covering this too. That
            # contract is about state authored before entry, and a derived copy
            # is not authored at all: erasing it costs a rebuild, not data
            # (round-16 review).
            #
            # The LEDGER is the exception, and only while history survives. Such
            # an agent is already named by that history, so its counter names
            # nothing new — and erasing it there is exactly what breaks the
            # monotonicity `is_stale()` rests on, since the trigger's next write
            # restarts the count.
            for table in tables:
                if table == ledger and survives:
                    continue
                purged += _rows_affected(
                    await self.db.execute(
                        f"DELETE FROM {table} WHERE agent_id = ?",
                        (self.agent_id,),
                    )
                )
        # Do not count the row the CLAIM just made. On PostgreSQL the exclusion
        # is acquired by inserting the watermark row when it is absent, and the
        # very next statement here deletes it — so every clean stint reported
        # one purged row, and `PurgeOutcome.PURGED` means "a real leak was found
        # and removed". Measured: a stint that never touched storage returned 1.
        # SQLite never saw it, because the claim is a no-op there — which is
        # also why the cases asserting `session_projection == 0` did not catch
        # it (round-20 review).

        if claim_created_watermark and purged:
            purged -= 1
        if purged:
            logger.info(
                "purged the session projection for %s (%s): %d row(s) across %s",
                self.agent_id, reason, purged, ", ".join(tables),
            )
        return purged

    async def purge_channel_messages_since(
        self, since_iso: str, *, reason: str = "ephemeral-leak",
    ) -> int:
        """Scoped EPHEMERAL leak-purge for the channels feature table (#2096).

        ``channel_messages`` is created by the channels feature, not core
        storage, so this primitive tolerates the table being absent (the
        feature was never loaded) and simply reports 0 rows purged. Like
        :meth:`purge_conversations_since`, only rows whose
        ``created_at >= since_iso`` are destroyed so a brief EPHEMERAL stint
        never wipes preexisting NORMAL channel history.

        Timestamp-shape safety (regression fix): the ``since_iso`` watermark is
        SQLite ``datetime('now')`` shape (``YYYY-MM-DD HH:MM:SS``, space
        separator), but ``channel_messages.created_at`` is written by the
        channels feature as ``message.timestamp.isoformat()`` (``T`` separator,
        microseconds, offset). A raw SQL ``created_at >= since_iso`` compares
        lexically, and ``'T'`` (0x54) > ``' '`` (0x20), so a same-UTC-day row
        authored BEFORE the watermark still sorts ``>=`` it and would be wiped —
        destroying preexisting NORMAL history. To be shape- and backend-agnostic
        (and to correctly handle rows already on disk in either format), we parse
        both the watermark and each row's timestamp to aware UTC datetimes and
        compare those, deleting only the ids that truly fall on/after the
        watermark. A row whose timestamp can't be parsed is purged (fail-safe:
        this is a privacy sweep, so an unparseable leaked row must not survive).
        The doomed ids are deleted in bounded batches so a high-volume session
        never trips the backend's bind-parameter ceiling and raises (which the
        caller would swallow, leaving the whole leak behind).
        """
        if not since_iso:
            return 0
        if not self._initialized:
            await self.initialize()
        if not await self.db.table_exists("channel_messages"):
            return 0
        from .async_conversation_store import _rows_affected

        since_dt = _parse_utc_datetime(since_iso)
        if since_dt is None:
            # Watermark itself is unparseable — we cannot scope the purge, and
            # deleting everything would destroy preexisting NORMAL history. Refuse
            # (mirrors the empty-watermark guard) rather than over-delete.
            logger.warning(
                "purge_channel_messages_since: unparseable watermark %r — "
                "refusing to purge (agent=%s, reason=%s)",
                since_iso, self.agent_id, reason,
            )
            return 0

        rows = await self.db.fetchall(
            "SELECT id, created_at FROM channel_messages WHERE agent_id = ?",
            (self.agent_id,),
        )
        doomed_ids = []
        for row in rows or []:
            created_dt = _parse_utc_datetime(row[1])
            # None => unparseable/empty: purge (fail-safe for a privacy sweep).
            if created_dt is None or created_dt >= since_dt:
                doomed_ids.append(row[0])

        if not doomed_ids:
            return 0

        purged = 0
        for start in range(0, len(doomed_ids), _DELETE_ID_BATCH):
            batch = doomed_ids[start:start + _DELETE_ID_BATCH]
            placeholders = ", ".join(["?"] * len(batch))
            affected = await self.db.execute_commit(
                f"DELETE FROM channel_messages "
                f"WHERE agent_id = ? AND id IN ({placeholders})",
                (self.agent_id, *batch),
            )
            purged += _rows_affected(affected)

        if purged:
            logger.info(
                "purge_channel_messages_since agent=%s since=%s reason=%s rows=%d",
                self.agent_id, since_iso, reason, purged,
            )
        return purged

    async def purge_trash_older_than(
        self,
        cutoff_iso: str,
        *,
        max_rows: int = 10_000,
        reason: str = "retention-janitor",
    ) -> int:
        """Retention-janitor primitive — facade delegator (#764).

        Hard-deletes soft-deleted conversation rows whose ``deleted_at``
        is older than the cutoff. Live rows are never touched.
        """
        if not self._initialized:
            await self.initialize()
        return await self.conversation.purge_trash_older_than(
            cutoff_iso, max_rows=max_rows, reason=reason,
        )

    async def purge_agent_graph_nodes(
        self, *, since_iso: Optional[str] = None
    ) -> int:
        """Hard-delete graph nodes owned by this agent (#767/#867).

        Used by the EPHEMERAL hard-purge defense-in-depth — agents in
        EPHEMERAL mode aren't supposed to write to ``graph_nodes`` at
        all, so any rows present are a privacy-layer leak. Scopes by
        the agent_id stored in the node's ``properties`` JSON.

        ``since_iso`` (when provided) further scopes to only nodes whose
        ``properties.created_at >= since_iso``, so the EPHEMERAL leak
        purge only destroys rows authored *during* the EPHEMERAL stint
        and leaves preexisting NORMAL nodes alone.
        """
        if not self._initialized:
            await self.initialize()
        return await self.graph.purge_agent_nodes(
            self.agent_id, since_iso=since_iso
        )

    async def purge_decayed_episodes(
        self,
        *,
        delete_threshold: float,
        grace_days: int,
        max_rows: int = 10_000,
        half_life_days: int = 30,
        reason: str = "forgetting",
    ) -> int:
        """Forgetting primitive — hard-delete *decayed* memory episodes (#1674).

        This is the deletion tier of the importance-decay forgetting curve, NOT
        an age-based sweep. An episode for THIS agent is eligible only when both:

        * it is older than ``grace_days`` (a faded-but-recent episode is never
          deleted — there is always a minimum lifetime), AND
        * its decay strength — ``calculate_decay(created_at, importance)`` on the
          same Ebbinghaus curve used for messages — has fallen below
          ``delete_threshold``.

        Because strength is importance-scaled, a high-importance episode decays
        far slower and survives much longer than a throwaway one of the same age;
        age alone never deletes anything. (Episodes don't yet carry access /
        applied / decay_protected signals — those arrive in P2 — so for now decay
        runs on importance + age, which is already the load-bearing dimension.)

        Eligible episodes are deleted oldest-first, capped at ``max_rows``. Each
        episode is mirrored into the knowledge graph as a node whose ``node_id``
        IS the episode id (see ``memory_consolidator``), so the paired graph node
        — and, via ``delete_node``, its edges — is removed too, leaving no
        orphans. The graph node is deleted BEFORE the episode row, and a row is
        removed ONLY if its node delete succeeded, so a node-delete failure (or a
        mid-sweep crash) can never orphan a graph node. Worst case: an episode
        whose row+node both survive and are retried next run.

        Returns the number of episodes removed. ``reason`` is informational
        (parity with ``purge_trash_older_than``; episodes carry no audit row).
        """
        if not self._initialized:
            await self.initialize()

        # Guard the cap: SQLite reads LIMIT -1 (any max_rows <= 0) as unbounded,
        # which would silently bypass the per-sweep cap. Matches
        # purge_trash_older_than. A zero/negative cap means "purge nothing".
        if max_rows <= 0:
            return 0

        from .memory_retriever import calculate_decay

        # Pre-filter to past-grace rows in SQL; the decay-strength test then runs
        # per row in Python. created_at is written by the consolidator as
        # ``datetime.now(utc).isoformat()`` ("T"+offset), so the cutoff MUST use
        # the same format to sort correctly.
        grace_cutoff_iso = (
            datetime.now(UTC) - timedelta(days=grace_days)
        ).isoformat()

        # Scan past-grace candidates oldest-first in bounded pages rather than
        # pulling the whole tail into memory: a table with many old-but-high-
        # importance episodes (not yet below threshold) could otherwise make the
        # nightly pass O(all old episodes) in memory. We page until we've
        # collected max_rows eligible ids or exhaust the candidates; only the
        # most-decayed (oldest, lowest-importance) survive the strength test.
        # Pagination is snapshot-stable because no rows are deleted mid-scan.
        page_size = max(max_rows, 1000)
        offset = 0
        episode_ids: list[str] = []
        while len(episode_ids) < max_rows:
            page = await self.db.fetchall(
                """
                SELECT id, created_at, importance, access_count FROM memory_episodes
                WHERE agent_id = ? AND created_at < ?
                ORDER BY created_at ASC
                LIMIT ? OFFSET ?
                """,
                (self.agent_id, grace_cutoff_iso, page_size, offset),
            )
            if not page:
                break
            for ep_id, created_at, importance, access_count in page:
                # access_count is the rehearsal signal (#1674 P2): episodes the
                # agent genuinely recalled extend their half-life and resist
                # deletion, exactly as messages do.
                strength = calculate_decay(
                    created_at,
                    importance=importance if importance is not None else 0.5,
                    access_count=access_count if access_count is not None else 0,
                    half_life_days=half_life_days,
                )
                if strength < delete_threshold:
                    episode_ids.append(ep_id)
                    if len(episode_ids) >= max_rows:
                        break
            if len(page) < page_size:
                break
            offset += page_size
        if not episode_ids:
            return 0

        # Drop the paired KG node (+ its edges) first — see docstring. Only
        # episodes whose node delete SUCCEEDS are then removed from the table:
        # deleting a row whose node delete failed would leave exactly the
        # orphan node the ordering is meant to prevent. A skipped episode keeps
        # both its row and node and is retried on the next sweep.
        deletable_ids = []
        for episode_id in episode_ids:
            try:
                await self.graph.delete_node(episode_id)
            except Exception as e:  # noqa: BLE001 - one bad node must not abort the sweep
                logger.warning(
                    "[retention] skipping episode %s — graph node delete failed: %s",
                    episode_id, e,
                )
                continue
            deletable_ids.append(episode_id)

        if not deletable_ids:
            return 0

        placeholders = ",".join("?" for _ in deletable_ids)
        await self.db.execute(
            f"DELETE FROM memory_episodes WHERE id IN ({placeholders})",
            tuple(deletable_ids),
        )
        return len(deletable_ids)

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
    
    async def add_node(self, node: GraphNode, *, capability: Any = None) -> None:
        """Add a node to the knowledge graph.

        ``capability`` is accepted (and ignored) so trusted governance callers
        can pass it uniformly whether they hold the raw facade or the
        privacy-enforcing wrapper, which is where it is actually enforced (#2672).
        """
        if not self._initialized:
            await self.initialize()
        await self.graph.add_node(node)

    async def compare_and_swap_node(
        self,
        node_id: str,
        expected: Optional[Dict[str, Any]],
        new_node: GraphNode,
        allowed_node_types: Optional[frozenset] = None,
        *,
        capability: Any = None,
    ) -> NodeSwapResult:
        """Atomically update a graph node's properties only if they still match.

        Facade delegator onto :meth:`AsyncGraphStore.compare_and_swap_node` —
        the race-free, properties-only conditional-update primitive.
        ``expected`` is the ``properties`` snapshot the caller last read
        (``None`` = compare-and-create); on a swap only ``new_node.properties``
        is written (``node_type`` / ``label`` are left as-is). ``allowed_node_types``
        (optional) constrains the effective node type the swap/create may touch
        — the privacy wrapper uses it to govern durable graph CAS in volatile
        modes. Returns a :class:`NodeSwapResult` (``swapped`` /
        ``predicate_failed`` / ``not_found`` / ``type_not_allowed``).
        """
        if not self._initialized:
            await self.initialize()
        return await self.graph.compare_and_swap_node(
            node_id, expected, new_node, allowed_node_types=allowed_node_types
        )

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
                       properties: Optional[Dict] = None,
                       *, capability: Any = None) -> None:
        """Add an edge between nodes.

        ``capability`` is accepted (and ignored) here for call-site uniformity
        with the privacy wrapper, which enforces it (#2672).
        """
        if not self._initialized:
            await self.initialize()
        await self.graph.add_edge(source_id, target_id, label, properties)

    async def delete_edge(self, source_id: str, target_id: str, label: str) -> None:
        """Remove a specific edge by its (source, target, label) triple."""
        if not self._initialized:
            await self.initialize()
        await self.graph.delete_edge(source_id, target_id, label)

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

    # --- Canonical Semantic Assertion Operations ---

    def _assertion_store(self) -> AsyncAssertionStore:
        if not self._initialized or self._assertions is None:
            raise RuntimeError(
                "Canonical assertion storage requires initialized, agent-bound AsyncStorage"
            )
        return self._assertions

    def semantic_assertion_binding(self) -> SemanticAssertionBinding:
        """Return non-authorizing storage metadata for internal consumers.

        The tenant and owner come from the private, agent-bound assertion
        store—not a caller-selected ``agent_id`` or tool payload.  Foreground
        producers must obtain the wrapper-issued governed binding from
        ``PrivacyEnforcingStorage``; this raw metadata never authorizes a
        feature to skip the current privacy policy.
        """
        store = self._assertion_store()
        return SemanticAssertionBinding(
            tenant_id=store.tenant_id,
            owning_agent_id=store.owning_agent_id,
            privacy_classification="normal",
            release_policy_reference="policy:privacy:normal-v1",
            visibility=Visibility.PRIVATE,
        )

    def legacy_graph_fact_migration(
        self,
        *,
        compatibility_read_enabled: bool = False,
        index_invalidator=None,
    ):
        """Return the explicit, agent-bound #2752 legacy fact migrator.

        The caller cannot supply an arbitrary tenant.  The returned service
        uses this already-authenticated storage's assertion authority and the
        graph ownership ledger, then routes every accepted proposal through
        :meth:`put_validated_assertion`.
        """
        from .legacy_fact_migration import LegacyGraphFactMigration

        return LegacyGraphFactMigration(
            self,
            compatibility_read_enabled=compatibility_read_enabled,
            index_invalidator=index_invalidator,
        )

    async def put_assertion(self, assertion, *, source_occurrences=(), operation_id: Optional[str] = None):
        """Govern canonical ingestion and return its required SHACL report.

        A public agent-bound storage facade is an ingestion boundary, not a
        migration authority.  It therefore cannot create an active canonical
        assertion without the accepted report committed beside it.
        """
        if not self._initialized:
            await self.initialize()
        return await self.semantic_validation_service().put_assertion(
            assertion,
            source_occurrences=source_occurrences,
            operation_id=operation_id,
        )

    async def get_assertion(self, assertion_id: str, *, include_inactive: bool = False):
        if not self._initialized:
            await self.initialize()
        return await self._assertion_store().get_assertion(assertion_id, include_inactive=include_inactive)

    async def get_assertion_revision(self, revision_id: str):
        if not self._initialized:
            await self.initialize()
        return await self._assertion_store().get_revision(revision_id)

    async def query_assertions(self, query=None):
        if not self._initialized:
            await self.initialize()
        return await self._assertion_store().query(query)

    async def list_assertion_revisions(self, assertion_id: str):
        if not self._initialized:
            await self.initialize()
        return await self._assertion_store().list_revisions(assertion_id)

    async def list_assertion_sources(self, assertion_id: str):
        if not self._initialized:
            await self.initialize()
        return await self._assertion_store().list_source_occurrences(assertion_id)

    async def list_assertion_revision_sources(self, revision_id: str):
        """Read exact revision provenance for a governed corpus example."""
        if not self._initialized:
            await self.initialize()
        return await self._assertion_store().list_revision_source_occurrences(revision_id)

    async def list_assertion_revision_sources_batch(self, revision_ids):
        """Read exact revision provenance for a bounded corpus page."""
        if not self._initialized:
            await self.initialize()
        return await self._assertion_store().list_revision_source_occurrences_batch(
            revision_ids
        )

    async def get_source_occurrence(self, source_occurrence_id: str):
        if not self._initialized:
            await self.initialize()
        return await self._assertion_store().get_source_occurrence(source_occurrence_id)

    async def get_derivation_inputs(self, revision_id: str):
        if not self._initialized:
            await self.initialize()
        return await self._assertion_store().derivation_inputs(revision_id)

    async def reactivate_inferred_assertion(self, assertion, *, operation_id: Optional[str] = None):
        if not self._initialized:
            await self.initialize()
        return await self._assertion_store().reactivate_inferred(
            assertion, operation_id=operation_id,
        )

    async def supersede_assertion(self, expected_predecessor_revision_id: str, replacement, *, source_occurrences=(), operation_id: Optional[str] = None):
        """Govern canonical supersession and return its required SHACL report."""
        if not self._initialized:
            await self.initialize()
        return await self.semantic_validation_service().supersede_assertion(
            expected_predecessor_revision_id,
            replacement,
            source_occurrences=source_occurrences,
            operation_id=operation_id,
        )

    async def append_assertion_source(
        self,
        expected_predecessor_revision_id: str,
        replacement,
        *,
        source_occurrences=(),
        operation_id: Optional[str] = None,
    ):
        """Append one direct source through the governed revision lifecycle."""
        if not self._initialized:
            await self.initialize()
        return await self.semantic_validation_service().append_assertion_source(
            expected_predecessor_revision_id,
            replacement,
            source_occurrences=source_occurrences,
            operation_id=operation_id,
        )

    async def _restore_explicit_fact_assertion(
        self,
        expected_terminal_revision_id: str,
        replacement,
        *,
        source_occurrences=(),
        operation_id: Optional[str] = None,
    ):
        """Restore a terminal direct shell through governed SHACL validation."""
        if not self._initialized:
            await self.initialize()
        return await self.semantic_validation_service()._restore_explicit_fact_assertion(
            expected_terminal_revision_id,
            replacement,
            source_occurrences=source_occurrences,
            operation_id=operation_id,
        )

    async def _replay_governed_assertion_operation(
        self,
        operation_id: str,
        binding,
    ):
        """Read one exact accepted governed assertion-write receipt.

        This is a ledger lookup, not a current-state reconstruction.  It is
        used by the privacy-owned explicit-fact path to make delayed retries
        stable after subsequent revisions have been committed.
        """
        if not self._initialized:
            await self.initialize()
        return await self._assertion_store().replay_governed_assertion_operation(
            operation_id,
            binding,
        )

    async def _terminalize_legacy_erased_explicit_fact_operation(
        self,
        operation_id: str,
        binding,
    ):
        """Fail closed on an exact semantic identity erased by a v3 store."""
        if not self._initialized:
            await self.initialize()
        return await self._assertion_store().terminalize_legacy_erased_explicit_fact_operation(
            operation_id,
            binding,
        )

    async def retract_assertion(self, assertion_id: str, expected_revision_id: str, *, operation_id: Optional[str] = None):
        if not self._initialized:
            await self.initialize()
        return await self._assertion_store().retract(
            assertion_id, expected_revision_id, operation_id=operation_id,
        )

    async def delete_assertion(
        self,
        assertion_id: str,
        expected_revision_id: str,
        *,
        operation_id: Optional[str] = None,
    ):
        if not self._initialized:
            await self.initialize()
        return await self._assertion_store().delete(
            assertion_id,
            expected_revision_id,
            operation_id=operation_id,
        )

    async def _delete_explicit_fact_assertion(
        self,
        assertion_id: str,
        expected_revision_id: str,
        *,
        operation_id: str,
        explicit_fact_selector,
    ):
        """Delete one adapter fact while binding its immutable selector."""
        if not self._initialized:
            await self.initialize()
        return await self._assertion_store().delete(
            assertion_id,
            expected_revision_id,
            operation_id=operation_id,
            explicit_fact_selector=explicit_fact_selector,
        )

    async def _replay_explicit_fact_forget_operation(
        self,
        operation_id: str,
        subject,
        predicate,
    ):
        """Read an exact explicit-fact delete or absent-result receipt."""
        if not self._initialized:
            await self.initialize()
        return await self._assertion_store().replay_explicit_fact_forget(
            operation_id,
            subject,
            predicate,
        )

    async def _record_explicit_fact_forget_noop(
        self,
        operation_id: str,
        subject,
        predicate,
    ):
        """Atomically persist a tenant-bound absent-result fact tombstone."""
        if not self._initialized:
            await self.initialize()
        return await self._assertion_store().record_explicit_fact_forget_noop(
            operation_id,
            subject,
            predicate,
        )

    async def invalidate_assertion_eligibility(self, assertion_id: str, expected_revision_id: str, *, operation_id: Optional[str] = None):
        """Withdraw a source's validation eligibility and cascade unsupported inference."""
        if not self._initialized:
            await self.initialize()
        return await self._assertion_store().invalidate_assertion_eligibility(
            assertion_id, expected_revision_id, operation_id=operation_id,
        )

    async def quarantine_assertion_for_validation(
        self,
        assertion_id: str,
        expected_revision_id: str,
        *,
        report_id: str,
        operation_id: Optional[str] = None,
    ):
        """Refuse partial validation repair outside the governed audit path."""
        raise RuntimeError(
            "Direct validation quarantine is unavailable; use "
            "semantic_validation_service().validate_current() or "
            "full_audit_and_repair() so the report and every repair commit atomically"
        )

    async def erase_assertion(self, assertion_id: str, *, operation_id: Optional[str] = None):
        if not self._initialized:
            await self.initialize()
        return await self._assertion_store().erase(assertion_id, operation_id=operation_id)

    async def assertion_checkpoint(self):
        if not self._initialized:
            await self.initialize()
        return await self._assertion_store().checkpoint()

    async def assertion_event_checkpoint(self):
        """Return the exact event cursor used by governed corpus deltas."""
        if not self._initialized:
            await self.initialize()
        from kestrel_sovereign.knowledge.corpus import CorpusCheckpoint

        checkpoint = await self._assertion_store().event_checkpoint()
        return CorpusCheckpoint(
            checkpoint.tenant_id,
            checkpoint.generation,
            checkpoint.latest_event_id,
        )

    async def assertion_changes_since(self, generation: int, *, limit: int = 100):
        if not self._initialized:
            await self.initialize()
        return await self._assertion_store().changes_since(generation, limit=limit)

    async def assertion_changes_after(self, checkpoint, *, limit: int = 100):
        """Read the public governed-corpus stream after one exact cursor."""
        if not self._initialized:
            await self.initialize()
        from kestrel_sovereign.knowledge.corpus import CorpusCheckpoint
        from kestrel_sovereign.storage.async_assertion_store import AssertionCheckpoint

        if not isinstance(checkpoint, CorpusCheckpoint):
            raise TypeError("checkpoint must be CorpusCheckpoint")
        return await self._assertion_store().changes_after(
            AssertionCheckpoint(
                checkpoint.tenant_id,
                checkpoint.generation,
                checkpoint.latest_event_id,
            ),
            limit=limit,
        )

    async def assertion_validation_statuses(self, assertions):
        """Read privacy-safe validation dispositions for the governed corpus."""
        if not self._initialized:
            await self.initialize()
        return await self._assertion_store().validation_statuses(assertions)

    def semantic_validation_service(self):
        """Return the tenant-bound SHACL service for this canonical store."""
        if not self._initialized:
            raise RuntimeError(
                "Governed semantic validation requires initialized, agent-bound AsyncStorage"
            )
        from .semantic_validation import GovernedSemanticValidationService

        return GovernedSemanticValidationService(self._assertion_store())

    async def put_validated_assertion(self, assertion, *, source_occurrences=(), **validation_options):
        """Validate a full tentative post-state before a canonical assertion write.

        This named entry point is equivalent to :meth:`put_assertion`; both
        public ingestion surfaces are governed so callers cannot select an
        unvalidated shortcut.
        """
        if not self._initialized:
            await self.initialize()
        return await self.semantic_validation_service().put_assertion(
            assertion,
            source_occurrences=source_occurrences,
            **validation_options,
        )

    async def supersede_validated_assertion(
        self,
        expected_predecessor_revision_id: str,
        replacement,
        *,
        source_occurrences=(),
        **validation_options,
    ):
        """Validate and atomically commit a canonical assertion replacement.

        This named entry point is equivalent to :meth:`supersede_assertion`;
        no public replacement surface can make data eligible without its SHACL
        report.
        """
        if not self._initialized:
            await self.initialize()
        return await self.semantic_validation_service().supersede_assertion(
            expected_predecessor_revision_id,
            replacement,
            source_occurrences=source_occurrences,
            **validation_options,
        )

    async def assertion_inference_inputs(self, query=None):
        if not self._initialized:
            await self.initialize()
        return await self._assertion_store().inference_inputs(query)

    async def semantic_recall_candidates(
        self, *, query, candidate_scan_limit, inference_profile, inference_limits=None, maintenance_limits=None,
    ):
        """Read bounded recall candidates through the canonical ledger seam."""
        if not self._initialized:
            await self.initialize()
        return await self._assertion_store().recall_candidates(
            query=query, candidate_scan_limit=candidate_scan_limit,
            inference_profile=inference_profile,
            inference_limits=inference_limits,
            maintenance_limits=maintenance_limits,
            semantic_capabilities=self.semantic_capabilities,
            rdf_codec=self._semantic_rdf_codec,
        )

    def semantic_assertion_vector_projection(self, profile):
        """Create the tenant-bound derived assertion-vector projection.

        This is intentionally an explicit capability, not a generic RAG
        vector accessor.  Its candidates have to be canonically hydrated
        before they can enter agent context.
        """
        if not self._initialized:
            raise RuntimeError("semantic vector projection requires initialized AsyncStorage")
        from .semantic_vector_projection import (
            SemanticAssertionVectorProjection,
            _resolve_host_semantic_vector_embedding_provider,
        )

        provider = _resolve_host_semantic_vector_embedding_provider(
            profile,
            self.__semantic_vector_llm_service,
            host_authority=self._assertion_tenant_capability,
        )
        return SemanticAssertionVectorProjection(self._assertion_store(), profile, provider)

    def _kite_release_vector_projection(self):
        """Return the deterministic local projection used only by Kite evidence.

        This is not a general embedding-provider selection API.  The endpoint
        that reaches it is guarded by the isolated test-agent flag, and this
        method additionally requires the same opt-in environment.  It still
        uses the production projection owner and its durable outbox/rebuild
        mechanics; only the local test embedding implementation is fixed.
        """
        if os.environ.get("KESTREL_KITE_RELEASE_EVIDENCE", "").strip().lower() not in {"1", "true", "yes", "on"}:
            raise RuntimeError("Kite release vector projection is unavailable")
        if not self._initialized:
            raise RuntimeError("Kite release vector projection requires initialized AsyncStorage")
        from types import SimpleNamespace
        from .semantic_vector_projection import SemanticVectorProfile

        class _LocalKiteEmbeddingService:
            def describe(self):
                return SimpleNamespace(
                    provider="kite-local", model="deterministic-v1", dim=4,
                    profile_id="kite-release-erasure-v1",
                )

            def semantic_vector_destination(self):
                return "local"

            async def aembed(self, text: str):
                digest = hashlib.sha256(text.encode("utf-8")).digest()
                return [float(int.from_bytes(digest[index:index + 2], "big") + 1) for index in range(0, 8, 2)]

        class _LocalKiteHost:
            def get_embedding_service(self):
                return _LocalKiteEmbeddingService()

        profile = SemanticVectorProfile(
            "kite-release-erasure-v1",
            hashlib.sha256(b"kite-release-erasure-v1").hexdigest(),
            provider="kite-local", model="deterministic-v1", dimension=4,
        )
        from .semantic_vector_projection import (
            SemanticAssertionVectorProjection,
            _resolve_host_semantic_vector_embedding_provider,
        )

        provider = _resolve_host_semantic_vector_embedding_provider(
            profile, _LocalKiteHost(), host_authority=self._assertion_tenant_capability,
        )
        return SemanticAssertionVectorProjection(self._assertion_store(), profile, provider)

    async def hydrate_semantic_recall_candidates(self, assertion_ids, **kwargs):
        if not self._initialized:
            await self.initialize()
        return await self._assertion_store().hydrate_recall_candidates(
            assertion_ids,
            semantic_capabilities=self.semantic_capabilities,
            rdf_codec=self._semantic_rdf_codec,
            **kwargs,
        )

    async def semantic_inference_state(self, profile):
        """Read the durable complete/incomplete status for one exact profile."""
        if not self._initialized:
            await self.initialize()
        from kestrel_sovereign.knowledge.inference import BoundedInferenceService

        return await BoundedInferenceService(self._assertion_store(), profile).closure_state()

    async def explain_semantic_inference(self, assertion_id: str, profile):
        """Read rule and premise-ID lineage for one tenant-local inferred claim."""
        if not self._initialized:
            await self.initialize()
        from kestrel_sovereign.knowledge.inference import BoundedInferenceService

        return await BoundedInferenceService(self._assertion_store(), profile).explain(assertion_id)

    async def materialize_semantic_inference(self, profile, *, limits=None, full_rebuild: bool = False):
        """Advance an incremental semantic closure, or run explicit repair mode."""
        if not self._initialized:
            await self.initialize()
        from kestrel_sovereign.knowledge.inference import BoundedInferenceService

        service = BoundedInferenceService(self._assertion_store(), profile, limits=limits)
        if full_rebuild:
            return await service.rebuild()
        return await service.materialize_incremental()

    async def revoke_semantic_inference(self):
        """Revoke this tenant's materializations after explicit disablement."""
        if not self._initialized:
            await self.initialize()
        from kestrel_sovereign.knowledge.inference import ENGINE_VERSION

        return await self._assertion_store().revoke_semantic_inference(
            ENGINE_VERSION
        )

    async def run_semantic_maintenance(
        self,
        inference_profile,
        *,
        inference_limits=None,
        maintenance_limits=None,
        semantic_capabilities=None,
        full_rebuild: bool = False,
    ):
        """Run the tenant's bounded incremental semantic-maintenance unit.

        ``full_rebuild`` is deliberately an explicit repair knob.  Ordinary
        sleep callers use the incremental path, which consumes the canonical
        assertion change checkpoint before invoking validation or inference.
        """
        if not self._initialized:
            await self.initialize()
        # Retention expiry is part of every real semantic-maintenance unit and
        # uses the storage-owned host clock.  It cannot be advanced by a caller.
        await self._assertion_store().sweep_expired_governed_artifacts()
        from kestrel_sovereign.knowledge.maintenance import SemanticMaintenanceService

        selected_capabilities = self._resolve_semantic_capabilities(
            semantic_capabilities
        )
        service = SemanticMaintenanceService(
            self._assertion_store(),
            inference_profile=inference_profile,
            inference_limits=inference_limits,
            limits=maintenance_limits,
            semantic_capabilities=selected_capabilities,
            rdf_codec=self._semantic_rdf_codec,
        )
        if full_rebuild:
            return await service.rebuild()
        return await service.run()

    async def semantic_maintenance_training_readiness(
        self,
        inference_profile,
        *,
        inference_limits=None,
        maintenance_limits=None,
        semantic_capabilities=None,
        allow_prior_verified_snapshot: bool = False,
        expected_checkpoint=None,
    ):
        """Return the durable semantic prerequisite for scheduled training.

        Scheduler consumers must ask the same coordinator that owns the
        maintenance checkpoint.  Reconstructing the capability identity here
        keeps the gate in lockstep with sleep maintenance when limits, shape
        pins, or inference profiles change.
        """
        if not self._initialized:
            await self.initialize()
        from kestrel_sovereign.knowledge.maintenance import SemanticMaintenanceService

        selected_capabilities = self._resolve_semantic_capabilities(
            semantic_capabilities
        )
        service = SemanticMaintenanceService(
            self._assertion_store(),
            inference_profile=inference_profile,
            inference_limits=inference_limits,
            limits=maintenance_limits,
            semantic_capabilities=selected_capabilities,
            rdf_codec=self._semantic_rdf_codec,
        )
        from kestrel_sovereign.storage.async_assertion_store import AssertionCheckpoint

        bound_checkpoint = None
        if expected_checkpoint is not None:
            bound_checkpoint = AssertionCheckpoint(
                expected_checkpoint.tenant_id,
                expected_checkpoint.generation,
                expected_checkpoint.latest_event_id,
            )
        return await service.training_readiness(
            allow_prior_verified_snapshot=allow_prior_verified_snapshot,
            expected_checkpoint=bound_checkpoint,
        )

    async def semantic_maintenance_capability_versions(
        self,
        inference_profile,
        *,
        inference_limits=None,
        maintenance_limits=None,
        semantic_capabilities=None,
    ):
        """Return the exact semantic capability pins used to verify a corpus."""
        if not self._initialized:
            await self.initialize()
        from kestrel_sovereign.knowledge.maintenance import SemanticMaintenanceService

        selected_capabilities = self._resolve_semantic_capabilities(
            semantic_capabilities
        )
        service = SemanticMaintenanceService(
            self._assertion_store(),
            inference_profile=inference_profile,
            inference_limits=inference_limits,
            limits=maintenance_limits,
            semantic_capabilities=selected_capabilities,
            rdf_codec=self._semantic_rdf_codec,
        )
        return service.capability_versions()

    async def governed_assertion_corpus_snapshot(
        self,
        *,
        policy,
        inference_profile,
        limits=None,
        inference_limits=None,
        maintenance_limits=None,
        semantic_capabilities=None,
        prior_verified_snapshot=None,
        allow_prior_verified_snapshot: bool = False,
        artifact_id: str,
        consumer_id: str,
        consumer_key_id: str,
        consumer_public_key: str,
        retention_seconds: float,
    ):
        """Produce the public immutable learning-corpus snapshot.

        This host service is intentionally the only feature-facing corpus
        ingress; it never returns the database handle or property graph.
        """
        from kestrel_sovereign.knowledge.corpus import (
            GovernedAssertionCorpusService,
            GovernedCorpusLimits,
        )

        snapshot = await GovernedAssertionCorpusService(self).snapshot(
            policy=policy,
            inference_profile=inference_profile,
            limits=limits if limits is not None else GovernedCorpusLimits(),
            inference_limits=inference_limits,
            maintenance_limits=maintenance_limits,
            semantic_capabilities=self._resolve_semantic_capabilities(
                semantic_capabilities
            ),
            prior_verified_snapshot=prior_verified_snapshot,
            allow_prior_verified_snapshot=allow_prior_verified_snapshot,
        )
        await self._register_produced_semantic_artifact(
            artifact_id=artifact_id,
            kind="corpus_manifest",
            consumer_id=consumer_id,
            consumer_key_id=consumer_key_id,
            consumer_public_key=consumer_public_key,
            checkpoint_generation=snapshot.checkpoint.generation,
            policy_pin=snapshot.policy.digest,
            capability_versions=snapshot.capability_versions,
            lineage=tuple(
                (example.assertion.assertion_id, example.assertion.revision_id)
                for example in snapshot.examples
            ),
            retention_seconds=retention_seconds,
            artifact_digest=snapshot.snapshot_hash,
        )
        return snapshot

    async def governed_assertion_corpus_changes_since(
        self,
        snapshot,
        *,
        policy,
        inference_profile,
        limits=None,
        inference_limits=None,
        maintenance_limits=None,
        semantic_capabilities=None,
        artifact_id: str,
        consumer_id: str,
        consumer_key_id: str,
        consumer_public_key: str,
        retention_seconds: float,
    ):
        """Read first-class governed additions/tombstones after a snapshot."""
        from kestrel_sovereign.knowledge.corpus import (
            GovernedAssertionCorpusService,
            GovernedCorpusLimits,
        )

        delta = await GovernedAssertionCorpusService(self).changes_since(
            snapshot,
            policy=policy,
            inference_profile=inference_profile,
            limits=limits if limits is not None else GovernedCorpusLimits(),
            inference_limits=inference_limits,
            maintenance_limits=maintenance_limits,
            semantic_capabilities=self._resolve_semantic_capabilities(
                semantic_capabilities
            ),
        )
        lineage_pairs = {
            (example.assertion.assertion_id, example.assertion.revision_id)
            for example in delta.additions
        }
        lineage_pairs.update(
            (tombstone.assertion_id, tombstone.revision_id)
            for tombstone in delta.tombstones
            if tombstone.assertion_id is not None
            and tombstone.revision_id is not None
        )
        await self._register_produced_semantic_artifact(
            artifact_id=artifact_id,
            kind="future_corpus_candidate",
            consumer_id=consumer_id,
            consumer_key_id=consumer_key_id,
            consumer_public_key=consumer_public_key,
            checkpoint_generation=delta.checkpoint.generation,
            policy_pin=policy.digest,
            capability_versions=snapshot.capability_versions,
            lineage=tuple(sorted(lineage_pairs)),
            retention_seconds=retention_seconds,
            artifact_digest=delta.snapshot_hash,
        )
        return delta

    async def repair_semantic_maintenance(
        self,
        inference_profile,
        *,
        inference_limits=None,
        maintenance_limits=None,
        semantic_capabilities=None,
    ):
        """Explicit full revalidation/rebuild using the normal maintenance service."""
        return await self.run_semantic_maintenance(
            inference_profile,
            inference_limits=inference_limits,
            maintenance_limits=maintenance_limits,
            semantic_capabilities=semantic_capabilities,
            full_rebuild=True,
        )

    def _resolve_semantic_capabilities(self, supplied):
        """Reject per-call semantic pin changes after agent boot.

        A durable maintenance profile, training readiness, and corpus snapshot
        must describe the exact same runtime.  Letting a feature override this
        selection would make a draft profile appear ready under stable pins.
        """
        if supplied is None:
            return self.semantic_capabilities
        if supplied != self.semantic_capabilities:
            raise ValueError(
                "semantic capabilities must match the agent-bound storage runtime"
            )
        return supplied

    def semantic_rdf_capability_report(self):
        """Return the capabilities active in this storage-owned RDF runtime."""
        return self._semantic_rdf_codec.capability_report

    def semantic_sparql12_read_adapter(self, backend, decode_row, **kwargs):
        """Build a draft read adapter from the agent-owned codec only.

        The stable runtime fails closed inside the codec; callers cannot turn
        on SPARQL 1.2 by attaching a query backend to a stable agent.
        """
        return self._semantic_rdf_codec.sparql12_read_adapter(
            backend, decode_row, **kwargs
        )

    async def export_assertion_snapshot(
        self,
        query=None,
        *,
        artifact_id: str,
        consumer_id: str,
        consumer_key_id: str,
        consumer_public_key: str,
        retention_seconds: float,
    ):
        if not self._initialized:
            await self.initialize()
        checkpoint, assertions = await self._assertion_store().export_snapshot(query)
        canonical = json.dumps(
            [assertion.to_mapping() for assertion in assertions],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        policy_pin = hashlib.sha256(
            b"kestrel:semantic-export-policy:v1"
        ).hexdigest()
        await self._register_produced_semantic_artifact(
            artifact_id=artifact_id,
            kind="export_snapshot",
            consumer_id=consumer_id,
            consumer_key_id=consumer_key_id,
            consumer_public_key=consumer_public_key,
            checkpoint_generation=checkpoint.generation,
            policy_pin=policy_pin,
            capability_versions=self.semantic_capabilities.capability_versions(),
            lineage=tuple(
                (assertion.assertion_id, assertion.revision_id)
                for assertion in assertions
            ),
            retention_seconds=retention_seconds,
            artifact_digest=digest,
        )
        return checkpoint, assertions

    async def _register_produced_semantic_artifact(
        self,
        *,
        artifact_id,
        kind,
        consumer_id,
        consumer_key_id,
        consumer_public_key,
        checkpoint_generation,
        policy_pin,
        capability_versions,
        lineage,
        retention_seconds,
        artifact_digest,
    ):
        """Seal producer-derived bytes and trusted runtime pins before return."""
        if not self._initialized:
            await self.initialize()
        if (
            not isinstance(retention_seconds, (int, float))
            or isinstance(retention_seconds, bool)
            or retention_seconds <= 0
        ):
            raise ValueError("retention_seconds must be positive")
        from kestrel_sovereign.knowledge.artifact_lifecycle import (
            GovernedArtifactKind,
            GovernedArtifactLineage,
            GovernedArtifactRegistration,
        )

        capability_pins = {
            name: hashlib.sha256(
                json.dumps(
                    {"name": name, "version": value},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            for name, value in sorted(dict(capability_versions).items())
        }
        clock_value = self._artifact_clock()
        if not isinstance(clock_value, datetime) or clock_value.tzinfo is None:
            raise ValueError("_artifact_clock must return an aware datetime")
        registration = GovernedArtifactRegistration(
            artifact_id=artifact_id,
            kind=GovernedArtifactKind(kind),
            tenant_id=self._assertion_store().tenant_id,
            consumer_id=consumer_id,
            consumer_key_id=consumer_key_id,
            consumer_public_key=consumer_public_key,
            checkpoint_generation=checkpoint_generation,
            policy_pin=policy_pin,
            capability_pins=capability_pins,
            lineage=tuple(
                GovernedArtifactLineage(assertion_id, revision_id)
                for assertion_id, revision_id in lineage
            ),
            retention_expires_at=(
                clock_value.astimezone(UTC)
                + timedelta(seconds=float(retention_seconds))
            ).isoformat().replace("+00:00", "Z"),
            artifact_digest=artifact_digest,
        )
        return await self._assertion_store().register_governed_artifact(registration)

    async def consume_governed_semantic_artifact(self, artifact_id, *, expected_generation: int):
        """Generation-fenced guard for a previously sealed semantic artifact."""
        if not self._initialized:
            await self.initialize()
        return await self._assertion_store().consume_governed_artifact(
            artifact_id, expected_generation=expected_generation
        )

    async def claim_governed_semantic_artifact_revocation(self, authentication, *, lease_seconds: float = 60.0):
        if not self._initialized:
            await self.initialize()
        return await self._assertion_store().claim_governed_artifact_revocation(
            authentication, lease_seconds=lease_seconds
        )

    async def acknowledge_governed_semantic_artifact_revocation(
        self, lease, proof,
    ):
        if not self._initialized:
            await self.initialize()
        return await self._assertion_store().acknowledge_governed_artifact_revocation(
            lease, proof
        )

    async def process_governed_semantic_artifact_revocation(
        self, authentication, owner, *, lease_seconds: float = 60.0,
    ):
        if not self._initialized:
            await self.initialize()
        return await self._assertion_store().process_governed_artifact_revocation(
            authentication, owner, lease_seconds=lease_seconds
        )

    async def sweep_expired_governed_semantic_artifacts(self, *, limit: int = 100):
        # Legacy/test storage instances without authenticated assertion-tenant
        # authority cannot own a governed artifact registry.  Treat that
        # capability absence as an unconfigured no-op; once authority exists,
        # every initialization/database/sweep failure still propagates to the
        # sleep fail-closed path.
        if self._assertion_tenant_capability is None:
            return 0
        if not self._initialized:
            await self.initialize()
        return await self._assertion_store().sweep_expired_governed_artifacts(
            limit=limit
        )

    async def governed_semantic_artifact_erasure_observation(self, *, expected_generation: int):
        if not self._initialized:
            await self.initialize()
        return await self._assertion_store().governed_artifact_erasure_observation(
            expected_generation=expected_generation
        )

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
    
    async def search_chunks(
        self, query: str, limit: int = 5, min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Search document chunks.

        ``min_score`` (#1404) is forwarded to the embedding-search
        candidate filter so weak semantic matches never enter the RRF
        merge; see AsyncRagStore.search_chunks.
        """
        if not self._initialized:
            await self.initialize()
        return await self.rag.search_chunks(query, limit, min_score=min_score)
    
    # --- Case Law / Audit Search ---
    
    async def search_case_law(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Orchestrates a semantic search for relevant "case law" (past audit failures).
        """
        if not self._initialized:
            await self.initialize()
        failures = await self.conversation.get_all_audit_failures()
        return await self.rag.search_case_law(query, failures, top_k)

    # --- Private Agent Identity Resources ---

    def _require_agent_resource_store(self) -> AgentResourceStore:
        if not self.agent_resources:
            raise ValueError("agent_id is required for private agent resources")
        return self.agent_resources

    async def create_agent_resource_version(
        self,
        resource_type: str,
        content: str,
        *,
        created_by: str,
        source: str,
        make_current: bool = True,
        signature: Optional[Dict[str, Any]] = None,
        anchoring_metadata: Optional[Dict[str, Any]] = None,
        public_metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentResourceVersion:
        """Create an encrypted private identity-resource version."""
        if not self._initialized:
            await self.initialize()
        resource = await self._require_agent_resource_store().create_version(
            resource_type,
            content,
            created_by=created_by,
            source=source,
            make_current=make_current,
            signature=signature,
            anchoring_metadata=anchoring_metadata,
            public_metadata=public_metadata,
        )
        if resource_type == SOUL_MARKDOWN_RESOURCE_TYPE:
            await self._record_soul_resource_reference(resource)
        return resource

    async def get_current_agent_resource(
        self,
        resource_type: str = SOUL_MARKDOWN_RESOURCE_TYPE,
    ) -> Optional[AgentResourceVersion]:
        """Load the current encrypted private identity resource."""
        if not self._initialized:
            await self.initialize()
        return await self._require_agent_resource_store().get_current(resource_type)

    async def get_agent_resource_public_metadata(
        self,
        resource_type: str = SOUL_MARKDOWN_RESOURCE_TYPE,
    ) -> Optional[Dict[str, Any]]:
        """Return hash/pointer metadata without private resource contents."""
        if not self._initialized:
            await self.initialize()
        return await self._require_agent_resource_store().get_public_metadata(
            resource_type
        )

    async def promote_soul_seed(
        self,
        content: str,
        *,
        created_by: Optional[str] = None,
        source: str = "agent_data/SOUL.md",
    ) -> AgentResourceVersion:
        """Promote a local SOUL.md seed/cache body into canonical storage."""
        if not self._initialized:
            await self.initialize()
        resource = await self._require_agent_resource_store().promote_soul_seed(
            content,
            created_by=created_by or self.agent_id,
            source=source,
        )
        await self._record_soul_resource_reference(resource)
        return resource

    async def _record_soul_resource_reference(
        self, resource: AgentResourceVersion
    ) -> None:
        """Record body-free KG facts that the SOUL resource exists."""
        properties = {
            "agent_id": self.agent_id,
            "resource_id": resource.resource_id,
            "resource_type": resource.resource_type,
            "current_version": resource.version,
            "content_hash": resource.content_hash,
            "content_bytes": resource.content_bytes,
            "private_body": True,
            "created_at": datetime.now(UTC).isoformat(),
            "provenance": resource.provenance,
        }
        await self.add_node(
            GraphNode(
                node_id=f"{self.agent_id}#soul",
                node_type="agent_identity_resource",
                label="Private SOUL resource",
                properties=properties,
            )
        )
        try:
            await self.add_edge(
                self.agent_id,
                f"{self.agent_id}#soul",
                "has_private_identity_resource",
                {"resource_type": resource.resource_type},
            )
        except Exception:
            logger.debug("Agent graph node missing while linking SOUL resource")
    
    # --- Backup/Restore Operations ---
    
    async def create_backup_blob(self, include_db: bool = True) -> bytes:
        """
        Creates a gzipped tar archive of selected artifacts and returns its bytes.
        Currently includes only the SQLite DB when include_db is True.

        The live database connection is never closed. We produce a consistent
        snapshot of the running DB with SQLite's online backup API and archive
        *that copy*, so concurrent reads/writes keep working and there is no
        close/re-init window. The (potentially large) tar+gzip runs off the
        event loop via ``asyncio.to_thread``.
        """
        if self.db_path == ":memory:":
            raise ValueError("Cannot backup an in-memory database")

        if not include_db:
            # Nothing to include yet — return an empty gzipped tar off-loop.
            return await asyncio.to_thread(self._tar_gzip_paths, [])

        if not self._initialized or not self.db:
            await self.initialize()

        if self.backend_type != "sqlite":
            # The online-backup snapshot path (backup_to) is a SQLite-only
            # facility; PostgreSQL has no equivalent on the backend contract.
            raise NotImplementedError(
                "create_backup_blob currently supports SQLite only "
                f"(backend_type={self.backend_type!r})"
            )

        # Snapshot the live DB into a temp file via the online backup API.
        # The shared connection stays open the whole time.
        tmp_dir = tempfile.mkdtemp(prefix="kestrel-backup-")
        snapshot_path = os.path.join(tmp_dir, "kestrel.db")
        try:
            await self.db.backend.backup_to(snapshot_path)
            # Archive the consistent copy off the event loop.
            return await asyncio.to_thread(
                self._tar_gzip_paths, [(snapshot_path, "kestrel.db")]
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    @staticmethod
    def _tar_gzip_paths(members: List[tuple]) -> bytes:
        """Build a gzipped tar of ``(path, arcname)`` members and return its bytes."""
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode='w:gz') as tar:
            for path, arcname in members:
                tar.add(path, arcname=arcname)
        buffer.seek(0)
        return buffer.read()
    
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
                    info_cursor = await backup_conn.execute(
                        "PRAGMA table_info(conversation_history)"
                    )
                    backup_info = await info_cursor.fetchall()
                    backup_cols = [
                        row[1]
                        for row in backup_info
                    ]
                    has_model = "model" in backup_cols
                    has_provider = "provider" in backup_cols
                    has_deleted_at = "deleted_at" in backup_cols
                    # Preserve the original created_at ORDER and select it +
                    # deleted_at so the restore is FAITHFUL (#F265): rewriting
                    # created_at to now() destroyed history ordering, and
                    # dropping deleted_at resurrected trashed (soft-deleted)
                    # messages. session_id rides inside ``metadata`` and is
                    # carried verbatim; the indexed column is re-derived from
                    # it below, so a backup taken before that column existed
                    # restores with it populated wherever the stored id is
                    # inside the column's contract, and NULL where it is not
                    # (#2958).
                    cursor = await backup_conn.execute(
                        "SELECT role, content, metadata"
                        + (", model" if has_model else ", NULL AS model")
                        + (", provider" if has_provider else ", NULL AS provider")
                        + ", created_at"
                        + (", deleted_at" if has_deleted_at else ", NULL AS deleted_at")
                        + " FROM conversation_history"
                        # Tie-break on the original row id: created_at is often
                        # second-granularity, so same-second turns must keep
                        # their original order — new ids are assigned in this
                        # order and get_conversation_history() sorts by id, so a
                        # tie here would swap user/assistant turns (codex P2).
                        + " ORDER BY created_at, id"
                    )
                    conversations = await cursor.fetchall()

                    # created_at/deleted_at come out of the SQLite backup as
                    # TEXT strings. Binding a string to a Postgres TIMESTAMP
                    # column fails: PostgresBackend._strip_tz only handles
                    # NORMALIZED on both backends, not just PostgreSQL (#3009).
                    #
                    # This restores from a backup's SQLite FILE, so `created_at`
                    # is whatever text the source database happened to hold —
                    # an older kestrel's spelling, an import, a hand-edited row.
                    # Passing it through verbatim (which the SQLite path did,
                    # because asyncpg was the only reason to convert) is the one
                    # writer in this codebase that can put a value into
                    # `conversation_history.created_at` that no reader can date.
                    # Every other writer takes CURRENT_TIMESTAMP or goes through
                    # `SovereignAdapter._restored_created_at`, which parses and
                    # re-spells; this now agrees with it.
                    #
                    # The column cannot enforce this itself: SQLite has no
                    # datetime type, so `TIMESTAMP` is NUMERIC affinity and an
                    # ISO string is stored as TEXT. The rule has to live at the
                    # writers until #3009 adds the CHECK.
                    #
                    # A value nothing can date is no longer written as-is:
                    # since #3009 the column carries a CHECK, so "as-is" is not
                    # an option the database still offers, and a restore that
                    # raised on one bad row of a hundred thousand would lose
                    # the whole history to save a field. It takes the stamp of
                    # its nearest readable neighbour instead — the same rule
                    # the boot migration applies, from `derived_stamp`, and the
                    # same one every reader was already applying to such a row
                    # on the fly. Nothing is invented and nothing is silent:
                    # the original text is logged, and the count is returned to
                    # the caller in `stats`. That is where the restore's report
                    # goes, rather than into `conversation_history_undated` —
                    # the migration needs that table because it overwrites the
                    # only copy of the original, and a restore does not: the
                    # backup it is reading still holds it.
                    is_pg = self.backend_type == "postgres"

                    def _ts(val):
                        """Bind a nullable stamp — `deleted_at`, which may be
                        genuinely absent and carries no canonical guarantee."""
                        if val is None:
                            return None
                        parsed = _parse_utc_datetime(val)
                        if parsed is None:
                            return val
                        naive_utc = parsed.replace(tzinfo=None)
                        return parsed if is_pg else naive_utc.strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )

                    # Computed over the whole ordered list, not row by row, so
                    # a run of undatable rows at the very START of a history
                    # can still borrow forward from the first readable row
                    # after it instead of falling to 1970.
                    stamps = fill_undatable([row[5] for row in conversations])

                    for (
                        role, content, metadata_json, model, provider,
                        created_at, deleted_at,
                    ), (stamp, origin) in zip(conversations, stamps):
                        if origin != "stored":
                            stats["messages_with_unreadable_created_at"] = (
                                stats.get("messages_with_unreadable_created_at", 0)
                                + 1
                            )
                            logger.warning(
                                "restore: created_at %r cannot be dated; "
                                "storing %s, taken from this row's %s (#3009)",
                                created_at, stamp, origin,
                            )
                        # Insert into the current database under this agent_id,
                        # PRESERVING created_at (ordering) and deleted_at (trash
                        # stays trash — a restore must not un-delete rows).
                        await self.db.execute_commit(
                            "INSERT INTO conversation_history "
                            "(agent_id, role, content, model, provider, metadata, "
                            "session_id, created_at, deleted_at) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (
                                self.agent_id, role, content, model, provider,
                                metadata_json,
                                column_session_id(metadata_json),
                                created_at_bind(self.backend_type, stamp),
                                _ts(deleted_at),
                            )
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
