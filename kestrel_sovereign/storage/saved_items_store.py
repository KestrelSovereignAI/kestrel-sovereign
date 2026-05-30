"""
Saved Items Store for Kestrel.

Unified storage for persisting stashes, files, conversation excerpts,
and structured items with embeddings for semantic search.

Features:
- Content deduplication via SHA256 hash
- Semantic search via embeddings
- Optional IPFS pinning for decentralized storage
- Structured item schema validation
"""
import hashlib
import json
import logging
import struct
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .async_database import AsyncDatabase

if TYPE_CHECKING:
    from kestrel_sovereign.filecoin_adapter import FilecoinAdapter

logger = logging.getLogger(__name__)


class SavedItemType(Enum):
    """Types of items that can be saved."""
    STASH = "stash"           # Persisted context stash
    FILE = "file"             # Saved file/document
    EXCERPT = "excerpt"       # Conversation excerpt
    STRUCTURED = "structured" # Typed item (recipe, story, etc.)


class SourceType(Enum):
    """Source of the saved content."""
    CONVERSATION = "conversation"  # From conversation history
    FILE = "file"                  # From file system
    URL = "url"                    # From web
    MANUAL = "manual"              # User/agent created


# =============================================================================
# Structured Item Schemas
# =============================================================================

# Built-in schema definitions for common structured item types
ITEM_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "recipe": {
        "id": "recipe",
        "name": "Recipe",
        "description": "A cooking recipe with ingredients and instructions",
        "required_fields": ["title", "ingredients", "instructions"],
        "optional_fields": ["prep_time", "cook_time", "servings", "cuisine", "difficulty", "notes"],
        "field_types": {
            "title": "string",
            "ingredients": "list",
            "instructions": "list",
            "prep_time": "string",
            "cook_time": "string",
            "servings": "integer",
            "cuisine": "string",
            "difficulty": "string",
            "notes": "string"
        }
    },
    "contact": {
        "id": "contact",
        "name": "Contact",
        "description": "A person or organization contact",
        "required_fields": ["name"],
        "optional_fields": ["email", "phone", "address", "organization", "notes", "relationship"],
        "field_types": {
            "name": "string",
            "email": "string",
            "phone": "string",
            "address": "string",
            "organization": "string",
            "notes": "string",
            "relationship": "string"
        }
    },
    "story": {
        "id": "story",
        "name": "Story/Memory",
        "description": "A personal story, memory, or narrative",
        "required_fields": ["title", "content"],
        "optional_fields": ["date", "location", "people", "emotions", "significance"],
        "field_types": {
            "title": "string",
            "content": "string",
            "date": "string",
            "location": "string",
            "people": "list",
            "emotions": "list",
            "significance": "string"
        }
    },
    "note": {
        "id": "note",
        "name": "Note",
        "description": "A general purpose note or reminder",
        "required_fields": ["content"],
        "optional_fields": ["title", "category", "due_date", "priority"],
        "field_types": {
            "content": "string",
            "title": "string",
            "category": "string",
            "due_date": "string",
            "priority": "string"
        }
    },
    "bookmark": {
        "id": "bookmark",
        "name": "Bookmark",
        "description": "A saved URL or reference",
        "required_fields": ["url", "title"],
        "optional_fields": ["description", "category", "favicon"],
        "field_types": {
            "url": "string",
            "title": "string",
            "description": "string",
            "category": "string",
            "favicon": "string"
        }
    },
    "health_record": {
        "id": "health_record",
        "name": "Health Record",
        "description": "Health-related information (medications, conditions, appointments)",
        "required_fields": ["record_type", "content"],
        "optional_fields": ["date", "provider", "notes", "next_steps"],
        "field_types": {
            "record_type": "string",
            "content": "string",
            "date": "string",
            "provider": "string",
            "notes": "string",
            "next_steps": "string"
        }
    }
}


def get_schema(schema_id: str) -> Optional[Dict[str, Any]]:
    """Get a schema definition by ID."""
    return ITEM_SCHEMAS.get(schema_id)


def list_schemas() -> List[Dict[str, Any]]:
    """List all available schemas."""
    return [
        {"id": s["id"], "name": s["name"], "description": s["description"]}
        for s in ITEM_SCHEMAS.values()
    ]


def validate_structured_content(content: Dict[str, Any], schema_id: str) -> Dict[str, Any]:
    """
    Validate structured content against a schema.

    Args:
        content: The content dict to validate
        schema_id: The schema ID to validate against

    Returns:
        Dict with 'valid' bool and 'errors' list

    Raises:
        ValueError: If schema_id is not found
    """
    schema = get_schema(schema_id)
    if not schema:
        raise ValueError(f"Unknown schema: {schema_id}")

    errors = []

    # Check required fields
    for field in schema["required_fields"]:
        if field not in content or content[field] is None:
            errors.append(f"Missing required field: {field}")
        elif content[field] == "" or content[field] == []:
            errors.append(f"Required field is empty: {field}")

    # Check field types
    field_types = schema.get("field_types", {})
    for field, value in content.items():
        if field in field_types and value is not None:
            expected_type = field_types[field]
            if expected_type == "string" and not isinstance(value, str):
                errors.append(f"Field '{field}' should be a string")
            elif expected_type == "integer" and not isinstance(value, int):
                errors.append(f"Field '{field}' should be an integer")
            elif expected_type == "list" and not isinstance(value, list):
                errors.append(f"Field '{field}' should be a list")

    return {
        "valid": len(errors) == 0,
        "errors": errors
    }


@dataclass
class SavedItem:
    """A saved item with embedding for semantic search."""
    id: str
    agent_id: str
    item_type: str          # SavedItemType value
    name: str
    content: str            # JSON or raw text
    summary: Optional[str] = None
    content_hash: Optional[str] = None
    ipfs_cid: Optional[str] = None
    embedding: Optional[List[float]] = None
    source_type: Optional[str] = None  # SourceType value
    source_ref: Optional[str] = None   # Original IDs/path
    schema_id: Optional[str] = None    # For structured items
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for storage."""
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "item_type": self.item_type,
            "name": self.name,
            "summary": self.summary,
            "content": self.content,
            "content_hash": self.content_hash,
            "ipfs_cid": self.ipfs_cid,
            "source_type": self.source_type,
            "source_ref": self.source_ref,
            "schema_id": self.schema_id,
            "tags": self.tags,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    @classmethod
    def from_row(cls, row: tuple) -> "SavedItem":
        """Create from database row."""
        return cls(
            id=row[0],
            agent_id=row[1],
            item_type=row[2],
            name=row[3],
            summary=row[4],
            content=row[5],
            content_hash=row[6],
            ipfs_cid=row[7],
            embedding=_deserialize_embedding(row[8]) if row[8] else None,
            source_type=row[9],
            source_ref=row[10],
            schema_id=row[11],
            tags=json.loads(row[12]) if row[12] else [],
            metadata=json.loads(row[13]) if row[13] else {},
            # Asyncpg returns ``datetime.datetime`` directly for
            # TIMESTAMP columns on Postgres; aiosqlite returns ISO
            # strings via the TEXT-affinity coercion. Accept either
            # shape so SavedItemsStore is portable across both
            # backends (caught by the vector-lift e2e validation
            # script: save_item had never been exercised on PG until
            # then).
            created_at=row[14] if isinstance(row[14], datetime) else (
                datetime.fromisoformat(row[14]) if row[14] else None
            ),
            updated_at=row[15] if isinstance(row[15], datetime) else (
                datetime.fromisoformat(row[15]) if row[15] else None
            ),
        )


def _now_for_backend(db, now: datetime) -> tuple:
    """Return ``(created_at, updated_at)`` bind values shaped for the
    target backend.

    - **Postgres** (asyncpg): pass ``datetime`` objects directly.
      Binding strings raises ``invalid input for query argument`` on
      TIMESTAMP columns.
    - **SQLite** (aiosqlite via Python's stdlib sqlite3 adapter): emit
      ``.isoformat()`` strings to preserve the historical on-disk
      encoding. Existing rows used the ``T`` separator; sqlite3's
      built-in datetime adapter would write the SPACE separator,
      which sorts differently under lexicographic ORDER BY and
      breaks ``list_items`` ordering. (Caught by codex review on the
      vector-lift validation PR.)
    """
    backend_type = getattr(db, "backend_type", None)
    if backend_type == "postgres":
        return (now, now)
    iso = now.isoformat()
    return (iso, iso)


def _serialize_embedding(embedding: List[float]) -> bytes:
    """Serialize embedding to bytes for SQLite storage."""
    return struct.pack(f'{len(embedding)}f', *embedding)


def _deserialize_embedding(data: bytes) -> List[float]:
    """Deserialize embedding from bytes."""
    count = len(data) // 4  # 4 bytes per float
    return list(struct.unpack(f'{count}f', data))


def _compute_content_hash(content: str) -> str:
    """Compute SHA256 hash of content for deduplication."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


class SavedItemsStore:
    """
    Store for saved items with embedding-based semantic search.

    Supports saving:
    - Stashes (persisted context)
    - Files (documents)
    - Excerpts (conversation snippets)
    - Structured items (recipes, stories, etc.)

    Features:
    - Content deduplication via SHA256 hash
    - Semantic search via embeddings
    - Optional IPFS pinning for decentralized storage
    - Structured item schema validation
    """

    def __init__(
        self,
        db: AsyncDatabase,
        agent_id: str,
        filecoin_adapter: Optional["FilecoinAdapter"] = None
    ):
        self.db = db
        self.agent_id = agent_id
        self._embedding_service = None
        self._filecoin_adapter = filecoin_adapter

    def _get_filecoin_adapter(self) -> Optional["FilecoinAdapter"]:
        """Get or lazy-load the FilecoinAdapter for IPFS operations."""
        if self._filecoin_adapter is not None:
            return self._filecoin_adapter

        # Try to create one if not provided
        try:
            from kestrel_sovereign.filecoin_adapter import FilecoinAdapter
            self._filecoin_adapter = FilecoinAdapter()
            if not self._filecoin_adapter.ipfs_is_available():
                logger.info("IPFS not available, items will be stored locally only")
                self._filecoin_adapter = None
        except Exception as e:
            logger.warning(f"FilecoinAdapter not available: {e}")
            self._filecoin_adapter = None

        return self._filecoin_adapter

    def _get_embedding_service(self):
        """Lazy load the embedding service."""
        if self._embedding_service is None:
            try:
                from kestrel_sovereign.llm.embedding_service import get_embedding_service
                self._embedding_service = get_embedding_service()
            except Exception as e:
                logger.warning(f"Embedding service not available: {e}")
                self._embedding_service = False  # Mark as unavailable
        return self._embedding_service if self._embedding_service else None

    async def save_item(
        self,
        item_type: str,
        name: str,
        content: str,
        summary: Optional[str] = None,
        source_type: Optional[str] = None,
        source_ref: Optional[str] = None,
        schema_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        compute_embedding: bool = True,
        deduplicate: bool = True,
        pin_to_ipfs: bool = False,
        validate_schema: bool = True
    ) -> SavedItem:
        """
        Save an item with optional embedding and IPFS pinning.

        Args:
            item_type: Type of item (stash, file, excerpt, structured)
            name: Human-readable name
            content: The content to save (JSON or text)
            summary: Optional summary for display/search
            source_type: Where the content came from
            source_ref: Reference to original (IDs, path, URL)
            schema_id: For structured items, the schema type
            tags: Optional tags for filtering
            metadata: Additional metadata
            compute_embedding: Whether to generate embedding
            deduplicate: Whether to check for existing content
            pin_to_ipfs: Whether to pin content to IPFS
            validate_schema: Whether to validate structured items against schema

        Returns:
            The saved item

        Raises:
            ValueError: If schema validation fails (when validate_schema=True)
        """
        content_hash = _compute_content_hash(content)
        now = datetime.now(timezone.utc)

        # Validate structured items against schema
        if item_type == SavedItemType.STRUCTURED.value and schema_id and validate_schema:
            try:
                content_dict = json.loads(content)
                validation = validate_structured_content(content_dict, schema_id)
                if not validation["valid"]:
                    raise ValueError(f"Schema validation failed: {', '.join(validation['errors'])}")
            except json.JSONDecodeError:
                raise ValueError("Structured item content must be valid JSON")

        # Check for an existing item representing the same logical record.
        # Raw content alone is not enough, or cross-type saves collapse into
        # one row and silently overwrite metadata.
        if deduplicate:
            existing = await self._find_duplicate(
                content_hash=content_hash,
                item_type=item_type,
                schema_id=schema_id,
                source_type=source_type,
                source_ref=source_ref,
            )
            if existing:
                logger.info(f"Found existing item with same content: {existing.id}")
                # Update the existing item
                return await self.update_item(
                    item_id=existing.id,
                    name=name,
                    summary=summary,
                    tags=tags,
                    metadata=metadata
                )

        # Generate ID
        item_id = str(uuid.uuid4())

        # Compute embedding if requested
        embedding = None
        embedding_blob = None
        if compute_embedding:
            embedding_service = self._get_embedding_service()
            if embedding_service:
                try:
                    # Embed summary if available, otherwise first 1000 chars of content
                    text_to_embed = summary or content[:1000]
                    embedding = await embedding_service.aembed(text_to_embed)
                    if embedding:
                        embedding_blob = _serialize_embedding(embedding)
                except Exception as e:
                    logger.warning(f"Failed to compute embedding: {e}")

        # Pin to IPFS if requested
        ipfs_cid = None
        if pin_to_ipfs:
            ipfs_cid = await self._pin_to_ipfs(content, content_hash)

        # Insert into database. Legacy ``embedding`` BYTEA / BLOB
        # column is written here unchanged; the parallel
        # ``embedding_vec`` column added by the Phase-2 migration is
        # populated by ``_write_embedding_vec`` below so the vector
        # search backend can pick it up.
        await self.db.execute(
            """INSERT INTO saved_items
               (id, agent_id, item_type, name, summary, content, content_hash,
                ipfs_cid, embedding, source_type, source_ref, schema_id, tags, metadata,
                created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                item_id,
                self.agent_id,
                item_type,
                name,
                summary,
                content,
                content_hash,
                ipfs_cid,
                embedding_blob,
                source_type,
                source_ref,
                schema_id,
                json.dumps(tags or []),
                json.dumps(metadata or {}),
                # Backend-aware timestamp binding (caught by codex
                # review on the vector-lift validation script):
                # - asyncpg requires a real ``datetime`` for
                #   TIMESTAMP columns; binding a string raises
                #   ``invalid input for query argument``.
                # - sqlite3's stdlib adapter encodes a ``datetime``
                #   as ``YYYY-MM-DD HH:MM:SS...`` (SPACE separator),
                #   but existing ``saved_items`` rows in SQLite were
                #   written with ``.isoformat()`` (``T`` separator).
                #   ``list_items`` orders this TEXT column
                #   lexicographically; mixing the two encodings
                #   sorts rows incorrectly (space < T, so newer
                #   rows would precede older same-day rows).
                # Preserve the existing SQLite shape while giving
                # Postgres what it actually needs.
                *_now_for_backend(self.db, now),
            )
        )
        if embedding is not None:
            await self._write_embedding_vec(item_id, embedding)
        await self.db.commit()

        return SavedItem(
            id=item_id,
            agent_id=self.agent_id,
            item_type=item_type,
            name=name,
            summary=summary,
            content=content,
            content_hash=content_hash,
            ipfs_cid=ipfs_cid,
            embedding=embedding,
            source_type=source_type,
            source_ref=source_ref,
            schema_id=schema_id,
            tags=tags or [],
            metadata=metadata or {},
            created_at=now,
            updated_at=now
        )

    async def _pin_to_ipfs(self, content: str, content_hash: str) -> Optional[str]:
        """
        Pin content to IPFS.

        Args:
            content: The content to pin
            content_hash: Pre-computed content hash

        Returns:
            IPFS CID if successful, None otherwise
        """
        adapter = self._get_filecoin_adapter()
        if not adapter:
            logger.info("IPFS not available, skipping pinning")
            return None

        try:
            from kestrel_sovereign.storage.providers.base import StorageTier

            result = adapter.store_content(
                content=content.encode('utf-8'),
                storage_tier=StorageTier.IPFS,
                encrypt=False,
                metadata={"type": "saved_item", "content_hash": content_hash}
            )

            if result.cid:
                logger.info(f"Pinned to IPFS: {result.cid}")
                return result.cid
            else:
                logger.warning("IPFS pinning returned no CID")
                return None

        except Exception as e:
            logger.warning(f"Failed to pin to IPFS: {e}")
            return None

    async def pin_item_to_ipfs(self, item_id: str) -> Optional[str]:
        """
        Pin an existing item to IPFS.

        Args:
            item_id: The item ID to pin

        Returns:
            IPFS CID if successful, None otherwise
        """
        item = await self.get_by_id(item_id)
        if not item:
            return None

        if item.ipfs_cid:
            logger.info(f"Item already pinned: {item.ipfs_cid}")
            return item.ipfs_cid

        ipfs_cid = await self._pin_to_ipfs(item.content, item.content_hash)
        if ipfs_cid:
            # Update database with CID
            await self.db.execute(
                "UPDATE saved_items SET ipfs_cid = ?, updated_at = ? WHERE id = ? AND agent_id = ?",
                # Backend-aware: datetime for PG, ISO string for
                # SQLite. Same rationale as ``save_item`` —
                # asyncpg refuses string TIMESTAMP binds, and SQLite
                # needs the ``T`` separator to keep lexicographic
                # ordering consistent with historical rows.
                (
                    ipfs_cid,
                    _now_for_backend(self.db, datetime.now(timezone.utc))[0],
                    item_id,
                    self.agent_id,
                )
            )
            await self.db.commit()

        return ipfs_cid

    async def update_item(
        self,
        item_id: str,
        name: Optional[str] = None,
        summary: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[SavedItem]:
        """Update an existing item's metadata."""
        existing = await self.get_by_id(item_id)
        if not existing:
            return None

        now = datetime.now(timezone.utc)
        updates = []
        params = []

        if name is not None:
            updates.append("name = ?")
            params.append(name)
        if summary is not None:
            updates.append("summary = ?")
            params.append(summary)
        if tags is not None:
            updates.append("tags = ?")
            params.append(json.dumps(tags))
        if metadata is not None:
            # Merge with existing metadata
            merged = {**existing.metadata, **metadata}
            updates.append("metadata = ?")
            params.append(json.dumps(merged))

        updates.append("updated_at = ?")
        # Backend-aware timestamp bind — see save_item.
        params.append(_now_for_backend(self.db, now)[0])
        params.append(item_id)
        params.append(self.agent_id)

        if updates:
            await self.db.execute(
                f"UPDATE saved_items SET {', '.join(updates)} WHERE id = ? AND agent_id = ?",
                tuple(params)
            )
            await self.db.commit()

        return await self.get_by_id(item_id)

    async def get_by_id(self, item_id: str) -> Optional[SavedItem]:
        """Get a saved item by ID."""
        row = await self.db.fetchone(
            """SELECT id, agent_id, item_type, name, summary, content, content_hash,
                      ipfs_cid, embedding, source_type, source_ref, schema_id,
                      tags, metadata, created_at, updated_at
               FROM saved_items WHERE id = ? AND agent_id = ?""",
            (item_id, self.agent_id)
        )
        return SavedItem.from_row(row) if row else None

    async def get_by_content_hash(self, content_hash: str) -> Optional[SavedItem]:
        """Get a saved item by content hash (for deduplication)."""
        row = await self.db.fetchone(
            """SELECT id, agent_id, item_type, name, summary, content, content_hash,
                      ipfs_cid, embedding, source_type, source_ref, schema_id,
                      tags, metadata, created_at, updated_at
               FROM saved_items WHERE content_hash = ? AND agent_id = ?""",
            (content_hash, self.agent_id)
        )
        return SavedItem.from_row(row) if row else None

    async def list_by_content_hash(self, content_hash: str) -> List[SavedItem]:
        """List saved items with the same content hash for this agent."""
        rows = await self.db.fetchall(
            """SELECT id, agent_id, item_type, name, summary, content, content_hash,
                      ipfs_cid, embedding, source_type, source_ref, schema_id,
                      tags, metadata, created_at, updated_at
               FROM saved_items WHERE content_hash = ? AND agent_id = ?""",
            (content_hash, self.agent_id)
        )
        return [SavedItem.from_row(row) for row in (rows or [])]

    async def _find_duplicate(
        self,
        content_hash: str,
        item_type: str,
        schema_id: Optional[str],
        source_type: Optional[str],
        source_ref: Optional[str],
    ) -> Optional[SavedItem]:
        """Find an existing row matching the same logical saved-item identity."""
        matches = await self.list_by_content_hash(content_hash)
        for item in matches:
            if (
                item.item_type == item_type
                and item.schema_id == schema_id
                and item.source_type == source_type
                and item.source_ref == source_ref
            ):
                return item
        return None

    async def list_items(
        self,
        item_type: Optional[str] = None,
        limit: int = 50
    ) -> List[SavedItem]:
        """List saved items, optionally filtered by type."""
        if item_type:
            rows = await self.db.fetchall(
                """SELECT id, agent_id, item_type, name, summary, content, content_hash,
                          ipfs_cid, embedding, source_type, source_ref, schema_id,
                          tags, metadata, created_at, updated_at
                   FROM saved_items
                   WHERE agent_id = ? AND item_type = ?
                   ORDER BY created_at DESC LIMIT ?""",
                (self.agent_id, item_type, limit)
            )
        else:
            rows = await self.db.fetchall(
                """SELECT id, agent_id, item_type, name, summary, content, content_hash,
                          ipfs_cid, embedding, source_type, source_ref, schema_id,
                          tags, metadata, created_at, updated_at
                   FROM saved_items
                   WHERE agent_id = ?
                   ORDER BY created_at DESC LIMIT ?""",
                (self.agent_id, limit)
            )
        return [SavedItem.from_row(row) for row in rows]

    async def _write_embedding_vec(
        self, item_id: str, embedding: List[float]
    ) -> None:
        """Write the embedding to the parallel ``embedding_vec`` column
        so it's discoverable by the vector backend.

        - On Postgres, formats the list as pgvector's text shape
          (``[v1,v2,…]``) and binds with a ``::vector`` cast.
        - On SQLite, packs to float32 little-endian bytes — same shape
          stored in the legacy ``embedding`` BLOB column, so the
          PurePythonBackend reads either column identically.
        - On any other dialect, treats the column as binary like
          SQLite.

        Errors here are non-fatal: the legacy ``embedding`` BYTEA / BLOB
        column is already written, so search degrades gracefully to
        the in-Python fallback path on the next ``search()`` call.
        """
        backend_type = getattr(self.db, "backend_type", None)
        try:
            if backend_type == "postgres":
                vec_text = "[" + ",".join(repr(float(v)) for v in embedding) + "]"
                await self.db.execute(
                    "UPDATE saved_items SET embedding_vec = ?::vector WHERE id = ?",
                    (vec_text, item_id),
                )
            else:
                await self.db.execute(
                    "UPDATE saved_items SET embedding_vec = ? WHERE id = ?",
                    (_serialize_embedding(embedding), item_id),
                )
        except Exception as e:
            # Most likely cause: the Phase-2 migration hasn't run yet
            # on this DB (the column doesn't exist). Log info and move
            # on — the legacy path keeps working unchanged.
            logger.info(
                "Could not write saved_items.embedding_vec for %s: %s. "
                "Vector search will use the in-Python fallback path "
                "until the next boot's migration runs.",
                item_id, e,
            )

    def _get_vector_session_factory(self):
        """Lazy-build a SQLAlchemy session factory pointed at the same
        DB as ``self.db``.

        Cached on the store so repeated searches reuse the same pool.
        Returns ``None`` if construction fails (e.g. ``from_pool``
        AsyncDatabase, in-memory SQLite, or any other unsupported
        backend shape) — callers then fall back to the legacy
        in-Python search path.
        """
        if getattr(self, "_sqla_factory", None) is not None:
            return self._sqla_factory
        # Don't keep retrying once we've decided it isn't available.
        if getattr(self, "_sqla_factory_unavailable", False):
            return None
        try:
            from kestrel_sovereign.storage.sqla import make_session_factory
            self._sqla_factory = make_session_factory(self.db)
            return self._sqla_factory
        except Exception as e:
            logger.info(
                "SQLAlchemy session factory unavailable for saved_items "
                "vector search (%s); falling back to in-Python search.",
                e,
            )
            self._sqla_factory_unavailable = True
            return None

    async def search(
        self,
        query: str,
        item_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Semantic search across saved items.

        Returns items sorted by relevance with scores.

        Phase-1 vector path: query embeddings are scored against stored
        embeddings via ``kestrel_sovereign.storage.vector``'s
        ``PurePythonBackend``. Same numpy-cosine math as the prior
        hand-rolled loop, but routed through the generic, shared
        primitives so a future Phase-2 PR can swap PG to pgvector by
        migrating the embedding column and flipping the backend choice
        — no further changes here. See kestrel-sovereign #1447 for the
        staged plan.

        Three fallbacks, in priority order:

        1. No embedding service available → return text-LIKE results
        2. SQLAlchemy session factory can't be built (e.g.
           ``AsyncDatabase`` from a bare pool) → fall back to legacy
           in-Python loop reading via ``self.db``
        3. Vector backend yields no rows (no stored embeddings yet) →
           fall back to text search

        Result shape matches the legacy code exactly:
        ``[{"item": <dict>, "score": <float>}, ...]``.
        """
        embedding_service = self._get_embedding_service()

        # Get query embedding
        query_embedding = None
        if embedding_service:
            try:
                query_embedding = await embedding_service.aembed(query)
            except Exception as e:
                logger.warning(f"Failed to embed query: {e}")

        if not query_embedding:
            # No embedding for the query → can't do semantic search at
            # all. Text-LIKE has the same fallback semantics the old
            # code had.
            return await self._text_search(query, item_type, limit)

        # Sovereign vector backend path. After Phase 2 of #1447
        # (BYTEA → vector(N) migration on PG) the dialect-dispatching
        # ``get_vector_backend`` is safe to use here: PG hits
        # ``PgVectorBackend`` for native pgvector kNN, SQLite stays on
        # ``PurePythonBackend`` numpy cosine.
        sf = self._get_vector_session_factory()
        if sf is not None:
            scored = await self._search_via_vector_backend(
                sf, query, query_embedding, item_type, limit
            )
            if scored is not None:
                return scored

        # No SQLA session factory available → fall back to the legacy
        # in-Python loop. Behaviourally identical.
        return await self._legacy_in_python_search(
            query, query_embedding, item_type, limit
        )

    async def _search_via_vector_backend(
        self,
        session_factory,
        query: str,
        query_embedding: List[float],
        item_type: Optional[str],
        limit: int,
    ) -> Optional[List[Dict[str, Any]]]:
        """Run kNN through ``kestrel_sovereign.storage.vector`` and
        materialize the top-k items via ``self.db``.

        Returns ``None`` (so the caller falls back to the legacy path)
        on any unexpected vector-backend failure — we'd rather degrade
        than 500.
        """
        try:
            # Local imports keep the legacy code paths free of the new
            # dependency surface when the vector backend isn't being
            # exercised.
            from kestrel_sovereign.storage.sqla import build_saved_item_spec
            from kestrel_sovereign.storage.vector import get_vector_backend

            # Pack query embedding into float32 little-endian bytes —
            # the shape both backends consume. The spec is built with
            # ``dimension=len(query_embedding)`` so any embedding-model
            # shape works (Ollama nomic-embed-text=768,
            # mxbai-embed-large=1024, OpenAI ada-002=1536, …) without a
            # hard-coded mismatch.
            packed = _serialize_embedding(query_embedding)
            spec = build_saved_item_spec(dimension=len(query_embedding))

            filter_kwargs: Dict[str, Any] = {"agent_id": self.agent_id}
            if item_type:
                filter_kwargs["item_type"] = item_type

            # Factory dispatch: PgVectorBackend on PG (post-Phase-2
            # migration), PurePythonBackend on SQLite.
            backend = get_vector_backend(session_factory, spec)
            top_k = await backend.knn(
                packed, k=limit, filter=filter_kwargs
            )
        except Exception as e:
            logger.warning(
                "Vector-backend search failed (%s); falling back to "
                "in-Python legacy path.", e,
            )
            return None

        if not top_k:
            # No stored embeddings for this agent's scope — fall back
            # to text search with the original query string (matching
            # the pre-#1447 behaviour exactly).
            return await self._text_search(query, item_type, limit)

        # Materialize: fetch full SavedItem rows by id, preserve the
        # backend's similarity ordering.
        scored: List[Dict[str, Any]] = []
        for item_id, score in top_k:
            row = await self.db.fetchone(
                """SELECT id, agent_id, item_type, name, summary, content,
                          content_hash, ipfs_cid, embedding, source_type,
                          source_ref, schema_id, tags, metadata,
                          created_at, updated_at
                   FROM saved_items WHERE id = ? AND agent_id = ?""",
                (item_id, self.agent_id),
            )
            if not row:
                # Row deleted between knn() and the lookup — skip it.
                continue
            scored.append({
                "item": SavedItem.from_row(row).to_dict(),
                "score": float(score),
            })

        return scored

    async def _legacy_in_python_search(
        self,
        query: str,
        query_embedding: List[float],
        item_type: Optional[str],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Fallback used when the SQLAlchemy session factory isn't
        available (in-memory SQLite, ``AsyncDatabase.from_pool``,
        etc.). Same shape as the pre-1447 search path."""

        # Get all items with embeddings
        if item_type:
            rows = await self.db.fetchall(
                """SELECT id, agent_id, item_type, name, summary, content, content_hash,
                          ipfs_cid, embedding, source_type, source_ref, schema_id,
                          tags, metadata, created_at, updated_at
                   FROM saved_items
                   WHERE agent_id = ? AND item_type = ? AND embedding IS NOT NULL""",
                (self.agent_id, item_type)
            )
        else:
            rows = await self.db.fetchall(
                """SELECT id, agent_id, item_type, name, summary, content, content_hash,
                          ipfs_cid, embedding, source_type, source_ref, schema_id,
                          tags, metadata, created_at, updated_at
                   FROM saved_items
                   WHERE agent_id = ? AND embedding IS NOT NULL""",
                (self.agent_id,)
            )

        if not rows:
            # Fall back to text search with the original query string —
            # matches the pre-#1447 behaviour exactly.
            return await self._text_search(query, item_type, limit)

        from kestrel_sovereign.llm.embedding_service import cosine_similarity

        scored = []
        for row in rows:
            item = SavedItem.from_row(row)
            if item.embedding:
                score = cosine_similarity(query_embedding, item.embedding)
                scored.append({
                    "item": item.to_dict(),
                    "score": score
                })

        # Sort by score descending
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    async def _text_search(
        self,
        query: str,
        item_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Fallback text search using LIKE."""
        query_lower = f"%{query.lower()}%"

        if item_type:
            rows = await self.db.fetchall(
                """SELECT id, agent_id, item_type, name, summary, content, content_hash,
                          ipfs_cid, embedding, source_type, source_ref, schema_id,
                          tags, metadata, created_at, updated_at
                   FROM saved_items
                   WHERE agent_id = ? AND item_type = ?
                     AND (LOWER(name) LIKE ? OR LOWER(summary) LIKE ? OR LOWER(content) LIKE ?)
                   ORDER BY created_at DESC LIMIT ?""",
                (self.agent_id, item_type, query_lower, query_lower, query_lower, limit)
            )
        else:
            rows = await self.db.fetchall(
                """SELECT id, agent_id, item_type, name, summary, content, content_hash,
                          ipfs_cid, embedding, source_type, source_ref, schema_id,
                          tags, metadata, created_at, updated_at
                   FROM saved_items
                   WHERE agent_id = ?
                     AND (LOWER(name) LIKE ? OR LOWER(summary) LIKE ? OR LOWER(content) LIKE ?)
                   ORDER BY created_at DESC LIMIT ?""",
                (self.agent_id, query_lower, query_lower, query_lower, limit)
            )

        return [
            {"item": SavedItem.from_row(row).to_dict(), "score": 1.0}
            for row in rows
        ]

    async def delete_item(self, item_id: str) -> bool:
        """Delete a saved item."""
        result = await self.db.execute(
            "DELETE FROM saved_items WHERE id = ? AND agent_id = ?",
            (item_id, self.agent_id)
        )
        await self.db.commit()
        return True

    async def get_item_count(self, item_type: Optional[str] = None) -> int:
        """Get count of saved items."""
        if item_type:
            row = await self.db.fetchone(
                "SELECT COUNT(*) FROM saved_items WHERE agent_id = ? AND item_type = ?",
                (self.agent_id, item_type)
            )
        else:
            row = await self.db.fetchone(
                "SELECT COUNT(*) FROM saved_items WHERE agent_id = ?",
                (self.agent_id,)
            )
        return row[0] if row else 0

    async def list_by_schema(
        self,
        schema_id: str,
        limit: int = 50
    ) -> List[SavedItem]:
        """List structured items by schema type."""
        rows = await self.db.fetchall(
            """SELECT id, agent_id, item_type, name, summary, content, content_hash,
                      ipfs_cid, embedding, source_type, source_ref, schema_id,
                      tags, metadata, created_at, updated_at
               FROM saved_items
               WHERE agent_id = ? AND schema_id = ?
               ORDER BY created_at DESC LIMIT ?""",
            (self.agent_id, schema_id, limit)
        )
        return [SavedItem.from_row(row) for row in rows]

    async def list_by_tag(
        self,
        tag: str,
        item_type: Optional[str] = None,
        limit: int = 50
    ) -> List[SavedItem]:
        """List items that have a specific tag."""
        # SQLite JSON search - look for tag in JSON array
        tag_pattern = f'%"{tag}"%'

        if item_type:
            rows = await self.db.fetchall(
                """SELECT id, agent_id, item_type, name, summary, content, content_hash,
                          ipfs_cid, embedding, source_type, source_ref, schema_id,
                          tags, metadata, created_at, updated_at
                   FROM saved_items
                   WHERE agent_id = ? AND item_type = ? AND tags LIKE ?
                   ORDER BY created_at DESC LIMIT ?""",
                (self.agent_id, item_type, tag_pattern, limit)
            )
        else:
            rows = await self.db.fetchall(
                """SELECT id, agent_id, item_type, name, summary, content, content_hash,
                          ipfs_cid, embedding, source_type, source_ref, schema_id,
                          tags, metadata, created_at, updated_at
                   FROM saved_items
                   WHERE agent_id = ? AND tags LIKE ?
                   ORDER BY created_at DESC LIMIT ?""",
                (self.agent_id, tag_pattern, limit)
            )
        return [SavedItem.from_row(row) for row in rows]

    async def get_all_tags(self) -> List[str]:
        """Get all unique tags across all items."""
        rows = await self.db.fetchall(
            "SELECT tags FROM saved_items WHERE agent_id = ?",
            (self.agent_id,)
        )

        all_tags = set()
        for row in rows:
            if row[0]:
                try:
                    tags = json.loads(row[0])
                    all_tags.update(tags)
                except json.JSONDecodeError:
                    pass

        return sorted(all_tags)

    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics about saved items."""
        total = await self.get_item_count()

        # Count by type
        type_counts = {}
        for item_type in SavedItemType:
            count = await self.get_item_count(item_type.value)
            if count > 0:
                type_counts[item_type.value] = count

        # Count by schema
        schema_rows = await self.db.fetchall(
            """SELECT schema_id, COUNT(*) FROM saved_items
               WHERE agent_id = ? AND schema_id IS NOT NULL
               GROUP BY schema_id""",
            (self.agent_id,)
        )
        schema_counts = {row[0]: row[1] for row in schema_rows}

        # Count items with IPFS
        ipfs_row = await self.db.fetchone(
            "SELECT COUNT(*) FROM saved_items WHERE agent_id = ? AND ipfs_cid IS NOT NULL",
            (self.agent_id,)
        )
        ipfs_count = ipfs_row[0] if ipfs_row else 0

        # Count items with embeddings
        embedding_row = await self.db.fetchone(
            "SELECT COUNT(*) FROM saved_items WHERE agent_id = ? AND embedding IS NOT NULL",
            (self.agent_id,)
        )
        embedding_count = embedding_row[0] if embedding_row else 0

        return {
            "total_items": total,
            "by_type": type_counts,
            "by_schema": schema_counts,
            "with_ipfs": ipfs_count,
            "with_embedding": embedding_count,
            "available_schemas": list_schemas()
        }

    async def save_structured_item(
        self,
        schema_id: str,
        content: Dict[str, Any],
        name: Optional[str] = None,
        summary: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        pin_to_ipfs: bool = False
    ) -> SavedItem:
        """
        Convenience method for saving structured items with validation.

        Args:
            schema_id: The schema type (recipe, contact, story, etc.)
            content: The structured content dict
            name: Item name (defaults to title/name from content)
            summary: Optional summary
            tags: Optional tags
            metadata: Optional metadata
            pin_to_ipfs: Whether to pin to IPFS

        Returns:
            The saved item

        Raises:
            ValueError: If schema validation fails
        """
        # Validate schema exists
        schema = get_schema(schema_id)
        if not schema:
            raise ValueError(f"Unknown schema: {schema_id}. Available: {[s['id'] for s in list_schemas()]}")

        # Auto-generate name from content if not provided
        if not name:
            name = content.get("title") or content.get("name") or f"{schema['name']} item"

        # Auto-generate summary if not provided
        if not summary:
            # Build summary from key fields
            summary_parts = []
            for field in schema["required_fields"][:2]:
                if field in content and content[field]:
                    val = content[field]
                    if isinstance(val, list):
                        val = ", ".join(str(v)[:50] for v in val[:3])
                    summary_parts.append(f"{field}: {str(val)[:100]}")
            summary = "; ".join(summary_parts) if summary_parts else name

        return await self.save_item(
            item_type=SavedItemType.STRUCTURED.value,
            name=name,
            content=json.dumps(content),
            summary=summary,
            schema_id=schema_id,
            tags=tags,
            metadata=metadata,
            source_type=SourceType.MANUAL.value,
            pin_to_ipfs=pin_to_ipfs,
            validate_schema=True
        )
