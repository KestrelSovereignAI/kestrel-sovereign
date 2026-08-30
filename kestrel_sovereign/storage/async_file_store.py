"""
Async File Store for Kestrel Storage.

Provides async file storage with content-addressable hashing and optional encryption.
Includes avatar storage with graph node relationships.
"""
import hashlib
import json
from datetime import datetime, UTC
from typing import Dict, Optional, Any

from .async_database import AsyncDatabase
from .async_graph_store import (
    AsyncGraphStore,
    GraphNode,
    reserve_provisional_agent_owner,
)
from .encryption import get_fernet, encrypt_bytes, decrypt_bytes
from kestrel_sovereign.kestrel_config.constants import MAX_FILE_SIZE


class AsyncFileStore:
    """Async file storage with content-addressable hashing."""

    def __init__(self, db: AsyncDatabase, agent_id: str = ""):
        self.db = db
        self.agent_id = agent_id
        self._fernet = get_fernet()

    def bind_agent(self, agent_id: str) -> None:
        """Bind ordinary file reads/writes to one tenant capability."""
        if not agent_id:
            raise ValueError("File ownership binding requires a non-empty agent_id")
        if self.agent_id and self.agent_id != agent_id:
            raise ValueError("File store is already bound to a different agent")
        self.agent_id = agent_id

    def _write_owner(self, metadata: Dict[str, Any]) -> str:
        declared = metadata.get("agent_id")
        if declared is not None and not isinstance(declared, str):
            raise ValueError("File metadata.agent_id must be a string")
        if self.agent_id and declared and declared != self.agent_id:
            raise ValueError("File owner does not match the bound agent")
        return self.agent_id or declared or ""

    def _require_matching_read_owner(self, agent_id: str) -> None:
        """Reject a per-call avatar read outside this bound capability."""
        if self.agent_id and self.agent_id != agent_id:
            raise ValueError("A bound file store cannot read another agent")

    @staticmethod
    def _avatar_node_id(
        agent_id: str, avatar_type: str, content_hash: str
    ) -> str:
        """Return a tenant/type namespace for content-addressed avatar nodes."""
        agent_digest = hashlib.sha256(agent_id.encode("utf-8")).hexdigest()
        type_digest = hashlib.sha256(avatar_type.encode("utf-8")).hexdigest()
        return f"avatar:{agent_digest}:{type_digest}:{content_hash}"

    def _reference_upsert_sql(self) -> str:
        if self.db.backend_type == "postgres":
            return """
                INSERT INTO file_owners
                    (content_hash, agent_id, original_name, metadata)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (content_hash, agent_id) DO UPDATE SET
                    original_name = EXCLUDED.original_name,
                    metadata = EXCLUDED.metadata
            """
        return """
            INSERT INTO file_owners
                (content_hash, agent_id, original_name, metadata)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(content_hash, agent_id) DO UPDATE SET
                original_name = excluded.original_name,
                metadata = excluded.metadata
        """

    # =========================================================================
    # Backend-Agnostic SQL Helpers
    # =========================================================================

    def _json_extract(self, column: str, path: str) -> str:
        """
        Return backend-appropriate JSON extraction SQL.

        SQLite: json_extract(column, '$.path')
        PostgreSQL: (column::jsonb)->>'path'

        Args:
            column: Column name containing JSON data
            path: JSON path (without $. prefix)

        Returns:
            SQL fragment for JSON extraction
        """
        if self.db.backend_type == "postgres":
            return f"({column}::jsonb)->>'{path}'"
        else:
            return f"json_extract({column}, '$.{path}')"

    def _order_by_created_at(self, column: str = "metadata", direction: str = "DESC") -> str:
        """
        Return backend-appropriate ordering by created_at from JSON metadata.

        SQLite: ORDER BY json_extract(metadata, '$.created_at') DESC
        PostgreSQL: ORDER BY (metadata::jsonb)->>'created_at' DESC

        Args:
            column: Column name containing JSON with created_at field
            direction: ASC or DESC

        Returns:
            SQL ORDER BY clause
        """
        return f"ORDER BY {self._json_extract(column, 'created_at')} {direction}"

    def _order_by_insertion(self, table_alias: str = "", direction: str = "DESC") -> str:
        """
        Return backend-appropriate ordering by insertion order.

        SQLite: ORDER BY rowid DESC
        PostgreSQL: Uses ctid as rough insertion order equivalent

        Note: For reliable ordering, prefer explicit timestamp columns.

        Args:
            table_alias: Optional table alias prefix (e.g., "e.")
            direction: ASC or DESC

        Returns:
            SQL ORDER BY clause
        """
        if self.db.backend_type == "postgres":
            # PostgreSQL doesn't have rowid, but for edges with composite PK
            # we should order by an explicit timestamp if available, or use ctid
            # ctid is a system column giving physical location (row insertion order)
            prefix = f"{table_alias}." if table_alias else ""
            return f"ORDER BY {prefix}ctid {direction}"
        else:
            prefix = f"{table_alias}." if table_alias else ""
            return f"ORDER BY {prefix}rowid {direction}"
    
    async def store_file(self, content: bytes, original_name: str,
                         metadata: Optional[Dict] = None) -> str:
        """Store a file and return its content hash."""
        if len(content) > MAX_FILE_SIZE:
            raise ValueError(
                f"File size ({len(content)} bytes) exceeds maximum "
                f"allowed size ({MAX_FILE_SIZE} bytes)"
            )
        content_hash = hashlib.sha256(content).hexdigest()
        meta = dict(metadata) if metadata else {}
        owner = self._write_owner(meta)
        to_store, was_encrypted = encrypt_bytes(content, self._fernet)
        if was_encrypted:
            meta["enc"] = True

        metadata_json = json.dumps(meta) if meta else None
        async with self.db.transaction():
            existing = await self.db.fetchone(
                "SELECT 1 FROM files WHERE content_hash = ?",
                (content_hash,),
            )
            if existing and owner:
                owner_rows = await self.db.fetchall(
                    "SELECT agent_id FROM file_owners WHERE content_hash = ?",
                    (content_hash,),
                )
                if not owner_rows:
                    raise ValueError("Cannot claim an unowned legacy file")

            await self.db.execute(
                "INSERT OR IGNORE INTO files "
                "(content_hash, original_name, content, metadata) "
                "VALUES (?, ?, ?, ?)",
                (content_hash, original_name, to_store, metadata_json),
            )
            if owner:
                await self.db.execute(
                    self._reference_upsert_sql(),
                    (content_hash, owner, original_name, metadata_json),
                )
        return content_hash
    
    async def retrieve_file(self, content_hash: str) -> Optional[bytes]:
        """Retrieve a file by its content hash.

        Returns:
            File content bytes, or None if file not found

        Raises:
            DecryptionError: If file is encrypted but decryption fails (wrong key)
        """
        if self.agent_id:
            row = await self.db.fetchone(
                "SELECT f.content, f.metadata FROM files f "
                "JOIN file_owners owners "
                "  ON owners.content_hash = f.content_hash "
                " AND owners.agent_id = ? "
                "WHERE f.content_hash = ?",
                (self.agent_id, content_hash),
            )
        else:
            row = await self.db.fetchone(
                "SELECT content, metadata FROM files WHERE content_hash = ?",
                (content_hash,),
            )
        if not row:
            return None

        content, metadata_json = row[0], row[1]
        meta = json.loads(metadata_json) if metadata_json else None
        return decrypt_bytes(content, self._fernet, meta)
    
    async def get_file_metadata(self, content_hash: str) -> Optional[Dict[str, Any]]:
        """Get file metadata by content hash."""
        if self.agent_id:
            row = await self.db.fetchone(
                "SELECT owners.metadata FROM file_owners owners "
                "JOIN files f ON f.content_hash = owners.content_hash "
                "WHERE owners.content_hash = ? AND owners.agent_id = ?",
                (content_hash, self.agent_id),
            )
        else:
            row = await self.db.fetchone(
                "SELECT metadata FROM files WHERE content_hash = ?",
                (content_hash,),
            )
        if row and row[0]:
            return json.loads(row[0])
        return None

    async def file_exists(self, content_hash: str) -> bool:
        """Return whether a file row exists for the given content hash."""
        if self.agent_id:
            row = await self.db.fetchone(
                "SELECT 1 FROM file_owners owners "
                "JOIN files f ON f.content_hash = owners.content_hash "
                "WHERE owners.content_hash = ? AND owners.agent_id = ?",
                (content_hash, self.agent_id),
            )
        else:
            row = await self.db.fetchone(
                "SELECT 1 FROM files WHERE content_hash = ?",
                (content_hash,),
            )
        return row is not None

    # =========================================================================
    # Avatar Storage Methods
    # =========================================================================

    async def store_avatar(
        self,
        image_data: bytes,
        agent_id: str,
        avatar_type: str = "primary",
        source_url: Optional[str] = None
    ) -> str:
        """
        Store avatar image as part of agent identity.

        The avatar hash is stored:
        1. As a file in content-addressable storage (encrypted)
        2. As a graph node linked to the agent
        3. As a property on the agent node itself (avatar_hash)

        This ensures the avatar is intrinsic to the agent's identity and
        travels with sovereignty exports.

        Args:
            image_data: Raw image bytes
            agent_id: Agent DID
            avatar_type: "primary", "selfie", "thumbnail", etc.
            source_url: Original URL (e.g., Replicate) for reference

        Returns:
            content_hash (SHA256)
        """
        metadata = {
            "type": "avatar",
            "avatar_type": avatar_type,
            "agent_id": agent_id,
            "source_url": source_url,
            "mime_type": "image/jpeg",
            "created_at": datetime.now(UTC).isoformat()
        }

        async with self.db.transaction():
            content_hash = await self.store_file(
                image_data, f"avatar_{avatar_type}.jpg", metadata
            )
            avatar_node_id = self._avatar_node_id(
                agent_id, avatar_type, content_hash
            )
            graph_metadata = {**metadata, "hash": content_hash}

            # Use the canonical graph writer so the node and edge receive
            # durable tenant-ownership witnesses together with their graph
            # rows. The graph id is tenant/type namespaced because identical
            # bytes do not imply shared avatar metadata (#2649).
            graph = AsyncGraphStore(self.db, agent_id=agent_id)
            # Some bootstrap/test callers store an avatar before the physical
            # agent root is inserted. The DID is still the canonical self-owner,
            # so reserve that witness; later root creation uses the same owner.
            # Lock both possibly-absent IDs first so source deletion and another
            # bootstrap writer observe the same canonical serialization order.
            await reserve_provisional_agent_owner(
                self.db,
                agent_id,
                additional_graph_node_ids=[avatar_node_id],
            )
            await graph.add_node(
                GraphNode(
                    node_id=avatar_node_id,
                    node_type="avatar",
                    label=f"Avatar ({avatar_type})",
                    properties=graph_metadata,
                )
            )
            await graph.add_edge(
                agent_id,
                avatar_node_id,
                "has_avatar",
                {"avatar_type": avatar_type},
            )

            # Update agent node's identity with avatar_hash (like constitution_hash)
            # This makes the avatar intrinsic to the agent's identity.
            if avatar_type == "primary":
                agent_node = await graph.get_node(agent_id)
                if agent_node is not None:
                    agent_node.properties["avatar_hash"] = content_hash
                    await graph.add_node(agent_node)

        return content_hash

    async def get_agent_avatar(
        self, agent_id: str, avatar_type: str = "primary"
    ) -> Optional[bytes]:
        """
        Retrieve agent's avatar by type.

        Args:
            agent_id: Agent DID
            avatar_type: "primary", "selfie", "thumbnail", etc.

        Returns:
            Avatar image bytes or None if not found

        Raises:
            DecryptionError: If avatar is encrypted but decryption fails (wrong key)
        """
        self._require_matching_read_owner(agent_id)
        # Build backend-agnostic query
        avatar_type_filter = self._json_extract("e.properties", "avatar_type")
        avatar_hash = self._json_extract("avatar.properties", "hash")
        order_by = self._order_by_created_at("avatar.properties")

        # Find avatar via graph edge relationship
        row = await self.db.fetchone(
            f"""SELECT f.content, f.metadata FROM graph_edges e
               JOIN graph_nodes avatar
                 ON avatar.node_id = e.target_id
                AND avatar.node_type = 'avatar'
               JOIN graph_node_owners avatar_owner
                 ON avatar_owner.node_id = avatar.node_id
                AND avatar_owner.agent_id = ?
               JOIN files f
                 ON f.content_hash = COALESCE({avatar_hash}, e.target_id)
               JOIN file_owners fo
                 ON fo.content_hash = f.content_hash AND fo.agent_id = ?
               JOIN graph_edge_owners eo
                 ON eo.source_id = e.source_id
                AND eo.target_id = e.target_id
                AND eo.label = e.label
                AND eo.agent_id = ?
               WHERE e.source_id = ? AND e.label = 'has_avatar'
               AND {avatar_type_filter} = ?
               {order_by}
               LIMIT 1""",
            (agent_id, agent_id, agent_id, agent_id, avatar_type)
        )

        if not row:
            return None

        content, metadata_json = row[0], row[1]
        meta = json.loads(metadata_json) if metadata_json else None
        return decrypt_bytes(content, self._fernet, meta)

    async def get_agent_avatar_hash(
        self, agent_id: str, avatar_type: str = "primary"
    ) -> Optional[str]:
        """
        Get avatar content hash for URL generation.

        For primary avatars, reads from agent node's avatar_hash property
        (the authoritative source, like constitution_hash). For other types,
        looks up via graph edges.

        Args:
            agent_id: Agent DID
            avatar_type: "primary", "selfie", "thumbnail", etc.

        Returns:
            Content hash (SHA256) or None if not found
        """
        self._require_matching_read_owner(agent_id)
        # For primary avatar, check agent node property first (authoritative)
        if avatar_type == "primary":
            agent_node = await AsyncGraphStore(
                self.db, agent_id=agent_id
            ).get_node(agent_id)
            if agent_node is not None:
                avatar_hash = agent_node.properties.get("avatar_hash")
                if avatar_hash and await AsyncFileStore(
                    self.db, agent_id=agent_id
                ).file_exists(avatar_hash):
                    return avatar_hash

        # Fallback to graph edge lookup (for non-primary or legacy data)
        # Build backend-agnostic query
        avatar_type_filter = self._json_extract("e.properties", "avatar_type")
        avatar_hash = self._json_extract("avatar.properties", "hash")
        order_by = self._order_by_insertion("e")

        row = await self.db.fetchone(
            f"""SELECT COALESCE({avatar_hash}, e.target_id)
               FROM graph_edges e
               JOIN graph_nodes avatar
                 ON avatar.node_id = e.target_id
                AND avatar.node_type = 'avatar'
               JOIN graph_node_owners avatar_owner
                 ON avatar_owner.node_id = avatar.node_id
                AND avatar_owner.agent_id = ?
               JOIN file_owners fo
                 ON fo.content_hash = COALESCE({avatar_hash}, e.target_id)
                AND fo.agent_id = ?
               JOIN graph_edge_owners eo
                 ON eo.source_id = e.source_id
                AND eo.target_id = e.target_id
                AND eo.label = e.label
                AND eo.agent_id = ?
               WHERE e.source_id = ? AND e.label = 'has_avatar'
               AND {avatar_type_filter} = ?
               {order_by} LIMIT 1""",
            (agent_id, agent_id, agent_id, agent_id, avatar_type)
        )
        return row[0] if row else None
