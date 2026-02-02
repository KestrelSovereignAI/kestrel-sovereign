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
from .encryption import get_fernet, encrypt_bytes, decrypt_bytes, DecryptionError


class AsyncFileStore:
    """Async file storage with content-addressable hashing."""

    def __init__(self, db: AsyncDatabase):
        self.db = db
        self._fernet = get_fernet()

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
        content_hash = hashlib.sha256(content).hexdigest()
        meta = dict(metadata) if metadata else {}
        to_store, was_encrypted = encrypt_bytes(content, self._fernet)
        if was_encrypted:
            meta["enc"] = True

        await self.db.execute_commit(
            "INSERT OR IGNORE INTO files (content_hash, original_name, content, metadata) VALUES (?, ?, ?, ?)",
            (content_hash, original_name, to_store, json.dumps(meta) if meta else None)
        )
        return content_hash
    
    async def retrieve_file(self, content_hash: str) -> Optional[bytes]:
        """Retrieve a file by its content hash.

        Returns:
            File content bytes, or None if file not found

        Raises:
            DecryptionError: If file is encrypted but decryption fails (wrong key)
        """
        row = await self.db.fetchone(
            "SELECT content, metadata FROM files WHERE content_hash = ?",
            (content_hash,)
        )
        if not row:
            return None

        content, metadata_json = row[0], row[1]
        meta = json.loads(metadata_json) if metadata_json else None
        return decrypt_bytes(content, self._fernet, meta)
    
    async def get_file_metadata(self, content_hash: str) -> Optional[Dict[str, Any]]:
        """Get file metadata by content hash."""
        row = await self.db.fetchone(
            "SELECT metadata FROM files WHERE content_hash = ?",
            (content_hash,)
        )
        if row and row[0]:
            return json.loads(row[0])
        return None

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

        content_hash = await self.store_file(
            image_data, f"avatar_{avatar_type}.jpg", metadata
        )

        # Create graph node for avatar
        await self.db.execute_commit(
            """INSERT OR REPLACE INTO graph_nodes (node_id, node_type, label, properties)
               VALUES (?, 'avatar', ?, ?)""",
            (content_hash, f"Avatar ({avatar_type})", json.dumps(metadata))
        )

        # Create edge: agent --has_avatar--> avatar_node
        await self.db.execute_commit(
            """INSERT OR REPLACE INTO graph_edges (source_id, target_id, label, properties)
               VALUES (?, ?, 'has_avatar', ?)""",
            (agent_id, content_hash, json.dumps({"avatar_type": avatar_type}))
        )

        # Update agent node's identity with avatar_hash (like constitution_hash)
        # This makes the avatar intrinsic to the agent's identity
        if avatar_type == "primary":
            row = await self.db.fetchone(
                "SELECT properties FROM graph_nodes WHERE node_id = ?",
                (agent_id,)
            )
            if row and row[0]:
                properties = json.loads(row[0])
                properties["avatar_hash"] = content_hash
                await self.db.execute_commit(
                    "UPDATE graph_nodes SET properties = ? WHERE node_id = ?",
                    (json.dumps(properties), agent_id)
                )

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
        # Build backend-agnostic query
        avatar_type_filter = self._json_extract("e.properties", "avatar_type")
        order_by = self._order_by_created_at("f.metadata")

        # Find avatar via graph edge relationship
        row = await self.db.fetchone(
            f"""SELECT f.content, f.metadata FROM files f
               JOIN graph_edges e ON f.content_hash = e.target_id
               WHERE e.source_id = ? AND e.label = 'has_avatar'
               AND {avatar_type_filter} = ?
               {order_by}
               LIMIT 1""",
            (agent_id, avatar_type)
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
        # For primary avatar, check agent node property first (authoritative)
        if avatar_type == "primary":
            row = await self.db.fetchone(
                "SELECT properties FROM graph_nodes WHERE node_id = ?",
                (agent_id,)
            )
            if row and row[0]:
                properties = json.loads(row[0])
                avatar_hash = properties.get("avatar_hash")
                if avatar_hash:
                    return avatar_hash

        # Fallback to graph edge lookup (for non-primary or legacy data)
        # Build backend-agnostic query
        avatar_type_filter = self._json_extract("e.properties", "avatar_type")
        order_by = self._order_by_insertion("e")

        row = await self.db.fetchone(
            f"""SELECT e.target_id FROM graph_edges e
               WHERE e.source_id = ? AND e.label = 'has_avatar'
               AND {avatar_type_filter} = ?
               {order_by} LIMIT 1""",
            (agent_id, avatar_type)
        )
        return row[0] if row else None
