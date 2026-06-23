"""Private agent identity resources.

SOUL documents are agent-owned private identity state. This store keeps their
canonical body encrypted at rest while exposing only hash/pointer metadata for
public identity surfaces.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from kestrel_sovereign.security.encryption import (
    decrypt_string_fernet,
    encrypt_string_fernet,
    get_agent_fernet,
)

from .async_database import AsyncDatabase


SOUL_MARKDOWN_RESOURCE_TYPE = "agent.soul.markdown"


@dataclass(frozen=True)
class AgentResourceVersion:
    """A decrypted private resource version attached to an agent identity."""

    id: str
    agent_id: str
    resource_id: str
    resource_type: str
    version: int
    is_current: bool
    content: str
    content_hash: str
    content_bytes: int
    encryption: str
    provenance: Dict[str, Any]
    public_metadata: Dict[str, Any]
    anchoring_metadata: Dict[str, Any]
    created_at: Any


class AgentResourceStore:
    """Encrypted store for private resources owned by one agent."""

    def __init__(self, db: AsyncDatabase, agent_id: str):
        if not agent_id:
            raise ValueError("agent_id is required for AgentResourceStore")
        self.db = db
        self.agent_id = agent_id
        self._fernet = get_agent_fernet(agent_id)

    async def create_version(
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
        """Create a new encrypted resource version.

        Args:
            resource_type: Stable type identifier, e.g. ``agent.soul.markdown``.
            content: Private plaintext body. Stored only as ciphertext.
            created_by: DID/tool/operator that created or promoted this version.
            source: Provenance source such as ``bootstrap`` or ``SOUL.md``.
            make_current: Whether this version becomes the selected version.
            signature: Optional detached signature/proof metadata.
            anchoring_metadata: Optional public anchoring/export pointers.
            public_metadata: Additional safe metadata. The body is never copied.
        """
        if not resource_type:
            raise ValueError("resource_type is required")
        if content is None:
            raise ValueError("content is required")

        existing = await self._current_or_latest(resource_type)
        resource_id = existing["resource_id"] if existing else str(uuid.uuid4())
        latest_version = await self._latest_version(resource_type)
        version = latest_version + 1

        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        encrypted, was_encrypted = encrypt_string_fernet(content, self._fernet)
        if not was_encrypted:
            raise RuntimeError("agent resource encryption was not applied")

        now = datetime.now(timezone.utc).isoformat()
        provenance = {
            "created_by": created_by,
            "source": source,
            "created_at": now,
        }
        if signature:
            provenance["signature"] = signature

        safe_public = {
            "resource_id": resource_id,
            "resource_type": resource_type,
            "version": version,
            "content_hash": content_hash,
            "content_bytes": len(content.encode("utf-8")),
            "current": bool(make_current),
        }
        if public_metadata:
            safe_public.update(public_metadata)
        safe_public.pop("content", None)
        safe_public.pop("body", None)
        safe_public.pop("plaintext", None)

        if make_current:
            await self.db.execute(
                """
                UPDATE agent_identity_resources
                SET is_current = 0
                WHERE agent_id = ? AND resource_type = ?
                """,
                (self.agent_id, resource_type),
            )

        version_id = str(uuid.uuid4())
        await self.db.execute(
            """
            INSERT INTO agent_identity_resources
                (id, agent_id, resource_id, resource_type, version, is_current,
                 content_ciphertext, content_hash, content_bytes, encryption,
                 provenance, public_metadata, anchoring_metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                version_id,
                self.agent_id,
                resource_id,
                resource_type,
                version,
                1 if make_current else 0,
                encrypted,
                content_hash,
                len(content.encode("utf-8")),
                "agent-fernet:v1",
                json.dumps(provenance, sort_keys=True),
                json.dumps(safe_public, sort_keys=True),
                json.dumps(anchoring_metadata or {}, sort_keys=True),
                now,
            ),
        )
        resource = await self.get_version(resource_type, version=version)
        if resource is None:
            raise RuntimeError("created resource version could not be loaded")
        return resource

    async def get_current(
        self, resource_type: str = SOUL_MARKDOWN_RESOURCE_TYPE
    ) -> Optional[AgentResourceVersion]:
        """Return the selected current version for a resource type."""
        row = await self.db.fetchone(
            """
            SELECT id, agent_id, resource_id, resource_type, version, is_current,
                   content_ciphertext, content_hash, content_bytes, encryption,
                   provenance, public_metadata, anchoring_metadata, created_at
            FROM agent_identity_resources
            WHERE agent_id = ? AND resource_type = ? AND is_current = 1
            ORDER BY version DESC
            LIMIT 1
            """,
            (self.agent_id, resource_type),
        )
        if not row:
            return None
        return self._row_to_resource(row)

    async def get_version(
        self,
        resource_type: str = SOUL_MARKDOWN_RESOURCE_TYPE,
        *,
        version: int,
    ) -> Optional[AgentResourceVersion]:
        row = await self.db.fetchone(
            """
            SELECT id, agent_id, resource_id, resource_type, version, is_current,
                   content_ciphertext, content_hash, content_bytes, encryption,
                   provenance, public_metadata, anchoring_metadata, created_at
            FROM agent_identity_resources
            WHERE agent_id = ? AND resource_type = ? AND version = ?
            """,
            (self.agent_id, resource_type, version),
        )
        if not row:
            return None
        return self._row_to_resource(row)

    async def get_public_metadata(
        self,
        resource_type: str = SOUL_MARKDOWN_RESOURCE_TYPE,
    ) -> Optional[Dict[str, Any]]:
        """Return body-free metadata suitable for public identity surfaces."""
        current = await self.get_current(resource_type)
        if not current:
            return None
        metadata = dict(current.public_metadata)
        metadata.update(
            {
                "resource_id": current.resource_id,
                "resource_type": current.resource_type,
                "version": current.version,
                "content_hash": current.content_hash,
                "content_bytes": current.content_bytes,
                "anchoring": current.anchoring_metadata,
            }
        )
        metadata.pop("content", None)
        metadata.pop("body", None)
        metadata.pop("plaintext", None)
        return metadata

    async def promote_soul_seed(
        self,
        content: str,
        *,
        created_by: Optional[str] = None,
        source: str = "agent_data/SOUL.md",
    ) -> AgentResourceVersion:
        """Promote a local SOUL.md seed/cache body into the canonical store."""
        return await self.create_version(
            SOUL_MARKDOWN_RESOURCE_TYPE,
            content,
            created_by=created_by or self.agent_id,
            source=source,
            make_current=True,
        )

    async def _latest_version(self, resource_type: str) -> int:
        row = await self.db.fetchone(
            """
            SELECT MAX(version) FROM agent_identity_resources
            WHERE agent_id = ? AND resource_type = ?
            """,
            (self.agent_id, resource_type),
        )
        return int(row[0] or 0) if row else 0

    async def _current_or_latest(self, resource_type: str) -> Optional[Dict[str, Any]]:
        row = await self.db.fetchone(
            """
            SELECT resource_id FROM agent_identity_resources
            WHERE agent_id = ? AND resource_type = ?
            ORDER BY is_current DESC, version DESC
            LIMIT 1
            """,
            (self.agent_id, resource_type),
        )
        if not row:
            return None
        return {"resource_id": row[0]}

    def _row_to_resource(self, row: tuple[Any, ...]) -> AgentResourceVersion:
        ciphertext = row[6]
        content = decrypt_string_fernet(ciphertext, {"enc": True}, self._fernet)
        return AgentResourceVersion(
            id=row[0],
            agent_id=row[1],
            resource_id=row[2],
            resource_type=row[3],
            version=int(row[4]),
            is_current=bool(row[5]),
            content=content,
            content_hash=row[7],
            content_bytes=int(row[8]),
            encryption=row[9],
            provenance=json.loads(row[10]) if row[10] else {},
            public_metadata=json.loads(row[11]) if row[11] else {},
            anchoring_metadata=json.loads(row[12]) if row[12] else {},
            created_at=row[13],
        )
