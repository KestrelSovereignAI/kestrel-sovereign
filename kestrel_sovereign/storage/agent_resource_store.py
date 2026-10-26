"""Private agent identity resources.

SOUL documents are agent-owned private identity state. This store keeps their
canonical body encrypted at rest while exposing only hash/pointer metadata for
public identity surfaces.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, Mapping, Optional

from kestrel_sovereign.execution_custody import is_execution_control_error
from kestrel_sovereign.security.encryption import (
    decrypt_string_fernet,
    encrypt_string_fernet,
    get_agent_fernet,
)

from .async_database import AsyncDatabase
from .db.interface import TransactionError


SOUL_MARKDOWN_RESOURCE_TYPE = "agent.soul.markdown"

# Provenance fields the store writes itself. A caller's extra provenance (the
# reason and actor of a SOUL revision, #3779) may add fields but never replace
# these. ``previous_version`` and ``previous_content_hash`` name the version a
# new current version replaced, read under the version lock, so no caller can
# claim a predecessor it inspected before another writer replaced it.
_RESERVED_PROVENANCE_FIELDS = frozenset(
    {
        "created_by",
        "source",
        "created_at",
        "signature",
        "previous_version",
        "previous_content_hash",
    }
)


def check_version_request(
    resource_type: str,
    content: Optional[str],
    provenance: Optional[Mapping[str, Any]] = None,
) -> None:
    """Refuse a resource-version request :meth:`AgentResourceStore.create_version`
    would refuse, before any write is attempted.

    Raises:
        ValueError: no ``resource_type`` or ``content``, or ``provenance``
            sets a field the store writes itself.
    """
    if not resource_type:
        raise ValueError("resource_type is required")
    if content is None:
        raise ValueError("content is required")
    reserved = _RESERVED_PROVENANCE_FIELDS.intersection(provenance or {})
    if reserved:
        raise ValueError(
            "provenance may not set store-written fields: "
            + ", ".join(sorted(reserved))
        )


@dataclass(frozen=True)
class ExpectedCurrentVersion:
    """The current version a new version's body was derived from (#3813).

    A writer that reads the current version and derives its new body from it
    (a rename of the SOUL) passes this, and the store refuses the write if a
    different version is current by the time it runs. Without it, the write
    would replace a version made current in between, such as a Sovereign's
    revision, with a body derived from the older text. ``version`` is
    ``None`` when no version was current.
    """

    version: Optional[int]


class ResourceVersionConflictError(RuntimeError):
    """The current version is no longer the one a write was derived from.

    Nothing was written. Read the current version again and derive the write
    from it.
    """

    def __init__(
        self, resource_type: str, *, expected: Optional[int], actual: Optional[int]
    ):
        self.resource_type = resource_type
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"the current {resource_type} version is {_describe_version(actual)}, "
            f"not {_describe_version(expected)} as when this write read it; "
            "nothing was written"
        )


def _describe_version(version: Optional[int]) -> str:
    return f"v{version}" if version is not None else "none"


def _check_expected_current(
    resource_type: str,
    existing: Optional[Mapping[str, Any]],
    expected_current: ExpectedCurrentVersion,
) -> None:
    """Raise unless ``existing`` is the current version ``expected_current`` names.

    ``existing`` is the row :meth:`AgentResourceStore._current_or_latest`
    read under the version lock.
    """
    actual = existing["version"] if existing and existing["is_current"] else None
    if actual != expected_current.version:
        raise ResourceVersionConflictError(
            resource_type, expected=expected_current.version, actual=actual
        )


@asynccontextmanager
async def surfacing_version_conflicts() -> AsyncIterator[None]:
    """Raise a version conflict as itself from a transaction that wrapped it.

    An owning transaction reports whatever its block raised as a
    :class:`TransactionError` caused by it. A refused write is an outcome its
    caller acts on, not a storage fault, so enter this outside the
    transaction the refusal is raised in. A wrapper carrying execution-control
    evidence is raised as it is.
    """
    try:
        yield
    except TransactionError as error:
        cause = error.__cause__
        while isinstance(cause, TransactionError):
            cause = cause.__cause__
        if (
            not isinstance(cause, ResourceVersionConflictError)
            or is_execution_control_error(error)
        ):
            raise
        # With its own cause: a cancellation that arrived while the refused
        # write rolled back is carried there, for the caller to propagate.
        raise cause from cause.__cause__


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
        provenance: Optional[Mapping[str, Any]] = None,
        expected_current: Optional[ExpectedCurrentVersion] = None,
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
            provenance: Additional provenance recorded on this version, such
                as why it was made and by whom. It may not set a field the
                store writes itself: ``created_by``, ``source``,
                ``created_at``, ``signature``, ``previous_version`` or
                ``previous_content_hash``.
            expected_current: The current version ``content`` was derived
                from. If a different version is current when the write runs,
                :class:`ResourceVersionConflictError` is raised and nothing is
                written. ``None`` writes over whatever is current.

        A version made current records the version it replaced, if one was
        current, as ``previous_version`` and ``previous_content_hash``.
        """
        check_version_request(resource_type, content, provenance)

        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        encrypted, was_encrypted = encrypt_string_fernet(content, self._fernet)
        if not was_encrypted:
            raise RuntimeError("agent resource encryption was not applied")

        # One atomic unit (#3779): allocating the version, clearing the
        # previous selection and inserting the replacement commit together or
        # not at all. Separately committed, a failed insert left the previous
        # version deselected and no version current, and the next boot then
        # promoted SOUL.md as a fresh seed with none of the failed write's
        # provenance. A caller's open transaction is joined, so a caller that
        # publishes more alongside the version (the SOUL's graph reference)
        # rolls back with it. Immediate: this unit reads before it writes.
        async with surfacing_version_conflicts(), self.db.transaction(immediate=True):
            await self._lock_versions(resource_type)
            existing = await self._current_or_latest(resource_type)
            if expected_current is not None:
                # Under the version lock: no other writer can make a version
                # current between this check and the insert (#3813).
                _check_expected_current(resource_type, existing, expected_current)
            resource_id = existing["resource_id"] if existing else str(uuid.uuid4())
            latest_version = await self._latest_version(resource_type)
            version = latest_version + 1

            now = datetime.now(timezone.utc)
            recorded_provenance = {
                **(provenance or {}),
                "created_by": created_by,
                "source": source,
                "created_at": now.isoformat(),
            }
            if signature:
                recorded_provenance["signature"] = signature
            if make_current and existing and existing["is_current"]:
                recorded_provenance["previous_version"] = existing["version"]
                recorded_provenance["previous_content_hash"] = existing["content_hash"]

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
                    encrypted.encode("utf-8"),
                    content_hash,
                    len(content.encode("utf-8")),
                    "agent-fernet:v1",
                    json.dumps(recorded_provenance, sort_keys=True),
                    json.dumps(safe_public, sort_keys=True),
                    json.dumps(anchoring_metadata or {}, sort_keys=True),
                    now.replace(tzinfo=None),
                ),
            )
            resource = await self.get_version(resource_type, version=version)
            if resource is None:
                raise RuntimeError("created resource version could not be loaded")
        return resource

    async def require_current_version(
        self,
        expected_current: ExpectedCurrentVersion,
        resource_type: str = SOUL_MARKDOWN_RESOURCE_TYPE,
    ) -> None:
        """Refuse as :meth:`create_version` would, and write nothing.

        For a writer that read the current version and found nothing to
        derive from it, but must not act on that read once another version is
        current: a rename whose SOUL names nobody still renames the agent
        everywhere else (#3813). Checked under the version lock and in the
        immediate transaction :meth:`create_version` writes under.

        Raises:
            ResourceVersionConflictError: ``expected_current`` is not the
                current version.
        """
        async with surfacing_version_conflicts(), self.db.transaction(immediate=True):
            await self._lock_versions(resource_type)
            _check_expected_current(
                resource_type,
                await self._current_or_latest(resource_type),
                expected_current,
            )

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
        provenance: Optional[Mapping[str, Any]] = None,
        expected_current: Optional[ExpectedCurrentVersion] = None,
    ) -> AgentResourceVersion:
        """Promote a local SOUL.md seed/cache body into the canonical store.

        ``expected_current`` is as for :meth:`create_version`.
        """
        return await self.create_version(
            SOUL_MARKDOWN_RESOURCE_TYPE,
            content,
            created_by=created_by or self.agent_id,
            source=source,
            make_current=True,
            provenance=provenance,
            expected_current=expected_current,
        )

    async def _lock_versions(self, resource_type: str) -> None:
        """Serialize version allocation for this agent's resource on PostgreSQL.

        Call inside the allocating transaction. SQLite already serializes the
        unit through its immediate transaction's writer slot.
        """
        if self.db.backend_type != "postgres":
            return
        await self.db.fetchone(
            "SELECT pg_advisory_xact_lock(hashtextextended(?, 0))",
            (f"agent-identity-resource:{self.agent_id}:{resource_type}",),
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
            SELECT resource_id, version, content_hash, is_current
            FROM agent_identity_resources
            WHERE agent_id = ? AND resource_type = ?
            ORDER BY is_current DESC, version DESC
            LIMIT 1
            """,
            (self.agent_id, resource_type),
        )
        if not row:
            return None
        return {
            "resource_id": row[0],
            "version": int(row[1]),
            "content_hash": row[2],
            "is_current": bool(row[3]),
        }

    def _row_to_resource(self, row: tuple[Any, ...]) -> AgentResourceVersion:
        ciphertext = row[6]
        # BYTEA on PostgreSQL and BLOB on new SQLite rows. Existing SQLite
        # databases may still contain the legacy text representation.
        if isinstance(ciphertext, (bytes, bytearray, memoryview)):
            ciphertext = bytes(ciphertext).decode("utf-8")
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
