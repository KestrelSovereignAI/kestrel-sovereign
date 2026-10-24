"""Transactional encrypted artifact persistence, independent of a producer.

This is an internal repository, not an authorization boundary. The host service
must hold its current authority/privacy operation guard around every call. A
ResourceScope is an owner selector, never proof of access. No HTTP route exposes
this repository directly. All publication bytes and metadata use AsyncDatabase.
Use the same database as KeyRotationService. Master-key rotation rewraps the
stable envelope root; it is not recovery from compromise of that root.
Minimal keyed publication tombstones persist to prevent erased-key replay.
Their cumulative storage is deliberately separate from live publication quotas;
the host must account for this growth when admitting ongoing publication work.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import asyncio
import base64
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import hashlib
import hmac
import json
import secrets
from typing import Callable
from uuid import uuid4

from kestrel_sdk.artifacts import (
    ArtifactCommit, ArtifactFile, ArtifactLink, ArtifactPage,
    ArtifactRef, ArtifactRevision, PublicationReceipt,
)
from kestrel_sdk.resources import ResourceConflict, ResourceError, ResourceNotFound, ResourceOutcomeUnknown, ResourceScope
from kestrel_sdk.security.encryption import get_fernet
from kestrel_sdk.security.aead import AEADCipher
from kestrel_sdk.security.exceptions import DecryptionError
from kestrel_sdk.storage.database.interface import TransactionError
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.execution_custody import execution_commit_outcome


class ArtifactIntegrityError(ResourceError):
    """Stored bytes or identity failed verification; never report saved success."""


class ArtifactQuotaExceeded(ResourceError):
    """An owner exceeded configured staging or retained-content bounds."""


@dataclass(frozen=True)
class ArtifactLimits:
    max_file_bytes: int = 16 * 1024 * 1024
    max_owner_bytes: int = 128 * 1024 * 1024
    max_revisions: int = 2048
    max_publications: int = 4096
    max_blobs: int = 4096
    max_links: int = 4096
    max_staging_files: int = 4096
    max_stored_bytes: int = 256 * 1024 * 1024
    max_metadata_bytes: int = 16 * 1024 * 1024
    staging_seconds: int = 3600

    def __post_init__(self):
        for value in vars(self).values():
            if type(value) is not int or value <= 0:
                raise ValueError("artifact limits must be positive integers")
        if self.max_file_bytes > self.max_owner_bytes:
            raise ValueError("file limit exceeds owner limit")


_SCHEMA = (
    """CREATE TABLE IF NOT EXISTS artifact_keys (
        key_id TEXT PRIMARY KEY, wrapped_key BLOB NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS artifact_owners (
        owner TEXT PRIMARY KEY, generation BIGINT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS artifact_blobs (
        owner TEXT NOT NULL, blob_key TEXT NOT NULL, size_bytes BIGINT NOT NULL,
        content BLOB NOT NULL,
        PRIMARY KEY (owner, blob_key))""",
    """CREATE TABLE IF NOT EXISTS artifact_staging (
        owner TEXT NOT NULL, publication_key TEXT NOT NULL, path_key TEXT NOT NULL,
        blob_key TEXT NOT NULL, file_key TEXT NOT NULL, expires_at TEXT NOT NULL,
        PRIMARY KEY (owner, publication_key, path_key))""",
    """CREATE TABLE IF NOT EXISTS artifact_heads (
        owner TEXT NOT NULL, artifact_id TEXT NOT NULL, revision_id TEXT NOT NULL,
        PRIMARY KEY (owner, artifact_id))""",
    """CREATE TABLE IF NOT EXISTS artifact_revisions (
        owner TEXT NOT NULL, artifact_id TEXT NOT NULL, revision_id TEXT NOT NULL,
        committed_at TEXT NOT NULL, payload BLOB NOT NULL,
        PRIMARY KEY (owner, artifact_id, revision_id))""",
    """CREATE TABLE IF NOT EXISTS artifact_blob_refs (
        owner TEXT NOT NULL, artifact_id TEXT NOT NULL, revision_id TEXT NOT NULL,
        blob_key TEXT NOT NULL, PRIMARY KEY (owner, artifact_id, revision_id, blob_key))""",
    """CREATE TABLE IF NOT EXISTS artifact_publications (
        owner TEXT NOT NULL, publication_key TEXT NOT NULL,
        artifact_id TEXT NOT NULL, receipt BLOB,
        PRIMARY KEY (owner, publication_key))""",
    """CREATE TABLE IF NOT EXISTS artifact_links (
        owner TEXT NOT NULL, artifact_id TEXT NOT NULL, revision_id TEXT NOT NULL,
        context_kind TEXT NOT NULL, context_id TEXT NOT NULL,
        PRIMARY KEY (owner, artifact_id, revision_id, context_kind, context_id))""",
)


def _now() -> datetime:
    return datetime.now(UTC)


def _stamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("artifact clock must be timezone-aware")
    return value.astimezone(UTC).isoformat(timespec="microseconds")


class ArtifactRepository:
    """Internal owner-scoped persistence; no implicit grants or plaintext mode."""

    def __init__(self, db: AsyncDatabase, *, storage_id: str,
                 limits: ArtifactLimits | None = None, clock: Callable[[], datetime] = _now):
        from kestrel_sdk._validation import stable_token
        self.db = db
        self.storage_id = stable_token(storage_id, "storage_id")
        self.limits = limits or ArtifactLimits()
        self.clock = clock
        self._index_key: bytes | None = None
        self._initialization_lock = asyncio.Lock()

    async def ensure_initialized(self) -> None:
        """Open lazily inside an authorized service operation, once per runtime.

        An unavailable optional store must not take down the rest of the host.
        Failure leaves the original envelope intact and remains retryable after
        the operator restores its key. No plaintext or replacement-key fallback.
        """
        if self._index_key is None:
            async with self._initialization_lock:
                if self._index_key is None:
                    try:
                        await self.initialize()
                    except TransactionError as error:
                        if execution_commit_outcome(error) is not None:
                            raise ResourceOutcomeUnknown("artifact initialization requires reconciliation") from error
                        if isinstance(error.__cause__, ArtifactIntegrityError):
                            raise error.__cause__ from None
                        raise

    async def initialize(self) -> None:
        self._require_outermost()
        wrapping_cipher = get_fernet()
        if wrapping_cipher is None:
            raise ArtifactIntegrityError("artifact persistence requires encryption")
        async with self.db.migration_lock("artifact-repository-v1"):
            for statement in _SCHEMA:
                # AsyncDatabase handles placeholders, but schema byte types differ.
                sql = statement.replace(" BLOB", " BYTEA") if self.db.backend_type == "postgres" else statement
                await self.db.execute(sql)
            row = await self.db.fetchone("SELECT wrapped_key FROM artifact_keys WHERE key_id = ?", ("v1",))
            if row is None:
                if await self.db.fetchone("SELECT 1 FROM artifact_owners LIMIT 1"):
                    raise ArtifactIntegrityError("artifact envelope key is missing; restore the original key")
                index_key = secrets.token_bytes(32)
                await self.db.execute("INSERT INTO artifact_keys (key_id, wrapped_key) VALUES (?, ?)",
                                      ("v1", wrapping_cipher.encrypt(index_key)))
            else:
                try:
                    index_key = wrapping_cipher.decrypt(bytes(row[0]))
                except DecryptionError:
                    raise ArtifactIntegrityError(
                        "artifact envelope cannot be opened; restore its original KESTREL_DATA_KEY "
                        "or complete offline key rotation"
                    ) from None
            if len(index_key) != 32:
                raise ArtifactIntegrityError("invalid artifact envelope key")
        for name, table, columns in (
            ("idx_artifact_refs_blob", "artifact_blob_refs", "owner, blob_key"),
            ("idx_artifact_staging_expiry", "artifact_staging", "expires_at, owner"),
            ("idx_artifact_staging_blob", "artifact_staging", "owner, blob_key"),
            ("idx_artifact_history", "artifact_revisions", "owner, artifact_id, committed_at, revision_id"),
            ("idx_artifact_publication_artifact", "artifact_publications", "owner, artifact_id"),
        ):
            await self.db.ensure_index(name, table, columns)
        self._index_key = index_key

    def _require_outermost(self) -> None:
        if self.db.owns_open_transaction is not False:
            raise ResourceConflict("artifact operation requires its own durable transaction")

    def _key(self) -> bytes:
        if self._index_key is None:
            raise RuntimeError("artifact repository must be initialized")
        return self._index_key

    def _address(self, domain: str, *identity: str | None) -> str:
        encoded = json.dumps(["kestrel-artifacts-index-v1", domain, *identity],
                             ensure_ascii=True, separators=(",", ":")).encode()
        return hmac.new(self._key(), encoded, hashlib.sha256).hexdigest()

    def _owner(self, scope: ResourceScope) -> str:
        scope = ResourceScope.model_validate(scope)
        # Explicit identity tuple: serialization or non-identity schema changes
        # cannot change owner addresses or their encryption namespace.
        return self._address("owner", scope.tenant_id, scope.principal_id, scope.agent_id)

    def _cipher(self, owner: str):
        # Envelope encryption: master-key rotation rewraps artifact_keys only.
        # A domain-separated PRF derives one content key per opaque owner.
        key = hmac.new(self._key(), b"kestrel-artifacts-cipher-v1\0" + owner.encode(), hashlib.sha256).digest()
        return AEADCipher(key)

    def _seal(self, owner: str, identity: str, content: bytes) -> bytes:
        return self._cipher(owner).encrypt(content, aad=identity.encode())

    def _open(self, owner: str, identity: str, content: bytes) -> bytes:
        try:
            return self._cipher(owner).decrypt(bytes(content), aad=identity.encode())
        except DecryptionError as error:
            raise ArtifactIntegrityError("artifact decryption failed") from error

    @asynccontextmanager
    async def _transaction_owner(self, owner: str, *, write: bool = True,
                                 guard: Callable[[], None] | None = None):
        self._require_outermost()
        try:
            async with self.db.transaction(immediate=write):
                if guard is not None:
                    guard()
                if write:
                    # PostgreSQL row lock for quota, CAS, GC and publication.
                    await self.db.execute("INSERT INTO artifact_owners (owner, generation) VALUES (?, 0) "
                                          "ON CONFLICT (owner) DO NOTHING", (owner,))
                    await self.db.execute("UPDATE artifact_owners SET generation = generation + 1 WHERE owner = ?", (owner,))
                yield owner
                # Authorization may expire while native database work awaits.
                # Refuse inside the transaction, while rollback is still possible.
                if guard is not None:
                    guard()
        except Exception as error:
            # Native pool-retirement failures can carry commit acknowledgments
            # without being TransactionError. Classify evidence, not its carrier;
            # cancellation and control BaseExceptions retain native propagation.
            if execution_commit_outcome(error) is not None:
                raise ResourceOutcomeUnknown("artifact transaction requires reconciliation") from error
            # SQLite wraps body errors only after its rollback succeeds. Keep
            # the SDK conflict/not-found contract, without masking cleanup loss.
            if isinstance(error, TransactionError) and isinstance(error.__cause__, ResourceError):
                raise error.__cause__ from None
            raise

    @asynccontextmanager
    async def _transaction(self, scope: ResourceScope, *, write: bool = True,
                           guard: Callable[[], None] | None = None):
        async with self._transaction_owner(self._owner(scope), write=write, guard=guard) as owner:
            yield owner

    async def _collect(self, owner: str) -> int:
        await self.db.execute("DELETE FROM artifact_staging WHERE owner = ? AND expires_at <= ?",
                              (owner, _stamp(self.clock())))
        return await self.db.execute(
            "DELETE FROM artifact_blobs WHERE owner = ? "
            "AND NOT EXISTS (SELECT 1 FROM artifact_staging s WHERE s.owner = artifact_blobs.owner AND s.blob_key = artifact_blobs.blob_key) "
            "AND NOT EXISTS (SELECT 1 FROM artifact_blob_refs r "
            "WHERE r.owner = artifact_blobs.owner AND r.blob_key = artifact_blobs.blob_key)",
            (owner,),
        )

    async def collect_staging(self, scope: ResourceScope) -> int:
        async with self._transaction(scope) as owner:
            return await self._collect(owner)

    async def collect_expired_staging(self, *, owner_limit: int = 100) -> int:
        """Host maintenance, never an operator endpoint; works without owner keys.

The host calls this periodically, including for owners who never return.
Cleanup rechecks expiry and references while holding each owner's writer lock.
"""
        self._require_outermost()
        if type(owner_limit) is not int or not 1 <= owner_limit <= 1000:
            raise ValueError("invalid staging sweep bound")
        owners = await self.db.fetchall(
            "SELECT owner FROM (SELECT owner FROM artifact_staging WHERE expires_at <= ? UNION "
            "SELECT b.owner FROM artifact_blobs b WHERE NOT EXISTS "
            "(SELECT 1 FROM artifact_blob_refs r WHERE r.owner = b.owner AND r.blob_key = b.blob_key) "
            "AND NOT EXISTS (SELECT 1 FROM artifact_staging s WHERE s.owner = b.owner AND s.blob_key = b.blob_key)) candidates "
            "ORDER BY owner LIMIT ?", (_stamp(self.clock()), owner_limit))
        removed = 0
        for (owner,) in owners:
            async with self._transaction_owner(owner):
                removed += await self._collect(owner)
        return removed

    async def _check_metadata_quota(self, owner: str, additional: int) -> None:
        revisions = await self.db.fetchone("SELECT COALESCE(SUM(length(payload)), 0) FROM artifact_revisions WHERE owner = ?", (owner,))
        receipts = await self.db.fetchone("SELECT COALESCE(SUM(length(receipt)), 0) FROM artifact_publications WHERE owner = ?", (owner,))
        if revisions[0] + receipts[0] + additional > self.limits.max_metadata_bytes:
            raise ArtifactQuotaExceeded("artifact owner exceeds metadata byte limit")

    def _file_key(self, owner: str, file: ArtifactFile) -> str:
        return self._address("file", owner, file.path, file.sha256, str(file.size_bytes), file.media_type)

    async def stage(self, scope: ResourceScope, content: bytes, *, publication_key: str,
                    path: str, media_type: str,
                    guard: Callable[[], None] | None = None) -> ArtifactFile:
        from kestrel_sdk._validation import stable_token
        stable_token(publication_key, "publication_key")
        if not isinstance(content, bytes):
            raise TypeError("artifact content must be bytes")
        if len(content) > self.limits.max_file_bytes:
            raise ArtifactQuotaExceeded("artifact file exceeds byte limit")
        file = ArtifactFile(path=path, media_type=media_type, size_bytes=len(content),
                            sha256=hashlib.sha256(content).hexdigest())
        async with self._transaction(scope, guard=guard) as owner:
            await self._collect(owner)
            publication = self._address("publication", owner, publication_key)
            if await self.db.fetchone("SELECT 1 FROM artifact_publications WHERE owner = ? AND publication_key = ?", (owner, publication)):
                raise ResourceConflict("publication key is closed to new staging")
            path_key = self._address("path", owner, file.path)
            file_key = self._file_key(owner, file)
            staged = await self.db.fetchone("SELECT file_key FROM artifact_staging WHERE owner = ? AND publication_key = ? AND path_key = ?",
                                            (owner, publication, path_key))
            if staged is not None:
                if staged[0] != file_key:
                    raise ResourceConflict("publication path already binds another file")
                await self._blob(owner, file)
                return file
            staging_count = await self.db.fetchone("SELECT COUNT(*) FROM artifact_staging WHERE owner = ?", (owner,))
            if staging_count[0] >= self.limits.max_staging_files:
                raise ArtifactQuotaExceeded("artifact owner exceeds staging file limit")
            blob_key = self._address("blob", owner, file.sha256)
            row = await self.db.fetchone("SELECT content, size_bytes FROM artifact_blobs WHERE owner = ? AND blob_key = ?",
                                         (owner, blob_key))
            sealed = self._seal(owner, "blob:" + blob_key, content)
            if row is not None:
                if row[1] != len(content) or self._open(owner, "blob:" + blob_key, row[0]) != content:
                    raise ArtifactIntegrityError("staged blob identity mismatch")
            else:
                used = await self.db.fetchone("SELECT COALESCE(SUM(size_bytes), 0), COUNT(*), COALESCE(SUM(length(content)), 0) FROM artifact_blobs WHERE owner = ?", (owner,))
                if used[0] + len(content) > self.limits.max_owner_bytes:
                    raise ArtifactQuotaExceeded("artifact owner exceeds byte limit")
                if used[1] >= self.limits.max_blobs or used[2] + len(sealed) > self.limits.max_stored_bytes:
                    raise ArtifactQuotaExceeded("artifact owner exceeds blob storage limit")
            until = _stamp(self.clock() + timedelta(seconds=self.limits.staging_seconds))
            await self.db.execute("INSERT INTO artifact_blobs (owner, blob_key, size_bytes, content) VALUES (?, ?, ?, ?) "
                                  "ON CONFLICT (owner, blob_key) DO NOTHING", (owner, blob_key, len(content), sealed))
            await self.db.execute("INSERT INTO artifact_staging (owner, publication_key, path_key, blob_key, file_key, expires_at) "
                                  "VALUES (?, ?, ?, ?, ?, ?)", (owner, publication, path_key, blob_key, file_key, until))
        return file

    async def _receipt(self, owner: str, key: str) -> PublicationReceipt | None:
        publication_key = self._address("publication", owner, key)
        row = await self.db.fetchone("SELECT receipt, artifact_id FROM artifact_publications WHERE owner = ? AND publication_key = ?",
                                     (owner, publication_key))
        if row is None:
            return None
        if row[0] is None or not await self.db.fetchone("SELECT 1 FROM artifact_heads WHERE owner = ? AND artifact_id = ?", (owner, row[1])):
            raise ResourceNotFound("publication belongs to a deleted artifact")
        receipt = PublicationReceipt.model_validate_json(self._open(owner, "receipt:" + publication_key, row[0]))
        if self._owner(receipt.scope) != owner or receipt.idempotency_key != key or receipt.ref.artifact_id != row[1]:
            raise ArtifactIntegrityError("publication identity mismatch")
        return receipt

    async def publication_status(self, scope: ResourceScope, key: str, *,
                                 guard: Callable[[], None] | None = None) -> PublicationReceipt | None:
        async with self._transaction(scope, write=False, guard=guard) as owner:
            return await self._receipt(owner, key)

    async def _blob(self, owner: str, file: ArtifactFile) -> bytes:
        blob_key = self._address("blob", owner, file.sha256)
        row = await self.db.fetchone("SELECT content, size_bytes FROM artifact_blobs WHERE owner = ? AND blob_key = ?", (owner, blob_key))
        if row is None:
            raise ResourceNotFound("required artifact bytes are unavailable")
        content = self._open(owner, "blob:" + blob_key, row[0])
        if row[1] != file.size_bytes or len(content) != file.size_bytes or hashlib.sha256(content).hexdigest() != file.sha256:
            raise ArtifactIntegrityError("artifact bytes failed verification")
        return content

    async def commit(self, scope: ResourceScope, request: ArtifactCommit, *,
                     guard: Callable[[], None] | None = None) -> PublicationReceipt:
        request = ArtifactCommit.model_validate(request)
        async with self._transaction(scope, guard=guard) as owner:
            previous = await self._receipt(owner, request.idempotency_key)
            if previous is not None:
                if previous.request_sha256 != request.digest():
                    raise ResourceConflict("publication key already binds another request")
                return previous
            await self._collect(owner)
            publication_key = self._address("publication", owner, request.idempotency_key)
            for file in request.manifest.files:
                staged = await self.db.fetchone("SELECT file_key FROM artifact_staging WHERE owner = ? AND publication_key = ? AND path_key = ?",
                                                (owner, publication_key, self._address("path", owner, file.path)))
                if staged is None or staged[0] != self._file_key(owner, file):
                    raise ResourceNotFound("file is not staged for this publication")
                await self._blob(owner, file)
            count = await self.db.fetchone("SELECT COUNT(*) FROM artifact_revisions WHERE owner = ?", (owner,))
            if count[0] >= self.limits.max_revisions:
                raise ArtifactQuotaExceeded("artifact owner exceeds revision limit")
            publications = await self.db.fetchone("SELECT COUNT(*) FROM artifact_publications WHERE owner = ? AND receipt IS NOT NULL", (owner,))
            if publications[0] >= self.limits.max_publications:
                raise ArtifactQuotaExceeded("artifact owner exceeds publication limit")
            artifact_id = request.artifact_id or str(uuid4())
            if request.artifact_id is not None:
                head = await self.db.fetchone("SELECT revision_id FROM artifact_heads WHERE owner = ? AND artifact_id = ?", (owner, artifact_id))
                if head is None:
                    raise ResourceNotFound("artifact is unavailable")
                if head[0] != request.expected_revision_id:
                    raise ResourceConflict("artifact head changed; recover staged edits")
            revision = ArtifactRevision(ref=ArtifactRef(artifact_id=artifact_id, revision_id=str(uuid4())),
                                        scope=scope, parent_revision_id=request.expected_revision_id,
                                        title=request.title, manifest=request.manifest,
                                        provenance=request.provenance, committed_at=self.clock())
            receipt = PublicationReceipt(ref=revision.ref, scope=scope, idempotency_key=request.idempotency_key,
                                         request_sha256=request.digest(), manifest_sha256=request.manifest.digest(),
                                         committed_at=revision.committed_at, storage_id=self.storage_id)
            payload = self._seal(owner, "revision:" + revision.ref.revision_id, revision.model_dump_json().encode())
            receipt_payload = self._seal(owner, "receipt:" + publication_key, receipt.model_dump_json().encode())
            await self._check_metadata_quota(owner, len(payload) + len(receipt_payload))
            await self.db.execute("INSERT INTO artifact_revisions (owner, artifact_id, revision_id, committed_at, payload) VALUES (?, ?, ?, ?, ?)",
                                  (owner, artifact_id, revision.ref.revision_id, _stamp(revision.committed_at),
                                   payload))
            await self.db.execute("INSERT INTO artifact_heads (owner, artifact_id, revision_id) VALUES (?, ?, ?) "
                                  "ON CONFLICT (owner, artifact_id) DO UPDATE SET revision_id = excluded.revision_id",
                                  (owner, artifact_id, revision.ref.revision_id))
            for digest in {file.sha256 for file in request.manifest.files}:
                blob_key = self._address("blob", owner, digest)
                await self.db.execute("INSERT INTO artifact_blob_refs (owner, artifact_id, revision_id, blob_key) VALUES (?, ?, ?, ?)",
                                      (owner, artifact_id, revision.ref.revision_id, blob_key))
            await self.db.execute("DELETE FROM artifact_staging WHERE owner = ? AND publication_key = ?", (owner, publication_key))
            await self.db.execute("INSERT INTO artifact_publications (owner, publication_key, artifact_id, receipt) VALUES (?, ?, ?, ?)",
                                  (owner, publication_key, artifact_id, receipt_payload))
            await self._collect(owner)
        return receipt

    async def _get(self, owner: str, ref: ArtifactRef) -> ArtifactRevision:
        row = await self.db.fetchone("SELECT payload FROM artifact_revisions WHERE owner = ? AND artifact_id = ? AND revision_id = ?",
                                     (owner, ref.artifact_id, ref.revision_id))
        if row is None:
            raise ResourceNotFound("artifact revision is unavailable")
        return self._decode_revision(owner, ref, row[0])

    def _decode_revision(self, owner: str, ref: ArtifactRef, payload: bytes) -> ArtifactRevision:
        revision = ArtifactRevision.model_validate_json(self._open(owner, "revision:" + ref.revision_id, payload))
        if revision.ref != ref or self._owner(revision.scope) != owner:
            raise ArtifactIntegrityError("artifact revision identity mismatch")
        return revision

    async def get(self, scope: ResourceScope, ref: ArtifactRef, *,
                  guard: Callable[[], None] | None = None) -> ArtifactRevision:
        async with self._transaction(scope, write=False, guard=guard) as owner:
            return await self._get(owner, ArtifactRef.model_validate(ref))

    async def head(self, scope: ResourceScope, artifact_id: str, *,
                   guard: Callable[[], None] | None = None) -> ArtifactRevision:
        from kestrel_sdk._validation import stable_token
        artifact_id = stable_token(artifact_id, "artifact_id")
        async with self._transaction(scope, write=False, guard=guard) as owner:
            # One SQL snapshot binds the committed head and its payload. A
            # timestamp-sorted history is not authoritative, even on one host.
            row = await self.db.fetchone(
                "SELECT h.revision_id, r.payload FROM artifact_heads h "
                "LEFT JOIN artifact_revisions r ON r.owner = h.owner AND r.artifact_id = h.artifact_id "
                "AND r.revision_id = h.revision_id WHERE h.owner = ? AND h.artifact_id = ?",
                (owner, artifact_id),
            )
            if row is None:
                raise ResourceNotFound("artifact is unavailable")
            if row[1] is None:
                raise ArtifactIntegrityError("artifact head revision is unavailable")
            return self._decode_revision(owner, ArtifactRef(artifact_id=artifact_id, revision_id=row[0]), row[1])

    async def read(self, scope: ResourceScope, ref: ArtifactRef, path: str, *,
                   guard: Callable[[], None] | None = None) -> bytes:
        async with self._transaction(scope, write=False, guard=guard) as owner:
            revision = await self._get(owner, ArtifactRef.model_validate(ref))
            for file in revision.manifest.files:
                if file.path == path:
                    return await self._blob(owner, file)
            raise ResourceNotFound("artifact file is unavailable")

    async def page(self, scope: ResourceScope, *, artifact_id: str | None = None,
                   cursor: str | None = None, limit: int = 50,
                   guard: Callable[[], None] | None = None) -> ArtifactPage:
        from kestrel_sdk._validation import stable_token
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("artifact page limit must be between 1 and 100")
        anchor = None
        if cursor is not None:
            try:
                if not isinstance(cursor, str) or not 1 <= len(cursor) <= 2048:
                    raise ValueError
                anchor = json.loads(base64.b64decode(cursor, altchars=b"-_", validate=True))
                if (not isinstance(anchor, list) or len(anchor) != 3 or anchor[2] != artifact_id
                        or not isinstance(anchor[0], str) or not isinstance(anchor[1], str)):
                    raise ValueError
                if _stamp(datetime.fromisoformat(anchor[0])) != anchor[0]:
                    raise ValueError
                stable_token(anchor[1], "cursor revision")
            except (ValueError, TypeError) as error:
                raise ValueError("invalid artifact page cursor") from error
        if artifact_id is not None:
            stable_token(artifact_id, "artifact_id")
        async with self._transaction(scope, write=False, guard=guard) as owner:
            sql = "SELECT r.artifact_id, r.revision_id, r.committed_at, r.payload FROM artifact_revisions r "
            if artifact_id is None:
                sql += ("JOIN artifact_heads h ON h.owner = r.owner AND h.artifact_id = r.artifact_id "
                        "AND h.revision_id = r.revision_id ")
            sql += "WHERE r.owner = ? "
            params = [owner]
            if artifact_id is not None:
                sql += "AND r.artifact_id = ? "
                params.append(artifact_id)
            if anchor is not None:
                sql += "AND (r.committed_at < ? OR (r.committed_at = ? AND r.revision_id < ?)) "
                params.extend((anchor[0], anchor[0], anchor[1]))
            sql += "ORDER BY r.committed_at DESC, r.revision_id DESC LIMIT ?"
            rows = await self.db.fetchall(sql, (*params, limit + 1))
            # Decode the payloads from this single SQL snapshot. A peer delete
            # after selection cannot turn a valid listing into a not-found error.
            items = tuple(self._decode_revision(owner, ArtifactRef(artifact_id=row[0], revision_id=row[1]), row[3])
                          for row in rows[:limit])
            next_cursor = None
            if len(rows) > limit:
                last = rows[limit - 1]
                next_cursor = base64.urlsafe_b64encode(json.dumps([last[2], last[1], artifact_id], separators=(",", ":")).encode()).decode()
            return ArtifactPage(items=items, next_cursor=next_cursor)

    async def link(self, scope: ResourceScope, link: ArtifactLink, *, remove: bool = False,
                   guard: Callable[[], None] | None = None) -> None:
        link = ArtifactLink.model_validate(link)
        async with self._transaction(scope, guard=guard) as owner:
            await self._get(owner, link.ref)
            values = (owner, link.ref.artifact_id, link.ref.revision_id, link.context_kind,
                      self._address("context", owner, link.context_kind, link.context_id))
            if remove:
                await self.db.execute("DELETE FROM artifact_links WHERE owner = ? AND artifact_id = ? AND revision_id = ? AND context_kind = ? AND context_id = ?", values)
            else:
                existing = await self.db.fetchone("SELECT 1 FROM artifact_links WHERE owner = ? AND artifact_id = ? AND revision_id = ? AND context_kind = ? AND context_id = ?", values)
                if existing:
                    return
                count = await self.db.fetchone("SELECT COUNT(*) FROM artifact_links WHERE owner = ?", (owner,))
                if count[0] >= self.limits.max_links:
                    raise ArtifactQuotaExceeded("artifact owner exceeds context link limit")
                await self.db.execute("INSERT INTO artifact_links (owner, artifact_id, revision_id, context_kind, context_id) VALUES (?, ?, ?, ?, ?) ON CONFLICT DO NOTHING", values)

    async def delete(self, scope: ResourceScope, expected_head: ArtifactRef, *,
                     guard: Callable[[], None] | None = None) -> None:
        expected_head = ArtifactRef.model_validate(expected_head)
        async with self._transaction(scope, guard=guard) as owner:
            head = await self.db.fetchone("SELECT revision_id FROM artifact_heads WHERE owner = ? AND artifact_id = ?", (owner, expected_head.artifact_id))
            if head is None:
                raise ResourceNotFound("artifact is unavailable")
            if head[0] != expected_head.revision_id:
                raise ResourceConflict("artifact head changed before deletion")
            for table in ("artifact_links", "artifact_blob_refs", "artifact_revisions", "artifact_heads"):
                await self.db.execute(f"DELETE FROM {table} WHERE owner = ? AND artifact_id = ?", (owner, expected_head.artifact_id))
            # Compact keyed tombstones preserve non-resurrection without imposing
            # a lifetime limit on new publications. Private receipts are erased.
            await self.db.execute("UPDATE artifact_publications SET receipt = NULL WHERE owner = ? AND artifact_id = ?", (owner, expected_head.artifact_id))
            # Delete only retired source bytes; unrelated staged edits must survive.
            await self._collect(owner)
