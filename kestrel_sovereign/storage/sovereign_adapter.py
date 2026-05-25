#!/usr/bin/env python3
"""
Sovereign Storage Adapter V2 - The Encrypted Merkle Forest.

This adapter implements the "Convergent Sharding" protocol:
1. Shards data by time (Conversations) and content (Files).
2. Uses Convergent Encryption (Key = HMAC(Content)) for deduplication.
3. Manages a Root Manifest DAG on IPFS.
"""

import abc
import asyncio
import json
import logging
import hashlib
import hmac
import os
from typing import (
    TYPE_CHECKING, Callable, Dict, Iterable, List, Optional, Any, Tuple,
)
from dataclasses import dataclass, field, asdict
from datetime import datetime, UTC
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from kestrel_sovereign.filecoin_adapter import FilecoinAdapter, StorageTier, StorageResult
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.car_builder import CARBuilder, CARReader

# Lazy-imported below inside import_agent — identity/__init__ pulls in
# the exporter chain which transitively loads features.bootstrap, and
# that chain imports back into storage. Type-only references can use
# the TYPE_CHECKING guard without triggering the circular load.
if TYPE_CHECKING:
    from kestrel_sovereign.identity.access_grant import DataAccessGrant

# Constants
SHARD_SIZE_LIMIT = 5 * 1024 * 1024  # 5MB per shard
MANIFEST_VERSION = "3.0"

@dataclass
class ShardMetadata:
    """Metadata for a single encrypted shard"""
    shard_id: str
    type: str               # 'conversation', 'file', 'graph'
    time_range: str         # '2025-11'
    cid: str                # IPFS Content ID
    content_hash: str       # SHA256 of plaintext
    size_bytes: int
    encryption_algo: str = "AES-GCM-256"
    key_derivation: str = "HMAC-SHA256"

@dataclass
class AssetDescriptor:
    """Describes a binary asset to be included in an export.

    Provided by downstream ``AssetCollector`` implementations so
    kestrel-sovereign can encrypt/upload the asset alongside conversation
    shards.
    """
    asset_type: str          # 'avatar', 'lora_weights', 'selfie', 'personality'
    asset_key: str           # unique key within this agent (e.g. "avatar_main")
    content_hash: str        # SHA256 hex of the plaintext bytes
    size_bytes: int
    ipfs_cid: Optional[str] = None   # skip upload if already on IPFS
    data: Optional[bytes] = None     # raw bytes (None when ipfs_cid is set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    encrypted: bool = False          # True if data is pre-encrypted


@dataclass
class AssetMetadata:
    """Manifest entry for an exported asset (parallel to ShardMetadata)."""
    asset_type: str
    asset_key: str
    cid: str                 # IPFS CID (or local fallback)
    content_hash: str
    size_bytes: int
    encrypted: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class AssetCollector(abc.ABC):
    """Protocol for downstream apps to attach binary assets to an export.

    Implementations (e.g. Frinz companion avatars) return a list of
    ``AssetDescriptor`` objects that will be encrypted and bundled into
    the sovereignty export.
    """

    @abc.abstractmethod
    async def collect_assets(self, agent_did: str) -> List[AssetDescriptor]:
        """Return all assets that should be exported for *agent_did*."""
        ...


class AssetRestorer(abc.ABC):
    """Protocol for downstream apps to restore exported assets on import.

    Symmetric counterpart to :class:`AssetCollector`. The export path
    serializes downstream-owned bytes into the sovereignty CAR; this
    protocol is how the receiving side gets those bytes back into its
    local stores.

    Lifecycle per import:

      1. ``SovereignStorageAdapter.import_agent`` walks
         ``manifest.assets``, fetches each block from the CAR, and
         decrypts those with a keyring entry (the
         convergent-encrypted ones). External-ref assets (blocks that
         are link nodes to IPFS) are surfaced as metadata-only on
         ``SovereignImportResult.assets_restored`` and skipped here —
         restorer Phase 2 will fetch them via the filecoin adapter.
      2. The adapter groups the resulting ``(AssetMetadata, bytes)``
         pairs by :attr:`asset_types` and routes each group to every
         restorer that declares the type.
      3. Restorers raise on any failure they want to surface; the
         adapter propagates the exception after writing a structured
         audit row (the host DB's conversation restore is in the same
         try-block, so an asset-restore failure rolls back nothing
         that was already written, but the next caller sees a clean
         error path).

    A restorer that handles multiple asset types declares them all in
    :attr:`asset_types`; the same restorer can receive a mix in one
    ``restore_assets`` call.
    """

    @property
    @abc.abstractmethod
    def asset_types(self) -> List[str]:
        """Asset type tokens this restorer accepts.

        Tokens match the ``asset_type`` field on
        :class:`AssetDescriptor`/``AssetMetadata`` (e.g.
        ``"fhir_resource"``, ``"avatar"``).
        """
        ...

    @abc.abstractmethod
    async def restore_assets(
        self,
        agent_did: str,
        assets: List[Tuple[AssetMetadata, bytes]],
    ) -> int:
        """Restore *assets* into local storage. Return count restored.

        ``assets`` is the subset of the import's assets matching one of
        this restorer's :attr:`asset_types`. Empty list means the
        adapter has nothing to hand off — restorer may treat that as a
        no-op or raise. ``bytes`` is decrypted plaintext for assets
        encrypted under the adapter's convergent keyring, OR raw block
        bytes for assets the caller pre-encrypted (the restorer
        decrypts in its own scheme).

        The returned count is recorded on
        :class:`SovereignImportResult` so callers can verify what
        landed without inspecting each restorer.
        """
        ...


@dataclass
class RootManifest:
    """The Root DAG Node - The Agent's State"""
    version: str
    timestamp: str
    agent_did: str
    shards: List[ShardMetadata]
    assets: List[AssetMetadata] = field(default_factory=list)
    index_cid: Optional[str] = None
    keyring_cid: Optional[str] = None
    previous_root: Optional[str] = None # For git-like history


@dataclass
class ImportCheck:
    """One statically-checkable property of an inbound sovereignty CAR."""
    name: str
    passed: bool
    weight: float
    detail: str = ""


@dataclass
class ImportContinuity:
    """Static integrity/continuity score for a sovereignty CAR import.

    A sovereignty CAR is a *static* archive of conversation + asset
    shards. The challenge/response ``ContinuityVerifier`` (identity
    package) cannot run here — there is no live respondent to answer
    challenges. This score is derived purely from statically checkable
    properties of the package, weighted by importance. 0.0–1.0.
    """
    overall_score: float
    checks: List[ImportCheck] = field(default_factory=list)
    verification_timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.verification_timestamp:
            self.verification_timestamp = datetime.now(UTC).isoformat()

    @property
    def checks_passed(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def checks_total(self) -> int:
        return len(self.checks)

    def is_verified(self, threshold: float = 0.7) -> bool:
        return self.overall_score >= threshold

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_score": self.overall_score,
            "checks_passed": self.checks_passed,
            "checks_total": self.checks_total,
            "checks": [asdict(c) for c in self.checks],
            "verification_timestamp": self.verification_timestamp,
        }


@dataclass
class SovereignImportResult:
    """Structured result of ``import_agent()``.

    Symmetric counterpart to ``export_agent()`` (which returns a CID
    string). ``status`` is one of ``imported`` | ``rejected`` | ``error``.
    A ``rejected`` result means the package failed verification and the
    host database was left untouched.

    ``assets_restored`` carries the manifest metadata for every asset
    in the CAR (matches pre-#1391 behavior). ``asset_payload_counts``
    is the per-asset-type count of payloads actually handed off to
    :class:`AssetRestorer` instances during import — when no restorers
    are wired (legacy callers), every value is ``0`` even though
    ``assets_restored`` is populated.
    """
    success: bool
    agent_did: str
    package_hash: str
    continuity: ImportContinuity
    status: str
    reject_reason: Optional[str] = None
    manifest_version: str = ""
    messages_restored: int = 0
    shards_restored: int = 0
    assets_restored: List[Dict[str, Any]] = field(default_factory=list)
    asset_payload_counts: Dict[str, int] = field(default_factory=dict)
    asset_payloads_skipped: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: str = ""
    source_cid: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "agent_did": self.agent_did,
            "package_hash": self.package_hash,
            "continuity": self.continuity.to_dict(),
            "status": self.status,
            "reject_reason": self.reject_reason,
            "manifest_version": self.manifest_version,
            "messages_restored": self.messages_restored,
            "shards_restored": self.shards_restored,
            "assets_restored": self.assets_restored,
            "asset_payload_counts": dict(self.asset_payload_counts),
            "asset_payloads_skipped": self.asset_payloads_skipped,
            "timestamp": self.timestamp,
            "source_cid": self.source_cid,
        }


class ConvergentEncryptor:
    """
    Handles deterministic encryption for deduplication.
    Key = HMAC(Content, User_Secret)
    """
    def __init__(self, user_secret: str):
        self.secret = user_secret.encode('utf-8')

    def derive_key(self, content: bytes) -> bytes:
        """Derive a 32-byte key from content + secret"""
        h = hmac.new(self.secret, content, hashlib.sha256)
        return h.digest()

    def encrypt(self, content: bytes) -> Tuple[bytes, bytes]:
        """
        Encrypt content deterministically.
        Returns (ciphertext, key).
        """
        key = self.derive_key(content)
        aesgcm = AESGCM(key)
        # Deterministic IV is required for deduplication.
        # We use the first 12 bytes of the content hash as IV.
        # Security Note: This leaks equality (if IV+Ciphertext matches, plaintext matches).
        # This is acceptable and desired for deduplication.
        nonce = hashlib.sha256(content).digest()[:12]
        ciphertext = aesgcm.encrypt(nonce, content, None)
        return ciphertext, key

    def decrypt(self, ciphertext: bytes, key: bytes) -> bytes:
        """Decrypt content"""
        aesgcm = AESGCM(key)
        # We need to recover the nonce. In this scheme, we can't easily recover
        # the nonce from the ciphertext alone without storing it.
        # FIX: We should prepend the nonce to the ciphertext.
        nonce = ciphertext[:12]
        actual_ciphertext = ciphertext[12:]
        return aesgcm.decrypt(nonce, actual_ciphertext, None)

    def encrypt_with_nonce_prefix(self, content: bytes) -> Tuple[bytes, bytes]:
        """Encrypt and prepend nonce for storage"""
        key = self.derive_key(content)
        aesgcm = AESGCM(key)
        nonce = hashlib.sha256(content).digest()[:12]
        ciphertext = aesgcm.encrypt(nonce, content, None)
        return nonce + ciphertext, key


class SovereignStorageAdapter:
    """
    V2 Adapter for Sharded, Deduplicated, Encrypted Storage.
    """

    def __init__(self, db: AsyncDatabase, user_secret: str, filecoin_adapter: Optional[FilecoinAdapter] = None, agent_id: str = ""):
        self.db = db
        self.agent_id = agent_id
        self.encryptor = ConvergentEncryptor(user_secret)
        self.adapter = filecoin_adapter or FilecoinAdapter()
        self.logger = logging.getLogger(__name__)

    def _now_sql(self) -> str:
        """Get SQL expression for current timestamp based on backend type."""
        if self.db.backend_type == "postgres":
            return "NOW()"
        return "datetime('now')"

    async def _get_conversations(self) -> List[Dict]:
        """Get all conversations from DB for this agent"""
        rows = await self.db.fetchall(
            "SELECT role, content, metadata, id FROM conversation_history WHERE agent_id = ? ORDER BY id ASC",
            (self.agent_id,)
        )
        return [
            {
                "role": row[0],
                "content": row[1],
                "metadata": json.loads(row[2]) if row[2] else {},
                "id": row[3]
            }
            for row in rows
        ]

    async def _shard_conversations(self) -> Dict[str, List[Dict]]:
        """
        Groups conversations by Month (YYYY-MM).
        Returns { '2025-11': [msgs...], '2025-10': [msgs...] }
        """
        shards = {}
        conversations = await self._get_conversations()
        for msg in conversations:
            # Extract timestamp from metadata. Falls back to current time if missing.
            # Note: The conversation_history table has created_at, but we use metadata.timestamp
            # for sharding consistency since metadata is what gets exported/imported.
            ts_str = msg.get("metadata", {}).get("timestamp", datetime.now(UTC).isoformat())
            try:
                dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                month_key = dt.strftime("%Y-%m")
            except ValueError:
                month_key = "unknown"

            if month_key not in shards:
                shards[month_key] = []
            shards[month_key].append(msg)
        return shards

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _upload_content(
        self, content: bytes, storage_tier: StorageTier, metadata: Optional[Dict] = None
    ) -> str:
        """Upload *content* and return its CID (or local fallback)."""
        result = await asyncio.to_thread(
            self.adapter.store_content,
            content=content,
            storage_tier=storage_tier,
            encrypt=False,
            metadata=metadata,
        )
        cid = result.ipfs_cid
        if not cid and result.storage_tier == StorageTier.LOCAL_ONLY:
            cid = f"local-{result.content_hash}"
        if not cid:
            raise RuntimeError("Failed to upload content")
        return cid

    async def _download_content(self, cid: str) -> bytes:
        """Download content by CID (handles local- fallback)."""
        return await asyncio.to_thread(
            self.adapter.retrieve_content,
            content_hash=cid.replace("local-", ""),
            ipfs_cid=cid if not cid.startswith("local-") else None,
        )

    def _encrypt_keyring(self, keyring: Dict[str, str]) -> bytes:
        """Encrypt the keyring dict and return nonce+ciphertext."""
        keyring_json = json.dumps(keyring).encode("utf-8")
        keyring_key = self.encryptor.derive_key(b"KESTREL_KEYRING_V2")
        aesgcm = AESGCM(keyring_key)
        nonce = os.urandom(12)
        return nonce + aesgcm.encrypt(nonce, keyring_json, None)

    def _decrypt_keyring(self, keyring_cipher: bytes) -> Dict[str, str]:
        """Decrypt a keyring blob and return the key map."""
        keyring_key = self.encryptor.derive_key(b"KESTREL_KEYRING_V2")
        aesgcm = AESGCM(keyring_key)
        nonce = keyring_cipher[:12]
        return json.loads(aesgcm.decrypt(nonce, keyring_cipher[12:], None).decode("utf-8"))

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    async def export_agent(
        self,
        agent_did: str,
        storage_tier: StorageTier = StorageTier.IPFS,
        asset_collector: Optional[AssetCollector] = None,
    ) -> str:
        """
        Export agent state as a single CAR v1 archive.

        All encrypted shards, assets, and the keyring are packed into one
        CAR file via ``CARBuilder``.  A single ``store_content()`` call
        produces one CID that represents the entire export.

        1. Encrypt conversation shards → add as raw blocks
        2. Collect & encrypt assets → add as raw/external-ref blocks
        3. Encrypt keyring → add as raw block
        4. Build dag-cbor manifest → set as CAR root
        5. Upload single CAR blob → return one CID
        """
        self.logger.info(f"🌲 Starting V3 CAR Export for {agent_did}...")

        builder = CARBuilder()
        shard_metadata_list: List[ShardMetadata] = []
        asset_metadata_list: List[AssetMetadata] = []
        keyring: Dict[str, str] = {}

        # 1. Process Conversation Shards
        conv_shards = await self._shard_conversations()
        for month, msgs in conv_shards.items():
            shard_content = json.dumps(msgs).encode("utf-8")
            ciphertext, key = self.encryptor.encrypt_with_nonce_prefix(shard_content)

            block_cid = builder.add_raw_block(ciphertext)

            meta = ShardMetadata(
                shard_id=f"conv_{month}",
                type="conversation",
                time_range=month,
                cid=block_cid,
                content_hash=hashlib.sha256(shard_content).hexdigest(),
                size_bytes=len(ciphertext),
            )
            shard_metadata_list.append(meta)
            keyring[meta.shard_id] = key.hex()
            self.logger.info(f"   Shard {month}: {len(msgs)} msgs -> {block_cid[:20]}...")

        # 2. Process Assets
        if asset_collector is not None:
            descriptors = await asset_collector.collect_assets(agent_did)
            for desc in descriptors:
                if desc.ipfs_cid:
                    # Already on IPFS — store a lightweight link node
                    block_cid = builder.add_external_ref(desc.ipfs_cid, ref_type=desc.asset_type)
                    encrypted = False
                else:
                    if desc.data is None:
                        raise ValueError(
                            f"AssetDescriptor {desc.asset_key!r} has no data and no ipfs_cid"
                        )
                    if desc.encrypted:
                        asset_cipher = desc.data
                    else:
                        asset_cipher, asset_key = self.encryptor.encrypt_with_nonce_prefix(desc.data)
                        keyring[f"asset_{desc.asset_key}"] = asset_key.hex()
                    block_cid = builder.add_raw_block(asset_cipher)
                    encrypted = True

                asset_metadata_list.append(AssetMetadata(
                    asset_type=desc.asset_type,
                    asset_key=desc.asset_key,
                    cid=block_cid,
                    content_hash=desc.content_hash,
                    size_bytes=desc.size_bytes,
                    encrypted=encrypted,
                    metadata=desc.metadata,
                ))
                self.logger.info(f"   Asset {desc.asset_key}: {desc.asset_type} -> {block_cid[:20]}...")

        # 3. Encrypt & add keyring
        keyring_cipher = self._encrypt_keyring(keyring)
        keyring_cid = builder.add_raw_block(keyring_cipher)

        # 4. Build manifest as dag-cbor root
        manifest = RootManifest(
            version=MANIFEST_VERSION,
            timestamp=datetime.now(UTC).isoformat(),
            agent_did=agent_did,
            shards=shard_metadata_list,
            assets=asset_metadata_list,
            keyring_cid=keyring_cid,
        )
        manifest_cid = builder.add_dag_cbor_block(asdict(manifest))
        builder.set_root(manifest_cid)

        # 5. Upload single CAR blob
        car_bytes = builder.build()
        root_cid = await self._upload_content(car_bytes, storage_tier)

        self.logger.info(
            f"✅ V3 CAR Export Complete. {builder.block_count} blocks, "
            f"{len(car_bytes)} bytes -> {root_cid}"
        )
        return root_cid

    # ------------------------------------------------------------------
    # Import
    # ------------------------------------------------------------------

    @staticmethod
    def _manifest_from_dict(data: dict) -> RootManifest:
        """Reconstruct a ``RootManifest`` from a plain dict."""
        data["shards"] = [ShardMetadata(**s) for s in data["shards"]]
        data["assets"] = [AssetMetadata(**a) for a in data.get("assets", [])]
        return RootManifest(**data)

    async def import_agent(
        self,
        package: Any,
        *,
        source_did: Optional[str] = None,
        verify_continuity: bool = True,
        continuity_threshold: float = 0.7,
        target_db_path: Optional[str] = None,
        grant: Optional["DataAccessGrant"] = None,
        host_did: Optional[str] = None,
        host_policy: Optional[
            Callable[["DataAccessGrant", str], bool]
        ] = None,
        grant_did_web_resolver: Optional[Callable[[str], Any]] = None,
        revoked_grant_ids: Optional[Iterable[str]] = None,
        asset_restorers: Optional[List[AssetRestorer]] = None,
    ) -> SovereignImportResult:
        """
        Import (restore) an agent from a sovereignty CAR archive.

        Symmetric counterpart to :meth:`export_agent`. ``package`` is
        either the CAR's root CID (``str``, downloaded here) or the raw
        CAR bytes (``bytes``).

        When ``verify_continuity`` is True (default) the package is run
        through :class:`SovereignImportVerifier` *before* any host
        database mutation. If the static continuity score is below
        ``continuity_threshold`` (or a critical check hard-fails) the
        import is rejected, the host DB is left untouched, and the
        rejected attempt is still recorded in ``agent_import_log``.

        Every attempt — imported, rejected, or errored — appends exactly
        one row to the append-only ``agent_import_log``, keyed by agent
        DID + source DID + package hash + continuity score + timestamp.

        Consent gate (#1379, follow-up to #1273): when ``grant`` is
        provided, owner-consent is verified AFTER CAR integrity but
        BEFORE any host-DB mutation. The CAR's structural integrity
        (block-hash + keyring decryptability) is the source-attestation
        equivalent of an AgentIdentityPackage signature; the grant adds
        the owner-side authorization (signed by the owner, names the
        manifest's source DID, targets the receiving agent's DID, not
        expired, not revoked). On consent rejection the import is
        refused with a distinct ``consent_*`` reject reason and the
        host DB is left untouched. ``grant=None`` preserves the
        pre-#1379 unauthenticated-host-trust behavior.

        Args:
            grant: Optional owner-signed :class:`DataAccessGrant`
                authorizing this import. Required to be paired with
                ``host_did`` so the grant has a receiver DID to bind
                against; ``host_did`` without ``grant`` is ignored.
            host_did: The receiving agent's own DID. Compared against
                ``grant.host_did`` during consent verification. Required
                when ``grant`` is provided.
            host_policy: Optional host-side filter callable evaluated
                AFTER consent verification returns ``ok=True``. Receives
                ``(grant, canonical_grant_id)``; the canonical id is the
                verifier-recomputed value, never ``grant.grant_id``.
                Returning ``False`` rejects the import with
                ``host_policy_rejected``. Ignored when ``grant`` is
                ``None``.
            grant_did_web_resolver: Optional resolver forwarded to
                :func:`verify_did_binding` for ``did:web:`` owners. The
                binding helper refuses-by-default without one for
                ``did:web:`` owners. Ignored when ``grant`` is ``None``.
            revoked_grant_ids: Optional iterable of canonical grant ids
                that are currently revoked. The recomputed canonical id
                is compared against this set. Sourced from a trusted
                registry — never from the grant payload. Ignored when
                ``grant`` is ``None``.
            asset_restorers: Optional list of :class:`AssetRestorer`
                instances that consume the CAR's asset payloads
                (#1391). Each manifest asset whose ``asset_type``
                matches a restorer's :attr:`AssetRestorer.asset_types`
                is decrypted (via the convergent keyring when the
                exporter let the adapter encrypt it, raw bytes when
                the exporter pre-encrypted) and handed to the
                restorer in a single ``restore_assets`` call per
                ``(restorer, asset_type)`` group. External-ref assets
                (link nodes pointing at IPFS) are skipped with a
                structured entry on ``asset_payloads_skipped`` —
                they're a Phase-2 follow-up. ``None`` preserves the
                pre-#1391 behavior bit-for-bit: ``assets_restored``
                still populates from the manifest, and
                ``asset_payload_counts`` is empty.
        """
        source_cid: Optional[str] = None
        try:
            if isinstance(package, (bytes, bytearray)):
                car_bytes = bytes(package)
            else:
                source_cid = str(package)
                car_bytes = await self._download_content(source_cid)
        except Exception as e:
            await self._log_import_attempt(
                agent_did="", source_did=source_did or "",
                package_hash="", continuity_score=0.0, status="error",
                reject_reason=f"package_fetch_failed: {e}",
                messages_restored=0, shards_restored=0,
                manifest_version="", source_cid=source_cid,
            )
            raise RuntimeError(f"Failed to fetch sovereignty package: {e}")

        package_hash = hashlib.sha256(car_bytes).hexdigest()
        self.logger.info(
            f"🌲 Sovereignty import: package {package_hash[:16]}… "
            f"({'cid ' + source_cid if source_cid else f'{len(car_bytes)} bytes'})"
        )

        verifier = SovereignImportVerifier(self.encryptor)
        continuity, reject_reason, manifest, reader = verifier.verify(car_bytes)
        agent_did = manifest.agent_did if manifest else ""
        src_did = source_did or agent_did
        manifest_version = manifest.version if manifest else ""

        if verify_continuity and (
            reject_reason is not None
            or not continuity.is_verified(continuity_threshold)
        ):
            reason = reject_reason or (
                f"continuity_below_threshold "
                f"({continuity.overall_score:.3f} < {continuity_threshold})"
            )
            self.logger.warning(
                f"⛔ Sovereignty import rejected ({reason}); host DB untouched"
            )
            await self._log_import_attempt(
                agent_did=agent_did, source_did=src_did,
                package_hash=package_hash,
                continuity_score=continuity.overall_score,
                status="rejected", reject_reason=reason,
                messages_restored=0, shards_restored=0,
                manifest_version=manifest_version, source_cid=source_cid,
            )
            return SovereignImportResult(
                success=False, agent_did=agent_did, package_hash=package_hash,
                continuity=continuity, status="rejected", reject_reason=reason,
                manifest_version=manifest_version,
                timestamp=(manifest.timestamp if manifest else ""),
                source_cid=source_cid,
            )

        # Consent gate (#1379) — runs AFTER CAR integrity (so the
        # caller's CAR-side attestation contract is satisfied) and
        # BEFORE any host-DB mutation (so a rejected grant leaves the
        # host DB untouched, same invariant as the continuity gate).
        # Lazy-import to break the storage→identity circular at module
        # load time (identity/__init__ pulls in the exporter chain
        # which transitively imports storage).
        consent_grant_id: Optional[str] = None
        if grant is not None:
            from kestrel_sovereign.identity.access_grant import (
                REJECT_HOST_POLICY,
            )
            from kestrel_sovereign.storage.sovereign_import_consent import (
                verify_car_import_consent,
            )
            if not host_did:
                consent_reason = (
                    "consent grant requires host_did to be passed to "
                    "import_agent so the grant's host_did field has a "
                    "receiver DID to bind against"
                )
                await self._log_import_attempt(
                    agent_did=agent_did, source_did=src_did,
                    package_hash=package_hash,
                    continuity_score=continuity.overall_score,
                    status="rejected", reject_reason=consent_reason,
                    messages_restored=0, shards_restored=0,
                    manifest_version=manifest_version, source_cid=source_cid,
                    grant_id=None,
                )
                return SovereignImportResult(
                    success=False, agent_did=agent_did,
                    package_hash=package_hash, continuity=continuity,
                    status="rejected", reject_reason=consent_reason,
                    manifest_version=manifest_version,
                    timestamp=(manifest.timestamp if manifest else ""),
                    source_cid=source_cid,
                )
            consent = await verify_car_import_consent(
                manifest, grant,
                host_did=host_did,
                revoked_grant_ids=revoked_grant_ids,
                did_web_resolver=grant_did_web_resolver,
            )
            consent_grant_id = consent.canonical_grant_id
            if not consent.ok:
                consent_reason = f"consent verification failed: {consent.reason}"
                self.logger.warning(
                    f"⛔ Sovereignty import rejected ({consent_reason}); "
                    f"host DB untouched"
                )
                await self._log_import_attempt(
                    agent_did=agent_did, source_did=src_did,
                    package_hash=package_hash,
                    continuity_score=continuity.overall_score,
                    status="rejected", reject_reason=consent_reason,
                    messages_restored=0, shards_restored=0,
                    manifest_version=manifest_version, source_cid=source_cid,
                    grant_id=consent_grant_id,
                )
                return SovereignImportResult(
                    success=False, agent_did=agent_did,
                    package_hash=package_hash, continuity=continuity,
                    status="rejected", reject_reason=consent_reason,
                    manifest_version=manifest_version,
                    timestamp=(manifest.timestamp if manifest else ""),
                    source_cid=source_cid,
                )
            if host_policy is not None and not host_policy(
                grant, consent.canonical_grant_id
            ):
                consent_reason = (
                    f"{REJECT_HOST_POLICY}: host policy rejected an "
                    f"otherwise-valid grant "
                    f"{consent.canonical_grant_id[:16]}"
                )
                self.logger.warning(
                    f"⛔ Sovereignty import rejected ({consent_reason}); "
                    f"host DB untouched"
                )
                await self._log_import_attempt(
                    agent_did=agent_did, source_did=src_did,
                    package_hash=package_hash,
                    continuity_score=continuity.overall_score,
                    status="rejected", reject_reason=consent_reason,
                    messages_restored=0, shards_restored=0,
                    manifest_version=manifest_version, source_cid=source_cid,
                    grant_id=consent_grant_id,
                )
                return SovereignImportResult(
                    success=False, agent_did=agent_did,
                    package_hash=package_hash, continuity=continuity,
                    status="rejected", reject_reason=consent_reason,
                    manifest_version=manifest_version,
                    timestamp=(manifest.timestamp if manifest else ""),
                    source_cid=source_cid,
                )

        asset_payload_counts: Dict[str, int] = {}
        asset_payloads_skipped: List[Dict[str, Any]] = []
        try:
            if manifest is None or reader is None:
                raise RuntimeError("package has no readable manifest")

            keyring_cipher = reader.get_block(manifest.keyring_cid)
            if keyring_cipher is None:
                raise ValueError(f"Keyring block {manifest.keyring_cid} not in CAR")
            keyring = self._decrypt_keyring(keyring_cipher)

            all_conversations: List[Dict] = []
            for shard_meta in manifest.shards:
                shard_cipher = reader.get_block(shard_meta.cid)
                if shard_cipher is None:
                    raise ValueError(f"Shard block {shard_meta.cid} not in CAR")
                shard_key_hex = keyring.get(shard_meta.shard_id)
                if not shard_key_hex:
                    raise ValueError(f"Missing key for shard {shard_meta.shard_id}")
                shard_json = self.encryptor.decrypt(
                    shard_cipher, bytes.fromhex(shard_key_hex)
                )
                all_conversations.extend(json.loads(shard_json.decode("utf-8")))

            await self.db.execute(
                "DELETE FROM conversation_history WHERE agent_id = ?",
                (self.agent_id,),
            )
            for msg in sorted(all_conversations, key=lambda m: m.get("id", 0)):
                metadata_json = json.dumps(msg.get("metadata", {}))
                await self.db.execute(
                    f"INSERT INTO conversation_history (agent_id, role, content, metadata, created_at) VALUES (?, ?, ?, ?, {self._now_sql()})",
                    (self.agent_id, msg["role"], msg["content"], metadata_json),
                )

            # Asset restoration (#1391) — runs AFTER conversation
            # restore so a restorer failure surfaces against an
            # already-restored conversation set. Pre-#1391 callers
            # (no ``asset_restorers``) skip this block entirely; the
            # metadata-only ``assets_restored`` field still populates
            # from the manifest below, matching the old behavior.
            if asset_restorers and manifest.assets:
                asset_payload_counts, asset_payloads_skipped = (
                    await self._restore_assets(
                        manifest=manifest,
                        reader=reader,
                        keyring=keyring,
                        asset_restorers=asset_restorers,
                    )
                )
        except Exception as e:
            await self._log_import_attempt(
                agent_did=agent_did, source_did=src_did,
                package_hash=package_hash,
                continuity_score=continuity.overall_score,
                status="error", reject_reason=f"restore_failed: {e}",
                messages_restored=0, shards_restored=0,
                manifest_version=manifest_version, source_cid=source_cid,
                grant_id=consent_grant_id,
            )
            raise

        await self._log_import_attempt(
            agent_did=agent_did, source_did=src_did,
            package_hash=package_hash,
            continuity_score=continuity.overall_score,
            status="imported", reject_reason=None,
            messages_restored=len(all_conversations),
            shards_restored=len(manifest.shards),
            manifest_version=manifest_version, source_cid=source_cid,
            grant_id=consent_grant_id,
        )
        asset_payload_total = sum(asset_payload_counts.values())
        self.logger.info(
            f"✅ Sovereignty import complete: {len(all_conversations)} messages, "
            f"{len(manifest.assets)} assets (continuity {continuity.overall_score:.3f}, "
            f"{asset_payload_total} asset payload(s) restored, "
            f"{len(asset_payloads_skipped)} skipped)"
        )
        return SovereignImportResult(
            success=True, agent_did=agent_did, package_hash=package_hash,
            continuity=continuity, status="imported",
            manifest_version=manifest_version,
            messages_restored=len(all_conversations),
            shards_restored=len(manifest.shards),
            assets_restored=[asdict(a) for a in manifest.assets],
            asset_payload_counts=asset_payload_counts,
            asset_payloads_skipped=asset_payloads_skipped,
            timestamp=manifest.timestamp, source_cid=source_cid,
        )

    # ------------------------------------------------------------------
    # Asset restoration (#1391)
    # ------------------------------------------------------------------

    async def _restore_assets(
        self,
        *,
        manifest: RootManifest,
        reader: CARReader,
        keyring: Dict[str, str],
        asset_restorers: List[AssetRestorer],
    ) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
        """Walk every asset in *manifest*, decrypt, route to restorers.

        Returns ``(payload_counts, skipped)``:

          * ``payload_counts`` — per ``asset_type`` count of payloads
            successfully restored. ``{"fhir_resource": 12, "avatar": 1}``.
          * ``skipped`` — list of ``{asset_key, asset_type, reason}``
            entries for assets the adapter couldn't hand off (external
            refs, missing block, no keyring entry where one was
            required). Surface-only — the import still succeeds.

        Raises whatever the restorer raises. By the time a restorer
        sees its assets, conversations have already been restored; an
        asset-side failure is auditable but not transactionally
        rollback-able under the current SQLite schema.
        """
        # Pre-compute the restorer-by-asset_type fan-out so we can
        # short-circuit assets nobody handles.
        restorers_for: Dict[str, List[AssetRestorer]] = {}
        for r in asset_restorers:
            types = r.asset_types or []
            for t in types:
                restorers_for.setdefault(t, []).append(r)

        # Group payloads by (restorer, asset_type) so each restorer
        # sees one batched call per type it claims.
        grouped: Dict[
            Tuple[int, str], List[Tuple[AssetMetadata, bytes]]
        ] = {}
        skipped: List[Dict[str, Any]] = []
        for asset in manifest.assets:
            restorers = restorers_for.get(asset.asset_type)
            if not restorers:
                # No restorer registered for this type — surface as
                # metadata-only so the caller can see what was passed
                # over.
                skipped.append({
                    "asset_key": asset.asset_key,
                    "asset_type": asset.asset_type,
                    "reason": "no_restorer_for_type",
                })
                continue

            block = reader.get_block(asset.cid)
            if block is None:
                # An external-ref block IS reachable via reader (it's
                # a dag-cbor link node), but we ALSO land here when the
                # asset's CID just isn't in the CAR. Either way, no
                # bytes to hand off — skip.
                skipped.append({
                    "asset_key": asset.asset_key,
                    "asset_type": asset.asset_type,
                    "reason": "asset_block_missing_from_car",
                })
                continue

            # External-ref detection: an external-ref block is a
            # dag-cbor link node shaped ``{"link": …, "type": …}``. A
            # raw-bytes asset block isn't cbor-encoded at all, so
            # ``get_dag_cbor_block`` raises rather than returning
            # None. Both paths land here as "not an external ref" —
            # only a successful decode + the link-shape signature
            # confirms.
            try:
                link_obj = reader.get_dag_cbor_block(asset.cid)
            except Exception:
                link_obj = None
            if isinstance(link_obj, dict) and "link" in link_obj:
                # Phase-2 follow-up will fetch via the filecoin
                # adapter; for now, skip with a structured reason.
                skipped.append({
                    "asset_key": asset.asset_key,
                    "asset_type": asset.asset_type,
                    "reason": "external_ref_not_yet_supported",
                })
                continue

            keyring_key_hex = keyring.get(f"asset_{asset.asset_key}")
            if keyring_key_hex:
                # Adapter-encrypted: decrypt with the per-asset key.
                try:
                    payload = self.encryptor.decrypt(
                        block, bytes.fromhex(keyring_key_hex),
                    )
                except Exception as e:
                    skipped.append({
                        "asset_key": asset.asset_key,
                        "asset_type": asset.asset_type,
                        "reason": f"asset_decrypt_failed: {e}",
                    })
                    continue
            else:
                # Caller pre-encrypted (encrypted=True on the
                # exported descriptor) — hand raw block bytes
                # through; the restorer decrypts in its own scheme.
                payload = block

            for r in restorers:
                grouped.setdefault(
                    (id(r), asset.asset_type), [],
                ).append((asset, payload))

        # Resolve restorer-id back to the restorer instance for the
        # call. Two restorers with the same id() can't co-exist (CPython
        # guarantees uniqueness for live objects), so this is stable.
        by_id = {id(r): r for r in asset_restorers}

        payload_counts: Dict[str, int] = {}
        for (restorer_id, asset_type), batch in grouped.items():
            restorer = by_id[restorer_id]
            restored_n = await restorer.restore_assets(
                manifest.agent_did, batch,
            )
            if not isinstance(restored_n, int):
                # Defensive: a buggy restorer returning None should not
                # silently zero out the count.
                restored_n = len(batch)
            payload_counts[asset_type] = (
                payload_counts.get(asset_type, 0) + restored_n
            )

        return payload_counts, skipped

    # ------------------------------------------------------------------
    # Append-only import audit log
    # ------------------------------------------------------------------

    def _import_log_pk(self) -> str:
        if self.db.backend_type == "postgres":
            return "BIGSERIAL PRIMARY KEY"
        return "INTEGER PRIMARY KEY AUTOINCREMENT"

    def _import_log_ts_col(self) -> str:
        if self.db.backend_type == "postgres":
            return "TIMESTAMPTZ DEFAULT NOW()"
        return "TEXT DEFAULT (datetime('now'))"

    async def _ensure_import_log(self) -> None:
        await self.db.execute(
            f"""
            CREATE TABLE IF NOT EXISTS agent_import_log (
                id {self._import_log_pk()},
                agent_did TEXT NOT NULL,
                source_did TEXT NOT NULL,
                host_agent_id TEXT,
                package_hash TEXT NOT NULL,
                continuity_score REAL,
                status TEXT NOT NULL,
                reject_reason TEXT,
                manifest_version TEXT,
                messages_restored INTEGER DEFAULT 0,
                shards_restored INTEGER DEFAULT 0,
                source_cid TEXT,
                grant_id TEXT,
                created_at {self._import_log_ts_col()}
            )
            """
        )
        await self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_import_log_agent ON agent_import_log(agent_did)"
        )
        await self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_import_log_source ON agent_import_log(source_did)"
        )
        # Additive migration for DBs created before #1379 introduced
        # the grant_id column. Both backends tolerate ADD COLUMN IF NOT
        # EXISTS, but SQLite older than 3.35 doesn't — so feature-check.
        try:
            await self.db.execute(
                "ALTER TABLE agent_import_log ADD COLUMN grant_id TEXT"
            )
        except Exception:
            # Column already exists (or backend refuses) — both are
            # benign for an additive migration.
            pass

    async def _log_import_attempt(
        self, *, agent_did: str, source_did: str, package_hash: str,
        continuity_score: float, status: str, reject_reason: Optional[str],
        messages_restored: int, shards_restored: int,
        manifest_version: str, source_cid: Optional[str],
        grant_id: Optional[str] = None,
    ) -> None:
        """Append exactly one row per import attempt.

        Append-only — there is deliberately no UPDATE/DELETE path. A
        failure to write the audit row must never mask the import
        outcome, so write failures are logged and swallowed.
        """
        try:
            await self._ensure_import_log()
            await self.db.execute(
                """
                INSERT INTO agent_import_log
                  (agent_did, source_did, host_agent_id, package_hash,
                   continuity_score, status, reject_reason, manifest_version,
                   messages_restored, shards_restored, source_cid, grant_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (agent_did, source_did, self.agent_id, package_hash,
                 continuity_score, status, reject_reason, manifest_version,
                 messages_restored, shards_restored, source_cid, grant_id),
            )
        except Exception as e:
            self.logger.error(f"Failed to write agent_import_log row: {e}")

    async def get_import_log(
        self, agent_did: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Read the append-only import log (newest first)."""
        await self._ensure_import_log()
        cols = [
            "id", "agent_did", "source_did", "host_agent_id", "package_hash",
            "continuity_score", "status", "reject_reason", "manifest_version",
            "messages_restored", "shards_restored", "source_cid", "grant_id",
            "created_at",
        ]
        col_sql = ", ".join(cols)
        if agent_did is None:
            rows = await self.db.fetchall(
                f"SELECT {col_sql} FROM agent_import_log ORDER BY id DESC LIMIT ?",
                (limit,),
            )
        else:
            rows = await self.db.fetchall(
                f"SELECT {col_sql} FROM agent_import_log WHERE agent_did = ? ORDER BY id DESC LIMIT ?",
                (agent_did, limit),
            )
        return [dict(zip(cols, r)) for r in rows]


class SovereignImportVerifier:
    """Derive a 0–1 continuity/integrity score for a static sovereignty CAR.

    The interactive ``ContinuityVerifier`` (identity package) needs a
    live respondent and an ``AgentIdentityPackage``; a sovereignty CAR
    has neither — it is a static archive of conversation + asset shards.
    This verifier instead scores statically checkable properties:

      * ``car_blocks_verified``  — every block hashes to its CID (critical)
      * ``manifest_wellformed``  — root dag-cbor decodes to a RootManifest (critical)
      * ``agent_did_present``    — manifest carries an agent DID
      * ``keyring_decrypts``     — keyring decrypts (ownership proxy: only
                                   the holder of the user secret can)
      * ``shards_decrypt``       — every shard decrypts under its key

    The two critical checks hard-reject (distinct structured reason)
    regardless of the weighted score.
    """

    def __init__(self, encryptor: ConvergentEncryptor) -> None:
        self._enc = encryptor

    def _decrypt_keyring(self, keyring_cipher: bytes) -> Dict[str, str]:
        keyring_key = self._enc.derive_key(b"KESTREL_KEYRING_V2")
        aesgcm = AESGCM(keyring_key)
        nonce = keyring_cipher[:12]
        return json.loads(
            aesgcm.decrypt(nonce, keyring_cipher[12:], None).decode("utf-8")
        )

    def verify(
        self, car_bytes: bytes
    ) -> Tuple[ImportContinuity, Optional[str], Optional[RootManifest], Optional[CARReader]]:
        checks: List[ImportCheck] = []
        reject_reason: Optional[str] = None
        reader: Optional[CARReader] = None
        manifest: Optional[RootManifest] = None

        car_ok = False
        try:
            reader = CARReader(car_bytes)
            car_ok = bool(reader.verify())
        except Exception:
            car_ok = False
        checks.append(ImportCheck(
            "car_blocks_verified", car_ok, 0.35,
            "" if car_ok else "CAR block-hash verification failed",
        ))
        if not car_ok:
            reject_reason = "car_verification_failed"

        man_ok = False
        if reader is not None and reject_reason is None:
            try:
                md = reader.get_dag_cbor_block(reader.root_cid)
                if md is None:
                    raise ValueError("root manifest block missing")
                manifest = SovereignStorageAdapter._manifest_from_dict(md)
                man_ok = True
            except Exception:
                man_ok = False
        checks.append(ImportCheck(
            "manifest_wellformed", man_ok, 0.20,
            "" if man_ok else "root manifest missing or malformed",
        ))
        if reject_reason is None and not man_ok:
            reject_reason = "manifest_missing"

        did_ok = bool(manifest and manifest.agent_did)
        checks.append(ImportCheck(
            "agent_did_present", did_ok, 0.10,
            "" if did_ok else "manifest has no agent_did",
        ))

        keyring: Optional[Dict[str, str]] = None
        kr_ok = False
        if man_ok and manifest is not None and reader is not None:
            try:
                kc = reader.get_block(manifest.keyring_cid)
                if kc is None:
                    raise ValueError("keyring block missing")
                keyring = self._decrypt_keyring(kc)
                kr_ok = isinstance(keyring, dict)
            except Exception:
                kr_ok = False
        checks.append(ImportCheck(
            "keyring_decrypts", kr_ok, 0.20,
            "" if kr_ok else "keyring undecryptable (wrong owner secret or tampered)",
        ))
        if reject_reason is None and man_ok and not kr_ok:
            reject_reason = "keyring_decrypt_failed"

        shard_total = len(manifest.shards) if manifest else 0
        shard_ok = 0
        if keyring is not None and manifest is not None and reader is not None:
            for sm in manifest.shards:
                try:
                    sc = reader.get_block(sm.cid)
                    kh = keyring.get(sm.shard_id)
                    if sc is None or not kh:
                        continue
                    self._enc.decrypt(sc, bytes.fromhex(kh))
                    shard_ok += 1
                except Exception:
                    continue
        shards_pass = shard_total == 0 or shard_ok == shard_total
        checks.append(ImportCheck(
            "shards_decrypt", shards_pass, 0.15,
            f"{shard_ok}/{shard_total} shards decrypted",
        ))
        if reject_reason is None and shard_total > 0 and shard_ok < shard_total:
            # Any shard that fails to resolve/decrypt is a hard reject:
            # otherwise a partially-broken multi-shard package scores
            # above threshold and then raises mid-restore instead of
            # returning the structured rejection the receiver promises.
            reject_reason = (
                "no_shards_decrypted" if shard_ok == 0 else "incomplete_shards"
            )

        total_w = sum(c.weight for c in checks) or 1.0
        score = sum(c.weight for c in checks if c.passed) / total_w
        return (
            ImportContinuity(overall_score=round(score, 4), checks=checks),
            reject_reason,
            manifest,
            reader,
        )
