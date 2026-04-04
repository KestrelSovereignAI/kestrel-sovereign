"""
Storacha Sync Target

Storacha (web3.storage) sync target using UCAN/DID authentication
for federated decentralized storage.
"""

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from kestrel_sovereign.storage.sync.manifest_manager import ManifestManagerMixin
from kestrel_sovereign.storage.sync.targets import (
    SyncTarget,
    SyncResult,
    TrustTier,
    _create_consistent_snapshot,
)

logger = logging.getLogger(__name__)


class StorachaTarget(ManifestManagerMixin, SyncTarget):
    """
    Storacha (web3.storage) sync target.

    Stores SQLite database snapshots on IPFS via the Storacha w3up protocol,
    using UCAN/DID authentication.  This is the preferred decentralised backup
    target -- the agent's Ed25519 identity keypair is used directly as the UCAN
    signing key, giving cryptographic proof of ownership without opaque API keys.

    Supports cold-start restore for ephemeral environments (Cloud Run scale-to-zero).

    Required environment variables:
        STORACHA_SPACE_DID  -- Space DID (from ``w3 space create``)
        STORACHA_AGENT_KEY  -- Ed25519 agent key (from ``w3 key create``)
        STORACHA_PROOF      -- Base64 delegation proof CAR

    Optional:
        STORACHA_STATE_CID  -- Explicit snapshot CID to restore from (overrides discovery)
    """

    SNAPSHOT_TAG = "kestrel-state"
    MANIFEST_TAG = "kestrel-manifest"

    def __init__(
        self,
        space_did: str,
        agent_key: str,
        proof: str,
        agent_id: str,
        state_dir: Optional[Path] = None,
    ):
        """
        Args:
            space_did:  Storacha space DID
            agent_key:  Ed25519 agent key string (multibase or raw base64)
            proof:      Base64-encoded UCAN delegation CAR
            agent_id:   Unique agent identifier for namespacing uploads
            state_dir:  Directory for local manifest files (defaults to cwd)
        """
        self.agent_id = agent_id
        self._state_dir = state_dir
        self._manifest_filename = f".storacha_manifest_{self.agent_id}.json"
        self._latest_cid: Optional[str] = None

        from kestrel_storage_storacha.storacha_ucan import StorachaUCAN
        from kestrel_storage_storacha.storacha_rest import StorachaRestClient
        from kestrel_sovereign.kestrel_config.defaults import get_storacha_gateway_url

        self._ucan = StorachaUCAN(
            agent_key=agent_key,
            space_did=space_did,
            proof=proof,
        )
        self._client = StorachaRestClient(
            ucan=self._ucan,
            gateway_url=get_storacha_gateway_url(),
        )

    @property
    def name(self) -> str:
        return f"storacha://{self.agent_id}"

    @property
    def trust_tier(self) -> TrustTier:
        return TrustTier.FEDERATED

    async def sync_snapshot(self, db_path: Path) -> SyncResult:
        """
        Upload a consistent SQLite snapshot to Storacha (IPFS/w3up).

        Uses sqlite3.backup() for a safe, consistent snapshot even with an
        active WAL.  Skips upload when content hash is unchanged.
        """
        timestamp = datetime.now(timezone.utc)
        try:
            content = _create_consistent_snapshot(db_path)
            content_hash = hashlib.sha256(content).hexdigest()

            manifest = self._load_local_manifest()
            if manifest and manifest.get("content_hash") == content_hash:
                logger.debug("Storacha snapshot unchanged (%s...), skipping", content_hash[:12])
                return SyncResult(
                    success=True,
                    target_name=self.name,
                    bytes_synced=0,
                    frames_synced=0,
                    timestamp=timestamp,
                    metadata={"cid": manifest.get("snapshot_cid"), "skipped": True},
                )

            result = await self._client.upload(
                content=content,
                filename=db_path.name,
            )
            cid = result["cid"]
            size = result["size"]

            logger.info("Uploaded snapshot to Storacha: %s (%d bytes)", cid, size)
            self._latest_cid = cid

            new_manifest: Dict[str, Any] = {
                "agent_id": self.agent_id,
                "snapshot_cid": cid,
                "snapshot_size": size,
                "uploaded_at": timestamp.isoformat(),
                "source_file": db_path.name,
                "content_hash": content_hash,
            }
            await self._upload_manifest(new_manifest)

            return SyncResult(
                success=True,
                target_name=self.name,
                bytes_synced=size,
                frames_synced=0,
                timestamp=timestamp,
                metadata={"cid": cid},
            )

        except Exception as e:
            logger.error("Failed to sync to Storacha: %s", e)
            return SyncResult(
                success=False,
                target_name=self.name,
                bytes_synced=0,
                frames_synced=0,
                timestamp=timestamp,
                error=str(e),
            )

    async def _upload_manifest(self, manifest: Dict[str, Any]) -> None:
        """Upload manifest JSON to Storacha and persist locally."""
        try:
            manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")
            result = await self._client.upload(
                content=manifest_bytes,
                filename=f"manifest_{self.agent_id}.json",
            )
            manifest["manifest_cid"] = result["cid"]
            self._save_local_manifest(manifest)
            logger.debug("Uploaded Storacha manifest: %s", result["cid"])
        except Exception as e:
            # Manifest upload is non-fatal -- snapshot is already safe on IPFS
            logger.warning("Failed to upload Storacha manifest (snapshot is safe): %s", e)
            self._save_local_manifest(manifest)

    async def restore_snapshot(self, dest_path: Path) -> Optional[SyncResult]:
        """
        Download the latest snapshot from Storacha and write to dest_path.

        CID resolution order:
        1. STORACHA_STATE_CID env var (explicit override)
        2. Local manifest file
        3. In-memory cache from last sync
        4. Storacha upload/list API query
        """
        timestamp = datetime.now(timezone.utc)
        try:
            snapshot_cid = await self._resolve_latest_cid()
            if not snapshot_cid:
                logger.info("No Storacha snapshot found to restore")
                return None

            logger.info("Restoring snapshot from Storacha: %s", snapshot_cid)
            content = await self._client.get_by_cid(snapshot_cid)

            if not content:
                logger.warning("Empty snapshot downloaded from Storacha")
                return None

            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dest_path, "wb") as f:
                f.write(content)

            logger.info("Restored %d bytes from Storacha to %s", len(content), dest_path)
            return SyncResult(
                success=True,
                target_name=self.name,
                bytes_synced=len(content),
                frames_synced=0,
                timestamp=timestamp,
                metadata={"cid": snapshot_cid, "action": "restore"},
            )

        except Exception as e:
            logger.error("Failed to restore from Storacha: %s", e)
            return SyncResult(
                success=False,
                target_name=self.name,
                bytes_synced=0,
                frames_synced=0,
                timestamp=timestamp,
                error=str(e),
            )

    async def _resolve_latest_cid(self) -> Optional[str]:
        """Find the latest snapshot CID."""
        # 1. Explicit env var override
        explicit = os.environ.get("STORACHA_STATE_CID")
        if explicit:
            logger.debug("Using STORACHA_STATE_CID env var: %s", explicit)
            return explicit

        # 2. Local manifest
        manifest = self._load_local_manifest()
        if manifest and manifest.get("snapshot_cid"):
            logger.debug("Using local Storacha manifest CID: %s", manifest["snapshot_cid"])
            return manifest["snapshot_cid"]

        # 3. In-memory from last sync
        if self._latest_cid:
            return self._latest_cid

        # 4. Query Storacha upload/list
        return await self._query_uploads_api()

    async def _query_uploads_api(self) -> Optional[str]:
        """Query Storacha upload/list to find the latest snapshot."""
        try:
            result = await self._client.list_uploads(size=50)
            uploads = result.get("results", [])
            if not uploads:
                return None

            # Look for manifest uploads (they reference snapshot CIDs)
            for upload in uploads:
                cid = upload.get("root", {}).get("/") if isinstance(upload.get("root"), dict) else None
                if not cid:
                    continue
                try:
                    content = await self._client.get_by_cid(str(cid))
                    manifest = json.loads(content)
                    if manifest.get("agent_id") == self.agent_id and manifest.get("snapshot_cid"):
                        logger.info("Found snapshot via Storacha upload list: %s", manifest["snapshot_cid"])
                        return manifest["snapshot_cid"]
                except Exception:
                    continue
            return None
        except Exception as e:
            logger.warning("Failed to query Storacha upload list: %s", e)
            return None

    async def sync_wal(self, wal_path: Path, position: int) -> SyncResult:
        """No-op -- content-addressed storage uses full snapshots, not WAL deltas."""
        return SyncResult(
            success=True,
            target_name=self.name,
            bytes_synced=0,
            frames_synced=0,
            timestamp=datetime.now(timezone.utc),
        )

    async def get_latest_position(self) -> Optional[int]:
        """Signal that WAL sync is not needed."""
        return 2**63

    async def health_check(self) -> bool:
        """Check Storacha bridge connectivity via a minimal list invocation."""
        try:
            await self._client.get_usage()
            return True
        except Exception as e:
            logger.warning("Storacha health check failed: %s", e)
            return False
