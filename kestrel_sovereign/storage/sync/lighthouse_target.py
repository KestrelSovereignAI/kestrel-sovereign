"""
Lighthouse Sync Target

Lighthouse (Filecoin) decentralized storage target with CID tracking
and cold-start restore support.
"""

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from kestrel_sovereign.storage.car_builder import CARBuilder, CARReader
from kestrel_sovereign.storage.sync.manifest_manager import ManifestManagerMixin
from kestrel_sovereign.storage.sync.targets import (
    SyncTarget,
    SyncResult,
    TrustTier,
    _create_consistent_snapshot,
)

logger = logging.getLogger(__name__)


class LighthouseTarget(ManifestManagerMixin, SyncTarget):
    """
    Lighthouse (Filecoin) storage target.

    Stores database snapshots and WAL on decentralized storage
    for maximum data sovereignty. Supports cold-start restore for
    ephemeral environments (Cloud Run scale-to-zero).

    CID tracking: After each snapshot upload, a manifest file is uploaded
    containing the snapshot CID. The manifest CID is stored locally and
    can also be found via the Lighthouse uploads API.
    """

    # Tag prefix for filtering uploads via Lighthouse API
    SNAPSHOT_TAG = "kestrel-state"
    MANIFEST_TAG = "kestrel-manifest"
    SNAPSHOT_FORMAT_CAR_V1 = "car-v1/raw-sqlite"

    def __init__(
        self,
        api_key: str,
        agent_id: str,
        state_dir: Optional[Path] = None,
    ):
        """
        Initialize Lighthouse target.

        Args:
            api_key: Lighthouse API key
            agent_id: Unique agent identifier for namespacing
            state_dir: Directory to store local state (latest CID tracking).
                       Defaults to parent of the DB path.
        """
        self.api_key = api_key
        self.agent_id = agent_id
        self._state_dir = state_dir
        self._manifest_filename = f".lighthouse_manifest_{self.agent_id}.json"
        self._latest_cid: Optional[str] = None

    @property
    def name(self) -> str:
        return f"lighthouse://{self.agent_id}"

    @property
    def trust_tier(self) -> TrustTier:
        return TrustTier.DELEGATED

    async def sync_snapshot(self, db_path: Path) -> SyncResult:
        """Upload snapshot to Lighthouse and update manifest.

        Uses sqlite3.backup() for a consistent snapshot and skips upload
        if the content hash matches the previous upload.
        """
        timestamp = datetime.now(timezone.utc)

        try:
            from kestrel_sovereign.storage.providers.lighthouse_rest import LighthouseRestClient

            # Use sqlite3.backup() for consistent snapshot (safe with active WAL)
            content = _create_consistent_snapshot(db_path)

            # Skip upload if content unchanged
            content_hash = hashlib.sha256(content).hexdigest()
            manifest = self._load_local_manifest()
            if manifest and manifest.get("content_hash") == content_hash:
                logger.debug(f"Snapshot unchanged (hash={content_hash[:12]}), skipping upload")
                return SyncResult(
                    success=True,
                    target_name=self.name,
                    bytes_synced=0,
                    frames_synced=0,
                    timestamp=timestamp,
                    metadata={"cid": manifest.get("snapshot_cid"), "skipped": True},
                )

            client = LighthouseRestClient(api_key=self.api_key)

            tag = f"{self.SNAPSHOT_TAG}-{self.agent_id}"
            car_bytes, payload_cid = self._build_snapshot_car(content)
            result = await client.upload_car(car_bytes=car_bytes, tag=tag)

            cid = result.get("Hash") or result.get("cid")
            size = int(result.get("Size", len(car_bytes)))

            if not cid:
                raise ValueError(f"No CID returned from Lighthouse upload: {result}")

            logger.info(f"Uploaded snapshot to Lighthouse: {cid} ({size} bytes)")
            self._latest_cid = cid

            # Upload manifest pointing to this snapshot
            new_manifest = {
                "agent_id": self.agent_id,
                "snapshot_cid": cid,
                "snapshot_size": size,
                "snapshot_format": self.SNAPSHOT_FORMAT_CAR_V1,
                "snapshot_payload_cid": payload_cid,
                "raw_snapshot_size": len(content),
                "uploaded_at": timestamp.isoformat(),
                "source_file": db_path.name,
                "content_hash": content_hash,
            }
            await self._upload_manifest(client, new_manifest)
            await client.close()

            return SyncResult(
                success=True,
                target_name=self.name,
                bytes_synced=size,
                frames_synced=0,
                timestamp=timestamp,
                metadata={"cid": cid, "format": self.SNAPSHOT_FORMAT_CAR_V1},
            )

        except Exception as e:
            logger.error(f"Failed to sync to Lighthouse: {e}")
            return SyncResult(
                success=False,
                target_name=self.name,
                bytes_synced=0,
                frames_synced=0,
                timestamp=timestamp,
                error=str(e),
            )

    async def _upload_manifest(self, client: Any, manifest: Dict[str, Any]) -> None:
        """Upload manifest file to Lighthouse and save locally."""
        try:
            manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")
            tag = f"{self.MANIFEST_TAG}-{self.agent_id}"

            result = await client.upload(
                content=manifest_bytes,
                filename=f"manifest_{self.agent_id}.json",
                tag=tag,
            )

            manifest_cid = result.get("Hash") or result.get("cid")
            manifest["manifest_cid"] = manifest_cid
            self._save_local_manifest(manifest)
            logger.debug(f"Uploaded manifest to Lighthouse: {manifest_cid}")

        except Exception as e:
            # Manifest upload failure is non-fatal -- snapshot is already safe
            logger.warning(f"Failed to upload manifest (snapshot is safe): {e}")
            self._save_local_manifest(manifest)

    async def restore_snapshot(self, dest_path: Path) -> Optional[SyncResult]:
        """
        Download the latest snapshot from Lighthouse and write to dest_path.

        Resolution order for finding the latest snapshot CID:
        1. LIGHTHOUSE_STATE_CID env var (explicit pointer)
        2. Local manifest file (from previous sync on this host)
        3. Lighthouse uploads API (search by tag)

        Args:
            dest_path: Where to write the restored database file

        Returns:
            SyncResult if restore succeeded, None if no snapshot found
        """
        timestamp = datetime.now(timezone.utc)

        try:
            snapshot_cid = await self._resolve_latest_cid()
            if not snapshot_cid:
                logger.info("No Lighthouse snapshot found to restore")
                return None

            logger.info(f"Restoring snapshot from Lighthouse: {snapshot_cid}")

            from kestrel_sovereign.storage.providers.lighthouse_rest import LighthouseRestClient

            client = LighthouseRestClient(api_key=self.api_key)
            content = await client.download(snapshot_cid)
            await client.close()

            if not content:
                logger.warning("Empty snapshot downloaded from Lighthouse")
                return None

            content = self._extract_snapshot_content(content)

            # Write to destination
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dest_path, "wb") as f:
                f.write(content)

            logger.info(
                f"Restored {len(content)} bytes from Lighthouse to {dest_path}"
            )

            return SyncResult(
                success=True,
                target_name=self.name,
                bytes_synced=len(content),
                frames_synced=0,
                timestamp=timestamp,
                metadata={"cid": snapshot_cid, "action": "restore"},
            )

        except Exception as e:
            logger.error(f"Failed to restore from Lighthouse: {e}")
            return SyncResult(
                success=False,
                target_name=self.name,
                bytes_synced=0,
                frames_synced=0,
                timestamp=timestamp,
                error=str(e),
            )

    def _build_snapshot_car(self, content: bytes) -> tuple[bytes, str]:
        """Pack a SQLite snapshot as a single-block CAR archive."""
        builder = CARBuilder()
        payload_cid = builder.add_raw_block(content)
        builder.set_root(payload_cid)
        return builder.build(), payload_cid

    def _extract_snapshot_content(self, content: bytes) -> bytes:
        """Return raw SQLite bytes from CAR snapshots, preserving legacy raw files."""
        try:
            reader = CARReader(content)
            if not reader.verify():
                raise ValueError("CAR verification failed")

            root_block = reader.get_block(reader.root_cid)
            if root_block is None:
                raise ValueError(f"CAR root block missing: {reader.root_cid}")
            return root_block
        except Exception as e:
            logger.debug(
                "Downloaded snapshot is not a CAR archive; treating as raw SQLite: %s",
                e,
            )
            return content

    async def _resolve_latest_cid(self) -> Optional[str]:
        """
        Find the latest snapshot CID using multiple resolution strategies.

        Returns:
            CID string or None
        """
        # 1. Explicit env var
        explicit_cid = os.environ.get("LIGHTHOUSE_STATE_CID")
        if explicit_cid:
            logger.debug(f"Using LIGHTHOUSE_STATE_CID env var: {explicit_cid}")
            return explicit_cid

        # 2. Local manifest
        manifest = self._load_local_manifest()
        if manifest and manifest.get("snapshot_cid"):
            logger.debug(f"Using local manifest CID: {manifest['snapshot_cid']}")
            return manifest["snapshot_cid"]

        # 3. In-memory cache from last sync
        if self._latest_cid:
            return self._latest_cid

        # 4. Lighthouse uploads API
        return await self._query_uploads_api()

    async def _query_uploads_api(self) -> Optional[str]:
        """Query Lighthouse uploads API to find the latest snapshot."""
        try:
            from kestrel_sovereign.storage.providers.lighthouse_rest import LighthouseRestClient

            client = LighthouseRestClient(api_key=self.api_key)
            result = await client.get_uploads()
            await client.close()

            uploads = result.get("fileList", [])
            if not isinstance(uploads, list):
                return None

            # Filter for our agent's snapshots by filename pattern
            tag = f"{self.SNAPSHOT_TAG}-{self.agent_id}"
            agent_snapshots = [
                u for u in uploads
                if u.get("tag") == tag or (
                    u.get("fileName", "").startswith("kestrel")
                    and u.get("fileName", "").endswith(".db")
                )
            ]

            if not agent_snapshots:
                # Try manifest files as fallback
                manifest_tag = f"{self.MANIFEST_TAG}-{self.agent_id}"
                manifests = [u for u in uploads if u.get("tag") == manifest_tag]
                if manifests:
                    # Get the most recent manifest
                    manifests.sort(
                        key=lambda u: u.get("createdAt", ""), reverse=True
                    )
                    manifest_cid = manifests[0].get("cid") or manifests[0].get("Hash")
                    if manifest_cid:
                        return await self._read_manifest_cid(manifest_cid)
                return None

            # Sort by creation time (most recent first)
            agent_snapshots.sort(
                key=lambda u: u.get("createdAt", ""), reverse=True
            )

            cid = agent_snapshots[0].get("cid") or agent_snapshots[0].get("Hash")
            logger.info(f"Found latest snapshot via uploads API: {cid}")
            return cid

        except Exception as e:
            logger.warning(f"Failed to query Lighthouse uploads API: {e}")
            return None

    async def _read_manifest_cid(self, manifest_cid: str) -> Optional[str]:
        """Download a manifest file and extract the snapshot CID."""
        try:
            from kestrel_sovereign.storage.providers.lighthouse_rest import LighthouseRestClient

            client = LighthouseRestClient(api_key=self.api_key)
            content = await client.download(manifest_cid)
            await client.close()
            manifest = json.loads(content)
            return manifest.get("snapshot_cid")
        except Exception as e:
            logger.warning(f"Failed to read manifest {manifest_cid}: {e}")
            return None

    async def sync_wal(self, wal_path: Path, position: int) -> SyncResult:
        """No-op for Lighthouse. WAL sync is not appropriate for content-addressed storage.

        Lighthouse uses full consistent snapshots via sync_snapshot(). Uploading
        the raw WAL file is wasteful (new CID on every write, not recoverable
        without the matching DB, and burns storage quota for no benefit).
        """
        return SyncResult(
            success=True,
            target_name=self.name,
            bytes_synced=0,
            frames_synced=0,
            timestamp=datetime.now(timezone.utc),
        )

    async def get_latest_position(self) -> Optional[int]:
        """Return max int to signal no WAL sync needed.

        Returning None caused SyncService to always sync from position 0.
        Returning a large value ensures the wal_position > last_pos check
        in SyncService._sync_pending() is never true.
        """
        return 2**63

    async def health_check(self) -> bool:
        """Check Lighthouse API connectivity."""
        try:
            from kestrel_sovereign.storage.providers.lighthouse_rest import LighthouseRestClient

            client = LighthouseRestClient(api_key=self.api_key)
            await client.get_balance()
            await client.close()
            return True
        except Exception as e:
            logger.warning(f"Lighthouse health check failed: {e}")
            return False
