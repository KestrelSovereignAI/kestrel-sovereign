"""
Sync Targets

Abstractions for sync destinations. SQLite changes can be replicated
to various targets including cloud storage and PostgreSQL.
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class SyncResult:
    """Result of a sync operation."""
    success: bool
    target_name: str
    bytes_synced: int
    frames_synced: int
    timestamp: datetime
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class SyncTarget(ABC):
    """Abstract base class for sync targets."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Target name for logging and identification."""
        ...

    @abstractmethod
    async def sync_snapshot(self, db_path: Path) -> SyncResult:
        """
        Sync a full database snapshot.

        Args:
            db_path: Path to SQLite database file

        Returns:
            SyncResult with status
        """
        ...

    @abstractmethod
    async def sync_wal(self, wal_path: Path, position: int) -> SyncResult:
        """
        Sync WAL changes from a specific position.

        Args:
            wal_path: Path to WAL file
            position: Byte offset to start from

        Returns:
            SyncResult with status
        """
        ...

    @abstractmethod
    async def get_latest_position(self) -> Optional[int]:
        """
        Get the latest synced WAL position.

        Returns:
            Byte offset of last sync, or None if no previous sync
        """
        ...

    async def health_check(self) -> bool:
        """Check if target is reachable and healthy."""
        return True


class S3Target(SyncTarget):
    """
    Amazon S3 / S3-compatible storage target.

    Stores database snapshots and WAL segments in S3,
    compatible with Litestream's format.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "kestrel/",
        region: str = "us-east-1",
        endpoint_url: Optional[str] = None,
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
    ):
        """
        Initialize S3 target.

        Args:
            bucket: S3 bucket name
            prefix: Key prefix for all objects
            region: AWS region
            endpoint_url: Custom endpoint for S3-compatible services
            access_key_id: AWS access key (uses env var if not provided)
            secret_access_key: AWS secret key (uses env var if not provided)
        """
        self.bucket = bucket
        self.prefix = prefix.rstrip("/") + "/"
        self.region = region
        self.endpoint_url = endpoint_url
        self._access_key_id = access_key_id
        self._secret_access_key = secret_access_key
        self._client = None

    @property
    def name(self) -> str:
        return f"s3://{self.bucket}/{self.prefix}"

    async def _get_client(self):
        """Get or create S3 client."""
        if self._client is None:
            try:
                import aioboto3
            except ImportError:
                raise ImportError(
                    "aioboto3 is required for S3 sync. Install with: pip install aioboto3"
                )

            session = aioboto3.Session()
            kwargs = {"region_name": self.region}
            if self.endpoint_url:
                kwargs["endpoint_url"] = self.endpoint_url
            if self._access_key_id:
                kwargs["aws_access_key_id"] = self._access_key_id
            if self._secret_access_key:
                kwargs["aws_secret_access_key"] = self._secret_access_key

            self._client = await session.client("s3", **kwargs).__aenter__()
        return self._client

    async def sync_snapshot(self, db_path: Path) -> SyncResult:
        """Upload full database snapshot to S3."""
        timestamp = datetime.now(timezone.utc)

        try:
            client = await self._get_client()

            # Read database file
            with open(db_path, "rb") as f:
                data = f.read()

            # Upload with timestamp in key
            key = f"{self.prefix}snapshots/{timestamp.strftime('%Y%m%d_%H%M%S')}.db"
            await client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=data,
                Metadata={"source_path": str(db_path)},
            )

            # Update latest pointer
            await client.put_object(
                Bucket=self.bucket,
                Key=f"{self.prefix}latest_snapshot",
                Body=key.encode(),
            )

            logger.info(f"Uploaded snapshot to {self.name}{key}")

            return SyncResult(
                success=True,
                target_name=self.name,
                bytes_synced=len(data),
                frames_synced=0,
                timestamp=timestamp,
                metadata={"key": key},
            )

        except Exception as e:
            logger.error(f"Failed to sync snapshot to S3: {e}")
            return SyncResult(
                success=False,
                target_name=self.name,
                bytes_synced=0,
                frames_synced=0,
                timestamp=timestamp,
                error=str(e),
            )

    async def sync_wal(self, wal_path: Path, position: int) -> SyncResult:
        """Upload WAL segment to S3."""
        timestamp = datetime.now(timezone.utc)

        try:
            client = await self._get_client()

            # Read WAL from position
            with open(wal_path, "rb") as f:
                f.seek(position)
                data = f.read()

            if not data:
                return SyncResult(
                    success=True,
                    target_name=self.name,
                    bytes_synced=0,
                    frames_synced=0,
                    timestamp=timestamp,
                )

            # Upload WAL segment
            key = f"{self.prefix}wal/{timestamp.strftime('%Y%m%d_%H%M%S')}_{position}.wal"
            await client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=data,
                Metadata={
                    "position": str(position),
                    "size": str(len(data)),
                },
            )

            # Update position marker
            new_position = position + len(data)
            await client.put_object(
                Bucket=self.bucket,
                Key=f"{self.prefix}wal_position",
                Body=str(new_position).encode(),
            )

            logger.debug(f"Uploaded WAL segment to {self.name}{key}")

            return SyncResult(
                success=True,
                target_name=self.name,
                bytes_synced=len(data),
                frames_synced=0,  # Would need frame parsing
                timestamp=timestamp,
                metadata={"key": key, "new_position": new_position},
            )

        except Exception as e:
            logger.error(f"Failed to sync WAL to S3: {e}")
            return SyncResult(
                success=False,
                target_name=self.name,
                bytes_synced=0,
                frames_synced=0,
                timestamp=timestamp,
                error=str(e),
            )

    async def get_latest_position(self) -> Optional[int]:
        """Get latest WAL position from S3."""
        try:
            client = await self._get_client()
            response = await client.get_object(
                Bucket=self.bucket,
                Key=f"{self.prefix}wal_position",
            )
            data = await response["Body"].read()
            return int(data.decode())
        except client.exceptions.NoSuchKey:
            return None
        except Exception as e:
            logger.warning(f"Failed to get WAL position from S3: {e}")
            return None

    async def health_check(self) -> bool:
        """Check S3 connectivity."""
        try:
            client = await self._get_client()
            await client.head_bucket(Bucket=self.bucket)
            return True
        except Exception as e:
            logger.warning(f"S3 health check failed: {e}")
            return False


class LighthouseTarget(SyncTarget):
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
        self._latest_cid: Optional[str] = None

    @property
    def name(self) -> str:
        return f"lighthouse://{self.agent_id}"

    @property
    def _manifest_path(self) -> Optional[Path]:
        """Path to local manifest file tracking the latest snapshot CID."""
        if self._state_dir:
            return self._state_dir / f".lighthouse_manifest_{self.agent_id}.json"
        return None

    def _load_local_manifest(self) -> Optional[Dict[str, Any]]:
        """Load the local manifest if it exists."""
        path = self._manifest_path
        if path and path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load local manifest: {e}")
        return None

    def _save_local_manifest(self, manifest: Dict[str, Any]) -> None:
        """Save manifest locally for quick CID lookup."""
        path = self._manifest_path
        if path:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(manifest, f, indent=2)
            except Exception as e:
                logger.warning(f"Failed to save local manifest: {e}")

    async def sync_snapshot(self, db_path: Path) -> SyncResult:
        """Upload snapshot to Lighthouse and update manifest."""
        timestamp = datetime.now(timezone.utc)

        try:
            from kestrel_sovereign.storage.providers.lighthouse_rest import LighthouseRestClient

            client = LighthouseRestClient(api_key=self.api_key)

            # Read and upload the database snapshot
            with open(db_path, "rb") as f:
                content = f.read()

            tag = f"{self.SNAPSHOT_TAG}-{self.agent_id}"
            result = await client.upload(
                content=content,
                filename=db_path.name,
                tag=tag,
            )

            cid = result.get("Hash") or result.get("cid")
            size = int(result.get("Size", len(content)))

            if not cid:
                raise ValueError(f"No CID returned from Lighthouse upload: {result}")

            logger.info(f"Uploaded snapshot to Lighthouse: {cid} ({size} bytes)")
            self._latest_cid = cid

            # Upload manifest pointing to this snapshot
            manifest = {
                "agent_id": self.agent_id,
                "snapshot_cid": cid,
                "snapshot_size": size,
                "uploaded_at": timestamp.isoformat(),
                "source_file": db_path.name,
            }
            await self._upload_manifest(client, manifest)
            await client.close()

            return SyncResult(
                success=True,
                target_name=self.name,
                bytes_synced=size,
                frames_synced=0,
                timestamp=timestamp,
                metadata={"cid": cid},
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
            # Manifest upload failure is non-fatal — snapshot is already safe
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
            uploads = await client.get_uploads()
            await client.close()

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
        """Upload WAL to Lighthouse (as full file for simplicity)."""
        # For Lighthouse, we upload the entire WAL as a snapshot
        # since IPFS content addressing means duplicates are deduplicated
        return await self.sync_snapshot(wal_path)

    async def get_latest_position(self) -> Optional[int]:
        """
        Get latest WAL position.

        Lighthouse uses full snapshot sync rather than incremental WAL,
        so position tracking is not applicable.
        """
        return None

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
