"""
Sync Targets

Abstractions for sync destinations. Each target declares a TrustTier
reflecting how much sovereignty the agent retains over its data:

    SOVEREIGN   Infrastructure we own and operate (self-hosted IPFS)
    FEDERATED   Agent-controlled auth, open protocol (Storacha/UCAN)
    DELEGATED   Third-party service with API key (Lighthouse)
    EXPEDIENT   Centralized cloud, fast but not sovereign (GCS, S3)

Write to all configured targets. Restore from most trusted.
"""

import hashlib
import json
import logging
import os
import sqlite3
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class TrustTier(Enum):
    """Trust hierarchy for persistence targets.

    Lower value = higher trust. Restore walks tiers in this order.
    """
    SOVEREIGN = 1   # Own infrastructure — self-hosted IPFS/Kubo
    FEDERATED = 2   # Agent-controlled auth — Storacha/UCAN
    DELEGATED = 3   # API-key gated — Lighthouse
    EXPEDIENT = 4   # Centralized cloud — GCS, S3


def _create_consistent_snapshot(db_path: Path) -> bytes:
    """Create a consistent snapshot using sqlite3.backup().

    Safe to call while the database is in use with an active WAL.
    Falls back to raw file read if backup fails (e.g. not a SQLite DB).
    """
    tmp_path = None
    try:
        src = sqlite3.connect(str(db_path))
        try:
            with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
                tmp_path = tmp.name
            dst = sqlite3.connect(tmp_path)
            try:
                src.backup(dst)
            finally:
                dst.close()
            with open(tmp_path, "rb") as f:
                return f.read()
        finally:
            src.close()
    except sqlite3.DatabaseError:
        logger.debug(f"sqlite3.backup() failed for {db_path}, using raw read")
        with open(db_path, "rb") as f:
            return f.read()
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


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

    @property
    def trust_tier(self) -> TrustTier:
        """Trust level of this target. Override in subclasses."""
        return TrustTier.EXPEDIENT

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

    @property
    def trust_tier(self) -> TrustTier:
        return TrustTier.EXPEDIENT

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


class GCSTarget(SyncTarget):
    """
    Google Cloud Storage sync target.

    Stores consistent SQLite snapshots in GCS with content-hash dedup.
    Uses Application Default Credentials (ADC) for auth.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "kestrel/",
        agent_id: str = "default",
        state_dir: Optional[Path] = None,
        project: Optional[str] = None,
        credentials_path: Optional[str] = None,
    ):
        self.bucket_name = bucket
        self.prefix = prefix.rstrip("/") + "/"
        self.agent_id = agent_id
        self._state_dir = state_dir
        self._project = project
        self._credentials_path = credentials_path
        self._last_content_hash: Optional[str] = None
        self._bucket = None

    @property
    def name(self) -> str:
        return f"gs://{self.bucket_name}/{self.prefix}{self.agent_id}"

    @property
    def trust_tier(self) -> TrustTier:
        return TrustTier.EXPEDIENT

    @property
    def _manifest_path(self) -> Optional[Path]:
        if self._state_dir:
            return self._state_dir / f".gcs_manifest_{self.agent_id}.json"
        return None

    def _load_local_manifest(self) -> Optional[Dict[str, Any]]:
        path = self._manifest_path
        if path and path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load GCS manifest: {e}")
        return None

    def _save_local_manifest(self, manifest: Dict[str, Any]) -> None:
        path = self._manifest_path
        if path:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(manifest, f, indent=2)
            except Exception as e:
                logger.warning(f"Failed to save GCS manifest: {e}")

    def _get_bucket(self):
        if self._bucket is None:
            from google.cloud import storage as gcs_storage
            kwargs = {}
            if self._project:
                kwargs["project"] = self._project
            if self._credentials_path:
                from google.oauth2 import service_account
                credentials = service_account.Credentials.from_service_account_file(
                    self._credentials_path
                )
                kwargs["credentials"] = credentials
                if not self._project:
                    kwargs["project"] = credentials.project_id
            client = gcs_storage.Client(**kwargs)
            self._bucket = client.bucket(self.bucket_name)
        return self._bucket

    async def sync_snapshot(self, db_path: Path) -> SyncResult:
        """Upload consistent snapshot to GCS with content-hash dedup."""
        import asyncio
        timestamp = datetime.now(timezone.utc)

        try:
            content = _create_consistent_snapshot(db_path)
            content_hash = hashlib.sha256(content).hexdigest()

            # Check local manifest for dedup
            manifest = self._load_local_manifest()
            if manifest and manifest.get("content_hash") == content_hash:
                logger.debug(f"GCS snapshot unchanged (hash={content_hash[:12]}), skipping")
                return SyncResult(
                    success=True,
                    target_name=self.name,
                    bytes_synced=0,
                    frames_synced=0,
                    timestamp=timestamp,
                    metadata={"skipped": True, "blob": manifest.get("blob_name")},
                )

            # Upload in thread (google-cloud-storage is synchronous)
            blob_name = (
                f"{self.prefix}{self.agent_id}/snapshots/"
                f"{timestamp.strftime('%Y%m%d_%H%M%S')}.db"
            )

            def _upload():
                bucket = self._get_bucket()
                blob = bucket.blob(blob_name)
                blob.upload_from_string(content, content_type="application/x-sqlite3")
                # Also update "latest" pointer
                latest = bucket.blob(f"{self.prefix}{self.agent_id}/latest.db")
                latest.upload_from_string(content, content_type="application/x-sqlite3")
                return blob_name

            result_name = await asyncio.to_thread(_upload)

            logger.info(f"Uploaded snapshot to gs://{self.bucket_name}/{result_name} ({len(content)} bytes)")

            new_manifest = {
                "agent_id": self.agent_id,
                "blob_name": result_name,
                "content_hash": content_hash,
                "size": len(content),
                "uploaded_at": timestamp.isoformat(),
            }
            self._save_local_manifest(new_manifest)

            return SyncResult(
                success=True,
                target_name=self.name,
                bytes_synced=len(content),
                frames_synced=0,
                timestamp=timestamp,
                metadata={"blob": result_name},
            )

        except Exception as e:
            logger.error(f"Failed to sync snapshot to GCS: {e}")
            return SyncResult(
                success=False,
                target_name=self.name,
                bytes_synced=0,
                frames_synced=0,
                timestamp=timestamp,
                error=str(e),
            )

    async def sync_wal(self, wal_path: Path, position: int) -> SyncResult:
        """No-op. GCS uses full consistent snapshots only."""
        return SyncResult(
            success=True,
            target_name=self.name,
            bytes_synced=0,
            frames_synced=0,
            timestamp=datetime.now(timezone.utc),
        )

    async def get_latest_position(self) -> Optional[int]:
        """Return max int to prevent WAL sync."""
        return 2**63

    async def restore_snapshot(self, dest_path: Path) -> Optional[SyncResult]:
        """Download the latest snapshot from GCS."""
        import asyncio
        timestamp = datetime.now(timezone.utc)

        try:
            def _download():
                bucket = self._get_bucket()
                blob = bucket.blob(f"{self.prefix}{self.agent_id}/latest.db")
                if not blob.exists():
                    return None
                return blob.download_as_bytes()

            content = await asyncio.to_thread(_download)
            if not content:
                logger.info("No GCS snapshot found to restore")
                return None

            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dest_path, "wb") as f:
                f.write(content)

            logger.info(f"Restored {len(content)} bytes from GCS to {dest_path}")
            return SyncResult(
                success=True,
                target_name=self.name,
                bytes_synced=len(content),
                frames_synced=0,
                timestamp=timestamp,
                metadata={"action": "restore"},
            )

        except Exception as e:
            logger.error(f"Failed to restore from GCS: {e}")
            return SyncResult(
                success=False,
                target_name=self.name,
                bytes_synced=0,
                frames_synced=0,
                timestamp=timestamp,
                error=str(e),
            )

    async def health_check(self) -> bool:
        """Check GCS bucket access."""
        import asyncio
        try:
            def _check():
                bucket = self._get_bucket()
                return bucket.exists()
            return await asyncio.to_thread(_check)
        except Exception as e:
            logger.warning(f"GCS health check failed: {e}")
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
    def trust_tier(self) -> TrustTier:
        return TrustTier.DELEGATED

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
            new_manifest = {
                "agent_id": self.agent_id,
                "snapshot_cid": cid,
                "snapshot_size": size,
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


class StorachaTarget(SyncTarget):
    """
    Storacha (web3.storage) sync target.

    Stores SQLite database snapshots on IPFS via the Storacha w3up protocol,
    using UCAN/DID authentication.  This is the preferred decentralised backup
    target — the agent's Ed25519 identity keypair is used directly as the UCAN
    signing key, giving cryptographic proof of ownership without opaque API keys.

    Supports cold-start restore for ephemeral environments (Cloud Run scale-to-zero).

    Required environment variables:
        STORACHA_SPACE_DID  — Space DID (from `w3 space create`)
        STORACHA_AGENT_KEY  — Ed25519 agent key (from `w3 key create`)
        STORACHA_PROOF      — Base64 delegation proof CAR

    Optional:
        STORACHA_STATE_CID  — Explicit snapshot CID to restore from (overrides discovery)
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
        self._latest_cid: Optional[str] = None

        from kestrel_sovereign.storage.providers.storacha_ucan import StorachaUCAN
        from kestrel_sovereign.storage.providers.storacha_rest import StorachaRestClient
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

    @property
    def _manifest_path(self) -> Optional[Path]:
        if self._state_dir:
            return self._state_dir / f".storacha_manifest_{self.agent_id}.json"
        return None

    def _load_local_manifest(self) -> Optional[Dict[str, Any]]:
        path = self._manifest_path
        if path and path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load Storacha local manifest: {e}")
        return None

    def _save_local_manifest(self, manifest: Dict[str, Any]) -> None:
        path = self._manifest_path
        if path:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(manifest, f, indent=2)
            except Exception as e:
                logger.warning(f"Failed to save Storacha local manifest: {e}")

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
            # Manifest upload is non-fatal — snapshot is already safe on IPFS
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
        """No-op — content-addressed storage uses full snapshots, not WAL deltas."""
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


class SovereignIPFSTarget(SyncTarget):
    """
    Self-hosted IPFS (Kubo) sync target.

    The sovereign default. Stores SQLite snapshots on infrastructure
    we own and operate. Requires only a reachable Kubo API endpoint —
    no third-party credentials.

    Trust tier: SOVEREIGN (highest)
    """

    def __init__(
        self,
        api_url: str,
        agent_id: str,
        state_dir: Optional[Path] = None,
    ):
        self.api_url = api_url.rstrip("/")
        self.agent_id = agent_id
        self._state_dir = state_dir
        self._client = None

    @property
    def name(self) -> str:
        return f"ipfs://{self.agent_id}"

    @property
    def trust_tier(self) -> TrustTier:
        return TrustTier.SOVEREIGN

    @property
    def _manifest_path(self) -> Optional[Path]:
        if self._state_dir:
            return self._state_dir / f".sovereign_ipfs_manifest_{self.agent_id}.json"
        return None

    def _load_local_manifest(self) -> Optional[Dict[str, Any]]:
        path = self._manifest_path
        if path and path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load sovereign IPFS manifest: {e}")
        return None

    def _save_local_manifest(self, manifest: Dict[str, Any]) -> None:
        path = self._manifest_path
        if path:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(manifest, f, indent=2)
            except Exception as e:
                logger.warning(f"Failed to save sovereign IPFS manifest: {e}")

    async def _get_client(self):
        """Get or create httpx async client."""
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client

    async def sync_snapshot(self, db_path: Path) -> SyncResult:
        """Upload consistent snapshot via Kubo /api/v0/add?pin=true."""
        timestamp = datetime.now(timezone.utc)

        try:
            content = _create_consistent_snapshot(db_path)
            content_hash = hashlib.sha256(content).hexdigest()

            # Content-hash dedup
            manifest = self._load_local_manifest()
            if manifest and manifest.get("content_hash") == content_hash:
                logger.debug(f"Sovereign IPFS snapshot unchanged (hash={content_hash[:12]}), skipping")
                return SyncResult(
                    success=True,
                    target_name=self.name,
                    bytes_synced=0,
                    frames_synced=0,
                    timestamp=timestamp,
                    metadata={"skipped": True, "cid": manifest.get("cid")},
                )

            client = await self._get_client()

            # Kubo API: POST /api/v0/add?pin=true
            filename = f"{self.agent_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}.db"
            response = await client.post(
                f"{self.api_url}/api/v0/add",
                params={"pin": "true", "quieter": "true"},
                files={"file": (filename, content, "application/x-sqlite3")},
            )
            response.raise_for_status()
            result_data = response.json()
            cid = result_data["Hash"]

            logger.info(
                f"Uploaded snapshot to sovereign IPFS: {cid} ({len(content)} bytes)"
            )

            new_manifest = {
                "agent_id": self.agent_id,
                "cid": cid,
                "content_hash": content_hash,
                "size": len(content),
                "uploaded_at": timestamp.isoformat(),
            }
            self._save_local_manifest(new_manifest)

            return SyncResult(
                success=True,
                target_name=self.name,
                bytes_synced=len(content),
                frames_synced=0,
                timestamp=timestamp,
                metadata={"cid": cid},
            )

        except Exception as e:
            logger.error(f"Failed to sync snapshot to sovereign IPFS: {e}")
            return SyncResult(
                success=False,
                target_name=self.name,
                bytes_synced=0,
                frames_synced=0,
                timestamp=timestamp,
                error=str(e),
            )

    async def sync_wal(self, wal_path: Path, position: int) -> SyncResult:
        """No-op. Sovereign IPFS uses full snapshots only."""
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

    async def restore_snapshot(self, dest_path: Path) -> Optional[SyncResult]:
        """Download latest snapshot from sovereign IPFS via /api/v0/cat."""
        timestamp = datetime.now(timezone.utc)

        # Resolve CID: env var → local manifest
        cid = os.environ.get("SOVEREIGN_IPFS_CID")
        if not cid:
            manifest = self._load_local_manifest()
            if manifest:
                cid = manifest.get("cid")

        if not cid:
            logger.debug("No sovereign IPFS CID available for restore")
            return None

        try:
            client = await self._get_client()
            response = await client.post(
                f"{self.api_url}/api/v0/cat",
                params={"arg": cid},
                timeout=120.0,
            )
            response.raise_for_status()
            content = response.content

            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dest_path, "wb") as f:
                f.write(content)

            logger.info(
                f"Restored snapshot from sovereign IPFS: {cid} ({len(content)} bytes)"
            )

            return SyncResult(
                success=True,
                target_name=self.name,
                bytes_synced=len(content),
                frames_synced=0,
                timestamp=timestamp,
                metadata={"cid": cid},
            )

        except Exception as e:
            logger.warning(f"Sovereign IPFS restore failed: {e}")
            return None

    async def health_check(self) -> bool:
        """Check Kubo connectivity via /api/v0/id."""
        try:
            client = await self._get_client()
            response = await client.post(
                f"{self.api_url}/api/v0/id",
                timeout=10.0,
            )
            response.raise_for_status()
            peer_info = response.json()
            logger.debug(f"Sovereign IPFS node: {peer_info.get('ID', 'unknown')}")
            return True
        except Exception as e:
            logger.warning(f"Sovereign IPFS health check failed: {e}")
            return False
