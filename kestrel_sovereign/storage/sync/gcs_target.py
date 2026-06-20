"""
GCS Sync Target

Google Cloud Storage sync target with content-hash deduplication.
"""

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from kestrel_sovereign.storage.sync.manifest_manager import ManifestManagerMixin
from kestrel_sovereign.storage.sync.retention import (
    RetentionItem,
    RetentionPolicy,
    classify,
    parse_timestamp,
)
from kestrel_sovereign.storage.sync.targets import (
    SyncTarget,
    SyncResult,
    TrustTier,
    _create_consistent_snapshot,
)

logger = logging.getLogger(__name__)


class GCSTarget(ManifestManagerMixin, SyncTarget):
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
        self._manifest_filename = f".gcs_manifest_{self.agent_id}.json"
        self._last_content_hash: Optional[str] = None
        self._bucket = None

    @property
    def name(self) -> str:
        return f"gs://{self.bucket_name}/{self.prefix}{self.agent_id}"

    @property
    def trust_tier(self) -> TrustTier:
        return TrustTier.EXPEDIENT

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

    async def prune(self, policy: RetentionPolicy) -> Dict[str, Any]:
        """Prune timestamped GCS snapshots while preserving latest.db."""
        import asyncio

        snapshots_prefix = f"{self.prefix}{self.agent_id}/snapshots/"
        latest_name = f"{self.prefix}{self.agent_id}/latest.db"

        def _list_and_delete() -> Dict[str, Any]:
            bucket = self._get_bucket()
            items: list[RetentionItem] = []
            blobs_by_name: Dict[str, Any] = {}
            for blob in bucket.list_blobs(prefix=snapshots_prefix):
                if blob.name == latest_name or not blob.name.endswith(".db"):
                    continue
                ts = parse_timestamp(blob.name) or parse_timestamp(
                    getattr(blob, "time_created", None)
                )
                if ts is None:
                    logger.debug("GCS retention skipped object without timestamp: %s", blob.name)
                    continue
                items.append(
                    RetentionItem(
                        key=blob.name,
                        name=Path(blob.name).name,
                        timestamp=ts,
                        data_class=classify({"key": blob.name, "role": "snapshot"}),
                    )
                )
                blobs_by_name[blob.name] = blob

            deleted = []
            for item in policy.deletions(items):
                blob = blobs_by_name[item.key]
                blob.delete()
                deleted.append(item.key)
            return {"deleted": len(deleted), "keys": deleted, "scanned": len(items)}

        return await asyncio.to_thread(_list_and_delete)

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
