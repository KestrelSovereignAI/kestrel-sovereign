"""
S3 Sync Target

Amazon S3 / S3-compatible storage target for database snapshots and WAL segments.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from kestrel_sovereign.storage.sync.retention import (
    RetentionItem,
    RetentionPolicy,
    classify,
    parse_timestamp,
)
from kestrel_sovereign.storage.sync.targets import SyncTarget, SyncResult, TrustTier

logger = logging.getLogger(__name__)


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

    async def prune(self, policy: RetentionPolicy) -> Dict[str, Any]:
        """Prune timestamped S3 snapshots and WAL segments."""
        client = await self._get_client()
        prefixes = (f"{self.prefix}snapshots/", f"{self.prefix}wal/")
        items: list[RetentionItem] = []

        for prefix in prefixes:
            continuation_token = None
            while True:
                kwargs: Dict[str, Any] = {"Bucket": self.bucket, "Prefix": prefix}
                if continuation_token:
                    kwargs["ContinuationToken"] = continuation_token
                response = await client.list_objects_v2(**kwargs)
                for obj in response.get("Contents", []):
                    key = obj.get("Key")
                    if not key or key in {
                        f"{self.prefix}latest_snapshot",
                        f"{self.prefix}wal_position",
                    }:
                        continue
                    ts = parse_timestamp(key) or parse_timestamp(obj.get("LastModified"))
                    if ts is None:
                        logger.debug("S3 retention skipped object without timestamp: %s", key)
                        continue
                    role = "wal" if "/wal/" in key else "snapshot"
                    items.append(
                        RetentionItem(
                            key=key,
                            name=Path(key).name,
                            timestamp=ts,
                            data_class=classify({"key": key, "role": role}),
                        )
                    )
                if not response.get("IsTruncated"):
                    break
                continuation_token = response.get("NextContinuationToken")
                if not continuation_token:
                    break

        deletions = policy.deletions(items)
        # S3 DeleteObjects accepts at most 1000 keys per request; batch so a
        # large backlog doesn't fail the whole prune (and delete nothing).
        for batch_start in range(0, len(deletions), 1000):
            batch = deletions[batch_start:batch_start + 1000]
            await client.delete_objects(
                Bucket=self.bucket,
                Delete={"Objects": [{"Key": item.key} for item in batch]},
            )
        return {
            "deleted": len(deletions),
            "keys": [item.key for item in deletions],
            "scanned": len(items),
        }

    async def health_check(self) -> bool:
        """Check S3 connectivity."""
        try:
            client = await self._get_client()
            await client.head_bucket(Bucket=self.bucket)
            return True
        except Exception as e:
            logger.warning(f"S3 health check failed: {e}")
            return False
