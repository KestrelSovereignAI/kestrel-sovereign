"""
Sync Targets

Abstractions for sync destinations. SQLite changes can be replicated
to various targets including cloud storage and PostgreSQL.
"""

import asyncio
import logging
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
    for maximum data sovereignty.
    """

    def __init__(
        self,
        api_key: str,
        agent_id: str,
    ):
        """
        Initialize Lighthouse target.

        Args:
            api_key: Lighthouse API key
            agent_id: Unique agent identifier for namespacing
        """
        self.api_key = api_key
        self.agent_id = agent_id

    @property
    def name(self) -> str:
        return f"lighthouse://{self.agent_id}"

    async def sync_snapshot(self, db_path: Path) -> SyncResult:
        """Upload snapshot to Lighthouse."""
        timestamp = datetime.now(timezone.utc)

        try:
            # Import Lighthouse SDK
            try:
                from lighthouseweb3 import Lighthouse
            except ImportError:
                raise ImportError(
                    "lighthouseweb3 is required. Install with: pip install lighthouseweb3"
                )

            lh = Lighthouse(token=self.api_key)

            # Upload file
            result = await asyncio.to_thread(
                lh.upload,
                source=str(db_path),
            )

            cid = result.get("Hash") or result.get("cid")
            size = result.get("Size", 0)

            logger.info(f"Uploaded snapshot to Lighthouse: {cid}")

            return SyncResult(
                success=True,
                target_name=self.name,
                bytes_synced=int(size),
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

    async def sync_wal(self, wal_path: Path, position: int) -> SyncResult:
        """Upload WAL to Lighthouse (as full file for simplicity)."""
        # For Lighthouse, we upload the entire WAL as a snapshot
        # since IPFS content addressing means duplicates are deduplicated
        return await self.sync_snapshot(wal_path)

    async def get_latest_position(self) -> Optional[int]:
        """
        Get latest WAL position.

        Note: Lighthouse doesn't support mutable state, so we'd need
        a separate mechanism to track position (local file or external service).
        """
        return None

    async def health_check(self) -> bool:
        """Check Lighthouse API connectivity."""
        try:
            from lighthouseweb3 import Lighthouse
            lh = Lighthouse(token=self.api_key)
            # Simple API call to verify connectivity
            await asyncio.to_thread(lh.getBalance)
            return True
        except Exception as e:
            logger.warning(f"Lighthouse health check failed: {e}")
            return False
