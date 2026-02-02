"""
Sync Protocol for Multi-Tier Storage

Handles synchronization of content between storage tiers:
- Browser ↔ Cloud backup
- Local IPFS ↔ Lighthouse cloud
- Cross-device sync via CID manifests

The protocol is designed to be:
- Resumable (can continue interrupted syncs)
- Conflict-aware (detects and reports conflicts)
- Bandwidth-efficient (only sync changed content)
"""

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from kestrel_sovereign.storage.providers.base import (
    StorageProvider,
    StorageResult,
    StorageTier,
    SyncItem,
    SyncManifest,
    SyncStatus,
)

logger = logging.getLogger(__name__)


class ConflictResolution(Enum):
    """How to resolve sync conflicts."""
    KEEP_LOCAL = "keep_local"      # Local version wins
    KEEP_REMOTE = "keep_remote"    # Remote version wins
    KEEP_NEWER = "keep_newer"      # Most recent wins
    KEEP_BOTH = "keep_both"        # Keep both, rename one
    MANUAL = "manual"              # Require user decision


@dataclass
class SyncConflict:
    """Represents a conflict between local and remote content."""
    content_hash: str
    local_result: StorageResult
    remote_result: StorageResult
    resolution: Optional[ConflictResolution] = None
    resolved_at: Optional[datetime] = None


@dataclass
class SyncProgress:
    """Progress tracking for sync operations."""
    total_items: int = 0
    completed_items: int = 0
    failed_items: int = 0
    skipped_items: int = 0
    total_bytes: int = 0
    transferred_bytes: int = 0
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    errors: List[str] = field(default_factory=list)

    @property
    def progress_percent(self) -> float:
        if self.total_items == 0:
            return 100.0
        return (self.completed_items / self.total_items) * 100

    @property
    def elapsed_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.start_time).total_seconds()

    @property
    def transfer_rate_bps(self) -> float:
        if self.elapsed_seconds == 0:
            return 0
        return self.transferred_bytes / self.elapsed_seconds


class SyncProtocol:
    """
    Protocol for synchronizing content between storage tiers.

    Handles:
    - Manifest generation (what needs to sync)
    - Incremental sync (only changed items)
    - Conflict detection and resolution
    - Progress tracking and callbacks
    """

    def __init__(
        self,
        conflict_resolution: ConflictResolution = ConflictResolution.KEEP_NEWER,
        progress_callback: Optional[Callable[[SyncProgress], None]] = None,
    ):
        """
        Initialize sync protocol.

        Args:
            conflict_resolution: Default conflict resolution strategy
            progress_callback: Optional callback for progress updates
        """
        self._conflict_resolution = conflict_resolution
        self._progress_callback = progress_callback
        self._conflicts: List[SyncConflict] = []

    async def generate_manifest(
        self,
        source: StorageProvider,
        target: StorageProvider,
    ) -> SyncManifest:
        """
        Generate a sync manifest comparing source and target.

        Args:
            source: Source storage provider
            target: Target storage provider

        Returns:
            SyncManifest with items to sync
        """
        logger.info(f"Generating sync manifest: {source.tier.value} → {target.tier.value}")

        # Get content from both providers
        source_content = await source.list_content(limit=10000)
        target_content = await target.list_content(limit=10000)

        # Build lookup sets
        source_hashes = {r.content_hash: r for r in source_content}
        target_hashes = {r.content_hash: r for r in target_content}

        manifest = SyncManifest(
            source_tier=source.tier,
            target_tier=target.tier,
            items=[],
            total_bytes=0,
        )

        # Find items in source but not in target
        for hash_id, result in source_hashes.items():
            if hash_id not in target_hashes:
                manifest.items.append(SyncItem(
                    content_hash=hash_id,
                    source_tier=source.tier,
                    target_tier=target.tier,
                    size_bytes=result.size_bytes,
                    status=SyncStatus.PENDING,
                ))
                manifest.total_bytes += result.size_bytes
            else:
                # Check for conflicts (same hash, different metadata)
                target_result = target_hashes[hash_id]
                if self._has_conflict(result, target_result):
                    self._conflicts.append(SyncConflict(
                        content_hash=hash_id,
                        local_result=result,
                        remote_result=target_result,
                    ))

        # Estimate cost if target supports it
        if hasattr(target, "estimate_cost"):
            manifest.estimated_cost_usd = await target.estimate_cost(manifest.total_bytes)

        logger.info(
            f"Manifest generated: {len(manifest.items)} items, "
            f"{manifest.total_bytes} bytes, "
            f"{len(self._conflicts)} conflicts"
        )

        return manifest

    async def execute_sync(
        self,
        manifest: SyncManifest,
        source: StorageProvider,
        target: StorageProvider,
        max_concurrent: int = 5,
    ) -> SyncProgress:
        """
        Execute sync based on manifest.

        Args:
            manifest: Sync manifest to execute
            source: Source provider
            target: Target provider
            max_concurrent: Max concurrent transfers

        Returns:
            SyncProgress with results
        """
        progress = SyncProgress(
            total_items=len(manifest.items),
            total_bytes=manifest.total_bytes,
        )

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)

        async def sync_item(item: SyncItem) -> bool:
            async with semaphore:
                try:
                    item.status = SyncStatus.IN_PROGRESS

                    # Retrieve from source
                    content = await source.retrieve(
                        item.content_hash,
                        encryption_key_hash=None,  # Will be re-encrypted at target
                    )

                    # Store to target
                    result = await target.store(content)

                    item.status = SyncStatus.COMPLETED
                    progress.completed_items += 1
                    progress.transferred_bytes += item.size_bytes

                    logger.debug(f"Synced: {item.content_hash[:16]}...")
                    return True

                except Exception as e:
                    item.status = SyncStatus.FAILED
                    item.error_message = str(e)
                    progress.failed_items += 1
                    progress.errors.append(f"{item.content_hash[:16]}: {e}")
                    logger.error(f"Sync failed for {item.content_hash[:16]}: {e}")
                    return False

        # Execute all syncs
        tasks = [sync_item(item) for item in manifest.items]
        await asyncio.gather(*tasks)

        # Report progress
        if self._progress_callback:
            self._progress_callback(progress)

        logger.info(
            f"Sync complete: {progress.completed_items}/{progress.total_items} items, "
            f"{progress.failed_items} failures"
        )

        return progress

    def _has_conflict(self, local: StorageResult, remote: StorageResult) -> bool:
        """Check if two results represent a conflict."""
        # Same content hash but different encryption or metadata
        if local.encryption_key_hash != remote.encryption_key_hash:
            return True
        if local.size_bytes != remote.size_bytes:
            return True
        return False

    def get_conflicts(self) -> List[SyncConflict]:
        """Get list of detected conflicts."""
        return self._conflicts.copy()

    async def resolve_conflict(
        self,
        conflict: SyncConflict,
        resolution: ConflictResolution,
        source: StorageProvider,
        target: StorageProvider,
    ) -> bool:
        """
        Resolve a sync conflict.

        Args:
            conflict: Conflict to resolve
            resolution: How to resolve it
            source: Source provider
            target: Target provider

        Returns:
            True if resolved successfully
        """
        try:
            if resolution == ConflictResolution.KEEP_LOCAL:
                # Push local version to target
                content = await source.retrieve(conflict.content_hash)
                await target.store(content)

            elif resolution == ConflictResolution.KEEP_REMOTE:
                # Pull remote version to local
                content = await target.retrieve(conflict.content_hash)
                await source.store(content)

            elif resolution == ConflictResolution.KEEP_NEWER:
                # Compare timestamps and keep newer
                if conflict.local_result.created_at > conflict.remote_result.created_at:
                    content = await source.retrieve(conflict.content_hash)
                    await target.store(content)
                else:
                    content = await target.retrieve(conflict.content_hash)
                    await source.store(content)

            elif resolution == ConflictResolution.KEEP_BOTH:
                # Keep both with different hashes (append timestamp to one)
                logger.warning("KEEP_BOTH resolution not yet implemented")
                return False

            conflict.resolution = resolution
            conflict.resolved_at = datetime.now(timezone.utc)
            return True

        except Exception as e:
            logger.error(f"Failed to resolve conflict: {e}")
            return False


@dataclass
class BrowserSyncState:
    """State for browser-to-cloud sync."""
    last_sync: Optional[datetime] = None
    local_content_hashes: Set[str] = field(default_factory=set)
    remote_content_hashes: Set[str] = field(default_factory=set)
    pending_uploads: List[str] = field(default_factory=list)
    pending_downloads: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage."""
        return {
            "last_sync": self.last_sync.isoformat() if self.last_sync else None,
            "local_content_hashes": list(self.local_content_hashes),
            "remote_content_hashes": list(self.remote_content_hashes),
            "pending_uploads": self.pending_uploads,
            "pending_downloads": self.pending_downloads,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BrowserSyncState":
        """Deserialize from kestrel_sovereign.storage."""
        return cls(
            last_sync=datetime.fromisoformat(data["last_sync"]) if data.get("last_sync") else None,
            local_content_hashes=set(data.get("local_content_hashes", [])),
            remote_content_hashes=set(data.get("remote_content_hashes", [])),
            pending_uploads=data.get("pending_uploads", []),
            pending_downloads=data.get("pending_downloads", []),
        )


class BrowserSyncProtocol:
    """
    Specialized sync protocol for browser IndexedDB to cloud.

    Optimized for:
    - Limited browser resources
    - Intermittent connectivity
    - User-initiated sync (not continuous)
    """

    def __init__(self, api_endpoint: str, auth_token: str):
        """
        Initialize browser sync.

        Args:
            api_endpoint: Base URL for sync API
            auth_token: JWT or API key for authentication
        """
        self.api_endpoint = api_endpoint.rstrip("/")
        self.auth_token = auth_token
        self._state: Optional[BrowserSyncState] = None

    async def get_sync_status(self) -> Dict[str, Any]:
        """Get current sync status."""
        import aiohttp

        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"Bearer {self.auth_token}"}
            async with session.get(
                f"{self.api_endpoint}/api/sovereign/status",
                headers=headers,
            ) as response:
                if response.status == 200:
                    return await response.json()
                raise Exception(f"Sync status failed: {response.status}")

    async def upload_encrypted_backup(
        self,
        encrypted_blob: bytes,
        content_hash: str,
    ) -> Dict[str, Any]:
        """
        Upload encrypted backup to cloud.

        Args:
            encrypted_blob: Encrypted content
            content_hash: SHA256 hash of original content

        Returns:
            Upload result with CID
        """
        import aiohttp

        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.auth_token}",
                "X-Content-Hash": content_hash,
            }
            async with session.post(
                f"{self.api_endpoint}/api/sovereign/sync",
                headers=headers,
                data=encrypted_blob,
            ) as response:
                if response.status == 200:
                    return await response.json()
                raise Exception(f"Upload failed: {response.status}")

    async def download_backup(self, content_hash: str) -> bytes:
        """
        Download encrypted backup from cloud.

        Args:
            content_hash: Hash of content to download

        Returns:
            Encrypted content bytes
        """
        import aiohttp

        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"Bearer {self.auth_token}"}
            async with session.get(
                f"{self.api_endpoint}/api/sovereign/restore",
                headers=headers,
                params={"content_hash": content_hash},
            ) as response:
                if response.status == 200:
                    return await response.read()
                raise Exception(f"Download failed: {response.status}")

    async def compute_sync_diff(
        self,
        local_hashes: Set[str],
    ) -> Dict[str, List[str]]:
        """
        Compute what needs to be synced.

        Args:
            local_hashes: Set of content hashes in local storage

        Returns:
            Dict with 'upload' and 'download' lists
        """
        status = await self.get_sync_status()
        remote_hashes = set(status.get("content_hashes", []))

        return {
            "upload": list(local_hashes - remote_hashes),
            "download": list(remote_hashes - local_hashes),
        }


def create_sync_protocol(
    conflict_resolution: ConflictResolution = ConflictResolution.KEEP_NEWER,
) -> SyncProtocol:
    """Create a standard sync protocol."""
    return SyncProtocol(conflict_resolution=conflict_resolution)


def create_browser_sync(api_endpoint: str, auth_token: str) -> BrowserSyncProtocol:
    """Create a browser sync protocol."""
    return BrowserSyncProtocol(api_endpoint=api_endpoint, auth_token=auth_token)
