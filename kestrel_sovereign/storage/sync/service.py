"""
Sync Service

Orchestrates SQLite replication to multiple targets.
This is the main entry point for the SQLite-first sync architecture.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
import json

from kestrel_sovereign.storage.sync.wal_listener import WALListener, WALChange
from kestrel_sovereign.storage.sync.targets import SyncTarget, SyncResult

logger = logging.getLogger(__name__)


@dataclass
class SyncStats:
    """Statistics for sync operations."""
    total_syncs: int = 0
    successful_syncs: int = 0
    failed_syncs: int = 0
    bytes_synced: int = 0
    last_sync: Optional[datetime] = None
    last_error: Optional[str] = None


@dataclass
class SyncState:
    """Persistent sync state."""
    db_path: str
    targets: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    last_snapshot: Optional[datetime] = None
    last_wal_sync: Optional[datetime] = None
    stats: SyncStats = field(default_factory=SyncStats)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "db_path": self.db_path,
            "targets": self.targets,
            "last_snapshot": self.last_snapshot.isoformat() if self.last_snapshot else None,
            "last_wal_sync": self.last_wal_sync.isoformat() if self.last_wal_sync else None,
            "stats": {
                "total_syncs": self.stats.total_syncs,
                "successful_syncs": self.stats.successful_syncs,
                "failed_syncs": self.stats.failed_syncs,
                "bytes_synced": self.stats.bytes_synced,
                "last_sync": self.stats.last_sync.isoformat() if self.stats.last_sync else None,
                "last_error": self.stats.last_error,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SyncState":
        stats_data = data.get("stats", {})
        return cls(
            db_path=data["db_path"],
            targets=data.get("targets", {}),
            last_snapshot=datetime.fromisoformat(data["last_snapshot"]) if data.get("last_snapshot") else None,
            last_wal_sync=datetime.fromisoformat(data["last_wal_sync"]) if data.get("last_wal_sync") else None,
            stats=SyncStats(
                total_syncs=stats_data.get("total_syncs", 0),
                successful_syncs=stats_data.get("successful_syncs", 0),
                failed_syncs=stats_data.get("failed_syncs", 0),
                bytes_synced=stats_data.get("bytes_synced", 0),
                last_sync=datetime.fromisoformat(stats_data["last_sync"]) if stats_data.get("last_sync") else None,
                last_error=stats_data.get("last_error"),
            ),
        )


class SyncService:
    """
    SQLite-first sync service.

    Monitors a SQLite database and replicates changes to configured targets.
    Supports both continuous WAL streaming and periodic snapshot backups.

    Usage:
        sync = SyncService(db_path="/path/to/agent.db")
        sync.add_target(S3Target(bucket="my-bucket"))
        await sync.start()
    """

    # Default intervals
    DEFAULT_WAL_POLL_INTERVAL = 0.1  # 100ms
    DEFAULT_SNAPSHOT_INTERVAL = 3600  # 1 hour
    DEFAULT_WAL_SYNC_INTERVAL = 1.0  # 1 second batch

    def __init__(
        self,
        db_path: str,
        state_file: Optional[str] = None,
        snapshot_interval: float = DEFAULT_SNAPSHOT_INTERVAL,
        wal_sync_interval: float = DEFAULT_WAL_SYNC_INTERVAL,
        on_sync: Optional[Callable[[SyncResult], None]] = None,
        on_error: Optional[Callable[[str, Exception], None]] = None,
    ):
        """
        Initialize sync service.

        Args:
            db_path: Path to SQLite database
            state_file: Path to persist sync state (default: {db_path}.sync)
            snapshot_interval: Seconds between full snapshots
            wal_sync_interval: Seconds between WAL batch syncs
            on_sync: Callback for successful syncs
            on_error: Callback for sync errors
        """
        self.db_path = Path(db_path)
        self.wal_path = Path(f"{db_path}-wal")
        self.state_file = Path(state_file) if state_file else Path(f"{db_path}.sync")

        self.snapshot_interval = snapshot_interval
        self.wal_sync_interval = wal_sync_interval

        self._on_sync = on_sync
        self._on_error = on_error

        self._targets: List[SyncTarget] = []
        self._wal_listener: Optional[WALListener] = None
        self._running = False
        self._state: Optional[SyncState] = None

        self._pending_changes: List[WALChange] = []
        self._sync_task: Optional[asyncio.Task] = None
        self._snapshot_task: Optional[asyncio.Task] = None

    def add_target(self, target: SyncTarget) -> None:
        """Add a sync target."""
        self._targets.append(target)
        logger.info(f"Added sync target: {target.name}")

    def remove_target(self, target_name: str) -> bool:
        """Remove a sync target by name."""
        for i, target in enumerate(self._targets):
            if target.name == target_name:
                self._targets.pop(i)
                logger.info(f"Removed sync target: {target_name}")
                return True
        return False

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def targets(self) -> List[str]:
        return [t.name for t in self._targets]

    @property
    def stats(self) -> Optional[SyncStats]:
        return self._state.stats if self._state else None

    async def start(self) -> None:
        """Start sync service."""
        if self._running:
            return

        # Load or create state
        await self._load_state()

        self._running = True
        logger.info(f"Sync service starting for {self.db_path}")

        # Start WAL listener
        self._wal_listener = WALListener(
            str(self.db_path),
            poll_interval=self.DEFAULT_WAL_POLL_INTERVAL,
            on_change=self._on_wal_change,
        )

        # Start background tasks
        self._sync_task = asyncio.create_task(self._wal_sync_loop())
        self._snapshot_task = asyncio.create_task(self._snapshot_loop())

        # Start WAL listening in background
        asyncio.create_task(self._wal_listener.start())

        logger.info(f"Sync service started with {len(self._targets)} targets")

    async def stop(self) -> None:
        """Stop sync service gracefully."""
        if not self._running:
            return

        self._running = False
        logger.info("Stopping sync service...")

        # Stop WAL listener
        if self._wal_listener:
            await self._wal_listener.stop()

        # Cancel tasks
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass

        if self._snapshot_task:
            self._snapshot_task.cancel()
            try:
                await self._snapshot_task
            except asyncio.CancelledError:
                pass

        # Final sync of pending changes
        if self._pending_changes:
            await self._sync_pending()

        # Save state
        await self._save_state()

        logger.info("Sync service stopped")

    async def force_snapshot(self) -> Dict[str, SyncResult]:
        """Force an immediate snapshot sync to all targets."""
        results = {}
        for target in self._targets:
            try:
                result = await target.sync_snapshot(self.db_path)
                results[target.name] = result
                self._update_stats(result)
            except Exception as e:
                logger.error(f"Snapshot failed for {target.name}: {e}")
                results[target.name] = SyncResult(
                    success=False,
                    target_name=target.name,
                    bytes_synced=0,
                    frames_synced=0,
                    timestamp=datetime.now(timezone.utc),
                    error=str(e),
                )
        return results

    async def health_check(self) -> Dict[str, bool]:
        """Check health of all targets."""
        results = {}
        for target in self._targets:
            try:
                results[target.name] = await target.health_check()
            except Exception as e:
                logger.warning(f"Health check failed for {target.name}: {e}")
                results[target.name] = False
        return results

    def _on_wal_change(self, change: WALChange) -> None:
        """Handle WAL change notification."""
        self._pending_changes.append(change)
        logger.debug(f"Queued {change.frame_count} frames for sync")

    async def _wal_sync_loop(self) -> None:
        """Background loop for WAL sync."""
        while self._running:
            try:
                await asyncio.sleep(self.wal_sync_interval)
                if self._pending_changes:
                    await self._sync_pending()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"WAL sync error: {e}")
                if self._on_error:
                    self._on_error("wal_sync", e)

    async def _snapshot_loop(self) -> None:
        """Background loop for periodic snapshots."""
        while self._running:
            try:
                await asyncio.sleep(self.snapshot_interval)

                # Check if snapshot is needed
                if self._state and self._state.last_snapshot:
                    elapsed = datetime.now(timezone.utc) - self._state.last_snapshot
                    if elapsed < timedelta(seconds=self.snapshot_interval):
                        continue

                await self.force_snapshot()
                if self._state:
                    self._state.last_snapshot = datetime.now(timezone.utc)
                    await self._save_state()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Snapshot error: {e}")
                if self._on_error:
                    self._on_error("snapshot", e)

    async def _sync_pending(self) -> None:
        """Sync pending WAL changes to all targets."""
        if not self._pending_changes:
            return

        changes = self._pending_changes.copy()
        self._pending_changes.clear()

        total_frames = sum(c.frame_count for c in changes)
        total_bytes = sum(c.bytes_changed for c in changes)

        logger.debug(f"Syncing {total_frames} frames ({total_bytes} bytes)")

        # Get current WAL position
        wal_position = 0
        if self.wal_path.exists():
            wal_position = self.wal_path.stat().st_size

        for target in self._targets:
            try:
                # Get target's last position
                last_pos = await target.get_latest_position() or 0

                if wal_position > last_pos:
                    result = await target.sync_wal(self.wal_path, last_pos)
                    self._update_stats(result)

                    if self._on_sync and result.success:
                        self._on_sync(result)

            except Exception as e:
                logger.error(f"WAL sync failed for {target.name}: {e}")
                if self._on_error:
                    self._on_error(target.name, e)

        if self._state:
            self._state.last_wal_sync = datetime.now(timezone.utc)

    def _update_stats(self, result: SyncResult) -> None:
        """Update sync statistics."""
        if not self._state:
            return

        self._state.stats.total_syncs += 1
        if result.success:
            self._state.stats.successful_syncs += 1
            self._state.stats.bytes_synced += result.bytes_synced
        else:
            self._state.stats.failed_syncs += 1
            self._state.stats.last_error = result.error

        self._state.stats.last_sync = result.timestamp

    async def _load_state(self) -> None:
        """Load sync state from file."""
        if self.state_file.exists():
            try:
                with open(self.state_file, encoding="utf-8") as f:
                    data = json.load(f)
                self._state = SyncState.from_dict(data)
                logger.debug(f"Loaded sync state from {self.state_file}")
            except Exception as e:
                logger.warning(f"Failed to load sync state: {e}")
                self._state = SyncState(db_path=str(self.db_path))
        else:
            self._state = SyncState(db_path=str(self.db_path))

    async def _save_state(self) -> None:
        """Save sync state to file."""
        if self._state:
            try:
                with open(self.state_file, "w", encoding="utf-8") as f:
                    json.dump(self._state.to_dict(), f, indent=2)
                logger.debug(f"Saved sync state to {self.state_file}")
            except Exception as e:
                logger.error(f"Failed to save sync state: {e}")
