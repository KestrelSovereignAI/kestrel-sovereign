"""
Sync Service

Event-driven snapshot service for SQLite persistence.
Snapshots are triggered by lifecycle events (shutdown, scheduled backup)
rather than continuous WAL polling.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any

from kestrel_sovereign.storage.sync.targets import SyncTarget, SyncResult, TrustTier

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
    # Change fingerprint of the DB at the last completed snapshot (#1674 P3).
    # Lets force_snapshot skip a full re-snapshot when nothing changed — so
    # idle agents stop re-dumping the whole DB (notably to S3) every cycle.
    last_fingerprint: Optional[str] = None
    stats: SyncStats = field(default_factory=SyncStats)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "db_path": self.db_path,
            "targets": self.targets,
            "last_snapshot": self.last_snapshot.isoformat() if self.last_snapshot else None,
            "last_fingerprint": self.last_fingerprint,
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
            last_fingerprint=data.get("last_fingerprint"),
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
    Event-driven snapshot service.

    A thin registry of sync targets with a force_snapshot() method.
    Snapshots are triggered by:
      - Shutdown (SIGTERM / graceful exit)
      - Scheduled task (every N hours)
      - Explicit command (!backup)

    No background polling. No WAL watching.
    """

    def __init__(
        self,
        db_path: str,
        state_file: Optional[str] = None,
        wal_sync_interval: Optional[float] = None,
        on_sync: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
    ):
        self.db_path = Path(db_path)
        self.state_file = Path(state_file) if state_file else Path(f"{db_path}.sync")
        self.wal_sync_interval = wal_sync_interval
        self.on_sync = on_sync
        self.on_error = on_error
        self._targets: List[SyncTarget] = []
        self._state: Optional[SyncState] = None
        self._poll_task: Optional[asyncio.Task] = None

    def add_target(self, target: SyncTarget) -> None:
        """Add a sync target."""
        self._targets.append(target)
        logger.info(f"Added sync target: {target.name} ({target.trust_tier.name})")

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
        """For backward compatibility. Always True if targets exist."""
        return len(self._targets) > 0

    @property
    def targets(self) -> List[str]:
        return [t.name for t in self._targets]

    @property
    def targets_by_trust(self) -> List[SyncTarget]:
        """Return targets sorted by trust tier (most trusted first)."""
        return sorted(self._targets, key=lambda t: t.trust_tier.value)

    @property
    def stats(self) -> Optional[SyncStats]:
        return self._state.stats if self._state else None

    async def start(self) -> None:
        """Load state and optionally start periodic sync if wal_sync_interval is set."""
        await self._load_state()
        if self.wal_sync_interval and self.wal_sync_interval > 0:
            self._poll_task = asyncio.create_task(self._periodic_sync())
        logger.info(
            f"Sync service ready for {self.db_path} "
            f"({len(self._targets)} targets: "
            f"{', '.join(f'{t.name}({t.trust_tier.name})' for t in self.targets_by_trust)})"
        )

    async def stop(self) -> None:
        """Cancel periodic sync and save state on shutdown."""
        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None
        await self._save_state()
        logger.info("Sync service stopped")

    async def _periodic_sync(self) -> None:
        """Background loop that triggers force_snapshot at wal_sync_interval."""
        while True:
            await asyncio.sleep(self.wal_sync_interval)
            try:
                await self.force_snapshot()
            except Exception as e:
                logger.error(f"Periodic sync error: {e}")

    def _compute_db_fingerprint(self) -> Optional[str]:
        """Cheap change signal for the DB: size + mtime of the main file and
        its WAL/SHM sidecars. Any write bumps mtime, so this never reports
        "unchanged" for a real change (it may over-report changes after a
        touch, which only costs one redundant snapshot). Returns None on any
        stat error — callers then never skip."""
        import os
        parts = []
        try:
            for suffix in ("", "-wal", "-shm"):
                p = f"{self.db_path}{suffix}"
                try:
                    st = os.stat(p)
                    parts.append(f"{suffix or 'db'}:{st.st_size}:{st.st_mtime_ns}")
                except FileNotFoundError:
                    parts.append(f"{suffix or 'db'}:absent")
            return "|".join(parts) if parts else None
        except Exception:  # noqa: BLE001
            return None

    async def snapshot_if_changed(self) -> Dict[str, SyncResult]:
        """Change-aware snapshot for the SCHEDULED backup cron (#1674 P3).

        Skips the whole pass when the DB fingerprint is unchanged since the last
        completed snapshot — an idle agent should not re-dump its entire DB
        (notably the full S3 upload) every cycle. Records the fingerprint only
        after a successful snapshot so a fully-failed attempt is retried. The
        fingerprint is conservative: it never skips a real change. Explicit
        ``!backup`` and shutdown keep using ``force_snapshot`` (always runs)."""
        fingerprint = self._compute_db_fingerprint()
        if (
            fingerprint is not None
            and self._state is not None
            and self._state.last_snapshot is not None
            and self._state.last_fingerprint == fingerprint
        ):
            logger.debug("Snapshot skipped — DB unchanged since last snapshot")
            return {
                "__unchanged__": SyncResult(
                    success=True,
                    target_name="__unchanged__",
                    bytes_synced=0,
                    frames_synced=0,
                    timestamp=datetime.now(timezone.utc),
                )
            }
        results = await self.force_snapshot()
        # Advance the fingerprint ONLY when every target succeeded. If any
        # target failed (transient per-target outage), leave the fingerprint
        # so the next cycle retries rather than skipping the failed target as
        # "unchanged" until some unrelated DB write happens to bump the
        # fingerprint. (No targets → all() is True → record, nothing to retry.)
        if (
            fingerprint is not None
            and self._state is not None
            and all(r.success for r in results.values())
        ):
            self._state.last_fingerprint = fingerprint
            await self._save_state()
        return results

    async def force_snapshot(self) -> Dict[str, SyncResult]:
        """Snapshot to all targets. Called on shutdown, scheduled backup, or !backup."""
        results = {}
        for target in self._targets:
            try:
                result = await target.sync_snapshot(self.db_path)
                results[target.name] = result
                self._update_stats(result)
                if self.on_sync:
                    self.on_sync(result)
            except Exception as e:
                logger.error(f"Snapshot failed for {target.name}: {e}")
                if self.on_error:
                    self.on_error(target.name, e)
                results[target.name] = SyncResult(
                    success=False,
                    target_name=target.name,
                    bytes_synced=0,
                    frames_synced=0,
                    timestamp=datetime.now(timezone.utc),
                    error=str(e),
                )

        if self._state:
            self._state.last_snapshot = datetime.now(timezone.utc)
            await self._save_state()

        return results

    async def restore_by_trust(self, dest_path: Path) -> Optional[SyncResult]:
        """Restore from the most trusted available target.

        Walks targets in trust order. First successful restore wins.
        """
        for target in self.targets_by_trust:
            if not hasattr(target, "restore_snapshot"):
                continue
            try:
                healthy = await target.health_check()
                if not healthy:
                    logger.info(f"Skipping restore from {target.name} (unhealthy)")
                    continue
                result = await target.restore_snapshot(dest_path)
                if result and result.success:
                    logger.info(
                        f"Restored from {target.name} "
                        f"(trust: {target.trust_tier.name}, "
                        f"{result.bytes_synced} bytes)"
                    )
                    return result
            except Exception as e:
                logger.warning(f"Restore from {target.name} failed: {e}")
                continue

        logger.warning("No target could provide a snapshot for restore")
        return None

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

    def _update_stats(self, result: SyncResult) -> None:
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
        if self.state_file.exists():
            try:
                with open(self.state_file, encoding="utf-8") as f:
                    data = json.load(f)
                self._state = SyncState.from_dict(data)
            except Exception as e:
                logger.warning(f"Failed to load sync state: {e}")
                self._state = SyncState(db_path=str(self.db_path))
        else:
            self._state = SyncState(db_path=str(self.db_path))

    async def _save_state(self) -> None:
        if self._state:
            try:
                with open(self.state_file, "w", encoding="utf-8") as f:
                    json.dump(self._state.to_dict(), f, indent=2)
            except Exception as e:
                logger.error(f"Failed to save sync state: {e}")
