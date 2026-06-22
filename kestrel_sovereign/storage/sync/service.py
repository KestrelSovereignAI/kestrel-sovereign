"""
Sync Service

Event-driven snapshot service for SQLite persistence.
Snapshots are triggered by lifecycle events (shutdown, scheduled backup)
rather than continuous WAL polling.
"""

import asyncio
import json
import logging
import os
import sqlite3
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any

from kestrel_sovereign.storage.sync.retention import (
    RetentionPolicy,
    load_retention_policy,
)
from kestrel_sovereign.storage.sync.targets import SyncTarget, SyncResult, TrustTier
from kestrel_sovereign.storage.tiered_manager import privacy_allows_remote_tiers

logger = logging.getLogger(__name__)

REMOTE_SYNC_TRUST_TIERS = {
    TrustTier.SOVEREIGN,
    TrustTier.FEDERATED,
    TrustTier.DELEGATED,
    TrustTier.EXPEDIENT,
}

DENIED_IDENTITY_TOKENS = (
    "malicious_agent",
    "no_constitution_agent",
    "test_ensemble_agent",
    "test_storage_agent",
    "gcs-live-test",
    "codex-live-check",
)

DENIED_PATH_PARTS = {
    "gcs-live-test",
    "codex-live-check",
}

DEFAULT_EMPTY_DB_WARNING_THRESHOLD = 5


@dataclass(frozen=True)
class RemoteTierPolicyContext:
    """Identity and instance facts needed before constructing remote targets."""

    identity: Optional[str] = None
    db_path: Optional[str] = None
    is_test_instance: bool = False
    has_constitution_anchor: Optional[bool] = None
    is_sovereign_identity: bool = True
    privacy_mode: Optional[str] = None


@dataclass(frozen=True)
class RemoteTierPolicyDecision:
    allowed: bool
    reason: Optional[str] = None


def _path_is_fixture_or_temp(db_path: Optional[str]) -> bool:
    if not db_path:
        return False
    path = Path(db_path)
    name = path.name.lower()
    if name in {"test.db", "binary_test.bin", "roundtrip.txt"}:
        return True
    if name.startswith("test") and name.endswith(".db"):
        return True
    if any(part.lower() in DENIED_PATH_PARTS for part in path.parts):
        return True
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path.absolute()
    temp_root = Path(tempfile.gettempdir()).resolve()
    return resolved == temp_root or temp_root in resolved.parents


def _remote_tiers_allowed(
    context: RemoteTierPolicyContext,
) -> RemoteTierPolicyDecision:
    """Authoritative hard gate for production remote backup tiers."""
    identity = (context.identity or "").strip()
    identity_lower = identity.lower()

    if context.privacy_mode and not privacy_allows_remote_tiers(context.privacy_mode):
        return RemoteTierPolicyDecision(False, "privacy_mode_local_only")
    if context.is_test_instance:
        return RemoteTierPolicyDecision(False, "test_instance")
    if identity_lower.startswith("did:test:"):
        return RemoteTierPolicyDecision(False, "test_did")
    if any(token in identity_lower for token in DENIED_IDENTITY_TOKENS):
        return RemoteTierPolicyDecision(False, "denied_fixture_identity")
    if not context.is_sovereign_identity:
        return RemoteTierPolicyDecision(False, "non_sovereign_identity")
    if context.has_constitution_anchor is False:
        return RemoteTierPolicyDecision(False, "missing_constitution_anchor")
    if _path_is_fixture_or_temp(context.db_path):
        return RemoteTierPolicyDecision(False, "fixture_or_temp_db_path")
    return RemoteTierPolicyDecision(True)


@dataclass
class SyncStats:
    """Statistics for sync operations."""
    total_syncs: int = 0
    successful_syncs: int = 0
    failed_syncs: int = 0
    empty_database_syncs: int = 0
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
    # Target names that were successfully covered at last_fingerprint. The
    # change-aware skip only applies when every CURRENT target is in this set —
    # otherwise a newly-added backup destination would never get its baseline
    # snapshot on an unchanged DB (#1674 P3 codex round 2).
    last_snapshot_targets: List[str] = field(default_factory=list)
    stats: SyncStats = field(default_factory=SyncStats)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "db_path": self.db_path,
            "targets": self.targets,
            "last_snapshot": self.last_snapshot.isoformat() if self.last_snapshot else None,
            "last_fingerprint": self.last_fingerprint,
            "last_snapshot_targets": self.last_snapshot_targets,
            "stats": {
                "total_syncs": self.stats.total_syncs,
                "successful_syncs": self.stats.successful_syncs,
                "failed_syncs": self.stats.failed_syncs,
                "empty_database_syncs": self.stats.empty_database_syncs,
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
            last_snapshot_targets=data.get("last_snapshot_targets", []),
            stats=SyncStats(
                total_syncs=stats_data.get("total_syncs", 0),
                successful_syncs=stats_data.get("successful_syncs", 0),
                failed_syncs=stats_data.get("failed_syncs", 0),
                empty_database_syncs=stats_data.get("empty_database_syncs", 0),
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
        policy_context: Optional[RemoteTierPolicyContext] = None,
        policy_context_provider: Optional[Callable[[], RemoteTierPolicyContext]] = None,
        retention_policy: Optional[RetentionPolicy] = None,
        empty_db_warning_threshold: int = DEFAULT_EMPTY_DB_WARNING_THRESHOLD,
    ):
        self.db_path = Path(db_path)
        self.state_file = Path(state_file) if state_file else Path(f"{db_path}.sync")
        self.wal_sync_interval = wal_sync_interval
        self.on_sync = on_sync
        self.on_error = on_error
        self._targets: List[SyncTarget] = []
        self._policy_skips: Dict[str, SyncResult] = {}
        self._policy_context = policy_context
        self._policy_context_provider = policy_context_provider
        self._retention_policy = retention_policy or load_retention_policy()
        self.empty_db_warning_threshold = empty_db_warning_threshold
        self._state: Optional[SyncState] = None
        self._poll_task: Optional[asyncio.Task] = None

    def add_target(self, target: SyncTarget) -> None:
        """Add a sync target."""
        if target.trust_tier in REMOTE_SYNC_TRUST_TIERS:
            decision = self._remote_target_policy_decision()
            if decision is None:
                self._append_target(target)
                return
            if not decision.allowed:
                self._record_policy_skip(target.name, decision.reason)
                logger.warning(
                    "Remote sync target skipped by policy before upload: %s (%s)",
                    target.name,
                    decision.reason,
                )
                return
        self._append_target(target)

    def _append_target(self, target: SyncTarget) -> None:
        self._targets.append(target)
        logger.info(f"Added sync target: {target.name} ({target.trust_tier.name})")

    def add_remote_target(
        self,
        target_name: str,
        trust_tier: TrustTier,
        factory: Callable[[], SyncTarget],
    ) -> bool:
        """Add a remote target, evaluating policy before constructing it."""
        if trust_tier in REMOTE_SYNC_TRUST_TIERS:
            decision = self._remote_target_policy_decision()
            if decision is None:
                self._append_target(factory())
                return True
            if not decision.allowed:
                self._record_policy_skip(target_name, decision.reason)
                logger.warning(
                    "Remote sync target skipped by policy before construction: %s (%s)",
                    target_name,
                    decision.reason,
                )
                return False
        self._append_target(factory())
        return True

    def _record_policy_skip(self, target_name: str, reason: Optional[str]) -> None:
        self._policy_skips[target_name] = SyncResult(
            success=True,
            target_name=target_name,
            bytes_synced=0,
            frames_synced=0,
            timestamp=datetime.now(timezone.utc),
            metadata={"skipped": True, "policy_denied": True, "reason": reason},
        )

    def _current_policy_context(self) -> Optional[RemoteTierPolicyContext]:
        if self._policy_context_provider:
            return self._policy_context_provider()
        return self._policy_context

    def _remote_target_policy_decision(self) -> Optional[RemoteTierPolicyDecision]:
        context = self._current_policy_context()
        if context is None:
            return None
        return _remote_tiers_allowed(context)

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
        return len(self._targets) > 0 or bool(self._policy_skips)

    @property
    def has_work(self) -> bool:
        """True when snapshots will produce upload or policy-skip results."""
        return self.is_running

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
        current_targets = {t.name for t in self._targets}
        if (
            fingerprint is not None
            and self._state is not None
            and self._state.last_snapshot is not None
            and self._state.last_fingerprint == fingerprint
            # Every current target must already be covered — else a newly-added
            # destination would never get its baseline snapshot on an idle DB.
            and current_targets.issubset(set(self._state.last_snapshot_targets))
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
            and results
            and all(r.success for r in results.values())
        ):
            completed_fingerprint = self._compute_db_fingerprint() or fingerprint
            self._state.last_fingerprint = completed_fingerprint
            self._state.last_snapshot_targets = sorted(
                r.target_name for r in results.values() if r.success
            )
            await self._save_state()
        return results

    async def force_snapshot(self) -> Dict[str, SyncResult]:
        """Snapshot to all targets. Called on shutdown, scheduled backup, or !backup."""
        results = dict(self._policy_skips)
        successful_snapshots = 0
        for target in self._targets:
            if target.trust_tier in REMOTE_SYNC_TRUST_TIERS:
                decision = self._remote_target_policy_decision()
                if decision is not None and not decision.allowed:
                    self._record_policy_skip(target.name, decision.reason)
                    results[target.name] = self._policy_skips[target.name]
                    logger.warning(
                        "Remote sync target skipped by policy before upload: %s (%s)",
                        target.name,
                        decision.reason,
                    )
                    continue
            try:
                result = await target.sync_snapshot(self.db_path)
                if result.success:
                    successful_snapshots += 1
                    await self._prune_after_success(target, result)
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
            self._warn_if_repeated_empty_syncs(successful_snapshots)
            self._state.last_snapshot = datetime.now(timezone.utc)
            await self._save_state()

        return results

    def _warn_if_repeated_empty_syncs(self, successful_snapshots: int) -> None:
        if not self._state:
            return
        if successful_snapshots <= 0:
            return
        threshold = self.empty_db_warning_threshold
        if threshold <= 0:
            return
        try:
            is_empty_pending = self._database_is_empty_and_bootstrap_pending()
        except Exception as e:  # noqa: BLE001
            logger.debug("Empty database sync check skipped: %s", e)
            return
        if not is_empty_pending:
            self._state.stats.empty_database_syncs = 0
            return
        self._state.stats.empty_database_syncs += successful_snapshots
        if self._state.stats.empty_database_syncs < threshold:
            return
        logger.warning(
            "Agent has been alive for %s syncs with no conversations. "
            "Possible misconfiguration.",
            self._state.stats.empty_database_syncs,
        )

    def _database_is_empty_and_bootstrap_pending(self) -> bool:
        with sqlite3.connect(str(self.db_path)) as conn:
            conversations = self._table_row_count(conn, "conversation_history")
            episodes = self._table_row_count(conn, "memory_episodes")
            if conversations > 0 or episodes > 0:
                return False
            return self._bootstrap_state_is_pending(conn)

    @staticmethod
    def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table_name,),
        ).fetchone()
        return row is not None

    @classmethod
    def _table_row_count(cls, conn: sqlite3.Connection, table_name: str) -> int:
        if not cls._table_exists(conn, table_name):
            return 0
        if table_name == "conversation_history":
            row = conn.execute("SELECT COUNT(*) FROM conversation_history").fetchone()
        elif table_name == "memory_episodes":
            row = conn.execute("SELECT COUNT(*) FROM memory_episodes").fetchone()
        else:
            raise ValueError(f"unsupported table for sync health check: {table_name}")
        return int(row[0]) if row else 0

    @classmethod
    def _bootstrap_state_is_pending(cls, conn: sqlite3.Connection) -> bool:
        states: List[str] = []
        if cls._table_exists(conn, "agent_metadata"):
            rows = conn.execute(
                """
                SELECT value FROM agent_metadata
                WHERE key = 'bootstrap_state'
                """
            ).fetchall()
            states.extend(str(row[0]).strip().lower() for row in rows if row and row[0])

        if cls._table_exists(conn, "graph_nodes"):
            rows = conn.execute(
                """
                SELECT properties FROM graph_nodes
                WHERE node_type = 'agent'
                """
            ).fetchall()
            for row in rows:
                if not row or not row[0]:
                    continue
                try:
                    properties = json.loads(row[0])
                except (TypeError, json.JSONDecodeError):
                    continue
                state = properties.get("bootstrap_state")
                if state:
                    states.append(str(state).strip().lower())

        return any(state == "pending" for state in states)

    async def _prune_after_success(
        self,
        target: SyncTarget,
        result: SyncResult,
    ) -> None:
        try:
            prune_result = await target.prune(self._retention_policy)
            if prune_result:
                metadata = dict(result.metadata or {})
                metadata["prune"] = prune_result
                result.metadata = metadata
        except Exception as e:  # noqa: BLE001
            logger.warning("Retention prune failed for %s: %s", target.name, e)
            metadata = dict(result.metadata or {})
            metadata["prune_error"] = str(e)
            result.metadata = metadata

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
                    f.flush()
                    os.fsync(f.fileno())
            except Exception as e:
                logger.error(f"Failed to save sync state: {e}")
