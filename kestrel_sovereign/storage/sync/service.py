"""
Sync Service

Event-driven snapshot service for SQLite persistence.
Snapshots are triggered by the scheduled backup and explicit requests rather
than continuous WAL polling. Shutdown uploads nothing: it records that a
remote snapshot is owed, and the next scheduled backup sends it, or the
service itself once the snapshot is past its deadline (#4028).
"""

import asyncio
import json
import logging
import math
import os
import sqlite3
import tempfile
from contextlib import closing
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any

from kestrel_sovereign._async_ownership import await_owned_task, owned_native_wait
from kestrel_sovereign.execution_custody import (
    execution_terminal_error,
    is_execution_control_error,
)
from kestrel_sovereign.storage.sync.quota import StorageQuota
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


def _owed_snapshot_deadline(value: object) -> float:
    """Validate an owed-snapshot deadline: finite seconds, not negative."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"owed snapshot deadline must be a number of seconds, not {value!r}")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"owed snapshot deadline must be finite and not negative, not {value!r}")
    return float(value)


OWED_SNAPSHOT_DEADLINE_ENV = "KESTREL_OWED_SNAPSHOT_DEADLINE_S"
#: The scheduled ``backup_snapshot`` runs every four hours; an hour more.
_OWED_SNAPSHOT_DEADLINE_FALLBACK_SECONDS = 5 * 3600.0


def owed_snapshot_deadline_from_environment() -> float:
    """The configured owed-snapshot deadline; refuses a malformed value."""
    raw = os.environ.get(OWED_SNAPSHOT_DEADLINE_ENV)
    if raw is None:
        return _OWED_SNAPSHOT_DEADLINE_FALLBACK_SECONDS
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(
            f"{OWED_SNAPSHOT_DEADLINE_ENV} must be a number of seconds, not {raw!r}"
        ) from error
    return _owed_snapshot_deadline(value)


#: How long a remote snapshot a shutdown recorded as owed may stay owed
#: before the sync service sends it itself (#4028). The scheduled
#: ``backup_snapshot`` normally sends it first. This is the delivery for an
#: agent that schedules no backup (one whose features exclude
#: SchedulerFeature, or whose backup schedule was paused).
#: ``KESTREL_OWED_SNAPSHOT_DEADLINE_S`` overrides it; 0 sends an owed
#: snapshot as soon as the service starts.
DEFAULT_OWED_SNAPSHOT_DEADLINE_SECONDS = owed_snapshot_deadline_from_environment()


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
    # When a shutdown last recorded that the remote copy is behind the
    # database: shutdown uploads nothing (#4028). Cleared by the first pass
    # that then reaches every target; None when nothing is owed.
    remote_snapshot_owed_since: Optional[datetime] = None
    # Counts the records. A pass may clear the marker only if no record was
    # made while it ran: the time alone cannot tell two records apart, since
    # a later record keeps the earlier time.
    remote_snapshot_owed_generation: int = 0
    stats: SyncStats = field(default_factory=SyncStats)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "db_path": self.db_path,
            "targets": self.targets,
            "last_snapshot": self.last_snapshot.isoformat() if self.last_snapshot else None,
            "last_fingerprint": self.last_fingerprint,
            "last_snapshot_targets": self.last_snapshot_targets,
            "remote_snapshot_owed_since": (
                self.remote_snapshot_owed_since.isoformat()
                if self.remote_snapshot_owed_since
                else None
            ),
            "remote_snapshot_owed_generation": self.remote_snapshot_owed_generation,
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
            remote_snapshot_owed_since=(
                datetime.fromisoformat(data["remote_snapshot_owed_since"])
                if data.get("remote_snapshot_owed_since")
                else None
            ),
            remote_snapshot_owed_generation=data.get(
                "remote_snapshot_owed_generation", 0
            ),
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


class SyncServiceStoppingError(RuntimeError):
    """A snapshot was asked for after the service began to stop."""


@dataclass
class SnapshotPass:
    """One ``force_snapshot`` pass in flight, on the task that asked for it."""

    task: Optional[asyncio.Task]
    #: Completes when the pass has left ``force_snapshot``; ``stop()`` waits
    #: on it rather than on the requester's whole task.
    done: "asyncio.Future[None]"
    #: The target the pass is uploading to now, if any.
    target: Optional[str] = None
    #: Why the service cancelled the pass, once it has.
    cancelled_at: Optional[str] = None
    #: Whether the pass has consumed that cancellation.
    acknowledged: bool = False
    #: Set once every target is done; the pass is no longer cancellable.
    settling: bool = False
    #: What the pass raised that is not its requester's own doing:
    #: process-control evidence, or a cancellation nobody requested (a
    #: native callback's). ``stop()`` reports it.
    retained_error: Optional[BaseException] = None


class SyncService:
    """
    Event-driven snapshot service.

    A thin registry of sync targets with a force_snapshot() method.
    Snapshots are triggered by:
      - Scheduled task (every N hours)
      - Explicit request

    Shutdown uploads nothing. It records that a remote snapshot is owed
    (``record_remote_snapshot_owed``); the next scheduled backup sends it and
    clears the record (#4028). A snapshot still owed
    ``owed_snapshot_deadline_seconds`` after the record is sent by the
    service's own worker, so an agent that schedules no backup still sends
    it (``_deliver_owed_snapshot``). A pass already in flight when the service
    stops is waited for, until the host's shutdown budget cancels it
    (``cancel_inflight_snapshots``).

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
        owed_snapshot_deadline_seconds: float = DEFAULT_OWED_SNAPSHOT_DEADLINE_SECONDS,
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
        self.owed_snapshot_deadline_seconds = _owed_snapshot_deadline(
            owed_snapshot_deadline_seconds
        )
        self._state: Optional[SyncState] = None
        self._poll_task: Optional[asyncio.Task] = None
        # Sends a snapshot still owed at its deadline; see
        # ``_deliver_owed_snapshot``.
        self._owed_delivery_task: Optional[asyncio.Task] = None
        self._passes: Dict[int, SnapshotPass] = {}
        self._stopping = False

    def add_target(self, target: SyncTarget) -> None:
        """Add a sync target."""
        if target.trust_tier in REMOTE_SYNC_TRUST_TIERS:
            decision = self._remote_target_policy_decision()
            if decision is None:
                self._append_target(target)
                return
            if not decision.allowed:
                self._record_policy_skip(target.name, decision.reason, kind=target.kind)
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

    def _record_policy_skip(
        self, target_name: str, reason: Optional[str], *, kind: str = ""
    ) -> None:
        """Record a destination the policy denied; ``kind`` is the target's
        when it was constructed, and unknown when the policy refused before
        construction (``add_remote_target``)."""
        self._policy_skips[target_name] = SyncResult(
            success=True,
            target_name=target_name,
            bytes_synced=0,
            frames_synced=0,
            timestamp=datetime.now(timezone.utc),
            metadata={"skipped": True, "policy_denied": True, "reason": reason},
            kind=kind,
            attempted=False,
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

    @property
    def retention_policy(self) -> RetentionPolicy:
        """The policy every target's prune applies; a target that prunes
        before uploading (Lighthouse near its quota) is built with it."""
        return self._retention_policy

    def storage_quotas(self) -> Dict[str, StorageQuota]:
        """Each configured target's storage quota, by target name, as the
        target last read it: this process's reading, else the one the state
        file kept from an earlier one. A target that never read one is
        absent (#3894)."""
        quotas: Dict[str, StorageQuota] = {}
        for target in self._targets:
            quota = getattr(target, "storage_quota", None)
            if not isinstance(quota, StorageQuota) and self._state is not None:
                recorded = (
                    self._state.targets.get(target.name)
                    if isinstance(self._state.targets, dict)
                    else None
                )
                if isinstance(recorded, dict):
                    quota = StorageQuota.from_dict(recorded.get("storage_quota"))
            if isinstance(quota, StorageQuota):
                quotas[target.name] = quota
        return quotas

    def _record_storage_quota(self, target: SyncTarget) -> None:
        """Keep the target's latest quota reading in the state file, so the
        health check still has it after a restart."""
        quota = getattr(target, "storage_quota", None)
        if not isinstance(quota, StorageQuota) or self._state is None:
            return
        if not isinstance(self._state.targets, dict):
            self._state.targets = {}
        recorded = self._state.targets.get(target.name)
        if not isinstance(recorded, dict):
            recorded = self._state.targets[target.name] = {}
        recorded["storage_quota"] = quota.to_dict()

    async def start(self) -> None:
        """Load state and optionally start periodic sync if wal_sync_interval is set."""
        await self._load_state()
        self._stopping = False
        if self.wal_sync_interval and self.wal_sync_interval > 0:
            self._poll_task = asyncio.create_task(self._periodic_sync())
        logger.info(
            f"Sync service ready for {self.db_path} "
            f"({len(self._targets)} targets: "
            f"{', '.join(f'{t.name}({t.trust_tier.name})' for t in self.targets_by_trust)})"
        )
        owed = self.remote_snapshot_owed
        if owed is not None and self._targets:
            due = owed + timedelta(seconds=self.owed_snapshot_deadline_seconds)
            delay = max(0.0, (due - datetime.now(timezone.utc)).total_seconds())
            logger.info(
                "A remote snapshot of %s has been owed since %s; the next "
                "scheduled backup_snapshot sends it, or this service at %s",
                self.db_path,
                owed.isoformat(),
                due.isoformat(),
            )
            self._owed_delivery_task = asyncio.create_task(
                self._deliver_owed_snapshot(delay),
                name=f"sync_owed_snapshot:{self.db_path}",
            )

    @property
    def remote_snapshot_owed(self) -> Optional[datetime]:
        """When a shutdown recorded that the remote copy is behind, or None."""
        return self._state.remote_snapshot_owed_since if self._state else None

    async def record_remote_snapshot_owed(self) -> bool:
        """Record that the remote copy is behind the database; True if owed.

        Shutdown calls this instead of uploading (#4028). An agent database
        is gigabytes and its upload takes many minutes, far past any stop the
        host can wait for, so a shutdown upload was always cut off by SIGKILL.
        The record survives the restart in the state file; the next scheduled
        ``backup_snapshot`` sends the snapshot even when the database looks
        unchanged, and clears the record once every target has it. A
        snapshot no backup has sent by ``owed_snapshot_deadline_seconds``
        after the record is sent by the service itself after its next
        ``start()``. An
        earlier record is kept, so the time says how long the copy has been
        behind. A service with no upload target owes nothing, and neither
        does a database unchanged since a pass that reached every target.
        """
        if self._state is None or not self._targets:
            return False
        if (
            self._state.remote_snapshot_owed_since is None
            and self._unchanged_since_complete_pass(self._compute_db_fingerprint())
        ):
            logger.info(
                "Remote copy of %s is current; no snapshot owed", self.db_path
            )
            return False
        if self._state.remote_snapshot_owed_since is None:
            self._state.remote_snapshot_owed_since = datetime.now(timezone.utc)
        self._state.remote_snapshot_owed_generation += 1
        await self._save_state()
        logger.info(
            "Remote snapshot of %s owed to %s since %s; shutdown uploads "
            "nothing, the next scheduled backup_snapshot sends it, or this "
            "service after its next start once the snapshot is past its "
            "deadline",
            self.db_path,
            ", ".join(self.targets),
            self._state.remote_snapshot_owed_since.isoformat(),
        )
        return True

    def cancel_inflight_snapshots(self, reason: str) -> List[str]:
        """Cancel every snapshot pass in flight; describe each one cancelled.

        The host calls this when its shutdown budget runs out (#4028) and a
        stopping agent is still waiting on an upload. Each cancelled pass
        records its current target as failed with ``reason``, skips the rest
        and returns its results normally, so its requester and ``stop()``
        both see a finished pass rather than a cancellation. An owed
        snapshot stays owed.
        """
        cancelled = []
        for snapshot_pass in list(self._passes.values()):
            if (
                snapshot_pass.cancelled_at is not None
                or snapshot_pass.settling
                or snapshot_pass.task is None
            ):
                continue
            snapshot_pass.cancelled_at = reason
            description = (
                f"snapshot to {snapshot_pass.target} cancelled at {reason}"
                if snapshot_pass.target is not None
                else f"snapshot of {self.db_path} cancelled at {reason}"
            )
            logger.warning("%s", description)
            snapshot_pass.task.cancel()
            cancelled.append(description)
        return cancelled

    async def stop(self) -> None:
        """Cancel the service's own workers, wait for passes in flight, and
        save state.

        The service's workers (the periodic sync and the owed-snapshot
        delivery) are cancelled, their uploads with them. A pass some other
        task started (a scheduled backup) is not cancelled here: it is remote
        work the stop waits for, until the host's shutdown budget cancels it
        through ``cancel_inflight_snapshots``.
        """
        self._stopping = True
        workers = [
            worker
            for worker in (self._poll_task, self._owed_delivery_task)
            if worker is not None
        ]
        # Captured now: a pass that finishes while the workers below are
        # joined still reports what it retained. Nothing new is admitted
        # after this. A worker's own pass ends with the worker.
        inflight = [
            snapshot_pass
            for snapshot_pass in self._passes.values()
            if snapshot_pass.task is None
            or all(snapshot_pass.task is not worker for worker in workers)
        ]
        failures: List[BaseException] = []
        if self._poll_task is not None:
            failure = await self._stop_worker(self._poll_task)
            if failure is None:
                self._poll_task = None
            else:
                failures.append(failure)
        if self._owed_delivery_task is not None:
            failure = await self._stop_worker(self._owed_delivery_task)
            if failure is None:
                self._owed_delivery_task = None
            else:
                failures.append(failure)
        retained = await self._join_snapshot_passes(inflight)
        if retained is not None:
            failures.append(retained)
        await self._save_state()
        if len(failures) > 1:
            raise execution_terminal_error(*failures, retain_originals=True)
        if failures:
            raise failures[0]
        logger.info("Sync service stopped")

    @staticmethod
    async def _stop_worker(worker: asyncio.Task) -> Optional[BaseException]:
        """Cancel and join one of the service's own workers.

        Returns the worker's failure, or None once it is retired. Raises the
        stop's own withdrawal.
        """
        if not worker.done():
            worker.cancel()
        outcome = await await_owned_task(worker)
        terminal = execution_terminal_error(
            outcome.error, outcome.cancellation,
            retain_originals=True, prefer_cancellation=True,
        )
        # Only our ordinary request to stop the worker is acknowledged.
        # Native debt and the stopping caller's withdrawal remain control;
        # the exact failed worker stays retained for a later stop observer.
        if terminal is not None and (
            is_execution_control_error(terminal)
            or not isinstance(terminal, asyncio.CancelledError)
            or outcome.cancellation is not None
        ):
            if outcome.cancellation is not None:
                raise terminal
            # The worker's own failure does not excuse the stop from waiting
            # for the other passes still in flight.
            return terminal
        return None

    async def _deliver_owed_snapshot(self, delay: float) -> None:
        """Send a snapshot still owed once its deadline has passed (#4028).

        Shutdown records a snapshot as owed and uploads nothing. The
        scheduled ``backup_snapshot`` normally sends it; an agent that
        schedules no backup has nothing else to. This worker sleeps until
        ``owed_snapshot_deadline_seconds`` after the record and then sends
        it, unless a pass has cleared it meanwhile. One attempt per start:
        a snapshot it cannot send stays owed for the next scheduled backup
        or the next start.
        """
        await asyncio.sleep(delay)
        owed = self.remote_snapshot_owed
        if owed is None:
            return
        logger.info(
            "Sending the remote snapshot of %s owed since %s: no backup has "
            "sent it within %.0fs",
            self.db_path,
            owed.isoformat(),
            self.owed_snapshot_deadline_seconds,
        )
        try:
            results = await self.snapshot_if_changed()
        except SyncServiceStoppingError:
            return
        except Exception as error:
            if is_execution_control_error(error):
                raise
            logger.error(
                "Owed remote snapshot of %s failed: %s: %s",
                self.db_path,
                type(error).__name__,
                error,
            )
            return
        if self.remote_snapshot_owed is not None:
            failed = sorted(
                name for name, result in results.items() if not result.success
            )
            logger.warning(
                "Remote snapshot of %s is still owed: %s",
                self.db_path,
                ", ".join(failed) if failed else "a later record keeps it owed",
            )

    async def _join_snapshot_passes(
        self, passes: List[SnapshotPass]
    ) -> Optional[BaseException]:
        """Wait for each of ``passes`` to finish; return what they retained."""
        current = asyncio.current_task()
        retained: List[BaseException] = []
        for snapshot_pass in passes:
            if snapshot_pass.task is current:
                continue
            # Declared as a join of the requester's task, so work inside the
            # pass that observes shutdown completion does not wait on this
            # stop in turn.
            async with owned_native_wait(
                () if snapshot_pass.task is None else (snapshot_pass.task,)
            ):
                await asyncio.wait({snapshot_pass.done})
            if snapshot_pass.retained_error is not None:
                retained.append(snapshot_pass.retained_error)
        if not retained:
            return None
        return execution_terminal_error(*retained, retain_originals=True)

    async def _periodic_sync(self) -> None:
        """Background loop that triggers force_snapshot at wal_sync_interval."""
        while True:
            await asyncio.sleep(self.wal_sync_interval)
            try:
                await self.force_snapshot()
            except Exception as e:
                if is_execution_control_error(e):
                    raise
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
        requests keep using ``force_snapshot`` (always runs), and so does a
        snapshot a shutdown recorded as owed (#4028): it is sent even when the
        database looks unchanged."""
        fingerprint = self._compute_db_fingerprint()
        if (
            self._state is not None
            and self._state.remote_snapshot_owed_since is None
            and self._unchanged_since_complete_pass(fingerprint)
        ):
            logger.debug("Snapshot skipped — DB unchanged since last snapshot")
            return {
                "__unchanged__": SyncResult(
                    success=True,
                    target_name="__unchanged__",
                    bytes_synced=0,
                    frames_synced=0,
                    timestamp=datetime.now(timezone.utc),
                    attempted=False,
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

    def _unchanged_since_complete_pass(self, fingerprint: Optional[str]) -> bool:
        """Whether ``fingerprint`` shows the database as it was at the last
        pass that reached every current target. An unreadable fingerprint
        (None) never counts as unchanged."""
        current_targets = {t.name for t in self._targets}
        return (
            fingerprint is not None
            and self._state is not None
            and self._state.last_snapshot is not None
            and self._state.last_fingerprint == fingerprint
            # Every current target must already be covered — else a newly-added
            # destination would never get its baseline snapshot on an idle DB.
            and current_targets.issubset(set(self._state.last_snapshot_targets))
        )

    async def force_snapshot(self) -> Dict[str, SyncResult]:
        """Snapshot to all targets: the scheduled backup, the periodic sync or
        an explicit request. Shutdown never calls this; it records the
        snapshot as owed instead (#4028).

        The pass runs on the caller's task, which can cancel it as before.
        ``stop()`` waits for it, and the host's shutdown budget cancels it
        through ``cancel_inflight_snapshots``."""
        snapshot_pass = self._admit_snapshot_pass()
        try:
            return await self._snapshot_to_targets(snapshot_pass)
        except BaseException as error:
            if is_execution_control_error(error) or (
                isinstance(error, asyncio.CancelledError)
                and snapshot_pass.task is not None
                and not snapshot_pass.task.cancelling()
            ):
                snapshot_pass.retained_error = error
            raise
        finally:
            self._passes.pop(id(snapshot_pass), None)
            if not snapshot_pass.done.done():
                snapshot_pass.done.set_result(None)

    def _admit_snapshot_pass(self) -> SnapshotPass:
        if self._stopping:
            raise SyncServiceStoppingError(
                f"sync service for {self.db_path} is stopping; no new snapshot"
            )
        snapshot_pass = SnapshotPass(
            task=asyncio.current_task(),
            done=asyncio.get_running_loop().create_future(),
        )
        self._passes[id(snapshot_pass)] = snapshot_pass
        return snapshot_pass

    @staticmethod
    def _acknowledges_cancellation(
        snapshot_pass: SnapshotPass, cancellation: asyncio.CancelledError
    ) -> bool:
        """Whether ``cancellation`` is the service's own and nothing else's.

        Only the cancellation ``cancel_inflight_snapshots`` requested is
        consumed. A requester's cancellation, or process-control evidence,
        still propagates.
        """
        if (
            snapshot_pass.cancelled_at is None
            or snapshot_pass.acknowledged
            or is_execution_control_error(cancellation)
        ):
            return False
        snapshot_pass.acknowledged = True
        task = snapshot_pass.task
        return task is None or task.uncancel() == 0

    def _every_target_reached(self, results: Dict[str, SyncResult]) -> bool:
        if not self._targets:
            return False
        # A destination the policy denies is owed nothing; its result is a
        # success that was never attempted.
        for target in self._targets:
            result = results.get(target.name)
            if result is None or not result.success:
                return False
        return True

    async def _snapshot_to_targets(
        self, snapshot_pass: SnapshotPass
    ) -> Dict[str, SyncResult]:
        # A record made while this pass runs is about a later state of the
        # database than the one this pass copies; only an earlier one can be
        # cleared by it.
        owed_at_start = self.remote_snapshot_owed
        owed_generation_at_start = (
            self._state.remote_snapshot_owed_generation if self._state else 0
        )
        results = dict(self._policy_skips)
        successful_snapshots = 0
        for target in self._targets:
            if snapshot_pass.cancelled_at is not None:
                # The previous target swallowed the service's cancellation.
                break
            if target.trust_tier in REMOTE_SYNC_TRUST_TIERS:
                decision = self._remote_target_policy_decision()
                if decision is not None and not decision.allowed:
                    self._record_policy_skip(target.name, decision.reason, kind=target.kind)
                    results[target.name] = self._policy_skips[target.name]
                    logger.warning(
                        "Remote sync target skipped by policy before upload: %s (%s)",
                        target.name,
                        decision.reason,
                    )
                    continue
            snapshot_pass.target = target.name
            result: Optional[SyncResult] = None
            try:
                # The result is what leaves this loop; the target does not.
                # Carry the target's bounded kind on a copy so the scheduled
                # backup can name which kind of destination failed (#3189);
                # a copy, so a target's own (possibly frozen) result object is
                # never mutated inside the failure envelope.
                result = replace(
                    await target.sync_snapshot(self.db_path), kind=target.kind
                )
                if result.success:
                    successful_snapshots += 1
                    await self._prune_after_success(target, result)
                results[target.name] = result
                self._update_stats(result)
                if self.on_sync:
                    self.on_sync(result)
            except asyncio.CancelledError as cancellation:
                if not self._acknowledges_cancellation(snapshot_pass, cancellation):
                    raise
                # An upload that finished before a cancelled prune still
                # reached its target.
                if result is not None and self.on_sync:
                    self.on_sync(result)
                if result is None:
                    result = SyncResult(
                        success=False,
                        target_name=target.name,
                        bytes_synced=0,
                        frames_synced=0,
                        timestamp=datetime.now(timezone.utc),
                        error=f"cancelled at {snapshot_pass.cancelled_at}",
                        kind=target.kind,
                    )
                results[target.name] = result
                self._update_stats(result)
                self._record_storage_quota(target)
                break
            except Exception as e:
                if is_execution_control_error(e):
                    raise
                logger.error(
                    "Snapshot failed for %s: %s: %s", target.name, type(e).__name__, e
                )
                if self.on_error:
                    self.on_error(target.name, e)
                results[target.name] = SyncResult(
                    success=False,
                    target_name=target.name,
                    bytes_synced=0,
                    frames_synced=0,
                    timestamp=datetime.now(timezone.utc),
                    error=f"{type(e).__name__}: {e}",
                    kind=target.kind,
                )
            self._record_storage_quota(target)
        snapshot_pass.target = None
        snapshot_pass.settling = True
        if (
            snapshot_pass.cancelled_at is not None
            and not snapshot_pass.acknowledged
            and snapshot_pass.task is not None
        ):
            # A target swallowed the cancellation; take back the request.
            snapshot_pass.acknowledged = True
            snapshot_pass.task.uncancel()

        if self._state:
            self._warn_if_repeated_empty_syncs(successful_snapshots)
            self._state.last_snapshot = datetime.now(timezone.utc)
            if (
                owed_at_start is not None
                and self._state.remote_snapshot_owed_generation
                == owed_generation_at_start
                and self._every_target_reached(results)
            ):
                self._state.remote_snapshot_owed_since = None
                logger.info(
                    "Owed remote snapshot of %s (owed since %s) sent to %s",
                    self.db_path,
                    owed_at_start.isoformat(),
                    ", ".join(self.targets),
                )
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
        # ``with`` on a sqlite3 connection ends a transaction; it does not
        # close the connection, so each backup leaked one.
        with closing(sqlite3.connect(str(self.db_path))) as conn:
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
            if is_execution_control_error(e):
                raise
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
