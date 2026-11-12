"""The sync service's owed snapshot and its passes in flight (#4028).

Shutdown no longer uploads: it records the remote snapshot as owed, and the
next scheduled ``backup_snapshot`` (``snapshot_if_changed``) sends it. A
snapshot still owed at its deadline is sent by the service's own worker, so
an agent that schedules no backup still sends it. A pass some other task
started is waited for by ``stop()`` until the host's shutdown budget cancels
it by name.
"""

from __future__ import annotations

import asyncio
import errno
import json
import logging
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import pytest

from kestrel_sovereign.storage.sync.service import (
    OWED_SNAPSHOT_DEADLINE_ENV,
    SyncService,
    SyncServiceStoppingError,
    SyncStateWriteError,
    owed_snapshot_deadline_from_environment,
)
from kestrel_sovereign.storage.sync.targets import (
    SyncResult,
    SyncTarget,
    TrustTier,
    _write_consistent_snapshot,
)


class _Target(SyncTarget):
    """A remote target that counts its uploads and can hold one open."""

    kind = "gcs"

    def __init__(self, name: str, *, blocks: bool = False, fails: bool = False):
        self._name = name
        self.blocks = blocks
        self.fails = fails
        self.calls = 0
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    @property
    def name(self) -> str:
        return self._name

    @property
    def trust_tier(self) -> TrustTier:
        return TrustTier.EXPEDIENT

    async def sync_snapshot(self, db_path: Path) -> SyncResult:
        self.calls += 1
        self.entered.set()
        if self.blocks:
            await self.release.wait()
        return SyncResult(
            success=not self.fails,
            target_name=self.name,
            bytes_synced=0 if self.fails else 1,
            frames_synced=0,
            timestamp=datetime.now(timezone.utc),
            error="upload refused" if self.fails else None,
        )

    async def sync_wal(self, wal_path: Path, position: int) -> SyncResult:
        raise AssertionError("the sync service never streams the WAL")

    async def get_latest_position(self):
        return None


def _database(tmp_path: Path) -> Path:
    db = tmp_path / "agent.db"
    with closing(sqlite3.connect(db)) as connection, connection:
        connection.execute("CREATE TABLE payload (value TEXT)")
        connection.execute("INSERT INTO payload VALUES ('first')")
    return db


async def _service(
    db: Path, *targets: SyncTarget, owed_deadline: float = 3600.0
) -> SyncService:
    service = SyncService(str(db), owed_snapshot_deadline_seconds=owed_deadline)
    for target in targets:
        service.add_target(target)
    await service.start()
    return service


def _persisted_owed(db: Path):
    return json.loads(Path(f"{db}.sync").read_text())["remote_snapshot_owed_since"]


@pytest.mark.asyncio
async def test_a_service_without_upload_targets_owes_nothing(tmp_path):
    db = _database(tmp_path)
    service = await _service(db)

    assert await service.record_remote_snapshot_owed() is False
    assert service.remote_snapshot_owed is None


@pytest.mark.asyncio
async def test_the_record_survives_a_restart_and_keeps_its_first_time(tmp_path):
    db = _database(tmp_path)
    target = _Target("gs://owed/agent")
    service = await _service(db, target)

    assert await service.record_remote_snapshot_owed() is True
    first = service.remote_snapshot_owed
    assert first is not None and _persisted_owed(db) == first.isoformat()
    assert await service.record_remote_snapshot_owed() is True
    assert service.remote_snapshot_owed == first, "a later record reset how long it is owed"
    assert target.calls == 0, "recording the snapshot as owed uploaded it"

    restarted = await _service(db, _Target("gs://owed/agent"))
    assert restarted.remote_snapshot_owed == first
    await service.stop()
    await restarted.stop()


def _write(db: Path, value: str) -> None:
    with closing(sqlite3.connect(db)) as connection, connection:
        connection.execute("INSERT INTO payload VALUES (?)", (value,))


@pytest.mark.asyncio
async def test_an_unchanged_database_owes_nothing(tmp_path):
    """The remote copy is current: a pass reached every target since."""
    db = _database(tmp_path)
    target = _Target("gs://owed/agent")
    service = await _service(db, target)
    await service.snapshot_if_changed()

    assert await service.record_remote_snapshot_owed() is False
    assert service.remote_snapshot_owed is None and _persisted_owed(db) is None

    _write(db, "after the backup")
    assert await service.record_remote_snapshot_owed() is True


@pytest.mark.asyncio
async def test_the_scheduled_backup_sends_an_owed_snapshot_of_an_unchanged_database(
    tmp_path,
):
    """A shutdown during a backup leaves the copy behind even when that
    backup then reaches every target and records the database as current."""
    db = _database(tmp_path)
    target = _Target("gs://owed/agent", blocks=True)
    service = await _service(db, target)
    target.release.set()
    await service.snapshot_if_changed()
    _write(db, "before the shutdown")
    target.release.clear()
    target.entered.clear()
    backup = asyncio.create_task(service.snapshot_if_changed())
    await asyncio.wait_for(target.entered.wait(), 3)

    assert await service.record_remote_snapshot_owed() is True
    target.release.set()
    await asyncio.wait_for(backup, 3)
    assert service.remote_snapshot_owed is not None
    calls = target.calls

    results = await service.snapshot_if_changed()

    assert target.calls == calls + 1, "an owed snapshot was skipped as unchanged"
    assert results[target.name].success
    assert service.remote_snapshot_owed is None and _persisted_owed(db) is None
    await service.snapshot_if_changed()
    assert target.calls == calls + 1, (
        "the change-aware skip did not return once nothing was owed"
    )


@pytest.mark.asyncio
async def test_a_record_made_while_a_pass_runs_outlives_an_earlier_record(tmp_path):
    """The record keeps its first time, so only its count can show that a
    second shutdown came after the pass copied the database."""
    db = _database(tmp_path)
    target = _Target("gs://owed/agent", blocks=True)
    service = await _service(db, target)
    await service.record_remote_snapshot_owed()
    first = service.remote_snapshot_owed
    backup = asyncio.create_task(service.snapshot_if_changed())
    await asyncio.wait_for(target.entered.wait(), 3)

    await service.record_remote_snapshot_owed()
    assert service.remote_snapshot_owed == first
    target.release.set()
    results = await asyncio.wait_for(backup, 3)

    assert results[target.name].success
    assert service.remote_snapshot_owed == first, (
        "a pass that predates the second record cleared it"
    )


@pytest.mark.asyncio
async def test_a_target_the_policy_denies_is_owed_nothing(tmp_path):
    from kestrel_sovereign.storage.sync.service import RemoteTierPolicyContext

    db = _database(tmp_path)
    policy = {"context": RemoteTierPolicyContext(identity="did:key:z6Mkowed")}
    service = SyncService(str(db), policy_context_provider=lambda: policy["context"])
    target = _Target("gs://owed/agent")
    service.add_target(target)
    await service.start()
    await service.record_remote_snapshot_owed()

    policy["context"] = RemoteTierPolicyContext(
        identity="did:key:z6Mkowed", is_test_instance=True
    )
    results = await service.snapshot_if_changed()

    assert target.calls == 0 and not results[target.name].attempted
    assert service.remote_snapshot_owed is None


@pytest.mark.asyncio
async def test_an_owed_snapshot_stays_owed_until_every_target_has_it(tmp_path):
    db = _database(tmp_path)
    working, failing = _Target("gs://owed/a"), _Target("s3://owed/b", fails=True)
    service = await _service(db, working, failing)
    await service.record_remote_snapshot_owed()

    await service.snapshot_if_changed()
    assert service.remote_snapshot_owed is not None

    failing.fails = False
    await service.snapshot_if_changed()
    assert service.remote_snapshot_owed is None


@pytest.mark.asyncio
async def test_a_pass_already_running_when_the_record_is_made_does_not_clear_it(
    tmp_path,
):
    """The pass copies the database as it was before the record."""
    db = _database(tmp_path)
    target = _Target("gs://owed/agent", blocks=True)
    service = await _service(db, target)
    running = asyncio.create_task(service.force_snapshot())
    await asyncio.wait_for(target.entered.wait(), 3)

    await service.record_remote_snapshot_owed()
    target.release.set()
    results = await asyncio.wait_for(running, 3)

    assert results[target.name].success
    assert service.remote_snapshot_owed is not None


@pytest.mark.asyncio
async def test_stop_waits_for_a_pass_until_it_is_cancelled_by_name(tmp_path, caplog):
    db = _database(tmp_path)
    target = _Target("gs://owed/blocked", blocks=True)
    service = await _service(db, target)
    backup = asyncio.create_task(service.snapshot_if_changed())
    await asyncio.wait_for(target.entered.wait(), 3)
    await service.record_remote_snapshot_owed()

    stopping = asyncio.create_task(service.stop())
    done, _ = await asyncio.wait({stopping}, timeout=0.2)
    assert not done, "stop() did not wait for the upload in flight"

    with caplog.at_level(logging.WARNING):
        cancelled = service.cancel_inflight_snapshots("shutdown budget")
    assert cancelled == ["snapshot to gs://owed/blocked cancelled at shutdown budget"]
    assert cancelled[0] in caplog.text

    await asyncio.wait_for(stopping, 3)
    results = await asyncio.wait_for(backup, 3)
    assert not results[target.name].success
    assert results[target.name].error == "cancelled at shutdown budget"
    assert asyncio.current_task().cancelling() == 0
    assert _persisted_owed(db) is not None, "a cancelled upload cleared the owed snapshot"
    assert service.cancel_inflight_snapshots("shutdown budget") == []


@pytest.mark.asyncio
async def test_the_budget_cancellation_skips_the_targets_after_it(tmp_path):
    db = _database(tmp_path)
    blocked, later = _Target("gs://owed/first", blocks=True), _Target("s3://owed/second")
    service = await _service(db, blocked, later)
    backup = asyncio.create_task(service.force_snapshot())
    await asyncio.wait_for(blocked.entered.wait(), 3)

    service.cancel_inflight_snapshots("shutdown budget")
    results = await asyncio.wait_for(backup, 3)

    assert later.calls == 0 and later.name not in results
    assert not backup.cancelled()


@pytest.mark.asyncio
async def test_the_requesters_cancellation_still_cancels_its_pass(tmp_path):
    db = _database(tmp_path)
    target = _Target("gs://owed/agent", blocks=True)
    service = await _service(db, target)
    backup = asyncio.create_task(service.force_snapshot())
    await asyncio.wait_for(target.entered.wait(), 3)

    backup.cancel()
    with pytest.raises(asyncio.CancelledError):
        await backup
    await asyncio.wait_for(service.stop(), 3)


@pytest.mark.asyncio
async def test_a_requester_cancelling_beside_the_budget_still_wins(tmp_path):
    db = _database(tmp_path)
    target = _Target("gs://owed/agent", blocks=True)
    service = await _service(db, target)
    backup = asyncio.create_task(service.force_snapshot())
    await asyncio.wait_for(target.entered.wait(), 3)

    service.cancel_inflight_snapshots("shutdown budget")
    backup.cancel()
    with pytest.raises(asyncio.CancelledError):
        await backup


@pytest.mark.asyncio
async def test_a_stopping_service_admits_no_new_pass(tmp_path):
    db = _database(tmp_path)
    target = _Target("gs://owed/agent")
    service = await _service(db, target)
    await service.stop()

    with pytest.raises(SyncServiceStoppingError):
        await service.force_snapshot()
    assert target.calls == 0


@pytest.mark.parametrize("deadline", [float("nan"), float("inf"), True, "60"])
def test_the_manager_refuses_a_deadline_that_is_not_a_finite_time(deadline):
    from kestrel_sovereign.multi_agent.agent_manager import AgentManager

    manager = AgentManager()
    with pytest.raises(ValueError, match="finite"):
        manager.arm_shutdown_remote_work_deadline(deadline)
    manager.arm_shutdown_remote_work_deadline(None)
    manager.arm_shutdown_remote_work_deadline(12.5)
    assert manager._shutdown_remote_work_deadline == 12.5


def test_a_failing_canceller_cancels_nothing_and_lets_the_drain_go_on(caplog):
    from kestrel_sovereign.multi_agent.agent_manager import (
        cancel_remote_work_at_shutdown_budget,
    )

    def failing(reason):
        raise RuntimeError("canceller broke")

    with caplog.at_level(logging.WARNING):
        assert cancel_remote_work_at_shutdown_budget(failing, "'agent'") == []
    assert "Could not cancel the retained remote work of 'agent'" in caplog.text
    assert cancel_remote_work_at_shutdown_budget(None, "'agent'") == []
    assert cancel_remote_work_at_shutdown_budget(
        lambda reason: [f"snapshot to x cancelled at {reason}"], "'agent'"
    ) == ["snapshot to x cancelled at shutdown budget"]


@pytest.mark.asyncio
async def test_an_unreadable_fingerprint_never_skips_the_scheduled_backup(
    tmp_path, monkeypatch
):
    """The change-aware skip needs a fingerprint it actually read."""
    db = _database(tmp_path)
    target = _Target("gs://owed/agent")
    service = await _service(db, target)
    await service.snapshot_if_changed()
    reads = []

    def fingerprint():
        reads.append(None)
        return None if len(reads) == 1 else service._state.last_fingerprint

    monkeypatch.setattr(service, "_compute_db_fingerprint", fingerprint)
    await service.snapshot_if_changed()

    assert target.calls == 2, "a failed fingerprint read was replaced by a second read"


@pytest.mark.asyncio
async def test_the_deadline_is_armed_for_one_fleet_shutdown():
    from kestrel_sovereign.multi_agent.agent_manager import AgentManager

    manager = AgentManager()
    manager.arm_shutdown_remote_work_deadline(1.0)
    await manager.shutdown_all()
    assert manager._shutdown_remote_work_deadline is None


async def _owed_then_restarted(
    tmp_path: Path, target: SyncTarget, *, owed_deadline: float
) -> tuple[Path, SyncService]:
    """A shutdown that records a snapshot as owed, then the next start."""
    db = _database(tmp_path)
    before = await _service(db, _Target(target.name))
    assert await before.record_remote_snapshot_owed() is True
    await before.stop()
    return db, await _service(db, target, owed_deadline=owed_deadline)


async def _until(predicate, timeout: float = 3.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        assert loop.time() < deadline, "timed out"
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_a_snapshot_owed_past_its_deadline_is_sent_at_start(tmp_path, caplog):
    """No scheduled backup is involved: the service sends it itself."""
    caplog.set_level(logging.INFO)
    target = _Target("gs://owed/agent")
    db, service = await _owed_then_restarted(tmp_path, target, owed_deadline=0)

    await _until(lambda: service.remote_snapshot_owed is None)

    assert target.calls == 1
    assert _persisted_owed(db) is None, "the owed marker was not cleared on disk"
    assert "no backup has sent it within 0s" in caplog.text
    await service.stop()


@pytest.mark.asyncio
async def test_the_deadline_passing_while_the_service_runs_sends_it(tmp_path):
    target = _Target("gs://owed/agent")
    _db, service = await _owed_then_restarted(tmp_path, target, owed_deadline=0.4)

    await asyncio.sleep(0.1)
    assert target.calls == 0, "the snapshot was sent before its deadline"
    await _until(lambda: service.remote_snapshot_owed is None)

    assert target.calls == 1
    await service.stop()


@pytest.mark.asyncio
async def test_a_scheduled_backup_that_sends_it_first_leaves_the_worker_nothing(tmp_path):
    target = _Target("gs://owed/agent")
    _db, service = await _owed_then_restarted(tmp_path, target, owed_deadline=0.3)

    db = service.db_path
    await service.snapshot_if_changed()
    assert target.calls == 1 and service.remote_snapshot_owed is None
    # A later write is the scheduled backup's to send, not the worker's.
    _write(db, "after the scheduled backup")
    await asyncio.sleep(0.5)

    assert target.calls == 1, "the worker sent a snapshot nobody owed any more"
    await service.stop()


@pytest.mark.asyncio
async def test_stop_cancels_the_worker_while_it_waits(tmp_path):
    target = _Target("gs://owed/agent")
    db, service = await _owed_then_restarted(tmp_path, target, owed_deadline=3600)

    await asyncio.wait_for(service.stop(), 3)

    assert target.calls == 0
    assert _persisted_owed(db) is not None


@pytest.mark.asyncio
async def test_stop_cancels_the_workers_upload_without_waiting_for_the_budget(tmp_path):
    """The worker is the service's own: its upload is not remote work the
    stop waits for, so a stopping agent is never held by it."""
    target = _Target("gs://owed/blocked", blocks=True)
    db, service = await _owed_then_restarted(tmp_path, target, owed_deadline=0)
    await asyncio.wait_for(target.entered.wait(), 3)

    await asyncio.wait_for(service.stop(), 3)

    assert target.calls == 1
    assert _persisted_owed(db) is not None, "a cancelled upload cleared the owed snapshot"
    assert service.cancel_inflight_snapshots("shutdown budget") == []


@pytest.mark.asyncio
async def test_a_failed_delivery_keeps_the_snapshot_owed(tmp_path, caplog):
    target = _Target("gs://owed/refused", fails=True)
    db, service = await _owed_then_restarted(tmp_path, target, owed_deadline=0)

    with caplog.at_level(logging.WARNING):
        await _until(lambda: "is still owed" in caplog.text)

    assert target.calls == 1
    assert service.remote_snapshot_owed is not None and _persisted_owed(db) is not None
    assert "gs://owed/refused" in caplog.text
    await service.stop()


@pytest.mark.parametrize("deadline", [-1.0, float("nan"), float("inf"), True, "60"])
def test_the_owed_snapshot_deadline_must_be_finite_seconds(tmp_path, deadline):
    with pytest.raises(ValueError, match="owed snapshot deadline"):
        SyncService(str(tmp_path / "agent.db"), owed_snapshot_deadline_seconds=deadline)


def test_the_deadline_comes_from_the_environment(monkeypatch):
    monkeypatch.delenv(OWED_SNAPSHOT_DEADLINE_ENV, raising=False)
    assert owed_snapshot_deadline_from_environment() == 5 * 3600
    monkeypatch.setenv(OWED_SNAPSHOT_DEADLINE_ENV, "0")
    assert owed_snapshot_deadline_from_environment() == 0
    for malformed in ("soon", "-5", "inf"):
        monkeypatch.setenv(OWED_SNAPSHOT_DEADLINE_ENV, malformed)
        with pytest.raises(ValueError, match="owed snapshot deadline|KESTREL_OWED"):
            owed_snapshot_deadline_from_environment()


class _CopyingTarget(_Target):
    """Copies the database as the shipped targets do (``sqlite3.backup``),
    then holds the upload open until released."""

    def __init__(self, name: str, remote: Path):
        super().__init__(name, blocks=True)
        self.remote = remote
        self.copies: list[Path] = []

    async def sync_snapshot(self, db_path: Path) -> SyncResult:
        copy = self.remote / f"{self.name.rsplit('/', 1)[-1]}-{len(self.copies)}.db"
        _write_consistent_snapshot(Path(db_path), copy)
        self.copies.append(copy)
        return await super().sync_snapshot(db_path)


def _rows(db: Path) -> int:
    with closing(sqlite3.connect(db)) as connection:
        return connection.execute("SELECT COUNT(*) FROM payload").fetchone()[0]


@pytest.mark.asyncio
async def test_a_write_during_the_upload_is_still_owed_at_shutdown(tmp_path):
    """The remote copy holds one row and the database two. The shutdown
    records the snapshot as owed, and the next backup after the restart
    uploads it rather than skipping it as unchanged."""
    db = _database(tmp_path)
    with closing(sqlite3.connect(db)) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
    remote = tmp_path / "remote"
    remote.mkdir()
    target = _CopyingTarget("gs://owed/agent", remote)
    service = await _service(db, target)
    backup = asyncio.create_task(service.snapshot_if_changed())
    await asyncio.wait_for(target.entered.wait(), 3)
    _write(db, "committed during the upload")
    target.release.set()
    results = await asyncio.wait_for(backup, 3)
    assert results[target.name].success
    assert (_rows(target.copies[-1]), _rows(db)) == (1, 2)

    assert await service.record_remote_snapshot_owed() is True, (
        "the backup recorded a write it never copied as backed up"
    )
    assert _persisted_owed(db) is not None
    await service.stop()

    after = _CopyingTarget("gs://owed/agent", remote)
    after.release.set()
    restarted = await _service(db, after)
    results = await restarted.snapshot_if_changed()

    assert "__unchanged__" not in results, "the next backup skipped the owed snapshot"
    assert results[after.name].success and _rows(after.copies[-1]) == 2
    assert restarted.remote_snapshot_owed is None and _persisted_owed(db) is None
    await restarted.stop()


@pytest.mark.asyncio
async def test_an_owed_record_that_was_not_written_raises_instead_of_reporting_owed(
    tmp_path, caplog
):
    db = _database(tmp_path)
    target = _Target("gs://owed/agent")
    service = await _service(db, target)
    state = Path(f"{db}.sync")
    state.mkdir()

    with caplog.at_level(logging.INFO):
        with pytest.raises(SyncStateWriteError, match=f"sync state {state} was not written"):
            await service.record_remote_snapshot_owed()
    assert "owed to gs://owed/agent since" not in caplog.text, (
        "a record that was not written was logged as owed"
    )

    # The record stays in memory, so the stop that follows writes it.
    assert service.remote_snapshot_owed is not None
    state.rmdir()
    await service.stop()
    assert _persisted_owed(db) is not None


@pytest.mark.asyncio
async def test_a_write_that_fails_keeps_the_previous_state_file(tmp_path, monkeypatch):
    """The state file is replaced whole: a failed write leaves the record an
    earlier shutdown made, and no staged file behind."""
    db = _database(tmp_path)
    service = await _service(db, _Target("gs://owed/agent"))
    assert await service.record_remote_snapshot_owed() is True
    state = Path(f"{db}.sync")
    before = state.read_bytes()

    def failing_fsync(descriptor):
        raise OSError(errno.EIO, "simulated fsync failure")

    with monkeypatch.context() as patch:
        patch.setattr(os, "fsync", failing_fsync)
        with pytest.raises(SyncStateWriteError, match="simulated fsync failure"):
            await service.record_remote_snapshot_owed()

    assert state.read_bytes() == before, "a failed write changed the state file"
    assert json.loads(before)["remote_snapshot_owed_generation"] == 1
    assert sorted(p.name for p in tmp_path.iterdir()) == ["agent.db", "agent.db.sync"]
    await service.stop()


@pytest.mark.asyncio
async def test_a_backup_whose_state_cannot_be_written_still_returns_its_uploads(
    tmp_path, caplog
):
    """Losing the write after a backup leaves an older fingerprint, which
    costs one more upload; the uploads themselves are reported."""
    db = _database(tmp_path)
    target = _Target("gs://owed/agent")
    service = await _service(db, target)
    state = Path(f"{db}.sync")
    state.mkdir()

    with caplog.at_level(logging.ERROR):
        results = await service.snapshot_if_changed()
        await service.stop()

    assert results[target.name].success and target.calls == 1
    assert f"sync state {state} was not written" in caplog.text


class _LandingTarget(_Target):
    """Copies the database as the shipped targets do and holds each upload
    until its own gate opens. ``landed`` is the copy of the upload that
    finished last: what a target that keeps one object then holds."""

    def __init__(self, name: str, remote: Path):
        super().__init__(name)
        self.remote = remote
        self.gates: list[asyncio.Event] = []
        self.landed: Path | None = None

    async def sync_snapshot(self, db_path: Path) -> SyncResult:
        copy = self.remote / f"copy-{len(self.gates)}.db"
        _write_consistent_snapshot(Path(db_path), copy)
        gate = asyncio.Event()
        self.gates.append(gate)
        await gate.wait()
        self.landed = copy
        return await super().sync_snapshot(db_path)


async def _landing_service(tmp_path: Path) -> tuple[Path, _LandingTarget, SyncService]:
    db = _database(tmp_path)
    with closing(sqlite3.connect(db)) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
    remote = tmp_path / "remote"
    remote.mkdir()
    target = _LandingTarget("gs://owed/agent", remote)
    return db, target, await _service(db, target)


@pytest.mark.asyncio
@pytest.mark.parametrize("older", ["force_snapshot", "snapshot_if_changed"])
async def test_an_older_pass_that_lands_last_leaves_the_snapshot_owed(tmp_path, older):
    """Two passes overlap and the older one's upload lands last, so the
    remote holds one row and the database two. The newer pass may not record
    the two-row database as backed up: the shutdown owes the snapshot and
    the next backup after the restart uploads it rather than skip it as
    unchanged."""
    db, target, service = await _landing_service(tmp_path)
    older_pass = asyncio.create_task(getattr(service, older)())
    await _until(lambda: len(target.gates) == 1)
    _write(db, "committed while the older pass uploads")
    newer_pass = asyncio.create_task(service.snapshot_if_changed())
    await _until(lambda: len(target.gates) == 2)
    target.gates[1].set()
    assert (await asyncio.wait_for(newer_pass, 3))[target.name].success
    target.gates[0].set()
    assert (await asyncio.wait_for(older_pass, 3))[target.name].success
    assert (_rows(target.landed), _rows(db)) == (1, 2)

    assert await service.record_remote_snapshot_owed() is True, (
        "a pass recorded the database as backed up while an older upload "
        "was still to land"
    )
    await service.stop()

    after = _CopyingTarget("gs://owed/agent", target.remote)
    after.release.set()
    restarted = await _service(db, after)
    results = await restarted.snapshot_if_changed()
    assert "__unchanged__" not in results, "the next backup skipped the owed snapshot"
    assert results[after.name].success and _rows(after.copies[-1]) == 2
    await restarted.stop()


@pytest.mark.asyncio
async def test_an_overlapped_pass_does_not_clear_an_owed_snapshot(tmp_path):
    """A pass that copied the database before the record lands after a
    newer pass reached every target: the remote is behind the record, so
    it stays owed."""
    db, target, service = await _landing_service(tmp_path)
    older_pass = asyncio.create_task(service.force_snapshot())
    await _until(lambda: len(target.gates) == 1)
    _write(db, "committed before the shutdown")
    assert await service.record_remote_snapshot_owed() is True
    newer_pass = asyncio.create_task(service.snapshot_if_changed())
    await _until(lambda: len(target.gates) == 2)
    target.gates[1].set()
    assert (await asyncio.wait_for(newer_pass, 3))[target.name].success
    target.gates[0].set()
    assert (await asyncio.wait_for(older_pass, 3))[target.name].success
    assert (_rows(target.landed), _rows(db)) == (1, 2)

    assert service.remote_snapshot_owed is not None, (
        "a pass cleared the owed snapshot while an older upload was still to land"
    )
    assert _persisted_owed(db) is not None
    await service.stop()


@pytest.mark.asyncio
async def test_a_pass_that_runs_alone_after_an_overlap_records_the_database(tmp_path):
    """An overlap costs one more upload, not the change-aware skip for good."""
    db, target, service = await _landing_service(tmp_path)
    first = asyncio.create_task(service.force_snapshot())
    await _until(lambda: len(target.gates) == 1)
    second = asyncio.create_task(service.snapshot_if_changed())
    await _until(lambda: len(target.gates) == 2)
    for gate in target.gates:
        gate.set()
    await asyncio.wait_for(asyncio.gather(first, second), 3)

    alone = asyncio.create_task(service.snapshot_if_changed())
    await _until(lambda: len(target.gates) == 3)
    target.gates[2].set()
    assert (await asyncio.wait_for(alone, 3))[target.name].success

    assert "__unchanged__" in await service.snapshot_if_changed()
    assert await service.record_remote_snapshot_owed() is False
    await service.stop()
