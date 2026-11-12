"""Host shutdown under its one budget, with a real booted agent (#4028).

Every ``kestrel restart`` ended in SIGKILL: each agent's shutdown started a
remote snapshot upload of a gigabyte-sized database, the per-agent bound
quarantined the agent, and the fleet's terminal drain then joined that
upload without a bound. Shutdown now records the snapshot as owed, and the
drain cancels any upload still running at the host shutdown budget, naming
it, so the lifespan returns before SIGKILL.

These tests run the real lifespan teardown (``server._lifespan_teardown_owner``)
under uvicorn's own lifespan driver, over a real ``AgentManager`` (or the
standalone server's single agent) and a real booted agent whose sync service
has a remote target the test controls.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from uvicorn.config import Config
from uvicorn.lifespan.on import LifespanOn

from kestrel_sovereign import server
from kestrel_sovereign.features.scheduler.feature import SchedulerFeature
from kestrel_sovereign.kestrel_agent import agent_shutdown_completion_owners
from kestrel_sovereign.multi_agent import agent_manager as agent_manager_module
from kestrel_sovereign.multi_agent import process_manager
from kestrel_sovereign.multi_agent.agent_manager import AgentManager
from kestrel_sovereign.storage.sync.service import SyncService
from kestrel_sovereign.storage.sync.targets import SyncResult, SyncTarget, TrustTier
from tests.shared.booted_agent import booted_agent

#: The host budget these tests configure: short enough that a hang shows as
#: a timeout well inside pytest's own, long enough to hold the per-agent
#: bound (agent_manager.SHUTDOWN_TIMEOUT) and the reserve.
_BUDGET_SECONDS = 20.0


class _RemoteTarget(SyncTarget):
    """A remote backup destination; a blocking one never finishes alone."""

    kind = "gcs"

    def __init__(self, name: str, *, blocks: bool):
        self._name = name
        self._blocks = blocks
        self.calls = 0
        self.uploads = 0
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
        if self._blocks:
            await self.release.wait()
        self.uploads += 1
        return SyncResult(
            success=True,
            target_name=self.name,
            bytes_synced=Path(db_path).stat().st_size,
            frames_synced=0,
            timestamp=datetime.now(timezone.utc),
        )

    async def sync_wal(self, wal_path: Path, position: int) -> SyncResult:
        raise AssertionError("the sync service never streams the WAL")

    async def get_latest_position(self):
        return None


@pytest.fixture
def host_budget(monkeypatch):
    """Configure the one host budget; the lifespan reads it at teardown."""
    monkeypatch.setattr(process_manager, "HOST_SHUTDOWN_BUDGET_SECONDS", _BUDGET_SECONDS)
    monkeypatch.setattr(
        process_manager,
        "HOST_SHUTDOWN_REMOTE_WORK_SECONDS",
        _BUDGET_SECONDS - process_manager.HOST_SHUTDOWN_RESERVE_SECONDS,
    )
    return _BUDGET_SECONDS


async def _install_sync(agent, target: SyncTarget) -> SyncService:
    """Give the booted agent a sync service with ``target``.

    A test instance's remote tiers are denied by policy, so the service the
    production boot builds can never carry an upload target here.
    """
    service = SyncService(db_path=agent.storage_path)
    service.add_target(target)
    await service.start()
    agent._sync_service = service
    agent._resource_custody().acquire("sync_service")
    return service


@asynccontextmanager
async def _serving_host(agent, name: str, *, topology: str = "fleet"):
    """A started host lifespan serving ``agent``: as a one-agent fleet, or as
    the standalone server's single agent."""
    app = FastAPI()
    manager = None
    if topology == "fleet":
        manager = AgentManager()
        manager._agents[name] = agent
        manager._agent_names[agent.did] = name
        app.state.agent_manager = manager
    else:
        app.state.agent = agent

    @asynccontextmanager
    async def lifespan(app):
        async with server._lifespan_teardown_owner(app):
            yield

    app.router.lifespan_context = lifespan
    driver = LifespanOn(Config(app=app, lifespan="on", log_config=None))
    await driver.startup()
    assert not driver.should_exit, "the test host did not start"
    yield manager, driver


def _owed_in_state_file(agent):
    state = json.loads(Path(f"{agent.storage_path}.sync").read_text())
    return state["remote_snapshot_owed_since"]


@pytest.mark.asyncio
@pytest.mark.parametrize("topology", ["fleet", "single"])
async def test_an_upload_blocked_at_its_target_does_not_hold_up_host_shutdown(
    tmp_path, host_budget, caplog, topology
):
    """A scheduled backup still uploading when the host stops is cancelled at
    the budget, by name, and the lifespan returns normally."""
    caplog.set_level(logging.INFO)
    target = _RemoteTarget("gs://budget-test/kestrel/blocked", blocks=True)
    async with booted_agent(
        tmp_path, name="BudgetBlocked", provenance="test:host_shutdown_budget",
        features=set(),
    ) as agent:
        service = await _install_sync(agent, target)
        backup = asyncio.create_task(service.snapshot_if_changed())
        lifespan = None
        try:
            async with _serving_host(
                agent, "BudgetBlocked", topology=topology
            ) as (manager, driver):
                await asyncio.wait_for(target.entered.wait(), 5)
                lifespan = asyncio.create_task(driver.shutdown())
                done, _ = await asyncio.wait({lifespan}, timeout=host_budget)
                assert lifespan in done, (
                    "host shutdown did not return within its budget; the drain "
                    "is still joining the blocked upload"
                )
                assert not driver.should_exit, "the lifespan shutdown failed"
            results = await asyncio.wait_for(backup, 5)
        finally:
            target.release.set()
            for task in (backup, lifespan):
                if task is not None:
                    await asyncio.wait({task}, timeout=30)

        if topology == "fleet":
            assert "All agents shut down" in caplog.text
            assert manager.get_agent("BudgetBlocked") is None
        assert "Application shutdown complete." in caplog.text
        assert (
            "snapshot to gs://budget-test/kestrel/blocked cancelled at shutdown budget"
            in caplog.text
        )
        assert results[target.name].error == "cancelled at shutdown budget"
        assert target.calls == 1 and target.uploads == 0, (
            "shutdown started an upload of its own"
        )
        assert _owed_in_state_file(agent) is not None, "no remote snapshot is owed"


@pytest.mark.asyncio
async def test_the_owed_snapshot_goes_out_with_the_next_scheduled_backup(
    tmp_path, host_budget, caplog
):
    caplog.set_level(logging.INFO)
    before = _RemoteTarget("gs://budget-test/kestrel/owed", blocks=False)
    async with booted_agent(
        tmp_path, name="BudgetOwed", provenance="test:host_shutdown_budget",
        features=set(),
    ) as agent:
        db_path = agent.storage_path
        await _install_sync(agent, before)
        scheduler = SchedulerFeature(SimpleNamespace(did=agent.did, features={}))
        scheduler.agent._sync_service = agent._sync_service
        assert json.loads(await scheduler._handle_backup_snapshot({}))["success"]
        assert before.uploads == 1
        # A write after the backup: the remote copy is now behind.
        await agent._raw_storage.db.execute(
            "CREATE TABLE IF NOT EXISTS budget_owed_probe (value TEXT)"
        )
        async with _serving_host(agent, "BudgetOwed") as (_manager, driver):
            await asyncio.wait_for(driver.shutdown(), host_budget)
        assert before.uploads == 1, "shutdown uploaded a snapshot"
        assert _owed_in_state_file(agent) is not None

    # The next process: a fresh sync service on the same database and state.
    after = _RemoteTarget("gs://budget-test/kestrel/owed", blocks=False)
    service = SyncService(db_path=db_path)
    service.add_target(after)
    await service.start()
    assert service.remote_snapshot_owed is not None
    scheduler = SchedulerFeature(SimpleNamespace(did="did:test:next", features={}))
    scheduler.agent._sync_service = service

    sent = json.loads(await scheduler._handle_backup_snapshot({}))

    assert sent["success"] and sent["targets"][after.name]["attempted"]
    assert after.uploads == 1
    assert service.remote_snapshot_owed is None
    assert json.loads(Path(f"{db_path}.sync").read_text())[
        "remote_snapshot_owed_since"
    ] is None, "the owed marker was not cleared on disk"
    assert "Owed remote snapshot" in caplog.text
    await service.stop()


@pytest.mark.asyncio
async def test_with_healthy_targets_shutdown_keeps_local_durability_within_the_agent_bound(
    tmp_path, host_budget, caplog
):
    caplog.set_level(logging.INFO)
    target = _RemoteTarget("gs://budget-test/kestrel/healthy", blocks=False)
    async with booted_agent(
        tmp_path, name="BudgetHealthy", provenance="test:host_shutdown_budget",
        features=set(),
    ) as agent:
        await _install_sync(agent, target)
        backend = agent._raw_storage.db.backend
        async with _serving_host(agent, "BudgetHealthy") as (manager, driver):
            loop = asyncio.get_running_loop()
            started = loop.time()
            await asyncio.wait_for(driver.shutdown(), host_budget)
            elapsed = loop.time() - started

        assert elapsed < agent_manager_module.SHUTDOWN_TIMEOUT, elapsed
        assert "exceeded its shutdown bound" not in caplog.text
        assert not manager._quarantined_shutdown_reapers
        assert not manager._quarantined_shutdown_history, "an agent was quarantined"
        assert "All agents shut down" in caplog.text
        assert "Application shutdown complete." in caplog.text
        # Local durability: the database closed and its WAL checkpointed.
        assert backend._connection is None
        wal = Path(f"{agent.storage_path}-wal")
        assert not wal.exists() or wal.stat().st_size == 0
        assert target.calls == 0, "shutdown uploaded a snapshot"
        assert _owed_in_state_file(agent) is not None


@pytest.mark.asyncio
async def test_cleanup_an_earlier_teardown_left_running_is_held_to_the_budget(
    tmp_path, host_budget, caplog
):
    """The standalone server first joins cleanup an earlier teardown left
    running. That join waits only until the budget, as the later one does."""
    caplog.set_level(logging.INFO)
    target = _RemoteTarget("gs://budget-test/kestrel/earlier", blocks=True)
    async with booted_agent(
        tmp_path, name="BudgetEarlier", provenance="test:host_shutdown_budget",
        features=set(),
    ) as agent:
        service = await _install_sync(agent, target)
        backup = asyncio.create_task(service.snapshot_if_changed())
        lifespan = None
        try:
            await asyncio.wait_for(target.entered.wait(), 5)
            # An earlier teardown whose own bound ran out: its durable tail
            # is retained, still waiting on the upload.
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(agent.shutdown(), 2)
            assert any(
                not owner.done() for owner in agent_shutdown_completion_owners(agent)
            ), "the earlier teardown left no cleanup running"

            async with _serving_host(
                agent, "BudgetEarlier", topology="single"
            ) as (_manager, driver):
                lifespan = asyncio.create_task(driver.shutdown())
                done, _ = await asyncio.wait({lifespan}, timeout=host_budget)
                assert lifespan in done, (
                    "host shutdown did not return within its budget; it is "
                    "joining the earlier teardown's cleanup without one"
                )
                assert not driver.should_exit, "the lifespan shutdown failed"
            results = await asyncio.wait_for(backup, 5)
        finally:
            target.release.set()
            for task in (backup, lifespan):
                if task is not None:
                    await asyncio.wait({task}, timeout=30)

        assert "Application shutdown complete." in caplog.text
        assert (
            "snapshot to gs://budget-test/kestrel/earlier cancelled at shutdown budget"
            in caplog.text
        )
        assert results[target.name].error == "cancelled at shutdown budget"
        assert target.uploads == 0
        assert _owed_in_state_file(agent) is not None


@pytest.mark.asyncio
async def test_without_a_scheduled_backup_the_owed_snapshot_goes_out_at_the_next_start(
    tmp_path, host_budget, caplog
):
    """An agent whose features schedule no backup still sends the snapshot
    its shutdown recorded as owed: its sync service does, once it is past
    its deadline."""
    caplog.set_level(logging.INFO)
    async with booted_agent(
        tmp_path, name="BudgetNoScheduler", provenance="test:host_shutdown_budget",
        features=set(),
    ) as agent:
        assert "SchedulerFeature" not in agent.features
        db_path = agent.storage_path
        await _install_sync(
            agent, _RemoteTarget("gs://budget-test/kestrel/unscheduled", blocks=False)
        )
        async with _serving_host(agent, "BudgetNoScheduler") as (_manager, driver):
            await asyncio.wait_for(driver.shutdown(), host_budget)
        assert _owed_in_state_file(agent) is not None

    # The next process: the sync service its boot starts, past the deadline.
    after = _RemoteTarget("gs://budget-test/kestrel/unscheduled", blocks=False)
    service = SyncService(db_path=db_path, owed_snapshot_deadline_seconds=0)
    service.add_target(after)
    await service.start()
    try:
        await asyncio.wait_for(after.entered.wait(), 5)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 5
        while service.remote_snapshot_owed is not None:
            assert loop.time() < deadline, "the owed snapshot was never cleared"
            await asyncio.sleep(0.01)
    finally:
        await service.stop()

    assert after.uploads == 1
    assert json.loads(Path(f"{db_path}.sync").read_text())[
        "remote_snapshot_owed_since"
    ] is None, "the owed marker was not cleared on disk"
    assert "Owed remote snapshot" in caplog.text
