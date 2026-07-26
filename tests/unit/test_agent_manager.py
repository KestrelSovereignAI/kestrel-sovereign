"""Unit tests for the in-process AgentManager."""

import asyncio
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from kestrel_sovereign.features import MandatoryFeatureReadinessError
from kestrel_sovereign.features.scheduler.runner import (
    AgentManagerHostedSchedulerExecutor,
    SchedulerExecution,
)
from kestrel_sovereign.identity.runtime_identity import IdentityReadinessError
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.multi_agent.agent_manager import (
    AgentManager,
    _AgentDIDLookupMode,
    _get_agent_did,
)
from kestrel_sovereign.multi_agent.config import LocalAgentConfig, MultiAgentConfig
from kestrel_sovereign.spawn.mandate import SpawnMandate
from kestrel_sovereign.storage import AsyncStorage, GraphNode
from tests.utils.aiosqlite_workers import aiosqlite_worker


def _make_mock_agent(agent_id: str = "did:pkh:eip155:1:0xABC"):
    """Create a mock KestrelAgent with the minimum interface."""
    agent = MagicMock()
    agent.agent_id = agent_id
    agent.initialize = AsyncMock()
    agent.shutdown = AsyncMock()
    agent.get_agent_card = AsyncMock()
    return agent


class TestAgentManagerBasics:
    """Test AgentManager get/list/remove without real agents."""

    def test_empty_manager(self):
        manager = AgentManager()
        assert manager.list_agents() == {}
        assert manager.get_agent("nonexistent") is None

    def test_get_agent_case_insensitive(self):
        manager = AgentManager()
        mock = _make_mock_agent()
        manager._agents["Claw"] = mock
        manager._agent_names[mock.agent_id] = "Claw"

        assert manager.get_agent("Claw") is mock
        assert manager.get_agent("claw") is mock
        assert manager.get_agent("CLAW") is mock
        assert manager.get_agent("unknown") is None

    def test_list_agents(self):
        manager = AgentManager()
        agent1 = _make_mock_agent("did:1")
        agent2 = _make_mock_agent("did:2")
        manager._agents["Alpha"] = agent1
        manager._agents["Beta"] = agent2

        result = manager.list_agents()
        assert len(result) == 2
        assert result["Alpha"] is agent1
        assert result["Beta"] is agent2

    def test_get_agent_name(self):
        manager = AgentManager()
        mock = _make_mock_agent("did:pkh:test")
        manager._agents["Emma"] = mock
        manager._agent_names["did:pkh:test"] = "Emma"

        assert manager.get_agent_name("did:pkh:test") == "Emma"
        assert manager.get_agent_name("did:unknown") is None

    @pytest.mark.asyncio
    async def test_local_agent_configs_by_did_includes_cold_agents(self, monkeypatch, tmp_path):
        """A host scheduler can resolve both loaded and autostart=false agents."""
        manager = AgentManager(base_data_dir=tmp_path)
        warm = _make_mock_agent("did:pkh:warm")
        manager._agents["Warm"] = warm
        cold_dir = tmp_path / "cold"
        config = MultiAgentConfig(
            agents={
                "Warm": LocalAgentConfig(data_dir="warm", port=8801, autostart=True),
                "Cold": LocalAgentConfig(data_dir="cold", port=8802, autostart=False),
            }
        )

        did_lookup = AsyncMock(return_value="did:pkh:cold")
        monkeypatch.setattr(
            "kestrel_sovereign.multi_agent.agent_manager._get_agent_did",
            did_lookup,
        )

        mapping = await manager.local_agent_configs_by_did(config)

        assert mapping["did:pkh:warm"][0] == "Warm"
        assert mapping["did:pkh:cold"][0] == "Cold"
        did_lookup.assert_awaited_once_with(
            str(cold_dir),
            mode=_AgentDIDLookupMode.COLD_READ_ONLY,
        )

    @pytest.mark.asyncio
    async def test_local_agent_configs_skips_unincepted_cold_agent_but_keeps_healthy_peer(
        self, monkeypatch, tmp_path,
    ):
        """One missing cold identity cannot abort the healthy scheduler fleet."""
        manager = AgentManager(base_data_dir=tmp_path)
        warm = _make_mock_agent("did:pkh:warm")
        manager._agents["Warm"] = warm
        config = MultiAgentConfig(
            agents={
                "Warm": LocalAgentConfig(
                    data_dir="warm", port=8801, autostart=True
                ),
                "Unincepted": LocalAgentConfig(
                    data_dir="unincepted", port=8802, autostart=False
                ),
            }
        )
        missing_identity = RuntimeError("identity database is not initialized")
        did_lookup = AsyncMock(side_effect=missing_identity)
        monkeypatch.setattr(
            "kestrel_sovereign.multi_agent.agent_manager._get_agent_did",
            did_lookup,
        )

        mapping = await manager.local_agent_configs_by_did(config)

        assert mapping == {"did:pkh:warm": ("Warm", config.agents["Warm"])}
        assert manager.cold_scheduler_identity_failures == [
            ("Unincepted", missing_identity)
        ]
        did_lookup.assert_awaited_once_with(
            str(tmp_path / "unincepted"),
            mode=_AgentDIDLookupMode.COLD_READ_ONLY,
        )

    @pytest.mark.asyncio
    async def test_local_agent_configs_skips_unresolved_autostart_agent_but_keeps_healthy_peer(
        self, monkeypatch, tmp_path,
    ):
        """A failed autostart tenant is not scheduler authority for its peer."""
        manager = AgentManager(base_data_dir=tmp_path)
        warm = _make_mock_agent("did:pkh:warm")
        manager._agents["Warm"] = warm
        config = MultiAgentConfig(
            agents={
                "Warm": LocalAgentConfig(data_dir="warm", port=8801),
                "Unresolved": LocalAgentConfig(
                    data_dir="unresolved", port=8802, autostart=True
                ),
            }
        )
        missing_identity = ValueError("local identity unavailable")
        monkeypatch.setattr(
            "kestrel_sovereign.multi_agent.agent_manager._get_agent_did",
            AsyncMock(side_effect=missing_identity),
        )

        mapping = await manager.local_agent_configs_by_did(config)

        assert mapping == {"did:pkh:warm": ("Warm", config.agents["Warm"])}
        assert manager.cold_scheduler_identity_failures == [
            ("Unresolved", missing_identity)
        ]
        assert manager.is_scheduler_agent_authorized("did:pkh:warm")
        assert not manager.is_scheduler_agent_authorized("did:pkh:unresolved")

    @pytest.mark.asyncio
    async def test_scheduler_preflight_recovers_wal_for_autostart_identity(
        self, monkeypatch, tmp_path,
    ):
        """Autostart authority uses normal WAL recovery before scheduler boot."""

        manager = AgentManager(base_data_dir=tmp_path)
        config = MultiAgentConfig(
            agents={
                "Recovering": LocalAgentConfig(
                    data_dir="recovering", port=8801, autostart=True
                ),
            }
        )
        did_lookup = AsyncMock(return_value="did:pkh:recovered")
        monkeypatch.setattr(
            "kestrel_sovereign.multi_agent.agent_manager._get_agent_did",
            did_lookup,
        )

        mapping = await manager.local_agent_configs_by_did(config)

        assert mapping == {
            "did:pkh:recovered": ("Recovering", config.agents["Recovering"])
        }
        assert manager.scheduler_authority_for("did:pkh:recovered") == (
            "Recovering",
            config.agents["Recovering"],
        )
        did_lookup.assert_awaited_once_with(
            str(tmp_path / "recovering"),
            mode=_AgentDIDLookupMode.INITIALIZATION,
        )

    @pytest.mark.asyncio
    async def test_autostart_preflight_authority_needs_no_runtime_hook(self):
        """A recovered startup DID is already protocol-seeded before load."""

        manager = AgentManager()
        manager.set_scheduler_polling_managed_by_host(True)
        config = LocalAgentConfig(data_dir="recovering", port=8801, autostart=True)
        agent_id = "did:pkh:recovered"
        manager._seed_scheduler_authority({agent_id: ("Recovering", config)})

        assert (
            await manager._begin_dynamic_scheduler_tenant_registration(
                "Recovering", agent_id, config
            )
            is None
        )
        assert not manager.scheduler_lifecycle_lock(agent_id).locked()

    @pytest.mark.asyncio
    async def test_dynamic_scheduler_registration_cancellation_joins_and_rolls_back(
        self,
    ):
        """Cancellation cannot expose scope or orphan protocol/authority state."""

        manager = AgentManager()
        manager.set_scheduler_polling_managed_by_host(True)
        agent_id = "did:scheduler:dynamic-cancel"
        config = LocalAgentConfig(data_dir="dynamic", port=8801)
        hook_started = asyncio.Event()
        release_hook = asyncio.Event()
        protocol_rolled_back = asyncio.Event()

        async def register(_name, _agent_id, _config):
            hook_started.set()
            await release_hook.wait()

            async def rollback():
                protocol_rolled_back.set()

            return rollback

        manager.set_scheduler_tenant_registration_hook(register)
        registration = asyncio.create_task(
            manager._begin_dynamic_scheduler_tenant_registration(
                "Dynamic",
                agent_id,
                config,
            )
        )
        await asyncio.wait_for(hook_started.wait(), timeout=1)
        assert manager.scheduler_authority_for(agent_id) == ("Dynamic", config)
        assert agent_id not in manager.scheduler_authorized_agent_ids()

        registration.cancel()
        await asyncio.sleep(0)
        assert not registration.done()
        release_hook.set()
        with pytest.raises(asyncio.CancelledError):
            await registration

        assert protocol_rolled_back.is_set()
        assert manager.scheduler_authority_for(agent_id) is None
        assert agent_id not in manager.scheduler_authorized_agent_ids()
        assert not manager.scheduler_lifecycle_lock(agent_id).locked()

    @pytest.mark.asyncio
    async def test_dynamic_scheduler_registration_rolls_back_on_onboarding_failure(
        self,
    ):
        """The scheduler lease spans publication and app-owned onboarding."""

        manager = AgentManager()
        manager.set_scheduler_polling_managed_by_host(True)
        agent_id = "did:scheduler:onboarding-failure"
        config = LocalAgentConfig(data_dir="dynamic", port=8801)
        protocol_rolled_back = asyncio.Event()

        async def register(_name, _agent_id, _config):
            async def rollback():
                protocol_rolled_back.set()

            return rollback

        manager.set_scheduler_tenant_registration_hook(register)
        pending = await manager._begin_dynamic_scheduler_tenant_registration(
            "Dynamic",
            agent_id,
            config,
        )
        agent = _make_mock_agent(agent_id)
        agent._dynamic_scheduler_tenant_registration = pending
        manager._initialize_agent = AsyncMock(return_value=agent)
        manager.set_agent_registration_hook(
            AsyncMock(side_effect=RuntimeError("onboarding failed"))
        )

        with pytest.raises(RuntimeError, match="onboarding failed"):
            await manager.load_agent("Dynamic", config)

        assert manager.list_agents() == {}
        assert protocol_rolled_back.is_set()
        assert manager.scheduler_authority_for(agent_id) is None
        assert agent_id not in manager.scheduler_authorized_agent_ids()
        assert not manager.scheduler_lifecycle_lock(agent_id).locked()
        agent.shutdown.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_pending_dynamic_registration_cannot_claim_before_onboarding_commit(
        self, tmp_path
    ):
        """A host runner cannot adopt rollback-owned rows before onboarding."""

        from kestrel_sovereign.features.scheduler.runner import (
            SCHEDULER_PROTOCOL_VERSION,
            SchedulerRunner,
        )
        from kestrel_sovereign.storage.async_database import AsyncDatabase
        from kestrel_sovereign.storage.db import SQLiteBackend

        backend = SQLiteBackend(str(tmp_path / "pending-registration.db"))
        await backend.connect()
        db = AsyncDatabase(backend)
        manager = AgentManager()
        manager.set_scheduler_polling_managed_by_host(True)
        agent_id = "did:scheduler:pending-onboarding"
        config = LocalAgentConfig(data_dir="dynamic", port=8801)
        registration_runner = SchedulerRunner(
            db,
            None,
            AsyncMock(),
            authorized_agent_ids=(agent_id,),
            owner_id="registration-owner",
        )
        pending = None
        try:
            async def register(_name, _agent_id, _config):
                durable_registration = (
                    await registration_runner.prepare_tenant_registration()
                )

                async def rollback() -> None:
                    await registration_runner.rollback_tenant_registration(
                        durable_registration
                    )

                rollback.scheduler_registration_nonce = (
                    durable_registration.registration_nonce
                )
                return rollback

            manager.set_scheduler_tenant_registration_hook(register)
            pending = await manager._begin_dynamic_scheduler_tenant_registration(
                "Dynamic", agent_id, config
            )
            assert pending is not None
            assert manager.scheduler_authority_for(agent_id) == ("Dynamic", config)
            assert not manager.is_scheduler_agent_authorized(agent_id)

            now = datetime.now(timezone.utc).isoformat()
            due = "2000-01-01T00:00:00+00:00"
            await db.execute(
                """
                INSERT INTO scheduled_tasks
                    (id, agent_id, task_name, cron_expression, args_json, enabled,
                     next_run_at, created_at, scheduler_protocol_version,
                     scheduler_registration_nonce)
                VALUES (?, ?, 'backup_snapshot', '* * * * *', '{}', 1, ?, ?, ?, ?)
                """,
                (
                    "pending-owned-schedule",
                    agent_id,
                    due,
                    now,
                    SCHEDULER_PROTOCOL_VERSION,
                    pending.registration_nonce,
                ),
            )
            host_runner = SchedulerRunner(
                db,
                None,
                AsyncMock(return_value="executed"),
                authorized_agent_ids=(agent_id,),
                authorized_agent_ids_provider=manager.scheduler_authorized_agent_ids,
                is_agent_authorized=manager.is_scheduler_agent_authorized,
                owner_id="host-runner",
            )

            # This is the former race: a scope publication here let the host
            # claim the row, clear its nonce, and make rollback retain it.
            await host_runner._tick()
            assert await db.fetchone(
                "SELECT scheduler_registration_nonce FROM scheduled_tasks WHERE id = ?",
                ("pending-owned-schedule",),
            ) == (pending.registration_nonce,)
            assert await db.fetchone(
                "SELECT COUNT(*) FROM task_execution_log WHERE task_id = ?",
                ("pending-owned-schedule",),
            ) == (0,)

            await pending.rollback()
            pending = None
            assert await db.fetchone(
                "SELECT COUNT(*) FROM scheduled_tasks WHERE id = ?",
                ("pending-owned-schedule",),
            ) == (0,)
            assert await db.fetchone(
                "SELECT COUNT(*) FROM task_execution_log WHERE task_id = ?",
                ("pending-owned-schedule",),
            ) == (0,)
        finally:
            if pending is not None:
                await pending.rollback()
            await db.close()

    @pytest.mark.asyncio
    async def test_hosted_cold_registration_validates_authority_without_releasing_owned_lock(
        self,
    ):
        """The explicit lock-owner path never reacquires or releases the DID lease."""

        manager = AgentManager()
        manager.set_scheduler_polling_managed_by_host(True)
        agent_id = "did:scheduler:configured-cold"
        config = LocalAgentConfig(data_dir="cold", port=8801, autostart=False)
        registration_hook = AsyncMock()
        manager.set_scheduler_tenant_registration_hook(registration_hook)
        lifecycle_lock = manager.scheduler_lifecycle_lock(agent_id)
        await lifecycle_lock.acquire()
        try:
            with pytest.raises(LookupError, match="without live manager authority"):
                await manager._begin_dynamic_scheduler_tenant_registration(
                    "Cold",
                    agent_id,
                    config,
                    scheduler_lifecycle_lock_held=True,
                )
            assert lifecycle_lock.locked()

            manager._seed_scheduler_authority(
                {agent_id: ("Cold", config)}
            )
            assert (
                await manager._begin_dynamic_scheduler_tenant_registration(
                    "Cold",
                    agent_id,
                    config,
                    scheduler_lifecycle_lock_held=True,
                )
                is None
            )
            assert lifecycle_lock.locked()
            registration_hook.assert_not_awaited()

            manager._scheduler_authority_by_name["Cold"] = "did:other"
            with pytest.raises(RuntimeError, match="does not match"):
                await manager._begin_dynamic_scheduler_tenant_registration(
                    "Cold",
                    agent_id,
                    config,
                    scheduler_lifecycle_lock_held=True,
                )
            assert lifecycle_lock.locked()
        finally:
            lifecycle_lock.release()

    @pytest.mark.asyncio
    async def test_cold_scheduler_load_refuses_did_mismatch_before_registration(
        self, tmp_path,
    ):
        """A cold wake must not publish tenant B under tenant A's claim."""
        manager = AgentManager(base_data_dir=tmp_path)
        tenant_b_agent = _make_mock_agent("did:pkh:tenant-b")
        manager._initialize_agent = AsyncMock(return_value=tenant_b_agent)
        config = LocalAgentConfig(
            data_dir="cold", port=8801, autostart=False
        )
        manager._seed_scheduler_authority(
            {"did:pkh:tenant-a": ("Cold", config)}
        )

        with pytest.raises(RuntimeError, match="does not match claimed DID"):
            await manager.load_agent(
                "Cold",
                config,
                expected_agent_id="did:pkh:tenant-a",
            )

        assert manager.list_agents() == {}
        assert manager.get_agent_name("did:pkh:tenant-b") is None
        tenant_b_agent.shutdown.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_cold_did_lookup_missing_database_never_creates_identity_artifacts(
        self, tmp_path,
    ):
        """Discovery of an unincepted cold config is a strictly read-only probe."""
        cold_dir = tmp_path / "unincepted"
        cold_dir.mkdir()

        with pytest.raises(ValueError, match="No agent found"):
            await _get_agent_did(str(cold_dir))

        assert list(cold_dir.iterdir()) == []
        assert not (cold_dir / "kestrel_prime.db").exists()
        assert not (cold_dir / "kestrel_prime.db-wal").exists()
        assert not (cold_dir / "kestrel_prime.db-shm").exists()

    @pytest.mark.asyncio
    async def test_cold_did_lookup_invalid_existing_database_never_writes_sidecars(
        self, tmp_path,
    ):
        """A malformed identity is reported without schema/WAL side effects."""
        cold_dir = tmp_path / "invalid"
        cold_dir.mkdir()
        db_path = cold_dir / "kestrel_prime.db"
        connection = sqlite3.connect(db_path)
        connection.execute("CREATE TABLE unrelated (value TEXT)")
        connection.commit()
        connection.close()
        before = {
            path.name: path.read_bytes()
            for path in cold_dir.iterdir()
        }

        with pytest.raises(ValueError, match="Could not read local agent identity"):
            await _get_agent_did(str(cold_dir))

        after = {
            path.name: path.read_bytes()
            for path in cold_dir.iterdir()
        }
        assert after == before

    @pytest.mark.asyncio
    async def test_cold_did_lookup_refuses_uncheckpointed_wal_without_mutation(
        self, tmp_path,
    ):
        """A cold identity probe must not ignore or materialize WAL state."""
        cold_dir = tmp_path / "wal-pending"
        cold_dir.mkdir()
        db_path = cold_dir / "kestrel_prime.db"
        connection = sqlite3.connect(db_path)
        connection.execute(
            "CREATE TABLE graph_nodes (node_id TEXT, node_type TEXT)"
        )
        connection.execute(
            "INSERT INTO graph_nodes VALUES (?, 'agent')",
            ("did:test:wal-pending",),
        )
        connection.commit()
        connection.close()
        wal_path = cold_dir / "kestrel_prime.db-wal"
        wal_path.write_bytes(b"uncheckpointed identity state")
        before = {
            path.name: path.read_bytes()
            for path in cold_dir.iterdir()
        }

        with pytest.raises(ValueError, match="WAL state is present"):
            await _get_agent_did(str(cold_dir))

        after = {
            path.name: path.read_bytes()
            for path in cold_dir.iterdir()
        }
        assert after == before

    @pytest.mark.asyncio
    async def test_initialization_did_lookup_recovers_existing_sqlite_wal(
        self, tmp_path,
    ):
        """Normal startup can consume a crash-recovery WAL that cold probes refuse."""
        agent_dir = tmp_path / "wal-recovery"
        agent_dir.mkdir()
        db_path = agent_dir / "kestrel_prime.db"
        connection = sqlite3.connect(db_path)
        try:
            assert connection.execute("PRAGMA journal_mode=WAL").fetchone() == ("wal",)
            connection.execute(
                "CREATE TABLE graph_nodes (node_id TEXT, node_type TEXT)"
            )
            connection.execute(
                "INSERT INTO graph_nodes VALUES (?, 'agent')",
                ("did:test:wal-recovery",),
            )
            connection.commit()
            assert (agent_dir / "kestrel_prime.db-wal").exists()

            with pytest.raises(ValueError, match="WAL state is present"):
                await _get_agent_did(str(agent_dir))

            assert await _get_agent_did(
                str(agent_dir),
                mode=_AgentDIDLookupMode.INITIALIZATION,
            ) == "did:test:wal-recovery"
        finally:
            connection.close()

    @pytest.mark.asyncio
    async def test_cold_did_lookup_stays_local_sqlite_with_postgres_environment(
        self, monkeypatch, tmp_path,
    ):
        """A host DB default must not select a foreign shared identity."""
        local_dir = tmp_path / "cold-agent"
        local_dir.mkdir()
        local_db = local_dir / "kestrel_prime.db"
        local_did = "did:test:local-cold-agent"

        storage = AsyncStorage(str(local_db), backend="sqlite")
        await storage.initialize()
        try:
            await storage.graph.add_node(
                GraphNode(local_did, "agent", "Local cold agent", {})
            )
        finally:
            await storage.close()

        before = {
            path.name: path.read_bytes()
            for path in local_dir.iterdir()
        }
        monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
        monkeypatch.setenv("KESTREL_DATABASE_URL", "postgresql://foreign-host/fleet")

        assert await _get_agent_did(str(local_dir)) == local_did
        after = {
            path.name: path.read_bytes()
            for path in local_dir.iterdir()
        }
        # The successful normal path is as important as malformed/missing
        # probes: it must not materialize or alter SQLite WAL/SHM sidecars.
        assert after == before
        assert "kestrel_prime.db-wal" not in after
        assert "kestrel_prime.db-shm" not in after

    @pytest.mark.asyncio
    async def test_remove_agent(self):
        manager = AgentManager()
        mock = _make_mock_agent("did:pkh:remove")
        manager._agents["Testbot"] = mock
        manager._agent_names["did:pkh:remove"] = "Testbot"

        result = await manager.remove_agent("Testbot")
        assert result is True
        assert manager.get_agent("Testbot") is None
        assert manager.get_agent_name("did:pkh:remove") is None
        mock.shutdown.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_remove_agent_revokes_live_scheduler_authority_before_future_cold_wake(
        self,
    ):
        """DELETE cannot be undone by the static startup config's old DID map."""
        manager = AgentManager()
        config = LocalAgentConfig(data_dir="managed", port=8801)
        mock = _make_mock_agent("did:pkh:removed")
        manager._agents["Managed"] = mock
        manager._agent_names[mock.agent_id] = "Managed"
        manager._seed_scheduler_authority({mock.agent_id: ("Managed", config)})

        assert await manager.remove_agent("Managed") is True
        assert not manager.is_scheduler_agent_authorized(mock.agent_id)
        manager._initialize_agent = AsyncMock()
        with pytest.raises(LookupError, match="no longer authorized"):
            await manager.load_agent(
                "Managed",
                config,
                expected_agent_id=mock.agent_id,
            )
        manager._initialize_agent.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_delete_unloaded_scheduler_agent_revokes_before_executor_wake(self):
        """DELETE of a configured cold tenant is a completed runtime removal."""
        manager = AgentManager()
        agent_id = "did:pkh:unloaded-delete"
        config = LocalAgentConfig(data_dir="cold", port=8801, autostart=False)
        manager._seed_scheduler_authority({agent_id: ("Cold", config)})
        manager._initialize_agent = AsyncMock()
        executor = AgentManagerHostedSchedulerExecutor(
            manager,
            {agent_id: ("Cold", config)},
        )
        execution = SchedulerExecution(
            id="execution-unloaded-delete",
            schedule_id="schedule-unloaded-delete",
            agent_id=agent_id,
            task_name="test_task",
            args={},
            scheduled_for="2026-07-25T00:00:00+00:00",
            idempotency_key="effect-unloaded-delete",
            attempt=1,
            owner="host",
        )

        assert await manager.remove_agent("Cold") is True
        assert not manager.is_scheduler_agent_authorized(agent_id)
        with pytest.raises(LookupError, match="No hosted agent configuration"):
            await executor.execute_scheduled(execution)
        manager._initialize_agent.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_failed_remove_restores_scheduler_authority(self):
        """A failed DELETE must not silently change desired state."""
        manager = AgentManager()
        config = LocalAgentConfig(data_dir="managed", port=8801)
        mock = _make_mock_agent("did:pkh:still-live")
        mock.shutdown.side_effect = RuntimeError("shutdown failed")
        manager._agents["Managed"] = mock
        manager._agent_names[mock.agent_id] = "Managed"
        manager._seed_scheduler_authority({mock.agent_id: ("Managed", config)})

        assert await manager.remove_agent("Managed") is False
        assert manager.is_scheduler_agent_authorized(mock.agent_id)

    @pytest.mark.asyncio
    async def test_delete_serializes_with_inflight_scheduler_lifecycle_lock(self):
        """DELETE waits for an in-flight dispatch lease, then revokes cold wake."""
        manager = AgentManager()
        config = LocalAgentConfig(data_dir="managed", port=8801)
        mock = _make_mock_agent("did:pkh:locked-delete")
        manager._agents["Managed"] = mock
        manager._agent_names[mock.agent_id] = "Managed"
        manager._seed_scheduler_authority({mock.agent_id: ("Managed", config)})
        dispatch_started = asyncio.Event()
        allow_dispatch_finish = asyncio.Event()

        async def in_flight_dispatch():
            async with manager.scheduler_lifecycle_lock(mock.agent_id):
                dispatch_started.set()
                await allow_dispatch_finish.wait()

        dispatch = asyncio.create_task(in_flight_dispatch())
        await asyncio.wait_for(dispatch_started.wait(), timeout=1)
        deletion = asyncio.create_task(manager.remove_agent("Managed"))
        await asyncio.sleep(0)
        assert not deletion.done()
        assert manager.is_scheduler_agent_authorized(mock.agent_id)

        allow_dispatch_finish.set()
        await dispatch
        assert await deletion is True
        assert not manager.is_scheduler_agent_authorized(mock.agent_id)

    @pytest.mark.asyncio
    async def test_hosted_effects_share_lifecycle_read_lease_before_delete(self):
        """Sibling schedules overlap, while DELETE drains both before revoking."""

        manager = AgentManager()
        config = LocalAgentConfig(data_dir="managed", port=8801)
        mock = _make_mock_agent("did:pkh:shared-scheduler-effects")
        manager._agents["Managed"] = mock
        manager._agent_names[mock.agent_id] = "Managed"
        manager._seed_scheduler_authority({mock.agent_id: ("Managed", config)})
        effects_started: list[str] = []
        both_started = asyncio.Event()
        release_effects = asyncio.Event()

        async def dispatch(_task_name, args):
            effects_started.append(args["effect"])
            if len(effects_started) == 2:
                both_started.set()
            await release_effects.wait()
            return "dispatched"

        mock.features = {
            "SchedulerFeature": SimpleNamespace(
                _dispatch_scheduled_task=dispatch,
            )
        }
        executor = AgentManagerHostedSchedulerExecutor(manager)

        def execution(effect: str) -> SchedulerExecution:
            return SchedulerExecution(
                id=f"execution-{effect}",
                schedule_id=f"schedule-{effect}",
                agent_id=mock.agent_id,
                task_name="test_task",
                args={"effect": effect},
                scheduled_for="2026-07-25T00:00:00+00:00",
                idempotency_key=f"effect-{effect}",
                attempt=1,
                owner="host",
            )

        scheduled = [
            asyncio.create_task(executor.execute_scheduled(execution(effect)))
            for effect in ("a", "b")
        ]
        deletion = None
        try:
            await asyncio.wait_for(both_started.wait(), timeout=1)
            assert set(effects_started) == {"a", "b"}

            deletion = asyncio.create_task(manager.remove_agent("Managed"))
            await asyncio.sleep(0.02)
            assert not deletion.done()
            assert manager.is_scheduler_agent_authorized(mock.agent_id)

            release_effects.set()
            assert await asyncio.gather(*scheduled) == ["dispatched", "dispatched"]
            assert await asyncio.wait_for(deletion, timeout=1) is True
            assert not manager.is_scheduler_agent_authorized(mock.agent_id)
        finally:
            release_effects.set()
            for task in (*scheduled, deletion):
                if task is not None and not task.done():
                    task.cancel()
            await asyncio.gather(
                *scheduled,
                *(() if deletion is None else (deletion,)),
                return_exceptions=True,
            )

    @pytest.mark.asyncio
    async def test_hosted_executor_uses_live_same_did_replacement_after_handoff(
        self,
    ):
        """A writer replacement cannot leave a stale warm agent dispatchable."""

        manager = AgentManager()
        config = LocalAgentConfig(data_dir="managed", port=8801)
        agent_id = "did:pkh:scheduler-replacement"
        original = _make_mock_agent(agent_id)
        replacement = _make_mock_agent(agent_id)
        original_dispatch = AsyncMock(return_value="original")
        replacement_dispatch = AsyncMock(return_value="replacement")
        original.features = {
            "SchedulerFeature": SimpleNamespace(
                _dispatch_scheduled_task=original_dispatch,
            )
        }
        replacement.features = {
            "SchedulerFeature": SimpleNamespace(
                _dispatch_scheduled_task=replacement_dispatch,
            )
        }
        manager._agents["Managed"] = original
        manager._agent_names[agent_id] = "Managed"
        manager._seed_scheduler_authority({agent_id: ("Managed", config)})
        warm_lookup_complete = asyncio.Event()
        allow_reader_admission = asyncio.Event()

        class HandoffProbeExecutor(AgentManagerHostedSchedulerExecutor):
            def _execution_lease_for(self, target_agent_id):
                base_lease = super()._execution_lease_for(target_agent_id)

                @asynccontextmanager
                async def delayed_reader_lease():
                    warm_lookup_complete.set()
                    await allow_reader_admission.wait()
                    async with base_lease:
                        yield

                return delayed_reader_lease()

        executor = HandoffProbeExecutor(manager)
        execution = SchedulerExecution(
            id="execution-replacement",
            schedule_id="schedule-replacement",
            agent_id=agent_id,
            task_name="test_task",
            args={},
            scheduled_for="2026-07-25T00:00:00+00:00",
            idempotency_key="effect-replacement",
            attempt=1,
            owner="host",
        )
        scheduled = asyncio.create_task(executor.execute_scheduled(execution))
        try:
            await asyncio.wait_for(warm_lookup_complete.wait(), timeout=1)
            async with manager.scheduler_lifecycle_lock(agent_id):
                manager._agents["Managed"] = replacement
            allow_reader_admission.set()

            assert await asyncio.wait_for(scheduled, timeout=1) == "replacement"
            original_dispatch.assert_not_awaited()
            replacement_dispatch.assert_awaited_once_with("test_task", {})
        finally:
            allow_reader_admission.set()
            if not scheduled.done():
                scheduled.cancel()
            await asyncio.gather(scheduled, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_delete_serializes_real_executor_cold_wake_before_registration(
        self,
    ):
        """DELETE cannot return 404 and lose a cold wake already holding its DID lock."""
        manager = AgentManager()
        agent_id = "did:pkh:cold-delete-race"
        config = LocalAgentConfig(data_dir="cold", port=8801, autostart=False)
        manager._seed_scheduler_authority({agent_id: ("Cold", config)})

        initialization_started = asyncio.Event()
        allow_initialization = asyncio.Event()
        dispatch_started = asyncio.Event()
        allow_dispatch_finish = asyncio.Event()

        async def dispatch(_task_name, _args):
            dispatch_started.set()
            await allow_dispatch_finish.wait()
            return "dispatched"

        cold = _make_mock_agent(agent_id)
        cold.features = {
            "SchedulerFeature": SimpleNamespace(
                _dispatch_scheduled_task=dispatch,
            )
        }

        async def initialize(_name, _config, **kwargs):
            assert kwargs == {"scheduler_lifecycle_lock_held": True}
            initialization_started.set()
            await allow_initialization.wait()
            return cold

        manager._initialize_agent = AsyncMock(side_effect=initialize)
        executor = AgentManagerHostedSchedulerExecutor(
            manager,
            {agent_id: ("Cold", config)},
        )
        execution = SchedulerExecution(
            id="execution-cold-delete-race",
            schedule_id="schedule-cold-delete-race",
            agent_id=agent_id,
            task_name="test_task",
            args={},
            scheduled_for="2026-07-25T00:00:00+00:00",
            idempotency_key="effect-cold-delete-race",
            attempt=1,
            owner="host",
        )

        scheduled = asyncio.create_task(executor.execute_scheduled(execution))
        await asyncio.wait_for(initialization_started.wait(), timeout=1)

        deletion = asyncio.create_task(manager.remove_agent("Cold"))
        await asyncio.sleep(0)
        assert not deletion.done()
        assert manager.is_scheduler_agent_authorized(agent_id)

        allow_initialization.set()
        await asyncio.wait_for(dispatch_started.wait(), timeout=1)
        assert manager.get_agent("Cold") is cold
        assert not deletion.done()

        allow_dispatch_finish.set()
        assert await asyncio.wait_for(scheduled, timeout=1) == "dispatched"
        assert await asyncio.wait_for(deletion, timeout=1) is True
        assert manager.get_agent("Cold") is None
        assert not manager.is_scheduler_agent_authorized(agent_id)
        cold.shutdown.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_shared_pg_hosted_cold_wake_cancellation_releases_delete_waiter(
        self,
        monkeypatch,
        tmp_path,
    ):
        """A configured cold wake skips dynamic reacquire and cancellation drains."""

        manager = AgentManager(base_data_dir=tmp_path)
        manager.set_scheduler_polling_managed_by_host(True)
        agent_id = "did:scheduler:shared-pg-cold-cancel"
        config = LocalAgentConfig(
            data_dir="cold",
            port=8801,
            autostart=False,
        )
        manager._seed_scheduler_authority({agent_id: ("Cold", config)})
        initialization_started = asyncio.Event()
        shutdown_finished = asyncio.Event()
        registration_hook = AsyncMock(
            side_effect=AssertionError(
                "configured cold wake must not enter dynamic registration"
            )
        )
        manager.set_scheduler_tenant_registration_hook(registration_hook)

        class ColdAgent:
            def __init__(self, *, did, **_kwargs):
                self.did = did
                self.agent_id = did
                self.features = {}

            async def initialize(self):
                initialization_started.set()
                await asyncio.Event().wait()

            async def shutdown(self):
                shutdown_finished.set()

        class TestLLMService:
            async def close(self):
                return None

        monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
        monkeypatch.setenv(
            "KESTREL_DATABASE_URL",
            "postgresql://scheduler-cold-test",
        )
        monkeypatch.setattr(
            "kestrel_sovereign.multi_agent.agent_manager._get_agent_did",
            AsyncMock(return_value=agent_id),
        )
        monkeypatch.setattr(
            "kestrel_sovereign.multi_agent.agent_manager.KestrelAgent",
            ColdAgent,
        )
        monkeypatch.setattr(
            "kestrel_sovereign.multi_agent.agent_manager.LLMService",
            TestLLMService,
        )
        monkeypatch.setattr(
            LocalAgentConfig,
            "validate_runtime",
            lambda self, **_kwargs: [],
        )
        executor = AgentManagerHostedSchedulerExecutor(manager)
        execution = SchedulerExecution(
            id="execution-shared-pg-cold-cancel",
            schedule_id="schedule-shared-pg-cold-cancel",
            agent_id=agent_id,
            task_name="wait_reconcile",
            args={},
            scheduled_for="2026-07-25T00:00:00+00:00",
            idempotency_key="effect-shared-pg-cold-cancel",
            attempt=1,
            owner="host",
        )

        scheduled = asyncio.create_task(executor.execute_scheduled(execution))
        deletion = None
        try:
            await asyncio.wait_for(initialization_started.wait(), timeout=1)
            assert manager.scheduler_lifecycle_lock(agent_id).locked()

            deletion = asyncio.create_task(manager.remove_agent("Cold"))
            await asyncio.sleep(0)
            assert not deletion.done()
            assert manager.is_scheduler_agent_authorized(agent_id)

            scheduled.cancel()
            with pytest.raises(asyncio.CancelledError):
                await scheduled
            assert shutdown_finished.is_set()
            assert await asyncio.wait_for(deletion, timeout=1) is True
            assert manager.scheduler_authority_for(agent_id) is None
            assert not manager.scheduler_lifecycle_lock(agent_id).locked()
            registration_hook.assert_not_awaited()
        finally:
            if not scheduled.done():
                scheduled.cancel()
            if deletion is not None and not deletion.done():
                deletion.cancel()
            await asyncio.gather(
                scheduled,
                *(
                    (deletion,)
                    if deletion is not None
                    else ()
                ),
                return_exceptions=True,
            )

    @pytest.mark.asyncio
    async def test_scheduler_cold_wake_receives_host_a2a_and_feature_route_onboarding(
        self,
        tmp_path,
    ):
        """A scheduler-loaded tenant is integrated like an autostart tenant."""
        from kestrel_sovereign import server
        from kestrel_sovereign.a2a.did_registry import install_a2a_did_resolver
        from kestrel_sovereign.a2a.inbound_authorization import (
            has_a2a_inbound_scoped_policy,
            install_a2a_inbound_sender_authorizer,
        )
        from kestrel_sovereign.a2a.envelope_signing import (
            bound_envelope_fields,
            canonical_message,
            sign_envelope,
            verify_inbound_envelope,
        )
        from kestrel_sovereign.a2a.types import Message, TaskSendParams, TextPart
        import kestrel_sovereign.endpoints.agent as agent_endpoint
        from kestrel_sovereign.features.peers.directory import PeerRequester
        from kestrel_sovereign.identity.did_web import build_verification_methods
        from kestrel_sovereign.identity.hybrid_keypair import generate_hybrid_keypair

        host = FastAPI()
        manager = AgentManager()
        host.state.agent_manager = manager
        host.state.agent = None
        host.state.demo_mode = False

        warm_did = "did:web:example.test:agent:warm"
        warm_keypair = generate_hybrid_keypair()
        warm = SimpleNamespace(
            agent_id=warm_did,
            identity=SimpleNamespace(
                is_hybrid=True,
                signing_did=warm_did,
                new_verification_methods=build_verification_methods(
                    warm_did, warm_keypair.public_keys()
                ),
            ),
            features={},
            shutdown=AsyncMock(),
        )
        manager._register_agent("Warm", warm)
        # Model initial fleet onboarding: a later cold wake must not replace
        # this warm recipient's verification/authorization seams.
        install_a2a_did_resolver(manager, recipient=warm)
        install_a2a_inbound_sender_authorizer(manager, recipient=warm)
        warm_resolver = warm.a2a_did_resolver.__self__
        warm_authorizer = warm.a2a_inbound_sender_authorizer

        router = APIRouter()

        @router.get("/cold-only")
        async def cold_only():
            return {"cold": True}

        feature = SimpleNamespace(
            enabled=True,
            receiver=None,
            get_router=lambda: router,
        )
        cold_did = "did:web:example.test:agent:cold"
        cold_keypair = generate_hybrid_keypair()
        # Production-shaped cold KestrelAgent: the local Peers adapter is
        # feature-internal, so these injected hosted fields are genuinely None.
        cold = KestrelAgent(
            did=cold_did,
            storage_path=str(tmp_path / "cold" / "kestrel_prime.db"),
        )
        cold.identity = SimpleNamespace(
            is_hybrid=True,
            signing_did=cold_did,
            new_verification_methods=build_verification_methods(
                cold_did, cold_keypair.public_keys()
            ),
        )
        # Normal AgentManager construction leaves these public injection attrs
        # empty. Its PeersFeature owns the live local-host route instead.
        # Onboarding must bind the immutable manager policy to this pair.
        live_router = SimpleNamespace(
            authorize_inbound_sender=AsyncMock(return_value=True),
        )
        live_requester = PeerRequester(cold_did, object())
        live_peers_feature = SimpleNamespace(
            hosted_peer_directory_context=lambda: (live_router, live_requester),
            get_router=lambda: None,
        )
        cold.features = {
            "ColdOnly": feature,
            "PeersFeature": live_peers_feature,
        }
        cold.peer_directory_router = None
        cold.peer_requester = None
        cold.task_manager = SimpleNamespace(
            create_task=AsyncMock(return_value=SimpleNamespace(id="cold-wake-a2a")),
        )
        cold.shutdown = AsyncMock()
        cold.wait_for_shutdown_completion = None
        cold._set_display_name = lambda _name: None
        config = LocalAgentConfig(data_dir="cold", port=8802, autostart=False)
        manager._seed_scheduler_authority({cold.agent_id: ("Cold", config)})
        manager._initialize_agent = AsyncMock(return_value=cold)
        manager.set_agent_registration_hook(
            lambda name, agent: server._onboard_host_registered_agent(
                host, manager, name, agent
            )
        )

        loaded = await manager.load_agent(
            "Cold", config, expected_agent_id=cold.agent_id
        )

        assert loaded is cold
        assert cold.a2a_did_resolver(warm_did)["id"] == warm_did
        assert warm.a2a_did_resolver(cold_did)["id"] == cold_did
        assert warm.a2a_did_resolver.__self__ is warm_resolver
        assert cold.a2a_did_resolver.__self__ is not warm.a2a_did_resolver.__self__
        assert warm.a2a_inbound_sender_authorizer is warm_authorizer
        assert cold.a2a_inbound_sender_authorizer is not warm_authorizer
        assert has_a2a_inbound_scoped_policy(cold) is True
        assert cold.a2a_inbound_sender_authorizer.requires_verified_sender is True
        assert cold.a2a_inbound_sender_authorizer.has_valid_current_scope() is False
        assert cold._a2a_host_manager is manager
        hosted_policy = manager.a2a_hosted_policy_for(cold)
        assert hosted_policy is not None
        assert hosted_policy.router is live_router
        assert hosted_policy.requester is live_requester

        # Exercise the actual verification seam, not merely resolver lookup:
        # a signed warm→cold same-host envelope must validate after the cold
        # tenant is registered through the scheduler path.
        timestamp = datetime.now(timezone.utc).isoformat()
        metadata = {"sender": warm_did}
        metadata["signature"] = sign_envelope(
            warm_keypair,
            sender=warm_did,
            task_id="cold-wake-a2a",
            message="scheduler woke cold peer",
            timestamp=timestamp,
            session_id="cold-wake-session",
            bound=bound_envelope_fields(metadata),
        )
        verdict = await verify_inbound_envelope(
            metadata,
            task_id="cold-wake-a2a",
            message="scheduler woke cold peer",
            session_id="cold-wake-session",
            resolver=cold.a2a_did_resolver,
            require_signed=True,
        )
        assert verdict.ok is True and verdict.verified is True

        # Exercise the recipient's real verified-send path under the manager
        # lease. The raw agent attrs remain None, so this would fail if the
        # hosted policy had not captured PeersFeature's effective context.
        send_metadata = {"sender": warm_did}
        send_metadata["signature"] = sign_envelope(
            warm_keypair,
            sender=warm_did,
            task_id="cold-wake-a2a",
            message=canonical_message(["scheduler woke cold peer"]),
            timestamp=datetime.now(timezone.utc).isoformat(),
            session_id="cold-wake-session",
            bound=bound_envelope_fields(send_metadata),
        )
        params = TaskSendParams(
            id="cold-wake-a2a",
            sessionId="cold-wake-session",
            message=Message(
                role="user", parts=[TextPart(text="scheduler woke cold peer")],
            ),
            metadata=send_metadata,
        )
        created = await agent_endpoint._create_verified_a2a_task(
            cold,
            params,
            params.message.parts,
            [],
            [],
        )
        assert created.id == "cold-wake-a2a"
        live_router.authorize_inbound_sender.assert_awaited_once_with(
            live_requester, warm_did,
        )
        assert cold.task_manager.create_task.await_count == 1
        with TestClient(host) as client:
            response = client.get("/cold-only")
            assert response.status_code == 200
            assert await manager.remove_agent("Cold") is True
            deleted_response = client.get("/cold-only")
        assert response.status_code == 200
        assert response.json() == {"cold": True}
        assert deleted_response.status_code == 404

    @pytest.mark.asyncio
    async def test_remove_nonexistent_agent(self):
        manager = AgentManager()
        result = await manager.remove_agent("ghost")
        assert result is False

    @pytest.mark.asyncio
    async def test_shutdown_all(self):
        manager = AgentManager()
        agent1 = _make_mock_agent("did:1")
        agent2 = _make_mock_agent("did:2")
        manager._agents["A"] = agent1
        manager._agents["B"] = agent2
        manager._agent_names["did:1"] = "A"
        manager._agent_names["did:2"] = "B"

        await manager.shutdown_all()
        assert len(manager._agents) == 0
        agent1.shutdown.assert_awaited_once()
        agent2.shutdown.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_remove_agent_joins_deferred_durable_close_before_unpublishing(
        self, tmp_path
    ):
        """One production removal call drains the real SQLite worker (#2713)."""
        from kestrel_sovereign.signals import (
            OrderedLockManager,
            SignalDispatcher,
            SignalLogStore,
            SourceRegistry,
        )
        from kestrel_sovereign.storage.db import SQLiteBackend

        manager = AgentManager()
        agent = KestrelAgent(
            did="did:test:manager-durable-shutdown",
            storage_path=str(tmp_path / "agent.db"),
        )
        backend = SQLiteBackend(str(tmp_path / "ledger.db"))
        await backend.connect()
        worker = aiosqlite_worker(backend._connection)
        log_store = SignalLogStore(backend)
        await log_store.initialize()
        dispatcher = SignalDispatcher(
            agent=agent,
            registry=SourceRegistry(),
            lock_manager=OrderedLockManager(),
            store=log_store,
        )
        await dispatcher.initialize_durable_delivery()
        agent.dispatcher = dispatcher

        original_release = dispatcher._durable_store.release_initial_reservations
        release_entered = asyncio.Event()
        allow_release = asyncio.Event()
        storage_closed: list[bool] = []

        async def block_release(*args, **kwargs):
            release_entered.set()
            await allow_release.wait()
            return await original_release(*args, **kwargs)

        class _Storage:
            async def close(self):
                storage_closed.append(True)
                await backend.close()

        dispatcher._durable_store.release_initial_reservations = block_release
        agent.features = {}
        agent.llm_service = None
        agent.task_manager = None
        agent.memory_system = None
        agent._sync_service = None
        agent.storage = _Storage()
        manager._agents["Managed"] = agent
        manager._agent_names[agent.agent_id] = "Managed"

        try:
            # The manager's normal outer timeout cancels the bounded agent
            # shutdown. It must then join the agent-owned continuation rather
            # than removing this routing entry or releasing its resources.
            with patch(
                "kestrel_sovereign.multi_agent.agent_manager.SHUTDOWN_TIMEOUT",
                0.05,
            ):
                remove_task = asyncio.create_task(manager.remove_agent("Managed"))
                await asyncio.wait_for(release_entered.wait(), timeout=1.0)
                await asyncio.sleep(0.1)

                assert manager.get_agent("Managed") is agent
                assert not remove_task.done()
                assert storage_closed == []

                allow_release.set()
                assert await asyncio.wait_for(remove_task, timeout=1.0) is True

            assert manager.get_agent("Managed") is None
            assert storage_closed == [True]
            assert backend._connection is None
            for _ in range(100):
                if not worker.is_alive():
                    break
                await asyncio.sleep(0)
            assert not worker.is_alive()
        finally:
            allow_release.set()
            dispatcher._durable_store.release_initial_reservations = original_release
            if backend._connection is not None:
                await backend.close()

    @pytest.mark.asyncio
    async def test_shutdown_handles_errors(self):
        """Shutdown should continue even if one agent errors."""
        manager = AgentManager()
        agent1 = _make_mock_agent("did:1")
        agent1.shutdown = AsyncMock(side_effect=Exception("boom"))
        agent2 = _make_mock_agent("did:2")
        manager._agents["A"] = agent1
        manager._agents["B"] = agent2
        manager._agent_names["did:1"] = "A"
        manager._agent_names["did:2"] = "B"

        with pytest.raises(ExceptionGroup, match="fleet agents failed"):
            await manager.shutdown_all()
        # Both are attempted, but a failed shutdown stays published. Removing
        # it would discard the lifecycle owner before durable cleanup can be
        # confirmed on a later retry. The aggregate is raised only after B has
        # received its own cleanup attempt.
        assert set(manager._agents) == {"A"}
        assert manager.get_agent("B") is None
        agent1.shutdown.assert_awaited_once()
        agent2.shutdown.assert_awaited_once()


class TestLoadFromConfig:
    """Test loading agents from MultiAgentConfig."""

    @pytest.mark.asyncio
    async def test_load_from_config_initializes_concurrently_and_registers_in_order(self):
        """Slow agents overlap without making fleet/UI order nondeterministic."""
        config = MultiAgentConfig(
            agents={
                "first": LocalAgentConfig(data_dir=Path("/tmp/first"), port=8801),
                "second": LocalAgentConfig(data_dir=Path("/tmp/second"), port=8802),
            }
        )
        manager = AgentManager(base_data_dir=Path("/tmp"))
        both_started = asyncio.Event()
        release = asyncio.Event()
        started = []

        async def initialize(name, _config):
            started.append(name)
            if len(started) == 2:
                both_started.set()
            await release.wait()
            return _make_mock_agent(f"did:{name}")

        manager._initialize_agent = initialize
        load_task = asyncio.create_task(manager.load_from_config(config))

        await asyncio.wait_for(both_started.wait(), timeout=1)
        assert started == ["first", "second"]
        release.set()

        assert await load_task == 2
        assert list(manager._agents) == ["first", "second"]

    @pytest.mark.asyncio
    async def test_cancelled_startup_shuts_down_completed_unregistered_agent(self):
        config = MultiAgentConfig(
            agents={
                "ready": LocalAgentConfig(data_dir=Path("/tmp/ready"), port=8801),
                "blocked": LocalAgentConfig(data_dir=Path("/tmp/blocked"), port=8802),
            }
        )
        manager = AgentManager(base_data_dir=Path("/tmp"))
        ready_agent = _make_mock_agent("did:ready")
        ready = asyncio.Event()
        block = asyncio.Event()

        async def initialize(name, _config):
            if name == "ready":
                ready.set()
                return ready_agent
            await block.wait()
            return _make_mock_agent("did:blocked")

        manager._initialize_agent = initialize
        load_task = asyncio.create_task(manager.load_from_config(config))
        await asyncio.wait_for(ready.wait(), timeout=1)
        await asyncio.sleep(0)

        load_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await load_task

        ready_agent.shutdown.assert_awaited_once()
        assert manager.list_agents() == {}

    @pytest.mark.asyncio
    @patch(
        "kestrel_sovereign.multi_agent.agent_manager._get_agent_did",
        new_callable=AsyncMock,
    )
    @patch("kestrel_sovereign.multi_agent.agent_manager.KestrelAgent")
    @patch("kestrel_sovereign.multi_agent.agent_manager.LLMService")
    async def test_failed_initializer_shuts_down_partial_agent(
        self, mock_llm_cls, mock_agent_cls, mock_get_did
    ):
        mock_get_did.return_value = "did:partial"
        partial = _make_mock_agent("did:partial")
        partial.initialize.side_effect = RuntimeError("init failed")
        mock_agent_cls.return_value = partial
        manager = AgentManager(base_data_dir=Path("/tmp"))
        config = LocalAgentConfig(data_dir=Path("/tmp/partial"), port=8801)

        with patch.object(LocalAgentConfig, "validate_runtime", return_value=[]):
            with pytest.raises(RuntimeError, match="init failed"):
                await manager._initialize_agent("partial", config)

        mock_get_did.assert_awaited_once_with(
            str(Path("/tmp/partial").resolve()),
            mode=_AgentDIDLookupMode.INITIALIZATION,
        )
        partial.shutdown.assert_awaited_once()

    @pytest.mark.asyncio
    @patch(
        "kestrel_sovereign.multi_agent.agent_manager._get_agent_did",
        new_callable=AsyncMock,
    )
    @patch("kestrel_sovereign.multi_agent.agent_manager.KestrelAgent")
    @patch("kestrel_sovereign.multi_agent.agent_manager.LLMService")
    async def test_in_process_agent_receives_resolved_identity_export_override(
        self,
        mock_llm_cls,
        mock_agent_cls,
        mock_get_did,
        tmp_path,
    ):
        mock_get_did.return_value = "did:claw"
        mock_agent = _make_mock_agent("did:claw")
        mock_agent_cls.return_value = mock_agent
        manager = AgentManager(base_data_dir=tmp_path)
        config = LocalAgentConfig(
            data_dir=Path("agent_data/claw"),
            identity_export_dir=Path("continuity"),
            port=8801,
        )

        with patch.object(LocalAgentConfig, "validate_runtime", return_value=[]):
            await manager._initialize_agent("claw", config)

        assert mock_agent_cls.call_args.kwargs["identity_export_dir"] == (
            tmp_path / "agent_data" / "claw" / "continuity"
        ).resolve()

    @pytest.mark.asyncio
    @patch(
        "kestrel_sovereign.multi_agent.agent_manager._get_agent_did",
        new_callable=AsyncMock,
    )
    @patch("kestrel_sovereign.multi_agent.agent_manager.KestrelAgent")
    @patch("kestrel_sovereign.multi_agent.agent_manager.LLMService")
    async def test_in_process_agent_defaults_export_binding_to_its_data_root(
        self,
        mock_llm_cls,
        mock_agent_cls,
        mock_get_did,
        tmp_path,
    ):
        mock_get_did.return_value = "did:claw"
        mock_agent_cls.return_value = _make_mock_agent("did:claw")
        manager = AgentManager(base_data_dir=tmp_path)
        config = LocalAgentConfig(
            data_dir=Path("agent_data/claw"),
            port=8801,
        )

        with patch.object(LocalAgentConfig, "validate_runtime", return_value=[]):
            await manager._initialize_agent("claw", config)

        agent_root = (tmp_path / "agent_data" / "claw").resolve()
        assert mock_agent_cls.call_args.kwargs["identity_export_dir"] == agent_root

    @pytest.mark.asyncio
    @patch("kestrel_sovereign.multi_agent.agent_manager._get_agent_did", new_callable=AsyncMock)
    @patch("kestrel_sovereign.multi_agent.agent_manager.KestrelAgent")
    @patch("kestrel_sovereign.multi_agent.agent_manager.LLMService")
    async def test_load_from_config_skips_non_autostart(
        self, mock_llm_cls, mock_agent_cls, mock_get_did
    ):
        """Agents with autostart=false should be skipped."""
        config = MultiAgentConfig(
            agents={
                "active": LocalAgentConfig(data_dir=Path("/tmp/active"), port=8801, autostart=True),
                "inactive": LocalAgentConfig(data_dir=Path("/tmp/inactive"), port=8802, autostart=False),
            }
        )

        mock_get_did.return_value = "did:active"
        mock_agent = _make_mock_agent("did:active")
        mock_agent_cls.return_value = mock_agent

        manager = AgentManager(base_data_dir=Path("/tmp"))

        # Patch validate_runtime to return no errors
        with patch.object(LocalAgentConfig, "validate_runtime", return_value=[]):
            loaded = await manager.load_from_config(config)

        assert loaded == 1
        assert manager.get_agent("active") is mock_agent
        assert manager.get_agent("inactive") is None

    @pytest.mark.asyncio
    @patch("kestrel_sovereign.multi_agent.agent_manager._get_agent_did", new_callable=AsyncMock)
    @patch("kestrel_sovereign.multi_agent.agent_manager.KestrelAgent")
    @patch("kestrel_sovereign.multi_agent.agent_manager.LLMService")
    async def test_load_from_config_handles_errors(
        self, mock_llm_cls, mock_agent_cls, mock_get_did
    ):
        """Failed agent loads should log error but not crash."""
        config = MultiAgentConfig(
            agents={
                "broken": LocalAgentConfig(data_dir=Path("/tmp/broken"), port=8801, autostart=True),
            }
        )

        # validate_runtime returns errors
        with patch.object(LocalAgentConfig, "validate_runtime", return_value=["missing db"]):
            manager = AgentManager(base_data_dir=Path("/tmp"))
            loaded = await manager.load_from_config(config)

        assert loaded == 0
        assert manager.get_agent("broken") is None

    @pytest.mark.asyncio
    @patch("kestrel_sovereign.multi_agent.agent_manager._get_agent_did", new_callable=AsyncMock)
    @patch("kestrel_sovereign.multi_agent.agent_manager.KestrelAgent")
    @patch("kestrel_sovereign.multi_agent.agent_manager.LLMService")
    async def test_load_from_config_records_init_failures(
        self, mock_llm_cls, mock_agent_cls, mock_get_did
    ):
        """Per-agent init failures are exposed via manager.init_failures so
        the lifespan handler can surface them via /health (#377 lifecycle
        hardening for multi-agent boot — codex review v3 followup).
        """
        from kestrel_sovereign.lifecycle_checks import NoLLMProvidersError

        config = MultiAgentConfig(
            agents={
                "good": LocalAgentConfig(data_dir=Path("/tmp/good"), port=8801, autostart=True),
                "muted": LocalAgentConfig(data_dir=Path("/tmp/muted"), port=8802, autostart=True),
            }
        )

        mock_get_did.side_effect = ["did:good", "did:muted"]
        good_agent = _make_mock_agent("did:good")
        muted_agent = _make_mock_agent("did:muted")
        # The muted agent's initialize raises the lifecycle hardening error.
        muted_agent.initialize = AsyncMock(
            side_effect=NoLLMProvidersError("no providers for muted")
        )
        mock_agent_cls.side_effect = [good_agent, muted_agent]

        with patch.object(LocalAgentConfig, "validate_runtime", return_value=[]):
            manager = AgentManager(base_data_dir=Path("/tmp"))
            loaded = await manager.load_from_config(config)

        assert loaded == 1
        assert manager.get_agent("good") is good_agent
        assert manager.get_agent("muted") is None

        failures = manager.init_failures
        assert len(failures) == 1
        name, exc = failures[0]
        assert name == "muted"
        assert isinstance(exc, NoLLMProvidersError)
        assert "no providers for muted" in str(exc)

    @pytest.mark.asyncio
    async def test_init_failures_resets_on_each_load(self):
        """A fresh load_from_config call clears prior failures."""
        manager = AgentManager(base_data_dir=Path("/tmp"))
        manager._init_failures = [("stale", RuntimeError("from a previous run"))]

        empty_config = MultiAgentConfig(agents={})
        loaded = await manager.load_from_config(empty_config)

        assert loaded == 0
        assert manager.init_failures == []

    @pytest.mark.asyncio
    @patch(
        "kestrel_sovereign.multi_agent.agent_manager._get_agent_did",
        new_callable=AsyncMock,
    )
    @patch("kestrel_sovereign.multi_agent.agent_manager.KestrelAgent")
    @patch("kestrel_sovereign.multi_agent.agent_manager.LLMService")
    async def test_mandatory_failure_never_publishes_partial_agent(
        self, mock_llm_cls, mock_agent_cls, mock_get_did
    ):
        config = MultiAgentConfig(
            agents={
                "secure": LocalAgentConfig(
                    data_dir=Path("/tmp/secure"), port=8801
                ),
                "broken": LocalAgentConfig(
                    data_dir=Path("/tmp/broken"), port=8802
                ),
            }
        )
        mock_get_did.side_effect = ["did:secure", "did:broken"]
        secure = _make_mock_agent("did:secure")
        broken = _make_mock_agent("did:broken")
        failure = MandatoryFeatureReadinessError(
            "SecurityFeature",
            "initialization",
            "could not initialize",
        )
        broken.initialize.side_effect = failure
        mock_agent_cls.side_effect = [secure, broken]

        with patch.object(LocalAgentConfig, "validate_runtime", return_value=[]):
            manager = AgentManager(base_data_dir=Path("/tmp"))
            loaded = await manager.load_from_config(config)

        assert loaded == 1
        assert manager.list_agents() == {"secure": secure}
        assert manager.get_agent("broken") is None
        broken.shutdown.assert_awaited_once()
        assert manager.init_failures == [("broken", failure)]

    @pytest.mark.asyncio
    async def test_identity_failure_keeps_healthy_peer_but_never_publishes_broken(
        self,
        caplog,
    ):
        """Fleet startup records a sanitized, non-invokable partial state."""
        config = MultiAgentConfig(
            agents={
                "healthy": LocalAgentConfig(
                    data_dir=Path("/tmp/healthy"), port=8801
                ),
                "broken": LocalAgentConfig(
                    data_dir=Path("/tmp/broken"), port=8802
                ),
            }
        )
        manager = AgentManager(base_data_dir=Path("/tmp"))
        healthy = _make_mock_agent("did:healthy")
        failure = IdentityReadinessError(
            "custody",
            cause_type="DecryptionError",
        )

        async def initialize(name, _config):
            if name == "broken":
                raise failure
            return healthy

        manager._initialize_agent = initialize
        loaded = await manager.load_from_config(config)

        assert loaded == 1
        assert manager.list_agents() == {"healthy": healthy}
        assert manager.get_agent("broken") is None
        assert manager.init_failures == [("broken", failure)]
        assert "identity_custody" in caplog.text
        assert "/tmp/broken" not in caplog.text


class TestCreateAgent:
    """Test create_agent (inception + load)."""

    @pytest.mark.asyncio
    async def test_create_agent_duplicate_name_raises(self):
        """Creating an agent with an existing name should raise ValueError."""
        manager = AgentManager()
        mock = _make_mock_agent("did:existing")
        manager._agents["Claw"] = mock
        manager._agent_names["did:existing"] = "Claw"

        with pytest.raises(ValueError, match="already exists"):
            await manager.create_agent("Claw")

    @pytest.mark.asyncio
    async def test_create_agent_duplicate_case_insensitive(self):
        """Duplicate check should be case-insensitive."""
        manager = AgentManager()
        mock = _make_mock_agent("did:existing")
        manager._agents["Claw"] = mock
        manager._agent_names["did:existing"] = "Claw"

        with pytest.raises(ValueError, match="already exists"):
            await manager.create_agent("claw")

    @pytest.mark.asyncio
    @patch("kestrel_sovereign.inception_service.create_kestrel_identity_async", new_callable=AsyncMock)
    @patch("kestrel_sovereign.multi_agent.agent_manager._get_agent_did", new_callable=AsyncMock)
    @patch("kestrel_sovereign.multi_agent.agent_manager.KestrelAgent")
    @patch("kestrel_sovereign.multi_agent.agent_manager.LLMService")
    async def test_create_agent_success(
        self, mock_llm_cls, mock_agent_cls, mock_get_did, mock_inception, tmp_path
    ):
        """create_agent should run inception and load the agent."""
        mock_get_did.return_value = "did:new-agent"
        mock_agent = _make_mock_agent("did:new-agent")
        mock_agent_cls.return_value = mock_agent

        manager = AgentManager(base_data_dir=tmp_path)

        with patch.object(LocalAgentConfig, "validate_runtime", return_value=[]):
            agent = await manager.create_agent("NewBot")

        mock_inception.assert_awaited_once()
        assert agent is mock_agent
        assert manager.get_agent("NewBot") is mock_agent
        # Fleet-idleness (#F235): a dynamically-created/spawned agent must get
        # the co-hosted-agents provider so its restart requests cannot bypass
        # the whole-fleet idle gate. Resolves live to the manager's agents.
        assert callable(agent._cohosted_agents_provider)
        assert agent in agent._cohosted_agents_provider()

    @pytest.mark.asyncio
    @patch("kestrel_sovereign.inception_service.create_kestrel_identity_async", new_callable=AsyncMock)
    @patch("kestrel_sovereign.multi_agent.agent_manager._get_agent_did", new_callable=AsyncMock)
    @patch("kestrel_sovereign.multi_agent.agent_manager.KestrelAgent")
    @patch("kestrel_sovereign.multi_agent.agent_manager.LLMService")
    async def test_create_agent_passes_parent_did(
        self, mock_llm_cls, mock_agent_cls, mock_get_did, mock_inception, tmp_path
    ):
        """create_agent should forward parent_did to inception service."""
        mock_get_did.return_value = "did:child"
        mock_agent = _make_mock_agent("did:child")
        mock_agent_cls.return_value = mock_agent

        manager = AgentManager(base_data_dir=tmp_path)

        with patch.object(LocalAgentConfig, "validate_runtime", return_value=[]):
            await manager.create_agent("ChildBot", parent_did="did:parent-abc")

        # Verify inception was called WITH parent_did
        mock_inception.assert_awaited_once()
        call_kwargs = mock_inception.call_args[1]
        assert call_kwargs["parent_did"] == "did:parent-abc"
        assert call_kwargs["agent_name"] == "ChildBot"


class TestSpawnAgent:
    """Test spawn_agent (delegation chain)."""

    @pytest.mark.asyncio
    @patch("kestrel_sovereign.inception_service.create_kestrel_identity_async", new_callable=AsyncMock)
    @patch("kestrel_sovereign.multi_agent.agent_manager._get_agent_did", new_callable=AsyncMock)
    @patch("kestrel_sovereign.multi_agent.agent_manager.KestrelAgent")
    @patch("kestrel_sovereign.multi_agent.agent_manager.LLMService")
    async def test_spawn_passes_parent_did_to_create(
        self, mock_llm_cls, mock_agent_cls, mock_get_did, mock_inception, tmp_path
    ):
        """spawn_agent should pass parent's DID to create_agent for delegation."""
        mock_get_did.return_value = "did:spawned-child"
        mock_child = _make_mock_agent("did:spawned-child")
        mock_agent_cls.return_value = mock_child

        manager = AgentManager(base_data_dir=tmp_path)

        parent = _make_mock_agent("did:parent-xyz")
        parent._private_key = None  # No key — skip signing
        parent.identity = None

        mandate = SpawnMandate(
            parent_did="did:parent-xyz",
            purpose="test spawn",
        )

        with patch.object(LocalAgentConfig, "validate_runtime", return_value=[]):
            await manager.spawn_agent("SpawnedBot", parent, mandate)

        # Verify inception received parent_did
        call_kwargs = mock_inception.call_args[1]
        assert call_kwargs["parent_did"] == "did:parent-xyz"

        # Verify parent-child tracking
        assert "SpawnedBot" in manager.get_children("did:parent-xyz")
        assert manager.get_mandate("SpawnedBot") is mandate
        assert mandate.child_did == "did:spawned-child"

    @pytest.mark.asyncio
    @patch("kestrel_sovereign.inception_service.create_kestrel_identity_async", new_callable=AsyncMock)
    @patch("kestrel_sovereign.multi_agent.agent_manager._get_agent_did", new_callable=AsyncMock)
    @patch("kestrel_sovereign.multi_agent.agent_manager.KestrelAgent")
    @patch("kestrel_sovereign.multi_agent.agent_manager.LLMService")
    async def test_spawn_wires_mandate_features_into_child(
        self, mock_llm_cls, mock_agent_cls, mock_get_did, mock_inception, tmp_path
    ):
        """The mandate's feature allowlist must reach the child KestrelAgent.

        Regression for #1946: spawn validated ``features_allowed`` but never
        threaded it into the child's config, so ``load_agent`` built the child
        with ``allowed_features=None`` and it loaded ALL features regardless of
        what the mandate permitted. This drives the real
        spawn_agent -> _do_spawn -> create_agent -> load_agent chain (only
        inception/DID/KestrelAgent/LLMService are mocked) and asserts the child
        is constructed with the allowlist as ``allowed_features``.
        """
        mock_get_did.return_value = "did:featured-child"
        mock_child = _make_mock_agent("did:featured-child")
        mock_agent_cls.return_value = mock_child

        manager = AgentManager(base_data_dir=tmp_path)

        parent = _make_mock_agent("did:parent-feat")
        parent._private_key = None  # No key — skip signing
        parent.identity = None

        mandate = SpawnMandate(
            parent_did="did:parent-feat",
            purpose="restricted child",
            features_allowed=["MemoryFeature", "WebSearchFeature"],
        )

        with patch.object(LocalAgentConfig, "validate_runtime", return_value=[]):
            await manager.spawn_agent("FeaturedBot", parent, mandate)

        # The child KestrelAgent must be built WITH the allowlist, not None.
        assert mock_agent_cls.call_count == 1
        child_kwargs = mock_agent_cls.call_args.kwargs
        assert child_kwargs["allowed_features"] == {"MemoryFeature", "WebSearchFeature"}

    @pytest.mark.asyncio
    @patch("kestrel_sovereign.inception_service.create_kestrel_identity_async", new_callable=AsyncMock)
    @patch("kestrel_sovereign.multi_agent.agent_manager._get_agent_did", new_callable=AsyncMock)
    @patch("kestrel_sovereign.multi_agent.agent_manager.KestrelAgent")
    @patch("kestrel_sovereign.multi_agent.agent_manager.LLMService")
    async def test_spawn_without_features_loads_all(
        self, mock_llm_cls, mock_agent_cls, mock_get_did, mock_inception, tmp_path
    ):
        """An empty (default) mandate allowlist means "load all" (allowed_features=None)."""
        mock_get_did.return_value = "did:open-child"
        mock_child = _make_mock_agent("did:open-child")
        mock_agent_cls.return_value = mock_child

        manager = AgentManager(base_data_dir=tmp_path)

        parent = _make_mock_agent("did:parent-open")
        parent._private_key = None
        parent.identity = None

        # No features_allowed → default empty list → load all features.
        mandate = SpawnMandate(parent_did="did:parent-open", purpose="open child")

        with patch.object(LocalAgentConfig, "validate_runtime", return_value=[]):
            await manager.spawn_agent("OpenBot", parent, mandate)

        assert mock_agent_cls.call_count == 1
        assert mock_agent_cls.call_args.kwargs["allowed_features"] is None

    @pytest.mark.asyncio
    @patch("kestrel_sovereign.inception_service.create_kestrel_identity_async", new_callable=AsyncMock)
    @patch("kestrel_sovereign.multi_agent.agent_manager._get_agent_did", new_callable=AsyncMock)
    @patch("kestrel_sovereign.multi_agent.agent_manager.KestrelAgent")
    @patch("kestrel_sovereign.multi_agent.agent_manager.LLMService")
    async def test_spawn_duplicate_name_raises(
        self, mock_llm_cls, mock_agent_cls, mock_get_did, mock_inception, tmp_path
    ):
        """spawn_agent should fail if child name already exists."""
        manager = AgentManager(base_data_dir=tmp_path)
        manager._agents["Existing"] = _make_mock_agent("did:existing")

        parent = _make_mock_agent("did:parent")
        parent._private_key = None
        parent.identity = None

        mandate = SpawnMandate(parent_did="did:parent", purpose="dupe test")

        with pytest.raises(ValueError, match="already exists"):
            await manager.spawn_agent("Existing", parent, mandate)
