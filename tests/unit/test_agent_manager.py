"""Unit tests for the in-process AgentManager."""

import asyncio
import os
import subprocess
import sys
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.testclient import TestClient

from kestrel_sovereign.features import MandatoryFeatureReadinessError
from kestrel_sovereign.features.isolated_runtime import (
    IsolatedRuntimeNamespaceError,
    derive_isolated_runtime_namespace,
    prepare_isolated_runtime_namespace,
    resolve_isolated_runtime_namespace,
)
from kestrel_sovereign.features.scheduler.runner import (
    AgentManagerHostedSchedulerExecutor,
    SchedulerExecution,
)
from kestrel_sovereign.identity.local_anchor import AgentDIDLookupMode
from kestrel_sovereign.identity.runtime_identity import IdentityReadinessError
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.knowledge import InferenceError, InferenceProfile, OntologyRef
from kestrel_sovereign.multi_agent.agent_manager import (
    AgentManager,
    ChildTerminationReconciliationError,
    RUNTIME_OFFBOARD_TIMEOUT_S,
    RuntimeOffboardingRetainedError,
    _parse_runtime_offboard_timeout,
)
from kestrel_sovereign.multi_agent.config import LocalAgentConfig, MultiAgentConfig
from kestrel_sovereign.spawn.mandate import SpawnMandate
from tests.utils.aiosqlite_workers import aiosqlite_worker


def _make_mock_agent(agent_id: str = "did:pkh:eip155:1:0xABC"):
    """Create a mock KestrelAgent with the minimum interface."""
    agent = MagicMock()
    agent.agent_id = agent_id
    agent.initialize = AsyncMock()
    agent.shutdown = AsyncMock()
    agent.get_agent_card = AsyncMock()
    return agent


def _exception_group_contains(
    error: BaseException, expected: type[BaseException]
) -> bool:
    if isinstance(error, expected):
        return True
    if isinstance(error, BaseExceptionGroup):
        return any(
            _exception_group_contains(item, expected) for item in error.exceptions
        )
    return False


def _exception_leaves(error: BaseException) -> list[BaseException]:
    if isinstance(error, BaseExceptionGroup):
        leaves: list[BaseException] = []
        for item in error.exceptions:
            leaves.extend(_exception_leaves(item))
        return leaves
    return [error]


@pytest.mark.parametrize("value", ["30s", "nan", "inf", "0", "-1"])
def test_runtime_offboard_timeout_rejects_invalid_values_actionably(value):
    with pytest.raises(
        ValueError,
        match=(
            "^KESTREL_RUNTIME_OFFBOARD_TIMEOUT_S must be finite and positive$"
        ),
    ):
        _parse_runtime_offboard_timeout(value)


@pytest.mark.parametrize(("value", "expected"), [("0.25", 0.25), (30, 30.0)])
def test_runtime_offboard_timeout_accepts_finite_positive_values(value, expected):
    assert _parse_runtime_offboard_timeout(value) == expected


def test_runtime_offboard_timeout_fresh_import_names_malformed_variable():
    env = os.environ.copy()
    env["KESTREL_RUNTIME_OFFBOARD_TIMEOUT_S"] = "30s"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import kestrel_sovereign.multi_agent.agent_manager",
        ],
        cwd=Path(__file__).parents[2],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    diagnostic = completed.stdout + completed.stderr
    assert "KESTREL_RUNTIME_OFFBOARD_TIMEOUT_S" in diagnostic
    assert "could not convert string to float" not in diagnostic


def test_runtime_offboard_timeout_has_independent_thirty_second_default():
    env = os.environ.copy()
    env.pop("KESTREL_RUNTIME_OFFBOARD_TIMEOUT_S", None)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from kestrel_sovereign.multi_agent.agent_manager import "
                "RUNTIME_OFFBOARD_TIMEOUT_S; print(RUNTIME_OFFBOARD_TIMEOUT_S)"
            ),
        ],
        cwd=Path(__file__).parents[2],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert completed.stdout.strip() == "30.0"
    if "KESTREL_RUNTIME_OFFBOARD_TIMEOUT_S" not in os.environ:
        assert RUNTIME_OFFBOARD_TIMEOUT_S == 30.0


def test_env_example_documents_runtime_offboard_timeout():
    env_example = (Path(__file__).parents[2] / ".env.example").read_text()
    assert "# KESTREL_RUNTIME_OFFBOARD_TIMEOUT_S=30" in env_example
    assert "# Default: 30 seconds" in env_example


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
            "kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did",
            did_lookup,
        )

        mapping = await manager.local_agent_configs_by_did(config)

        assert mapping["did:pkh:warm"][0] == "Warm"
        assert mapping["did:pkh:cold"][0] == "Cold"
        did_lookup.assert_awaited_once_with(
            str(cold_dir),
            mode=AgentDIDLookupMode.COLD_READ_ONLY,
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
            "kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did",
            did_lookup,
        )

        mapping = await manager.local_agent_configs_by_did(config)

        assert mapping == {"did:pkh:warm": ("Warm", config.agents["Warm"])}
        assert manager.cold_scheduler_identity_failures == [
            ("Unincepted", missing_identity)
        ]
        did_lookup.assert_awaited_once_with(
            str(tmp_path / "unincepted"),
            mode=AgentDIDLookupMode.COLD_READ_ONLY,
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
            "kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did",
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
            "kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did",
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
            mode=AgentDIDLookupMode.INITIALIZATION,
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
    @pytest.mark.parametrize("cancel_onboarding", [False, True])
    async def test_rejected_onboarding_withdraws_before_a2a_reader_or_cleanup(
        self, cancel_onboarding: bool
    ) -> None:
        """A failed/cancelled hook cannot expose its partial publication."""

        manager = AgentManager()
        agent = _make_mock_agent("did:test:rejected-onboarding")
        config = LocalAgentConfig(data_dir="rejected", port=8801)
        hook_entered = asyncio.Event()
        allow_failure = asyncio.Event()
        cleanup_started = asyncio.Event()
        allow_cleanup = asyncio.Event()

        async def reject_onboarding(_name, registered_agent) -> None:
            manager.install_a2a_hosted_policy(
                registered_agent,
                resolver=object(),
                authorizer=object(),
                router=object(),
                requester=object(),
            )
            hook_entered.set()
            await allow_failure.wait()
            raise RuntimeError("host onboarding rejected")

        original_cleanup = manager._discard_unpublished_initialized_agents

        async def pause_cleanup(*args, **kwargs) -> bool:
            cleanup_started.set()
            await allow_cleanup.wait()
            return await original_cleanup(*args, **kwargs)

        manager._initialize_agent = AsyncMock(return_value=agent)
        manager.set_agent_registration_hook(reject_onboarding)
        manager._discard_unpublished_initialized_agents = pause_cleanup
        load = asyncio.create_task(manager.load_agent("Rejected", config))
        await asyncio.wait_for(hook_entered.wait(), timeout=1.0)

        async def observe_reader():
            async with manager.a2a_execution_lease():
                return (
                    manager.get_agent("Rejected"),
                    manager.a2a_hosted_policy_for(agent),
                )

        # Queue the reader while onboarding owns the writer.  It can proceed
        # only after that writer releases, exactly where the old failure path
        # exposed routing while it had merely started slow cleanup.
        reader = asyncio.create_task(observe_reader())
        await asyncio.sleep(0)
        if cancel_onboarding:
            load.cancel()
        else:
            allow_failure.set()

        try:
            await asyncio.wait_for(cleanup_started.wait(), timeout=1.0)
            observed_agent, observed_policy = await asyncio.wait_for(reader, timeout=1.0)
            assert observed_agent is None
            assert observed_policy is None
        finally:
            allow_cleanup.set()

        if cancel_onboarding:
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(load, timeout=1.0)
        else:
            with pytest.raises(RuntimeError, match="host onboarding rejected"):
                await asyncio.wait_for(load, timeout=1.0)
        agent.shutdown.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_batch_rejected_onboarding_withdraws_before_a2a_reader_or_cleanup(
        self,
    ) -> None:
        """The ordered batch registrar has the same no-reader failure boundary."""

        manager = AgentManager()
        agent = _make_mock_agent("did:test:batch-rejected-onboarding")
        config = LocalAgentConfig(data_dir="batch-rejected", port=8801)
        hook_entered = asyncio.Event()
        allow_failure = asyncio.Event()
        cleanup_started = asyncio.Event()
        allow_cleanup = asyncio.Event()

        async def reject_onboarding(_name, registered_agent) -> None:
            manager.install_a2a_hosted_policy(
                registered_agent,
                resolver=object(),
                authorizer=object(),
                router=object(),
                requester=object(),
            )
            hook_entered.set()
            await allow_failure.wait()
            raise RuntimeError("batch host onboarding rejected")

        original_cleanup = manager._discard_unpublished_initialized_agents

        async def pause_cleanup(*args, **kwargs) -> bool:
            cleanup_started.set()
            await allow_cleanup.wait()
            return await original_cleanup(*args, **kwargs)

        manager._initialize_agent = AsyncMock(return_value=agent)
        manager.set_agent_registration_hook(reject_onboarding)
        manager._discard_unpublished_initialized_agents = pause_cleanup
        batch = asyncio.create_task(
            manager.load_from_config(MultiAgentConfig(agents={"Rejected": config}))
        )
        await asyncio.wait_for(hook_entered.wait(), timeout=1.0)

        async def observe_reader():
            async with manager.a2a_execution_lease():
                return (
                    manager.get_agent("Rejected"),
                    manager.a2a_hosted_policy_for(agent),
                )

        reader = asyncio.create_task(observe_reader())
        await asyncio.sleep(0)
        allow_failure.set()
        try:
            await asyncio.wait_for(cleanup_started.wait(), timeout=1.0)
            assert await asyncio.wait_for(reader, timeout=1.0) == (None, None)
        finally:
            allow_cleanup.set()

        assert await asyncio.wait_for(batch, timeout=1.0) == 0
        assert manager.init_failures[0][0] == "Rejected"
        agent.shutdown.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_load_agent_cancellation_waiting_for_a2a_publication_cleans_unpublished_dynamic_registration(
        self,
    ):
        """A cancelled cold publication releases its invisible scheduler tenant."""

        manager = AgentManager()
        manager.set_scheduler_polling_managed_by_host(True)
        agent_id = "did:scheduler:cancelled-single-publication"
        config = LocalAgentConfig(data_dir="dynamic", port=8801)
        protocol_rolled_back = asyncio.Event()

        async def register(_name, _agent_id, _config):
            async def rollback() -> None:
                protocol_rolled_back.set()

            return rollback

        manager.set_scheduler_tenant_registration_hook(register)
        pending = await manager._begin_dynamic_scheduler_tenant_registration(
            "Dynamic", agent_id, config
        )
        assert pending is not None
        agent = _make_mock_agent(agent_id)
        agent._dynamic_scheduler_tenant_registration = pending
        initialize_entered = asyncio.Event()
        allow_initialize_return = asyncio.Event()

        async def initialize(*_args, **_kwargs):
            initialize_entered.set()
            await allow_initialize_return.wait()
            return agent

        manager._initialize_agent = initialize
        load = asyncio.create_task(manager.load_agent("Dynamic", config))
        await asyncio.wait_for(initialize_entered.wait(), timeout=1.0)
        await manager._a2a_lifecycle_lock.acquire()
        try:
            allow_initialize_return.set()
            await asyncio.sleep(0)
            assert not load.done()

            load.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(load, timeout=1.0)
        finally:
            manager._a2a_lifecycle_lock.release()

        assert manager.list_agents() == {}
        assert manager.scheduler_authority_for(agent_id) is None
        assert agent_id not in manager.scheduler_authorized_agent_ids()
        assert not manager.scheduler_lifecycle_lock(agent_id).locked()
        assert protocol_rolled_back.is_set()
        agent.shutdown.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_load_from_config_cancellation_waiting_for_a2a_publication_cleans_every_unpublished_result(
        self,
    ):
        """One blocked batch publication cannot strand any initialized sibling."""

        manager = AgentManager()
        manager.set_scheduler_polling_managed_by_host(True)
        configs = {
            "Alpha": LocalAgentConfig(data_dir="alpha", port=8801),
            "Beta": LocalAgentConfig(data_dir="beta", port=8802),
        }
        agent_ids = {
            "Alpha": "did:scheduler:cancelled-batch-alpha",
            "Beta": "did:scheduler:cancelled-batch-beta",
        }
        rolled_back: set[str] = set()

        async def register(name, _agent_id, _config):
            async def rollback() -> None:
                rolled_back.add(name)

            return rollback

        manager.set_scheduler_tenant_registration_hook(register)
        agents = {}
        for name, config in configs.items():
            pending = await manager._begin_dynamic_scheduler_tenant_registration(
                name, agent_ids[name], config
            )
            assert pending is not None
            agent = _make_mock_agent(agent_ids[name])
            agent._dynamic_scheduler_tenant_registration = pending
            agents[name] = agent

        initialized = set()
        all_initialized = asyncio.Event()
        allow_initialize_return = asyncio.Event()

        async def initialize(name, *_args, **_kwargs):
            initialized.add(name)
            if len(initialized) == len(agents):
                all_initialized.set()
            await allow_initialize_return.wait()
            return agents[name]

        manager._initialize_agent = initialize
        batch = asyncio.create_task(
            manager.load_from_config(MultiAgentConfig(agents=configs))
        )
        await asyncio.wait_for(all_initialized.wait(), timeout=1.0)
        await manager._a2a_lifecycle_lock.acquire()
        try:
            allow_initialize_return.set()
            await asyncio.sleep(0)
            assert not batch.done()

            batch.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(batch, timeout=1.0)
        finally:
            manager._a2a_lifecycle_lock.release()

        assert manager.list_agents() == {}
        assert rolled_back == {"Alpha", "Beta"}
        for name, agent in agents.items():
            assert manager.scheduler_authority_for(agent_ids[name]) is None
            assert agent_ids[name] not in manager.scheduler_authorized_agent_ids()
            assert not manager.scheduler_lifecycle_lock(agent_ids[name]).locked()
            agent.shutdown.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_load_from_config_cancellation_releases_every_batch_admission(
        self,
    ) -> None:
        """Cancellation during the first final release cannot strand its peers."""

        manager = AgentManager()
        configs = {
            "Alpha": LocalAgentConfig(data_dir="alpha", port=8801),
            "Beta": LocalAgentConfig(data_dir="beta", port=8802),
        }
        agents = {
            "Alpha": _make_mock_agent("did:test:batch-release-alpha"),
            "Beta": _make_mock_agent("did:test:batch-release-beta"),
        }
        final_publication_holds_state_lock = asyncio.Event()

        async def initialize(name, *_args, **_kwargs):
            return agents[name]

        async def hold_state_lock_after_final_publication(name, _agent) -> None:
            if name == "Beta":
                await manager._lock.acquire()
                final_publication_holds_state_lock.set()

        manager._initialize_agent = initialize
        manager.set_agent_registration_hook(hold_state_lock_after_final_publication)
        batch = asyncio.create_task(
            manager.load_from_config(MultiAgentConfig(agents=configs))
        )
        await asyncio.wait_for(final_publication_holds_state_lock.wait(), timeout=1.0)
        try:
            # The batch has committed both agents and is blocked retiring its
            # first admission. Its cancellation must join both release tasks.
            batch.cancel()
            await asyncio.sleep(0)
        finally:
            manager._lock.release()

        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(batch, timeout=1.0)

        assert manager.list_agents() == agents
        assert manager._agent_operations == {}

    @pytest.mark.asyncio
    async def test_committed_load_returns_success_after_cancellation_during_admission_release(
        self,
    ) -> None:
        """A completed load is success even when its final release sees cancel."""

        manager = AgentManager()
        agent = _make_mock_agent("did:test:committed-load")
        release_is_blocked = asyncio.Event()

        async def hold_state_lock_after_registration(_name, _agent) -> None:
            await manager._lock.acquire()
            release_is_blocked.set()

        manager._initialize_agent = AsyncMock(return_value=agent)
        manager.set_agent_registration_hook(hold_state_lock_after_registration)
        load = asyncio.create_task(
            manager.load_agent(
                "CommittedLoad", LocalAgentConfig(data_dir="load", port=8801)
            )
        )
        await asyncio.wait_for(release_is_blocked.wait(), timeout=1.0)
        try:
            load.cancel()
            await asyncio.sleep(0)
        finally:
            manager._lock.release()

        assert await asyncio.wait_for(load, timeout=1.0) is agent
        assert manager.get_agent("CommittedLoad") is agent
        assert manager._agent_operations == {}

    @pytest.mark.asyncio
    async def test_committed_create_returns_success_after_cancellation_during_admission_release(
        self, monkeypatch, tmp_path
    ) -> None:
        """A committed create preserves its persistence handoff after cancel."""

        manager = AgentManager(base_data_dir=tmp_path)
        agent = _make_mock_agent("did:test:committed-create")
        release_is_blocked = asyncio.Event()

        async def hold_state_lock_after_registration(_name, _agent) -> None:
            await manager._lock.acquire()
            release_is_blocked.set()

        manager._data_key_custody_conflict = lambda: None
        manager._initialize_agent = AsyncMock(return_value=agent)
        manager.set_agent_registration_hook(hold_state_lock_after_registration)
        monkeypatch.setattr(
            "kestrel_sovereign.inception_service.create_kestrel_identity_async",
            AsyncMock(),
        )
        create = asyncio.create_task(manager.create_agent("CommittedCreate"))
        await asyncio.wait_for(release_is_blocked.wait(), timeout=1.0)
        try:
            create.cancel()
            await asyncio.sleep(0)
        finally:
            manager._lock.release()

        assert await asyncio.wait_for(create, timeout=1.0) is agent
        assert manager.get_agent("CommittedCreate") is agent
        assert "CommittedCreate" in manager._created_configs
        assert manager._agent_operations == {}

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
    async def test_remove_agent_preserves_refund_failure_with_cancellation(self):
        """A cancellation outcome cannot erase a settled refund failure."""

        manager = AgentManager()
        child = _make_mock_agent("did:test:cancelled-refund-failure")
        manager._agents["Child"] = child
        manager._agent_names[child.agent_id] = "Child"
        budget_entry = (object(), object())
        manager._child_budgets["Child"] = budget_entry
        refund_failure = RuntimeError("refund provider unavailable")

        async def fail_refund(name: str) -> bool:
            assert name == "Child"
            assert manager._child_budgets[name] is budget_entry
            raise refund_failure

        manager._release_child_budget_cancellation_safe = fail_refund
        manager._reconcile_fully_removed_child_tracking = AsyncMock(
            return_value=True
        )

        with pytest.raises(BaseExceptionGroup) as raised:
            await manager.remove_agent("Child")

        assert str(raised.value).startswith(
            "Agent 'Child' removal had terminal budget outcomes"
        )
        assert len(raised.value.exceptions) == 2
        assert isinstance(raised.value.exceptions[0], asyncio.CancelledError)
        assert raised.value.exceptions[1] is refund_failure
        assert manager.get_agent("Child") is None
        assert manager._child_budgets["Child"] is budget_entry

    @pytest.mark.asyncio
    async def test_remove_agent_securely_deletes_only_owned_hosted_runtime_tree(
        self, tmp_path
    ):
        manager = AgentManager(base_data_dir=tmp_path)
        did = "did:pkh:hosted-offboarding"
        scope = resolve_isolated_runtime_namespace(
            manager._isolated_runtime_root,
            derive_isolated_runtime_namespace(did),
        )
        prepare_isolated_runtime_namespace(
            scope,
            did,
            relative_directories=(("feature_venvs", "feature-safe"),),
        )
        (scope.path / "feature_venvs" / "feature-safe" / "credential").write_text(
            "tenant-secret"
        )
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "keep").write_text("unrelated")
        (scope.path / "outside-link").symlink_to(outside, target_is_directory=True)
        agent = _make_mock_agent(did)
        agent.did = did
        agent.isolated_runtime_scope = scope
        manager._agents["Hosted"] = agent
        manager._agent_names[did] = "Hosted"

        assert (
            await manager.remove_agent(
                "Hosted",
                offboard_runtime=True,
            )
            is True
        )

        assert not scope.path.exists()
        assert scope.root.is_dir()
        assert (outside / "keep").read_text() == "unrelated"

    @pytest.mark.asyncio
    async def test_slow_runtime_offboarding_is_bounded_outside_lifecycle_locks(
        self, monkeypatch
    ):
        manager = AgentManager()
        hosted = _make_mock_agent("did:pkh:slow-offboard")
        other = _make_mock_agent("did:pkh:unrelated")
        manager._agents.update({"Hosted": hosted, "Other": other})
        manager._agent_names.update(
            {hosted.agent_id: "Hosted", other.agent_id: "Other"}
        )
        started = threading.Event()
        release = threading.Event()

        def slow_cleanup(_agent):
            started.set()
            release.wait(timeout=5)

        monkeypatch.setattr(
            "kestrel_sovereign.features.isolated_runtime.remove_agent_runtime_namespace",
            slow_cleanup,
        )
        monkeypatch.setattr(
            "kestrel_sovereign.multi_agent.agent_manager.RUNTIME_OFFBOARD_TIMEOUT_S",
            0.05,
        )

        deletion = asyncio.create_task(
            manager.remove_agent("Hosted", offboard_runtime=True)
        )
        try:
            assert await asyncio.to_thread(started.wait, 1)
            # The slow filesystem worker cannot retain the manager-wide or
            # A2A lifecycle locks needed by an unrelated agent removal.
            assert await asyncio.wait_for(manager.remove_agent("Other"), 0.5)
            with pytest.raises(RuntimeOffboardingRetainedError) as raised:
                await deletion
            assert raised.value.metadata["cause_type"] == "TimeoutError"
            assert raised.value.metadata["runtime_cleanup_pending"] is True
            assert raised.value.metadata["runtime_cleanup_state"] == "pending"
            assert "still pending" in str(raised.value)
            assert "was retained" not in str(raised.value)
            assert "runtime_path" not in raised.value.metadata
            assert manager.get_agent("Hosted") is None
            assert manager.quarantined_shutdowns()
        finally:
            release.set()
        await manager.drain_quarantined_shutdowns()
        completed = list(manager.quarantined_shutdowns().values())
        assert completed
        assert all(record["pending"] is False for record in completed)
        assert all(record["failure"] is None for record in completed)

    @pytest.mark.asyncio
    async def test_cancelled_runtime_offboarding_is_retained_and_does_not_wedge_peers(
        self, monkeypatch
    ):
        manager = AgentManager()
        hosted = _make_mock_agent("did:pkh:cancelled-cleanup")
        other = _make_mock_agent("did:pkh:cancelled-peer")
        manager._agents.update({"Hosted": hosted, "Other": other})
        manager._agent_names.update(
            {hosted.agent_id: "Hosted", other.agent_id: "Other"}
        )
        started = threading.Event()
        release = threading.Event()

        def slow_cleanup(_agent):
            started.set()
            release.wait(timeout=5)

        monkeypatch.setattr(
            "kestrel_sovereign.features.isolated_runtime.remove_agent_runtime_namespace",
            slow_cleanup,
        )
        monkeypatch.setattr(
            "kestrel_sovereign.multi_agent.agent_manager.RUNTIME_OFFBOARD_TIMEOUT_S",
            5.0,
        )
        deletion = asyncio.create_task(
            manager.remove_agent("Hosted", offboard_runtime=True)
        )
        try:
            assert await asyncio.to_thread(started.wait, 1)
            deletion.cancel()
            with pytest.raises(BaseExceptionGroup) as raised:
                await deletion
            assert _exception_group_contains(raised.value, asyncio.CancelledError)
            assert _exception_group_contains(
                raised.value, RuntimeOffboardingRetainedError
            )
            retained = next(
                error
                for error in _exception_leaves(raised.value)
                if isinstance(error, RuntimeOffboardingRetainedError)
            )
            assert retained.metadata["runtime_cleanup_pending"] is True
            assert retained.metadata["runtime_cleanup_state"] == "pending"
            assert manager.get_agent("Hosted") is None
            assert await asyncio.wait_for(manager.remove_agent("Other"), 0.5)
        finally:
            release.set()
        await manager.drain_quarantined_shutdowns()

    @pytest.mark.asyncio
    async def test_shutdown_all_preserves_hosted_runtime_tree(self, tmp_path):
        manager = AgentManager(base_data_dir=tmp_path)
        did = "did:pkh:hosted-restart"
        scope = resolve_isolated_runtime_namespace(
            manager._isolated_runtime_root,
            derive_isolated_runtime_namespace(did),
        )
        prepare_isolated_runtime_namespace(
            scope,
            did,
            relative_directories=(("feature_venvs", "feature-safe"),),
        )
        credential = scope.path / "feature_venvs" / "feature-safe" / "credential"
        credential.write_text("survive-restart")
        agent = _make_mock_agent(did)
        agent.did = did
        agent.isolated_runtime_scope = scope
        manager._agents["Hosted"] = agent
        manager._agent_names[did] = "Hosted"

        await manager.shutdown_all()

        assert manager.get_agent("Hosted") is None
        assert credential.read_text() == "survive-restart"
        assert (scope.path / ".kestrel-runtime-owner").is_file()

    @pytest.mark.asyncio
    async def test_delete_endpoint_defaults_to_state_preserving_stop(self):
        from kestrel_sovereign.endpoints.models import delete_agent

        manager = SimpleNamespace(remove_agent=AsyncMock(return_value=True))
        request = SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(agent_manager=manager))
        )

        result = await delete_agent.__wrapped__(request, "Hosted")

        manager.remove_agent.assert_awaited_once_with(
            "Hosted",
            offboard_runtime=False,
        )
        assert result["success"] is True
        assert result["runtime_offboarded"] is False
        assert result["runtime_retained_for_restart"] is True

    @pytest.mark.asyncio
    async def test_delete_endpoint_default_does_not_rewrite_autostart_config(
        self,
        tmp_path,
    ):
        from kestrel_sovereign.endpoints.models import delete_agent

        config_path = tmp_path / "multi_agent.toml"
        config = MultiAgentConfig(
            agents={
                "Hosted": LocalAgentConfig(
                    data_dir=Path("agent_data/hosted"),
                    port=8801,
                    autostart=True,
                )
            }
        )
        config.save(config_path)
        before = config_path.read_bytes()
        manager = SimpleNamespace(remove_agent=AsyncMock(return_value=True))
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(
                    agent_manager=manager,
                    multi_agent_config_path=config_path,
                    multi_agent_config=config,
                )
            )
        )

        result = await delete_agent.__wrapped__(request, "Hosted")

        assert config_path.read_bytes() == before
        assert "Hosted" in MultiAgentConfig.from_file(config_path).agents
        assert result["runtime_retained_for_restart"] is True

    @pytest.mark.asyncio
    async def test_delete_endpoint_explicit_offboard_removes_autostart_registration(
        self,
        tmp_path,
    ):
        from kestrel_sovereign.endpoints.models import delete_agent

        config_path = tmp_path / "multi_agent.toml"
        config = MultiAgentConfig(
            agents={
                "Hosted": LocalAgentConfig(
                    data_dir=Path("agent_data/hosted"),
                    port=8801,
                    autostart=True,
                )
            }
        )
        config.save(config_path)
        manager = SimpleNamespace(
            resolve_registered_agent_id=AsyncMock(return_value="did:test:hosted"),
            remove_agent=AsyncMock(return_value=True),
        )
        state = SimpleNamespace(
            agent_manager=manager,
            multi_agent_config_path=config_path,
            multi_agent_config=config,
        )
        request = SimpleNamespace(app=SimpleNamespace(state=state))

        result = await delete_agent.__wrapped__(
            request,
            "Hosted",
            offboard_runtime=True,
        )

        manager.remove_agent.assert_awaited_once_with(
            "Hosted",
            offboard_runtime=True,
            known_agent_id="did:test:hosted",
        )
        assert MultiAgentConfig.from_file(config_path).agents == {}
        assert state.multi_agent_config.agents == {}
        assert result["runtime_offboarded"] is True
        assert result["persisted_registration_removed"] is True

    @pytest.mark.asyncio
    async def test_delete_endpoint_refuses_destructive_auto_discovery_deployment(self):
        from kestrel_sovereign.endpoints.models import delete_agent

        manager = SimpleNamespace(remove_agent=AsyncMock(return_value=True))
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(
                    agent_manager=manager,
                    multi_agent_config_path=None,
                )
            )
        )

        with pytest.raises(HTTPException) as raised:
            await delete_agent.__wrapped__(
                request,
                "Hosted",
                offboard_runtime=True,
            )

        assert raised.value.status_code == 409
        assert "auto-discovered" in raised.value.detail
        manager.remove_agent.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_delete_endpoint_retains_config_removal_while_cleanup_is_pending(
        self,
        tmp_path,
    ):
        from kestrel_sovereign.endpoints.models import delete_agent

        config_path = tmp_path / "multi_agent.toml"
        config = MultiAgentConfig(
            agents={
                "Hosted": LocalAgentConfig(
                    data_dir=Path("agent_data/hosted"),
                    port=8801,
                )
            }
        )
        config.save(config_path)
        pending = RuntimeOffboardingRetainedError(
            agent_name="Hosted",
            agent_id="did:test:pending",
            runtime_path=tmp_path / "private-runtime",
            cause=TimeoutError("private shutdown handoff"),
            cleanup_pending=True,
        )
        manager = SimpleNamespace(
            resolve_registered_agent_id=AsyncMock(return_value="did:test:pending"),
            remove_agent=AsyncMock(side_effect=pending),
        )
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(
                    agent_manager=manager,
                    multi_agent_config_path=config_path,
                    multi_agent_config=config,
                )
            )
        )

        with pytest.raises(HTTPException) as raised:
            await delete_agent.__wrapped__(
                request,
                "Hosted",
                offboard_runtime=True,
            )

        assert raised.value.status_code == 409
        assert raised.value.detail["runtime_cleanup_state"] == "pending"
        assert MultiAgentConfig.from_file(config_path).agents == {}
        assert str(tmp_path) not in str(raised.value.detail)

    @pytest.mark.asyncio
    async def test_delete_endpoint_restores_registration_when_removal_is_refused(
        self,
        tmp_path,
    ):
        from kestrel_sovereign.endpoints.models import delete_agent

        config_path = tmp_path / "multi_agent.toml"
        original = LocalAgentConfig(
            data_dir=Path("agent_data/hosted"),
            port=8801,
            autostart=True,
        )
        config = MultiAgentConfig(agents={"Hosted": original})
        config.save(config_path)
        manager = SimpleNamespace(
            resolve_registered_agent_id=AsyncMock(return_value="did:test:hosted"),
            remove_agent=AsyncMock(return_value=False),
        )
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(
                    agent_manager=manager,
                    multi_agent_config_path=config_path,
                    multi_agent_config=config,
                )
            )
        )

        with pytest.raises(HTTPException) as raised:
            await delete_agent.__wrapped__(
                request,
                "Hosted",
                offboard_runtime=True,
            )

        assert raised.value.status_code == 404
        restored = MultiAgentConfig.from_file(config_path)
        assert restored.agents["Hosted"] == original

    @pytest.mark.asyncio
    async def test_delete_endpoint_reports_unpublished_agent_with_retained_runtime(
        self, tmp_path
    ):
        from kestrel_sovereign.endpoints.models import delete_agent

        retained = RuntimeOffboardingRetainedError(
            agent_name="Hosted",
            agent_id="did:pkh:retained",
            runtime_path=tmp_path / "runtime" / "tenant",
            cause=IsolatedRuntimeNamespaceError("foreign owner"),
        )
        manager = SimpleNamespace(remove_agent=AsyncMock(side_effect=retained))
        request = SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(agent_manager=manager))
        )

        with pytest.raises(HTTPException) as raised:
            await delete_agent.__wrapped__(request, "Hosted")

        assert raised.value.status_code == 409
        assert raised.value.detail == retained.metadata
        assert raised.value.detail["agent_removed"] is True
        assert raised.value.detail["runtime_retained"] is True
        assert "runtime_path" not in raised.value.detail
        assert str(tmp_path) not in str(raised.value.detail)

    @pytest.mark.asyncio
    async def test_delete_endpoint_extracts_retained_runtime_from_nested_group(
        self, tmp_path
    ):
        from kestrel_sovereign.endpoints.models import delete_agent

        class _PrivateRefundFailure(RuntimeError):
            pass

        retained = RuntimeOffboardingRetainedError(
            agent_name="Hosted",
            agent_id="did:pkh:compound-retained",
            runtime_path=tmp_path / "runtime" / "tenant",
            cause=IsolatedRuntimeNamespaceError("foreign owner"),
        )
        refund_failure = _PrivateRefundFailure(
            f"refund failed with secret at {tmp_path / 'wallet.json'}"
        )
        terminal = BaseExceptionGroup(
            "removal outcomes",
            [ExceptionGroup("cleanup", [retained]), refund_failure],
        )
        manager = SimpleNamespace(remove_agent=AsyncMock(side_effect=terminal))
        request = SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(agent_manager=manager))
        )

        with pytest.raises(HTTPException) as raised:
            await delete_agent.__wrapped__(request, "Hosted")

        assert raised.value.status_code == 409
        detail = raised.value.detail
        assert detail["code"] == "runtime_offboarding_retained"
        assert detail["agent_removed"] is True
        assert detail["runtime_retained"] is True
        assert detail["runtime_cleanup_pending"] is False
        assert detail["runtime_cleanup_state"] == "retained"
        assert detail["retained_agents"] == ["Hosted"]
        assert detail["compound_outcome"] is True
        assert detail["retained_outcome_count"] == 1
        assert detail["additional_outcome_count"] == 1
        assert detail["additional_outcome_types"] == ["RuntimeError"]
        assert "runtime_path" not in detail
        assert "wallet.json" not in str(detail)
        assert str(tmp_path) not in str(detail)

    @pytest.mark.asyncio
    async def test_delete_endpoint_reports_cleanup_timeout_as_pending_not_permanent(
        self, tmp_path
    ):
        from kestrel_sovereign.endpoints.models import delete_agent

        pending = RuntimeOffboardingRetainedError(
            agent_name="Hosted",
            agent_id="did:pkh:pending-cleanup",
            runtime_path=tmp_path / "runtime" / "tenant",
            cause=TimeoutError("private timeout detail"),
            cleanup_pending=True,
        )
        manager = SimpleNamespace(remove_agent=AsyncMock(side_effect=pending))
        request = SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(agent_manager=manager))
        )

        with pytest.raises(HTTPException) as raised:
            await delete_agent.__wrapped__(request, "Hosted")

        assert raised.value.status_code == 409
        assert raised.value.detail["runtime_cleanup_pending"] is True
        assert raised.value.detail["runtime_cleanup_state"] == "pending"
        assert "timeout detail" not in str(raised.value.detail)
        assert str(tmp_path) not in str(raised.value.detail)

    def test_retained_metadata_hides_private_internal_exception_type(self, tmp_path):
        class _PrivateOwnerMarkerMissing(IsolatedRuntimeNamespaceError):
            pass

        retained = RuntimeOffboardingRetainedError(
            agent_name="Hosted",
            agent_id="did:pkh:private-cause",
            runtime_path=tmp_path / "runtime" / "tenant",
            cause=_PrivateOwnerMarkerMissing("missing"),
        )

        assert retained.metadata["cause_type"] == "IsolatedRuntimeNamespaceError"

    @pytest.mark.asyncio
    async def test_remove_agent_refuses_foreign_runtime_owner_without_deleting(
        self, tmp_path
    ):
        manager = AgentManager(base_data_dir=tmp_path)
        did = "did:pkh:requested-offboarding"
        scope = resolve_isolated_runtime_namespace(
            manager._isolated_runtime_root,
            derive_isolated_runtime_namespace(did),
        )
        prepare_isolated_runtime_namespace(scope, "did:pkh:different-owner")
        retained = scope.path / "credential"
        retained.write_text("must-remain")
        agent = _make_mock_agent(did)
        agent.did = did
        agent.isolated_runtime_scope = scope
        manager._agents["Hosted"] = agent
        manager._agent_names[did] = "Hosted"
        config = LocalAgentConfig(data_dir="managed", port=8801)
        manager._seed_scheduler_authority({did: ("Hosted", config)})

        with pytest.raises(RuntimeOffboardingRetainedError) as raised:
            await manager.remove_agent(
                "Hosted",
                offboard_runtime=True,
            )

        assert raised.value.metadata["runtime_retained"] is True
        assert raised.value.metadata["runtime_cleanup_pending"] is False
        assert raised.value.metadata["runtime_cleanup_state"] == "retained"
        assert raised.value.runtime_path == scope.path
        assert manager.get_agent("Hosted") is None
        assert manager.get_agent_name(did) is None
        assert not manager.is_scheduler_agent_authorized(did)
        assert retained.read_text() == "must-remain"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("scheduler_authorized", (False, True))
    async def test_case_variant_remove_shuts_down_exact_live_agent_before_cleanup(
        self,
        monkeypatch,
        scheduler_authorized,
        tmp_path,
    ):
        from kestrel_sovereign.features import isolated_runtime

        manager = AgentManager(base_data_dir=tmp_path)
        did = "did:pkh:case-variant-offboarding"
        scope = resolve_isolated_runtime_namespace(
            manager._isolated_runtime_root,
            derive_isolated_runtime_namespace(did),
        )
        prepare_isolated_runtime_namespace(
            scope,
            did,
            relative_directories=(("feature_venvs", "feature-safe"),),
        )
        credential = scope.path / "feature_venvs" / "feature-safe" / "credential"
        credential.write_text("case-sensitive-custody")
        agent = _make_mock_agent(did)
        agent.did = did
        agent.isolated_runtime_scope = scope
        manager._agents["Hosted"] = agent
        manager._agent_names[did] = "Hosted"
        if scheduler_authorized:
            manager._seed_scheduler_authority(
                {
                    did: (
                        "Hosted",
                        LocalAgentConfig(data_dir="managed", port=8801),
                    )
                }
            )
        real_remove = isolated_runtime.remove_agent_runtime_namespace
        cleanup_calls = 0

        def assert_shutdown_precedes_cleanup(candidate):
            nonlocal cleanup_calls
            cleanup_calls += 1
            assert candidate is agent
            assert agent.shutdown.await_count == 1
            return real_remove(candidate)

        monkeypatch.setattr(
            isolated_runtime,
            "remove_agent_runtime_namespace",
            assert_shutdown_precedes_cleanup,
        )

        assert await manager.remove_agent(
            "hosted",
            offboard_runtime=True,
        )

        agent.shutdown.assert_awaited_once_with()
        assert cleanup_calls == 1
        assert manager.get_agent("Hosted") is None
        assert manager.get_agent("hosted") is None
        assert manager.get_agent_name(did) is None
        assert "Hosted" not in manager._agents
        assert not scope.path.exists()
        assert not manager.is_scheduler_agent_authorized(did)

    @pytest.mark.asyncio
    async def test_delete_case_edited_registration_cannot_bypass_live_shutdown(
        self,
        tmp_path,
    ):
        from kestrel_sovereign.endpoints.models import delete_agent

        manager = AgentManager(base_data_dir=tmp_path)
        did = "did:pkh:case-edited-registration"
        scope = resolve_isolated_runtime_namespace(
            manager._isolated_runtime_root,
            derive_isolated_runtime_namespace(did),
        )
        prepare_isolated_runtime_namespace(scope, did)
        agent = _make_mock_agent(did)
        agent.did = did
        agent.isolated_runtime_scope = scope
        manager._agents["Hosted"] = agent
        manager._agent_names[did] = "Hosted"
        config_path = tmp_path / "multi_agent.toml"
        config = MultiAgentConfig(
            agents={
                "hosted": LocalAgentConfig(
                    data_dir="agent_data/hosted",
                    port=8801,
                )
            }
        )
        config.save(config_path)
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(
                    agent_manager=manager,
                    multi_agent_config_path=config_path,
                    multi_agent_config=config,
                )
            )
        )

        response = await delete_agent.__wrapped__(
            request,
            "HOSTED",
            offboard_runtime=True,
        )

        assert response["success"] is True
        agent.shutdown.assert_awaited_once_with()
        assert manager.get_agent("Hosted") is None
        assert manager.get_agent("hosted") is None
        assert not scope.path.exists()
        assert MultiAgentConfig.from_file(config_path).agents == {}
        assert request.app.state.multi_agent_config.agents == {}

    @pytest.mark.asyncio
    async def test_explicit_offboard_removes_authorized_cold_agent_runtime(
        self,
        tmp_path,
    ):
        manager = AgentManager(base_data_dir=tmp_path)
        did = "did:pkh:cold-offboarding"
        scope = resolve_isolated_runtime_namespace(
            manager._isolated_runtime_root,
            derive_isolated_runtime_namespace(did),
        )
        prepare_isolated_runtime_namespace(
            scope,
            did,
            relative_directories=(("feature_venvs", "feature-safe"),),
        )
        (scope.path / "feature_venvs" / "feature-safe" / "credential").write_text(
            "cold-credential"
        )
        manager._seed_scheduler_authority(
            {
                did: (
                    "Cold",
                    LocalAgentConfig(
                        data_dir="agent_data/cold",
                        port=8801,
                        autostart=False,
                    ),
                )
            }
        )

        assert await manager.remove_agent("Cold", offboard_runtime=True) is True

        assert not scope.path.exists()
        assert not manager.is_scheduler_agent_authorized(did)

    @pytest.mark.asyncio
    async def test_explicit_offboard_removes_witnessed_registered_cold_runtime(
        self,
        monkeypatch,
        tmp_path,
    ):
        """A known stopped registration is not a false 404 or duplicate sweep."""

        from kestrel_sovereign.features import isolated_runtime

        manager = AgentManager(base_data_dir=tmp_path)
        did = "did:pkh:registered-cold-offboarding"
        scope = resolve_isolated_runtime_namespace(
            manager._isolated_runtime_root,
            derive_isolated_runtime_namespace(did),
        )
        prepare_isolated_runtime_namespace(
            scope,
            did,
            relative_directories=(("feature_venvs", "feature-safe"),),
        )
        credential = scope.path / "feature_venvs" / "feature-safe" / "credential"
        credential.write_text("registered-cold-credential")
        real_remove = isolated_runtime.remove_isolated_runtime_namespace
        cleanup_calls = 0

        def count_cleanup(*args, **kwargs):
            nonlocal cleanup_calls
            cleanup_calls += 1
            return real_remove(*args, **kwargs)

        monkeypatch.setattr(
            isolated_runtime,
            "remove_isolated_runtime_namespace",
            count_cleanup,
        )

        assert await manager.remove_agent(
            "Cold",
            offboard_runtime=True,
            known_agent_id=did,
        )

        assert cleanup_calls == 1
        assert not scope.path.exists()
        assert manager.get_agent("Cold") is None
        assert not manager.is_scheduler_agent_authorized(did)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("cleanup_outcome", ("success", "failure", "pending"))
    async def test_delete_endpoint_offboards_registered_unloaded_agent_truthfully(
        self,
        cleanup_outcome,
        monkeypatch,
        tmp_path,
    ):
        from kestrel_sovereign.endpoints.models import delete_agent
        from kestrel_sovereign.features import isolated_runtime

        manager = AgentManager(base_data_dir=tmp_path)
        did = f"did:pkh:cold-endpoint-{cleanup_outcome}"
        scope = resolve_isolated_runtime_namespace(
            manager._isolated_runtime_root,
            derive_isolated_runtime_namespace(did),
        )
        prepare_isolated_runtime_namespace(
            scope,
            did,
            relative_directories=(("feature_venvs", "feature-safe"),),
        )
        credential = scope.path / "feature_venvs" / "feature-safe" / "credential"
        credential.write_text(f"cold-{cleanup_outcome}-credential")
        config_path = tmp_path / f"multi-agent-{cleanup_outcome}.toml"
        local = LocalAgentConfig(
            data_dir=Path("agent_data/cold"),
            port=8801,
            autostart=False,
        )
        config = MultiAgentConfig(agents={"Cold": local})
        config.save(config_path)
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(
                    agent_manager=manager,
                    multi_agent_config_path=config_path,
                    multi_agent_config=config,
                )
            )
        )
        anchor = AsyncMock(return_value=did)
        monkeypatch.setattr(
            "kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did",
            anchor,
        )
        real_remove = isolated_runtime.remove_isolated_runtime_namespace
        cleanup_calls = 0
        cleanup_started = threading.Event()
        cleanup_release = threading.Event()

        def controlled_cleanup(*args, **kwargs):
            nonlocal cleanup_calls
            cleanup_calls += 1
            if cleanup_outcome == "failure":
                raise PermissionError("private operator runtime path")
            if cleanup_outcome == "pending":
                cleanup_started.set()
                cleanup_release.wait(timeout=5)
            return real_remove(*args, **kwargs)

        monkeypatch.setattr(
            isolated_runtime,
            "remove_isolated_runtime_namespace",
            controlled_cleanup,
        )
        if cleanup_outcome == "pending":
            monkeypatch.setattr(
                "kestrel_sovereign.multi_agent.agent_manager.RUNTIME_OFFBOARD_TIMEOUT_S",
                0.01,
            )

        try:
            if cleanup_outcome == "success":
                result = await delete_agent.__wrapped__(
                    request,
                    "Cold",
                    offboard_runtime=True,
                )
                assert result["runtime_offboarded"] is True
                assert result["persisted_registration_removed"] is True
                assert not scope.path.exists()
            else:
                with pytest.raises(HTTPException) as raised:
                    await delete_agent.__wrapped__(
                        request,
                        "Cold",
                        offboard_runtime=True,
                    )
                assert raised.value.status_code == 409
                expected_state = (
                    "pending" if cleanup_outcome == "pending" else "retained"
                )
                assert raised.value.detail["runtime_cleanup_state"] == expected_state
                assert str(tmp_path) not in str(raised.value.detail)
                assert credential.read_text() == f"cold-{cleanup_outcome}-credential"

            assert cleanup_calls == 1
            assert MultiAgentConfig.from_file(config_path).agents == {}
            assert request.app.state.multi_agent_config.agents == {}
            assert manager.get_agent("Cold") is None
            anchor.assert_awaited_once_with(
                str(local.resolve_data_dir(tmp_path)),
                mode=AgentDIDLookupMode.COLD_READ_ONLY,
            )
        finally:
            cleanup_release.set()
            if cleanup_outcome == "pending":
                await manager.drain_quarantined_shutdowns()

    @pytest.mark.asyncio
    async def test_delete_endpoint_refuses_unresolved_registered_identity_before_mutation(
        self,
        monkeypatch,
        tmp_path,
    ):
        from kestrel_sovereign.endpoints.models import delete_agent

        manager = AgentManager(base_data_dir=tmp_path)
        config_path = tmp_path / "multi_agent.toml"
        local = LocalAgentConfig(
            data_dir=Path("agent_data/unresolved"),
            port=8801,
            autostart=False,
        )
        config = MultiAgentConfig(agents={"Unresolved": local})
        config.save(config_path)
        before = config_path.read_bytes()
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(
                    agent_manager=manager,
                    multi_agent_config_path=config_path,
                    multi_agent_config=config,
                )
            )
        )
        monkeypatch.setattr(
            "kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did",
            AsyncMock(side_effect=OSError("private anchor path")),
        )

        with pytest.raises(HTTPException) as raised:
            await delete_agent.__wrapped__(
                request,
                "Unresolved",
                offboard_runtime=True,
            )

        assert raised.value.status_code == 409
        assert "identity is unavailable" in raised.value.detail
        assert config_path.read_bytes() == before
        assert "Unresolved" in request.app.state.multi_agent_config.agents
        assert manager._inflight_runtime_offboardings == {}

    @pytest.mark.asyncio
    async def test_offboarding_failure_and_cancellation_never_republish_dead_agent(
        self,
    ):
        manager = AgentManager()
        agent = _make_mock_agent("did:pkh:cancelled-offboarding")
        manager._agents["Hosted"] = agent
        manager._agent_names[agent.agent_id] = "Hosted"
        config = LocalAgentConfig(data_dir="managed", port=8801)
        manager._seed_scheduler_authority({agent.agent_id: ("Hosted", config)})
        retained = RuntimeOffboardingRetainedError(
            agent_name="Hosted",
            agent_id=agent.agent_id,
            runtime_path=Path("operator/runtime/tenant"),
            cause=IsolatedRuntimeNamespaceError("foreign owner"),
        )
        manager._finish_agent_runtime_offboarding = AsyncMock(
            return_value=(True, retained)
        )

        with pytest.raises(BaseExceptionGroup) as raised:
            await manager.remove_agent("Hosted", offboard_runtime=True)

        assert any(
            isinstance(error, asyncio.CancelledError)
            for error in raised.value.exceptions
        )
        assert any(
            isinstance(error, RuntimeOffboardingRetainedError)
            for error in raised.value.exceptions
        )
        assert manager.get_agent("Hosted") is None
        assert manager.get_agent_name(agent.agent_id) is None
        assert not manager.is_scheduler_agent_authorized(agent.agent_id)
        agent.shutdown.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_offboarding_and_budget_failures_are_both_reported_after_unpublish(
        self,
    ):
        manager = AgentManager()
        agent = _make_mock_agent("did:pkh:offboard-and-refund")
        manager._agents["Hosted"] = agent
        manager._agent_names[agent.agent_id] = "Hosted"
        budget_entry = (object(), object())
        manager._child_budgets["Hosted"] = budget_entry
        retained = RuntimeOffboardingRetainedError(
            agent_name="Hosted",
            agent_id=agent.agent_id,
            runtime_path=Path("operator/runtime/tenant"),
            cause=IsolatedRuntimeNamespaceError("foreign owner"),
        )
        manager._finish_agent_runtime_offboarding = AsyncMock(
            return_value=(False, retained)
        )

        async def fail_refund(name: str) -> bool:
            assert name == "Hosted"
            assert manager._child_budgets[name] is budget_entry
            raise RuntimeError("refund failed")

        manager._release_child_budget_cancellation_safe = fail_refund

        with pytest.raises(ExceptionGroup) as raised:
            await manager.remove_agent("Hosted", offboard_runtime=True)

        assert any(
            isinstance(error, RuntimeOffboardingRetainedError)
            for error in raised.value.exceptions
        )
        assert any("refund failed" in str(error) for error in raised.value.exceptions)
        assert manager.get_agent("Hosted") is None
        assert manager.get_agent_name(agent.agent_id) is None
        assert manager._child_budgets["Hosted"] is budget_entry

    @pytest.mark.asyncio
    async def test_terminate_child_prunes_tracking_when_runtime_is_retained(self):
        manager = AgentManager()
        parent_did = "did:pkh:retained-parent"
        child = _make_mock_agent("did:pkh:retained-child")
        manager._agents["Child"] = child
        manager._agent_names[child.agent_id] = "Child"
        manager._parent_children[parent_did] = ["Child"]
        mandate = SpawnMandate(parent_did=parent_did, purpose="offboard")
        manager._child_mandates["Child"] = mandate
        retained = RuntimeOffboardingRetainedError(
            agent_name="Child",
            agent_id=child.agent_id,
            runtime_path=Path("operator/runtime/tenant"),
            cause=IsolatedRuntimeNamespaceError("foreign owner"),
        )
        manager._finish_agent_runtime_offboarding = AsyncMock(
            return_value=(False, retained)
        )

        with pytest.raises(RuntimeOffboardingRetainedError):
            await manager.terminate_child(
                parent_did,
                "Child",
                offboard_runtime=True,
            )

        assert manager.get_agent("Child") is None
        assert manager.get_children(parent_did) == []
        assert manager.get_mandate("Child") is None

    @pytest.mark.asyncio
    async def test_terminate_children_continues_after_retained_and_grouped_outcomes(
        self,
    ):
        manager = AgentManager()
        parent_did = "did:pkh:cascade-parent"
        manager._parent_children[parent_did] = ["First", "Second", "Third"]
        for name in ("First", "Second", "Third"):
            manager._child_mandates[name] = SpawnMandate(
                parent_did=parent_did, purpose="cascade"
            )
        retained = RuntimeOffboardingRetainedError(
            agent_name="First",
            agent_id="did:pkh:first",
            runtime_path=Path("operator/runtime/first"),
            cause=OSError("retained"),
        )
        grouped = BaseExceptionGroup(
            "cancelled retained cleanup",
            [
                asyncio.CancelledError(),
                RuntimeOffboardingRetainedError(
                    agent_name="Second",
                    agent_id="did:pkh:second",
                    runtime_path=Path("operator/runtime/second"),
                    cause=OSError("retained"),
                ),
            ],
        )
        manager.remove_agent = AsyncMock(side_effect=[retained, grouped, True])

        with pytest.raises(BaseExceptionGroup) as raised:
            await manager.terminate_children(
                parent_did,
                offboard_runtime=True,
            )

        assert manager.remove_agent.await_args_list == [
            (("First",), {"offboard_runtime": True}),
            (("Second",), {"offboard_runtime": True}),
            (("Third",), {"offboard_runtime": True}),
        ]
        assert _exception_group_contains(
            raised.value, RuntimeOffboardingRetainedError
        )
        assert _exception_group_contains(raised.value, asyncio.CancelledError)
        assert manager.get_children(parent_did) == []

    @pytest.mark.asyncio
    async def test_terminate_child_still_removes_parent_after_descendant_retention(
        self,
    ):
        manager = AgentManager()
        parent_did = "did:pkh:root-parent"
        child = _make_mock_agent("did:pkh:cascade-child")
        manager._agents["Child"] = child
        manager._agent_names[child.agent_id] = "Child"
        manager._parent_children[parent_did] = ["Child"]
        manager._child_mandates["Child"] = SpawnMandate(
            parent_did=parent_did, purpose="cascade"
        )
        descendant_retained = RuntimeOffboardingRetainedError(
            agent_name="Grandchild",
            agent_id="did:pkh:grandchild",
            runtime_path=Path("operator/runtime/grandchild"),
            cause=OSError("retained"),
        )
        manager.terminate_children = AsyncMock(side_effect=descendant_retained)

        async def remove_child(name: str, *, offboard_runtime: bool) -> bool:
            assert name == "Child"
            assert offboard_runtime is True
            manager._agents.pop(name)
            manager._agent_names.pop(child.agent_id)
            return True

        manager.remove_agent = AsyncMock(side_effect=remove_child)

        with pytest.raises(RuntimeOffboardingRetainedError):
            await manager.terminate_child(
                parent_did,
                "Child",
                offboard_runtime=True,
            )

        manager.remove_agent.assert_awaited_once_with(
            "Child", offboard_runtime=True
        )
        assert manager.get_children(parent_did) == []

    @pytest.mark.asyncio
    async def test_terminate_child_types_post_removal_reconciliation_failure(self):
        manager = AgentManager()
        parent_did = "did:pkh:reconcile-parent"
        manager._parent_children[parent_did] = ["Child"]
        manager.remove_agent = AsyncMock(return_value=True)
        cause = OSError("private reconciliation path /operator/runtime")
        manager._prune_child_tracking_if_fully_removed = AsyncMock(
            side_effect=cause
        )

        with pytest.raises(ChildTerminationReconciliationError) as raised:
            await manager.terminate_child(
                parent_did,
                "Child",
                offboard_runtime=True,
            )

        assert raised.value.cause is cause
        assert raised.value.metadata == {
            "code": "child_termination_reconciliation_failed",
            "child_name": "Child",
            "cause_type": "OSError",
        }
        assert "/operator/runtime" not in str(raised.value)
        manager.remove_agent.assert_awaited_once_with(
            "Child", offboard_runtime=True
        )

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
    async def test_stale_delete_never_removes_same_name_replacement(self):
        """A DELETE holding DID A's lock cannot unpublish replacement DID B."""

        manager = AgentManager()
        old_config = LocalAgentConfig(data_dir="old", port=8801)
        new_config = LocalAgentConfig(data_dir="new", port=8802)
        old = _make_mock_agent("did:pkh:old-delete-target")
        replacement = _make_mock_agent("did:pkh:same-name-replacement")
        manager._agents["Managed"] = old
        manager._agent_names[old.agent_id] = "Managed"
        manager._seed_scheduler_authority(
            {old.agent_id: ("Managed", old_config)}
        )

        lifecycle_lock = manager.scheduler_lifecycle_lock(old.agent_id)
        await lifecycle_lock.acquire_read()
        deletion = asyncio.create_task(manager.remove_agent("Managed"))
        try:
            while lifecycle_lock._waiting_writers == 0:
                await asyncio.sleep(0)

            async with manager._a2a_lifecycle_lock:
                manager._agents["Managed"] = replacement
                manager._agent_names.pop(old.agent_id)
                manager._agent_names[replacement.agent_id] = "Managed"
                manager._revoke_scheduler_authority("Managed", old.agent_id)
                manager._scheduler_revoked_names.discard("Managed")
                manager._scheduler_authority_by_did[replacement.agent_id] = (
                    "Managed",
                    new_config,
                )
                manager._scheduler_authority_by_name["Managed"] = replacement.agent_id
                manager._scheduler_execution_scope.add(replacement.agent_id)
        finally:
            lifecycle_lock.release_read()

        assert await asyncio.wait_for(deletion, timeout=1) is False
        assert manager.get_agent("Managed") is replacement
        replacement.shutdown.assert_not_awaited()
        assert manager.is_scheduler_agent_authorized(replacement.agent_id)

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
            # Mirrors the real constructor envelope: each agent's service is
            # bound to that agent's own data root, so per-agent usage rows
            # cannot collapse into one shared database (#2769).
            def __init__(self, database_url=None, agent_data_dir=None):
                self.agent_data_dir = agent_data_dir

            async def close(self):
                return None

        monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
        monkeypatch.setenv(
            "KESTREL_DATABASE_URL",
            "postgresql://scheduler-cold-test",
        )
        monkeypatch.setattr(
            "kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did",
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
        import kestrel_sovereign.endpoints.agent as agent_endpoint
        from kestrel_sovereign import server
        from kestrel_sovereign.a2a.did_registry import install_a2a_did_resolver
        from kestrel_sovereign.a2a.envelope_signing import (
            bound_envelope_fields,
            canonical_message,
            sign_envelope,
            verify_inbound_envelope,
        )
        from kestrel_sovereign.a2a.inbound_authorization import (
            has_a2a_inbound_scoped_policy,
            install_a2a_inbound_sender_authorizer,
        )
        from kestrel_sovereign.a2a.types import Message, TaskSendParams, TextPart
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
    async def test_remove_agent_quarantines_deferred_durable_close_before_unpublishing(
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
            # shutdown. It withdraws the routing entry within that advertised
            # bound, while a retained quarantine reaper remains the sole owner
            # of the dispatcher release and SQLite close.
            with patch(
                "kestrel_sovereign.multi_agent.agent_manager.SHUTDOWN_TIMEOUT",
                0.05,
            ):
                remove_task = asyncio.create_task(manager.remove_agent("Managed"))
                await asyncio.wait_for(release_entered.wait(), timeout=1.0)
                assert await asyncio.wait_for(remove_task, timeout=1.0) is True

                assert manager.get_agent("Managed") is None
                assert storage_closed == []
                quarantined = manager.quarantined_shutdowns()
                assert len(quarantined) == 1
                assert next(iter(quarantined.values()))["pending"] is True

                allow_release.set()
                # A reaper owns a real SQLite worker, whose completion is not
                # ordered by a count of event-loop turns. Join the manager's
                # explicit terminal lifecycle boundary instead of letting test
                # teardown close that worker's backend underneath its release.
                assert await asyncio.wait_for(
                    manager.drain_quarantined_shutdowns(), timeout=1.0
                ) is False

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
            # Keep the production-shaped owner alive even when an assertion
            # above fails.  Closing the backend first would manufacture the
            # very post-close owner-release race this regression covers.
            if manager._quarantined_shutdown_reapers:
                await asyncio.wait_for(
                    manager.drain_quarantined_shutdowns(), timeout=1.0
                )
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
        "kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did",
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
            mode=AgentDIDLookupMode.INITIALIZATION,
        )
        partial.shutdown.assert_awaited_once()

    @pytest.mark.asyncio
    @patch(
        "kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did",
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
        "kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did",
        new_callable=AsyncMock,
    )
    @patch("kestrel_sovereign.multi_agent.agent_manager.KestrelAgent")
    @patch("kestrel_sovereign.multi_agent.agent_manager.LLMService")
    async def test_hosted_telegram_resolver_is_bound_before_agent_initialize(
        self,
        mock_llm_cls,
        mock_agent_cls,
        mock_get_did,
        tmp_path,
    ):
        """A host resolver is selected by local agent scope, never by input."""

        mock_get_did.return_value = "did:telegram:one"
        mock_agent_cls.return_value = _make_mock_agent("did:telegram:one")
        resolver = object()
        resolver_factory = MagicMock(return_value=resolver)
        manager = AgentManager(
            base_data_dir=tmp_path,
            hosted_telegram_route_attestation_resolver_factory=resolver_factory,
        )
        config = LocalAgentConfig(data_dir=Path("agent_data/telegram"), port=8801)

        with patch.object(LocalAgentConfig, "validate_runtime", return_value=[]):
            await manager._initialize_agent("telegram", config)

        resolver_factory.assert_called_once_with(
            "telegram", "did:telegram:one", config
        )
        assert (
            mock_agent_cls.call_args.kwargs[
                "hosted_telegram_route_attestation_resolver"
            ]
            is resolver
        )
        mock_agent_cls.return_value.initialize.assert_awaited_once()

    @pytest.mark.asyncio
    @patch(
        "kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did",
        new_callable=AsyncMock,
    )
    @patch("kestrel_sovereign.multi_agent.agent_manager.KestrelAgent")
    @patch("kestrel_sovereign.multi_agent.agent_manager.LLMService")
    async def test_each_agent_llm_service_is_bound_to_its_own_data_dir(
        self,
        mock_llm_cls,
        mock_agent_cls,
        mock_get_did,
        tmp_path,
        monkeypatch,
    ):
        """Per-agent usage rows must follow the agent, not the process env.

        Asserts the resolved directory itself, not merely that the keyword was
        passed: ``LLMService(agent_data_dir=None)`` would keep the keyword while
        silently falling back to ``KESTREL_DB_PATH`` and reinstating #2769.
        """
        monkeypatch.setenv("KESTREL_DB_PATH", str(tmp_path / "agent_data" / "claw"))
        mock_agent_cls.side_effect = lambda **kw: _make_mock_agent(kw["did"])

        manager = AgentManager(base_data_dir=tmp_path)
        seen = {}
        with patch.object(LocalAgentConfig, "validate_runtime", return_value=[]):
            for name in ("claw", "meridian"):
                mock_get_did.return_value = f"did:{name}"
                await manager._initialize_agent(
                    name,
                    LocalAgentConfig(data_dir=Path("agent_data") / name, port=8801),
                )
                seen[name] = mock_llm_cls.call_args.kwargs["agent_data_dir"]

        for name, passed in seen.items():
            assert passed == (tmp_path / "agent_data" / name).resolve()

        # Two agents in one process must not collapse onto one usage DB — the
        # exact condition that produced the live misattribution.
        assert seen["claw"] != seen["meridian"]

    @pytest.mark.asyncio
    @patch(
        "kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did",
        new_callable=AsyncMock,
    )
    @patch("kestrel_sovereign.multi_agent.agent_manager.KestrelAgent")
    @patch("kestrel_sovereign.multi_agent.agent_manager.LLMService")
    async def test_in_process_agent_receives_semantic_inference_profile(
        self,
        mock_llm_cls,
        mock_agent_cls,
        mock_get_did,
        tmp_path,
    ):
        """The managed config, not a sidecar TOML, controls nightly closure."""
        mock_get_did.return_value = "did:claw"
        mock_agent_cls.return_value = _make_mock_agent("did:claw")
        manager = AgentManager(base_data_dir=tmp_path)
        config = LocalAgentConfig(
            data_dir=Path("agent_data/claw"),
            port=8801,
            semantic_inference={
                "enabled": True,
                "rdfs_version": "1.0.0",
                "ontology": {
                    "namespace": "http://www.w3.org/2000/01/rdf-schema#",
                    "version": "1.0.0",
                    "content_digest": "e362812917fddab7cfab3dc35553ad292725e8f264e05f376077340e91034db5",
                    "compatibility_profile": "semantic-kb-v1",
                },
                "limits": {"max_source_assertions": 17},
            },
        )

        with patch.object(LocalAgentConfig, "validate_runtime", return_value=[]):
            await manager._initialize_agent("claw", config)

        profile = mock_agent_cls.call_args.kwargs["semantic_inference_profile"]
        assert profile == InferenceProfile(
            OntologyRef(
                "http://www.w3.org/2000/01/rdf-schema#",
                "1.0.0",
                "e362812917fddab7cfab3dc35553ad292725e8f264e05f376077340e91034db5",
                "semantic-kb-v1",
            ),
            "1.0.0",
        )
        assert mock_agent_cls.call_args.kwargs[
            "semantic_inference_limits"
        ].max_source_assertions == 17
        assert mock_agent_cls.call_args.kwargs["semantic_inference_configured"] is True

    @pytest.mark.asyncio
    @patch(
        "kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did",
        new_callable=AsyncMock,
    )
    @patch("kestrel_sovereign.multi_agent.agent_manager.KestrelAgent")
    @patch("kestrel_sovereign.multi_agent.agent_manager.LLMService")
    async def test_in_process_agent_receives_exact_semantic_capability_selection(
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
            semantic_capabilities={
                "mode": "experimental",
                "rdf12": {
                    "capability": "rdf-profile:rdf12-cr-20260407-experimental",
                    "version": "0.1.0",
                },
                "sparql12": {
                    "capability": "query-profile:sparql12-20260605-experimental",
                    "version": "0.1.0",
                },
                "shacl12": {
                    "capability": "validation-profile:shacl12-core-20260602-experimental",
                    "version": "0.1.0",
                },
                "shape_set": {
                    "identifier": "kestrel-assertion-shapes-shacl12-experimental",
                    "version": "0.1.0",
                },
            },
        )

        with patch.object(LocalAgentConfig, "validate_runtime", return_value=[]):
            await manager._initialize_agent("claw", config)

        selected = mock_agent_cls.call_args.kwargs["semantic_capabilities"]
        assert selected.allow_experimental is True
        assert selected.shape_set.identifier == "kestrel-assertion-shapes-shacl12-experimental"
        assert mock_agent_cls.call_args.kwargs["semantic_capabilities_configured"] is True

    @pytest.mark.asyncio
    @patch(
        "kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did",
        new_callable=AsyncMock,
    )
    @patch("kestrel_sovereign.multi_agent.agent_manager.KestrelAgent")
    @patch("kestrel_sovereign.multi_agent.agent_manager.LLMService")
    async def test_in_process_agent_rejects_unavailable_pinned_inference_profile(
        self,
        mock_llm_cls,
        mock_agent_cls,
        mock_get_did,
        tmp_path,
    ):
        """Rule resources are validated before the agent can publish ready."""
        mock_get_did.return_value = "did:claw"
        manager = AgentManager(base_data_dir=tmp_path)
        config = LocalAgentConfig(
            data_dir=Path("agent_data/claw"),
            port=8801,
            semantic_inference={
                "enabled": True,
                "rdfs_version": "9.9.9",
                "ontology": {
                    "namespace": "http://www.w3.org/2000/01/rdf-schema#",
                    "version": "1.0.0",
                    "content_digest": "e362812917fddab7cfab3dc35553ad292725e8f264e05f376077340e91034db5",
                    "compatibility_profile": "semantic-kb-v1",
                },
            },
        )

        with patch.object(LocalAgentConfig, "validate_runtime", return_value=[]):
            with pytest.raises(InferenceError, match="RDFS"):
                await manager._initialize_agent("claw", config)

        mock_agent_cls.assert_not_called()

    @pytest.mark.asyncio
    @patch(
        "kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did",
        new_callable=AsyncMock,
    )
    @patch("kestrel_sovereign.multi_agent.agent_manager.KestrelAgent")
    @patch("kestrel_sovereign.multi_agent.agent_manager.LLMService")
    async def test_sqlite_agent_keeps_storage_derived_isolated_runtime_layout(
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
        kwargs = mock_agent_cls.call_args.kwargs
        assert kwargs["identity_export_dir"] == agent_root
        assert "isolated_runtime_root" not in kwargs
        assert "isolated_runtime_namespace" not in kwargs
        assert "isolated_runtime_hosted" not in kwargs

    @pytest.mark.asyncio
    async def test_postgres_factory_constructs_real_agent_with_derived_namespace(
        self, monkeypatch, tmp_path
    ):
        """Exercise the real derive -> KestrelAgent validation seam."""
        monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
        monkeypatch.setenv("KESTREL_DATABASE_URL", "postgresql://host/kestrel")
        did = "did:kestrel:real-hosted-construction"
        llm_service = MagicMock()
        llm_service.providers = []
        manager = AgentManager(base_data_dir=tmp_path)
        config = LocalAgentConfig(data_dir=Path("agent_data/real"), port=8801)
        (tmp_path / "agent_data" / "real").mkdir(parents=True)

        with patch.object(
            LocalAgentConfig, "validate_runtime", return_value=[]
        ), patch(
            "kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did",
            new=AsyncMock(return_value=did),
        ), patch(
            "kestrel_sovereign.multi_agent.agent_manager.LLMService",
            return_value=llm_service,
        ), patch.object(
            KestrelAgent, "initialize", new=AsyncMock()
        ):
            agent = await manager._initialize_agent("Real", config)

        expected_namespace = derive_isolated_runtime_namespace(did)
        assert type(agent) is KestrelAgent
        assert len(expected_namespace) <= 64
        assert str(agent.isolated_runtime_namespace) == expected_namespace
        assert agent.isolated_runtime_path == (
            tmp_path / "isolated_feature_runtime" / expected_namespace
        ).resolve()
        assert agent.isolated_runtime_legacy_root == (
            tmp_path / "agent_data" / "real" / "feature_venvs"
        ).resolve()

    @pytest.mark.asyncio
    @patch(
        "kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did",
        new_callable=AsyncMock,
    )
    @patch("kestrel_sovereign.multi_agent.agent_manager.KestrelAgent")
    @patch("kestrel_sovereign.multi_agent.agent_manager.LLMService")
    async def test_postgres_host_factory_supplies_distinct_isolated_runtime_scopes(
        self,
        mock_llm_cls,
        mock_agent_cls,
        mock_get_did,
        monkeypatch,
        tmp_path,
    ):
        """The real in-process factory owns scoped mutable feature state."""
        agent_a = _make_mock_agent("did:tenant-a")
        agent_b = _make_mock_agent("did:tenant-b")
        mock_agent_cls.side_effect = [agent_a, agent_b]
        mock_get_did.side_effect = ["did:tenant-a", "did:tenant-b"]
        monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
        monkeypatch.setenv("KESTREL_DATABASE_URL", "postgresql://host/kestrel")
        manager = AgentManager(base_data_dir=tmp_path)
        config = LocalAgentConfig(data_dir=Path("agent_data/companion"), port=8801)

        with patch.object(LocalAgentConfig, "validate_runtime", return_value=[]):
            await manager._initialize_agent("Companion A", config)
            await manager._initialize_agent("Companion B", config)

        first, second = (call.kwargs for call in mock_agent_cls.call_args_list)
        expected_root = (tmp_path / "isolated_feature_runtime").resolve()
        assert first["isolated_runtime_root"] == expected_root
        assert second["isolated_runtime_root"] == expected_root
        assert first["isolated_runtime_hosted"] is True
        assert second["isolated_runtime_hosted"] is True
        expected_legacy_root = (
            tmp_path / "agent_data" / "companion" / "feature_venvs"
        ).resolve()
        assert first["isolated_runtime_legacy_root"] == expected_legacy_root
        assert second["isolated_runtime_legacy_root"] == expected_legacy_root
        assert first["isolated_runtime_namespace"] == (
            derive_isolated_runtime_namespace("did:tenant-a")
        )
        assert second["isolated_runtime_namespace"] == (
            derive_isolated_runtime_namespace("did:tenant-b")
        )
        assert first["isolated_runtime_namespace"] != second["isolated_runtime_namespace"]

    @pytest.mark.asyncio
    @patch("kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did", new_callable=AsyncMock)
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
    @patch("kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did", new_callable=AsyncMock)
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
    @patch("kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did", new_callable=AsyncMock)
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
        "kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did",
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
    @patch("kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did", new_callable=AsyncMock)
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
    @patch("kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did", new_callable=AsyncMock)
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

    @pytest.mark.asyncio
    async def test_concurrent_same_name_create_is_admitted_once_before_inception(
        self, monkeypatch, tmp_path
    ):
        """One name owner reaches inception; the concurrent loser does not."""

        manager = AgentManager(base_data_dir=tmp_path)
        inception_started = asyncio.Event()
        allow_inception = asyncio.Event()
        inception_calls = 0
        child = _make_mock_agent("did:test:single-create-owner")

        async def inception(**_kwargs):
            nonlocal inception_calls
            inception_calls += 1
            inception_started.set()
            await allow_inception.wait()

        async def initialize(*_args, **_kwargs):
            return child

        monkeypatch.setattr(
            "kestrel_sovereign.inception_service.create_kestrel_identity_async",
            inception,
        )
        manager._initialize_agent = initialize

        first = asyncio.create_task(manager.create_agent("SameName"))
        await asyncio.wait_for(inception_started.wait(), timeout=1.0)
        with pytest.raises(ValueError, match="already being initialized or created"):
            await manager.create_agent("samename")
        with pytest.raises(ValueError, match="already being initialized or created"):
            await manager.load_agent(
                "SAMENAME", LocalAgentConfig(data_dir="other", port=8802)
            )

        allow_inception.set()
        assert await asyncio.wait_for(first, timeout=1.0) is child
        assert inception_calls == 1
        assert manager.list_agents() == {"SameName": child}
        assert manager.get_agent_name(child.agent_id) == "SameName"
        child.shutdown.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_shutdown_fences_pre_inception_create_without_deleting_identity(
        self, monkeypatch, tmp_path
    ):
        """An old create cannot publish after shutdown or block later reuse."""

        manager = AgentManager(base_data_dir=tmp_path)
        inception_started = asyncio.Event()
        allow_inception = asyncio.Event()
        child = _make_mock_agent("did:test:stale-create")

        async def inception(**_kwargs):
            inception_started.set()
            await allow_inception.wait()

        async def initialize(*_args, **_kwargs):
            return child

        monkeypatch.setattr(
            "kestrel_sovereign.inception_service.create_kestrel_identity_async",
            inception,
        )
        manager._initialize_agent = initialize

        create = asyncio.create_task(manager.create_agent("BeforeShutdown"))
        await asyncio.wait_for(inception_started.wait(), timeout=1.0)
        shutdown = asyncio.create_task(manager.shutdown_all())
        while not manager._agent_registration_sealed:
            await asyncio.sleep(0)
        # Inception itself is unbounded external work.  Shutdown fences its
        # later publication but does not hang waiting for that ordinary create.
        assert await asyncio.wait_for(shutdown, timeout=1.0) is None

        allow_inception.set()
        with pytest.raises(RuntimeError, match="began before manager shutdown"):
            await asyncio.wait_for(create, timeout=1.0)

        assert manager.list_agents() == {}
        assert manager._created_configs == {}
        child.shutdown.assert_not_awaited()
        # Inception data remains deliberately intact; a canceled/stale create
        # never guesses whether it is safe to delete an identity directory.
        assert (tmp_path / "agent_data" / "BeforeShutdown").is_dir()
        # The stale admission is rejected before it can initialize a runtime
        # around identity data written before shutdown.
        assert manager._agent_operations == {}

        replacement = _make_mock_agent("did:test:post-shutdown-create")

        async def initialize_replacement(*_args, **_kwargs):
            return replacement

        manager._initialize_agent = initialize_replacement
        assert await manager.load_agent(
            "BeforeShutdown", LocalAgentConfig(data_dir="replacement", port=8802)
        ) is replacement
        assert manager.get_agent("BeforeShutdown") is replacement

    @pytest.mark.asyncio
    async def test_concurrent_same_name_load_rejects_before_second_initialization(self):
        """Direct cold loads share the same admission, not just create_agent."""

        manager = AgentManager()
        initialized = asyncio.Event()
        allow_first = asyncio.Event()
        child = _make_mock_agent("did:test:single-load-owner")
        initialize_calls = 0

        async def initialize(*_args, **_kwargs):
            nonlocal initialize_calls
            initialize_calls += 1
            initialized.set()
            await allow_first.wait()
            return child

        manager._initialize_agent = initialize
        config = LocalAgentConfig(data_dir="same", port=8801)
        first = asyncio.create_task(manager.load_agent("SameLoad", config))
        await asyncio.wait_for(initialized.wait(), timeout=1.0)
        with pytest.raises(ValueError, match="already being initialized or created"):
            await manager.load_agent("sameload", config)

        allow_first.set()
        assert await asyncio.wait_for(first, timeout=1.0) is child
        assert initialize_calls == 1
        assert manager.list_agents() == {"SameLoad": child}
        assert manager._agent_names == {child.agent_id: "SameLoad"}

    @pytest.mark.asyncio
    async def test_cancelled_operation_release_waits_for_state_lock(
        self,
    ) -> None:
        """A completed admission cannot be stranded by cancellation in finally."""

        manager = AgentManager()
        admission, owns_admission = await manager._admit_agent_operation(
            "Completed", kind="test"
        )
        assert owns_admission

        await manager._lock.acquire()
        release = asyncio.create_task(manager._release_agent_operation(admission))
        try:
            await asyncio.sleep(0)
            release.cancel()

            # The owned release is blocked on the state lock. Releasing that
            # lock must finish its mutation before the caller's cancellation
            # can propagate.
            manager._lock.release()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(release, timeout=1.0)
        finally:
            if manager._lock.locked():
                manager._lock.release()

        assert manager._agent_operations == {}


class TestSpawnAgent:
    """Test spawn_agent (delegation chain)."""

    @pytest.mark.asyncio
    @patch("kestrel_sovereign.inception_service.create_kestrel_identity_async", new_callable=AsyncMock)
    @patch("kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did", new_callable=AsyncMock)
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
    @patch("kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did", new_callable=AsyncMock)
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
    @patch("kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did", new_callable=AsyncMock)
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
    @patch("kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did", new_callable=AsyncMock)
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
