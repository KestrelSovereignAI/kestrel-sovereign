"""Unit tests for the in-process AgentManager."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from kestrel_sovereign.features import MandatoryFeatureReadinessError
from kestrel_sovereign.identity.runtime_identity import IdentityReadinessError
from kestrel_sovereign.multi_agent.agent_manager import AgentManager
from kestrel_sovereign.multi_agent.config import LocalAgentConfig, MultiAgentConfig
from kestrel_sovereign.spawn.mandate import SpawnMandate


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

        await manager.shutdown_all()
        # Both should have been attempted
        assert len(manager._agents) == 0


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
            child = await manager.spawn_agent("SpawnedBot", parent, mandate)

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
