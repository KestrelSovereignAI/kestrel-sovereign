"""Unit tests for the in-process AgentManager."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from kestrel_sovereign.rookery.agent_manager import AgentManager
from kestrel_sovereign.rookery.config import LocalAgentConfig, RookeryConfig


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
    """Test loading agents from RookeryConfig."""

    @pytest.mark.asyncio
    @patch("kestrel_sovereign.rookery.agent_manager._get_agent_did", new_callable=AsyncMock)
    @patch("kestrel_sovereign.rookery.agent_manager.KestrelAgent")
    @patch("kestrel_sovereign.rookery.agent_manager.LLMService")
    async def test_load_from_config_skips_non_autostart(
        self, mock_llm_cls, mock_agent_cls, mock_get_did
    ):
        """Agents with autostart=false should be skipped."""
        config = RookeryConfig(
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
    @patch("kestrel_sovereign.rookery.agent_manager._get_agent_did", new_callable=AsyncMock)
    @patch("kestrel_sovereign.rookery.agent_manager.KestrelAgent")
    @patch("kestrel_sovereign.rookery.agent_manager.LLMService")
    async def test_load_from_config_handles_errors(
        self, mock_llm_cls, mock_agent_cls, mock_get_did
    ):
        """Failed agent loads should log error but not crash."""
        config = RookeryConfig(
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
    @patch("kestrel_sovereign.rookery.agent_manager._get_agent_did", new_callable=AsyncMock)
    @patch("kestrel_sovereign.rookery.agent_manager.KestrelAgent")
    @patch("kestrel_sovereign.rookery.agent_manager.LLMService")
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
