"""
Unit tests for the rename command.

Tests the agent renaming functionality in the Bootstrap Feature.
"""

import pytest
import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


class MockStorage:
    """Mock storage for testing."""

    def __init__(self):
        self.nodes = {}

    async def get_node(self, node_id: str):
        """Get a node by ID."""
        return self.nodes.get(node_id)

    async def add_node(self, node):
        """Add or update a node."""
        self.nodes[node.node_id] = node


class MockDB:
    """Mock database for testing."""

    def __init__(self):
        self.data = {}
        self.executed = []

    async def execute(self, query: str, params: tuple = None):
        """Mock execute."""
        self.executed.append((query, params))
        if params and len(params) >= 4:
            key = (params[0], params[1])
            self.data[key] = params[2]

    async def fetchall(self, query: str, params: tuple = None):
        """Mock fetchall."""
        key = (params[0], params[1]) if params and len(params) >= 2 else None
        if key and key in self.data:
            return [(self.data[key],)]
        return []


class MockAgent:
    """Mock agent for testing."""

    def __init__(self, agent_id="did:test:123", agent_name="TestAgent"):
        self.did = agent_id
        self.agent_id = agent_id
        self._agent_name = agent_name
        self.storage = MockStorage()
        self._raw_storage = MagicMock()
        self._raw_storage.db = MockDB()
        self.context_builder = MagicMock()
        self.bootstrap_service = MagicMock()
        self.bootstrap_service.agent_name = agent_name
        self.bootstrap_service.agent_data_path = None


class MockNode:
    """Mock graph node."""

    def __init__(self, node_id, properties=None, label=None):
        self.node_id = node_id
        self.properties = properties or {}
        self.label = label or ""


@pytest.fixture
def mock_agent():
    """Create a mock agent."""
    return MockAgent()


@pytest.fixture
def temp_agent_dir(tmp_path):
    """Create a temporary agent data directory."""
    agent_dir = tmp_path / "agent_data" / "test_agent"
    agent_dir.mkdir(parents=True)
    return agent_dir


class TestRenameValidation:
    """Tests for rename input validation."""

    @pytest.mark.asyncio
    async def test_exactly_64_character_name_is_allowed(self, mock_agent):
        """The documented 64-character boundary should succeed and persist."""
        from kestrel_sovereign.features.bootstrap.feature import BootstrapFeature

        feature = BootstrapFeature(mock_agent)
        await feature.initialize()
        mock_agent._agent_name = "OldName"
        mock_agent.storage.nodes[mock_agent.agent_id] = MockNode(
            mock_agent.agent_id,
            properties={"name": "OldName"},
            label="OldName",
        )
        valid_name = "A" * 64

        result = await feature.rename_agent(valid_name)
        node = await mock_agent.storage.get_node(mock_agent.agent_id)

        assert "Renamed" in result
        assert mock_agent._agent_name == valid_name
        assert mock_agent.bootstrap_service.agent_name == valid_name
        assert node.properties["name"] == valid_name
        assert node.label == valid_name
        assert mock_agent._raw_storage.db.data[(mock_agent.agent_id, "name")] == valid_name

    @pytest.mark.asyncio
    async def test_too_long_name_does_not_mutate_any_rename_targets(self, mock_agent, temp_agent_dir):
        """Rejecting a long name should leave memory, graph, metadata, and SOUL.md unchanged."""
        from kestrel_sovereign.features.bootstrap.feature import BootstrapFeature

        mock_agent.bootstrap_service.agent_data_path = temp_agent_dir
        soul_path = temp_agent_dir / "SOUL.md"
        original_soul = "# SOUL.md - You Are OldName\n\nYou're OldName."
        soul_path.write_text(original_soul)
        mock_agent._agent_name = "OldName"
        mock_agent.bootstrap_service.agent_name = "OldName"
        mock_agent.storage.nodes[mock_agent.agent_id] = MockNode(
            mock_agent.agent_id,
            properties={"name": "OldName"},
            label="OldName",
        )
        feature = BootstrapFeature(mock_agent)
        await feature.initialize()

        result = await feature.rename_agent("A" * 65)
        node = await mock_agent.storage.get_node(mock_agent.agent_id)

        assert "too long" in result.lower()
        assert mock_agent._agent_name == "OldName"
        assert mock_agent.bootstrap_service.agent_name == "OldName"
        assert node.properties["name"] == "OldName"
        assert node.label == "OldName"
        assert mock_agent._raw_storage.db.executed == []
        assert soul_path.read_text() == original_soul

    @pytest.mark.asyncio
    async def test_rename_core_raises_value_error_for_invalid_names(self, mock_agent):
        """The shared core should fail loudly; the tool wrapper converts errors to text."""
        from kestrel_sovereign.features.bootstrap.feature import rename_agent_core

        with pytest.raises(ValueError, match="empty"):
            await rename_agent_core(mock_agent, "")
        with pytest.raises(ValueError, match="too long"):
            await rename_agent_core(mock_agent, "A" * 65)


class TestRenameExecution:
    """Tests for rename execution."""

    @pytest.mark.asyncio
    async def test_rename_updates_agent_name(self, mock_agent):
        """Renaming should update the agent's internal name."""
        from kestrel_sovereign.features.bootstrap.feature import BootstrapFeature

        feature = BootstrapFeature(mock_agent)
        await feature.initialize()

        # Set up the agent node
        mock_agent.storage.nodes[mock_agent.agent_id] = MockNode(
            mock_agent.agent_id,
            properties={"name": "OldName"},
            label="OldName"
        )

        result = await feature.rename_agent("NewName")

        assert "NewName" in result
        assert mock_agent._agent_name == "NewName"

    @pytest.mark.asyncio
    async def test_rename_updates_bootstrap_service(self, mock_agent):
        """Renaming should update the bootstrap service name."""
        from kestrel_sovereign.features.bootstrap.feature import BootstrapFeature

        feature = BootstrapFeature(mock_agent)
        await feature.initialize()

        mock_agent.storage.nodes[mock_agent.agent_id] = MockNode(
            mock_agent.agent_id,
            properties={"name": "OldName"},
        )

        await feature.rename_agent("NewName")

        assert mock_agent.bootstrap_service.agent_name == "NewName"

    @pytest.mark.asyncio
    async def test_rename_returns_confirmation(self, mock_agent):
        """Renaming should return a confirmation message."""
        from kestrel_sovereign.features.bootstrap.feature import BootstrapFeature

        feature = BootstrapFeature(mock_agent)
        await feature.initialize()

        mock_agent._agent_name = "OldName"
        mock_agent.storage.nodes[mock_agent.agent_id] = MockNode(
            mock_agent.agent_id,
            properties={"name": "OldName"},
        )

        result = await feature.rename_agent("NewName")

        assert "OldName" in result
        assert "NewName" in result
        assert "Renamed" in result


class TestSoulMdUpdate:
    """Tests for SOUL.md name updates during rename."""

    @pytest.mark.asyncio
    async def test_rename_updates_soul_header(self, mock_agent, temp_agent_dir):
        """Renaming should update the SOUL.md header."""
        from kestrel_sovereign.features.bootstrap.feature import BootstrapFeature

        # Set up agent with temp directory
        mock_agent.bootstrap_service.agent_data_path = temp_agent_dir

        # Create a SOUL.md with old name
        soul_path = temp_agent_dir / "SOUL.md"
        soul_path.write_text("# SOUL.md - You Are OldName\n\nContent here")

        feature = BootstrapFeature(mock_agent)
        await feature.initialize()

        mock_agent.storage.nodes[mock_agent.agent_id] = MockNode(
            mock_agent.agent_id,
            properties={"name": "OldName"},
        )
        mock_agent._agent_name = "OldName"

        result = await feature.rename_agent("NewName")

        # Check SOUL.md was updated
        updated_content = soul_path.read_text()
        assert "NewName" in updated_content
        assert "SOUL.md updated" in result

    @pytest.mark.asyncio
    async def test_rename_updates_soul_content(self, mock_agent, temp_agent_dir):
        """Renaming should update name references in SOUL.md content."""
        from kestrel_sovereign.features.bootstrap.feature import BootstrapFeature

        mock_agent.bootstrap_service.agent_data_path = temp_agent_dir

        # Create a SOUL.md with name references
        soul_path = temp_agent_dir / "SOUL.md"
        soul_path.write_text(
            "# SOUL.md - You Are OldName\n\n"
            "You're OldName, a Kestrel agent.\n"
            "I'm OldName. Born today."
        )

        feature = BootstrapFeature(mock_agent)
        await feature.initialize()

        mock_agent.storage.nodes[mock_agent.agent_id] = MockNode(
            mock_agent.agent_id,
            properties={"name": "OldName"},
        )
        mock_agent._agent_name = "OldName"

        await feature.rename_agent("NewName")

        updated_content = soul_path.read_text()
        assert "You're NewName" in updated_content
        assert "I'm NewName" in updated_content

    @pytest.mark.asyncio
    async def test_rename_without_soul_succeeds(self, mock_agent, temp_agent_dir):
        """Renaming without SOUL.md should still succeed."""
        from kestrel_sovereign.features.bootstrap.feature import BootstrapFeature

        mock_agent.bootstrap_service.agent_data_path = temp_agent_dir
        # No SOUL.md exists

        feature = BootstrapFeature(mock_agent)
        await feature.initialize()

        mock_agent.storage.nodes[mock_agent.agent_id] = MockNode(
            mock_agent.agent_id,
            properties={"name": "OldName"},
        )

        result = await feature.rename_agent("NewName")

        assert "NewName" in result
        assert "SOUL.md updated" not in result


class TestRenameEdgeCases:
    """Tests for edge cases in renaming."""

    @pytest.mark.asyncio
    async def test_rename_empty_name_rejected(self, mock_agent):
        """Empty name should be rejected."""
        from kestrel_sovereign.features.bootstrap.feature import BootstrapFeature

        feature = BootstrapFeature(mock_agent)
        await feature.initialize()

        result = await feature.rename_agent("")

        assert "provide a new name" in result.lower() or "usage" in result.lower()

    @pytest.mark.asyncio
    async def test_rename_whitespace_only_rejected(self, mock_agent):
        """Whitespace-only name should be rejected."""
        from kestrel_sovereign.features.bootstrap.feature import BootstrapFeature

        feature = BootstrapFeature(mock_agent)
        await feature.initialize()

        result = await feature.rename_agent("   ")

        assert "provide a new name" in result.lower() or "usage" in result.lower()

    @pytest.mark.asyncio
    async def test_rename_long_name_rejected(self, mock_agent):
        """Names over 64 characters should be rejected."""
        from kestrel_sovereign.features.bootstrap.feature import BootstrapFeature

        feature = BootstrapFeature(mock_agent)
        await feature.initialize()

        long_name = "A" * 65
        result = await feature.rename_agent(long_name)

        assert "too long" in result.lower()

    @pytest.mark.asyncio
    async def test_rename_strips_whitespace(self, mock_agent):
        """Names should have whitespace stripped."""
        from kestrel_sovereign.features.bootstrap.feature import BootstrapFeature

        feature = BootstrapFeature(mock_agent)
        await feature.initialize()

        mock_agent.storage.nodes[mock_agent.agent_id] = MockNode(
            mock_agent.agent_id,
            properties={"name": "OldName"},
        )

        result = await feature.rename_agent("  NewName  ")

        assert mock_agent._agent_name == "NewName"
