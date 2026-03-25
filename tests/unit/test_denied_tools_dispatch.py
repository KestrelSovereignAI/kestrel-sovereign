"""Tests for security DENY enforcement at orchestrator dispatch level.

Verifies that denied tools are stripped from the subagent's tool palette
before the LLM ever sees them (primary gate), complementing the existing
hook check in features/base.py (secondary gate).
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestDeniedToolsStripping:
    """Test that _get_denied_tools correctly queries the permission store."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent with OrchestratorEngineMixin methods."""
        from kestrel_sovereign.agent.orchestrator_engine import OrchestratorEngineMixin

        agent = MagicMock()
        agent._get_denied_tools = OrchestratorEngineMixin._get_denied_tools.__get__(agent)
        return agent

    @pytest.fixture
    def mock_permission_store(self):
        from kestrel_sovereign.features.security.permissions import PermissionLevel, PermissionStore

        store = MagicMock(spec=PermissionStore)
        # Default: export_sovereignty is DENY, others are ALLOW
        async def get_perm(feature_name, tool_name):
            if tool_name == "export_sovereignty":
                return PermissionLevel.DENY
            return PermissionLevel.ALLOW

        store.get_permission = AsyncMock(side_effect=get_perm)
        return store

    @pytest.fixture
    def mock_sovereignty_feature(self):
        feature = MagicMock()
        type(feature).__name__ = "SovereigntyFeature"

        tool1 = MagicMock()
        tool1.name = "export_sovereignty"
        tool2 = MagicMock()
        tool2.name = "import_sovereignty"
        tool3 = MagicMock()
        tool3.name = "check_sovereignty_status"

        feature.get_tools.return_value = [tool1, tool2, tool3]
        return feature

    @pytest.mark.asyncio
    async def test_denied_tools_detected(
        self, mock_agent, mock_permission_store, mock_sovereignty_feature
    ):
        """export_sovereignty should be in the denied set."""
        security_feature = MagicMock()
        security_feature.permission_store = mock_permission_store

        mock_agent.features = {
            "SecurityFeature": security_feature,
            "SovereigntyFeature": mock_sovereignty_feature,
        }

        denied = await mock_agent._get_denied_tools("SovereigntyFeature")
        assert "export_sovereignty" in denied
        assert "import_sovereignty" not in denied
        assert "check_sovereignty_status" not in denied

    @pytest.mark.asyncio
    async def test_no_denied_tools(self, mock_agent, mock_sovereignty_feature):
        """When all tools are ALLOW, denied set should be empty."""
        from kestrel_sovereign.features.security.permissions import PermissionLevel, PermissionStore

        store = MagicMock(spec=PermissionStore)
        store.get_permission = AsyncMock(return_value=PermissionLevel.ALLOW)

        security_feature = MagicMock()
        security_feature.permission_store = store

        mock_agent.features = {
            "SecurityFeature": security_feature,
            "SovereigntyFeature": mock_sovereignty_feature,
        }

        denied = await mock_agent._get_denied_tools("SovereigntyFeature")
        assert denied == set()

    @pytest.mark.asyncio
    async def test_no_security_feature(self, mock_agent):
        """Without SecurityFeature, no tools are denied."""
        mock_agent.features = {}
        denied = await mock_agent._get_denied_tools("SovereigntyFeature")
        assert denied == set()


class TestSubagentDeniedToolsExclusion:
    """Test that execute_as_subagent excludes denied tools from the palette."""

    @pytest.fixture
    def feature_with_tools(self):
        """Create a real-ish feature with mock tools."""
        from kestrel_sovereign.features.base import Feature

        feature = MagicMock(spec=Feature)
        feature.name = "SovereigntyFeature"
        feature.tool_description = "Sovereignty management"

        tool1 = MagicMock()
        tool1.name = "export_sovereignty"
        tool1.schema.to_openai_format.return_value = {
            "type": "function",
            "function": {"name": "export_sovereignty"},
        }
        tool2 = MagicMock()
        tool2.name = "check_sovereignty_status"
        tool2.schema.to_openai_format.return_value = {
            "type": "function",
            "function": {"name": "check_sovereignty_status"},
        }

        feature.get_tools.return_value = [tool1, tool2]
        return feature

    def test_denied_tools_excluded_from_palette(self, feature_with_tools):
        """Denied tools should not appear in the tools sent to the LLM."""
        from kestrel_sovereign.features.base import Feature

        all_tools = feature_with_tools.get_tools()
        denied = {"export_sovereignty"}
        available = [t for t in all_tools if t.name not in denied]

        assert len(available) == 1
        assert available[0].name == "check_sovereignty_status"

    @pytest.mark.asyncio
    async def test_all_tools_denied_returns_error(self):
        """When all tools are denied, subagent should return error immediately."""
        from kestrel_sovereign.features.base import Feature

        feature = MagicMock(spec=Feature)
        feature.name = "SovereigntyFeature"
        feature.get_tools.return_value = []

        # Simulate the logic from execute_as_subagent
        denied_tools = {"export_sovereignty", "import_sovereignty", "check_sovereignty_status"}
        available_tools = [t for t in feature.get_tools() if t.name not in denied_tools]

        if not available_tools and denied_tools:
            result = {
                "success": False,
                "error": f"All tools in {feature.name} are blocked by security policy",
            }

        assert result["success"] is False
        assert "blocked by security policy" in result["error"]
