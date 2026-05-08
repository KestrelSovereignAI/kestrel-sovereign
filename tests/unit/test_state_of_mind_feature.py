"""Direct contracts for the StateOfMind feature."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from kestrel_sdk.tools.result import ToolResultStatus
from kestrel_sovereign.features.state_of_mind import StateOfMindFeature


@pytest.mark.asyncio
async def test_state_of_mind_requires_llm_service():
    feature = StateOfMindFeature(agent=SimpleNamespace())

    result = await feature.get_state_of_mind()

    # Migrated to ToolResult (#1061 wave 19): missing LLM service
    # surfaces as ERROR with the diagnostic in result.error.
    assert result.status is ToolResultStatus.ERROR
    assert "LLM service not available" in result.error


@pytest.mark.asyncio
async def test_state_of_mind_formats_via_profile_service():
    llm_service = MagicMock()
    llm_service.get_state_of_mind.return_value = {"provider": "anthropic", "model": "auto"}
    agent = SimpleNamespace(llm_service=llm_service)
    feature = StateOfMindFeature(agent=agent)
    profile_service = MagicMock()
    profile_service.format_state_of_mind.return_value = "formatted state"

    with patch(
        "kestrel_sovereign.llm.constitutional_profile.get_profile_service",
        return_value=profile_service,
    ):
        result = await feature.get_state_of_mind()

    assert result.status is ToolResultStatus.OK
    assert result.confirmation == "formatted state"
    profile_service.format_state_of_mind.assert_called_once()
