"""Direct contracts for the StateOfMind feature."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from kestrel_sovereign.features.state_of_mind import StateOfMindFeature


@pytest.mark.asyncio
async def test_state_of_mind_requires_llm_service():
    feature = StateOfMindFeature(agent=SimpleNamespace())

    result = await feature.get_state_of_mind()

    assert result == "Error: LLM service not available"


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

    assert result == "formatted state"
    profile_service.format_state_of_mind.assert_called_once()
