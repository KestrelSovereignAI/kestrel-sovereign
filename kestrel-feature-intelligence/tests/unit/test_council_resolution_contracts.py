"""Contracts for council config path and auto model resolution."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from kestrel_feature_intelligence.council.deliberation import _initialize_adapters
from kestrel_feature_intelligence.council.feature import CONFIG_PATH
from kestrel_feature_intelligence.council.models import CouncilMember


def test_council_feature_uses_repo_root_config_path():
    assert CONFIG_PATH == Path("council_config.toml")


@pytest.mark.asyncio
async def test_initialize_adapters_resolves_auto_models_before_adapter_init():
    member = CouncilMember(
        name="Claude",
        provider="anthropic",
        model="auto",
        role="constitutional_reviewer",
    )

    with patch(
        "kestrel_feature_intelligence.council.deliberation.resolve_provider_default",
        return_value="claude-opus-4-5-20251101",
    ) as mock_resolve:
        with patch(
            "kestrel_feature_intelligence.council.deliberation._get_adapter_for_provider",
            new=AsyncMock(return_value=("client", "adapter")),
        ) as mock_get_adapter:
            adapters = await _initialize_adapters([member])

    mock_resolve.assert_called_once_with("anthropic")
    mock_get_adapter.assert_awaited_once_with("anthropic", "claude-opus-4-5-20251101")
    assert adapters["Claude"] == ("client", "adapter", "claude-opus-4-5-20251101")
