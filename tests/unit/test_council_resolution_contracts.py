"""Contracts for council config path and auto model resolution."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from kestrel_sovereign.features.council.deliberation import _initialize_adapters
from kestrel_sovereign.features.council.feature import _resolve_council_config_path
from kestrel_sovereign.features.council.models import CouncilMember


def test_council_feature_resolves_config_under_project_dir(tmp_path, monkeypatch):
    """Council config lives in the resolved project dir — not CWD-relative.

    The historical ``Path('council_config.toml')`` resolved to whatever the
    operator's CWD happened to be, which silently broke pip-installed users
    whose CWD wasn't a Kestrel project. The path now goes through
    ``kestrel_sovereign.paths.project_dir`` so ``KESTREL_HOME`` /
    marker-walk-up / ``~/.kestrel`` all do the right thing.
    """
    monkeypatch.setenv("KESTREL_HOME", str(tmp_path))
    from kestrel_sovereign import paths
    paths.reset_cache()

    assert _resolve_council_config_path() == tmp_path / "council_config.toml"


@pytest.mark.asyncio
async def test_initialize_adapters_resolves_auto_models_before_adapter_init():
    member = CouncilMember(
        name="Claude",
        provider="anthropic",
        model="auto",
        role="constitutional_reviewer",
    )

    with patch(
        "kestrel_sovereign.features.council.deliberation.resolve_provider_default",
        return_value="claude-opus-4-5-20251101",
    ) as mock_resolve:
        with patch(
            "kestrel_sovereign.features.council.deliberation._get_adapter_for_provider",
            new=AsyncMock(return_value=("client", "adapter")),
        ) as mock_get_adapter:
            adapters = await _initialize_adapters([member])

    mock_resolve.assert_called_once_with("anthropic")
    mock_get_adapter.assert_awaited_once_with("anthropic", "claude-opus-4-5-20251101")
    assert adapters["Claude"] == ("client", "adapter", "claude-opus-4-5-20251101")
