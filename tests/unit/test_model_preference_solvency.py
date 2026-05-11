import logging

import pytest

from kestrel_sovereign.agent.model_preference import ModelPreferenceMixin


class _Agent(ModelPreferenceMixin):
    def __init__(self, wallet=None):
        self.wallet = wallet
        self._current_model_preference = None


@pytest.mark.asyncio
async def test_check_solvency_skips_when_wallet_feature_disabled(caplog):
    agent = _Agent(wallet=None)

    with caplog.at_level(logging.ERROR):
        override = await agent.check_solvency()

    assert override is None
    assert "Solvency check failed" not in caplog.text
