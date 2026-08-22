"""Command-handler contracts for privacy mode commands."""

from unittest.mock import AsyncMock, MagicMock

import inspect
import pytest

from kestrel_sovereign.command_handler import CommandHandler
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.privacy import PrivacyMode
from kestrel_sovereign.storage.privacy_wrapper import (
    PRIVACY_TRANSITION_RETRY_MESSAGE,
    PrivacyEnforcingStorage,
)


def test_privacy_command_handler_is_explicitly_async():
    agent = MagicMock()
    handler = CommandHandler(agent)

    assert inspect.iscoroutinefunction(handler._cmd_privacy)


@pytest.mark.asyncio
async def test_privacy_command_delegates_to_agent_level_transition():
    agent = MagicMock()
    agent.set_privacy_mode = AsyncMock(return_value="Privacy mode changed from normal to isolated.")
    agent.privacy_agent.get_status = MagicMock(return_value="Current privacy mode: normal")

    handler = CommandHandler(agent)

    result = await handler._cmd_privacy("!privacy isolated")

    assert result == "Privacy mode changed from normal to isolated."
    agent.set_privacy_mode.assert_awaited_once_with(PrivacyMode.ISOLATED)


@pytest.mark.asyncio
async def test_privacy_command_without_mode_returns_current_status():
    agent = MagicMock()
    agent.privacy_agent.get_status = MagicMock(return_value="Current privacy mode: normal")

    handler = CommandHandler(agent)

    result = await handler._cmd_privacy("!privacy")

    assert result == "Current privacy mode: normal"
    agent.privacy_agent.get_status.assert_called_once_with()


@pytest.mark.asyncio
async def test_privacy_command_reports_retry_and_succeeds_after_fact_lease():
    """!privacy does not propagate or claim a refused transition."""
    from kestrel_sovereign.features.privacy.feature import (
        PrivacyTransitionDecision,
    )

    wrapper = PrivacyEnforcingStorage(MagicMock(), PrivacyMode.NORMAL)
    agent = KestrelAgent.__new__(KestrelAgent)
    agent._privacy_mode = PrivacyMode.NORMAL
    agent.storage = wrapper
    agent.features = {}
    agent.llm_service = None
    agent.privacy_agent = MagicMock()
    agent.privacy_agent.evaluate_transition = MagicMock(
        side_effect=lambda mode: PrivacyTransitionDecision(
            target=mode,
            requires_confirmation=False,
        )
    )
    agent.privacy_agent.set_mode = MagicMock(
        return_value="Privacy mode changed from normal to isolated."
    )
    handler = CommandHandler(agent)

    wrapper._acquire_explicit_fact_lease()
    try:
        refused = await handler._cmd_privacy("!privacy isolated")
        assert refused == PRIVACY_TRANSITION_RETRY_MESSAGE
        assert agent.privacy_mode is PrivacyMode.NORMAL
        assert wrapper.privacy_mode is PrivacyMode.NORMAL
    finally:
        wrapper._release_explicit_fact_lease()

    applied = await handler._cmd_privacy("!privacy isolated")
    assert applied == "Privacy mode changed from normal to isolated."
    assert agent.privacy_mode is PrivacyMode.ISOLATED
    assert wrapper.privacy_mode is PrivacyMode.ISOLATED
