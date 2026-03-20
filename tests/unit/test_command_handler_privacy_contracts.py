"""Command-handler contracts for privacy mode commands."""

from unittest.mock import AsyncMock, MagicMock

import inspect
import pytest

from kestrel_sovereign.command_handler import CommandHandler
from kestrel_sovereign.privacy import PrivacyMode


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
