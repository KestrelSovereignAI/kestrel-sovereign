"""Command-handler contracts for constitution verification commands."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sovereign.command_handler import CommandHandler


@pytest.mark.asyncio
async def test_verify_constitution_returns_success_and_records_verification_state():
    agent = MagicMock()
    agent._verify_constitution_integrity = AsyncMock(return_value=(True, "Constitution integrity verified."))
    agent.enter_safe_mode = AsyncMock()
    agent._constitution_verified = None

    handler = CommandHandler(agent)

    result = await handler.handle("!verify-constitution")

    assert result == "✅ Constitution integrity verified."
    assert agent._constitution_verified is True
    agent.enter_safe_mode.assert_not_awaited()


@pytest.mark.asyncio
async def test_verify_constitution_enters_safe_mode_on_failure():
    agent = MagicMock()
    agent._verify_constitution_integrity = AsyncMock(return_value=(False, "Integrity failure"))
    agent.enter_safe_mode = AsyncMock()
    agent._constitution_verified = None

    handler = CommandHandler(agent)

    result = await handler.handle("!verify-constitution")

    assert result == "🚨 Integrity failure\n\nAgent has entered SAFE MODE. Contact administrator."
    assert agent._constitution_verified is False
    agent.enter_safe_mode.assert_awaited_once_with("Integrity failure")
