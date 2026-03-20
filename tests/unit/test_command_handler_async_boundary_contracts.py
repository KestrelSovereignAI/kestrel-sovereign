"""Contracts for explicit sync/async boundaries in CommandHandler."""

import inspect
from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sovereign.command_handler import CommandHandler


def test_privacy_save_handler_is_explicitly_async():
    agent = MagicMock()
    handler = CommandHandler(agent)

    assert inspect.iscoroutinefunction(handler._cmd_privacy_save)


@pytest.mark.asyncio
async def test_handle_awaits_privacy_save_without_leaking_coroutine():
    agent = MagicMock()
    agent.privacy_agent.save_isolated_session = AsyncMock(return_value="Saved 2 messages.")

    handler = CommandHandler(agent)

    result = await handler.handle("!privacy-save")

    assert result == "Saved 2 messages."
    agent.privacy_agent.save_isolated_session.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_handle_accepts_custom_awaitable_results_via_isawaitable():
    class CustomAwaitable:
        def __await__(self):
            async def _inner():
                return "custom result"

            return _inner().__await__()

    agent = MagicMock()
    handler = CommandHandler(agent)
    handler._command_handlers["!custom-awaitable"] = lambda _user_input: CustomAwaitable()

    result = await handler.handle("!custom-awaitable")

    assert result == "custom result"


def test_create_agent_handler_is_explicitly_async():
    agent = MagicMock()
    handler = CommandHandler(agent)

    assert inspect.iscoroutinefunction(handler._cmd_create_agent)


@pytest.mark.asyncio
async def test_handle_awaits_create_agent_command():
    agent = MagicMock()
    agent.create_trusted_agent = AsyncMock(return_value="Created trusted agent 'buddy'")

    handler = CommandHandler(agent)

    result = await handler.handle("!create-agent buddy")

    assert result == "Created trusted agent 'buddy'"
    agent.create_trusted_agent.assert_awaited_once_with("buddy")


def test_anchor_handler_is_explicitly_async():
    agent = MagicMock()
    handler = CommandHandler(agent)

    assert inspect.iscoroutinefunction(handler._cmd_anchor)


@pytest.mark.asyncio
async def test_handle_awaits_anchor_command():
    agent = MagicMock()
    agent.anchor_memory_state = AsyncMock(return_value="Successfully anchored memory state.")

    handler = CommandHandler(agent)

    result = await handler.handle("!anchor")

    assert result == "Successfully anchored memory state."
    agent.anchor_memory_state.assert_awaited_once_with()
