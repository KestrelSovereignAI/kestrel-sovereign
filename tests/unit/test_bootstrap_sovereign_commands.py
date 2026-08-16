"""Regression coverage for command routing while bootstrap is required."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from kestrel_sovereign.auth import CallerContext
from kestrel_sovereign.command_handler import CommandHandler
from kestrel_sovereign.kestrel_agent import KestrelAgent


def _bootstrap_agent() -> KestrelAgent:
    """Build the smallest initialized agent that exercises ``process_input``."""
    agent = KestrelAgent.__new__(KestrelAgent)
    agent.storage = object()
    agent._safe_mode = False
    agent._constitution_audit_pending = False
    agent._maybe_refresh_user_byok_resolver = AsyncMock()
    agent._genesis_audit_cognition_block = AsyncMock(return_value=None)
    agent._maybe_audit = AsyncMock()

    @asynccontextmanager
    async def _noop_lifecycle():
        yield

    agent._turn_lifecycle = _noop_lifecycle
    agent.bootstrap_service = MagicMock()
    agent.bootstrap_service.is_bootstrap_needed = AsyncMock(return_value=True)
    agent._handle_bootstrap = AsyncMock(return_value="onboarding greeting")
    agent._post_response_pipeline = AsyncMock()
    agent.command_handler = CommandHandler(agent)
    return agent


@pytest.mark.asyncio
async def test_bootstrap_reanchor_command_reaches_sovereign_command_handler():
    agent = _bootstrap_agent()
    caller = CallerContext.sovereign()
    real_handle = agent.command_handler.handle
    agent.command_handler.handle = AsyncMock(wraps=real_handle)

    unauthorized = await agent.process_input("!reanchor-constitution")
    result = await agent.process_input("!reanchor-constitution", caller=caller)

    assert unauthorized.startswith(
        "🚨 Unauthorized: !reanchor-constitution requires sovereign authority"
    )
    assert result.startswith("Usage: !reanchor-constitution")
    assert agent.command_handler.handle.await_args_list == [
        call("!reanchor-constitution", caller=None),
        call("!reanchor-constitution", caller=caller),
    ]
    agent._handle_bootstrap.assert_not_awaited()


@pytest.mark.asyncio
async def test_bootstrap_safe_mode_command_reaches_sovereign_command_handler():
    agent = _bootstrap_agent()
    caller = CallerContext.sovereign()
    real_handle = agent.command_handler.handle
    agent.command_handler.handle = AsyncMock(wraps=real_handle)

    unauthorized = await agent.process_input("!safe-mode")
    result = await agent.process_input("!safe-mode", caller=caller)

    assert unauthorized.startswith(
        "🚨 Unauthorized: !safe-mode requires sovereign authority"
    )
    assert result == "✅ Normal operation mode. No integrity issues detected."
    assert agent.command_handler.handle.await_args_list == [
        call("!safe-mode", caller=None),
        call("!safe-mode", caller=caller),
    ]
    agent._handle_bootstrap.assert_not_awaited()


@pytest.mark.asyncio
async def test_bootstrap_non_recovery_command_returns_explicit_command_result():
    agent = _bootstrap_agent()
    agent.command_handler.handle = AsyncMock(return_value="command unexpectedly ran")

    result = await agent.process_input("!tasks")

    assert result.startswith("❌ Command unavailable during bootstrap: !tasks")
    assert "onboarding greeting" not in result
    agent.command_handler.handle.assert_not_awaited()
    agent._handle_bootstrap.assert_not_awaited()
    agent._post_response_pipeline.assert_not_awaited()


@pytest.mark.asyncio
async def test_bootstrap_passthrough_tracks_sovereign_commands(monkeypatch):
    future_command = "!future-sovereign-recovery"
    agent = _bootstrap_agent()
    monkeypatch.setattr(
        CommandHandler,
        "SOVEREIGN_COMMANDS",
        CommandHandler.SOVEREIGN_COMMANDS | {future_command},
    )
    agent.command_handler.handle = AsyncMock(return_value="future recovery handled")

    result = await agent.process_input(future_command)

    assert result == "future recovery handled"
    agent.command_handler.handle.assert_awaited_once_with(future_command, caller=None)
    agent._handle_bootstrap.assert_not_awaited()
