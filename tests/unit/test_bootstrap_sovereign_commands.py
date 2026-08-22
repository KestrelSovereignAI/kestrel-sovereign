"""Regression coverage for command routing while bootstrap is required."""

from contextlib import asynccontextmanager
from types import MethodType
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from kestrel_sovereign.auth import CallerContext
from kestrel_sovereign.command_handler import CommandHandler
from kestrel_sovereign.command_policy import (
    RECOVERY_COMMAND_POLICY,
    RECOVERY_COMMANDS,
    SAFE_MODE_COMMANDS,
    SOVEREIGN_COMMANDS,
)
from kestrel_sovereign.kestrel_agent import KestrelAgent


def _bootstrap_agent() -> KestrelAgent:
    """Build the smallest initialized agent that exercises ``process_input``."""
    agent = KestrelAgent.__new__(KestrelAgent)
    agent.storage = object()
    agent._safe_mode = False
    agent._constitution_audit_pending = False
    agent.did = "did:test:bootstrap"
    agent.privacy_agent = MagicMock()
    agent.privacy_agent.get_status.return_value = "Privacy: normal"
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
async def test_bootstrap_verify_constitution_returns_blocked_diagnostic():
    agent = _bootstrap_agent()
    del agent._genesis_audit_cognition_block
    agent._ensure_genesis_audit_ready = AsyncMock(
        side_effect=AssertionError("verification command must bypass genesis cognition")
    )

    async def _blocked_audit(_agent):
        return None, "audit marker unavailable", False

    agent._run_explicit_constitution_audit = MethodType(_blocked_audit, agent)

    result = await agent.process_input("!verify-constitution")

    assert result == (
        "🚨 Constitution verification could not start because its durable "
        "audit marker was unavailable. Agent remains in SAFE MODE."
    )
    agent._ensure_genesis_audit_ready.assert_not_awaited()
    agent._handle_bootstrap.assert_not_awaited()


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("!status", "Agent ID: did:test:bootstrap\nPrivacy: normal"),
        ("!help", "Kestrel Agent Commands"),
    ],
)
@pytest.mark.asyncio
async def test_bootstrap_allows_read_only_diagnostic_commands(command, expected):
    agent = _bootstrap_agent()

    result = await agent.process_input(command)

    assert expected in result
    agent._handle_bootstrap.assert_not_awaited()


@pytest.mark.parametrize(
    "command",
    ["!get-privacy-mode", "!privacy-status", "!bootstrap-status"],
)
@pytest.mark.asyncio
async def test_bootstrap_allows_established_readiness_diagnostics(command):
    agent = _bootstrap_agent()
    agent.command_handler.handle = AsyncMock(return_value="diagnostic handled")

    result = await agent.process_input(command)

    assert result == "diagnostic handled"
    agent.command_handler.handle.assert_awaited_once_with(command, caller=None)
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
async def test_ordinary_prompt_containing_command_tokens_stays_in_onboarding():
    agent = _bootstrap_agent()
    agent.command_handler.handle = AsyncMock(return_value="command unexpectedly ran")
    prompt = "Please explain !status and !reanchor-constitution before we begin."

    result = await agent.process_input(prompt)

    assert result == "onboarding greeting"
    agent.command_handler.handle.assert_not_awaited()
    agent._handle_bootstrap.assert_awaited_once()
    agent._post_response_pipeline.assert_awaited_once()


@pytest.mark.asyncio
async def test_one_shot_legacy_policy_cannot_split_routing_from_authorization():
    agent = _bootstrap_agent()
    one_shot_sovereign_commands = iter(["!reanchor-constitution"])
    agent.command_handler.SOVEREIGN_COMMANDS = one_shot_sovereign_commands

    result = await agent.process_input("!reanchor-constitution")

    assert result.startswith(
        "🚨 Unauthorized: !reanchor-constitution requires sovereign authority"
    )
    assert next(one_shot_sovereign_commands) == "!reanchor-constitution"
    agent._handle_bootstrap.assert_not_awaited()


def test_recovery_policy_is_complete_immutable_and_authority_derived():
    assert RECOVERY_COMMANDS == frozenset(
        {
            "!verify-constitution",
            "!status",
            "!help",
            "!safe-mode",
            "!reanchor-constitution",
            "!get-privacy-mode",
            "!privacy-status",
            "!bootstrap-status",
        }
    )
    assert SAFE_MODE_COMMANDS == frozenset(
        {
            "!verify-constitution",
            "!status",
            "!help",
            "!safe-mode",
            "!reanchor-constitution",
        }
    )
    assert SOVEREIGN_COMMANDS == frozenset(
        {"!safe-mode", "!reanchor-constitution"}
    )
    assert SOVEREIGN_COMMANDS == frozenset(
        command
        for command, rule in RECOVERY_COMMAND_POLICY.items()
        if rule.requires_sovereign
    )
    with pytest.raises(TypeError):
        RECOVERY_COMMAND_POLICY["!status"] = True
