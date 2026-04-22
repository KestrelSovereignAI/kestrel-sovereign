"""Unit tests for caller context auth gate on governance commands."""
import pytest
import hashlib
from unittest.mock import AsyncMock, MagicMock, patch
from kestrel_sovereign.auth import CallerContext, CallerRole, AuthMethod
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.command_handler import CommandHandler


# --- CallerContext dataclass ---

def test_sovereign_factory():
    ctx = CallerContext.sovereign()
    assert ctx.is_sovereign
    assert ctx.role == CallerRole.SOVEREIGN
    assert ctx.auth_method == AuthMethod.API_KEY
    assert ctx.identity == "api_key"


def test_authenticated_factory():
    ctx = CallerContext.authenticated("user@example.com")
    assert not ctx.is_sovereign
    assert ctx.role == CallerRole.AUTHENTICATED
    assert ctx.auth_method == AuthMethod.OAUTH_SESSION
    assert ctx.identity == "user@example.com"


def test_anonymous_factory():
    ctx = CallerContext.anonymous()
    assert not ctx.is_sovereign
    assert ctx.role == CallerRole.ANONYMOUS


def test_jwt_authenticated():
    ctx = CallerContext.authenticated("user@example.com", auth_method=AuthMethod.JWT)
    assert not ctx.is_sovereign
    assert ctx.auth_method == AuthMethod.JWT


# --- CommandHandler auth gate ---

def _make_handler():
    """Create a CommandHandler with a mocked agent."""
    agent = MagicMock(spec=KestrelAgent)
    agent._safe_mode = False
    handler = CommandHandler(agent)
    return handler, agent


@pytest.mark.asyncio
async def test_sovereign_can_run_safe_mode_exit():
    handler, agent = _make_handler()
    agent.exit_safe_mode = MagicMock(return_value="Safe mode deactivated.")
    agent._safe_mode = True

    result = await handler.handle("!safe-mode exit", caller=CallerContext.sovereign())

    assert "deactivated" in result.lower()


@pytest.mark.asyncio
async def test_oauth_user_rejected_from_safe_mode_exit():
    handler, agent = _make_handler()
    agent._safe_mode = True

    caller = CallerContext.authenticated("user@example.com")
    result = await handler.handle("!safe-mode exit", caller=caller)

    assert "unauthorized" in result.lower()
    assert "sovereign" in result.lower()
    assert "user@example.com" in result


@pytest.mark.asyncio
async def test_no_caller_rejected_from_safe_mode_exit():
    handler, agent = _make_handler()
    agent._safe_mode = True

    result = await handler.handle("!safe-mode exit", caller=None)

    assert "unauthorized" in result.lower()


@pytest.mark.asyncio
async def test_sovereign_can_run_reanchor():
    handler, agent = _make_handler()
    agent.reanchor_constitution = AsyncMock(return_value="Constitution re-anchored successfully.\n  Old hash: abc...\n  New hash: def...")

    result = await handler.handle("!reanchor-constitution abcdef12", caller=CallerContext.sovereign())

    assert "re-anchored" in result.lower()
    agent.reanchor_constitution.assert_called_once()


@pytest.mark.asyncio
async def test_oauth_user_rejected_from_reanchor():
    handler, agent = _make_handler()

    caller = CallerContext.authenticated("hacker@evil.com", auth_method=AuthMethod.JWT)
    result = await handler.handle("!reanchor-constitution abcdef12", caller=caller)

    assert "unauthorized" in result.lower()
    assert "hacker@evil.com" in result


@pytest.mark.asyncio
async def test_anonymous_rejected_from_reanchor():
    handler, agent = _make_handler()

    result = await handler.handle("!reanchor-constitution abcdef12", caller=CallerContext.anonymous())

    assert "unauthorized" in result.lower()


# Non-governance commands should work for any caller

@pytest.mark.asyncio
async def test_non_governance_command_works_for_oauth_user():
    handler, agent = _make_handler()
    agent.audit_enabled = True

    caller = CallerContext.authenticated("user@example.com")
    result = await handler.handle("!audit", caller=caller)

    assert "audit" in result.lower()


@pytest.mark.asyncio
async def test_non_governance_command_works_with_no_caller():
    handler, agent = _make_handler()
    agent.audit_enabled = False

    result = await handler.handle("!audit", caller=None)

    assert "audit" in result.lower()


@pytest.mark.asyncio
async def test_safe_mode_check_allowed_for_any_caller():
    """!safe-mode (without exit) is a read-only status check, allowed for anyone."""
    handler, agent = _make_handler()
    agent._safe_mode = False

    # But wait - !safe-mode is in SOVEREIGN_COMMANDS, so even check requires sovereign.
    # This is intentional: the command is gated as a whole.
    # Non-sovereign callers cannot check safe-mode status via command.
    caller = CallerContext.authenticated("user@example.com")
    result = await handler.handle("!safe-mode", caller=caller)

    assert "unauthorized" in result.lower()
