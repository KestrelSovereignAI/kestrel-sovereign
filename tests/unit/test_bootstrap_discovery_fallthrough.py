"""Regression: ``_handle_bootstrap`` falls through on discovery LLM error.

Pre-fix, ``BootstrapService.process_discovery_message`` swallowed any
``llm_service.generate_with_messages`` failure and returned a hardcoded
"I'm having trouble thinking right now…" string. The agent's
``_handle_bootstrap`` then forwarded that string back to the caller —
including OpenAI-compat clients hitting ``/v1/chat/completions``, where
it landed verbatim as the assistant's first reply. Open WebUI users on
fresh pip installs hit this on every first message: the canned string
looked like a real model response and hid the actual problem.

The new contract:

  - ``process_discovery_message`` propagates LLM errors instead of
    swallowing them. (Tested in ``tests/integration/test_bootstrap_flow.py``.)
  - ``_handle_bootstrap`` catches the propagated error in its DISCOVERY
    branch, calls ``skip_discovery`` to auto-complete bootstrap, and
    returns ``None`` so ``process_input`` falls through to the agent's
    full LLM chain on this very turn.

This test pins the second leg of the contract end-to-end via a stubbed
``KestrelAgent`` instance with the relevant attributes set by hand —
spinning up a real agent for a routing test is too heavy and noisy.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sovereign.bootstrap import BootstrapState
from kestrel_sovereign.kestrel_agent import KestrelAgent


@pytest.mark.asyncio
async def test_handle_bootstrap_falls_through_when_discovery_llm_fails():
    """An LLM error during DISCOVERY should auto-complete bootstrap and
    return ``None`` so the caller continues to normal processing."""
    agent = KestrelAgent.__new__(KestrelAgent)  # bypass __init__ — we only need the method

    bootstrap = MagicMock()
    bootstrap.get_bootstrap_state = AsyncMock(return_value=BootstrapState.DISCOVERY)
    bootstrap.process_discovery_message = AsyncMock(
        side_effect=RuntimeError("Ollama unreachable")
    )
    bootstrap.skip_discovery = AsyncMock(return_value="ok")
    agent.bootstrap_service = bootstrap

    privacy_agent = MagicMock()
    privacy_agent.add_conversation = AsyncMock()
    agent.privacy_agent = privacy_agent

    result = await agent._handle_bootstrap("hello", session_id=None)

    assert result is None, (
        "expected None so process_input falls through to the normal LLM "
        "path; got a string which would be returned to the OpenAI-compat "
        "client verbatim"
    )
    bootstrap.skip_discovery.assert_awaited_once()


@pytest.mark.asyncio
async def test_handle_bootstrap_pending_persists_user_message_and_greeting():
    """PENDING branch — the very first message ever — MUST persist
    BOTH the user's input AND the wake-up greeting to
    ``conversation_history``. Pre-fix only the greeting landed, so the
    user's first message was silently dropped and never available for
    future recall. Closes #1486.

    Surfaced after #1481/v0.21.1 enabled user-role rows in
    MemoryRetriever — once the recall path actually used user content,
    "the first turn is missing" became observable in end-to-end
    smoke tests.
    """
    agent = KestrelAgent.__new__(KestrelAgent)

    bootstrap = MagicMock()
    bootstrap.get_bootstrap_state = AsyncMock(return_value=BootstrapState.PENDING)
    bootstrap.generate_wake_up_message = AsyncMock(
        return_value="Hi, what should I call you?"
    )
    bootstrap.set_bootstrap_state = AsyncMock()
    agent.bootstrap_service = bootstrap

    privacy_agent = MagicMock()
    privacy_agent.add_conversation = AsyncMock()
    agent.privacy_agent = privacy_agent

    result = await agent._handle_bootstrap(
        "My favorite hobby is sailing.", session_id="sess-1",
    )

    # Greeting is returned to the caller as the assistant's response.
    assert result == "Hi, what should I call you?"

    # Two persistence calls: user first, then assistant.
    calls = privacy_agent.add_conversation.await_args_list
    assert len(calls) == 2, (
        f"expected user + assistant rows persisted; got {len(calls)} calls"
    )
    # Order matters: user-message persisted BEFORE greeting so the row
    # IDs reflect the actual order of the turn.
    assert calls[0].args[0] == "user"
    assert calls[0].args[1] == "My favorite hobby is sailing."
    assert calls[0].kwargs.get("session_id") == "sess-1"
    assert calls[1].args[0] == "assistant"
    assert calls[1].args[1] == "Hi, what should I call you?"
    assert calls[1].kwargs.get("session_id") == "sess-1"


@pytest.mark.asyncio
async def test_handle_bootstrap_pending_does_not_persist_user_when_greeting_fails():
    """If the wake-up-message generation itself fails, we return None so
    the caller falls through to the normal LLM path — which then persists
    the user message on its own. We must NOT double-persist here, otherwise
    the row lands twice.
    """
    agent = KestrelAgent.__new__(KestrelAgent)

    bootstrap = MagicMock()
    bootstrap.get_bootstrap_state = AsyncMock(return_value=BootstrapState.PENDING)
    bootstrap.generate_wake_up_message = AsyncMock(
        side_effect=RuntimeError("template render failed"),
    )
    bootstrap.set_bootstrap_state = AsyncMock()
    agent.bootstrap_service = bootstrap

    privacy_agent = MagicMock()
    privacy_agent.add_conversation = AsyncMock()
    agent.privacy_agent = privacy_agent

    result = await agent._handle_bootstrap("hello", session_id=None)

    assert result is None
    # State transition must NOT happen on failure (we stay PENDING for
    # the next try) AND no conversation rows get written here.
    bootstrap.set_bootstrap_state.assert_not_awaited()
    privacy_agent.add_conversation.assert_not_awaited()


@pytest.mark.asyncio
async def test_handle_bootstrap_swallows_skip_discovery_failure():
    """Even if ``skip_discovery`` itself fails on the auto-complete path,
    we still return None so the user's message reaches normal processing.
    A stuck DISCOVERY state will get cleared by the 1-hour timeout in
    ``is_bootstrap_needed``; we must not block the user's chat on that."""
    agent = KestrelAgent.__new__(KestrelAgent)

    bootstrap = MagicMock()
    bootstrap.get_bootstrap_state = AsyncMock(return_value=BootstrapState.DISCOVERY)
    bootstrap.process_discovery_message = AsyncMock(
        side_effect=RuntimeError("Ollama unreachable")
    )
    bootstrap.skip_discovery = AsyncMock(side_effect=RuntimeError("DB locked"))
    agent.bootstrap_service = bootstrap

    privacy_agent = MagicMock()
    privacy_agent.add_conversation = AsyncMock()
    agent.privacy_agent = privacy_agent

    result = await agent._handle_bootstrap("hello", session_id=None)

    assert result is None
