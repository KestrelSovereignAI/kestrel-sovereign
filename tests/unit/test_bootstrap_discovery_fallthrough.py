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
