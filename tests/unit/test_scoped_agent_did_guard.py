"""One guard for "the DID this agent's self-scoped rows are bound to".

Four sites scoped a shared table to the calling agent and each carried
its own copy of the check (observability #3215, the A2A task list, the
consent log and audit anchors #3229/#3230); they had drifted — one gated
on truthiness alone and would bind a non-string value as a query
parameter. `resolve_scoped_agent_did` is the single copy.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from kestrel_sovereign.features.storage_access import (
    AgentIdentityUnavailable,
    resolve_scoped_agent_did,
)


def test_returns_the_agents_did():
    assert resolve_scoped_agent_did(SimpleNamespace(did="did:test:x")) == "did:test:x"
    mock = MagicMock()
    mock.did = "did:test:explicit"
    assert resolve_scoped_agent_did(mock) == "did:test:explicit"


@pytest.mark.parametrize(
    "agent",
    [
        pytest.param(SimpleNamespace(did=None), id="none"),
        pytest.param(SimpleNamespace(did=""), id="empty"),
        pytest.param(SimpleNamespace(did=123), id="non-string"),
        pytest.param(SimpleNamespace(agent_name="only-a-display-name"), id="absent"),
        pytest.param(MagicMock(), id="magicmock-fabrication"),
        pytest.param(None, id="no-agent"),
    ],
)
def test_refuses_anything_that_is_not_a_did(agent):
    """Not "unscoped": a store gating on ``if agent_id:`` reads every row
    for an empty string, and a MagicMock's fabricated attribute is truthy."""
    with pytest.raises(AgentIdentityUnavailable, match="identity"):
        resolve_scoped_agent_did(agent)
    assert issubclass(AgentIdentityUnavailable, RuntimeError)


# ---------------------------------------------------------------------------
# The `!tasks` command routes through the guard (it had no coverage before)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tasks_command_scopes_by_the_guarded_did():
    from unittest.mock import AsyncMock

    from kestrel_sovereign.command_handler import CommandHandler

    agent = MagicMock()
    agent.did = "did:test:recipient"
    task_manager = MagicMock()
    task_manager.task_store.list_tasks = AsyncMock(return_value=[])
    handler = CommandHandler(agent, task_manager=task_manager)

    assert await handler._cmd_tasks("!tasks") == "📋 No tasks found"
    task_manager.task_store.list_tasks.assert_awaited_once_with(
        recipient_agent_id="did:test:recipient", limit=10
    )


@pytest.mark.asyncio
async def test_tasks_command_refuses_without_a_did():
    from unittest.mock import AsyncMock

    from kestrel_sovereign.command_handler import CommandHandler

    agent = MagicMock()  # a fabricated `did`, not an identity
    task_manager = MagicMock()
    task_manager.task_store.list_tasks = AsyncMock(return_value=[])
    handler = CommandHandler(agent, task_manager=task_manager)

    assert await handler._cmd_tasks("!tasks") == "❌ Task recipient identity unavailable"
    task_manager.task_store.list_tasks.assert_not_awaited()
