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
