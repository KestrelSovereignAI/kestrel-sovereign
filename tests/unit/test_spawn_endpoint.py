"""Tests for the inlined spawn panel endpoint.

Inlined from the archived ``kestrel-feature-spawn`` package — see
``kestrel_sovereign/endpoints/spawn.py``. Covers both the
agent-attached AgentManager case (single-agent mode) and the
``request.app.state.agent_manager`` fallback (multi-agent mode).
The fallback path was missing in the original package and was
caught by codex during the inline (#1149 round 3).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from kestrel_sovereign.endpoints import spawn as spawn_endpoints


def _make_request(*, agent_manager=None):
    """Build a minimal FastAPI Request stand-in with an app.state."""
    request = MagicMock()
    request.app.state.agent_manager = agent_manager
    return request


@pytest.mark.asyncio
async def test_get_spawn_children_returns_empty_when_no_manager(monkeypatch):
    """No AgentManager attached anywhere — empty payload, not crash."""
    agent = SimpleNamespace(agent_id="parent-did", _agent_manager=None)
    request = _make_request(agent_manager=None)
    monkeypatch.setattr(spawn_endpoints, "get_agent", lambda r: agent)

    result = await spawn_endpoints.get_spawn_children(request)
    assert result == {
        "children": [],
        "count": 0,
        "delegation_chain": {},
        "history": [],
    }


@pytest.mark.asyncio
async def test_get_spawn_children_uses_agent_attached_manager(monkeypatch):
    """Single-agent mode — manager is on agent._agent_manager."""
    manager = MagicMock()
    manager.get_children.return_value = []
    manager._lifecycle = None
    agent = SimpleNamespace(agent_id="parent-did", _agent_manager=manager)
    request = _make_request(agent_manager=None)
    monkeypatch.setattr(spawn_endpoints, "get_agent", lambda r: agent)

    result = await spawn_endpoints.get_spawn_children(request)
    assert result["count"] == 0
    # The agent's manager was consulted (top-level call + delegation chain recursion).
    manager.get_children.assert_called_with("parent-did")
    assert manager.get_children.call_count >= 1


@pytest.mark.asyncio
async def test_get_spawn_children_falls_back_to_app_state_manager(monkeypatch):
    """Multi-agent mode — agent has no manager attached, but
    ``request.app.state.agent_manager`` does. Without the fallback
    introduced in #1149 round 3, the panel reported empty even when
    the app-level manager held children/lifecycle state.
    """
    manager = MagicMock()
    manager.get_children.return_value = []
    manager._lifecycle = None
    # Agent has no manager; app.state does.
    agent = SimpleNamespace(agent_id="parent-did", _agent_manager=None)
    request = _make_request(agent_manager=manager)
    monkeypatch.setattr(spawn_endpoints, "get_agent", lambda r: agent)

    result = await spawn_endpoints.get_spawn_children(request)
    assert result["count"] == 0
    # The app-level manager was consulted via fallback (top + delegation chain)
    manager.get_children.assert_called_with("parent-did")
    assert manager.get_children.call_count >= 1


@pytest.mark.asyncio
async def test_spawn_history_filtered_by_parent_did(monkeypatch):
    """In multi-agent mode the lifecycle's ``_tracked`` and
    ``_results`` are shared across all loaded parents. The spawn
    history endpoint must filter by ``agent.agent_id`` so opening
    A's panel never shows B's children. Codex round 4 of #1149
    caught this leak.
    """
    # Lifecycle holds active spawns from BOTH parent A and parent B.
    tracked_a = SimpleNamespace(
        child_name="child-of-a",
        child_did="did:child:a",
        parent_did="did:parent:A",
        started_at="2026-05-09T10:00:00Z",
        result=None,
    )
    tracked_b = SimpleNamespace(
        child_name="child-of-b",
        child_did="did:child:b",
        parent_did="did:parent:B",
        started_at="2026-05-09T11:00:00Z",
        result=None,
    )

    # And one terminated result from each.
    from kestrel_sovereign.spawn.lifecycle import SpawnResult, SpawnStatus
    from decimal import Decimal
    result_a = SpawnResult(
        child_name="finished-a",
        child_did="did:child:a-old",
        status=SpawnStatus.COMPLETED,
        started_at="2026-05-09T09:00:00Z",
        parent_did="did:parent:A",
        budget_consumed=Decimal("0"),
    )
    result_b = SpawnResult(
        child_name="finished-b",
        child_did="did:child:b-old",
        status=SpawnStatus.COMPLETED,
        started_at="2026-05-09T08:00:00Z",
        parent_did="did:parent:B",
        budget_consumed=Decimal("0"),
    )

    lifecycle = SimpleNamespace(
        _tracked={"child-of-a": tracked_a, "child-of-b": tracked_b},
        _results={"finished-a": result_a, "finished-b": result_b},
    )
    manager = MagicMock()
    manager._lifecycle = lifecycle
    manager.get_children.return_value = []

    # Request comes from agent A.
    agent_a = SimpleNamespace(agent_id="did:parent:A", _agent_manager=None)
    request = _make_request(agent_manager=manager)
    monkeypatch.setattr(spawn_endpoints, "get_agent", lambda r: agent_a)

    result = await spawn_endpoints.get_spawn_children(request)
    history_names = {h["child_name"] for h in result["history"]}
    # Only A's children appear; B's are filtered out.
    assert history_names == {"child-of-a", "finished-a"}, (
        f"Expected only parent A's history, got {history_names}"
    )


def test_get_agent_manager_helper_prefers_agent_then_app_state():
    """Direct unit test of the helper — agent.attached wins over app.state."""
    agent_mgr = MagicMock(name="agent-attached")
    app_mgr = MagicMock(name="app-state")
    agent = SimpleNamespace(_agent_manager=agent_mgr)
    request = _make_request(agent_manager=app_mgr)

    # When agent has one, that wins
    assert spawn_endpoints._get_agent_manager(agent, request=request) is agent_mgr

    # When agent has none, fall back to app.state
    agent_no_mgr = SimpleNamespace(_agent_manager=None)
    assert spawn_endpoints._get_agent_manager(agent_no_mgr, request=request) is app_mgr

    # When neither has one, return None
    request_no_state = _make_request(agent_manager=None)
    assert spawn_endpoints._get_agent_manager(agent_no_mgr, request=request_no_state) is None
