from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sdk.tools.result import ToolResultStatus
from kestrel_sovereign.features.todo.feature import TODO_NODE_TYPE, TodoFeature
from kestrel_sovereign.storage.async_graph_store import GraphNode


def _make_agent():
    agent = MagicMock()
    agent.did = "did:test:todo-agent"
    agent.agent_id = agent.did
    agent._active_session_id = "session-1"
    agent._get_current_turn_id = MagicMock(return_value="turn-1")
    agent.storage = MagicMock()
    agent.storage.graph = MagicMock()
    agent.storage.graph.add_node = AsyncMock()
    agent.storage.graph.add_edge = AsyncMock()
    agent.storage.graph.get_node = AsyncMock(return_value=None)
    agent.storage.graph.query_nodes_by_type_and_property = AsyncMock(return_value=[])
    return agent


async def _make_feature(agent=None):
    feature = TodoFeature(agent or _make_agent())
    await feature.initialize()
    return feature


def _todo_node(
    todo_id="todo:1",
    *,
    title="Monitor Talon job",
    status="open",
    scope="session",
    priority="normal",
    owner="did:test:todo-agent",
    agent_id="did:test:todo-agent",
    links=None,
    terminal_condition="PR merged and runtime verified",
    superseded_by=None,
):
    return GraphNode(
        node_id=todo_id,
        node_type=TODO_NODE_TYPE,
        label=title,
        properties={
            "id": todo_id,
            "agent_id": agent_id,
            "title": title,
            "description": "",
            "scope": scope,
            "status": status,
            "priority": priority,
            "owner": owner,
            "links": links or [],
            "terminal_condition": terminal_condition,
            "next_check_at": None,
            "source_turn": {"turn_id": "turn-1", "session_id": "session-1"},
            "superseded_by": superseded_by,
            "created_at": "2026-06-20T10:00:00+00:00",
            "updated_at": "2026-06-20T10:00:00+00:00",
            "completed_at": None,
        },
    )


@pytest.mark.asyncio
async def test_tools_are_discoverable():
    feature = await _make_feature()
    tool_names = {tool.name for tool in feature.get_tools()}

    assert {
        "todo_add",
        "todo_update",
        "todo_link_task",
        "todo_list",
        "todo_complete",
        "todo_rollup",
    }.issubset(tool_names)
    assert feature.promote_tools_on_startup is True


@pytest.mark.asyncio
async def test_todo_add_persists_session_scoped_todo_with_terminal_condition():
    feature = await _make_feature()

    result = await feature.todo_add(
        title="Monitor Talon job until terminal",
        description="Keep polling until a PR is ready or failed",
        scope="session",
        status="in_progress",
        priority="high",
        links=[{"type": "talon_job", "target": "job-123"}],
        terminal_condition="Talon job terminal and reviewer verification run",
        next_check_at="2026-06-20T11:00:00+00:00",
    )

    assert result.status is ToolResultStatus.OK
    feature.agent.storage.graph.add_node.assert_awaited()
    persisted = feature.agent.storage.graph.add_node.await_args[0][0]
    assert persisted.node_type == TODO_NODE_TYPE
    assert persisted.properties["scope"] == "session"
    assert persisted.properties["status"] == "in_progress"
    assert (
        persisted.properties["terminal_condition"]
        == "Talon job terminal and reviewer verification run"
    )
    assert persisted.properties["links"][0]["type"] == "talon_job"
    assert persisted.properties["source_turn"] == {"turn_id": "turn-1", "session_id": "session-1"}


@pytest.mark.asyncio
async def test_todo_add_rejects_done_status():
    feature = await _make_feature()

    result = await feature.todo_add(title="Already finished", status="done")

    assert result.status is ToolResultStatus.ERROR
    assert "cannot create a todo already marked done" in result.error
    feature.agent.storage.graph.add_node.assert_not_awaited()


@pytest.mark.asyncio
async def test_todo_update_preserves_terminal_condition_and_status_transition():
    agent = _make_agent()
    agent.storage.graph.get_node = AsyncMock(return_value=_todo_node(status="waiting"))
    feature = await _make_feature(agent)

    result = await feature.todo_update(
        todo_id="todo:1",
        status="in_progress",
        next_check_at="2026-06-20T12:00:00+00:00",
    )

    assert result.status is ToolResultStatus.OK
    persisted = agent.storage.graph.add_node.await_args[0][0]
    assert persisted.properties["status"] == "in_progress"
    assert persisted.properties["next_check_at"] == "2026-06-20T12:00:00+00:00"
    assert persisted.properties["terminal_condition"] == "PR merged and runtime verified"
    assert persisted.properties["completed_at"] is None


@pytest.mark.asyncio
async def test_todo_update_cannot_bypass_terminal_condition_with_done_status():
    agent = _make_agent()
    agent.storage.graph.get_node = AsyncMock(return_value=_todo_node(status="in_progress"))
    feature = await _make_feature(agent)

    result = await feature.todo_update(todo_id="todo:1", status="done")

    assert result.status is ToolResultStatus.ERROR
    assert "use todo_complete" in result.error
    agent.storage.graph.add_node.assert_not_awaited()


@pytest.mark.asyncio
async def test_todo_link_task_adds_link_and_edge():
    agent = _make_agent()
    agent.storage.graph.get_node = AsyncMock(return_value=_todo_node())
    feature = await _make_feature(agent)

    result = await feature.todo_link_task(
        todo_id="todo:1",
        link_type="github_issue",
        target="KestrelSovereignAI/kestrel-sovereign#1832",
        title="Add active todo queue",
        url="https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/1832",
    )

    assert result.status is ToolResultStatus.OK
    persisted = agent.storage.graph.add_node.await_args[0][0]
    assert persisted.properties["links"][0]["type"] == "github_issue"
    agent.storage.graph.add_edge.assert_awaited()
    assert agent.storage.graph.add_edge.await_args[0][:3] == (
        "todo:1",
        "github_issue:KestrelSovereignAI/kestrel-sovereign#1832",
        "linked_to",
    )


@pytest.mark.asyncio
async def test_todo_list_filters_and_excludes_done_and_superseded_by_default():
    agent = _make_agent()
    agent.storage.graph.query_nodes_by_type_and_property = AsyncMock(
        return_value=[
            _todo_node("todo:open", status="open"),
            _todo_node("todo:done", status="done"),
            _todo_node("todo:old", status="waiting", superseded_by="todo:new"),
        ]
    )
    feature = await _make_feature(agent)

    result = await feature.todo_list(scope="session", status="open", limit=25)

    assert result.status is ToolResultStatus.OK
    assert result.data["count"] == 1
    assert result.data["todos"][0]["id"] == "todo:open"
    call = agent.storage.graph.query_nodes_by_type_and_property.await_args
    assert call.args[0] == TODO_NODE_TYPE
    assert call.kwargs["filters"] == {
        "agent_id": "did:test:todo-agent",
        "scope": "session",
        "status": "open",
    }


@pytest.mark.asyncio
async def test_todo_complete_refuses_done_until_terminal_condition_satisfied():
    agent = _make_agent()
    agent.storage.graph.get_node = AsyncMock(return_value=_todo_node())
    feature = await _make_feature(agent)

    result = await feature.todo_complete(todo_id="todo:1", outcome="done")

    assert result.status is ToolResultStatus.ERROR
    assert "terminal_condition is not satisfied" in result.error
    agent.storage.graph.add_node.assert_not_awaited()


@pytest.mark.asyncio
async def test_todo_complete_marks_done_with_evidence_when_terminal_satisfied():
    agent = _make_agent()
    agent.storage.graph.get_node = AsyncMock(return_value=_todo_node())
    feature = await _make_feature(agent)

    result = await feature.todo_complete(
        todo_id="todo:1",
        outcome="done",
        evidence="PR merged and runtime checked",
        terminal_condition_satisfied=True,
    )

    assert result.status is ToolResultStatus.OK
    persisted = agent.storage.graph.add_node.await_args[0][0]
    assert persisted.properties["status"] == "done"
    assert persisted.properties["completion_evidence"] == "PR merged and runtime checked"
    assert persisted.properties["terminal_condition_satisfied"] is True
    assert persisted.properties["completed_at"] is not None


@pytest.mark.asyncio
async def test_todo_complete_can_cancel_without_terminal_satisfaction():
    agent = _make_agent()
    agent.storage.graph.get_node = AsyncMock(return_value=_todo_node())
    feature = await _make_feature(agent)

    result = await feature.todo_complete(
        todo_id="todo:1",
        outcome="cancelled",
        evidence="User stopped the work",
    )

    assert result.status is ToolResultStatus.OK
    persisted = agent.storage.graph.add_node.await_args[0][0]
    assert persisted.properties["status"] == "cancelled"


@pytest.mark.asyncio
async def test_todo_rollup_counts_active_items_and_github_talon_links():
    agent = _make_agent()
    agent.storage.graph.query_nodes_by_type_and_property = AsyncMock(
        return_value=[
            _todo_node(
                "todo:issue",
                status="in_progress",
                scope="global",
                links=[{"type": "github_issue", "target": "#1832"}],
            ),
            _todo_node(
                "todo:talon",
                status="waiting",
                scope="repo",
                links=[{"type": "talon_job", "target": "job-42"}],
            ),
            _todo_node("todo:done", status="done", scope="session"),
        ]
    )
    feature = await _make_feature(agent)

    result = await feature.todo_rollup()

    assert result.status is ToolResultStatus.OK
    assert result.data["counts"]["by_status"] == {"in_progress": 1, "waiting": 1}
    assert result.data["counts"]["by_scope"] == {"global": 1, "repo": 1}
    assert result.data["counts"]["linked_systems"]["github_issue"] == 1
    assert result.data["counts"]["linked_systems"]["talon_job"] == 1
    assert [item["id"] for item in result.data["waiting_or_in_progress"]] == [
        "todo:issue",
        "todo:talon",
    ]
