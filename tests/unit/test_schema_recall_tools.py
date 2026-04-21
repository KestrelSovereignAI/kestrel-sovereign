"""Unit tests for the schema-aware recall tools on MemoryFeature.

Covers: recall_action_items, update_action_item, recall_decisions,
recall_interactions, confirm_person_match.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from kestrel_sovereign.features.memory.feature import MemoryFeature
from kestrel_sovereign.storage.async_graph_store import Edge, GraphNode


def _make_mock_db():
    db = MagicMock()
    db.execute = AsyncMock()
    db.fetchall = AsyncMock(return_value=[])
    db.fetchone = AsyncMock(return_value=None)
    db.table_exists = AsyncMock(return_value=True)
    return db


def _make_agent(db=None):
    agent = MagicMock()
    agent.did = "did:test:recall-agent"
    agent.agent_id = agent.did
    agent.features = {}
    agent.storage = MagicMock()
    agent.storage.db = db or _make_mock_db()
    agent.storage.graph = MagicMock()
    agent.storage.graph.add_edge = AsyncMock()
    agent.storage.graph.add_node = AsyncMock()
    agent.storage.graph.get_node = AsyncMock(return_value=None)
    agent.storage.graph.get_edges = AsyncMock(return_value=[])
    agent.storage.graph.get_nodes_by_type = AsyncMock(return_value=[])
    agent.storage.graph.query_nodes_by_type_and_property = AsyncMock(return_value=[])
    agent.bootstrap_service = MagicMock()
    agent.bootstrap_service.agent_data_path = None
    return agent


@pytest_asyncio.fixture
async def feature():
    agent = _make_agent()
    f = MemoryFeature(agent)
    await f.initialize()
    return f


# =============================================================================
# recall_action_items
# =============================================================================


def _action_node(node_id, text, status="pending", agent_id="did:test:recall-agent",
                 created_at="2026-04-19T10:00:00+00:00", assignee=None, due_date=None,
                 source_message_id=None):
    return GraphNode(
        node_id=node_id,
        node_type="action_item",
        label=text[:120],
        properties={
            "text": text,
            "status": status,
            "agent_id": agent_id,
            "created_at": created_at,
            "assignee_concept_id": assignee,
            "due_date": due_date,
            "source_message_id": source_message_id,
            "confidence": 0.7,
        },
    )


class TestRecallActionItems:

    @pytest.mark.asyncio
    async def test_returns_own_items(self, feature):
        feature.agent.storage.graph.query_nodes_by_type_and_property = AsyncMock(return_value=[
            _action_node("action:did:test:recall-agent:a", "Call mom"),
            _action_node("action:did:test:recall-agent:b", "Ship PR", status="done",
                         created_at="2026-04-19T09:00:00+00:00"),
        ])
        result = await feature.recall_action_items()
        assert result["count"] == 2
        # Sorted by created_at desc (SQL ORDER BY)
        assert result["action_items"][0]["text"] == "Call mom"
        assert result["action_items"][1]["status"] == "done"

    @pytest.mark.asyncio
    async def test_filters_by_status(self, feature):
        feature.agent.storage.graph.query_nodes_by_type_and_property = AsyncMock(return_value=[
            _action_node("a1", "Pending one", status="pending"),
        ])
        result = await feature.recall_action_items(status="pending")
        assert result["count"] == 1
        assert result["action_items"][0]["text"] == "Pending one"

    @pytest.mark.asyncio
    async def test_filters_by_assignee(self, feature):
        feature.agent.storage.graph.query_nodes_by_type_and_property = AsyncMock(return_value=[
            _action_node("a1", "Call mom", assignee="concept:agent:alice"),
        ])
        result = await feature.recall_action_items(assignee_concept_id="concept:agent:alice")
        assert result["count"] == 1
        assert result["action_items"][0]["assignee_concept_id"] == "concept:agent:alice"

    @pytest.mark.asyncio
    async def test_filters_by_days_window(self, feature):
        from datetime import datetime, timezone
        recent = datetime.now(timezone.utc).isoformat()
        feature.agent.storage.graph.query_nodes_by_type_and_property = AsyncMock(return_value=[
            _action_node("a1", "Recent", created_at=recent),
        ])
        result = await feature.recall_action_items(days=7)
        assert result["count"] == 1
        assert result["action_items"][0]["text"] == "Recent"

    @pytest.mark.asyncio
    async def test_passes_filters_to_query(self, feature):
        """Verify that agent_id, status, and created_since are pushed into SQL filters."""
        feature.agent.storage.graph.query_nodes_by_type_and_property = AsyncMock(return_value=[])
        await feature.recall_action_items(status="pending", days=7, assignee_concept_id="concept:x:alice")
        call = feature.agent.storage.graph.query_nodes_by_type_and_property.await_args
        assert call.args[0] == "action_item"
        filters = call.kwargs.get("filters") or call.args[1] if len(call.args) > 1 else call.kwargs["filters"]
        assert filters["agent_id"] == "did:test:recall-agent"
        assert filters["status"] == "pending"
        assert filters["assignee_concept_id"] == "concept:x:alice"
        assert call.kwargs.get("created_since") is not None

    @pytest.mark.asyncio
    async def test_rejects_invalid_status(self, feature):
        result = await feature.recall_action_items(status="nonsense")
        assert result["success"] is False
        assert "pending/done/cancelled" in result["error"]

    @pytest.mark.asyncio
    async def test_rejects_invalid_days(self, feature):
        for bad in (0, -1, 10_000):
            result = await feature.recall_action_items(days=bad)
            assert result["success"] is False

    @pytest.mark.asyncio
    async def test_rejects_invalid_limit(self, feature):
        for bad in (0, -5, 500):
            result = await feature.recall_action_items(limit=bad)
            assert result["success"] is False


# =============================================================================
# update_action_item
# =============================================================================


class TestUpdateActionItem:

    @pytest.mark.asyncio
    async def test_updates_status(self, feature):
        feature.agent.storage.graph.get_node = AsyncMock(return_value=_action_node(
            "action_1", "Call mom"
        ))
        result = await feature.update_action_item(item_id="action_1", status="done")
        assert result["success"] is True
        # Must have upserted a node with the new status
        feature.agent.storage.graph.add_node.assert_awaited()
        persisted = feature.agent.storage.graph.add_node.await_args[0][0]
        assert persisted.properties["status"] == "done"

    @pytest.mark.asyncio
    async def test_missing_item(self, feature):
        feature.agent.storage.graph.get_node = AsyncMock(return_value=None)
        result = await feature.update_action_item(item_id="missing", status="done")
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_wrong_node_type(self, feature):
        feature.agent.storage.graph.get_node = AsyncMock(return_value=GraphNode(
            node_id="some-decision", node_type="decision", label="x", properties={},
        ))
        result = await feature.update_action_item(item_id="some-decision", status="done")
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_cross_agent_mutation_blocked(self, feature):
        """Mutating another agent's action item via known node_id must fail."""
        feature.agent.storage.graph.get_node = AsyncMock(return_value=_action_node(
            "action_x", "Their item", agent_id="did:other",
        ))
        result = await feature.update_action_item(item_id="action_x", status="done")
        assert result["success"] is False
        feature.agent.storage.graph.add_node.assert_not_called()

    @pytest.mark.asyncio
    async def test_rejects_invalid_status(self, feature):
        result = await feature.update_action_item(item_id="action_1", status="bogus")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_no_fields_to_update(self, feature):
        result = await feature.update_action_item(item_id="action_1")
        assert result["success"] is False
        assert "no fields" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_multi_field_update(self, feature):
        feature.agent.storage.graph.get_node = AsyncMock(return_value=_action_node(
            "action_1", "Do thing"
        ))
        result = await feature.update_action_item(
            item_id="action_1",
            status="pending",
            due_date="2026-05-01",
            assignee_concept_id="concept:agent-1:alice",
        )
        assert result["success"] is True
        persisted = feature.agent.storage.graph.add_node.await_args[0][0]
        assert persisted.properties["status"] == "pending"
        assert persisted.properties["due_date"] == "2026-05-01"
        assert persisted.properties["assignee_concept_id"] == "concept:agent-1:alice"


# =============================================================================
# recall_decisions
# =============================================================================


class TestRecallDecisions:

    @pytest.mark.asyncio
    async def test_returns_own_decisions_only(self, feature):
        node_mine = GraphNode(
            node_id="decision:did:test:recall-agent:abc",
            node_type="decision",
            label="Move to Brooklyn",
            properties={
                "text": "move to brooklyn",
                "agent_id": "did:test:recall-agent",
                "created_at": "2026-04-19T10:00:00",
                "confidence": 0.8,
            },
        )
        # query_nodes_by_type_and_property filters by agent_id in SQL,
        # so only the agent's own nodes are returned.
        feature.agent.storage.graph.query_nodes_by_type_and_property = AsyncMock(
            return_value=[node_mine]
        )

        result = await feature.recall_decisions()
        labels = [d["label"] for d in result["decisions"]]
        assert "Move to Brooklyn" in labels
        assert result["count"] == 1

    @pytest.mark.asyncio
    async def test_empty(self, feature):
        result = await feature.recall_decisions()
        assert result["decisions"] == []
        assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_passes_agent_filter_to_query(self, feature):
        """Verify agent_id filter is pushed into SQL."""
        feature.agent.storage.graph.query_nodes_by_type_and_property = AsyncMock(return_value=[])
        await feature.recall_decisions()
        call = feature.agent.storage.graph.query_nodes_by_type_and_property.await_args
        assert call.args[0] == "decision"
        filters = call.kwargs.get("filters") or call.args[1] if len(call.args) > 1 else call.kwargs["filters"]
        assert filters["agent_id"] == "did:test:recall-agent"


# =============================================================================
# recall_interactions
# =============================================================================


class TestRecallInteractions:

    @pytest.mark.asyncio
    async def test_returns_mention_edges_with_properties(self, feature):
        edges = [
            Edge(
                source_id="message:did:test:recall-agent:msg-1",
                target_id="concept:did:test:recall-agent:alice",
                label="mentions",
                properties={"sentiment": "positive", "topics": ["weekend"]},
            ),
            Edge(
                source_id="message:did:test:recall-agent:msg-2",
                target_id="concept:did:test:recall-agent:alice",
                label="other_edge",
                properties={},
            ),
        ]
        feature.agent.storage.graph.get_edges = AsyncMock(return_value=edges)
        result = await feature.recall_interactions(person_concept_id="concept:did:test:recall-agent:alice")
        # Only 'mentions' edges are interactions
        assert result["count"] == 1
        assert result["interactions"][0]["properties"]["sentiment"] == "positive"

    @pytest.mark.asyncio
    async def test_empty(self, feature):
        result = await feature.recall_interactions(person_concept_id="concept:agent-1:nobody")
        assert result["interactions"] == []


# =============================================================================
# confirm_person_match
# =============================================================================


class TestConfirmPersonMatch:

    @pytest.mark.asyncio
    async def test_resolves_ambiguous_match(self, feature):
        feature.agent.storage.graph.get_node = AsyncMock(return_value=GraphNode(
            node_id="concept:did:test:recall-agent:alice-smith",
            node_type="concept",
            label="Alice Smith",
            properties={},
        ))
        feature.agent.storage.graph.delete_edge = AsyncMock()
        result = await feature.confirm_person_match(
            message_id="msg-1",
            mentioned_label="alice",
            concept_id="concept:did:test:recall-agent:alice-smith",
        )
        assert result["success"] is True
        # A mentions edge must be written with the canonical concept
        feature.agent.storage.graph.add_edge.assert_awaited()
        call = feature.agent.storage.graph.add_edge.await_args
        assert call.args[2] == "mentions"
        props = call.kwargs.get("properties") or {}
        assert props.get("confirmed") is True
        assert props.get("resolved_from") == "alice"

    @pytest.mark.asyncio
    async def test_actually_removes_ambiguous_edge(self, feature):
        """Critical: confirmation must supersede the ambiguous edge, not
        accumulate parallel edges. After confirm, recall on the ambiguous
        label-concept must not return this message."""
        feature.agent.storage.graph.get_node = AsyncMock(return_value=GraphNode(
            node_id="concept:did:test:recall-agent:alice-smith",
            node_type="concept",
            label="Alice Smith",
            properties={},
        ))
        feature.agent.storage.graph.delete_edge = AsyncMock()
        result = await feature.confirm_person_match(
            message_id="msg-1",
            mentioned_label="alice",
            concept_id="concept:did:test:recall-agent:alice-smith",
        )
        assert result["success"] is True
        assert result["ambiguous_edge_removed"] is True
        # delete_edge called with the ambiguous (guessed) target
        feature.agent.storage.graph.delete_edge.assert_awaited_once()
        del_call = feature.agent.storage.graph.delete_edge.await_args
        assert del_call.args[0] == "message:did:test:recall-agent:msg-1"
        assert del_call.args[1] == "concept:did:test:recall-agent:alice"
        assert del_call.args[2] == "mentions"

    @pytest.mark.asyncio
    async def test_no_self_delete_when_ambiguous_matches_canonical(self, feature):
        """If the label-concept and confirmed concept happen to resolve
        to the same node id, we must not delete the edge we just wrote."""
        feature.agent.storage.graph.get_node = AsyncMock(return_value=GraphNode(
            node_id="concept:did:test:recall-agent:alice",
            node_type="concept",
            label="Alice",
            properties={},
        ))
        feature.agent.storage.graph.delete_edge = AsyncMock()
        result = await feature.confirm_person_match(
            message_id="msg-1",
            mentioned_label="alice",
            concept_id="concept:did:test:recall-agent:alice",
        )
        assert result["success"] is True
        assert result["ambiguous_edge_removed"] is False
        feature.agent.storage.graph.delete_edge.assert_not_called()

    @pytest.mark.asyncio
    async def test_unknown_concept(self, feature):
        feature.agent.storage.graph.get_node = AsyncMock(return_value=None)
        result = await feature.confirm_person_match(
            message_id="msg-1",
            mentioned_label="alice",
            concept_id="concept:agent-1:ghost",
        )
        assert result["success"] is False


# =============================================================================
# due_date validation
# =============================================================================


class TestDueDateValidation:

    @pytest.mark.asyncio
    async def test_rejects_invalid_due_date(self, feature):
        feature.agent.storage.graph.get_node = AsyncMock(return_value=_action_node(
            "action_1", "t"
        ))
        result = await feature.update_action_item(
            item_id="action_1", due_date="not-a-date"
        )
        assert result["success"] is False
        assert "iso-8601" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_accepts_iso_date(self, feature):
        feature.agent.storage.graph.get_node = AsyncMock(return_value=_action_node(
            "action_1", "t"
        ))
        result = await feature.update_action_item(
            item_id="action_1", due_date="2026-05-01"
        )
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_accepts_iso_datetime(self, feature):
        feature.agent.storage.graph.get_node = AsyncMock(return_value=_action_node(
            "action_1", "t"
        ))
        result = await feature.update_action_item(
            item_id="action_1", due_date="2026-05-01T14:00:00+00:00"
        )
        assert result["success"] is True


# =============================================================================
# recall_interactions agent scoping
# =============================================================================


class TestRecallInteractionsAgentScope:

    @pytest.mark.asyncio
    async def test_filters_out_other_agents_message_edges(self, feature):
        """Person concept ids can be shared across agents in theory; the
        recall tool must only return edges whose source message node is
        prefixed with this agent's id."""
        edges = [
            Edge(
                source_id="message:did:test:recall-agent:my-msg",
                target_id="concept:x:alice",
                label="mentions",
                properties={"sentiment": "positive"},
            ),
            Edge(
                source_id="message:did:other-agent:their-msg",
                target_id="concept:x:alice",
                label="mentions",
                properties={"sentiment": "negative"},
            ),
        ]
        feature.agent.storage.graph.get_edges = AsyncMock(return_value=edges)
        result = await feature.recall_interactions(person_concept_id="concept:x:alice")
        assert result["count"] == 1
        assert result["interactions"][0]["message_node_id"].endswith("my-msg")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
