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


class TestRecallActionItems:

    @pytest.mark.asyncio
    async def test_returns_rows(self, feature):
        feature._db.fetchall = AsyncMock(return_value=[
            ("action_1", "msg-1", "Call mom", "pending", None, None, 0.7, "2026-04-19T10:00:00"),
            ("action_2", "msg-2", "Ship PR", "done", None, "2026-04-20", 0.9, "2026-04-19T09:00:00"),
        ])
        result = await feature.recall_action_items()
        assert result["count"] == 2
        assert result["action_items"][0]["text"] == "Call mom"
        assert result["action_items"][1]["status"] == "done"

    @pytest.mark.asyncio
    async def test_filters_by_status(self, feature):
        feature._db.fetchall = AsyncMock(return_value=[])
        await feature.recall_action_items(status="pending")
        sql = feature._db.fetchall.call_args[0][0]
        params = feature._db.fetchall.call_args[0][1]
        assert "status = ?" in sql
        assert "pending" in params

    @pytest.mark.asyncio
    async def test_rejects_invalid_status(self, feature):
        result = await feature.recall_action_items(status="nonsense")
        assert result["success"] is False
        assert "pending/done/cancelled" in result["error"]

    @pytest.mark.asyncio
    async def test_filters_by_days_window(self, feature):
        feature._db.fetchall = AsyncMock(return_value=[])
        await feature.recall_action_items(days=7)
        sql = feature._db.fetchall.call_args[0][0]
        assert "created_at >= ?" in sql

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
        feature._db.fetchone = AsyncMock(return_value=("action_1",))
        result = await feature.update_action_item(item_id="action_1", status="done")
        assert result["success"] is True
        update_call = next(
            c for c in feature._db.execute.call_args_list
            if "UPDATE action_items" in c[0][0]
        )
        sql = update_call[0][0]
        params = update_call[0][1]
        assert "status = ?" in sql
        assert "done" in params

    @pytest.mark.asyncio
    async def test_missing_item(self, feature):
        feature._db.fetchone = AsyncMock(return_value=None)
        result = await feature.update_action_item(item_id="missing", status="done")
        assert result["success"] is False
        assert "not found" in result["error"].lower()

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
        feature._db.fetchone = AsyncMock(return_value=("action_1",))
        result = await feature.update_action_item(
            item_id="action_1",
            status="pending",
            due_date="2026-05-01",
            assignee_concept_id="concept:agent-1:alice",
        )
        assert result["success"] is True
        update_call = next(
            c for c in feature._db.execute.call_args_list
            if "UPDATE action_items" in c[0][0]
        )
        sql = update_call[0][0]
        # All three fields should appear in the SET clause
        assert sql.count("?") >= 5  # 3 updates + item_id + agent_id


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
        node_other = GraphNode(
            node_id="decision:did:other:xyz",
            node_type="decision",
            label="Other agent's decision",
            properties={
                "agent_id": "did:other",
                "created_at": "2026-04-19T10:00:00",
            },
        )
        feature.agent.storage.graph.get_nodes_by_type = AsyncMock(
            return_value=[node_mine, node_other]
        )

        result = await feature.recall_decisions()
        labels = [d["label"] for d in result["decisions"]]
        assert "Move to Brooklyn" in labels
        assert "Other agent's decision" not in labels

    @pytest.mark.asyncio
    async def test_empty(self, feature):
        result = await feature.recall_decisions()
        assert result["decisions"] == []
        assert result["count"] == 0


# =============================================================================
# recall_interactions
# =============================================================================


class TestRecallInteractions:

    @pytest.mark.asyncio
    async def test_returns_mention_edges_with_properties(self, feature):
        edges = [
            Edge(
                source_id="message:agent-1:msg-1",
                target_id="concept:agent-1:alice",
                label="mentions",
                properties={"sentiment": "positive", "topics": ["weekend"]},
            ),
            Edge(
                source_id="message:agent-1:msg-2",
                target_id="concept:agent-1:alice",
                label="other_edge",
                properties={},
            ),
        ]
        feature.agent.storage.graph.get_edges = AsyncMock(return_value=edges)
        result = await feature.recall_interactions(person_concept_id="concept:agent-1:alice")
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
            node_id="concept:agent-1:alice-smith",
            node_type="concept",
            label="Alice Smith",
            properties={},
        ))
        result = await feature.confirm_person_match(
            message_id="msg-1",
            mentioned_label="alice",
            concept_id="concept:agent-1:alice-smith",
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
    async def test_unknown_concept(self, feature):
        feature.agent.storage.graph.get_node = AsyncMock(return_value=None)
        result = await feature.confirm_person_match(
            message_id="msg-1",
            mentioned_label="alice",
            concept_id="concept:agent-1:ghost",
        )
        assert result["success"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
