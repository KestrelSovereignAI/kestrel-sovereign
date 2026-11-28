"""Unit tests for the schema-aware recall tools on MemoryFeature.

Covers: recall_action_items, update_action_item, recall_decisions,
recall_interactions, confirm_person_match.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from kestrel_sdk.tools.result import ToolResultStatus
from kestrel_sovereign.features.memory.feature import MemoryFeature
from kestrel_sovereign.storage.async_graph_store import GraphNode


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


AGENT_ID = "did:test:recall-agent"
OTHER_AGENT_ID = "did:test:other-agent"


@pytest_asyncio.fixture
async def real(tmp_path):
    """MemoryFeature over a real agent store behind the real privacy wrapper.

    The recall tools are aliases of the generic recall read (#4155), which
    reads only through PrivacyEnforcingStorage with a graph bound to the agent.
    """
    from tests.utils.recall_storage import governed_storage

    async with governed_storage(str(tmp_path / "kestrel.db"), AGENT_ID) as (storage, raw):
        agent = MagicMock()
        agent.did = AGENT_ID
        agent.agent_id = AGENT_ID
        agent.storage = storage
        f = MemoryFeature.__new__(MemoryFeature)
        f.agent = agent
        f.agent_id = AGENT_ID
        yield f, raw


def _other_agent_graph(raw):
    from kestrel_sovereign.storage.async_graph_store import AsyncGraphStore

    return AsyncGraphStore(raw.db, agent_id=OTHER_AGENT_ID)


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
    async def test_returns_own_items(self, real):
        feature, raw = real
        await raw.graph.add_node(_action_node(f"action:{AGENT_ID}:a", "Call mom"))
        await raw.graph.add_node(_action_node(
            f"action:{AGENT_ID}:b", "Ship PR", status="done",
            created_at="2026-04-19T09:00:00+00:00",
        ))
        result = await feature.recall_action_items()
        assert result.status is ToolResultStatus.OK
        assert result.data["count"] == 2
        items = result.data["action_items"]
        # Sorted by created_at desc (SQL ORDER BY)
        assert items[0]["text"] == "Call mom"
        assert items[1]["status"] == "done"
        assert items[0]["home"]["canonical_home"] == "graph"

    @pytest.mark.asyncio
    async def test_filters_by_status(self, real):
        feature, raw = real
        await raw.graph.add_node(_action_node(f"action:{AGENT_ID}:a1", "Pending one"))
        await raw.graph.add_node(_action_node(f"action:{AGENT_ID}:a2", "Done one", status="done"))
        result = await feature.recall_action_items(status="pending")
        assert result.status is ToolResultStatus.OK
        assert result.data["count"] == 1
        assert result.data["action_items"][0]["text"] == "Pending one"

    @pytest.mark.asyncio
    async def test_filters_by_assignee(self, real):
        feature, raw = real
        await raw.graph.add_node(_action_node(
            f"action:{AGENT_ID}:a1", "Call mom", assignee="concept:agent:alice"
        ))
        await raw.graph.add_node(_action_node(f"action:{AGENT_ID}:a2", "Unassigned"))
        result = await feature.recall_action_items(assignee_concept_id="concept:agent:alice")
        assert result.status is ToolResultStatus.OK
        assert result.data["count"] == 1
        assert result.data["action_items"][0]["assignee_concept_id"] == "concept:agent:alice"

    @pytest.mark.asyncio
    async def test_filters_by_days_window(self, real):
        from datetime import datetime, timezone
        feature, raw = real
        recent = datetime.now(timezone.utc).isoformat()
        await raw.graph.add_node(_action_node(f"action:{AGENT_ID}:a1", "Recent", created_at=recent))
        await raw.graph.add_node(_action_node(
            f"action:{AGENT_ID}:a2", "Old", created_at="2020-01-01T00:00:00+00:00"
        ))
        result = await feature.recall_action_items(days=7)
        assert result.status is ToolResultStatus.OK
        assert result.data["count"] == 1
        assert result.data["action_items"][0]["text"] == "Recent"

    @pytest.mark.asyncio
    async def test_filters_combine_and_stay_agent_scoped(self, real):
        """status, assignee, the days window and the agent scope all hold at once."""
        from datetime import datetime, timezone
        feature, raw = real
        recent = datetime.now(timezone.utc).isoformat()
        wanted = _action_node(
            f"action:{AGENT_ID}:hit", "Hit", created_at=recent, assignee="concept:x:alice"
        )
        await raw.graph.add_node(wanted)
        await raw.graph.add_node(_action_node(
            f"action:{AGENT_ID}:done", "Done", status="done", created_at=recent,
            assignee="concept:x:alice",
        ))
        await raw.graph.add_node(_action_node(
            f"action:{AGENT_ID}:old", "Old", created_at="2020-01-01T00:00:00+00:00",
            assignee="concept:x:alice",
        ))
        await _other_agent_graph(raw).add_node(_action_node(
            f"action:{OTHER_AGENT_ID}:theirs", "Theirs", created_at=recent,
            assignee="concept:x:alice", agent_id=OTHER_AGENT_ID,
        ))
        result = await feature.recall_action_items(
            status="pending", days=7, assignee_concept_id="concept:x:alice"
        )
        assert [item["id"] for item in result.data["action_items"]] == [wanted.node_id]

    @pytest.mark.asyncio
    async def test_rejects_invalid_status(self, feature):
        result = await feature.recall_action_items(status="nonsense")
        assert result.status is ToolResultStatus.ERROR
        assert "pending, done, cancelled" in result.error

    @pytest.mark.asyncio
    async def test_rejects_invalid_days(self, feature):
        for bad in (0, -1, 10_000):
            result = await feature.recall_action_items(days=bad)
            assert result.status is ToolResultStatus.ERROR

    @pytest.mark.asyncio
    async def test_rejects_invalid_limit(self, feature):
        for bad in (0, -5, 500):
            result = await feature.recall_action_items(limit=bad)
            assert result.status is ToolResultStatus.ERROR


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
        assert result.status is ToolResultStatus.OK
        # Must have upserted a node with the new status
        feature.agent.storage.graph.add_node.assert_awaited()
        persisted = feature.agent.storage.graph.add_node.await_args[0][0]
        assert persisted.properties["status"] == "done"

    @pytest.mark.asyncio
    async def test_missing_item(self, feature):
        feature.agent.storage.graph.get_node = AsyncMock(return_value=None)
        result = await feature.update_action_item(item_id="missing", status="done")
        assert result.status is ToolResultStatus.ERROR
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_wrong_node_type(self, feature):
        feature.agent.storage.graph.get_node = AsyncMock(return_value=GraphNode(
            node_id="some-decision", node_type="decision", label="x", properties={},
        ))
        result = await feature.update_action_item(item_id="some-decision", status="done")
        assert result.status is ToolResultStatus.ERROR
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_cross_agent_mutation_blocked(self, feature):
        """Mutating another agent's action item via known node_id must fail."""
        feature.agent.storage.graph.get_node = AsyncMock(return_value=_action_node(
            "action_x", "Their item", agent_id="did:other",
        ))
        result = await feature.update_action_item(item_id="action_x", status="done")
        assert result.status is ToolResultStatus.ERROR
        feature.agent.storage.graph.add_node.assert_not_called()

    @pytest.mark.asyncio
    async def test_rejects_invalid_status(self, feature):
        result = await feature.update_action_item(item_id="action_1", status="bogus")
        assert result.status is ToolResultStatus.ERROR

    @pytest.mark.asyncio
    async def test_no_fields_to_update(self, feature):
        result = await feature.update_action_item(item_id="action_1")
        assert result.status is ToolResultStatus.ERROR
        assert "no fields" in result.error.lower()

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
        assert result.status is ToolResultStatus.OK
        persisted = feature.agent.storage.graph.add_node.await_args[0][0]
        assert persisted.properties["status"] == "pending"
        assert persisted.properties["due_date"] == "2026-05-01"
        assert persisted.properties["assignee_concept_id"] == "concept:agent-1:alice"


# =============================================================================
# recall_decisions
# =============================================================================


class TestRecallDecisions:

    @pytest.mark.asyncio
    async def test_returns_own_decisions_only(self, real):
        feature, raw = real
        node_mine = GraphNode(
            node_id=f"decision:{AGENT_ID}:abc",
            node_type="decision",
            label="Move to Brooklyn",
            properties={
                "text": "move to brooklyn",
                "agent_id": AGENT_ID,
                "created_at": "2026-04-19T10:00:00",
                "confidence": 0.8,
            },
        )
        await raw.graph.add_node(node_mine)
        await _other_agent_graph(raw).add_node(GraphNode(
            node_id=f"decision:{OTHER_AGENT_ID}:def",
            node_type="decision",
            label="Theirs",
            properties={"text": "theirs", "agent_id": OTHER_AGENT_ID,
                        "created_at": "2026-04-20T10:00:00"},
        ))

        result = await feature.recall_decisions()
        assert result.status is ToolResultStatus.OK
        labels = [d["label"] for d in result.data["decisions"]]
        assert labels == ["Move to Brooklyn"]
        assert result.data["count"] == 1

    @pytest.mark.asyncio
    async def test_empty(self, real):
        feature, _ = real
        result = await feature.recall_decisions()
        assert result.status is ToolResultStatus.OK
        assert result.data["decisions"] == []
        assert result.data["count"] == 0

    @pytest.mark.asyncio
    async def test_reads_both_decision_types_newest_first(self, real):
        """One query over both decision types (rv-4244-1 split ``decision``)."""
        feature, raw = real
        for node_type, stamp in (("decision", "2026-04-18"), ("strategy_decision", "2026-04-19")):
            await raw.graph.add_node(GraphNode(
                node_id=f"decision:{AGENT_ID}:{node_type}",
                node_type=node_type,
                label=node_type,
                properties={"text": node_type, "agent_id": AGENT_ID,
                            "created_at": f"{stamp}T00:00:00+00:00"},
            ))
        result = await feature.recall_decisions()
        assert [d["label"] for d in result.data["decisions"]] == [
            "strategy_decision", "decision",
        ]


# =============================================================================
# recall_interactions
# =============================================================================


async def _mentioning_message(storage, raw, concept_id, label="mentions", properties=None):
    """A live conversation row, its message node, and an edge to ``concept_id``."""
    row_id = await storage.add_conversation("user", "a message")
    node_id = f"message:{AGENT_ID}:{row_id}"
    await raw.graph.add_node(GraphNode(
        node_id=node_id, node_type="message", label=f"Message {row_id}",
        properties={"message_id": str(row_id), "agent_id": AGENT_ID,
                    "created_at": f"2026-04-1{row_id}T00:00:00+00:00"},
    ))
    await raw.graph.add_edge(node_id, concept_id, label, properties=properties or {})
    return node_id


async def _concept(raw, concept_id):
    await raw.graph.add_node(GraphNode(
        node_id=concept_id, node_type="concept", label="alice",
        properties={"agent_id": AGENT_ID, "category": "proper_noun"},
    ))


class TestRecallInteractions:

    @pytest.mark.asyncio
    async def test_returns_mention_edges_with_properties(self, real):
        feature, raw = real
        concept = f"concept:{AGENT_ID}:alice"
        await _concept(raw, concept)
        storage = feature.agent.storage
        await _mentioning_message(
            storage, raw, concept, properties={"sentiment": "positive", "topics": ["weekend"]}
        )
        await _mentioning_message(storage, raw, concept, label="other_edge")
        result = await feature.recall_interactions(person_concept_id=concept)
        # Only 'mentions' edges are interactions
        assert result.status is ToolResultStatus.OK
        assert result.data["count"] == 1
        assert result.data["interactions"][0]["properties"]["sentiment"] == "positive"

    @pytest.mark.asyncio
    async def test_empty(self, real):
        feature, _ = real
        result = await feature.recall_interactions(person_concept_id="concept:agent-1:nobody")
        assert result.status is ToolResultStatus.OK
        assert result.data["interactions"] == []


# =============================================================================
# confirm_person_match
# =============================================================================


class TestConfirmPersonMatch:

    @pytest.mark.asyncio
    async def test_resolves_ambiguous_match(self, feature):
        nodes = {
            "concept:did:test:recall-agent:alice-smith": GraphNode(
                node_id="concept:did:test:recall-agent:alice-smith",
                node_type="concept", label="Alice Smith", properties={},
            ),
            "concept:did:test:recall-agent:alice": GraphNode(
                node_id="concept:did:test:recall-agent:alice",
                node_type="concept", label="alice", properties={"category": "proper_noun"},
            ),
        }
        feature.agent.storage.graph.get_node = AsyncMock(side_effect=lambda node_id: nodes.get(node_id))
        feature.agent.storage.graph.delete_edge = AsyncMock()
        result = await feature.confirm_person_match(
            message_id="msg-1",
            mentioned_label="alice",
            concept_id="concept:did:test:recall-agent:alice-smith",
        )
        assert result.status is ToolResultStatus.OK
        # A mentions edge must be written with the canonical concept, then
        # the answer recorded as an alias from the ambiguous node so the
        # next mention resolves without asking again (#3259).
        writes = feature.agent.storage.graph.add_edge.await_args_list
        assert [c.args[2] for c in writes] == ["mentions"]
        (canonical,) = writes
        assert canonical.args[1] == "concept:did:test:recall-agent:alice-smith"
        props = canonical.kwargs.get("properties") or {}
        assert props.get("confirmed") is True
        assert props.get("resolved_from") == "alice"
        # The answer is recorded on the mention's own node, single-valued,
        # so the next mention resolves without asking and a correction
        # overwrites (#3259).
        recorded = feature.agent.storage.graph.add_node.await_args.args[0]
        assert recorded.node_id == "concept:did:test:recall-agent:alice"
        assert recorded.properties["resolved_to"] == "concept:did:test:recall-agent:alice-smith"
        assert recorded.properties["category"] == "proper_noun"  # the rest of the node is kept

    @pytest.mark.asyncio
    async def test_attempts_to_remove_ambiguous_edge(self, feature):
        """Critical: confirmation must issue a delete for the ambiguous
        edge. Round-5 finding: AsyncGraphStore.delete_edge does not
        report affected rows or raise on no-op, so we phrase the
        outcome as 'remove attempted' rather than 'removed'."""
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
        assert result.status is ToolResultStatus.OK
        assert result.data["ambiguous_remove_attempted"] is True
        # Honesty: confirmation must NOT claim the edge was removed —
        # only that delete_edge was issued.
        assert "removed" not in result.confirmation.lower() or (
            "not verified" in result.confirmation.lower()
        )
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
        assert result.status is ToolResultStatus.OK
        assert result.data["ambiguous_remove_attempted"] is False
        feature.agent.storage.graph.delete_edge.assert_not_called()

    @pytest.mark.asyncio
    async def test_unknown_concept(self, feature):
        feature.agent.storage.graph.get_node = AsyncMock(return_value=None)
        result = await feature.confirm_person_match(
            message_id="msg-1",
            mentioned_label="alice",
            concept_id="concept:agent-1:ghost",
        )
        assert result.status is ToolResultStatus.ERROR


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
        assert result.status is ToolResultStatus.ERROR
        assert "iso-8601" in result.error.lower()

    @pytest.mark.asyncio
    async def test_accepts_iso_date(self, feature):
        feature.agent.storage.graph.get_node = AsyncMock(return_value=_action_node(
            "action_1", "t"
        ))
        result = await feature.update_action_item(
            item_id="action_1", due_date="2026-05-01"
        )
        assert result.status is ToolResultStatus.OK

    @pytest.mark.asyncio
    async def test_accepts_iso_datetime(self, feature):
        feature.agent.storage.graph.get_node = AsyncMock(return_value=_action_node(
            "action_1", "t"
        ))
        result = await feature.update_action_item(
            item_id="action_1", due_date="2026-05-01T14:00:00+00:00"
        )
        assert result.status is ToolResultStatus.OK


# =============================================================================
# recall_interactions agent scoping
# =============================================================================


class TestRecallInteractionsAgentScope:

    @pytest.mark.asyncio
    async def test_filters_out_other_agents_message_edges(self, real):
        """The recall tool must only return this agent's own messages' mentions."""
        feature, raw = real
        other = _other_agent_graph(raw)
        concept = f"concept:{OTHER_AGENT_ID}:alice"
        await other.add_node(GraphNode(
            node_id=concept, node_type="concept", label="alice",
            properties={"agent_id": OTHER_AGENT_ID},
        ))
        message = f"message:{OTHER_AGENT_ID}:their-msg"
        await other.add_node(GraphNode(
            node_id=message, node_type="message", label="Message their-msg",
            properties={"message_id": "1", "agent_id": OTHER_AGENT_ID},
        ))
        await other.add_edge(message, concept, "mentions", properties={"sentiment": "negative"})
        result = await feature.recall_interactions(person_concept_id=concept)
        assert result.status is ToolResultStatus.OK
        assert result.data["count"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestConfirmPersonMatchRecordingFailure:
    @pytest.mark.asyncio
    async def test_a_recording_failure_is_partial_and_the_cleanup_still_runs(self, feature):
        """The canonical edge is in place, so a failure to record the answer
        for future mentions is PARTIAL, not failed, and the ambiguous-edge
        removal is still attempted (#3259 round 2)."""
        nodes = {
            "concept:did:test:recall-agent:alice-smith": GraphNode(
                node_id="concept:did:test:recall-agent:alice-smith",
                node_type="concept", label="Alice Smith", properties={},
            ),
            "concept:did:test:recall-agent:alice": GraphNode(
                node_id="concept:did:test:recall-agent:alice",
                node_type="concept", label="alice", properties={"category": "proper_noun"},
            ),
        }
        feature.agent.storage.graph.get_node = AsyncMock(side_effect=lambda node_id: nodes.get(node_id))
        feature.agent.storage.graph.add_node = AsyncMock(side_effect=RuntimeError("node write refused"))
        feature.agent.storage.graph.delete_edge = AsyncMock()
        result = await feature.confirm_person_match(
            message_id="msg-1",
            mentioned_label="alice",
            concept_id="concept:did:test:recall-agent:alice-smith",
        )
        assert result.status is ToolResultStatus.PARTIAL
        assert "will ask again" in result.error and "node write refused" in result.error
        assert result.data["future_mentions_recorded"] is False
        assert result.data["ambiguous_remove_attempted"] is True
        feature.agent.storage.graph.delete_edge.assert_awaited_once()
        assert [c.args[2] for c in feature.agent.storage.graph.add_edge.await_args_list] == ["mentions"]
