"""Unit tests for the epistemic status layer (#650).

Covers:
- MemoryMetadata epistemic fields (backward compat)
- EmotionalTagger cue-based certainty/source inference
- SchemaRouter epistemic fields on graph nodes
- SchemaRouter.mark_superseded helper
- Recall tools: include_superseded filter behavior
- MemoryRetriever certainty scoring
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from kestrel_sovereign.storage.memory_models import (
    ClaimSource,
    DEFAULT_CERTAINTY_BY_SOURCE,
    MemoryMetadata,
    TemporalValidity,
)
from kestrel_sovereign.storage.emotional_tagger import EmotionalTagger
from kestrel_sovereign.storage.async_graph_store import Edge, GraphNode
from kestrel_sovereign.storage.schema_router import (
    ACTION_ITEM_NODE_TYPE,
    DECISION_NODE_TYPE,
    SchemaRouter,
)
from kestrel_sovereign.storage.memory_retriever import MemoryRetriever
from kestrel_sovereign.features.memory.feature import MemoryFeature


# =============================================================================
# MemoryMetadata epistemic fields
# =============================================================================


class TestMemoryMetadataEpistemic:

    def test_defaults_are_none(self):
        meta = MemoryMetadata()
        assert meta.claim_certainty is None
        assert meta.claim_source is None
        assert meta.temporal_validity is None

    def test_round_trip_to_dict_from_dict(self):
        meta = MemoryMetadata(
            claim_certainty=0.85,
            claim_source="direct",
            temporal_validity="durable",
        )
        d = meta.to_dict()
        assert d["claim_certainty"] == 0.85
        assert d["claim_source"] == "direct"
        assert d["temporal_validity"] == "durable"

        restored = MemoryMetadata.from_dict(d)
        assert restored.claim_certainty == 0.85
        assert restored.claim_source == "direct"
        assert restored.temporal_validity == "durable"

    def test_backward_compat_missing_fields(self):
        """Old metadata dicts without epistemic fields should load fine."""
        old = {"emotional_valence": 0.5, "importance": 0.7}
        meta = MemoryMetadata.from_dict(old)
        assert meta.claim_certainty is None
        assert meta.claim_source is None
        assert meta.temporal_validity is None
        assert meta.emotional_valence == 0.5

    def test_merge_preserves_existing_keys(self):
        meta = MemoryMetadata(claim_certainty=0.9, claim_source="direct")
        merged = meta.merge_with({"enc": True, "session_id": "xyz"})
        assert merged["enc"] is True
        assert merged["claim_certainty"] == 0.9


class TestClaimSourceEnum:

    def test_values(self):
        assert ClaimSource.DIRECT.value == "direct"
        assert ClaimSource.OBSERVED.value == "observed"
        assert ClaimSource.INFERRED.value == "inferred"
        assert ClaimSource.HEARSAY.value == "hearsay"

    def test_default_certainties(self):
        assert DEFAULT_CERTAINTY_BY_SOURCE["direct"] == 0.85
        assert DEFAULT_CERTAINTY_BY_SOURCE["hearsay"] == 0.35


class TestTemporalValidityEnum:

    def test_values(self):
        assert TemporalValidity.DURABLE.value == "durable"
        assert TemporalValidity.EPHEMERAL.value == "ephemeral"
        assert TemporalValidity.MOMENT.value == "moment"


# =============================================================================
# EmotionalTagger epistemic cue detection
# =============================================================================


class TestEmotionalTaggerEpistemic:

    @pytest.fixture
    def tagger(self):
        return EmotionalTagger()

    @pytest.mark.asyncio
    async def test_direct_statement_high_certainty(self, tagger):
        result = await tagger.analyze("I've decided to move to Brooklyn", "user")
        assert result.claim_source == "direct"
        assert result.claim_certainty == 0.85

    @pytest.mark.asyncio
    async def test_hedge_i_think_lowers_certainty(self, tagger):
        result = await tagger.analyze("I think I might move to Brooklyn", "user")
        assert result.claim_source == "direct"
        # "I think" (-0.15) + "might" (-0.10) = 0.85 - 0.25 = 0.60
        assert result.claim_certainty < 0.85
        assert result.claim_certainty <= 0.65

    @pytest.mark.asyncio
    async def test_maybe_lowers_certainty(self, tagger):
        result = await tagger.analyze("Maybe I'll try the new restaurant", "user")
        assert result.claim_source == "direct"
        assert result.claim_certainty == 0.65  # 0.85 - 0.20

    @pytest.mark.asyncio
    async def test_hearsay_shifts_source(self, tagger):
        result = await tagger.analyze(
            "Apparently Sarah is moving to Brooklyn", "user"
        )
        assert result.claim_source == "hearsay"
        assert result.claim_certainty <= DEFAULT_CERTAINTY_BY_SOURCE["hearsay"]

    @pytest.mark.asyncio
    async def test_i_heard_shifts_to_hearsay(self, tagger):
        result = await tagger.analyze("I heard that the office is closing", "user")
        assert result.claim_source == "hearsay"

    @pytest.mark.asyncio
    async def test_someone_told_me_is_hearsay(self, tagger):
        result = await tagger.analyze("Someone told me the project is cancelled", "user")
        assert result.claim_source == "hearsay"

    @pytest.mark.asyncio
    async def test_assistant_messages_get_no_epistemic(self, tagger):
        result = await tagger.analyze("I'll help you with that.", "assistant")
        assert result.claim_source is None
        assert result.claim_certainty is None
        assert result.temporal_validity is None

    @pytest.mark.asyncio
    async def test_ephemeral_temporal_validity(self, tagger):
        result = await tagger.analyze("I'm feeling tired right now", "user")
        assert result.temporal_validity == "ephemeral"

    @pytest.mark.asyncio
    async def test_moment_temporal_validity(self, tagger):
        result = await tagger.analyze("This just happened: the server crashed", "user")
        assert result.temporal_validity == "moment"

    @pytest.mark.asyncio
    async def test_no_temporal_cue_is_none(self, tagger):
        result = await tagger.analyze("My sister lives in Portland", "user")
        assert result.temporal_validity is None

    @pytest.mark.asyncio
    async def test_certainty_clamps_to_zero(self, tagger):
        """Stacking multiple hedges should not go below 0.0."""
        result = await tagger.analyze(
            "I guess maybe I'm not sure, possibly, I think", "user"
        )
        assert result.claim_certainty >= 0.0

    @pytest.mark.asyncio
    async def test_batch_includes_epistemic(self, tagger):
        results = await tagger.analyze_batch([
            {"content": "I've decided to stay", "role": "user"},
            {"content": "Maybe I'll change my mind", "role": "user"},
        ])
        assert results[0].claim_certainty == 0.85
        assert results[1].claim_certainty < 0.85


# =============================================================================
# SchemaRouter epistemic fields on graph nodes
# =============================================================================


def _make_mock_graph(person_rows=None):
    graph = MagicMock()
    graph.db = MagicMock()
    graph.db.fetchall = AsyncMock(return_value=person_rows or [])
    graph.db.execute = AsyncMock()
    graph.get_node = AsyncMock(return_value=None)
    graph.add_node = AsyncMock()
    graph.add_edge = AsyncMock()
    graph.get_edges = AsyncMock(return_value=[])
    return graph


@pytest_asyncio.fixture
async def router():
    graph = _make_mock_graph([])
    db = MagicMock()
    db.execute = AsyncMock()
    db.fetchall = AsyncMock(return_value=[])
    db.fetchone = AsyncMock(return_value=None)
    return SchemaRouter(graph=graph, db=db, agent_id="agent-1")


class TestSchemaRouterEpistemic:

    @pytest.mark.asyncio
    async def test_decision_nodes_carry_epistemic_fields(self, router):
        await router.route(
            message_id="msg-1",
            content="I've decided to move to Brooklyn.",
            concepts=[],
            role="user",
        )
        decision_nodes = [
            c.args[0] for c in router.graph.add_node.await_args_list
            if c.args[0].node_type == DECISION_NODE_TYPE
        ]
        assert decision_nodes
        props = decision_nodes[0].properties
        assert props["claim_certainty"] == DEFAULT_CERTAINTY_BY_SOURCE["direct"]
        assert props["claim_source"] == "direct"
        assert props["temporal_validity"] == "durable"
        assert props["superseded_by"] is None
        assert props["contradicts"] == []

    @pytest.mark.asyncio
    async def test_action_item_nodes_carry_epistemic_fields(self, router):
        await router.route(
            message_id="msg-2",
            content="I need to call the doctor tomorrow.",
            concepts=[],
            role="user",
        )
        action_nodes = [
            c.args[0] for c in router.graph.add_node.await_args_list
            if c.args[0].node_type == ACTION_ITEM_NODE_TYPE
        ]
        assert action_nodes
        props = action_nodes[0].properties
        assert props["claim_certainty"] == DEFAULT_CERTAINTY_BY_SOURCE["direct"]
        assert props["claim_source"] == "direct"
        assert props["superseded_by"] is None
        assert props["contradicts"] == []


# =============================================================================
# SchemaRouter.mark_superseded
# =============================================================================


class TestMarkSuperseded:

    @pytest.mark.asyncio
    async def test_writes_supersedes_edge_and_updates_old_node(self, router):
        old_node = GraphNode(
            node_id="decision:agent-1:old",
            node_type="decision",
            label="Move to Brooklyn",
            properties={
                "agent_id": "agent-1",
                "text": "move to brooklyn",
                "superseded_by": None,
                "contradicts": [],
            },
        )
        new_node = GraphNode(
            node_id="decision:agent-1:new",
            node_type="decision",
            label="Stay in Manhattan",
            properties={
                "agent_id": "agent-1",
                "text": "stay in manhattan",
            },
        )

        async def _get_node(node_id):
            if node_id == "decision:agent-1:old":
                return old_node
            if node_id == "decision:agent-1:new":
                return new_node
            return None

        router.graph.get_node = AsyncMock(side_effect=_get_node)

        await router.mark_superseded(
            "decision:agent-1:old",
            "decision:agent-1:new",
            "Changed mind about moving",
        )

        # Check edge written: new → old with label "supersedes"
        edge_calls = router.graph.add_edge.await_args_list
        supersedes_edges = [
            c for c in edge_calls if c.args[2] == "supersedes"
        ]
        assert len(supersedes_edges) == 1
        edge_call = supersedes_edges[0]
        assert edge_call.args[0] == "decision:agent-1:new"
        assert edge_call.args[1] == "decision:agent-1:old"
        assert edge_call.kwargs["properties"]["reason"] == "Changed mind about moving"

        # Check old node updated with superseded_by and contradicts
        node_calls = router.graph.add_node.await_args_list
        old_updates = [
            c.args[0] for c in node_calls
            if c.args[0].node_id == "decision:agent-1:old"
        ]
        assert old_updates
        updated_props = old_updates[-1].properties
        assert updated_props["superseded_by"] == "decision:agent-1:new"
        assert "decision:agent-1:new" in updated_props["contradicts"]

    @pytest.mark.asyncio
    async def test_raises_on_missing_old_node(self, router):
        router.graph.get_node = AsyncMock(return_value=None)
        with pytest.raises(ValueError, match="Old node"):
            await router.mark_superseded("missing", "new", "reason")

    @pytest.mark.asyncio
    async def test_raises_on_missing_new_node(self, router):
        old_node = GraphNode(
            node_id="old", node_type="decision", label="x",
            properties={"agent_id": "agent-1"},
        )

        async def _get_node(node_id):
            return old_node if node_id == "old" else None

        router.graph.get_node = AsyncMock(side_effect=_get_node)
        with pytest.raises(ValueError, match="New node"):
            await router.mark_superseded("old", "missing", "reason")

    @pytest.mark.asyncio
    async def test_raises_on_cross_agent(self, router):
        old_node = GraphNode(
            node_id="old", node_type="decision", label="x",
            properties={"agent_id": "other-agent"},
        )
        new_node = GraphNode(
            node_id="new", node_type="decision", label="y",
            properties={"agent_id": "agent-1"},
        )

        async def _get_node(node_id):
            return {"old": old_node, "new": new_node}.get(node_id)

        router.graph.get_node = AsyncMock(side_effect=_get_node)
        with pytest.raises(ValueError, match="does not belong"):
            await router.mark_superseded("old", "new", "reason")


# =============================================================================
# Recall tools: include_superseded filter
# =============================================================================


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


def _decision_node(node_id, text, agent_id="did:test:recall-agent",
                    superseded_by=None, created_at="2026-04-19T10:00:00+00:00"):
    return GraphNode(
        node_id=node_id,
        node_type="decision",
        label=text[:120],
        properties={
            "text": text,
            "agent_id": agent_id,
            "created_at": created_at,
            "confidence": 0.7,
            "claim_certainty": 0.85,
            "claim_source": "direct",
            "superseded_by": superseded_by,
            "contradicts": [],
        },
    )


def _action_node(node_id, text, agent_id="did:test:recall-agent",
                  superseded_by=None, created_at="2026-04-19T10:00:00+00:00"):
    return GraphNode(
        node_id=node_id,
        node_type="action_item",
        label=text[:120],
        properties={
            "text": text,
            "status": "pending",
            "agent_id": agent_id,
            "created_at": created_at,
            "confidence": 0.7,
            "claim_certainty": 0.85,
            "claim_source": "direct",
            "superseded_by": superseded_by,
            "contradicts": [],
        },
    )


class TestRecallDecisionsSuperseded:

    @pytest.mark.asyncio
    async def test_filters_superseded_by_default(self, feature):
        feature.agent.storage.graph.get_nodes_by_type = AsyncMock(return_value=[
            _decision_node("d1", "Move to Brooklyn"),
            _decision_node("d2", "Stay in Manhattan", superseded_by="d1"),
        ])
        result = await feature.recall_decisions()
        assert result["count"] == 1
        assert result["decisions"][0]["text"] == "Move to Brooklyn"

    @pytest.mark.asyncio
    async def test_includes_superseded_when_requested(self, feature):
        feature.agent.storage.graph.get_nodes_by_type = AsyncMock(return_value=[
            _decision_node("d1", "Move to Brooklyn"),
            _decision_node("d2", "Stay in Manhattan", superseded_by="d1"),
        ])
        result = await feature.recall_decisions(include_superseded=True)
        assert result["count"] == 2

    @pytest.mark.asyncio
    async def test_decision_output_includes_epistemic_fields(self, feature):
        feature.agent.storage.graph.get_nodes_by_type = AsyncMock(return_value=[
            _decision_node("d1", "Move to Brooklyn"),
        ])
        result = await feature.recall_decisions()
        d = result["decisions"][0]
        assert d["claim_certainty"] == 0.85
        assert d["claim_source"] == "direct"
        assert d["superseded_by"] is None


class TestRecallActionItemsSuperseded:

    @pytest.mark.asyncio
    async def test_filters_superseded_by_default(self, feature):
        feature.agent.storage.graph.get_nodes_by_type = AsyncMock(return_value=[
            _action_node("a1", "Call doctor"),
            _action_node("a2", "Call clinic", superseded_by="a1"),
        ])
        result = await feature.recall_action_items()
        assert result["count"] == 1
        assert result["action_items"][0]["text"] == "Call doctor"

    @pytest.mark.asyncio
    async def test_includes_superseded_when_requested(self, feature):
        feature.agent.storage.graph.get_nodes_by_type = AsyncMock(return_value=[
            _action_node("a1", "Call doctor"),
            _action_node("a2", "Call clinic", superseded_by="a1"),
        ])
        result = await feature.recall_action_items(include_superseded=True)
        assert result["count"] == 2

    @pytest.mark.asyncio
    async def test_action_output_includes_epistemic_fields(self, feature):
        feature.agent.storage.graph.get_nodes_by_type = AsyncMock(return_value=[
            _action_node("a1", "Call doctor"),
        ])
        result = await feature.recall_action_items()
        item = result["action_items"][0]
        assert item["claim_certainty"] == 0.85
        assert item["claim_source"] == "direct"
        assert item["superseded_by"] is None


class TestRecallInteractionsSuperseded:

    @pytest.mark.asyncio
    async def test_filters_superseded_interactions(self, feature):
        edges = [
            Edge(
                source_id="message:did:test:recall-agent:msg-1",
                target_id="concept:did:test:recall-agent:alice",
                label="mentions",
                properties={"sentiment": "positive"},
            ),
            Edge(
                source_id="message:did:test:recall-agent:msg-2",
                target_id="concept:did:test:recall-agent:alice",
                label="mentions",
                properties={"sentiment": "negative", "superseded_by": "msg-3"},
            ),
        ]
        feature.agent.storage.graph.get_edges = AsyncMock(return_value=edges)
        result = await feature.recall_interactions(
            person_concept_id="concept:did:test:recall-agent:alice"
        )
        assert result["count"] == 1
        assert result["interactions"][0]["properties"]["sentiment"] == "positive"

    @pytest.mark.asyncio
    async def test_includes_superseded_interactions(self, feature):
        edges = [
            Edge(
                source_id="message:did:test:recall-agent:msg-1",
                target_id="concept:did:test:recall-agent:alice",
                label="mentions",
                properties={"sentiment": "positive"},
            ),
            Edge(
                source_id="message:did:test:recall-agent:msg-2",
                target_id="concept:did:test:recall-agent:alice",
                label="mentions",
                properties={"sentiment": "negative", "superseded_by": "msg-3"},
            ),
        ]
        feature.agent.storage.graph.get_edges = AsyncMock(return_value=edges)
        result = await feature.recall_interactions(
            person_concept_id="concept:did:test:recall-agent:alice",
            include_superseded=True,
        )
        assert result["count"] == 2


# =============================================================================
# mark_claim_superseded tool
# =============================================================================


class TestMarkClaimSupersededTool:

    @pytest.mark.asyncio
    async def test_success(self, feature):
        old_node = GraphNode(
            node_id="decision:did:test:recall-agent:old",
            node_type="decision",
            label="Old decision",
            properties={
                "agent_id": "did:test:recall-agent",
                "superseded_by": None,
                "contradicts": [],
            },
        )
        new_node = GraphNode(
            node_id="decision:did:test:recall-agent:new",
            node_type="decision",
            label="New decision",
            properties={"agent_id": "did:test:recall-agent"},
        )

        async def _get_node(node_id):
            if node_id == old_node.node_id:
                return old_node
            if node_id == new_node.node_id:
                return new_node
            return None

        feature.agent.storage.graph.get_node = AsyncMock(side_effect=_get_node)
        result = await feature.mark_claim_superseded(
            old_id=old_node.node_id,
            new_id=new_node.node_id,
            reason="Changed my mind",
        )
        assert result["success"] is True
        assert result["old_id"] == old_node.node_id
        assert result["new_id"] == new_node.node_id

    @pytest.mark.asyncio
    async def test_missing_old_node(self, feature):
        feature.agent.storage.graph.get_node = AsyncMock(return_value=None)
        result = await feature.mark_claim_superseded(
            old_id="missing", new_id="new", reason="test"
        )
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_empty_reason(self, feature):
        result = await feature.mark_claim_superseded(
            old_id="old", new_id="new", reason=""
        )
        assert result["success"] is False
        assert "reason" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_no_graph_store(self, feature):
        feature.agent.storage = MagicMock(spec=[])  # No graph attribute
        result = await feature.mark_claim_superseded(
            old_id="old", new_id="new", reason="test"
        )
        assert result["success"] is False


# =============================================================================
# MemoryRetriever certainty scoring
# =============================================================================


class TestRetrieverCertaintyScoring:

    def test_certainty_score_returns_value(self):
        retriever = MemoryRetriever(None, None)
        assert retriever._score_certainty({"claim_certainty": 0.85}) == 0.85
        assert retriever._score_certainty({"claim_certainty": 0.35}) == 0.35

    def test_certainty_score_neutral_when_missing(self):
        retriever = MemoryRetriever(None, None)
        assert retriever._score_certainty({}) == 0.5
        assert retriever._score_certainty({"claim_certainty": None}) == 0.5

    def test_weights_sum_to_one(self):
        retriever = MemoryRetriever(None, None)
        total = (
            retriever.WEIGHT_SEMANTIC
            + retriever.WEIGHT_EMOTIONAL
            + retriever.WEIGHT_IMPORTANCE
            + retriever.WEIGHT_RECENCY
            + retriever.WEIGHT_ACCESS
            + retriever.WEIGHT_CERTAINTY
        )
        assert abs(total - 1.0) < 0.001

    def test_high_certainty_scores_higher(self):
        """Messages with high claim_certainty should score slightly higher."""
        retriever = MemoryRetriever(None, None)
        base_meta = {
            "emotional_valence": 0.0,
            "importance": 0.5,
            "access_count": 0,
        }

        high = dict(base_meta, claim_certainty=0.95)
        low = dict(base_meta, claim_certainty=0.20)

        score_high = retriever._calculate_score(
            content="I decided to move",
            query="move",
            metadata=high,
            emotional_context=None,
            created_at=None,
            expanded_concepts=[],
        )
        score_low = retriever._calculate_score(
            content="I decided to move",
            query="move",
            metadata=low,
            emotional_context=None,
            created_at=None,
            expanded_concepts=[],
        )
        assert score_high > score_low


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
