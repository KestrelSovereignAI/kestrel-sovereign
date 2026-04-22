"""Tests for the epistemic status layer (#680).

Covers:
- MemoryMetadata epistemic fields (claim_certainty, claim_source, temporal_validity)
- EmotionalTagger._detect_epistemic_status cue-based certainty inference
- Non-claim detection (questions/greetings → None)
- SchemaRouter epistemic provenance flow to graph nodes
- mark_superseded: node type restriction, agent ownership, supersession semantics
- recall_* include_superseded filter
- MemoryRetriever certainty weighting
- Cross-agent mutation isolation (regression guard from #676)
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest
import pytest_asyncio

from kestrel_sovereign.storage.memory_models import MemoryMetadata
from kestrel_sovereign.storage.emotional_tagger import EmotionalTagger
from kestrel_sovereign.storage.async_graph_store import GraphNode
from kestrel_sovereign.storage.schema_router import (
    ACTION_ITEM_NODE_TYPE,
    CLAIM_SHAPED_NODE_TYPES,
    DECISION_NODE_TYPE,
    SchemaRouter,
    _extract_epistemic_fields,
)


# =============================================================================
# MemoryMetadata epistemic fields
# =============================================================================


class TestMemoryMetadataEpistemicFields:

    def test_default_epistemic_fields_are_none(self):
        meta = MemoryMetadata()
        assert meta.claim_certainty is None
        assert meta.claim_source is None
        assert meta.temporal_validity is None

    def test_to_dict_includes_epistemic_fields(self):
        meta = MemoryMetadata(
            claim_certainty=0.85,
            claim_source="direct",
            temporal_validity="durable",
        )
        d = meta.to_dict()
        assert d["claim_certainty"] == 0.85
        assert d["claim_source"] == "direct"
        assert d["temporal_validity"] == "durable"

    def test_from_dict_round_trips(self):
        original = MemoryMetadata(
            claim_certainty=0.42,
            claim_source="hearsay",
            temporal_validity="ephemeral",
        )
        restored = MemoryMetadata.from_dict(original.to_dict())
        assert restored.claim_certainty == 0.42
        assert restored.claim_source == "hearsay"
        assert restored.temporal_validity == "ephemeral"

    def test_from_dict_handles_missing_epistemic_fields(self):
        """Backward compat: old metadata without epistemic fields."""
        meta = MemoryMetadata.from_dict({"importance": 0.7})
        assert meta.claim_certainty is None
        assert meta.claim_source is None
        assert meta.temporal_validity is None

    def test_merge_with_preserves_epistemic_fields(self):
        meta = MemoryMetadata(claim_certainty=0.9, claim_source="direct")
        merged = meta.merge_with({"some_existing_key": "value"})
        assert merged["claim_certainty"] == 0.9
        assert merged["claim_source"] == "direct"
        assert merged["some_existing_key"] == "value"


# =============================================================================
# EmotionalTagger._detect_epistemic_status
# =============================================================================


class TestEpistemicDetection:

    def setup_method(self):
        self.tagger = EmotionalTagger()

    def test_high_certainty_cues(self):
        result = self.tagger._detect_epistemic_status(
            "I've decided to move to Brooklyn."
        )
        assert result.get("claim_certainty") is not None
        assert result["claim_certainty"] >= 0.7

    def test_medium_certainty_cues(self):
        result = self.tagger._detect_epistemic_status(
            "I think we should use Postgres for this."
        )
        assert result.get("claim_certainty") is not None
        assert 0.4 <= result["claim_certainty"] <= 0.8

    def test_low_certainty_cues(self):
        result = self.tagger._detect_epistemic_status(
            "Maybe we could try a different approach."
        )
        assert result.get("claim_certainty") is not None
        assert result["claim_certainty"] <= 0.5

    def test_question_returns_empty(self):
        """Non-claim messages: questions should return empty dict (→ None fields)."""
        result = self.tagger._detect_epistemic_status(
            "What do you think about this approach?"
        )
        assert result == {}

    def test_greeting_returns_empty(self):
        """Non-claim messages: greetings should return empty dict (→ None fields)."""
        result = self.tagger._detect_epistemic_status("Hello!")
        assert result == {}

    def test_hearsay_source_detection(self):
        result = self.tagger._detect_epistemic_status(
            "I've heard that the team is switching to Go."
        )
        assert result.get("claim_source") == "hearsay"

    def test_observed_source_detection(self):
        result = self.tagger._detect_epistemic_status(
            "I noticed the build was failing on main."
        )
        assert result.get("claim_source") == "observed"

    def test_inferred_source_detection(self):
        result = self.tagger._detect_epistemic_status(
            "Based on the metrics, therefore the migration must be working."
        )
        assert result.get("claim_source") == "inferred"

    def test_inferred_source_deduction(self):
        result = self.tagger._detect_epistemic_status(
            "Given that the tests pass, I'm concluding the fix is correct."
        )
        assert result.get("claim_source") == "inferred"

    def test_direct_source_for_assertions(self):
        result = self.tagger._detect_epistemic_status(
            "I've decided to take the job offer."
        )
        assert result.get("claim_source") == "direct"

    def test_durable_temporal_validity(self):
        result = self.tagger._detect_epistemic_status(
            "I've decided to always use type hints from now on."
        )
        assert result.get("temporal_validity") == "durable"

    def test_moment_temporal_validity(self):
        result = self.tagger._detect_epistemic_status(
            "Right now I'm feeling pretty confident about the decision."
        )
        assert result.get("temporal_validity") == "moment"

    def test_ephemeral_temporal_validity(self):
        result = self.tagger._detect_epistemic_status(
            "For now I think we should keep the current setup until the migration."
        )
        assert result.get("temporal_validity") == "ephemeral"

    @pytest.mark.asyncio
    async def test_analyze_integrates_epistemic_fields(self):
        """Full analyze() pipeline populates epistemic fields on MemoryMetadata."""
        meta = await self.tagger.analyze(
            "I've decided to move to Brooklyn.", role="user"
        )
        assert meta.claim_certainty is not None
        assert meta.claim_certainty >= 0.7
        assert meta.claim_source == "direct"
        assert meta.temporal_validity == "durable"

    @pytest.mark.asyncio
    async def test_analyze_question_leaves_epistemic_none(self):
        """Questions should result in None epistemic fields on MemoryMetadata."""
        meta = await self.tagger.analyze(
            "What do you think about moving?", role="user"
        )
        assert meta.claim_certainty is None
        assert meta.claim_source is None
        assert meta.temporal_validity is None


# =============================================================================
# _extract_epistemic_fields helper
# =============================================================================


class TestExtractEpistemicFields:

    def test_extracts_present_fields(self):
        meta = {"claim_certainty": 0.8, "claim_source": "direct", "importance": 0.9}
        result = _extract_epistemic_fields(meta)
        assert result == {"claim_certainty": 0.8, "claim_source": "direct"}

    def test_returns_none_when_no_epistemic_fields(self):
        assert _extract_epistemic_fields({"importance": 0.5}) is None

    def test_returns_none_for_none_metadata(self):
        assert _extract_epistemic_fields(None) is None


# =============================================================================
# SchemaRouter epistemic provenance flow
# =============================================================================


def _make_mock_graph():
    graph = MagicMock()
    graph.db = MagicMock()
    graph.db.fetchall = AsyncMock(return_value=[])
    graph.db.execute = AsyncMock()
    graph.get_node = AsyncMock(return_value=None)
    graph.add_node = AsyncMock()
    graph.add_edge = AsyncMock()
    graph.get_edges = AsyncMock(return_value=[])
    return graph


@pytest_asyncio.fixture
async def router():
    graph = _make_mock_graph()
    db = MagicMock()
    db.execute = AsyncMock()
    db.fetchall = AsyncMock(return_value=[])
    db.fetchone = AsyncMock(return_value=None)
    return SchemaRouter(graph=graph, db=db, agent_id="agent-1")


class TestSchemaRouterEpistemicProvenance:

    @pytest.mark.asyncio
    async def test_decision_inherits_epistemic_from_metadata(self, router):
        metadata = {
            "claim_certainty": 0.9,
            "claim_source": "direct",
            "temporal_validity": "durable",
        }
        await router.route(
            message_id="msg-ep-1",
            content="I've decided to use Postgres.",
            concepts=[],
            role="user",
            metadata=metadata,
        )
        decision_nodes = [
            c.args[0] for c in router.graph.add_node.await_args_list
            if c.args[0].node_type == DECISION_NODE_TYPE
        ]
        assert decision_nodes
        props = decision_nodes[0].properties
        assert props["claim_certainty"] == 0.9
        assert props["claim_source"] == "direct"
        assert props["temporal_validity"] == "durable"

    @pytest.mark.asyncio
    async def test_action_item_inherits_epistemic_from_metadata(self, router):
        metadata = {
            "claim_certainty": 0.6,
            "claim_source": "inferred",
            "temporal_validity": "ephemeral",
        }
        await router.route(
            message_id="msg-ep-2",
            content="I need to update the RFC.",
            concepts=[],
            role="user",
            metadata=metadata,
        )
        action_nodes = [
            c.args[0] for c in router.graph.add_node.await_args_list
            if c.args[0].node_type == ACTION_ITEM_NODE_TYPE
        ]
        assert action_nodes
        props = action_nodes[0].properties
        assert props["claim_certainty"] == 0.6
        assert props["claim_source"] == "inferred"
        assert props["temporal_validity"] == "ephemeral"

    @pytest.mark.asyncio
    async def test_no_metadata_no_epistemic_on_nodes(self, router):
        """When metadata is None, nodes should not have epistemic fields."""
        await router.route(
            message_id="msg-ep-3",
            content="I've decided to skip the meeting.",
            concepts=[],
            role="user",
            metadata=None,
        )
        decision_nodes = [
            c.args[0] for c in router.graph.add_node.await_args_list
            if c.args[0].node_type == DECISION_NODE_TYPE
        ]
        assert decision_nodes
        props = decision_nodes[0].properties
        assert "claim_certainty" not in props


# =============================================================================
# mark_superseded
# =============================================================================


def _make_mock_agent(agent_id="agent-1"):
    """Build a mock agent with graph store for mark_superseded tests."""
    agent = MagicMock()
    agent.did = agent_id
    agent.storage = MagicMock()
    agent.storage.graph = _make_mock_graph()
    return agent


def _make_memory_feature(agent):
    """Instantiate MemoryFeature with minimal mocking."""
    from kestrel_sovereign.features.memory.feature import MemoryFeature
    feature = MemoryFeature.__new__(MemoryFeature)
    feature.agent = agent
    feature.agent_id = agent.did
    feature.storage = agent.storage
    feature._db = MagicMock()
    feature._memory_system = None
    return feature


class TestMarkSuperseded:

    @pytest.mark.asyncio
    async def test_supersede_decision_succeeds(self):
        agent = _make_mock_agent("agent-1")
        feature = _make_memory_feature(agent)

        old_node = GraphNode(
            node_id="decision:agent-1:old",
            node_type="decision",
            label="old decision",
            properties={"agent_id": "agent-1", "text": "use MySQL"},
        )
        new_node = GraphNode(
            node_id="decision:agent-1:new",
            node_type="decision",
            label="new decision",
            properties={"agent_id": "agent-1", "text": "use Postgres"},
        )

        async def _get_node(node_id):
            if node_id == old_node.node_id:
                return old_node
            if node_id == new_node.node_id:
                return new_node
            return None

        agent.storage.graph.get_node = AsyncMock(side_effect=_get_node)

        result = await feature.mark_superseded(
            old_id=old_node.node_id,
            new_id=new_node.node_id,
            reason="Changed database",
        )
        assert result["success"] is True

        # Verify supersedes edge was written
        edge_calls = agent.storage.graph.add_edge.await_args_list
        assert any(
            c.args[0] == new_node.node_id
            and c.args[1] == old_node.node_id
            and c.args[2] == "supersedes"
            for c in edge_calls
        )

        # Verify old node got superseded_by property
        node_calls = agent.storage.graph.add_node.await_args_list
        assert any(
            c.args[0].node_id == old_node.node_id
            and c.args[0].properties.get("superseded_by") == new_node.node_id
            for c in node_calls
        )

    @pytest.mark.asyncio
    async def test_supersede_action_item_succeeds(self):
        agent = _make_mock_agent("agent-1")
        feature = _make_memory_feature(agent)

        old_node = GraphNode(
            node_id="action:agent-1:old",
            node_type="action_item",
            label="old action",
            properties={"agent_id": "agent-1"},
        )
        new_node = GraphNode(
            node_id="action:agent-1:new",
            node_type="action_item",
            label="new action",
            properties={"agent_id": "agent-1"},
        )

        async def _get_node(node_id):
            if node_id == old_node.node_id:
                return old_node
            if node_id == new_node.node_id:
                return new_node
            return None

        agent.storage.graph.get_node = AsyncMock(side_effect=_get_node)

        result = await feature.mark_superseded(
            old_id=old_node.node_id,
            new_id=new_node.node_id,
        )
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_rejects_non_claim_node_type(self):
        """mark_superseded MUST restrict to claim-shaped node types."""
        agent = _make_mock_agent("agent-1")
        feature = _make_memory_feature(agent)

        concept_node = GraphNode(
            node_id="concept:agent-1:foo",
            node_type="concept",
            label="foo",
            properties={"agent_id": "agent-1"},
        )
        decision_node = GraphNode(
            node_id="decision:agent-1:bar",
            node_type="decision",
            label="bar",
            properties={"agent_id": "agent-1"},
        )

        async def _get_node(node_id):
            if node_id == concept_node.node_id:
                return concept_node
            if node_id == decision_node.node_id:
                return decision_node
            return None

        agent.storage.graph.get_node = AsyncMock(side_effect=_get_node)

        # Trying to supersede a concept node → rejected
        result = await feature.mark_superseded(
            old_id=concept_node.node_id,
            new_id=decision_node.node_id,
        )
        assert result["success"] is False
        assert "concept" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_rejects_non_claim_replacement_node(self):
        """Replacement node must also be a claim type."""
        agent = _make_mock_agent("agent-1")
        feature = _make_memory_feature(agent)

        decision_node = GraphNode(
            node_id="decision:agent-1:old",
            node_type="decision",
            label="old",
            properties={"agent_id": "agent-1"},
        )
        concept_node = GraphNode(
            node_id="concept:agent-1:new",
            node_type="concept",
            label="new",
            properties={"agent_id": "agent-1"},
        )

        async def _get_node(node_id):
            if node_id == decision_node.node_id:
                return decision_node
            if node_id == concept_node.node_id:
                return concept_node
            return None

        agent.storage.graph.get_node = AsyncMock(side_effect=_get_node)

        result = await feature.mark_superseded(
            old_id=decision_node.node_id,
            new_id=concept_node.node_id,
        )
        assert result["success"] is False
        assert "claim type" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_cross_agent_mutation_returns_not_found(self):
        """REGRESSION GUARD (#676): mark_superseded MUST enforce per-agent
        ownership. A node belonging to agent-2 must not be mutated by agent-1."""
        agent = _make_mock_agent("agent-1")
        feature = _make_memory_feature(agent)

        # Old node belongs to agent-2
        other_agent_node = GraphNode(
            node_id="decision:agent-2:secret",
            node_type="decision",
            label="their decision",
            properties={"agent_id": "agent-2", "text": "secret info"},
        )
        own_node = GraphNode(
            node_id="decision:agent-1:mine",
            node_type="decision",
            label="my decision",
            properties={"agent_id": "agent-1"},
        )

        async def _get_node(node_id):
            if node_id == other_agent_node.node_id:
                return other_agent_node
            if node_id == own_node.node_id:
                return own_node
            return None

        agent.storage.graph.get_node = AsyncMock(side_effect=_get_node)

        # agent-1 tries to supersede agent-2's node → not found
        result = await feature.mark_superseded(
            old_id=other_agent_node.node_id,
            new_id=own_node.node_id,
        )
        assert result["success"] is False
        assert "not found" in result["error"].lower()

        # No edge or node mutation should have happened
        agent.storage.graph.add_edge.assert_not_awaited()
        agent.storage.graph.add_node.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_cross_agent_new_node_returns_not_found(self):
        """Cross-agent guard also applies to the replacement (new) node."""
        agent = _make_mock_agent("agent-1")
        feature = _make_memory_feature(agent)

        own_node = GraphNode(
            node_id="decision:agent-1:mine",
            node_type="decision",
            label="my decision",
            properties={"agent_id": "agent-1"},
        )
        other_node = GraphNode(
            node_id="decision:agent-2:theirs",
            node_type="decision",
            label="their decision",
            properties={"agent_id": "agent-2"},
        )

        async def _get_node(node_id):
            if node_id == own_node.node_id:
                return own_node
            if node_id == other_node.node_id:
                return other_node
            return None

        agent.storage.graph.get_node = AsyncMock(side_effect=_get_node)

        result = await feature.mark_superseded(
            old_id=own_node.node_id,
            new_id=other_node.node_id,
        )
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_nonexistent_old_node_returns_not_found(self):
        agent = _make_mock_agent("agent-1")
        feature = _make_memory_feature(agent)
        agent.storage.graph.get_node = AsyncMock(return_value=None)

        result = await feature.mark_superseded(
            old_id="decision:agent-1:ghost",
            new_id="decision:agent-1:new",
        )
        assert result["success"] is False
        assert "not found" in result["error"].lower()


# =============================================================================
# recall_* include_superseded filter
# =============================================================================


class TestRecallSupersededFilter:

    @pytest.mark.asyncio
    async def test_recall_decisions_excludes_superseded_by_default(self):
        agent = _make_mock_agent("agent-1")
        feature = _make_memory_feature(agent)

        active_node = GraphNode(
            node_id="decision:agent-1:active",
            node_type="decision",
            label="active",
            properties={"agent_id": "agent-1", "text": "use Postgres", "created_at": "2026-04-01T00:00:00+00:00"},
        )
        superseded_node = GraphNode(
            node_id="decision:agent-1:old",
            node_type="decision",
            label="old",
            properties={
                "agent_id": "agent-1",
                "text": "use MySQL",
                "created_at": "2026-03-01T00:00:00+00:00",
                "superseded_by": "decision:agent-1:active",
            },
        )

        agent.storage.graph.query_nodes_by_type_and_property = AsyncMock(
            return_value=[active_node, superseded_node]
        )

        # Default: exclude superseded
        result = await feature.recall_decisions(limit=25, include_superseded=False)
        assert result["count"] == 1
        assert result["decisions"][0]["id"] == active_node.node_id

    @pytest.mark.asyncio
    async def test_recall_decisions_includes_superseded_when_asked(self):
        agent = _make_mock_agent("agent-1")
        feature = _make_memory_feature(agent)

        active_node = GraphNode(
            node_id="decision:agent-1:active",
            node_type="decision",
            label="active",
            properties={"agent_id": "agent-1", "text": "use Postgres", "created_at": "2026-04-01T00:00:00+00:00"},
        )
        superseded_node = GraphNode(
            node_id="decision:agent-1:old",
            node_type="decision",
            label="old",
            properties={
                "agent_id": "agent-1",
                "text": "use MySQL",
                "created_at": "2026-03-01T00:00:00+00:00",
                "superseded_by": "decision:agent-1:active",
            },
        )

        agent.storage.graph.query_nodes_by_type_and_property = AsyncMock(
            return_value=[active_node, superseded_node]
        )

        result = await feature.recall_decisions(limit=25, include_superseded=True)
        assert result["count"] == 2

    @pytest.mark.asyncio
    async def test_recall_action_items_excludes_superseded_by_default(self):
        agent = _make_mock_agent("agent-1")
        feature = _make_memory_feature(agent)

        active = GraphNode(
            node_id="action:agent-1:active",
            node_type="action_item",
            label="active",
            properties={"agent_id": "agent-1", "status": "pending", "text": "do X", "created_at": "2026-04-01T00:00:00+00:00"},
        )
        superseded = GraphNode(
            node_id="action:agent-1:old",
            node_type="action_item",
            label="old",
            properties={
                "agent_id": "agent-1",
                "status": "pending",
                "text": "do Y",
                "created_at": "2026-03-01T00:00:00+00:00",
                "superseded_by": "action:agent-1:active",
            },
        )

        agent.storage.graph.query_nodes_by_type_and_property = AsyncMock(
            return_value=[active, superseded]
        )

        result = await feature.recall_action_items(limit=25, include_superseded=False)
        assert result["count"] == 1
        assert result["action_items"][0]["id"] == active.node_id

    @pytest.mark.asyncio
    async def test_recall_returns_epistemic_fields(self):
        agent = _make_mock_agent("agent-1")
        feature = _make_memory_feature(agent)

        node = GraphNode(
            node_id="decision:agent-1:ep",
            node_type="decision",
            label="epistemic decision",
            properties={
                "agent_id": "agent-1",
                "text": "use Postgres",
                "created_at": "2026-04-01T00:00:00+00:00",
                "claim_certainty": 0.9,
                "claim_source": "direct",
                "temporal_validity": "durable",
            },
        )

        agent.storage.graph.query_nodes_by_type_and_property = AsyncMock(
            return_value=[node]
        )

        result = await feature.recall_decisions(limit=25)
        decision = result["decisions"][0]
        assert decision["claim_certainty"] == 0.9
        assert decision["claim_source"] == "direct"
        assert decision["temporal_validity"] == "durable"


# =============================================================================
# MemoryRetriever certainty weighting
# =============================================================================


class TestRetrieverCertaintyWeighting:

    def test_weights_sum_to_one(self):
        from kestrel_sovereign.storage.memory_retriever import MemoryRetriever
        total = (
            MemoryRetriever.WEIGHT_SEMANTIC
            + MemoryRetriever.WEIGHT_EMOTIONAL
            + MemoryRetriever.WEIGHT_IMPORTANCE
            + MemoryRetriever.WEIGHT_RECENCY
            + MemoryRetriever.WEIGHT_ACCESS
            + MemoryRetriever.WEIGHT_CERTAINTY
        )
        assert abs(total - 1.0) < 1e-9

    def test_certainty_score_with_value(self):
        from kestrel_sovereign.storage.memory_retriever import MemoryRetriever
        retriever = MemoryRetriever.__new__(MemoryRetriever)
        assert retriever._score_certainty({"claim_certainty": 0.9}) == 0.9

    def test_certainty_score_without_value(self):
        from kestrel_sovereign.storage.memory_retriever import MemoryRetriever
        retriever = MemoryRetriever.__new__(MemoryRetriever)
        assert retriever._score_certainty({}) == 0.5

    def test_certainty_score_none_value(self):
        from kestrel_sovereign.storage.memory_retriever import MemoryRetriever
        retriever = MemoryRetriever.__new__(MemoryRetriever)
        assert retriever._score_certainty({"claim_certainty": None}) == 0.5

    def test_high_certainty_boosts_score(self):
        """A message with high certainty should score higher than one without."""
        from kestrel_sovereign.storage.memory_retriever import MemoryRetriever
        retriever = MemoryRetriever.__new__(MemoryRetriever)
        retriever.linker = None

        base_args = dict(
            content="test content about decisions",
            query="decisions",
            emotional_context=None,
            created_at=None,
            expanded_concepts=[],
        )

        score_high = retriever._calculate_score(
            metadata={"claim_certainty": 0.95, "importance": 0.5}, **base_args
        )
        score_none = retriever._calculate_score(
            metadata={"importance": 0.5}, **base_args
        )
        assert score_high > score_none


# =============================================================================
# CLAIM_SHAPED_NODE_TYPES constant
# =============================================================================


class TestClaimShapedNodeTypes:

    def test_includes_decision_and_action_item(self):
        assert "decision" in CLAIM_SHAPED_NODE_TYPES
        assert "action_item" in CLAIM_SHAPED_NODE_TYPES

    def test_excludes_concept(self):
        assert "concept" not in CLAIM_SHAPED_NODE_TYPES


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
