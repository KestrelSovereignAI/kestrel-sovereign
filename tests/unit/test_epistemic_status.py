"""Tests for epistemic status features (issue #670).

Covers all five must-fix items:
1. mark_superseded() rejects non-claim node types
2. Epistemic fields flow from message metadata to claim nodes
3. Interaction supersession not claimed in tool descriptions
4. contradicted_by semantics match docs (direction: "this node is contradicted BY those")
5. _detect_epistemic_status() docstring matches implementation
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sovereign.storage.async_graph_store import GraphNode
from kestrel_sovereign.storage.emotional_tagger import EmotionalTagger
from kestrel_sovereign.storage.memory_models import (
    ClaimSource,
    DEFAULT_CERTAINTY_BY_SOURCE,
    MemoryMetadata,
    TemporalValidity,
)
from kestrel_sovereign.storage.schema_router import (
    ACTION_ITEM_NODE_TYPE,
    CLAIM_NODE_TYPES,
    DECISION_NODE_TYPE,
    SchemaRouter,
    mark_superseded,
)


# =============================================================================
# Helpers
# =============================================================================


def _make_mock_graph(nodes=None):
    """Build a mock graph store pre-loaded with optional nodes."""
    node_store = {n.node_id: n for n in (nodes or [])}
    graph = MagicMock()
    graph.db = MagicMock()
    graph.db.fetchall = AsyncMock(return_value=[])
    graph.db.execute = AsyncMock()
    graph.db.execute_commit = AsyncMock()
    graph.get_node = AsyncMock(side_effect=lambda nid: node_store.get(nid))
    graph.add_node = AsyncMock(
        side_effect=lambda n: node_store.update({n.node_id: n})
    )
    graph.add_edge = AsyncMock()
    graph.get_edges = AsyncMock(return_value=[])
    graph.get_nodes_by_type = AsyncMock(return_value=[])
    return graph


def _decision_node(node_id, agent_id="agent-1", **extra_props):
    props = {"agent_id": agent_id, "text": "test decision", **extra_props}
    return GraphNode(
        node_id=node_id,
        node_type=DECISION_NODE_TYPE,
        label="test decision",
        properties=props,
    )


def _action_node(node_id, agent_id="agent-1", **extra_props):
    props = {"agent_id": agent_id, "text": "test action", "status": "pending", **extra_props}
    return GraphNode(
        node_id=node_id,
        node_type=ACTION_ITEM_NODE_TYPE,
        label="test action",
        properties=props,
    )


def _concept_node(node_id):
    return GraphNode(
        node_id=node_id,
        node_type="concept",
        label="some concept",
        properties={},
    )


def _skill_node(node_id):
    return GraphNode(
        node_id=node_id,
        node_type="skill",
        label="some skill",
        properties={},
    )


# =============================================================================
# Fix #1: mark_superseded rejects non-claim node types
# =============================================================================


class TestMarkSupersededNodeTypeValidation:

    @pytest.mark.asyncio
    async def test_rejects_concept_node(self):
        old = _concept_node("concept:agent-1:weather")
        new = _decision_node("decision:agent-1:abc")
        graph = _make_mock_graph([old, new])

        result = await mark_superseded(graph, old.node_id, new.node_id, "test")
        assert result["success"] is False
        assert result["error_code"] == "INVALID_NODE_TYPE"
        assert "concept" in result["error"]

    @pytest.mark.asyncio
    async def test_rejects_skill_node(self):
        old = _skill_node("skill:agent-1:cooking")
        new = _decision_node("decision:agent-1:abc")
        graph = _make_mock_graph([old, new])

        result = await mark_superseded(graph, old.node_id, new.node_id, "test")
        assert result["success"] is False
        assert result["error_code"] == "INVALID_NODE_TYPE"
        assert "skill" in result["error"]

    @pytest.mark.asyncio
    async def test_rejects_new_node_of_non_claim_type(self):
        old = _decision_node("decision:agent-1:abc")
        new = _concept_node("concept:agent-1:weather")
        graph = _make_mock_graph([old, new])

        result = await mark_superseded(graph, old.node_id, new.node_id, "test")
        assert result["success"] is False
        assert result["error_code"] == "INVALID_NODE_TYPE"

    @pytest.mark.asyncio
    async def test_accepts_decision_node(self):
        old = _decision_node("decision:agent-1:old")
        new = _decision_node("decision:agent-1:new")
        graph = _make_mock_graph([old, new])

        result = await mark_superseded(graph, old.node_id, new.node_id, "changed mind")
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_accepts_action_item_node(self):
        old = _action_node("action:agent-1:old")
        new = _action_node("action:agent-1:new")
        graph = _make_mock_graph([old, new])

        result = await mark_superseded(graph, old.node_id, new.node_id, "replaced")
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_rejects_nonexistent_old_node(self):
        new = _decision_node("decision:agent-1:new")
        graph = _make_mock_graph([new])

        result = await mark_superseded(graph, "nonexistent", new.node_id, "test")
        assert result["success"] is False
        assert result["error_code"] == "NODE_NOT_FOUND"

    @pytest.mark.asyncio
    async def test_rejects_nonexistent_new_node(self):
        old = _decision_node("decision:agent-1:old")
        graph = _make_mock_graph([old])

        result = await mark_superseded(graph, old.node_id, "nonexistent", "test")
        assert result["success"] is False
        assert result["error_code"] == "NODE_NOT_FOUND"

    @pytest.mark.asyncio
    async def test_claim_node_types_constant_covers_expected(self):
        assert ACTION_ITEM_NODE_TYPE in CLAIM_NODE_TYPES
        assert DECISION_NODE_TYPE in CLAIM_NODE_TYPES
        assert "concept" not in CLAIM_NODE_TYPES
        assert "skill" not in CLAIM_NODE_TYPES


# =============================================================================
# Fix #4: contradicted_by semantics and supersedes edge direction
# =============================================================================


class TestContradictedBySemantics:
    """Pin the direction: `contradicted_by` on old node lists nodes that
    contradict it. Edge direction: new → old with label 'supersedes'."""

    @pytest.mark.asyncio
    async def test_old_node_gets_contradicted_by_field(self):
        old = _decision_node("decision:agent-1:old")
        new = _decision_node("decision:agent-1:new")
        graph = _make_mock_graph([old, new])

        await mark_superseded(graph, old.node_id, new.node_id, "changed")

        # Old node should have been updated via add_node
        updated = graph.add_node.await_args_list[-1].args[0]
        # The call that sets superseded_by is on the old node
        # Find the call for the old node
        old_updates = [
            c.args[0] for c in graph.add_node.await_args_list
            if c.args[0].node_id == old.node_id
        ]
        assert old_updates
        props = old_updates[-1].properties
        assert props["superseded_by"] == new.node_id
        assert new.node_id in props["contradicted_by"]

    @pytest.mark.asyncio
    async def test_supersedes_edge_points_new_to_old(self):
        old = _decision_node("decision:agent-1:old")
        new = _decision_node("decision:agent-1:new")
        graph = _make_mock_graph([old, new])

        await mark_superseded(graph, old.node_id, new.node_id, "reason")

        # Edge: new → old, label "supersedes"
        edge_calls = [
            c for c in graph.add_edge.await_args_list
            if c.args[2] == "supersedes"
        ]
        assert len(edge_calls) == 1
        call = edge_calls[0]
        assert call.args[0] == new.node_id  # source = new
        assert call.args[1] == old.node_id  # target = old

    @pytest.mark.asyncio
    async def test_contradicted_by_accumulates(self):
        """Multiple supersessions should accumulate in contradicted_by list."""
        old = _decision_node("decision:agent-1:old")
        new1 = _decision_node("decision:agent-1:new1")
        new2 = _decision_node("decision:agent-1:new2")
        graph = _make_mock_graph([old, new1, new2])

        await mark_superseded(graph, old.node_id, new1.node_id, "first")
        await mark_superseded(graph, old.node_id, new2.node_id, "second")

        # The graph mock accumulates updates via the side_effect
        final_node = graph.add_node.await_args_list[-1].args[0]
        # Find the last update to old node
        old_updates = [
            c.args[0] for c in graph.add_node.await_args_list
            if c.args[0].node_id == old.node_id
        ]
        props = old_updates[-1].properties
        assert new1.node_id in props["contradicted_by"]
        assert new2.node_id in props["contradicted_by"]


# =============================================================================
# Fix #2: Epistemic fields flow from message metadata to claim nodes
# =============================================================================


class TestEpistemicFlowToClaimNodes:

    @pytest.fixture
    def router(self):
        graph = _make_mock_graph([])
        db = MagicMock()
        db.execute = AsyncMock()
        db.fetchall = AsyncMock(return_value=[])
        db.fetchone = AsyncMock(return_value=None)
        return SchemaRouter(graph=graph, db=db, agent_id="agent-1")

    @pytest.mark.asyncio
    async def test_decision_node_inherits_epistemic_from_metadata(self, router):
        metadata = {
            "claim_source": "inferred",
            "claim_certainty": 0.5,
            "temporal_validity": "durable",
        }
        await router.route(
            message_id="msg-1",
            content="I've decided to move to Brooklyn.",
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
        assert props["claim_source"] == "inferred"
        assert props["claim_certainty"] == 0.5
        assert props["temporal_validity"] == "durable"

    @pytest.mark.asyncio
    async def test_action_item_inherits_epistemic_from_metadata(self, router):
        metadata = {
            "claim_source": "hearsay",
            "claim_certainty": 0.3,
            "temporal_validity": "ephemeral",
        }
        await router.route(
            message_id="msg-2",
            content="I need to call the doctor.",
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
        assert props["claim_source"] == "hearsay"
        assert props["claim_certainty"] == 0.3
        assert props["temporal_validity"] == "ephemeral"

    @pytest.mark.asyncio
    async def test_no_metadata_means_no_epistemic_fields(self, router):
        await router.route(
            message_id="msg-3",
            content="I've decided to use Postgres.",
            concepts=[],
            role="user",
        )
        decision_nodes = [
            c.args[0] for c in router.graph.add_node.await_args_list
            if c.args[0].node_type == DECISION_NODE_TYPE
        ]
        assert decision_nodes
        props = decision_nodes[0].properties
        assert "claim_source" not in props
        assert "claim_certainty" not in props

    @pytest.mark.asyncio
    async def test_partial_metadata_only_includes_present_fields(self, router):
        metadata = {"claim_source": "direct"}
        await router.route(
            message_id="msg-4",
            content="I've decided to ship it.",
            concepts=[],
            role="user",
            metadata=metadata,
        )
        decision_nodes = [
            c.args[0] for c in router.graph.add_node.await_args_list
            if c.args[0].node_type == DECISION_NODE_TYPE
        ]
        props = decision_nodes[0].properties
        assert props["claim_source"] == "direct"
        assert "claim_certainty" not in props


# =============================================================================
# Fix #5: _detect_epistemic_status docstring matches implementation
# =============================================================================


class TestDetectEpistemicStatus:

    def setup_method(self):
        self.tagger = EmotionalTagger()

    def test_returns_none_for_assistant_messages(self):
        result = self.tagger._detect_epistemic_status("I decided to help.", "assistant")
        assert result is None

    def test_returns_none_for_pure_questions(self):
        result = self.tagger._detect_epistemic_status("What time is it?", "user")
        assert result is None

    def test_returns_none_for_greetings(self):
        result = self.tagger._detect_epistemic_status("Hello!", "user")
        assert result is None

    def test_returns_none_for_farewells(self):
        result = self.tagger._detect_epistemic_status("Goodbye, see you later", "user")
        assert result is None

    def test_direct_certainty_for_plain_statement(self):
        """All user statements start at direct certainty unless cues downgrade."""
        result = self.tagger._detect_epistemic_status("I went to the store.", "user")
        assert result is not None
        assert result["claim_source"] == "direct"
        assert result["claim_certainty"] == DEFAULT_CERTAINTY_BY_SOURCE["direct"]

    def test_hedging_downgrades_to_inferred(self):
        result = self.tagger._detect_epistemic_status("I think it might rain.", "user")
        assert result is not None
        assert result["claim_source"] == "inferred"
        assert result["claim_certainty"] < DEFAULT_CERTAINTY_BY_SOURCE["direct"]

    def test_hearsay_detected(self):
        result = self.tagger._detect_epistemic_status(
            "Apparently the office is closing early.", "user"
        )
        assert result is not None
        assert result["claim_source"] == "hearsay"
        assert result["claim_certainty"] == DEFAULT_CERTAINTY_BY_SOURCE["hearsay"]

    def test_i_heard_is_hearsay(self):
        result = self.tagger._detect_epistemic_status(
            "I heard they're laying off the team.", "user"
        )
        assert result is not None
        assert result["claim_source"] == "hearsay"

    def test_ephemeral_temporal_detection(self):
        result = self.tagger._detect_epistemic_status(
            "I'm at the store right now.", "user"
        )
        assert result is not None
        assert result["temporal_validity"] == "ephemeral"

    def test_moment_temporal_detection(self):
        result = self.tagger._detect_epistemic_status(
            "I just sneezed.", "user"
        )
        assert result is not None
        assert result["temporal_validity"] == "moment"

    def test_no_temporal_cue_returns_none_temporal(self):
        result = self.tagger._detect_epistemic_status("I live in Austin.", "user")
        assert result is not None
        assert result["temporal_validity"] is None

    def test_certainty_clamped_above_zero(self):
        """Even with extreme hedging, certainty never drops below 0.05."""
        result = self.tagger._detect_epistemic_status(
            "I'm not sure, maybe it's possibly wrong.", "user"
        )
        assert result is not None
        assert result["claim_certainty"] >= 0.05

    def test_mixed_question_and_statement_is_not_pure_question(self):
        """A message with both statement and question should be treated as a claim."""
        result = self.tagger._detect_epistemic_status(
            "I went to the store. Did you need anything?", "user"
        )
        assert result is not None
        assert result["claim_source"] == "direct"


# =============================================================================
# MemoryMetadata epistemic fields
# =============================================================================


class TestMemoryMetadataEpistemic:

    def test_default_epistemic_fields_are_none(self):
        meta = MemoryMetadata()
        assert meta.claim_source is None
        assert meta.claim_certainty is None
        assert meta.temporal_validity is None

    def test_to_dict_includes_epistemic_fields(self):
        meta = MemoryMetadata(
            claim_source="direct",
            claim_certainty=0.9,
            temporal_validity="durable",
        )
        d = meta.to_dict()
        assert d["claim_source"] == "direct"
        assert d["claim_certainty"] == 0.9
        assert d["temporal_validity"] == "durable"

    def test_from_dict_round_trips(self):
        original = MemoryMetadata(
            claim_source="hearsay",
            claim_certainty=0.3,
            temporal_validity="ephemeral",
        )
        d = original.to_dict()
        restored = MemoryMetadata.from_dict(d)
        assert restored.claim_source == "hearsay"
        assert restored.claim_certainty == 0.3
        assert restored.temporal_validity == "ephemeral"

    def test_from_dict_handles_missing_epistemic_fields(self):
        meta = MemoryMetadata.from_dict({"emotional_valence": 0.5})
        assert meta.claim_source is None
        assert meta.claim_certainty is None
        assert meta.temporal_validity is None


# =============================================================================
# EmotionalTagger integration (epistemic fields populated in analyze())
# =============================================================================


class TestEmotionalTaggerEpistemicIntegration:

    @pytest.mark.asyncio
    async def test_analyze_populates_epistemic_for_user_claims(self):
        tagger = EmotionalTagger()
        meta = await tagger.analyze("I went to the store today.", "user")
        assert meta.claim_source == "direct"
        assert meta.claim_certainty is not None
        assert meta.claim_certainty > 0

    @pytest.mark.asyncio
    async def test_analyze_no_epistemic_for_assistant(self):
        tagger = EmotionalTagger()
        meta = await tagger.analyze("Here is your answer.", "assistant")
        assert meta.claim_source is None
        assert meta.claim_certainty is None

    @pytest.mark.asyncio
    async def test_analyze_hedging_lowers_certainty(self):
        tagger = EmotionalTagger()
        meta = await tagger.analyze("I think the project is going well.", "user")
        assert meta.claim_source == "inferred"
        assert meta.claim_certainty < DEFAULT_CERTAINTY_BY_SOURCE["direct"]

    @pytest.mark.asyncio
    async def test_analyze_question_has_no_epistemic(self):
        tagger = EmotionalTagger()
        meta = await tagger.analyze("What do you think?", "user")
        assert meta.claim_source is None


# =============================================================================
# Enums
# =============================================================================


class TestEnums:

    def test_claim_source_values(self):
        assert ClaimSource.DIRECT.value == "direct"
        assert ClaimSource.OBSERVED.value == "observed"
        assert ClaimSource.INFERRED.value == "inferred"
        assert ClaimSource.HEARSAY.value == "hearsay"

    def test_temporal_validity_values(self):
        assert TemporalValidity.DURABLE.value == "durable"
        assert TemporalValidity.EPHEMERAL.value == "ephemeral"
        assert TemporalValidity.MOMENT.value == "moment"

    def test_default_certainty_covers_all_sources(self):
        for src in ClaimSource:
            assert src.value in DEFAULT_CERTAINTY_BY_SOURCE


# =============================================================================
# Fix #3: No interaction supersession claims
# =============================================================================


class TestNoInteractionSupersessionClaims:
    """mark_superseded operates on nodes only. Interactions (mentions edges)
    are NOT supersedable via this API. This test pins that contract."""

    @pytest.mark.asyncio
    async def test_mark_superseded_only_accepts_claim_types(self):
        """Exhaustive: every non-claim type is rejected."""
        for node_type in ("concept", "skill", "message", "episode"):
            old = GraphNode(
                node_id=f"{node_type}:test",
                node_type=node_type,
                label="test",
                properties={},
            )
            new = _decision_node("decision:agent-1:new")
            graph = _make_mock_graph([old, new])
            result = await mark_superseded(graph, old.node_id, new.node_id)
            assert result["success"] is False, f"Should reject {node_type}"
            assert result["error_code"] == "INVALID_NODE_TYPE"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
