"""Unit tests for the schema-aware routing module.

Covers the extractors, the 3-pass person resolver, and the SchemaRouter
orchestration itself. Recall tools on MemoryFeature have their own suite.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from kestrel_sovereign.storage.async_graph_store import GraphNode
from kestrel_sovereign.storage.schema_router import (
    ActionItemExtractor,
    DecisionExtractor,
    DECISION_NODE_TYPE,
    PersonMatch,
    PersonResolver,
    SchemaRouter,
    extract_interaction_sentiment,
)


# =============================================================================
# Extractors
# =============================================================================


class TestActionItemExtractor:

    def setup_method(self):
        self.ex = ActionItemExtractor()

    def test_extracts_first_person_commitments(self):
        items = self.ex.extract(
            "I need to call mom tomorrow. I should update the PRD. I'll review the design."
        )
        assert any("call mom" in i.lower() for i in items)
        assert any("update the prd" in i.lower() for i in items)
        assert any("review the design" in i.lower() for i in items)

    def test_extracts_todo_markers(self):
        items = self.ex.extract("TODO: fix the flaky test. Remind me to ship the PR.")
        assert any("fix the flaky test" in i.lower() for i in items)
        assert any("ship the pr" in i.lower() for i in items)

    def test_dedupes_repeated_extractions(self):
        items = self.ex.extract("I need to sleep. I need to sleep.")
        assert len(items) == 1

    def test_ignores_statements_without_commitment(self):
        items = self.ex.extract("The weather is nice today. Tests passed.")
        assert items == []

    def test_does_not_capture_whitespace_only_matches(self):
        items = self.ex.extract("I need to  ")
        assert items == []


class TestDecisionExtractor:

    def setup_method(self):
        self.ex = DecisionExtractor()

    def test_extracts_explicit_decisions(self):
        decisions = self.ex.extract(
            "I've decided to move to Brooklyn. My decision is to take the job."
        )
        assert any("move to brooklyn" in d.lower() for d in decisions)
        assert any("take the job" in d.lower() for d in decisions)

    def test_extracts_we_decisions(self):
        decisions = self.ex.extract("We've decided to use Postgres.")
        assert any("use postgres" in d.lower() for d in decisions)

    def test_empty_on_non_decision(self):
        assert self.ex.extract("Just thinking out loud.") == []


class TestInteractionSentiment:

    def test_positive(self):
        sentiment, topics = extract_interaction_sentiment(
            "I really appreciate how mom handled the weekend call."
        )
        assert sentiment == "positive"
        assert "weekend" in topics or "handled" in topics

    def test_negative(self):
        sentiment, _ = extract_interaction_sentiment(
            "Work was frustrating today; I'm stressed about the deadline."
        )
        assert sentiment == "negative"

    def test_mixed(self):
        sentiment, _ = extract_interaction_sentiment(
            "The meeting was great but the follow-up felt disappointed."
        )
        assert sentiment == "mixed"

    def test_none_when_no_cues(self):
        sentiment, _ = extract_interaction_sentiment("The file was saved successfully.")
        assert sentiment is None

    def test_topics_cap_at_five(self):
        content = " ".join(
            f"word{i}" for i in range(20)
        ) + " actual keywords here like project deadline review followup"
        _, topics = extract_interaction_sentiment(content)
        assert len(topics) <= 5


# =============================================================================
# PersonResolver (3-pass)
# =============================================================================


def _make_mock_graph(person_rows=None):
    """Build a mock graph with db.fetchall stubbed for person listing."""
    graph = MagicMock()
    graph.db = MagicMock()
    graph.db.fetchall = AsyncMock(return_value=person_rows or [])
    graph.db.execute = AsyncMock()
    graph.get_node = AsyncMock(return_value=None)
    graph.add_node = AsyncMock()
    graph.add_edge = AsyncMock()
    graph.get_edges = AsyncMock(return_value=[])
    return graph


class TestPersonResolverPasses:

    @pytest.mark.asyncio
    async def test_no_existing_returns_new(self):
        resolver = PersonResolver(_make_mock_graph([]))
        m = await resolver.resolve("Alice", "agent-1")
        assert m.status == "new"
        assert m.concept_id is None

    @pytest.mark.asyncio
    async def test_exact_match_pass_1(self):
        graph = _make_mock_graph([
            ("concept:agent-1:alice smith", "Alice Smith"),
            ("concept:agent-1:bob", "Bob"),
        ])
        resolver = PersonResolver(graph)
        m = await resolver.resolve("alice smith", "agent-1")
        assert m.status == "exact"
        assert m.concept_id == "concept:agent-1:alice smith"

    @pytest.mark.asyncio
    async def test_fuzzy_first_name_pass_2(self):
        graph = _make_mock_graph([
            ("concept:agent-1:robert", "Robert"),
        ])
        resolver = PersonResolver(graph)
        m = await resolver.resolve("Rob", "agent-1")
        assert m.status == "fuzzy"
        assert m.concept_id == "concept:agent-1:robert"

    @pytest.mark.asyncio
    async def test_collision_pass_3_flags_pending(self):
        graph = _make_mock_graph([
            ("concept:agent-1:alice a", "Alice A"),
            ("concept:agent-1:alice b", "Alice B"),
        ])
        resolver = PersonResolver(graph)
        m = await resolver.resolve("Alice", "agent-1")
        assert m.status == "pending"
        assert m.concept_id is None
        assert len(m.candidates) == 2

    @pytest.mark.asyncio
    async def test_does_not_auto_merge_on_collision(self):
        """Critical: with multiple same-first-name candidates, never pick one."""
        graph = _make_mock_graph([
            ("concept:agent-1:jon doe", "Jon Doe"),
            ("concept:agent-1:jon smith", "Jon Smith"),
            ("concept:agent-1:jon lee", "Jon Lee"),
        ])
        resolver = PersonResolver(graph)
        m = await resolver.resolve("Jon", "agent-1")
        assert m.status == "pending"
        assert m.concept_id is None

    @pytest.mark.asyncio
    async def test_empty_name_returns_new(self):
        resolver = PersonResolver(_make_mock_graph([]))
        m = await resolver.resolve("  ", "agent-1")
        assert m.status == "new"


# =============================================================================
# SchemaRouter orchestration
# =============================================================================


@pytest_asyncio.fixture
async def router():
    graph = _make_mock_graph([])
    db = MagicMock()
    db.execute = AsyncMock()
    db.fetchall = AsyncMock(return_value=[])
    db.fetchone = AsyncMock(return_value=None)
    r = SchemaRouter(graph=graph, db=db, agent_id="agent-1")
    return r


class TestSchemaRouterOrchestration:

    @pytest.mark.asyncio
    async def test_ensure_tables_creates_action_items(self, router):
        await router.ensure_tables()
        sqls = [c[0][0] for c in router.db.execute.call_args_list]
        assert any("CREATE TABLE IF NOT EXISTS action_items" in s for s in sqls)
        assert any("idx_action_items_agent_status" in s for s in sqls)

    @pytest.mark.asyncio
    async def test_route_user_message_writes_action_items(self, router):
        summary = await router.route(
            message_id="msg-1",
            content="I need to finalize the RFC by Friday.",
            concepts=[],
            role="user",
        )
        assert summary["action_items"] == 1
        insert_calls = [
            c for c in router.db.execute.call_args_list
            if "INSERT INTO action_items" in c[0][0]
        ]
        assert len(insert_calls) == 1

    @pytest.mark.asyncio
    async def test_route_user_message_writes_decisions_as_graph_nodes(self, router):
        summary = await router.route(
            message_id="msg-2",
            content="I've decided to move to Brooklyn.",
            concepts=[],
            role="user",
        )
        assert summary["decisions"] == 1
        router.graph.add_node.assert_awaited()
        node = router.graph.add_node.await_args[0][0]
        assert isinstance(node, GraphNode)
        assert node.node_type == DECISION_NODE_TYPE

    @pytest.mark.asyncio
    async def test_route_enriches_mention_edges_for_person_concepts(self, router):
        summary = await router.route(
            message_id="msg-3",
            content="I love when mom calls on Sundays.",
            concepts=["mom", "sunday"],
            role="user",
        )
        # Only 'mom' is person-shaped; 'sunday' is temporal.
        assert summary["interactions"] == 1
        edge_calls = router.graph.add_edge.await_args_list
        assert any(
            call.args[2] == "mentions"
            and call.kwargs.get("properties", {}).get("sentiment") == "positive"
            for call in edge_calls
        )

    @pytest.mark.asyncio
    async def test_route_skips_assistant_role(self, router):
        summary = await router.route(
            message_id="msg-4",
            content="I've decided to format that reply nicely.",
            concepts=[],
            role="assistant",
        )
        assert summary["action_items"] == 0
        assert summary["decisions"] == 0
        router.db.execute.assert_not_called()
        router.graph.add_node.assert_not_called()

    @pytest.mark.asyncio
    async def test_route_failure_in_one_lane_does_not_block_others(self, router):
        """Best-effort routing: action item write fails, decisions still persist."""
        async def _exec(sql, *args):
            if "INSERT INTO action_items" in sql:
                raise Exception("db down for actions only")

        router.db.execute = AsyncMock(side_effect=_exec)
        summary = await router.route(
            message_id="msg-5",
            content="I need to buy milk. I've decided to skip the meeting.",
            concepts=[],
            role="user",
        )
        # Action items lane failed, decisions lane succeeded.
        assert summary["action_items"] == 0
        assert summary["decisions"] == 1

    @pytest.mark.asyncio
    async def test_pending_person_match_surfaced_in_summary(self, router):
        # Two Alice concepts exist — mentioning "Alice" should flag pending.
        router.graph.db.fetchall = AsyncMock(return_value=[
            ("concept:agent-1:alice one", "Alice One"),
            ("concept:agent-1:alice two", "Alice Two"),
        ])
        summary = await router.route(
            message_id="msg-6",
            content="Thanks so much Alice for everything.",
            concepts=["alice"],
            role="user",
        )
        assert len(summary["pending_person_matches"]) == 1
        pending = summary["pending_person_matches"][0]
        assert pending["mentioned_label"] == "alice"
        assert len(pending["candidates"]) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
