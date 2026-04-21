"""Unit tests for the schema-aware routing module.

Covers the extractors, the 3-pass person resolver, and the SchemaRouter
orchestration itself. Recall tools on MemoryFeature have their own suite.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from kestrel_sovereign.storage.associative_linker import LinkedConcept
from kestrel_sovereign.storage.async_graph_store import GraphNode
from kestrel_sovereign.storage.schema_router import (
    ACTION_ITEM_NODE_TYPE,
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
    async def test_fuzzy_requires_minimum_evidence(self):
        """'Al' → 'Alice' is NOT enough evidence. Guards against false merges."""
        graph = _make_mock_graph([
            ("concept:agent-1:alice", "Alice"),
        ])
        resolver = PersonResolver(graph)
        m = await resolver.resolve("Al", "agent-1")
        # 2-char input is below the min shared prefix length
        assert m.status == "new"
        assert m.concept_id is None

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
    async def test_ensure_tables_is_noop(self, router):
        """Action items now live as graph nodes; no per-feature table creation."""
        await router.ensure_tables()
        # No DDL executed
        router.db.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_route_user_message_writes_action_items_as_graph_nodes(self, router):
        summary = await router.route(
            message_id="msg-1",
            content="I need to finalize the RFC by Friday.",
            concepts=[],
            role="user",
        )
        assert summary["action_items"] == 1
        action_node_calls = [
            c.args[0] for c in router.graph.add_node.await_args_list
            if c.args[0].node_type == ACTION_ITEM_NODE_TYPE
        ]
        assert len(action_node_calls) == 1
        node = action_node_calls[0]
        assert node.properties["status"] == "pending"
        assert node.properties["agent_id"] == "agent-1"

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
            concepts=[
                LinkedConcept(node_id="concept:agent-1:mom", label="mom", category="person"),
                LinkedConcept(node_id="concept:agent-1:sunday", label="sunday", category="time"),
            ],
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
        router.graph.add_node.assert_not_called()

    @pytest.mark.asyncio
    async def test_route_failure_in_one_lane_does_not_block_others(self, router):
        """Best-effort routing: action_item node write fails, decisions
        should still persist."""
        async def _add_node(node):
            if node.node_type == ACTION_ITEM_NODE_TYPE:
                raise Exception("graph down for actions only")

        router.graph.add_node = AsyncMock(side_effect=_add_node)
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
    async def test_is_idempotent_on_duplicate_message_processing(self, router):
        """Re-routing the same message must upsert the same action_item and
        decision nodes, not create new ones. Deterministic node_ids from
        (message_id, text) are the guard; graph upsert-by-node_id is the
        mechanism."""
        content = "I need to ship the feature. I've decided to use Postgres."

        await router.route(
            message_id="msg-idempotent",
            content=content,
            concepts=[],
            role="user",
        )
        await router.route(
            message_id="msg-idempotent",
            content=content,
            concepts=[],
            role="user",
        )

        action_ids = [
            c.args[0].node_id for c in router.graph.add_node.await_args_list
            if c.args[0].node_type == "action_item"
        ]
        decision_ids = [
            c.args[0].node_id for c in router.graph.add_node.await_args_list
            if c.args[0].node_type == "decision"
        ]
        # Two passes → two calls each, but both passes produce the same id.
        assert len(action_ids) >= 2
        assert len(set(action_ids)) == 1
        assert len(decision_ids) >= 2
        assert len(set(decision_ids)) == 1

    @pytest.mark.asyncio
    async def test_idempotent_action_preserves_user_status(self, router):
        """If the user marked an action done, reprocessing must not reset
        status to 'pending'. The preservation must happen only when
        get_node is called with the SAME deterministic id the router would
        produce — not whenever get_node is called at all."""
        from kestrel_sovereign.storage.async_graph_store import GraphNode as GN
        from kestrel_sovereign.storage.schema_router import (
            _deterministic_action_node_id,
        )

        content = "I need to Ship PR."
        # Extract the text the way the extractor would, then compute the
        # deterministic id the router will use.
        extracted = router.action_extractor.extract(content)
        assert extracted, "extractor precondition for this test"
        text = extracted[0]
        expected_id = _deterministic_action_node_id("agent-1", "msg-repeat", text)

        # Mock get_node to return existing state ONLY when asked for the
        # matching id. A different id must return None — otherwise the test
        # proves nothing about identity stability.
        async def _get_node(node_id):
            if node_id == expected_id:
                return GN(
                    node_id=expected_id,
                    node_type="action_item",
                    label="stale label",
                    properties={
                        "status": "done",
                        "agent_id": "agent-1",
                        "created_at": "2026-04-01T00:00:00+00:00",
                        "text": text,
                    },
                )
            return None

        router.graph.get_node = AsyncMock(side_effect=_get_node)

        await router.route(
            message_id="msg-repeat",
            content=content,
            concepts=[],
            role="user",
        )
        persisted = [
            c.args[0] for c in router.graph.add_node.await_args_list
            if c.args[0].node_type == "action_item"
        ]
        assert persisted
        assert persisted[-1].node_id == expected_id
        assert persisted[-1].properties["status"] == "done"
        assert persisted[-1].properties["created_at"] == "2026-04-01T00:00:00+00:00"

    @pytest.mark.asyncio
    async def test_unrelated_existing_node_does_not_pollute_new_action(self, router):
        """Regression guard: if get_node is called with the new action's id
        and the store has no such node, preservation must not accidentally
        inherit from some other action item."""
        from kestrel_sovereign.storage.async_graph_store import GraphNode as GN

        async def _get_node(node_id):
            # Unrelated existing node with a different id
            return None

        router.graph.get_node = AsyncMock(side_effect=_get_node)

        await router.route(
            message_id="msg-first-time",
            content="I need to file taxes.",
            concepts=[],
            role="user",
        )
        persisted = [
            c.args[0] for c in router.graph.add_node.await_args_list
            if c.args[0].node_type == "action_item"
        ]
        assert persisted
        # New node → default status pending, no inherited done state
        assert persisted[-1].properties["status"] == "pending"

    @pytest.mark.asyncio
    async def test_router_uses_concept_node_id_directly(self, router):
        """Router must use LinkedConcept.node_id as-is instead of
        reconstructing it from the label. This replaces the old pinning
        test from #646 — the coupling is gone because the linker now
        provides the canonical node_id."""
        await router.route(
            message_id="msg-shape",
            content="mom called.",
            concepts=[
                LinkedConcept(
                    node_id="concept:agent-1:mom",
                    label="mom",
                    category="person",
                ),
            ],
            role="user",
        )
        edge_targets = [
            c.args[1] for c in router.graph.add_edge.await_args_list
            if c.args[2] == "mentions"
        ]
        assert "concept:agent-1:mom" in edge_targets

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
            concepts=[
                LinkedConcept(
                    node_id="concept:agent-1:alice",
                    label="alice",
                    category="person",
                ),
            ],
            role="user",
        )
        assert len(summary["pending_person_matches"]) == 1
        pending = summary["pending_person_matches"][0]
        assert pending["mentioned_label"] == "alice"
        assert len(pending["candidates"]) == 2


# =============================================================================
# Privacy suppression hook
# =============================================================================


class TestRoutingSuppression:
    """The `skip_schema_routing` metadata flag must prevent the router from
    running in process_message. EPHEMERAL/ISOLATED callers set this to
    avoid persisting routing output to storage they don't want to keep."""

    def test_suppressed_helper_true(self):
        from kestrel_sovereign.storage.memory_system import _routing_suppressed
        assert _routing_suppressed({"skip_schema_routing": True}) is True

    def test_suppressed_helper_false(self):
        from kestrel_sovereign.storage.memory_system import _routing_suppressed
        assert _routing_suppressed({}) is False
        assert _routing_suppressed(None) is False
        assert _routing_suppressed({"skip_schema_routing": False}) is False

    def test_suppressed_helper_truthy_coercion(self):
        from kestrel_sovereign.storage.memory_system import _routing_suppressed
        # Any truthy value suppresses — string values from settings files
        # often come through as strings rather than bools.
        assert _routing_suppressed({"skip_schema_routing": "yes"}) is True


# =============================================================================
# Upsert semantics verification
# =============================================================================


class TestUpsertSemantics:
    """Pins the contract the router depends on: graph.add_edge upserts by
    (source, target, label). If that ever changes, interaction enrichment
    would silently start duplicating instead of updating."""

    def test_edge_upsert_sql_is_insert_or_replace_or_on_conflict(self):
        import inspect
        from kestrel_sovereign.storage.async_graph_store import AsyncGraphStore
        source = inspect.getsource(AsyncGraphStore._upsert_edge_sql)
        assert "INSERT OR REPLACE" in source or "ON CONFLICT" in source

    def test_node_upsert_sql_is_insert_or_replace_or_on_conflict(self):
        import inspect
        from kestrel_sovereign.storage.async_graph_store import AsyncGraphStore
        source = inspect.getsource(AsyncGraphStore._upsert_node_sql)
        assert "INSERT OR REPLACE" in source or "ON CONFLICT" in source


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
