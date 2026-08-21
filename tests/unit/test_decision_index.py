"""STRATEGY.yaml decisions must be reachable through the graph (#2851).

Decisions lived in two stores that never met: `strategy_add_decision` appended
to STRATEGY.yaml, while `recall_decisions` queried a `decision` node type that
nothing ever wrote. On Emma that meant `recall_decisions` returned 0 while a
66 KB file held the real record, and `mark_superseded` — the strongest primitive
in the graph-reasoning surface — had nothing to operate on.

YAML stays canonical. The graph is a derived index, and these tests pin the
properties that keep that direction honest.
"""

import json
from unittest.mock import MagicMock

import pytest

from kestrel_sovereign.features.strategic_memory.decision_index import (
    decision_entries,
    project_decisions,
    strategy_decision_node_id,
)


AGENT = "did:test:strategy"


class _FakeGraph:
    """Stands in for AsyncGraphStore, with real compare-and-swap semantics.

    The swap actually compares the stored properties against the caller's
    snapshot and refuses when they differ. A double that always succeeded
    would let the projection pass every test while still losing a concurrent
    ``mark_superseded`` in production — the double has to be faithful to the
    property under test, not merely to the signature.
    """

    def __init__(self, *, fail_on=None, fail_read_on=None):
        self.nodes = {}
        self.writes = 0
        self.deleted = []
        self._fail_on = fail_on
        self._fail_read_on = fail_read_on
        #: Fires once, between the projection's read and its swap.
        self.on_before_swap = None

    async def get_node(self, node_id):
        if self._fail_read_on and self._fail_read_on in node_id:
            raise RuntimeError("graph read refused")
        return self.nodes.get(node_id)

    async def get_nodes_by_type(self, node_type):
        return [n for n in self.nodes.values() if n.node_type == node_type]

    async def delete_node(self, node_id):
        self.deleted.append(node_id)
        self.nodes.pop(node_id, None)

    async def add_node(self, node):
        self.writes += 1
        if self._fail_on and self._fail_on in node.node_id:
            raise RuntimeError("graph write refused")
        self.nodes[node.node_id] = node

    async def compare_and_swap_node(self, node_id, expected, new_node):
        from kestrel_sovereign.storage.async_graph_store import NodeSwapResult

        if self.on_before_swap is not None:
            hook, self.on_before_swap = self.on_before_swap, None
            hook(self)

        self.writes += 1
        if self._fail_on and self._fail_on in node_id:
            raise RuntimeError("graph write refused")

        current = self.nodes.get(node_id)
        if current is None:
            if expected is not None:
                return NodeSwapResult.NOT_FOUND
            self.nodes[node_id] = new_node
            return NodeSwapResult.SWAPPED
        if (current.properties or None) != (expected or None):
            return NodeSwapResult.PREDICATE_FAILED
        # Properties-only, like the real primitive: type and label are set at
        # creation and are not rewritten by a swap.
        current.properties = new_node.properties
        return NodeSwapResult.SWAPPED


def _entry(decision, date="2026-07-01", rationale="because", impact="", session=""):
    return {
        "date": date,
        "session": session,
        "decision": decision,
        "rationale": rationale,
        "impact": impact,
    }


class TestProjection:
    @pytest.mark.asyncio
    async def test_decisions_become_reachable_nodes(self):
        graph = _FakeGraph()
        entries = [_entry("adopt durable leases"), _entry("stop using cron")]

        report = await project_decisions(graph, AGENT, entries)

        assert report["projected"] == 2
        assert len(graph.nodes) == 2
        for node in graph.nodes.values():
            assert node.node_type == "decision"
            # Both are load-bearing: recall_decisions filters on agent_id and
            # orders on created_at, so a node missing either is unreachable.
            assert node.properties["agent_id"] == AGENT
            assert node.properties["created_at"] == "2026-07-01"
            assert node.properties["claim_source"] == "strategy_yaml"

    @pytest.mark.asyncio
    async def test_reprojection_upserts_rather_than_duplicating(self):
        graph = _FakeGraph()
        entries = [_entry("adopt durable leases")]

        await project_decisions(graph, AGENT, entries)
        await project_decisions(graph, AGENT, entries)
        await project_decisions(graph, AGENT, entries)

        assert len(graph.nodes) == 1, (
            "node identity must be a function of the entry, or every rebuild "
            "mints duplicates"
        )

    @pytest.mark.asyncio
    async def test_editing_rationale_does_not_orphan_the_node(self):
        """A typo fix in rationale must edit the node, not mint a second."""
        graph = _FakeGraph()
        await project_decisions(graph, AGENT, [_entry("x", rationale="beacuse")])
        await project_decisions(graph, AGENT, [_entry("x", rationale="because")])

        assert len(graph.nodes) == 1
        assert list(graph.nodes.values())[0].properties["rationale"] == "because"

    @pytest.mark.asyncio
    async def test_entry_without_text_is_skipped(self):
        graph = _FakeGraph()
        report = await project_decisions(
            graph, AGENT, [_entry(""), {"date": "2026-07-01"}, "not a dict"]
        )
        assert report["projected"] == 0
        assert report["skipped"] == 3
        assert graph.nodes == {}


class TestSupersessionSurvivesRebuild:
    """The property that makes rebuilding safe."""

    @pytest.mark.asyncio
    async def test_rebuild_does_not_un_supersede_a_decision(self):
        """Supersession is graph-owned and YAML cannot express it.

        If a rebuild flattened it, a decision the agent had explicitly retired
        would silently return to `recall_decisions` — the index quietly
        overruling a judgement the agent made.
        """
        graph = _FakeGraph()
        entries = [_entry("use cron")]
        await project_decisions(graph, AGENT, entries)

        node_id = strategy_decision_node_id(AGENT, entries[0])
        # mark_superseded writes these; YAML has no field for them.
        graph.nodes[node_id].properties["superseded_by"] = "decision:other"
        graph.nodes[node_id].properties["superseded_at"] = "2026-07-05T00:00:00Z"
        graph.nodes[node_id].properties["superseded_reason"] = "moved to leases"

        await project_decisions(graph, AGENT, entries)

        props = graph.nodes[node_id].properties
        assert props["superseded_by"] == "decision:other"
        assert props["superseded_at"] == "2026-07-05T00:00:00Z"
        assert props["superseded_reason"] == "moved to leases"

    @pytest.mark.asyncio
    async def test_yaml_content_still_refreshes_on_rebuild(self):
        """Preserving graph state must not freeze the YAML-owned fields."""
        graph = _FakeGraph()
        await project_decisions(graph, AGENT, [_entry("x", impact="small")])
        node_id = strategy_decision_node_id(AGENT, _entry("x"))
        graph.nodes[node_id].properties["superseded_by"] = "decision:other"

        await project_decisions(graph, AGENT, [_entry("x", impact="large")])

        props = graph.nodes[node_id].properties
        assert props["impact"] == "large", "YAML remains the source for its own fields"
        assert props["superseded_by"] == "decision:other"


class TestIndexNeverOwnsTheTruth:
    @pytest.mark.asyncio
    async def test_missing_graph_store_is_not_an_error(self):
        """YAML is already on disk; a missing index must not look like failure."""
        report = await project_decisions(None, AGENT, [_entry("x")])
        assert report["projected"] == 0
        assert report["skipped_reason"] == "no_graph_store"

    @pytest.mark.asyncio
    async def test_one_failed_node_does_not_stop_the_rest(self):
        graph = _FakeGraph(fail_on="strategy")
        report = await project_decisions(graph, AGENT, [_entry("a"), _entry("b")])
        assert report["failed"] == 2
        assert report["projected"] == 0

    def test_decision_entries_tolerates_a_malformed_file(self):
        assert decision_entries(None) == []
        assert decision_entries({}) == []
        assert decision_entries({"decisions": "not a list"}) == []
        assert decision_entries({"decisions": [{"decision": "x"}, "junk"]}) == [
            {"decision": "x"}
        ]


class TestEndToEndThroughTheFeature:
    """The ticket's verify gate: record a decision, then recall it."""

    @pytest.mark.asyncio
    async def test_recorded_decision_is_retrievable_and_supersedable(self, tmp_path):
        from kestrel_sovereign.features.strategic_memory.feature import (
            StrategicMemoryFeature,
        )

        graph = _FakeGraph()
        agent = MagicMock()
        agent.agent_id = AGENT
        agent.agent_data_dir = str(tmp_path)
        agent.storage = MagicMock()
        agent.storage.graph = graph

        feature = StrategicMemoryFeature(agent)
        await feature.initialize()

        result = await feature.strategy_add_decision(
            decision="adopt durable leases", rationale="cron drops them"
        )
        assert result.data["recorded"] is True

        # Reachable the way recall_decisions reaches it.
        decisions = [
            n for n in graph.nodes.values()
            if n.node_type == "decision"
            and (n.properties or {}).get("agent_id") == AGENT
        ]
        assert len(decisions) == 1, "a recorded decision must be recallable"
        assert "durable leases" in decisions[0].label

        # And mark_superseded's preconditions hold: claim-shaped type, owned by
        # this agent, resolvable by id.
        from kestrel_sovereign.storage.schema_router import CLAIM_SHAPED_NODE_TYPES

        node = decisions[0]
        assert node.node_type in CLAIM_SHAPED_NODE_TYPES
        assert await graph.get_node(node.node_id) is not None

    @pytest.mark.asyncio
    async def test_existing_yaml_is_indexed_at_load(self, tmp_path):
        """Decisions recorded before the index existed must become reachable."""
        import yaml as _yaml

        from kestrel_sovereign.features.strategic_memory.feature import (
            StrategicMemoryFeature,
        )

        (tmp_path / "STRATEGY.yaml").write_text(
            _yaml.dump({
                "version": 1,
                "decisions": [
                    _entry("keep STRATEGY.yaml canonical"),
                    _entry("graph is a derived index", date="2026-07-02"),
                ],
            }),
            encoding="utf-8",
        )

        graph = _FakeGraph()
        agent = MagicMock()
        agent.agent_id = AGENT
        agent.agent_data_dir = str(tmp_path)
        agent.storage = MagicMock()
        agent.storage.graph = graph

        feature = StrategicMemoryFeature(agent)
        await feature.initialize()

        assert len(graph.nodes) == 2, (
            "a pre-existing STRATEGY.yaml must be indexed at load, or the "
            "index is not rebuildable from its source"
        )

    @pytest.mark.asyncio
    async def test_yaml_write_still_succeeds_when_the_graph_fails(self, tmp_path):
        """The index must never be able to fail a canonical write."""
        from kestrel_sovereign.features.strategic_memory.feature import (
            StrategicMemoryFeature,
        )

        agent = MagicMock()
        agent.agent_id = AGENT
        agent.agent_data_dir = str(tmp_path)
        agent.storage = MagicMock()
        agent.storage.graph = _FakeGraph(fail_on="strategy")

        feature = StrategicMemoryFeature(agent)
        await feature.initialize()

        result = await feature.strategy_add_decision(
            decision="ship it", rationale="tests pass"
        )

        assert result.data["recorded"] is True
        on_disk = (tmp_path / "STRATEGY.yaml").read_text(encoding="utf-8")
        assert "ship it" in on_disk

    @pytest.mark.asyncio
    async def test_projection_never_writes_back_to_yaml(self, tmp_path):
        """A derived index that edits its source is no longer derived."""
        import yaml as _yaml

        from kestrel_sovereign.features.strategic_memory.feature import (
            StrategicMemoryFeature,
        )

        path = tmp_path / "STRATEGY.yaml"
        path.write_text(
            _yaml.dump({"version": 1, "decisions": [_entry("x")]}), encoding="utf-8"
        )
        before = path.read_text(encoding="utf-8")

        agent = MagicMock()
        agent.agent_id = AGENT
        agent.agent_data_dir = str(tmp_path)
        agent.storage = MagicMock()
        agent.storage.graph = _FakeGraph()

        feature = StrategicMemoryFeature(agent)
        await feature.initialize()
        await feature._reindex_decisions()

        assert path.read_text(encoding="utf-8") == before


# --- Review findings on the #2851 implementation -----------------------------


class TestProductionIdentity:
    """The projector must use an identity the real feature actually has."""

    @pytest.mark.asyncio
    async def test_projection_uses_the_agents_identity_not_a_feature_attribute(
        self, tmp_path
    ):
        """``self.agent_id`` does not exist on any feature.

        Reading it raised AttributeError into the best-effort handler, so the
        index never populated in production at all — while the tests passed,
        because they assigned ``feature.agent_id`` by hand. The mutants died
        against a world only the tests contained.
        """
        from kestrel_sovereign.features.strategic_memory.feature import (
            StrategicMemoryFeature,
        )

        agent = MagicMock()
        agent.agent_id = AGENT
        agent.storage.graph = _FakeGraph()
        feature = StrategicMemoryFeature(agent)

        assert not hasattr(feature, "agent_id"), (
            "if a feature ever gains its own agent_id this test is checking "
            "the wrong thing"
        )
        feature._data = {"decisions": [_entry("index me")]}

        report = await feature._reindex_decisions()

        assert report["projected"] == 1
        assert agent.storage.graph.nodes


class TestReconciliation:
    """YAML is canonical, so removals there must reach the index."""

    @pytest.mark.asyncio
    async def test_a_decision_removed_from_yaml_stops_being_reachable(self):
        graph = _FakeGraph()
        await project_decisions(graph, AGENT, [_entry("keep me"), _entry("drop me")])
        assert len(graph.nodes) == 2

        report = await project_decisions(graph, AGENT, [_entry("keep me")])

        assert report["removed"] == 1
        assert [n.label for n in graph.nodes.values()] == ["keep me"]

    @pytest.mark.asyncio
    async def test_reconcile_leaves_other_agents_and_foreign_nodes_alone(self):
        from kestrel_sovereign.storage.async_graph_store import GraphNode

        graph = _FakeGraph()
        await project_decisions(graph, AGENT, [_entry("mine")])
        graph.nodes["other-agent"] = GraphNode(
            node_id="other-agent",
            node_type="decision",
            label="theirs",
            properties={"agent_id": "did:test:other", "source": "strategic_memory"},
        )
        graph.nodes["hand-written"] = GraphNode(
            node_id="hand-written",
            node_type="decision",
            label="not a projection",
            properties={"agent_id": AGENT, "source": "somewhere_else"},
        )

        await project_decisions(graph, AGENT, [])

        assert graph.deleted == [strategy_decision_node_id(AGENT, _entry("mine"))]
        assert "other-agent" in graph.nodes
        assert "hand-written" in graph.nodes


class TestConcurrentSupersession:
    """A rebuild must not clobber a supersession that lands mid-projection."""

    @pytest.mark.asyncio
    async def test_a_supersession_landing_mid_projection_is_not_overwritten(self):
        entry = _entry("supersede me")
        node_id = strategy_decision_node_id(AGENT, entry)
        graph = _FakeGraph()
        await project_decisions(graph, AGENT, [entry])

        def _supersede(g):
            g.nodes[node_id].properties = {
                **(g.nodes[node_id].properties or {}),
                "superseded_by": "decision:newer",
            }

        graph.on_before_swap = _supersede

        report = await project_decisions(graph, AGENT, [entry])

        # The write must NOT land on a stale snapshot; the supersession stands.
        assert graph.nodes[node_id].properties["superseded_by"] == "decision:newer"
        assert report["failed"] == 1

    @pytest.mark.asyncio
    async def test_a_failed_read_does_not_write_a_node_without_its_graph_state(self):
        """Treating a failed read as "absent" would revive a superseded decision."""
        entry = _entry("read fails")
        node_id = strategy_decision_node_id(AGENT, entry)
        graph = _FakeGraph()
        await project_decisions(graph, AGENT, [entry])
        graph.nodes[node_id].properties["superseded_by"] = "decision:newer"
        graph._fail_read_on = node_id

        report = await project_decisions(graph, AGENT, [entry])

        assert report["failed"] == 1
        assert graph.nodes[node_id].properties["superseded_by"] == "decision:newer"

    @pytest.mark.asyncio
    async def test_a_decision_that_never_reached_yaml_is_not_indexed(self, tmp_path):
        """The index must not publish what the canonical file does not contain.

        The entry is appended to ``_data`` before the write is attempted, so a
        failed write leaves it in memory. Projecting unconditionally would make
        ``recall_decisions`` return a decision that exists nowhere on disk —
        a derived index inventing its own source.
        """
        from kestrel_sovereign.features.strategic_memory.feature import (
            StrategicMemoryFeature,
        )

        agent = MagicMock()
        agent.agent_id = AGENT
        agent.agent_data_dir = str(tmp_path)
        agent.storage = MagicMock()
        agent.storage.graph = _FakeGraph()

        feature = StrategicMemoryFeature(agent)
        await feature.initialize()
        agent.storage.graph.nodes.clear()

        def _failing_save():
            from kestrel_sovereign.features.strategic_memory.feature import (
                _SaveOutcome,
            )

            return _SaveOutcome(persisted=False, error="disk full")

        feature._save = _failing_save

        result = await feature.strategy_add_decision(
            decision="never written", rationale="the disk was full"
        )

        assert result.data["persisted"] is False
        assert agent.storage.graph.nodes == {}
