"""STRATEGY.yaml decisions must be reachable through the graph (#2851).

Decisions lived in two stores that never met: `strategy_add_decision` appended
to STRATEGY.yaml, while `recall_decisions` queried a `decision` node type that
nothing ever wrote. On Emma that meant `recall_decisions` returned 0 while a
66 KB file held the real record, and `mark_superseded` — the strongest primitive
in the graph-reasoning surface — had nothing to operate on.

YAML stays canonical. The graph is a derived index, and these tests pin the
properties that keep that direction honest.
"""

import copy
import json
from datetime import date
from unittest.mock import MagicMock

import pytest

from kestrel_sovereign.features.strategic_memory.decision_index import (
    decision_entries,
    project_decisions,
    strategy_decision_node_id,
    supersede_entries,
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
            # The entry's day at midnight UTC, per the graph contract (#3255).
            assert node.properties["created_at"] == "2026-07-01T00:00:00+00:00"
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


class TestSupersessionIsTheFiles:
    """A decision's supersession lives in STRATEGY.yaml, like the rest of it (#4154).

    It used to be graph-owned: the projection carried it across a rebuild and
    never read the entry's ``superseded_by``. A decision the canonical file
    marked superseded therefore came back from ``recall_decisions`` as current.
    """

    @pytest.mark.asyncio
    async def test_entry_supersession_projects_to_the_node(self):
        graph = _FakeGraph()
        entry = {
            **_entry("use cron"),
            "superseded_by": "decision:leases",
            # A hand-written day loads as a date; the graph stores JSON.
            "superseded_at": date(2026, 7, 5),
            "superseded_reason": "moved to leases",
        }

        await project_decisions(graph, AGENT, [entry])

        props = graph.nodes[strategy_decision_node_id(AGENT, entry)].properties
        assert props["superseded_by"] == "decision:leases"
        assert props["superseded_at"] == "2026-07-05"
        assert props["superseded_reason"] == "moved to leases"
        json.dumps(props)

    @pytest.mark.asyncio
    async def test_an_entry_without_supersession_projects_none(self):
        """Absent, not empty: an unsuperseded node keeps the shape it always had."""
        graph = _FakeGraph()
        entry = {**_entry("use cron"), "superseded_by": "", "superseded_at": None}

        await project_decisions(graph, AGENT, [entry])

        props = graph.nodes[strategy_decision_node_id(AGENT, entry)].properties
        for key in ("superseded_by", "superseded_at", "superseded_reason"):
            assert key not in props

    @pytest.mark.asyncio
    async def test_yaml_content_still_refreshes_on_rebuild(self):
        """A superseded entry's own fields keep following the file."""
        graph = _FakeGraph()
        superseded = {"superseded_by": "decision:other"}
        await project_decisions(
            graph, AGENT, [{**_entry("x", impact="small"), **superseded}]
        )
        node_id = strategy_decision_node_id(AGENT, _entry("x"))

        await project_decisions(
            graph, AGENT, [{**_entry("x", impact="large"), **superseded}]
        )

        props = graph.nodes[node_id].properties
        assert props["impact"] == "large", "YAML remains the source for its own fields"
        assert props["superseded_by"] == "decision:other"

    def test_supersede_entries_marks_only_the_named_decision(self):
        entries = [
            {**_entry("keep"), "superseded_reason": "stale"},
            {**_entry("retire"), "superseded_reason": "an earlier reason"},
            "not a dict",
        ]
        node_id = strategy_decision_node_id(AGENT, entries[1])

        matched = supersede_entries(
            entries, AGENT, node_id,
            superseded_by="decision:new", superseded_at="2026-07-05T00:00:00+00:00",
        )

        assert matched == 1
        assert "superseded_by" not in entries[0]
        assert entries[0]["superseded_reason"] == "stale"
        assert entries[1]["superseded_by"] == "decision:new"
        assert entries[1]["superseded_at"] == "2026-07-05T00:00:00+00:00"
        assert "superseded_reason" not in entries[1], (
            "superseding again without a reason must not keep the old one"
        )
        assert supersede_entries(
            entries, AGENT, "decision:absent",
            superseded_by="decision:new", superseded_at="2026-07-05",
        ) == 0


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
    """A rebuild must not write over a node changed under it mid-projection.

    The file still decides: the next reindex projects it onto whatever the
    concurrent writer left. What the swap prevents is a stale snapshot
    landing on top of a newer one.
    """

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
        """A write made without reading the node cannot be conditional on it."""
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


# --- #4154: supersession has one home, STRATEGY.yaml --------------------------


def _write_strategy_file(path, decisions):
    import yaml as _yaml

    path.write_text(
        _yaml.dump(
            {"version": 1, "vision": "be useful", "decisions": decisions},
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _read_strategy_file(path):
    import yaml as _yaml

    return _yaml.safe_load(path.read_text(encoding="utf-8"))


def _decision_in(data, text):
    return next(d for d in data["decisions"] if d["decision"] == text)


_OLD = _entry("run reconciliation from cron")
_NEW = _entry("run reconciliation on durable leases", date="2026-07-02")
OLD_ID = strategy_decision_node_id(AGENT, _OLD)
NEW_ID = strategy_decision_node_id(AGENT, _NEW)


class TestSupersessionThroughTheGraphPath:
    """recall_decisions and mark_superseded against a real graph, as an agent has it."""

    @pytest.fixture
    async def raw(self, tmp_path):
        from kestrel_sovereign.storage import AsyncStorage

        async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT) as raw:
            yield raw

    @pytest.fixture
    def strategy_path(self, tmp_path):
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        return agent_dir / "STRATEGY.yaml"

    def _agent(self, raw, strategy_path):
        from kestrel_sovereign.privacy import PrivacyMode
        from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage

        agent = MagicMock()
        agent.agent_id = AGENT
        agent.did = AGENT
        agent.agent_data_dir = str(strategy_path.parent)
        agent.storage = PrivacyEnforcingStorage(raw, PrivacyMode.NORMAL)
        agent.features = {}
        agent.get_feature = lambda name: agent.features.get(name)
        return agent

    async def _strategic(self, agent):
        from kestrel_sovereign.features.strategic_memory.feature import (
            StrategicMemoryFeature,
        )

        feature = StrategicMemoryFeature(agent)
        await feature.initialize()
        agent.features["StrategicMemoryFeature"] = feature
        return feature

    def _memory(self, agent):
        from kestrel_sovereign.features.memory.feature import MemoryFeature

        feature = MemoryFeature.__new__(MemoryFeature)
        feature.agent = agent
        feature.agent_id = AGENT
        return feature

    async def _recalled(self, memory, **kwargs):
        from kestrel_sdk.tools.result import ToolResultStatus

        result = await memory.recall_decisions(**kwargs)
        assert result.status is ToolResultStatus.OK, result
        return {d["text"]: d for d in result.data["decisions"]}

    async def _supersedes_edges(self, raw):
        edges = await raw.graph.get_edges(NEW_ID, direction="out")
        return [(e.source_id, e.target_id) for e in edges if e.label == "supersedes"]

    async def test_yaml_superseded_by_projects_to_node(self, raw, strategy_path):
        _write_strategy_file(
            strategy_path,
            [
                {**_OLD, "superseded_by": NEW_ID, "superseded_at": date(2026, 7, 5)},
                _NEW,
            ],
        )
        agent = self._agent(raw, strategy_path)
        await self._strategic(agent)
        memory = self._memory(agent)

        current = await self._recalled(memory)
        everything = await self._recalled(memory, include_superseded=True)

        assert list(current) == [_NEW["decision"]], (
            "a decision STRATEGY.yaml marks superseded must not come back as current"
        )
        assert set(everything) == {_OLD["decision"], _NEW["decision"]}
        assert everything[_OLD["decision"]]["superseded_by"] == NEW_ID
        node = await raw.graph.get_node(OLD_ID)
        assert node.properties["superseded_at"] == "2026-07-05"

    async def test_graph_supersede_writes_yaml(
        self, raw, strategy_path, monkeypatch, caplog
    ):
        from kestrel_sdk.tools.result import ToolResultStatus
        from kestrel_sovereign.features.strategic_memory import feature as strategic_module
        from kestrel_sovereign.features.strategic_memory.atomic_file import file_digest
        from kestrel_sovereign.features.strategic_memory.decision_index import (
            supersession_of,
        )

        _write_strategy_file(strategy_path, [_OLD, _NEW])
        agent = self._agent(raw, strategy_path)
        strategic = await self._strategic(agent)
        memory = self._memory(agent)
        before = strategy_path.read_bytes()

        atomic_writes = []
        replace = strategic_module._replace_atomically

        def spy(path, content):
            # The digest check runs before the replace: the file is still the
            # one this feature read.
            atomic_writes.append((path, file_digest(path.read_bytes())))
            replace(path, content)

        monkeypatch.setattr(strategic_module, "_replace_atomically", spy)
        caplog.set_level("WARNING")

        result = await memory.mark_superseded(
            old_id=OLD_ID, new_id=NEW_ID, reason="cron drops leases"
        )

        assert result.status is ToolResultStatus.OK, result
        assert result.data["yaml_written"] is True
        assert result.data["property_updated"] is True
        assert result.data["edge_written"] is True
        assert atomic_writes == [(strategy_path, file_digest(before))]
        assert not list(strategy_path.parent.glob("*.tmp")), "no staging file left"
        assert strategic._disk_digest == file_digest(strategy_path.read_bytes())

        on_disk = _read_strategy_file(strategy_path)
        old_entry = _decision_in(on_disk, _OLD["decision"])
        assert old_entry["superseded_by"] == NEW_ID
        assert old_entry["superseded_reason"] == "cron drops leases"
        assert old_entry["superseded_at"]
        assert "superseded_by" not in _decision_in(on_disk, _NEW["decision"])
        assert on_disk["vision"] == "be useful"

        node = await raw.graph.get_node(OLD_ID)
        assert supersession_of(node.properties) == supersession_of(old_entry), (
            "the node must be the file's supersession, re-projected"
        )
        assert await self._supersedes_edges(raw) == [(NEW_ID, OLD_ID)]
        assert list(await self._recalled(memory)) == [_NEW["decision"]]
        assert "disagree" not in caplog.text, (
            "the node trailing a write this call just made is not a disagreement"
        )

        # The graph is not the only record: a restart re-projects it from the file.
        restarted = self._agent(raw, strategy_path)
        await self._strategic(restarted)
        assert list(await self._recalled(self._memory(restarted))) == [_NEW["decision"]]

    @pytest.mark.parametrize("superseded_in", ["graph", "file"])
    async def test_file_wins_on_disagreement(
        self, raw, strategy_path, caplog, superseded_in
    ):
        from kestrel_sovereign.storage.async_graph_store import GraphNode

        _write_strategy_file(strategy_path, [_OLD, _NEW])
        agent = self._agent(raw, strategy_path)
        await self._strategic(agent)
        if superseded_in == "graph":
            # The graph-only supersession mark_superseded wrote before #4154.
            node = await raw.graph.get_node(OLD_ID)
            await raw.graph.add_node(GraphNode(
                node_id=OLD_ID,
                node_type=node.node_type,
                label=node.label,
                properties={**node.properties, "superseded_by": NEW_ID,
                            "superseded_at": "2026-07-05T00:00:00+00:00"},
            ))
        else:
            # An operator marks it superseded by hand.
            _write_strategy_file(strategy_path, [{**_OLD, "superseded_by": NEW_ID}, _NEW])
        caplog.set_level("WARNING")

        restarted = self._agent(raw, strategy_path)
        await self._strategic(restarted)

        current = await self._recalled(self._memory(restarted))
        if superseded_in == "graph":
            assert set(current) == {_OLD["decision"], _NEW["decision"]}
            assert "superseded_by" not in (await raw.graph.get_node(OLD_ID)).properties
        else:
            assert list(current) == [_NEW["decision"]]
        warnings = [
            r for r in caplog.records
            if r.levelname == "WARNING" and "disagree" in r.getMessage()
        ]
        assert len(warnings) == 1
        assert OLD_ID in warnings[0].getMessage()
        assert _OLD["decision"] not in warnings[0].getMessage(), "ids, not text"

    _OPERATOR_CHOICE = "decision:chosen-by-the-operator"

    @pytest.mark.parametrize("edit", ["same_decision", "elsewhere"])
    async def test_concurrent_hand_edit_refused(
        self, raw, strategy_path, caplog, edit
    ):
        """One hand edit between the read and the write refuses the write.

        It is not re-read and written again: the supersession was decided on
        what the graph showed, and an edit to the same decision would be
        overwritten by an answer decided without seeing it.
        """
        import yaml as _yaml

        from kestrel_sdk.tools.result import ToolResultStatus
        from kestrel_sovereign.features.strategic_memory.decision_index import (
            supersession_of,
        )

        _write_strategy_file(strategy_path, [_OLD, _NEW])
        agent = self._agent(raw, strategy_path)
        strategic = await self._strategic(agent)
        memory = self._memory(agent)
        node_before = (await raw.graph.get_node(OLD_ID)).properties
        write = strategic._write_strategy
        attempts = []

        def hand_edit_then_write(data, expected_digest):
            # The operator saves the file once, after the supersede read it
            # and before it writes.
            if not attempts:
                edited = _read_strategy_file(strategy_path)
                if edit == "same_decision":
                    _decision_in(edited, _OLD["decision"])["superseded_by"] = (
                        self._OPERATOR_CHOICE
                    )
                else:
                    edited["vision"] = "edited by hand"
                strategy_path.write_text(
                    _yaml.dump(edited, sort_keys=False), encoding="utf-8"
                )
            attempts.append(strategy_path.read_bytes())
            return write(data, expected_digest)

        strategic._write_strategy = hand_edit_then_write
        caplog.set_level("WARNING")

        result = await memory.mark_superseded(old_id=OLD_ID, new_id=NEW_ID)

        assert result.status is ToolResultStatus.ERROR, result
        assert result.data["conflict"] is True
        assert result.data["yaml_written"] is False
        assert result.data["edge_written"] is False
        assert result.data["reread_from_disk"] is True
        assert "changed on disk" in result.error
        assert len(attempts) == 1, "refused, not re-read and written again"
        hand_edited = attempts[0]
        assert strategy_path.read_bytes() == hand_edited, "the operator's edit stands"
        assert not list(strategy_path.parent.glob("*.tmp"))
        assert await self._supersedes_edges(raw) == []
        assert strategic._data == _read_strategy_file(strategy_path), (
            "the refusal re-read the file, so a retry is decided on what it holds"
        )

        node = await raw.graph.get_node(OLD_ID)
        old_entry = _decision_in(_read_strategy_file(strategy_path), _OLD["decision"])
        assert supersession_of(node.properties) == supersession_of(old_entry), (
            "the node follows the file as edited, and holds nothing of this call"
        )
        disagreements = [
            r for r in caplog.records
            if r.levelname == "WARNING" and "disagree" in r.getMessage()
        ]
        if edit == "same_decision":
            assert node.properties["superseded_by"] == self._OPERATOR_CHOICE
            assert len(disagreements) == 1, "the file won over the node, logged"
        else:
            assert node.properties == node_before
            assert disagreements == []

        # Seen through recall now, the edit can be superseded deliberately.
        del strategic._write_strategy
        retried = await memory.mark_superseded(old_id=OLD_ID, new_id=NEW_ID)
        assert retried.status is ToolResultStatus.OK, retried
        on_disk = _read_strategy_file(strategy_path)
        assert _decision_in(on_disk, _OLD["decision"])["superseded_by"] == NEW_ID
        if edit == "elsewhere":
            assert on_disk["vision"] == "edited by hand", "the retry keeps the edit"

    async def test_an_edit_since_load_is_refused_once(self, raw, strategy_path):
        """An edit made while the agent ran, before the supersede: the graph
        the caller decided on predates it, so this too is a conflict -- once."""
        from kestrel_sdk.tools.result import ToolResultStatus

        _write_strategy_file(strategy_path, [_OLD, _NEW])
        agent = self._agent(raw, strategy_path)
        await self._strategic(agent)
        memory = self._memory(agent)
        _write_strategy_file(
            strategy_path, [{**_OLD, "superseded_by": self._OPERATOR_CHOICE}, _NEW]
        )
        edited = strategy_path.read_bytes()

        refused = await memory.mark_superseded(old_id=OLD_ID, new_id=NEW_ID)

        assert refused.status is ToolResultStatus.ERROR, refused
        assert refused.data["conflict"] is True
        assert strategy_path.read_bytes() == edited
        recalled = await self._recalled(memory, include_superseded=True)
        assert recalled[_OLD["decision"]]["superseded_by"] == self._OPERATOR_CHOICE
        assert await self._supersedes_edges(raw) == []

        retried = await memory.mark_superseded(old_id=OLD_ID, new_id=NEW_ID)
        assert retried.status is ToolResultStatus.OK, retried
        assert _decision_in(_read_strategy_file(strategy_path), _OLD["decision"])[
            "superseded_by"
        ] == NEW_ID

    async def test_a_reprojection_that_does_not_land_is_partial(
        self, raw, strategy_path, monkeypatch
    ):
        """The file holds it, so the next reindex projects it; until then the
        result must not claim recall already filters the decision out."""
        from kestrel_sdk.tools.result import ToolResultStatus

        _write_strategy_file(strategy_path, [_OLD, _NEW])
        agent = self._agent(raw, strategy_path)
        strategic = await self._strategic(agent)

        async def refused(*args, **kwargs):
            raise RuntimeError("graph write refused")

        monkeypatch.setattr(raw.graph, "compare_and_swap_node", refused)

        result = await self._memory(agent).mark_superseded(old_id=OLD_ID, new_id=NEW_ID)

        assert result.status is ToolResultStatus.PARTIAL, result
        assert result.data["yaml_written"] is True
        assert result.data["property_updated"] is False
        assert "not re-projected" in result.error
        assert _decision_in(_read_strategy_file(strategy_path), _OLD["decision"])[
            "superseded_by"
        ] == NEW_ID
        monkeypatch.undo()
        await strategic._reindex_decisions()
        assert (await raw.graph.get_node(OLD_ID)).properties["superseded_by"] == NEW_ID

    async def test_a_stopped_edge_write_is_not_reported_as_partial(
        self, raw, strategy_path, monkeypatch
    ):
        """An execution-control error from the graph is a Stop, not a failure
        to fold into the result."""
        from kestrel_sovereign.execution_custody import ExecutionAuthorityError

        _write_strategy_file(strategy_path, [_OLD, _NEW])
        agent = self._agent(raw, strategy_path)
        await self._strategic(agent)

        async def stopped(*args, **kwargs):
            raise ExecutionAuthorityError("execution was stopped")

        monkeypatch.setattr(raw.graph, "add_edge", stopped)

        with pytest.raises(ExecutionAuthorityError):
            await self._memory(agent).mark_superseded(old_id=OLD_ID, new_id=NEW_ID)

    async def test_a_stopped_read_back_is_not_reported_as_unprojected(
        self, raw, strategy_path, monkeypatch
    ):
        from kestrel_sovereign.execution_custody import ExecutionAuthorityError

        _write_strategy_file(strategy_path, [_OLD, _NEW])
        agent = self._agent(raw, strategy_path)
        strategic = await self._strategic(agent)

        async def stopped(*args, **kwargs):
            raise ExecutionAuthorityError("execution was stopped")

        monkeypatch.setattr(raw.graph, "get_node", stopped)

        with pytest.raises(ExecutionAuthorityError):
            await strategic.supersede_decision(
                OLD_ID, NEW_ID, superseded_at="2026-07-05T00:00:00+00:00"
            )

    async def test_a_decision_the_file_no_longer_holds_is_not_superseded(
        self, raw, strategy_path
    ):
        from kestrel_sdk.tools.result import ToolResultStatus

        _write_strategy_file(strategy_path, [_OLD, _NEW])
        agent = self._agent(raw, strategy_path)
        await self._strategic(agent)
        _write_strategy_file(strategy_path, [_NEW])
        removed = strategy_path.read_bytes()

        result = await self._memory(agent).mark_superseded(old_id=OLD_ID, new_id=NEW_ID)

        assert result.status is ToolResultStatus.ERROR, result
        assert result.data["conflict"] is True
        assert strategy_path.read_bytes() == removed
        assert await self._supersedes_edges(raw) == []
        assert await raw.graph.get_node(OLD_ID) is None, (
            "re-read, the file no longer holds it, so neither does the index"
        )

    @pytest.mark.parametrize("failure", ["write_fails", "file_removed"])
    async def test_an_unwritten_supersession_is_not_kept_in_memory(
        self, raw, strategy_path, monkeypatch, failure
    ):
        """The change was applied to the in-memory copy before the write; a
        write that does not happen must take it back out, or the next save
        would carry a supersession nobody recorded."""
        from kestrel_sdk.tools.result import ToolResultStatus
        from kestrel_sovereign.features.strategic_memory import feature as strategic_module

        _write_strategy_file(strategy_path, [_OLD, _NEW])
        agent = self._agent(raw, strategy_path)
        strategic = await self._strategic(agent)
        node_before = (await raw.graph.get_node(OLD_ID)).properties
        before = strategy_path.read_bytes()
        if failure == "write_fails":
            def refuse(path, content):
                raise OSError("disk full")

            monkeypatch.setattr(strategic_module, "_replace_atomically", refuse)
        else:
            strategy_path.unlink()

        result = await self._memory(agent).mark_superseded(old_id=OLD_ID, new_id=NEW_ID)

        assert result.status is ToolResultStatus.ERROR, result
        assert result.data["yaml_written"] is False
        assert result.data["conflict"] is (failure == "file_removed")
        assert result.data["reread_from_disk"] is False
        assert not any("superseded_by" in d for d in strategic._data["decisions"])
        assert (await raw.graph.get_node(OLD_ID)).properties == node_before
        assert await self._supersedes_edges(raw) == []
        if failure == "write_fails":
            assert strategy_path.read_bytes() == before
        else:
            assert not strategy_path.exists(), "a removed file is not recreated"

    async def test_a_decision_an_unchanged_file_does_not_hold_is_refused(
        self, raw, strategy_path
    ):
        _write_strategy_file(strategy_path, [_OLD, _NEW])
        agent = self._agent(raw, strategy_path)
        strategic = await self._strategic(agent)
        before = strategy_path.read_bytes()
        data_before = copy.deepcopy(strategic._data)

        outcome = await strategic.supersede_decision(
            "decision:not-in-the-file", NEW_ID,
            superseded_at="2026-07-05T00:00:00+00:00",
        )

        assert not outcome.recorded
        assert outcome.conflict
        assert not outcome.reread, "the file did not change, so there is nothing to re-read"
        assert "holds no decision" in outcome.error
        assert strategy_path.read_bytes() == before
        assert strategic._data == data_before

    @pytest.mark.parametrize("mode", ["EPHEMERAL", "ISOLATED", "DEIDENTIFIED"])
    async def test_refused_in_a_volatile_privacy_mode(self, raw, strategy_path, mode):
        """The graph wrapper refused this supersession in these modes (#2672);
        writing STRATEGY.yaml instead must not make it a durable record."""
        from kestrel_sdk.tools.result import ToolResultStatus
        from kestrel_sovereign.privacy import PrivacyMode
        from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage

        _write_strategy_file(strategy_path, [_OLD, _NEW])
        agent = self._agent(raw, strategy_path)
        strategic = await self._strategic(agent)
        node_before = (await raw.graph.get_node(OLD_ID)).properties
        file_before = strategy_path.read_bytes()
        agent.storage = PrivacyEnforcingStorage(raw, getattr(PrivacyMode, mode))

        result = await self._memory(agent).mark_superseded(
            old_id=OLD_ID, new_id=NEW_ID, reason="said in a volatile session"
        )

        assert result.status is ToolResultStatus.ERROR, result
        assert "privacy mode" in result.error
        assert result.data["yaml_written"] is False
        assert strategy_path.read_bytes() == file_before
        assert not any("superseded_by" in d for d in strategic._data["decisions"])
        assert (await raw.graph.get_node(OLD_ID)).properties == node_before
        assert await self._supersedes_edges(raw) == []

    async def test_refused_without_strategic_memory(self, raw, strategy_path):
        """Writing the graph alone would be undone by the next reindex."""
        from kestrel_sdk.tools.result import ToolResultStatus

        _write_strategy_file(strategy_path, [_OLD, _NEW])
        agent = self._agent(raw, strategy_path)
        await self._strategic(agent)
        agent.features.clear()
        node_before = (await raw.graph.get_node(OLD_ID)).properties
        file_before = strategy_path.read_bytes()

        result = await self._memory(agent).mark_superseded(old_id=OLD_ID, new_id=NEW_ID)

        assert result.status is ToolResultStatus.ERROR, result
        assert "strategic memory is not loaded" in result.error
        assert (await raw.graph.get_node(OLD_ID)).properties == node_before
        assert strategy_path.read_bytes() == file_before
        assert await self._supersedes_edges(raw) == []


class TestProjectionRacingASupersession:
    @pytest.mark.asyncio
    async def test_a_projection_in_flight_does_not_revive_the_decision(self, tmp_path):
        """A reindex that read the entries before a supersession was written
        must not land after it and put the decision back to current."""
        import asyncio

        from kestrel_sovereign.features.strategic_memory.feature import (
            StrategicMemoryFeature,
        )

        class _PausingGraph(_FakeGraph):
            def __init__(self):
                super().__init__()
                self.armed = False
                self.paused = asyncio.Event()
                self.release = asyncio.Event()

            async def get_node(self, node_id):
                if self.armed and node_id == OLD_ID:
                    self.armed = False
                    self.paused.set()
                    await self.release.wait()
                return await super().get_node(node_id)

        _write_strategy_file(tmp_path / "STRATEGY.yaml", [_OLD])
        graph = _PausingGraph()
        agent = MagicMock()
        agent.agent_id = AGENT
        agent.agent_data_dir = str(tmp_path)
        agent.storage = MagicMock()
        agent.storage.graph = graph
        feature = StrategicMemoryFeature(agent)
        await feature.initialize()

        graph.armed = True
        reindex = asyncio.create_task(feature._reindex_decisions())
        await graph.paused.wait()
        supersede = asyncio.create_task(feature.supersede_decision(
            OLD_ID, NEW_ID, superseded_at="2026-07-05T00:00:00+00:00"
        ))
        for _ in range(20):
            await asyncio.sleep(0)
        graph.release.set()
        await reindex
        outcome = await supersede

        assert outcome.recorded and outcome.projected
        assert graph.nodes[OLD_ID].properties["superseded_by"] == NEW_ID
