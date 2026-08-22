"""Patterns and blockers move out of the prompt and into an addressable ledger (#2954).

Three defects motivated this, all observed on Emma's live instance:

* ``strategy_resolve_blocker`` matched on the ``issue`` key and deleted *every*
  row that shared it — one call returned ``removed_count: 10``. Rows had no
  identity, so a row could not be resolved individually.
* ``patterns_learned`` was append-only and reached 358 rows / 175 KB inside a
  266 KB ``STRATEGY.yaml`` that the BootstrapLoader truncates to 20,000
  characters by head+tail byte offset. Under 7% of the file reached the agent,
  and *which* 7% was a function of byte position.
* 57 of 110 blockers referenced GitHub issues that had already closed, with no
  path from live state back to the ledger.

``STRATEGY.yaml`` stays canonical for the standing brief. The two growing logs
become ``STRATEGY_LEDGER.yaml`` — canonical, but reached by query through the
graph projection rather than by injection, exactly as #2851 settled it for
decisions.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml as _yaml

from kestrel_sovereign.features.bootstrap.loader import (
    DEFAULT_BOOTSTRAP_FILES,
    DEFAULT_MAX_CHARS_PER_FILE,
)
from kestrel_sovereign.features.strategic_memory.feature import StrategicMemoryFeature
from kestrel_sovereign.features.strategic_memory.ledger import (
    BLOCKERS_KEY,
    LEDGER_FILENAME,
    PATTERNS_KEY,
    StrategyLedger,
    blocker_row_id,
    pattern_row_id,
)
from kestrel_sovereign.features.strategic_memory.ledger_index import (
    BLOCKER_NODE_TYPE,
    PATTERN_NODE_TYPE,
    ledger_node_id,
    project_ledger,
    search_rows,
)


AGENT = "did:test:ledger"


class _FakeGraph:
    """Stands in for AsyncGraphStore, with real compare-and-swap semantics.

    The swap actually compares stored properties against the caller's snapshot
    and refuses when they differ. A double that always succeeded would let the
    projection pass every test while still clobbering a concurrent write in
    production.
    """

    def __init__(self, *, fail_on=None, fail_read_on=None):
        self.nodes = {}
        self.writes = 0
        self.deleted = []
        self._fail_on = fail_on
        self._fail_read_on = fail_read_on

    async def get_node(self, node_id):
        if self._fail_read_on and self._fail_read_on in node_id:
            raise RuntimeError("graph read refused")
        return self.nodes.get(node_id)

    async def get_nodes_by_type(self, node_type):
        return [n for n in self.nodes.values() if n.node_type == node_type]

    async def delete_node(self, node_id):
        self.deleted.append(node_id)
        self.nodes.pop(node_id, None)

    async def compare_and_swap_node(self, node_id, expected, new_node):
        from kestrel_sovereign.storage.async_graph_store import NodeSwapResult

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
        current.properties = new_node.properties
        return NodeSwapResult.SWAPPED


async def _feature(tmp_path, *, graph=None, strategy=None):
    agent = MagicMock()
    agent.agent_id = AGENT
    agent.agent_data_dir = str(tmp_path)
    agent.storage = MagicMock()
    agent.storage.graph = graph if graph is not None else _FakeGraph()
    if strategy is not None:
        (tmp_path / "STRATEGY.yaml").write_text(
            _yaml.dump(strategy), encoding="utf-8"
        )
    feature = StrategicMemoryFeature(agent)
    await feature.initialize()
    return feature


# ---------------------------------------------------------------------------
# The split itself
# ---------------------------------------------------------------------------


class TestTheLedgerIsNotABootstrapFile:
    def test_ledger_is_absent_from_the_bootstrap_file_list(self):
        """Adding it there would undo the entire point of the split."""
        assert LEDGER_FILENAME not in DEFAULT_BOOTSTRAP_FILES
        assert "STRATEGY.yaml" in DEFAULT_BOOTSTRAP_FILES

    @pytest.mark.asyncio
    async def test_migration_leaves_strategy_yaml_under_the_truncation_cap(
        self, tmp_path
    ):
        """The acceptance bar: STRATEGY.yaml must load whole after migration."""
        bulky = {
            "version": 1,
            "vision": "Ship the sovereign agent.",
            "milestones": [{"name": "M1", "status": "in_progress"}],
            PATTERNS_KEY: [
                {
                    "pattern": f"pattern number {i} " + ("x" * 400),
                    "implication": "y" * 200,
                }
                for i in range(358)
            ],
            BLOCKERS_KEY: [
                {"issue": f"#{i}", "title": f"blocker {i}", "severity": "high"}
                for i in range(80)
            ],
        }
        path = tmp_path / "STRATEGY.yaml"
        path.write_text(_yaml.dump(bulky), encoding="utf-8")
        assert len(path.read_text(encoding="utf-8")) > DEFAULT_MAX_CHARS_PER_FILE

        await _feature(tmp_path)

        after = path.read_text(encoding="utf-8")
        assert len(after) < DEFAULT_MAX_CHARS_PER_FILE, (
            "the whole point of moving the logs out is that the brief loads "
            "without truncation"
        )
        assert "pattern number 0" not in after
        assert "blocker 0" not in after

    @pytest.mark.asyncio
    async def test_migration_moves_every_row_without_loss(self, tmp_path):
        await _feature(
            tmp_path,
            strategy={
                "version": 1,
                PATTERNS_KEY: [{"pattern": f"p{i}"} for i in range(297)],
                BLOCKERS_KEY: [
                    {"issue": f"#{i}", "title": f"b{i}"} for i in range(57)
                ],
            },
        )

        ledger = _yaml.safe_load(
            (tmp_path / LEDGER_FILENAME).read_text(encoding="utf-8")
        )
        assert len(ledger[PATTERNS_KEY]) == 297
        assert len(ledger[BLOCKERS_KEY]) == 57
        assert {r["pattern"] for r in ledger[PATTERNS_KEY]} == {
            f"p{i}" for i in range(297)
        }

        strategy = _yaml.safe_load(
            (tmp_path / "STRATEGY.yaml").read_text(encoding="utf-8")
        )
        assert PATTERNS_KEY not in strategy
        assert BLOCKERS_KEY not in strategy
        # A breadcrumb, so a human opening the file does not conclude the rows
        # were lost.
        assert strategy["ledger_file"] == LEDGER_FILENAME

    @pytest.mark.asyncio
    async def test_the_empty_patterns_key_collision_is_folded_not_ignored(
        self, tmp_path
    ):
        """``patterns`` (empty) sat beside ``patterns_learned`` (358 rows).

        Two names for one concept is the shape of defect where a reader binds
        to the wrong one and reports a truthful zero against nothing. Migration
        folds any rows found under the stray key in and drops it.
        """
        await _feature(
            tmp_path,
            strategy={
                "version": 1,
                "patterns": [{"pattern": "written under the stray key"}],
                PATTERNS_KEY: [{"pattern": "written under the real key"}],
            },
        )

        ledger = _yaml.safe_load(
            (tmp_path / LEDGER_FILENAME).read_text(encoding="utf-8")
        )
        assert {r["pattern"] for r in ledger[PATTERNS_KEY]} == {
            "written under the stray key",
            "written under the real key",
        }
        strategy = _yaml.safe_load(
            (tmp_path / "STRATEGY.yaml").read_text(encoding="utf-8")
        )
        assert "patterns" not in strategy

    def test_the_default_template_no_longer_ships_the_stray_key(self):
        assert "patterns" not in StrategicMemoryFeature._DEFAULT_TEMPLATE
        assert PATTERNS_KEY not in StrategicMemoryFeature._DEFAULT_TEMPLATE
        assert BLOCKERS_KEY not in StrategicMemoryFeature._DEFAULT_TEMPLATE

    @pytest.mark.asyncio
    async def test_rerunning_migration_converges_instead_of_doubling(
        self, tmp_path
    ):
        feature = await _feature(
            tmp_path,
            strategy={"version": 1, PATTERNS_KEY: [{"pattern": "only once"}]},
        )
        # Simulate an interrupted run: the rows are back in STRATEGY.yaml while
        # the ledger already holds them.
        feature._data[PATTERNS_KEY] = [{"pattern": "only once"}]
        feature._migrate_ledger_sections()

        assert len(feature._ledger.patterns) == 1

    @pytest.mark.asyncio
    async def test_a_failed_ledger_write_leaves_the_rows_in_strategy_yaml(
        self, tmp_path
    ):
        """No ordering may lose a row. If the ledger cannot be written, the
        old file keeps its copy — duplicated is recoverable, gone is not."""
        agent = MagicMock()
        agent.agent_id = AGENT
        agent.agent_data_dir = str(tmp_path)
        agent.storage = MagicMock()
        agent.storage.graph = _FakeGraph()
        (tmp_path / "STRATEGY.yaml").write_text(
            _yaml.dump({"version": 1, PATTERNS_KEY: [{"pattern": "keep me"}]}),
            encoding="utf-8",
        )

        feature = StrategicMemoryFeature(agent)
        with patch.object(StrategyLedger, "save", return_value="disk full"):
            await feature.initialize()

        strategy = _yaml.safe_load(
            (tmp_path / "STRATEGY.yaml").read_text(encoding="utf-8")
        )
        assert strategy[PATTERNS_KEY] == [{"pattern": "keep me"}]


# ---------------------------------------------------------------------------
# Row identity — the removed_count: 10 defect
# ---------------------------------------------------------------------------


class TestRowsAreIndividuallyAddressable:
    @pytest.mark.asyncio
    async def test_every_migrated_row_gets_an_id(self, tmp_path):
        feature = await _feature(
            tmp_path,
            strategy={
                "version": 1,
                PATTERNS_KEY: [{"pattern": "a"}, {"pattern": "b"}],
                BLOCKERS_KEY: [{"issue": "#1", "title": "t"}],
            },
        )
        ids = [r["id"] for r in feature._ledger.patterns + feature._ledger.blockers]
        assert all(ids)
        assert len(set(ids)) == len(ids)

    def test_byte_identical_rows_still_get_distinct_ids(self):
        """Two rows with identical content are still two rows."""
        ledger = StrategyLedger(None)
        first = ledger.add_blocker("#2877", "same", "high", "me")
        second = ledger.add_blocker("#2877", "same", "high", "me")
        assert first["id"] != second["id"]

    def test_ids_are_content_derived_so_reprojection_upserts(self):
        assert pattern_row_id({"pattern": "x"}) == pattern_row_id({"pattern": " X "})
        assert blocker_row_id({"issue": "#1", "title": "t"}) != blocker_row_id(
            {"issue": "#1", "title": "other"}
        )

    def test_existing_ids_are_never_rewritten(self):
        """An id is an address; rewriting one breaks every outside reference."""
        ledger = StrategyLedger(None)
        ledger.data[BLOCKERS_KEY] = [{"id": "blk_hand_written", "issue": "#1", "title": "t"}]
        ledger.normalize()
        assert ledger.blockers[0]["id"] == "blk_hand_written"

    @pytest.mark.asyncio
    async def test_resolving_one_blocker_leaves_its_siblings_alone(self, tmp_path):
        """#2877 carried ten rows; one call used to take all ten."""
        feature = await _feature(tmp_path, strategy={"version": 1})
        for n in range(10):
            await feature.strategy_add_blocker(
                issue="#2877", title=f"symptom {n}", severity="high"
            )
        target = feature._ledger.blockers[3]["id"]

        result = await feature.strategy_resolve_blocker(issue=target)

        assert result.status.value == "ok"
        assert result.data["removed_count"] == 1
        active = [b for b in feature._ledger.blockers if not b.get("resolved_at")]
        assert len(active) == 9
        assert target not in [b["id"] for b in active]

    @pytest.mark.asyncio
    async def test_an_ambiguous_issue_key_is_refused_not_bulk_deleted(
        self, tmp_path
    ):
        feature = await _feature(tmp_path, strategy={"version": 1})
        for n in range(10):
            await feature.strategy_add_blocker(
                issue="#2877", title=f"symptom {n}", severity="high"
            )

        result = await feature.strategy_resolve_blocker(issue="#2877")

        assert result.status.value == "error"
        assert result.data["ambiguous"] is True
        assert len(result.data["candidate_ids"]) == 10
        assert result.data["removed_count"] == 0
        assert all(not b.get("resolved_at") for b in feature._ledger.blockers)

    @pytest.mark.asyncio
    async def test_a_unique_issue_key_still_resolves(self, tmp_path):
        """The handle the agent already uses must keep working."""
        feature = await _feature(tmp_path, strategy={"version": 1})
        await feature.strategy_add_blocker(issue="#42", title="only one")

        result = await feature.strategy_resolve_blocker(issue="#42")

        assert result.status.value == "ok"
        assert feature._ledger.blockers[0]["resolved_at"]

    @pytest.mark.asyncio
    async def test_a_resolved_blocker_is_kept_as_history_not_deleted(
        self, tmp_path
    ):
        feature = await _feature(tmp_path, strategy={"version": 1})
        await feature.strategy_add_blocker(issue="#42", title="was blocking")
        await feature.strategy_resolve_blocker(issue="#42", resolution="shipped")

        rows = feature._ledger.blockers
        assert len(rows) == 1, "the row is the only record the blocker existed"
        assert rows[0]["resolution"] == "shipped"
        # ...but it is out of the active view the briefing reads.
        assert feature._strategy_data_view()[BLOCKERS_KEY] == []


# ---------------------------------------------------------------------------
# Supersession — no more append-only growth
# ---------------------------------------------------------------------------


class TestSupersession:
    @pytest.mark.asyncio
    async def test_a_superseded_pattern_drops_out_of_the_active_set(self, tmp_path):
        feature = await _feature(tmp_path, strategy={"version": 1})
        added = await feature.strategy_add_pattern("cron is reliable enough")
        pattern_id = added.data["pattern_id"]

        result = await feature.strategy_supersede_pattern(
            pattern_id=pattern_id, reason="durable leases replaced it"
        )

        assert result.status.value == "ok"
        assert feature._strategy_data_view()[PATTERNS_KEY] == []
        # Kept as history, not deleted.
        assert len(feature._ledger.patterns) == 1
        assert feature._ledger.patterns[0]["superseded_reason"] == (
            "durable leases replaced it"
        )

    @pytest.mark.asyncio
    async def test_superseding_an_unknown_id_is_refused(self, tmp_path):
        feature = await _feature(tmp_path, strategy={"version": 1})
        result = await feature.strategy_supersede_pattern(pattern_id="pat_nope")
        assert result.status.value == "error"
        assert result.data["superseded"] is False

    @pytest.mark.asyncio
    async def test_a_replacement_that_does_not_exist_is_refused(self, tmp_path):
        """Recording a pointer to a row that is not there is worse than none."""
        feature = await _feature(tmp_path, strategy={"version": 1})
        added = await feature.strategy_add_pattern("old")

        result = await feature.strategy_supersede_pattern(
            pattern_id=added.data["pattern_id"], superseded_by="pat_missing"
        )

        assert result.status.value == "error"
        assert not feature._ledger.patterns[0].get("superseded_at")

    @pytest.mark.asyncio
    async def test_superseding_twice_is_refused(self, tmp_path):
        feature = await _feature(tmp_path, strategy={"version": 1})
        added = await feature.strategy_add_pattern("old")
        pattern_id = added.data["pattern_id"]
        await feature.strategy_supersede_pattern(pattern_id=pattern_id)

        result = await feature.strategy_supersede_pattern(pattern_id=pattern_id)

        assert result.status.value == "error"


# ---------------------------------------------------------------------------
# The graph projection — same contract as the decision index (#2851)
# ---------------------------------------------------------------------------


class TestProjection:
    @pytest.mark.asyncio
    async def test_ledger_rows_become_reachable_nodes(self, tmp_path):
        graph = _FakeGraph()
        await _feature(
            tmp_path,
            graph=graph,
            strategy={
                "version": 1,
                PATTERNS_KEY: [{"pattern": "leases beat cron"}],
                BLOCKERS_KEY: [{"issue": "#7", "title": "waiting on review"}],
            },
        )

        by_type = {n.node_type for n in graph.nodes.values()}
        assert by_type == {PATTERN_NODE_TYPE, BLOCKER_NODE_TYPE}
        for node in graph.nodes.values():
            # Load-bearing: scoped queries filter on agent_id, and a node
            # missing it is written but unreachable.
            assert node.properties["agent_id"] == AGENT
            assert node.properties["claim_source"] == "strategy_ledger_yaml"
            assert node.properties["row_id"]

    @pytest.mark.asyncio
    async def test_reprojection_upserts_rather_than_duplicating(self):
        graph = _FakeGraph()
        data = {
            PATTERNS_KEY: [{"id": "pat_1", "pattern": "x"}],
            BLOCKERS_KEY: [{"id": "blk_1", "issue": "#1", "title": "t"}],
        }
        for _ in range(3):
            await project_ledger(graph, AGENT, data)

        assert len(graph.nodes) == 2

    @pytest.mark.asyncio
    async def test_a_row_removed_from_the_ledger_stops_being_reachable(self):
        graph = _FakeGraph()
        await project_ledger(
            graph,
            AGENT,
            {PATTERNS_KEY: [{"id": "pat_1", "pattern": "keep"},
                            {"id": "pat_2", "pattern": "drop"}]},
        )
        assert len(graph.nodes) == 2

        report = await project_ledger(
            graph, AGENT, {PATTERNS_KEY: [{"id": "pat_1", "pattern": "keep"}]}
        )

        assert report["removed"] == 1
        assert [n.label for n in graph.nodes.values()] == ["keep"]

    @pytest.mark.asyncio
    async def test_reconcile_leaves_other_agents_and_foreign_nodes_alone(self):
        from kestrel_sovereign.storage.async_graph_store import GraphNode

        graph = _FakeGraph()
        await project_ledger(
            graph, AGENT, {PATTERNS_KEY: [{"id": "pat_1", "pattern": "mine"}]}
        )
        graph.nodes["other-agent"] = GraphNode(
            node_id="other-agent",
            node_type=PATTERN_NODE_TYPE,
            label="theirs",
            properties={"agent_id": "did:test:other", "source": "strategic_memory"},
        )
        graph.nodes["hand-written"] = GraphNode(
            node_id="hand-written",
            node_type=PATTERN_NODE_TYPE,
            label="not a projection",
            properties={"agent_id": AGENT, "source": "somewhere_else"},
        )

        await project_ledger(graph, AGENT, {PATTERNS_KEY: []})

        assert graph.deleted == [
            ledger_node_id(PATTERN_NODE_TYPE, AGENT, "pat_1")
        ]
        assert "other-agent" in graph.nodes
        assert "hand-written" in graph.nodes

    @pytest.mark.asyncio
    async def test_a_failed_read_does_not_write_a_node_without_its_graph_state(self):
        """Treating a failed read as "absent" would revive a superseded row."""
        node_id = ledger_node_id(PATTERN_NODE_TYPE, AGENT, "pat_1")
        graph = _FakeGraph()
        rows = {PATTERNS_KEY: [{"id": "pat_1", "pattern": "x"}]}
        await project_ledger(graph, AGENT, rows)
        graph.nodes[node_id].properties["superseded_by"] = "pat_newer"
        graph._fail_read_on = node_id

        report = await project_ledger(graph, AGENT, rows)

        assert report["failed"] == 1
        assert graph.nodes[node_id].properties["superseded_by"] == "pat_newer"

    @pytest.mark.asyncio
    async def test_the_canonical_file_wins_over_graph_owned_supersession(self):
        """The ledger *can* express supersession, unlike STRATEGY.yaml decisions.

        Preserving the graph's copy unconditionally would make un-superseding a
        row in the canonical file impossible to apply.
        """
        node_id = ledger_node_id(PATTERN_NODE_TYPE, AGENT, "pat_1")
        graph = _FakeGraph()
        await project_ledger(
            graph, AGENT, {PATTERNS_KEY: [{"id": "pat_1", "pattern": "x"}]}
        )
        graph.nodes[node_id].properties["superseded_by"] = "pat_stale"

        await project_ledger(
            graph,
            AGENT,
            {PATTERNS_KEY: [{"id": "pat_1", "pattern": "x",
                             "superseded_by": "pat_current",
                             "superseded_at": "2026-08-21"}]},
        )

        props = graph.nodes[node_id].properties
        assert props["superseded_by"] == "pat_current"
        assert props["status"] == "superseded"

    @pytest.mark.asyncio
    async def test_graph_supersession_survives_where_the_ledger_is_silent(self):
        node_id = ledger_node_id(PATTERN_NODE_TYPE, AGENT, "pat_1")
        graph = _FakeGraph()
        rows = {PATTERNS_KEY: [{"id": "pat_1", "pattern": "x"}]}
        await project_ledger(graph, AGENT, rows)
        graph.nodes[node_id].properties["superseded_by"] = "pat_newer"

        await project_ledger(graph, AGENT, rows)

        assert graph.nodes[node_id].properties["superseded_by"] == "pat_newer"

    @pytest.mark.asyncio
    async def test_missing_graph_store_is_not_an_error(self):
        """The ledger is already on disk; a missing index is not a failure."""
        report = await project_ledger(
            None, AGENT, {PATTERNS_KEY: [{"id": "pat_1", "pattern": "x"}]}
        )
        assert report["projected"] == 0
        assert report["skipped_reason"] == "no_graph_store"

    @pytest.mark.asyncio
    async def test_a_ledger_write_still_succeeds_when_the_graph_fails(
        self, tmp_path
    ):
        """The index must never be able to fail a canonical write."""
        feature = await _feature(
            tmp_path, graph=_FakeGraph(fail_on="strategy"), strategy={"version": 1}
        )

        result = await feature.strategy_add_pattern("ship it")

        assert result.data["recorded"] is True
        on_disk = (tmp_path / LEDGER_FILENAME).read_text(encoding="utf-8")
        assert "ship it" in on_disk

    @pytest.mark.asyncio
    async def test_projection_never_writes_back_to_the_ledger(self, tmp_path):
        """A derived index that edits its source is no longer derived."""
        feature = await _feature(
            tmp_path,
            strategy={"version": 1, PATTERNS_KEY: [{"pattern": "x"}]},
        )
        before = (tmp_path / LEDGER_FILENAME).read_text(encoding="utf-8")

        await feature._reindex_ledger()

        assert (tmp_path / LEDGER_FILENAME).read_text(encoding="utf-8") == before

    @pytest.mark.asyncio
    async def test_a_row_that_never_reached_the_ledger_is_not_indexed(
        self, tmp_path
    ):
        graph = _FakeGraph()
        feature = await _feature(tmp_path, graph=graph, strategy={"version": 1})
        graph.nodes.clear()
        feature._ledger.save = MagicMock(return_value="disk full")

        result = await feature.strategy_add_pattern("never written")

        assert result.data["persisted"] is False
        assert graph.nodes == {}


# ---------------------------------------------------------------------------
# The query layer that replaced prompt injection
# ---------------------------------------------------------------------------


class TestQueryLayer:
    @pytest.mark.asyncio
    async def test_search_reaches_a_pattern_the_prompt_would_have_truncated(
        self, tmp_path
    ):
        feature = await _feature(
            tmp_path,
            strategy={
                "version": 1,
                PATTERNS_KEY: [{"pattern": f"filler {i}"} for i in range(400)]
                + [{"pattern": "axolotls regenerate limbs"}],
            },
        )

        result = await feature.strategy_search(query="axolotls")

        assert result.status.value == "ok"
        assert result.data["count"] == 1
        assert "axolotl" in result.data["matches"][0]["row"]["pattern"]

    @pytest.mark.asyncio
    async def test_search_can_be_scoped_to_one_kind(self, tmp_path):
        feature = await _feature(tmp_path, strategy={"version": 1})
        await feature.strategy_add_pattern("review latency is the bottleneck")
        await feature.strategy_add_blocker(issue="#1", title="review latency")

        patterns = await feature.strategy_search(query="latency", kind="patterns")
        blockers = await feature.strategy_search(query="latency", kind="blockers")

        assert [m["kind"] for m in patterns.data["matches"]] == ["pattern"]
        assert [m["kind"] for m in blockers.data["matches"]] == ["blocker"]

    @pytest.mark.asyncio
    async def test_search_rejects_an_unknown_kind_rather_than_searching_all(
        self, tmp_path
    ):
        feature = await _feature(tmp_path, strategy={"version": 1})
        result = await feature.strategy_search(query="x", kind="garbage")
        assert result.status.value == "error"

    def test_the_id_digest_does_not_manufacture_matches(self):
        """Ids are hex, so substring-matching them scores unrelated rows.

        Observed: querying "pattern 357" returned a row whose text was
        "pattern 158" at a perfect score, because its id happened to contain
        "357". A search that confidently ranks the wrong row is worse than one
        that finds nothing.
        """
        data = {
            PATTERNS_KEY: [
                {"id": "pat_aa357bb", "pattern": "pattern 158"},
                {"id": "pat_ffffff0", "pattern": "pattern 357"},
            ]
        }
        matches = search_rows(data, "pattern 357")
        by_text = {m["row"]["pattern"]: m["score"] for m in matches}
        assert by_text["pattern 357"] == 1.0
        # It may still surface on the shared word "pattern" -- this is an
        # any-token search. What it must not do is claim both terms matched.
        assert by_text["pattern 158"] < 1.0
        assert matches[0]["row"]["pattern"] == "pattern 357"

    def test_a_row_is_still_findable_by_pasting_its_exact_id(self):
        data = {PATTERNS_KEY: [{"id": "pat_abc123", "pattern": "x"}]}
        assert [m["id"] for m in search_rows(data, "pat_abc123")] == ["pat_abc123"]

    def test_search_excludes_retired_rows_by_default(self):
        data = {
            PATTERNS_KEY: [
                {"id": "pat_1", "pattern": "retired idea", "superseded_at": "2026-01-01"},
            ]
        }
        assert search_rows(data, "retired") == []
        assert len(search_rows(data, "retired", include_retired=True)) == 1

    @pytest.mark.asyncio
    async def test_the_view_says_how_many_patterns_it_withheld(self, tmp_path):
        """A capped list that reads like a complete one is the same defect as
        a prompt truncated at a byte offset, only quieter."""
        feature = await _feature(
            tmp_path,
            strategy={
                "version": 1,
                PATTERNS_KEY: [{"pattern": f"p{i}"} for i in range(60)],
            },
        )

        result = await feature.strategy_view(section="patterns")

        assert "35 more active pattern(s) not shown" in result.confirmation
        assert "strategy_search" in result.confirmation


# ---------------------------------------------------------------------------
# Live GitHub reconciliation
# ---------------------------------------------------------------------------


class TestBlockerReconciliation:
    @staticmethod
    def _strategy_with_repos():
        return {
            "version": 1,
            "morning_signal_config": {"scan_repos": ["Owner/repo"]},
        }

    @pytest.mark.asyncio
    async def test_a_closed_issue_is_reported_but_not_applied_by_default(
        self, tmp_path
    ):
        feature = await _feature(
            tmp_path,
            strategy={
                **self._strategy_with_repos(),
                BLOCKERS_KEY: [{"issue": "#2877", "title": "stale"}],
            },
        )

        with patch(
            "kestrel_sovereign.features.strategic_memory.blocker_reconcile."
            "get_github_token",
            return_value="t",
        ), patch(
            "kestrel_sovereign.features.strategic_memory.blocker_reconcile."
            "github_api_get",
            new=AsyncMock(return_value={"state": "closed"}),
        ):
            result = await feature.strategy_reconcile_blockers()

        assert result.status.value == "partial", "a report-only run must say so"
        assert result.data["closed_count"] == 1
        assert result.data["applied"] is False
        assert not feature._ledger.blockers[0].get("resolved_at")

    @pytest.mark.asyncio
    async def test_apply_resolves_only_the_closed_rows(self, tmp_path):
        feature = await _feature(
            tmp_path,
            strategy={
                **self._strategy_with_repos(),
                BLOCKERS_KEY: [
                    {"issue": "#1", "title": "closed upstream"},
                    {"issue": "#2", "title": "still open"},
                ],
            },
        )
        states = {1: {"state": "closed"}, 2: {"state": "open"}}

        async def _fake_get(path, token, **kwargs):
            return states[int(path.rsplit("/", 1)[-1])]

        with patch(
            "kestrel_sovereign.features.strategic_memory.blocker_reconcile."
            "get_github_token",
            return_value="t",
        ), patch(
            "kestrel_sovereign.features.strategic_memory.blocker_reconcile."
            "github_api_get",
            new=_fake_get,
        ):
            result = await feature.strategy_reconcile_blockers(apply="yes")

        assert result.status.value == "ok"
        by_issue = {b["issue"]: b for b in feature._ledger.blockers}
        assert by_issue["#1"]["resolved_at"]
        assert not by_issue["#2"].get("resolved_at")

    @pytest.mark.asyncio
    async def test_a_missing_token_is_a_failure_not_zero_stale_blockers(
        self, tmp_path
    ):
        """Reporting "0 stale blockers" off a check that never ran is the same
        shape of lie this ticket was filed about."""
        feature = await _feature(
            tmp_path,
            strategy={
                **self._strategy_with_repos(),
                BLOCKERS_KEY: [{"issue": "#1", "title": "t"}],
            },
        )

        with patch(
            "kestrel_sovereign.features.strategic_memory.blocker_reconcile."
            "get_github_token",
            return_value=None,
        ):
            result = await feature.strategy_reconcile_blockers(apply="yes")

        assert result.status.value == "error"
        assert "GITHUB_TOKEN" in result.error
        assert result.data["applied"] is False

    @pytest.mark.asyncio
    async def test_an_unlookupable_row_is_not_reported_as_still_blocking(
        self, tmp_path
    ):
        feature = await _feature(
            tmp_path,
            strategy={
                **self._strategy_with_repos(),
                BLOCKERS_KEY: [{"issue": "not-a-number", "title": "t"}],
            },
        )

        with patch(
            "kestrel_sovereign.features.strategic_memory.blocker_reconcile."
            "get_github_token",
            return_value="t",
        ):
            result = await feature.strategy_reconcile_blockers()

        report = result.data["report"]
        assert report["open"] == []
        assert len(report["unresolvable"]) == 1


# ---------------------------------------------------------------------------
# The file split must not empty the surfaces that read blockers
# ---------------------------------------------------------------------------


class TestExistingReadersStillSeeBlockers:
    @pytest.mark.asyncio
    async def test_the_briefing_and_dispatch_read_the_merged_view(self, tmp_path):
        feature = await _feature(
            tmp_path,
            strategy={
                "version": 1,
                BLOCKERS_KEY: [
                    {"issue": "#9", "title": "still blocking", "severity": "critical"}
                ],
            },
        )

        view = feature._strategy_data_view()
        assert [b["title"] for b in view[BLOCKERS_KEY]] == ["still blocking"]
        # ...and the merged view is a copy: merging into ``_data`` would put the
        # rows back into whatever ``_save()`` writes to STRATEGY.yaml.
        assert BLOCKERS_KEY not in feature._data

        rendered = await feature.strategy_view(section="blockers")
        assert "still blocking" in rendered.confirmation
        assert "id:" in rendered.confirmation


class TestUnreadableLedgerCannotDeleteTheIndex:
    """An unreadable ledger is not an empty one.

    Reconciliation derives its keep-set from the rows it is given, so a failed
    parse reads as "every row was deleted" and takes the derived index with it
    — a parse error escalated into data loss. The guard lives in the projector
    rather than only at its caller, because a bare mapping cannot express the
    difference and the next call site added would omit it.
    """

    @pytest.mark.asyncio
    async def test_an_unreadable_ledger_projects_nothing_and_deletes_nothing(self):
        from kestrel_sovereign.features.strategic_memory.ledger_index import (
            project_ledger,
        )

        graph = _FakeGraph()
        rows = {PATTERNS_KEY: [{"id": "p1", "pattern": "keep me"}], BLOCKERS_KEY: []}
        await project_ledger(graph, AGENT, rows)
        assert len(graph.nodes) == 1

        class _UnreadableLedger:
            readable = False
            load_error = "while parsing a block mapping"
            data = {PATTERNS_KEY: [], BLOCKERS_KEY: []}

        report = await project_ledger(graph, AGENT, _UnreadableLedger())

        assert report["skipped_reason"] == "ledger_unavailable"
        assert report["removed"] == 0
        assert len(graph.nodes) == 1, "the valid index must survive a parse failure"

    @pytest.mark.asyncio
    async def test_a_readable_ledger_still_reconciles_removals(self):
        """The guard must not disable reconciliation for a healthy ledger."""
        from kestrel_sovereign.features.strategic_memory.ledger_index import (
            project_ledger,
        )

        graph = _FakeGraph()
        await project_ledger(
            graph, AGENT,
            {PATTERNS_KEY: [{"id": "p1", "pattern": "a"}, {"id": "p2", "pattern": "b"}],
             BLOCKERS_KEY: []},
        )
        assert len(graph.nodes) == 2

        class _ReadableLedger:
            readable = True
            load_error = None
            data = {PATTERNS_KEY: [{"id": "p1", "pattern": "a"}], BLOCKERS_KEY: []}

        report = await project_ledger(graph, AGENT, _ReadableLedger())

        assert report["removed"] == 1
        assert len(graph.nodes) == 1
