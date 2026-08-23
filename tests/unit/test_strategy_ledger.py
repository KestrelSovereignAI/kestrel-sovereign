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

import copy
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
    MEMBERSHIP_READ_CAP,
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

    def __init__(self, *, fail_on=None, fail_read_on=None, fail_query_on=None):
        self.nodes = {}
        self.writes = 0
        self.deleted = []
        self._fail_on = fail_on
        self._fail_read_on = fail_read_on
        self._fail_query_on = fail_query_on

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

    async def query_nodes_by_type_and_property(
        self, node_type, filters=None, *, created_since=None,
        order_by_created=True, limit=200,
    ):
        """Mirrors AsyncGraphStore: equality filters, created_at DESC, limit.

        ``TestRecallAgainstRealGraphStore`` below runs the same consumer
        against the real store, so this double is a convenience rather than
        the only thing the recall path is ever proven against.
        """
        if self._fail_query_on and self._fail_query_on in node_type:
            raise RuntimeError("graph query refused")
        rows = [n for n in self.nodes.values() if n.node_type == node_type]
        for key, value in (filters or {}).items():
            rows = [n for n in rows if (n.properties or {}).get(key) == value]
        if created_since is not None:
            rows = [
                n for n in rows
                if str((n.properties or {}).get("created_at") or "") >= created_since
            ]
        if order_by_created:
            rows.sort(
                key=lambda n: str((n.properties or {}).get("created_at") or ""),
                reverse=True,
            )
        return rows[: max(1, min(int(limit), 10000))]


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


class TestTheIndexHasAConsumer:
    """The projection is read back through the graph, not through the file.

    #2851 shipped a decision projection whose only consumer was itself, and the
    gap stayed invisible because every reader went to YAML. #3051 scopes the
    fix for patterns and blockers to exactly this: a structural consumer that
    queries the projected node types. Without one, "reachable by query" is a
    claim about a write nobody checked.
    """

    @pytest.mark.asyncio
    async def test_recall_patterns_reads_the_graph_not_the_ledger(self, tmp_path):
        feature = await _feature(tmp_path)
        await feature.strategy_add_pattern(
            pattern="reviews stall on ambiguous ownership",
            implication="name an owner per row",
        )

        result = await feature.recall_patterns()

        assert result.data["count"] == 1
        row = result.data["patterns"][0]
        assert row["text"] == "reviews stall on ambiguous ownership"
        assert row["implication"] == "name an owner per row"
        # Proves the answer came through the projection: node_id is the graph's
        # address, and nothing in the YAML row carries it.
        assert row["node_id"].startswith(f"{PATTERN_NODE_TYPE}:{AGENT}:")

    @pytest.mark.asyncio
    async def test_recall_reads_the_index_even_when_the_file_is_gone(self, tmp_path):
        """The index is a genuine second copy, not a view over the loaded file."""
        feature = await _feature(tmp_path)
        await feature.strategy_add_pattern(pattern="a durable observation")
        graph = feature.agent.storage.graph

        # Empty the in-memory ledger. A reader that had been going to YAML all
        # along would now report zero.
        feature._ledger.data[PATTERNS_KEY] = []

        result = await feature.recall_patterns()
        assert result.data["count"] == 1
        assert result.data["patterns"][0]["text"] == "a durable observation"
        # The row still comes back -- that is the claim here. It is also now
        # reported as a divergence, because an in-memory ledger holding none
        # of it and an index holding one of it do disagree.
        assert result.status.value == "partial"
        assert result.data["orphaned_count"] == 1
        # Count the pattern nodes specifically. ``len(graph.nodes)`` would also
        # count decision and blocker nodes, which is a different claim than the
        # one this test is making.
        pattern_nodes = [
            n for n in graph.nodes.values() if n.node_type == PATTERN_NODE_TYPE
        ]
        assert len(pattern_nodes) == 1

    @pytest.mark.asyncio
    async def test_recall_blockers_reads_the_graph(self, tmp_path):
        feature = await _feature(tmp_path)
        await feature.strategy_add_blocker(
            issue="owner/repo#42", title="CI runner is wedged", severity="high"
        )

        result = await feature.recall_blockers()

        assert result.data["count"] == 1
        row = result.data["blockers"][0]
        assert row["text"] == "CI runner is wedged"
        assert row["issue"] == "owner/repo#42"
        assert row["severity"] == "high"
        assert row["node_id"].startswith(f"{BLOCKER_NODE_TYPE}:{AGENT}:")

    @pytest.mark.asyncio
    async def test_a_recalled_blocker_carries_its_repository(self, tmp_path):
        """#3064: the projection dropped ``repo``.

        ``#42`` names an issue only relative to a repository, and the ledger
        records the field for exactly that reason. Losing it in the index
        hands every consumer an ambiguous reference -- in the same change
        that added repository-identity guards elsewhere.
        """
        feature = await _feature(tmp_path)
        await feature.strategy_add_blocker(
            issue="#42",
            title="CI runner is wedged",
            severity="high",
            repo="owner/repo",
        )

        result = await feature.recall_blockers()

        row = result.data["blockers"][0]
        assert row["issue"] == "#42"
        assert row["repo"] == "owner/repo", (
            "a bare issue number without its repo is not a reference"
        )

    @pytest.mark.asyncio
    async def test_superseded_patterns_are_excluded_by_default(self, tmp_path):
        feature = await _feature(tmp_path)
        added = await feature.strategy_add_pattern(pattern="held once, not now")
        pattern_id = added.data["pattern_id"]
        await feature.strategy_supersede_pattern(
            pattern_id=pattern_id, reason="measured again"
        )

        active = await feature.recall_patterns()
        assert active.data["count"] == 0

        everything = await feature.recall_patterns(include_superseded=True)
        assert everything.data["count"] == 1
        assert everything.data["patterns"][0]["status"] == "superseded"

    @pytest.mark.asyncio
    async def test_resolved_blockers_are_excluded_by_default(self, tmp_path):
        feature = await _feature(tmp_path)
        added = await feature.strategy_add_blocker(issue="#7", title="waiting on infra")
        await feature.strategy_resolve_blocker(
            issue=added.data["blocker_id"], resolution="infra landed"
        )

        active = await feature.recall_blockers()
        assert active.data["count"] == 0

        everything = await feature.recall_blockers(include_resolved=True)
        assert everything.data["count"] == 1
        assert everything.data["blockers"][0]["status"] == "resolved"

    @pytest.mark.asyncio
    async def test_a_query_failure_is_not_reported_as_zero(self, tmp_path):
        """The defect the whole ticket was filed on, in its smallest form."""
        graph = _FakeGraph(fail_query_on=PATTERN_NODE_TYPE)
        feature = await _feature(tmp_path, graph=graph)

        result = await feature.recall_patterns()

        assert result.status.value == "error"
        assert "could not read the strategy index" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_an_empty_index_over_a_populated_ledger_is_not_a_clean_zero(
        self, tmp_path
    ):
        """recall_decisions returned a truthful 0 while YAML held the real ones.

        That is the shape #2851 was filed on. Here the same divergence is
        reported instead of being rendered as an answer.
        """
        feature = await _feature(tmp_path)
        await feature.strategy_add_pattern(pattern="canonical but unindexed")
        # Lose the index without touching the canonical file — a database
        # restore from before the projection, or a projection that never ran.
        feature.agent.storage.graph.nodes.clear()

        result = await feature.recall_patterns()

        assert result.status.value == "partial"
        assert result.data["count"] == 0
        assert result.data["canonical_expected"] == 1
        assert result.data["missing_count"] == 1
        assert result.data["index_stale"] is True
        assert "stale or was never built" in (result.error or "")

    @pytest.mark.asyncio
    async def test_a_partial_index_is_not_a_clean_answer(self, tmp_path):
        """#3064: non-emptiness was taken as proof the index was complete.

        One projected row standing in for two canonical ones returned a short
        list with an ok status -- a check answering "is there anything?" while
        the caller asked "is this all of it?".
        """
        feature = await _feature(tmp_path)
        await feature.strategy_add_pattern(pattern="the first observation")
        await feature.strategy_add_pattern(pattern="the second observation")
        graph = feature.agent.storage.graph
        # Drop exactly one projected node. The ledger still holds both.
        victim = next(
            node_id
            for node_id, node in graph.nodes.items()
            if node.node_type == PATTERN_NODE_TYPE
        )
        del graph.nodes[victim]

        result = await feature.recall_patterns()

        assert result.data["count"] == 1
        assert result.status.value == "partial", (
            "a list missing a canonical row is not a clean answer"
        )
        assert result.data["index_stale"] is True
        assert result.data["missing_count"] == 1
        assert result.data["canonical_expected"] == 2

    @pytest.mark.asyncio
    async def test_membership_not_count_decides_staleness(self, tmp_path):
        """An index of the right size and the wrong contents is still stale.

        Comparing counts would pass here: one node in, one node out.
        """
        feature = await _feature(tmp_path)
        await feature.strategy_add_pattern(pattern="the canonical observation")
        graph = feature.agent.storage.graph
        victim, node = next(
            (nid, n)
            for nid, n in graph.nodes.items()
            if n.node_type == PATTERN_NODE_TYPE
        )
        del graph.nodes[victim]
        # A node of the same type and agent, but not a row the ledger holds.
        node.node_id = f"{PATTERN_NODE_TYPE}:{AGENT}:pat_stranger"
        node.properties = {**node.properties, "row_id": "pat_stranger"}
        graph.nodes[node.node_id] = node

        result = await feature.recall_patterns()

        assert result.data["count"] == 1
        assert result.status.value == "partial"
        assert result.data["missing_count"] == 1

    @pytest.mark.asyncio
    async def test_a_retired_row_still_answering_as_current_is_a_divergence(
        self, tmp_path
    ):
        """One-directional membership certified retired guidance as current.

        Projection is best-effort, so a write that fails after a supersession
        leaves the old node marked active while the canonical active set no
        longer holds it. Nothing is *missing*, so a check that only asks
        "did I get everything?" returns ok with completeness_checked True --
        confidently wrong rather than merely short.
        """
        feature = await _feature(tmp_path)
        added = await feature.strategy_add_pattern(pattern="old guidance")
        graph = feature.agent.storage.graph
        # The projection of the supersession fails. The canonical file records
        # it; the index still answers with the active row.
        graph._fail_on = PATTERN_NODE_TYPE
        await feature.strategy_supersede_pattern(
            pattern_id=added.data["pattern_id"], reason="retired"
        )

        result = await feature.recall_patterns()

        assert result.data["count"] == 1, "the stale node is still returned"
        assert result.data["canonical_expected"] == 0
        assert result.status.value == "partial", (
            "an index answering with a row the ledger retired is not ok"
        )
        assert result.data["index_stale"] is True
        assert result.data["misfiled_count"] == 1, (
            "the ledger retired it and the index still marks it current"
        )

    @pytest.mark.asyncio
    async def test_a_graph_retired_row_is_not_reported_as_missing(self, tmp_path):
        """The projection preserves graph-owned supersession by design.

        A node carrying superseded_by while the ledger is silent is retired on
        purpose, and reprojection keeps it that way -- so calling its absence
        from the active page a divergence would report a permanent staleness
        no restart could ever clear. Only the reverse (current in the index,
        retired in the ledger) is a divergence.
        """
        feature = await _feature(tmp_path)
        first = await feature.strategy_add_pattern(pattern="graph retires me")
        await feature.strategy_add_pattern(pattern="still active")
        graph = feature.agent.storage.graph
        node = next(
            n
            for n in graph.nodes.values()
            if n.node_type == PATTERN_NODE_TYPE
            and n.properties["row_id"] == first.data["pattern_id"]
        )
        node.properties["superseded_by"] = "pat_replacement"
        await feature._reindex_ledger()
        assert node.properties["status"] == "superseded"

        result = await feature.recall_patterns()

        assert result.data["count"] == 1
        assert result.status.value == "ok", (
            "a row the index retired on its own is present, not missing"
        )
        assert result.data["completeness_checked"] is True

    @pytest.mark.asyncio
    async def test_a_divergence_behind_the_page_is_still_reported(self, tmp_path):
        """The check reads the whole scoped projection, not the caller's page.

        An orphan that sorts behind every canonical row is invisible to a
        LIMIT and present in the database, so a page-derived check certifies a
        clean index while a larger limit returns deleted guidance.
        """
        feature = await _feature(tmp_path)
        for n in range(3):
            await feature.strategy_add_pattern(pattern=f"observation {n}")
        graph = feature.agent.storage.graph
        node = next(
            n for n in graph.nodes.values() if n.node_type == PATTERN_NODE_TYPE
        )
        node.properties = {**node.properties, "row_id": "pat_stranger"}

        result = await feature.recall_patterns(limit=2)

        assert result.data["count"] == 2, "the page itself is still bounded"
        assert result.status.value == "partial"
        assert result.data["index_stale"] is True
        assert result.data["completeness_checked"] is True, (
            "the limit no longer defeats the check"
        )
        assert result.data["orphaned_count"] == 1
        assert result.data["missing_count"] == 1

    @pytest.mark.asyncio
    async def test_a_stale_retired_recall_names_a_fallback_that_can_answer(
        self, tmp_path
    ):
        """strategy_search excludes retired rows unless told otherwise.

        Recommending it bare for a retired-mode recall promises a fallback
        that structurally cannot return the rows the error is about.
        """
        feature = await _feature(tmp_path)
        added = await feature.strategy_add_pattern(pattern="a retired observation")
        await feature.strategy_supersede_pattern(
            pattern_id=added.data["pattern_id"], reason="no longer holds"
        )
        feature.agent.storage.graph.nodes.clear()

        result = await feature.recall_patterns(include_superseded=True)

        assert result.status.value == "partial"
        assert "include_retired=True" in (result.error or "")

        # And the named path actually returns it.
        found = await feature.strategy_search(
            query="retired observation", include_retired=True
        )
        assert found.data["count"] == 1
        assert (await feature.strategy_search(query="retired observation")).data[
            "count"
        ] == 0

    @pytest.mark.asyncio
    async def test_superseded_recall_is_measured_against_superseded_rows(
        self, tmp_path
    ):
        """#3064: include_superseded=True still compared against ACTIVE rows.

        A ledger holding only retired rows against an empty index therefore
        returned a clean zero -- the baseline answered a different question
        than the query did.
        """
        feature = await _feature(tmp_path)
        await feature.strategy_add_pattern(pattern="a since-retired observation")
        for row in feature._ledger.patterns:
            row["superseded_at"] = "2026-01-01"
            row["superseded_reason"] = "no longer holds"
        feature._ledger.save()
        feature.agent.storage.graph.nodes.clear()

        active = await feature.recall_patterns()
        assert active.data["canonical_expected"] == 0, (
            "no ACTIVE rows are expected, so the empty page is the right answer"
        )
        assert active.status.value == "partial", (
            "but membership is status-agnostic: the index is missing a row the "
            "ledger holds, and that is true whichever page was asked for"
        )
        assert active.data["missing_count"] == 1

        everything = await feature.recall_patterns(include_superseded=True)

        assert everything.data["count"] == 0
        assert everything.status.value == "partial", (
            "the ledger holds a superseded row the index does not"
        )
        assert everything.data["canonical_expected"] == 1
        assert everything.data["index_stale"] is True

    @pytest.mark.asyncio
    async def test_a_bounded_page_over_a_healthy_index_is_clean(self, tmp_path):
        """The limit bounds the answer without weakening the claim about it.

        Reading membership from the whole scoped projection means a short page
        is just a short page -- neither a false alarm nor an unrun check.
        """
        feature = await _feature(tmp_path)
        for n in range(4):
            await feature.strategy_add_pattern(pattern=f"observation {n}")

        result = await feature.recall_patterns(limit=2)

        assert result.status.value == "ok"
        assert result.data["count"] == 2
        assert result.data["completeness_checked"] is True
        assert "index_stale" not in result.data

    @pytest.mark.asyncio
    async def test_a_blocker_reopened_by_hand_is_not_a_clean_zero(self, tmp_path):
        """Whether the INDEX may retire a row is a property of the section.

        A pattern node may carry a graph-owned supersession the ledger is
        silent about. A blocker's status is a pure function of the ledger's
        resolved_at, so the same shape there is a projection that has not
        landed -- and an operator reopening a blocker by editing YAML would
        otherwise get a certified-clean empty list over an active blocker.
        """
        feature = await _feature(tmp_path)
        await feature.strategy_add_blocker(
            issue="owner/repo#42", title="CI runner is wedged", severity="high"
        )
        row_id = feature._ledger.data[BLOCKERS_KEY][0]["id"]
        await feature.strategy_resolve_blocker(issue=row_id, resolution="fixed")
        # Reopened by hand, and the reprojection cannot land.
        feature.agent.storage.graph._fail_on = BLOCKER_NODE_TYPE
        for row in feature._ledger.data[BLOCKERS_KEY]:
            row.pop("resolved_at", None)
            row.pop("resolution", None)
        feature._ledger.save()

        result = await feature.recall_blockers()

        assert result.data["count"] == 0
        assert result.data["canonical_expected"] == 1
        assert result.status.value == "partial", (
            "an empty list over an active blocker is the original bug shape"
        )
        assert result.data["misfiled_count"] == 1

    @pytest.mark.asyncio
    async def test_two_canonical_rows_on_one_id_are_reported(self, tmp_path):
        """A set of ids hides multiplicity; the ledger is hand-editable.

        Both rows project to the same node, the second overwrites the first,
        and one canonical row is unreachable however healthy the projection
        is. Reprojecting cannot fix it, so the message must not tell the
        caller to restart and expect a different result.
        """
        feature = await _feature(tmp_path)
        await feature.strategy_add_pattern(pattern="the first")
        rows = feature._ledger.data[PATTERNS_KEY]
        duplicate = dict(rows[0])
        duplicate["pattern"] = "a genuinely different observation"
        rows.append(duplicate)
        feature._ledger.save()
        await feature._reindex_ledger()

        result = await feature.recall_patterns()

        assert result.data["canonical_expected"] == 2, (
            "two rows are two rows, whatever ids they carry"
        )
        assert result.data["count"] == 1
        assert result.status.value == "partial"
        assert result.data["colliding_count"] == 1
        assert "duplicate ids" in (result.error or "")

    @pytest.mark.asyncio
    async def test_a_saturated_membership_read_is_not_a_complete_one(
        self, tmp_path, monkeypatch
    ):
        """A capped read cannot be told apart from a whole one.

        query_nodes_by_type_and_property clamps its limit, so a projection
        larger than the cap comes back looking exactly like a small complete
        one -- and every row past the cap would read as missing.
        """
        from kestrel_sovereign.features.strategic_memory import ledger_index

        feature = await _feature(tmp_path)
        for n in range(2):
            await feature.strategy_add_pattern(pattern=f"observation {n}")
        monkeypatch.setattr(ledger_index, "MEMBERSHIP_READ_CAP", 1)

        result = await feature.recall_patterns()

        assert result.status.value == "ok", (
            "the rows are still a real answer; only the check could not run"
        )
        assert result.data["completeness_checked"] is False
        assert result.data["completeness_unchecked_reason"] == (
            "index_exceeds_membership_cap"
        )
        assert "index_stale" not in result.data

    @pytest.mark.asyncio
    async def test_a_hand_written_node_is_not_this_projections_orphan(
        self, tmp_path
    ):
        """Membership describes what THIS projection wrote, nothing else.

        recall_nodes already filters on source so a hand-written node of the
        same type stays out of an answer claiming to describe the ledger. The
        membership read has to apply the same boundary, or someone else's node
        becomes a permanent orphan this feature reports and cannot fix.
        """
        from kestrel_sovereign.storage.async_graph_store import GraphNode

        feature = await _feature(tmp_path)
        await feature.strategy_add_pattern(pattern="a real observation")
        graph = feature.agent.storage.graph
        node_id = f"{PATTERN_NODE_TYPE}:{AGENT}:hand_written"
        graph.nodes[node_id] = GraphNode(
            node_id=node_id,
            node_type=PATTERN_NODE_TYPE,
            label="not ours",
            properties={
                "agent_id": AGENT,
                "row_id": "hand_written",
                "status": "active",
                "source": "some_other_writer",
            },
        )

        result = await feature.recall_patterns()

        assert result.status.value == "ok"
        assert result.data["count"] == 1, "and it is not in the answer either"
        assert "orphaned_count" not in result.data

    @pytest.mark.asyncio
    async def test_an_unreadable_membership_query_names_the_unrun_check(
        self, tmp_path
    ):
        """A failed membership read is not agreement, and not a failed recall.

        The rows already returned are a real answer; only the check could not
        run, and it says which.
        """
        feature = await _feature(tmp_path)
        await feature.strategy_add_pattern(pattern="an indexed observation")
        graph = feature.agent.storage.graph
        original = graph.query_nodes_by_type_and_property

        async def _refuse_membership(node_type, filters=None, **kwargs):
            # Discriminate on the cap, not on the absence of a status filter:
            # include_superseded=True also omits status, so that reading would
            # make the double refuse the page read as well.
            if kwargs.get("limit") == MEMBERSHIP_READ_CAP:
                raise RuntimeError("graph query refused")
            return await original(node_type, filters=filters, **kwargs)

        graph.query_nodes_by_type_and_property = _refuse_membership

        result = await feature.recall_patterns()

        assert result.status.value == "ok"
        assert result.data["count"] == 1
        assert result.data["completeness_checked"] is False
        assert result.data["completeness_unchecked_reason"] == (
            "index_membership_unavailable"
        )

    @pytest.mark.asyncio
    async def test_an_unreadable_ledger_names_the_unrun_check(self, tmp_path):
        """No trustworthy baseline means the check cannot run -- and says so."""
        feature = await _feature(tmp_path)
        await feature.strategy_add_pattern(pattern="an indexed observation")
        # The real path: the index is already projected, then the canonical
        # file is corrupted underneath it and reloaded. Setting the flag by
        # hand would test the flag.
        (tmp_path / LEDGER_FILENAME).write_text(
            "patterns_learned: [unclosed\n", encoding="utf-8"
        )
        feature._ledger.load()
        assert not feature._ledger.readable

        result = await feature.recall_patterns()

        assert result.status.value == "ok"
        assert result.data["count"] == 1
        assert result.data["completeness_checked"] is False
        assert result.data["completeness_unchecked_reason"] == "ledger_unreadable"

    @pytest.mark.asyncio
    async def test_a_text_less_row_is_not_reported_missing_forever(self, tmp_path):
        """The projection skips rows with no text; the check must skip them too.

        Otherwise every recall reports a permanent staleness no reprojection
        can clear.
        """
        feature = await _feature(tmp_path)
        await feature.strategy_add_pattern(pattern="a real observation")
        # ``patterns`` is a property that builds a fresh list, so appending to
        # it changes nothing. Write through ``data``, which is the ledger.
        feature._ledger.data[PATTERNS_KEY].append(
            {"id": "pat_blank", "pattern": "   "}
        )
        feature._ledger.save()
        await feature._reindex_ledger()

        result = await feature.recall_patterns()

        assert result.status.value == "ok"
        assert result.data["count"] == 1
        assert result.data["completeness_checked"] is True

    @pytest.mark.asyncio
    async def test_a_genuinely_empty_ledger_recalls_cleanly(self, tmp_path):
        """The divergence check must not make an empty agent look broken."""
        feature = await _feature(tmp_path)

        result = await feature.recall_patterns()

        assert result.status.value == "ok"
        assert result.data["count"] == 0
        assert "index_stale" not in result.data

    @pytest.mark.asyncio
    async def test_recall_refuses_without_an_agent_identity(self, tmp_path):
        """Scoping is the tenancy boundary; unscoped is a refusal, not a list."""
        feature = await _feature(tmp_path)
        for attribute in ("agent_id", "did", "id"):
            setattr(feature.agent, attribute, None)

        result = await feature.recall_blockers()

        assert result.status.value == "error"
        assert "agent_id" in (result.error or "")

    @pytest.mark.asyncio
    async def test_recall_does_not_return_a_foreign_agents_rows(self, tmp_path):
        from kestrel_sovereign.storage.async_graph_store import GraphNode

        feature = await _feature(tmp_path)
        await feature.strategy_add_pattern(pattern="mine")
        graph = feature.agent.storage.graph
        graph.nodes["strategy_pattern:did:other:x"] = GraphNode(
            node_id="strategy_pattern:did:other:x",
            node_type=PATTERN_NODE_TYPE,
            label="theirs",
            properties={
                "agent_id": "did:test:someone-else",
                "text": "theirs",
                "status": "active",
                "source": "strategic_memory",
            },
        )

        result = await feature.recall_patterns()

        assert [r["text"] for r in result.data["patterns"]] == ["mine"]

    @pytest.mark.asyncio
    async def test_recall_ignores_a_node_this_projection_did_not_write(self, tmp_path):
        """A same-typed node from elsewhere is not part of the ledger's answer."""
        from kestrel_sovereign.storage.async_graph_store import GraphNode

        feature = await _feature(tmp_path)
        await feature.strategy_add_pattern(pattern="projected")
        graph = feature.agent.storage.graph
        graph.nodes["strategy_pattern:handwritten"] = GraphNode(
            node_id="strategy_pattern:handwritten",
            node_type=PATTERN_NODE_TYPE,
            label="hand-written",
            properties={
                "agent_id": AGENT,
                "text": "hand-written",
                "status": "active",
                "source": "somewhere_else",
            },
        )

        result = await feature.recall_patterns()

        assert [r["text"] for r in result.data["patterns"]] == ["projected"]

    @pytest.mark.asyncio
    async def test_limit_is_validated_not_silently_clamped(self, tmp_path):
        feature = await _feature(tmp_path)

        assert (await feature.recall_patterns(limit="ten")).status.value == "error"
        assert (await feature.recall_patterns(limit=0)).status.value == "error"
        assert (await feature.recall_patterns(limit=201)).status.value == "error"
        assert (
            await feature.recall_patterns(include_superseded="yes")
        ).status.value == "error"

    @pytest.mark.asyncio
    async def test_limit_bounds_the_returned_rows(self, tmp_path):
        feature = await _feature(tmp_path)
        for n in range(4):
            await feature.strategy_add_pattern(pattern=f"observation {n}")

        result = await feature.recall_patterns(limit=2)

        assert result.data["count"] == 2
        assert result.data["limit_requested"] == 2


class TestRecallAgainstRealGraphStore:
    """The consumer, proven against AsyncGraphStore on real SQLite.

    The double above mirrors the production query, but a double is only ever
    evidence about itself. These run the same projection and the same recall
    through the real store so the JSON-path filtering, ordering and scoping are
    the ones production actually applies.
    """

    @pytest.mark.asyncio
    async def test_projected_rows_are_recallable_from_a_real_store(self, tmp_path):
        from kestrel_sovereign.storage.async_database import AsyncDatabase
        from kestrel_sovereign.storage.async_graph_store import AsyncGraphStore

        database = await AsyncDatabase.sqlite(str(tmp_path / "graph.db"))
        try:
            graph = AsyncGraphStore(database)
            feature = await _feature(tmp_path, graph=graph)

            await feature.strategy_add_pattern(
                pattern="the double is not the store",
                implication="prove the consumer against production",
            )
            await feature.strategy_add_blocker(
                issue="#11", title="blocked on a real query", severity="critical"
            )

            patterns = await feature.recall_patterns()
            assert patterns.status.value == "ok"
            assert patterns.data["count"] == 1
            assert patterns.data["patterns"][0]["text"] == "the double is not the store"

            blockers = await feature.recall_blockers()
            assert blockers.status.value == "ok"
            assert blockers.data["count"] == 1
            assert blockers.data["blockers"][0]["issue"] == "#11"
            assert blockers.data["blockers"][0]["severity"] == "critical"
        finally:
            await database.close()

    @pytest.mark.asyncio
    async def test_supersession_reaches_the_real_index(self, tmp_path):
        """Retiring a row in YAML must remove it from the real query's answer."""
        from kestrel_sovereign.storage.async_database import AsyncDatabase
        from kestrel_sovereign.storage.async_graph_store import AsyncGraphStore

        database = await AsyncDatabase.sqlite(str(tmp_path / "graph.db"))
        try:
            graph = AsyncGraphStore(database)
            feature = await _feature(tmp_path, graph=graph)
            added = await feature.strategy_add_pattern(pattern="true until it wasn't")

            await feature.strategy_supersede_pattern(
                pattern_id=added.data["pattern_id"], reason="re-measured"
            )

            assert (await feature.recall_patterns()).data["count"] == 0
            everything = await feature.recall_patterns(include_superseded=True)
            assert everything.data["count"] == 1
            assert everything.data["patterns"][0]["status"] == "superseded"
        finally:
            await database.close()

    @pytest.mark.asyncio
    async def test_a_real_store_scopes_recall_to_this_agent(self, tmp_path):
        """The agent filter is pushed into SQL; prove it there, not in a double."""
        from kestrel_sovereign.storage.async_database import AsyncDatabase
        from kestrel_sovereign.storage.async_graph_store import (
            AsyncGraphStore,
            GraphNode,
        )

        database = await AsyncDatabase.sqlite(str(tmp_path / "graph.db"))
        try:
            graph = AsyncGraphStore(database)
            feature = await _feature(tmp_path, graph=graph)
            await feature.strategy_add_pattern(pattern="mine")

            await graph.add_node(GraphNode(
                node_id="strategy_pattern:did:test:stranger:z",
                node_type=PATTERN_NODE_TYPE,
                label="theirs",
                properties={
                    "agent_id": "did:test:stranger",
                    "text": "theirs",
                    "status": "active",
                    "source": "strategic_memory",
                },
            ))

            result = await feature.recall_patterns()
            assert [r["text"] for r in result.data["patterns"]] == ["mine"]
        finally:
            await database.close()


class TestUnreadableLedgerRefusesEveryToolThatTouchesIt:
    """Fail closed, and say which failure it is.

    An empty in-memory ledger answers "no patterns", "no blocker found with
    issue X", "no matches" — all of them truthful about the object in memory
    and all of them false about the agent's record. The reviewer's rule: every
    reader and mutator refuses before inspecting or modifying ``ledger.data``.
    """

    @staticmethod
    async def _broken(tmp_path):
        (tmp_path / LEDGER_FILENAME).write_text(
            "patterns_learned: [unclosed\n", encoding="utf-8"
        )
        feature = await _feature(tmp_path)
        assert not feature._ledger.readable
        return feature

    @pytest.mark.asyncio
    async def test_add_pattern_does_not_mutate_before_refusing(self, tmp_path):
        feature = await self._broken(tmp_path)

        result = await feature.strategy_add_pattern(pattern="would be phantom")

        assert result.status.value == "error"
        assert result.data["ledger_unavailable"] is True
        # The row must not be sitting in memory looking recorded.
        assert feature._ledger.data.get(PATTERNS_KEY, []) == []

    @pytest.mark.asyncio
    async def test_search_refuses_rather_than_reporting_no_matches(self, tmp_path):
        feature = await self._broken(tmp_path)

        result = await feature.strategy_search(query="anything")

        assert result.status.value == "error"
        assert result.data["ledger_unavailable"] is True

    @pytest.mark.asyncio
    async def test_supersede_refuses_rather_than_reporting_not_found(self, tmp_path):
        feature = await self._broken(tmp_path)

        result = await feature.strategy_supersede_pattern(pattern_id="pat_abc")

        assert result.status.value == "error"
        assert result.data["ledger_unavailable"] is True
        assert "no pattern found" not in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_resolve_refuses_rather_than_reporting_not_found(self, tmp_path):
        feature = await self._broken(tmp_path)

        result = await feature.strategy_resolve_blocker(issue="#42")

        assert result.status.value == "error"
        assert result.data["ledger_unavailable"] is True
        assert "no blocker found" not in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_reconcile_refuses_rather_than_reporting_a_clean_bill(self, tmp_path):
        feature = await self._broken(tmp_path)

        result = await feature.strategy_reconcile_blockers()

        assert result.status.value == "error"
        assert result.data["ledger_unavailable"] is True

    @pytest.mark.asyncio
    async def test_ledger_sections_of_the_view_refuse(self, tmp_path):
        feature = await self._broken(tmp_path)

        for section in ("patterns", "blockers"):
            result = await feature.strategy_view(section=section)
            assert result.status.value == "error", section
            assert result.data["ledger_unavailable"] is True, section

    @pytest.mark.asyncio
    async def test_the_standing_brief_still_renders_with_a_caveat(self, tmp_path):
        """A broken ledger must not take down a working STRATEGY.yaml.

        The mirror of the rule the loader already follows in the other
        direction. ``all`` renders what is genuine and names what is missing.
        """
        (tmp_path / LEDGER_FILENAME).write_text(
            "patterns_learned: [unclosed\n", encoding="utf-8"
        )
        feature = await _feature(
            tmp_path, strategy={"vision": "Ship the tortoise", "milestones": []}
        )
        assert not feature._ledger.readable

        result = await feature.strategy_view(section="all")

        assert result.status.value == "partial"
        assert "Ship the tortoise" in result.confirmation
        assert result.data["ledger_unavailable"] is True
        assert "Patterns and blockers are missing" in (result.error or "")

    @pytest.mark.asyncio
    async def test_vision_is_unaffected_by_a_broken_ledger(self, tmp_path):
        """A view that never held ledger content loses nothing, so says nothing.

        Caveating ``!strategy vision`` with "patterns and blockers are missing"
        would report a loss that view never had. Only ``all`` genuinely drops a
        section it would otherwise have rendered.
        """
        feature = await self._broken(tmp_path)
        feature._data["vision"] = "Still readable"

        result = await feature.strategy_view(section="vision")

        assert result.status.value == "ok"
        assert "Still readable" in result.confirmation
        assert "ledger_unavailable" not in (result.data or {})


class TestTheDefaultTemplateIsNotSharedBetweenAgents:
    """One agent's strategic record must not be born inside another's.

    ``_data = dict(self._DEFAULT_TEMPLATE)`` was a shallow copy, so the
    ``decisions`` / ``milestones`` / ``stakeholders`` lists were the SAME
    objects on every instance in the process. ``strategy_add_decision``
    appended into the class-level template, and the next agent created was
    born holding them — then ``_save()`` wrote them to its own STRATEGY.yaml.

    Found while adding the index consumers: a full-suite run projected four
    graph nodes where one was expected, because earlier tests had been
    appending into the shared list all along. The bug predates this ticket;
    it is fixed here because it lives in this file and is one line.
    """

    @staticmethod
    async def _fresh(tmp_path, name):
        agent = MagicMock()
        agent.agent_id = f"did:test:{name}"
        agent.agent_data_dir = str(tmp_path / name)
        (tmp_path / name).mkdir(parents=True, exist_ok=True)
        agent.storage = MagicMock()
        agent.storage.graph = _FakeGraph()
        feature = StrategicMemoryFeature(agent)
        await feature.initialize()
        return feature

    @pytest.mark.asyncio
    async def test_a_decision_does_not_leak_into_the_next_agent(self, tmp_path):
        first = await self._fresh(tmp_path, "agent_a")
        await first.strategy_add_decision(
            decision="Agent A private decision", rationale="its own reasons"
        )

        second = await self._fresh(tmp_path, "agent_b")

        assert second._data.get("decisions") == [], (
            "a second agent must not be born holding the first agent's decisions"
        )
        on_disk = _yaml.safe_load(
            (tmp_path / "agent_b" / "STRATEGY.yaml").read_text(encoding="utf-8")
        )
        assert on_disk.get("decisions") == [], (
            "and the leak must not have been written to its STRATEGY.yaml"
        )

    @pytest.mark.asyncio
    async def test_the_class_level_template_is_never_mutated(self, tmp_path):
        before = copy.deepcopy(StrategicMemoryFeature._DEFAULT_TEMPLATE)

        feature = await self._fresh(tmp_path, "agent_c")
        await feature.strategy_add_decision(decision="mutating?", rationale="r")
        feature._data.setdefault("milestones", []).append({"name": "M1"})
        feature._data.setdefault("stakeholders", []).append({"name": "S1"})

        assert StrategicMemoryFeature._DEFAULT_TEMPLATE == before

    @pytest.mark.asyncio
    async def test_each_agent_gets_its_own_container_objects(self, tmp_path):
        first = await self._fresh(tmp_path, "agent_d")
        second = await self._fresh(tmp_path, "agent_e")

        for key in ("milestones", "stakeholders", "decisions"):
            assert first._data[key] is not second._data[key], key
            assert (
                first._data[key] is not StrategicMemoryFeature._DEFAULT_TEMPLATE[key]
            ), key


class TestRecallLimitIsNotUnderReported:
    """A page of retired rows must not hide the active rows behind it.

    Filtering status in Python after a ``LIMIT``-ed query means asking for 25
    active patterns returns zero whenever the 25 most recent happen to be
    superseded — an honest-looking count with a hundred active rows just past
    the page boundary. The predicates are exact equalities, so they belong in
    SQL where the limit can mean what the caller asked for.
    """

    @staticmethod
    async def _with_retired_newest(tmp_path, graph):
        feature = await _feature(tmp_path, graph=graph)
        # Oldest first, so the superseded rows sort newest under created_at
        # DESC and would occupy the whole first page.
        added = []
        for n in range(3):
            r = await feature.strategy_add_pattern(pattern=f"still true {n}")
            added.append(r.data["pattern_id"])
        for n in range(3):
            r = await feature.strategy_add_pattern(pattern=f"no longer true {n}")
            await feature.strategy_supersede_pattern(
                pattern_id=r.data["pattern_id"], reason="re-measured"
            )
        # recorded_at is a date, so force a strict ordering that puts the
        # retired rows unambiguously newest.
        for row in feature._ledger.patterns:
            if row["pattern"].startswith("no longer"):
                row["recorded_at"] = "2099-01-01"
        feature._ledger.save()
        await feature._reindex_ledger()
        return feature

    @pytest.mark.asyncio
    async def test_active_rows_survive_a_page_full_of_retired_ones(self, tmp_path):
        feature = await self._with_retired_newest(tmp_path, _FakeGraph())

        result = await feature.recall_patterns(limit=3)

        assert result.data["count"] == 3, (
            "three active patterns exist; a page of superseded rows must not "
            "consume the limit"
        )
        assert all(
            r["text"].startswith("still true") for r in result.data["patterns"]
        )

    @pytest.mark.asyncio
    async def test_same_holds_against_a_real_graph_store(self, tmp_path):
        from kestrel_sovereign.storage.async_database import AsyncDatabase
        from kestrel_sovereign.storage.async_graph_store import AsyncGraphStore

        database = await AsyncDatabase.sqlite(str(tmp_path / "graph.db"))
        try:
            feature = await self._with_retired_newest(
                tmp_path, AsyncGraphStore(database)
            )

            result = await feature.recall_patterns(limit=3)

            assert result.data["count"] == 3
            assert all(
                r["text"].startswith("still true") for r in result.data["patterns"]
            )
        finally:
            await database.close()


class TestDataLossGuardsAreDefended:
    """The two P1 guards that reconciliation and persistence depend on.

    Both were correct in the tree and killed no mutant: removing the
    fail-closed check in ``save()``, and letting ``resolve_row_repo`` bind an
    unqualified reference to the first configured repository, each left the
    suite green. A guard nothing tests is one refactor from being deleted as
    dead weight, and both of these protect against silent data loss rather
    than a visible error.
    """

    def test_save_refuses_to_overwrite_a_ledger_it_could_not_parse(self, tmp_path):
        """An unreadable ledger must be left on disk untouched.

        ``load()`` leaves the in-memory sections empty when the parse fails,
        so writing them back is not an update — it deletes everything the
        parse could not reach.
        """
        from kestrel_sovereign.features.strategic_memory.ledger import StrategyLedger

        path = tmp_path / "STRATEGY_LEDGER.yaml"
        malformed = "version: 1\npatterns: [unclosed\n"
        path.write_text(malformed, encoding="utf-8")

        ledger = StrategyLedger(path)
        ledger.load()
        assert ledger.load_error, "precondition: the file must fail to parse"

        error = ledger.save()

        assert error is not None, "save() must refuse, not silently succeed"
        assert "Refusing to write" in error
        assert path.read_text(encoding="utf-8") == malformed, (
            "the malformed file must be byte-identical after a refused save"
        )

    def test_an_unqualified_issue_is_ambiguous_across_several_repositories(self):
        """`#42` is not an issue identifier; `owner/repo#42` is.

        Binding an unqualified reference to the first configured repository
        containing that number resolved a blocker whose issue was open in one
        project because a different project had closed its own issue 42.
        """
        from kestrel_sovereign.features.strategic_memory.blocker_reconcile import (
            AMBIGUOUS_REPO,
            resolve_row_repo,
        )

        repo, problem = resolve_row_repo(
            {"issue": "#42"}, ["owner/alpha", "owner/beta"]
        )

        assert repo is None, "must not guess a repository"
        assert problem == AMBIGUOUS_REPO

    def test_a_row_that_names_its_repository_is_never_ambiguous(self):
        """The guard must refuse guesses without refusing known answers."""
        from kestrel_sovereign.features.strategic_memory.blocker_reconcile import (
            resolve_row_repo,
        )

        declared, problem = resolve_row_repo(
            {"issue": "#42", "repo": "owner/alpha"}, ["owner/alpha", "owner/beta"]
        )
        assert (declared, problem) == ("owner/alpha", None)

        qualified, problem = resolve_row_repo(
            {"issue": "owner/beta#42"}, ["owner/alpha", "owner/beta"]
        )
        assert (qualified, problem) == ("owner/beta", None)

        single, problem = resolve_row_repo({"issue": "#42"}, ["owner/alpha"])
        assert (single, problem) == ("owner/alpha", None), (
            "one configured repository makes an unqualified reference unambiguous"
        )
