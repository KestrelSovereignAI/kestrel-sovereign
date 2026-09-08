"""Strategic-memory projections stamp ``created_at`` under the graph contract (#3255).

The ledger and decision file carry day-granular dates, and some rows carry no
date. The projections used to copy those values into ``properties.created_at``
as-is: a bare date sorts before every timestamp of its day, and an empty
string is either a Postgres purge cast error or, after #3227, a row outside
leak coverage for good. One rule at the projection seam now turns a date into
midnight UTC of that date, normalises a full timestamp, and leaves the key
absent when nothing usable is known.

The migration is the re-projection itself: both projections run at feature
initialize and after every mutation, and they replace ``created_at`` from the
ledger, so the reindex test here is the proof that existing bare-date and
empty-string rows are rewritten.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from kestrel_sovereign.features.strategic_memory.decision_index import (
    _entry_properties,
    _label_for,
    project_decisions,
    strategy_decision_node_id,
)
from kestrel_sovereign.features.strategic_memory.ledger import BLOCKERS_KEY, PATTERNS_KEY
from kestrel_sovereign.features.strategic_memory.ledger_index import (
    BLOCKER_NODE_TYPE,
    PATTERN_NODE_TYPE,
    _blocker_properties,
    _label,
    _pattern_properties,
    ledger_node_id,
    project_ledger,
)
from kestrel_sovereign.features.strategic_memory.timestamps import (
    contract_created_at,
    stamp_created_at,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.async_graph_store import AsyncGraphStore, GraphNode

AGENT = "did:test:strategy-stamps"
MIDNIGHT = "2026-07-01T00:00:00+00:00"


@pytest.mark.parametrize(
    "value, expected",
    [
        ("2026-07-01", MIDNIGHT),
        (date(2026, 7, 1), MIDNIGHT),
        ("2026-07-01T09:30:00+00:00", "2026-07-01T09:30:00+00:00"),
        ("2026-07-01T09:30:00Z", "2026-07-01T09:30:00+00:00"),
        ("2026-07-01T09:30:00", "2026-07-01T09:30:00+00:00"),  # naive is UTC
        ("2026-07-01T11:30:00+02:00", "2026-07-01T09:30:00+00:00"),
        (datetime(2026, 7, 1, 9, 30, tzinfo=timezone(timedelta(hours=2))), "2026-07-01T07:30:00+00:00"),
        (datetime(2026, 7, 1, 9, 30), "2026-07-01T09:30:00+00:00"),
        (" 2026-07-01 ", MIDNIGHT),
        ("", None),
        ("   ", None),
        (None, None),
        (7, None),
        ("not a date", None),
        ("2026-02-30", None),
    ],
)
def test_contract_created_at(value, expected, new_york_clock):
    assert contract_created_at(value) == expected


def test_stamp_leaves_the_key_absent_when_nothing_is_known():
    props = {"agent_id": AGENT, "created_at": "stale"}
    assert "created_at" not in stamp_created_at(props, "")
    assert "created_at" not in stamp_created_at({"agent_id": AGENT}, None)
    assert stamp_created_at({"agent_id": AGENT}, "2026-07-01")["created_at"] == MIDNIGHT


def test_every_writer_stamps_the_contract_shape():
    decision = _entry_properties(AGENT, {"decision": "adopt leases", "date": "2026-07-01"})
    pattern = _pattern_properties(AGENT, {"id": "pat_1", "pattern": "x", "recorded_at": "2026-07-01"})
    blocker = _blocker_properties(AGENT, {"id": "blk_1", "title": "t", "blocked_since": "2026-07-01"})
    assert decision["created_at"] == pattern["created_at"] == blocker["created_at"] == MIDNIGHT


def test_every_writer_leaves_the_key_absent_for_a_dateless_row():
    assert "created_at" not in _entry_properties(AGENT, {"decision": "adopt leases"})
    assert "created_at" not in _pattern_properties(AGENT, {"id": "pat_1", "pattern": "x", "recorded_at": ""})
    assert "created_at" not in _blocker_properties(AGENT, {"id": "blk_1", "title": "t"})


@pytest.mark.asyncio
async def test_reindex_rewrites_bare_date_and_empty_stamps(tmp_path):
    """The migration is the re-projection: nodes written by the OLD writers
    (bare date, empty string) are replaced on the next reindex."""
    db = await AsyncDatabase.sqlite(str(tmp_path / "kestrel.db"))
    try:
        graph = AsyncGraphStore(db, agent_id=AGENT)
        pattern_row = {"id": "pat_1", "pattern": "x", "recorded_at": "2026-07-01"}
        empty_row = {"id": "pat_2", "pattern": "y"}
        blocker_row = {"id": "blk_1", "issue": "#1", "title": "t", "blocked_since": "2026-07-02"}
        relabeled_row = {"id": "blk_2", "issue": "#2", "title": "renamed", "blocked_since": "2026-07-04"}
        entry = {"decision": "adopt leases", "date": "2026-07-03"}
        # What the old projection left behind, verbatim in shape, under the
        # projection's own node ids (a row that carries an ``id`` keeps it;
        # the minting helpers apply only to rows without one) and under the
        # labels the old projection wrote from the same text. On the live
        # agents the label matches, so the rewrite goes through the
        # compare-and-swap door; one row carries a stale label so the
        # add_node door is exercised too.
        pat_1 = ledger_node_id(PATTERN_NODE_TYPE, AGENT, pattern_row["id"])
        pat_2 = ledger_node_id(PATTERN_NODE_TYPE, AGENT, empty_row["id"])
        blk_1 = ledger_node_id(BLOCKER_NODE_TYPE, AGENT, blocker_row["id"])
        blk_2 = ledger_node_id(BLOCKER_NODE_TYPE, AGENT, relabeled_row["id"])
        dec_1 = strategy_decision_node_id(AGENT, entry)
        old = {
            pat_1: (PATTERN_NODE_TYPE, _label(pattern_row["pattern"]), "2026-07-01"),
            pat_2: (PATTERN_NODE_TYPE, _label(empty_row["pattern"]), ""),
            blk_1: (BLOCKER_NODE_TYPE, _label(blocker_row["title"]), "2026-07-02"),
            blk_2: (BLOCKER_NODE_TYPE, "old title", "2026-07-04"),
            dec_1: ("decision", _label_for(entry), "2026-07-03"),
        }
        for node_id, (node_type, label, stamp) in old.items():
            await graph.add_node(GraphNode(
                node_id=node_id, node_type=node_type, label=label,
                properties={"agent_id": AGENT, "created_at": stamp, "source": "strategic_memory"},
            ))

        report = await project_ledger(
            graph, AGENT,
            {PATTERNS_KEY: [pattern_row, empty_row], BLOCKERS_KEY: [blocker_row, relabeled_row]},
        )
        assert report["failed"] == 0, report
        await project_decisions(graph, AGENT, [entry])

        after = {}
        for node_id in old:
            node = await graph.get_node(node_id)
            assert node is not None, node_id
            after[node_id] = node.properties
        assert after[pat_1]["created_at"] == "2026-07-01T00:00:00+00:00"
        assert "created_at" not in after[pat_2]
        assert after[blk_1]["created_at"] == "2026-07-02T00:00:00+00:00"
        assert after[blk_2]["created_at"] == "2026-07-04T00:00:00+00:00"
        assert after[dec_1]["created_at"] == "2026-07-03T00:00:00+00:00"
    finally:
        await db.close()
