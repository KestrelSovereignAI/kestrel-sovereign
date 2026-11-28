"""The generic paged recall read of a registered node type (#4155, rv-4135-1).

Every test here runs against a real SQLite agent store behind the real
``PrivacyEnforcingStorage``, with a real assertion store, because the
properties under test (ownership scope, keyset paging, canonical assertion
status, message trash state) live in those layers and not in a double.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from kestrel_sdk.tools.result import ToolResultStatus
from kestrel_sovereign.features import registered_recall
from kestrel_sovereign.features.memory.feature import MemoryFeature
from kestrel_sovereign.features.registered_recall import (
    ALIAS_REMOVAL_RELEASE,
    FIXED_RECALL_ALIASES,
    MAX_PAGE_SIZE,
    WITHHELD_REASONS,
    RecallRefused,
    list_recall_types,
    read_page,
)
from kestrel_sovereign.features.save.feature import SaveFeature
from kestrel_sovereign.features.strategic_memory.feature import (
    StrategicMemoryFeature,
)
from kestrel_sovereign.knowledge import AssertionQuery, AssertionStatus
from kestrel_sovereign.privacy import PrivacyMode
from kestrel_sovereign.storage.async_graph_store import AsyncGraphStore, GraphNode
from kestrel_sovereign.storage.node_type_registry import (
    derive_governance,
    list_node_types,
)
from tests.utils.recall_storage import governed_storage

AGENT = "did:test:recall-agent"
OTHER = "did:test:other-agent"

SERVED = {
    "action_item",
    "decision",
    "strategy_decision",
    "strategy_pattern",
    "strategy_blocker",
    "message",
}


@pytest.fixture
async def stores(tmp_path):
    async with governed_storage(str(tmp_path / "kestrel.db"), AGENT) as pair:
        yield pair


def _agent(storage, tmp_path):
    agent = MagicMock()
    agent.agent_id = AGENT
    agent.did = AGENT
    agent.agent_data_dir = str(tmp_path)
    agent.storage = storage
    agent.features = {}
    agent.get_feature = lambda name: agent.features.get(name)
    return agent


async def _save(agent) -> SaveFeature:
    feature = SaveFeature(agent)
    await feature.initialize()
    return feature


def _memory(agent) -> MemoryFeature:
    feature = MemoryFeature.__new__(MemoryFeature)
    feature.agent = agent
    feature.agent_id = AGENT
    return feature


async def _strategic(agent) -> StrategicMemoryFeature:
    feature = StrategicMemoryFeature(agent)
    await feature.initialize()
    agent.features["StrategicMemoryFeature"] = feature
    return feature


def _decision(index, *, created_at, agent=AGENT, **extra):
    return GraphNode(
        node_id=f"decision:{agent}:{index:016x}",
        node_type="decision",
        label=f"decision {index}",
        properties={
            "text": f"decision {index}",
            "agent_id": agent,
            "confidence": 0.8,
            "source_message_id": str(index),
            **({"created_at": created_at} if created_at is not None else {}),
            **extra,
        },
    )


async def _recall(save, **kwargs):
    result = await save.recall(**kwargs)
    assert result.status is ToolResultStatus.OK, result
    return result.data


async def _all_pages(save, node_type, limit, **kwargs):
    seen, cursor, pages = [], "", 0
    while True:
        data = await _recall(save, node_type=node_type, limit=limit, cursor=cursor, **kwargs)
        seen.extend(row["node_id"] for row in data["results"])
        pages += 1
        cursor = data["next_cursor"] or ""
        if not cursor:
            return seen, pages


# ---------------------------------------------------------------------------
# recall_types
# ---------------------------------------------------------------------------


async def test_recall_types_lists_registry(stores, tmp_path):
    storage, raw = stores
    save = await _save(_agent(storage, tmp_path))

    result = await save.recall_types()

    assert result.status is ToolResultStatus.OK, result
    listed = {row["node_type"]: row for row in result.data["types"]}
    registry = await list_node_types(raw.db)
    assert sorted(listed) == [entry.node_type for entry in registry]
    for entry in registry:
        row = listed[entry.node_type]
        for key, value in entry.to_dict().items():
            assert row[key] == value, (entry.node_type, key)
        governance = derive_governance(entry.node_type)
        assert row["governed"] is governance.governed
        assert row["governed_reason"] == governance.reason
        assert row["origin"] == ("projection" if entry.is_projection else "original")
        if row["generic_read"]:
            assert row["withheld_reason"] is None
        else:
            assert row["withheld_reason"], entry.node_type
    assert set(result.data["served"]) == SERVED
    assert listed["strategy_pattern"]["governed"] is True
    assert listed["strategy_blocker"]["governed"] is True
    assert listed["fact"]["governed"] is True
    assert listed["fact"]["generic_read"] is False
    assert "#4159" in listed["fact"]["withheld_reason"]
    assert listed["decision"]["canonical_home"] == "graph"
    assert listed["decision"]["is_projection"] is False
    assert listed["strategy_decision"]["canonical_home"] == "file"
    assert listed["strategy_decision"]["canonical_location"] == "STRATEGY.yaml"
    assert listed["strategy_decision"]["is_projection"] is True
    assert "API keys" in listed["feature_config"]["withheld_reason"]


async def test_every_served_governed_type_has_a_canonical_status_mapping(stores):
    _, raw = stores
    for described in await list_recall_types(raw.db):
        if described.served and described.governance.governed:
            module = __import__(described.governance.producer.module, fromlist=["x"])
            assert callable(getattr(module, "graph_node_assertion_ids", None))


async def test_an_unregistered_or_withheld_type_is_refused(stores, tmp_path):
    storage, _ = stores
    save = await _save(_agent(storage, tmp_path))

    unregistered = await save.recall(node_type="no_such_type")
    withheld = await save.recall(node_type="feature_config")

    assert unregistered.status is ToolResultStatus.ERROR
    assert unregistered.data["refused"] == "unregistered"
    assert withheld.status is ToolResultStatus.ERROR
    assert withheld.data["refused"] == "withheld"
    assert "API keys" in withheld.error


# ---------------------------------------------------------------------------
# Paging, cap, scope, privacy
# ---------------------------------------------------------------------------


async def test_generic_read_is_paged_and_capped(stores, tmp_path):
    storage, raw = stores
    expected = []
    # Inserted out of node-id order, so storage order cannot stand in for the
    # node_id tie-break the keyset needs.
    for index in sorted(range(MAX_PAGE_SIZE + 37), key=lambda i: (i * 7919) % 241):
        # Ties on created_at, and unstamped rows, are where a keyset that is
        # not total would repeat or skip a row.
        created_at = None if index % 11 == 0 else f"2026-07-{1 + index % 9:02d}T00:00:00+00:00"
        node = _decision(index, created_at=created_at)
        await raw.graph.add_node(node)
        expected.append(node.node_id)
    save = await _save(_agent(storage, tmp_path))

    clamped = await _recall(save, node_type="decision", limit=MAX_PAGE_SIZE * 10)
    assert clamped["limit"] == MAX_PAGE_SIZE
    assert clamped["limit_clamped"] is True
    assert clamped["count"] == MAX_PAGE_SIZE
    assert clamped["next_cursor"]

    seen, pages = await _all_pages(save, "decision", 7)

    assert len(seen) == len(set(seen)), "a row came back twice"
    assert sorted(seen) == sorted(expected), "a row was skipped"
    assert pages == -(-len(expected) // 7)
    # Newest first, unstamped rows last.
    stamps = [
        (await raw.graph.get_node(node_id)).properties.get("created_at") for node_id in seen
    ]
    stamped = [s for s in stamps if s is not None]
    assert stamped == sorted(stamped, reverse=True)
    assert all(s is None for s in stamps[len(stamped):])


async def test_a_cursor_for_another_type_is_refused(stores, tmp_path):
    storage, raw = stores
    for index in range(3):
        await raw.graph.add_node(_decision(index, created_at="2026-07-01T00:00:00+00:00"))
    save = await _save(_agent(storage, tmp_path))
    first = await _recall(save, node_type="decision", limit=1)

    other = await save.recall(node_type="action_item", cursor=first["next_cursor"])
    garbage = await save.recall(node_type="decision", cursor="not-a-cursor")

    assert other.data["refused"] == "bad_cursor"
    assert garbage.data["refused"] == "bad_cursor"


async def test_generic_read_agent_scoped(stores, tmp_path):
    storage, raw = stores
    mine = _decision(1, created_at="2026-07-01T00:00:00+00:00")
    await raw.graph.add_node(mine)
    other_graph = AsyncGraphStore(raw.db, agent_id=OTHER)
    theirs = _decision(2, created_at="2026-07-02T00:00:00+00:00", agent=OTHER)
    await other_graph.add_node(theirs)
    save = await _save(_agent(storage, tmp_path))

    data = await _recall(save, node_type="decision", limit=50)
    lookup = await save.recall_get(theirs.node_id, node_type="decision")

    assert [row["node_id"] for row in data["results"]] == [mine.node_id]
    assert lookup.status is ToolResultStatus.ERROR
    # And the aliases see exactly the same.
    decisions = await _memory(_agent(storage, tmp_path)).recall_decisions()
    assert [d["id"] for d in decisions.data["decisions"]] == [mine.node_id]


async def test_a_row_owned_here_but_naming_another_agent_is_not_returned(stores, tmp_path):
    """The SQL agent_id filter holds even where the ownership ledger would not."""
    storage, raw = stores
    mine = _decision(1, created_at="2026-07-01T00:00:00+00:00")
    await raw.graph.add_node(mine)
    theirs = _decision(2, created_at="2026-07-02T00:00:00+00:00", agent=OTHER)
    await AsyncGraphStore(raw.db, agent_id=OTHER).add_node(theirs)
    await raw.db.execute(
        "INSERT INTO graph_node_owners (node_id, agent_id) VALUES (?, ?)",
        (theirs.node_id, AGENT),
    )
    assert await raw.graph.get_node(theirs.node_id) is not None, "owned here too"
    save = await _save(_agent(storage, tmp_path))

    data = await _recall(save, node_type="decision", limit=50)
    lookup = await save.recall_get(theirs.node_id, node_type="decision")

    assert [row["node_id"] for row in data["results"]] == [mine.node_id]
    assert lookup.data["missing"] == "foreign"


async def test_an_unbound_graph_store_is_refused(stores, tmp_path):
    storage, raw = stores
    await raw.graph.add_node(_decision(1, created_at="2026-07-01T00:00:00+00:00"))
    raw.graph = AsyncGraphStore(raw.db)

    with pytest.raises(RecallRefused) as refused:
        await read_page(
            storage=storage, registry_db=raw.db, agent_id=AGENT, node_types=("decision",)
        )

    assert refused.value.reason == "unscoped"


@pytest.mark.parametrize(
    "mode", [PrivacyMode.EPHEMERAL, PrivacyMode.ISOLATED, PrivacyMode.DEIDENTIFIED]
)
async def test_generic_read_through_privacy_wrapper(stores, tmp_path, mode):
    storage, raw = stores
    await raw.graph.add_node(_decision(1, created_at="2026-07-01T00:00:00+00:00"))
    agent = _agent(storage, tmp_path)
    save = await _save(agent)
    assert (await _recall(save, node_type="decision"))["count"] == 1

    # The raw store, unwrapped, is refused outright.
    with pytest.raises(RecallRefused) as unwrapped:
        await read_page(
            storage=raw, registry_db=raw.db, agent_id=AGENT, node_types=("decision",)
        )
    assert unwrapped.value.reason == "not_privacy_wrapped"

    strategic = await _strategic(agent)
    memory = _memory(agent)
    storage.set_privacy_mode(mode)
    results = {
        "recall": await save.recall(node_type="decision"),
        "recall_get": await save.recall_get(
            _decision(1, created_at=None).node_id, node_type="decision"
        ),
        # Every alias reads through the same gate.
        "recall_decisions": await memory.recall_decisions(),
        "recall_action_items": await memory.recall_action_items(),
        "recall_interactions": await memory.recall_interactions(
            person_concept_id=f"concept:{AGENT}:alice"
        ),
        "recall_patterns": await strategic.recall_patterns(),
        "recall_blockers": await strategic.recall_blockers(),
    }

    for name, result in results.items():
        assert result.status is ToolResultStatus.ERROR, name
        assert result.data["refused"] == "privacy", name


# ---------------------------------------------------------------------------
# Governed types: canonical status
# ---------------------------------------------------------------------------


async def _pattern_and_its_assertion(storage, tmp_path):
    agent = _agent(storage, tmp_path)
    strategic = await _strategic(agent)
    added = await strategic.strategy_add_pattern(
        pattern="reviews find real defects", implication="keep them"
    )
    assert added.status is ToolResultStatus.OK, added
    assertions = await storage.query_assertions(AssertionQuery(limit=100))
    assert len(assertions) == 1, "the ledger projection wrote the row's assertion"
    save = await _save(agent)
    data = await _recall(save, node_type="strategy_pattern")
    assert data["count"] == 1
    assert data["results"][0]["canonical_status"] == "active"
    return agent, strategic, save, assertions[0], data["results"][0]["node_id"]


async def test_retracted_assertion_node_not_returned(stores, tmp_path):
    storage, _ = stores
    agent, strategic, save, assertion, node_id = await _pattern_and_its_assertion(
        storage, tmp_path
    )

    await storage.retract_assertion(
        assertion.assertion_id, assertion.revision_id, operation_id="test:retract:4155"
    )

    data = await _recall(save, node_type="strategy_pattern")
    assert data["count"] == 0
    assert data["withheld"] == {"canonical_inactive": 1}
    assert (await strategic.recall_patterns()).data["count"] == 0
    lookup = await save.recall_get(node_id, node_type="strategy_pattern")
    assert lookup.status is ToolResultStatus.ERROR
    assert lookup.data["missing"] == "canonical_inactive"


async def test_erased_assertion_node_not_returned(stores, tmp_path):
    storage, _ = stores
    agent, strategic, save, assertion, node_id = await _pattern_and_its_assertion(
        storage, tmp_path
    )

    await storage.erase_assertion(assertion.assertion_id, operation_id="test:erase:4155")

    # Erasure deletes the assertion and keeps no record of it, so the read
    # cannot tell it from one never written: it is withheld as absent.
    for include_retired in (False, True):
        data = await _recall(
            save, node_type="strategy_pattern", include_retired=include_retired
        )
        assert data["count"] == 0, include_retired
        assert data["withheld"] == {"canonical_absent": 1}
    assert (await strategic.recall_patterns(include_superseded=True)).data["count"] == 0
    lookup = await save.recall_get(node_id, node_type="strategy_pattern", include_retired=True)
    assert lookup.data["missing"] == "canonical_absent"


async def test_a_retired_pattern_returns_only_on_request_labelled_retracted(stores, tmp_path):
    storage, _ = stores
    agent, strategic, save, assertion, node_id = await _pattern_and_its_assertion(
        storage, tmp_path
    )
    pattern_id = (await storage.get_node(node_id)).properties["row_id"]

    await strategic.strategy_supersede_pattern(pattern_id=pattern_id, reason="re-measured")

    current = await _recall(save, node_type="strategy_pattern")
    retired = await _recall(save, node_type="strategy_pattern", include_retired=True)
    assert current["count"] == 0
    assert [row["node_id"] for row in retired["results"]] == [node_id]
    assert retired["results"][0]["canonical_status"] == AssertionStatus.RETRACTED.value


async def test_a_governed_row_with_no_assertion_is_withheld_as_absent(stores, tmp_path):
    """ANONYMOUS writes no canonical assertion, and erasure leaves none (#4155 review).

    The two cannot be told apart, so the row stays withheld, under its own
    reason, and the ledger's note names that cause rather than a hand edit.
    A row written in a mode that does write assertions is still returned.
    """
    storage, _ = stores
    agent, strategic, save, assertion, kept = await _pattern_and_its_assertion(
        storage, tmp_path
    )
    storage.set_privacy_mode(PrivacyMode.ANONYMOUS)

    added = await strategic.strategy_add_pattern(pattern="learned while anonymous")

    assert added.status is ToolResultStatus.OK, added
    nodes = await storage.graph.query_nodes_by_type_and_property(
        "strategy_pattern", filters={"agent_id": AGENT}
    )
    assert len(nodes) == 2, "the graph projection wrote the row"
    assert [a.assertion_id for a in await storage.query_assertions(AssertionQuery(limit=100))] == [
        assertion.assertion_id
    ], "the assertion projection did not: anonymous mode refuses the write"

    data = await _recall(save, node_type="strategy_pattern")
    recalled = await strategic.recall_patterns()

    assert [row["node_id"] for row in data["results"]] == [kept]
    assert data["withheld"] == {"canonical_absent": 1}
    assert "erased" in data["withheld_reasons"]["canonical_absent"]
    assert [row["node_id"] for row in recalled.data["patterns"]] == [kept]
    assert recalled.data["withheld"] == {"canonical_absent": 1}
    assert recalled.status is ToolResultStatus.PARTIAL
    assert "no canonical assertion" in recalled.error
    assert "This read ran in anonymous mode" in recalled.error
    assert "reactivated by hand" not in recalled.error, (
        "an absent assertion is not a retraction undone by hand"
    )


# ---------------------------------------------------------------------------
# message: trash state at the source
# ---------------------------------------------------------------------------


_EPOCH = datetime(2026, 7, 1, tzinfo=timezone.utc)


async def _message_with_node(storage, raw, text, *, concept=None):
    row_id = await storage.add_conversation("user", text)
    node_id = f"message:{AGENT}:{row_id}"
    await raw.graph.add_node(GraphNode(
        node_id=node_id,
        node_type="message",
        label=f"Message {row_id}",
        properties={
            "message_id": str(row_id),
            "agent_id": AGENT,
            # Later rows are newer, so the newest message has the highest id.
            "created_at": (_EPOCH + timedelta(minutes=row_id)).isoformat(),
        },
    ))
    if concept is not None:
        await raw.graph.add_edge(
            node_id, concept, "mentions", properties={"sentiment": "positive", "topics": []}
        )
    return row_id, node_id


async def test_trashed_message_not_returned(stores, tmp_path):
    storage, raw = stores
    kept_row, kept = await _message_with_node(storage, raw, "kept")
    trashed_row, trashed = await _message_with_node(storage, raw, "trashed")
    save = await _save(_agent(storage, tmp_path))
    assert (await _recall(save, node_type="message"))["count"] == 2

    assert await storage.delete_conversation_message(trashed_row, AGENT) is True

    # Recorded on #4155 (rv-4135-1 asked for the check): trashing a message
    # stamps conversation_history.deleted_at and leaves its graph node, so the
    # node is reachable through the graph after the message is in Trash.
    assert await raw.graph.get_node(trashed) is not None
    data = await _recall(save, node_type="message")
    assert [row["node_id"] for row in data["results"]] == [kept]
    assert data["withheld"] == {"source_trashed": 1}
    lookup = await save.recall_get(trashed, node_type="message")
    assert lookup.data["missing"] == "source_trashed"


async def test_an_archived_message_is_live_and_a_purged_one_is_not(stores, tmp_path):
    """Trash and erasure withhold a message; the consolidator's decay does not.

    ``memory_consolidator._archive_decayed`` stamps ``archived_at`` on any
    message whose strength decays below its floor, so archived means "faded",
    not trashed or erased. ``get_conversation_message`` hides archived rows,
    which is why the read no longer asks it.
    """
    storage, raw = stores
    archived_row, archived = await _message_with_node(storage, raw, "faded")
    purged_row, purged = await _message_with_node(storage, raw, "destroyed")
    # The decay archive's own write: archived_at on the row, nothing else.
    await raw.db.execute(
        "UPDATE conversation_history SET archived_at = ? WHERE id = ?",
        (_EPOCH.isoformat(), archived_row),
    )
    assert await storage.purge_conversation_message(purged_row, AGENT) is True
    assert await storage.get_conversation_message(archived_row) is None, (
        "the old liveness check: an archived row reads as gone"
    )
    save = await _save(_agent(storage, tmp_path))

    data = await _recall(save, node_type="message")
    archived_lookup = await save.recall_get(archived, node_type="message")
    purged_lookup = await save.recall_get(purged, node_type="message")

    assert [row["node_id"] for row in data["results"]] == [archived]
    assert data["withheld"] == {"source_trashed": 1}
    assert data["withheld_reasons"] == {"source_trashed": WITHHELD_REASONS["source_trashed"]}
    assert archived_lookup.status is ToolResultStatus.OK
    assert purged_lookup.data["missing"] == "source_trashed"
    assert "Trash or purged" in purged_lookup.error


async def test_live_message_ids_reads_trash_state_only(stores):
    """The conversation store's id-only liveness read the generic read uses."""
    storage, raw = stores
    live = await storage.add_conversation("user", "live")
    archived = await storage.add_conversation("user", "archived")
    trashed = await storage.add_conversation("user", "trashed")
    purged = await storage.add_conversation("user", "purged")
    await raw.db.execute(
        "UPDATE conversation_history SET archived_at = ? WHERE id = ?",
        (_EPOCH.isoformat(), archived),
    )
    assert await storage.delete_conversation_message(trashed, AGENT) is True
    assert await storage.purge_conversation_message(purged, AGENT) is True
    async with governed_storage(str(raw.db_path), OTHER) as (other, _):
        theirs = await other.add_conversation("user", "another agent's")

    found = await raw.conversation.live_message_ids(
        [live, archived, trashed, purged, theirs, 10_000]
    )

    assert found == {live, archived}
    assert await raw.conversation.live_message_ids([]) == set()
    # More ids than one IN clause takes.
    many = range(1, raw.conversation._LIVE_ID_CHUNK * 2 + 7)
    assert await raw.conversation.live_message_ids(many) == {live, archived}


async def _mentioned_messages(storage, raw, *, total, trashed_newest):
    """``total`` messages mentioning one person, the newest ``trashed_newest`` in Trash."""
    concept = f"concept:{AGENT}:alice"
    await raw.graph.add_node(GraphNode(
        node_id=concept, node_type="concept", label="alice",
        properties={"agent_id": AGENT, "category": "proper_noun"},
    ))
    made = [
        await _message_with_node(storage, raw, f"alice {n}", concept=concept)
        for n in range(total)
    ]
    for row_id, _ in made[total - trashed_newest:]:
        assert await storage.delete_conversation_message(row_id, AGENT) is True
    live_newest_first = [node_id for _, node_id in reversed(made[: total - trashed_newest])]
    return concept, live_newest_first


async def test_recall_interactions_fills_its_page_past_trashed_messages(stores, tmp_path):
    """A page whose newest rows are withheld is not returned short (#4155 review).

    30 messages mention the person and the 10 newest are in Trash, so 20 live
    interactions exist. One bounded query of 10 finds only trashed messages;
    the read queries on until the page is full and says what it withheld.
    """
    storage, raw = stores
    concept, live = await _mentioned_messages(storage, raw, total=30, trashed_newest=10)
    agent = _agent(storage, tmp_path)
    memory = _memory(agent)
    save = await _save(agent)

    interactions = await memory.recall_interactions(person_concept_id=concept, limit=10)
    generic = await _recall(save, node_type="message", limit=10)

    assert interactions.status is ToolResultStatus.OK, interactions
    assert [i["message_node_id"] for i in interactions.data["interactions"]] == live[:10]
    assert interactions.data["withheld"] == {"source_trashed": 10}
    assert interactions.data["stopped_short"] is False
    assert interactions.data["next_cursor"], "ten more live interactions remain"
    assert [row["node_id"] for row in generic["results"]] == live[:10]
    assert generic["withheld"] == {"source_trashed": 10}

    rest = await _recall(save, node_type="message", limit=10, cursor=generic["next_cursor"])
    assert [row["node_id"] for row in rest["results"]] == live[10:]
    assert rest["withheld"] == {}, "a withheld row is counted on the page that passed it"


async def test_a_page_stopped_at_the_scan_bound_says_so(stores, tmp_path, monkeypatch):
    """A short page is partial when the read stopped before the type ran out."""
    storage, raw = stores
    concept, live = await _mentioned_messages(storage, raw, total=30, trashed_newest=10)
    monkeypatch.setattr(registered_recall, "MAX_SCAN_QUERIES", 1)
    agent = _agent(storage, tmp_path)
    memory = _memory(agent)
    save = await _save(agent)

    interactions = await memory.recall_interactions(person_concept_id=concept, limit=10)
    generic = await save.recall(node_type="message", limit=10)

    for result, key in ((interactions, "interactions"), (generic, "results")):
        assert result.status is ToolResultStatus.PARTIAL, result
        assert result.data[key] == []
        assert result.data["withheld"] == {"source_trashed": 10}
        assert result.data["stopped_short"] is True
        assert result.data["next_cursor"]
        assert "older rows past next_cursor were not read" in result.error
    rest = await _recall(save, node_type="message", limit=10, cursor=generic.data["next_cursor"])
    assert [row["node_id"] for row in rest["results"]] == live[:10]


# ---------------------------------------------------------------------------
# Home and projection
# ---------------------------------------------------------------------------


async def test_result_names_home_and_projection(stores, tmp_path):
    storage, raw = stores
    agent = _agent(storage, tmp_path)
    extracted = _decision(1, created_at="2026-07-03T00:00:00+00:00")
    await raw.graph.add_node(extracted)
    strategic = await _strategic(agent)
    await strategic.strategy_add_decision(decision="adopt durable leases", rationale="cron drops them")
    save = await _save(agent)

    rows = (await _recall(save, node_type="decision,strategy_decision", limit=10))["results"]

    homes = {row["node_type"]: row["home"] for row in rows}
    assert homes["decision"] == {
        "registered_type": "decision",
        "owning_feature": "core",
        "canonical_home": "graph",
        "canonical_location": "graph_nodes",
        "is_projection": False,
        "origin": "original",
    }
    assert homes["strategy_decision"]["canonical_home"] == "file"
    assert homes["strategy_decision"]["canonical_location"] == "STRATEGY.yaml"
    assert homes["strategy_decision"]["is_projection"] is True
    assert homes["strategy_decision"]["origin"] == "projection"
    assert homes["strategy_decision"]["owning_feature"] == "StrategicMemoryFeature"

    # A projected decision still typed ``decision`` (pre-split writer or
    # restore) is labelled with the file home, as mark_superseded treats it.
    projected = next(row for row in rows if row["node_type"] == "strategy_decision")
    await raw.db.execute(
        "UPDATE graph_nodes SET node_type = 'decision' WHERE node_id = ?",
        (projected["node_id"],),
    )
    relabelled = (await _recall(save, node_type="decision", limit=10))["results"]
    marked = next(row for row in relabelled if row["node_id"] == projected["node_id"])
    assert marked["home"]["registered_type"] == "strategy_decision"
    assert marked["home"]["is_projection"] is True


# ---------------------------------------------------------------------------
# The fixed tools are aliases
# ---------------------------------------------------------------------------


async def _world(storage, raw, tmp_path):
    agent = _agent(storage, tmp_path)
    strategic = await _strategic(agent)
    for index, stamp in enumerate(("2026-07-03", "2026-06-30")):
        await raw.graph.add_node(_decision(index, created_at=f"{stamp}T00:00:00+00:00"))
    await raw.graph.add_node(_decision(
        9, created_at="2026-07-04T00:00:00+00:00", superseded_by=_decision(0, created_at=None).node_id
    ))
    await strategic.strategy_add_decision(decision="adopt durable leases", rationale="r")
    for index, status in enumerate(("pending", "done")):
        await raw.graph.add_node(GraphNode(
            node_id=f"action:{AGENT}:{index}",
            node_type="action_item",
            label=f"item {index}",
            properties={"text": f"item {index}", "status": status, "agent_id": AGENT,
                        "created_at": f"2026-07-0{index + 1}T00:00:00+00:00"},
        ))
    await raw.graph.add_node(GraphNode(
        node_id=f"action:{AGENT}:old",
        node_type="action_item",
        label="old",
        properties={"text": "old", "status": "pending", "agent_id": AGENT,
                    "created_at": "2026-07-05T00:00:00+00:00",
                    "superseded_by": f"action:{AGENT}:0"},
    ))
    kept = await strategic.strategy_add_pattern(pattern="kept pattern")
    retired = await strategic.strategy_add_pattern(pattern="retired pattern")
    await strategic.strategy_supersede_pattern(
        pattern_id=retired.data["pattern_id"], reason="no longer holds"
    )
    assert kept.status is ToolResultStatus.OK
    await strategic.strategy_add_blocker(issue="#1", title="open blocker", severity="high")
    resolved = await strategic.strategy_add_blocker(issue="#2", title="closed blocker")
    await strategic.strategy_resolve_blocker(
        issue=resolved.data["blocker_id"], resolution="landed"
    )
    concept = f"concept:{AGENT}:alice"
    await raw.graph.add_node(GraphNode(
        node_id=concept, node_type="concept", label="alice",
        properties={"agent_id": AGENT, "category": "proper_noun"},
    ))
    await _message_with_node(storage, raw, "talked to alice", concept=concept)
    trashed_row, _ = await _message_with_node(storage, raw, "alice again", concept=concept)
    await _message_with_node(storage, raw, "no mention")
    await storage.delete_conversation_message(trashed_row, AGENT)
    return agent, strategic, concept


async def test_alias_parity(stores, tmp_path):
    storage, raw = stores
    agent, strategic, concept = await _world(storage, raw, tmp_path)
    memory = _memory(agent)
    save = await _save(agent)

    async def generic(node_type, **kwargs):
        data = await _recall(save, node_type=node_type, limit=25, **kwargs)
        return [row["node_id"] for row in data["results"]]

    checks = [
        ("recall_decisions", memory.recall_decisions, "decisions", "id", "decision,strategy_decision", "include_superseded"),
        ("recall_action_items", memory.recall_action_items, "action_items", "id", "action_item", "include_superseded"),
        ("recall_patterns", strategic.recall_patterns, "patterns", "node_id", "strategy_pattern", "include_superseded"),
        ("recall_blockers", strategic.recall_blockers, "blockers", "node_id", "strategy_blocker", "include_resolved"),
    ]
    for tool, call, key, id_key, node_type, flag in checks:
        for retired in (False, True):
            result = await call(**{flag: retired})
            assert result.status in (ToolResultStatus.OK, ToolResultStatus.PARTIAL), (tool, result)
            ids = [row[id_key] for row in result.data[key]]
            assert ids, (tool, retired)
            assert ids == await generic(node_type, include_retired=retired), (tool, retired)
            assert all("home" in row for row in result.data[key])
            assert result.data["deprecated_alias"]["removal_release"] == ALIAS_REMOVAL_RELEASE
        # The default leaves retired rows out, the flag brings them back.
        assert len(await generic(node_type)) < len(await generic(node_type, include_retired=True))

    interactions = await memory.recall_interactions(person_concept_id=concept)
    page = await read_page(
        storage=storage,
        registry_db=raw.db,
        conversation_store=raw.conversation,
        agent_id=AGENT,
        node_types=("message",),
        linked_to=("mentions", concept),
    )
    assert [i["message_node_id"] for i in interactions.data["interactions"]] == [
        row.node.node_id for row in page.rows
    ]
    assert len(page.rows) == 1, "the trashed message's mention is not returned"
    assert interactions.data["interactions"][0]["properties"]["sentiment"] == "positive"


def test_fixed_tools_are_marked_deprecated_with_their_removal_release():
    owners = {"MemoryFeature": MemoryFeature, "StrategicMemoryFeature": StrategicMemoryFeature}
    assert {alias.tool for alias in FIXED_RECALL_ALIASES} == {
        "recall_decisions",
        "recall_action_items",
        "recall_patterns",
        "recall_blockers",
        "recall_interactions",
    }
    for alias in FIXED_RECALL_ALIASES:
        method = getattr(owners[alias.feature], alias.tool)
        description = method._tool_schema["description"]
        assert description.startswith("Deprecated alias of recall("), alias.tool
        assert f"removed in {alias.removal_release}" in description, alias.tool
        assert alias.removal_release == ALIAS_REMOVAL_RELEASE


def test_no_knowledge_prefixed_tools():
    from kestrel_sovereign.feature_inventory import build_inventory

    inventory = build_inventory()
    names = [tool.name for feature in inventory.features for tool in feature.tools]
    assert names, "the inventory found no tools at all"
    assert [name for name in names if name.startswith("knowledge_")] == []
    assert "recall_types" in names
