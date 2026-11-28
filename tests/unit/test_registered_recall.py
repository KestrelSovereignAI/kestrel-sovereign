"""The generic paged recall read of a registered node type (#4155, rv-4135-1).

Every test here runs against a real SQLite agent store behind the real
``PrivacyEnforcingStorage``, with a real assertion store, because the
properties under test (ownership scope, keyset paging, canonical assertion
status, message trash state) live in those layers and not in a double.
"""

from __future__ import annotations

import base64
import dataclasses
import importlib
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from kestrel_sdk.tools.result import ToolResultStatus
from kestrel_sovereign.a2a.task_manager import TaskManager
from kestrel_sovereign.execution_custody import ExecutionAuthorityError
from kestrel_sovereign.features import registered_recall
from kestrel_sovereign.features.registered_recall import (
    MAX_PAGE_SIZE,
    WITHHELD_REASONS,
    RecallRefused,
    ScopeCarrier,
    ServedType,
    SourceCheck,
    list_recall_types,
    read_node,
    read_page,
)
from kestrel_sovereign.features.save import feature as save_module
from kestrel_sovereign.features.save.feature import SaveFeature
from kestrel_sovereign.features.strategic_memory.feature import (
    StrategicMemoryFeature,
)
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.knowledge import AssertionQuery, AssertionStatus
from kestrel_sovereign.privacy import PrivacyMode
from kestrel_sovereign.storage.async_graph_store import AsyncGraphStore, GraphNode
from kestrel_sovereign.storage.node_type_registry import (
    Governance,
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


async def _strategic(agent) -> StrategicMemoryFeature:
    feature = StrategicMemoryFeature(agent)
    await feature.initialize()
    agent.features["StrategicMemoryFeature"] = feature
    return feature


def _decision(index, *, created_at, agent=AGENT, node_type="decision", **extra):
    return GraphNode(
        node_id=f"{node_type}:{agent}:{index:016x}",
        node_type=node_type,
        label=f"{node_type} {index}",
        properties={
            "text": f"{node_type} {index}",
            "agent_id": agent,
            "confidence": 0.8,
            **({"created_at": created_at} if created_at is not None else {}),
            **extra,
        },
    )


async def _recall(save, **kwargs):
    result = await save.recall(**kwargs)
    assert result.status is ToolResultStatus.OK, result
    return result.data


def _ids(data):
    return [row["node_id"] for row in data["results"]]


async def _all_pages(save, node_type, limit, **kwargs):
    seen, cursor, pages = [], "", 0
    while True:
        data = await _recall(save, node_type=node_type, limit=limit, cursor=cursor, **kwargs)
        seen.extend(_ids(data))
        pages += 1
        cursor = data["next_cursor"] or ""
        if not cursor:
            return seen, pages


# ---------------------------------------------------------------------------
# recall_types and the served-type declarations
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


def test_every_served_type_declares_scope_source_and_canonical_requirement():
    """C3 (au-4155-1): no served type goes without its three declarations.

    The three fields have no default, so a type added to ``_SERVED`` without
    one of them fails to construct; a default added to a field later fails
    here. The table pins what each type declares (Findings 5 and 6, and the
    message scoping of rv-4155-1).
    """
    fields = {f.name: f for f in dataclasses.fields(ServedType)}
    for name in ("scope", "source", "canonical_required"):
        assert fields[name].default is dataclasses.MISSING, name
        assert fields[name].default_factory is dataclasses.MISSING, name
    served = registered_recall._SERVED
    assert set(served) == SERVED
    for node_type, declaration in served.items():
        assert isinstance(declaration.scope, ScopeCarrier), node_type
        assert isinstance(declaration.source, SourceCheck), node_type
        assert type(declaration.canonical_required) is bool, node_type
    assert {
        node_type: (d.scope, d.source, d.canonical_required) for node_type, d in served.items()
    } == {
        "action_item": (ScopeCarrier.AGENT_ID_PROPERTY, SourceCheck.SOURCE_MESSAGE_ID, False),
        "decision": (ScopeCarrier.AGENT_ID_PROPERTY, SourceCheck.SOURCE_MESSAGE_ID, False),
        "strategy_decision": (ScopeCarrier.AGENT_ID_PROPERTY, SourceCheck.NONE, False),
        "strategy_pattern": (ScopeCarrier.AGENT_ID_PROPERTY, SourceCheck.NONE, True),
        "strategy_blocker": (ScopeCarrier.AGENT_ID_PROPERTY, SourceCheck.NONE, True),
        "message": (ScopeCarrier.NODE_ID_PREFIX, SourceCheck.OWN_ROW, False),
    }
    with pytest.raises(TypeError):
        ServedType(scope=None, source=SourceCheck.NONE, canonical_required=False)
    with pytest.raises(TypeError):
        ServedType(
            scope=ScopeCarrier.AGENT_ID_PROPERTY, source=SourceCheck.NONE, canonical_required=None
        )
    with pytest.raises(ValueError):
        ServedType(
            scope=ScopeCarrier.AGENT_ID_PROPERTY,
            source=SourceCheck.OWN_ROW,
            canonical_required=False,
        )


async def test_every_canonical_required_type_is_served_with_a_mapper(stores):
    """Finding 11: named types, so a missing mapper fails here, not elsewhere.

    ``withheld_reason`` withholds a governed type whose producer has no
    mapper, so asking "is every served governed type mapped" could never
    fail. This names the types that must be served and checks the mapper
    directly.
    """
    _, raw = stores
    required = {t for t, d in registered_recall._SERVED.items() if d.canonical_required}
    assert required == {"strategy_pattern", "strategy_blocker"}
    listed = {d.entry.node_type: d for d in await list_recall_types(raw.db)}
    for node_type in sorted(required):
        described = listed[node_type]
        assert described.served, (node_type, described.withheld)
        assert described.governance.governed, node_type
        module = importlib.import_module(described.governance.producer.module)
        assert callable(getattr(module, "graph_node_assertion_ids", None)), node_type


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


def _governance_override(monkeypatch, node_type, governance):
    real = registered_recall.derive_governance

    def derive(name, **kwargs):
        return governance if name == node_type else real(name, **kwargs)

    monkeypatch.setattr(registered_recall, "derive_governance", derive)


async def test_a_canonical_required_type_governance_calls_ungoverned_is_refused(
    stores, tmp_path, monkeypatch
):
    """Finding 6: the canonical filter fails closed, not open.

    A producer/ontology pin mismatch, a broken vocabulary package or a
    missing producer module makes ``derive_governance`` report a ledger type
    ungoverned. The read then refuses the type instead of serving its graph
    nodes with no canonical check, so a retracted pattern stays out.
    """
    storage, _ = stores
    agent, _strategic_feature, save, assertion, node_id = await _pattern_and_its_assertion(
        storage, tmp_path
    )
    await storage.retract_assertion(
        assertion.assertion_id, assertion.revision_id, operation_id="test:retract:f6"
    )
    pin = "ledger_assertions binds ontology kestrel-vocab@1.3.0, not kestrel-vocab@1.2.0"
    _governance_override(
        monkeypatch, "strategy_pattern", Governance("strategy_pattern", False, pin)
    )

    for include_retired in (False, True):
        refused = await save.recall(node_type="strategy_pattern", include_retired=include_retired)
        assert refused.status is ToolResultStatus.ERROR, refused
        assert refused.data["refused"] == "withheld"
        assert "ungoverned" in refused.error and pin in refused.error
    lookup = await save.recall_get(node_id, node_type="strategy_pattern", include_retired=True)
    assert lookup.data["refused"] == "withheld"
    listed = {row["node_type"]: row for row in (await save.recall_types()).data["types"]}
    assert listed["strategy_pattern"]["generic_read"] is False
    assert pin in listed["strategy_pattern"]["withheld_reason"]
    assert listed["strategy_blocker"]["generic_read"] is True


@pytest.mark.parametrize("producer_module", [None, "kestrel_sovereign.features.registered_recall"])
async def test_a_canonical_required_type_with_no_mapping_producer_is_refused(
    stores, tmp_path, monkeypatch, producer_module
):
    """Governed with no producer, or one that maps no node: refused, not served.

    The listing names the reason rather than failing for every type.
    """
    storage, _ = stores
    agent, _strategic_feature, save, _assertion, node_id = await _pattern_and_its_assertion(
        storage, tmp_path
    )
    producer = derive_governance("strategy_pattern").producer
    if producer_module is not None:
        producer = dataclasses.replace(producer, module=producer_module)
    _governance_override(
        monkeypatch,
        "strategy_pattern",
        Governance("strategy_pattern", True, "test", None if producer_module is None else producer),
    )

    refused = await save.recall(node_type="strategy_pattern")
    lookup = await save.recall_get(node_id, node_type="strategy_pattern")
    listing = await save.recall_types()

    assert refused.data["refused"] == "withheld"
    assert "maps no graph node to its assertion" in refused.error
    assert lookup.data["refused"] == "withheld"
    assert listing.status is ToolResultStatus.OK, listing
    listed = {row["node_type"]: row for row in listing.data["types"]}
    assert listed["strategy_pattern"]["generic_read"] is False
    assert listed["strategy_blocker"]["generic_read"] is True


async def test_a_governed_type_not_declared_canonical_required_is_refused(
    stores, tmp_path, monkeypatch
):
    """A type that became governed is not served until its declaration says so."""
    storage, raw = stores
    await raw.graph.add_node(_decision(1, created_at="2026-07-01T00:00:00+00:00"))
    producer = derive_governance("strategy_pattern").producer
    _governance_override(
        monkeypatch, "decision", Governance("decision", True, "test producer", producer)
    )
    save = await _save(_agent(storage, tmp_path))

    refused = await save.recall(node_type="decision")

    assert refused.data["refused"] == "withheld"
    assert "does not require canonical status" in refused.error


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


async def test_a_cursor_for_another_type_or_not_a_string_is_refused(stores, tmp_path):
    storage, raw = stores
    for index in range(3):
        await raw.graph.add_node(_decision(index, created_at="2026-07-01T00:00:00+00:00"))
    save = await _save(_agent(storage, tmp_path))
    first = await _recall(save, node_type="decision", limit=1)

    other = await save.recall(node_type="action_item", cursor=first["next_cursor"])
    garbage = await save.recall(node_type="decision", cursor="not-a-cursor")
    not_a_string = await save.recall(node_type="decision", cursor=17)
    nested = await save.recall(
        node_type="decision", cursor=base64.urlsafe_b64encode(b"[" * 200_000).decode()
    )

    assert other.data["refused"] == "bad_cursor"
    assert garbage.data["refused"] == "bad_cursor"
    assert not_a_string.data["refused"] == "bad_cursor"
    assert nested.data["refused"] == "bad_cursor"


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

    assert _ids(data) == [mine.node_id]
    assert lookup.status is ToolResultStatus.ERROR
    assert lookup.data["missing"] == "not_found"


async def test_a_row_owned_here_but_naming_another_agent_is_not_returned(stores, tmp_path):
    """The agent_id property carrier holds even where the ownership ledger would not."""
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

    assert _ids(data) == [mine.node_id]
    assert lookup.data["missing"] == "foreign"


async def test_an_unbound_graph_store_is_refused(stores, tmp_path):
    storage, raw = stores
    await raw.graph.add_node(_decision(1, created_at="2026-07-01T00:00:00+00:00"))
    raw.graph = AsyncGraphStore(raw.db)

    with pytest.raises(RecallRefused) as refused:
        await read_page(
            storage=storage,
            registry_db=raw.db,
            agent_id=AGENT,
            node_types=("decision",),
            conversation_store=raw.conversation,
        )

    assert refused.value.reason == "unscoped"


async def test_a_missing_or_foreign_conversation_store_is_refused(stores, tmp_path):
    """A type whose source is a message is refused when that cannot be checked."""
    storage, raw = stores
    async with governed_storage(str(raw.db_path), OTHER) as (_, other_raw):
        for node_types in (("message",), ("decision",), ("action_item",)):
            with pytest.raises(RecallRefused) as missing:
                await read_page(
                    storage=storage, registry_db=raw.db, agent_id=AGENT, node_types=node_types
                )
            assert missing.value.reason == "unavailable", node_types
            with pytest.raises(RecallRefused) as foreign:
                await read_page(
                    storage=storage,
                    registry_db=raw.db,
                    agent_id=AGENT,
                    node_types=node_types,
                    conversation_store=other_raw.conversation,
                )
            assert foreign.value.reason == "unscoped", node_types
    # A type with no message source needs no conversation store.
    page = await read_page(
        storage=storage, registry_db=raw.db, agent_id=AGENT, node_types=("strategy_decision",)
    )
    assert page.rows == ()


@pytest.mark.parametrize(
    "mode", [PrivacyMode.EPHEMERAL, PrivacyMode.ISOLATED, PrivacyMode.DEIDENTIFIED]
)
async def test_generic_read_through_privacy_wrapper(stores, tmp_path, mode):
    storage, raw = stores
    await raw.graph.add_node(_decision(1, created_at="2026-07-01T00:00:00+00:00"))
    save = await _save(_agent(storage, tmp_path))
    assert (await _recall(save, node_type="decision"))["count"] == 1

    # The raw store, unwrapped, is refused outright.
    with pytest.raises(RecallRefused) as unwrapped:
        await read_page(
            storage=raw, registry_db=raw.db, agent_id=AGENT, node_types=("decision",)
        )
    assert unwrapped.value.reason == "not_privacy_wrapped"

    storage.set_privacy_mode(mode)
    results = {
        "recall": await save.recall(node_type="decision"),
        "recall_get": await save.recall_get(
            _decision(1, created_at=None).node_id, node_type="decision"
        ),
    }

    for name, result in results.items():
        assert result.status is ToolResultStatus.ERROR, name
        assert result.data["refused"] == "privacy", name


async def test_a_store_failure_is_a_failed_result_and_control_is_raised(
    stores, tmp_path, monkeypatch
):
    """Finding 7: only a refusal was turned into a result before."""
    storage, _ = stores
    save = await _save(_agent(storage, tmp_path))

    async def broken(**_):
        raise RuntimeError("the graph store is gone")

    monkeypatch.setattr(save_module, "read_page", broken)
    monkeypatch.setattr(save_module, "read_node", broken)
    failed_page = await save.recall(node_type="decision")
    failed_lookup = await save.recall_get("decision:x", node_type="decision")
    for result in (failed_page, failed_lookup):
        assert result.status is ToolResultStatus.ERROR, result
        assert "the graph store is gone" in result.error

    async def withdrawn(**_):
        raise ExecutionAuthorityError("original authority withdrawn")

    monkeypatch.setattr(save_module, "read_page", withdrawn)
    monkeypatch.setattr(save_module, "read_node", withdrawn)
    with pytest.raises(ExecutionAuthorityError):
        await save.recall(node_type="decision")
    with pytest.raises(ExecutionAuthorityError):
        await save.recall_get("decision:x", node_type="decision")


def test_recall_types_command_routes_to_recall_types(tmp_path):
    """Finding 10: a command routes by its first word.

    ``!recall`` is recall's, so ``!recall types`` ran a saved-items search
    for "types". This registers SaveFeature through the agent's own wiring
    and routes through the real TaskManager.
    """
    manager = TaskManager.__new__(TaskManager)
    manager._providers = {}
    manager._skill_to_agent = {}
    manager._command_to_skill = {}
    save = SaveFeature(_agent(MagicMock(), tmp_path))
    KestrelAgent._wire_feature_a2a(SimpleNamespace(task_manager=manager), save)

    assert manager.get_agent_for_command("!recall-types") == (save.name, "recall_types")


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
    assert _ids(retired) == [node_id]
    assert retired["results"][0]["canonical_status"] == AssertionStatus.RETRACTED.value


async def test_anonymous_mode_refuses_governed_types_and_their_rows_read_as_absent(
    stores, tmp_path
):
    """ANONYMOUS writes no canonical assertions (rv-4155-1, Finding 2).

    The read refuses the governed types there, naming the mode. A row written
    in it has no assertion, so a later read in a mode that checks canonical
    status withholds it as absent, while a row written with its assertion is
    still returned. Ungoverned types still read in ANONYMOUS.
    """
    storage, raw = stores
    agent, strategic, save, assertion, kept = await _pattern_and_its_assertion(
        storage, tmp_path
    )
    await raw.graph.add_node(_decision(1, created_at="2026-07-01T00:00:00+00:00"))
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

    for node_type in ("strategy_pattern", "strategy_blocker"):
        refused = await save.recall(node_type=node_type)
        assert refused.status is ToolResultStatus.ERROR, node_type
        assert refused.data["refused"] == "privacy", node_type
        assert "anonymous" in refused.error, node_type
    lookup = await save.recall_get(kept, node_type="strategy_pattern")
    assert lookup.data["refused"] == "privacy"
    assert (await _recall(save, node_type="decision"))["count"] == 1

    storage.set_privacy_mode(PrivacyMode.NORMAL)
    data = await _recall(save, node_type="strategy_pattern")
    assert _ids(data) == [kept]
    assert data["withheld"] == {"canonical_absent": 1}
    assert "anonymous" in data["withheld_reasons"]["canonical_absent"]


# ---------------------------------------------------------------------------
# message: scope by owner table and id prefix, trash state of its own row
# ---------------------------------------------------------------------------


_EPOCH = datetime(2026, 7, 1, tzinfo=timezone.utc)


def _message_properties(row_id, *, stamp_agent):
    return {
        "message_id": str(row_id),
        **({"agent_id": AGENT} if stamp_agent else {}),
        # Later rows are newer, so the newest message has the highest id.
        "created_at": (_EPOCH + timedelta(minutes=row_id)).isoformat(),
    }


async def _message_with_node(storage, raw, text, *, stamp_agent=True):
    row_id = await storage.add_conversation("user", text)
    node_id = f"message:{AGENT}:{row_id}"
    await raw.graph.add_node(GraphNode(
        node_id=node_id,
        node_type="message",
        label=f"Message {row_id}",
        properties=_message_properties(row_id, stamp_agent=stamp_agent),
    ))
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
    assert _ids(data) == [kept]
    assert data["withheld"] == {"source_trashed": 1}
    lookup = await save.recall_get(trashed, node_type="message")
    assert lookup.data["missing"] == "source_trashed"


async def test_a_message_without_an_agent_id_property_is_served(stores, tmp_path):
    """C2 (au-4155-1): owner row and id prefix scope a message, not the property."""
    storage, raw = stores
    row_id, node_id = await _message_with_node(storage, raw, "no property", stamp_agent=False)
    stored = await raw.graph.get_node(node_id)
    assert "agent_id" not in stored.properties, "the bound store stamps no property"
    save = await _save(_agent(storage, tmp_path))

    data = await _recall(save, node_type="message")
    lookup = await save.recall_get(node_id, node_type="message")

    assert _ids(data) == [node_id]
    assert data["withheld"] == {}
    assert lookup.status is ToolResultStatus.OK, lookup
    assert lookup.data["result"]["node_id"] == node_id


async def _insert_legacy_message_node(raw, agent_id, row_id):
    """A message node as a pre-#2649 writer left it: no agent_id, no owner row."""
    await raw.db.execute(
        "INSERT OR IGNORE INTO graph_nodes (node_id, node_type, label, properties) "
        "VALUES (?, 'agent', ?, ?)",
        (agent_id, agent_id, json.dumps({})),
    )
    node_id = f"message:{agent_id}:{row_id}"
    await raw.db.execute(
        "INSERT INTO graph_nodes (node_id, node_type, label, properties) VALUES (?, ?, ?, ?)",
        (node_id, "message", f"Message {row_id}",
         json.dumps(_message_properties(row_id, stamp_agent=False))),
    )
    return node_id


async def test_a_legacy_message_node_is_served_after_the_ownership_backfill(stores, tmp_path):
    """rv-4155-1 (4): an upgraded database's message nodes stay readable.

    Message nodes written before #2649 carry no ``properties.agent_id``; the
    ownership backfill records their owner from the id prefix and does not
    stamp the property. Pushing ``agent_id`` into SQL for ``message`` (as
    eca5d166 did) would drop every such node silently. Another agent's legacy
    node is owned by that agent and returned by neither read.
    """
    storage, raw = stores
    mine_row = await storage.add_conversation("user", "before the owner table")
    mine = await _insert_legacy_message_node(raw, AGENT, mine_row)
    async with governed_storage(str(raw.db_path), OTHER) as (other, other_raw):
        theirs_row = await other.add_conversation("user", "another agent's")
        theirs = await _insert_legacy_message_node(other_raw, OTHER, theirs_row)
    assert await raw.graph.get_node(mine) is None, "unowned before the backfill"

    await raw.db._backfill_graph_ownership()

    owners = dict(await raw.db.fetchall(
        "SELECT node_id, agent_id FROM graph_node_owners WHERE node_id IN (?, ?)",
        (mine, theirs),
    ))
    assert owners == {mine: AGENT, theirs: OTHER}
    assert "agent_id" not in (await raw.graph.get_node(mine)).properties
    save = await _save(_agent(storage, tmp_path))

    data = await _recall(save, node_type="message")
    page = await read_page(
        storage=storage,
        registry_db=raw.db,
        agent_id=AGENT,
        node_types=("message",),
        conversation_store=raw.conversation,
    )
    found = await read_node(
        storage=storage,
        registry_db=raw.db,
        agent_id=AGENT,
        node_type="message",
        node_id=mine,
        conversation_store=raw.conversation,
    )
    other_lookup = await read_node(
        storage=storage,
        registry_db=raw.db,
        agent_id=AGENT,
        node_type="message",
        node_id=theirs,
        conversation_store=raw.conversation,
    )

    assert _ids(data) == [mine]
    assert [row.node.node_id for row in page.rows] == [mine]
    assert page.withheld == {}
    assert found.row is not None and found.row.node.node_id == mine
    assert other_lookup.row is None and other_lookup.missing == "not_found"


async def test_a_message_node_owned_here_with_another_agents_prefix_is_foreign(
    stores, tmp_path
):
    """The id prefix is the message scope carrier: failing it is withheld and counted."""
    storage, raw = stores
    _, mine = await _message_with_node(storage, raw, "mine")
    row_id = await storage.add_conversation("user", "a live row of mine")
    stray = GraphNode(
        node_id=f"message:{OTHER}:{row_id}",
        node_type="message",
        label=f"Message {row_id}",
        properties=_message_properties(row_id, stamp_agent=False),
    )
    await AsyncGraphStore(raw.db, agent_id=OTHER).add_node(stray)
    await raw.db.execute(
        "INSERT INTO graph_node_owners (node_id, agent_id) VALUES (?, ?)",
        (stray.node_id, AGENT),
    )
    assert await raw.graph.get_node(stray.node_id) is not None, "owned here too"
    save = await _save(_agent(storage, tmp_path))

    data = await _recall(save, node_type="message")
    lookup = await save.recall_get(stray.node_id, node_type="message")

    assert _ids(data) == [mine]
    assert data["withheld"] == {"foreign": 1}
    assert lookup.data["missing"] == "foreign"
    assert "not found" in lookup.error


async def test_an_archived_message_is_live_and_a_purged_one_is_not(stores, tmp_path):
    """Trash and erasure withhold a message; the consolidator's decay does not.

    ``memory_consolidator._archive_decayed`` stamps ``archived_at`` on any
    message whose strength decays below its floor, so archived means "faded",
    not trashed or erased. ``get_conversation_message`` hides archived rows,
    which is why the read does not ask it.
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
        "the other liveness check: an archived row reads as gone"
    )
    save = await _save(_agent(storage, tmp_path))

    data = await _recall(save, node_type="message")
    archived_lookup = await save.recall_get(archived, node_type="message")
    purged_lookup = await save.recall_get(purged, node_type="message")

    assert _ids(data) == [archived]
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


async def _messages(storage, raw, *, total, trashed_newest):
    """``total`` messages, the newest ``trashed_newest`` in Trash."""
    made = [await _message_with_node(storage, raw, f"message {n}") for n in range(total)]
    for row_id, _ in made[total - trashed_newest:]:
        assert await storage.delete_conversation_message(row_id, AGENT) is True
    return [node_id for _, node_id in reversed(made[: total - trashed_newest])]


async def test_a_page_is_filled_past_withheld_rows(stores, tmp_path):
    """A page whose newest rows are withheld is not returned short (#4155 review).

    30 messages exist and the 10 newest are in Trash, so 20 are live. One
    bounded query of 10 finds only trashed messages; the read queries on
    until the page is full and says what it withheld.
    """
    storage, raw = stores
    live = await _messages(storage, raw, total=30, trashed_newest=10)
    save = await _save(_agent(storage, tmp_path))

    first = await _recall(save, node_type="message", limit=10)

    assert _ids(first) == live[:10]
    assert first["withheld"] == {"source_trashed": 10}
    assert first["stopped_short"] is False
    assert first["next_cursor"], "ten more live messages remain"
    rest = await _recall(save, node_type="message", limit=10, cursor=first["next_cursor"])
    assert _ids(rest) == live[10:]
    assert rest["withheld"] == {}, "a withheld row is counted on the page that passed it"


async def test_a_page_stopped_at_the_scan_bound_says_so(stores, tmp_path, monkeypatch):
    """A short page is partial when the read stopped before the type ran out."""
    storage, raw = stores
    live = await _messages(storage, raw, total=30, trashed_newest=10)
    monkeypatch.setattr(registered_recall, "MAX_SCAN_QUERIES", 1)
    save = await _save(_agent(storage, tmp_path))

    result = await save.recall(node_type="message", limit=10)

    assert result.status is ToolResultStatus.PARTIAL, result
    assert result.data["results"] == []
    assert result.data["withheld"] == {"source_trashed": 10}
    assert result.data["stopped_short"] is True
    assert result.data["next_cursor"]
    assert "older rows past next_cursor were not read" in result.error
    rest = await _recall(save, node_type="message", limit=10, cursor=result.data["next_cursor"])
    assert _ids(rest) == live[:10]


# ---------------------------------------------------------------------------
# action_item and decision: the message each was extracted from
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("node_type", ["action_item", "decision"])
async def test_a_row_whose_source_message_is_gone_is_withheld(stores, tmp_path, node_type):
    """Finding 5: an extracted row is served only while its source message is live.

    ``schema_router`` writes the message text into the row and names the
    message as ``source_message_id``. Trash and purge leave the row in place
    (no cascade), so the read withholds it and counts it. A row with no
    source message, and an archived source, are served.
    """
    storage, raw = stores
    rows = {}
    for name in ("live", "archived", "trashed", "purged"):
        rows[name] = await storage.add_conversation("user", f"call my lawyer ({name})")
    nodes = {
        name: _decision(index, created_at=f"2026-07-0{index + 1}T00:00:00+00:00",
                        node_type=node_type, source_message_id=str(row_id))
        for index, (name, row_id) in enumerate(rows.items())
    }
    nodes["unsourced"] = _decision(5, created_at="2026-07-06T00:00:00+00:00",
                                   node_type=node_type, source_message_id=None)
    nodes["unverifiable"] = _decision(6, created_at="2026-07-07T00:00:00+00:00",
                                      node_type=node_type, source_message_id="not-a-row")
    for node in nodes.values():
        await raw.graph.add_node(node)
    await raw.db.execute(
        "UPDATE conversation_history SET archived_at = ? WHERE id = ?",
        (_EPOCH.isoformat(), rows["archived"]),
    )
    assert await storage.delete_conversation_message(rows["trashed"], AGENT) is True
    assert await storage.purge_conversation_message(rows["purged"], AGENT) is True
    save = await _save(_agent(storage, tmp_path))

    data = await _recall(save, node_type=node_type, limit=50)

    assert set(_ids(data)) == {
        nodes[name].node_id for name in ("live", "archived", "unsourced")
    }
    assert data["withheld"] == {"source_trashed": 2, "source_unverifiable": 1}
    for name in ("trashed", "purged"):
        assert await raw.graph.get_node(nodes[name].node_id) is not None, "no purge cascade"
        lookup = await save.recall_get(nodes[name].node_id, node_type=node_type)
        assert lookup.data["missing"] == "source_trashed", name
    unverifiable = await save.recall_get(nodes["unverifiable"].node_id, node_type=node_type)
    assert unverifiable.data["missing"] == "source_unverifiable"


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


def test_no_knowledge_prefixed_tools():
    from kestrel_sovereign.feature_inventory import build_inventory

    inventory = build_inventory()
    names = [tool.name for feature in inventory.features for tool in feature.tools]
    assert names, "the inventory found no tools at all"
    assert [name for name in names if name.startswith("knowledge_")] == []
    assert "recall_types" in names
