"""Person resolution reads through the privacy-governed graph facade (#3228).

``PersonResolver`` reached for ``graph.db``; the privacy-governing graph proxy
refuses that handle by design (#2672), ``SchemaRouter.route`` caught the
refusal as a warning, and every person resolution on a real agent silently
returned a zero summary — after the interaction edge for the first person had
already been written. These tests drive the production wiring: a real
``PrivacyEnforcingStorage`` facade over real storage, no raw reach-through, no
warning, and no partial writes.
"""

from __future__ import annotations

import logging

import pytest
import pytest_asyncio

from kestrel_sovereign.privacy import PrivacyMode
from kestrel_sovereign.storage.associative_linker import LinkedConcept
from kestrel_sovereign.storage.async_graph_store import GraphNode
from kestrel_sovereign.storage.async_storage import AsyncStorage
from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage, PrivacyViolationError
from kestrel_sovereign.storage.schema_router import SchemaRouter

AGENT = "did:test:agent-a"


@pytest_asyncio.fixture
async def governed(tmp_path):
    """The production shape: MemorySystem hands SchemaRouter the privacy
    wrapper's graph proxy, never the raw store."""
    storage = AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT)
    await storage.initialize()
    wrapper = PrivacyEnforcingStorage(storage, PrivacyMode.NORMAL)
    try:
        yield wrapper, storage
    finally:
        await storage.close()


async def _seed_people(graph, agent, *labels):
    for label in labels:
        await graph.add_node(GraphNode(
            node_id=f"concept:{agent}:{label.lower()}", node_type="concept",
            label=label, properties={"agent_id": agent, "mention_count": 1},
        ))


async def _seed_message(graph, agent, message_id):
    """The linker creates the message node before the router runs; an edge
    whose endpoints the agent does not both own is refused by the store."""
    await graph.add_node(GraphNode(
        node_id=f"message:{agent}:{message_id}", node_type="message",
        label=message_id, properties={"agent_id": agent},
    ))


def _mention(agent, label):
    return LinkedConcept(node_id=f"concept:{agent}:{label.lower()}", label=label, category="person")


async def _route(router, message_id, content, *labels):
    await _seed_message(router.graph, router.agent_id, message_id)
    return await router.route(
        message_id=message_id, content=content,
        concepts=[_mention(router.agent_id, label) for label in labels], role="user",
    )


class _Collect(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@pytest.fixture
def router_log():
    handler = _Collect()
    target = logging.getLogger("kestrel_sovereign.storage.schema_router")
    target.addHandler(handler)
    try:
        yield handler
    finally:
        target.removeHandler(handler)


def test_the_governed_facade_still_refuses_the_raw_handle(governed):
    """The proxy's default-deny boundary is intact; the fix is the reader.
    The router hands its resolver the very facade it was given, so the
    resolver's graph refuses the raw handle too (a resolver built over the
    underlying store would answer the same reads but bypass the boundary)."""
    wrapper, storage = governed
    with pytest.raises(PrivacyViolationError, match="refuses to forward 'db'"):
        wrapper.graph.db
    router = SchemaRouter(graph=wrapper.graph, db=storage.db, agent_id=AGENT)
    assert router.person_resolver.graph is router.graph
    with pytest.raises(PrivacyViolationError, match="refuses to forward 'db'"):
        router.person_resolver.graph.db


@pytest.mark.asyncio
async def test_resolution_through_the_facade_without_warning(governed, router_log):
    wrapper, storage = governed
    await _seed_people(wrapper.graph, AGENT, "Alice Smith", "Robert", "Jon Doe", "Jon Lee")
    router = SchemaRouter(graph=wrapper.graph, db=storage.db, agent_id=AGENT)
    resolver = router.person_resolver

    exact = await resolver.resolve("alice smith", AGENT)
    assert (exact.status, exact.concept_id) == ("exact", f"concept:{AGENT}:alice smith")
    fuzzy = await resolver.resolve("Rob", AGENT)
    assert (fuzzy.status, fuzzy.concept_id) == ("fuzzy", f"concept:{AGENT}:robert")
    ambiguous = await resolver.resolve("Jon", AGENT)
    assert ambiguous.status == "pending" and len(ambiguous.candidates) == 2
    nobody = await resolver.resolve("Zelda", AGENT)
    assert (nobody.status, nobody.concept_id) == ("new", None)

    # Through route(): the linker has already created the mention's own
    # concept node, so resolution lands on it and the edge is enriched.
    await _seed_people(wrapper.graph, AGENT, "Robert")
    summary = await _route(router, "msg-1", "Lunch with Robert was great.", "Robert")
    assert summary["interactions"] == 1
    assert summary["pending_person_matches"] == []
    edges = await wrapper.graph.get_edges(f"message:{AGENT}:msg-1")
    assert [e.target_id for e in edges] == [f"concept:{AGENT}:robert"]
    assert edges[0].properties["sentiment"]
    assert not any("Interaction enrichment failed" in m for m in router_log.messages), router_log.messages


@pytest.mark.asyncio
async def test_a_failing_resolution_leaves_no_partial_edges(governed, router_log, monkeypatch):
    """A message naming two people: if the second resolution raises, the
    first person's edge must not have been written, and the summary's zero
    must be the truth."""
    wrapper, storage = governed
    await _seed_people(wrapper.graph, AGENT, "Alice", "Bob")
    router = SchemaRouter(graph=wrapper.graph, db=storage.db, agent_id=AGENT)

    calls = []
    real_resolve = router.person_resolver.resolve

    async def flaky(label, agent_id, **kwargs):
        calls.append(label)
        if label == "Bob":
            raise RuntimeError("resolver backend hiccup")
        return await real_resolve(label, agent_id, **kwargs)

    monkeypatch.setattr(router.person_resolver, "resolve", flaky)
    summary = await _route(router, "msg-2", "Alice and Bob argued about lunch.", "Alice", "Bob")
    assert summary["interactions"] == 0
    assert calls == ["Alice", "Bob"]
    edges = await wrapper.graph.get_edges(f"message:{AGENT}:msg-2")
    assert edges == [], "an edge was written before every person resolved"
    assert any("Interaction enrichment failed" in m for m in router_log.messages)


@pytest.mark.asyncio
async def test_all_people_resolve_then_all_edges_are_written(governed, router_log):
    wrapper, storage = governed
    await _seed_people(wrapper.graph, AGENT, "Alice", "Bob")
    router = SchemaRouter(graph=wrapper.graph, db=storage.db, agent_id=AGENT)
    summary = await _route(router, "msg-3", "Alice and Bob argued about lunch.", "Alice", "Bob")
    assert summary["interactions"] == 2
    targets = {e.target_id for e in await wrapper.graph.get_edges(f"message:{AGENT}:msg-3")}
    assert targets == {f"concept:{AGENT}:alice", f"concept:{AGENT}:bob"}
    assert not any("Interaction enrichment failed" in m for m in router_log.messages)


@pytest.mark.asyncio
async def test_memory_system_wires_the_governed_graph_into_person_resolution(tmp_path, router_log):
    """The production constructor: MemorySystem hands SchemaRouter the privacy
    wrapper's graph proxy, the router hands it to its resolver, and a message
    through the linker + router enriches its person with no warning."""
    from kestrel_sovereign.storage.memory_system import MemorySystem

    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT) as raw:
        wrapper = PrivacyEnforcingStorage(raw, PrivacyMode.NORMAL)
        ms = MemorySystem(storage=raw, agent_id=AGENT, privacy_storage=wrapper)
        await ms.initialize()

        assert ms.router.person_resolver.graph is ms.router.graph
        with pytest.raises(PrivacyViolationError, match="refuses to forward 'db'"):
            ms.router.graph.db

        # The linker does not tag a sentence-initial capitalised word as a name.
        concepts = await ms.linker.extract_and_link("m-1", "Today Robert fixed the sink.", AGENT)
        people = [c for c in concepts if c.category in ("person", "proper_noun")]
        assert [c.label for c in people] == ["robert"]
        summary = await ms.router.route(message_id="m-1", content="Today Robert fixed the sink.", concepts=concepts, role="user")
        assert summary["interactions"] == 1
        edges = await ms.router.graph.get_edges(f"message:{AGENT}:m-1")
        assert {e.target_id for e in edges} >= {f"concept:{AGENT}:robert"}
        assert not any("Interaction enrichment failed" in m for m in router_log.messages), router_log.messages


class _FailingWrites:
    """Delegate to the governed facade, but make the n-th call of one writer
    raise, so a lane fails part-way through its writes."""

    def __init__(self, inner, method, fail_on_call):
        self._inner, self._method, self._fail_on, self.calls = inner, method, fail_on_call, 0

    def __getattr__(self, name):
        target = getattr(self._inner, name)
        if name != self._method:
            return target

        async def wrapped(*args, **kwargs):
            self.calls += 1
            if self.calls == self._fail_on:
                raise RuntimeError(f"{name} failed on call {self.calls}")
            return await target(*args, **kwargs)

        return wrapped


@pytest.mark.asyncio
async def test_an_edge_write_failing_part_way_reports_the_edges_that_landed(governed, router_log):
    """Two people, the second edge write fails: the first edge is in the
    graph, and the summary says one, not zero."""
    wrapper, storage = governed
    await _seed_people(wrapper.graph, AGENT, "Alice", "Bob")
    router = SchemaRouter(graph=wrapper.graph, db=storage.db, agent_id=AGENT)
    await _seed_message(router.graph, AGENT, "msg-4")
    router.graph = _FailingWrites(wrapper.graph, "add_edge", fail_on_call=2)

    summary = await router.route(
        message_id="msg-4", content="Alice and Bob argued about lunch.",
        concepts=[_mention(AGENT, "Alice"), _mention(AGENT, "Bob")], role="user",
    )
    targets = [e.target_id for e in await wrapper.graph.get_edges(f"message:{AGENT}:msg-4")]
    assert targets == [f"concept:{AGENT}:alice"]
    assert summary["interactions"] == 1
    assert summary["pending_person_matches"] == []
    assert any("Interaction enrichment failed" in m for m in router_log.messages)


@pytest.mark.asyncio
async def test_an_action_item_write_failing_part_way_reports_the_nodes_that_landed(governed, router_log, monkeypatch):
    wrapper, storage = governed
    router = SchemaRouter(graph=wrapper.graph, db=storage.db, agent_id=AGENT)
    await _seed_message(router.graph, AGENT, "msg-5")
    monkeypatch.setattr(router.action_extractor, "extract_with_evidence", lambda content: [("file the report", "need to"), ("call the bank", "must")])
    monkeypatch.setattr(router.decision_extractor, "extract", lambda content: [])
    router.graph = _FailingWrites(wrapper.graph, "add_node", fail_on_call=2)

    summary = await router.route(message_id="msg-5", content="irrelevant", concepts=[], role="user")
    assert summary["action_items"] == 1
    nodes = await wrapper.graph.get_nodes_by_type("action_item")
    assert [n.properties["text"] for n in nodes] == ["file the report"]
    assert any("Action item routing failed" in m for m in router_log.messages)


@pytest.mark.asyncio
async def test_a_decision_write_failing_part_way_reports_the_nodes_that_landed(governed, router_log, monkeypatch):
    wrapper, storage = governed
    router = SchemaRouter(graph=wrapper.graph, db=storage.db, agent_id=AGENT)
    await _seed_message(router.graph, AGENT, "msg-6")
    monkeypatch.setattr(router.action_extractor, "extract_with_evidence", lambda content: [])
    monkeypatch.setattr(router.decision_extractor, "extract", lambda content: ["use postgres", "ship friday"])
    router.graph = _FailingWrites(wrapper.graph, "add_node", fail_on_call=2)

    summary = await router.route(message_id="msg-6", content="irrelevant", concepts=[], role="user")
    assert summary["decisions"] == 1
    nodes = await wrapper.graph.get_nodes_by_type("decision")
    assert [n.properties["text"] for n in nodes] == ["use postgres"]
    assert any("Decision routing failed" in m for m in router_log.messages)
