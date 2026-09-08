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
import uuid

import pytest
import pytest_asyncio

from kestrel_sovereign.privacy import PrivacyMode
from kestrel_sovereign.storage.associative_linker import LinkedConcept
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.async_graph_store import AsyncGraphStore, GraphNode
from kestrel_sovereign.storage.async_storage import AsyncStorage
from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage, PrivacyViolationError
from kestrel_sovereign.storage.schema_router import PersonResolver, SchemaRouter

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
async def test_one_agent_never_lists_another_agents_people_on_a_shared_graph(tmp_path):
    """Two tenant-bound facades over ONE database: the shared-backend shape.
    Agent B's people are invisible to agent A's resolver."""
    from kestrel_sovereign.storage.db import SQLiteBackend

    raw = SQLiteBackend(str(tmp_path / "shared.db"))
    await raw.connect()
    db = AsyncDatabase(raw)
    await db._init_schema()
    db._initialized = True
    try:
        agent_a, agent_b = f"did:test:a-{uuid.uuid4().hex}", f"did:test:b-{uuid.uuid4().hex}"
        graph_a, graph_b = AsyncGraphStore(db, agent_id=agent_a), AsyncGraphStore(db, agent_id=agent_b)
        await _seed_people(graph_a, agent_a, "Alice")
        await _seed_people(graph_b, agent_b, "Alice", "Bob")

        assert {cid for cid, _ in await PersonResolver(graph_a)._list_person_concepts(agent_a)} == {
            f"concept:{agent_a}:alice"
        }
        assert {cid for cid, _ in await PersonResolver(graph_b)._list_person_concepts(agent_b)} == {
            f"concept:{agent_b}:alice", f"concept:{agent_b}:bob"
        }
        # Even asked about the other agent's prefix, a bound store yields nothing.
        assert await PersonResolver(graph_a)._list_person_concepts(agent_b) == []
    finally:
        await db.close()


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

    async def flaky(label, agent_id):
        calls.append(label)
        if label == "Bob":
            raise RuntimeError("resolver backend hiccup")
        return await real_resolve(label, agent_id)

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
