"""Person resolution reaches its fuzzy and pending passes on the production path (#3259).

The linker creates the mention's own concept node before the router runs,
so ``PersonResolver.resolve`` used to match every mention to itself and the
console's confirm-person flow never had anything to confirm. The linker also
split a multi-word name into one concept per token and dropped a lone
sentence-initial capitalised word. These tests drive the real linker and
router over a real privacy-governed graph, no doubles.
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from kestrel_sovereign.privacy import PrivacyMode
from kestrel_sovereign.storage.associative_linker import AssociativeLinker
from kestrel_sovereign.storage.async_storage import AsyncStorage
from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage
from kestrel_sovereign.storage.schema_router import PersonResolver, SchemaRouter

AGENT = "did:test:people"


@pytest_asyncio.fixture
async def pipeline(tmp_path):
    storage = AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT)
    await storage.initialize()
    wrapper = PrivacyEnforcingStorage(storage, PrivacyMode.NORMAL)
    graph = wrapper.graph
    try:
        yield AssociativeLinker(graph), SchemaRouter(graph=graph, db=storage.db, agent_id=AGENT), graph
    finally:
        await storage.close()


async def _say(linker, router, message_id, text):
    concepts = await linker.extract_and_link(message_id, text, AGENT)
    summary = await router.route(message_id=message_id, content=text, concepts=concepts, role="user")
    people = sorted(c.label for c in concepts if c.category in ("person", "proper_noun"))
    return people, summary


@pytest.mark.asyncio
async def test_a_first_name_shared_by_two_people_is_pending_through_the_pipeline(pipeline):
    linker, router, _graph = pipeline
    assert (await _say(linker, router, "m1", "I helped Jon Doe move."))[0] == ["jon doe"]
    assert (await _say(linker, router, "m2", "I called Jon Lee about the sink."))[0] == ["jon lee"]
    people, summary = await _say(linker, router, "m3", "Thanks Jon for everything.")
    assert people == ["jon"]
    assert summary["interactions"] == 1
    (pending,) = summary["pending_person_matches"]
    assert (pending["mentioned_label"], pending["message_id"]) == ("jon", "m3")
    assert sorted(pending["candidates"]) == [f"concept:{AGENT}:jon doe", f"concept:{AGENT}:jon lee"]


@pytest.mark.asyncio
async def test_one_other_person_with_the_first_name_is_a_fuzzy_match(pipeline):
    linker, router, _graph = pipeline
    await _say(linker, router, "m1", "I helped Jon Doe move.")
    people, summary = await _say(linker, router, "m2", "Thanks Jon for everything.")
    assert people == ["jon"] and summary["pending_person_matches"] == []
    match = await router.person_resolver.resolve("jon", AGENT, self_node_id=f"concept:{AGENT}:jon")
    assert (match.status, match.concept_id) == ("fuzzy", f"concept:{AGENT}:jon doe")


@pytest.mark.asyncio
async def test_a_mention_never_resolves_to_itself(pipeline):
    linker, router, _graph = pipeline
    await _say(linker, router, "m1", "Thanks Jon for everything.")
    resolver: PersonResolver = router.person_resolver
    own = f"concept:{AGENT}:jon"
    assert (await resolver.resolve("jon", AGENT, self_node_id=own)).status == "new"
    # Without the exclusion the mention's own node is an exact match: the
    # old behaviour, still available to a caller that resolves a free label.
    assert (await resolver.resolve("jon", AGENT)).concept_id == own


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text, people",
    [
        ("I helped Jon Doe move.", ["jon doe"]),
        ("Robert fixed the sink.", []),  # a sentence-initial capital is never a name
        ("Jon Doe helped me move.", ["doe"]),  # ...so a leading name loses its first token
        ("Today Robert fixed the sink.", ["robert"]),
        ("I met Mary Ann Smith yesterday.", ["mary ann smith"]),
        ("Talked to Alice, then Bob.", ["alice", "bob"]),
        ("We chose Jon Doe. Jon Lee objected.", ["jon doe", "lee"]),
    ],
)
async def test_linker_names(pipeline, text, people):
    linker, router, _graph = pipeline
    assert (await _say(linker, router, "m1", text))[0] == people


@pytest.mark.asyncio
async def test_a_classified_word_next_to_a_name_is_not_part_of_it(pipeline):
    """Review round 1 P1: "Robert Monday" is Robert, on Monday."""
    linker, router, graph = pipeline
    people, _summary = await _say(linker, router, "m1", "Lunch with Robert Monday was great.")
    assert people == ["robert"]
    assert (await graph.get_node(f"concept:{AGENT}:monday")).properties["category"] == "time"
    assert (await graph.get_node(f"concept:{AGENT}:robert")).properties["category"] == "proper_noun"
    assert await graph.get_node(f"concept:{AGENT}:robert monday") is None


@pytest.mark.asyncio
async def test_candidates_are_people_never_a_month(pipeline):
    linker, router, _graph = pipeline
    await _say(linker, router, "m1", "It all happened in March.")
    await _say(linker, router, "m2", "I called Marcus about it.")
    people, summary = await _say(linker, router, "m3", "I met Marc again.")
    assert people == ["marc"]
    assert summary["pending_person_matches"] == []
    match = await router.person_resolver.resolve("marc", AGENT, self_node_id=f"concept:{AGENT}:marc")
    assert (match.status, match.concept_id) == ("fuzzy", f"concept:{AGENT}:marcus")


@pytest.mark.asyncio
async def test_a_confirmed_match_converges(pipeline):
    """Review round 1 P2: after the user confirms who "Jon" is, the next
    mention resolves to that person and does not ask again."""
    from types import SimpleNamespace

    from kestrel_sovereign.features.memory.feature import MemoryFeature

    linker, router, graph = pipeline
    await _say(linker, router, "m1", "I helped Jon Doe move.")
    await _say(linker, router, "m2", "I called Jon Lee about the sink.")
    _people, summary = await _say(linker, router, "m3", "Thanks Jon for everything.")
    assert len(summary["pending_person_matches"]) == 1

    feature = SimpleNamespace(agent=SimpleNamespace(storage=SimpleNamespace(graph=graph)), agent_id=AGENT)
    confirm = getattr(MemoryFeature.confirm_person_match, "__wrapped__", MemoryFeature.confirm_person_match)
    result = await confirm(feature, message_id="m3", mentioned_label="jon", concept_id=f"concept:{AGENT}:jon doe")
    assert result.error is None, result.error

    _people, summary = await _say(linker, router, "m4", "Thanks Jon again.")
    assert summary["pending_person_matches"] == []
    match = await router.person_resolver.resolve("jon", AGENT, self_node_id=f"concept:{AGENT}:jon")
    assert (match.status, match.concept_id) == ("exact", f"concept:{AGENT}:jon doe")


@pytest.mark.asyncio
async def test_a_later_confirmation_overwrites_the_earlier_one(pipeline):
    """Review round 2 P1: a correction must take. The answer is a
    single-valued property on the mention's node, so there is never a second
    answer to pick between."""
    from types import SimpleNamespace

    from kestrel_sovereign.features.memory.feature import MemoryFeature

    linker, router, graph = pipeline
    await _say(linker, router, "m1", "I helped Jon Doe move.")
    await _say(linker, router, "m2", "I called Jon Lee about the sink.")
    await _say(linker, router, "m3", "Thanks Jon for everything.")
    feature = SimpleNamespace(agent=SimpleNamespace(storage=SimpleNamespace(graph=graph)), agent_id=AGENT)
    confirm = getattr(MemoryFeature.confirm_person_match, "__wrapped__", MemoryFeature.confirm_person_match)
    own = f"concept:{AGENT}:jon"
    assert (await confirm(feature, message_id="m3", mentioned_label="jon", concept_id=f"concept:{AGENT}:jon doe")).error is None
    assert (await router.person_resolver.resolve("jon", AGENT, self_node_id=own)).concept_id == f"concept:{AGENT}:jon doe"
    assert (await confirm(feature, message_id="m3", mentioned_label="jon", concept_id=f"concept:{AGENT}:jon lee")).error is None
    assert (await router.person_resolver.resolve("jon", AGENT, self_node_id=own)).concept_id == f"concept:{AGENT}:jon lee"
    assert (await graph.get_node(own)).properties["resolved_to"] == f"concept:{AGENT}:jon lee"


@pytest.mark.asyncio
async def test_the_resolved_person_keeps_accumulating_mentions(pipeline):
    """Review round 2 P2: after a confirmation, later mentions of the label
    are the confirmed person's interactions too."""
    from types import SimpleNamespace

    from kestrel_sovereign.features.memory.feature import MemoryFeature

    linker, router, graph = pipeline
    await _say(linker, router, "m1", "I helped Jon Doe move.")
    await _say(linker, router, "m2", "I called Jon Lee about the sink.")
    await _say(linker, router, "m3", "Thanks Jon for everything.")
    feature = SimpleNamespace(agent=SimpleNamespace(storage=SimpleNamespace(graph=graph)), agent_id=AGENT)
    confirm = getattr(MemoryFeature.confirm_person_match, "__wrapped__", MemoryFeature.confirm_person_match)
    await confirm(feature, message_id="m3", mentioned_label="jon", concept_id=f"concept:{AGENT}:jon doe")
    _people, summary = await _say(linker, router, "m4", "Thanks Jon again.")
    assert summary["pending_person_matches"] == [] and summary["interactions"] == 2
    into_doe = {e.source_id for e in await graph.get_edges(f"concept:{AGENT}:jon doe", direction="in") if e.label == "mentions"}
    assert f"message:{AGENT}:m4" in into_doe


@pytest.mark.asyncio
async def test_a_legacy_month_is_refused_on_read_without_a_re_mention(pipeline):
    """Review round 2 P2: a node written before categories were stored is
    classified on read with the linker's own keyword passes."""
    from kestrel_sovereign.storage.async_graph_store import GraphNode

    linker, router, graph = pipeline
    for label in ("march", "brooklyn", "marcus"):
        await graph.add_node(GraphNode(
            node_id=f"concept:{AGENT}:{label}", node_type="concept", label=label,
            properties={"agent_id": AGENT, "mention_count": 1},
        ))
    match = await router.person_resolver.resolve("marc", AGENT, self_node_id=f"concept:{AGENT}:marc")
    assert (match.status, match.concept_id) == ("fuzzy", f"concept:{AGENT}:marcus")


@pytest.mark.asyncio
async def test_a_node_with_associations_but_no_recorded_answer_is_not_exact(pipeline):
    linker, router, graph = pipeline
    await _say(linker, router, "m1", "Lunch with Alice was happy and calm.")
    edges = [e.label for e in await graph.get_edges(f"concept:{AGENT}:alice", direction="out")]
    assert "associated_with" in edges
    match = await router.person_resolver.resolve("alice", AGENT, self_node_id=f"concept:{AGENT}:alice")
    assert match.status == "new"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text, people",
    [
        ("I saw Jon And Doe yesterday.", ["doe", "jon"]),  # a capitalised stop word ends a run
        ("Waiting For Godot was long.", ["godot"]),
        ("I saw Alice, Bob went home.", ["alice", "bob"]),  # trailing punctuation ends a run
    ],
)
async def test_run_boundaries(pipeline, text, people):
    linker, router, _graph = pipeline
    assert (await _say(linker, router, "m1", text))[0] == people


@pytest.mark.asyncio
async def test_a_mention_stamps_the_category_on_an_existing_node(pipeline):
    """The refresh on every mention is what lets a node written before
    categories were stored acquire one; classify-on-read covers the
    resolver, this covers the record."""
    from kestrel_sovereign.storage.async_graph_store import GraphNode

    linker, router, graph = pipeline
    await graph.add_node(GraphNode(
        node_id=f"concept:{AGENT}:alice", node_type="concept", label="alice",
        properties={"agent_id": AGENT, "mention_count": 1},
    ))
    await _say(linker, router, "m1", "I talked to Alice about the project.")
    node = await graph.get_node(f"concept:{AGENT}:alice")
    assert node.properties["category"] == "proper_noun"
    assert node.properties["mention_count"] == 2
