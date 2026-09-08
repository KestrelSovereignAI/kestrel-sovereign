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
    assert summary["pending_person_matches"] == [{
        "mentioned_label": "jon",
        "candidates": [f"concept:{AGENT}:jon doe", f"concept:{AGENT}:jon lee"],
        "message_id": "m3",
    }]


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
