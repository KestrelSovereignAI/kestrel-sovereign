"""search_memory returns compact, size-bounded hits (#3784).

On a live agent ``search_memory`` returned 431,601 characters for five hits:
each hit carried its whole message (setup turns with inlined documents run to
a megabyte) and its whole metadata envelope, concept URIs repeating the DID
included. A hit is now a pointer plus an excerpt, the whole result fits the
orchestrator's tool-result cap, and ``read_message_by_id`` reads the rest.
"""
import json
from types import SimpleNamespace

import pytest
from unittest.mock import AsyncMock, MagicMock

from kestrel_sdk.tools.result import ToolResult, ToolResultStatus
from kestrel_sovereign.features.base import (
    largest_fitting_result,
    serialized_result_len,
)
from kestrel_sovereign.features.memory.feature import MemoryFeature
from kestrel_sovereign.features.memory.search_results import (
    DEFAULT_SNIPPET_CHARS,
    ECHO_CHARS,
    ELLIPSIS,
    MAX_HIT_CONCEPTS,
    MIN_SNIPPET_CHARS,
    STORED_SESSION_ID_CHARS,
    SearchHit,
    bare_concept_name,
    bare_concepts,
    bounded_echo,
    fit_search_result,
    locate_match,
    snippet_bounds,
)
from kestrel_sovereign.storage.async_conversation_store import (
    SEARCH_MATCH_EXACT,
    SEARCH_MATCH_TERMS,
    AsyncConversationStore,
    _strip_search_wrappers,
    search_match_projection,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase

AGENT = "did:pkh:eip155:1:0x00000000000000000000000000000000000c3784"
#: The default orchestrator cap (``KESTREL_MAX_TOOL_RESULT_CHARS``), pinned so
#: the regression bound does not move with the environment.
RESULT_CAP = 8000
COMPACT_KEYS = {
    "message_id", "session_id", "role", "timestamp", "score", "match",
    "snippet", "snippet_start", "snippet_end", "content_chars",
}


@pytest.fixture
def result_cap(monkeypatch):
    """Pin the orchestrator's per-result cap; return a setter for others."""
    import kestrel_sovereign.kestrel_agent as kestrel_agent

    def set_cap(value: int) -> int:
        monkeypatch.setattr(kestrel_agent, "MAX_TOOL_RESULT_CHARS", value)
        return value

    set_cap(RESULT_CAP)
    return set_cap


@pytest.fixture
async def store(tmp_path):
    db = await AsyncDatabase.sqlite(str(tmp_path / "search.db"))
    try:
        yield AsyncConversationStore(db, agent_id=AGENT)
    finally:
        await db.close()


async def _add(store, role: str, content: str, **kwargs) -> int:
    """Persist one message and return its row id."""
    await store.add_conversation(role, content, **kwargs)
    row = await store.db.fetchone(
        "SELECT MAX(id) FROM conversation_history WHERE agent_id = ?", (AGENT,)
    )
    return int(row[0])


def _feature(store) -> MemoryFeature:
    feature = MemoryFeature.__new__(MemoryFeature)
    feature.agent_id = AGENT
    feature._get_conversation_store = lambda: store
    return feature


def _in_live_turn(feature: MemoryFeature, session_id=None) -> MemoryFeature:
    """Make ``feature``'s calls run inside the agent's live turn.

    The two turn-lifecycle accessors a feature reads through
    ``Feature._owns_live_turn`` and ``Feature._turn_session_id``.
    """
    feature.agent = SimpleNamespace(
        owns_live_turn=lambda: True,
        get_turn_bound_session_id=lambda: session_id,
    )
    return feature


def _size(result, tool_name: str) -> int:
    return serialized_result_len(result, tool_name=tool_name)


def _huge_message(marker: str, chars: int = 1_100_000) -> str:
    half = chars // 2
    return ("lorem ipsum " * (half // 12)) + f" {marker} " + ("dolor sit " * (half // 10))


# -- The regression the issue asks for ---------------------------------------


@pytest.mark.asyncio
async def test_a_million_character_message_cannot_make_a_large_search_result(
    store, result_cap
):
    """A store holding ≥1M-character messages must not produce a search result
    larger than the cap: every hit is an excerpt, never the message."""
    for copy in range(3):
        await _add(
            store, "assistant", _huge_message(f"the scheduler worker died (copy {copy})")
        )
    await _add(store, "assistant", "Yes: the scheduler worker died at 03:00.")
    feature = _feature(store)

    result = await feature.search_memory(query="scheduler worker died", limit=5)

    assert result.status is ToolResultStatus.OK
    assert _size(result, "search_memory") <= RESULT_CAP
    # Measured as the orchestrator does: json.dumps of the wrapped envelope.
    wrapped = {**result.to_dict(), "tool": "search_memory", "success": True}
    assert len(json.dumps(wrapped)) <= RESULT_CAP
    hits = result.data["results"]
    assert result.data["count"] == len(hits) == 4
    assert result.data["omitted"] == 0 and result.data["truncated"] is False
    huge = [hit for hit in hits if hit["content_chars"] >= 1_000_000]
    assert len(huge) == 3
    for hit in huge:
        assert "scheduler worker died" in hit["snippet"]
        assert hit["snippet"].startswith(ELLIPSIS) and hit["snippet"].endswith(ELLIPSIS)
        assert hit["snippet_end"] - hit["snippet_start"] <= DEFAULT_SNIPPET_CHARS


@pytest.mark.asyncio
async def test_twenty_huge_hits_shrink_and_drop_but_stay_under_the_cap(store, result_cap):
    """Many large matches: snippets shrink first, then the lowest-ranked hits
    are dropped — counted and named, never silently."""
    result_cap(3000)
    for copy in range(20):
        await _add(
            store, "assistant", _huge_message(f"rotation drill {copy}", chars=40_000)
        )
    feature = _feature(store)

    result = await feature.search_memory(query="rotation drill", limit=20)

    assert _size(result, "search_memory") <= 3000
    assert result.status is ToolResultStatus.PARTIAL
    data = result.data
    assert data["matched"] == 20
    assert data["omitted"] == 20 - data["count"] > 0
    assert data["truncated"] is True
    assert f"{data['omitted']} of 20 matching message(s) omitted" in result.error
    assert "3,000 characters" in result.error
    for hit in data["results"]:
        assert hit["snippet_end"] - hit["snippet_start"] == MIN_SNIPPET_CHARS
    # Best first: the hits kept are the newest (highest-ranked) ones.
    kept_ids = [hit["message_id"] for hit in data["results"]]
    assert kept_ids == sorted(kept_ids, reverse=True)


@pytest.mark.asyncio
async def test_snippets_shrink_before_any_hit_is_dropped(store, result_cap):
    result_cap(5000)
    for copy in range(12):
        await _add(
            store, "assistant", _huge_message(f"tidal model {copy}", chars=20_000)
        )
    feature = _feature(store)

    result = await feature.search_memory(query="tidal model", limit=12)

    assert result.status is ToolResultStatus.OK
    assert result.data["count"] == 12 and result.data["omitted"] == 0
    widths = {hit["snippet_end"] - hit["snippet_start"] for hit in result.data["results"]}
    assert len(widths) == 1
    width = widths.pop()
    assert MIN_SNIPPET_CHARS <= width < DEFAULT_SNIPPET_CHARS
    assert _size(result, "search_memory") <= 5000


@pytest.mark.asyncio
async def test_the_smallest_cap_still_fits_with_a_huge_query(store, result_cap):
    """The query is the caller's own text; a 26,000-character message passed
    as the query must not fill the result by itself."""
    result_cap(1000)
    query = "needle " + ("x" * 26_000)
    await _add(store, "assistant", query)
    feature = _feature(store)

    result = await feature.search_memory(query=query, limit=5)

    assert _size(result, "search_memory") <= 1000
    assert len(result.data["query"]) < ECHO_CHARS + 40
    assert "chars omitted]" in result.data["query"]


# -- The compact shape --------------------------------------------------------


@pytest.mark.asyncio
async def test_hits_carry_no_metadata_envelope_and_bare_concepts(store, result_cap):
    message_id = await _add(
        store, "user",
        "The scheduler worker died at noon while Nellie was away.",
        metadata={
            "extracted_concepts": [
                f"concept:{AGENT}:scheduler",
                "nellie",
                "nellie",
                f"concept:{AGENT}:noon",
            ],
            "schema_routing": {
                "pending_person_matches": [
                    {"candidates": [f"concept:{AGENT}:nellie"]},
                ],
            },
            "stashed": False,
            "stash_id": None,
            "excluded_from_context": False,
            "excluded_reason": None,
            "summarized_into": None,
        },
        session_id="session-3784",
    )
    feature = _feature(store)

    result = await feature.search_memory(query="worker died", limit=5)

    assert result.status is ToolResultStatus.OK
    [hit] = result.data["results"]
    assert set(hit) == COMPACT_KEYS | {"concepts"}
    assert hit["message_id"] == message_id
    assert hit["session_id"] == "session-3784"
    assert hit["role"] == "user"
    assert hit["timestamp"]
    assert hit["score"] == 1.0 and hit["match"] == SEARCH_MATCH_EXACT
    assert hit["concepts"] == ["scheduler", "nellie", "noon"]
    encoded = json.dumps(result.to_dict())
    assert "concept:" not in encoded
    assert AGENT not in encoded
    for lifecycle in ("stash", "excluded", "summarized", "schema_routing"):
        assert lifecycle not in encoded


@pytest.mark.asyncio
async def test_term_fallback_hits_report_their_score(store, result_cap):
    await _add(
        store, "assistant", "Meridian reviewed the first Kestrel agent's memory context."
    )
    feature = _feature(store)

    result = await feature.search_memory(
        query="Meridian first Kestrel agent context management", limit=5
    )

    [hit] = result.data["results"]
    assert hit["match"] == SEARCH_MATCH_TERMS
    assert 0.6 <= hit["score"] < 1.0


@pytest.mark.asyncio
async def test_the_active_question_is_suppressed_with_a_real_store(store, result_cap):
    """search_history now returns row ids, so the active-question echo filter
    (which compares ids) works against the real store, not only a mock."""
    historical = await _add(
        store, "user", "What is my favorite planet?", session_id="session-a"
    )
    await _add(store, "assistant", "You said Neptune.", session_id="session-a")
    current = await _add(
        store, "user", "What is my favorite planet?", session_id="session-b"
    )
    feature = _in_live_turn(_feature(store), session_id="session-b")

    result = await feature.search_memory(query="favorite planet", limit=10)

    ids = [hit["message_id"] for hit in result.data["results"]]
    assert historical in ids
    assert current not in ids


@pytest.mark.asyncio
async def test_searching_an_older_session_keeps_its_last_question(store, result_cap):
    """The active question lives in the TURN's session. Searching another
    session must not take that session's newest question for the active one
    and hide it (#3784 review: the filter read the searched session)."""
    historical = await _add(
        store, "user", "What is my favorite planet?", session_id="session-a"
    )
    await _add(store, "assistant", "You said Neptune.", session_id="session-a")
    await _add(
        store, "user", "Do you remember what I asked about planets?",
        session_id="session-b",
    )
    feature = _in_live_turn(_feature(store), session_id="session-b")

    result = await feature.search_memory(
        query="favorite planet", limit=10, session_id="session-a"
    )

    assert result.status is ToolResultStatus.OK
    assert [hit["message_id"] for hit in result.data["results"]] == [historical]


@pytest.mark.asyncio
async def test_no_question_is_hidden_outside_a_live_turn(store, result_cap):
    """A search run outside any turn (a command, a cron tick) has no active
    question: the newest stored question is history like any other."""
    newest = await _add(store, "user", "What is my favorite planet?")
    feature = _feature(store)

    result = await feature.search_memory(query="favorite planet", limit=10)

    assert [hit["message_id"] for hit in result.data["results"]] == [newest]


@pytest.mark.asyncio
async def test_a_wake_turns_own_row_is_its_active_question(store, result_cap):
    """A signal wake persists its own user row, so in a wake turn that row is
    the active one; the session's last ordinary question is history."""
    historical = await _add(
        store, "user", "What is my favorite planet?", session_id="session-a"
    )
    await _add(
        store, "user", "Autonomous wake: check the planet question",
        session_id="session-a",
        metadata={"signal_wake": {"source": "cron.self_followup"}},
    )
    feature = _in_live_turn(_feature(store), session_id="session-a")

    result = await feature.search_memory(query="favorite planet", limit=10)

    assert historical in [hit["message_id"] for hit in result.data["results"]]


@pytest.mark.asyncio
async def test_an_operator_notice_after_the_question_does_not_displace_it(
    store, result_cap
):
    """Operator notices are persisted as user rows DURING a turn, after its
    question: the question is still the active row."""
    current = await _add(
        store, "user", "What is my favorite planet?", session_id="session-b"
    )
    await _add(
        store, "user", "Operator notice: favorite planet queue drained",
        session_id="session-b", metadata={"operator_signal": True},
    )
    feature = _in_live_turn(_feature(store), session_id="session-b")

    result = await feature.search_memory(query="favorite planet", limit=10)

    assert current not in [hit["message_id"] for hit in result.data["results"]]


@pytest.mark.asyncio
async def test_search_history_rows_carry_identity_and_match(store):
    exact_id = await _add(store, "user", "kestrel memory search")
    terms_id = await _add(
        store, "assistant", "our kestrel agent has great memory and search capabilities"
    )

    rows = await store.search_history("kestrel memory search", limit=10)

    by_id = {row["id"]: row for row in rows}
    assert by_id[exact_id]["match"] == SEARCH_MATCH_EXACT
    assert by_id[exact_id]["score"] == 1.0
    assert by_id[terms_id]["match"] == SEARCH_MATCH_TERMS
    assert by_id[terms_id]["created_at"] is not None


def _legacy_row_quoting(memory: str, said: str) -> str:
    """A legacy user row: a bare RELEVANT MEMORIES block, then the turn."""
    filler = "".join(
        f"[Memory {n}] Assistant: filler line {n} about other things entirely\n"
        for n in range(2, 30)
    )
    return (
        "--- RELEVANT MEMORIES (from past conversations) ---\n"
        "NOTE: These are retrieved from earlier conversations, not the current session.\n"
        f"[Memory 1] User: {memory}\n{filler}--- END MEMORIES ---\n"
        f"<user_input>\n{said}\n</user_input>"
    )


@pytest.mark.asyncio
async def test_a_snippet_shows_what_was_said_not_a_quoted_memory(store, result_cap):
    """Search admits hits on the wrapper-free projection, so the snippet is
    cut around the match there: a recalled memory quoted inside a legacy
    row's bare RELEVANT MEMORIES block must not supply it (#3784 review)."""
    message_id = await _add(
        store, "user",
        _legacy_row_quoting(
            "the scheduler worker died at 03:00", "the scheduler worker died at 04:00"
        ),
    )
    feature = _feature(store)

    [hit] = (
        await feature.search_memory(query="scheduler worker died", limit=5)
    ).data["results"]

    assert hit["message_id"] == message_id
    assert "the scheduler worker died at 04:00" in hit["snippet"]
    assert "03:00" not in hit["snippet"]
    # The range still indexes the text the reader returns.
    read = await feature.read_message_by_id(message_id, offset=hit["snippet_start"])
    window = read.data["content"][: hit["snippet_end"] - hit["snippet_start"]]
    assert hit["snippet"].strip(ELLIPSIS) == " ".join(window.split())


def test_locate_match_skips_wrapper_text_and_maps_back_to_the_source():
    text = _legacy_row_quoting("deploy failed at noon", "deploy failed at dusk")
    start, end = locate_match(text, "deploy failed", ("deploy", "failed"))
    assert text[start:end].lower() == "deploy failed"
    assert text[end:].startswith(" at dusk")
    # Only the wrappers match: there is nothing to point at.
    assert locate_match(text, "noon", ("noon",)) == (0, 0)


def test_search_match_projection_is_the_admission_projection():
    """The offset-keeping projection is exactly the one search admits on."""
    samples = [
        "",
        "plain text",
        _legacy_row_quoting("a", "b"),
        "<retrieved_context><memories>m</memories></retrieved_context>x<user_input>y",
        "--- RELEVANT MEMORIES truncated heading\nbody\nmore",
        "pre <documents>d\ne</documents> mid --- END MEMORIES --- post </user_input>",
    ]
    for text in samples:
        projection = search_match_projection(text)
        assert projection.text == _strip_search_wrappers(text)
        for index, character in enumerate(projection.text):
            start, end = projection.source_span(index, index + 1)
            assert 0 <= start <= end <= len(text)
            if end - start == 1:
                assert text[start] == character


# -- Reading a hit in full ----------------------------------------------------


@pytest.mark.asyncio
async def test_a_hits_snippet_start_is_an_offset_into_the_read(store, result_cap):
    await _add(store, "assistant", _huge_message("the vault key rotated"))
    feature = _feature(store)
    [hit] = (await feature.search_memory(query="vault key rotated", limit=5)).data["results"]

    read = await feature.read_message_by_id(
        hit["message_id"], offset=hit["snippet_start"]
    )

    assert read.status is ToolResultStatus.OK
    assert _size(read, "read_message_by_id") <= RESULT_CAP
    assert read.data["offset"] == hit["snippet_start"]
    assert read.data["total_chars"] == hit["content_chars"]
    window = read.data["content"][: hit["snippet_end"] - hit["snippet_start"]]
    assert hit["snippet"].strip(ELLIPSIS) == " ".join(window.split())
    assert read.data["next_offset"] == hit["snippet_start"] + read.data["length"]
    assert read.data["truncated"] is True


@pytest.mark.asyncio
async def test_read_message_by_id_pages_a_whole_message(store, result_cap):
    text = "".join(f"line {n} with \"quotes\" and é\n" for n in range(6000))
    message_id = await _add(store, "assistant", text)
    feature = _feature(store)

    parts, offset, chunks = [], 0, 0
    while True:
        read = await feature.read_message_by_id(message_id, offset=offset)
        assert read.status is ToolResultStatus.OK
        assert _size(read, "read_message_by_id") <= RESULT_CAP
        parts.append(read.data["content"])
        chunks += 1
        if read.data["next_offset"] is None:
            assert "End of message." in read.confirmation
            break
        assert read.data["next_offset"] > offset
        offset = read.data["next_offset"]
        assert chunks < 1000
    assert "".join(parts) == text
    assert chunks > 1


@pytest.mark.asyncio
async def test_read_message_by_id_honours_length(store, result_cap):
    message_id = await _add(store, "assistant", "abcdefghij" * 100)
    feature = _feature(store)

    read = await feature.read_message_by_id(message_id, offset=10, length=25)

    assert read.data["content"] == ("abcdefghij" * 100)[10:35]
    assert read.data["next_offset"] == 35


@pytest.mark.asyncio
async def test_read_message_by_id_strips_sent_form_like_search(store, result_cap):
    message_id = await _add(
        store, "user",
        "<retrieved_context>\n<memories>old</memories>\n</retrieved_context>\n"
        "<user_input>\nwhat about Article IV?\n</user_input>",
        metadata={"sent_form": True},
    )
    feature = _feature(store)

    read = await feature.read_message_by_id(message_id)

    assert read.data["content"] == "what about Article IV?"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kwargs", "fragment"),
    [
        ({"message_id": "abc"}, "message_id must be an integer"),
        ({"message_id": 0}, "message_id must be in"),
        ({"message_id": 1, "offset": -1}, "offset must be in"),
        ({"message_id": 1, "length": 0}, "length must be in"),
        ({"message_id": 999}, "No live message found with id 999"),
        ({"message_id": 1, "offset": 50}, "offset 50 is past the end of message 1"),
    ],
)
async def test_read_message_by_id_refuses_bad_requests(store, result_cap, kwargs, fragment):
    await _add(store, "assistant", "short")
    feature = _feature(store)

    read = await feature.read_message_by_id(**kwargs)

    assert read.status is ToolResultStatus.ERROR
    assert fragment in read.error


@pytest.mark.asyncio
async def test_read_message_by_id_session_guard(store, result_cap):
    message_id = await _add(
        store, "assistant", "inside one session", session_id="session-a"
    )
    feature = _feature(store)

    refused = await feature.read_message_by_id(message_id, session_id="session-b")
    allowed = await feature.read_message_by_id(message_id, session_id="session-a")

    assert refused.status is ToolResultStatus.ERROR
    assert "does not belong to session" in refused.error
    assert allowed.data["content"] == "inside one session"
    assert allowed.data["session_id"] == "session-a"


@pytest.mark.asyncio
async def test_a_huge_stored_session_id_cannot_starve_a_read(store, result_cap):
    """Stored session ids are unbounded and copied into every chunk: an
    8,000-character one under the 8,000 cap used to leave room for one
    character, then return it anyway, over the cap (#3784 review)."""
    session_id = "s" * 8000
    message_id = await _add(
        store, "assistant", "x" * 30_000, session_id=session_id
    )
    feature = _feature(store)

    read = await feature.read_message_by_id(message_id)

    assert read.status is ToolResultStatus.OK
    assert _size(read, "read_message_by_id") <= RESULT_CAP
    assert read.data["length"] > 5000
    assert read.data["session_id"].startswith("s" * STORED_SESSION_ID_CHARS + ELLIPSIS)
    assert "chars omitted]" in read.data["session_id"]


@pytest.mark.asyncio
async def test_a_huge_stored_session_id_cannot_crowd_out_search_hits(store, result_cap):
    message_id = await _add(
        store, "assistant", "the beacon relit", session_id="s" * 8000
    )
    feature = _feature(store)

    result = await feature.search_memory(query="beacon relit", limit=5)

    assert result.status is ToolResultStatus.OK
    assert _size(result, "search_memory") <= RESULT_CAP
    [hit] = result.data["results"]
    assert hit["message_id"] == message_id
    assert len(hit["session_id"]) < 600


@pytest.mark.asyncio
async def test_read_labels_are_left_out_visibly_when_no_character_fits_beside_them(
    store, result_cap
):
    """At the smallest cap a label that escapes to twice its length leaves no
    room for content: the labels go, named, and the chunk still fits."""
    result_cap(1000)
    message_id = await _add(
        store, "assistant", "y" * 5000, session_id='"' * STORED_SESSION_ID_CHARS
    )
    feature = _feature(store)

    read = await feature.read_message_by_id(message_id)

    assert read.status is ToolResultStatus.OK
    assert _size(read, "read_message_by_id") <= 1000
    assert read.data["labels_omitted"] == ["session_id", "role", "timestamp"]
    assert "session_id" not in read.data
    assert read.data["length"] >= 1
    assert "left out to fit" in read.confirmation


@pytest.mark.asyncio
async def test_a_read_that_cannot_fit_one_character_fails_instead(
    store, result_cap, monkeypatch
):
    """Never the old unmeasured one-character fallback: a read whose bare
    envelope exceeds the cap fails, and the failure itself is small."""
    import kestrel_sovereign.features.memory.feature as memory_feature

    monkeypatch.setattr(memory_feature, "orchestrator_result_cap", lambda: 200)
    message_id = await _add(store, "assistant", "z" * 5000)
    feature = _feature(store)

    read = await feature.read_message_by_id(message_id)

    assert read.status is ToolResultStatus.ERROR
    assert "cannot fit the next character" in read.error
    assert "content" not in (read.data or {})


# -- The recall_emotional fallback is the same search -------------------------


@pytest.mark.asyncio
async def test_recall_emotional_fallback_is_compact_and_bounded(store, result_cap):
    for copy in range(4):
        await _add(
            store, "assistant", _huge_message(f"lighthouse keeper {copy}", chars=300_000)
        )
    feature = _feature(store)
    feature._memory_system = None
    feature.agent = SimpleNamespace(memory_system=None)

    result = await feature.recall_emotional(query="lighthouse keeper", mood="neutral")

    assert result.status is ToolResultStatus.PARTIAL
    assert "emotional weighting" in result.error
    assert _size(result, "recall_emotional") <= RESULT_CAP
    hits = result.data["fallback_results"]
    assert len(hits) == 4
    assert all(set(hit) == COMPACT_KEYS for hit in hits)


# -- Helpers ------------------------------------------------------------------


def test_snippet_bounds_center_the_match_and_slide_back_from_the_end():
    assert snippet_bounds(50, 10, 15, 400) == (0, 50)
    assert snippet_bounds(10_000, 5_000, 5_010, 300) == (4_900, 5_200)
    assert snippet_bounds(10_000, 20, 25, 300) == (0, 300)
    assert snippet_bounds(10_000, 9_990, 9_995, 300) == (9_700, 10_000)


def test_locate_match_prefers_the_phrase_then_the_earliest_term():
    text = "Alpha beta. Later: GAMMA delta, then beta gamma."
    assert locate_match(text, "gamma delta", ("gamma", "delta")) == (19, 30)
    assert locate_match(text, "delta beta", ("delta", "beta")) == (6, 10)
    assert locate_match(text, "zeta", ("zeta",)) == (0, 0)


def test_locate_match_keeps_positions_when_lowercasing_changes_length():
    # "İ" lowercases to two characters, so str.lower() positions would drift.
    text = "İİİİ then the needle"
    start, end = locate_match(text, "NEEDLE", ("needle",))
    assert text[start:end] == "needle"


def test_bare_concepts_collapse_uris_dedupe_and_cap():
    names = [f"concept:{AGENT}:c{n}" for n in range(MAX_HIT_CONCEPTS + 3)]
    metadata = {"extracted_concepts": ["c0", *names, 7]}
    assert bare_concepts(metadata, AGENT) == tuple(f"c{n}" for n in range(MAX_HIT_CONCEPTS))
    assert bare_concept_name("concept:did:other:agent:moon", None) == "moon"
    assert bare_concepts(None, AGENT) == ()
    assert bare_concepts({"extracted_concepts": "not-a-list"}, AGENT) == ()


def test_bounded_echo_marks_what_it_drops():
    assert bounded_echo(None) is None
    assert bounded_echo("short") == "short"
    long = "q" * (ECHO_CHARS + 500)
    echoed = bounded_echo(long)
    assert echoed.startswith("q" * ECHO_CHARS + ELLIPSIS)
    assert "[500 chars omitted]" in echoed


def _hit(n: int, text: str) -> SearchHit:
    return SearchHit(
        message_id=n, session_id="s", role="assistant", timestamp="t", score=1.0,
        match=SEARCH_MATCH_EXACT, text=text, match_start=0, match_end=1, concepts=(),
    )


def test_fit_search_result_fails_visibly_when_nothing_fits():
    def render(entries, omitted):
        return ToolResult.ok(confirmation="x" * 5000, data={"results": entries})

    result = fit_search_result([_hit(1, "abc")], render, cap=1000, tool_name="search_memory")

    assert result.status is ToolResultStatus.ERROR
    assert "cannot fit even an empty result" in result.error


def test_largest_fitting_result_returns_the_largest_measured_fit():
    def build(n):
        return ToolResult.ok(confirmation="c", data={"body": "x" * n})

    base = serialized_result_len(build(0), tool_name="t")
    found = largest_fitting_result(build, 0, 10_000, cap=base + 123, tool_name="t")

    assert found is not None
    n, result = found
    assert n == 123
    assert serialized_result_len(result, tool_name="t") == base + 123
    assert largest_fitting_result(build, 50, 100, cap=base + 10, tool_name="t") is None


def test_largest_fitting_result_never_returns_an_unmeasured_result():
    """A non-monotone build (a dropped ellipsis can shorten a wider snippet)
    may cost a smaller n, never a result over the cap."""
    sizes = {n: (5 if n % 2 else 50) for n in range(0, 21)}

    def build(n):
        return ToolResult.ok(confirmation="c", data={"body": "x" * sizes[n]})

    base = serialized_result_len(build(0), tool_name="t") - sizes[0]
    found = largest_fitting_result(build, 0, 20, cap=base + 10, tool_name="t")

    assert found is not None
    n, result = found
    assert n % 2 == 1
    assert serialized_result_len(result, tool_name="t") <= base + 10


@pytest.mark.asyncio
async def test_search_memory_store_failure_still_fails():
    feature = MemoryFeature.__new__(MemoryFeature)
    broken = MagicMock()
    broken.search_history = AsyncMock(side_effect=RuntimeError("disk on fire"))
    feature._get_conversation_store = lambda: broken

    result = await feature.search_memory(query="anything", limit=5)

    assert result.status is ToolResultStatus.ERROR
    assert "disk on fire" in result.error
