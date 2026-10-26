"""#3740: ranged reads of a text, each chunk sized to the tool-result cap.

``paged_text_result`` is the one paging envelope the ranged file readers
share. Each test measures results with ``serialized_result_len``, the
orchestrator's own measure, so "fits the cap" means the model sees the chunk
whole rather than a head-and-tail preview.
"""

from __future__ import annotations

import hashlib
import json

import pytest
from kestrel_sdk.tools.result import ToolResultStatus

from kestrel_sovereign import kestrel_agent
from kestrel_sovereign.agent.orchestrator_engine import _PRUNE_MIN_RESULT_CHARS
from kestrel_sovereign.features.base import (
    OrchestratorResultRoom,
    bind_orchestrator_result_room,
    orchestrator_result_cap,
    serialized_result_len,
)
from kestrel_sovereign.features.paged_text import (
    MAX_BATCH_CHARS,
    RESPONSE_TEXT_RESERVE,
    PageRequest,
    PageRequestError,
    page_request,
    paged_text_result,
)

TOOL = "read_test_file"
DATA = {"repo": "owner/repo", "path": "docs/big.md", "ref": "main"}
CALL = {"repo": "owner/repo", "path": "docs/big.md", "ref": "main"}


@pytest.fixture(autouse=True)
def default_cap(monkeypatch):
    monkeypatch.setattr(kestrel_agent, "MAX_TOOL_RESULT_CHARS", 8000)


def _read(text: str, offset=0, limit=None):
    return paged_text_result(
        text,
        page_request(offset, limit),
        tool_name=TOOL,
        subject="docs/big.md at main",
        data=DATA,
        call_arguments=CALL,
    )


def _fits(result) -> bool:
    return serialized_result_len(result, tool_name=TOOL) <= orchestrator_result_cap()


def _read_all(text: str, limit=None):
    """Follow ``next_offset`` from 0, as the tool description tells a model."""
    results, offset = [], 0
    while True:
        result = _read(text, offset, limit)
        assert result.status is ToolResultStatus.OK, result.error
        results.append(result)
        if result.data["next_offset"] is None:
            return results
        offset = result.data["next_offset"]


def _fifty_thousand_characters() -> str:
    """Text that grows under JSON escaping: newlines, quotes, backslashes,
    accents, CJK and characters outside the BMP (twelve escaped characters
    each)."""
    unit = 'Línea "uno" — C:\\path\\x 漢字 🦅\n\tplain ascii text follows here. '
    return (unit * (50_000 // len(unit) + 1))[:50_000]


def test_a_50k_text_read_in_chunks_reassembles_byte_identically():
    text = _fifty_thousand_characters()

    results = _read_all(text)

    assert len(results) > 1
    assert "".join(r.data["content"] for r in results).encode("utf-8") == text.encode("utf-8")
    offset = 0
    for result in results:
        data = result.data
        assert _fits(result)
        assert data["offset"] == offset
        assert data["returned_chars"] == len(data["content"]) > 0
        assert data["total_chars"] == 50_000
        assert data["content_sha256"] == hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert {k: data[k] for k in DATA} == DATA
        offset += data["returned_chars"]
        assert data["next_offset"] == (offset if offset < 50_000 else None)
        assert data["truncated"] is (offset < 50_000)
    assert offset == 50_000


def test_each_chunk_is_as_large_as_fits():
    """The default chunk is not a fixed size: one more character would pass
    the cap, so a read takes as few calls as the cap allows."""
    text = "a" * 50_000

    results = _read_all(text)

    cap = orchestrator_result_cap()
    for result in results[:-1]:
        # Each "a" serializes to one character; a chunk one longer would
        # also name offsets a digit longer at most.
        assert cap - serialized_result_len(result, tool_name=TOOL) < 4
    # Plain ASCII loses only the envelope to the cap: seven calls for 50k.
    assert len(results) == 7


def _read_batched(text: str):
    """Read from 0 as a model making every batched call in one response.

    Returns the rounds (one per response) as lists of results. Each batched
    call is checked as it is made: it fits the cap, starts where it asked to
    and holds exactly the ``batch_limit`` it asked for.
    """
    rounds = [[_read(text)]]
    while rounds[-1][-1].data["next_offset"] is not None:
        last = rounds[-1][-1].data
        if "batch_offsets" not in last:
            # No batch reads further than the next call alone.
            rounds.append([_read(text, last["next_offset"])])
            continue
        limit, offsets = last["batch_limit"], last["batch_offsets"]
        batch = [_read(text, offset, limit) for offset in offsets]
        for offset, result in zip(offsets, batch):
            assert result.status is ToolResultStatus.OK, result.error
            assert _fits(result), (offset, serialized_result_len(result, tool_name=TOOL))
            assert result.data["offset"] == offset
            assert result.data["returned_chars"] == min(limit, len(text) - offset)
        rounds.append(batch)
    return rounds


#: Texts whose characters serialize to one, two, six and twelve characters,
#: mixed, uniform, and with a dense block that only some windows cross.
BATCH_TEXTS = {
    "ascii": "a" * 50_000,
    "mixed": _fifty_thousand_characters(),
    "astral": "🦅" * 50_000,
    "bmp": "é" * 50_000,
    "escapes": '"\\\n' * 17_000,
    "astral_block": "a" * 20_000 + "🦅" * 5_000 + "b" * 25_000,
}


@pytest.mark.parametrize("name", sorted(BATCH_TEXTS))
def test_batched_calls_read_on_from_next_offset_without_a_gap(name):
    """#3740: every batch offset, called with ``limit=batch_limit``, returns
    exactly that many characters under the cap, so all of a batch's calls can
    go in one response and their chunks, joined, are the text."""
    text = BATCH_TEXTS[name]

    rounds = _read_batched(text)

    chunks = [result.data["content"] for batch in rounds for result in batch]
    assert "".join(chunks) == text
    offset = 0
    for result in (result for batch in rounds for result in batch):
        assert result.data["offset"] == offset
        offset += result.data["returned_chars"]


@pytest.mark.parametrize("name", sorted(BATCH_TEXTS))
def test_a_batch_is_bounded_and_starts_at_next_offset(name):
    text = BATCH_TEXTS[name]
    most = MAX_BATCH_CHARS // orchestrator_result_cap()
    assert most == 3

    for batch in _read_batched(text):
        # What one response's calls add to the turn stays inside the bound.
        assert sum(
            serialized_result_len(result, tool_name=TOOL) for result in batch
        ) <= MAX_BATCH_CHARS
        for result in batch:
            data = result.data
            if data["next_offset"] is None:
                assert "batch_offsets" not in data and "batch_limit" not in data
                assert "End of file" in result.confirmation
                continue
            if "batch_offsets" not in data:
                # Only when the next call alone reads the rest.
                remaining = len(text) - data["next_offset"]
                assert _read(text, data["next_offset"]).data["returned_chars"] == remaining
                continue
            offsets, limit = data["batch_offsets"], data["batch_limit"]
            assert 2 <= len(offsets) <= most
            assert offsets[0] == data["next_offset"]
            assert offsets == list(range(offsets[0], offsets[-1] + 1, limit))
            assert offsets[-1] < len(text)
            # A batch stops at the end of the text rather than past it, and
            # is never shorter than it needs to be to get there.
            assert len(offsets) == min(most, -(-(len(text) - offsets[0]) // limit))
            assert f"limit={limit}" in result.confirmation


def test_astral_text_is_read_inside_the_tool_loop_budget_when_batched():
    """Each character outside the BMP serializes to twelve, so 50,000 of them
    are 600,000 characters: about eighty chunks under the 8,000-character
    cap, more than the tool loop's 45-response default budget allows when
    each takes a response. Three to a response, they take under thirty."""
    text = BATCH_TEXTS["astral"]

    rounds = _read_batched(text)

    assert sum(len(batch) for batch in rounds) > 45
    assert len(rounds) < 30


def test_a_cap_too_large_for_two_chunks_in_a_batch_offers_none(monkeypatch):
    """A batch of one is only the next call: a cap over half of
    MAX_BATCH_CHARS leaves following next_offset as the way on."""
    monkeypatch.setattr(kestrel_agent, "MAX_TOOL_RESULT_CHARS", MAX_BATCH_CHARS // 2 + 1)
    text = "🦅" * 50_000

    result = _read(text)

    assert result.data["next_offset"] == result.data["returned_chars"]
    assert "batch_offsets" not in result.data and "batch_limit" not in result.data
    assert "for the next chunk" in result.confirmation
    assert _fits(result)


# ---------------------------------------------------------------------------
# Inside the orchestrator's tool loop: the room the turn has left
# ---------------------------------------------------------------------------


def _call_chars(offset: int, limit: int) -> int:
    """A continuation call's arguments, as the orchestrator counts them."""
    return len(json.dumps({**CALL, "offset": offset, "limit": limit}))


def test_a_chunk_fits_its_share_of_its_batchs_room():
    """Two calls share 6,000 characters of room: the orchestrator would prune
    a result over 3,000 before the model read it, so the chunk is as large as
    fits 3,000 rather than the cap."""
    room = OrchestratorResultRoom(chars=6_000, calls=2, later_chars=0)

    with bind_orchestrator_result_room(room):
        result = _read(BATCH_TEXTS["astral"])

    size = serialized_result_len(result, tool_name=TOOL)
    # One more character is twelve more, and a digit in each offset.
    assert 3_000 - 20 < size <= 3_000
    # A later response has no room for two chunks: no batch is offered.
    assert "batch_offsets" not in result.data
    assert f"offset={result.data['next_offset']}" in result.confirmation


@pytest.mark.parametrize("later", [60_000, 24_000, 15_000, 11_000])
@pytest.mark.parametrize("name", ["astral", "mixed", "ascii"])
def test_a_batch_fits_the_room_a_later_response_has(name, later):
    """#3740: a batch's calls, their results and the response's text fit what
    a later response has, so the orchestrator prunes none of its results
    before the model reads them; each still returns exactly ``batch_limit``."""
    text = BATCH_TEXTS[name]
    with bind_orchestrator_result_room(
        OrchestratorResultRoom(chars=60_000, calls=1, later_chars=later)
    ):
        first = _read(text)
    offsets, limit = first.data["batch_offsets"], first.data["batch_limit"]
    arguments = sum(_call_chars(offset, limit) for offset in offsets)

    # The batch's own room when it runs, at its smallest: the response's text
    # used its whole reserve, and this batch's results kept 200 characters
    # each once pruned (``later_chars`` already allowed for that).
    room = OrchestratorResultRoom(
        chars=later - RESPONSE_TEXT_RESERVE - arguments,
        calls=len(offsets),
        later_chars=0,
    )
    with bind_orchestrator_result_room(room):
        batch = [_read(text, offset, limit) for offset in offsets]

    for offset, result in zip(offsets, batch):
        assert result.status is ToolResultStatus.OK, result.error
        assert result.data["returned_chars"] == min(limit, len(text) - offset)
        assert "fewer than limit" not in result.confirmation
    assert sum(
        serialized_result_len(result, tool_name=TOOL) for result in batch
    ) <= room.chars


def test_batches_get_fewer_and_smaller_as_the_turn_fills():
    text = BATCH_TEXTS["astral"]

    def covered(later: int):
        with bind_orchestrator_result_room(
            OrchestratorResultRoom(chars=60_000, calls=1, later_chars=later)
        ):
            data = _read(text).data
        if "batch_offsets" not in data:
            return 0, 0
        return len(data["batch_offsets"]), len(data["batch_offsets"]) * data["batch_limit"]

    roomy, tighter, tight, full = (
        covered(later) for later in (60_000, 18_000, 12_000, 9_000)
    )

    assert roomy[0] == 3
    assert roomy[1] > tighter[1] > tight[1] > 0
    assert tight[0] == 2
    # With room for little more than one full chunk, two smaller ones read
    # less than the next call alone would: following next_offset is the way.
    assert full == (0, 0)


def test_a_batched_call_given_less_room_than_its_batch_expected_says_so():
    """A response whose own text outran the reserve leaves its batch less room
    than it was sized for. Each call returns what fits its share, and says
    that the next chunk starts at its own end, not at the next batch offset,
    rather than return a chunk the orchestrator would prune unread."""
    text = BATCH_TEXTS["astral"]
    with bind_orchestrator_result_room(
        OrchestratorResultRoom(chars=60_000, calls=1, later_chars=60_000)
    ):
        first = _read(text)
    offsets, limit = first.data["batch_offsets"], first.data["batch_limit"]

    with bind_orchestrator_result_room(
        OrchestratorResultRoom(chars=12_000, calls=3, later_chars=0)
    ):
        short = _read(text, offsets[0], limit)

    assert short.status is ToolResultStatus.OK
    returned = short.data["returned_chars"]
    assert 0 < returned < limit
    assert serialized_result_len(short, tool_name=TOOL) <= 4_000
    assert short.data["content"] == text[offsets[0]: offsets[0] + returned]
    assert short.data["next_offset"] == offsets[0] + returned
    assert f"fewer than limit={limit}" in short.confirmation
    assert f"starts at offset={offsets[0] + returned}" in short.confirmation


def test_a_turn_with_no_room_left_fails_short_enough_to_be_kept():
    """With no room for a chunk, the read says where to read on, in a result
    the orchestrator never prunes: it keeps any result of
    ``_PRUNE_MIN_RESULT_CHARS`` or fewer whole."""
    text = "a" * 10_000_000
    with bind_orchestrator_result_room(
        OrchestratorResultRoom(chars=300, calls=1, later_chars=0)
    ):
        result = _read(text, offset=9_999_999)

    assert result.status is ToolResultStatus.ERROR
    assert "no room left" in result.error
    assert result.error.endswith("from offset=9999999")
    assert not result.data
    assert (
        serialized_result_len(result, tool_name="git_show_file")
        <= _PRUNE_MIN_RESULT_CHARS
    )


def test_offset_zero_is_the_default_and_names_the_next_offset():
    text = "x" * 20_000

    result = _read(text)

    assert result.data["offset"] == 0
    assert result.data["next_offset"] == result.data["returned_chars"]
    assert result.data["content"] == text[: result.data["returned_chars"]]
    assert f"offset={result.data['next_offset']}" in result.confirmation


def test_offset_at_the_end_returns_an_empty_final_chunk():
    text = "abc" * 1000

    result = _read(text, offset=len(text))

    assert result.status is ToolResultStatus.OK
    assert result.data["content"] == ""
    assert result.data["returned_chars"] == 0
    assert result.data["total_chars"] == len(text)
    assert result.data["next_offset"] is None
    assert "End of file" in result.confirmation


def test_offset_past_the_end_fails_and_reports_the_length():
    text = "abc" * 1000

    result = _read(text, offset=len(text) + 1)

    assert result.status is ToolResultStatus.ERROR
    assert "past the end" in result.error
    assert result.data["total_chars"] == len(text)
    assert "content" not in result.data


def test_a_limit_larger_than_the_remainder_returns_the_remainder():
    text = "0123456789" * 30

    result = _read(text, offset=290, limit=500)

    assert result.data["content"] == text[290:]
    assert result.data["returned_chars"] == 10
    assert result.data["next_offset"] is None


def test_a_limit_inside_the_text_returns_exactly_that_many():
    text = "0123456789" * 30

    result = _read(text, offset=5, limit=12)

    assert result.data["content"] == text[5:17]
    assert result.data["returned_chars"] == 12
    assert result.data["next_offset"] == 17
    assert result.data["truncated"] is True


def test_a_limit_over_the_cap_is_shrunk_to_fit():
    text = "y" * 50_000

    result = _read(text, limit=40_000)

    assert _fits(result)
    assert result.data["returned_chars"] < 8000
    assert result.data["next_offset"] == result.data["returned_chars"]


def test_an_empty_text_is_one_empty_final_chunk():
    result = _read("")

    assert result.status is ToolResultStatus.OK
    assert result.data["content"] == ""
    assert result.data["total_chars"] == 0
    assert result.data["next_offset"] is None


def test_the_smallest_cap_still_moves_forward_through_astral_text(monkeypatch):
    """Every character here escapes to twelve; a 1,000-character cap still
    returns at least one per call, so following next_offset ends."""
    monkeypatch.setattr(kestrel_agent, "MAX_TOOL_RESULT_CHARS", 1000)
    text = "🦅" * 600

    results = _read_all(text)

    assert "".join(r.data["content"] for r in results) == text
    assert all(_fits(r) and r.data["returned_chars"] > 0 for r in results)


def test_identifying_fields_too_large_to_leave_room_fail_rather_than_overflow(monkeypatch):
    monkeypatch.setattr(kestrel_agent, "MAX_TOOL_RESULT_CHARS", 1000)

    result = paged_text_result(
        "text",
        PageRequest(),
        tool_name=TOOL,
        subject="x",
        data={"path": "p" * 2000},
        call_arguments={"path": "p" * 2000},
    )

    assert result.status is ToolResultStatus.ERROR
    assert "cannot fit" in result.error


@pytest.mark.parametrize(
    ("offset", "limit", "expected"),
    [
        (0, None, PageRequest(0, None)),
        (None, None, PageRequest(0, None)),
        ("120", "40", PageRequest(120, 40)),
        (" 7 ", None, PageRequest(7, None)),
        (3.0, 9.0, PageRequest(3, 9)),
    ],
)
def test_page_request_accepts_integers_in_the_forms_models_send(offset, limit, expected):
    assert page_request(offset, limit) == expected


@pytest.mark.parametrize(
    ("offset", "limit"),
    [
        (-1, None),
        (True, None),
        ("ten", None),
        (2.5, None),
        (0, 0),
        (0, -5),
        (0, False),
        (0, "all"),
    ],
)
def test_page_request_rejects_unusable_values(offset, limit):
    with pytest.raises(PageRequestError):
        page_request(offset, limit)
