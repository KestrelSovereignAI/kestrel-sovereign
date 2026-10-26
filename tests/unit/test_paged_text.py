"""#3740: ranged reads of a text, each chunk sized to the tool-result cap.

``paged_text_result`` is the one paging envelope the ranged file readers
share. Each test measures results with ``serialized_result_len``, the
orchestrator's own measure, so "fits the cap" means the model sees the chunk
whole rather than a head-and-tail preview.
"""

from __future__ import annotations

import hashlib

import pytest
from kestrel_sdk.tools.result import ToolResultStatus

from kestrel_sovereign import kestrel_agent
from kestrel_sovereign.features.base import orchestrator_result_cap, serialized_result_len
from kestrel_sovereign.features.paged_text import (
    MAX_BATCH_CHARS,
    PageRequest,
    PageRequestError,
    page_request,
    paged_text_result,
)

TOOL = "read_test_file"
DATA = {"repo": "owner/repo", "path": "docs/big.md", "ref": "main"}


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
            offsets, limit = data["batch_offsets"], data["batch_limit"]
            assert 1 <= len(offsets) <= most
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
