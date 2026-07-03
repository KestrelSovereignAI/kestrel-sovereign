"""Pagination for read_attachment (#2134, F086).

A large document must not be silently truncated to an unreadable preview:
each read returns a chunk sized so the *serialized* ToolResult fits under the
orchestrator's ``MAX_TOOL_RESULT_CHARS`` cap, plus offset/total metadata so the
model can request the next chunk, and a confirmation that states range + total.

The cap is applied downstream to ``len(json.dumps(_serialize_tool_result(res)))``
(orchestrator_engine.py), so these tests assert against that *serialized* shape
rather than the raw character count — JSON-escaping of quotes, backslashes, and
non-ASCII/emoji expands content well past its ``len()``.
"""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock


def _history_with(att):
    return [{"role": "user", "metadata": {"attachments": [att]}}]


def _make(att, *, bytes_):
    from kestrel_sovereign.features.attachments.feature import AttachmentsFeature
    storage = MagicMock()
    storage.get_conversation_history = AsyncMock(return_value=_history_with(att))
    storage.retrieve_file = AsyncMock(return_value=bytes_)
    agent = MagicMock()
    agent.agent_id = "did:test"
    agent._active_session_id = None
    agent.storage = storage
    feat = AttachmentsFeature(agent)
    feat.storage = storage
    return feat, storage


def _doc_att(h, name="big.txt", mime="text/plain"):
    return {"hash": h, "kind": "document", "mime": mime, "name": name}


def _serialized_len(res):
    from kestrel_sovereign.features.base import _serialize_tool_result
    return len(json.dumps(_serialize_tool_result(res)))


async def _read_all(feat, h):
    """Walk the pagination chain and return (reassembled_text, chunk_count),
    asserting every chunk's serialized result fits under the orchestrator cap."""
    from kestrel_sovereign.kestrel_agent import MAX_TOOL_RESULT_CHARS
    parts = []
    offset = 0
    count = 0
    while True:
        res = await feat.read_attachment(h, offset=offset)
        assert res.status == "ok"
        assert _serialized_len(res) <= MAX_TOOL_RESULT_CHARS, (
            f"chunk at offset {offset} serialized to {_serialized_len(res)} "
            f"chars, over the {MAX_TOOL_RESULT_CHARS} cap"
        )
        parts.append(res.data["content"])
        count += 1
        nxt = res.data["next_offset"]
        if nxt is None:
            assert res.data["truncated"] is False
            break
        assert nxt > offset, "pagination must make forward progress"
        offset = nxt
        assert count < 100_000, "pagination did not terminate"
    return "".join(parts), count


# --- serialized chunk fits under the cap, across escaping-heavy content ------

@pytest.mark.parametrize(
    "unit",
    [
        "x",        # plain ASCII — cheapest to serialize
        '"',        # double-quote — escapes to \" (2x)
        "\\",       # backslash — escapes to \\ (2x)
        "\n",       # newline — escapes to \n (2x)
        "é",        # 2-byte UTF-8; json.dumps default escapes to é (6x)
        "😀",       # emoji; surrogate pair 😀 (12x)
    ],
)
@pytest.mark.asyncio
async def test_every_chunk_fits_under_cap_regardless_of_escaping(unit):
    from kestrel_sovereign.kestrel_agent import MAX_TOOL_RESULT_CHARS

    h = "a" * 64
    # A document several caps long, built entirely from the worst-case unit so
    # naive len()-based sizing would blow the serialized cap.
    doc = unit * (MAX_TOOL_RESULT_CHARS * 3)
    feat, _ = _make(_doc_att(h), bytes_=doc.encode("utf-8"))

    reassembled, chunks = await _read_all(feat, h)
    # No silent loss: the concatenated chunks reconstruct the whole document.
    assert reassembled == doc
    # A worst-case-escaping doc must actually paginate (more than one chunk).
    if unit != "x":
        assert chunks > 1


# --- range + total metadata, no silent truncation ---------------------------

@pytest.mark.asyncio
async def test_first_chunk_reports_range_and_total():
    from kestrel_sovereign.kestrel_agent import MAX_TOOL_RESULT_CHARS

    h = "b" * 64
    total = MAX_TOOL_RESULT_CHARS * 2 + 50
    feat, _ = _make(_doc_att(h), bytes_=b"x" * total)
    res = await feat.read_attachment(h)
    assert res.status == "ok"
    assert res.data["offset"] == 0
    assert res.data["total"] == total
    assert res.data["truncated"] is True
    assert res.data["next_offset"] == res.data["length"]
    assert len(res.data["content"]) == res.data["length"]
    # Confirmation must state the range read AND the total (no silent trunc).
    assert str(total) in res.confirmation
    assert str(res.data["length"]) in res.confirmation


@pytest.mark.asyncio
async def test_next_offset_returns_following_chunk():
    h = "c" * 64
    # Two ASCII chunks: first fills a chunk, tail is the remainder.
    from kestrel_sovereign.kestrel_agent import MAX_TOOL_RESULT_CHARS
    body = ("A" * (MAX_TOOL_RESULT_CHARS * 2)) + ("B" * 500)
    feat, _ = _make(_doc_att(h), bytes_=body.encode())

    first = await feat.read_attachment(h)
    assert set(first.data["content"]) == {"A"}
    nxt = first.data["next_offset"]
    assert nxt == first.data["length"]

    second = await feat.read_attachment(h, offset=nxt)
    assert second.status == "ok"
    assert second.data["offset"] == nxt
    # The chain must ultimately reach the "B" tail with no loss.
    reassembled, _ = await _read_all(feat, h)
    assert reassembled == body


# --- explicit length is honored as an upper bound ---------------------------

@pytest.mark.asyncio
async def test_small_length_is_honored():
    from kestrel_sovereign.kestrel_agent import MAX_TOOL_RESULT_CHARS
    h = "d" * 64
    total = MAX_TOOL_RESULT_CHARS * 2
    feat, _ = _make(_doc_att(h), bytes_=b"x" * total)
    res = await feat.read_attachment(h, offset=0, length=100)
    assert res.data["length"] == 100
    assert res.data["content"] == "x" * 100
    assert res.data["next_offset"] == 100


@pytest.mark.asyncio
async def test_oversized_length_shrinks_to_fit_the_cap():
    from kestrel_sovereign.kestrel_agent import MAX_TOOL_RESULT_CHARS
    h = "d" * 64
    total = MAX_TOOL_RESULT_CHARS * 4
    feat, _ = _make(_doc_att(h), bytes_=b"x" * total)
    res = await feat.read_attachment(h, offset=0, length=10_000_000)
    # Requested far more than fits — the returned chunk is bounded by the cap.
    assert _serialized_len(res) <= MAX_TOOL_RESULT_CHARS
    assert res.data["length"] < total


# --- small document: single complete read -----------------------------------

@pytest.mark.asyncio
async def test_small_document_reads_fully_with_no_next():
    h = "e" * 64
    feat, _ = _make(_doc_att(h), bytes_=b"hello world")
    res = await feat.read_attachment(h)
    assert res.data["content"] == "hello world"
    assert res.data["offset"] == 0
    assert res.data["total"] == 11
    assert res.data["next_offset"] is None
    assert res.data["truncated"] is False
    assert "11" in res.confirmation


@pytest.mark.asyncio
async def test_offset_beyond_end_returns_empty_and_no_next():
    h = "f" * 64
    feat, _ = _make(_doc_att(h), bytes_=b"short")
    res = await feat.read_attachment(h, offset=1000)
    assert res.status == "ok"
    assert res.data["content"] == ""
    assert res.data["next_offset"] is None
    assert res.data["total"] == 5
