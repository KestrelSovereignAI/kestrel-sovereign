"""
Wave 5E — server-side ``strip_revise_sentinels`` helper.

Voice/TTS, bridge, and any other downstream consumer of the agent's
streaming chunks would otherwise leak the in-band sentinel as
literal ``\\x1eKESTREL:REVISE:...`` text. The helper strips complete
sentinels before re-publishing chunks. Codex P1 of #1089 caught the
TTS-reading-aloud regression.
"""
from kestrel_sovereign.agent.streaming import (
    REVISE_SENTINEL_PREFIX,
    REVISE_SENTINEL_SUFFIX,
    strip_revise_sentinels,
    _build_revise_sentinel,
)
from kestrel_sdk.llm import ToolCallStarted


def test_no_sentinel_passthrough():
    """A chunk without a sentinel should be returned unchanged —
    fast-path the common case (every non-revise chunk on the stream)."""
    assert strip_revise_sentinels("Hello world") == "Hello world"
    assert strip_revise_sentinels("") == ""
    assert strip_revise_sentinels("\x1e but not a sentinel") == "\x1e but not a sentinel"


def test_single_sentinel_stripped():
    sentinel = _build_revise_sentinel(
        ToolCallStarted(index=0, id="tc1", name="x"),
    )
    chunk = f"pre-tool {sentinel}post-tool"
    out = strip_revise_sentinels(chunk)
    assert out == "pre-tool post-tool"
    assert REVISE_SENTINEL_PREFIX not in out


def test_multiple_sentinels_in_one_chunk():
    """Two ToolCallStarted markers might land in a single yield —
    strip both."""
    a = _build_revise_sentinel(ToolCallStarted(index=0, id="tc1", name="a"))
    b = _build_revise_sentinel(ToolCallStarted(index=1, id="tc2", name="b"))
    chunk = f"pre {a} mid {b} post"
    out = strip_revise_sentinels(chunk)
    assert out == "pre  mid  post"


def test_split_sentinel_falls_through_at_helper_layer():
    """The server-side helper handles complete sentinels only.
    A split sentinel (no closing \\x1e in the same chunk) is the
    caller's responsibility to buffer — but the helper must not
    crash and must not corrupt the chunk before the prefix.

    In practice this path is unreachable on the server side because
    the agent emits the sentinel as a single Python ``yield``; only
    the client side has to worry about ReadableStream re-chunking.
    """
    chunk = f"pre {REVISE_SENTINEL_PREFIX}{{\"index\":0,"  # no closing \x1e
    out = strip_revise_sentinels(chunk)
    # Pre-prefix slice preserved; everything from prefix on dropped.
    assert out == "pre "
    assert REVISE_SENTINEL_PREFIX not in out


def test_chunk_that_is_only_a_sentinel():
    """The dedicated yield in agent/streaming.py emits the sentinel
    as a chunk of its own. The helper must produce empty string
    rather than dropping the chunk on the floor (caller decides
    whether to forward an empty chunk or skip it)."""
    sentinel = _build_revise_sentinel(
        ToolCallStarted(index=0, id="tc1", name="x"),
    )
    out = strip_revise_sentinels(sentinel)
    assert out == ""
