"""Ranged reads of a text, each chunk sized to the tool-result cap (#3740).

The orchestrator replaces a tool result whose serialized form is over
``MAX_TOOL_RESULT_CHARS`` with a preview of its first and last characters, so
a tool that returns a whole file loses the file's middle and nothing can bring
it back. A paged read returns one chunk per call instead: the longest that
fits under the cap together with its envelope, measured the way the
orchestrator measures it.

Each chunk reports where it starts (``offset``), how many characters it holds
(``returned_chars``), the length of the whole text (``total_chars``) and where
the next chunk starts (``next_offset``, ``None`` at the end). Offsets count the
characters (code points) of the text the tool read, so the chunks of one text,
joined in order, are that text exactly. ``content_sha256`` is the digest of the
whole text: a chunk whose digest differs from the previous one's was cut from
a different version of the file.

A turn's tool loop stops after a budget of model responses, and following
``next_offset`` costs one response per chunk. Serialized, a character can
take twelve characters (one outside the BMP) or six (any other non-ASCII), so
a heavily escaped text needs many chunks: 50,000 such characters take about
eighty at the default cap, more than the loop allows. A chunk that is not the
last therefore also offers a batch, when the cap leaves room for one (see
:data:`MAX_BATCH_CHARS`): ``batch_offsets``, starting at
``next_offset``, and ``batch_limit``. A call at each of those offsets with
``limit=batch_limit`` returns exactly ``batch_limit`` characters (the rest of
the text, at its end) and fits the cap, so all of them can be made in one
response, and their chunks, joined in order, continue the text from
``next_offset`` without a gap.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from itertools import accumulate
from typing import Any, Mapping, Optional

from kestrel_sdk.tools.result import ToolResult

from kestrel_sovereign.features.base import (
    largest_fitting_result,
    orchestrator_result_cap,
    serialized_result_len,
)

_INTEGER_TEXT = re.compile(r"\s*[+-]?\d+\s*")

#: The most characters a batch's results add to the turn in one model
#: response: three results at the default 8,000-character cap. The
#: orchestrator prunes a turn's oldest tool results once its messages near the
#: model's context window, which it takes as 32,768 tokens (about 92,000
#: characters) when it cannot resolve the model's own; a larger batch, with
#: the system prompt, could lose its own first chunks to that pruning before
#: the model reads them. Three to a response, 50,000 characters outside the
#: BMP (600,000 once serialized) take under thirty responses, inside the tool
#: loop's default budget.
MAX_BATCH_CHARS = 24_000


class PageRequestError(ValueError):
    """A paged read's ``offset`` or ``limit`` is not a usable integer."""


@dataclass(frozen=True)
class PageRequest:
    """Where a paged read starts, and the most characters it may return.

    ``limit`` of ``None`` means as many as fit under the tool-result cap; a
    given ``limit`` is shrunk further to fit.
    """

    offset: int = 0
    limit: Optional[int] = None


@dataclass(frozen=True)
class _Batch:
    """Chunks of ``limit`` characters at ``offsets``, each known to fit."""

    limit: int
    offsets: tuple[int, ...]


def page_request(offset: Any = 0, limit: Any = None) -> PageRequest:
    """Validate a tool call's ``offset`` and ``limit``.

    Model-supplied arguments arrive as integers, integer strings (from a
    ``!command``) or integral floats, and ``None`` means not given; anything
    else, a boolean, a negative ``offset`` or a ``limit`` below 1 raises
    :class:`PageRequestError`.
    """
    start = 0 if offset is None else _page_int(offset, "offset", lo=0)
    wanted = None if limit is None else _page_int(limit, "limit", lo=1)
    return PageRequest(offset=start, limit=wanted)


def _page_int(value: Any, name: str, *, lo: int) -> int:
    if isinstance(value, bool):
        raise PageRequestError(f"{name} must be an integer, got {value!r}")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, float) and value.is_integer():
        parsed = int(value)
    elif isinstance(value, str) and _INTEGER_TEXT.fullmatch(value):
        parsed = int(value)
    else:
        raise PageRequestError(f"{name} must be an integer, got {value!r}")
    if parsed < lo:
        raise PageRequestError(f"{name} must be at least {lo}, got {parsed}")
    return parsed


@lru_cache(maxsize=4096)
def _escaped_width(char: str) -> int:
    """How many characters ``char`` takes in the orchestrator's serialization.

    The orchestrator serializes a result with ``json.dumps`` defaults, which
    escape every character outside printable ASCII: ``é`` takes six and
    ``🦅`` twelve. Escaping is per character, so a chunk's serialized length
    is the sum over its characters.
    """
    return len(json.dumps(char)) - 2


def paged_text_result(
    text: str,
    page: PageRequest,
    *,
    tool_name: str,
    subject: str,
    data: Mapping[str, Any],
) -> ToolResult:
    """One chunk of ``text``, as a result that fits the tool-result cap.

    Args:
        text: The whole text being read.
        page: Where the chunk starts and its most characters.
        tool_name: The calling tool, named in the "call again" instruction and
            counted in the result's measured size.
        subject: What the text is (``"README.md at main"``), for messages.
        data: Fields identifying the text (repo, path, ref) that every result
            carries ahead of the position fields and the chunk. They must be
            the same on every call for the same text: a batch is sized
            against them.

    An ``offset`` equal to the text's length returns an empty final chunk; an
    ``offset`` past it fails and reports ``total_chars``. While characters
    remain, a chunk holds at least one, so following ``next_offset`` always
    reaches the end.
    """
    total = len(text)
    digest = hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()
    start = page.offset
    if start > total:
        return ToolResult.failed(
            f"offset {start} is past the end of {subject} ({total} characters)",
            data={**data, "offset": start, "total_chars": total, "content_sha256": digest},
        )
    requested_end = total if page.limit is None else min(start + page.limit, total)

    def result(
        first: int,
        end: int,
        chunk: str,
        batch: Optional[_Batch],
        *,
        has_more: bool,
        returned: Optional[int] = None,
    ) -> ToolResult:
        confirmation = f"Read {subject}: characters {first}–{end} of {total}."
        fields: dict[str, Any] = {
            **data,
            "offset": first,
            "returned_chars": len(chunk) if returned is None else returned,
            "total_chars": total,
            "next_offset": end if has_more else None,
            "truncated": has_more,
        }
        if not has_more:
            confirmation += " End of file."
        elif batch is None:
            confirmation += (
                f" More remains; call {tool_name} again with offset={end} "
                f"for the next chunk."
            )
        else:
            confirmation += (
                f" More remains; call {tool_name} again with offset={end}, or "
                f"read on faster: call it once for each offset in batch_offsets "
                f"with limit={batch.limit}, all in one response."
            )
            fields["batch_limit"] = batch.limit
            fields["batch_offsets"] = list(batch.offsets)
        fields["content_sha256"] = digest
        fields["content"] = chunk
        return ToolResult.ok(confirmation=confirmation, data=fields)

    cap = orchestrator_result_cap()
    # Each of a batch's results is at most the cap; a batch of one chunk is
    # only the next call, so a cap too large for two offers none.
    batch_chunks = MAX_BATCH_CHARS // cap
    if batch_chunks < 2:
        batch_chunks = 0
    # The largest envelope any chunk of this text can have: every position it
    # names is at most ``total``, so none has more digits than ``total``, and
    # a batch names at most ``batch_chunks`` offsets. A chunk whose escaped
    # characters fit in what the cap leaves fits whatever its position.
    widest = _Batch(limit=total, offsets=(total,) * batch_chunks) if batch_chunks else None
    envelope = max(
        serialized_result_len(
            result(total, total, "", widest, has_more=True, returned=total),
            tool_name=tool_name,
        ),
        serialized_result_len(
            result(total, total, "", None, has_more=False, returned=total),
            tool_name=tool_name,
        ),
    )
    room = cap - envelope

    # Serialized, every character takes at least one character, so neither a
    # chunk nor a batched one is longer than the cap: the escaped widths of
    # the characters a chunk and the batch after it can cover are all a batch
    # needs, without serializing the rest of a large file.
    reach = min(total, start + cap * (batch_chunks + 1))
    widths = [0, *accumulate(map(_escaped_width, text[start:reach]))]

    def escaped_len(first: int, end: int) -> int:
        return widths[end - start] - widths[first - start]

    def batch_after(end: int) -> Optional[_Batch]:
        """The batch from ``end`` with the largest chunks that all fit."""
        best: Optional[_Batch] = None
        lo, hi = 1, min(total - end, room)
        while lo <= hi:
            size = (lo + hi) // 2
            offsets = tuple(range(end, min(total, end + batch_chunks * size), size))
            if all(
                escaped_len(offset, min(offset + size, total)) <= room
                for offset in offsets
            ):
                best = _Batch(limit=size, offsets=offsets)
                lo = size + 1
            else:
                hi = size - 1
        return best

    def build(end: int) -> ToolResult:
        has_more = end < total
        return result(
            start,
            end,
            text[start:end],
            batch_after(end) if has_more and batch_chunks else None,
            has_more=has_more,
        )

    shortest = start + 1 if start < requested_end else start
    longest = min(requested_end, start + cap)
    # A batched call asks for a window whose escaped characters fit in
    # ``room``. Every chunk it could return is a prefix of that window under
    # an envelope no larger than ``envelope``, so every candidate fits and
    # the search returns the whole ``batch_limit``.
    fit = largest_fitting_result(
        build, shortest, longest, cap=cap, tool_name=tool_name
    )
    if fit is None:
        return ToolResult.failed(
            f"{tool_name} cannot fit the next character of {subject} under the "
            f"{cap:,}-character tool-result cap",
            data={**data, "offset": start, "total_chars": total},
        )
    return fit[1]
