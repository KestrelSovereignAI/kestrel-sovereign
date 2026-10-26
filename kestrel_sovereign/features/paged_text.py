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

The cap is not the only way a chunk can be lost. Before each model call the
orchestrator prunes the turn to the model's context window, replacing tool
results with a stub oldest first, and every call's arguments and every pruned
result's stub stay in the turn: eighty calls leave about 20,000 characters of
them. Inside the orchestrator's tool loop a read therefore also sizes to the
room the turn has left (:func:`~kestrel_sovereign.features.base.orchestrator_result_room`):
a chunk to its share of its own batch's room, and the batch it offers so that
its calls' arguments and results, beside the text of the response making
them, fit what a later response has. As the turn fills, batches get fewer and
smaller rather than being pruned before the model reads them.
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
    orchestrator_result_room,
    serialized_result_len,
)

_INTEGER_TEXT = re.compile(r"\s*[+-]?\d+\s*")

#: The most characters a batch's results add to the turn in one model
#: response: three results at the default 8,000-character cap. Three to a
#: response, 50,000 characters outside the BMP (600,000 once serialized) take
#: about thirty responses, inside the tool loop's default budget. Inside the
#: orchestrator's loop a batch is smaller still when the turn has less room
#: left than this.
MAX_BATCH_CHARS = 24_000

#: Characters a batch sized to the turn's room leaves for the text of the
#: response that makes its calls: the orchestrator counts that text beside
#: the calls' arguments and results.
RESPONSE_TEXT_RESERVE = 1_000


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
    call_arguments: Mapping[str, Any],
) -> ToolResult:
    """One chunk of ``text``, as a result that reaches the model whole.

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
        call_arguments: The calling tool's arguments other than ``offset``
            and ``limit``, which a call reading on repeats. The orchestrator
            keeps every call's arguments in the turn, so a batch is sized
            against them too.

    An ``offset`` equal to the text's length returns an empty final chunk; an
    ``offset`` past it fails and reports ``total_chars``. While characters
    remain and the turn has room for one, a chunk holds at least one, so
    following ``next_offset`` always reaches the end.
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
        if page.limit is not None and end < requested_end:
            # Fewer than the limit asked for: a model reading a batch must not
            # go on from the batch's next offset, past what this one missed.
            confirmation += (
                f" That is fewer than limit={page.limit}: the next chunk "
                f"starts at offset={end}, not at a later batch offset."
            )
        if not has_more:
            confirmation += " End of file."
        elif batch is None:
            confirmation += (
                f" More remains; call {tool_name} again with offset={end} "
                f"for the next chunk."
            )
        else:
            confirmation += (
                f" More remains: call {tool_name} once for each offset in "
                f"batch_offsets with limit={batch.limit}, all in one response "
                f"(or with offset={end} for the next chunk alone)."
            )
            fields["batch_limit"] = batch.limit
            fields["batch_offsets"] = list(batch.offsets)
        fields["content_sha256"] = digest
        fields["content"] = chunk
        return ToolResult.ok(confirmation=confirmation, data=fields)

    cap = orchestrator_result_cap()
    room = orchestrator_result_room()
    # This result reaches the model whole when it fits the cap and, inside the
    # orchestrator's tool loop, its share of the room its batch has.
    own_cap = cap if room is None else min(cap, room.share())
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
    # A call reading on, as the orchestrator counts its arguments: every
    # position it names is at most ``total``.
    call_chars = len(json.dumps({**call_arguments, "offset": total, "limit": total}))
    later = None if room is None else room.later_chars - RESPONSE_TEXT_RESERVE

    def chunk_room(calls: int) -> int:
        """Escaped characters each of a batch of ``calls`` chunks may hold.

        Each result fits the cap, and in the orchestrator's loop the batch's
        calls and results fit what a later response has: equal shares, so
        each result fits its own share when the batch runs.
        """
        per_result = cap
        if later is not None:
            per_result = min(cap, (later - calls * call_chars) // calls)
        return per_result - envelope

    # Serialized, every character takes at least one character, so neither a
    # chunk nor a batched one is longer than the cap: the escaped widths of
    # the characters a chunk and the batch after it can cover are all a batch
    # needs, without serializing the rest of a large file.
    reach = min(total, start + cap * (batch_chunks + 1))
    widths = [0, *accumulate(map(_escaped_width, text[start:reach]))]

    def escaped_len(first: int, end: int) -> int:
        return widths[end - start] - widths[first - start]

    def batch_of(end: int, calls: int) -> Optional[_Batch]:
        """The batch of at most ``calls`` chunks from ``end`` with the
        largest chunks that all fit."""
        fits = chunk_room(calls)
        best: Optional[_Batch] = None
        lo, hi = 1, min(total - end, fits)
        while lo <= hi:
            size = (lo + hi) // 2
            offsets = tuple(range(end, min(total, end + calls * size), size))
            if all(
                escaped_len(offset, min(offset + size, total)) <= fits
                for offset in offsets
            ):
                best = _Batch(limit=size, offsets=offsets)
                lo = size + 1
            else:
                hi = size - 1
        return best

    def covered(batch: _Batch) -> int:
        return min(total, batch.offsets[-1] + batch.limit) - batch.offsets[0]

    def batch_after(end: int) -> Optional[_Batch]:
        """The batch from ``end`` that reads furthest; of two that read as
        far, the one with fewer calls. None when the next call alone reads as
        far, as it does when the rest of the text fits one chunk or the turn
        has room for little more than one: a batch of one chunk is only the
        next call."""
        best: Optional[_Batch] = None
        for calls in range(batch_chunks, 0, -1):
            batch = batch_of(end, calls)
            if batch is not None and (best is None or covered(batch) >= covered(best)):
                best = batch
        return best if best is not None and len(best.offsets) > 1 else None

    def build(end: int) -> ToolResult:
        has_more = end < total
        return result(
            start,
            end,
            text[start:end],
            batch_after(end) if has_more and batch_chunks else None,
            has_more=has_more,
        )

    def fits(candidate: ToolResult) -> bool:
        return serialized_result_len(candidate, tool_name=tool_name) <= own_cap

    # A batched call asks for a window whose escaped characters fit its share
    # under an envelope no larger than ``envelope``, so the whole window fits
    # and the call returns exactly ``batch_limit``. Tried first, because only
    # a shorter chunk carries the "fewer than limit" note, so a search could
    # miss the whole window by probing one character short of it.
    if requested_end - start <= own_cap:
        whole = build(requested_end)
        if fits(whole):
            return whole
    shortest = start + 1 if start < requested_end else start
    longest = min(requested_end - 1, start + own_cap)
    fit = largest_fitting_result(
        build, shortest, longest, cap=own_cap, tool_name=tool_name
    )
    if fit is not None:
        return fit[1]
    if own_cap < cap:
        # Without the identifying fields: the orchestrator never prunes a
        # result this short, so the model sees where to read on even in a
        # turn with no room for anything longer.
        return ToolResult.failed(
            f"no room left in this turn for more of the text; read on in a "
            f"later turn from offset={start}"
        )
    return ToolResult.failed(
        f"{tool_name} cannot fit the next character of {subject} under the "
        f"{cap:,}-character tool-result cap",
        data={**data, "offset": start, "total_chars": total},
    )
