"""Compact, size-bounded conversation-search results (#3784).

``search_memory`` used to return each hit's whole message and its whole
metadata envelope: every lifecycle field, and concept URIs that each repeat the
agent's full DID. One stored message can be a megabyte (setup turns with
inlined documents, A2A wakes quoting whole questions), so five hits reached
431,601 characters on a live agent. The orchestrator then kept only the first
2,000 and last 500 of them, and a subagent's prompt received all of them.

A hit is now a pointer plus an excerpt: ``message_id``, ``session_id``,
``role``, ``timestamp``, ``score`` and a snippet around the match, with the
character range the snippet came from. The range indexes the same text
``read_message_by_id`` returns, so ``snippet_start`` is an offset to read on
from. Concepts are bare names, and no other metadata is carried.

The whole result is then fitted under the orchestrator's per-result cap,
measured the way the orchestrator measures it, so it reaches the model whole
on every path. Snippets shrink first; then the lowest-ranked hits are dropped,
and the result says how many and why.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from kestrel_sdk.tools.result import ToolResult
from kestrel_sovereign.features.base import largest_fitting_result
from kestrel_sovereign.storage.async_conversation_store import (
    SearchTerms,
    search_match_projection,
)
from kestrel_sovereign.storage.session_id_column import SESSION_ID_MAX_LENGTH

#: The widest excerpt a hit carries, in characters of message text.
DEFAULT_SNIPPET_CHARS = 400
#: The narrowest excerpt before hits are dropped instead.
MIN_SNIPPET_CHARS = 80
#: Bare concept names kept per hit.
MAX_HIT_CONCEPTS = 5
#: How much of a caller's own argument (query, session id) is echoed back.
ECHO_CHARS = 120
#: How much of a STORED session id a hit or a read carries. Every id the
#: session column can hold passes whole; a longer legacy value is clipped,
#: visibly, rather than copied into every hit and every chunk (#3784).
STORED_SESSION_ID_CHARS = SESSION_ID_MAX_LENGTH

ELLIPSIS = "…"

#: Builds the tool's result from compact hits and how many were left out.
RenderHits = Callable[[List[Dict[str, Any]], int], ToolResult]


@dataclass(frozen=True)
class SearchHit:
    """One matched message, located once, excerpted at any width."""

    message_id: Optional[int]
    session_id: Optional[str]
    role: str
    timestamp: Optional[str]
    score: float
    match: Optional[str]
    text: str
    match_start: int
    match_end: int
    concepts: Tuple[str, ...]

    def compact(self, width: int) -> Dict[str, Any]:
        """The hit as returned to the model, its snippet at most ``width``."""
        total = len(self.text)
        start, end = snippet_bounds(total, self.match_start, self.match_end, width)
        body = " ".join(self.text[start:end].split())
        entry: Dict[str, Any] = {
            "message_id": self.message_id,
            "session_id": self.session_id,
            "role": self.role,
            "timestamp": self.timestamp,
            "score": round(self.score, 3),
            "match": self.match,
            "snippet": (
                (ELLIPSIS if start > 0 else "")
                + body
                + (ELLIPSIS if end < total else "")
            ),
            "snippet_start": start,
            "snippet_end": end,
            "content_chars": total,
        }
        if self.concepts:
            entry["concepts"] = list(self.concepts)
        return entry


def snippet_bounds(
    total: int, match_start: int, match_end: int, width: int
) -> Tuple[int, int]:
    """The ``[start, end)`` window of at most ``width`` characters to excerpt.

    The match begins a third of the way in, so more of what follows it is
    kept than of what precedes it; the window slides back from the end of the
    text rather than coming up short.
    """
    if total <= width:
        return 0, total
    start = max(0, min(match_start - width // 3, total - width))
    return start, start + width


def locate_match(text: str, query: str, tokens: Sequence[str]) -> Tuple[int, int]:
    """Where in ``text`` the hit matched: the query as a phrase, else a term.

    Looked for in the wrapper-free projection search admits hits on
    (:func:`search_match_projection`), then mapped back to offsets in
    ``text``: a legacy row can quote an older message inside a bare RELEVANT
    MEMORIES block, and searching ``text`` itself would excerpt the quote
    rather than what was said. Case-insensitive, like the search. A phrase
    match wins; otherwise the earliest of the query's search terms. ``(0, 0)``
    when neither occurs outside the wrappers (an empty query matches every
    row).
    """
    projection = search_match_projection(text)
    haystack = projection.text
    lowered = haystack.lower()
    # ``str.lower`` keeps positions for almost every text; where it does not
    # (a character that lowercases to two), fall back to a case-insensitive
    # regex, which reports positions in the projection itself.
    same_positions = len(lowered) == len(haystack)

    def find(needle: str) -> Optional[Tuple[int, int]]:
        if same_positions:
            folded = needle.lower()
            index = lowered.find(folded)
            return (index, index + len(folded)) if index >= 0 else None
        found = re.search(re.escape(needle), haystack, re.IGNORECASE)
        return found.span() if found else None

    phrase = query.strip()
    span = find(phrase) if phrase else None
    if span is None:
        for token in tokens:
            found = find(token)
            if found is not None and (span is None or found[0] < span[0]):
                span = found
    if span is None:
        return 0, 0
    return projection.source_span(*span)


def bare_concept_name(value: str, agent_id: Optional[str]) -> str:
    """``concept:<agent did>:<name>`` collapsed to ``<name>``."""
    if agent_id:
        prefix = f"concept:{agent_id}:"
        if value.startswith(prefix):
            return value[len(prefix):]
    if value.startswith("concept:"):
        return value.rsplit(":", 1)[-1]
    return value


def bare_concepts(
    metadata: Optional[Mapping[str, Any]], agent_id: Optional[str]
) -> Tuple[str, ...]:
    """Up to :data:`MAX_HIT_CONCEPTS` distinct bare concept names of a row."""
    raw = metadata.get("extracted_concepts") if isinstance(metadata, Mapping) else None
    if not isinstance(raw, (list, tuple)):
        return ()
    names: List[str] = []
    for value in raw:
        if not isinstance(value, str):
            continue
        name = bare_concept_name(value, agent_id).strip()
        if name and name not in names:
            names.append(name)
            if len(names) >= MAX_HIT_CONCEPTS:
                break
    return tuple(names)


def timestamp_text(value: Any) -> Optional[str]:
    """A stored ``created_at`` as text: ISO for a datetime, as stored otherwise."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def message_text(row: Mapping[str, Any]) -> str:
    """A stored row's content as the text a hit and a read both index."""
    content = row.get("content")
    if isinstance(content, str):
        return content
    return "" if content is None else str(content)


def search_hits(
    rows: Sequence[Mapping[str, Any]],
    query: str,
    *,
    agent_id: Optional[str],
) -> List[SearchHit]:
    """Locate each ``search_history`` row's match in its returned text.

    ``rows`` carry the content the model would read — for user rows, already
    stripped of sent-form wrappers — so a hit's offsets agree with
    ``read_message_by_id``.
    """
    terms = SearchTerms.compile(query)
    tokens = terms.tokens if terms is not None else ()
    hits: List[SearchHit] = []
    for row in rows:
        text = message_text(row)
        metadata = row.get("metadata")
        session_id = metadata.get("session_id") if isinstance(metadata, Mapping) else None
        match_start, match_end = locate_match(text, query, tokens)
        score = row.get("score")
        hits.append(SearchHit(
            message_id=row.get("id"),
            session_id=stored_session_id(session_id),
            role=str(row.get("role") or ""),
            timestamp=timestamp_text(row.get("created_at")),
            score=float(score) if isinstance(score, (int, float)) else 0.0,
            match=row.get("match"),
            text=text,
            match_start=match_start,
            match_end=match_end,
            concepts=bare_concepts(metadata, agent_id),
        ))
    return hits


def bounded_echo(value: Any, limit: int = ECHO_CHARS) -> Optional[str]:
    """``value`` as text, at most ``limit`` characters plus a visible marker.

    The query is the caller's own text; a 26,000-character message passed as
    the query would otherwise fill the result by itself.
    """
    if value is None:
        return None
    text = str(value)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}{ELLIPSIS} [{len(text) - limit:,} chars omitted]"


def stored_session_id(value: Any) -> Optional[str]:
    """A row's stored session id as a hit or a read carries it.

    Stored ids are not bounded the way minted ones are, and one is copied
    into every hit and every chunk of a read, so a pathological value would
    otherwise crowd out the content it labels.
    """
    return bounded_echo(value, STORED_SESSION_ID_CHARS)


def omission_notice(omitted: int, matched: int, cap: int) -> str:
    """Why hits are missing from a result, and how to see them."""
    return (
        f"{omitted} of {matched} matching message(s) omitted to keep this "
        f"result under {cap:,} characters; narrow the query, pass session_id, "
        f"or lower limit to see them"
    )


def fit_search_result(
    hits: Sequence[SearchHit],
    render: RenderHits,
    *,
    cap: int,
    tool_name: str,
) -> ToolResult:
    """The fullest result ``render`` builds from ``hits`` within ``cap``.

    Every hit is kept with the widest snippet that fits; if not even
    :data:`MIN_SNIPPET_CHARS` fits for all of them, the lowest-ranked hits are
    dropped and ``render`` is told how many. A result over the cap is never
    returned: if not even the empty one fits, the call fails, visibly.
    """
    count = len(hits)

    def build(width: int, keep: int) -> ToolResult:
        return render([hit.compact(width) for hit in hits[:keep]], count - keep)

    widest = largest_fitting_result(
        lambda width: build(width, count),
        MIN_SNIPPET_CHARS,
        DEFAULT_SNIPPET_CHARS,
        cap=cap,
        tool_name=tool_name,
    )
    if widest is not None:
        return widest[1]
    most = largest_fitting_result(
        lambda keep: build(MIN_SNIPPET_CHARS, keep),
        0,
        count - 1,
        cap=cap,
        tool_name=tool_name,
    )
    if most is not None:
        return most[1]
    return ToolResult.failed(
        f"{tool_name} cannot fit even an empty result under the {cap:,}-character "
        f"tool-result cap"
    )
