"""Per-turn continuation snapshots for stateless Responses API calls (#4198).

A reasoning model's encrypted reasoning is valid only after the exact history
the provider saw before it. OpenAI documents that changing that history — a
rewritten or dropped earlier item, changed ``tools``, ``instructions`` or
``text.format`` — makes the API drop the reasoning
(https://developers.openai.com/api/docs/guides/keep-prefix-unchanged), and
that replayed output must keep its original order
(https://developers.openai.com/api/docs/guides/reasoning).

Kestrel's tool loops rebuild their message list for every continuation, and
that rebuild is not the request the provider saw: it omits the operator notice
the first request carried, re-encodes tool arguments, prunes old results, and
offers the tools the registry exposes now. So the continuation is never
rebuilt from the live message list. When a request is serialized, the adapter
captures what it sent — every input item, the request's prefix-bearing
settings and tool declarations — and, when the reply arrives, the reply's
output items exactly as the provider emitted them. That is a
:class:`ResponsesSnapshot`. The next request of the same chain replays the
snapshot unchanged and may only append: tool outputs, new operator or user
turns, and newly exposed tools as a developer ``additional_tools`` item.

A snapshot is bound to the configured route that produced it (its route
name, its own adapter and SDK client, so two routes never share one), the
model requested, the response id and served model the provider returned, and
a digest of every provider-visible request setting outside the input items.
It lives only in memory, in a registry the turn lifecycle opens when a turn
starts and expires when the turn ends (:func:`responses_continuation_scope`).
Nothing here is persisted.

A snapshot that can no longer be continued is retired, and the registry keeps
a bounded record of why (:class:`RetiredSnapshot`, one of
:data:`RETIREMENT_REASONS`). A request that still continues a retired chain is
told it was retired and why, never that there was no snapshot.

The registry also carries the turn's context budget: the most text the turn's
tool loop pruned its own messages to (:func:`note_turn_context_budget`). A
continuation may only append, so one whose replay exceeds that budget cannot
honour the pruning and ends its chain instead.
"""

from __future__ import annotations

import contextlib
import contextvars
import threading
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

from kestrel_sovereign.turn_scope import turn_scoped

#: The most snapshots one turn keeps. A turn's main loop and each feature
#: subagent it runs keep one chain each; a turn with more retires its oldest.
MAX_SNAPSHOTS_PER_TURN = 16

#: The most retirement records one turn keeps. Routine ones (superseded or
#: evicted) are forgotten first, so a chain that ended for a reason keeps its
#: record however long another chain runs.
MAX_RETIRED_PER_TURN = 32

#: The most routes one turn remembers holding a snapshot on.
MAX_ROUTES_PER_TURN = 64

#: Why a snapshot was retired or a reply's snapshot never kept. A bounded
#: vocabulary: evidence names one of these and nothing else.
SUPERSEDED = "superseded"  # a continuation of the chain replaced it
EVICTED = "evicted"  # the turn held MAX_SNAPSHOTS_PER_TURN newer ones
PREFIX_PRUNED = "prefix_pruned"  # the caller no longer carries part of its prefix
PREFIX_CHANGED = "prefix_changed"  # the caller inserted into its prefix
BUDGET_EXCEEDED = "budget_exceeded"  # its replay exceeds the turn's context budget
REQUEST_SETTINGS_CHANGED = "request_settings_changed"  # a bound setting changed
SERVED_MODEL_MISMATCH = "served_model_mismatch"  # another model answered it
PREFIX_MISMATCH = "prefix_mismatch"  # the provider reported dropping replayed input
UNANCHORED_ASSISTANT_MESSAGE = "unanchored_assistant_message"
UNKNOWN_TOOL_OUTPUT = "unknown_tool_output"
DUPLICATE_TOOL_OUTPUT = "duplicate_tool_output"
UNSUPPORTED_MESSAGE_ROLE = "unsupported_message_role"
PARENT_RETIRED = "parent_retired"  # the chain it extended was retired first
UNIDENTIFIED_RESPONSE = "unidentified_response"  # its reply named no model or id
RETIREMENT_REASONS = frozenset({
    SUPERSEDED, EVICTED, PREFIX_PRUNED, PREFIX_CHANGED, BUDGET_EXCEEDED,
    REQUEST_SETTINGS_CHANGED, SERVED_MODEL_MISMATCH, PREFIX_MISMATCH,
    UNANCHORED_ASSISTANT_MESSAGE, UNKNOWN_TOOL_OUTPUT, DUPLICATE_TOOL_OUTPUT,
    UNSUPPORTED_MESSAGE_ROLE, PARENT_RETIRED, UNIDENTIFIED_RESPONSE,
})


@dataclass(frozen=True)
class ResponsesAnchor:
    """How the agent's message list refers to one provider reply.

    The tool loop replays a reply as an assistant message carrying its text
    and its tool calls. A reply with tool calls is found by its call ids,
    which the provider minted; a reply without any by its text. The message
    found must still carry the reply as the caller received it: its
    ``content``, and its tool calls, whose ids, names and arguments
    (compared by value) ``calls_digest`` digests. A caller that kept the
    calls but excluded or rewrote the text beside them, or rewrote a call,
    removed something the reply's replay would restore.
    """

    content: str
    call_ids: Tuple[str, ...]
    calls_digest: str


@dataclass(frozen=True)
class ResponsesSnapshot:
    """One captured continuation prefix and what it is bound to.

    ``items`` is every input item the chain has sent, followed by the latest
    reply's output items, each as the JSON text it was captured as.
    ``settings`` is the JSON text of the request settings that are part of
    the prefix (tools, reasoning, text format, include, store).
    ``request_digest`` is the digest the chain's requests are bound to: every
    provider-visible request field, ``extra_body`` included, other than the
    input items and the per-request tool offer (see
    :func:`~kestrel_sovereign.llm.openai_responses.request_digest`).
    ``anchors`` refers to every reply the chain absorbed, oldest first.
    ``carried`` holds, for each of those replies, the items the request that
    produced it built from the caller's messages, before the reply; the
    first entry's items from ``notices_from`` on are the turn notices that
    followed its user message (an operator notice), which the tool loop does
    not carry back. A continuation must still carry every other one of them,
    unchanged, or its chain ends.
    ``turn_request`` is the JSON text of the chain's first user message, used
    to confirm a chain whose replies carried no tool calls. ``route`` names
    the configured route; ``adapter`` and ``client`` are the route's own
    adapter and SDK client, held so their identities are never reused while
    the snapshot lives. ``declared_tools`` names every tool the chain
    declared, in the order it declared them, and ``tool_definitions`` holds
    each one's declaration as sent, so a tool redefined under a declared name
    is noticed.
    """

    route: str
    client: object
    requested_model: str
    served_model: str
    response_id: str
    settings: str
    items: Tuple[str, ...]
    declared_tools: Tuple[str, ...]
    anchors: Tuple[ResponsesAnchor, ...]
    turn_request: Optional[str]
    adapter: object = None
    request_digest: str = ""
    carried: Tuple[Tuple[str, ...], ...] = ()
    notices_from: int = 0
    tool_definitions: Tuple[str, ...] = ()


@dataclass(frozen=True)
class RouteKey:
    """A route the turn held a snapshot on, as :meth:`ResponsesRoute.serves` reads it."""

    route: str
    client: object
    adapter: object


@dataclass(frozen=True)
class RetiredSnapshot:
    """What the registry remembers of a snapshot it retired, and why.

    Enough to recognise a request that still continues the chain (its route,
    model and anchors), and nothing of its items: no encrypted reasoning, no
    message text beyond the anchors' and no tool-call arguments beyond their
    digest.
    """

    route: str
    adapter: object
    client: object
    requested_model: str
    response_id: str
    anchors: Tuple[ResponsesAnchor, ...]
    turn_request: Optional[str]
    reason: str

    @classmethod
    def of(cls, snapshot: ResponsesSnapshot, reason: str) -> "RetiredSnapshot":
        if reason not in RETIREMENT_REASONS:
            raise ValueError(f"Unknown snapshot retirement reason {reason!r}.")
        return cls(
            route=snapshot.route, adapter=snapshot.adapter, client=snapshot.client,
            requested_model=snapshot.requested_model, response_id=snapshot.response_id,
            anchors=snapshot.anchors, turn_request=snapshot.turn_request, reason=reason,
        )


class ResponsesContinuations:
    """The snapshots one turn's Responses calls captured.

    Mutated only by the adapter that serializes the turn's calls, under a
    lock so a feature subagent running beside the main loop cannot interleave
    a replacement. After :meth:`expire` nothing is kept or returned.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshots: list[ResponsesSnapshot] = []
        self._retired: list[RetiredSnapshot] = []
        self._routes: list[RouteKey] = []
        self._expired = False
        self._context_budget: Optional[int] = None

    @property
    def expired(self) -> bool:
        return self._expired

    @property
    def context_budget(self) -> Optional[int]:
        """The turn's context budget in characters, or ``None`` if none was set."""
        with self._lock:
            return self._context_budget

    def limit_context(self, chars: int) -> None:
        """Set the turn's context budget (see :func:`note_turn_context_budget`)."""
        if isinstance(chars, bool) or not isinstance(chars, int) or chars <= 0:
            raise ValueError(f"A context budget is a positive number of characters, not {chars!r}.")
        with self._lock:
            self._context_budget = chars

    def snapshots(self) -> Tuple[ResponsesSnapshot, ...]:
        """Every live snapshot, newest first."""
        with self._lock:
            if self._expired:
                return ()
            return tuple(reversed(self._snapshots))

    def retired(self) -> Tuple[RetiredSnapshot, ...]:
        """Every retirement the registry still remembers, newest first."""
        with self._lock:
            if self._expired:
                return ()
            return tuple(reversed(self._retired))

    def routes(self) -> Tuple[RouteKey, ...]:
        """Every route the turn kept or retired a snapshot on.

        Unlike :meth:`snapshots` and :meth:`retired`, eviction does not forget
        a route: a turn that held a snapshot on one stays a snapshot turn
        there. Past :data:`MAX_ROUTES_PER_TURN` routes the least recently used
        one without a live snapshot is forgotten, never one with.
        """
        with self._lock:
            if self._expired:
                return ()
            return tuple(self._routes)

    def _touch(self, snapshot: ResponsesSnapshot) -> None:
        """Remember ``snapshot``'s route, most recently used last.

        Past :data:`MAX_ROUTES_PER_TURN`, the least recently used route that
        holds no live snapshot is forgotten; a route with one never is.
        """
        self._routes = [
            key for key in self._routes
            if not (
                key.route == snapshot.route and key.client is snapshot.client
                and key.adapter is snapshot.adapter
            )
        ]
        self._routes.append(RouteKey(snapshot.route, snapshot.client, snapshot.adapter))
        while len(self._routes) > MAX_ROUTES_PER_TURN:
            idle = next((
                key for key in self._routes
                if not any(
                    kept.route == key.route and kept.client is key.client
                    and kept.adapter is key.adapter
                    for kept in self._snapshots
                )
            ), None)
            if idle is None:
                break
            self._routes.remove(idle)

    def retirement_of(self, snapshot: ResponsesSnapshot) -> Optional[str]:
        """Why ``snapshot`` was retired, if the registry remembers it."""
        with self._lock:
            for record in reversed(self._retired):
                if record.response_id == snapshot.response_id and record.route == snapshot.route:
                    return record.reason
            return None

    def record(
        self,
        snapshot: ResponsesSnapshot,
        *,
        replaces: Optional[ResponsesSnapshot] = None,
    ) -> bool:
        """Keep ``snapshot``, retiring the one it extends as superseded.

        Returns ``False`` when the registry has expired or ``replaces`` is no
        longer live (another request of the same chain extended or retired it
        first), in which case nothing is kept: the caller records why. Keeping
        more than :data:`MAX_SNAPSHOTS_PER_TURN` retires the oldest as evicted.
        """
        with self._lock:
            if self._expired:
                return False
            if replaces is not None:
                if not any(kept is replaces for kept in self._snapshots):
                    return False
                self._remove(replaces, SUPERSEDED)
            self._snapshots.append(snapshot)
            self._touch(snapshot)
            while len(self._snapshots) > MAX_SNAPSHOTS_PER_TURN:
                self._remove(self._snapshots[0], EVICTED)
            return True

    def discard(self, snapshot: ResponsesSnapshot, reason: str) -> None:
        """Retire ``snapshot`` for ``reason``: its chain can no longer be continued."""
        with self._lock:
            self._remove(snapshot, reason)

    def retire(self, snapshot: ResponsesSnapshot, reason: str) -> None:
        """Remember that ``snapshot``, never kept, may not be continued."""
        with self._lock:
            if not self._expired:
                self._remember(RetiredSnapshot.of(snapshot, reason))
                self._touch(snapshot)

    def _remove(self, snapshot: ResponsesSnapshot, reason: str) -> None:
        record = RetiredSnapshot.of(snapshot, reason)
        kept = [entry for entry in self._snapshots if entry is not snapshot]
        if len(kept) != len(self._snapshots):
            self._snapshots = kept
            self._remember(record)

    def _remember(self, record: RetiredSnapshot) -> None:
        self._retired.append(record)
        while len(self._retired) > MAX_RETIRED_PER_TURN:
            routine = next(
                (index for index, kept in enumerate(self._retired) if kept.reason in _ROUTINE),
                0,
            )
            del self._retired[routine]

    def expire(self) -> None:
        """End the registry with its turn; every snapshot and record is dropped."""
        with self._lock:
            self._expired = True
            self._snapshots = []
            self._retired = []
            self._routes = []


#: Retirements that record no reason a chain ended, only that it moved on.
_ROUTINE = frozenset({SUPERSEDED, EVICTED})


_CONTINUATIONS: contextvars.ContextVar[Optional[ResponsesContinuations]] = (
    contextvars.ContextVar("kestrel_responses_continuations", default=None)
)


def current_responses_continuations() -> Optional[ResponsesContinuations]:
    """The running turn's registry, or ``None`` outside a turn."""
    registry = _CONTINUATIONS.get()
    if registry is None or registry.expired:
        return None
    return registry


def note_turn_context_budget(chars: int) -> None:
    """Record the context budget the turn's tool loop prunes its messages to.

    ``chars`` is the most characters of message text, tool-call arguments,
    tool results and tool declarations one of its requests may carry. The
    OpenAI Responses path measures a continuation, exactly as it will be
    sent, against it (#4198). Outside a turn there is nothing to record.
    """
    registry = current_responses_continuations()
    if registry is not None:
        registry.limit_context(chars)


@contextlib.contextmanager
def bind_responses_continuations(
    registry: Optional[ResponsesContinuations],
) -> Iterator[None]:
    """Re-present a captured turn registry on a task that predates the turn."""
    token = _CONTINUATIONS.set(registry)
    try:
        yield
    finally:
        _CONTINUATIONS.reset(token)


@contextlib.contextmanager
def responses_continuation_scope() -> Iterator[ResponsesContinuations]:
    """Open one turn's registry and expire it when the turn ends."""
    registry = ResponsesContinuations()
    token = _CONTINUATIONS.set(registry)
    try:
        yield registry
    finally:
        registry.expire()
        _CONTINUATIONS.reset(token)


def _capture_responses_continuations(
    _agent: object,
) -> Optional[ResponsesContinuations]:
    return _CONTINUATIONS.get()


turn_scoped(
    "responses_continuations",
    variables=(_CONTINUATIONS,),
    capture=_capture_responses_continuations,
    bind=bind_responses_continuations,
)


__all__ = [
    "BUDGET_EXCEEDED",
    "DUPLICATE_TOOL_OUTPUT",
    "EVICTED",
    "MAX_RETIRED_PER_TURN",
    "MAX_ROUTES_PER_TURN",
    "MAX_SNAPSHOTS_PER_TURN",
    "PARENT_RETIRED",
    "PREFIX_CHANGED",
    "PREFIX_MISMATCH",
    "PREFIX_PRUNED",
    "REQUEST_SETTINGS_CHANGED",
    "RETIREMENT_REASONS",
    "ResponsesAnchor",
    "ResponsesContinuations",
    "ResponsesSnapshot",
    "RetiredSnapshot",
    "RouteKey",
    "SERVED_MODEL_MISMATCH",
    "SUPERSEDED",
    "UNANCHORED_ASSISTANT_MESSAGE",
    "UNIDENTIFIED_RESPONSE",
    "UNKNOWN_TOOL_OUTPUT",
    "UNSUPPORTED_MESSAGE_ROLE",
    "bind_responses_continuations",
    "current_responses_continuations",
    "note_turn_context_budget",
    "responses_continuation_scope",
]
