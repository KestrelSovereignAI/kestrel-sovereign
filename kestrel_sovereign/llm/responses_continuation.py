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

A snapshot is bound to the route that produced it (its configured client,
not just its vendor and base URL), the model requested and to the response id
and served model the provider returned. It lives only in memory, in a registry
the turn lifecycle opens when a turn starts and expires when the turn ends
(:func:`responses_continuation_scope`). Nothing here is persisted.

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


@dataclass(frozen=True)
class ResponsesAnchor:
    """How the agent's message list refers to one provider reply.

    The tool loop replays a reply as an assistant message carrying its text
    and its tool calls. A reply with tool calls is found by its call ids,
    which the provider minted; a reply without any by its text.
    """

    content: str
    call_ids: Tuple[str, ...]


@dataclass(frozen=True)
class ResponsesSnapshot:
    """One captured continuation prefix and what it is bound to.

    ``items`` is every input item the chain has sent, followed by the latest
    reply's output items, each as the JSON text it was captured as.
    ``settings`` is the JSON text of the request settings that are part of
    the prefix (tools, reasoning, text format, include, store).
    ``anchors`` refers to every reply the chain absorbed, oldest first.
    ``turn_request`` is the JSON text of the chain's first user message, used
    to confirm a chain whose replies carried no tool calls. ``route`` labels
    the route; ``client`` is the route's own SDK client, held so its identity
    is never reused while the snapshot lives. ``declared_tools`` names every
    tool the chain declared, in the order it declared them.
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


class ResponsesContinuations:
    """The snapshots one turn's Responses calls captured.

    Mutated only by the adapter that serializes the turn's calls, under a
    lock so a feature subagent running beside the main loop cannot interleave
    a replacement. After :meth:`expire` nothing is kept or returned.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshots: list[ResponsesSnapshot] = []
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

    def record(
        self,
        snapshot: ResponsesSnapshot,
        *,
        replaces: Optional[ResponsesSnapshot] = None,
    ) -> bool:
        """Keep ``snapshot``, retiring the one it extends.

        Returns ``False`` when the registry has expired or ``replaces`` is no
        longer live (another request of the same chain extended or retired it
        first), in which case nothing is kept.
        """
        with self._lock:
            if self._expired:
                return False
            if replaces is not None:
                if not any(kept is replaces for kept in self._snapshots):
                    return False
                self._snapshots = [
                    kept for kept in self._snapshots if kept is not replaces
                ]
            self._snapshots.append(snapshot)
            if len(self._snapshots) > MAX_SNAPSHOTS_PER_TURN:
                del self._snapshots[: len(self._snapshots) - MAX_SNAPSHOTS_PER_TURN]
            return True

    def discard(self, snapshot: ResponsesSnapshot) -> None:
        """Retire ``snapshot``: its chain can no longer be continued."""
        with self._lock:
            self._snapshots = [
                kept for kept in self._snapshots if kept is not snapshot
            ]

    def expire(self) -> None:
        """End the registry with its turn; every snapshot is dropped."""
        with self._lock:
            self._expired = True
            self._snapshots = []


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
    "MAX_SNAPSHOTS_PER_TURN",
    "ResponsesAnchor",
    "ResponsesContinuations",
    "ResponsesSnapshot",
    "bind_responses_continuations",
    "current_responses_continuations",
    "note_turn_context_budget",
    "responses_continuation_scope",
]
