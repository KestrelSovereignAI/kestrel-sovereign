"""Progress an LLM call reports while its answer arrives (#3552, #3357).

A bound on a call's silence (the orchestrator's ``LLMCallWatchdog``) cannot
tell a long answer from a hung provider unless the call says when the provider
sent something. A streamed call shows it is alive by yielding items, but not
every provider event becomes an item: a tool call's arguments are held until
the call is whole, and thinking that is not displayed yields nothing. A
non-streaming call yields nothing at all until it returns.

So an adapter calls :func:`report_call_progress` for each provider event it
reads, streamed or not, whether or not that event yields anything. Whoever
bounds the call listens with :func:`call_progress_listener`. A report carries
no content and is never yielded, so a consumer of the stream sees exactly
what it saw without it. A report with no listener does nothing, so adapters
report unconditionally.

An adapter that receives its answer in one piece reports nothing, and a bound
around it covers the whole call.

The service also says which route and model each attempt of the call waits on
(:func:`report_call_attempt`, #2658), so a bound that gives up on a silent
call can name the provider that went silent.

A call that falls back across routes is not one wait but one per candidate,
and each candidate's silence is measured against its own route (#1968). The
service's dispatch loops therefore run every candidate attempt through
:func:`bound_attempt_stream` or :func:`bound_attempt_call`, which hand it to
the :class:`AttemptBound` bounding the call, if any. A candidate that goes its
route's bound without progress raises :class:`AttemptInactivityTimeout` inside
the loop, which treats it as that candidate's failure: the loop moves on to
the next candidate it already admitted, or fails the call, exactly as for any
other failure; a route whose harness owns its transport fails the call
instead, as on the harness's own stall. With no bound listening, both run the
attempt as they found it.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import Context, ContextVar
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Iterator,
    Mapping,
    Optional,
    Protocol,
    TypeVar,
)

from .output_ceiling import IncompleteGenerationError

_T = TypeVar("_T")

#: The phase a candidate attempt was abandoned in: a streamed attempt that had
#: yielded nothing yet, a streamed attempt between two items, or an attempt
#: that returns its answer in one piece.
FIRST_TOKEN_PHASE = "first-token"
INTER_CHUNK_PHASE = "inter-chunk"
NON_STREAMING_PHASE = "non-streaming"


class AttemptInactivityTimeout(IncompleteGenerationError):
    """One candidate route of a bounded call went its bound without progress.

    Raised inside the dispatch loop (#1968), so the loop treats it as that
    candidate's failure. ``provider`` and ``model`` name the candidate,
    ``phase`` what it was waiting for (:data:`FIRST_TOKEN_PHASE`,
    :data:`INTER_CHUNK_PHASE` or :data:`NON_STREAMING_PHASE`) and ``timeout``
    the bound, in seconds, its route was given.
    """

    def __init__(
        self, *, provider: str, model: str, phase: str, timeout: float,
    ) -> None:
        self.provider = provider
        self.model = model
        self.phase = phase
        self.timeout = timeout
        self.summary = f"timeout after {int(timeout)}s"
        super().__init__(
            f"Route {provider} (model {model}) produced nothing for "
            f"{timeout:g}s ({phase}) and was abandoned"
        )


class AttemptBound(Protocol):
    """Bounds each candidate attempt of the call it listens to (#1968).

    ``route`` is the candidate's provider entry, as routing resolved it.
    """

    def attempt_stream(
        self,
        route: Mapping[str, Any],
        provider: str,
        model: str,
        stream: AsyncIterator[_T],
    ) -> AsyncIterator[_T]:
        """``stream``'s items, each wait for the next one bounded."""

    def attempt_call(
        self,
        route: Mapping[str, Any],
        provider: str,
        model: str,
        pending: Awaitable[_T],
    ) -> Awaitable[_T]:
        """``pending``'s result, its wait bounded."""


_LISTENER: ContextVar[Optional[Callable[[], None]]] = ContextVar(
    "kestrel_llm_call_progress_listener", default=None,
)
_ATTEMPT_LISTENER: ContextVar[Optional[Callable[[str, str], None]]] = ContextVar(
    "kestrel_llm_call_attempt_listener", default=None,
)
_ATTEMPT_BOUND: ContextVar[Optional[AttemptBound]] = ContextVar(
    "kestrel_llm_call_attempt_bound", default=None,
)


def inherit_call_progress(context: Context) -> None:
    """Pin one observation scope on a previously unbound lazy source context.

    Sources can be constructed before the watchdog's first pull. Authority,
    turn and privacy bindings stay those captured at construction; a consumer
    cannot replace an already-pinned call listener with its own.
    """
    variables = (_LISTENER, _ATTEMPT_LISTENER, _ATTEMPT_BOUND)
    if any(context.get(variable) is not None for variable in variables):
        return
    for variable in variables:
        context.run(variable.set, variable.get())


def report_call_progress() -> None:
    """Tell whoever bounds the current call that the provider sent something."""
    listener = _LISTENER.get()
    if listener is not None:
        listener()


def report_call_attempt(provider: str, model: str) -> None:
    """Tell whoever bounds the current call which route and model it now awaits."""
    listener = _ATTEMPT_LISTENER.get()
    if listener is not None:
        listener(provider, model)


def bound_attempt_stream(
    route: Optional[Mapping[str, Any]],
    provider: str,
    model: str,
    stream: AsyncIterator[_T],
) -> AsyncIterator[_T]:
    """``stream``, one candidate's streamed attempt, under the call's bound.

    ``route`` is the candidate's provider entry. An attempt with none (the
    private inference lease route, which is no configured route and never
    falls back) and an attempt no bound listens to run as they are, under
    whatever bounds the whole call.
    """
    bound = _ATTEMPT_BOUND.get()
    if bound is None or route is None:
        return stream
    return bound.attempt_stream(route, provider, model, stream)


async def bound_attempt_call(
    route: Optional[Mapping[str, Any]],
    provider: str,
    model: str,
    pending: Awaitable[_T],
) -> _T:
    """Await ``pending``, one candidate's non-streaming attempt, under the
    call's bound; see :func:`bound_attempt_stream`."""
    bound = _ATTEMPT_BOUND.get()
    if bound is None or route is None:
        return await pending
    return await bound.attempt_call(route, provider, model, pending)


@contextmanager
def call_progress_listener(
    listener: Callable[[], None],
    attempt_listener: Optional[Callable[[str, str], None]] = None,
    attempt_bound: Optional[AttemptBound] = None,
) -> Iterator[None]:
    """Route the progress of calls awaited inside this block to ``listener``,
    the route and model of each attempt to ``attempt_listener``, and each
    candidate attempt to ``attempt_bound``."""
    token = _LISTENER.set(listener)
    attempt_token = _ATTEMPT_LISTENER.set(attempt_listener)
    bound_token = _ATTEMPT_BOUND.set(attempt_bound)
    try:
        yield
    finally:
        _ATTEMPT_BOUND.reset(bound_token)
        _ATTEMPT_LISTENER.reset(attempt_token)
        _LISTENER.reset(token)


__all__ = [
    "AttemptBound",
    "AttemptInactivityTimeout",
    "FIRST_TOKEN_PHASE",
    "INTER_CHUNK_PHASE",
    "NON_STREAMING_PHASE",
    "bound_attempt_call",
    "bound_attempt_stream",
    "call_progress_listener",
    "report_call_attempt",
    "report_call_progress",
]
