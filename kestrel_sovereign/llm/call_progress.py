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
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable, Iterator, Optional

_LISTENER: ContextVar[Optional[Callable[[], None]]] = ContextVar(
    "kestrel_llm_call_progress_listener", default=None,
)
_ATTEMPT_LISTENER: ContextVar[Optional[Callable[[str, str], None]]] = ContextVar(
    "kestrel_llm_call_attempt_listener", default=None,
)


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


@contextmanager
def call_progress_listener(
    listener: Callable[[], None],
    attempt_listener: Optional[Callable[[str, str], None]] = None,
) -> Iterator[None]:
    """Route the progress of calls awaited inside this block to ``listener``,
    and the route and model of each attempt to ``attempt_listener``."""
    token = _LISTENER.set(listener)
    attempt_token = _ATTEMPT_LISTENER.set(attempt_listener)
    try:
        yield
    finally:
        _ATTEMPT_LISTENER.reset(attempt_token)
        _LISTENER.reset(token)


__all__ = ["call_progress_listener", "report_call_attempt", "report_call_progress"]
