"""Progress a non-streaming LLM call reports while its answer arrives (#3552).

A streamed call shows it is alive by yielding items. A non-streaming call
yields nothing until it returns, so a bound on its silence (the orchestrator's
``LLMCallWatchdog``) cannot tell a long answer from a hung provider unless the
call says when part of its answer has arrived.

An adapter that reads a non-streaming answer incrementally, over a stream it
does not expose, calls :func:`report_call_progress` for each part it receives.
Whoever bounds the call listens with :func:`call_progress_listener`. A report
with no listener does nothing, so adapters report unconditionally.

An adapter that receives its answer in one piece reports nothing, and a bound
around it covers the whole call.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable, Iterator, Optional

_LISTENER: ContextVar[Optional[Callable[[], None]]] = ContextVar(
    "kestrel_llm_call_progress_listener", default=None,
)


def report_call_progress() -> None:
    """Tell whoever bounds the current call that part of its answer arrived."""
    listener = _LISTENER.get()
    if listener is not None:
        listener()


@contextmanager
def call_progress_listener(listener: Callable[[], None]) -> Iterator[None]:
    """Route the progress of calls awaited inside this block to ``listener``."""
    token = _LISTENER.set(listener)
    try:
        yield
    finally:
        _LISTENER.reset(token)


__all__ = ["call_progress_listener", "report_call_progress"]
