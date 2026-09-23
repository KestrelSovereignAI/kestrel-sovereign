"""Dispatch-scoped context: the Signal currently being dispatched.

The dispatcher publishes the in-flight :class:`~kestrel_sdk.signals.Signal`
in a per-async-task ``ContextVar`` for the duration of a dispatch, so code
that runs *inside* a handler/turn can observe the envelope that drove it —
without changing the ``ActionHandler = Callable[[dict], ...]`` contract
(handlers still receive only the validated payload).

The first consumer is the Talon coordinator's orchestrator/workflow
correlation stamping (kestrel-talon#53 contract): when a talon dispatch is
driven by a ``kind == "workflow.stage"`` signal, the coordinator reads the
workflow run id off ``Signal.session_id`` (the workflows feature sets
``session_id=run.run_id``) and the stage name off the payload / causation
chain, and stamps them onto the outgoing talon invocation.

A ``ContextVar`` (not agent-level mutable state) keeps concurrent dispatches
isolated: each ``enqueue_signal`` background task sets/resets its own copy.
Reads outside any dispatch return ``None`` — callers treat that as a direct
(non-signal-driven) invocation.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import TYPE_CHECKING, Iterator, Optional

from kestrel_sovereign.turn_scope import turn_scoped

if TYPE_CHECKING:  # pragma: no cover - import cycle guard, typing only
    from kestrel_sdk.signals import Signal

_current_signal: ContextVar[Optional["Signal"]] = ContextVar(
    "kestrel_current_signal", default=None
)


def get_current_signal() -> Optional["Signal"]:
    """The Signal whose dispatch the calling task is running inside, or None."""
    return _current_signal.get()


def set_current_signal(signal: Optional["Signal"]) -> Token:
    """Publish ``signal`` for the current task. Returns the reset token."""
    return _current_signal.set(signal)


def reset_current_signal(token: Token) -> None:
    """Restore the previous value (always call from a ``finally``)."""
    _current_signal.reset(token)


@contextmanager
def bind_current_signal(signal: Optional["Signal"]) -> Iterator[None]:
    """Re-present a captured Signal on a task whose context predates it.

    ``set_current_signal`` runs only on the dispatching task.  A turn's tools
    that run on a task created before that dispatch (the codex app-server
    reader loop) would otherwise read ``None`` — and a guard keyed on the
    driving signal, such as the A2A recipient-decline bookkeeping in
    ``EventManagerMixin``, silently stops recognising its own dispatch.
    Binding ``None`` explicitly clears a stale signal the task inherited.
    """
    token = _current_signal.set(signal)
    try:
        yield
    finally:
        _current_signal.reset(token)


turn_scoped(
    "current_signal",
    variables=(_current_signal,),
    capture=lambda _agent: get_current_signal(),
    bind=bind_current_signal,
)
