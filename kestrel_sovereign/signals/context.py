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

from contextvars import ContextVar, Token
from typing import TYPE_CHECKING, Optional

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
