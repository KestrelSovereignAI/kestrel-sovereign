"""Last-instant precondition revalidation for a COGNITION dispatch.

A source can have a precondition that is true when the scheduler decides to
fire and false by the time the turn actually starts. The scheduler's
``self_followup`` privacy guard is the motivating case (#3101 review P1): it
checks ``hides_persisted_user_content(agent)`` and then calls
``await dispatcher.dispatch_signal(...)``. There is no ``await`` between those
two statements, but ``dispatch_signal`` is itself a suspension point — durable
admission, event persistence and lock acquisition all await before the
COGNITION route reaches ``process_input``. A privacy transition landing in that
window produced a turn carrying resurrected conversation content under a mode
that forbids it; that was reproduced against the real dispatcher.

The obvious repair — hold the agent's privacy-transition lock across
check-and-dispatch, mirroring the write side in ``_create_schedule`` — is
**wrong here**, and that is worth stating in the code rather than rediscovering.
The system's lock-order invariant is CONVERSATION (via ``_turn_lifecycle``)
**before** the transition lock; see the comment in
``StreamingMixin.process_input_streaming``. Taking the transition lock in the
scheduler and then entering ``process_input``, which acquires CONVERSATION,
inverts that pair and reintroduces the AB-BA wedge this repository has already
fixed once.

So the guard is evaluated **twice**, at two different kinds of boundary, and
the second one is the load-bearing one:

1. At the last *synchronous* instant of the dispatcher pipeline, before the
   cognition turn is even created. This is an early refusal, not a
   serialization: it closes the long I/O-bound stretch (durable admission,
   event persistence, lock acquisition, constitution anchoring) cheaply, and
   it is the only check available for a duck-typed agent whose
   ``process_input`` cannot take a precondition.
2. **Inside the turn's own ``CONVERSATION`` → privacy-transition lock span**,
   immediately before the prompt is consumed. The dispatcher hands the guard
   down as ``process_input(..., turn_precondition=...)`` and
   ``KestrelAgent.process_input`` calls it as its first act inside that span.

Check 1 alone was not enough, and three successive review rounds each found
that same defect one door further along: ``await_monitored_execution`` creates
the execution task and yields, and ``process_input`` awaits readiness
(byok refresh, the genesis gate, the periodic constitution audit) before it
consumes the prompt. A transition landing in *those* suspension points still
reached a turn carrying the persisted intent.

What makes check 2 different in kind is that it is not another read-side test
racing a write-side transition — it runs *inside the span the writer must
acquire*. ``set_privacy_mode`` / ``confirm_privacy_transition`` take
``KestrelAgent._privacy_transition_lock``; the turn now holds that lock from
before the guard runs until the turn ends, in the same
CONVERSATION-then-transition order the streaming turn has always used. So a
transition either lands before the guard (and the guard sees it and refuses)
or blocks until the turn is over. There is no third interleaving, which is
why this is the last door rather than a fourth one.

The guard itself stays deliberately **sync**: a coroutine guard would put an
``await`` back between the check and the prompt, reopening the window inside
the very span that exists to close it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from kestrel_sdk.signals import Signal, SourceRegistration

#: Called with the signal being dispatched and the dispatching agent.
#: Returns ``None`` to allow the turn, or a short operator-readable reason to
#: refuse it. The reason reaches ``signal_log`` and the scheduler's execution
#: record, so it must describe the refusal WITHOUT quoting the payload.
PreTurnGuard = Callable[[Signal, Any], Optional[str]]


class TurnPreconditionRefused(RuntimeError):
    """A source's precondition no longer held at the turn's own lock span.

    Raised by the ``turn_precondition`` callable the dispatcher hands to
    ``process_input``, from inside the turn's CONVERSATION → transition span
    and before the prompt is consumed. The turn unwinds without persisting or
    processing anything; the dispatcher maps it to
    ``Status.DROPPED_VALIDATION`` so the occurrence is recorded as a refusal
    rather than as a turn that ran.

    An exception rather than a return value on purpose: ``process_input``
    returns the agent's response text, so a refusal returned in-band would be
    indistinguishable from a turn that ran and said something — the exact
    "accept that produces no turn but reports success" shape #3101 exists to
    prevent.
    """


@dataclass
class SourceRegistrationWithPreTurnGuard(SourceRegistration):
    """Source registration that revalidates a precondition before its turn.

    A Core-side subclass for the same reason as
    :class:`~kestrel_sovereign.signals.prompt_overrides.SourceRegistrationWithPromptOverride`:
    the canonical dataclass lives in ``kestrel_sdk.signals`` and does not carry
    this field yet. It remains an ordinary ``SourceRegistration`` to the
    registry and the dispatcher, which reads the attribute defensively.
    """

    pre_turn_guard: Optional[PreTurnGuard] = None


__all__ = [
    "PreTurnGuard",
    "SourceRegistrationWithPreTurnGuard",
    "TurnPreconditionRefused",
]
