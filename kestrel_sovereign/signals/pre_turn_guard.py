"""Source-declared pre-turn admission, evaluated inside the turn's own span.

#3310. A COGNITION source sometimes has a precondition the turn must still
satisfy at the instant it consumes its prompt — the motivating case is a
privacy mode that may no longer permit the persisted intent the wake is
carrying. Such a check is only worth anything if nothing can change the
answer between the check and the prompt being used.

Before #3310 there was nowhere to put it. The check kept being moved one door
further along — schedule creation, fire time, the last synchronous instant
before handoff — and a transition kept landing in a later ``await``, because
the non-streaming turn serialized against nothing a privacy transition takes.
``KestrelAgent.process_input`` now holds CONVERSATION (via ``_turn_lifecycle``)
and then the privacy-transition mutex for its whole body, which finally makes
one region authoritative.

This module is the seam that puts the source's check *in* that region:

* a source declares ``pre_turn_guard`` on its registration (via
  :class:`SourceRegistrationWithPreTurnGuard`, or any registration object
  carrying the attribute — the registry and dispatcher both read it with
  ``getattr`` so an author can combine it with other adapter subclasses);
* :class:`~kestrel_sovereign.signals.dispatcher.SignalDispatcher` binds the
  signal to it and hands the zero-argument result to ``process_input``; and
* ``process_input`` runs it as the FIRST operation inside the span, and raises
  :class:`PreTurnRefusal` when it refuses. The dispatcher maps that to
  ``Status.DROPPED_VALIDATION`` — no turn ran, and the occurrence says so.

The guard is **synchronous by contract**. The whole value of the region is
that it contains no suspension point between the check and the prompt being
consumed; a guard that could ``await`` would put one back. The registry
rejects a coroutine function at registration time and ``process_input``
refuses an awaitable return at call time.

The seam is deliberately small. It is the one place a source's pre-turn
precondition is evaluated — not a second one next to an existing check — so
it can be deleted rather than grown.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from kestrel_sdk.signals import Signal, SourceRegistration

#: A source's pre-turn admission: given the signal about to become a turn,
#: return ``None`` to admit it or a refusal reason to stop it. Synchronous.
PreTurnGuard = Callable[[Signal], Optional[str]]

#: The same guard with its signal already bound, as ``process_input`` receives
#: it. The turn evaluates an admission decision; it does not read signals.
BoundPreTurnGuard = Callable[[], Optional[str]]


class PreTurnRefusal(Exception):
    """A source's pre-turn guard refused the turn from inside its own span.

    Raised by ``KestrelAgent.process_input`` while it holds CONVERSATION and
    the privacy-transition mutex, before any of the turn body runs. It is a
    policy decision, not a failure: the dispatcher maps it to
    ``Status.DROPPED_VALIDATION``. A guard that *raises* something else is a
    bug in the guard and is deliberately left to propagate as ``FAILED``.
    """


@dataclass
class SourceRegistrationWithPreTurnGuard(SourceRegistration):
    """Source registration that declares a pre-turn admission guard.

    COGNITION-only: a guard exists to stop a turn, and ACTION / ARTIFACT
    dispatches have no turn to stop. The registry enforces that, rejects a
    non-callable or coroutine-function guard, and folds the guard into the
    source's contract signature so a re-registration that swaps the guard is
    a mismatch rather than silently keeping the old one.
    """

    pre_turn_guard: Optional[PreTurnGuard] = None


__all__ = [
    "BoundPreTurnGuard",
    "PreTurnGuard",
    "PreTurnRefusal",
    "SourceRegistrationWithPreTurnGuard",
]
