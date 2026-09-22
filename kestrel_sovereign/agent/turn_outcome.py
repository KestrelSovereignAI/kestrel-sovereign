"""How a cognition turn ended, computed once and delivered to every span.

Issue #3159. A turn span used to say only *that* it ended — and on several
paths not even that:

* a cooperative checkpoint Stop returns normally, so the turn ended UNSET and
  read exactly like a completed one;
* a pre-admission Stop raises ``InvocationCancelledError``, so the turn ended
  ERROR and read exactly like a failure;
* ``CancelledError``, ``GeneratorExit`` and the safe-mode early return escaped
  the streaming path's ``except Exception`` / ``else`` pair entirely, so the
  span was never ended and never exported at all.

The outcome is derived from the request lifecycle's own durable record that a
cooperative Stop was acknowledged for THIS turn's generation, never from the
exception type: a Stop can surface as ``CancelledError``, and so can a client
disconnect or a host shutdown. The exception type only separates a genuine
failure from the cancellation family.

The receipt remains the sole authority for *why* a turn was stopped. Nothing
here copies a receipt's ``reason`` or ``actor_id`` onto a span: Phoenix has its
own custody and retention (``docs/architecture/security/PHOENIX_TRACE_CUSTODY.md``)
and the Navigator renders raw attributes.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from kestrel_sovereign.agent.invocation import (
    InvocationCancelledError,
    InvocationSelfFencedError,
    current_invocation_id,
)
from kestrel_sovereign.telemetry import TurnOutcome

logger = logging.getLogger(__name__)


# Cancellation arrives in several shapes, none of which says WHO cancelled.
# Everything here resolves to stopped/disconnected/interrupted by consulting
# the lifecycle; every OTHER ``Exception`` is a real failure.
_CANCELLATION_SHAPES = (
    asyncio.CancelledError,
    InvocationCancelledError,
    InvocationSelfFencedError,
)


def _is_cancellation_shaped(error: BaseException) -> bool:
    if isinstance(error, _CANCELLATION_SHAPES):
        return True
    # KeyboardInterrupt / SystemExit: the host is going away, which is an
    # interruption of the turn rather than a defect in it.
    return not isinstance(error, Exception)


def cooperative_stop_recorded(agent: object) -> bool:
    """Whether the lifecycle recorded an acknowledged Stop for THIS turn.

    ``is_request_cancelled`` is generation-aware: it resolves the calling
    task's own delivery generation, so a fresh redelivery reusing the same
    request id is not poisoned by an older Stop. A self-fence is excluded
    deliberately — an infrastructure lease that became unsafe is not evidence
    that an operator asked for anything (see ``InvocationSelfFencedError``).
    """

    is_cancelled = getattr(agent, "is_request_cancelled", None)
    if not callable(is_cancelled):
        return False
    request_id = current_invocation_id()
    if not isinstance(request_id, str) or not request_id:
        request_id = getattr(agent, "_current_request_id", None)
    if not isinstance(request_id, str) or not request_id:
        return False
    # Structural guard, not a swallowed error: ``StreamingMixin`` and
    # ``TurnLifecycleMixin`` are reusable and can be hosted by a minimal or
    # duck-typed object with no lifecycle state at all. Such a host has no
    # Stop record to read, and a mock's truthy return must not be mistaken
    # for one.
    if not isinstance(
        getattr(agent, "_cancelled_request_generations", None), set
    ) and not isinstance(getattr(agent, "_cancelled_requests", None), set):
        return False
    if not is_cancelled(request_id):
        return False
    self_fenced = getattr(agent, "is_request_self_fenced", None)
    if callable(self_fenced) and self_fenced(request_id):
        return False
    return True


def resolve_turn_outcome(
    agent: object,
    error: Optional[BaseException],
) -> TurnOutcome:
    """Classify one turn's exit. One value, computed once, for every span."""

    if isinstance(error, GeneratorExit):
        # The consumer closed the stream. That is true even when the
        # disconnect also cancelled the request, and it is the more specific
        # fact of the two.
        return TurnOutcome.DISCONNECTED
    if error is not None and not _is_cancellation_shaped(error):
        return TurnOutcome.FAILED
    if cooperative_stop_recorded(agent):
        return TurnOutcome.STOPPED
    if error is None:
        return TurnOutcome.COMPLETED
    return TurnOutcome.INTERRUPTED


def publish_turn_outcome(
    agent: object,
    turn_id: Optional[str],
    outcome: TurnOutcome,
) -> None:
    """Deliver the computed outcome to feature-owned turn roots.

    The observability feature opens its own turn root at USER_PROMPT_SUBMIT and
    used to end it on the SDK ``Stop`` hook — which the strict-audit cancel
    paths skip, leaving those turns open until the Timeline's abandoned cap.
    Core computes the outcome once and hands it over here so both spans agree;
    core never imports the feature, and a listener that raises must not be able
    to fail a turn that has already run.
    """

    if turn_id is None or not isinstance(outcome, TurnOutcome):
        return
    listeners = getattr(agent, "_turn_outcome_listeners", None)
    if not isinstance(listeners, list) or not listeners:
        return
    for listener in tuple(listeners):
        try:
            listener(turn_id, outcome)
        except Exception:  # noqa: BLE001 - telemetry must not fail cognition
            logger.warning(
                "Turn outcome listener raised for turn %s", turn_id, exc_info=True
            )


__all__ = [
    "TurnOutcome",
    "cooperative_stop_recorded",
    "publish_turn_outcome",
    "resolve_turn_outcome",
]
