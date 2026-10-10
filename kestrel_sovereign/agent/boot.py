"""Explicit, ordered, rollback-safe agent boot state machine (#2522).

``KestrelAgent.initialize()`` used to be a ~1,000-line method whose body was
guarded by ``if self._raw_storage is None:``. Because ``_raw_storage`` was
assigned near the very top, any later failure left the guard "satisfied": a
second ``initialize()`` skipped almost the entire body and ran only the tail
(salvage worker, spawn mandate, provider readiness, ``on_agent_ready`` hooks)
against a partially-initialized agent. There was no phase journal, no
reverse-order rollback, and no terminal failed state.

This module supplies the small, agent-agnostic primitives that make boot an
explicit state machine:

* :class:`BootPhaseState` — the four states an agent's boot can be in.
* :class:`AgentBootError` — raised when ``initialize()`` is called in a state
  that forbids it (already running, or previously failed and not closed).
* :class:`BootPhase` — one named phase: a coroutine plus the
  ``retained`` resources it deliberately keeps on failure (the documented
  resumable contract).
* :class:`BootContext` — carries cross-phase values and, crucially, the
  **reverse-order rollback stack**. Each phase pushes an undo action as it
  acquires a resource, which also puts that resource in the agent's
  :class:`~kestrel_sovereign.agent.custody.ResourceCustody`; on failure the
  stack unwinds LIFO, and a resource leaves custody only when its undo
  reports :attr:`~kestrel_sovereign.agent.custody.ReleaseOutcome.RELEASED`
  for an owner known to report a failed release truthfully.
* :func:`run_boot_sequence` — the orchestrator: runs phases in order,
  records a journal, and on *any* exception (including
  :class:`asyncio.CancelledError`) unwinds the rollback stack to completion
  before re-raising and marking the agent ``FAILED``.

The phase *bodies* live on ``KestrelAgent`` (they need the full agent surface);
this module owns only the sequencing/rollback contract so it can be unit-tested
in isolation.
"""

from __future__ import annotations

import asyncio
import enum
import logging
from dataclasses import dataclass
from typing import Awaitable, Callable, List, Optional

from kestrel_sovereign.agent.custody import ReleaseOutcome, ResourceCustody

logger = logging.getLogger(__name__)

# An undo action: a zero-arg coroutine function that releases one resource and
# reports what it did. Anything but ``ReleaseOutcome.RELEASED`` keeps it held.
RollbackAction = Callable[[], Awaitable[ReleaseOutcome]]
# A phase body: a coroutine function taking the shared BootContext.
PhaseBody = Callable[["BootContext"], Awaitable[None]]
# Callback the orchestrator uses to publish state transitions onto the agent.
StateSetter = Callable[["BootPhaseState"], None]


class BootPhaseState(enum.Enum):
    """The state of an agent's boot sequence.

    ``READY`` is the *only* state in which readiness may have fired; it is set
    exactly once, after every phase commits. ``FAILED`` is terminal: the
    resources a partial boot acquired have been rolled back (any whose
    release custody could not confirm stay held, #3522), and a fresh
    ``initialize()`` is refused until the agent is closed/reconstructed, so a
    retry can never run readiness on partial state.
    """

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    READY = "ready"
    FAILED = "failed"


class AgentBootError(RuntimeError):
    """Raised when ``initialize()`` is invoked in a disallowed boot state.

    Two cases:

    * boot is already ``IN_PROGRESS`` — concurrent/re-entrant boot is refused;
    * boot previously ``FAILED`` — the partial state was rolled back and the
      caller must construct a fresh agent (or ``shutdown()`` this one) before
      retrying. This is the explicit "close-required" error the retry contract
      demands — never a silent re-run over partial state.
    """


@dataclass
class _Undo:
    label: str
    action: RollbackAction


@dataclass
class BootPhase:
    """One boot phase: a name, its coroutine body, and its retained resources."""

    name: str
    body: PhaseBody
    #: Human-readable resources this phase deliberately RETAINS on failure
    #: (never rolled back) — e.g. a durable identity graph node that a later
    #: retry reuses. Recorded so the rollback audit distinguishes "released"
    #: from "deliberately kept".
    retained: tuple[str, ...] = ()

    async def run(self, ctx: "BootContext") -> None:
        await self.body(ctx)


class BootContext:
    """Cross-phase carrier plus the reverse-order rollback stack.

    Phases store values other phases need on the context (only
    :attr:`early_agent_node` today) and register an undo action the moment
    they acquire a resource. The undo stack is LIFO: :meth:`run_rollback`
    releases resources in the exact reverse of acquisition order.

    Every resource registered with :meth:`on_rollback` is held in
    :attr:`custody` until a release of it is confirmed (#3522).
    """

    def __init__(
        self,
        logger_: Optional[logging.Logger] = None,
        custody: Optional[ResourceCustody] = None,
    ) -> None:
        self.logger = logger_ or logger
        #: The tracker of held resources. An agent passes its own, so what a
        #: rollback could not release stays held for its shutdown.
        self.custody = custody if custody is not None else ResourceCustody()
        self._undo: List[_Undo] = []
        #: Phases that fully committed, in order (the journal).
        self.committed_phases: List[str] = []
        #: Resources deliberately retained across a failure (audit trail).
        self.retained_resources: List[str] = []
        # ---- cross-phase carry -------------------------------------------
        #: The agent identity graph node loaded in the storage phase and read
        #: again by the providers/sync phase (constitution-anchor gate).
        self.early_agent_node: object = None

    def on_rollback(
        self, label: str, action: RollbackAction, owner: Optional[str] = None
    ) -> None:
        """Register an undo action for a resource this phase just acquired.

        The resource is held in :attr:`custody` under ``label``, released by
        ``owner`` (default: ``label``), until a release of it is confirmed.
        """
        self.custody.acquire(label, owner)
        self._undo.append(_Undo(label, action))

    def on_rollback_unheld(self, label: str, action: RollbackAction) -> None:
        """Register an undo for something :attr:`custody` does not hold.

        For a collection whose members enter custody one by one, such as an
        agent's features, and for registrations that live only in the agent
        object, such as its signal sources. The undo's outcome is reported
        like any other.
        """
        self._undo.append(_Undo(label, action))

    def note_retained(self, resource: str) -> None:
        self.retained_resources.append(resource)

    @property
    def rollback_labels(self) -> List[str]:
        """Labels of the pending undo actions, acquisition order (for tests)."""
        return [u.label for u in self._undo]

    async def _unwind(self) -> List[str]:
        """Pop and run every undo LIFO, tolerating per-step failure/cancel.

        A step that raises, is cancelled, or whose release custody cannot
        confirm leaves its resource held and is listed in
        :attr:`retained_resources`; the unwind goes on with the rest.
        """
        released: List[str] = []
        while self._undo:
            undo = self._undo.pop()
            try:
                outcome = await self.custody.release(undo.label, undo.action)
            except asyncio.CancelledError:
                # Teardown must complete even under cancellation — record and
                # keep going rather than abandoning the remaining resources.
                self.logger.warning(
                    "boot rollback step '%s' was cancelled; continuing teardown",
                    undo.label,
                )
                self.note_retained(undo.label)
                released.append(f"{undo.label} (cancelled)")
                continue
            except Exception as exc:  # noqa: BLE001 - teardown is best-effort
                self.logger.warning(
                    "boot rollback step '%s' failed: %s",
                    undo.label,
                    exc,
                    exc_info=True,
                )
                self.note_retained(undo.label)
                released.append(f"{undo.label} (error)")
                continue
            if outcome is ReleaseOutcome.RELEASED:
                released.append(undo.label)
                continue
            self.note_retained(undo.label)
            released.append(f"{undo.label} ({outcome.value})")
        return released

    async def run_rollback(self) -> List[str]:
        """Unwind the rollback stack to completion, even under cancellation.

        The unwind runs as a separate task awaited via :func:`asyncio.wait`,
        which — unlike :func:`asyncio.wait_for` — does NOT cancel its inner
        task when *our own* await is cancelled. So a boot cancelled mid-phase
        still releases every acquired resource before the ``CancelledError``
        propagates out of :func:`run_boot_sequence`.
        """
        task = asyncio.ensure_future(self._unwind())
        while True:
            try:
                done, _pending = await asyncio.wait({task})
                return task.result() if task in done else []
            except asyncio.CancelledError:
                if task.done():
                    # Unwind already finished; swallow so the orchestrator can
                    # re-raise the ORIGINAL boot exception (which may itself be
                    # the CancelledError) after a complete teardown.
                    return task.result()
                # Otherwise keep waiting for the shielded unwind to finish.
                continue


async def run_boot_sequence(
    phases: List[BootPhase],
    ctx: BootContext,
    set_state: StateSetter,
) -> None:
    """Run ``phases`` in order as an explicit, rollback-safe state machine.

    * Sets ``IN_PROGRESS`` up front.
    * Runs each phase; on success records it in the journal and notes its
      retained resources.
    * On ANY exception (including :class:`asyncio.CancelledError`) unwinds the
      rollback stack to completion, marks the agent ``FAILED``, and re-raises.
    * Sets ``READY`` only after every phase commits — readiness can never fire
      on partial state.
    """
    set_state(BootPhaseState.IN_PROGRESS)
    try:
        for phase in phases:
            ctx.logger.debug("boot: entering phase '%s'", phase.name)
            await phase.run(ctx)
            ctx.committed_phases.append(phase.name)
            for resource in phase.retained:
                ctx.note_retained(resource)
            ctx.logger.debug("boot: committed phase '%s'", phase.name)
    except BaseException as exc:  # noqa: BLE001 - includes CancelledError
        failed_after = ctx.committed_phases[-1] if ctx.committed_phases else "<none>"
        released = await ctx.run_rollback()
        set_state(BootPhaseState.FAILED)
        ctx.logger.warning(
            "agent boot FAILED (last committed phase: %s; %r). "
            "Rolled back %d resource(s): %s. Retained: %s.",
            failed_after,
            exc,
            len(released),
            released,
            ctx.retained_resources or "none",
        )
        raise
    set_state(BootPhaseState.READY)
