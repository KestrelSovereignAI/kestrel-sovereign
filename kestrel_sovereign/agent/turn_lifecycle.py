"""Shared turn lifecycle for non-streaming and streaming entry points.

Per SIGNAL_DISPATCHER.md §Concern 1: `process_input` and
`process_input_streaming` both write conversation history with no
serialization today; two turns can interleave. Heartbeat ([heartbeat.py:247])
fires `process_input` without checking whether a user turn is in flight.

This mixin provides the **single boundary** where the `CONVERSATION` lock
is acquired/released. Both entry points wrap their inner traced bodies in
`async with self._turn_lifecycle():`. The lock manager is the same
instance the SignalDispatcher will use for its registered resource locks
(Phase 1 — already shipped via the SDK), so cross-system invariants hold.

The dispatcher does NOT pre-acquire `CONVERSATION` for COGNITION sources
(Phase 1 §Concern 2). The turn lifecycle here is the sole owner.
"""

from __future__ import annotations

import contextvars
import logging
import time
import warnings
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Iterator, Optional
from uuid import uuid4

from kestrel_sdk.signals import CausationFrame, ResourceLock
from kestrel_sovereign.signals import OrderedLockManager

logger = logging.getLogger(__name__)


# Per-task storage for the in-flight cognition turn's causation chain.
# Using `contextvars.ContextVar` instead of an agent attribute closes
# the race the #906 review caught: when two COGNITION signals dispatch
# concurrently, an agent-level `_current_chain` attribute can be
# overwritten by signal B before signal A's turn actually enters its
# CONVERSATION-locked body. ContextVar is task-local, so each
# dispatch's chain only flows down its own asyncio.Task tree (which
# is exactly what we want — `TaskManager.create_task` runs inside the
# turn's task and reads the right value via the provider).
#
# Default `[]` means "no signal-driven chain" — direct HTTP user
# input or tests that don't drive cognition see an empty chain
# (the provider returns None for empty chains, so no metadata is
# attached to outbound tasks in those cases).
_CURRENT_CHAIN: contextvars.ContextVar[list[CausationFrame]] = (
    contextvars.ContextVar("kestrel_signals_current_chain", default=[])
)
_CURRENT_TURN_ID: contextvars.ContextVar[Optional[str]] = (
    contextvars.ContextVar("kestrel_agent_current_turn_id", default=None)
)


@dataclass(frozen=True)
class _TurnSessionBinding:
    """A turn/session pair explicitly carried across a task boundary."""

    agent: object
    turn_id: Optional[str]
    session_id: Optional[str]


_BOUND_TURN_SESSION: contextvars.ContextVar[Optional[_TurnSessionBinding]] = (
    contextvars.ContextVar("kestrel_agent_bound_turn_session", default=None)
)


def _normalize_session_id(session_id: object) -> Optional[str]:
    if isinstance(session_id, str) and session_id.strip():
        return session_id.strip()
    return None


def capture_turn_session_binding(agent: object) -> _TurnSessionBinding:
    """Capture ``agent``'s authoritative live-turn session for later binding.

    Context variables are copied when a task is created.  Long-lived transport
    readers therefore cannot see a turn that began after the reader task.  A
    caller on the owning turn uses this helper while building its callback,
    then :func:`bind_turn_session` re-presents the captured pair while the
    callback runs on the reader task.

    The capture never derives authority from an arbitrary transport argument,
    logging context, or the agent-global session alone.  It either preserves an
    already-bound live pair (for nested task boundaries) or asks the lifecycle
    accessor while the calling task owns the live turn.  An explicit binding
    for this agent takes precedence even when it is unbound or stale: callback
    code must not replace its captured authority with an ambient turn copied
    into the task that happens to invoke it.  Entering a new turn clears any
    inherited binding, because the task then owns that turn outright.  An
    out-of-turn or session-less capture is represented explicitly as unbound.
    """
    live_turn_id = getattr(agent, "_live_turn_id", None)
    propagated = _BOUND_TURN_SESSION.get()
    if propagated is not None and propagated.agent is agent:
        if propagated.turn_id and propagated.turn_id == live_turn_id:
            return propagated
        return _TurnSessionBinding(agent, None, None)

    turn_id = _CURRENT_TURN_ID.get()
    if turn_id and turn_id == live_turn_id:
        resolve = getattr(agent, "get_turn_bound_session_id", None)
        if not callable(resolve):
            # Compatibility for agent shapes from the 0.53 -> 0.54 migration
            # window. Keep capture aligned with Feature._turn_session_id's
            # direct-read resolution until the private alias is removed.
            resolve = getattr(agent, "_get_turn_bound_session_id", None)
        try:
            session_id = resolve() if callable(resolve) else None
        except Exception:  # noqa: BLE001 - unknown host shapes stay unbound
            session_id = None
        return _TurnSessionBinding(
            agent, turn_id, _normalize_session_id(session_id)
        )

    return _TurnSessionBinding(agent, None, None)


@contextmanager
def bind_turn_session(
    binding: _TurnSessionBinding,
) -> Iterator[None]:
    """Temporarily expose a captured lifecycle binding in the current task."""
    token = _BOUND_TURN_SESSION.set(binding)
    try:
        yield
    finally:
        _BOUND_TURN_SESSION.reset(token)


class TurnLifecycleMixin:
    """Provides `_turn_lifecycle` and the per-agent state it needs.

    `KestrelAgent.__init__` initializes `self._lock_manager`; the
    `_get_lock_manager` accessor lazy-creates one for tests/callers that
    bypass `__init__` via `KestrelAgent.__new__` (mirrors the existing
    `_get_privacy_transition_lock` pattern in `kestrel_agent.py`).

    The in-flight cognition turn's causation chain lives in a module-
    level `ContextVar` (see `_CURRENT_CHAIN` above). The dispatcher
    `_set_current_chain` before invoking `process_input` for a
    COGNITION signal so any outbound A2A tasks created during the turn
    (via `TaskManager.create_task`) can carry the chain forward in
    their metadata. Without this, A→B→A loops would restart at depth
    1 every iteration and dispatcher cycle detection would never fire
    (#905 review P1). Using a ContextVar instead of an agent attribute
    keeps concurrent dispatches isolated from each other (#906 review
    P1 — concurrent dispatchers could overwrite an agent-level
    attribute before the woken turn actually entered the CONVERSATION
    lock).
    """

    _lock_manager: OrderedLockManager

    def _get_lock_manager(self) -> OrderedLockManager:
        """Return the shared OrderedLockManager, lazy-creating one if the
        owning class skipped __init__ (tests using `KestrelAgent.__new__`)."""
        mgr = getattr(self, "_lock_manager", None)
        if mgr is None:
            mgr = OrderedLockManager()
            self._lock_manager = mgr
        return mgr

    def _get_current_chain(self) -> Optional[list[CausationFrame]]:
        """Return the in-flight turn's causation chain, or None when no
        cognition signal triggered the current turn (e.g. direct HTTP
        user input). Reads the per-task ContextVar.

        Returns None for empty chains so callers (TaskManager provider)
        can use truthiness without a separate `len` check.
        """
        chain = _CURRENT_CHAIN.get()
        return chain if chain else None

    def _get_current_turn_id(self) -> Optional[str]:
        """Return the current agent turn id for per-turn observability."""
        return _CURRENT_TURN_ID.get()

    def get_turn_bound_session_id(self) -> Optional[str]:
        """The chat session of the turn the CALLING task belongs to, or None.

        The one honest answer to "which chat window is this code running for",
        and the only safe way to read `_active_session_id` (#2877). Neither
        half is sufficient alone:

        - `_active_session_id` is an agent-global attribute. Read on its own
          from work that is NOT the live turn (a cron ACTION tick, a detached
          background task), it returns whatever *concurrent* chat turn happens
          to be in flight — cross-wiring unattended work into a stranger's
          window.
        - `_CURRENT_TURN_ID` is a ContextVar, and a ContextVar is COPIED into
          child tasks at creation. A task detached from turn A therefore keeps
          reporting turn A's id forever, including long after A exited. So a
          truthy turn id does not mean "a turn is live", only "this task was
          born inside one".

        Pairing them closes both: `_live_turn_id` is the agent-scoped mirror of
        *which turn holds the CONVERSATION lock right now*, so requiring the
        task-local id to equal it means the caller owns the live turn and the
        session it is reading is that turn's own. A transport callback may also
        explicitly carry a pair captured through this accessor across a known
        task boundary; the pair is accepted only while that exact turn remains
        live. When present, that explicit binding is authoritative even if it
        is unbound or stale; an executor captured off-turn cannot borrow an
        unrelated ambient turn merely because its reader task copied that
        turn's ContextVar. A task that enters `_turn_lifecycle` clears any
        inherited binding and owns the new turn outright. The detached task
        from turn A sees `A != B` while turn B runs, and `None` after A ended —
        both resolve to None, i.e. "no chat window", which callers treat as
        system-initiated.

        Returns None outside a turn, for a session-less turn, and for any task
        that merely inherited a finished turn's context.
        """
        live_turn_id = getattr(self, "_live_turn_id", None)
        propagated = _BOUND_TURN_SESSION.get()
        if propagated is not None and propagated.agent is self:
            if propagated.turn_id and propagated.turn_id == live_turn_id:
                return _normalize_session_id(propagated.session_id)
            return None

        turn_id = _CURRENT_TURN_ID.get()
        if turn_id and turn_id == live_turn_id:
            return _normalize_session_id(
                getattr(self, "_active_session_id", None)
            )
        return None

    def _get_turn_bound_session_id(self) -> Optional[str]:
        """Compatibility alias for :meth:`get_turn_bound_session_id`.

        Deprecated in Sovereign 0.53.0 and planned for removal in 0.54.0.
        Feature packages must use the public accessor. The private name remains
        for one compatibility release so older packages and agent doubles can
        migrate without silently losing turn attribution.
        """
        warnings.warn(
            "_get_turn_bound_session_id() is deprecated since Sovereign "
            "0.53.0 and will be removed in 0.54.0; use "
            "get_turn_bound_session_id() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_turn_bound_session_id()

    def _set_current_chain(
        self, chain: Optional[list[CausationFrame]]
    ) -> contextvars.Token:
        """Set the in-flight turn's causation chain.

        Returns a Token the caller MUST pass to `_clear_current_chain`
        in a `finally` block. The token-based reset preserves any
        outer chain context (matters for the rare reentrant case
        where a turn's outbound task triggers another COGNITION
        signal that runs inline)."""
        return _CURRENT_CHAIN.set(list(chain) if chain else [])

    def _clear_current_chain(
        self, token: Optional[contextvars.Token] = None
    ) -> None:
        """Restore the chain context to what it was before the matching
        `_set_current_chain`. If no token is provided (defensive call
        from a path that didn't capture one), reset to the default
        empty chain."""
        if token is not None:
            try:
                _CURRENT_CHAIN.reset(token)
            except (ValueError, LookupError) as e:
                # Token from a different context — best-effort fall
                # back to clearing rather than raising.
                logger.debug(
                    "ContextVar.reset failed (cross-context token?); "
                    "clearing to default: %s", e,
                )
                _CURRENT_CHAIN.set([])
        else:
            _CURRENT_CHAIN.set([])

    @asynccontextmanager
    async def _turn_lifecycle(self) -> AsyncIterator[str]:
        """Enter a turn: acquire CONVERSATION, yield a fresh turn_id,
        release on exit (normal or exception).

        The yielded `turn_id` is opaque to callers today; Phase 5 (#894 —
        A2A causation chain propagation) will plumb it into Signal
        envelopes so dispatcher-driven cognition can mark its CausationFrame
        with the receiving turn.

        Entry and exit log at INFO, and the acquisition carries an agent-scoped
        label. Both exist because of #2770: a turn that stalled inside this
        region left `process_input called` as the last record for that agent, so
        an operator could not tell a working turn from a wedged one, nor which
        turn was holding the lock. Begin/end at DEBUG was invisible in
        production. Two INFO lines per turn is a deliberate trade for a bounded
        region that can otherwise silently hold an agent hostage for minutes.
        """
        turn_id = f"turn_{uuid4().hex[:12]}"
        mgr = self._get_lock_manager()
        label = f"{getattr(self, 'agent_name', None) or 'agent'} {turn_id}"
        started = time.monotonic()
        async with mgr.acquire({ResourceLock.CONVERSATION}, label=label):
            logger.info("turn_lifecycle: %s begin", label)
            token = _CURRENT_TURN_ID.set(turn_id)
            # Agent-scoped mirror of "which turn is LIVE" — i.e. which one
            # holds the CONVERSATION lock and therefore owns the value of
            # `_active_session_id`. Set and cleared inside the lock, so at most
            # one turn is ever live. `get_turn_bound_session_id` compares it
            # against the task-local id to tell a caller that owns the turn
            # from one that merely inherited its ContextVar (#2877).
            self._live_turn_id = turn_id
            # A background task created inside an explicitly-bound callback
            # inherits that binding. If it later enters its own cognition turn
            # (the signal-dispatch wake path), turn entry is the unambiguous
            # ownership boundary: the new turn's lifecycle/session authority
            # supersedes the callback binding it inherited. Passive callbacks
            # never enter this boundary, so their explicit stale/unbound veto
            # remains intact (#2928 review P1).
            bound_token = _BOUND_TURN_SESSION.set(None)
            try:
                yield turn_id
            finally:
                _BOUND_TURN_SESSION.reset(bound_token)
                _CURRENT_TURN_ID.reset(token)
                self._live_turn_id = None
                # Clear the per-turn active session on exit so an out-of-turn
                # caller (e.g. a CLI/system-filed request_restart after a chat
                # turn) cannot read a stale session and misroute its wake into
                # an old chat window (#1809). Set inside the turn body by
                # process_input / the streaming turn; both run under this lock.
                self._active_session_id = None
                # Duration on the exit line so a slow turn is measurable from
                # the log alone, without correlating two timestamps by hand.
                logger.info(
                    "turn_lifecycle: %s end after %.1fs",
                    label,
                    time.monotonic() - started,
                )
