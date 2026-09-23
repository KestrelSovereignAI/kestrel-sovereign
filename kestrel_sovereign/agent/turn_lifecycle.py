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

import asyncio
import contextvars
import logging
import time
import warnings
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Iterator, Optional
from uuid import uuid4

from kestrel_sdk.signals import CausationFrame, ResourceLock

from kestrel_sovereign.agent.invocation import (
    InvocationCancelledError,
    current_invocation_id,
    invocation_log_correlation,
    validate_invocation_id,
)
from kestrel_sovereign.signals import OrderedLockManager
from kestrel_sovereign.telemetry import (
    KESTREL_TURN_ID,
    current_turn_id as telemetry_current_turn_id,
    span_trace_identity,
    turn_span_scope,
)
from kestrel_sovereign.turn_scope import turn_scoped

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


@contextmanager
def bind_current_chain(chain: Optional[list[CausationFrame]]) -> Iterator[None]:
    """Re-present a captured causation chain on a task that predates the turn.

    The dispatcher publishes the chain on its own task before entering the
    turn, so work the turn runs on an older task (the codex app-server reader)
    would otherwise read an empty chain and emit signals / outbound A2A tasks
    with no lineage (#3114). The value is copied so the foreign task cannot
    mutate the turn's list.
    """
    token = _CURRENT_CHAIN.set(list(chain) if chain else [])
    try:
        yield
    finally:
        _CURRENT_CHAIN.reset(token)


turn_scoped(
    "causation_chain",
    variables=(_CURRENT_CHAIN,),
    capture=lambda _agent: list(_CURRENT_CHAIN.get()),
    bind=bind_current_chain,
)


@dataclass(frozen=True)
class _TurnSessionBinding:
    """Authority to act as part of one agent's live turn.

    Two kinds exist, and they are deliberately distinguishable:

    - ``lifecycle=True`` is published by ``_active_turn_scope`` itself on the
      task that enters the turn.  Genuine descendants of that task inherit it
      by ordinary ContextVar copy.  Its session resolves live from the agent.
    - ``lifecycle=False`` is a pair captured on the owning turn with
      :func:`capture_turn_session_binding` and explicitly re-presented on a
      foreign task with :func:`bind_turn_session`.

    Every live-turn gate reads this binding and never the raw turn id.  The
    raw id is observability and is carried onto foreign tasks freely (#3114);
    a task that holds only a copied turn id — without this pairing — is not
    admitted as part of the turn.
    """

    agent: object
    turn_id: Optional[str]
    session_id: Optional[str]
    lifecycle: bool = False


_BOUND_TURN_SESSION: contextvars.ContextVar[Optional[_TurnSessionBinding]] = (
    contextvars.ContextVar("kestrel_agent_bound_turn_session", default=None)
)


@dataclass(slots=True)
class _FeatureTransitionAncestry:
    """Task-tree capability for one CONVERSATION-owned feature transition."""

    agent: object
    owner_task: object | None
    active: bool = True


_FEATURE_TRANSITION_ANCESTRY: contextvars.ContextVar[
    Optional[_FeatureTransitionAncestry]
] = contextvars.ContextVar("kestrel_feature_transition_ancestry", default=None)
_COMMITTED_FEATURE_TRANSITION_AGENT: contextvars.ContextVar[object | None] = (
    contextvars.ContextVar(
        "kestrel_committed_feature_transition_agent",
        default=None,
    )
)


def _normalize_session_id(session_id: object) -> Optional[str]:
    if isinstance(session_id, str) and session_id.strip():
        return session_id.strip()
    return None


def _live_turn_binding(agent: object) -> Optional[_TurnSessionBinding]:
    """The calling task's binding to ``agent``'s LIVE turn, or ``None``.

    The single ownership test behind every live-turn gate.  It requires the
    explicit ``_BOUND_TURN_SESSION`` pairing for this agent and that the paired
    turn is the one holding CONVERSATION right now.  A task that merely carries
    a copied turn id — a detached descendant of a finished turn, or a foreign
    task onto which the turn id was carried for observability — does not pass.
    """
    propagated = _BOUND_TURN_SESSION.get()
    if (
        propagated is None
        or propagated.agent is not agent
        or not propagated.turn_id
        or propagated.turn_id != getattr(agent, "_live_turn_id", None)
    ):
        return None
    return propagated


@contextmanager
def publish_turn_ownership(agent: object, turn_id: str) -> Iterator[None]:
    """Publish the lifecycle's own binding for ``turn_id`` on this task.

    Used by ``_active_turn_scope``; a task that enters a turn owns it outright,
    so this also supersedes any binding it inherited from the callback or tool
    that spawned it.
    """
    token = _BOUND_TURN_SESSION.set(
        _TurnSessionBinding(agent, turn_id, None, lifecycle=True)
    )
    try:
        yield
    finally:
        _BOUND_TURN_SESSION.reset(token)


def capture_turn_session_binding(agent: object) -> _TurnSessionBinding:
    """Capture ``agent``'s authoritative live-turn session for later binding.

    Context variables are copied when a task is created.  Long-lived transport
    readers therefore cannot see a turn that began after the reader task.  A
    caller on the owning turn uses this helper while building its callback,
    then :func:`bind_turn_session` re-presents the captured pair while the
    callback runs on the reader task.

    The capture never derives authority from an arbitrary transport argument,
    logging context, the raw turn id, or the agent-global session alone.  It
    either preserves an already-captured live pair (for nested task
    boundaries) or, on a task carrying the lifecycle's own binding, resolves
    the session through the lifecycle accessor.  An explicit binding for this
    agent takes precedence even when it is unbound or stale: callback code must
    not replace its captured authority with an ambient turn copied into the
    task that happens to invoke it.  Entering a new turn replaces any inherited
    binding, because the task then owns that turn outright.  An out-of-turn or
    session-less capture is represented explicitly as unbound.
    """
    live = _live_turn_binding(agent)
    if live is None:
        return _TurnSessionBinding(agent, None, None)
    if not live.lifecycle:
        return live

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
        agent, live.turn_id, _normalize_session_id(session_id)
    )


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


turn_scoped(
    "turn_session",
    variables=(_BOUND_TURN_SESSION,),
    capture=capture_turn_session_binding,
    bind=bind_turn_session,
)


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

    def get_current_turn_id(self) -> Optional[str]:
        """Return the canonical cooperative-Stop address of this task's turn.

        A value for attribution (todo metadata, ``origin_turn_id``, dispatch
        logs, span stamping). Tool executors carry it onto the foreign tasks
        they run turn work on (#3114), so it is not evidence that the caller
        owns the live turn; the live-turn gates use ``_live_turn_binding``.
        """

        return telemetry_current_turn_id()

    def _get_current_turn_id(self) -> Optional[str]:
        """Compatibility alias for callers predating the public accessor."""

        return self.get_current_turn_id()

    def _turn_request_index(self) -> dict[str, tuple[str, int | None]]:
        """Return the live turn-to-request index, creating it for test doubles."""

        index = getattr(self, "_turn_request_ids", None)
        if index is None:
            index = {}
            self._turn_request_ids = index
        if not isinstance(index, dict):
            raise TypeError("turn request index has an invalid type")
        return index

    def _register_turn_request_id(
        self,
        turn_id: str,
        request_id: str,
        generation: int | None,
    ) -> None:
        """Bind one freshly-created observable turn to its cancellation key."""

        for field_name, value in (("turn_id", turn_id), ("request_id", request_id)):
            try:
                validate_invocation_id(value)
            except ValueError as error:
                raise ValueError(
                    f"{field_name} must be a concrete string"
                ) from error
        if generation is not None and (
            not isinstance(generation, int)
            or isinstance(generation, bool)
            or generation <= 0
        ):
            raise ValueError("request generation must be a positive integer")
        index = self._turn_request_index()
        if turn_id in index:
            raise RuntimeError("turn_id is already bound to a request")
        index[turn_id] = (request_id, generation)

    def resolve_turn_request_id(self, turn_id: str) -> Optional[str]:
        """Resolve an active observable turn to its process-local cancel key."""

        if not isinstance(turn_id, str) or not turn_id.strip():
            return None
        binding = self._turn_request_index().get(turn_id)
        if binding is None:
            return None
        if (
            not isinstance(binding, tuple)
            or len(binding) != 2
            or not isinstance(binding[0], str)
            or not binding[0]
        ):
            raise TypeError("turn request index contains an invalid request identity")
        return binding[0]

    def active_turn_request_bindings(
        self,
    ) -> dict[str, tuple[str, int | None]]:
        """Snapshot exact live turn cancellation addresses atomically."""

        return dict(self._turn_request_index())

    def active_turn_request_ids(self) -> dict[str, str]:
        """Snapshot live observable-turn addresses for cancellation inventory."""

        return {
            turn_id: binding[0]
            for turn_id, binding in self.active_turn_request_bindings().items()
        }

    def _turn_trace_index(self) -> dict[str, tuple[str, str]]:
        """Return live turn-to-span correlations, creating it for test doubles."""

        index = getattr(self, "_turn_trace_identities", None)
        if index is None:
            index = {}
            self._turn_trace_identities = index
        if not isinstance(index, dict):
            raise TypeError("turn trace identity index has an invalid type")
        return index

    def bind_current_turn_trace_identity(
        self,
        trace_id: str,
        span_id: str,
    ) -> bool:
        """Correlate the live canonical turn with one observable turn span.

        The mapping is evidence only. Cancellation still resolves exclusively
        through the lifecycle-owned turn-to-request index. Only a task paired
        with the live turn (``_live_turn_binding``) may bind it: the raw turn
        id is carried onto foreign tasks for observability and is not
        ownership (#3114).
        """

        live = _live_turn_binding(self)
        if live is None:
            return False
        turn_id = live.turn_id
        for field_name, value, length in (
            ("trace_id", trace_id, 32),
            ("span_id", span_id, 16),
        ):
            if (
                not isinstance(value, str)
                or len(value) != length
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{field_name} must be a lowercase W3C hex identity")
        self._turn_trace_index()[turn_id] = (trace_id, span_id)
        return True

    def bind_current_turn_span(self, span: object) -> bool:
        """Bind a concrete OTel span to the live turn when it is valid.

        Gated like :meth:`bind_current_turn_trace_identity` on the explicit
        live-turn pairing, never on the raw turn id alone.
        """

        live = _live_turn_binding(self)
        if live is None:
            return False
        turn_id = live.turn_id
        set_attribute = getattr(span, "set_attribute", None)
        if callable(set_attribute):
            set_attribute(KESTREL_TURN_ID, turn_id)
        trace_id, span_id = span_trace_identity(span)
        if trace_id is None or span_id is None:
            return False
        return self.bind_current_turn_trace_identity(trace_id, span_id)

    def active_turn_trace_identities(self) -> dict[str, tuple[str, str]]:
        """Snapshot optional observability correlations for live turns."""

        return dict(self._turn_trace_index())

    def _turn_outcome_listener_registry(self) -> list:
        """Return the outcome listener list, creating it for test doubles."""

        listeners = getattr(self, "_turn_outcome_listeners", None)
        if listeners is None:
            listeners = []
            self._turn_outcome_listeners = listeners
        if not isinstance(listeners, list):
            raise TypeError("turn outcome listener registry has an invalid type")
        return listeners

    def add_turn_outcome_listener(self, listener) -> None:
        """Subscribe a feature-owned turn root to this agent's turn outcomes.

        The contract (#3159), for the consumer kestrel-feature-observability
        #118, which reaches it duck-typed and never imports core:

        * **When.** ``listener(turn_id, outcome)`` is called exactly once per
          turn whose lifecycle minted a turn address, on EVERY exit — normal
          return, early return, exception, ``CancelledError``, and the stream
          close paths that skip the SDK ``Stop`` hook, which is why a feature
          cannot derive this for itself. It runs in the turn entry point's own
          ``finally``, after the core turn span carries the same outcome. A
          turn refused before its address existed has no ``turn_id`` and is
          not published.
        * **Synchronous.** The call is not awaited, because it also runs while
          ``GeneratorExit`` propagates, where an ``await`` would raise. A
          listener must not block and must not schedule work on the turn.
        * **Never raises into the turn.** An exception from a listener is
          logged and swallowed; the turn's own exit is unchanged.

        ``outcome`` is a ``str`` enum member, so a consumer may compare it to
        the plain strings (``completed``/``failed``/``stopped``/
        ``disconnected``/``interrupted``) without importing core.
        """

        if not callable(listener):
            raise TypeError("turn outcome listener must be callable")
        listeners = self._turn_outcome_listener_registry()
        if listener not in listeners:
            listeners.append(listener)

    def remove_turn_outcome_listener(self, listener) -> None:
        """Unsubscribe a listener; unknown listeners are a no-op."""

        listeners = self._turn_outcome_listener_registry()
        if listener in listeners:
            listeners.remove(listener)

    def _unregister_turn_request_id(
        self,
        turn_id: str,
        request_id: str,
        generation: int | None,
    ) -> None:
        """Remove only the exact lifecycle binding that this turn registered."""

        index = self._turn_request_index()
        current = index.get(turn_id)
        if current is None:
            return
        if current != (request_id, generation):
            raise RuntimeError("turn request cleanup does not own the live binding")
        del index[turn_id]

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
        - A task-local turn marker is COPIED into child tasks at creation. A
          task detached from turn A therefore keeps reporting turn A forever,
          including long after A exited. So a truthy marker does not mean "a
          turn is live", only "this task was born inside one".

        Pairing them closes both: `_live_turn_id` is the agent-scoped mirror of
        *which turn holds the CONVERSATION lock right now*, and the task-local
        marker is the lifecycle's own `_BOUND_TURN_SESSION` binding published
        at turn entry, so requiring the two to agree means the caller belongs
        to the live turn and the session it is reading is that turn's own. The
        raw `_CURRENT_TURN_ID` is deliberately NOT consulted: it is carried
        onto foreign tasks for observability (#3114) and is not ownership. A
        transport callback may also explicitly carry a pair captured through
        this accessor across a known task boundary; the pair is accepted only
        while that exact turn remains live. When present, that explicit binding
        is authoritative even if it is unbound or stale; an executor captured
        off-turn cannot borrow an unrelated ambient turn merely because its
        reader task copied that turn's context. A task that enters
        `_turn_lifecycle` publishes its own binding and owns the new turn
        outright. The detached task from turn A sees `A != B` while turn B
        runs, and `None` after A ended — both resolve to None, i.e. "no chat
        window", which callers treat as system-initiated.

        Returns None outside a turn, for a session-less turn, and for any task
        that merely inherited a finished turn's context.
        """
        live = _live_turn_binding(self)
        if live is None:
            return None
        if live.lifecycle:
            return _normalize_session_id(
                getattr(self, "_active_session_id", None)
            )
        return _normalize_session_id(live.session_id)

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

    async def _await_host_context_publication(self) -> None:
        """Wait until server startup has published host-owned prompt policy.

        A standalone scheduler (and other ready hooks) can wake during
        ``KestrelAgent.initialize()`` while the host feature lifecycle is still
        being assembled by the server.  The server installs one shared event
        before initialization and sets it only after the host context registry
        has been bound.  Multi-agent initialization can still be absent from
        the manager's fan-out at that instant, so gate release also reconciles
        the shared publication generation before cognition may continue.
        Directly-created/test agents have no shared state and retain their
        established behavior.
        """

        gate = getattr(self, "_host_context_publication_gate", None)
        if gate is not None and not gate.is_set():
            await gate.wait()
        self._synchronize_host_context_publication()

    def _synchronize_host_context_publication(self) -> None:
        """Bind the manager's latest host registry at the cognition barrier.

        An agent is deliberately not routable until ``initialize()`` returns,
        but its ready hooks can start cognition during initialization.  A
        manager fan-out therefore cannot be the only publication mechanism:
        the still-unregistered agent may be waiting on the same event that the
        host is about to release.  The manager shares a generation box with
        every constructed agent; this synchronous check makes rebinding atomic
        with leaving the gate and is also safe when a late cold wake observes
        an already-set gate.
        """

        state = getattr(self, "_host_context_publication_state", None)
        if state is None:
            return
        generation = getattr(state, "generation", None)
        if type(generation) is not int or generation < 0:
            raise RuntimeError("host context publication state is invalid")
        if getattr(self, "_host_context_publication_generation", None) == generation:
            return

        validate_registry = getattr(
            self, "validate_host_context_clause_registry", None
        )
        bind_registry = getattr(self, "bind_host_context_clause_registry", None)
        if not callable(validate_registry) or not callable(bind_registry):
            raise RuntimeError("agent cannot synchronize host context publication")

        registry = getattr(state, "registry", None)
        validate_registry(registry)
        bind_registry(registry)
        self._host_context_publication_generation = generation

    @asynccontextmanager
    async def feature_config_transition(self) -> AsyncIterator[None]:
        """Serialize one config/context transition with cognition turns.

        Prompts consume an immutable rendered clause snapshot while feature
        tools consume the feature's live config.  Holding the same
        ``CONVERSATION`` resource as a turn prevents either half of a turn from
        observing a different config generation.
        """

        mgr = self._get_lock_manager()
        label = f"{getattr(self, 'agent_name', None) or 'agent'} feature-config"
        async with mgr.acquire({ResourceLock.CONVERSATION}, label=label):
            ancestry = _FeatureTransitionAncestry(
                agent=self,
                owner_task=asyncio.current_task(),
            )
            token = _FEATURE_TRANSITION_ANCESTRY.set(ancestry)
            try:
                yield
            finally:
                # A task created by an uncommitted hook inherits the same object.
                # Invalidating it before restoring this task's ContextVar stops a
                # detached descendant from waiting until commit and laundering
                # its pre-commit authority into a later cognition turn.
                ancestry.active = False
                _FEATURE_TRANSITION_ANCESTRY.reset(token)

    @contextmanager
    def committed_feature_transition_cognition(self) -> Iterator[None]:
        """Admit same-task cognition after a feature generation commits.

        Feature mutation hooks run while their owner holds ``CONVERSATION``.
        Re-entering cognition from an arbitrary hook would expose whichever
        subset of config, clauses, tools, and enablement that hook has already
        changed.  Only a lifecycle seam that has completed the whole commit may
        opt in here; currently that is runtime ``on_agent_ready``.
        """

        token = _COMMITTED_FEATURE_TRANSITION_AGENT.set(self)
        try:
            yield
        finally:
            _COMMITTED_FEATURE_TRANSITION_AGENT.reset(token)

    def _capture_committed_feature_transition_delegation(
        self,
    ) -> _FeatureTransitionAncestry | None:
        """Capture authority that an isolated invocation child may re-own."""

        ancestry = _FEATURE_TRANSITION_ANCESTRY.get()
        if (
            ancestry is None
            or ancestry.agent is not self
            or not ancestry.active
            or ancestry.owner_task is not asyncio.current_task()
            or _COMMITTED_FEATURE_TRANSITION_AGENT.get() is not self
        ):
            return None
        return ancestry

    @contextmanager
    def _bind_committed_feature_transition_delegation(
        self,
        ancestry: _FeatureTransitionAncestry,
    ) -> Iterator[None]:
        """Re-own captured committed-transition authority in one child task."""

        if (
            not isinstance(ancestry, _FeatureTransitionAncestry)
            or ancestry.agent is not self
            or not ancestry.active
            or _COMMITTED_FEATURE_TRANSITION_AGENT.get() is not self
        ):
            raise RuntimeError("committed feature-transition authority expired")
        delegated = _FeatureTransitionAncestry(
            agent=self,
            owner_task=asyncio.current_task(),
        )
        token = _FEATURE_TRANSITION_ANCESTRY.set(delegated)
        try:
            yield
        finally:
            delegated.active = False
            _FEATURE_TRANSITION_ANCESTRY.reset(token)

    def _caller_belongs_to_live_turn(self) -> bool:
        """Whether this task is executing as part of this agent's live turn.

        The turn owner is authoritative.  A provider callback that runs on a
        reader-spawned task is also admitted only when it carries the explicit
        binding captured by :func:`capture_turn_session_binding`; a detached
        task that merely inherited the turn id, or the lifecycle's own binding
        by ordinary task creation, is not allowed to bypass the conversation
        lock.
        """

        if asyncio.current_task() is getattr(self, "_live_turn_task", None):
            return True
        live = _live_turn_binding(self)
        return live is not None and not live.lifecycle

    @asynccontextmanager
    async def privacy_transition(self) -> AsyncIterator[None]:
        """Serialize a privacy transition with complete cognition turns.

        External callers acquire ``CONVERSATION`` before the privacy mutex, so
        a prompt assembled under the old privacy policy cannot remain in flight
        after a restrictive transition reports success.  An in-turn command or
        explicitly-bound provider tool already belongs to the live turn and
        therefore acquires only the task-reentrant privacy mutex, avoiding a
        recursive ``CONVERSATION`` deadlock.  The global lock order remains
        CONVERSATION -> privacy everywhere.
        """

        transition_lock = self._get_privacy_transition_lock()
        mgr = self._get_lock_manager()
        if self._caller_belongs_to_live_turn() or mgr.is_owned_by_current_task(
            ResourceLock.CONVERSATION
        ):
            async with transition_lock:
                yield
            return

        label = f"{getattr(self, 'agent_name', None) or 'agent'} privacy-transition"
        async with mgr.acquire({ResourceLock.CONVERSATION}, label=label):
            async with transition_lock:
                yield

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
        await self._await_host_context_publication()
        # This identifier is now a durable public Stop address, not a log-only
        # convenience token.  Keep the full UUID entropy so fleet-scale turns
        # cannot collide onto the same cancellation target.
        turn_id = f"turn_{uuid4().hex}"
        mgr = self._get_lock_manager()
        label = f"{getattr(self, 'agent_name', None) or 'agent'} {turn_id}"
        started = time.monotonic()
        holder = mgr.holder(ResourceLock.CONVERSATION)
        current_task = asyncio.current_task()
        transition_ancestry = _FEATURE_TRANSITION_ANCESTRY.get()
        if (
            transition_ancestry is not None
            and transition_ancestry.agent is self
        ):
            if not transition_ancestry.active:
                raise RuntimeError(
                    "cognition cannot start from an expired feature transition"
                )
            if transition_ancestry.owner_task is not current_task:
                if _COMMITTED_FEATURE_TRANSITION_AGENT.get() is self:
                    raise RuntimeError(
                        "committed feature-transition cognition cannot cross "
                        "a task boundary"
                    )
                raise RuntimeError(
                    "cognition cannot start before the feature transition "
                    "generation is fully committed"
                )
            if _COMMITTED_FEATURE_TRANSITION_AGENT.get() is not self:
                raise RuntimeError(
                    "cognition cannot start before the feature transition "
                    "generation is fully committed"
                )
        if (
            holder is not None
            and mgr.is_owned_by_current_task(ResourceLock.CONVERSATION)
            and current_task is not getattr(self, "_live_turn_task", None)
        ):
            if _COMMITTED_FEATURE_TRANSITION_AGENT.get() is not self:
                raise RuntimeError(
                    "cognition cannot start before the feature transition "
                    "generation is fully committed"
                )
            # The committed ready phase still owns CONVERSATION in this task,
            # or has explicitly delegated that exact hold to the isolated
            # invocation child. Reuse that boundary; an arbitrary mid-transition
            # hook is rejected above, and a genuine live turn is excluded so
            # recursive process_input cannot replace the outer turn's authority.
            async with self._active_turn_scope(turn_id, label, started):
                yield turn_id
            return

        async with mgr.acquire({ResourceLock.CONVERSATION}, label=label):
            async with self._active_turn_scope(turn_id, label, started):
                yield turn_id

    @asynccontextmanager
    async def _active_turn_scope(
        self,
        turn_id: str,
        label: str,
        started: float,
    ) -> AsyncIterator[None]:
        """Publish one live turn inside an already-owned conversation bound."""

        logger.info("turn_lifecycle: %s begin", label)
        turn_scope = turn_span_scope(turn_id)
        turn_scope.__enter__()
        # Agent-scoped mirror of "which turn is LIVE" — i.e. which one holds
        # the CONVERSATION lock and therefore owns `_active_session_id`.
        self._live_turn_id = turn_id
        self._live_turn_task = asyncio.current_task()
        # A background task created inside an explicitly-bound callback inherits
        # that binding. New turn entry supersedes it with the lifecycle's own
        # binding, which is what every live-turn gate consults.
        ownership = publish_turn_ownership(self, turn_id)
        ownership.__enter__()
        request_id = current_invocation_id()
        request_generation = None
        request_binding_registered = False
        try:
            if request_id is not None:
                generation_accessor = getattr(
                    self,
                    "_request_generation_for_current_task",
                    None,
                )
                if callable(generation_accessor):
                    request_generation = generation_accessor(request_id)
                self._register_turn_request_id(
                    turn_id,
                    request_id,
                    request_generation,
                )
                request_binding_registered = True
                await_turn_admission = getattr(
                    self,
                    "await_durable_turn_admission",
                    None,
                )
                if callable(await_turn_admission):
                    durable_binding_admitted = await await_turn_admission(
                        turn_id,
                        request_id,
                        request_generation,
                    )
                    if not durable_binding_admitted:
                        raise InvocationCancelledError(
                            "turn was stopped before durable admission "
                            f"({invocation_log_correlation(turn_id)})"
                        )
            yield
        finally:
            try:
                if request_binding_registered:
                    self._unregister_turn_request_id(
                        turn_id,
                        request_id,
                        request_generation,
                    )
            finally:
                ownership.__exit__(None, None, None)
                self._turn_trace_index().pop(turn_id, None)
                turn_scope.__exit__(None, None, None)
                self._live_turn_id = None
                self._live_turn_task = None
                # An out-of-turn caller must never reuse a stale chat session.
                self._active_session_id = None
                logger.info(
                    "turn_lifecycle: %s end after %.1fs",
                    label,
                    time.monotonic() - started,
                )
