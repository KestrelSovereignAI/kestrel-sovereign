"""Declaration-site registry of turn-scoped context (#3114).

A cognition turn publishes several values in ``ContextVar``\\ s: the part
collector, the causation chain, the turn id, the dispatching signal, the caller
binding, and so on.  A ``ContextVar`` is copied into a task only when that task
is *created*.  Transports that run a turn's tools on a task spawned **before**
the turn published — the codex app-server's long-lived reader loop is the live
case — therefore see every one of those values at its default unless the tool
executor re-presents it.

That mistake was made once per value, in each executor, and fixed one bind at a
time (#2081, #2672, #2965, #3112, #3114).  The executor's ``with`` block was the
only enumeration of "the turn-scoped values", so a new one was covered only by
someone remembering to extend that block in every place it was copied.

This module inverts the ownership.  The module that declares a turn-scoped
``ContextVar`` also declares how to carry it, next to the declaration::

    _CURRENT_CHAIN = contextvars.ContextVar(...)
    turn_scoped(
        "causation_chain",
        variables=(_CURRENT_CHAIN,),
        capture=lambda agent: ...,
        bind=bind_current_chain,
    )

Every closure that runs turn work on a foreign task captures the whole set with
:func:`capture_turn_scope` on the owning task and re-presents it with
:meth:`TurnScopeSnapshot.bind`.  A new turn-scoped value is covered the moment
its declaration registers; no executor changes.

Each carrier keeps its own semantics.  ``capture`` and ``bind`` are the
declaring module's own functions, so a value that is an authority grant (the
turn/session binding, the revocable caller binding, the transition-lock
reentry token) keeps its validation and revocation rules; a plain observability
value (the turn id, the causation chain) is copied as-is.

Registration happens at import of the declaring module.  That is sufficient: a
``ContextVar`` cannot hold a non-default value before the module that creates it
has run, so a carrier that is not yet registered has nothing to carry.
"""

from __future__ import annotations

import contextlib
import contextvars
import threading
from dataclasses import dataclass
from typing import Any, Callable, ContextManager, Iterator

Capture = Callable[[object], Any]
Bind = Callable[[Any], ContextManager[Any]]


@dataclass(frozen=True)
class TurnScopedCarrier:
    """How one turn-scoped value crosses a task boundary.

    ``variables`` are the ``ContextVar`` objects this carrier is responsible
    for.  They make the registry enumerable by the thing that matters — a
    ContextVar a turn publishes — so completeness can be checked against a live
    context rather than a hand-maintained list of names.
    """

    name: str
    variables: tuple[contextvars.ContextVar, ...]
    capture: Capture
    bind: Bind


_REGISTRY_LOCK = threading.Lock()
_CARRIERS: dict[str, TurnScopedCarrier] = {}


def turn_scoped(
    name: str,
    *,
    variables: tuple[contextvars.ContextVar, ...],
    capture: Capture,
    bind: Bind,
) -> TurnScopedCarrier:
    """Declare a turn-scoped value at its definition site.

    ``capture(agent)`` runs on the task that owns the turn and returns whatever
    must be re-presented; ``bind(captured)`` is a context manager that
    re-presents it on a foreign task.  ``capture`` receives the agent the work
    belongs to, because some values (the transition-lock reentry token, the
    turn/session binding) are defined relative to one agent.

    Registration is idempotent for an identical declaration (a module reload in
    tests) and refuses anything else: two carriers for one name, or one
    ``ContextVar`` claimed by two carriers, would be two writers for one value.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("turn-scoped carrier name must be a non-empty string")
    if not variables or not all(
        isinstance(var, contextvars.ContextVar) for var in variables
    ):
        raise TypeError("turn-scoped carrier must declare its ContextVars")
    if not callable(capture) or not callable(bind):
        raise TypeError("turn-scoped carrier needs callable capture and bind")
    carrier = TurnScopedCarrier(name, tuple(variables), capture, bind)
    with _REGISTRY_LOCK:
        existing = _CARRIERS.get(name)
        if existing is not None and _same_declaration(existing, carrier):
            _CARRIERS[name] = carrier
            return carrier
        if existing is not None:
            raise ValueError(f"turn-scoped carrier {name!r} is already declared")
        claimed = {
            id(var): other.name
            for other in _CARRIERS.values()
            for var in other.variables
        }
        for var in carrier.variables:
            owner = claimed.get(id(var))
            if owner is not None:
                raise ValueError(
                    f"ContextVar {var.name!r} is already carried by {owner!r}"
                )
        _CARRIERS[name] = carrier
    return carrier


def _same_declaration(left: TurnScopedCarrier, right: TurnScopedCarrier) -> bool:
    """Whether ``right`` re-declares ``left`` (same name and var names)."""
    return left.name == right.name and [
        var.name for var in left.variables
    ] == [var.name for var in right.variables]


def turn_scoped_carriers() -> tuple[TurnScopedCarrier, ...]:
    """Every declared carrier, in declaration order."""
    with _REGISTRY_LOCK:
        return tuple(_CARRIERS.values())


def turn_scoped_variables() -> frozenset[contextvars.ContextVar]:
    """Every ``ContextVar`` some carrier is responsible for."""
    return frozenset(
        var for carrier in turn_scoped_carriers() for var in carrier.variables
    )


@dataclass(frozen=True)
class TurnScopeSnapshot:
    """Captured turn-scoped values, re-presentable on any task."""

    captured: tuple[tuple[TurnScopedCarrier, Any], ...]

    @contextlib.contextmanager
    def bind(self) -> Iterator[None]:
        """Re-present every captured value for the duration of the block.

        Binds in declaration order and unwinds in reverse, so the foreign task
        observes exactly the owning turn's values and its own context is
        restored afterwards.
        """
        with contextlib.ExitStack() as stack:
            for carrier, value in self.captured:
                stack.enter_context(carrier.bind(value))
            yield


def capture_turn_scope(agent: object) -> TurnScopeSnapshot:
    """Capture every declared turn-scoped value on the calling (owning) task.

    Call this where the closure is BUILT — on the task that owns the turn —
    never inside the closure, which may run on a task whose context predates
    the turn.
    """
    return TurnScopeSnapshot(
        tuple(
            (carrier, carrier.capture(agent))
            for carrier in turn_scoped_carriers()
        )
    )
