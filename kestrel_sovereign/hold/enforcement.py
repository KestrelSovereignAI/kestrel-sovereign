"""Universal turn-start enforcement for durable Hold state.

The storage layer owns the latch and its transaction boundary.  This module
owns the one semantic decision made from that snapshot: a turn may begin only
when neither the host nor the addressed agent latch is active.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from .state import EffectiveHoldState, HoldStateError


class HoldEnforcementUnavailableError(HoldStateError):
    """A production runtime could not bind its load-bearing Hold store."""


class HeldWorkDisposition(str, Enum):
    """The exhaustive treatment of work observed while Hold is active."""

    SKIPPED = "skipped"
    DEFERRED = "deferred"
    REFUSED = "refused"


class HoldTurnRefusal(RuntimeError):
    """Typed refusal raised before a held agent begins a turn.

    ``effective_state`` is the exact immutable host/agent snapshot observed at
    the store's read transaction.  Downstream transports can therefore choose
    their own disposition without parsing prose or re-reading a potentially
    newer latch.
    """

    code = "agent_held"

    def __init__(
        self,
        *,
        agent_id: str,
        effective_state: EffectiveHoldState,
    ) -> None:
        if not effective_state.held:
            raise ValueError("a Hold refusal requires an active latch")
        self.agent_id = agent_id
        self.effective_state = effective_state
        self.host_hold = effective_state.host
        self.agent_hold = effective_state.agent
        self.metadata = {
            "code": self.code,
            "disposition": HeldWorkDisposition.REFUSED.value,
            "agent_id": agent_id,
            "host_hold": effective_state.host,
            "agent_hold": effective_state.agent,
        }
        super().__init__(f"Agent {agent_id!r} is held and cannot begin a turn")


def require_context_hold_store(context: Any) -> Any:
    """Return a host context's Hold store or fail before agents can start."""

    store = getattr(context, "hold_store", None)
    if store is not None:
        return store
    reason = getattr(context, "backend_error", "") or "Hold store is unavailable"
    raise HoldEnforcementUnavailableError(
        f"Host cannot admit agent turns without durable Hold state: {reason}"
    )


async def build_bound_host_context(agent: Any, *, config: Any = None) -> Any:
    """Open the standalone host context and bind its Hold store to ``agent``."""

    from kestrel_sovereign.host_features.context import build_host_context

    context = await build_host_context(config=config)
    agent._hold_store = require_context_hold_store(context)
    return context


async def close_bound_host_context(context: Any) -> None:
    """Close a standalone context after its agent has stopped."""

    if context is None:
        return
    factory = getattr(context, "session_factory", None)
    try:
        if factory is not None:
            await factory.close()
    finally:
        db = getattr(context, "db", None)
        if db is not None and hasattr(db, "close"):
            await db.close()


async def get_effective_hold_state(agent: Any) -> EffectiveHoldState | None:
    """Read the effective Hold snapshot for a bound runtime object.

    Unbound library/test objects retain their pre-Hold construction contract.
    Production factories bind the load-bearing store before initialization.
    """

    # Mock/proxy objects may synthesize arbitrary attributes from ``getattr``.
    # A Hold store is load-bearing only after the runtime explicitly binds the
    # concrete instance attribute, so inspect the instance namespace directly.
    namespace = getattr(agent, "__dict__", None)
    store = namespace.get("_hold_store") if isinstance(namespace, dict) else None
    if store is None:
        return None
    agent_id = getattr(agent, "did", None) or getattr(agent, "agent_id", None)
    if not isinstance(agent_id, str) or not agent_id:
        raise HoldEnforcementUnavailableError(
            "Cannot enforce Hold without a concrete agent DID"
        )
    return await store.get_effective(agent_id)


async def require_turn_start_allowed(agent: Any) -> EffectiveHoldState | None:
    """Linearize one turn admission against host and agent Hold latches.

    The ``HoldStore.get_effective`` transaction is the linearization point.
    A Hold committed before that snapshot refuses the turn; a Hold committed
    after it applies to the next turn.  A release committed after a refusal
    does not rewrite the already-observed evidence carried by the exception.

    Unbound objects retain the library/test construction contract that predates
    Hold.  Production factories are responsible for binding the store before
    initialization, and fail closed through :func:`require_context_hold_store`.
    """

    agent_id = getattr(agent, "did", None) or getattr(agent, "agent_id", None)
    effective = await get_effective_hold_state(agent)
    if effective is None:
        return None
    if effective.held:
        from .metrics import record_held_work_disposition

        record_held_work_disposition(
            disposition=HeldWorkDisposition.REFUSED.value,
            source="turn",
        )
        raise HoldTurnRefusal(agent_id=agent_id, effective_state=effective)
    return effective


__all__ = [
    "HoldEnforcementUnavailableError",
    "HoldTurnRefusal",
    "HeldWorkDisposition",
    "build_bound_host_context",
    "close_bound_host_context",
    "get_effective_hold_state",
    "require_context_hold_store",
    "require_turn_start_allowed",
]
