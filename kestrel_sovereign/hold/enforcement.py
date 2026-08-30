"""Universal turn-start enforcement for durable Hold state.

The storage layer owns the latch and its transaction boundary.  This module
owns the one semantic decision made from that snapshot: a turn may begin only
when neither the host nor the addressed agent latch is active.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from .state import EffectiveHoldState, HoldState, HoldStateError


logger = logging.getLogger(__name__)


# A streamed ``!command`` enters through both public turn seams: streaming
# first, then the non-streaming command implementation.  Both seams must stay
# independently load-bearing, but they must linearize the *same turn* at one
# latch snapshot.  This task-local binding lets only the streaming delegate
# reuse the exact snapshot it already observed; a different agent, task, or
# later top-level call still performs its own durable read.
_turn_admission_snapshot: ContextVar[
    tuple[Any, asyncio.Task[Any], EffectiveHoldState | None] | None
] = ContextVar("kestrel_hold_turn_admission_snapshot", default=None)


@contextmanager
def _reuse_turn_admission_snapshot(
    agent: Any,
    effective_state: EffectiveHoldState | None,
):
    owner_task = asyncio.current_task()
    if owner_task is None:
        raise RuntimeError("Hold admission reuse requires a running asyncio task")
    token = _turn_admission_snapshot.set((agent, owner_task, effective_state))
    try:
        yield
    finally:
        _turn_admission_snapshot.reset(token)


class HoldEnforcementUnavailableError(HoldStateError):
    """A production runtime could not bind its load-bearing Hold store."""


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
            "agent_id": agent_id,
            "host_hold": effective_state.host,
            "agent_hold": effective_state.agent,
        }
        super().__init__(f"Agent {agent_id!r} is held and cannot begin a turn")

    @staticmethod
    def _latch_payload(latch: HoldState | None) -> dict[str, Any] | None:
        if latch is None:
            return None
        return {
            "scope": latch.scope.value,
            "target_id": latch.target_id,
            "reason": latch.reason,
            "actor_id": latch.actor_id,
            "set_at": latch.set_at,
            "hold_receipt_id": latch.hold_receipt_id,
            "revision": latch.revision,
        }

    def wire_payload(self) -> dict[str, Any]:
        """Return the stable, exact refusal envelope for non-Python callers."""

        return {
            "code": self.code,
            "message": str(self),
            "agent_id": self.agent_id,
            "host_hold": self._latch_payload(self.host_hold),
            "agent_hold": self._latch_payload(self.agent_hold),
        }

    def wire_json(self) -> str:
        """Serialize the refusal without turning it into agent-authored prose."""

        return json.dumps(
            self.wire_payload(),
            sort_keys=True,
            separators=(",", ":"),
        )

    def as_http_exception(self):
        """Map this intentional refusal to the canonical HTTP error envelope."""

        from kestrel_sovereign.api_errors import ApiHTTPException

        return ApiHTTPException(
            status_code=423,
            code=self.code,
            message=str(self),
            details=[self.wire_payload()],
        )


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
    try:
        store = require_context_hold_store(context)
    except BaseException as binding_failure:
        await _close_context_after_startup_failure(
            context,
            binding_failure,
            phase="Hold binding",
        )
        raise
    agent._hold_store = store
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
        hold_database = getattr(context, "hold_database", None)
        try:
            if (
                hold_database is not None
                and hold_database is not db
                and hasattr(hold_database, "close")
            ):
                await hold_database.close()
        finally:
            if db is not None and hasattr(db, "close"):
                await db.close()


async def _close_context_after_startup_failure(
    context: Any,
    startup_failure: BaseException,
    *,
    phase: str,
) -> None:
    """Join context cleanup before propagating a failed startup phase."""

    from kestrel_sovereign.kestrel_agent import await_lifecycle_task_completion

    close_task = asyncio.create_task(
        close_bound_host_context(context),
        name=f"hold_context:{phase}:close",
    )
    cleanup_cancelled, cleanup_failure = await await_lifecycle_task_completion(
        close_task
    )
    if cleanup_failure is not None:
        logger.warning(
            "Hold context cleanup failed after %s: %s",
            phase,
            cleanup_failure,
            exc_info=(
                type(cleanup_failure),
                cleanup_failure,
                cleanup_failure.__traceback__,
            ),
        )
    if cleanup_cancelled and not isinstance(
        startup_failure, asyncio.CancelledError
    ):
        raise asyncio.CancelledError() from startup_failure


async def initialize_with_bound_hold_context(
    agent: Any,
    *,
    config: Any = None,
) -> Any:
    """Bind Hold and initialize one standalone agent as one owned lifecycle."""

    context = await build_bound_host_context(agent, config=config)
    try:
        await agent.initialize()
    except BaseException as startup_failure:
        await _close_context_after_startup_failure(
            context,
            startup_failure,
            phase="agent initialization",
        )
        raise
    return context


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

    reused = _turn_admission_snapshot.get()
    if (
        reused is not None
        and reused[0] is agent
        and reused[1] is asyncio.current_task()
    ):
        return reused[2]

    # Hold enforcement is enabled only by the explicit factory binding above.
    # Dynamic proxy objects (notably MagicMock-backed library consumers) may
    # fabricate any missing attribute on access; treating that fabricated value
    # as a store both breaks the pre-Hold unbound-object contract and attempts to
    # await a non-store. Read the instance namespace so only an actual binding
    # activates this load-bearing seam.
    try:
        store = vars(agent).get("_hold_store")
    except TypeError:
        store = None
    if store is None:
        return None
    agent_id = getattr(agent, "did", None) or getattr(agent, "agent_id", None)
    if not isinstance(agent_id, str) or not agent_id:
        raise HoldEnforcementUnavailableError(
            "Cannot enforce Hold without a concrete agent DID"
        )
    effective = await store.get_effective(agent_id)
    if effective.held:
        raise HoldTurnRefusal(agent_id=agent_id, effective_state=effective)
    return effective


__all__ = [
    "HoldEnforcementUnavailableError",
    "HoldTurnRefusal",
    "build_bound_host_context",
    "close_bound_host_context",
    "initialize_with_bound_hold_context",
    "require_context_hold_store",
    "require_turn_start_allowed",
]
