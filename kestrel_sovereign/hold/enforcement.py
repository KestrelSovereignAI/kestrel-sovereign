"""Universal turn-start enforcement for durable Hold state.

The storage layer owns the latch and its transaction boundary. This module
owns the one semantic decision made from that snapshot: a turn may begin only
when neither the host nor the addressed agent latch is active.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import contextmanager
from contextvars import ContextVar
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kestrel_sovereign.host_features.storage import HostDatabaseLaunchContext

from .state import EffectiveHoldState, HoldState, HoldStateError

logger = logging.getLogger(__name__)


# A streamed ``!command`` enters through both public turn seams: streaming
# first, then the non-streaming command implementation. Both seams remain
# independently load-bearing, but the same turn must linearize at one latch
# snapshot. A copied ContextVar is not sufficient authority: only the exact
# task that captured the snapshot, or the explicitly adopted decorated
# execution task, may reuse it.
_turn_admission_snapshot: ContextVar[
    tuple[Any, asyncio.Task[Any], EffectiveHoldState | None] | None
] = ContextVar("kestrel_hold_turn_admission_snapshot", default=None)

# Signal dispatch owns a source-specific disposition (skip or defer), while a
# direct caller owns the interactive refusal.  Context is copied into child
# tasks, so bind the waiver to the exact task executing ``process_input``;
# otherwise an unrelated child turn could inherit permission to suppress its
# own refusal metric.
_source_disposition_owner: ContextVar[
    tuple[Any, asyncio.Task[Any]] | None
] = ContextVar("kestrel_hold_source_disposition_owner", default=None)


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


@contextmanager
def _adopt_turn_admission_snapshot(
    agent: Any,
    *,
    from_task: asyncio.Task[Any] | None,
):
    """Transfer a reused admission only to the decorated execution owner."""

    current_task = asyncio.current_task()
    inherited = _turn_admission_snapshot.get()
    token = None
    if (
        current_task is not None
        and from_task is not None
        and inherited is not None
        and inherited[0] is agent
        and inherited[1] is from_task
    ):
        token = _turn_admission_snapshot.set((agent, current_task, inherited[2]))
    try:
        yield
    finally:
        if token is not None:
            _turn_admission_snapshot.reset(token)


@contextmanager
def _adopt_source_disposition_owner(
    agent: Any,
    *,
    from_task: asyncio.Task[Any] | None,
):
    """Transfer source disposition ownership to the invocation execution task.

    ``bind_async_invocation`` deliberately moves production work into one
    isolated child task.  That child is part of the same ingress attempt, but
    arbitrary descendants are not: require an exact parent-task match before
    rebinding the ownership tuple.
    """

    current_task = asyncio.current_task()
    inherited = _source_disposition_owner.get()
    token = None
    if (
        current_task is not None
        and from_task is not None
        and inherited is not None
        and inherited[0] is agent
        and inherited[1] is from_task
    ):
        token = _source_disposition_owner.set((agent, current_task))
    try:
        yield
    finally:
        if token is not None:
            _source_disposition_owner.reset(token)


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
    the store's read transaction. Transports can therefore choose their own
    disposition without parsing prose or re-reading a potentially newer latch.
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
        """Return the stable refusal envelope for non-Python callers."""

        return {
            "code": self.code,
            "disposition": HeldWorkDisposition.REFUSED.value,
            "message": str(self),
            "agent_id": self.agent_id,
            "host_hold": self._latch_payload(self.host_hold),
            "agent_hold": self._latch_payload(self.agent_hold),
        }

    def wire_json(self) -> str:
        """Serialize the refusal without rendering agent-authored prose."""

        return json.dumps(
            self.wire_payload(),
            sort_keys=True,
            separators=(",", ":"),
        )

    def as_http_exception(self):
        """Map this refusal to the canonical HTTP error envelope."""

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


async def build_bound_host_context(
    agent: Any,
    *,
    config: Any = None,
    agent_data_root: str | Path | None = None,
    host_database_launch_context: HostDatabaseLaunchContext | None = None,
) -> Any:
    """Open a standalone host context and bind its Hold store to ``agent``."""

    from kestrel_sovereign.host_features.context import build_host_context

    if host_database_launch_context is None and agent_data_root is not None:
        from kestrel_sovereign.host_features.storage import (
            resolve_host_database_launch_context,
        )

        # A standalone launcher may select an agent root without exporting it
        # as KESTREL_DB_PATH. Resolve Hold from that exact launch description,
        # while retaining an explicit KESTREL_HOST_DB_PATH as operator authority.
        launch_env = dict(os.environ)
        launch_env["KESTREL_DB_PATH"] = str(agent_data_root)
        from kestrel_sovereign.paths import project_dir

        host_database_launch_context = resolve_host_database_launch_context(
            env=launch_env,
            project_root=project_dir(),
        )
    context_kwargs: dict[str, Any] = {"config": config}
    if host_database_launch_context is not None:
        context_kwargs["host_database_launch_context"] = (
            host_database_launch_context
        )
    context = await build_host_context(**context_kwargs)
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

    from kestrel_sovereign.host_features.context import close_host_context

    await close_host_context(context)


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
    if cleanup_cancelled and not isinstance(startup_failure, asyncio.CancelledError):
        raise asyncio.CancelledError() from startup_failure


async def initialize_with_bound_hold_context(
    agent: Any,
    *,
    config: Any = None,
    agent_data_root: str | Path | None = None,
    host_database_launch_context: HostDatabaseLaunchContext | None = None,
) -> Any:
    """Bind Hold and initialize one standalone agent as one owned lifecycle."""

    binding_kwargs: dict[str, Any] = {"config": config}
    if agent_data_root is not None:
        binding_kwargs["agent_data_root"] = agent_data_root
    if host_database_launch_context is not None:
        binding_kwargs["host_database_launch_context"] = (
            host_database_launch_context
        )
    context = await build_bound_host_context(agent, **binding_kwargs)
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


def _hold_scoped_agent_did(agent: Any) -> str:
    """The DID this agent's Hold latches are scoped to.

    Hold scopes a shared table to the calling agent, so its identity comes
    from the one guard (#3251), never an inline did/agent_id chain. The guard
    raises a sibling of ValueError; converting here keeps this seam's
    documented failure type, which every caller fails closed on.
    """

    from kestrel_sovereign.features.storage_access import (
        AgentIdentityUnavailable,
        resolve_scoped_agent_did,
    )

    try:
        return resolve_scoped_agent_did(agent)
    except AgentIdentityUnavailable as error:
        raise HoldEnforcementUnavailableError(
            "Cannot enforce Hold without a concrete agent DID"
        ) from error


async def get_effective_hold_state(agent: Any) -> EffectiveHoldState | None:
    """Read the effective Hold snapshot for an explicitly bound runtime."""

    # Use the instance namespace so proxy objects cannot fabricate a binding
    # through ``__getattr__`` and accidentally activate this authority seam.
    try:
        store = vars(agent).get("_hold_store")
    except TypeError:
        store = None
    if store is None:
        return None

    agent_id = _hold_scoped_agent_did(agent)
    try:
        return await store.get_effective(agent_id)
    except HoldStateError:
        raise
    except Exception as exc:
        # Backend failures at the willingness boundary are not ordinary turn
        # failures.  Preserve the typed outage so durable ingress can return
        # its exact lease without spending an execution attempt.
        raise HoldEnforcementUnavailableError(
            "Durable Hold state could not be read"
        ) from exc


async def require_turn_start_allowed(agent: Any) -> EffectiveHoldState | None:
    """Linearize one turn admission against host and agent Hold latches.

    ``HoldStore.get_effective`` is the linearization point. A Hold committed
    before that snapshot refuses this turn; a Hold committed after it applies
    to the next turn. A release committed after a refusal cannot rewrite the
    exact evidence already captured by :class:`HoldTurnRefusal`.

    Direct library/test instances retain the pre-Hold unbound construction
    contract. Production factories bind the store before initialization and
    fail closed through :func:`require_context_hold_store`.
    """

    reused = _turn_admission_snapshot.get()
    if (
        reused is not None
        and reused[0] is agent
        and reused[1] is asyncio.current_task()
    ):
        return reused[2]

    effective = await get_effective_hold_state(agent)
    if effective is None:
        return None
    if effective.held:
        owner = _source_disposition_owner.get()
        if not (
            owner is not None
            and owner[0] is agent
            and owner[1] is asyncio.current_task()
        ):
            from .metrics import record_held_work_disposition

            record_held_work_disposition(
                disposition=HeldWorkDisposition.REFUSED.value,
                source="interactive",
            )
        raise HoldTurnRefusal(
            agent_id=_hold_scoped_agent_did(agent), effective_state=effective
        )
    return effective


@contextmanager
def source_owns_hold_disposition(agent: Any):
    """Let one exact signal-execution task own its final Hold disposition."""

    owner_task = asyncio.current_task()
    if owner_task is None:
        raise RuntimeError("Hold source disposition requires an asyncio task")
    token = _source_disposition_owner.set((agent, owner_task))
    try:
        yield
    finally:
        _source_disposition_owner.reset(token)


__all__ = [
    "HeldWorkDisposition",
    "HoldEnforcementUnavailableError",
    "HoldTurnRefusal",
    "build_bound_host_context",
    "close_bound_host_context",
    "get_effective_hold_state",
    "initialize_with_bound_hold_context",
    "require_context_hold_store",
    "require_turn_start_allowed",
    "source_owns_hold_disposition",
]
