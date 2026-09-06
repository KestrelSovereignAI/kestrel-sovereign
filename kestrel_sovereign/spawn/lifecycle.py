"""
Spawned Agent Lifecycle Management.

Manages TTL monitoring, auto-termination, result collection, and cleanup
for spawned child agents. Supports ephemeral (auto-cleanup) and persistent
(survives restarts) modes.

Lifecycle:
    1. Parent spawns child via AgentManager.spawn_agent()
    2. SpawnedAgentLifecycle.register() starts TTL monitoring
    3. On task completion or TTL expiry → auto-terminate + collect results
    4. Ephemeral resources (temp dirs, in-memory DBs) are cleaned up
    5. Hook events fire for AGENT_SPAWN and AGENT_TERMINATE
"""

import asyncio
import logging
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, Optional

from kestrel_sdk.hooks.base import HookEvent, HookInput

from kestrel_sovereign.hooks.manager import HooksManager
from kestrel_sovereign.spawn.mandate import (
    PersistedSpawnMandateExpiredError,
    SpawnMandate,
    remaining_spawn_ttl_seconds,
)

logger = logging.getLogger(__name__)

_MAX_AUTOMATIC_TERMINATION_ATTEMPTS = 3
_COLD_TTL_RETIREMENT_RETRY_SECONDS = 30.0


def _is_expected_termination_outcome(error: BaseException) -> bool:
    """Accept lifecycle failures/cancellation, never process-control signals."""

    if isinstance(error, (Exception, asyncio.CancelledError)):
        return True
    if isinstance(error, BaseExceptionGroup):
        return all(_is_expected_termination_outcome(item) for item in error.exceptions)
    return False


def _typed_termination_proves_removal(
    manager: object,
    child_name: str,
    error: BaseException,
) -> bool:
    """Require Core typing plus authoritative routing absence before finalize."""

    from kestrel_sovereign.multi_agent.agent_manager import (
        ChildTerminationNotPerformedError,
        ChildTerminationReconciliationError,
        RuntimeOffboardingNotPerformedError,
        RuntimeOffboardingRetainedError,
    )

    leaves: list[BaseException] = []

    def collect(candidate: BaseException) -> None:
        if isinstance(candidate, BaseExceptionGroup):
            for nested in candidate.exceptions:
                collect(nested)
            return
        leaves.append(candidate)

    collect(error)
    supported = (
        RuntimeOffboardingRetainedError,
        RuntimeOffboardingNotPerformedError,
        ChildTerminationReconciliationError,
        ChildTerminationNotPerformedError,
        asyncio.CancelledError,
    )
    if not leaves or any(not isinstance(item, supported) for item in leaves):
        return False
    if any(
        isinstance(item, ChildTerminationNotPerformedError)
        and item.child_name.casefold() == child_name.casefold()
        for item in leaves
    ):
        return False
    private_get_agent = getattr(
        type(manager), "_get_agent_for_lifecycle", None
    )
    get_agent = (
        (lambda name: private_get_agent(manager, name))
        if callable(private_get_agent)
        else None
    )
    if get_agent is None:
        get_agent = getattr(manager, "get_agent", None)
    if not callable(get_agent):
        return False
    try:
        return get_agent(child_name) is None
    except Exception:
        return False


def _manager_proves_child_absent(
    manager: object,
    *,
    parent_did: str,
    child_name: str,
) -> bool:
    """Require both routing and parent-edge absence before local finalization."""

    private_get_agent = getattr(
        type(manager), "_get_agent_for_lifecycle", None
    )
    get_agent = (
        (lambda name: private_get_agent(manager, name))
        if callable(private_get_agent)
        else None
    )
    if get_agent is None:
        get_agent = getattr(manager, "get_agent", None)
    get_children = getattr(manager, "get_children", None)
    if not callable(get_agent) or not callable(get_children):
        return False
    try:
        if get_agent(child_name) is not None:
            return False
        children = get_children(parent_did)
    except Exception:
        return False
    if not isinstance(children, (list, tuple, set, frozenset)):
        return False
    return not any(
        isinstance(candidate, str)
        and candidate.casefold() == child_name.casefold()
        for candidate in children
    )


def _manager_proves_child_live(
    manager: object,
    *,
    parent_did: str,
    child_name: str,
) -> bool:
    """Require both live routing and the parent edge before reopening restart."""

    private_get_agent = getattr(type(manager), "_get_agent_for_lifecycle", None)
    get_agent = (
        (lambda name: private_get_agent(manager, name))
        if callable(private_get_agent)
        else None
    )
    if get_agent is None:
        get_agent = getattr(manager, "get_agent", None)
    get_children = getattr(manager, "get_children", None)
    if not callable(get_agent) or not callable(get_children):
        return False
    try:
        if get_agent(child_name) is None:
            return False
        children = get_children(parent_did)
    except Exception:
        return False
    if not isinstance(children, (list, tuple, set, frozenset)):
        return False
    return any(
        isinstance(candidate, str)
        and candidate.casefold() == child_name.casefold()
        for candidate in children
    )


@dataclass(frozen=True)
class _TerminalRetirementTarget:
    """One exact edge whose restart rail shares a terminal cascade."""

    child_name: str
    child_did: str
    parent_did: str


def _terminal_retirement_tree(
    manager: object,
    *,
    child_name: str,
    child_did: str,
    parent_did: str,
) -> tuple[_TerminalRetirementTarget, ...]:
    """Snapshot an exact descendant tree before cascade removal mutates it."""

    targets = [
        _TerminalRetirementTarget(
            child_name=child_name,
            child_did=child_did,
            parent_did=parent_did,
        )
    ]
    get_children = getattr(manager, "get_children", None)
    class_get_mandate = getattr(type(manager), "get_mandate", None)
    get_mandate = (
        (lambda name: class_get_mandate(manager, name))
        if callable(class_get_mandate)
        else None
    )
    class_unsettled_witnesses = getattr(
        type(manager), "unsettled_spawn_authority_witnesses", None
    )
    durable_witnesses = (
        class_unsettled_witnesses(manager)
        if callable(class_unsettled_witnesses)
        else (
            getattr(type(manager), "active_spawn_authority_witnesses")(manager)
            if callable(
                getattr(type(manager), "active_spawn_authority_witnesses", None)
            )
            else ()
        )
    )
    if not isinstance(durable_witnesses, (list, tuple, set, frozenset)):
        raise TypeError("manager authority projection must be a concrete collection")
    durable_by_parent: dict[str, list[_TerminalRetirementTarget]] = {}
    for witness in durable_witnesses:
        descendant_name = getattr(witness, "child_name", None)
        descendant_did = getattr(witness, "child_did", None)
        descendant_parent_did = getattr(witness, "parent_did", None)
        mandate = getattr(witness, "mandate", None)
        if (
            not isinstance(descendant_name, str)
            or not descendant_name
            or not isinstance(descendant_did, str)
            or not descendant_did
            or not isinstance(descendant_parent_did, str)
            or not descendant_parent_did
            or not isinstance(mandate, SpawnMandate)
            or mandate.child_did != descendant_did
            or mandate.parent_did != descendant_parent_did
        ):
            raise RuntimeError(
                "Refusing terminal cascade with invalid durable authority"
            )
        durable_by_parent.setdefault(descendant_parent_did, []).append(
            _TerminalRetirementTarget(
                child_name=descendant_name,
                child_did=descendant_did,
                parent_did=descendant_parent_did,
            )
        )

    if (
        not callable(get_children)
        or not callable(get_mandate)
    ) and not durable_by_parent:
        return tuple(targets)

    pending = [child_did]
    visited = {child_did}
    known_names = {child_name.casefold(): child_did}
    while pending:
        descendant_parent_did = pending.pop()
        candidates: dict[str, _TerminalRetirementTarget] = {
            target.child_did: target
            for target in durable_by_parent.get(descendant_parent_did, ())
        }
        if callable(get_children) and callable(get_mandate):
            children = get_children(descendant_parent_did)
            if not isinstance(children, (list, tuple, set, frozenset)):
                raise TypeError(
                    "manager child projection must be a concrete collection"
                )
            for descendant_name in children:
                if not isinstance(descendant_name, str) or not descendant_name:
                    raise RuntimeError(
                        "terminal cascade contains an invalid child name"
                    )
                mandate = get_mandate(descendant_name)
                if (
                    not isinstance(mandate, SpawnMandate)
                    or not isinstance(mandate.child_did, str)
                    or not mandate.child_did
                ):
                    raise RuntimeError(
                        "Refusing terminal cascade without stable descendant authority"
                    )
                runtime_target = _TerminalRetirementTarget(
                    child_name=descendant_name,
                    child_did=mandate.child_did,
                    parent_did=descendant_parent_did,
                )
                durable_target = candidates.get(mandate.child_did)
                if (
                    durable_target is not None
                    and durable_target.child_name.casefold()
                    != descendant_name.casefold()
                ):
                    raise RuntimeError(
                        "Refusing terminal cascade with conflicting authority"
                    )
                candidates[mandate.child_did] = runtime_target

        for target in candidates.values():
            known_did = known_names.get(target.child_name.casefold())
            if known_did is not None and known_did != target.child_did:
                raise RuntimeError(
                    "Refusing terminal cascade with conflicting authority"
                )
            if target.child_did in visited:
                raise RuntimeError("Refusing terminal cascade with cyclic authority")
            known_names[target.child_name.casefold()] = target.child_did
            visited.add(target.child_did)
            targets.append(target)
            pending.append(target.child_did)
    return tuple(targets)


async def _settle_terminal_retirement_tree(
    manager: object,
    targets: tuple[_TerminalRetirementTarget, ...],
    owned_intents: dict[tuple[str, str], bool],
    *,
    reopen_live_intents: bool = True,
) -> None:
    """Finalize removed targets and reopen only intents this attempt owns."""

    record_retirement = getattr(manager, "record_expired_spawn_retirement", None)
    cancel_retirement = getattr(
        type(manager),
        "cancel_terminal_spawn_retirement",
        None,
    )
    revoke_scheduler = getattr(
        type(manager),
        "revoke_terminal_spawn_scheduler_authority",
        None,
    )
    # Descendants were removed leaf-first, so settle their witnesses leaf-first
    # as well. A crash anywhere leaves every unfinalized record at ``retiring``.
    for target in reversed(targets):
        if _manager_proves_child_absent(
            manager,
            parent_did=target.parent_did,
            child_name=target.child_name,
        ):
            if callable(revoke_scheduler):
                await revoke_scheduler(
                    manager,
                    target.child_name,
                    target.child_did,
                )
            if callable(record_retirement):
                record_retirement(
                    target.child_name,
                    expected_child_did=target.child_did,
                )
            continue
        if (
            reopen_live_intents
            and owned_intents.get((target.child_name, target.child_did), False)
            and _manager_proves_child_live(
                manager,
                parent_did=target.parent_did,
                child_name=target.child_name,
            )
            and callable(cancel_retirement)
        ):
            cancel_retirement(
                manager,
                target.child_name,
                expected_child_did=target.child_did,
            )


class SpawnStatus(str, Enum):
    """Status of a spawned agent's lifecycle."""

    RUNNING = "running"
    COMPLETED = "completed"
    TERMINATED = "terminated"
    TIMED_OUT = "timed_out"
    FAILED = "failed"


class SpawnMode(str, Enum):
    """Storage mode for spawned agents."""

    EPHEMERAL = "ephemeral"
    PERSISTENT = "persistent"


@dataclass(frozen=True)
class TerminationRefusalState:
    """Operator-visible state for a live child whose TTL removal was refused."""

    automatic_termination_attempts: int
    requested_status: SpawnStatus
    recorded_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def as_metadata(self) -> Dict[str, Any]:
        """Return a detached, stable operator-facing representation."""

        return {
            "termination_not_performed": True,
            "automatic_termination_attempts": self.automatic_termination_attempts,
            "automatic_retries_exhausted": True,
            "operator_action_required": True,
            "retry_termination": True,
            "requested_status": self.requested_status.value,
            "recorded_at": self.recorded_at,
        }


@dataclass
class SpawnResult:
    """Result collected from a spawned child agent after termination.

    Attributes:
        child_name: Name of the child agent.
        child_did: DID of the child agent.
        status: Final lifecycle status.
        output_artifacts: Any data the child produced.
        budget_consumed: Total amount spent from delegated budget.
        started_at: ISO timestamp of when the child was registered.
        ended_at: ISO timestamp of when the child completed/terminated.
        finalized_from_absence: True only when local lifecycle finalization used
            manager routing and parent-edge absence after a termination call
            returned False. This is not evidence about runtime-tree custody.
    """

    child_name: str
    child_did: str
    status: SpawnStatus
    output_artifacts: Dict[str, Any] = field(default_factory=dict)
    budget_consumed: Decimal = Decimal("0")
    started_at: str = ""
    ended_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    # parent_did identifies which parent agent spawned this child.
    # Required for filtering history by parent in multi-agent mode
    # where the lifecycle is shared across all loaded agents
    # (#1149 round 4 — without this the spawn panel could leak
    # other parents' history). Defaulted to "" for back-compat with
    # any existing serialized SpawnResult dataclasses.
    parent_did: str = ""
    finalized_from_absence: bool = False


@dataclass
class _TrackedChild:
    """Internal tracking state for a registered child agent."""

    child_name: str
    child_did: str
    parent_did: str
    mode: SpawnMode
    ttl_seconds: int
    purpose: str
    temp_dir: Optional[str] = None
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    ttl_task: Optional[asyncio.Task] = None
    result: Optional[SpawnResult] = None
    automatic_termination_attempts: int = 0
    termination_refusal: Optional[TerminationRefusalState] = None


class SpawnedAgentLifecycle:
    """Manages lifecycle of spawned child agents.

    Responsibilities:
    - TTL monitoring with background tasks
    - Auto-termination on TTL expiry
    - Result collection from child agents
    - Cleanup of ephemeral resources (temp directories)
    - Firing hook events for spawn/terminate
    """

    def __init__(
        self,
        agent_manager: Any,
        hooks_manager: Optional[HooksManager] = None,
    ):
        """Initialize the lifecycle manager.

        Args:
            agent_manager: The AgentManager that owns the spawned agents.
            hooks_manager: Optional HooksManager for firing spawn/terminate events.
        """
        self._agent_manager = agent_manager
        self._hooks_manager = hooks_manager
        self._tracked: Dict[str, _TrackedChild] = {}
        self._results: Dict[str, SpawnResult] = {}
        # A valid finite host witness can remain intentionally cold
        # (``autostart=false``) or fail before runtime publication.  It still
        # needs a process-owned deadline so its name/cap slot cannot survive
        # the signed lifetime merely because no child runtime was available to
        # own the normal tracked timer.
        self._cold_ttl_tasks: dict[tuple[str, str], asyncio.Task] = {}
        self._lock = asyncio.Lock()
        # The lifecycle lock serializes finalizers, but lock ownership alone
        # does not identify which child the active finalizer owns. Manager-side
        # direct removal of a different child must still retire that exact
        # record rather than mistaking the unrelated lock holder for its owner.
        self._finalization_owner_counts: dict[tuple[str, str], int] = {}
        # The lifecycle object is intentionally created lazily by SpawnFeature,
        # often after AgentManager has cold-loaded every agent.  Adopt the
        # manager's durable mandate projections at construction so public
        # terminate/TTL paths see the same children as the authority maps.
        # Construction may happen while AgentManager is still onboarding a
        # restored child.  Adopt the records now, but let the publication
        # commit arm each TTL explicitly so a reaper cannot race that commit.
        self.restore_from_manager(arm_ttls=False)

    def _claim_finalization(self, child_name: str, child_did: str) -> None:
        key = (child_name, child_did)
        self._finalization_owner_counts[key] = (
            self._finalization_owner_counts.get(key, 0) + 1
        )

    def _release_finalization(self, child_name: str, child_did: str) -> None:
        key = (child_name, child_did)
        remaining = self._finalization_owner_counts.get(key, 0) - 1
        if remaining > 0:
            self._finalization_owner_counts[key] = remaining
        else:
            self._finalization_owner_counts.pop(key, None)
            tracked = self._tracked.get(child_name)
            if (
                tracked is not None
                and tracked.child_did == child_did
                and _manager_proves_child_absent(
                    self._agent_manager,
                    parent_did=tracked.parent_did,
                    child_name=child_name,
                )
                and self._durable_retirement_is_settled(child_name, child_did)
            ):
                # A direct manager prune retains tracking while a finalizer is
                # queued so that owner can reconcile normally. If that owner
                # is cancelled before taking the lifecycle lock, its finally
                # is the last custody boundary and must retire the now-absent
                # child rather than leak it as perpetually running.
                self.retire_persisted_child(
                    child_name,
                    expected_child_did=child_did,
                )

    def _durable_retirement_is_settled(
        self,
        child_name: str,
        child_did: str,
    ) -> bool:
        """Whether lifecycle tracking may release its last terminal owner."""

        from kestrel_sovereign.spawn.authority_registry import (
            SpawnAuthorityRegistry,
        )

        registry = vars(self._agent_manager).get("_spawn_authority_registry")
        if not isinstance(registry, SpawnAuthorityRegistry):
            return True
        try:
            witness = registry.get(child_did)
        except Exception:
            logger.exception(
                "Could not prove durable retirement settled for child %r; "
                "lifecycle custody retained",
                child_name,
            )
            return False
        return witness is None or (
            witness.retired
            and witness.child_name.casefold() == child_name.casefold()
        )

    @staticmethod
    def _remaining_ttl_seconds(created_at: str, ttl_seconds: int) -> float:
        """Return a persisted mandate's remaining lifetime, never a fresh TTL."""

        return remaining_spawn_ttl_seconds(created_at, ttl_seconds)

    def restore_from_manager(self, *, arm_ttls: bool = True) -> None:
        """Adopt every durable child authority already projected by the manager."""

        # Read concrete manager state: permissive proxies (notably test
        # MagicMocks) can fabricate an apparent authority map via getattr.
        mandates = vars(self._agent_manager).get("_child_mandates", {})
        if not isinstance(mandates, dict):
            raise TypeError("agent manager child mandates must be a mapping")
        relationships = vars(self._agent_manager).get("_parent_children", {})
        if not isinstance(relationships, dict):
            raise TypeError("agent manager parent relationships must be a mapping")
        for child_name, mandate in tuple(mandates.items()):
            parent_ids = [
                parent_did
                for parent_did, children in relationships.items()
                if child_name in children
            ]
            if len(parent_ids) > 1:
                raise RuntimeError(
                    f"Child {child_name!r} has multiple lifecycle parents"
                )
            self.restore_persisted_child(
                child_name,
                mandate,
                authority_parent_did=parent_ids[0] if parent_ids else None,
                arm_ttl=arm_ttls,
            )

    def _arm_restored_ttl_if_possible(self, tracked: _TrackedChild) -> None:
        """Arm a restored ephemeral timer once called inside a running loop."""

        if tracked.mode is not SpawnMode.EPHEMERAL or tracked.ttl_task is not None:
            return
        remaining = self._remaining_ttl_seconds(
            tracked.started_at,
            tracked.ttl_seconds,
        )
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning(
                "Restored child %r outside a running event loop; TTL will "
                "be armed when lifecycle state is next adopted",
                tracked.child_name,
            )
            return
        tracked.ttl_task = loop.create_task(
            self._ttl_monitor(tracked.child_name, tracked.child_did, remaining),
            name=f"spawn_ttl:{tracked.child_name}",
        )

    def arm_cold_authority_ttl(
        self,
        child_name: str,
        mandate: Any,
        *,
        authority_parent_did: Optional[str] = None,
    ) -> None:
        """Own a finite witness deadline even while its runtime stays cold."""

        child_did = getattr(mandate, "child_did", None)
        parent_did = authority_parent_did or getattr(mandate, "parent_did", None)
        ttl_seconds = getattr(mandate, "ttl_seconds", None)
        created_at = getattr(mandate, "created_at", None)
        if not isinstance(child_name, str) or not child_name:
            raise TypeError("cold authority name must be a non-empty string")
        if not isinstance(child_did, str) or not child_did:
            raise TypeError("cold authority child DID must be a non-empty string")
        if not isinstance(parent_did, str) or not parent_did:
            raise TypeError("cold authority parent DID must be a non-empty string")
        if not isinstance(ttl_seconds, int) or isinstance(ttl_seconds, bool):
            raise TypeError("cold authority TTL must be an integer")
        if not isinstance(created_at, str):
            raise TypeError("cold authority created_at must be a string")
        if ttl_seconds <= 0:
            return

        tracked = self._tracked.get(child_name)
        if tracked is not None:
            if tracked.child_did != child_did or tracked.parent_did != parent_did:
                raise RuntimeError(
                    f"Conflicting lifecycle authority for child {child_name!r}"
                )
            self._arm_restored_ttl_if_possible(tracked)
            return

        key = (child_name.casefold(), child_did)
        current = self._cold_ttl_tasks.get(key)
        if current is not None and not current.done():
            return
        remaining = self._remaining_ttl_seconds(created_at, ttl_seconds)
        task = asyncio.create_task(
            self._ttl_monitor(child_name, child_did, remaining),
            name=f"cold_spawn_ttl:{child_name}",
        )
        self._retain_cold_ttl_owner(key, task)

    def _retain_cold_ttl_owner(
        self,
        key: tuple[str, str],
        task: asyncio.Task,
    ) -> asyncio.Task:
        """Keep one exact deadline owner while no child runtime is published."""

        current = self._cold_ttl_tasks.get(key)
        if (
            current is not None
            and current is not task
            and current is not asyncio.current_task()
            and not current.done()
        ):
            if task is not asyncio.current_task():
                task.cancel()
            return current
        self._cold_ttl_tasks[key] = task

        def release(completed: asyncio.Task) -> None:
            if not completed.cancelled():
                completed.exception()
            if self._cold_ttl_tasks.get(key) is completed:
                self._cold_ttl_tasks.pop(key, None)

        task.add_done_callback(release)
        return task

    def restore_persisted_child(
        self,
        child_name: str,
        mandate: Any,
        *,
        authority_parent_did: Optional[str] = None,
        arm_ttl: bool = True,
    ) -> None:
        """Rehydrate one cold-loaded child without replaying the spawn hook."""

        child_did = getattr(mandate, "child_did", None)
        parent_did = getattr(mandate, "parent_did", None)
        ttl_seconds = getattr(mandate, "ttl_seconds", None)
        purpose = getattr(mandate, "purpose", "")
        created_at = getattr(mandate, "created_at", "")
        if not isinstance(child_name, str) or not child_name:
            raise TypeError("persisted child name must be a non-empty string")
        if not isinstance(child_did, str) or not child_did:
            raise TypeError("persisted child DID must be a non-empty string")
        if not isinstance(parent_did, str) or not parent_did:
            raise TypeError("persisted parent DID must be a non-empty string")
        if authority_parent_did is not None:
            if not isinstance(authority_parent_did, str) or not authority_parent_did:
                raise TypeError("authority parent DID must be a non-empty string")
            parent_did = authority_parent_did
        if not isinstance(ttl_seconds, int) or isinstance(ttl_seconds, bool):
            raise TypeError("persisted spawn TTL must be an integer")
        if not isinstance(purpose, str) or not isinstance(created_at, str):
            raise TypeError("persisted spawn purpose and created_at must be strings")

        existing = self._tracked.get(child_name)
        if existing is not None:
            if (
                existing.child_did != child_did
                or existing.parent_did != parent_did
                or existing.ttl_seconds != ttl_seconds
            ):
                raise RuntimeError(
                    f"Conflicting lifecycle authority for child {child_name!r}"
                )
            if arm_ttl:
                self._arm_restored_ttl_if_possible(existing)
            return

        mode = SpawnMode.PERSISTENT if ttl_seconds <= 0 else SpawnMode.EPHEMERAL
        tracked = _TrackedChild(
            child_name=child_name,
            child_did=child_did,
            parent_did=parent_did,
            mode=mode,
            ttl_seconds=ttl_seconds,
            purpose=purpose,
            started_at=created_at,
        )
        if mode is SpawnMode.EPHEMERAL and arm_ttl:
            self._arm_restored_ttl_if_possible(tracked)
        self._tracked[child_name] = tracked

    def arm_restored_child_ttl(
        self,
        child_name: str,
        *,
        expected_child_did: Optional[str] = None,
    ) -> None:
        """Commit a restored child's TTL after host onboarding completes.

        Expiry at this boundary is a rejected load, not an immediately queued
        reaper: callers must still be able to withdraw publication and report
        failure instead of returning an already-unroutable agent.
        """

        tracked = self._tracked.get(child_name)
        if tracked is None or (
            expected_child_did is not None
            and tracked.child_did != expected_child_did
        ):
            raise RuntimeError(
                f"Restored lifecycle authority for child {child_name!r} is unavailable"
            )
        if tracked.mode is not SpawnMode.EPHEMERAL or tracked.ttl_task is not None:
            return
        cold_key = (child_name.casefold(), tracked.child_did)
        cold_owner = self._cold_ttl_tasks.pop(cold_key, None)
        if cold_owner is not None and not cold_owner.done():
            # Transfer the original host deadline instead of resetting the
            # signed lifetime after a slow or retried cold boot.
            tracked.ttl_task = cold_owner
            return
        if self._remaining_ttl_seconds(
            tracked.started_at,
            tracked.ttl_seconds,
        ) <= 0:
            raise PersistedSpawnMandateExpiredError(
                "Persisted spawn mandate expired during onboarding"
            )
        self._arm_restored_ttl_if_possible(tracked)

    def withdraw_persisted_child(
        self,
        child_name: str,
        *,
        expected_child_did: Optional[str] = None,
        preserve_ttl_owner: bool = False,
    ) -> bool:
        """Withdraw exact tracking without touching a replacement.

        A failed cold publication may return its already-running deadline to
        host-only custody. Other withdrawals cancel the timer as before.
        """

        tracked = self._tracked.get(child_name)
        if tracked is None or (
            expected_child_did is not None
            and tracked.child_did != expected_child_did
        ):
            return False
        self._tracked.pop(child_name, None)
        if tracked.ttl_task is not None:
            if preserve_ttl_owner and not tracked.ttl_task.done():
                self._retain_cold_ttl_owner(
                    (child_name.casefold(), tracked.child_did),
                    tracked.ttl_task,
                )
            elif tracked.ttl_task is not asyncio.current_task():
                tracked.ttl_task.cancel()
        return True

    def disarm_persisted_child(
        self,
        child_name: str,
        *,
        expected_child_did: Optional[str] = None,
    ) -> bool:
        """Cancel an exact child's timer while retaining terminal reconciliation."""

        tracked = self._tracked.get(child_name)
        if tracked is None or (
            expected_child_did is not None
            and tracked.child_did != expected_child_did
        ):
            return False
        if (
            tracked.ttl_task is not None
            and tracked.ttl_task is not asyncio.current_task()
        ):
            tracked.ttl_task.cancel()
        tracked.ttl_task = None
        return True

    def retire_persisted_child(
        self,
        child_name: str,
        *,
        expected_child_did: Optional[str] = None,
    ) -> bool:
        """Retire direct-removal tracking without stealing lifecycle cleanup.

        An explicit termination, result report, or TTL expiry holds ``_lock``
        while it asks AgentManager to remove the child. That lifecycle owner
        still needs its local record to publish the terminal result and hook,
        so manager pruning only disarms its timer. A direct manager removal has
        no such owner and withdraws the exact record immediately. When that
        removal is an ordinary Stop, however, its still-active finite authority
        keeps the original signed deadline under host-only custody. Terminal or
        destructive retirement has already advanced the witness out of
        ``active`` and therefore cancels the timer. Both paths avoid cancelling
        the TTL task when it is the current task.
        """

        tracked = self._tracked.get(child_name)
        if tracked is None or (
            expected_child_did is not None
            and tracked.child_did != expected_child_did
        ):
            return False
        if self._finalization_owner_counts.get((child_name, tracked.child_did), 0):
            # The exact child's finalizer may have claimed ownership before it
            # reached the global lifecycle lock. Cancelling its timer here
            # strands ``_tracked``: the task's ``finally`` releases only its
            # owner count. Let that queued owner observe manager-side removal
            # and finish local reconciliation; the active finalizer will
            # disarm any sibling timer when it publishes its terminal result.
            return True
        preserve_ttl_owner = False
        if (
            tracked.mode is SpawnMode.EPHEMERAL
            and tracked.ttl_task is not None
            and not tracked.ttl_task.done()
        ):
            from kestrel_sovereign.spawn.authority_registry import (
                SpawnAuthorityRegistry,
            )

            registry = vars(self._agent_manager).get("_spawn_authority_registry")
            if isinstance(registry, SpawnAuthorityRegistry):
                witness = registry.get(tracked.child_did)
                resolve_parent = getattr(
                    type(self._agent_manager),
                    "resolve_spawn_authority_parent_did",
                    None,
                )
                witness_parent_did = (
                    resolve_parent(
                        self._agent_manager,
                        child_name,
                        tracked.child_did,
                        witness.parent_did,
                    )
                    if witness is not None and callable(resolve_parent)
                    else (witness.parent_did if witness is not None else None)
                )
                preserve_ttl_owner = bool(
                    witness is not None
                    and witness.active
                    and witness.child_name.casefold() == child_name.casefold()
                    and witness_parent_did == tracked.parent_did
                    and witness.mandate.ttl_seconds > 0
                )
        return self.withdraw_persisted_child(
            child_name,
            expected_child_did=expected_child_did,
            preserve_ttl_owner=preserve_ttl_owner,
        )

    def create_ephemeral_dir(self) -> str:
        """Create a temporary directory for an ephemeral child agent.

        Returns:
            Path to the created temporary directory.
        """
        return tempfile.mkdtemp(prefix="kestrel_spawn_")

    async def register(
        self,
        child_name: str,
        child_did: str,
        parent_did: str,
        ttl_seconds: int = 3600,
        mode: SpawnMode = SpawnMode.EPHEMERAL,
        purpose: str = "",
        temp_dir: Optional[str] = None,
        started_at: Optional[str] = None,
    ) -> None:
        """Register a spawned child for lifecycle tracking.

        Starts a TTL background task that will auto-terminate the child
        when the TTL expires.

        Args:
            child_name: Name of the child agent.
            child_did: DID of the child agent.
            parent_did: DID of the parent agent.
            ttl_seconds: Time-to-live in seconds before auto-termination.
            mode: EPHEMERAL (auto-cleanup) or PERSISTENT (survives restarts).
            purpose: Purpose description from the spawn mandate.
            temp_dir: Path to temp directory (for ephemeral mode cleanup).
        """
        signed_start = started_at is not None
        preexisting = child_name in self._tracked
        effective_started_at = (
            started_at
            if started_at is not None
            else datetime.now(timezone.utc).isoformat()
        )
        tracked = self._tracked.get(child_name)
        if tracked is not None:
            if (
                tracked.child_did != child_did
                or tracked.parent_did != parent_did
                or tracked.mode is not mode
                or tracked.ttl_seconds != ttl_seconds
                or (started_at is not None and tracked.started_at != started_at)
            ):
                raise RuntimeError(
                    f"Conflicting lifecycle authority for child {child_name!r}"
                )
            # AgentManager may already have armed the signed TTL at governance
            # commit.  The feature-side registration adds hook/history metadata
            # without replacing that exact timer or resetting its lifetime.
            tracked.purpose = purpose
            if temp_dir is not None:
                tracked.temp_dir = temp_dir
        else:
            tracked = _TrackedChild(
                child_name=child_name,
                child_did=child_did,
                parent_did=parent_did,
                mode=mode,
                ttl_seconds=ttl_seconds,
                purpose=purpose,
                temp_dir=temp_dir,
                started_at=effective_started_at,
            )

        # Persistent children deliberately have no automatic expiry.  This is
        # also the shape restored from a durable ``ttl_seconds <= 0`` mandate.
        if mode is SpawnMode.EPHEMERAL and tracked.ttl_task is None:
            remaining = (
                self._remaining_ttl_seconds(
                    tracked.started_at,
                    tracked.ttl_seconds,
                )
                if signed_start or preexisting
                else ttl_seconds
            )
            tracked.ttl_task = asyncio.create_task(
                self._ttl_monitor(child_name, child_did, remaining),
                name=f"spawn_ttl:{child_name}",
            )

        self._tracked[child_name] = tracked

        logger.info(
            "Registered child '%s' (DID: %s) — TTL: %ds, mode: %s",
            child_name, child_did[:30], ttl_seconds, mode.value,
        )

        # Fire AGENT_SPAWN hook
        await self._fire_hook(
            HookEvent.AGENT_SPAWN,
            parent_did=parent_did,
            child_did=child_did,
            child_name=child_name,
            spawn_purpose=purpose,
        )

    async def report_result(
        self,
        child_name: str,
        output_artifacts: Optional[Dict[str, Any]] = None,
        budget_consumed: Decimal = Decimal("0"),
        status: SpawnStatus = SpawnStatus.COMPLETED,
    ) -> Optional[SpawnResult]:
        """Report a result from a child agent and trigger cleanup.

        Called when a child finishes its task. Stores the result, cancels
        the TTL timer, and initiates cleanup.

        Args:
            child_name: Name of the child agent.
            output_artifacts: Data produced by the child.
            budget_consumed: Amount of budget spent.
            status: Final status (defaults to COMPLETED).

        Returns:
            The SpawnResult, or None if the child is not tracked.
        """
        initially_tracked = self._tracked.get(child_name)
        if initially_tracked is None:
            logger.warning("report_result for untracked child '%s'", child_name)
            return None
        owned_child_did = initially_tracked.child_did
        self._claim_finalization(child_name, owned_child_did)
        # Hold the lifecycle lock so report_result and the TTL monitor can't
        # BOTH finalize the same child concurrently (double cleanup / result
        # clobber). The lock was created but never acquired (#1729). Idempotent:
        # if the child is already finalized, return the existing result.
        try:
            async with self._lock:
                tracked = self._tracked.get(child_name)
                if tracked is None or tracked.child_did != owned_child_did:
                    return None
                if tracked.result is not None:
                    return tracked.result  # already finalized (e.g. by TTL)
                result = SpawnResult(
                    child_name=child_name,
                    child_did=tracked.child_did,
                    status=status,
                    output_artifacts=output_artifacts or {},
                    budget_consumed=budget_consumed,
                    started_at=tracked.started_at,
                    parent_did=tracked.parent_did,
                )

                # Terminate and clean up
                terminated = await self._terminate_and_cleanup(
                    child_name,
                    status,
                    result=result,
                )

                return result if terminated else None
        finally:
            self._release_finalization(child_name, owned_child_did)

    def get_result(self, child_name: str) -> Optional[SpawnResult]:
        """Retrieve the result for a terminated child.

        Args:
            child_name: Name of the child agent.

        Returns:
            SpawnResult if available, None otherwise.
        """
        return self._results.get(child_name)

    def pop_result(self, child_name: str) -> Optional[SpawnResult]:
        """Retrieve and remove the result for a terminated child.

        Args:
            child_name: Name of the child agent.

        Returns:
            SpawnResult if available, None otherwise.
        """
        return self._results.pop(child_name, None)

    def is_tracked(self, child_name: str) -> bool:
        """Check if a child is currently being tracked."""
        return child_name in self._tracked

    def get_tracked_children(self) -> list[str]:
        """Return names of all currently tracked children."""
        return list(self._tracked.keys())

    def get_termination_refusal(
        self, child_name: str
    ) -> Optional[Dict[str, Any]]:
        """Return detached operator state without marking the child finalized."""

        tracked = self._tracked.get(child_name)
        if tracked is None or tracked.termination_refusal is None:
            return None
        return tracked.termination_refusal.as_metadata()

    async def terminate(
        self,
        child_name: str,
        reason: str = "explicit termination",
        *,
        offboard_runtime: bool = False,
    ) -> Optional[SpawnResult]:
        """Explicitly terminate a tracked child.

        Args:
            child_name: Name of the child to terminate.
            reason: Human-readable reason for termination.
            offboard_runtime: Explicit destructive runtime-deprovision intent.
                Ordinary TTL, result, and parent-shutdown paths leave this
                false so a restart retains child state.

        Returns:
            SpawnResult if the child was tracked, None otherwise.
        """
        initially_tracked = self._tracked.get(child_name)
        if initially_tracked is None:
            return None
        owned_child_did = initially_tracked.child_did
        self._claim_finalization(child_name, owned_child_did)
        # Serialize explicit termination with result reports and TTL expiry.
        # A refused manager removal must leave the exact child and its timer
        # available to a later retry instead of publishing a false terminal
        # result.
        try:
            async with self._lock:
                tracked = self._tracked.get(child_name)
                if tracked is None or tracked.child_did != owned_child_did:
                    return None

                result = SpawnResult(
                    child_name=child_name,
                    child_did=tracked.child_did,
                    status=SpawnStatus.TERMINATED,
                    started_at=tracked.started_at,
                    parent_did=tracked.parent_did,
                )
                terminated = await self._terminate_and_cleanup(
                    child_name,
                    SpawnStatus.TERMINATED,
                    reason=reason,
                    offboard_runtime=offboard_runtime,
                    result=result,
                )

                return result if terminated else None
        finally:
            self._release_finalization(child_name, owned_child_did)

    async def shutdown(self) -> None:
        """Shut down all tracked children and clean up.

        Called when the parent agent or the entire system shuts down.
        Cascading: terminates all tracked children.
        """
        children = list(self._tracked.keys())
        terminal_outcomes: list[BaseException] = []
        for child_name in children:
            try:
                result = await self.terminate(child_name, reason="parent shutdown")
                if result is None and self.is_tracked(child_name):
                    from kestrel_sovereign.multi_agent.agent_manager import (
                        ChildTerminationNotPerformedError,
                    )

                    terminal_outcomes.append(
                        ChildTerminationNotPerformedError(child_name=child_name)
                    )
            except BaseException as exc:
                if not _is_expected_termination_outcome(exc):
                    raise
                terminal_outcomes.append(exc)
        if len(terminal_outcomes) == 1:
            raise terminal_outcomes[0]
        if terminal_outcomes:
            raise BaseExceptionGroup(
                "One or more spawned children retained terminal cleanup",
                terminal_outcomes,
            )

    async def _ttl_monitor(
        self,
        child_name: str,
        child_did: str,
        ttl_seconds: int,
    ) -> None:
        """Background task that auto-terminates a child when TTL expires."""
        try:
            await asyncio.sleep(ttl_seconds)
        except asyncio.CancelledError:
            return

        exact = self._tracked.get(child_name)
        if exact is None:
            canonical_name = child_name.casefold()
            matches = [
                name
                for name, candidate in self._tracked.items()
                if name.casefold() == canonical_name
                and candidate.child_did == child_did
            ]
            if len(matches) > 1:
                raise RuntimeError(
                    "Case-insensitive lifecycle tracking is ambiguous at TTL expiry"
                )
            if matches:
                # A cold owner starts from the durable registry spelling, while
                # multi_agent.toml may preserve a case-only spelling accepted by
                # roster reconciliation. Finalization must use the live routing
                # key; the immutable child DID prevents a stale timer from
                # selecting a same-name replacement.
                child_name = matches[0]

        self._claim_finalization(child_name, child_did)
        # Same lock as report_result so TTL expiry and a just-in-time result
        # report can't both finalize the child (#1729). Idempotent.
        try:
            async with self._lock:
                tracked = self._tracked.get(child_name)
                if tracked is None:
                    # An ordinary explicit Stop withdraws running work but is
                    # intentionally restartable until its signed TTL.  Its
                    # live timer remains the expiry owner after local tracking
                    # is removed, so the retained host witness cannot become a
                    # permanent name/cap reservation.
                    from kestrel_sovereign.kestrel_agent import (
                        await_lifecycle_task_completion,
                    )

                    try:
                        retirement = asyncio.create_task(
                            self._retire_stopped_authority_at_expiry(
                                child_name,
                                child_did,
                            ),
                            name=f"expired_stopped_spawn_authority:{child_name}",
                        )
                        cancelled, failure = await await_lifecycle_task_completion(
                            retirement
                        )
                        if failure is not None:
                            raise failure
                        if cancelled:
                            raise asyncio.CancelledError()
                        settled = bool(retirement.result())
                    except BaseException as exc:
                        if not _is_expected_termination_outcome(exc):
                            raise
                        logger.error(
                            "Cold authority retirement failed for child '%s'; "
                            "retained owner will retry: %s",
                            child_name,
                            exc,
                        )
                        settled = False
                    if not settled:
                        retry = asyncio.create_task(
                            self._ttl_monitor(
                                child_name,
                                child_did,
                                _COLD_TTL_RETIREMENT_RETRY_SECONDS,
                            ),
                            name=f"retry_cold_spawn_retirement:{child_name}",
                        )
                        self._retain_cold_ttl_owner(
                            (child_name.casefold(), child_did),
                            retry,
                        )
                    return
                if (
                    tracked.child_did != child_did
                    or tracked.result is not None
                ):
                    return  # already finalized by report_result

                logger.info(
                    "TTL expired for child '%s' after %ds",
                    child_name,
                    ttl_seconds,
                )

                result = SpawnResult(
                    child_name=child_name,
                    child_did=tracked.child_did,
                    status=SpawnStatus.TIMED_OUT,
                    started_at=tracked.started_at,
                    parent_did=tracked.parent_did,
                )
                try:
                    terminated = await self._terminate_and_cleanup(
                        child_name,
                        SpawnStatus.TIMED_OUT,
                        reason="TTL expired",
                        result=result,
                    )
                    if not terminated:
                        still_tracked = self._tracked.get(child_name)
                        if (
                            still_tracked is not None
                            and still_tracked.termination_refusal is None
                        ):
                            logger.warning(
                                "TTL termination was refused for child '%s'; "
                                "tracking and periodic retry remain active",
                                child_name,
                            )
                except BaseException as exc:
                    if not _is_expected_termination_outcome(exc):
                        raise
                    logger.error(
                        "TTL termination retained cleanup for child '%s': %s",
                        child_name,
                        exc,
                    )
                    still_tracked = self._tracked.get(child_name)
                    if (
                        still_tracked is not None
                        and still_tracked.child_did == child_did
                        and still_tracked.result is None
                        and still_tracked.ttl_task is asyncio.current_task()
                    ):
                        # Runtime removal can succeed before the durable witness
                        # advances from retiring to retired. Keep both the local
                        # record and one automatic owner until that terminal
                        # write settles; explicit report_result remains a second
                        # safe retry path in the meantime.
                        still_tracked.ttl_task = asyncio.create_task(
                            self._ttl_monitor(
                                child_name,
                                child_did,
                                _COLD_TTL_RETIREMENT_RETRY_SECONDS,
                            ),
                            name=f"retry_live_spawn_retirement:{child_name}",
                        )
        finally:
            self._release_finalization(child_name, child_did)

    async def _retire_stopped_authority_at_expiry(
        self,
        child_name: str,
        child_did: str,
    ) -> bool:
        """Finalize an absent child's witness after stopping its live subtree."""

        from kestrel_sovereign.spawn.authority_registry import (
            SpawnAuthorityRegistry,
        )

        registry = vars(self._agent_manager).get("_spawn_authority_registry")
        if not isinstance(registry, SpawnAuthorityRegistry):
            return False
        witness = registry.get(child_did)
        if (
            witness is None
            or witness.child_name.casefold() != child_name.casefold()
        ):
            return True
        if witness.retired:
            return True
        if not _manager_proves_child_absent(
            self._agent_manager,
            parent_did=witness.parent_did,
            child_name=child_name,
        ):
            return False
        begin_retirements = getattr(
            type(self._agent_manager),
            "begin_terminal_spawn_retirements",
            None,
        )
        if not callable(begin_retirements):
            return False
        begin_fence = getattr(
            type(self._agent_manager),
            "begin_terminal_descendant_spawn_fence",
            None,
        )
        end_fence = getattr(
            type(self._agent_manager),
            "end_terminal_descendant_spawn_fence",
            None,
        )
        if not callable(begin_fence) or not callable(end_fence):
            return False
        fence = await begin_fence(self._agent_manager, witness.child_did)
        try:
            # The fence joins pre-existing descendant spawns and excludes new
            # ones, making this a complete authority snapshot. Persist the
            # whole crash-ordered denial before stopping any live descendant.
            targets = _terminal_retirement_tree(
                self._agent_manager,
                child_name=witness.child_name,
                child_did=witness.child_did,
                parent_did=witness.parent_did,
            )
            owned_intents = begin_retirements(
                self._agent_manager,
                tuple((target.child_name, target.child_did) for target in targets),
            )
            terminate_children = getattr(
                type(self._agent_manager),
                "terminate_children",
                None,
            )
            if not callable(terminate_children):
                return False
            await terminate_children(self._agent_manager, witness.child_did)
            await _settle_terminal_retirement_tree(
                self._agent_manager,
                targets,
                owned_intents,
            )
            settled = registry.get(child_did)
            return settled is None or settled.retired
        finally:
            end_fence(self._agent_manager, fence)

    async def _terminate_and_cleanup(
        self,
        child_name: str,
        status: SpawnStatus,
        reason: str = "",
        *,
        offboard_runtime: bool = False,
        result: Optional[SpawnResult] = None,
        _terminal_descendant_fence_active: bool = False,
    ) -> bool:
        """Terminate the child in AgentManager and clean up ephemeral resources.

        Args:
            child_name: Name of the child to terminate.
            status: The status that caused termination.
            reason: Human-readable reason.
            offboard_runtime: Explicit destructive tenant-runtime intent.
        """
        tracked = self._tracked.get(child_name)
        if tracked is None:
            return False

        terminal_retirement = status in {
            SpawnStatus.COMPLETED,
            SpawnStatus.FAILED,
            SpawnStatus.TIMED_OUT,
        }
        if terminal_retirement and not _terminal_descendant_fence_active:
            begin_fence = vars(self._agent_manager).get(
                "begin_terminal_descendant_spawn_fence"
            )
            class_begin_fence = (
                getattr(
                    type(self._agent_manager),
                    "begin_terminal_descendant_spawn_fence",
                    None,
                )
                if not callable(begin_fence)
                else None
            )
            if callable(begin_fence) or callable(class_begin_fence):
                fence = (
                    await begin_fence(tracked.child_did)
                    if callable(begin_fence)
                    else await class_begin_fence(self._agent_manager, tracked.child_did)
                )
                try:
                    return await self._terminate_and_cleanup(
                        child_name,
                        status,
                        reason,
                        offboard_runtime=offboard_runtime,
                        result=result,
                        _terminal_descendant_fence_active=True,
                    )
                finally:
                    end_fence = vars(self._agent_manager).get(
                        "end_terminal_descendant_spawn_fence"
                    )
                    if callable(end_fence):
                        end_fence(fence)
                    else:
                        class_end_fence = getattr(
                            type(self._agent_manager),
                            "end_terminal_descendant_spawn_fence",
                            None,
                        )
                        if callable(class_end_fence):
                            class_end_fence(self._agent_manager, fence)
        retirement_targets = (
            _terminal_retirement_tree(
                self._agent_manager,
                child_name=child_name,
                child_did=tracked.child_did,
                parent_did=tracked.parent_did,
            )
            if terminal_retirement
            else ()
        )
        retirement_intents: dict[tuple[str, str], bool] = {}
        reopen_live_intents = status is not SpawnStatus.TIMED_OUT
        begin_retirements = getattr(
            type(self._agent_manager),
            "begin_terminal_spawn_retirements",
            None,
        )
        begin_retirement = getattr(
            type(self._agent_manager),
            "begin_terminal_spawn_retirement",
            None,
        )
        if terminal_retirement and callable(begin_retirements):
            try:
                retirement_intents.update(
                    begin_retirements(
                        self._agent_manager,
                        tuple(
                            (target.child_name, target.child_did)
                            for target in retirement_targets
                        ),
                    )
                )
            except BaseException:
                await _settle_terminal_retirement_tree(
                    self._agent_manager,
                    retirement_targets,
                    retirement_intents,
                    reopen_live_intents=reopen_live_intents,
                )
                raise
        elif terminal_retirement and callable(begin_retirement):
            try:
                for target in retirement_targets:
                    target_key = (target.child_name, target.child_did)
                    retirement_intents[target_key] = bool(
                        begin_retirement(
                            self._agent_manager,
                            target.child_name,
                            expected_child_did=target.child_did,
                        )
                    )
            except BaseException:
                await _settle_terminal_retirement_tree(
                    self._agent_manager,
                    retirement_targets,
                    retirement_intents,
                    reopen_live_intents=reopen_live_intents,
                )
                raise

        if status is SpawnStatus.TIMED_OUT:
            fence_expired = getattr(
                type(self._agent_manager),
                "fence_expired_spawn_routes",
                None,
            )
            if callable(fence_expired):
                fence_expired(
                    self._agent_manager,
                    tuple(
                        (target.child_name, target.child_did)
                        for target in retirement_targets
                    ),
                )

        # Terminate via AgentManager (handles cascading grandchildren). The
        # calling entry point already marked the exact child as the owner that
        # still needs this record after manager-side relationship pruning.
        termination_failure: BaseException | None = None
        reconciliation_cancelled = False
        terminated = False
        finalized_from_absence = False
        try:
            if offboard_runtime:
                terminated = await self._agent_manager.terminate_child(
                    tracked.parent_did,
                    child_name,
                    offboard_runtime=True,
                )
            else:
                terminated = await self._agent_manager.terminate_child(
                    tracked.parent_did,
                    child_name,
                )
        except BaseException as exc:
            if not _is_expected_termination_outcome(exc):
                raise
            if not _typed_termination_proves_removal(
                self._agent_manager,
                child_name,
                exc,
            ):
                await _settle_terminal_retirement_tree(
                    self._agent_manager,
                    retirement_targets,
                    retirement_intents,
                    reopen_live_intents=reopen_live_intents,
                )
                raise
            termination_failure = exc
            logger.error(
                "Failed to terminate child '%s' via AgentManager: %s",
                child_name,
                exc,
            )

        if status is SpawnStatus.TIMED_OUT and (
            terminated or termination_failure is not None
        ):
            await_reconciliation = vars(self._agent_manager).get(
                "await_child_termination_reconciliation",
            )
            if callable(await_reconciliation):
                reconciliation_result = await await_reconciliation(
                    tracked.parent_did, child_name
                )
            else:
                class_reconciliation = getattr(
                    type(self._agent_manager),
                    "await_child_termination_reconciliation",
                    None,
                )
                reconciliation_result = (
                    await class_reconciliation(
                        self._agent_manager,
                        tracked.parent_did,
                        child_name,
                    )
                    if callable(class_reconciliation)
                    else None
                )
            if reconciliation_result is not None:
                terminated, reconciliation_cancelled = reconciliation_result
                if not terminated and termination_failure is not None:
                    raise termination_failure

        if not terminated and termination_failure is None:
            if _manager_proves_child_absent(
                self._agent_manager,
                parent_did=tracked.parent_did,
                child_name=child_name,
            ):
                # A concurrent removal may already have shut down and
                # unpublished the child and pruned its parent edge. Finalize
                # local lifecycle custody exactly once. Routing absence alone
                # does not prove that destructive runtime offboarding ran.
                terminated = True
                finalized_from_absence = True
                logger.warning(
                    "Finalizing child %r from routing absence; "
                    "offboard_runtime=%s and runtime custody is unknown",
                    child_name,
                    offboard_runtime,
                )
                if offboard_runtime:
                    from kestrel_sovereign.multi_agent.agent_manager import (
                        RuntimeOffboardingNotPerformedError,
                    )

                    termination_failure = RuntimeOffboardingNotPerformedError(
                        agent_name=child_name,
                        agent_id=tracked.child_did,
                        cleanup_state="custody_unknown",
                    )
            else:
                await _settle_terminal_retirement_tree(
                    self._agent_manager,
                    retirement_targets,
                    retirement_intents,
                    reopen_live_intents=reopen_live_intents,
                )
                # A TTL attempt is itself the timer task and is about to
                # finish. Bound automatic retries so a genuine refusal cannot
                # leak one task and one warning per TTL forever. Explicit
                # retries remain available while the live child stays tracked.
                if tracked.ttl_task is asyncio.current_task():
                    tracked.automatic_termination_attempts += 1
                    attempts = tracked.automatic_termination_attempts
                    if attempts < _MAX_AUTOMATIC_TERMINATION_ATTEMPTS:
                        tracked.ttl_task = asyncio.create_task(
                            self._ttl_monitor(
                                child_name,
                                tracked.child_did,
                                tracked.ttl_seconds,
                            )
                        )
                    else:
                        tracked.termination_refusal = TerminationRefusalState(
                            automatic_termination_attempts=attempts,
                            requested_status=status,
                        )
                        logger.error(
                            "Automatic termination was refused %d times for "
                            "child %r; automatic retries stopped and explicit "
                            "operator termination remains available",
                            attempts,
                            child_name,
                        )
                return False

        if terminal_retirement:
            await _settle_terminal_retirement_tree(
                self._agent_manager,
                retirement_targets,
                retirement_intents,
                reopen_live_intents=reopen_live_intents,
            )

        # Only an authoritative manager success (or its typed terminal
        # exception path, which is surfaced below) may disarm TTL and publish
        # lifecycle completion. A bare False retains every retry handle.
        if result is None:
            result = SpawnResult(
                child_name=child_name,
                child_did=tracked.child_did,
                status=status,
                started_at=tracked.started_at,
                parent_did=tracked.parent_did,
            )
        result.finalized_from_absence = finalized_from_absence
        tracked.termination_refusal = None
        tracked.result = result
        self._results[child_name] = result
        if (
            (status is not SpawnStatus.TERMINATED or offboard_runtime)
            and tracked.ttl_task
            and tracked.ttl_task is not asyncio.current_task()
            and not tracked.ttl_task.done()
        ):
            tracked.ttl_task.cancel()

        # Clean up ephemeral temp directory
        if tracked.mode == SpawnMode.EPHEMERAL and tracked.temp_dir:
            try:
                shutil.rmtree(tracked.temp_dir, ignore_errors=True)
                logger.info(
                    "Cleaned up ephemeral dir for '%s': %s",
                    child_name,
                    tracked.temp_dir,
                )
            except Exception as e:
                logger.error(
                    "Failed to clean up temp dir for '%s': %s",
                    child_name,
                    e,
                )

        # Fire AGENT_TERMINATE hook
        termination_reason = reason or status.value
        await self._fire_hook(
            HookEvent.AGENT_TERMINATE,
            parent_did=tracked.parent_did,
            child_did=tracked.child_did,
            child_name=child_name,
            termination_reason=termination_reason,
        )

        # Remove from tracking
        self._tracked.pop(child_name, None)

        logger.info(
            "Child '%s' lifecycle ended: status=%s, reason=%s",
            child_name,
            status.value,
            termination_reason,
        )
        terminal_outcomes: list[BaseException] = []
        if reconciliation_cancelled:
            terminal_outcomes.append(asyncio.CancelledError())
        if termination_failure is not None:
            terminal_outcomes.append(termination_failure)
        if len(terminal_outcomes) == 1:
            raise terminal_outcomes[0]
        if terminal_outcomes:
            raise BaseExceptionGroup(
                "Child termination finalized with terminal outcomes",
                terminal_outcomes,
            )
        return terminated

    async def _fire_hook(
        self,
        event: HookEvent,
        parent_did: str = "",
        child_did: str = "",
        child_name: str = "",
        spawn_purpose: str = "",
        termination_reason: str = "",
    ) -> None:
        """Fire a hook event if a HooksManager is configured."""
        if self._hooks_manager is None:
            return

        hook_input = HookInput(
            session_id=f"spawn:{child_name}",
            hook_event_name=event.value,
            parent_did=parent_did,
            child_did=child_did,
            child_name=child_name,
            spawn_purpose=spawn_purpose,
            termination_reason=termination_reason,
        )

        try:
            await self._hooks_manager.execute_hooks(event, hook_input)
        except Exception as e:
            logger.error("Hook execution failed for %s: %s", event.value, e)
