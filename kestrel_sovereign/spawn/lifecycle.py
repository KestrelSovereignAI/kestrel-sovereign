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
from kestrel_sovereign.spawn.mandate import remaining_spawn_ttl_seconds

logger = logging.getLogger(__name__)

_MAX_AUTOMATIC_TERMINATION_ATTEMPTS = 3


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

    def cleanup_authority_child_did(
        self,
        *,
        parent_did: str,
        child_name: str,
    ) -> Optional[str]:
        """Return the exact child retained for lifecycle cleanup.

        A signed TTL ceases to authorize new governance at expiry, and its
        parent may already be unloaded. The lifecycle finalizer still needs a
        cleanup-only capability for the child it claimed before calling the
        manager. An exhausted automatic-finalization refusal retains that same
        obligation after the finalizer exits so a later parent cascade can
        retry it. No ordinary inactive tracker record grants this authority.
        """

        tracked = self._tracked.get(child_name)
        if tracked is None or tracked.parent_did != parent_did:
            return None
        if not self._tracked_child_has_cleanup_authority(tracked):
            return None
        return tracked.child_did

    def _tracked_child_has_cleanup_authority(self, tracked: _TrackedChild) -> bool:
        """Whether one exact tracker still owns cleanup, never governance."""

        if self._finalization_owner_counts.get(
            (tracked.child_name, tracked.child_did), 0
        ) > 0:
            return True
        return self._tracked_child_is_cleanup_retained(tracked)

    def _tracked_child_is_cleanup_retained(self, tracked: _TrackedChild) -> bool:
        """Whether cleanup remains pending after automatic finalization exits."""

        return tracked.termination_refusal is not None or (
            tracked.mode is SpawnMode.EPHEMERAL
            and tracked.ttl_seconds > 0
            and self._remaining_ttl_seconds(
                tracked.started_at, tracked.ttl_seconds
            )
            <= 0
        )

    def cleanup_authority_children(
        self,
        *,
        parent_did: str,
    ) -> tuple[tuple[str, str], ...]:
        """Snapshot every child currently retained for cleanup.

        Governance queries intentionally exclude expired or revoked mandates.
        A cascading teardown has a separate obligation: it must still visit a
        descendant whose lifecycle finalizer already claimed cleanup before
        that authority expired, including a refusal retained after bounded
        automatic retries end. Return exact name/DID pairs so manager removal
        can retain its same-name replacement fence.
        """

        return tuple(
            (tracked.child_name, tracked.child_did)
            for tracked in self._tracked.values()
            if tracked.parent_did == parent_did
            and self._tracked_child_has_cleanup_authority(tracked)
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
        if self._remaining_ttl_seconds(
            tracked.started_at,
            tracked.ttl_seconds,
        ) <= 0:
            raise RuntimeError("Persisted spawn mandate expired during onboarding")
        self._arm_restored_ttl_if_possible(tracked)

    def withdraw_persisted_child(
        self,
        child_name: str,
        *,
        expected_child_did: Optional[str] = None,
    ) -> bool:
        """Withdraw an exact child's timer without touching a replacement."""

        tracked = self._tracked.get(child_name)
        if tracked is None or (
            expected_child_did is not None
            and tracked.child_did != expected_child_did
        ):
            return False
        self._tracked.pop(child_name, None)
        if (
            tracked.ttl_task is not None
            and tracked.ttl_task is not asyncio.current_task()
        ):
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
        no such owner and withdraws the exact record immediately. Both paths
        avoid cancelling the TTL task when it is the current task.
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
        return self.withdraw_persisted_child(
            child_name,
            expected_child_did=expected_child_did,
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
        tracked = _TrackedChild(
            child_name=child_name,
            child_did=child_did,
            parent_did=parent_did,
            mode=mode,
            ttl_seconds=ttl_seconds,
            purpose=purpose,
            temp_dir=temp_dir,
            started_at=(
                started_at
                if started_at is not None
                else datetime.now(timezone.utc).isoformat()
            ),
        )

        # Persistent children deliberately have no automatic expiry.  This is
        # also the shape restored from a durable ``ttl_seconds <= 0`` mandate.
        if mode is SpawnMode.EPHEMERAL:
            remaining = (
                self._remaining_ttl_seconds(
                    tracked.started_at,
                    tracked.ttl_seconds,
                )
                if signed_start
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

    def cleanup_retained_child_did(
        self,
        *,
        parent_did: str,
        child_name: str,
    ) -> Optional[str]:
        """Return cleanup-only identity after signed governance expires.

        This does not restore the parent's delegation authority. It only lets
        the lifecycle owner expose and retry termination for an ephemeral
        child whose TTL has elapsed but whose cleanup is still pending.
        """

        tracked = self._tracked.get(child_name)
        if tracked is None or tracked.parent_did != parent_did:
            return None
        cleanup_retained = self._tracked_child_is_cleanup_retained(tracked)
        return tracked.child_did if cleanup_retained else None

    def get_cleanup_retained_children(self, *, parent_did: str) -> list[str]:
        """List one parent's expired children without granting governance."""

        return [
            child_name
            for child_name in self._tracked
            if self.cleanup_retained_child_did(
                parent_did=parent_did,
                child_name=child_name,
            )
            is not None
        ]

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

        self._claim_finalization(child_name, child_did)
        # Same lock as report_result so TTL expiry and a just-in-time result
        # report can't both finalize the child (#1729). Idempotent.
        try:
            async with self._lock:
                tracked = self._tracked.get(child_name)
                if (
                    tracked is None
                    or tracked.child_did != child_did
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
                    # The local lifecycle record and ephemeral resources have
                    # already been reconciled. Keep the background TTL monitor
                    # terminal while preserving the manager outcome in logs.
                    logger.error(
                        "TTL termination retained cleanup for child '%s': %s",
                        child_name,
                        exc,
                    )
        finally:
            self._release_finalization(child_name, child_did)

    async def _terminate_and_cleanup(
        self,
        child_name: str,
        status: SpawnStatus,
        reason: str = "",
        *,
        offboard_runtime: bool = False,
        result: Optional[SpawnResult] = None,
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

        # Terminate via AgentManager (handles cascading grandchildren). The
        # calling entry point already marked the exact child as the owner that
        # still needs this record after manager-side relationship pruning.
        termination_failure: BaseException | None = None
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
                raise
            termination_failure = exc
            logger.error(
                "Failed to terminate child '%s' via AgentManager: %s",
                child_name,
                exc,
            )

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
            tracked.ttl_task
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
        if termination_failure is not None:
            raise termination_failure
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
