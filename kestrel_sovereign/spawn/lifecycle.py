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

logger = logging.getLogger(__name__)


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
        tracked = _TrackedChild(
            child_name=child_name,
            child_did=child_did,
            parent_did=parent_did,
            mode=mode,
            ttl_seconds=ttl_seconds,
            purpose=purpose,
            temp_dir=temp_dir,
        )

        # Start TTL monitoring
        tracked.ttl_task = asyncio.create_task(
            self._ttl_monitor(child_name, ttl_seconds)
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
        # Hold the lifecycle lock so report_result and the TTL monitor can't
        # BOTH finalize the same child concurrently (double cleanup / result
        # clobber). The lock was created but never acquired (#1729). Idempotent:
        # if the child is already finalized, return the existing result.
        async with self._lock:
            tracked = self._tracked.get(child_name)
            if tracked is None:
                logger.warning("report_result for untracked child '%s'", child_name)
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

            tracked.result = result
            self._results[child_name] = result

            # Cancel TTL timer since the child is done
            if tracked.ttl_task and not tracked.ttl_task.done():
                tracked.ttl_task.cancel()

            # Terminate and clean up
            await self._terminate_and_cleanup(child_name, status)

            return result

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

    async def terminate(
        self,
        child_name: str,
        reason: str = "explicit termination",
    ) -> Optional[SpawnResult]:
        """Explicitly terminate a tracked child.

        Args:
            child_name: Name of the child to terminate.
            reason: Human-readable reason for termination.

        Returns:
            SpawnResult if the child was tracked, None otherwise.
        """
        tracked = self._tracked.get(child_name)
        if tracked is None:
            return None

        # Cancel TTL timer
        if tracked.ttl_task and not tracked.ttl_task.done():
            tracked.ttl_task.cancel()

        result = SpawnResult(
            child_name=child_name,
            child_did=tracked.child_did,
            status=SpawnStatus.TERMINATED,
            started_at=tracked.started_at,
            parent_did=tracked.parent_did,
        )
        tracked.result = result
        self._results[child_name] = result

        await self._terminate_and_cleanup(
            child_name, SpawnStatus.TERMINATED, reason=reason
        )

        return result

    async def shutdown(self) -> None:
        """Shut down all tracked children and clean up.

        Called when the parent agent or the entire system shuts down.
        Cascading: terminates all tracked children.
        """
        children = list(self._tracked.keys())
        for child_name in children:
            await self.terminate(child_name, reason="parent shutdown")
        self._tracked.clear()

    async def _ttl_monitor(self, child_name: str, ttl_seconds: int) -> None:
        """Background task that auto-terminates a child when TTL expires."""
        try:
            await asyncio.sleep(ttl_seconds)
        except asyncio.CancelledError:
            return

        # Same lock as report_result so TTL expiry and a just-in-time result
        # report can't both finalize the child (#1729). Idempotent.
        async with self._lock:
            tracked = self._tracked.get(child_name)
            if tracked is None or tracked.result is not None:
                return  # already finalized by report_result

            logger.info("TTL expired for child '%s' after %ds", child_name, ttl_seconds)

            result = SpawnResult(
                child_name=child_name,
                child_did=tracked.child_did,
                status=SpawnStatus.TIMED_OUT,
                started_at=tracked.started_at,
                parent_did=tracked.parent_did,
            )
            tracked.result = result
            self._results[child_name] = result

            await self._terminate_and_cleanup(
                child_name, SpawnStatus.TIMED_OUT, reason="TTL expired"
            )

    async def _terminate_and_cleanup(
        self,
        child_name: str,
        status: SpawnStatus,
        reason: str = "",
    ) -> None:
        """Terminate the child in AgentManager and clean up ephemeral resources.

        Args:
            child_name: Name of the child to terminate.
            status: The status that caused termination.
            reason: Human-readable reason.
        """
        tracked = self._tracked.get(child_name)
        if tracked is None:
            return

        # Terminate via AgentManager (handles cascading grandchildren)
        try:
            await self._agent_manager.terminate_child(
                tracked.parent_did, child_name
            )
        except Exception as e:
            logger.error(
                "Failed to terminate child '%s' via AgentManager: %s",
                child_name, e,
            )

        # Clean up ephemeral temp directory
        if tracked.mode == SpawnMode.EPHEMERAL and tracked.temp_dir:
            try:
                shutil.rmtree(tracked.temp_dir, ignore_errors=True)
                logger.info(
                    "Cleaned up ephemeral dir for '%s': %s",
                    child_name, tracked.temp_dir,
                )
            except Exception as e:
                logger.error(
                    "Failed to clean up temp dir for '%s': %s",
                    child_name, e,
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
            child_name, status.value, termination_reason,
        )

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
