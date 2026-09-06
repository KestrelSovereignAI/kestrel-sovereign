"""
In-process multi-agent manager for Kestrel.

Manages multiple KestrelAgent instances within a single process.
Used for Cloud Run deployments where running separate processes per agent
is impractical, and for any deployment that wants multi-agent in one server.

Replaces ProcessManager for in-process use; ProcessManager is still
available for local dev (separate OS processes per agent).
"""

import asyncio
import copy
import inspect
import logging
import math
import os
import stat
import sys
import tempfile
import time
from collections import deque
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Awaitable, Callable, List, Mapping, Optional

from kestrel_sovereign._async_rwlock import AsyncReaderWriterLock
from kestrel_sovereign.identity.local_anchor import (
    AgentDIDLookupMode,
    read_anchor_agent_did,
    read_anchor_agent_did_sync,
)
from kestrel_sovereign.identity.runtime_identity import IdentityReadinessError
from kestrel_sovereign.kestrel_agent import (
    KestrelAgent,
    arm_host_authority_deadline,
    await_agent_shutdown_completion,
    await_lifecycle_task_completion,
    disarm_host_authority_deadline,
)
from kestrel_sovereign.kestrel_config.constants import SHUTDOWN_TIMEOUT
from kestrel_sovereign.llm.service import LLMService
from kestrel_sovereign.spawn.authority_registry import (
    PendingSpawnAuthority,
    SpawnAuthorityRegistry,
    SpawnAuthorityWitness,
    _identity_anchor_birth_status,
)
from kestrel_sovereign.spawn.delegated_wallet import (
    _default_currency_for,
    create_delegated_wallet,
    has_durable_delegated_child_wallet_provisioning_contract,
    release_delegated_wallet,
)
from kestrel_sovereign.spawn.mandate import (
    PersistedSpawnMandateExpiredError,
    SpawnMandate,
    remaining_spawn_ttl_seconds,
    sign_mandate,
    validate_spawn_max_child_depth,
)

from .config import (
    RETIRED_SPAWN_MARKER,
    LocalAgentConfig,
    MultiAgentConfig,
    spawn_retirement_denies_startup,
)

logger = logging.getLogger(__name__)

_BATCH_STAGE_CALLBACK: ContextVar[
    Optional[Callable[[str, KestrelAgent], Awaitable[None]]]
] = ContextVar("kestrel_batch_stage_callback", default=None)
_BATCH_ACTIVE_BOOT_SEMAPHORE: ContextVar[Optional[asyncio.Semaphore]] = ContextVar(
    "kestrel_batch_active_boot_semaphore",
    default=None,
)

_QUARANTINED_SHUTDOWN_HISTORY_LIMIT = 128
_UNSAFE_QUARANTINED_FAILURE_LIMIT = 128
_UNSAFE_REMOVAL_BUDGET_RELEASE_FAILURE_LIMIT = 128
_QUARANTINED_METADATA_TEXT_LIMIT = 256
_RUNTIME_OFFBOARD_TIMEOUT_ENV = "KESTREL_RUNTIME_OFFBOARD_TIMEOUT_S"
_DEFAULT_RUNTIME_OFFBOARD_TIMEOUT_S = 30.0


@dataclass(frozen=True, slots=True)
class HostedIsolatedRuntimeLifecyclePolicy:
    """Host-supplied lifecycle policy bound to one loaded agent."""

    idle_timeout_seconds: float | None = None
    idle_timeouts: Mapping[str, float | None] = field(default_factory=dict)
    telemetry_observer: Callable[[object], object] | None = None


def _parse_runtime_offboard_timeout(value: object) -> float:
    """Parse the operator cleanup bound with one actionable error contract."""

    try:
        timeout = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"{_RUNTIME_OFFBOARD_TIMEOUT_ENV} must be finite and positive"
        ) from None
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError(
            f"{_RUNTIME_OFFBOARD_TIMEOUT_ENV} must be finite and positive"
        )
    return timeout


RUNTIME_OFFBOARD_TIMEOUT_S = _parse_runtime_offboard_timeout(
    os.environ.get(
        _RUNTIME_OFFBOARD_TIMEOUT_ENV,
        str(_DEFAULT_RUNTIME_OFFBOARD_TIMEOUT_S),
    )
)


def public_exception_type_name(error: BaseException) -> str:
    """Return the first non-private exception class for API-safe metadata."""

    for candidate in type(error).__mro__:
        if issubclass(candidate, BaseException) and not candidate.__name__.startswith(
            "_"
        ):
            return candidate.__name__
    return "RuntimeError"


class RuntimeOffboardingRetainedError(RuntimeError):
    """A stopped agent was unpublished without observed runtime-tree removal.

    Runtime removal is deliberately not transactional with process shutdown:
    once shutdown succeeds, restoring routing or scheduler authority would
    expose a dead agent. This error gives administrative callers an explicit,
    machine-readable custody outcome without including filesystem error text
    that may contain host details. ``cleanup_pending`` distinguishes a bounded
    wait whose manager-owned worker may still succeed from a completed cleanup
    failure that definitively retained the tree.
    """

    def __init__(
        self,
        *,
        agent_name: str,
        agent_id: str,
        runtime_path: Optional[Path],
        cause: BaseException,
        cleanup_pending: bool = False,
    ) -> None:
        self.agent_name = agent_name
        self.agent_id = agent_id
        self.runtime_path = runtime_path
        self.cause = cause
        self.cleanup_pending = cleanup_pending
        self.metadata = {
            "code": "runtime_offboarding_retained",
            "agent": agent_name,
            "agent_id": agent_id,
            "agent_removed": True,
            "runtime_retained": True,
            "runtime_cleanup_pending": cleanup_pending,
            "runtime_cleanup_state": "pending" if cleanup_pending else "retained",
            "cause_type": public_exception_type_name(cause),
        }
        if cleanup_pending:
            message = (
                f"Agent {agent_name!r} was shut down and unpublished; secure "
                "runtime offboarding is still pending in manager-owned cleanup."
            )
        else:
            message = (
                f"Agent {agent_name!r} was shut down and unpublished, but secure "
                "runtime offboarding failed; the tenant tree was retained."
            )
        super().__init__(message)


class RuntimeOffboardingNotPerformedError(RuntimeError):
    """A destructive request stopped the agent but deleted no runtime tree.

    This is intentionally distinct from a failed deletion: an already-absent
    namespace normally has no tree to retain, while a storage-backed agent has
    no hosted namespace that Core is authorized to delete. A lifecycle race
    may know only that routing is already absent; ``custody_unknown`` keeps that
    narrower evidence from becoming a false deletion claim. Administrative
    callers must never translate these outcomes into ``runtime_offboarded=true``.
    """

    def __init__(
        self,
        *,
        agent_name: str,
        agent_id: str,
        cleanup_state: str,
    ) -> None:
        if cleanup_state not in {
            "already_absent",
            "not_hosted",
            "custody_unknown",
        }:
            raise ValueError("invalid runtime offboarding no-op state")
        self.agent_name = agent_name
        self.agent_id = agent_id
        self.cleanup_state = cleanup_state
        self.custody_unknown = cleanup_state == "custody_unknown"
        self.metadata = {
            "code": "runtime_offboarding_not_performed",
            "agent": agent_name,
            "agent_id": agent_id,
            "agent_removed": True,
            "runtime_offboard_requested": True,
            "runtime_offboarded": False,
            "runtime_cleanup_pending": False,
            "runtime_cleanup_state": cleanup_state,
            "runtime_already_absent": cleanup_state == "already_absent",
        }
        if self.custody_unknown:
            self.metadata.update(
                {
                    "runtime_already_absent": False,
                    "runtime_custody_known": False,
                    "runtime_retention_unknown": True,
                }
            )
            message = (
                f"Agent {agent_name!r} was already shut down and unpublished; "
                "this request performed no secure runtime offboarding, so "
                "runtime custody requires operator reconciliation."
            )
        elif cleanup_state == "already_absent":
            self.metadata.update(
                {
                    "runtime_retained": False,
                    "hosted_runtime_configured": True,
                }
            )
            message = (
                f"Agent {agent_name!r} was shut down and unpublished; its hosted "
                "runtime namespace was already absent, so no tree was deleted."
            )
        else:
            self.metadata.update(
                {
                    "runtime_retained": True,
                    "hosted_runtime_configured": False,
                }
            )
            message = (
                f"Agent {agent_name!r} was shut down and unpublished; it has no "
                "hosted runtime namespace for secure offboarding."
            )
        super().__init__(message)


class ChildTerminationReconciliationError(RuntimeError):
    """A removed child needs operator reconciliation of manager bookkeeping.

    This typed lifecycle outcome lets tool callers distinguish an operational
    post-removal reconciliation failure from an arbitrary exception.  The
    original exception remains available to server-side operators only; its
    text is never part of the public metadata contract.
    """

    def __init__(self, *, child_name: str, cause: Exception) -> None:
        self.child_name = child_name
        self.cause = cause
        self.metadata = {
            "code": "child_termination_reconciliation_failed",
            "child_name": child_name,
            "cause_type": public_exception_type_name(cause),
        }
        super().__init__(
            f"Child {child_name!r} was removed, but lifecycle bookkeeping "
            "requires operator reconciliation."
        )


class ChildTerminationNotPerformedError(RuntimeError):
    """The named child was not removed after descendant teardown was attempted.

    Descendant terminal outcomes still need to reach the tool caller, but none
    of them proves that the requested child was stopped.  This leaf makes that
    negative result explicit so a grouped descendant failure cannot be
    misreported as successful termination of the named child.
    """

    def __init__(self, *, child_name: str) -> None:
        self.child_name = child_name
        self.metadata = {
            "code": "child_termination_not_performed",
            "child_name": child_name,
            "agent_removed": False,
        }
        super().__init__(f"Child {child_name!r} was not removed.")


class PersistedSpawnParentUnavailableError(RuntimeError):
    """A signed child receipt cannot be verified without its live parent."""


def _agent_runtime_path(agent: object) -> Optional[Path]:
    scope = getattr(agent, "isolated_runtime_scope", None)
    path = getattr(scope, "path", None)
    return path if isinstance(path, Path) else None


def _contains_lifecycle_cancellation(error: BaseException | None) -> bool:
    if isinstance(error, asyncio.CancelledError):
        return True
    if isinstance(error, BaseExceptionGroup):
        return any(_contains_lifecycle_cancellation(item) for item in error.exceptions)
    return False


def _uncommitted_spawn_not_hosted_cancellation(
    error: BaseException,
) -> Optional[bool]:
    """Classify the one cleanup no-op an uncommitted spawn may reconcile.

    A storage-backed child has no hosted namespace to offboard. Once its
    process is shut down, rollback can still be complete, but only after the
    caller verifies that routing and the delegated hold are gone. Accept one
    Core-typed ``not_hosted`` outcome, optionally grouped solely with lifecycle
    cancellation. Every other custody state or operational/programmer failure
    remains visible to the spawn owner.
    """

    leaves: list[BaseException] = []

    def collect(candidate: BaseException) -> None:
        if isinstance(candidate, BaseExceptionGroup):
            for nested in candidate.exceptions:
                collect(nested)
            return
        leaves.append(candidate)

    collect(error)
    no_hosted = [
        candidate
        for candidate in leaves
        if isinstance(candidate, RuntimeOffboardingNotPerformedError)
        and candidate.cleanup_state == "not_hosted"
    ]
    if len(no_hosted) != 1 or any(
        not isinstance(
            candidate,
            (RuntimeOffboardingNotPerformedError, asyncio.CancelledError),
        )
        for candidate in leaves
    ):
        return None
    if any(
        isinstance(candidate, RuntimeOffboardingNotPerformedError)
        and candidate.cleanup_state != "not_hosted"
        for candidate in leaves
    ):
        return None
    return any(isinstance(candidate, asyncio.CancelledError) for candidate in leaves)


def _is_lifecycle_terminal_outcome(error: BaseException) -> bool:
    """Whether a lifecycle owner may aggregate this outcome and keep sweeping."""

    if isinstance(error, (Exception, asyncio.CancelledError)):
        return True
    if isinstance(error, BaseExceptionGroup):
        return all(_is_lifecycle_terminal_outcome(item) for item in error.exceptions)
    return False


def _raise_lifecycle_outcomes(message: str, outcomes: list[BaseException]) -> None:
    if not outcomes:
        return
    if len(outcomes) == 1:
        raise outcomes[0]
    raise BaseExceptionGroup(message, outcomes)


@dataclass(frozen=True)
class InflightRuntimeOffboarding:
    """Manager-owned filesystem deletion started after terminal shutdown."""

    agent_name: str
    agent_id: str
    runtime_path: Optional[Path]
    task: "asyncio.Task[object]"


@dataclass
class RuntimeOffboardingAdmission:
    """Caller-visible witness that destructive filesystem work was started.

    Administrative callers may mutate durable desired state before invoking
    :meth:`AgentManager.remove_agent`.  A cold agent has no routing entry from
    which those callers could infer whether runtime cleanup was admitted, so
    the manager marks this witness at the actual task-admission boundary.
    """

    started: bool = False


@dataclass(frozen=True)
class SpawnRetirementAdmission:
    """Exact crash-ordering intent owned by one destructive removal."""

    witness: SpawnAuthorityWitness
    transitioned: bool


def _runtime_offboarding_outcome_error(
    *,
    agent_name: str,
    agent_id: str,
    result: object,
) -> Optional[BaseException]:
    """Translate one typed filesystem custody result without path inference."""

    from kestrel_sovereign.features.isolated_runtime import (
        RuntimeNamespaceCleanupOutcome,
    )

    if result is RuntimeNamespaceCleanupOutcome.REMOVED:
        return None
    if (
        result is RuntimeNamespaceCleanupOutcome.ALREADY_ABSENT
        or result is RuntimeNamespaceCleanupOutcome.NOT_HOSTED
    ):
        return RuntimeOffboardingNotPerformedError(
            agent_name=agent_name,
            agent_id=agent_id,
            cleanup_state=result.value,
        )
    return TypeError("secure runtime offboarding returned an invalid outcome")


def _bounded_shutdown_metadata(value: object) -> str:
    """Keep operator-visible quarantine metadata safe and bounded."""

    text = str(value).replace("\x00", " ").replace("\r", " ").replace("\n", " ")
    if len(text) > _QUARANTINED_METADATA_TEXT_LIMIT:
        return text[: _QUARANTINED_METADATA_TEXT_LIMIT - 1] + "…"
    return text


@dataclass(frozen=True)
class A2AHostedPolicy:
    """Manager-owned immutable inbound policy for one published recipient."""

    generation: int
    recipient: object
    recipient_id: str
    resolver: object
    authorizer: object
    router: object
    requester: object


@dataclass
class QuarantinedShutdownReaper:
    """Observable ownership record for cleanup that outlived agent removal.

    The control plane has already withdrawn this generation from routing and
    fenced any delegated budget.  The reaper retains an exact shutdown or
    refund task, so durable storage and a blocked spend/refund cannot be
    reclaimed or garbage-collected underneath cancellation-resistant work.
    """

    reaper_id: str
    agent_name: str
    canonical_agent_name: str
    agent_id: str
    task: "asyncio.Future[object]"
    started_monotonic: float
    runtime_outcome_required: bool = False
    completed_monotonic: Optional[float] = None
    failure: Optional[str] = None


@dataclass(frozen=True)
class QuarantinedShutdownHistory:
    """Bounded, metadata-only outcome retained after a cleanup task settles."""

    reaper_id: str
    agent_name: str
    canonical_agent_name: str
    agent_id: str
    started_monotonic: float
    completed_monotonic: float
    failure: Optional[str]


@dataclass
class InflightRemovalBudgetRelease:
    """One ordinary delegated-budget release admitted before a terminal drain."""

    release_id: str
    child_name: str
    task: "asyncio.Future[object]"
    started_monotonic: float


@dataclass(frozen=True)
class UnsafeRemovalBudgetReleaseFailure:
    """Bounded, acknowledgeable evidence that an ordinary refund failed."""

    release_id: str
    child_name: str
    started_monotonic: float
    completed_monotonic: float
    failure: str


@dataclass
class AgentOperationAdmission:
    """One named create/load/spawn operation that owns publication or rollback.

    A routing name is a single-writer resource.  The operation that reserves it
    owns every initialized result until it either publishes that exact result or
    shuts it down.  Spawn operations extend the same ownership through their
    budget and mandate commit, which gives a terminal fleet shutdown one
    joinable boundary instead of a best-effort ``_pending_spawns`` count.
    """

    name: str
    canonical_name: str
    kind: str
    registration_epoch: int
    owner_task: "asyncio.Task[object] | None"
    # Identity resolution precedes agent construction. A concurrently loading
    # signed child must be able to discover that an already-admitted operation
    # is resolving its parent even before that parent enters
    # ``_initializing_agents``.
    resolved_agent_id: str | None = None
    agent_id_resolution_event: asyncio.Event | None = None
    candidate_staged_event: asyncio.Event | None = None
    spawn_task: "asyncio.Future[object] | None" = None
    child: Optional[KestrelAgent] = None
    spawn_parent: Optional[KestrelAgent] = None
    # Publication and spawn-governance commits are deliberately separate. A
    # spawned child receives its delegated custody before create/load makes it
    # routable, but its spawning operation still owns it until the parent
    # relationship and cap-slot bookkeeping commit below.
    published: bool = False
    committed: bool = False
    # The final signed mandate is projected into the restrictive runtime maps
    # before a live child becomes routable.  It is still provisional until the
    # later budget + governance commit converts the spawn slot, but every
    # route and ready hook must already observe the child's depth/feature
    # ceiling during that interval.
    provisional_spawn_authority: bool = False
    # True only while this spawn owns one slot in ``_pending_spawns``.  The
    # final governance commit converts that reservation into a child mandate
    # under the same lock, so concurrent restored children cannot overflow the
    # cap and concurrent successful spawns do not double-count one another.
    spawn_slot_active: bool = False
    # A cold restore can make one already-published pending spawn the loser at
    # governance commit. Other pending spawns wait for this exact slot to reach
    # terminal rollback instead of also rejecting against the same transient
    # over-cap snapshot.
    spawn_cap_rejected: bool = False
    spawn_slot_terminal: "asyncio.Future[None] | None" = None
    # A ``remove_agent(False)`` rollback inspection found this operation's
    # child or hold still live.  The public spawn tail must not let a later
    # cancellation overwrite the rollback failure with a bare cancellation.
    rollback_incomplete: bool = False
    # The signed edge is written before the final governance commit. Retain its
    # exact storage witness from before the await so rollback can revoke even an
    # add whose commit succeeded immediately before caller cancellation.
    spawn_receipt_graph: object | None = None
    spawn_receipt_source_id: str | None = None
    spawn_receipt_target_id: str | None = None
    spawn_receipt_unsigned_properties: dict[str, object] | None = None
    # A config-driven host must select this child on its next boot. The exact
    # registration is retained until the spawn commits so rollback can remove
    # only the row this operation wrote, without erasing later operator edits.
    spawn_startup_config: LocalAgentConfig | None = None
    spawn_startup_config_path: Path | None = None
    # The local config is known after inception and before nested load. Keep it
    # on the owning admission so both the host authority registry and restart
    # roster can become durable before the child is published.
    spawn_candidate_config: LocalAgentConfig | None = None
    spawn_authority_witness_did: str | None = None
    spawn_authority_witness_mandate: SpawnMandate | None = None
    # Before inception can create a discoverable database, a durable registry
    # reservation denies restart for this name/data slot. The final signed
    # witness atomically replaces it once the child's DID exists.
    spawn_authority_pending_id: str | None = None
    # A live spawn installs this private initializer handoff before entering
    # create -> load. The load path must await it after initialization and
    # before routing publication, so the final-child-DID signed receipt is
    # durable before any peer can address the child.
    before_publish: Optional[Callable[[KestrelAgent], Awaitable[None]]] = None
    # If publication does not commit after ``before_publish`` starts, this
    # inverse must run while the unpublished child's storage is still live.
    # A failed inverse leaves cleanup with the outer spawn owner; closing the
    # graph first would make a possibly-committed signed receipt irrevocable.
    before_publish_rollback: Optional[
        Callable[[KestrelAgent], Awaitable[None]]
    ] = None
    unpublished_cleanup_deferred_to_spawn: bool = False
    # The signed host-authority watchdog cancelled this exact operation. Batch
    # startup converts that owned cancellation into an ordinary per-agent expiry
    # failure after cleanup, instead of treating it as operator cancellation of
    # the entire fleet load.
    host_authority_expired: bool = False


@dataclass(frozen=True, slots=True)
class TerminalDescendantSpawnFence:
    """One owner's lease excluding new child spawns from a parent DID."""

    parent_did: str
    nonce: int


class _DynamicSchedulerTenantRegistration:
    """Own one runtime tenant's scheduler authority until host onboarding."""

    def __init__(
        self,
        manager: "AgentManager",
        name: str,
        agent_id: str,
        config: LocalAgentConfig,
        lifecycle_lock: AsyncReaderWriterLock,
        rollback_protocol: Optional[Callable[[], Awaitable[None]]],
    ) -> None:
        self._manager = manager
        self._name = name
        self._agent_id = agent_id
        self._config = config
        self._lifecycle_lock = lifecycle_lock
        self._rollback_protocol = rollback_protocol
        self.registration_nonce = getattr(
            rollback_protocol, "scheduler_registration_nonce", None
        )
        self._finished = False

    def commit(self) -> None:
        """Publish the registration by opening scheduler execution scope."""

        if self._finished:
            return
        self._finished = True
        # Preparing the durable registration deliberately does not make the
        # DID executable: post-load and host onboarding can still fail after
        # that point, and a claim would adopt its private registration rows.
        # This synchronous publication precedes releasing the lifecycle writer
        # so a host scheduler can observe the DID only after onboarding has
        # committed and no rollback can still own its durable state.
        self._manager._scheduler_execution_scope.add(self._agent_id)
        self._lifecycle_lock.release()

    async def rollback(self) -> None:
        """Revoke scope first, then remove registration-owned durable state."""

        if self._finished:
            return
        self._finished = True
        manager = self._manager
        manager._scheduler_execution_scope.discard(self._agent_id)
        failure: Optional[BaseException] = None
        try:
            if self._rollback_protocol is not None:
                await self._rollback_protocol()
        except BaseException as error:
            failure = error
        finally:
            entry = manager._scheduler_authority_by_did.get(self._agent_id)
            if entry == (self._name, self._config):
                manager._scheduler_authority_by_did.pop(self._agent_id, None)
                if (
                    manager._scheduler_authority_by_name.get(self._name)
                    == self._agent_id
                ):
                    manager._scheduler_authority_by_name.pop(self._name, None)
            self._lifecycle_lock.release()
        if failure is not None:
            raise failure


def _has_shutdown_completion_contract(agent: object) -> bool:
    """Whether this object exposes Kestrel's joinable shutdown continuation.

    The manager's construction seam is intentionally patchable in unit tests,
    where ``KestrelAgent`` itself may be replaced with a ``MagicMock``.  Check
    the async contract rather than using ``isinstance`` against a patched
    module global.
    """
    return inspect.iscoroutinefunction(
        getattr(agent, "wait_for_shutdown_completion", None)
    )


def _has_shutdown_reaper_handoff_contract(agent: object) -> bool:
    """Whether an agent can safely retain its own timed-out shutdown task.

    Only the concrete agent lifecycle supplies this contract.  Looking on the
    class avoids accidentally treating a ``MagicMock``-fabricated attribute as
    proof that an arbitrary test/legacy object can keep its durable storage
    alive after the manager withdraws it from routing.
    """

    return callable(getattr(type(agent), "handoff_shutdown_to_reaper", None))


def _loaded_agent_did(agent: object) -> Optional[str]:
    """Return a concrete agent DID without trusting a dynamic test proxy."""

    for attribute in ("did", "agent_id"):
        value = getattr(agent, attribute, None)
        if isinstance(value, str) and value:
            return value
    return None


def _loaded_agent_bound_dids(agent: object) -> frozenset[str]:
    """Return the stable DID and verified runtime-identity aliases.

    A rotated agent keeps its legacy DID as the manager routing identity while
    signing new artifacts as its successor DID.  Only expand aliases when the
    loaded identity itself contains the stable DID; an unrelated mutable
    identity object must not manufacture a parent binding.
    """

    stable_did = _loaded_agent_did(agent)
    if stable_did is None:
        return frozenset()
    return _identity_bound_dids(stable_did, vars(agent).get("identity"))


def _identity_bound_dids(stable_did: str, identity: object) -> frozenset[str]:
    """Return verified aliases only when an identity includes its stable anchor."""

    identity_dids = {
        candidate
        for candidate in (
            getattr(identity, "legacy_did", None),
            getattr(identity, "new_did", None),
        )
        if isinstance(candidate, str) and candidate
    }
    if stable_did not in identity_dids:
        return frozenset({stable_did})
    return frozenset({stable_did, *identity_dids})


@dataclass
class _HostContextPublicationState:
    """Shared host registry generation observed by initializing agents."""

    registry: object | None = None
    generation: int = 0


class AgentManager:
    """In-process multi-agent manager.

    Holds multiple KestrelAgent instances keyed by name.
    Each agent gets its own LLMService (mutable model preference state)
    and its own storage (SQLite file or Postgres with agent_id isolation).
    """

    def __init__(
        self,
        base_data_dir: Optional[Path] = None,
        *,
        startup_config_path: Optional[Path] = None,
        startup_roster_enabled: bool = True,
        hosted_telegram_route_attestation_resolver_factory: Optional[
            Callable[[str, str, LocalAgentConfig], object]
        ] = None,
        hosted_isolated_runtime_lifecycle_policy_factory: Optional[
            Callable[
                [str, str, LocalAgentConfig],
                HostedIsolatedRuntimeLifecyclePolicy | None,
            ]
        ] = None,
        shared_postgres_backend: object | None = None,
    ):
        if shared_postgres_backend is not None:
            from kestrel_sovereign.storage.db.postgres import PostgresBackend

            if not isinstance(shared_postgres_backend, PostgresBackend):
                raise TypeError("shared_postgres_backend must be a PostgresBackend")
            if not shared_postgres_backend.is_connected:
                raise ValueError("shared_postgres_backend must already be connected")
        self._agents: dict[str, KestrelAgent] = {}
        self._agent_names: dict[str, str] = {}  # agent_id -> name (reverse lookup)
        self._parent_children: dict[str, list[str]] = {}  # parent_did -> [child_name]
        self._child_mandates: dict[str, SpawnMandate] = {}  # child_name -> mandate
        # Legacy unsigned lineage is restrictive attribution, not a grant of
        # parental authority. Keep it explicit so restored children cannot be
        # mistaken for unrestricted roots merely because no mandate was cached.
        self._non_governing_spawn_lineage: set[str] = set()
        # child_name -> (DelegatedWallet, parent_wallet) for budgeted children, so
        # termination can release the unspent hold back to the parent (#2113).
        self._child_budgets: dict[str, tuple] = {}
        self._base_data_dir = (base_data_dir or Path.cwd()).expanduser().resolve()
        self._spawn_authority_registry = SpawnAuthorityRegistry(self._base_data_dir)
        if type(startup_roster_enabled) is not bool:
            raise TypeError("startup_roster_enabled must be a bool")
        self._startup_roster_enabled = startup_roster_enabled
        configured_roster = (
            startup_config_path
            if startup_config_path is not None
            else self._base_data_dir / "multi_agent.toml"
        )
        configured_roster = Path(configured_roster).expanduser()
        self._startup_config_path = (
            configured_roster.resolve()
            if startup_roster_enabled and configured_roster.is_file()
            else None
        )
        self._host_context_clause_registry = None
        self._host_context_publication_gate: asyncio.Event | None = None
        self._host_context_publication_state = _HostContextPublicationState()
        # A multi-agent host owns one mutable isolated-feature root.  The
        # per-agent namespace is derived below from the stable DID rather than
        # accepting the routing name as a path component.
        self._isolated_runtime_root = self._base_data_dir / "isolated_feature_runtime"
        # Host-owned, agent-scoped Telegram route evidence is injected before
        # feature initialization. No request payload can select this resolver.
        self._hosted_telegram_route_attestation_resolver_factory = (
            hosted_telegram_route_attestation_resolver_factory
        )
        self._hosted_isolated_runtime_lifecycle_policy_factory = (
            hosted_isolated_runtime_lifecycle_policy_factory
        )
        # The server owns this backend and closes it only after the manager has
        # terminally drained every child. Each hosted child gets the exact same
        # operational pool and delegates advisory sessions back to this owner.
        self._shared_postgres_backend = shared_postgres_backend
        self._lock = asyncio.Lock()
        # Inbound hosted A2A verification/authorization/task persistence holds
        # a shared reader lease from DID resolution through create_task.
        # Registration, onboarding, policy replacement, and removal take the
        # exclusive writer lease, so no request can commit work against a
        # sender or recipient topology that changed after its trust decision.
        # Independent recipients can nevertheless verify and persist together.
        self._a2a_lifecycle_lock = AsyncReaderWriterLock()
        self._a2a_policy_generation = 0
        self._a2a_hosted_policies: dict[str, A2AHostedPolicy] = {}
        # Registration captures the current epoch before initialization and
        # verifies it again while publishing, so an initializer that races a
        # fleet shutdown can never publish after the shutdown sweep observed
        # an empty fleet.  This is intentionally separate from the temporary
        # reaper-handoff seal below: it fences publication for one fleet sweep
        # without changing the manager's established reuse semantics.
        self._agent_registration_shutdown_epoch = 0
        self._agent_registration_sealed = False
        # Name admission is intentionally independent of published routing.
        # ``_agents`` cannot represent a create/load that has entered inception
        # or feature initialization, so using it as the duplicate guard lets
        # two same-name operations initialize independently and the loser
        # overwrite the winner's routing/reverse mapping.  The admission stays
        # owned until publication or terminal rollback completes.
        self._agent_operations: dict[str, AgentOperationAdmission] = {}
        # A verified agent can run wake-capable ready hooks before routing is
        # published. Preserve its exact DID/name/object binding across that
        # private window so control features see the same spawn depth ceiling
        # they will have after ``_register_agent`` commits.
        self._prepared_agent_names: dict[str, tuple[str, KestrelAgent]] = {}
        # A signed child is synchronously removed from every public routing
        # view at its admission deadline, before cancellation-resistant ready
        # work is joined.  The exact object remains in the private topology
        # maps until the normal receipt/runtime rollback owner removes it.
        self._expired_spawn_authority_routes: dict[str, KestrelAgent] = {}
        # A deadline may have to abandon the *wait*, never the work owner. Keep
        # cancellation-resistant admission tasks strongly owned until they
        # actually settle, then consume their terminal outcome.
        self._expired_admission_operations: set[asyncio.Task] = set()
        # Deadline racing runs an admission await in a child task. Preserve
        # the parent operation identity so a ready hook that requests its own
        # removal is treated as the spawn owner, not as an unrelated remover
        # that must join the very spawn awaiting that hook.
        self._admission_delegate_tasks: dict[
            asyncio.Task, AgentOperationAdmission
        ] = {}
        # A timed-out agent is no longer routable, but its exact shutdown task
        # may still own an active durable cognition delivery and its storage.
        # Retain those tasks independently of ``_agents`` so removal/restart
        # has a finite control-plane bound without abandoning durable state.
        self._quarantined_shutdown_reapers: dict[str, QuarantinedShutdownReaper] = {}
        # Completed asyncio tasks retain their coroutine frames and exception
        # tracebacks. Keep only bounded metadata after either safe completion
        # or an unsafe failure; active cleanup alone owns a live task.
        self._quarantined_shutdown_history: deque[QuarantinedShutdownHistory] = deque(
            maxlen=_QUARANTINED_SHUTDOWN_HISTORY_LIMIT
        )
        # Unsafe outcomes remain operator-visible until explicit process/host
        # remediation, but retain only strings/timestamps — never the finished
        # task or its traceback-bearing coroutine frame.
        self._unsafe_quarantined_shutdown_failures: dict[
            str, QuarantinedShutdownHistory
        ] = {}
        self._unsafe_quarantined_shutdown_failure_evictions = 0
        self._unsafe_quarantined_shutdown_failure_evictions_acknowledged_through = 0
        # The bounded failure history can evict individual unsafe records.
        # Once that happens, retaining every evicted routing name would itself
        # be unbounded.  Reserve *all* reuse until aggregate eviction evidence
        # is acknowledged instead: this is deliberately conservative, but it
        # is fixed-size and cannot silently reuse a name whose exact unsafe
        # cleanup evidence was discarded.
        self._unsafe_quarantined_shutdown_failure_overflow_reserved = False
        self._next_shutdown_reaper_id = 0
        # Ordinary stop-then-release cleanup is normally joined by the DELETE
        # that started it.  It still becomes removal-owned work immediately
        # after routing withdrawal, however, so a concurrent terminal drain
        # must be able to join it too.  Keep active tasks only until they
        # settle, but retain bounded metadata for an unsafe result: a release
        # can fail while a drain is blocked acquiring ``_lock`` and must not
        # become an implicit terminal success before that drain linearizes.
        self._inflight_removal_budget_releases: dict[
            str, InflightRemovalBudgetRelease
        ] = {}
        # This is the admission index.  Every owner that reaches a release for
        # one child while ``_lock`` is held must receive this *exact* task;
        # starting a second refund while the first override is suspended can
        # otherwise credit the same delegated allocation twice.
        self._inflight_removal_budget_releases_by_child: dict[
            str, InflightRemovalBudgetRelease
        ] = {}
        # Secure tenant-tree deletion runs in a worker after the agent is
        # stopped. It is admitted while ``_lock`` still linearizes removal,
        # then awaited only after every manager/A2A/scheduler lock is released.
        # A concurrent terminal drain therefore owns the exact worker even if
        # the administrative caller times out or is cancelled.
        self._inflight_runtime_offboardings: dict[int, InflightRuntimeOffboarding] = {}
        self._unsafe_removal_budget_release_failures: dict[
            str, UnsafeRemovalBudgetReleaseFailure
        ] = {}
        self._unsafe_removal_budget_release_failure_evictions = 0
        self._unsafe_removal_budget_release_failure_evictions_acknowledged_through = 0
        self._next_removal_budget_release_id = 0
        # A quarantined reaper is normally an intentional escape hatch for a
        # bounded single-agent DELETE. A terminal drain temporarily seals all
        # removal-cleanup admissions so every reaper *and ordinary budget
        # release* admitted before its linearization point is visible to that
        # drain. The separate drain lock makes that temporary seal exclusive:
        # one drain cannot reopen admissions beneath another drain that still
        # owns the terminal boundary.
        self._quarantined_shutdown_handoffs_sealed = False
        self._quarantined_shutdown_handoffs_open = asyncio.Event()
        self._quarantined_shutdown_handoffs_open.set()
        self._quarantined_shutdown_drain_lock = asyncio.Lock()
        # Reserved-port allocator (#1729 → #2358 codex rounds 4-6). A bare
        # monotonic counter avoided unload-reuse but couldn't express "this
        # port is taken by the HOST" without starving every port below it.
        # Reservations are NEVER removed (an unloaded agent's port stays
        # reserved — the #1729 guarantee); allocation scans for the first
        # free in-range port instead of incrementing.
        self._reserved_ports: set[int] = set()
        self._port_scan_start = 8801
        # Hard cap on dynamically-spawned agents so a runaway spawn loop can't
        # exhaust ports / resources (#1729).
        self._max_spawned_agents = int(os.environ.get("KESTREL_MAX_SPAWNED_AGENTS", "64"))
        # Bound cold-start parallelism so larger fleets gain overlap without
        # stampeding provider SDK initialization or exhausting DB descriptors.
        self._init_concurrency = max(
            1, int(os.environ.get("KESTREL_AGENT_INIT_CONCURRENCY", "4"))
        )
        # In-flight spawns whose mandate isn't registered yet (counts toward the
        # cap under the lock so concurrent spawns can't race past it).
        self._pending_spawns = 0
        self._rejected_spawn_slot_waiters: set[asyncio.Future[None]] = set()
        # Terminal parent removal snapshots a complete descendant tree only
        # after joining every earlier parent-owned spawn.  A per-owner token
        # keeps the exclusion closed until all overlapping finalizers settle.
        self._terminal_descendant_spawn_fences: dict[str, set[int]] = {}
        self._terminal_descendant_spawn_fence_members: dict[int, set[str]] = {}
        self._next_terminal_descendant_spawn_fence_nonce = 0
        # Per-agent initialization failures recorded by load_from_config so
        # the FastAPI lifespan can surface them via /health (#377 lifecycle
        # hardening for multi-agent boot).
        self._init_failures: list[tuple[str, Exception]] = []
        # A host scheduler needs a DID map for cold agents too.  A missing
        # local identity for an explicitly non-autostart agent is operationally
        # useful information, but must not roll back unrelated healthy agents.
        self._cold_scheduler_identity_failures: list[tuple[str, Exception]] = []
        # Scheduler authority is a *live desired-state registry*, not a frozen
        # copy of multi_agent.toml.  A host scheduler may wake a cold agent long
        # after startup, while DELETE is an administrative decision that must
        # take effect immediately even though the startup config remains on
        # disk for an intentional future restart.
        self._scheduler_authority_by_did: dict[
            str, tuple[str, LocalAgentConfig]
        ] = {}
        self._scheduler_authority_by_name: dict[str, str] = {}
        # Execution scope is intentionally separate from config authority:
        # dynamic registration publishes config first, activates the durable
        # protocol row second, then becomes visible to the host runner.
        self._scheduler_execution_scope: set[str] = set()
        self._scheduler_revoked_names: set[str] = set()
        self._scheduler_revoked_dids: set[str] = set()
        # Set only by the shared-PostgreSQL server topology before loading any
        # agents. SchedulerFeature then omits its agent-scoped polling runner;
        # the host runner is the sole executor and shares this manager's live
        # authority/lifecycle locks.
        self._scheduler_polling_managed_by_host = False
        # Private candidates undergoing app-owned integration. This staging
        # map is never consulted by routing/list APIs; it exists only so the
        # onboarding hook can install an A2A policy before publication.
        self._onboarding_agents: dict[str, KestrelAgent] = {}
        # Constructed hosted candidates remain private while a startup batch
        # initializes. Their locally loaded identity keys may verify a child's
        # signature during the early boot preflight, but they are never valid
        # parent authority for projection/publication until routed normally.
        self._initializing_agents: dict[str, KestrelAgent] = {}
        # A signed cold child consumes its fleet-cap slot as soon as preflight
        # verifies the receipt. The reservation is private (not governance or
        # routing authority) and is converted into projected authority only
        # after its parent is normally published.
        self._preflight_spawn_reservations: dict[
            str, tuple[KestrelAgent, str]
        ] = {}
        # The hosted executor and DELETE share this lock.  It is deliberately
        # exposed through ``scheduler_lifecycle_lock`` rather than having each
        # caller manufacture a private lock, which was the race allowing a
        # frozen scheduler map to resurrect a deleted tenant.
        self._scheduler_lifecycle_locks: dict[str, AsyncReaderWriterLock] = {}
        # The server owns app-level registration work (A2A resolver, feature
        # routes, and static assets).  It installs this hook before loading the
        # configured fleet so a scheduler cold wake cannot bypass onboarding.
        self._agent_registration_hook: Optional[
            Callable[
                [str, KestrelAgent],
                Awaitable[Optional[Callable[[], Awaitable[None]]]],
            ]
        ] = None
        # Shared-PostgreSQL hosts install this after the long-lived scheduler
        # storage is ready. It durably prepares a runtime-created DID and
        # returns an async rollback for state seeded before onboarding commits.
        self._scheduler_tenant_registration_hook: Optional[
            Callable[
                [str, str, LocalAgentConfig],
                Awaitable[Optional[Callable[[], Awaitable[None]]]],
            ]
        ] = None
        # LocalAgentConfig per agent created at runtime via create_agent —
        # consumed by the create-agent endpoint to persist registrations.
        self._created_configs: dict[str, "LocalAgentConfig"] = {}

    def _isolated_runtime_scope(self, agent_did: str) -> tuple[Path, str]:
        """Return this host's canonical mutable runtime scope for one agent.

        DIDs are stable across an agent rename, but their punctuation is not a
        portable directory name.  A SHA-256 namespace is deterministic,
        traversal-free, and collision-resistant, while KestrelAgent still
        performs the authoritative root-containment validation.
        """
        from kestrel_sovereign.features.isolated_runtime import (
            derive_isolated_runtime_namespace,
        )

        return (
            self._isolated_runtime_root,
            derive_isolated_runtime_namespace(agent_did),
        )

    @staticmethod
    def _hosted_agent_runtime_factory_configured(
        db_backend: str,
        database_url: Optional[str],
    ) -> bool:
        """Mirror the factory condition that supplies an explicit PG scope."""

        return db_backend.lower() == "postgres" and bool(database_url)

    def set_agent_registration_hook(
        self,
        hook: Optional[
            Callable[
                [str, KestrelAgent],
                Awaitable[Optional[Callable[[], Awaitable[None]]]],
            ]
        ],
    ) -> None:
        """Install the host-owned onboarding hook for every future register.

        The hook is intentionally an async contract: a cold scheduler wake
        does not return an agent to the executor until its host integrations
        are ready.  Existing loaded agents are handled by the normal startup
        pass; callers installing a hook before ``load_from_config`` cover both
        initial and dynamic registration through this one seam.
        """
        self._agent_registration_hook = hook

    def set_scheduler_tenant_registration_hook(
        self,
        hook: Optional[
            Callable[
                [str, str, LocalAgentConfig],
                Awaitable[Optional[Callable[[], Awaitable[None]]]],
            ]
        ],
    ) -> None:
        """Install the host-owned dynamic scheduler protocol boundary."""

        self._scheduler_tenant_registration_hook = hook

    def a2a_lifecycle_lease(self) -> AsyncReaderWriterLock:
        """Return the exclusive hosted A2A topology mutation lease."""
        return self._a2a_lifecycle_lock

    def a2a_execution_lease(self):
        """Return a shared hosted A2A verification-and-commit lease."""

        return self._a2a_lifecycle_lock.read()

    def install_a2a_hosted_policy(
        self,
        recipient: object,
        *,
        resolver: object,
        authorizer: object,
        router: object,
        requester: object,
    ) -> A2AHostedPolicy:
        """Publish immutable recipient policy while holding the lifecycle lease."""
        recipient_id = _loaded_agent_did(recipient)
        name = (
            self._agent_names.get(recipient_id)
            if isinstance(recipient_id, str)
            else None
        )
        if name is None:
            staged_names = [
                candidate_name
                for candidate_name, candidate in self._onboarding_agents.items()
                if candidate is recipient
            ]
            if len(staged_names) == 1:
                name = staged_names[0]
        if (
            not isinstance(recipient_id, str)
            or not recipient_id
            or not isinstance(name, str)
            or self._spawn_route_is_fenced(name, recipient)
            or (
                self._agents.get(name) is not recipient
                and self._onboarding_agents.get(name) is not recipient
            )
        ):
            raise RuntimeError(
                "Cannot install hosted A2A policy for an unpublished recipient"
            )
        self._a2a_policy_generation += 1
        policy = A2AHostedPolicy(
            generation=self._a2a_policy_generation,
            recipient=recipient,
            recipient_id=recipient_id,
            resolver=resolver,
            authorizer=authorizer,
            router=router,
            requester=requester,
        )
        self._a2a_hosted_policies[recipient_id] = policy
        return policy

    async def replace_a2a_hosted_policy(
        self,
        recipient: object,
        *,
        resolver: object,
        authorizer: object,
        router: object,
        requester: object,
    ) -> A2AHostedPolicy:
        """Writer-side API for an authorized live hosted-policy replacement."""
        async with self._a2a_lifecycle_lock:
            return self.install_a2a_hosted_policy(
                recipient,
                resolver=resolver,
                authorizer=authorizer,
                router=router,
                requester=requester,
            )

    def a2a_hosted_policy_for(
        self,
        recipient: object,
    ) -> Optional[A2AHostedPolicy]:
        """Return policy only while this exact recipient remains published."""
        recipient_id = _loaded_agent_did(recipient)
        name = (
            self._agent_names.get(recipient_id)
            if isinstance(recipient_id, str)
            else None
        )
        if (
            not isinstance(recipient_id, str)
            or not isinstance(name, str)
            or self._spawn_route_is_fenced(name, recipient)
            or self._agents.get(name) is not recipient
        ):
            return None
        policy = self._a2a_hosted_policies.get(recipient_id)
        if (
            policy is None
            or policy.recipient is not recipient
            or policy.recipient_id != recipient_id
        ):
            return None
        return policy

    async def authorize_a2a_legacy_unsigned_sender(
        self,
        recipient: object,
        claimed_sender: str,
        policy: A2AHostedPolicy,
    ) -> Optional[str]:
        """Authorize the sole hosted unsigned compatibility path.

        A pre-ceremony sender has no cryptographic signing DID, so accepting
        its ``metadata.sender`` is safe only when it is the unambiguous current
        *published display identity* of a loaded local agent, that agent is
        incapable of hybrid signing, and the recipient's immutable directory
        policy authorizes the sender's stable agent id. The display identity is
        resolved back through the manager's immutable routing mapping before
        authorization; an unsigned caller never chooses a routing key. This
        method is called while :meth:`a2a_execution_lease` is held;
        registration/removal/replacement therefore cannot change either
        endpoint or the policy during the directory await.
        """

        if (
            not isinstance(claimed_sender, str)
            or not claimed_sender
            or self.a2a_hosted_policy_for(recipient) is not policy
        ):
            return None
        matches: list[tuple[str, KestrelAgent, str]] = []
        for routing_name, candidate in self._agents.items():
            sender_id = _loaded_agent_did(candidate)
            if (
                not isinstance(sender_id, str)
                or self._agent_names.get(sender_id) != routing_name
                or self._agents.get(routing_name) is not candidate
                or self._spawn_route_is_fenced(routing_name, candidate)
                or self._published_a2a_display_identity(candidate) != claimed_sender
            ):
                continue
            matches.append((routing_name, candidate, sender_id))

        # Display names are mutable and not unique by construction. An
        # unsigned compatibility sender may be accepted only when the current
        # hosted fleet has exactly one matching published display identity.
        if len(matches) != 1:
            return None
        routing_name, sender, sender_id = matches[0]
        if sender is recipient or (
            self._agent_names.get(sender_id) != routing_name
            or self._agents.get(routing_name) is not sender
        ):
            return None

        identity = getattr(sender, "identity", None)
        # A loaded hybrid identity must never deliberately downgrade to the
        # unsigned transport.  Treat any retained hybrid key/material as a
        # signing capability even if a caller corrupts ``is_hybrid``.
        if identity is not None and (
            getattr(identity, "is_hybrid", False) is True
            or getattr(identity, "hybrid_keypair", None) is not None
            or bool(getattr(identity, "new_verification_methods", None))
        ):
            return None

        authorize = getattr(
            policy.authorizer,
            "authorize_legacy_local_sender_with_policy",
            None,
        )
        if not callable(authorize):
            return None
        try:
            result = authorize(
                sender_id,
                router=policy.router,
                requester=policy.requester,
            )
            if inspect.isawaitable(result):
                result = await result
        except Exception:  # noqa: BLE001 - recipient policy provider boundary
            logger.warning(
                "Hosted legacy A2A sender authorization failed",
                exc_info=True,
            )
            return None
        return sender_id if result is True else None

    async def _authorize_host_attested_local_a2a_route(
        self,
        *,
        sender: object,
        requester: object,
        peer: object,
    ) -> tuple[str, object]:
        """Resolve one live sender→recipient route under the caller's lease."""

        from kestrel_sovereign.features.peers.directory import (
            PeerAccessDeniedError,
            PeerIdentity,
            PeerNotFoundError,
            PeerProtocolError,
            PeerRequester,
        )

        if not isinstance(requester, PeerRequester) or not isinstance(
            peer, PeerIdentity
        ):
            raise PeerProtocolError("Local A2A route is malformed")
        sender_policy = self.a2a_hosted_policy_for(sender)
        sender_id = _loaded_agent_did(sender)
        if (
            sender_policy is None
            or sender_policy.requester is not requester
            or requester.identity != sender_id
        ):
            raise PeerAccessDeniedError("Local A2A sender is no longer published")

        recipient_name = self._agent_names.get(peer.agent_id)
        recipient = (
            self._agents.get(recipient_name)
            if isinstance(recipient_name, str)
            else None
        )
        recipient_policy = (
            self.a2a_hosted_policy_for(recipient)
            if recipient is not None
            else None
        )
        if (
            recipient is None
            or recipient_policy is None
            or recipient_policy.recipient_id != peer.agent_id
            or recipient_name != peer.routing_key
        ):
            raise PeerNotFoundError("Local A2A recipient is no longer published")

        authorize = getattr(
            recipient_policy.router,
            "authorize_inbound_sender",
            None,
        )
        if not callable(authorize):
            raise PeerAccessDeniedError("Local A2A recipient policy is unavailable")
        authorized = authorize(recipient_policy.requester, sender_id)
        if inspect.isawaitable(authorized):
            authorized = await authorized
        if (
            authorized is not True
            or self.a2a_hosted_policy_for(sender) is not sender_policy
            or self.a2a_hosted_policy_for(recipient) is not recipient_policy
        ):
            raise PeerAccessDeniedError("Local A2A sender is not authorized")
        return sender_id, recipient

    async def get_host_attested_local_a2a_task(
        self,
        *,
        sender: object,
        requester: object,
        peer: object,
        task_id: str,
    ) -> dict[str, object]:
        """Read a same-host result only through its durable creator."""

        from kestrel_sovereign.features.peers.directory import (
            PeerNotFoundError,
            PeerProtocolError,
        )

        if not isinstance(task_id, str) or not task_id:
            raise PeerProtocolError("Local A2A task read is malformed")
        async with self.a2a_execution_lease():
            sender_id, recipient = await self._authorize_host_attested_local_a2a_route(
                sender=sender,
                requester=requester,
                peer=peer,
            )
            task_manager = getattr(recipient, "task_manager", None)
            if task_manager is None:
                raise PeerNotFoundError("Local A2A task does not exist")
            recipient_id = _loaded_agent_did(recipient)
            if recipient_id is None:
                raise PeerNotFoundError("Local A2A task does not exist")
            task = await task_manager.get_task_for_creator(
                task_id,
                sender_id,
                recipient_agent_id=recipient_id,
            )
            if task is None:
                # Unknown and cross-principal ids share one public result.
                raise PeerNotFoundError("Local A2A task does not exist")
            return task.model_dump()

    async def subscribe_host_attested_local_a2a_task(
        self,
        *,
        sender: object,
        requester: object,
        peer: object,
        task_id: str,
    ):
        """Stream a same-host task only after creator-scoped admission."""

        from kestrel_sovereign.features.peers.directory import (
            PeerNotFoundError,
            PeerProtocolError,
            PeerSubscriptionEvent,
        )

        if not isinstance(task_id, str) or not task_id:
            raise PeerProtocolError("Local A2A task subscription is malformed")
        subscription = None
        async with self.a2a_execution_lease():
            sender_id, recipient = await self._authorize_host_attested_local_a2a_route(
                sender=sender,
                requester=requester,
                peer=peer,
            )
            task_manager = getattr(recipient, "task_manager", None)
            recipient_id = _loaded_agent_did(recipient)
            if recipient_id is None:
                raise PeerNotFoundError("Local A2A task does not exist")
            if task_manager is None or await task_manager.get_task_for_creator(
                task_id,
                sender_id,
                recipient_agent_id=recipient_id,
            ) is None:
                raise PeerNotFoundError("Local A2A task does not exist")
            subscription = task_manager.subscribe(
                task_id,
                creator_agent_id=sender_id,
                recipient_agent_id=recipient_id,
            )
            try:
                first = await anext(subscription)
            except StopAsyncIteration as exc:
                raise PeerNotFoundError("Local A2A task does not exist") from exc
        try:
            yield PeerSubscriptionEvent(
                event=first.get("event"),
                data=first.get("data"),
            )
            async for event in subscription:
                yield PeerSubscriptionEvent(
                    event=event.get("event"),
                    data=event.get("data"),
                )
        finally:
            await subscription.aclose()

    async def cancel_host_attested_local_a2a_task(
        self,
        *,
        sender: object,
        requester: object,
        peer: object,
        task_id: str,
        payload: object,
    ) -> dict[str, object]:
        """Cancel same-host work through a non-serializable authority seam.

        Pre-ceremony agents cannot sign a cancellation envelope. The local
        router capability is bound to the exact published sender during host
        onboarding; this method revalidates both endpoints and the recipient's
        live directory policy under one lifecycle lease before the recipient
        store applies its atomic creator/recipient predicate.
        """

        from collections.abc import Mapping

        from kestrel_sovereign.a2a.task_manager import (
            TaskCancellationAuthorizationError,
        )
        from kestrel_sovereign.features.peers.directory import (
            PeerAccessDeniedError,
            PeerNotFoundError,
            PeerProtocolError,
            PeerTaskConflictError,
        )

        if (
            not isinstance(task_id, str)
            or not task_id
            or not isinstance(payload, Mapping)
        ):
            raise PeerProtocolError("Local cancellation request is malformed")
        reason = payload.get("reason")
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 4096:
            raise PeerProtocolError("Local cancellation reason is invalid")

        async with self.a2a_execution_lease():
            sender_id, recipient = await self._authorize_host_attested_local_a2a_route(
                sender=sender,
                requester=requester,
                peer=peer,
            )

            task_manager = getattr(recipient, "task_manager", None)
            if task_manager is None:
                raise PeerNotFoundError("Local cancellation recipient is unavailable")
            recipient_id = _loaded_agent_did(recipient)
            if recipient_id is None:
                raise PeerNotFoundError("Local cancellation recipient is unavailable")
            current = await task_manager.get_task_for_creator(
                task_id,
                sender_id,
                recipient_agent_id=recipient_id,
            )
            if current is None:
                raise PeerNotFoundError("Local cancellation task does not exist")
            try:
                task = await task_manager.cancel_task(
                    task_id,
                    reason=reason,
                    agent_name=sender_id,
                    recipient_agent_id=peer.agent_id,
                )
            except TaskCancellationAuthorizationError as exc:
                raise PeerAccessDeniedError(
                    "Local cancellation actor is not authorized"
                ) from exc
            except ValueError as exc:
                raise PeerTaskConflictError(
                    "Local cancellation conflicts with task state"
                ) from exc

            state = getattr(getattr(task, "status", None), "state", None)
            return {
                "id": getattr(task, "id", task_id),
                "status": getattr(state, "value", state),
                "cancellation_receipt": (
                    getattr(task, "metadata", None) or {}
                ).get("cancellation_receipt"),
            }

    @staticmethod
    def _published_a2a_display_identity(agent: object) -> Optional[str]:
        """Return the live display identity published by ``/api/agents``.

        Agent cards (and therefore ``PeersFeature``) expose the agent's live
        effective name, not the manager's immutable routing key. Keep this
        lookup synchronous and in-memory so it is safe under the A2A lifecycle
        lease and cannot race a storage read with unpublication. Test doubles
        may not implement ``resolve_effective_name``; their live ``_agent_name``
        remains the equivalent published identity.
        """

        resolver = getattr(agent, "resolve_effective_name", None)
        if callable(resolver):
            try:
                value = resolver(default=None)
            except Exception:  # noqa: BLE001 - fail closed at identity boundary
                return None
        else:
            value = getattr(agent, "_agent_name", None)
        return value if isinstance(value, str) and value.strip() else None

    def _revoke_a2a_hosted_policy(self, recipient: object) -> None:
        recipient_id = _loaded_agent_did(recipient)
        if not isinstance(recipient_id, str):
            return
        policy = self._a2a_hosted_policies.get(recipient_id)
        if policy is not None and policy.recipient is recipient:
            self._a2a_policy_generation += 1
            self._a2a_hosted_policies.pop(recipient_id, None)

    def a2a_sender_identity_witness(
        self,
        signing_did: str,
    ) -> tuple[str, object | None, object | None, str]:
        """Snapshot one sender identity while the caller holds the A2A lease.

        The object references detect replacement even when a new agent reuses
        the same DID and keys; the digest binds the exact verification methods
        consumed by cryptographic verification. ``external`` is a legitimate
        no-local-match state, while ``ambiguous`` always fails closed.
        """
        from kestrel_sovereign.a2a.did_registry import (
            local_a2a_verification_document,
        )
        from kestrel_sovereign.a2a.envelope_signing import (
            verification_document_fingerprint,
        )

        matches = []
        for agent in self._agents.values():
            document = local_a2a_verification_document(agent, signing_did)
            if document is not None:
                matches.append((agent, getattr(agent, "identity", None), document))
        if not matches:
            return ("external", None, None, "")
        if len(matches) != 1:
            return ("ambiguous", None, None, "")
        agent, identity, document = matches[0]
        return (
            "local",
            agent,
            identity,
            verification_document_fingerprint(document),
        )

    def set_scheduler_polling_managed_by_host(self, managed: bool) -> None:
        """Declare that one host runner exclusively owns scheduler polling."""

        self._scheduler_polling_managed_by_host = bool(managed)

    def scheduler_lifecycle_lock(self, agent_id: str) -> AsyncReaderWriterLock:
        """Return the exclusive lifecycle writer for a scheduler DID."""
        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("scheduler lifecycle lock requires a non-empty DID")
        return self._scheduler_lifecycle_locks.setdefault(
            agent_id, AsyncReaderWriterLock()
        )

    def scheduler_execution_lease(self, agent_id: str):
        """Return a shared execution lease drained by lifecycle writers.

        Scheduled effects hold this lease through their terminal scheduler CAS.
        Removal, rollout authority mutation, and cold initialization retain the
        exclusive lifecycle writer, closing admission before they mutate live
        desired state or routing.
        """

        return self.scheduler_lifecycle_lock(agent_id).read()

    def scheduler_authority_for(
        self, agent_id: str
    ) -> Optional[tuple[str, LocalAgentConfig]]:
        """Return currently desired scheduler authority for ``agent_id``.

        A ``None`` result is a fail-closed authorization denial.  In
        particular, a DID revoked by DELETE remains absent even if the static
        startup config still contains it.
        """
        return self._scheduler_authority_by_did.get(agent_id)

    def is_scheduler_agent_authorized(self, agent_id: str) -> bool:
        """Whether ``agent_id`` remains in the live scheduler desired state."""
        return agent_id in self._scheduler_execution_scope

    def scheduler_authorized_agent_ids(self) -> tuple[str, ...]:
        """Return a snapshot of the live scheduler authorization set."""
        return tuple(self._scheduler_execution_scope)

    def _scheduler_cold_authority_ancestors(
        self,
        agent_id: str,
    ) -> tuple[tuple[str, str, LocalAgentConfig], ...]:
        """Return a cold signed parent chain in root-first load order."""

        current_did = agent_id
        seen = {agent_id}
        ancestors: list[tuple[str, str, LocalAgentConfig]] = []
        while True:
            witness = self._spawn_authority_registry.get(current_did)
            if witness is None or not witness.active:
                break
            parent_did = witness.parent_did
            if parent_did in seen:
                raise RuntimeError("Host spawn authority contains a cycle")
            seen.add(parent_did)
            authority_candidates = (
                *self._agents.values(),
                *self._initializing_agents.values(),
            )
            if any(
                parent_did in _loaded_agent_bound_dids(candidate)
                for candidate in authority_candidates
            ):
                break
            authority = self._scheduler_authority_by_did.get(parent_did)
            if authority is None:
                raise LookupError(
                    "Refusing scheduler cold wake because its signed parent "
                    "is not authorized by live host desired state"
                )
            parent_name, parent_config = authority
            ancestors.append((parent_did, parent_name, parent_config))
            current_did = parent_did
        ancestors.reverse()
        return tuple(ancestors)

    async def _load_scheduler_cold_authority_ancestors(
        self,
        agent_id: str,
        agent_name: str,
    ) -> None:
        """Load every missing scheduler-authorized parent before its child."""

        for (
            parent_did,
            parent_name,
            parent_config,
        ) in self._scheduler_cold_authority_ancestors(agent_id):
            await self._await_admitted_parent_candidate(
                agent_name,
                parent_did,
            )
            if any(
                parent_did in _loaded_agent_bound_dids(candidate)
                for candidate in (
                    *self._agents.values(),
                    *self._initializing_agents.values(),
                )
            ):
                continue
            await self.load_agent(
                parent_name,
                parent_config,
                expected_agent_id=parent_did,
            )

    def _seed_scheduler_authority(
        self, mapping: dict[str, tuple[str, LocalAgentConfig]]
    ) -> None:
        """Replace configured scheduler authority while preserving revocations."""
        active = {
            agent_id: entry
            for agent_id, entry in mapping.items()
            if entry[0] not in self._scheduler_revoked_names
            and agent_id not in self._scheduler_revoked_dids
        }
        self._scheduler_authority_by_did = active
        self._scheduler_authority_by_name = {
            name: agent_id for agent_id, (name, _config) in active.items()
        }
        self._scheduler_execution_scope = set(active)
        for agent_id in active:
            self.scheduler_lifecycle_lock(agent_id)

    def _revoke_scheduler_authority(
        self, name: str, agent_id: Optional[str]
    ) -> Optional[tuple[str, LocalAgentConfig]]:
        """Remove one tenant from live scheduler authority, returning its entry."""
        self._scheduler_revoked_names.add(name)
        if isinstance(agent_id, str) and agent_id:
            self._scheduler_revoked_dids.add(agent_id)
        known_id = agent_id or self._scheduler_authority_by_name.get(name)
        if not isinstance(known_id, str) or not known_id:
            return None
        entry = self._scheduler_authority_by_did.pop(known_id, None)
        self._scheduler_execution_scope.discard(known_id)
        self._scheduler_authority_by_name.pop(name, None)
        return entry

    def _restore_scheduler_authority(
        self, agent_id: Optional[str], entry: Optional[tuple[str, LocalAgentConfig]]
    ) -> None:
        """Restore authority only when an administrative removal did not land."""
        if entry is None or not isinstance(agent_id, str) or not agent_id:
            return
        name, _config = entry
        self._scheduler_revoked_names.discard(name)
        self._scheduler_revoked_dids.discard(agent_id)
        self._scheduler_authority_by_did[agent_id] = entry
        self._scheduler_authority_by_name[name] = agent_id
        self._scheduler_execution_scope.add(agent_id)

    async def revoke_terminal_spawn_scheduler_authority(
        self,
        name: str,
        agent_id: str,
    ) -> None:
        """Drain and revoke one terminal spawned DID from scheduler execution.

        Terminal spawn retirement is identity-scoped.  A later replacement may
        legitimately reuse the display name, so this path records only the old
        DID as revoked and removes a name index only when it still identifies
        that exact DID.
        """

        if not isinstance(name, str) or not name:
            raise ValueError("terminal scheduler revocation requires an agent name")
        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("terminal scheduler revocation requires an agent DID")
        async with self.scheduler_lifecycle_lock(agent_id):
            async with self._lock:
                authority = self._scheduler_authority_by_did.get(agent_id)
                if authority is not None and self._canonical_agent_name(
                    authority[0]
                ) != self._canonical_agent_name(name):
                    raise RuntimeError(
                        "Refusing terminal scheduler revocation with conflicting "
                        "name authority"
                    )
                self._scheduler_revoked_dids.add(agent_id)
                self._scheduler_execution_scope.discard(agent_id)
                self._scheduler_authority_by_did.pop(agent_id, None)
                for indexed_name, indexed_did in tuple(
                    self._scheduler_authority_by_name.items()
                ):
                    if indexed_did == agent_id:
                        self._scheduler_authority_by_name.pop(indexed_name, None)

    async def _begin_dynamic_scheduler_tenant_registration(
        self,
        name: str,
        agent_id: str,
        config: LocalAgentConfig,
        *,
        scheduler_lifecycle_lock_held: bool = False,
    ) -> Optional[_DynamicSchedulerTenantRegistration]:
        """Register a new hosted DID before any feature post-load mutation."""

        if not self._scheduler_polling_managed_by_host:
            return None
        if scheduler_lifecycle_lock_held:
            # Hosted cold execution already owns this DID's non-reentrant
            # lifecycle lease. It is authorized configured state, not a
            # dynamic tenant registration: validate the exact live authority
            # under the manager state lock, then leave ownership untouched.
            async with self._lock:
                existing = self._scheduler_authority_by_did.get(agent_id)
                if existing is None:
                    raise LookupError(
                        "refusing hosted scheduler cold initialization without "
                        "live manager authority"
                    )
                if (
                    existing != (name, config)
                    or self._scheduler_authority_by_name.get(name) != agent_id
                ):
                    raise RuntimeError(
                        "refusing hosted scheduler cold initialization because "
                        "manager authority does not match the claimed tenant"
                    )
            return None
        lifecycle_lock = self.scheduler_lifecycle_lock(agent_id)
        await lifecycle_lock.acquire()
        lock_owned = True
        authority_added = False
        try:
            async with self._lock:
                existing = self._scheduler_authority_by_did.get(agent_id)
                if existing is not None:
                    if existing != (name, config):
                        raise RuntimeError(
                            "scheduler tenant DID is already registered to a "
                            "different hosted configuration"
                        )
                    lifecycle_lock.release()
                    lock_owned = False
                    return None
                if (
                    name in self._scheduler_revoked_names
                    or agent_id in self._scheduler_revoked_dids
                ):
                    raise LookupError(
                        "refusing to reauthorize a scheduler tenant removed "
                        "during this host process"
                    )
                known_id = self._scheduler_authority_by_name.get(name)
                if known_id is not None and known_id != agent_id:
                    raise RuntimeError(
                        "scheduler tenant name is already registered to a "
                        "different hosted DID"
                    )
                hook = self._scheduler_tenant_registration_hook
                if hook is None:
                    raise RuntimeError(
                        "shared PostgreSQL host scheduler is not ready to "
                        "register a runtime-created tenant"
                    )
                self._scheduler_authority_by_did[agent_id] = (name, config)
                self._scheduler_authority_by_name[name] = agent_id
                authority_added = True

            registration_task = asyncio.create_task(
                hook(name, agent_id, config),
                name=f"scheduler_tenant_registration:{name}",
            )
            cancelled, failure = await await_lifecycle_task_completion(
                registration_task
            )
            if failure is not None:
                raise failure
            rollback_protocol = registration_task.result()
            registration = _DynamicSchedulerTenantRegistration(
                self,
                name,
                agent_id,
                config,
                lifecycle_lock,
                rollback_protocol,
            )
            if cancelled:
                rollback_task = asyncio.create_task(
                    registration.rollback(),
                    name=f"scheduler_tenant_registration_cancel:{name}",
                )
                _, rollback_failure = await await_lifecycle_task_completion(
                    rollback_task
                )
                lock_owned = False
                if rollback_failure is not None:
                    raise rollback_failure
                raise asyncio.CancelledError()
            return registration
        except BaseException:
            if authority_added and lock_owned:
                self._scheduler_execution_scope.discard(agent_id)
                entry = self._scheduler_authority_by_did.get(agent_id)
                if entry == (name, config):
                    self._scheduler_authority_by_did.pop(agent_id, None)
                    if self._scheduler_authority_by_name.get(name) == agent_id:
                        self._scheduler_authority_by_name.pop(name, None)
            if lock_owned:
                lifecycle_lock.release()
            raise

    async def _on_agent_registered(self, name: str, agent: KestrelAgent) -> None:
        """Finish app-owned onboarding before routing publication.

        The historical name is retained for the host integration seam, but
        callers deliberately run this while the candidate is still private.
        Feature routes are live-gated through the manager, so a mounted route
        cannot reach this agent until the later atomic registration commit.
        """
        hook = self._agent_registration_hook
        if hook is not None:
            rollback = await hook(name, agent)
            if rollback is not None and not callable(rollback):
                raise TypeError("agent registration hook rollback must be callable")
            if rollback is not None:
                agent._host_onboarding_rollback = rollback

    async def _complete_registered_agent_readiness(
        self,
        agent: KestrelAgent,
    ) -> bool:
        """Finish deferred hooks after onboarding releases the topology writer.

        A ready hook may start a cognition turn whose tools acquire a shared
        A2A execution lease.  Registration therefore must not run it while
        retaining the exclusive lifecycle writer used for publication and
        app-owned onboarding.  Drive the hook task to settlement so caller
        cancellation cannot strand a successfully published agent forever in
        its deferred-ready state. Return whether the caller was cancelled while
        the manager-owned hook task settled. Once publication has committed,
        that cancellation cannot turn the load into a reported failure: callers
        such as ``create_agent`` still have a persistence handoff to finish.
        """

        gate = self._host_context_publication_gate
        complete_readiness = getattr(
            agent,
            "complete_deferred_agent_readiness",
            None,
        )
        if gate is not None and gate.is_set() and callable(complete_readiness):
            deadline = self._spawn_admission_deadline(agent)
            route_name = next(
                (
                    name
                    for name, candidate in self._agents.items()
                    if candidate is agent
                ),
                None,
            )
            if deadline is not None and route_name is None:
                raise RuntimeError(
                    "Cannot complete signed child readiness without an exact route"
                )

            async def complete_with_authority_deadline() -> None:
                if deadline is None:
                    await complete_readiness()
                    return
                assert route_name is not None
                await self._await_before_spawn_admission_deadline(
                    complete_readiness,
                    deadline=deadline,
                    phase="agent readiness",
                    on_expiry=lambda: self._fence_expired_spawn_route(
                        route_name,
                        agent,
                    ),
                )

            task = asyncio.create_task(
                complete_with_authority_deadline(),
                name=(
                    "agent_deferred_readiness:"
                    f"{_loaded_agent_did(agent) or 'unknown'}"
                ),
            )
            cancelled, failure = await await_lifecycle_task_completion(task)
            if isinstance(failure, asyncio.CancelledError):
                # ``on_agent_ready`` is a best-effort post-publication phase.
                # Real agents consume a hook-owned child cancellation inside
                # that phase, but keep this boundary defensive for compatible
                # agent implementations: reporting load cancellation here
                # would leave a routable committed agent while its caller
                # skips persistence handoff.  Caller cancellation is carried
                # separately by ``cancelled`` because the manager-owned task
                # is shielded and driven to settlement.
                logger.warning(
                    "Deferred best-effort readiness cancelled for published "
                    "agent %r; preserving the committed load",
                    _loaded_agent_did(agent),
                )
                return cancelled
            if failure is not None:
                raise failure
            return cancelled
        return False

    async def complete_deferred_agent_readiness(self) -> None:
        """Complete startup readiness behind the registration writer boundary.

        The server opens the host-context gate after publishing host policy.
        A registration may already have inserted an agent into ``_agents`` but
        still be awaiting app-owned onboarding under the A2A lifecycle writer.
        Taking that same writer here makes the startup sweep linearize either
        before publication or after onboarding; it can never run a feature's
        ``on_agent_ready`` hook in the partial interval between them.

        Registrations that publish after this sweep retain the complementary
        post-onboarding call in the registration path.
        """

        async with self._a2a_lifecycle_lock:
            agents = list(self._agents.values())

        seen: set[int] = set()
        for agent in agents:
            if id(agent) in seen:
                continue
            seen.add(id(agent))
            route_name = next(
                (
                    name
                    for name, candidate in self._agents.items()
                    if candidate is agent
                ),
                None,
            )
            try:
                cancelled = await self._complete_registered_agent_readiness(agent)
            except PersistedSpawnMandateExpiredError as exc:
                # A finite child can expire while the server-owned context gate
                # is opening. Its exact route is already fenced by the deadline
                # owner; preserve the host's normal per-agent failure isolation
                # so one terminal child cannot withhold healthy peers.
                failed_name = route_name or _loaded_agent_did(agent) or "unknown"
                logger.error(
                    "Deferred readiness expired for agent %r: %s",
                    failed_name,
                    exc,
                    exc_info=True,
                )
                self._init_failures.append((failed_name, exc))
                continue
            if cancelled:
                raise asyncio.CancelledError()

    async def _initialize_agent(
        self,
        name: str,
        config: LocalAgentConfig,
        *,
        scheduler_lifecycle_lock_held: bool = False,
    ) -> KestrelAgent:
        """Construct and initialize an agent without publishing it to the fleet.

        Keeping initialization separate from registration lets startup initialize
        independent configured agents concurrently, then publish successes in
        deterministic config order. Dynamic ``load_agent`` still uses the same
        path and registration point.

        Args:
            name: Agent name (used as routing key).
            config: LocalAgentConfig with data_dir, port, autostart.

        Returns:
            Initialized KestrelAgent instance.

        Raises:
            ValueError: If agent data directory is invalid.
        """
        resolved_dir = config.resolve_data_dir(self._base_data_dir)
        identity_export_dir = (
            config.resolve_identity_export_dir(self._base_data_dir) or resolved_dir
        )

        # Validate the data directory
        errors = config.validate_runtime(base_dir=self._base_data_dir)
        if errors:
            raise ValueError(f"Agent '{name}' validation failed: {'; '.join(errors)}")

        # Get DID from the agent's database
        agent_did = await read_anchor_agent_did(
            str(resolved_dir),
            mode=AgentDIDLookupMode.INITIALIZATION,
        )
        async with self._a2a_lifecycle_lock:
            admission = self._agent_operations.get(
                self._canonical_agent_name(name)
            )
            if admission is not None:
                if not self._operation_is_admitted(admission):
                    raise RuntimeError(
                        "Refusing agent initialization because its manager "
                        "operation is no longer admitted"
                    )
                if (
                    admission.resolved_agent_id is not None
                    and admission.resolved_agent_id != agent_did
                ):
                    raise RuntimeError(
                        "Agent operation resolved conflicting durable identities"
                    )
                admission.resolved_agent_id = agent_did
                if admission.agent_id_resolution_event is not None:
                    admission.agent_id_resolution_event.set()

        # Check database backend
        db_backend = os.environ.get("KESTREL_DB_BACKEND", "sqlite")
        database_url = os.environ.get("KESTREL_DATABASE_URL")

        db_path = str(resolved_dir / "kestrel_prime.db")

        # Each agent gets its own LLMService (mutable model state). Bind it to
        # THIS agent's data root: an in-process host shares one environment
        # across every agent, so ``KESTREL_DB_PATH`` cannot name each agent's
        # directory and usage rows would all land in one agent's DB (#2769).
        llm_service = LLMService(agent_data_dir=resolved_dir)
        hosted_telegram_resolver = None
        if self._hosted_telegram_route_attestation_resolver_factory is not None:
            hosted_telegram_resolver = (
                self._hosted_telegram_route_attestation_resolver_factory(
                    name, agent_did, config
                )
            )
        hosted_runtime_configured = self._hosted_agent_runtime_factory_configured(
            db_backend,
            database_url,
        )
        lifecycle_policy = None
        if (
            hosted_runtime_configured
            and self._hosted_isolated_runtime_lifecycle_policy_factory is not None
        ):
            lifecycle_policy = (
                self._hosted_isolated_runtime_lifecycle_policy_factory(
                    name, agent_did, config
                )
            )
            if lifecycle_policy is not None and not isinstance(
                lifecycle_policy, HostedIsolatedRuntimeLifecyclePolicy
            ):
                raise TypeError(
                    "hosted isolated runtime lifecycle policy factory returned "
                    "an invalid policy"
                )

        # Build allowed_features set from config (None = load all)
        allowed_features = set(config.features) if config.features is not None else None
        # Materialization is an exact, tenant-local operator approval.  Parse
        # the per-agent multi_agent.toml block before construction so an
        # invalid profile fails this agent's startup rather than silently
        # disabling its approved closure.
        semantic_inference_profile = None
        semantic_inference_limits = None
        semantic_maintenance_limits = None
        semantic_maintenance_allow_prior_verified_snapshot = False
        semantic_capabilities = None
        semantic_inference_configured = config.semantic_inference is not None
        semantic_maintenance_configured = config.semantic_maintenance is not None
        semantic_capabilities_configured = config.semantic_capabilities is not None
        if semantic_inference_configured:
            from kestrel_sovereign.knowledge.inference import (
                inference_limits_from_config,
                inference_profile_from_config,
                validate_inference_profile,
            )

            semantic_inference_profile = inference_profile_from_config(
                config.semantic_inference
            )
            semantic_inference_limits = inference_limits_from_config(
                config.semantic_inference
            )
            if semantic_inference_profile is not None:
                validate_inference_profile(semantic_inference_profile)
        if config.semantic_maintenance is not None:
            from kestrel_sovereign.knowledge.maintenance import (
                maintenance_allows_prior_verified_snapshot,
                maintenance_limits_from_config,
            )

            semantic_maintenance_limits = maintenance_limits_from_config(
                config.semantic_maintenance
            )
            semantic_maintenance_allow_prior_verified_snapshot = (
                maintenance_allows_prior_verified_snapshot(
                    config.semantic_maintenance
                )
            )
        if config.semantic_capabilities is not None:
            from kestrel_sovereign.knowledge.capabilities import (
                semantic_capabilities_from_config,
            )

            semantic_capabilities = semantic_capabilities_from_config(
                config.semantic_capabilities
            )

        agent: Optional[KestrelAgent] = None
        scheduler_registration: Optional[
            _DynamicSchedulerTenantRegistration
        ] = None
        try:
            scheduler_registration = (
                await self._begin_dynamic_scheduler_tenant_registration(
                    name,
                    agent_did,
                    config,
                    scheduler_lifecycle_lock_held=(
                        scheduler_lifecycle_lock_held
                    ),
                )
            )
            if hosted_runtime_configured:
                runtime_root, runtime_namespace = self._isolated_runtime_scope(
                    agent_did
                )
                shared_postgres_pool = (
                    self._shared_postgres_backend.operational_pool
                    if self._shared_postgres_backend is not None
                    else None
                )
                agent = KestrelAgent(
                    did=agent_did,
                    storage_path=db_path,
                    llm_service=llm_service,
                    database_url=database_url,
                    db_backend="postgres",
                    pg_pool=shared_postgres_pool,
                    shared_postgres_advisory_backend=(
                        self._shared_postgres_backend
                    ),
                    allowed_features=allowed_features,
                    host_context_clause_registry=self._host_context_clause_registry,
                    hosted_telegram_route_attestation_resolver=hosted_telegram_resolver,
                    identity_export_dir=identity_export_dir,
                    isolated_runtime_root=runtime_root,
                    isolated_runtime_namespace=runtime_namespace,
                    isolated_runtime_legacy_root=resolved_dir / "feature_venvs",
                    isolated_runtime_hosted=True,
                    isolated_runtime_idle_timeout_seconds=(
                        lifecycle_policy.idle_timeout_seconds
                        if lifecycle_policy is not None
                        else None
                    ),
                    isolated_runtime_idle_timeouts=(
                        lifecycle_policy.idle_timeouts
                        if lifecycle_policy is not None
                        else None
                    ),
                    isolated_runtime_telemetry_observer=(
                        lifecycle_policy.telemetry_observer
                        if lifecycle_policy is not None
                        else None
                    ),
                    semantic_inference_profile=semantic_inference_profile,
                    semantic_inference_limits=semantic_inference_limits,
                    semantic_maintenance_limits=semantic_maintenance_limits,
                    semantic_capabilities=semantic_capabilities,
                    semantic_inference_configured=semantic_inference_configured,
                    semantic_maintenance_configured=semantic_maintenance_configured,
                    semantic_capabilities_configured=semantic_capabilities_configured,
                    semantic_maintenance_allow_prior_verified_snapshot=(
                        semantic_maintenance_allow_prior_verified_snapshot
                    ),
                )
            else:
                agent = KestrelAgent(
                    did=agent_did,
                    storage_path=db_path,
                    llm_service=llm_service,
                    allowed_features=allowed_features,
                    host_context_clause_registry=self._host_context_clause_registry,
                    hosted_telegram_route_attestation_resolver=hosted_telegram_resolver,
                    identity_export_dir=identity_export_dir,
                    semantic_inference_profile=semantic_inference_profile,
                    semantic_inference_limits=semantic_inference_limits,
                    semantic_maintenance_limits=semantic_maintenance_limits,
                    semantic_capabilities=semantic_capabilities,
                    semantic_inference_configured=semantic_inference_configured,
                    semantic_maintenance_configured=semantic_maintenance_configured,
                    semantic_capabilities_configured=semantic_capabilities_configured,
                    semantic_maintenance_allow_prior_verified_snapshot=(
                        semantic_maintenance_allow_prior_verified_snapshot
                    ),
                )

            self._initializing_agents[name] = agent
            if admission is not None and admission.candidate_staged_event is not None:
                admission.candidate_staged_event.set()
            # ``_initializing_agents`` becomes visible before storage has read
            # this candidate's durable spawned_by edge. Concurrent children may
            # wait on this explicit evidence boundary, but must never infer that
            # an as-yet-unread parent is a root.
            agent._agent_manager_authority_evidence_loaded = False
            agent._agent_manager_authority_evidence_event = asyncio.Event()
            # Durable evidence and routable publication are distinct boundaries.
            # A concurrently restoring signed child may verify this candidate as
            # its parent as soon as evidence is loaded, but it cannot project the
            # relation until the parent has committed routing.  Give that child a
            # cancellation-safe boundary to await without polling manager state.
            agent._agent_manager_published = False
            agent._agent_manager_publication_event = asyncio.Event()
            # Publish this before feature initialization. A shared PostgreSQL
            # host must never briefly arm an agent-scoped runner that lacks the
            # manager's live authority and per-DID lifecycle lock.
            agent._scheduler_polling_managed_by_host = (
                self._scheduler_polling_managed_by_host
            )
            agent._host_context_publication_gate = (
                self._host_context_publication_gate
            )
            # Keep the live state box attached before initialize: ready hooks
            # can enter cognition while the agent is still absent from
            # ``_agents`` and therefore invisible to manager fan-out.  Leave
            # the observed generation unset until the cognition barrier or
            # registration seam performs a validated bind; construction alone
            # is not proof that the prompt builder already consumes it.
            agent._host_context_publication_state = (
                self._host_context_publication_state
            )
            agent._host_context_publication_generation = None
            # Authority admission owns the ready boundary independently of the
            # server's context-publication gate. A manager used without the
            # server must still keep wake-capable hooks private until the
            # durable spawn relation has been verified and routing published.
            defer_readiness = getattr(
                agent,
                "defer_agent_readiness_to_host",
                None,
            )
            if callable(defer_readiness):
                defer_readiness()
            else:
                # Compatibility for narrow test/legacy candidates that expose
                # the earlier manager-owned readiness seam only.
                agent._host_ready_hooks_deferred = True
            if scheduler_registration is not None:
                agent._dynamic_scheduler_tenant_registration = (
                    scheduler_registration
                )
            stage_callback = _BATCH_STAGE_CALLBACK.get()
            active_boot_semaphore = _BATCH_ACTIVE_BOOT_SEMAPHORE.get()
            active_boot_slot_held = False

            async def verify_host_authority_before_active_boot(
                candidate: KestrelAgent,
            ) -> None:
                nonlocal active_boot_slot_held
                if stage_callback is not None:
                    # Storage has now frozen this candidate's durable mandate.
                    # Release the bounded construction slot while every batch
                    # candidate reaches the same evidence boundary. Receipt
                    # repair also runs outside that active-boot slot: a child
                    # can wait for its parent's repaired evidence, so holding a
                    # concurrency-one slot here would deadlock the parent.
                    # Child-first config order therefore cannot hide a staged
                    # parent's own mandate at concurrency=1.
                    if (
                        active_boot_semaphore is not None
                        and active_boot_slot_held
                    ):
                        active_boot_semaphore.release()
                        active_boot_slot_held = False
                    await stage_callback(name, candidate)
                await self._recover_interrupted_spawn_receipt(name, candidate)
                # An active host witness and an unsigned local edge are a
                # recoverable crash window, not verified parent authority.
                # Publish evidence only after repair has durably restored the
                # signed receipt so a concurrently staged descendant cannot
                # recursively reject this parent based on the stale edge.
                candidate._agent_manager_authority_evidence_loaded = True
                candidate._agent_manager_authority_evidence_event.set()
                if stage_callback is not None and active_boot_semaphore is not None:
                    await active_boot_semaphore.acquire()
                    active_boot_slot_held = True
                await self._await_initializing_parent_evidence(
                    name,
                    candidate,
                )
                async with self._a2a_lifecycle_lock:
                    admission = self._agent_operations.get(
                        self._canonical_agent_name(name)
                    )
                    if admission is None or not self._operation_is_admitted(admission):
                        raise RuntimeError(
                            "Refusing active agent initialization because the "
                            "manager operation is no longer admitted"
                        )
                    async with self._lock:
                        self._verify_agent_authority(name, candidate)

            agent._host_authority_preflight = (
                verify_host_authority_before_active_boot
            )
            agent._host_authority_expiry_callback = (
                lambda candidate: self._expire_host_authority_candidate(
                    name, candidate
                )
            )
            if active_boot_semaphore is not None:
                await active_boot_semaphore.acquire()
                active_boot_slot_held = True
            try:
                await agent.initialize()
            finally:
                if active_boot_semaphore is not None and active_boot_slot_held:
                    active_boot_semaphore.release()
                    active_boot_slot_held = False
        except BaseException as initialization_failure:
            if agent is not None:
                await self._discard_unpublished_initialized_agent(name, agent)
                if isinstance(
                    initialization_failure,
                    PersistedSpawnMandateExpiredError,
                ):
                    agent_id = _loaded_agent_did(agent)
                    if not isinstance(agent_id, str) or not agent_id:
                        raise RuntimeError(
                            "Expired spawned candidate has no stable identity"
                        ) from initialization_failure
                    self.record_expired_spawn_retirement(
                        name,
                        expected_child_did=agent_id,
                        data_dir=config.resolve_data_dir(self._base_data_dir),
                    )
            else:
                if scheduler_registration is not None:
                    rollback_task = asyncio.create_task(
                        scheduler_registration.rollback(),
                        name=f"scheduler_tenant_rollback:{name}",
                    )
                    _, rollback_failure = await await_lifecycle_task_completion(
                        rollback_task
                    )
                    if rollback_failure is not None:
                        logger.error(
                            "Failed to roll back scheduler registration for %r",
                            name,
                            exc_info=(
                                type(rollback_failure),
                                rollback_failure,
                                rollback_failure.__traceback__,
                            ),
                        )
                try:
                    await llm_service.close()
                except Exception:
                    logger.warning(
                        "Failed to close LLM service for uninitialized agent %r",
                        name,
                        exc_info=True,
                    )
            raise
        # Spawn-mandate enforcement (restricted_tools hook + spawn_mandate attach)
        # is reattached inside KestrelAgent.initialize() from the persisted
        # delegation edge (#2137), so it covers every boot path — not just this
        # one — uniformly.

        assert agent is not None
        return agent

    @staticmethod
    def _commit_dynamic_scheduler_registration(agent: KestrelAgent) -> None:
        """Commit a pending dynamic scheduler tenant after host onboarding."""

        registration = getattr(
            agent, "_dynamic_scheduler_tenant_registration", None
        )
        if not isinstance(registration, _DynamicSchedulerTenantRegistration):
            return
        registration.commit()
        delattr(agent, "_dynamic_scheduler_tenant_registration")

    @staticmethod
    def _commit_host_onboarding(agent: KestrelAgent) -> None:
        """Retire rollback custody after routing publication commits."""

        vars(agent).pop("_host_onboarding_rollback", None)

    @staticmethod
    async def _rollback_host_onboarding(
        name: str,
        agent: KestrelAgent,
    ) -> tuple[bool, Optional[BaseException]]:
        """Join app-state rollback while the registration writer is held."""

        onboarding_rollback = vars(agent).pop(
            "_host_onboarding_rollback",
            None,
        )
        if onboarding_rollback is None:
            return False, None
        rollback_task = asyncio.create_task(
            onboarding_rollback(),
            name=f"host_onboarding_rollback:{name}",
        )
        return await await_lifecycle_task_completion(rollback_task)

    @staticmethod
    async def _shutdown_unregistered_agent(
        name: str, agent: KestrelAgent
    ) -> None:
        """Release a fully or partially initialized agent never published.

        Concurrent fleet startup can be cancelled after one initializer has
        completed but before deterministic registration begins. Such agents
        are invisible to ``shutdown_all()``, so the startup path owns their
        cleanup explicitly.
        """
        cancelled = False
        onboarding_failure: Optional[BaseException] = None
        onboarding_rollback = vars(agent).pop(
            "_host_onboarding_rollback",
            None,
        )
        if onboarding_rollback is not None:
            rollback_task = asyncio.create_task(
                onboarding_rollback(),
                name=f"host_onboarding_rollback:{name}",
            )
            rollback_cancelled, onboarding_failure = (
                await await_lifecycle_task_completion(rollback_task)
            )
            cancelled = cancelled or rollback_cancelled
        rollback_failure: Optional[BaseException] = None
        registration = getattr(
            agent, "_dynamic_scheduler_tenant_registration", None
        )
        if isinstance(registration, _DynamicSchedulerTenantRegistration):
            # Stop future selection before feature shutdown can yield. The
            # owned rollback task keeps the global protocol cleanup joinable
            # across repeated caller cancellation.
            rollback_task = asyncio.create_task(
                registration.rollback(),
                name=f"scheduler_tenant_rollback:{name}",
            )
            rollback_cancelled, rollback_failure = (
                await await_lifecycle_task_completion(rollback_task)
            )
            cancelled = cancelled or rollback_cancelled
            delattr(agent, "_dynamic_scheduler_tenant_registration")
        try:
            await asyncio.wait_for(agent.shutdown(), timeout=SHUTDOWN_TIMEOUT)
        except asyncio.CancelledError:
            cancelled = True
            logger.warning(
                "Unregistered agent %r shutdown was cancelled; "
                "joining durable cleanup before propagating cancellation",
                name,
            )
        except (asyncio.TimeoutError, Exception) as exc:
            logger.warning(
                "Failed to clean up unregistered agent %r: %s",
                name,
                exc,
                exc_info=True,
            )
        # A bounded KestrelAgent shutdown can hand durable dispatcher release
        # plus storage close to its own shielded continuation.  This startup
        # cleanup path is still its lifecycle owner, so join that continuation
        # rather than dropping the only reference while its SQLite backend is
        # live. Lightweight test doubles do not expose this contract.
        if _has_shutdown_completion_contract(agent):
            cancelled = await await_agent_shutdown_completion(agent) or cancelled
        if cancelled:
            raise asyncio.CancelledError()
        failures = [
            failure
            for failure in (onboarding_failure, rollback_failure)
            if failure is not None
        ]
        if failures:
            _raise_lifecycle_outcomes(
                "Unregistered agent integration rollback failed",
                failures,
            )

    def _withdraw_initialized_agent(self, name: str, agent: KestrelAgent) -> None:
        """Synchronously withdraw one initialized result from published state.

        Callers that are withdrawing a result which was ever published must
        hold ``_a2a_lifecycle_lock``.  The no-publication cleanup path owns the
        result privately, so it can use the same one source of truth without a
        lifecycle lease.  Shutdown intentionally remains outside this method:
        it can run arbitrary slow cleanup only after topology is no longer
        observable.
        """

        # Wake any concurrently initializing descendants before withdrawing
        # this private candidate.  The false marker makes the wake a failure,
        # never evidence that the parent was verified.
        authority_evidence_event = vars(agent).get(
            "_agent_manager_authority_evidence_event"
        )
        if isinstance(authority_evidence_event, asyncio.Event):
            authority_evidence_event.set()
        publication_event = vars(agent).get("_agent_manager_publication_event")
        if isinstance(publication_event, asyncio.Event):
            # Wake descendants on failure too.  The false publication marker
            # below makes the wake a refusal, never counterfeit authority.
            agent._agent_manager_published = False
            publication_event.set()
        self._revoke_a2a_hosted_policy(agent)
        if self._onboarding_agents.get(name) is agent:
            self._onboarding_agents.pop(name, None)
        if self._initializing_agents.get(name) is agent:
            self._initializing_agents.pop(name, None)
        if self._agents.get(name) is agent:
            self._agents.pop(name, None)
        agent_id = _loaded_agent_did(agent)
        if agent_id is not None and self._agent_names.get(agent_id) == name:
            self._agent_names.pop(agent_id, None)
        if (
            agent_id is not None
            and self._prepared_agent_names.get(agent_id) == (name, agent)
        ):
            self._prepared_agent_names.pop(agent_id, None)
        self._withdraw_restored_spawn_authority(name, agent_id)
        withdrawn_bound_dids = _loaded_agent_bound_dids(agent)
        surviving_candidates: list[KestrelAgent] = []
        for candidate in (
            *self._agents.values(),
            *self._onboarding_agents.values(),
            *self._initializing_agents.values(),
            *(prepared[1] for prepared in self._prepared_agent_names.values()),
            *(
                admission.child
                for admission in self._agent_operations.values()
                if admission.child is not None
            ),
        ):
            if candidate is agent or any(
                candidate is existing for existing in surviving_candidates
            ):
                continue
            surviving_candidates.append(candidate)

        def retains_unsigned_lineage(candidate: KestrelAgent, bound_did: str) -> bool:
            mandate = vars(candidate).get("_persisted_spawn_mandate")
            return (
                bound_did in _loaded_agent_bound_dids(candidate)
                and isinstance(mandate, SpawnMandate)
                and not mandate.parent_signature
            )

        for bound_did in withdrawn_bound_dids:
            guarded_by_survivor = any(
                retains_unsigned_lineage(candidate, bound_did)
                for candidate in surviving_candidates
            )
            if guarded_by_survivor:
                self._non_governing_spawn_lineage.add(bound_did)
            else:
                self._non_governing_spawn_lineage.discard(bound_did)
        self._release_preflight_spawn_reservation(name, agent, agent_id)
        if vars(agent).get("_agent_manager_authority_prepared") is self:
            agent._agent_manager_authority_prepared = None
        if vars(agent).get("_agent_manager") is self:
            agent._agent_manager = None
        canonical_name = self._canonical_agent_name(name)
        if self._expired_spawn_authority_routes.get(canonical_name) is agent:
            self._expired_spawn_authority_routes.pop(canonical_name, None)
        disarm_host_authority_deadline(agent)

    async def _discard_unpublished_initialized_agent(
        self, name: str, agent: KestrelAgent
    ) -> None:
        """Withdraw and clean up one initialized agent that never committed.

        A dynamic scheduler registration deliberately remains private until
        host onboarding commits.  This method is therefore the one cleanup
        owner for every initialized result that cannot be published, whether
        publication was rejected, onboarding failed, or cancellation arrived
        while waiting for the A2A lifecycle writer.
        """

        expiry_cleanup = vars(agent).get(
            "_host_authority_expiry_cleanup_task"
        )
        if (
            isinstance(expiry_cleanup, asyncio.Task)
            and expiry_cleanup is not asyncio.current_task()
        ):
            _cancelled, failure = await await_lifecycle_task_completion(
                expiry_cleanup
            )
            if failure is not None:
                raise failure
            return

        # Prepared restored authority is a real cap/topology reservation even
        # before routing publication. Withdraw it under the same lock order as
        # preparation so a concurrent spawn or restore cannot admit against a
        # half-rolled-back projection. An initializer cancelled while waiting
        # for that boundary remains privately owned and must not reacquire the
        # writer that prevented it from preparing in the first place.
        if vars(agent).get("_agent_manager_authority_prepared") is self:
            async with self._a2a_lifecycle_lock:
                async with self._lock:
                    self._withdraw_initialized_agent(name, agent)
        else:
            self._withdraw_initialized_agent(name, agent)
        await self._shutdown_unregistered_agent(name, agent)

    async def _discard_unpublished_initialized_agents(
        self,
        agents: list[tuple[str, KestrelAgent]],
        *,
        already_withdrawn: bool = False,
    ) -> bool:
        """Finish cleanup for every unpublished initialization despite cancel.

        A batch load can have several fully initialized agents waiting behind
        the A2A lifecycle writer.  Run each cleanup in its own owned task and
        join every one through the lifecycle helper, so cancelling the batch
        cannot strand either an invisible agent or its pending scheduler
        registration.  Returns whether the caller observed cancellation after
        all cleanup reached a terminal result.
        """

        cleanup = (
            self._shutdown_unregistered_agent
            if already_withdrawn
            else self._discard_unpublished_initialized_agent
        )
        cleanup_tasks = [
            asyncio.create_task(
                cleanup(name, agent),
                name=f"agent_unpublished_cleanup:{name}",
            )
            for name, agent in agents
        ]
        cancelled = False
        failures: list[Exception] = []
        fatal: Optional[BaseException] = None
        for cleanup_task in cleanup_tasks:
            cleanup_cancelled, cleanup_failure = (
                await await_lifecycle_task_completion(cleanup_task)
            )
            cancelled = cancelled or cleanup_cancelled
            if cleanup_failure is None:
                continue
            if isinstance(cleanup_failure, asyncio.CancelledError):
                cancelled = True
            elif isinstance(cleanup_failure, Exception):
                failures.append(cleanup_failure)
            elif fatal is None:
                fatal = cleanup_failure
        # The cleanup tasks are now terminal, so preserve the caller's
        # cancellation in preference to any cleanup error.  Otherwise a batch
        # onboarding failure can turn a later cancellation into an apparently
        # ordinary startup failure after it already completed all cleanup.
        if cancelled:
            raise asyncio.CancelledError()
        if fatal is not None:
            raise fatal
        if failures:
            raise ExceptionGroup(
                "One or more unpublished initialized agents failed to clean up",
                failures,
            )
        return False

    def _restore_persisted_spawn_authority(
        self,
        name: str,
        agent: KestrelAgent,
        agent_id: str,
        *,
        project: bool = True,
        allow_initializing_parent: bool = False,
    ) -> None:
        """Project one durable ``spawned_by`` edge into runtime indexes.

        The maps are caches used by the existing control paths, not an
        authority source.  A fresh spawn projects its final signed mandate as
        restrictive provisional authority before publication; ``_do_spawn``
        later converts its reserved cap slot at the budget + governance commit.
        """

        # Read concrete instance state rather than ``getattr`` so permissive
        # test/proxy objects cannot synthesize an apparent authority value.
        mandate = vars(agent).get("_persisted_spawn_mandate")
        if mandate is None:
            return
        if not isinstance(mandate, SpawnMandate):
            raise TypeError("persisted spawn authority must be a SpawnMandate")

        admission = self._agent_operations.get(self._canonical_agent_name(name))
        live_spawn_admission = (
            admission is not None
            and admission.kind in {"spawn", "direct-spawn-test"}
            and not admission.committed
        )

        # Legacy spawned_by edges are unsigned attribution. They continue to
        # re-apply restrictions on the child, but can never recreate parental
        # governance. A signed receipt is restored only when its parent is
        # loaded and the signature verifies against that parent's live,
        # trusted identity.
        if not mandate.parent_signature:
            self._refuse_unrestored_delegated_budget(name, agent)
            if project:
                self._non_governing_spawn_lineage.update(
                    _loaded_agent_bound_dids(agent)
                )
            return
        parent_items = list(self._agents.items())
        if (
            live_spawn_admission
            and admission is not None
            and admission.spawn_parent is not None
            and not any(
                published is admission.spawn_parent
                for _name, published in parent_items
            )
        ):
            # The public spawn admission owns the exact parent object even for
            # compatibility callers that did not first publish that root in
            # this manager.  Never generalize this to cold restoration.
            parent_items.append(
                ("<admitted-spawn-parent>", admission.spawn_parent)
            )
        if allow_initializing_parent:
            for candidate_name, candidate in self._initializing_agents.items():
                if not any(
                    published is candidate for _name, published in parent_items
                ):
                    parent_items.append((candidate_name, candidate))
        parent_matches = [
            (candidate_name, candidate)
            for candidate_name, candidate in parent_items
            if mandate.parent_did in _loaded_agent_bound_dids(candidate)
        ]
        if len(parent_matches) > 1:
            raise RuntimeError(
                "Persisted spawn mandate parent identity is ambiguous"
            )
        if not parent_matches:
            raise PersistedSpawnParentUnavailableError(
                "Refusing to publish a signed child before its parent authority "
                "is loaded"
            )
        _parent_name, parent = parent_matches[0]
        parent_lineage = vars(parent).get("_persisted_spawn_mandate")
        if (
            any(
                parent_did in self._non_governing_spawn_lineage
                for parent_did in _loaded_agent_bound_dids(parent)
            )
            or (
                isinstance(parent_lineage, SpawnMandate)
                and not parent_lineage.parent_signature
            )
        ):
            raise RuntimeError(
                "Refusing to restore signed authority through a non-governing parent"
            )
        authority_parent_did = _loaded_agent_did(parent)
        if authority_parent_did is None:
            raise PersistedSpawnParentUnavailableError(
                "Refusing to publish a signed child without stable parent authority"
            )
        parent_state = vars(parent)
        parent_identity = parent_state.get("identity")
        parent_private_key = parent_state.get("_private_key")
        public_key = None
        public_key_getter = getattr(parent_private_key, "public_key", None)
        if callable(public_key_getter):
            public_key = public_key_getter()
        from kestrel_sovereign.spawn.mandate import verify_mandate

        if not verify_mandate(
            mandate,
            public_key,
            parent_identity=parent_identity,
        ):
            raise RuntimeError("Persisted spawn mandate signature is invalid")

        if (
            mandate.ttl_seconds > 0
            and remaining_spawn_ttl_seconds(
                mandate.created_at,
                mandate.ttl_seconds,
            ) <= 0
        ):
            raise PersistedSpawnMandateExpiredError(
                "Persisted spawn mandate has expired"
            )

        if mandate.child_did != agent_id:
            raise RuntimeError(
                "Persisted spawn mandate child DID does not match the loaded agent"
            )
        parent_did = authority_parent_did
        if not isinstance(parent_did, str) or not parent_did or parent_did == agent_id:
            raise RuntimeError("Persisted spawn mandate has an invalid parent DID")

        canonical_name = self._canonical_agent_name(name)
        for recorded_parent, children in self._parent_children.items():
            if recorded_parent == parent_did:
                continue
            if any(
                self._canonical_agent_name(child) == canonical_name
                for child in children
            ):
                raise RuntimeError(
                    f"Agent {name!r} is already attached to a different parent"
                )

        existing = self._child_mandates.get(name)
        if existing is not None:
            existing_parent_did = existing.parent_did
            existing_parent_matches = [
                candidate
                for candidate in self._agents.values()
                if existing_parent_did in _loaded_agent_bound_dids(candidate)
            ]
            if len(existing_parent_matches) == 1:
                existing_parent_did = (
                    _loaded_agent_did(existing_parent_matches[0])
                    or existing_parent_did
                )
            if (
                existing_parent_did != parent_did
                or existing.child_did != agent_id
            ):
                raise RuntimeError(
                    f"Agent {name!r} has conflicting persisted spawn authority"
                )

        # Treat the durable child->parent edges as a directed forest.  Walk
        # DIDs rather than routing names so load order and renames cannot hide
        # a cycle (A spawned_by B, B spawned_by A).  Any already-corrupt loop
        # is also refused instead of being extended into recursive control
        # paths such as terminate_child and delegation-chain construction.
        authority_candidates = list(self._agents.values())
        for candidate in self._initializing_agents.values():
            if not any(candidate is existing for existing in authority_candidates):
                authority_candidates.append(candidate)

        def canonical_authority_did(claimed_did: str) -> str:
            matches = [
                candidate
                for candidate in authority_candidates
                if claimed_did in _loaded_agent_bound_dids(candidate)
            ]
            if len(matches) > 1:
                raise RuntimeError(
                    "Persisted spawn mandate parent identity is ambiguous"
                )
            if len(matches) == 1:
                return _loaded_agent_did(matches[0]) or claimed_did
            return claimed_did

        authority_edges = list(self._child_mandates.values())
        for candidate in self._initializing_agents.values():
            staged = vars(candidate).get("_persisted_spawn_mandate")
            if not isinstance(staged, SpawnMandate) or not staged.parent_signature:
                continue
            if staged.child_did not in _loaded_agent_bound_dids(candidate):
                raise RuntimeError(
                    "Persisted spawn mandate child DID does not match its "
                    "initializing agent"
                )
            if not any(
                edge.child_did == staged.child_did
                and edge.parent_did == staged.parent_did
                for edge in authority_edges
            ):
                authority_edges.append(staged)

        parent_by_child: dict[str, str] = {}
        for recorded in authority_edges:
            recorded_child = recorded.child_did
            recorded_parent = canonical_authority_did(recorded.parent_did)
            recorded_child = canonical_authority_did(recorded_child)
            if not isinstance(recorded_child, str) or not recorded_child:
                continue
            prior_parent = parent_by_child.get(recorded_child)
            if prior_parent is not None and prior_parent != recorded_parent:
                raise RuntimeError(
                    "Persisted spawn authority has conflicting parents for one DID"
                )
            parent_by_child[recorded_child] = recorded_parent
        ancestor = parent_did
        visited: set[str] = set()
        while ancestor in parent_by_child:
            if ancestor == agent_id or ancestor in visited:
                raise RuntimeError("Persisted spawn authority contains a cycle")
            visited.add(ancestor)
            ancestor = parent_by_child[ancestor]
        if ancestor == agent_id:
            raise RuntimeError("Persisted spawn authority contains a cycle")

        # A signed child cannot restore more delegation depth than its parent
        # still owns.  This is the restart equivalent of _run_admitted_spawn's
        # live depth reduction, except a persisted signed receipt is immutable:
        # reject an oversized edge rather than rewriting its signed fields.
        parent_mandate = next(
            (
                recorded
                for child_name, recorded in self._child_mandates.items()
                if self._canonical_agent_name(child_name)
                == self._canonical_agent_name(_parent_name)
            ),
            None,
        )
        if parent_mandate is None:
            staged_parent_mandate = vars(parent).get("_persisted_spawn_mandate")
            if (
                isinstance(staged_parent_mandate, SpawnMandate)
                and staged_parent_mandate.parent_signature
            ):
                parent_mandate = staged_parent_mandate
        if parent_mandate is not None:
            if parent_mandate.child_did not in _loaded_agent_bound_dids(parent):
                raise RuntimeError(
                    "Persisted parent mandate does not match its loaded identity"
                )
            parent_depth = getattr(parent_mandate, "max_child_depth", 0)
            allowed_depth = parent_depth - 1
            if parent_depth <= 0 or mandate.max_child_depth > allowed_depth:
                raise RuntimeError(
                    "Persisted spawn mandate exceeds the parent's remaining "
                    "child-depth authority"
                )

        reservation = self._preflight_spawn_reservations.get(canonical_name)
        owns_reservation = (
            reservation is not None
            and reservation[0] is agent
            and reservation[1] == agent_id
        )
        durable_witness = self._spawn_authority_registry.get(agent_id)
        # A cold receipt arbitrates ahead of live spawn admissions: those
        # admissions remain rollback owners until governance commit.  Count
        # every durable/cold/retained slot, but neither this receipt's own
        # preflight reservation nor a still-active live admission that must
        # yield if the cold receipt consumes the final slot.
        live_admission_slots = sum(
            1
            for candidate in self._agent_operations.values()
            if candidate.spawn_slot_active
        )
        cold_restore_slots = max(
            0,
            self._spawn_cap_slots_in_use()
            - int(owns_reservation)
            - live_admission_slots,
        )
        if (
            existing is None
            and not live_spawn_admission
            and durable_witness is None
            and cold_restore_slots >= self._max_spawned_agents
        ):
            raise RuntimeError(
                "Persisted spawn authority exceeds the configured spawned-agent cap"
            )
        # Signature, identity, ancestry, and TTL are verified above. Only now
        # may unsupported cold delegated-wallet custody refuse publication;
        # doing this first would mask an expired receipt and prevent its host
        # witness from advancing to terminal retirement.
        self._refuse_unrestored_delegated_budget(name, agent)
        if not project:
            return

        self._non_governing_spawn_lineage.difference_update(
            _loaded_agent_bound_dids(agent)
        )
        children = self._parent_children.setdefault(parent_did, [])
        if not any(
            self._canonical_agent_name(child) == canonical_name for child in children
        ):
            children.append(name)
        self._child_mandates[name] = mandate
        if live_spawn_admission:
            # This is enforcement, not yet a governance/cap commit.  The
            # admission remains the rollback owner and live-spawn TTL
            # registration still occurs only after ``spawn_agent`` succeeds.
            admission.provisional_spawn_authority = True
        else:
            self._ensure_spawn_lifecycle().restore_persisted_child(
                name,
                mandate,
                authority_parent_did=parent_did,
                arm_ttl=False,
            )
        if owns_reservation and not live_spawn_admission:
            self._preflight_spawn_reservations.pop(canonical_name, None)
            if vars(agent).get("_agent_manager_authority_reserved") is self:
                agent._agent_manager_authority_reserved = None

    def _prepare_agent_authority(
        self, name: str, agent: KestrelAgent
    ) -> None:
        """Reserve verified durable authority before hosted readiness may fire.

        Callers hold the A2A lifecycle writer and manager state lock. The
        projection is deliberately committed before wake-capable ready hooks;
        unpublished cleanup withdraws it if any later onboarding stage fails.
        """

        agent_id = _loaded_agent_did(agent)
        if not isinstance(agent_id, str) or not agent_id:
            raise RuntimeError(
                f"Cannot prepare agent {name!r} without a concrete agent DID"
            )
        prepared = self._prepared_agent_names.get(agent_id)
        if prepared is not None and prepared != (name, agent):
            raise RuntimeError(
                "Cannot prepare an agent DID already owned by another private "
                f"registration: {agent_id!r}"
            )
        bound_name = self._agent_names.get(agent_id)
        if (
            bound_name is not None
            and self._agents.get(bound_name) is not agent
        ):
            raise RuntimeError(
                "Cannot prepare an agent DID already routed under a different "
                f"name: {agent_id!r} -> {bound_name!r}"
            )
        self._restore_persisted_spawn_authority(
            name,
            agent,
            agent_id,
        )
        self._prepared_agent_names[agent_id] = (name, agent)
        # Ready hooks can invoke agent-owned control features. Bind those
        # features to this host manager at the same boundary that projects
        # their verified authority; unpublished cleanup removes the binding
        # if readiness or onboarding does not commit.
        agent._agent_manager = self
        agent._agent_manager_authority_prepared = self

    async def _recover_interrupted_spawn_receipt(
        self,
        name: str,
        agent: KestrelAgent,
    ) -> None:
        """Repair only the crash window after a signed host witness commits.

        The host registry is an independent, durable, signed authority rail.
        If the process died after committing it but before replacing the
        child's unsigned inception edge, a hosted restart may copy that exact
        receipt back after verifying its slot, signature, and live parent.
        Direct child boot never invokes this repair boundary.
        """

        agent_id = _loaded_agent_did(agent)
        if not isinstance(agent_id, str) or not agent_id:
            return
        witness = self._spawn_authority_registry.get(agent_id)
        local_mandate = vars(agent).get("_persisted_spawn_mandate")
        if (
            witness is None
            or not witness.active
            or (
                isinstance(local_mandate, SpawnMandate)
                and bool(local_mandate.parent_signature)
            )
        ):
            return

        admission = self._agent_operations.get(self._canonical_agent_name(name))
        candidate_config = (
            admission.spawn_candidate_config if admission is not None else None
        )
        if not isinstance(candidate_config, LocalAgentConfig):
            candidate_config = self._created_configs.get(name)
        if (
            witness.child_name.casefold() != name.casefold()
            or not isinstance(candidate_config, LocalAgentConfig)
            or not self._spawn_authority_registry.same_data_slot(
                candidate_config,
                witness.config,
            )
        ):
            raise RuntimeError(
                "Refusing interrupted spawn receipt repair for a different host slot"
            )
        if local_mandate is not None:
            if not isinstance(local_mandate, SpawnMandate):
                raise TypeError("persisted spawn authority must be a SpawnMandate")
            unsigned_witness = copy.deepcopy(witness.mandate)
            unsigned_witness.parent_signature = None
            if witness.proposal_created_at is not None:
                unsigned_witness.created_at = witness.proposal_created_at
            if local_mandate.to_dict() != unsigned_witness.to_dict():
                raise RuntimeError(
                    "Refusing interrupted spawn receipt repair over changed local authority"
                )

        raw_storage = vars(agent).get("_raw_storage")
        graph = getattr(raw_storage, "graph", None)
        if graph is None:
            raise RuntimeError(
                "Cannot repair an interrupted spawn receipt without durable storage"
            )
        repaired_mandate = copy.deepcopy(witness.mandate)
        vars(agent)["_persisted_spawn_mandate"] = repaired_mandate
        try:
            # The unsigned local edge did not arm the KestrelAgent preflight.
            # Establish signed expiry custody before waiting on parent evidence
            # or performing the receipt repair itself.
            arm_host_authority_deadline(agent, repaired_mandate)
            await self._await_initializing_parent_evidence(name, agent)
            async with self._lock:
                self._restore_persisted_spawn_authority(
                    name,
                    agent,
                    agent_id,
                    project=False,
                    allow_initializing_parent=True,
                )
            await graph.add_trusted_cross_agent_edge(
                agent_id,
                witness.parent_did,
                "spawned_by",
                properties=repaired_mandate.to_edge_properties(),
            )
        except BaseException:
            vars(agent)["_persisted_spawn_mandate"] = local_mandate
            raise

    def _verify_agent_authority(
        self,
        name: str,
        agent: KestrelAgent,
        *,
        _verification_stack: Optional[set[int]] = None,
    ) -> None:
        """Validate durable authority without publishing its runtime caches.

        This is the early hosted-boot boundary used after storage opens but
        before any active provider/feature phase.  The later preparation step
        repeats the validation under the governance commit lock and projects
        the verified relation for readiness and publication.
        """

        verification_stack = (
            _verification_stack if _verification_stack is not None else set()
        )
        identity = id(agent)
        if identity in verification_stack:
            raise RuntimeError("Persisted spawn authority contains a cycle")
        verification_stack.add(identity)
        try:
            agent_id = _loaded_agent_did(agent)
            if not isinstance(agent_id, str) or not agent_id:
                raise RuntimeError(
                    f"Cannot verify agent {name!r} without a concrete agent DID"
                )
            mandate = vars(agent).get("_persisted_spawn_mandate")
            admission = self._agent_operations.get(
                self._canonical_agent_name(name)
            )
            candidate_config = (
                admission.spawn_candidate_config
                if admission is not None
                else None
            )
            if not isinstance(candidate_config, LocalAgentConfig):
                candidate_config = self._created_configs.get(name)
            host_witness = self._spawn_authority_registry.get(agent_id)
            pending_witness = self._spawn_authority_registry.pending_for_slot(
                child_name=name,
                config=(
                    candidate_config
                    if isinstance(candidate_config, LocalAgentConfig)
                    else None
                ),
            )
            owns_pending_witness = (
                pending_witness is not None
                and admission is not None
                and admission.kind in {"spawn", "direct-spawn-test"}
                and admission.spawn_authority_pending_id
                == pending_witness.reservation_id
            )
            if pending_witness is not None and not owns_pending_witness:
                raise RuntimeError(
                    "Refusing a child denied restart by pending spawn authority"
                )
            # A standalone root has no spawn relation. Its private manager can
            # share the host registry with sibling roots whose descendants are
            # free to reuse its runtime name. Only a configured or mandated
            # child participates in the registry's child-slot namespace.
            slot_witness = (
                self._spawn_authority_registry.authoritative_for_slot(
                    child_name=name,
                    config=(
                        candidate_config
                        if isinstance(candidate_config, LocalAgentConfig)
                        else None
                    ),
                )
                if self._startup_roster_enabled
                or isinstance(candidate_config, LocalAgentConfig)
                or isinstance(mandate, SpawnMandate)
                else None
            )
            if (
                slot_witness is not None
                and slot_witness.child_did != agent_id
            ):
                raise RuntimeError(
                    "host spawn witness binds this name or data slot to a "
                    "different child DID"
                )
            if host_witness is not None:
                if not host_witness.active:
                    raise RuntimeError(
                        "Refusing a child denied restart by its host spawn witness"
                    )
                if host_witness.child_name.casefold() != name.casefold():
                    raise RuntimeError(
                        "Host spawn witness names a different local child"
                    )
                if (
                    not isinstance(mandate, SpawnMandate)
                    or not mandate.parent_signature
                ):
                    raise RuntimeError(
                        "host spawn witness requires a matching signed local receipt"
                    )
                if mandate.to_dict() != host_witness.mandate.to_dict():
                    raise RuntimeError(
                        "Host spawn witness does not match the signed local receipt"
                    )
            if isinstance(mandate, SpawnMandate) and mandate.parent_signature:
                staged_parent_matches = [
                    (candidate_name, candidate)
                    for candidate_name, candidate in self._initializing_agents.items()
                    if candidate is not agent
                    and mandate.parent_did in _loaded_agent_bound_dids(candidate)
                ]
                if len(staged_parent_matches) > 1:
                    raise RuntimeError(
                        "Persisted spawn mandate parent identity is ambiguous"
                    )
                if staged_parent_matches:
                    parent_name, staged_parent = staged_parent_matches[0]
                    if (
                        vars(staged_parent).get(
                            "_agent_manager_authority_evidence_loaded"
                        )
                        is False
                    ):
                        raise PersistedSpawnParentUnavailableError(
                            "Refusing to verify signed child authority before "
                            "its initializing parent has loaded durable evidence"
                        )
                    self._verify_agent_authority(
                        parent_name,
                        staged_parent,
                        _verification_stack=verification_stack,
                    )

            self._restore_persisted_spawn_authority(
                name,
                agent,
                agent_id,
                project=False,
                allow_initializing_parent=True,
            )
            if not isinstance(mandate, SpawnMandate) or not mandate.parent_signature:
                return
            if host_witness is None:
                if isinstance(candidate_config, LocalAgentConfig):
                    # Backfill a verified pre-registry receipt once. Future
                    # boots can then distinguish it from a root even if the
                    # child-owned edge is lost.
                    self._spawn_authority_registry.record_active(
                        child_name=name,
                        child_did=agent_id,
                        mandate=mandate,
                        config=candidate_config,
                        max_authority_slots=self._max_spawned_agents,
                    )
                    # The batch cold-witness snapshot was taken before this
                    # legacy receipt acquired its first host record.  Give a
                    # finite backfill an independent deadline immediately: a
                    # later boot-phase failure withdraws the private candidate,
                    # but must not leave its new name/cap reservation immortal.
                    if mandate.ttl_seconds > 0:
                        self._ensure_spawn_lifecycle().arm_cold_authority_ttl(
                            name,
                            mandate,
                            authority_parent_did=mandate.parent_did,
                        )
            canonical_name = self._canonical_agent_name(name)
            existing = self._preflight_spawn_reservations.get(canonical_name)
            if existing is not None:
                if existing != (agent, agent_id):
                    raise RuntimeError(
                        f"Agent {name!r} has a conflicting authority preflight reservation"
                    )
                return
            # _restore_persisted_spawn_authority already evaluated the cap while
            # excluding only this candidate's (as-yet absent) reservation. Commit
            # the slot in the same manager-lock critical section as verification.
            self._preflight_spawn_reservations[canonical_name] = (agent, agent_id)
            agent._agent_manager_authority_reserved = self
        finally:
            verification_stack.remove(identity)

    async def _await_initializing_parent_evidence(
        self,
        name: str,
        agent: KestrelAgent,
        *,
        _wait_stack: Optional[set[int]] = None,
    ) -> None:
        """Wait for every staged ancestor's durable authority read.

        ``_initializing_agents`` is published before storage opens so hosted
        lifecycle ownership is visible early.  That visibility is not evidence
        that a candidate is a governing root.  A concurrently loading signed
        child therefore waits outside manager locks until each staged ancestor
        has crossed its durable-evidence callback.  Withdrawal wakes the same
        event while leaving the marker false, which fails the child closed.
        """

        mandate = vars(agent).get("_persisted_spawn_mandate")
        if not isinstance(mandate, SpawnMandate) or not mandate.parent_signature:
            return
        wait_stack = _wait_stack if _wait_stack is not None else set()
        identity = id(agent)
        if identity in wait_stack:
            raise RuntimeError("Persisted spawn authority contains a cycle")
        wait_stack.add(identity)
        try:
            await self._await_admitted_parent_candidate(
                name,
                mandate.parent_did,
            )
            staged_parent_matches = [
                (candidate_name, candidate)
                for candidate_name, candidate in self._initializing_agents.items()
                if candidate is not agent
                and mandate.parent_did in _loaded_agent_bound_dids(candidate)
            ]
            if len(staged_parent_matches) > 1:
                raise RuntimeError(
                    "Persisted spawn mandate parent identity is ambiguous"
                )
            if not staged_parent_matches:
                return
            parent_name, staged_parent = staged_parent_matches[0]
            parent_state = vars(staged_parent)
            evidence_loaded = parent_state.get(
                "_agent_manager_authority_evidence_loaded"
            )
            evidence_event = parent_state.get(
                "_agent_manager_authority_evidence_event"
            )
            if evidence_loaded is False:
                if not isinstance(evidence_event, asyncio.Event):
                    raise PersistedSpawnParentUnavailableError(
                        "Initializing parent has no durable authority evidence boundary"
                    )
                await evidence_event.wait()
                if (
                    vars(staged_parent).get(
                        "_agent_manager_authority_evidence_loaded"
                    )
                    is not True
                ):
                    raise PersistedSpawnParentUnavailableError(
                        "Initializing parent withdrew before loading durable authority"
                    )
            await self._await_initializing_parent_evidence(
                parent_name,
                staged_parent,
                _wait_stack=wait_stack,
            )
        finally:
            wait_stack.remove(identity)

    async def _await_admitted_parent_candidate(
        self,
        name: str,
        parent_did: str,
    ) -> None:
        """Wait for an already-admitted parent to resolve and become visible.

        An admission is published before its immutable anchor read. Snapshot
        the contemporaneous operations so a child cannot reject a valid parent
        merely because that read or subsequent construction lost the scheduler
        race. Failed operations wake the same boundaries and remain absent,
        preserving the later fail-closed parent lookup.
        """

        canonical_name = self._canonical_agent_name(name)
        async with self._lock:
            visible_parent_candidates = {
                id(candidate): candidate
                for candidate_name, candidate in (
                    tuple(self._agents.items())
                    + tuple(self._initializing_agents.items())
                )
                if self._canonical_agent_name(candidate_name) != canonical_name
                and parent_did in _loaded_agent_bound_dids(candidate)
            }
            if len(visible_parent_candidates) > 1:
                raise RuntimeError(
                    "Persisted spawn mandate parent identity is ambiguous"
                )
            if visible_parent_candidates:
                return
            contemporaries = [
                candidate
                for candidate in self._agent_operations.values()
                if candidate.canonical_name != canonical_name
            ]
        unresolved = [
            candidate.agent_id_resolution_event
            for candidate in contemporaries
            if candidate.agent_id_resolution_event is not None
            and not candidate.agent_id_resolution_event.is_set()
        ]
        if unresolved:
            await asyncio.gather(*(event.wait() for event in unresolved))
        matching = [
            candidate
            for candidate in contemporaries
            if candidate.resolved_agent_id == parent_did
        ]
        if len(matching) > 1:
            raise RuntimeError(
                "Persisted spawn mandate parent identity is ambiguous"
            )
        if not matching:
            return
        staged = matching[0].candidate_staged_event
        if staged is not None and not staged.is_set():
            await staged.wait()

    async def _await_initializing_parent_publication(
        self,
        name: str,
        agent: KestrelAgent,
    ) -> None:
        """Join a verified dynamic parent's routing publication.

        Dynamic loads initialize independently.  A signed child can therefore
        finish active boot after its parent's durable evidence is verified but
        before that parent publishes.  Authority projection intentionally reads
        only routable parents, so wait outside all manager/A2A locks for the
        initializing parent to publish or withdraw before entering that commit.
        Fleet batch startup does not call this helper; it already registers its
        fully initialized candidates in deterministic parent-first order.
        """

        mandate = vars(agent).get("_persisted_spawn_mandate")
        if not isinstance(mandate, SpawnMandate) or not mandate.parent_signature:
            return
        parent_matches = [
            (candidate_name, candidate)
            for candidate_name, candidate in self._initializing_agents.items()
            if candidate is not agent
            and mandate.parent_did in _loaded_agent_bound_dids(candidate)
        ]
        if len(parent_matches) > 1:
            raise RuntimeError("Persisted spawn mandate parent identity is ambiguous")
        if not parent_matches:
            return
        parent_name, parent = parent_matches[0]
        publication_event = vars(parent).get("_agent_manager_publication_event")
        if not isinstance(publication_event, asyncio.Event):
            raise PersistedSpawnParentUnavailableError(
                "Initializing parent has no routing publication boundary"
            )
        if self._agents.get(parent_name) is not parent:
            await publication_event.wait()
        if (
            vars(parent).get("_agent_manager_published") is not True
            or self._agents.get(parent_name) is not parent
        ):
            raise PersistedSpawnParentUnavailableError(
                "Initializing parent withdrew before routing publication"
            )

    def _release_preflight_spawn_reservation(
        self,
        name: str,
        agent: KestrelAgent,
        agent_id: Optional[str],
    ) -> None:
        """Release only the exact private reservation owned by ``agent``."""

        canonical_name = self._canonical_agent_name(name)
        reservation = self._preflight_spawn_reservations.get(canonical_name)
        if reservation == (agent, agent_id):
            self._preflight_spawn_reservations.pop(canonical_name, None)
        if vars(agent).get("_agent_manager_authority_reserved") is self:
            agent._agent_manager_authority_reserved = None

    def _provisional_spawn_authority_slots(self) -> int:
        """Projected child rows still owned by live spawn reservations."""

        projected_names = {
            self._canonical_agent_name(name)
            for name in self._child_mandates
        }
        provisional = 0
        for admission in self._agent_operations.values():
            if not (
                admission.provisional_spawn_authority
                and admission.spawn_slot_active
            ):
                continue
            if admission.canonical_name not in projected_names:
                raise RuntimeError(
                    "Provisional spawn authority is missing its child projection"
                )
            provisional += 1
        return provisional

    def _spawn_cap_slots_in_use(self) -> int:
        """Count projected, pending, and cold-preflight authority once each."""

        pending_authority = self._spawn_authority_registry.pending()
        live_pending_ids = {
            admission.spawn_authority_pending_id
            for admission in self._agent_operations.values()
            if admission.spawn_slot_active
            and admission.spawn_authority_pending_id is not None
        }
        cold_pending_slots = sum(
            1
            for reservation in pending_authority
            if reservation.reservation_id not in live_pending_ids
        )
        durable_child_dids = {
            witness.child_did
            for witness in self.active_spawn_authority_witnesses()
        }
        projected_child_dids = set(durable_child_dids)
        unbound_projected_names: set[str] = set()
        projected_names: set[str] = set()
        for child_name, mandate in self._child_mandates.items():
            projected_names.add(self._canonical_agent_name(child_name))
            child_did = getattr(mandate, "child_did", None)
            if isinstance(child_did, str) and child_did:
                projected_child_dids.add(child_did)
            else:
                # Compatibility fixtures and a corrupt in-memory projection
                # still consume a slot; never let malformed authority widen
                # the fleet cap.
                unbound_projected_names.add(
                    self._canonical_agent_name(child_name)
                )
        projected_pending_slots = sum(
            1
            for admission in self._agent_operations.values()
            if admission.spawn_slot_active
            and (
                (
                    admission.provisional_spawn_authority
                    and admission.canonical_name in projected_names
                )
                or admission.spawn_authority_witness_did
                in durable_child_dids
            )
        )
        unprojected_preflight_slots = sum(
            1
            for canonical_name, (_agent, child_did) in (
                self._preflight_spawn_reservations.items()
            )
            if child_did not in projected_child_dids
            and canonical_name not in unbound_projected_names
        )
        return (
            len(projected_child_dids)
            + len(unbound_projected_names)
            + self._pending_spawns
            + cold_pending_slots
            + unprojected_preflight_slots
            - projected_pending_slots
        )

    def active_spawn_authority_witnesses(
        self,
    ) -> tuple[SpawnAuthorityWitness, ...]:
        """Return the durable authorities that still permit child restart."""

        return tuple(
            witness
            for witness in self._spawn_authority_registry.records()
            if witness.active
        )

    def unsettled_spawn_authority_witnesses(
        self,
    ) -> tuple[SpawnAuthorityWitness, ...]:
        """Return active or retiring authority still needing terminal custody."""

        return tuple(
            witness
            for witness in self._spawn_authority_registry.records()
            if not witness.retired
        )

    def _spawned_descendant_names(
        self,
        parent_dids: set[str],
    ) -> set[str]:
        """Project live and restartable direct children for parent identities."""

        descendants = {
            child_name
            for parent_did in parent_dids
            for child_name in self._parent_children.get(parent_did, ())
        }
        descendants.update(
            witness.child_name
            for witness in self.active_spawn_authority_witnesses()
            if witness.parent_did in parent_dids
        )
        return descendants

    def _retained_spawn_cleanup_slots(self) -> int:
        """Cap slots retained by quarantine rather than active spawn owners.

        ``_pending_spawns`` includes both ordinary in-flight reservations and
        failed-cleanup reservations transferred to quarantine. Cold restoration
        is allowed to arbitrate ahead of the former (the active spawn becomes
        the rollback loser at governance commit), but it must never bypass the
        latter because that child is still live and has no active admission
        left to lose.

        Callers that participate in async admission hold ``_lock``. Direct
        registration is a synchronous compatibility seam and observes the same
        event-loop-atomic snapshot.
        """

        active_slots = sum(
            1
            for admission in self._agent_operations.values()
            if admission.spawn_slot_active
        )
        retained = self._pending_spawns - active_slots
        if retained < 0:
            raise RuntimeError("Spawn cap reservation accounting underflowed")
        return retained

    @staticmethod
    async def _run_hosted_agent_ready_hooks(agent: KestrelAgent) -> None:
        """Cross the deferred ready boundary for a concrete Kestrel agent."""

        complete = getattr(type(agent), "complete_deferred_agent_readiness", None)
        bound_complete = None
        if complete is not None:
            bound_complete = lambda: complete(agent)
        else:
            # Some host-bound compatibility objects expose the seam directly
            # on the instance. Read ``vars`` so a MagicMock cannot fabricate
            # readiness capability merely because an attribute was queried.
            instance_complete = vars(agent).get(
                "complete_deferred_agent_readiness"
            )
            if callable(instance_complete):
                bound_complete = instance_complete
        if bound_complete is not None:
            gate = getattr(agent, "_host_context_publication_gate", None)
            if gate is None or gate.is_set():
                await bound_complete()
            # A closed server-owned context gate keeps the hook deferred for
            # the server's post-publication sweep. Authority is already
            # admitted, but cognition cannot precede host policy publication.
            return
        ready = getattr(type(agent), "run_agent_ready_hooks", None)
        if ready is not None:
            await ready(agent)

    @staticmethod
    def _spawn_admission_deadline(agent: KestrelAgent) -> Optional[float]:
        """Return one monotonic deadline for an ephemeral signed admission."""

        mandate = vars(agent).get("_persisted_spawn_mandate")
        if (
            not isinstance(mandate, SpawnMandate)
            or not mandate.parent_signature
            or mandate.ttl_seconds <= 0
        ):
            return None
        owned_deadline = vars(agent).get("_host_authority_deadline_monotonic")
        if isinstance(owned_deadline, (int, float)) and not isinstance(
            owned_deadline, bool
        ):
            return float(owned_deadline)
        remaining = remaining_spawn_ttl_seconds(
            mandate.created_at,
            mandate.ttl_seconds,
        )
        return asyncio.get_running_loop().time() + remaining

    @staticmethod
    def _require_spawn_admission_time(
        deadline: Optional[float],
        *,
        phase: str,
    ) -> None:
        if (
            deadline is not None
            and asyncio.get_running_loop().time() >= deadline
        ):
            raise PersistedSpawnMandateExpiredError(
                f"Persisted spawn mandate expired before {phase}"
            )

    async def _await_before_spawn_admission_deadline(
        self,
        operation: Callable[[], Awaitable[object]],
        *,
        deadline: Optional[float],
        phase: str,
        on_expiry: Optional[Callable[[], None]] = None,
        admission: Optional[AgentOperationAdmission] = None,
    ) -> object:
        """Run one admission await without resetting the signed deadline.

        ``asyncio.wait_for`` waits for a cancellation-suppressing coroutine to
        finish.  That is unsafe at an authority boundary: an already-published
        child would remain routable after its signed lifetime. Race an owned
        task against the deadline instead, fence synchronously, and then join
        cancellation-resistant work before destructive cleanup can overtake it.
        """

        if deadline is None:
            return await operation()
        try:
            self._require_spawn_admission_time(deadline, phase=phase)
        except PersistedSpawnMandateExpiredError:
            if on_expiry is not None:
                on_expiry()
            raise
        remaining = deadline - asyncio.get_running_loop().time()
        task = asyncio.create_task(
            operation(),
            name=f"spawn_admission:{phase}",
        )
        if admission is not None:
            self._admission_delegate_tasks[task] = admission

            def retire_delegate(completed: asyncio.Task) -> None:
                self._admission_delegate_tasks.pop(completed, None)

            task.add_done_callback(retire_delegate)
        try:
            done, _pending = await asyncio.wait(
                {task}, timeout=remaining, return_when=asyncio.FIRST_COMPLETED
            )
        except asyncio.CancelledError:
            task.cancel()
            await await_lifecycle_task_completion(task)
            raise
        if task not in done:
            task.cancel()
            self._retain_expired_admission_operation(task)
            if on_expiry is not None:
                on_expiry()
            # Fencing is synchronous, but cleanup must not overtake a hook that
            # retained cancellation and still holds a direct agent reference.
            # Join its terminal outcome before the caller rolls back routes,
            # storage, or runtime credentials.
            _cancelled, failure = await await_lifecycle_task_completion(task)
            if failure is not None:
                logger.warning(
                    "Expired spawn admission work ended with %s during %s",
                    type(failure).__name__,
                    phase,
                )
            raise PersistedSpawnMandateExpiredError(
                f"Persisted spawn mandate expired during {phase}"
            )
        result = task.result()
        # Completion and the timer can become runnable in the same loop turn.
        # Recheck the signed clock so a just-late result cannot reopen admission.
        try:
            self._require_spawn_admission_time(deadline, phase=phase)
        except PersistedSpawnMandateExpiredError:
            if on_expiry is not None:
                on_expiry()
            raise
        return result

    def _retain_expired_admission_operation(self, task: asyncio.Task) -> None:
        """Own a cancellation-resistant operation beyond its expired wait."""

        self._expired_admission_operations.add(task)

        def retire(completed: asyncio.Task) -> None:
            self._expired_admission_operations.discard(completed)
            try:
                completed.exception()
            except (asyncio.CancelledError, Exception):
                pass

        task.add_done_callback(retire)

    def _fence_expired_spawn_route(
        self, name: str, agent: KestrelAgent
    ) -> None:
        """Synchronously revoke one exact child's public routing authority."""

        canonical_name = self._canonical_agent_name(name)
        candidates = (
            self._agents.get(name),
            self._onboarding_agents.get(name),
            self._initializing_agents.get(name),
        )
        prepared = self._prepared_agent_names.get(_loaded_agent_did(agent) or "")
        admission = self._agent_operations.get(canonical_name)
        if (
            any(candidate is agent for candidate in candidates)
            or (prepared is not None and prepared == (name, agent))
            or (admission is not None and admission.child is agent)
        ):
            self._expired_spawn_authority_routes[canonical_name] = agent

    def fence_expired_spawn_routes(
        self,
        targets: tuple[tuple[str, str], ...],
    ) -> None:
        """Withdraw exact expired routes while their cleanup remains retryable."""

        if not isinstance(targets, tuple):
            raise TypeError("expired spawn route targets must be a tuple")
        resolved: list[tuple[str, KestrelAgent]] = []
        for target in targets:
            if not isinstance(target, tuple) or len(target) != 2:
                raise TypeError("expired spawn route target must be a name/DID tuple")
            child_name, child_did = target
            if not isinstance(child_name, str) or not child_name:
                raise TypeError("expired spawn route name must be non-empty")
            if not isinstance(child_did, str) or not child_did:
                raise TypeError("expired spawn route DID must be non-empty")
            agent = self._get_agent_for_lifecycle(child_name)
            if agent is None:
                continue
            if _loaded_agent_did(agent) != child_did:
                raise RuntimeError("Refusing to fence a different spawned child")
            resolved.append((child_name, agent))
        for child_name, agent in resolved:
            self._fence_expired_spawn_route(child_name, agent)

    def _spawn_route_is_fenced(self, name: str, agent: object) -> bool:
        return (
            self._expired_spawn_authority_routes.get(
                self._canonical_agent_name(name)
            )
            is agent
        )

    def _expire_host_authority_candidate(
        self, name: str, agent: KestrelAgent
    ) -> None:
        """Fence expiry now; clean a batch-orphaned private boot independently."""

        self._fence_expired_spawn_route(name, agent)
        if not self._spawn_route_is_fenced(name, agent):
            # Ignore a stale timer from an earlier occupant after this routing
            # name has completed release and become eligible for reuse.
            return
        admission = self._agent_operations.get(self._canonical_agent_name(name))
        if admission is not None:
            if admission.published:
                # Published readiness owns receipt-first rollback. Starting the
                # private-candidate cleanup here would race that rollback and
                # assign destructive custody twice.
                return
            admission.host_authority_expired = True
            # Active boot normally cancels its own task below. Once initialize
            # has returned, however, the candidate can remain private while a
            # batch waits for a sibling or a dynamic child waits for its parent
            # to publish. Cancel the admission owner so that exact load/spawn
            # path performs its cancellation-safe unpublished cleanup.
            owner_task = admission.owner_task
            if isinstance(owner_task, asyncio.Task) and not owner_task.done():
                owner_task.cancel()
                return
        active_boot = vars(agent).get("_host_authority_active_boot_task")
        if isinstance(active_boot, asyncio.Task) and not active_boot.done():
            # The KestrelAgent watchdog cancels this owner immediately after
            # the callback returns. Its load/spawn exception path owns receipt
            # rollback and runtime shutdown.
            return
        existing = vars(agent).get("_host_authority_expiry_cleanup_task")
        if isinstance(existing, asyncio.Task):
            return
        cleanup = asyncio.create_task(
            self._discard_unpublished_initialized_agent(name, agent),
            name=f"expired_spawn_admission_cleanup:{name}",
        )
        agent._host_authority_expiry_cleanup_task = cleanup

        def observe(completed: asyncio.Task) -> None:
            try:
                completed.exception()
            except (asyncio.CancelledError, Exception):
                pass

        cleanup.add_done_callback(observe)

    @asynccontextmanager
    async def _a2a_writer_before_spawn_admission_deadline(
        self,
        *,
        deadline: Optional[float],
        phase: str,
    ):
        """Acquire the publication writer inside the same signed lifetime."""

        acquired = False

        async def acquire() -> None:
            nonlocal acquired
            await self._a2a_lifecycle_lock.acquire()
            acquired = True

        try:
            await self._await_before_spawn_admission_deadline(
                acquire,
                deadline=deadline,
                phase=phase,
            )
            yield
        finally:
            if acquired:
                self._a2a_lifecycle_lock.release()

    async def _run_hosted_agent_ready_hooks_before_mandate_expiry(
        self,
        name: str,
        agent: KestrelAgent,
        *,
        deadline: Optional[float],
        admission: AgentOperationAdmission,
    ) -> None:
        """Keep wake-capable readiness inside a signed ephemeral lifetime.

        Authority is projected before ready hooks so they can observe the
        child's restrictions, but publication/TTL commit happens afterward.
        Bound that otherwise-unarmed interval with the receipt's exact
        remaining lifetime. ``wait_for`` delivers cancellation to the hook at
        the deadline and joins its cleanup before onboarding rolls back.
        """

        if deadline is None:
            task = asyncio.create_task(
                self._run_hosted_agent_ready_hooks(agent),
                name=f"agent_deferred_readiness:{name}",
            )
            caller_cancelled, failure = await await_lifecycle_task_completion(
                task
            )
            if isinstance(failure, asyncio.CancelledError):
                # Ready hooks are best-effort after publication. A compatible
                # agent can surface a hook-owned cancellation instead of
                # consuming it internally; that must not turn a committed
                # route into a reported load failure.
                logger.warning(
                    "Deferred best-effort readiness cancelled for published "
                    "agent %r; preserving the committed load",
                    name,
                )
                return
            if failure is not None:
                raise failure
            if caller_cancelled:
                logger.info(
                    "Agent %r finished deferred readiness after caller "
                    "cancellation; preserving the committed load result",
                    name,
                )
            return

        await self._await_before_spawn_admission_deadline(
            lambda: self._run_hosted_agent_ready_hooks(agent),
            deadline=deadline,
            phase="agent readiness",
            on_expiry=lambda: self._fence_expired_spawn_route(name, agent),
            admission=admission,
        )

    async def _rollback_published_agent_after_readiness_failure(
        self,
        name: str,
        agent: KestrelAgent,
        admission: AgentOperationAdmission,
        *,
        scheduler_lifecycle_lock_held: bool = False,
    ) -> bool:
        """Cascade one published candidate after its ready boundary fails.

        Ready hooks run only after host onboarding and routing publication
        commit.  A hook may therefore have created descendants before it
        fails or its signed deadline expires.  Tear those descendants down
        first, then withdraw the candidate, so failure cannot strand a
        routable orphan.  The cleanup task is joined cancellation-safely.
        """

        async def rollback() -> None:
            failures: list[BaseException] = []
            rollback_admission = (
                admission
                if admission.kind in {"spawn", "direct-spawn-test"}
                else None
            )
            receipt_revoked = rollback_admission is None
            if rollback_admission is not None:
                try:
                    await self._downgrade_uncommitted_spawn_receipt(
                        rollback_admission,
                        agent,
                    )
                    receipt_revoked = True
                except BaseException as error:
                    # The signed receipt lives in the child's graph. Never
                    # close that graph (or its runtime) until the exact durable
                    # authority edge has been downgraded. The outer spawn owner
                    # retains the witness and will retry through its handoff.
                    failures.append(error)
            agent_id = _loaded_agent_did(agent)
            if receipt_revoked and isinstance(agent_id, str) and agent_id:
                try:
                    await self.terminate_children(agent_id)
                except BaseException as error:
                    # A typed descendant terminal outcome can mean the child
                    # was already removed but a refund/reconciliation step
                    # failed. It must not strand the rejected parent routable.
                    failures.append(error)
            if receipt_revoked:
                try:
                    removed = await self.remove_agent(
                        name,
                        offboard_runtime=rollback_admission is not None,
                        _spawn_rollback_admission=rollback_admission,
                        _scheduler_lifecycle_lock_held=(
                            scheduler_lifecycle_lock_held
                        ),
                        _preserve_scheduler_authority=(
                            scheduler_lifecycle_lock_held
                        ),
                    )
                    if not removed and self._agents.get(name) is agent:
                        raise RuntimeError(
                            f"Readiness rollback did not withdraw agent {name!r}"
                        )
                except BaseException as error:
                    no_hosted_cancellation = (
                        _uncommitted_spawn_not_hosted_cancellation(error)
                        if rollback_admission is not None
                        else None
                    )
                    if no_hosted_cancellation is None:
                        failures.append(error)
                    elif no_hosted_cancellation:
                        failures.append(asyncio.CancelledError())
            if failures:
                _raise_lifecycle_outcomes(
                    "Readiness rollback encountered cleanup failures",
                    failures,
                )

        cleanup = asyncio.create_task(
            rollback(),
            name=f"agent_readiness_rollback:{name}",
        )
        cancelled, failure = await await_lifecycle_task_completion(cleanup)
        if failure is not None:
            raise failure
        return cancelled

    async def _finish_published_agent_readiness(
        self,
        name: str,
        agent: KestrelAgent,
        admission: AgentOperationAdmission,
        *,
        deadline: Optional[float],
        failure_description: str,
        scheduler_lifecycle_lock_held: bool = False,
    ) -> None:
        """Run post-publication readiness or cascade its published effects."""

        try:
            await self._run_hosted_agent_ready_hooks_before_mandate_expiry(
                name,
                agent,
                deadline=deadline,
                admission=admission,
            )
            # Ready hooks are wake-capable and may remove their own agent; a
            # concurrent DELETE/shutdown can do the same while the hook awaits.
            # Re-enter the topology writer and verify the exact object before
            # load/create reports success or persists a resurrecting config.
            async def verify_publication() -> None:
                async with self._a2a_lifecycle_lock:
                    async with self._lock:
                        if (
                            self.get_agent(name) is not agent
                            or not admission.published
                            or not self._operation_is_admitted(admission)
                        ):
                            raise RuntimeError(
                                f"Agent {name!r} was withdrawn during readiness"
                            )

            verification = asyncio.create_task(
                verify_publication(),
                name=f"agent_readiness_publication_check:{name}",
            )
            _verification_cancelled, verification_failure = (
                await self._await_before_spawn_admission_deadline(
                    lambda: await_lifecycle_task_completion(verification),
                    deadline=deadline,
                    phase="agent readiness publication verification",
                    on_expiry=lambda: self._fence_expired_spawn_route(
                        name, agent
                    ),
                    admission=admission,
                )
            )
            if verification_failure is not None:
                raise verification_failure
        except BaseException as readiness_failure:
            cleanup_cancelled = False
            cleanup_failure: BaseException | None = None
            try:
                cleanup_cancelled = (
                    await self._rollback_published_agent_after_readiness_failure(
                        name,
                        agent,
                        admission,
                        scheduler_lifecycle_lock_held=(
                            scheduler_lifecycle_lock_held
                        ),
                    )
                )
            except BaseException as error:
                cleanup_failure = error
            if self.get_agent(name) is not agent:
                admission.published = False
            outcomes = [readiness_failure]
            if cleanup_cancelled:
                outcomes.append(asyncio.CancelledError())
            if cleanup_failure is not None:
                outcomes.append(cleanup_failure)
            _raise_lifecycle_outcomes(failure_description, outcomes)

    async def _run_host_onboarding_before_mandate_expiry(
        self,
        name: str,
        agent: KestrelAgent,
        *,
        deadline: Optional[float],
    ) -> None:
        """Keep private host integration inside a signed child lifetime."""

        await self._await_before_spawn_admission_deadline(
            lambda: self._on_agent_registered(name, agent),
            deadline=deadline,
            phase="onboarding",
        )

    def _commit_restored_child_ttl(
        self, name: str, agent: KestrelAgent
    ) -> None:
        """Arm an adopted TTL only after every onboarding stage committed."""

        mandate = vars(agent).get("_persisted_spawn_mandate")
        if not isinstance(mandate, SpawnMandate) or not mandate.parent_signature:
            return
        admission = self._agent_operations.get(self._canonical_agent_name(name))
        if (
            admission is not None
            and admission.kind in {"spawn", "direct-spawn-test"}
            and not admission.committed
        ):
            return
        self._ensure_spawn_lifecycle().arm_restored_child_ttl(
            name,
            expected_child_did=_loaded_agent_did(agent),
        )
        disarm_host_authority_deadline(agent)

    def record_expired_spawn_retirement(
        self,
        child_name: str,
        *,
        expected_child_did: str,
        data_dir: Optional[Path] = None,
    ) -> Optional[Path]:
        """Persist a fail-closed tombstone for config-less auto-discovery.

        Spawn TTL termination intentionally retains the primary agent database.
        Without a separate terminal marker, config-less startup discovers that
        database again and retries a receipt which has already expired.  Bind
        the marker to the exact child DID and create it only after routing and
        parental authority have both been withdrawn.
        """

        if not isinstance(child_name, str) or not child_name:
            raise TypeError("retired spawn name must be a non-empty string")
        if not isinstance(expected_child_did, str) or not expected_child_did:
            raise TypeError("retired spawn DID must be a non-empty string")
        if self._get_agent_for_lifecycle(child_name) is not None:
            raise RuntimeError("Refusing to retire a still-routable spawned agent")
        if self.get_mandate(child_name) is not None:
            raise RuntimeError("Refusing to retire active spawned authority")

        # Retire the independent host authority even when the child has no
        # discoverable local database (test doubles, external storage, or
        # already-offboarded runtime). This is the load-bearing restart fence.
        self._spawn_authority_registry.retire(
            child_name=child_name,
            child_did=expected_child_did,
        )

        resolved_dir = (
            Path(data_dir)
            if data_dir is not None
            else self._base_data_dir / "agent_data" / child_name
        )
        if not resolved_dir.is_dir() or not (
            resolved_dir / "kestrel_prime.db"
        ).is_file():
            # Nothing here can be found by config-less auto-discovery. Test
            # doubles and already-deprovisioned children legitimately take
            # this path; no tombstone is needed when no discoverable database
            # remains.
            return None
        marker = resolved_dir / RETIRED_SPAWN_MARKER
        payload = f"{expected_child_did}\n"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(marker, flags, 0o600)
        except FileExistsError:
            if marker.is_file() and marker.read_text(encoding="utf-8") == payload:
                self._fsync_spawn_retirement_directory(resolved_dir)
                return marker
            # A name/data-directory can legitimately be reused after a prior
            # identity was retired. Replace its stale tombstone only after the
            # database itself proves that this expiry owner names the current
            # identity; otherwise an old lifecycle callback could retire a new
            # same-name agent. The temporary file is durable before the atomic
            # rename, and the directory fsync below makes that rename durable.
            anchored_did = read_anchor_agent_did_sync(
                str(resolved_dir),
                mode=AgentDIDLookupMode.INSPECTION,
            )
            if anchored_did != expected_child_did:
                raise RuntimeError(
                    "Retired spawn marker conflicts with the current database identity"
                ) from None
            temp_descriptor, temp_name = tempfile.mkstemp(
                prefix=f".{RETIRED_SPAWN_MARKER}.",
                dir=resolved_dir,
            )
            temp_path = Path(temp_name)
            try:
                with os.fdopen(temp_descriptor, "w", encoding="utf-8") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temp_path, marker)
            finally:
                temp_path.unlink(missing_ok=True)
            self._fsync_spawn_retirement_directory(resolved_dir)
            return marker
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        self._fsync_spawn_retirement_directory(resolved_dir)
        return marker

    def begin_terminal_spawn_retirement(
        self,
        child_name: str,
        *,
        expected_child_did: str,
    ) -> bool:
        """Persist restart denial before terminal lifecycle removal starts.

        This intent deliberately precedes routing and parent-edge withdrawal.
        A process crash after the manager removes the child but before the
        final tombstone is therefore fail-closed instead of resurrecting a
        COMPLETED, FAILED, or expired child at the next boot.
        """

        return self.begin_terminal_spawn_retirements(
            ((child_name, expected_child_did),)
        )[(child_name, expected_child_did)]

    def begin_terminal_spawn_retirements(
        self,
        targets: tuple[tuple[str, str], ...],
    ) -> dict[tuple[str, str], bool]:
        """Persist one complete terminal cascade before removing any child."""

        if not isinstance(targets, tuple):
            raise TypeError("retiring spawn targets must be a tuple")
        for child_name, expected_child_did in targets:
            if not isinstance(child_name, str) or not child_name:
                raise TypeError("retiring spawn name must be a non-empty string")
            if not isinstance(expected_child_did, str) or not expected_child_did:
                raise TypeError("retiring spawn DID must be a non-empty string")
            agent = self._get_agent_for_lifecycle(child_name)
            if agent is None:
                # A concurrent manager removal can win immediately before the
                # lifecycle finalizer acquires custody. The exact host witness
                # is still sufficient to close restart; an absent witness is a
                # legacy child with no independent authority record to advance.
                continue
            if _loaded_agent_did(agent) != expected_child_did:
                raise RuntimeError(
                    "Refusing retirement intent for a different child"
                )
            mandate = self.get_mandate(child_name)
            if (
                not isinstance(mandate, SpawnMandate)
                or mandate.child_did != expected_child_did
            ):
                raise RuntimeError(
                    "Refusing retirement intent without active authority"
                )

        admissions = self._spawn_authority_registry.admit_retirements(
            targets=targets
        )
        return {
            target: transitioned
            for target, (_exists, transitioned) in admissions.items()
        }

    def cancel_terminal_spawn_retirement(
        self,
        child_name: str,
        *,
        expected_child_did: str,
    ) -> bool:
        """Reopen restart only after the manager proves the child stayed live."""

        agent = self._get_agent_for_lifecycle(child_name)
        if agent is None or _loaded_agent_did(agent) != expected_child_did:
            raise RuntimeError("Refusing to cancel retirement for an absent child")
        mandate = self.get_mandate(child_name)
        if (
            not isinstance(mandate, SpawnMandate)
            or mandate.child_did != expected_child_did
        ):
            raise RuntimeError("Refusing to cancel retirement without live authority")
        return self._spawn_authority_registry.cancel_retirement(
            child_name=child_name,
            child_did=expected_child_did,
        )

    async def _begin_destructive_spawn_retirement(
        self,
        name: str,
        *,
        known_agent_id: Optional[str],
    ) -> Optional[SpawnRetirementAdmission]:
        """Crash-order one exact hosted witness before generic offboarding.

        ``terminate_child`` has a lifecycle owner that already advances this
        state.  Administrative DELETE reaches ``remove_agent`` directly, so
        this canonical primitive must close the same restart rail itself.
        """

        async with self._lock:
            published_name, agent = self._published_agent_binding(name)
            child_did = _loaded_agent_did(agent) if agent is not None else None
            if not isinstance(child_did, str) or not child_did:
                child_did = known_agent_id
            effective_name = published_name or name
        if not isinstance(child_did, str) or not child_did:
            return None
        witness = self._spawn_authority_registry.get(child_did)
        if witness is None:
            return None
        if witness.child_name.casefold() != effective_name.casefold():
            raise RuntimeError(
                "Refusing destructive spawn retirement for a different routing name"
            )
        if witness.retired:
            return SpawnRetirementAdmission(witness=witness, transitioned=False)
        _exists, transitioned = self._spawn_authority_registry.admit_retirement(
            child_name=witness.child_name,
            child_did=witness.child_did,
        )
        return SpawnRetirementAdmission(
            witness=witness,
            transitioned=transitioned,
        )

    async def _settle_destructive_spawn_retirement(
        self,
        admission: Optional[SpawnRetirementAdmission],
    ) -> None:
        """Cancel a refused removal or finalize one whose custody is gone."""

        if admission is None or admission.witness.retired:
            return
        witness = admission.witness
        async with self._a2a_lifecycle_lock:
            async with self._lock:
                current = self._get_agent_for_lifecycle(witness.child_name)
                current_did = (
                    _loaded_agent_did(current) if current is not None else None
                )
                mandate = self.get_mandate(witness.child_name)
                exact_live_authority = (
                    current_did == witness.child_did
                    and isinstance(mandate, SpawnMandate)
                    and mandate.child_did == witness.child_did
                )
                scheduler_authority = self.scheduler_authority_for(
                    witness.child_did
                )
                exact_scheduler_authority = (
                    scheduler_authority is not None
                    and scheduler_authority[0].casefold()
                    == witness.child_name.casefold()
                )
                if exact_live_authority or exact_scheduler_authority:
                    if admission.transitioned:
                        self._spawn_authority_registry.cancel_retirement(
                            child_name=witness.child_name,
                            child_did=witness.child_did,
                        )
                    return
                # A different same-name identity or deferred cleanup still owns
                # this slot. Keep the crash-ordered denial instead of retiring
                # across ambiguous live state.
                if current is not None:
                    return
                self._spawn_authority_registry.retire(
                    child_name=witness.child_name,
                    child_did=witness.child_did,
                )
                if mandate is not None:
                    return
                self.record_expired_spawn_retirement(
                    witness.child_name,
                    expected_child_did=witness.child_did,
                    data_dir=witness.config.resolve_data_dir(self._base_data_dir),
                )

    @staticmethod
    def _fsync_spawn_retirement_directory(directory: Path) -> None:
        """Make creation of a retirement marker durable on POSIX hosts."""

        if os.name == "nt":  # pragma: no cover - directory fsync is POSIX-only
            return
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        descriptor = os.open(directory, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _record_loaded_spawn_expiry(
        self,
        name: str,
        agent: KestrelAgent,
        config: LocalAgentConfig,
    ) -> None:
        """Retire one cleaned-up restored child after post-boot expiry."""

        agent_id = _loaded_agent_did(agent)
        if not isinstance(agent_id, str) or not agent_id:
            raise RuntimeError("Expired spawned candidate has no stable identity")
        self.record_expired_spawn_retirement(
            name,
            expected_child_did=agent_id,
            data_dir=config.resolve_data_dir(self._base_data_dir),
        )

    def _ensure_spawn_lifecycle(self):
        """Return the manager-owned lifecycle used by cold and live spawns.

        Cold authority restoration is a startup responsibility. It cannot
        wait for a later SpawnFeature tool call, because an expired child may
        otherwise remain routable forever on a host that never invokes that
        feature. Construction is synchronous and idempotent; TTL tasks arm
        when the signed receipt is adopted inside the running startup loop.
        """

        from kestrel_sovereign.spawn.lifecycle import SpawnedAgentLifecycle

        lifecycle = getattr(self, "_lifecycle", None)
        if not isinstance(lifecycle, SpawnedAgentLifecycle):
            lifecycle = SpawnedAgentLifecycle(self)
            self._lifecycle = lifecycle
        return lifecycle

    def _refuse_unrestored_delegated_budget(
        self, name: str, agent: KestrelAgent
    ) -> None:
        """Fail closed before publishing a cold child without budget custody.

        The live spawn path owns the hold from inception through governance
        commit. A cold load currently has no async provider reconciliation
        seam at this synchronous publication boundary, so accepting a positive
        persisted allocation would give the child its ordinary wallet and
        silently discard the signed spend ceiling. Refuse that load until a
        durable delegated-wallet restoration protocol is implemented.
        """

        mandate = vars(agent).get("_persisted_spawn_mandate")
        if not isinstance(mandate, SpawnMandate):
            return
        try:
            budget = Decimal(str(mandate.budget_allocation))
        except (ArithmeticError, TypeError, ValueError):
            raise RuntimeError("Persisted spawn budget is invalid") from None
        if not budget.is_finite() or budget <= 0:
            return

        admission = self._agent_operations.get(self._canonical_agent_name(name))
        live_spawn_owns_custody = (
            admission is not None
            and admission.kind in {"spawn", "direct-spawn-test"}
            and not admission.committed
        )
        if not live_spawn_owns_custody:
            raise RuntimeError(
                "Refusing to publish a cold budgeted child before delegated "
                "wallet custody is durably restored"
            )

    def _restore_all_verifiable_spawn_authority(self) -> None:
        """Reconcile signed child receipts after either load order."""

        for child_name, child in tuple(self._agents.items()):
            child_id = _loaded_agent_did(child)
            if child_id is None:
                continue
            existing = self._child_mandates.get(child_name)
            if existing is not None:
                # This projection was signature-verified when the child was
                # published. A supported non-cascading parent withdrawal may
                # leave that child running; unrelated registrations must not
                # reopen its proof against a parent that is intentionally no
                # longer loaded. Still refuse a corrupted name/DID binding.
                if existing.child_did != child_id:
                    raise RuntimeError(
                        f"Agent {child_name!r} has conflicting persisted spawn "
                        "authority"
                    )
                continue
            self._restore_persisted_spawn_authority(
                child_name,
                child,
                child_id,
            )

    def _registration_order_for_initialized_agents(self, items):
        """Return a stable parent-before-child order for one startup batch.

        Initialization remains fully concurrent. Only publication is ordered,
        using the signed receipt's parent DID as a dependency when that parent
        is another successfully initialized member of this same batch.
        Unknown parents stay in their original relative order and fail closed
        at ``_register_agent``; cycles likewise reach the verifier rather than
        being guessed into a relationship.
        """

        batch_dids = {
            bound_did
            for *_prefix, agent in items
            for bound_did in _loaded_agent_bound_dids(agent)
        }
        emitted_dids = {
            bound_did
            for agent in self._agents.values()
            for bound_did in _loaded_agent_bound_dids(agent)
        }
        pending = list(items)
        ordered = []
        while pending:
            progress = False
            deferred = []
            for item in pending:
                agent = item[-1]
                mandate = vars(agent).get("_persisted_spawn_mandate")
                parent_did = (
                    mandate.parent_did
                    if isinstance(mandate, SpawnMandate)
                    and mandate.parent_signature
                    else None
                )
                if (
                    parent_did is not None
                    and parent_did in batch_dids
                    and parent_did not in emitted_dids
                ):
                    deferred.append(item)
                    continue
                ordered.append(item)
                emitted_dids.update(_loaded_agent_bound_dids(agent))
                progress = True
            if not progress:
                ordered.extend(deferred)
                break
            pending = deferred
        return ordered

    def _withdraw_restored_spawn_authority(
        self, name: str, agent_id: Optional[str]
    ) -> None:
        """Roll back authority projected by a registration that did not commit."""

        mandate = self._child_mandates.get(name)
        if mandate is None or mandate.child_did != agent_id:
            return
        from kestrel_sovereign.spawn.lifecycle import SpawnedAgentLifecycle

        admission = self._agent_operations.get(
            self._canonical_agent_name(name)
        )
        lifecycle = getattr(self, "_lifecycle", None)
        if isinstance(lifecycle, SpawnedAgentLifecycle):
            witness = (
                self._spawn_authority_registry.get(agent_id)
                if isinstance(agent_id, str)
                else None
            )
            live_spawn_rollback = (
                admission is not None
                and admission.kind in {"spawn", "direct-spawn-test"}
                and not admission.committed
            )
            preserve_ttl_owner = (
                witness is not None
                and witness.active
                and witness.child_name.casefold() == name.casefold()
                and witness.mandate.ttl_seconds > 0
                and not live_spawn_rollback
            )
            lifecycle.withdraw_persisted_child(
                name,
                expected_child_did=agent_id,
                preserve_ttl_owner=preserve_ttl_owner,
            )
        parent_ids = [
            parent_did
            for parent_did, children in self._parent_children.items()
            if name in children
        ]
        if len(parent_ids) > 1:
            raise RuntimeError(
                f"Agent {name!r} is attached to multiple parent identities"
            )
        parent_did = parent_ids[0] if parent_ids else mandate.parent_did
        self._prune_child_relationship_and_mandate(parent_did, name)
        if (
            admission is not None
            and admission.provisional_spawn_authority
            and (
                admission.child is None
                or _loaded_agent_did(admission.child) == agent_id
            )
        ):
            admission.provisional_spawn_authority = False

    def _register_agent(
        self,
        name: str,
        agent: KestrelAgent,
        *,
        arm_restored_ttl: bool = True,
    ) -> None:
        """Publish one fully initialized agent to the co-hosted fleet."""
        agent_id = _loaded_agent_did(agent)
        if not isinstance(agent_id, str) or not agent_id:
            raise RuntimeError(
                f"Cannot publish agent {name!r} without a concrete agent DID"
            )
        # The operation admission normally proves these are absent.  Keep the
        # publication seam defensive too: callers/tests may construct manager
        # state directly, and routing must never overwrite a different live
        # name or reverse DID binding.
        existing_name = next(
            (
                routing_name
                for routing_name in self._agents
                if routing_name.casefold() == name.casefold()
            ),
            None,
        )
        if existing_name is not None and self._agents[existing_name] is not agent:
            raise ValueError(f"Agent '{name}' already exists")
        bound_name = self._agent_names.get(agent_id)
        if bound_name is not None and bound_name != name:
            raise RuntimeError(
                "Cannot publish an agent DID already routed under a different "
                f"name: {agent_id!r} -> {bound_name!r}"
            )
        prepared = self._prepared_agent_names.get(agent_id)
        if prepared is not None and prepared != (name, agent):
            raise RuntimeError(
                "Cannot publish an agent DID prepared by another registration: "
                f"{agent_id!r}"
            )
        attached_manager = vars(agent).get("_agent_manager")
        if attached_manager is not None and attached_manager is not self:
            raise RuntimeError("Cannot publish an agent attached to another manager")
        # Registration is the final routing publication seam, and not every
        # caller comes through hosted initialization (the standalone Spawn
        # feature is one intentional example). Recheck the independent host
        # witness here so no alternate registrar can turn a missing local
        # receipt into root authority.
        self._verify_agent_authority(name, agent)
        was_published = self._agents.get(name) is agent
        parent_children_before = {
            parent: list(children)
            for parent, children in self._parent_children.items()
        }
        child_mandates_before = dict(self._child_mandates)

        # Construction snapshots host context before ``initialize()``.  Host
        # feature publication can finish while a cold agent is still starting,
        # after the manager's fan-out has already walked ``_agents``.  The turn
        # barrier handles cognition during that window; rebind again at this
        # final admission seam so the agent cannot become routable with a stale
        # snapshot.  Validate before adding either routing entry: a namespace
        # collision must reject onboarding without making the agent briefly
        # addressable.
        registry = self._host_context_clause_registry
        validate_registry = getattr(
            agent, "validate_host_context_clause_registry", None
        )
        bind_registry = getattr(agent, "bind_host_context_clause_registry", None)
        if registry is not None and (
            not callable(validate_registry) or not callable(bind_registry)
        ):
            raise RuntimeError(
                f"Cannot publish agent {name!r} without host context registry support"
            )
        if callable(validate_registry) and callable(bind_registry):
            validate_registry(registry)
            bind_registry(registry)

        # Refresh every shared publication pointer before the agent becomes
        # routable.  This also records that the direct rebind above consumed the
        # current generation, avoiding redundant work on its first normal turn.
        agent._host_context_publication_gate = self._host_context_publication_gate
        agent._host_context_publication_state = self._host_context_publication_state
        agent._host_context_publication_generation = (
            self._host_context_publication_state.generation
        )
        self._agents[name] = agent
        self._agent_names[agent_id] = name
        try:
            self._restore_persisted_spawn_authority(name, agent, agent_id)
            self._restore_all_verifiable_spawn_authority()
        except BaseException:
            from kestrel_sovereign.spawn.lifecycle import SpawnedAgentLifecycle

            lifecycle = getattr(self, "_lifecycle", None)
            if isinstance(lifecycle, SpawnedAgentLifecycle):
                for restored_name in set(self._child_mandates).difference(
                    child_mandates_before
                ):
                    lifecycle.withdraw_persisted_child(restored_name)
            self._parent_children.clear()
            self._parent_children.update(parent_children_before)
            self._child_mandates.clear()
            self._child_mandates.update(child_mandates_before)
            if not was_published:
                self._agents.pop(name, None)
                if self._agent_names.get(agent_id) == name:
                    self._agent_names.pop(agent_id, None)
            raise
        # SpawnFeature is agent-owned and has no request/app-state context.
        # Give every hosted agent the same manager that published it so its
        # control tools cannot silently manufacture a lineage-empty lightweight
        # manager after restart.
        agent._agent_manager = self
        if self._prepared_agent_names.get(agent_id) == (name, agent):
            self._prepared_agent_names.pop(agent_id, None)
        # Publish the registered routing key as the human display name so the
        # observability emitter (and any other consumer) attributes events to
        # the real agent name instead of falling back to "unknown" (#2461).
        # This is the authoritative override: ``KestrelAgent.__init__`` and
        # ``initialize()`` already stamp progressively better floors (#2602), and
        # ``_set_display_name`` also mirrors the name onto the owning LLMService
        # so LLM-call spans carry ``kestrel.agent_name`` (#2573). Guarded via
        # ``getattr`` because a few unit-test doubles stand in for a real agent.
        setter = getattr(agent, "_set_display_name", None)
        if callable(setter):
            setter(name)
        else:
            agent.agent_name = name
        if arm_restored_ttl:
            try:
                self._commit_restored_child_ttl(name, agent)
            except BaseException:
                self._withdraw_initialized_agent(name, agent)
                raise
        # Fleet-idleness (#F235): give EVERY agent — including ones created or
        # spawned after startup — a live view of all co-hosted agents, so
        # RestartCoordinator can gate a whole-host restart on the whole fleet
        # being idle. Installed here at the single registration point so a
        # dynamically-added agent can never bypass the gate. Resolves live, so
        # each agent sees agents registered after it.
        agent._cohosted_agents_provider = lambda: list(self._agents.values())
        if self._initializing_agents.get(name) is agent:
            self._initializing_agents.pop(name, None)
        publication_event = vars(agent).get("_agent_manager_publication_event")
        if isinstance(publication_event, asyncio.Event):
            agent._agent_manager_published = True
            publication_event.set()
        logger.info(f"Loaded agent '{name}' (DID: {agent.agent_id[:30]}...)")

    @staticmethod
    def _canonical_agent_name(name: str) -> str:
        """Return the sole identity used for case-insensitive name admission."""

        return name.casefold()

    def _published_agent_binding(
        self,
        name: str,
    ) -> tuple[Optional[str], Optional[KestrelAgent]]:
        """Resolve one case-insensitive routing name to its exact publication.

        Name admission forbids canonical duplicates, but removal is a security
        boundary and must not silently choose if internal state is corrupt.
        Returning the exact dictionary key ensures shutdown, reverse-map
        withdrawal, scheduler revocation, and runtime deletion all describe
        the same live agent even when an API/config spelling differs by case.
        """

        exact = self._agents.get(name)
        if exact is not None:
            return name, exact
        canonical = self._canonical_agent_name(name)
        matches = [
            (routing_name, agent)
            for routing_name, agent in self._agents.items()
            if self._canonical_agent_name(routing_name) == canonical
        ]
        if len(matches) > 1:
            raise RuntimeError(
                "Case-insensitive agent routing state is ambiguous; removal refused"
            )
        return matches[0] if matches else (None, None)

    def _scheduler_authority_binding_by_name(
        self,
        name: str,
    ) -> tuple[Optional[str], Optional[str]]:
        """Resolve one canonical routing name to exact scheduler authority."""

        canonical = self._canonical_agent_name(name)
        matches = [
            (routing_name, agent_id)
            for agent_id, (routing_name, _config) in self._scheduler_authority_by_did.items()
            if self._canonical_agent_name(routing_name) == canonical
        ]
        if len(matches) > 1:
            raise RuntimeError(
                "Case-insensitive scheduler authority is ambiguous; removal refused"
            )
        return matches[0] if matches else (None, None)

    def _quarantined_cleanup_name_is_reserved(self, canonical_name: str) -> bool:
        """Whether active or unsafe quarantined cleanup owns this name.

        A completed *successful* reaper is only operational history.  A
        failed/cancelled reaper is different: its storage and, for delegated
        children, its hold can require a later explicit remediation.  Keep the
        exact case-insensitive routing identity unavailable until that unsafe
        outcome is acknowledged.  If bounded history has evicted any unsafe
        record, the aggregate overflow reservation blocks every name until its
        explicit acknowledgement rather than retaining an unbounded name set.
        """

        retained_name_key = _bounded_shutdown_metadata(canonical_name)
        return (
            any(
                record.canonical_agent_name == retained_name_key
                for record in self._quarantined_shutdown_reapers.values()
            )
            or any(
                record.canonical_agent_name == retained_name_key
                for record in self._unsafe_quarantined_shutdown_failures.values()
            )
            or self._unsafe_quarantined_shutdown_failure_overflow_reserved
        )

    def _retained_child_tracking_name_is_reserved(
        self, canonical_name: str
    ) -> bool:
        """Whether an unpruned mandate or parent edge still owns this name."""

        return any(
            self._canonical_agent_name(child_name) == canonical_name
            for child_name in self._child_mandates
        ) or any(
            self._canonical_agent_name(child_name) == canonical_name
            for children in self._parent_children.values()
            for child_name in children
        )

    async def _admit_agent_operation(
        self, name: str, *, kind: str
    ) -> tuple[AgentOperationAdmission, bool]:
        """Reserve ``name`` through one operation's publish/rollback boundary.

        The returned boolean says whether this caller created the reservation.
        Nested create -> load calls made by the same task borrow their parent's
        admission; a second task is rejected before it can initialize or write
        inception data.  Lock order matches removal (A2A lifecycle then state)
        so the admission and shutdown epoch form one linearizable boundary.
        """

        canonical_name = self._canonical_agent_name(name)
        owner_task = asyncio.current_task()
        async with self._a2a_lifecycle_lock:
            if self._agent_registration_sealed:
                raise RuntimeError(
                    "Refusing agent registration because the manager is shutting down"
                )
            async with self._lock:
                # Re-check after the state-lock await because shutdown seals
                # without awaiting the A2A writer (to avoid drain deadlocks).
                if self._agent_registration_sealed:
                    raise RuntimeError(
                        "Refusing agent registration because the manager is shutting down"
                    )
                admitted = self._agent_operations.get(canonical_name)
                if admitted is not None:
                    if admitted.owner_task is owner_task:
                        if (
                            admitted.registration_epoch
                            != self._agent_registration_shutdown_epoch
                        ):
                            # Reopening the manager admits only work that
                            # begins after the completed shutdown.  A nested
                            # create -> load from an older operation must not
                            # treat that reopen as permission to initialize a
                            # new runtime around persisted identity data.
                            raise RuntimeError(
                                "Refusing agent registration because this operation "
                                "began before manager shutdown"
                            )
                        return admitted, False
                    raise ValueError(
                        f"Agent '{name}' is already being initialized or created"
                    )
                existing_name = next(
                    (
                        routing_name
                        for routing_name in self._agents
                        if routing_name.casefold() == canonical_name
                    ),
                    None,
                )
                if existing_name is not None:
                    raise ValueError(f"Agent '{name}' already exists")
                # A failed quarantined refund restores this exact allocation so
                # an operator can retry safely.  Reusing its name for a new
                # identity before that cleanup resolves would make restoration
                # ambiguous and could attach an old hold to a new child.
                unresolved_hold_name = next(
                    (
                        child_name
                        for child_name in self._child_budgets
                        if self._canonical_agent_name(child_name) == canonical_name
                    ),
                    None,
                )
                if unresolved_hold_name is not None:
                    raise RuntimeError(
                        f"Agent '{name}' has an unresolved delegated budget cleanup"
                    )
                # A bounded DELETE may already have withdrawn this name while
                # a quarantined reaper owns its storage or a fenced delegated
                # refund.  The hold is temporarily absent while that reaper
                # runs, so ``_child_budgets`` alone has a gap in which a new
                # same-name create could race a failed refund restoration.
                # Keep the routing name unavailable until the exact reaper
                # settles; on failure the restored hold then continues the
                # reservation until an explicit safe retry succeeds.
                if self._quarantined_cleanup_name_is_reserved(canonical_name):
                    raise RuntimeError(
                        f"Agent '{name}' has unresolved quarantined cleanup"
                    )
                # Successful quarantine retires its task record before the
                # terminal drain reconciles the deliberately retained parent
                # edge and mandate.  Keep that tracking authoritative during
                # the callback-to-reconciliation gap so a new same-name child
                # cannot inherit the old relationship.
                if self._retained_child_tracking_name_is_reserved(canonical_name):
                    raise RuntimeError(
                        f"Agent '{name}' has retained child lifecycle tracking"
                    )
                pending_spawn = self._spawn_authority_registry.pending_for_slot(
                    child_name=name,
                    config=None,
                )
                if pending_spawn is not None:
                    raise RuntimeError(
                        f"Agent '{name}' has unresolved pending spawn authority"
                    )
                admission = AgentOperationAdmission(
                    name=name,
                    canonical_name=canonical_name,
                    kind=kind,
                    registration_epoch=self._agent_registration_shutdown_epoch,
                    owner_task=owner_task,
                    agent_id_resolution_event=asyncio.Event(),
                    candidate_staged_event=asyncio.Event(),
                )
                self._agent_operations[canonical_name] = admission
                return admission, True

    async def _release_agent_operation(
        self, admission: AgentOperationAdmission
    ) -> None:
        """Release an operation's name only after its owner reached a terminal state.

        This is itself a terminal ownership handoff.  A cancelled caller can
        otherwise be interrupted while waiting for the state lock and strand a
        completed admission forever.  Keep an owned
        release task and join it through the common lifecycle contract.
        """

        async def release() -> None:
            # Operation admission is state-only after its owner reaches a
            # terminal result.  Do not reacquire the A2A writer here: callers
            # may already be waiting for an admission that is blocked behind a
            # writer they deliberately hold (for example, cancellation while
            # publication is queued).  Taking it again would turn that normal
            # ownership handoff into a re-entrancy deadlock.
            async with self._lock:
                if (
                    self._agent_operations.get(admission.canonical_name)
                    is admission
                ):
                    self._agent_operations.pop(admission.canonical_name, None)
                if admission.agent_id_resolution_event is not None:
                    admission.agent_id_resolution_event.set()
                if admission.candidate_staged_event is not None:
                    admission.candidate_staged_event.set()

        release_task = asyncio.create_task(
            release(), name=f"agent_operation_release:{admission.canonical_name}"
        )
        cancelled, failure = await await_lifecycle_task_completion(release_task)
        if failure is not None:
            raise failure
        if cancelled:
            raise asyncio.CancelledError()

    async def _settle_owned_lifecycle_tasks(
        self,
        tasks: list["asyncio.Future[object]"],
        *,
        description: str,
    ) -> bool:
        """Join every owned task before reporting cancellation or failures.

        A caller's cancellation must not make a later owner invisible merely
        because an earlier join observed it first.  Each task is independently
        cancellation-safe, and this collector preserves that same terminal
        boundary for a group of related lifecycle owners.  Cancellation wins
        only after all work has settled; otherwise every terminal failure is
        retained in one group rather than dropping failures after the first.
        """

        cancelled = False
        failures: list[BaseException] = []
        for task in tasks:
            task_cancelled, failure = await await_lifecycle_task_completion(task)
            cancelled = cancelled or task_cancelled
            if failure is None:
                continue
            if isinstance(failure, asyncio.CancelledError):
                cancelled = True
            else:
                failures.append(failure)
        if cancelled:
            return True
        if failures:
            raise BaseExceptionGroup(description, failures)
        return False

    async def _release_agent_operations(
        self, admissions: list[AgentOperationAdmission]
    ) -> bool:
        """Retire every admission before returning any observed cancellation."""

        releases = [
            asyncio.create_task(
                self._release_agent_operation(admission),
                name=f"agent_operation_final_release:{admission.canonical_name}",
            )
            for admission in admissions
        ]
        return await self._settle_owned_lifecycle_tasks(
            releases,
            description="One or more agent operation admissions failed to release",
        )

    async def _retire_spawn_slot_and_admission(
        self,
        admission: AgentOperationAdmission,
        *,
        spawn_slot_admitted: bool,
    ) -> bool:
        """Retire a spawn's cap slot and name admission as one terminal tail."""

        async def retire_spawn_slot() -> None:
            if not spawn_slot_admitted or not admission.spawn_slot_active:
                return
            async with self._lock:
                if admission.spawn_slot_active:
                    self._pending_spawns -= 1
                    admission.spawn_slot_active = False
                    self._resolve_spawn_slot_terminal(admission)

        tasks: list["asyncio.Future[object]"] = [
            asyncio.create_task(
                self._release_agent_operation(admission),
                name=f"agent_operation_final_release:{admission.canonical_name}",
            )
        ]
        if spawn_slot_admitted:
            tasks.append(
                asyncio.create_task(
                    retire_spawn_slot(),
                    name=f"spawn_slot_retirement:{admission.canonical_name}",
                )
            )
        return await self._settle_owned_lifecycle_tasks(
            tasks,
            description="One or more spawn lifecycle owners failed to retire",
        )

    def _resolve_spawn_slot_terminal(
        self,
        admission: AgentOperationAdmission,
    ) -> None:
        """Wake pending governance commits after one cap slot is terminal."""

        terminal = admission.spawn_slot_terminal
        if terminal is not None and not terminal.done():
            terminal.set_result(None)
        if terminal is not None:
            self._rejected_spawn_slot_waiters.discard(terminal)

    def _operation_is_admitted(self, admission: AgentOperationAdmission) -> bool:
        """Whether a named operation may still publish or commit state.

        The caller must hold ``_a2a_lifecycle_lock``.  Checking at publication
        rather than only before initialization closes the interval in which a
        slow load could otherwise publish just after ``shutdown_all`` finishes.
        """

        return (
            self._agent_operations.get(admission.canonical_name) is admission
            and not self._agent_registration_sealed
            and admission.registration_epoch == self._agent_registration_shutdown_epoch
        )

    def _seal_agent_registration_for_shutdown_all(self) -> None:
        """Fence publications that overlap the current fleet shutdown."""

        # Publishing is serialized by ``_a2a_lifecycle_lock``, but a normal
        # removal deliberately holds that writer while joining an ordinary
        # refund.  Waiting for it here would deadlock the terminal drain that
        # must join the same refund.  This state transition itself has no
        # await point, so it is atomic on the manager's event loop; each
        # publisher still verifies it while holding the A2A lifecycle writer.
        if not self._agent_registration_sealed:
            self._agent_registration_sealed = True
            self._agent_registration_shutdown_epoch += 1

    def _reopen_agent_registration_after_shutdown_all(self) -> None:
        """Allow registrations that begin only after the fleet sweep returns."""

        # This runs without an await point immediately before the caller
        # releases the terminal-drain lock and returns.  A task that began
        # before the sweep still has the previous epoch and remains fenced;
        # one that begins afterward observes the new epoch normally.
        self._agent_registration_sealed = False

    async def load_agent(
        self,
        name: str,
        config: LocalAgentConfig,
        *,
        expected_agent_id: Optional[str] = None,
        scheduler_lifecycle_lock_held: bool = False,
    ) -> KestrelAgent:
        """Create, initialize, and register one agent.

        This is the dynamic/single-agent loading path. Fleet startup uses the
        same initializer and registrar but batches the independent initialization
        awaits in :meth:`load_from_config`.  ``expected_agent_id`` is the
        host-scheduler authorization boundary for a cold wake: verify it before
        publishing the agent to routing so a stale/misconfigured local identity
        cannot execute a schedule claimed for another tenant.
        """
        if expected_agent_id is not None and not scheduler_lifecycle_lock_held:
            # Direct callers receive the same cold-initialization serialization
            # as the hosted scheduler. The executor passes ``True`` only while
            # it owns this exact manager-managed lifecycle writer.
            async with self.scheduler_lifecycle_lock(expected_agent_id):
                return await self.load_agent(
                    name,
                    config,
                    expected_agent_id=expected_agent_id,
                    scheduler_lifecycle_lock_held=True,
                )

        if expected_agent_id is not None:
            authority = self.scheduler_authority_for(expected_agent_id)
            if authority is None:
                raise LookupError(
                    "Refusing hosted scheduler cold wake for an agent that is "
                    "no longer authorized by the live host desired state"
                )
            authorized_name, authorized_config = authority
            if authorized_name != name or authorized_config != config:
                raise RuntimeError(
                    "Refusing hosted scheduler cold wake: the live host "
                    "authority does not match the claimed configuration"
                )
            existing = self._agents.get(name)
            if existing is not None:
                existing_did = _loaded_agent_did(existing)
                if existing_did != expected_agent_id:
                    raise RuntimeError(
                        "Refusing hosted scheduler cold wake: configured agent "
                        f"{name!r} is already loaded as {existing_did!r}, not "
                        f"the claimed DID {expected_agent_id!r}"
                    )
                return existing
            await self._load_scheduler_cold_authority_ancestors(
                expected_agent_id,
                name,
            )
        admission, owns_admission = await self._admit_agent_operation(
            name, kind="load"
        )
        admission.spawn_candidate_config = config.model_copy(deep=True)
        try:
            agent = await self._initialize_agent(
                name,
                config,
                scheduler_lifecycle_lock_held=scheduler_lifecycle_lock_held,
            )
            actual_agent_id = _loaded_agent_did(agent)
            if (
                expected_agent_id is not None
                and actual_agent_id != expected_agent_id
            ):
                cleanup_cancelled = (
                    await self._discard_unpublished_initialized_agents(
                        [(name, agent)]
                    )
                )
                if cleanup_cancelled:
                    raise asyncio.CancelledError()
                raise RuntimeError(
                    "Refusing hosted scheduler cold wake: initialized agent DID "
                    f"{actual_agent_id!r} does not match claimed DID "
                    f"{expected_agent_id!r}"
                )
            admission_deadline = self._spawn_admission_deadline(agent)
            committed = False
            withdrawn_after_onboarding_failure = False
            try:
                # Parent publication is an admission wait, not post-initialize
                # dead space. If the staged parent withdraws, this candidate is
                # still owned by the cleanup path below.
                await self._await_initializing_parent_publication(name, agent)
                if admission.before_publish is not None:
                    await self._await_before_spawn_admission_deadline(
                        lambda: admission.before_publish(agent),
                        deadline=admission_deadline,
                        phase="prepublication authority",
                    )
                    # A live spawn signs and restamps its mandate inside the
                    # prepublication callback. The unsigned proposal above has
                    # no enforceable deadline; use the final signed timestamp
                    # for every wake-capable await that follows it.
                    admission_deadline = self._spawn_admission_deadline(agent)
                async with self._a2a_writer_before_spawn_admission_deadline(
                    deadline=admission_deadline,
                    phase="authority preparation",
                ):
                    if not self._operation_is_admitted(admission):
                        raise RuntimeError(
                            "Refusing agent readiness because the manager is shutting down"
                        )
                    async with self._lock:
                        self._prepare_agent_authority(name, agent)
                async with self._a2a_writer_before_spawn_admission_deadline(
                    deadline=admission_deadline,
                    phase="registration publication",
                ):
                    if not self._operation_is_admitted(admission):
                        raise RuntimeError(
                            "Refusing agent registration because the manager is shutting down"
                        )
                    try:
                        # Host integration remains private and is bounded by
                        # the signed lifetime.  Arm that exact remaining TTL
                        # before the no-await routing publication commit.
                        self._onboarding_agents[name] = agent
                        await self._run_host_onboarding_before_mandate_expiry(
                            name,
                            agent,
                            deadline=admission_deadline,
                        )
                        self._require_spawn_admission_time(
                            admission_deadline,
                            phase="routing publication",
                        )
                        self._commit_restored_child_ttl(name, agent)
                        self._register_agent(
                            name,
                            agent,
                            arm_restored_ttl=False,
                        )
                        self._onboarding_agents.pop(name, None)
                        self._commit_dynamic_scheduler_registration(agent)
                    except BaseException as registration_failure:
                        # The registration hook can install hosted A2A policy
                        # and app routes before it reports failure.  Withdraw
                        # that partial publication *while this same writer is
                        # still held*, then perform the potentially slow agent
                        # shutdown below.  Releasing the writer first lets an
                        # inbound reader observe a rejected recipient or lets a
                        # DELETE claim a second shutdown owner.
                        rollback_cancelled, rollback_failure = (
                            await self._rollback_host_onboarding(name, agent)
                        )
                        self._withdraw_initialized_agent(name, agent)
                        withdrawn_after_onboarding_failure = True
                        if rollback_failure is not None:
                            _raise_lifecycle_outcomes(
                                "Agent onboarding and serialized rollback failed",
                                [registration_failure, rollback_failure],
                            )
                        if rollback_cancelled:
                            raise asyncio.CancelledError()
                        raise
                    committed = True
                    admission.published = True
                await self._finish_published_agent_readiness(
                    name,
                    agent,
                    admission,
                    deadline=admission_deadline,
                    failure_description=(
                        "Agent readiness and its published rollback failed"
                    ),
                    scheduler_lifecycle_lock_held=(
                        scheduler_lifecycle_lock_held
                    ),
                )
                self._commit_host_onboarding(agent)
                return agent
            except BaseException as onboarding_failure:
                if not committed:
                    rollback_cancelled = False
                    rollback_failure: BaseException | None = None
                    if admission.before_publish_rollback is not None:
                        rollback_task = asyncio.create_task(
                            admission.before_publish_rollback(agent),
                            name=f"agent_before_publish_rollback:{name}",
                        )
                        rollback_cancelled, rollback_failure = (
                            await await_lifecycle_task_completion(rollback_task)
                        )
                    if rollback_failure is None:
                        is_uncommitted_spawn_candidate = (
                            admission.kind in {"spawn", "direct-spawn-test"}
                            and admission.child is agent
                            and not admission.committed
                        )
                        if is_uncommitted_spawn_candidate:
                            # The outer spawn owns destructive runtime cleanup.
                            # Retain this exact private candidate so its hosted
                            # namespace and credentials can still be offboarded.
                            admission.unpublished_cleanup_deferred_to_spawn = True
                            if rollback_cancelled and not _contains_lifecycle_cancellation(
                                onboarding_failure
                            ):
                                _raise_lifecycle_outcomes(
                                    "Agent onboarding failed while prepublication "
                                    "rollback observed cancellation",
                                    [
                                        onboarding_failure,
                                        asyncio.CancelledError(),
                                    ],
                                )
                        else:
                            cleanup_cancelled = (
                                await self._discard_unpublished_initialized_agents(
                                    [(name, agent)],
                                    already_withdrawn=(
                                        withdrawn_after_onboarding_failure
                                    ),
                                )
                            )
                            if cleanup_cancelled or rollback_cancelled:
                                raise asyncio.CancelledError()
                    else:
                        # The nested load must not close the graph underneath a
                        # signed receipt whose write outcome is ambiguous. The
                        # outer spawn still owns ``admission.child`` and will
                        # retry revocation before shutting this private child
                        # down (or hand it to observable quarantine).
                        admission.unpublished_cleanup_deferred_to_spawn = True
                        _raise_lifecycle_outcomes(
                            "Agent onboarding and prepublication rollback failed",
                            [onboarding_failure, rollback_failure],
                        )
                if (
                    admission.kind == "load"
                    and isinstance(
                        onboarding_failure,
                        PersistedSpawnMandateExpiredError,
                    )
                ):
                    self._record_loaded_spawn_expiry(name, agent, config)
                raise
        finally:
            if owns_admission:
                release_cancelled = await self._release_agent_operations([admission])
                # Once the agent is fully published, returning cancellation
                # would report failure while leaving a live agent and a config
                # handoff that the caller may never persist. The admission is
                # now terminal, so retain the successful return instead.
                if release_cancelled and not admission.published:
                    raise asyncio.CancelledError()

    def _reconcile_spawn_authority_restart_roster(
        self,
        config: MultiAgentConfig,
        *,
        authority_roots: frozenset[str] | None = None,
    ) -> MultiAgentConfig:
        """Reconcile desired startup state with durable host authority.

        The authority registry is committed before ``multi_agent.toml``. A
        crash between those writes must therefore make the registry repair the
        in-memory and durable restart roster, never omit the signed child.
        Conversely, terminal intent is committed before routing/config cleanup;
        a crash in that window must remove the exact retired identity from the
        roster instead of attempting and refusing it on every later boot.
        """

        # The registry and its spawn-cap accounting are host-global even when
        # this ProcessManager-launched root restores only its own authority
        # closure. Reap every positively ownerless, no-birth reservation before
        # applying that read scope; per-reservation producer locks preserve live
        # sibling attempts.
        pending = self._spawn_authority_registry.pending()
        self._spawn_authority_registry.reap_orphaned_pending_without_birth(
            reservation_ids=tuple(
                reservation.reservation_id for reservation in pending
            )
        )
        # Re-read after the locked liveness reconciliation.  A live producer's
        # reservation remains selected and continues to suppress startup; a
        # crashed pre-birth producer no longer consumes its name or cap slot.
        pending = self._selected_pending_spawn_authority(authority_roots)
        all_records = self._spawn_authority_registry.records()

        # A cold ``autostart=false`` child never reaches load-time TTL
        # verification, so its host witness must advance here. The cap and
        # registry are host-global, so terminal reconciliation precedes the
        # standalone root's read scope; otherwise an offline sibling's expired
        # witness can permanently consume the shared cap. Retire the complete
        # durable descendant tree: authority cannot outlive the parent edge that
        # granted it. Existing crash-ordered ``retiring`` records take the same
        # recovery path, but a live in-process child prevents this startup helper
        # from stealing an active finalizer.
        terminal_roots = [
            witness
            for witness in all_records
            if witness.state == "retiring"
            or (
                witness.active
                and witness.mandate.ttl_seconds > 0
                and remaining_spawn_ttl_seconds(
                    witness.mandate.created_at,
                    witness.mandate.ttl_seconds,
                )
                <= 0
            )
        ]
        records_by_parent: dict[str, list[SpawnAuthorityWitness]] = {}
        for witness in all_records:
            if witness.state != "retired":
                records_by_parent.setdefault(witness.parent_did, []).append(witness)
        terminal_dids: set[str] = set()
        for root in terminal_roots:
            tree: list[SpawnAuthorityWitness] = []
            frontier = [root]
            seen: set[str] = set()
            while frontier:
                current = frontier.pop()
                if current.child_did in seen:
                    raise RuntimeError("Host spawn authority contains a cycle")
                seen.add(current.child_did)
                tree.append(current)
                frontier.extend(records_by_parent.get(current.child_did, ()))
            if any(
                self._get_agent_for_lifecycle(witness.child_name) is not None
                for witness in tree
            ):
                continue
            terminal_dids.update(seen)
        terminal_records = tuple(
            witness
            for witness in all_records
            if witness.child_did in terminal_dids and not witness.retired
        )
        if terminal_records:
            self._spawn_authority_registry.admit_retirements(
                targets=tuple(
                    (witness.child_name, witness.child_did)
                    for witness in terminal_records
                )
            )
            by_did = {witness.child_did: witness for witness in terminal_records}
            depth: dict[str, int] = {}

            def authority_depth(witness: SpawnAuthorityWitness) -> int:
                if witness.child_did in depth:
                    return depth[witness.child_did]
                parent = by_did.get(witness.parent_did)
                value = 0 if parent is None else authority_depth(parent) + 1
                depth[witness.child_did] = value
                return value

            for witness in sorted(
                terminal_records,
                key=authority_depth,
                reverse=True,
            ):
                data_dir = witness.config.resolve_data_dir(self._base_data_dir)
                identity_db = data_dir / "kestrel_prime.db"
                identity_present = False
                try:
                    identity_db.stat()
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    logger.warning(
                        "Could not inspect expired spawn witness %r; leaving its "
                        "retirement unresolved for this startup: %s",
                        witness.child_name,
                        exc,
                    )
                    continue
                else:
                    identity_present = True
                if identity_present:
                    try:
                        anchored_did = read_anchor_agent_did_sync(
                            str(data_dir),
                            mode=AgentDIDLookupMode.INSPECTION,
                        )
                    except Exception as exc:
                        # This tenant remains restart-denied by its durable
                        # ``retiring`` witness, but must not withhold unrelated
                        # healthy agents from fleet startup.
                        logger.warning(
                            "Could not inspect expired spawn witness %r; leaving "
                            "its retirement unresolved for this startup: %s",
                            witness.child_name,
                            exc,
                        )
                        continue
                    if anchored_did != witness.child_did:
                        # The old witness still expires, but an operator may
                        # have placed a legitimate replacement identity in its
                        # former slot.  Finalize only the old registry record;
                        # never stamp that replacement's database with the old
                        # child's retirement marker.
                        self._spawn_authority_registry.retire(
                            child_name=witness.child_name,
                            child_did=witness.child_did,
                        )
                        continue
                self.record_expired_spawn_retirement(
                    witness.child_name,
                    expected_child_did=witness.child_did,
                    data_dir=data_dir,
                )
        # Runtime restoration remains scoped even though terminal cleanup is a
        # host-global prerequisite. Re-read after retirement so this root never
        # adopts a sibling's child and never restores a record just finalized.
        records = self._selected_spawn_authority_records(authority_roots)

        runtime_host = None
        if self._startup_config_path is not None:
            # Backend/bootstrap awaits can leave the caller's startup snapshot
            # older than an operator edit.  Durable agent policy is file-owned,
            # so apply the authority repair exactly once to a fresh snapshot.
            # The caller's host projection can contain process-local listen
            # overrides and is restored only on the returned runtime model.
            runtime_host = config.host.model_copy(deep=True)
            reconciled = MultiAgentConfig.from_file(
                self._startup_config_path
            ).model_copy(deep=True)
        else:
            reconciled = config.model_copy(deep=True)
        changed = False
        for reservation in pending:
            matching = [
                name
                for name, candidate in reconciled.agents.items()
                if name.casefold() == reservation.child_name.casefold()
                or (
                    isinstance(candidate, LocalAgentConfig)
                    and self._spawn_authority_registry.same_data_slot(
                        candidate,
                        reservation.config,
                    )
                )
            ]
            if len(matching) > 1:
                raise RuntimeError(
                    "Pending spawn authority has an ambiguous startup registration"
                )
            if matching:
                del reconciled.agents[matching[0]]
                changed = True
        for witness in records:
            matching = [
                name
                for name, candidate in reconciled.agents.items()
                if name.casefold() == witness.child_name.casefold()
                or (
                    isinstance(candidate, LocalAgentConfig)
                    and self._spawn_authority_registry.same_data_slot(
                        candidate,
                        witness.config,
                    )
                )
            ]
            if len(matching) > 1:
                raise RuntimeError(
                    "Host spawn witness has an ambiguous startup registration"
                )
            if not witness.active:
                if not matching:
                    continue
                existing_name = matching[0]
                existing_config = reconciled.agents[existing_name]
                # A finalized DID must not suppress a legitimate replacement
                # that an operator has incepted under the old name or data
                # directory. Bind destructive reconciliation to the immutable
                # local birth record, just as runtime offboarding does.
                if not isinstance(existing_config, LocalAgentConfig):
                    continue
                resolved_dir = existing_config.resolve_data_dir(self._base_data_dir)
                try:
                    anchored_did = read_anchor_agent_did_sync(
                        str(resolved_dir),
                        mode=AgentDIDLookupMode.INSPECTION,
                    )
                except Exception as exc:
                    # The row cannot be proven to belong to this terminal DID,
                    # so fail closed by leaving it untouched. Normal batch
                    # initialization will isolate a missing/unreadable identity
                    # to this tenant instead of making roster reconciliation a
                    # fleet-wide startup gate.
                    logger.warning(
                        "Could not verify terminal spawn witness %r against its "
                        "startup registration; preserving the row: %s",
                        witness.child_name,
                        exc,
                    )
                    continue
                if anchored_did != witness.child_did:
                    continue
                del reconciled.agents[existing_name]
                changed = True
                continue
            if matching:
                existing_name = matching[0]
                existing_config = reconciled.agents[existing_name]
                if (
                    existing_name.casefold() != witness.child_name.casefold()
                    or not isinstance(existing_config, LocalAgentConfig)
                    or not self._spawn_authority_registry.same_data_slot(
                        existing_config,
                        witness.config,
                    )
                ):
                    raise RuntimeError(
                        "Host spawn witness conflicts with the startup roster"
                    )
                continue
            reconciled.agents[witness.child_name] = witness.config.model_copy(
                deep=True
            )
            changed = True
        type(reconciled).model_validate(reconciled.model_dump())
        if changed and self._startup_config_path is not None:
            type(reconciled).model_validate(reconciled.model_dump())
            reconciled.save(self._startup_config_path)
        if runtime_host is not None:
            reconciled = reconciled.model_copy(deep=True)
            reconciled.host = runtime_host
            type(reconciled).model_validate(reconciled.model_dump())
        return reconciled

    def _selected_spawn_authority_records(
        self,
        authority_roots: frozenset[str] | None,
    ) -> tuple[SpawnAuthorityWitness, ...]:
        """Select the durable authority records governed by this manager."""

        records = self._spawn_authority_registry.records()
        if authority_roots is not None:
            # Standalone roots launched by ProcessManager share the host
            # registry but not one in-process fleet. Select only the transitive
            # authority closure whose parent is loaded in this process; a
            # sibling root's children must remain invisible here.
            closure = set(authority_roots)
            selected: list = []
            remaining = list(records)
            while remaining:
                next_remaining = []
                advanced = False
                for witness in remaining:
                    if witness.parent_did in closure:
                        selected.append(witness)
                        closure.add(witness.child_did)
                        advanced = True
                    else:
                        next_remaining.append(witness)
                if not advanced:
                    break
                remaining = next_remaining
            return tuple(selected)
        return records

    def _selected_pending_spawn_authority(
        self,
        authority_roots: frozenset[str] | None,
    ) -> tuple[PendingSpawnAuthority, ...]:
        """Select pre-inception denials governed by this manager's roots."""

        pending = self._spawn_authority_registry.pending()
        if authority_roots is None:
            return pending
        closure = set(authority_roots)
        remaining = list(self._spawn_authority_registry.records())
        while remaining:
            next_remaining = []
            advanced = False
            for witness in remaining:
                if witness.parent_did in closure:
                    closure.add(witness.child_did)
                    advanced = True
                else:
                    next_remaining.append(witness)
            if not advanced:
                break
            remaining = next_remaining
        return tuple(
            reservation
            for reservation in pending
            if reservation.parent_did in closure
        )

    async def _required_cold_authority_names(
        self,
        config: MultiAgentConfig,
        *,
        authority_roots: frozenset[str] | None,
    ) -> frozenset[str]:
        """Find cold ancestors that active signed children require at boot.

        ``autostart=false`` remains the operator's durable policy.  This only
        widens the effective boot set when an explicitly active descendant
        cannot verify or publish without that ancestor in memory.
        """

        active_records = tuple(
            witness
            for witness in self._selected_spawn_authority_records(authority_roots)
            if witness.active
        )
        if not active_records:
            return frozenset()

        config_names = {
            name.casefold(): name
            for name, candidate in config.agents.items()
            if isinstance(candidate, LocalAgentConfig)
        }
        by_child_did = {witness.child_did: witness for witness in active_records}
        frontier: list[str] = []
        for witness in active_records:
            name = config_names.get(witness.child_name.casefold())
            if name is None:
                continue
            candidate = config.agents[name]
            if isinstance(candidate, LocalAgentConfig) and candidate.autostart:
                frontier.append(witness.child_did)
        if not frontier:
            return frozenset()

        loaded_dids = set(authority_roots or ())
        for loaded_agent in self._agents.values():
            loaded_dids.update(_loaded_agent_bound_dids(loaded_agent))

        required_parent_dids: set[str] = set()
        while frontier:
            child_did = frontier.pop()
            witness = by_child_did.get(child_did)
            if witness is None or witness.parent_did in loaded_dids:
                continue
            parent_did = witness.parent_did
            if parent_did in required_parent_dids:
                continue
            required_parent_dids.add(parent_did)
            if parent_did in by_child_did:
                frontier.append(parent_did)

        required_names: set[str] = set()
        unresolved_root_dids = set(required_parent_dids)
        for parent_did in required_parent_dids:
            parent_witness = by_child_did.get(parent_did)
            if parent_witness is None:
                continue
            unresolved_root_dids.discard(parent_did)
            name = config_names.get(parent_witness.child_name.casefold())
            if name is None:
                continue
            candidate = config.agents[name]
            if isinstance(candidate, LocalAgentConfig) and not candidate.autostart:
                required_names.add(name)

        if not unresolved_root_dids:
            return frozenset(required_names)

        witnessed_names = {
            witness.child_name.casefold() for witness in active_records
        }
        matched_roots: dict[str, str] = {}
        for name, candidate in config.agents.items():
            if (
                not isinstance(candidate, LocalAgentConfig)
                or candidate.autostart
                or name.casefold() in witnessed_names
            ):
                continue
            try:
                resolved_dir = candidate.resolve_data_dir(self._base_data_dir)
                anchored_did = await read_anchor_agent_did(
                    str(resolved_dir),
                    mode=AgentDIDLookupMode.INSPECTION,
                )
            except Exception:
                logger.debug(
                    "Cold startup candidate %r could not be read while "
                    "resolving authority dependencies",
                    name,
                    exc_info=True,
                )
                continue
            bound_dids = frozenset({anchored_did})
            if anchored_did not in unresolved_root_dids:
                try:
                    from kestrel_sovereign.identity.runtime_identity import (
                        load_agent_identity,
                    )
                    from kestrel_sovereign.identity.signing import get_key_id

                    legacy_key_id = (
                        None
                        if anchored_did.startswith("did:web:")
                        else get_key_id(anchored_did)
                    )
                    identity = await asyncio.to_thread(
                        load_agent_identity,
                        legacy_key_id,
                        storage_dir=resolved_dir,
                    )
                    bound_dids = _identity_bound_dids(anchored_did, identity)
                except Exception:
                    logger.debug(
                        "Cold startup candidate %r could not load its verified "
                        "identity aliases while resolving authority dependencies",
                        name,
                        exc_info=True,
                    )
                    continue
            matched_dids = bound_dids.intersection(unresolved_root_dids)
            if not matched_dids:
                continue
            for matched_did in matched_dids:
                previous = matched_roots.get(matched_did)
                if previous is not None:
                    raise RuntimeError(
                        "Cold authority parent has ambiguous startup registrations"
                    )
                matched_roots[matched_did] = name
            required_names.add(name)

        return frozenset(required_names)

    def reconcile_spawn_authority_restart_roster(
        self,
        config: MultiAgentConfig,
        *,
        authority_roots: frozenset[str] | None = None,
    ) -> MultiAgentConfig:
        """Return the effective restart roster before any dependent bootstrap."""

        return self._reconcile_spawn_authority_restart_roster(
            config,
            authority_roots=authority_roots,
        )

    async def restore_spawn_authority_tree(self, root_agent: KestrelAgent) -> int:
        """Restore only descendants governed by one standalone root agent."""

        roots = _loaded_agent_bound_dids(root_agent)
        if not roots:
            raise RuntimeError("Standalone spawn restoration requires a stable root DID")
        return await self.load_from_config(
            MultiAgentConfig(agents={}),
            authority_roots=roots,
        )

    async def shutdown_spawn_authority_tree(self, root_agent: KestrelAgent) -> None:
        """Stop running descendants and retain cold authority for restart.

        Ordinary feature shutdown is a Stop, not a terminal Hold: persistent
        signed witnesses remain restartable by design.  Registry-only finite
        descendants still need an owned deadline while this process remains
        alive, even though there is no runtime for ``terminate_children`` to
        visit.
        """

        roots = _loaded_agent_bound_dids(root_agent)
        if not roots:
            raise RuntimeError("Standalone spawn shutdown requires a stable root DID")
        finite_witnesses = tuple(
            witness
            for witness in self._selected_spawn_authority_records(
                frozenset(roots)
            )
            if witness.active and witness.mandate.ttl_seconds > 0
        )
        if finite_witnesses:
            lifecycle = self._ensure_spawn_lifecycle()
            for witness in finite_witnesses:
                lifecycle.arm_cold_authority_ttl(
                    witness.child_name,
                    witness.mandate,
                    authority_parent_did=self.resolve_spawn_authority_parent_did(
                        witness.child_name,
                        witness.child_did,
                        witness.parent_did,
                    ),
                )
        outcomes: list[BaseException] = []
        for root_did in sorted(roots):
            try:
                await self.terminate_children(root_did, offboard_runtime=False)
            except BaseException as exc:
                outcomes.append(exc)
        try:
            if await self.drain_quarantined_shutdowns():
                outcomes.append(asyncio.CancelledError())
        except BaseException as exc:
            outcomes.append(exc)
        _raise_lifecycle_outcomes(
            "Standalone spawn descendant shutdown had terminal failures",
            outcomes,
        )

    async def load_from_config(
        self,
        config: MultiAgentConfig,
        *,
        authority_roots: frozenset[str] | None = None,
        restart_roster_reconciled: bool = False,
    ) -> int:
        """Load all autostart agents from a MultiAgentConfig.

        Per-agent failures are recorded in ``self._init_failures`` so the
        FastAPI lifespan handler can surface them via ``/health`` (lifecycle
        hardening #377 — without this, a multi-agent host whose providers
        all failed to initialize would silently report a healthy startup).

        Args:
            config: Effective host configuration to load.
            authority_roots: Optional restart-authority roots for standalone
                or scoped restoration.
            restart_roster_reconciled: The caller already reconciled ``config``
                against durable spawn authority and has kept that exact roster
                coupled to another authorization preflight. The server uses
                this after seeding the shared scheduler protocol; ordinary
                callers leave it false so restart authority is still enforced.

        Returns:
            Number of agents successfully loaded.
        """
        if not restart_roster_reconciled:
            config = self.reconcile_spawn_authority_restart_roster(
                config,
                authority_roots=authority_roots,
            )
        # Host authority has a deadline even when its child never publishes.
        # Arm every selected finite witness before filtering autostart rows or
        # attempting provider/storage initialization; a skipped or failed load
        # must not convert a signed TTL into permanent name/cap ownership.
        finite_witnesses = tuple(
            witness
            for witness in self._selected_spawn_authority_records(authority_roots)
            if witness.active and witness.mandate.ttl_seconds > 0
        )
        if finite_witnesses:
            lifecycle = self._ensure_spawn_lifecycle()
            for witness in finite_witnesses:
                lifecycle.arm_cold_authority_ttl(
                    witness.child_name,
                    witness.mandate,
                    authority_parent_did=witness.parent_did,
                )
        required_cold_authority_names = await self._required_cold_authority_names(
            config,
            authority_roots=authority_roots,
        )
        loaded = 0
        # Reset failure list — fresh load attempt.
        self._init_failures = []
        # RESERVE every configured port and the host's own port (codex P1
        # rounds 4-5 on #2358): a runtime-created agent must never take a port
        # something already owns — once persisted, the next boot fails
        # MultiAgentConfig's port-conflict validation and bricks startup. A
        # high host port must NOT starve the range below it (round 6).
        host_port = getattr(getattr(config, "host", None), "port", None)
        if isinstance(host_port, int):
            self._reserved_ports.add(host_port)
        for agent_cfg in config.agents.values():
            if isinstance(agent_cfg, LocalAgentConfig):
                self._reserved_ports.add(agent_cfg.port)
        pending: list[tuple[str, LocalAgentConfig]] = []
        for name, agent_cfg in config.agents.items():
            if not isinstance(agent_cfg, LocalAgentConfig):
                logger.info(f"Skipping remote agent '{name}' (not supported in-process)")
                continue
            if not agent_cfg.autostart and name not in required_cold_authority_names:
                logger.info(f"Skipping agent '{name}' (autostart=false)")
                continue
            if spawn_retirement_denies_startup(
                agent_cfg.resolve_data_dir(self._base_data_dir)
            ):
                continue
            if authority_roots is not None:
                existing_name = next(
                    (
                        published_name
                        for published_name in self._agents
                        if published_name.casefold() == name.casefold()
                    ),
                    None,
                )
                if existing_name is not None:
                    existing = self._agents[existing_name]
                    existing_did = _loaded_agent_did(existing)
                    witness = (
                        self._spawn_authority_registry.get(existing_did)
                        if isinstance(existing_did, str)
                        else None
                    )
                    if (
                        witness is not None
                        and witness.active
                        and witness.child_name.casefold() == name.casefold()
                        and self._spawn_authority_registry.same_data_slot(
                            witness.config,
                            agent_cfg,
                        )
                    ):
                        logger.info(
                            "Standalone spawn restore already has child %r loaded",
                            name,
                        )
                        continue
            pending.append((name, agent_cfg))

        # Admit every configured name before running concurrent initialization.
        # Each admission is held by this batch task until the exact result has
        # either published or completed its single cleanup attempt.  That makes
        # a concurrent dynamic create/load fail before duplicate initialization.
        admitted: list[tuple[str, LocalAgentConfig, AgentOperationAdmission]] = []
        try:
            for name, agent_cfg in pending:
                try:
                    admission, owns_admission = await self._admit_agent_operation(
                        name, kind="batch-load"
                    )
                    assert owns_admission
                    admission.spawn_candidate_config = agent_cfg.model_copy(deep=True)
                except Exception as exc:
                    logger.error("Failed to admit configured agent %r: %s", name, exc)
                    self._init_failures.append((name, exc))
                    continue
                admitted.append((name, agent_cfg, admission))

            # Agent storage, provider construction, and feature initialization
            # are independent. Run them concurrently, then register successful
            # results in config order so UI/fleet order remains stable.
            init_slots = asyncio.Semaphore(self._init_concurrency)
            stage_lock = asyncio.Lock()
            all_candidates_staged = asyncio.Event()
            stage_arrivals: set[str] = set()

            async def _stage_candidate(
                name: str,
                _agent: Optional[KestrelAgent] = None,
            ) -> None:
                """Wait until every admitted batch identity is staged or failed."""

                async with stage_lock:
                    stage_arrivals.add(self._canonical_agent_name(name))
                    if len(stage_arrivals) == len(admitted):
                        all_candidates_staged.set()
                await all_candidates_staged.wait()

            async def _bounded_initialize(name, agent_cfg, admission):
                stage_token = _BATCH_STAGE_CALLBACK.set(_stage_candidate)
                semaphore_token = _BATCH_ACTIVE_BOOT_SEMAPHORE.set(init_slots)
                try:
                    try:
                        initialized = await self._initialize_agent(name, agent_cfg)
                    except BaseException as initialization_failure:
                        # A signed sibling can already be waiting for this exact
                        # admission's immutable DID or construction boundary.
                        # Complete both before joining the batch stage barrier:
                        # the barrier itself waits for that sibling, so deferring
                        # these wakeups to the batch-wide final release deadlocks
                        # startup when this initializer failed before its anchor
                        # read or candidate construction.
                        if admission.agent_id_resolution_event is not None:
                            admission.agent_id_resolution_event.set()
                        if admission.candidate_staged_event is not None:
                            admission.candidate_staged_event.set()
                        await _stage_candidate(name)
                        if (
                            admission.host_authority_expired
                            and _contains_lifecycle_cancellation(
                                initialization_failure
                            )
                        ):
                            raise PersistedSpawnMandateExpiredError(
                                "Persisted spawn mandate expired during batch "
                                "initialization"
                            ) from initialization_failure
                        raise
                    try:
                        await _stage_candidate(name)
                    except asyncio.CancelledError:
                        # A lightweight/test initializer may not participate in
                        # the construction barrier itself. Once it returned a
                        # private agent, this wrapper owns that result even if
                        # cancellation lands while terminal staging unblocks.
                        await self._discard_unpublished_initialized_agent(
                            name,
                            initialized,
                        )
                        raise
                    return initialized
                finally:
                    _BATCH_ACTIVE_BOOT_SEMAPHORE.reset(semaphore_token)
                    _BATCH_STAGE_CALLBACK.reset(stage_token)

            init_tasks = []
            for name, agent_cfg, admission in admitted:
                init_task = asyncio.create_task(
                    _bounded_initialize(name, agent_cfg, admission),
                    name=f"agent_batch_initialize:{admission.canonical_name}",
                )
                # A finite child owns only its initializer during the concurrent
                # phase. asyncio cannot run the new task until this loop yields,
                # so the watchdog can never observe the earlier batch owner.
                admission.owner_task = init_task
                init_tasks.append(init_task)
            try:
                results = await asyncio.gather(*init_tasks, return_exceptions=True)
            except BaseException as initial_failure:
                # ``gather`` cancellation cancels pending initializers.  Results
                # that already initialized are invisible to shutdown_all(), so
                # this batch still owns their cleanup.  Cancellation wins only
                # after all of those ownership transfers have settled.
                settled = await asyncio.gather(*init_tasks, return_exceptions=True)
                initialized = [
                    (name, result)
                    for (name, _, _), result in zip(admitted, settled)
                    if not isinstance(result, BaseException)
                ]
                try:
                    cleanup_cancelled = (
                        await self._discard_unpublished_initialized_agents(initialized)
                    )
                except BaseException:
                    if isinstance(initial_failure, asyncio.CancelledError):
                        raise asyncio.CancelledError()
                    raise
                if isinstance(initial_failure, asyncio.CancelledError) or cleanup_cancelled:
                    raise asyncio.CancelledError()
                raise initial_failure

            fatal = next(
                (
                    result
                    for result in results
                    if isinstance(result, BaseException)
                    and not isinstance(result, Exception)
                ),
                None,
            )
            if fatal is not None:
                initialized = [
                    (name, result)
                    for (name, _, _), result in zip(admitted, results)
                    if not isinstance(result, BaseException)
                ]
                try:
                    cleanup_cancelled = (
                        await self._discard_unpublished_initialized_agents(initialized)
                    )
                except BaseException:
                    if isinstance(fatal, asyncio.CancelledError):
                        raise asyncio.CancelledError()
                    raise
                if isinstance(fatal, asyncio.CancelledError) or cleanup_cancelled:
                    raise asyncio.CancelledError()
                raise fatal

            unpublished = {
                name: result
                for (name, _, _), result in zip(admitted, results)
                if not isinstance(result, BaseException)
            }
            try:
                initialized_for_registration = []
                for (name, _, admission), result in zip(admitted, results):
                    if isinstance(result, BaseException):
                        e = result
                        if isinstance(e, IdentityReadinessError):
                            logger.error(
                                "Failed to load agent '%s': %s "
                                "(code=%s, cause_type=%s)",
                                name,
                                e,
                                e.error_code,
                                e.cause_type,
                            )
                        else:
                            logger.error(
                                f"Failed to load agent '{name}': {e}",
                                exc_info=(type(e), e, e.__traceback__),
                            )
                        self._init_failures.append((name, e))
                        continue
                    initialized_for_registration.append(
                        (name, admission, result)
                    )

                for name, admission, result in (
                    self._registration_order_for_initialized_agents(
                        initialized_for_registration
                    )
                ):
                    try:
                        withdrawn_after_onboarding_failure = False
                        admission_deadline = self._spawn_admission_deadline(result)
                        async with self._a2a_writer_before_spawn_admission_deadline(
                            deadline=admission_deadline,
                            phase="authority preparation",
                        ):
                            if not self._operation_is_admitted(admission):
                                raise RuntimeError(
                                    "Agent initialization completed after manager shutdown began"
                                )
                            async with self._lock:
                                self._prepare_agent_authority(name, result)
                        async with self._a2a_writer_before_spawn_admission_deadline(
                            deadline=admission_deadline,
                            phase="registration publication",
                        ):
                            if not self._operation_is_admitted(admission):
                                raise RuntimeError(
                                    "Agent initialization completed after manager shutdown began"
                                )
                            try:
                                self._onboarding_agents[name] = result
                                await self._run_host_onboarding_before_mandate_expiry(
                                    name,
                                    result,
                                    deadline=admission_deadline,
                                )
                                self._require_spawn_admission_time(
                                    admission_deadline,
                                    phase="routing publication",
                                )
                                self._commit_restored_child_ttl(name, result)
                                self._register_agent(
                                    name,
                                    result,
                                    arm_restored_ttl=False,
                                )
                                self._onboarding_agents.pop(name, None)
                                self._commit_dynamic_scheduler_registration(result)
                            except BaseException as registration_failure:
                                # Mirror the single-load path: an onboarding
                                # rejection is withdrawn before releasing the
                                # A2A writer, while cleanup remains outside the
                                # writer so it cannot block inbound topology.
                                rollback_cancelled, rollback_failure = (
                                    await self._rollback_host_onboarding(
                                        name,
                                        result,
                                    )
                                )
                                self._withdraw_initialized_agent(name, result)
                                withdrawn_after_onboarding_failure = True
                                if rollback_failure is not None:
                                    _raise_lifecycle_outcomes(
                                        "Batch onboarding and serialized rollback failed",
                                        [registration_failure, rollback_failure],
                                    )
                                if rollback_cancelled:
                                    raise asyncio.CancelledError()
                                raise
                            admission.published = True
                        unpublished.pop(name, None)
                        await self._finish_published_agent_readiness(
                            name,
                            result,
                            admission,
                            deadline=admission_deadline,
                            failure_description=(
                                "Batch agent readiness and its published rollback failed"
                            ),
                        )
                        self._commit_host_onboarding(result)
                        loaded += 1
                    except BaseException as onboarding_failure:
                        # Claim before the first cleanup await.  If that cleanup
                        # fails, the outer batch handler must not discover and
                        # shut down this same initialized agent a second time.
                        claimed = unpublished.pop(name, None)
                        cleanup_failure: Optional[BaseException] = None
                        cleanup_cancelled = False
                        if claimed is not None:
                            try:
                                cleanup_cancelled = (
                                    await self._discard_unpublished_initialized_agents(
                                        [(name, claimed)],
                                        already_withdrawn=withdrawn_after_onboarding_failure,
                                    )
                                )
                            except BaseException as error:
                                cleanup_failure = error
                        if (
                            isinstance(onboarding_failure, asyncio.CancelledError)
                            or cleanup_cancelled
                            or isinstance(cleanup_failure, asyncio.CancelledError)
                        ):
                            raise asyncio.CancelledError()
                        if cleanup_failure is not None:
                            if isinstance(onboarding_failure, Exception) and isinstance(
                                cleanup_failure, Exception
                            ):
                                raise ExceptionGroup(
                                    "Agent onboarding and its claimed cleanup failed",
                                    [onboarding_failure, cleanup_failure],
                                )
                            raise cleanup_failure
                        if isinstance(
                            onboarding_failure,
                            PersistedSpawnMandateExpiredError,
                        ):
                            agent_config = config.agents[name]
                            if not isinstance(agent_config, LocalAgentConfig):
                                raise RuntimeError(
                                    "Expired local candidate has no local configuration"
                                )
                            self._record_loaded_spawn_expiry(
                                name,
                                result,
                                agent_config,
                            )
                        if not isinstance(onboarding_failure, Exception):
                            raise onboarding_failure
                        logger.error(
                            "Failed to onboard agent %r after initialization: %s",
                            name,
                            onboarding_failure,
                            exc_info=True,
                        )
                        self._init_failures.append((name, onboarding_failure))
                        continue
            except BaseException as publication_failure:
                # ``unpublished`` contains only agents this path still owns;
                # failing onboarding claimed its result before cleanup above.
                cleanup_failure: Optional[BaseException] = None
                cleanup_cancelled = False
                if unpublished:
                    claimed_unpublished = list(unpublished.items())
                    unpublished.clear()
                    try:
                        cleanup_cancelled = (
                            await self._discard_unpublished_initialized_agents(
                                claimed_unpublished
                            )
                        )
                    except BaseException as error:
                        cleanup_failure = error
                if (
                    isinstance(publication_failure, asyncio.CancelledError)
                    or cleanup_cancelled
                    or isinstance(cleanup_failure, asyncio.CancelledError)
                ):
                    raise asyncio.CancelledError()
                if cleanup_failure is not None:
                    if isinstance(publication_failure, Exception) and isinstance(
                        cleanup_failure, Exception
                    ):
                        raise ExceptionGroup(
                            "Batch publication and claimed cleanup failed",
                            [publication_failure, cleanup_failure],
                        )
                    raise cleanup_failure
                raise
            return loaded
        finally:
            # No batch admission is reusable until all initialization,
            # publication, or claimed cleanup paths above have settled.
            release_cancelled = await self._release_agent_operations(
                [admission for _, _, admission in admitted]
            )
            if release_cancelled:
                raise asyncio.CancelledError()

    @property
    def init_failures(self) -> list[tuple[str, Exception]]:
        """Read-only view of per-agent initialization failures from the last
        ``load_from_config`` call. Used by the FastAPI lifespan to surface
        lifecycle errors (e.g. ``NoLLMProvidersError`` or an identity
        readiness failure) via ``/health``.
        """
        return list(self._init_failures)

    @property
    def cold_scheduler_identity_failures(self) -> list[tuple[str, Exception]]:
        """Unavailable configured identities omitted from scheduler routing.

        These are deliberately distinct from ``init_failures``: an autostart
        tenant may already have failed boot, and a cold tenant may be
        unincepted.  In either case the host can still route healthy peers,
        while readiness reports that this tenant's schedules are unavailable.
        """

        return list(self._cold_scheduler_identity_failures)

    def get_agent(self, name: str) -> Optional[KestrelAgent]:
        """Get an agent by name (case-insensitive)."""
        # Try exact match first
        agent = self._agents.get(name)
        if agent and not self._spawn_route_is_fenced(name, agent):
            return agent
        # Try case-insensitive
        name_lower = name.lower()
        for key, agent in self._agents.items():
            if (
                key.lower() == name_lower
                and not self._spawn_route_is_fenced(key, agent)
            ):
                return agent
        return None

    def _get_agent_for_lifecycle(self, name: str) -> Optional[KestrelAgent]:
        """Resolve private cleanup ownership even after public route fencing."""

        exact = self._agents.get(name)
        if exact is not None:
            return exact
        canonical_name = self._canonical_agent_name(name)
        return next(
            (
                agent
                for routing_name, agent in self._agents.items()
                if self._canonical_agent_name(routing_name) == canonical_name
            ),
            None,
        )

    def resolve_spawn_authority_parent_did(
        self,
        child_name: str,
        child_did: str,
        claimed_parent_did: str,
    ) -> str:
        """Map a verified signed parent alias to the manager's stable lineage DID."""

        if not isinstance(child_name, str) or not child_name:
            raise TypeError("spawn authority child name must be non-empty")
        if not isinstance(child_did, str) or not child_did:
            raise TypeError("spawn authority child DID must be non-empty")
        if not isinstance(claimed_parent_did, str) or not claimed_parent_did:
            raise TypeError("spawn authority parent DID must be non-empty")

        canonical_name = self._canonical_agent_name(child_name)
        projections = [
            mandate
            for name, mandate in self._child_mandates.items()
            if self._canonical_agent_name(name) == canonical_name
        ]
        if len(projections) > 1:
            raise RuntimeError("Spawn authority projection is ambiguous")
        if projections:
            mandate = projections[0]
            if (
                mandate.child_did != child_did
                or mandate.parent_did != claimed_parent_did
            ):
                raise RuntimeError("Spawn authority projection conflicts with witness")
            parents = [
                parent_did
                for parent_did, children in self._parent_children.items()
                if any(
                    self._canonical_agent_name(name) == canonical_name
                    for name in children
                )
            ]
            if len(parents) > 1:
                raise RuntimeError("Spawn authority projection has multiple parents")
            if parents:
                return parents[0]

        matches = [
            agent
            for agent in self._agents.values()
            if claimed_parent_did in _loaded_agent_bound_dids(agent)
        ]
        if len(matches) > 1:
            raise RuntimeError("Spawn authority parent identity is ambiguous")
        if matches:
            return _loaded_agent_did(matches[0]) or claimed_parent_did
        return claimed_parent_did

    def list_agents(self) -> dict[str, KestrelAgent]:
        """Return all loaded agents as {name: agent}."""
        return {
            name: agent
            for name, agent in self._agents.items()
            if not self._spawn_route_is_fenced(name, agent)
        }

    def validate_host_context_clause_registry(self, registry) -> None:
        """Preflight one host registry against every currently loaded agent."""

        for agent in self._agents.values():
            agent.validate_host_context_clause_registry(registry)

    def bind_host_context_clause_registry(self, registry) -> None:
        """Publish host context to loaded, initializing, and future agents."""

        self.validate_host_context_clause_registry(registry)
        agents = tuple(self._agents.values())
        for agent in agents:
            agent.bind_host_context_clause_registry(registry)
        self._host_context_clause_registry = registry
        state = self._host_context_publication_state
        state.registry = registry
        state.generation += 1
        for agent in agents:
            agent._host_context_publication_state = state
            agent._host_context_publication_generation = state.generation

    def set_host_context_publication_gate(self, gate: asyncio.Event) -> None:
        """Gate current and future agent turns until host policy is published."""

        self._host_context_publication_gate = gate
        for agent in self._agents.values():
            agent._host_context_publication_gate = gate
            agent._host_context_publication_state = (
                self._host_context_publication_state
            )

    async def local_agent_configs_by_did(
        self,
        config: MultiAgentConfig,
    ) -> dict[str, tuple[str, LocalAgentConfig]]:
        """Return every local configured agent keyed by its durable DID.

        Unlike :meth:`list_agents`, this includes ``autostart=false`` agents.
        A host-owned scheduler needs that complete map to wake a cold target
        after it atomically claims a due row in the shared PostgreSQL database.
        Loaded agents provide their in-memory identity. Explicitly cold
        identities use an immutable local read; autostart identities instead
        use the same read/write WAL-recovery lookup as imminent initialization.
        """
        mapping: dict[str, tuple[str, LocalAgentConfig]] = {}
        self._cold_scheduler_identity_failures = []
        for name, agent_config in config.agents.items():
            if not isinstance(agent_config, LocalAgentConfig):
                continue
            resolved_dir = agent_config.resolve_data_dir(self._base_data_dir)
            if spawn_retirement_denies_startup(resolved_dir):
                # Retirement is desired-state denial, not merely an autostart
                # preference.  A host scheduler must not reauthorize the same
                # row for a cold wake after normal startup correctly skipped it.
                continue

            agent = self._agents.get(name)
            agent_id = _loaded_agent_did(agent) if agent is not None else None
            if not isinstance(agent_id, str) or not agent_id:
                try:
                    # An autostart entry is about to open this exact local
                    # identity database during normal startup. Let shared
                    # scheduler preflight take that same recovery path, so an
                    # interrupted WAL is replayed before scheduler authority is
                    # seeded. Explicitly cold entries remain immutable-read-only
                    # until a scheduler cold wake is authorized to initialize.
                    lookup_mode = (
                        AgentDIDLookupMode.INITIALIZATION
                        if agent_config.autostart
                        else AgentDIDLookupMode.COLD_READ_ONLY
                    )
                    agent_id = await read_anchor_agent_did(
                        str(resolved_dir),
                        mode=lookup_mode,
                    )
                except Exception as exc:
                    # An autostart agent may already have failed its independent
                    # boot attempt.  Do not let a second cold-DID lookup roll
                    # back healthy PostgreSQL tenants; omit only the unresolved
                    # DID from scheduler authority and surface its failure in
                    # readiness.  This is equally important for an explicitly
                    # cold entry: neither state is safe to authorize without a
                    # durable local identity witness.
                    logger.warning(
                        "Skipping unavailable scheduler identity for agent %r: %s",
                        name,
                        exc,
                        exc_info=True,
                    )
                    self._cold_scheduler_identity_failures.append((name, exc))
                    continue

            if agent_id in mapping:
                existing_name = mapping[agent_id][0]
                raise ValueError(
                    "Local multi-agent configuration maps the same DID to "
                    f"both {existing_name!r} and {name!r}: {agent_id!r}"
                )
            mapping[agent_id] = (name, agent_config)
        self._seed_scheduler_authority(mapping)
        return mapping

    async def resolve_registered_agent_id(
        self,
        name: str,
        config: LocalAgentConfig,
    ) -> str:
        """Resolve a persisted local registration to one durable DID.

        Destructive offboarding calls this before mutating ``multi_agent.toml``.
        A loaded or scheduler-authorized identity is accepted only when it is
        coherent with the requested registration; an otherwise cold identity
        is read through the immutable anchor path.  Failure is intentionally a
        refusal, never permission to guess a namespace from the routing name.
        """

        if not isinstance(config, LocalAgentConfig):
            raise ValueError("Registered agent is not a local hosted agent.")
        canonical_name = self._canonical_agent_name(name)
        async with self._lock:
            loaded_matches = [
                agent
                for routing_name, agent in self._agents.items()
                if self._canonical_agent_name(routing_name) == canonical_name
            ]
            authority_matches = [
                (agent_id, entry)
                for agent_id, entry in self._scheduler_authority_by_did.items()
                if self._canonical_agent_name(entry[0]) == canonical_name
            ]
        if len(loaded_matches) > 1 or len(authority_matches) > 1:
            raise ValueError(
                "Registered agent identity is ambiguous; offboarding was refused."
            )

        loaded_id = (
            _loaded_agent_did(loaded_matches[0]) if loaded_matches else None
        )
        authority_id = authority_matches[0][0] if authority_matches else None
        if loaded_id and authority_id and loaded_id != authority_id:
            raise ValueError(
                "Registered agent identity conflicts with live scheduler authority; "
                "offboarding was refused."
            )
        if loaded_id:
            return loaded_id
        if authority_id:
            authority_config = authority_matches[0][1][1]
            if authority_config != config:
                raise ValueError(
                    "Persisted agent registration changed from scheduler authority; "
                    "offboarding was refused."
                )
            return authority_id

        try:
            resolved_dir = config.resolve_data_dir(self._base_data_dir)
            agent_id = await read_anchor_agent_did(
                str(resolved_dir),
                mode=AgentDIDLookupMode.COLD_READ_ONLY,
            )
        except Exception as exc:
            raise ValueError(
                "Registered agent identity is unavailable; offboarding was refused."
            ) from exc
        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError(
                "Registered agent identity is unavailable; offboarding was refused."
            )

        # Recheck after I/O so a concurrent cold wake cannot replace the
        # registration with a different DID between witness and DELETE.
        async with self._lock:
            _published_name, current = self._published_agent_binding(name)
            current_id = _loaded_agent_did(current) if current is not None else None
            _authority_name, known_id = self._scheduler_authority_binding_by_name(
                name
            )
            did_authority = self._scheduler_authority_by_did.get(agent_id)
        if (current_id and current_id != agent_id) or (
            known_id and known_id != agent_id
        ) or (
            did_authority is not None
            and self._canonical_agent_name(did_authority[0]) != canonical_name
        ):
            raise ValueError(
                "Registered agent identity changed during offboarding admission; "
                "offboarding was refused."
            )
        return agent_id

    def get_agent_name(self, agent_id: str) -> Optional[str]:
        """Get the name for an agent by its DID."""
        name = self._agent_names.get(agent_id)
        if name is None:
            return None
        agent = self._agents.get(name)
        if agent is not None and self._spawn_route_is_fenced(name, agent):
            return None
        return name

    async def remove_agent(
        self,
        name: str,
        *,
        offboard_runtime: bool = False,
        known_agent_id: Optional[str] = None,
        known_agent_config: Optional[LocalAgentConfig] = None,
        offboarding_admission: Optional[RuntimeOffboardingAdmission] = None,
        _spawn_rollback_admission: Optional[AgentOperationAdmission] = None,
        _scheduler_lifecycle_lock_held: bool = False,
        _preserve_scheduler_authority: bool = False,
        _terminal_descendant_fence_active: bool = False,
        _prestarted_spawn_retirement: Optional[
            SpawnRetirementAdmission
        ] = None,
    ) -> bool:
        """Stop/unpublish an agent and optionally offboard its runtime tree.

        Destructive filesystem cleanup is admitted before unpublication, but
        it is awaited only after all manager, A2A, and scheduler lifecycle
        locks have been released. A slow filesystem therefore cannot wedge
        unrelated tenant lifecycle operations. The wait is bounded; timeout
        or caller cancellation leaves the exact cleanup worker manager-owned
        and reports cleanup as pending without restoring a stopped agent.
        """

        if _spawn_rollback_admission is not None and (
            _spawn_rollback_admission.canonical_name
            != self._canonical_agent_name(name)
        ):
            raise ValueError("spawn rollback admission does not own this agent name")
        if _prestarted_spawn_retirement is not None:
            if not offboard_runtime or _spawn_rollback_admission is not None:
                raise ValueError(
                    "prestarted spawn retirement requires destructive offboarding"
                )
            if (
                _prestarted_spawn_retirement.witness.child_name.casefold()
                != name.casefold()
            ):
                raise ValueError(
                    "prestarted spawn retirement does not own this agent name"
                )
        if _preserve_scheduler_authority and not _scheduler_lifecycle_lock_held:
            raise ValueError(
                "scheduler authority may be preserved only by its current "
                "cold-wake lifecycle owner"
            )
        if _spawn_rollback_admission is None:
            join_cancelled = await self._join_active_spawn_before_removal(name)
            if join_cancelled:
                raise asyncio.CancelledError()
        if (
            offboard_runtime
            and _spawn_rollback_admission is None
            and not _terminal_descendant_fence_active
        ):
            async with self._lock:
                _published_name, published = self._published_agent_binding(name)
                parent_dids = set(_loaded_agent_bound_dids(published))
                parent_did = _loaded_agent_did(published)
                if not isinstance(parent_did, str) or not parent_did:
                    parent_did = known_agent_id
                if not isinstance(parent_did, str) or not parent_did:
                    _authority_name, parent_did = (
                        self._scheduler_authority_binding_by_name(name)
                    )
                if isinstance(parent_did, str):
                    parent_dids.add(parent_did)
                    existing_descendants = self._spawned_descendant_names(
                        parent_dids
                    )
                    if any(
                        child_name in self._child_budgets
                        for child_name in existing_descendants
                    ):
                        raise ValueError(
                            f"Cannot remove '{name}' directly: it has budgeted child "
                            "agents. Terminate children first."
                        )
                    if existing_descendants:
                        raise RuntimeError(
                            "Refusing destructive removal of an agent with "
                            "active spawned descendants"
                        )
            if parent_dids:
                fences = []
                try:
                    for fenced_parent_did in sorted(parent_dids):
                        fences.append(
                            await self.begin_terminal_descendant_spawn_fence(
                                fenced_parent_did
                            )
                        )
                    async with self._lock:
                        descendants = self._spawned_descendant_names(parent_dids)
                        if any(
                            child_name not in self._child_budgets
                            for child_name in descendants
                        ):
                            raise RuntimeError(
                                "Refusing destructive removal of an agent with "
                                "active spawned descendants"
                            )
                    return await self.remove_agent(
                        name,
                        offboard_runtime=offboard_runtime,
                        known_agent_id=known_agent_id,
                        known_agent_config=known_agent_config,
                        offboarding_admission=offboarding_admission,
                        _spawn_rollback_admission=_spawn_rollback_admission,
                        _scheduler_lifecycle_lock_held=(_scheduler_lifecycle_lock_held),
                        _preserve_scheduler_authority=(_preserve_scheduler_authority),
                        _terminal_descendant_fence_active=True,
                        _prestarted_spawn_retirement=(
                            _prestarted_spawn_retirement
                        ),
                    )
                finally:
                    for fence in reversed(fences):
                        self.end_terminal_descendant_spawn_fence(fence)

        if offboarding_admission is not None and (
            not offboard_runtime
            or not isinstance(offboarding_admission, RuntimeOffboardingAdmission)
        ):
            raise TypeError(
                "offboarding_admission requires destructive offboarding and a "
                "RuntimeOffboardingAdmission witness"
            )

        destructive_spawn_witness = _prestarted_spawn_retirement
        if offboard_runtime and _spawn_rollback_admission is None:
            if destructive_spawn_witness is None:
                destructive_spawn_witness = (
                    await self._begin_destructive_spawn_retirement(
                        name,
                        known_agent_id=known_agent_id,
                    )
                )

        pending_offboarding: list[InflightRuntimeOffboarding] = []
        removed = False
        primary_failure: BaseException | None = None
        try:
            removed = await self._remove_agent_serialized(
                name,
                offboard_runtime=offboard_runtime,
                known_agent_id=known_agent_id,
                known_agent_config=known_agent_config,
                pending_offboarding=pending_offboarding,
                offboarding_admission=offboarding_admission,
                scheduler_lifecycle_lock_held=(
                    _scheduler_lifecycle_lock_held
                ),
                preserve_scheduler_authority=_preserve_scheduler_authority,
            )
        except BaseException as exc:
            primary_failure = exc

        # A record enters ``pending_offboarding`` only after the cleanup task
        # has been created and registered under the manager lock. Mark the
        # witness even when later reconciliation/cancellation fails so a cold
        # DELETE never restores autostart registration over admitted cleanup.
        if pending_offboarding and offboarding_admission is not None:
            offboarding_admission.started = True

        offboarding_cancelled = False
        offboarding_failure: BaseException | None = None
        if pending_offboarding:
            if len(pending_offboarding) != 1:
                raise RuntimeError("one agent removal admitted multiple offboardings")
            (
                offboarding_cancelled,
                offboarding_failure,
            ) = await self._finish_agent_runtime_offboarding(
                pending_offboarding[0],
                cancellation_already_observed=_contains_lifecycle_cancellation(
                    primary_failure
                ),
            )

        retirement_failure: BaseException | None = None
        try:
            await self._settle_destructive_spawn_retirement(
                destructive_spawn_witness
            )
        except BaseException as exc:
            retirement_failure = exc

        outcomes: list[BaseException] = []
        if primary_failure is not None:
            outcomes.append(primary_failure)
        if offboarding_cancelled and not _contains_lifecycle_cancellation(
            primary_failure
        ):
            outcomes.append(asyncio.CancelledError())
        if offboarding_failure is not None:
            outcomes.append(offboarding_failure)
        if retirement_failure is not None:
            outcomes.append(retirement_failure)
        if len(outcomes) == 1:
            raise outcomes[0]
        if outcomes:
            raise BaseExceptionGroup(
                "Agent removal had multiple terminal outcomes", outcomes
            )
        return removed

    async def _join_active_spawn_before_removal(self, name: str) -> bool:
        """Fence DELETE behind the spawn that still owns receipt rollback.

        A published child can remain inside provider I/O before governance
        commit.  Its graph is the sole durable receipt-revocation handle, so a
        concurrent remover must join that spawn before it may shut the child
        down.  The spawn's own terminal exception belongs to its caller; this
        remover continues against the authoritative post-spawn state.
        """

        async with self._lock:
            admission = self._agent_operations.get(
                self._canonical_agent_name(name)
            )
            spawn_task = (
                admission.spawn_task
                if admission is not None
                and admission.kind in {"spawn", "direct-spawn-test"}
                else None
            )
        if spawn_task is None:
            return False
        current_task = asyncio.current_task()
        delegated_admission = (
            self._admission_delegate_tasks.get(current_task)
            if current_task is not None
            else None
        )
        if spawn_task is current_task or delegated_admission is admission:
            if admission is not None and not admission.committed:
                # Ready hooks run inside the spawn owner.  Allowing that same
                # task to shut down its child closes the only graph handle before
                # the outer rollback can revoke the signed receipt.  The explicit
                # rollback admission bypasses this join and remains the sole
                # legal same-task removal before governance commits.
                raise RuntimeError(
                    "Cannot remove an uncommitted spawned agent before its "
                    "governance commit; the spawn owner must revoke its receipt first"
                )
            return False
        cancelled, _failure = await await_lifecycle_task_completion(spawn_task)
        return cancelled

    async def _remove_agent_serialized(
        self,
        name: str,
        *,
        offboard_runtime: bool,
        known_agent_id: Optional[str],
        known_agent_config: Optional[LocalAgentConfig],
        pending_offboarding: list[InflightRuntimeOffboarding],
        offboarding_admission: Optional[RuntimeOffboardingAdmission],
        scheduler_lifecycle_lock_held: bool = False,
        preserve_scheduler_authority: bool = False,
    ) -> bool:
        """Stop and unpublish an agent while serializing with cold wakes.

        ``offboard_runtime`` is an explicit destructive intent. Ordinary host
        shutdown/restart leaves it false so an agent's isolated-feature venv,
        credentials, configuration, and state survive process teardown. Only
        administrative deprovisioning and rollback of an uncommitted child may
        set it true.

        The shared per-DID lock is acquired before authority is revoked and
        held through shutdown/unpublication.  A hosted executor that holds the
        same lock cannot observe stale config and re-load this agent after an
        administrative DELETE. If shutdown fails and the agent remains live,
        restore the exact prior authority. If shutdown succeeds but requested
        offboarding fails, routing and authority stay withdrawn and
        :class:`RuntimeOffboardingRetainedError` reports the retained tree.
        """
        if type(offboard_runtime) is not bool:
            raise TypeError("offboard_runtime must be a bool")
        if known_agent_id is not None and (
            not offboard_runtime
            or type(known_agent_id) is not str
            or not known_agent_id
        ):
            raise TypeError(
                "known_agent_id requires destructive offboarding and a non-empty DID"
            )
        if known_agent_config is not None and (
            known_agent_id is None
            or not offboard_runtime
            or not isinstance(known_agent_config, LocalAgentConfig)
        ):
            raise TypeError(
                "known_agent_config requires destructive offboarding, a known DID, "
                "and a LocalAgentConfig"
            )

        async with self._lock:
            published_name, current = self._published_agent_binding(name)
            agent_id = _loaded_agent_did(current) if current is not None else None
            if agent_id and known_agent_id and agent_id != known_agent_id:
                raise ValueError(
                    "Registered agent identity does not match the loaded agent; "
                    "offboarding was refused."
                )
            authority_name, authority_id = self._scheduler_authority_binding_by_name(
                name
            )
            if not isinstance(agent_id, str) or not agent_id:
                # A scheduler can be partway through a cold load while the
                # agent is still absent from ``_agents``.  The live authority
                # registry is the only durable-in-process witness for that
                # configured DID at this point.  Use it to wait on the same
                # lifecycle lock and revoke the desired state, rather than
                # returning a misleading 404 and letting the already-claimed
                # cold wake publish the agent immediately afterwards.
                agent_id = authority_id
            if agent_id and known_agent_id and agent_id != known_agent_id:
                raise ValueError(
                    "Registered agent identity does not match scheduler authority; "
                    "offboarding was refused."
                )
            if not agent_id and known_agent_id:
                agent_id = known_agent_id
            # From this point onward, every routing mutation uses the exact
            # published/authorized spelling. A case-variant administrative
            # request must never miss a live process yet still reach its
            # canonical DID's destructive cleanup.
            name = published_name or authority_name or name

        if not isinstance(agent_id, str) or not agent_id:
            async with self._a2a_lifecycle_lock:
                return await self._remove_agent_without_scheduler_lifecycle(
                    name,
                    offboard_runtime=offboard_runtime,
                    pending_offboarding=pending_offboarding,
                    offboarding_admission=offboarding_admission,
                )

        cold_identity_offboarding: Optional[
            tuple[
                str,
                Optional[tuple[str, LocalAgentConfig]],
                Optional[LocalAgentConfig],
            ]
        ] = None
        @asynccontextmanager
        async def scheduler_writer():
            if scheduler_lifecycle_lock_held:
                # Hosted cold execution owns this exact DID's non-reentrant
                # writer until load_agent returns. Readiness rollback is part
                # of that same operation and must not acquire it recursively.
                yield
                return
            async with self.scheduler_lifecycle_lock(agent_id):
                yield

        async with scheduler_writer():
            async with self._a2a_lifecycle_lock:
                # Re-read after waiting: another lifecycle operation may have
                # completed first. The exact published key may differ from the
                # request only by case; retain that key through every mutation.
                published_name, current = self._published_agent_binding(name)
                if published_name is not None:
                    name = published_name
                current_did = (
                    _loaded_agent_did(current) if current is not None else None
                )
                authority = self.scheduler_authority_for(agent_id)
                if current is None:
                    # DELETE won the race with a cold scheduler load. Revoke
                    # desired state while the lifecycle writers are held, then
                    # admit destructive I/O only after releasing them. The
                    # separate manager-lock admission below owns the terminal
                    # seal check and inflight registration atomically.
                    if authority is not None and self._canonical_agent_name(
                        authority[0]
                    ) == self._canonical_agent_name(name):
                        name = authority[0]
                        revoked = self._revoke_scheduler_authority(name, agent_id)
                        if offboard_runtime:
                            cold_identity_offboarding = (
                                name,
                                revoked,
                                revoked[1] if revoked is not None else known_agent_config,
                            )
                        else:
                            logger.info(
                                "Revoked cold scheduler authority for agent %r",
                                name,
                            )
                            reconciliation_cancelled = (
                                await self._reconcile_fully_removed_child_tracking()
                            )
                            if reconciliation_cancelled:
                                raise asyncio.CancelledError()
                            return True
                    elif known_agent_id == agent_id:
                        if authority is not None:
                            raise ValueError(
                                "Registered agent DID belongs to a different routing "
                                "name; offboarding was refused."
                            )
                        if self._has_budgeted_descendants(
                            name,
                            known_agent_id=agent_id,
                        ):
                            raise ValueError(
                                f"Cannot remove '{name}' directly: it has budgeted "
                                "child agents. Use terminate_child, which cascades "
                                "and releases nested budgets leaf-first (#2113)."
                            )
                        identity_config = (
                            known_agent_config or self._created_configs.get(name)
                        )
                        if (
                            self._hosted_agent_runtime_factory_configured(
                                os.environ.get("KESTREL_DB_BACKEND", "sqlite"),
                                os.environ.get("KESTREL_DATABASE_URL"),
                            )
                            and identity_config is None
                        ):
                            raise ValueError(
                                "Registered hosted agent configuration is unavailable; "
                                "offboarding was refused before filesystem cleanup."
                            )
                        revoked = self._revoke_scheduler_authority(name, agent_id)
                        cold_identity_offboarding = (
                            name,
                            revoked,
                            identity_config,
                        )
                    else:
                        return await self._remove_agent_without_scheduler_lifecycle(
                            name,
                            offboard_runtime=offboard_runtime,
                            pending_offboarding=pending_offboarding,
                            offboarding_admission=offboarding_admission,
                        )
                else:
                    if current_did != agent_id:
                        # This DELETE resolved ``name`` to ``agent_id`` before it
                        # waited for that DID's scheduler writer. A replacement
                        # identity may publish after an earlier same-name DELETE
                        # completes. Never remove that replacement while holding
                        # the obsolete identity's lifecycle lock.
                        logger.info(
                            "Agent %r changed identity while waiting for removal; "
                            "refusing to remove the replacement",
                            name,
                        )
                        return False

                    revoked = None
                    if (
                        not preserve_scheduler_authority
                        and authority is not None
                        and self._canonical_agent_name(
                            authority[0]
                        ) == self._canonical_agent_name(name)
                    ):
                        revoked = self._revoke_scheduler_authority(
                            authority[0], agent_id
                        )
                    try:
                        removed = await self._remove_agent_without_scheduler_lifecycle(
                            name,
                            offboard_runtime=offboard_runtime,
                            pending_offboarding=pending_offboarding,
                            offboarding_admission=offboarding_admission,
                        )
                    except BaseException:
                        if self._published_agent_binding(name)[1] is not None:
                            self._restore_scheduler_authority(agent_id, revoked)
                        raise
                    if not removed:
                        self._restore_scheduler_authority(agent_id, revoked)
                    else:
                        (
                            onboarding_cancelled,
                            onboarding_failure,
                        ) = await self._rollback_host_onboarding(name, current)
                        onboarding_outcomes: list[BaseException] = []
                        if onboarding_cancelled:
                            onboarding_outcomes.append(asyncio.CancelledError())
                        if onboarding_failure is not None:
                            onboarding_outcomes.append(onboarding_failure)
                        _raise_lifecycle_outcomes(
                            f"Agent {name!r} removal and provisional host "
                            "onboarding rollback had terminal outcomes",
                            onboarding_outcomes,
                        )
                    return removed

        assert cold_identity_offboarding is not None
        cold_name, revoked, cold_config = cold_identity_offboarding
        admitted, admission_cancelled = (
            await self._admit_agent_runtime_offboarding_identity(
                name=cold_name,
                agent_id=agent_id,
                config=cold_config,
                revoked=revoked,
                pending_offboarding=pending_offboarding,
            )
        )
        if not admitted:
            logger.warning(
                "Refusing identity offboarding of agent %r after terminal manager "
                "shutdown handoffs were sealed",
                cold_name,
            )
            if admission_cancelled:
                raise asyncio.CancelledError()
            return False
        reconciliation_cancelled = (
            await self._reconcile_fully_removed_child_tracking()
        )
        logger.info(
            "Offboarded registered cold agent %r by durable identity",
            cold_name,
        )
        if admission_cancelled or reconciliation_cancelled:
            raise asyncio.CancelledError()
        return True

    def _handoff_shutdown_to_quarantined_reaper(
        self,
        *,
        name: str,
        agent: KestrelAgent,
        shutdown_task: "asyncio.Future[object]",
        offboard_runtime: bool,
    ) -> bool:
        """Retain durable shutdown cleanup without extending a removal timeout.

        The handoff is deliberately opt-in.  A legacy/test agent without the
        concrete lifecycle contract continues through the conservative join
        path below; only an agent that explicitly promises to preserve its
        durable owner/storage may be withdrawn while its reaper is live.
        """
        if self._quarantined_shutdown_handoffs_sealed:
            return False
        if not _has_shutdown_reaper_handoff_contract(agent):
            return False
        handoff = getattr(type(agent), "handoff_shutdown_to_reaper")
        shutdown_reaper = handoff(agent, shutdown_task)
        if not isinstance(shutdown_reaper, asyncio.Future):
            raise TypeError(
                "agent shutdown reaper handoff must return an asyncio future"
            )

        async def finish_shutdown_and_optional_offboarding() -> object:
            await shutdown_reaper
            if offboard_runtime:
                (
                    cleanup_cancelled,
                    cleanup_failure,
                ) = await self._offboard_agent_runtime_namespace(agent)
                if cleanup_failure is not None:
                    if cleanup_cancelled:
                        raise BaseExceptionGroup(
                            "Runtime offboarding had multiple terminal outcomes",
                            [asyncio.CancelledError(), cleanup_failure],
                        )
                    raise cleanup_failure
                if cleanup_cancelled:
                    raise asyncio.CancelledError()
                from kestrel_sovereign.features.isolated_runtime import (
                    RuntimeNamespaceCleanupOutcome,
                )

                return RuntimeNamespaceCleanupOutcome.REMOVED
            return None

        # The retained reaper always owns durable agent shutdown. It owns
        # tenant-tree deletion only when the original caller carried explicit
        # offboarding intent; timeout/cancellation must never turn an ordinary
        # host restart into deprovisioning.
        reaper_task = asyncio.create_task(
            finish_shutdown_and_optional_offboarding(),
            name=(
                f"agent_runtime_offboard:{name}"
                if offboard_runtime
                else f"agent_shutdown_reaper:{name}"
            ),
        )

        self._retain_quarantined_cleanup(
            name=name,
            agent_id=_loaded_agent_did(agent) or "<unknown>",
            task=reaper_task,
            runtime_outcome_required=offboard_runtime,
        )
        logger.warning(
            "Handed agent %r to quarantined shutdown cleanup; routing is "
            "withdrawn but durable owner/storage remain retained until it settles.",
            name,
        )
        return True

    async def _offboard_agent_runtime_namespace(
        self,
        agent: KestrelAgent,
    ) -> tuple[bool, Optional[BaseException]]:
        """Join deletion owned by an already-retained shutdown reaper.

        This path is intentionally unbounded only inside the manager-owned
        quarantine task; it never runs under manager, A2A, or scheduler
        lifecycle locks. Direct administrative deletion uses the bounded
        inflight path below.
        """

        from kestrel_sovereign.features.isolated_runtime import (
            remove_agent_runtime_namespace,
        )

        cleanup_task = asyncio.create_task(
            asyncio.to_thread(remove_agent_runtime_namespace, agent),
            name="isolated_runtime_namespace_quarantined_offboard",
        )
        cancelled, failure = await await_lifecycle_task_completion(cleanup_task)
        if failure is not None:
            return cancelled, failure
        result = cleanup_task.result()
        agent_id = _loaded_agent_did(agent) or "<unknown>"
        agent_name = self._agent_names.get(agent_id)
        if type(agent_name) is not str or not agent_name:
            agent_name = agent_id
        return cancelled, _runtime_offboarding_outcome_error(
            agent_name=agent_name,
            agent_id=agent_id,
            result=result,
        )

    def _start_agent_runtime_offboarding(
        self,
        *,
        name: str,
        agent: KestrelAgent,
    ) -> InflightRuntimeOffboarding:
        """Admit secure deletion while removal still owns the manager lock."""

        from kestrel_sovereign.features.isolated_runtime import (
            remove_agent_runtime_namespace,
        )

        cleanup_task = asyncio.create_task(
            asyncio.to_thread(remove_agent_runtime_namespace, agent),
            name=f"isolated_runtime_namespace_offboard:{name}",
        )
        record = InflightRuntimeOffboarding(
            agent_name=name,
            agent_id=_loaded_agent_did(agent) or "<unknown>",
            runtime_path=_agent_runtime_path(agent),
            task=cleanup_task,
        )
        record_key = id(cleanup_task)
        self._inflight_runtime_offboardings[record_key] = record

        def retire_inflight(_task: "asyncio.Future[object]") -> None:
            self._inflight_runtime_offboardings.pop(record_key, None)

        cleanup_task.add_done_callback(retire_inflight)
        return record

    async def _admit_agent_runtime_offboarding_identity(
        self,
        *,
        name: str,
        agent_id: str,
        config: Optional[LocalAgentConfig],
        revoked: Optional[tuple[str, LocalAgentConfig]],
        pending_offboarding: list[InflightRuntimeOffboarding],
    ) -> tuple[bool, bool]:
        """Atomically seal-check and register one cold-identity deletion.

        Callers deliberately release scheduler and A2A lifecycle writers
        before entering this method.  The manager lock is the sole
        linearization boundary shared with terminal drain sealing: either the
        inflight task is registered before the seal and the drain joins it, or
        the sealed refusal restores desired-state revocation without starting
        filesystem work.  Cancellation is remembered only after that ownership
        decision has completed.
        """

        acquire = asyncio.create_task(
            self._lock.acquire(),
            name="agent_manager:admit_identity_runtime_offboarding",
        )
        cancelled, failure = await await_lifecycle_task_completion(acquire)
        if failure is not None:
            raise RuntimeError(
                "Unable to serialize identity runtime offboarding admission"
            ) from failure
        try:
            if self._quarantined_shutdown_handoffs_sealed:
                if revoked is not None:
                    self._restore_scheduler_authority(agent_id, revoked)
                else:
                    self._scheduler_revoked_names.discard(name)
                    self._scheduler_revoked_dids.discard(agent_id)
                return False, cancelled
            try:
                record = self._start_agent_runtime_offboarding_identity(
                    name=name,
                    agent_id=agent_id,
                    config=config,
                )
            except BaseException:
                if revoked is not None:
                    self._restore_scheduler_authority(agent_id, revoked)
                else:
                    self._scheduler_revoked_names.discard(name)
                    self._scheduler_revoked_dids.discard(agent_id)
                raise
            pending_offboarding.append(record)
            return True, cancelled
        finally:
            self._lock.release()

    def _start_agent_runtime_offboarding_identity(
        self,
        *,
        name: str,
        agent_id: str,
        config: Optional[LocalAgentConfig] = None,
    ) -> InflightRuntimeOffboarding:
        """Register deletion for a cold agent while the caller owns ``_lock``."""

        from kestrel_sovereign.features.isolated_runtime import (
            remove_runtime_namespace,
            resolve_legacy_isolated_runtime_root,
            resolve_isolated_runtime_namespace,
        )

        scope = None
        if self._hosted_agent_runtime_factory_configured(
            os.environ.get("KESTREL_DB_BACKEND", "sqlite"),
            os.environ.get("KESTREL_DATABASE_URL"),
        ):
            root, namespace = self._isolated_runtime_scope(agent_id)
            scope = resolve_isolated_runtime_namespace(root, namespace)
        legacy_root = None
        if scope is not None and config is not None:
            resolved_data_dir = config.resolve_data_dir(self._base_data_dir)
            try:
                data_dir_metadata = resolved_data_dir.stat(follow_symlinks=False)
            except FileNotFoundError:
                data_dir_metadata = None
            if data_dir_metadata is not None:
                if not stat.S_ISDIR(data_dir_metadata.st_mode):
                    raise ValueError(
                        "Registered agent data directory is unsafe; offboarding "
                        "was refused."
                    )
                legacy_root = resolve_legacy_isolated_runtime_root(
                    resolved_data_dir / "feature_venvs",
                    scope,
                )

        # A concurrent destructive owner may already have admitted this DID
        # before routing disappeared.  Join that exact worker instead of
        # launching two recursive deletions against the same tenant tree.
        for existing in self._inflight_runtime_offboardings.values():
            if existing.agent_id == agent_id:
                return existing
        cleanup_task = asyncio.create_task(
            asyncio.to_thread(
                remove_runtime_namespace,
                scope,
                agent_id,
                legacy_root,
            ),
            name=f"isolated_runtime_namespace_offboard:{name}",
        )
        record = InflightRuntimeOffboarding(
            agent_name=name,
            agent_id=agent_id,
            runtime_path=scope.path if scope is not None else None,
            task=cleanup_task,
        )
        record_key = id(cleanup_task)
        self._inflight_runtime_offboardings[record_key] = record

        def retire_inflight(_task: "asyncio.Future[object]") -> None:
            self._inflight_runtime_offboardings.pop(record_key, None)

        cleanup_task.add_done_callback(retire_inflight)
        return record

    def _retain_timed_out_runtime_offboarding(
        self, record: InflightRuntimeOffboarding
    ) -> None:
        """Publish bounded operational ownership unless a terminal drain owns it."""

        if self._quarantined_shutdown_handoffs_sealed:
            # The inflight registry was populated before the seal and is part
            # of the terminal drain's join set. Registering a second owner
            # after the seal would instead race the drain's empty snapshot.
            return
        self._retain_quarantined_cleanup(
            name=record.agent_name,
            agent_id=record.agent_id,
            task=record.task,
            runtime_outcome_required=True,
        )

    async def _finish_agent_runtime_offboarding(
        self,
        record: InflightRuntimeOffboarding,
        *,
        cancellation_already_observed: bool,
    ) -> tuple[bool, Optional[BaseException]]:
        """Bound an administrative wait without cancelling filesystem cleanup."""

        def retained(
            cause: BaseException,
            *,
            cleanup_pending: bool = False,
        ) -> RuntimeOffboardingRetainedError:
            return RuntimeOffboardingRetainedError(
                agent_name=record.agent_name,
                agent_id=record.agent_id,
                runtime_path=record.runtime_path,
                cause=cause,
                cleanup_pending=cleanup_pending,
            )

        if cancellation_already_observed and not record.task.done():
            self._retain_timed_out_runtime_offboarding(record)
            return False, retained(
                RuntimeError("runtime offboarding continues after cancellation"),
                cleanup_pending=True,
            )

        try:
            done, _ = await asyncio.wait(
                {record.task}, timeout=RUNTIME_OFFBOARD_TIMEOUT_S
            )
        except asyncio.CancelledError:
            self._retain_timed_out_runtime_offboarding(record)
            return True, retained(
                RuntimeError("runtime offboarding continues after cancellation"),
                cleanup_pending=True,
            )

        if not done:
            self._retain_timed_out_runtime_offboarding(record)
            return False, retained(
                TimeoutError(
                    "secure runtime offboarding exceeded its administrative wait bound"
                ),
                cleanup_pending=True,
            )
        if record.task.cancelled():
            return cancellation_already_observed, retained(
                RuntimeError("secure runtime offboarding task was cancelled")
            )
        failure = record.task.exception()
        if failure is None:
            result = record.task.result()
            outcome_error = _runtime_offboarding_outcome_error(
                agent_name=record.agent_name,
                agent_id=record.agent_id,
                result=result,
            )
            if outcome_error is None:
                return cancellation_already_observed, None
            if isinstance(outcome_error, RuntimeOffboardingNotPerformedError):
                return cancellation_already_observed, outcome_error
            return cancellation_already_observed, retained(outcome_error)
        if not isinstance(failure, Exception):
            raise failure
        return cancellation_already_observed, retained(failure)

    def _retain_quarantined_cleanup(
        self,
        *,
        name: str,
        agent_id: str,
        task: "asyncio.Future[object]",
        runtime_outcome_required: bool = False,
    ) -> str:
        """Keep one live cleanup task, then collapse it to bounded metadata."""

        if self._quarantined_shutdown_handoffs_sealed:
            raise RuntimeError(
                "cannot retain quarantined cleanup after the terminal manager "
                "drain has been sealed"
        )

        self._next_shutdown_reaper_id += 1
        # Reaper ids are operator-facing retained metadata too.  Put the
        # monotonic discriminator first (so truncation cannot collide) and
        # bound the whole value before it becomes a map key.
        reaper_id = _bounded_shutdown_metadata(
            f"{self._next_shutdown_reaper_id}:{name}"
        )
        record = QuarantinedShutdownReaper(
            reaper_id=reaper_id,
            agent_name=_bounded_shutdown_metadata(name),
            # A full canonical user name is not retained in bounded failure
            # metadata.  Prefix collisions only over-reserve admission, which
            # is fail-closed until acknowledgement.
            canonical_agent_name=_bounded_shutdown_metadata(
                self._canonical_agent_name(name)
            ),
            agent_id=_bounded_shutdown_metadata(agent_id),
            task=task,
            started_monotonic=time.monotonic(),
            runtime_outcome_required=runtime_outcome_required,
        )
        self._quarantined_shutdown_reapers[reaper_id] = record

        def observe_reaper_completion(task: "asyncio.Future[object]") -> None:
            record.completed_monotonic = time.monotonic()
            if task.cancelled():
                record.failure = "shutdown reaper was cancelled"
            else:
                failure = task.exception()
                if failure is None and record.runtime_outcome_required:
                    failure = _runtime_offboarding_outcome_error(
                        agent_name=record.agent_name,
                        agent_id=record.agent_id,
                        result=task.result(),
                    )
                if failure is not None:
                    record.failure = _bounded_shutdown_metadata(
                        f"{public_exception_type_name(failure)}: {failure}"
                    )
            # Do not keep a completed Task: it retains coroutine locals and,
            # on failure, the full traceback. Operators still receive a
            # bounded history entry with the safety outcome.
            self._quarantined_shutdown_reapers.pop(record.reaper_id, None)
            history = QuarantinedShutdownHistory(
                reaper_id=record.reaper_id,
                agent_name=record.agent_name,
                canonical_agent_name=record.canonical_agent_name,
                agent_id=record.agent_id,
                started_monotonic=record.started_monotonic,
                completed_monotonic=record.completed_monotonic or time.monotonic(),
                failure=record.failure,
            )
            if history.failure is None:
                self._quarantined_shutdown_history.append(history)
            else:
                self._unsafe_quarantined_shutdown_failures[history.reaper_id] = history
                while (
                    len(self._unsafe_quarantined_shutdown_failures)
                    > _UNSAFE_QUARANTINED_FAILURE_LIMIT
                ):
                    self._unsafe_quarantined_shutdown_failures.pop(
                        next(iter(self._unsafe_quarantined_shutdown_failures))
                    )
                    # The individual record is intentionally discarded to
                    # preserve the hard metadata bound.  Keep one aggregate
                    # reservation instead of an unbounded set of evicted
                    # names; acknowledgement below is the explicit boundary
                    # for reopening admission.
                    self._unsafe_quarantined_shutdown_failure_overflow_reserved = (
                        True
                    )
                    self._unsafe_quarantined_shutdown_failure_evictions += 1
            if record.failure is None:
                logger.info(
                    "Quarantined shutdown reaper %s completed for agent %r",
                    record.reaper_id,
                    record.agent_name,
                )
            else:
                logger.error(
                    "Quarantined shutdown reaper %s remains unsafe for agent %r: %s",
                    record.reaper_id,
                    record.agent_name,
                    record.failure,
                )

        task.add_done_callback(observe_reaper_completion)
        return reaper_id

    def quarantined_shutdowns(self) -> dict[str, dict[str, object]]:
        """Return operational status for cleanup retained after removal.

        This is intentionally metadata-only: it exposes neither the agent's
        storage handle nor a foreign tenant's signal payload, while allowing a
        host operator to distinguish a still-draining reaper from a completed
        or failed one.
        """
        active = {
            reaper_id: {
                "agent_name": record.agent_name,
                "agent_id": record.agent_id,
                "pending": not record.task.done(),
                "started_monotonic": record.started_monotonic,
                "completed_monotonic": record.completed_monotonic,
                "failure": record.failure,
            }
            for reaper_id, record in self._quarantined_shutdown_reapers.items()
        }
        history = {
            record.reaper_id: {
                "agent_name": record.agent_name,
                "agent_id": record.agent_id,
                "pending": False,
                "started_monotonic": record.started_monotonic,
                "completed_monotonic": record.completed_monotonic,
                "failure": record.failure,
            }
            for record in self._quarantined_shutdown_history
        }
        unsafe_failures = {
            record.reaper_id: {
                "agent_name": record.agent_name,
                "agent_id": record.agent_id,
                "pending": False,
                "started_monotonic": record.started_monotonic,
                "completed_monotonic": record.completed_monotonic,
                "failure": record.failure,
            }
            for record in self._unsafe_quarantined_shutdown_failures.values()
        }
        return {**history, **unsafe_failures, **active}

    @property
    def unsafe_quarantined_shutdown_failure_eviction_count(self) -> int:
        """Return unsafe quarantine evictions not yet operator-acknowledged."""

        return (
            self._unsafe_quarantined_shutdown_failure_evictions
            - self._unsafe_quarantined_shutdown_failure_evictions_acknowledged_through
        )

    def acknowledge_unsafe_quarantined_shutdown_failure_evictions(self) -> int:
        """Acknowledge the aggregate evidence for evicted unsafe reapers.

        Individual records remain bounded and require their own acknowledgement.
        An eviction cannot retain its exact record, so this acknowledgement
        clears the single fail-closed overflow reservation for every eviction
        observed through this checkpoint.  Later evictions reserve admission
        again until this method is called again.
        """

        unacknowledged = self.unsafe_quarantined_shutdown_failure_eviction_count
        self._unsafe_quarantined_shutdown_failure_evictions_acknowledged_through = (
            self._unsafe_quarantined_shutdown_failure_evictions
        )
        # Individual evicted records are intentionally not retained.  This
        # explicit aggregate acknowledgement is therefore the remediation
        # boundary for the conservative overflow reservation.
        self._unsafe_quarantined_shutdown_failure_overflow_reserved = False
        return unacknowledged

    def acknowledge_unsafe_quarantined_shutdown_failure(self, reaper_id: str) -> bool:
        """Remove an operator-acknowledged unsafe outcome from bounded history."""

        return self._unsafe_quarantined_shutdown_failures.pop(reaper_id, None) is not None

    def unsafe_removal_budget_release_failures(self) -> dict[str, dict[str, object]]:
        """Return bounded evidence for ordinary refunds that failed after removal.

        A successful ordinary release has no post-removal state to expose. A
        failed one is different: routing may already be withdrawn, so dropping
        the completed task would let a terminal drain falsely certify the
        manager. Keep metadata only until the host operator explicitly
        acknowledges it; the finished task and its traceback remain unretained.
        """

        return {
            record.release_id: {
                "child_name": record.child_name,
                "started_monotonic": record.started_monotonic,
                "completed_monotonic": record.completed_monotonic,
                "failure": record.failure,
            }
            for record in self._unsafe_removal_budget_release_failures.values()
        }

    @property
    def unsafe_removal_budget_release_failure_eviction_count(self) -> int:
        """Return unsafe ordinary-release evictions not yet acknowledged."""

        return (
            self._unsafe_removal_budget_release_failure_evictions
            - self._unsafe_removal_budget_release_failure_evictions_acknowledged_through
        )

    def acknowledge_unsafe_removal_budget_release_failure_evictions(self) -> int:
        """Acknowledge aggregate evidence for evicted ordinary-release failures."""

        unacknowledged = self.unsafe_removal_budget_release_failure_eviction_count
        self._unsafe_removal_budget_release_failure_evictions_acknowledged_through = (
            self._unsafe_removal_budget_release_failure_evictions
        )
        return unacknowledged

    def acknowledge_unsafe_removal_budget_release_failure(self, release_id: str) -> bool:
        """Remove an operator-acknowledged ordinary-release failure record."""

        return (
            self._unsafe_removal_budget_release_failures.pop(release_id, None)
            is not None
        )

    def _removal_budget_release_ids_for_child(self, child_name: str) -> set[str]:
        """Snapshot live or retained ordinary-release evidence for one child.

        This is used only within one ``shutdown_all`` sweep to avoid reporting
        a newly admitted release first through ``attempt_removal`` and then
        again through its retained unsafe metadata.  It intentionally does not
        acknowledge anything: a later drain must still surface the failure.
        """

        release_ids = {
            release_id
            for release_id, record in self._unsafe_removal_budget_release_failures.items()
            if record.child_name == _bounded_shutdown_metadata(child_name)
        }
        active = self._inflight_removal_budget_releases_by_child.get(child_name)
        if active is not None:
            release_ids.add(active.release_id)
        return release_ids

    async def _set_quarantined_shutdown_handoffs_sealed(self, sealed: bool) -> bool:
        """Set the temporary reaper-handoff seal at a drain boundary.

        Quarantined handoffs and ordinary budget-release tasks are both
        admitted while they hold ``_lock``. Waiting for that lock therefore
        makes setting the seal the exact linearization point: cleanup either
        completed admission before the seal and is drained below, or starts
        while sealed and takes the conservative no-removal path. Keep the
        acquisition cancellation-safe so a terminal owner cannot be cancelled
        in the small gap before it owns the state transition.
        """

        acquire = asyncio.create_task(
            self._lock.acquire(),
            name="agent_manager:set_quarantined_shutdown_handoff_seal",
        )
        cancelled, failure = await await_lifecycle_task_completion(acquire)
        if failure is not None:
            raise RuntimeError(
                "Unable to update quarantined shutdown handoff seal"
            ) from failure
        try:
            self._quarantined_shutdown_handoffs_sealed = sealed
            if sealed:
                self._quarantined_shutdown_handoffs_open.clear()
            else:
                self._quarantined_shutdown_handoffs_open.set()
        finally:
            self._lock.release()
        return cancelled

    async def _acquire_quarantined_shutdown_drain(self) -> bool:
        """Serialize terminal drains despite caller cancellation."""

        acquire = asyncio.create_task(
            self._quarantined_shutdown_drain_lock.acquire(),
            name="agent_manager:drain_quarantined_shutdowns",
        )
        cancelled, failure = await await_lifecycle_task_completion(acquire)
        if failure is not None:
            raise RuntimeError(
                "Unable to acquire quarantined shutdown drain"
            ) from failure
        return cancelled

    async def drain_quarantined_shutdowns(
        self,
        *,
        reported_budget_release_failures: Optional[set[str]] = None,
    ) -> bool:
        """Join every active quarantined shutdown before retiring this manager.

        ``remove_agent`` deliberately remains a bounded control-plane operation:
        it can unpublish an agent after handing its exact durable cleanup task to
        quarantine.  A caller that is itself about to retire the *manager* has a
        different responsibility.  It must join those retained tasks before the
        surrounding server can close the event loop or any shared resource they
        still use.  Otherwise a late owner-release transaction can run against
        storage that the next teardown phase has already closed.

        Returns whether this join observed caller cancellation.  Like the other
        lifecycle joins, it still drains every retained reaper first so
        cancellation cannot orphan a SQLite worker or durable runtime owner.
        """

        # This public operation is a terminal lifecycle boundary, not a
        # best-effort status poll.  Serialize the drain and seal it first so a
        # concurrent bounded ``remove_agent`` cannot hand work to a registry
        # this drain has already declared empty.  The seal is released after
        # every attempt: a failure that leaves an agent or a delegated hold
        # retained must remain retryable by a later startup/server shutdown.
        cancelled = await self._acquire_quarantined_shutdown_drain()
        try:
            self._seal_agent_registration_for_shutdown_all()
            spawn_cancelled, spawn_failures = (
                await self._join_admitted_spawn_operations()
            )
            cancelled = cancelled or spawn_cancelled
            result = await self._drain_quarantined_shutdowns_while_locked(
                cancelled=cancelled,
                reported_budget_release_failures=reported_budget_release_failures,
            )
            _raise_lifecycle_outcomes(
                "One or more admitted spawns had terminal failures",
                spawn_failures,
            )
            return result
        finally:
            self._reopen_agent_registration_after_shutdown_all()
            self._quarantined_shutdown_drain_lock.release()

    async def _drain_quarantined_shutdowns_while_locked(
        self,
        *,
        cancelled: bool,
        reported_budget_release_failures: Optional[set[str]],
    ) -> bool:
        """Drain terminal cleanup while the caller owns the drain lock."""

        failures: list[Exception] = []
        observed_reapers: set[str] = set()
        observed_runtime_offboardings: set[int] = set()
        observed_cleanup_tasks: set[int] = set()
        observed_budget_releases: set[str] = set()
        # ``shutdown_all`` may have already joined and reported a release
        # before this terminal drain gets to its retained unsafe metadata.
        # That evidence must remain for a later drain, but one fleet-shutdown
        # ExceptionGroup must name the release only once.
        reported_budget_release_failures = (
            set()
            if reported_budget_release_failures is None
            else reported_budget_release_failures
        )

        def sanitized_terminal_failure(
            *, owner: str, owner_id: str, failure: BaseException
        ) -> RuntimeError:
            """Describe terminal cleanup without reflecting exception text."""

            failure_type = public_exception_type_name(failure)
            return RuntimeError(
                f"{owner} {owner_id!r} failed ({failure_type}); inspect "
                "server-side lifecycle diagnostics"
            )

        def record_reaper_failure(
            reaper_id: str, failure: BaseException
        ) -> None:
            failures.append(
                sanitized_terminal_failure(
                    owner="Quarantined shutdown reaper",
                    owner_id=reaper_id,
                    failure=failure,
                )
            )

        def record_budget_release_failure(
            release_id: str, failure: BaseException
        ) -> None:
            failures.append(
                sanitized_terminal_failure(
                    owner="Ordinary child budget release",
                    owner_id=release_id,
                    failure=failure,
                )
            )

        def record_retained_failure(*, owner: str, owner_id: str) -> None:
            """Report bounded retained evidence without inventing its type."""

            failures.append(
                RuntimeError(
                    f"{owner} {owner_id!r} retained an unacknowledged cleanup "
                    "failure; inspect server-side lifecycle diagnostics"
                )
            )

        sealed = False
        try:
            cancelled = (
                await self._set_quarantined_shutdown_handoffs_sealed(True)
            ) or cancelled
            sealed = True

            while (
                self._quarantined_shutdown_reapers
                or self._inflight_runtime_offboardings
                or self._inflight_removal_budget_releases
            ):
                records = tuple(self._quarantined_shutdown_reapers.values())
                for record in records:
                    # Completion callbacks remove their record after collapsing it
                    # to bounded history.  A snapshot can therefore include a task
                    # another callback has already retired; it is still the exact
                    # task this manager must observe once.
                    if record.reaper_id in observed_reapers:
                        continue
                    observed_reapers.add(record.reaper_id)
                    observed_cleanup_tasks.add(id(record.task))
                    join_cancelled, failure = await await_lifecycle_task_completion(
                        record.task
                    )
                    cancelled = cancelled or join_cancelled
                    classified_outcome = False
                    if failure is None and record.runtime_outcome_required:
                        failure = _runtime_offboarding_outcome_error(
                            agent_name=record.agent_name,
                            agent_id=record.agent_id,
                            result=record.task.result(),
                        )
                        classified_outcome = failure is not None
                    if failure is not None:
                        if classified_outcome and isinstance(failure, Exception):
                            failures.append(failure)
                        else:
                            record_reaper_failure(record.reaper_id, failure)

                runtime_offboardings = tuple(
                    self._inflight_runtime_offboardings.items()
                )
                for record_key, runtime_offboarding in runtime_offboardings:
                    if record_key in observed_runtime_offboardings:
                        continue
                    observed_runtime_offboardings.add(record_key)
                    if id(runtime_offboarding.task) in observed_cleanup_tasks:
                        continue
                    observed_cleanup_tasks.add(id(runtime_offboarding.task))
                    (
                        join_cancelled,
                        offboarding_failure,
                    ) = await await_lifecycle_task_completion(runtime_offboarding.task)
                    cancelled = cancelled or join_cancelled
                    classified_outcome = False
                    if offboarding_failure is None:
                        offboarding_failure = _runtime_offboarding_outcome_error(
                            agent_name=runtime_offboarding.agent_name,
                            agent_id=runtime_offboarding.agent_id,
                            result=runtime_offboarding.task.result(),
                        )
                        classified_outcome = offboarding_failure is not None
                    if offboarding_failure is not None:
                        if classified_outcome and isinstance(
                            offboarding_failure, Exception
                        ):
                            failures.append(offboarding_failure)
                        else:
                            failures.append(
                                sanitized_terminal_failure(
                                    owner="Secure runtime offboarding",
                                    owner_id=runtime_offboarding.agent_name,
                                    failure=offboarding_failure,
                                )
                            )

                # A normal DELETE waits for this exact task itself.  It is
                # nevertheless removal-owned as soon as it is admitted, so a
                # concurrent terminal owner must join it after unpublishing
                # has made both the routing and delegated-hold maps empty.
                # These tasks are deliberately separate from quarantine
                # metadata: a successful ordinary release is not a deferred
                # cleanup state that operators need to inspect.
                budget_releases = tuple(self._inflight_removal_budget_releases.values())
                for budget_release in budget_releases:
                    if budget_release.release_id in observed_budget_releases:
                        continue
                    observed_budget_releases.add(budget_release.release_id)
                    (
                        join_cancelled,
                        release_failure,
                    ) = await await_lifecycle_task_completion(budget_release.task)
                    cancelled = cancelled or join_cancelled
                    if (
                        release_failure is not None
                        and budget_release.release_id
                        not in reported_budget_release_failures
                    ):
                        record_budget_release_failure(
                            budget_release.release_id,
                            release_failure,
                        )

                # A terminal task's done callback is responsible for dropping its
                # live task reference.  Yield once so a task which completed before
                # this join started cannot keep the loop spinning on stale active
                # metadata.  Use the same cancellation-safe join contract here:
                # a second cancellation must not interrupt this manager between
                # observing a reaper and its callback retiring the live task.
                callback_yield = asyncio.create_task(asyncio.sleep(0))
                yield_cancelled, yield_failure = await await_lifecycle_task_completion(
                    callback_yield
                )
                cancelled = cancelled or yield_cancelled
                if yield_failure is not None:
                    raise RuntimeError(
                        "Unable to retire completed quarantined shutdown metadata"
                    ) from yield_failure

            # A task can finish (and its callback can collapse it to unsafe
            # metadata) before this drain gets its first registry snapshot.  That
            # outcome is still unresolved terminal cleanup, so surface every
            # unacknowledged failure exactly once alongside failures observed live.
            for reaper_id, record in self._unsafe_quarantined_shutdown_failures.items():
                if reaper_id not in observed_reapers:
                    record_retained_failure(
                        owner="Quarantined shutdown reaper",
                        owner_id=reaper_id,
                    )

            # An ordinary release may fail before this drain acquires the
            # manager lock. Its completion callback then drops the live task,
            # so retained bounded evidence is the only way to prevent the
            # terminal boundary from falsely succeeding.
            for (
                release_id,
                record,
            ) in self._unsafe_removal_budget_release_failures.items():
                if (
                    release_id not in observed_budget_releases
                    and release_id not in reported_budget_release_failures
                ):
                    record_retained_failure(
                        owner="Ordinary child budget release",
                        owner_id=release_id,
                    )

            # A direct ``terminate_child`` retains its parent edge while a
            # bounded DELETE has handed cleanup to quarantine.  The terminal
            # drain has sealed new handoffs and joined every retained task, so
            # reconcile only now: a failed refund has restored its hold and an
            # unsafe reaper still reserves the name, while a successful refund
            # can finally release the mandate and spawn-cap edge.
            reconciliation = asyncio.create_task(
                self._prune_all_fully_removed_child_tracking(),
                name="agent_manager:reconcile_completed_child_terminations",
            )
            reconciliation_cancelled, reconciliation_failure = (
                await await_lifecycle_task_completion(reconciliation)
            )
            cancelled = cancelled or reconciliation_cancelled
            if reconciliation_failure is not None:
                raise RuntimeError(
                    "Unable to reconcile completed child terminations"
                ) from reconciliation_failure

            # Eviction never acknowledges an unsafe result.  The retained
            # metadata is intentionally bounded, so preserve the loss as an
            # aggregate terminal failure rather than letting 129+ failures be
            # hidden after every surviving record is acknowledged.
            evictions = self.unsafe_quarantined_shutdown_failure_eviction_count
            if evictions:
                failures.append(
                    RuntimeError(
                        f"{evictions} unsafe quarantined shutdown failure "
                        "record(s) were evicted before acknowledgement"
                    )
                )

            budget_release_evictions = (
                self.unsafe_removal_budget_release_failure_eviction_count
            )
            if budget_release_evictions:
                failures.append(
                    RuntimeError(
                        f"{budget_release_evictions} unsafe ordinary budget "
                        "release failure record(s) were evicted before acknowledgement"
                    )
                )
        finally:
            if sealed:
                cancelled = (
                    await self._set_quarantined_shutdown_handoffs_sealed(False)
                ) or cancelled

        # Preserve both cancellation and every sanitized cleanup failure after
        # all owned work has settled. A BaseExceptionGroup is required because
        # asyncio cancellation is a BaseException; putting it in an
        # ExceptionGroup raises TypeError and loses the failures already
        # collected. A successful join retains the established ``True`` return
        # value so callers can choose when to re-raise cancellation themselves.
        if cancelled and failures:
            raise BaseExceptionGroup(
                "Quarantined shutdown drain observed cancellation and failures",
                [asyncio.CancelledError(), *failures],
            )
        if failures:
            raise ExceptionGroup(
                "One or more quarantined shutdown reapers or ordinary budget "
                "releases failed",
                failures,
            )
        return cancelled

    async def _remove_agent_without_scheduler_lifecycle(
        self,
        name: str,
        *,
        offboard_runtime: bool,
        pending_offboarding: list[InflightRuntimeOffboarding],
        offboarding_admission: Optional[RuntimeOffboardingAdmission],
    ) -> bool:
        """Shutdown and remove an agent with an explicit state-retention policy.

        Returns:
            True if agent was found and removed.
        """
        # A manager that is inside its terminal drain cannot safely accept a
        # new bounded handoff. Refuse promptly rather than extending a
        # single-agent DELETE into an unbounded join or letting the current
        # drain return without observing the new reaper.
        async with self._lock:
            if self._quarantined_shutdown_handoffs_sealed:
                logger.warning(
                    "Refusing removal of agent %r after terminal manager shutdown "
                    "handoffs were sealed",
                    name,
                )
                return False

        # Refuse to remove an agent that still has budgeted descendants (#2113):
        # remove_agent is a single-agent primitive, so releasing this agent's hold
        # while a budgeted grandchild still holds from its (about-to-be-removed)
        # wallet would strand the grandchild's unspent budget on its later
        # release. Such teardown must go through terminate_child, which cascades
        # and releases nested budgets leaf-first. terminate_child/shutdown_all
        # remove descendants BEFORE the parent, so they never trip this.
        if self._has_budgeted_descendants(name):
            raise ValueError(
                f"Cannot remove '{name}' directly: it has budgeted child agents. "
                f"Use terminate_child, which cascades and releases nested budgets "
                f"leaf-first (#2113)."
            )

        # A spawn can reserve a delegated hold before its agent is published.
        # There is no process to stop in that state, but a DELETE/shutdown must
        # still refund the hold.  Treat it as a completed removal rather than
        # silently retaining money because there is no routing entry.
        ordinary_budget_release: Optional[InflightRemovalBudgetRelease] = None
        handoff_offboarding_pending: Optional[
            RuntimeOffboardingRetainedError
        ] = None
        unpublished_offboarding_failure: Optional[
            RuntimeOffboardingRetainedError
        ] = None
        async with self._lock:
            if self._quarantined_shutdown_handoffs_sealed:
                logger.warning(
                    "Refusing removal of agent %r after terminal manager shutdown "
                    "handoffs were sealed",
                    name,
                )
                return False
            unpublished_hold = name not in self._agents and name in self._child_budgets
            if unpublished_hold:
                unpublished_agent_id = self._budgeted_child_agent_id(name)
                if offboard_runtime:
                    unpublished_config = self._created_configs.get(name)
                    hosted_factory = self._hosted_agent_runtime_factory_configured(
                        os.environ.get("KESTREL_DB_BACKEND", "sqlite"),
                        os.environ.get("KESTREL_DATABASE_URL"),
                    )
                    if unpublished_agent_id is None or (
                        hosted_factory and unpublished_config is None
                    ):
                        unpublished_offboarding_failure = (
                            RuntimeOffboardingRetainedError(
                                agent_name=name,
                                agent_id="<unknown>",
                                runtime_path=None,
                                cause=RuntimeError(
                                    "unpublished child identity is unavailable"
                                ),
                            )
                        )
                    else:
                        pending_offboarding.append(
                            self._start_agent_runtime_offboarding_identity(
                                name=name,
                                agent_id=unpublished_agent_id,
                                config=unpublished_config,
                            )
                        )
                ordinary_budget_release = self._start_child_budget_release(name)
        if unpublished_hold:
            assert ordinary_budget_release is not None
            try:
                release_cancelled = (
                    await self._await_child_budget_release_cancellation_safe(
                        ordinary_budget_release.task
                    )
                )
            except asyncio.CancelledError:
                # The admitted release has reached a terminal state before its
                # cancellation-safe join propagates. Reconcile the now-settled
                # ownership maps before reporting cancellation to DELETE.
                release_cancelled = True
            reconciliation_cancelled = (
                await self._reconcile_fully_removed_child_tracking()
            )
            if release_cancelled or reconciliation_cancelled:
                raise asyncio.CancelledError()
            if unpublished_offboarding_failure is not None:
                raise unpublished_offboarding_failure
            return True

        async with self._lock:
            if self._quarantined_shutdown_handoffs_sealed:
                logger.warning(
                    "Refusing removal of agent %r after terminal manager shutdown "
                    "handoffs were sealed",
                    name,
                )
                return False
            agent = self._agents.get(name)
            if agent is None:
                return False

            # ``wait_for(agent.shutdown())`` loses the actual task on timeout
            # and on cancellation.  KestrelAgent deliberately runs its
            # cancellation tail before re-raising, so that task is the primary
            # evidence that cleanup is terminal even when no deferred
            # continuation was necessary.
            shutdown_task = asyncio.create_task(
                agent.shutdown(), name=f"agent_shutdown:{name}"
            )
            caller_cancelled = False
            shutdown_timed_out = False
            shutdown_handed_off = False
            budget_handed_off = False
            ordinary_budget_release = None
            try:
                await asyncio.wait_for(
                    asyncio.shield(shutdown_task), timeout=SHUTDOWN_TIMEOUT
                )
            except asyncio.CancelledError:
                caller_cancelled = asyncio.current_task().cancelling() > 0
                if not shutdown_task.done():
                    shutdown_task.cancel()
                shutdown_handed_off = self._handoff_shutdown_to_quarantined_reaper(
                    name=name,
                    agent=agent,
                    shutdown_task=shutdown_task,
                    offboard_runtime=offboard_runtime,
                )
                if (
                    shutdown_handed_off
                    and offboard_runtime
                    and offboarding_admission is not None
                ):
                    offboarding_admission.started = True
                if shutdown_handed_off and offboard_runtime:
                    handoff_offboarding_pending = RuntimeOffboardingRetainedError(
                        agent_name=name,
                        agent_id=_loaded_agent_did(agent) or "<unknown>",
                        runtime_path=_agent_runtime_path(agent),
                        cause=RuntimeError(
                            "runtime offboarding is owned by quarantined shutdown"
                        ),
                        cleanup_pending=True,
                    )
                if shutdown_handed_off:
                    logger.warning(
                        "Agent '%s' shutdown was cancelled; durable cleanup is "
                        "quarantined while control-plane removal continues.",
                        name,
                    )
                else:
                    logger.warning(
                        "Agent '%s' shutdown was cancelled; joining its actual "
                        "shutdown task and durable cleanup before unpublishing.",
                        name,
                    )
            except asyncio.TimeoutError:
                shutdown_timed_out = True
                shutdown_task.cancel()
                shutdown_handed_off = self._handoff_shutdown_to_quarantined_reaper(
                    name=name,
                    agent=agent,
                    shutdown_task=shutdown_task,
                    offboard_runtime=offboard_runtime,
                )
                if (
                    shutdown_handed_off
                    and offboard_runtime
                    and offboarding_admission is not None
                ):
                    offboarding_admission.started = True
                if shutdown_handed_off and offboard_runtime:
                    handoff_offboarding_pending = RuntimeOffboardingRetainedError(
                        agent_name=name,
                        agent_id=_loaded_agent_did(agent) or "<unknown>",
                        runtime_path=_agent_runtime_path(agent),
                        cause=TimeoutError(
                            "runtime offboarding awaits quarantined shutdown"
                        ),
                        cleanup_pending=True,
                    )
                if shutdown_handed_off:
                    logger.warning(
                        "Agent '%s' exceeded its shutdown bound; durable cleanup "
                        "is quarantined while control-plane removal continues.",
                        name,
                    )
                else:
                    logger.warning(
                        "Agent '%s' shutdown timed out; joining its actual "
                        "shutdown task before deciding removal.",
                        name,
                    )
            except Exception as exc:
                # A normal shutdown error has no general proof that the agent
                # reached its cancellation tail.  Keep the historical safe
                # behaviour for that case; timeouts/cancellation are handled
                # below by joining the task that did run the tail.
                logger.warning(
                    "Agent '%s' shutdown failed; retaining it until cleanup "
                    "can be confirmed: %s",
                    name,
                    exc,
                    exc_info=True,
                )
                return False

            if not shutdown_handed_off:
                (
                    join_cancelled,
                    shutdown_failure,
                ) = await await_lifecycle_task_completion(shutdown_task)
                caller_cancelled = caller_cancelled or join_cancelled
                if shutdown_failure is not None and not isinstance(
                    shutdown_failure, asyncio.CancelledError
                ):
                    logger.warning(
                        "Agent '%s' shutdown task failed; retaining it until "
                        "cleanup can be confirmed: %s",
                        name,
                        shutdown_failure,
                        exc_info=(
                            type(shutdown_failure),
                            shutdown_failure,
                            shutdown_failure.__traceback__,
                        ),
                    )
                    return False
                if shutdown_timed_out and shutdown_failure is None:
                    logger.warning(
                        "Agent '%s' exceeded its shutdown budget but completed "
                        "while being joined; continuing durable cleanup.",
                        name,
                    )

                # A tail that spent its dispatcher guard continues in the
                # agent-owned completion task.  Do not unpublish the agent or
                # release its delegated budget until that task has released the
                # owner and closed storage. Legacy lifecycle implementations
                # remain on this conservative path; KestrelAgent's explicit
                # handoff contract above is what permits bounded removal.
                if _has_shutdown_completion_contract(agent):
                    caller_cancelled = (
                        await await_agent_shutdown_completion(agent)
                    ) or caller_cancelled

                if offboard_runtime:
                    # Admit the destructive worker while the manager lock still
                    # linearizes this removal, but never await filesystem I/O
                    # here. The public wrapper joins it only after scheduler,
                    # A2A, and manager lifecycle locks have all been released.
                    pending_offboarding.append(
                        self._start_agent_runtime_offboarding(
                            name=name,
                            agent=agent,
                        )
                    )

            # Successful shutdown always withdraws routing. Requested
            # offboarding may have completed, failed with retained state, or
            # been handed to an explicit reaper; none of those states permits
            # republishing a process whose shutdown already completed.
            self._agents.pop(name, None)
            self._agent_names.pop(agent.agent_id, None)
            self._non_governing_spawn_lineage.difference_update(
                _loaded_agent_bound_dids(agent)
            )
            self._revoke_a2a_hosted_policy(agent)
            if vars(agent).get("_agent_manager") is self:
                agent._agent_manager = None
            if shutdown_handed_off:
                # Keep both quarantine handoffs under the same lock that
                # linearizes terminal sealing.  This synchronous fence is
                # still immediately after routing withdrawal, as before.
                budget_handed_off = (
                    self._handoff_child_budget_release_to_quarantined_reaper(
                        name, agent_id=_loaded_agent_did(agent) or "<unknown>"
                    )
                )
            if not budget_handed_off:
                # Admission is deliberately before we release ``_lock``. A
                # terminal drain that follows routing withdrawal can then
                # either observe this exact ordinary release or seal before a
                # removal begins; it can never see both routing and the hold
                # gone while this task is still unowned.
                ordinary_budget_release = self._start_child_budget_release(name)
            logger.info(f"Agent '{name}' shut down")

        # Release THIS agent's own budget hold AFTER it is stopped (#2113):
        # releasing before shutdown would let a still-running child spend
        # already-refunded funds. remove_agent is a SINGLE-AGENT primitive — it
        # does not cascade. Budgeted subtrees are torn down via terminate_child
        # (cascade) or shutdown_all (leaf-first), which release nested holds in
        # the correct order; directly remove_agent-ing a budgeted PARENT is not a
        # supported budget teardown (folded into #2348 with reload durability).
        # Idempotent — a no-op when those paths already released this entry.
        # The routing entry is gone, so this release is now the last mutation
        # required for removal.  Retain its task through repeated cancellation
        # before propagating cancellation to the caller; otherwise a closed
        # child can strand its delegated hold.
        # The shutdown reaper proves the agent's durable owner/storage are
        # retained, but a still-running cognition can also be blocked inside
        # DelegatedWallet.spend(). In that case the budget handoff above keeps
        # DELETE bounded. Otherwise this is the same ordinary stop-then-
        # release task DELETE has always joined, now admitted early enough for
        # a terminal manager drain to join it too.
        release_cancelled = False
        release_failure: Optional[Exception] = None
        if ordinary_budget_release is not None:
            try:
                release_cancelled = (
                    await self._await_child_budget_release_cancellation_safe(
                        ordinary_budget_release.task
                    )
                )
            except asyncio.CancelledError:
                # The release join is terminal even when it must propagate a
                # caller cancellation. Finish child-name reconciliation first.
                release_cancelled = True
            except Exception as exc:
                # The task completion callback has already retained bounded
                # unsafe-refund evidence. Continue reconciliation so a runtime
                # offboarding failure cannot be hidden by this independent
                # budget failure, then surface both below.
                release_failure = exc
        # Direct DELETE is also a supported child-removal path. Reconcile its
        # parent edge before releasing the same A2A lifecycle boundary used by
        # name admission. Quarantined cleanup remains reserved because the
        # pruning predicate treats its live reaper as authoritative ownership.
        reconciliation_cancelled = await self._reconcile_fully_removed_child_tracking()
        removal_cancelled = (
            caller_cancelled or release_cancelled or reconciliation_cancelled
        )
        terminal_outcomes: list[BaseException] = []
        if removal_cancelled:
            terminal_outcomes.append(asyncio.CancelledError())
        if release_failure is not None:
            terminal_outcomes.append(release_failure)
        if handoff_offboarding_pending is not None:
            terminal_outcomes.append(handoff_offboarding_pending)
        _raise_lifecycle_outcomes(
            f"Agent {name!r} removal had terminal budget outcomes",
            terminal_outcomes,
        )
        return True

    def _allocate_port(self) -> int:
        """Pick the first FREE in-range agent port and reserve it.

        Reservations cover configured agents, the host itself, and every
        prior runtime allocation — and are never released (an unloaded
        agent's port stays reserved, the #1729 guarantee). Scanning instead
        of incrementing means a high host port can't starve the range.
        """
        candidate = self._port_scan_start
        while candidate in self._reserved_ports:
            candidate += 1
            if candidate > 65535:
                raise ValueError("No free agent ports remain in 8801-65535.")
        self._reserved_ports.add(candidate)
        return candidate

    def _data_key_custody_conflict(self) -> Optional[str]:
        """Return a custody-conflict message if the process ``KESTREL_DATA_KEY``
        disagrees with the one persisted in the resolved home ``.env`` (#2468).

        Reuses the setup path's resolver so runtime inception enforces the same
        contract: encrypting a new identity with a key the home ``.env`` does not
        persist would brick that agent on the next boot. Returns ``None`` when
        custody is coherent (matching keys, only-persisted, or only-exported).
        Best-effort: never raises — a resolver/import failure must not block the
        running fleet, only a *positive* conflict does.
        """
        try:
            from kestrel_sovereign.setup.steps.keys import (
                DATA_KEY_ENV,
                read_persisted_data_key,
                resolve_data_key_authority,
            )

            # Read the *same* ``.env`` whose ``KESTREL_DATA_KEY`` actually
            # seeded ``os.environ`` at boot. ``server.py`` loads the resolved
            # project home first (``override=False`` → wins), then CWD only to
            # fill gaps. ``_base_data_dir`` is hard-wired to ``Path.cwd()``, so
            # comparing against it would diverge whenever ``kestrel start`` is
            # launched from a source checkout under an explicit ``KESTREL_HOME``
            # — producing a false conflict (or missing a real one) (#2468).
            try:
                from kestrel_sovereign.paths import project_dir as _resolve_project_dir

                env_path = _resolve_project_dir() / ".env"
            except Exception:
                env_path = self._base_data_dir / ".env"
            persisted = read_persisted_data_key(env_path)
            exported = os.environ.get(DATA_KEY_ENV)
            _, conflict = resolve_data_key_authority(
                persisted, exported, env_name=env_path.name
            )
            return conflict
        except Exception:
            return None

    async def create_agent(
        self,
        name: str,
        parent_did: str = None,
        features: Optional[List[str]] = None,
        mandate: Optional[SpawnMandate] = None,
    ) -> KestrelAgent:
        """Create a new agent via inception and load it.

        Runs the inception service to generate a new DID and database,
        then loads the agent into the manager.

        Args:
            name: Name for the new agent (used as directory name and routing key).
            parent_did: Optional DID of parent agent for delegation chain.
            features: Optional allowlist of feature class names the agent may
                load. ``None`` loads all discovered features (backward
                compatible); a list restricts loading to those class names
                (mandatory features are always loaded regardless). Threaded
                into the agent's ``LocalAgentConfig`` so the restriction
                actually reaches ``load_agent`` / ``discover_features`` (#1946).
            mandate: Optional SpawnMandate authorizing a spawned child. It is
                accepted only inside the manager-owned ``spawn_agent``
                operation, which signs and promotes the final child-DID receipt
                before publication. Direct callers must use ``spawn_agent``.

        Returns:
            The newly created and initialized KestrelAgent.

        Raises:
            ValueError: If an agent with this name already exists or inception fails.
        """
        admission, owns_admission = await self._admit_agent_operation(
            name, kind="create"
        )
        try:
            if mandate is not None and admission.kind not in {
                "spawn",
                "direct-spawn-test",
            }:
                raise ValueError(
                    "Direct create_agent cannot accept a spawn mandate; use "
                    "spawn_agent so the parent signs the final child-DID receipt"
                )
            # Custody guard (#2468): runtime inception (POST /api/agents,
            # spawned children) reaches ``create_kestrel_identity_async``
            # without the setup resolver, so verify — through the *same*
            # resolver setup uses — that the KESTREL_DATA_KEY this process
            # would encrypt with matches the one persisted in the resolved home
            # ``.env``. Refuse a split brain before any identity is written.
            custody_conflict = self._data_key_custody_conflict()
            if custody_conflict:
                raise ValueError(custody_conflict)

            # Allocate the port BEFORE inception: failing allocation afterward
            # would leave an orphaned identity directory.
            port = self._allocate_port()

            config = LocalAgentConfig(
                data_dir=Path("agent_data") / name,
                port=port,
                autostart=True,
                features=features,
            )
            admission.spawn_candidate_config = config.model_copy(deep=True)
            agent_dir = self._base_data_dir / "agent_data" / name

            # A child database becomes eligible for config-less auto-discovery
            # as soon as inception writes it, before the child's final DID is
            # available for signing. Persist a slot-level restart denial first;
            # the prepublication callback atomically promotes it to the final
            # signed witness. Cancellation deliberately retains this denial
            # because inception may already have committed durable identity.
            if mandate is not None:
                # A valid cold identity in this slot predates this admission.
                # Reject it before reserving capacity: an inception
                # FileExistsError cannot distinguish that old tenant from a
                # partial birth by this attempt after the fact.
                identity_db = agent_dir / "kestrel_prime.db"
                if _identity_anchor_birth_status(identity_db) is True:
                    self._reserved_ports.discard(port)
                    raise ValueError(
                        f"Agent data slot for {name!r} already contains an "
                        "agent identity"
                    )
                try:
                    pending = self._spawn_authority_registry.reserve_pending(
                        child_name=name,
                        parent_did=mandate.parent_did,
                        mandate=mandate,
                        config=config,
                        max_authority_slots=self._max_spawned_agents,
                    )
                except Exception:
                    self._reserved_ports.discard(port)
                    raise
                admission.spawn_authority_pending_id = pending.reservation_id

            try:
                from kestrel_sovereign.inception_service import (
                    create_kestrel_identity_async,
                )

                agent_dir.mkdir(parents=True, exist_ok=True)
            except BaseException as pre_inception_failure:
                cleanup_failure: BaseException | None = None
                pending_id = admission.spawn_authority_pending_id
                if pending_id is not None:
                    try:
                        self._spawn_authority_registry.withdraw_pending(
                            reservation_id=pending_id,
                            child_name=name,
                        )
                        admission.spawn_authority_pending_id = None
                    except BaseException as error:
                        cleanup_failure = error
                self._reserved_ports.discard(port)
                if cleanup_failure is not None:
                    _raise_lifecycle_outcomes(
                        "Pre-inception setup and pending authority rollback failed",
                        [pre_inception_failure, cleanup_failure],
                    )
                raise

            try:
                await create_kestrel_identity_async(
                    output_dir=str(agent_dir),
                    agent_name=name,
                    parent_did=parent_did,
                    spawn_mandate=mandate,
                )
            except BaseException as error:
                # Inception removes the SQLite file when its guarded mint phases
                # fail cleanly. Release the pending denial only with that
                # positive evidence. Later failures can occur after the atomic
                # identity graph commit, so a remaining database is ambiguous
                # and stays denied just like cancellation/process death.
                pending_id = admission.spawn_authority_pending_id
                identity_db = agent_dir / "kestrel_prime.db"
                identity_birth_absent = False
                try:
                    identity_db.stat()
                except FileNotFoundError:
                    identity_birth_absent = True
                except OSError as inspection_error:
                    # ``Path.exists()`` would collapse this uncertainty into
                    # absence and withdraw the only restart denial.  Retain the
                    # reservation until recovery can prove that no DB exists.
                    logger.warning(
                        "Could not prove failed inception for %r stopped before "
                        "identity birth; retaining pending authority: %s",
                        name,
                        inspection_error,
                    )
                if pending_id is not None and identity_birth_absent:
                    self._spawn_authority_registry.withdraw_pending(
                        reservation_id=pending_id,
                        child_name=name,
                    )
                    admission.spawn_authority_pending_id = None
                elif pending_id is not None:
                    # The denial remains because durable birth is ambiguous,
                    # but this producer has reached a terminal outcome.  Do not
                    # advertise it as live until process exit: reconciliation
                    # in this same host may now claim the liveness lock and reap
                    # an ownerless, pre-birth SQLite shell.
                    self._spawn_authority_registry.release_pending_producer(
                        reservation_id=pending_id,
                    )
                self._reserved_ports.discard(port)
                if not isinstance(error, Exception):
                    raise
                raise ValueError(f"Inception failed for '{name}': {error}")
            agent = await self.load_agent(name, config)
            # Retain only an actually published configuration for the endpoint
            # persistence handoff.  A stale pre-shutdown operation must not
            # leave an in-memory config which a later request could persist.
            self._created_configs[name] = config
            return agent
        finally:
            if owns_admission:
                release_cancelled = await self._release_agent_operations([admission])
                # ``load_agent`` records publication on this outer admission,
                # even when create owns it through the nested call. Do not
                # turn that committed create into a false cancellation while
                # merely retiring its now-terminal admission.
                if release_cancelled and not admission.published:
                    raise asyncio.CancelledError()

    def _parent_feature_names(self, parent_agent: KestrelAgent) -> set[str]:
        """Feature class names available to the parent — the ceiling a child's
        ``features_allowed`` may not exceed."""
        features = getattr(parent_agent, "features", None) or {}
        try:
            return {
                type(feat).__name__ if not isinstance(key, str) else key
                for key, feat in features.items()
            }
        except AttributeError:
            return set()

    def _validate_mandate_subset(
        self, parent_agent: KestrelAgent, mandate: SpawnMandate
    ) -> None:
        """Refuse a mandate that grants the child MORE than the parent (F277).

        Uses the shared ``ScopedConstitution`` narrowing rules so this consumer
        cannot drift from the spawn-constraint contract: a child's
        ``features_allowed`` must be a subset of the parent's features, and its
        ``additional_constraints`` must be restrictions (never
        ``grant_features`` / ``override_constitution`` / ``remove_restrictions``).
        """
        from kestrel_sovereign.spawn.scoped_constitution import ScopedConstitution

        scoped = ScopedConstitution(
            base_constitution="",
            additional_constraints=getattr(mandate, "additional_constraints", {}) or {},
            features_allowed=list(getattr(mandate, "features_allowed", []) or []),
            parent_features=self._parent_feature_names(parent_agent),
        )
        ok, msg = scoped.validate_constraints()
        if not ok:
            raise ValueError(f"Spawn refused: {msg}")

    # ------------------------------------------------------------------
    # Per-child spawn budgets (#2113): hold from the parent on spawn, route the
    # child's spend through a ceiling'd DelegatedWallet, release the unspent hold
    # on termination.
    # ------------------------------------------------------------------

    @staticmethod
    def _mandate_budget(mandate: SpawnMandate) -> Decimal:
        """The mandate's requested budget as a Decimal (0 when unset/invalid)."""
        raw = getattr(mandate, "budget_allocation", 0) or 0
        try:
            return Decimal(str(raw))
        except Exception:  # noqa: BLE001 — a malformed budget is treated as none
            return Decimal("0")

    def _validate_budget_precondition(
        self, parent_agent: KestrelAgent, mandate: SpawnMandate
    ) -> None:
        """Refuse a positive budget the parent can't back (#2113), before spawn."""
        budget = self._mandate_budget(mandate)
        if budget <= 0:
            return
        # Budgets are enforced IN-PROCESS only: the ceiling + hold live in memory
        # and are released on termination/shutdown. A persistent (non-TTL) child
        # could outlive the process and be reloaded WITHOUT the delegated wrapper,
        # bypassing the cap — so restrict budgets to ephemeral (TTL-bounded)
        # children, which are torn down within the process and never reloaded.
        # Durable budgets for persistent children (persist `spent` + rehydrate on
        # load + crash reconciliation) are tracked in #2348.
        ttl = getattr(mandate, "ttl_seconds", 0) or 0
        if ttl <= 0:
            raise ValueError(
                "Spawn refused: a per-child budget requires an ephemeral child "
                "(ttl_seconds > 0). Budgets are enforced in-process and are not yet "
                "durable across a reload of a persistent child (#2348). Set a TTL, "
                "or spawn without a budget."
            )
        parent_wallet = getattr(parent_agent, "wallet", None)
        if parent_wallet is None:
            raise ValueError(
                "Spawn refused: a per-child budget requires the parent to have a "
                "funded wallet (enable the wallet feature). Spawn without a budget "
                "or fund the parent's wallet."
            )
        if has_durable_delegated_child_wallet_provisioning_contract(parent_wallet):
            # A persistent provider's in-memory balance is explicitly only a
            # local snapshot. Its atomic reserve-and-provision transaction is
            # the affordability authority and also recognizes exact retries,
            # so a synchronous preflight here would reject an already-held
            # budget or a concurrent-process deposit before the provider can
            # decide safely.
            return
        currency = _default_currency_for(parent_wallet)
        if not parent_wallet.can_afford(budget, currency):
            raise ValueError(
                f"Spawn refused: parent wallet cannot afford the requested budget "
                f"of {budget}."
            )

    async def _apply_delegated_budget(
        self,
        name: str,
        parent_agent: KestrelAgent,
        child: KestrelAgent,
        mandate: SpawnMandate,
        *,
        admission: Optional[AgentOperationAdmission] = None,
    ) -> None:
        """Hold the budget from the parent and point the child's wallet at a
        ceiling'd DelegatedWallet (#2113). No-op for budget<=0."""
        budget = self._mandate_budget(mandate)
        if budget <= 0:
            return
        parent_wallet = getattr(parent_agent, "wallet", None)
        if parent_wallet is None:
            return  # precondition already refused this; defensive.
        # Do not remove the spawn candidate from this helper when the provider
        # refuses the allocation. ``_do_spawn`` owns that failure and
        # must first downgrade its durable signed receipt, then close storage
        # and withdraw routing. A nested cleanup here used to destroy the graph
        # before the receipt-first rollback could revoke that authority.
        delegated = await create_delegated_wallet(
            parent_wallet=parent_wallet,
            parent_did=parent_agent.agent_id,
            child_did=child.agent_id,
            budget=budget,
        )

        # Provider I/O may have yielded to terminal shutdown or a direct DELETE.
        # Claim the exact hold *before* any post-provider await. A positive
        # allocation is durable provider state already, so cancellation while
        # waiting for lifecycle locks must leave rollback an entry it can
        # release. The subsequent fenced check still controls whether this
        # spawn may commit governance state; it never makes an allocation
        # disappear from the cleanup owner's view.
        self._child_budgets[name] = (delegated, parent_wallet)
        if admission is not None:
            async with self._a2a_lifecycle_lock:
                async with self._lock:
                    if admission.published:
                        admitted = self._spawn_operation_is_admitted(
                            admission,
                            child,
                        )
                    else:
                        # Prepublication custody is installed by the spawn's
                        # private load callback. The exact admitted operation
                        # and child witness are authoritative here; requiring a
                        # routing binding would force custody back after ready.
                        admitted = (
                            self._operation_is_admitted(admission)
                            and admission.child is child
                            and self._agents.get(admission.name) is None
                        )
                    if not admitted:
                        raise RuntimeError(
                            "Spawn was fenced while its delegated budget was being created"
                        )
        child.wallet = delegated
        child.wallet_agent = delegated
        # Also expose it as ``_delegated_wallet`` so the spawn-status endpoint
        # reports live budget_spent / budget_remaining (#2113).
        child._delegated_wallet = delegated
        logger.info(
            "Applied delegated budget %s to child '%s' — spend now ceiling'd (#2113).",
            budget, name,
        )

    def _budgeted_child_agent_id(self, child_name: str) -> Optional[str]:
        """Resolve one held child's durable DID without requiring publication."""

        agent = self._agents.get(child_name)
        agent_id = _loaded_agent_did(agent) if agent is not None else None
        if isinstance(agent_id, str) and agent_id:
            return agent_id
        mandate = self._child_mandates.get(child_name)
        mandate_id = getattr(mandate, "child_did", None)
        if isinstance(mandate_id, str) and mandate_id:
            return mandate_id
        entry = self._child_budgets.get(child_name)
        delegated = entry[0] if isinstance(entry, tuple) and entry else None
        allocation = getattr(delegated, "allocation", None)
        allocation_id = getattr(allocation, "child_did", None)
        return allocation_id if isinstance(allocation_id, str) and allocation_id else None

    def _has_budgeted_descendants(
        self,
        name: str,
        *,
        known_agent_id: Optional[str] = None,
    ) -> bool:
        """True if any descendant of ``name`` still holds a budget (#2113).

        Keys on ``_child_budgets`` membership (not just the parent-child graph),
        so an already-released descendant — remove_agent pops its budget but not
        its ``_parent_children`` entry — no longer counts.
        """
        seen: set = set()

        def visit(n: str, did: Optional[str] = None) -> bool:
            did = did or self._budgeted_child_agent_id(n)
            if not did:
                return False
            for child in self._parent_children.get(did, []):
                if child in seen:
                    continue
                seen.add(child)
                if child in self._child_budgets:
                    return True
                if visit(child):
                    return True
            return False

        return visit(name, known_agent_id)

    async def _release_child_budget(self, child_name: str) -> None:
        """Credit a terminated child's unspent budget back to its parent (#2113).

        The admitted release task is the terminal owner of this refund: a
        provider failure must reach it so removal and a later terminal drain
        retain unsafe evidence instead of certifying an unknown refund state.
        Keep the hold entry until the provider confirms the refund: a durable
        provider can safely retry the same allocation idempotently, while a
        legacy provider's uncertain outcome remains visibly held and refuses a
        duplicate credit rather than disappearing from lifecycle ownership.
        The cascade case (a budgeted child with budgeted descendants) is
        handled by ``terminate_child`` recursing — each descendant releases
        its own hold.
        """
        entry = self._child_budgets.get(child_name)
        if entry is None:
            return
        delegated, parent_wallet = entry
        await self._release_child_budget_entry(
            child_name, delegated=delegated, parent_wallet=parent_wallet
        )
        # A concurrent terminal owner can only join this exact release task;
        # it must not make the hold look released before this provider call
        # returns. Remove the original entry only after confirmed success.
        if self._child_budgets.get(child_name) is entry:
            self._child_budgets.pop(child_name, None)

    async def _release_child_budget_entry(
        self, child_name: str, *, delegated, parent_wallet
    ) -> None:
        """Release an already-reserved hold through its terminal owner."""

        returned = await release_delegated_wallet(delegated, parent_wallet)
        logger.info(
            "Released delegated budget for '%s': returned %s to parent (#2113).",
            child_name,
            returned,
        )

    def _handoff_child_budget_release_to_quarantined_reaper(
        self, child_name: str, *, agent_id: str
    ) -> bool:
        """Fence a budget now and retain its blocked refund outside DELETE.

        A cancellation-resistant cognition turn can hold the delegated wallet's
        spend lock while arbitrary provider I/O is blocked.  Fencing is
        synchronous, so no later spend can begin; the retained reaper then
        waits for the in-flight transfer and performs exactly one refund.
        """

        if self._quarantined_shutdown_handoffs_sealed:
            return False

        entry = self._child_budgets.pop(child_name, None)
        if entry is None:
            return False
        delegated, parent_wallet = entry
        delegated.fence_spending()

        async def release() -> None:
            try:
                returned = await release_delegated_wallet(delegated, parent_wallet)
            except BaseException:
                # Quarantine fenced the wallet before withdrawing routing, but
                # its refund is still an ordinary ownership obligation.  Keep
                # the *exact* allocation entry retryable after a provider
                # failure/cancellation, just as the non-quarantined path does.
                # A new same-name create is refused while this entry remains,
                # so restoration cannot attach old money to a new identity.
                if child_name not in self._child_budgets:
                    self._child_budgets[child_name] = entry
                raise
            logger.info(
                "Released quarantined delegated budget for '%s': returned %s to parent (#2113).",
                child_name,
                returned,
            )

        task = asyncio.create_task(
            release(), name=f"agent_budget_release_quarantine:{child_name}"
        )
        self._retain_quarantined_cleanup(
            name=child_name,
            agent_id=agent_id,
            task=task,
        )
        logger.warning(
            "Handed delegated budget for %r to quarantined cleanup after immediate spend fence.",
            child_name,
        )
        return True

    def _record_unsafe_removal_budget_release_failure(
        self,
        record: InflightRemovalBudgetRelease,
        failure: BaseException,
    ) -> None:
        """Retain bounded evidence for one terminal ordinary-release failure."""

        unsafe = UnsafeRemovalBudgetReleaseFailure(
            release_id=record.release_id,
            child_name=_bounded_shutdown_metadata(record.child_name),
            started_monotonic=record.started_monotonic,
            completed_monotonic=time.monotonic(),
            failure=_bounded_shutdown_metadata(
                "budget release was cancelled"
                if isinstance(failure, asyncio.CancelledError)
                else f"{type(failure).__name__}: {failure}"
            ),
        )
        self._unsafe_removal_budget_release_failures[unsafe.release_id] = unsafe
        while (
            len(self._unsafe_removal_budget_release_failures)
            > _UNSAFE_REMOVAL_BUDGET_RELEASE_FAILURE_LIMIT
        ):
            self._unsafe_removal_budget_release_failures.pop(
                next(iter(self._unsafe_removal_budget_release_failures))
            )
            self._unsafe_removal_budget_release_failure_evictions += 1
        logger.error(
            "Ordinary child budget release %s remains unsafe for child %r: %s",
            unsafe.release_id,
            unsafe.child_name,
            unsafe.failure,
        )

    def _start_child_budget_release(
        self, child_name: str
    ) -> InflightRemovalBudgetRelease:
        """Admit or join one ordinary child-budget release under ``_lock``."""

        admitted = self._inflight_removal_budget_releases_by_child.get(child_name)
        if admitted is not None:
            return admitted

        if self._quarantined_shutdown_handoffs_sealed:
            raise RuntimeError(
                "cannot start ordinary budget release after the terminal manager "
                "drain has been sealed"
            )
        # Reserve a release id *before* calling the legacy override seam. A
        # synchronous override can mutate its hold and raise before it returns
        # an awaitable; that is still an unsafe terminal outcome which must be
        # visible to a later drain and require explicit acknowledgement.
        self._next_removal_budget_release_id += 1
        release_id = f"ordinary-budget-release:{self._next_removal_budget_release_id}"
        started_monotonic = time.monotonic()
        try:
            release_awaitable = self._release_child_budget_cancellation_safe(
                child_name
            )
            # ``ensure_future`` preserves the existing override seam: legacy
            # managers may return an already-created Future instead of a bare
            # coroutine from ``_release_child_budget_cancellation_safe``.
            budget_release = asyncio.ensure_future(release_awaitable)
        except BaseException as failure:
            # A failed Future gives the caller the same await/cancellation
            # contract as an asynchronously failed release, while ensuring the
            # just-reserved record can retain its evidence synchronously.
            budget_release = asyncio.get_running_loop().create_future()
            if isinstance(failure, asyncio.CancelledError):
                budget_release.cancel()
            else:
                budget_release.set_exception(failure)
        if isinstance(budget_release, asyncio.Task):
            budget_release.set_name(f"agent_budget_release:{child_name}")
        record = InflightRemovalBudgetRelease(
            release_id=release_id,
            child_name=child_name,
            task=budget_release,
            started_monotonic=started_monotonic,
        )
        self._inflight_removal_budget_releases[release_id] = record
        self._inflight_removal_budget_releases_by_child[child_name] = record

        def observe_budget_release_completion(
            task: "asyncio.Future[object]",
        ) -> None:
            # A completed task retains its traceback-bearing frame. Remove it
            # immediately, but never discard an unsafe terminal outcome: this
            # callback can run while a terminal drain is still waiting to
            # acquire ``_lock`` and therefore has not yet observed the task.
            self._inflight_removal_budget_releases.pop(record.release_id, None)
            if (
                self._inflight_removal_budget_releases_by_child.get(child_name)
                is record
            ):
                self._inflight_removal_budget_releases_by_child.pop(child_name, None)
            if task.cancelled():
                self._record_unsafe_removal_budget_release_failure(
                    record, asyncio.CancelledError()
                )
                return
            failure = task.exception()
            if failure is not None:
                self._record_unsafe_removal_budget_release_failure(record, failure)

        # ``Future.add_done_callback`` schedules callbacks for a later loop
        # turn when the override returned an already-completed Future. Retire
        # that result now instead: an immediate DELETE/shutdown retry must
        # admit a fresh release rather than coalescing to an old failed Future.
        if budget_release.done():
            observe_budget_release_completion(budget_release)
        else:
            budget_release.add_done_callback(observe_budget_release_completion)
        return record

    async def _await_child_budget_release_cancellation_safe(
        self, budget_release: "asyncio.Future[object]"
    ) -> bool:
        """Join one admitted budget release before reporting cancellation."""

        release_cancelled, release_failure = await await_lifecycle_task_completion(
            budget_release
        )
        # The task's done callback retains unsafe refund evidence before this
        # join returns.  Once that owned task has settled, the caller's
        # cancellation must still win over its provider error; DELETE otherwise
        # reports an ordinary failure even though its cancellation contract was
        # observed and cleanup fully completed.
        if release_cancelled:
            raise asyncio.CancelledError()
        if release_failure is not None:
            raise release_failure
        # ``_release_child_budget_cancellation_safe`` predates the task
        # admission wrapper and promises a boolean: ``True`` means its own
        # cancellation-safe cleanup observed caller cancellation after the
        # refund completed. The wrapper's join cancellation is only one half
        # of that contract; preserve an override's completed ``True`` result
        # too. Legacy overrides returning ``None`` remain a normal success.
        return release_cancelled or budget_release.result() is True

    async def _release_child_budget_cancellation_safe(self, child_name: str) -> bool:
        """Release one removed child's hold before reporting caller cancellation."""

        budget_release = asyncio.create_task(
            self._release_child_budget(child_name),
            name=f"agent_budget_release:{child_name}",
        )
        return await self._await_child_budget_release_cancellation_safe(budget_release)

    async def spawn_agent(
        self,
        name: str,
        parent_agent: KestrelAgent,
        mandate: SpawnMandate,
    ) -> KestrelAgent:
        """Create a child agent governed by a SpawnMandate.

        The child is created via inception, registered under the parent's
        DID in the parent-child tracking map, and its mandate is stored.

        Args:
            name: Name for the child agent.
            parent_agent: The parent KestrelAgent requesting the spawn.
            mandate: SpawnMandate describing purpose, budget, TTL, etc.

        Returns:
            The newly created and initialized child KestrelAgent.

        Raises:
            ValueError: If an agent with this name already exists or inception fails.
        """
        # The public object is a proposal owned by the caller.  Snapshot it
        # before validation and, critically, before the first await so caller
        # mutation cannot change the signed authority after it was admitted.
        mandate = copy.deepcopy(mandate)
        parent_did = _loaded_agent_did(parent_agent)
        if (
            parent_did is None
            or mandate.parent_did not in _loaded_agent_bound_dids(parent_agent)
        ):
            raise ValueError(
                "Spawn mandate parent DID does not match the requesting agent"
            )
        if mandate.child_did is not None:
            raise ValueError(
                "Spawn mandate child DID must be unset until inception"
            )
        # The caller may address a rotated parent by its successor signing DID.
        # New receipts persist the manager's stable routing DID so graph edges,
        # termination, and restart indexes retain one canonical parent key.
        mandate.parent_did = parent_did
        validate_spawn_max_child_depth(mandate.max_child_depth)

        # Subset-of-parent validation (F277): a mandate must only ever RESTRICT
        # the child relative to the parent — it may never grant features the
        # parent lacks or add constitution-weakening constraints. Enforce this
        # before any inception work, so an over-broad mandate is refused rather
        # than silently producing a child with more authority than its parent.
        self._validate_mandate_subset(parent_agent, mandate)

        # Budget precondition (#2113): a positive budget requires a funded parent
        # wallet to hold from, or the ceiling would be advertised-but-unenforced.
        # Validated before any inception work so it is refused rather than
        # producing an uncapped child.
        self._validate_budget_precondition(parent_agent, mandate)

        admission, owns_admission = await self._admit_agent_operation(
            name, kind="spawn"
        )
        assert owns_admission
        # Publish the parent binding before the operation task can run. A
        # terminal descendant fence can now either join this exact task or
        # make its cap admission fail; there is no unclassified middle state.
        admission.spawn_parent = parent_agent
        spawn_operation = asyncio.create_task(
            self._run_admitted_spawn(
                name,
                parent_agent,
                mandate,
                admission,
            ),
            name=f"agent_spawn:{admission.canonical_name}",
        )
        # Nested create -> load admission follows the operation task, not the
        # request/cognition task that happens to call ``spawn_agent``. Terminal
        # drains therefore join exactly through rollback/commit and cannot be
        # held open by unrelated caller work after this method returns.
        admission.owner_task = spawn_operation
        admission.spawn_task = spawn_operation
        return await spawn_operation

    async def _run_admitted_spawn(
        self,
        name: str,
        parent_agent: KestrelAgent,
        mandate: SpawnMandate,
        admission: AgentOperationAdmission,
    ) -> KestrelAgent:
        """Own one public spawn through cap retirement and admission release."""

        spawn_slot_admitted = False

        # Spawn caps (#1729): bound runaway spawning. The check + reservation run
        # under the manager lock so concurrent spawn_agent calls can't all read
        # the same count and blow past the cap (codex r2). ``_pending_spawns``
        # counts in-flight spawns whose mandate isn't registered yet.
        try:
            async with self._lock:
                if any(
                    self._terminal_descendant_spawn_fences.get(bound_did)
                    for bound_did in _loaded_agent_bound_dids(parent_agent)
                ):
                    raise RuntimeError(
                        "Spawn refused: parent is in terminal retirement"
                    )
                if any(
                    bound_did in self._non_governing_spawn_lineage
                    for bound_did in _loaded_agent_bound_dids(parent_agent)
                ):
                    raise ValueError(
                        "Spawn refused: unsigned restored lineage does not grant "
                        "delegation authority."
                    )
                in_use = self._spawn_cap_slots_in_use()
                if in_use >= self._max_spawned_agents:
                    raise ValueError(
                        f"Spawn refused: at the spawned-agent cap "
                        f"({self._max_spawned_agents}). Set KESTREL_MAX_SPAWNED_AGENTS to raise."
                    )
                # Depth cap — if the PARENT was itself spawned and its mandate marks
                # it a leaf (max_child_depth <= 0), it may not spawn further.
                parent_name = self._agent_names.get(parent_agent.agent_id)
                if parent_name is None:
                    prepared = self._prepared_agent_names.get(
                        parent_agent.agent_id
                    )
                    if prepared is not None and prepared[1] is parent_agent:
                        parent_name = prepared[0]
                parent_mandate = (
                    self._child_mandates.get(parent_name) if parent_name else None
                )
                if parent_mandate is not None and getattr(
                    parent_mandate, "max_child_depth", 0
                ) <= 0:
                    raise ValueError(
                        f"Spawn refused: parent '{parent_name}' is at its max child depth "
                        f"(mandate max_child_depth={getattr(parent_mandate, 'max_child_depth', 0)})."
                )
                self._pending_spawns += 1
                admission.spawn_slot_active = True
                admission.spawn_slot_terminal = (
                    asyncio.get_running_loop().create_future()
                )
                spawn_slot_admitted = True
            # DECREMENT remaining depth on delegation (codex r2): a non-leaf
            # spawned parent's child must have strictly less depth, regardless of
            # what the caller put in the mandate — otherwise depth never shrinks.
            if parent_mandate is not None:
                allowed = getattr(parent_mandate, "max_child_depth", 0) - 1
                if getattr(mandate, "max_child_depth", 0) > allowed:
                    mandate.max_child_depth = max(allowed, 0)
            return await self._do_spawn(name, parent_agent, mandate, admission)
        finally:
            release_cancelled = await self._retire_spawn_slot_and_admission(
                admission,
                spawn_slot_admitted=spawn_slot_admitted,
            )
            # The spawn body can be propagating a BaseExceptionGroup which
            # preserves both cancellation and a rollback failure.  Retirement
            # observes the same pending cancellation, but must not replace
            # that already-terminal evidence with a bare CancelledError.
            if (
                release_cancelled
                and sys.exception() is None
                and not admission.committed
                and not admission.rollback_incomplete
            ):
                raise asyncio.CancelledError()

    async def _do_spawn(
        self,
        name: str,
        parent_agent: KestrelAgent,
        mandate: SpawnMandate,
        admission: Optional[AgentOperationAdmission] = None,
    ) -> KestrelAgent:
        """Sign, create, then atomically commit or roll back one child spawn."""
        # A couple of focused feature tests exercise this private seam directly.
        # Give those calls the same named ownership contract as public spawn,
        # rather than preserving a second untracked creation path.
        if admission is None:
            admission, owns_admission = await self._admit_agent_operation(
                name, kind="direct-spawn-test"
            )
            assert owns_admission
            admission.spawn_parent = parent_agent

            async def run_direct_spawn() -> KestrelAgent:
                try:
                    return await self._do_spawn(
                        name, parent_agent, mandate, admission
                    )
                finally:
                    release_cancelled = await self._release_agent_operations(
                        [admission]
                    )
                    if (
                        release_cancelled
                        and sys.exception() is None
                        and not admission.committed
                    ):
                        raise asyncio.CancelledError()

            spawn_operation = asyncio.create_task(
                run_direct_spawn(),
                name=f"agent_direct_spawn:{admission.canonical_name}",
            )
            admission.owner_task = spawn_operation
            admission.spawn_task = spawn_operation
            return await spawn_operation

        assert admission is not None
        admission.spawn_parent = parent_agent
        async with self._lock:
            if any(
                self._terminal_descendant_spawn_fences.get(bound_did)
                for bound_did in _loaded_agent_bound_dids(parent_agent)
            ):
                raise RuntimeError("Spawn refused: parent is in terminal retirement")
        # Resolve the parent's signing material now. The final signature is
        # created only after inception returns the child's DID; signing before
        # that point would bind ``child_did=None`` and cannot authorize the
        # identity that was actually born.
        # Hybrid parents (post-rotation ceremony) get an additional
        # ``parent_identity`` arg so the mandate is signed with both
        # Ed25519 and ML-DSA-65; legacy parents fall through to the
        # bare-hex ECDSA path. The parent's runtime identity is set
        # on the agent at startup by KestrelAgent.__init__ (#999).
        # Resolve the child's feature ceiling BEFORE signing so the signed
        # mandate — and the spawned_by edge inception persists from it — records
        # the ACTUAL ceiling, explicit or inherited (#1946). An empty list is a
        # real (empty) allowlist; only ``None`` means "load all".
        explicit_features = getattr(mandate, "features_allowed", None)
        if explicit_features:
            child_features = list(explicit_features)
        else:
            # No explicit allowlist ⇒ inherit the PARENT's feature ceiling, NOT
            # "load all discovered features" (F277 / codex P1). Otherwise a
            # restricted parent could spawn a broader-than-itself child simply by
            # omitting features_allowed. A parent with no resolvable feature set
            # (degenerate/test doubles) falls back to None (load all).
            child_features = sorted(self._parent_feature_names(parent_agent)) or None
            # Persist the INHERITED ceiling onto the mandate so it is durable on
            # the edge and enforced on every boot path (#2226) — not just via
            # this process's config threading. Without this, an inherited-ceiling
            # child persists an empty features_allowed and, on a direct restart
            # outside AgentManager, would escape its ceiling and load everything.
            if child_features:
                mandate.features_allowed = list(child_features)

        parent_private_key = getattr(parent_agent, '_private_key', None)
        parent_identity = getattr(parent_agent, 'identity', None)
        # A hybrid parent (rotated or born-hybrid #2397) signs via its
        # hybrid keypair and needs no legacy private key; a legacy
        # parent signs with its ECDSA key.
        parent_is_hybrid = bool(parent_identity is not None and parent_identity.is_hybrid)
        mandate.parent_signature = None
        child: Optional[KestrelAgent] = None

        async def persist_final_spawn_receipt(candidate: KestrelAgent) -> None:
            """Bind authority and custody before ``load_agent`` publishes."""

            admission.child = candidate
            mandate.child_did = candidate.agent_id
            raw_storage = vars(candidate).get("_raw_storage")
            if raw_storage is None:
                raise RuntimeError(
                    "Spawned child has no durable storage for its signed mandate"
                )
            # A caller-created mandate is only a proposal until the final child
            # DID exists. Establish the signed window at this last prepublication
            # seam so slow inception cannot consume the child's authority TTL.
            proposal_created_at = mandate.created_at
            mandate.created_at = datetime.now(timezone.utc).isoformat()
            sign_mandate(
                mandate,
                parent_private_key,
                parent_identity=parent_identity if parent_is_hybrid else None,
            )
            graph = getattr(raw_storage, "graph", None)
            if graph is None:
                raise RuntimeError(
                    "Spawned child has no durable graph for its signed mandate"
                )
            properties = mandate.to_edge_properties()
            admission.spawn_receipt_graph = graph
            admission.spawn_receipt_source_id = candidate.agent_id
            admission.spawn_receipt_target_id = parent_agent.agent_id
            admission.spawn_receipt_unsigned_properties = {
                **properties,
                "created_at": proposal_created_at,
                "parent_signature": None,
            }
            vars(candidate)["_persisted_spawn_mandate"] = mandate
            # The candidate completed boot from its unsigned inception edge.
            # Once the final receipt is signed, this operation must own the
            # exact deadline before any registry/graph/budget work can wait.
            arm_host_authority_deadline(candidate, mandate)
            candidate_config = admission.spawn_candidate_config
            if not isinstance(candidate_config, LocalAgentConfig):
                raise RuntimeError(
                    "Spawned child has no local configuration for host authority"
                )
            pending_id = admission.spawn_authority_pending_id
            if pending_id is None:
                raise RuntimeError(
                    "Spawned child has no pre-inception restart denial"
                )
            self._spawn_authority_registry.promote_pending(
                reservation_id=pending_id,
                child_name=name,
                child_did=candidate.agent_id,
                mandate=mandate,
                config=candidate_config,
                proposal_created_at=proposal_created_at,
            )
            admission.spawn_authority_pending_id = None
            admission.spawn_authority_witness_did = candidate.agent_id
            admission.spawn_authority_witness_mandate = copy.deepcopy(mandate)
            # Config-driven selection is also durable before any ready hook or
            # routing publication. The registry repairs this row if the host
            # crashes between the two atomic writes.
            self._persist_spawn_startup_registration(admission)
            signed_deadline = self._spawn_admission_deadline(candidate)
            await self._await_before_spawn_admission_deadline(
                lambda: graph.add_trusted_cross_agent_edge(
                    candidate.agent_id,
                    parent_agent.agent_id,
                    "spawned_by",
                    properties=properties,
                ),
                deadline=signed_deadline,
                phase="signed receipt persistence",
                on_expiry=lambda: self._fence_expired_spawn_route(name, candidate),
                admission=admission,
            )
            # Ready hooks are wake-capable and the child becomes routable
            # immediately afterward. Install delegated custody while the
            # candidate is still private so neither hook nor peer traffic can
            # spend through its unrestricted boot wallet.
            await self._await_before_spawn_admission_deadline(
                lambda: self._apply_delegated_budget(
                    name,
                    parent_agent,
                    candidate,
                    mandate,
                    admission=admission,
                ),
                deadline=signed_deadline,
                phase="delegated budget custody",
                on_expiry=lambda: self._fence_expired_spawn_route(name, candidate),
                admission=admission,
            )

        admission.before_publish = persist_final_spawn_receipt

        async def revoke_failed_prepublication_receipt(
            candidate: KestrelAgent,
        ) -> None:
            await self._downgrade_uncommitted_spawn_receipt(admission, candidate)

        admission.before_publish_rollback = revoke_failed_prepublication_receipt
        try:
            child = await self.create_agent(
                name,
                parent_did=parent_agent.agent_id,
                features=child_features,
                mandate=mandate,
            )
            admission.child = child
            await self._ensure_spawn_operation_admitted(admission, child)

            if admission.spawn_receipt_graph is None:
                if admission.kind != "direct-spawn-test":
                    raise RuntimeError(
                        "Spawned child reached publication without a durable signed receipt"
                    )
                # Focused compatibility tests exercise this private seam with
                # an in-memory create_agent double, so no durable graph exists
                # to run ``before_publish`` and fill the final DID. Keep that
                # test-only ownership path internally coherent before the
                # lifecycle adopts its mandate; production spawns still fail
                # closed above unless their signed receipt was persisted.
                child_did = _loaded_agent_did(child)
                if child_did is None:
                    raise RuntimeError(
                        "Direct spawn test returned a child without a concrete DID"
                    )
                mandate.child_did = child_did

            # Runtime enforcement (spawn_mandate attach + restricted_tools hook)
            # is applied uniformly in load_agent from the persisted delegation
            # edge (#2137), which already ran for this child inside create_agent.
            parent_did = parent_agent.agent_id
            while True:
                capacity_waiter: asyncio.Future[None] | None = None
                async with self._a2a_lifecycle_lock:
                    async with self._lock:
                        if not self._spawn_operation_is_admitted(admission, child):
                            raise RuntimeError(
                                "Spawn was fenced before its budget and mandate could commit"
                            )
                        if (
                            mandate.ttl_seconds > 0
                            and remaining_spawn_ttl_seconds(
                                mandate.created_at,
                                mandate.ttl_seconds,
                            ) <= 0
                        ):
                            raise RuntimeError(
                                "Spawn mandate expired before governance commit"
                            )
                        # A cold load may restore a persisted child while this
                        # spawn is doing inception/provider I/O. Exactly one
                        # pending spawn becomes the rollback loser. Later
                        # contenders wait for that exact reservation to reach a
                        # terminal commit/rollback boundary, rather than all
                        # rejecting against the same transient over-cap count.
                        if admission.kind == "spawn":
                            in_use = self._spawn_cap_slots_in_use()
                            if in_use > self._max_spawned_agents:
                                capacity_waiter = next(
                                    (
                                        waiter
                                        for waiter in self._rejected_spawn_slot_waiters
                                        if waiter is not admission.spawn_slot_terminal
                                        and not waiter.done()
                                    ),
                                    None,
                                )
                                if capacity_waiter is None:
                                    admission.spawn_cap_rejected = True
                                    terminal = admission.spawn_slot_terminal
                                    if terminal is None:
                                        raise RuntimeError(
                                            "Rejected spawn has no cap-slot lifecycle"
                                        )
                                    self._rejected_spawn_slot_waiters.add(terminal)
                                    raise ValueError(
                                        f"Spawn refused: restored authority consumed the "
                                        f"last spawned-agent cap slot "
                                        f"({self._max_spawned_agents})."
                                    )
                            else:
                                if not admission.spawn_slot_active:
                                    raise RuntimeError(
                                        "Spawn cap reservation was lost before commit"
                                    )
                                self._pending_spawns -= 1
                                admission.spawn_slot_active = False
                                self._resolve_spawn_slot_terminal(admission)
                        if capacity_waiter is None:
                            if (
                                admission.provisional_spawn_authority
                                and self._child_mandates.get(name) is not mandate
                            ):
                                raise RuntimeError(
                                    "Spawn lost its provisional authority before "
                                    "governance commit"
                                )
                            children = self._parent_children.setdefault(
                                parent_did,
                                [],
                            )
                            if name not in children:
                                children.append(name)
                            self._child_mandates[name] = mandate
                            lifecycle = self._ensure_spawn_lifecycle()
                            lifecycle.restore_persisted_child(
                                name,
                                mandate,
                                authority_parent_did=parent_did,
                                arm_ttl=True,
                            )
                            disarm_host_authority_deadline(child)
                            self._persist_spawn_startup_registration(
                                admission,
                            )
                            admission.committed = True
                            admission.provisional_spawn_authority = False
                if capacity_waiter is None:
                    break
                if mandate.ttl_seconds <= 0:
                    await asyncio.shield(capacity_waiter)
                    continue
                remaining_ttl = remaining_spawn_ttl_seconds(
                    mandate.created_at,
                    mandate.ttl_seconds,
                )
                if remaining_ttl <= 0:
                    self._fence_expired_spawn_route(admission.name, child)
                    raise RuntimeError(
                        "Spawn mandate expired before governance commit"
                    )
                try:
                    await asyncio.wait_for(
                        asyncio.shield(capacity_waiter),
                        timeout=remaining_ttl,
                    )
                except TimeoutError:
                    self._fence_expired_spawn_route(admission.name, child)
                    raise RuntimeError(
                        "Spawn mandate expired before governance commit"
                    ) from None

            logger.info(
                f"Spawned child '{name}' (DID: {child.agent_id[:30]}...) "
                f"for parent '{parent_did[:30]}...' — purpose: {mandate.purpose}"
            )
            return child
        except BaseException as spawn_failure:
            if child is None:
                child = admission.child
            if child is not None and not admission.committed:
                cleanup_failure: Optional[BaseException] = None
                cleanup_cancelled = False
                try:
                    cleanup_cancelled = await self._rollback_uncommitted_spawn(
                        admission, child
                    )
                except BaseException as error:
                    cleanup_failure = error
                authority_expired = (
                    vars(child).get("_host_authority_boot_expired") is True
                )
                if authority_expired and _contains_lifecycle_cancellation(
                    spawn_failure
                ):
                    expiry_failure = PersistedSpawnMandateExpiredError(
                        "Persisted spawn mandate expired during active host admission"
                    )
                    retained_spawn_failure: BaseException | None = None
                    if isinstance(spawn_failure, BaseExceptionGroup):
                        _cancelled, retained_spawn_failure = spawn_failure.split(
                            asyncio.CancelledError
                        )
                    outcomes: list[BaseException] = [expiry_failure]
                    if retained_spawn_failure is not None:
                        outcomes.append(retained_spawn_failure)
                    if cleanup_failure is not None:
                        outcomes.append(cleanup_failure)
                    if len(outcomes) == 1:
                        raise expiry_failure from spawn_failure
                    _raise_lifecycle_outcomes(
                        "Spawn authority expiry and its owned rollback failed",
                        outcomes,
                    )
                # A rollback that proves its uncommitted child or exact hold
                # is still live is not safely subsumed by the caller's
                # cancellation.  Preserve both facts so a cancellation cannot
                # make a routable, mandate-less child look like a completed
                # rollback to its lifecycle owner.
                if cleanup_failure is not None and (
                    isinstance(spawn_failure, asyncio.CancelledError)
                    or cleanup_cancelled
                ):
                    cancellation = (
                        spawn_failure
                        if isinstance(spawn_failure, asyncio.CancelledError)
                        else asyncio.CancelledError()
                    )
                    raise BaseExceptionGroup(
                        "Spawn cancellation and its owned rollback failed",
                        [cancellation, cleanup_failure],
                    )
                if (
                    isinstance(spawn_failure, asyncio.CancelledError)
                    or cleanup_cancelled
                    or isinstance(cleanup_failure, asyncio.CancelledError)
                ):
                    raise asyncio.CancelledError()
                if cleanup_failure is not None:
                    if isinstance(spawn_failure, Exception) and isinstance(
                        cleanup_failure, Exception
                    ):
                        raise ExceptionGroup(
                            "Spawn and its owned rollback failed",
                            [spawn_failure, cleanup_failure],
                        )
                    raise cleanup_failure
            raise

    def _spawn_operation_is_admitted(
        self, admission: AgentOperationAdmission, child: KestrelAgent
    ) -> bool:
        """Whether this exact child may still commit its spawn bookkeeping.

        Callers hold the A2A lifecycle writer.  Checking the published object
        identity prevents a concurrent DELETE from turning an old child into a
        successful spawn result after it has been withdrawn.
        """

        child_id = _loaded_agent_did(child)
        # ``_do_spawn`` is a long-standing private feature-test seam whose
        # lightweight fake ``create_agent`` deliberately does not publish a
        # runtime child.  Public ``spawn_agent`` is the production path and
        # requires the exact routing/reverse binding before it can commit.
        if admission.kind == "direct-spawn-test" and self._agents.get(
            admission.name
        ) is None:
            return self._operation_is_admitted(admission)
        return (
            self._operation_is_admitted(admission)
            and self._agents.get(admission.name) is child
            and not self._spawn_route_is_fenced(admission.name, child)
            and isinstance(child_id, str)
            and self._agent_names.get(child_id) == admission.name
        )

    async def _ensure_spawn_operation_admitted(
        self, admission: AgentOperationAdmission, child: KestrelAgent
    ) -> None:
        """Fail a stale spawn before it makes another irreversible mutation."""

        async with self._a2a_lifecycle_lock:
            if not self._spawn_operation_is_admitted(admission, child):
                raise RuntimeError(
                    "Spawn was fenced before its child could commit"
                )

    async def _rollback_uncommitted_spawn(
        self, admission: AgentOperationAdmission, child: KestrelAgent
    ) -> bool:
        """Own rollback of a child whose spawn never committed its mandate.

        ``remove_agent`` remains the sole runtime/budget cleanup primitive.  A
        concurrent DELETE may have already claimed that child, in which case
        its ``False`` is a completed handoff rather than permission to issue a
        second shutdown.  A ``False`` is not itself proof of that handoff,
        though: inspect the authoritative routing/hold state before accepting
        it, so a failed spawn cannot leave a routable uncommitted child behind.
        Return cancellation only after that chosen owner and the inspection
        have both reached terminal state.
        """

        receipt_cleanup = asyncio.create_task(
            self._downgrade_uncommitted_spawn_receipt(admission, child),
            name=f"rollback_spawn_receipt:{admission.name}",
        )
        receipt_cancelled, receipt_failure = (
            await await_lifecycle_task_completion(receipt_cleanup)
        )

        if receipt_failure is not None:
            # A failed revocation is ambiguous whether the signed edge remains
            # durable.  Publication does not make that ambiguity safe: the
            # failed spawn still owns the child until an idempotent revocation
            # succeeds and the matching runtime is withdrawn.  Retain that
            # cleanup owner for both prepublication and published children.
            handoff_cancelled = await self._handoff_failed_spawn_cleanup(
                admission,
                child,
            )
            admission.rollback_incomplete = True
            if handoff_cancelled:
                raise BaseExceptionGroup(
                    "Spawn receipt revocation failed during cancellation",
                    [asyncio.CancelledError(), receipt_failure],
                )
            raise receipt_failure

        startup_cleanup = asyncio.create_task(
            self._withdraw_uncommitted_spawn_startup_registration(admission),
            name=f"rollback_spawn_startup_roster:{admission.name}",
        )
        startup_cancelled, startup_failure = (
            await await_lifecycle_task_completion(startup_cleanup)
        )
        if startup_failure is not None:
            handoff_cancelled = await self._handoff_failed_spawn_cleanup(
                admission,
                child,
            )
            admission.rollback_incomplete = True
            failures: list[BaseException] = [startup_failure]
            if handoff_cancelled:
                failures.insert(0, asyncio.CancelledError())
            _raise_lifecycle_outcomes(
                "Spawn startup-roster rollback failed",
                failures,
            )

        runtime_cleanup = asyncio.create_task(
            self._rollback_uncommitted_spawn_runtime(admission, child),
            name=f"rollback_spawn_runtime:{admission.name}",
        )
        runtime_cancelled, runtime_failure = (
            await await_lifecycle_task_completion(runtime_cleanup)
        )
        failures = [
            failure
            for failure in (receipt_failure, runtime_failure)
            if failure is not None
        ]
        if failures:
            admission.rollback_incomplete = True
            handoff_cancelled = await self._handoff_failed_spawn_cleanup(
                admission,
                child,
            )
            if handoff_cancelled:
                failures.insert(0, asyncio.CancelledError())
            _raise_lifecycle_outcomes(
                "Spawn receipt and runtime rollback failed",
                failures,
            )

        witness_cleanup = asyncio.create_task(
            self._withdraw_uncommitted_spawn_authority_witness(admission),
            name=f"rollback_spawn_authority_witness:{admission.name}",
        )
        witness_cancelled, witness_failure = (
            await await_lifecycle_task_completion(witness_cleanup)
        )
        if witness_failure is not None:
            handoff_cancelled = await self._handoff_failed_spawn_cleanup(
                admission,
                child,
            )
            admission.rollback_incomplete = True
            failures = [witness_failure]
            if handoff_cancelled:
                failures.insert(0, asyncio.CancelledError())
            _raise_lifecycle_outcomes(
                "Spawn authority-witness rollback failed",
                failures,
            )
        return (
            receipt_cancelled
            or startup_cancelled
            or witness_cancelled
            or runtime_cancelled
            or receipt_cleanup.result()
            or startup_cleanup.result()
            or witness_cleanup.result()
            or runtime_cleanup.result()
        )

    def _persist_spawn_startup_registration(
        self,
        admission: AgentOperationAdmission,
    ) -> None:
        """Add a prepublication child candidate to a config-driven roster."""

        config_path = self._startup_config_path
        if config_path is None:
            return
        created = admission.spawn_candidate_config
        if not isinstance(created, LocalAgentConfig):
            created = self._created_configs.get(admission.name)
        if not isinstance(created, LocalAgentConfig):
            raise RuntimeError(
                "Spawned child has no local configuration for restart selection"
            )
        current = MultiAgentConfig.from_file(config_path)
        collision = next(
            (
                existing
                for existing in current.agents
                if existing.casefold() == admission.name.casefold()
            ),
            None,
        )
        if collision is not None:
            if (
                collision == admission.name
                and current.agents[collision] == created
            ):
                admission.spawn_startup_config = copy.deepcopy(created)
                admission.spawn_startup_config_path = config_path
                return
            raise RuntimeError(
                "Spawned child startup registration conflicts with the current "
                f"host roster: {collision!r}"
            )
        current.agents[admission.name] = copy.deepcopy(created)
        type(current).model_validate(current.model_dump())
        current.save(config_path)
        admission.spawn_startup_config = copy.deepcopy(created)
        admission.spawn_startup_config_path = config_path

    async def _withdraw_uncommitted_spawn_startup_registration(
        self,
        admission: AgentOperationAdmission,
    ) -> bool:
        """Remove only the exact startup row written by this failed spawn."""

        expected = admission.spawn_startup_config
        config_path = admission.spawn_startup_config_path
        if expected is None and config_path is None:
            return False
        if not isinstance(expected, LocalAgentConfig) or config_path is None:
            raise RuntimeError("Uncommitted spawn startup witness is incomplete")
        current = MultiAgentConfig.from_file(config_path)
        stored = current.agents.get(admission.name)
        if stored != expected:
            raise RuntimeError(
                "Refusing to erase a changed spawned-child startup registration"
            )
        del current.agents[admission.name]
        type(current).model_validate(current.model_dump())
        current.save(config_path)
        admission.spawn_startup_config = None
        admission.spawn_startup_config_path = None
        return False

    def _withdraw_committed_spawn_startup_registration(
        self,
        child_name: str,
        *,
        expected_child_did: str,
    ) -> Optional[tuple[Path, str, LocalAgentConfig]]:
        """Remove an exact spawned-child desired-state row before offboarding.

        The public agent DELETE endpoint already removes a witnessed config row
        before destructive cleanup.  ``terminate_child`` is a second destructive
        entry point, so it must perform the same desired-state transition rather
        than leaving a committed spawn row able to recreate the child at boot.
        A runtime-created config is an exact witness.  After restart, bind the
        current row back to the loaded child's durable identity before mutation.
        """

        config_path = self._startup_config_path
        if config_path is None:
            return None
        current = MultiAgentConfig.from_file(config_path)
        matching = [
            name
            for name in current.agents
            if name.casefold() == child_name.casefold()
        ]
        if len(matching) > 1:
            raise RuntimeError(
                "Spawned-child startup registration is ambiguous; offboarding "
                "was refused"
            )
        if not matching:
            return None
        persisted_name = matching[0]
        stored = current.agents[persisted_name]
        if not isinstance(stored, LocalAgentConfig):
            raise RuntimeError(
                "Spawned child is not a local startup registration"
            )
        created = self._created_configs.get(persisted_name)
        row_is_bound = isinstance(created, LocalAgentConfig) and created == stored
        if not row_is_bound:
            resolved_dir = stored.resolve_data_dir(self._base_data_dir)
            try:
                anchored_did = read_anchor_agent_did_sync(
                    str(resolved_dir),
                    mode=AgentDIDLookupMode.INSPECTION,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Spawned-child startup registration identity is unavailable; "
                    "offboarding was refused"
                ) from exc
            if anchored_did != expected_child_did:
                raise RuntimeError(
                    "Spawned-child startup registration changed identity; "
                    "offboarding was refused"
                )
        del current.agents[persisted_name]
        type(current).model_validate(current.model_dump())
        current.save(config_path)
        return config_path, persisted_name, stored

    def _restore_committed_spawn_startup_registration(
        self,
        witness: tuple[Path, str, LocalAgentConfig],
    ) -> None:
        """Compensate desired state only when destructive cleanup never began."""

        config_path, persisted_name, expected = witness
        current = MultiAgentConfig.from_file(config_path)
        matching = [
            name
            for name in current.agents
            if name.casefold() == persisted_name.casefold()
        ]
        if matching:
            if len(matching) == 1 and current.agents[matching[0]] == expected:
                return
            raise RuntimeError(
                "Spawned-child startup registration changed during offboarding; "
                "refusing rollback overwrite"
            )
        current.agents[persisted_name] = expected
        type(current).model_validate(current.model_dump())
        current.save(config_path)

    async def _downgrade_uncommitted_spawn_receipt(
        self,
        admission: AgentOperationAdmission,
        child: KestrelAgent,
    ) -> bool:
        """Atomically revoke authority while preserving restrictive lineage."""

        witness_did = admission.spawn_authority_witness_did
        witness_mandate = admission.spawn_authority_witness_mandate
        if witness_did is not None or witness_mandate is not None:
            if not isinstance(witness_did, str) or not isinstance(
                witness_mandate, SpawnMandate
            ):
                raise RuntimeError("Uncommitted host spawn witness is incomplete")
            existing = self._spawn_authority_registry.get(witness_did)
            if (
                existing is None
                or existing.child_name.casefold() != admission.name.casefold()
                or existing.mandate.to_dict() != witness_mandate.to_dict()
            ):
                raise RuntimeError(
                    "Uncommitted host spawn witness changed during rollback"
                )
            # This is the first durable rollback mutation.  A crash after the
            # child edge is downgraded or the startup row is removed must find a
            # terminal intent, never an active witness that attempts repair.
            self._spawn_authority_registry.begin_retirement(
                child_name=admission.name,
                child_did=witness_did,
            )

        graph = admission.spawn_receipt_graph
        source_id = admission.spawn_receipt_source_id
        target_id = admission.spawn_receipt_target_id
        if graph is not None:
            if not isinstance(source_id, str) or not isinstance(target_id, str):
                raise RuntimeError("Uncommitted spawn receipt witness is incomplete")
            unsigned_properties = admission.spawn_receipt_unsigned_properties
            if unsigned_properties is None:
                raise RuntimeError("Uncommitted spawn receipt rollback is incomplete")
            replace_edge = getattr(graph, "add_trusted_cross_agent_edge", None)
            if not callable(replace_edge):
                raise RuntimeError("Spawn receipt graph cannot revoke its authority")
            await replace_edge(
                source_id,
                target_id,
                "spawned_by",
                properties=unsigned_properties,
            )
            admission.spawn_receipt_graph = None
            admission.spawn_receipt_source_id = None
            admission.spawn_receipt_target_id = None
            admission.spawn_receipt_unsigned_properties = None
            async with self._lock:
                persisted = vars(child).get("_persisted_spawn_mandate")
                if isinstance(persisted, SpawnMandate):
                    unsigned_created_at = unsigned_properties.get("created_at")
                    if unsigned_created_at is not None and not isinstance(
                        unsigned_created_at, str
                    ):
                        raise RuntimeError(
                            "Uncommitted spawn proposal timestamp is unavailable"
                        )
                    replacement = {"parent_signature": None}
                    if isinstance(unsigned_created_at, str):
                        replacement["created_at"] = unsigned_created_at
                    vars(child)["_persisted_spawn_mandate"] = replace(
                        persisted, **replacement
                    )
                child_id = _loaded_agent_did(child)
                if isinstance(child_id, str) and child_id:
                    self._withdraw_restored_spawn_authority(
                        admission.name,
                        child_id,
                    )
                    self._non_governing_spawn_lineage.update(
                        _loaded_agent_bound_dids(child)
                    )

        # Either lifecycle custody committed (success path) or signed authority
        # has now been durably denied and locally revoked (this rollback path).
        # Do not leave the temporary watchdog able to cancel later cleanup.
        disarm_host_authority_deadline(child)

        return False

    async def _withdraw_uncommitted_spawn_authority_witness(
        self,
        admission: AgentOperationAdmission,
    ) -> bool:
        """Settle the witness after desired state and runtime are durably gone."""

        witness_did = admission.spawn_authority_witness_did
        witness_mandate = admission.spawn_authority_witness_mandate
        if witness_did is not None or witness_mandate is not None:
            if not isinstance(witness_did, str) or not isinstance(
                witness_mandate, SpawnMandate
            ):
                raise RuntimeError("Uncommitted host spawn witness is incomplete")
            existing = self._spawn_authority_registry.get(witness_did)
            if existing is not None:
                if (
                    existing.child_name.casefold() != admission.name.casefold()
                    or existing.mandate.to_dict() != witness_mandate.to_dict()
                ):
                    raise RuntimeError(
                        "Uncommitted host spawn witness changed during rollback"
                    )
                if existing.active:
                    # A non-discoverable private runtime may have no terminal
                    # marker to retain. Runtime cleanup has already completed,
                    # so its still-active prepublication witness can be erased.
                    self._spawn_authority_registry.withdraw_active(
                        child_name=admission.name,
                        child_did=witness_did,
                        mandate=witness_mandate,
                    )
                # Retiring/retired witnesses are durable restart denials. Keep
                # that tombstone; only release this admission's cleanup handle.
            admission.spawn_authority_witness_did = None
            admission.spawn_authority_witness_mandate = None
        return False

    async def _rollback_uncommitted_spawn_runtime(
        self,
        admission: AgentOperationAdmission,
        child: KestrelAgent,
    ) -> bool:
        """Remove the live child and budget after its receipt is withdrawn."""

        if admission.unpublished_cleanup_deferred_to_spawn:
            discard_task = asyncio.create_task(
                self._discard_unpublished_initialized_agent(
                    admission.name,
                    child,
                ),
                name=f"rollback_private_spawn_shutdown:{admission.name}",
            )
            discard_cancelled, discard_failure = (
                await await_lifecycle_task_completion(discard_task)
            )
            if discard_failure is not None:
                # The child may still own live resources. Keep its isolated
                # credentials/files and the retry marker intact until shutdown
                # completion is positively confirmed.
                _raise_lifecycle_outcomes(
                    "Private spawn shutdown failed; runtime namespace retained",
                    [discard_failure],
                )
            offboard_task = asyncio.create_task(
                self._offboard_agent_runtime_namespace(child),
                name=f"rollback_private_spawn_offboard:{admission.name}",
            )
            offboard_cancelled, offboard_failure = (
                await await_lifecycle_task_completion(offboard_task)
            )
            failures: list[BaseException] = []
            if offboard_failure is not None:
                failures.append(offboard_failure)
            elif offboard_task.done() and not offboard_task.cancelled():
                nested_cancelled, nested_failure = offboard_task.result()
                offboard_cancelled = offboard_cancelled or nested_cancelled
                if nested_failure is not None:
                    no_hosted_cancellation = (
                        _uncommitted_spawn_not_hosted_cancellation(
                            nested_failure
                        )
                    )
                    if no_hosted_cancellation is None:
                        failures.append(nested_failure)
                    else:
                        offboard_cancelled = (
                            offboard_cancelled or no_hosted_cancellation
                        )
            _raise_lifecycle_outcomes(
                "Private spawn shutdown and runtime offboarding failed",
                failures,
            )
            # A private spawn can already own a delegated wallet even though
            # it never reached routing publication. The spawn admission owns
            # this cancellation-safe refund through terminal completion; do
            # not clear its cleanup marker while money or a discoverable
            # primary database can still survive the rejected spawn.
            budget_cancelled = (
                await self._release_child_budget_cancellation_safe(
                    admission.name
                )
            )
            child_id = _loaded_agent_did(child)
            if not isinstance(child_id, str) or not child_id:
                raise RuntimeError(
                    "Rejected spawned candidate has no stable identity to retire"
                )
            self.record_expired_spawn_retirement(
                admission.name,
                expected_child_did=child_id,
            )
            admission.unpublished_cleanup_deferred_to_spawn = False
            return discard_cancelled or offboard_cancelled or budget_cancelled

        descendant_cancelled = False
        descendant_failure: BaseException | None = None
        child_id = _loaded_agent_did(child)
        if isinstance(child_id, str) and child_id:
            descendant_cleanup = asyncio.create_task(
                self.terminate_children(child_id, offboard_runtime=True),
                name=f"rollback_spawn_descendants:{admission.name}",
            )
            descendant_cancelled, descendant_failure = (
                await await_lifecycle_task_completion(descendant_cleanup)
            )
            if descendant_failure is not None:
                no_hosted_cancellation = (
                    _uncommitted_spawn_not_hosted_cancellation(
                        descendant_failure
                    )
                )
                if no_hosted_cancellation is not None:
                    # Storage-backed descendants have no isolated namespace to
                    # delete.  Their destructive rollback is nevertheless
                    # terminal once routing, desired-state rows, and any
                    # discoverable primary database have been retired.
                    descendant_cancelled = (
                        descendant_cancelled or no_hosted_cancellation
                    )
                    descendant_failure = None

        cleanup = asyncio.create_task(
            self.remove_agent(
                admission.name,
                offboard_runtime=True,
                _spawn_rollback_admission=admission,
            ),
            name=f"rollback_uncommitted_spawn:{admission.name}",
        )
        cancelled, failure = await await_lifecycle_task_completion(cleanup)
        no_hosted_cleanup = False
        if failure is not None:
            no_hosted_cancellation = _uncommitted_spawn_not_hosted_cancellation(
                failure
            )
            if no_hosted_cancellation is not None:
                # remove_agent already proved successful shutdown and routing
                # withdrawal. Preserve any cancellation, then use the same
                # authoritative resource inspection as a concurrent-removal
                # handoff before accepting this private rollback.
                no_hosted_cleanup = True
                cancelled = cancelled or no_hosted_cancellation
            else:
                if isinstance(
                    failure,
                    (
                        RuntimeOffboardingRetainedError,
                        RuntimeOffboardingNotPerformedError,
                    ),
                ):
                    admission.rollback_incomplete = True
                raise failure
        inspection = asyncio.create_task(
            self._child_runtime_or_delegated_hold_is_live(admission.name),
            name=f"rollback_uncommitted_spawn_inspect:{admission.name}",
        )
        inspection_cancelled, inspection_failure = (
            await await_lifecycle_task_completion(inspection)
        )
        if inspection_failure is not None:
            raise inspection_failure
        child_live, hold_live = inspection.result()
        if child_live or hold_live:
            admission.rollback_incomplete = True
            live_resources = ", ".join(
                resource
                for resource, is_live in (
                    ("routable child", child_live),
                    ("delegated budget hold", hold_live),
                )
                if is_live
            )
            raise RuntimeError(
                "Rollback of uncommitted spawn "
                f"{admission.name!r} did not remove its live {live_resources}"
            )
        child_id = _loaded_agent_did(child)
        if not isinstance(child_id, str) or not child_id:
            raise RuntimeError(
                "Rejected spawned candidate has no stable identity to retire"
            )
        # A storage-backed spawn has no hosted namespace for destructive
        # offboarding. Its primary database remains discoverable by design, so
        # rollback is not terminal until that exact DID is durably tombstoned.
        # Hosted spawns normally removed their namespace already; the helper is
        # a no-op when no discoverable primary database remains.
        self.record_expired_spawn_retirement(
            admission.name,
            expected_child_did=child_id,
        )
        outcomes = []
        if descendant_cancelled:
            outcomes.append(asyncio.CancelledError())
        if descendant_failure is not None:
            outcomes.append(descendant_failure)
        _raise_lifecycle_outcomes(
            "Uncommitted spawn descendant rollback failed",
            outcomes,
        )
        return cancelled or inspection_cancelled

    async def _handoff_failed_spawn_cleanup(
        self,
        admission: AgentOperationAdmission,
        child: KestrelAgent,
    ) -> bool:
        """Retain an ambiguous receipt and its live storage until revocation.

        This is the last-resort owner after a prepublication receipt write or a
        published spawn's later governance commit failed. It deliberately
        retries the idempotent restrictive replacement before withdrawing the
        exact runtime through the rollback path appropriate to its publication
        state. A durable backend outage remains visible in the quarantine
        registry and keeps the routing name reserved instead of abandoning
        signed power.

        The manager state lock is the admission boundary shared with the drain
        seal. A drain joins already-admitted spawns before sealing, so this
        spawn must not reacquire the drain lock that its joining owner holds.
        A defensive gate handles any already-sealed direct drain without
        creating an unregistered task behind its snapshot. A spawn cap slot
        transfers to the retained task at the same boundary and is released
        only after the child runtime and delegated hold are both gone.

        Returns whether cancellation was observed while completing admission.
        """

        slot_transferred = False

        async def revoke_then_shutdown() -> None:
            while admission.spawn_receipt_graph is not None:
                try:
                    await self._downgrade_uncommitted_spawn_receipt(
                        admission,
                        child,
                    )
                except Exception:
                    logger.exception(
                        "Failed spawn receipt revocation remains pending for %r",
                        admission.name,
                    )
                    await asyncio.sleep(1.0)
            while (
                admission.spawn_startup_config is not None
                or admission.spawn_startup_config_path is not None
            ):
                try:
                    await self._withdraw_uncommitted_spawn_startup_registration(
                        admission
                    )
                except Exception:
                    logger.exception(
                        "Failed spawn startup-roster withdrawal remains pending "
                        "for %r",
                        admission.name,
                    )
                    await asyncio.sleep(1.0)
            while True:
                try:
                    await self._rollback_uncommitted_spawn_runtime(
                        admission,
                        child,
                    )
                except Exception:
                    logger.exception(
                        "Failed spawn runtime cleanup remains pending for %r",
                        admission.name,
                    )
                    await asyncio.sleep(1.0)
                    continue
                break
            while (
                admission.spawn_authority_witness_did is not None
                or admission.spawn_authority_witness_mandate is not None
            ):
                try:
                    await self._withdraw_uncommitted_spawn_authority_witness(
                        admission
                    )
                except Exception:
                    logger.exception(
                        "Failed spawn authority-witness settlement remains pending "
                        "for %r",
                        admission.name,
                    )
                    await asyncio.sleep(1.0)

            if slot_transferred:
                async with self._lock:
                    if self._pending_spawns <= 0:
                        raise RuntimeError(
                            "Quarantined spawn cap ownership underflowed"
                        )
                    self._pending_spawns -= 1
                    self._resolve_spawn_slot_terminal(admission)

        cancelled = False
        cleanup: asyncio.Task[None] | None = None
        admission_failure: BaseException | None = None
        while True:
            acquire = asyncio.create_task(
                self._lock.acquire(),
                name=f"failed_spawn_cleanup_admission:{admission.name}",
            )
            acquire_cancelled, acquire_failure = (
                await await_lifecycle_task_completion(acquire)
            )
            cancelled = cancelled or acquire_cancelled
            if acquire_failure is not None:
                raise RuntimeError(
                    "Unable to admit failed spawn cleanup"
                ) from acquire_failure
            wait_for_open: asyncio.Event | None = None
            try:
                if self._quarantined_shutdown_handoffs_sealed:
                    wait_for_open = self._quarantined_shutdown_handoffs_open
                else:
                    cleanup = asyncio.create_task(
                        revoke_then_shutdown(),
                        name=f"failed_spawn_receipt_cleanup:{admission.name}",
                    )
                    try:
                        self._retain_quarantined_cleanup(
                            name=admission.name,
                            agent_id=_loaded_agent_did(child) or "<unknown>",
                            task=cleanup,
                        )
                    except BaseException as exc:
                        # The task has not had an event-loop turn while this
                        # code holds the lock. Cancel it before leaving the
                        # admission boundary so an unregistered cleanup can
                        # never escape.
                        cleanup.cancel()
                        admission_failure = exc
                    else:
                        if admission.spawn_slot_active:
                            slot_transferred = True
                            admission.spawn_slot_active = False
                        else:
                            # Governance commit may already have converted the
                            # pending reservation into a projected mandate.
                            # Receipt downgrade removes that projection, so a
                            # still-live cleanup needs one replacement slot.
                            self._pending_spawns += 1
                            slot_transferred = True
            finally:
                self._lock.release()
            if wait_for_open is None:
                break
            wait = asyncio.create_task(
                wait_for_open.wait(),
                name=f"failed_spawn_cleanup_wait_for_drain:{admission.name}",
            )
            wait_cancelled, wait_failure = (
                await await_lifecycle_task_completion(wait)
            )
            cancelled = cancelled or wait_cancelled
            if wait_failure is not None:
                raise RuntimeError(
                    "Unable to wait for failed spawn cleanup admission"
                ) from wait_failure

        if admission_failure is not None:
            assert cleanup is not None
            cleanup_cancelled, cleanup_failure = (
                await await_lifecycle_task_completion(cleanup)
            )
            cancelled = cancelled or cleanup_cancelled
            if cleanup_failure is not None and not isinstance(
                cleanup_failure, asyncio.CancelledError
            ):
                raise BaseExceptionGroup(
                    "Failed cleanup admission left multiple terminal outcomes",
                    [admission_failure, cleanup_failure],
                )
            raise admission_failure
        return cancelled

    async def _child_runtime_or_delegated_hold_is_live(
        self, child_name: str
    ) -> tuple[bool, bool]:
        """Read the authoritative child routing and delegated-hold state."""

        async with self._lock:
            return (
                self._agents.get(child_name) is not None,
                child_name in self._child_budgets,
            )

    def _prune_child_relationship_and_mandate(
        self, parent_did: str, child_name: str
    ) -> None:
        """Forget one fully removed child from parent spawn-cap bookkeeping."""

        mandate = self._child_mandates.get(child_name)
        from kestrel_sovereign.spawn.lifecycle import SpawnedAgentLifecycle

        lifecycle = getattr(self, "_lifecycle", None)
        if isinstance(lifecycle, SpawnedAgentLifecycle):
            lifecycle.retire_persisted_child(
                child_name,
                expected_child_did=(
                    mandate.child_did if isinstance(mandate, SpawnMandate) else None
                ),
            )
        children = self._parent_children.get(parent_did)
        if children is not None:
            try:
                children.remove(child_name)
            except ValueError:
                pass
            if not children:
                self._parent_children.pop(parent_did, None)
        self._child_mandates.pop(child_name, None)

    async def _prune_child_tracking_if_fully_removed(
        self, parent_did: str, child_name: str
    ) -> bool:
        """Prune a parent edge only after every child cleanup owner is gone."""

        async with self._lock:
            child_live = self._agents.get(child_name) is not None
            hold_live = child_name in self._child_budgets
            quarantined_cleanup_live = self._quarantined_cleanup_name_is_reserved(
                self._canonical_agent_name(child_name)
            )
            if child_live or hold_live or quarantined_cleanup_live:
                return False
            self._prune_child_relationship_and_mandate(parent_did, child_name)
            return True

    def _record_destructive_child_retirement(
        self,
        child_name: str,
        *,
        child_id: Optional[str],
        startup_registration: Optional[tuple[Path, str, LocalAgentConfig]],
    ) -> None:
        """Deny rediscovery after a destructive removal reaches terminal state."""

        if not isinstance(child_id, str) or not child_id:
            return
        retirement_config = (
            startup_registration[2]
            if startup_registration is not None
            else self._created_configs.get(child_name)
        )
        self.record_expired_spawn_retirement(
            child_name,
            expected_child_did=child_id,
            data_dir=(
                retirement_config.resolve_data_dir(self._base_data_dir)
                if isinstance(retirement_config, LocalAgentConfig)
                else None
            ),
        )
        self._created_configs.pop(child_name, None)

    async def await_child_termination_reconciliation(
        self,
        parent_did: str,
        child_name: str,
    ) -> tuple[bool, bool]:
        """Join this child's deferred cleanup before terminal retirement.

        ``terminate_child`` is intentionally bounded and may return after a
        slow shutdown or delegated refund has moved to quarantine.  A TTL
        owner has a stricter boundary: it must not publish the durable
        retirement marker until every cleanup owner for that exact child has
        settled and the parent edge is actually pruned.  Join only matching
        reapers, preserve cancellation until ownership is reconciled, and
        leave failed cleanup fail-closed in the quarantine reservation.

        Returns ``(reconciled, cancellation_observed)``.
        """

        if not isinstance(parent_did, str) or not parent_did:
            raise TypeError("parent DID must be a non-empty string")
        if not isinstance(child_name, str) or not child_name:
            raise TypeError("child name must be a non-empty string")
        retained_name = _bounded_shutdown_metadata(
            self._canonical_agent_name(child_name)
        )
        cancelled = False
        cleanup_failed = False

        while True:
            records = tuple(
                record
                for record in self._quarantined_shutdown_reapers.values()
                if record.canonical_agent_name == retained_name
            )
            if not records:
                break
            for record in records:
                join_cancelled, failure = await await_lifecycle_task_completion(
                    record.task
                )
                cancelled = cancelled or join_cancelled
                if failure is not None:
                    cleanup_failed = True

            # Completion callbacks collapse live tasks into bounded history or
            # unsafe evidence. Yield through the cancellation-safe join so the
            # next ownership snapshot cannot spin on a completed live record.
            callback_yield = asyncio.create_task(asyncio.sleep(0))
            yield_cancelled, yield_failure = await await_lifecycle_task_completion(
                callback_yield
            )
            cancelled = cancelled or yield_cancelled
            if yield_failure is not None:
                raise RuntimeError(
                    "Unable to observe quarantined child cleanup completion"
                ) from yield_failure
            if cleanup_failed:
                logger.error(
                    "Deferred cleanup failed for child %r; TTL retirement "
                    "remains reserved for operator reconciliation",
                    child_name,
                )
                return False, cancelled

        reconciliation = asyncio.create_task(
            self._prune_child_tracking_if_fully_removed(parent_did, child_name),
            name=f"ttl_termination_reconcile:{child_name}",
        )
        reconciliation_cancelled, reconciliation_failure = (
            await await_lifecycle_task_completion(reconciliation)
        )
        cancelled = cancelled or reconciliation_cancelled
        if reconciliation_failure is not None:
            raise RuntimeError(
                "Unable to reconcile deferred child cleanup"
            ) from reconciliation_failure
        return bool(reconciliation.result()), cancelled

    async def _prune_all_fully_removed_child_tracking(self) -> None:
        """Reconcile completed child removals at a serialized lifecycle boundary.

        A bounded ``remove_agent`` may return after handing shutdown and/or a
        delegated refund to quarantine.  Its parent edge is deliberately kept
        while that reaper owns the name. Direct removal and the terminal drain
        both call this helper while holding the A2A lifecycle boundary, which
        excludes same-name admission between this check and pruning.
        """

        async with self._lock:
            for parent_did, children in tuple(self._parent_children.items()):
                for child_name in tuple(children):
                    child_live = self._agents.get(child_name) is not None
                    hold_live = child_name in self._child_budgets
                    quarantined_cleanup_live = (
                        self._quarantined_cleanup_name_is_reserved(
                            self._canonical_agent_name(child_name)
                        )
                    )
                    if not (
                        child_live or hold_live or quarantined_cleanup_live
                    ):
                        self._prune_child_relationship_and_mandate(
                            parent_did, child_name
                        )

    async def _reconcile_fully_removed_child_tracking(self) -> bool:
        """Own child-tracking reconciliation through repeated cancellation."""

        reconciliation = asyncio.create_task(
            self._prune_all_fully_removed_child_tracking(),
            name="agent_manager:reconcile_removed_child_tracking",
        )
        cancelled, failure = await await_lifecycle_task_completion(reconciliation)
        if failure is not None:
            raise failure
        return cancelled

    async def _join_admitted_spawn_operations(
        self,
    ) -> tuple[bool, list[BaseException]]:
        """Join every spawn admitted before the shutdown registration fence.

        A plain create/load has no child hold or mandate to commit after the
        fleet sweep; its captured epoch prevents publication once shutdown
        reopens for later operations.  A spawn is different: it can resume from
        an already-published child into budget/mandate commit, so shutdown joins
        that full operation before it snapshots live routing.  The spawn's own
        failure is not independently a fleet failure: surviving runtime state
        is what the subsequent sweep authoritatively reports.
        """

        cancelled = False
        failures: list[BaseException] = []
        observed: set[str] = set()
        current_task = asyncio.current_task()
        while True:
            async with self._lock:
                operations = tuple(
                    operation
                    for operation in self._agent_operations.values()
                    if operation.kind in {"spawn", "direct-spawn-test"}
                    and operation.spawn_task is not None
                    and operation.spawn_task is not current_task
                )
            pending = [
                operation
                for operation in operations
                if operation.canonical_name not in observed
            ]
            if not pending:
                return cancelled, failures
            for operation in pending:
                observed.add(operation.canonical_name)
                task = operation.spawn_task
                assert task is not None
                join_cancelled, failure = await await_lifecycle_task_completion(
                    task
                )
                cancelled = cancelled or join_cancelled
                # A spawn owns its rollback until this join.  Ordinary spawn
                # failures are represented by the authoritative state swept
                # below, but a BaseExceptionGroup preserves cancellation plus
                # a rollback failure and cannot be allowed to abort that
                # sweep.  Defer it so the terminal report retains both facts
                # after every live agent has received cleanup.
                if isinstance(failure, BaseExceptionGroup) and not isinstance(
                    failure, Exception
                ):
                    expected, unexpected = failure.split(
                        (asyncio.CancelledError, Exception)
                    )
                    if unexpected is not None:
                        raise unexpected
                    assert expected is not None
                    failures.append(expected)
                elif failure is not None and not isinstance(
                    failure, (Exception, asyncio.CancelledError)
                ):
                    raise failure

    async def begin_terminal_descendant_spawn_fence(
        self,
        parent_did: str,
    ) -> TerminalDescendantSpawnFence:
        """Join earlier child spawns and exclude later ones until settlement.

        ``spawn_parent`` is attached to an admission before its task becomes
        runnable, so the manager-lock snapshot sees every operation admitted
        before this fence.  The spawn cap admission checks the same lock and
        refuses operations admitted afterward.
        """

        if not isinstance(parent_did, str) or not parent_did:
            raise ValueError("terminal descendant spawn fence requires a parent DID")
        async with self._lock:
            self._next_terminal_descendant_spawn_fence_nonce += 1
            token = TerminalDescendantSpawnFence(
                parent_did=parent_did,
                nonce=self._next_terminal_descendant_spawn_fence_nonce,
            )
            self._terminal_descendant_spawn_fence_members[token.nonce] = set()

        cancelled = False
        joined_spawns: set[asyncio.Future[object]] = set()
        try:
            while True:
                async with self._lock:
                    members = self._terminal_descendant_spawn_fence_members[token.nonce]
                    durable_by_parent: dict[
                        str, dict[str, tuple[str, SpawnMandate]]
                    ] = {}
                    for witness in self.unsettled_spawn_authority_witnesses():
                        if (
                            not isinstance(witness.child_name, str)
                            or not witness.child_name
                            or not isinstance(witness.child_did, str)
                            or not witness.child_did
                            or not isinstance(witness.parent_did, str)
                            or not witness.parent_did
                            or not isinstance(witness.mandate, SpawnMandate)
                            or witness.mandate.child_did != witness.child_did
                            or witness.mandate.parent_did != witness.parent_did
                        ):
                            raise RuntimeError(
                                "Refusing terminal descendant fence with invalid "
                                "durable authority"
                            )
                        siblings = durable_by_parent.setdefault(
                            witness.parent_did,
                            {},
                        )
                        previous = siblings.get(witness.child_did)
                        if previous is not None and (
                            self._canonical_agent_name(previous[0])
                            != self._canonical_agent_name(witness.child_name)
                        ):
                            raise RuntimeError(
                                "Refusing terminal descendant fence with conflicting "
                                "durable authority"
                            )
                        siblings[witness.child_did] = (
                            witness.child_name,
                            witness.mandate,
                        )
                    discovered = {parent_did}
                    discovered_names: dict[str, str] = {}
                    pending = [parent_did]
                    while pending:
                        descendant_parent_did = pending.pop()
                        candidates = dict(
                            durable_by_parent.get(descendant_parent_did, {})
                        )
                        for child_name in self._parent_children.get(
                            descendant_parent_did, ()
                        ):
                            mandate = self._child_mandates.get(child_name)
                            child_did = (
                                mandate.child_did
                                if isinstance(mandate, SpawnMandate)
                                else None
                            )
                            if not isinstance(child_did, str) or not child_did:
                                child = self._get_agent_for_lifecycle(child_name)
                                child_did = _loaded_agent_did(child)
                            if not isinstance(child_did, str) or not child_did:
                                raise RuntimeError(
                                    "Refusing terminal descendant fence without "
                                    "stable descendant authority"
                                )
                            durable = candidates.get(child_did)
                            if durable is not None and (
                                self._canonical_agent_name(durable[0])
                                != self._canonical_agent_name(child_name)
                            ):
                                raise RuntimeError(
                                    "Refusing terminal descendant fence with "
                                    "conflicting authority"
                                )
                            candidates[child_did] = (child_name, mandate)
                        for child_did, (child_name, _mandate) in candidates.items():
                            known_did = discovered_names.get(
                                self._canonical_agent_name(child_name)
                            )
                            if known_did is not None and known_did != child_did:
                                raise RuntimeError(
                                    "Refusing terminal descendant fence with "
                                    "conflicting authority"
                                )
                            if child_did in discovered:
                                raise RuntimeError(
                                    "Refusing terminal descendant fence with "
                                    "cyclic authority"
                                )
                            discovered_names[
                                self._canonical_agent_name(child_name)
                            ] = child_did
                            discovered.add(child_did)
                            pending.append(child_did)
                    for discovered_did in discovered - members:
                        self._terminal_descendant_spawn_fences.setdefault(
                            discovered_did, set()
                        ).add(token.nonce)
                    members.update(discovered)
                    current_task = asyncio.current_task()
                    earlier_spawns = tuple(
                        admission.spawn_task
                        for admission in self._agent_operations.values()
                        if admission.kind in {"spawn", "direct-spawn-test"}
                        and admission.spawn_task is not None
                        and admission.spawn_task is not current_task
                        and admission.spawn_task not in joined_spawns
                        and admission.spawn_parent is not None
                        and bool(
                            members.intersection(
                                _loaded_agent_bound_dids(admission.spawn_parent)
                            )
                        )
                    )
                if not earlier_spawns:
                    break
                for spawn_task in earlier_spawns:
                    joined_spawns.add(spawn_task)
                    task_cancelled, _failure = await await_lifecycle_task_completion(
                        spawn_task
                    )
                    cancelled = cancelled or task_cancelled
        except BaseException:
            self.end_terminal_descendant_spawn_fence(token)
            raise
        if cancelled:
            self.end_terminal_descendant_spawn_fence(token)
            raise asyncio.CancelledError()
        return token

    def end_terminal_descendant_spawn_fence(
        self,
        token: TerminalDescendantSpawnFence,
    ) -> None:
        """Release exactly one terminal descendant-spawn exclusion lease."""

        if not isinstance(token, TerminalDescendantSpawnFence):
            raise TypeError("terminal descendant spawn fence token is invalid")
        members = self._terminal_descendant_spawn_fence_members.pop(token.nonce, None)
        if members is None:
            raise RuntimeError("terminal descendant spawn fence is not active")
        for parent_did in members:
            owners = self._terminal_descendant_spawn_fences.get(parent_did)
            if owners is None or token.nonce not in owners:
                raise RuntimeError("terminal descendant spawn fence is corrupt")
            owners.remove(token.nonce)
            if not owners:
                self._terminal_descendant_spawn_fences.pop(parent_did, None)

    def get_children(self, parent_did: str) -> list[str]:
        """Get list of child agent names for a parent DID."""
        return list(self._parent_children.get(parent_did, []))

    def get_mandate(self, child_name: str) -> Optional[SpawnMandate]:
        """Get the SpawnMandate for a child agent."""
        return self._child_mandates.get(child_name)

    async def terminate_child(
        self,
        parent_did: str,
        child_name: str,
        *,
        offboard_runtime: bool = False,
    ) -> bool:
        """Terminate a specific child agent and its descendants.

        Removes the child from the parent-child map, terminates any
        grandchildren (cascading), then shuts down the child itself.

        Args:
            parent_did: DID of the parent agent.
            child_name: Name of the child to terminate.
            offboard_runtime: Explicit destructive intent. The compatibility
                default stops/unpublishes the child while retaining its
                isolated runtime for restart; only an approved deprovisioning
                path may set this true.

        Returns:
            True if the child was found and terminated.
        """
        children = self._parent_children.get(parent_did, [])
        if child_name not in children:
            return False
        if type(offboard_runtime) is not bool:
            raise TypeError("offboard_runtime must be a bool")

        terminal_outcomes: list[BaseException] = []

        # Cascade: terminate grandchildren first. A retained runtime tree is a
        # terminal offboarding outcome, not permission to abandon later
        # siblings or the now-stopped parent. Preserve it for the caller after
        # every reachable lifecycle target has received its teardown attempt.
        child_agent = self.get_agent(child_name)
        child_id = _loaded_agent_did(child_agent) if child_agent is not None else None
        if not isinstance(child_id, str) or not child_id:
            mandate = self._child_mandates.get(child_name)
            child_id = (
                mandate.child_did
                if isinstance(mandate, SpawnMandate)
                and isinstance(mandate.child_did, str)
                and mandate.child_did
                else None
            )
        if child_agent is not None:
            try:
                if offboard_runtime:
                    await self.terminate_children(
                        child_agent.agent_id,
                        offboard_runtime=True,
                    )
                else:
                    await self.terminate_children(child_agent.agent_id)
            except BaseException as exc:
                if not _is_lifecycle_terminal_outcome(exc):
                    raise
                terminal_outcomes.append(exc)

        # NB: the child's own budget hold is released inside remove_agent below
        # (stop-then-release), after the cascade above has already stopped and
        # released every descendant — so refunds flow up leaf-first (#2113).

        # Shutdown the child before mutating retry bookkeeping.  A false
        # removal result or a failed/cancelled refund leaves either routing or
        # its exact delegated hold live; keeping the relationship and mandate
        # lets the ordinary terminate_child path retry safely.  ``remove_agent``
        # can also re-raise cancellation only after its owned shutdown/refund
        # tail settled.  In that terminal case reconcile the authoritative
        # state before propagating cancellation, or the removed child keeps a
        # stale mandate that consumes a spawn-cap slot forever.
        startup_registration = None
        offboarding_admission = None
        destructive_spawn_retirement = None
        if offboard_runtime:
            if (
                self._startup_config_path is not None
                and (not isinstance(child_id, str) or not child_id)
            ):
                raise RuntimeError(
                    "Cannot destructively terminate a child without a stable identity"
                )
            if isinstance(child_id, str) and child_id:
                destructive_spawn_retirement = (
                    await self._begin_destructive_spawn_retirement(
                        child_name,
                        known_agent_id=child_id,
                    )
                )
                try:
                    startup_registration = (
                        self._withdraw_committed_spawn_startup_registration(
                            child_name,
                            expected_child_did=child_id,
                        )
                    )
                except Exception:
                    await self._settle_destructive_spawn_retirement(
                        destructive_spawn_retirement
                    )
                    raise
            if startup_registration is not None:
                offboarding_admission = RuntimeOffboardingAdmission()

        removed = False
        try:
            retirement_kwargs = (
                {}
                if destructive_spawn_retirement is None
                else {
                    "_prestarted_spawn_retirement": destructive_spawn_retirement
                }
            )
            if offboarding_admission is not None:
                removed = await self.remove_agent(
                    child_name,
                    offboard_runtime=offboard_runtime,
                    offboarding_admission=offboarding_admission,
                    **retirement_kwargs,
                )
            else:
                removed = await self.remove_agent(
                    child_name,
                    offboard_runtime=offboard_runtime,
                    **retirement_kwargs,
                )
        except BaseException as exc:
            if (
                startup_registration is not None
                and offboarding_admission is not None
                and not offboarding_admission.started
            ):
                self._restore_committed_spawn_startup_registration(
                    startup_registration
                )
            if not _is_lifecycle_terminal_outcome(exc):
                raise
            retirement_settlement = asyncio.create_task(
                self._settle_destructive_spawn_retirement(
                    destructive_spawn_retirement
                ),
                name=f"terminate_child_retirement_settle:{child_name}",
            )
            (
                retirement_cancelled,
                retirement_failure,
            ) = await await_lifecycle_task_completion(retirement_settlement)
            if retirement_failure is not None:
                terminal_outcomes.append(retirement_failure)
            if retirement_cancelled:
                terminal_outcomes.append(asyncio.CancelledError())
            reconciliation = asyncio.create_task(
                self._prune_child_tracking_if_fully_removed(parent_did, child_name),
                name=f"terminate_child_terminal_reconcile:{child_name}",
            )
            (
                reconciliation_cancelled,
                reconciliation_failure,
            ) = await await_lifecycle_task_completion(reconciliation)
            if reconciliation_failure is not None:
                if not isinstance(reconciliation_failure, Exception):
                    raise reconciliation_failure
                logger.error(
                    "Unable to reconcile terminal termination of child %r",
                    child_name,
                    exc_info=(
                        type(reconciliation_failure),
                        reconciliation_failure,
                        reconciliation_failure.__traceback__,
                    ),
                )
                terminal_outcomes.append(
                    ChildTerminationReconciliationError(
                        child_name=child_name,
                        cause=reconciliation_failure,
                    )
                )
            if reconciliation_cancelled:
                terminal_outcomes.append(asyncio.CancelledError())
            if (
                offboard_runtime
                and reconciliation_failure is None
                and reconciliation.result()
            ):
                try:
                    self._record_destructive_child_retirement(
                        child_name,
                        child_id=child_id,
                        startup_registration=startup_registration,
                    )
                except Exception as retirement_failure:
                    terminal_outcomes.append(retirement_failure)
            terminal_outcomes.append(exc)
            _raise_lifecycle_outcomes(
                f"Child {child_name!r} termination had terminal failures",
                terminal_outcomes,
            )
        if not removed:
            if (
                startup_registration is not None
                and offboarding_admission is not None
                and not offboarding_admission.started
            ):
                self._restore_committed_spawn_startup_registration(
                    startup_registration
                )
            if terminal_outcomes:
                terminal_outcomes.append(
                    ChildTerminationNotPerformedError(child_name=child_name)
                )
            _raise_lifecycle_outcomes(
                f"Child {child_name!r} descendant termination failed",
                terminal_outcomes,
            )
            return False

        # ``True`` means routing has been withdrawn, not necessarily that a
        # timeout/cancellation-resistant shutdown or fenced refund is done:
        # remove_agent may have handed either to quarantine.  Keep the parent
        # edge and mandate until the authoritative maps *and* the per-name
        # reaper reservation prove the complete removal.  A terminal drain
        # performs the same reconciliation after it joins the reaper, so a
        # late failed refund that restores its hold cannot race this pruning.
        reconciliation = asyncio.create_task(
            self._prune_child_tracking_if_fully_removed(parent_did, child_name),
            name=f"terminate_child_reconcile:{child_name}",
        )
        (
            reconciliation_cancelled,
            reconciliation_failure,
        ) = await await_lifecycle_task_completion(reconciliation)
        if reconciliation_failure is not None:
            if not isinstance(reconciliation_failure, Exception):
                raise reconciliation_failure
            logger.error(
                "Unable to reconcile completed termination of child %r",
                child_name,
                exc_info=(
                    type(reconciliation_failure),
                    reconciliation_failure,
                    reconciliation_failure.__traceback__,
                ),
            )
            terminal_outcomes.append(
                ChildTerminationReconciliationError(
                    child_name=child_name,
                    cause=reconciliation_failure,
                )
            )
        if reconciliation_cancelled:
            terminal_outcomes.append(asyncio.CancelledError())
        if (
            offboard_runtime
            and reconciliation_failure is None
            and reconciliation.result()
        ):
            self._record_destructive_child_retirement(
                child_name,
                child_id=child_id,
                startup_registration=startup_registration,
            )
        if reconciliation_failure is None and reconciliation.result():
            logger.info(
                f"Terminated child '{child_name}' of parent '{parent_did[:30]}...'"
            )
        else:
            logger.info(
                "Child %r routing was withdrawn, but its retry tracking remains "
                "owned by delegated cleanup.",
                child_name,
            )
        _raise_lifecycle_outcomes(
            f"Child {child_name!r} descendant termination failed",
            terminal_outcomes,
        )
        return True

    async def terminate_children(
        self,
        parent_did: str,
        *,
        offboard_runtime: bool = False,
    ) -> int:
        """Terminate all children of a parent agent (cascading).

        Args:
            parent_did: DID of the parent whose children to terminate.
            offboard_runtime: Whether to securely delete each descendant's
                hosted runtime after shutdown. Defaults to retention.

        Returns:
            Number of children terminated.
        """
        children = list(self._parent_children.get(parent_did, []))
        if type(offboard_runtime) is not bool:
            raise TypeError("offboard_runtime must be a bool")
        count = 0
        terminal_outcomes: list[BaseException] = []
        for child_name in children:
            try:
                if offboard_runtime:
                    terminated = await self.terminate_child(
                        parent_did,
                        child_name,
                        offboard_runtime=True,
                    )
                else:
                    terminated = await self.terminate_child(parent_did, child_name)
                if terminated:
                    count += 1
            except BaseException as exc:
                if not _is_lifecycle_terminal_outcome(exc):
                    raise
                terminal_outcomes.append(exc)
        _raise_lifecycle_outcomes(
            f"One or more children of {parent_did!r} had terminal failures",
            terminal_outcomes,
        )
        return count

    async def shutdown_all(self) -> None:
        """Gracefully shut down every agent, then report any terminal failures.

        Fleet teardown is an ownership boundary: one agent's failed durable
        continuation must never strand a later agent's dispatcher or storage.
        Keep successful removals reflected in the relationship maps, retain
        failed agents as published, and surface all ordinary failures only once
        every candidate has received its shutdown attempt.
        """
        # A fleet shutdown and a direct terminal drain have the same ownership
        # boundary.  Own that boundary *before* attempting removal: otherwise a
        # drain can seal between this sweep's snapshot and its first DELETE,
        # making a live agent look like a false ``remove_agent(False)`` failure.
        # This also serializes concurrent fleet shutdowns, so the second owner
        # takes a fresh snapshot after the first has finished every live agent.
        cancelled = await self._acquire_quarantined_shutdown_drain()
        try:
            self._seal_agent_registration_for_shutdown_all()
            await self._shutdown_all_while_drain_locked(cancelled=cancelled)
        finally:
            self._reopen_agent_registration_after_shutdown_all()
            self._quarantined_shutdown_drain_lock.release()

    async def _shutdown_all_while_drain_locked(self, *, cancelled: bool) -> None:
        """Perform one fleet sweep while owning the terminal drain boundary."""

        # Stop + release budgeted children leaf-first (#2113): reverse insertion
        # order is leaf-first (a descendant is always spawned after its ancestor),
        # so each is quiesced and its unspent hold refunded UP into its (budgeted)
        # parent before that parent is released to the root. remove_agent does the
        # stop-then-release. (A follow-up covers durable reconciliation across an
        # *ungraceful* crash.)
        # Fence first, then join each pre-fence spawn through its full create ->
        # budget -> mandate commit/rollback boundary.  Plain create/load work is
        # publication-fenced by its captured epoch and remains intentionally
        # bounded by its own initializer; waiting for arbitrary provider or
        # inception I/O here would turn a normal fleet shutdown into a hang.
        spawn_join_cancelled, spawn_failures = (
            await self._join_admitted_spawn_operations()
        )
        cancelled = spawn_join_cancelled or cancelled
        failures: list[BaseException] = list(spawn_failures)
        removed_names: set[str] = set()
        attempted_names: set[str] = set()
        reported_budget_release_failures: set[str] = set()

        def fully_removed(name: str) -> bool:
            """Whether neither a routable agent nor a delegated hold remains."""
            return name not in self._agents and name not in self._child_budgets

        def record_failure(name: str, failure: BaseException) -> None:
            failures.append(failure)
            logger.warning("Could not completely shut down agent %r: %s", name, failure)

        async def attempt_removal(name: str, *, unpublished_hold: bool = False) -> None:
            """Attempt one removal without allowing it to abort the fleet sweep."""
            nonlocal cancelled
            attempted_names.add(name)
            failure_recorded = False
            admitted_budget_release: Optional[InflightRemovalBudgetRelease] = None
            release_ids_before_attempt = self._removal_budget_release_ids_for_child(
                name
            )
            try:
                if unpublished_hold:
                    async with self._lock:
                        admitted_budget_release = self._start_child_budget_release(name)
                    release_cancelled = await self._await_child_budget_release_cancellation_safe(
                        admitted_budget_release.task
                    )
                    cancelled = cancelled or release_cancelled
                    removed = fully_removed(name)
                else:
                    removed = await self.remove_agent(
                        name,
                        offboard_runtime=False,
                    )
            except asyncio.CancelledError:
                # The single-agent primitive completes its durable tail before
                # propagating cancellation.  Continue sweeping later agents;
                # only prune this name when the authoritative maps confirm it.
                cancelled = True
                removed = fully_removed(name)
                if not removed:
                    record_failure(
                        name,
                        RuntimeError("removal was interrupted before completion"),
                    )
                    failure_recorded = True
            except BaseExceptionGroup as exc:
                # A lifecycle owner can retain cancellation and an ordinary
                # cleanup failure in one group.  As with admitted spawns,
                # report that terminal evidence only after every later fleet
                # member has received its own removal attempt.
                expected, unexpected = exc.split(
                    (asyncio.CancelledError, Exception)
                )
                if unexpected is not None:
                    raise unexpected
                assert expected is not None
                removed = fully_removed(name)
                record_failure(name, expected)
                failure_recorded = True
                if admitted_budget_release is not None:
                    reported_budget_release_failures.add(
                        admitted_budget_release.release_id
                    )
                else:
                    reported_budget_release_failures.update(
                        self._removal_budget_release_ids_for_child(name)
                        - release_ids_before_attempt
                    )
            except Exception as exc:
                removed = fully_removed(name)
                record_failure(name, exc)
                failure_recorded = True
                if admitted_budget_release is not None:
                    # Keep its retained unsafe metadata for a later terminal
                    # drain, but this fleet shutdown already names the exact
                    # release failure above.
                    reported_budget_release_failures.add(
                        admitted_budget_release.release_id
                    )
                else:
                    # A published agent admits its release inside
                    # ``remove_agent``. Attribute only evidence created by
                    # this attempt, leaving pre-existing unsafe records for
                    # the terminal drain to report.
                    reported_budget_release_failures.update(
                        self._removal_budget_release_ids_for_child(name)
                        - release_ids_before_attempt
                    )

            # A concurrent DELETE can complete the snapshotted target while
            # this sweep is waiting on its per-DID lifecycle writer. Its later
            # ``False`` means there was nothing left for *this* caller to do,
            # not that fleet teardown failed. Only accept that result when the
            # authoritative maps prove both routing and the delegated hold are
            # gone; a live agent or retained hold remains a real failure.
            if not removed:
                removed = fully_removed(name)

            if removed:
                removed_names.add(name)
            elif not failure_recorded:
                record_failure(
                    name,
                    RuntimeError("remove_agent returned False; agent remains published"),
                )

        for child_name in reversed(list(self._child_budgets.keys())):
            # A partially completed spawn/boot can leave a delegated hold
            # without ever publishing its agent.  It has no live process to
            # stop, but the leaf-first refund is still required before its
            # parent's hold may be released.
            await attempt_removal(
                child_name,
                unpublished_hold=child_name not in self._agents,
            )

        # A failed child remains published and must not be retried as an
        # unrelated root agent.  Every other agent still receives one attempt.
        names = [name for name in self._agents if name not in attempted_names]
        for name in names:
            await attempt_removal(name)

        # ``remove_agent`` is intentionally allowed to return once it has
        # quarantined cancellation-resistant cleanup.  Fleet/server shutdown
        # is the terminal lifecycle owner, however: do not let a following
        # server teardown close the event loop or shared storage while one of
        # those retained reapers is still releasing an owner or closing SQLite.
        try:
            cancelled = (
                await self._drain_quarantined_shutdowns_while_locked(
                    cancelled=False,
                    reported_budget_release_failures=reported_budget_release_failures
                )
            ) or cancelled
        except BaseExceptionGroup as exc:
            drain_failures, non_failures = exc.split(Exception)
            cancellation_group: BaseExceptionGroup | None = None
            if non_failures is not None:
                cancellation_group, unexpected = non_failures.split(
                    asyncio.CancelledError
                )
                if unexpected is not None:
                    raise unexpected
            cancelled = cancelled or cancellation_group is not None
            if drain_failures is not None:
                failures.append(drain_failures)
            logger.warning(
                "Quarantined agent shutdown cleanup retained cancellation or "
                "failure outcomes",
                exc_info=True,
            )
        except asyncio.CancelledError:
            cancelled = True
        except Exception as exc:
            failures.append(exc)
            logger.warning(
                "Quarantined agent shutdown cleanup did not complete safely: %s",
                exc,
                exc_info=True,
            )

        # A bounded DELETE may initially look fully removed after handing a
        # fenced refund to quarantine.  The terminal drain above can then
        # observe that refund fail and restore its exact hold.  Re-evaluate the
        # authoritative routing/hold state before pruning relationships: a
        # restored hold needs its mandate and parent edge for normal retry.
        removed_names = {
            name for name in removed_names if fully_removed(name)
        }

        # Only successful removals may disappear from relationship state. A
        # failed child stays visible to its parent and retains its mandate for a
        # future recovery/termination attempt.
        for child_name in removed_names:
            self._child_mandates.pop(child_name, None)
        for parent_did, children in list(self._parent_children.items()):
            remaining_children = [
                child_name for child_name in children if child_name not in removed_names
            ]
            if remaining_children:
                self._parent_children[parent_did] = remaining_children
            else:
                self._parent_children.pop(parent_did, None)

        remaining_agents = list(self._agents)
        remaining_holds = list(self._child_budgets)
        if remaining_agents or remaining_holds:
            logger.warning(
                "Fleet shutdown incomplete; agents still published=%s, "
                "delegated holds still active=%s",
                remaining_agents,
                remaining_holds,
            )
        else:
            logger.info("All agents shut down")

        if cancelled and failures:
            raise BaseExceptionGroup(
                "Fleet shutdown observed cancellation and agent failures",
                [asyncio.CancelledError(), *failures],
            )
        if cancelled:
            raise asyncio.CancelledError()
        if failures:
            raise BaseExceptionGroup(
                "One or more fleet agents failed to shut down", failures
            )
