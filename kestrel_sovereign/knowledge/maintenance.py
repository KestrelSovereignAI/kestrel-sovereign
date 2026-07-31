"""Incremental, restart-safe semantic maintenance after memory consolidation.

This module is intentionally a coordinator, not another assertion writer.  It
consumes the canonical change stream, invokes the existing governed SHACL and
inference services, and records only content-free operational evidence.  In
particular, contradiction detection creates review candidates; it never uses a
model or chooses which competing reported fact is true.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import logging
import math
from time import monotonic
from typing import TYPE_CHECKING, Mapping, Sequence
from uuid import uuid4

from kestrel_sovereign.storage.async_assertion_store import (
    AssertionChange,
    AssertionCheckpoint,
    MaintenanceLeaseLostError,
)
from kestrel_sovereign.storage.db.interface import TransactionError

from .assertion import Assertion, AssertionQuery, AssertionStatus, DirectLineage
from .inference import (
    BoundedInferenceService,
    ClosureStatus,
    InferenceLimits,
    InferenceProfile,
    InferenceReconciliationResult,
)
from .registry import (
    ArtifactPin,
    ExperimentalCapabilityError,
    KnowledgeRegistryError,
    ResourceKind,
    get_knowledge_registry,
)
from .shacl_validation import (
    ShapeSetReference,
    ShaclValidationLimits,
    ShaclValidationReport,
    ValidationState,
)

if TYPE_CHECKING:
    from kestrel_sovereign.storage.async_assertion_store import AsyncAssertionStore


logger = logging.getLogger(__name__)


class SemanticMaintenanceError(ValueError):
    """A semantic-maintenance configuration or invariant is invalid."""


class SemanticMaintenanceStatus(str, Enum):
    """Observable terminal states for one bounded maintenance attempt."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    FAILED = "failed"
    NO_OP = "no_op"


@dataclass(frozen=True, slots=True)
class SemanticMaintenanceTrainingReadiness:
    """Content-free durable prerequisite for a training-corpus consumer.

    ``ready`` is true only when the active semantic capability has a complete
    maintenance checkpoint at the current assertion cursor. Historical
    maintenance evidence is not an immutable training-corpus snapshot and
    therefore never makes stale data ready.
    """

    ready: bool
    reason: str | None
    using_prior_verified_snapshot: bool = False


@dataclass(frozen=True, slots=True)
class SemanticMaintenanceLimits:
    """Finite budgets for one incremental maintenance unit."""

    max_wall_time_seconds: float = 15.0
    max_assertions: int = 100
    max_derivations: int = 1_000
    max_shapes: int = 1
    max_reports: int = 32
    max_context_assertions: int = 100

    def __post_init__(self) -> None:
        for field_name in (
            "max_assertions",
            "max_derivations",
            "max_shapes",
            "max_reports",
            "max_context_assertions",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or not 1 <= value <= 1_000:
                raise SemanticMaintenanceError(
                    f"{field_name} must be an integer in [1, 1000]"
                )
        if (
            not isinstance(self.max_wall_time_seconds, (int, float))
            or isinstance(self.max_wall_time_seconds, bool)
            or not math.isfinite(self.max_wall_time_seconds)
            or self.max_wall_time_seconds <= 0
        ):
            raise SemanticMaintenanceError(
                "max_wall_time_seconds must be a positive finite number"
            )


def maintenance_limits_from_config(config: Mapping[str, object]) -> SemanticMaintenanceLimits:
    """Parse the optional operator-owned ``semantic_maintenance`` budget.

    ``semantic_inference`` remains an exact inference-profile contract.  The
    maintenance table is a sibling so changing a sleep budget cannot silently
    alter the approved rule profile.
    """
    if not isinstance(config, Mapping):
        raise SemanticMaintenanceError("[semantic_maintenance] must be a table")
    allowed = {
        "allow_prior_verified_snapshot",
        "max_wall_time_seconds",
        "max_assertions",
        "max_derivations",
        "max_shapes",
        "max_reports",
        "max_context_assertions",
    }
    unexpected = set(config).difference(allowed)
    if unexpected:
        raise SemanticMaintenanceError(
            "semantic maintenance configuration has unsupported fields: "
            + ", ".join(sorted(map(str, unexpected)))
        )
    if (
        "allow_prior_verified_snapshot" in config
        and type(config["allow_prior_verified_snapshot"]) is not bool
    ):
        raise SemanticMaintenanceError(
            "semantic maintenance allow_prior_verified_snapshot must be a boolean"
        )
    if config.get("allow_prior_verified_snapshot") is True:
        raise SemanticMaintenanceError(
            "prior verified snapshot consumption is unavailable until a durable "
            "governed corpus snapshot exists"
        )
    values: dict[str, int | float] = {}
    for name in allowed - {
        "allow_prior_verified_snapshot",
        "max_wall_time_seconds",
    }:
        if name in config:
            if type(config[name]) is not int:
                raise SemanticMaintenanceError(
                    f"semantic maintenance limit {name} must be an integer"
                )
            values[name] = config[name]  # type: ignore[assignment]
    if "max_wall_time_seconds" in config:
        value = config["max_wall_time_seconds"]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise SemanticMaintenanceError(
                "semantic maintenance limit max_wall_time_seconds must be a number"
            )
        values["max_wall_time_seconds"] = value
    return SemanticMaintenanceLimits(**values)


def maintenance_allows_prior_verified_snapshot(config: Mapping[str, object]) -> bool:
    """Reject the retired stale-corpus escape hatch.

    This compatibility helper remains importable while #2751 owns the actual
    immutable snapshot boundary. It never authorizes stale corpus use.
    """

    maintenance_limits_from_config(config)
    return False


@dataclass(frozen=True, slots=True)
class SemanticMaintenanceResult:
    """Privacy-safe aggregate outcome returned to sleep and metrics callers."""

    run_id: str
    status: SemanticMaintenanceStatus
    reason: str | None
    source_generation: int
    checkpoint_generation: int
    changes_consumed: int
    assertions_validated: int
    assertions_inferred: int
    assertions_retracted: int
    contradictions: int
    supersession_candidates: int
    expired_assertions: int
    orphan_provenance: int
    invalid_eligibility: int
    reports_created: int
    backlog_assertions: int
    backlog_reports: int
    duration_ms: int
    capability_versions: dict[str, str]

    @property
    def complete(self) -> bool:
        return self.status in (
            SemanticMaintenanceStatus.COMPLETE,
            SemanticMaintenanceStatus.NO_OP,
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "status": self.status.value,
            "reason": self.reason,
            "source_generation": self.source_generation,
            "checkpoint_generation": self.checkpoint_generation,
            "changes_consumed": self.changes_consumed,
            "assertions_validated": self.assertions_validated,
            "assertions_inferred": self.assertions_inferred,
            "assertions_retracted": self.assertions_retracted,
            "contradictions": self.contradictions,
            "supersession_candidates": self.supersession_candidates,
            "expired_assertions": self.expired_assertions,
            "orphan_provenance": self.orphan_provenance,
            "invalid_eligibility": self.invalid_eligibility,
            "reports_created": self.reports_created,
            "backlog_assertions": self.backlog_assertions,
            "backlog_reports": self.backlog_reports,
            "duration_ms": self.duration_ms,
            "capability_versions": dict(self.capability_versions),
        }


@dataclass(frozen=True, slots=True)
class _MaintenanceLease:
    holder_id: str
    fencing_token: int


@dataclass(frozen=True, slots=True)
class _MaintenanceState:
    profile_key: str
    checkpoint_generation: int
    checkpoint_event_id: str | None
    status: SemanticMaintenanceStatus
    repair_cursor_revision_id: str | None
    repair_active: bool
    repair_mode: str | None
    repair_scan_complete: bool
    repair_checkpoint_generation: int | None
    repair_checkpoint_event_id: str | None
    repair_reconcile_cursor_derivation_id: str | None
    audit_assertion_id: str | None
    audit_assertion_revision_id: str | None
    audit_competitor_cursor_revision_id: str | None


@dataclass(frozen=True, slots=True)
class _ChangeBatch:
    changes: tuple[AssertionChange, ...]
    backlog: int
    ineligible_changes: int


class _MaintenanceBudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class _AuditOutcome:
    """Completed audit counts and any report-budget work left for the next unit."""

    counts: dict[str, int]
    report_budget_exhausted: bool = False
    remaining_assertions: int = 0
    competitor_backlog: bool = False
    audit_assertion_id: str | None = None
    audit_assertion_revision_id: str | None = None
    audit_competitor_cursor_revision_id: str | None = None


class SemanticMaintenanceService:
    """Run one bounded incremental semantic-maintenance batch for one tenant.

    The assertion store is already authenticated and tenant-bound.  A durable
    lease additionally excludes concurrent sleep and explicit repair workers
    across processes; it does not confer read or write authority.
    """

    _LEASE_SECONDS = 60.0

    def __init__(
        self,
        assertion_store: "AsyncAssertionStore",
        *,
        inference_profile: InferenceProfile | None,
        inference_limits: InferenceLimits | None = None,
        limits: SemanticMaintenanceLimits | None = None,
        shape_set: ShapeSetReference = ShapeSetReference(
            "kestrel-assertion-shapes", "1.0.0"
        ),
        validation_capability: str = "validation-profile:shacl-core-20170720",
        validation_profile_version: str | None = None,
    ) -> None:
        from kestrel_sovereign.storage.async_assertion_store import AsyncAssertionStore

        if not isinstance(assertion_store, AsyncAssertionStore):
            raise TypeError("semantic maintenance requires an agent-bound assertion store")
        if inference_profile is not None and not isinstance(
            inference_profile, InferenceProfile
        ):
            raise SemanticMaintenanceError("inference_profile must be InferenceProfile or null")
        if inference_limits is not None and not isinstance(inference_limits, InferenceLimits):
            raise SemanticMaintenanceError("inference_limits must be InferenceLimits or null")
        if not isinstance(shape_set, ShapeSetReference):
            raise SemanticMaintenanceError("shape_set must be ShapeSetReference")
        if not isinstance(validation_capability, str) or not validation_capability:
            raise SemanticMaintenanceError("validation_capability must be non-empty")
        self._assertions = assertion_store
        self._database = assertion_store._database  # noqa: SLF001 - peer persistence service
        self.inference_profile = inference_profile
        self.inference_limits = inference_limits or InferenceLimits()
        self.limits = limits or SemanticMaintenanceLimits()
        self.shape_set = shape_set
        self.validation_capability = validation_capability
        self.validation_profile_version = validation_profile_version

    async def run(self, *, full_rebuild: bool = False) -> SemanticMaintenanceResult:
        """Advance changed semantic work; full rebuild is an explicit repair.

        The ordinary path never calls a full validation audit merely because a
        timer fired.  A profile/shapes change has a new profile key and is the
        deliberate exception: its current assertions are queued under the same
        finite budget rather than treated as an unbounded nightly closure.
        """
        started = monotonic()
        source_checkpoint = await self._assertions.checkpoint()
        # ``checkpoint()`` reports the latest canonical generation, which can
        # advance for proof-ledger bookkeeping without an assertion outbox
        # event.  State/result cursors that retain an event ID must instead
        # retain that event's generation or ``changes_after()`` rejects them.
        initial = await self._event_checkpoint(source_checkpoint)
        try:
            capability_versions = self._capability_versions()
        except SemanticMaintenanceError:
            # Do not let a stale capability digest turn an unavailable or
            # unverifiable validation profile into an apparent no-op.  There
            # is no trustworthy profile key under which to persist this
            # result, so readiness will independently deny it as well.
            return self._result(
                run_id=f"failed:capability:{source_checkpoint.generation}",
                status=SemanticMaintenanceStatus.FAILED,
                reason="semantic_maintenance_capability_unavailable",
                source_generation=source_checkpoint.generation,
                checkpoint_generation=0,
                backlog_assertions=1,
                duration_ms=self._duration_ms(started),
                capability_versions={},
            )
        profile_key = _digest(capability_versions)
        holder_id = uuid4().hex
        lease = await self._acquire_lease(holder_id)
        if lease is None:
            return self._result(
                run_id=f"busy:{initial.generation}",
                status=SemanticMaintenanceStatus.PARTIAL,
                reason="semantic_maintenance_busy",
                source_generation=source_checkpoint.generation,
                checkpoint_generation=0,
                backlog_assertions=1,
                duration_ms=self._duration_ms(started),
                capability_versions=capability_versions,
            )

        run_id = ""
        state: _MaintenanceState | None = None
        state_checkpoint: AssertionCheckpoint | None = None
        state_checkpoint_generation = 0
        state_matches = False
        force_current_scan = False
        repair_checkpoint: AssertionCheckpoint | None = None
        repair_page_cursor: str | None = None
        repair_mode: str | None = None
        repair_scan_complete = False
        repair_reconcile_cursor: str | None = None
        try:
            # State is only trustworthy once this worker holds the tenant's
            # cross-process lease.  A former holder must never pick up an old
            # checkpoint and later overwrite its successor's state.
            state = await self._state()
            state_matches = state is not None and state.profile_key == profile_key
            state_checkpoint = (
                await self._event_checkpoint(
                    AssertionCheckpoint(
                        self._assertions.tenant_id,
                        state.checkpoint_generation,
                        state.checkpoint_event_id,
                    )
                )
                if state is not None
                else None
            )
            state_checkpoint_generation = (
                state_checkpoint.generation if state_checkpoint is not None else 0
            )
            # A fresh tenant with no canonical generation has no assertions,
            # derivations, validation focus, or provenance to inspect. Record
            # its selected capability checkpoint without constructing either
            # engine; this is the cheap no-change sleep path from first boot.
            if not full_rebuild and initial.generation == 0:
                run_id = f"noop:{profile_key[:16]}:0"
                result = self._result(
                    run_id=run_id,
                    status=SemanticMaintenanceStatus.NO_OP,
                    reason=None,
                    source_generation=source_checkpoint.generation,
                    checkpoint_generation=0,
                    duration_ms=self._duration_ms(started),
                    capability_versions=capability_versions,
                )
                result = await self._record_state(
                    result,
                    profile_key,
                    lease,
                    checkpoint_event_id=None,
                )
                return result
            if (
                not full_rebuild
                and state_matches
                and state_checkpoint is not None
                and state.status
                in (SemanticMaintenanceStatus.COMPLETE, SemanticMaintenanceStatus.NO_OP)
                and self._checkpoint_matches(
                    state_checkpoint,
                    initial,
                )
            ):
                run_id = f"noop:{profile_key[:16]}:{initial.generation}"
                result = self._result(
                    run_id=run_id,
                    status=SemanticMaintenanceStatus.NO_OP,
                    reason=None,
                    source_generation=source_checkpoint.generation,
                    checkpoint_generation=initial.generation,
                    duration_ms=self._duration_ms(started),
                    capability_versions=capability_versions,
                )
                result = await self._record_state(
                    result,
                    profile_key,
                    lease,
                    checkpoint_event_id=initial.latest_event_id,
                )
                return result

            # A capability change and an explicit repair both scan the active
            # graph in deterministic pages.  The repair mode, scan completion,
            # and obsolete-proof cursor are all durable: a normal sleep must
            # resume an interrupted explicit repair under its original policy,
            # not reinterpret it as an incremental no-context invocation.
            force_current_scan = (
                full_rebuild
                or not state_matches
                or (state_matches and state.repair_active)
            )
            source_changes: tuple[AssertionChange, ...] = ()
            fallback_full_audit = False
            if force_current_scan:
                if state_matches and state.repair_active:
                    repair_cursor = state.repair_cursor_revision_id
                    repair_mode = state.repair_mode or (
                        "full_rebuild" if full_rebuild else "profile_change"
                    )
                    repair_scan_complete = state.repair_scan_complete
                    repair_reconcile_cursor = (
                        state.repair_reconcile_cursor_derivation_id
                    )
                    # v3 repair rows did not retain the scan origin.  Their
                    # durable maintenance checkpoint is the only safe lower
                    # bound: replaying too much is idempotent, skipping an
                    # event observed by a former repair worker is not.
                    repair_checkpoint = await self._event_checkpoint(
                        AssertionCheckpoint(
                            self._assertions.tenant_id,
                            (
                                state.repair_checkpoint_generation
                                if state.repair_checkpoint_generation is not None
                                else state.checkpoint_generation
                            ),
                            (
                                state.repair_checkpoint_event_id
                                if state.repair_checkpoint_generation is not None
                                else state.checkpoint_event_id
                            ),
                        )
                    )
                else:
                    repair_cursor = None
                    repair_checkpoint = await self._event_checkpoint(initial)
                    repair_mode = "full_rebuild" if full_rebuild else "profile_change"
                repair_page_cursor = repair_cursor
                if repair_scan_complete:
                    target_assertions = ()
                    backlog = 0
                    changes_consumed = 0
                    next_repair_cursor = None
                else:
                    target_assertions, backlog = await self._current_scan_page(
                        repair_cursor
                    )
                    changes_consumed = len(target_assertions)
                    next_repair_cursor = (
                        target_assertions[-1].revision_id if backlog else None
                    )
                ineligible_changes = 0
                processed_checkpoint: AssertionCheckpoint | None = None
                # Keep the selected page retryable on any later partial path.
                # Completion is decided only after the proof-reconciliation
                # phase has also drained.
                repair_active = True
            else:
                if state_checkpoint is None:
                    raise SemanticMaintenanceError(
                        "semantic maintenance state is missing its checkpoint"
                    )
                batch = await self._changes_after(
                    state_checkpoint
                )
                source_changes = batch.changes
                changes_consumed = len(source_changes)
                backlog = batch.backlog
                ineligible_changes = batch.ineligible_changes
                target_assertions, fallback_full_audit = await self._current_targets(
                    source_changes,
                    started=started,
                )
                processed_checkpoint = (
                    AssertionCheckpoint(
                        self._assertions.tenant_id,
                        source_changes[-1].generation,
                        source_changes[-1].event_id,
                    )
                    if source_changes
                    else None
                )
                next_repair_cursor = None
                repair_active = False
                if fallback_full_audit:
                    # Opaque erasure or a non-local deleted revision cannot
                    # be acknowledged from a focused target list. Restart a
                    # bounded active scan instead of treating this sleep unit
                    # as permission for ``export_snapshot()``.
                    target_assertions, scan_backlog = await self._current_scan_page(
                        None
                    )
                    force_current_scan = True
                    repair_checkpoint = await self._event_checkpoint(initial)
                    repair_page_cursor = None
                    repair_mode = "current_scan"
                    repair_scan_complete = False
                    repair_reconcile_cursor = None
                    backlog = max(backlog, scan_backlog)
                    next_repair_cursor = (
                        target_assertions[-1].revision_id if scan_backlog else None
                    )
                    repair_active = True
                    fallback_full_audit = False
            target_key = tuple(
                f"{item.assertion_id}:{item.revision_id}" for item in target_assertions
            )
            run_id = _digest(
                {
                    "profile": profile_key,
                    "source_generation": initial.generation,
                    "targets": target_key,
                    "repair_mode": repair_mode,
                }
            )
            await self._record_running(
                run_id,
                profile_key,
                source_checkpoint.generation,
                lease,
            )
            self._check_budget(started)

            reports_created = 0
            assertions_validated = 0
            if target_assertions or fallback_full_audit:
                async with self._assertions.maintenance_fence(
                    holder_id=lease.holder_id,
                    fencing_token=lease.fencing_token,
                    lease_seconds=self._LEASE_SECONDS,
                ):
                    report, created = await self._revalidate(
                        run_id,
                        target_assertions,
                        started,
                    )
                reports_created += int(created)
                assertions_validated = len(target_assertions)
                if report.state is ValidationState.INCOMPLETE:
                    result = self._result(
                        run_id=run_id,
                        status=SemanticMaintenanceStatus.PARTIAL,
                        reason="validation_incomplete",
                        source_generation=source_checkpoint.generation,
                        checkpoint_generation=(
                            state_checkpoint_generation if state_matches else 0
                        ),
                        changes_consumed=changes_consumed,
                        assertions_validated=assertions_validated,
                        reports_created=reports_created,
                        invalid_eligibility=ineligible_changes,
                        backlog_assertions=max(1, backlog),
                        backlog_reports=1,
                        duration_ms=self._duration_ms(started),
                        capability_versions=capability_versions,
                    )
                    result = await self._record_state(
                        result,
                        profile_key,
                        lease,
                        checkpoint_event_id=(
                            state.checkpoint_event_id if state_matches else None
                        ),
                        repair_cursor_revision_id=(
                            repair_page_cursor if force_current_scan else None
                        ),
                        repair_active=force_current_scan,
                        repair_mode=(repair_mode if force_current_scan else None),
                        repair_scan_complete=(
                            repair_scan_complete if force_current_scan else False
                        ),
                        repair_checkpoint=repair_checkpoint,
                        repair_reconcile_cursor_derivation_id=(
                            repair_reconcile_cursor if force_current_scan else None
                        ),
                        audit_assertion_id=(
                            state.audit_assertion_id if state_matches else None
                        ),
                        audit_assertion_revision_id=(
                            state.audit_assertion_revision_id
                            if state_matches
                            else None
                        ),
                        audit_competitor_cursor_revision_id=(
                            state.audit_competitor_cursor_revision_id
                            if state_matches
                            else None
                        ),
                    )
                    return result

            if not await self._renew_lease(lease):
                raise SemanticMaintenanceError("semantic_maintenance_lease_lost")
            async with self._assertions.maintenance_fence(
                holder_id=lease.holder_id,
                fencing_token=lease.fencing_token,
                lease_seconds=self._LEASE_SECONDS,
            ):
                audit_outcome = await self._audit_targets(
                    run_id,
                    target_assertions,
                    started,
                    reports_created=reports_created,
                    lease=lease,
                    audit_assertion_id=(
                        state.audit_assertion_id if state_matches else None
                    ),
                    audit_assertion_revision_id=(
                        state.audit_assertion_revision_id if state_matches else None
                    ),
                    audit_competitor_cursor_revision_id=(
                        state.audit_competitor_cursor_revision_id
                        if state_matches
                        else None
                    ),
                )
            audit = audit_outcome.counts
            reports_created += audit["reports_created"]
            if audit_outcome.report_budget_exhausted:
                result = self._result(
                    run_id=run_id,
                    status=SemanticMaintenanceStatus.PARTIAL,
                    reason="report_budget",
                    source_generation=source_checkpoint.generation,
                    checkpoint_generation=(
                        state_checkpoint_generation if state_matches else 0
                    ),
                    changes_consumed=changes_consumed,
                    assertions_validated=assertions_validated,
                    assertions_retracted=audit["retracted"],
                    reports_created=reports_created,
                    contradictions=audit["contradictions"],
                    supersession_candidates=audit["supersession_candidates"],
                    expired_assertions=audit["expired"],
                    orphan_provenance=audit["orphans"],
                    invalid_eligibility=(
                        audit["invalid_eligibility"] + ineligible_changes
                    ),
                    backlog_assertions=(
                        backlog + max(1, audit_outcome.remaining_assertions)
                    ),
                    backlog_reports=1,
                    duration_ms=self._duration_ms(started),
                    capability_versions=capability_versions,
                )
                result = await self._record_state(
                    result,
                    profile_key,
                    lease,
                    checkpoint_event_id=(
                        state.checkpoint_event_id if state_matches else None
                    ),
                        repair_cursor_revision_id=(
                            repair_page_cursor if force_current_scan else None
                        ),
                        repair_active=force_current_scan,
                        repair_mode=(repair_mode if force_current_scan else None),
                        repair_scan_complete=(
                            repair_scan_complete if force_current_scan else False
                        ),
                        repair_checkpoint=repair_checkpoint,
                        repair_reconcile_cursor_derivation_id=(
                            repair_reconcile_cursor if force_current_scan else None
                        ),
                        audit_assertion_id=(
                            audit_outcome.audit_assertion_id
                        or (state.audit_assertion_id if state_matches else None)
                    ),
                    audit_assertion_revision_id=(
                        audit_outcome.audit_assertion_revision_id
                        or (
                            state.audit_assertion_revision_id
                            if state_matches
                            else None
                        )
                    ),
                    audit_competitor_cursor_revision_id=(
                        audit_outcome.audit_competitor_cursor_revision_id
                        or (
                            state.audit_competitor_cursor_revision_id
                            if state_matches
                            else None
                        )
                    ),
                )
                return result
            if audit_outcome.competitor_backlog:
                result = self._result(
                    run_id=run_id,
                    status=SemanticMaintenanceStatus.PARTIAL,
                    reason="contradiction_context_budget",
                    source_generation=source_checkpoint.generation,
                    checkpoint_generation=(
                        state_checkpoint_generation if state_matches else 0
                    ),
                    changes_consumed=changes_consumed,
                    assertions_validated=assertions_validated,
                    assertions_retracted=audit["retracted"],
                    reports_created=reports_created,
                    contradictions=audit["contradictions"],
                    supersession_candidates=audit["supersession_candidates"],
                    expired_assertions=audit["expired"],
                    orphan_provenance=audit["orphans"],
                    invalid_eligibility=(
                        audit["invalid_eligibility"] + ineligible_changes
                    ),
                    backlog_assertions=max(1, backlog),
                    backlog_reports=1,
                    duration_ms=self._duration_ms(started),
                    capability_versions=capability_versions,
                )
                result = await self._record_state(
                    result,
                    profile_key,
                    lease,
                    checkpoint_event_id=(
                        state.checkpoint_event_id if state_matches else None
                    ),
                        repair_cursor_revision_id=(
                            repair_page_cursor if force_current_scan else None
                        ),
                        repair_active=force_current_scan,
                        repair_mode=(repair_mode if force_current_scan else None),
                        repair_scan_complete=(
                            repair_scan_complete if force_current_scan else False
                        ),
                        repair_checkpoint=repair_checkpoint,
                        repair_reconcile_cursor_derivation_id=(
                            repair_reconcile_cursor if force_current_scan else None
                        ),
                        audit_assertion_id=audit_outcome.audit_assertion_id,
                    audit_assertion_revision_id=(
                        audit_outcome.audit_assertion_revision_id
                    ),
                    audit_competitor_cursor_revision_id=(
                        audit_outcome.audit_competitor_cursor_revision_id
                    ),
                )
                return result
            self._check_budget(started)
            inferred = 0
            retracted = audit["retracted"]
            if self.inference_profile is not None:
                if not await self._renew_lease(lease):
                    raise SemanticMaintenanceError("semantic_maintenance_lease_lost")
                bounded_inference = replace(
                    self.inference_limits,
                    max_source_assertions=min(
                        self.inference_limits.max_source_assertions,
                        self._primary_assertion_limit
                        + self.limits.max_context_assertions,
                    ),
                    max_generated_assertions=min(
                        self.inference_limits.max_generated_assertions,
                        self.limits.max_derivations,
                    ),
                    max_wall_time_seconds=min(
                        self.inference_limits.max_wall_time_seconds,
                        max(0.001, self.limits.max_wall_time_seconds - (monotonic() - started)),
                    ),
                )
                inference_service = BoundedInferenceService(
                    self._assertions,
                    self.inference_profile,
                    limits=bounded_inference,
                )
                async with self._assertions.maintenance_fence(
                    holder_id=lease.holder_id,
                    fencing_token=lease.fencing_token,
                    lease_seconds=self._LEASE_SECONDS,
                ):
                    materialized = await inference_service.materialize_targets(
                        tuple(item.assertion_id for item in target_assertions),
                        max_context_assertions=min(
                            self.limits.max_context_assertions,
                            max(
                                0,
                                inference_service.limits.max_source_assertions
                                - len(target_assertions),
                            ),
                        ),
                    )
                inferred = materialized.generated_assertions
                retracted += materialized.retracted_assertions
                if materialized.status is not ClosureStatus.COMPLETE:
                    result = self._result(
                        run_id=run_id,
                        status=SemanticMaintenanceStatus.PARTIAL,
                        reason=materialized.incomplete_reason or "inference_incomplete",
                        source_generation=source_checkpoint.generation,
                        checkpoint_generation=(
                            state_checkpoint_generation if state_matches else 0
                        ),
                        changes_consumed=changes_consumed,
                        assertions_validated=assertions_validated,
                        assertions_inferred=inferred,
                        assertions_retracted=retracted,
                        reports_created=reports_created,
                        contradictions=audit["contradictions"],
                        supersession_candidates=audit["supersession_candidates"],
                        expired_assertions=audit["expired"],
                        orphan_provenance=audit["orphans"],
                        invalid_eligibility=(
                            audit["invalid_eligibility"] + ineligible_changes
                        ),
                        backlog_assertions=max(1, backlog),
                        duration_ms=self._duration_ms(started),
                        capability_versions=capability_versions,
                    )
                    result = await self._record_state(
                        result,
                        profile_key,
                        lease,
                        checkpoint_event_id=(
                            state.checkpoint_event_id if state_matches else None
                        ),
                        repair_cursor_revision_id=(
                            repair_page_cursor if force_current_scan else None
                        ),
                        repair_active=force_current_scan,
                        repair_mode=(repair_mode if force_current_scan else None),
                        repair_scan_complete=(
                            repair_scan_complete if force_current_scan else False
                        ),
                        repair_checkpoint=repair_checkpoint,
                        repair_reconcile_cursor_derivation_id=(
                            repair_reconcile_cursor if force_current_scan else None
                        ),
                        audit_assertion_id=(
                            state.audit_assertion_id if state_matches else None
                        ),
                        audit_assertion_revision_id=(
                            state.audit_assertion_revision_id
                            if state_matches
                            else None
                        ),
                        audit_competitor_cursor_revision_id=(
                            state.audit_competitor_cursor_revision_id
                            if state_matches
                            else None
                        ),
                    )
                    return result

            self._check_budget(started)
            reconciliation: InferenceReconciliationResult | None = None
            if force_current_scan and not backlog:
                # A completed assertion scan is only the first half of a
                # profile/ontology repair.  Retire stale active proof rows in
                # deterministic pages before maintenance may publish a
                # complete checkpoint.  A scan-only fallback for an opaque
                # deletion has no changed profile to reconcile.
                repair_scan_complete = True
                if repair_mode in ("full_rebuild", "profile_change"):
                    if not await self._renew_lease(lease):
                        raise SemanticMaintenanceError("semantic_maintenance_lease_lost")
                    async with self._assertions.maintenance_fence(
                        holder_id=lease.holder_id,
                        fencing_token=lease.fencing_token,
                        lease_seconds=self._LEASE_SECONDS,
                    ):
                        reconciliation = (
                            await BoundedInferenceService.reconcile_obsolete_derivations_page(
                                self._assertions,
                                active_profile_key=(
                                    self.inference_profile.key
                                    if self.inference_profile is not None
                                    else None
                                ),
                                cursor=repair_reconcile_cursor,
                                max_derivations=self.limits.max_derivations,
                                run_id=run_id,
                            )
                        )
                    retracted += reconciliation.retracted_assertions
                    repair_reconcile_cursor = reconciliation.next_cursor
            reconciliation_backlog = (
                reconciliation.backlog if reconciliation is not None else 0
            )
            repair_active = force_current_scan and bool(
                backlog or reconciliation_backlog
            )
            # A maintenance unit may write validation reports, audit repairs,
            # or inferred assertions.  Those are fresh canonical changes, not
            # inputs that this unit has validated and audited end-to-end.  The
            # durable cursor must therefore stop at the input batch (or the
            # pre-work snapshot for a capability/full-rebuild scan), never at
            # the mutable checkpoint observed after those writes.
            #
            # A generation is a transaction boundary rather than a cursor: a
            # supersession can emit several events at it, so partial state
            # keeps the final processed event ID as well as its generation.
            post_work_replay_required = False
            post_work_replay_reason: str | None = None
            if force_current_scan and not backlog:
                if repair_checkpoint is None:
                    raise SemanticMaintenanceError("repair scan is missing its origin checkpoint")
                # The lexical scan cursor does not order concurrent writes.
                # Keep the checkpoint taken before the first repair page and
                # replay every outbox event after it on the next ordinary
                # maintenance unit.  Advancing to this page's ``initial``
                # checkpoint would lose a write whose revision ID sorts
                # behind the already-persisted repair cursor.
                durable_checkpoint = repair_checkpoint
                post_work_replay_required = bool(
                    await self._assertions.changes_after(
                        repair_checkpoint,
                        limit=1,
                    )
                )
                if post_work_replay_required:
                    post_work_replay_reason = "repair_change_replay"
            elif backlog:
                if processed_checkpoint is not None:
                    durable_checkpoint = processed_checkpoint
                elif state_matches:
                    if state_checkpoint is None:
                        raise SemanticMaintenanceError(
                            "semantic maintenance state is missing its checkpoint"
                        )
                    durable_checkpoint = state_checkpoint
                else:
                    durable_checkpoint = AssertionCheckpoint(
                        self._assertions.tenant_id, 0, None
                    )
            else:
                if processed_checkpoint is not None:
                    durable_checkpoint = processed_checkpoint
                elif state_matches:
                    if state_checkpoint is None:
                        raise SemanticMaintenanceError(
                            "semantic maintenance state is missing its checkpoint"
                        )
                    durable_checkpoint = state_checkpoint
                else:
                    durable_checkpoint = AssertionCheckpoint(
                        self._assertions.tenant_id, 0, None
                    )
                # Normal incremental materialization can create derived
                # assertions after its selected input events.  They are not
                # covered merely because this unit's inference call returned
                # COMPLETE: they must be observed as inputs by another
                # bounded maintenance unit before training/readiness can
                # claim closure.
                if not force_current_scan and not backlog:
                    self._check_budget(started)
                    post_work_replay_required = bool(
                        await self._assertions.changes_after(
                            durable_checkpoint,
                            limit=1,
                        )
                    )
                    if post_work_replay_required:
                        post_work_replay_reason = "change_replay"
            status = (
                SemanticMaintenanceStatus.PARTIAL
                if backlog or reconciliation_backlog or post_work_replay_required
                else SemanticMaintenanceStatus.COMPLETE
            )
            result = self._result(
                run_id=run_id,
                status=status,
                reason=(
                    "assertion_budget"
                    if backlog
                    else "derivation_budget"
                    if reconciliation_backlog
                    else post_work_replay_reason
                    if post_work_replay_required
                    else None
                ),
                source_generation=source_checkpoint.generation,
                checkpoint_generation=durable_checkpoint.generation,
                changes_consumed=changes_consumed,
                assertions_validated=assertions_validated,
                assertions_inferred=inferred,
                assertions_retracted=retracted,
                contradictions=audit["contradictions"],
                supersession_candidates=audit["supersession_candidates"],
                expired_assertions=audit["expired"],
                orphan_provenance=audit["orphans"],
                invalid_eligibility=(
                    audit["invalid_eligibility"] + ineligible_changes
                ),
                reports_created=reports_created,
                backlog_assertions=max(
                    backlog,
                    reconciliation_backlog,
                    int(post_work_replay_required),
                ),
                duration_ms=self._duration_ms(started),
                capability_versions=capability_versions,
            )
            # A partial batch has completed its bounded unit.  Its checkpoint
            # is therefore durable; the nonzero backlog says exactly why the
            # next sleep must take another unit rather than claiming closure.
            result = await self._record_state(
                result,
                profile_key,
                lease,
                checkpoint_event_id=durable_checkpoint.latest_event_id,
                repair_cursor_revision_id=next_repair_cursor,
                repair_active=repair_active,
                repair_mode=(repair_mode if repair_active else None),
                repair_scan_complete=(repair_scan_complete if repair_active else False),
                repair_checkpoint=(repair_checkpoint if repair_active else None),
                repair_reconcile_cursor_derivation_id=(
                    repair_reconcile_cursor if repair_active else None
                ),
            )
            return result
        except _MaintenanceBudgetExceeded:
            result = self._result(
                run_id=run_id or f"budget:{initial.generation}",
                status=SemanticMaintenanceStatus.PARTIAL,
                reason="wall_time",
                source_generation=source_checkpoint.generation,
                checkpoint_generation=state_checkpoint_generation,
                backlog_assertions=1,
                duration_ms=self._duration_ms(started),
                capability_versions=capability_versions,
            )
            if run_id:
                # A timeout after selecting a repair page must leave that
                # page retryable.  Recording only the run result used to
                # discard the newly introduced scan cursor and restart from
                # the first lexical page forever.
                result = await self._record_state(
                    result,
                    profile_key,
                    lease,
                    checkpoint_event_id=(
                        state.checkpoint_event_id if state_matches and state else None
                    ),
                    repair_cursor_revision_id=(
                        repair_page_cursor if force_current_scan else None
                    ),
                    repair_active=force_current_scan,
                    repair_mode=(repair_mode if force_current_scan else None),
                    repair_scan_complete=(
                        repair_scan_complete if force_current_scan else False
                    ),
                    repair_checkpoint=repair_checkpoint,
                    repair_reconcile_cursor_derivation_id=(
                        repair_reconcile_cursor if force_current_scan else None
                    ),
                    audit_assertion_id=(
                        state.audit_assertion_id if state_matches and state else None
                    ),
                    audit_assertion_revision_id=(
                        state.audit_assertion_revision_id
                        if state_matches and state
                        else None
                    ),
                    audit_competitor_cursor_revision_id=(
                        state.audit_competitor_cursor_revision_id
                        if state_matches and state
                        else None
                    ),
                )
            else:
                await self._record_run_if_current(result, profile_key, lease)
            return result
        except MaintenanceLeaseLostError:
            result = self._result(
                run_id=run_id or f"failed:{initial.generation}",
                status=SemanticMaintenanceStatus.FAILED,
                reason="semantic_maintenance_lease_lost",
                source_generation=source_checkpoint.generation,
                checkpoint_generation=state_checkpoint_generation,
                backlog_assertions=1,
                duration_ms=self._duration_ms(started),
                capability_versions=capability_versions,
            )
            await self._record_run_if_current(result, profile_key, lease)
            return result
        except Exception as error:
            logger.exception(
                "semantic maintenance failed",
                extra={"semantic_maintenance_tenant_id": self._assertions.tenant_id},
            )
            result = self._result(
                run_id=run_id or f"failed:{initial.generation}",
                status=SemanticMaintenanceStatus.FAILED,
                reason=(str(error) if isinstance(error, SemanticMaintenanceError) else type(error).__name__),
                source_generation=source_checkpoint.generation,
                checkpoint_generation=state_checkpoint_generation,
                backlog_assertions=1,
                duration_ms=self._duration_ms(started),
                capability_versions=capability_versions,
            )
            await self._record_run_if_current(result, profile_key, lease)
            return result
        finally:
            await self._release_lease(lease)

    async def rebuild(self) -> SemanticMaintenanceResult:
        """Run the explicit, page-oriented semantic repair path under one lease."""
        return await self.run(full_rebuild=True)

    @property
    def _primary_assertion_limit(self) -> int:
        """Keep one maintenance unit within both primary-work budgets."""
        if self.inference_profile is None:
            return self.limits.max_assertions
        return min(
            self.limits.max_assertions,
            self.inference_limits.max_source_assertions,
        )

    async def training_readiness(
        self,
        *,
        allow_prior_verified_snapshot: bool = False,
        expected_checkpoint: AssertionCheckpoint | None = None,
    ) -> SemanticMaintenanceTrainingReadiness:
        """Read whether a scheduled training consumer may use semantic data.

        This is deliberately a read-only check over the durable state written
        by maintenance. It uses the same capability identity as ``run()``, so
        a changed ontology, shape profile, or effective maintenance budget
        invalidates an earlier verification. A historical complete run is not
        a durable corpus snapshot, so stale semantic data remains blocked until
        the governed snapshot boundary provides one.
        """

        if type(allow_prior_verified_snapshot) is not bool:
            raise SemanticMaintenanceError(
                "allow_prior_verified_snapshot must be a boolean"
            )
        if allow_prior_verified_snapshot:
            raise SemanticMaintenanceError(
                "prior verified snapshot consumption is unavailable until a durable "
                "governed corpus snapshot exists"
            )
        raw_current = await self._assertions.checkpoint()
        current = await self._event_checkpoint(raw_current)
        if expected_checkpoint is not None:
            if not isinstance(expected_checkpoint, AssertionCheckpoint):
                raise SemanticMaintenanceError(
                    "expected_checkpoint must be AssertionCheckpoint or null"
                )
            normalized_expected = await self._event_checkpoint(expected_checkpoint)
            if not self._checkpoint_matches(current, normalized_expected):
                return SemanticMaintenanceTrainingReadiness(
                    False,
                    "semantic_maintenance_checkpoint_changed",
                )
        try:
            profile_key = _digest(self._capability_versions())
        except SemanticMaintenanceError:
            return SemanticMaintenanceTrainingReadiness(
                False,
                "semantic_maintenance_capability_unavailable",
            )
        state = await self._state()
        state_checkpoint = (
            await self._event_checkpoint(
                AssertionCheckpoint(
                    self._assertions.tenant_id,
                    state.checkpoint_generation,
                    state.checkpoint_event_id,
                )
            )
            if state is not None
            else None
        )
        if (
            state is not None
            and state_checkpoint is not None
            and state.profile_key == profile_key
            and state.status
            in (SemanticMaintenanceStatus.COMPLETE, SemanticMaintenanceStatus.NO_OP)
            and self._checkpoint_matches(
                state_checkpoint,
                current,
            )
        ):
            return SemanticMaintenanceTrainingReadiness(True, None)

        if state is None:
            reason = "semantic_maintenance_state_missing"
        elif state.profile_key != profile_key:
            reason = "semantic_maintenance_capability_mismatch"
        elif state.status not in (
            SemanticMaintenanceStatus.COMPLETE,
            SemanticMaintenanceStatus.NO_OP,
        ):
            reason = f"semantic_maintenance_{state.status.value}"
        else:
            reason = "semantic_maintenance_checkpoint_behind"
        return SemanticMaintenanceTrainingReadiness(False, reason)

    def capability_versions(self) -> dict[str, str]:
        """Return the exact pins/budgets that define maintenance readiness.

        A governed corpus stores this content-free identity alongside its
        checkpoint, so consumers cannot reuse a snapshot after a shape,
        ontology, rule, or maintenance-budget change.
        """
        return dict(self._capability_versions())

    async def _current_scan_page(
        self, cursor: str | None
    ) -> tuple[tuple[Assertion, ...], int]:
        """Read one indexed active page and a count-only repair backlog."""
        targets = tuple(
            await self._assertions.query(
                AssertionQuery(limit=self._primary_assertion_limit, cursor=cursor)
            )
        )
        remaining = await self._assertions.active_assertion_count(cursor=cursor)
        return targets, max(0, remaining - len(targets))

    async def _event_checkpoint(
        self, checkpoint: AssertionCheckpoint
    ) -> AssertionCheckpoint:
        """Normalize a replay cursor to the generation of its event.

        The canonical generation also advances for proof-ledger membership
        changes, which deliberately emit no assertion outbox event.  A raw
        ``checkpoint()`` can therefore carry a current generation newer than
        its latest event.  ``changes_after()`` correctly rejects that invalid
        pair; repair origins must retain the event's own generation instead.
        """
        if checkpoint.latest_event_id is None:
            return checkpoint
        row = await self._database.fetchone(
            "SELECT generation FROM semantic_projection_outbox "
            "WHERE tenant_id = ? AND event_id = ? "
            "UNION ALL "
            "SELECT generation FROM semantic_projection_erasure_outbox "
            "WHERE tenant_id = ? AND event_id = ? LIMIT 1",
            (
                self._assertions.tenant_id,
                checkpoint.latest_event_id,
                self._assertions.tenant_id,
                checkpoint.latest_event_id,
            ),
        )
        return (
            checkpoint
            if row is None
            else AssertionCheckpoint(
                checkpoint.tenant_id,
                int(row[0]),
                checkpoint.latest_event_id,
            )
        )

    async def _changes_after(self, checkpoint: AssertionCheckpoint) -> _ChangeBatch:
        """Take one event-bounded batch without splitting cursor semantics.

        The assertion-store API owns the ordering and its legacy recovery;
        maintenance only decides how many events one sleep unit may consume.
        A second one-event probe preserves the exact assertion budget at the
        API's maximum page size.
        """
        work_limit = self._primary_assertion_limit
        changes = await self._assertions.changes_after(
            checkpoint,
            limit=min(1_000, work_limit + 1),
        )
        selected = tuple(changes[:work_limit])
        backlog = max(0, len(changes) - len(selected))
        if backlog == 0 and work_limit == 1_000 and selected:
            last = selected[-1]
            probe = await self._assertions.changes_after(
                AssertionCheckpoint(
                    self._assertions.tenant_id,
                    last.generation,
                    last.event_id,
                ),
                limit=1,
            )
            backlog = len(probe)
        return _ChangeBatch(
            changes=selected,
            backlog=backlog,
            ineligible_changes=sum(1 for item in selected if not item.eligible),
        )

    async def _current_targets(
        self,
        changes: Sequence[AssertionChange],
        *,
        started: float,
    ) -> tuple[tuple[Assertion, ...], bool]:
        """Resolve active validation/audit focus for lifecycle changes.

        Current reads intentionally hide inactive assertions.  That is right
        for application callers but insufficient for maintenance: a deletion
        or retraction can invalidate a nearby active shape even though the
        changed assertion itself is no longer readable as active.  We retain
        each change record, inspect its historical revision, and collect a
        bounded active term-neighbourhood.  An opaque erasure, a missing
        revision, or an oversized neighbourhood cannot be safely localized,
        so the caller performs the bounded full-audit fallback before it may
        acknowledge the event cursor.
        """
        targets: dict[str, Assertion] = {}
        context_target_ids: set[str] = set()
        queried_neighbourhoods: dict[tuple[str, object], bool] = {}
        fallback_full_audit = False
        for change in changes:
            self._check_budget(started)
            if fallback_full_audit:
                # Once a deletion cannot be safely localized, the caller
                # will use the bounded current-scan path.  Additional
                # neighbour probes cannot improve that fallback and would
                # merely multiply the work for one input batch.
                break
            if change.assertion_id is None:
                fallback_full_audit = True
                continue
            current = await self._assertions.get_assertion(
                change.assertion_id,
                include_inactive=True,
            )
            if current is not None and current.status is AssertionStatus.ACTIVE:
                targets[current.assertion_id] = current
                continue

            historical = (
                await self._assertions.get_revision(change.revision_id)
                if change.revision_id is not None
                else current
            )
            if historical is None:
                fallback_full_audit = True
                continue
            related, overflow, has_active_neighbour = await self._affected_active_targets(
                historical,
                remaining_context_assertions=(
                    self.limits.max_context_assertions - len(context_target_ids)
                ),
                started=started,
                known_assertion_ids=set(targets),
                queried_neighbourhoods=queried_neighbourhoods,
            )
            for assertion in related:
                if assertion.assertion_id in targets:
                    continue
                if len(context_target_ids) >= self.limits.max_context_assertions:
                    overflow = True
                    break
                targets[assertion.assertion_id] = assertion
                context_target_ids.add(assertion.assertion_id)
            # A removed fact with no active term neighbour can still affect a
            # non-local shape (for example, a required relationship), so it
            # must use the full-audit path rather than becoming a no-op.
            if overflow or not has_active_neighbour:
                fallback_full_audit = True

        ordered = tuple(sorted(targets.values(), key=lambda item: item.assertion_id))
        if len(ordered) > self._primary_assertion_limit:
            return ordered[: self._primary_assertion_limit], True
        return ordered, fallback_full_audit

    async def _affected_active_targets(
        self,
        removed: Assertion,
        *,
        remaining_context_assertions: int,
        started: float,
        known_assertion_ids: set[str],
        queried_neighbourhoods: dict[tuple[str, object], bool],
    ) -> tuple[tuple[Assertion, ...], bool, bool]:
        """Find a globally bounded active term-neighbourhood for a deletion.

        Deletions often share terms.  Cache those indexed probes and spend one
        shared context allowance across the whole input batch rather than
        granting every deleted revision another three full-sized queries.
        """

        query_specs = (
            ("subject", removed.subject),
            ("predicate", removed.predicate),
            ("object", removed.object),
        )
        related: dict[str, Assertion] = {}
        overflow = False
        has_active_neighbour = False
        for field, term in query_specs:
            self._check_budget(started)
            query_key = (field, term)
            previous_has_active = queried_neighbourhoods.get(query_key)
            if previous_has_active is not None:
                has_active_neighbour = has_active_neighbour or previous_has_active
                continue
            remaining = remaining_context_assertions - len(related)
            if remaining <= 0:
                # We cannot establish a new relation without exceeding the
                # unit-wide context cap, so defer to the bounded full scan.
                return (
                    tuple(
                        sorted(related.values(), key=lambda item: item.assertion_id)
                    ),
                    True,
                    has_active_neighbour,
                )
            query_limit = min(1_000, remaining + 1)
            if field == "subject":
                query = AssertionQuery(subject=removed.subject, limit=query_limit)
            elif field == "predicate":
                query = AssertionQuery(predicate=removed.predicate, limit=query_limit)
            else:
                query = AssertionQuery(object=removed.object, limit=query_limit)
            matches = await self._assertions.query(query)
            has_matches = bool(matches)
            queried_neighbourhoods[query_key] = has_matches
            has_active_neighbour = has_active_neighbour or has_matches
            if len(matches) == query_limit:
                overflow = True
            for assertion in matches:
                if assertion.assertion_id in known_assertion_ids:
                    continue
                if assertion.assertion_id in related:
                    continue
                if len(related) >= remaining_context_assertions:
                    overflow = True
                    break
                related[assertion.assertion_id] = assertion
            if overflow:
                break
        ordered = tuple(sorted(related.values(), key=lambda item: item.assertion_id))
        return ordered, overflow, has_active_neighbour

    async def _revalidate(
        self,
        run_id: str,
        targets: Sequence[Assertion],
        started: float,
    ) -> tuple[ShaclValidationReport, bool]:
        self._check_budget(started)
        row = await self._database.fetchone(
            "SELECT report_mapping FROM semantic_validation_reports "
            "WHERE tenant_id = ? AND run_id = ? ORDER BY report_id ASC LIMIT 1",
            (self._assertions.tenant_id, run_id),
        )
        if row is not None:
            return ShaclValidationReport.from_mapping(json.loads(row[0])), False
        from kestrel_sovereign.storage.semantic_validation import GovernedSemanticValidationService

        validation = GovernedSemanticValidationService(self._assertions)
        options = {
            "shape_set": self.shape_set,
            "validation_capability": self.validation_capability,
            "profile_version": self.validation_profile_version,
            "limits": ShaclValidationLimits(
                max_shapes=self.limits.max_shapes,
                max_wall_time_seconds=max(
                    0.001, self.limits.max_wall_time_seconds - (monotonic() - started)
                )
            ),
            "run_id": run_id,
            # Repair pages use the same bounded focused contract as sleep.
            # Full-tenant context is never concealed behind a one-assertion
            # repair page: a non-local shape returns an explicit incomplete
            # report until the operator supplies a sufficiently bounded local
            # context strategy instead of materializing the whole tenant.
            "bounded_focus_only": True,
        }
        report = await validation.validate_current(
            assertion_ids=tuple(item.assertion_id for item in targets),
            **options,
        )
        return report, True

    async def _audit_targets(
        self,
        run_id: str,
        targets: Sequence[Assertion],
        started: float,
        *,
        reports_created: int,
        lease: _MaintenanceLease,
        audit_assertion_id: str | None,
        audit_assertion_revision_id: str | None,
        audit_competitor_cursor_revision_id: str | None,
    ) -> _AuditOutcome:
        counts = {
            "contradictions": 0,
            "supersession_candidates": 0,
            "expired": 0,
            "orphans": 0,
            "invalid_eligibility": 0,
            "retracted": 0,
            "reports_created": 0,
        }
        now = datetime.now(timezone.utc)
        for position, assertion in enumerate(targets):
            self._check_budget(started)
            current = await self._assertions.get_assertion(assertion.assertion_id)
            if current is None or current.status is not AssertionStatus.ACTIVE:
                continue
            if _is_expired(current, now):
                counts["expired"] += 1
                created = await self._record_candidate(
                    "expired_assertion",
                    "deterministic_action_applied",
                    {"assertion_id": current.assertion_id, "revision_id": current.revision_id},
                    started=started,
                    reports_created=reports_created + counts["reports_created"],
                    lease=lease,
                )
                if created is None:
                    return _AuditOutcome(
                        counts,
                        report_budget_exhausted=True,
                        remaining_assertions=len(targets) - position,
                    )
                counts["reports_created"] += int(created)
                await self._assertions.retract(
                    current.assertion_id,
                    current.revision_id,
                    operation_id=f"semantic-maintenance-expiry:{run_id}:{current.revision_id}",
                )
                counts["retracted"] += 1
                continue
            if isinstance(current.lineage, DirectLineage):
                sources = await self._assertions.list_source_occurrences(
                    current.assertion_id
                )
                if {
                    item.source_occurrence_id for item in sources
                } != set(current.lineage.source_occurrence_ids):
                    counts["orphans"] += 1
                    counts["invalid_eligibility"] += 1
                    created = await self._record_candidate(
                        "orphan_provenance",
                        "deterministic_action_applied",
                        {"assertion_id": current.assertion_id, "revision_id": current.revision_id},
                        started=started,
                        reports_created=reports_created + counts["reports_created"],
                        lease=lease,
                    )
                    if created is None:
                        return _AuditOutcome(
                            counts,
                            report_budget_exhausted=True,
                            remaining_assertions=len(targets) - position,
                        )
                    counts["reports_created"] += int(created)
                    await self._assertions.invalidate_assertion_eligibility(
                        current.assertion_id,
                        current.revision_id,
                        operation_id=(
                            "semantic-maintenance-orphan-provenance:"
                            f"{run_id}:{current.revision_id}"
                        ),
                    )
                    continue
            competitor_cursor = (
                audit_competitor_cursor_revision_id
                if (
                    audit_assertion_id == current.assertion_id
                    and audit_assertion_revision_id == current.revision_id
                )
                else None
            )
            competing = await self._assertions.query(
                AssertionQuery(
                    subject=current.subject,
                    predicate=current.predicate,
                    # Competitors are bounded contextual reads, never an
                    # unbounded tenant scan. A wider conflicting set remains
                    # visible through a later maintenance unit.
                    # This is context work rather than primary maintenance
                    # work. Its own explicit budget prevents one contested
                    # predicate from consuming every changed assertion slot.
                    limit=self.limits.max_context_assertions,
                    cursor=competitor_cursor,
                    exclude_assertion_ids=(current.assertion_id,),
                )
            )
            for other in competing:
                if other.object.identity_mapping() == current.object.identity_mapping():
                    continue
                evidence = {
                    "assertion_ids": sorted((current.assertion_id, other.assertion_id)),
                    "revision_ids": sorted((current.revision_id, other.revision_id)),
                }
                # A single review artifact carries both the contradiction and
                # possible supersession evidence.  It is deliberately not an
                # instruction to select either assertion, and one artifact per
                # pair keeps the report budget exact.
                created = await self._record_candidate(
                    "contradiction_candidate",
                    "review_required",
                    evidence,
                    started=started,
                    reports_created=reports_created + counts["reports_created"],
                    lease=lease,
                )
                if created is None:
                    return _AuditOutcome(
                        counts,
                        report_budget_exhausted=True,
                        remaining_assertions=len(targets) - position,
                        audit_assertion_id=current.assertion_id,
                        audit_assertion_revision_id=current.revision_id,
                        audit_competitor_cursor_revision_id=competitor_cursor,
                    )
                # The durable evidence digest deduplicates a pair when both
                # sides changed in this batch. Do not use assertion-ID order
                # as a shortcut: it misses changed-vs-unchanged pairs.
                if created:
                    counts["contradictions"] += 1
                    counts["reports_created"] += 1
                    counts["supersession_candidates"] += 1
            if len(competing) == self.limits.max_context_assertions:
                probe = await self._assertions.query(
                    AssertionQuery(
                        subject=current.subject,
                        predicate=current.predicate,
                        limit=1,
                        cursor=competing[-1].revision_id,
                        exclude_assertion_ids=(current.assertion_id,),
                    )
                )
                if probe:
                    return _AuditOutcome(
                        counts,
                        competitor_backlog=True,
                        audit_assertion_id=current.assertion_id,
                        audit_assertion_revision_id=current.revision_id,
                        audit_competitor_cursor_revision_id=(
                            competing[-1].revision_id
                        ),
                    )
        return _AuditOutcome(counts)

    async def _record_candidate(
        self,
        kind: str,
        status: str,
        evidence: Mapping[str, object],
        *,
        started: float,
        reports_created: int,
        lease: _MaintenanceLease,
    ) -> bool | None:
        """Write one new audit candidate, or signal an exhausted report budget.

        ``False`` means this exact report is already durable and costs no
        budget.  ``None`` means a new report is required but its write budget
        has been exhausted; callers must leave the checkpoint behind it.
        """
        self._check_budget(started)
        encoded = _json(evidence)
        digest = _digest({"kind": kind, "evidence": json.loads(encoded)})
        report_id = digest[:32]
        now = _now()
        try:
            async with self._database.transaction():
                await self._renew_fenced_lease_in_transaction(lease)
                existing = await self._database.fetchone(
                    "SELECT 1 FROM semantic_maintenance_reports "
                    "WHERE tenant_id = ? AND evidence_digest = ?",
                    (self._assertions.tenant_id, digest),
                )
                if existing is not None:
                    return False
                self._check_budget(started)
                if reports_created >= self.limits.max_reports:
                    return None
                await self._database.execute(
                    "INSERT INTO semantic_maintenance_reports "
                    "(tenant_id, report_id, report_kind, evidence_digest, status, evidence_mapping, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        self._assertions.tenant_id,
                        report_id,
                        kind,
                        digest,
                        status,
                        encoded,
                        now,
                        now,
                    ),
                )
        except TransactionError as error:
            if isinstance(error.__cause__, MaintenanceLeaseLostError):
                raise error.__cause__ from error
            if isinstance(error.__cause__, _MaintenanceBudgetExceeded):
                raise error.__cause__ from error
            raise
        return True

    async def _state(self) -> _MaintenanceState | None:
        row = await self._database.fetchone(
            "SELECT profile_key, checkpoint_generation, checkpoint_event_id, status, "
            "repair_cursor_revision_id, repair_active, repair_mode, repair_scan_complete, "
            "repair_checkpoint_generation, repair_checkpoint_event_id, "
            "repair_reconcile_cursor_derivation_id, audit_assertion_id, "
            "audit_assertion_revision_id, "
            "audit_competitor_cursor_revision_id "
            "FROM semantic_maintenance_state "
            "WHERE tenant_id = ?",
            (self._assertions.tenant_id,),
        )
        return (
            None
            if row is None
            else _MaintenanceState(
                str(row[0]),
                int(row[1]),
                (str(row[2]) if row[2] is not None else None),
                SemanticMaintenanceStatus(str(row[3])),
                (str(row[4]) if row[4] is not None else None),
                bool(row[5]),
                (str(row[6]) if row[6] is not None else None),
                bool(row[7]),
                (int(row[8]) if row[8] is not None else None),
                (str(row[9]) if row[9] is not None else None),
                (str(row[10]) if row[10] is not None else None),
                (str(row[11]) if row[11] is not None else None),
                (str(row[12]) if row[12] is not None else None),
                (str(row[13]) if row[13] is not None else None),
            )
        )

    async def _record_running(
        self,
        run_id: str,
        profile_key: str,
        source_generation: int,
        lease: _MaintenanceLease,
    ) -> None:
        now = _now()
        try:
            async with self._database.transaction():
                await self._renew_fenced_lease_in_transaction(lease)
                await self._database.execute(
                    "INSERT INTO semantic_maintenance_runs "
                    "(tenant_id, run_id, profile_key, source_generation, status, reason, result_mapping, started_at, completed_at) "
                    "VALUES (?, ?, ?, ?, 'running', NULL, '{}', ?, NULL) "
                    "ON CONFLICT(tenant_id, run_id) DO UPDATE SET "
                    "profile_key = excluded.profile_key, "
                    "source_generation = excluded.source_generation, "
                    "status = 'running', reason = NULL, "
                    "result_mapping = '{}', started_at = excluded.started_at, "
                    "completed_at = NULL",
                    (
                        self._assertions.tenant_id,
                        run_id,
                        profile_key,
                        source_generation,
                        now,
                    ),
                )
        except TransactionError as error:
            if isinstance(error.__cause__, MaintenanceLeaseLostError):
                raise SemanticMaintenanceError("semantic_maintenance_lease_lost") from error
            raise

    async def _record_run(
        self,
        result: SemanticMaintenanceResult,
        profile_key: str,
        lease: _MaintenanceLease,
    ) -> None:
        """Persist a terminal run only while this exact lease remains current."""
        try:
            async with self._database.transaction():
                await self._renew_fenced_lease_in_transaction(lease)
                await self._write_terminal_run_in_transaction(result, profile_key)
        except TransactionError as error:
            if isinstance(error.__cause__, MaintenanceLeaseLostError):
                raise SemanticMaintenanceError("semantic_maintenance_lease_lost") from error
            raise
        self._log_result(result)

    async def _record_run_if_current(
        self,
        result: SemanticMaintenanceResult,
        profile_key: str,
        lease: _MaintenanceLease,
    ) -> bool:
        """Best-effort terminal evidence for an already-failed worker.

        A result returned to a stale caller is useful diagnostics, but it must
        not become durable evidence after a successor owns the fence.
        """

        try:
            await self._record_run(result, profile_key, lease)
        except SemanticMaintenanceError as error:
            if str(error) == "semantic_maintenance_lease_lost":
                return False
            raise
        return True

    async def _write_terminal_run_in_transaction(
        self,
        result: SemanticMaintenanceResult,
        profile_key: str,
    ) -> None:
        """Write terminal evidence under the caller's already-renewed fence."""
        now = _now()
        await self._database.execute(
            "INSERT INTO semantic_maintenance_runs "
            "(tenant_id, run_id, profile_key, source_generation, status, reason, result_mapping, started_at, completed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(tenant_id, run_id) DO UPDATE SET status = excluded.status, reason = excluded.reason, result_mapping = excluded.result_mapping, completed_at = excluded.completed_at",
            (
                self._assertions.tenant_id,
                result.run_id,
                profile_key,
                result.source_generation,
                result.status.value,
                result.reason,
                _json(result.to_mapping()),
                now,
                now,
            ),
        )

    def _log_result(self, result: SemanticMaintenanceResult) -> None:
        logger.info(
            "semantic_maintenance",
            extra={
                "semantic_maintenance_status": result.status.value,
                "semantic_maintenance_reason": result.reason,
                "semantic_maintenance_changes_consumed": result.changes_consumed,
                "semantic_maintenance_assertions_validated": result.assertions_validated,
                "semantic_maintenance_assertions_inferred": result.assertions_inferred,
                "semantic_maintenance_assertions_retracted": result.assertions_retracted,
                "semantic_maintenance_contradictions": result.contradictions,
                "semantic_maintenance_backlog": result.backlog_assertions,
                "semantic_maintenance_duration_ms": result.duration_ms,
                "semantic_maintenance_capability_versions": result.capability_versions,
            },
        )

    async def _record_state(
        self,
        result: SemanticMaintenanceResult,
        profile_key: str,
        lease: _MaintenanceLease,
        *,
        checkpoint_event_id: str | None = None,
        repair_cursor_revision_id: str | None = None,
        repair_active: bool = False,
        repair_mode: str | None = None,
        repair_scan_complete: bool = False,
        repair_checkpoint: AssertionCheckpoint | None = None,
        repair_reconcile_cursor_derivation_id: str | None = None,
        audit_assertion_id: str | None = None,
        audit_assertion_revision_id: str | None = None,
        audit_competitor_cursor_revision_id: str | None = None,
    ) -> SemanticMaintenanceResult:
        try:
            async with self._database.transaction():
                await self._renew_fenced_lease_in_transaction(lease)
                # Terminal readiness is published under the same canonical
                # tenant lock as assertion mutations.  The lock order is
                # deliberately lease then tenant, matching erasure and
                # maintenance-generated writes.  A source event arriving
                # after the earlier replay probe therefore either commits
                # before this lock and downgrades closure, or waits until the
                # PARTIAL/COMPLETE state below is already durable.
                async with self._assertions.inference_publication():
                    if result.complete:
                        current_source = await self._assertions.checkpoint()
                        current_cursor = await self._event_checkpoint(current_source)
                        proposed_cursor = AssertionCheckpoint(
                            self._assertions.tenant_id,
                            result.checkpoint_generation,
                            checkpoint_event_id,
                        )
                        if not self._checkpoint_matches(
                            proposed_cursor,
                            current_cursor,
                        ):
                            result = replace(
                                result,
                                status=SemanticMaintenanceStatus.PARTIAL,
                                reason="source_changed_during_closure",
                                source_generation=current_source.generation,
                                backlog_assertions=max(1, result.backlog_assertions),
                            )
                    # Keep terminal evidence and its authoritative state in one
                    # lease-fenced, tenant-serialized transaction. A stale
                    # holder cannot overwrite a successor's terminal row, and
                    # a crash cannot commit one half without the other.
                    await self._write_terminal_run_in_transaction(result, profile_key)
                    changed = await self._database.execute(
                        "INSERT INTO semantic_maintenance_state "
                        "(tenant_id, profile_key, checkpoint_generation, checkpoint_event_id, run_id, status, capability_versions, repair_cursor_revision_id, repair_active, repair_mode, repair_scan_complete, repair_checkpoint_generation, repair_checkpoint_event_id, repair_reconcile_cursor_derivation_id, audit_assertion_id, audit_assertion_revision_id, audit_competitor_cursor_revision_id, updated_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                        "ON CONFLICT(tenant_id) DO UPDATE SET "
                        "profile_key = excluded.profile_key, "
                        "checkpoint_generation = excluded.checkpoint_generation, "
                        "checkpoint_event_id = excluded.checkpoint_event_id, "
                        "run_id = excluded.run_id, status = excluded.status, "
                        "capability_versions = excluded.capability_versions, "
                        "repair_cursor_revision_id = excluded.repair_cursor_revision_id, "
                        "repair_active = excluded.repair_active, "
                        "repair_mode = excluded.repair_mode, "
                        "repair_scan_complete = excluded.repair_scan_complete, "
                        "repair_checkpoint_generation = excluded.repair_checkpoint_generation, "
                        "repair_checkpoint_event_id = excluded.repair_checkpoint_event_id, "
                        "repair_reconcile_cursor_derivation_id = excluded.repair_reconcile_cursor_derivation_id, "
                        "audit_assertion_id = excluded.audit_assertion_id, "
                        "audit_assertion_revision_id = excluded.audit_assertion_revision_id, "
                        "audit_competitor_cursor_revision_id = excluded.audit_competitor_cursor_revision_id, "
                        "updated_at = excluded.updated_at",
                        (
                            self._assertions.tenant_id,
                            profile_key,
                            result.checkpoint_generation,
                            checkpoint_event_id,
                            result.run_id,
                            result.status.value,
                            _json(result.capability_versions),
                            repair_cursor_revision_id,
                            int(repair_active),
                            repair_mode,
                            int(repair_scan_complete),
                            (
                                repair_checkpoint.generation
                                if repair_checkpoint is not None
                                else None
                            ),
                            (
                                repair_checkpoint.latest_event_id
                                if repair_checkpoint is not None
                                else None
                            ),
                            repair_reconcile_cursor_derivation_id,
                            audit_assertion_id,
                            audit_assertion_revision_id,
                            audit_competitor_cursor_revision_id,
                            _now(),
                        ),
                    )
        except TransactionError as error:
            if isinstance(error.__cause__, MaintenanceLeaseLostError):
                raise SemanticMaintenanceError("semantic_maintenance_lease_lost") from error
            raise
        if changed != 1:
            raise SemanticMaintenanceError("semantic_maintenance_lease_lost")
        self._log_result(result)
        return result

    async def _database_time(self) -> float:
        if self._database.backend_type == "postgres":
            value = await self._database.fetchval(
                "SELECT EXTRACT(EPOCH FROM CURRENT_TIMESTAMP)", ()
            )
        else:
            value = await self._database.fetchval(
                "SELECT CAST(strftime('%s', 'now') AS REAL)", ()
            )
        return float(value)

    async def _acquire_lease(self, holder_id: str) -> _MaintenanceLease | None:
        async with self._database.transaction():
            now = await self._database_time()
            row = await self._database.fetchone(
                "INSERT INTO semantic_maintenance_leases "
                "(tenant_id, holder_id, fencing_token, expires_at, updated_at) "
                "VALUES (?, ?, 1, ?, ?) "
                "ON CONFLICT(tenant_id) DO UPDATE SET "
                "holder_id = excluded.holder_id, "
                "fencing_token = semantic_maintenance_leases.fencing_token + 1, "
                "expires_at = excluded.expires_at, "
                "updated_at = excluded.updated_at "
                "WHERE semantic_maintenance_leases.expires_at <= ? "
                "RETURNING fencing_token",
                (
                    self._assertions.tenant_id,
                    holder_id,
                    now + self._LEASE_SECONDS,
                    _now(),
                    now,
                ),
            )
        return (
            None
            if row is None
            else _MaintenanceLease(holder_id, int(row[0]))
        )

    async def _release_lease(self, lease: _MaintenanceLease) -> None:
        await self._database.execute(
            "DELETE FROM semantic_maintenance_leases "
            "WHERE tenant_id = ? AND holder_id = ? AND fencing_token = ?",
            (self._assertions.tenant_id, lease.holder_id, lease.fencing_token),
        )

    async def _renew_fenced_lease_in_transaction(
        self, lease: _MaintenanceLease
    ) -> None:
        """Renew the exact lease while its maintenance write transaction is open."""
        now = await self._database_time()
        changed = await self._database.execute(
            "UPDATE semantic_maintenance_leases SET expires_at = ?, updated_at = ? "
            "WHERE tenant_id = ? AND holder_id = ? AND fencing_token = ? "
            "AND expires_at > ?",
            (
                now + self._LEASE_SECONDS,
                _now(),
                self._assertions.tenant_id,
                lease.holder_id,
                lease.fencing_token,
                now,
            ),
        )
        if changed != 1:
            raise MaintenanceLeaseLostError("semantic_maintenance_lease_lost")

    async def _renew_lease(self, lease: _MaintenanceLease) -> bool:
        async with self._database.transaction():
            now = await self._database_time()
            changed = await self._database.execute(
                "UPDATE semantic_maintenance_leases SET expires_at = ?, updated_at = ? "
                "WHERE tenant_id = ? AND holder_id = ? AND fencing_token = ? AND expires_at > ?",
                (
                    now + self._LEASE_SECONDS,
                    _now(),
                    self._assertions.tenant_id,
                    lease.holder_id,
                    lease.fencing_token,
                    now,
                ),
            )
        return changed == 1

    def _capability_versions(self) -> dict[str, str]:
        validation_pins = self._validation_artifact_pins()
        maintenance_budget = {
            "max_wall_time_seconds": self.limits.max_wall_time_seconds,
            "max_assertions": self.limits.max_assertions,
            "max_derivations": self.limits.max_derivations,
            "max_shapes": self.limits.max_shapes,
            "max_reports": self.limits.max_reports,
            "max_context_assertions": self.limits.max_context_assertions,
        }
        versions = {
            "semantic_maintenance": "v3",
            "maintenance_budget": _digest(maintenance_budget),
            "shape_set": f"{self.shape_set.identifier}@{self.shape_set.version}",
            "validation_capability": self.validation_capability,
            "validation_profile_version": self.validation_profile_version or "registry-selected",
            # A capability string is only a selector.  The selected profile,
            # its imports, and the shape-set import closure are the immutable
            # validation contract that must invalidate an earlier no-op when
            # the registry selects different bytes under the same selector.
            "validation_artifact_pins": _digest(
                [
                    {
                        "identifier": pin.identifier,
                        "version": str(pin.version),
                        "sha256": pin.sha256,
                    }
                    for pin in validation_pins
                ]
            ),
        }
        if self.inference_profile is not None:
            versions["inference_profile"] = self.inference_profile.key
            versions["rule_profile"] = self.inference_profile.rule_profile_version
            versions["ontology"] = (
                f"{self.inference_profile.ontology.namespace}@"
                f"{self.inference_profile.ontology.version}"
            )
        return versions

    def _validation_artifact_pins(self) -> tuple[ArtifactPin, ...]:
        """Resolve the exact, verified SHACL artifacts used by maintenance.

        ``validation_capability`` is intentionally not itself a durable
        identity: the registry can select a different pin for the same
        capability.  Mirror the validator's capability/shape compatibility
        checks before any early no-op or readiness decision, and retain every
        verified profile/shape import pin in a canonical order.
        """

        try:
            registry = get_knowledge_registry()
            profile = registry.select_capability(self.validation_capability)
            if profile.resource.kind is not ResourceKind.VALIDATION_PROFILE:
                raise SemanticMaintenanceError(
                    "semantic_maintenance_validation_capability_unavailable"
                )
            if (
                self.validation_profile_version is not None
                and str(profile.resource.version) != self.validation_profile_version
            ):
                raise SemanticMaintenanceError(
                    "semantic_maintenance_validation_capability_unavailable"
                )
            shapes = registry.resolve_capability(
                self.shape_set.identifier,
                self.shape_set.version,
            )
        except (ExperimentalCapabilityError, KnowledgeRegistryError) as error:
            raise SemanticMaintenanceError(
                "semantic_maintenance_validation_capability_unavailable"
            ) from error

        if shapes.resource.kind is not ResourceKind.SHAPE_SET:
            raise SemanticMaintenanceError(
                "semantic_maintenance_validation_capability_unavailable"
            )
        shape_imports = {
            (item.identifier, str(item.version)) for item in shapes.import_closure
        }
        if (profile.resource.identifier, str(profile.resource.version)) not in shape_imports:
            raise SemanticMaintenanceError(
                "semantic_maintenance_validation_capability_unavailable"
            )

        pins = {
            (pin.identifier, str(pin.version), pin.sha256): pin
            for pin in (
                profile.resource.pin,
                *profile.artifact_pins,
                shapes.resource.pin,
                *shapes.artifact_pins,
            )
        }
        return tuple(
            pins[key]
            for key in sorted(pins)
        )

    @staticmethod
    def _checkpoint_matches(
        checkpoint: AssertionCheckpoint,
        current: AssertionCheckpoint,
    ) -> bool:
        """Compare the assertion event cursor without mistaking DB generation.

        The store's generation increments for some canonical bookkeeping that
        emits no assertion-stream event.  Once a durable checkpoint has an
        event ID, that ID is therefore the authoritative stream position; a
        generation-only comparison would make an unchanged tenant look
        perpetually behind.  Legacy checkpoints with no event ID retain the
        conservative generation comparison.
        """

        if checkpoint.latest_event_id is not None or current.latest_event_id is not None:
            return checkpoint.latest_event_id == current.latest_event_id
        return checkpoint.generation == current.generation

    def _check_budget(self, started: float) -> None:
        if monotonic() - started > self.limits.max_wall_time_seconds:
            raise _MaintenanceBudgetExceeded()

    @staticmethod
    def _duration_ms(started: float) -> int:
        return max(0, int((monotonic() - started) * 1000))

    @staticmethod
    def _result(
        *,
        run_id: str,
        status: SemanticMaintenanceStatus,
        reason: str | None,
        source_generation: int,
        checkpoint_generation: int,
        capability_versions: dict[str, str],
        changes_consumed: int = 0,
        assertions_validated: int = 0,
        assertions_inferred: int = 0,
        assertions_retracted: int = 0,
        contradictions: int = 0,
        supersession_candidates: int = 0,
        expired_assertions: int = 0,
        orphan_provenance: int = 0,
        invalid_eligibility: int = 0,
        reports_created: int = 0,
        backlog_assertions: int = 0,
        backlog_reports: int = 0,
        duration_ms: int = 0,
    ) -> SemanticMaintenanceResult:
        return SemanticMaintenanceResult(
            run_id=run_id,
            status=status,
            reason=reason,
            source_generation=source_generation,
            checkpoint_generation=checkpoint_generation,
            changes_consumed=changes_consumed,
            assertions_validated=assertions_validated,
            assertions_inferred=assertions_inferred,
            assertions_retracted=assertions_retracted,
            contradictions=contradictions,
            supersession_candidates=supersession_candidates,
            expired_assertions=expired_assertions,
            orphan_provenance=orphan_provenance,
            invalid_eligibility=invalid_eligibility,
            reports_created=reports_created,
            backlog_assertions=backlog_assertions,
            backlog_reports=backlog_reports,
            duration_ms=duration_ms,
            capability_versions=capability_versions,
        )


def _is_expired(assertion: Assertion, now: datetime) -> bool:
    if assertion.valid_time is None or assertion.valid_time.end is None:
        return False
    return datetime.fromisoformat(
        assertion.valid_time.end.value.replace("Z", "+00:00")
    ) <= now


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(value: object) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


__all__ = [
    "SemanticMaintenanceError",
    "SemanticMaintenanceLimits",
    "SemanticMaintenanceResult",
    "SemanticMaintenanceService",
    "SemanticMaintenanceStatus",
    "SemanticMaintenanceTrainingReadiness",
    "maintenance_allows_prior_verified_snapshot",
    "maintenance_limits_from_config",
]
