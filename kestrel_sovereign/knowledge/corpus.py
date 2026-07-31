"""Backend-neutral governed assertion corpus contract for learning consumers.

This is deliberately a *read* boundary.  Consumers receive immutable,
tenant-scoped assertion examples together with the evidence needed to decide
whether a later adapter or other derived artifact must be invalidated.  The
contract never exposes a database handle, graph table, or filesystem path.

The caller owns the substantive release policy.  There is intentionally no
permissive default: a learning consumer must name the source, consent,
privacy, grounding, and derivation classes it accepts.  That makes a corpus
selection auditable and prevents a newly added source class from silently
becoming trainable data.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from time import monotonic
from types import MappingProxyType
from typing import Mapping, Protocol, Sequence

from .assertion import (
    Assertion,
    AssertionQuery,
    AssertionStatus,
    DerivedLineage,
    DirectLineage,
    EpistemicState,
    OntologyRef,
    SourceOccurrence,
    Visibility,
)
from .maintenance import SemanticMaintenanceTrainingReadiness
from .shacl_validation import ValidationState, ValidationWriteAction


CORPUS_SCHEMA_VERSION = 1
"""Wire/schema version for the governed corpus value contracts."""


class GovernedCorpusError(ValueError):
    """The requested corpus cannot be produced safely."""


class GovernedCorpusUnavailable(GovernedCorpusError):
    """A required semantic checkpoint or host capability is unavailable."""


class GovernedCorpusBudgetExceeded(GovernedCorpusError):
    """A bounded corpus request cannot complete within its declared budget."""


class CorpusEligibilityReason(str, Enum):
    INCLUDED = "included"
    LIFECYCLE = "lifecycle_ineligible"
    EPISTEMIC_STATE = "epistemic_state_disallowed"
    PRIVACY = "privacy_classification_disallowed"
    CONSENT = "consent_reference_disallowed"
    VISIBILITY = "visibility_disallowed"
    GROUNDING = "grounding_disallowed"
    SOURCE = "source_class_disallowed"
    DERIVATION = "derivation_disallowed"
    DERIVATION_INPUT = "derivation_input_ineligible"
    ONTOLOGY = "ontology_pin_disallowed"
    VALIDATION = "validation_not_conformant"


@dataclass(frozen=True, slots=True)
class CorpusCheckpoint:
    """Public, tenant-scoped cursor for a governed corpus stream."""

    tenant_id: str
    generation: int
    latest_event_id: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.tenant_id, str) or not self.tenant_id:
            raise GovernedCorpusError("checkpoint tenant_id must be non-empty")
        if type(self.generation) is not int or self.generation < 0:
            raise GovernedCorpusError("checkpoint generation must be a non-negative integer")
        if self.latest_event_id is not None and (
            not isinstance(self.latest_event_id, str) or not self.latest_event_id
        ):
            raise GovernedCorpusError("checkpoint latest_event_id must be non-empty or null")


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _ordered_texts(values: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise GovernedCorpusError(f"{name} must be an ordered sequence")
    if not values or any(not isinstance(value, str) or not value for value in values):
        raise GovernedCorpusError(f"{name} must contain non-empty values")
    normalized = tuple(values)
    if len(set(normalized)) != len(normalized):
        raise GovernedCorpusError(f"{name} must not contain duplicates")
    return tuple(sorted(normalized))


@dataclass(frozen=True, slots=True)
class GovernedCorpusLimits:
    """Hard request budgets; budget exhaustion fails closed instead of truncating."""

    max_assertions: int = 100
    max_serialized_bytes: int = 1_000_000
    max_wall_time_seconds: float = 10.0
    max_derivation_depth: int = 32

    def __post_init__(self) -> None:
        if type(self.max_assertions) is not int or not 1 <= self.max_assertions <= 999:
            raise GovernedCorpusError("max_assertions must be an integer in [1, 999]")
        if type(self.max_serialized_bytes) is not int or not 1 <= self.max_serialized_bytes <= 50_000_000:
            raise GovernedCorpusError(
                "max_serialized_bytes must be an integer in [1, 50000000]"
            )
        if (
            not isinstance(self.max_wall_time_seconds, (int, float))
            or isinstance(self.max_wall_time_seconds, bool)
            or not math.isfinite(self.max_wall_time_seconds)
            or self.max_wall_time_seconds <= 0
        ):
            raise GovernedCorpusError("max_wall_time_seconds must be positive and finite")
        if type(self.max_derivation_depth) is not int or not 1 <= self.max_derivation_depth <= 128:
            raise GovernedCorpusError("max_derivation_depth must be an integer in [1, 128]")


@dataclass(frozen=True, slots=True)
class GovernedCorpusPolicy:
    """Explicit release policy selected by a learning consumer/operator.

    All allow-lists are required.  A caller must consciously opt in to each
    source/consent/privacy/grounding class; this class is a policy value, not
    an inference from current database contents.
    """

    policy_id: str
    policy_version: str
    accepted_epistemic_states: tuple[EpistemicState | str, ...]
    accepted_visibility: tuple[Visibility | str, ...]
    accepted_privacy_classifications: tuple[str, ...]
    accepted_consent_references: tuple[str, ...]
    accepted_grounding_classes: tuple[str, ...]
    accepted_source_kinds: tuple[str, ...]
    accepted_ontology_pins: tuple[OntologyRef, ...]
    accepted_semantic_capability_versions: tuple[tuple[str, str], ...]
    allow_inferred: bool = False
    accepted_derivation_profiles: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.policy_id, str) or not self.policy_id:
            raise GovernedCorpusError("policy_id must be non-empty")
        if not isinstance(self.policy_version, str) or not self.policy_version:
            raise GovernedCorpusError("policy_version must be non-empty")
        states = tuple(EpistemicState(value) for value in self.accepted_epistemic_states)
        visibility = tuple(Visibility(value) for value in self.accepted_visibility)
        if not states or len(set(states)) != len(states):
            raise GovernedCorpusError("accepted_epistemic_states must be a non-empty set")
        if not visibility or len(set(visibility)) != len(visibility):
            raise GovernedCorpusError("accepted_visibility must be a non-empty set")
        if type(self.allow_inferred) is not bool:
            raise GovernedCorpusError("allow_inferred must be a boolean")
        ontology_pins = tuple(self.accepted_ontology_pins)
        if not ontology_pins or any(
            not isinstance(item, OntologyRef) for item in ontology_pins
        ):
            raise GovernedCorpusError(
                "accepted_ontology_pins must contain OntologyRef values"
            )
        if len(set(ontology_pins)) != len(ontology_pins):
            raise GovernedCorpusError("accepted_ontology_pins must not contain duplicates")
        capability_pairs = tuple(self.accepted_semantic_capability_versions)
        if not capability_pairs or any(
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not item[0]
            or not isinstance(item[1], str)
            or not item[1]
            for item in capability_pairs
        ):
            raise GovernedCorpusError(
                "accepted_semantic_capability_versions must contain non-empty string pairs"
            )
        if len({key for key, _ in capability_pairs}) != len(capability_pairs):
            raise GovernedCorpusError(
                "accepted_semantic_capability_versions must not duplicate keys"
            )
        profiles = self.accepted_derivation_profiles
        if self.allow_inferred and not profiles:
            raise GovernedCorpusError(
                "accepted_derivation_profiles is required when inferred assertions are allowed"
            )
        if not self.allow_inferred and profiles:
            raise GovernedCorpusError(
                "accepted_derivation_profiles requires allow_inferred=true"
            )
        object.__setattr__(self, "accepted_epistemic_states", tuple(sorted(states, key=str)))
        object.__setattr__(self, "accepted_visibility", tuple(sorted(visibility, key=str)))
        object.__setattr__(
            self, "accepted_privacy_classifications",
            _ordered_texts(self.accepted_privacy_classifications, "accepted_privacy_classifications"),
        )
        object.__setattr__(
            self, "accepted_consent_references",
            _ordered_texts(self.accepted_consent_references, "accepted_consent_references"),
        )
        object.__setattr__(
            self, "accepted_grounding_classes",
            _ordered_texts(self.accepted_grounding_classes, "accepted_grounding_classes"),
        )
        object.__setattr__(
            self, "accepted_source_kinds",
            _ordered_texts(self.accepted_source_kinds, "accepted_source_kinds"),
        )
        object.__setattr__(
            self, "accepted_derivation_profiles",
            _ordered_texts(profiles, "accepted_derivation_profiles") if profiles else (),
        )
        object.__setattr__(
            self,
            "accepted_ontology_pins",
            tuple(
                sorted(
                    ontology_pins,
                    key=lambda item: (
                        item.namespace,
                        item.version,
                        item.content_digest,
                        item.compatibility_profile,
                    ),
                )
            ),
        )
        object.__setattr__(
            self,
            "accepted_semantic_capability_versions",
            tuple(sorted(capability_pairs)),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": CORPUS_SCHEMA_VERSION,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "accepted_epistemic_states": [value.value for value in self.accepted_epistemic_states],
            "accepted_visibility": [value.value for value in self.accepted_visibility],
            "accepted_privacy_classifications": list(self.accepted_privacy_classifications),
            "accepted_consent_references": list(self.accepted_consent_references),
            "accepted_grounding_classes": list(self.accepted_grounding_classes),
            "accepted_source_kinds": list(self.accepted_source_kinds),
            "accepted_ontology_pins": [
                item.to_mapping() for item in self.accepted_ontology_pins
            ],
            "accepted_semantic_capability_versions": [
                {"name": name, "version": version}
                for name, version in self.accepted_semantic_capability_versions
            ],
            "allow_inferred": self.allow_inferred,
            "accepted_derivation_profiles": list(self.accepted_derivation_profiles),
        }

    @property
    def digest(self) -> str:
        return _digest(self.to_mapping())


@dataclass(frozen=True, slots=True)
class CorpusValidationStatus:
    """Latest validation disposition for one current assertion revision."""

    state: ValidationState | None
    action: ValidationWriteAction | None
    shape_set_id: str | None = None
    shape_set_version: str | None = None
    validation_profile_version: str | None = None


@dataclass(frozen=True, slots=True)
class CorpusEligibilityDecision:
    """Content-free explanation for one included or excluded assertion."""

    included: bool
    reason: CorpusEligibilityReason
    policy_digest: str
    validation_state: ValidationState | None


@dataclass(frozen=True, slots=True)
class GovernedCorpusExample:
    """One immutable, lineage-carrying governed example."""

    assertion: Assertion
    source_occurrences: tuple[SourceOccurrence, ...]
    validation: CorpusValidationStatus
    decision: CorpusEligibilityDecision
    content_hash: str
    split_key: str

    def __post_init__(self) -> None:
        if not self.decision.included or self.decision.reason is not CorpusEligibilityReason.INCLUDED:
            raise GovernedCorpusError("corpus examples must carry an included eligibility decision")
        if tuple(sorted(self.source_occurrences, key=lambda item: (item.received_at.value, item.source_occurrence_id))) != self.source_occurrences:
            raise GovernedCorpusError("source_occurrences must use canonical ordering")

    def canonical_mapping(self) -> dict[str, object]:
        return {
            "assertion": self.assertion.to_mapping(),
            "sources": [source.to_mapping() for source in self.source_occurrences],
            "validation": {
                "state": self.validation.state.value if self.validation.state else None,
                "action": self.validation.action.value if self.validation.action else None,
                "shape_set_id": self.validation.shape_set_id,
                "shape_set_version": self.validation.shape_set_version,
                "validation_profile_version": self.validation.validation_profile_version,
            },
            "policy_digest": self.decision.policy_digest,
        }


@dataclass(frozen=True, slots=True)
class GovernedCorpusObservability:
    """Privacy-safe request summary: counts and hashes only, never examples."""

    considered: int
    included: int
    excluded: Mapping[str, int]
    snapshot_hash: str
    policy_digest: str
    checkpoint_generation: int

    def __post_init__(self) -> None:
        if self.considered < 0 or self.included < 0 or self.checkpoint_generation < 0:
            raise GovernedCorpusError("observability counts must be non-negative")
        normalized = dict(self.excluded)
        if any(not isinstance(key, str) or not key or type(value) is not int or value < 0 for key, value in normalized.items()):
            raise GovernedCorpusError("excluded observability must contain non-negative integer counts")
        object.__setattr__(self, "excluded", MappingProxyType(dict(sorted(normalized.items()))))


@dataclass(frozen=True, slots=True)
class GovernedCorpusSnapshot:
    """Immutable, deterministic corpus at one verified semantic checkpoint."""

    schema_version: int
    tenant_id: str
    checkpoint: CorpusCheckpoint
    capability_versions: Mapping[str, str]
    policy: GovernedCorpusPolicy
    examples: tuple[GovernedCorpusExample, ...]
    snapshot_hash: str
    observability: GovernedCorpusObservability
    verified: bool = True

    def __post_init__(self) -> None:
        if self.schema_version != CORPUS_SCHEMA_VERSION:
            raise GovernedCorpusError("unsupported corpus snapshot schema version")
        if self.checkpoint.tenant_id != self.tenant_id:
            raise GovernedCorpusError("snapshot checkpoint tenant does not match snapshot tenant")
        versions = dict(self.capability_versions)
        if any(not isinstance(key, str) or not key or not isinstance(value, str) or not value for key, value in versions.items()):
            raise GovernedCorpusError("snapshot capability_versions must be non-empty string mappings")
        object.__setattr__(self, "capability_versions", MappingProxyType(dict(sorted(versions.items()))))
        if tuple(sorted(self.examples, key=lambda item: (item.assertion.assertion_id, item.assertion.revision_id))) != self.examples:
            raise GovernedCorpusError("corpus examples must use canonical ordering")
        if not self.verified:
            raise GovernedCorpusError("a corpus snapshot cannot represent unverified content")


@dataclass(frozen=True, slots=True)
class GovernedCorpusTombstone:
    """A first-class lifecycle/eligibility invalidation for a prior snapshot."""

    event_id: str
    assertion_id: str | None
    revision_id: str | None
    operation: str
    generation: int
    reason: str


@dataclass(frozen=True, slots=True)
class GovernedCorpusDelta:
    """Incremental additions and invalidations after an immutable snapshot."""

    since_checkpoint: CorpusCheckpoint
    checkpoint: CorpusCheckpoint
    additions: tuple[GovernedCorpusExample, ...]
    tombstones: tuple[GovernedCorpusTombstone, ...]
    snapshot_hash: str
    observability: GovernedCorpusObservability

    @property
    def since_generation(self) -> int:
        return self.since_checkpoint.generation

    @property
    def checkpoint_generation(self) -> int:
        return self.checkpoint.generation


class GovernedCorpusStorage(Protocol):
    """Public host capability implemented by ``AsyncStorage`` and its privacy facade."""

    async def assertion_event_checkpoint(self) -> CorpusCheckpoint: ...

    async def assertion_changes_after(self, checkpoint, *, limit: int = 100): ...

    async def assertion_inference_inputs(self, query: AssertionQuery | None = None) -> list[Assertion]: ...

    async def get_derivation_inputs(self, revision_id: str) -> list[Assertion]: ...

    async def list_assertion_revision_sources(self, revision_id: str) -> list[SourceOccurrence]: ...

    async def list_assertion_revision_sources_batch(self, revision_ids: Sequence[str]) -> Mapping[str, Sequence[SourceOccurrence]]: ...

    async def assertion_validation_statuses(self, assertions: Sequence[Assertion]) -> Mapping[str, CorpusValidationStatus]: ...

    async def semantic_maintenance_training_readiness(self, inference_profile, *, inference_limits=None, maintenance_limits=None, allow_prior_verified_snapshot: bool = False, expected_checkpoint: CorpusCheckpoint | None = None) -> SemanticMaintenanceTrainingReadiness: ...

    async def semantic_maintenance_capability_versions(self, inference_profile, *, inference_limits=None, maintenance_limits=None) -> Mapping[str, str]: ...


class GovernedAssertionCorpusService:
    """Construct snapshots/deltas exclusively through a host storage capability."""

    def __init__(self, storage: GovernedCorpusStorage) -> None:
        self._storage = storage

    async def snapshot(
        self,
        *,
        policy: GovernedCorpusPolicy,
        inference_profile,
        limits: GovernedCorpusLimits = GovernedCorpusLimits(),
        inference_limits=None,
        maintenance_limits=None,
        prior_verified_snapshot: GovernedCorpusSnapshot | None = None,
        allow_prior_verified_snapshot: bool = False,
    ) -> GovernedCorpusSnapshot:
        if not isinstance(limits, GovernedCorpusLimits):
            raise GovernedCorpusError("limits must be GovernedCorpusLimits")
        try:
            async with asyncio.timeout(limits.max_wall_time_seconds):
                return await self._snapshot(
                    policy=policy,
                    inference_profile=inference_profile,
                    limits=limits,
                    inference_limits=inference_limits,
                    maintenance_limits=maintenance_limits,
                    prior_verified_snapshot=prior_verified_snapshot,
                    allow_prior_verified_snapshot=allow_prior_verified_snapshot,
                )
        except TimeoutError as error:
            raise GovernedCorpusBudgetExceeded(
                "max_wall_time_seconds exhausted"
            ) from error

    async def _snapshot(
        self,
        *,
        policy: GovernedCorpusPolicy,
        inference_profile,
        limits: GovernedCorpusLimits,
        inference_limits=None,
        maintenance_limits=None,
        prior_verified_snapshot: GovernedCorpusSnapshot | None = None,
        allow_prior_verified_snapshot: bool = False,
    ) -> GovernedCorpusSnapshot:
        if not isinstance(policy, GovernedCorpusPolicy):
            raise GovernedCorpusError("policy must be GovernedCorpusPolicy")
        if type(allow_prior_verified_snapshot) is not bool:
            raise GovernedCorpusError("allow_prior_verified_snapshot must be a boolean")
        if prior_verified_snapshot is not None and not isinstance(
            prior_verified_snapshot, GovernedCorpusSnapshot
        ):
            raise GovernedCorpusError(
                "prior_verified_snapshot must be GovernedCorpusSnapshot or null"
            )
        started = monotonic()
        checkpoint = await self._event_checkpoint()
        readiness = await self._storage.semantic_maintenance_training_readiness(
            inference_profile,
            inference_limits=inference_limits,
            maintenance_limits=maintenance_limits,
            expected_checkpoint=checkpoint,
        )
        capability_versions = dict(await self._storage.semantic_maintenance_capability_versions(
            inference_profile,
            inference_limits=inference_limits,
            maintenance_limits=maintenance_limits,
        ))
        if capability_versions != dict(
            policy.accepted_semantic_capability_versions
        ):
            raise GovernedCorpusUnavailable(
                "semantic_capability_versions_mismatch"
            )
        if not readiness.ready:
            # A caller-held dataclass is not host-verifiable durable evidence.
            # Do not turn an untrusted object (even one with matching hashes)
            # into a stale-corpus escape hatch.  A future persisted manifest
            # registry may implement this explicitly; until then the only safe
            # policy is fail-closed.
            if allow_prior_verified_snapshot or prior_verified_snapshot is not None:
                raise GovernedCorpusUnavailable(
                    "prior_verified_snapshot_reuse_requires_host_persistence"
                )
            raise GovernedCorpusUnavailable(
                readiness.reason or "semantic_maintenance_unverified"
            )
        candidates = await self._storage.assertion_inference_inputs(
            AssertionQuery(limit=limits.max_assertions + 1)
        )
        if len(candidates) > limits.max_assertions:
            # A full page proves only that the database may have more data.  A
            # corpus must not silently train on a prefix; callers raise the
            # declared budget then request a new immutable snapshot.
            raise GovernedCorpusBudgetExceeded("max_assertions exhausted")
        snapshot = await self._build_snapshot(
            candidates=candidates,
            checkpoint=checkpoint,
            capability_versions=capability_versions,
            policy=policy,
            limits=limits,
            started=started,
        )
        final_checkpoint = await self._event_checkpoint()
        if not self._same_checkpoint(checkpoint, final_checkpoint):
            raise GovernedCorpusUnavailable("semantic_checkpoint_changed_during_snapshot")
        return snapshot

    async def changes_since(
        self,
        snapshot: GovernedCorpusSnapshot,
        *,
        policy: GovernedCorpusPolicy,
        inference_profile,
        limits: GovernedCorpusLimits = GovernedCorpusLimits(),
        inference_limits=None,
        maintenance_limits=None,
    ) -> GovernedCorpusDelta:
        if not isinstance(limits, GovernedCorpusLimits):
            raise GovernedCorpusError("limits must be GovernedCorpusLimits")
        try:
            async with asyncio.timeout(limits.max_wall_time_seconds):
                return await self._changes_since(
                    snapshot,
                    policy=policy,
                    inference_profile=inference_profile,
                    limits=limits,
                    inference_limits=inference_limits,
                    maintenance_limits=maintenance_limits,
                )
        except TimeoutError as error:
            raise GovernedCorpusBudgetExceeded(
                "max_wall_time_seconds exhausted"
            ) from error

    async def _changes_since(
        self,
        snapshot: GovernedCorpusSnapshot,
        *,
        policy: GovernedCorpusPolicy,
        inference_profile,
        limits: GovernedCorpusLimits,
        inference_limits=None,
        maintenance_limits=None,
    ) -> GovernedCorpusDelta:
        if not isinstance(snapshot, GovernedCorpusSnapshot):
            raise GovernedCorpusError("snapshot must be GovernedCorpusSnapshot")
        if snapshot.policy.digest != policy.digest:
            raise GovernedCorpusError("incremental reads require the same corpus policy")
        started = monotonic()
        head = await self._event_checkpoint()
        readiness = await self._storage.semantic_maintenance_training_readiness(
            inference_profile,
            inference_limits=inference_limits,
            maintenance_limits=maintenance_limits,
            expected_checkpoint=head,
        )
        if not readiness.ready:
            raise GovernedCorpusUnavailable(readiness.reason or "semantic_maintenance_unverified")
        capability_versions = dict(await self._storage.semantic_maintenance_capability_versions(
            inference_profile,
            inference_limits=inference_limits,
            maintenance_limits=maintenance_limits,
        ))
        if dict(snapshot.capability_versions) != capability_versions:
            raise GovernedCorpusUnavailable("semantic_capability_versions_mismatch")
        if capability_versions != dict(
            policy.accepted_semantic_capability_versions
        ):
            raise GovernedCorpusUnavailable(
                "semantic_capability_versions_mismatch"
            )
        changes = await self._storage.assertion_changes_after(
            snapshot.checkpoint,
            limit=limits.max_assertions + 1,
        )
        if len(changes) > limits.max_assertions:
            raise GovernedCorpusBudgetExceeded("max_assertions exhausted")
        if changes:
            last_change = changes[-1]
            if last_change.event_id != head.latest_event_id:
                raise GovernedCorpusBudgetExceeded(
                    "change page does not reach the semantic checkpoint"
                )
        elif not self._same_checkpoint(snapshot.checkpoint, head):
            raise GovernedCorpusUnavailable(
                "semantic_change_stream_did_not_reach_checkpoint"
            )
        changed_ids = tuple(sorted({change.assertion_id for change in changes if change.eligible and change.assertion_id}))
        additions: tuple[GovernedCorpusExample, ...] = ()
        excluded: dict[str, int] = {}
        if changed_ids:
            candidates = await self._storage.assertion_inference_inputs(
                AssertionQuery(assertion_ids=changed_ids, limit=limits.max_assertions)
            )
            built, excluded = await self._examples(candidates, policy, limits, started)
            additions = tuple(built)
        tombstones = tuple(
            GovernedCorpusTombstone(
                event_id=change.event_id,
                assertion_id=change.assertion_id,
                revision_id=change.revision_id,
                operation=change.operation,
                generation=change.generation,
                reason=self._tombstone_reason(change.operation),
            )
            for change in changes
            if not change.eligible
        )
        snapshot_hash = _digest({
            "base": snapshot.snapshot_hash,
            "checkpoint": [head.generation, head.latest_event_id],
            "additions": [example.content_hash for example in additions],
            "tombstones": [tombstone.__dict__ if hasattr(tombstone, "__dict__") else {
                "event_id": tombstone.event_id, "assertion_id": tombstone.assertion_id,
                "revision_id": tombstone.revision_id, "operation": tombstone.operation,
                "generation": tombstone.generation, "reason": tombstone.reason,
            } for tombstone in tombstones],
        })
        observability = GovernedCorpusObservability(
            considered=len(changes), included=len(additions), excluded=dict(sorted(excluded.items())),
            snapshot_hash=snapshot_hash, policy_digest=policy.digest,
            checkpoint_generation=head.generation,
        )
        final_checkpoint = await self._event_checkpoint()
        if not self._same_checkpoint(head, final_checkpoint):
            raise GovernedCorpusUnavailable("semantic_checkpoint_changed_during_incremental_read")
        return GovernedCorpusDelta(
            snapshot.checkpoint,
            CorpusCheckpoint(head.tenant_id, head.generation, head.latest_event_id),
            additions,
            tombstones,
            snapshot_hash,
            observability,
        )

    async def _build_snapshot(self, *, candidates, checkpoint, capability_versions, policy, limits, started) -> GovernedCorpusSnapshot:
        examples, excluded = await self._examples(candidates, policy, limits, started)
        canonical = {
            "schema_version": CORPUS_SCHEMA_VERSION,
            "tenant_id": checkpoint.tenant_id,
            "checkpoint": [checkpoint.generation, checkpoint.latest_event_id],
            "capability_versions": dict(sorted(capability_versions.items())),
            "policy": policy.to_mapping(),
            "examples": [example.content_hash for example in examples],
        }
        snapshot_hash = _digest(canonical)
        observability = GovernedCorpusObservability(
            considered=len(candidates), included=len(examples), excluded=dict(sorted(excluded.items())),
            snapshot_hash=snapshot_hash, policy_digest=policy.digest,
            checkpoint_generation=checkpoint.generation,
        )
        return GovernedCorpusSnapshot(
            CORPUS_SCHEMA_VERSION, checkpoint.tenant_id,
            CorpusCheckpoint(checkpoint.tenant_id, checkpoint.generation, checkpoint.latest_event_id),
            dict(sorted(capability_versions.items())), policy,
            tuple(examples), snapshot_hash, observability,
        )

    async def _examples(self, candidates, policy, limits, started) -> tuple[list[GovernedCorpusExample], dict[str, int]]:
        self._check_time(limits, started)
        validations = dict(
            await self._storage.assertion_validation_statuses(tuple(candidates))
        )
        direct_revision_ids = tuple(
            assertion.revision_id
            for assertion in candidates
            if isinstance(assertion.lineage, DirectLineage)
            and self._eligibility(
                assertion,
                validations.get(
                    assertion.assertion_id, CorpusValidationStatus(None, None)
                ),
                policy,
            )
            is CorpusEligibilityReason.INCLUDED
        )
        source_cache = await self._source_batch(direct_revision_ids)
        examples: list[GovernedCorpusExample] = []
        excluded: dict[str, int] = {}
        used_bytes = 0
        governed_revisions: set[str] = set()
        for assertion in sorted(candidates, key=lambda item: (item.assertion_id, item.revision_id)):
            self._check_time(limits, started)
            validation = validations.get(
                assertion.assertion_id, CorpusValidationStatus(None, None)
            )
            reason, sources = await self._govern_lineage(
                assertion,
                policy=policy,
                validation=validation,
                validation_cache=validations,
                limits=limits,
                started=started,
                depth=0,
                stack=frozenset(),
                governed_revisions=governed_revisions,
                source_cache=source_cache,
            )
            if reason is not CorpusEligibilityReason.INCLUDED:
                excluded[reason.value] = excluded.get(reason.value, 0) + 1
                continue
            decision = CorpusEligibilityDecision(True, reason, policy.digest, validation.state)
            content_hash = _digest({
                "assertion": assertion.to_mapping(),
                "sources": [source.to_mapping() for source in sources],
                "validation": [validation.state.value if validation.state else None, validation.action.value if validation.action else None, validation.shape_set_id, validation.shape_set_version, validation.validation_profile_version],
                "policy_digest": policy.digest,
            })
            split_key = _digest({"v": 1, "assertion_id": assertion.assertion_id, "revision_id": assertion.revision_id})
            example = GovernedCorpusExample(assertion, sources, validation, decision, content_hash, split_key)
            used_bytes += len(_canonical_json(example.canonical_mapping()).encode("utf-8"))
            if used_bytes > limits.max_serialized_bytes:
                raise GovernedCorpusBudgetExceeded("max_serialized_bytes exhausted")
            examples.append(example)
        return examples, excluded

    async def _govern_lineage(
        self,
        assertion: Assertion,
        *,
        policy: GovernedCorpusPolicy,
        validation: CorpusValidationStatus,
        validation_cache: dict[str, CorpusValidationStatus],
        limits: GovernedCorpusLimits,
        started: float,
        depth: int,
        stack: frozenset[str],
        governed_revisions: set[str],
        source_cache: dict[str, tuple[SourceOccurrence, ...]],
    ) -> tuple[CorpusEligibilityReason, tuple[SourceOccurrence, ...]]:
        """Enforce policy recursively over exact derivation revisions."""
        self._check_time(limits, started)
        if depth > limits.max_derivation_depth or assertion.revision_id in stack:
            return CorpusEligibilityReason.DERIVATION_INPUT, ()
        governed_revisions.add(assertion.revision_id)
        if len(governed_revisions) > limits.max_assertions:
            raise GovernedCorpusBudgetExceeded(
                "max_assertions exhausted by derivation traversal"
            )
        reason = self._eligibility(assertion, validation, policy)
        if reason is not CorpusEligibilityReason.INCLUDED:
            return reason, ()
        if isinstance(assertion.lineage, DirectLineage):
            if assertion.revision_id not in source_cache:
                source_cache.update(await self._source_batch((assertion.revision_id,)))
            sources = source_cache.get(assertion.revision_id, ())
            if not sources or any(
                source.source_kind not in policy.accepted_source_kinds
                for source in sources
            ):
                return CorpusEligibilityReason.SOURCE, ()
            return CorpusEligibilityReason.INCLUDED, sources

        inputs = tuple(
            await self._storage.get_derivation_inputs(assertion.revision_id)
        )
        expected_revision_ids = assertion.lineage.input_revision_ids
        if tuple(item.revision_id for item in inputs) != expected_revision_ids:
            return CorpusEligibilityReason.DERIVATION_INPUT, ()
        prospective_work = governed_revisions.union(expected_revision_ids)
        if len(prospective_work) > limits.max_assertions:
            raise GovernedCorpusBudgetExceeded(
                "max_assertions exhausted by derivation traversal"
            )
        current_inputs = await self._storage.assertion_inference_inputs(
            AssertionQuery(
                assertion_ids=tuple(item.assertion_id for item in inputs),
                limit=len(inputs),
            )
        )
        current_by_assertion = {
            item.assertion_id: item.revision_id for item in current_inputs
        }
        if any(
            current_by_assertion.get(item.assertion_id) != item.revision_id
            for item in inputs
        ):
            return CorpusEligibilityReason.DERIVATION_INPUT, ()
        missing_validation_assertions = tuple(
            sorted(
                (
                    item
                    for item in inputs
                    if item.assertion_id not in validation_cache
                ),
                key=lambda item: (item.assertion_id, item.revision_id),
            )
        )
        if missing_validation_assertions:
            validation_cache.update(
                await self._storage.assertion_validation_statuses(
                    missing_validation_assertions
                )
            )
        missing_direct_sources = tuple(
            item.revision_id
            for item in inputs
            if isinstance(item.lineage, DirectLineage)
            and item.revision_id not in source_cache
            and self._eligibility(
                item,
                validation_cache.get(
                    item.assertion_id, CorpusValidationStatus(None, None)
                ),
                policy,
            )
            is CorpusEligibilityReason.INCLUDED
        )
        if missing_direct_sources:
            source_cache.update(await self._source_batch(missing_direct_sources))
        transitive_sources: dict[str, SourceOccurrence] = {}
        nested_stack = stack.union((assertion.revision_id,))
        for item in inputs:
            item_validation = validation_cache.get(
                item.assertion_id, CorpusValidationStatus(None, None)
            )
            item_reason, item_sources = await self._govern_lineage(
                item,
                policy=policy,
                validation=item_validation,
                validation_cache=validation_cache,
                limits=limits,
                started=started,
                depth=depth + 1,
                stack=nested_stack,
                governed_revisions=governed_revisions,
                source_cache=source_cache,
            )
            if item_reason is not CorpusEligibilityReason.INCLUDED:
                return CorpusEligibilityReason.DERIVATION_INPUT, ()
            for source in item_sources:
                transitive_sources[source.source_occurrence_id] = source
        return (
            CorpusEligibilityReason.INCLUDED,
            tuple(
                sorted(
                    transitive_sources.values(),
                    key=lambda item: (
                        item.received_at.value,
                        item.source_occurrence_id,
                    ),
                )
            ),
        )

    @staticmethod
    def _eligibility(assertion: Assertion, validation: CorpusValidationStatus, policy: GovernedCorpusPolicy) -> CorpusEligibilityReason:
        if assertion.status is not AssertionStatus.ACTIVE:
            return CorpusEligibilityReason.LIFECYCLE
        if validation.state is not ValidationState.CONFORMS or validation.action not in (ValidationWriteAction.ACCEPT, ValidationWriteAction.ACCEPT_WITH_REPORT):
            return CorpusEligibilityReason.VALIDATION
        if assertion.epistemic_state not in policy.accepted_epistemic_states:
            return CorpusEligibilityReason.EPISTEMIC_STATE
        if assertion.visibility not in policy.accepted_visibility:
            return CorpusEligibilityReason.VISIBILITY
        if assertion.privacy_classification not in policy.accepted_privacy_classifications:
            return CorpusEligibilityReason.PRIVACY
        if assertion.release_policy_reference not in policy.accepted_consent_references:
            return CorpusEligibilityReason.CONSENT
        if assertion.confidence_basis not in policy.accepted_grounding_classes:
            return CorpusEligibilityReason.GROUNDING
        if assertion.ontology_version not in policy.accepted_ontology_pins:
            return CorpusEligibilityReason.ONTOLOGY
        if isinstance(assertion.lineage, DerivedLineage):
            if not policy.allow_inferred or assertion.lineage.profile_version not in policy.accepted_derivation_profiles:
                return CorpusEligibilityReason.DERIVATION
        elif not isinstance(assertion.lineage, DirectLineage):
            return CorpusEligibilityReason.DERIVATION
        return CorpusEligibilityReason.INCLUDED

    async def _source_batch(
        self, revision_ids: Sequence[str]
    ) -> dict[str, tuple[SourceOccurrence, ...]]:
        """Load exact-revision provenance in one tenant-bound host call."""
        if not revision_ids:
            return {}
        normalized = tuple(dict.fromkeys(revision_ids))
        rows = await self._storage.list_assertion_revision_sources_batch(normalized)
        return {
            revision_id: tuple(
                sorted(
                    rows.get(revision_id, ()),
                    key=lambda item: (
                        item.received_at.value,
                        item.source_occurrence_id,
                    ),
                )
            )
            for revision_id in normalized
        }

    @staticmethod
    def _tombstone_reason(operation: str) -> str:
        """Preserve the persisted invalidation cause without inventing one."""
        if not isinstance(operation, str) or not operation:
            return "unknown_invalidation"
        return operation

    @staticmethod
    def _check_time(limits: GovernedCorpusLimits, started: float) -> None:
        if monotonic() - started > limits.max_wall_time_seconds:
            raise GovernedCorpusBudgetExceeded("max_wall_time_seconds exhausted")

    async def _event_checkpoint(self) -> CorpusCheckpoint:
        try:
            checkpoint = await self._storage.assertion_event_checkpoint()
        except Exception as error:
            if isinstance(error, GovernedCorpusError):
                raise
            raise GovernedCorpusUnavailable(
                "semantic_event_checkpoint_unavailable"
            ) from error
        if not isinstance(checkpoint, CorpusCheckpoint):
            raise GovernedCorpusUnavailable("semantic_event_checkpoint_invalid")
        return checkpoint

    @staticmethod
    def _same_checkpoint(left, right) -> bool:
        """Compare exact public cursors without assuming a backend row order."""
        return (
            left.tenant_id == right.tenant_id
            and left.generation == right.generation
            and left.latest_event_id == right.latest_event_id
        )

__all__ = [
    "CORPUS_SCHEMA_VERSION", "CorpusCheckpoint", "CorpusEligibilityDecision", "CorpusEligibilityReason",
    "CorpusValidationStatus", "GovernedAssertionCorpusService", "GovernedCorpusBudgetExceeded",
    "GovernedCorpusDelta", "GovernedCorpusError", "GovernedCorpusExample", "GovernedCorpusLimits",
    "GovernedCorpusObservability", "GovernedCorpusPolicy", "GovernedCorpusSnapshot",
    "GovernedCorpusStorage", "GovernedCorpusTombstone", "GovernedCorpusUnavailable",
]
