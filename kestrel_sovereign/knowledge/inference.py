"""Deterministic, bounded RDFS and allowlisted OWL 2 RL materialization.

This module intentionally implements the small, reviewed rule set itself.
Calling an RDF/OWL library's closure helper would hide both the rule identity
and the complete premise set needed to revoke an inferred assertion later.
The materializer consumes only an already tenant-bound assertion store; it has
no tenant argument and never reads an ontology from the network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import logging
import math
from collections import deque
from importlib import resources
from time import monotonic
from typing import Callable, Iterable, Mapping, Sequence, TYPE_CHECKING
from uuid import uuid4

from .assertion import (
    Assertion,
    AssertionQuery,
    AssertionObject,
    AssertionStatus,
    DerivedLineage,
    EpistemicState,
    IRI,
    Literal,
    OntologyRef,
    derive_assertion_id,
)
from .registry import (
    KnowledgeRegistryError,
    ResourceKind,
    SemanticVersion,
    get_knowledge_registry,
)

if TYPE_CHECKING:
    from kestrel_sovereign.storage.async_assertion_store import AsyncAssertionStore


logger = logging.getLogger(__name__)

ENGINE_VERSION = "semantic-kb-materializer-v1"
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
RDFS_SUBCLASS = "http://www.w3.org/2000/01/rdf-schema#subClassOf"
RDFS_SUBPROPERTY = "http://www.w3.org/2000/01/rdf-schema#subPropertyOf"
RDFS_DOMAIN = "http://www.w3.org/2000/01/rdf-schema#domain"
RDFS_RANGE = "http://www.w3.org/2000/01/rdf-schema#range"
OWL_EQUIVALENT_CLASS = "http://www.w3.org/2002/07/owl#equivalentClass"
OWL_EQUIVALENT_PROPERTY = "http://www.w3.org/2002/07/owl#equivalentProperty"
OWL_INVERSE_OF = "http://www.w3.org/2002/07/owl#inverseOf"
OWL_TRANSITIVE_PROPERTY = "http://www.w3.org/2002/07/owl#TransitiveProperty"
OWL_SYMMETRIC_PROPERTY = "http://www.w3.org/2002/07/owl#SymmetricProperty"
OWL_PROPERTY_CHAIN_AXIOM = "http://www.w3.org/2002/07/owl#propertyChainAxiom"


class InferenceError(ValueError):
    """The pinned inference contract or a materialization request is invalid."""


class ClosureStatus(str, Enum):
    RUNNING = "running"
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class InferenceLimits:
    """Finite operator-configured limits for one closure calculation."""

    max_source_assertions: int = 10_000
    max_iterations: int = 64
    max_generated_assertions: int = 100_000
    max_wall_time_seconds: float = 30.0
    max_memory_items: int = 200_000

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value < 1
            for value in (
                self.max_source_assertions,
                self.max_iterations,
                self.max_generated_assertions,
                self.max_memory_items,
            )
        ):
            raise InferenceError("inference integer limits must be positive integers")
        if (
            not isinstance(self.max_wall_time_seconds, (int, float))
            or isinstance(self.max_wall_time_seconds, bool)
            or not math.isfinite(self.max_wall_time_seconds)
            or self.max_wall_time_seconds <= 0
        ):
            raise InferenceError("max_wall_time_seconds must be a positive finite number")


@dataclass(frozen=True, slots=True)
class InferenceProfile:
    """An exact, operator-selected rule and ontology interpretation.

    Resource versions are required values rather than capabilities selected by
    name.  This is what prevents a package upgrade from silently widening a
    tenant's inference profile.
    """

    ontology: OntologyRef
    rdfs_version: str
    owl2rl_version: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.ontology, OntologyRef):
            raise InferenceError("inference profile requires an OntologyRef")
        if not isinstance(self.rdfs_version, str) or not self.rdfs_version:
            raise InferenceError("rdfs_version must be an exact non-empty version")
        if self.owl2rl_version is not None and (
            not isinstance(self.owl2rl_version, str) or not self.owl2rl_version
        ):
            raise InferenceError("owl2rl_version must be an exact version or null")
        try:
            SemanticVersion.parse(self.rdfs_version)
            if self.owl2rl_version is not None:
                SemanticVersion.parse(self.owl2rl_version)
        except ValueError as error:
            raise InferenceError("inference profile rule versions must be exact semantic versions") from error

    @property
    def key(self) -> str:
        payload = {
            "ontology": self.ontology.to_mapping(),
            "rdfs_version": self.rdfs_version,
            "owl2rl_version": self.owl2rl_version,
        }
        return "sha256:" + hashlib.sha256(_canonical_json(payload).encode()).hexdigest()

    @property
    def rule_profile_version(self) -> str:
        return (
            f"rdfs-v1@{self.rdfs_version}"
            + (f"+owl2rl-kestrel-v1@{self.owl2rl_version}" if self.owl2rl_version else "")
        )


def inference_profile_from_config(config: Mapping[str, object]) -> InferenceProfile | None:
    """Load one explicit, tenant-local operator approval from agent configuration.

    Inference is disabled unless the per-agent configuration says otherwise.
    The profile carries only exact versions and a content-addressed ontology;
    accepting a name-only or ``latest`` selector here would let an unrelated
    package update broaden a tenant's materialization rules.
    """
    if not isinstance(config, Mapping):
        raise InferenceError("[semantic_inference] must be a table")
    allowed_fields = {
        "enabled", "rdfs_version", "owl2rl_version", "ontology", "limits",
    }
    unexpected_fields = set(config).difference(allowed_fields)
    if unexpected_fields:
        raise InferenceError(
            "semantic inference configuration has unsupported fields: "
            + ", ".join(sorted(map(str, unexpected_fields)))
        )
    # Validate the paired budget at the same explicit approval boundary.
    # Agent startup later retains the parsed object and passes it to sleep.
    inference_limits_from_config(config)
    enabled = config.get("enabled", False)
    if type(enabled) is not bool:
        raise InferenceError("semantic inference enabled must be a boolean")
    if not enabled:
        return None

    ontology = config.get("ontology")
    if not isinstance(ontology, Mapping):
        raise InferenceError("enabled semantic inference requires an [semantic_inference.ontology] table")
    required_ontology_fields = (
        "namespace",
        "version",
        "content_digest",
        "compatibility_profile",
    )
    unexpected_ontology_fields = set(ontology).difference(required_ontology_fields)
    if unexpected_ontology_fields:
        raise InferenceError(
            "semantic inference ontology has unsupported fields: "
            + ", ".join(sorted(map(str, unexpected_ontology_fields)))
        )
    if any(not isinstance(ontology.get(field), str) or not ontology[field] for field in required_ontology_fields):
        raise InferenceError("semantic inference ontology requires exact namespace, version, content_digest, and compatibility_profile")

    rdfs_version = config.get("rdfs_version")
    owl2rl_version = config.get("owl2rl_version")
    if not isinstance(rdfs_version, str) or not rdfs_version:
        raise InferenceError("enabled semantic inference requires an exact rdfs_version")
    if owl2rl_version is not None and (not isinstance(owl2rl_version, str) or not owl2rl_version):
        raise InferenceError("owl2rl_version must be an exact version or omitted")
    return InferenceProfile(
        OntologyRef(
            str(ontology["namespace"]),
            str(ontology["version"]),
            str(ontology["content_digest"]),
            str(ontology["compatibility_profile"]),
        ),
        rdfs_version,
        owl2rl_version,
    )


def inference_limits_from_config(config: Mapping[str, object]) -> InferenceLimits:
    """Parse one strict, operator-owned materialization budget.

    Limits live beside the exact rule/ontology approval so a deployment cannot
    silently inherit an unreviewed library or process default.  Omitting the
    optional table preserves the bounded service defaults; every supplied
    value, however, must have the exact expected primitive type.
    """
    if not isinstance(config, Mapping):
        raise InferenceError("[semantic_inference] must be a table")
    raw_limits = config.get("limits")
    if raw_limits is None:
        return InferenceLimits()
    if not isinstance(raw_limits, Mapping):
        raise InferenceError("[semantic_inference.limits] must be a table")
    allowed_fields = {
        "max_source_assertions",
        "max_iterations",
        "max_generated_assertions",
        "max_wall_time_seconds",
        "max_memory_items",
    }
    unexpected_fields = set(raw_limits).difference(allowed_fields)
    if unexpected_fields:
        raise InferenceError(
            "semantic inference limits have unsupported fields: "
            + ", ".join(sorted(map(str, unexpected_fields)))
        )
    values: dict[str, int | float] = {}
    for name in (
        "max_source_assertions",
        "max_iterations",
        "max_generated_assertions",
        "max_memory_items",
    ):
        if name in raw_limits:
            value = raw_limits[name]
            if type(value) is not int:
                raise InferenceError(f"semantic inference limit {name} must be an integer")
            values[name] = value
    if "max_wall_time_seconds" in raw_limits:
        value = raw_limits["max_wall_time_seconds"]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise InferenceError(
                "semantic inference limit max_wall_time_seconds must be a number"
            )
        values["max_wall_time_seconds"] = value
    try:
        return InferenceLimits(**values)
    except TypeError as error:  # Defensive: keep this parser's public error stable.
        raise InferenceError("invalid semantic inference limits") from error


def validate_inference_profile(profile: InferenceProfile) -> None:
    """Validate the exact local artifacts selected by an inference profile.

    Parsing a profile only proves that its values have the expected shape.
    Startup must also prove that each pinned capability resolves in this
    installed build and that the local artifact still has the reviewed shape.
    Keeping this at the profile boundary lets startup and materialization share
    one validation contract.
    """
    if not isinstance(profile, InferenceProfile):
        raise InferenceError("semantic inference validation requires an InferenceProfile")
    registry = get_knowledge_registry()
    try:
        ontology_version = SemanticVersion.parse(profile.ontology.version)
    except ValueError as error:
        raise InferenceError("the pinned ontology version must be an exact semantic version") from error
    ontology_matches = [
        resource
        for resource in registry.resources
        if resource.kind is ResourceKind.ONTOLOGY
        and resource.namespace == profile.ontology.namespace
        and resource.version == ontology_version
    ]
    if len(ontology_matches) != 1:
        raise InferenceError("the pinned ontology namespace and version are unavailable")
    ontology = ontology_matches[0]
    if profile.ontology.content_digest != ontology.sha256:
        raise InferenceError("the pinned ontology digest does not match the local registry")
    if profile.ontology.compatibility_profile != registry.contract_version:
        raise InferenceError("the pinned ontology compatibility profile is unsupported")
    try:
        rdfs = registry.resolve_capability("rdfs-v1", profile.rdfs_version)
    except KnowledgeRegistryError as error:
        raise InferenceError("the pinned RDFS rule profile is unavailable") from error
    if rdfs.resource.kind is not ResourceKind.RULE_PROFILE:
        raise InferenceError("rdfs-v1 must resolve to a rule profile")
    rdfs_profile = _read_profile(rdfs.resource.package_resource, "rdfs-v1")
    expected_rdfs = {
        "rdfs:subClassOf", "rdfs:subPropertyOf", "rdfs:domain", "rdfs:range",
    }
    if set(rdfs_profile.get("entailment", ())) != expected_rdfs:
        raise InferenceError("the pinned RDFS profile has an unsupported rule allowlist")
    if profile.owl2rl_version is None:
        return
    try:
        owl = registry.resolve_capability(
            "owl2rl-kestrel-v1", profile.owl2rl_version
        )
    except KnowledgeRegistryError as error:
        raise InferenceError("the pinned OWL rule profile is unavailable") from error
    if owl.resource.kind is not ResourceKind.RULE_PROFILE:
        raise InferenceError("owl2rl-kestrel-v1 must resolve to a rule profile")
    rule_profile = _read_profile(owl.resource.package_resource, "owl2rl-kestrel-v1")
    expected = {
        "rdfs:subClassOf", "rdfs:subPropertyOf", "rdfs:domain", "rdfs:range",
        "owl:equivalentClass", "owl:equivalentProperty", "owl:inverseOf",
        "owl:TransitiveProperty", "owl:SymmetricProperty", "owl:propertyChainAxiom:max-3",
    }
    allowed = rule_profile.get("allows")
    if not isinstance(allowed, list) or set(allowed) != expected:
        raise InferenceError("the pinned OWL profile has an unsupported rule allowlist")


@dataclass(frozen=True, slots=True)
class ClosureState:
    """Durable status for the latest attempt at one profile's closure."""

    profile_key: str
    run_id: str
    source_generation: int
    status: ClosureStatus
    incomplete_reason: str | None
    updated_at: str

    @property
    def complete(self) -> bool:
        return self.status is ClosureStatus.COMPLETE


@dataclass(frozen=True, slots=True)
class MaterializationResult:
    run_id: str
    profile_key: str
    source_generation: int
    checkpoint_generation: int
    status: ClosureStatus
    incomplete_reason: str | None
    source_assertions: int
    generated_assertions: int
    active_derivations: int
    retracted_assertions: int
    incremental: bool

    @property
    def complete(self) -> bool:
        return self.status is ClosureStatus.COMPLETE


@dataclass(frozen=True, slots=True)
class InferenceReconciliationResult:
    """One bounded page retiring active proofs from an obsolete profile.

    Maintenance owns the durable cursor; this result deliberately contains no
    assertion values.  A page can retire proof-ledger entries without
    retracting a conclusion when an independent active proof still grounds it.
    """

    retired_derivations: int
    retracted_assertions: int
    next_cursor: str | None
    backlog: int


@dataclass(frozen=True, slots=True)
class DerivationExplanation:
    derivation_id: str
    rule_id: str
    rule_profile_version: str
    ontology: OntologyRef
    run_id: str
    generated_at: str
    premise_revision_ids: tuple[str, ...]


@dataclass(slots=True)
class _Derivation:
    rule_id: str
    premises: tuple[str, ...]

    @property
    def signature(self) -> str:
        return _sha256({"rule_id": self.rule_id, "premises": self.premises})


@dataclass(slots=True)
class _Fact:
    assertion_id: str
    subject: IRI
    predicate: IRI
    object: AssertionObject
    depth: int
    source_assertion: Assertion | None = None
    derivations: dict[str, _Derivation] = field(default_factory=dict)

    @property
    def is_source(self) -> bool:
        return self.source_assertion is not None


class _BudgetExceeded(RuntimeError):
    def __init__(self, reason: str) -> None:
        self.reason = reason


def _budget_exhaustion_from(error: BaseException) -> _BudgetExceeded | None:
    """Return a materialization budget error preserved as an explicit cause.

    The storage backends deliberately translate exceptions raised in a
    transaction to their public transaction error.  ``__cause__`` retains the
    original bounded-work outcome, which is semantically distinct from a
    persistence failure.  Follow only explicit causes (not incidental context)
    and guard against a malformed cyclic exception chain.
    """
    current: BaseException | None = error
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        if isinstance(current, _BudgetExceeded):
            return current
        visited.add(id(current))
        current = current.__cause__
    return None


class BoundedInferenceService:
    """The sole materializer for the pinned RDFS / OWL 2 RL profile.

    The supplied store is already bound to one authenticated tenant.  The
    service deliberately does not accept a tenant id, raw database, RDF graph,
    or ``latest`` profile selector.
    """

    def __init__(
        self,
        store: "AsyncAssertionStore",
        profile: InferenceProfile,
        *,
        limits: InferenceLimits | None = None,
    ) -> None:
        from kestrel_sovereign.storage.async_assertion_store import AsyncAssertionStore

        if not isinstance(store, AsyncAssertionStore):
            raise InferenceError("BoundedInferenceService requires an agent-bound AsyncAssertionStore")
        if limits is not None and not isinstance(limits, InferenceLimits):
            raise InferenceError("BoundedInferenceService limits must be InferenceLimits")
        self._store = store
        self.profile = profile
        self.limits = limits or InferenceLimits()
        self._load_and_verify_rule_profile()

    def _load_and_verify_rule_profile(self) -> None:
        validate_inference_profile(self.profile)

    async def closure_state(self) -> ClosureState | None:
        tenant_id = self._store.tenant_id
        row = await self._store._database.fetchone(  # noqa: SLF001 - internal peer service
            "SELECT run_id, source_generation, status, incomplete_reason, updated_at "
            "FROM semantic_inference_state WHERE tenant_id = ? AND profile_key = ?",
            (tenant_id, self.profile.key),
        )
        if row is None:
            return None
        return ClosureState(
            self.profile.key,
            str(row[0]),
            int(row[1]),
            ClosureStatus(str(row[2])),
            str(row[3]) if row[3] is not None else None,
            str(row[4]),
        )

    async def explain(self, assertion_id: str) -> tuple[DerivationExplanation, ...]:
        """Return active, tenant-local lineage without exposing assertion text."""
        if not isinstance(assertion_id, str) or not assertion_id:
            raise InferenceError("assertion_id must be non-empty")
        tenant_id = self._store.tenant_id
        rows = await self._store._database.fetchall(  # noqa: SLF001
            "SELECT d.derivation_id, d.rule_id, d.rule_profile_version, "
            "d.ontology_namespace, d.ontology_version, d.ontology_digest, d.run_id, d.generated_at "
            "FROM semantic_inference_derivations d "
            "WHERE d.tenant_id = ? AND d.derived_assertion_id = ? AND d.active = 1 "
            "ORDER BY d.rule_id ASC, d.derivation_id ASC",
            (tenant_id, assertion_id),
        )
        explanations: list[DerivationExplanation] = []
        for row in rows:
            premises = await self._store._database.fetchall(  # noqa: SLF001
                "SELECT input_revision_id FROM semantic_inference_derivation_inputs "
                "WHERE tenant_id = ? AND derivation_id = ? ORDER BY ordinal ASC",
                (tenant_id, row[0]),
            )
            explanations.append(
                DerivationExplanation(
                    str(row[0]), str(row[1]), str(row[2]),
                    OntologyRef(str(row[3]), str(row[4]), str(row[5]), self.profile.ontology.compatibility_profile),
                    str(row[6]), str(row[7]), tuple(str(item[0]) for item in premises),
                )
            )
        return tuple(explanations)

    async def materialize_incremental(self) -> MaterializationResult:
        """Process a changed-assertion batch; never treats a partial closure as complete."""
        return await self._materialize(full_rebuild=False)

    async def rebuild(self) -> MaterializationResult:
        """Explicit repair path that recomputes the profile from current direct assertions."""
        return await self._materialize(full_rebuild=True)

    async def materialize_targets(
        self,
        assertion_ids: Sequence[str],
        *,
        max_context_assertions: int = 0,
    ) -> MaterializationResult:
        """Materialize a bounded, caller-selected direct-fact work unit.

        This is deliberately distinct from :meth:`materialize_incremental`.
        The latter is a complete graph closure and therefore cannot be hidden
        behind a sleep maintenance assertion budget.  A target unit reads only
        the selected current sources, publishes only conclusions proven by
        those sources, and never replaces the profile-wide proof ledger.
        Lifecycle writes already deactivate proofs that name withdrawn input
        revisions, so additive target publication remains safe across pages.
        """
        target_ids = tuple(sorted(set(assertion_ids)))
        if type(max_context_assertions) is not int or max_context_assertions < 0:
            raise InferenceError("max_context_assertions must be a non-negative integer")
        if not target_ids:
            checkpoint = await self._store.checkpoint()
            return MaterializationResult(
                "inference:targets:empty",
                self.profile.key,
                checkpoint.generation,
                checkpoint.generation,
                ClosureStatus.COMPLETE,
                None,
                0,
                0,
                0,
                0,
                True,
            )
        if len(target_ids) > self.limits.max_source_assertions:
            raise InferenceError(
                "targeted materialization exceeds max_source_assertions"
            )

        initial_checkpoint = await self._store.checkpoint()
        run_id = "inference:targets:" + _sha256(
            {
                "profile_key": self.profile.key,
                "generation": initial_checkpoint.generation,
                "assertion_ids": target_ids,
            }
        )[:40]
        started = monotonic()
        try:
            sources = await self._target_source_facts(
                target_ids,
                max_context_assertions=max_context_assertions,
                started=started,
            )
            facts = self._close(sources, started)
        except _BudgetExceeded as error:
            return MaterializationResult(
                run_id,
                self.profile.key,
                initial_checkpoint.generation,
                initial_checkpoint.generation,
                ClosureStatus.INCOMPLETE,
                error.reason,
                0,
                0,
                0,
                0,
                True,
            )

        try:
            async with self._store.inference_publication():
                locked_checkpoint = await self._store.checkpoint()
                if locked_checkpoint.generation != initial_checkpoint.generation:
                    return MaterializationResult(
                        run_id,
                        self.profile.key,
                        initial_checkpoint.generation,
                        locked_checkpoint.generation,
                        ClosureStatus.INCOMPLETE,
                        "source_changed_during_closure",
                        len(sources),
                        0,
                        0,
                        0,
                        True,
                    )
                generated, derivations = await self._persist_facts(
                    facts, run_id, started
                )
                await self._write_active_derivations(
                    facts, run_id, started, replace=False
                )
                final_checkpoint = await self._store.checkpoint()
        except Exception as error:
            budget_error = _budget_exhaustion_from(error)
            if budget_error is not None:
                return MaterializationResult(
                    run_id,
                    self.profile.key,
                    initial_checkpoint.generation,
                    initial_checkpoint.generation,
                    ClosureStatus.INCOMPLETE,
                    budget_error.reason,
                    len(sources),
                    0,
                    0,
                    0,
                    True,
                )
            raise
        return MaterializationResult(
            run_id,
            self.profile.key,
            initial_checkpoint.generation,
            final_checkpoint.generation,
            ClosureStatus.COMPLETE,
            None,
            len(sources),
            generated,
            derivations,
            0,
            True,
        )

    async def _materialize(self, *, full_rebuild: bool) -> MaterializationResult:
        prior = await self.closure_state()
        initial_checkpoint = await self._store.checkpoint()
        if (
            not full_rebuild
            and prior is not None
            and prior.complete
            and prior.source_generation == initial_checkpoint.generation
        ):
            return MaterializationResult(
                prior.run_id, self.profile.key, initial_checkpoint.generation,
                initial_checkpoint.generation, ClosureStatus.COMPLETE, None,
                0, 0, 0, 0, True,
            )

        # Sleep callers advance from their last durable assertion generation.
        # The closure calculation remains set-based (so a batch with a deleted
        # premise is just as correct as an inserted premise), but consuming the
        # tenant-bound change stream makes the incremental checkpoint explicit
        # and avoids treating this normal path as an operator repair rebuild.
        run_id = self._run_id(initial_checkpoint.generation, full_rebuild)
        changes = ()
        await self._record_run(
            run_id, initial_checkpoint.generation, ClosureStatus.RUNNING, None,
            {"incremental": not full_rebuild, "changed_assertions": len(changes)}, complete=False,
        )
        started = monotonic()
        try:
            if not full_rebuild and prior is not None:
                self._check_time(started)
                changes = tuple(
                    await self._store.changes_since(
                        prior.source_generation,
                        limit=min(1000, self.limits.max_source_assertions),
                    )
                )
                self._check_time(started)
            sources = await self._source_facts(started)
            facts = self._close(sources, started)
        except _BudgetExceeded as error:
            await self._record_run(
                run_id, initial_checkpoint.generation, ClosureStatus.INCOMPLETE, error.reason,
                {"incremental": not full_rebuild, "changed_assertions": len(changes)}, complete=True,
            )
            logger.info(
                "semantic_inference_incomplete",
                extra={
                    "inference_event": "semantic_inference_incomplete",
                    "inference_tenant_id": self._store.tenant_id,
                    "inference_profile_key": self.profile.key,
                    "inference_reason": error.reason,
                    "inference_source_generation": initial_checkpoint.generation,
                },
            )
            return MaterializationResult(
                run_id, self.profile.key, initial_checkpoint.generation,
                initial_checkpoint.generation, ClosureStatus.INCOMPLETE, error.reason,
                0, 0, 0, 0, not full_rebuild,
            )
        except Exception as error:
            await self._record_run(
                run_id, initial_checkpoint.generation, ClosureStatus.FAILED, type(error).__name__,
                {"incremental": not full_rebuild, "changed_assertions": len(changes)}, complete=True,
            )
            raise

        # A closure is calculated from a point-in-time source generation, but
        # it is published only under the assertion store's tenant lock.  The
        # check must live *inside* that lock: checking immediately before
        # persistence leaves a window in which a direct write can advance the
        # generation and then be incorrectly covered by a COMPLETE checkpoint.
        try:
            async with self._store.inference_publication():
                locked_checkpoint = await self._store.checkpoint()
                if locked_checkpoint.generation != initial_checkpoint.generation:
                    reason = "source_changed_during_closure"
                    await self._record_run(
                        run_id, initial_checkpoint.generation, ClosureStatus.INCOMPLETE, reason,
                        {"incremental": not full_rebuild, "changed_assertions": len(changes)}, complete=True,
                    )
                    return MaterializationResult(
                        run_id, self.profile.key, initial_checkpoint.generation,
                        locked_checkpoint.generation, ClosureStatus.INCOMPLETE, reason,
                        len(sources), 0, 0, 0, not full_rebuild,
                    )

                retracted = await self._reconcile_stale(facts, run_id, started)
                generated, derivations = await self._persist_facts(
                    facts, run_id, started
                )
                await self._replace_active_derivations(facts, run_id, started)
                final_checkpoint = await self._store.checkpoint()
                await self._record_run(
                    run_id, final_checkpoint.generation, ClosureStatus.COMPLETE, None,
                    {
                        "incremental": not full_rebuild,
                        "changed_assertions": len(changes),
                        "source_assertions": len(sources),
                        "generated_assertions": generated,
                        "active_derivations": derivations,
                        "retracted_assertions": retracted,
                    },
                    complete=True,
                )
        except Exception as error:
            budget_error = _budget_exhaustion_from(error)
            if budget_error is not None:
                # ``inference_publication()`` uses the canonical store's
                # transaction boundary.  SQLite and PostgreSQL wrap an error
                # raised in that scope in ``TransactionError`` after rolling
                # back, so inspect the causal chain here rather than allowing
                # a publication-time budget limit to be recorded as FAILED.
                await self._record_run(
                    run_id,
                    initial_checkpoint.generation,
                    ClosureStatus.INCOMPLETE,
                    budget_error.reason,
                    {
                        "incremental": not full_rebuild,
                        "changed_assertions": len(changes),
                        "phase": "publication",
                    },
                    complete=True,
                )
                return MaterializationResult(
                    run_id,
                    self.profile.key,
                    initial_checkpoint.generation,
                    initial_checkpoint.generation,
                    ClosureStatus.INCOMPLETE,
                    budget_error.reason,
                    len(sources),
                    0,
                    0,
                    0,
                    not full_rebuild,
                )
            # The publication transaction has rolled back before this handler
            # runs. Record its terminal state outside that transaction so a
            # failed reconciliation, persistence, ledger replacement, or
            # completion checkpoint never leaves a durable RUNNING marker.
            await self._record_run(
                run_id,
                initial_checkpoint.generation,
                ClosureStatus.FAILED,
                type(error).__name__,
                {
                    "incremental": not full_rebuild,
                    "changed_assertions": len(changes),
                    "phase": "publication",
                },
                complete=True,
            )
            raise
        logger.info(
            "semantic_inference_complete",
            extra={
                "inference_event": "semantic_inference_complete",
                "inference_tenant_id": self._store.tenant_id,
                "inference_profile_key": self.profile.key,
                "inference_source_count": len(sources),
                "inference_generated_count": generated,
                "inference_derivation_count": derivations,
                "inference_retracted_count": retracted,
                "inference_checkpoint_generation": final_checkpoint.generation,
            },
        )
        return MaterializationResult(
            run_id, self.profile.key, initial_checkpoint.generation,
            final_checkpoint.generation, ClosureStatus.COMPLETE, None,
            len(sources), generated, derivations, retracted, not full_rebuild,
        )

    async def _source_facts(self, started: float) -> dict[str, _Fact]:
        """Stream eligible direct facts without retaining an unbounded store scan."""
        cursor: str | None = None
        sources: dict[str, _Fact] = {}
        while True:
            self._check_time(started)
            # Retained source facts and the current result page coexist while
            # this loop filters it.  Fit the page in the remaining budget,
            # then remove each item as it becomes a retained fact (or is
            # discarded) so the accounting remains true throughout the scan.
            page_size = min(
                1000,
                self.limits.max_source_assertions,
                max(1, self.limits.max_memory_items - len(sources)),
            )
            page = deque(
                await self._store.inference_inputs_page(
                    cursor=cursor, limit=page_size
                )
            )
            self._check_time(started)
            if not page:
                return sources
            received_count = len(page)
            self._check_memory(sources, transient_items=len(page))
            next_cursor = page[-1].revision_id
            while page:
                assertion = page.popleft()
                self._check_time(started)
                if (
                    assertion.epistemic_state is EpistemicState.INFERRED
                    or assertion.ontology_version != self.profile.ontology
                ):
                    self._check_memory(sources, transient_items=len(page))
                    continue
                if len(sources) >= self.limits.max_source_assertions:
                    raise _BudgetExceeded("source_assertions")
                sources[assertion.assertion_id] = _Fact(
                    assertion.assertion_id,
                    assertion.subject,
                    assertion.predicate,
                    assertion.object,
                    0,
                    assertion,
                )
                self._check_memory(sources, transient_items=len(page))
            if received_count < page_size:
                return sources
            cursor = next_cursor

    async def _target_source_facts(
        self,
        assertion_ids: Sequence[str],
        *,
        max_context_assertions: int,
        started: float,
    ) -> dict[str, _Fact]:
        """Read only a target page's eligible direct inputs.

        Targeted maintenance must not turn an assertion-ID filter into a
        source-graph cursor walk.  The store applies tenant, lifecycle, and
        eligibility filters before this bounded page reaches memory.
        """
        self._check_time(started)
        selected = await self._store.inference_inputs(
            AssertionQuery(assertion_ids=tuple(assertion_ids), limit=len(assertion_ids))
        )
        self._check_time(started)
        facts: dict[str, _Fact] = {}

        def add(assertion: Assertion) -> None:
            if (
                assertion.epistemic_state is EpistemicState.INFERRED
                or assertion.ontology_version != self.profile.ontology
                or assertion.assertion_id in facts
            ):
                return
            if len(facts) >= self.limits.max_source_assertions:
                raise _BudgetExceeded("source_assertions")
            facts[assertion.assertion_id] = _Fact(
                assertion.assertion_id,
                assertion.subject,
                assertion.predicate,
                assertion.object,
                0,
                assertion,
            )
            self._check_memory(facts)

        for assertion in selected:
            add(assertion)

        # A one-hop, term-indexed neighbourhood is explicit context rather
        # than a disguised closure scan. It is enough to join a changed fact
        # to a directly adjacent schema/data fact, and a larger dependency is
        # deferred to a later bounded target or explicit repair.
        remaining_context_reads = min(
            max_context_assertions,
            self.limits.max_source_assertions - len(facts),
        )
        for target in tuple(facts.values()):
            for term in (
                {"subject": target.subject},
                {"predicate": target.predicate},
                {"object": target.object},
            ):
                if remaining_context_reads <= 0:
                    # Context is an explicit bounded dependency, not a hint.
                    # Once the caller has spent its allowance, prove that the
                    # next term index has no additional row before declaring
                    # the target closure complete.  The probe is deliberately
                    # not materialized into the fact set.
                    if max_context_assertions and await self._has_context_row(
                        term,
                        cursor=None,
                        excluded_assertion_ids=tuple(facts),
                        started=started,
                    ):
                        raise _BudgetExceeded("context_assertions")
                    continue
                query = AssertionQuery(**term, limit=remaining_context_reads)
                self._check_time(started)
                context_page = await self._store.inference_inputs(query)
                # Count every row read, including a target repeated by one of
                # its term indexes. A distinct-fact cap alone would let a
                # dense neighbourhood consume arbitrary database work.
                remaining_context_reads -= len(context_page)
                for assertion in context_page:
                    add(assertion)
                if len(context_page) == query.limit and await self._has_context_row(
                    term,
                    cursor=context_page[-1].revision_id,
                    excluded_assertion_ids=tuple(facts),
                    started=started,
                ):
                    raise _BudgetExceeded("context_assertions")
        return facts

    async def _has_context_row(
        self,
        term: Mapping[str, AssertionObject],
        *,
        cursor: str | None,
        excluded_assertion_ids: Sequence[str],
        started: float,
    ) -> bool:
        """Probe a term index without admitting another context fact.

        The probe is the bounded overflow test paired with every context page.
        Without it, a page that fills ``max_context_assertions`` is
        indistinguishable from a complete neighbourhood and maintenance can
        checkpoint a truncated inference closure.
        """
        self._check_time(started)
        return bool(
            await self._store.inference_inputs(
                AssertionQuery(
                    **term,
                    limit=1,
                    cursor=cursor,
                    exclude_assertion_ids=tuple(excluded_assertion_ids),
                )
            )
        )

    @staticmethod
    async def reconcile_obsolete_derivations_page(
        store: "AsyncAssertionStore",
        *,
        active_profile_key: str | None,
        cursor: str | None,
        max_derivations: int,
        run_id: str,
    ) -> InferenceReconciliationResult:
        """Retire one deterministic page of active proofs from prior profiles.

        A new ontology/rule profile must not merely add its own proof rows:
        conclusions justified only by an earlier profile remain active until
        those rows are retired.  This operation pages the proof ledger by
        derivation ID, deactivates obsolete rows, and retracts only conclusions
        left with no independent active proof.  It is safe to retry because
        inactive rows are excluded from each subsequent page.
        """
        if type(max_derivations) is not int or not 1 <= max_derivations <= 1_000:
            raise InferenceError("max_derivations must be an integer in [1, 1000]")
        if cursor is not None and (not isinstance(cursor, str) or not cursor):
            raise InferenceError("derivation reconciliation cursor must be non-empty or null")
        if active_profile_key is not None and (
            not isinstance(active_profile_key, str) or not active_profile_key
        ):
            raise InferenceError("active_profile_key must be a non-empty string or null")
        if not isinstance(run_id, str) or not run_id:
            raise InferenceError("derivation reconciliation run_id must be non-empty")

        database = store._database  # noqa: SLF001 - canonical inference ledger peer
        clauses = ["tenant_id = ?", "active = 1"]
        params: list[object] = [store.tenant_id]
        if active_profile_key is not None:
            clauses.append("profile_key <> ?")
            params.append(active_profile_key)
        if cursor is not None:
            clauses.append("derivation_id > ?")
            params.append(cursor)
        rows = await database.fetchall(
            "SELECT derivation_id, derived_assertion_id, derived_revision_id "
            "FROM semantic_inference_derivations WHERE "
            + " AND ".join(clauses)
            + " ORDER BY derivation_id ASC LIMIT ?",
            tuple(params + [max_derivations + 1]),
        )
        page = rows[:max_derivations]
        backlog = max(0, len(rows) - len(page))
        if not page:
            return InferenceReconciliationResult(0, 0, None, 0)

        retracted = 0
        # The maintenance fence, when present, is renewed by every nested
        # canonical mutation.  The peer publication transaction provides the
        # same tenant serialization when this helper is used by another repair
        # entry point.
        async with store.inference_publication():
            for derivation_id, _, _ in page:
                await database.execute(
                    "UPDATE semantic_inference_derivations SET active = 0 "
                    "WHERE tenant_id = ? AND derivation_id = ? AND active = 1",
                    (store.tenant_id, str(derivation_id)),
                )
            # Proof membership is lifecycle-relevant even when every affected
            # conclusion has another live proof, so publish a fresh generation
            # before checking conclusions for a necessary retraction.
            await store._advance_generation()  # noqa: SLF001 - inference publication peer
            for _, assertion_id, revision_id in page:
                remaining = await database.fetchval(
                    "SELECT COUNT(*) FROM semantic_inference_derivations "
                    "WHERE tenant_id = ? AND derived_assertion_id = ? AND "
                    "derived_revision_id = ? AND active = 1",
                    (store.tenant_id, str(assertion_id), str(revision_id)),
                )
                if int(remaining or 0):
                    continue
                current = await store.get_assertion(str(assertion_id))
                if (
                    current is None
                    or current.status is not AssertionStatus.ACTIVE
                    or current.epistemic_state is not EpistemicState.INFERRED
                    or current.revision_id != str(revision_id)
                ):
                    continue
                receipt = await store.retract(
                    current.assertion_id,
                    current.revision_id,
                    operation_id=(
                        "inference-profile-reconcile:"
                        f"{run_id}:{current.revision_id}"
                    ),
                )
                retracted += len(receipt.retracted)
        return InferenceReconciliationResult(
            retired_derivations=len(page),
            retracted_assertions=retracted,
            next_cursor=(str(page[-1][0]) if backlog else None),
            backlog=backlog,
        )

    def _close(self, facts: dict[str, _Fact], started: float) -> dict[str, _Fact]:
        self._check_memory(facts)
        for iteration in range(1, self.limits.max_iterations + 1):
            self._check_time(started)
            snapshot = tuple(facts[key] for key in sorted(facts))
            added = self._apply_rules(snapshot, facts, iteration, started)
            self._check_memory(facts)
            if not added:
                return facts
        raise _BudgetExceeded("iterations")

    def _apply_rules(
        self,
        snapshot: Sequence[_Fact],
        facts: dict[str, _Fact],
        depth: int,
        started: float,
    ) -> bool:
        added = False

        def add(rule_id: str, subject: IRI, predicate: IRI, object_: AssertionObject, premises: Sequence[_Fact]) -> None:
            nonlocal added
            self._check_time(started)
            premise_ids = tuple(item.assertion_id for item in premises)
            if len(set(premise_ids)) != len(premise_ids):
                return
            assertion_id = derive_assertion_id(
                tenant_id=self._store.tenant_id, subject=subject, predicate=predicate, object=object_,
            )
            if assertion_id in premise_ids:
                return
            candidate = _Derivation(rule_id, premise_ids)
            existing = facts.get(assertion_id)
            if existing is None:
                if len(facts) - self._source_count(facts) >= self.limits.max_generated_assertions:
                    raise _BudgetExceeded("generated_assertions")
                facts[assertion_id] = _Fact(assertion_id, subject, predicate, object_, depth, None, {candidate.signature: candidate})
                added = True
            elif not existing.is_source and candidate.signature not in existing.derivations:
                existing.derivations[candidate.signature] = candidate
            self._check_memory(facts)

        subclass = [fact for fact in snapshot if fact.predicate.value == RDFS_SUBCLASS and isinstance(fact.object, IRI)]
        subproperty = [fact for fact in snapshot if fact.predicate.value == RDFS_SUBPROPERTY and isinstance(fact.object, IRI)]
        domains = [fact for fact in snapshot if fact.predicate.value == RDFS_DOMAIN and isinstance(fact.object, IRI)]
        ranges = [fact for fact in snapshot if fact.predicate.value == RDFS_RANGE and isinstance(fact.object, IRI)]

        for left in subclass:
            for right in subclass:
                self._check_time(started)
                if left.object == right.subject:
                    add("rdfs:subClassOf-transitive", left.subject, IRI(RDFS_SUBCLASS), right.object, (left, right))
        for instance in snapshot:
            self._check_time(started)
            if instance.predicate.value != RDF_TYPE or not isinstance(instance.object, IRI):
                continue
            for hierarchy in subclass:
                self._check_time(started)
                if instance.object == hierarchy.subject:
                    add("rdfs:type-subClassOf", instance.subject, IRI(RDF_TYPE), hierarchy.object, (instance, hierarchy))
        for left in subproperty:
            for right in subproperty:
                self._check_time(started)
                if left.object == right.subject:
                    add("rdfs:subPropertyOf-transitive", left.subject, IRI(RDFS_SUBPROPERTY), right.object, (left, right))
        for statement in snapshot:
            self._check_time(started)
            for hierarchy in subproperty:
                self._check_time(started)
                if statement.predicate == hierarchy.subject:
                    add("rdfs:subPropertyOf", statement.subject, hierarchy.object, statement.object, (statement, hierarchy))
            for domain in domains:
                self._check_time(started)
                if statement.predicate == domain.subject:
                    add("rdfs:domain", statement.subject, IRI(RDF_TYPE), domain.object, (statement, domain))
            if isinstance(statement.object, IRI):
                for range_ in ranges:
                    self._check_time(started)
                    if statement.predicate == range_.subject:
                        add("rdfs:range", statement.object, IRI(RDF_TYPE), range_.object, (statement, range_))

        if self.profile.owl2rl_version is None:
            return added

        equivalents_class = [fact for fact in snapshot if fact.predicate.value == OWL_EQUIVALENT_CLASS and isinstance(fact.object, IRI)]
        equivalents_property = [fact for fact in snapshot if fact.predicate.value == OWL_EQUIVALENT_PROPERTY and isinstance(fact.object, IRI)]
        inverses = [fact for fact in snapshot if fact.predicate.value == OWL_INVERSE_OF and isinstance(fact.object, IRI)]
        transitive = {
            fact.subject for fact in snapshot
            if fact.predicate.value == RDF_TYPE and isinstance(fact.object, IRI) and fact.object.value == OWL_TRANSITIVE_PROPERTY
        }
        symmetric = {
            fact.subject for fact in snapshot
            if fact.predicate.value == RDF_TYPE and isinstance(fact.object, IRI) and fact.object.value == OWL_SYMMETRIC_PROPERTY
        }
        for equivalent in equivalents_class:
            self._check_time(started)
            add("owl:equivalentClass-left", equivalent.subject, IRI(RDFS_SUBCLASS), equivalent.object, (equivalent,))
            add("owl:equivalentClass-right", equivalent.object, IRI(RDFS_SUBCLASS), equivalent.subject, (equivalent,))
        for equivalent in equivalents_property:
            self._check_time(started)
            add("owl:equivalentProperty-left", equivalent.subject, IRI(RDFS_SUBPROPERTY), equivalent.object, (equivalent,))
            add("owl:equivalentProperty-right", equivalent.object, IRI(RDFS_SUBPROPERTY), equivalent.subject, (equivalent,))
        for inverse in inverses:
            for statement in snapshot:
                self._check_time(started)
                if statement.predicate == inverse.subject and isinstance(statement.object, IRI):
                    add("owl:inverseOf-forward", statement.object, inverse.object, statement.subject, (inverse, statement))
                if statement.predicate == inverse.object and isinstance(statement.object, IRI):
                    add("owl:inverseOf-reverse", statement.object, inverse.subject, statement.subject, (inverse, statement))
        for statement in snapshot:
            self._check_time(started)
            if statement.predicate in symmetric and isinstance(statement.object, IRI):
                add("owl:SymmetricProperty", statement.object, statement.predicate, statement.subject, (statement,))
        for first in snapshot:
            self._check_time(started)
            if first.predicate not in transitive or not isinstance(first.object, IRI):
                continue
            for second in snapshot:
                self._check_time(started)
                if second.predicate == first.predicate and second.subject == first.object:
                    add("owl:TransitiveProperty", first.subject, first.predicate, second.object, (first, second))
        for chain_axiom in snapshot:
            self._check_time(started)
            if chain_axiom.predicate.value != OWL_PROPERTY_CHAIN_AXIOM:
                continue
            chain = _property_chain(chain_axiom.object)
            if chain is None:
                continue
            for path in _matching_paths(snapshot, chain, lambda: self._check_time(started)):
                self._check_time(started)
                add(f"owl:propertyChainAxiom:{len(chain)}", path[0].subject, chain_axiom.subject, path[-1].object, (chain_axiom, *path))
        return added

    @staticmethod
    def _source_count(facts: Iterable[_Fact] | dict[str, _Fact]) -> int:
        values = facts.values() if isinstance(facts, dict) else facts
        return sum(1 for fact in values if fact.is_source)

    def _check_time(self, started: float) -> None:
        if monotonic() - started > self.limits.max_wall_time_seconds:
            raise _BudgetExceeded("wall_time")

    def _check_memory(
        self,
        facts: dict[str, _Fact],
        *,
        transient_items: int = 0,
    ) -> None:
        item_count = (
            len(facts)
            + sum(len(fact.derivations) for fact in facts.values())
            + transient_items
        )
        if item_count > self.limits.max_memory_items:
            raise _BudgetExceeded("memory")

    async def _reconcile_stale(
        self,
        facts: dict[str, _Fact],
        run_id: str,
        started: float,
    ) -> int:
        desired = set(facts)
        retracted = 0
        cursor: str | None = None
        while True:
            self._check_time(started)
            page_size = min(1000, self.limits.max_memory_items)
            page = await self._store.inference_inputs_page(
                cursor=cursor, limit=page_size
            )
            self._check_time(started)
            if not page:
                return retracted
            self._check_memory(facts, transient_items=len(page))
            next_cursor = page[-1].revision_id
            # Mutating the result page in place avoids retaining a second
            # page-sized stale list under the same inference memory limit.
            page.sort(key=lambda item: item.assertion_id)
            for assertion in page:
                if not (
                    assertion.epistemic_state is EpistemicState.INFERRED
                    and isinstance(assertion.lineage, DerivedLineage)
                    and assertion.lineage.engine_version == ENGINE_VERSION
                    and (
                        assertion.assertion_id not in desired
                        or assertion.ontology_version != self.profile.ontology
                        or assertion.lineage.profile_version
                        != self.profile.rule_profile_version
                    )
                ):
                    continue
                self._check_time(started)
                # A previous stale root can transitively retract this assertion.
                # Re-read the current pointer rather than turning that normal
                # lifecycle cascade into an optimistic-concurrency failure.
                current = await self._store.get_assertion(
                    assertion.assertion_id, include_inactive=True
                )
                if (
                    current is None
                    or current.status is not AssertionStatus.ACTIVE
                    or current.revision_id != assertion.revision_id
                ):
                    continue
                await self._store.retract(
                    assertion.assertion_id,
                    assertion.revision_id,
                    operation_id=f"inference-retract:{run_id}:{assertion.revision_id}",
                )
                retracted += 1
            if len(page) < page_size:
                return retracted
            cursor = next_cursor
        return retracted

    async def _persist_facts(
        self,
        facts: dict[str, _Fact],
        run_id: str,
        started: float,
    ) -> tuple[int, int]:
        generated = [fact for fact in facts.values() if not fact.is_source]
        revision_ids = {
            fact.assertion_id: fact.source_assertion.revision_id
            for fact in facts.values() if fact.source_assertion is not None
        }
        persisted: list[_Fact] = []
        for fact in sorted(generated, key=lambda item: (item.depth, item.assertion_id)):
            self._check_time(started)
            # A conclusion can have an alternate proof discovered later than
            # its first proof.  The lexicographically first signature is not
            # necessarily topologically publishable (it may name an inferred
            # premise from a later proof path), so choose deterministically
            # among the premise sets whose revisions are already available.
            available_derivations = [
                derivation
                for derivation in fact.derivations.values()
                if all(premise in revision_ids for premise in derivation.premises)
            ]
            if not available_derivations:
                raise InferenceError("closure fact has no publishable derivation")
            primary = min(available_derivations, key=lambda item: item.signature)
            inputs = tuple(revision_ids[premise] for premise in primary.premises)
            assertion = self._derived_assertion(fact, run_id, primary, inputs)
            current = await self._store.get_assertion(fact.assertion_id, include_inactive=True)
            if current is None:
                result = await self._store.publish_inferred_assertion(
                    assertion,
                    operation_id=f"inference-put:{run_id}:{fact.assertion_id}",
                )
            elif current.status is AssertionStatus.ACTIVE:
                result = None
            elif current.epistemic_state is EpistemicState.RETRACTED and isinstance(current.lineage, DerivedLineage):
                result = await self._store.reactivate_inferred(assertion, operation_id=f"inference-reactivate:{run_id}:{fact.assertion_id}")
            else:
                # A direct assertion with this exact canonical identity is
                # stronger source knowledge.  Leave it current and do not
                # write an inference ledger row for an assertion we do not own.
                continue
            current = result.assertion if result is not None else await self._store.get_assertion(fact.assertion_id)
            if current is None or current.status is not AssertionStatus.ACTIVE:
                raise InferenceError("materialized assertion did not remain active")
            revision_ids[fact.assertion_id] = current.revision_id
            persisted.append(fact)
        # The ledger contains every independent derivation, not just each
        # canonical revision's deterministic primary lineage.
        return len(persisted), sum(len(fact.derivations) for fact in persisted)

    def _derived_assertion(
        self,
        fact: _Fact,
        run_id: str,
        derivation: _Derivation,
        input_revision_ids: tuple[str, ...],
    ) -> Assertion:
        generated_at = _utc_now()
        revision_id = "inference:" + _sha256({"run_id": run_id, "assertion_id": fact.assertion_id})[:40]
        return Assertion(
            tenant_id=self._store.tenant_id,
            owning_agent_id=self._store.owning_agent_id,
            subject=fact.subject,
            predicate=fact.predicate,
            object=fact.object,
            revision_id=revision_id,
            confidence="1",
            confidence_method=ENGINE_VERSION,
            confidence_basis=self.profile.key,
            epistemic_state=EpistemicState.INFERRED,
            asserted_at=generated_at,
            ontology_version=self.profile.ontology,
            lineage=DerivedLineage(
                rule_id=derivation.rule_id,
                engine_version=ENGINE_VERSION,
                profile_version=self.profile.rule_profile_version,
                input_revision_ids=input_revision_ids,
                input_digest=_sha256({"premises": input_revision_ids}),
                run_id=run_id,
                generated_at=generated_at,
                derivation_reference="urn:kestrel:inference:" + derivation.signature,
            ),
            privacy_classification="normal",
            release_policy_reference="policy:inference-private-v1",
        )

    async def _replace_active_derivations(
        self,
        facts: dict[str, _Fact],
        run_id: str,
        started: float,
    ) -> None:
        await self._write_active_derivations(facts, run_id, started, replace=True)

    async def _write_active_derivations(
        self,
        facts: dict[str, _Fact],
        run_id: str,
        started: float,
        *,
        replace: bool,
    ) -> None:
        tenant_id = self._store.tenant_id
        database = self._store._database  # noqa: SLF001 - same internal persistence boundary
        # A materialization can change only the active proof set for an
        # existing conclusion.  That still changes the lifecycle graph used
        # by governed supersession planning, even though no canonical
        # assertion revision is written.  Capture the semantic membership
        # before replacement so that a real proof/input change advances the
        # tenant generation in this same publication transaction.
        before_membership = await self._active_derivation_membership()
        if replace:
            await database.execute(
                "UPDATE semantic_inference_derivations SET active = 0 "
                "WHERE tenant_id = ? AND profile_key = ?",
                (tenant_id, self.profile.key),
            )
        for fact in sorted((item for item in facts.values() if not item.is_source), key=lambda item: item.assertion_id):
            self._check_time(started)
            current = await self._store.get_assertion(fact.assertion_id)
            if current is None or current.epistemic_state is not EpistemicState.INFERRED:
                continue
            for derivation in sorted(fact.derivations.values(), key=lambda item: item.signature):
                self._check_time(started)
                derivation_id = "derivation:" + _sha256({
                    "profile_key": self.profile.key,
                    "assertion_id": fact.assertion_id,
                    "rule_id": derivation.rule_id,
                    "premises": derivation.premises,
                })
                premises = []
                for premise in derivation.premises:
                    self._check_time(started)
                    premise_assertion = await self._store.get_assertion(premise)
                    if premise_assertion is None:
                        raise InferenceError("active closure premise disappeared during ledger write")
                    premises.append(premise_assertion.revision_id)
                await database.execute(
                    "INSERT INTO semantic_inference_derivations "
                    "(tenant_id, derivation_id, derived_assertion_id, derived_revision_id, rule_id, profile_key, "
                    "rule_profile_version, ontology_namespace, ontology_version, ontology_digest, run_id, generated_at, active) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1) "
                    "ON CONFLICT(tenant_id, derivation_id) DO UPDATE SET "
                    "derived_revision_id = excluded.derived_revision_id, run_id = excluded.run_id, "
                    "generated_at = excluded.generated_at, active = 1",
                    (
                        tenant_id, derivation_id, fact.assertion_id, current.revision_id,
                        derivation.rule_id, self.profile.key, self.profile.rule_profile_version,
                        self.profile.ontology.namespace, self.profile.ontology.version,
                        self.profile.ontology.content_digest, run_id, _utc_now(),
                    ),
                )
                await database.execute(
                    "DELETE FROM semantic_inference_derivation_inputs WHERE tenant_id = ? AND derivation_id = ?",
                    (tenant_id, derivation_id),
                )
                for ordinal, revision_id in enumerate(premises):
                    await database.execute(
                        "INSERT INTO semantic_inference_derivation_inputs "
                        "(tenant_id, derivation_id, input_revision_id, ordinal) VALUES (?, ?, ?, ?)",
                        (tenant_id, derivation_id, revision_id, ordinal),
                    )
        if before_membership != await self._active_derivation_membership():
            # ``inference_publication()`` owns the tenant lock, so this is
            # the same atomic CAS fence observed by governed validation.
            # Lifecycle mutations already advance this generation whenever
            # they deactivate proofs; this covers the materializer-only path
            # that adds/removes an alternate active proof without changing a
            # canonical assertion revision.
            await self._store._advance_generation()  # noqa: SLF001

    async def _active_derivation_membership(
        self,
    ) -> tuple[tuple[str, str, str, tuple[str, ...]], ...]:
        """Return this profile's active proof graph without run metadata.

        A new run ID or generated timestamp does not change lifecycle
        planning.  The exact active derivation identity, conclusion revision,
        and ordered premise revisions do.  Keeping this comparison here makes
        generation advancement depend on the semantic ledger state rather
        than the implementation's deactivate-then-upsert mechanics.
        """
        rows = await self._store._database.fetchall(  # noqa: SLF001
            "SELECT d.derivation_id, d.derived_assertion_id, d.derived_revision_id, "
            "i.input_revision_id FROM semantic_inference_derivations d "
            "LEFT JOIN semantic_inference_derivation_inputs i "
            "ON i.tenant_id = d.tenant_id AND i.derivation_id = d.derivation_id "
            "WHERE d.tenant_id = ? AND d.profile_key = ? AND d.active = 1 "
            "ORDER BY d.derivation_id ASC, i.ordinal ASC",
            (self._store.tenant_id, self.profile.key),
        )
        membership: list[tuple[str, str, str, tuple[str, ...]]] = []
        derivation_id: str | None = None
        derived_assertion_id = ""
        derived_revision_id = ""
        premises: list[str] = []
        for row in rows:
            row_derivation_id = str(row[0])
            if derivation_id is not None and row_derivation_id != derivation_id:
                membership.append(
                    (derivation_id, derived_assertion_id, derived_revision_id, tuple(premises))
                )
                premises = []
            if row_derivation_id != derivation_id:
                derivation_id = row_derivation_id
                derived_assertion_id = str(row[1])
                derived_revision_id = str(row[2])
            if row[3] is not None:
                premises.append(str(row[3]))
        if derivation_id is not None:
            membership.append(
                (derivation_id, derived_assertion_id, derived_revision_id, tuple(premises))
            )
        return tuple(membership)

    async def _record_run(
        self,
        run_id: str,
        source_generation: int,
        status: ClosureStatus,
        incomplete_reason: str | None,
        result: dict[str, object],
        *,
        complete: bool,
    ) -> None:
        tenant_id = self._store.tenant_id
        now = _utc_now()
        result_mapping = _canonical_json(result)
        database = self._store._database  # noqa: SLF001
        await database.execute(
            "INSERT INTO semantic_inference_runs "
            "(tenant_id, run_id, profile_key, ontology_namespace, ontology_version, ontology_digest, "
            "source_generation, status, incomplete_reason, result_mapping, started_at, completed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(tenant_id, run_id) DO UPDATE SET source_generation = excluded.source_generation, "
            "status = excluded.status, incomplete_reason = excluded.incomplete_reason, "
            "result_mapping = excluded.result_mapping, completed_at = excluded.completed_at",
            (
                tenant_id, run_id, self.profile.key, self.profile.ontology.namespace,
                self.profile.ontology.version, self.profile.ontology.content_digest,
                source_generation, status.value, incomplete_reason, result_mapping, now,
                now if complete else None,
            ),
        )
        await database.execute(
            "INSERT INTO semantic_inference_state "
            "(tenant_id, profile_key, run_id, source_generation, status, incomplete_reason, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(tenant_id, profile_key) DO UPDATE SET run_id = excluded.run_id, "
            "source_generation = excluded.source_generation, status = excluded.status, "
            "incomplete_reason = excluded.incomplete_reason, updated_at = excluded.updated_at",
            (tenant_id, self.profile.key, run_id, source_generation, status.value, incomplete_reason, now),
        )

    def _run_id(self, generation: int, full_rebuild: bool) -> str:
        # Retrying an incomplete incremental checkpoint repeats the identical
        # run ID and generated revision IDs.  An explicit repair is purposely
        # a fresh auditable action even when it starts from the same snapshot.
        if not full_rebuild:
            return "inference:incremental:" + _sha256(
                {"profile_key": self.profile.key, "generation": generation}
            )[:40]
        return f"inference:rebuild:{generation}:{uuid4().hex}"


InferenceService = BoundedInferenceService
SemanticInferenceService = BoundedInferenceService


def _matching_paths(
    snapshot: Sequence[_Fact],
    chain: tuple[IRI, ...],
    check_budget: Callable[[], None],
) -> Iterable[tuple[_Fact, ...]]:
    paths: list[tuple[_Fact, ...]] = [(fact,) for fact in snapshot if fact.predicate == chain[0]]
    for predicate in chain[1:]:
        next_paths: list[tuple[_Fact, ...]] = []
        for path in paths:
            check_budget()
            tail = path[-1]
            if not isinstance(tail.object, IRI):
                continue
            for candidate in snapshot:
                check_budget()
                if candidate.predicate == predicate and candidate.subject == tail.object:
                    next_paths.append((*path, candidate))
        paths = next_paths
    return tuple(paths)


def _property_chain(value: AssertionObject) -> tuple[IRI, ...] | None:
    """Decode the canonical-store representation of a 2- or 3-property chain.

    Canonical assertions cannot retain RDF list blank nodes.  The reviewed
    profile therefore admits only a JSON string array in the assertion's
    object position; the parser accepts no Turtle, SPARQL, remote reference,
    or arbitrary executable rule encoding.
    """
    if not isinstance(value, Literal):
        return None
    try:
        decoded = json.loads(value.lexical_form)
    except json.JSONDecodeError:
        return None
    if not isinstance(decoded, list) or not 2 <= len(decoded) <= 3 or any(not isinstance(item, str) for item in decoded):
        return None
    try:
        return tuple(IRI(item) for item in decoded)
    except ValueError:
        return None


def _read_profile(resource_name: str, expected_name: str) -> dict[str, object]:
    target = resources.files("kestrel_sovereign")
    for part in resource_name.split("/"):
        target = target.joinpath(part)
    try:
        decoded = json.loads(target.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise InferenceError(f"cannot read pinned inference profile {resource_name!r}") from error
    if not isinstance(decoded, dict) or decoded.get("profile") != expected_name:
        raise InferenceError(f"pinned inference profile {resource_name!r} has an unexpected shape")
    if decoded.get("network_access") is not False or decoded.get("forward_only") is not True:
        raise InferenceError(f"pinned inference profile {resource_name!r} is not a local forward-only profile")
    return decoded


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


__all__ = [
    "BoundedInferenceService",
    "ClosureState",
    "ClosureStatus",
    "DerivationExplanation",
    "ENGINE_VERSION",
    "InferenceError",
    "InferenceLimits",
    "InferenceProfile",
    "InferenceReconciliationResult",
    "inference_limits_from_config",
    "inference_profile_from_config",
    "validate_inference_profile",
    "InferenceService",
    "MaterializationResult",
    "SemanticInferenceService",
]
