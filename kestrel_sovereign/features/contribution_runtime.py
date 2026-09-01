"""Atomic lifecycle wiring for SDK feature contributions.

The SDK owns declaration and validation contracts.  Sovereign owns the live
registries and retains the exact canonical objects returned for one feature
enable (or host-feature start) transition until its matching teardown.
"""

from __future__ import annotations

import asyncio
import weakref
from dataclasses import dataclass, field, replace
from typing import Callable, Iterable

from kestrel_sdk.features import (
    ContributionContractError,
    ContextClauseRegistration,
    FeatureContributionSet,
    FeaturePermissionDefaults,
    SetupStepClassification,
    SetupStepRegistration,
    await_contribution_result,
    order_setup_step_registrations,
    validate_contribution_owner_uniqueness,
    validate_feature_contributions,
)
from kestrel_sdk.operator import ServiceScope

from kestrel_sovereign.agent.system_prompt_assembler import (
    AGENTS_FILENAME,
    SYNTHETIC_HOST_AUDIT_NAMES,
    TORTOISE_DOCTRINE_FILENAME,
)
from kestrel_sovereign.features.bootstrap.loader import DEFAULT_BOOTSTRAP_FILES
from kestrel_sovereign.operator import (
    ExecutionTargetRegistration,
    OperatorRegistrationSet,
    OperatorRuntimeRegistry,
)
from kestrel_sovereign.signals import (
    CLAIM_CONTRIBUTION,
    RegistrationPolicy,
    RegistrationState,
    SourceRegistry,
)
from kestrel_sovereign.waits import WaitRegistry


class FeatureContributionRuntimeError(RuntimeError):
    """A contribution transition cannot be committed or exactly reversed."""


class FeatureContributionCollectionError(FeatureContributionRuntimeError):
    """A sanitized failure from one declarative collection boundary.

    The exact feature and fixed boundary remain inspectable, while the public
    message omits both the feature representation and original exception text.
    The original failure is deliberately discarded rather than retained as an
    exception cause that production traceback logging could reveal.
    """

    _STAGES_BY_GETTER = {
        "contribution_owner": "contribution collection",
        "get_tools": "tool collection",
        "get_service_registrations": "service collection",
        "get_wait_provider_registrations": "wait-provider collection",
        "get_workflow_registrations": "workflow collection",
        "get_feature_permission_defaults": "permission-default collection",
        "get_setup_step_registrations": "setup-step collection",
        "get_context_clause_registrations": "context-clause collection",
        "render_context_clauses": "context-clause rendering",
        "validate_feature_contributions": "contribution validation",
        "validate_contribution_owner_uniqueness": "contribution validation",
    }
    _UNKNOWN_GETTER = "unknown contribution boundary"

    def __init__(self, feature: object, getter: str) -> None:
        self.feature = feature
        self.getter = (
            getter if getter in self._STAGES_BY_GETTER else self._UNKNOWN_GETTER
        )
        self.stage = self._STAGES_BY_GETTER.get(
            getter, "contribution collection"
        )
        super().__init__(
            f"feature contribution failure during {self.stage} "
            f"({self.getter})"
        )


@dataclass(frozen=True, slots=True)
class PermissionDefaultRegistration:
    """Exact feature-name projection of one SDK permission descriptor."""

    owner: str
    feature_name: str
    defaults: FeaturePermissionDefaults


@dataclass(frozen=True, slots=True)
class ResolvedContextClause:
    """Immutable prompt bytes resolved at one feature lifecycle transition."""

    owner: str
    name: str
    priority: int
    body: str
    registration: ContextClauseRegistration

    @property
    def identity(self) -> tuple[str, str]:
        return (self.owner, self.name)


class ContextClauseRegistry:
    """Lifecycle-owned cache of already-rendered feature context clauses."""

    def __init__(self) -> None:
        self._clauses: dict[tuple[str, str], ResolvedContextClause] = {}
        self._external_registries: tuple[ContextClauseRegistry, ...] = ()
        # An agent registry depends on the host registry for prompt assembly.
        # The reverse weak edge lets a later host-feature start preflight
        # against already-bound agents without retaining stopped agents or
        # making independent agents conflict with one another.
        self._dependent_registries: weakref.WeakSet[ContextClauseRegistry] = (
            weakref.WeakSet()
        )
        self._reserved_audit_name_provider: (
            Callable[[], Iterable[str]] | None
        ) = None

    _SYNTHETIC_HOST_AUDIT_NAMES = SYNTHETIC_HOST_AUDIT_NAMES
    _RESERVED_AUDIT_NAMES = frozenset(
        set(DEFAULT_BOOTSTRAP_FILES)
        | {AGENTS_FILENAME, TORTOISE_DOCTRINE_FILENAME}
        | _SYNTHETIC_HOST_AUDIT_NAMES
    )

    def _local_reserved_audit_names(self) -> frozenset[str]:
        names = set(self._RESERVED_AUDIT_NAMES)
        if self._reserved_audit_name_provider is not None:
            names.update(self._reserved_audit_name_provider())
        return frozenset(names)

    def _visible_reserved_audit_names(self) -> frozenset[str]:
        names = set(self._local_reserved_audit_names())
        for registry in self._external_registries:
            names.update(registry._local_reserved_audit_names())
        for registry in self._dependent_registries:
            names.update(registry._local_reserved_audit_names())
        return frozenset(names)

    def validate_declared_names(self, names: Iterable[str]) -> tuple[str, ...]:
        """Validate host-visible audit names without invoking renderers."""

        values = tuple(names)
        if len(set(values)) != len(values):
            raise FeatureContributionRuntimeError(
                "duplicate context-clause name in one registration batch"
            )
        reserved = next(
            (
                name
                for name in values
                if name in self._visible_reserved_audit_names()
                or name.casefold().endswith(".md")
            ),
            None,
        )
        if reserved is not None:
            raise FeatureContributionRuntimeError(
                f"context-clause name {reserved!r} is a reserved host audit name"
            )
        return values

    def _validate_names(
        self,
        values: tuple[ResolvedContextClause, ...],
        *,
        resident: Iterable[ResolvedContextClause],
    ) -> None:
        names = self.validate_declared_names(clause.name for clause in values)
        resident_names = {clause.name for clause in resident}
        conflict = next((name for name in names if name in resident_names), None)
        if conflict is not None:
            raise FeatureContributionRuntimeError(
                f"context-clause name is already registered: {conflict!r}"
            )

    def _external_clauses(self) -> tuple[ResolvedContextClause, ...]:
        return tuple(
            clause
            for registry in self._external_registries
            for clause in registry.snapshot()
        )

    def _dependent_clauses(self) -> tuple[ResolvedContextClause, ...]:
        return tuple(
            clause
            for registry in self._dependent_registries
            for clause in registry.snapshot()
        )

    def validate_reserved_audit_names(
        self, names: Iterable[str]
    ) -> tuple[str, ...]:
        """Ensure prospective bootstrap audit names do not shadow clauses."""

        values = tuple(names)
        if len(set(values)) != len(values):
            raise FeatureContributionRuntimeError(
                "duplicate bootstrap audit name"
            )
        synthetic_conflict = next(
            (
                name
                for name in values
                if name in self._SYNTHETIC_HOST_AUDIT_NAMES
            ),
            None,
        )
        if synthetic_conflict is not None:
            raise FeatureContributionRuntimeError(
                f"bootstrap name {synthetic_conflict!r} is a reserved host "
                "audit name"
            )
        resident_names = {
            clause.name
            for clause in (
                *self._clauses.values(),
                *self._external_clauses(),
                *self._dependent_clauses(),
            )
        }
        conflict = next((name for name in values if name in resident_names), None)
        if conflict is not None:
            raise FeatureContributionRuntimeError(
                f"context-clause name is already registered: {conflict!r}"
            )
        return values

    def bind_reserved_audit_name_provider(
        self, provider: Callable[[], Iterable[str]]
    ) -> None:
        """Bind one live bootstrap namespace after an atomic conflict check."""

        self.validate_reserved_audit_names(provider())
        self._reserved_audit_name_provider = provider

    def has_audit_name(self, name: str) -> bool:
        """Whether visible clause or bootstrap state owns one audit name."""

        return name in self._visible_reserved_audit_names() or any(
            clause.name == name
            for clause in (
                *self._clauses.values(),
                *self._external_clauses(),
                *self._dependent_clauses(),
            )
        )

    def validate_external_registries(
        self, registries: Iterable[ContextClauseRegistry]
    ) -> tuple[ContextClauseRegistry, ...]:
        values = tuple(registries)
        if any(registry is self for registry in values):
            raise FeatureContributionRuntimeError(
                "context-clause registry cannot depend on itself"
            )
        external = tuple(
            clause for registry in values for clause in registry.snapshot()
        )
        self._validate_names(external, resident=self._clauses.values())
        external_reserved = {
            name
            for registry in values
            for name in registry._local_reserved_audit_names()
        }
        conflict = next(
            (
                clause.name
                for clause in self._clauses.values()
                if clause.name in external_reserved
            ),
            None,
        )
        if conflict is not None:
            raise FeatureContributionRuntimeError(
                f"context-clause name {conflict!r} is a reserved host audit name"
            )
        return values

    def bind_external_registries(
        self, registries: Iterable[ContextClauseRegistry]
    ) -> None:
        """Atomically bind host registries after validating bare audit names."""

        values = self.validate_external_registries(registries)
        for registry in self._external_registries:
            if registry not in values:
                registry._dependent_registries.discard(self)
        for registry in values:
            registry._dependent_registries.add(self)
        self._external_registries = values

    def validate_register_batch(
        self, clauses: Iterable[ResolvedContextClause]
    ) -> tuple[ResolvedContextClause, ...]:
        values = tuple(clauses)
        identities = [clause.identity for clause in values]
        if len(set(identities)) != len(identities):
            raise FeatureContributionRuntimeError(
                "duplicate context-clause registration identity"
            )
        self._validate_names(
            values,
            resident=(
                *self._clauses.values(),
                *self._external_clauses(),
                *self._dependent_clauses(),
            ),
        )
        conflict = next(
            (identity for identity in identities if identity in self._clauses),
            None,
        )
        if conflict is not None:
            raise FeatureContributionRuntimeError(
                f"context clause is already registered for {conflict!r}"
            )
        return values

    def register_batch(
        self, clauses: Iterable[ResolvedContextClause]
    ) -> tuple[ResolvedContextClause, ...]:
        values = self.validate_register_batch(clauses)
        self._clauses.update((clause.identity, clause) for clause in values)
        return values

    def validate_unregister_batch(
        self, clauses: Iterable[ResolvedContextClause]
    ) -> tuple[ResolvedContextClause, ...]:
        values = tuple(clauses)
        if any(self._clauses.get(clause.identity) is not clause for clause in values):
            raise FeatureContributionRuntimeError(
                "active context-clause registration identity does not match"
            )
        return values

    def unregister_batch(self, clauses: Iterable[ResolvedContextClause]) -> None:
        values = self.validate_unregister_batch(clauses)
        for clause in values:
            del self._clauses[clause.identity]

    def replace_batch(
        self,
        current: Iterable[ResolvedContextClause],
        replacement: Iterable[ResolvedContextClause],
    ) -> tuple[ResolvedContextClause, ...]:
        old_values = self.validate_unregister_batch(current)
        new_values = tuple(replacement)
        old_identities = {clause.identity for clause in old_values}
        new_identities = [clause.identity for clause in new_values]
        if len(set(new_identities)) != len(new_identities):
            raise FeatureContributionRuntimeError(
                "duplicate context-clause registration identity"
            )
        self._validate_names(
            new_values,
            resident=(
                *(
                    clause
                    for clause in self._clauses.values()
                    if clause.identity not in old_identities
                ),
                *self._external_clauses(),
                *self._dependent_clauses(),
            ),
        )
        if any(
            identity in self._clauses and identity not in old_identities
            for identity in new_identities
        ):
            raise FeatureContributionRuntimeError(
                "replacement context clause conflicts with an active registration"
            )
        for clause in old_values:
            del self._clauses[clause.identity]
        self._clauses.update((clause.identity, clause) for clause in new_values)
        return new_values

    def snapshot(self) -> tuple[ResolvedContextClause, ...]:
        """Return a load-order-independent immutable prompt snapshot."""

        return tuple(
            sorted(
                self._clauses.values(),
                key=lambda clause: (clause.priority, clause.name, clause.owner),
            )
        )


class CompositeContextClauseRegistry:
    """Read-only deterministic union of host and agent lifecycle registries."""

    def __init__(self, *registries: ContextClauseRegistry) -> None:
        self._registries = tuple(registry for registry in registries if registry)

    def snapshot(self) -> tuple[ResolvedContextClause, ...]:
        clauses = tuple(
            clause
            for registry in self._registries
            for clause in registry.snapshot()
        )
        names = [clause.name for clause in clauses]
        if len(set(names)) != len(names):
            raise FeatureContributionRuntimeError(
                "composite context-clause audit names are not globally unique"
            )
        return tuple(
            sorted(
                clauses,
                key=lambda clause: (clause.priority, clause.name, clause.owner),
            )
        )

    def validate_reserved_audit_names(
        self, names: Iterable[str]
    ) -> tuple[str, ...]:
        """Ensure bootstrap audit names do not shadow any union member."""

        values = tuple(names)
        if len(set(values)) != len(values):
            raise FeatureContributionRuntimeError(
                "duplicate bootstrap audit name"
            )
        resident_names = {clause.name for clause in self.snapshot()}
        conflict = next((name for name in values if name in resident_names), None)
        if conflict is not None:
            raise FeatureContributionRuntimeError(
                f"context-clause name is already registered: {conflict!r}"
            )
        return values


class PermissionDefaultsRegistry:
    """Lifecycle-owned permission defaults, keyed by exact feature name."""

    def __init__(self) -> None:
        self._registrations: dict[str, PermissionDefaultRegistration] = {}

    def register(self, registration: PermissionDefaultRegistration) -> None:
        if registration.feature_name in self._registrations:
            raise FeatureContributionRuntimeError(
                f"permission defaults already registered for "
                f"{registration.feature_name!r}"
            )
        self._registrations[registration.feature_name] = registration

    def unregister(self, registration: PermissionDefaultRegistration) -> None:
        if self._registrations.get(registration.feature_name) is not registration:
            raise FeatureContributionRuntimeError(
                "permission-default registration identity does not match"
            )
        del self._registrations[registration.feature_name]

    def get(self, feature_name: str) -> FeaturePermissionDefaults | None:
        registration = self._registrations.get(feature_name)
        return None if registration is None else registration.defaults

    def registration(self, feature_name: str) -> PermissionDefaultRegistration | None:
        return self._registrations.get(feature_name)

    def __len__(self) -> int:
        return len(self._registrations)


class SetupStepRegistry:
    """Exact active setup steps with SDK-defined deterministic ordering."""

    def __init__(self) -> None:
        from kestrel_sovereign.setup.steps import OPTIONAL, ORDERED

        core_steps = (
            *(
                SetupStepRegistration(
                    owner="core:setup",
                    name=name,
                    step=step,
                    classification=SetupStepClassification.DEFAULT,
                    order=index,
                )
                for index, (name, step) in enumerate(ORDERED)
            ),
            *(
                SetupStepRegistration(
                    owner="core:setup",
                    name=name,
                    step=step,
                    classification=SetupStepClassification.OPTIONAL,
                    order=10_000 + index,
                )
                for index, (name, step) in enumerate(OPTIONAL)
            ),
        )
        self._registrations: dict[str, SetupStepRegistration] = {
            registration.name: registration for registration in core_steps
        }

    def preflight(self, registrations: Iterable[SetupStepRegistration]) -> None:
        incoming = tuple(registrations)
        names = [item.name for item in incoming]
        if len(set(names)) != len(names):
            raise FeatureContributionRuntimeError(
                "duplicate setup step name in contribution transition"
            )
        conflicts = set(names).intersection(self._registrations)
        if conflicts:
            raise FeatureContributionRuntimeError(
                f"setup step already registered: {sorted(conflicts)[0]}"
            )
        order_setup_step_registrations(
            (*tuple(self._registrations.values()), *incoming)
        )

    def register_batch(
        self,
        registrations: tuple[SetupStepRegistration, ...],
        *,
        prevalidated: bool = False,
    ) -> None:
        if not prevalidated:
            self.preflight(registrations)
        self._registrations.update((item.name, item) for item in registrations)

    def unregister_batch(self, registrations: tuple[SetupStepRegistration, ...]) -> None:
        for registration in registrations:
            if self._registrations.get(registration.name) is not registration:
                raise FeatureContributionRuntimeError(
                    "setup-step registration identity does not match"
                )
        for registration in registrations:
            del self._registrations[registration.name]

    def get(self, name: str) -> SetupStepRegistration | None:
        return self._registrations.get(name)

    def ordered(self) -> tuple[SetupStepRegistration, ...]:
        return order_setup_step_registrations(tuple(self._registrations.values()))

    async def run(self, name: str, context: object) -> object:
        """Execute one active setup step through the SDK awaitable boundary."""
        registration = self._registrations.get(name)
        if registration is None:
            raise KeyError(f"unknown setup step: {name}")
        return await await_contribution_result(registration.step(context))

    def __len__(self) -> int:
        return len(self._registrations)


@dataclass(frozen=True, slots=True)
class PreparedFeatureContributions:
    """Collected-once, validated objects for one prospective transition."""

    feature: object
    owner: str
    feature_name: str
    contributions: FeatureContributionSet


@dataclass(frozen=True, slots=True)
class ContributionRejection:
    """One feature refused activation, and why — reported, never silent."""

    feature: object
    feature_name: str
    reason: str


@dataclass(frozen=True, slots=True)
class PreparedTransition:
    """The prospective active set: what may activate, and what was refused.

    A collision with an ALREADY-registered key is a capability gap for the
    offending feature, not an identity gap for the agent, so it must not be
    able to abort boot for every agent on the host (issue #2951). The rejected
    features are carried here rather than raised, and
    :meth:`activatable` derives the pairing every caller needs so none of them
    has to remember to skip a rejected feature.
    """

    accepted: tuple[PreparedFeatureContributions, ...] = ()
    rejected: tuple[ContributionRejection, ...] = ()

    def activatable(
        self, features: Iterable[object]
    ) -> tuple[tuple[object, PreparedFeatureContributions], ...]:
        """``(feature, prepared)`` for exactly the features that may activate.

        Callers used to build ``{id(item.feature): item}`` and index into it
        per feature, which raises ``KeyError`` the moment a feature is excluded.
        Deriving the list here means a new call site cannot forget the skip —
        the pairing simply does not contain what must not be activated.
        """
        by_feature = {id(item.feature): item for item in self.accepted}
        return tuple(
            (feature, by_feature[id(feature)])
            for feature in features
            if id(feature) in by_feature
        )

    def only(self) -> PreparedFeatureContributions:
        """The single accepted item — or raise the rejection that refused it.

        The one-feature paths (registering or enabling a named feature) are not
        the boot batch: the caller asked for exactly this feature, and there is
        no fleet to keep up by continuing without it. A rejection there IS that
        operation's failure, so it surfaces as one.
        """
        if self.rejected:
            raise FeatureContributionRuntimeError(self.rejected[0].reason)
        return self.accepted[0]

    def rejection_for(self, feature: object) -> ContributionRejection | None:
        """The rejection recorded for *feature*, if it was refused."""
        for rejection in self.rejected:
            if rejection.feature is feature:
                return rejection
        return None

    def __iter__(self):
        """Iterate the ACCEPTED set, so ``for item in prepared`` still reads true."""
        return iter(self.accepted)

    def __len__(self) -> int:
        return len(self.accepted)


@dataclass(frozen=True, slots=True)
class ActiveFeatureContributions:
    """The exact teardown capability retained for one active lifecycle."""

    prepared: PreparedFeatureContributions
    operator_registrations: OperatorRegistrationSet
    permission_registration: PermissionDefaultRegistration | None
    context_clauses: tuple[ResolvedContextClause, ...] = ()
    execution_target_registrations: tuple[OperatorRegistrationSet, ...] = ()
    # Quarantine is deliberately best-effort across independent registries.
    # Operator registration sets are the exception: their authenticated
    # withdrawal consumes a one-shot issuance seal. If a later registry fails,
    # retain which exact capabilities already completed so a retry can finish
    # instead of presenting a consumed seal again and becoming unrecoverable.
    quarantined_operator_set_ids: set[int] = field(
        default_factory=set,
        repr=False,
        compare=False,
    )


class FeatureContributionRuntime:
    """Compose SDK declarations into existing lifecycle-owned registries.

    ``prepare_transition`` calls each SDK contribution getter exactly once and
    validates the complete prospective owner/key set without mutation.
    ``activate`` then commits one prepared member atomically.  ``deactivate``
    accepts only the same feature instance and exact retained registration
    objects, so one owner can never tear down another owner's state.
    """

    def __init__(
        self,
        *,
        operator_registry: OperatorRuntimeRegistry,
        wait_registry: WaitRegistry,
        source_registry: SourceRegistry,
        permission_defaults_registry: PermissionDefaultsRegistry | None = None,
        setup_step_registry: SetupStepRegistry | None = None,
        context_clause_registry: ContextClauseRegistry | None = None,
    ) -> None:
        self.operator_registry = operator_registry
        self.wait_registry = wait_registry
        self.source_registry = source_registry
        self.permission_defaults_registry = (
            permission_defaults_registry
            if permission_defaults_registry is not None
            else PermissionDefaultsRegistry()
        )
        self.setup_step_registry = (
            setup_step_registry
            if setup_step_registry is not None
            else SetupStepRegistry()
        )
        self.context_clause_registry = (
            context_clause_registry
            if context_clause_registry is not None
            else ContextClauseRegistry()
        )
        self._active: dict[int, ActiveFeatureContributions] = {}

    def prepare_transition(self, features: Iterable[object]) -> PreparedTransition:
        """Collect once and validate a complete prospective active set.

        Returns what may activate AND what was refused. A feature clashing with
        an already-registered key is excluded rather than raised, so one stale
        third-party package cannot abort boot for every agent on the host
        (issue #2951).
        """

        feature_values = tuple(features)
        if len({id(feature) for feature in feature_values}) != len(feature_values):
            raise FeatureContributionRuntimeError(
                "a feature instance appears twice in one contribution transition"
            )
        if any(id(feature) in self._active for feature in feature_values):
            raise FeatureContributionRuntimeError(
                "feature contributions are already active"
            )

        prepared = tuple(self._collect(feature) for feature in feature_values)
        active_prepared = tuple(item.prepared for item in self._active.values())
        try:
            validate_contribution_owner_uniqueness(
                item.owner for item in (*active_prepared, *prepared)
            )
        except Exception as exc:
            failing = self._first_owner_conflict(active_prepared, prepared)
            if failing is None:
                raise
            raise FeatureContributionCollectionError(
                failing.feature,
                "validate_contribution_owner_uniqueness",
            ) from exc
        rejections = self._preflight_keys(prepared, validate_setup_order=True)
        refused = {id(rejection.feature) for rejection in rejections}
        return PreparedTransition(
            accepted=tuple(
                item for item in prepared if id(item.feature) not in refused
            ),
            rejected=rejections,
        )

    def activate(
        self, prepared: PreparedFeatureContributions
    ) -> ActiveFeatureContributions:
        """Atomically commit one previously prepared feature contribution set."""

        if id(prepared.feature) in self._active:
            raise FeatureContributionRuntimeError(
                "feature contributions are already active"
            )
        # Recheck against state that may have changed since the preparation.
        validate_contribution_owner_uniqueness(
            [
                *(item.prepared.owner for item in self._active.values()),
                prepared.owner,
            ]
        )
        # One feature, committing now: a clash here IS this activation's
        # failure and belongs to the caller that asked for it. The per-feature
        # transaction boundary is unchanged — only the whole-set gate moved.
        rejections = self._preflight_keys((prepared,), validate_setup_order=False)
        if rejections:
            raise FeatureContributionRuntimeError(rejections[0].reason)

        values = prepared.contributions
        resolved_context_clauses = self._resolve_context_clauses(prepared)
        self.context_clause_registry.validate_register_batch(
            resolved_context_clauses
        )
        operator_set: OperatorRegistrationSet | None = None
        registered_waits = []
        registered_sources = []
        permission_registration: PermissionDefaultRegistration | None = None
        setup_registered = False
        context_registered = False
        try:
            operator_set = self.operator_registry.register(
                prepared.owner,
                services=values.services,
                workflows=values.workflows,
            )
            for registration in values.wait_providers:
                self.wait_registry.register(registration.provider)
                registered_waits.append(registration)
            sources = tuple(
                source
                for workflow in values.workflows
                for source in workflow.sources
            )
            if sources:
                # The registry records the claim AND reports what this
                # activation acquired, so the rollback below releases exactly
                # that — not a claim the feature already held from registering
                # the same source imperatively (#3053).
                with self.source_registry.claims_acquired(
                    prepared.feature
                ) as acquired:
                    self.source_registry.register_batch(
                        sources,
                        RegistrationPolicy.MANDATORY,
                        owner=prepared.feature,
                        role=CLAIM_CONTRIBUTION,
                    )
                registered_sources.extend(acquired)
            if values.permission_defaults is not None:
                candidate_permission_registration = PermissionDefaultRegistration(
                    owner=prepared.owner,
                    feature_name=prepared.feature_name,
                    defaults=values.permission_defaults,
                )
                self.permission_defaults_registry.register(
                    candidate_permission_registration
                )
                # This variable is a rollback capability. Do not expose it to
                # rollback until the registry commit succeeded, or a failed
                # register can mask the original error with an identity error.
                permission_registration = candidate_permission_registration
            # The complete transition was topologically validated during
            # preparation. Individual members may reference a later member's
            # step, so do not revalidate an intentionally partial commit.
            self.setup_step_registry.register_batch(
                values.setup_steps, prevalidated=True
            )
            setup_registered = True
            self.context_clause_registry.register_batch(resolved_context_clauses)
            context_registered = True
        except Exception:
            if context_registered:
                self.context_clause_registry.unregister_batch(
                    resolved_context_clauses
                )
            if setup_registered:
                self.setup_step_registry.unregister_batch(values.setup_steps)
            if permission_registration is not None:
                self.permission_defaults_registry.unregister(permission_registration)
            # Release exactly what THIS activation acquired: another holder may
            # already depend on the source, and a claim the feature held before
            # this activation is not this rollback's to drop (#3053).
            self.source_registry.release_acquired(
                list(reversed(registered_sources)), prepared.feature
            )
            for registration in reversed(registered_waits):
                self.wait_registry.deregister(
                    registration.name, registration.provider
                )
            if operator_set is not None:
                self.operator_registry.unregister(operator_set)
            raise

        active = ActiveFeatureContributions(
            prepared=prepared,
            operator_registrations=operator_set,
            permission_registration=permission_registration,
            context_clauses=resolved_context_clauses,
        )
        self._active[id(prepared.feature)] = active
        return active

    def register_execution_targets(
        self,
        feature: object,
        registrations: Iterable[ExecutionTargetRegistration],
    ) -> OperatorRegistrationSet:
        """Attach Sovereign execution targets to one active feature lifecycle.

        SDK 0.36 intentionally has no execution-target contribution getter.
        Features that discover targets imperatively therefore register through
        this scoped path, which binds the exact returned capability to the
        feature instance for rollback/disable teardown.
        """

        active = self._active.get(id(feature))
        if active is None or active.prepared.feature is not feature:
            raise FeatureContributionRuntimeError(
                "execution targets require an active owning feature"
            )
        registration_set = self.operator_registry.register(
            active.prepared.owner,
            execution_targets=registrations,
        )
        try:
            self._active[id(feature)] = replace(
                active,
                execution_target_registrations=(
                    *active.execution_target_registrations,
                    registration_set,
                ),
            )
        except Exception:
            self.operator_registry.unregister(registration_set)
            raise
        return registration_set

    def deactivate(self, feature: object) -> bool:
        """Remove only ``feature``'s exact retained contribution objects."""

        active = self._active.get(id(feature))
        if active is None:
            return False
        if active.prepared.feature is not feature:
            raise FeatureContributionRuntimeError(
                "active feature contribution identity does not match"
            )
        values = active.prepared.contributions
        # No per-lifecycle source list any more: the registry holds the claims,
        # so teardown just lets go of what this feature held. A source another
        # holder still needs simply stays (#3053).
        # Validate every exact inverse before mutating any registry — the
        # ownership transfer below included. Handing the source to the heir
        # before validation meant an UNRELATED mismatch (a missing wait
        # provider, say) raised after a registry had already been mutated
        # feature stayed active recording the same source: two owners, and one
        # more copy appended on every retry (#2951).
        for registration in values.wait_providers:
            if not self.wait_registry.contains(
                registration.name, registration.provider
            ):
                raise FeatureContributionRuntimeError(
                    "active wait-provider registration identity does not match"
                )
        for source in (
            source
            for workflow in values.workflows
            for source in workflow.sources
        ):
            # Every successful owner-scoped activation claims each declared
            # source, equivalent incumbents included — so a MISSING claim is not
            # "not ours", it is drift. Skipping silently let deactivate() strip
            # the other capabilities and erase `_active` as though the exact
            # inverse had succeeded (#3053).
            if feature not in self.source_registry.owners_of(source.name):
                raise FeatureContributionRuntimeError(
                    "active signal-source registration identity does not match"
                )
            current = self.source_registry.get(source.name)
            # CONTRACT, not object identity. A feature holding a claim on an
            # equivalent incumbent never registered that object, so demanding
            # `is` rejected a teardown that was entirely correct. What the
            # inverse actually needs to know is that the source still carries
            # the contract this claim was granted against (#3053).
            if current is None or not SourceRegistry.contract_equivalent(
                current, source
            ):
                raise FeatureContributionRuntimeError(
                    "active signal-source registration identity does not match"
                )
        if active.permission_registration is not None:
            if (
                self.permission_defaults_registry.registration(
                    active.permission_registration.feature_name
                )
                is not active.permission_registration
            ):
                raise FeatureContributionRuntimeError(
                    "active permission-default registration identity does not match"
                )
        for setup in values.setup_steps:
            if self.setup_step_registry.get(setup.name) is not setup:
                raise FeatureContributionRuntimeError(
                    "active setup-step registration identity does not match"
                )
        self.context_clause_registry.validate_unregister_batch(
            active.context_clauses
        )
        self.operator_registry.validate_registration_set(
            active.operator_registrations
        )
        for registration_set in active.execution_target_registrations:
            self.operator_registry.validate_registration_set(registration_set)

        for registration_set in reversed(active.execution_target_registrations):
            self.operator_registry.unregister(registration_set)
        self.operator_registry.unregister(active.operator_registrations)
        for registration in values.wait_providers:
            self.wait_registry.deregister(registration.name, registration.provider)
        if active.permission_registration is not None:
            self.permission_defaults_registry.unregister(
                active.permission_registration
            )
        self.setup_step_registry.unregister_batch(values.setup_steps)
        self.context_clause_registry.unregister_batch(active.context_clauses)
        # Past every validation, in the same mutating stretch as the other
        # unregistrations: drop this feature's claims. The registry removes each
        # source only when its last holder lets go.
        self.source_registry.release_all(feature)
        del self._active[id(feature)]
        return True

    def quarantine(self, feature: object) -> bool:
        """Fail closed after drift prevents the ordinary exact inverse.

        ``deactivate`` deliberately validates the complete inverse before it
        mutates anything. If an operator or external component has already
        removed one retained capability, that atomic inverse refuses to run.
        Runtime rollback still needs a repair seam: remove every surviving
        capability only when its exact retained identity is present, tolerate
        capabilities that are already absent, and never remove a replacement
        object. Context clauses are checked and withdrawn first so a failed
        quarantine cannot leave feature-owned prompt bytes published while a
        caller reports the feature disabled.
        """

        active = self._active.get(id(feature))
        if active is None:
            return False
        if active.prepared.feature is not feature:
            raise FeatureContributionRuntimeError(
                "active feature contribution identity does not match"
            )

        # A different object at the same context identity is not ours to erase.
        # Refuse before any quarantine mutation rather than risk deleting a
        # replacement clause published by another lifecycle generation.
        for clause in active.context_clauses:
            resident = self.context_clause_registry._clauses.get(clause.identity)
            if resident is not None and resident is not clause:
                raise FeatureContributionRuntimeError(
                    "feature context clauses could not be quarantined"
                )
        exact_context = tuple(
            clause
            for clause in active.context_clauses
            if self.context_clause_registry._clauses.get(clause.identity) is clause
        )
        if exact_context:
            self.context_clause_registry.unregister_batch(exact_context)

        failed = False

        def attempt(operation) -> None:
            nonlocal failed
            try:
                operation()
            except Exception:
                # The public error below is deliberately fixed text. Registry
                # exceptions can include third-party names or representations.
                failed = True

        def quarantine_operator_set(registration_set: OperatorRegistrationSet) -> None:
            nonlocal failed
            capability_id = id(registration_set)
            if capability_id in active.quarantined_operator_set_ids:
                return
            try:
                withdrawn = self.operator_registry.quarantine_registration_set(
                    registration_set
                )
            except Exception:
                # Keep the retained capability unmarked so a later repair can
                # retry its authenticated withdrawal.
                failed = True
                return
            if withdrawn is not True:
                failed = True
                return
            active.quarantined_operator_set_ids.add(capability_id)

        for registration_set in reversed(active.execution_target_registrations):
            quarantine_operator_set(registration_set)
        quarantine_operator_set(active.operator_registrations)

        for registration in active.prepared.contributions.wait_providers:
            if self.wait_registry.contains(registration.name, registration.provider):
                attempt(
                    lambda item=registration: self.wait_registry.deregister(
                        item.name, item.provider
                    )
                )

        permission = active.permission_registration
        if permission is not None:
            resident_permission = self.permission_defaults_registry.registration(
                permission.feature_name
            )
            if resident_permission is permission:
                attempt(
                    lambda: self.permission_defaults_registry.unregister(permission)
                )
            elif resident_permission is not None:
                failed = True

        exact_setup = tuple(
            registration
            for registration in active.prepared.contributions.setup_steps
            if self.setup_step_registry.get(registration.name) is registration
        )
        if exact_setup:
            attempt(lambda: self.setup_step_registry.unregister_batch(exact_setup))
        if any(
            (resident := self.setup_step_registry.get(registration.name)) is not None
            and resident is not registration
            for registration in active.prepared.contributions.setup_steps
        ):
            failed = True

        sources = tuple(
            source
            for workflow in active.prepared.contributions.workflows
            for source in workflow.sources
        )
        attempt(lambda: self.source_registry.quarantine_claims(feature, sources))
        if failed:
            raise FeatureContributionRuntimeError(
                "feature contributions could not be quarantined"
            ) from None

        del self._active[id(feature)]
        return True

    def is_active(self, feature: object) -> bool:
        active = self._active.get(id(feature))
        return active is not None and active.prepared.feature is feature

    def active_owners(self) -> tuple[str, ...]:
        return tuple(item.prepared.owner for item in self._active.values())

    def active_context_clauses(self) -> tuple[ResolvedContextClause, ...]:
        """Return only core-owned rendered bytes; no feature code runs here."""

        return self.context_clause_registry.snapshot()

    def refresh_context_clauses(
        self, feature: object
    ) -> tuple[ResolvedContextClause, ...]:
        """Resolve a deliberate configuration transition, never a turn read."""

        active = self._active.get(id(feature))
        if active is None or active.prepared.feature is not feature:
            raise FeatureContributionRuntimeError(
                "context-clause refresh requires an active owning feature"
            )
        replacement = self._resolve_context_clauses(active.prepared)
        committed = self.context_clause_registry.replace_batch(
            active.context_clauses, replacement
        )
        self._active[id(feature)] = replace(active, context_clauses=committed)
        return committed

    def refresh_all_context_clauses(
        self,
    ) -> tuple[ResolvedContextClause, ...]:
        """Atomically republish every active clause after a host transition.

        Privacy and other host-owned configuration can change what an
        out-of-tree renderer is allowed to disclose without invoking a feature
        tool. Resolve every renderer before touching the registry, then replace
        the complete agent-owned set in one validated mutation so prompts never
        observe a mixture of pre- and post-transition bytes.
        """

        active_items = tuple(self._active.items())
        replacements = tuple(
            (
                key,
                active,
                self._resolve_context_clauses(active.prepared),
            )
            for key, active in active_items
        )
        return self._commit_all_context_clause_replacements(replacements)

    def suppress_all_context_clauses(
        self,
    ) -> tuple[ResolvedContextClause, ...]:
        """Fail closed to empty bodies without executing feature code."""

        replacements = tuple(
            (
                key,
                active,
                tuple(replace(clause, body="") for clause in active.context_clauses),
            )
            for key, active in tuple(self._active.items())
        )
        return self._commit_all_context_clause_replacements(replacements)

    def _commit_all_context_clause_replacements(
        self,
        replacements: tuple[
            tuple[
                int,
                ActiveFeatureContributions,
                tuple[ResolvedContextClause, ...],
            ],
            ...,
        ],
    ) -> tuple[ResolvedContextClause, ...]:
        current = tuple(
            clause
            for _key, active, _replacement in replacements
            for clause in active.context_clauses
        )
        requested = tuple(
            clause
            for _key, _active, replacement in replacements
            for clause in replacement
        )
        committed = self.context_clause_registry.replace_batch(current, requested)
        offset = 0
        for key, active, replacement in replacements:
            end = offset + len(replacement)
            self._active[key] = replace(
                active,
                context_clauses=committed[offset:end],
            )
            offset = end
        return committed

    @staticmethod
    def _resolve_context_clauses(
        prepared: PreparedFeatureContributions,
    ) -> tuple[ResolvedContextClause, ...]:
        sanitized_error: FeatureContributionCollectionError | None = None
        try:
            resolved = []
            for registration in prepared.contributions.context_clauses:
                body = registration.renderer()
                if not isinstance(body, str):
                    raise TypeError("context clause renderer must return str")
                resolved.append(
                    ResolvedContextClause(
                        owner=registration.owner,
                        name=registration.name,
                        priority=registration.priority,
                        body=body,
                        registration=registration,
                    )
                )
            return tuple(resolved)
        except (Exception, asyncio.CancelledError):
            # Renderer exceptions are arbitrary out-of-tree objects and may
            # carry user-authored text or credentials in their message.  Do
            # not retain them as a cause: API/startup error boundaries format
            # complete exception chains into production logs.
            sanitized_error = FeatureContributionCollectionError(
                prepared.feature, "render_context_clauses"
            )
        # Raising outside the active handler is load-bearing: ``from None``
        # hides an exception context from formatting but still retains the raw
        # object (and any secret text) on ``__context__``.
        raise sanitized_error

    @staticmethod
    def _collect(feature: object) -> PreparedFeatureContributions:
        from kestrel_sdk.features.base import Feature
        from kestrel_sdk.features.host_base import HostFeature

        if not isinstance(feature, (Feature, HostFeature)):
            # Existing embedders and lifecycle tests may use duck-typed feature
            # doubles. They predate the SDK contribution contract and have no
            # declarative state to collect; do not mistake MagicMock's dynamic
            # attributes for real contribution methods.
            owner = f"legacy:{id(feature)}"
            feature_name = getattr(feature, "name", type(feature).__name__)
            contributions = validate_feature_contributions(
                owner,
                tool_names=(),
                services=(),
                wait_providers=(),
                workflows=(),
                permission_defaults=None,
                setup_steps=(),
                context_clauses=(),
            )
            return PreparedFeatureContributions(
                feature=feature,
                owner=owner,
                feature_name=feature_name,
                contributions=contributions,
            )
        try:
            owner = feature.contribution_owner
        except Exception as exc:
            raise FeatureContributionCollectionError(
                feature, "contribution_owner"
            ) from exc
        feature_name = getattr(feature, "name", type(feature).__name__)
        # These are deliberately single reads: getters may return bound or
        # generated implementation objects whose identity is lifecycle state.
        tools = FeatureContributionRuntime._call_collection_getter(
            feature,
            "get_tools",
            materialize=True,
            optional=True,
        )
        try:
            tool_names = tuple(tool.name for tool in tools)
        except Exception as exc:
            raise FeatureContributionCollectionError(feature, "get_tools") from exc
        services = FeatureContributionRuntime._call_collection_getter(
            feature, "get_service_registrations", materialize=True
        )
        waits = FeatureContributionRuntime._call_collection_getter(
            feature, "get_wait_provider_registrations", materialize=True
        )
        workflows = FeatureContributionRuntime._call_collection_getter(
            feature, "get_workflow_registrations", materialize=True
        )
        permissions = FeatureContributionRuntime._call_collection_getter(
            feature, "get_feature_permission_defaults", materialize=False
        )
        setup_steps = FeatureContributionRuntime._call_collection_getter(
            feature, "get_setup_step_registrations", materialize=True
        )
        context_clauses = FeatureContributionRuntime._call_collection_getter(
            feature,
            "get_context_clause_registrations",
            materialize=True,
            optional=True,
            discard_cause=True,
        )
        try:
            contributions = validate_feature_contributions(
                owner,
                tool_names=tool_names,
                services=services,
                wait_providers=waits,
                workflows=workflows,
                permission_defaults=permissions,
                setup_steps=setup_steps,
                context_clauses=context_clauses,
            )
            if isinstance(feature, Feature):
                agent_id = getattr(feature.agent, "agent_id", None) or getattr(
                    feature.agent, "did", None
                )
                for registration in contributions.services:
                    if registration.descriptor.scope is not ServiceScope.AGENT:
                        raise ContributionContractError(
                            "agent Feature services must use ServiceScope.AGENT"
                        )
                    if registration.agent_id != agent_id:
                        raise ContributionContractError(
                            "agent Feature service agent_id must match its agent"
                        )
            else:
                for registration in contributions.services:
                    if registration.descriptor.scope is not ServiceScope.HOST:
                        raise ContributionContractError(
                            "HostFeature services must use ServiceScope.HOST"
                        )
        except Exception as exc:
            raise FeatureContributionCollectionError(
                feature, "validate_feature_contributions"
            ) from exc
        return PreparedFeatureContributions(
            feature=feature,
            owner=owner,
            feature_name=feature_name,
            contributions=contributions,
        )

    @staticmethod
    def _call_collection_getter(
        feature: object,
        getter: str,
        *,
        materialize: bool,
        optional: bool = False,
        discard_cause: bool = False,
    ) -> object:
        """Read one SDK getter once and keep all failures at one typed edge."""

        sanitized_error: FeatureContributionCollectionError | None = None
        try:
            method = getattr(feature, getter, None)
            if method is None and optional:
                return ()
            value = method()
            return tuple(value) if materialize else value
        except (Exception, asyncio.CancelledError) as exc:
            if discard_cause:
                # Context clauses are user-authored prompt material and their
                # getters commonly read secret-bearing feature configuration.
                # Startup/API logging formats complete exception chains, so the
                # fixed boundary must not retain arbitrary out-of-tree text.
                sanitized_error = FeatureContributionCollectionError(feature, getter)
            if isinstance(exc, asyncio.CancelledError):
                if sanitized_error is None:
                    raise
            elif sanitized_error is None:
                raise FeatureContributionCollectionError(feature, getter) from exc
        # Do not raise this while the untrusted exception is being handled:
        # Python would retain it through ``__context__`` even with ``from None``.
        raise sanitized_error

    @staticmethod
    def _first_owner_conflict(
        active: tuple[PreparedFeatureContributions, ...],
        prepared: tuple[PreparedFeatureContributions, ...],
    ) -> PreparedFeatureContributions | None:
        """Return the incoming member that first duplicates an active owner."""

        seen = {item.owner for item in active}
        for item in prepared:
            if item.owner in seen:
                return item
            seen.add(item.owner)
        return None

    def _preflight_keys(
        self,
        prepared: tuple[PreparedFeatureContributions, ...],
        *,
        validate_setup_order: bool,
    ) -> tuple[ContributionRejection, ...]:
        """Validate a prospective set. TWO failure classes, resolved differently.

        **Transition-invalid** — two features in this same batch claiming one
        key, or a setup order the resulting set cannot satisfy. The proposed set
        is incoherent and there is no subset to prefer, so this still raises.

        **Feature-rejected** — one feature clashing with something ALREADY
        registered. That is a capability gap for *that feature*; it is not an
        identity gap for the agent, and it must not abort boot for every agent
        on the host. Core registers the same generic sources under ``OPTIONAL``
        ("nothing ever raises"), so the identical collision was survivable on
        the core path and fatal here (issue #2951). Returned as rejections for
        the caller to exclude and report.
        """
        service_refs = []
        workflow_names = []
        wait_names = []
        source_names = []
        permission_names = []
        setup_steps = []
        context_names = []
        for item in prepared:
            values = item.contributions
            service_refs.extend(registration.reference for registration in values.services)
            workflow_names.extend(registration.name for registration in values.workflows)
            wait_names.extend(registration.name for registration in values.wait_providers)
            source_names.extend(
                source.name
                for workflow in values.workflows
                for source in workflow.sources
            )
            if values.permission_defaults is not None:
                permission_names.append(item.feature_name)
            setup_steps.extend(values.setup_steps)
            context_names.extend(
                registration.name for registration in values.context_clauses
            )

        self._require_unique(service_refs, "service reference")
        self._require_unique(workflow_names, "workflow name")
        self._require_unique(wait_names, "wait-provider name")
        self._require_unique(source_names, "workflow source name")
        self._require_unique(permission_names, "permission feature name")
        self._require_unique(
            [registration.name for registration in setup_steps], "setup step name"
        )
        self._require_unique(context_names, "context-clause name")
        self.context_clause_registry.validate_declared_names(context_names)

        rejections = tuple(
            ContributionRejection(item.feature, item.feature_name, reason)
            for item, reason in (
                (item, self._active_conflict(item))
                for item in prepared
            )
            if reason is not None
        )

        if validate_setup_order:
            # Over the ACCEPTED steps only: a rejected feature never activates,
            # so its steps are not part of the resulting set and ordering the
            # batch around them would validate a set that will not exist.
            refused = {id(rejection.feature) for rejection in rejections}
            self.setup_step_registry.preflight(
                tuple(
                    registration
                    for item in prepared
                    if id(item.feature) not in refused
                    for registration in item.contributions.setup_steps
                )
            )
        return rejections

    def _active_conflict(self, item: PreparedFeatureContributions) -> str | None:
        """Why ONE feature cannot activate against what is already registered.

        Returns the first reason, or None. Per-feature by construction: the flat
        key lists above cannot say which feature contributed a colliding name,
        and a rejection that cannot name its feature cannot exclude it either.
        """
        values = item.contributions
        for registration in values.services:
            if self.operator_registry.resolve_service(registration.reference) is not None:
                return "service registration conflicts with an active service"
        for registration in values.workflows:
            if self.operator_registry.get_workflow_registration(registration.name) is not None:
                return f"workflow actor already registered: {registration.name}"
        for registration in values.wait_providers:
            if self.wait_registry.get(registration.name) is not None:
                return f"wait provider already registered: {registration.name}"
        for workflow in values.workflows:
            for source in workflow.sources:
                try:
                    SourceRegistry._validate(source)
                except Exception as exc:  # noqa: BLE001 - reject the feature, not the boot
                    return f"invalid signal source {source.name!r}: {exc}"
                existing = self.source_registry.get(source.name)
                if existing is None:
                    continue
                # The registry already knows what "the same source" means, and
                # it is not the NAME. A byte-identical re-registration — the
                # common case when core and a feature ship the same generic
                # source through an extraction — is a no-op success under every
                # declared policy; only a DIFFERENT contract is a clash.
                if not SourceRegistry.contract_equivalent(existing, source):
                    return (
                        "signal source already registered with a different "
                        f"contract: {source.name}"
                    )
        if values.permission_defaults is not None:
            if self.permission_defaults_registry.registration(item.feature_name) is not None:
                return f"permission defaults already registered for {item.feature_name}"
        # Unconditional: `preflight` ALSO raises on a name already registered,
        # so gating this on the order-validating path left a setup-step
        # collision aborting the whole boot — the very blast radius this split
        # exists to remove, still intact for one key type (#2951).
        for registration in values.setup_steps:
            if self.setup_step_registry.get(registration.name) is not None:
                return f"setup step already registered: {registration.name}"
        for registration in values.context_clauses:
            if self.context_clause_registry.has_audit_name(registration.name):
                return (
                    "context clause already registered: "
                    f"{registration.name}"
                )
        return None

    @staticmethod
    def _require_unique(values: list[object], description: str) -> None:
        if len(set(values)) != len(values):
            raise FeatureContributionRuntimeError(
                f"duplicate {description} in contribution transition"
            )


__all__ = [
    "ActiveFeatureContributions",
    "FeatureContributionCollectionError",
    "FeatureContributionRuntime",
    "FeatureContributionRuntimeError",
    "PermissionDefaultRegistration",
    "ContributionRejection",
    "PermissionDefaultsRegistry",
    "PreparedFeatureContributions",
    "PreparedTransition",
    "SetupStepRegistry",
]
