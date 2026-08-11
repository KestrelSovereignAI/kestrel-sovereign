"""Atomic lifecycle wiring for SDK feature contributions.

The SDK owns declaration and validation contracts.  Sovereign owns the live
registries and retains the exact canonical objects returned for one feature
enable (or host-feature start) transition until its matching teardown.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

from kestrel_sdk.features import (
    ContributionContractError,
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

from kestrel_sovereign.operator import (
    ExecutionTargetRegistration,
    OperatorRegistrationSet,
    OperatorRuntimeRegistry,
)
from kestrel_sovereign.signals import RegistrationPolicy, SourceRegistry
from kestrel_sovereign.waits import WaitRegistry


class FeatureContributionRuntimeError(RuntimeError):
    """A contribution transition cannot be committed or exactly reversed."""


@dataclass(frozen=True, slots=True)
class PermissionDefaultRegistration:
    """Exact feature-name projection of one SDK permission descriptor."""

    owner: str
    feature_name: str
    defaults: FeaturePermissionDefaults


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
class ActiveFeatureContributions:
    """The exact teardown capability retained for one active lifecycle."""

    prepared: PreparedFeatureContributions
    operator_registrations: OperatorRegistrationSet
    permission_registration: PermissionDefaultRegistration | None
    execution_target_registrations: tuple[OperatorRegistrationSet, ...] = ()


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
        self._active: dict[int, ActiveFeatureContributions] = {}

    def prepare_transition(
        self, features: Iterable[object]
    ) -> tuple[PreparedFeatureContributions, ...]:
        """Collect once and validate a complete prospective active set."""

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
        validate_contribution_owner_uniqueness(
            item.owner for item in (*active_prepared, *prepared)
        )
        self._preflight_keys(prepared, validate_setup_order=True)
        return prepared

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
        self._preflight_keys((prepared,), validate_setup_order=False)

        values = prepared.contributions
        operator_set: OperatorRegistrationSet | None = None
        registered_waits = []
        registered_sources = []
        permission_registration: PermissionDefaultRegistration | None = None
        setup_registered = False
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
                self.source_registry.register_batch(
                    sources, RegistrationPolicy.MANDATORY
                )
                registered_sources.extend(sources)
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
        except Exception:
            if setup_registered:
                self.setup_step_registry.unregister_batch(values.setup_steps)
            if permission_registration is not None:
                self.permission_defaults_registry.unregister(permission_registration)
            for source in reversed(registered_sources):
                if self.source_registry.get(source.name) is source:
                    self.source_registry.unregister(source.name)
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
        sources = tuple(
            source for workflow in values.workflows for source in workflow.sources
        )

        # Validate every exact inverse before mutating any registry.
        for registration in values.wait_providers:
            if not self.wait_registry.contains(
                registration.name, registration.provider
            ):
                raise FeatureContributionRuntimeError(
                    "active wait-provider registration identity does not match"
                )
        for source in sources:
            if self.source_registry.get(source.name) is not source:
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
        for source in sources:
            self.source_registry.unregister(source.name)
        if active.permission_registration is not None:
            self.permission_defaults_registry.unregister(
                active.permission_registration
            )
        self.setup_step_registry.unregister_batch(values.setup_steps)
        del self._active[id(feature)]
        return True

    def is_active(self, feature: object) -> bool:
        active = self._active.get(id(feature))
        return active is not None and active.prepared.feature is feature

    def active_owners(self) -> tuple[str, ...]:
        return tuple(item.prepared.owner for item in self._active.values())

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
            )
            return PreparedFeatureContributions(
                feature=feature,
                owner=owner,
                feature_name=feature_name,
                contributions=contributions,
            )
        owner = feature.contribution_owner
        feature_name = getattr(feature, "name", type(feature).__name__)
        tools = tuple(feature.get_tools()) if hasattr(feature, "get_tools") else ()
        tool_names = tuple(tool.name for tool in tools)
        # These are deliberately single reads: getters may return bound or
        # generated implementation objects whose identity is lifecycle state.
        services = feature.get_service_registrations()
        waits = feature.get_wait_provider_registrations()
        workflows = feature.get_workflow_registrations()
        permissions = feature.get_feature_permission_defaults()
        setup_steps = feature.get_setup_step_registrations()
        contributions = validate_feature_contributions(
            owner,
            tool_names=tool_names,
            services=services,
            wait_providers=waits,
            workflows=workflows,
            permission_defaults=permissions,
            setup_steps=setup_steps,
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
        return PreparedFeatureContributions(
            feature=feature,
            owner=owner,
            feature_name=feature_name,
            contributions=contributions,
        )

    def _preflight_keys(
        self,
        prepared: tuple[PreparedFeatureContributions, ...],
        *,
        validate_setup_order: bool,
    ) -> None:
        service_refs = []
        workflow_names = []
        wait_names = []
        source_names = []
        permission_names = []
        setup_steps = []
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

        self._require_unique(service_refs, "service reference")
        self._require_unique(workflow_names, "workflow name")
        self._require_unique(wait_names, "wait-provider name")
        self._require_unique(source_names, "workflow source name")
        self._require_unique(permission_names, "permission feature name")
        self._require_unique(
            [registration.name for registration in setup_steps], "setup step name"
        )

        for reference in service_refs:
            if self.operator_registry.resolve_service(reference) is not None:
                raise FeatureContributionRuntimeError(
                    "service registration conflicts with an active service"
                )
        for name in workflow_names:
            if self.operator_registry.get_workflow_registration(name) is not None:
                raise FeatureContributionRuntimeError(
                    f"workflow actor already registered: {name}"
                )
        for name in wait_names:
            if self.wait_registry.get(name) is not None:
                raise FeatureContributionRuntimeError(
                    f"wait provider already registered: {name}"
                )
        for item in prepared:
            for workflow in item.contributions.workflows:
                for source in workflow.sources:
                    SourceRegistry._validate(source)
                    if self.source_registry.get(source.name) is not None:
                        raise FeatureContributionRuntimeError(
                            f"signal source already registered: {source.name}"
                        )
        for name in permission_names:
            if self.permission_defaults_registry.registration(name) is not None:
                raise FeatureContributionRuntimeError(
                    f"permission defaults already registered for {name}"
                )
        if validate_setup_order:
            self.setup_step_registry.preflight(tuple(setup_steps))
        else:
            for registration in setup_steps:
                if self.setup_step_registry.get(registration.name) is not None:
                    raise FeatureContributionRuntimeError(
                        f"setup step already registered: {registration.name}"
                    )

    @staticmethod
    def _require_unique(values: list[object], description: str) -> None:
        if len(set(values)) != len(values):
            raise FeatureContributionRuntimeError(
                f"duplicate {description} in contribution transition"
            )


__all__ = [
    "ActiveFeatureContributions",
    "FeatureContributionRuntime",
    "FeatureContributionRuntimeError",
    "PermissionDefaultRegistration",
    "PermissionDefaultsRegistry",
    "PreparedFeatureContributions",
    "SetupStepRegistry",
]
