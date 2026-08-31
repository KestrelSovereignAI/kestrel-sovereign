"""Lifecycle-owned registries for generic SDK operator contracts.

This module is deliberately an in-memory lifecycle boundary, not a workflow
engine.  It owns active services, workflow actors, and opaque execution target
handles while their contributing lifecycle is active.  Durable runs,
artifacts, HTTP projection, and feature-specific behavior belong elsewhere.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import RLock

from kestrel_sdk.features import (
    WorkflowRegistration,
    validate_contribution_owner_uniqueness,
)
from kestrel_sdk.operator import (
    ExecutionTargetDescriptor,
    ExecutionTargetReference,
    OperatorAuthorizationError,
    OperatorContext,
    ServiceReference,
    ServiceRegistration,
    ServiceRequirement,
)


class OperatorRegistrationError(RuntimeError):
    """Base class for rejected operator-runtime registration transitions."""


class OperatorRegistrationConflictError(OperatorRegistrationError):
    """A prospective registration conflicts with active runtime state."""


class OperatorRegistrationIdentityError(OperatorRegistrationError):
    """Lifecycle teardown did not present the exact active objects."""


class ExecutionTargetUnavailableError(OperatorAuthorizationError):
    """An execution target cannot be disclosed or used by this request."""


_TARGET_UNAVAILABLE = "execution target is unavailable"
_TARGET_CONFLICT = "execution target registration conflicts with an active target"


@dataclass(frozen=True, slots=True)
class ExecutionTargetRegistration:
    """Lifecycle owner, safe descriptor, opaque handle, and entitlements.

    ``entitled_capabilities`` is copied and must be a subset of the public
    descriptor's capabilities.  Omitting it entitles the descriptor's complete
    closed capability set.  The handle never appears in projections or errors.
    """

    owner: str
    descriptor: ExecutionTargetDescriptor
    handle: object
    entitled_capabilities: frozenset[str] | None = None

    def __post_init__(self) -> None:
        _validate_owner(self.owner)
        if not isinstance(self.descriptor, ExecutionTargetDescriptor):
            raise TypeError("descriptor must be an ExecutionTargetDescriptor")
        if self.handle is None:
            raise ValueError("handle must not be None")
        if isinstance(self.entitled_capabilities, (str, bytes)):
            raise TypeError("entitled_capabilities must contain capability names")
        entitled = (
            self.descriptor.capabilities
            if self.entitled_capabilities is None
            else frozenset(self.entitled_capabilities)
        )
        if not entitled.issubset(self.descriptor.capabilities):
            raise ValueError(
                "entitled_capabilities must be a subset of descriptor capabilities"
            )
        object.__setattr__(self, "entitled_capabilities", entitled)

    @property
    def identity(self) -> tuple[str, str, str]:
        """Return the exact public lifecycle identity for this target."""

        return (
            self.owner,
            self.descriptor.tenant_id,
            self.descriptor.target_id,
        )


@dataclass(frozen=True, slots=True)
class OperatorRegistrationSet:
    """Exact objects committed by one atomic lifecycle transition."""

    owner: str
    services: tuple[ServiceRegistration, ...] = ()
    workflows: tuple[WorkflowRegistration, ...] = ()
    execution_targets: tuple[ExecutionTargetRegistration, ...] = ()


class OperatorRuntimeRegistry:
    """Authoritative active registry for generic operator integrations.

    Registration batches are validated in full before any map changes.  The
    returned :class:`OperatorRegistrationSet` is the teardown capability:
    unregistering requires that exact set, each exact registration object, and
    each registration's original opaque implementation object.  Registrations
    are never represented as an ownership stack, so removing one lifecycle
    cannot reveal a superseded inactive implementation.
    """

    def __init__(self, *, clock: Callable[[], datetime] | None = None) -> None:
        self._clock = clock or (lambda: datetime.now(UTC))
        self._lock = RLock()
        self._services: dict[ServiceReference, ServiceRegistration] = {}
        self._workflows: dict[str, WorkflowRegistration] = {}
        self._targets: dict[
            tuple[str, str], ExecutionTargetRegistration
        ] = {}
        self._active_sets: dict[int, OperatorRegistrationSet] = {}

    def register(
        self,
        owner: str,
        *,
        services: Iterable[ServiceRegistration] = (),
        workflows: Iterable[WorkflowRegistration] = (),
        execution_targets: Iterable[ExecutionTargetRegistration] = (),
    ) -> OperatorRegistrationSet:
        """Atomically register one lifecycle's exact contribution objects."""

        _validate_owner(owner)
        service_values = self._typed_tuple(
            services, ServiceRegistration, "services"
        )
        workflow_values = self._typed_tuple(
            workflows, WorkflowRegistration, "workflows"
        )
        target_values = self._typed_tuple(
            execution_targets,
            ExecutionTargetRegistration,
            "execution_targets",
        )
        for registration in (*service_values, *workflow_values, *target_values):
            if registration.owner != owner:
                raise OperatorRegistrationIdentityError(
                    "registration owner does not match lifecycle owner"
                )

        service_keys = tuple(item.reference for item in service_values)
        workflow_keys = tuple(item.name for item in workflow_values)
        target_keys = tuple(
            (item.descriptor.tenant_id, item.descriptor.target_id)
            for item in target_values
        )
        self._require_unique(service_keys, "service identity")
        self._require_unique(workflow_keys, "workflow name")
        self._require_unique(target_keys, "execution target identity")

        registration_set = OperatorRegistrationSet(
            owner=owner,
            services=service_values,
            workflows=workflow_values,
            execution_targets=target_values,
        )
        with self._lock:
            if any(key in self._services for key in service_keys):
                raise OperatorRegistrationConflictError(
                    "service registration conflicts with an active service"
                )
            if any(key in self._workflows for key in workflow_keys):
                raise OperatorRegistrationConflictError(
                    "workflow registration conflicts with an active workflow"
                )
            if any(key in self._targets for key in target_keys):
                raise OperatorRegistrationConflictError(_TARGET_CONFLICT)

            self._services.update(zip(service_keys, service_values, strict=True))
            self._workflows.update(zip(workflow_keys, workflow_values, strict=True))
            self._targets.update(zip(target_keys, target_values, strict=True))
            self._active_sets[id(registration_set)] = registration_set
        return registration_set

    def unregister(self, registration_set: OperatorRegistrationSet) -> None:
        """Atomically remove only one exact active lifecycle registration set."""

        if not isinstance(registration_set, OperatorRegistrationSet):
            raise TypeError("registration_set must be an OperatorRegistrationSet")
        with self._lock:
            active = self._validate_registration_set_locked(registration_set)
            for registration in active.services:
                del self._services[registration.reference]
            for registration in active.workflows:
                del self._workflows[registration.name]
            for registration in active.execution_targets:
                del self._targets[
                    (registration.descriptor.tenant_id, registration.descriptor.target_id)
                ]
            del self._active_sets[id(active)]

    def quarantine_registration_set(
        self, registration_set: OperatorRegistrationSet
    ) -> bool:
        """Withdraw exact survivors from one drifted active registration set.

        Ordinary :meth:`unregister` is an atomic exact inverse and therefore
        refuses a partially missing set. Lifecycle recovery has a different
        job: remove each retained object that is still exactly resident,
        tolerate an already-absent object, preserve any replacement, and retire
        the original set capability so a complete generation can be prepared.
        """

        if not isinstance(registration_set, OperatorRegistrationSet):
            raise TypeError("registration_set must be an OperatorRegistrationSet")
        with self._lock:
            active = self._active_sets.get(id(registration_set))
            if active is not registration_set:
                return False
            for registration in active.services:
                resident = self._services.get(registration.reference)
                if (
                    resident is registration
                    and resident.service is registration.service
                ):
                    del self._services[registration.reference]
            for registration in active.workflows:
                resident = self._workflows.get(registration.name)
                if resident is registration and resident.actor is registration.actor:
                    del self._workflows[registration.name]
            for registration in active.execution_targets:
                key = (
                    registration.descriptor.tenant_id,
                    registration.descriptor.target_id,
                )
                resident = self._targets.get(key)
                if (
                    resident is registration
                    and resident.handle is registration.handle
                ):
                    del self._targets[key]
            del self._active_sets[id(active)]
            return True

    def validate_registration_set(
        self, registration_set: OperatorRegistrationSet
    ) -> None:
        """Validate an exact teardown capability without mutating the registry."""

        if not isinstance(registration_set, OperatorRegistrationSet):
            raise TypeError("registration_set must be an OperatorRegistrationSet")
        with self._lock:
            self._validate_registration_set_locked(registration_set)

    def resolve_service(self, reference: ServiceReference) -> object | None:
        """Resolve an exact service without crossing host/agent namespaces."""

        if not isinstance(reference, ServiceReference):
            raise TypeError("reference must be a ServiceReference")
        with self._lock:
            registration = self._services.get(reference)
            return None if registration is None else registration.service

    def resolve_compatible_service(
        self, requirement: ServiceRequirement
    ) -> object | None:
        """Resolve the highest stable compatible active service version."""

        if not isinstance(requirement, ServiceRequirement):
            raise TypeError("requirement must be a ServiceRequirement")
        with self._lock:
            candidates = [
                registration
                for reference, registration in self._services.items()
                if reference.name == requirement.name
                and reference.scope is requirement.scope
                and reference.agent_id == requirement.agent_id
                and requirement.accepts(registration.descriptor)
            ]
            if not candidates:
                return None
            selected = max(
                candidates,
                key=lambda item: self._stable_version_key(item.descriptor.version),
            )
            return selected.service

    def resolve_workflow_actor(self, name: str) -> Callable[..., object] | None:
        """Return the active actor for an exact workflow name, if present."""

        if not isinstance(name, str):
            raise TypeError("name must be a string")
        with self._lock:
            registration = self._workflows.get(name)
            return None if registration is None else registration.actor

    def get_workflow_registration(
        self, name: str
    ) -> WorkflowRegistration | None:
        """Return the active exact workflow registration for runtime wiring."""

        if not isinstance(name, str):
            raise TypeError("name must be a string")
        with self._lock:
            return self._workflows.get(name)

    async def resolve_execution_target(
        self,
        reference: ExecutionTargetReference,
        context: OperatorContext,
    ) -> object:
        """Return an authorized opaque handle or one sanitized denial."""

        if not isinstance(reference, ExecutionTargetReference):
            raise TypeError("reference must be an ExecutionTargetReference")
        self._require_context(context)
        try:
            context.require_fresh(self._trusted_now())
            with self._lock:
                registration = self._targets.get(
                    (context.tenant_id, reference.target_id)
                )
                if registration is None:
                    raise ExecutionTargetUnavailableError(_TARGET_UNAVAILABLE)
                descriptor = registration.descriptor
                if (
                    not context.matches_tenant(descriptor.tenant_id)
                    or descriptor.boundary_id != reference.boundary_id
                    or not context.allows_boundary(reference.boundary_id)
                    or reference.capability not in descriptor.capabilities
                    or reference.capability not in registration.entitled_capabilities
                    or not context.allows_capability(reference.capability)
                ):
                    raise ExecutionTargetUnavailableError(_TARGET_UNAVAILABLE)
                return registration.handle
        except OperatorAuthorizationError as error:
            if isinstance(error, ExecutionTargetUnavailableError):
                raise
            raise ExecutionTargetUnavailableError(_TARGET_UNAVAILABLE) from None

    def list_execution_target_descriptors(
        self, context: OperatorContext
    ) -> tuple[ExecutionTargetDescriptor, ...]:
        """Project only authorized, entitled, browser-safe target fields."""

        self._require_context(context)
        try:
            context.require_fresh(self._trusted_now())
        except OperatorAuthorizationError:
            raise ExecutionTargetUnavailableError(_TARGET_UNAVAILABLE) from None
        with self._lock:
            registrations = tuple(self._targets.values())

        projected: list[ExecutionTargetDescriptor] = []
        for registration in registrations:
            descriptor = registration.descriptor
            if (
                not context.matches_tenant(descriptor.tenant_id)
                or not context.allows_boundary(descriptor.boundary_id)
            ):
                continue
            capabilities = frozenset(
                capability
                for capability in registration.entitled_capabilities
                if context.allows_capability(capability)
            )
            if not capabilities:
                continue
            projected.append(
                ExecutionTargetDescriptor(
                    target_id=descriptor.target_id,
                    target_kind=descriptor.target_kind,
                    display_name=descriptor.display_name,
                    tenant_id=descriptor.tenant_id,
                    boundary_id=descriptor.boundary_id,
                    capabilities=capabilities,
                )
            )
        projected.sort(
            key=lambda item: (item.display_name, item.target_kind, item.target_id)
        )
        return tuple(projected)

    def project_execution_targets(
        self, context: OperatorContext
    ) -> tuple[dict[str, object], ...]:
        """Return the complete browser wire projection without opaque state."""

        return tuple(
            descriptor.to_dict()
            for descriptor in self.list_execution_target_descriptors(context)
        )

    def _registration_set_is_exact(
        self, registration_set: OperatorRegistrationSet
    ) -> bool:
        for registration in registration_set.services:
            active = self._services.get(registration.reference)
            if active is not registration or active.service is not registration.service:
                return False
        for registration in registration_set.workflows:
            active = self._workflows.get(registration.name)
            if active is not registration or active.actor is not registration.actor:
                return False
        for registration in registration_set.execution_targets:
            active = self._targets.get(
                (registration.descriptor.tenant_id, registration.descriptor.target_id)
            )
            if active is not registration or active.handle is not registration.handle:
                return False
        return True

    def _validate_registration_set_locked(
        self, registration_set: OperatorRegistrationSet
    ) -> OperatorRegistrationSet:
        """Return the exact active set while ``self._lock`` is held."""

        active = self._active_sets.get(id(registration_set))
        if active is not registration_set:
            raise OperatorRegistrationIdentityError(
                "operator registration set is not active"
            )
        if not self._registration_set_is_exact(active):
            raise OperatorRegistrationIdentityError(
                "active operator registration identity does not match"
            )
        return active

    def _trusted_now(self) -> datetime:
        instant = self._clock()
        if not isinstance(instant, datetime):
            raise TypeError("clock must return a datetime")
        if instant.tzinfo is None or instant.utcoffset() is None:
            raise ValueError("clock must return a timezone-aware datetime")
        return instant

    @staticmethod
    def _require_context(context: OperatorContext) -> None:
        if not isinstance(context, OperatorContext):
            raise TypeError("context must be an OperatorContext")

    @staticmethod
    def _typed_tuple(
        values: Iterable[object], expected: type[object], field_name: str
    ) -> tuple:
        if isinstance(values, (str, bytes)):
            raise TypeError(f"{field_name} must contain {expected.__name__} values")
        try:
            result = tuple(values)
        except TypeError as error:
            raise TypeError(
                f"{field_name} must contain {expected.__name__} values"
            ) from error
        if not all(isinstance(item, expected) for item in result):
            raise TypeError(f"{field_name} must contain {expected.__name__} values")
        return result

    @staticmethod
    def _require_unique(values: tuple[object, ...], description: str) -> None:
        if len(set(values)) != len(values):
            raise OperatorRegistrationConflictError(
                f"duplicate {description} in registration batch"
            )

    @staticmethod
    def _stable_version_key(version: str) -> tuple[int, int, int]:
        major, minor, patch = version.split(".")
        return (int(major), int(minor), int(patch))


def _validate_owner(owner: object) -> None:
    """Reuse the SDK's canonical lifecycle-owner token validation."""

    validate_contribution_owner_uniqueness((owner,))


__all__ = [
    "ExecutionTargetRegistration",
    "ExecutionTargetUnavailableError",
    "OperatorRegistrationConflictError",
    "OperatorRegistrationError",
    "OperatorRegistrationIdentityError",
    "OperatorRegistrationSet",
    "OperatorRuntimeRegistry",
]
