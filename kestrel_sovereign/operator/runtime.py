"""Lifecycle-owned registries for generic SDK operator contracts.

This module is deliberately an in-memory lifecycle boundary, not a workflow
engine.  It owns active services, workflow actors, and opaque execution target
handles while their contributing lifecycle is active.  Durable runs,
artifacts, HTTP projection, and feature-specific behavior belong elsewhere.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
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
    ServiceScope,
)


class ServiceDiscoveryIdentity:
    """Opaque identity comparison for fresh retained discovery operations.

    This is metadata, never a credential or current checker. It is neither
    serialized nor persisted. An equal identity cannot authorize a service
    operation, and a closed operation's checker can never be revived by it.
    """

    __slots__ = ("_parts",)

    def __new__(cls, *args, **kwargs):
        raise TypeError("discovery identities are issued by retained host discovery")

    @classmethod
    def _from_registry(cls, registry, registration, issued_set, seal):
        value = object.__new__(cls)
        object.__setattr__(
            value, "_parts", (registry, registration, issued_set, seal, None, None, None, None)
        )
        return value

    def _with_host_binding(self, binding, generation, publication, publication_generation):
        if type(generation) is not int or (
            publication_generation is not None and type(publication_generation) is not int
        ):
            raise OperatorAuthorizationError("host discovery generation unavailable")
        value = object.__new__(type(self))
        object.__setattr__(
            value,
            "_parts",
            (*self._parts[:4], binding, generation, publication, publication_generation),
        )
        return value

    def matches(self, other):
        """Compare only trusted exact identities and plain host generations.

        A match supplies no authority. Consumers MUST separately retain the
        fresh operation's checker and their independent capability policy.
        """
        if (
            type(self) is not ServiceDiscoveryIdentity
            or type(other) is not ServiceDiscoveryIdentity
        ):
            return False
        first, second = getattr(self, "_parts", ()), getattr(other, "_parts", ())
        if (
            type(first) is not tuple
            or type(second) is not tuple
            or len(first) != 8
            or len(second) != 8
        ):
            return False
        if any(
            value is not None and type(value) is not int
            for value in (first[5], first[7], second[5], second[7])
        ):
            return False
        return (
            all(first[index] is second[index] for index in (0, 1, 2, 3, 4, 6))
            and first[5] == second[5]
            and first[7] == second[7]
        )

    def __setattr__(self, name, value):
        raise AttributeError("discovery identity is immutable")

    def __repr__(self):
        return "<opaque host service discovery identity>"

    def __reduce_ex__(self, protocol):
        raise TypeError("discovery identity is not serializable")


class _ServiceOperationIssuer:
    """Lexical issuance provenance inside the existing discovery owner."""
    __slots__ = ("operation", "service", "identity", "current")

    def __init__(self, operation, current):
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "service", operation.service)
        object.__setattr__(self, "identity", operation.discovery_identity)
        object.__setattr__(self, "current", current)

    def __setattr__(self, name, value):
        raise AttributeError("host operation issuance is immutable")

    def __repr__(self):
        return "<opaque original host operation issuance>"

    def __reduce_ex__(self, protocol):
        raise TypeError("host operation issuance is not serializable")


@dataclass(frozen=True, slots=True)
class ServiceOperation:
    """Exact retained registration discovery, never a capability grant.

    Consumers must retain their independent original run/content policies.
    The checker is synchronous and valid only inside the original frame.
    """
    service: object
    check_current: Callable[[], None]
    discovery_identity: ServiceDiscoveryIdentity | None = None
    _issuer: _ServiceOperationIssuer | None = field(default=None, init=False, repr=False, compare=False)

    @classmethod
    def _issue(cls, service, current, discovery_identity):
        operation = cls(service, current, discovery_identity)
        object.__setattr__(operation, "_issuer", _ServiceOperationIssuer(operation, current))
        return operation

    def require_issued(self):
        """Check this exact original record and its still-live lexical owner.

        This verifies discovery provenance only. It supplies no run/content
        policy. Ordinary API1 two-field descriptions keep their existing
        check_current behavior, but cannot satisfy this new provenance seam.
        Trusted Python extensions remain trusted; this is not a Python sandbox.
        """
        issuer = self._issuer
        if (type(issuer) is not _ServiceOperationIssuer or issuer.operation is not self
                or issuer.service is not self.service or issuer.identity is not self.discovery_identity
                or issuer.current is not self.check_current):
            raise OperatorAuthorizationError("original issued host discovery is unavailable")
        if issuer.current() is not None:
            raise OperatorAuthorizationError("host discovery checker must return None or raise")


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
    _registry_seal: object | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )


class OperatorRuntimeRegistry:
    """Authoritative active registry for generic operator integrations.

    Registration batches are validated in full before any map changes.  The
    returned :class:`OperatorRegistrationSet` is the teardown capability:
    unregistering requires that exact set, each exact registration object, and
    each registration's original opaque implementation object.  Registrations
    are never represented as an ownership stack, so removing one lifecycle
    cannot reveal a superseded inactive implementation.
    """

    host_service_identity_api_version = 1

    def __init__(self, *, clock: Callable[[], datetime] | None = None) -> None:
        self._clock = clock or (lambda: datetime.now(UTC))
        self._lock = RLock()
        self._services: dict[ServiceReference, ServiceRegistration] = {}
        self._workflows: dict[str, WorkflowRegistration] = {}
        self._targets: dict[
            tuple[str, str], ExecutionTargetRegistration
        ] = {}
        self._active_sets: dict[int, OperatorRegistrationSet] = {}
        # Independent issuance provenance. This survives active-set index drift
        # while binding a unique seal to one exact returned set. A caller cannot
        # reuse its own legitimate seal on a wrapper around another owner's
        # publicly projected registration.
        self._issued_set_seals: dict[
            int, tuple[OperatorRegistrationSet, object]
        ] = {}

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

            seal = object()
            object.__setattr__(registration_set, "_registry_seal", seal)
            self._services.update(zip(service_keys, service_values, strict=True))
            self._workflows.update(zip(workflow_keys, workflow_values, strict=True))
            self._targets.update(zip(target_keys, target_values, strict=True))
            self._active_sets[id(registration_set)] = registration_set
            self._issued_set_seals[id(registration_set)] = (
                registration_set,
                seal,
            )
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
            del self._issued_set_seals[id(active)]
            object.__setattr__(active, "_registry_seal", None)

    def quarantine_registration_set(
        self, registration_set: OperatorRegistrationSet
    ) -> bool:
        """Withdraw exact survivors from one drifted active registration set.

        Ordinary :meth:`unregister` is an atomic exact inverse and therefore
        refuses a partially missing set. Lifecycle recovery has a different
        job: remove each retained object that is still exactly resident,
        tolerate an already-absent object or set ledger, preserve any
        replacement, and retire the original set capability so a complete
        generation can be prepared.
        """

        if not isinstance(registration_set, OperatorRegistrationSet):
            raise TypeError("registration_set must be an OperatorRegistrationSet")
        with self._lock:
            issued = self._issued_set_seals.get(id(registration_set))
            if (
                issued is None
                or issued[0] is not registration_set
                or issued[1] is not registration_set._registry_seal
            ):
                return False
            active = self._active_sets.get(id(registration_set))
            had_active_set = active is registration_set
            if active is not None and not had_active_set:
                return False
            if active is None:
                active = registration_set
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
            if had_active_set:
                del self._active_sets[id(active)]
            del self._issued_set_seals[id(active)]
            object.__setattr__(active, "_registry_seal", None)
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
                for registration in self._services.values()
                if service_registration_satisfies(registration, requirement)
            ]
            if not candidates:
                return None
            selected = max(
                candidates,
                key=lambda item: self._stable_version_key(item.descriptor.version),
            )
            return selected.service

    @asynccontextmanager
    async def host_service_operation(self, requirement: ServiceRequirement):
        """Retain one exact HOST registration epoch, including ABA removal.

        Lookup IDs are discovery only. No OperatorContext is created here.
        """
        if not isinstance(requirement, ServiceRequirement) or requirement.scope is not ServiceScope.HOST:
            raise OperatorAuthorizationError("explicit host service scope required")
        with self._lock:
            candidates = [registration for registration in self._services.values()
                          if service_registration_satisfies(registration, requirement)]
            if not candidates:
                raise OperatorAuthorizationError("host service unavailable")
            registration = max(candidates, key=lambda item: self._stable_version_key(item.descriptor.version))
            original = next((value for value in self._active_sets.values()
                             if any(item is registration for item in value.services)), None)
            seal = None if original is None else original._registry_seal
            if original is None or seal is None:
                raise OperatorAuthorizationError("host registration unavailable")
        active = True
        def check():
            with self._lock:
                if (not active or self._services.get(registration.reference) is not registration
                        or self._active_sets.get(id(original)) is not original
                        or original._registry_seal is not seal
                        or self._issued_set_seals.get(id(original)) != (original, seal)):
                    raise OperatorAuthorizationError("original host registration ended")
        try:
            check()
            identity = ServiceDiscoveryIdentity._from_registry(self, registration, original, seal)
            yield ServiceOperation._issue(registration.service, check, identity)
            check()
        finally:
            active = False

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
        issued = self._issued_set_seals.get(id(registration_set))
        if (
            active is not registration_set
            or issued is None
            or issued[0] is not registration_set
            or issued[1] is not registration_set._registry_seal
        ):
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


def service_registration_satisfies(
    registration: ServiceRegistration, requirement: ServiceRequirement
) -> bool:
    """Whether ``registration`` is a stable compatible candidate for ``requirement``.

    The single compatibility rule: the registry resolves services by it, and
    host-feature start order (#3501) puts a consumer after exactly the
    providers it would resolve. Host and agent namespaces never cross.
    """

    return registration.agent_id == requirement.agent_id and requirement.accepts(
        registration.descriptor
    )


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
    "service_registration_satisfies",
]
