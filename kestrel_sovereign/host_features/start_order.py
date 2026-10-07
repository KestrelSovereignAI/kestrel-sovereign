"""Start host features provider-before-consumer (issue #3501).

A host feature provides a service through ``get_service_registrations()`` and
consumes one by resolving a ``ServiceRequirement`` through the host operator
registry. Host features start one at a time, so a consumer whose start hook
launches workers would otherwise race every provider still waiting its turn:
the provider's registration is already active, but its own ``on_host_start``
has not yet made the service ready. Talon, Eye and Flight hit exactly that
against Workflows' run service on every boot.

The services a host feature consumes are the ``ServiceRequirement`` values it
holds once constructed -- the same objects it later hands to
``resolve_compatible_service``. They are found in the feature's instance state:
attributes of the feature itself, of objects whose class comes from the
feature's own top-level package, and of the built-in containers among them.
Nothing else is entered, so a router, lock or third-party client is never
walked. Attributes are read from instance ``__dict__`` and ``__slots__``
storage, never through ``getattr``, so no property, custom descriptor or
``__getattr__`` of the feature runs. A requirement built only inside a start
hook, or held only at module level, is not visible here.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from types import MemberDescriptorType
from typing import TypeVar

from kestrel_sdk.operator import ServiceRequirement, ServiceResolver

from kestrel_sovereign.features.contribution_runtime import (
    PreparedFeatureContributions,
)
from kestrel_sovereign.operator import service_registration_satisfies

logger = logging.getLogger(__name__)

#: Most objects one feature's requirement scan visits. Only objects that can
#: hold a requirement count, so this bounds a pathological graph rather than an
#: ordinary feature.
REQUIREMENT_SCAN_BUDGET = 4096

_CONTAINERS = (tuple, list, set, frozenset)

_Feature = TypeVar("_Feature")


@dataclass(frozen=True, slots=True)
class RequirementScan:
    """The ``ServiceRequirement`` values one host feature holds.

    ``complete`` is false when the scan stopped at
    :data:`REQUIREMENT_SCAN_BUDGET`; ``requirements`` is then what it found
    before stopping.
    """

    requirements: tuple[ServiceRequirement, ...]
    complete: bool


def scan_service_requirements(feature: object) -> RequirementScan:
    """Find the ``ServiceRequirement`` values ``feature`` holds, in scan order."""

    package = _package_of(type(feature))
    found: dict[ServiceRequirement, None] = {}
    seen: set[int] = set()
    pending: list[object] = [feature]
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        if len(seen) >= REQUIREMENT_SCAN_BUDGET:
            return RequirementScan(tuple(found), complete=False)
        seen.add(id(value))
        if issubclass(type(value), ServiceRequirement):
            found[value] = None
            continue
        pending.extend(
            item for item in _members(value) if _may_hold_requirement(item, package)
        )
    return RequirementScan(tuple(found), complete=True)


def order_host_features_for_start(
    pairs: Sequence[tuple[_Feature, PreparedFeatureContributions]],
    active_services: ServiceResolver,
) -> tuple[tuple[_Feature, PreparedFeatureContributions], ...]:
    """Order activatable ``(feature, prepared)`` pairs provider-before-consumer.

    A provider is placed immediately before its first consumer; every other
    feature keeps its discovery order, so features with no dependency between
    them start exactly as discovered. A requirement satisfied by no feature in
    ``pairs`` and by no already-active service is logged and adds no ordering
    constraint. Features in a dependency cycle are logged and start in
    discovery order relative to each other, still after the providers outside
    the cycle that they consume. Nothing is ever dropped.
    """

    count = len(pairs)
    names = [prepared.feature_name for _, prepared in pairs]
    providers: list[set[int]] = [set() for _ in range(count)]
    via: dict[tuple[int, int], set[str]] = {}
    for consumer, (feature, _) in enumerate(pairs):
        scan = scan_service_requirements(feature)
        if not scan.complete:
            logger.warning(
                "Host feature %s requirement scan stopped after %d objects; "
                "its start order uses the %d service requirements found so far",
                names[consumer],
                REQUIREMENT_SCAN_BUDGET,
                len(scan.requirements),
            )
        for requirement in scan.requirements:
            matches = {
                provider
                for provider, (_, prepared) in enumerate(pairs)
                if any(
                    service_registration_satisfies(registration, requirement)
                    for registration in prepared.contributions.services
                )
            }
            self_provided = consumer in matches
            matches.discard(consumer)
            for provider in matches:
                providers[consumer].add(provider)
                via.setdefault((consumer, provider), set()).add(requirement.name)
            if (
                not matches
                and not self_provided
                and active_services.resolve_compatible_service(requirement) is None
            ):
                logger.warning(
                    "Host feature %s requires service %s >= %s, which no host "
                    "feature provides; it is not ordered after any provider",
                    names[consumer],
                    requirement.name,
                    requirement.minimum_version,
                )

    reachable = [_reachable(index, providers) for index in range(count)]
    component_of: dict[int, int] = {}
    components: list[list[int]] = []
    for index in range(count):
        if index in component_of:
            continue
        members = [
            other
            for other in range(count)
            if other == index
            or (other in reachable[index] and index in reachable[other])
        ]
        for member in members:
            component_of[member] = len(components)
        components.append(members)
        if len(members) > 1:
            edges = "; ".join(
                f"{names[consumer]} requires "
                f"{', '.join(sorted(via[(consumer, provider)]))} from {names[provider]}"
                for consumer in members
                for provider in sorted(providers[consumer])
                if provider in members
            )
            logger.error(
                "Host features %s form a service dependency cycle (%s); they "
                "start in discovery order",
                ", ".join(names[member] for member in members),
                edges,
            )

    order: list[int] = []
    placed: set[int] = set()

    def place(component: int) -> None:
        # The components form a DAG, so marking before recursing cannot hide
        # an unplaced provider.
        placed.add(component)
        upstream = {
            component_of[provider]
            for member in components[component]
            for provider in providers[member]
        } - {component}
        for provider_component in sorted(upstream, key=lambda c: components[c][0]):
            if provider_component not in placed:
                place(provider_component)
        order.extend(components[component])

    for index in range(count):
        if component_of[index] not in placed:
            place(component_of[index])
    return tuple(pairs[index] for index in order)


def _reachable(start: int, providers: list[set[int]]) -> set[int]:
    """Every feature ``start`` transitively consumes from."""

    reached: set[int] = set()
    pending = list(providers[start])
    while pending:
        index = pending.pop()
        if index not in reached:
            reached.add(index)
            pending.extend(providers[index])
    return reached


def _members(value: object) -> Iterator[object]:
    """Values ``value`` holds, read without running any of its own code.

    Types come from ``type()``, never ``isinstance``: a test double's
    ``__class__`` can claim to be a container or a requirement it is not.
    """

    kind = type(value)
    if issubclass(kind, dict):
        yield from dict.values(value)
        return
    for container in _CONTAINERS:
        if issubclass(kind, container):
            yield from container.__iter__(value)
            return
    try:
        attributes = object.__getattribute__(value, "__dict__")
    except AttributeError:
        attributes = {}
    if isinstance(attributes, dict):
        yield from attributes.values()
    for cls in kind.__mro__:
        for member in vars(cls).values():
            if isinstance(member, MemberDescriptorType):
                try:
                    yield member.__get__(value, cls)
                except AttributeError:
                    continue


def _may_hold_requirement(value: object, package: str) -> bool:
    kind = type(value)
    return (
        issubclass(kind, (ServiceRequirement, dict, *_CONTAINERS))
        or _package_of(kind) == package
    )


def _package_of(cls: type) -> str:
    return cls.__module__.partition(".")[0]


__all__ = [
    "REQUIREMENT_SCAN_BUDGET",
    "RequirementScan",
    "order_host_features_for_start",
    "scan_service_requirements",
]
