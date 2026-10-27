"""Distinct SDK environment and driver factory discovery; no host fallback."""

from importlib.metadata import entry_points
from collections.abc import Mapping

from kestrel_sdk.computer_use import (
    COMPUTER_DRIVER_ENTRY_POINT_GROUP, ENVIRONMENT_PROVIDER_ENTRY_POINT_GROUP,
    ComputerDriver, EnvironmentProvider,
)
from kestrel_sdk.resources import ResourceConflict, ResourceNotFound


def discover_factories(group: str) -> dict:
    if group not in {COMPUTER_DRIVER_ENTRY_POINT_GROUP, ENVIRONMENT_PROVIDER_ENTRY_POINT_GROUP}:
        raise ValueError("not a visual computer factory group")
    factories = {}
    for point in entry_points(group=group):
        if point.name in factories:
            raise ResourceConflict("duplicate visual computer factory")
        factory = point.load()
        if not callable(factory):
            raise TypeError("visual computer entry point is not a factory")
        factories[point.name] = factory
    return factories


def construct_adapters(*, providers: Mapping, drivers: Mapping) -> tuple[dict, dict]:
    """Only trusted host settings enter factories; HTTP input cannot select them."""
    def construct(group, settings, protocol):
        factories = discover_factories(group)
        result = {}
        for name, private_settings in settings.items():
            if name not in factories:
                raise ResourceNotFound("configured visual computer adapter is unavailable")
            if not isinstance(private_settings, Mapping):
                raise TypeError("private visual adapter settings must be a mapping")
            adapter = factories[name](settings=private_settings)
            if not isinstance(adapter, protocol):
                raise TypeError("visual adapter does not implement its SDK protocol")
            result[name] = adapter
        return result
    return (construct(ENVIRONMENT_PROVIDER_ENTRY_POINT_GROUP, providers, EnvironmentProvider),
            construct(COMPUTER_DRIVER_ENTRY_POINT_GROUP, drivers, ComputerDriver))
