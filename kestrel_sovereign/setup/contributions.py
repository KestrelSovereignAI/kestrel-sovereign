"""Pre-boot discovery for SDK setup-step contributions.

Setup runs before an agent exists, so it cannot discover declarations by
constructing an SDK ``Feature`` with a placeholder agent.  Packages instead
publish a small, setup-only entry point whose value is either a
``SetupStepRegistration``, a tuple of registrations, or a zero-argument
provider returning that tuple::

    [project.entry-points."kestrel_sovereign.setup_steps"]
    voice = "kestrel_feature_voice.setup:get_setup_step_registrations"

The provider module is deliberately separate from the feature implementation;
loading setup metadata must not boot or instantiate a feature.
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from kestrel_sdk.features import (
    ContributionContractError,
    SetupFlow,
    SetupStepClassification,
    SetupStepRegistration,
    await_contribution_result,
)

from kestrel_sovereign.features.contribution_runtime import SetupStepRegistry
from kestrel_sovereign.setup.context import Flow, SetupContext


SETUP_STEP_ENTRY_POINT_GROUP = "kestrel_sovereign.setup_steps"


class SetupContributionDiscoveryError(RuntimeError):
    """Installed setup declarations are invalid or cannot be loaded."""


@dataclass(frozen=True, slots=True)
class DiscoveredSetupSteps:
    """Complete validated registry plus package-slug selection aliases."""

    registry: SetupStepRegistry
    aliases: dict[str, tuple[str, ...]]

    def selected(self, name_or_slug: str) -> tuple[SetupStepRegistration, ...]:
        registration = self.registry.get(name_or_slug)
        if registration is not None:
            return (registration,)
        names = self.aliases.get(name_or_slug)
        if names is None:
            return ()
        selected_names = set(names)
        return tuple(
            registration
            for registration in self.registry.ordered()
            if registration.name in selected_names
        )


class ContributedSetupContext:
    """SDK-stable view over Sovereign's richer historic wizard context."""

    def __init__(self, context: SetupContext) -> None:
        self._context = context

    @property
    def project_dir(self) -> Path:
        return self._context.project_dir

    @property
    def agent_data_root(self) -> Path:
        return self._context.agent_data_root

    @property
    def flow(self) -> SetupFlow:
        return (
            SetupFlow.CHECK
            if self._context.flow is Flow.CHECK
            else SetupFlow.SETUP
        )

    @property
    def prompter(self) -> Any:
        return self._context.prompter

    def record(self, message: str) -> None:
        self._context.record(message)

    def block(self, message: str) -> None:
        self._context.block(message)


def discover_core_setup_steps() -> DiscoveredSetupSteps:
    """Return only built-in recovery steps without inspecting plugin metadata."""

    return DiscoveredSetupSteps(registry=SetupStepRegistry(), aliases={})


def discover_setup_steps(
    entry_points: Iterable[object] | None = None,
) -> DiscoveredSetupSteps:
    """Discover and validate all built-in and installed setup steps.

    ``entry_points`` is an explicit test/embedder seam.  Production callers
    omit it and read only the dedicated setup-step entry-point group.
    """
    if entry_points is None:
        try:
            available = importlib.metadata.entry_points()
        except Exception as exc:  # noqa: BLE001 - dependency discovery boundary
            raise SetupContributionDiscoveryError(
                f"could not read installed setup-step entry points: {exc}"
            ) from exc
        if hasattr(available, "select"):
            entry_points = available.select(group=SETUP_STEP_ENTRY_POINT_GROUP)
        else:
            entry_points = available.get(SETUP_STEP_ENTRY_POINT_GROUP, [])

    registry = SetupStepRegistry()
    aliases: dict[str, tuple[str, ...]] = {}
    registrations: list[SetupStepRegistration] = []
    for entry_point in sorted(
        tuple(entry_points),
        key=lambda item: (
            str(getattr(item, "name", "")),
            str(getattr(item, "value", "")),
        ),
    ):
        slug = str(getattr(entry_point, "name", "")).strip()
        if not slug:
            raise SetupContributionDiscoveryError(
                "installed setup-step entry point has an empty slug"
            )
        if slug in aliases:
            raise SetupContributionDiscoveryError(
                f"duplicate setup-step entry-point slug: {slug}"
            )
        try:
            loaded = entry_point.load()
            declared = loaded() if callable(loaded) else loaded
        except Exception as exc:  # noqa: BLE001 - third-party plugin boundary
            raise SetupContributionDiscoveryError(
                f"could not load contributed setup steps for {slug!r}: {exc}"
            ) from exc
        if inspect.isawaitable(declared):
            if inspect.iscoroutine(declared):
                declared.close()
            raise SetupContributionDiscoveryError(
                f"setup-step provider {slug!r} must return declarations "
                "synchronously"
            )
        values = (
            (declared,)
            if isinstance(declared, SetupStepRegistration)
            else declared
        )
        if not isinstance(values, tuple) or not all(
            isinstance(value, SetupStepRegistration) for value in values
        ):
            raise SetupContributionDiscoveryError(
                f"setup-step provider {slug!r} must return a "
                "SetupStepRegistration or tuple of registrations"
            )
        aliases[slug] = tuple(value.name for value in values)
        registrations.extend(values)

    try:
        registry.register_batch(tuple(registrations))
        ordered = registry.ordered()
        keys_index = next(
            (
                index
                for index, registration in enumerate(ordered)
                if registration.owner == "core:setup" and registration.name == "keys"
            ),
            None,
        )
        early_default = next(
            (
                registration
                for registration in (
                    () if keys_index is None else ordered[:keys_index]
                )
                if registration.owner != "core:setup"
                and registration.classification is SetupStepClassification.DEFAULT
            ),
            None,
        )
        if early_default is not None:
            raise ContributionContractError(
                f"contributed DEFAULT setup step {early_default.name!r} orders "
                "before the core 'keys' custody boundary"
            )
    except (ContributionContractError, ValueError, TypeError, RuntimeError) as exc:
        raise SetupContributionDiscoveryError(
            f"invalid installed setup-step contributions: {exc}"
        ) from exc
    return DiscoveredSetupSteps(registry=registry, aliases=aliases)


def run_setup_step(
    registration: SetupStepRegistration,
    context: SetupContext,
) -> object:
    """Execute a sync or async registration from the synchronous CLI."""
    step_context: object = (
        context
        if registration.owner == "core:setup"
        else ContributedSetupContext(context)
    )
    result = registration.step(step_context)
    if inspect.isawaitable(result):
        return asyncio.run(await_contribution_result(result))
    return result


def missing_setup_step_message(
    name_or_slug: str,
    discovered: DiscoveredSetupSteps,
) -> str:
    """Return an actionable error without importing an absent package."""
    from kestrel_sovereign.feature_registry import load_registry

    package_info = load_registry().get(name_or_slug)
    if package_info is not None and package_info.installable:
        try:
            importlib.metadata.distribution(package_info.package)
        except importlib.metadata.PackageNotFoundError:
            return (
                f"Setup step/feature slug {name_or_slug!r} requires missing "
                f"package {package_info.package!r}. Install it with "
                f"`kestrel feature install {name_or_slug}`, then retry."
            )
        return (
            f"Installed package {package_info.package!r} does not contribute "
            f"setup step/slug {name_or_slug!r} via "
            f"{SETUP_STEP_ENTRY_POINT_GROUP!r}. Upgrade or repair that package."
        )

    valid = sorted(
        {
            *(registration.name for registration in discovered.registry.ordered()),
            *discovered.aliases,
        }
    )
    return (
        f"Unknown setup step/feature slug: {name_or_slug!r}. "
        f"Valid installed choices: {', '.join(valid)}"
    )


__all__ = [
    "ContributedSetupContext",
    "DiscoveredSetupSteps",
    "SETUP_STEP_ENTRY_POINT_GROUP",
    "SetupContributionDiscoveryError",
    "discover_core_setup_steps",
    "discover_setup_steps",
    "missing_setup_step_message",
    "run_setup_step",
]
