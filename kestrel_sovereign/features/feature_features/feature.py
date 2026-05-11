"""Agent-facing FeatureFeature surface (#1151)."""

from __future__ import annotations

import importlib
from typing import Any

from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult

from kestrel_sovereign.features import (
    discover_entrypoint_feature_classes,
    discover_feature_modules,
    find_feature_class,
)
from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.features.feature_features.workflows import (
    DEFAULT_BRANCH,
    DEFAULT_PROMPT_PACK_CONSTRAINT,
    DEFAULT_REPOSITORY,
    feature_feature_workflow_payloads,
)
from kestrel_sovereign.features.feature_features.signals import (
    build_feature_feature_registrations,
)


class FeatureFeaturesFeature(Feature):
    """Discover features and expose FeatureFeature workflow templates."""

    @property
    def tool_description(self) -> str:
        return (
            "Discover installed Kestrel features and return signed-workflow "
            "templates for proposing new tools or feature packages."
        )

    async def initialize(self) -> None:
        self._register_signal_sources()
        return None

    def _register_signal_sources(self) -> None:
        registry = getattr(self.agent, "signal_registry", None)
        if registry is None:
            return
        for registration in build_feature_feature_registrations(self.agent):
            if registry.get(registration.name) is not None:
                continue
            registry.register(registration)

    @tool(
        name="feature_explore",
        description="List discoverable feature modules and feature classes.",
        category=ToolCategory.UTILITY,
        command_prefix="!feature-explore",
    )
    async def feature_explore(
        self,
        include_entrypoints: bool = False,
    ) -> ToolResult:
        """List installed core features and optionally entry-point features.

        Args:
            include_entrypoints: Include installed package entry points.
        """

        core = _core_feature_inventory()
        entrypoints: list[dict[str, str]] = []
        if include_entrypoints:
            entrypoints = [
                {"name": name, "class": cls.__name__, "module": cls.__module__}
                for name, cls in sorted(
                    discover_entrypoint_feature_classes().items()
                )
            ]
        return ToolResult.ok(
            f"Discovered {len(core)} core feature module(s).",
            data={
                "core_features": core,
                "entrypoint_features": entrypoints,
                "counts": {
                    "core": len(core),
                    "entrypoints": len(entrypoints),
                },
            },
        )

    @tool(
        name="feature_feature_workflows",
        description=(
            "Return unsigned FeatureFeature workflow specs for workflow_define."
        ),
        category=ToolCategory.SYSTEM,
        command_prefix="!feature-workflows",
    )
    async def feature_feature_workflows(
        self,
        kind: str = "all",
        repository: str = DEFAULT_REPOSITORY,
        branch: str = DEFAULT_BRANCH,
        prompt_pack_constraint: str = DEFAULT_PROMPT_PACK_CONSTRAINT,
    ) -> ToolResult:
        """Return FeatureFeature workflow specs.

        Args:
            kind: all, tool, or package.
            repository: GitHub repository used by the ci_green gate.
            branch: Git branch used by the ci_green gate.
            prompt_pack_constraint: Red-team prompt package version range.
        """

        try:
            payloads = feature_feature_workflow_payloads(
                kind=kind,  # type: ignore[arg-type]
                repository=repository,
                branch=branch,
                prompt_pack_constraint=prompt_pack_constraint,
            )
        except ValueError as exc:
            return ToolResult.failed(str(exc))

        return ToolResult.ok(
            f"Prepared {len(payloads)} FeatureFeature workflow spec(s).",
            data={"workflows": payloads},
        )


def _core_feature_inventory() -> list[dict[str, Any]]:
    rows = []
    for module_path in sorted(discover_feature_modules()):
        module = importlib.import_module(module_path)
        feature_class = find_feature_class(module)
        if feature_class is None:
            continue
        rows.append(
            {
                "module": module_path,
                "name": _feature_module_name(module_path),
                "class": feature_class.__name__,
            }
        )
    return rows


def _feature_module_name(module_path: str) -> str:
    if module_path.endswith(".feature"):
        return module_path.split(".")[-2]
    return module_path.split(".")[-1]


__all__ = ["FeatureFeaturesFeature"]
