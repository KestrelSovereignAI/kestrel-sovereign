"""Agent-facing FeatureFeature surface (#1151)."""

from __future__ import annotations

import importlib
from typing import Any

from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult, ToolResultStatus

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
    FEATURE_PROPOSE_PACKAGE_WORKFLOW_NAME,
    FEATURE_PROPOSE_TOOL_WORKFLOW_NAME,
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

    @tool(
        name="feature_feature_define_workflows",
        description=(
            "Define and sign FeatureFeature workflow specs through WorkflowsFeature."
        ),
        category=ToolCategory.SYSTEM,
        command_prefix="!feature-define-workflows",
    )
    async def feature_feature_define_workflows(
        self,
        kind: str = "all",
        repository: str = DEFAULT_REPOSITORY,
        branch: str = DEFAULT_BRANCH,
        prompt_pack_constraint: str = DEFAULT_PROMPT_PACK_CONSTRAINT,
    ) -> ToolResult:
        """Register FeatureFeature workflow definitions.

        Args:
            kind: all, tool, or package.
            repository: GitHub repository used by the ci_green gate.
            branch: Git branch used by the ci_green gate.
            prompt_pack_constraint: Red-team prompt package version range.
        """

        workflow_feature = self._workflow_feature()
        if workflow_feature is None:
            return ToolResult.failed("WorkflowsFeature is not available")
        try:
            payloads = feature_feature_workflow_payloads(
                kind=kind,  # type: ignore[arg-type]
                repository=repository,
                branch=branch,
                prompt_pack_constraint=prompt_pack_constraint,
            )
        except ValueError as exc:
            return ToolResult.failed(str(exc))

        defined: list[dict[str, Any]] = []
        for name, payload in payloads.items():
            result = await workflow_feature.workflow_define(payload)
            defined.append(
                {
                    "name": name,
                    "status": result.status.value,
                    "data": result.data,
                    "error": result.error,
                }
            )
            if result.status is not ToolResultStatus.OK:
                return ToolResult.failed(
                    f"Failed to define FeatureFeature workflow {name!r}: "
                    f"{result.error}",
                    data={"defined": defined},
                )

        return ToolResult.ok(
            f"Defined {len(defined)} FeatureFeature workflow definition(s).",
            data={"defined": defined},
        )

    @tool(
        name="feature_feature_run",
        description="Run a FeatureFeature workflow through WorkflowsFeature.",
        category=ToolCategory.SYSTEM,
        command_prefix="!feature-run",
    )
    async def feature_feature_run(
        self,
        kind: str,
        params: dict,
        version: int = 0,
    ) -> ToolResult:
        """Start a FeatureFeature workflow run.

        Args:
            kind: tool or package.
            params: Workflow run parameters.
            version: Specific workflow definition version, or latest active.
        """

        workflow_feature = self._workflow_feature()
        if workflow_feature is None:
            return ToolResult.failed("WorkflowsFeature is not available")
        if not isinstance(params, dict):
            return ToolResult.failed("FeatureFeature run params must be an object")

        try:
            workflow_name = _workflow_name_for_kind(kind)
        except ValueError as exc:
            return ToolResult.failed(str(exc))

        return await workflow_feature.workflow_run(
            workflow_name,
            params=params,
            version=version,
        )

    @tool(
        name="feature_feature_runtime_status",
        description="Check FeatureFeature workflow signal-source readiness.",
        category=ToolCategory.SYSTEM,
        command_prefix="!feature-runtime-status",
    )
    async def feature_feature_runtime_status(self) -> ToolResult:
        """Report whether FeatureFeature workflow sources are registered.

        ACTION sources also report whether a callable provider is available.
        """

        registry = getattr(self.agent, "signal_registry", None)
        rows = []
        missing_registered: list[str] = []
        missing_action_providers: list[str] = []
        missing_cognition_prompts: list[str] = []
        for registration in build_feature_feature_registrations(self.agent):
            actual_registration = (
                registry.get(registration.name) if registry is not None else None
            )
            active_registration = actual_registration or registration
            registered = actual_registration is not None
            if not registered:
                missing_registered.append(registration.name)
            mode = active_registration.default_mode.value
            provider_name = _resolve_provider_name(self.agent, registration.name)
            registered_handler_ready = _registered_handler_ready(
                actual_registration
            )
            action_provider_ready = (
                mode != "action"
                or registered_handler_ready
                or provider_name is not None
            )
            prompt_ready = (
                mode != "cognition"
                or (
                    active_registration.prompt_template is not None
                    and active_registration.prompt_template.exists()
                )
            )
            if mode == "action" and not action_provider_ready:
                missing_action_providers.append(registration.name)
            if mode == "cognition" and not prompt_ready:
                missing_cognition_prompts.append(registration.name)
            rows.append(
                {
                    "name": registration.name,
                    "mode": mode,
                    "registered": registered,
                    "provider": provider_name,
                    "registered_handler": registered_handler_ready,
                    "prompt_template_exists": prompt_ready,
                    "ready": registered and action_provider_ready and prompt_ready,
                }
            )

        ok = (
            not missing_registered
            and not missing_action_providers
            and not missing_cognition_prompts
        )
        data = {
            "sources": rows,
            "missing_registered": missing_registered,
            "missing_action_providers": missing_action_providers,
            "missing_cognition_prompts": missing_cognition_prompts,
            "ready": ok,
        }
        if not ok:
            return ToolResult.failed(
                "FeatureFeature runtime is not ready.",
                data=data,
            )
        return ToolResult.ok("FeatureFeature runtime is ready.", data=data)

    def _workflow_feature(self) -> Any:
        features = getattr(self.agent, "features", None)
        if not isinstance(features, dict):
            return None
        workflow = features.get("WorkflowsFeature")
        if workflow is not None:
            return workflow
        for feature in features.values():
            if (
                feature.__class__.__name__ == "WorkflowsFeature"
                or (
                    hasattr(feature, "workflow_define")
                    and hasattr(feature, "workflow_run")
                )
            ):
                return feature
        return None


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


def _workflow_name_for_kind(kind: str) -> str:
    if kind == "tool":
        return FEATURE_PROPOSE_TOOL_WORKFLOW_NAME
    if kind == "package":
        return FEATURE_PROPOSE_PACKAGE_WORKFLOW_NAME
    raise ValueError("kind must be one of: tool, package")


def _resolve_provider_name(agent: Any, source: str) -> str | None:
    stage = source.split(".")[-1]
    for name in (
        f"feature_feature_{stage}",
        f"feature_feature_handle_{stage}",
    ):
        if callable(getattr(agent, name, None)):
            return name
    return None


def _registered_handler_ready(registration: Any) -> bool:
    if registration is None:
        return False
    handler = getattr(registration, "handler", None)
    return callable(handler) and not bool(
        getattr(handler, "_feature_feature_requires_agent_provider", False)
    )


__all__ = ["FeatureFeaturesFeature"]
