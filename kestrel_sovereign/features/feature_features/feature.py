"""Agent-facing FeatureFeature surface (#1151)."""

from __future__ import annotations

import importlib
import json
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
        include_loaded_tools: bool = False,
    ) -> ToolResult:
        """List installed core features and optionally entry-point features.

        Args:
            include_entrypoints: Include installed package entry points.
            include_loaded_tools: Include active runtime features and tool metadata.
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
        loaded_features = []
        if include_loaded_tools:
            loaded_features = _loaded_feature_inventory(self.agent)
        return ToolResult.ok(
            f"Discovered {len(core)} core feature module(s).",
            data={
                "core_features": core,
                "entrypoint_features": entrypoints,
                "loaded_features": loaded_features,
                "counts": {
                    "core": len(core),
                    "entrypoints": len(entrypoints),
                    "loaded": len(loaded_features),
                },
            },
        )

    @tool(
        name="feature_context_status",
        description="Show which feature and direct tools are visible to the LLM.",
        category=ToolCategory.UTILITY,
        command_prefix="!feature-context-status",
    )
    async def feature_context_status(self) -> ToolResult:
        """Report the current context-visibility profile."""

        profile = _context_profile(self.agent)
        loaded_features = _loaded_feature_inventory(self.agent)
        direct_tools = _direct_tool_inventory(self.agent)
        visible_features = [
            row for row in loaded_features if row["visible_in_context"]
        ]
        visible_direct_tools = [
            row for row in direct_tools if row["visible_in_context"]
        ]
        return ToolResult.ok(
            (
                f"{len(visible_features)} feature dispatcher(s) and "
                f"{len(visible_direct_tools)} direct tool(s) visible."
            ),
            data={
                "profile": profile,
                "features": loaded_features,
                "direct_tools": direct_tools,
                "counts": {
                    "visible_features": len(visible_features),
                    "hidden_features": len(loaded_features) - len(visible_features),
                    "visible_direct_tools": len(visible_direct_tools),
                    "hidden_direct_tools": len(direct_tools)
                    - len(visible_direct_tools),
                },
            },
        )

    @tool(
        name="feature_focus",
        description=(
            "Persistently focus LLM context on selected features/tools without changing permissions."
        ),
        category=ToolCategory.SYSTEM,
        command_prefix="!feature-focus",
    )
    async def feature_focus(
        self,
        features: list[str] | None = None,
        tools: list[str] | None = None,
    ) -> ToolResult:
        """Hide non-selected feature dispatchers/direct tools until reset.

        Args:
            features: Feature class names or dispatcher tool names to keep visible.
            tools: Direct tool names to keep visible.
        """

        keep_features = _clean_name_set(features)
        keep_tools = _clean_name_set(tools)
        if not keep_features and not keep_tools:
            return ToolResult.failed("Provide at least one feature or tool to focus.")

        _ensure_context_profile(self.agent)
        loaded_features = _loaded_feature_inventory(self.agent)
        direct_tools = _direct_tool_inventory(self.agent)
        control_features = _control_feature_names(self.agent)
        selected_tool_features = {
            row["feature_tool_name"]
            for row in direct_tools
            if row["name"] in keep_tools and row["feature_tool_name"] is not None
        }

        hidden_features = {
            row["tool_name"]
            for row in loaded_features
            if row["tool_name"] not in control_features
            and row["class"] not in keep_features
            and row["tool_name"] not in keep_features
            and row["tool_name"] not in selected_tool_features
        }
        if keep_features:
            hidden_tools = {
                row["name"]
                for row in direct_tools
                if row["feature_tool_name"] not in keep_features
                and row["feature_class"] not in keep_features
                and row["name"] not in keep_tools
                and row["feature_tool_name"] not in control_features
            }
        else:
            hidden_tools = {
                row["name"]
                for row in direct_tools
                if row["name"] not in keep_tools
                and row["feature_tool_name"] not in control_features
            }

        setattr(self.agent, "_tool_context_hidden_features", hidden_features)
        setattr(self.agent, "_tool_context_hidden_tools", hidden_tools)
        _refresh_cached_features_prompt(self.agent)
        return await self.feature_context_status()

    @tool(
        name="feature_unfocus",
        description=(
            "Persistently hide selected features/tools from LLM context without denying access."
        ),
        category=ToolCategory.SYSTEM,
        command_prefix="!feature-unfocus",
    )
    async def feature_unfocus(
        self,
        features: list[str] | None = None,
        tools: list[str] | None = None,
        reset: bool = False,
    ) -> ToolResult:
        """Collapse selected tools out of context until reset.

        Args:
            features: Feature class names or dispatcher tool names to hide.
            tools: Direct tool names to hide.
            reset: Clear all context-hidden features/tools.
        """

        _ensure_context_profile(self.agent)
        reset_requested = _coerce_bool(reset)
        if reset_requested is None:
            return ToolResult.failed("reset must be a boolean.")
        if reset_requested:
            setattr(self.agent, "_tool_context_hidden_features", set())
            setattr(self.agent, "_tool_context_hidden_tools", set())
            _refresh_cached_features_prompt(self.agent)
            return await self.feature_context_status()

        hide_features = _clean_name_set(features)
        hide_tools = _clean_name_set(tools)
        if not hide_features and not hide_tools:
            return ToolResult.failed("Provide features/tools to hide, or reset=True.")

        control_features = _control_feature_names(self.agent)
        hidden_features = set(getattr(self.agent, "_tool_context_hidden_features"))
        for row in _loaded_feature_inventory(self.agent):
            if row["tool_name"] in control_features:
                continue
            if row["tool_name"] in hide_features or row["class"] in hide_features:
                hidden_features.add(row["tool_name"])
        hidden_tools = set(getattr(self.agent, "_tool_context_hidden_tools"))
        hidden_tools.update(hide_tools)
        setattr(self.agent, "_tool_context_hidden_features", hidden_features)
        setattr(self.agent, "_tool_context_hidden_tools", hidden_tools)
        _refresh_cached_features_prompt(self.agent)
        return await self.feature_context_status()

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


def _loaded_feature_inventory(agent: Any) -> list[dict[str, Any]]:
    rows = []
    hidden_features = _hidden_features(agent)
    for key, feature in sorted(_agent_features(agent).items()):
        tool_name = getattr(feature, "tool_name", key)
        class_name = feature.__class__.__name__
        tools = []
        for agent_tool in _safe_feature_tools(feature):
            tools.append(
                {
                    "name": agent_tool.name,
                    "description": agent_tool.schema.description,
                    "category": (
                        agent_tool.schema.category.value
                        if agent_tool.schema.category
                        else None
                    ),
                    "estimated_context_tokens": _estimate_tool_tokens(
                        agent_tool.schema.to_openai_format()
                    ),
                }
            )
        rows.append(
            {
                "registry_key": key,
                "class": class_name,
                "tool_name": tool_name,
                "module": feature.__class__.__module__,
                "tool_count": len(tools),
                "visible_in_context": (
                    tool_name not in hidden_features
                    and class_name not in hidden_features
                ),
                "tools": tools,
            }
        )
    return rows


def _direct_tool_inventory(agent: Any) -> list[dict[str, Any]]:
    hidden_tools = _hidden_tools(agent)
    hidden_features = _hidden_features(agent)
    direct_tools = getattr(agent, "_direct_tools", {})
    tool_to_feature = getattr(agent, "_tool_to_feature", {})
    if not isinstance(direct_tools, dict):
        return []
    rows = []
    features_by_tool_name = {
        getattr(feature, "tool_name", key): feature
        for key, feature in _agent_features(agent).items()
    }
    for name, agent_tool in sorted(direct_tools.items()):
        feature_tool_name = tool_to_feature.get(name)
        feature = features_by_tool_name.get(feature_tool_name)
        feature_class = feature.__class__.__name__ if feature is not None else None
        rows.append(
            {
                "name": name,
                "feature_tool_name": feature_tool_name,
                "feature_class": feature_class,
                "visible_in_context": (
                    name not in hidden_tools
                    and feature_tool_name not in hidden_features
                    and feature_class not in hidden_features
                ),
                "estimated_context_tokens": _estimate_tool_tokens(
                    agent_tool.schema.to_openai_format()
                ),
            }
        )
    return rows


def _agent_features(agent: Any) -> dict[str, Any]:
    features = getattr(agent, "features", None)
    return features if isinstance(features, dict) else {}


def _safe_feature_tools(feature: Any) -> list[Any]:
    get_tools = getattr(feature, "get_tools", None)
    if not callable(get_tools):
        return []
    try:
        return list(get_tools())
    except Exception:
        return []


def _context_profile(agent: Any) -> dict[str, list[str]]:
    _ensure_context_profile(agent)
    return {
        "hidden_features": sorted(_hidden_features(agent)),
        "hidden_tools": sorted(_hidden_tools(agent)),
    }


def _ensure_context_profile(agent: Any) -> None:
    if not isinstance(getattr(agent, "_tool_context_hidden_features", None), set):
        setattr(agent, "_tool_context_hidden_features", set())
    if not isinstance(getattr(agent, "_tool_context_hidden_tools", None), set):
        setattr(agent, "_tool_context_hidden_tools", set())


def _refresh_cached_features_prompt(agent: Any) -> None:
    build_prompt = getattr(agent, "_build_features_prompt_section", None)
    if callable(build_prompt) and hasattr(agent, "_cached_features_prompt"):
        setattr(agent, "_cached_features_prompt", build_prompt())


def _hidden_features(agent: Any) -> set[str]:
    _ensure_context_profile(agent)
    return {str(item) for item in getattr(agent, "_tool_context_hidden_features")}


def _hidden_tools(agent: Any) -> set[str]:
    _ensure_context_profile(agent)
    return {str(item) for item in getattr(agent, "_tool_context_hidden_tools")}


def _clean_name_set(values: Any) -> set[str]:
    if values is None:
        return set()
    if isinstance(values, str):
        raw_value = values.strip()
        if not raw_value:
            return set()
        if raw_value.startswith("["):
            try:
                decoded = json.loads(raw_value)
            except json.JSONDecodeError:
                decoded = None
            if isinstance(decoded, list):
                values = decoded
            else:
                values = [raw_value]
        else:
            values = raw_value.split(",")
    elif not isinstance(values, (list, tuple, set)):
        return {str(values).strip()} if str(values).strip() else set()
    return {str(item).strip() for item in values if str(item).strip()}


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off", ""}:
            return False
        return None
    return bool(value)


def _control_feature_names(agent: Any) -> set[str]:
    controls = {"FeatureFeaturesFeature", "feature_features_feature"}
    for feature in _agent_features(agent).values():
        if feature.__class__.__name__ == "FeatureFeaturesFeature":
            controls.add(feature.tool_name)
    return controls


def _estimate_tool_tokens(openai_tool: dict[str, Any]) -> int:
    body = json.dumps(openai_tool, sort_keys=True, separators=(",", ":"))
    return max(1, (len(body) + 3) // 4)


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
