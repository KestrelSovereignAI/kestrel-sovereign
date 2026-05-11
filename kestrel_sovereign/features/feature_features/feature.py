"""Agent-facing FeatureFeature surface (#1151)."""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import os
import shlex
from pathlib import Path
from typing import Any

from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.features.base import Feature as SDKBaseFeature
from kestrel_sdk.tools.result import ToolResult, ToolResultStatus

from kestrel_sovereign.features import (
    discover_entrypoint_feature_classes,
    discover_feature_class_by_name,
    discover_feature_modules,
    find_feature_class,
    get_disabled_features,
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
        self._register_default_providers()
        return None

    def get_tools(self) -> list[Any]:
        tools = super().get_tools()
        for agent_tool in tools:
            if agent_tool.name == "feature_discover":
                agent_tool.parse_command_args = _parse_feature_discover_command
            elif agent_tool.name == "feature_add":
                agent_tool.parse_command_args = _parse_feature_add_command
            elif agent_tool.name == "feature_remove":
                agent_tool.parse_command_args = _parse_feature_remove_command
            elif agent_tool.name == "feature_focus":
                agent_tool.parse_command_args = _parse_feature_focus_command
            elif agent_tool.name == "feature_unfocus":
                agent_tool.parse_command_args = _parse_feature_unfocus_command
        return tools

    def _register_signal_sources(self) -> None:
        registry = getattr(self.agent, "signal_registry", None)
        if registry is None:
            return
        for registration in build_feature_feature_registrations(self.agent):
            if registry.get(registration.name) is not None:
                continue
            registry.register(registration)

    def _register_default_providers(self) -> None:
        if getattr(self.agent, "feature_feature_file_github_epic", None) is None:
            setattr(
                self.agent,
                "feature_feature_file_github_epic",
                self._feature_feature_file_github_epic,
            )
            setattr(
                self.agent,
                "feature_feature_file_github_epic_requires_github_token",
                True,
            )
        if getattr(self.agent, "feature_feature_assign_talon_chunks", None) is None:
            setattr(
                self.agent,
                "feature_feature_assign_talon_chunks",
                self._feature_feature_assign_talon_chunks,
            )
            setattr(
                self.agent,
                "feature_feature_assign_talon_chunks_requires_talon",
                True,
            )
        if getattr(self.agent, "feature_feature_tests_pass", None) is None:
            setattr(
                self.agent,
                "feature_feature_tests_pass",
                self._feature_feature_tests_pass,
            )
        if getattr(self.agent, "feature_feature_lint_clean", None) is None:
            setattr(
                self.agent,
                "feature_feature_lint_clean",
                self._feature_feature_lint_clean,
            )
        if getattr(self.agent, "feature_feature_boundary_scan", None) is None:
            setattr(
                self.agent,
                "feature_feature_boundary_scan",
                self._feature_feature_boundary_scan,
            )
        if getattr(self.agent, "feature_feature_red_team_review", None) is None:
            setattr(
                self.agent,
                "feature_feature_red_team_review",
                self._feature_feature_red_team_review,
            )
        if getattr(self.agent, "feature_feature_ci_green", None) is None:
            setattr(
                self.agent,
                "feature_feature_ci_green",
                self._feature_feature_ci_green,
            )
            setattr(
                self.agent,
                "feature_feature_ci_green_requires_ci_token",
                True,
            )
        if getattr(self.agent, "feature_feature_audit_anchor", None) is None:
            setattr(
                self.agent,
                "feature_feature_audit_anchor",
                self._feature_feature_audit_anchor,
            )
            setattr(
                self.agent,
                "feature_feature_audit_anchor_requires_audit_anchor",
                True,
            )

    async def _feature_feature_file_github_epic(self, payload: dict) -> dict[str, Any]:
        """Create the FeatureFeature GitHub epic for a proposed feature change."""

        repository = _payload_string(payload, "repository")
        if repository is None or "/" not in repository:
            raise RuntimeError("repository must be owner/repo")
        token = _github_token()
        if not token:
            raise RuntimeError("GITHUB_TOKEN or .env GITHUB_TOKEN is required")

        title = _github_epic_title(payload)
        body = _github_epic_body(payload)
        issue = await _github_create_issue(
            repository,
            token,
            {
                "title": title,
                "body": body,
            },
        )
        if not isinstance(issue, dict):
            raise RuntimeError(f"failed to create GitHub issue in {repository}")
        issue_number = issue.get("number")
        if not _is_strict_positive_int(issue_number):
            raise RuntimeError(f"GitHub issue response missing number in {repository}")
        return {
            "status": "ok",
            "repository": repository,
            "issue_number": issue_number,
            "issue_url": issue.get("html_url"),
            "title": title,
        }

    async def _feature_feature_assign_talon_chunks(
        self,
        payload: dict,
    ) -> dict[str, Any]:
        """Dispatch FeatureFeature implementation work to Talon."""

        repository = _payload_string(payload, "repository")
        if repository is None or "/" not in repository:
            raise RuntimeError("repository must be owner/repo")
        talon = _talon_feature(self.agent)
        if talon is None or not callable(getattr(talon, "talon_claim", None)):
            raise RuntimeError("TalonCoordinatorFeature is not available")

        issue_numbers = _talon_issue_numbers(payload)
        if not issue_numbers:
            raise RuntimeError(
                "issue_number or talon_issue_numbers is required to assign Talon"
            )

        dispatches: list[dict[str, Any]] = []
        for issue_number in issue_numbers:
            result = await talon.talon_claim(
                repo=repository,
                issue=issue_number,
                max_iterations=_payload_positive_int(payload, "max_iterations"),
                max_turns=_payload_positive_int(payload, "max_turns"),
                backend=_payload_string(payload, "talon_backend"),
                model=_payload_string(payload, "talon_model"),
                auth_lane=_payload_string(payload, "talon_auth_lane"),
                skip_clarification=_payload_optional_bool(
                    payload, "skip_clarification"
                ),
                worktree=_payload_optional_bool(payload, "worktree", default=True),
                self_review=_payload_optional_bool(payload, "self_review"),
            )
            if getattr(result, "status", None) is not ToolResultStatus.OK:
                error = getattr(result, "error", None) or getattr(
                    result, "confirmation", None
                )
                raise RuntimeError(
                    error or f"talon_claim failed for {repository}#{issue_number}"
                )
            data = getattr(result, "data", None)
            dispatches.append(
                data if isinstance(data, dict) else {"issue_number": issue_number}
            )

        return {
            "status": "ok",
            "repository": repository,
            "issues": issue_numbers,
            "dispatches": dispatches,
        }

    async def _feature_feature_tests_pass(
        self,
        payload: dict,
    ) -> dict[str, Any]:
        """Run the FeatureFeature tests_pass quality gate."""

        suite = _payload_string(payload, "suite") or "unit"
        command = _quality_command(
            payload,
            keys=("test_command", "tests_command", "command"),
            default=_default_test_command(suite),
        )
        marker = await _run_quality_command(self.agent, payload, command)
        passed = marker["exit_code"] == 0
        return {
            **marker,
            "status": "ok" if passed else "failed",
            "suite": suite,
            "failed": 0 if passed else 1,
            "errors": 0 if passed else 1,
        }

    async def _feature_feature_lint_clean(
        self,
        payload: dict,
    ) -> dict[str, Any]:
        """Run the FeatureFeature lint_clean quality gate."""

        scopes = _payload_string_list(payload, "scopes") or ["changed"]
        command = _quality_command(
            payload,
            keys=("lint_command", "command"),
            default=[
                "uv",
                "run",
                "--no-sync",
                "python",
                "run_tests.py",
                "--kestrel",
                "--skip-check",
                "--validate-only",
            ],
        )
        marker = await _run_quality_command(self.agent, payload, command)
        passed = marker["exit_code"] == 0
        return {
            **marker,
            "status": "ok" if passed else "failed",
            "scopes": scopes,
            "violations": 0 if passed else 1,
            "errors": 0 if passed else 1,
        }

    async def _feature_feature_boundary_scan(
        self,
        payload: dict,
    ) -> dict[str, Any]:
        """Emit the changed patch for the constitutional boundary gate."""

        snapshot = await _feature_change_snapshot(self.agent, payload)
        return {
            **snapshot,
            "status": "ok",
            "scan": "constitutional_boundary",
        }

    async def _feature_feature_red_team_review(
        self,
        payload: dict,
    ) -> dict[str, Any]:
        """Emit the changed patch for the red-team review gate."""

        snapshot = await _feature_change_snapshot(self.agent, payload)
        return {
            **snapshot,
            "status": "ok",
            "review": "red_team",
        }

    async def _feature_feature_ci_green(
        self,
        payload: dict,
    ) -> dict[str, Any]:
        """Acknowledge the ci_green stage before the runner polls GitHub checks."""

        return {
            "status": "ok",
            "repository": _payload_string(payload, "repository"),
            "branch": _payload_string(payload, "branch"),
            "workflow_run_id": _payload_string(payload, "workflow_run_id"),
            "workflow_stage_name": _payload_string(payload, "workflow_stage_name"),
        }

    async def _feature_feature_audit_anchor(
        self,
        payload: dict,
    ) -> dict[str, Any]:
        """Anchor audit logs after an irreversible FeatureFeature publish."""

        audit_anchor = _audit_anchor_feature(self.agent)
        if audit_anchor is None or not callable(
            getattr(audit_anchor, "anchor_audit", None)
        ):
            raise RuntimeError("AuditAnchorFeature is not available")

        result = await audit_anchor.anchor_audit()
        status = getattr(result, "status", None)
        data = getattr(result, "data", None)
        if status is not ToolResultStatus.OK:
            raise RuntimeError(
                getattr(result, "error", None)
                or getattr(result, "confirmation", None)
                or "audit anchor failed"
            )
        return {
            "status": "ok",
            "audit_anchor_status": getattr(status, "value", str(status)),
            "audit_anchor": data if isinstance(data, dict) else {},
            "workflow_run_id": _payload_string(payload, "workflow_run_id"),
            "workflow_stage_name": _payload_string(payload, "workflow_stage_name"),
        }

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
        name="feature_discover",
        description=(
            "Search the feature catalog with provenance, load state, and context visibility."
        ),
        category=ToolCategory.UTILITY,
        command_prefix="!feature-discover",
    )
    async def feature_discover(
        self,
        include_entrypoints: bool = True,
        include_loaded: bool = True,
        include_tools: bool = False,
        loaded_only: bool = False,
        limit: int = 50,
        query: str = "",
    ) -> ToolResult:
        """Return a searchable feature catalog for runtime feature selection.

        Args:
            include_entrypoints: Include installed package entry-point features.
            include_loaded: Include active runtime-only features that are not discoverable.
            include_tools: Include tool rows for loaded features.
            loaded_only: Return only features that are active in this agent runtime.
            limit: Maximum number of catalog rows to return.
            query: Optional text matched against class/module/tool names and docs.
        """

        include_entrypoints_requested = _coerce_bool(include_entrypoints)
        include_loaded_requested = _coerce_bool(include_loaded)
        include_tools_requested = _coerce_bool(include_tools)
        loaded_only_requested = _coerce_bool(loaded_only)
        if (
            include_entrypoints_requested is None
            or include_loaded_requested is None
            or include_tools_requested is None
            or loaded_only_requested is None
        ):
            return ToolResult.failed(
                "include_entrypoints, include_loaded, include_tools, and loaded_only must be booleans."
            )
        limit_value = _coerce_positive_int(limit)
        if limit_value is None:
            return ToolResult.failed("limit must be a positive integer.")

        catalog = _feature_catalog(
            self.agent,
            include_entrypoints=include_entrypoints_requested,
            include_loaded=include_loaded_requested,
            include_tools=True,
        )
        if loaded_only_requested:
            catalog = [row for row in catalog if row["loaded"]]
        filtered = _filter_feature_catalog(catalog, query)[:limit_value]
        response_features = (
            filtered
            if include_tools_requested
            else [_without_catalog_tools(row) for row in filtered]
        )
        loaded_count = sum(1 for row in filtered if row["loaded"])
        visible_count = sum(1 for row in filtered if row["visible_in_context"])
        return ToolResult.ok(
            f"Discovered {len(filtered)} matching feature(s).",
            data={
                "query": str(query or ""),
                "features": response_features,
                "counts": {
                    "matched": len(filtered),
                    "loaded": loaded_count,
                    "visible_in_context": visible_count,
                    "hidden_from_context": loaded_count - visible_count,
                    "total_catalog": len(catalog),
                },
                "actions": {
                    "load": "feature_add",
                    "unload": "feature_remove",
                    "focus_context": "feature_focus",
                    "hide_context": "feature_unfocus",
                    "context_status": "feature_context_status",
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
        name="feature_add",
        description="Load an installed/discoverable feature into this agent runtime.",
        category=ToolCategory.SYSTEM,
        command_prefix="!feature-add",
    )
    async def feature_add(
        self,
        feature: str,
        pre_explore: bool = False,
    ) -> ToolResult:
        """Load an installed feature class into the active agent.

        Args:
            feature: Feature class, module, or shorthand name.
            pre_explore: Promote the feature's direct tools immediately.
        """

        feature_name = str(feature).strip()
        if not feature_name:
            return ToolResult.failed("feature is required.")
        pre_explore_requested = _coerce_bool(pre_explore)
        if pre_explore_requested is None:
            return ToolResult.failed("pre_explore must be a boolean.")

        existing = _resolve_loaded_feature(self.agent, feature_name)
        if existing is not None:
            return ToolResult.ok(
                f"Feature {existing.name} is already loaded.",
                data={"feature": _loaded_feature_row(self.agent, existing)},
            )

        feature_class = discover_feature_class_by_name(feature_name)
        if feature_class is None:
            return ToolResult.failed(f"No discoverable feature matches {feature_name!r}.")
        if feature_class.__name__ in get_disabled_features():
            return ToolResult.failed(
                f"Feature {feature_class.__name__} is disabled by configuration."
            )
        allowed_features = getattr(self.agent, "_allowed_features", None)
        if allowed_features is not None:
            from kestrel_sovereign.multi_agent.config import MANDATORY_FEATURES

            if (
                feature_class.__name__ not in set(allowed_features)
                and feature_class.__name__ not in MANDATORY_FEATURES
            ):
                return ToolResult.failed(
                    f"Feature {feature_class.__name__} is not allowed by this agent profile."
                )

        instance = None
        try:
            instance = feature_class(self.agent)
            if _has_runtime_router(instance):
                return ToolResult.failed(
                    f"Feature {feature_class.__name__} exposes HTTP routes and "
                    "cannot be added at runtime yet."
                )
            register_feature = getattr(self.agent, "_register_feature", None)
            if callable(register_feature):
                await register_feature(instance)
            else:
                await _register_feature_fallback(self.agent, instance)
            await _notify_features_loaded(self.agent)
            if pre_explore_requested:
                register_direct_tools = getattr(
                    self.agent, "_register_explored_feature_tools", None
                )
                if callable(register_direct_tools):
                    register_direct_tools(instance)
            _refresh_cached_features_prompt(self.agent)
        except Exception as exc:
            if instance is not None:
                try:
                    await _rollback_runtime_feature(self.agent, instance)
                except Exception as rollback_exc:
                    return ToolResult.failed(
                        f"Failed to load feature {feature_class.__name__}: {exc}; "
                        f"rollback also failed: {rollback_exc}"
                    )
            return ToolResult.failed(
                f"Failed to load feature {feature_class.__name__}: {exc}"
            )

        return ToolResult.ok(
            f"Loaded feature {feature_class.__name__}.",
            data={"feature": _loaded_feature_row(self.agent, instance)},
        )

    @tool(
        name="feature_remove",
        description="Unload a non-mandatory feature from this agent runtime.",
        category=ToolCategory.SYSTEM,
        command_prefix="!feature-remove",
    )
    async def feature_remove(self, feature: str) -> ToolResult:
        """Unload a runtime feature and remove its promoted direct tools.

        Args:
            feature: Feature class, dispatcher tool, registry key, or shorthand name.
        """

        feature_name = str(feature).strip()
        if not feature_name:
            return ToolResult.failed("feature is required.")
        loaded = _resolve_loaded_feature(self.agent, feature_name)
        if loaded is None:
            return ToolResult.failed(f"Feature {feature_name!r} is not loaded.")

        from kestrel_sovereign.multi_agent.config import MANDATORY_FEATURES

        if loaded.__class__.__name__ in MANDATORY_FEATURES:
            return ToolResult.failed(
                f"Feature {loaded.__class__.__name__} is mandatory and cannot be removed."
            )
        if loaded.__class__.__name__ == "FeatureFeaturesFeature":
            return ToolResult.failed("FeatureFeaturesFeature cannot remove itself.")
        if _has_runtime_router(loaded):
            return ToolResult.failed(
                f"Feature {loaded.__class__.__name__} exposes HTTP routes and "
                "cannot be removed at runtime yet."
            )

        disable_feature = getattr(self.agent, "_disable_feature", None)
        if callable(disable_feature):
            await disable_feature(loaded.name)
        else:
            await _disable_feature_fallback(self.agent, loaded)
        _refresh_cached_features_prompt(self.agent)

        return ToolResult.ok(
            f"Removed feature {loaded.__class__.__name__}.",
            data={"removed": loaded.__class__.__name__},
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
        try:
            run_params = _coerce_object_params(params, "FeatureFeature run params")
            run_version = _coerce_nonnegative_int(version, "version")
        except ValueError as exc:
            return ToolResult.failed(str(exc))

        try:
            workflow_name = _workflow_name_for_kind(kind)
        except ValueError as exc:
            return ToolResult.failed(str(exc))

        return await workflow_feature.workflow_run(
            workflow_name,
            params=run_params,
            version=run_version,
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
        missing_provider_requirements: list[dict[str, Any]] = []
        missing_cognition_prompts: list[str] = []
        missing_workflow_prerequisites: list[dict[str, Any]] = []
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
            provider_requirements = _provider_requirement_status(
                self.agent,
                provider_name,
            )
            provider_requirements_ready = all(
                item["ready"] for item in provider_requirements
            )
            action_provider_ready = (
                mode != "action"
                or registered_handler_ready
                or (
                    provider_name is not None
                    and provider_requirements_ready
                )
            )
            prompt_ready = (
                mode != "cognition"
                or (
                    active_registration.prompt_template is not None
                    and active_registration.prompt_template.exists()
                )
            )
            workflow_prerequisites = _workflow_prerequisite_status(
                self.agent,
                registration.name,
            )
            workflow_prerequisites_ready = all(
                item["ready"] for item in workflow_prerequisites
            )
            if mode == "action" and not action_provider_ready:
                missing_action_providers.append(registration.name)
            if mode == "action" and not registered_handler_ready:
                for requirement in provider_requirements:
                    if requirement["ready"]:
                        continue
                    missing_provider_requirements.append(
                        {
                            "source": registration.name,
                            "provider": provider_name,
                            "requirement": requirement["name"],
                        }
                    )
            if mode == "cognition" and not prompt_ready:
                missing_cognition_prompts.append(registration.name)
            for prerequisite in workflow_prerequisites:
                if not prerequisite["ready"]:
                    missing_workflow_prerequisites.append(
                        {
                            "source": registration.name,
                            "prerequisite": prerequisite["name"],
                        }
                    )
            rows.append(
                {
                    "name": registration.name,
                    "stage": _source_stage(registration.name),
                    "mode": mode,
                    "registered": registered,
                    "provider": provider_name,
                    "registered_handler": registered_handler_ready,
                    "provider_requirements": provider_requirements,
                    "provider_requirements_ready": provider_requirements_ready,
                    "action_provider_ready": action_provider_ready,
                    "prompt_template_exists": prompt_ready,
                    "workflow_prerequisites": workflow_prerequisites,
                    "workflow_prerequisites_ready": workflow_prerequisites_ready,
                    "ready": (
                        registered
                        and action_provider_ready
                        and prompt_ready
                        and workflow_prerequisites_ready
                    ),
                }
            )

        ok = (
            not missing_registered
            and not missing_action_providers
            and not missing_provider_requirements
            and not missing_cognition_prompts
            and not missing_workflow_prerequisites
        )
        blocking_sources = [row for row in rows if not row["ready"]]
        data = {
            "sources": rows,
            "blocking_sources": blocking_sources,
            "missing_registered": missing_registered,
            "missing_action_providers": missing_action_providers,
            "missing_provider_requirements": missing_provider_requirements,
            "missing_cognition_prompts": missing_cognition_prompts,
            "missing_workflow_prerequisites": missing_workflow_prerequisites,
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


def _parse_feature_discover_command(user_input: str) -> dict[str, Any]:
    parts = _split_command(user_input)
    if not parts or parts[0].lower() != "!feature-discover":
        return {}

    args: dict[str, Any] = {}
    query_parts: list[str] = []
    for token in parts[1:]:
        if "=" in token:
            key, value = token.split("=", 1)
            key = key.strip("-")
            if key in {
                "include_entrypoints",
                "include_loaded",
                "include_tools",
                "loaded_only",
            }:
                parsed_bool = _coerce_bool(value)
                args[key] = value if parsed_bool is None else parsed_bool
            elif key == "limit":
                try:
                    args[key] = int(value)
                except ValueError:
                    args[key] = value
            elif key == "query":
                args[key] = value
            else:
                query_parts.append(token)
        elif token == "--include-tools":
            args["include_tools"] = True
        elif token == "--no-entrypoints":
            args["include_entrypoints"] = False
        elif token == "--loaded-only":
            args["loaded_only"] = True
        else:
            query_parts.append(token)

    if query_parts and "query" not in args:
        args["query"] = " ".join(query_parts)
    return args


def _parse_feature_add_command(user_input: str) -> dict[str, Any]:
    parts = _split_command(user_input)
    if not parts or parts[0].lower() != "!feature-add":
        return {}

    args: dict[str, Any] = {}
    feature_parts: list[str] = []
    for token in parts[1:]:
        if token in {"--pre-explore", "--pre_explore"}:
            args["pre_explore"] = True
        elif token in {"--no-pre-explore", "--no-pre_explore"}:
            args["pre_explore"] = False
        elif _option_key(token) in {"pre_explore", "pre-explore"}:
            value = token.split("=", 1)[1]
            parsed = _coerce_bool(value)
            args["pre_explore"] = value if parsed is None else parsed
        elif _option_key(token) == "feature":
            args["feature"] = token.split("=", 1)[1]
        else:
            feature_parts.append(token)
    if feature_parts and "feature" not in args:
        args["feature"] = " ".join(feature_parts)
    return args


def _parse_feature_remove_command(user_input: str) -> dict[str, Any]:
    parts = _split_command(user_input)
    if not parts or parts[0].lower() != "!feature-remove":
        return {}
    if len(parts) <= 1:
        return {}
    if _option_key(parts[1]) == "feature":
        return {"feature": parts[1].split("=", 1)[1]}
    return {"feature": " ".join(parts[1:])}


def _parse_feature_focus_command(user_input: str) -> dict[str, Any]:
    parts = _split_command(user_input)
    if not parts or parts[0].lower() != "!feature-focus":
        return {}
    return _parse_feature_context_selection(parts[1:])


def _parse_feature_unfocus_command(user_input: str) -> dict[str, Any]:
    parts = _split_command(user_input)
    if not parts or parts[0].lower() != "!feature-unfocus":
        return {}
    return _parse_feature_context_selection(parts[1:], allow_reset=True)


def _parse_feature_context_selection(
    tokens: list[str],
    *,
    allow_reset: bool = False,
) -> dict[str, Any]:
    args: dict[str, Any] = {}
    positional_features: list[str] = []
    feature_names: list[str] = []
    tool_names: list[str] = []
    active_bucket = "features"

    for token in tokens:
        if allow_reset and token == "--reset":
            args["reset"] = True
            continue
        if allow_reset and _option_key(token) == "reset":
            value = token.split("=", 1)[1]
            parsed = _coerce_bool(value)
            args["reset"] = value if parsed is None else parsed
            continue
        if token in {"--feature", "--features"}:
            active_bucket = "features"
            continue
        if token in {"--tool", "--tools"}:
            active_bucket = "tools"
            continue
        if _option_key(token) in {"feature", "features"}:
            feature_names.extend(sorted(_clean_name_set(token.split("=", 1)[1])))
            continue
        if _option_key(token) in {"tool", "tools"}:
            tool_names.extend(sorted(_clean_name_set(token.split("=", 1)[1])))
            continue
        if active_bucket == "tools":
            tool_names.append(token)
        else:
            positional_features.append(token)

    feature_names.extend(positional_features)
    if feature_names:
        args["features"] = feature_names
    if tool_names:
        args["tools"] = tool_names
    return args


def _split_command(user_input: str) -> list[str]:
    try:
        return shlex.split(user_input.strip())
    except ValueError:
        return user_input.strip().split()


def _option_key(token: str) -> str | None:
    if "=" not in token:
        return None
    key = token.split("=", 1)[0].strip()
    return key.lstrip("-") or None


def _feature_catalog(
    agent: Any,
    *,
    include_entrypoints: bool,
    include_loaded: bool,
    include_tools: bool,
) -> list[dict[str, Any]]:
    loaded_rows = _loaded_feature_inventory(agent)
    loaded_by_class = {row["class"]: row for row in loaded_rows}
    loaded_by_module = {row["module"]: row for row in loaded_rows}
    catalog: list[dict[str, Any]] = []
    seen_classes: set[str] = set()
    allowed_features = getattr(agent, "_allowed_features", None)
    allowed_set = set(allowed_features) if allowed_features is not None else None
    disabled = get_disabled_features()

    for module_path in sorted(discover_feature_modules()):
        module = importlib.import_module(module_path)
        feature_class = find_feature_class(module)
        if feature_class is None:
            continue
        loaded = loaded_by_class.get(feature_class.__name__) or loaded_by_module.get(
            module_path
        )
        catalog.append(
            _feature_catalog_row(
                feature_class,
                source="core",
                provenance=module_path,
                loaded=loaded,
                disabled=feature_class.__name__ in disabled,
                allowed=_feature_allowed(feature_class.__name__, allowed_set),
                include_tools=include_tools,
            )
        )
        seen_classes.add(feature_class.__name__)

    if include_entrypoints:
        for entrypoint_name, feature_class in sorted(
            discover_entrypoint_feature_classes().items()
        ):
            if feature_class.__name__ in seen_classes:
                continue
            loaded = loaded_by_class.get(feature_class.__name__) or loaded_by_module.get(
                feature_class.__module__
            )
            catalog.append(
                _feature_catalog_row(
                    feature_class,
                    source="entrypoint",
                    provenance=entrypoint_name,
                    loaded=loaded,
                    disabled=feature_class.__name__ in disabled,
                    allowed=_feature_allowed(feature_class.__name__, allowed_set),
                    include_tools=include_tools,
                )
            )
            seen_classes.add(feature_class.__name__)

    if include_loaded:
        for loaded in loaded_rows:
            if loaded["class"] in seen_classes:
                continue
            catalog.append(
                {
                    "name": _feature_module_name(loaded["module"]),
                    "class": loaded["class"],
                    "module": loaded["module"],
                    "source": "runtime",
                    "provenance": loaded["registry_key"],
                    "summary": None,
                    "docs": [],
                    "source_path": None,
                    "loaded": True,
                    "visible_in_context": loaded["visible_in_context"],
                    "tool_name": loaded["tool_name"],
                    "tool_count": loaded["tool_count"],
                    "tools": loaded["tools"] if include_tools else [],
                    "tools_discoverable": loaded["tools_discoverable"],
                    "direct_tool_invokable": loaded["direct_tool_invokable"],
                    "subagent_capable": loaded["subagent_capable"],
                    "subagent_executable": loaded["subagent_executable"],
                    "workflow_invocation_modes": loaded["workflow_invocation_modes"],
                    "subagent_workflow_executable": loaded[
                        "subagent_workflow_executable"
                    ],
                    "disabled": loaded["class"] in disabled,
                    "allowed": _feature_allowed(loaded["class"], allowed_set),
                }
            )
            seen_classes.add(loaded["class"])

    return sorted(catalog, key=lambda row: (not row["loaded"], row["class"].lower()))


def _feature_catalog_row(
    feature_class: type,
    *,
    source: str,
    provenance: str,
    loaded: dict[str, Any] | None,
    disabled: bool,
    allowed: bool,
    include_tools: bool,
) -> dict[str, Any]:
    module = inspect.getmodule(feature_class)
    module_name = feature_class.__module__
    source_path = _feature_source_path(feature_class)
    return {
        "name": _feature_module_name(module_name),
        "class": feature_class.__name__,
        "module": module_name,
        "source": source,
        "provenance": provenance,
        "summary": _feature_summary(feature_class),
        "docs": _feature_docs(module, source_path),
        "source_path": str(source_path) if source_path is not None else None,
        "loaded": loaded is not None,
        "visible_in_context": (
            bool(loaded["visible_in_context"]) if loaded is not None else False
        ),
        "tool_name": loaded["tool_name"] if loaded is not None else None,
        "tool_count": loaded["tool_count"] if loaded is not None else 0,
        "tools": loaded["tools"] if include_tools and loaded is not None else [],
        "tools_discoverable": bool(loaded["tools_discoverable"])
        if loaded is not None
        else False,
        "direct_tool_invokable": (
            bool(loaded["direct_tool_invokable"]) if loaded is not None else False
        ),
        "subagent_capable": _feature_class_supports_subagent(feature_class),
        "subagent_executable": (
            bool(loaded["subagent_executable"])
            if loaded is not None
            else False
        ),
        "workflow_invocation_modes": (
            loaded["workflow_invocation_modes"]
            if loaded is not None
            else _workflow_invocation_modes(
                direct_tool_invokable=False,
                subagent_executable=False,
            )
        ),
        "subagent_workflow_executable": (
            bool(loaded["subagent_workflow_executable"])
            if loaded is not None
            else False
        ),
        "disabled": disabled,
        "allowed": allowed,
    }


def _feature_allowed(class_name: str, allowed_features: set[str] | None) -> bool:
    if allowed_features is None:
        return True
    from kestrel_sovereign.multi_agent.config import MANDATORY_FEATURES

    return class_name in allowed_features or class_name in MANDATORY_FEATURES


def _feature_source_path(feature_class: type) -> Path | None:
    try:
        path = Path(inspect.getfile(feature_class))
    except (OSError, TypeError):
        return None
    return path


def _feature_summary(feature_class: type) -> str | None:
    doc = inspect.getdoc(feature_class)
    if not doc:
        return None
    return doc.split("\n\n", 1)[0]


def _feature_docs(module: Any, source_path: Path | None) -> list[str]:
    docs: list[str] = []
    module_doc = inspect.getdoc(module) if module is not None else None
    if module_doc:
        docs.append(module_doc.split("\n\n", 1)[0])
    if source_path is not None:
        for name in ("README.md", "README.rst", "SKILL.md"):
            candidate = source_path.parent / name
            if candidate.exists():
                docs.append(str(candidate))
    return docs


def _filter_feature_catalog(
    catalog: list[dict[str, Any]],
    query: Any,
) -> list[dict[str, Any]]:
    text = str(query or "").strip().lower()
    if not text:
        return catalog
    terms = [term for term in text.replace(",", " ").split() if term]
    if not terms:
        return catalog

    def haystack(row: dict[str, Any]) -> str:
        values = [
            row.get("name"),
            row.get("class"),
            row.get("module"),
            row.get("source"),
            row.get("provenance"),
            row.get("summary"),
            row.get("tool_name"),
            *(row.get("docs") or []),
        ]
        for tool_row in row.get("tools") or []:
            values.extend(
                [
                    tool_row.get("name"),
                    tool_row.get("description"),
                    tool_row.get("category"),
                ]
            )
        return " ".join(str(value).lower() for value in values if value)

    return [row for row in catalog if all(term in haystack(row) for term in terms)]


def _without_catalog_tools(row: dict[str, Any]) -> dict[str, Any]:
    response_row = dict(row)
    response_row["tools"] = []
    return response_row


def _loaded_feature_inventory(agent: Any) -> list[dict[str, Any]]:
    rows = []
    hidden_features = _hidden_features(agent)
    direct_tool_names_by_feature = _direct_tool_names_by_feature(agent)
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
        tools_discoverable = bool(tools)
        direct_tool_invokable = bool(direct_tool_names_by_feature.get(tool_name))
        subagent_capable = _feature_supports_subagent(feature)
        subagent_executable = _feature_supports_subagent(feature)
        workflow_invocation_modes = _workflow_invocation_modes(
            direct_tool_invokable=direct_tool_invokable,
            subagent_executable=subagent_executable,
        )
        rows.append(
            {
                "registry_key": key,
                "class": class_name,
                "tool_name": tool_name,
                "module": feature.__class__.__module__,
                "tool_count": len(tools),
                "tools_discoverable": tools_discoverable,
                "direct_tool_invokable": direct_tool_invokable,
                "subagent_capable": subagent_capable,
                "subagent_executable": subagent_executable,
                "workflow_invocation_modes": workflow_invocation_modes,
                "subagent_workflow_executable": subagent_executable,
                "visible_in_context": (
                    tool_name not in hidden_features
                    and class_name not in hidden_features
                ),
                "tools": tools,
            }
        )
    return rows


def _loaded_feature_row(agent: Any, feature: Any) -> dict[str, Any]:
    for row in _loaded_feature_inventory(agent):
        if row["class"] == feature.__class__.__name__:
            return row
    return {
        "registry_key": getattr(feature, "name", feature.__class__.__name__),
        "class": feature.__class__.__name__,
        "tool_name": getattr(feature, "tool_name", None),
        "module": feature.__class__.__module__,
        "tool_count": 0,
        "tools_discoverable": False,
        "direct_tool_invokable": False,
        "subagent_capable": _feature_supports_subagent(feature),
        "subagent_executable": _feature_supports_subagent(feature),
        "workflow_invocation_modes": _workflow_invocation_modes(
            direct_tool_invokable=False,
            subagent_executable=_feature_supports_subagent(feature),
        ),
        "subagent_workflow_executable": _feature_supports_subagent(feature),
        "visible_in_context": True,
        "tools": [],
    }


def _resolve_loaded_feature(agent: Any, name: str) -> Any | None:
    target = _normalize_lookup(name)
    if not target:
        return None
    for key, feature in _agent_features(agent).items():
        aliases = {
            _normalize_lookup(key),
            _normalize_lookup(getattr(feature, "name", "")),
            _normalize_lookup(feature.__class__.__name__),
            _normalize_lookup(feature.__class__.__name__.removesuffix("Feature")),
            _normalize_lookup(getattr(feature, "tool_name", "")),
        }
        if target in aliases:
            return feature
    return None


async def _register_feature_fallback(agent: Any, feature: Any) -> None:
    await feature.initialize()
    if not isinstance(getattr(agent, "features", None), dict):
        setattr(agent, "features", {})
    _agent_features(agent)[feature.name] = feature
    await feature.on_enable()


async def _disable_feature_fallback(agent: Any, feature: Any) -> None:
    await feature.on_disable()
    await feature.shutdown()
    features = _agent_features(agent)
    for key, value in list(features.items()):
        if value is feature:
            del features[key]


async def _rollback_runtime_feature(agent: Any, feature: Any) -> None:
    if feature not in _agent_features(agent).values():
        return
    disable_feature = getattr(agent, "_disable_feature", None)
    if callable(disable_feature):
        await disable_feature(getattr(feature, "name", feature.__class__.__name__))
    else:
        await _disable_feature_fallback(agent, feature)
    _refresh_cached_features_prompt(agent)


def _has_runtime_router(feature: Any) -> bool:
    get_router = getattr(feature.__class__, "get_router", None)
    return get_router not in (Feature.get_router, SDKBaseFeature.get_router, None)


def _feature_supports_subagent(feature: Any) -> bool:
    return _feature_class_supports_subagent(feature.__class__)


def _feature_class_supports_subagent(feature_class: type) -> bool:
    return (
        callable(getattr(feature_class, "handle_task", None))
        and callable(getattr(feature_class, "execute_as_subagent", None))
    )


def _workflow_invocation_modes(
    *,
    direct_tool_invokable: bool,
    subagent_executable: bool,
) -> list[str]:
    modes = []
    if direct_tool_invokable:
        modes.append("direct_tool")
    if subagent_executable:
        modes.append("subagent")
    return modes


def _direct_tool_names_by_feature(agent: Any) -> dict[str, set[str]]:
    direct_tools = getattr(agent, "_direct_tools", {})
    tool_to_feature = getattr(agent, "_tool_to_feature", {})
    if not isinstance(direct_tools, dict) or not isinstance(tool_to_feature, dict):
        return {}
    rows: dict[str, set[str]] = {}
    for tool_name in direct_tools:
        owner = tool_to_feature.get(tool_name)
        if not isinstance(owner, str) or not owner:
            continue
        rows.setdefault(owner, set()).add(tool_name)
    return rows


async def _notify_features_loaded(agent: Any) -> None:
    for feature in list(_agent_features(agent).values()):
        await feature.post_all_features_loaded(agent)


def _normalize_lookup(name: Any) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


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
                "invocation_mode": "direct_tool",
                "invokable": True,
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


def _coerce_positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 1 else None
    if isinstance(value, str):
        try:
            parsed = int(value.strip())
        except ValueError:
            return None
        return parsed if parsed >= 1 else None
    return None


def _coerce_nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = int(value.strip())
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer") from exc
    else:
        raise ValueError(f"{name} must be an integer")
    if parsed < 0:
        raise ValueError(f"{name} must be >= 0")
    return parsed


def _coerce_object_params(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        raw_value = value.strip()
        if not raw_value:
            return {}
        try:
            decoded = json.loads(raw_value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{name} must be a JSON object") from exc
        if isinstance(decoded, dict):
            return decoded
        raise ValueError(f"{name} must be a JSON object")
    raise ValueError(f"{name} must be an object")


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


def _github_epic_title(payload: dict[str, Any]) -> str:
    feature_name = _payload_string(payload, "feature_name")
    package_name = _payload_string(payload, "package_name")
    target_tool_name = _payload_string(payload, "target_tool_name")
    subject = feature_name or package_name or target_tool_name or "feature proposal"
    if target_tool_name and target_tool_name != subject:
        subject = f"{subject}: {target_tool_name}"
    return f"[EPIC] Feature proposal: {subject}"


def _github_epic_body(payload: dict[str, Any]) -> str:
    summary = _payload_string(payload, "summary") or "No summary provided."
    lines = [
        "## FeatureFeature Proposal",
        "",
        summary,
        "",
        "## Proposal Parameters",
        "",
        "```json",
        json.dumps(_redacted_payload(payload), indent=2, sort_keys=True),
        "```",
        "",
        "## Workflow",
        "",
        "Created by the FeatureFeature `file_github_epic` provider.",
    ]
    return "\n".join(lines)


def _redacted_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        str(key): _redacted_value(str(key), value)
        for key, value in payload.items()
    }


def _redacted_value(key_text: str, value: Any) -> Any:
    if _secretish_key(key_text):
        return "<redacted>"
    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            child_key = str(key)
            redacted[child_key] = _redacted_value(child_key, item)
        return redacted
    if isinstance(value, list):
        return [_redacted_value("", item) for item in value]
    if isinstance(value, tuple):
        return [_redacted_value("", item) for item in value]
    return value


def _secretish_key(key_text: str) -> bool:
    return any(
        secret in key_text.lower()
        for secret in (
            "auth",
            "credential",
            "key",
            "password",
            "passwd",
            "private",
            "secret",
            "token",
        )
    )


def _payload_string(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _payload_positive_int(payload: dict[str, Any], key: str) -> int | None:
    value = payload.get(key)
    if _is_strict_positive_int(value):
        return value
    return None


def _payload_optional_bool(
    payload: dict[str, Any],
    key: str,
    *,
    default: bool | None = None,
) -> bool | None:
    value = payload.get(key, default)
    return value if isinstance(value, bool) else default


def _payload_string_list(payload: dict[str, Any], key: str) -> list[str]:
    value = payload.get(key)
    if value is None:
        return []
    if isinstance(value, str):
        raw_value = value.strip()
        if not raw_value:
            return []
        if raw_value.startswith("["):
            try:
                decoded = json.loads(raw_value)
            except json.JSONDecodeError:
                decoded = None
            if isinstance(decoded, list):
                value = decoded
            else:
                value = raw_value.split(",")
        else:
            value = raw_value.split(",")
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _quality_command(
    payload: dict[str, Any],
    *,
    keys: tuple[str, ...],
    default: list[str],
) -> list[str]:
    command = _custom_quality_command(payload, keys=keys)
    return default if command is None else command


def _custom_quality_command(
    payload: dict[str, Any],
    *,
    keys: tuple[str, ...],
) -> list[str] | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return shlex.split(value)
        if isinstance(value, (list, tuple)) and value:
            command = [str(item) for item in value if str(item)]
            if command:
                return command
    return None


def _default_test_command(suite: str) -> list[str]:
    suite_paths = {
        "unit": "tests/unit/",
        "integration": "tests/integration/",
        "llm": "tests/llm/",
    }
    target = suite_paths.get(suite, suite)
    return ["uv", "run", "--no-sync", "pytest", target, "-q"]


async def _run_quality_command(
    agent: Any,
    payload: dict[str, Any],
    command: list[str],
    *,
    output_limit: int | None = 4000,
) -> dict[str, Any]:
    cwd = _quality_command_cwd(payload)
    timeout_seconds = _quality_timeout_seconds(payload)
    runner = getattr(agent, "feature_feature_command_runner", None)
    if callable(runner):
        result = runner(command=command, cwd=str(cwd), timeout=timeout_seconds)
        if inspect.isawaitable(result):
            result = await result
        return _quality_command_marker(command, cwd, result, output_limit=output_limit)

    env = dict(os.environ)
    env.setdefault("KESTREL_AUDIT_MODE", "skip")
    process = None
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(cwd),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(),
            timeout=timeout_seconds,
        )
    except FileNotFoundError as exc:
        return _quality_command_marker(
            command,
            cwd,
            {
                "exit_code": 127,
                "stdout": "",
                "stderr": str(exc),
            },
            output_limit=output_limit,
        )
    except asyncio.TimeoutError:
        if process is not None:
            try:
                process.kill()
                await process.communicate()
            except ProcessLookupError:
                pass
        return _quality_command_marker(
            command,
            cwd,
            {
                "exit_code": 124,
                "stdout": "",
                "stderr": f"timed out after {timeout_seconds:g}s",
            },
            output_limit=output_limit,
        )

    return _quality_command_marker(
        command,
        cwd,
        {
            "exit_code": process.returncode,
            "stdout": stdout_bytes.decode(errors="replace"),
            "stderr": stderr_bytes.decode(errors="replace"),
        },
        output_limit=output_limit,
    )


async def _feature_change_snapshot(
    agent: Any,
    payload: dict[str, Any],
) -> dict[str, Any]:
    base_ref = _payload_string(payload, "base_ref")
    if base_ref is None:
        base_ref = _payload_string(payload, "base_branch") or "origin/main"

    merge_base: str | None = None
    command = _custom_quality_command(payload, keys=("diff_command", "patch_command"))
    if command is None:
        merge_base_marker = await _run_quality_command(
            agent,
            payload,
            ["git", "merge-base", base_ref, "HEAD"],
        )
        if merge_base_marker["exit_code"] != 0:
            stderr = (
                merge_base_marker["stderr"]
                or merge_base_marker["stdout"]
                or "merge-base command failed"
            )
            raise RuntimeError(f"feature change snapshot failed: {stderr}")
        merge_base_lines = merge_base_marker["stdout"].strip().splitlines()
        if not merge_base_lines:
            raise RuntimeError("feature change snapshot failed: empty merge-base")
        merge_base = merge_base_lines[0]
        command = ["git", "diff", "--no-ext-diff", merge_base]

    marker = await _run_quality_command(agent, payload, command, output_limit=None)
    if marker["exit_code"] != 0:
        stderr = marker["stderr"] or marker["stdout"] or "diff command failed"
        raise RuntimeError(f"feature change snapshot failed: {stderr}")

    untracked_patch = await _untracked_files_patch(agent, payload)
    patch = marker["stdout"]
    if untracked_patch:
        patch = f"{patch.rstrip()}\n{untracked_patch}" if patch else untracked_patch
    changed_files = _changed_files_from_patch(patch)
    return {
        "command": marker["command"],
        "cwd": marker["cwd"],
        "base_ref": base_ref,
        "merge_base": merge_base,
        "patch": patch,
        "changed_files": changed_files,
        "changed_file_count": len(changed_files),
    }


async def _untracked_files_patch(agent: Any, payload: dict[str, Any]) -> str:
    include_untracked = _payload_optional_bool(
        payload,
        "include_untracked",
        default=True,
    )
    if not include_untracked:
        return ""
    marker = await _run_quality_command(
        agent,
        payload,
        ["git", "ls-files", "--others", "--exclude-standard"],
        output_limit=None,
    )
    if marker["exit_code"] != 0:
        stderr = marker["stderr"] or marker["stdout"] or "ls-files command failed"
        raise RuntimeError(f"feature change snapshot failed: {stderr}")

    cwd = Path(marker["cwd"])
    chunks: list[str] = []
    for file_name in marker["stdout"].splitlines():
        if not file_name.strip():
            continue
        path = (cwd / file_name).resolve()
        if not path.is_relative_to(cwd) or not path.is_file():
            continue
        chunks.append(_new_file_patch(file_name, path))
    return "\n".join(chunk for chunk in chunks if chunk)


def _new_file_patch(file_name: str, path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    patch = (
        f"diff --git a/{file_name} b/{file_name}\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        f"+++ b/{file_name}"
    )
    if not lines:
        return patch
    added = "\n".join(f"+{line}" for line in lines)
    return f"{patch}\n@@ -0,0 +1,{len(lines)} @@\n{added}"


def _quality_command_marker(
    command: list[str],
    cwd: Path,
    result: Any,
    *,
    output_limit: int | None = 4000,
) -> dict[str, Any]:
    if not isinstance(result, dict):
        result = {"exit_code": 1, "stdout": "", "stderr": str(result)}
    exit_code = result.get(
        "exit_code",
        result.get("returncode", result.get("return_code")),
    )
    if isinstance(exit_code, bool) or not isinstance(exit_code, int):
        exit_code = 1
    return {
        "command": command,
        "cwd": str(cwd),
        "exit_code": exit_code,
        "stdout": _truncate_output(result.get("stdout"), limit=output_limit),
        "stderr": _truncate_output(result.get("stderr"), limit=output_limit),
    }


def _quality_command_cwd(payload: dict[str, Any]) -> Path:
    for key in ("cwd", "worktree_path", "repository_path"):
        value = _payload_string(payload, key)
        if value is not None:
            return Path(value).expanduser().resolve()
    return Path.cwd()


def _quality_timeout_seconds(payload: dict[str, Any]) -> float:
    value = payload.get("timeout_seconds", payload.get("quality_timeout_seconds"))
    if isinstance(value, bool):
        return 600.0
    if isinstance(value, (int, float)) and value > 0:
        return float(value)
    if isinstance(value, str):
        try:
            parsed = float(value.strip())
        except ValueError:
            return 600.0
        if parsed > 0:
            return parsed
    return 600.0


def _truncate_output(value: Any, *, limit: int | None = 4000) -> str:
    text = value if isinstance(value, str) else ""
    if limit is None:
        return text
    if len(text) <= limit:
        return text
    return text[:limit] + "\n<truncated>"


def _changed_files_from_patch(patch: str) -> list[str]:
    files: list[str] = []
    current_path: str | None = None
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            current_path = None
            if len(parts) >= 4 and parts[3].startswith("b/"):
                current_path = parts[3].removeprefix("b/")
            continue
        if not line.startswith("+++ b/"):
            if line == "+++ /dev/null" and current_path:
                files.append(current_path)
            continue
        path = line.removeprefix("+++ b/").strip()
        if path and path != "/dev/null":
            files.append(path)
    return list(dict.fromkeys(files))


def _talon_issue_numbers(payload: dict[str, Any]) -> list[int]:
    numbers: list[int] = []
    for key in ("issue_number", "talon_issue_number", "github_issue_number"):
        value = payload.get(key)
        if _is_strict_positive_int(value):
            numbers.append(value)
    for item in _iter_issue_candidates(payload.get("talon_issue_numbers")):
        if _is_strict_positive_int(item):
            numbers.append(item)
    for item in _iter_issue_candidates(payload.get("issues")):
        if _is_strict_positive_int(item):
            numbers.append(item)
    for chunk in payload.get("chunks", []):
        if not isinstance(chunk, dict):
            continue
        for key in ("issue_number", "talon_issue_number", "github_issue_number"):
            value = chunk.get(key)
            if _is_strict_positive_int(value):
                numbers.append(value)
                break
    return list(dict.fromkeys(numbers))


def _iter_issue_candidates(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        candidates = []
        for item in value:
            if isinstance(item, dict):
                candidates.extend(
                    item.get(key)
                    for key in (
                        "number",
                        "issue",
                        "issue_number",
                        "talon_issue_number",
                        "github_issue_number",
                    )
                )
            else:
                candidates.append(item)
        return candidates
    return [value]


def _is_strict_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _github_token() -> str | None:
    from kestrel_sovereign.features.strategic_memory.github_integration import (
        get_github_token,
    )

    return get_github_token() or os.environ.get("GH_TOKEN")


def _ci_green_token() -> str | None:
    return os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")


async def _github_create_issue(
    repository: str,
    token: str,
    body: dict[str, Any],
) -> Any:
    from kestrel_sovereign.features.strategic_memory.github_integration import (
        github_api_post,
    )

    return await github_api_post(f"/repos/{repository}/issues", token, body)


def _resolve_provider_name(agent: Any, source: str) -> str | None:
    stage = _source_stage(source)
    for name in (
        f"feature_feature_{stage}",
        f"feature_feature_handle_{stage}",
    ):
        if callable(getattr(agent, name, None)):
            return name
    return None


def _source_stage(source: str) -> str:
    return source.split(".")[-1]


def _registered_handler_ready(registration: Any) -> bool:
    if registration is None:
        return False
    handler = getattr(registration, "handler", None)
    return callable(handler) and not bool(
        getattr(handler, "_feature_feature_requires_agent_provider", False)
    )


def _provider_requirement_status(
    agent: Any,
    provider_name: str | None,
) -> list[dict[str, Any]]:
    if provider_name is None:
        return []
    statuses: list[dict[str, Any]] = []
    if bool(getattr(agent, f"{provider_name}_requires_github_token", False)):
        statuses.append({"name": "github_token", "ready": bool(_github_token())})
    if bool(getattr(agent, f"{provider_name}_requires_ci_token", False)):
        statuses.append({"name": "ci_token", "ready": bool(_ci_green_token())})
    if bool(getattr(agent, f"{provider_name}_requires_talon", False)):
        statuses.append(
            {
                "name": "TalonCoordinatorFeature",
                "ready": _talon_feature(agent) is not None,
            }
        )
    if bool(getattr(agent, f"{provider_name}_requires_audit_anchor", False)):
        statuses.append(
            {
                "name": "AuditAnchorFeature",
                "ready": _audit_anchor_feature(agent) is not None,
            }
        )
    return statuses


def _workflow_prerequisite_status(
    agent: Any,
    source: str,
) -> list[dict[str, Any]]:
    if source != "feature_features.red_team_review":
        return []
    return [
        {
            "name": "workflow_red_team_prompt_pack_resolver",
            "ready": _red_team_prompt_pack_resolver_ready(agent),
        },
        {
            "name": "workflow_red_team_attestation_resolver",
            "ready": _red_team_attestation_resolver_ready(agent),
        },
    ]


def _red_team_prompt_pack_resolver_ready(agent: Any) -> bool:
    if callable(getattr(agent, "workflow_red_team_prompt_pack_resolver", None)):
        return True
    features = getattr(agent, "features", None)
    if not isinstance(features, dict):
        return False
    workflow = features.get("WorkflowsFeature")
    if workflow is None:
        for feature in features.values():
            if feature.__class__.__name__ == "WorkflowsFeature":
                workflow = feature
                break
    runner = getattr(workflow, "runner", None)
    return callable(getattr(runner, "red_team_prompt_pack_resolver", None))


def _red_team_attestation_resolver_ready(agent: Any) -> bool:
    if callable(getattr(agent, "workflow_red_team_attestation_resolver", None)):
        return True
    features = getattr(agent, "features", None)
    if not isinstance(features, dict):
        return False
    workflow = features.get("WorkflowsFeature")
    if workflow is None:
        for feature in features.values():
            if feature.__class__.__name__ == "WorkflowsFeature":
                workflow = feature
                break
    runner = getattr(workflow, "runner", None)
    return callable(getattr(runner, "red_team_attestation_resolver", None))


def _talon_feature(agent: Any) -> Any | None:
    features = getattr(agent, "features", None)
    if not isinstance(features, dict):
        return None
    talon = features.get("TalonCoordinatorFeature") or features.get("talon")
    if talon is not None:
        return talon
    for feature in features.values():
        if feature.__class__.__name__ == "TalonCoordinatorFeature":
            return feature
        if callable(getattr(feature, "talon_claim", None)):
            return feature
    return None


def _audit_anchor_feature(agent: Any) -> Any | None:
    features = getattr(agent, "features", None)
    if not isinstance(features, dict):
        return None
    audit_anchor = features.get("AuditAnchorFeature") or features.get("audit_anchor")
    if audit_anchor is not None:
        return audit_anchor
    for feature in features.values():
        if feature.__class__.__name__ == "AuditAnchorFeature":
            return feature
        if callable(getattr(feature, "anchor_audit", None)):
            return feature
    return None


__all__ = ["FeatureFeaturesFeature"]
