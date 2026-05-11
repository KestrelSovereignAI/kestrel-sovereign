"""Tests for the FeatureFeature workflow contracts (#1151)."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
from kestrel_sdk.signals import SignalMode
from kestrel_sdk.tools.base import ToolCategory, ToolParameter, ToolSchema
from kestrel_sdk.tools.result import ToolResult, ToolResultStatus

from kestrel_sovereign.features.feature_features.feature import (
    FeatureFeaturesFeature,
)
from kestrel_sovereign.features.feature_features.signals import (
    build_feature_feature_registrations,
)
from kestrel_sovereign.features.feature_features.workflows import (
    FEATURE_FEATURES_COMPENSATION_SOURCES,
    FEATURE_FEATURES_REVIEWER_SOURCES,
    FEATURE_FEATURES_STAGE_ORDER,
    FEATURE_FEATURES_STAGE_SOURCES,
    FEATURE_PROPOSE_PACKAGE_WORKFLOW_NAME,
    FEATURE_PROPOSE_TOOL_WORKFLOW_NAME,
    feature_feature_workflow_payloads,
    feature_propose_package_spec_payload,
    feature_propose_tool_spec_payload,
)
from kestrel_sovereign.features.workflows.models import (
    TriggerKind,
    WorkflowSpec,
)
from kestrel_sovereign.features.workflows.schema import validate_spec_payload
from kestrel_sovereign.signals.registry import SourceRegistry


def _mock_agent_tool(name: str, description: str = "mock tool"):
    return SimpleNamespace(
        name=name,
        schema=ToolSchema(
            name=name,
            description=description,
            category=ToolCategory.UTILITY,
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="query",
                )
            ],
        ),
    )


def _mock_feature(class_name: str, tool_name: str, tools: list):
    cls = type(class_name, (), {})
    feature = cls()
    feature.name = class_name
    feature.tool_name = tool_name
    feature.tool_description = f"{class_name} description"
    feature.get_tools = lambda: tools
    return feature


@pytest.mark.parametrize(
    ("name", "payload"),
    [
        (FEATURE_PROPOSE_TOOL_WORKFLOW_NAME, feature_propose_tool_spec_payload()),
        (
            FEATURE_PROPOSE_PACKAGE_WORKFLOW_NAME,
            feature_propose_package_spec_payload(),
        ),
    ],
)
def test_feature_feature_workflow_payloads_are_valid_unsigned_specs(
    name,
    payload,
):
    validate_spec_payload(payload)
    spec = WorkflowSpec.from_dict(payload)

    assert spec.name == name
    assert spec.version == 1
    assert spec.triggers[0].kind == TriggerKind.MANUAL
    assert [stage.name for stage in spec.stages] == list(
        FEATURE_FEATURES_STAGE_ORDER
    )


def test_feature_feature_workflow_gates_pin_required_review_path():
    spec = WorkflowSpec.from_dict(feature_propose_tool_spec_payload())
    stages = {stage.name: stage for stage in spec.stages}

    assert stages["constitutional_review"].signal_mode == SignalMode.COGNITION
    assert stages["constitutional_review"].gate.type == (
        "constitution_echo_verified"
    )
    assert stages["constitutional_review"].compensate == (
        FEATURE_FEATURES_COMPENSATION_SOURCES["constitutional_review"]
    )
    assert stages["tests_pass"].gate.type == "tests_pass"
    assert stages["lint_clean"].gate.type == "lint_clean"
    assert stages["ci_green"].gate.type == "ci_green"
    assert stages["ci_green"].gate.params["repo_param"] == "repository"
    assert stages["ci_green"].gate.params["branch_param"] == "branch"
    assert "repository" in spec.params_schema["required"]
    assert "branch" in spec.params_schema["required"]
    assert stages["constitutional_boundary_scan"].gate.type == (
        "constitutional_boundary_clean"
    )
    assert stages["red_team_review"].gate.type == "red_team_clear"
    assert stages["red_team_review"].gate.params["reviewer_pool"] == (
        "codex",
        "claude",
    )
    assert stages["council_review"].gate.type == "council_approve"
    assert stages["publish"].irreversible is True
    assert stages["publish"].compensate == "compensate_record_only"
    forbidden = stages["constitutional_boundary_scan"].gate.params[
        "forbidden_modules"
    ]
    assert "kestrel_sovereign.constitution" in forbidden
    assert "kestrel_sovereign.features.constitution" in forbidden


def test_feature_feature_workflow_edges_are_sequential():
    spec = WorkflowSpec.from_dict(feature_propose_package_spec_payload())

    assert [(edge.from_stage, edge.to_stage) for edge in spec.edges] == list(
        zip(FEATURE_FEATURES_STAGE_ORDER, FEATURE_FEATURES_STAGE_ORDER[1:])
    )


def test_feature_feature_payload_selection_and_ci_branch_override():
    payloads = feature_feature_workflow_payloads(
        "tool",
        repository="ExampleOrg/example-repo",
        branch="codex/example",
    )

    assert sorted(payloads) == [FEATURE_PROPOSE_TOOL_WORKFLOW_NAME]
    spec = WorkflowSpec.from_dict(payloads[FEATURE_PROPOSE_TOOL_WORKFLOW_NAME])
    ci_stage = next(stage for stage in spec.stages if stage.name == "ci_green")
    assert ci_stage.gate.params["repo"] == "ExampleOrg/example-repo"
    assert ci_stage.gate.params["branch"] == "codex/example"


@pytest.mark.asyncio
async def test_feature_features_tool_returns_workflow_payloads():
    feature = FeatureFeaturesFeature(SimpleNamespace())
    result = await feature.feature_feature_workflows(kind="package")

    assert result.status is ToolResultStatus.OK
    assert sorted(result.data["workflows"]) == [
        FEATURE_PROPOSE_PACKAGE_WORKFLOW_NAME
    ]


@pytest.mark.asyncio
async def test_feature_features_define_workflows_uses_workflows_feature():
    class StubWorkflowsFeature:
        def __init__(self):
            self.defined = []

        async def workflow_define(self, spec):
            self.defined.append(spec)
            return ToolResult.ok(
                "defined",
                data={"name": spec["name"], "version": spec["version"]},
            )

    workflows = StubWorkflowsFeature()
    feature = FeatureFeaturesFeature(
        SimpleNamespace(features={"WorkflowsFeature": workflows})
    )

    result = await feature.feature_feature_define_workflows(kind="tool")

    assert result.status is ToolResultStatus.OK
    assert [spec["name"] for spec in workflows.defined] == [
        FEATURE_PROPOSE_TOOL_WORKFLOW_NAME
    ]
    assert result.data["defined"][0]["data"]["name"] == (
        FEATURE_PROPOSE_TOOL_WORKFLOW_NAME
    )


@pytest.mark.asyncio
async def test_feature_explore_can_include_loaded_tool_inventory():
    model_feature = _mock_feature(
        "ModelFeature",
        "model_feature",
        [_mock_agent_tool("list_models")],
    )
    agent = SimpleNamespace(features={"ModelFeature": model_feature})
    feature = FeatureFeaturesFeature(agent)

    result = await feature.feature_explore(include_loaded_tools=True)

    assert result.status is ToolResultStatus.OK
    loaded = result.data["loaded_features"]
    assert loaded[0]["class"] == "ModelFeature"
    assert loaded[0]["tools"][0]["name"] == "list_models"
    assert loaded[0]["tools"][0]["estimated_context_tokens"] > 0


@pytest.mark.asyncio
async def test_feature_context_status_reports_hidden_direct_tools():
    agent = SimpleNamespace(
        features={},
        _direct_tools={"list_models": _mock_agent_tool("list_models")},
        _tool_to_feature={"list_models": "model_feature"},
        _tool_context_hidden_tools={"list_models"},
        _tool_context_hidden_features=set(),
    )
    feature = FeatureFeaturesFeature(agent)

    result = await feature.feature_context_status()

    assert result.status is ToolResultStatus.OK
    assert result.data["counts"]["hidden_direct_tools"] == 1
    assert result.data["direct_tools"][0]["visible_in_context"] is False


@pytest.mark.asyncio
async def test_feature_focus_keeps_selected_feature_and_control_surface():
    model_feature = _mock_feature("ModelFeature", "model_feature", [])
    memory_feature = _mock_feature("MemoryFeature", "memory_feature", [])
    feature_feature_tool = _mock_agent_tool("feature_unfocus")
    agent = SimpleNamespace(
        features={
            "FeatureFeaturesFeature": None,
            "ModelFeature": model_feature,
            "MemoryFeature": memory_feature,
        },
        _direct_tools={"feature_unfocus": feature_feature_tool},
        _tool_to_feature={"feature_unfocus": "feature_features_feature"},
    )
    feature = FeatureFeaturesFeature(agent)
    agent.features["FeatureFeaturesFeature"] = feature

    result = await feature.feature_focus(features=["ModelFeature"])

    assert result.status is ToolResultStatus.OK
    hidden = set(result.data["profile"]["hidden_features"])
    assert "memory_feature" in hidden
    assert "model_feature" not in hidden
    assert "feature_features_feature" not in hidden
    assert "feature_unfocus" not in result.data["profile"]["hidden_tools"]


@pytest.mark.asyncio
async def test_feature_focus_accepts_schema_advertised_string_targets():
    model_feature = _mock_feature("ModelFeature", "model_feature", [])
    memory_feature = _mock_feature("MemoryFeature", "memory_feature", [])
    agent = SimpleNamespace(
        features={
            "ModelFeature": model_feature,
            "MemoryFeature": memory_feature,
        }
    )
    feature = FeatureFeaturesFeature(agent)

    result = await feature.feature_focus(features="ModelFeature")

    assert result.status is ToolResultStatus.OK
    hidden = set(result.data["profile"]["hidden_features"])
    assert "memory_feature" in hidden
    assert "model_feature" not in hidden


@pytest.mark.asyncio
async def test_feature_focus_on_tool_keeps_owning_feature_unhidden():
    model_feature = _mock_feature("ModelFeature", "model_feature", [])
    memory_feature = _mock_feature("MemoryFeature", "memory_feature", [])
    agent = SimpleNamespace(
        features={
            "ModelFeature": model_feature,
            "MemoryFeature": memory_feature,
        },
        _direct_tools={"list_models": _mock_agent_tool("list_models")},
        _tool_to_feature={"list_models": "model_feature"},
    )
    feature = FeatureFeaturesFeature(agent)

    result = await feature.feature_focus(tools=["list_models"])

    hidden = set(result.data["profile"]["hidden_features"])
    assert "model_feature" not in hidden
    assert "memory_feature" in hidden
    assert result.data["direct_tools"][0]["visible_in_context"] is True


@pytest.mark.asyncio
async def test_feature_unfocus_hides_and_resets_context_profile():
    model_feature = _mock_feature("ModelFeature", "model_feature", [])
    agent = SimpleNamespace(features={"ModelFeature": model_feature})
    feature = FeatureFeaturesFeature(agent)

    hidden = await feature.feature_unfocus(features=["ModelFeature"])
    reset = await feature.feature_unfocus(reset=True)

    assert hidden.status is ToolResultStatus.OK
    assert hidden.data["profile"]["hidden_features"] == ["model_feature"]
    assert reset.status is ToolResultStatus.OK
    assert reset.data["profile"]["hidden_features"] == []


@pytest.mark.asyncio
async def test_feature_unfocus_parses_schema_advertised_reset_string():
    model_feature = _mock_feature("ModelFeature", "model_feature", [])
    agent = SimpleNamespace(features={"ModelFeature": model_feature})
    feature = FeatureFeaturesFeature(agent)

    false_reset = await feature.feature_unfocus(reset="false")
    await feature.feature_unfocus(features="ModelFeature")
    true_reset = await feature.feature_unfocus(reset="true")

    assert false_reset.status is ToolResultStatus.ERROR
    assert false_reset.error == "Provide features/tools to hide, or reset=True."
    assert true_reset.status is ToolResultStatus.OK
    assert true_reset.data["profile"]["hidden_features"] == []


@pytest.mark.asyncio
async def test_feature_focus_refreshes_cached_feature_prompt():
    model_feature = _mock_feature("ModelFeature", "model_feature", [])
    agent = SimpleNamespace(
        features={"ModelFeature": model_feature},
        _cached_features_prompt="stale",
    )
    agent._build_features_prompt_section = lambda: "fresh"
    feature = FeatureFeaturesFeature(agent)

    result = await feature.feature_focus(features="ModelFeature")

    assert result.status is ToolResultStatus.OK
    assert agent._cached_features_prompt == "fresh"


@pytest.mark.asyncio
async def test_feature_features_define_workflows_stops_on_define_failure():
    class StubWorkflowsFeature:
        async def workflow_define(self, spec):
            return SimpleNamespace(
                status=ToolResultStatus.ERROR,
                data=None,
                error=f"boom:{spec['name']}",
            )

    feature = FeatureFeaturesFeature(
        SimpleNamespace(features={"WorkflowsFeature": StubWorkflowsFeature()})
    )

    result = await feature.feature_feature_define_workflows(kind="tool")

    assert result.status is ToolResultStatus.ERROR
    assert result.data["defined"][0]["error"] == (
        f"boom:{FEATURE_PROPOSE_TOOL_WORKFLOW_NAME}"
    )


@pytest.mark.asyncio
async def test_feature_features_run_delegates_to_workflows_feature():
    class StubWorkflowsFeature:
        def __init__(self):
            self.calls = []

        async def workflow_run(self, name, params=None, version=0):
            self.calls.append((name, params, version))
            return ToolResult.ok("ran", data={"run_id": "run-1"})

    workflows = StubWorkflowsFeature()
    feature = FeatureFeaturesFeature(
        SimpleNamespace(features={"WorkflowsFeature": workflows})
    )

    result = await feature.feature_feature_run(
        "package",
        params={
            "feature_name": "demo",
            "package_name": "kestrel-demo",
            "repository": "Org/repo",
            "branch": "codex/demo",
        },
        version=2,
    )

    assert result.status is ToolResultStatus.OK
    assert workflows.calls == [
        (
            FEATURE_PROPOSE_PACKAGE_WORKFLOW_NAME,
            {
                "feature_name": "demo",
                "package_name": "kestrel-demo",
                "repository": "Org/repo",
                "branch": "codex/demo",
            },
            2,
        )
    ]


@pytest.mark.asyncio
async def test_feature_features_run_rejects_unknown_kind_and_non_object_params():
    feature = FeatureFeaturesFeature(
        SimpleNamespace(features={"WorkflowsFeature": SimpleNamespace()})
    )

    bad_kind = await feature.feature_feature_run("all", params={})
    bad_params = await feature.feature_feature_run("tool", params=[])

    assert bad_kind.status is ToolResultStatus.ERROR
    assert bad_params.status is ToolResultStatus.ERROR


@pytest.mark.asyncio
async def test_feature_feature_runtime_status_reports_missing_action_providers():
    registry = SourceRegistry()
    agent = SimpleNamespace(signal_registry=registry)
    feature = FeatureFeaturesFeature(agent)
    await feature.initialize()

    result = await feature.feature_feature_runtime_status()

    assert result.status is ToolResultStatus.ERROR
    assert result.data["missing_registered"] == []
    assert "feature_features.file_github_epic" in (
        result.data["missing_action_providers"]
    )
    explore = next(
        row
        for row in result.data["sources"]
        if row["name"] == "feature_features.explore"
    )
    assert explore["ready"] is True


@pytest.mark.asyncio
async def test_feature_feature_runtime_status_passes_with_registered_providers():
    registry = SourceRegistry()
    agent = SimpleNamespace(signal_registry=registry)
    for source in (
        *FEATURE_FEATURES_STAGE_SOURCES.values(),
        *FEATURE_FEATURES_COMPENSATION_SOURCES.values(),
    ):
        stage = source.split(".")[-1]
        if stage in {"explore", "design_plan", "constitutional_review"}:
            continue

        async def provider(payload, *, stage=stage):
            return {"status": "ok", "stage": stage, "payload": payload}

        setattr(agent, f"feature_feature_{stage}", provider)

    feature = FeatureFeaturesFeature(agent)
    await feature.initialize()

    result = await feature.feature_feature_runtime_status()

    assert result.status is ToolResultStatus.OK
    assert result.data["ready"] is True
    assert result.data["missing_action_providers"] == []


@pytest.mark.asyncio
async def test_feature_feature_runtime_status_accepts_pre_registered_handler():
    registry = SourceRegistry()
    agent = SimpleNamespace(signal_registry=registry)
    registration = next(
        reg
        for reg in build_feature_feature_registrations(agent)
        if reg.name == "feature_features.file_github_epic"
    )

    async def handler(payload):
        return {"status": "ok", "payload": payload}

    registry.register(replace(registration, handler=handler))
    feature = FeatureFeaturesFeature(agent)
    await feature.initialize()

    result = await feature.feature_feature_runtime_status()

    row = next(
        item
        for item in result.data["sources"]
        if item["name"] == "feature_features.file_github_epic"
    )
    assert row["registered_handler"] is True
    assert "feature_features.file_github_epic" not in (
        result.data["missing_action_providers"]
    )


@pytest.mark.asyncio
async def test_feature_feature_runtime_status_rejects_required_explore_handler():
    registry = SourceRegistry()
    agent = SimpleNamespace(signal_registry=registry)
    registration = next(
        reg
        for reg in build_feature_feature_registrations(agent)
        if reg.name == "feature_features.explore"
    )

    async def handler(payload):
        return {"status": "ok", "payload": payload}

    setattr(handler, "_feature_feature_requires_agent_provider", True)
    registry.register(replace(registration, handler=handler))
    feature = FeatureFeaturesFeature(agent)
    await feature.initialize()

    result = await feature.feature_feature_runtime_status()

    assert "feature_features.explore" in (
        result.data["missing_action_providers"]
    )


@pytest.mark.asyncio
async def test_feature_feature_runtime_status_reports_missing_prompt(tmp_path):
    registry = SourceRegistry()
    agent = SimpleNamespace(signal_registry=registry)
    registration = next(
        reg
        for reg in build_feature_feature_registrations(agent)
        if reg.name == "feature_features.design_plan"
    )
    registry.register(
        replace(registration, prompt_template=tmp_path / "missing.md")
    )
    feature = FeatureFeaturesFeature(agent)
    await feature.initialize()

    result = await feature.feature_feature_runtime_status()

    assert result.status is ToolResultStatus.ERROR
    assert "feature_features.design_plan" in (
        result.data["missing_cognition_prompts"]
    )


@pytest.mark.asyncio
async def test_feature_explore_lists_feature_features_itself():
    feature = FeatureFeaturesFeature(SimpleNamespace())
    result = await feature.feature_explore()

    assert result.status is ToolResultStatus.OK
    names = {row["name"] for row in result.data["core_features"]}
    classes = {row["class"] for row in result.data["core_features"]}
    assert "feature_features" in names
    assert "FeatureFeaturesFeature" in classes


@pytest.mark.asyncio
async def test_feature_features_initializes_workflow_signal_sources():
    registry = SourceRegistry()
    agent = SimpleNamespace(signal_registry=registry)
    feature = FeatureFeaturesFeature(agent)

    await feature.initialize()

    for registration in build_feature_feature_registrations(agent):
        assert registry.get(registration.name) is not None
    constitutional = registry.get("feature_features.constitutional_review")
    assert constitutional is not None
    assert constitutional.require_constitution_echo is True
    assert constitutional.constitution_injection == "full"
    assert constitutional.prompt_template is not None
    assert constitutional.prompt_template.exists()
    assert "feature_features/prompts" in constitutional.prompt_template.as_posix()
    for source in FEATURE_FEATURES_REVIEWER_SOURCES.values():
        reviewer = registry.get(source)
        assert reviewer is not None
        assert reviewer.prompt_template is not None
        assert reviewer.prompt_template.exists()
