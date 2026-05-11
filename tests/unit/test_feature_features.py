"""Tests for the FeatureFeature workflow contracts (#1151)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from kestrel_sdk.signals import SignalMode
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
