"""Tests for the FeatureFeature workflow contracts (#1151)."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from jsonschema import Draft202012Validator, ValidationError
from kestrel_sdk.features.base import Feature as SDKBaseFeature
from kestrel_sdk.signals import SignalMode
from kestrel_sdk.tools.base import ToolCategory, ToolParameter, ToolSchema
from kestrel_sdk.tools.result import ToolResult, ToolResultStatus

from kestrel_sovereign.features.feature_features.feature import (
    FeatureFeaturesFeature,
)
from kestrel_sovereign.features.base import Feature
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
from kestrel_sovereign.features.workflows.feature import WorkflowsFeature
from kestrel_sovereign.features.workflows.schema import validate_spec_payload
from kestrel_sovereign.identity.runtime_identity import AgentIdentity
from kestrel_sovereign.security.crypto_suite import (
    ALG_ECDSA_SECP256K1_SHA256,
    get_suite,
)
from kestrel_sovereign.signals import (
    OrderedLockManager,
    SignalDispatcher,
    SignalLogStore,
)
from kestrel_sovereign.signals.constitution_canary import CanaryStatus
from kestrel_sovereign.signals.registry import SourceRegistry
from kestrel_sovereign.storage.db import SQLiteBackend


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


class RuntimeDemoFeature(Feature):
    @property
    def tool_description(self):
        return "Runtime demo feature"

    async def initialize(self):
        self.initialized = True

    async def on_enable(self):
        self.enabled = True

    async def on_disable(self):
        self.disabled = True

    async def shutdown(self):
        self.shutdown_called = True


class RouterDemoFeature(RuntimeDemoFeature):
    def get_router(self):
        return object()


class ExternalSdkDemoFeature(SDKBaseFeature):
    @property
    def tool_description(self):
        return "External SDK demo feature"

    async def initialize(self):
        self.initialized = True


class ExternalSdkToolOnlyFeature(SDKBaseFeature):
    name = "ExternalSdkToolOnlyFeature"
    tool_name = "external_sdk_tool_only"

    @property
    def tool_description(self):
        return "External SDK tool-only demo feature"

    async def initialize(self):
        return None

    def get_tools(self):
        return [_mock_agent_tool("github_issue_fetch", "Fetch a GitHub issue")]


def _identity(did: str = "did:web:feature-feature.example") -> AgentIdentity:
    suite = get_suite(ALG_ECDSA_SECP256K1_SHA256)
    return AgentIdentity(
        legacy_did=did,
        legacy_keypair=suite.generate_keypair(),
        legacy_did_document={},
    )


class _FeatureFeatureSmokeAgent:
    def __init__(self, backend: SQLiteBackend):
        self.identity = _identity()
        self.did = self.identity.legacy_did
        self.storage = SimpleNamespace(db=backend)
        self.background_tasks: list[asyncio.Task] = []
        self.signal_registry = SourceRegistry()
        self.dispatcher = None
        self.process_input_prompts: list[str] = []

    async def process_input(self, prompt: str, **kwargs):
        del kwargs
        self.process_input_prompts.append(prompt)
        reviewer = _reviewer_from_prompt(prompt)
        if reviewer is None:
            return "ok"
        canary = _red_team_canary_from_prompt(prompt)
        family = reviewer
        return json.dumps(
            {
                "canary": canary,
                "reviewer": reviewer,
                "model_family": family,
                "model": f"{family}-review-model",
                "blockers": [],
                "tokens": 1,
                "cost_usd": 0.0,
            }
        )

    def _track_background_task(self, coro, *, name: str):
        task = asyncio.create_task(coro, name=name)
        self.background_tasks.append(task)
        return task

    def get_constitution_hash(self) -> str:
        return "a" * 64

    def _get_governing_constitution(self) -> str:
        return "Article I. Test constitution."

    def get_anchored_doctrine_files(self) -> dict:
        return {}

    def verify_constitution_echo(self, **kwargs) -> CanaryStatus:
        del kwargs
        return CanaryStatus.VERIFIED


class _SmokeTalonFeature:
    def __init__(self):
        self.claims: list[dict] = []

    async def talon_claim(self, **kwargs):
        self.claims.append(kwargs)
        return ToolResult.ok(
            "claimed",
            data={
                "job_id": "talon-job-1",
                "issue_number": kwargs["issue"],
                "status": "dispatched",
            },
        )

    async def talon_status(self):
        return ToolResult.ok(
            "status",
            data={
                "jobs": [
                    {
                        "id": "talon-job-1",
                        "status": "complete",
                    }
                ]
            },
        )


class _SmokeAuditAnchorFeature:
    def __init__(self):
        self.calls = 0

    async def anchor_audit(self):
        self.calls += 1
        return ToolResult.ok(
            "anchored",
            data={"anchor_id": "anchor-1", "root_hash": "b" * 64},
        )


def _reviewer_from_prompt(prompt: str) -> str | None:
    match = re.search(r"'reviewer': '([^']+)'", prompt)
    if match is None:
        return None
    return match.group(1)


def _red_team_canary_from_prompt(prompt: str) -> str:
    match = re.search(r"'canary': '([a-f0-9]{64})'", prompt)
    assert match is not None
    return match.group(1)


async def _smoke_command_runner(command, cwd, timeout):
    del cwd, timeout
    if command[:2] == ["git", "merge-base"]:
        return {"exit_code": 0, "stdout": "base-sha\n", "stderr": ""}
    if command[:3] == ["git", "diff", "--no-ext-diff"]:
        return {
            "exit_code": 0,
            "stdout": (
                "diff --git a/kestrel_sovereign/features/demo.py "
                "b/kestrel_sovereign/features/demo.py\n"
                "new file mode 100644\n"
                "--- /dev/null\n"
                "+++ b/kestrel_sovereign/features/demo.py\n"
                "@@ -0,0 +1,2 @@\n"
                "+def demo_tool():\n"
                "+    return 'ok'\n"
            ),
            "stderr": "",
        }
    if command[:3] == ["git", "ls-files", "--others"]:
        return {"exit_code": 0, "stdout": "", "stderr": ""}
    return {"exit_code": 0, "stdout": "ok\n", "stderr": ""}


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
    assert "issue_number" in spec.params_schema["properties"]
    assert "talon_issue_numbers" in spec.params_schema["properties"]
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


def test_feature_feature_workflow_params_allow_talon_assignment_inputs():
    tool = WorkflowSpec.from_dict(feature_propose_tool_spec_payload())
    package = WorkflowSpec.from_dict(feature_propose_package_spec_payload())

    Draft202012Validator(tool.params_schema).validate(
        {
            "feature_name": "DemoFeature",
            "target_tool_name": "demo_tool",
            "repository": "Org/repo",
            "branch": "codex/demo",
            "talon_issue_numbers": [7, 8],
            "issues": [{"issue": 10}],
            "chunks": [{"number": 9}],
            "talon_backend": "codex",
            "talon_model": "gpt-5.5",
            "skip_clarification": True,
            "worktree": True,
            "self_review": False,
        }
    )
    Draft202012Validator(package.params_schema).validate(
        {
            "package_name": "kestrel-demo",
            "repository": "Org/repo",
            "branch": "codex/demo",
            "issue_number": 7,
            "max_iterations": 3,
            "max_turns": 5,
        }
    )
    with pytest.raises(ValidationError):
        Draft202012Validator(tool.params_schema).validate(
            {
                "feature_name": "DemoFeature",
                "target_tool_name": "demo_tool",
                "repository": "Org/repo",
                "branch": "codex/demo",
                "issue_number": True,
            }
        )


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
async def test_feature_discover_returns_searchable_catalog_with_provenance():
    module_path = RuntimeDemoFeature.__module__
    agent = SimpleNamespace(features={"RuntimeDemoFeature": RuntimeDemoFeature(None)})
    feature = FeatureFeaturesFeature(agent)

    with (
        patch(
            "kestrel_sovereign.features.feature_features.feature.discover_feature_modules",
            return_value=[module_path],
        ),
        patch(
            "kestrel_sovereign.features.feature_features.feature.find_feature_class",
            return_value=RuntimeDemoFeature,
        ),
        patch(
            "kestrel_sovereign.features.feature_features.feature.discover_entrypoint_feature_classes",
            return_value={},
        ),
    ):
        result = await feature.feature_discover(query="runtime demo")

    assert result.status is ToolResultStatus.OK
    assert result.data["actions"]["load"] == "feature_add"
    assert result.data["counts"]["matched"] == 1
    row = result.data["features"][0]
    assert row["class"] == "RuntimeDemoFeature"
    assert row["source"] == "core"
    assert row["provenance"] == module_path
    assert row["loaded"] is True
    assert row["visible_in_context"] is True
    assert row["allowed"] is True
    assert row["disabled"] is False
    assert row["docs"]


@pytest.mark.asyncio
async def test_feature_discover_includes_runtime_context_and_tool_rows():
    model_feature = _mock_feature(
        "ModelFeature",
        "model_feature",
        [_mock_agent_tool("list_models", "List available LLM models")],
    )
    agent = SimpleNamespace(
        features={"ModelFeature": model_feature},
        _tool_context_hidden_features={"model_feature"},
        _tool_context_hidden_tools=set(),
    )
    feature = FeatureFeaturesFeature(agent)

    with (
        patch(
            "kestrel_sovereign.features.feature_features.feature.discover_feature_modules",
            return_value=[],
        ),
        patch(
            "kestrel_sovereign.features.feature_features.feature.discover_entrypoint_feature_classes",
            return_value={},
        ),
    ):
        result = await feature.feature_discover(query="list_models", include_tools=True)

    assert result.status is ToolResultStatus.OK
    row = result.data["features"][0]
    assert row["source"] == "runtime"
    assert row["loaded"] is True
    assert row["visible_in_context"] is False
    assert row["tools"][0]["name"] == "list_models"
    assert row["tools_discoverable"] is True
    assert row["direct_tool_invokable"] is False
    assert row["subagent_capable"] is False
    assert row["subagent_executable"] is False
    assert row["workflow_invocation_modes"] == []
    assert row["subagent_workflow_executable"] is False
    assert result.data["counts"]["hidden_from_context"] == 1


@pytest.mark.asyncio
async def test_feature_discover_distinguishes_sdk_tool_only_from_subagent():
    sdk_feature = ExternalSdkToolOnlyFeature(SimpleNamespace())
    agent = SimpleNamespace(features={"GitHubFeature": sdk_feature})
    feature = FeatureFeaturesFeature(agent)

    with (
        patch(
            "kestrel_sovereign.features.feature_features.feature.discover_feature_modules",
            return_value=[],
        ),
        patch(
            "kestrel_sovereign.features.feature_features.feature.discover_entrypoint_feature_classes",
            return_value={},
        ),
    ):
        result = await feature.feature_discover(query="github", include_tools=True)

    assert result.status is ToolResultStatus.OK
    row = result.data["features"][0]
    assert row["class"] == "ExternalSdkToolOnlyFeature"
    assert row["tools_discoverable"] is True
    assert row["direct_tool_invokable"] is False
    assert row["tools"][0]["name"] == "github_issue_fetch"
    assert row["subagent_capable"] is False
    assert row["subagent_executable"] is False
    assert row["subagent_workflow_executable"] is False
    assert row["workflow_invocation_modes"] == []


@pytest.mark.asyncio
async def test_feature_discover_marks_promoted_direct_tools_invokable():
    sdk_feature = ExternalSdkToolOnlyFeature(SimpleNamespace())
    agent = SimpleNamespace(
        features={"GitHubFeature": sdk_feature},
        _direct_tools={"github_issue_fetch": _mock_agent_tool("github_issue_fetch")},
        _tool_to_feature={"github_issue_fetch": "external_sdk_tool_only"},
    )
    feature = FeatureFeaturesFeature(agent)

    with (
        patch(
            "kestrel_sovereign.features.feature_features.feature.discover_feature_modules",
            return_value=[],
        ),
        patch(
            "kestrel_sovereign.features.feature_features.feature.discover_entrypoint_feature_classes",
            return_value={},
        ),
    ):
        result = await feature.feature_discover(query="github", include_tools=True)

    assert result.status is ToolResultStatus.OK
    row = result.data["features"][0]
    assert row["tools_discoverable"] is True
    assert row["direct_tool_invokable"] is True
    assert row["subagent_executable"] is False
    assert row["workflow_invocation_modes"] == ["direct_tool"]


@pytest.mark.asyncio
async def test_feature_discover_searches_tools_without_returning_tool_rows_by_default():
    model_feature = _mock_feature(
        "ModelFeature",
        "model_feature",
        [_mock_agent_tool("list_models", "List available LLM models")],
    )
    agent = SimpleNamespace(features={"ModelFeature": model_feature})
    feature = FeatureFeaturesFeature(agent)

    with (
        patch(
            "kestrel_sovereign.features.feature_features.feature.discover_feature_modules",
            return_value=[],
        ),
        patch(
            "kestrel_sovereign.features.feature_features.feature.discover_entrypoint_feature_classes",
            return_value={},
        ),
    ):
        result = await feature.feature_discover(query="list_models")

    assert result.status is ToolResultStatus.OK
    assert result.data["counts"]["matched"] == 1
    assert result.data["features"][0]["class"] == "ModelFeature"
    assert result.data["features"][0]["tools"] == []


@pytest.mark.asyncio
async def test_feature_discover_accepts_string_limit_from_tool_schema():
    agent = SimpleNamespace(features={})
    feature = FeatureFeaturesFeature(agent)

    with (
        patch(
            "kestrel_sovereign.features.feature_features.feature.discover_feature_modules",
            return_value=[],
        ),
        patch(
            "kestrel_sovereign.features.feature_features.feature.discover_entrypoint_feature_classes",
            return_value={},
        ),
    ):
        result = await feature.feature_discover(limit="10")

    assert result.status is ToolResultStatus.OK


@pytest.mark.asyncio
async def test_feature_discover_loaded_only_filters_unloaded_core_features():
    module_path = RuntimeDemoFeature.__module__
    feature = FeatureFeaturesFeature(SimpleNamespace(features={}))

    with (
        patch(
            "kestrel_sovereign.features.feature_features.feature.discover_feature_modules",
            return_value=[module_path],
        ),
        patch(
            "kestrel_sovereign.features.feature_features.feature.find_feature_class",
            return_value=RuntimeDemoFeature,
        ),
        patch(
            "kestrel_sovereign.features.feature_features.feature.discover_entrypoint_feature_classes",
            return_value={},
        ),
    ):
        result = await feature.feature_discover(loaded_only="true")

    assert result.status is ToolResultStatus.OK
    assert result.data["features"] == []


def test_feature_discover_command_parser_keeps_multi_word_query():
    feature = FeatureFeaturesFeature(SimpleNamespace())
    tools = {tool.name: tool for tool in feature.get_tools()}

    args = tools["feature_discover"].parse_command_args(
        "!feature-discover runtime demo include_tools=true limit=3 --loaded-only"
    )

    assert args == {
        "query": "runtime demo",
        "include_tools": True,
        "limit": 3,
        "loaded_only": True,
    }


def test_feature_lifecycle_command_parsers_accept_agent_friendly_forms():
    feature = FeatureFeaturesFeature(SimpleNamespace())
    tools = {tool.name: tool for tool in feature.get_tools()}

    add_args = tools["feature_add"].parse_command_args(
        '!feature-add "Runtime Demo Feature" --pre-explore'
    )
    remove_args = tools["feature_remove"].parse_command_args(
        "!feature-remove feature=RuntimeDemoFeature"
    )
    focus_args = tools["feature_focus"].parse_command_args(
        "!feature-focus --feature ModelFeature MemoryFeature --tool list_models"
    )
    unfocus_args = tools["feature_unfocus"].parse_command_args(
        "!feature-unfocus features=model_feature,memory_feature --reset"
    )
    dashed_add_args = tools["feature_add"].parse_command_args(
        "!feature-add RuntimeDemoFeature --pre-explore=true"
    )
    dashed_focus_args = tools["feature_focus"].parse_command_args(
        "!feature-focus --features=ModelFeature --tools=list_models"
    )

    assert add_args == {
        "feature": "Runtime Demo Feature",
        "pre_explore": True,
    }
    assert remove_args == {"feature": "RuntimeDemoFeature"}
    assert focus_args == {
        "features": ["ModelFeature", "MemoryFeature"],
        "tools": ["list_models"],
    }
    assert unfocus_args == {
        "features": ["memory_feature", "model_feature"],
        "reset": True,
    }
    assert dashed_add_args == {
        "feature": "RuntimeDemoFeature",
        "pre_explore": True,
    }
    assert dashed_focus_args == {
        "features": ["ModelFeature"],
        "tools": ["list_models"],
    }


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
    assert result.data["direct_tools"][0]["invocation_mode"] == "direct_tool"
    assert result.data["direct_tools"][0]["invokable"] is True


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
async def test_feature_add_loads_discoverable_runtime_feature():
    agent = SimpleNamespace(
        features={},
        _cached_features_prompt="stale",
    )
    agent._build_features_prompt_section = lambda: "fresh"

    async def register_feature(instance):
        await instance.initialize()
        agent.features[instance.name] = instance
        await instance.on_enable()

    agent._register_feature = register_feature
    feature = FeatureFeaturesFeature(agent)

    with (
        patch(
            "kestrel_sovereign.features.feature_features.feature.discover_feature_class_by_name",
            return_value=RuntimeDemoFeature,
        ),
        patch(
            "kestrel_sovereign.features.feature_features.feature.get_disabled_features",
            return_value=set(),
        ),
    ):
        result = await feature.feature_add("runtime-demo")

    assert result.status is ToolResultStatus.OK
    assert "RuntimeDemoFeature" in agent.features
    assert agent.features["RuntimeDemoFeature"].initialized is True
    assert agent.features["RuntimeDemoFeature"].enabled is True
    assert agent._cached_features_prompt == "fresh"
    assert result.data["feature"]["class"] == "RuntimeDemoFeature"


@pytest.mark.asyncio
async def test_feature_add_can_pre_explore_loaded_feature_tools():
    tool = _mock_agent_tool("runtime_status")
    agent = SimpleNamespace(features={})
    registered = []

    async def register_feature(instance):
        instance.get_tools = lambda: [tool]
        await instance.initialize()
        agent.features[instance.name] = instance

    agent._register_feature = register_feature
    agent._register_explored_feature_tools = lambda instance: registered.append(
        instance.tool_name
    )
    feature = FeatureFeaturesFeature(agent)

    with (
        patch(
            "kestrel_sovereign.features.feature_features.feature.discover_feature_class_by_name",
            return_value=RuntimeDemoFeature,
        ),
        patch(
            "kestrel_sovereign.features.feature_features.feature.get_disabled_features",
            return_value=set(),
        ),
    ):
        result = await feature.feature_add("RuntimeDemoFeature", pre_explore="true")

    assert result.status is ToolResultStatus.OK
    assert registered == ["runtime_demo_feature"]


@pytest.mark.asyncio
async def test_feature_add_validates_pre_explore_before_loading():
    agent = SimpleNamespace(features={})
    called = False

    async def register_feature(instance):
        nonlocal called
        called = True
        agent.features[instance.name] = instance

    agent._register_feature = register_feature
    feature = FeatureFeaturesFeature(agent)

    result = await feature.feature_add("RuntimeDemoFeature", pre_explore="maybe")

    assert result.status is ToolResultStatus.ERROR
    assert result.error == "pre_explore must be a boolean."
    assert called is False
    assert agent.features == {}


@pytest.mark.asyncio
async def test_feature_add_respects_agent_allowed_feature_profile():
    agent = SimpleNamespace(
        features={},
        _allowed_features={"OtherFeature"},
    )
    feature = FeatureFeaturesFeature(agent)

    with (
        patch(
            "kestrel_sovereign.features.feature_features.feature.discover_feature_class_by_name",
            return_value=RuntimeDemoFeature,
        ),
        patch(
            "kestrel_sovereign.features.feature_features.feature.get_disabled_features",
            return_value=set(),
        ),
    ):
        result = await feature.feature_add("RuntimeDemoFeature")

    assert result.status is ToolResultStatus.ERROR
    assert "not allowed" in result.error
    assert agent.features == {}


@pytest.mark.asyncio
async def test_feature_add_notifies_existing_features_after_runtime_load():
    class ExistingFeature(RuntimeDemoFeature):
        async def post_all_features_loaded(self, agent):
            self.post_loaded_count = getattr(self, "post_loaded_count", 0) + 1

    existing = ExistingFeature(SimpleNamespace())
    agent = SimpleNamespace(features={existing.name: existing})

    async def register_feature(instance):
        await instance.initialize()
        agent.features[instance.name] = instance

    agent._register_feature = register_feature
    feature = FeatureFeaturesFeature(agent)

    with (
        patch(
            "kestrel_sovereign.features.feature_features.feature.discover_feature_class_by_name",
            return_value=RuntimeDemoFeature,
        ),
        patch(
            "kestrel_sovereign.features.feature_features.feature.get_disabled_features",
            return_value=set(),
        ),
    ):
        result = await feature.feature_add("RuntimeDemoFeature")

    assert result.status is ToolResultStatus.OK
    assert existing.post_loaded_count == 1


@pytest.mark.asyncio
async def test_feature_add_rolls_back_when_post_load_notification_fails():
    agent = SimpleNamespace(features={})
    registered = []

    async def register_feature(instance):
        registered.append(instance)
        await instance.initialize()
        agent.features[instance.name] = instance
        await instance.on_enable()

    agent._register_feature = register_feature
    feature = FeatureFeaturesFeature(agent)

    with (
        patch(
            "kestrel_sovereign.features.feature_features.feature.discover_feature_class_by_name",
            return_value=RuntimeDemoFeature,
        ),
        patch(
            "kestrel_sovereign.features.feature_features.feature.get_disabled_features",
            return_value=set(),
        ),
        patch(
            "kestrel_sovereign.features.feature_features.feature._notify_features_loaded",
            side_effect=RuntimeError("post-load boom"),
        ),
    ):
        result = await feature.feature_add("RuntimeDemoFeature")

    assert result.status is ToolResultStatus.ERROR
    assert "post-load boom" in result.error
    assert agent.features == {}
    assert registered[0].disabled is True
    assert registered[0].shutdown_called is True


@pytest.mark.asyncio
async def test_feature_add_rejects_router_backed_features():
    agent = SimpleNamespace(features={})
    feature = FeatureFeaturesFeature(agent)

    with (
        patch(
            "kestrel_sovereign.features.feature_features.feature.discover_feature_class_by_name",
            return_value=RouterDemoFeature,
        ),
        patch(
            "kestrel_sovereign.features.feature_features.feature.get_disabled_features",
            return_value=set(),
        ),
    ):
        result = await feature.feature_add("RouterDemoFeature")

    assert result.status is ToolResultStatus.ERROR
    assert "exposes HTTP routes" in result.error
    assert agent.features == {}


@pytest.mark.asyncio
async def test_feature_remove_unloads_feature_and_refreshes_context():
    agent = SimpleNamespace(
        features={},
        _cached_features_prompt="stale",
    )
    runtime = RuntimeDemoFeature(agent)
    agent.features[runtime.name] = runtime
    agent._build_features_prompt_section = lambda: "fresh"
    feature = FeatureFeaturesFeature(agent)

    result = await feature.feature_remove("runtime-demo")

    assert result.status is ToolResultStatus.OK
    assert "RuntimeDemoFeature" not in agent.features
    assert runtime.disabled is True
    assert runtime.shutdown_called is True
    assert agent._cached_features_prompt == "fresh"


@pytest.mark.asyncio
async def test_feature_remove_rejects_router_backed_features():
    agent = SimpleNamespace(features={})
    runtime = RouterDemoFeature(agent)
    agent.features[runtime.name] = runtime
    feature = FeatureFeaturesFeature(agent)

    result = await feature.feature_remove("router-demo")

    assert result.status is ToolResultStatus.ERROR
    assert "exposes HTTP routes" in result.error
    assert agent.features[runtime.name] is runtime


@pytest.mark.asyncio
async def test_feature_remove_allows_external_sdk_features_without_router_override():
    agent = SimpleNamespace(features={})
    runtime = ExternalSdkDemoFeature(agent)
    agent.features[runtime.__class__.__name__] = runtime
    feature = FeatureFeaturesFeature(agent)

    result = await feature.feature_remove("external-sdk-demo")

    assert result.status is ToolResultStatus.OK
    assert agent.features == {}


@pytest.mark.asyncio
async def test_feature_remove_rejects_self_removal():
    agent = SimpleNamespace(features={})
    feature = FeatureFeaturesFeature(agent)
    agent.features[feature.name] = feature

    result = await feature.feature_remove("feature-features")

    assert result.status is ToolResultStatus.ERROR
    assert "cannot remove itself" in result.error


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
async def test_feature_features_run_accepts_json_string_params_and_version():
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
        "tool",
        params=json.dumps(
            {
                "feature_name": "demo",
                "target_tool_name": "demo_tool",
                "repository": "Org/repo",
                "branch": "codex/demo",
            }
        ),
        version="0",
    )

    assert result.status is ToolResultStatus.OK
    assert workflows.calls == [
        (
            FEATURE_PROPOSE_TOOL_WORKFLOW_NAME,
            {
                "feature_name": "demo",
                "target_tool_name": "demo_tool",
                "repository": "Org/repo",
                "branch": "codex/demo",
            },
            0,
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
    assert "object" in bad_params.error


@pytest.mark.asyncio
async def test_feature_feature_workflow_runs_end_to_end_with_default_providers(
    tmp_path,
    monkeypatch,
):
    backend = SQLiteBackend(str(tmp_path / "feature-feature-smoke.db"))
    await backend.connect()
    agent = _FeatureFeatureSmokeAgent(backend)
    talon = _SmokeTalonFeature()
    audit_anchor = _SmokeAuditAnchorFeature()
    agent.workflow_red_team_prompt_pack_resolver = lambda _constraint: {
        "name": "kestrel-red-team-prompts",
        "version": "1.0.0",
        "prompt_hash": "c" * 64,
    }
    agent.workflow_red_team_attestation_resolver = lambda reviewer: {
        "model_family": reviewer,
        "constitution_hash": agent.get_constitution_hash(),
    }
    agent.workflow_council_approve_provider = lambda *args: {
        "approved_dids": ["did:kestrel:one", "did:kestrel:two"]
    }
    agent.feature_feature_command_runner = _smoke_command_runner

    signal_store = SignalLogStore(backend)
    await signal_store.initialize()
    agent.dispatcher = SignalDispatcher(
        agent=agent,
        registry=agent.signal_registry,
        lock_manager=OrderedLockManager(),
        store=signal_store,
    )
    workflows = WorkflowsFeature(agent)
    feature = FeatureFeaturesFeature(agent)
    agent.features = {
        "WorkflowsFeature": workflows,
        "FeatureFeaturesFeature": feature,
        "TalonCoordinatorFeature": talon,
        "AuditAnchorFeature": audit_anchor,
    }

    created_issues: list[dict] = []
    merged_prs: list[dict] = []

    async def create_issue(repository, token, body):
        created_issues.append(
            {"repository": repository, "token": token, "body": body}
        )
        return {
            "number": 42,
            "html_url": "https://github.com/Org/repo/issues/42",
        }

    async def find_open_pr(repository, branch, token):
        return {
            "number": 17,
            "html_url": f"https://github.com/{repository}/pull/17",
            "head": {"sha": "head-sha-1", "ref": branch},
        }

    async def merge_pr(repository, pr_number, token, body):
        merged_prs.append(
            {
                "repository": repository,
                "pr_number": pr_number,
                "token": token,
                "body": body,
            }
        )
        return {"merged": True, "sha": "merge-sha-1"}

    monkeypatch.setattr(
        "kestrel_sovereign.features.feature_features.feature._github_token",
        lambda: "token-1",
    )
    monkeypatch.setattr(
        "kestrel_sovereign.features.feature_features.feature._ci_green_token",
        lambda: "token-1",
    )
    monkeypatch.setattr(
        "kestrel_sovereign.features.feature_features.feature._github_create_issue",
        create_issue,
    )
    monkeypatch.setattr(
        "kestrel_sovereign.features.feature_features.feature."
        "_github_find_open_pull_request_for_branch",
        find_open_pr,
    )
    monkeypatch.setattr(
        "kestrel_sovereign.features.feature_features.feature."
        "_github_merge_pull_request",
        merge_pr,
    )

    try:
        await workflows.initialize()
        await feature.initialize()
        assert workflows.runner is not None
        workflows.runner.ci_green_provider = lambda gate, result: {
            "check_runs": [
                {
                    "name": name,
                    "status": "completed",
                    "conclusion": "success",
                }
                for name in gate.params["required_checks"]
            ],
            "required_checks": list(gate.params["required_checks"]),
        }

        defined = await feature.feature_feature_define_workflows(kind="tool")
        assert defined.status is ToolResultStatus.OK

        run = await feature.feature_feature_run(
            "tool",
            params={
                "feature_name": "demo feature",
                "target_tool_name": "demo_tool",
                "summary": "Add a demo tool",
                "repository": "Org/repo",
                "branch": "codex/demo-tool",
            },
        )

        assert run.status is ToolResultStatus.OK
        assert run.data["status"] == "waiting"
        resumed = await workflows.workflow_resume(run.data["run_id"])
        assert resumed.status is ToolResultStatus.OK
        assert resumed.data["status"] == "completed"
        assert created_issues[0]["body"]["title"] == (
            "[EPIC] Feature proposal: demo feature: demo_tool"
        )
        assert talon.claims[0]["issue"] == 42
        assert merged_prs == [
            {
                "repository": "Org/repo",
                "pr_number": 17,
                "token": "token-1",
                "body": {"merge_method": "merge", "sha": "head-sha-1"},
            }
        ]
        assert audit_anchor.calls == 1

        status = await workflows.workflow_status(run.data["run_id"])
        assert status.status is ToolResultStatus.OK
        assert status.data["params"]["issue_number"] == 42
        assert status.data["params"]["talon_job_ids"] == ["talon-job-1"]
        assert status.data["params"]["publish_pr_number"] == 17
        assert status.data["params"]["publish_pr_head_sha"] == "head-sha-1"

        history = await workflows.workflow_history(run.data["run_id"])
        assert history.status is ToolResultStatus.OK
        assert [link["stage_name"] for link in history.data["links"]] == list(
            FEATURE_FEATURES_STAGE_ORDER
        )
        assert all(link["gate_outcome"] == "pass" for link in history.data["links"])
    finally:
        pending = [task for task in agent.background_tasks if not task.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        await backend.close()


@pytest.mark.asyncio
async def test_feature_feature_runtime_status_reports_missing_action_providers():
    registry = SourceRegistry()
    agent = SimpleNamespace(signal_registry=registry)
    feature = FeatureFeaturesFeature(agent)
    await feature.initialize()

    with (
        patch(
            "kestrel_sovereign.features.feature_features.feature._github_token",
            return_value="gh-test",
        ),
        patch(
            "kestrel_sovereign.features.feature_features.feature._ci_green_token",
            return_value="gh-test",
        ),
    ):
        result = await feature.feature_feature_runtime_status()

    assert result.status is ToolResultStatus.ERROR
    assert result.data["missing_registered"] == []
    assert "feature_features.file_github_epic" not in (
        result.data["missing_action_providers"]
    )
    assert "feature_features.assign_talon_chunks" in (
        result.data["missing_action_providers"]
    )
    assert {
        "source": "feature_features.assign_talon_chunks",
        "provider": "feature_feature_assign_talon_chunks",
        "requirement": "TalonCoordinatorFeature",
    } in result.data["missing_provider_requirements"]
    assert {
        "source": "feature_features.red_team_review",
        "prerequisite": "workflow_red_team_prompt_pack_resolver",
    } in result.data["missing_workflow_prerequisites"]
    assert {
        "source": "feature_features.red_team_review",
        "prerequisite": "workflow_red_team_attestation_resolver",
    } in result.data["missing_workflow_prerequisites"]
    explore = next(
        row
        for row in result.data["sources"]
        if row["name"] == "feature_features.explore"
    )
    assert explore["ready"] is True
    assign = next(
        row
        for row in result.data["sources"]
        if row["name"] == "feature_features.assign_talon_chunks"
    )
    assert assign["stage"] == "assign_talon_chunks"
    assert assign["provider_requirements"] == [
        {"name": "TalonCoordinatorFeature", "ready": False}
    ]
    assert assign["provider_requirements_ready"] is False
    assert any(
        row["name"] == "feature_features.assign_talon_chunks"
        for row in result.data["blocking_sources"]
    )


@pytest.mark.asyncio
async def test_feature_feature_runtime_status_requires_github_token_for_default_epic_provider():
    registry = SourceRegistry()
    agent = SimpleNamespace(signal_registry=registry)
    feature = FeatureFeaturesFeature(agent)
    await feature.initialize()

    with (
        patch(
            "kestrel_sovereign.features.feature_features.feature._github_token",
            return_value=None,
        ),
        patch(
            "kestrel_sovereign.features.feature_features.feature._ci_green_token",
            return_value=None,
        ),
    ):
        result = await feature.feature_feature_runtime_status()

    assert result.status is ToolResultStatus.ERROR
    assert "feature_features.file_github_epic" in (
        result.data["missing_action_providers"]
    )
    assert "feature_features.ci_green" in (
        result.data["missing_action_providers"]
    )
    assert {
        "source": "feature_features.file_github_epic",
        "provider": "feature_feature_file_github_epic",
        "requirement": "github_token",
    } in result.data["missing_provider_requirements"]
    assert {
        "source": "feature_features.ci_green",
        "provider": "feature_feature_ci_green",
        "requirement": "ci_token",
    } in result.data["missing_provider_requirements"]
    row = next(
        item
        for item in result.data["sources"]
        if item["name"] == "feature_features.file_github_epic"
    )
    assert row["ready"] is False


@pytest.mark.asyncio
async def test_feature_feature_runtime_status_requires_github_token_for_default_ci_provider():
    async def file_github_epic(_payload):
        return {"status": "ok"}

    registry = SourceRegistry()
    agent = SimpleNamespace(
        signal_registry=registry,
        feature_feature_file_github_epic=file_github_epic,
    )
    feature = FeatureFeaturesFeature(agent)
    await feature.initialize()

    with (
        patch(
            "kestrel_sovereign.features.feature_features.feature._github_token",
            return_value="gh-test",
        ),
        patch(
            "kestrel_sovereign.features.feature_features.feature._ci_green_token",
            return_value=None,
        ),
    ):
        result = await feature.feature_feature_runtime_status()

    assert result.status is ToolResultStatus.ERROR
    assert "feature_features.file_github_epic" not in (
        result.data["missing_action_providers"]
    )
    assert "feature_features.ci_green" in (
        result.data["missing_action_providers"]
    )
    row = next(
        item
        for item in result.data["sources"]
        if item["name"] == "feature_features.ci_green"
    )
    assert row["action_provider_ready"] is False
    assert row["ready"] is False


@pytest.mark.asyncio
async def test_feature_feature_runtime_status_rejects_empty_github_token():
    registry = SourceRegistry()
    agent = SimpleNamespace(signal_registry=registry)
    feature = FeatureFeaturesFeature(agent)
    await feature.initialize()

    with (
        patch(
            "kestrel_sovereign.features.feature_features.feature._github_token",
            return_value="",
        ),
        patch(
            "kestrel_sovereign.features.feature_features.feature._ci_green_token",
            return_value="",
        ),
    ):
        result = await feature.feature_feature_runtime_status()

    assert result.status is ToolResultStatus.ERROR
    assert "feature_features.file_github_epic" in (
        result.data["missing_action_providers"]
    )
    assert "feature_features.ci_green" in (
        result.data["missing_action_providers"]
    )


@pytest.mark.asyncio
async def test_feature_feature_runtime_status_marks_talon_provider_ready_when_loaded():
    async def talon_claim(**kwargs):
        return ToolResult.ok("dispatched", data={"kwargs": kwargs})

    registry = SourceRegistry()
    agent = SimpleNamespace(
        signal_registry=registry,
        features={"TalonCoordinatorFeature": SimpleNamespace(talon_claim=talon_claim)},
    )
    feature = FeatureFeaturesFeature(agent)
    await feature.initialize()

    with (
        patch(
            "kestrel_sovereign.features.feature_features.feature._github_token",
            return_value="gh-test",
        ),
        patch(
            "kestrel_sovereign.features.feature_features.feature._ci_green_token",
            return_value="gh-test",
        ),
    ):
        result = await feature.feature_feature_runtime_status()

    assert result.status is ToolResultStatus.ERROR
    assert "feature_features.assign_talon_chunks" not in (
        result.data["missing_action_providers"]
    )


@pytest.mark.asyncio
async def test_feature_feature_runtime_status_reports_missing_prompt_pack_resolver():
    registry = SourceRegistry()
    agent = SimpleNamespace(signal_registry=registry)
    feature = FeatureFeaturesFeature(agent)
    await feature.initialize()

    with (
        patch(
            "kestrel_sovereign.features.feature_features.feature._github_token",
            return_value="gh-test",
        ),
        patch(
            "kestrel_sovereign.features.feature_features.feature._ci_green_token",
            return_value="gh-test",
        ),
    ):
        result = await feature.feature_feature_runtime_status()

    red_team = next(
        item
        for item in result.data["sources"]
        if item["name"] == "feature_features.red_team_review"
    )
    assert result.status is ToolResultStatus.ERROR
    assert red_team["workflow_prerequisites_ready"] is False
    assert {
        "source": "feature_features.red_team_review",
        "prerequisite": "workflow_red_team_prompt_pack_resolver",
    } in result.data["missing_workflow_prerequisites"]


@pytest.mark.asyncio
async def test_feature_feature_runtime_status_reports_missing_attestation_resolver():
    registry = SourceRegistry()
    agent = SimpleNamespace(
        signal_registry=registry,
        workflow_council_approve_provider=lambda *args: {
            "approved_dids": ["did:kestrel:one", "did:kestrel:two"]
        },
        workflow_red_team_prompt_pack_resolver=lambda _constraint: {
            "name": "kestrel-red-team-prompts",
            "version": "1.0.0",
            "prompt_hash": "a" * 64,
            "prompt": "review",
        },
    )
    feature = FeatureFeaturesFeature(agent)
    await feature.initialize()

    with (
        patch(
            "kestrel_sovereign.features.feature_features.feature._github_token",
            return_value="gh-test",
        ),
        patch(
            "kestrel_sovereign.features.feature_features.feature._ci_green_token",
            return_value="gh-test",
        ),
    ):
        result = await feature.feature_feature_runtime_status()

    red_team = next(
        item
        for item in result.data["sources"]
        if item["name"] == "feature_features.red_team_review"
    )
    assert {
        "source": "feature_features.red_team_review",
        "prerequisite": "workflow_red_team_prompt_pack_resolver",
    } not in result.data["missing_workflow_prerequisites"]
    assert red_team["workflow_prerequisites_ready"] is False
    assert {
        "source": "feature_features.red_team_review",
        "prerequisite": "workflow_red_team_attestation_resolver",
    } in result.data["missing_workflow_prerequisites"]


@pytest.mark.asyncio
async def test_feature_feature_runtime_status_accepts_red_team_resolvers():
    registry = SourceRegistry()
    agent = SimpleNamespace(
        signal_registry=registry,
        workflow_red_team_prompt_pack_resolver=lambda _constraint: {
            "name": "kestrel-red-team-prompts",
            "version": "1.0.0",
            "prompt_hash": "a" * 64,
            "prompt": "review",
        },
        workflow_red_team_attestation_resolver=lambda _reviewers: {},
    )
    feature = FeatureFeaturesFeature(agent)
    await feature.initialize()

    with (
        patch(
            "kestrel_sovereign.features.feature_features.feature._github_token",
            return_value="gh-test",
        ),
        patch(
            "kestrel_sovereign.features.feature_features.feature._ci_green_token",
            return_value="gh-test",
        ),
    ):
        result = await feature.feature_feature_runtime_status()

    red_team = next(
        item
        for item in result.data["sources"]
        if item["name"] == "feature_features.red_team_review"
    )
    assert red_team["workflow_prerequisites_ready"] is True
    assert not any(
        item["source"] == "feature_features.red_team_review"
        for item in result.data["missing_workflow_prerequisites"]
    )


@pytest.mark.asyncio
async def test_feature_feature_runtime_status_reports_missing_council_provider():
    registry = SourceRegistry()
    agent = SimpleNamespace(
        signal_registry=registry,
        workflow_red_team_prompt_pack_resolver=lambda _constraint: {
            "name": "kestrel-red-team-prompts",
            "version": "1.0.0",
            "prompt_hash": "a" * 64,
            "prompt": "review",
        },
        workflow_red_team_attestation_resolver=lambda _reviewers: {},
    )
    feature = FeatureFeaturesFeature(agent)
    await feature.initialize()

    with (
        patch(
            "kestrel_sovereign.features.feature_features.feature._github_token",
            return_value="gh-test",
        ),
        patch(
            "kestrel_sovereign.features.feature_features.feature._ci_green_token",
            return_value="gh-test",
        ),
    ):
        result = await feature.feature_feature_runtime_status()

    council = next(
        item
        for item in result.data["sources"]
        if item["name"] == "feature_features.council_review"
    )
    assert council["workflow_prerequisites_ready"] is False
    assert {
        "source": "feature_features.council_review",
        "prerequisite": "workflow_council_approve_provider",
    } in result.data["missing_workflow_prerequisites"]


@pytest.mark.asyncio
async def test_feature_feature_runtime_status_rejects_workflows_wrapper_without_resolver():
    class WorkflowsFeature:
        def __init__(self, agent):
            self.agent = agent
            self.runner = SimpleNamespace(
                council_approve_provider=self._council_approve_provider
            )

        async def _council_approve_provider(self, gate, run, stage, link):
            return {"status": "failed", "reason": "council resolver unavailable"}

    registry = SourceRegistry()
    agent = SimpleNamespace(
        signal_registry=registry,
        workflow_red_team_prompt_pack_resolver=lambda _constraint: {
            "name": "kestrel-red-team-prompts",
            "version": "1.0.0",
            "prompt_hash": "a" * 64,
            "prompt": "review",
        },
        workflow_red_team_attestation_resolver=lambda _reviewers: {},
    )
    agent.features = {"WorkflowsFeature": WorkflowsFeature(agent)}
    feature = FeatureFeaturesFeature(agent)
    await feature.initialize()

    with (
        patch(
            "kestrel_sovereign.features.feature_features.feature._github_token",
            return_value="gh-test",
        ),
        patch(
            "kestrel_sovereign.features.feature_features.feature._ci_green_token",
            return_value="gh-test",
        ),
    ):
        result = await feature.feature_feature_runtime_status()

    assert {
        "source": "feature_features.council_review",
        "prerequisite": "workflow_council_approve_provider",
    } in result.data["missing_workflow_prerequisites"]


@pytest.mark.asyncio
async def test_feature_feature_runtime_status_accepts_workflows_council_provider():
    registry = SourceRegistry()
    runner = SimpleNamespace(council_approve_provider=lambda *args: None)
    agent = SimpleNamespace(
        signal_registry=registry,
        features={"WorkflowsFeature": SimpleNamespace(runner=runner)},
        workflow_red_team_prompt_pack_resolver=lambda _constraint: {
            "name": "kestrel-red-team-prompts",
            "version": "1.0.0",
            "prompt_hash": "a" * 64,
            "prompt": "review",
        },
        workflow_red_team_attestation_resolver=lambda _reviewers: {},
    )
    feature = FeatureFeaturesFeature(agent)
    await feature.initialize()

    with (
        patch(
            "kestrel_sovereign.features.feature_features.feature._github_token",
            return_value="gh-test",
        ),
        patch(
            "kestrel_sovereign.features.feature_features.feature._ci_green_token",
            return_value="gh-test",
        ),
    ):
        result = await feature.feature_feature_runtime_status()

    council = next(
        item
        for item in result.data["sources"]
        if item["name"] == "feature_features.council_review"
    )
    assert council["workflow_prerequisites_ready"] is True
    assert not any(
        item["source"] == "feature_features.council_review"
        for item in result.data["missing_workflow_prerequisites"]
    )


@pytest.mark.asyncio
async def test_feature_features_installs_github_epic_provider():
    registry = SourceRegistry()
    agent = SimpleNamespace(signal_registry=registry)
    feature = FeatureFeaturesFeature(agent)

    await feature.initialize()

    assert callable(agent.feature_feature_file_github_epic)
    with patch(
        "kestrel_sovereign.features.feature_features.feature._github_token",
        return_value="",
    ):
        with pytest.raises(RuntimeError, match="GITHUB_TOKEN"):
            await agent.feature_feature_file_github_epic(
                {"repository": "Org/repo", "feature_name": "DemoFeature"}
            )


@pytest.mark.asyncio
async def test_feature_features_github_epic_provider_creates_issue():
    registry = SourceRegistry()
    agent = SimpleNamespace(signal_registry=registry)
    feature = FeatureFeaturesFeature(agent)
    await feature.initialize()
    created = {}

    async def fake_create(repository, token, body):
        created.update({"repository": repository, "token": token, "body": body})
        return {"number": 42, "html_url": "https://github.test/org/repo/issues/42"}

    with (
        patch(
            "kestrel_sovereign.features.feature_features.feature._github_token",
            return_value="gh-test",
        ),
        patch(
            "kestrel_sovereign.features.feature_features.feature._github_create_issue",
            side_effect=fake_create,
        ),
    ):
        result = await agent.feature_feature_file_github_epic(
            {
                "repository": "Org/repo",
                "feature_name": "DemoFeature",
                "target_tool_name": "demo_tool",
                "summary": "Add a demo tool",
                "api_token": "do-not-log",
                "password": "no-password",
                "credential_ref": "no-credential",
                "provider": {"secret_key": "nested-secret"},
                "chunks": [{"token": "chunk-secret", "auth_header": "no-auth"}],
            }
        )

    assert result["status"] == "ok"
    assert result["issue_number"] == 42
    assert created["repository"] == "Org/repo"
    assert created["token"] == "gh-test"
    assert created["body"]["title"] == (
        "[EPIC] Feature proposal: DemoFeature: demo_tool"
    )
    assert "labels" not in created["body"]
    assert "<redacted>" in created["body"]["body"]
    assert "do-not-log" not in created["body"]["body"]
    assert "no-password" not in created["body"]["body"]
    assert "no-credential" not in created["body"]["body"]
    assert "nested-secret" not in created["body"]["body"]
    assert "chunk-secret" not in created["body"]["body"]
    assert "no-auth" not in created["body"]["body"]


@pytest.mark.asyncio
async def test_feature_features_assign_talon_provider_uses_explicit_issue():
    class StubTalon:
        def __init__(self):
            self.calls = []

        async def talon_claim(self, **kwargs):
            self.calls.append(kwargs)
            return ToolResult.ok(
                "dispatched",
                data={"dispatched": True, "job_id": "job-1"},
            )

    registry = SourceRegistry()
    talon = StubTalon()
    agent = SimpleNamespace(
        signal_registry=registry,
        features={"TalonCoordinatorFeature": talon},
    )
    feature = FeatureFeaturesFeature(agent)
    await feature.initialize()

    async def fake_create(repository, token, body):
        return {"number": 42, "html_url": "https://github.test/org/repo/issues/42"}

    with (
        patch(
            "kestrel_sovereign.features.feature_features.feature._github_token",
            return_value="gh-test",
        ),
        patch(
            "kestrel_sovereign.features.feature_features.feature._github_create_issue",
            side_effect=fake_create,
        ),
    ):
        epic = await agent.feature_feature_file_github_epic(
            {
                "repository": "Org/repo",
                "feature_name": "DemoFeature",
            }
        )
        result = await agent.feature_feature_assign_talon_chunks(
            {
                "repository": "Org/repo",
                "issue_number": epic["issue_number"],
                "talon_backend": "codex",
                "talon_model": "gpt-5.5",
                "skip_clarification": True,
                "self_review": False,
            }
        )

    assert epic["issue_number"] == 42
    assert result["status"] == "ok"
    assert result["issues"] == [42]
    assert result["dispatches"][0]["job_id"] == "job-1"
    assert talon.calls == [
        {
            "repo": "Org/repo",
            "issue": 42,
            "max_iterations": None,
            "max_turns": None,
            "backend": "codex",
            "model": "gpt-5.5",
            "auth_lane": None,
            "skip_clarification": True,
            "worktree": True,
            "self_review": False,
        }
    ]


@pytest.mark.asyncio
async def test_feature_features_assign_talon_provider_dispatches_issue_list():
    class StubTalon:
        def __init__(self):
            self.issues = []

        async def talon_claim(self, repo, issue, **kwargs):
            self.issues.append((repo, issue, kwargs))
            return ToolResult.ok("dispatched", data={"issue": issue})

    talon = StubTalon()
    agent = SimpleNamespace(features={"talon": talon})
    feature = FeatureFeaturesFeature(agent)
    await feature.initialize()

    result = await agent.feature_feature_assign_talon_chunks(
        {
            "repository": "Org/repo",
            "talon_issue_numbers": [7, 8],
            "chunks": [{"issue_number": 9}, {"issue_number": 8}],
            "max_iterations": 3,
            "worktree": False,
        }
    )

    assert result["status"] == "ok"
    assert result["issues"] == [7, 8, 9]
    assert [(repo, issue) for repo, issue, _ in talon.issues] == [
        ("Org/repo", 7),
        ("Org/repo", 8),
        ("Org/repo", 9),
    ]
    assert talon.issues[0][2]["max_iterations"] == 3
    assert talon.issues[0][2]["worktree"] is False


@pytest.mark.asyncio
async def test_feature_features_assign_talon_provider_requires_issue_and_talon():
    feature = FeatureFeaturesFeature(SimpleNamespace(features={}))
    await feature.initialize()

    with pytest.raises(RuntimeError, match="TalonCoordinatorFeature"):
        await feature.agent.feature_feature_assign_talon_chunks(
            {"repository": "Org/repo", "issue_number": 1}
        )

    talon = SimpleNamespace(
        talon_claim=lambda **kwargs: ToolResult.ok("not awaited")
    )
    agent = SimpleNamespace(features={"talon": talon})
    feature = FeatureFeaturesFeature(agent)
    await feature.initialize()

    with pytest.raises(RuntimeError, match="issue_number"):
        await agent.feature_feature_assign_talon_chunks({"repository": "Org/repo"})


@pytest.mark.asyncio
async def test_feature_features_implement_chunks_provider_passes_completed_jobs():
    class StubTalon:
        async def talon_status(self):
            return ToolResult.ok(
                "status",
                data={
                    "jobs": [
                        {"id": "job-1", "status": "complete", "issue": 7},
                        {"id": "job-2", "status": "complete", "issue": 8},
                    ]
                },
            )

    agent = SimpleNamespace(features={"TalonCoordinatorFeature": StubTalon()})
    feature = FeatureFeaturesFeature(agent)
    await feature.initialize()

    result = await agent.feature_feature_implement_chunks(
        {
            "talon_dispatches": [
                {"job_id": "job-1", "issue": 7},
                {"message_id": "job-2", "issue": 8},
            ]
        }
    )

    assert result["status"] == "ok"
    assert result["talon_job_ids"] == ["job-1", "job-2"]
    assert result["completed"] == 2
    assert result["jobs"][0]["issue"] == 7


@pytest.mark.asyncio
async def test_feature_features_implement_chunks_provider_fails_running_jobs():
    class StubTalon:
        async def talon_status(self):
            return ToolResult.ok(
                "status",
                data={
                    "jobs": [
                        {"id": "job-1", "status": "complete"},
                        {"id": "job-2", "status": "running"},
                    ]
                },
            )

    agent = SimpleNamespace(features={"talon": StubTalon()})
    feature = FeatureFeaturesFeature(agent)
    await feature.initialize()

    with pytest.raises(RuntimeError, match="running=1"):
        await agent.feature_feature_implement_chunks(
            {"talon_job_ids": ["job-1", "job-2"], "talon_wait_seconds": 0}
        )


@pytest.mark.asyncio
async def test_feature_features_implement_chunks_provider_polls_running_jobs():
    class StubTalon:
        def __init__(self):
            self.calls = 0

        async def talon_status(self):
            self.calls += 1
            status = "running" if self.calls == 1 else "complete"
            return ToolResult.ok(
                "status",
                data={"jobs": [{"id": "job-1", "status": status}]},
            )

    talon = StubTalon()
    agent = SimpleNamespace(features={"talon": talon})
    feature = FeatureFeaturesFeature(agent)
    await feature.initialize()

    result = await agent.feature_feature_implement_chunks(
        {
            "talon_job_ids": ["job-1"],
            "talon_wait_seconds": 1,
            "talon_poll_seconds": 0.01,
        }
    )

    assert result["status"] == "ok"
    assert result["completed"] == 1
    assert talon.calls == 2


@pytest.mark.asyncio
async def test_feature_features_implement_chunks_provider_requires_job_ids():
    class StubTalon:
        async def talon_status(self):
            return ToolResult.ok("status", data={"jobs": []})

    agent = SimpleNamespace(features={"talon": StubTalon()})
    feature = FeatureFeaturesFeature(agent)
    await feature.initialize()

    with pytest.raises(RuntimeError, match="talon_job_ids"):
        await agent.feature_feature_implement_chunks({})


@pytest.mark.asyncio
async def test_feature_features_installs_quality_gate_providers():
    commands = []

    async def command_runner(**kwargs):
        commands.append(kwargs)
        return {
            "exit_code": 0,
            "stdout": "ok",
            "stderr": "",
        }

    agent = SimpleNamespace(feature_feature_command_runner=command_runner)
    feature = FeatureFeaturesFeature(agent)
    await feature.initialize()

    tests = await agent.feature_feature_tests_pass(
        {
            "suite": "unit",
            "test_command": ["uv", "run", "--no-sync", "pytest", "tests/unit", "-q"],
            "cwd": "/tmp",
        }
    )
    lint = await agent.feature_feature_lint_clean(
        {
            "scopes": ["changed"],
            "lint_command": "uv run --no-sync python run_tests.py --kestrel --skip-check --validate-only",
            "cwd": "/tmp",
        }
    )

    assert tests["status"] == "ok"
    assert tests["suite"] == "unit"
    assert tests["failed"] == 0
    assert tests["errors"] == 0
    assert lint["status"] == "ok"
    assert lint["scopes"] == ["changed"]
    assert lint["violations"] == 0
    assert lint["errors"] == 0
    assert commands[0]["command"] == [
        "uv",
        "run",
        "--no-sync",
        "pytest",
        "tests/unit",
        "-q",
    ]
    assert commands[1]["command"] == [
        "uv",
        "run",
        "--no-sync",
        "python",
        "run_tests.py",
        "--kestrel",
        "--skip-check",
        "--validate-only",
    ]


@pytest.mark.asyncio
async def test_feature_features_quality_gate_providers_report_failures():
    async def command_runner(**kwargs):
        return {
            "exit_code": 1,
            "stdout": "failed",
            "stderr": "boom",
        }

    agent = SimpleNamespace(feature_feature_command_runner=command_runner)
    feature = FeatureFeaturesFeature(agent)
    await feature.initialize()

    tests = await agent.feature_feature_tests_pass({"suite": "unit"})
    lint = await agent.feature_feature_lint_clean({"scopes": "changed,package"})

    assert tests["status"] == "failed"
    assert tests["exit_code"] == 1
    assert tests["failed"] == 1
    assert tests["errors"] == 1
    assert lint["status"] == "failed"
    assert lint["scopes"] == ["changed", "package"]
    assert lint["violations"] == 1
    assert lint["errors"] == 1


@pytest.mark.asyncio
async def test_feature_features_installs_review_diff_providers():
    commands = []

    async def command_runner(**kwargs):
        commands.append(kwargs)
        if kwargs["command"][:2] == ["git", "merge-base"]:
            return {
                "exit_code": 0,
                "stdout": "abc123\n",
                "stderr": "",
            }
        if kwargs["command"][:2] == ["git", "ls-files"]:
            return {
                "exit_code": 0,
                "stdout": "",
                "stderr": "",
            }
        return {
            "exit_code": 0,
            "stdout": (
                "diff --git a/demo.py b/demo.py\n"
                "+++ b/demo.py\n"
                "@@\n"
                "+import math\n"
            ),
            "stderr": "",
        }

    agent = SimpleNamespace(feature_feature_command_runner=command_runner)
    feature = FeatureFeaturesFeature(agent)
    await feature.initialize()

    boundary = await agent.feature_feature_boundary_scan(
        {
            "base_ref": "origin/main",
            "cwd": "/tmp",
        }
    )
    red_team = await agent.feature_feature_red_team_review(
        {
            "base_ref": "origin/main",
            "cwd": "/tmp",
        }
    )

    assert boundary["status"] == "ok"
    assert boundary["scan"] == "constitutional_boundary"
    assert boundary["patch"].startswith("diff --git")
    assert boundary["changed_files"] == ["demo.py"]
    assert red_team["status"] == "ok"
    assert red_team["review"] == "red_team"
    assert red_team["patch"] == boundary["patch"]
    assert commands[0]["command"] == [
        "git",
        "merge-base",
        "origin/main",
        "HEAD",
    ]
    assert commands[1]["command"] == [
        "git",
        "diff",
        "--no-ext-diff",
        "abc123",
    ]


@pytest.mark.asyncio
async def test_feature_features_installs_ci_green_passthrough_provider(monkeypatch):
    monkeypatch.setattr(
        "kestrel_sovereign.features.feature_features.feature._github_token",
        lambda: None,
    )
    agent = SimpleNamespace()
    feature = FeatureFeaturesFeature(agent)

    await feature.initialize()

    result = await agent.feature_feature_ci_green(
        {
            "repository": "Org/repo",
            "branch": "codex/demo",
            "workflow_run_id": "run-1",
            "workflow_stage_name": "ci_green",
        }
    )

    assert result == {
        "status": "ok",
        "repository": "Org/repo",
        "branch": "codex/demo",
        "workflow_run_id": "run-1",
        "workflow_stage_name": "ci_green",
    }


@pytest.mark.asyncio
async def test_feature_features_publish_provider_merges_branch_pr(monkeypatch):
    find_calls = []
    merge_calls = []

    async def find_pr(repository, branch, token):
        find_calls.append((repository, branch, token))
        return {
            "number": 17,
            "html_url": "https://github.example/pr/17",
        }

    async def merge_pr(repository, pr_number, token, body):
        merge_calls.append((repository, pr_number, token, body))
        return {"merged": True, "sha": "abc123"}

    monkeypatch.setattr(
        "kestrel_sovereign.features.feature_features.feature._github_token",
        lambda: "token-1",
    )
    monkeypatch.setattr(
        "kestrel_sovereign.features.feature_features.feature._github_find_open_pull_request_for_branch",
        find_pr,
    )
    monkeypatch.setattr(
        "kestrel_sovereign.features.feature_features.feature._github_merge_pull_request",
        merge_pr,
    )
    agent = SimpleNamespace()
    feature = FeatureFeaturesFeature(agent)
    await feature.initialize()

    result = await agent.feature_feature_publish(
        {
            "repository": "Org/repo",
            "branch": "codex/demo",
            "publish_pr_number": 17,
            "publish_pr_head_sha": "head-abc",
            "workflow_run_id": "run-1",
            "workflow_stage_name": "publish",
        }
    )

    assert find_calls == [("Org/repo", "codex/demo", "token-1")]
    assert merge_calls == [
        (
            "Org/repo",
            17,
            "token-1",
            {"merge_method": "merge", "sha": "head-abc"},
        )
    ]
    assert result["status"] == "ok"
    assert result["merged"] is True
    assert result["pull_request_number"] == 17
    assert result["pull_request_url"] == "https://github.example/pr/17"
    assert result["merge_sha"] == "abc123"
    assert result["workflow_run_id"] == "run-1"


@pytest.mark.asyncio
async def test_feature_features_ci_green_provider_captures_publish_sha(
    monkeypatch,
):
    async def find_pr(repository, branch, token):
        return {
            "number": 17,
            "html_url": "https://github.example/pr/17",
            "head": {"sha": "head-abc"},
        }

    monkeypatch.setattr(
        "kestrel_sovereign.features.feature_features.feature._github_token",
        lambda: "token-1",
    )
    monkeypatch.setattr(
        "kestrel_sovereign.features.feature_features.feature._github_find_open_pull_request_for_branch",
        find_pr,
    )
    agent = SimpleNamespace()
    feature = FeatureFeaturesFeature(agent)
    await feature.initialize()

    result = await agent.feature_feature_ci_green(
        {"repository": "Org/repo", "branch": "codex/demo"}
    )

    assert result["publish_pr_number"] == 17
    assert result["publish_pr_head_sha"] == "head-abc"
    assert result["publish_pr_url"] == "https://github.example/pr/17"


@pytest.mark.asyncio
async def test_feature_features_publish_provider_requires_reviewed_sha(
    monkeypatch,
):
    async def find_pr(repository, branch, token):
        return {"number": 17, "head": {"sha": "new-head"}}

    monkeypatch.setattr(
        "kestrel_sovereign.features.feature_features.feature._github_token",
        lambda: "token-1",
    )
    monkeypatch.setattr(
        "kestrel_sovereign.features.feature_features.feature._github_find_open_pull_request_for_branch",
        find_pr,
    )
    agent = SimpleNamespace()
    feature = FeatureFeaturesFeature(agent)
    await feature.initialize()

    with pytest.raises(RuntimeError, match="publish_pr_head_sha"):
        await agent.feature_feature_publish(
            {
                "repository": "Org/repo",
                "branch": "codex/demo",
                "publish_pr_number": 17,
            }
        )


@pytest.mark.asyncio
async def test_feature_features_publish_provider_requires_reviewed_pr_number(
    monkeypatch,
):
    async def find_pr(repository, branch, token):
        return {"number": 18, "head": {"sha": "head-abc"}}

    monkeypatch.setattr(
        "kestrel_sovereign.features.feature_features.feature._github_token",
        lambda: "token-1",
    )
    monkeypatch.setattr(
        "kestrel_sovereign.features.feature_features.feature._github_find_open_pull_request_for_branch",
        find_pr,
    )
    agent = SimpleNamespace()
    feature = FeatureFeaturesFeature(agent)
    await feature.initialize()

    with pytest.raises(RuntimeError, match="publish_pr_number"):
        await agent.feature_feature_publish(
            {
                "repository": "Org/repo",
                "branch": "codex/demo",
                "publish_pr_number": 17,
                "publish_pr_head_sha": "head-abc",
            }
        )


@pytest.mark.asyncio
async def test_feature_features_publish_provider_requires_github_token(
    monkeypatch,
):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setattr(
        "kestrel_sovereign.features.feature_features.feature._github_token",
        lambda: None,
    )
    agent = SimpleNamespace()
    feature = FeatureFeaturesFeature(agent)
    await feature.initialize()

    with pytest.raises(RuntimeError, match="GITHUB_TOKEN"):
        await agent.feature_feature_publish(
            {"repository": "Org/repo", "branch": "codex/demo"}
        )


@pytest.mark.asyncio
async def test_feature_features_publish_provider_fails_ambiguous_pr(
    monkeypatch,
):
    async def find_pr(repository, branch, token):
        raise RuntimeError("multiple open pull requests found for Org/repo:codex/demo")

    monkeypatch.setattr(
        "kestrel_sovereign.features.feature_features.feature._github_token",
        lambda: "token-1",
    )
    monkeypatch.setattr(
        "kestrel_sovereign.features.feature_features.feature._github_find_open_pull_request_for_branch",
        find_pr,
    )
    agent = SimpleNamespace()
    feature = FeatureFeaturesFeature(agent)
    await feature.initialize()

    with pytest.raises(RuntimeError, match="multiple open pull requests"):
        await agent.feature_feature_publish(
            {
                "repository": "Org/repo",
                "branch": "codex/demo",
                "publish_pr_number": 17,
                "publish_pr_head_sha": "head-abc",
            }
        )


@pytest.mark.asyncio
async def test_feature_features_audit_anchor_provider_delegates_to_feature():
    class StubAuditAnchor:
        def __init__(self):
            self.called = 0

        async def anchor_audit(self):
            self.called += 1
            return ToolResult.ok(
                "anchored",
                data={"anchor_id": "anchor-1", "entries_count": 3},
            )

    audit_anchor = StubAuditAnchor()
    agent = SimpleNamespace(features={"AuditAnchorFeature": audit_anchor})
    feature = FeatureFeaturesFeature(agent)

    await feature.initialize()
    result = await agent.feature_feature_audit_anchor(
        {
            "workflow_run_id": "run-1",
            "workflow_stage_name": "audit_anchor",
        }
    )

    assert audit_anchor.called == 1
    assert result["status"] == "ok"
    assert result["audit_anchor_status"] == "ok"
    assert result["audit_anchor"]["anchor_id"] == "anchor-1"
    assert result["workflow_run_id"] == "run-1"


@pytest.mark.asyncio
async def test_feature_features_audit_anchor_provider_requires_loaded_feature():
    agent = SimpleNamespace(features={})
    feature = FeatureFeaturesFeature(agent)

    await feature.initialize()

    with pytest.raises(RuntimeError, match="AuditAnchorFeature"):
        await agent.feature_feature_audit_anchor({})


@pytest.mark.asyncio
async def test_feature_features_audit_anchor_provider_rejects_partial_anchor():
    class PartialAuditAnchor:
        async def anchor_audit(self):
            return ToolResult.partial(
                "anchored hash only",
                error="file storage failed",
                data={"anchor_id": "anchor-1"},
            )

    agent = SimpleNamespace(features={"AuditAnchorFeature": PartialAuditAnchor()})
    feature = FeatureFeaturesFeature(agent)

    await feature.initialize()

    with pytest.raises(RuntimeError, match="file storage failed"):
        await agent.feature_feature_audit_anchor({})


@pytest.mark.asyncio
async def test_feature_features_review_diff_providers_fail_closed():
    async def command_runner(**kwargs):
        return {
            "exit_code": 128,
            "stdout": "",
            "stderr": "fatal: bad revision",
        }

    agent = SimpleNamespace(feature_feature_command_runner=command_runner)
    feature = FeatureFeaturesFeature(agent)
    await feature.initialize()

    with pytest.raises(RuntimeError, match="feature change snapshot failed"):
        await agent.feature_feature_boundary_scan({"base_ref": "missing/main"})


@pytest.mark.asyncio
async def test_feature_features_review_diff_provider_does_not_truncate_patch():
    long_patch = "+++ b/demo.py\n" + ("+value = 1\n" * 500) + "+import math\n"

    async def command_runner(**kwargs):
        if kwargs["command"][:2] == ["git", "merge-base"]:
            return {
                "exit_code": 0,
                "stdout": "abc123\n",
                "stderr": "",
            }
        if kwargs["command"][:2] == ["git", "ls-files"]:
            return {
                "exit_code": 0,
                "stdout": "",
                "stderr": "",
            }
        return {
            "exit_code": 0,
            "stdout": long_patch,
            "stderr": "",
        }

    agent = SimpleNamespace(feature_feature_command_runner=command_runner)
    feature = FeatureFeaturesFeature(agent)
    await feature.initialize()

    result = await agent.feature_feature_boundary_scan({})

    assert result["patch"] == long_patch
    assert "<truncated>" not in result["patch"]
    assert result["patch"].endswith("+import math\n")


@pytest.mark.asyncio
async def test_feature_features_review_diff_provider_includes_untracked_files(
    tmp_path,
):
    untracked = tmp_path / "new_feature.py"
    untracked.write_text("from kestrel_sovereign.constitution import guard\n")

    async def command_runner(**kwargs):
        if kwargs["command"][:2] == ["git", "merge-base"]:
            return {
                "exit_code": 0,
                "stdout": "abc123\n",
                "stderr": "",
            }
        if kwargs["command"][:2] == ["git", "ls-files"]:
            return {
                "exit_code": 0,
                "stdout": "new_feature.py\n",
                "stderr": "",
            }
        return {
            "exit_code": 0,
            "stdout": "",
            "stderr": "",
        }

    agent = SimpleNamespace(feature_feature_command_runner=command_runner)
    feature = FeatureFeaturesFeature(agent)
    await feature.initialize()

    result = await agent.feature_feature_boundary_scan({"cwd": str(tmp_path)})

    assert "+++ b/new_feature.py" in result["patch"]
    assert "@@ -0,0 +1,1 @@" in result["patch"]
    assert "+from kestrel_sovereign.constitution import guard" in result["patch"]
    assert result["changed_files"] == ["new_feature.py"]


@pytest.mark.asyncio
async def test_feature_features_review_diff_provider_reports_deleted_files():
    async def command_runner(**kwargs):
        if kwargs["command"][:2] == ["git", "merge-base"]:
            return {
                "exit_code": 0,
                "stdout": "abc123\n",
                "stderr": "",
            }
        if kwargs["command"][:2] == ["git", "ls-files"]:
            return {
                "exit_code": 0,
                "stdout": "",
                "stderr": "",
            }
        return {
            "exit_code": 0,
            "stdout": (
                "diff --git a/removed.py b/removed.py\n"
                "deleted file mode 100644\n"
                "--- a/removed.py\n"
                "+++ /dev/null\n"
                "@@ -1 +0,0 @@\n"
                "-import math\n"
            ),
            "stderr": "",
        }

    agent = SimpleNamespace(feature_feature_command_runner=command_runner)
    feature = FeatureFeaturesFeature(agent)
    await feature.initialize()

    result = await agent.feature_feature_boundary_scan({})

    assert result["changed_files"] == ["removed.py"]
    assert result["changed_file_count"] == 1


@pytest.mark.asyncio
async def test_feature_feature_runtime_status_passes_with_registered_providers():
    registry = SourceRegistry()
    agent = SimpleNamespace(
        signal_registry=registry,
        workflow_council_approve_provider=lambda *args: {
            "approved_dids": ["did:kestrel:one", "did:kestrel:two"]
        },
        workflow_red_team_prompt_pack_resolver=lambda _constraint: {
            "name": "kestrel-red-team-prompts",
            "version": "1.0.0",
            "prompt_hash": "a" * 64,
            "prompt": "review",
        },
        workflow_red_team_attestation_resolver=lambda _reviewers: {},
    )
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
    assert row["ready"] is True
    assert "feature_features.file_github_epic" not in (
        result.data["missing_action_providers"]
    )
    assert not any(
        item["source"] == "feature_features.file_github_epic"
        for item in result.data["missing_provider_requirements"]
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
