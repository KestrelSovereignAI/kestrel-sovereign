"""Workflows execute real decorated tools through native primary task storage."""

import inspect
import json
from types import SimpleNamespace

import pytest
from kestrel_sovereign._vendor.a2a.types import Task, TaskState
from kestrel_sdk.tools.result import ToolResult, ToolResultStatus

from kestrel_sovereign.a2a.projections import embedded_json_value
from kestrel_sovereign.execution_custody import ExecutionAuthorityError
from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.features.tasks.feature import TaskFeature
from tests.unit.test_a2a_native_task_manager import manager as manager  # noqa: PLC0414


class WorkflowProvider(Feature):
    @property
    def tool_description(self):
        return "Real local workflow tools"

    async def initialize(self):
        pass


class ModelProvider(WorkflowProvider):
    @tool(name="list_models", description="List deterministic test models")
    async def list_models(self, provider: str = "all") -> dict:
        return {"models": ["model-a", "model-b"], "provider": provider}

    @tool(name="get_current_model", description="Read the selected test model")
    async def get_current_model(self) -> dict:
        return {"model": "model-a", "provider": "test-provider"}


class MemoryProvider(WorkflowProvider):
    @tool(name="memory_status", description="Return the selected real tool outcome")
    async def memory_status(self, context: dict | None = None):
        self.calls += 1
        result = self.supplier(self.calls)
        if inspect.isawaitable(result):
            result = await result
        return result if context is None else {"context": context, "value": result}


class WalletProvider(WorkflowProvider):
    @tool(name="check_balance", description="Read a deterministic test balance")
    async def check_balance(self) -> dict:
        return {"balance": "12.50", "currency": "TEST"}


@pytest.fixture
def providers(manager):
    agent = SimpleNamespace(
        did=manager.did, task_manager=manager, _execution_custody=None
    )
    model, memory, wallet = (
        ModelProvider(agent),
        MemoryProvider(agent),
        WalletProvider(agent),
    )
    memory.calls = 0
    memory.supplier = lambda _attempt: {"episodes": 42, "total_tokens": 150000}
    for name, provider in (
        ("model_agent", model),
        ("memory_feature", memory),
        ("wallet_feature", wallet),
    ):
        provider.name = name
        manager.register_provider(name, provider.get_agent_skills(), provider)
    return SimpleNamespace(model=model, memory=memory, wallet=wallet)


@pytest.fixture
def task_feature(manager, providers):
    feature = TaskFeature(SimpleNamespace(did=manager.did))
    feature.set_task_manager(manager)
    return feature


def payload(result, index=0):
    return result.data["results"][index]["result"]["result"]


async def test_single_step_retains_an_official_task_and_result(task_feature, manager):
    result = await task_feature.run_workflow(
        [{"feature": "model_agent", "skill": "get_current_model"}]
    )
    assert result.status is ToolResultStatus.OK
    assert result.data["workflow_steps"] == result.data["completed"] == 1
    assert isinstance(result.data["results"][0]["result"], dict)
    assert result.data["failed"] == 0 and payload(result)["model"] == "model-a"
    [(task_id,)] = await manager.task_store.backend.fetch_all(
        "SELECT id FROM a2a_tasks"
    )
    task = await manager.get_task_for_recipient(task_id, manager.did)
    assert (
        isinstance(task, Task) and task.status.state == TaskState.TASK_STATE_COMPLETED
    )
    recognized, value = embedded_json_value(task.artifacts[0].parts[0])
    assert recognized and value == result.data["results"][0]["result"]


async def test_multi_step_workflow_runs_in_order_and_retains_metadata(task_feature):
    result = await task_feature.run_workflow(
        [
            {"feature": "model_agent", "skill": "get_current_model"},
            {"feature": "memory_feature", "skill": "memory_status"},
            {"feature": "wallet_feature", "skill": "check_balance"},
        ]
    )
    assert result.status is ToolResultStatus.OK
    assert result.data["workflow_steps"] == result.data["completed"] == 3
    assert result.data["failed"] == 0
    assert payload(result, 0)["model"] == "model-a"
    assert payload(result, 1)["episodes"] == 42
    assert payload(result, 2)["balance"] == "12.50"
    assert [
        (step["step"], step["feature"], step["skill"])
        for step in result.data["results"]
    ] == [
        (0, "model_agent", "get_current_model"),
        (1, "memory_feature", "memory_status"),
        (2, "wallet_feature", "check_balance"),
    ]
    assert (
        isinstance(result.data["total_duration_ms"], int)
        and result.data["total_duration_ms"] >= 0
    )
    assert all(
        isinstance(step["duration_ms"], int) and step["duration_ms"] >= 0
        for step in result.data["results"]
    )


@pytest.mark.parametrize(
    "arguments", [{"provider": "selected"}, '{"provider":"selected"}']
)
async def test_object_and_json_encoded_tool_arguments(task_feature, arguments):
    result = await task_feature.run_workflow(
        [{"feature": "model_agent", "skill": "list_models", "args": arguments}]
    )
    assert (
        result.status is ToolResultStatus.OK
        and payload(result)["provider"] == "selected"
    )


@pytest.mark.parametrize("reference", ["{{steps.0.result}}", "{{prev.result}}"])
async def test_references_are_resolved_into_typed_native_arguments(
    task_feature, reference
):
    result = await task_feature.run_workflow(
        [
            {"feature": "model_agent", "skill": "get_current_model"},
            {
                "feature": "memory_feature",
                "skill": "memory_status",
                "args": {"context": reference},
            },
        ]
    )
    assert result.status is ToolResultStatus.OK
    assert payload(result, 1)["context"] == result.data["results"][0]["result"]


@pytest.mark.parametrize(
    "arguments,expected",
    [
        ([], "list"),
        (None, "NoneType"),
        ('["not","an","object"]', "str decoding to list"),
        ("{not json", "str: invalid JSON"),
    ],
)
async def test_invalid_arguments_keep_the_step_target_and_reason(
    task_feature, arguments, expected
):
    result = await task_feature.run_workflow(
        [{"feature": "model_agent", "skill": "list_models", "args": arguments}]
    )
    assert result.status is ToolResultStatus.ERROR and result.data["failed"] == 1
    step = result.data["results"][0]
    assert (
        step["status"] == "failed"
        and "Step 0 (model_agent.list_models)" in step["error"]
    )
    assert expected in step["error"]


async def test_unknown_provider_has_a_clear_failure(task_feature):
    result = await task_feature.run_workflow(
        [{"feature": "missing", "skill": "missing"}]
    )
    assert result.status is ToolResultStatus.ERROR and result.data["failed"] == 1
    assert "Unknown agent: missing" in result.data["results"][0]["error"]


async def test_native_skill_rejection_keeps_its_reason(task_feature, manager):
    result = await task_feature.run_workflow(
        [{"feature": "model_agent", "skill": "missing"}]
    )
    assert result.status is ToolResultStatus.ERROR and result.data["failed"] == 1
    step = result.data["results"][0]
    assert step["status"] == "failed"
    assert "Unknown skill" in step.get("error", "")
    assert await manager.task_store.backend.fetch_all(
        "SELECT status FROM a2a_tasks"
    ) == [("rejected",)]


async def test_failure_does_not_hide_the_successful_steps(task_feature):
    result = await task_feature.run_workflow(
        [
            {"feature": "model_agent", "skill": "get_current_model"},
            {"feature": "missing", "skill": "missing"},
            {"feature": "wallet_feature", "skill": "check_balance"},
        ]
    )
    assert result.status is ToolResultStatus.PARTIAL
    assert result.data["completed"] == 2 and result.data["failed"] == 1
    assert [step["status"] for step in result.data["results"]] == [
        "completed",
        "failed",
        "completed",
    ]


@pytest.mark.parametrize("steps", [[], None, "invalid"])
async def test_invalid_workflow_input_is_rejected(task_feature, steps):
    result = await task_feature.run_workflow(steps)
    assert result.status is ToolResultStatus.ERROR and "non-empty list" in result.error


@pytest.mark.parametrize(
    "step", ["invalid", {"feature": "model_agent"}, {"skill": "list_models"}]
)
async def test_invalid_step_keeps_a_clear_failure(task_feature, step):
    result = await task_feature.run_workflow([step])
    assert result.status is ToolResultStatus.ERROR and result.data["failed"] == 1
    assert result.data["results"][0]["error"]


@pytest.mark.parametrize("method", ["list_available_skills", "run_workflow"])
async def test_a_missing_manager_is_an_explicit_failure(method):
    feature = TaskFeature(agent=None)
    result = await getattr(feature, method)(*([[]] if method == "run_workflow" else []))
    assert result.status is ToolResultStatus.ERROR and "not available" in result.error


@pytest.mark.parametrize(
    "outcome,expected,reason",
    [
        (ToolResult.failed("Unavailable"), ToolResultStatus.ERROR, "Unavailable"),
        (
            ToolResult.partial(confirmation="Some data", error="Incomplete"),
            ToolResultStatus.PARTIAL,
            "Incomplete",
        ),
        (
            {"success": False, "error": "Tool refused the operation"},
            ToolResultStatus.ERROR,
            "Tool refused the operation",
        ),
    ],
)
async def test_returned_semantic_failure_is_not_reported_as_success(
    task_feature, providers, manager, outcome, expected, reason
):
    providers.memory.supplier = lambda _attempt: outcome
    result = await task_feature.run_workflow(
        [{"feature": "memory_feature", "skill": "memory_status"}]
    )
    assert result.status is expected
    assert reason in result.data["results"][0]["error"]
    # Explicit ToolResult errors produce native FAILED receipts. Partial and
    # dictionary results retain their business semantics in the artifact.
    native_status = (
        "failed"
        if isinstance(outcome, ToolResult) and outcome.status is ToolResultStatus.ERROR
        else "completed"
    )
    assert await manager.task_store.backend.fetch_all(
        "SELECT status FROM a2a_tasks"
    ) == [(native_status,)]
    assert providers.memory.calls == 1


async def test_raised_tool_error_is_reported_without_stopping_other_steps(
    task_feature, providers
):
    def unavailable(_attempt):
        raise OSError("Tool storage unavailable")

    providers.memory.supplier = unavailable
    result = await task_feature.run_workflow(
        [
            {"feature": "memory_feature", "skill": "memory_status"},
            {"feature": "model_agent", "skill": "get_current_model"},
        ]
    )
    assert result.status is ToolResultStatus.PARTIAL
    assert "Tool storage unavailable" in result.data["results"][0]["error"]
    assert result.data["completed"] == result.data["failed"] == 1


async def test_explicit_retry_reports_the_attempt_that_succeeded(
    task_feature, providers
):
    providers.memory.supplier = lambda attempt: (
        ToolResult.failed("Retry")
        if attempt == 1
        else ToolResult.ok("Ready", data={"attempt": attempt})
    )
    result = await task_feature.run_workflow(
        [
            {
                "feature": "memory_feature",
                "skill": "memory_status",
                "max_retries": 1,
                "retry_delay_ms": 0,
            }
        ]
    )
    assert result.status is ToolResultStatus.OK and providers.memory.calls == 2
    assert len(result.data["results"]) == 1
    assert result.data["results"][0]["attempts"] == 2
    assert result.data["results"][0]["result"]["data"]["attempt"] == 2


async def test_execution_control_keeps_its_original_identity_and_is_not_retried(
    task_feature, providers
):
    denial = ExecutionAuthorityError("Original execution authority was withdrawn")

    def deny(_attempt):
        raise denial

    providers.memory.supplier = deny
    with pytest.raises(ExecutionAuthorityError) as observed:
        await task_feature.run_workflow(
            [
                {
                    "feature": "memory_feature",
                    "skill": "memory_status",
                    "max_retries": 2,
                    "retry_delay_ms": 0,
                }
            ]
        )
    assert observed.value is denial and providers.memory.calls == 1


async def test_nested_results_and_task_projections_remain_json_serializable(
    task_feature, providers, manager
):
    providers.memory.supplier = lambda _attempt: {
        "nested": ToolResult.ok("Nested", data={"count": 3})
    }
    result = await task_feature.run_workflow(
        [{"feature": "memory_feature", "skill": "memory_status"}]
    )
    assert result.status is ToolResultStatus.OK
    assert payload(result)["nested"]["data"]["count"] == 3
    json.dumps(result.to_dict(), allow_nan=False)
    [(task_id,)] = await manager.task_store.backend.fetch_all(
        "SELECT id FROM a2a_tasks"
    )
    for projection in (
        await task_feature.check_task_status(task_id),
        await task_feature.get_task_result(task_id),
    ):
        assert projection.status is ToolResultStatus.OK
        json.dumps(projection.to_dict(), allow_nan=False)
    stored = await manager.get_task_for_recipient(task_id, manager.did)
    recognized, value = embedded_json_value(stored.artifacts[0].parts[0])
    assert recognized and value == result.data["results"][0]["result"]


async def test_catalog_names_the_actual_registered_providers_and_skills(task_feature):
    result = await task_feature.list_available_skills()
    assert result.status is ToolResultStatus.OK
    assert result.data["feature_count"] == 3 and result.data["skill_count"] == 4
    assert set(result.data["features"]) == {
        "model_agent",
        "memory_feature",
        "wallet_feature",
    }
    skills = result.data["features"]["model_agent"]["skills"]
    assert {skill["skill"] for skill in skills} == {"list_models", "get_current_model"}
    assert all(
        skill["description"]
        for value in result.data["features"].values()
        for skill in value["skills"]
    )


def test_classify_step_result_unit_table():
    """Classify semantic outcomes across the utility's explicit accepted value forms."""
    from kestrel_sdk.tools.result import ToolResult

    cases = [
        ("completed", {"status": "ok", "confirmation": "ok"}, "completed", None),
        ("completed", {"status": "error", "error": "x"}, "failed", "x"),
        (
            "completed",
            {"status": "partial", "confirmation": "c", "error": "e"},
            "partial",
            "e",
        ),
        ("completed", ToolResult.ok("ok"), "completed", None),
        ("completed", ToolResult.failed("raw err"), "failed", "raw err"),
        (
            "completed",
            ToolResult.partial(confirmation="c", error="raw partial"),
            "partial",
            "raw partial",
        ),
        (
            "completed",
            {
                "success": True,
                "result": {"status": "ok", "confirmation": "ok"},
                "tool": "t",
            },
            "completed",
            None,
        ),
        (
            "completed",
            {
                "success": True,
                "result": {"status": "error", "error": "wrapped err"},
                "tool": "t",
            },
            "failed",
            "wrapped err",
        ),
        (
            "completed",
            {
                "success": True,
                "result": {
                    "status": "partial",
                    "confirmation": "c",
                    "error": "wrapped partial",
                },
                "tool": "t",
            },
            "partial",
            "wrapped partial",
        ),
        (
            "completed",
            {"success": True, "result": ToolResult.ok("ok"), "tool": "t"},
            "completed",
            None,
        ),
        (
            "completed",
            {
                "success": True,
                "result": ToolResult.failed("raw wrapped err"),
                "tool": "t",
            },
            "failed",
            "raw wrapped err",
        ),
        (
            "completed",
            {
                "success": True,
                "result": ToolResult.partial(
                    confirmation="c", error="raw wrapped partial"
                ),
                "tool": "t",
            },
            "partial",
            "raw wrapped partial",
        ),
        ("completed", {"success": False, "error": "legacy"}, "failed", "legacy"),
        ("completed", {"success": True, "model": "m"}, "completed", None),
        ("failed", {"status": "ok"}, "failed", None),
        ("failed", ToolResult.ok("would have been ok"), "failed", None),
        ("completed", "string result", "completed", None),
        ("completed", None, "completed", None),
    ]
    for transport, data, expected_status, expected_err in cases:
        actual_status, actual_err = TaskFeature._classify_step_result(transport, data)
        assert actual_status == expected_status, (
            f"transport={transport!r} data={data!r} → expected {expected_status}, got {actual_status}"
        )
        if expected_err is not None:
            assert actual_err == expected_err, (
                f"transport={transport!r} data={data!r} → expected error {expected_err!r}, got {actual_err!r}"
            )


@pytest.mark.parametrize("domain_status", ["ok", "partial"])
async def test_domain_status_is_not_mistaken_for_a_tool_result_envelope(
    task_feature,
    providers,
    domain_status,
):
    providers.memory.supplier = lambda _attempt: {"status": domain_status, "items": [1]}
    result = await task_feature.run_workflow(
        [
            {"feature": "memory_feature", "skill": "memory_status"},
        ]
    )
    assert result.status is ToolResultStatus.OK
    assert result.data["results"][0]["status"] == "completed"


def test_native_terminal_failures_override_an_optimistic_result():
    for state in ["failed", "canceled", "rejected"]:
        assert TaskFeature._classify_step_result(
            state,
            ToolResult.ok("Returned"),
            failure_message="Native refusal",
        ) == ("failed", "Native refusal")
