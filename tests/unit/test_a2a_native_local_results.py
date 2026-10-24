"""Local commands and real workflows must preserve exact JSON integer results."""

import os
from types import SimpleNamespace

from kestrel_sdk.tools.result import ToolResult, ToolResultStatus

from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.features.tasks.feature import TaskFeature
from tests.unit.test_a2a_native_task_manager import manager as manager


class ExactResultFeature(Feature):
    @property
    def tool_description(self):
        return "Preserve exact local result data"

    async def initialize(self):
        pass

    @tool(name="list_names", description="Return a POSIX file name losslessly")
    async def list_names(self) -> ToolResult:
        self.effects += 1
        return ToolResult.ok(
            confirmation="Listed names",
            data={"names": [os.fsdecode(b"report-\xff.txt")]},
        )

    @tool(name="consume_names", description="Consume the exact returned file name")
    async def consume_names(self, value: dict) -> ToolResult:
        self.observed = value
        assert os.fsencode(value["data"]["names"][0]) == b"report-\xff.txt"
        return ToolResult.ok(confirmation="Consumed the file name")

    @tool(
        name="exact_result", description="Return an integer beyond binary64 precision"
    )
    async def exact_result(self) -> ToolResult:
        self.effects += 1
        return ToolResult.ok(
            confirmation="Returned exact identifiers",
            data={"small": 3, "snowflake": 1153921230456789012},
        )

    @tool(name="consume_exact_result", description="Consume the previous exact result")
    async def consume_exact_result(self, value: dict) -> ToolResult:
        assert type(value["data"]["small"]) is int
        assert value["data"]["snowflake"] == 1153921230456789012
        self.observed = value
        return ToolResult.ok(confirmation="Consumed the exact result")


def provider(manager):
    feature = ExactResultFeature(
        SimpleNamespace(did=manager.did, _execution_custody=None)
    )
    feature.name = "exact"
    feature.effects = 0
    manager.register_provider(
        "exact",
        feature.get_agent_skills(),
        feature,
        {"!exact": "exact_result", "!names": "list_names"},
    )
    return feature


async def test_local_command_retains_integer_types_and_complete_values(manager):
    feature = provider(manager)
    result = await manager.execute_command("!exact")
    assert feature.effects == 1
    assert isinstance(result.get("data"), dict), (
        "Command discarded its structured JSON result"
    )
    assert type(result["data"]["small"]) is int
    assert result["data"]["snowflake"] == 1153921230456789012


async def test_local_command_preserves_surrogateescaped_names_after_one_effect(manager):
    feature = provider(manager)
    result = await manager.execute_command("!names")
    assert os.fsencode(result["data"]["names"][0]) == b"report-\xff.txt"
    assert feature.effects == 1
    assert await manager.task_store.backend.fetch_all(
        "SELECT status FROM a2a_tasks"
    ) == [("completed",)]


async def test_workflow_carries_surrogateescaped_names_without_repeating_effect(
    manager,
):
    feature = provider(manager)
    workflow = TaskFeature(SimpleNamespace(did=manager.did))
    workflow.set_task_manager(manager)
    result = await workflow.run_workflow(
        [
            {"feature": "exact", "skill": "list_names"},
            {
                "feature": "exact",
                "skill": "consume_names",
                "args": {"value": "{{prev.result}}"},
            },
        ]
    )
    assert result.status is ToolResultStatus.OK
    assert result.data["completed"] == 2 and result.data["failed"] == 0
    assert feature.effects == 1
    assert os.fsencode(feature.observed["data"]["names"][0]) == b"report-\xff.txt"


async def test_real_workflow_carries_exact_result_into_the_next_typed_tool(manager):
    feature = provider(manager)
    workflow = TaskFeature(SimpleNamespace(did=manager.did))
    workflow.set_task_manager(manager)
    result = await workflow.run_workflow(
        [
            {"feature": "exact", "skill": "exact_result"},
            {
                "feature": "exact",
                "skill": "consume_exact_result",
                "args": {"value": "{{prev.result}}"},
            },
        ]
    )
    assert result.status is ToolResultStatus.OK
    assert result.data["completed"] == 2 and result.data["failed"] == 0
    assert feature.effects == 1
    assert feature.observed["data"]["snowflake"] == 1153921230456789012


async def test_exact_result_bytes_survive_primary_storage_and_official_sdk_json(
    manager,
):
    from kestrel_sdk.native_a2a.types import Task
    from google.protobuf.json_format import MessageToDict, ParseDict

    from kestrel_sovereign.a2a.projections import embedded_json_value

    provider(manager)
    task = await manager.execute_skill("exact", "exact_result", {}, sync=True)
    stored = await manager.get_task_for_recipient(task.id, manager.did)
    decoded = ParseDict(MessageToDict(stored), Task())
    part = decoded.artifacts[0].parts[0]
    assert part.HasField("raw") and part.media_type == "application/json"
    recognized, value = embedded_json_value(part)
    assert recognized and type(value["data"]["small"]) is int
    assert value["data"]["snowflake"] == 1153921230456789012
