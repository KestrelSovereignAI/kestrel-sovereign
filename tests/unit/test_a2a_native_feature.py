"""Real decorated feature tools produce native task values without wire models."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from kestrel_sovereign._vendor.a2a.helpers.proto_helpers import new_task, new_text_artifact, new_text_message
from kestrel_sovereign._vendor.a2a.types import TaskState
from kestrel_sdk.tools.result import ToolResult

from kestrel_sovereign.a2a.projections import embedded_json_value
from kestrel_sovereign.features.base import Feature, tool


class NativeFeature(Feature):
    @property
    def tool_description(self):
        return "Exercise native feature task results"

    async def initialize(self):
        pass

    @tool(name="native_echo", description="Return structured values unchanged")
    async def native_echo(self, value: dict) -> dict:
        return value

    @tool(name="native_failure", description="Return a typed tool failure")
    async def native_failure(self) -> ToolResult:
        return ToolResult.failed("Deliberate tool rejection")


def feature():
    return NativeFeature(SimpleNamespace())


def task(**metadata):
    result = new_task(
        "native-task",
        "native-context",
        TaskState.TASK_STATE_SUBMITTED,
        history=[new_text_message("request")],
    )
    result.metadata.update(metadata)
    return result


async def test_native_feature_preserves_history_and_existing_artifacts_and_produces_native_ids():
    original = task(skill="native_echo", args={"value": {"nested": [None, {}, []]}})
    before = original.history[0].SerializeToString()
    existing = new_text_artifact("earlier", "Keep this")
    original.artifacts.append(existing)
    result = await feature().handle_task(original)
    assert result.status.state == TaskState.TASK_STATE_COMPLETED
    assert result.history[0].SerializeToString() == before
    assert result.artifacts[0] == existing
    assert result.artifacts[1].artifact_id and result.status.message.message_id
    assert embedded_json_value(result.artifacts[1].parts[0])[1]["result"] == {
        "nested": [None, {}, []]
    }


@pytest.mark.parametrize(
    "metadata",
    [{}, {"skill": 1}, {"skill": "missing"}, {"skill": "native_echo", "args": []}],
)
async def test_missing_unknown_or_malformed_skills_are_rejected_without_execution(
    metadata,
):
    result = await feature().handle_task(task(**metadata))
    assert result.status.state == TaskState.TASK_STATE_REJECTED
    assert result.status.message.message_id
    assert list(result.artifacts) == []


async def test_native_feature_does_not_report_a_typed_tool_error_as_completed():
    result = await feature().handle_task(task(skill="native_failure"))
    assert result.status.state == TaskState.TASK_STATE_FAILED
    assert embedded_json_value(result.artifacts[0].parts[0])[1]["status"] == "error"


async def test_native_feature_refuses_private_task_objects():
    with pytest.raises(TypeError, match="official A2A Task"):
        await feature().handle_task({"sessionId": "legacy"})


def test_dynamic_tools_no_longer_carry_a_second_cached_skill_model():
    exposed = feature().get_tools()
    assert {item.name for item in exposed} == {"native_echo", "native_failure"}
    assert all(not hasattr(item, "agent_skill") for item in exposed)


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"skill": "missing"},
        {"skill": "native_echo", "args": {"value": {"nested": [None, {}]}}},
        {"skill": "native_echo", "args": {}},
        {"skill": "native_failure"},
    ],
)
async def test_all_feature_status_messages_identify_their_parent_task(metadata):
    result = await feature().handle_task(task(**metadata))
    assert result.status.message.task_id == result.id
    assert result.status.message.context_id == result.context_id
