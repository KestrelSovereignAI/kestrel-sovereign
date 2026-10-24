"""Native inbox regressions for #1433, #1946 and #1525 on real storage.

Requests and replies remain distinct, sender identity comes from ownership,
filters query the complete inbox, inputs are Parts, and Artifacts are outputs.
"""

from datetime import UTC, datetime, timedelta

from kestrel_sdk.native_a2a.helpers.proto_helpers import (
    new_data_artifact,
    new_data_part,
    new_text_artifact,
    new_text_message,
)
from kestrel_sdk.native_a2a.types import Part, Role, SendMessageRequest, TaskState
from kestrel_sdk.tools.result import ToolResultStatus

from tests.unit.test_a2a_native_consumers import inbox as inbox  # noqa: PLC0414


async def admit(
    inbox, text="PING-1428", *, sender="did:kestrel:emma", task_type="peer", parts=()
):
    manager, _feature, _agent = inbox
    message = new_text_message(text, role=Role.ROLE_USER)
    message.parts.extend(parts)
    return await manager.create_task(
        SendMessageRequest(
            tenant="tenant",
            message=message,
            metadata={"sender": "untrusted metadata", "task_type": task_type},
        ),
        creator_agent_id=sender,
    )


async def test_check_task_status_returns_request_content_for_submitted_task(inbox):
    _manager, feature, _agent = inbox
    task = await admit(inbox)
    result = await feature.check_task_status(task.id)
    assert result.status is ToolResultStatus.OK
    assert result.data["request_content"] == "PING-1428"
    assert result.data["sender"] == "did:kestrel:emma"
    assert result.data["status"] == "submitted"
    assert result.data["message"] is None
    assert "PING-1428" in result.confirmation
    assert "did:kestrel:emma" in result.confirmation
    assert "untrusted metadata" not in result.confirmation


async def test_check_task_status_returns_both_request_and_reply_for_completed_task(
    inbox,
):
    _manager, feature, _agent = inbox
    task = await admit(inbox, "ping?")
    assert (
        await feature.respond_to_a2a_task(task.id, "pong")
    ).status is ToolResultStatus.OK
    result = await feature.check_task_status(task.id)
    assert result.status is ToolResultStatus.OK
    assert result.data["request_content"] == "ping?"
    assert result.data["message"] == "pong"
    assert "ping?" in result.confirmation and "pong" in result.confirmation


async def test_list_my_tasks_includes_request_content_per_row(inbox):
    _manager, feature, _agent = inbox
    first = await admit(inbox, "please reply with PONG")
    second = await admit(inbox, "urgent: status", sender="did:kestrel:claw")
    result = await feature.list_my_tasks(limit=10)
    assert result.status is ToolResultStatus.OK
    rows = {row["task_id"]: row for row in result.data["tasks"]}
    assert len(rows) == 2
    assert rows[first.id]["request_content"] == "please reply with PONG"
    assert rows[first.id]["sender"] == "did:kestrel:emma"
    assert rows[second.id]["request_content"] == "urgent: status"
    assert rows[second.id]["sender"] == "did:kestrel:claw"


async def test_list_my_tasks_status_filter_queries_full_table_not_pending(inbox):
    _manager, feature, _agent = inbox
    completed = await admit(inbox, "ping")
    await feature.respond_to_a2a_task(completed.id, "PONG")
    pending = await admit(inbox, "waiting")
    default = await feature.list_my_tasks()
    assert [row["task_id"] for row in default.data["tasks"]] == [pending.id]
    result = await feature.list_my_tasks(status="completed", limit=5)
    assert result.status is ToolResultStatus.OK
    assert [row["task_id"] for row in result.data["tasks"]] == [completed.id]
    assert result.data["tasks"][0]["status"] == "completed"


async def test_list_my_tasks_invalid_status_rejected(inbox):
    _manager, feature, _agent = inbox
    result = await feature.list_my_tasks(status="bogus")
    assert result.status is ToolResultStatus.ERROR
    assert "Invalid task status" in result.error


async def test_list_my_tasks_type_filter_overfetches_then_truncates(inbox):
    manager, feature, agent = inbox
    matching = await admit(inbox, "matching request", task_type="foo")
    created = [matching]
    for _ in range(2):
        other = await admit(inbox, "unrelated request", task_type="other")
        created.append(other)
    for offset, task in enumerate(created):
        await manager.task_store.backend.execute(
            "UPDATE a2a_tasks SET created_at = ? WHERE id = ?",
            (
                manager.task_store.to_timestamp_param(
                    datetime(2026, 10, 20, tzinfo=UTC) + timedelta(seconds=offset)
                ),
                task.id,
            ),
        )
        await feature.respond_to_a2a_task(task.id, "done")
    latest = await manager.list_task_snapshots(
        recipient_agent_id=agent.did,
        status=TaskState.TASK_STATE_COMPLETED,
        limit=1,
    )
    assert len(latest) == 1 and latest[0].task.id != matching.id
    result = await feature.list_my_tasks(status="completed", task_type="foo", limit=1)
    assert result.status is ToolResultStatus.OK
    assert [row["task_id"] for row in result.data["tasks"]] == [matching.id]
    assert result.data["tasks"][0]["task_type"] == "foo"


async def test_get_task_result_includes_request_content_for_completed_task(inbox):
    _manager, feature, _agent = inbox
    task = await admit(inbox, "please ack")
    await feature.respond_to_a2a_task(task.id, "acknowledged")
    result = await feature.get_task_result(task.id)
    assert result.status is ToolResultStatus.OK
    assert result.data["request_content"] == "please ack"
    assert result.data["message"] == "acknowledged"
    assert result.data["sender"] == "did:kestrel:emma"


async def test_check_task_status_preserves_sender_input_parts(inbox):
    _manager, feature, _agent = inbox
    task = await admit(
        inbox,
        "orchestrate",
        parts=[
            Part(text="step one", filename="plan.txt"),
            Part(text="step two"),
            new_data_part({"ref_type": "memory", "id": "m1"}),
        ],
    )
    result = await feature.check_task_status(task.id)
    assert result.status is ToolResultStatus.OK
    assert result.data["request_content"] == "orchestrate\nstep one\nstep two"
    parts = result.data["request"]["parts"]
    assert parts[1] == {"text": "step one", "filename": "plan.txt"}
    assert parts[2] == {"text": "step two"}
    assert parts[3]["data"] == {"ref_type": "memory", "id": "m1"}
    assert result.data["artifacts"] == []


async def test_check_task_status_preserves_ordered_native_output_artifacts(inbox):
    manager, feature, agent = inbox
    task = await admit(inbox)
    plan = new_text_artifact("plan", "step one", description="operating model")
    plan.parts.append(Part(text="step two"))
    reference = new_data_artifact("references", {"ref_type": "memory", "id": "m1"})
    reference.metadata.update({"kind": "reference"})
    for artifact in (plan, reference):
        await manager.add_artifact(task.id, artifact, recipient_agent_id=agent.did)
    result = await feature.check_task_status(task.id)
    assert result.status is ToolResultStatus.OK
    artifacts = result.data["artifacts"]
    assert [artifact["name"] for artifact in artifacts] == ["plan", "references"]
    assert [part["text"] for part in artifacts[0]["parts"]] == ["step one", "step two"]
    assert artifacts[0]["description"] == "operating model"
    assert artifacts[1]["metadata"] == {"kind": "reference"}
    assert artifacts[1]["parts"][0]["data"] == {"ref_type": "memory", "id": "m1"}
