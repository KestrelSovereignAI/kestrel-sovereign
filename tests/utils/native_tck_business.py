"""Execute the official TCK's artifact scenarios through the real TaskManager."""

from __future__ import annotations

from kestrel_sdk.native_a2a.helpers.proto_helpers import (
    new_data_artifact,
    new_raw_artifact,
    new_text_artifact,
    new_text_message,
    new_url_artifact,
)
from kestrel_sdk.native_a2a.types import TaskState

from kestrel_sovereign.a2a.stores.unified.task_store import NativeTaskSnapshot
from kestrel_sovereign.a2a.task_manager import TaskManager


async def execute_tck_task(snapshot: NativeTaskSnapshot, *, manager: TaskManager) -> None:
    """Supply business results prescribed by scenarios/core_operations.feature.

    The message ID is the TCK's documented scenario selector. All artifacts,
    lifecycle transitions and responses still use the canonical native manager.
    """
    request = snapshot.input_request
    if request is None:
        raise ValueError("TCK business execution requires the original admitted request")
    message_id = request.message.message_id
    if message_id.startswith("tck-working-"):
        await manager.update_status(
            snapshot.task.id, TaskState.TASK_STATE_WORKING,
            message=new_text_message("Work is active until cancelled"),
            recipient_agent_id=snapshot.ownership.recipient_agent_id,
        )
        return
    if message_id.startswith("tck-input-required-"):
        await manager.update_status(
            snapshot.task.id, TaskState.TASK_STATE_INPUT_REQUIRED,
            message=new_text_message("Please provide the next input"),
            recipient_agent_id=snapshot.ownership.recipient_agent_id,
        )
        return
    if message_id.startswith("tck-reject-task-"):
        await manager.update_status(
            snapshot.task.id, TaskState.TASK_STATE_REJECTED,
            message=new_text_message("rejected"),
            recipient_agent_id=snapshot.ownership.recipient_agent_id,
        )
        return
    artifacts = []
    if message_id.startswith("tck-artifact-file-url-"):
        artifacts = [new_url_artifact(
            "output", "https://example.com/output.txt",
            media_type="text/plain", filename="output.txt",
        )]
    elif message_id.startswith("tck-artifact-file-"):
        artifacts = [new_raw_artifact(
            "output", b"Generated file content",
            media_type="text/plain", filename="output.txt",
        )]
    elif message_id.startswith("tck-artifact-text-"):
        artifacts = [new_text_artifact("output", "Generated text content")]
    elif message_id.startswith("tck-artifact-data-"):
        artifacts = [new_data_artifact("output", {"key": "value", "count": 42})]
    await manager.complete_task(
        snapshot.task.id, "Hello from TCK", artifacts=artifacts,
        recipient_agent_id=snapshot.ownership.recipient_agent_id,
    )
