"""Creator input persistence and content integrity on the real primary backend."""

import json

import pytest
from google.protobuf.json_format import MessageToDict

from kestrel_sovereign.a2a.stores.unified.task_store import TaskInputConflictError
from tests.unit.test_a2a_native_task_manager import manager as manager
from tests.unit.test_a2a_native_task_manager import request


async def test_initial_request_is_retained_with_its_task_in_one_admission(manager):
    original = request(nested={"list": [None, {}, []]})
    task = await manager.create_task(original, creator_agent_id="did:creator")
    snapshot = await manager.get_task_snapshot(task.id, recipient_agent_id=manager.did)
    assert snapshot.input_revision == 0 and snapshot.input_request == original
    assert snapshot.cognition_wake_eligible is True
    assert await manager.task_store.backend.fetch_one(
        "SELECT COUNT(*) FROM a2a_task_inputs"
    ) == (1,)
    altered = MessageToDict(original)
    altered["message"]["parts"][0]["text"] = "Changed after admission"
    await manager.task_store.backend.execute(
        "UPDATE a2a_task_inputs SET request_json = ? WHERE task_id = ?",
        (json.dumps(altered), task.id),
    )
    corrupted = await manager.get_task_snapshot(task.id, recipient_agent_id=manager.did)
    with pytest.raises(TaskInputConflictError, match="admitted request digest"):
        _ = corrupted.input_request
