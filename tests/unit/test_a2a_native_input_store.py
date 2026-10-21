"""Creator input persistence and content integrity on the real primary backend."""

import json

import pytest
from google.protobuf.json_format import MessageToDict

from kestrel_sovereign.a2a.stores.unified.task_store import TaskInputConflictError
from tests.unit.test_a2a_native_task_manager import manager as manager
from tests.unit.test_a2a_native_task_manager import request
from tests.utils.native_push import managed_push_service


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


async def test_inline_callback_secrets_are_not_persisted_or_exposed_to_cognition(
    manager,
    monkeypatch,
):
    from kestrel_sovereign.a2a.native_input import native_request_digest
    from tests.unit.test_a2a_native_input import request as inline_request

    original = inline_request()
    original.configuration.task_push_notification_config.url = (
        "https://1.1.1.1/private-path?secret=do-not-persist"
    )
    async with managed_push_service(manager, monkeypatch, pause_delivery=True):
        task = await manager.create_task(original, creator_agent_id="did:creator")
        row = await manager.task_store.backend.fetch_one(
            "SELECT request_json, request_digest FROM a2a_task_inputs WHERE task_id = ?",
            (task.id,),
        )
        for secret in [
            "1.1.1.1",
            "do-not-persist",
            "callback-validation-secret",
            "callback-authorization-secret",
        ]:
            assert secret not in row[0]
        snapshot = await manager.get_task_snapshot(
            task.id, recipient_agent_id=manager.did
        )
        assert not snapshot.input_request.configuration.HasField(
            "task_push_notification_config"
        )
        assert snapshot.input_request.message == original.message
        assert snapshot.input_wire_request_digest == native_request_digest(
            original, "SendMessage"
        )
        assert snapshot.input_request_digest == row[1]
        assert (
            native_request_digest(snapshot.input_request, "SendMessage")
            != snapshot.input_wire_request_digest
        )


async def test_inline_secret_changes_conflict_even_when_execution_projection_is_identical(
    manager,
):
    from a2a.types import SendMessageRequest

    from tests.unit.test_a2a_native_input import request as inline_request

    initial = await manager.create_task(request(), creator_agent_id="did:creator")
    continuation = inline_request()
    continuation.message.task_id = initial.id
    continuation.message.context_id = initial.context_id
    first, appended = await manager.task_store.append_creator_input(
        continuation, creator_agent_id="did:creator"
    )
    assert appended and first.input_revision == 1
    stored = await manager.task_store.backend.fetch_one(
        "SELECT request_json FROM a2a_task_inputs WHERE task_id = ? AND input_revision = 1",
        (initial.id,),
    )
    assert "callback-authorization-secret" not in stored[0]

    replay = SendMessageRequest.FromString(continuation.SerializeToString())
    repeated, appended = await manager.task_store.append_creator_input(
        replay, creator_agent_id="did:creator"
    )
    assert not appended and repeated.input_revision == 1
    replay.configuration.task_push_notification_config.authentication.credentials = (
        "different-secret"
    )
    with pytest.raises(TaskInputConflictError, match="Message ID"):
        await manager.task_store.append_creator_input(
            replay, creator_agent_id="did:creator"
        )
    current = await manager.get_task_snapshot(
        initial.id, recipient_agent_id=manager.did
    )
    assert current.input_revision == 1
    assert not current.input_request.configuration.HasField(
        "task_push_notification_config"
    )


async def test_cognition_refuses_callback_material_even_if_the_execution_checksum_is_rewritten(
    manager,
):
    import hashlib

    from google.protobuf.json_format import MessageToJson

    from tests.unit.test_a2a_native_input import request as inline_request

    original = inline_request(callback=False)
    task = await manager.create_task(original, creator_agent_id="did:creator")
    corrupted = inline_request()
    corrupted.message.CopyFrom(original.message)
    await manager.task_store.backend.execute(
        "UPDATE a2a_task_inputs SET request_json = ?, request_digest = ? WHERE task_id = ?",
        (
            MessageToJson(corrupted),
            hashlib.sha256(corrupted.SerializeToString(deterministic=True)).hexdigest(),
            task.id,
        ),
    )
    snapshot = await manager.get_task_snapshot(task.id, recipient_agent_id=manager.did)
    with pytest.raises(TaskInputConflictError, match="Callback credentials"):
        _ = snapshot.input_request


@pytest.mark.parametrize("digest", [None, "", "not-a-sha256", "g" * 64])
async def test_cognition_requires_a_well_formed_original_wire_identity(manager, digest):
    from dataclasses import replace

    original = request()
    task = await manager.create_task(original, creator_agent_id="did:creator")
    snapshot = await manager.get_task_snapshot(task.id, recipient_agent_id=manager.did)
    invalid = replace(snapshot, input_wire_request_digest=digest)
    with pytest.raises(TaskInputConflictError, match="wire request digest"):
        _ = invalid.input_request
