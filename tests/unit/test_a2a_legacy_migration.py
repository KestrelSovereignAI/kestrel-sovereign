"""Offline draft snapshot conversion preserves data without minting authority."""

import copy
import json

import pytest
from a2a.types import TaskState
from google.protobuf.json_format import MessageToDict

from kestrel_sovereign.a2a.extensions import (
    PEER_STOP_URI,
    QUESTION_CORRELATION_URI,
    SKILL_HINT_URI,
)
from kestrel_sovereign.a2a.migrate_legacy import (
    LegacyTaskMigrationError,
    convert_legacy_task,
)
from kestrel_sovereign.a2a.stores.unified.context_store import ContextOwnership


@pytest.fixture
def archived():
    return {
        "id": "old-task",
        "session_id": "old-session",
        "status": "completed",
        "creator_agent_id": "did:actual:sender",
        "recipient_agent_id": "did:recipient",
        "history": [
            {
                "role": "user",
                "parts": [
                    {"type": "text", "text": "Input"},
                    {"type": "data", "data": {"nested": [None, {}, []]}},
                ],
            }
        ],
        "message": {"role": "agent", "parts": [{"type": "text", "text": "Answer"}]},
        "artifacts": [
            {
                "name": "file",
                "index": 0,
                "lastChunk": True,
                "parts": [
                    {
                        "type": "file",
                        "file": {
                            "name": "data.bin",
                            "mimeType": "application/octet-stream",
                            "bytes": "AAEC",
                        },
                    }
                ],
            }
        ],
        "metadata": {
            "sender": "forged",
            "sender_verified": True,
            "signature": "old-proof",
            "skill": "prior-skill",
            "a2a_verb": "question",
            "business": {"keep": 42},
        },
    }


def owner(**changes):
    return ContextOwnership(
        **(
            {
                "context_id": "reviewed-context",
                "tenant": "tenant",
                "recipient_agent_id": "did:recipient",
                "creator_agent_id": "did:actual:sender",
            }
            | changes
        )
    )


def test_offline_conversion_retains_parts_and_assigns_stable_native_ids(archived):
    before = copy.deepcopy(archived)
    task = convert_legacy_task(archived, ownership=owner())
    assert task == convert_legacy_task(archived, ownership=owner())
    assert archived == before and task.id == "old-task"
    assert (
        task.context_id == "reviewed-context"
        and task.status.state == TaskState.TASK_STATE_COMPLETED
    )
    assert not task.status.HasField("timestamp")
    assert task.history[0].role and task.history[0].message_id
    assert (
        task.history[0].task_id == task.id
        and task.history[0].context_id == task.context_id
    )
    assert MessageToDict(task.history[0].parts[1])["data"] == {"nested": [None, {}, []]}
    assert task.artifacts[0].parts[0].raw == b"\x00\x01\x02"
    metadata = MessageToDict(task.metadata)
    assert (
        "sender" not in metadata
        and "sender_verified" not in metadata
        and "signature" not in metadata
    )
    assert metadata[SKILL_HINT_URI]["skill_id"] == "prior-skill"
    assert metadata[QUESTION_CORRELATION_URI]["question_id"] == task.id
    assert metadata["business"] == {"keep": 42}


def test_draft_metadata_cannot_introduce_native_peer_control(archived):
    archived["metadata"][PEER_STOP_URI] = {"version": 1, "intent": {}}
    with pytest.raises(LegacyTaskMigrationError, match="Mixed native/draft"):
        convert_legacy_task(archived, ownership=owner())


@pytest.mark.parametrize("field", ["creator_agent_id", "recipient_agent_id"])
def test_claimed_metadata_cannot_replace_reviewed_durable_provenance(archived, field):
    with pytest.raises(LegacyTaskMigrationError, match="conflicts"):
        convert_legacy_task(archived, ownership=owner(**{field: "did:impostor"}))
    del archived[field]
    # Missing columns require the explicit reviewed plan, never metadata.
    result = convert_legacy_task(archived, ownership=owner())
    assert result.context_id == "reviewed-context"
    with pytest.raises(TypeError, match="reviewed ContextOwnership"):
        convert_legacy_task(archived, ownership=None)


@pytest.mark.parametrize(
    "state", ["submitted", "working", "input-required", "auth-required", "unknown"]
)
def test_live_or_unknown_tasks_cannot_be_silently_settled_during_upgrade(
    archived, state
):
    archived["status"] = state
    with pytest.raises(LegacyTaskMigrationError):
        convert_legacy_task(archived, ownership=owner())


@pytest.mark.parametrize(
    "part",
    [
        {"type": "future", "body": "unknown"},
        {"type": "text", "text": 1},
        {"type": "text", "text": "x", "hidden": "discarded?"},
        {"type": "data", "data": [1, 2]},
        {"type": "data", "data": {"bad": float("nan")}},
        {"type": "file", "file": {"bytes": "invalid!"}},
        {"type": "file", "file": {"uri": "https://example.test/a", "bytes": "AAEC"}},
        {"type": "file", "file": {}},
    ],
)
def test_malformed_historical_content_fails_without_dropping_data(archived, part):
    archived["history"][0]["parts"] = [part]
    before = json.dumps(archived, sort_keys=True)
    with pytest.raises(LegacyTaskMigrationError):
        convert_legacy_task(archived, ownership=owner())
    assert json.dumps(archived, sort_keys=True) == before


def test_migration_does_not_fetch_file_uris_or_require_remote_clients(archived):
    archived["artifacts"][0]["parts"] = [
        {
            "type": "file",
            "file": {
                "uri": "https://example.test/output",
                "name": "output.txt",
                "mimeType": "text/plain",
            },
        }
    ]
    task = convert_legacy_task(archived, ownership=owner())
    assert task.artifacts[0].parts[0].url == "https://example.test/output"
    assert task.artifacts[0].parts[0].filename == "output.txt"


def test_sql_timestamps_are_not_interpreted_as_fabricated_native_status_time(archived):
    from datetime import UTC, datetime

    archived["updated_at"] = datetime(2025, 4, 1, tzinfo=UTC)
    task = convert_legacy_task(archived, ownership=owner())
    assert not task.status.HasField("timestamp")
    assert archived["updated_at"].year == 2025


def test_explicit_draft_artifact_append_becomes_one_native_artifact(archived):
    archived["artifacts"] = [
        {
            "index": 0,
            "name": "output",
            "metadata": {"version": 1},
            "parts": [{"type": "text", "text": "first"}],
        },
        {
            "index": 0,
            "append": True,
            "lastChunk": True,
            "name": "output",
            "metadata": {"version": 1},
            "parts": [{"type": "text", "text": "second"}],
        },
    ]
    task = convert_legacy_task(archived, ownership=owner())
    assert len(task.artifacts) == 1
    assert [part.text for part in task.artifacts[0].parts] == ["first", "second"]
    assert (
        MessageToDict(task.status.message.metadata)[QUESTION_CORRELATION_URI]["kind"]
        == "answer"
    )


@pytest.mark.parametrize(
    "fault", ["missing", "ambiguous", "closed", "name", "metadata"]
)
def test_ambiguous_artifact_append_requires_a_reviewed_plan(archived, fault):
    first = {"index": 0, "name": "output", "parts": [{"type": "text", "text": "first"}]}
    last = {"index": 0, "append": True, "parts": [{"type": "text", "text": "last"}]}
    archived["artifacts"] = [first, last]
    if fault == "missing":
        last["index"] = 1
    elif fault == "ambiguous":
        archived["artifacts"].insert(1, copy.deepcopy(first))
    elif fault == "closed":
        first["lastChunk"] = True
    elif fault == "name":
        last["name"] = "another artifact"
    else:
        last["metadata"] = {"different": True}
    with pytest.raises(LegacyTaskMigrationError):
        convert_legacy_task(archived, ownership=owner())


def test_reviewed_tenant_cannot_override_existing_durable_tenant(archived):
    archived["tenant"] = "original-tenant"
    with pytest.raises(LegacyTaskMigrationError, match="durable tenant"):
        convert_legacy_task(archived, ownership=owner())


def test_historical_cancellation_does_not_invent_a_question_answer(archived):
    archived["status"] = "canceled"
    archived["message"] = {
        "role": "agent",
        "parts": [{"type": "text", "text": "Task canceled by did:actual:sender"}],
    }
    task = convert_legacy_task(archived, ownership=owner())
    assert MessageToDict(task.metadata)[QUESTION_CORRELATION_URI]["kind"] == "question"
    assert QUESTION_CORRELATION_URI not in MessageToDict(task.status.message.metadata)
    assert task.status.message.parts[0].text == "Task canceled by did:actual:sender"
