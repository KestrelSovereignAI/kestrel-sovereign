"""Native protobuf storage, authority, and atomic context admission contracts."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from kestrel_sdk.native_a2a.helpers.proto_helpers import (
    new_data_message,
    new_task,
    new_text_artifact,
    new_text_message,
)
from kestrel_sdk.native_a2a.types import Task, TaskState, TaskStatus
from kestrel_sdk.native_a2a.utils.errors import InvalidParamsError
from google.protobuf.json_format import MessageToDict
from google.protobuf.timestamp_pb2 import Timestamp

from kestrel_sovereign.a2a.native_context import NativeRecipientBinding
from kestrel_sovereign.a2a.scope import NativeRecipientScope
from kestrel_sovereign.a2a.stores.base import json_dumps, json_loads
from kestrel_sovereign.a2a.stores.unified.context_store import ContextStore
from kestrel_sovereign.a2a.stores.unified.task_store import (
    TaskAlreadyExistsError,
    TaskMutationAuthorizationError,
    TaskStore,
)
from kestrel_sovereign.storage.db.interface import QueryError, TransactionError


@pytest_asyncio.fixture
async def store(native_a2a_backend_factory):
    backend = native_a2a_backend_factory()
    await backend.connect()
    tasks = TaskStore(backend, scope=NativeRecipientScope("did:kestrel:bob"))
    try:
        await tasks.initialize()
        yield tasks
    finally:
        await backend.close()


def task(state=TaskState.TASK_STATE_SUBMITTED):
    return new_task(
        "task-one", "context-one", state, history=[new_text_message("original request")]
    )


async def create(store, value):
    await store.create(
        value,
        creator_agent_id="did:kestrel:alice",
        recipient_agent_id="did:kestrel:bob",
    )


async def get(store):
    return await store.get_for_creator(
        "task-one", "did:kestrel:alice", recipient_agent_id="did:kestrel:bob"
    )


async def wake_candidate_ids(store, *, live, terminal):
    return {
        candidate.task.id
        for candidate in await store.list_cognition_wake_candidates(
            recipient_agent_id=store.scope.recipient_id,
            live_changed_since=live, terminal_changed_since=terminal,
        )
    }


async def age_task_revision(store, task_id, timestamp):
    await store.backend.execute(
        "UPDATE a2a_tasks SET lifecycle_updated_at = ?, updated_at = ? WHERE id = ?",
        (store.to_timestamp_param(timestamp), store.to_timestamp_param(timestamp), task_id),
    )


async def test_wake_horizons_are_independent_for_live_and_terminal_native_tasks(store):
    for name, state in (
        ("live-submitted", TaskState.TASK_STATE_SUBMITTED),
        ("live-working", TaskState.TASK_STATE_WORKING),
        ("terminal", TaskState.TASK_STATE_COMPLETED),
    ):
        value = task(state)
        value.id = name
        await create(store, value)
    foreign = TaskStore(store.backend, scope=NativeRecipientScope("did:kestrel:other"))
    value = task()
    value.id = "foreign-live"
    await foreign.create(
        value, creator_agent_id="did:kestrel:alice", recipient_agent_id=foreign.scope.recipient_id,
    )
    now = store.now_utc()
    past, future = now - timedelta(days=1), now + timedelta(days=1)
    assert await wake_candidate_ids(store, live=past, terminal=future) == {"live-submitted", "live-working"}
    assert await wake_candidate_ids(store, live=future, terminal=past) == {"terminal"}
    assert await wake_candidate_ids(store, live=past, terminal=past) == {"live-submitted", "live-working", "terminal"}


@pytest.mark.parametrize("transition", ["lifecycle", "terminal", "cancel"])
async def test_each_native_lifecycle_writer_restamps_the_wake_revision(store, transition):
    original = task()
    await create(store, original)
    horizon = store.now_utc() - timedelta(days=14)
    await age_task_revision(store, original.id, horizon - timedelta(days=16))
    assert not await wake_candidate_ids(store, live=horizon, terminal=horizon)
    outcome = Task.FromString(original.SerializeToString())
    if transition == "lifecycle":
        outcome.status.state = TaskState.TASK_STATE_WORKING
        committed = await store.save_recipient_lifecycle(
            outcome, baseline=original, recipient_agent_id=store.scope.recipient_id,
            expected_state=TaskState.TASK_STATE_SUBMITTED,
        )
    elif transition == "terminal":
        outcome.status.state = TaskState.TASK_STATE_COMPLETED
        committed = await store.save_recipient_terminal_outcome(
            outcome, baseline=original, recipient_agent_id=store.scope.recipient_id,
            operation_id="wake-window-terminal",
        )
    else:
        committed = await store.cancel_if_authorized(original.id, actor_agent_id="did:kestrel:alice")
    assert committed is not None
    snapshot = await store.get_task_snapshot(original.id, recipient_agent_id=store.scope.recipient_id)
    assert snapshot.lifecycle_revision == 1
    assert await wake_candidate_ids(store, live=horizon, terminal=horizon) == {original.id}


async def test_artifact_progress_never_reopens_an_expired_native_wake_window(store):
    value = task(TaskState.TASK_STATE_WORKING)
    await create(store, value)
    horizon = store.now_utc() - timedelta(days=14)
    await age_task_revision(store, value.id, horizon - timedelta(days=16))
    before = await store.backend.fetch_one(
        "SELECT lifecycle_revision, lifecycle_updated_at, updated_at FROM a2a_tasks WHERE id = ?",
        (value.id,),
    )
    await store.add_artifact(
        value.id, new_text_artifact("progress", "still going"),
        recipient_agent_id=store.scope.recipient_id,
    )
    after = await store.backend.fetch_one(
        "SELECT lifecycle_revision, lifecycle_updated_at, updated_at FROM a2a_tasks WHERE id = ?",
        (value.id,),
    )
    assert after[:2] == before[:2] and after[2] != before[2]
    assert not await wake_candidate_ids(store, live=horizon, terminal=horizon)
    for expected, state in (
        (TaskState.TASK_STATE_WORKING, TaskState.TASK_STATE_INPUT_REQUIRED),
        (TaskState.TASK_STATE_INPUT_REQUIRED, TaskState.TASK_STATE_WORKING),
    ):
        assert await store.update_status(
            value.id, TaskStatus(state=state), recipient_agent_id=store.scope.recipient_id,
            expected_state=expected,
        )
        candidates = await wake_candidate_ids(store, live=horizon, terminal=horizon)
        assert candidates == ({value.id} if state == TaskState.TASK_STATE_WORKING else set())
    snapshot = await store.get_task_snapshot(value.id, recipient_agent_id=store.scope.recipient_id)
    assert snapshot.lifecycle_revision == 2


@pytest.mark.parametrize("native_a2a_backend_factory", ["sqlite"], indirect=True)
async def test_sqlite_wake_boundary_normalizes_native_space_and_iso_timestamps(store):
    assert store.backend.backend_type == "sqlite"
    value = task()
    await create(store, value)
    now = store.now_utc().replace(microsecond=0)
    stamp = (now - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S")
    await store.backend.execute(
        "UPDATE a2a_tasks SET status = 'completed', lifecycle_updated_at = NULL, updated_at = ? WHERE id = ?",
        (stamp, value.id),
    )
    assert await wake_candidate_ids(store, live=now, terminal=now - timedelta(hours=3)) == {value.id}
    assert not await wake_candidate_ids(store, live=now, terminal=now - timedelta(hours=1))


@pytest.mark.parametrize("native_a2a_backend_factory", ["postgres"], indirect=True)
async def test_postgres_native_database_fence_versions_direct_status_writes(store):
    assert store.backend.backend_type == "postgres"
    value = task()
    await create(store, value)
    horizon = store.now_utc() - timedelta(days=1)
    for revision, state in enumerate(("working", "input-required", "working"), start=1):
        await age_task_revision(store, value.id, horizon - timedelta(days=29))
        await store.backend.execute("UPDATE a2a_tasks SET status = ? WHERE id = ?", (state, value.id))
        snapshot = await store.get_task_snapshot(value.id, recipient_agent_id=store.scope.recipient_id)
        assert snapshot.lifecycle_revision == revision
        candidates = await wake_candidate_ids(store, live=horizon, terminal=horizon)
        assert candidates == ({value.id} if state == "working" else set())
    await store.backend.execute(
        "INSERT INTO a2a_tasks (id, session_id, task_type, status, creator_agent_id, "
        "recipient_agent_id, tenant, history, metadata, artifacts) "
        "SELECT ?, session_id, task_type, 'submitted', creator_agent_id, "
        "recipient_agent_id, tenant, history, metadata, artifacts FROM a2a_tasks WHERE id = ?",
        ("direct-native-insert", value.id),
    )
    assert (await store.backend.fetch_one(
        "SELECT lifecycle_updated_at IS NOT NULL FROM a2a_tasks WHERE id = ?", ("direct-native-insert",)
    ))[0] is True
    assert await wake_candidate_ids(store, live=horizon, terminal=horizon) == {value.id, "direct-native-insert"}


async def test_native_payload_round_trip_preserves_nested_structs_and_message_identity(
    store,
):
    original = task()
    original.history.append(
        new_data_message({"null": None, "array": [None, {}, [], 3.5], "map": {}})
    )
    original.artifacts.append(
        new_text_artifact("result", "payload", artifact_id="artifact-one")
    )
    original.status.message.CopyFrom(new_text_message("queued"))
    original.metadata.update({"nested": {"array": [None, {}, []]}, "empty": {}})
    await create(store, original)
    restored = await get(store)
    assert isinstance(restored, Task)
    assert restored == original
    assert restored.history[0].message_id == original.history[0].message_id
    assert restored.artifacts[0].artifact_id == "artifact-one"


@pytest.mark.parametrize("present", [False, True])
async def test_optional_empty_text_message_and_metadata_presence_survives_storage(
    store, present
):
    original = task()
    if present:
        original.status.message.CopyFrom(new_text_message(""))
        original.metadata.SetInParent()
    await create(store, original)
    restored = await get(store)
    assert restored.status.HasField("message") is present
    assert restored.HasField("metadata") is present
    assert restored == original


@pytest.mark.parametrize(
    "stamp",
    [None, (0, 0), (0, 1), (-62135596800, 999999999), (253402300799, 999999999)],
)
async def test_status_timestamp_preserves_absence_epoch_and_nanoseconds(store, stamp):
    original = task()
    original.status.ClearField("timestamp")
    if stamp is not None:
        original.status.timestamp.CopyFrom(Timestamp(seconds=stamp[0], nanos=stamp[1]))
    await create(store, original)
    for _ in range(2):
        assert (await get(store)).status == original.status


async def test_status_cas_persists_the_supplied_native_timestamp(store):
    await create(store, task())
    working = TaskStatus(
        state=TaskState.TASK_STATE_WORKING,
        timestamp=Timestamp(seconds=5, nanos=123456789),
    )
    assert await store.update_status(
        "task-one",
        working,
        recipient_agent_id="did:kestrel:bob",
        expected_state=TaskState.TASK_STATE_SUBMITTED,
    )
    assert (await get(store)).status == working
    assert not await store.update_status(
        "task-one",
        TaskStatus(state=TaskState.TASK_STATE_FAILED),
        recipient_agent_id="did:kestrel:bob",
        expected_state=TaskState.TASK_STATE_SUBMITTED,
    )
    assert (await get(store)).status == working


@pytest.mark.parametrize(
    "state", [TaskState.TASK_STATE_AUTH_REQUIRED, TaskState.TASK_STATE_INPUT_REQUIRED]
)
async def test_interrupted_states_remain_cancellable(store, state):
    await create(store, task(state))
    canceled = await store.cancel_if_authorized(
        "task-one",
        actor_agent_id="did:kestrel:alice",
        expected_recipient_agent_id="did:kestrel:bob",
        operation_id="cancel-one",
    )
    assert canceled.status.state == TaskState.TASK_STATE_CANCELED
    assert canceled.status.HasField("timestamp")
    assert (await get(store)) == canceled
    assert (
        MessageToDict(canceled.metadata)["cancellation_receipt"]["actor_agent_id"]
        == "did:kestrel:alice"
    )


@pytest.mark.parametrize(
    "state",
    [
        TaskState.TASK_STATE_COMPLETED,
        TaskState.TASK_STATE_FAILED,
        TaskState.TASK_STATE_CANCELED,
        TaskState.TASK_STATE_REJECTED,
    ],
)
async def test_terminal_states_cannot_be_replaced_by_recipient_cas(store, state):
    value = task()
    await create(store, value)
    value.status.state = state
    if state == TaskState.TASK_STATE_CANCELED:
        await store.cancel_if_authorized("task-one", actor_agent_id="did:kestrel:alice")
    else:
        assert await store.save_recipient_terminal_outcome(
            value,
            baseline=await store.get_for_recipient(value.id, store.scope.recipient_id),
            recipient_agent_id="did:kestrel:bob",
            operation_id="outcome-one",
        )
    with pytest.raises(ValueError, match="terminal"):
        await store.update_status(
            "task-one",
            TaskStatus(state=TaskState.TASK_STATE_WORKING),
            recipient_agent_id="did:kestrel:bob",
            expected_state=state,
        )
    with pytest.raises(ValueError, match="terminal"):
        await store.add_artifact(
            "task-one",
            new_text_artifact("late", "late"),
            recipient_agent_id="did:kestrel:bob",
        )
    assert (
        await store.cancel_if_authorized("task-one", actor_agent_id="did:kestrel:alice")
        is None
    )
    assert (await get(store)).status.state == state


async def test_a_sender_cannot_mint_a_cancellation_receipt_in_metadata(store):
    value = task()
    value.metadata.update(
        {"cancellation_receipt": {"actor_agent_id": "forged"}, "ordinary": [None, {}]}
    )
    await create(store, value)
    assert MessageToDict((await get(store)).metadata) == {"ordinary": [None, {}]}
    raw = await store.backend.fetch_one(
        "SELECT metadata FROM a2a_tasks WHERE id = ?", ("task-one",)
    )
    assert json_loads(raw[0]) == {"ordinary": [None, {}]}


async def test_readback_does_not_treat_ordinary_metadata_as_a_cancellation_receipt(
    store,
):
    await create(store, task())
    await store.backend.execute(
        "UPDATE a2a_tasks SET metadata = ? WHERE id = ?",
        (
            json_dumps({"cancellation_receipt": {"actor_agent_id": "forged"}}),
            "task-one",
        ),
    )
    assert MessageToDict((await get(store)).metadata) == {}


async def test_native_reads_and_mutations_require_the_exact_durable_principal(store):
    await create(store, task())
    assert (
        await store.get_for_creator(
            "task-one", "did:kestrel:mallory", recipient_agent_id="did:kestrel:bob"
        )
        is None
    )
    assert (
        await store.get_for_creator(
            "task-one", "did:kestrel:alice", recipient_agent_id="did:kestrel:mallory"
        )
        is None
    )
    assert not await store.update_status(
        "task-one",
        TaskStatus(state=TaskState.TASK_STATE_WORKING),
        recipient_agent_id="did:kestrel:mallory",
        expected_state=TaskState.TASK_STATE_SUBMITTED,
    )
    assert (
        await store.cancel_if_authorized(
            "task-one", actor_agent_id="did:kestrel:mallory"
        )
        is None
    )
    assert (await get(store)).status.state == TaskState.TASK_STATE_SUBMITTED


async def test_duplicate_task_admission_rolls_back_its_context_reservation(store):
    await create(store, task())
    contexts = ContextStore(store.backend)
    await contexts.initialize()
    binding = NativeRecipientBinding(
        "recipient", store.scope.recipient_id, store.scope.tenant, "JSONRPC"
    )
    duplicate = task()
    duplicate.context_id = "other-context"
    with pytest.raises(TransactionError) as failed:
        async with store.backend.transaction():
            await contexts.reserve(
                "other-context", binding.scope, creator_agent_id="did:kestrel:mallory"
            )
            await create(store, duplicate)
    assert isinstance(failed.value.__cause__, TaskAlreadyExistsError)
    assert await store.backend.fetch_all("SELECT * FROM a2a_contexts") == []
    assert (await get(store)).context_id == "context-one"


@pytest.mark.parametrize(
    "state",
    [
        TaskState.TASK_STATE_SUBMITTED,
        TaskState.TASK_STATE_WORKING,
        TaskState.TASK_STATE_INPUT_REQUIRED,
        TaskState.TASK_STATE_AUTH_REQUIRED,
    ],
)
async def test_a_recipient_can_reject_any_live_native_task(store, state):
    value = task(state)
    await create(store, value)
    value.status.state = TaskState.TASK_STATE_REJECTED
    assert await store.save_recipient_terminal_outcome(
        value,
        baseline=await store.get_for_recipient(value.id, store.scope.recipient_id),
        recipient_agent_id="did:kestrel:bob",
        operation_id="rejection-one",
    )
    assert (await get(store)) == value
    assert (
        await store.get_recipient_terminal_operation_id("task-one", "did:kestrel:bob")
        == "rejection-one"
    )
    with pytest.raises(QueryError, match="terminal"):
        await store.backend.execute(
            "UPDATE a2a_tasks SET status = 'working' WHERE id = ?", ("task-one",)
        )
    assert (await get(store)).status.state == TaskState.TASK_STATE_REJECTED


async def test_context_filter_is_native_and_still_recipient_scoped(store):
    await create(store, task())
    assert [
        value.id
        for value in await store.list_tasks(
            recipient_agent_id="did:kestrel:bob", context_id="context-one"
        )
    ] == ["task-one"]
    assert (
        await store.list_tasks(recipient_agent_id="did:kestrel:bob", context_id="other")
        == []
    )
    assert (
        await store.list_tasks(
            recipient_agent_id="did:kestrel:mallory", context_id="context-one"
        )
        == []
    )


@pytest.mark.parametrize("terminal", [False, True])
async def test_a_recipient_outcome_cannot_move_an_existing_task_to_another_context(
    store, terminal
):
    original = task()
    await create(store, original)
    outcome = task(
        TaskState.TASK_STATE_COMPLETED if terminal else TaskState.TASK_STATE_WORKING
    )
    outcome.context_id = "another-context"
    if terminal:
        saved = await store.save_recipient_terminal_outcome(
            outcome,
            baseline=await store.get_for_recipient(
                outcome.id, store.scope.recipient_id
            ),
            recipient_agent_id="did:kestrel:bob",
            operation_id="wrong-context",
        )
    else:
        saved = await store.save_recipient_lifecycle(
            outcome,
            baseline=await store.get_for_recipient(
                outcome.id, store.scope.recipient_id
            ),
            recipient_agent_id="did:kestrel:bob",
            expected_state=TaskState.TASK_STATE_SUBMITTED,
        )
    assert not saved
    assert await get(store) == original


async def test_a_partial_stored_timestamp_fails_instead_of_inventing_read_time(store):
    await create(store, task())
    await store.backend.execute(
        "UPDATE a2a_tasks SET status_timestamp_seconds = 0, status_timestamp_nanos = NULL WHERE id = ?",
        ("task-one",),
    )
    with pytest.raises(ValueError, match="timestamp is incomplete"):
        await get(store)


@pytest.mark.parametrize(
    "stamp", [(-62135596801, 0), (253402300800, 0), (0, -1), (0, 1000000000)]
)
async def test_invalid_native_timestamps_do_not_write_a_task(store, stamp):
    value = task()
    value.status.timestamp.CopyFrom(Timestamp(seconds=stamp[0], nanos=stamp[1]))
    with pytest.raises(ValueError):
        await create(store, value)
    assert await store.backend.fetch_all("SELECT id FROM a2a_tasks") == []


async def test_missing_required_native_message_fields_are_rejected_before_storage(
    store,
):
    value = task()
    value.status.message.SetInParent()
    with pytest.raises(InvalidParamsError):
        await create(store, value)
    assert await store.backend.fetch_all("SELECT id FROM a2a_tasks") == []


@pytest.mark.parametrize("operation", ["status", "lifecycle", "terminal"])
async def test_lifecycle_writes_reject_incomplete_native_messages(store, operation):
    original = task()
    await create(store, original)
    changed = task(
        TaskState.TASK_STATE_REJECTED
        if operation == "terminal"
        else TaskState.TASK_STATE_WORKING
    )
    changed.status.message.SetInParent()
    with pytest.raises(InvalidParamsError):
        if operation == "status":
            await store.update_status(
                "task-one",
                changed.status,
                recipient_agent_id="did:kestrel:bob",
                expected_state=TaskState.TASK_STATE_SUBMITTED,
            )
        elif operation == "lifecycle":
            await store.save_recipient_lifecycle(
                changed,
                baseline=await store.get_for_recipient(
                    changed.id, store.scope.recipient_id
                ),
                recipient_agent_id="did:kestrel:bob",
                expected_state=TaskState.TASK_STATE_SUBMITTED,
            )
        else:
            await store.save_recipient_terminal_outcome(
                changed,
                baseline=await store.get_for_recipient(
                    changed.id, store.scope.recipient_id
                ),
                recipient_agent_id="did:kestrel:bob",
                operation_id="bad-outcome",
            )
    assert (await get(store)) == original


@pytest.mark.parametrize("value", [None, {}, "legacy task"])
async def test_the_store_does_not_accept_private_wire_objects(store, value):
    with pytest.raises(TypeError, match="official A2A protobuf"):
        await create(store, value)
    assert await store.backend.fetch_all("SELECT id FROM a2a_tasks") == []


async def test_an_older_empty_schema_does_not_make_column_order_authoritative(
    native_a2a_backend_factory,
):
    backend = native_a2a_backend_factory()
    await backend.connect()
    try:
        await backend.execute_script("""
            CREATE TABLE a2a_tasks (
                id TEXT PRIMARY KEY, session_id TEXT, user_id TEXT,
                task_type TEXT NOT NULL, status TEXT, message TEXT,
                artifacts TEXT, history TEXT, metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        tasks = TaskStore(backend, scope=NativeRecipientScope("did:kestrel:bob"))
        await tasks.initialize()
        value = task()
        await create(tasks, value)
        assert (await get(tasks)) == value
        await tasks.initialize()
        assert (await get(tasks)) == value
    finally:
        await backend.close()


@pytest.fixture(params=[{"tenant": "other"}, {"recipient_id": "did:kestrel:other"}])
def foreign(store, request):
    return TaskStore(store.backend, scope=replace(store.scope, **request.param))


async def test_every_task_read_is_bound_to_the_selected_recipient_and_tenant(
    store, foreign
):
    await create(store, task())
    assert await foreign._get_unscoped("task-one") is None
    assert (
        await foreign.get_for_creator(
            "task-one", "did:kestrel:alice", recipient_agent_id="did:kestrel:bob"
        )
        is None
    )
    assert await foreign.get_for_principal("task-one", "did:kestrel:alice") is None
    assert await foreign.get_for_recipient("task-one", "did:kestrel:bob") is None
    assert (
        await foreign.get_cancellation_snapshot(
            "task-one", recipient_agent_id="did:kestrel:bob"
        )
        is None
    )
    assert not await foreign.is_task_recipient("task-one", "did:kestrel:bob")
    assert await foreign.get_pending_tasks(recipient_agent_id="did:kestrel:bob") == []
    assert await foreign.list_tasks(recipient_agent_id="did:kestrel:bob") == []
    assert (
        await foreign.list_cognition_wake_candidates(
            recipient_agent_id="did:kestrel:bob",
            live_changed_since=datetime(1970, 1, 1, tzinfo=UTC),
            terminal_changed_since=datetime(1970, 1, 1, tzinfo=UTC),
        )
        == []
    )
    assert await get(store) is not None


async def test_all_lifecycle_writers_refuse_a_foreign_scope_even_for_the_actual_creator(
    store, foreign
):
    original = task()
    await create(store, original)
    working = task(TaskState.TASK_STATE_WORKING)
    assert not await foreign.update_status(
        "task-one",
        working.status,
        recipient_agent_id="did:kestrel:bob",
        expected_state=TaskState.TASK_STATE_SUBMITTED,
    )
    assert not await foreign.save_recipient_lifecycle(
        working,
        baseline=await store.get_for_recipient(working.id, store.scope.recipient_id),
        recipient_agent_id="did:kestrel:bob",
        expected_state=TaskState.TASK_STATE_SUBMITTED,
    )
    completed = task(TaskState.TASK_STATE_COMPLETED)
    assert not await foreign.save_recipient_terminal_outcome(
        completed,
        baseline=await store.get_for_recipient(completed.id, store.scope.recipient_id),
        recipient_agent_id="did:kestrel:bob",
        operation_id="foreign-outcome",
    )
    assert (
        await foreign.cancel_if_authorized(
            "task-one", actor_agent_id="did:kestrel:alice"
        )
        is None
    )
    with pytest.raises(TaskMutationAuthorizationError):
        await foreign.add_artifact(
            "task-one",
            new_text_artifact("foreign", "must not persist"),
            recipient_agent_id="did:kestrel:bob",
        )
    assert (await get(store)) == original
    assert (
        await store.get_recipient_terminal_operation_id("task-one", "did:kestrel:bob")
        is None
    )


async def test_terminal_receipts_and_retention_do_not_cross_the_bound_scope(
    store, foreign
):
    completed = task(TaskState.TASK_STATE_COMPLETED)
    await create(store, completed)
    # A real recipient-owned transition supplies a non-null private receipt.
    second = task()
    second.id = "task-two"
    await create(store, second)
    second.status.state = TaskState.TASK_STATE_COMPLETED
    assert await store.save_recipient_terminal_outcome(
        second,
        baseline=await store.get_for_recipient(second.id, store.scope.recipient_id),
        recipient_agent_id="did:kestrel:bob",
        operation_id="private-terminal-receipt",
    )
    assert (
        await store.get_recipient_terminal_operation_id("task-two", "did:kestrel:bob")
        == "private-terminal-receipt"
    )
    assert (
        await foreign.get_recipient_terminal_operation_id("task-two", "did:kestrel:bob")
        is None
    )
    assert not await foreign.delete("task-one")
    assert await foreign.cleanup_old(older_than_days=-1) == 0
    assert (await get(store)) == completed
    assert await store.cleanup_old(older_than_days=-1) == 2


async def test_cancellation_receipt_is_private_to_its_recipient_and_tenant(
    store, foreign
):
    await create(store, task())
    canceled = await store.cancel_if_authorized(
        "task-one",
        actor_agent_id="did:kestrel:alice",
        operation_id="private-cancel-receipt",
    )
    assert canceled.status.state == TaskState.TASK_STATE_CANCELED
    assert (
        await store.get_cancellation_operation_id("task-one")
        == "private-cancel-receipt"
    )
    assert await foreign.get_cancellation_operation_id("task-one") is None
    assert (
        await foreign.get_cancellation_snapshot(
            "task-one", recipient_agent_id="did:kestrel:bob"
        )
        is None
    )


async def test_creation_cannot_override_the_store_recipient_with_a_caller_argument(
    store,
):
    with pytest.raises(TaskMutationAuthorizationError, match="selected store"):
        await store.create(
            task(),
            creator_agent_id="did:kestrel:alice",
            recipient_agent_id="did:kestrel:other",
        )
    assert await store.backend.fetch_all("SELECT id FROM a2a_tasks") == []


@pytest.mark.parametrize(
    "scope", [None, "did:kestrel:bob", {"recipient_id": "did:kestrel:bob"}]
)
async def test_the_store_requires_a_host_selected_scope(store, scope):
    with pytest.raises(TypeError, match="server-owned native recipient scope"):
        TaskStore(store.backend, scope=scope)


async def test_task_metadata_does_not_select_its_recipient_or_tenant(store, foreign):
    value = task()
    value.metadata.update(
        {"tenant": foreign.scope.tenant, "recipient_id": foreign.scope.recipient_id}
    )
    await create(store, value)
    assert await store.backend.fetch_all(
        "SELECT recipient_agent_id, tenant FROM a2a_tasks"
    ) == [(store.scope.recipient_id, store.scope.tenant)]
    assert (await get(store)) == value
    assert await foreign._get_unscoped("task-one") is None


@pytest.mark.parametrize("cancel", [False, True])
async def test_outcomes_preserve_chunks_and_history_added_after_the_baseline(
    store, cancel
):
    original = task()
    original.artifacts.append(new_text_artifact("body", "first", artifact_id="body"))
    await create(store, original)
    baseline = await get(store)
    outcome = Task.FromString(baseline.SerializeToString())
    outcome.artifacts.append(new_text_artifact("result", "done", artifact_id="result"))
    outcome.metadata.update({"handler": [None, False, {}, []]})
    outcome.status.state = (
        TaskState.TASK_STATE_CANCELED if cancel else TaskState.TASK_STATE_COMPLETED
    )
    await store.add_artifact(
        "task-one",
        new_text_artifact("body", "second", artifact_id="body"),
        recipient_agent_id="did:kestrel:bob",
        append=True,
    )
    latest = await get(store)
    progress = new_text_message(
        "concurrent progress", task_id=latest.id, context_id=latest.context_id
    )
    latest.history.append(progress)
    latest.status.state = TaskState.TASK_STATE_WORKING
    latest.metadata.update({"progress": {"keep": [None, {}]}})
    assert await store.save_recipient_lifecycle(
        latest,
        baseline=baseline,
        recipient_agent_id="did:kestrel:bob",
        expected_state=TaskState.TASK_STATE_SUBMITTED,
    )
    if cancel:
        committed = await store.cancel_if_authorized(
            "task-one",
            actor_agent_id="did:kestrel:bob",
            task_payload=outcome,
            task_baseline=baseline,
            operation_id="cancel-with-payload",
        )
        assert committed.status.message.task_id == committed.id
        assert committed.status.message.context_id == committed.context_id
    else:
        committed = await store.save_recipient_terminal_outcome(
            outcome,
            baseline=baseline,
            recipient_agent_id="did:kestrel:bob",
            operation_id="complete-with-payload",
        )
    assert committed == await get(store)
    assert [item.artifact_id for item in committed.artifacts] == ["body", "result"]
    assert [part.text for part in committed.artifacts[0].parts] == ["first", "second"]
    assert progress in committed.history
    metadata = MessageToDict(committed.metadata)
    assert metadata["progress"] == {"keep": [None, {}]}
    assert metadata["handler"] == [None, False, {}, []]


@pytest.mark.parametrize("cancel", [False, True])
async def test_conflicting_handler_outcome_does_not_write_or_mint_a_receipt(
    store, cancel
):
    from kestrel_sovereign.a2a.outcomes import TaskOutcomeConflictError

    original = task()
    original.artifacts.append(new_text_artifact("body", "original", artifact_id="body"))
    await create(store, original)
    baseline = await get(store)
    await store.add_artifact(
        "task-one",
        new_text_artifact("body", "independent writer", artifact_id="body"),
        recipient_agent_id="did:kestrel:bob",
    )
    before = await get(store)
    outcome = Task.FromString(baseline.SerializeToString())
    outcome.artifacts[0].parts[0].text = "conflicting handler"
    outcome.status.state = (
        TaskState.TASK_STATE_CANCELED if cancel else TaskState.TASK_STATE_COMPLETED
    )
    with pytest.raises(TaskOutcomeConflictError, match="Artifact changed"):
        if cancel:
            await store.cancel_if_authorized(
                "task-one",
                actor_agent_id="did:kestrel:bob",
                task_payload=outcome,
                task_baseline=baseline,
                operation_id="refused-cancellation",
            )
        else:
            await store.save_recipient_terminal_outcome(
                outcome,
                baseline=baseline,
                recipient_agent_id="did:kestrel:bob",
                operation_id="refused-completion",
            )
    assert await get(store) == before
    row = await store.backend.fetch_one(
        "SELECT cancel_operation_id, terminal_operation_id, lifecycle_revision FROM a2a_tasks WHERE id = ?",
        ("task-one",),
    )
    assert row == (None, None, 0)


async def test_the_returned_outcome_contains_only_durable_authority_metadata(store):
    await create(store, task())
    baseline = await get(store)
    outcome = Task.FromString(baseline.SerializeToString())
    outcome.status.state = TaskState.TASK_STATE_COMPLETED
    outcome.metadata.update(
        {"ordinary": [None, {}], "cancellation_receipt": {"actor_agent_id": "forged"}}
    )
    committed = await store.save_recipient_terminal_outcome(
        outcome,
        baseline=baseline,
        recipient_agent_id="did:kestrel:bob",
        operation_id="complete",
    )
    assert MessageToDict(committed.metadata) == {"ordinary": [None, {}]}
    assert committed == await get(store)
