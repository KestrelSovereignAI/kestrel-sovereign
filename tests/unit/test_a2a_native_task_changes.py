"""Original SQLite/PostgreSQL transactions journal subscribed task mutations."""

import pytest
import pytest_asyncio
from a2a.types import Artifact, TaskState, TaskStatus

from kestrel_sovereign.a2a.scope import NativeRecipientScope
from kestrel_sovereign.a2a.stores.unified.base import NativeStoreConflict
from kestrel_sovereign.a2a.stores.unified.task_change_store import TaskChangeStore
from kestrel_sovereign.storage.db.interface import TransactionError
from tests.unit.test_a2a_native_task_manager import (
    manager as manager,  # noqa: PLC0414 - fixture dependency
)
from tests.unit.test_a2a_native_task_manager import request
from tests.utils.native_push import managed_push_service


@pytest_asyncio.fixture(autouse=True)
async def push_runtime(manager, monkeypatch):
    async with managed_push_service(
        manager, monkeypatch, pause_delivery=True
    ):
        yield


async def _subscribed_task(manager, params):
    params.configuration.task_push_notification_config.url = "https://1.1.1.1/callback"
    return await manager.create_task(params, creator_agent_id="did:creator")


async def test_initial_task_journals_only_its_scope_and_reference(manager):
    assert await manager.task_store.backend.table_exists("a2a_task_changes")
    task = await _subscribed_task(manager, request(secret="never-journal-content"))
    rows = await manager.task_store.backend.fetch_all("SELECT * FROM a2a_task_changes")
    assert len(rows) == 1
    assert rows[0][1:] == (manager.did, "tenant", task.id, None)
    assert "never-journal-content" not in repr(rows)
    pending = await manager.task_store.changes.pending(limit=10)
    assert pending[0].task_id == task.id and pending[0].sequence > 0


@pytest.mark.parametrize("admission", ["configuration", "consumer"])
async def test_capture_requires_both_original_admission_facts(manager, admission):
    task = await _subscribed_task(manager, request())
    backend = manager.task_store.backend
    before = await manager.task_store.changes.pending(limit=10)
    if admission == "configuration":
        revoke = "UPDATE a2a_push_configs SET state = 'deleting'"
        restore = "UPDATE a2a_push_configs SET state = 'active'"
    else:
        revoke = "UPDATE durable_signal_consumers SET active = FALSE"
        restore = "UPDATE durable_signal_consumers SET active = TRUE"
    await backend.execute(revoke)
    try:
        await backend.execute(
            "UPDATE a2a_tasks SET status = 'working' WHERE id = ?", (task.id,)
        )
        assert await manager.task_store.changes.pending(limit=10) == before
    finally:
        await backend.execute(restore)


@pytest.mark.parametrize(
    "mutation", ["status", "artifact", "input", "cancel", "direct"]
)
async def test_every_committed_task_change_has_an_original_transaction_reference(
    manager, mutation
):
    task = await _subscribed_task(manager, request())
    changes = manager.task_store.changes
    before = await changes.pending(limit=10)
    if mutation == "status":
        assert await manager.task_store.update_status(
            task.id,
            TaskStatus(state=TaskState.TASK_STATE_WORKING),
            recipient_agent_id=manager.did,
            expected_state=TaskState.TASK_STATE_SUBMITTED,
        )
    elif mutation == "artifact":
        await manager.task_store.add_artifact(
            task.id,
            Artifact(artifact_id="artifact", parts=[{"text": "private artifact"}]),
            recipient_agent_id=manager.did,
        )
    elif mutation == "input":
        continuation = request()
        continuation.message.task_id = task.id
        await manager.continue_task(continuation, creator_agent_id="did:creator")
    elif mutation == "cancel":
        await manager.cancel_task(
            task.id, agent_name="did:creator", recipient_agent_id=manager.did
        )
    else:
        # A second producer that knows only the canonical SQL cannot bypass
        # notification persistence by omitting an in-memory callback.
        await manager.task_store.backend.execute(
            "UPDATE a2a_tasks SET status = 'working' WHERE id = ?", (task.id,)
        )
    after = await changes.pending(limit=10)
    assert len(after) == len(before) + 1
    assert after[-1].task_id == task.id and after[-1].sequence > before[-1].sequence
    assert "private artifact" not in repr(after)


async def test_rollback_retires_neither_the_task_change_nor_its_reference(manager):
    task = await _subscribed_task(manager, request())
    changes = manager.task_store.changes
    before = await changes.pending(limit=10)
    original = ValueError("rollback the original transaction")
    with pytest.raises(TransactionError) as caught:
        async with manager.task_store.write_transaction():
            await manager.task_store.backend.execute(
                "UPDATE a2a_tasks SET status = 'working' WHERE id = ?", (task.id,)
            )
            assert len(await changes.pending(limit=10)) == len(before) + 1
            raise original
    assert caught.value.__cause__ is original
    assert await changes.pending(limit=10) == before
    assert (
        await manager.task_store.get_for_recipient(task.id, manager.did)
    ).status.state == TaskState.TASK_STATE_SUBMITTED


async def test_failed_compare_and_swap_produces_no_task_change(manager):
    task = await _subscribed_task(manager, request())
    before = await manager.task_store.changes.pending(limit=10)
    assert not await manager.task_store.update_status(
        task.id,
        TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
        recipient_agent_id=manager.did,
        expected_state=TaskState.TASK_STATE_WORKING,
    )
    assert await manager.task_store.changes.pending(limit=10) == before


@pytest.mark.parametrize(
    "scope",
    [
        NativeRecipientScope("did:other", "tenant"),
        NativeRecipientScope("did:kestrel:recipient", "other-tenant"),
    ],
)
async def test_foreign_recipient_and_tenant_cannot_read_or_mark_changes(manager, scope):
    task = await _subscribed_task(manager, request())
    change = (await manager.task_store.changes.pending(limit=10))[0]
    foreign = TaskChangeStore(manager.task_store.backend, scope=scope)
    assert await foreign.pending(limit=10) == []
    with pytest.raises(NativeStoreConflict):
        await foreign.mark_forwarded(change, "forged-event")
    with pytest.raises(NativeStoreConflict):
        async with foreign.write_transaction():
            await foreign.lock_task_boundary(task.id, creator_id="did:creator")


async def test_registration_boundary_and_snapshot_keep_original_task_authority(manager):
    task = await _subscribed_task(manager, request())
    changes = manager.task_store.changes
    with pytest.raises(NativeStoreConflict):
        await changes.enqueue_snapshot(task.id, creator_id="did:impostor")
    async with changes.write_transaction():
        boundary = await changes.lock_task_boundary(task.id, creator_id="did:creator")
        snapshot = await changes.enqueue_snapshot(task.id, creator_id="did:creator")
        assert snapshot > boundary > 0
    with pytest.raises(ValueError):
        await changes.mark_forwarded((await changes.pending(limit=10))[0], "")


async def test_task_cleanup_removes_its_metadata_only_journal(manager):
    task = await _subscribed_task(manager, request())
    assert len(await manager.task_store.changes.pending(limit=10)) == 1
    configs = await manager.push_service.worker.outbox.configurations.list(
        task.id, creator_id="did:creator", limit=10
    )
    assert len(configs) == 1
    await manager.push_service.worker.delete(
        task.id, configs[0].config.id, creator_id="did:creator"
    )
    assert await manager.task_store.delete(task.id)
    assert await manager.task_store.changes.pending(limit=10) == []


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
async def test_invalid_journal_batch_never_reads_storage(manager, limit):
    with pytest.raises(ValueError, match="positive"):
        await manager.task_store.changes.pending(limit=limit)
