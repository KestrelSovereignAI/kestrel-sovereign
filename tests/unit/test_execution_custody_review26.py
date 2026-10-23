"""Actual wrapped read control and host factory COMMIT/retirement consumers."""



import asyncio
from types import SimpleNamespace

import pytest

from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError,
    ExecutionCustody,
    ProcessRuntimeExecutionFence,
    bind_execution_runtime,
    execution_commit_outcome,
    is_execution_control_error,
)
from kestrel_sovereign.services.payer_resolver import open_host_db
from kestrel_sovereign.services.process_host_database import _ProcessHostBackend
from kestrel_sovereign.storage.async_database import (
    AsyncDatabase,
    DatabaseInitializationCleanupError,
)
from tests.unit.test_execution_custody_review24 import host_accessor


@pytest.mark.asyncio
async def test_actual_a2a_store_retirement_joins_every_owner_and_preserves_control(tmp_path, monkeypatch):
    from kestrel_sdk.storage.database import QueryError

    from kestrel_sovereign.a2a.scope import NativeRecipientScope
    from kestrel_sovereign.a2a.task_manager import create_task_manager
    from kestrel_sovereign.execution_custody import ExecutionCommitOutcomeError
    manager = await create_task_manager(str(tmp_path / "all-stores.db"), scope=NativeRecipientScope("did:test:recipient"))
    stores = [manager.feedback_store, manager.memory_service, manager.observability_store, manager.session_service, manager.task_store]
    completed = []
    closers = [(store, store.close) for store in stores]
    for index, store in enumerate(stores):
        original = store.close

        async def close_boundary(index=index, original=original):
            await original()
            completed.append(index)
            if index == 0:
                raise ValueError("ordinary first store close failure")
            if index == 1:
                raise QueryError("second store uncertain custody") from ExecutionCommitOutcomeError("unknown")

        monkeypatch.setattr(store, "close", close_boundary)
    with pytest.raises(BaseException) as result:
        await manager.close()
    assert completed == list(range(5)), "a failed store retirement skipped another resource owner"
    assert execution_commit_outcome(result.value) == "unknown", "first ordinary failure eclipsed original control"
    for store, closer in closers:
        monkeypatch.setattr(store, "close", closer)
    await manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", [
    "causation", "submitted", "create_session", "create_event", "create_observation", "create_readback",
    "status_event", "status_observation", "status_notification", "status_memory", "completion_callback",
    "worker", "worker_cancel", "command", "cancel_callback",
])
async def test_actual_a2a_producers_and_projections_propagate_original_control(tmp_path, monkeypatch, stage):
    from kestrel_sovereign._vendor.a2a.helpers.proto_helpers import new_text_message
    from kestrel_sovereign._vendor.a2a.types import AgentSkill, Role, SendMessageRequest, TaskState
    from kestrel_sdk.storage.database import QueryError

    from kestrel_sovereign.a2a.scope import NativeRecipientScope
    from kestrel_sovereign.a2a.task_manager import create_task_manager
    from kestrel_sovereign.execution_custody import ExecutionCommitOutcomeError

    manager = await create_task_manager(str(tmp_path / 'producer.db'), scope=NativeRecipientScope('did:test:recipient'))
    marker = ExecutionCommitOutcomeError("unknown")

    def fail_sync(*args, **kwargs):
        raise QueryError("local wrapped producer control") from marker

    async def fail_async(*args, **kwargs):
        fail_sync()

    params = SendMessageRequest(message=new_text_message('fixture', role=Role.ROLE_USER))
    task = await manager.create_task(params,  creator_agent_id="did:test:creator")
    if stage.startswith("status_") or stage == "completion_callback":
        task = await manager.update_status(task.id, TaskState.TASK_STATE_WORKING, agent_name="did:test:recipient", recipient_agent_id="did:test:recipient")
    targets = {
        "causation": (manager, "_causation_chain_provider", fail_sync),
        "submitted": (manager, "_on_task_submitted", fail_sync),
        "create_session": (manager.session_service, "get_session", fail_async),
        "create_event": (manager.session_service, "append_event", fail_async),
        "create_observation": (manager.observability_store, "log_tool_call", fail_async),
        "create_readback": (manager.task_store, "_get_unscoped", fail_async),
        "status_event": (manager.session_service, "append_event", fail_async),
        "status_observation": (manager.observability_store, "log_agent_response", fail_async),
        "status_notification": (manager, "_notify_status_update", fail_async),
        "status_memory": (manager, "_save_to_memory", fail_async),
        "completion_callback": (manager, "_on_task_complete", fail_sync),
        "cancel_callback": (manager, "_on_task_cancelled", fail_sync),
    }
    if stage in targets:
        owner, attribute, function = targets[stage]
        monkeypatch.setattr(owner, attribute, function)
    try:
        with pytest.raises(BaseException) as result:
            if stage == "causation":
                manager.register_provider(
                    "feature", [AgentSkill(id="fixture", name="Fixture", description="Fixture", tags=["test"])],
                    SimpleNamespace(handle_task=fail_async),
                )
                await manager.execute_skill("feature", "fixture", {}, sync=True)
            elif stage == "submitted" or stage.startswith("create_"):
                fresh_params = SendMessageRequest(message=new_text_message('new fixture', role=Role.ROLE_USER))
                await manager.create_task(fresh_params,  creator_agent_id="did:test:creator")
            elif stage == "cancel_callback":
                await manager.cancel_task(task.id, agent_name="did:test:creator")
            elif stage in {"worker", "worker_cancel"}:
                async def handler_failure(_task, *, arguments=None):
                    if stage == "worker_cancel":
                        raise asyncio.CancelledError("wrapped original custody") from marker
                    fail_sync()
                await manager._execute_async(SimpleNamespace(handle_task=handler_failure), task, "did:test:recipient")
            elif stage == "command":
                manager.register_provider(
                    "feature", [AgentSkill(id="fixture", name="Fixture", description="Fixture", tags=["test"])],
                    SimpleNamespace(handle_task=fail_async), command_prefixes={"!fixture":"fixture"},
                )
                await manager.execute_command("!fixture")
            else:
                await manager.complete_task(task.id, "fixture completion", agent_name="did:test:recipient", recipient_agent_id="did:test:recipient")
        assert execution_commit_outcome(result.value) == "unknown", "A2A producer/projection erased original control"
        assert is_execution_control_error(result.value)
        if stage in {"worker", "worker_cancel"}:
            assert (await manager.task_store._get_unscoped(task.id)).status.state == TaskState.TASK_STATE_SUBMITTED, "original control became a fresh FAILED/CANCELED write"
    finally:
        await manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["save", "payload"])
@pytest.mark.parametrize("control", ["denied", "unknown"])
async def test_actual_a2a_terminal_outcome_never_normalizes_control_into_completion(tmp_path, monkeypatch, stage, control):
    from kestrel_sovereign._vendor.a2a.helpers.proto_helpers import new_text_message
    from kestrel_sovereign._vendor.a2a.types import Role, SendMessageRequest, Task, TaskState
    from kestrel_sdk.storage.database import QueryError

    from kestrel_sovereign.a2a.scope import NativeRecipientScope
    from kestrel_sovereign.a2a.task_manager import create_task_manager
    from kestrel_sovereign.execution_custody import ExecutionCommitOutcomeError

    manager = await create_task_manager(str(tmp_path / 'terminal.db'), scope=NativeRecipientScope('did:test:recipient'))
    task = await manager.create_task(
        SendMessageRequest(message=new_text_message('fixture', role=Role.ROLE_USER)),

        creator_agent_id="did:test:creator",
    )
    baseline = Task.FromString(task.SerializeToString())
    task.status.state = TaskState.TASK_STATE_COMPLETED
    original_save = manager._save_recipient_execution_result
    original_read = manager.task_store._get_unscoped
    marker = ExecutionCommitOutcomeError("unknown") if control == "unknown" else ExecutionAuthorityError("original authority retired")

    async def wrapped_control():
        raise QueryError("local legacy custody wrapper") from marker

    async def save_then_lose_ack(*args, **kwargs):
        assert await original_save(*args, **kwargs)
        if stage == "save":
            await wrapped_control()
        raise ValueError("ordinary lost ACK; matching token permits ordinary reconciliation")

    async def denied_payload(*args, **kwargs):
        await wrapped_control()

    monkeypatch.setattr(manager, "_save_recipient_execution_result", save_then_lose_ack)
    if stage == "payload":
        monkeypatch.setattr(manager.task_store, "get_for_recipient", denied_payload)
    try:
        with pytest.raises(BaseException) as result:
            await manager._persist_execution_outcome(
                task, baseline=baseline, authority_agent_id="did:test:recipient", expected_state=TaskState.TASK_STATE_SUBMITTED,
            )
        assert is_execution_control_error(result.value), "durable reconciliation normalized control into completion"
        assert execution_commit_outcome(result.value) == ("unknown" if control == "unknown" else None)
        assert (await original_read(task.id)).status.state == TaskState.TASK_STATE_COMPLETED
    finally:
        await manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("read", ["one", "all"])
@pytest.mark.parametrize("control", ["denied", "unknown"])
@pytest.mark.parametrize("cleanup", ["cancel", "error"])
async def test_real_transaction_read_control_cause_survives_failed_rollback(tmp_path, monkeypatch, read, control, cleanup):
    owner, db, _ = await host_accessor(tmp_path)
    original_rollback = db.backend._connection.rollback

    async def failed_rollback():
        if cleanup == "cancel":
            raise asyncio.CancelledError("local cleanup cancellation")
        raise ValueError("local cleanup failure")

    monkeypatch.setattr(db.backend._connection, "rollback", failed_rollback)
    try:
        with pytest.raises(BaseException) as caught:
            async with db.transaction():
                if control == "denied":
                    owner._execution_custody.fence.retire()
                else:
                    owner._execution_custody.preserve_commit_uncertainty("unknown")
                if read == "one":
                    await db.fetchone("SELECT COUNT(*) FROM service_providers")
                else:
                    await db.fetchall("SELECT id FROM service_providers")
        assert is_execution_control_error(caught.value), "cleanup overwrote the real QueryError control cause"
        assert execution_commit_outcome(caught.value) == ("unknown" if control == "unknown" else None)
    finally:
        monkeypatch.setattr(db.backend._connection, "rollback", original_rollback)
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("cleanup", ["cancel", "error"])
async def test_actual_host_factory_preserves_interrupted_schema_commit_and_retirement_owner(tmp_path, monkeypatch, cleanup):
    path = tmp_path / "agent_data"
    path.mkdir()

    async def no_schema(_db):
        return None

    seed = await AsyncDatabase.sqlite(str(path / "host.db"), schema_initializer=no_schema)
    await seed.close()
    owner = SimpleNamespace(_execution_custody=ExecutionCustody(ProcessRuntimeExecutionFence("host-factory")))
    commit_entered, close_entered, release_close = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original_connect, original_close = _ProcessHostBackend.connect, AsyncDatabase.close
    databases = []

    async def connect_with_native_ack_boundary(backend):
        await original_connect(backend)
        original_commit = backend._connection.commit

        async def commit_ack():
            await original_commit()
            commit_entered.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(backend._connection, "commit", commit_ack)

    async def close_boundary(database):
        databases.append(database)
        close_entered.set()
        await release_close.wait()
        if cleanup == "error":
            raise ValueError("actual host factory close failed")
        await original_close(database)

    monkeypatch.setattr(_ProcessHostBackend, "connect", connect_with_native_ack_boundary)
    monkeypatch.setattr(AsyncDatabase, "close", close_boundary)
    task = None
    try:
        with bind_execution_runtime(owner):
            task = asyncio.create_task(open_host_db(storage_path=path / "agent.db"))
        await commit_entered.wait()
        task.cancel()
        await close_entered.wait()
        if cleanup == "cancel":
            task.cancel()
        release_close.set()
        result = (await asyncio.gather(task, return_exceptions=True))[0]
        assert execution_commit_outcome(result) == "unknown", "factory discarded the original COMMIT receipt"
        assert is_execution_control_error(result)
        assert len(databases) == 1
        if cleanup == "error":
            assert isinstance(result, DatabaseInitializationCleanupError)
            assert result.retirement_owner is databases[0]
            assert isinstance(result.cleanup_error, ValueError)
            assert databases[0].backend._connection is not None, "unretired native worker lost its owner"
        else:
            assert databases[0].backend._connection is None, "receipt returned before native retirement joined"
        with pytest.raises(ExecutionAuthorityError) as latched:
            owner._execution_custody.require_work()
        assert execution_commit_outcome(latched.value) == "unknown"
    finally:
        release_close.set()
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        for database in databases:
            await original_close(database)
        monkeypatch.setattr(AsyncDatabase, "close", original_close)
    verifier = await AsyncDatabase.sqlite(str(path / "host.db"), schema_initializer=no_schema)
    try:
        assert await verifier.fetchone("SELECT name FROM sqlite_master WHERE name='host_service_keys'") == ("host_service_keys",)
    finally:
        await verifier.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("reconciliation", ["cancel", "error"])
async def test_actual_a2a_cancel_commit_receipt_survives_failed_readback(tmp_path, monkeypatch, reconciliation):
    from kestrel_sovereign._vendor.a2a.helpers.proto_helpers import new_text_message
    from kestrel_sovereign._vendor.a2a.types import Role, SendMessageRequest, TaskState
    from kestrel_sdk.storage.database import QueryError

    from kestrel_sovereign.a2a.scope import NativeRecipientScope
    from kestrel_sovereign.a2a.task_manager import create_task_manager
    from kestrel_sovereign.execution_custody import ExecutionCommitOutcomeError

    manager = await create_task_manager(str(tmp_path / 'task.db'), scope=NativeRecipientScope('did:test:recipient'))
    task = await manager.create_task(
        SendMessageRequest(message=new_text_message('fixture', role=Role.ROLE_USER), metadata={"sender": "did:test:creator"}),

        creator_agent_id="did:test:creator",
    )
    original_cancel = manager.task_store.cancel_if_authorized
    original_read = manager.task_store._get_unscoped

    async def commit_then_lose_wrapped_ack(*args, **kwargs):
        committed = await original_cancel(*args, **kwargs)
        assert committed is not None and committed.status.state == TaskState.TASK_STATE_CANCELED
        try:
            raise ExecutionCommitOutcomeError("unknown")
        except ExecutionCommitOutcomeError as control:
            raise QueryError("local legacy storage wrapper") from control

    async def failed_read(*_args, **_kwargs):
        if reconciliation == "cancel":
            raise asyncio.CancelledError("local reconciliation cancellation")
        raise ValueError("local reconciliation failure")

    monkeypatch.setattr(manager.task_store, "cancel_if_authorized", commit_then_lose_wrapped_ack)
    monkeypatch.setattr(manager.task_store, "get_for_principal", failed_read)
    try:
        with pytest.raises(BaseException) as caught:
            await manager.cancel_task(task.id, agent_name="did:test:creator")
        assert execution_commit_outcome(caught.value) == "unknown", "readback replaced original committed cancellation cause"
        assert is_execution_control_error(caught.value)
        assert (await original_read(task.id)).status.state == TaskState.TASK_STATE_CANCELED
    finally:
        await manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["backend", "factory", "backend_cancel", "factory_cancel", "dual_cancel"])
async def test_actual_database_close_preserves_wrapped_control_from_either_owner(tmp_path, monkeypatch, source):
    from kestrel_sdk.storage.database import QueryError

    from kestrel_sovereign.execution_custody import ExecutionCommitOutcomeError

    database = await AsyncDatabase.sqlite(str(tmp_path / "close.db"))
    original_close = database.backend.close
    control = ExecutionCommitOutcomeError("unknown")

    async def fail_factory():
        if source in {"factory", "dual_cancel"}:
            raise QueryError("local factory wrapper") from control
        if source == "factory_cancel":
            raise asyncio.CancelledError("local factory wrapper") from control
        raise ValueError("local factory failure")

    async def close_backend_then_fail():
        await original_close()
        if source == "backend":
            raise QueryError("local backend wrapper") from control
        if source == "backend_cancel":
            raise asyncio.CancelledError("local backend wrapper") from control
        if source == "factory_cancel":
            return
        if source == "dual_cancel":
            raise asyncio.CancelledError("local backend cancellation")
        raise ValueError("local backend failure")

    monkeypatch.setattr(database, "dispose_cached_sqla_factory", fail_factory)
    monkeypatch.setattr(database.backend, "close", close_backend_then_fail)
    try:
        with pytest.raises(BaseException) as result:
            await database.close()
        assert execution_commit_outcome(result.value) == "unknown"
        assert is_execution_control_error(result.value)
        assert database.backend._connection is None, "primary native close did not join"
    finally:
        await original_close()
