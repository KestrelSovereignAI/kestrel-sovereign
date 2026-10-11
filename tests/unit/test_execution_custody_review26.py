"""Actual wrapped read control and host factory COMMIT/retirement consumers."""

import asyncio
from types import SimpleNamespace

import pytest

from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError, ExecutionCustody, ProcessRuntimeExecutionFence,
    bind_execution_runtime, execution_commit_outcome, is_execution_control_error,
)
from kestrel_sovereign.services.payer_resolver import open_host_db
from kestrel_sovereign.services.process_host_database import _ProcessHostBackend
from kestrel_sovereign.storage.async_database import AsyncDatabase, DatabaseInitializationCleanupError
from tests.unit.test_execution_custody_review24 import host_accessor


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
    from kestrel_sdk.storage.database import QueryError
    from kestrel_sovereign.a2a.task_manager import create_task_manager
    from kestrel_sovereign.a2a.types import TaskSendParams, Message, TextPart, TaskState
    from kestrel_sovereign.execution_custody import ExecutionCommitOutcomeError

    manager = await create_task_manager(str(tmp_path / "task.db"))
    task = await manager.create_task(
        TaskSendParams(message=Message(role="user", parts=[TextPart(text="fixture")]), metadata={"sender": "did:test:creator"}),
        agent_name="did:test:recipient",
        creator_agent_id="did:test:creator",
    )
    original_cancel = manager.task_store.cancel_if_authorized
    original_read = manager.task_store._get_unscoped

    async def commit_then_lose_wrapped_ack(*args, **kwargs):
        committed = await original_cancel(*args, **kwargs)
        assert committed is not None and committed.status.state is TaskState.CANCELED
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
        assert (await original_read(task.id)).status.state is TaskState.CANCELED
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
