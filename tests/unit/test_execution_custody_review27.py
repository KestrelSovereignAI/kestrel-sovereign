"""Real-store regressions for review27 control propagation and retirement."""
import asyncio

import pytest
from a2a.helpers.proto_helpers import new_text_message
from a2a.types import Role, SendMessageRequest, TaskState
from kestrel_sdk.storage.database import QueryError

from kestrel_sovereign.a2a.scope import NativeRecipientScope
from kestrel_sovereign.a2a.task_manager import create_task_manager
from kestrel_sovereign.execution_custody import (
    ExecutionCommitOutcomeError,
    execution_commit_outcome,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("carrier", [QueryError, ValueError])
async def test_actual_shutdown_joins_every_store_without_losing_native_cancel_receipt(tmp_path, monkeypatch, carrier):
    manager = await create_task_manager(str(tmp_path / "shutdown.db"), scope=NativeRecipientScope("did:test:recipient"))
    task = await manager.create_task(
        SendMessageRequest(message=new_text_message("fixture", role=Role.ROLE_USER)),
        creator_agent_id="did:test:creator",
    )
    entered = asyncio.Event()

    async def worker():
        entered.set()
        await asyncio.Event().wait()

    running = manager._track_execution_task(worker(), task.id, "did:test:recipient")
    await entered.wait()
    original_cancel = manager.task_store.cancel_if_authorized

    async def lost_ack(*args, **kwargs):
        committed = await original_cancel(*args, **kwargs)
        assert committed.status.state == TaskState.TASK_STATE_CANCELED
        raise carrier("original shutdown COMMIT acknowledgement lost") from ExecutionCommitOutcomeError("unknown")

    async def read_failed(*args, **kwargs):
        raise OSError("authorized readback unavailable")

    monkeypatch.setattr(manager.task_store, "cancel_if_authorized", lost_ack)
    monkeypatch.setattr(manager.task_store, "get_for_principal", read_failed)
    stores = [manager.feedback_store, manager.memory_service, manager.observability_store, manager.session_service, manager.task_store]
    closed = []
    for index, store in enumerate(stores):
        original = store.close

        async def close(index=index, original=original):
            await original()
            closed.append(index)

        monkeypatch.setattr(store, "close", close)
    failure = None
    try:
        await manager.close()
    except BaseException as error:
        failure = error
    assert running.done(), "shutdown abandoned its own worker"
    assert closed == list(range(5)), "shutdown skipped a native store owner"
    assert failure is not None and execution_commit_outcome(failure) == "unknown", "joined shutdown discarded original native cancellation receipt"

from types import SimpleNamespace

from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError,
    is_execution_control_error,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["status", "list", "respond_read", "respond_write", "artifact", "cancel_direction", "cancel_write"])
@pytest.mark.parametrize("carrier", [QueryError, ValueError])
async def test_public_task_tools_preserve_native_control_receipts(tmp_path, monkeypatch, route, carrier):
    from kestrel_sovereign.features.tasks.feature import TaskFeature

    recipient = "did:test:recipient"
    manager = await create_task_manager(str(tmp_path / "public-tools.db"), scope=NativeRecipientScope("did:test:recipient"))
    task = await manager.create_task(
        SendMessageRequest(message=new_text_message("fixture", role=Role.ROLE_USER)),
        creator_agent_id="did:test:creator",
    )
    feature = TaskFeature(SimpleNamespace(did=recipient, features={}))
    feature.set_task_manager(manager)
    marker = ExecutionCommitOutcomeError("unknown")
    method = {
        "status": "get_task_snapshot", "list": "list_task_snapshots",
        "respond_read": "get_task_for_recipient", "respond_write": "update_status",
        "artifact": "add_artifact", "cancel_direction": "is_task_recipient",
        "cancel_write": "cancel_task",
    }[route]
    original = getattr(manager, method)
    calls = []

    async def native_then_control(*args, **kwargs):
        result = await original(*args, **kwargs)
        assert result is not None
        calls.append(True)
        raise carrier("original native operation receipt") from marker

    monkeypatch.setattr(manager, method, native_then_control)
    try:
        with pytest.raises(BaseException) as caught:
            if route == "status":
                await feature._get_task_status_data(task.id)
            elif route == "list":
                await feature.list_my_tasks()
            elif route.startswith("respond"):
                await feature.respond_to_a2a_task(task.id, "fixture response")
            elif route == "artifact":
                await feature.attach_artifact_to_a2a_task(task.id, "fixture", "native payload")
            else:
                await feature.cancel_task(task.id, "fixture cancel")
        assert execution_commit_outcome(caught.value) == "unknown"
        assert calls == [True], "native operation was retried after an unresolved receipt"
        durable = await manager.task_store._get_unscoped(task.id)
        expected = TaskState.TASK_STATE_COMPLETED if route == "respond_write" else TaskState.TASK_STATE_CANCELED if route == "cancel_write" else TaskState.TASK_STATE_SUBMITTED
        assert durable.status.state == expected
        assert len(durable.artifacts or []) == (route == "artifact")
    finally:
        await manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["skill", "workflow", "background", "background_cancel"])
@pytest.mark.parametrize("control", ["denied", "unknown"])
async def test_registered_feature_and_actual_workflow_never_publish_or_retry_control(tmp_path, monkeypatch, route, control):
    from kestrel_sovereign.features.base import Feature, tool
    from kestrel_sovereign.features.tasks.feature import TaskFeature
    from kestrel_sovereign.storage.async_database import AsyncDatabase

    manager = await create_task_manager(str(tmp_path / "registered.db"), scope=NativeRecipientScope("did:test:recipient"))
    effects = await AsyncDatabase.sqlite(str(tmp_path / "effects.db"))
    await effects.execute("CREATE TABLE effects (id INTEGER PRIMARY KEY)")
    marker = ExecutionCommitOutcomeError("unknown") if control == "unknown" else ExecutionAuthorityError("narrower child authority lost")
    calls, submitted = [], []
    original_create = manager.create_task

    async def observe_create(*args, **kwargs):
        task = await original_create(*args, **kwargs)
        submitted.append(task.id)
        return task

    monkeypatch.setattr(manager, "create_task", observe_create)

    class Effects(Feature):
        async def initialize(self):
            pass

        @property
        def tool_description(self):
            return "actual registered fixture effect"

        @tool(name="effect", description="fixture durable effect")
        async def effect(self):
            calls.append(True)
            await effects.execute("INSERT INTO effects VALUES (?)", (len(calls),))
            if route == "background_cancel":
                raise asyncio.CancelledError("original child cancellation carrier") from marker
            raise QueryError("child custody wrapper after durable effect") from marker

    feature = Effects(SimpleNamespace(did=manager.did, task_manager=manager))
    manager.register_provider(feature.tool_name, feature.get_agent_skills(), feature)
    workflow = TaskFeature(SimpleNamespace(did=manager.did, task_manager=manager))
    workflow.set_task_manager(manager)
    try:
        with pytest.raises(BaseException) as caught:
            if route == "skill":
                await manager.execute_skill(feature.tool_name, "effect", {})
            elif route == "workflow":
                await workflow.run_workflow([{"feature": feature.tool_name, "skill": "effect", "args": {}, "max_retries": 2, "retry_delay_ms": 0}])
            else:
                await manager.execute_skill(feature.tool_name, "effect", {}, sync=False)
                await asyncio.gather(*tuple(manager._execution_tasks), return_exceptions=True)
                await asyncio.sleep(0)
                assert not manager._execution_tasks
                await manager.drain_execution_tasks()
        assert is_execution_control_error(caught.value)
        assert calls == [True], "workflow retried a narrower child's unresolved effect"
        assert await effects.fetchval("SELECT COUNT(*) FROM effects") == 1
        assert len(submitted) == 1
        durable = await manager.task_store._get_unscoped(submitted[0])
        assert durable.status.state == TaskState.TASK_STATE_SUBMITTED, "registered handler published control as a new FAILED result"
    finally:
        try:
            await manager.close()
        except BaseException as retirement:
            assert is_execution_control_error(retirement)
        await effects.close()


@pytest.mark.asyncio
async def test_lifecycle_join_keeps_owned_cancellation_original_control():
    from kestrel_sovereign.kestrel_agent import await_lifecycle_task_completion
    marker = ExecutionCommitOutcomeError("unknown")
    async def retire():
        await asyncio.sleep(0)
        raise asyncio.CancelledError("actual owned child retirement") from marker
    task = asyncio.create_task(retire())
    cancelled, failure = await await_lifecycle_task_completion(task)
    assert task.done() and not cancelled
    assert failure is not None and execution_commit_outcome(failure) == "unknown"


def test_explicit_lifecycle_groups_keep_strongest_control_but_ignore_context():
    unknown = ExecutionCommitOutcomeError("unknown")
    committed = ExecutionCommitOutcomeError("committed")
    wrapped = asyncio.CancelledError("owned child cancellation")
    wrapped.__cause__ = unknown
    group = BaseExceptionGroup("actual lifecycle aggregate", [wrapped, committed, ValueError("cleanup")])
    assert is_execution_control_error(group)
    assert execution_commit_outcome(group) == "unknown"
    ordinary = ValueError("only incidental handling context")
    ordinary.__context__ = group
    assert not is_execution_control_error(ordinary)
    assert execution_commit_outcome(ordinary) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["load", "persist", "policy"])
async def test_actual_feature_config_does_not_treat_control_as_missing_or_best_effort(tmp_path, monkeypatch, stage):
    from kestrel_sovereign.features.base import Feature
    from kestrel_sovereign.storage.async_database import AsyncDatabase
    from kestrel_sovereign.storage.async_storage import AsyncStorage

    db = await AsyncDatabase.sqlite(str(tmp_path / "config.db"))
    storage = AsyncStorage(backend=db.backend)
    storage.db = db
    marker = ExecutionCommitOutcomeError("unknown")

    class Config(Feature):
        async def initialize(self):
            pass
        @property
        def tool_description(self):
            return "native config fixture"
    feature = Config(SimpleNamespace(storage=storage))
    before = await db.fetchval("SELECT COUNT(*) FROM graph_nodes")

    if stage == "policy":
        def fail_policy():
            raise QueryError("wrapped policy control") from marker
        monkeypatch.setattr(storage, "allows_persistent_writes", fail_policy, raising=False)
    else:
        name = "get_node" if stage == "load" else "add_node"
        original = getattr(storage, name)
        async def fail_after_native(*args, **kwargs):
            await original(*args, **kwargs)
            raise QueryError("wrapped native config control") from marker
        monkeypatch.setattr(storage, name, fail_after_native)
    try:
        with pytest.raises(BaseException) as caught:
            if stage == "load":
                await feature.load_persisted_config()
            else:
                await feature.persist_config({"fixture": "value"})
        assert execution_commit_outcome(caught.value) == "unknown"
        assert await db.fetchval("SELECT COUNT(*) FROM graph_nodes") == before + (stage == "persist")
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["task_manager", "host", "storage", "feature"])
@pytest.mark.parametrize("carrier", ["error", "cancel", "group"])
async def test_actual_agent_shutdown_retains_control_through_prefix_and_native_tail(tmp_path, monkeypatch, source, carrier):
    from kestrel_sovereign.kestrel_agent import KestrelAgent
    from kestrel_sovereign.storage.async_database import AsyncDatabase
    from kestrel_sovereign.storage.async_storage import AsyncStorage

    async def nothing():
        pass

    agent = KestrelAgent(did="did:test:shutdown-control", storage_path=str(tmp_path / "agent.db"), llm_service=SimpleNamespace(close=nothing))
    db = await AsyncDatabase.sqlite(str(tmp_path / "tail.db"))
    storage = AsyncStorage(backend=db.backend)
    storage.db = db
    agent.storage = agent._raw_storage = storage
    agent.features = {}
    agent.task_manager = await create_task_manager(str(tmp_path / "manager.db"), scope=NativeRecipientScope(agent.did))
    completed = []
    marker = ExecutionCommitOutcomeError("unknown")

    def fail():
        wrapper = asyncio.CancelledError("native retirement cancellation carrier") if carrier == "cancel" else QueryError("native retirement error carrier")
        if carrier == "group":
            cancellation = asyncio.CancelledError("owned retirement with unknown receipt")
            cancellation.__cause__ = marker
            raise BaseExceptionGroup("owned retirement failures", [cancellation, ValueError("ordinary cleanup failure")])
        raise wrapper from marker

    original_storage_close = storage.close
    async def storage_close():
        await original_storage_close()
        completed.append("storage")
        if source == "storage":
            fail()
    monkeypatch.setattr(storage, "close", storage_close)

    original_manager_close = agent.task_manager.close
    async def manager_close():
        await original_manager_close()
        completed.append("manager")
        if source == "task_manager":
            fail()
    monkeypatch.setattr(agent.task_manager, "close", manager_close)

    if source == "host":
        host = await AsyncDatabase.sqlite(str(tmp_path / "host.db"))
        original_host_close = host.close
        async def host_close():
            await original_host_close()
            completed.append("host")
            fail()
        monkeypatch.setattr(host, "close", host_close)
        agent._resolved_local_host_db = host
    if source == "feature":
        async def feature_close():
            completed.append("feature")
            fail()
        agent.features = {"Fixture": SimpleNamespace(shutdown=feature_close)}

    failure = None
    try:
        await asyncio.wait_for(agent.shutdown(), 30)
    except BaseException as error:
        failure = error
    assert "manager" in completed and "storage" in completed, "control interrupted retirement of another real owner"
    assert failure is not None and execution_commit_outcome(failure) == "unknown", "actual agent shutdown discarded original control"
