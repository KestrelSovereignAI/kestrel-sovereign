"""Actual preference and late accounting owners retain native control (#3569)."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from tests.utils.original_failure import OriginalFailure
from kestrel_sdk.storage.database import QueryError

from kestrel_sovereign._async_ownership import await_owned_task
from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError,
    ExecutionCommitOutcomeError,
    _explicit_control_chain,
    is_execution_control_error,
)
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.llm import modality_recording
from kestrel_sovereign.llm.invocation_context import LLMInvocationContext
from kestrel_sovereign.llm.modality_recording import ModalityCall
from kestrel_sovereign.llm.service import LLMService
from kestrel_sovereign.storage.async_database import AsyncDatabase


def assert_recorded_preference_rows(rows, route):
    """The original single write now includes atomic provenance (#2652)."""
    mandate = {"model": "model", "vendor": "local", "route": "unit"}
    expected = {
        "model": {
            KestrelAgent.MODEL_PREFERENCE_KEY: mandate,
            KestrelAgent.MODEL_PREFERENCE_PROVENANCE_KEY: {
                "mandate": mandate, "provenance": None,
            },
        },
        "embedding": {KestrelAgent.EMBEDDING_ROUTE_KEY: None},
        "overrides": {KestrelAgent.EMBEDDING_MODEL_OVERRIDES_KEY: {}},
    }[route]
    assert len(rows) == len(expected)
    assert {row[0]: json.loads(row[1]) for row in rows} == expected


def carrier(kind):
    error = (
        asyncio.CancelledError("original cancellation")
        if kind.startswith("cancel")
        else QueryError("original native receipt")
    )
    error.__cause__ = (
        ExecutionCommitOutcomeError("unknown")
        if "unknown" in kind
        else ExecutionAuthorityError("original authority")
    )
    return ExceptionGroup("original group", [error]) if kind == "group" else error


def service_owner(service_class=LLMService):
    service = service_class.__new__(service_class)
    service._preference_persistence_tasks = set()
    service._preference_persistence_callback = None
    service._embedding_route_persistence_callback = None
    service._route_embedding_model_persistence_callback = None
    service.providers = []
    service._remote_route_condition = asyncio.Condition()
    service._remote_route_closed = False
    service._remote_lease = None
    service._usage_db = None
    service._usage_db_owned = False
    service._route_embedding_model_overrides = {}
    return service


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind", ["error", "unknown", "cancel", "cancel_unknown", "group", "ordinary"]
)
@pytest.mark.parametrize("route", ["model", "embedding", "overrides"])
async def test_installed_agent_preference_callback_retains_completed_native_error(
    tmp_path, monkeypatch, route, kind
):
    db = await AsyncDatabase.sqlite(str(tmp_path / "preference.db"))
    await db.execute(
        "CREATE TABLE IF NOT EXISTS agent_metadata (agent_id TEXT, key TEXT, value TEXT, updated_at TEXT, PRIMARY KEY (agent_id, key))"
    )
    service = service_owner()
    agent = KestrelAgent.__new__(KestrelAgent)
    agent.did = "did:test:preference"
    agent.llm_service = service
    agent._raw_storage = SimpleNamespace(db=db)
    agent.privacy_agent = SimpleNamespace(
        privacy_config=SimpleNamespace(allows_cloud_llm=lambda: False)
    )
    original = db.execute
    failure = (
        OSError("ordinary persistence problem") if kind == "ordinary" else carrier(kind)
    )
    writes = []

    async def native_then_receipt(*args, **kwargs):
        result = await original(*args, **kwargs)
        writes.append(True)
        raise failure

    monkeypatch.setattr(db, "execute", native_then_receipt)
    completed = asyncio.Event()
    if route == "model":
        service.set_preference_persistence_callback(agent._persist_model_preference)
        task = service._schedule_preference_persistence("model", "local", "unit")
    elif route == "embedding":
        service.set_embedding_route_persistence_callback(agent._persist_embedding_route)
        task = service._schedule_embedding_route_persistence(None)
    else:
        service.set_route_embedding_model_persistence_callback(
            agent._persist_route_embedding_models
        )
        task = service._schedule_route_embedding_model_persistence()
    task.add_done_callback(lambda done: completed.set())
    try:
        await asyncio.wait_for(completed.wait(), 2)
        assert_recorded_preference_rows(
            await db.fetchall("SELECT key, value FROM agent_metadata"), route,
        )
        assert writes == [True]
        if kind == "ordinary":
            await service.drain_preference_persistence()
        else:
            with pytest.raises(BaseException) as caught:
                await service.drain_preference_persistence()
            assert is_execution_control_error(caught.value)
        assert not service._preference_persistence_tasks
    finally:
        await await_owned_task(task)
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind", ["error", "unknown", "cancel", "cancel_unknown", "group", "ordinary"]
)
@pytest.mark.parametrize("boundary", ["receipt", "channel", "terminal_channel"])
async def test_actual_dispatch_admission_cannot_normalize_native_control(
    tmp_path, monkeypatch, kind, boundary
):
    from tests.unit.test_execution_custody_review29 import native_owner, retire
    from kestrel_sovereign.signals import (
        SignalDispatcher,
        SignalLogStore,
        SourceRegistry,
        OrderedLockManager,
    )
    from kestrel_sovereign.features.channels.feature import (
        ChannelFeature,
        InboundAdmissionDisposition,
    )
    from kestrel_sovereign.features.channels.models import (
        ChannelMessage,
        MessageDirection,
    )
    from kestrel_sovereign.signals.sources.channels import (
        build_signal_for_channel_message,
    )

    agent, db = await native_owner(tmp_path)
    logs = SignalLogStore(db.backend)
    await logs.initialize()
    registry = SourceRegistry()
    agent.signal_registry = registry
    dispatcher = SignalDispatcher(
        agent=agent, registry=registry, lock_manager=OrderedLockManager(), store=logs
    )
    agent.dispatcher = dispatcher
    feature = ChannelFeature(agent)
    agent.features = {"ChannelFeature": feature}
    failure = (
        OSError("ordinary admission problem") if kind == "ordinary" else carrier(kind)
    )
    entered = asyncio.Event()

    async def native_run(*args, **kwargs):
        await db.execute("INSERT INTO effects VALUES (1)")
        entered.set()
        raise failure

    monkeypatch.setattr(dispatcher, "_run", native_run)
    message = ChannelMessage(
        channel_type="telegram",
        direction=MessageDirection.INBOUND,
        sender="555",
        recipient="42",
        content="native admission",
        id="telegram:v2:bot:42:update:101",
        metadata={"dedupe_key": "telegram:v2:bot:42:update:101"},
    )
    message._kestrel_cursor_owned_protocol = True
    try:
        await feature.initialize()

        async def invoke():
            if boundary == "receipt":
                handle = await dispatcher.enqueue_signal(
                    build_signal_for_channel_message(message, target_agent=agent.did)
                )
                return await handle.wait_for_durable_admission()
            if boundary == "channel":
                return await feature.handle_inbound(message)
            return await feature.handle_terminal_inbound(
                message, disposition="unsupported_update"
            )

        if kind == "ordinary":
            result = await invoke()
            if boundary != "receipt":
                assert result.disposition is InboundAdmissionDisposition.RETRYABLE
        else:
            with pytest.raises(BaseException) as caught:
                await invoke()
            assert caught.value is failure
        assert entered.is_set()
        assert await db.fetchval("SELECT COUNT(*) FROM effects") == 1
    finally:
        await retire(agent)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind", ["error", "unknown", "cancel", "cancel_unknown", "group"]
)
@pytest.mark.parametrize("boundary", ["drain", "close"])
async def test_late_modality_native_effect_retains_control_after_task_discard(
    tmp_path, monkeypatch, kind, boundary
):
    db = await AsyncDatabase.sqlite(str(tmp_path / "modality.db"))
    await db.execute("CREATE TABLE effects (id INTEGER PRIMARY KEY)")
    release = asyncio.Event()
    entered = asyncio.Event()
    completed = asyncio.Event()
    failure = carrier(kind)
    retired = []

    class NativeService(LLMService):
        async def _log_llm_call(self, **kwargs):
            await db.execute("INSERT INTO effects VALUES (1)")
            entered.set()
            await release.wait()
            raise failure

    service = service_owner(NativeService)

    async def close_resource():
        retired.append("client")

    service.providers = [
        {"name": "native owner", "client": SimpleNamespace(close=close_resource)}
    ]
    monkeypatch.setattr(modality_recording, "USAGE_RECORD_TIMEOUT", 0.01)
    caller = asyncio.create_task(
        service.record_modality_call(
            ModalityCall(
                modality="embedding",
                provider="unit",
                model="unit",
                duration_ms=1,
                success=True,
                context=LLMInvocationContext(),
            )
        )
    )
    tasks = ()
    try:
        await asyncio.wait_for(entered.wait(), 2)
        tasks = tuple(service._pending_modality_records())
        assert len(tasks) == 1
        tasks[0].add_done_callback(lambda done: completed.set())
        await asyncio.wait_for(caller, 2)
        assert not tasks[0].done(), "fixture did not cross the late-record deadline"
        release.set()
        await asyncio.wait_for(completed.wait(), 2)
        assert await db.fetchval("SELECT COUNT(*) FROM effects") == 1
        with pytest.raises(BaseException) as caught:
            await (
                service.close()
                if boundary == "close"
                else service.drain_modality_records()
            )
        assert is_execution_control_error(caught.value)
        if boundary == "close":
            assert retired == ["client"], "control skipped a later resource owner"
    finally:
        release.set()
        await await_owned_task(caller)
        for task in tasks:
            await await_owned_task(task)
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind", ["error", "unknown", "cancel", "cancel_unknown", "group"]
)
@pytest.mark.parametrize("stage", ["adapter", "client"])
async def test_close_retains_timed_out_native_resource_and_joins_original_receipt(
    tmp_path, monkeypatch, kind, stage
):
    from kestrel_sovereign.llm import service as service_module

    db = await AsyncDatabase.sqlite(str(tmp_path / "resource.db"))
    await db.execute("CREATE TABLE effects (id INTEGER PRIMARY KEY)")
    failure = carrier(kind)
    source = OriginalFailure.capture(failure)
    retired = []
    release_native_close = asyncio.Event()

    class Resource:
        async def close(self):
            await db.execute("INSERT INTO effects VALUES (1)")
            native_close_entered.set()
            await release_native_close.wait()
            raise failure

        aclose = close

    async def later():
        retired.append("later")

    service = service_owner()
    resource = Resource()
    service.providers = [{stage: resource}, {"client": SimpleNamespace(close=later)}]
    native_close_entered = asyncio.Event()
    original_wait = asyncio.wait

    async def wait_after_native_setup(awaitable, *, timeout):
        # This regression concerns the resource's terminal receipt after
        # a real write, not how quickly a SQLite worker can finish that write.
        # Keep the actual timeout owner, but arm its short test deadline only
        # after the intended resource boundary has positively been entered.
        if timeout == 0.01 and any(
            task.get_name().startswith(f"llm-close:('{stage}',") for task in awaitable
        ):
            await asyncio.wait_for(native_close_entered.wait(), timeout=5)
        return await original_wait(awaitable, timeout=timeout)

    monkeypatch.setattr(service_module, "CLIENT_CLOSE_TIMEOUT", 0.01)
    monkeypatch.setattr(asyncio, "wait", wait_after_native_setup)
    try:
        with pytest.raises(RuntimeError, match="retained cleanup owners"):
            await service.close()
        assert native_close_entered.is_set()
        assert service.has_pending_close_owners()
        failed_owner = service._llm_close_tasks[(stage, id(resource))]
        assert retired == ["later"]
        assert service._remote_route_closed is True
        assert await db.fetchval("SELECT COUNT(*) FROM effects") == 1
        release_native_close.set()
        with pytest.raises(BaseException) as caught:
            await service.wait_for_close_completion()
        source.assert_retained(caught.value, resource=resource, task=failed_owner)
        # A native close raising a control receipt has not proved transport
        # retirement. Preserve its exact owner even after its task settles.
        assert service.has_pending_close_owners()
        assert service._llm_close_tasks[(stage, id(resource))] is failed_owner
        assert failed_owner.done()
        with pytest.raises(BaseException) as caught:
            await service.close()
        source.assert_retained(caught.value, resource=resource, task=failed_owner)
        assert retired == ["later"]
        assert service._llm_close_tasks[(stage, id(resource))] is failed_owner
        assert await db.fetchval("SELECT COUNT(*) FROM effects") == 1
    finally:
        release_native_close.set()
        for task in tuple(vars(service).get("_llm_close_tasks", {}).values()):
            if task is not None:
                await await_owned_task(task)
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("owner", ["preference", "modality"])
async def test_drain_cancellation_joins_all_native_writers_before_strongest_receipt(
    tmp_path, owner, monkeypatch
):
    db = await AsyncDatabase.sqlite(str(tmp_path / "siblings.db"))
    await db.execute("CREATE TABLE effects (id INTEGER PRIMARY KEY)")
    service = service_owner()
    from kestrel_sovereign.llm import service as service_module

    module = service_module if owner == "preference" else modality_recording
    actual_wait = module.await_owned_task
    withdrawals = []

    async def observed_wait(task, *args, **kwargs):
        outcome = await actual_wait(task, *args, **kwargs)
        if outcome.cancellation is not None:
            withdrawals.append(outcome.cancellation)
        return outcome

    monkeypatch.setattr(module, "await_owned_task", observed_wait)
    started, release, complete = asyncio.Event(), asyncio.Event(), asyncio.Event()
    failure = carrier("unknown")

    async def first():
        await db.execute("INSERT INTO effects VALUES (1)")
        raise failure

    async def second():
        await db.execute("INSERT INTO effects VALUES (2)")
        started.set()
        await release.wait()

    tasks = (asyncio.create_task(first()), asyncio.create_task(second()))
    pending = (
        service._preference_persistence_tasks
        if owner == "preference"
        else service._pending_modality_records()
    )
    done = (
        service._handle_preference_persistence_done
        if owner == "preference"
        else service._modality_record_done
    )
    for task in tasks:
        pending.add(task)
        task.add_done_callback(done)
    tasks[0].add_done_callback(lambda task: complete.set())
    drainer = None
    entered_drain = asyncio.Event()

    async def drain():
        entered_drain.set()
        await (
            service.drain_preference_persistence()
            if owner == "preference"
            else service.drain_modality_records()
        )

    try:
        await asyncio.wait_for(started.wait(), 5)
        await asyncio.wait_for(complete.wait(), 5)
        drainer = asyncio.create_task(drain())
        await entered_drain.wait()
        drainer.cancel()
        assert not tasks[1].done()
        release.set()
        result = await await_owned_task(drainer)
        chain = tuple(_explicit_control_chain(result.error))
        assert failure in chain and withdrawals and withdrawals[0] in chain
        latched = getattr(service, "_preference_persistence_control" if owner == "preference" else "_modality_record_control")
        assert latched is failure
        assert withdrawals[0] not in tuple(_explicit_control_chain(latched))
        for _ in range(3):
            with pytest.raises(BaseException) as later:
                await (service.drain_preference_persistence() if owner == "preference" else service.drain_modality_records())
            assert later.value is failure
            assert withdrawals[0] not in tuple(_explicit_control_chain(later.value))
        assert all(task.done() for task in tasks)
        assert not pending
        assert await db.fetchval("SELECT COUNT(*) FROM effects") == 2
    finally:
        release.set()
        if drainer is not None:
            await await_owned_task(drainer)
        for task in tasks:
            await await_owned_task(task)
        await db.close()
