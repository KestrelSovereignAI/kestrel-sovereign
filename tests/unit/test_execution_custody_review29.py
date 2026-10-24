"""Native effects cannot become retryable signal non-delivery (#3569)."""

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from kestrel_sdk.signals import SignalHandle, SignalMode, SignalResult, Status
from kestrel_sdk.storage.database import QueryError

from kestrel_sovereign._async_ownership import await_owned_task
from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError,
    ExecutionCommitOutcomeError,
    is_execution_control_error,
)
from kestrel_sovereign.features.base import Feature
from kestrel_sovereign.features.peers.feature import PeersFeature
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.signals import (
    OrderedLockManager,
    SignalDispatcher,
    SignalLogStore,
    SourceRegistry,
    delivery,
)
from kestrel_sovereign.signals.sources.a2a_question_answered import (
    build_a2a_question_answered_registration,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.async_storage import AsyncStorage


def control_carrier(kind):
    marker = (
        ExecutionCommitOutcomeError("unknown")
        if kind in {"unknown", "cancel_unknown"}
        else ExecutionAuthorityError("narrower child lost authority")
    )
    wrapper = (
        asyncio.CancelledError("native child cancellation")
        if kind in {"cancel", "cancel_unknown"}
        else QueryError("native child receipt")
    )
    wrapper.__cause__ = marker
    if kind == "group":
        return ExceptionGroup("native child group", [wrapper])
    return wrapper


async def native_owner(tmp_path, *, agent_class=KestrelAgent):
    async def nothing():
        pass

    agent = agent_class(
        did="did:test:terminal-delivery",
        storage_path=str(tmp_path / "agent.db"),
        llm_service=SimpleNamespace(close=nothing),
    )
    db = await AsyncDatabase.sqlite(str(tmp_path / "native.db"))
    await db.execute("CREATE TABLE effects (id INTEGER PRIMARY KEY)")
    storage = AsyncStorage(backend=db.backend)
    storage.db = db
    agent.storage = agent._raw_storage = storage
    agent._runtime_publication_ready.set()
    return agent, db


async def retire(agent):
    try:
        await agent.shutdown()
    except BaseException as error:
        assert is_execution_control_error(error)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind", ["error", "cancel", "unknown", "cancel_unknown", "group"]
)
@pytest.mark.parametrize("route", ["inline", "supervisor", "detached", "enqueue"])
async def test_terminal_delivery_retains_control_before_retry_normalization(
    tmp_path, kind, route
):
    agent, db = await native_owner(tmp_path)

    class Native(Feature):
        async def initialize(self):
            pass

        @property
        def tool_description(self):
            return "native delivery ownership"

    feature = Native(agent)
    agent.features = {"Native": feature}
    failure = control_carrier(kind)
    settlements = []
    completed = asyncio.Event()

    async def effect():
        await db.execute("INSERT INTO effects VALUES (1)")
        raise failure

    async def undelivered(_outcome):
        settlements.append("retry")

    try:
        if route == "enqueue":
            task = delivery.harvest_detached_delivery(
                feature._track_owned_background_task,
                effect,
                label="native enqueue",
                task_name="native-enqueue",
            )
        else:
            task = agent._track_background_task(effect(), name="native-effect")
            task.add_done_callback(lambda _done: completed.set())
            await completed.wait()
            handle = SignalHandle(signal_id="native-effect", task=task)
            if route == "inline":
                with pytest.raises(BaseException) as caught:
                    await delivery.await_terminal_delivery(handle, label="native inline")
                assert is_execution_control_error(caught.value)
                assert settlements == []
                return
            if route == "supervisor":
                task = delivery.supervise_terminal_delivery(
                    feature,
                    handle,
                    label="native supervisor",
                    task_name="native-supervisor",
                    on_undelivered=undelivered,
                )
            else:

                async def enqueue():
                    return handle

                task = delivery.harvest_detached_delivery(
                    feature._track_owned_background_task,
                    enqueue,
                    label="native detached",
                    task_name="native-detached",
                )
        outcome = await await_owned_task(task)
        assert outcome.error is not None and is_execution_control_error(
            outcome.error
        ), "native effect became ordinary non-delivery"
        assert await db.fetchval("SELECT COUNT(*) FROM effects") == 1
        assert settlements == [], "control failure authorized a retry checkpoint"
    finally:
        try:
            assert await db.fetchval("SELECT COUNT(*) FROM effects") == 1
        finally:
            await retire(agent)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["error", "cancel", "unknown", "cancel_unknown", "group"])
@pytest.mark.parametrize("inline", [True, False])
async def test_actual_peer_delivery_does_not_restore_native_effect_as_waiting(tmp_path, monkeypatch, kind, inline):
    from kestrel_sovereign.a2a.question_supervisor import NativeQuestionSupervisor
    from kestrel_sovereign.a2a.scope import NativeRecipientScope
    from kestrel_sovereign.signals import DurableConsumerRegistration
    from kestrel_sovereign.signals.sources.a2a_question_answered import DURABLE_COGNITION_CONSUMER_ID
    from tests.unit.test_a2a_native_question_store import reserve, response
    from tests.utils.native_a2a import manager_for

    failure = control_carrier(kind)

    class NativeAgent(KestrelAgent):
        async def process_input(self, prompt):
            await self.storage.db.execute("INSERT INTO effects VALUES (1)")
            raise failure

    agent, db = await native_owner(tmp_path, agent_class=NativeAgent)
    manager = manager_for(db.backend, NativeRecipientScope(agent.did), owns_backend=False)
    await manager.initialize()
    agent.task_manager = manager
    store = manager.question_store
    clock = SimpleNamespace(now=datetime.now(timezone.utc))
    p = SimpleNamespace(outbound=manager.outbound_store, store=store, clock=clock)
    question, dispatch, request = await reserve(p)
    submitting = await manager.outbound_store.begin_submission(dispatch, operation_id="native-send")
    await store.acknowledge(submitting, response(question, request), operation_id="native-answer")
    logs = SignalLogStore(db.backend)
    await logs.initialize()
    registry = SourceRegistry()
    registry.register(build_a2a_question_answered_registration())
    dispatcher = SignalDispatcher(agent=agent, registry=registry, lock_manager=OrderedLockManager(), store=logs)
    agent.dispatcher = dispatcher
    await dispatcher.initialize_durable_delivery()
    await dispatcher.register_durable_consumer(DurableConsumerRegistration(
        consumer_id=DURABLE_COGNITION_CONSUMER_ID, source="a2a.question_answered", agent_id=agent.did, max_attempts=0,
    ))
    handles = []
    enqueue = dispatcher.enqueue_durable_cognition

    async def remember(*args, **kwargs):
        handle = await enqueue(*args, **kwargs)
        handles.append(handle)
        return handle

    monkeypatch.setattr(dispatcher, "enqueue_durable_cognition", remember)
    feature = PeersFeature(agent)
    agent.features = {"Peers": feature}
    supervisor = NativeQuestionSupervisor(store=store, open_dispatch=None, dispatcher=dispatcher, interval=1, is_current=lambda: True)
    try:
        if inline:
            await supervisor.handoff_wakes()
        else:
            owner = feature._track_owned_runtime_task(supervisor.handoff_wakes(), name="native-question-handoff")
            outcome = await await_owned_task(owner)
            if outcome.error is not None:
                raise outcome.error
        assert len(handles) == 1
        terminal = await await_owned_task(handles[0].task)
        assert terminal.error is not None and is_execution_control_error(terminal.error)
        assert await db.fetchval("SELECT COUNT(*) FROM effects") == 1
        assert (await store.get(question.question_id)).status == "RESOLVED"
        assert not any(task.get_name().startswith("a2a_question_answered_retry:") for task in agent._background_tasks)
    finally:
        await retire(agent)



@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind", ["error", "cancel", "unknown", "cancel_unknown", "group"]
)
@pytest.mark.parametrize("delivered", [True, False])
@pytest.mark.parametrize("settle_noise", [False, True])
async def test_supervisor_callback_control_cannot_reenter_retry_cleanup(
    tmp_path, kind, delivered, settle_noise
):
    agent, db = await native_owner(tmp_path)

    class Native(Feature):
        async def initialize(self):
            pass

        @property
        def tool_description(self):
            return "native callback ownership"

    feature = Native(agent)
    agent.features = {"Native": feature}
    callbacks = []

    async def result():
        return SignalResult(
            signal_id="callback",
            status=Status.OK if delivered else Status.FAILED,
            mode=SignalMode.COGNITION,
            duration_ms=1,
        )

    async def effect():
        callbacks.append("effect")
        await db.execute("INSERT INTO effects VALUES (1)")
        raise control_carrier(kind)

    async def undelivered(_outcome):
        if delivered:
            callbacks.append("retry")
        else:
            await effect()

    def settled():
        if settle_noise:
            raise ValueError("later local bookkeeping failure")

    handle = SignalHandle(
        signal_id="callback",
        task=agent._track_background_task(result(), name="callback-result"),
    )
    try:
        task = delivery.supervise_terminal_delivery(
            feature,
            handle,
            label="callback",
            task_name="callback-supervisor",
            on_delivered=effect,
            on_undelivered=undelivered,
            on_settled=settled,
        )
        outcome = await await_owned_task(task)
        assert outcome.error is not None and is_execution_control_error(outcome.error)
        assert callbacks == ["effect"], "control reentered non-delivery restoration"
        assert await db.fetchval("SELECT COUNT(*) FROM effects") == 1
    finally:
        await retire(agent)
