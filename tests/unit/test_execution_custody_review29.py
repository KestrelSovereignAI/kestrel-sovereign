"""Native effects cannot become retryable signal non-delivery (#3569)."""

import asyncio
from datetime import datetime, timedelta, timezone
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
from kestrel_sovereign.features.peers.feature import PeersFeature, _QuestionClaim
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
from kestrel_sovereign.storage.async_pending_a2a_question_store import (
    PendingA2AQuestionStore,
)
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
@pytest.mark.parametrize(
    "kind", ["error", "cancel", "unknown", "cancel_unknown", "group"]
)
@pytest.mark.parametrize("inline", [True, False])
async def test_actual_peer_delivery_does_not_restore_native_effect_as_waiting(
    tmp_path, kind, inline
):
    failure = control_carrier(kind)

    class NativeAgent(KestrelAgent):
        async def process_input(self, prompt):
            await self.storage.db.execute("INSERT INTO effects VALUES (1)")
            raise failure

    agent, db = await native_owner(tmp_path, agent_class=NativeAgent)
    store = PendingA2AQuestionStore(db, agent.did)
    agent.pending_a2a_questions = store
    await store.insert(
        task_id="question",
        recipient="Peer",
        original_question="question",
        origin_turn_id=None,
        origin_session_id="session",
        deadline=datetime.now(timezone.utc) + timedelta(hours=1),
    )
    assert await store.mark_resolved(
        "question", state="completed", reply_text="reply", claim="original-claim",
    )
    logs = SignalLogStore(db.backend)
    await logs.initialize()
    registry = SourceRegistry()
    registry.register(build_a2a_question_answered_registration())
    dispatcher = SignalDispatcher(
        agent=agent, registry=registry, lock_manager=OrderedLockManager(), store=logs
    )
    agent.dispatcher = dispatcher
    feature = PeersFeature(agent)
    feature._own_name = "Sender"
    agent.features = {"Peers": feature}
    args = dict(
        task_id="question",
        recipient="Peer",
        original_question="question",
        sess_id="session",
        state="completed",
        reply_text="reply",
        causation_chain=None,
        claim=_QuestionClaim(store, "original-claim"),
    )
    try:
        if inline:
            with pytest.raises(BaseException) as caught:
                fired = await feature._fire_question_answered_signal(**args)
                if not fired:
                    await feature._restore_pending_question_waiting(**args)
            assert is_execution_control_error(caught.value)
        else:
            assert await feature._fire_question_answered_signal(
                **args, await_delivery=False
            )
            tasks = tuple(feature._owned_background_tasks)
            assert len(tasks) == 1
            outcome = await await_owned_task(tasks[0])
            assert outcome.error is not None and is_execution_control_error(
                outcome.error
            )
        assert await db.fetchval("SELECT COUNT(*) FROM effects") == 1
        assert (await store.get("question")).status == "RESOLVED"
        assert not any(
            task.get_name().startswith("a2a_question_answered_retry:")
            for task in agent._background_tasks
        )
    finally:
        try:
            assert await db.fetchval("SELECT COUNT(*) FROM effects") == 1
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
