"""Real native ledgers, durable signal admissions and restarted question recovery."""

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import timedelta
from types import SimpleNamespace

import pytest
from a2a.types import TaskState

from kestrel_sovereign.a2a.question_supervisor import NativeQuestionSupervisor
from kestrel_sovereign.a2a.stores.unified.outbound_store import DispatchState
from kestrel_sovereign.signals import (
    DurableConsumerRegistration,
    OrderedLockManager,
    SignalDispatcher,
    SignalLogStore,
    SourceRegistry,
)
from kestrel_sovereign.signals.sources.a2a_question_answered import (
    DURABLE_COGNITION_CONSUMER_ID,
    build_a2a_question_answered_registration,
    build_signal_for_question_answered,
)
from tests.unit.test_a2a_native_question_store import reserve, response
from tests.unit.test_a2a_native_question_store import questions as questions
from tests.unit.test_durable_signal_delivery import _Agent


async def dispatcher_for(p):
    logs = SignalLogStore(p.backend)
    await logs.initialize()
    agent = _Agent(p.store.sender_scope.recipient_id)
    registry = SourceRegistry()
    registry.register(build_a2a_question_answered_registration())
    dispatcher = SignalDispatcher(
        agent=agent, registry=registry, lock_manager=OrderedLockManager(), store=logs
    )
    await dispatcher.initialize_durable_delivery()
    await dispatcher.register_durable_consumer(
        DurableConsumerRegistration(
            consumer_id=DURABLE_COGNITION_CONSUMER_ID,
            source="a2a.question_answered",
            agent_id=agent.did,
            max_attempts=0,
        )
    )
    return agent, dispatcher


@asynccontextmanager
async def no_peer_read(_dispatch_id):
    raise AssertionError("A retained terminal answer needs no peer request")
    yield


async def test_taskless_answer_and_receipt_commit_together_and_survive_restart(
    questions,
):
    p = questions
    q, d, request = await reserve(p)
    submitting = await p.outbound.begin_submission(d, operation_id="original-send")
    answer = response(q, request, taskless=True)
    receipt = await p.store.acknowledge(
        submitting, answer, operation_id="native-response"
    )
    assert receipt.state is DispatchState.COMPLETE and receipt.remote_task_id is None
    stored = await p.store.for_dispatch(d.dispatch_id)
    assert (
        stored.receiver_settled
        and json.loads(stored.response_json)["parts"][0]["text"] == "The result is 42"
    )
    assert not await p.store.unsettled()
    agent, dispatcher = await dispatcher_for(p)
    try:
        supervisor = NativeQuestionSupervisor(
            store=p.store,
            open_dispatch=no_peer_read,
            dispatcher=dispatcher,
            interval=1,
            is_current=lambda: True,
        )
        await supervisor.recover_once()
        assert not await p.store.pending_wakes()
        # Simulate a producer restart: the canonical durable event remains one
        # selected delivery even when cognition has not been started yet.
        supervisor = NativeQuestionSupervisor(
            store=p.store,
            open_dispatch=no_peer_read,
            dispatcher=dispatcher,
            interval=1,
            is_current=lambda: True,
        )
        await supervisor.recover_once()
        events = await p.backend.fetch_one("SELECT COUNT(*) FROM durable_signal_events")
        deliveries = await p.backend.fetch_one(
            "SELECT COUNT(*) FROM durable_signal_deliveries"
        )
        assert events[0] == deliveries[0] == 1
    finally:
        await dispatcher.shutdown_durable_delivery()
        await asyncio.gather(*agent.tasks, return_exceptions=True)


async def test_invalid_taskless_answer_rolls_back_delivery_receipt_and_wake(questions):
    from kestrel_sovereign.storage.db.interface import TransactionError
    from kestrel_sovereign.a2a.stores.unified.question_store import QuestionConflict

    p = questions
    q, d, request = await reserve(p)
    submitting = await p.outbound.begin_submission(d, operation_id="original-send")
    answer = response(q, request, taskless=True, qid="another-question")
    with pytest.raises(
        (TransactionError, QuestionConflict), match="outstanding question identity"
    ):
        await p.store.acknowledge(submitting, answer, operation_id="bad-response")
    assert await p.outbound.get(d.dispatch_id) == submitting
    assert not (await p.store.get(q.question_id)).receiver_settled
    assert not await p.store.pending_wakes()


async def test_answer_after_clock_deadline_creates_ordered_deadline_and_late_waves(
    questions,
):
    p = questions
    q, d, request = await reserve(p)
    submitting = await p.outbound.begin_submission(d, operation_id="original-send")
    p.clock.now += timedelta(seconds=11)
    await p.store.acknowledge(
        submitting, response(q, request), operation_id="late-response"
    )
    wakes = await p.store.pending_wakes()
    assert [wake.wave for wake in wakes] == ["deadline", "late-answer"]
    agent, dispatcher = await dispatcher_for(p)
    try:
        supervisor = NativeQuestionSupervisor(
            store=p.store,
            open_dispatch=no_peer_read,
            dispatcher=dispatcher,
            interval=1,
            is_current=lambda: True,
        )
        await supervisor.recover_once()
        assert not await p.store.pending_wakes()
        rows = await p.backend.fetch_all(
            "SELECT event_id FROM durable_signal_events ORDER BY source_sequence"
        )
        assert [row[0] for row in rows] == [wake.event_id for wake in wakes]
    finally:
        await dispatcher.shutdown_durable_delivery()
        await asyncio.gather(*agent.tasks, return_exceptions=True)


async def test_unknown_cancellation_is_read_without_resending(questions):
    from a2a.types import CancelTaskRequest

    p = questions
    q, d, request = await reserve(p)
    submitting = await p.outbound.begin_submission(d, operation_id="send")
    active = response(q, request, state=TaskState.TASK_STATE_WORKING)
    dispatch = await p.outbound.acknowledge(submitting, active, operation_id="ack")
    canceling = await p.outbound.begin_cancellation(
        dispatch,
        CancelTaskRequest(id=active.id, tenant=dispatch.tenant),
        operation_id="cancel",
    )
    dispatch = await p.outbound.retain_cancellation_failure(
        canceling, may_have_sent=True, operation_id="lost-cancel-ack"
    )
    p.clock.now += timedelta(seconds=11)
    reads = []

    class ReadOnlyPeer:
        async def get_task(self, current):
            reads.append(current)
            return SimpleNamespace(dispatch=current, response=active)

    @asynccontextmanager
    async def retained(_dispatch_id):
        yield ReadOnlyPeer(), dispatch

    agent, dispatcher = await dispatcher_for(p)
    try:
        supervisor = NativeQuestionSupervisor(
            store=p.store,
            open_dispatch=retained,
            dispatcher=dispatcher,
            interval=1,
            is_current=lambda: True,
        )
        await supervisor.recover_once()
        await supervisor.recover_once()
        assert len(reads) == 2
        assert (
            await p.outbound.get(d.dispatch_id)
        ).cancel_state is DispatchState.UNKNOWN
        assert (await p.store.get(q.question_id)).status == "EXPIRED"
    finally:
        await dispatcher.shutdown_durable_delivery()
        await asyncio.gather(*agent.tasks, return_exceptions=True)


def test_question_signal_uses_local_dispatch_and_original_session_for_taskless_reply():
    payload = dict(
        question_id="question",
        request={
            "messageId": "question-input",
            "role": "ROLE_USER",
            "parts": [{"text": "Why?"}],
        },
        dispatch_id="dispatch",
        recipient_id="did:peer",
        original_question="Why?",
        reply_text="é" * 9000,
        state="rejected",
        origin_session_id="local-chat",
        late=False,
        withdrawn=False,
    )
    signal = build_signal_for_question_answered(
        event_id="event", payload=payload, target_agent="did:sender"
    )
    assert signal.id == signal.dedupe_key == "event"
    assert signal.session_id == "local-chat"
    assert signal.payload["truncated"]
    assert 'get_peer_task_result("dispatch")' in signal.payload["reply_text"]
    assert len(signal.payload["reply_text"].encode()) <= 8192
    assert "task_id" not in signal.payload and payload["reply_text"] == "é" * 9000


async def test_crash_after_durable_admission_before_outbox_checkpoint_deduplicates(
    questions,
):
    p = questions
    q, d, request = await reserve(p)
    submitting = await p.outbound.begin_submission(d, operation_id="send")
    await p.store.acknowledge(submitting, response(q, request), operation_id="answer")
    candidate = (await p.store.pending_wakes())[0]
    claimed = await p.store.claim_wake(candidate)
    agent, dispatcher = await dispatcher_for(p)
    try:
        first = build_signal_for_question_answered(
            event_id=claimed.event_id,
            payload=json.loads(claimed.payload_json),
            target_agent=agent.did,
        )
        handle = await dispatcher.enqueue_durable_cognition(
            first,
            source_event_id=claimed.event_id,
            consumer_id=DURABLE_COGNITION_CONSUMER_ID,
        )
        assert (await handle.wait_for_durable_admission()).acknowledged
        # Process loss here leaves both a committed delivery and an unretired
        # outbox claim. A restarted producer must retire the same event once.
        p.clock.now += timedelta(seconds=6)
        replacement = NativeQuestionSupervisor(
            store=p.store,
            open_dispatch=no_peer_read,
            dispatcher=dispatcher,
            interval=1,
            is_current=lambda: True,
        )
        await replacement.recover_once()
        assert not await p.store.pending_wakes()
        assert (
            await p.backend.fetch_one("SELECT COUNT(*) FROM durable_signal_events")
        )[0] == 1
        assert (
            await p.backend.fetch_one("SELECT COUNT(*) FROM durable_signal_deliveries")
        )[0] == 1
    finally:
        await dispatcher.shutdown_durable_delivery()
        await asyncio.gather(*agent.tasks, return_exceptions=True)


async def test_privacy_elided_question_wake_rehydrates_from_owned_outbox_after_restart(
    questions,
):
    from types import MethodType
    from kestrel_sovereign.agent.event_manager import EventManagerMixin
    from kestrel_sovereign.privacy import get_privacy_preset
    from tests.unit.test_durable_signal_delivery import _HoldSnapshots, _held_state

    p = questions
    q, d, request = await reserve(p)
    submitting = await p.outbound.begin_submission(d, operation_id="send")
    await p.store.acknowledge(
        submitting, response(q, request, taskless=True), operation_id="answer"
    )
    agent_a, dispatcher_a = await dispatcher_for(p)
    agent_a.privacy_config = get_privacy_preset("ephemeral")
    agent_a._hold_store = _HoldSnapshots(_held_state(agent_a.did))
    try:
        supervisor = NativeQuestionSupervisor(
            store=p.store,
            open_dispatch=no_peer_read,
            dispatcher=dispatcher_a,
            interval=1,
            is_current=lambda: True,
        )
        await supervisor.handoff_wakes()
        [delivery] = await dispatcher_a.list_durable_deliveries(
            consumer_id=DURABLE_COGNITION_CONSUMER_ID
        )
        assert delivery.event.payload == {"_privacy_gated": "none"}
        [owed] = await p.store.pending_wakes()
        assert owed.handed_off_at is None
    finally:
        await dispatcher_a.shutdown_durable_delivery()
        await asyncio.gather(*agent_a.tasks, return_exceptions=True)

    agent_b, dispatcher_b = await dispatcher_for(p)
    agent_b.privacy_config = get_privacy_preset("ephemeral")
    agent_b.task_manager = SimpleNamespace(
        questions_for_actor=lambda did: p.store if did == agent_b.did else None
    )
    agent_b.durable_rehydratable_sources = (
        EventManagerMixin.durable_rehydratable_sources
    )
    agent_b.rehydrate_durable_cognition_signal = MethodType(
        EventManagerMixin.rehydrate_durable_cognition_signal, agent_b
    )
    prompts = []

    async def resumed(prompt):
        prompts.append(prompt)
        return "Handled the original answer"

    agent_b.process_input = resumed
    try:
        await dispatcher_b.start_durable_cognition_consumer(
            DURABLE_COGNITION_CONSUMER_ID
        )
        async with asyncio.timeout(5):
            while True:
                [delivery] = await dispatcher_b.list_durable_deliveries(
                    consumer_id=DURABLE_COGNITION_CONSUMER_ID
                )
                if delivery.status == "acknowledged":
                    break
                await asyncio.sleep(0.01)
        assert len(prompts) == 1 and "The result is 42" in prompts[0]
        assert q.dispatch_id in prompts[0] and q.original_question in prompts[0]
    finally:
        await dispatcher_b.shutdown_durable_delivery()
        await asyncio.gather(*agent_b.tasks, return_exceptions=True)


async def test_unsent_question_retires_with_local_failure_without_peer_reads(questions):
    p = questions
    q, d, _ = await reserve(p)
    submitting = await p.outbound.begin_submission(d, operation_id="attempt")
    await p.outbound.record_no_delivery(submitting, operation_id="http-not-admitted")
    agent, dispatcher = await dispatcher_for(p)
    try:
        supervisor = NativeQuestionSupervisor(
            store=p.store,
            open_dispatch=no_peer_read,
            dispatcher=dispatcher,
            interval=1,
            is_current=lambda: True,
        )
        await supervisor.recover_once()
        assert not await p.store.unsettled() and not await p.store.pending_wakes()
        row = await p.store.get(q.question_id)
        assert row.status == "RESOLVED" and row.response_json is None
        [delivery] = await dispatcher.list_durable_deliveries(
            consumer_id=DURABLE_COGNITION_CONSUMER_ID
        )
        assert delivery.event.payload["state"] == "failed"
        assert "not delivered" in delivery.event.payload["reply_text"]
    finally:
        await dispatcher.shutdown_durable_delivery()
        await asyncio.gather(*agent.tasks, return_exceptions=True)


async def test_live_original_submission_is_not_reconciled_as_a_crashed_sender(
    questions,
):
    p = questions
    q, d, _ = await reserve(p)
    with p.outbound.active_submission(d):
        submitting = await p.outbound.begin_submission(d, operation_id="sending-now")
        supervisor = NativeQuestionSupervisor(
            store=p.store,
            open_dispatch=no_peer_read,
            dispatcher=None,
            interval=1,
            is_current=lambda: True,
        )
        await supervisor._observe(q)
        assert await p.outbound.get(d.dispatch_id) == submitting
    assert not p.outbound.submission_active(d.dispatch_id)


async def test_full_taskless_question_result_is_read_locally_and_detects_corruption(
    questions,
):
    from kestrel_sovereign.a2a.stores.unified.question_store import QuestionConflict
    from kestrel_sovereign.features.peers.feature import PeersFeature
    from tests.utils.native_a2a import manager_for

    p = questions
    q, d, request = await reserve(p)
    submitting = await p.outbound.begin_submission(d, operation_id="send")
    await p.store.acknowledge(
        submitting, response(q, request, taskless=True), operation_id="answer"
    )
    manager = manager_for(p.backend, p.outbound.sender_scope, owns_backend=False)
    await manager.initialize()
    try:
        feature = PeersFeature(
            SimpleNamespace(
                did=p.outbound.sender_scope.recipient_id, task_manager=manager
            )
        )
        result = await feature.get_peer_task_result(d.dispatch_id)
        assert (
            result.data["response_kind"] == "message"
            and result.data["remote_task_id"] is None
        )
        assert result.data["reply_text"] == "The result is 42"
        assert result.data["message"]["parts"][0]["text"] == "The result is 42"
        assert "task" not in result.data
        stored = await p.store.get(q.question_id)
        forged = json.loads(stored.response_json)
        forged["parts"][0]["text"] = "Changed after native admission"
        await p.backend.execute(
            "UPDATE a2a_native_questions SET response_json = ? WHERE question_id = ?",
            (json.dumps(forged), q.question_id),
        )
        with pytest.raises(QuestionConflict, match="differs from its native receipt"):
            await feature.get_peer_task_result(d.dispatch_id)
    finally:
        await manager.close()


async def test_data_only_answer_retains_native_parts_and_can_hand_off_without_text(
    questions,
):
    from a2a.types import Part
    from google.protobuf.json_format import MessageToDict, ParseDict

    p = questions
    q, d, request = await reserve(p)
    submitting = await p.outbound.begin_submission(d, operation_id="send")
    answer = response(q, request, taskless=True)
    del answer.parts[:]
    answer.parts.append(ParseDict({"data": {"result": {"value": 42}}}, Part()))
    receipt = await p.store.acknowledge(submitting, answer, operation_id="data-answer")
    [wake] = await p.store.pending_wakes()
    payload = json.loads(wake.payload_json)
    assert payload["reply_text"] == ""
    assert payload["requires_full_response"] is True
    assert payload["request"] == MessageToDict(request.message)
    retained = await p.store.retained_response(receipt)
    assert MessageToDict(retained.parts[0].data) == {"result": {"value": 42}}
    agent, dispatcher = await dispatcher_for(p)
    try:
        supervisor = NativeQuestionSupervisor(
            store=p.store,
            open_dispatch=no_peer_read,
            dispatcher=dispatcher,
            interval=1,
            is_current=lambda: True,
        )
        await supervisor.recover_once()
        assert not await p.store.pending_wakes()
        [delivery] = await dispatcher.list_durable_deliveries(
            consumer_id=DURABLE_COGNITION_CONSUMER_ID
        )
        assert delivery.event.payload["reply_text"] == ""
    finally:
        await dispatcher.shutdown_durable_delivery()
        await asyncio.gather(*agent.tasks, return_exceptions=True)


async def test_slow_peer_read_does_not_delay_another_questions_deadline_wake(questions):
    p = questions
    first, dispatch, request = await reserve(p, "slow-peer-question")
    submitting = await p.outbound.begin_submission(dispatch, operation_id="send")
    active = response(first, request, state=TaskState.TASK_STATE_WORKING)
    acknowledged = await p.outbound.acknowledge(submitting, active, operation_id="ack")
    entered, release = asyncio.Event(), asyncio.Event()

    @asynccontextmanager
    async def slow_peer(_dispatch_id):
        entered.set()
        await release.wait()
        yield SimpleNamespace(), acknowledged

    agent, dispatcher = await dispatcher_for(p)
    supervisor = NativeQuestionSupervisor(
        store=p.store,
        open_dispatch=slow_peer,
        dispatcher=dispatcher,
        interval=0.01,
        is_current=lambda: True,
    )
    owner = asyncio.create_task(supervisor.run())
    try:
        async with asyncio.timeout(5):
            await entered.wait()
            second, _, _ = await reserve(p, "new-deadline-while-read-blocked")
            p.clock.now += timedelta(seconds=11)
            supervisor.notify()
            while (
                await p.store.get(second.question_id)
            ).status != "EXPIRED" or await p.store.pending_wakes():
                await asyncio.sleep(0.01)
        assert not release.is_set() and not owner.done()
        events = await p.backend.fetch_all(
            "SELECT source_event_id FROM durable_signal_events"
        )
        assert (
            len(events) == 2
        )  # Each deadline was handed off while the read remained blocked.
    finally:
        owner.cancel()
        await asyncio.gather(owner, return_exceptions=True)
        await dispatcher.shutdown_durable_delivery()
        await asyncio.gather(*agent.tasks, return_exceptions=True)


async def test_replaced_question_runtime_cannot_handoff_an_old_outbox(questions):
    p = questions
    q, d, request = await reserve(p)
    submitting = await p.outbound.begin_submission(d, operation_id="send")
    await p.store.acknowledge(submitting, response(q, request), operation_id="answer")
    agent, dispatcher = await dispatcher_for(p)
    try:
        supervisor = NativeQuestionSupervisor(
            store=p.store,
            open_dispatch=no_peer_read,
            dispatcher=dispatcher,
            interval=1,
            is_current=lambda: False,
        )
        with pytest.raises(RuntimeError, match="replaced runtime"):
            await supervisor.handoff_wakes()
        [wake] = await p.store.pending_wakes()
        assert wake.handed_off_at is None and wake.claim_token is None
        assert not await dispatcher.list_durable_deliveries(
            consumer_id=DURABLE_COGNITION_CONSUMER_ID
        )
    finally:
        await dispatcher.shutdown_durable_delivery()
        await asyncio.gather(*agent.tasks, return_exceptions=True)
