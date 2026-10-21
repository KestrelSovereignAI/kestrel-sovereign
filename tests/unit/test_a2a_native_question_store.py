"""Real primary question correlation, native receipt authority and durable wake claims."""

import asyncio
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

import pytest
import pytest_asyncio
from google.protobuf.json_format import ParseDict
from a2a.types import (
    AgentInterface,
    Message,
    Part,
    Role,
    SendMessageRequest,
    Task,
    TaskState,
    TaskStatus,
)

from kestrel_sovereign.a2a.extensions import QUESTION_CORRELATION_URI
from kestrel_sovereign.a2a.scope import NativeRecipientScope
from kestrel_sovereign.a2a.stores.unified.outbound_store import OutboundStore
from kestrel_sovereign.a2a.stores.unified.question_store import (
    NativeQuestionStore,
    QuestionConflict,
)


@pytest_asyncio.fixture
async def questions(native_a2a_backend_factory):
    backend = native_a2a_backend_factory()
    await backend.connect()
    outbound = OutboundStore(
        backend, sender_scope=NativeRecipientScope("did:sender", "local-tenant")
    )
    clock = SimpleNamespace(now=datetime(2026, 10, 20, 20, 0, tzinfo=UTC))
    store = NativeQuestionStore(
        outbound, clock=lambda: clock.now, claim_lease=timedelta(seconds=5)
    )
    await outbound.initialize()
    await store.initialize()
    try:
        yield SimpleNamespace(
            backend=backend, outbound=outbound, store=store, clock=clock
        )
    finally:
        await backend.close()


def correlation(kind, question_id):
    return {
        QUESTION_CORRELATION_URI: {
            "version": 1,
            "kind": kind,
            "question_id": question_id,
        }
    }


async def reserve(p, question_id="question-one", *, parts=None):
    request = SendMessageRequest(
        tenant="peer-tenant",
        metadata=correlation("question", question_id),
        message=Message(
            message_id=uuid4().hex,
            role=Role.ROLE_USER,
            parts=[Part(text="What is the result?")] if parts is None else parts,
        ),
    )
    dispatch = await p.outbound.reserve(
        uuid4().hex,
        request,
        recipient=NativeRecipientScope("did:peer", request.tenant),
        signing_did="did:web:peer.test",
        interface=AgentInterface(
            url="https://peer.test/a2a",
            protocol_binding="JSONRPC",
            protocol_version="1.0",
            tenant=request.tenant,
        ),
        dispatch_tool="send_a2a_question",
        verb="question",
    )
    question = await p.store.reserve(
        dispatch,
        request,
        origin_turn_id="local-turn",
        origin_session_id="local-session",
        deadline=p.clock.now + timedelta(seconds=10),
    )
    return question, dispatch, request


def response(
    question, request, *, state=TaskState.TASK_STATE_COMPLETED, qid=None, taskless=False
):
    answer = Message(
        message_id=uuid4().hex,
        role=Role.ROLE_AGENT,
        context_id="remote-context",
        parts=[Part(text="The result is 42")],
        metadata=correlation("answer", qid or question.question_id),
    )
    if taskless:
        return answer
    answer.task_id = "remote-task"
    return Task(
        id=answer.task_id,
        context_id=answer.context_id,
        metadata=request.metadata,
        status=TaskStatus(state=state, message=answer),
        history=[request.message],
    )


async def acknowledge(p, dispatch, answer):
    submitting = await p.outbound.begin_submission(dispatch, operation_id=uuid4().hex)
    return await p.outbound.acknowledge(submitting, answer, operation_id=uuid4().hex)


async def test_question_intent_precedes_delivery_and_keeps_local_lineage(questions):
    p = questions
    q, d, request = await reserve(p)
    assert q.question_id != d.dispatch_id and d.remote_task_id is None
    assert (q.recipient_id, q.recipient_tenant) == ("did:peer", "peer-tenant")
    assert (q.origin_turn_id, q.origin_session_id) == ("local-turn", "local-session")
    assert not await p.store.pending_wakes()
    with pytest.raises(QuestionConflict, match="already has durable"):
        await p.store.reserve(
            d,
            request,
            origin_turn_id="local-turn",
            origin_session_id="local-session",
            deadline=p.clock.now + timedelta(seconds=20),
        )


@pytest.mark.parametrize("tamper", ["request", "snapshot", "recipient"])
async def test_question_intent_refuses_unbound_or_changed_dispatch(questions, tamper):
    p = questions
    q, d, request = await reserve(p)
    if tamper == "request":
        request.message.parts[0].text = "Different question"
    elif tamper == "snapshot":
        d = replace(d, dispatch_id="foreign-dispatch")
    else:
        d = replace(d, recipient_id="did:foreign-peer")
    with pytest.raises(QuestionConflict):
        await p.store.reserve(
            d,
            request,
            origin_turn_id="local-turn",
            origin_session_id="local-session",
            deadline=p.clock.now + timedelta(seconds=20),
        )
    assert await p.store.get(q.question_id) == q


@pytest.mark.parametrize("taskless", [False, True])
async def test_authorized_correlated_answer_commits_one_complete_wake(
    questions, taskless
):
    p = questions
    q, d, request = await reserve(p)
    answer = response(q, request, taskless=taskless)
    accepted = await acknowledge(p, d, answer)
    settled = await p.store.record_answer(q.question_id, accepted, answer)
    assert settled.status == "RESOLVED" and settled.receiver_settled == 1
    wakes = await p.store.pending_wakes()
    assert len(wakes) == 1 and wakes[0].wave == "answer"
    payload = json.loads(wakes[0].payload_json)
    assert (
        payload["question_id"] == q.question_id
        and payload["dispatch_id"] == d.dispatch_id
    )
    assert payload["reply_text"] == "The result is 42"
    assert json.loads(settled.response_json).get("id") == (
        None if taskless else "remote-task"
    )
    assert await p.store.record_answer(q.question_id, accepted, answer) == settled
    assert await p.store.pending_wakes() == wakes


@pytest.mark.parametrize(
    "tamper", ["uncorrelated", "uncommitted", "peer", "tenant", "body"]
)
async def test_answer_claim_never_authorizes_settlement_or_wake(questions, tamper):
    p = questions
    q, d, request = await reserve(p)
    answer = response(
        q, request, qid="other-question" if tamper == "uncorrelated" else None
    )
    accepted = await acknowledge(p, d, answer)
    if tamper == "uncommitted":
        accepted = replace(accepted, response_digest="forged")
    elif tamper == "peer":
        accepted = replace(accepted, recipient_id="did:impostor")
    elif tamper == "tenant":
        accepted = replace(accepted, tenant="another-tenant")
    elif tamper == "body":
        answer.status.message.parts[0].text = "Changed after authorized read"
    with pytest.raises(QuestionConflict):
        await p.store.record_answer(q.question_id, accepted, answer)
    assert await p.store.get(q.question_id) == q and not await p.store.pending_wakes()


async def test_question_wake_claim_is_scoped_fenced_and_recoverable_after_restart(
    questions,
):
    p = questions
    q, d, request = await reserve(p)
    answer = response(q, request)
    accepted = await acknowledge(p, d, answer)
    await p.store.record_answer(q.question_id, accepted, answer)
    wake = (await p.store.pending_wakes())[0]
    outcomes = await asyncio.gather(p.store.claim_wake(wake), p.store.claim_wake(wake))
    claimed = next(x for x in outcomes if x is not None)
    assert sum(x is not None for x in outcomes) == 1
    p.clock.now += timedelta(seconds=6)
    restarted = NativeQuestionStore(
        p.outbound, clock=lambda: p.clock.now, claim_lease=timedelta(seconds=5)
    )
    replacement = await restarted.claim_wake(wake)
    assert (
        replacement.event_id == claimed.event_id
        and replacement.claim_token != claimed.claim_token
    )
    assert not await p.store.confirm_handoff(claimed)
    assert await restarted.confirm_handoff(replacement)
    assert not await restarted.pending_wakes()
    assert not await restarted.claim_wake(wake)


async def test_deadline_wake_precedes_late_answer_and_survives_restart(questions):
    p = questions
    q, d, request = await reserve(p)
    with pytest.raises(QuestionConflict, match="deadline"):
        await p.store.expire(q.question_id)
    p.clock.now += timedelta(seconds=11)
    expired = await p.store.expire(q.question_id)
    assert expired.status == "EXPIRED" and not expired.receiver_settled
    answer = response(q, request)
    accepted = await acknowledge(p, d, answer)
    settled = await p.store.record_answer(q.question_id, accepted, answer)
    assert settled.status == "EXPIRED" and settled.receiver_settled
    deadline, late = await p.store.pending_wakes()
    assert deadline.wave == "deadline" and late.wave == "late-answer"
    assert not json.loads(deadline.payload_json)["withdrawn"]
    assert await p.store.claim_wake(late) is None
    first = await p.store.claim_wake(deadline)
    assert await p.store.confirm_handoff(first)
    assert await p.store.claim_wake(late) is not None


async def test_foreign_actor_and_local_tenant_have_no_question_or_wake_access(
    questions,
):
    p = questions
    q, d, request = await reserve(p)
    answer = response(q, request)
    accepted = await acknowledge(p, d, answer)
    await p.store.record_answer(q.question_id, accepted, answer)
    wake = (await p.store.pending_wakes())[0]
    for scope in [
        NativeRecipientScope("did:other", "local-tenant"),
        NativeRecipientScope("did:sender", "other-tenant"),
    ]:
        other = NativeQuestionStore(
            OutboundStore(p.backend, sender_scope=scope), clock=lambda: p.clock.now
        )
        assert await other.get(q.question_id) is None
        assert not await other.pending_wakes() and await other.claim_wake(wake) is None
        with pytest.raises(QuestionConflict):
            await other.record_answer(q.question_id, accepted, answer)


async def test_withdrawn_deadline_requires_native_canceled_task_evidence(questions):
    p = questions
    q, d, request = await reserve(p)
    p.clock.now += timedelta(seconds=11)
    with pytest.raises(QuestionConflict, match="canceled-task receipt"):
        await p.store.expire(q.question_id, withdrawal=d)
    answer = response(q, request, state=TaskState.TASK_STATE_CANCELED)
    accepted = await acknowledge(p, d, answer)
    expired = await p.store.expire(q.question_id, withdrawal=accepted)
    assert expired.receiver_settled
    wake = (await p.store.pending_wakes())[0]
    assert json.loads(wake.payload_json)["withdrawn"]
    await p.store.record_answer(q.question_id, accepted, answer)
    assert len(await p.store.pending_wakes()) == 1


async def test_question_intent_retains_complete_structured_input_without_inventing_text(
    questions,
):
    p = questions
    q, d, request = await reserve(
        p,
        parts=[
            ParseDict(
                {"data": {"question": {"operation": "count", "items": [1, 2, 3]}}},
                Part(),
            )
        ],
    )
    assert q.original_question == ""
    preserved = json.loads(q.request_json)
    assert preserved["message"]["messageId"] == request.message.message_id
    assert preserved["message"]["parts"][0]["data"]["question"]["items"] == [1, 2, 3]
    answer = response(q, request, taskless=True)
    submitting = await p.outbound.begin_submission(
        d, operation_id="send-structured-question"
    )
    await p.store.acknowledge(
        submitting, answer, operation_id="answer-structured-question"
    )
    [wake] = await p.store.pending_wakes()
    assert json.loads(wake.payload_json)["request"] == preserved["message"]
