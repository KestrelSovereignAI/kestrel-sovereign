"""Native wait ownership, terminal evidence and protected event identity."""

from uuid import uuid4

import pytest
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
from kestrel_sdk.tools import Outcome
from kestrel_sdk.tools.result import ToolResult

from kestrel_sovereign.a2a.scope import NativeRecipientScope
from kestrel_sovereign.features.peers.wait_provider import A2AWaitable
from kestrel_sovereign.waits.engine import (
    TERMINAL_EVENT_AT_KEY,
    TERMINAL_EVENT_DETAIL_KEY,
    TERMINAL_EVENT_FINAL_KEY,
    TERMINAL_EVENT_KEY,
    TERMINAL_EVENT_VIEW_KEY,
)
from tests.unit.test_a2a_native_question_store import questions as questions


async def dispatch(p, verb="task"):
    request = SendMessageRequest(
        tenant="peer-tenant",
        message=Message(
            message_id=uuid4().hex,
            role=Role.ROLE_USER,
            parts=[Part(text="Native assignment")],
        ),
    )
    return await p.outbound.reserve(
        uuid4().hex,
        request,
        recipient=NativeRecipientScope("did:peer", request.tenant),
        signing_did="did:web:peer.test",
        interface=AgentInterface(
            url="https://peer.test/a2a",
            tenant=request.tenant,
            protocol_binding="JSONRPC",
            protocol_version="1.0",
        ),
        dispatch_tool="send_a2a_task",
        verb=verb,
    )


def provider(p, result=None):
    calls = []

    class Peers:
        def _native_outbound_store(self):
            return p.outbound

        async def get_peer_task_result(self, dispatch_id):
            calls.append(dispatch_id)
            return (
                result
                if result is not None
                else ToolResult.failed("Peer temporarily unreachable")
            )

    return A2AWaitable(Peers()), calls


async def observed(p, d, state):
    submitting = await p.outbound.begin_submission(d, operation_id="send")
    task = Task(
        id="peer-task", context_id="peer-context", status=TaskStatus(state=state)
    )
    return await p.outbound.acknowledge(submitting, task, operation_id="ack")


@pytest.mark.parametrize("verb", ["task", "question", "message"])
async def test_only_sender_owned_task_dispatches_can_arm_a_generic_wait(
    questions, verb
):
    p = questions
    d = await dispatch(p, verb)
    waiter, _ = provider(p)
    assert await waiter.owns_handle(d.dispatch_id) is (verb == "task")
    assert not await waiter.owns_handle("unowned")
    if verb != "task":
        assert (await waiter.poll(d.dispatch_id)).outcome is Outcome.PENDING


@pytest.mark.parametrize(
    "state,outcome",
    [
        (TaskState.TASK_STATE_COMPLETED, Outcome.DONE),
        (TaskState.TASK_STATE_FAILED, Outcome.FAILED),
        (TaskState.TASK_STATE_REJECTED, Outcome.FAILED),
        (TaskState.TASK_STATE_CANCELED, Outcome.FAILED),
    ],
)
async def test_local_native_terminal_receipt_ends_wait_without_network(
    questions, state, outcome
):
    p = questions
    d = await dispatch(p)
    await observed(p, d, state)
    waiter, calls = provider(p)
    result = await waiter.poll(d.dispatch_id)
    assert result.outcome is outcome
    assert result.data[TERMINAL_EVENT_FINAL_KEY] is True
    assert result.data["dispatch_id"] == d.dispatch_id and not calls


async def test_reserved_dispatch_stays_pending_and_unowned_watch_fails_clearly(
    questions,
):
    p = questions
    d = await dispatch(p)
    waiter, calls = provider(p)
    assert (await waiter.poll(d.dispatch_id)).outcome is Outcome.PENDING
    assert (await waiter.poll("unowned")).outcome is Outcome.FAILED
    assert not calls


async def test_proven_unsent_dispatch_ends_wait_without_inventing_remote_identity(
    questions,
):
    p = questions
    d = await dispatch(p)
    submitting = await p.outbound.begin_submission(d, operation_id="send")
    await p.outbound.record_no_delivery(submitting, operation_id="not-admitted")
    waiter, calls = provider(p)
    result = await waiter.poll(d.dispatch_id)
    assert result.outcome is Outcome.FAILED and result.data["state"] == "not_sent"
    assert result.data[TERMINAL_EVENT_FINAL_KEY] is True and not calls
    assert "remote_task_id" not in result.data


async def test_unreachable_peer_preserves_pending_native_wait(questions):
    p = questions
    d = await dispatch(p)
    await observed(p, d, TaskState.TASK_STATE_WORKING)
    waiter, calls = provider(p)
    assert (await waiter.poll(d.dispatch_id)).outcome is Outcome.PENDING
    assert calls == [d.dispatch_id]


@pytest.mark.parametrize(
    "key",
    [
        TERMINAL_EVENT_AT_KEY,
        TERMINAL_EVENT_DETAIL_KEY,
        TERMINAL_EVENT_FINAL_KEY,
        TERMINAL_EVENT_KEY,
        TERMINAL_EVENT_VIEW_KEY,
    ],
)
async def test_peer_payload_cannot_choose_wait_event_identity_or_finality(
    questions, key
):
    p = questions
    d = await dispatch(p)
    await observed(p, d, TaskState.TASK_STATE_WORKING)
    waiter, calls = provider(
        p,
        ToolResult.ok(
            "Native result", data={"state": "completed", key: "peer-forgery"}
        ),
    )
    result = await waiter.poll(d.dispatch_id)
    assert result.outcome is Outcome.DONE and key not in result.data
    assert result.data["dispatch_id"] == d.dispatch_id and calls == [d.dispatch_id]
