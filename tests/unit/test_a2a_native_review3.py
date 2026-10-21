"""Full-review defects reproduced through official SDK bindings and real storage."""

import json
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from a2a.utils.errors import A2AError
from google.protobuf.json_format import MessageToDict

from kestrel_sovereign.a2a.extensions import (
    question_correlation_metadata,
)
from kestrel_sovereign.a2a.native_client import (
    NativePeerProtocolError,
    NativePeerSession,
)
from kestrel_sovereign.a2a.question_supervisor import NativeQuestionSupervisor
from kestrel_sovereign.a2a.stores.unified.outbound_store import (
    DispatchState,
    OutboundDispatchConflict,
)
from kestrel_sovereign.a2a.stores.unified.question_store import NativeQuestionStore
from tests.unit.test_a2a_native_client import peer_session as peer_session
from tests.unit.test_a2a_native_client import recipient_identity as recipient_identity
from tests.unit.test_a2a_native_client import request_for, reserve
from tests.unit.test_a2a_native_question_store import questions as questions
from tests.unit.test_a2a_native_question_supervisor import dispatcher_for


def response_with_wire_variation(request, task, protocol, fault):
    body = MessageToDict(task)
    if fault == "unknown-field":
        body["createdAt"] = "2026-10-21T00:00:00Z"
    elif fault == "nonfinite":
        body["artifacts"] = [
            {"artifactId": "bad-stats", "parts": [{"data": {"mean": float("nan")}}]}
        ]
    if protocol == "JSONRPC":
        body = {
            "jsonrpc": "2.0",
            "id": json.loads(request.content)["id"],
            "result": body,
        }
    if fault == "envelope-list":
        body = []
    return httpx.Response(
        200,
        text="{" if fault == "malformed-json" else json.dumps(body),
        headers={"content-type": "application/json"},
    )


@pytest.mark.parametrize(
    "fault", ["unknown-field", "envelope-list", "malformed-json", "nonfinite"]
)
async def test_sdk_peer_read_ignores_unknown_fields_and_refuses_malformed_values(
    peer_session, fault
):
    p = peer_session
    dispatch, wire = await reserve(p)
    accepted = await p.session.send(dispatch, wire)
    requests = []

    def respond(request):
        requests.append(request)
        return response_with_wire_variation(
            request, accepted.response, p.arguments["interface"].protocol_binding, fault
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        session = NativePeerSession(**(p.arguments | {"http_client": http}))
        try:
            if fault == "unknown-field":
                # A2A 1.0.1 section 5.7 recommends forward-compatible field
                # handling. Unknown fields cannot become task authority.
                receipt = await session.get_task(accepted.dispatch)
                assert receipt.response == accepted.response
                assert (
                    receipt.dispatch.remote_task_id == accepted.dispatch.remote_task_id
                )
                assert receipt.dispatch.recipient_id == accepted.dispatch.recipient_id
                assert await p.store.get(dispatch.dispatch_id) == receipt.dispatch
            else:
                with pytest.raises(
                    (A2AError, NativePeerProtocolError, OutboundDispatchConflict)
                ):
                    await session.get_task(accepted.dispatch)
                assert await p.store.get(dispatch.dispatch_id) == accepted.dispatch
            assert len(requests) == 1
        finally:
            await session.close()


@pytest.mark.parametrize(
    "fault", ["unknown-field", "envelope-list", "malformed-json", "nonfinite"]
)
async def test_sdk_question_read_does_not_strand_another_deadline(peer_session, fault):
    p = peer_session
    clock = SimpleNamespace(now=datetime.now(UTC))
    questions = NativeQuestionStore(p.store, clock=lambda: clock.now)
    await questions.initialize()
    signer = replace(
        p.arguments["signer"],
        extensions=frozenset(
            item.uri for item in p.arguments["card"].capabilities.extensions
        ),
    )
    session = NativePeerSession(
        **(p.arguments | {"signer": signer, "question_store": questions})
    )

    async def reserve_question(deadline):
        question_id = uuid4().hex
        request = request_for(p)
        request.metadata.update(question_correlation_metadata("question", question_id))
        dispatch, wire = await session.reserve(
            request, dispatch_tool="send_a2a_question", verb="question"
        )
        await questions.reserve(
            dispatch,
            wire,
            origin_turn_id="original-turn",
            origin_session_id="original-session",
            deadline=deadline,
        )
        return question_id, dispatch, wire

    try:
        bad_id, dispatch, wire = await reserve_question(
            clock.now + timedelta(seconds=60)
        )
        accepted = await session.send(dispatch, wire)
        other_id, _, _ = await reserve_question(clock.now + timedelta(seconds=10))
        original_question = await questions.get(bad_id)

        def respond(request):
            return response_with_wire_variation(
                request,
                accepted.response,
                p.arguments["interface"].protocol_binding,
                fault,
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
            malformed = NativePeerSession(
                **(
                    p.arguments
                    | {
                        "http_client": http,
                        "signer": signer,
                        "question_store": questions,
                    }
                )
            )

            @asynccontextmanager
            async def open_dispatch(dispatch_id):
                yield malformed, await p.store.get(dispatch_id)

            agent, dispatcher = await dispatcher_for(
                SimpleNamespace(backend=p.admission.backend, store=questions)
            )
            supervisor = NativeQuestionSupervisor(
                store=questions,
                open_dispatch=open_dispatch,
                dispatcher=dispatcher,
                interval=1,
                max_interval=60,
                is_current=lambda: True,
            )
            try:
                await supervisor._read_pass()
                current = await p.store.get(dispatch.dispatch_id)
                if fault == "unknown-field":
                    assert current.revision == accepted.dispatch.revision + 1
                    assert current.remote_task_id == accepted.dispatch.remote_task_id
                    assert current.response_digest == accepted.dispatch.response_digest
                    assert current.recipient_id == accepted.dispatch.recipient_id
                else:
                    assert current == accepted.dispatch
                assert await questions.get(bad_id) == original_question
                clock.now += timedelta(seconds=11)
                await supervisor._deadline_pass()
                await supervisor.handoff_wakes()
                assert (await questions.get(other_id)).status == "EXPIRED"
                row = await p.admission.backend.fetch_one(
                    "SELECT event_id FROM a2a_native_question_wakes WHERE question_id = ?",
                    (other_id,),
                )
                assert (await questions.wake_for_event(row[0])).handed_off_at
                assert (
                    await p.store.get(dispatch.dispatch_id)
                ).state is DispatchState.ACKNOWLEDGED
            finally:
                await dispatcher.shutdown_durable_delivery()
                import asyncio

                await asyncio.gather(*agent.tasks, return_exceptions=True)
                await malformed.close()
    finally:
        await session.close()


@pytest.mark.parametrize("kind", ["parse", "value", "type", "database"])
async def test_peer_decode_boundary_preserves_original_execution_and_storage_errors(
    peer_session, kind
):
    from google.protobuf.json_format import ParseError

    from kestrel_sovereign.execution_custody import (
        ExecutionCommitOutcomeError,
        is_execution_control_error,
    )
    from kestrel_sovereign.storage.db.interface import QueryError

    p = peer_session
    dispatch, wire = await reserve(p)
    accepted = await p.session.send(dispatch, wire)
    error = {
        "parse": ParseError,
        "value": ValueError,
        "type": TypeError,
        "database": QueryError,
    }[kind]("Original native outcome")
    if kind != "database":
        error.__cause__ = ExecutionCommitOutcomeError("unknown")
    p.state.response_error = error
    with pytest.raises(type(error)) as failure:
        await p.session.get_task(accepted.dispatch)
    assert failure.value is error
    assert is_execution_control_error(failure.value) is (kind != "database")
    assert await p.store.get(dispatch.dispatch_id) == accepted.dispatch


async def test_identical_working_responses_back_off_but_actual_progress_resets_the_delay(
    questions,
):
    from a2a.types import TaskState

    from kestrel_sovereign.a2a.native_client import NativeDispatchReceipt
    from tests.unit.test_a2a_native_question_store import reserve as reserve_question
    from tests.unit.test_a2a_native_question_store import response

    p = questions
    q, dispatch, request = await reserve_question(p, deadline_seconds=3600)
    submitting = await p.outbound.begin_submission(
        dispatch, operation_id="original-send"
    )
    active = response(q, request, state=TaskState.TASK_STATE_WORKING)
    await p.store.acknowledge(submitting, active, operation_id="original-working")
    elapsed = SimpleNamespace(now=0)
    reads = []

    class Peer:
        async def get_task(self, dispatch):
            reads.append(elapsed.now)
            committed = await p.store.acknowledge(
                dispatch, active, operation_id=uuid4().hex
            )
            return NativeDispatchReceipt(committed, active)

    @asynccontextmanager
    async def open_dispatch(dispatch_id):
        yield Peer(), await p.outbound.get(dispatch_id)

    supervisor = NativeQuestionSupervisor(
        store=p.store,
        open_dispatch=open_dispatch,
        dispatcher=None,
        interval=1,
        max_interval=60,
        clock=lambda: elapsed.now,
        is_current=lambda: True,
    )
    for tick in range(301):
        elapsed.now = tick
        await supervisor._read_pass()
    assert len(reads) <= 10, (
        f"Unchanged Task responses reset the capped backoff: {len(reads)} reads"
    )
    assert supervisor._next_read[q.question_id][1] == 60
    active.status.message.parts[0].text = "The peer made new progress"
    elapsed.now = supervisor._next_read[q.question_id][0]
    await supervisor._read_pass()
    assert supervisor._next_read[q.question_id][1] == 1
