"""Native SDK messages drive the real Stop policy and durable receipt rail."""

from contextlib import asynccontextmanager
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest
import pytest_asyncio
from a2a.client.errors import A2AClientError
from a2a.helpers.proto_helpers import new_text_message
from a2a.types import Message, Role, SendMessageRequest
from a2a.utils.errors import ContentTypeNotSupportedError, InvalidParamsError
from starlette.applications import Starlette

from kestrel_sovereign.a2a.extensions import (
    PEER_STOP_URI,
    SKILL_HINT_URI,
    parse_message_metadata,
)
from kestrel_sovereign.a2a.native_card import create_native_agent_card
from kestrel_sovereign.a2a.native_client import NativePeerSession
from kestrel_sovereign.a2a.native_routes import NativeA2AHTTP
from kestrel_sovereign.a2a.projections import embedded_json_value
from kestrel_sovereign.a2a.request_handler import NativeA2ARequestHandler
from tests.unit import (
    test_a2a_native_client,
    test_a2a_native_http,
    test_peer_stop_signals,
)
from tests.utils.native_a2a import manager_for
from tests.utils.native_a2a_security import PRINCIPAL

wire = test_a2a_native_http.wire
rail = test_peer_stop_signals.rail
recipient_identity = test_a2a_native_client.recipient_identity


@pytest_asyncio.fixture
async def control(wire, rail, recipient_identity):
    rail.agent.did = wire.admission.scope.recipient_id
    rail.agent.task_manager = wire.manager
    identity, _ = recipient_identity
    card = create_native_agent_card(
        identity=identity, scope=wire.admission.scope,
        routing_key=wire.admission.guard.routing_key,
        public_base_url="http://native-core.test", name="Peer",
        description="Native Stop peer", version="1", skills=wire.card.skills,
        peer_stop=True,
    )
    handler = NativeA2ARequestHandler(
        wire.manager, wire.admission.guard, card, peer_stop_target=rail.agent
    )
    binding = NativeA2AHTTP(handler, public_base_url="http://native-core.test")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(Starlette(routes=[binding.route])),
        base_url="http://native-core.test",
        headers={"A2A-Version": "1.0"},
    ) as http:
        def client(*, streaming=False, activated=True):
            return wire.client(
                streaming=streaming,
                polling=False,
                selected_card=card,
                http_client=http,
                sender_interceptor=replace(
                    wire.interceptor,
                    extensions=frozenset(item.uri for item in card.capabilities.extensions)
                    - (set() if activated else {PEER_STOP_URI}),
                ),
            )

        yield SimpleNamespace(
            wire=wire, rail=rail, card=card, handler=handler, client=client, http=http
        )


@pytest_asyncio.fixture
async def sender(control, recipient_identity):
    from kestrel_sovereign.a2a.scope import NativeRecipientScope
    from kestrel_sovereign.features.peers.directory import (
        PeerAccessDeniedError,
        PeerIdentity,
        PeerRequester,
    )
    from kestrel_sovereign.features.peers.feature import PeersFeature

    c = control
    manager = manager_for(
        c.wire.admission.backend,
        NativeRecipientScope(PRINCIPAL, "sender-tenant"), owns_backend=False,
    )
    await manager.initialize()
    agent = SimpleNamespace(did=PRINCIPAL, agent_id=PRINCIPAL, task_manager=manager)
    requester = PeerRequester(PRINCIPAL, object())
    peer = PeerIdentity(c.wire.admission.scope.recipient_id, "peer", "registered-agent")
    state = SimpleNamespace(allowed=True, admissions=0)
    identity, document = recipient_identity

    class Directory:
        async def resolve_peer(self, actor, name):
            assert actor is requester and name == "Peer"
            return peer if state.allowed else None

        @asynccontextmanager
        async def open_a2a_session(self, actor, selected):
            assert actor is requester and selected is peer

            async def authorize():
                state.admissions += 1
                if not state.allowed:
                    raise PeerAccessDeniedError("Peer route revoked")

            session = NativePeerSession(
                card=c.card, signing_did=identity.signing_did,
                trusted_document=document, recipient=c.wire.admission.scope,
                interface=next(item for item in c.card.supported_interfaces
                               if item.protocol_binding == c.wire.protocol),
                signer=replace(c.wire.interceptor,
                               extensions=frozenset(item.uri for item in c.card.capabilities.extensions)),
                http_client=c.http, outbound_store=manager.outbound_store,
                authorize=authorize,
            )
            try:
                yield session
            finally:
                await session.close()

    feature = PeersFeature(agent)
    feature._peer_router = Directory()
    feature._peer_requester = requester
    feature._own_name = "Sender"
    try:
        yield SimpleNamespace(control=c, manager=manager, state=state, feature=feature)
    finally:
        await manager.close()


def request(**changes):
    intent = {
        "scope": "agent", "target": None, "reason": "Please stop",
        "cascade": False, "correlation_id": "native-stop-1",
    } | changes
    return SendMessageRequest(
        message=new_text_message(
            intent["reason"] or "", role=Role.ROLE_USER, context_id="stop-context"
        ),
        metadata={PEER_STOP_URI: {"version": 1, "intent": intent}},
    )


async def receipt(control, params=None, *, streaming=False):
    params = request() if params is None else params
    params.tenant = control.wire.admission.scope.tenant
    events = [event async for event in control.client(streaming=streaming).send_message(
        params
    )]
    assert len(events) == 1 and events[0].WhichOneof("payload") == "message"
    message = events[0].message
    assert isinstance(message, Message) and message.role == Role.ROLE_AGENT
    assert message.context_id == "stop-context" and not message.task_id
    present, value = embedded_json_value(message.parts[0])
    assert present
    return value


@pytest.mark.parametrize("streaming", [False, True])
async def test_native_stop_returns_receipted_control_without_task_or_cognition(
    control, streaming
):
    agent = control.rail.agent
    agent._active_request_ids.add("turn-1")
    result = await receipt(control, streaming=streaming)
    assert result["recorded"] is True and result["receipt_kind"] == "operation"
    assert result["stop_outcomes"][0]["disposition"] == "stopped"
    assert agent.cancelled == ["turn-1"]
    replay = await receipt(control, streaming=streaming)
    assert replay["stop_outcomes"] == result["stop_outcomes"]
    assert agent.cancelled == ["turn-1"]
    page = await control.rail.receipts.list_receipts(limit=10)
    assert len(page.receipts) == 1
    assert page.receipts[0].actor_id == PRINCIPAL
    assert control.wire.submissions == []
    assert await control.wire.manager.task_store.backend.fetch_one(
        "SELECT COUNT(*) FROM a2a_tasks"
    ) == (0,)


@pytest.mark.parametrize("changes", [{"cascade": True}, {"scope": "host"}])
async def test_native_stop_keeps_dispatcher_policy_refusals_receipted(control, changes):
    result = await receipt(control, request(**changes))
    assert result["recorded"] is True
    assert result["stop_outcomes"][0]["disposition"] == "refused"
    assert control.rail.agent.cancelled == []


async def test_native_stop_requires_current_recipient_authorization(control):
    await control.wire.admission.backend.execute("DELETE FROM native_test_principals")
    with pytest.raises(A2AClientError):
        await receipt(control)
    assert control.rail.agent.cancelled == []
    assert not (await control.rail.receipts.list_receipts(limit=10)).receipts


async def test_native_stop_preserves_signed_causation_cycle_policy(control):
    from datetime import UTC, datetime

    from kestrel_sdk.signals import CausationFrame

    from kestrel_sovereign.a2a.extensions import (
        causation_metadata,
        serialize_chain_for_metadata,
    )

    frame = CausationFrame(
        agent_id=control.rail.agent.did, source="a2a.peer_stop", signal_id="loop",
        turn_id=None, depth=1, emitted_at=datetime.now(UTC),
    )
    params = request()
    params.metadata.update(causation_metadata(serialize_chain_for_metadata([frame])))
    result = await receipt(control, params)
    assert result["signal_receipt"]["status"] == "dropped_cycle"
    assert result["stop_outcomes"][0]["disposition"] == "refused"
    assert result["recorded"] is True and control.rail.agent.cancelled == []


async def test_native_stop_cannot_activate_its_own_extension(control):
    params = request()
    params.tenant = control.wire.admission.scope.tenant
    with pytest.raises(InvalidParamsError):
        _ = [event async for event in control.client(activated=False).send_message(params)]
    assert control.rail.agent.cancelled == []
    assert not (await control.rail.receipts.list_receipts(limit=10)).receipts


@pytest.mark.parametrize("fault", ["version", "actor", "skill", "task", "reason", "role"])
async def test_native_stop_rejects_malformed_or_ambiguous_control(control, fault):
    params = request()
    if fault == "version":
        params.metadata[PEER_STOP_URI]["version"] = True
    elif fault == "actor":
        params.metadata[PEER_STOP_URI]["intent"]["actor_id"] = "did:forged:sender"
    elif fault == "skill":
        params.metadata[SKILL_HINT_URI] = {"version": 1, "skill_id": "message"}
    elif fault == "task":
        params.message.task_id = "another-task"
    elif fault == "reason":
        params.message.parts[0].text = "Different reason"
    else:
        params.message.role = Role.ROLE_AGENT
    with pytest.raises(InvalidParamsError):
        await receipt(control, params)
    assert control.rail.agent.cancelled == []
    assert control.wire.submissions == []
    assert not (await control.rail.receipts.list_receipts(limit=10)).receipts


async def test_native_stop_card_and_handler_bind_the_same_recipient(control):
    with pytest.raises(ValueError, match="task manager"):
        NativeA2ARequestHandler(
            control.wire.manager, control.wire.admission.guard, control.card
        )
    control.rail.agent.did = "did:wrong:recipient"
    with pytest.raises(ValueError, match="differs"):
        NativeA2ARequestHandler(
            control.wire.manager, control.wire.admission.guard, control.card,
            peer_stop_target=control.rail.agent,
        )


@pytest.mark.parametrize("fault", ["input", "output"])
async def test_native_stop_checks_content_modes_before_control_effects(control, fault):
    params = request()
    if fault == "input":
        params.message.parts[0].media_type = "image/png"
    else:
        params.configuration.accepted_output_modes.append("text/plain")
    with pytest.raises(ContentTypeNotSupportedError):
        await receipt(control, params)
    assert control.rail.agent.cancelled == []
    assert not (await control.rail.receipts.list_receipts(limit=10)).receipts


def test_native_stop_metadata_uses_the_canonical_intent_schema():
    from google.protobuf.json_format import MessageToDict

    claims = parse_message_metadata(MessageToDict(request().metadata))
    assert claims.peer_stop["correlation_id"] == "native-stop-1"


async def test_public_stop_tool_uses_signed_native_session_and_dispatch_ledger(sender):
    sender.control.rail.agent._active_request_ids.add("turn-1")
    result = await sender.feature.stop_peer("Peer", reason="Please stop")
    assert result.status == "ok" and result.data["stopped"] is True
    assert sender.control.rail.agent.cancelled == ["turn-1"]
    dispatch = await sender.manager.outbound_store.get(result.data["dispatch_id"])
    assert dispatch.state.value == "complete" and dispatch.response_kind == "message"
    assert dispatch.dispatch_tool == "stop_peer" and dispatch.remote_task_id is None
    assert sender.state.admissions >= 2
    assert sender.control.wire.submissions == []


async def test_public_stop_tool_cannot_reach_a_denied_peer(sender):
    sender.state.allowed = False
    result = await sender.feature.stop_peer("Peer")
    assert result.status == "error"
    assert sender.control.rail.agent.cancelled == []
    assert not (await sender.control.rail.receipts.list_receipts(limit=10)).receipts


@pytest.mark.parametrize("lost_responses", [1, 3])
async def test_native_stop_retries_only_the_same_receipted_intent(sender, lost_responses):
    import json

    calls = []

    async def lose_receipt(response):
        calls.append(response.request)
        if len(calls) <= lost_responses:
            await response.aclose()
            raise httpx.ReadError("Stop receipt lost", request=response.request)

    sender.control.http.event_hooks["response"].append(lose_receipt)
    sender.control.rail.agent._active_request_ids.add("turn-1")
    result = await sender.feature.stop_peer("Peer")
    assert len(calls) == (2 if lost_responses == 1 else 3), result.to_dict()
    assert sender.control.rail.agent.cancelled == ["turn-1"]
    page = await sender.control.rail.receipts.list_receipts(limit=10)
    assert len(page.receipts) == 1
    if lost_responses == 1:
        assert result.status == "ok" and result.data["stopped"] is True
    else:
        assert result.status == "error"
        assert result.data["delivery_state"] == "unknown"
        dispatch = await sender.manager.outbound_store.get(result.data["dispatch_id"])
        assert dispatch.state.value == "unknown"
    payloads = [json.loads(call.content) for call in calls]
    intents = [body.get("params", body)["metadata"][PEER_STOP_URI]["intent"]
               for body in payloads]
    assert all(intent == intents[0] for intent in intents)
    proofs = [call.headers["Kestrel-A2A-Sender-Proof"] for call in calls]
    assert len(set(proofs)) == len(proofs)


async def test_native_stop_preserves_local_response_error_without_retry(sender):
    original = TypeError("Local response hook failed")
    calls = []

    async def fail_locally(response):
        calls.append(response.request)
        raise original

    sender.control.http.event_hooks["response"].append(fail_locally)
    with pytest.raises(TypeError) as caught:
        await sender.feature.stop_peer("Peer")
    assert caught.value is original and len(calls) == 1


async def test_stop_retry_policy_never_replays_ordinary_peer_work(sender):
    calls = []

    async def lose_receipt(response):
        calls.append(response.request)
        await response.aclose()
        raise httpx.ReadError("Task receipt lost", request=response.request)

    sender.control.http.event_hooks["response"].append(lose_receipt)
    result = await sender.feature.send_a2a_message("Peer", "Do this once")
    assert result.status == "error" and result.data["delivery_state"] == "unknown"
    assert len(calls) == 1
    assert len(sender.control.wire.submissions) == 1
